"""Read retained focused rows with the existing common-point comparison."""
from pathlib import Path
import json,sys,hashlib
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
sys.path[:0]=[str(ROOT/'tools'),str(ROOT)]
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers
FINAL=ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1'
HERE=Path(__file__).resolve().parent
RUNTIME='a7577a9a33292ffb5d675ccf1b9c3667db3ab3613d77ccd270f1efc9e9802008'
REVISION='856e76e669618ec8d0f799b9c8e6f86819dcd986'
correction=policy.validate_correction(ROOT/'docs/benchmarkfolder/policy-v2/model-correction.json')
inputs={};results={}
def bound(p):
    return {'path':policy.relative(p),'sha256':workers.digest(p),'bytes':p.stat().st_size}
def load(binding_name,case,backends):
    path=FINAL/binding_name;binding=policy.checked_bindings(path,correction)
    inputs[policy.relative(path)]=workers.digest(path)
    protocol,records=workers.checked_series(policy.path(binding['report']),minimum=2)
    found={}
    for backend in backends:
        key=f'{case}-candidate-{backend}';target=binding['targets'][key]
        assert target['backend']==backend and target['role']=='candidate_corrected'
        assert target['generation_source_identity']['revision']==REVISION
        assert target['reducer_runtime']['identity_sha256']==RUNTIME
        for repetition in (0,1,'fresh'):
            source=policy.path(binding['report'])/'data'/f'{key}-{repetition}.json'
            assert binding['run_files_sha256'][policy.relative(source)]==workers.digest(source)
            data=policy.read(source);inputs[policy.relative(source)]=workers.digest(source)
            expected_exe=target['fresh_executable_sha256'] if repetition=='fresh' else target['executable_sha256']
            assert data['executable_sha256']==expected_exe
            assert data['backend']==backend and data['case']==case
            assert data['shared_numerical_settings']['selected_backend_qp_integrand'] is True
            found[f'{backend}-{repetition}']=data
    return found
families={
'complex':load('focused-ninja-nrs-v1/runtime-bindings/complex.json','control-complex',[6,8]),
'mixed':{**load('focused-ninja-nrs-v1/runtime-bindings/mixed-ninja.json','control-mixed',[6]),**load('representative/runtime-bindings/mixed.json','control-mixed',[8])},
'near-singular':{**load('focused-ninja-nrs-v1/runtime-bindings/near-singular-nrs.json','control-forced-qp',[8]),**load('representative/runtime-bindings/ninja-near-singular.json','control-forced-qp',[6])}}
for name,records in families.items():
    identities={(r['points_sha256'],r['parameter_card_sha256'],json.dumps(r['answer_dimensions'],sort_keys=True),json.dumps(r['shared_numerical_settings'],sort_keys=True)) for r in records.values()}
    assert len(identities)==1,(name,'Different inputs/settings/dimensions')
    keys=[[(x['point'],x['order']) for x in r['rows']] for r in records.values()]
    assert all(key==keys[0] for key in keys)
    rows={key:record['rows'] for key,record in records.items()}
    comparison=policy.compare_rows(rows)
    histograms={key:{'saved':r['status_counts'],'timed':r['timed_status_counts']} for key,r in records.items()}
    if name=='near-singular':
        assert all(r['status_counts']=={'322':10} and r['timed_status_counts']=={'322':10} for key,r in records.items() if key.startswith('8-'))
        assert all(r['status_counts']=={'328':10} and r['timed_status_counts']=={'328':10} for key,r in records.items() if key.startswith('6-'))
    count=comparison['pairs']*len(keys[0])*4
    results[name]={'comparison':comparison,'physical_components':count,'records':len(records),'rows_per_record':len(keys[0]),'status_histograms':histograms,'dimensions':next(iter(records.values()))['answer_dimensions'],'points_sha256':next(iter(records.values()))['points_sha256'],'physical_card_sha256':next(iter(records.values()))['parameter_card_sha256']}
receipt={'scope':'Only current-cohort Ninja6/NinjaRS8 fixed10 focused correctness; all total/split Born/finite/pole components, two persistent commands and fresh-process data','passed':all(x['comparison']['passed'] for x in results.values()),'results':results,'physical_components':sum(x['physical_components'] for x in results.values()),'comparison_pairs':sum(x['comparison']['pairs'] for x in results.values()),'runtime_identity_sha256':RUNTIME,'numerical_revision':REVISION,'sources':{'script':bound(Path(__file__)),'existing_comparison':bound(ROOT/'tools/ninjars_benchmark_policy_v2.py'),'existing_series_reader':bound(ROOT/'tools/ninjars_benchmark_workers.py')},'inputs_sha256':inputs,'amplitude_calls':0,'performance_claim':False,'remaining':['HHrotation/state/direction/directQP and complexdirectQP customcontrols remainheld; HH100 first10 anchor notyetavailable.']}
out=HERE/'physical-comparison.json'
if out.exists():raise RuntimeError('Preserve prior comparison; nooverwrite')
out.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'passed':receipt['passed'],'physical_components':receipt['physical_components'],'comparison_pairs':receipt['comparison_pairs'],'result':bound(out)}))
if not receipt['passed']:raise SystemExit(1)
