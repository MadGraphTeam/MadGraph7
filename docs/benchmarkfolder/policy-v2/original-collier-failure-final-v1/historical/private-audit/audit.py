#!/usr/bin/env python3
"""Preserve/analyze the failed actual Collier calibration; no physics imports."""
from pathlib import Path
import collections,datetime,hashlib,itertools,json,math,os,shutil
ROOT=Path('/shared/external/madloop_ninjars/madgraph7_madloop_ninjars')
HERE=Path(__file__).resolve().parent
REPORT=ROOT/'docs/benchmarkfolder/policy-v2/reference-runs/corrected-original-core-v2-fresh-20261021'
OUTPUT=ROOT/'.benchmark-work/policy-v2/reference-runs/corrected-original-core-v2-fresh-20261021/workers/qcd-4-baseline-7/outputs/baseline/qcd-4'
DIRECTORY=OUTPUT/'SubProcesses/P0_uux_zzgg'
COPIES={}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def preserve(source,destination):
 source=Path(source);destination=HERE/destination;destination.parent.mkdir(parents=True,exist_ok=True)
 if destination.exists():raise RuntimeError('Refuse to overwrite preserved failure evidence: '+str(destination))
 destination.write_bytes(source.read_bytes());COPIES[str(source.relative_to(ROOT))]={'retained':str(destination.relative_to(HERE)),'sha256':digest(destination),'bytes':destination.stat().st_size}
 return destination
def load(p):return json.loads(Path(p).read_text())
def cdigest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
for p in sorted(REPORT.rglob('*')):
 if p.is_file():preserve(p,Path('raw/report')/p.relative_to(REPORT))
preserve(ROOT/'.benchmark-work/logs/corrected-original-core-v2-fresh-20261021.log','raw/collector-combined-stdout-stderr.log')
for name in ['values.dat','timing.dat','timed_status.dat','dimensions.dat','check_sa.f']:
 preserve(DIRECTORY/name,Path('raw/worker')/name)
for name in ['Cards/param_card.dat','Cards/MadLoopParams.dat','generation.json','compilation.json','correction-build.json','correction-source.json']:
 preserve(OUTPUT/name,Path('raw/actual-output')/name)
preserve(DIRECTORY/'MadLoop5_resources/MadLoopParams.dat','raw/actual-resource-MadLoopParams.dat')
for name in ['ninjars_benchmark.py','ninjars_benchmark_worker.f','ninjars_benchmark_workers.py','ninjars_benchmark_references_v2.py','ninjars_benchmark_policy_v2.py']:
 preserve(ROOT/'tools'/name,Path('source/frozen-tools')/name)
correction=load(ROOT/'docs/benchmarkfolder/policy-v2/model-correction.json')
preserve(ROOT/'docs/benchmarkfolder/policy-v2/model-correction.json','source/model-correction.json')
build=load(REPORT/'builds/qcd-4-baseline-7/build.json');compiled=load(REPORT/'builds/qcd-4-baseline-7/compiled-sources.json')
assert compiled['files_sha256']==cdigest(compiled['files'])==build['generated_sources_sha256']
for name,entry in compiled['files'].items():
 original=OUTPUT/name;assert original.stat().st_size==entry['bytes'] and digest(original)==entry['sha256'],name
 preserve(original,Path('source/actual-compiled-tree')/name)
assert {k:v for k,v in compiled['files'].items() if k.startswith('Source/MODEL/')}=={k:v for k,v in correction['cases']['qcd-4']['fresh_after_manifest']['files'].items() if k.startswith('Source/MODEL/')}
assert digest(DIRECTORY/'check')==build['executable_sha256'];assert digest(DIRECTORY/'check_sa.f')==build['driver_sha256']
assert digest(OUTPUT/'Cards/param_card.dat')==build['parameter_card_sha256'];assert digest(OUTPUT/'Cards/MadLoopParams.dat')==build['madloop_card_sha256']
assert (DIRECTORY/'MadLoop5_resources/MadLoopParams.dat').read_bytes()==(OUTPUT/'Cards/MadLoopParams.dat').read_bytes()
candidates=[DIRECTORY/'param_card.dat']+[DIRECTORY.joinpath(*(['..']*level),'Cards/param_card.dat') for level in range(6)];effective=next(p for p in candidates if p.is_file());assert effective.read_bytes()==(OUTPUT/'Cards/param_card.dat').read_bytes()
libs={};libroot=OUTPUT/'lib';targets=list(libroot.iterdir())
for p in list(targets):
 if p.is_dir():targets.extend(p.rglob('*'))
for p in sorted(targets):
 if p.suffix in ['.a','.so','.dylib'] or '.so.' in p.name:
  assert p.is_file();libs[str(p.relative_to(libroot))]={'sha256':digest(p),'bytes':p.stat().st_size,'resolved':str(p.resolve().relative_to(ROOT))}
rebuild=load(OUTPUT/'correction-build.json');assert {k:v['sha256'] for k,v in libs.items()}==rebuild['libraries_after']
rows=[]
for line in (DIRECTORY/'values.dat').read_text().splitlines():
 f=line.split();assert len(f)==8
 rows.append({'point':int(f[0]),'order':int(f[1]),'status':int(f[2]),'coefficients':list(map(float,f[3:7])),'accuracy':float(f[7])})
assert [(v['point'],v['order']) for v in rows]==[(p,o) for p in range(1,101) for o in [0,1]]
assert all(all(math.isfinite(v) for v in r['coefficients']+[r['accuracy']]) for r in rows)
counts=dict(collections.Counter(str(r['status']) for r in rows if r['order']==0));timed={}
for line in (DIRECTORY/'timed_status.dat').read_text().splitlines():
 a,b=map(int,line.split());timed[str(a)]=b
assert timed==counts and sum(counts.values())==100
failure=load(REPORT/'rejected/qcd-4-failure.json')
for name in ['values.dat','timing.dat','timed_status.dat','dimensions.dat']:assert failure['raw_outputs'][name]==(DIRECTORY/name).read_text(),name
assert failure['stage']=='calibration' and failure['backend']==7 and failure['command_sequence']==1
settings=build['shared_numerical_settings'];assert settings['mlstabthres']==1e-9 and settings['collierrequiredaccuracy']==1e-12 and settings['colliercomputeuvpoles'] and settings['colliercomputeirpoles'] and settings['collieruseinternalstabilitytest']
mlcard=(OUTPUT/'Cards/MadLoopParams.dat').read_text();lines=mlcard.splitlines();assert lines[lines.index('#MLReductionLib')+1]=='7'
scalars={1:load(REPORT/'data/qcd-4-baseline-1-calibration.json')['rows'],6:load(REPORT/'data/qcd-4-baseline-6-calibration.json')['rows'],7:rows};scales={}
for backend,data in scalars.items():
 for row in data:scales[row['point']]=max(scales.get(row['point'],0.),*map(abs,row['coefficients']))
compares=[]
for left,right in itertools.combinations(scalars,2):
 for a,b in zip(scalars[left],scalars[right]):
  assert (a['point'],a['order'])==(b['point'],b['order'])
  for column,(x,y) in enumerate(zip(a['coefficients'],b['coefficients'])):
   error=abs(x-y);tol=1e-7*max(abs(x),abs(y))+1e-10*scales[a['point']]
   compares.append({'backends':[left,right],'point':a['point'],'order':a['order'],'column':column,'values':[x,y],'absolute_error':error,'tolerance':tol,'fraction_of_tolerance':error/tol if tol else 0.,'passed':error<=tol})
exceptional=[r for r in rows if r['status']==410]
point_context={str(point):{str(backend):next(v for v in data if v['point']==point and v['order']==0) for backend,data in scalars.items()} for point in sorted({v['point'] for v in exceptional})}
for p,data in point_context.items():
 data['point_scale_S']=scales[int(p)]
log=(REPORT/'logs/worker-qcd-4-baseline-7.log').read_text(errors='replace')
assert 'No quadruple precision will be used.' in log
references={
 'internal_error_shift':{'path':'SubProcesses/P0_uux_zzgg/TIR_interface.f','first_line':178,'last_line':190},
 'selected_qp_capability_table':{'path':'SubProcesses/P0_uux_zzgg/loop_matrix.f','first_line':261,'last_line':263},
 'selected_qp_discovery':{'path':'SubProcesses/P0_uux_zzgg/loop_matrix.f','first_line':514,'last_line':538},
 'dp_accuracy_failure_and_no_fallback':{'path':'SubProcesses/P0_uux_zzgg/loop_matrix.f','first_line':1508,'last_line':1569},
 'accuracy_aggregation_and_collier_first_value':{'path':'SubProcesses/P0_uux_zzgg/loop_matrix.f','first_line':1935,'last_line':1969},
 'return_code_zero_when_not_stable':{'path':'SubProcesses/MadLoopCommons.f','first_line':313,'last_line':327},
 'strict_worker_status_parser':{'file':'source/frozen-tools/ninjars_benchmark.py','first_line':418,'last_line':425}}
for entry in references.values():
 if 'path' in entry:entry['file']='source/actual-compiled-tree/'+entry['path']
 source=HERE/entry['file'];entry['sha256']=digest(source);entry['lines']=[{'line':i,'text':t} for i,t in enumerate(source.read_text().splitlines(),1) if entry['first_line']<=i<=entry['last_line']]
result={'schema':'original-collier-qcd4-strict-calibration-failure-audit-v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Pure-stdlib offline diagnosis of actual failed original-helper calibration; no new evaluation or policy change','terminal_status':load(REPORT/'run-status.json'),'controller_and_worker_pids_absent':{str(pid):not Path(f'/proc/{pid}').exists() for pid in [1973783,1995662,3208654,442282]},'all_200_rows_and_coefficients_finite':True,'physical_points':100,'row_count':200,'point_status_counts':counts,'exceptional_rows':exceptional,'point_context':point_context,'fixed_threshold':1e-9,'collier_required_accuracy':1e-12,'selected_libraries':'7 only; no selected QP-capable backend','source_bindings':{'live_generated_sources_checked':len(compiled['files']),'compiled_manifest_sha256':compiled['files_sha256'],'driver_sha256':build['driver_sha256'],'executable_sha256':build['executable_sha256'],'exact_corrected_model_postimages':True,'effective_parameter_card':str(effective.relative_to(ROOT)),'libraries_match_forced_original_rebuild':True,'libraries':libs},'source_path':references,'warmup_log_context':{'no_qp_warning_count':log.count('No quadruple precision will be used.'),'collier_cache_consistency_reinitialization_warnings':log.count('consistency check in MadLoop failed'),'scope':'Warnings were retained during warmup; unit6 is redirected during measured calibration. Cache reset warnings are preserved, not established as the failure cause.'},'cross_backend_value_diagnostic':{'scalar_comparisons':len(compares),'pairs':3,'maximum_fraction_of_tolerance':max(v['fraction_of_tolerance'] for v in compares),'max_case':max(compares,key=lambda v:v['fraction_of_tolerance']),'failures':[v for v in compares if not v['passed']],'passed':all(v['passed'] for v in compares),'scope':'Compares finite raw values including rejected Collier rows; does not override return410 or make failed stability gate pass.'},'interpretation':'The original Collier internal-error stability estimate exceeds the unchanged strict threshold at points17/52. Generic original MadLoop exhausts the sole selected backend7, has no selected QP fallback, and returns H4/T1/U0=410. The frozen parser correctly rejects it. Source/card/model/driver bindings are consistent; raw output is complete and finite. Cross-backend numerical agreement, if passing, is a separate weaker observable and does not certify the required stability gate.','actions':{'signals':[],'new_jobs':[],'model_imports':[],'source_edits':[],'automatic_retries':0}}
(HERE/'parsed-collier-rows.json').write_text(json.dumps(rows,indent=2)+'\n');(HERE/'comparison-checks.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in compares));(HERE/'copies.json').write_text(json.dumps(COPIES,indent=2)+'\n');(HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['point_status_counts','exceptional_rows','cross_backend_value_diagnostic','source_bindings','warmup_log_context']},indent=2))
