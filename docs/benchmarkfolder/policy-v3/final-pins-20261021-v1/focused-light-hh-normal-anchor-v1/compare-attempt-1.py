#!/usr/bin/env python3
"""Read-only HH ordinary calibration versus already completed explicit-QP rows."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'tools'))
import ninjars_benchmark_policy_v2 as policy

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def bound(path):return {"path":str(path.relative_to(ROOT)),"sha256":sha(path),"bytes":path.stat().st_size}
def main():
    base=ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1'
    light=base/'light-timing/runs/run-v1';focused=base/'focused-light-controls-v2'
    records={
        'normal-original-Ninja6':(light/'data/supplement-gg-hh-baseline-6-calibration.json','f4d465550263c9782c63329700b02b7a62873d145dca6876775a9f7ca7791cbc'),
        'normal-final-NinjaRS8':(light/'data/supplement-gg-hh-candidate-8-calibration.json','1e4ccb8c27ffd366c130b1cf68a777affc2ea78dc06008b45e39ecfef22bf24b'),
        'explicit-QP-forward-NinjaRS8':(focused/'run-v1/jobs/hh-qp-forward-8/result.json',''),
        'explicit-QP-reverse-NinjaRS8':(focused/'run-v1/jobs/hh-qp-reverse-8/result.json','')}
    light_result=policy.verified(read(light/'result.json'),schema=read(light/'result.json')['schema'])
    manifest=read(focused/'manifest.json')['files'];spec=read(focused/'specification.json')
    source=policy.check_file(spec['points']['control-gg-hh']['source']);selected=policy.check_file(spec['points']['control-gg-hh']['selected'])
    lines=source.read_bytes().splitlines(keepends=True)
    assert len(lines)==41 and lines[0].strip()==b'10' and b'2\n'+b''.join(lines[1:9])==selected.read_bytes()
    groups={};bindings={};raw_bindings={};histograms={};card=None;dimensions={'allocated':1,'loop_orders':1,'born_orders':0}
    for label,(path,expected) in records.items():
        actual=sha(path)
        if expected:assert actual==expected
        else:assert manifest[str(path.relative_to(focused))]['sha256']==actual
        data=read(path);assert data['parameter_card_sha256']==(card or data['parameter_card_sha256']);card=data['parameter_card_sha256'];assert data['answer_dimensions']==dimensions
        if label.startswith('normal'):
            assert data['repetition']==-1 and data['points_sha256']==sha(source) and len(data['rows'])==20
            assert light_result['inputs_sha256'][str(path.relative_to(ROOT))]==actual
            assert data['case']=='supplement-gg-hh' and data['backend']==(6 if 'Ninja6' in label else 8)
            assert data['status_counts']=={str(210+data['backend']):10}
            for name,item in data['raw_capture'].items():policy.check_file(item);raw_bindings[label+'/'+name]=item
        else:
            assert data['backend']==8 and data['mode']==(4 if 'forward' in label else 5) and data['points_sha256']==sha(selected)
            assert data['runtime_identity_sha256']==spec['cohort']['runtime_identity_sha256'] and data['status_counts']=={'120':2}
            assert data['shared_numerical_settings']['selected_backend_qp_integrand'] is True
            assert all(row['status']==120 and row['accuracy']==-1. for row in data['rows'])
        rows=[row for row in data['rows'] if row['point']<=2]
        assert [(row['point'],row['order']) for row in rows]==[(1,0),(1,1),(2,0),(2,1)]
        groups[label]=rows;bindings[label]=bound(path);histograms[label]=data['status_counts']
    target_path=light/'targets/supplement-gg-hh-candidate-8.json';target=policy.verified(read(target_path),schema=read(target_path)['schema'])
    assert target['reducer_runtime']['identity_sha256']==spec['cohort']['runtime_identity_sha256']
    comparison=policy.compare_rows(groups);assert comparison['passed']
    receipt={'status':'passed','scope':'Read-only normal full-sum HH calibration first2 versus unchanged-state HH explicit-QP forward/reverse; no new calls or benchmark inference',
        'comparison':comparison,'physical_components':comparison['pairs']*4*4,'points':2,'orders':[0,1],'source_stream':bound(source),'selected_stream':bound(selected),
        'physical_card_sha256':card,'answer_dimensions':dimensions,'histograms':histograms,'records':bindings,'normal_raw_capture':raw_bindings,
        'light_dataset':bound(light/'result.json'),'actual_candidate_target':bound(target_path),'focused_manifest':bound(focused/'manifest.json'),
        'runtime_identity_sha256':target['reducer_runtime']['identity_sha256'],'comparator':bound(ROOT/'tools/ninjars_benchmark_policy_v2.py'),
        'limits':['Modified-mass/scale state targets excluded','120 accuracy -1 establishes explicitQP scope, not stability or automatic rescue','Two points only; no HH100 claim','Timing acceptance belongs exclusively to the independent LIGHT6 authority']}
    destination=Path(__file__).parent/'comparison.json';assert not destination.exists();destination.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'passed':True,'pairs':comparison['pairs'],'physical_components':receipt['physical_components'],'receipt_sha256':sha(destination)}))
if __name__=='__main__':main()
