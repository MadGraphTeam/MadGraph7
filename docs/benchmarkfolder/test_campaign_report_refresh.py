"""Closed-evidence selection and timing deferral checks; no native execution."""
import importlib.util
import json
from pathlib import Path
from copy import deepcopy
import hashlib
import pytest

HERE = Path(__file__).resolve().parent

def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

REFRESH = module('campaign_report_refresh')
REPORT = module('campaign_report')


def write(root, name, value):
    raw = (json.dumps(value, sort_keys=True) + '\n').encode()
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(raw)
    return {'bytes':len(raw), 'sha256':hashlib.sha256(raw).hexdigest()}


def fixture(root):
    summary = 'campaigns/test/numerical-checkpoints/one/summary.json'
    identity = write(root, summary, {'case_ids':['one'], 'case_count':1,
        'saved_points_per_lane':100, 'lane_count':6, 'performance_acceptance_eligible':False})
    index = 'campaigns/test/numerical-checkpoints/one/index.json'
    checksum = write(root,index, {'schema':'madloop-tirust-closed-numerical-checkpoint-v1',
        'performance_acceptance_eligible':False, 'files':{
        'numerical-checkpoints/one/summary.json':identity}})['sha256']
    selection = {'schema':'madloop-tirust-closed-report-selection-v1','campaign':'test',
        'checkpoints':{index:checksum},'closed_cases':['one'],'packages':{},'documents':{},
        'performance_status':'deferred to Hua-Sheng','final_performance_accepted':False}
    return selection, {'id':'test'}, {'cases':[{'id':'one'},{'id':'active'}]}


def test_explicit_checkpoint_closure_passes_without_active_data(tmp_path):
    selected,manifest,config=fixture(tmp_path)
    write(tmp_path,'selection.json',selected)
    evidence=REPORT.Evidence(tmp_path)
    assert REFRESH.selection(evidence,'selection.json',manifest,config)==selected
    assert not any('active' in name for name in evidence.files)


@pytest.mark.parametrize('mutation',[
    lambda d: d.update(closed_cases=['one','active']),
    lambda d: d.update(closed_cases=['one','one']),
    lambda d: d.update(final_performance_accepted=True),
    lambda d: d.update(performance_status='passed'),
    lambda d: d.update(campaign='different'),
    lambda d: d['checkpoints'].update({next(iter(d['checkpoints'])):'0'*64}),
])
def test_selection_rejects_promotion_and_substitution(tmp_path,mutation):
    selected,manifest,config=fixture(tmp_path)
    mutation(selected)
    write(tmp_path,'selection.json',selected)
    with pytest.raises(ValueError):
        REFRESH.selection(REPORT.Evidence(tmp_path),'selection.json',manifest,config)


def test_bound_payload_change_is_not_accepted(tmp_path):
    selected,manifest,config=fixture(tmp_path)
    write(tmp_path,'selection.json',selected)
    (tmp_path/'campaigns/test/numerical-checkpoints/one/summary.json').write_text('{}')
    with pytest.raises(ValueError):
        REFRESH.selection(REPORT.Evidence(tmp_path),'selection.json',manifest,config)
