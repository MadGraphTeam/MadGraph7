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


def next_fixture(root, change=lambda summary: None):
    cases = ["case" + str(i) for i in range(11)]
    manifest = {"schema": "madloop-tirust-campaign-v1", "id": "fresh",
                "baseline": {"revision": "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f"},
                "candidate": {"exporter_identity": {"dirty": False, "files_sha256": "new-source", "revision": "new-revision"},
                              "numerical_identity_sha256": "new-numerical"}}
    manifest_pin = {"path": "fresh/manifest.json", "sha256": write(root, "fresh/manifest.json", manifest)["sha256"]}
    summary = {"campaign_binding": {"id": "fresh", "manifest_sha256": manifest_pin["sha256"]},
               "gate": {"passed": False, "complete": False, "failed_cases": [], "validation_errors": []},
               "cases": [{"id": case, "status": "pending", "complete": False, "retained_attempts": [],
                          "performance": {"pairs": 0, "geometric_mean_ratio": None},
                          "correctness": [{"backend": b, "status": "pending", "samples": 0, "passed": None} for b in (1, 6, 7, 8, 9)],
                          "timings": [{"backend": b, "status": "pending", "samples": 0, "median_seconds_per_point": None} for b in (1, 6, 7, 8, 9)]}
                         for case in cases]}
    change(summary)
    selection = {"manifest": manifest_pin, "summary": {"path": "fresh/summary.json",
                 "sha256": write(root, "fresh/summary.json", summary)["sha256"]}}
    report = {"campaign_binding": {"id": "old"}, "registered_source_sha256": "old-source",
              "numerical": {"cases": [{"id": case} for case in cases]}}
    return selection, report


def test_new_registration_is_bound_without_overlaying_old_measurements(tmp_path):
    selected, report = next_fixture(tmp_path)
    frozen = deepcopy(report)
    result = REFRESH.next_campaign(REPORT.Evidence(tmp_path), selected, report)
    assert result["id"] == "fresh" and result["case_count"] == 11
    assert result["displayed_measurements_from_this_campaign"] is False
    assert result["fresh_numerical_qualification"] == "pending"
    assert report == frozen


@pytest.mark.parametrize("change", [
    lambda d: d["campaign_binding"].update(id="other"),
    lambda d: d["cases"][0]["timings"][0].update(samples=1),
    lambda d: d["cases"][0]["correctness"][0].update(passed=True),
    lambda d: d["cases"][0]["performance"].update(geometric_mean_ratio=1.),
    lambda d: d["cases"][0].update(retained_attempts=["observed"]),
    lambda d: d["gate"].update(passed=True),
])
def test_new_registration_cannot_smuggle_unbound_measurements(tmp_path, change):
    selected, report = next_fixture(tmp_path, change)
    with pytest.raises(ValueError):
        REFRESH.next_campaign(REPORT.Evidence(tmp_path), selected, report)
