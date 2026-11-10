#!/usr/bin/env python3
"""Pure integrity and consistency check of this independent audit packet."""
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
EXPECTED='8f5f7110cc101dc79e8fe75d5dd1ef342e1e72797986a5a5026680e28d4496be'
def sha(raw): return hashlib.sha256(raw).hexdigest()
def encoded(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def main():
    index=json.loads((HERE/'index.json').read_bytes());assert index['selected_package_index_sha256']==EXPECTED
    assert {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}==set(index['files'])|{'index.json'}
    payload={}
    for name,entry in index['files'].items():
        path=Path(name);assert not path.is_absolute() and '..' not in path.parts and str(path)==name
        p=HERE
        for part in path.parts: p=p/part;assert not p.is_symlink()
        raw=p.read_bytes();assert sha(raw)==entry['sha256'] and len(raw)==entry['bytes'];payload[name]=raw
    proof=json.loads(payload['proof.json']);cases=json.loads(payload['cases.json']);scan=json.loads(payload['runtime-console-scan.json'])
    assert proof['passed'] is True and proof['expected_package_index_sha256']==EXPECTED
    assert proof['relocated_replay_exactly_equal'] is True and proof['sealed_package_unchanged'] is True
    assert proof['native_or_process_execution'] is False and proof['original_generated_work_reads'] is False
    assert proof['performance_acceptance_eligible'] is False and proof['overall_goal_qualified'] is False
    assert [x['phase'] for x in proof['passes']]==['selected_package_only','relocated_no_workspace']
    assert len({x['replay_sha256'] for x in proof['passes']})==1
    for p in proof['passes']:
        assert p['passed'] is True and p['index_sha256']==EXPECTED
        assert (p['case_count'],p['lane_count'],p['saved_point_lane_observations'])==(11,66,6600)
        assert (p['retained_members'],p['retained_bytes'])==(1221,157210978)
        assert p['unexpected_denied_operations']==0 and p['blocked_virtual_sentinel_metadata_probes']==3
        assert len(p['negative_controls'])==5 and len(p['blocked_operations'])==8
        assert sum(x['intentional_negative'] for x in p['blocked_operations'])==5
        assert p['runtime_console_scan_sha256']==sha(encoded(scan))
    assert len(cases['rows'])==11 and cases['registered_regression_tolerance']==3e-6 and cases['ordinary_target']==1e-8
    assert len({x['case'] for x in cases['rows']})==11
    expected={('baseline',1),*(('candidate',n) for n in (1,6,7,8,9))}
    for row in cases['rows']:
        assert row['regression_pass'] is True and len(row['lanes'])==6
        assert {(x['source'],x['backend']) for x in row['lanes']}==expected
        for lane in row['lanes']:
            assert lane['regression_pass'] is True and lane['points']==100
            assert lane['accepted_dp_count']+lane['accepted_qp_count']==100
            assert sum(lane['status_counts'].values())==100
    assert cases['tighter_target_miss_cases']==[r['case'] for r in cases['rows'] if not r['ordinary_target_pass']]
    assert set(cases['tighter_target_miss_cases'])=={'qcd_n3_uux_zgg','qcd_n4_uux_zzgg','ew_n4_epem_mupmumaa'}
    assert scan['file_count']==len(scan['files'])==66 and scan['total_warning_lines']==0
    assert all('/experiments/numerical/' in name and name.endswith('-output.txt') and item['matches']==[] for name,item in scan['files'].items())
    print(json.dumps({'passed':True,'package_index':EXPECTED,'case_count':11,'lane_count':66,'runtime_console_W03_lines':0,'native_executed':False},sort_keys=True))
if __name__=='__main__': main()
