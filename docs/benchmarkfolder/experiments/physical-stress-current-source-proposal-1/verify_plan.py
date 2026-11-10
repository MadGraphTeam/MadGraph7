#!/usr/bin/env python3
"""Validate a source-only proposal. Never launch native/build/CLI commands."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent

def sha(raw): return hashlib.sha256(raw).hexdigest()

def verify():
    assert Path.cwd() == ROOT
    plan = json.loads((HERE / 'proposal.json').read_bytes())
    assert plan['native_executed'] is plan['preparation_executed'] is False
    assert plan['manifest_sha256'] == 'ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0'
    bindings = json.loads((HERE / 'input-bindings.json').read_bytes())
    for name, checksum in bindings.items(): assert sha((ROOT / name).read_bytes()) == checksum, name
    index = HERE / 'source-index.json'
    if index.exists():
        for name, record in json.loads(index.read_bytes())['files'].items():
            raw = (HERE / name).read_bytes()
            assert sha(raw) == record['sha256'] and len(raw) == record['size'], name
    old = (HERE / 'protocol/tirust_benchmark_worker.f').read_text()
    new = (HERE / 'protocol/stress-worker.f').read_text()
    needle = "      OPEN(6,FILE='/dev/null',STATUS='OLD')\n"
    assert old.count(needle) == 1
    assert new == old.replace(needle, 'C     Untimed stress diagnostics retain all native/host stdout.\n')
    for case in plan['cases']:
        cid = case['case']['id']; n = case['case']['nfinal'] + 2
        point = HERE / 'points' / cid
        data = (point / 'selected.dat').read_bytes(); metadata = json.loads((point / 'selected.json').read_bytes())
        assert sha(data) == metadata['sha256'] == case['point_sha256']
        rows = ['2']
        for parent in metadata['parents']:
            suffix = '' if parent['role'] == 'ordinary-control' else '-stress'
            raw = (point / ('parent' + suffix + '.dat')).read_bytes()
            assert sha(raw) == parent['sha256']
            lines = raw.decode().splitlines(); start = 1 + (parent['source_point_index'] - 1) * n
            rows.extend(lines[start:start+n])
        assert ('\n'.join(rows)+'\n').encode() == data
        assert sha((HERE / 'cards' / cid / 'param_card.dat').read_bytes()) == case['physical_card_sha256']
        for source in ('baseline', 'candidate'):
            command = (HERE / 'commands' / (source+'-'+cid+'.mg5')).read_text().splitlines()
            assert sum(line.startswith('output standalone ') for line in command) == 1
            assert 'output standalone ' + str(Path(plan['work']) / 'exports' / source / cid) in command
            assert sum(line.startswith('set tirust ') for line in command) == (source == 'candidate')
            assert ('generate ' + case['case']['process']) in command
    assert len(plan['requests']) == 24
    assert len({(r['case'],r['source'],r['backend']) for r in plan['requests']}) == 24
    for request in plan['requests']:
        assert request['backend'] in ([1] if request['source']=='baseline' else [1,6,7,8,9])
        assert request['MLReductionLib'] == str(request['backend'])
        assert request['argv'][:4] == ['taskset','-c','12','./check']
        assert request['startup_seconds'] == request['traversal_seconds'] == 180
        assert request['retry'] is False
        assert request['requires_common_numerical_lock'] == (request['backend'] in (8,9))
    assert plan['stability']['MLStabThres']==1e-6 and plan['stability']['CTModeRun']==-1
    assert not any(plan[k] for k in ('performance_eligible','fixed_QP_admitted','adaptive_bridge_admitted','production_changes_admitted'))
    return {'passed':True,'bound_read_only_inputs':len(bindings),'cases':4,'source_control_requests':4,'candidate_requests':20,'source_only':True,'commands_executed':0,'driver_delta':'stdout retention only'}

if __name__ == '__main__': print(json.dumps(verify(),indent=2,sort_keys=True))
