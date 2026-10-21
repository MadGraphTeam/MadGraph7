#!/usr/bin/env python3
"""Retire exact stopped jobs; never resume an interrupted numerical interval."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())
OLD = HERE.parent / 'ninjars-mp-tensor-allocation-process-v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def binding(path):
    return {'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path)}

def write(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,sort_keys=True)
        stream.write('\n')

def snapshot(expected):
    p = Path('/proc') / str(expected['pid'])
    raw = (p / 'stat').read_text()
    fields = raw[raw.rfind(')')+2:].split()
    argv = (p / 'cmdline').read_bytes().decode().split('\0')[:-1]
    result = {'pid':expected['pid'],'ppid':int(fields[1]),'state':fields[0],
        'pgid':int(fields[2]),'sid':int(fields[3]),'start_ticks':int(fields[19]),
        'argv':argv,'cwd':os.readlink(p / 'cwd'),'stat':raw,
        'fds':{fd.name:os.readlink(fd) for fd in (p / 'fd').iterdir()}}
    for name in ['pid','ppid','pgid','sid','start_ticks','argv','cwd']:
        assert result[name] == expected[name],(name,result[name],expected[name])
    assert result['state'] == 'T'
    return result

def main():
    assert Path.cwd() == ROOT
    pause_path = OLD / 'pause-20261021.json'
    pause = json.loads(pause_path.read_text())
    assert pause['completed_qcd3_pairs_zero_based'] == list(range(12))
    assert pause['completed_sample_count'] == 25 and not pause['gate_passed']
    retention_path = OLD / 'pause-retention.json'
    retention = json.loads(retention_path.read_text())
    for name,record in retention['files'].items():
        path = OLD / name
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes'],name
    expected = pause['process_snapshot']['processes']
    assert [(p['pid'],p['start_ticks']) for p in expected] == [(2896495,57322835),(3369596,57422045)]
    before = [snapshot(p) for p in expected]
    # pidfds prevent signaling a reused PID between validation and termination.
    fds = {p['pid']:os.pidfd_open(p['pid']) for p in before}
    record = {'schema':'allocation-pause-retirement-v1','started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'authorization':'Root explicitly authorizes termination of the exact verified stopped controller and worker; user resumed; no SIGCONT or accepted pause-spanning interval',
        'controller':binding(Path(__file__)),'pause':binding(pause_path),
        'old_retention':binding(retention_path),'old_retention_verified_files':len(retention['files']),
        'before':before,'signals':[],'numerical_execution':False,'old_measurements_unchanged':True}
    write(HERE / 'recovery-started.json',record)
    try:
        for p in before:
            snapshot(p)
            signal.pidfd_send_signal(fds[p['pid']],signal.SIGKILL)
            record['signals'].append({'pid':p['pid'],'signal':'SIGKILL','via':'pidfd','never_SIGCONT':True})
        deadline = time.monotonic()+5
        states = {}
        while time.monotonic() < deadline:
            states = {}
            for p in before:
                path = Path('/proc') / str(p['pid']) / 'stat'
                if not path.exists():
                    states[str(p['pid'])] = 'absent'
                    continue
                raw = path.read_text()
                fields = raw[raw.rfind(')')+2:].split()
                assert int(fields[19]) == p['start_ticks']
                states[str(p['pid'])] = fields[0]
            if all(s in ['absent','Z'] for s in states.values()):
                break
            time.sleep(.05)
        assert all(s in ['absent','Z'] for s in states.values()),states
        record['after_states'] = states
        record['status'] = 'retired-without-resumption'
    finally:
        for fd in fds.values():
            os.close(fd)
        record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(HERE / 'recovery-result.json',record)
    print(json.dumps({'status':record['status'],'states':record['after_states'],
        'result':binding(HERE / 'recovery-result.json')}),flush=True)

if __name__ == '__main__':
    main()
