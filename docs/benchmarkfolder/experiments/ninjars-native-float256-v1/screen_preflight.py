#!/usr/bin/env python3
"""Retain three fixed host-only CPU intervals before any screening sample."""
import sys
sys.dont_write_bytecode = True
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def cpu_stat(cpu):
    for line in Path('/proc/stat').read_text().splitlines():
        fields=line.split()
        if fields[0]==f'cpu{cpu}':return [int(v) for v in fields[1:9]]
    raise AssertionError(cpu)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu',type=int,required=True)
    parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    assert Path.cwd()==ROOT and args.cpu in os.sched_getaffinity(0) and args.cpu not in (2,207)
    assert args.attempt and all(c.isalnum() or c in '-_' for c in args.attempt)
    destination=HERE/'supplement'/('preflight-'+args.attempt)
    destination.mkdir(exist_ok=False)
    topology=subprocess.check_output(['lscpu','-J'],text=True)
    (destination/'lscpu.json').write_text(topology)
    sibling_path=Path(f'/sys/devices/system/cpu/cpu{args.cpu}/topology/thread_siblings_list')
    siblings=sibling_path.read_text().strip()
    allowed=os.sched_getaffinity(0)
    expanded=set()
    for part in siblings.split(','):
        if '-' in part:
            start,end=map(int,part.split('-')); expanded.update(range(start,end+1))
        else:expanded.add(int(part))
    assert expanded & allowed=={args.cpu},'Require one exposed thread on selected core'
    intervals=[]
    for _ in range(3):
        before=cpu_stat(args.cpu); start=time.monotonic_ns()
        time.sleep(3)
        after=cpu_stat(args.cpu); elapsed=time.monotonic_ns()-start
        deltas=[b-a for a,b in zip(before,after)]
        assert all(v>=0 for v in deltas) and sum(deltas)>0
        # I/O wait counts as busy: the qualification is conservative.
        busy=1-deltas[3]/sum(deltas)
        intervals.append({'before':before,'after':after,'deltas':deltas,
            'elapsed_ns':elapsed,'busy_fraction':busy,'qualified':busy<=.05})
    result={'complete':True,'qualified':all(r['qualified'] for r in intervals),'cpu':args.cpu,
        'script_sha256':sha(Path(__file__)),'plan_sha256':sha(HERE/'SCREENING-PLAN.md'),
        'interval_seconds':3,'interval_count':3,'maximum_busy_fraction':.05,
        'iowait_policy':'counts as busy','intervals':intervals,'thread_siblings_list':siblings,
        'allowed_affinity':sorted(allowed),'topology_sha256':sha(destination/'lscpu.json'),
        'performance_samples_collected':0,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (destination/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'result':str((destination/'result.json').relative_to(ROOT)),
        'sha256':sha(destination/'result.json'),'qualified':result['qualified'],
        'busy_fractions':[r['busy_fraction'] for r in intervals]}))


if __name__=='__main__':main()
