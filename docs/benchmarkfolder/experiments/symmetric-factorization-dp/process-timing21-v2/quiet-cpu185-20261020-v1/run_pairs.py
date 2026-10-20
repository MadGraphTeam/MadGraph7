#!/usr/bin/env python3
"""Fixed21 paired original-Ninja process trial; requires a coordinated window."""
import sys
sys.dont_write_bytecode = True
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
sys.path[:0] = [str(ROOT/'tools'), str(BASE/'process-qcd3-v1')]
import run as control
import ninjars_benchmark_policy_v2 as policy
from ninjars_report_v2 import compare

WORK = ROOT/'.benchmark-work/experiments/symmetric-factorization-dp/process-timing21-v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def snapshot(cpus):
    wanted={'cpu'+str(c) for c in cpus}
    counters={}
    for line in Path('/proc/stat').read_text().splitlines():
        row=line.split()
        if row[0] in wanted:
            counters[row[0]]=list(map(int,row[1:]))
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        monotonic=time.monotonic(),cpu_jiffies=counters,loadavg=read('/proc/loadavg'),
        frequency_khz={str(c):read(f'/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq') for c in cpus})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--cpu',type=int,required=True)
    parser.add_argument('--quiet-confirmed',action='store_true',required=True)
    args=parser.parse_args()
    os.sched_setaffinity(0,{2})
    report=HERE/args.run_id
    report.mkdir(exist_ok=False)
    prepared=json.loads((HERE/'preparation.json').read_text())
    assert prepared['complete'] and not prepared['amplitudes_evaluated']
    assert sha(HERE/'PLAN.md')==prepared['plan_sha256']
    reference=json.loads((control.REFERENCE/'data/candidate-qcd-3-6-0.json').read_text())
    exact=json.loads((BASE/'process-qcd3-v1/backend-6/result.json').read_text())
    assert exact['complete'] and sha(BASE/'process-qcd3-v1/backend-6/result.json')==prepared['prerequisite_ninja6_sha256']
    for variant,built in prepared['builds'].items():
        output=WORK/variant
        sub=next(output.glob('SubProcesses/P*_*'))
        assert sha(sub/'check')==built['executable_sha256']
        assert control.bench.generated_source_manifest(output)['files_sha256']==built['generated_sources_sha256']
        assert policy.library_hashes(output)==built['libraries']
        assert {n:policy.bound_file(p) for n,p in policy.runtime_metadata_files(output).items()}==built['metadata']
        assert {n:sha(output/n) for n in policy.RUNTIME_LINK_INPUTS}==built['link_inputs']
    siblings=[]
    for group in Path(f'/sys/devices/system/cpu/cpu{args.cpu}/topology/thread_siblings_list').read_text().strip().split(','):
        values=list(map(int,group.split('-')))
        siblings.extend(range(values[0],values[-1]+1))
    cpus=sorted(set(siblings+[207]))
    first=snapshot(cpus)
    time.sleep(5)
    last=snapshot(cpus)
    idle={}
    for c in siblings:
        delta=[b-a for a,b in zip(first['cpu_jiffies']['cpu'+str(c)],last['cpu_jiffies']['cpu'+str(c)])]
        idle[str(c)]=(delta[3]+delta[4])/sum(delta[:8])
    passed=idle[str(args.cpu)]>=.90 and all(v>=.80 for v in idle.values())
    dump(report/'preflight.json',dict(cpu=args.cpu,siblings=siblings,idle=idle,passed=passed,snapshots=[first,last],
        core_id=read(f'/sys/devices/system/cpu/cpu{args.cpu}/topology/core_id'),
        package_id=read(f'/sys/devices/system/cpu/cpu{args.cpu}/topology/physical_package_id'),
        numa_nodes=[p.name for p in Path(f'/sys/devices/system/cpu/cpu{args.cpu}').glob('node*')],
        background_cpu=207,background_siblings=read('/sys/devices/system/cpu/cpu207/topology/thread_siblings_list'),
        kernel=list(os.uname()),clock_ticks=os.sysconf('SC_CLK_TCK')))
    assert passed,'CPU preflight rejected; no amplitude launched'
    dump(report/'protocol.json',dict(plan_sha256=sha(HERE/'PLAN.md'),preparation_sha256=sha(HERE/'preparation.json'),
        runner_sha256=sha(Path(__file__)),cpu=args.cpu,pairs=21,loops=1,points=100,backend=6,
        original_first_on_odd_pairs=True,all_samples_retained=True,quiet_confirmed=True,
        point_sha256=sha(control.POINTS),reference_sha256=sha(control.REFERENCE/'data/candidate-qcd-3-6-0.json')))
    (report/'run_pairs.py').write_bytes(Path(__file__).read_bytes())
    control.WORK=WORK
    result=dict(complete=False,acceptance_eligible=False,all_samples_retained=True,measurements=[],comparisons=[])
    dump(report/'result.json',result)
    stop=threading.Event()
    def monitor():
        with (report/'activity.jsonl').open('x') as stream:
            while not stop.is_set():
                stream.write(json.dumps(snapshot(cpus))+'\n')
                stream.flush()
                stop.wait(5)
            stream.write(json.dumps(snapshot(cpus))+'\n')
    watcher=threading.Thread(target=monitor)
    watcher.start()
    try:
        for pair in range(1,22):
            groups={}
            order=('original','factor') if pair%2 else ('factor','original')
            for position,variant in enumerate(order,1):
                target=report/f'pair-{pair:02d}-{variant}'
                target.mkdir()
                settings=control.configure(variant,6,reference,target)
                sub=control.directory(variant)
                for name in ('values.dat','timing.dat','timed_status.dat','dimensions.dat'):
                    (sub/name).unlink(missing_ok=True)
                record=control.command(['taskset','-c',str(args.cpu),'./check',str(control.POINTS),'1'],sub,target/'run.log')
                record.update(pair=pair,position=position,variant=variant,shared_numerical_settings=settings)
                result['measurements'].append(record)
                dump(report/'result.json',result)
                for name in ('values.dat','timing.dat','timed_status.dat','dimensions.dat'):
                    if (sub/name).is_file():
                        shutil.copy2(sub/name,target/name)
                assert record['returncode']==0
                dims=list(map(int,(target/'dimensions.dat').read_text().split()))
                rows=control.read_rows(target/'values.dat',6,dims)
                expected=exact['measurements']['original' if variant=='original' else 'factor-trace']['rows']
                assert rows==expected,'Uninstrumented rows/status/accuracy changed from correctness control'
                times=list(map(float,(target/'timing.dat').read_text().split()))
                assert len(times)==2 and all(math.isfinite(t) and t>0 for t in times)
                histogram={}
                for line in (target/'timed_status.dat').read_text().splitlines():
                    code,count=map(int,line.split())
                    control.bench.validate_status(code,6)
                    assert count>0 and str(code) not in histogram
                    histogram[str(code)]=count
                assert sum(histogram.values())==100
                record.update(rows=rows,dimensions=dims,timed_status_counts=histogram,
                    cold_seconds=times[0],timed_seconds=times[1],seconds_per_point=times[1]/100,
                    values_sha256=sha(target/'values.dat'),exact_correctness_stream=True)
                groups[variant]=rows
                dump(report/'result.json',result)
            checked=compare(groups)
            result['comparisons'].append(checked)
            assert checked['passed']
            dump(report/'result.json',result)
            print('pair',pair,'of21 complete',flush=True)
        assert len(result['measurements'])==42 and len(result['comparisons'])==21
        import numpy as np
        original=np.array([next(r['seconds_per_point'] for r in result['measurements'] if r['pair']==p and r['variant']=='original') for p in range(1,22)])
        factored=np.array([next(r['seconds_per_point'] for r in result['measurements'] if r['pair']==p and r['variant']=='factor') for p in range(1,22)])
        indices=np.random.default_rng(410083).integers(0,21,size=(10000,21))
        distribution=np.median(factored[indices],axis=1)/np.median(original[indices],axis=1)
        result['statistics']=dict(ratio_of_medians=float(np.median(factored)/np.median(original)),
            ci95=np.quantile(distribution,[.025,.975]).tolist(),pair_ratios=(factored/original).tolist(),
            original_median_seconds_per_point=float(np.median(original)),factor_median_seconds_per_point=float(np.median(factored)),
            bootstrap_seed=410083,bootstrap_resamples=10000)
        result['complete']=True
    except Exception as error:
        result['error']=repr(error)
        raise
    finally:
        stop.set()
        watcher.join()
        dump(report/'result.json',result)


if __name__=='__main__':
    main()
