#!/usr/bin/env python3
"""Coordinated paired process trial with retained CPU/SMT interference samples."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def snapshot(cpus):
    wanted={'cpu'+str(cpu) for cpu in cpus}
    counters={}
    for row in Path('/proc/stat').read_text().splitlines():
        fields=row.split()
        if fields[0] in wanted:
            counters[fields[0]]=[int(value) for value in fields[1:]]
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        monotonic_seconds=time.monotonic(),cpu_jiffies=counters,
        scaling_khz={str(cpu):read(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq')
            for cpu in cpus},load_average=read('/proc/loadavg'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--cpu',type=int,default=3)
    parser.add_argument('--background-cpu',type=int,default=207)
    args=parser.parse_args()
    # This light sampling controller uses a different core from evaluations.
    os.sched_setaffinity(0,{2})
    monitor=HERE/(args.run_id+'-monitor')
    monitor.mkdir(exist_ok=False)
    cpu_root=Path(f'/sys/devices/system/cpu/cpu{args.cpu}')
    siblings=[int(value) for value in (cpu_root/'topology/thread_siblings_list')
              .read_text().strip().split(',')]
    cpus=sorted(set(siblings+[args.background_cpu]))
    topology=dict(evaluation_cpu=args.cpu,siblings=siblings,
        core_id=read(cpu_root/'topology/core_id'),
        package_id=read(cpu_root/'topology/physical_package_id'),
        numa_nodes=[p.name for p in cpu_root.glob('node*')],
        background_cpu=args.background_cpu,
        background_siblings=read(f'/sys/devices/system/cpu/cpu{args.background_cpu}/topology/thread_siblings_list'),
        background_numa_nodes=[p.name for p in Path(
            f'/sys/devices/system/cpu/cpu{args.background_cpu}').glob('node*')],
        kernel=list(os.uname()),clock_ticks=os.sysconf('SC_CLK_TCK'),
        scope='Locally coordinated quiet window: other task-owned numerical and '
        'compiler jobs paused except the explicitly allowed corrected-original '
        'reference worker on the background CPU. Shared host is not exclusively '
        'reserved; CPU/SMT activity is retained rather than assumed absent.')
    (monitor/'topology.json').write_text(json.dumps(topology,indent=2)+'\n')
    reference=HERE/'qcd3-sharing-trace-v2'
    command=[sys.executable,str(HERE/'run_shared.py'),'--case','qcd-3',
        '--variants','shared','--run-id',args.run_id,'--repetitions','7',
        '--loops','1','--cpu',str(args.cpu),'--quiet','--maximum-ratio','1.0',
        '--reference-values',str(reference/'original-0-values.dat'),
        '--reference-status',str(reference/'original-0-timed_status.dat')]
    (monitor/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    (monitor/'controller.py').write_bytes(Path(__file__).read_bytes())
    samples=[]
    stop=threading.Event()
    def sample():
        with (monitor/'samples.jsonl').open('w') as stream:
            while not stop.is_set():
                row=snapshot(cpus)
                samples.append(row)
                stream.write(json.dumps(row)+'\n')
                stream.flush()
                stop.wait(5)
    sampler=threading.Thread(target=sample)
    sampler.start()
    code=None
    started=time.monotonic()
    try:
        with (monitor/'run.log').open('w') as log:
            code=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
    finally:
        stop.set()
        sampler.join()
        final=snapshot(cpus)
        samples.append(final)
        with (monitor/'samples.jsonl').open('a') as stream:
            stream.write(json.dumps(final)+'\n')
        activity={}
        for cpu in cpus:
            key='cpu'+str(cpu)
            delta=[b-a for a,b in zip(samples[0]['cpu_jiffies'][key],
                                     samples[-1]['cpu_jiffies'][key])]
            # Linux guest fields are included in user/nice; avoid double count.
            total=sum(delta[:8])
            idle=delta[3]+delta[4]
            activity[key]=dict(jiffy_deltas=delta,busy_fraction=(
                (total-idle)/total if total else None))
        result=dict(returncode=code,wall_seconds=time.monotonic()-started,
            samples=len(samples),activity=activity,
            controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Activity includes compilation and initialization as well as '
            'timed evaluations. The per-five-second raw samples permit inspection '
            'of changing sibling/background load; no samples or slow pairs excluded.')
        (monitor/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(code)


if __name__=='__main__':
    main()
