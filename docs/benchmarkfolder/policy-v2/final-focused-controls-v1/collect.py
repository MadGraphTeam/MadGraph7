#!/usr/bin/env python3
"""Collect only the explicitly selected final focused targets and bind results."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

sys.dont_write_bytecode = True
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
REPORT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT),str(ROOT/'tools')]
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_references_v2 as references

EXPECTED = '8b454519cb6818ef1fafe37378930c276f7e85394a2333ebc9fdb00e75726cff'
TREE = '34aed45638a2e18b18135179e749317c3cf9a0fdd3bb404e43fe68d23837c6de'
ARCHIVE = '800a941faf3cdfdab764fdffe1d5ab73f84984a786aa194596b753af6b6f5d42'


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def identity(pid):
    raw = (Path('/proc')/str(pid)/'stat').read_text()
    fields = raw[raw.rfind(')')+2:].split()
    return dict(pid=pid,pgid=os.getpgid(pid),start_ticks=int(fields[19]))


def freeze():
    if subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip() != 'madloop_ninjars':
        raise RuntimeError('Wrong branch')
    subprocess.run(['git','diff','--exit-code','b0ea105a04afa34304d437d2956e4d4672360670','--',*bench.GENERATOR_PATHS,'vendor/ninjars'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    if bench.content_digest(bench.generator_tree_state()) != TREE:
        raise RuntimeError('Final numerical generator inputs changed')
    if references.workers.digest(ROOT/'HEPTools/ninjars/lib/libninjars.a') != ARCHIVE:
        raise RuntimeError('Final installed archive changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection',choices=('complex','mixed-legacy'))
    parser.add_argument('--authorization',required=True)
    args = parser.parse_args()
    spec_path = REPORT/'specs'/(args.collection+'.json')
    spec = policy.read(spec_path)
    if os.sched_getaffinity(0) != {spec['cpu']}:
        raise RuntimeError('Use the declared singleton CPU affinity')
    if os.getpgrp() != os.getpid():os.setsid()
    os.environ.update(TMPDIR=str(ROOT/'.ninjars-build/tmp'),SYMBOLICA_LOCK_DIR=str(ROOT/'.ninjars-build/symbolica-locks'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',RAYON_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    ledger = REPORT/'ledgers'/args.collection
    ledger.mkdir(parents=True,exist_ok=False)
    write(ledger/'launch.json',dict(schema='final-focused-collection-launch-v1',collection=args.collection,
        started_utc=stamp(),controller=identity(os.getpid()),authorization=args.authorization,
        orchestration_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        generation_revision='b0ea105a04afa34304d437d2956e4d4672360670',numerical_revision='d05b59f3277ce0d727cb2816d8b7ad45107e58c4',
        spec=policy.bound_file(spec_path),cpu=spec['cpu'],backends=spec['backends'],phase_timeout_seconds=21600))
    print(json.dumps({'collection':args.collection,'controller':identity(os.getpid())}),flush=True)
    original_popen = subprocess.Popen
    lock = threading.Lock()
    processes = []
    def tracked_popen(*values,**kwargs):
        process = original_popen(*values,**kwargs)
        try:record = identity(process.pid)
        except (FileNotFoundError,ProcessLookupError):record = dict(pid=process.pid,already_exited=True)
        command = kwargs.get('args',values[0] if values else None)
        if isinstance(command,(list,tuple)):command = [str(x) for x in command]
        record.update(command=command,cwd=str(kwargs.get('cwd',ROOT)),started_utc=stamp())
        with lock:
            processes.append((process,record))
            with (ledger/'descendant-starts.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        return process
    subprocess.Popen = tracked_popen
    def interrupted(signum,_frame):
        raise TimeoutError('Focused collection interrupted by signal '+str(signum))
    signal.signal(signal.SIGALRM,interrupted)
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGHUP,interrupted)
    signal.alarm(21600)
    start = time.monotonic()
    try:
        freeze()
        inventory = policy.verified(policy.read(ROOT/'docs/benchmarkfolder/policy-v2/final-core-generation-v1/runtime-inventory.json'),'fresh-generated-runtime-inventory-v2')
        signature = inventory['runtime_groups']['8'][EXPECTED]['signature']
        binding_spec = references.collect_timing(spec_path)
        run = policy.path(binding_spec['report'])
        role_path = run/'role-bindings.json'
        correction = ROOT/'docs/benchmarkfolder/policy-v2/model-correction.json'
        policy.bind_run(binding_spec['report'],binding_spec['role_map'],binding_spec['outputs'],correction,role_path)
        runtime_path = REPORT/'runtime-bindings'/(args.collection+'.json')
        policy.refresh_runtime_bindings(role_path,correction,runtime_path)
        binding = policy.checked_bindings(runtime_path,policy.validate_correction(correction),live=True)
        protocol,records = references.workers.checked_series(run,minimum=2)
        targets = {}
        for key,target in binding['targets'].items():
            if target['reducer_runtime'] != signature:
                raise RuntimeError('Focused target differs from the complete final runtime')
            first = policy.read(run/'data'/(key+'-0.json'))
            fresh = policy.read(run/'data'/(key+'-fresh.json'))
            targets[key] = dict(status_counts=first['status_counts'],dimensions=first['answer_dimensions'],
                parity=references.workers.numerical_parity(first,fresh,protocol['configuration']),
                runtime=target['reducer_runtime']['identity_sha256'],executable=target['executable_sha256'])
        freeze()
        write(ledger/'result.json',dict(status='passed',collection=args.collection,finished_utc=stamp(),
            seconds=time.monotonic()-start,runtime_binding=policy.bound_file(runtime_path),targets=targets,
            records=len(records),performance_acceptance_claim=False))
        print(json.dumps({'collection':args.collection,'status':'passed','targets':targets}),flush=True)
    except BaseException as error:
        write(ledger/'failure.json',dict(status='failed',error_type=type(error).__name__,error=str(error),finished_utc=stamp()))
        (ledger/'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        subprocess.Popen = original_popen
        remaining = []
        for process,record in processes:
            if process.poll() is not None:continue
            try:
                if identity(process.pid)['start_ticks'] != record.get('start_ticks'):continue
                if os.getpgid(process.pid) == process.pid:os.killpg(process.pid,signal.SIGKILL)
                else:process.kill()
                process.wait(timeout=10)
            except (FileNotFoundError,ProcessLookupError):pass
            if process.poll() is None:remaining.append(record)
        write(ledger/'children-exited.json',dict(checked_utc=stamp(),all_children_exited=not remaining,remaining=remaining))
        if remaining:raise RuntimeError('Owned focused children remain')


if __name__ == '__main__':
    main()
