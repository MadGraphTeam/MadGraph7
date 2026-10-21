#!/usr/bin/env python3
"""Bound the already reviewed control operations and retain owned processes."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
REPORT = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path[:0] = [str(REPORT), str(ROOT/'tools'), str(ROOT)]
import execute
import ninjars_benchmark as bench


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def identity(pid):
    raw = (Path('/proc')/str(pid)/'stat').read_text()
    return dict(pid=pid, pgid=os.getpgid(pid), start_ticks=int(raw[raw.rfind(')')+2:].split()[19]))


def freeze():
    subprocess.run(['git', 'diff', '--exit-code', 'b0ea105a04afa34304d437d2956e4d4672360670',
                    '--', *bench.GENERATOR_PATHS, 'vendor/ninjars'], cwd=ROOT, check=True,
                   stdout=subprocess.DEVNULL)
    if bench.content_digest(bench.generator_tree_state()) != '34aed45638a2e18b18135179e749317c3cf9a0fdd3bb404e43fe68d23837c6de':
        raise RuntimeError('Final generator inputs changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'compare-rotation'))
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--authorization', required=True)
    args = parser.parse_args()
    if os.sched_getaffinity(0) != {190}:
        raise RuntimeError('This handoff owns only CPU190')
    if os.getpgrp() != os.getpid():
        os.setsid()
    os.environ.update(TMPDIR=str(ROOT/'.ninjars-build/tmp'), SYMBOLICA_LOCK_DIR=str(ROOT/'.ninjars-build/symbolica-locks'),
                      OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', RAYON_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      PYTHONDONTWRITEBYTECODE='1')
    ledger = REPORT/'ledgers'/args.action
    ledger.mkdir(parents=True, exist_ok=False)
    write(ledger/'launch.json', dict(schema='final-variant-owned-phase-v1', controller=identity(os.getpid()),
        authorization=args.authorization, action=args.action, started_utc=stamp(), timeout_seconds=21600,
        orchestration_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        generation_revision='b0ea105a04afa34304d437d2956e4d4672360670',
        numerical_revision='d05b59f3277ce0d727cb2816d8b7ad45107e58c4',
        source={str(p.relative_to(ROOT)):execute.sha(p) for p in (Path(__file__), REPORT/'execute.py', REPORT/'controls_driver.py')},
        binding=str(args.binding) if args.binding else None))
    original_popen = subprocess.Popen
    processes = []
    lock = threading.Lock()
    def tracked_popen(*values, **kwargs):
        process = original_popen(*values, **kwargs)
        try:
            record = identity(process.pid)
        except (FileNotFoundError, ProcessLookupError):
            record = dict(pid=process.pid, already_exited=True)
        command = kwargs.get('args', values[0] if values else None)
        if isinstance(command, (list, tuple)):
            command = [str(x) for x in command]
        record.update(command=command, cwd=str(kwargs.get('cwd', ROOT)), started_utc=stamp())
        with lock:
            processes.append((process, record))
            with (ledger/'descendant-starts.jsonl').open('a') as stream:
                stream.write(json.dumps(record)+'\n')
        return process
    subprocess.Popen = tracked_popen
    def interrupted(signum, _frame):
        raise TimeoutError('Variant phase interrupted by signal '+str(signum))
    for signum in (signal.SIGALRM, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, interrupted)
    signal.alarm(21600)
    start = time.monotonic()
    print(json.dumps(dict(action=args.action, controller=identity(os.getpid()))), flush=True)
    try:
        freeze()
        if args.action == 'prepare':
            execute.prepare()
        elif args.action == 'run':
            execute.run(['control-rotation', 'control-rebinding', 'control-rebinding-fresh', 'control-near-singular'], 190)
        else:
            if args.binding is None:
                raise RuntimeError('compare-rotation requires the actual fresh HH binding')
            execute.compare_rotation(args.binding)
        freeze()
        write(ledger/'result.json', dict(status='passed', action=args.action, seconds=time.monotonic()-start,
              finished_utc=stamp(), performance_acceptance_claim=False))
    except BaseException as error:
        write(ledger/'failure.json', dict(status='failed', action=args.action, error_type=type(error).__name__,
              error=str(error), finished_utc=stamp()))
        (ledger/'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        signal.alarm(0)
        subprocess.Popen = original_popen
        terminated, remaining = [], []
        for process, record in processes:
            if process.poll() is not None:
                continue
            try:
                if identity(process.pid)['start_ticks'] != record.get('start_ticks'):
                    continue
                if os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait(timeout=10)
                terminated.append(record)
            except (FileNotFoundError, ProcessLookupError):
                pass
            if process.poll() is None:
                remaining.append(record)
        write(ledger/'children-exited.json', dict(checked_utc=stamp(), all_children_exited=not remaining,
              exceptionally_terminated=terminated, remaining=remaining))
        if remaining:
            raise RuntimeError('Owned variant children remain')


if __name__ == '__main__':
    main()
