#!/usr/bin/env python3
"""Retain one exact final-release timing controller; never retry its pairs."""
import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())
sys.path.insert(0, str(ROOT / 'tools'))
import ninjars_benchmark_pair_v3 as pair
import ninjars_benchmark_policy_v2 as policy
import ninjars_supplement_v2 as supplement


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


owner = load('final_timing_core_owner', BASE / 'core-correctness/launch.py')
SPECS = {
    'core': (BASE / 'timing-pair/specification-held-v1.json',
             'de65bb6bb168bf427be3fc5c9fc1cb2eb2dc77a386c24528015228a1c4e8f813'),
    'loop': (BASE / 'loop-execution/specification-pair-held-v2.json',
             '56cee094b28452ba481c998410d6095b0f46d4737f6b200dea31587cf08c2d47'),
}


def read(path):
    return json.loads(Path(path).read_text())


def prepare(stage, handoff, quiet_proof):
    """Pure source/input checks. Neither collector nor an engine is started."""
    specification, expected = SPECS[stage]
    if supplement.digest(specification) != expected:
        raise ValueError('The exact reviewed timing specification changed')
    quiet = read(quiet_proof)
    required = {'schema': 'final-timing-quiet-handoff-v1', 'root_authorized': True,
                'stage': stage, 'cpu': 207, 'quiet_cpu207': True,
                'all_other_numerical_and_compiler_work_released': True,
                'specification_sha256': expected,
                'collector_handoff_sha256': supplement.digest(handoff)}
    if any(quiet.get(k) != v or (type(v) is bool and type(quiet.get(k)) is not bool)
           for k, v in required.items()):
        raise ValueError('A matching explicit root quiet handoff is required')
    evidence = quiet.get('release_evidence')
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('The quiet handoff must bind retained release evidence')
    for item in evidence:
        supplement.checked(item)
    authorization = read(handoff)
    if stage == 'core':
        spec = pair.validate_spec(read(specification))
        config = read(supplement.checked(spec['configuration']))
        correction = policy.validate_correction(supplement.checked(spec['correction']), live=True)
        pair.preflight(spec, config, correction)
        required_handoff = {'root_authorized': True, 'phase': pair.PAIR_SCHEMA,
                            'cpu': 207, 'specification_sha256': expected}
        command = [sys.executable, '-B', str(ROOT / 'tools/ninjars_benchmark_pair_v3.py'),
                   '--spec', str(specification), '--execute-numerical']
    else:
        loop = load('final_timing_loop_collector', BASE / 'loop-execution/loop_pair.py')
        spec, _ = loop.validate_spec(read(specification))
        required_handoff = {'root_authorized': True, 'phase': loop.SCHEMA, 'cpu': 207,
                            'specification_sha256': expected, 'quiet_cpu207': True}
        command = [sys.executable, '-B', str(BASE / 'loop-execution/loop_pair.py'),
                   'collect', '--spec', str(specification), '--execute-numerical']
    if authorization != required_handoff:
        raise ValueError('The collector handoff does not match this exact stage')
    if spec['pair_targets'] != [['baseline', 6], ['candidate', 8]] or spec['repetitions'] != 7:
        raise ValueError('Only the reviewed seven Ninja6/NinjaRS8 pairs may run')
    report, work = policy.path(spec['report']), policy.path(spec['work'])
    if report.exists() or work.exists():
        raise ValueError('Retain the existing attempt; there is no automatic restart or append')
    if not {207}.issubset(os.sched_getaffinity(0)) or not os.sched_getaffinity(0).issubset({6, 7, 207}):
        raise ValueError('Use taskset -c6,7,207; numerical workers alone are pinned207')
    return {'stage': stage, 'specification': supplement.bound(specification),
            'collector_handoff': supplement.bound(handoff), 'quiet_proof': supplement.bound(quiet_proof),
            'candidate_cohort': spec['candidate_cohort'], 'report': spec['report'], 'work': spec['work'],
            'command': command, 'sources': {name: supplement.bound(path) for name, path in
                [('launcher', Path(__file__)), ('ownership', BASE / 'core-correctness/launch.py'),
                 ('ownership_core_import', BASE / 'core-correctness/core_collect.py'),
                 ('pair_import', ROOT / 'tools/ninjars_benchmark_pair_v3.py')]}}


def terminal_ledger(report):
    """A successful controller must leave every owned group reaped and absent."""
    filename = report / 'process-ledger.jsonl'
    events = [json.loads(line) for line in filename.read_text().splitlines()]
    spawned, reaped = {}, {}
    for event in events:
        identity = tuple(event[k] for k in ('pid', 'pgid', 'start_ticks'))
        if any(type(value) is not int or value <= 0 for value in identity) or identity[0] != identity[1]:
            raise ValueError('Invalid owned child identity')
        if event['event'] == 'spawn':
            if identity in spawned:
                raise ValueError('Duplicate owned spawn')
            spawned[identity] = event
        elif event['event'] == 'reaped':
            if (identity not in spawned or identity in reaped or event.get('reaped') is not True or
                    type(event.get('returncode')) is not int or event['returncode'] != 0):
                raise ValueError('Unmatched or unsuccessful owned reap')
            reaped[identity] = event
        else:
            raise ValueError('Unknown process custody event')
    if not spawned or set(spawned) != set(reaped):
        raise ValueError('Missing owned child reap')
    for pid, pgid, ticks in spawned:
        actual = owner.identity(pid)
        if actual and actual['start_ticks'] == ticks:
            raise RuntimeError('An owned child remains live')
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('An owned process group remains present')
    status = read(report / 'run-status.json')
    if (status.get('status') != 'complete' or not status.get('requested_cases') or
            status.get('requested_cases') != status.get('completed_cases')):
        raise ValueError('The actual timing series is incomplete')
    return {'ledger': supplement.bound(filename), 'spawned_groups': len(spawned),
            'all_reaped_exit0_and_absent': True, 'status': supplement.bound(report / 'run-status.json')}


def run_owned(command, report, attempt, metadata):
    """Retain ownership before delivering any interruption received during spawn."""
    attempt.mkdir(parents=True, exist_ok=False)
    child = owned = None
    started = time.monotonic()
    interruptions = []
    deferred = True
    completed = False
    ownership = None
    failure = None
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}

    def interrupted(signum, _frame):
        interruptions.append({'utc': datetime.datetime.now(datetime.UTC).isoformat(), 'signal': signum})
        if not deferred:
            raise KeyboardInterrupt('Timing launcher interrupted; owned cleanup follows')

    for sig in previous:
        signal.signal(sig, interrupted)
    env = dict(os.environ)
    env.update({name: '1' for name in pair.THREADS})
    try:
        with (attempt / 'stdout').open('xb') as out, (attempt / 'stderr').open('xb') as err:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
            owned = owner.identity(child.pid)
            if owned is None or owned['pgid'] != child.pid:
                raise RuntimeError('Controller ownership could not be recorded')
            owner.save(attempt / 'launch.json', {**metadata, 'controller_identity': owned,
                'launcher_identity': owner.identity(os.getpid()), 'command': command,
                'started_utc': datetime.datetime.now(datetime.UTC).isoformat(),
                'thread_environment': {name: '1' for name in pair.THREADS},
                'worker_cpu': 207, 'automatic_retry': False, 'in_place_resume': False})
            deferred = False
            if interruptions:
                raise KeyboardInterrupt('Deferred interruption after durable controller ownership')
            code = child.wait()
            if code:
                owner.cleanup_verified_workers(report, attempt)
            else:
                ownership = terminal_ledger(report)
                if owner.identity(child.pid) is not None:
                    raise RuntimeError('The reaped controller identity remains present')
                try:
                    os.killpg(child.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError('The controller process group survives its leader')
                completed = True
            return code
    except BaseException as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
        deferred = True
        if child is not None and child.poll() is None:
            actual = owner.identity(child.pid)
            if actual is None or (owned is not None and actual['start_ticks'] != owned['start_ticks']):
                raise RuntimeError('Refusing to signal a controller with unavailable/reused identity') from error
            child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        owner.cleanup_verified_workers(report, attempt)
        raise
    finally:
        deferred = True
        try:
            if child is not None:
                controller_absent = owner.identity(child.pid) is None
                try:
                    os.killpg(child.pid, 0)
                except ProcessLookupError:
                    group_absent = True
                else:
                    group_absent = False
                owner.save(attempt / 'result.json', {'status': 'complete' if completed else 'execution-not-complete',
                    'returncode': child.returncode, 'elapsed_seconds': time.monotonic() - started,
                    'finished_utc': datetime.datetime.now(datetime.UTC).isoformat(),
                    'interruptions': interruptions, 'failure': failure,
                    'controller_identity': owned, 'controller_absent': controller_absent,
                    'controller_group_absent': group_absent, 'terminal_custody': ownership,
                    'automatic_retry': False, 'in_place_resume': False,
                    'scientific_gate_decision': 'Requires the existing pair authority; this is an execution receipt only'})
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=tuple(SPECS))
    parser.add_argument('--handoff', required=True, type=Path)
    parser.add_argument('--quiet-proof', required=True, type=Path)
    parser.add_argument('--attempt', required=True, type=Path)
    parser.add_argument('--execute-numerical', action='store_true')
    args = parser.parse_args()
    if not args.execute_numerical:
        raise ValueError('No controller launch without explicit numerical execution')
    metadata = prepare(args.stage, args.handoff, args.quiet_proof)
    if args.attempt.exists():
        raise ValueError('Preserve the retained launch attempt')
    # The collector receives its exact-byte retained handoff, not mutable ambient text.
    args.attempt.parent.mkdir(parents=True, exist_ok=True)
    handoff = args.attempt.parent / (args.attempt.name + '-collector-handoff.json')
    quiet = args.attempt.parent / (args.attempt.name + '-quiet-proof.json')
    for source, target in ((args.handoff, handoff), (args.quiet_proof, quiet)):
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
    metadata['retained_handoff'] = supplement.bound(handoff)
    metadata['retained_quiet_proof'] = supplement.bound(quiet)
    command = [*metadata['command'], '--handoff', str(handoff)]
    return run_owned(command, policy.path(metadata['report']), args.attempt, metadata)


if __name__ == '__main__':
    raise SystemExit(main())
