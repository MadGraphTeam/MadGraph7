#!/usr/bin/env python3
"""One admitted bounded release build, using the existing owned-group runner."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.tirust-build/cayley-guarded-adaptive-build-proposal-2'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def interrupted(signum, frame):
    raise InterruptedError(f'Build interrupted by signal {signum}')

def deadline_expired(signum, frame):
    raise TimeoutError('The single 600-second build deadline expired')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-admitted', action='store_true', required=True)
    parser.parse_args()
    assert Path.cwd().resolve() == ROOT
    plan = json.loads((HERE / 'plan.json').read_text())
    started = time.monotonic()
    finish_by = started + plan['overall_seconds']
    out = WORK / 'attempt1'
    out.mkdir(exist_ok=False)
    cache = WORK / 'build-cache'
    target = WORK / 'target'
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    signal.signal(signal.SIGALRM, deadline_expired)
    record = {'schema': 'cayley-guarded-adaptive-build-v1', 'complete': False,
              'root_admission_received': True, 'environment': plan['environment'],
              'command': plan['build_command'], 'overall_seconds': plan['overall_seconds'],
              'cpu': 11, 'cpu_exclusive': False, 'jobs': 1,
              'address_space_limit_bytes_per_build_process': 8589934592,
              'address_space_limit_bytes_per_cache_copy_process': 4294967296,
              'automatic_retries': 0, 'loader_or_native_test_executed': False,
              'protected_inputs_before': {}, 'commands': [], 'started_unix': time.time()}
    pins = {}
    signal.setitimer(signal.ITIMER_REAL, max(0.001, finish_by - time.monotonic()))
    try:
        pins = load('guarded_source_verifier', HERE / 'verify_source.py').verify()
        record['protected_inputs_before'] = pins
        supervisor = load('existing_bounded_group_runner', ROOT / plan['supervisor'])
        # The outer absolute alarm must share the helper's atomic spawn/reap
        # signal deferral. Change only this imported module's in-memory tuple;
        # the reviewed helper source and every numerical source stay untouched.
        supervisor.SIGNALS = (*supervisor.SIGNALS, signal.SIGALRM)
        cache.mkdir(exist_ok=False)
        target.mkdir(exist_ok=False)
        for label, command in [('copy-cache', plan['cache_copy_command']), ('release-build', plan['build_command'])]:
            remaining = finish_by - time.monotonic()
            assert remaining > 0, 'Single build budget exhausted'
            progress = {}
            record['commands'].append({'label': label, 'command': command, 'progress': progress})
            with (out / f'{label}.log').open('w') as log:
                code = supervisor.run_plain_capped(command, log, cwd=str(ROOT), env=plan['environment'],
                                                   progress=progress, seconds=remaining)
            assert code == 0, (label, code)
            assert time.monotonic() < finish_by, 'Single build budget exhausted'
        # Demand actual recompiled core/bridge from private source manifests;
        # cached dependency reuse remains an observed Cargo property.
        artifacts = []
        for line in (out / 'release-build.log').read_text().splitlines():
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if message.get('reason') == 'compiler-artifact':
                artifacts.append(message)
        expected_manifests = {
            'iregi-rust': WORK / 'sources/pole-development/crates/iregi-rust/Cargo.toml',
            'madloop_tirust_adaptive_private': WORK / 'sources/adaptive-qp-rescue-prototype3/Cargo.toml',
        }
        selected = {}
        for name, manifest in expected_manifests.items():
            matches = [m for m in artifacts if m['target']['name'] == name.replace('-', '_')]
            assert len(matches) == 1, name
            message = matches[0]
            assert Path(message['manifest_path']).resolve() == manifest.resolve()
            assert not message['fresh'] and not message['profile']['test']
            assert message['profile']['opt_level'] == '3'
            selected[name] = message
        for message in artifacts:
            if message['package_id'].startswith('path+'):
                assert Path(message['manifest_path']).resolve().is_relative_to(WORK / 'sources')
        bridge = selected['madloop_tirust_adaptive_private']
        libraries = [Path(p) for p in bridge['filenames'] if p.endswith('.so')]
        assert len(libraries) == 1 and libraries[0].resolve().is_relative_to(target)
        runtime = out / 'runtime'
        runtime.mkdir()
        destination = runtime / 'libmadloop_tirust_adaptive_private.so'
        shutil.copyfile(libraries[0], destination)
        assert sha(destination) == sha(libraries[0])
        record['private_library_not_loaded'] = {'path': str(destination), 'sha256': sha(destination),
                                               'bytes': destination.stat().st_size}
        record['cargo_artifacts'] = artifacts
        record['protected_inputs_after'] = {name: sha(Path(name)) for name in pins}
        assert record['protected_inputs_after'] == pins, 'Protected input drift'
        assert time.monotonic() < finish_by, 'Single build budget exhausted'
        record['complete'] = True
        record['elapsed_seconds'] = time.monotonic() - started
        record['finished_unix'] = time.time()
        (out / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
        assert time.monotonic() < finish_by, 'Single build publication budget exhausted'
    except BaseException as error:
        record['complete'] = False
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        # The existing supervisor reaps owned children before unwinding here.
        # Failure-only metadata retention may finish after the deadline;
        # it never changes failure into a successful build qualification.
        signal.setitimer(signal.ITIMER_REAL, 0)
        elapsed = time.monotonic() - started
        if elapsed >= plan['overall_seconds']:
            record['complete'] = False
            record['budget_exhausted'] = True
        if not record['complete']:
            record['elapsed_seconds'] = elapsed
            record['finished_unix'] = time.time()
            record['failure_metadata_only'] = True
            record['protected_input_closure_complete'] = record.get('protected_inputs_after') == pins and bool(pins)
            (out / 'execution.json').write_text(json.dumps(record, indent=2) + '\n')
    assert record['complete']
    print(json.dumps({'complete': True, 'execution': str(out / 'execution.json'),
                      'private_library_not_loaded': record['private_library_not_loaded']}, indent=2))

if __name__ == '__main__':
    main()
