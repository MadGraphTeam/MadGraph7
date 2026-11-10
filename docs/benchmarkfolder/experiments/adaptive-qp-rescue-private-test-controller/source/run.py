#!/usr/bin/env python3
"""Prepared only: a separately authorized one-shot private Rust test process."""
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import sys
import time

ROOT = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent
BUILD = ROOT / 'docs/benchmarkfolder/experiments/adaptive-qp-rescue-private-build'
BUILD_INDEX = '1a4af7f8f8be9ce8880ebc9ce466f66afec961be4c24b6978cdd8fdfa28fc8d8'
RUNNER = ROOT / 'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2'
RUNNER_INDEX = '69f76225a439dd325c4c08369a46ae5654102f1a95c19d6c3e7ca83682f0a3f4'
OUT = ROOT / 'docs/benchmarkfolder/experiments/adaptive-qp-rescue-private-tests/attempt1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(folder, name, expected=None):
    index = folder / name
    if expected:
        assert sha(index) == expected
    entries = json.loads(index.read_text())['files']
    paths = list(folder.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(folder)) for p in paths if p.is_file()} == set(entries) | {name}
    result = {str(index): sha(index)}
    for relative, entry in entries.items():
        path = folder / relative
        assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
        assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
        result[str(path)] = entry['sha256']
    return result


def interrupted(signum, frame):
    raise InterruptedError(f'Test controller received signal {signum}')


def main():
    assert ROOT.name == 'madgraph7_madloop_tirust'
    pins = inventory(BUILD, 'evidence-index.json', BUILD_INDEX)
    pins.update(inventory(RUNNER, 'source-index.json', RUNNER_INDEX))
    pins.update(inventory(HERE, 'source-index.json'))
    build = json.loads((BUILD / 'attempt-3/execution.json').read_text())
    assert build['complete'] and not build['native_tests_executed']
    assert build['protected_inputs_before'] == build['protected_inputs_after']
    pins.update(build['protected_inputs_after'])
    artifact = build['test_executable_not_run']
    binary = Path(artifact['cargo_record']['executable'])
    assert artifact['sha256'] == 'd22953ed3b955adf7dcce3831259060cec9e974816d432a4a86fd000b1afe8b7'
    assert sha(binary) == artifact['sha256']
    pins[str(binary)] = artifact['sha256']
    assert all(sha(Path(name)) == digest for name, digest in pins.items())
    env = build['environment'].copy()
    assert all(env[name] == '1' for name in ('RUST_TEST_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'))
    taskset = shutil.which('taskset', path=env['PATH'])
    assert taskset and str(Path(taskset).resolve()) in pins
    command = [taskset, '-c', '11', str(binary), '--test-threads=1', '--nocapture']
    spec = importlib.util.spec_from_file_location('reviewed_adaptive_test_supervisor', RUNNER / 'supervise.py')
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    # Busy admission consumes no attempt and never starts a native process.
    with (ROOT / '.tirust-build/numerical-tests.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Shared numerical lock is busy; no attempt created and no retry loop.')
            return 75
        assert all(sha(Path(name)) == digest for name, digest in pins.items())
        OUT.mkdir(parents=True, exist_ok=False)
        record = {'schema': 1, 'build_index_sha256': BUILD_INDEX,
                  'source_index_sha256': build['source_index_sha256'],
                  'test_executable_sha256': artifact['sha256'], 'command': command,
                  'cwd': str(ROOT), 'environment': env, 'cpu': 11, 'threads': 1,
                  'inclusive_deadline_seconds': 180, 'automatic_retry_count': 0,
                  'started_unix': time.time(), 'complete': False,
                  'protected_inputs_before': pins, 'process': {},
                  'physical_amplitude_qualified': False, 'production_promoted': False}
        try:
            with (OUT / 'tests.log').open('w') as stream:
                code = supervisor.run_plain_capped(command, stream, cwd=str(ROOT), env=env,
                    progress=record['process'], seconds=180)
            record['returncode'] = code
            assert code == 0, f'Tests returned {code}; no automatic retry.'
            log = (OUT / 'tests.log').read_text()
            result = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out;', log)
            assert result == [('20', '0', '0', '0', '0')], result
            names = re.findall(r'^test ([A-Za-z0-9_:]+) \.\.\. (?:ok|.*)$', log, re.M)
            assert len(names) == 20 and len(set(names)) == 20
            record['test_names'] = names
            record['test_result'] = {'passed': 20, 'failed': 0, 'ignored': 0}
            record['complete'] = True
        except BaseException as error:
            record['error'] = f'{type(error).__name__}: {error}'
            raise
        finally:
            record['finished_unix'] = time.time()
            record['protected_inputs_after'] = {name: sha(Path(name)) for name in pins}
            if record['protected_inputs_after'] != pins:
                record['complete'] = False
                record['protected_input_drift'] = True
            record['log_sha256'] = sha(OUT / 'tests.log') if (OUT / 'tests.log').exists() else None
            (OUT / 'execution.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        assert record['complete']
        print(json.dumps({'complete': True, 'passed': 20, 'physical_amplitude_qualified': False}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
