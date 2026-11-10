#!/usr/bin/env python3
"""One compile-only attempt with the frozen, reviewed owned-process supervisor."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import sys
import time

ROOT = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / '.tirust-build/adaptive-qp-rescue-prototype2'
RUNNER = ROOT / 'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2'
SOURCE_INDEX = '4655bdd1d12271d65c2bfb10256ed9d52fa28acc1dde048097bb169fca28c829'
RUNNER_INDEX = '69f76225a439dd325c4c08369a46ae5654102f1a95c19d6c3e7ca83682f0a3f4'
OUT = ROOT / '.tirust-build/adaptive-qp-rescue-build/attempt2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(folder, index_name, expected=None):
    index = folder / index_name
    if expected is not None:
        assert sha(index) == expected, str(index)
    entries = json.loads(index.read_text())['files']
    actual = list(folder.rglob('*'))
    assert not any(p.is_symlink() for p in actual)
    assert {str(p.relative_to(folder)) for p in actual if p.is_file()} == set(entries) | {index_name}
    pins = {str(index): sha(index)}
    for name, record in entries.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        path = folder / name
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256'], str(path)
        pins[str(path)] = record['sha256']
    return pins


def interrupted(signum, frame):
    raise InterruptedError(f'Controller received signal {signum}')


def main():
    assert ROOT.name == 'madgraph7_madloop_tirust'
    pins = inventory(SOURCE, 'source-index.json', SOURCE_INDEX)
    pins.update(inventory(RUNNER, 'source-index.json', RUNNER_INDEX))
    pins.update(inventory(HERE, 'source-index.json'))
    for name, digest in json.loads((SOURCE / 'source-bindings.json').read_text())['files'].items():
        path = ROOT / name
        assert sha(path) == digest, name
        pins[str(path)] = digest
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    assert protocol['cpu'] == 11 and protocol['compile_seconds'] == 900
    assert protocol['automatic_retry_count'] == 0 and not protocol['native_authorized']
    env_script = ROOT / protocol['environment_script']
    pins[str(env_script)] = sha(env_script)
    # The caller sources the recorded toolchain script. Compiler injection,
    # loader overrides and credentials are not inherited into the build.
    allowed = ['PATH', 'HOME', 'USER', 'LOGNAME', 'TMPDIR', 'LANG', 'LC_ALL',
               'CARGO_HOME', 'RUSTUP_HOME', 'XDG_CACHE_HOME', 'SYMBOLICA_LOCK_DIR',
               'SYMBOLICA_HIDE_BANNER', 'RUST_MIN_STACK', 'RUST_TEST_THREADS',
               'CC', 'CXX', 'FC']
    env = {name: os.environ[name] for name in allowed if name in os.environ}
    env.update(protocol['environment'])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['RUST_TEST_THREADS'] = '1'
    cargo = shutil.which('cargo', path=env['PATH'])
    rustc = shutil.which('rustc', path=env['PATH'])
    taskset = shutil.which('taskset', path=env['PATH'])
    assert cargo and rustc and taskset
    for value in [cargo, rustc, taskset, env.get('CC'), env.get('CXX'), env.get('FC')]:
        if value:
            pins[str(Path(value).resolve())] = sha(Path(value).resolve())
    # Mechanical command correction only: remove GNU timeout's extra grace;
    # the exact reviewed supervisor owns the inclusive 900-second deadline.
    old = protocol['build_command']
    assert old[:7] == ['taskset', '-c', '11', 'timeout', '--signal=TERM', '--kill-after=10s', '900s']
    assert old[7] == 'cargo' and '--no-run' in old and '--locked' in old
    command = [taskset, '-c', '11', cargo] + old[8:]
    spec = importlib.util.spec_from_file_location('reviewed_adaptive_build_supervisor', RUNNER / 'supervise.py')
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    OUT.mkdir(parents=True, exist_ok=False)
    record = {'schema': 1, 'source_index_sha256': SOURCE_INDEX,
              'supervisor_source_index_sha256': RUNNER_INDEX,
              'command': command, 'cwd': str(ROOT), 'environment': env,
              'compile_seconds': 900, 'cpu': 11, 'jobs': 1,
              'native_tests_authorized': False, 'native_tests_executed': False,
              'automatic_retry_count': 0, 'complete': False,
              'protected_inputs_before': pins, 'started_unix': time.time(),
              'process': {}}
    try:
        record['tool_versions'] = {}
        for label, tool in [('cargo', cargo), ('rustc', rustc)]:
            progress = {}
            with (OUT / f'{label}-version.log').open('w') as stream:
                code = supervisor.run_plain_capped([taskset, '-c', '11', tool, '--version'], stream,
                    cwd=str(ROOT), env=env, progress=progress, seconds=30)
            record['tool_versions'][label] = progress
            assert code == 0, (label, code)
        assert all(sha(Path(name)) == digest for name, digest in pins.items())
        budget_path = ROOT / '.tirust-build/adaptive-qp-rescue-build/further-build-budget.json'
        if budget_path.exists():
            budget = json.loads(budget_path.read_text())
        else:
            budget = {'started_monotonic': time.monotonic(), 'seconds': 900, 'reason': 'Root-authorized further compile-only budget; no automatic native tests.'}
            budget_path.write_text(json.dumps(budget, indent=2, sort_keys=True) + '\n')
        remaining = 900 - (time.monotonic() - budget['started_monotonic'])
        assert remaining > 0, 'Further build budget exhausted.'
        record['shared_further_build_budget'] = budget
        record['compile_seconds_this_attempt'] = remaining
        with (OUT / 'build.log').open('w') as stream:
            code = supervisor.run_plain_capped(command, stream, cwd=str(ROOT), env=env,
                progress=record['process'], seconds=remaining)
        record['returncode'] = code
        assert code == 0, f'Cargo returned {code}; no retry permitted.'
        artifacts = []
        for line in (OUT / 'build.log').read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get('reason') == 'compiler-artifact' and item.get('executable'):
                path = Path(item['executable'])
                artifacts.append({'cargo_record': item, 'sha256': sha(path), 'bytes': path.stat().st_size})
        assert len(artifacts) == 1
        assert artifacts[0]['cargo_record']['target']['name'] == 'madloop_tirust_adaptive_private'
        assert artifacts[0]['cargo_record']['profile']['test']
        record['test_executable_not_run'] = artifacts[0]
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
        record['artifact_sha256'] = {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}
        (OUT / 'execution.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    assert record['complete']
    print(json.dumps({'complete': True, 'native_tests_executed': False,
                     'execution': str(OUT / 'execution.json')}, sort_keys=True))


if __name__ == '__main__':
    main()
