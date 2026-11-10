#!/usr/bin/env python3
"""Prepared optimized private build; execution requires root authorization."""
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
SOURCE = ROOT / '.tirust-build/adaptive-qp-rescue-prototype3'
RUNNER = ROOT / 'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2'
SOURCE_INDEX = 'f5550b8719e76e177ad6b73d456cc655815770eb7092a8fabdc776c415c03213'
RUNNER_INDEX = '69f76225a439dd325c4c08369a46ae5654102f1a95c19d6c3e7ca83682f0a3f4'
OUT = ROOT / '.tirust-build/adaptive-qp-rescue-release-build/attempt1'


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
    previous = ROOT / 'docs/benchmarkfolder/experiments/adaptive-qp-rescue-private-build'
    pins.update(inventory(previous, 'evidence-index.json', '1a4af7f8f8be9ce8880ebc9ce466f66afec961be4c24b6978cdd8fdfa28fc8d8'))
    prior = json.loads((previous / 'attempt-3/execution.json').read_text())
    assert prior['complete'] and not prior['native_tests_executed']
    artifact = prior['test_executable_not_run']
    assert artifact['sha256'] == 'd22953ed3b955adf7dcce3831259060cec9e974816d432a4a86fd000b1afe8b7'
    test_path = Path(artifact['cargo_record']['executable'])
    assert sha(test_path) == artifact['sha256']
    pins[str(test_path)] = artifact['sha256']
    debug_package = ROOT / 'docs/benchmarkfolder/experiments/adaptive-qp-rescue-private-cdylib-build'
    pins.update(inventory(debug_package, 'evidence-index.json', 'd3e1442560aa572c6a3dd2964a9685ee423e58fbc275e0027e9b2c0f2693f03b'))
    debug_record = json.loads((debug_package / 'attempt1/execution.json').read_text())
    debug_library = debug_record['private_library_not_loaded']
    assert debug_library['sha256'] == '63298bdd242b99dd46d654debe6304f5b4fdff955d5082d8c50c3b2ff01ae9ea'
    for name in (debug_library['path'], debug_library['isolated_copy']):
        assert sha(Path(name)) == debug_library['sha256']
        pins[name] = debug_library['sha256']
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    assert protocol['cpu'] == 11 and protocol['compile_seconds'] == 900
    assert protocol['automatic_retry_count'] == 0 and not protocol['native_authorized']
    env_script = ROOT / protocol['environment_script']
    pins[str(env_script)] = sha(env_script)
    # Exact positive environment already recorded by the successful test build.
    env = prior['environment'].copy()
    assert env['CARGO_BUILD_JOBS'] == '1'
    assert not any(name in env for name in ('RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS', 'RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER'))
    cargo = shutil.which('cargo', path=env['PATH'])
    rustc = shutil.which('rustc', path=env['PATH'])
    taskset = shutil.which('taskset', path=env['PATH'])
    assert cargo and rustc and taskset
    for value in [cargo, rustc, taskset, env.get('CC'), env.get('CXX'), env.get('FC')]:
        if value:
            pins[str(Path(value).resolve())] = sha(Path(value).resolve())
    command = [cargo, 'build', '--release', '--manifest-path',
               str(SOURCE.relative_to(ROOT) / 'Cargo.toml'), '--locked', '--offline',
               '--features', 'poles-only', '--lib', '--message-format=json']
    spec = importlib.util.spec_from_file_location('reviewed_adaptive_build_supervisor', RUNNER / 'supervise.py')
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    OUT.mkdir(parents=True, exist_ok=False)
    record = {'schema': 1, 'source_index_sha256': SOURCE_INDEX,
              'supervisor_source_index_sha256': RUNNER_INDEX,
              'command': command, 'cwd': str(ROOT), 'environment': env,
              'compile_seconds': 900, 'cpu_affinity_override': None, 'jobs': 1,
              'native_tests_authorized': False, 'native_tests_executed': False,
              'automatic_retry_count': 0, 'complete': False,
              'protected_inputs_before': pins, 'started_unix': time.time(),
              'process': {}}
    budget_start = time.monotonic()
    try:
        record['tool_versions'] = {}
        for label, tool in [('cargo', cargo), ('rustc', rustc)]:
            progress = {}
            with (OUT / f'{label}-version.log').open('w') as stream:
                code = supervisor.run_plain_capped([tool, '--version'], stream,
                    cwd=str(ROOT), env=env, progress=progress, seconds=min(30, max(0, 900 - (time.monotonic() - budget_start))))
            record['tool_versions'][label] = progress
            assert code == 0, (label, code)
        assert all(sha(Path(name)) == digest for name, digest in pins.items())
        remaining = 900 - (time.monotonic() - budget_start)
        assert remaining > 0, 'The single release-build budget is exhausted.'
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
            if item.get('reason') == 'compiler-artifact' and item['target']['name'] == 'madloop_tirust_adaptive_private':
                assert not item['profile']['test'] and item['executable'] is None
                for name in item['filenames']:
                    path = Path(name)
                    artifacts.append({'path': name, 'sha256': sha(path), 'bytes': path.stat().st_size,
                                      'cargo_record': item})
        libraries = [a for a in artifacts if a['path'].endswith('.so')]
        assert len(libraries) == 1
        runtime = OUT / 'runtime'
        runtime.mkdir()
        copied = runtime / 'libmadloop_tirust_adaptive_private.so'
        shutil.copyfile(libraries[0]['path'], copied)
        assert sha(copied) == libraries[0]['sha256']
        assert libraries[0]['cargo_record']['profile']['opt_level'] == '3'
        record['private_library_not_loaded'] = {**libraries[0], 'isolated_copy': str(copied)}
        record['cargo_artifacts'] = artifacts
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
        record['artifact_sha256'] = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file()}
        (OUT / 'execution.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    assert record['complete']
    print(json.dumps({'complete': True, 'native_tests_executed': False,
                     'execution': str(OUT / 'execution.json')}, sort_keys=True))


if __name__ == '__main__':
    main()
