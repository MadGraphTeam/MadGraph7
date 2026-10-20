#!/usr/bin/env python3
"""Compile the isolated transport-only crate; do not execute any tests."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.benchmark-work/experiments/ninjars-mp-transport-audit-v1'
CACHE = ROOT / '.benchmark-work/experiments/ninjars-mp-tensor-allocation-v1/build/release'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path.cwd() == ROOT
    prepared = json.loads((HERE / 'bridge-prepared.json').read_text())
    assert prepared['complete'] and not prepared['executed']
    for name, digest in prepared['proposed_files'].items():
        assert sha(WORK / 'bridge' / name) == digest, name
    build = WORK / 'build'
    build.mkdir(exist_ok=False)
    # Private byte copies/reflinks, never hard links to mutable cached artifacts.
    copy = ['taskset', '-c', '185', 'cp', '-a', '--reflink=auto', str(CACHE), str(build / 'release')]
    subprocess.run(copy, cwd=ROOT, check=True)
    env = dict(os.environ)
    env.update(CARGO_TARGET_DIR=str(WORK / 'target'), CARGO_BUILD_BUILD_DIR=str(build), CARGO_BUILD_JOBS='1')
    command = ['taskset', '-c', '185', 'cargo', 'test', '--manifest-path', str(WORK / 'bridge/Cargo.toml'),
               '--locked', '--offline', '--release', '--lib', '--features', 'fenv-test-support',
               '--no-run', '--message-format=json']
    result = {'complete': False, 'executed': False, 'production_modified': False,
              'preparation_sha256': sha(HERE / 'bridge-prepared.json'),
              'builder_sha256': sha(Path(__file__)), 'copy_command': copy, 'command': command,
              'cache_policy': 'Private copies/reflinks; no shared-cache mutation or hard links.',
              'environment': {key: env[key] for key in ('CARGO_TARGET_DIR', 'CARGO_BUILD_BUILD_DIR', 'CARGO_BUILD_JOBS')},
              'rustc': subprocess.check_output(['rustc', '-Vv'], text=True),
              'cargo': subprocess.check_output(['cargo', '--version'], text=True)}
    start = time.monotonic()
    try:
        with (HERE / 'bridge-build-1.stdout').open('x') as out, (HERE / 'bridge-build-1.stderr').open('x') as err:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
            result['pid'] = process.pid
            try:
                result['returncode'] = process.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        assert result['returncode'] == 0
        artifacts = []
        for line in (HERE / 'bridge-build-1.stdout').read_text().splitlines():
            record = json.loads(line)
            if record.get('reason') == 'compiler-artifact' and record.get('executable'):
                path = Path(record['executable'])
                artifacts.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        assert len(artifacts) == 1, artifacts
        result['executables'] = artifacts
        for name, digest in prepared['proposed_files'].items():
            assert sha(WORK / 'bridge' / name) == digest, name
        result['complete'] = True
    except BaseException as error:
        result['failure'] = repr(error)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - start
        result['logs'] = {name: sha(HERE / name) for name in ('bridge-build-1.stdout', 'bridge-build-1.stderr')}
        with (HERE / 'bridge-build.json').open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
