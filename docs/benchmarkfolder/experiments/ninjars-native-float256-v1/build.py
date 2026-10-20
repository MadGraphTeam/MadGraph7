#!/usr/bin/env python3
"""One authorized serial release-test compilation, CPU190; never executes tests."""
import sys
sys.dont_write_bytecode = True
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT/'.benchmark-work/experiments/ninjars-native-float256-v1'
CACHE = ROOT/'.benchmark-work/experiments/ninjars-mp-transport-audit-v1/build/release'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert Path.cwd() == ROOT
    assert not (HERE/'build.json').exists()
    prepared = json.loads((HERE/'prepared.json').read_text())
    for name, digest in prepared['study_files'].items():
        assert sha(WORK/'bridge'/name) == digest, name
    build = WORK/'build'
    build.mkdir(exist_ok=False)
    copy = ['taskset', '-c', '190', 'cp', '-a', '--reflink=auto', str(CACHE), str(build/'release')]
    subprocess.run(copy, cwd=ROOT, check=True)
    env = dict(os.environ)
    env.update(CARGO_TARGET_DIR=str(WORK/'target'), CARGO_BUILD_BUILD_DIR=str(build),
               CARGO_BUILD_JOBS='1', TMPDIR=str(ROOT/'.ninjars-build/tmp'))
    command = ['taskset', '-c', '190', 'cargo', 'test', '--manifest-path', str(WORK/'bridge/Cargo.toml'),
               '--locked', '--offline', '--release', '--lib', '--features', 'fenv-test-support',
               '--no-run', '--message-format=json']
    result = {'complete': False, 'executed': False, 'production_modified': False,
              'authorised_scope': 'One serial release-test compile only; MP/test execution held',
              'prepared_sha256': sha(HERE/'prepared.json'), 'builder_sha256': sha(Path(__file__)),
              'copy_command': copy, 'cache_policy': 'Private reflinks/copies; no hard links or shared-cache mutation',
              'command': command, 'environment': {k: env[k] for k in
                  ('CARGO_TARGET_DIR', 'CARGO_BUILD_BUILD_DIR', 'CARGO_BUILD_JOBS', 'TMPDIR')},
              'rustc': subprocess.check_output(['rustc', '-Vv'], text=True),
              'cargo': subprocess.check_output(['cargo', '--version'], text=True),
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    start = time.monotonic()
    try:
        with (HERE/'build-1.stdout').open('x') as out, (HERE/'build-1.stderr').open('x') as err:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
            result['pid'] = process.pid
            (HERE/'build-launch.json').write_text(json.dumps(result, indent=2)+'\n')
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
        for line in (HERE/'build-1.stdout').read_text().splitlines():
            row = json.loads(line)
            if row.get('reason') == 'compiler-artifact' and row.get('executable'):
                path = Path(row['executable'])
                artifacts.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
        assert len(artifacts) == 1, artifacts
        result['executables'] = artifacts
        for name, digest in prepared['study_files'].items():
            assert sha(WORK/'bridge'/name) == digest, name
        result['complete'] = True
    except BaseException as error:
        result['failure'] = repr(error)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-start
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result['logs'] = {n: sha(HERE/n) for n in ('build-1.stdout', 'build-1.stderr')}
        with (HERE/'build.json').open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
