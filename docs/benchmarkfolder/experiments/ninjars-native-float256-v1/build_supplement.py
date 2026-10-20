#!/usr/bin/env python3
"""Separate test-only relink; preserves every original source/artifact binding."""
import sys
sys.dont_write_bytecode = True
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.benchmark-work/experiments/ninjars-native-float256-v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert Path.cwd() == ROOT
    prepared = json.loads((HERE/'prepared.json').read_text())
    original = json.loads((HERE/'build.json').read_text())
    original_exe = ROOT/original['executables'][0]['path']
    assert sha(original_exe) == original['executables'][0]['sha256']
    for name, digest in prepared['study_files'].items():
        assert sha(WORK/'bridge'/name) == digest, name
    assert json.loads((HERE/'run-convergence/result.json').read_text())['complete']
    destination = HERE/'supplement'
    destination.mkdir(exist_ok=False)
    bridge = WORK/'supplement/bridge'
    bridge.parent.mkdir(exist_ok=False)
    shutil.copytree(WORK/'bridge', bridge)
    helper = HERE/'native_supplement_tests.rs'
    shutil.copy2(helper, bridge/'src/native_supplement_tests.rs')
    with (bridge/'src/native_study_tests.rs').open('a') as stream:
        stream.write('\n#[path = "native_supplement_tests.rs"]\nmod supplement;\n')
    source_hashes = {str(path.relative_to(bridge)): sha(path) for path in sorted(bridge.rglob('*'))
                     if path.is_file() and '__pycache__' not in path.parts}
    for name in ('Cargo.toml', 'src/lib.rs', 'src/native_study_tests.rs', 'src/native_supplement_tests.rs'):
        target = destination/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(bridge/name, target)
    env = dict(os.environ)
    env.update(CARGO_TARGET_DIR=str(WORK/'target'), CARGO_BUILD_BUILD_DIR=str(WORK/'build'),
               CARGO_BUILD_JOBS='1', TMPDIR=str(ROOT/'.ninjars-build/tmp'))
    command = ['taskset','-c','190','cargo','test','--manifest-path',str(bridge/'Cargo.toml'),
               '--locked','--offline','--release','--lib','--features','fenv-test-support',
               '--no-run','--message-format=json']
    result = {'complete': False, 'executed': False, 'production_modified': False,
        'original_build_sha256': sha(HERE/'build.json'), 'original_executable_sha256': sha(original_exe),
        'screening_plan_sha256': sha(HERE/'SCREENING-PLAN.md'), 'builder_sha256': sha(Path(__file__)),
        'supplement_helper_sha256': sha(helper), 'source_hashes': source_hashes, 'command': command,
        'cache_policy': 'Existing study-only build cache; different local package path/artifact disambiguator; original source and binary must remain exact',
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'performance_claim': False}
    start = time.monotonic()
    try:
        with (destination/'build.stdout').open('x') as out, (destination/'build.stderr').open('x') as err:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
            result['pid'] = process.pid
            (destination/'build-launch.json').write_text(json.dumps(result,indent=2)+'\n')
            try:
                result['returncode'] = process.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        assert result['returncode'] == 0
        rows = [json.loads(line) for line in (destination/'build.stdout').read_text().splitlines()]
        assert [r for r in rows if r.get('reason') == 'build-finished'] == [{'reason':'build-finished','success':True}]
        artifacts = [{'path':str(Path(r['executable']).relative_to(ROOT)), 'sha256':sha(Path(r['executable']))}
                     for r in rows if r.get('reason') == 'compiler-artifact' and r.get('executable')]
        assert len(artifacts) == 1 and ROOT/artifacts[0]['path'] != original_exe
        assert sha(original_exe) == result['original_executable_sha256']
        for name,digest in prepared['study_files'].items(): assert sha(WORK/'bridge'/name) == digest,name
        for name,digest in source_hashes.items(): assert sha(bridge/name) == digest,name
        result['executables'] = artifacts
        result['complete'] = True
    except BaseException as error:
        result['failure'] = repr(error)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-start
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result['logs'] = {name:sha(destination/name) for name in ('build.stdout','build.stderr')}
        with (destination/'build-result.json').open('x') as stream:
            json.dump(result,stream,indent=2)
            stream.write('\n')


if __name__ == '__main__': main()
