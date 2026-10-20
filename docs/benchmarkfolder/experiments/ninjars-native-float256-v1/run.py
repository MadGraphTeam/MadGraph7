#!/usr/bin/env python3
"""Held serial study runner. Invoke only after root's explicit MP handoff."""
import sys
sys.dont_write_bytecode = True
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT/'.benchmark-work/experiments/ninjars-native-float256-v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
TESTS = {'focused': 'native_study_tests::policies_rebind_fail_recover_and_converge',
         'replay': 'native_study_tests::all_requests_three_policies',
         'convergence': 'native_study_tests::discrepancies_at_256_and_384'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('unit', *TESTS))
    parser.add_argument('--cpu', type=int, required=True)
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    assert args.cpu not in (185, 207)
    prepared = json.loads((HERE/'prepared.json').read_text())
    build = json.loads((HERE/'build.json').read_text())
    assert build['complete'] and not build['executed']
    assert build['prepared_sha256'] == sha(HERE/'prepared.json')
    for name, digest in prepared['study_files'].items():
        assert sha(WORK/'bridge'/name) == digest, name
    executable = ROOT/build['executables'][0]['path']
    assert sha(executable) == build['executables'][0]['sha256']
    source = WORK/'actual-mp-requests.jsonl'
    assert sha(source) == prepared['capture']['uncompressed_sha256']
    if args.stage != 'unit':
        assert json.loads((HERE/'run-unit/result.json').read_text())['complete']
    if args.stage in ('replay', 'convergence'):
        assert json.loads((HERE/'run-focused/result.json').read_text())['complete']
    if args.stage == 'convergence':
        assert json.loads((HERE/'run-replay/result.json').read_text())['complete']
    destination = HERE/('run-'+args.stage)
    destination.mkdir(exist_ok=False)
    command = ['taskset', '-c', str(args.cpu), str(executable)]
    if args.stage != 'unit':
        command += ['--ignored', '--exact', TESTS[args.stage]]
    command += ['--test-threads=1', '--nocapture']
    env = dict(os.environ)
    lockdir = ROOT/'.ninjars-build/symbolica-locks'
    assert lockdir.is_dir()
    env.setdefault('SYMBOLICA_LOCK_DIR', str(lockdir))
    env.update(TMPDIR=str(ROOT/'.ninjars-build/tmp'),
               NINJARS_NATIVE_STUDY_INPUT=str(source),
               NINJARS_NATIVE_STUDY_OUTPUT=str(destination/'raw.jsonl'))
    if args.stage == 'convergence':
        env['NINJARS_NATIVE_STUDY_REPLAY'] = str(HERE/'run-replay/raw.jsonl')
    ldd = subprocess.check_output(['ldd', str(executable)], text=True)
    assert 'not found' not in ldd
    (destination/'ldd.txt').write_text(ldd)
    libraries = {}
    for name in re.findall(r'(?:=>\s+)?(/\S+)', ldd):
        path = Path(name).resolve(strict=True)
        libraries[name] = {'resolved': str(path), 'sha256': sha(path)}
    result = {'complete': False, 'stage': args.stage, 'production_modified': False,
              'command': command, 'cpu': args.cpu, 'timeout_seconds': 1800,
              'runner_sha256': sha(Path(__file__)), 'prepared_sha256': sha(HERE/'prepared.json'),
              'build_sha256': sha(HERE/'build.json'), 'input_sha256': sha(source),
              'executable_sha256': sha(executable), 'shared_libraries': libraries,
              'symbolica_lock_dir': env['SYMBOLICA_LOCK_DIR'],
              'performance_claim': False, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    start = time.monotonic()
    try:
        with (destination/'stdout').open('x') as out, (destination/'stderr').open('x') as err:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=out, stderr=err, start_new_session=True)
            result['pid'] = process.pid
            (destination/'launch.json').write_text(json.dumps(result, indent=2)+'\n')
            try:
                result['returncode'] = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        assert result['returncode'] == 0
        stdout = (destination/'stdout').read_text()
        assert ('18 passed; 0 failed;' if args.stage == 'unit' else '1 passed; 0 failed;') in stdout
        if args.stage != 'unit':
            rows = [json.loads(line) for line in (destination/'raw.jsonl').read_text().splitlines()]
            assert rows[-1]['kind'] == 'summary' and rows[-1]['complete'] is True
            result['summary'] = rows[-1]
        for name, digest in prepared['study_files'].items():
            assert sha(WORK/'bridge'/name) == digest, name
        assert sha(executable) == result['executable_sha256'] and sha(source) == result['input_sha256']
        for name, record in libraries.items():
            assert str(Path(name).resolve(strict=True)) == record['resolved']
            assert sha(Path(record['resolved'])) == record['sha256']
        result['complete'] = True
    except BaseException as error:
        result['failure'] = repr(error)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-start
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        result['files'] = {name: sha(destination/name) for name in ('stdout', 'stderr', 'raw.jsonl', 'ldd.txt')
                           if (destination/name).is_file()}
        with (destination/'result.json').open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
