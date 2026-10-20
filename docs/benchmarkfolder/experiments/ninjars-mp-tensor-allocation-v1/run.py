#!/usr/bin/env python3
"""Run a sealed isolated test binary, retaining all output and failed attempts."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.benchmark-work/experiments/ninjars-mp-tensor-allocation-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {str(p.relative_to(WORK / 'bridge')): sha(p) for p in sorted((WORK / 'bridge').rglob('*')) if p.is_file() and 'target' not in p.parts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['unit', 'capture-all', 'golden'])
    parser.add_argument('--build', default='build-attempt-3.jsonl')
    parser.add_argument('--name', required=True)
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    records = [json.loads(line) for line in (HERE / args.build).read_text().splitlines()]
    assert records[-1] == {'reason': 'build-finished', 'success': True}
    candidates = [r['executable'] for r in records if r.get('reason') == 'compiler-artifact' and r.get('executable') and r['profile']['test']]
    assert len(candidates) == 1
    executable = Path(candidates[0])
    assert executable.is_relative_to(WORK)
    command = ['taskset', '-c', '212', str(executable), '--test-threads=1', '--nocapture']
    env = os.environ.copy()
    declared_env = {'RUST_BACKTRACE': '1'}
    if args.mode == 'capture-all':
        command += ['--ignored', 'tensor::capture_tests::actual_madloop_requests_match_dense_expansions_exactly', '--exact']
        declared_env.update(NINJARS_TENSOR_REPLAY=str(WORK / 'actual-mp-requests.jsonl'), NINJARS_REPLAY_ALL='1')
    elif args.mode == 'golden':
        command += ['--ignored', 'tensor::capture_tests::all_golden_mp_inputs_match_original_numerator_state', '--exact']
    env.update(declared_env)
    before = source_hashes()
    provenance = json.loads((HERE / 'provenance.json').read_text())
    production_before = {name: sha(ROOT / 'vendor/ninjars' / name) for name in provenance['original_files']}
    assert production_before == provenance['original_files']
    assert sha(WORK / 'bridge/src/tensor_dense_reference.rs') == provenance['original_files']['src/tensor.rs']
    assert sha(WORK / 'actual-mp-requests.jsonl') == provenance['capture']['uncompressed_sha256']
    installed = ROOT / 'HEPTools/ninjars/lib/libninjars.a'
    installed_before = sha(installed)
    destination = HERE / args.name
    destination.mkdir(exist_ok=False)
    record = {'mode': args.mode, 'command': command, 'environment_overrides': declared_env,
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'executable_sha256': sha(executable), 'build_record': args.build,
              'build_record_sha256': sha(HERE / args.build), 'source_before': before,
              'installed_archive_sha256': installed_before, 'acceptance_evidence': False}
    (destination / 'started.json').write_text(json.dumps(record, indent=2) + '\n')
    start = time.monotonic()
    with (destination / 'stdout.log').open('xb') as stdout, (destination / 'stderr.log').open('xb') as stderr:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
    record.update(returncode=completed.returncode, elapsed_seconds=time.monotonic()-start,
                  source_unchanged=before == source_hashes(),
                  production_unchanged=all(sha(ROOT/'vendor/ninjars'/name) == digest for name,digest in production_before.items()),
                  installed_archive_unchanged=sha(installed) == installed_before,
                  executable_unchanged=sha(executable) == record['executable_sha256'],
                  stdout_sha256=sha(destination/'stdout.log'), stderr_sha256=sha(destination/'stderr.log'))
    (destination / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    assert all(record[key] for key in ['source_unchanged', 'production_unchanged', 'installed_archive_unchanged', 'executable_unchanged'])
    print(json.dumps({'mode': args.mode, 'returncode': completed.returncode, 'elapsed_seconds': record['elapsed_seconds'], 'result': str(destination/'result.json')}))
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
