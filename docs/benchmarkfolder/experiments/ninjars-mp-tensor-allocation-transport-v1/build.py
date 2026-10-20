#!/usr/bin/env python3
"""Private serial compilation only; requires an explicit coordination reason."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from prepare import HERE, ROOT, WORK, binding, sha, write


def sources():
    base = WORK / 'bridge'
    return {str(path.relative_to(base)): sha(path) for path in sorted(base.rglob('*')) if path.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorized-build', required=True)
    args = parser.parse_args()
    assert Path.cwd() == ROOT and args.authorized_build.strip()
    assert Path(os.environ['TMPDIR']).resolve().is_relative_to(ROOT)
    os.sched_setaffinity(0, {234})
    provenance = json.loads((HERE / 'provenance.json').read_text())
    before = sources()
    assert before == provenance['combined_files']
    prior_cache = ROOT / '.benchmark-work/experiments/ninjars-mp-tensor-allocation-v1/build'
    private_cache = WORK / 'build'
    if not private_cache.exists():
        # Copy-on-write when supported; ordinary independent copies otherwise.
        subprocess.run(['cp', '-a', '--reflink=auto', str(prior_cache), str(private_cache)], check=True)
    attempts = HERE / 'builds'
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f'attempt-{len(list(attempts.iterdir())) + 1}'
    attempt.mkdir()
    shutil.copytree(WORK / 'bridge', attempt / 'source')
    env = dict(os.environ, CARGO_TARGET_DIR=str(WORK / 'target'),
        CARGO_BUILD_BUILD_DIR=str(private_cache), CARGO_BUILD_JOBS='1',
        TMP=os.environ['TMPDIR'], TEMP=os.environ['TMPDIR'], TEMPDIR=os.environ['TMPDIR'])
    command = ['cargo', 'test', '--manifest-path', str(WORK / 'bridge/Cargo.toml'),
        '--locked', '--offline', '--release', '--lib', '--features', 'fenv-test-support',
        '--no-run', '--message-format=json']
    record = {'schema': 'corrected-transport-allocation-build-v1',
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'coordination': args.authorized_build, 'command': command, 'affinity': sorted(os.sched_getaffinity(0)),
        'environment': {name: env[name] for name in ['CARGO_TARGET_DIR', 'CARGO_BUILD_BUILD_DIR', 'CARGO_BUILD_JOBS', 'TMPDIR']},
        'cache_source': str(prior_cache.relative_to(ROOT)), 'cache_policy': 'independent copy/reflink; never modify source cache',
        'sources': before, 'provenance': binding(HERE / 'provenance.json'),
        'controller': binding(Path(__file__).resolve()),
        'rustc': subprocess.check_output(['rustc', '-vV'], text=True),
        'cargo': subprocess.check_output(['cargo', '-V'], text=True),
        'installed_before': binding(ROOT / 'HEPTools/ninjars/lib/libninjars.a'),
        'production_changed': False, 'numerical_execution': False}
    write(attempt / 'started.json', record)
    start = time.monotonic()
    try:
        with (attempt / 'cargo.jsonl').open('x') as stdout, (attempt / 'cargo.stderr').open('x') as stderr:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        record['exit_code'] = completed.returncode
        assert sources() == before
        assert completed.returncode == 0
        rows = [json.loads(line) for line in (attempt / 'cargo.jsonl').read_text().splitlines()]
        assert rows[-1] == {'reason': 'build-finished', 'success': True}
        binaries = [Path(row['executable']) for row in rows if row.get('reason') == 'compiler-artifact'
                    and row.get('executable') and row['profile']['test']]
        assert len(binaries) == 1 and binaries[0].is_relative_to(WORK)
        record['executable'] = binding(binaries[0])
        record['status'] = 'compiled-not-executed'
    except BaseException as exc:
        record['status'] = 'failed'
        record['failure'] = repr(exc)
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - start
        record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        record['installed_after'] = binding(ROOT / 'HEPTools/ninjars/lib/libninjars.a')
        record['installed_archive_changed_by_other_owner'] = record['installed_after'] != record['installed_before']
        write(attempt / 'result.json', record)
    print(json.dumps({'status': record['status'], 'result': str((attempt / 'result.json').relative_to(ROOT))}))


if __name__ == '__main__':
    main()
