#!/usr/bin/env python3
"""Compile only the exact final allocation source; never run a numerical engine."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())
PRIOR = HERE.parent / 'ninjars-mp-tensor-allocation-transport-v1'
WORK = ROOT / '.benchmark-work/experiments/ninjars-mp-tensor-allocation-process-v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

def binding(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'bytes': path.stat().st_size}

def sources(directory):
    return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}

def main():
    assert Path.cwd() == ROOT
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'madloop_ninjars'
    assert not WORK.exists() and not (HERE / 'build-result.json').exists()
    os.sched_setaffinity(0, {234})
    seal = json.loads((PRIOR / 'manifest.json').read_text())
    for name, item in seal['files'].items():
        p = PRIOR / name
        assert sha(p) == item['sha256'] and p.stat().st_size == item['bytes'], name
    before = sources(PRIOR / 'original')
    assert sha(PRIOR / 'production-tensor.rs') == '240bbc1f9c8937393227c16a6ce4887ee68524a3f021a8a93656151159445d70'
    WORK.mkdir(parents=True)
    bridge = WORK / 'bridge'
    shutil.copytree(PRIOR / 'original', bridge)
    shutil.copyfile(PRIOR / 'production-tensor.rs', bridge / 'src/tensor.rs')
    after = sources(bridge)
    assert [n for n in before if before[n] != after[n]] == ['src/tensor.rs']
    assert set(before) == set(after)
    retained = HERE / 'source'
    shutil.copytree(bridge, retained)
    # Independent reflink/copy; the correctness cache remains immutable.
    cache = ROOT / '.benchmark-work/experiments/ninjars-mp-tensor-allocation-transport-v1/build'
    subprocess.run(['cp', '-a', '--reflink=auto', str(cache), str(WORK / 'build')], check=True)
    baseline = ROOT / '.benchmark-work/policy-v2/install-export-abi3-zero-sign-v1/source/HEPTools/ninjars-unavailable/lib/libninjars.a'
    assert sha(baseline) == '832fc0d1198814b9cc80a65e47bc7aa5dc204f7d25701f206c910cd188fbe80b'
    env = dict(os.environ, CARGO_TARGET_DIR=str(WORK / 'target'),
        CARGO_BUILD_BUILD_DIR=str(WORK / 'build'), CARGO_BUILD_JOBS='1')
    command = ['cargo', 'build', '--manifest-path', str(bridge / 'Cargo.toml'),
        '--locked', '--offline', '--release', '--lib', '--message-format=json']
    result = {'schema': 'allocation-process-staticlib-build-v1', 'status': 'building',
        'coordination': 'root permits isolated CPU234 jobs1 build preparation; no numerical engine',
        'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'affinity': sorted(os.sched_getaffinity(0)), 'command': command,
        'environment': {n: env[n] for n in ['CARGO_TARGET_DIR', 'CARGO_BUILD_BUILD_DIR', 'CARGO_BUILD_JOBS', 'CARGO_HOME', 'TMPDIR']},
        'controller': binding(Path(__file__)), 'plan': binding(HERE / 'PLAN.md'),
        'correctness_manifest': binding(PRIOR / 'manifest.json'),
        'original_sources': before, 'allocation_sources': after,
        'baseline_archive': binding(baseline), 'cache_source': str(cache.relative_to(ROOT)),
        'rustc': subprocess.check_output(['rustc', '-vV'], text=True),
        'cargo': subprocess.check_output(['cargo', '-V'], text=True),
        'production_changed': False, 'numerical_execution': False, 'acceptance_eligible': False}
    write(HERE / 'build-started.json', result)
    start = time.monotonic()
    try:
        with (HERE / 'cargo.jsonl').open('x') as stdout, (HERE / 'cargo.stderr').open('x') as stderr:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            write(HERE / 'build-pids.json', {'controller': os.getpid(), 'cargo': child.pid})
            print(json.dumps({'build_started': True, 'controller': os.getpid(), 'cargo': child.pid}), flush=True)
            result['exit_code'] = child.wait()
        assert result['exit_code'] == 0
        assert sources(bridge) == after and sources(PRIOR / 'original') == before
        assert sha(baseline) == result['baseline_archive']['sha256']
        archive = WORK / 'target/release/libmadloop_ninjars.a'
        result['allocation_archive'] = binding(archive)
        result['status'] = 'compiled-not-executed'
    except BaseException as exc:
        result['status'] = 'failed'
        result['failure'] = repr(exc)
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic() - start
        result['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(HERE / 'build-result.json', result)
    print(json.dumps({'status': result['status'], 'archive': result['allocation_archive']}), flush=True)

if __name__ == '__main__':
    main()
