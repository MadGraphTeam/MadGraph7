#!/usr/bin/env python3
"""Build isolated binaries on CPU212. Never installs or runs either binary."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import runpy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.benchmark-work/experiments/ninjars-mp-tensor-allocation-performance-v1'
CRATE = WORK / 'crate'
CACHE = ROOT / '.benchmark-work/experiments/ninjars-mp-tensor-allocation-v1/build'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    assert Path.cwd() == ROOT
    assert len(sys.argv) == 2 and sys.argv[1] == '--authorized-build'
    assert Path(os.environ['TMPDIR']).resolve().is_relative_to(ROOT)
    frozen_inputs = runpy.run_path(str(HERE / 'prepare.py'))['verified_frozen_inputs']()
    os.sched_setaffinity(0, {212})
    attempts = HERE / 'builds'
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f'attempt-{len(list(attempts.iterdir())) + 1}'
    attempt.mkdir()
    source = attempt / 'source'
    source.mkdir()
    for name in ('common.rs', 'kernel.rs', 'capture.rs'):
        shutil.copyfile(HERE / name, CRATE / 'src' / name)
    inputs = {str(p.relative_to(CRATE)): sha(p) for p in sorted(CRATE.rglob('*')) if p.is_file()}
    for relative in inputs:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CRATE / relative, target)
    plan = json.loads((HERE / 'plan.json').read_text())
    assert inputs['src/original.rs'] == plan['original_tensor_sha256']
    assert inputs['src/allocation.rs'] == plan['prototype_tensor_sha256']
    command = ['cargo', 'build', '--manifest-path', str(CRATE / 'Cargo.toml'),
               '--locked', '--offline', '--release', '--bins', '--message-format=json-render-diagnostics', '-j1']
    env = dict(os.environ, CARGO_TARGET_DIR=str(WORK / 'target'), CARGO_BUILD_BUILD_DIR=str(WORK / 'build'),
               CARGO_BUILD_JOBS='1')
    record = {'schema': 'pure-tensor-build-v1', 'started_utc': now(), 'status': 'running',
              'command': command, 'cwd': str(ROOT), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'plan_identity': plan['identity'], 'sources': inputs, 'verified_frozen_inputs': frozen_inputs,
              'environment': {k: env.get(k) for k in ('TMPDIR', 'RUSTFLAGS', 'CARGO_ENCODED_RUSTFLAGS',
                  'CARGO_TARGET_DIR', 'CARGO_BUILD_BUILD_DIR', 'CARGO_BUILD_JOBS', 'CC', 'CXX', 'FC')},
              'rustc': subprocess.check_output(['rustc', '--version', '--verbose'], text=True),
              'cargo': subprocess.check_output(['cargo', '--version'], text=True),
              'production_changes': False, 'numerical_execution': False}
    write(attempt / 'started.json', record)
    try:
        if not (WORK / 'build').exists():
            (WORK / 'build').mkdir()
            subprocess.run(['cp', '-a', '--reflink=auto', str(CACHE) + '/.', str(WORK / 'build')], check=True)
        with (attempt / 'cargo.jsonl').open('x') as stdout, (attempt / 'cargo.stderr').open('x') as stderr:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr).returncode
        record['exit_code'] = code
        assert inputs == {str(p.relative_to(CRATE)): sha(p) for p in sorted(CRATE.rglob('*')) if p.is_file()}
        if code:
            raise RuntimeError(f'cargo failed with exit {code}')
        executables = {}
        for line in (attempt / 'cargo.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row.get('reason') == 'compiler-artifact' and row.get('executable'):
                executable = Path(row['executable'])
                executables[row['target']['name']] = {'path': str(executable.relative_to(ROOT)),
                    'sha256': sha(executable), 'bytes': executable.stat().st_size}
        assert set(executables) == {'tensor-capture', 'tensor-kernel'}
        record['executables'] = executables
        symbols = subprocess.check_output(['nm', '-C', str(ROOT / executables['tensor-kernel']['path'])], text=True)
        (attempt / 'kernel-symbols.txt').write_text(symbols)
        # Keep exact matching lines as evidence; matches are reviewed, not hidden.
        patterns = ['SymbolicaAmplitude', 'SymbolicaMasters', 'SymbolicaCuts', 'SymbolicaHighCuts',
                    'evaluate_batch_with_inputs', 'license', 'License', 'oneloopmaster', 'amplitude::', 'geometry::']
        record['kernel_symbol_matches'] = {pattern: [line for line in symbols.splitlines() if pattern in line]
                                            for pattern in patterns}
        record['kernel_symbols_sha256'] = sha(attempt / 'kernel-symbols.txt')
        dynamic = subprocess.check_output(['readelf', '-d', str(ROOT / executables['tensor-kernel']['path'])], text=True)
        (attempt / 'kernel-dynamic.txt').write_text(dynamic)
        record['kernel_dynamic_sha256'] = sha(attempt / 'kernel-dynamic.txt')
        record['status'] = 'compiled-not-executed'
    except BaseException as exc:
        record['status'] = 'failed'
        record['failure'] = repr(exc)
        raise
    finally:
        record['finished_utc'] = now()
        record['all_compiler_children_exited'] = True
        write(attempt / 'result.json', record)
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
