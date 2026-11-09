#!/usr/bin/env python3
"""Build a private historical bridge at 512 working bits; no native evaluation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[5]
WORK = ROOT / '.tirust-build/ew-threshold-512'
HISTORICAL = ROOT / '.tirust-build/bridge-poles'
GUARD = REPORT.parent / 'guard-audit'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def name(path):
    path = path.resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def main():
    if Path.cwd().resolve() != ROOT:
        raise SystemExit('Run from the MadGraph repository root')
    if WORK.exists() or (REPORT / 'build.json').exists() or (REPORT / 'build.log').exists():
        raise SystemExit('Refusing to overwrite a prior private build or evidence')
    audit = json.loads((GUARD / 'audit.json').read_text())
    recipe = ROOT / 'docs/benchmarkfolder/experiments/n1-profile/bridge/build-metadata.json'
    old = json.loads(recipe.read_text())
    original = HISTORICAL / 'src/lib.rs'
    assert digest(original) == audit['diagnostic_policy']['source_before_sha256']
    source = original.read_text()
    assert source.count('const MP_BITS: u32 = 256;') == 1
    assert source.count('Float::with_val(MP_BITS, 2).pow(136).inv()') == 2
    source = source.replace('const MP_BITS: u32 = 256;', 'const MP_BITS: u32 = 512;')
    source = source.replace('Float::with_val(MP_BITS, 2).pow(136).inv()',
        'Float::with_val(MP_BITS, 2).pow(u64::from(MP_BITS - 120)).inv()')
    source = source.replace('// 256-bit arithmetic with substantial headroom above binary128.',
        '// 512-bit diagnostic arithmetic; input binary128 semantics unchanged.')
    assert hashlib.sha256(source.encode()).hexdigest() == audit['diagnostic_policy']['source_after_sha256']
    WORK.mkdir(parents=True)
    (WORK / 'src').mkdir()
    (WORK / 'build').mkdir()
    (WORK / 'runtime').mkdir()
    (REPORT / 'source').mkdir()
    for filename in ('Cargo.toml', 'Cargo.lock'):
        shutil.copy2(HISTORICAL / filename, REPORT / 'source' / filename)
    shutil.copy2(original, REPORT / 'source/lib.rs.original')
    (WORK / 'src/lib.rs').write_text(source)
    (REPORT / 'source/lib.rs').write_text(source)
    shutil.copy2(GUARD / 'diagnostic-512-source.patch', REPORT / 'source/diagnostic-512-source.patch')
    command = list(old['command'])
    command[0] = shutil.which('rustc')
    assert command[0], 'Source tools/tirust_env.sh first'
    command[command.index('--crate-name') + 1] = 'madloop_tirust_ew512'
    source_index = next(i for i, value in enumerate(command) if value.endswith('/src/lib.rs'))
    command[source_index] = str(WORK / 'src/lib.rs')
    command[command.index('--out-dir') + 1] = str(WORK / 'build')
    command = ['taskset', '-c', '6-8', *command]
    deps = ROOT / '.tirust-build/cargo-build/release/deps'
    native = ROOT / '.tirust-build/cargo-build/release/build/gmp-mpfr-sys-e3a3f2f14732793a/out/lib'
    # The isolated compiler reads the same prebuilt transitive artifacts. It
    # receives no Cargo target directory and cannot publish an installation.
    protected_paths = [p for p in deps.iterdir() if p.is_file() and p.suffix in ('.rlib', '.rmeta', '.so')]
    protected_paths += list(native.glob('*.a'))
    protected_paths += [original, HISTORICAL / 'Cargo.toml', HISTORICAL / 'Cargo.lock', recipe,
        GUARD / 'audit.json', GUARD / 'diagnostic-512-source.patch',
        Path(command[3]), Path(__file__), WORK / 'src/lib.rs',
        REPORT.parent / 'execution.json', REPORT.parent / 'summary.json',
        REPORT.parent / 'selected-inputs.json', REPORT.parent / 'selected-inputs.dat']
    for rel, expected in old['input_sha256'].items():
        if rel.endswith('.rlib'):
            assert digest(ROOT / rel) == expected, rel
    protected = {name(p): digest(p) for p in protected_paths}
    result = {'schema': 1, 'scope': 'Private diagnostic only; no production installation or policy change.',
        'input_bits': 113, 'working_bits': 512, 'gate': '2^-392',
        'lightlike_gate': '2^-100 (unchanged)', 'cache_policy': 'Fresh private namespace; disable result caching in replay.',
        'other_gate_consumers': audit['interpretation']['other_consumers'],
        'gate_accuracy_limit': audit['interpretation']['roundoff_model'],
        'command': command, 'compile_cpu_affinity': '6-8', 'timeout_seconds': 1800,
        'controlled_environment': {k: os.environ.get(k) for k in ('RUST_MIN_STACK', 'CC', 'CXX')},
        'protected_inputs_before': protected, 'source_before_sha256': digest(original),
        'source_after_sha256': digest(WORK / 'src/lib.rs'), 'complete': False,
        'native_evaluation_executed': False, 'performance_eligible': False,
        'concurrent_load': 'Installer compilation CPUs2-5 and unrelated diagnostic measurements CPU9 may overlap.'}
    write(REPORT / 'preparation.json', result)
    process = None
    started = time.monotonic()
    try:
        result['rustc_version'] = subprocess.check_output([command[3], '--version', '--verbose'], text=True, timeout=10)
        with (REPORT / 'build.log').open('x') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
            result['compiler_pid'] = process.pid
            result['observed_affinity'] = sorted(os.sched_getaffinity(process.pid))
            # taskset may not yet have exec'd; the command is authoritative,
            # and the compiler's affinity is sampled again after launch.
            time.sleep(.1)
            if process.poll() is None:
                result['compiler_affinity_after_launch'] = sorted(os.sched_getaffinity(process.pid))
                assert set(result['compiler_affinity_after_launch']) <= {6, 7, 8}
            process.wait(timeout=1800)
            result['returncode'] = process.returncode
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        library = WORK / 'build/libmadloop_tirust_ew512.so'
        assert library.is_file()
        shutil.copy2(library, WORK / 'runtime/libmadloop_tirust.so')
        nm = subprocess.check_output(['nm', '-D', '--defined-only', str(library)], text=True, timeout=30)
        (REPORT / 'exports.txt').write_text(nm)
        exports = [row.split()[-1] for row in nm.splitlines() if row.split()[-1].startswith('ml_tirust_')]
        historical_runtime = ROOT / '.tirust-build/ew-threshold-tensor-replay-runtime/libmadloop_tirust.so'
        historical_nm = subprocess.check_output(['nm', '-D', '--defined-only', str(historical_runtime)], text=True, timeout=30)
        historical_exports = [row.split()[-1] for row in historical_nm.splitlines() if row.split()[-1].startswith('ml_tirust_')]
        assert exports == historical_exports and len(exports) == 12
        (REPORT / 'dynamic.txt').write_text(subprocess.check_output(['readelf', '-d', str(library)], text=True, timeout=30))
        result['library_sha256'] = digest(library)
        result['runtime_library'] = name(WORK / 'runtime/libmadloop_tirust.so')
        result['abi_exports'] = exports
        result['historical_dso_sha256'] = digest(historical_runtime)
        result['complete'] = True
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        result['elapsed_seconds_not_a_benchmark'] = time.monotonic() - started
        result['protected_inputs_after'] = {name(p): digest(p) for p in protected_paths}
        result['protected_inputs_unchanged'] = protected == result['protected_inputs_after']
        result['build_log_sha256'] = digest(REPORT / 'build.log') if (REPORT / 'build.log').exists() else None
        write(REPORT / 'build.json', result)
        if not result['protected_inputs_unchanged']:
            raise RuntimeError('A protected diagnostic dependency changed during compilation')


if __name__ == '__main__':
    main()
