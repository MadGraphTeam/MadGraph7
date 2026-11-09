#!/usr/bin/env python3
"""Bounded private tensor replay. Build/run only in the assigned native slot."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
from prepare import ROOT, REPORT, WORK, RUNTIME, sources


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def name(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def terminate(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def verify():
    manifest = sources()
    selected = json.loads((REPORT / 'selected-inputs.json').read_text())
    checks = dict(manifest['sources_and_link_inputs'])
    checks[name(REPORT / 'source-manifest.json')] = selected['source_manifest_sha256']
    for item in ('trace', 'summary', 'execution'):
        checks[name(REPORT.parent / (item + '.json'))] = selected['parent_' + item + '_sha256']
    for path in (REPORT / 'selected-inputs.dat', WORK / 'selected-inputs.dat'):
        checks[name(path)] = selected['input_stream_sha256']
    checks[name(WORK / 'replay.f90')] = digest(REPORT / 'replay.f90')
    checks[name(REPORT / 'selected-inputs.json')] = digest(REPORT / 'selected-inputs.json')
    checks[name(Path(__file__))] = digest(Path(__file__))
    for path, expected in checks.items():
        if digest(ROOT / path) != expected:
            raise ValueError('Protected input changed: ' + path)
    return checks


def build():
    if (REPORT / 'compilation.json').exists() or (REPORT / 'build.log').exists():
        raise ValueError('Refusing to overwrite prior build evidence')
    protected = verify()
    fc = shutil.which('gfortran')
    if fc is None:
        raise ValueError('Load tools/tirust_env.sh before building')
    compile_command = [fc, '-O2', '-ffree-line-length-none', '-I' + str(RUNTIME),
        '-J' + str(WORK), '-c', str(WORK / 'replay.f90'), '-o', str(WORK / 'replay.o')]
    before = shlex.split((ROOT / 'HEPTools/ninjars/lib/ninjars-link-before.txt').read_text())
    native = shlex.split((ROOT / 'HEPTools/ninjars/lib/ninjars-native-libs.txt').read_text())
    link_command = [fc, '-o', str(WORK / 'replay'), str(WORK / 'replay.o'),
        str(RUNTIME / 'libtirust.a'), *before, str(ROOT / 'HEPTools/ninjars/lib/libninjars.a'),
        *native, str(RUNTIME / 'libmadloop_tirust.so'), '-Wl,-rpath,' + str(RUNTIME), '-lquadmath']
    result = {'compile_command': compile_command, 'link_command': link_command,
        'compiler': fc, 'compiler_sha256': digest(Path(fc)), 'timeout_seconds': 180,
        'run_script_sha256': digest(Path(__file__)), 'protected_inputs_before': protected,
        'complete': False, 'performance_eligible': False}
    deadline = time.monotonic() + 180
    try:
        result['compiler_version'] = subprocess.check_output([fc, '--version'], text=True, timeout=10)
        with (REPORT / 'build.log').open('x') as log:
            for command in (compile_command, link_command):
                completed = subprocess.run(command, cwd=WORK, stdout=log, stderr=subprocess.STDOUT,
                    timeout=max(.1, deadline - time.monotonic()), check=False)
                result['returncode'] = completed.returncode
                completed.check_returncode()
        result['artifacts_sha256'] = {name(WORK / p): digest(WORK / p)
            for p in ('replay.f90', 'replay.o', 'replay', 'selected-inputs.dat')}
        result['complete'] = True
    except BaseException as exc:
        result['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        result['protected_inputs_after'] = verify()
        result['protected_inputs_unchanged'] = result['protected_inputs_after'] == protected
        if (REPORT / 'build.log').exists():
            result['build_log_sha256'] = digest(REPORT / 'build.log')
        dump(REPORT / 'compilation.json', result)


def run(cpu):
    if (REPORT / 'execution.json').exists() or (REPORT / 'output.log').exists():
        raise ValueError('Refusing to overwrite prior native evidence')
    protected = verify()
    compiled = json.loads((REPORT / 'compilation.json').read_text())
    assert compiled['complete'] and compiled['protected_inputs_unchanged']
    assert compiled['run_script_sha256'] == digest(Path(__file__))
    for path, expected in compiled['artifacts_sha256'].items():
        assert digest(ROOT / path) == expected, path
    selected = json.loads((REPORT / 'selected-inputs.json').read_text())
    command = ['taskset', '-c', str(cpu), str(WORK / 'replay'), str(WORK / 'selected-inputs.dat')]
    env = dict(os.environ, LD_LIBRARY_PATH=str(RUNTIME) + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', RAYON_NUM_THREADS='1', SYMBOLICA_HIDE_BANNER='1')
    # Both metadata resolution and actual process maps must identify the private DSO.
    ldd = subprocess.check_output(['ldd', str(WORK / 'replay')], env=env, text=True, timeout=10)
    (REPORT / 'ldd.txt').write_text(ldd)
    assert str(RUNTIME / 'libmadloop_tirust.so') in ldd
    artifacts = dict(compiled['artifacts_sha256'])
    for p in (REPORT / 'compilation.json', REPORT / 'build.log', REPORT / 'run.py', REPORT / 'ldd.txt'):
        artifacts[name(p)] = digest(p)
    outcome = {'command': command, 'cpu': cpu, 'timeout_seconds': 180, 'complete': False,
        'performance_eligible': False, 'input_stream_sha256': selected['input_stream_sha256'],
        'source_manifest_sha256': selected['source_manifest_sha256'], 'artifacts_sha256': artifacts,
        'protected_inputs_before': protected, 'loaded_libraries_sha256': {},
        'lock': '.tirust-build/numerical-tests.lock', 'lock_policy': 'exclusive nonblocking acquisition',
        'runtime_policy': 'historical TIRust wrapper/module/DSO; unchanged NinjaRS archive; 256-bit working precision',
        'controlled_environment': {k: env[k] for k in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS', 'SYMBOLICA_HIDE_BANNER')}}
    process = None
    try:
        with ExitStack() as cleanup:
            lock = cleanup.enter_context((ROOT / '.tirust-build/numerical-tests.lock').open('a'))
            # A conflicting lane causes an explicit setup failure, never a hidden wait.
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            outcome['lock_acquired'] = True
            with (REPORT / 'output.log').open('x') as log:
                started = time.monotonic()
                process = subprocess.Popen(command, cwd=WORK, env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                # Cleanup terminates a timed-out child before releasing the lock.
                cleanup.callback(terminate, process)
                maps = ''
                while process.poll() is None and time.monotonic() - started < 5:
                    try:
                        maps = Path('/proc') .joinpath(str(process.pid), 'maps').read_text()
                    except FileNotFoundError:
                        break
                    if str(RUNTIME / 'libmadloop_tirust.so') in maps:
                        break
                    time.sleep(.01)
                (REPORT / 'process-maps.txt').write_text(maps)
                paths = {row.split(maxsplit=5)[5] for row in maps.splitlines()
                    if len(row.split(maxsplit=5)) == 6 and row.split(maxsplit=5)[5].startswith('/')}
                libraries = {name(Path(p)): digest(Path(p)) for p in paths if '.so' in Path(p).name}
                outcome['loaded_libraries_sha256'] = libraries
                assert name(RUNTIME / 'libmadloop_tirust.so') in libraries, 'Private DSO missing from actual process maps'
                process.wait(timeout=max(.1, 180 - (time.monotonic() - started)))
                outcome['returncode'] = process.returncode
                outcome['elapsed_seconds_not_a_benchmark'] = time.monotonic() - started
                outcome['complete'] = process.returncode == 0
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, command)
    except BaseException as exc:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        if process is not None:
            outcome['returncode'] = process.returncode
        outcome['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        outcome['protected_inputs_after'] = verify()
        outcome['protected_inputs_unchanged'] = outcome['protected_inputs_after'] == protected
        for file, field in (('output.log', 'output_sha256'), ('process-maps.txt', 'process_maps_sha256')):
            if (REPORT / file).exists():
                outcome[field] = digest(REPORT / file)
        dump(REPORT / 'execution.json', outcome)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build', 'run'))
    parser.add_argument('--cpu', type=int, default=9)
    args = parser.parse_args()
    build() if args.action == 'build' else run(args.cpu)
