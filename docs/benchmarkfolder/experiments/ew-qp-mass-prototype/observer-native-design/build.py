#!/usr/bin/env python3
"""Guarded, bounded direct Fortran build; default invocation executes nothing.

prepared.json supplies explicit argv recipes and a complete input inventory.
The review hash binds that document and BOTH executable Python sources. This
guard records a caller's review; computing the hash is not authorization.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
WORK = ROOT / '.tirust-build/ew-qp-mass-prototype/observer-native-design'
PREPARED = HERE / 'prepared.json'
COMPILER = Path('/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/bin/gfortran')
COMPILER_SHA256 = '30c6495dcb18cc324629fc230302d9cd8d81143f0ca1635d752b3f8552d48cae'
CPU = 8
BUILD_SECONDS = 180.0


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def root_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError('Artifact leaves repository: ' + str(path))
    return resolved


def work_path(value):
    path = root_path(value)
    if not path.is_relative_to(WORK.resolve()):
        raise ValueError('Build/run output leaves isolated work directory: ' + str(path))
    return path


def writable_path(value):
    """Inspect lexical components BEFORE resolving a writable destination."""
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = Path(os.path.abspath(path))
    if WORK.resolve() != WORK or not path.is_relative_to(WORK):
        raise ValueError('Writable path leaves the canonical isolated tree: ' + str(path))
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError('Symlink in writable path: ' + str(component))
        if component == ROOT:
            break
    if path.is_file() and path.stat().st_nlink != 1:
        raise ValueError('Writable file has hard-link aliases: ' + str(path))
    return path


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def load_prepared():
    if Path.cwd().resolve() != ROOT:
        raise ValueError('Run from the MadGraph repository root: ' + str(ROOT))
    record = json.loads(PREPARED.read_text())
    if record['schema'] != 'ew-qp-mass-observer-prepare-v1':
        raise ValueError('Unexpected preparation schema')
    if root_path(record['work']) != WORK.resolve():
        raise ValueError('Unexpected prepared work directory')
    if set(record['arms']) != {'original', 'candidate'}:
        raise ValueError('Exactly original and candidate arms are required')
    if digest(HERE.parent / 'evidence-index.json') != record['source_package_index_sha256']:
        raise ValueError('Frozen source package index changed')
    return record


def review_identity():
    parts = {'prepared_sha256': digest(PREPARED),
             'build_py_sha256': digest(HERE / 'build.py'),
             'run_py_sha256': digest(HERE / 'run.py')}
    encoded = json.dumps(parts, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest(), parts


def authorize(supplied):
    identity, parts = review_identity()
    if supplied != identity:
        raise ValueError('Review hash does not match the prepared inputs and both scripts')
    return identity, parts


def verify_artifacts(prepared, skip=()):
    omitted = {root_path(path) for path in skip}
    for name, expected in prepared['artifacts'].items():
        path = root_path(name)
        if path in omitted:
            continue
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Prepared artifact changed or is missing: ' + name)


def all_build_outputs(prepared):
    return [writable_path(path) for arm in prepared['arms'].values()
            for path in arm['build_outputs']]


def atomic_json(path, value):
    """Update only a report already reserved by this invocation."""
    temporary = Path(str(path) + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def reserve_report(path, initial):
    with path.open('x') as stream:
        json.dump(initial, stream, indent=2, sort_keys=True)
        stream.write('\n')


def execution_environment(runtime_dirs=()):
    if os.environ.get('LD_PRELOAD') or os.environ.get('LD_AUDIT'):
        raise ValueError('LD_PRELOAD/LD_AUDIT are not qualified for this experiment')
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    if runtime_dirs:
        # No inherited empty component: it would search the current directory.
        inherited = [part for part in env.get('LD_LIBRARY_PATH', '').split(':') if part]
        env['LD_LIBRARY_PATH'] = ':'.join([str(path) for path in runtime_dirs] + inherited)
    return env


def bounded_command(argv, cwd, log, deadline, env):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Global execution deadline exhausted before launch')
    taskset = shutil.which('taskset')
    if taskset is None:
        raise ValueError('taskset is unavailable')
    command = [taskset, '-c', str(CPU)] + list(argv)
    event = {'command': command, 'cwd': relative(cwd), 'timeout_seconds': remaining,
             'started_unix': time.time()}
    start = time.monotonic()
    process = None
    try:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        event['pid'] = process.pid
        event['returncode'] = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        event['timed_out'] = False
    except BaseException as error:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        event['returncode'] = None if process is None else process.returncode
        event['timed_out'] = isinstance(error, (TimeoutError, subprocess.TimeoutExpired))
        event['error'] = {'type': type(error).__name__, 'message': str(error)}
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        event['elapsed_seconds'] = time.monotonic() - start
    return event


def validate_recipe(prepared, arm, argv, compile_only):
    if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
        raise ValueError('Recipes must be explicit nonempty argv lists')
    if Path(argv[0]) != COMPILER or argv.count('-o') != 1:
        raise ValueError('Recipe does not use the pinned compiler and explicit output')
    if ('-c' in argv) != compile_only:
        raise ValueError('Compile/link recipe mismatch')
    output = writable_path(argv[argv.index('-o') + 1])
    allowed = {writable_path(path) for path in arm['build_outputs']}
    process_dir = writable_path(arm['process_dir'])
    if output not in allowed or not output.is_relative_to(process_dir):
        raise ValueError('Unlisted or symlinked build output: ' + str(output))
    if not compile_only and output != work_path(arm['executable']):
        raise ValueError('Link output differs from the declared executable')
    if compile_only:
        source = work_path(argv[argv.index('-c') + 1])
        if not source.is_file() or source.suffix.lower() not in ('.f', '.f90'):
            raise ValueError('Invalid staged Fortran source: ' + str(source))
        if prepared['artifacts'].get(relative(source)) != digest(source):
            raise ValueError('Compile source is not bound by preparation: ' + str(source))
        if source in allowed:
            raise ValueError('Compile source is also declared writable: ' + str(source))
    # Bind every explicit file operand, and every header/module visible in the
    # private include search path. New objects/modules are separately declared
    # outputs; ambient files cannot quietly enter the reviewed build.
    search_dirs = [process_dir]
    for index, token in enumerate(argv[1:], 1):
        if token.startswith('-I'):
            directory = argv[index + 1] if token == '-I' else token[2:]
            search_dirs.append(work_path(directory))
        if token.startswith('-J'):
            directory = argv[index + 1] if token == '-J' else token[2:]
            if writable_path(directory) != process_dir:
                raise ValueError('Module output directory differs from private process directory')
        if token.startswith('/') and Path(token).is_file():
            operand = work_path(token)
            if operand not in allowed and relative(operand) not in prepared['artifacts']:
                raise ValueError('Uninventoried recipe input: ' + str(operand))
    for directory in set(search_dirs):
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in ('.mod', '.smod', '.inc', '.h'):
                if path not in allowed and relative(path) not in prepared['artifacts']:
                    raise ValueError('Uninventoried visible include/module: ' + str(path))
    return output


def build(prepared, supplied):
    identity, parts = authorize(supplied)
    verify_artifacts(prepared)
    if digest(COMPILER) != COMPILER_SHA256:
        raise ValueError('Pinned compiler bytes changed')
    outputs = all_build_outputs(prepared)
    if len(set(outputs)) != len(outputs):
        raise ValueError('Original/candidate build outputs overlap')
    for arm in prepared['arms'].values():
        writable_path(arm['process_dir'])
        for argv in arm['compile_commands']:
            validate_recipe(prepared, arm, argv, True)
        validate_recipe(prepared, arm, arm['link_command'], False)
    env = execution_environment()
    report_path = HERE / 'build.json'
    report = {'schema': 'ew-qp-mass-observer-build-v1', 'state': 'reserved',
              'review_hash': identity, 'review_sources': parts, 'cpu': CPU, 'jobs': 1,
              'deadline_seconds': BUILD_SECONDS, 'native_reducer_calls': 0,
              'commands': [], 'outputs': {}, 'performance_eligible': False,
              'environment': {key: value for key, value in sorted(env.items())
                              if key.startswith(('NIX_', 'LD_', 'OMP_', 'OPENBLAS_', 'RAYON_',
                                                 'MKL_', 'NUMEXPR_'))}}
    reserve_report(report_path, report)
    try:
        with (ROOT / '.tirust-build/numerical-tests.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            verify_artifacts(prepared)
            deadline = time.monotonic() + BUILD_SECONDS
            report['state'] = 'running'
            atomic_json(report_path, report)
            with (HERE / 'build.log').open('x') as log:
                for name in ('original', 'candidate'):
                    arm = prepared['arms'][name]
                    for argv in arm['compile_commands'] + [arm['link_command']]:
                        validate_recipe(prepared, arm, argv, '-c' in argv)
                        log.write(json.dumps({'arm': name, 'argv': argv}) + '\n')
                        log.flush()
                        event = bounded_command(argv, work_path(arm['process_dir']), log, deadline, env)
                        event['arm'] = name
                        report['commands'].append(event)
                        atomic_json(report_path, report)
                        if event.get('error') or event['returncode'] != 0:
                            raise RuntimeError('Build command failed; no retry: ' + str(event))
            verify_artifacts(prepared, skip=outputs)
            for path in outputs:
                if not path.is_file():
                    raise ValueError('Declared build output missing: ' + str(path))
                report['outputs'][relative(path)] = digest(path)
            report['state'] = 'complete'
    except BaseException as error:
        report['state'] = 'failed'
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        # Retain every produced file, including outputs of an incomplete build.
        for path in outputs:
            if path.is_file():
                report['outputs'][relative(path)] = digest(path)
        if (HERE / 'build.log').exists():
            report['build_log_sha256'] = digest(HERE / 'build.log')
        atomic_json(report_path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show-review-hash', action='store_true')
    parser.add_argument('--execute-reviewed-source', metavar='SHA256')
    args = parser.parse_args()
    prepared = load_prepared()
    identity, parts = review_identity()
    if args.show_review_hash or not args.execute_reviewed_source:
        print(json.dumps({'state': 'source_only_no_execution', 'review_hash': identity,
                          'review_sources': parts, 'cpu': CPU, 'jobs': 1,
                          'build_seconds': BUILD_SECONDS}, indent=2))
        return
    build(prepared, args.execute_reviewed_source)


if __name__ == '__main__':
    main()
