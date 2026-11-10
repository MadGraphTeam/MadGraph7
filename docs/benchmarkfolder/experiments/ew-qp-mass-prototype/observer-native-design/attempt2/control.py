#!/usr/bin/env python3
"""Append-only attempt 2 controller. Default invocation is read-only.

A matching review hash identifies reviewed sources; it is not execution
authorization. Build, card-only smoke, and four diagnostic runs are separate
explicit actions. No action retries a failed prior action.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
PRIOR = HERE.parent
WORK = ROOT / '.tirust-build/ew-qp-mass-prototype/observer-native-design/attempt2'
CPU = 8
BUILD_SECONDS, SMOKE_SECONDS = 180.0, 30.0
PROCESS_SECONDS, TOTAL_SECONDS = 120.0, 480.0
COMPILER = Path('/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/bin/gfortran')
COMPILER_SHA256 = '30c6495dcb18cc324629fc230302d9cd8d81143f0ca1635d752b3f8552d48cae'
OUTPUTS = ('mass-observer.hex', 'observer-run.log', 'param.log')
_pending_signal = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def root_path(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    resolved = path.resolve()
    require(resolved.is_relative_to(ROOT), 'Path outside repository: ' + str(path))
    return resolved


def work_path(value):
    path = root_path(value)
    require(path.is_relative_to(WORK), 'Path outside attempt 2: ' + str(path))
    return path


def writable(value):
    path = Path(value)
    path = Path(os.path.abspath(path if path.is_absolute() else ROOT / path))
    require(WORK.resolve() == WORK and path.is_relative_to(WORK), 'Nonprivate output path: ' + str(path))
    for component in (path, *path.parents):
        require(not component.is_symlink(), 'Symlink in writable path: ' + str(component))
        if component == ROOT:
            break
    require(not path.is_file() or path.stat().st_nlink == 1, 'Hard-linked output: ' + str(path))
    return path


def inventory(directory):
    require(not any(p.is_symlink() for p in directory.rglob('*')), 'Symlink in private file tree')
    return {str(p.relative_to(directory)): digest(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    require(not temporary.exists(), 'Unfinished report update exists: ' + str(temporary))
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
    temporary.replace(path)


def reserve(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def load_prepared():
    require(Path.cwd().resolve() == ROOT, 'Run from the MadGraph repository root')
    prepared = json.loads((HERE / 'prepared.json').read_text())
    require(prepared['schema'] == 'ew-qp-mass-observer-attempt2-prepare-v1', 'Wrong preparation schema')
    require(root_path(prepared['work']) == WORK, 'Wrong attempt 2 directory')
    require(set(prepared['arms']) == {'original', 'candidate'}, 'Two observer arms required')
    return prepared


def review_identity():
    names = ('prepared.json', 'control.py', 'prepare.py', 'check_observed.f', 'check_setpara.f')
    parts = {name: digest(HERE / name) for name in names}
    encoded = json.dumps(parts, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest(), parts


def verify_inputs(prepared):
    check_interrupt()
    for section in ('artifacts', 'prior_evidence', 'source_files'):
        for name, expected in prepared[section].items():
            require(digest(root_path(name)) == expected, 'Changed ' + section + ': ' + name)


def environment(prepared, reducers=False):
    env = os.environ.copy()
    require(not env.get('LD_PRELOAD') and not env.get('LD_AUDIT'), 'Loader injection variables are forbidden')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS',
                 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    if reducers:
        dirs = sorted({str(work_path(info['path']).parent) for info in prepared['expected_runtime'].values()})
        old = [x for x in env.get('LD_LIBRARY_PATH', '').split(':') if x]
        env['LD_LIBRARY_PATH'] = ':'.join(dirs + old)
    return env


class ExecutionInterrupted(BaseException):
    pass


def check_interrupt():
    if _pending_signal is not None:
        raise ExecutionInterrupted('Controller interrupted by signal ' + str(_pending_signal))


@contextmanager
def signal_cleanup():
    def interrupted(number, _frame):
        global _pending_signal
        # Defer every signal until the owner reaches a safe cleanup boundary.
        # Children inherit neither blocked masks nor these caught handlers.
        _pending_signal = number
    old = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield
    finally:
        for number, handler in old.items():
            signal.signal(number, handler)


def bounded(argv, cwd, log, deadline, env):
    check_interrupt()
    remaining = deadline - time.monotonic()
    require(remaining > 0, 'Execution deadline exhausted before launch')
    taskset = shutil.which('taskset')
    require(taskset is not None, 'taskset unavailable')
    command = [taskset, '-c', str(CPU), *argv]
    event = {'command': command, 'cwd': relative(cwd), 'timeout_seconds': remaining,
             'started_unix': time.time(), 'returncode': None, 'timed_out': False}
    start, process = time.monotonic(), None
    try:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        event['pid'] = process.pid
        while True:
            check_interrupt()
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, deadline-start)
            try:
                event['returncode'] = process.wait(timeout=min(0.2, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
    except BaseException as error:
        event['timed_out'] = isinstance(error, (TimeoutError, subprocess.TimeoutExpired))
        event['error'] = {'type': type(error).__name__, 'message': str(error)}
        # Return the failure event after cleanup so callers retain its process
        # identity before rejecting the stage. Signals never launch a retry.
    finally:
        if process is not None:
            # Also clean orphaned descendants after the direct child exits.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            event['returncode'] = process.returncode
            event['process_group_cleanup'] = 'SIGKILL sent if group existed; direct child reaped'
        event['elapsed_seconds'] = time.monotonic() - start
        log.flush()
    return event


def successful(event):
    return event['returncode'] == 0 and not event['timed_out'] and not event.get('error')


def check_recipe(prepared, job):
    argv = job['argv']
    require(argv[0] == str(COMPILER) and argv.count('-o') == 1, 'Unpinned compiler/output')
    output = writable(argv[argv.index('-o') + 1])
    require(relative(output) in prepared['build_outputs'], 'Undeclared build output')
    require(not output.exists(), 'Build output already exists; no retry')
    if '-c' in argv:
        source = work_path(argv[argv.index('-c') + 1])
        require(relative(source) in prepared['artifacts'], 'Unbound compile source')
    search = [work_path(job['cwd'])]
    for arg in argv[1:]:
        if arg.startswith('-I'):
            search.append(work_path(arg[2:]))
        if arg.startswith('-J'):
            require(writable(arg[2:]) == work_path(job['cwd']), 'Wrong module output directory')
        if arg.startswith('/') and Path(arg).is_file():
            name = relative(work_path(arg))
            require(name in prepared['artifacts'] or name in prepared['build_outputs'], 'Unbound recipe operand')
    for directory in search:
        for path in directory.iterdir():
            if path.is_file() and path.suffix.lower() in ('.mod', '.smod', '.inc', '.h'):
                require(relative(path) in prepared['artifacts'], 'Unbound visible include/module')


def verify_build(prepared, identity):
    report = json.loads((HERE / 'build.json').read_text())
    require(report['state'] == 'complete' and report['review_hash'] == identity, 'Matching successful build required')
    require(set(report['outputs']) == set(prepared['build_outputs']), 'Incomplete built outputs')
    verify_inputs(prepared)
    for name, expected in report['outputs'].items():
        require(digest(work_path(name)) == expected, 'Built output changed: ' + name)
    return report


def build(prepared, report):
    require(digest(COMPILER) == COMPILER_SHA256, 'Compiler changed')
    deadline = time.monotonic() + BUILD_SECONDS
    report['commands'], report['outputs'] = [], {}
    env = environment(prepared)
    report['environment'] = {key: value for key, value in sorted(env.items())
                             if key.startswith(('NIX_', 'LD_', 'OMP_', 'OPENBLAS_', 'RAYON_', 'MKL_', 'NUMEXPR_'))
                             or key in ('PATH', 'LIBRARY_PATH', 'COMPILER_PATH', 'GCC_EXEC_PREFIX', 'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'GFORTRAN_INCLUDE_PATH')}
    frozen_projection = {key: value for key, value in env.items()
                         if key.startswith(('NIX_', 'LD_', 'OMP_', 'OPENBLAS_', 'RAYON_', 'MKL_', 'NUMEXPR_'))}
    require(frozen_projection == prepared['prior_build_environment'], 'Historical compiler environment projection changed')
    for key in ('LIBRARY_PATH', 'COMPILER_PATH', 'GCC_EXEC_PREFIX', 'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'GFORTRAN_INCLUDE_PATH'):
        require(not env.get(key), 'Unpinned compiler search override: '+key)
    with (HERE / 'build.log').open('x') as log:
        for job in prepared['build_jobs']:
            verify_inputs(prepared)
            check_recipe(prepared, job)
            log.write(json.dumps(job) + '\n');log.flush()
            event = bounded(job['argv'], work_path(job['cwd']), log, deadline, env)
            event['name'] = job['name'];report['commands'].append(event)
            atomic_json(HERE / 'build.json', report)
            require(successful(event), 'Build failed; no retry: ' + job['name'])
    verify_inputs(prepared)
    for name in prepared['build_outputs']:
        report['outputs'][name] = digest(work_path(name))
    report['build_log_sha256'] = digest(HERE / 'build.log')


def loaded_libraries(text, cwd):
    loaded = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('linux-vdso.'):
            continue
        require('not found' not in line, 'Unresolved library: ' + line)
        match = re.fullmatch(r'(\S+)\s+=>\s+(\S+)\s+\(0x[0-9a-fA-F]+\)', line)
        if match:
            soname, spelling = match.groups()
        else:
            match = re.fullmatch(r'(\S+)\s+\(0x[0-9a-fA-F]+\)', line)
            require(match is not None, 'Unrecognised loader line: ' + line)
            spelling = match[1];soname = Path(spelling).name
        path = Path(spelling)
        path = (path if path.is_absolute() else cwd / path).resolve(strict=True)
        require(soname not in loaded, 'Duplicate runtime SONAME')
        loaded[soname] = {'path': str(path), 'sha256': digest(path)}
    return loaded


def verify_case_inputs(directory, expected):
    for name, sha in expected.items():
        path = writable(directory / name)
        require(path.is_file() and digest(path) == sha, 'Runtime input changed: ' + name)


def stage_case(prepared, name, arm='original', backend=None):
    target = writable(WORK / 'runs' / name)
    target.mkdir(parents=True, exist_ok=False)
    source = work_path(prepared['arms'][arm]['process_dir'])
    for path in (source/'MadLoop5_resources').rglob('*'):
        require(not path.is_symlink(), 'Symlink in staged resource source')
        if path.is_file():
            require(prepared['artifacts'].get(relative(path)) == digest(path), 'Unbound resource source')
    for filename in ('param_card.dat', 'selected.dat'):
        require(prepared['artifacts'].get(relative(source/filename)) == digest(source/filename), 'Unbound runtime source')
    shutil.copytree(source / 'MadLoop5_resources', target / 'MadLoop5_resources')
    for filename in ('param_card.dat', 'selected.dat'):
        shutil.copyfile(source / filename, target / filename)
    before = inventory(target)
    changes = {}
    if backend is not None:
        point_lines = [s.split() for s in (target / 'selected.dat').read_text().splitlines() if s.strip()]
        require(point_lines[0] == ['2'] and len(point_lines) == 11, 'Wrong saved point count')
        require(all(len(r) == 4 and all(math.isfinite(float(x)) for x in r) for r in point_lines[1:]), 'Bad points')
        card = target / 'MadLoop5_resources/MadLoopParams.dat'
        lines = card.read_text().splitlines(keepends=True)
        values = {'MLReductionLib': str(backend), 'CTModeRun': '4', 'CTModeInit': '1',
                  'UseQPIntegrandForNinjaRS': '.TRUE.', 'UseQPIntegrandForTIRust': '.TRUE.',
                  'TIRustComputeFinite': '.TRUE.'}
        for label, value in values.items():
            where = [i for i, s in enumerate(lines) if s.strip().lower() == '#' + label.lower()]
            require(len(where) == 1 and where[0]+1 < len(lines), 'Missing/duplicate card key: ' + label)
            index = where[0]+1;changes[label] = {'original': lines[index].strip(), 'new': value}
            lines[index] = value+'\n'
        card.write_text(''.join(lines))
    after = inventory(target)
    inputs = {'arm': arm, 'backend': backend, 'before': before, 'after': after, 'card_changes': changes,
              'source_process_dir': relative(source), 'parameter_path': 'MadLoop5_resources/param_card.dat'}
    reserve(target / 'inputs.json', inputs)
    return target, inputs


def smoke_output(directory, prepared):
    # Exact model record shapes are supplied by the reviewed smoke source contract.
    lines = (directory / 'card-smoke.hex').read_text().splitlines()
    contract = prepared['smoke']['output_contract']
    require(lines and lines[0] == contract['header'] and lines[-1] == 'COMPLETE_CARD_ONLY', 'Incomplete card-only smoke')
    require(len(lines) == 12, 'Wrong smoke record count')
    names = ('MT', 'MZ', 'MW', 'MH', 'WT', 'WZ', 'WW', 'WH', 'MU')
    require(set(contract['model_words']) == set(names), 'Wrong model smoke contract')
    for line, label in zip(lines[1:10], names):
        row = line.split()
        require(row == ['MODEL', label, *contract['model_words'][label]], 'Wrong model smoke bits: '+label)
    flags = lines[10].split()
    require(len(flags) == 6 and flags[0] == 'MODEL_IEEE_FLAGS' and all(f in ('T', 'F') for f in flags[1:]), 'Malformed model IEEE flags')
    require((directory / 'param.log').stat().st_size > 0, 'Empty model parameter log')
    observed = inventory(directory)
    allowed = set(json.loads((directory/'inputs.json').read_text())['after']) | {'inputs.json', 'ldd.txt', 'symbols.txt', 'stdout.log', 'param.log', 'card-smoke.hex'}
    require(set(observed) == allowed, 'Unexpected/missing card-only files')
    return observed


def smoke(prepared, report, identity):
    verify_build(prepared, identity)
    directory, inputs = stage_case(prepared, 'card-only')
    deadline = time.monotonic() + SMOKE_SECONDS
    exe = work_path(prepared['smoke']['executable'])
    report.update(directory=relative(directory), inputs_sha256=digest(directory/'inputs.json'),
                  executable_sha256=digest(exe), build_json_sha256=digest(HERE/'build.json'))
    env = environment(prepared)
    for label, argv in [('loader', [shutil.which('ldd'), str(exe)]), ('symbols', [shutil.which('nm'), '-a', str(exe)])]:
        require(argv[0] is not None, label+' tool missing')
        filename = 'ldd.txt' if label == 'loader' else 'symbols.txt'
        with (directory/filename).open('x') as log:
            report[label] = bounded(argv, directory, log, deadline, env)
        require(successful(report[label]), 'Smoke inspection failed')
    loaded = loaded_libraries((directory/'ldd.txt').read_text(), directory)
    require(not any(re.search(r'tirust|ninjars|ninja|collier', x, re.I) for x in loaded), 'Reducer DSO in card-only smoke')
    require(not re.search(r'ml_tirust|mnrs|tirust|ninjars|ninja|collier', (directory/'symbols.txt').read_text(), re.I), 'Reducer symbol in card-only smoke')
    report['loaded_libraries'] = loaded
    for name, info in loaded.items():
        require(prepared['prior_system_runtime'].get(name) == info, 'Smoke system runtime changed: '+name)
    verify_case_inputs(directory, inputs['after'])
    with (directory/'stdout.log').open('x') as log:
        report['evaluation'] = bounded([str(exe)], directory, log, deadline, env)
    require(successful(report['evaluation']), 'Card-only smoke failed; no retry')
    verify_case_inputs(directory, inputs['after']);verify_build(prepared, identity)
    report['files_after'] = smoke_output(directory, prepared)


def verify_smoke(prepared, identity):
    report = json.loads((HERE/'smoke.json').read_text())
    require(report['state'] == 'complete' and report['review_hash'] == identity, 'Matching successful smoke required')
    require(report['build_json_sha256'] == digest(HERE/'build.json'), 'Smoke/build identity mismatch')
    directory = work_path(report['directory'])
    require(inventory(directory) == report['files_after'], 'Smoke artifacts changed')
    smoke_output(directory, prepared)
    return report


def run(prepared, report, identity):
    verify_build(prepared, identity);verify_smoke(prepared, identity)
    report.update(build_json_sha256=digest(HERE/'build.json'), smoke_json_sha256=digest(HERE/'smoke.json'), cases=[])
    cases = [(a, b, *stage_case(prepared, a+'-'+str(b), a, b)) for a in ('original', 'candidate') for b in (8, 9)]
    require(all(x[3]['before'] == cases[0][3]['before'] for x in cases), 'Different frozen arm inputs')
    env = environment(prepared, reducers=True)
    report['environment'] = {k: env[k] for k in ('LD_LIBRARY_PATH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')}
    deadline = time.monotonic() + TOTAL_SECONDS
    first_loaded = None
    for arm, backend, directory, inputs in cases:
        case = {'arm': arm, 'backend': backend, 'directory': relative(directory), 'complete': False,
                'inputs_sha256': digest(directory/'inputs.json')}
        report['cases'].append(case)
        process_deadline = min(deadline, time.monotonic()+PROCESS_SECONDS)
        try:
            verify_build(prepared, identity);verify_case_inputs(directory, inputs['after'])
            exe = work_path(prepared['arms'][arm]['executable'])
            case['executable_sha256'] = digest(exe)
            ldd = shutil.which('ldd');require(ldd is not None, 'ldd unavailable')
            with (directory/'ldd.txt').open('x') as log:
                case['loader_command'] = bounded([ldd, str(exe)], directory, log, process_deadline, env)
            require(successful(case['loader_command']), 'Loader inspection failed')
            loaded = loaded_libraries((directory/'ldd.txt').read_text(), directory)
            for name, info in prepared['expected_runtime'].items():
                require(loaded.get(name) == {'path': str(work_path(info['path'])), 'sha256': info['sha256']}, 'Pinned DSO mismatch: '+name)
            system = {name: info for name, info in loaded.items() if name not in prepared['expected_runtime']}
            require(system == prepared['prior_system_runtime'], 'System runtime changed from attempt 1')
            if first_loaded is None:first_loaded = loaded
            require(loaded == first_loaded, 'Runtime identities changed between cases')
            case['loaded_libraries'] = loaded
            verify_case_inputs(directory, inputs['after'])
            with (directory/'stdout.log').open('x') as log:
                case['evaluation'] = bounded([str(exe), str(directory/'selected.dat')], directory, log, process_deadline, env)
            verify_case_inputs(directory, inputs['after']);verify_build(prepared, identity)
            require(successful(case['evaluation']), 'Diagnostic process failed; no retry')
            require(all((directory/n).is_file() and (directory/n).stat().st_size for n in OUTPUTS), 'Missing/empty diagnostic output')
            require((directory/'observer-run.log').read_text().splitlines()[-1].strip() == 'COMPLETE_DIAGNOSTIC_ONLY', 'Missing driver completion marker')
            case['complete'] = True
        except BaseException as error:
            case['error'] = {'type': type(error).__name__, 'message': str(error)}
            raise
        finally:
            try:
                case['files_after'] = inventory(directory)
            except BaseException as error:
                case['complete'] = False
                case['inventory_error'] = {'type': type(error).__name__, 'message': str(error)}
                raise
            finally:
                atomic_json(HERE/'run.json', report)


def execute(prepared, mode, identity, parts):
    path = HERE/(mode+'.json')
    report = {'schema': 'ew-qp-mass-observer-attempt2-'+mode+'-v1', 'state': 'reserved',
              'review_hash': identity, 'review_sources': parts, 'cpu': CPU, 'jobs': 1,
              'automatic_retry': False, 'host_stability_qualified': False, 'performance_eligible': False,
              'build_seconds': BUILD_SECONDS, 'smoke_seconds': SMOKE_SECONDS,
              'process_seconds': PROCESS_SECONDS, 'total_seconds': TOTAL_SECONDS}
    reserve(path, report)
    try:
        with signal_cleanup(), (ROOT/'.tirust-build/numerical-tests.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            verify_inputs(prepared)
            report['state'] = 'running';atomic_json(path, report)
            if mode == 'build':build(prepared, report)
            elif mode == 'smoke':smoke(prepared, report, identity)
            else:run(prepared, report, identity)
            check_interrupt()
            report['state'] = 'complete'
    except BaseException as error:
        report['state'] = 'failed';report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        # Retain partial artifacts even when a compiler/driver was interrupted.
        if mode == 'build':
            report['outputs'] = {n: digest(work_path(n)) for n in prepared['build_outputs'] if work_path(n).is_file()}
            if (HERE/'build.log').is_file():report['build_log_sha256'] = digest(HERE/'build.log')
        if mode == 'smoke' and (WORK/'runs/card-only').is_dir():report['files_after'] = inventory(WORK/'runs/card-only')
        atomic_json(path, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('build', 'smoke', 'run'))
    parser.add_argument('--show-review-hash', action='store_true')
    parser.add_argument('--execute-reviewed-source', metavar='SHA256')
    args = parser.parse_args()
    prepared = load_prepared();identity, parts = review_identity()
    if args.show_review_hash or not args.execute_reviewed_source:
        print(json.dumps({'state': 'source_only_no_execution', 'review_hash': identity, 'review_sources': parts,
                          'compiler_jobs': len(prepared['build_jobs']), 'cpu': CPU}, indent=2))
        return
    require(args.mode is not None, 'An explicit mode is required')
    require(args.execute_reviewed_source == identity, 'Review hash mismatch')
    execute(prepared, args.mode, identity, parts)


if __name__ == '__main__':
    main()
