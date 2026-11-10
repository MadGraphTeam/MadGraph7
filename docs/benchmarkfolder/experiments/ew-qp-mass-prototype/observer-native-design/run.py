#!/usr/bin/env python3
"""Four bounded fixed-QP diagnostics; default invocation executes nothing.

Each arm/backend receives private resource/card/point copies. A reviewed source
hash, successful matching build, unchanged inventory and pinned loaded DSOs are
required before evaluation. Numerical status interpretation belongs to analysis.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time

sys.dont_write_bytecode = True
import build as design

HERE, ROOT, WORK = design.HERE, design.ROOT, design.WORK
PROCESS_SECONDS = 120.0
TOTAL_SECONDS = 480.0
OUTPUT_NAMES = ('mass-observer.hex', 'observer-run.log')


def verify_build(prepared, identity):
    path = HERE / 'build.json'
    result = json.loads(path.read_text())
    if result['state'] != 'complete' or result['review_hash'] != identity:
        raise ValueError('A successful build of this exact reviewed source is required')
    design.verify_artifacts(prepared, skip=design.all_build_outputs(prepared))
    for name, expected in result['outputs'].items():
        if design.digest(design.work_path(name)) != expected:
            raise ValueError('Built artifact changed: ' + name)
    for arm in prepared['arms'].values():
        if design.relative(design.work_path(arm['executable'])) not in result['outputs']:
            raise ValueError('Executable is absent from the build evidence')
    return result


def bound_input(prepared, path):
    path = design.work_path(path)
    name = design.relative(path)
    if prepared['artifacts'].get(name) != design.digest(path):
        raise ValueError('Runtime input is not bound by preparation: ' + name)
    return path


def file_inventory(directory):
    if any(path.is_symlink() for path in directory.rglob('*')):
        raise ValueError('Unexpected symlink in private runtime files: ' + str(directory))
    return {str(path.relative_to(directory)): design.digest(path)
            for path in sorted(directory.rglob('*')) if path.is_file()}


def card_changes(text, backend):
    changes = {'MLReductionLib': str(backend), 'CTModeRun': '4', 'CTModeInit': '1',
               'UseQPIntegrandForNinjaRS': '.TRUE.', 'UseQPIntegrandForTIRust': '.TRUE.',
               'TIRustComputeFinite': '.TRUE.'}
    lines = text.splitlines(keepends=True)
    applied = {}
    for name, value in changes.items():
        positions = [i for i, line in enumerate(lines) if line.strip().lower() == '#' + name.lower()]
        if len(positions) != 1 or positions[0] + 1 >= len(lines):
            raise ValueError('Missing/duplicate card setting: ' + name)
        index = positions[0] + 1
        applied[name] = {'original': lines[index].strip(), 'new': value}
        lines[index] = value + '\n'
    return ''.join(lines), applied


def prepare_case(prepared, arm_name, backend):
    arm = prepared['arms'][arm_name]
    source = design.work_path(arm['process_dir'])
    target = design.writable_path(WORK / 'runs' / (arm_name + '-' + str(backend)))
    target.mkdir(parents=True, exist_ok=False)
    resources = design.work_path(source / 'MadLoop5_resources')
    if not resources.is_dir():
        raise ValueError('Historical resource directory missing')
    for path in sorted(resources.rglob('*')):
        if path.is_symlink():
            raise ValueError('Runtime resources must be private regular copies: ' + str(path))
        if path.is_file():
            bound_input(prepared, path)
    shutil.copytree(resources, target / 'MadLoop5_resources', symlinks=False)
    for name in ('param_card.dat', 'selected.dat'):
        shutil.copy2(bound_input(prepared, source / name), target / name)
    point_lines = [line.split() for line in (target / 'selected.dat').read_text().splitlines() if line.strip()]
    if point_lines[0] != ['2'] or len(point_lines) != 11:
        raise ValueError('Exactly the two frozen five-leg points are required')
    if any(len(line) != 4 or not all(math.isfinite(float(v)) for v in line) for line in point_lines[1:]):
        raise ValueError('Malformed frozen point stream')
    before = file_inventory(target)
    card = target / 'MadLoop5_resources/MadLoopParams.dat'
    text, changes = card_changes(card.read_text(), backend)
    card.write_text(text)
    after = file_inventory(target)
    changed = sorted(name for name in before if before[name] != after[name])
    if any(name != 'MadLoop5_resources/MadLoopParams.dat' for name in changed):
        raise ValueError('Unexpected staged runtime input change')
    inputs = {'arm': arm_name, 'backend': backend, 'before': before, 'after': after,
              'card_changes': changes, 'source_process_dir': design.relative(source),
              'scope': 'Fixed CTModeRun=4, initialization mode 1; no host stability acceptance claim'}
    design.reserve_report(target / 'inputs.json', inputs)
    return target, inputs


def verify_case_inputs(target, expected):
    design.writable_path(target)
    for name, sha256 in expected.items():
        path = design.writable_path(target / name)
        if not path.is_file() or design.digest(path) != sha256:
            raise ValueError('Frozen private runtime input changed: ' + str(path))


def loaded_libraries(text, cwd):
    loaded = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('linux-vdso.'):
            continue
        if 'not found' in line:
            raise ValueError('Unresolved runtime library: ' + line)
        match = re.fullmatch(r'(\S+)\s+=>\s+(\S+)\s+\(0x[0-9a-fA-F]+\)', line)
        if match:
            soname, spelling = match.groups()
        else:
            match = re.fullmatch(r'(\S+)\s+\(0x[0-9a-fA-F]+\)', line)
            if not match:
                raise ValueError('Unrecognised ldd record: ' + line)
            spelling = match[1]
            soname = Path(spelling).name
        path = Path(spelling)
        if not path.is_absolute():
            path = cwd / path
        path = path.resolve(strict=True)
        if soname in loaded:
            raise ValueError('Duplicate runtime library: ' + soname)
        loaded[soname] = {'path': str(path), 'sha256': design.digest(path)}
    return loaded


def runtime_environment(prepared):
    expected = prepared['expected_runtime']
    if not {'libninja.so.0', 'libcollier.so', 'libmadloop_tirust.so'} <= set(expected):
        raise ValueError('All three native shared libraries must be pinned')
    paths = []
    for record in expected.values():
        path = design.work_path(record['path'])
        if design.digest(path) != record['sha256']:
            raise ValueError('Pinned runtime library changed: ' + str(path))
        if path.parent not in paths:
            paths.append(path.parent)
    return design.execution_environment(paths)


def check_loaded(prepared, loaded):
    for soname, expected in prepared['expected_runtime'].items():
        wanted = {'path': str(design.work_path(expected['path'])), 'sha256': expected['sha256']}
        if loaded.get(soname) != wanted:
            raise ValueError('Runtime path/hash mismatch for ' + soname)


def run(prepared, supplied):
    identity, parts = design.authorize(supplied)
    compiled = verify_build(prepared, identity)
    env = runtime_environment(prepared)
    ldd = shutil.which('ldd')
    if ldd is None:
        raise ValueError('ldd is unavailable')
    report_path = HERE / 'run.json'
    report = {'schema': 'ew-qp-mass-observer-run-v1', 'state': 'reserved',
              'review_hash': identity, 'review_sources': parts,
              'build_json_sha256': design.digest(HERE / 'build.json'),
              'cpu': design.CPU, 'process_seconds': PROCESS_SECONDS,
              'total_seconds': TOTAL_SECONDS, 'maximum_evaluations': 4,
              'automatic_retry': False, 'performance_eligible': False,
              'host_stability_qualified': False, 'cases': [],
              'environment': {key: env[key] for key in ('LD_LIBRARY_PATH', 'OMP_NUM_THREADS',
                              'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS', 'MKL_NUM_THREADS',
                              'NUMEXPR_NUM_THREADS')}}
    design.reserve_report(report_path, report)
    try:
        with (ROOT / '.tirust-build/numerical-tests.lock').open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            verify_build(prepared, identity)
            # All private inputs are frozen before the first evaluator starts.
            cases = [(name, backend, *prepare_case(prepared, name, backend))
                     for name in ('original', 'candidate') for backend in (8, 9)]
            if any(inputs['before'] != cases[0][3]['before'] for _, _, _, inputs in cases[1:]):
                raise ValueError('The arms do not share identical frozen points, cards and resources')
            total_deadline = time.monotonic() + TOTAL_SECONDS
            first_loaded = None
            report['state'] = 'running'
            design.atomic_json(report_path, report)
            for arm_name, backend, target, inputs in cases:
                case = {'arm': arm_name, 'backend': backend, 'directory': design.relative(target),
                        'inputs_sha256': design.digest(target / 'inputs.json'), 'complete': False}
                report['cases'].append(case)
                deadline = min(total_deadline, time.monotonic() + PROCESS_SECONDS)
                try:
                    verify_build(prepared, identity)
                    verify_case_inputs(target, inputs['after'])
                    executable = design.work_path(prepared['arms'][arm_name]['executable'])
                    case['executable_sha256'] = design.digest(executable)
                    with (target / 'ldd.txt').open('x') as log:
                        case['loader_command'] = design.bounded_command([ldd, str(executable)], target,
                                                                         log, deadline, env)
                    if case['loader_command'].get('error') or case['loader_command']['returncode'] != 0:
                        raise RuntimeError('Runtime verification failed before evaluation')
                    loaded = loaded_libraries((target / 'ldd.txt').read_text(), target)
                    check_loaded(prepared, loaded)
                    if first_loaded is None:
                        first_loaded = loaded
                    elif loaded != first_loaded:
                        raise ValueError('Runtime dependency identity changed between cases')
                    case['loaded_libraries'] = loaded
                    verify_case_inputs(target, inputs['after'])
                    with (target / 'stdout.log').open('x') as log:
                        case['evaluation'] = design.bounded_command(
                            [str(executable), str(target / 'selected.dat')], target, log, deadline, env)
                    verify_case_inputs(target, inputs['after'])
                    verify_build(prepared, identity)
                    case['files_after'] = file_inventory(target)
                    case['complete'] = (not case['evaluation'].get('error') and
                                        case['evaluation']['returncode'] == 0 and
                                        all((target / name).is_file() and (target / name).stat().st_size > 0
                                            for name in OUTPUT_NAMES))
                    if not case['complete']:
                        case['error'] = {'type': 'IncompleteDiagnostic',
                                         'message': 'Nonzero exit, timeout or missing observer output; no retry'}
                except Exception as error:
                    case['complete'] = False
                    case['error'] = {'type': type(error).__name__, 'message': str(error)}
                    # Identity failures are not numerical refusals. Stop before
                    # any further evaluator can consume changed artifacts.
                    raise
                finally:
                    try:
                        case['files_after'] = file_inventory(target)
                    except Exception as error:
                        case['complete'] = False
                        case['inventory_error'] = {'type': type(error).__name__, 'message': str(error)}
                        raise
                    finally:
                        design.atomic_json(report_path, report)
            report['state'] = 'complete' if all(case['complete'] for case in report['cases']) else 'incomplete'
    except BaseException as error:
        report['state'] = 'failed'
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        design.atomic_json(report_path, report)
    return report['state'] == 'complete'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--show-review-hash', action='store_true')
    parser.add_argument('--execute-reviewed-source', metavar='SHA256')
    args = parser.parse_args()
    prepared = design.load_prepared()
    identity, parts = design.review_identity()
    if args.show_review_hash or not args.execute_reviewed_source:
        print(json.dumps({'state': 'source_only_no_execution', 'review_hash': identity,
                          'review_sources': parts, 'cpu': design.CPU,
                          'maximum_evaluations': 4, 'process_seconds': PROCESS_SECONDS,
                          'total_seconds': TOTAL_SECONDS}, indent=2))
        return
    if not run(prepared, args.execute_reviewed_source):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
