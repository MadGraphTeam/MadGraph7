#!/usr/bin/env python3
"""Compile/run only the source-derived wrapper policy fixture, pinned to CPU 11.

--prepare-only copies the pinned wrapper/preparation and makes two explicit
mutants without launching a compiler or executable. Results are exclusive
directories; reruns must use a new --run-name. All subprocesses run from the
repository root. No native reducer, model library, or physical amplitude links.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
PROTOTYPE = HERE.parents[1]
ROOT = PROTOTYPE.parents[3]
PREPARATION_SHA = 'e4f6235e7c7d3d337c567a0acd248cfa49daaaa4d265f842f4ebf5a54a81602f'
WRAPPER_SHA = 'be514aaffe9e478ce86bbfe1d83f6fdc7f08ff249b66898a7aba4a440787cfb0'


def require(value, label):
    if not value:
        raise ValueError(label)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path):
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                sha256=sha(path))


def keep_exact(path, data):
    if path.exists():
        require(path.read_bytes() == data, 'Existing retained source differs: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(data)


def prepare():
    preparation = PROTOTYPE / 'preparation.json'
    wrapper = PROTOTYPE / 'born_rescue.f'
    require(sha(preparation) == PREPARATION_SHA, 'Revision 2 preparation identity changed')
    metadata = json.loads(preparation.read_bytes())
    require(metadata['revision'] == 2, 'Expected revision 2')
    require(sha(wrapper) == WRAPPER_SHA == metadata['files_sha256'][wrapper.relative_to(ROOT).as_posix()],
            'Actual wrapper is not bound to the pinned preparation')
    source = wrapper.read_text()
    require(source.count('      SUBROUTINE ') == 1 and
            source.startswith('      SUBROUTINE ML5_0_BORN_WITH_RESCUE(') and
            source.rstrip().endswith('      END'), 'Expected one complete actual wrapper')
    restore = '      CALL IEEE_SET_STATUS(ENTRY_STATUS)\n'
    require(source.count(restore) == 2, 'Expected exactly two restore sites')
    position = source.rindex(restore)
    missing_restore = source[:position] + source[position + len(restore):]
    snapshot = '      CALL IEEE_GET_STATUS(ENTRY_STATUS)\n'
    require(source.count(snapshot) == 1, 'Expected one entry snapshot')
    changed_rounding = source.replace(snapshot, snapshot +
        '      CALL IEEE_SET_ROUNDING_MODE(IEEE_NEAREST)\n')
    keep_exact(HERE / 'preparation-used.json', preparation.read_bytes())
    keep_exact(HERE / 'wrapper-used.f', wrapper.read_bytes())
    keep_exact(HERE / 'mutation-missing-final-restore.f', missing_restore.encode())
    keep_exact(HERE / 'mutation-changed-rounding.f', changed_rounding.encode())
    return {name: identity(HERE / name) for name in (
        'preparation-used.json', 'wrapper-used.f', 'rounding_driver.f90',
        'mutation-missing-final-restore.f', 'mutation-changed-rounding.f', 'run_rounding.py')}


def parse_output(text, negative):
    rows = [line.split() for line in text.splitlines() if line.strip()]
    require(rows and rows[0][0] == 'FLAG_BITS' and len(rows[0]) == 6, 'Missing flag identities')
    supports, cases = [], []
    for row in rows[1:-1]:
        if row[0] == 'SUPPORT':
            require(len(row) == 4, 'Support frame shape')
            supports.append(list(map(int, row[1:])))
        else:
            require(row[0] == 'CASE' and len(row) == 19, 'Case frame shape')
            values = list(map(int, row[1:17]))
            require(all(re.fullmatch('[0-9A-F]{16}', value) for value in row[17:]), 'Answer bit frame')
            cases.append(dict(mode=values[0], action=values[1], entry_flag_seed=values[2],
                              entry_trap_policy=values[3], error_bits=values[-1],
                              observed_flags=values[7], expected_flags=values[8]))
    require([row[0] for row in supports] == [1, 2, 3, 4, 5], 'Incomplete rounding support inventory')
    require(all(dp in (0, 1) and qp in (0, 1) for _, dp, qp in supports), 'Invalid support value')
    supported = [mode for mode, dp, _ in supports if dp]
    expected = [(mode, action, seed, trap) for mode in supported
                for action in range(1, 5) for seed in range(4) for trap in range(2)]
    require([(r['mode'], r['action'], r['entry_flag_seed'], r['entry_trap_policy']) for r in cases]
            == expected, 'Incomplete/reordered fixture cases')
    require(rows[-1][0] == 'RESULT' and len(rows[-1]) == 4, 'Missing final result')
    result = list(map(int, rows[-1][1:]))
    failures = sum(bool(row['error_bits']) for row in cases)
    require(result == [len(supported), len(expected), failures], 'Final count mismatch')
    require(failures > 0 if negative else failures == 0, 'Unexpected policy outcome')
    return dict(rounding_support=[dict(mode=mode, dp=bool(dp), qp=bool(qp)) for mode, dp, qp in supports],
                cases=len(cases), failing_cases=failures, cases_by_action={
                    str(action): sum(row['action'] == action for row in cases) for action in range(1, 5)},
                observed_error_bits=sorted({row['error_bits'] for row in cases}),
                assertions=cases if negative else 'All per-case assertions passed; raw frames retained.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-name', default='run-1')
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, 'Run from the repository root')
    inputs = prepare()
    if args.prepare_only:
        print(json.dumps(dict(prepared=True, compiled=False, inputs=inputs), indent=2))
        return
    require(re.fullmatch(r'run-[1-9][0-9]*', args.run_name), 'Use an exclusive run-N directory')
    compiler = shutil.which('gfortran')
    require(compiler, 'gfortran is required')
    taskset = shutil.which('taskset')
    require(taskset, 'taskset is required')
    result_dir = HERE / args.run_name
    work = HERE / 'work' / args.run_name
    result_dir.mkdir()
    work.mkdir(parents=True)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    record = dict(schema=1, complete=False, cpu=11, scope='Source-derived wrapper IEEE policy fixture with explicit DP/QP stubs.',
        physical_amplitudes=False, native_reducers=False, performance_eligible=False,
        source_inputs=inputs, preparation_sha256=PREPARATION_SHA, wrapper_sha256=WRAPPER_SHA,
        compiler=dict(path=compiler, sha256=sha(Path(compiler))), runs=[],
        limitations=[
            'This fixture tests wrapper control and IEEE state, not physical matrix elements or reducer accuracy.',
            'QP stubs use exact constants; the rounding-sensitive operation is the actual wrapper conversion from QP to DP.',
            'The enabled trap policy tests preservation of divide-by-zero halting without deliberately triggering it.',
            'Sticky entry flags are underflow/inexact; failed-DP invalid is deliberately discarded, and new QP conversion flags are checked.',
            'Executables/module files remain in ignored work/; their hashes and exact commands are retained.'])
    configurations = [('original-O0', 'wrapper-used.f', '-O0', None),
                      ('original-O3', 'wrapper-used.f', '-O3', None),
                      ('missing-final-restore', 'mutation-missing-final-restore.f', '-O0', 9),
                      ('changed-rounding', 'mutation-changed-rounding.f', '-O0', 0)]
    try:
        version = subprocess.check_output([taskset, '-c', '11', compiler, '--version'], cwd=ROOT, text=True)
        keep_exact(result_dir / 'compiler-version.txt', version.encode())
        record['compiler']['version'] = version.splitlines()[0]
        for name, wrapper, optimization, mutation_bit in configurations:
            directory = work / name
            directory.mkdir()
            executable = directory / 'check-rounding'
            command = [taskset, '-c', '11', compiler, optimization, '-frounding-math',
                       '-fno-unsafe-math-optimizations', '-fcheck=all', '-ffixed-line-length-none',
                       '-J' + str(directory), '-I' + str(directory),
                       str(HERE / 'rounding_driver.f90'), str(HERE / wrapper), '-o', str(executable)]
            run = dict(name=name, compile_command=command, compile_cwd=str(ROOT), mutation=mutation_bit is not None)
            record['runs'].append(run)
            built = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=90)
            keep_exact(result_dir / (name + '-build.stdout'), built.stdout)
            keep_exact(result_dir / (name + '-build.stderr'), built.stderr)
            run['compile_returncode'] = built.returncode
            require(built.returncode == 0, 'Fixture compilation failed: ' + name)
            run['executable'] = identity(executable)
            linked = subprocess.check_output([taskset, '-c', '11', 'ldd', str(executable)], cwd=ROOT)
            keep_exact(result_dir / (name + '-ldd.txt'), linked)
            require(not any(token in linked.lower() for token in (b'tirust', b'ninjars', b'symbolica', b'collier')),
                    'Unexpected native reducer dependency')
            command = [taskset, '-c', '11', str(executable)]
            run['execute_command'] = command
            executed = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=30)
            keep_exact(result_dir / (name + '.stdout'), executed.stdout)
            keep_exact(result_dir / (name + '.stderr'), executed.stderr)
            run['returncode'] = executed.returncode
            require(executed.returncode == (91 if mutation_bit is not None else 0), 'Unexpected fixture exit: ' + name)
            run['summary'] = parse_output(executed.stdout.decode(), mutation_bit is not None)
            if mutation_bit is not None:
                require(any(bits & (1 << mutation_bit) for bits in run['summary']['observed_error_bits']),
                        'Mutation failed without exercising its intended assertion')
        record['complete'] = True
    except BaseException as error:
        record['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        record['source_inputs_after'] = {name: identity(HERE / name) for name in inputs}
        record['protected_inputs_unchanged'] = record['source_inputs_after'] == inputs
        record['raw_files'] = [identity(path) for path in sorted(result_dir.iterdir()) if path.is_file()]
        with (result_dir / 'execution.json').open('x') as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        require(record['protected_inputs_unchanged'], 'Fixture sources changed during execution')
    print(json.dumps(dict(complete=True, result=identity(result_dir / 'execution.json'),
                         runs=[dict(name=row['name'], cases=row['summary']['cases'],
                                    failures=row['summary']['failing_cases']) for row in record['runs']],
                         performance_eligible=False), indent=2))


if __name__ == '__main__':
    main()
