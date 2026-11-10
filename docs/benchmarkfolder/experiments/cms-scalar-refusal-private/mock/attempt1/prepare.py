#!/usr/bin/env python3
"""Freeze a two-mock runtime proposal using ELF reads only; no process launch."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CMS = HERE.parent
ROOT = CMS.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def main():
    assert Path.cwd().resolve() == ROOT
    assert not (HERE/'prepared.json').exists()
    assert not (HERE/'source-index.json').exists()
    build = json.loads((CMS/'build.json').read_text())
    original = json.loads((CMS/'prepared.json').read_text())
    environment = json.loads((ROOT/'.tirust-build/qp-mass-general-implementation/combined-qualification/prepared.json').read_text())
    assert sha(CMS/'build.json') == 'f098a042744800a934b41b390dd2a3ff191670b3bc04701cbcb19c311508ff10'
    assert build['state'] == 'complete' and build['input_hashes_after'] == 'unchanged'
    assert len(build['jobs']) == 17
    module_spec = importlib.util.spec_from_file_location('cms_mock_elf_reader', HERE/'elf.py')
    elf = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(elf)
    prior_runtime = environment['runtime_checks'][0]
    loader = prior_runtime['loader']
    allowed_names = {'libgfortran.so.5', 'libm.so.6', 'libgcc_s.so.1', 'libc.so.6', 'libquadmath.so.0'}
    allowed = {name: info for name, info in prior_runtime['expected'].items() if name in allowed_names}
    allowed[Path(loader).name] = prior_runtime['expected'][loader]
    inputs = dict(original['inputs'])
    inputs.update(build['outputs'])
    inputs.update(build['module_outputs'])
    for name in ('source-index.json', 'prepared.json', 'build.json', 'source-build-evidence/manifest.json'):
        inputs[relative(CMS/name)] = sha(CMS/name)
    for directory in (HERE, CMS/'classifier/mock-qualification'):
        for path in sorted(directory.rglob('*')):
            assert not path.is_symlink()
            if path.is_file():
                inputs[relative(path)] = sha(path)
    for name in ('caller/validate_fixture.py', 'caller/fixture/cases.json'):
        inputs[relative(CMS/name)] = sha(CMS/name)
    checks, mock_jobs, declarations, runtime_objects = [], [], {}, {}
    for kind, executable, seconds, outputs in (
        ('classifier', CMS/'classifier/build/classifier_mock', 15, []),
        ('caller', CMS/'caller/fixture/build/caller_mock', 10, ['caller-mock.log'])):
        assert sha(executable) == build['outputs'][relative(executable)]
        declaration = elf.inspect(executable)
        assert declaration['interpreter'] == loader
        declarations[kind] = declaration
        queue = list(declaration['needed']) + [Path(loader).name]
        closure = {}
        while queue:
            name = queue.pop()
            if name in closure:
                continue
            assert name in allowed, ('Unbound ELF dependency', name)
            info = allowed[name]
            assert sha(info['path']) == info['sha256']
            closure[name] = info
            queue.extend(elf.inspect(info['path'])['needed'])
        runtime_objects.update(closure)
        for suffix in ('loader', 'run'):
            directory = HERE/(kind+'-'+suffix)
            directory.mkdir()
            assert not any(directory.iterdir())
        checks.append({'executable': relative(executable), 'loader': loader, 'expected': closure,
                       'allow_system_subset': False,
                       'job': {'name': kind+'-loader', 'directory': kind+'-loader',
                               'argv': [loader, '--list', str(executable)],
                               'seconds_max': 5, 'expected_outputs': []}})
        mock_jobs.append({'name': kind+'-mock', 'directory': kind+'-run', 'argv': [str(executable)],
                          'seconds_max': seconds, 'expected_outputs': outputs, 'validator': kind})
    system = dict(original['system_files'])
    system.update({info['path']: info['sha256'] for info in runtime_objects.values()})
    for name, digest in inputs.items():
        assert sha(ROOT/name) == digest, name
    cases = json.loads((CMS/'caller/fixture/cases.json').read_text())
    assert len(cases['cases']) == 91 and len(cases['models']) == 2
    prepared = {'schema': 'cms-scalar-refusal-mock-proposal-v1', 'cpu': 8, 'processes_at_once': 1,
                'execution_seconds_max': 35, 'loader_checks_seconds_max': 10,
                'classifier_seconds_max': 15, 'caller_seconds_max': 10, 'automatic_retry': False,
                'compiler_environment': original['compiler_environment'], 'taskset': original['taskset'],
                'inputs': dict(sorted(inputs.items())), 'system_files': system,
                'runtime_directories': sorted({str(Path(info['path']).parent) for info in runtime_objects.values()}),
                'elf_declarations': declarations, 'loader_checks': checks, 'mock_jobs': mock_jobs,
                'classifier_validator': relative(CMS/'classifier/mock-qualification/validate.py'),
                'caller_validator': relative(CMS/'caller/validate_fixture.py'),
                'caller_cases': relative(CMS/'caller/fixture/cases.json'),
                'classifier_expected_records': ['CLASSIFIER_MOCK_COMPLETE 8 8',
                    'CLASSIFIER_MOCK_COMPLETE 16 8', 'COMPLETE_CLASSIFIER_MOCK_ONLY'],
                'caller_logical_cases': 91, 'caller_mass_variants': 2, 'caller_primary_cases': 182,
                'caller_total_reduction_calls': 546, 'caller_expected_journal_rows': len(cases['expected_journal_rows']),
                'shared_reducer_lock_required': False,
                'ieee_scope': 'Finiteness/NaN predicates, exact buffer bits and MadLoop refusal latches only; sticky flags, traps and rounding transparency are not qualified',
                'scope': 'Only recorded pure-Fortran mocks; no AVH analytic evaluation, Rust, real scalar or backend8/9 execution'}
    (HERE/'prepared.json').write_text(json.dumps(prepared, sort_keys=True, indent=2)+'\n')
    files = {str(path.relative_to(HERE)): sha(path) for path in sorted(HERE.rglob('*')) if path.is_file()}
    index = {'schema': 'cms-scalar-refusal-mock-source-index-v1', 'files': files,
             'prepared_sha256': sha(HERE/'prepared.json'), 'parent_build_sha256': sha(CMS/'build.json'),
             'process_launched': False}
    (HERE/'source-index.json').write_text(json.dumps(index, sort_keys=True, indent=2)+'\n')
    print(json.dumps({'prepared_sha256': sha(HERE/'prepared.json'),
                      'source_index_sha256': sha(HERE/'source-index.json'),
                      'inputs': len(inputs), 'system_files': len(system), 'process_launched': False}))


if __name__ == '__main__':
    main()
