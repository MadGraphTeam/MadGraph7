#!/usr/bin/env python3
"""Freeze existing generated sources and compiler recipes; never execute them."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENV_PARENT = ROOT / '.tirust-build/qp-mass-general-implementation/combined-qualification/prepared.json'
ENV_SHA = '906cc984cc59068b89823ef7108f3f80f28e373e2ee8a15dfe9df7abb86c2a1e'
MODULE = re.compile(r'^\s*module\s+(?!procedure\b)([A-Za-z][A-Za-z0-9_]*)\s*(?:!.*)?$', re.M | re.I)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def resolve(value):
    p = Path(value)
    p = p if p.is_absolute() else ROOT / p
    p = p.resolve()
    assert p.is_relative_to(ROOT), p
    return p


def write_new(name, value):
    with (HERE / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def main():
    assert Path.cwd().resolve() == ROOT
    assert not (HERE / 'prepared.json').exists()
    assert not (HERE / 'source-index.json').exists()
    assert sha(ENV_PARENT) == ENV_SHA
    environment = json.loads(ENV_PARENT.read_text())
    plans = [(HERE / name, json.loads((HERE / name).read_text())) for name in
             ('classifier/compile-plan.json', 'caller/compile-plan.json')]
    jobs = []
    for _, plan in plans:
        jobs.extend(plan['jobs'])
    assert jobs and len({j['name'] for j in jobs}) == len(jobs)
    outputs, modules, search_dirs = set(), set(), {ROOT}
    for job in jobs:
        argv = job['argv']
        assert argv[0] == environment['compiler'] and argv.count('-o') == 1
        output = resolve(argv[argv.index('-o') + 1])
        assert output.is_relative_to(HERE) and not output.exists()
        outputs.add(rel(output))
        output.parent.mkdir(parents=True, exist_ok=True)
        cwd = resolve(job['cwd'])
        assert cwd == ROOT or cwd.is_relative_to(HERE)
        cwd.mkdir(parents=True, exist_ok=True)
        search_dirs.add(cwd)
        jdirs = [resolve(a[2:]) for a in argv if a.startswith('-J')]
        for directory in jdirs:
            assert directory.is_relative_to(HERE)
            directory.mkdir(parents=True, exist_ok=True)
        search_dirs.update(resolve(a[2:]) for a in argv if a.startswith('-I'))
        if '-c' in argv:
            source = resolve(argv[argv.index('-c') + 1])
            assert source.is_relative_to(HERE) and source.is_file()
            names = MODULE.findall(source.read_text())
            assert not names or len(jdirs) == 1
            modules.update(rel(jdirs[0] / (n.lower() + '.mod')) for n in names)
    assert len(outputs) == len(jobs), 'Every command must have a unique output'
    assert not any(resolve(name).exists() for name in modules)
    inputs = {}
    for p in sorted(HERE.rglob('*')):
        assert not p.is_symlink(), p
        if p.is_file():
            assert p.suffix.lower() not in ('.o', '.a', '.so'), p
            inputs[rel(p)] = sha(p)
    parents = json.loads((HERE / 'parents.json').read_text())['parent_files']
    inputs.update(parents)
    inputs.update(json.loads((HERE / 'retained-inputs/point20-inputs.json').read_text())['source_files'])
    inputs[rel(ENV_PARENT)] = ENV_SHA
    for name in ('.tirust-build/qp-mass-general-implementation/combined-qualification/control.py',
                 'docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/attempt2/control.py',
                 'tools/tirust_env.sh'):
        inputs[name] = sha(ROOT / name)
    for directory in search_dirs:
        assert directory.is_dir(), directory
        for p in sorted(directory.iterdir()):
            if p.is_file() and p.suffix.lower() in ('.mod', '.smod', '.inc', '.h'):
                inputs[rel(p)] = sha(p)
    for name, expected in inputs.items():
        assert sha(ROOT / name) == expected, name
    spec = {'schema': 'cms-scalar-refusal-build-proposal-v1',
            'state': 'source_prepared_not_built', 'cpu': 8, 'jobs': 1,
            'build_seconds_max': 180, 'automatic_retry': False,
            'compiler': environment['compiler'], 'compiler_environment': environment['compiler_environment'],
            'system_files': environment['system_files'], 'taskset': environment['taskset'],
            'inputs': dict(sorted(inputs.items())), 'build_jobs': jobs,
            'build_outputs': sorted(outputs), 'module_outputs': sorted(modules),
            'component_plans': {rel(p): sha(p) for p, _ in plans},
            'scope': 'Compiler/typechecking and mock executable construction only; no loader, mock, scalar, amplitude or Rust execution',
            'native_execution': 'Not part of this controller; component plans describe unexecuted bounded mock proposals'}
    write_new('prepared.json', spec)
    package = {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
    index = {'schema': 'cms-scalar-refusal-source-index-v1', 'state': 'source_only',
             'package_files': package, 'prepared_sha256': sha(HERE / 'prepared.json'),
             'compiler_commands': len(jobs), 'compiler_cap_seconds': 180, 'compiler_launched': False,
             'native_launched': False, 'production_sources_unchanged': parents,
             'scope': 'Private additive classifier refusal, checked caller prototype and unexecuted Fortran mocks'}
    write_new('source-index.json', index)
    print(json.dumps({'files': len(package), 'bound_inputs': len(inputs), 'build_commands': len(jobs),
                      'source_index_sha256': sha(HERE / 'source-index.json'),
                      'prepared_sha256': sha(HERE / 'prepared.json')}, sort_keys=True))


if __name__ == '__main__':
    main()
