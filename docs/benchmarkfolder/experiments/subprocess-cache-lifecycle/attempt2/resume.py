#!/usr/bin/env python3
"""Append-only shell-binding correction; reuse the frozen generated sources.

Attempt1 stopped at the first make query because /bin/bash does not exist in
the pinned Nix environment. No subprocess/driver compiler or native worker ran.
This wrapper changes only the report directory and the make command-line SHELL
binding. All original protocol, source/package checks, deadlines, cleanup,
objects, worker inputs and numerical lock remain in the frozen qualify.py.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
ORIGINAL = ROOT/'docs/benchmarkfolder/experiments/subprocess-cache-lifecycle'
REPORT = ORIGINAL/'attempt2'
INDEX = REPORT/'resume-freeze.json'
ORIGINAL_FREEZE = '2f971799fa749a7591736967da1e119655670b21fd304b01e452d601bb131c50'
GENERATED_FREEZE = 'c1da2ccf47ad5f8c654422f8eb56a140c93bf42c0655b13bcf7334ad6ebfe33e'
LINKS = ['source-freeze.json', 'generated-source-freeze.json', 'materialized-source.json',
         'generation.json', 'cli-bootstrap-provenance.json', 'DESIGN.md', 'generate.mg5.in',
         'driver.f.in', 'observer.c', 'analyze.py', 'qualify.py', 'test_analyze.py']
PRIOR_FAILURES = ['execution/link-info.json', 'execution/link-info.txt']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_module():
    original = json.loads((ORIGINAL/'source-freeze.json').read_text())
    if digest(ORIGINAL/'source-freeze.json') != ORIGINAL_FREEZE:
        raise RuntimeError('Original source freeze changed')
    if digest(ORIGINAL/'generated-source-freeze.json') != GENERATED_FREEZE:
        raise RuntimeError('Generated source freeze changed')
    for name, expected in original['protocol'].items():
        if digest(ORIGINAL/name) != expected:
            raise RuntimeError('Original protocol changed: '+name)
    spec = importlib.util.spec_from_file_location('frozen_lifecycle_qualifier', ORIGINAL/'qualify.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.frozen()
    return module


def bootstrap_check():
    record = json.loads((ORIGINAL/'cli-bootstrap-provenance.json').read_text())
    for name, item in record['files'].items():
        if digest(ROOT/name) != item['sha256']:
            raise RuntimeError('Retained CLI bootstrap artifact changed: '+name)


def freeze():
    source_module()
    bootstrap_check()
    found = shutil.which('bash')
    if found is None:
        raise RuntimeError('Pinned development environment does not provide Bash')
    shell = Path(found).resolve()
    if not str(shell).startswith('/nix/store/') or not os.access(shell, os.X_OK):
        raise RuntimeError('Bash must be the executable from the pinned Nix environment')
    for name in LINKS:
        (REPORT/name).symlink_to('../'+name)
    record = {'schema_version':1, 'kind':'append-only shell-binding attempt2',
              'runner_sha256':digest(Path(__file__)),
              'shell':{'path':str(shell),'sha256':digest(shell)},
              'original_inputs':{name:digest(ORIGINAL/name) for name in LINKS},
              'input_symlinks':{name:'../'+name for name in LINKS},
              'attempt1_failure':{name:digest(ORIGINAL/name) for name in PRIOR_FAILURES},
              'cpu':10, 'native_execution_authorized':False,
              'changes':['report directory is attempt2', 'each make argv adds SHELL=<pinned absolute Bash>'],
              'preserved':'No CLI regeneration, generated-source changes, or changes to the frozen numerical protocol.'}
    with INDEX.open('x') as stream:
        json.dump(record,stream,indent=2,sort_keys=True)
        stream.write('\n')


def resume(stage):
    record = json.loads(INDEX.read_text())
    if record['runner_sha256'] != digest(Path(__file__)):
        raise RuntimeError('Attempt2 runner changed')
    shell = record['shell']['path']
    if digest(shell) != record['shell']['sha256']:
        raise RuntimeError('Pinned Bash changed')
    for name, expected in record['attempt1_failure'].items():
        if name not in PRIOR_FAILURES or digest(ORIGINAL/name) != expected:
            raise RuntimeError('Retained attempt1 failure changed: '+name)
    if set(record['attempt1_failure']) != set(PRIOR_FAILURES):
        raise RuntimeError('Incomplete attempt1 failure binding')
    for name, expected in record['original_inputs'].items():
        path = REPORT/name
        if (not path.is_symlink() or record['input_symlinks'].get(name) != '../'+name
                or os.readlink(path) != '../'+name or digest(path) != expected):
            raise RuntimeError('Attempt2 original input binding changed: '+name)
    module = source_module()
    bootstrap_check()
    module.REPORT = REPORT
    original_command = module.command

    def shell_bound_command(args, name, *positional, **keyword):
        args = list(args)
        if args and str(args[0]) == 'make':
            args.append('SHELL='+shell)
        return original_command(args, name, *positional, **keyword)

    module.command = shell_bound_command
    try:
        {'build':module.build, 'run':module.run}[stage]()
    finally:
        bootstrap_check()
        if digest(shell) != record['shell']['sha256']:
            raise RuntimeError('Pinned Bash changed during stage')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze','build','run'])
    argument = parser.parse_args()
    if Path.cwd() != ROOT:
        parser.error('All work must start from the MadGraph repository root')
    if argument.stage == 'freeze':
        freeze()
    else:
        resume(argument.stage)


if __name__ == '__main__':
    main()
