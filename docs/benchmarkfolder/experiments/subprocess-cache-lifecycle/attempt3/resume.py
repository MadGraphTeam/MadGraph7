#!/usr/bin/env python3
"""Bind the exact calibrated linker-input closure before fresh attempt3 links.

The previous plain link revealed a Nix compiler-selected libgcc_s linker script
outside compiler -print-file-name results. Its map is calibration evidence,
not numerical qualification. No numerical worker has run. This attempt adds
only the frozen concrete input closure and make's -W lifecycle_driver.o to
force fresh links. It inherits attempt2's absolute Bash binding and all
original object/source/runtime/cache/IEEE checks and execution limits.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
BASE = ROOT/'docs/benchmarkfolder/experiments/subprocess-cache-lifecycle'
PRIOR = BASE/'attempt2'
REPORT = BASE/'attempt3'
INDEX = REPORT/'resume-freeze.json'
PRIOR_INDEX = '59933ea215e6f93199655d039ec206667a5f9e43c73a766bca7a8ae9d14ce381'
PRIOR_RUNNER = 'eff4e1e890e37806c1c552c62450f484bda89993cdea1b19fe17b461c4c89113'
MAP_HASH = '62f2e2b80445f1299d6cbe8eed44c3265480b4ce058247d12dce267cb80e0171'
PLAIN_HASH = '7b9ea41996aacd066f25b2eee8f60546c52e70d58c818fc24592aa7e2116e35e'
SCRIPT = Path('/nix/store/b7kx9bkjsma9wslr1cg0316m5jy450l8-gcc-15.2.0-libgcc/lib/libgcc_s.so')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prior_module():
    if digest(PRIOR/'resume.py') != PRIOR_RUNNER or digest(PRIOR/'resume-freeze.json') != PRIOR_INDEX:
        raise RuntimeError('Attempt2 runner or index changed')
    spec = importlib.util.spec_from_file_location('frozen_attempt2', PRIOR/'resume.py')
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)
    return prior


def calibrated_inputs(module):
    compressed = PRIOR/'execution/lifecycle_plain.map.gz'
    raw = gzip.decompress(compressed.read_bytes())
    if hashlib.sha256(raw).hexdigest() != MAP_HASH:
        raise RuntimeError('Attempt2 calibration map changed')
    retained = module.WORK/'attempt2-retained'
    if digest(retained/'lifecycle_plain.map') != MAP_HASH or digest(retained/'lifecycle_plain') != PLAIN_HASH:
        raise RuntimeError('Retained failed-attempt artifacts changed')
    entries = {}
    for name in re.findall(r'^LOAD (.+)$', raw.decode().split('\x00')[0], re.M):
        path = Path(name)
        if not path.is_absolute():
            path = module.OUTPUT/'SubProcesses'/path
        if not path.is_file():
            raise RuntimeError('Unresolved actual LOAD input: '+str(path))
        entries[str(path.resolve())] = digest(path)
    # This observer object was already compiled in attempt2 but not linked.
    observer = module.OUTPUT/'SubProcesses/observer_observed.o'
    entries[str(observer.resolve())] = digest(observer)
    if str(SCRIPT) not in entries:
        raise RuntimeError('Expected libgcc_s linker script absent from actual LOAD inputs')
    return entries


def freeze():
    prior = prior_module()
    module = prior.source_module()
    prior.bootstrap_check()
    entries = calibrated_inputs(module)
    recipe = (PRIOR/'execution/link-plain.txt').read_text().splitlines()
    lines = [line for line in recipe if ' -o lifecycle_plain lifecycle_driver.o ' in line]
    if len(lines) != 1:
        raise RuntimeError('Cannot identify the unique actual plain link command')
    argv = shlex.split(lines[0])
    argv.insert(1,'-###')
    module.REPORT = REPORT
    # -### prints compiler/link commands without executing them.
    module.command(argv,'compiler-link-search',module.OUTPUT/'SubProcesses',30)
    module.verify_hashes(entries)
    for name in prior.LINKS:
        (REPORT/name).symlink_to('../'+name)
    evidence = [PRIOR/'resume.py',PRIOR/'resume-freeze.json',PRIOR/'failure.json',
                PRIOR/'execution/link-plain.txt',PRIOR/'execution/link-plain.json',
                PRIOR/'execution/lifecycle_plain.map.gz',PRIOR/'execution/lifecycle_plain.map-gzip.json']
    record = {'schema_version':1,'kind':'exact calibrated linker-input closure before fresh links',
        'runner_sha256':digest(Path(__file__)),
        'prior_evidence':{str(path):digest(path) for path in evidence},
        'input_symlinks':{name:'../'+name for name in prior.LINKS},
        'link_inputs_before':entries,'calibration_map_raw_sha256':MAP_HASH,
        'retained_plain_sha256':PLAIN_HASH,
        'libgcc_script':{'path':str(SCRIPT),'sha256':digest(SCRIPT),'text':SCRIPT.read_text(),
                        'actual_load_libgcc_inputs':{path:value for path,value in entries.items() if 'libgcc' in path}},
        'search_evidence':{name:digest(REPORT/'execution'/name) for name in
                           ('compiler-link-search.txt','compiler-link-search.json')},
        'fresh_link_rule':'Add make -W lifecycle_driver.o only to link-plain and link-observed commands; never replace object bytes.',
        'cpu':10,'native_execution_authorized':False,
        'scope':'No broad unbound-input exemption; exact frozen closure supplements original compiler discovery. No numerical worker has run.'}
    with INDEX.open('x') as stream:
        json.dump(record,stream,indent=2,sort_keys=True)
        stream.write('\n')


def resume(stage):
    record = json.loads(INDEX.read_text())
    if digest(Path(__file__)) != record['runner_sha256']:
        raise RuntimeError('Attempt3 runner changed')
    for name, expected in record['prior_evidence'].items():
        if digest(name) != expected:
            raise RuntimeError('Prior failed-attempt evidence changed: '+name)
    for name, target in record['input_symlinks'].items():
        path = REPORT/name
        if target != '../'+name or not path.is_symlink() or os.readlink(path) != target:
            raise RuntimeError('Attempt3 input symlink changed: '+name)
    for name, expected in record['search_evidence'].items():
        if digest(REPORT/'execution'/name) != expected:
            raise RuntimeError('Compiler search evidence changed')
    prior = prior_module()
    original_source_module = prior.source_module

    def source_module_with_closure():
        module = original_source_module()
        if calibrated_inputs(module) != record['link_inputs_before']:
            raise RuntimeError('Concrete calibrated input closure changed')
        original_external_inputs = module.external_inputs
        original_command = module.command
        bound_inputs = {}

        def external_inputs(directory, expanded, compiler, deadline):
            values = original_external_inputs(directory,expanded,compiler,deadline)
            module.verify_hashes(record['link_inputs_before'])
            values.update(record['link_inputs_before'])
            bound_inputs.clear()
            bound_inputs.update(values)
            return values

        def fresh_link_command(args,name,*positional,**keyword):
            args = list(args)
            if name in ('link-plain','link-observed'):
                if not args or args[0] != 'make':
                    raise RuntimeError('Unexpected link command')
                module.verify_hashes(record['link_inputs_before'])
                args += ['-W','lifecycle_driver.o']
            result = original_command(args,name,*positional,**keyword)
            if name in ('link-plain','link-observed'):
                module.verify_hashes(bound_inputs)
                directory = module.OUTPUT/'SubProcesses'
                map_path = directory/('lifecycle_'+name.removeprefix('link-')+'.map')
                for loaded in re.findall(r'^LOAD (.+)$',map_path.read_text(),re.M):
                    path = Path(loaded)
                    if not path.is_absolute():
                        path = directory/path
                    resolved = str(path.resolve())
                    if not path.is_file() or resolved not in bound_inputs or digest(path) != bound_inputs[resolved]:
                        raise RuntimeError('Actual link input was not bound before this link: '+loaded)
            return result

        module.external_inputs = external_inputs
        module.command = fresh_link_command
        return module

    prior.source_module = source_module_with_closure
    prior.REPORT = REPORT
    # Keep prior.INDEX at the immutable attempt2 freeze; its Bash and original
    # source checks remain active, while prior.REPORT routes all new evidence.
    prior.resume(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','build','run'])
    argument = parser.parse_args()
    if Path.cwd() != ROOT:
        parser.error('All work must start from the MadGraph repository root')
    if argument.stage == 'freeze':
        freeze()
    else:
        resume(argument.stage)


if __name__ == '__main__':
    main()
