#!/usr/bin/env python3
"""Append-only loader closure correction; reuse the frozen scientific controller."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
ROOT = PARENT.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(name, value):
    target = HERE/name
    temporary = target.with_suffix(target.suffix+'.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
    temporary.replace(target)


def closure(check, elf, base):
    allowed = {Path(name).name: info for name, info in check['expected'].items()}
    executable = ROOT/check['executable']
    declared = elf.inspect(executable)
    base.require(declared['interpreter'] == check['loader'], 'Changed ELF interpreter')
    pending = declared['needed'] + [Path(check['loader']).name]
    result = {}
    while pending:
        name = pending.pop()
        if name in result:
            continue
        base.require(name in allowed, 'Unbound ELF dependency: '+name)
        info = allowed[name]
        base.require(sha(info['path']) == info['sha256'], 'Changed runtime object: '+name)
        result[name] = info
        pending.extend(elf.inspect(info['path'])['needed'])
    return {'declarations': declared, 'closure': result}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('verify', 'run'))
    action = cli.parse_args().action
    assert Path.cwd().resolve() == ROOT
    correction = json.loads((HERE/'prepared.json').read_text())
    for name, digest in correction['inputs'].items():
        assert sha(ROOT/name) == digest, name
    base = load('mass_loader_original_controller', PARENT/'control.py')
    elf = load('mass_loader_elf_reader', ROOT/correction['elf_reader'])
    spec = json.loads((PARENT/'prepared.json').read_text())
    original = copy.deepcopy(spec)
    derived = {check['job']['name']: closure(check, elf, base) for check in spec['runtime_checks']}
    base.require(derived == correction['elf_closures'], 'ELF closure differs from source freeze')
    fixture = spec['runtime_checks'][0]
    base.require(fixture['job']['name'] == 'loader-fixture', 'Wrong fixture binding')
    fixture['expected'] = correction['fixture_expected']
    fixture['allow_system_subset'] = False
    fixture.pop('required')
    fixture['job']['log_file'] = correction['fixture_loader_log']
    proof = copy.deepcopy(spec)
    proof['runtime_checks'][0] = original['runtime_checks'][0]
    base.require(proof == original, 'Scientific scope changed')
    base.require(set(fixture['expected']) == {'libgfortran.so.5', 'libm.so.6', 'libgcc_s.so.1',
                                            'libc.so.6', Path(fixture['loader']).name},
                 'Fixture closure is not exactly five pinned runtime objects')
    verify_original = base.verify_inputs

    def verify_all(value):
        verify_original(value)
        for name, digest in correction['inputs'].items():
            base.require(sha(ROOT/name) == digest, 'Changed correction/failure input: '+name)

    base.verify_inputs = verify_all
    base.save = save
    base.verify_inputs(spec)
    base.environment(spec)
    base.executable_outputs(spec)
    base.require(base.sha(base.OLD) == base.OLD_SHA, 'Reviewed cleanup changed')
    if action == 'verify':
        print(json.dumps({'verified_original_inputs': len(spec['inputs']),
                          'correction_inputs': len(correction['inputs']), 'elf_closures': len(derived),
                          'scientific_scope_unchanged': True, 'process_launched': False}))
        return
    base.require(8 in os.sched_getaffinity(0), 'CPU8 unavailable')
    base.require(not (HERE/'run.json').exists(), 'Corrected attempt already exists; no retry')
    cleanup = load('mass_loader_reviewed_cleanup', base.OLD)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    report = {'state': 'running', 'action': 'run', 'manifest_sha256': sha(PARENT/'prepared.json'),
              'correction_manifest_sha256': sha(HERE/'prepared.json'),
              'controller_sha256': sha(__file__), 'original_controller_sha256': sha(PARENT/'control.py'),
              'cleanup_sha256': base.OLD_SHA, 'jobs': [], 'started_unix': time.time(),
              'inherited_affinity': sorted(os.sched_getaffinity(0)),
              'loadavg_before': Path('/proc/loadavg').read_text().strip(), 'automatic_retry': False,
              'scientific_scope_unchanged': True, 'prior_failed_run_sha256': sha(PARENT/'run.json')}
    save('run.json', report)
    try:
        with cleanup.signal_cleanup():
            base.run(spec, report, cleanup)
        report['state'] = 'complete'
    except BaseException as error:
        report['state'] = 'failed'
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        try:
            base.verify_inputs(spec)
            base.executable_outputs(spec)
            report['input_hashes_after'] = 'unchanged'
        except BaseException as error:
            report['state'] = 'failed'
            report['input_hashes_after'] = 'FAILED: '+str(error)
        report['finished_unix'] = time.time()
        report['loadavg_after'] = Path('/proc/loadavg').read_text().strip()
        save('run.json', report)
    print(json.dumps({'state': report['state'], 'jobs': len(report['jobs']),
                      'error': report.get('error')}, sort_keys=True))
    sys.exit(0 if report['state'] == 'complete' else 1)


if __name__ == '__main__':
    main()
