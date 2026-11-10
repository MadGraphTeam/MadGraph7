#!/usr/bin/env python3
"""Bounded pure-Fortran mock qualification; verify never launches a process."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CMS = HERE.parents[1]
ROOT = CMS.parents[1]
BASE = ROOT / '.tirust-build/qp-mass-general-implementation/combined-qualification/control.py'
BASE_SHA = '57b95030a7b70facc3c276767b4d84596edbf9b692817ef81bd6c20c211d30d3'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def save(value):
    target = HERE / 'run.json'
    temporary = HERE / 'run.json.tmp'
    with temporary.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')
    temporary.replace(target)


def inventory(directory, base):
    result = {}
    for p in directory.rglob('*'):
        base.require(not p.is_symlink(), 'Symlink in mock working directory')
        if p.is_file():
            result[str(p.relative_to(directory))] = base.sha(p)
    return result


def execute(job, spec, report, base, cleanup, env, deadline):
    directory = HERE / job['directory']
    base.require(directory.resolve().is_relative_to(HERE), 'Working directory outside mock scope')
    base.require(directory.is_dir() and not directory.is_symlink(), 'Missing/nonprivate mock working directory')
    before = inventory(directory, base)
    base.require(not before, 'Existing mock data; no retry')
    log = HERE / (job['name'] + '.log')
    base.require(not log.exists(), 'Existing process log; no retry')
    with log.open('x') as stream:
        event = cleanup.bounded(job['argv'], directory, stream,
                                min(deadline, time.monotonic()+job['seconds_max']), env)
    event.update(name=job['name'], log_file=str(log.relative_to(ROOT)), log_sha256=base.sha(log),
                 capture='merged stdout and stderr; empty stderr is not separately asserted')
    try:
        event['output_files'] = inventory(directory, base)
        event['output_closure_matches'] = set(event['output_files']) == set(job['expected_outputs'])
    except BaseException as error:
        event['output_inventory_error'] = str(error)
        event['output_closure_matches'] = False
    report['jobs'].append(event)
    save(report)
    base.require(event['returncode'] == 0 and not event['timed_out'] and not event.get('error'),
                 'Process failed; no retry: ' + job['name'])
    base.require(event['output_closure_matches'], 'Missing/unexpected mock output: '+job['name'])
    base.verify_inputs(spec)
    return event, log, directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='verify', choices=('verify', 'run'))
    action = parser.parse_args().action
    import hashlib
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('Reviewed process controller changed')
    base = load('cms_mock_reviewed_base', BASE)
    base.require(Path.cwd().resolve() == ROOT, 'Run from the MadGraph repository root')
    index = json.loads((HERE / 'source-index.json').read_text())
    for name, digest in index['files'].items():
        base.require(base.sha(HERE/name) == digest, 'Attempt2 source changed: '+name)
    spec = json.loads((HERE / 'prepared.json').read_text())
    base.require(spec['schema'] == 'cms-scalar-refusal-mock-attempt2-proposal-v1', 'Wrong attempt2 proposal')
    base.require(spec['cpu'] == 8 and spec['processes_at_once'] == 1 and spec['execution_seconds_max'] == 35,
                 'Wrong bounded mock scope')
    base.verify_inputs(spec)
    env = base.environment(spec)
    base.require(base.sha(base.OLD) == base.OLD_SHA, 'Reviewed child cleanup changed')
    if action == 'verify':
        print(json.dumps({'verified_inputs': len(spec['inputs']), 'loader_checks': len(spec['loader_checks']),
                          'mock_cases': len(spec['mock_jobs']), 'process_launched': False}))
        return
    base.require(8 in os.sched_getaffinity(0), 'CPU8 unavailable')
    base.require(not (HERE/'run.json').exists(), 'Mock qualification already attempted; no retry')
    cleanup = load('cms_mock_reviewed_cleanup', base.OLD)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    report = {'schema': 'cms-scalar-refusal-mock-attempt2-result-v1', 'state': 'running',
              'prior_failed_run_sha256': spec['prior_failed_run_sha256'],
              'source_index_sha256': base.sha(HERE/'source-index.json'),
              'target_runtime_selection': 'Explicit target PT_INTERP --library-path; outer environment unchanged',
              'prepared_sha256': base.sha(HERE/'prepared.json'), 'controller_sha256': base.sha(__file__),
              'reviewed_base_sha256': BASE_SHA, 'cleanup_sha256': base.OLD_SHA,
              'started_unix': time.time(), 'inherited_affinity': sorted(os.sched_getaffinity(0)),
              'loadavg_before': Path('/proc/loadavg').read_text().strip(), 'jobs': [],
              'automatic_retry': False, 'shared_reducer_lock_required': False,
              'runtime_environment': {k: env.get(k) for k in ('LD_LIBRARY_PATH', 'OMP_NUM_THREADS',
                  'OPENBLAS_NUM_THREADS', 'RAYON_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
              'physical_accuracy_qualified': False, 'full_madloop_host_rescue_qualified': False,
              'hardware_ieee_flags_traps_rounding_qualified': False}
    with (HERE/'run.json').open('x') as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write('\n')
    deadline = time.monotonic()+spec['execution_seconds_max']
    try:
        with cleanup.signal_cleanup():
            for check in spec['loader_checks']:
                base.verify_inputs(spec)
                base.require(base.elf_interpreter(ROOT/check['executable']) == check['loader'],
                             'ELF interpreter changed')
                event, log, directory = execute(check['job'], spec, report, base, cleanup, env, deadline)
                loaded = cleanup.loaded_libraries(log.read_text(), directory)
                base.verify_runtime(loaded, check)
                event['resolved_runtime'] = loaded
                save(report)
            validations = {}
            for job in spec['mock_jobs']:
                base.verify_inputs(spec)
                event, log, directory = execute(job, spec, report, base, cleanup, env, deadline)
                if job['validator'] == 'classifier':
                    validator = load('cms_classifier_mock_validator', ROOT/spec['classifier_validator'])
                    result = validator.validate_text(log.read_text())
                elif job['validator'] == 'caller':
                    base.require(not log.read_text().strip(), 'Unexpected caller stdout/stderr')
                    validator = load('cms_caller_mock_validator', ROOT/spec['caller_validator'])
                    result = validator.validate_text((directory/'caller-mock.log').read_text(),
                        json.loads((ROOT/spec['caller_cases']).read_text()))
                else:
                    raise ValueError('Unknown mock validator')
                validations[job['validator']] = result
                report['validations'] = validations
                save(report)
            base.require(set(validations) == {'classifier', 'caller'}, 'Incomplete mock qualification')
            cleanup.check_interrupt()
            base.require(time.monotonic() <= deadline, 'Inclusive mock qualification deadline exceeded')
        report['state'] = 'complete'
    except BaseException as error:
        report['state'] = 'failed'
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        try:
            base.verify_inputs(spec)
            report['protected_inputs_after'] = 'unchanged'
        except BaseException as error:
            report['state'] = 'failed'
            report['protected_inputs_after'] = 'FAILED: '+str(error)
        report['finished_unix'] = time.time()
        report['loadavg_after'] = Path('/proc/loadavg').read_text().strip()
        save(report)
    print(json.dumps({'state': report['state'], 'processes': len(report['jobs']),
                      'error': report.get('error')}, sort_keys=True))
    sys.exit(0 if report['state'] == 'complete' else 1)


if __name__ == '__main__':
    main()
