#!/usr/bin/env python3
"""Verify the private CMS source package or perform its one bounded build.

This controller has no native/mock execution action. Build remains subject to
the parent's explicit admission; source presence is not authorization.
"""
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
ROOT = HERE.parents[1]
BASE = ROOT / '.tirust-build/qp-mass-general-implementation/combined-qualification/control.py'
BASE_SHA = '57b95030a7b70facc3c276767b4d84596edbf9b692817ef81bd6c20c211d30d3'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'build'))
    args = parser.parse_args()
    import hashlib
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
        raise ValueError('Previously reviewed build controller changed')
    base = load('cms_reviewed_build_controller', BASE)
    base.HERE = HERE
    base.require(Path.cwd().resolve() == ROOT, 'Run from the MadGraph repository root')
    base.require(base.sha(base.OLD) == base.OLD_SHA, 'Previously reviewed owned-child cleanup changed')
    cleanup = load('cms_reviewed_owned_child_cleanup', base.OLD)
    spec = json.loads((HERE / 'prepared.json').read_text())
    base.require(spec['schema'] == 'cms-scalar-refusal-build-proposal-v1', 'Wrong private build proposal')
    base.require(spec['cpu'] == 8 and spec['jobs'] == 1 and spec['build_seconds_max'] == 180,
                 'Wrong bounded build scope')
    base.verify_inputs(spec)
    base.environment(spec)
    if args.action == 'verify':
        print(json.dumps({'verified_inputs': len(spec['inputs']), 'build_commands': len(spec['build_jobs']),
                          'compiler_launched': False, 'native_launched': False}))
        return
    base.require(8 in os.sched_getaffinity(0), 'CPU8 unavailable')
    base.require(not (HERE / 'build.json').exists(), 'Build already attempted; no retry')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    report = {'schema': 'cms-scalar-refusal-build-result-v1', 'state': 'running', 'action': 'build',
              'manifest_sha256': base.sha(HERE / 'prepared.json'),
              'controller_sha256': base.sha(__file__), 'reused_controller_sha256': BASE_SHA,
              'cleanup_sha256': base.OLD_SHA, 'jobs': [], 'automatic_retry': False,
              'started_unix': time.time(), 'inherited_affinity': sorted(os.sched_getaffinity(0)),
              'loadavg_before': Path('/proc/loadavg').read_text().strip(),
              'mock_or_analytic_execution': False}
    base.save('build.json', report)
    try:
        with cleanup.signal_cleanup():
            base.build(spec, report, cleanup)
        report['state'] = 'complete'
    except BaseException as error:
        report['state'] = 'failed'
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        try:
            base.verify_inputs(spec)
            report['input_hashes_after'] = 'unchanged'
        except BaseException as error:
            report['state'] = 'failed'
            report['input_hashes_after'] = 'FAILED: ' + str(error)
        report['finished_unix'] = time.time()
        report['loadavg_after'] = Path('/proc/loadavg').read_text().strip()
        base.save('build.json', report)
    print(json.dumps({'state': report['state'], 'jobs': len(report['jobs']),
                      'error': report.get('error')}, sort_keys=True))
    sys.exit(0 if report['state'] == 'complete' else 1)


if __name__ == '__main__':
    main()
