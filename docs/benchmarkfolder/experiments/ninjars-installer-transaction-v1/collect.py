#!/usr/bin/env python3
"""Engine-free red/green installer proof; run only after the quiet handoff."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(suite, log):
    with log.open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return {'tests':result.testsRun, 'failures':len(result.failures),
            'errors':len(result.errors), 'skipped':len(result.skipped),
            'passed':result.wasSuccessful(), 'log_sha256':sha(log)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=int, required=True)
    parser.add_argument('--quiet-released', required=True,
                        help='Explicit root handoff identifying the released window')
    args = parser.parse_args()
    assert Path.cwd() == ROOT and args.quiet_released.strip()
    assert args.cpu in os.sched_getaffinity(0)
    os.sched_setaffinity(0,{args.cpu})
    sys.path.insert(0,str(ROOT))
    sys.dont_write_bytecode = True
    os.environ['TMPDIR'] = str(ROOT/'.ninjars-build/tmp')
    source_names = ['madgraph/various/ninjars_install.py',
                    'tests/unit_tests/various/test_ninjars_install.py']
    before = {name:sha(ROOT/name) for name in source_names}
    original_path = HERE/'original/ninjars_install.py'
    assert sha(original_path) == 'aabedacfb872d3d3527dab19f162fc2733efc9fe05b52aec42d4b2d4145c06de'
    spec = importlib.util.spec_from_file_location('ninjars_installer_before_transaction',original_path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    from tests.unit_tests.various import test_ninjars_install as tests
    red_names = ['test_alias_publication_failure_rolls_back_previous_install_and_retry',
                 'test_common_lib_mkdir_failure_restores_previous_install_and_retry',
                 'test_post_commit_cleanup_failure_keeps_success_and_recovery_diagnostic']
    # Swap only this test module's imported installer object. No live source
    # file is replaced, and the final module remains the normal MG7 import.
    with mock.patch.object(tests,'ninjars_install',original):
        red = run(unittest.TestSuite(tests.TestNinjaRSInstaller(name) for name in red_names),
                  HERE/'original-failure-tests.log')
    assert (red['tests'],red['failures'],red['errors'],red['passed']) == (3,2,1,False),red
    modules = ['tests.unit_tests.various.test_ninjars_install',
               'tests.unit_tests.various.test_ninjars_statistics',
               'tests.unit_tests.various.test_ninjars_integration_boundaries',
               'tests.unit_tests.various.test_ninjars_upstream']
    green = run(unittest.defaultTestLoader.loadTestsFromNames(modules), HERE/'focused-tests.log')
    assert green['passed'] and green['tests'] == 37 and not green['skipped'],green
    assert before == {name:sha(ROOT/name) for name in source_names}
    final = HERE/'final'
    final.mkdir()
    for name in source_names:
        shutil.copy2(ROOT/name,final/Path(name).name)
    result = {'schema':'ninjars-installer-transaction-validation-v1', 'passed':True,
        'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'quiet_released':args.quiet_released,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'collector_sha256':sha(Path(__file__)), 'original_source_sha256':sha(original_path),
        'original_failure_tests':red_names,'original_red':red,'final_green':green,
        'source_files':before,'test_modules':modules,'compiler_calls':0,'numerical_engine_calls':0,
        'old_failure_reproduction':'Private original module is injected only into the test module; production files are not reverted.',
        'new_regression_tests':7,
        'coverage':['alias publication rollback and successful retry','common directory creation failure and retry',
            'first-install rollback with no previous destination','exact custom and dangling symlink restoration',
            'regular user files and hidden sentinels preserved','symlinked common library directory',
            'post-commit cleanup failure remains successful with diagnostic',
            'rollback failure retains recovery files and clear error','KeyboardInterrupt rollback']}
    with (HERE/'validation.json').open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
