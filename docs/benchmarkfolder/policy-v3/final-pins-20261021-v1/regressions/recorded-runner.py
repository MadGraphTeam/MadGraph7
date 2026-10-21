"""Run actual repository software suites without loading a reduction engine."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path.cwd().resolve()
OUT = ROOT / '.benchmark-work/final-dependency-regression-20261021-v1'
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
MODULES = {
    'focused': [
        'tests.unit_tests.loop.test_ninjars_fortran',
        'tests.unit_tests.various.test_ninjars_install',
        'tests.unit_tests.various.test_ninjars_statistics',
        'tests.unit_tests.various.test_ninjars_integration_boundaries',
        'tests.unit_tests.various.test_ninjars_upstream',
    ],
    'exporters': ['tests.unit_tests.loop.test_loop_exporters'],
    'fks': ['tests.unit_tests.iolibs.test_export_fks'],
    'polynomial': [
        'tests.unit_tests.various.test_q_polynomial',
        'tests.unit_tests.various.test_q_polynomial_compiled',
        'tests.unit_tests.various.test_q_polynomial_factorized',
        'tests.unit_tests.iolibs.test_helas_call_writers',
    ],
}

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source_files():
    paths = subprocess.check_output([
        'git', 'ls-files', '-z', 'madgraph', 'aloha', 'Template',
        'tests/unit_tests', 'tests/input_files/IOTestsComparison',
        'vendor/ninjars'], text=True).split('\0')
    paths += [str(Path(__file__).relative_to(ROOT))]
    return {p: sha(ROOT/p) for p in sorted(set(paths))
            if p and (ROOT/p).is_file()}

def names(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from names(test)
        else:
            yield test.id()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('suite', choices=MODULES)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    result_path = OUT/f'{args.suite}-{args.attempt}-result.json'
    assert not result_path.exists()
    before = source_files()
    revision = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    start_utc = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    if args.suite == 'fks':
        import madgraph.interface.master_interface as commands
        from tests import test_manager
        from tests.unit_tests.iolibs import test_export_fks
        original_init = commands.MasterCmd.__init__
        def bounded_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.options['nb_core'] = 2
        commands.MasterCmd.__init__ = bounded_init
        method = 'test_w_nlo_gen_gosam'
        assert method in {item[0] for item in test_manager.to_bypass}
        setattr(test_export_fks.TestFKSOutput, method,
                unittest.skip('Existing @bypass_for_py3: CLI forbids GoSam with low_mem_multicore_nlo_generation on Python3')(
                    getattr(test_export_fks.TestFKSOutput, method)))
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES[args.suite])
    ids = list(names(suite))
    assert ids and len(ids) == len(set(ids))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    after = source_files()
    changed = sorted(k for k in before.keys() | after.keys()
                     if before.get(k) != after.get(k))
    record = {
        'schema': 'ninjars-final-dependency-software-regression-v1',
        'scope': 'Software generation, configuration, mock installation and compiled recording backend; no real NinjaRS/Symbolica evaluations.',
        'source_revision': revision,
        'source_files_before': before,
        'source_files_after': after,
        'changed_source_files': changed,
        'source_unchanged': not changed,
        'modules': MODULES[args.suite],
        'test_ids': ids,
        'tests': result.testsRun,
        'failures': [(t.id(), msg) for t,msg in result.failures],
        'errors': [(t.id(), msg) for t,msg in result.errors],
        'skipped': [(t.id(), msg) for t,msg in result.skipped],
        'suite_successful': result.wasSuccessful(),
        'passed_without_skips': result.wasSuccessful() and not result.skipped,
        'started_utc': start_utc,
        'elapsed_seconds': time.monotonic()-start,
        'affinity': sorted(os.sched_getaffinity(0)),
        'python': sys.version,
        'gfortran': subprocess.check_output(['gfortran','--version'],text=True).splitlines()[0],
        'command': sys.argv,
        'test_harness_override': (
            'FKS MasterCmd receives nb_core=2; existing Python3 GoSam bypass marked unittest.skip'
            if args.suite == 'fks' else None),
    }
    assert result.testsRun == len(ids)
    with result_path.open('x') as stream:
        json.dump(record,stream,indent=2,sort_keys=True)
        stream.write('\n')
    print(json.dumps({k:record[k] for k in (
        'tests','suite_successful','passed_without_skips','elapsed_seconds',
        'source_unchanged','changed_source_files')},sort_keys=True))
    return 0 if result.wasSuccessful() and not changed else 1

if __name__ == '__main__':
    raise SystemExit(main())
