#!/usr/bin/env python3
"""Pure-Python checks; subprocess launch is forbidden in this test process."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('portable_fixture_verifier', HERE/'verify.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class Tests(unittest.TestCase):
    def test_complete_replay_and_failure_semantics(self):
        value = verify.verify(HERE, check_inventory=False)
        self.assertEqual((value['normal_cases'], value['positive_invalid_trap_controls'],
                          value['dp_exact_pairs'], value['qp_transport_pairs']), (216, 2, 90, 18))
        self.assertEqual(value['captured_events']['amplitude_worker_jobs'], 0)

    def test_modified_and_truncated_journals_rejected(self):
        module = verify.load_frozen_validator(HERE)
        root = HERE/'payload'
        commands = verify.read(root/verify.BASE/'fixture/commands.json')
        job = commands['runs'][0]
        matrix = verify.read(root/verify.BASE/'fixture/argument-matrix.json')
        text = (root/job['cwd']/'caller.hex').read_text()
        module.parse_case(text, job, matrix)
        for changed in [text.rsplit('\n', 2)[0]+'\n', text+'EXTRA\n', text.replace('STATUS 1 0', 'STATUS 1 1')]:
            self.assertNotEqual(text, changed)
            with self.assertRaises(ValueError):
                module.parse_case(changed, job, matrix)

    def test_closed_inventory_rejects_extra_missing_changed(self):
        with tempfile.TemporaryDirectory(dir=HERE) as raw:
            root = Path(raw)
            (root/'a.txt').write_text('bound\n')
            expected = {'a.txt': verify.sha(root/'a.txt')}
            self.assertEqual(verify.closed_inventory(root, expected), 1)
            (root/'extra.txt').write_text('unbound\n')
            with self.assertRaises(ValueError): verify.closed_inventory(root, expected)
            (root/'extra.txt').unlink()
            (root/'a.txt').write_text('changed\n')
            with self.assertRaises(ValueError): verify.closed_inventory(root, expected)
            (root/'a.txt').unlink()
            with self.assertRaises(ValueError): verify.closed_inventory(root, expected)

    def test_all_fixture_outputs_are_preserved(self):
        root = HERE/'payload'
        commands = verify.read(root/verify.BASE/'fixture/commands.json')
        self.assertEqual(len(commands['runs']), 218)
        for job in commands['runs']:
            directory = root/job['cwd']
            self.assertEqual({x.name for x in directory.iterdir()}, {'caller.hex', 'stdout.log'})


if __name__ == '__main__':
    stream = io.StringIO()
    with patch.object(subprocess, 'Popen', side_effect=AssertionError('Native process forbidden')):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    record = dict(schema='qp-mass-portable-source-tests-v1', tests=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  native_launched=False, output=stream.getvalue(),
                  verify_sha256=verify.sha(HERE/'verify.py'), test_sha256=verify.sha(Path(__file__)))
    if not (HERE/'index.json').exists():
        (HERE/'source-tests.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    print(json.dumps(record, indent=2, sort_keys=True))
    sys.exit(0 if result.wasSuccessful() else 1)
