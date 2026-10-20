#!/usr/bin/env python3
"""Read-only real-evidence and corruption checks for the prospective reuse guard."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
SPEC = importlib.util.spec_from_file_location('tensor_reuse_check',
    V2 / 'final-runtime-audit-abi3-tensor-factor-v1/check-reuse.py')
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)
# Use the unchanged completed inventory as both sides of the intended reuse
# boundary. Calling check() never writes its result or launches a subprocess.
CHECKER.HERE = V2 / 'final-runtime-audit-fixed'
READ = CHECKER.policy.read
INVENTORY = CHECKER.HERE / 'inventory.json'


class ReuseGuardTests(unittest.TestCase):
    def test_all_thirteen_existing_postimages_and_runtimes_are_eligible(self):
        result = CHECKER.check()
        self.assertTrue(result['passed'])
        self.assertEqual(len(result['cases']), 13)
        self.assertEqual(result['numerical_evaluations'], 0)

    def corrupt_current_inventory(self, mutation):
        seen = 0
        def read(filename):
            nonlocal seen
            data = READ(filename)
            if Path(filename) == INVENTORY:
                seen += 1
                if seen == 2:
                    data = copy.deepcopy(data)
                    mutation(data)
                    data.pop('identity_sha256')
                    data = CHECKER.policy.seal(data)
            return data
        return mock.patch.object(CHECKER.policy, 'read', side_effect=read)

    def test_changed_runtime_signature_rejects_reuse(self):
        with self.corrupt_current_inventory(lambda d:
                d['cases']['qcd-3']['signatures'].__setitem__('8', '0' * 64)):
            with self.assertRaisesRegex(RuntimeError, 'Runtime/interface metadata changed'):
                CHECKER.check()

    def test_changed_module_metadata_rejects_reuse(self):
        def mutate(data):
            metadata = data['cases']['qcd-3']['runtime_metadata']
            metadata[next(iter(metadata))] = '0' * 64
        with self.corrupt_current_inventory(mutate):
            with self.assertRaisesRegex(RuntimeError, 'Runtime/interface metadata changed'):
                CHECKER.check()

    def test_changed_exact_model_postimage_rejects_reuse(self):
        inventory = READ(INVENTORY)
        compiled = CHECKER.policy.path(inventory['cases']['qcd-3']['compiled_manifest']['path'])
        def read(filename):
            data = READ(filename)
            if Path(filename) == compiled:
                data = copy.deepcopy(data)
                data['files']['Source/MODEL/model_functions.f']['sha256'] = '0' * 64
            return data
        with mock.patch.object(CHECKER.policy, 'read', side_effect=read):
            with self.assertRaisesRegex(RuntimeError, 'wrong exact model postimages'):
                CHECKER.check()


if __name__ == '__main__':
    unittest.main(verbosity=2)
