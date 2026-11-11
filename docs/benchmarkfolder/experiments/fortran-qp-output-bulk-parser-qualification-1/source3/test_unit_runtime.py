"""Pure ELF-byte and retained-failure tests; no loader/compiler invocation."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent;ROOT=Path.cwd()
spec=importlib.util.spec_from_file_location('bulk_unit_runtime_tests',HERE/'unit_runtime.py')
runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
FAILED=ROOT/'.tirust-build/fortran-qp-output-bulk-unit-work-2/build/original'


class RuntimeTests(unittest.TestCase):
    def test_retained_failed_elf_has_unit_specific_five_provider_closure(self):
        pre=json.loads((FAILED/'prelink.json').read_text())
        derived=runtime.derive(FAILED/'check',pre)
        self.assertEqual(set(derived['sonames']),runtime.REQUIRED_SONAMES)
        self.assertEqual(len(derived['files']),5)
        self.assertNotIn('libquadmath.so.0',derived['sonames'])
        self.assertTrue(derived['actual_interpreter_query_still_required'])
        self.assertEqual(derived['compiler_or_loader_invocations'],0)
    def test_profile_only_quadmath_expected_provider_is_rejected(self):
        pre=json.loads((FAILED/'prelink.json').read_text())
        derived=runtime.derive(FAILED/'check',pre)
        old=json.loads((ROOT/'.tirust-build/fortran-qp-output-bulk-unit-controller-2/input-pins.json').read_text())
        with self.assertRaisesRegex(ValueError,'unit-specific dependency closure'):
            runtime.validate_expected(derived,old['expected_runtime'])
        runtime.validate_expected(derived,derived['files'])
    def test_unbound_selected_provider_rejected_before_query(self):
        pre=json.loads((FAILED/'prelink.json').read_text())
        derived=runtime.derive(FAILED/'check',pre)
        victim=derived['sonames']['libgfortran.so.5']
        del pre['bound'][victim]
        with self.assertRaisesRegex(ValueError,'not prebound'):
            runtime.derive(FAILED/'check',pre)
    def test_interpreter_must_be_prebound(self):
        pre=json.loads((FAILED/'prelink.json').read_text())
        derived=runtime.derive(FAILED/'check',pre)
        del pre['bound'][derived['sonames']['ld-linux-x86-64.so.2']]
        with self.assertRaisesRegex(ValueError,'Interpreter was not prebound'):
            runtime.derive(FAILED/'check',pre)
    def test_truncated_and_non_elf_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'not-elf'
            for value in [b'',b'not an ELF',b'\x7fELF\x02\x01\x01']:
                p.write_bytes(value)
                with self.assertRaises(ValueError):runtime.describe(p)


if __name__=='__main__':unittest.main(verbosity=2)
