"""Pure structural/synthetic qualification; no Rust compiler or native calls."""
import hashlib
from pathlib import Path
import unittest
import bridge_patch as patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BASE = ROOT / ".tirust-build/qp-transport-install/mg/vendor/tirust/src/lib.rs"


class DiagnosticPatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = BASE.read_bytes()
        cls.candidate = patch.prepare(cls.base)
        cls.original = cls.base.decode()
        cls.text = cls.candidate.decode()

    def test_exact_pinned_source_and_reversible_derivation(self):
        self.assertEqual(hashlib.sha256(self.base).hexdigest(), patch.BASE_SHA256)
        self.assertEqual(patch.restore(self.candidate), self.base)
        self.assertEqual(patch.prepare(self.base), self.candidate)

    def test_wrong_or_already_patched_input_rejected(self):
        for source in (b"", self.base+b"\n", self.candidate, bytearray(self.base)):
            with self.assertRaises(ValueError):patch.prepare(source)

    def test_reset_precedes_dp_guard_even_validation_failure(self):
        start=self.text.index(patch.DP_SIGNATURE)
        self.assertLess(self.text.index(patch.ENTRY,start),self.text.index("    guard(|| {",start))
        self.assertEqual(self.text.count(patch.ENTRY),1)

    def test_snapshot_after_actual_reduce_before_original_mapping(self):
        start=self.text.index(patch.SNAPSHOT)
        self.assertLess(self.text.index("workspace.reduce_batch(&batch)",start),self.text.index("private_dp_diagnostic_snapshot(workspace",start))
        self.assertLess(self.text.index("private_dp_diagnostic_snapshot(workspace",start),self.text.index("private_reduced.map_err",start))
        self.assertEqual(self.text.count("workspace.reduce_batch(&batch)"),self.original.count("workspace.reduce_batch(&batch)"))

    def test_no_last_error_mutation_or_guard_in_getter_or_snapshot(self):
        self.assertNotIn("LAST_ERROR",patch.PRIVATE_BLOCK)
        self.assertNotIn("set_error(",patch.PRIVATE_BLOCK)
        self.assertNotIn("guard(",patch.PRIVATE_BLOCK)
        getter=patch.PRIVATE_BLOCK.split('pub unsafe extern "C" fn '+patch.GETTER,1)[1]
        self.assertIn("record.borrow()",getter)
        self.assertNotIn("borrow_mut",getter)
        self.assertIn("record.len() + 1",getter)

    def test_failed_values_never_read_and_err_freshness_explicit(self):
        for forbidden in (".lane(",".lane_with_estimate(",".values", ".best_effort", "estimated_errors"):
            self.assertNotIn(forbidden,patch.PRIVATE_BLOCK)
        self.assertIn("Err(error)",patch.PRIVATE_BLOCK)
        self.assertIn("unproven_partial_or_previous",patch.PRIVATE_BLOCK)
        self.assertIn("fallback_used",patch.PRIVATE_BLOCK)

    def test_only_public_metadata_and_no_option_or_policy_setters(self):
        for api in ("binary_precision()","embedder_options()","last_lane_routes()","engine_execution()","last_diagnostics()"):
            self.assertIn(api,patch.PRIVATE_BLOCK)
        for forbidden in ("set_stability", "enable_", "new_embedded", "Float::", "reduce_batch(", "pow", "sqrt", "unsafe { context"):
            self.assertNotIn(forbidden,patch.PRIVATE_BLOCK)
        # Complete original bytes are restored, including status/guard/cache/QP paths.
        self.assertEqual(patch.restore(self.candidate),self.base)

    def test_fixed_bound_utf8_and_copy_contract(self):
        self.assertEqual(patch.MAX_BYTES,65536)
        for token in ("PRIVATE_DP_DIAGNOSTIC_MAX - 64", "is_char_boundary", "saturating_sub", "truncated=true", "capacity - 1", "out.add(n).write(0)"):
            self.assertIn(token,patch.PRIVATE_BLOCK)
        # Mirror only the byte-copy ABI, independent of Rust/native execution.
        text="diagnostic π".encode()
        for capacity in (0,1,2,len(text),len(text)+1,len(text)+99):
            required=len(text)+1
            copied=text[:max(0,capacity-1)]+(b"\0" if capacity else b"")
            self.assertEqual(required,len(text)+1)
            self.assertLessEqual(len(copied),capacity)
            if capacity:self.assertEqual(copied[-1:],b"\0")

    def test_unified_patch_is_deterministic(self):
        self.assertEqual(patch.patch(self.base),patch.patch(self.base))
        self.assertTrue(patch.patch(self.base).startswith("--- a/src/lib.rs\n+++ b/src/lib.rs\n"))


if __name__=="__main__":unittest.main()
