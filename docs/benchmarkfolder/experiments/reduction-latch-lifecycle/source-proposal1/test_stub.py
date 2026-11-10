"""Pure source-contract tests. No compiler, loader, reducer or amplitude runs."""
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("latch_stub_check", HERE / "stub_check.py")
stub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stub)


class StubSourceTests(unittest.TestCase):
    def setUp(self):
        self.before,self.candidate = stub.templates()

    def test_exact_two_assignment_change_and_existing_sequence(self):
        self.assertEqual(self.candidate,self.before.replace(stub.INIT,stub.INSERT+stub.INIT,1))
        self.assertEqual(self.candidate.count(stub.INIT),1)

    def test_fragments_bind_actual_masks_and_three_transitions(self):
        for source in (self.before,self.candidate):
            parts = stub.fragments(source)
            self.assertEqual(parts["automatic_policy"].count("FPE_IN_DP_REDUCTION=.False."),3)
            self.assertIn("ACC(I)=1.0d0",parts["automatic_policy"])
            self.assertIn("IF(FPE_IN_QP_REDUCTION) THEN",parts["automatic_policy"])
        self.assertEqual(stub.fragments(self.candidate)["call_initialization"],
                         stub.INSERT+stub.fragments(self.before)["call_initialization"])

    def test_actual_public_wrapper_both_availability_paths(self):
        for available in (False,True):
            text = stub.render(self.candidate,available)
            self.assertFalse(any(line.startswith("##") for line in text.splitlines()))
            self.assertNotIn("%(proc_prefix)",text)
            self.assertEqual(text.count("SUBROUTINE P_SLOOPMATRIX(P_USER,ANS)"),1)
            self.assertEqual(text.count("SUBROUTINE P_SLOOPMATRIX_RETRY_IMPL(P_USER,ANS)"),int(available))
            self.assertLess(text.index("FPE_IN_DP_REDUCTION=.FALSE."),text.index("MP_DONE=.FALSE."))

    def test_before_entry_does_not_receive_candidate_reset(self):
        text = stub.fragments(self.before)["call_initialization"]
        self.assertNotIn("FPE_IN_",text)

    def test_fixed_form_column_six_continuation_is_preserved(self):
        self.assertEqual(stub.fixed("     $ A,B"),"     $ A,B\n")
        for available in (False,True):
            lines=stub.render(self.candidate,available).splitlines()
            self.assertFalse(any(line.startswith("           $") for line in lines))

    def test_unknown_template_condition_fails_closed(self):
        with self.assertRaises(ValueError):
            stub.conditional("## if(unknown) {\nEND\n## }",False)

    def test_environment_removes_all_loader_overrides(self):
        from unittest.mock import patch
        with patch.dict(stub.os.environ,{"LD_PRELOAD":"bad","LD_LIBRARY_PATH":"bad","LD_AUDIT":"bad","LD_ANYTHING":"bad"}):
            env = stub.clean_environment(Path("/tmp/example"))
        self.assertFalse(any(key.startswith("LD_") for key in env))

    def test_source_labels_scope_and_execution_bounds(self):
        text = (HERE / "stub_check.py").read_text()
        self.assertIn("no complete host",stub.SCOPE)
        self.assertIn('dir="/tmp"',text)
        self.assertIn('"compiler_commands": 4',text)
        self.assertIn('"stub_executions": 32',text)
        self.assertNotIn("ctypes",text)
        self.assertNotIn("cargo",text)


if __name__ == "__main__":
    unittest.main()
