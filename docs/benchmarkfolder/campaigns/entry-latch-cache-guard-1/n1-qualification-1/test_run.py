"""Supervision tests only; no CLI generation, compiler or reducer is launched."""
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("n1_supervisor_test", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class N1SupervisionTests(unittest.TestCase):
    def test_exact_two_case_cli_schedule_and_caps(self):
        rows = run.schedule()
        self.assertEqual([r["cap_seconds"] for r in rows], [600., 900., 1200., 1800., 1800.])
        self.assertEqual(len(rows), 5)
        for row in rows[:3]:
            self.assertEqual(row["command"][:3], ["taskset", "-c", "12-15"])
            i = row["command"].index("--cases")
            self.assertEqual(row["command"][i+1:i+3], run.CASES)
        for row in rows[3:]:
            command = row["command"]
            self.assertIn("--warmup-only", command)
            self.assertNotIn("--timeout", command)
            i = command.index("--backends")
            self.assertEqual(command[i+1:i+6], ["1", "6", "7", "8", "9"])
            self.assertEqual(command[-2:], ["--cpu", "12"])
            self.assertEqual(command[command.index("--campaign")+1], run.CAMPAIGN)
        self.assertEqual([r["case"] for r in rows[3:]], run.CASES)

    def test_pinned_support_is_imported_without_historical_execution(self):
        with patch.object(run.subprocess, "Popen") as launch:
            cleanup, preparation, numerical = run.load_support()
        launch.assert_not_called()
        self.assertEqual(preparation.StageDeadline.__module__, "n1_preparation_support")
        self.assertIs(preparation.OwnedChild, cleanup.OwnedChild)
        self.assertEqual(numerical.CAMPAIGN, run.CAMPAIGN)
        self.assertEqual(numerical.EXPECTED_MANIFEST, run.MANIFEST_SHA)
        self.assertEqual(numerical.EXPECTED_LANES, [("baseline", 1), ("candidate", 1),
            ("candidate", 6), ("candidate", 7), ("candidate", 8), ("candidate", 9)])

    def test_changed_support_is_rejected_before_import(self):
        with patch.object(run, "digest", return_value="changed"), self.assertRaises(ValueError):
            run.load_support()

    def test_default_invocation_never_activates_or_spawns(self):
        with patch.object(sys, "argv", ["run.py"]), patch.object(run, "verify_seal"), \
             patch.object(run.campaign, "activate_campaign") as activate, \
             patch.object(run.subprocess, "Popen") as launch, redirect_stdout(io.StringIO()) as stream:
            run.main()
        activate.assert_not_called(); launch.assert_not_called()
        value = json.loads(stream.getvalue())
        self.assertEqual(value["status"], "dry-run-no-execution")
        self.assertEqual(value["worker_deadline_seconds"], 3600.)
        self.assertFalse(value["performance_acceptance_eligible"])

    def test_cleanup_never_allocates_more_than_remaining_reserved_time(self):
        calls = []
        child = SimpleNamespace(stop=lambda **kwargs: calls.append(kwargs) or {"survivors": []})
        with patch.object(run.time, "monotonic", return_value=100.):
            run.cleanup_owned(child, 109.)
        self.assertAlmostEqual(sum(calls[0].values()), 9.)
        child.stop = lambda **kwargs: {"survivors": [{"pid": 1}]}
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            run.cleanup_owned(child, time.monotonic()+60)

    def stage_fixture(self, numeric=False, returncode=0, drift=False, cleanup_failure=False):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        folder = Path(temporary.name)
        stack = ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch.object(run, "HERE", folder))
        stack.enter_context(patch.object(run, "verify_seal"))
        stack.enter_context(patch.object(run.campaign, "verify_campaign"))
        stack.enter_context(patch.object(run, "configurations", side_effect=[{}, {"drift": True}] if drift else lambda: {}))
        stack.enter_context(patch.object(run, "declared_cards", return_value={}))
        stack.enter_context(patch.object(run, "verify_builds", return_value={"bound": "inputs"}))
        process = SimpleNamespace(pid=12345)
        launch = stack.enter_context(patch.object(run.subprocess, "Popen", return_value=process))
        calls = []
        class Owned:
            def __init__(self, process): pass
            def wait(self, requests): return returncode
            def stop(self, **kwargs):
                calls.append(kwargs)
                return {"survivors": [{"pid": 12345}] if cleanup_failure else []}
        class Deadline:
            def __enter__(self):
                self.end = time.monotonic()+60.; self.handoff=False; self.deferred=False
                return self
            def __exit__(self, *args): pass
        cleanup = SimpleNamespace(OwnedChild=Owned)
        preparation = SimpleNamespace(StageDeadline=Deadline)
        observed = stack.enter_context(patch.object(run, "read", wraps=run.read))
        numerical = SimpleNamespace(observed_result=lambda *args: {"status": "passed" if returncode == 0 else "failed"})
        row = run.schedule()[3 if numeric else 1]
        return folder, row, {"stages": []}, (cleanup, preparation, numerical), launch, calls

    def test_stage_delegates_exact_command_and_records_success(self):
        folder,row,state,support,launch,calls = self.stage_fixture()
        run.run_stage(row, {}, {}, {}, SimpleNamespace(check=lambda: None), state, support)
        self.assertEqual(state["stages"][0]["status"], "passed")
        self.assertEqual(support[1].STAGE_SECONDS, 900.)
        self.assertEqual(support[1].CLEANUP_RESERVE, 60.)
        self.assertEqual(launch.call_args.args[0], row["command"])
        self.assertTrue(launch.call_args.kwargs["start_new_session"])
        self.assertTrue((folder/row["id"]/"configuration-after.json").is_file())

    def test_command_failure_retained_and_raises(self):
        folder,row,state,support,launch,calls = self.stage_fixture(returncode=2)
        with self.assertRaisesRegex(RuntimeError, "Preparation command failed"):
            run.run_stage(row, {}, {}, {}, SimpleNamespace(check=lambda: None), state, support)
        self.assertEqual(state["stages"][0]["status"], "failed")
        self.assertEqual(state["stages"][0]["returncode"], 2)
        self.assertEqual(launch.call_count, 1)

    def test_configuration_drift_blocks_success_without_restoration(self):
        folder,row,state,support,launch,calls = self.stage_fixture(drift=True)
        with self.assertRaisesRegex(ValueError, "Configuration drift"):
            run.run_stage(row, {}, {}, {}, SimpleNamespace(check=lambda: None), state, support)
        retained = run.read(folder/row["id"]/"configuration-after.json")
        self.assertEqual(retained, {"drift": True})
        self.assertEqual(state["stages"][0]["status"], "failed")

    def test_cleanup_uncertainty_blocks_numerical_replay_and_later_work(self):
        folder,row,state,support,launch,calls = self.stage_fixture(numeric=True, cleanup_failure=True)
        support[2].observed_result = lambda *args: self.fail("Uncertain cleanup must not qualify results")
        with self.assertRaisesRegex(RuntimeError, "unresolved"):
            run.run_stage(row, {}, {}, {}, SimpleNamespace(check=lambda: None), state, support)
        self.assertEqual(state["stages"][0]["status"], "failed")
        self.assertIn("cleanup_error", state["stages"][0])

    def test_numerical_failure_keeps_original_replay_record(self):
        folder,row,state,support,launch,calls = self.stage_fixture(numeric=True, returncode=1)
        with self.assertRaisesRegex(RuntimeError, "Numerical case failed"):
            run.run_stage(row, {}, {}, {}, SimpleNamespace(check=lambda: None), state, support)
        self.assertEqual(run.read(folder/row["id"]/"numerical-result.json"), {"status": "failed"})
        self.assertEqual(state["stages"][0]["status"], "failed")

    def test_signal_after_wait_cannot_be_published_as_success(self):
        folder,row,state,support,launch,calls = self.stage_fixture()
        count=[0]
        def check():
            count[0]+=1
            if count[0]>=2: raise InterruptedError("requested stop")
        with self.assertRaises(InterruptedError):
            run.run_stage(row, {}, {}, {}, SimpleNamespace(check=check), state, support)
        self.assertEqual(state["stages"][0]["status"], "failed")
        self.assertTrue(calls)


if __name__ == "__main__":
    unittest.main()
