"""Preparation control-flow checks without executing subprocesses or reducers."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("current_preparation", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class TestPreparation(unittest.TestCase):
    def test_dry_run_cannot_launch_or_activate_or_change_affinity(self):
        with patch.object(sys, "argv", [str(HERE / "run.py")]), \
             patch.object(run, "verify_seal"), \
             patch.object(run.subprocess, "Popen", side_effect=AssertionError("launch")), \
             patch.object(run.campaign, "activate_campaign", side_effect=AssertionError("activate")), \
             patch.object(run.os, "sched_setaffinity", side_effect=AssertionError("affinity")), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            run.main()
        value = json.loads(output.getvalue())
        self.assertEqual(value["status"], "dry-run-no-execution")
        self.assertEqual(value["per_stage_seconds"], 3600.)
        self.assertFalse(value["affinity_exclusive"])
        self.assertEqual([s["id"] for s in value["stages"]],
                         ["reuse-baseline", "build-baseline", "generate-candidate", "assert-declared-cards", "build-candidate"])

    def test_no_numerical_commands_or_partial_candidate_build(self):
        for stage in run.schedule():
            if stage["command"] is None:
                self.assertEqual(stage["action"], "assert_declared_cards")
                continue
            argv = stage["command"]
            self.assertEqual(argv[:3], ["taskset", "-c", "12-15"])
            self.assertIn("tools/tirust_benchmark.py", argv)
            self.assertEqual(argv[argv.index("--jobs") + 1], "4")
            self.assertEqual(argv[argv.index("--campaign") + 1], run.CAMPAIGN)
            self.assertNotIn("--cases", argv)
            self.assertNotIn("--timeout", argv)

    def test_expired_owned_cleanup_does_not_extend_stage_allowance(self):
        owned = Mock()
        owned.stop.return_value = {"survivors": []}
        with patch.object(run.time, "monotonic", return_value=100.):
            run.cleanup_owned(owned, 100.)
        self.assertEqual(owned.stop.call_args.kwargs,
                         {"cooperative_seconds": 0., "terminate_seconds": 0., "kill_seconds": 0.})
        owned.stop.return_value = {"survivors": [{"pid": 123}]}
        with patch.object(run.time, "monotonic", return_value=100.), self.assertRaisesRegex(RuntimeError, "incomplete"):
            run.cleanup_owned(owned, 105.)

    def test_nonzero_child_is_retained_and_cleanup_called(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            destination = Path(temp)
            stack.enter_context(patch.object(run, "HERE", destination))
            stack.enter_context(patch.object(run, "verify_seal"))
            stack.enter_context(patch.object(run, "configurations", return_value={"config": "original"}))
            stack.enter_context(patch.object(run, "source_snapshot", return_value={"files_sha256": "original"}))
            stack.enter_context(patch.object(run.bench, "source_identity", return_value={"files_sha256": "original"}))
            stack.enter_context(patch.object(run.campaign, "verify_campaign"))
            check = stack.enter_context(patch.object(run, "check_preparation"))
            process = Mock(pid=123)
            launch = stack.enter_context(patch.object(run.subprocess, "Popen", return_value=process))
            owned = Mock()
            owned.wait.return_value = 7
            owned.stop.return_value = {"survivors": [], "events": [], "observed_owned_processes": []}
            stack.enter_context(patch.object(run, "OwnedChild", return_value=owned))
            state = {"status": "running", "stages": []}
            with self.assertRaisesRegex(RuntimeError, "exited 7"):
                run.run_stage({"id": "synthetic", "command": ["mock-only"]}, {}, {"config": "original"}, {}, Mock(), state)
            launch.assert_called_once()
            owned.stop.assert_called_once()
            check.assert_not_called()
            entry = state["stages"][0]
            self.assertEqual(entry["status"], "failed")
            self.assertEqual(entry["returncode"], 7)
            self.assertEqual(entry["cleanup"]["survivors"], [])
            self.assertTrue((destination / "synthetic/configuration-after-failure.json").is_file())
            self.assertTrue((destination / "synthetic/source-after-failure.json").is_file())

    def test_card_assertion_rejects_wrong_hash_before_build(self):
        case = {"id": "example"}
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            work = Path(temp)
            stack.enter_context(patch.object(run, "WORK", work))
            stack.enter_context(patch.object(run.campaign, "verify_physical_card_origins", return_value={"cards": {"example": {"param_card_sha256": "wrong"}}}))
            for source in ("baseline", "candidate"):
                output = work / "outputs" / source / "example"
                run.dump(output / "generation.json", {"case": case, "source": source,
                    "campaign_binding": {"id": run.CAMPAIGN, "manifest_sha256": run.EXPECTED_MANIFEST},
                    "parameter_card_sha256": "wrong"})
                (output / "Cards").mkdir()
                (output / "Cards/param_card.dat").write_text("actual different bytes")
            with self.assertRaisesRegex(ValueError, "physical card differs"):
                run.assert_declared_cards({"cases": [case]})

    def test_deadline_error_and_signal_requests_are_not_swallowed(self):
        with self.assertRaises(TimeoutError):
            run.StageDeadline().expired(None, None)
        requests = run.SignalRequests()
        requests.requested.append(15)
        with self.assertRaises(InterruptedError):
            requests.check()

    def test_alarm_defers_without_inheriting_a_blocked_child_mask(self):
        deadline = run.StageDeadline()
        deadline.handoff = True
        deadline.deferred = False
        deadline.expired(None, None)
        self.assertTrue(deadline.deferred)
        deadline.handoff = False
        with self.assertRaises(TimeoutError):
            deadline.expired(None, None)
        self.assertNotIn("pthread_sigmask", (HERE / "run.py").read_text())


if __name__ == "__main__":
    unittest.main()
