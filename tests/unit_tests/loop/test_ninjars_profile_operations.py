"""Profiler child ownership and interruption controls, without perf or physics."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from tools import ninjars_profile_v2 as profile


class NinjaRSProfileOperationsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            dir=profile.ROOT / ".ninjars-build/tmp", prefix="profile-operations-")
        self.addCleanup(self.temp.cleanup)
        self.report = Path(self.temp.name)
        self.children = []
        self.addCleanup(self.reap_fallback)

    def reap_fallback(self):
        for process in self.children:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5.)

    def sleeping_child(self):
        process = profile.workers.subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            cwd=self.report, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
        self.children.append(process)
        return process

    def test_term_and_hup_reap_sleeping_session_and_restore_handlers(self):
        original_subprocess = profile.workers.subprocess
        for signum in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=signum):
                previous = signal.getsignal(signum)
                with self.assertRaisesRegex(SystemExit, "Profiler interrupted"):
                    with profile.owned_runtime(self.report):
                        child = self.sleeping_child()
                        self.assertEqual(os.getpgid(child.pid), child.pid)
                        os.kill(os.getpid(), signum)
                self.assertIsNotNone(child.poll())
                self.assertEqual(signal.getsignal(signum), previous)
                self.assertIs(profile.workers.subprocess, original_subprocess)

    def test_exception_before_worker_assignment_reaps_registered_child(self):
        child = None

        def interrupted_constructor():
            nonlocal child
            child = self.sleeping_child()
            raise RuntimeError("Worker READY failed before ownership transfer")

        with self.assertRaisesRegex(RuntimeError, "READY failed"):
            with profile.owned_runtime(self.report):
                interrupted_constructor()
        self.assertIsNotNone(child.poll())
        ledger = (self.report / "process-ledger.jsonl").read_text()
        self.assertIn('"event": "spawn"', ledger)
        self.assertIn('"reaped": true', ledger)

    def test_all_thread_counts_are_pinned_without_losing_runtime_environment(self):
        supplied = {name: "8" for name in profile.pair.THREADS}
        supplied["LD_LIBRARY_PATH"] = "the-bound-runtime"
        with mock.patch.object(profile.workers, "environment", return_value=supplied):
            actual = profile.profile_environment(self.report)
        self.assertEqual({name: actual[name] for name in profile.pair.THREADS},
                         {name: "1" for name in profile.pair.THREADS})
        self.assertEqual(actual["LD_LIBRARY_PATH"], "the-bound-runtime")

    def test_perf_flush_wait_is_bounded_for_the_owned_escalation_path(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("perf", 10.)
        with self.assertRaises(subprocess.TimeoutExpired):
            profile.stop_profiler(process)
        process.send_signal.assert_called_once_with(signal.SIGINT)
        process.wait.assert_called_once_with(timeout=10.)

    def test_already_exited_perf_is_never_signalled(self):
        process = mock.Mock()
        process.poll.return_value = 0
        process.wait.return_value = 0
        self.assertEqual(profile.stop_profiler(process), 0)
        process.send_signal.assert_not_called()


if __name__ == "__main__":
    unittest.main()
