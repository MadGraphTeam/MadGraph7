"""Fake controllers and detached sleeping workers; no MadLoop/reducer calls."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from process_cleanup import OwnedChild, SignalRequests, identity, matches


WORKER = """
import json,os,signal,sys,time
from pathlib import Path
if sys.argv[2]=='ignore':
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
Path(sys.argv[1]).write_text(json.dumps({'pid':os.getpid()}))
time.sleep(90)
"""
CONTROLLER = """
import json,os,signal,subprocess,sys,time
from pathlib import Path
folder=Path(sys.argv[1]); mode=sys.argv[2]
if mode=='ignore':
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
child=subprocess.Popen([sys.executable,'-B','-c',sys.argv[3],str(folder/'worker.json'),mode],start_new_session=True)
try:
    while not (folder/'worker.json').exists():time.sleep(.01)
    (folder/'controller.json').write_text(json.dumps({'pid':os.getpid(),'phase':mode}))
    time.sleep(90)
finally:
    if mode!='ignore':
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        child.wait(timeout=3)
"""


class CleanupTests(unittest.TestCase):
    def spawn(self, folder, phase):
        child = subprocess.Popen([sys.executable, "-B", "-c", CONTROLLER, str(folder), phase, WORKER],
                                 start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        owned = OwnedChild(child)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            owned.sample()
            try:
                control = json.loads((folder / "controller.json").read_text())
                worker = json.loads((folder / "worker.json").read_text())
                token = identity(worker["pid"])
                if control["phase"] == phase and token:
                    owned.sample()
                    self.assertIn(worker["pid"], owned.records)
                    return owned, token
            except (FileNotFoundError, ValueError):
                pass
            time.sleep(.01)
        owned.stop(cooperative_seconds=.2, terminate_seconds=.2, kill_seconds=.2)
        self.fail("Fake controller failed to reach the requested phase")

    def interrupted_phase(self, phase):
        with tempfile.TemporaryDirectory(prefix="tirust-cleanup-fake-") as name:
            with SignalRequests() as requests:
                owned, worker = self.spawn(Path(name), phase)
                timer = threading.Timer(.05, os.kill, (os.getpid(), signal.SIGTERM))
                timer.start()
                try:
                    with self.assertRaises(InterruptedError):
                        owned.wait(requests, poll=.01)
                finally:
                    timer.join()
                    result = owned.stop(cooperative_seconds=2., terminate_seconds=.3, kill_seconds=.3)
                self.assertEqual(result["survivors"], [])
                self.assertFalse(matches(worker))
                self.assertFalse(matches(owned.root))
                self.assertEqual(result["events"][0]["signal"], "SIGINT")

    def test_supervisor_sigterm_during_startup_cleans_detached_worker(self):
        self.interrupted_phase("startup")

    def test_supervisor_sigterm_during_wait_cleans_detached_worker(self):
        self.interrupted_phase("waiting")

    def test_forced_fallback_cleans_only_owned_sessions(self):
        with tempfile.TemporaryDirectory(prefix="tirust-cleanup-fake-") as name:
            owned, worker = self.spawn(Path(name), "ignore")
            result = owned.stop(cooperative_seconds=.1, terminate_seconds=.1, kill_seconds=2.)
            self.assertEqual(result["survivors"], [])
            self.assertFalse(matches(worker))
            self.assertFalse(matches(owned.root))
            self.assertIn("SIGKILL", [event["signal"] for event in result["events"]])

    def test_recycled_start_identity_cannot_authorize_group_signal(self):
        with tempfile.TemporaryDirectory(prefix="tirust-cleanup-fake-") as name:
            owned, _ = self.spawn(Path(name), "startup")
            original = owned.records
            try:
                owned.records = {pid: {**record, "start_time_ticks": record["start_time_ticks"] + 1}
                                 for pid, record in original.items()}
                with mock.patch("process_cleanup.os.killpg") as send:
                    self.assertEqual(owned.signal_groups(signal.SIGTERM), [])
                    send.assert_not_called()
            finally:
                owned.records = original
                self.assertEqual(owned.stop(cooperative_seconds=2.)["survivors"], [])


if __name__ == "__main__":
    unittest.main()
