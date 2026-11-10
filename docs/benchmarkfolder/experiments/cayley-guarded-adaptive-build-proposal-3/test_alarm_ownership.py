#!/usr/bin/env python3
"""Synthetic signals and fake children only; never spawn a real process."""
import importlib.util
import io
import json
from pathlib import Path
import signal
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PLAN = json.loads((HERE / 'plan.json').read_text())

def helper():
    spec = importlib.util.spec_from_file_location('fresh_synthetic_ownership_helper', ROOT / PLAN['supervisor'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def deadline(signum, frame):
    raise TimeoutError('synthetic outer deadline')

class OwnershipTests(unittest.TestCase):
    def exercise(self, corrected, phase):
        module = helper()
        if corrected:
            module.SIGNALS = (*module.SIGNALS, signal.SIGALRM)
        trace = []
        progress = {}
        class FakeChild:
            pid = 424242
            returncode = None
            def wait(self):
                trace.append('wait')
                if phase == 'reap':
                    signal.raise_signal(signal.SIGALRM)
                self.returncode = 0
                return 0
        child = FakeChild()
        def fake_popen(*args, **kwargs):
            trace.append('spawn')
            if phase == 'spawn':
                signal.raise_signal(signal.SIGALRM)
            return child
        def fake_killpg(pid, signum):
            self.assertEqual(pid, child.pid)
            self.assertEqual(signum, signal.SIGKILL)
            trace.append('kill')
        previous = signal.signal(signal.SIGALRM, deadline)
        try:
            with patch.object(module.subprocess, 'Popen', fake_popen), \
                 patch.object(module.os, 'killpg', fake_killpg), \
                 patch.object(module.os, 'waitid', return_value=SimpleNamespace(si_pid=child.pid)):
                with self.assertRaises((InterruptedError, TimeoutError)):
                    module.run_plain_capped(['synthetic-never-executed'], io.StringIO(),
                        cwd=str(ROOT), env={}, progress=progress, seconds=5)
        finally:
            signal.signal(signal.SIGALRM, previous)
        return trace, progress

    def test_alarm_between_spawn_and_ownership_is_reaped(self):
        trace, progress = self.exercise(True, 'spawn')
        self.assertEqual(trace, ['spawn', 'kill', 'wait'])
        self.assertTrue(progress['process_spawned'])
        self.assertTrue(progress['process_reaped'])

    def test_alarm_during_reaping_does_not_repeat_stale_pid_cleanup(self):
        trace, progress = self.exercise(True, 'reap')
        self.assertEqual(trace, ['spawn', 'kill', 'wait'])
        self.assertTrue(progress['process_reaped'])

    def test_original_missing_alarm_deferral_is_discriminated(self):
        trace, progress = self.exercise(False, 'spawn')
        self.assertEqual(trace, ['spawn'])
        self.assertNotIn('process_reaped', progress)
        self.assertNotIn('process_spawned', progress)

if __name__ == '__main__':
    unittest.main()
