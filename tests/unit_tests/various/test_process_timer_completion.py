"""Deterministic ProcessTimer regression tests; no child is launched."""

import ast
import io
import logging
import os
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[3]


def extract(path, name, owner=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if owner is not None:
        nodes = next(node for node in nodes
                     if isinstance(node, ast.ClassDef) and node.name == owner).body
    return next(node for node in nodes
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)


class Clock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def perf_counter(self):
        return self.now

    def time(self):
        return 1000.0 + self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class Harness:
    """One manual scheduler controls Popen, waiter publication and RSS samples."""

    def __init__(self, duration=.013, returncode=0, rss=(41,)):
        self.clock = Clock()
        self.duration = duration
        self.returncode = returncode
        self.rss = list(rss)
        self.samples = []
        self.spawns = []
        self.waits = []
        self.files = []
        self.threads = []
        self.events = []
        self.child = None
        self.spawn_error = None
        self.wait_error = None
        self.ps_error = None
        self.ps_output = None
        self.during_ps = None
        self.compile_calls = []
        self.call_calls = []
        self.ignore_terminate = False
        self.never_exit = False
        self.load()

    def opened(self, *args, **kwargs):
        stream = io.StringIO()
        self.files.append(stream)
        return stream

    def thread_factory(self, target, daemon):
        harness = self

        class ManualThread:
            def __init__(self):
                self.target = target
                self.daemon = daemon
                self.started = False
                self.finished = False
                self.join_calls = 0

            def start(self):
                self.started = True

            def run(self):
                if not self.finished:
                    try:
                        self.target()
                    finally:
                        self.finished = True

            def join(self, timeout=None):
                self.join_calls += 1
                raise AssertionError('Cleanup must not join a running child')

        thread = ManualThread()
        harness.threads.append(thread)
        return thread

    def event_factory(self):
        harness = self

        class ManualEvent:
            def __init__(self):
                self.flag = False

            def set(self):
                self.flag = True

            def is_set(self):
                return self.flag

            def wait(self, timeout=None):
                harness.waits.append(timeout)
                if not self.flag:
                    end = harness.child.end
                    if timeout is None or end <= harness.clock.now + timeout:
                        harness.clock.now = max(harness.clock.now, end)
                        harness.threads[-1].run()
                    else:
                        harness.clock.now += timeout
                return self.flag

        event = ManualEvent()
        harness.events.append(event)
        return event

    def popen(self, command, **kwargs):
        harness = self
        if isinstance(command, str) and command.startswith('ps -p '):
            if self.ps_error is not None:
                raise self.ps_error
            self.samples.append(self.clock.now)
            index = min(len(self.samples) - 1, len(self.rss) - 1)
            output = (self.ps_output if self.ps_output is not None else
                      ('RSS\n%s\n' % self.rss[index]).encode())

            class Flash:
                def communicate(self):
                    if harness.during_ps is not None:
                        harness.during_ps()
                    return output, None

            return Flash()
        self.spawns.append((command, kwargs))
        if self.spawn_error is not None:
            raise self.spawn_error

        class Child:
            pid = 1234

            def __init__(self):
                self.end = harness.clock.now + harness.duration
                self.returncode = None
                self.terminate_calls = 0
                self.kill_calls = 0
                self.wait_calls = 0

            def poll(self):
                if (self.returncode is None and not harness.never_exit and
                        harness.clock.now >= self.end):
                    self.returncode = harness.returncode
                return self.returncode

            def wait(self):
                self.wait_calls += 1
                if harness.wait_error is not None:
                    raise harness.wait_error
                harness.clock.now = max(harness.clock.now, self.end)
                self.returncode = harness.returncode
                return self.returncode

            def terminate(self):
                self.terminate_calls += 1
                if not harness.ignore_terminate:
                    self.returncode = -15

            def kill(self):
                self.kill_calls += 1
                self.returncode = -9

        self.child = Child()
        return self.child

    def compile(self, **kwargs):
        self.compile_calls.append(kwargs)
        self.clock.now += .01
        return 0

    def call(self, command, **kwargs):
        self.call_calls.append((command, kwargs))
        self.clock.now += self.duration
        return self.returncode

    def load(self):
        misc_path = SOURCE_ROOT / 'madgraph/various/misc.py'
        interface_path = SOURCE_ROOT / 'madgraph/interface/madevent_interface.py'
        subprocess = SimpleNamespace(Popen=self.popen, PIPE=object(), call=self.call)
        namespace = dict(time=self.clock, subprocess=subprocess,
                         threading=SimpleNamespace(Event=self.event_factory,
                                                   Thread=self.thread_factory),
                         open=self.opened, os=SimpleNamespace(
                             devnull='unused-devnull', path=SimpleNamespace(isfile=lambda _: False)))
        node = extract(misc_path, 'ProcessTimer')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(misc_path), 'exec'), namespace)
        self.Timer = namespace['ProcessTimer']
        node = extract(interface_path, 'make_and_run', 'MadLoopInitializer')
        node.decorator_list = []
        namespace.update(misc=SimpleNamespace(ProcessTimer=self.Timer, compile=self.compile),
                         pjoin=os.path.join, logging=logging)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(interface_path), 'exec'), namespace)
        self.make_and_run = namespace['make_and_run']

    def timer(self):
        timer = self.Timer(['unused-check'])
        timer.execute()
        return timer


class ProcessTimerCompletionTests(unittest.TestCase):
    def test_quick_child_is_not_charged_the_rss_interval(self):
        harness = Harness(duration=.013)
        compiled, elapsed, ram = harness.make_and_run('unused-directory', checkRam=True)
        self.assertAlmostEqual(compiled, .01)
        self.assertAlmostEqual(elapsed, .013)
        self.assertEqual(ram, 41)
        self.assertEqual(harness.clock.sleeps, [])
        self.assertIn(.2, harness.waits)
        self.assertTrue(harness.threads[0].daemon)
        self.assertEqual(harness.child.wait_calls, 1)
        self.assertEqual(harness.child.terminate_calls, 0)
        self.assertTrue(all(stream.closed for stream in harness.files))

    def test_long_child_retains_periodic_rss_maximum(self):
        harness = Harness(duration=.55, rss=(128, 64, 512))
        _, elapsed, ram = harness.make_and_run('unused-directory', checkRam=True)
        self.assertAlmostEqual(elapsed, .55)
        self.assertEqual(ram, 512)
        self.assertEqual(len(harness.samples), 3)
        self.assertAlmostEqual(harness.samples[1] - harness.samples[0], .2)
        self.assertAlmostEqual(harness.samples[2] - harness.samples[1], .2)

    def test_exit_during_ps_accepts_missing_or_invalid_rss(self):
        for output in (b'RSS\n', b'', b'RSS\nexited\n'):
            with self.subTest(output=output):
                harness = Harness()
                harness.ps_output = output
                timer = harness.timer()

                def exit_during_sample():
                    harness.clock.now = harness.child.end
                    harness.threads[0].run()

                harness.during_ps = exit_during_sample
                self.assertFalse(timer.poll())
                self.assertAlmostEqual(timer.t1 - timer.t0, .013)
                self.assertEqual(timer.max_rss_memory, 0)
                self.assertFalse(timer.execution_state)

    def test_delayed_poll_preserves_published_completion_timestamp(self):
        harness = Harness()
        timer = harness.timer()
        self.assertTrue(timer.wait(.2))
        completion = timer.t1
        harness.clock.now += 5.0
        self.assertFalse(timer.poll())
        self.assertFalse(timer.check_execution_state())
        self.assertEqual(timer.t1, completion)
        self.assertAlmostEqual(timer.t1 - timer.t0, .013)
        self.assertEqual(harness.samples, [])

    def test_observed_exit_waits_for_timestamp_publication(self):
        harness = Harness()
        timer = harness.timer()
        harness.clock.now = harness.child.end
        self.assertEqual(harness.child.poll(), 0)
        self.assertIsNone(timer.t1)
        self.assertFalse(timer.poll())
        self.assertIn(None, harness.waits)
        self.assertIsNotNone(timer.t1)
        self.assertTrue(timer._completion.is_set())
        self.assertAlmostEqual(timer.t1 - timer.t0, .013)

    def test_nonzero_exit_preserves_make_and_run_failure_result(self):
        harness = Harness(returncode=7)
        self.assertEqual(harness.make_and_run('unused-directory', checkRam=True),
                         (None, None, None))
        self.assertEqual(harness.child.returncode, 7)
        self.assertEqual(harness.child.wait_calls, 1)

    def test_waiter_exception_is_published_and_propagated(self):
        harness = Harness()
        failure = OSError('fake wait failed')
        harness.wait_error = failure
        with self.assertRaises(OSError) as caught:
            harness.make_and_run('unused-directory', checkRam=True)
        self.assertIs(caught.exception, failure)
        self.assertTrue(harness.events[0].is_set())
        self.assertTrue(harness.threads[0].finished)
        self.assertEqual(harness.child.wait_calls, 1)

    def test_spawn_failure_cleanup_keeps_the_original_exception(self):
        harness = Harness()
        failure = OSError('fake executable missing')
        harness.spawn_error = failure
        with self.assertRaises(OSError) as caught:
            harness.make_and_run('unused-directory', checkRam=True)
        self.assertIs(caught.exception, failure)
        self.assertIsNone(harness.child)
        self.assertEqual(harness.threads, [])

    def test_active_execute_rejects_reuse_without_spawning_again(self):
        harness = Harness()
        timer = harness.timer()
        first_child = timer.p
        with self.assertRaises(RuntimeError):
            timer.execute()
        self.assertIs(timer.p, first_child)
        self.assertEqual(len(harness.spawns), 1)
        self.assertEqual(len(harness.threads), 1)
        timer.close()

    def test_completed_timer_can_execute_a_second_child(self):
        harness = Harness()
        timer = harness.timer()
        self.assertTrue(timer.wait(.2))
        first_child = timer.p
        first_completion = timer.t1
        first_event = timer._completion
        timer.execute()
        self.assertIsNot(timer.p, first_child)
        self.assertIsNot(timer._completion, first_event)
        self.assertIsNone(timer.t1)
        self.assertEqual(timer.t0, first_completion)
        self.assertTrue(timer.wait(.2))
        self.assertAlmostEqual(timer.t1 - timer.t0, .013)
        self.assertEqual(first_child.wait_calls, 1)
        self.assertEqual(timer.p.wait_calls, 1)
        self.assertEqual(len(harness.spawns), 2)
        self.assertFalse(timer.poll())
        timer.close()

    def test_failed_spawn_cleanup_allows_retry_of_same_timer(self):
        harness = Harness()
        timer = harness.Timer(['unused-check'])
        failure = OSError('fake first spawn failed')
        harness.spawn_error = failure
        try:
            with self.assertRaises(OSError) as caught:
                timer.execute()
            self.assertIs(caught.exception, failure)
        finally:
            timer.close()
        self.assertFalse(timer.execution_state)
        self.assertIsNone(timer.p)
        harness.spawn_error = None
        timer.execute()
        self.assertTrue(timer.wait(.2))
        self.assertAlmostEqual(timer.t1 - timer.t0, .013)
        self.assertFalse(timer.poll())
        self.assertEqual(len(harness.spawns), 2)
        timer.close()

    def test_sampling_exception_signals_child_cleanup(self):
        harness = Harness()
        harness.never_exit = True
        harness.ignore_terminate = True
        failure = OSError('fake ps launch failed')
        harness.ps_error = failure
        with self.assertRaises(OSError) as caught:
            harness.make_and_run('unused-directory', checkRam=True)
        self.assertIs(caught.exception, failure)
        self.assertEqual(harness.child.terminate_calls, 1)
        self.assertEqual(harness.threads[0].join_calls, 0)

    def test_close_of_term_ignoring_child_remains_nonblocking(self):
        harness = Harness()
        harness.never_exit = True
        harness.ignore_terminate = True
        timer = harness.timer()
        now = harness.clock.now
        timer.close()
        self.assertEqual(harness.clock.now, now)
        self.assertEqual(harness.child.terminate_calls, 1)
        self.assertIsNone(harness.child.poll())
        self.assertEqual(harness.threads[0].join_calls, 0)
        self.assertFalse(harness.threads[0].finished)

    def test_close_with_kill_preserves_kill_selection(self):
        harness = Harness()
        harness.never_exit = True
        timer = harness.timer()
        timer.close(kill=True)
        self.assertEqual(harness.child.kill_calls, 1)
        self.assertEqual(harness.child.terminate_calls, 0)
        self.assertEqual(harness.threads[0].join_calls, 0)

    def test_non_ram_path_preserves_synchronous_call_and_wall_clock(self):
        harness = Harness(duration=.013)
        compiled, elapsed, ram = harness.make_and_run('unused-directory', checkRam=False)
        self.assertAlmostEqual(compiled, .01)
        self.assertAlmostEqual(elapsed, .013)
        self.assertIsNone(ram)
        self.assertEqual(harness.call_calls[0][0], './check')
        self.assertEqual(harness.threads, [])

    def test_real_thread_event_wiring_with_in_memory_fake_child(self):
        clock = Clock()
        entered = threading.Event()
        release = threading.Event()

        class Child:
            returncode = None

            def wait(self):
                entered.set()
                if not release.wait(2):
                    raise AssertionError('Test did not release fake child')
                self.returncode = 0
                return 0

            def poll(self):
                return self.returncode

            def terminate(self):
                release.set()

        child = Child()
        namespace = dict(time=clock, threading=threading,
                         subprocess=SimpleNamespace(Popen=lambda *args, **kwargs: child))
        path = SOURCE_ROOT / 'madgraph/various/misc.py'
        node = extract(path, 'ProcessTimer')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
        timer = namespace['ProcessTimer'](['unused-check'])
        try:
            timer.execute()
            self.assertTrue(entered.wait(2))
            clock.now += .013
            release.set()
            self.assertTrue(timer.wait(2))
            self.assertAlmostEqual(timer.t1 - timer.t0, .013)
            self.assertEqual(child.returncode, 0)
            self.assertFalse(timer.check_execution_state())
        finally:
            release.set()
            timer.close()
            if hasattr(timer, '_waiter'):
                timer._waiter.join(2)
        self.assertFalse(timer._waiter.is_alive())


if __name__ == '__main__':
    unittest.main(verbosity=2)
