"""Recoverable clustering failures and fatal reweighting setup errors."""

import math
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from madgraph import MG5DIR
from madgraph.core import base_objects
from madgraph.various import cluster, fxfx_ewsudakov as fx
from tests.unit_tests.ewsudakov.test_fxfx_catalogue import CatalogueDriver
from tests.unit_tests.ewsudakov import test_sudakov_factor_limits as limits


@unittest.skipIf(limits.np is None, "NumPy is required for FxFx event fixtures")
class TestFxFxFailures(unittest.TestCase):
    def setUp(self):
        fx.fks_reset_accounting()
        self.addCleanup(fx.fks_reset_accounting)
        event_id = mock.patch.object(fx, "CURRENT_EVENT_ID", 0)
        event_id.start()
        self.addCleanup(event_id.stop)
        self.fixture = limits.TestSudakovFactorLimits()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.driver = limits.FactorDriver()
        self.event = limits.ww_event([
            limits.np.array([50., 10., 5., math.sqrt(2375.)])])
        self.event.wgt = -2.
        self.event.parse_reweight()
        self.event.reweight_data = {"1001": -2., "1101": 3.}
        self.event.reweight_order = ["1001", "1101"]

    def incompatible_history(self):
        # gg -> WWg conserves charge, but the history asks for gg -> q.
        # No quark sign or flavour can make that clustering conserve charge.
        self.event[0].pid = self.event[1].pid = 21
        self.event.tag = '<clustering step_1="(1,5);mom=1"></clustering>'
        self.driver.model = {
            "particle_dict": {
                **{pid: base_objects.Particle({
                    "pdg_code": pid, "charge": 2. / 3 if pid % 2 == 0 else -1. / 3,
                    "self_antipart": False}) for pid in range(1, 7)},
                21: base_objects.Particle({"pdg_code": 21, "charge": 0., "self_antipart": True}),
                24: base_objects.Particle({"pdg_code": 24, "charge": 1., "self_antipart": False}),
            },
            "interactions": [],
        }

    def test_charge_failure_gives_unit_factors_and_counts_once(self):
        self.incompatible_history()
        source = str(self.event)
        for policy in ("accept", "unit_weight"):
            fx.fks_reset_accounting()
            fx.CURRENT_EVENT_ID = 1
            with self.subTest(policy=policy), \
                    mock.patch.object(fx, "FKS_FALLBACK_POLICY", policy), \
                    self.assertLogs(fx.logger, level="WARNING") as messages:
                for name in ("_compute_ewsudakov_fxfx_reweight",
                             "_compute_density_ewsudakov_reweight"):
                    for xi in (1., 1.5, 3.):
                        self.driver.mW2_cluster_scale = xi
                        result = getattr(self.driver, name)(self.event, None)
                        self.fixture.assert_factors(self.driver, result, [1., 1., 1.], self.event)
                self.assertEqual(fx.FKS_QUALITY_COUNTS["events_clustering_failed"], 1)
            self.assertTrue(any("Global charge conservation FAILED" in line
                                and "event 1" in line for line in messages.output))
        self.assertEqual(str(self.event), source)

    def test_math_domain_failure_gives_unit_factors_with_diagnostic(self):
        fx.CURRENT_EVENT_ID = 1
        for name in ("_compute_ewsudakov_fxfx_reweight",
                     "_compute_density_ewsudakov_reweight"):
            with self.subTest(driver=name), \
                    mock.patch.object(fx, "fxfx_merge_particles_kinematics",
                                      side_effect=ValueError("math domain error")), \
                    self.assertLogs(fx.logger, level="WARNING") as messages:
                result = getattr(self.driver, name)(self.event, None)
                self.fixture.assert_factors(self.driver, result, [1., 1., 1.], self.event)
            self.assertTrue(any("math domain error" in line for line in messages.output))
        self.assertEqual(fx.FKS_QUALITY_COUNTS["events_clustering_failed"], 1)

    def test_unexpected_clustering_errors_propagate(self):
        for error in (RuntimeError("unexpected mapping failure"),
                      ValueError("invalid mapping implementation")):
            with self.subTest(error=error), \
                    mock.patch.object(fx, "fxfx_merge_particles_kinematics", side_effect=error):
                with self.assertRaises(type(error)) as raised:
                    self.driver._cluster_fxfx_event(self.event)
                self.assertIs(raised.exception, error)
        self.assertEqual(fx.FKS_QUALITY_COUNTS["events_clustering_failed"], 0)

    def test_preparation_errors_propagate_without_exiting_python(self):
        _, tag = self.driver._cluster_fxfx_event(self.event)
        # A library with the wrong canonical flavours is an invalid setup,
        # rather than an event for which unit factors would be appropriate.
        module = SimpleNamespace(original_pdg_list_dict={tag: (tag[0], [24, 24])})
        for name in ("_compute_ewsudakov_fxfx_reweight",
                     "_compute_density_ewsudakov_reweight"):
            with self.subTest(driver=name):
                with self.assertRaisesRegex(RuntimeError, "Order in particle momenta") as raised:
                    getattr(self.driver, name)(self.event, module)
                self.assertIn("expected [24, 24]", str(raised.exception))
                self.assertIn("got [24, -24]", str(raised.exception))
        self.assertEqual(fx.FKS_QUALITY_COUNTS["events_clustering_failed"], 0)

    def test_catalogue_skips_bad_history_and_continues(self):
        self.incompatible_history()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.lhe"
            bad = str(self.event)
            self.event.tag = '<clustering step_1="(1,5);mom=21"></clustering>'
            path.write_text('<LesHouchesEvents version="3.0">\n' + bad
                            + str(self.event) + '</LesHouchesEvents>\n')
            driver = CatalogueDriver(path)
            self.addCleanup(driver.lhe_input.close)
            driver.model = self.driver.model
            driver.sudakov_xi_scan = [1.5]
            with self.assertLogs(fx.logger, level="WARNING"):
                driver._ensure_fxfx_cluster_catalogue()
            self.assertTrue(driver._fxfx_cluster_catalog_ready)
            self.assertEqual(driver.fxfx_clustered_final_states, {((21, 21), (-24, 24))})
            self.assertEqual(fx.FKS_QUALITY_COUNTS["events_clustering_failed"], 0)


class TestMulticoreCreatorFailure(unittest.TestCase):
    def test_failed_creator_stops_a_running_pickle_waiter(self):
        # The existing scheduler must cancel an active setup_f2py_interface
        # waiter, as well as the jobs that have not started yet.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "worker.py"
            script.write_text(textwrap.dedent('''\
                from pathlib import Path
                import sys
                import time

                sys.path.insert(0, sys.argv[1])
                from madgraph.interface import reweight_interface

                root = Path(sys.argv[2])
                ready = root / "waiting"
                if sys.argv[3] == "create":
                    deadline = time.monotonic() + 5
                    while not ready.exists() and time.monotonic() < deadline:
                        time.sleep(.01)
                    raise RuntimeError("catalogue creator failed before rwgt.pkl")

                interface = reweight_interface.ReweightInterface.__new__(
                    reweight_interface.ReweightInterface)
                interface.exitted = True
                interface.rwgt_dir = None
                interface.me_dir = str(root)
                interface.multicore = "wait"
                sleep = time.sleep
                def waiting(seconds):
                    ready.touch()
                    sleep(seconds)
                reweight_interface.time.sleep = waiting
                interface.setup_f2py_interface()
                raise AssertionError("waiter loaded an unpublished pickle")
                '''))
            scheduler = cluster.MultiCore(nb_core=2)
            processes, failures = [], []
            start = cluster.misc.Popen

            def record_process(*args, **kwargs):
                process = start(*args, **kwargs)
                processes.append(process)
                return process

            def wait():
                try:
                    scheduler.wait(str(root), lambda *args: None)
                except Exception as error:
                    failures.append(error)

            try:
                with mock.patch.object(cluster.misc, "Popen", side_effect=record_process):
                    for mode in ("create", "wait"):
                        scheduler.submit(sys.executable, [str(script), MG5DIR, str(root), mode],
                                         stdout=subprocess.DEVNULL)
                    thread = threading.Thread(target=wait, daemon=True)
                    thread.start()
                    thread.join(10)
                    self.assertFalse(thread.is_alive(), "creator failure left a worker waiting")
                    self.assertTrue((root / "waiting").exists())
                    self.assertEqual(len(failures), 1)
                    self.assertIn("non zero status", str(failures[0]))
                    self.assertEqual(len(processes), 2)
                    self.assertTrue(all(p.poll() is not None for p in processes))
                    self.assertTrue(all(p.returncode != 0 for p in processes))
                    self.assertFalse((root / "rw_me" / "rwgt.pkl").exists())
            finally:
                scheduler.remove()
                scheduler.lock.set()


if __name__ == "__main__":
    unittest.main()
