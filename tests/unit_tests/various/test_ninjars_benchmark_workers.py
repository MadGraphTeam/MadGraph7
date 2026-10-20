"""Persistent worker protocol failures must terminate children, never reuse data."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
spec = importlib.util.spec_from_file_location(
    "ninjars_benchmark_workers", ROOT / "tools/ninjars_benchmark_workers.py")
workers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workers)


class PersistentWorkerProtocolTest(unittest.TestCase):
    def setUp(self):
        (ROOT / ".benchmark-work").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="ninjars-worker-test-",
                                                     dir=ROOT / ".benchmark-work")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def fake(self, body, timeout=2.):
        path = self.directory / "fake.py"
        path.write_text("import sys,time\n" + textwrap.dedent(body))
        return workers.Worker([sys.executable, str(path)], self.directory,
                              self.directory / "stdout.log", timeout=timeout)

    def test_diagnostic_backpressure_cannot_block_protocol(self):
        worker = self.fake("""
            with open(sys.argv[1], 'w', buffering=1) as pipe:
                sys.stdout.write('diagnostic\\n' * 100000)
                sys.stdout.flush()
                pipe.write('READY 1 100 1 1 1 .1\\n')
                for line in sys.stdin:
                    command,seq,loops = line.split()
                    if command == 'STOP':
                        pipe.write('BYE\\n')
                        break
                    pipe.write(f'DONE {seq} {loops}\\n')
        """)
        try:
            worker.run(1)
            worker.run(2)
            self.assertEqual(worker.sequence, 2)
        finally:
            worker.close()
        self.assertEqual(worker.process.returncode, 0)
        self.assertGreater((self.directory / "stdout.log").stat().st_size, 65536)
        self.assertFalse(worker.pipe.exists())

    def test_wrong_sequence_cannot_acknowledge_a_command(self):
        worker = self.fake("""
            with open(sys.argv[1], 'w', buffering=1) as pipe:
                pipe.write('READY 1 100 1 1 1 .1\\n')
                sys.stdin.readline()
                pipe.write('DONE 99 1\\n')
                time.sleep(60)
        """)
        try:
            with self.assertRaisesRegex(workers.ProtocolError, "Unexpected worker completion"):
                worker.run(1)
        finally:
            worker.close(graceful=False)
        self.assertIsNotNone(worker.process.returncode)

    def test_exit_before_ready_fails_without_hanging(self):
        with self.assertRaisesRegex(workers.ProtocolError, "exited"):
            self.fake("sys.exit(7)\n")
        self.assertFalse((self.directory / "worker-protocol.fifo").exists())

    def test_nonzero_exit_after_bye_is_not_success(self):
        worker = self.fake("""
            with open(sys.argv[1], 'w', buffering=1) as pipe:
                pipe.write('READY 1 100 1 1 1 .1\\n')
                sys.stdin.readline()
                pipe.write('BYE\\n')
            sys.exit(9)
        """)
        with self.assertRaisesRegex(workers.ProtocolError, "after BYE"):
            worker.close()
        self.assertEqual(worker.process.returncode, 9)
        self.assertFalse(worker.pipe.exists())

    def test_timeout_reaps_the_owned_child(self):
        with self.assertRaisesRegex(workers.ProtocolError, "timeout"):
            self.fake("time.sleep(60)\n", timeout=.15)
        self.assertFalse((self.directory / "worker-protocol.fifo").exists())

    def test_local_parameter_card_cannot_shadow_recorded_input(self):
        output = self.directory / "output"
        directory = output / "SubProcesses/P0"
        directory.mkdir(parents=True)
        (output / "Cards").mkdir()
        expected = b"BLOCK MASS\n 6 173.0\n"
        (output / "Cards/param_card.dat").write_bytes(expected)
        workers.validate_param_card(directory, expected)
        (directory / "param_card.dat").write_bytes(b"BLOCK MASS\n 6 150.0\n")
        with self.assertRaisesRegex(RuntimeError, "different parameter card"):
            workers.validate_param_card(directory, expected)


class PersistentSeriesAcceptanceTest(unittest.TestCase):
    def setUp(self):
        (ROOT / ".benchmark-work").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="ninjars-series-",
                                                dir=ROOT / ".benchmark-work")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.report = self.directory / "experiment"
        self.report.mkdir()
        (self.report / "sources").mkdir()
        descriptor = workers.protocol_descriptor()
        for name in workers.SOURCE_NAMES:
            (self.report / "sources" / name).write_bytes((ROOT / "tools" / name).read_bytes())
        self.identity = workers.bench.content_digest(descriptor)
        self.config = {"points_per_case": 1, "relative_tolerance": 1e-7,
                       "scale_tolerance": 1e-10,
                       "cases": [{"id": "toy", "process": "u u~ > z [virt=QCD]"}]}
        workers.bench.dump(self.report / "protocol.json", {
            "protocol_descriptor": descriptor, "protocol_identity_sha256": self.identity,
            "repetitions": 2, "configuration": self.config,
            "sources": "baseline", "backends": [6]})
        workers.bench.dump(self.report / "run-status.json",
                          {"status": "complete", "requested_cases": ["toy"],
                           "completed_cases": ["toy"]})
        self.base = {
            "case": "toy", "source": "baseline", "backend": 6, "cpu": 0,
            "benchmark_protocol": workers.SCHEME, "protocol_identity_sha256": self.identity,
            "executable_sha256": "exe", "generated_sources_sha256": "sources",
            "points_sha256": "points", "parameter_card_sha256": "card",
            "shared_numerical_settings": {}, "loops": 1, "seconds": .1,
            "seconds_per_point": .1, "status_counts": {"216": 1},
            "timed_status_counts": {"216": 1},
            "answer_dimensions": {"allocated": 1, "loop_orders": 1, "born_orders": 1},
            "rows": [{"point": 1, "order": order, "status": 216,
                      "coefficients": [1., .2, .03, -.04], "accuracy": 1e-12}
                     for order in (0, 1)]}
        workers.bench.dump(self.report / "builds/toy-baseline-6/build.json",
                          {"executable_sha256": "exe", "generated_sources_sha256": "sources"})
        for rep in ("calibration", 0, 1):
            workers.bench.dump(self.report / f"data/toy-baseline-6-{rep}.json",
                              dict(self.base, repetition=rep))

    def test_incomplete_series_cannot_be_validated(self):
        workers.checked_series(self.report)
        workers.bench.dump(self.report / "run-status.json",
                          {"status": "failed", "requested_cases": ["toy"],
                           "completed_cases": ["toy"]})
        with self.assertRaisesRegex(RuntimeError, "incomplete or failed"):
            workers.checked_series(self.report)

    def test_changed_executable_or_missing_repetition_cannot_pass(self):
        path = self.report / "data/toy-baseline-6-1.json"
        workers.bench.dump(path, dict(self.base, repetition=1, executable_sha256="other"))
        with self.assertRaisesRegex(RuntimeError, "identity changed"):
            workers.checked_series(self.report)
        path.unlink()
        with self.assertRaisesRegex(RuntimeError, "missing or noncontiguous"):
            workers.checked_series(self.report)

    def test_changed_protocol_source_requires_revalidation(self):
        path = self.report / "sources/ninjars_benchmark_worker.f"
        path.write_text(path.read_text()+"\n")
        with self.assertRaisesRegex(RuntimeError, "source differs"):
            workers.checked_series(self.report)

    def test_fresh_parity_checks_each_points_rescue_outcome(self):
        fresh = dict(self.base)
        current = dict(self.base)
        fresh["rows"] = [dict(row, point=point, status=216 if point == 1 else 328)
                         for point in (1, 2) for row in self.base["rows"]]
        current["rows"] = [dict(row, point=point, status=328 if point == 1 else 216)
                           for point in (1, 2) for row in self.base["rows"]]
        fresh["status_counts"] = current["status_counts"] = {"216": 1, "328": 1}
        with self.assertRaisesRegex(RuntimeError, "rescue outcomes"):
            workers.numerical_parity(current, fresh, self.config)

    def test_missing_backend_and_qp_coverage_cannot_be_promoted(self):
        # No fresh reference files exist, so validation must fail before
        # producing a document that could certify this partial Ninja-only run.
        output = self.directory / "coverage.json"
        with self.assertRaises((FileNotFoundError, RuntimeError)):
            workers.validate_coverage([self.report], output)
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(RuntimeError, "at least 7"):
            workers.checked_series(self.report, minimum=7, require_core=True)

    def test_failed_physics_gate_is_published_as_outstanding(self):
        output = self.directory / "published"
        source_path = self.report / "data/toy-baseline-6-0.json"
        marker = self.report / "protocol.json"
        evidence = {
            "passed": True, "schema": workers.SCHEME,
            "protocol_identity_sha256": self.identity,
            "coverage": {"dp_backends": [1, 6, 7, 8], "qp_backends": [1, 6, 8],
                         "loop_induced": True, "multiple_split_orders": True,
                         "minimum_retained_commands": 2},
            "files_sha256": {str(marker.relative_to(ROOT)): workers.digest(marker)}}
        evidence["evidence_sha256"] = workers.bench.content_digest(evidence)
        evidence_path = self.directory / "evidence.json"
        workers.bench.dump(evidence_path, evidence)
        protocol = {"configuration": self.config, "repetitions": 7,
                    "protocol_identity_sha256": self.identity}
        with mock.patch.object(workers, "checked_series",
                               return_value=(protocol, {("toy", "baseline", 6, 0):
                                                         (source_path, dict(self.base, repetition=0))})), \
                mock.patch.object(workers.bench, "REPORT", output), \
                mock.patch.object(workers.bench, "compare", side_effect=RuntimeError("physics mismatch")), \
                mock.patch.object(workers.bench, "compare_legacy"), \
                mock.patch.object(workers.bench, "compare_performance"):
            result = workers.promote(self.report, evidence_path)
        self.assertEqual(result["toy"]["agreement"]["status"], "outstanding")
        record = json.loads((output / "data/baseline-toy-6-0.json").read_text())
        self.assertTrue(record["protocol_series_complete"])
        self.assertEqual(record["protocol_validation_sha256"], evidence["evidence_sha256"])


if __name__ == "__main__":
    unittest.main()
