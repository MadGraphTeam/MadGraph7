"""Fail closed on stale physics inputs and incomplete worker shutdown."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as benchmark
import tirust_benchmark_worker as protocol


class TestTIRustBenchmarkIntegrity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.report = self.directory / "report"
        self.case = {"id": "example", "model": "loop_sm", "process": "u u~ > z [virt=QCD]",
                     "nfinal": 1, "role": "main"}
        self.config = {"cases": [self.case], "points_per_case": 100, "repetitions": 7,
                       "minimum_batch_seconds": 1., "relative_tolerance": 3e-6,
                       "performance_upper_ratio": 1.05, "backends": [1, 6, 7, 8, 9]}
        self.outputs, self.directories = {}, {}
        for source in ("baseline", "candidate"):
            output = self.directory / source
            subprocess = output / "SubProcesses/P0_example"
            subprocess.mkdir(parents=True)
            self.outputs[source], self.directories[source] = output, subprocess
            (output / "Cards").mkdir()
            card = output / "Cards/param_card.dat"
            card.write_text("same physics card\n")
            (subprocess / "check").write_text("same executable\n")
            benchmark.dump(output / "generation.json", {
                "case": self.case, "source": source, "default_chain": "6|7|1",
                "source_identity": {"revision": benchmark.BASELINE_REVISION, "files_sha256": "source"},
                "parameter_card_sha256": benchmark.digest(card)})
            benchmark.dump(output / "compilation.json", {
                "executable_sha256": benchmark.digest(subprocess / "check"), "sources": {},
                "masses": [0., 0., 91.], "compiler": "GNU Fortran test compiler",
                "settings": {"Source/make_opts": "FC=gfortran\n"}})
        self.points = self.report / "points/example.dat"
        self.points.parent.mkdir(parents=True)
        self.points.write_text("frozen point bytes\n")
        benchmark.dump(self.points.with_suffix(".json"), {
            "sha256": benchmark.digest(self.points),
            "inputs": {"incoming_masses": [0., 0.], "masses": [91.], "count": 100}})
        benchmark.dump(self.report / "cases.json", self.config)
        for target, value in (
                ("REPORT", self.report),
                ("output_dir", lambda source, case: self.outputs[source]),
                ("subprocess_dir", lambda source, case: self.directories[source]),
                ("generated_sources", lambda output: {}),
                ("configure", lambda source, case, chain, config: {"mlreductionlib": "|".join(map(str, chain))}),
                ("runtime_identity", lambda directory: {"libraries": {"/lib/common.so": "same"}}),
                ("machine_identity", lambda cpu: {"selected_cpu": cpu})):
            item = patch.object(benchmark, target, value)
            item.start()
            self.addCleanup(item.stop)

    def launch(self):
        report = self.report / "test-start"
        report.mkdir(exist_ok=True)
        with ExitStack() as stack, patch.object(protocol, "Worker") as child:
            result = benchmark.start_worker(stack, "candidate", self.case, [9],
                                            self.config, 0, report)
            self.assertEqual(result[0], child.return_value)
        return json.loads((report / "candidate-inputs.json").read_text())

    def test_records_the_card_fortran_actually_resolves(self):
        result = self.launch()
        self.assertEqual(result["resolved_parameter_card"],
                         str(self.outputs["candidate"] / "Cards/param_card.dat"))

    def test_local_parameter_card_shadow_is_rejected_before_worker(self):
        (self.directories["candidate"] / "param_card.dat").write_text("different physics\n")
        with patch.object(protocol, "Worker") as child:
            with self.assertRaisesRegex(RuntimeError, "Fortran resolves"):
                self.launch()
            child.assert_not_called()

    def test_modified_canonical_card_and_stale_mass_points_are_rejected(self):
        card = self.outputs["candidate"] / "Cards/param_card.dat"
        original = card.read_bytes()
        card.write_text("changed mass card\n")
        with self.assertRaisesRegex(RuntimeError, "frozen generation"):
            self.launch()
        card.write_bytes(original)
        metadata = json.loads(self.points.with_suffix(".json").read_text())
        metadata["inputs"]["masses"] = [92.]
        benchmark.dump(self.points.with_suffix(".json"), metadata)
        with self.assertRaisesRegex(RuntimeError, "point masses"):
            self.launch()

    def fake_start(self, close_failure=None, point_mismatch=False, library_mismatch=False):
        self.closed = []

        def start(stack, source, case, chain, config, cpu, report):
            child = Mock(directory=self.directories[source], source=source)

            def close():
                self.closed.append(source)
                if source == close_failure:
                    raise protocol.ProtocolError("failed STOP/BYE")

            child.close.side_effect = close
            stack.callback(child.close)
            compilation = json.loads((self.outputs[source] / "compilation.json").read_text())
            checksum = "different" if library_mismatch and source == "candidate" else "same"
            benchmark.dump(report / (source + "-inputs.json"), {
                "points_sha256": "different" if point_mismatch and source == "candidate" else "same",
                "compilation": compilation, "runtime": {"libraries": {"/lib/common.so": checksum}}})
            return child, {"mlreductionlib": chain, "ctmoderun": -1}
        return start

    @staticmethod
    def observation(child, chain, loops, destination):
        return {"seconds": 1., "seconds_per_point": .01,
                "rows": [{"point": 1, "order": 0, "coefficients": [1., 2., 3., 4.]}]}

    def test_shutdown_failure_never_publishes_success(self):
        with patch.object(benchmark, "start_worker", side_effect=self.fake_start("candidate")), \
             patch.object(benchmark, "observe", side_effect=self.observation):
            with self.assertRaisesRegex(protocol.ProtocolError, "STOP/BYE"):
                benchmark.pair(self.case, [6, 7, 1], [9, 6, 7, 1], self.config, 0, "tirust-first")
        report = self.report / "runs/example/tirust-first"
        self.assertFalse((report / "summary.json").exists())
        self.assertTrue((report / "failure.json").exists())
        self.assertEqual(self.closed, ["candidate", "baseline"])

    def test_success_closes_workers_and_retains_exact_protocol_sources(self):
        with patch.object(benchmark, "start_worker", side_effect=self.fake_start()), \
             patch.object(benchmark, "observe", side_effect=self.observation):
            result = benchmark.pair(self.case, [1], [9], self.config, 0, "backend-9")
        self.assertTrue(result["passed"])
        self.assertEqual(self.closed, ["candidate", "baseline"])
        report = self.report / "runs/example/backend-9"
        specification = json.loads((report / "specification.json").read_text())
        for name, checksum in specification["protocol_sources"].items():
            self.assertEqual(benchmark.digest(report / "protocol-sources" / name), checksum)
        self.assertTrue((report / "summary.json").exists())
        self.assertFalse((report / "failure.json").exists())

    def test_reserved_label_cannot_misidentify_a_chain(self):
        for label in ("tirust-first", "backend-9"):
            with self.subTest(label=label), patch.object(benchmark, "start_worker") as start:
                with self.assertRaisesRegex(RuntimeError, "chain"):
                    benchmark.pair(self.case, [1], [1], self.config, 0, label)
                start.assert_not_called()

    def test_point_or_runtime_mismatch_is_rejected_before_observation(self):
        for field in ("point_mismatch", "library_mismatch"):
            with self.subTest(field=field), \
                 patch.object(benchmark, "start_worker", side_effect=self.fake_start(**{field: True})), \
                 patch.object(benchmark, "observe") as observe:
                with self.assertRaises(RuntimeError):
                    benchmark.pair(self.case, [1], [9], self.config, 0, field)
                observe.assert_not_called()

    def test_pair_evidence_is_accepted_by_independent_reparse(self):
        # Exercise actual start_worker, observe, pair and retained-data parser
        # together; only the numerical executable is a deterministic fake.
        class FakeWorker:
            def __init__(self, command, directory, log, env=None):
                self.directory = directory
                self.code = 219 if "candidate" in directory.parts else 211

            def run(self, loops):
                (self.directory / "dimensions.dat").write_text("0 0 0\n")
                (self.directory / "timing.dat").write_text(".1 1.0\n")
                (self.directory / "timed_status.dat").write_text(f"{self.code} {100*loops}\n")
                (self.directory / "values.dat").write_text("".join(
                    f"{point} 0 {self.code} 1 2 3 4 1e-10\n" for point in range(1, 101)))

            def close(self):
                pass

        with patch.object(protocol, "Worker", FakeWorker):
            result = benchmark.pair(self.case, [1], [9], self.config, 0, "backend-9")
        report = self.report / "runs/example/backend-9"
        validated, anchors = benchmark.validate_retained_result(report, "backend-9", self.case, self.config)
        self.assertEqual(validated, result)
        self.assertEqual(set(anchors), {"baseline", "candidate"})


if __name__ == "__main__":
    unittest.main()
