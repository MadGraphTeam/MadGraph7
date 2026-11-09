"""Fail closed on stale physics inputs and incomplete worker shutdown."""
from contextlib import ExitStack
import hashlib
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
import tirust_benchmark_compiler as compiler


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
        self.dependencies = {"archives": {}, "metadata": {}}
        self.flags = {"schema": compiler.SCHEMA, "environment": {}, "scopes": {
            "process": {"recipes": {"loop.f": {"compiler": [{"sha256": "compiler", "version": "test"}],
                                                    "flags": ["-O2"]}}}}}
        template = ROOT / "tools/tirust_benchmark_worker.f"
        render_parameters = {"prefix": "TEST_", "nexternal": 3, "nborn": 0}
        driver_sha256 = hashlib.sha256(benchmark.render_worker(template.read_text(), render_parameters).encode()).hexdigest()
        self.sources = {"SubProcesses/P0_example/check_sa.f": driver_sha256}
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
                "executable_sha256": benchmark.digest(subprocess / "check"), "sources": self.sources,
                "masses": [0., 0., 91.], "compiler": "GNU Fortran test compiler",
                "link_inputs_before": self.dependencies, "link_inputs_after": self.dependencies,
                "effective_compiler_flags": self.flags,
                "numerical_build_from_sources": True, "worker_template_sha256": benchmark.digest(template),
                "worker_render_parameters": render_parameters, "driver_sha256": driver_sha256,
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
                ("generated_sources", lambda output: self.sources),
                ("configure", lambda source, case, chain, config: {"mlreductionlib": "|".join(map(str, chain))}),
                ("dependency_inputs", lambda directory: self.dependencies),
                ("runtime_identity", lambda directory: {"libraries": {"/lib/common.so": "same"}, **self.dependencies}),
                ("machine_identity", lambda cpu: {"selected_cpu": cpu})):
            item = patch.object(benchmark, target, value)
            item.start()
            self.addCleanup(item.stop)
        item = patch.object(compiler, "capture_effective_flags", lambda directory, env=None: self.flags)
        item.start()
        self.addCleanup(item.stop)
        item = patch.object(benchmark.fcntl, "flock")
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

    def test_diagnostic_template_is_explicit_bound_and_ineligible(self):
        template = self.directory / "historical-worker.f"
        template.write_bytes((ROOT / "tools/tirust_benchmark_worker.f").read_bytes())
        report = self.report / "historical-start"
        report.mkdir()
        with ExitStack() as stack, patch.object(protocol, "Worker"):
            benchmark.start_worker(stack, "candidate", self.case, [9], self.config, 0,
                                   report, diagnostic_template=template)
        inputs = json.loads((report / "candidate-inputs.json").read_text())
        self.assertFalse(inputs["diagnostic_worker_template"]["performance_acceptance_eligible"])
        benchmark.verify_observation_inputs("candidate", self.case, inputs)
        template.write_text("different historical template")
        with self.assertRaisesRegex(RuntimeError, "Diagnostic worker template changed"):
            benchmark.verify_observation_inputs("candidate", self.case, inputs)

    def test_diagnostic_template_cannot_bypass_compilation_binding(self):
        template = self.directory / "wrong-worker.f"
        template.write_text("not the compiled worker")
        with ExitStack() as stack, patch.object(protocol, "Worker") as child:
            with self.assertRaisesRegex(RuntimeError, "Worker template differs"):
                benchmark.start_worker(stack, "candidate", self.case, [9], self.config, 0,
                                       self.report, diagnostic_template=template)
            child.assert_not_called()

    def test_post_observation_checks_bound_inputs(self):
        inputs = self.launch()
        self.assertEqual(benchmark.verify_observation_inputs("candidate", self.case, inputs)["points_sha256"],
                         inputs["points_sha256"])
        changed = self.directories["candidate"] / "check"
        original = changed.read_text()
        changed.write_text("different executable")
        with self.assertRaisesRegex(RuntimeError, "Executable changed"):
            benchmark.verify_observation_inputs("candidate", self.case, inputs)
        changed.write_text(original)
        with patch.object(benchmark, "runtime_identity", return_value={
                "libraries": {"/lib/common.so": "changed"}, **self.dependencies}):
            with self.assertRaisesRegex(RuntimeError, "Runtime dependencies changed"):
                benchmark.verify_observation_inputs("candidate", self.case, inputs)
        self.points.write_text("changed saved point bytes")
        with self.assertRaisesRegex(RuntimeError, "Frozen points changed"):
            benchmark.verify_observation_inputs("candidate", self.case, inputs)

    def test_post_observation_rejects_new_numerical_card_or_source(self):
        inputs = self.launch()
        card = self.outputs["candidate"] / "Cards/MadLoopParams.dat"
        card.write_text("new settings")
        with self.assertRaisesRegex(RuntimeError, "Numerical cards changed"):
            benchmark.verify_observation_inputs("candidate", self.case, inputs)
        card.unlink()
        with patch.object(benchmark, "generated_sources", return_value={"loop.f": "changed"}):
            with self.assertRaisesRegex(RuntimeError, "Generated sources changed"):
                benchmark.verify_observation_inputs("candidate", self.case, inputs)
        (self.directories["candidate"] / "param_card.dat").write_text("late shadow card")
        with self.assertRaisesRegex(RuntimeError, "Physical parameter card changed"):
            benchmark.verify_observation_inputs("candidate", self.case, inputs)

    def test_foreign_campaign_records_are_rejected_before_launch(self):
        for name in ("generation.json", "compilation.json"):
            path = self.outputs["candidate"] / name
            original = path.read_text()
            record = json.loads(original)
            record["campaign_binding"] = {"id": "another-candidate", "manifest_sha256": "different"}
            benchmark.dump(path, record)
            with self.subTest(name=name), patch.object(protocol, "Worker") as child:
                with self.assertRaisesRegex(RuntimeError, "another campaign"):
                    self.launch()
                child.assert_not_called()
            path.write_text(original)

    def test_fixed_precision_modes_do_not_enter_automatic_worker(self):
        for settings in ({"ctmoderun": 4, "ctmodeinit": 1},
                         {"ctmoderun": -1, "ctmodeinit": 4}):
            with self.subTest(settings=settings), patch.object(benchmark, "configure", return_value=settings):
                with patch.object(protocol, "Worker") as child:
                    with self.assertRaisesRegex(ValueError, "requires CTMode"):
                        self.launch()
                    child.assert_not_called()
        benchmark.validate_worker_modes({"ctmoderun": -1, "ctmodeinit": 1})

    def test_initialization_failure_retains_raw_point_and_status(self):
        raw = {"warmup.dat": "2 39 440 1.0 2.0 3.0 4.0\n"}
        with patch.object(protocol, "Worker", side_effect=protocol.MeasurementError("warmup-status 39 440", raw)):
            with ExitStack() as stack, self.assertRaises(protocol.MeasurementError):
                benchmark.start_worker(stack, "candidate", self.case, [9], self.config, 0, self.report)
        path = self.report / "candidate-initialization"
        self.assertEqual((path / "warmup.dat").read_text(), raw["warmup.dat"])
        self.assertEqual(json.loads((path / "failure.json").read_text())["error_class"], "MeasurementError")

    def test_local_parameter_card_shadow_is_rejected_before_worker(self):
        (self.directories["candidate"] / "param_card.dat").write_text("different physics\n")
        with patch.object(protocol, "Worker") as child:
            with self.assertRaisesRegex(RuntimeError, "Fortran resolves"):
                self.launch()
            child.assert_not_called()

    def test_changed_static_archive_cannot_describe_an_old_executable(self):
        with patch.object(benchmark, "dependency_inputs", return_value={"archives": {"/lib/backend.a": "new"},
                                                                      "metadata": {}}):
            with self.assertRaisesRegex(RuntimeError, "link-time evidence"):
                self.launch()

    def test_missing_link_evidence_cannot_qualify(self):
        path = self.outputs["candidate"] / "compilation.json"
        record = json.loads(path.read_text())
        del record["link_inputs_after"]
        benchmark.dump(path, record)
        with self.assertRaisesRegex(RuntimeError, "link-time evidence"):
            self.launch()

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
                "points_sha256": "different" if point_mismatch and source == "candidate" else benchmark.digest(self.points),
                "points_path": str(self.points), "points_metadata_sha256": benchmark.digest(self.points.with_suffix(".json")),
                "numerical_cards": {}, "param_card_sha256": benchmark.digest(self.outputs[source] / "Cards/param_card.dat"),
                "resolved_parameter_card": str(self.outputs[source] / "Cards/param_card.dat"),
                "generation": json.loads((self.outputs[source] / "generation.json").read_text()),
                "compilation": compilation, "runtime": {"libraries": {"/lib/common.so": checksum}, **self.dependencies}})
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

    def test_retry_selection_preserves_prior_raw_evidence(self):
        base = self.report / "runs/example"
        original = base / "tirust-first"
        original.mkdir(parents=True)
        raw = original / "retained-raw.txt"
        raw.write_text("original ineligible timing evidence\n")
        checksum = benchmark.digest(raw)
        benchmark.dump(original / "summary.json", {"acceptance_eligible": False,
            "minimum_batch_duration_met": False, "numerical_pass": True})
        retry = base / "tirust-first-attempt-2"
        benchmark.prepare_retry(base, "tirust-first", retry, "Repeat with full-duration calibration")
        retry.mkdir()
        benchmark.select_attempt(base, "tirust-first", retry, "Repeat with full-duration calibration")
        self.assertEqual(benchmark.selected_run_directory(base, "tirust-first"), retry)
        self.assertEqual(benchmark.digest(raw), checksum)
        history = json.loads((base / "attempt-selection-history.jsonl").read_text())
        self.assertEqual(history["previous_directory"], "tirust-first")
        self.assertEqual(history["directory"], "tirust-first-attempt-2")

    def test_qualified_runtime_failure_cannot_be_replaced_by_a_retry(self):
        base = self.report / "runs/example"
        benchmark.dump(base / "tirust-first/summary.json", {"acceptance_eligible": True,
            "minimum_batch_duration_met": True, "numerical_pass": True, "passed": False})
        with self.assertRaisesRegex(ValueError, "qualified result is authoritative"):
            benchmark.prepare_retry(base, "tirust-first", base / "tirust-first-attempt-2", "Try again")

    def test_retry_cannot_be_selected_without_a_prior_registered_reason(self):
        base = self.report / "runs/example"
        with self.assertRaisesRegex(ValueError, "registered before measurement"):
            benchmark.select_attempt(base, "tirust-first", base / "tirust-first-attempt-2", "Try again")

    def test_retry_selection_rejects_paths_outside_the_process(self):
        base = self.report / "runs/example"
        benchmark.dump(base / "selected-attempts.json", {
            "tirust-first": {"directory": "../different-process/tirust-first"}})
        with self.assertRaisesRegex(ValueError, "Invalid selected attempt"):
            benchmark.selected_run_directory(base, "tirust-first")

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
            def __init__(self, command, directory, log, env=None, timeout=3600):
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
