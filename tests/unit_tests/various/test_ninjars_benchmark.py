"""Acceptance controls for the retained numerical/performance benchmark."""
import importlib.util
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "ninjars_benchmark", ROOT / "tools/ninjars_benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BenchmarkAcceptanceTest(unittest.TestCase):
    @staticmethod
    def persistent_fixture(root):
        report = root / "report"
        source = root / "experiment"
        names = ("ninjars_benchmark.py", "ninjars_benchmark_workers.py",
                 "ninjars_benchmark_worker.f")
        descriptor = {"schema": "persistent-worker-v1",
                      "sources": {name: hashlib.sha256(name.encode()).hexdigest()
                                  for name in names}}
        identity = benchmark.content_digest(descriptor)
        benchmark.dump(source / "protocol.json",
                       {"protocol_descriptor": descriptor, "protocol_identity_sha256": identity})
        (source / "sources").mkdir()
        for name in names:
            (source / "sources" / name).write_text(name)
        evidence_file = root / "coverage.txt"
        evidence_file.write_text("independent representative parity evidence")
        evidence = {
            "passed": True, "schema": "persistent-worker-v1",
            "protocol_identity_sha256": identity,
            "coverage": {"dp_backends": [1, 6, 7, 8], "qp_backends": [1, 6, 8],
                         "loop_induced": True, "multiple_split_orders": True,
                         "minimum_retained_commands": 2},
            "files_sha256": {"coverage.txt": hashlib.sha256(evidence_file.read_bytes()).hexdigest()}}
        validation = benchmark.content_digest(evidence)
        evidence["evidence_sha256"] = validation
        benchmark.dump(report / "inputs/worker-protocols" / (validation + ".json"), evidence)
        original = {"source": "candidate", "case": "case", "backend": 8, "repetition": 0,
                    "benchmark_protocol": "persistent-worker-v1",
                    "protocol_identity_sha256": identity, "seconds_per_point": .01,
                    "acceptance_eligible": False}
        original_path = source / "data/case-candidate-8-0.json"
        benchmark.dump(original_path, original)
        checksum = hashlib.sha256(original_path.read_bytes()).hexdigest()
        targets = [("baseline", b) for b in (1, 6, 7)] + [("candidate", b) for b in (1, 6, 7, 8)]
        manifest = {"complete": True, "benchmark_protocol": "persistent-worker-v1",
                    "protocol_identity_sha256": identity,
                    "protocol_validation_sha256": validation, "repetitions": 7,
                    "source_run": "experiment",
                    "measurements": {f"{s}-case-{b}-{r}.json": checksum
                                     for s, b in targets for r in range(7)}}
        manifest_hash = benchmark.content_digest(manifest)
        benchmark.dump(report / "inputs/persistent-series" / (manifest_hash + ".json"), manifest)
        promoted = dict(original, acceptance_eligible=True,
                        protocol_validation_sha256=validation, protocol_series_complete=True,
                        protocol_repetitions=7, protocol_run_manifest_sha256=manifest_hash,
                        protocol_source_measurement_sha256=checksum)
        return report, promoted

    def test_persistent_promotion_is_bound_to_original_timing_and_parity_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, promoted = self.persistent_fixture(root)
            with patch.object(benchmark, "ROOT", root):
                identity = benchmark.validate_benchmark_protocol(promoted, report)
                self.assertEqual(identity[0], "persistent-worker-v1")
                changed = dict(promoted, seconds_per_point=.001)
                with self.assertRaisesRegex(RuntimeError, "differs from its original"):
                    benchmark.validate_benchmark_protocol(changed, report)
                (root / "coverage.txt").write_text("modified after validation")
                with self.assertRaisesRegex(RuntimeError, "Changed protocol evidence"):
                    benchmark.validate_benchmark_protocol(promoted, report)

    def test_persistent_promotion_requires_all_backends_and_repetitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report, promoted = self.persistent_fixture(root)
            path = report / "inputs/persistent-series" / (
                promoted["protocol_run_manifest_sha256"] + ".json")
            manifest = json.loads(path.read_text())
            del manifest["measurements"]["baseline-case-1-6.json"]
            checksum = benchmark.content_digest(manifest)
            benchmark.dump(path.with_name(checksum + ".json"), manifest)
            promoted["protocol_run_manifest_sha256"] = checksum
            with patch.object(benchmark, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "omits a required backend"):
                    benchmark.validate_benchmark_protocol(promoted, report)
            with self.assertRaisesRegex(RuntimeError, "unsupported benchmark_protocol"):
                benchmark.validate_benchmark_protocol({})

    def test_performance_cannot_mix_separately_consistent_protocols(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(benchmark, "REPORT", Path(tmp)):
            identities = []
            for policy in ("first", "second"):
                descriptor = {"schema": "fresh-process-v1", "policy": policy,
                              "sources": {name: hashlib.sha256(b"fixture").hexdigest()
                                          for name in ("ninjars_benchmark.py",
                                                       "ninjars_benchmark_driver.f")}}
                identity = benchmark.content_digest(descriptor)
                directory = benchmark.REPORT / "inputs/benchmark-protocols" / identity
                benchmark.dump(directory.with_suffix(".json"), descriptor)
                directory.mkdir()
                for name in descriptor["sources"]:
                    (directory / name).write_bytes(b"fixture")
                identities.append(identity)
            for source, backend in (("baseline", 6), ("candidate", 8)):
                for rep in range(7):
                    record = {"source": source, "case": "case", "backend": backend,
                              "repetition": rep, "benchmark_protocol": "fresh-process-v1",
                              "protocol_identity_sha256": identities[0],
                              "paired_baseline_candidate": True, "measurement_run_id": "run",
                              "seconds_per_point": 1. if source == "baseline" else .9,
                              "points_sha256": "points", "parameter_card_sha256": "card",
                              "cpu": 1, "executable_sha256": source,
                              "shared_numerical_settings": {"threshold": 1e-9}}
                    benchmark.dump(benchmark.measurement_path(record), record)
            config = {"repetitions": 7, "max_runtime_ratio": 1.05, "seed": 1}
            benchmark.compare_performance({"id": "case"}, config)
            for path in (benchmark.REPORT / "data").glob("candidate-case-8-*.json"):
                record = json.loads(path.read_text())
                record["protocol_identity_sha256"] = identities[1]
                benchmark.dump(path, record)
            with self.assertRaisesRegex(RuntimeError, "different benchmark protocols"):
                benchmark.compare_performance({"id": "case"}, config)

    def test_common_timing_noise_preserves_paired_ratio(self):
        reference = [1., 1.5, .8, 1.2, 1.8, .9, 1.4]
        fast = [x * .9 for x in reference]
        result = benchmark.runtime_statistics(reference, fast, 1.05)
        self.assertEqual(result["status"], "passed")
        for endpoint in result["ratio_confidence_interval_95"]:
            self.assertAlmostEqual(endpoint, .9)
        slow = benchmark.runtime_statistics(reference, [x*1.2 for x in reference], 1.05)
        self.assertEqual(slow["status"], "failed")

    def test_uncertain_or_insufficient_measurement_cannot_pass(self):
        result = benchmark.runtime_statistics([1.]*7, [.9, .95, 1., 1.03, 1.1, 1.2, 1.3], 1.05)
        self.assertLess(result["runtime_ratio"], 1.05)
        self.assertEqual(result["status"], "unresolved")
        self.assertFalse(result["passed"])
        with self.assertRaises(RuntimeError):
            benchmark.runtime_statistics([1.]*6, [.8]*6, 1.05)

    def test_fallback_cannot_count_as_requested_backend(self):
        for backend, codes in {1: (211, 319), 6: (216, 318), 7: (217,),
                               8: (218, 312)}.items():
            for code in codes:
                benchmark.validate_status(code, backend)
        for code in (216, 318, 212, 210, 238, 418):
            with self.assertRaises(RuntimeError, msg=str(code)):
                benchmark.validate_status(code, 8)

    def test_repeated_timing_cannot_hide_changed_physics_or_executable(self):
        reference = {key: "fixed" for key in (
            "points_sha256", "parameter_card_sha256", "executable_sha256",
            "shared_numerical_settings", "answer_dimensions", "cpu")}
        reference["rows"] = [{"point": 1, "order": 0,
                              "coefficients": [1., .01, -.02, -.03]}]
        config = {"relative_tolerance": 1e-7, "scale_tolerance": 1e-10}
        changed = copy.deepcopy(reference)
        benchmark.validate_repetition(changed, reference, config)
        changed["rows"][0]["coefficients"][1] += 1e-5
        with self.assertRaisesRegex(RuntimeError, "Numerical result"):
            benchmark.validate_repetition(changed, reference, config)
        changed = copy.deepcopy(reference)
        changed["executable_sha256"] = "rebuilt"
        with self.assertRaisesRegex(RuntimeError, "executable_sha256"):
            benchmark.validate_repetition(changed, reference, config)

    def test_source_patch_is_retained_and_missing_generation_is_not_current_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(benchmark, "REPORT", root / "report"), \
                    patch.object(benchmark, "WORK", root / "work"):
                dirty_patch = b"diff --git a/generator.py b/generator.py\n"
                identity = benchmark.retain_source_identity(
                    {"status": "reconstructed", "revision": "historical"}, dirty_patch)
                retained = benchmark.REPORT / identity["dirty_patch_path"]
                self.assertEqual(retained.read_bytes(), dirty_patch)
                self.assertEqual(identity["dirty_patch_sha256"],
                                 hashlib.sha256(dirty_patch).hexdigest())
                case = {"id": "older-output"}
                benchmark.dump(benchmark.output_dir("candidate", case) / "generation.json",
                               {"seconds": 1})
                unknown = benchmark.generation_identity("candidate", case)
                self.assertEqual(unknown["status"], "not-recorded")
                self.assertIsNone(unknown["revision"])

    def test_generated_source_manifest_hashes_link_targets_and_excludes_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "Source").mkdir()
            (output / "SubProcesses").mkdir()
            source = output / "Source/model.f"
            source.write_text("      END\n")
            include = output / "Source/parameters.inc"
            include.write_text("      INTEGER N\n")
            (output / "SubProcesses/parameters.inc").symlink_to(include)
            obj = output / "Source/model.o"
            obj.write_bytes(b"first binary")
            first = benchmark.generated_source_manifest(output)
            self.assertIn("SubProcesses/parameters.inc", first["files"])
            obj.write_bytes(b"second binary")
            self.assertEqual(benchmark.generated_source_manifest(output), first)
            include.write_text("      INTEGER N,M\n")
            changed = benchmark.generated_source_manifest(output)
            self.assertNotEqual(changed["files_sha256"], first["files_sha256"])
            self.assertNotEqual(changed["files"]["SubProcesses/parameters.inc"],
                                first["files"]["SubProcesses/parameters.inc"])

    def test_rejected_repetition_is_quarantined_without_accepted_record(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(benchmark, "REPORT", Path(tmp)):
            record = {key: "fixed" for key in (
                "points_sha256", "parameter_card_sha256", "executable_sha256",
                "shared_numerical_settings", "answer_dimensions", "cpu")}
            record.update(source="candidate", case="test", backend=7, repetition=0,
                          rows=[{"point": 1, "order": 0, "coefficients": [1., 2., 3., 4.]}])
            config = {"relative_tolerance": 1e-7, "scale_tolerance": 1e-10}
            benchmark.accept_repetition(record, config)
            bad = copy.deepcopy(record)
            bad["repetition"] = 1
            bad["rows"][0]["coefficients"][1] = 20.
            # Also remove a previously written invalid record from older harnesses.
            benchmark.dump(benchmark.measurement_path(bad), bad)
            with self.assertRaisesRegex(RuntimeError, "Numerical result"):
                benchmark.accept_repetition(bad, config)
            self.assertFalse(benchmark.measurement_path(bad).exists())
            self.assertTrue(benchmark.measurement_path(record).exists())
            rejected = list((benchmark.REPORT / "rejected").glob("*.json"))
            self.assertEqual(len(rejected), 1)
            evidence = json.loads(rejected[0].read_text())
            self.assertEqual(evidence["measurement"], bad)
            self.assertIn("Numerical result", evidence["reason"])

    def test_common_stability_settings_are_written_and_case_overrides_win(self):
        from madgraph.various.banner import MadLoopParam
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(benchmark, "REPORT", root / "report"), \
                    patch.object(benchmark, "WORK", root / "work"):
                case = {"id": "settings"}
                output = benchmark.output_dir("candidate", case)
                (output / "Cards").mkdir(parents=True)
                (output / "SubProcesses").mkdir()
                card = output / "Cards/MadLoopParams.dat"
                card.write_text("#MLReductionLib\n6\n")
                config = {"madloop_parameters": {
                    "MLStabThres": 1e-9, "COLLIERRequiredAccuracy": 1e-10}}
                settings = benchmark.configure("candidate", case, 8, config=config)
                written = MadLoopParam(str(card))
                self.assertEqual(written["MLStabThres"], 1e-9)
                self.assertEqual(written["COLLIERRequiredAccuracy"], 1e-10)
                self.assertEqual(settings["mlstabthres"], 1e-9)
                case["madloop_parameters"] = {"MLStabThres": 1e-20}
                benchmark.configure("candidate", case, 8, config=config)
                written = MadLoopParam(str(card))
                self.assertEqual(written["MLStabThres"], 1e-20)
                self.assertEqual(written["COLLIERRequiredAccuracy"], 1e-10)


if __name__ == "__main__":
    unittest.main()
