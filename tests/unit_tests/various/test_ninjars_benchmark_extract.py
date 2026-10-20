"""Completed-case extraction keeps failures visible and rejects incomplete evidence."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark_extract as extract


class CompletedCaseExtractionTest(unittest.TestCase):
    def setUp(self):
        (ROOT / ".benchmark-work").mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="ninjars-extract-test-",
                                                 dir=ROOT / ".benchmark-work")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.source = self.directory / "failed"
        self.destination = self.directory / "completed-cases"
        self.status = {
            "status": "failed", "acceptance_eligible": False,
            "completed_cases": ["toy-a", "toy-b"],
            "requested_cases": ["toy-a", "toy-b", "unfinished"],
            "error_type": "RuntimeError", "error": "Backend 7 returned status 410",
        }
        self.dump("run-status.json", self.status)
        descriptor = extract.workers.protocol_descriptor()
        identity = extract.workers.bench.content_digest(descriptor)
        for name in extract.workers.SOURCE_NAMES:
            extract.copy_file(ROOT / "tools" / name, self.source / "sources" / name)
        config = {
            "points_per_case": 1, "relative_tolerance": 1e-7, "scale_tolerance": 1e-10,
            "cases": [{"id": case, "process": "u u~ > z [virt=QCD]"}
                      for case in self.status["requested_cases"]],
        }
        self.dump("protocol.json", {
            "protocol_descriptor": descriptor, "protocol_identity_sha256": identity,
            "repetitions": 7, "configuration": config, "sources": "both",
            "backends": [1, 6, 7, 8],
        })
        for case in self.status["completed_cases"]:
            for source, backend in extract.TARGETS:
                code = 210 + backend
                base = {
                    "case": case, "source": source, "backend": backend, "cpu": 0,
                    "benchmark_protocol": extract.workers.SCHEME,
                    "protocol_identity_sha256": identity,
                    "executable_sha256": "e" * 64, "generated_sources_sha256": "a" * 64,
                    "points_sha256": "b" * 64, "parameter_card_sha256": "c" * 64,
                    "shared_numerical_settings": {}, "loops": 1, "seconds": .1,
                    "seconds_per_point": .1, "status_counts": {str(code): 1},
                    "timed_status_counts": {str(code): 1},
                    "answer_dimensions": {"allocated": 1, "loop_orders": 1, "born_orders": 1},
                    "rows": [{"point": 1, "order": order, "status": code,
                              "coefficients": [1., .2, .03, -.04], "accuracy": 1e-12}
                             for order in (0, 1)],
                    "measurement_run_id": "one-original-run",
                    "paired_baseline_candidate": True,
                }
                self.dump(f"builds/{case}-{source}-{backend}/build.json", {
                    "executable_sha256": "e" * 64, "generated_sources_sha256": "a" * 64,
                })
                for rep in ("calibration", *range(7)):
                    self.dump(f"data/{case}-{source}-{backend}-{rep}.json",
                              dict(base, repetition=rep))
        log = self.source / "logs/unfinished.log"
        log.parent.mkdir()
        log.write_text("retained failure output\n")
        self.dump("rejected/unfinished-failure.json", {
            "case": "unfinished", "source": "baseline", "backend": 7,
            "run_status": "failed", "diagnostic_log": str(log),
            "raw_outputs": {"timed_status.dat": "217 98\n410 2\n"},
        })
        # The entire parent snapshot must retain partial later-case evidence.
        self.dump("data/unfinished-baseline-6-0.json", {"partial": True})

    def dump(self, name, data):
        extract.workers.bench.dump(self.source / name, data)

    def test_complete_prefix_survives_later_failure_without_altering_source(self):
        before = extract.inventory(self.source)
        result = extract.extract_completed(self.source, self.destination)
        self.assertEqual(extract.inventory(self.source), before)
        self.assertEqual(result["selected_cases"], ["toy-a", "toy-b"])
        self.assertEqual(result["measurements"], 2 * 7 * 7)
        self.assertFalse(result["published"])
        derived = extract.read_json(self.destination / "run-status.json")
        self.assertEqual(derived["scope"], "completed-case-subset")
        self.assertEqual(derived["original_status"], "failed")
        self.assertEqual(extract.read_json(self.source / "run-status.json")["status"], "failed")
        self.assertTrue((self.destination / "parent-run/data/unfinished-baseline-6-0.json").is_file())
        self.assertFalse((self.destination / "data/unfinished-baseline-6-0.json").exists())
        self.assertEqual(extract.verify_extraction(self.destination), result)

    def test_empty_nonprefix_duplicate_and_unstopped_runs_are_rejected(self):
        for status in (
                dict(self.status, completed_cases=[]),
                dict(self.status, completed_cases=["toy-b"]),
                dict(self.status, completed_cases=["toy-a", "toy-a"]),
                dict(self.status, requested_cases=["toy-a", "toy-a", "unfinished"]),
                dict(self.status, status="incomplete"),
                dict(self.status, completed_cases=self.status["requested_cases"])):
            with self.subTest(status=status):
                self.dump("run-status.json", status)
                with self.assertRaises(RuntimeError):
                    extract.extract_completed(self.source, self.destination)
                self.assertFalse(self.destination.exists())

    def test_selected_case_failure_or_cleanup_failure_cannot_be_hidden(self):
        for name, data in (
                ("other-name.json", {"case": "toy-a", "error": "bad"}),
                ("toy-a-cleanup-failure.json", {"errors": ["nonzero exit"]})):
            with self.subTest(name=name):
                self.dump("rejected/" + name, data)
                with self.assertRaisesRegex(RuntimeError, "rejection or cleanup"):
                    extract.extract_completed(self.source, self.destination)
                (self.source / "rejected" / name).unlink()

    def test_missing_target_or_repetition_fails_the_frozen_validator(self):
        (self.source / "data/toy-b-candidate-8-6.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing or noncontiguous"):
            extract.extract_completed(self.source, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.directory.glob(".*.pending-*")), [])

    def test_changed_copy_is_rejected_even_if_json_meaning_is_unchanged(self):
        copy = extract.copy_file

        def corrupt(source, destination):
            copy(source, destination)
            if destination.name == "toy-a-baseline-6-0.json" and destination.parent.name == "data":
                destination.write_text(destination.read_text() + "\n")

        with mock.patch.object(extract, "copy_file", side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, "differs from source"):
                extract.extract_completed(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_source_mutation_during_extraction_is_rejected(self):
        copy = extract.copy_file
        mutated = False

        def mutate(source, destination):
            nonlocal mutated
            copy(source, destination)
            if not mutated:
                mutated = True
                (self.source / "late-change.txt").write_text("changed while extracting")

        with mock.patch.object(extract, "copy_file", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "source run changed"):
                extract.extract_completed(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_original_or_snapshot_failure_mutation_invalidates_derivation(self):
        extract.extract_completed(self.source, self.destination)
        snapshot = self.destination / "parent-run/rejected/unfinished-failure.json"
        snapshot.write_text(snapshot.read_text() + "\n")
        with self.assertRaisesRegex(RuntimeError, "differs from its derivation"):
            extract.verify_extraction(self.destination)

    def test_existing_destination_and_manual_case_selection_are_forbidden(self):
        self.destination.mkdir()
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            extract.extract_completed(self.source, self.destination)
        with self.assertRaises(TypeError):
            extract.extract_completed(self.source, self.directory / "other", cases=["toy-a"])


if __name__ == "__main__":
    unittest.main()
