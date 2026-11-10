"""Pure retained replay and explicitly synthetic admission/coverage fixtures."""
import contextlib
import copy
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[1]
SELECTION = "experiments/current-source-report-supplement-proposal-1/selection.json"
DOSSIER = "campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-1.json"


def load(name, path):
    old = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = old


BUILD = load("source_progress_build", HERE / "proposed/build_report.py")
HELPER = load("source_progress_helper", HERE / "proposed/current_source_supplement.py")
REFRESH = load("source_progress_refresh", HERE / "proposed/campaign_report_refresh.py")
CONFIG = json.loads((DOCS / "campaigns/entry-latch-cache-guard-1/cases.json").read_text())


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


@contextlib.contextmanager
def synthetic_case(change=None):
    """Synthetic metadata only; the replay call is mocked explicitly below."""
    with tempfile.TemporaryDirectory(prefix="synthetic-report-coverage-") as directory:
        root = Path(directory)
        payload = write(root, "payload.txt", b"SYNTHETIC; no physical values\n")
        lanes = [{"source": source, "backend": backend, "status": "complete",
            "matched_saved_points": 100, "numerical_agreement_passed": True,
            "ordinary_target_passed": True, "performance_acceptance_eligible": False,
            "accepted_backend_precision_counts": {str(backend) + ":DP": 100}}
            for source, backend in HELPER.LANES]
        case = {"case": CONFIG["cases"][0]["id"], "status": "passed", "returncode": 0, "lanes": lanes}
        if change:
            change(case)
        closure = {"manifest_sha256": HELPER.MANIFEST, "performance_acceptance_eligible": False,
            "case_ids": [case["case"]], "cases": [case], "files": {
                "docs/benchmarkfolder/payload.txt": {"sha256": payload["sha256"],
                    "bytes": (root / "payload.txt").stat().st_size}}}
        closure_pin = write(root, "closure.json", closure)
        verifier = write(root, "portable/verify.py", b"# SYNTHETIC placeholder; never executed\n")
        portable_pin = write(root, "portable/index.json", {"closure_index_sha256": closure_pin["sha256"],
            "files": {"verify.py": {"sha256": verifier["sha256"],
                "bytes": (root / "portable/verify.py").stat().st_size}}})
        selected = {"closure_index": closure_pin, "replay_index": portable_pin}
        result = {"index_sha256": closure_pin["sha256"], "cases": [case["case"]], "lanes_replayed": 6,
            "native_executions": 0, "payloads_verified": 1, "performance_acceptance": False,
            "regression_and_target_passed": all(row["numerical_agreement_passed"]
                and row["ordinary_target_passed"] for row in lanes)}
        with patch.object(HELPER, "ADMITTED_NUMERICAL", (selected,)), \
                patch.object(HELPER, "replay", return_value=result):
            yield root, selected, result


class ProposedInputs(BUILD.ReportInputs):
    """Explicit candidate mapping for the NEW dossier path, not existing evidence."""
    def add(self, name, expected=None):
        if name != DOSSIER or (self.root / name).exists():
            return super().add(name, expected)
        raw = (HERE / "historical-dossier.json").read_bytes()
        identity = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        if expected is not None and identity["sha256"] != expected:
            raise ValueError("Proposed new dossier hash differs")
        self.files[name] = identity
        return raw


class TestSupplement(unittest.TestCase):
    def collect_synthetic(self, selected, root):
        return HELPER.numerical(BUILD.ReportInputs(root), [selected], {}, CONFIG)

    def test_actual_closed_capsules_only_and_no_process_or_foreign_calls(self):
        inputs = ProposedInputs(DOCS)
        before = hashlib.sha256((HERE / "historical-dossier.json").read_bytes()).hexdigest()
        with patch.object(subprocess, "Popen", side_effect=AssertionError("process launch")), \
                patch.object(ctypes, "CDLL", side_effect=AssertionError("foreign call")), \
                patch.object(ctypes, "PyDLL", side_effect=AssertionError("foreign call")):
            if sys.flags.optimize:
                with self.assertRaisesRegex(ValueError, "unoptimized Python"):
                    HELPER.validate(inputs, SELECTION, HELPER.HISTORICAL, DOSSIER)
                return
            view, context = HELPER.validate(inputs, SELECTION, HELPER.HISTORICAL, DOSSIER)
        self.assertEqual(view, json.loads((HERE / "preview.json").read_text())["source_supplement"])
        self.assertEqual((view["numerical"]["closed_cases"], view["numerical"]["closed_lanes"]), (9, 54))
        self.assertEqual(sum(row["status"] == "pending" for row in view["numerical"]["cases"]), 2)
        self.assertEqual(sum(row["accepted_dp"] for row in view["numerical"]["cases"]), 5387)
        self.assertEqual(sum(row["accepted_qp"] for row in view["numerical"]["cases"]), 13)
        n3 = next(row for row in view["numerical"]["cases"] if row["id"] == "qcd_n3_uux_zgg")
        self.assertEqual(n3["status"], "qualified")
        self.assertTrue(n3["regression_pass"])
        self.assertFalse(n3["ordinary_target_pass"])
        misses = view["numerical"]["target_misses"]
        self.assertEqual({row["backend"] for row in misses}, {6, 7, 8, 9})
        self.assertTrue(all(row["scalar_misses"] == 56 and row["distinct_points"] == 28 for row in misses))
        self.assertTrue(all(row["columns"] == {"0": 0, "1": 56, "2": 0, "3": 0} for row in misses))
        self.assertTrue(any("375 unchanged inherited baseline objects" in note
                            and "full-build receipt was not located" in note
                            for note in view["numerical"]["build_scope_notes"]))
        self.assertFalse(view["delivery_complete"])
        self.assertEqual(view["local"]["total_lanes"], 66)
        self.assertTrue(context.startswith("campaigns/entry-latch-cache-guard-1/report-supplements/"))
        self.assertNotEqual(context, "campaigns/invariant-symmetry-1/report-context.json")
        self.assertIn("PRECISION_STREAM_TODO.md", inputs.files)
        self.assertEqual(hashlib.sha256((HERE / "historical-dossier.json").read_bytes()).hexdigest(), before)
        ratios = [row["ratio"] for row in view["cache"]["rows"]]
        self.assertGreater(ratios[1], 1)
        self.assertAlmostEqual(ratios[2], 0.9968262831537023)
        self.assertAlmostEqual(ratios[3], 0.9968824621004743)

    def test_synthetic_complete_case_does_not_fill_other_ten(self):
        with synthetic_case() as (root, selected, _):
            result = self.collect_synthetic(selected, root)
        self.assertEqual(result["closed_lanes"], 6)
        self.assertEqual(result["qualified_cases"], 1)
        self.assertEqual(sum(row["status"] == "pending" for row in result["cases"]), 10)

    def test_synthetic_target_miss_retains_regression_qualification(self):
        with synthetic_case(lambda c: c["lanes"][-1].update(ordinary_target_passed=False)) as (root, selected, _):
            row = self.collect_synthetic(selected, root)["cases"][0]
        self.assertEqual(row["status"], "qualified")
        self.assertTrue(row["regression_pass"])
        self.assertFalse(row["ordinary_target_pass"])

    def test_synthetic_regression_failure_remains_visible(self):
        def fail(case):
            case.update(status="failed", returncode=1)
            case["lanes"][-1]["numerical_agreement_passed"] = False
        with synthetic_case(fail) as (root, selected, _):
            row = self.collect_synthetic(selected, root)["cases"][0]
        self.assertEqual(row["status"], "failed")
        self.assertFalse(row["regression_pass"])
        self.assertEqual(row["accepted_dp"], 600)

    def test_synthetic_duplicate_cases_are_rejected(self):
        with synthetic_case() as (root, selected, _):
            with self.assertRaisesRegex(ValueError, "Duplicate numerical"):
                HELPER.numerical(BUILD.ReportInputs(root), [selected, selected], {}, CONFIG)

    def test_synthetic_missing_lane_is_rejected(self):
        with synthetic_case(lambda c: c["lanes"].pop()) as (root, selected, _):
            with self.assertRaisesRegex(ValueError, "six-lane"):
                self.collect_synthetic(selected, root)

    def test_synthetic_reordered_backend_is_rejected(self):
        with synthetic_case(lambda c: c["lanes"].reverse()) as (root, selected, _):
            with self.assertRaisesRegex(ValueError, "six-lane"):
                self.collect_synthetic(selected, root)

    def test_synthetic_incomplete_saved_points_are_rejected(self):
        with synthetic_case(lambda c: c["lanes"][-1].update(matched_saved_points=99)) as (root, selected, _):
            with self.assertRaisesRegex(ValueError, "Unqualified numerical lane"):
                self.collect_synthetic(selected, root)

    def test_synthetic_precision_count_loss_is_rejected(self):
        with synthetic_case(lambda c: c["lanes"][-1].update(accepted_backend_precision_counts={"9:DP": 99})) as (root, selected, _):
            with self.assertRaisesRegex(ValueError, "precision counts"):
                self.collect_synthetic(selected, root)

    def test_synthetic_lane_cannot_promote_performance(self):
        with synthetic_case(lambda c: c["lanes"][-1].update(performance_acceptance_eligible=True)) as (root, selected, _):
            with self.assertRaisesRegex(ValueError, "Unqualified numerical lane"):
                self.collect_synthetic(selected, root)

    def test_synthetic_indexed_payload_corruption_is_rejected(self):
        with synthetic_case() as (root, selected, _):
            (root / "payload.txt").write_text("corrupted synthetic payload")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                self.collect_synthetic(selected, root)

    def test_synthetic_replay_pass_cannot_override_failed_lane(self):
        with synthetic_case(lambda c: c["lanes"][-1].update(ordinary_target_passed=False)) as (root, selected, result):
            result["regression_and_target_passed"] = True
            with self.assertRaisesRegex(ValueError, "pass flags differ"):
                self.collect_synthetic(selected, root)

    def test_unadmitted_checkpoint_is_rejected_before_reads(self):
        inputs = types.SimpleNamespace(json=lambda *a: self.fail("unadmitted read"))
        with self.assertRaisesRegex(ValueError, "explicitly admitted"):
            HELPER.numerical(inputs, [{"closure_index": {}, "replay_index": {}}], {}, CONFIG)

    def test_mixed_admitted_and_unknown_list_is_rejected_before_any_read(self):
        inputs = types.SimpleNamespace(json=lambda *a: self.fail("unadmitted list read"))
        selected = copy.deepcopy(HELPER.ADMITTED_NUMERICAL[0])
        unknown = copy.deepcopy(selected)
        unknown["closure_index"]["sha256"] = "0" * 64
        with patch.object(HELPER, "replay") as replay:
            with self.assertRaisesRegex(ValueError, "explicitly admitted"):
                HELPER.numerical(inputs, [selected, unknown], {}, CONFIG)
        replay.assert_not_called()

    def test_unsafe_or_old_context_is_rejected_before_proof_reads(self):
        selected = json.loads((HERE / "selection.json").read_text())
        for name in ("../escape.json", "campaigns/invariant-symmetry-1/report-context.json"):
            with self.subTest(name=name):
                current = dict(selected, context_path=name)
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    write(root, "selection.json", current)
                    with self.assertRaises(ValueError):
                        HELPER.validate(BUILD.ReportInputs(root), "selection.json", HELPER.HISTORICAL, DOSSIER)

    def test_changed_cache_pin_is_rejected_before_proof_reads(self):
        selected = json.loads((HERE / "selection.json").read_text())
        selected["cache_measurement"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write(root, "selection.json", selected)
            with self.assertRaisesRegex(ValueError, "exact admitted pins"):
                HELPER.validate(BUILD.ReportInputs(root), "selection.json", HELPER.HISTORICAL, DOSSIER)

    def test_cli_passes_explicit_supplement_without_starting_build(self):
        class Parsed(Exception):
            pass
        argv = ["build_report.py", "--campaign", HELPER.HISTORICAL,
                "--campaign-report", DOSSIER, "--supplement", SELECTION, "--verify-only"]
        with patch.object(sys, "argv", argv), patch.object(BUILD, "prepare_report", side_effect=Parsed) as prepare:
            with self.assertRaises(Parsed):
                BUILD.main()
        prepare.assert_called_once_with(BUILD.ROOT, HELPER.HISTORICAL, DOSSIER, SELECTION)

    def test_historical_dossier_changes_only_the_explicit_document_role(self):
        old = json.loads((DOCS / "campaigns/invariant-symmetry-1/report-snapshots/closed11-raw-dp-pilot-1.json").read_text())
        new = json.loads((HERE / "historical-dossier.json").read_text())
        self.assertEqual(old["criteria"], new["criteria"])
        self.assertFalse(new["delivery_complete"])
        for value in (old, new):
            value.pop("files")
            value.pop("selection_path")
            value["refresh"].pop("precision_stream_todo_path")
        self.assertEqual(old, new)

    def test_historical_role_default_retains_strict_original_path(self):
        self.assertEqual(REFRESH.precision_document({"documents": {"PRECISION_STREAM_TODO.md": "old"}}),
                         "PRECISION_STREAM_TODO.md")

    def test_archived_document_role_has_exact_old_git_bytes(self):
        selected = json.loads((HERE / "historical-selection.json").read_text())
        document = REFRESH.HISTORICAL_PRECISION_DOCUMENT
        self.assertEqual(REFRESH.precision_document(selected), document["path"])
        self.assertEqual(hashlib.sha256((HERE / "historical-PRECISION_STREAM_TODO.md").read_bytes()).hexdigest(),
                         document["sha256"])
        old = json.loads((DOCS / "campaigns/invariant-symmetry-1/report-snapshots/closed11-raw-dp-pilot-selection-1.json").read_text())
        self.assertEqual({k: v for k, v in old.items() if k != "documents"},
            {k: v for k, v in selected.items() if k not in ("documents", "historical_documents")})
        self.assertEqual({k: v for k, v in old["documents"].items() if k != "PRECISION_STREAM_TODO.md"},
            {k: v for k, v in selected["documents"].items() if k != document["path"]})

    def test_unknown_changed_or_ambiguous_historical_role_is_rejected(self):
        base = json.loads((HERE / "historical-selection.json").read_text())
        mutations = [
            lambda s: s["historical_documents"].update(other={}),
            lambda s: s["historical_documents"]["precision_stream_todo"].update(sha256="0" * 64),
            lambda s: s["historical_documents"]["precision_stream_todo"].update(path="different.md"),
            lambda s: s["documents"].update({"PRECISION_STREAM_TODO.md": "old"}),
            lambda s: s["documents"].pop(REFRESH.HISTORICAL_PRECISION_DOCUMENT["path"]),
        ]
        for change in mutations:
            selected = copy.deepcopy(base)
            change(selected)
            with self.subTest(selection=selected["historical_documents"]), self.assertRaises(ValueError):
                REFRESH.precision_document(selected)

    def test_typst_scope_panel_and_appendix_preserve_original_body(self):
        old = (DOCS / "benchmark.typ").read_text()
        proposed = (HERE / "proposed/benchmark.typ").read_text()
        start = proposed.index('#if report.at("source_supplement", default: none)')
        end = proposed.index('#if campaign != none {', start)
        original_body_and_appendix = proposed[:start] + proposed[end:]
        self.assertTrue(original_body_and_appendix.startswith(old))
        self.assertIn('report.at("source_supplement", default: none)', original_body_and_appendix[len(old):])
        self.assertFalse((HERE / "proposed/style.typ").exists())


if __name__ == "__main__":
    unittest.main()
