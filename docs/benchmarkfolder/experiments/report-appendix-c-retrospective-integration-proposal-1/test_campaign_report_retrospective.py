"""Invented retrospective plot capsules; no actual campaign or plotter reads.

The proposed report adapter authenticates retained bytes and never executes a
proof, plotter, native worker or Typst. All package contents below are synthetic.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def module():
    spec = importlib.util.spec_from_file_location("synthetic_retrospective_refresh", HERE / "campaign_report_refresh.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class NoReads:
    def add(self, *args, **kwargs):
        raise AssertionError("A retrospective capsule was read before admission")
    json = add


class MemoryEvidence:
    def __init__(self, entries):
        self.entries, self.reads, self.files = dict(entries), [], {}

    def add(self, name, expected=None):
        self.reads.append(name)
        raw = self.entries[name]
        if expected is not None and sha(raw) != expected:
            raise ValueError("Synthetic retrospective hash differs: " + name)
        self.files[name] = {"sha256": sha(raw), "bytes": len(raw)}
        return raw

    def json(self, name, expected=None):
        return json.loads(self.add(name, expected))


CASES = ("qcd_n2_gg_ttx", "qcd_n3_uux_zgg", "ew_n2_epem_mupmum", "qcd_n4_uux_zzgg")
BACKENDS = (1, 6, 7, 8, 9)
METRICS = ("accepted-internal-estimate", "finite-disagreement-current-cuttools")


class Capsule:
    """A 43-payload invented delivery, authenticated without executing its code."""
    def __init__(self):
        self.prefix = "experiments/synthetic-retrospective"
        self.checkpoint = {"index": "experiments/synthetic-checkpoint/index.json", "sha256": "a" * 64}
        self.outcomes, self.comparisons = {}, {}
        qp_codes = {1: 329, 6: 328, 8: 322, 9: 323}
        for case in CASES:
            accepted, comparisons = {}, {}
            for backend in BACKENDS:
                rows = []
                for point in range(1, 101):
                    qp = point > 95 and backend in qp_codes
                    rows.append({"point": point, "order": 0, "status": qp_codes[backend] if qp else 210 + backend,
                        "precision": "QP" if qp else "DP", "actual_backend": backend,
                        "accuracy": 0. if point <= 3 else 1e-9,
                        "coefficients": [1., 2., 0., 0.]})
                precision, statuses = {}, {}
                for row in rows:
                    precision[row["precision"]] = precision.get(row["precision"], 0) + 1
                    key = str(row["status"]);statuses[key] = statuses.get(key, 0) + 1
                accepted[str(backend)] = {"accepted_rows": rows, "accepted_precision_counts": precision,
                    "accepted_status_counts": statuses, "exact_zero_internal_estimates": 3, "unavailable_points": 0,
                    "retained_diagnostic": {"scope": "invented ordinary accepted post-rescue output"}}
                comparisons[str(backend)] = {"n": 100, "exact_zero_metrics": 2,
                    "exceedances_1e8": 4, "exceedances_3e6": 1}
            self.outcomes[case] = {"case": {"id": case, "process": "invented " + case},
                "accepted_outcomes": accepted, "policy": {"scope": "synthetic selected ordinary policy"}}
            self.comparisons[case] = comparisons
        self.proof_updates, self.figure_updates, self.index_updates = {}, {}, {}
        self.omitted = set()
        self.seal()

    def seal(self):
        checkpoint, plotter = self.checkpoint["sha256"], "b" * 64
        payloads = {}
        figure_files = {}
        for case in CASES:
            for metric in METRICS:
                stem = case + "-" + metric
                for suffix, raw in ((".png", b"\x89PNG\r\n\x1a\nsynthetic figure"),
                                    (".pdf", b"%PDF-1.4\n% synthetic export"),
                                    (".json", encoded({"scope": "invented plot data", "case": case, "metric": metric})),
                                    ("-points.json", encoded({"scope": "invented retained per-point statuses", "points": list(range(1, 101))}))):
                    name = stem + suffix
                    if "figures/" + name in self.omitted: continue
                    payloads["figures/" + name] = raw
                    figure_files[name] = {"sha256": sha(raw), "bytes": len(raw)}
        outcomes = encoded(self.outcomes)
        payloads["figures/accepted-outcomes.json"] = outcomes
        figure_files["accepted-outcomes.json"] = {"sha256": sha(outcomes), "bytes": len(outcomes)}
        figures = {"schema": "madloop-stability-retrospective-main-closed11-figures-v1",
            "case_ids": list(CASES), "figures": 8, "new_native_evaluations": 0, "performance_eligible": False,
            "closed_capsule_index_sha256": checkpoint, "plotter_source_index_sha256": plotter,
            "files": figure_files, **self.figure_updates}
        payloads["figures/index.json"] = encoded(figures)
        figures_sha = sha(payloads["figures/index.json"])
        payloads["comparison-summary.json"] = encoded({"schema": "madloop-stability-retrospective-main-comparison-summary-v1",
            "closed_capsule_index_sha256": checkpoint, "figures_index_sha256": figures_sha, "cases": self.comparisons})
        adapter = b'raise AssertionError("The report must not execute the retained adapter")\n'
        proof = {"schema": "madloop-stability-retrospective-main-portable-proof-v1", "passed": True,
            "negative_controls_passed": 5, "unexpected_original_or_native_accesses": 0,
            "relocated_files_unchanged": True, "new_native_evaluations": 0,
            "closed_capsule_index_sha256": checkpoint, "figures_index_sha256": figures_sha,
            "adapter_sha256": sha(adapter), "plotter_source_index_sha256": plotter,
            "verification": {"verified": True}, **self.proof_updates}
        payloads["portable-proof.json"] = encoded(proof)
        payloads.update({"adapter.py": adapter,
            "verify.py": b'raise AssertionError("The report must not execute the retained proof")\n',
            "plotter.py": b'raise AssertionError("The report must not execute the retained plotter")\n',
            "source-index.json": encoded({"scope": "invented closed input inventory"}),
            "README.md": b"Invented diagnostic delivery; no physical measurement.",
            "source/status-journal.json": encoded({"retained_failure": "synthetic controller failure"}),
            "source/points.dat": b"100\ninvented retained points"})
        self.bindings = {"checkpoint": checkpoint, "figures": figures_sha, "source": sha(payloads["source-index.json"]),
            "portable_proof": sha(payloads["portable-proof.json"]), "adapter": sha(adapter), "plotter": plotter}
        self.index = {"schema": "madloop-stability-retrospective-main-delivery-v1", "case_count": 4,
            "figures": 8, "physical_points_per_case": 100, "new_native_evaluations": 0,
            "performance_eligible": False, "figures_index_sha256": figures_sha,
            "source_index_sha256": self.bindings["source"], "portable_proof_sha256": self.bindings["portable_proof"],
            "files": {name: {"sha256": sha(raw), "bytes": len(raw)} for name, raw in payloads.items()},
            **self.index_updates}
        self.payloads = payloads
        self.refresh_index()

    def refresh_index(self):
        raw = encoded(self.index)
        self.pin = {"index": self.prefix + "/index.json", "sha256": sha(raw)}
        self.entries = {self.prefix + "/" + name: raw for name, raw in self.payloads.items()}
        self.entries[self.pin["index"]] = raw

    @contextmanager
    def admitted(self, refresh, evidence=None):
        evidence = evidence or MemoryEvidence(self.entries)
        with patch.object(refresh, "RETROSPECTIVE_PIN", self.pin), \
                patch.object(refresh, "RETROSPECTIVE_BINDINGS", self.bindings), \
                patch("subprocess.Popen", side_effect=AssertionError("Native processes forbidden")), \
                patch("ctypes.CDLL", side_effect=AssertionError("Native loading forbidden")):
            yield evidence


class SyntheticRetrospectiveCapsules(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.R = module()

    def collect(self, capsule):
        with capsule.admitted(self.R) as evidence:
            return self.R.retrospective_plots(evidence, capsule.pin, capsule.checkpoint), evidence

    def test_exact_four_cases_twenty_accepted_lanes_and_all_forty_three_files_are_bound(self):
        capsule = Capsule();before = deepcopy(capsule.outcomes)
        result, evidence = self.collect(capsule)
        self.assertEqual(len(capsule.payloads), 43)
        self.assertEqual(set(evidence.files), set(capsule.entries))
        self.assertEqual(len(evidence.files), 44)
        self.assertEqual(result["accepted_outcomes"], before)
        self.assertEqual([case["id"] for case in result["cases"]], list(CASES))
        for case in result["cases"]:
            self.assertEqual(case["process"], "invented " + case["id"])
            self.assertEqual([lane["backend"] for lane in case["lanes"]], list(BACKENDS))
            self.assertEqual([plot["metric"] for plot in case["plots"]], list(METRICS))
            for plot in case["plots"]:
                for key in ("figure_path", "export_pdf_path", "figure_data_path"):
                    self.assertIn(plot[key], evidence.files)
            for lane in case["lanes"]:
                self.assertEqual(lane["accepted_DP"] + lane["accepted_QP"], 100)
                self.assertEqual(lane["accepted_QP"], 0 if lane["backend"] == 7 else 5)
                self.assertEqual(lane["zero_estimates"], 3)
                self.assertEqual(lane["unavailable"], 0)
        self.assertEqual(len([name for name in evidence.files if name.endswith(".png")]), 8)
        self.assertEqual(len([name for name in evidence.files if name.endswith(".pdf")]), 8)
        for key in ("performance_eligible", "independent_truth_qualified", "main_campaign_modified"):
            self.assertIs(result[key], False)
        self.assertIn("ordinary accepted-output", result["scope"])
        self.assertEqual(capsule.outcomes, before)

    def test_unadmitted_pin_and_mismatched_checkpoint_reject_before_any_reads(self):
        capsule = Capsule()
        with capsule.admitted(self.R):
            for pin, checkpoint in (({**capsule.pin, "sha256": "0" * 64}, capsule.checkpoint),
                                    (capsule.pin, {**capsule.checkpoint, "sha256": "0" * 64})):
                with self.subTest(pin=pin, checkpoint=checkpoint), self.assertRaises(ValueError):
                    self.R.retrospective_plots(NoReads(), pin, checkpoint)

    def test_image_status_point_proof_and_unrendered_source_byte_drift_rejects(self):
        for name in ("figures/" + CASES[0] + "-" + METRICS[0] + ".png", "figures/accepted-outcomes.json",
                     "figures/" + CASES[0] + "-" + METRICS[0] + "-points.json", "portable-proof.json", "source/status-journal.json"):
            with self.subTest(name=name):
                capsule = Capsule();evidence = MemoryEvidence(capsule.entries)
                evidence.entries[capsule.prefix + "/" + name] += b"invented byte drift"
                with capsule.admitted(self.R, evidence), self.assertRaises(ValueError):
                    self.R.retrospective_plots(evidence, capsule.pin, capsule.checkpoint)

    def test_inventory_size_count_and_missing_figure_set_rejects(self):
        for defect in ("bytes", "count", "figure", "points"):
            with self.subTest(defect=defect):
                capsule = Capsule()
                if defect == "bytes":
                    capsule.index["files"]["source/points.dat"]["bytes"] += 1;capsule.refresh_index()
                elif defect == "count":
                    del capsule.payloads["README.md"];del capsule.index["files"]["README.md"];capsule.refresh_index()
                else:
                    suffix = ".png" if defect == "figure" else "-points.json"
                    capsule.omitted.add("figures/" + CASES[0] + "-" + METRICS[0] + suffix);capsule.seal()
                    # Keep the required inventory length so a missing image or
                    # status sidecar cannot hide behind the file-count guard.
                    raw = b"invented replacement, never a figure or point record"
                    capsule.payloads["replacement.txt"] = raw
                    capsule.index["files"]["replacement.txt"] = {"sha256": sha(raw), "bytes": len(raw)}
                    capsule.refresh_index()
                    self.assertEqual(len(capsule.index["files"]), 43)
                with capsule.admitted(self.R) as evidence, self.assertRaises(ValueError):
                    self.R.retrospective_plots(evidence, capsule.pin, capsule.checkpoint)

    def test_case_backend_point_status_and_precision_omissions_or_duplicates_reject(self):
        for defect in ("case", "backend", "points", "duplicate", "boolean_point", "boolean_order", "missing_status", "status120", "rejected420", "backend_alias", "boolean_backend", "precision", "precision_count", "status_count", "zero_count", "boolean_precision_count", "boolean_status_count", "boolean_zero_count", "boolean_unavailable"):
            with self.subTest(defect=defect):
                capsule = Capsule();lane = capsule.outcomes[CASES[0]]["accepted_outcomes"]["1"]
                if defect == "case": del capsule.outcomes[CASES[-1]]
                elif defect == "backend": del capsule.outcomes[CASES[0]]["accepted_outcomes"]["9"]
                elif defect == "points": lane["accepted_rows"].pop()
                elif defect == "duplicate": lane["accepted_rows"][1] = deepcopy(lane["accepted_rows"][0])
                elif defect == "boolean_point": lane["accepted_rows"][0]["point"] = True
                elif defect == "boolean_order": lane["accepted_rows"][0]["order"] = False
                elif defect == "missing_status": del lane["accepted_rows"][0]["status"]
                elif defect == "status120": lane["accepted_rows"][0]["status"] = 120
                elif defect == "rejected420": lane["accepted_rows"][0]["status"] = 420
                elif defect == "backend_alias": lane["accepted_rows"][0]["actual_backend"] = 9
                elif defect == "boolean_backend": lane["accepted_rows"][0]["actual_backend"] = True
                elif defect == "precision": lane["accepted_rows"][0]["precision"] = "QP"
                elif defect == "precision_count": lane["accepted_precision_counts"]["DP"] -= 1
                elif defect == "status_count": lane["accepted_status_counts"]["211"] -= 1
                elif defect == "zero_count": lane["exact_zero_internal_estimates"] -= 1
                elif defect in {"boolean_precision_count", "boolean_status_count"}:
                    for row in lane["accepted_rows"]:
                        row.update(status=329 if row["point"] == 100 else 211,
                            precision="QP" if row["point"] == 100 else "DP")
                    lane["accepted_precision_counts"] = {"DP": 99, "QP": True if defect == "boolean_precision_count" else 1}
                    lane["accepted_status_counts"] = {"211": 99, "329": True if defect == "boolean_status_count" else 1}
                elif defect == "boolean_zero_count":
                    for row in lane["accepted_rows"]: row["accuracy"] = 0. if row["point"] == 1 else 1e-9
                    lane["exact_zero_internal_estimates"] = True
                else: lane["unavailable_points"] = False
                capsule.seal()
                with capsule.admitted(self.R) as evidence, self.assertRaises(ValueError):
                    self.R.retrospective_plots(evidence, capsule.pin, capsule.checkpoint)

    def test_portable_proof_and_inventory_dependency_binding_must_match(self):
        for defect in ("passed", "controls", "unexpected", "relocated", "native", "checkpoint", "adapter", "plotter", "verified", "figure_cases"):
            with self.subTest(defect=defect):
                capsule = Capsule()
                if defect == "figure_cases": capsule.figure_updates["case_ids"] = list(reversed(CASES))
                else:
                    key, value = {"passed": ("passed", False), "controls": ("negative_controls_passed", 4),
                        "unexpected": ("unexpected_original_or_native_accesses", 1), "relocated": ("relocated_files_unchanged", False),
                        "native": ("new_native_evaluations", 1), "checkpoint": ("closed_capsule_index_sha256", "0" * 64),
                        "adapter": ("adapter_sha256", "0" * 64), "plotter": ("plotter_source_index_sha256", "0" * 64),
                        "verified": ("verification", {"verified": False})}[defect]
                    capsule.proof_updates[key] = value
                capsule.seal()
                with capsule.admitted(self.R) as evidence, self.assertRaises(ValueError):
                    self.R.retrospective_plots(evidence, capsule.pin, capsule.checkpoint)

    def test_comparison_point_counts_and_complete_case_backend_coverage_are_required(self):
        for defect in ("n", "boolean", "negative", "threshold_order", "zero_overlap", "case", "backend"):
            with self.subTest(defect=defect):
                capsule = Capsule()
                if defect == "n": capsule.comparisons[CASES[0]]["1"]["n"] = 99
                elif defect == "boolean": capsule.comparisons[CASES[0]]["1"]["exact_zero_metrics"] = True
                elif defect == "negative": capsule.comparisons[CASES[0]]["1"]["exceedances_1e8"] = -1
                elif defect == "threshold_order": capsule.comparisons[CASES[0]]["1"]["exceedances_3e6"] = 5
                elif defect == "zero_overlap": capsule.comparisons[CASES[0]]["1"]["exact_zero_metrics"] = 100
                elif defect == "case": del capsule.comparisons[CASES[-1]]
                else: del capsule.comparisons[CASES[0]]["9"]
                capsule.seal()
                with capsule.admitted(self.R) as evidence, self.assertRaises(ValueError):
                    self.R.retrospective_plots(evidence, capsule.pin, capsule.checkpoint)

    def test_exact_selected_schema_cannot_enable_performance_or_new_native_execution(self):
        for key, value in (("performance_eligible", True), ("new_native_evaluations", 1), ("physical_points_per_case", 99)):
            with self.subTest(key=key):
                capsule = Capsule();capsule.index_updates[key] = value;capsule.seal()
                with capsule.admitted(self.R) as evidence, self.assertRaises(ValueError):
                    self.R.retrospective_plots(evidence, capsule.pin, capsule.checkpoint)


class SyntheticRetrospectiveRouting(unittest.TestCase):
    def test_absent_and_null_preserve_old_output_and_present_only_adds_appendix(self):
        refresh = module()
        report = {"numerical": {"cases": [], "status": "open"}, "criteria": {"stress_accuracy": "open"},
            "delivery_complete": False, "paired": {"status": "deferred"}, "cohorts": {"status": "open"},
            "registered_source_sha256": "invented-source6"}
        before = deepcopy(report)
        checkpoint = {"index": "synthetic/checkpoint.json", "sha256": "a" * 64}
        selected = {"schema": refresh.CURRENT_SCHEMA, "closed_cases": [], "performance_status": "deferred to Hua-Sheng",
            "local_handoff_path": "synthetic-handoff", "historical_archive": {}, "implementation_packages": {},
            "numerical_checkpoint": checkpoint}
        implementation = SimpleNamespace(collect=lambda *args, **kwargs: {}, apply=lambda *args: None)
        def sibling(name):
            self.assertEqual(name, "campaign_report_implementation", "An unexpected helper was imported")
            return implementation
        appendix = {"cases": [], "performance_eligible": False, "independent_truth_qualified": False}
        pin = {"index": "synthetic/figures/evidence-index.json", "sha256": "b" * 64}
        with patch.object(refresh, "archive", return_value={}), \
                patch.object(refresh, "preserve_historical_disclosures"), \
                patch.object(refresh, "bind_index", return_value={"schema_version": 1, "new_native_evaluations": 0,
                    "files": {"analysis.json": {"sha256": refresh.N3_ANALYSIS_JSON_SHA}}}), \
                patch.object(refresh, "sibling", side_effect=sibling), \
                patch.object(refresh, "retrospective_plots", return_value=appendix) as collect:
            absent = refresh.current_supplement(NoReads(), selected, report)
            null = refresh.current_supplement(NoReads(), {**selected, "appendix_c_retrospective": None}, report)
            self.assertEqual(absent, null)
            self.assertNotIn("appendix_c_retrospective", absent)
            collect.assert_not_called()
            evidence = NoReads()
            present = refresh.current_supplement(evidence, {**selected, "appendix_c_retrospective": pin}, report)
            collect.assert_called_once_with(evidence, pin, checkpoint)
        self.assertEqual(present.pop("appendix_c_retrospective"), appendix)
        self.assertEqual(present, absent)
        self.assertEqual(report, before)
        self.assertIs(present["final_performance_accepted"], False)

    def test_unadmitted_selection_rejects_before_checkpoint_imports_or_reads(self):
        refresh = module()
        pin = {"index": "synthetic/unadmitted/evidence-index.json", "sha256": "a" * 64}
        selected = {"schema": refresh.CURRENT_SCHEMA, "campaign": "synthetic", "numerical_checkpoint": {},
            "closed_cases": [], "documents": {}, "packages": {}, "implementation_packages": {},
            "historical_archive": {}, "local_handoff_path": "synthetic", "snapshot_date": "synthetic",
            "performance_status": "deferred to Hua-Sheng", "final_performance_accepted": False,
            "current_lifecycle": None, "appendix_c_retrospective": pin}
        with patch.object(refresh, "sibling", side_effect=AssertionError("Checkpoint imported before admission")), \
                patch.object(refresh, "require_retrospective_pin", wraps=refresh.require_retrospective_pin) as admitted:
            with self.assertRaises(ValueError): refresh.current_selection(NoReads(), selected, {"id": "synthetic"}, {})
        admitted.assert_called_once_with(pin)


if __name__ == "__main__":
    unittest.main()
