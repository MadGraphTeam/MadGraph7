"""Synthetic report-admission tests; no real results, workers or rendering run.

The invented numbers and byte strings below exercise rejection boundaries. They
are not additional numerical observations or substitute qualification evidence.
"""
from copy import deepcopy
from decimal import Decimal
import gzip
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("candidate_diagnostics", Path(__file__).with_name("campaign_report_diagnostics.py"))
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


class SyntheticPackage:
    def __init__(self, role, documents=None, payloads=None, raw=None):
        self.index = {"schema": D.INDEX_SCHEMAS[role]}
        if role in D.SCOPES:
            self.index["scope"] = D.SCOPES[role]
        if role == "rank6_feynman_sampled":
            self.index.update(native_reducer_calls=0, additional_physics_evaluations_during_replay=0,
                reference_repositories_changed=False, collier_cache_changed=False)
        elif role == "rank6_feynman_all_components":
            self.index.update(native_reducer_calls_in_experiment=0, quadrature_calls_during_replay=0,
                old_evidence_changed=False, collier_cache_changed=False, total_floating_point_error_formally_certified=False)
        elif role == "adaptive_independent_backend8":
            self.index.update(native_executed=False, truth_or_stability_or_performance_qualified=False,
                antecedent_indices={"docs/benchmarkfolder/experiments/" + D.CLOSED["adaptive_fixed_replay"][0]: D.CLOSED["adaptive_fixed_replay"][1],
                "docs/benchmarkfolder/experiments/ew-qp-mass-prototype/combined-qualification-1/evidence-index.json": "4f2555b2d4e707e4308bda68bed306a0193b60959f90cccf8e2253bfc65cba73"})
        self.documents, self.payloads, self.bytes = documents or {}, payloads or {}, raw or {}

    def json(self, name):
        return self.documents[name]

    def raw(self, name):
        return self.bytes[name]

    def payload(self, name):
        return self.payloads[name]

    def payload_json(self, name):
        return D.decode(self.payload(name))


def comparison(residual="1e-17", lower="0", upper="1e-7"):
    return {"symmetric_residual": residual,
        "estimate_agrees_at_3e6": Decimal(residual) <= Decimal("3e-6"),
        "estimate_agrees_at_1e8": Decimal(residual) <= Decimal("1e-8"),
        "analytic_error_only_interval": {"symmetric_residual_lower": lower, "symmetric_residual_upper": upper,
            "quadrature_error_only_agreement_at_3e6": Decimal(upper) <= Decimal("3e-6"),
            "quadrature_error_only_disagreement_at_3e6": Decimal(lower) > Decimal("3e-6"),
            "excludes_finite_precision_node_weight_arithmetic_error": True}}


def sampled_fixture():
    mappings = {"scalar": (0, [0, 0, 0, 0]), "odd_time": (1, [1, 0, 0, 0]),
        "time_time": (5, [2, 0, 0, 0]), "space_space": (7, [0, 2, 0, 0]),
        "index55": (55, [1, 1, 1, 1]), "index111": (111, [1, 1, 1, 2]), "index161": (161, [3, 1, 1, 1])}
    components = {}
    for name, (index, powers) in mappings.items():
        parts = {}
        for part in ("real", "imaginary"):
            missed = index in (55, 111, 161) and (index, part) != (161, "imaginary")
            parts[part] = {"TIRust_QP": comparison(), "COLLIER": comparison("1e-3", "9e-4", "2e-3") if missed else comparison()}
        components[name] = {"index": index, "powers": powers, "parts": parts}
    controls = {"TIRust_QP": {"all_four_controls_agree_at_1e8": True}}
    analysis = {"schema": 1, "finite_truth_assigned": False, "native_reducer_calls": 0,
        "retained_backend_states": deepcopy(D.STATES), "orders": [
            {"n": n, "components": deepcopy(components), "lowrank_convention_checks_by_estimate": deepcopy(controls)} for n in (4, 6, 8, 10, 18)],
        "raw_result_sha256": {"synthetic-result": D.sha(b"synthetic quadrature record; no calculation")}}
    summary = {"schema": 1, "scope": "seven Cartesian finite components of synthetic N6/R6 fixture108 only",
        "sampled_components": 7, "sampled_real_imaginary_parts": 14, "orders_binary64": [4, 6, 8, 10, 18],
        "order_decimal80": 8, "endpoint": 18, "native_reducer_calls_in_this_experiment": 0, "performance_qualified": False,
        "total_floating_point_error_formally_certified": False, "tirust_maximum_observed_symmetric_residual": "1e-17",
        "tirust_all_observed_parts_agree_at_3e6": True, "tirust_all_observed_parts_agree_at_1e8": True,
        "lowrank_convention_checks": controls, "collier_failed_disputed_parts_at_3e6": 5, "disputed_parts": 6,
        "collier_maximum_observed_symmetric_residual": "1e-3", "tirust_maximum_analytic_error_only_upper_residual": "1e-7",
        "tirust_all_parts_agree_at_3e6_with_analytic_error_only": True,
        "collier_failed_disputed_parts_with_analytic_error_only": 5, "retained_backend_states": deepcopy(D.STATES)}
    return SyntheticPackage("rank6_feynman_sampled", {"summary.json": summary, "analysis.json": analysis},
        raw={"data/synthetic-result.json.gz": gzip.compress(b"synthetic quadrature record; no calculation")})


def all_components_fixture():
    components = []
    for index in range(210):
        parts = {}
        for offset, part in enumerate(("real", "imaginary")):
            ordinal = 2 * index + offset
            tirust = comparison("1e-19", "0", "2e-8" if ordinal < 63 else "1e-9")
            if ordinal < 90:
                collier = comparison("1e-9", "0", "2e-9")
            elif ordinal < 92:
                collier = comparison("1e-9", "0", "2e-8")
            elif ordinal < 94:
                collier = comparison("1e-7", "0", "2e-7")
            elif ordinal < 298:
                collier = comparison("1e-7", "9e-8", "2e-7")
            else:
                collier = comparison("1e-3", "9e-4", "2e-3")
            parts[part] = {"TIRust_QP": tirust, "COLLIER": collier}
        components.append({"index": index, "parts": parts})
    summaries = {}
    for backend in D.STATES:
        parts = [p[backend] for c in components for p in c["parts"].values()]
        summary = {"parts": 420, "maximum_symmetric_residual": max((p["symmetric_residual"] for p in parts), key=Decimal)}
        for name, threshold in (("3e6", "3e-6"), ("1e8", "1e-8")):
            passed = sum(p["estimate_agrees_at_" + name] for p in parts)
            agree = sum(Decimal(p["analytic_error_only_interval"]["symmetric_residual_upper"]) <= Decimal(threshold) for p in parts)
            disagree = sum(Decimal(p["analytic_error_only_interval"]["symmetric_residual_lower"]) > Decimal(threshold) for p in parts)
            summary.update({"pass_" + name: passed, "fail_" + name: 420 - passed,
                "analytic_error_only_agree_" + name: agree, "analytic_error_only_disagree_" + name: disagree,
                "analytic_error_only_inconclusive_" + name: 420 - agree - disagree})
        summaries[backend] = summary
    analysis = {"schema": 1, "scope": "all 210 Cartesian finite components of synthetic N6/rank6 fixture108",
        "execution_status": "completed estimates; total floating error not formally certified", "failure": None,
        "attempted_order": 18, "unavailable_proposed_orders": [], "retained_backend_states": deepcopy(D.STATES),
        "total_floating_point_error_formally_certified": False, "precision": {"significand_bits": 64},
        "orders": [{"n": n, "components": deepcopy(components), "summary": deepcopy(summaries)} for n in (8, 16, 18)]}
    summary = deepcopy(analysis)
    for row in summary["orders"]:
        del row["components"]
    summary.update(native_reducer_calls_in_experiment=0, performance_qualified=False)
    return SyntheticPackage("rank6_feynman_all_components", {"summary.json": summary, "analysis.json.gz": analysis})


def fixed_fixture():
    payloads = {name: ("synthetic " + name).encode() for name in (
        "attempt/case/stdout.log", "attempt/case/mass-observer.hex", "original/stdout.log", "original/mass-observer.hex", "frozen/journal-parser.py")}
    payloads.update({"attempt/execution.json": b'{"complete":false,"native_launched":true}',
        "attempt/analysis.json": b'{"failures":[[1,"point_input_or_model_bits"],[2,"point_input_or_model_bits"]]}'})
    mapping = {"new_console": "attempt/case/stdout.log", "new_journal": "attempt/case/mass-observer.hex",
        "old_console": "original/stdout.log", "old_journal": "original/mass-observer.hex", "parser": "frozen/journal-parser.py"}
    refused = [{"point": 2, "group": g, "old_status": 1, "new_status": 0, "new_errors": [],
        "actual_working_precision_bits": None, "independent_numerical_truth_available": False,
        "new_raw_slots": ["RAW_FINITE", "RAW_IR1", "RAW_IR2", "RAW_UV1"]} for g in range(1, 38)]
    served = [{"point": p, "group": g, "old_served_output_bits_preserved": True}
        for p in (1, 2) for g in range(1, 201) if not (p == 2 and g < 38)]
    value = {"schema": "adaptive-physical-same-input-diagnostic-v2-offline-classification-only",
        "invariants_passed": True, "failures": [], "observed_native_calls": 400, "refusals_now_served": 37,
        "new_W03": [], "old_W03": ["synthetic warning"] * 74, "fixed_CTMode4_is_checked": False,
        "physical_accuracy_qualified": False, "stability_or_performance_qualified": False,
        "actual_per_call_working_precision_available": False, "native_execution_performed_by_this_comparison": False,
        "observed_sums_have_no_recorded_native_omissions": True, "previously_refused_calls": refused,
        "previously_served_calls": served, "Born_bit_checks": [{"point": p, "bits_preserved": True, "flat_indices": [1, 5],
            "old": ["synthetic word"], "new": ["synthetic word"]} for p in (1, 2)],
        "point_outputs": [{arm: {"PUBLIC_STATUS": [["INTEGER", "120"]]} for arm in ("old", "new")} for p in (1, 2)],
        "offline_input_sha256": {key: D.sha(payloads[name]) for key, name in mapping.items()},
        "offline_analyzer_sha256": D.sha(b"synthetic analyzer; never executed")}
    receipt = {"original_native_attempt_repeated": False, "original_controller_record_modified": False,
        "native_executed_by_offline_work": False, "offline_python_comparison_exit_code": 0,
        "Born_flat_indices_checked_exactly_at_each_point": [1, 5], "checked_stability_qualified": False,
        "physical_accuracy_qualified": False, "performance_qualified": False, "per_call_working_precision_available": False}
    return SyntheticPackage("adaptive_fixed_replay", {"corrected-analysis.json": value, "offline-receipt.json": receipt}, payloads,
        {"offline-successor/analyze.py": b"synthetic analyzer; never executed"})


def independent_fixture():
    value = {"schema": "adaptive-fixed-QP-independent8-diagnostic-audit-v1", "failures": [],
        "physical_input_and_Born_parity_passed": True, "native_served_counts": {"8": 400, "9": 400},
        "native_integral_input_bits_identical": False, "same_input_independent_truth_qualified": False,
        "checked_stability_qualified": False, "physical_accuracy_qualified": False,
        "per_call_working_precision_available": False, "performance_qualified": False,
        "W03": {"adaptive9": [], "guarded8": []}, "identical_nonselection_card_resource_files": [str(n) for n in range(7)],
        "native_integral_input_differences": [{"point": 2, "group": g, "label": "NATIVE_SCALE", "adaptive9": ["synthetic QP"],
            "backend8": ["synthetic DP"]} for g in range(1, 201)],
        "residuals": [{"point": p, "flat_index": i, "symmetric_relative_difference": "2e-16",
            "within_1e_8": True, "within_3e_6": True} for p in (1, 2) for i in range(1, 9)]}
    return SyntheticPackage("adaptive_independent_backend8", {"analysis.json": value})


def masked_fixture():
    proof = {"failures": [], "formerly_served_calls": 489, "formerly_served_outputs_preserved": True,
        "selected_input_Born_bits_match": True, "selected_input_Born_records": 101086,
        "original_served_records_sha256": "a" * 64, "adaptive_served_records_sha256": "a" * 64,
        "original_input_Born_sha256": "b" * 64, "adaptive_input_Born_sha256": "b" * 64,
        "passes": [{arm: {"calls": 200, "mode": mode, "served": served, "refused": refused,
                "cache_enabled_true": 200, "check_pass_true": checked} for arm, served, refused in (("original", 163, 37), ("adaptive", 200, 0))}
            for mode, checked in ((4, 0), (5, 200), (4, 0))],
        "refusal_transitions": [{"old_status": 1, "new_status": 0, "new_errors": []} for _ in range(111)]}
    workers = {}
    for arm, status, served, refused, warnings in (("original", "420", 489, 111, 2), ("adaptive", "323", 600, 0, 0)):
        workers[arm + "-masked"] = {"exit_code": 0, "named_control_complete": True, "W03_count": warnings,
            "journal_summary": {"complete": True, "partial": False, "points": [
                {"point": 1, "public_status": "219", "native_calls": 0,
                    "accuracy_bits": [["R8", 1, "3D631D88E3AB0553"], ["R8", 2, "3D631D88E3AB0553"]]},
                {"point": 2, "public_status": status, "native_calls": 600, "served_calls": served, "native_refusals": refused,
                    "pass_modes": [4, 5, 4], "complete_passes": 3, "model_snapshots": 2, "rotation_label_available": False,
                    "accuracy_bits": [["R8", i, "0000000000000000" if arm == "adaptive" else "3FF0000000000000"] for i in (1, 2)]}]}}
        workers[arm + "-trapped"] = {"exit_code": -8, "named_control_complete": False, "journal_disposition": "synthetic missing journal",
            "calls": [{"index": 2, "point": 2, "returned": False, "stage": "WARMUP"}]}
    for fault in ("invalid", "divide_by_zero", "overflow"):
        workers["fault-" + fault] = {"exit_code": -8, "control_assessment": {"fault_control_demonstrated": True,
            "physical_qualification": False, "policy_failures": []}}
    value = {"schema": "checked-trap-one-experiment-offline-outcome-v1", "masked_adaptive_checked_outputs_qualified": True,
        "masked_adaptive_QP_three_pass_coverage": True, "combined_checked_trap_coverage_qualified": False,
        "exact_hardware_exception_at_trap_identified": False, "independent_physical_accuracy_qualified": False,
        "per_call_working_precision_available": False, "performance_qualified": False,
        "native_or_loader_executed_by_reanalysis": False, "original_records_modified": False,
        "raw_record_proof": proof, "workers": workers}
    audit = {"masked_checked_QP_pass": True, "combined_trap_gate_pass": False, "exact_trap_exception_code_captured": False,
        "independent_physical_truth_or_performance_qualified": False, "formerly_served_preserved": 489,
        "selected_input_Born_records": 101086, "served_output_records_sha256": "a" * 64, "selected_input_Born_sha256": "b" * 64,
        "observed_pass_modes": [4, 5, 4], "original_served_by_pass": [163, 163, 163], "original_refused_by_pass": [37, 37, 37],
        "adaptive_served_by_pass": [200, 200, 200], "adaptive_refused_by_pass": [0, 0, 0],
        "check_pass_true_by_pass": [0, 200, 0], "cache_enabled_true_by_pass": [200, 200, 200],
        "ordinary_point_both_status": 219, "ordinary_point_accuracy": 5.432914922319511e-13,
        "threshold_original_status": 420, "threshold_original_accuracy": 1,
        "threshold_adaptive_status": 323, "threshold_adaptive_accuracy": 0}
    return SyntheticPackage("adaptive_masked_trap", {"offline-analysis.json": value, "independent-audit.json": audit,
        "symbolization.json": {"exact_hardware_exception_identified": False, "instruction_at_both_fault_PCs": "DIVSD %xmm1,%xmm0; bytesf20f5ec1"}})


class SyntheticDiagnosticAdmissionTests(unittest.TestCase):
    def test_sampled_scope_and_summary_join(self):
        p = sampled_fixture()
        got = D.diagnostic(p, "rank6_feynman_sampled")
        self.assertEqual((got["components"], got["parts"]), (7, 14))
        self.assertFalse(got["whole_rank_qualified"])
        self.assertIn("Seven samples do not qualify all 210", D.note("rank6_feynman_sampled", got)["text"])
        p.documents["summary.json"]["sampled_components"] = 210
        with self.assertRaises(ValueError):
            D.diagnostic(p, "rank6_feynman_sampled")

    def test_sampled_rejects_summary_raw_and_certification_changes(self):
        for key, value in (("tirust_maximum_observed_symmetric_residual", "0"),
                           ("total_floating_point_error_formally_certified", True), ("performance_qualified", True)):
            with self.subTest(key=key):
                p = sampled_fixture()
                p.documents["summary.json"][key] = value
                with self.assertRaises(ValueError): D.diagnostic(p, "rank6_feynman_sampled")
        p = sampled_fixture()
        p.bytes["data/synthetic-result.json.gz"] = gzip.compress(b"changed synthetic bytes")
        with self.assertRaises(ValueError): D.diagnostic(p, "rank6_feynman_sampled")

    def test_all_components_estimate_and_analytic_counts_stay_distinct(self):
        p = all_components_fixture()
        got = D.diagnostic(p, "rank6_feynman_all_components")
        self.assertEqual(got["tirust_estimate_passes_1e8"], 420)
        self.assertEqual(got["tirust_analytic_error_only_agree_1e8"], 357)
        self.assertEqual(got["tirust_analytic_error_only_inconclusive_1e8"], 63)
        text = D.note("rank6_feynman_all_components", got)["text"]
        self.assertIn("420 TIRust passes at 3e-6", text)
        self.assertIn("357 agreements and 63 inconclusive", text)
        p.documents["summary.json"]["orders"][-1]["summary"]["TIRust_QP"]["analytic_error_only_agree_1e8"] = 420
        with self.assertRaises(ValueError): D.diagnostic(p, "rank6_feynman_all_components")

    def test_all_components_rejects_missing_component_and_wrong_precision(self):
        for defect in ("missing", "precision", "finite_error"):
            with self.subTest(defect=defect):
                p = all_components_fixture()
                if defect == "missing": p.documents["analysis.json.gz"]["orders"][-1]["components"].pop()
                elif defect == "precision": p.documents["summary.json"]["precision"]["significand_bits"] = 113
                else: p.documents["analysis.json.gz"]["total_floating_point_error_formally_certified"] = True
                with self.assertRaises(ValueError): D.diagnostic(p, "rank6_feynman_all_components")

    def test_fixed_status120_does_not_promote_stability_or_precision(self):
        p = fixed_fixture()
        got = D.diagnostic(p, "adaptive_fixed_replay")
        self.assertFalse(got["checked_stability_qualified"])
        self.assertFalse(got["per_call_working_precision_available"])
        self.assertFalse(got["original_controller_complete"])
        self.assertEqual(got["previously_served_bitwise_preserved"], 363)
        for key in ("fixed_CTMode4_is_checked", "actual_per_call_working_precision_available", "physical_accuracy_qualified"):
            with self.subTest(key=key):
                q = fixed_fixture();q.documents["corrected-analysis.json"][key] = True
                with self.assertRaises(ValueError): D.diagnostic(q, "adaptive_fixed_replay")

    def test_fixed_preserves_original_failure_and_same_input_coverage(self):
        for defect in ("retry", "failure", "duplicate", "served", "raw"):
            with self.subTest(defect=defect):
                p = fixed_fixture()
                if defect == "retry": p.documents["offline-receipt.json"]["original_native_attempt_repeated"] = True
                elif defect == "failure": p.payloads["attempt/execution.json"] = b'{"complete":true,"native_launched":true}'
                elif defect == "duplicate": p.documents["corrected-analysis.json"]["previously_served_calls"][0]["group"] = 2
                elif defect == "served": p.documents["corrected-analysis.json"]["previously_served_calls"][0]["old_served_output_bits_preserved"] = False
                else: p.payloads["original/stdout.log"] = b"changed synthetic input"
                with self.assertRaises(ValueError): D.diagnostic(p, "adaptive_fixed_replay")

    def test_independent_backend8_preserves_scale_difference(self):
        p = independent_fixture()
        got = D.diagnostic(p, "adaptive_independent_backend8")
        self.assertTrue(got["same_physical_inputs"])
        self.assertFalse(got["identical_native_inputs"])
        self.assertEqual(got["native_scale_differences"], 200)
        for defect in ("identical", "missing", "wrong_point", "truth", "antecedent"):
            with self.subTest(defect=defect):
                q = independent_fixture();v = q.documents["analysis.json"]
                if defect == "identical": v["native_integral_input_bits_identical"] = True
                elif defect == "missing": v["native_integral_input_differences"].pop()
                elif defect == "wrong_point": v["native_integral_input_differences"][0]["point"] = 1
                elif defect == "truth": v["same_input_independent_truth_qualified"] = True
                else: q.index["antecedent_indices"].clear()
                with self.assertRaises(ValueError): D.diagnostic(q, "adaptive_independent_backend8")

    def test_masked_success_retains_false_trap_gate(self):
        p = masked_fixture()
        got = D.diagnostic(p, "adaptive_masked_trap")
        self.assertTrue(got["masked_checked_outputs_qualified"])
        self.assertFalse(got["combined_trap_gate"])
        self.assertEqual(got["masked_qp_calls_served"], 600)
        self.assertEqual(got["previously_served_bitwise_preserved"], 489)
        self.assertIn("combined trap gate remains false", D.note("adaptive_masked_trap", got)["text"])

    def test_masked_rejects_trap_exception_rotation_and_digest_promotions(self):
        for defect in ("trap_gate", "hardware", "rotation", "trapped_exit", "journal", "digest", "pass", "accuracy"):
            with self.subTest(defect=defect):
                p = masked_fixture();v = p.documents["offline-analysis.json"]
                if defect == "trap_gate": v["combined_checked_trap_coverage_qualified"] = True
                elif defect == "hardware": p.documents["symbolization.json"]["exact_hardware_exception_identified"] = True
                elif defect == "rotation": v["workers"]["adaptive-masked"]["journal_summary"]["points"][1]["rotation_label_available"] = True
                elif defect == "trapped_exit": v["workers"]["adaptive-trapped"]["exit_code"] = 0
                elif defect == "journal": v["workers"]["adaptive-trapped"]["journal_summary"] = {}
                elif defect == "digest": v["raw_record_proof"]["adaptive_served_records_sha256"] = "c" * 64
                elif defect == "pass": v["raw_record_proof"]["passes"][1]["adaptive"]["mode"] = 4
                else: v["workers"]["adaptive-masked"]["journal_summary"]["points"][1]["accuracy_bits"][0][2] = "3FF0000000000000"
                with self.assertRaises(ValueError): D.diagnostic(p, "adaptive_masked_trap")

    def test_unknown_schema_role_or_changed_closed_pin_is_rejected_before_reads(self):
        class NoReads:
            def add(self, *args): raise AssertionError("Unadmitted selection read evidence")
        for role in D.ROLES:
            p = D.default_selection()[role];p["sha256"] = "0" * 64
            with self.subTest(role=role):
                with self.assertRaises(ValueError): D.Package(NoReads(), p, role)
        p = fixed_fixture();p.index["schema"] = "future-schema"
        with self.assertRaises(ValueError): D.diagnostic(p, "adaptive_fixed_replay")
        with self.assertRaises(ValueError): D.diagnostic(p, "adaptive_unknown")
        with self.assertRaises(ValueError): D.collect(NoReads(), {})

    def test_strict_json_paths_and_unbounded_analytic_upper(self):
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError): D.decode(raw)
        for path in ("../escape", "/absolute", "a\\b", "a//b", ""):
            with self.subTest(path=path):
                with self.assertRaises(ValueError): D.safe(path)
        with self.assertRaises(ValueError): D.number("Infinity")
        self.assertTrue(D.upper_bound("Infinity").is_infinite())
        p = sampled_fixture()
        interval = p.documents["analysis.json"]["orders"][0]["components"]["scalar"]["parts"]["real"]["TIRust_QP"]["analytic_error_only_interval"]
        interval["symmetric_residual_upper"] = "Infinity"
        interval["quadrature_error_only_agreement_at_3e6"] = False
        D.diagnostic(p, "rank6_feynman_sampled")

    def test_synthetic_closed_loader_rejects_index_schema_and_identity_defects(self):
        # Patch the role seal only inside this synthetic loader test. Every
        # byte, path and identity here is invented and independent of real data.
        role = "adaptive_fixed_replay"
        name = "experiments/synthetic-closed-loader/index.json"
        payload = b'{"synthetic":true}'
        good = {"schema": D.INDEX_SCHEMAS[role], "scope": D.SCOPES[role],
            "files": {"synthetic.json": {"bytes": len(payload), "sha256": D.sha(payload)}}}
        class MemoryEvidence:
            def __init__(self, entries): self.entries = entries
            def add(self, key, expected=None):
                result = self.entries[key]
                if expected != D.sha(result): raise ValueError("Synthetic identity differs")
                return result
        for defect in (None, "extra_key", "size", "digest", "unsafe", "boolean_size"):
            with self.subTest(defect=defect):
                index = deepcopy(good)
                if defect == "extra_key": index["future"] = True
                elif defect == "size": index["files"]["synthetic.json"]["bytes"] += 1
                elif defect == "digest": index["files"]["synthetic.json"]["sha256"] = "0" * 64
                elif defect == "unsafe": index["files"]["../synthetic.json"] = index["files"].pop("synthetic.json")
                elif defect == "boolean_size": index["files"]["synthetic.json"]["bytes"] = True
                raw = json.dumps(index).encode()
                pin = {"index": name, "sha256": D.sha(raw)}
                evidence = MemoryEvidence({name: raw, "experiments/synthetic-closed-loader/synthetic.json": payload})
                with patch.dict(D.CLOSED, {role: ("synthetic-closed-loader/index.json", pin["sha256"])}):
                    if defect is None:
                        package = D.Package(evidence, pin, role)
                        self.assertEqual(package.json("synthetic.json"), {"synthetic": True})
                        with self.assertRaises(ValueError): package.raw("unindexed.json")
                    else:
                        with self.assertRaises(ValueError): D.Package(evidence, pin, role)

    def test_display_residual_is_compact_conservative_and_keeps_full_primitive(self):
        for exact in ("4.761122417782263584E-17", "9.62443003469919693E-20", "2.684112094650290016E-16"):
            shown = D.display_upper(exact)
            self.assertGreaterEqual(Decimal(shown), Decimal(exact))
            self.assertLess(len(shown), 12)


if __name__ == "__main__":
    unittest.main()
