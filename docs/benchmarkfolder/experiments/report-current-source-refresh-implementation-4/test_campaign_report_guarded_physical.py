"""Synthetic/memory report tests; no native worker, build or rendering run.

Every receipt, source and output below is invented. Tests patch synthetic seals
locally to exercise validation, never amend or reinterpret real physical data.
"""
from contextlib import contextmanager
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("candidate_guarded_physical", Path(__file__).with_name("campaign_report_guarded_physical.py"))
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)

ANALYZER = b'''import sys
assert __debug__
assert sys.dont_write_bytecode is True
def analyze(): return "synthetic text analysis only"
'''
LOCALIZED = b'''import sys
assert __debug__
assert sys.dont_write_bytecode is True
def verify(): return {"verified":True}
'''
VERIFIER = b'''from pathlib import Path
import importlib.util
import json
import sys
import tempfile
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
def verify():
    assert __debug__, "outer assertions must survive"
    assert __name__ != "__main__", "retained CLI must not run"
    assert HERE.is_relative_to(Path("/tmp"))
    assert tempfile.gettempdir() == "/tmp"
    assert load("synthetic_analyzer", HERE / "analyze.py").analyze() == "synthetic text analysis only"
    assert load("synthetic_localized", HERE / "localization/verify.py").verify()["verified"]
    index = json.loads((HERE / "evidence-index.json").read_bytes())
    assert {str(p.relative_to(HERE)) for p in HERE.rglob("*") if p.is_file()} == set(index["files"]) | {"evidence-index.json"}
    summary = json.loads((HERE / "summary.json").read_bytes())
    return {"verified":True,"payload_files":len(index["files"]),"text_inputs":264,
        "native_identity_only":83,"execution_complete_as_recorded":True,"native_loader_or_process_executed":False,
        "offline_result":{k:summary[k] for k in ("gate_observations","qualified_gates","checked_trap_gate_qualified",
            "forced_qp_trap_gate_qualified","fault_controls_demonstrated","independent_physical_truth_qualified",
            "per_call_working_precision_qualified","performance_qualified")}}
if __name__ == "__main__": raise AssertionError("synthetic CLI must not run")
'''


def encoded(value):
    return json.dumps(value, sort_keys=True).encode("utf-8")


def view(precision, code, calls, modes, models, word):
    return {"accepted_precision": precision, "public_code": code, "native_calls": calls, "served_calls": calls,
        "native_refusals": 0, "groups": calls, "pass_modes": modes, "model_snapshots": models,
        "accuracy_bits": [["R8", 1, word], ["R8", 2, word]]}


def paths(views):
    return {"FPE_flags_can_change_rescue_selection": True, "cleaner_flags_do_not_qualify_host_stability": True,
        "path_identity_is_not_a_numerical_acceptance_requirement": True, "per_call_working_precision_qualified": False,
        "points": [{"point": p, "before": deepcopy(v), "after": deepcopy(v), "accepted_precision_changed": False,
            "observed_call_count_changed": False, "status_or_accuracy_or_pass_usage_changed": False} for p, v in enumerate(views, 1)]}


def summary_fixture():
    workers = {}
    for mode in ("checked", "forced-qp"):
        for policy in ("masked", "trapped"):
            masked = policy == "masked"
            worker = {"kind": "physical", "mode": mode, "policy": policy, "exit_code": 0 if masked else -8,
                "control_assessment": {"actual_per_call_precision_available": False, "physical_qualification": False,
                    "physical_worker_completed_with_requested_IEEE_policy": masked, "policy_failures": []},
                "journal_disposition": "complete" if masked else "absent after interrupted worker; no observed journal or physical qualification",
                "retained_offline_assessment_failure": None if masked else {"type": "FileNotFoundError",
                    "message": "/unavailable/synthetic/" + mode + "-trapped/mass-observer.hex"},
                "numerical_assessment": None}
            workers[mode + "-" + policy] = worker
    workers["checked-masked"]["numerical_assessment"] = {"checked_outputs_qualified": True, "errors": [],
        "independent_physical_accuracy_qualified": False, "qp_coverage_observed": True, "rotation_identity_qualified": False,
        "points": [{"point": p, "status": code, "errors": [], "qp_observed": qp, "qp_checked_sequence_observed": qp,
            "rotation_identity_qualified": False} for p, code, qp in ((1, [2, 1, 9], False), (2, [3, 2, 3], True))]}
    workers["forced-qp-masked"]["numerical_assessment"] = {"errors": [], "fixed_qp_outputs_qualified": True,
        "independent_physical_accuracy_qualified": False, "per_call_working_precision_qualified": False,
        "public120_means_unchecked": True, "stability_checked": False}
    for name in ("invalid", "divide_by_zero", "overflow"):
        workers["fault-" + name] = {"kind": "fault", "exit_code": -8, "control_assessment": {
            "fault_control_demonstrated": True, "physical_qualification": False, "policy_failures": []}}
    checked_views = [view("DP", 219, 0, [], 0, "3D631D88E3AB0553"), view("QP", 323, 600, [4, 5, 4], 2, "0000000000000000")]
    fixed_views = [view("unchecked_QP", 120, 200, [4], 1, "BFF0000000000000") for _ in range(2)]
    comparisons = {name: {"numerical_agreement_reference": True, "passed": True, "ordinary_target_passed": True,
        "maximum_relative_error": 0.0, "strict_bitwise_outputs_exact": True, "path_changes": paths(views)} for name, views in (
            ("closed-adaptive-checked__checked-masked", checked_views), ("closed-adaptive-fixed__forced-qp-masked", fixed_views))}
    original = {"numerical_agreement_reference": False, "passed": True, "ordinary_target_passed": False,
        "maximum_relative_error": 0.9241097961391376, "strict_bitwise_outputs_exact": False, "path_changes": paths(checked_views)}
    original["path_changes"]["points"][1]["before"].update(public_code=420, accepted_precision="not_accepted", native_refusals=111, served_calls=489)
    comparisons["closed-original-checked__checked-masked"] = original
    observations = {}
    for mode in ("checked", "forced-qp"):
        observations[mode] = {"policy_numerical_agreement": False, **{name: {"masked": not (
            mode == "forced-qp" and name == "host_stability_acceptance"), "trapped": False} for name in (
                "IEEE_policy_worker_completion", "host_stability_acceptance", "numerical_to_closed_adaptive", "observed_QP_path_coverage")}}
    return {"schema": "guarded-adaptive-physical-closed-summary-v1", "execution_complete": True,
        "execution_sha256": G.EXECUTION_SHA256, "source_index_sha256": G.SOURCE_SHA256,
        "artifact_binding_sha256": G.BINDING_SHA256, "library_sha256": G.LIBRARY_SHA256,
        "build_receipt_sha256": G.BUILD_SHA256, "localization_evidence_index_sha256": G.LOCALIZATION_SHA256,
        "text_inputs": 264, "native_identity_only": 83, "fault_controls_demonstrated": True, **G.FALSE_FLAGS,
        "diagnostic_elapsed_seconds_not_benchmark": 1.0, "workers": workers, "comparisons": comparisons,
        "qualified_gates": {mode: {k: False for k in G.GATE_KEYS} for mode in ("checked", "forced-qp")},
        "gate_observations": observations, "trapped_instruction_sites": [
            {"case": "checked-trapped", "ELF_address": "0x11fcecb", "instruction": "divsd %xmm1,%xmm0",
                "symbol": "iregi_rust::direct::exec::gram::gram_indicator::<f64>", "not_in_cayley_cancellation_symbol": True,
                "actual_register_operands_or_exception_subtype_available": False},
            {"case": "forced-qp-trapped", "ELF_address": "0x2523cc3", "instruction": "divsd %xmm0,%xmm1",
                "symbol": "oneloop::native::dilog::series::<numerica::domains::float::multiprecision::Float>",
                "not_in_cayley_cancellation_symbol": True, "actual_register_operands_or_exception_subtype_available": False}]}


class MemoryEvidence:
    def __init__(self, entries): self.entries, self.reads = entries, []
    def add(self, name, expected=None):
        self.reads.append(name)
        raw = self.entries[name]
        if expected is not None and G.sha(raw) != expected: raise ValueError("Synthetic identity differs")
        return raw


class SyntheticPacket:
    def __init__(self, verifier=VERIFIER, analyzer=ANALYZER):
        self.summary = summary_fixture()
        identities = {"synthetic/native-" + str(i): {"payload_retained": False, "sha256": "a" * 64} for i in range(80)}
        identities.update({
            ".tirust-build/cayley-guarded-adaptive-physical-proposal-2/run1/runtime-guarded/libmadloop_tirust.so": {
                "payload_retained": False, "sha256": G.LIBRARY_SHA256},
            ".tirust-build/adaptive-qp-rescue-release-build/attempt1/runtime/libmadloop_tirust_adaptive_private.so": {
                "payload_retained": False, "sha256": G.OLD_LIBRARY_SHA256},
            ".tirust-build/qp-mass-general-implementation/combined-qualification/current-runtime/libmadloop_tirust.so": {
                "payload_retained": False, "sha256": G.MAIN_LIBRARY_SHA256}})
        self.manifest = {"schema": "guarded-physical-portable-text-v1", "native_payloads_retained": False,
            "historical_native_paths_required": False, "native_loader_or_process_executed_by_capture": False, "native_identities": identities}
        self.other = {"verify.py": verifier, "analyze.py": analyzer, "localization/verify.py": LOCALIZED,
            "payloads/synthetic.txt.gz": gzip.compress(b"synthetic frozen text; no scientific data"),
            "capture.py": b"raise AssertionError('Synthetic capturer must not run')\n"}

    @contextmanager
    def opened(self, index_edit=None, pin_edit=None):
        payloads = {**self.other, "summary.json": encoded(self.summary), "manifest.json": encoded(self.manifest)}
        for number in (1, 2):
            payloads[f"audit-attempt{number}/failure.json"] = encoded({"schema": f"guarded-portable-audit-harness-failure-v{number}",
                "exit_code": 1, "native_or_loader_executed": False, "numerical_runtime_or_captured_evidence_modified": False,
                "failure": "synthetic optional verifier-test hook denial"})
        index = {"schema": G.INDEX_SCHEMA, "scope": G.INDEX_SCOPE, "files": {
            name: {"bytes": len(raw), "sha256": G.sha(raw)} for name, raw in payloads.items()}}
        if index_edit is not None: index_edit(index)
        index_raw = encoded(index)
        prefix = str(Path(G.INDEX).parent)
        entries = {prefix + "/" + name: raw for name, raw in payloads.items()}
        entries[G.INDEX] = index_raw
        pins = {name: G.sha(payloads[name]) for name in G.PINNED_FILES}
        if pin_edit is not None: pin_edit(pins)
        with patch.object(G, "INDEX_SHA256", G.sha(index_raw)), patch.object(G, "PINNED_FILES", pins):
            yield MemoryEvidence(entries), G.default_selection()


class SyntheticGuardedPhysicalTests(unittest.TestCase):
    def test_masked_success_is_separate_from_all_false_joint_gates(self):
        packet = SyntheticPacket()
        with packet.opened() as (evidence, selected): result = G.collect(evidence, selected)
        p = result["primitives"][G.ROLE]
        self.assertEqual(p["checked_masked_statuses"], [219, 323])
        self.assertEqual(p["checked_masked_QP_calls"], 600)
        self.assertEqual(p["forced_masked_QP_calls"], 400)
        self.assertFalse(p["forced_masked_stability_checked"])
        self.assertTrue(p["masked_adaptive_reference_outputs_bitwise_exact"])
        self.assertTrue(all(v is False for row in p["qualified_gates"].values() for v in row.values()))
        self.assertEqual(p["trapped_workers_SIGFPE"], 2)
        self.assertEqual(len(p["trapped_journal_failures"]), 2)
        self.assertEqual(len(p["retained_optional_verifier_test_failures"]), 2)
        self.assertFalse(p["production_promoted"])
        self.assertFalse(result["performance_acceptance_eligible"])
        self.assertNotIn("diagnostic_elapsed_seconds_not_benchmark", p)

    def test_original_comparator_regression_pass_is_not_numerical_reference(self):
        packet = SyntheticPacket()
        with packet.opened() as (evidence, selected): result = G.collect(evidence, selected)
        original = result["primitives"][G.ROLE]["comparisons"]["closed-original-checked__checked-masked"]
        self.assertTrue(original["passed"])
        self.assertFalse(original["numerical_agreement_reference"])
        self.assertFalse(original["ordinary_target_passed"])
        self.assertEqual(original["maximum_relative_error"], 0.9241097961391376)
        self.assertIn("nonreference diagnostic", result["notes"][0]["text"])
        packet.summary["comparisons"]["closed-original-checked__checked-masked"]["numerical_agreement_reference"] = True
        with packet.opened() as (evidence, selected):
            with self.assertRaises(ValueError): G.collect(evidence, selected)

    def test_joint_truth_precision_performance_and_promotion_changes_are_rejected(self):
        for defect in (*G.FALSE_FLAGS, "qualified_gate", "promotion"):
            with self.subTest(defect=defect):
                packet = SyntheticPacket()
                if defect == "qualified_gate": packet.summary["qualified_gates"]["checked"]["numerical_agreement"] = True
                elif defect == "promotion": packet.summary["production_promoted"] = True
                else: packet.summary[defect] = True
                with packet.opened() as (evidence, selected):
                    with self.assertRaises(ValueError): G.collect(evidence, selected)

    def test_requested_qp_cannot_become_per_call_precision_or_checked_fixed120(self):
        for defect in ("per_call", "fixed_checked", "fixed_accuracy", "fixed_precision", "missing_qp"):
            with self.subTest(defect=defect):
                packet = SyntheticPacket()
                fixed = packet.summary["comparisons"]["closed-adaptive-fixed__forced-qp-masked"]
                if defect == "per_call": fixed["path_changes"]["per_call_working_precision_qualified"] = True
                elif defect == "fixed_checked": packet.summary["workers"]["forced-qp-masked"]["numerical_assessment"]["stability_checked"] = True
                elif defect == "fixed_accuracy": fixed["path_changes"]["points"][0]["after"]["accuracy_bits"][0][2] = "0000000000000000"
                elif defect == "fixed_precision": fixed["path_changes"]["points"][0]["after"]["accepted_precision"] = "QP"
                else: packet.summary["comparisons"]["closed-adaptive-checked__checked-masked"]["path_changes"]["points"][1]["after"]["native_calls"] = 0
                with packet.opened() as (evidence, selected):
                    with self.assertRaises(ValueError): G.collect(evidence, selected)

    def test_trap_failures_missing_journals_and_new_sites_remain_disclosed(self):
        for defect in ("missing_worker", "successful_trap", "journal", "missing_error", "operands", "old_cayley"):
            with self.subTest(defect=defect):
                packet = SyntheticPacket();worker = packet.summary["workers"]["checked-trapped"]
                if defect == "missing_worker": del packet.summary["workers"]["forced-qp-trapped"]
                elif defect == "successful_trap": worker["exit_code"] = 0
                elif defect == "journal": worker["journal_disposition"] = "complete"
                elif defect == "missing_error": worker["retained_offline_assessment_failure"] = None
                elif defect == "operands": packet.summary["trapped_instruction_sites"][1]["actual_register_operands_or_exception_subtype_available"] = True
                else: packet.summary["trapped_instruction_sites"][0]["symbol"] = "cayley_cancellation::<f64>"
                with packet.opened() as (evidence, selected):
                    with self.assertRaises(ValueError): G.collect(evidence, selected)

    def test_libraries_and_native_hash_only_scope_cannot_be_conflated(self):
        for defect in ("same_library", "native_payload", "native_capture"):
            with self.subTest(defect=defect):
                packet = SyntheticPacket()
                if defect == "same_library":
                    key = ".tirust-build/adaptive-qp-rescue-release-build/attempt1/runtime/libmadloop_tirust_adaptive_private.so"
                    packet.manifest["native_identities"][key]["sha256"] = G.LIBRARY_SHA256
                elif defect == "native_payload": next(iter(packet.manifest["native_identities"].values()))["payload_retained"] = True
                else: packet.manifest["native_loader_or_process_executed_by_capture"] = True
                with packet.opened() as (evidence, selected):
                    with self.assertRaises(ValueError): G.collect(evidence, selected)

    def test_index_source_input_and_verifier_drift_rejected_without_native_reads(self):
        for drift in (G.INDEX, "analyze.py", "payloads/synthetic.txt.gz", "verify.py", "summary.json"):
            with self.subTest(drift=drift):
                packet = SyntheticPacket()
                with packet.opened() as (evidence, selected):
                    name = drift if drift == G.INDEX else str(Path(G.INDEX).parent / drift)
                    evidence.entries[name] += b"synthetic changed byte"
                    with self.assertRaises(ValueError): G.collect(evidence, selected)
                    self.assertTrue(all(name in evidence.entries and not name.startswith("/") for name in evidence.reads))
        packet = SyntheticPacket()
        with packet.opened(pin_edit=lambda p: p.update({"verify.py": "0" * 64})) as (evidence, selected):
            with self.assertRaises(ValueError): G.collect(evidence, selected)

    def test_optimized_callers_rejected_before_any_proof_executes(self):
        class NoReads:
            def raw(self, name): raise AssertionError("Optimized caller read proof")
        with patch.object(G.sys, "flags", SimpleNamespace(optimize=1)):
            with self.assertRaisesRegex(ValueError, "without -O/-OO"): G.replay(NoReads())

    def test_bytecode_tempdir_and_import_path_restored_on_success_and_failure(self):
        previous_flag, previous_tempdir, previous_path = sys.dont_write_bytecode, tempfile.tempdir, list(sys.path)
        bad_analyzer = ANALYZER + b'\nsys.path.append("synthetic leaked path")\nraise AssertionError("synthetic imported proof failure")\n'
        try:
            for initial in (True, False):
                for analyzer, success in ((ANALYZER, True), (bad_analyzer, False)):
                    with self.subTest(initial=initial, success=success):
                        sys.dont_write_bytecode = initial
                        tempfile.tempdir = "/unavailable/synthetic-temp-setting"
                        packet = SyntheticPacket(analyzer=analyzer)
                        with packet.opened() as (evidence, selected):
                            if success: G.collect(evidence, selected)
                            else:
                                with self.assertRaises(ValueError): G.collect(evidence, selected)
                        self.assertIs(sys.dont_write_bytecode, initial)
                        self.assertEqual(tempfile.tempdir, "/unavailable/synthetic-temp-setting")
                        self.assertEqual(sys.path, previous_path)
        finally:
            sys.dont_write_bytecode, tempfile.tempdir = previous_flag, previous_tempdir
            sys.path[:] = previous_path

    def test_imports_create_no_bytecode_and_exact_inventory_changes_rejected(self):
        polluted = VERIFIER.replace(b'    return {"verified":True', b'    (HERE / "unexpected.pyc").write_bytes(b"synthetic pollution")\n    return {"verified":True')
        packet = SyntheticPacket(verifier=polluted)
        with packet.opened() as (evidence, selected):
            with self.assertRaises(ValueError): G.collect(evidence, selected)

    def test_caller_project_and_site_paths_are_removed_during_proof_imports(self):
        checking = ANALYZER.replace(b'assert __debug__',
            b'assert __debug__\nassert not any("synthetic-untrusted" in p for p in sys.path)')
        previous = list(sys.path)
        try:
            sys.path.extend(["/tmp/synthetic-untrusted-project", "/tmp/synthetic-untrusted-site-packages"])
            packet = SyntheticPacket(analyzer=checking)
            with packet.opened() as (evidence, selected): G.collect(evidence, selected)
            self.assertEqual(sys.path, previous + ["/tmp/synthetic-untrusted-project", "/tmp/synthetic-untrusted-site-packages"])
        finally:
            sys.path[:] = previous

    def test_unknown_schema_paths_and_native_payloads_fail_closed(self):
        for mutation in (lambda i: i.update(schema="future"), lambda i: i.update(extra=True),
                         lambda i: i["files"]["verify.py"].update(bytes=True)):
            packet = SyntheticPacket()
            with packet.opened(index_edit=mutation) as (evidence, selected):
                with self.assertRaises(ValueError): G.collect(evidence, selected)
        for name, raw in (("native.so", b"synthetic"), ("native.txt", b"\x7fELF native"),
                          ("native.txt.gz", gzip.compress(b"!<arch> synthetic")), ("__pycache__/x.pyc", b"synthetic")):
            packet = SyntheticPacket();packet.other[name] = raw
            with packet.opened() as (evidence, selected):
                with self.assertRaises(ValueError): G.collect(evidence, selected)
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'):
            with self.assertRaises(ValueError): G.decode(raw)
        for selected in ({}, {"unknown": {}}, {G.ROLE: {}, "extra": {}}):
            with self.assertRaises(ValueError): G.collect(MemoryEvidence({}), selected)


if __name__ == "__main__":
    unittest.main()
