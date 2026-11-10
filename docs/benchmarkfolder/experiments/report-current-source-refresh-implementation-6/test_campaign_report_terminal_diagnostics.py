"""Invented terminal stress/CMS evidence; no real capsules or workers are read.

These small virtual indices verify report admission, dependency boundaries and
retained scientific failures. All observations, sources and pins are synthetic.
"""
from contextlib import contextmanager
from copy import deepcopy
import gzip
import importlib.util
from itertools import combinations
from pathlib import Path
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("terminal_fixture_helpers", Path(__file__).with_name("test_campaign_report_current_diagnostics.py"))
F = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(F)
C = F.C
ROLE = "current_physical_stress"


def proof_source(paths, checks=None, dependency=None, extra_path=None):
    expected = set(paths)
    if extra_path:
        expected.add(extra_path)
    source = ("import hashlib\nimport json\nimport sys\nassert __debug__\n"
        "assert sys.dont_write_bytecode is True\n"
        "def verify(*, read_bytes):\n"
        f"    assert set(read_bytes.paths) == {expected!r}\n")
    for name, digest in (checks or {}).items():
        source += f"    assert hashlib.sha256(read_bytes({name!r})).hexdigest() == {digest!r}\n"
    source += ("    result = {'verified':True, 'index_sha256':hashlib.sha256(read_bytes('evidence-index.json')).hexdigest(),\n"
               "              'summary':json.loads(read_bytes('summary.json'))}\n")
    if dependency:
        source += f"    result['preparation_index_sha256'] = {dependency!r}\n"
    source += "    return result\n"
    return source.encode()


class Packet:
    def __init__(self, value, namespace, stage="prepared", preparation=None, own=None, extra_callback_path=None):
        self.value, self.namespace, self.stage, self.preparation = value, namespace, stage, preparation
        self.prefix = "experiments/synthetic-terminal/" + namespace
        self.name = self.prefix + "/evidence-index.json"
        self.payloads = {"summary.json": F.encoded(value), "extra.txt": b"synthetic nonclaim payload",
                         "logs/capture.txt.gz": gzip.compress(b"synthetic raw physical log; no alias")}
        self.payloads.update(own or {})
        self.paths = {"evidence-index.json", "verify_evidence.py", *self.payloads}
        checks = {name: F.sha(raw) for name, raw in self.payloads.items()}
        callback_paths = set(self.paths)
        if preparation:
            callback_paths |= {"preparation/" + name for name in preparation.paths}
            checks.update({"preparation/" + name: F.sha(raw) for name, raw in preparation.payloads.items()})
            checks["preparation/evidence-index.json"] = preparation.pin["sha256"]
        self.payloads["verify_evidence.py"] = proof_source(callback_paths, checks,
            preparation.pin["sha256"] if preparation else None, extra_callback_path)
        self.index = {"schema_version": 1, "files": {name: {"kind": "file", "size": len(raw), "sha256": F.sha(raw)}
            for name, raw in self.payloads.items()}}
        if preparation:
            self.index["dependencies"] = {"preparation": {"sibling": preparation.namespace,
                "index_sha256": preparation.pin["sha256"], "verifier_sha256": preparation.contract["verifier_sha256"]}}
        self.reseal()

    def reseal(self):
        self.raw_index = F.encoded(self.index)
        self.pin = {"index": self.name, "sha256": F.sha(self.raw_index)}
        self.contract = {**self.pin, "format": "stress", "stage": self.stage, "summary": "summary.json",
            "summary_sha256": F.sha(self.payloads["summary.json"]), "verifier_sha256": F.sha(self.payloads["verify_evidence.py"])}
        if self.preparation:
            self.contract.update(reader_dependency={"prefix": "preparation", "pin": deepcopy(self.preparation.pin)},
                ancestors=[deepcopy(self.preparation.pin)], setup_provenance={"kind": "inherited"})
        self.entries = {self.prefix + "/" + name: raw for name, raw in self.payloads.items()}
        self.entries[self.name] = self.raw_index


def trace(rows=None):
    rows = rows or []
    return {"rows": rows, "malformed": [], "observed_call_keys": list(range(len(rows))),
            "observed_points": sorted({row["point"] for row in rows}), "nonfinite_row_count": 0}


def terminal_fixture(preparation_sha, mode="subset"):
    requests = []
    for position, wanted in enumerate(C.expected(ROLE)):
        checked = mode == "subset" and position < 4
        state = "host_checked_complete" if checked else ("deferred_before_launch" if mode == "subset" and position == 4 else "not_started")
        final = [{"point": point, "order": order, "status": 210 + wanted["backend"], "alias_backend": wanted["backend"],
                  "alias_precision": "DP", "raw_coefficients": ["1.0", "0.0", "0.0", "0.0"],
                  "raw_accuracy": "0.0", "nonfinite_tokens": []}
                 for point in (1, 2) for order in (0, 1)] if checked else []
        warm = [{"point": 1 + ordinal % 2, "order": 0, "status": 210 + wanted["backend"], "finite": "1.0"} for ordinal in range(7)] if checked else []
        row = {"request_id": wanted["id"], **{key: wanted[key] for key in ("case", "label", "source", "backend")},
            "state": state, "host_checked_complete_stream": checked, "expected_final_points": 2,
            "qualified_final_points": [1, 2] if checked else [], "raw_final_points_observed": [1, 2] if checked else [],
            "warmup_calls_observed": len(warm), "warmup_trace": trace(warm), "final_trace": trace(final),
            "accepted_precision_counts": {"DP": 2 if checked else 0, "QP": 0}, "warnings": [] if checked else None,
            "nonfinite_rows": 0, "parser_failure": None, "native_failure": None, "controller_failure": None,
            "fatal_infrastructure_failure": None, "binding_errors": [], "coefficient_comparison_eligible": checked}
        requests.append(row)
    if mode == "lock_failure":
        requests[0].update(state="controller_failed", controller_failure={"type": "LockFailure", "message": "synthetic exclusive lock unavailable"})
    checked_rows = [row for row in requests if row["host_checked_complete_stream"]]
    comparisons = []
    for ordinal, (left, right) in enumerate(combinations(checked_rows, 2)):
        comparisons.append({"case": left["case"], "left": left["request_id"], "right": right["request_id"],
            "passed": ordinal == 0, "ordinary_target_passed": False, "tolerance": 3e-6,
            "ordinary_target_tolerance": 1e-8, "failures": [] if ordinal == 0 else [{"point": 2, "order": 1,
                "reason": "synthetic finite disagreement", "left": "1.0", "right": "2.0"}]})
    coverage = []
    for case in C.STRESS_CASES:
        rows = [row for row in checked_rows if row["case"] == case]
        pairs = [row for row in comparisons if row["case"] == case]
        subset = all(row["passed"] for row in pairs) if pairs else None
        coverage.append({"case": case, "requested_lanes": 6, "host_checked_lanes": len(rows), "six_lane_coverage": False,
            "compared_pairs": len(pairs), "compared_subset_agreement": subset, "six_lane_agreement": False})
    histogram = {}
    for row in requests:
        histogram[row["state"]] = histogram.get(row["state"], 0) + 1
    return {"role": "current-physical-stress-terminal", "scope": "four frozen physical stress streams under original strict automatic policy",
        "candidate_revision": C.REVISION, "candidate_generator_sha256": C.GENERATOR, "candidate_DSO_sha256": C.DSO,
        "campaign_manifest_sha256": C.MANIFEST, "source_index_sha256": C.STRESS_SOURCE,
        "prepared_dependency_index_sha256": preparation_sha, "requested": 24, "stage_status": "failed", "root_exit_code": 1,
        "completed_orchestration": 4 if checked_rows else 1, "host_checked_complete_streams": len(checked_rows),
        "state_counts": histogram, "request_rows": requests, "case_coverage": coverage, "comparisons": comparisons,
        "six_lane_all_cases_agree": False, "independent_physical_accuracy_qualified": False,
        "EW_analytic_oracle_qualified": False, "performance_eligible": False, "warmup_counts_are_observed_only": True}


@contextmanager
def stress_registry(value=None, mode="subset", own=None, extra_callback_path=None):
    preparation = Packet(F.stress_fixture(), "stress-preparation")
    value = deepcopy(value) if value is not None else terminal_fixture(preparation.pin["sha256"], mode)
    value["prepared_dependency_index_sha256"] = preparation.pin["sha256"]
    terminal = Packet(value, "stress-terminal", "native_terminal", preparation, own, extra_callback_path)
    evidence = F.MemoryEvidence({**preparation.entries, **terminal.entries})
    with patch.object(C, "ADMITTED", {ROLE: [preparation.contract, terminal.contract]}):
        yield evidence, preparation, terminal


def cms_second_fixture():
    value = F.cms_fixture("native_terminal")
    value.update(overall_status="incomplete", failure=None, terminal_exit_code=0, completed_requests=6,
        worker_launch_count=6, amplitude_launch_count=6, physics_loader_launch_count=1)
    value["stages"]["run"] = {"status": "passed", "exit_code": 0}
    value["gates"].update(native_run_passed=True, numerically_qualified=False)
    for position, row in enumerate(value["requests"]):
        row.update(state="startup_timeout" if position < 2 else "warmup_rejected", controller_setup_started=True,
            execution_started=True, amplitude_observed=True, numerically_qualified=False)
        F.set_phase(row, "warmup", "unexecuted" if position == 0 else "accepted")
        if position >= 2:
            rows = row["phase_outcomes"]["warmup"]
            rows[-1].update(state="rejected", status=420)
            row["phase_counts"]["warmup"].update(accepted=41, rejected=1)
        F.set_phase(row, "final", "unexecuted")
    return value


@contextmanager
def cms_registry(value=None):
    previous = F.Capsule("current_cms", F.cms_fixture("native_terminal"), namespace="cms-original-failed")
    old_entries, old_contract, old_pin = previous.build()
    current_value = deepcopy(value) if value is not None else cms_second_fixture()
    current_value["binding"].update(proposal_index_sha256="d" * 64, supervisor_index_sha256="e" * 64)
    current = F.Capsule("current_cms", current_value, namespace="cms-second-incomplete")
    entries, contract, pin = current.build()
    contract.update(ancestors=[old_pin], setup_provenance={"kind": "inherited"})
    evidence = F.MemoryEvidence({**old_entries, **entries})
    with patch.object(C, "ADMITTED", {"current_cms": [old_contract, contract]}):
        yield evidence, pin


class SyntheticCompositeReaderTests(unittest.TestCase):
    def test_composite_paths_bind_both_physical_inventories_and_indices(self):
        with stress_registry() as (evidence, preparation, terminal):
            own = C.IndexedReader(evidence, terminal.pin, terminal.contract)
            parent = C.IndexedReader(evidence, preparation.pin, preparation.contract)
            reader = C.CompositeReader(own, parent)
            expected = terminal.paths | {"preparation/" + name for name in preparation.paths}
            self.assertEqual(reader.paths, expected)
            self.assertEqual(reader("evidence-index.json"), terminal.raw_index)
            self.assertEqual(reader("preparation/evidence-index.json"), preparation.raw_index)
            self.assertEqual(reader.raw_index, terminal.raw_index)
            self.assertEqual(reader.index, terminal.index)
            C.verified_summary(evidence, ROLE, terminal.pin)

    def test_compressed_physical_names_are_not_decompressed_callback_aliases(self):
        with stress_registry() as (evidence, preparation, terminal):
            reader = C.CompositeReader(C.IndexedReader(evidence, terminal.pin, terminal.contract),
                C.IndexedReader(evidence, preparation.pin, preparation.contract))
            for prefix, packet in (("", terminal), ("preparation/", preparation)):
                self.assertEqual(reader(prefix + "logs/capture.txt.gz"), packet.payloads["logs/capture.txt.gz"])
                self.assertTrue(reader(prefix + "logs/capture.txt.gz").startswith(b"\x1f\x8b"))
                self.assertNotIn(prefix + "logs/capture.txt", reader.paths)
                with self.assertRaises(ValueError): reader(prefix + "logs/capture.txt")

    def test_unknown_missing_cyclic_or_wrong_preparation_dependencies_reject_before_reads(self):
        for defect in ("unknown", "not_ancestor", "self_cycle", "ancestor_cycle", "wrong_prefix"):
            with self.subTest(defect=defect), stress_registry() as (evidence, preparation, terminal):
                if defect == "unknown":
                    bad = {"index": "experiments/synthetic-terminal/unadmitted/evidence-index.json", "sha256": "a" * 64}
                    terminal.contract["reader_dependency"]["pin"] = bad;terminal.contract["ancestors"] = [bad]
                elif defect == "not_ancestor": terminal.contract["ancestors"] = []
                elif defect == "self_cycle":
                    terminal.contract["reader_dependency"]["pin"] = terminal.pin;terminal.contract["ancestors"] = [terminal.pin]
                elif defect == "ancestor_cycle": preparation.contract["ancestors"] = [terminal.pin]
                else: terminal.contract["reader_dependency"]["prefix"] = "other"
                with self.assertRaises(ValueError): C.collect(evidence, {ROLE: terminal.pin})
                self.assertEqual(evidence.reads, [])

    def test_own_reserved_preparation_namespace_is_rejected(self):
        for reserved in ("preparation", "preparation/extra.txt"):
            with self.subTest(reserved=reserved), stress_registry(own={reserved: b"synthetic collision"}) as (evidence, preparation, terminal):
                with self.assertRaisesRegex(ValueError, "Reserved preparation"): C.collect(evidence, {ROLE: terminal.pin})
        with self.subTest(reserved="compressed_text logical alias"), stress_registry() as (evidence, preparation, terminal):
            terminal.index["compressed_text"] = {"preparation/fake.txt": {"compressed_path": "logs/capture.txt.gz"}}
            terminal.reseal();evidence.entries.update(terminal.entries);C.ADMITTED[ROLE][1] = terminal.contract
            with self.assertRaisesRegex(ValueError, "Reserved preparation"): C.collect(evidence, {ROLE: terminal.pin})

    def test_index_dependency_declaration_must_match_all_admitted_parent_identities(self):
        for field in ("sibling", "index_sha256", "verifier_sha256"):
            with self.subTest(field=field), stress_registry() as (evidence, preparation, terminal):
                terminal.index["dependencies"]["preparation"][field] = "wrong" if field == "sibling" else "0" * 64
                terminal.reseal();evidence.entries.update(terminal.entries)
                C.ADMITTED[ROLE][1] = terminal.contract
                with self.assertRaisesRegex(ValueError, "dependency declaration"): C.collect(evidence, {ROLE: terminal.pin})

    def test_missing_or_tampered_parent_nonclaim_bytes_are_hard_errors(self):
        for defect in ("missing", "tampered"):
            with self.subTest(defect=defect), stress_registry() as (evidence, preparation, terminal):
                key = preparation.prefix + "/extra.txt"
                if defect == "missing": del evidence.entries[key]
                else: evidence.entries[key] += b" synthetic drift"
                with self.assertRaises((KeyError, ValueError)): C.collect(evidence, {ROLE: terminal.pin})

    def test_verifier_rejects_extra_callback_paths_and_missing_dependency_receipt(self):
        with stress_registry(extra_callback_path="preparation/unindexed-extra.txt") as (evidence, preparation, terminal):
            with self.assertRaises(AssertionError): C.collect(evidence, {ROLE: terminal.pin})
        with stress_registry() as (evidence, preparation, terminal):
            source = terminal.payloads["verify_evidence.py"].replace(b"result['preparation_index_sha256']", b"result['omitted_dependency']")
            terminal.payloads["verify_evidence.py"] = source
            terminal.index["files"]["verify_evidence.py"].update(size=len(source), sha256=F.sha(source))
            terminal.reseal();evidence.entries.update(terminal.entries);C.ADMITTED[ROLE][1] = terminal.contract
            with self.assertRaisesRegex(ValueError, "proof omitted"): C.collect(evidence, {ROLE: terminal.pin})


class SyntheticStressTerminalTests(unittest.TestCase):
    def test_four_checked_streams_and_five_failed_comparisons_do_not_qualify_full_coverage(self):
        with stress_registry() as (evidence, preparation, terminal): result = C.collect(evidence, {ROLE: terminal.pin})
        row = result["current_campaign_diagnostics"][ROLE]
        self.assertTrue(row["integrity_verified"])
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["coverage"]["host_checked_requests"], 4)
        self.assertEqual(row["coverage"]["state_counts"], {"host_checked_complete": 4, "deferred_before_launch": 1, "not_started": 19})
        self.assertEqual(len(row["comparisons"]), 6)
        self.assertEqual(sum(not item["passed"] for item in row["comparisons"]), 5)
        self.assertTrue(all(row["gates"][gate] is False for gate in ("host_checked", "full_stream", "agreement", "two_point_target_1e8")))
        self.assertEqual(row["ancestry"][0]["status"], "prepared")
        self.assertEqual(len(row["ancestry"][0]["request_rows"]), 24)
        F.assert_no_broad_mutation(self, result)

    def test_agreeing_four_lane_subset_cannot_qualify_six_lanes_or_all_cases(self):
        with stress_registry() as (evidence, preparation, terminal):
            value = deepcopy(terminal.value)
            for comparison in value["comparisons"]:
                comparison.update(passed=True, ordinary_target_passed=True, failures=[])
            value["case_coverage"][0]["compared_subset_agreement"] = True
            row = C.stress_terminal(value, terminal.contract)
        self.assertEqual(len(row["comparisons"]), 6)
        self.assertTrue(all(item["passed"] for item in row["comparisons"]))
        self.assertTrue(row["case_coverage"][0]["compared_subset_agreement"])
        self.assertEqual(row["case_coverage"][0]["host_checked_lanes"], 4)
        self.assertFalse(row["case_coverage"][0]["six_lane_coverage"])
        self.assertFalse(row["case_coverage"][0]["six_lane_agreement"])
        self.assertEqual(row["coverage"]["host_checked_requests"], 4)
        self.assertEqual(row["status"], "failed")
        self.assertTrue(all(row["gates"][gate] is False for gate in ("host_checked", "full_stream", "agreement", "two_point_target_1e8")))
        self.assertFalse(row["broad_qualified"])
        self.assertFalse(row["performance_eligible"])

    def test_all_24_terminal_identities_states_and_histogram_are_required(self):
        for defect in ("missing", "duplicate", "backend", "unknown_state", "histogram", "qualified_flag"):
            with self.subTest(defect=defect), stress_registry() as (evidence, preparation, terminal):
                value = deepcopy(terminal.value)
                if defect == "missing": value["request_rows"].pop()
                elif defect == "duplicate": value["request_rows"][1] = deepcopy(value["request_rows"][0])
                elif defect == "backend": value["request_rows"][0]["backend"] = 9
                elif defect == "unknown_state": value["request_rows"][4]["state"] = "passed"
                elif defect == "histogram": value["state_counts"]["not_started"] -= 1
                else: value["request_rows"][4]["host_checked_complete_stream"] = True
                with self.assertRaises(ValueError): C.stress_terminal(value, terminal.contract)

    def test_rejected_w03_unchecked_nonfinite_and_raw_failures_remain_retained(self):
        with stress_registry() as (evidence, preparation, terminal):
            value = deepcopy(terminal.value);row = value["request_rows"][5]
            row.update(state="physical_rejected", warnings=[{"text": "##W03 synthetic unstable contribution"}], nonfinite_rows=1,
                raw_final_points_observed=[1], native_failure={"message": "synthetic native refusal"},
                parser_failure={"token": "NaN", "message": "synthetic nonfinite token"})
            row["final_trace"] = trace([{"point": 1, "order": 0, "status": 120, "alias_backend": row["backend"],
                "alias_precision": "QP", "raw_coefficients": ["NaN", "+Infinity", "0.0", "0.0"],
                "raw_accuracy": "-1.0", "nonfinite_tokens": ["NaN", "+Infinity"]}])
            row["final_trace"]["nonfinite_row_count"] = 1
            value["state_counts"].update(not_started=18, physical_rejected=1);value["completed_orchestration"] = 5
            result = C.stress_terminal(value, terminal.contract)
        retained = result["request_rows"][5]["retained"]
        self.assertEqual(retained, row)
        self.assertEqual(retained["final_trace"]["rows"][0]["status"], 120)
        self.assertEqual(retained["final_trace"]["rows"][0]["raw_coefficients"][:2], ["NaN", "+Infinity"])
        self.assertEqual(retained["final_trace"]["rows"][0]["nonfinite_tokens"], ["NaN", "+Infinity"])
        self.assertFalse(result["request_rows"][5]["numerically_qualified"])
        self.assertEqual(result["coverage"]["host_checked_requests"], 4)

    def test_checked_stream_rejects_warnings_failures_nonfinite_or_missing_final_rows(self):
        for defect in ("warning", "nonfinite", "failure", "missing_point", "duplicate_order", "status120", "rejected_status420", "alias_contradiction", "nan_forged_counter", "accuracy", "backend", "precision"):
            with self.subTest(defect=defect), stress_registry() as (evidence, preparation, terminal):
                value = deepcopy(terminal.value);row = value["request_rows"][0]
                if defect == "warning": row["warnings"] = ["##W03 synthetic"]
                elif defect == "nonfinite": row["nonfinite_rows"] = 1
                elif defect == "failure": row["controller_failure"] = {"message": "synthetic retained failure"}
                elif defect == "missing_point": row["final_trace"]["rows"].pop()
                elif defect == "duplicate_order": row["final_trace"]["rows"][1] = deepcopy(row["final_trace"]["rows"][0])
                elif defect == "status120": row["final_trace"]["rows"][0]["status"] = 120
                elif defect == "rejected_status420": row["final_trace"]["rows"][0]["status"] = 420
                elif defect == "alias_contradiction": row["final_trace"]["rows"][0]["status"] = 219
                elif defect == "nan_forged_counter": row["final_trace"]["rows"][0]["raw_coefficients"][0] = "NaN"
                elif defect == "accuracy": row["final_trace"]["rows"][0]["raw_accuracy"] = "1.0"
                elif defect == "backend": row["final_trace"]["rows"][0]["alias_backend"] = 9
                else: row["final_trace"]["rows"][0]["alias_precision"] = "requested_QP"
                with self.assertRaises(ValueError): C.stress_terminal(value, terminal.contract)

    def test_two_final_points_and_observed_warmup_calls_are_distinct_counts(self):
        with stress_registry() as (evidence, preparation, terminal): result = C.collect(evidence, {ROLE: terminal.pin})
        row = result["current_campaign_diagnostics"][ROLE]
        self.assertEqual(row["coverage"]["qualified_final_points"], 8)
        self.assertEqual(row["coverage"]["warmup_calls_observed"], 28)
        self.assertIsNone(row["amplitude_evaluations"])
        self.assertTrue(row["coverage"]["amplitude_observations_are_trace_only"])
        self.assertEqual(len(row["request_rows"][0]["retained"]["final_trace"]["rows"]), 4)
        self.assertEqual(len(row["request_rows"][0]["retained"]["warmup_trace"]["rows"]), 7)

    def test_terminal_one_lock_failure_is_failed_and_is_not_prepared_success(self):
        with stress_registry(mode="lock_failure") as (evidence, preparation, terminal): result = C.collect(evidence, {ROLE: terminal.pin})
        row = result["current_campaign_diagnostics"][ROLE]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["coverage"]["host_checked_requests"], 0)
        self.assertEqual(row["request_rows"][0]["state"], "controller_failed")
        self.assertEqual(row["request_rows"][0]["retained"]["controller_failure"]["type"], "LockFailure")
        self.assertEqual(row["ancestry"][0]["status"], "prepared")
        self.assertFalse(row["gates"]["native_orchestration_passed"])

    def test_partial_subset_or_scientific_flags_cannot_be_promoted(self):
        for defect in ("all_cases", "case_coverage", "oracle", "performance", "truth", "comparison_missing"):
            with self.subTest(defect=defect), stress_registry() as (evidence, preparation, terminal):
                value = deepcopy(terminal.value)
                if defect == "all_cases": value["six_lane_all_cases_agree"] = True
                elif defect == "case_coverage": value["case_coverage"][0]["six_lane_coverage"] = True
                elif defect == "oracle": value["EW_analytic_oracle_qualified"] = True
                elif defect == "performance": value["performance_eligible"] = True
                elif defect == "truth": value["independent_physical_accuracy_qualified"] = True
                else: value["comparisons"].pop()
                with self.assertRaises(ValueError): C.stress_terminal(value, terminal.contract)


class SyntheticSecondCmsTerminalTests(unittest.TestCase):
    def test_six_completed_workers_with_no_final_outcomes_remain_incomplete(self):
        with cms_registry() as (evidence, pin): result = C.collect(evidence, {"current_cms": pin})
        row = result["current_campaign_diagnostics"]["current_cms"]
        self.assertEqual(row["status"], "incomplete")
        self.assertEqual(row["coverage"]["completed_requests"], 6)
        self.assertEqual(row["coverage"]["worker_requests_started"], 6)
        self.assertEqual(row["amplitude_launch_count"], 6)
        self.assertIsNone(row["amplitude_evaluations"])
        self.assertIsNone(row["coverage"]["unexecuted_amplitude_streams"])
        self.assertEqual(row["coverage"]["host_checked_requests"], 0)
        self.assertEqual(row["coverage"]["complete_final_streams"], 0)
        self.assertEqual(row["coverage"]["phase_order_outcomes"]["final"], {
            "accepted": 0, "rejected": 0, "unchecked": 0, "unexecuted": 252})
        self.assertEqual(row["coverage"]["phase_order_outcomes"]["warmup"]["accepted"], 206)
        self.assertEqual([r["state"] for r in row["request_rows"]], ["startup_timeout"] * 2 + ["warmup_rejected"] * 4)
        self.assertTrue(all(r["retained"]["phase_counts"]["final"]["unexecuted"] == 42 for r in row["request_rows"]))
        self.assertFalse(row["gates"]["host_checked_full21"])
        self.assertEqual(row["ancestry"][0]["status"], "failed")
        self.assertEqual(row["ancestry"][0]["amplitude_evaluations"], 0)
        F.assert_no_broad_mutation(self, result)

    def test_warmup_acceptance_cannot_qualify_unexecuted_final_streams(self):
        value = cms_second_fixture()
        self.assertGreater(sum(row["phase_counts"]["warmup"]["accepted"] for row in value["requests"]), 0)
        self.assertEqual(sum(row["phase_counts"]["final"]["unexecuted"] for row in value["requests"]), 252)
        for defect in ("request_qualified", "numerical_gate", "candidate_complete"):
            with self.subTest(defect=defect):
                bad = deepcopy(value)
                if defect == "request_qualified": bad["requests"][1].update(state="host_checked", numerically_qualified=True)
                elif defect == "numerical_gate": bad["gates"]["numerically_qualified"] = True
                else: bad["five_current_reducers_complete"] = True
                with cms_registry(bad) as (evidence, pin):
                    with self.assertRaises(ValueError): C.collect(evidence, {"current_cms": pin})


if __name__ == "__main__":
    unittest.main()
