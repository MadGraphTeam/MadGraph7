"""Synthetic current-diagnostic admission tests; no real proof or native reads.

Tiny virtual capsules and their invented Python verifier exercise policy and
integrity separately. All source identities below are fixture bindings, not new
stress/CMS observations. No filesystem capsule, worker, build or Typst runs.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("synthetic_current_diagnostics_target", Path(__file__).with_name("campaign_report_current_diagnostics.py"))
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)

VERIFIER = b'''import hashlib
import json
import sys
assert __debug__, "synthetic proof assertions must survive"
assert sys.dont_write_bytecode is True
def verify(*, read_bytes):
    assert __name__ != "__main__"
    raw = read_bytes("evidence-index.json")
    assert set(read_bytes.paths) == {"evidence-index.json", "verify_evidence.py", "summary.json", "extra.txt"}
    assert read_bytes("extra.txt") == b"synthetic nonclaim payload"
    return {"verified":True, "index_sha256":hashlib.sha256(raw).hexdigest(),
            "summary":json.loads(read_bytes("summary.json"))}
'''


def encoded(value):
    return json.dumps(value, sort_keys=True).encode("utf-8")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class MemoryEvidence:
    def __init__(self, entries=None):
        self.entries, self.reads = entries or {}, []

    def add(self, name, expected=None):
        self.reads.append(name)
        raw = self.entries[name]
        if expected is not None and sha(raw) != expected:
            raise ValueError("Synthetic indexed hash differs")
        return raw


class NoReads:
    def add(self, *args, **kwargs):
        raise AssertionError("Pending/unadmitted selection attempted an evidence read")


def stress_fixture():
    value = {"scope": "current-source preparation and preflight only", "role": "current-physical-stress-preparation",
        "passed": True, "preparation_qualified": True, "closed_stages": ["prepare", "preflight"],
        "native_state": "pending", "candidate_revision": C.REVISION, "baseline_revision": C.BASELINE,
        "candidate_generator_sha256": C.GENERATOR, "campaign_manifest_sha256": C.MANIFEST,
        "source_index_sha256": C.STRESS_SOURCE, "candidate_DSO_sha256": C.DSO,
        "fresh_CLI_exports": 8, "fresh_full_builds": 8, "mass_preflights": 8,
        "amplitude_evaluations": 0, "physical_stress_qualified": False, "pending_native_requests": 24,
        "performance_eligible": False, "EW_analytic_oracle_qualified": False}
    value["request_rows"] = [dict(row, state="not_started", numerically_qualified=False,
        observed_statuses=None, incomplete_warnings=None, failure=None) for row in C.expected("current_physical_stress")]
    return value


def phase_fixture(state="unexecuted"):
    status = 120 if state == "unchecked" else (323 if state == "accepted" else (420 if state == "rejected" else None))
    rows = [{"point": point, "order": order, "state": state, "status": status}
            for point in range(1, 22) for order in (0, 1)]
    return rows, {name: 42 if name == state else 0 for name in C.COUNTS}


def set_phase(request, name, state="unexecuted"):
    rows, counts = phase_fixture(state)
    request.setdefault("phase_outcomes", {})[name] = rows
    request.setdefault("phase_counts", {})[name] = counts


def cms_fixture(stage="prepared"):
    requests = [{**{key: row[key] for key in ("id", "source", "backend")}, "state": "pending",
        "controller_setup_started": False, "execution_started": False,
        "numerically_qualified": False, "amplitude_observed": False} for row in C.expected("current_cms")]
    for request in requests:
        set_phase(request, "warmup")
        set_phase(request, "final")
    value = {"schema_version": 1, "role": "current-cms-replay", "closed": True, "stage": stage,
        "binding": {**deepcopy(C.CMS_SCIENCE_BINDING), "geometry_source": "synthetic serialized 21-point geometry",
            "proposal_index_sha256": C.CMS_PROPOSAL, "supervisor_index_sha256": C.CMS_SUPERVISOR},
        "performance_eligible": False, "requested_requests": 6,
        "primary_policy_agreement": False, "five_current_reducers_complete": False,
        "gates": {"independent_finite_term_oracle": False, "native_run_passed": False, "numerically_qualified": False,
            "preparation_passed": True, "preflight_passed": True},
        "amplitude_launch_count": 0, "completed_requests": 0, "worker_launch_count": 0,
        "physics_loader_launch_count": 0, "requests": requests, "overall_status": "prepared", "failure": None,
        "stages": {"run": {"status": "not_started"}}}
    if stage != "prepared":
        value["overall_status"] = "failed"
        value["stages"]["run"] = {"status": "failed", "exit_code": 1}
        for row in requests:
            row.update(state="unexecuted")
        requests[0].update(state="failed", controller_setup_started=True)
        value["failure"] = {"type": "RuntimeError", "message": "synthetic baseline controller failed before amplitude launch"}
    return value


def started_cms(classification="rejected", all_requests=False):
    """Terminal numerical outcomes, separate from integrity-verifier success."""
    value = cms_fixture("native_terminal")
    value["stages"]["run"] = {"status": "terminal", "exit_code": 0}
    selected = value["requests"] if all_requests else value["requests"][:1]
    for row in selected:
        row.update(state="host_checked" if classification == "accepted" else ("incomplete" if classification == "unchecked" else "failed"),
            controller_setup_started=True, execution_started=True, numerically_qualified=classification == "accepted",
            amplitude_observed=True)
        set_phase(row, "warmup", "accepted")
        set_phase(row, "final", classification)
    value.update(amplitude_launch_count=len(selected), worker_launch_count=len(selected), physics_loader_launch_count=1,
        completed_requests=len(selected), overall_status="complete" if classification == "accepted" and all_requests else (
            "incomplete" if classification == "unchecked" else "failed"))
    value["gates"]["amplitude_execution_started"] = True
    value["failure"] = {"type": "NumericalRejection", "message": "synthetic checked rejection"} if classification == "rejected" else None
    if classification == "accepted" and all_requests:
        value["gates"].update(host_checked_full21=True, agreement_3e6=True, ordinary_1e8=True)
    return value


class Capsule:
    """Invented index, summary and verifier, admitted only by a test patch."""
    def __init__(self, role, value=None, stage=None, verifier=VERIFIER, namespace=None):
        self.role = role
        self.format = "stress" if role == "current_physical_stress" else "cms"
        self.value = value if value is not None else (stress_fixture() if self.format == "stress" else cms_fixture())
        self.stage = stage or ("prepared" if self.format == "stress" else self.value["stage"])
        self.verifier = verifier
        self.namespace = namespace or role

    def build(self, index_edit=None, contract_edit=None):
        prefix = "experiments/synthetic-current-diagnostics/" + self.namespace
        index_name = prefix + "/evidence-index.json"
        payloads = {"verify_evidence.py": self.verifier, "summary.json": encoded(self.value), "extra.txt": b"synthetic nonclaim payload"}
        files = {}
        for name, raw in payloads.items():
            identity = {"sha256": sha(raw)}
            identity.update({"kind": "file", "size": len(raw)} if self.format == "stress" else {"codec": "utf8", "bytes": len(raw)})
            files[name] = identity
        index = {"schema_version": 1, "files": files}
        if index_edit is not None:
            index_edit(index)
        index_raw = encoded(index)
        contract = {"index": index_name, "sha256": sha(index_raw), "format": self.format, "stage": self.stage,
            "summary": "summary.json", "summary_sha256": sha(payloads["summary.json"]), "verifier_sha256": sha(self.verifier)}
        if self.format == "cms":
            contract["control_binding"] = {key: self.value["binding"][key]
                for key in ("proposal_index_sha256", "supervisor_index_sha256")}
        if contract_edit is not None:
            contract_edit(contract)
        entries = {prefix + "/" + name: raw for name, raw in payloads.items()}
        entries[index_name] = index_raw
        pin = {"index": contract["index"], "sha256": contract["sha256"]}
        return entries, contract, pin

    @contextmanager
    def admitted(self, index_edit=None, contract_edit=None):
        entries, contract, pin = self.build(index_edit, contract_edit)
        with patch.object(C, "ADMITTED", {self.role: contract}):
            yield MemoryEvidence(entries), {self.role: pin}


def assert_no_broad_mutation(test, result):
    test.assertIs(result["main_campaign_modified"], False)
    test.assertIs(result["performance_acceptance_eligible"], False)
    for row in result["current_campaign_diagnostics"].values():
        test.assertIs(row["broad_qualified"], False)
        test.assertIs(row["performance_eligible"], False)


class SyntheticCurrentDiagnosticTests(unittest.TestCase):
    def test_absent_roles_are_pending_without_reading_any_directory(self):
        with patch.object(C, "ADMITTED", {}):
            self.assertEqual(C.default_selection(), {})
            result = C.collect(NoReads(), {})
        self.assertEqual(result["packages"], {})
        self.assertEqual(set(result["current_campaign_diagnostics"]), C.ROLES)
        for role, count in (("current_physical_stress", 24), ("current_cms", 6)):
            row = result["current_campaign_diagnostics"][role]
            self.assertEqual(row["status"], "pending")
            self.assertFalse(row["integrity_verified"])
            self.assertIsNone(row["amplitude_evaluations"])
            self.assertEqual(len(row["request_rows"]), count)
            self.assertTrue(all(r["state"] == "pending" and r["execution_started"] is None for r in row["request_rows"]))
            self.assertIsNone(row["coverage"]["amplitude_requests_started"])
            self.assertIsNone(row["coverage"]["completed_requests"])
        assert_no_broad_mutation(self, result)

    def test_unadmitted_or_unknown_selection_rejected_before_evidence_reads(self):
        for role in C.ROLES:
            with self.subTest(role=role), patch.object(C, "ADMITTED", {}):
                with self.assertRaises(ValueError): C.collect(NoReads(), {role: {"index": "experiments/synthetic/index.json", "sha256": "a" * 64}})
        with self.assertRaises(ValueError): C.collect(NoReads(), {"future-role": {}})
        packet = Capsule("current_cms")
        with packet.admitted() as (evidence, selection):
            selection[packet.role]["sha256"] = "0" * 64
            with self.assertRaises(ValueError): C.collect(NoReads(), selection)

    def test_any_unadmitted_pin_rejects_mixed_selection_before_any_read(self):
        packet = Capsule("current_cms")
        with packet.admitted() as (evidence, selection):
            selection["current_physical_stress"] = {"index": "experiments/unadmitted/evidence-index.json", "sha256": "b" * 64}
            with self.assertRaises(ValueError): C.collect(evidence, selection)
            self.assertEqual(evidence.reads, [])

    def test_stress_preparation_preserves_24_unstarted_requests_and_zero_amplitudes(self):
        packet = Capsule("current_physical_stress")
        with packet.admitted() as (evidence, selection):
            original_selection = deepcopy(selection)
            result = C.collect(evidence, selection)
            self.assertEqual(selection, original_selection)
        row = result["current_campaign_diagnostics"][packet.role]
        self.assertEqual((row["status"], row["stage"]), ("prepared", "prepared"))
        self.assertTrue(row["integrity_verified"])
        self.assertEqual(row["amplitude_evaluations"], 0)
        self.assertEqual(row["coverage"], {"expected_requests": 24, "pending_requests": 24, "amplitude_requests_started": 0,
            "worker_requests_started": 0, "completed_requests": 0})
        self.assertTrue(all(r["state"] == "unexecuted" and r["execution_started"] is False and r["numerically_qualified"] is False for r in row["request_rows"]))
        self.assertTrue(all(r["retained"]["state"] == "not_started" for r in row["request_rows"]))
        self.assertIn("extra.txt", {name.rsplit("/", 1)[-1] for name in evidence.reads})
        assert_no_broad_mutation(self, result)

    def test_stress_preparation_rejects_executed_rows_or_scientific_promotion(self):
        for defect in ("execution", "amplitudes", "broad", "oracle", "performance", "source"):
            with self.subTest(defect=defect):
                value = stress_fixture()
                if defect == "execution": value["request_rows"][0].update(state="host_checked", observed_statuses=[219])
                elif defect == "amplitudes": value["amplitude_evaluations"] = 1
                elif defect == "broad": value["physical_stress_qualified"] = True
                elif defect == "oracle": value["EW_analytic_oracle_qualified"] = True
                elif defect == "performance": value["performance_eligible"] = True
                else: value["source_index_sha256"] = "0" * 64
                with Capsule("current_physical_stress", value).admitted() as (evidence, selection):
                    with self.assertRaises(ValueError): C.collect(evidence, selection)

    def test_cms_prepared_retains_six_pending_warmup_and_final_streams(self):
        packet = Capsule("current_cms")
        with packet.admitted() as (evidence, selection): result = C.collect(evidence, selection)
        row = result["current_campaign_diagnostics"][packet.role]
        self.assertEqual(row["status"], "prepared")
        self.assertTrue(row["integrity_verified"])
        self.assertEqual(row["coverage"]["pending_requests"], 6)
        self.assertEqual(row["coverage"]["amplitude_requests_started"], 0)
        for request in row["request_rows"]:
            self.assertEqual(request["state"], "pending")
            for phase in ("warmup", "final"):
                retained = request["retained"]
                self.assertEqual(len(retained["phase_outcomes"][phase]), 42)
                self.assertEqual(retained["phase_counts"][phase]["unexecuted"], 42)
                self.assertTrue(all(row["state"] == "unexecuted" for row in retained["phase_outcomes"][phase]))
        assert_no_broad_mutation(self, result)

    def test_cms_minimal_serialized_contract_uses_science_binding_and_global_failure(self):
        value = cms_fixture("native_terminal")
        self.assertNotIn("inputs", value)
        self.assertTrue(set(C.CMS_SCIENCE_BINDING) <= set(value["binding"]))
        self.assertTrue(value["binding"]["geometry_source"])
        for request in value["requests"]:
            self.assertEqual(set(request), {"id", "source", "backend", "state", "controller_setup_started",
                "execution_started", "amplitude_observed", "numerically_qualified", "phase_outcomes", "phase_counts"})
        with Capsule("current_cms", value).admitted() as (evidence, selection): result = C.collect(evidence, selection)
        row = result["current_campaign_diagnostics"]["current_cms"]
        self.assertEqual(row["status"], "failed")
        self.assertTrue(row["retained_summary"]["requests"][0]["controller_setup_started"])
        self.assertFalse(row["retained_summary"]["gates"]["native_run_passed"])
        self.assertFalse(row["retained_summary"]["gates"]["numerically_qualified"])
        self.assertFalse(row["retained_summary"]["performance_eligible"])

    def test_cms_setup_failure_remains_failed_without_a_native_attempt(self):
        value = cms_fixture()
        value.update(stage="setup_terminal", overall_status="failed", failure={"type": "SetupFailure", "message": "synthetic preflight failed"})
        value["stages"].update(prepare={"status": "passed", "exit_code": 0}, preflight={"status": "failed", "exit_code": 1})
        value["gates"]["preflight_passed"] = False
        with Capsule("current_cms", value).admitted() as (evidence, selection): result = C.collect(evidence, selection)
        row = result["current_campaign_diagnostics"]["current_cms"]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["stage"], "setup_terminal")
        self.assertEqual(row["amplitude_evaluations"], 0)
        self.assertEqual(row["coverage"]["unexecuted_amplitude_streams"], 6)
        self.assertEqual(row["coverage"]["worker_requests_started"], 0)
        cms_note = next(note for note in result["notes"] if note["title"] == "Current CMS coverage")
        self.assertIn("Setup failed before a native attempt", cms_note["text"])
        self.assertNotIn("native orchestrator failed", cms_note["text"])
        assert_no_broad_mutation(self, result)

    def test_cms_common_science_and_geometry_cannot_change_with_controller_contract(self):
        for field, expected in C.CMS_SCIENCE_BINDING.items():
            with self.subTest(field=field):
                value = cms_fixture()
                changed = not expected if type(expected) is bool else (expected + 1 if type(expected) in (int, float) else "synthetic-changed-" + expected)
                value["binding"][field] = changed
                with Capsule("current_cms", value).admitted() as (evidence, selection):
                    with self.assertRaises(ValueError): C.collect(evidence, selection)
        for defect in ("geometry", "oracle", "performance", "request_count", "numerical_gate", "native_run_gate", "primary_agreement", "candidate_complete"):
            with self.subTest(defect=defect):
                value = cms_fixture()
                if defect == "geometry": value["binding"]["geometry_source"] = ""
                elif defect == "oracle": value["gates"]["independent_finite_term_oracle"] = True
                elif defect == "performance": value["performance_eligible"] = True
                elif defect == "request_count": value["requested_requests"] = 5
                elif defect == "numerical_gate": value["gates"]["numerically_qualified"] = True
                elif defect == "native_run_gate": value["gates"]["native_run_passed"] = True
                elif defect == "primary_agreement": value["primary_policy_agreement"] = True
                else: value["five_current_reducers_complete"] = True
                with Capsule("current_cms", value).admitted() as (evidence, selection):
                    with self.assertRaises(ValueError): C.collect(evidence, selection)

    def test_cms_controller_failure_is_failed_with_all_six_amplitudes_unexecuted(self):
        packet = Capsule("current_cms", cms_fixture("native_terminal"))
        with packet.admitted() as (evidence, selection): result = C.collect(evidence, selection)
        row = result["current_campaign_diagnostics"][packet.role]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["stage"], "native_terminal")
        self.assertTrue(row["integrity_verified"])
        self.assertEqual(row["amplitude_evaluations"], 0)
        self.assertEqual(row["coverage"]["unexecuted_amplitude_streams"], 6)
        self.assertEqual(row["coverage"]["unstarted_worker_requests"], 6)
        self.assertEqual(row["coverage"]["amplitude_requests_unobserved"], 6)
        self.assertEqual(row["coverage"]["amplitude_requests_started"], 0)
        self.assertEqual(row["request_rows"][0]["state"], "failed")
        self.assertTrue(all(r["state"] == "unexecuted" for r in row["request_rows"][1:]))
        self.assertTrue(all(not r["execution_started"] for r in row["request_rows"]))
        self.assertEqual(row["retained_summary"]["stages"]["run"]["exit_code"], 1)
        self.assertIn("before amplitude launch", row["retained_summary"]["failure"]["message"])
        assert_no_broad_mutation(self, result)

    def test_valid_scientific_failure_passes_integrity_but_is_not_qualified(self):
        packet = Capsule("current_cms", started_cms("rejected"))
        with packet.admitted() as (evidence, selection): result = C.collect(evidence, selection)
        row = result["current_campaign_diagnostics"][packet.role]
        self.assertTrue(row["integrity_verified"])
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["coverage"]["amplitude_requests_started"], 1)
        self.assertEqual(row["amplitude_launch_count"], 1)
        self.assertIsNone(row["amplitude_evaluations"])
        self.assertEqual(row["coverage"]["host_checked_requests"], 0)
        self.assertEqual(row["request_rows"][0]["retained"]["phase_counts"]["final"]["rejected"], 42)
        self.assertFalse(row["request_rows"][0]["numerically_qualified"])
        assert_no_broad_mutation(self, result)

    def test_worker_initialization_failure_can_precede_any_observed_amplitude(self):
        value = cms_fixture("native_terminal")
        value["requests"][0].update(execution_started=True, amplitude_observed=False)
        value["failure"] = {"type": "InitializationFailure", "message": "synthetic physics worker failed before bootstrap"}
        value["worker_launch_count"] = 1
        value["physics_loader_launch_count"] = 1
        with Capsule("current_cms", value).admitted() as (evidence, selection): result = C.collect(evidence, selection)
        row = result["current_campaign_diagnostics"]["current_cms"]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["amplitude_launch_count"], 0)
        self.assertEqual(row["coverage"]["amplitude_requests_started"], 0)
        self.assertEqual(row["coverage"]["worker_requests_started"], 1)
        self.assertIsNone(row["coverage"]["unexecuted_amplitude_streams"])
        self.assertEqual(row["coverage"]["unstarted_worker_requests"], 5)
        self.assertEqual(row["coverage"]["amplitude_requests_unobserved"], 6)
        self.assertEqual(row["coverage"]["host_checked_requests"], 0)
        self.assertIsNone(row["amplitude_evaluations"])
        self.assertTrue(row["integrity_verified"])
        assert_no_broad_mutation(self, result)

    def test_unchecked_status120_is_retained_without_checked_agreement(self):
        value = started_cms("unchecked")
        with Capsule("current_cms", value).admitted() as (evidence, selection): result = C.collect(evidence, selection)
        row = result["current_campaign_diagnostics"]["current_cms"]
        self.assertEqual(row["status"], "incomplete")
        self.assertEqual(row["coverage"]["unchecked_final_order_rows"], 42)
        self.assertEqual(row["coverage"]["host_checked_requests"], 0)
        self.assertTrue(all(r["status"] == 120 for r in row["request_rows"][0]["retained"]["phase_outcomes"]["final"]))
        for defect in ("request_qualified", "host_gate", "agreement_gate", "target_gate"):
            with self.subTest(defect=defect):
                bad = deepcopy(value)
                if defect == "request_qualified": bad["requests"][0].update(state="host_checked", numerically_qualified=True)
                else: bad["gates"][{"host_gate": "host_checked_full21", "agreement_gate": "agreement_3e6", "target_gate": "ordinary_1e8"}[defect]] = True
                with Capsule("current_cms", bad).admitted() as (evidence, selection):
                    with self.assertRaises(ValueError): C.collect(evidence, selection)

    def test_duplicate_missing_or_substituted_requests_are_rejected(self):
        for role in C.ROLES:
            for defect in ("missing", "duplicate", "backend"):
                with self.subTest(role=role, defect=defect):
                    value = stress_fixture() if role == "current_physical_stress" else cms_fixture()
                    rows = value["request_rows" if role == "current_physical_stress" else "requests"]
                    if defect == "missing": rows.pop()
                    elif defect == "duplicate": rows[1] = deepcopy(rows[0])
                    else: rows[0]["backend"] = 9
                    with Capsule(role, value).admitted() as (evidence, selection):
                        with self.assertRaises(ValueError): C.collect(evidence, selection)

    def test_cms_phase_requires_all_42_rows_and_exact_nonnegative_count_partition(self):
        for phase in ("warmup", "final"):
            for defect in ("missing_row", "extra_row", "sum", "negative", "boolean", "unknown_count"):
                with self.subTest(phase=phase, defect=defect):
                    value = cms_fixture();request = value["requests"][0]
                    rows, counts = request["phase_outcomes"][phase], request["phase_counts"][phase]
                    if defect == "missing_row": rows.pop()
                    elif defect == "extra_row": rows.append(deepcopy(rows[0]))
                    elif defect == "sum": counts["unexecuted"] = 41
                    elif defect == "negative": counts.update(accepted=-1, unexecuted=43)
                    elif defect == "boolean": counts["accepted"] = False
                    else: counts["future"] = 0
                    with Capsule("current_cms", value).admitted() as (evidence, selection):
                        with self.assertRaises(ValueError): C.collect(evidence, selection)

    def test_cms_phase_point_order_pairs_require_exact_typed_ordered_coverage(self):
        for phase in ("warmup", "final"):
            for defect in ("duplicate", "misordered", "point_bool", "order_bool", "missing_point", "missing_order", "point_range", "order_range"):
                with self.subTest(phase=phase, defect=defect):
                    value = cms_fixture();rows = value["requests"][0]["phase_outcomes"][phase]
                    if defect == "duplicate": rows[1] = deepcopy(rows[0])
                    elif defect == "misordered": rows[0], rows[1] = rows[1], rows[0]
                    elif defect == "point_bool": rows[0]["point"] = True
                    elif defect == "order_bool": rows[0]["order"] = False
                    elif defect == "missing_point": del rows[0]["point"]
                    elif defect == "missing_order": del rows[0]["order"]
                    elif defect == "point_range": rows[-1]["point"] = 22
                    else: rows[1]["order"] = 2
                    with Capsule("current_cms", value).admitted() as (evidence, selection):
                        with self.assertRaises(ValueError): C.collect(evidence, selection)

    def test_cms_phase_states_must_match_counts_and_native_status(self):
        for phase in ("warmup", "final"):
            for defect in ("row_count_contradiction", "accepted_status120", "missing_state", "invalid_state", "unexecuted_status"):
                with self.subTest(phase=phase, defect=defect):
                    value = started_cms("accepted", all_requests=True)
                    request = value["requests"][0]
                    rows, counts = request["phase_outcomes"][phase], request["phase_counts"][phase]
                    if defect == "row_count_contradiction": counts.update(accepted=41, unchecked=1)
                    elif defect == "accepted_status120": rows[0]["status"] = 120
                    elif defect == "missing_state": del rows[0]["state"]
                    elif defect == "invalid_state": rows[0]["state"] = "pending"
                    else:
                        set_phase(request, phase)
                        request["phase_outcomes"][phase][0]["status"] = 219
                    with Capsule("current_cms", value).admitted() as (evidence, selection):
                        with self.assertRaises(ValueError): C.collect(evidence, selection)

    def test_cms_legacy_or_missing_phase_maps_are_rejected(self):
        for defect in ("legacy_shape", "missing_outcomes", "missing_counts", "missing_warmup", "missing_final"):
            with self.subTest(defect=defect):
                value = cms_fixture();request = value["requests"][0]
                if defect == "legacy_shape":
                    outcomes, counts = request.pop("phase_outcomes"), request.pop("phase_counts")
                    for name in ("warmup", "final"): request[name] = {"rows": outcomes[name], "counts": counts[name]}
                elif defect == "missing_outcomes": del request["phase_outcomes"]
                elif defect == "missing_counts": del request["phase_counts"]
                elif defect == "missing_warmup": del request["phase_outcomes"]["warmup"]
                else: del request["phase_counts"]["final"]
                with Capsule("current_cms", value).admitted() as (evidence, selection):
                    with self.assertRaises(ValueError): C.collect(evidence, selection)

    def test_cms_cannot_claim_checked_requests_without_launches_or_completions(self):
        for defect in ("no_launches", "no_completions", "no_worker", "unobserved_amplitude"):
            with self.subTest(defect=defect):
                value = started_cms("accepted", all_requests=True)
                if defect == "no_launches":
                    value["amplitude_launch_count"] = 0
                    value["gates"]["amplitude_execution_started"] = False
                elif defect == "no_completions": value["completed_requests"] = 0
                elif defect == "no_worker": value["worker_launch_count"] = 0
                else: value["requests"][0]["amplitude_observed"] = False
                with Capsule("current_cms", value).admitted() as (evidence, selection):
                    with self.assertRaises(ValueError): C.collect(evidence, selection)

    def test_positive_launches_are_not_reported_as_evaluation_counts_or_broad_acceptance(self):
        value = started_cms("accepted", all_requests=True)
        with Capsule("current_cms", value).admitted() as (evidence, selection): result = C.collect(evidence, selection)
        row = result["current_campaign_diagnostics"]["current_cms"]
        self.assertEqual(row["status"], "complete")
        self.assertEqual(row["coverage"]["host_checked_requests"], 6)
        self.assertEqual(row["amplitude_launch_count"], 6)
        self.assertIsNone(row["amplitude_evaluations"])
        self.assertIsNone(row["coverage"]["unexecuted_amplitude_streams"])
        self.assertEqual(row["coverage"]["unstarted_worker_requests"], 0)
        self.assertEqual(row["coverage"]["amplitude_requests_unobserved"], 0)
        assert_no_broad_mutation(self, result)

    def test_index_hash_payload_size_and_nonclaim_tampering_are_hard_errors(self):
        for role in C.ROLES:
            for defect in ("index_tamper", "summary_tamper", "nonclaim_tamper", "size", "hash"):
                with self.subTest(role=role, defect=defect):
                    packet = Capsule(role)
                    def edit(index):
                        if defect == "size": index["files"]["summary.json"]["size" if role == "current_physical_stress" else "bytes"] += 1
                        elif defect == "hash": index["files"]["summary.json"]["sha256"] = "0" * 64
                    with packet.admitted(index_edit=edit) as (evidence, selection):
                        if defect.endswith("tamper"):
                            index = selection[role]["index"]
                            name = index if defect == "index_tamper" else str(Path(index).parent / ("summary.json" if defect == "summary_tamper" else "extra.txt"))
                            evidence.entries[name] += b" synthetic altered bytes"
                        with self.assertRaises(ValueError): C.collect(evidence, selection)

    def test_verifier_and_summary_pins_and_replay_receipt_must_match(self):
        for defect in ("verifier_pin", "summary_pin", "verified_false", "index_mismatch", "summary_mismatch"):
            with self.subTest(defect=defect):
                source = VERIFIER
                if defect == "verified_false": source = source.replace(b'"verified":True', b'"verified":False')
                elif defect == "index_mismatch": source = source.replace(b'hashlib.sha256(raw).hexdigest()', b'"0"*64')
                elif defect == "summary_mismatch": source = source.replace(b'json.loads(read_bytes("summary.json"))', b'{}')
                def edit(contract):
                    if defect in ("verifier_pin", "summary_pin"): contract["verifier_sha256" if defect == "verifier_pin" else "summary_sha256"] = "0" * 64
                with Capsule("current_cms", verifier=source).admitted(contract_edit=edit) as (evidence, selection):
                    with self.assertRaises(ValueError): C.collect(evidence, selection)

    def test_bytecode_setting_is_restored_after_success_or_verifier_exception(self):
        previous = sys.dont_write_bytecode
        try:
            for initial in (True, False):
                for source, success in ((VERIFIER, True), (VERIFIER + b'\nraise RuntimeError("synthetic verifier import failure")\n', False),
                        (VERIFIER.replace(b'    raw = read_bytes("evidence-index.json")', b'    raise RuntimeError("synthetic verifier call failure")'), False)):
                    with self.subTest(initial=initial, success=success):
                        sys.dont_write_bytecode = initial
                        with Capsule("current_cms", verifier=source).admitted() as (evidence, selection):
                            if success: C.collect(evidence, selection)
                            else:
                                with self.assertRaises(RuntimeError): C.collect(evidence, selection)
                        self.assertIs(sys.dont_write_bytecode, initial)
        finally:
            sys.dont_write_bytecode = previous


class SyntheticAdmittedSuccessorTests(unittest.TestCase):
    """Future namespaces here are invented; no future capsule is admitted."""
    def registry(self):
        old = Capsule("current_cms", cms_fixture("native_terminal"), namespace="cms-old-closed-001")
        old_entries, old_contract, old_pin = old.build()
        successor_value = cms_fixture()
        successor_value["binding"].update(proposal_index_sha256="d" * 64, supervisor_index_sha256="e" * 64)
        successor = Capsule("current_cms", successor_value, namespace="cms-future-mechanical-999")
        new_entries, new_contract, new_pin = successor.build()
        new_contract.update(ancestors=[deepcopy(old_pin)], setup_provenance={"kind": "inherited", "source": deepcopy(old_pin)})
        return {**old_entries, **new_entries}, [old_contract, new_contract], old_pin, new_pin

    def test_registry_dict_and_list_defaults_keep_first_contract_not_newest(self):
        entries, contracts, old_pin, new_pin = self.registry()
        with patch.object(C, "ADMITTED", {"current_cms": contracts[0]}):
            self.assertEqual(C.default_selection(), {"current_cms": old_pin})
        with patch.object(C, "ADMITTED", {"current_cms": contracts}):
            self.assertEqual(C.default_selection(), {"current_cms": old_pin})
            result = C.collect(MemoryEvidence(entries), C.default_selection())
        row = result["current_campaign_diagnostics"]["current_cms"]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["evidence"], old_pin["index"])
        self.assertNotEqual(row["evidence"], new_pin["index"])
        self.assertEqual(row["ancestry"], [])

    def test_explicit_mechanical_successor_keeps_science_and_failed_ancestor(self):
        entries, contracts, old_pin, new_pin = self.registry()
        evidence = MemoryEvidence(entries)
        with patch.object(C, "ADMITTED", {"current_cms": contracts}):
            result = C.collect(evidence, {"current_cms": new_pin})
        row = result["current_campaign_diagnostics"]["current_cms"]
        self.assertEqual(row["status"], "prepared")
        self.assertEqual(row["setup_provenance"]["kind"], "inherited")
        self.assertEqual(row["retained_summary"]["binding"]["proposal_index_sha256"], "d" * 64)
        self.assertEqual(row["retained_summary"]["binding"]["supervisor_index_sha256"], "e" * 64)
        for key, expected in C.CMS_SCIENCE_BINDING.items():
            self.assertEqual(row["retained_summary"]["binding"][key], expected)
        self.assertEqual(len(row["ancestry"]), 1)
        ancestor = row["ancestry"][0]
        self.assertEqual(ancestor["evidence"], old_pin["index"])
        self.assertEqual(ancestor["index_sha256"], old_pin["sha256"])
        self.assertEqual(ancestor["status"], "failed")
        self.assertEqual(ancestor["coverage"]["unexecuted_amplitude_streams"], 6)
        self.assertEqual(len(ancestor["request_rows"]), 6)
        self.assertEqual(ancestor["request_rows"][0]["state"], "failed")
        self.assertEqual(ancestor["retained_summary"]["failure"]["type"], "RuntimeError")
        self.assertIn(old_pin["index"], evidence.reads)
        self.assertIn(new_pin["index"], evidence.reads)
        self.assertEqual(result["packages"]["current_cms"], new_pin)
        text = next(note["text"] for note in result["notes"] if note["title"] == "Current CMS coverage")
        self.assertIn("does not claim fresh CLI exports or builds", text)
        self.assertIn("failed ancestor capsules", text)
        assert_no_broad_mutation(self, result)

    def test_changed_controller_source_requires_the_matching_explicit_contract(self):
        entries, contracts, old_pin, new_pin = self.registry()
        contracts[1]["control_binding"] = deepcopy(contracts[0]["control_binding"])
        with patch.object(C, "ADMITTED", {"current_cms": contracts}):
            with self.assertRaisesRegex(ValueError, "controller/proposal binding"):
                C.collect(MemoryEvidence(entries), {"current_cms": new_pin})

    def test_unknown_future_pin_is_rejected_before_any_evidence_read(self):
        entries, contracts, old_pin, new_pin = self.registry()
        unknown = {"index": "experiments/synthetic-current-diagnostics/cms-future-unadmitted/evidence-index.json", "sha256": "f" * 64}
        with patch.object(C, "ADMITTED", {"current_cms": contracts}):
            with self.assertRaises(ValueError): C.collect(NoReads(), {"current_cms": unknown})

    def test_ancestry_must_be_explicit_admitted_acyclic_and_unambiguous_before_reads(self):
        for defect in ("missing_ancestor", "unadmitted_ancestor", "cycle", "ambiguous_registry"):
            with self.subTest(defect=defect):
                entries, contracts, old_pin, new_pin = self.registry()
                if defect == "missing_ancestor": contracts[1]["ancestors"] = []
                elif defect == "unadmitted_ancestor": contracts[1]["ancestors"] = [{
                    "index": "experiments/synthetic-current-diagnostics/unknown-ancestor/evidence-index.json", "sha256": "a" * 64}]
                elif defect == "cycle": contracts[0]["ancestors"] = [deepcopy(new_pin)]
                else: contracts.append(deepcopy(contracts[1]))
                evidence = MemoryEvidence(entries)
                with patch.object(C, "ADMITTED", {"current_cms": contracts}):
                    with self.assertRaises(ValueError): C.collect(evidence, {"current_cms": new_pin})
                self.assertEqual(evidence.reads, [])


class SyntheticCurrentRoutingTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("synthetic_current_routing_target", Path(__file__).with_name("campaign_report_implementation.py"))
        self.implementation = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.implementation)
        self.legacy = {"schema": "synthetic-legacy-supplement", "packages": {"adaptive_debug": {
            "index": "experiments/synthetic-legacy/index.json", "sha256": "a" * 64}},
            "primitives": {"adaptive_debug": {"synthetic_tensor_calls": 7}},
            "notes": [{"title": "Synthetic retained legacy note", "text": "Synthetic only", "evidence": "synthetic"}],
            "production_promotion": "pending separate receipt", "main_campaign_modified": False,
            "performance_acceptance_eligible": False, "synthetic_acceptance_sentinel": {"timing_gate": "unchanged"}}
        self.private = SimpleNamespace(ROLES=frozenset())

    @contextmanager
    def routed(self):
        with patch.object(self.implementation, "closed_diagnostics", return_value=self.private), \
             patch.object(self.implementation, "guarded_diagnostic_modules", return_value=[]), \
             patch.object(self.implementation, "_collect_legacy", side_effect=lambda evidence, selected: deepcopy(self.legacy)) as legacy, \
             patch.object(self.implementation, "current_diagnostics", return_value=C) as current:
            yield legacy, current

    def test_current_routing_merges_only_packages_notes_and_separate_coverage(self):
        packet = Capsule("current_cms")
        with packet.admitted() as (evidence, selection), self.routed() as (legacy, current):
            selection["adaptive_debug"] = deepcopy(self.legacy["packages"]["adaptive_debug"])
            result = self.implementation.collect(evidence, selection, current=True)
        self.assertEqual(result["primitives"], self.legacy["primitives"])
        self.assertEqual(result["synthetic_acceptance_sentinel"], self.legacy["synthetic_acceptance_sentinel"])
        self.assertEqual(set(result["packages"]), {"adaptive_debug", "current_cms"})
        self.assertEqual(len(result["notes"]), 3)
        self.assertEqual(result["current_campaign_diagnostics"]["current_cms"]["status"], "prepared")
        self.assertEqual(result["current_campaign_diagnostics"]["current_physical_stress"]["status"], "pending")
        self.assertFalse(result["main_campaign_modified"])
        self.assertFalse(result["performance_acceptance_eligible"])
        self.assertEqual(legacy.call_args.args[1], {"adaptive_debug": self.legacy["packages"]["adaptive_debug"]})
        current.assert_called_once_with()

    def test_default_routing_rejects_current_roles_without_collecting_evidence(self):
        with self.routed() as (legacy, current):
            with self.assertRaises(ValueError): self.implementation.collect(NoReads(), {"current_cms": {
                "index": "experiments/synthetic-current/index.json", "sha256": "a" * 64}})
        legacy.assert_not_called()
        current.assert_not_called()

    def test_default_legacy_behavior_is_preserved_and_current_pending_does_not_probe(self):
        selection = {"adaptive_debug": deepcopy(self.legacy["packages"]["adaptive_debug"])}
        with self.routed() as (legacy, current):
            result = self.implementation.collect(NoReads(), selection)
            self.assertEqual(result, self.legacy)
            current.assert_not_called()
        with self.routed(), patch.object(C, "ADMITTED", {}):
            result = self.implementation.collect(NoReads(), selection, current=True)
        self.assertEqual(result["primitives"], self.legacy["primitives"])
        self.assertEqual(result["packages"], self.legacy["packages"])
        self.assertTrue(all(row["status"] == "pending" for row in result["current_campaign_diagnostics"].values()))


if __name__ == "__main__":
    unittest.main()
