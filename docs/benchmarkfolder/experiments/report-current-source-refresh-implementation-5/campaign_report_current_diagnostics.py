"""Explicit closed current stress/CMS capsules, independent of main acceptance.

ADMITTED contains only root-admitted exact closed dependency pins.
An absent role never probes a directory. A supplied unadmitted pin is an error.
Portable verification establishes integrity; it does not mean physics passed.
"""
import hashlib
import json
import math
from pathlib import PurePosixPath
import sys


ROLES = frozenset({"current_physical_stress", "current_cms"})
ADMITTED = {
    "current_physical_stress": {
        "index": "experiments/physical-stress-current-portable-4-preflight-1/evidence-index.json",
        "sha256": "49cb1ec298bbd2f8b942b02c3e81b60b53cd2f14a5cdaa56d0fa93452c175c22",
        "verifier_sha256": "3850e1dcfbd3cecfc2c11eabd090e5ae174b0d0fbad6da11b6d94557c5e72f24",
        "summary": "qualification-summary.json",
        "summary_sha256": "6b5c2450098343b7eac059cb791697817b514fde81959c90414a9980bf9fe4d9",
        "format": "stress", "stage": "prepared",
    },
    "current_cms": {
        "index": "experiments/current-cms-replay-retention-1/evidence-index.json",
        "sha256": "7ef5f6a3f7a88449d67d56cb21e80e5ad1e75433e8e066ab07e3abf2744237ed",
        "verifier_sha256": "af78d87e1c1ba85c9c4a64e8b0e4f872d4c3daff7825d1d8464b4d71e822352c",
        "summary": "summary.json",
        "summary_sha256": "9c48193127c4366e7da61e5c4e604cce5c145abc3545063f30fdcbbd3a74a67b",
        "format": "cms", "stage": "native_terminal",
        "control_binding": {
            "proposal_index_sha256": "9c1703b96ec3933b9f8644a4850db2d9742579a109d4fdb7e72d3d776f0eba2e",
            "supervisor_index_sha256": "edf6d8cacfe1ad72cdb94584cb6d9a49e022e976c52d0668d670b702c28c84d6",
        },
        "setup_provenance": {"kind": "fresh", "source_attempt": "current CMS attempt 1",
                             "scope": "Actual source1 preparation/preflight; offline retention performs no export/build."},
        "failure_scope": "Private-link validation rejected a generated external directory symlink; no physics amplitude was evaluated.",
    },
}
MANIFEST = "ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0"
GENERATOR = "a90f80545696a648dbcfdfb8de73b1e319364edf2b93d02d4458c7b9656b8b19"
REVISION = "2c13246ffa63aa357ad1e29e5e79aff63d88dc7e"
BASELINE = "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f"
DSO = "eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9"
STRESS_SOURCE = "c192146977008a7dec8c3d1dce479eeaaf53e382ab4aad5599db6665dfe42f25"
CMS_PROPOSAL = "9c1703b96ec3933b9f8644a4850db2d9742579a109d4fdb7e72d3d776f0eba2e"
CMS_SUPERVISOR = "edf6d8cacfe1ad72cdb94584cb6d9a49e022e976c52d0668d670b702c28c84d6"
CMS_SCIENCE_BINDING = {
    "campaign_manifest_sha256": MANIFEST, "current_revision": REVISION,
    "baseline_revision": BASELINE, "current_native_dso_sha256": DSO,
    "generator_inventory_sha256": GENERATOR,
    "points_sha256": "a0b0bf140915a5b65600eda9dc6ca87f7e4749f6d3b1cab3ecf4b28616051b60",
    "physical_card_sha256": "96158fedb398f3a7957bff10db0a7101281eb57ef0e9973e9c6e43c727efb8ec",
    "point_count": 21, "MLStabThres": 0.001, "OSThres": 1e-8,
    "regression_tolerance": 3e-6, "ordinary_target": 1e-8,
    "complex_squared_mass": "M*M - i*M*Gamma",
    "width_sign": "positive widths; negative imaginary squared masses",
    "geometry_modified": False,
}
LABELS = ("baseline-1-auto", "candidate-1-auto", "candidate-6-auto",
          "candidate-7-auto", "candidate-8-auto", "candidate-9-auto")
STRESS_CASES = ("qcd_n2_gg_ttx", "qcd_n3_uux_zgg", "qcd_n4_uux_zzgg", "ew_n3_epem_ttxa")
COUNTS = {"accepted", "rejected", "unchecked", "unexecuted"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def safe(value):
    require(isinstance(value, str) and value and "\\" not in value, "Invalid current diagnostic path")
    path = PurePosixPath(value)
    require(path.parts and not path.is_absolute() and ".." not in path.parts and str(path) == value,
            "Unsafe current diagnostic path")
    return value


def decode(raw):
    def invalid(value):
        raise ValueError("Nonfinite current diagnostic JSON: " + value)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate current diagnostic JSON key")
            result[key] = value
        return result
    def finite(value):
        result = float(value)
        require(math.isfinite(result), "Nonfinite current diagnostic JSON number")
        return result
    return json.loads(raw, parse_constant=invalid, parse_float=finite, object_pairs_hook=pairs)


def label_identity(label):
    source, backend, automatic = label.split("-")
    require(automatic == "auto", "Current diagnostic is not automatic mode")
    return source, int(backend)


def expected(role):
    if role == "current_physical_stress":
        return [{"id": case + "/" + label, "case": case, "label": label,
                 "source": label_identity(label)[0], "backend": label_identity(label)[1],
                 "expected_points": 2, "expected_order_rows": None}
                for case in STRESS_CASES for label in LABELS]
    require(role == "current_cms", "Unknown current diagnostic role")
    return [{"id": label, "source": label_identity(label)[0], "backend": label_identity(label)[1],
             "expected_points": 21, "expected_order_rows": 42} for label in LABELS]


def pending(role):
    rows = [dict(row, state="pending", execution_started=None, numerically_qualified=False)
            for row in expected(role)]
    return {"status": "pending", "stage": None, "integrity_verified": False,
            "evidence": None, "amplitude_evaluations": None, "request_rows": rows,
            "coverage": {"expected_requests": len(rows), "pending_requests": len(rows),
                         "amplitude_requests_started": None, "worker_requests_started": None,
                         "completed_requests": None},
            "broad_qualified": False, "performance_eligible": False,
            "reason": "No explicit admitted closed capsule selected; active directories are not inspected."}


def default_selection():
    # Registry order is explicit source policy: adding a later closed capsule
    # never replaces the existing default merely because its namespace is newer.
    return {role: {"index": contracts(role)[0]["index"], "sha256": contracts(role)[0]["sha256"]}
            for role in ADMITTED if contracts(role)}


def contracts(role):
    value = ADMITTED.get(role, [])
    return [value] if isinstance(value, dict) else value


class IndexedReader:
    """An exact virtual capsule; .paths is its complete indexed inventory."""
    def __init__(self, evidence, pin, contract):
        self.evidence, self.contract = evidence, contract
        self.name = safe(pin["index"])
        self.prefix = str(PurePosixPath(self.name).parent)
        self.raw_index = evidence.add(self.name, pin["sha256"])
        self.index = decode(self.raw_index)
        require(type(self.index.get("schema_version")) is int and self.index["schema_version"] == 1
                and isinstance(self.index.get("files"), dict)
                and self.index["files"], "Unknown or empty current diagnostic index")
        self.files = self.index["files"]
        for name in self.files:
            safe(name)
        require("evidence-index.json" not in self.files, "Current diagnostic index indexes itself")
        self.paths = frozenset({"evidence-index.json", *self.files})
        # Authenticate every indexed physical byte, not only claim-bearing files.
        for name in self.files:
            self(name)

    def __call__(self, name):
        safe(name)
        if name == "evidence-index.json":
            return self.raw_index
        item = self.files.get(name)
        require(isinstance(item, dict) and digest(item.get("sha256")), "Unbound current diagnostic input")
        if self.contract["format"] == "stress":
            require(item.get("kind") == "file", "Current stress payload is not a regular file")
            size = item.get("size")
        else:
            require(item.get("codec") in {"utf8", "gzip-utf8"}, "Unknown CMS text codec")
            size = item.get("bytes")
        require(type(size) is int and size >= 0, "Invalid current diagnostic byte count")
        raw = self.evidence.add(self.prefix + "/" + name, item["sha256"])
        require(len(raw) == size, "Current diagnostic payload byte count differs")
        return raw


def admitted_contract(role, pin):
    require(isinstance(pin, dict) and set(pin) == {"index", "sha256"}
            and digest(pin.get("sha256")), "Current diagnostic requires an exact lowercase index pin")
    safe(pin["index"])
    matches = [item for item in contracts(role)
               if pin == {"index": item["index"], "sha256": item["sha256"]}]
    require(len(matches) == 1, "Current diagnostic capsule is not explicitly admitted")
    return matches[0]


def verified_summary(evidence, role, pin):
    contract = admitted_contract(role, pin)
    reader = IndexedReader(evidence, pin, contract)
    source = reader("verify_evidence.py")
    require(hashlib.sha256(source).hexdigest() == contract["verifier_sha256"],
            "Current diagnostic verifier source differs")
    summary_raw = reader(contract["summary"])
    require(hashlib.sha256(summary_raw).hexdigest() == contract["summary_sha256"],
            "Current diagnostic summary pin differs")
    summary = decode(summary_raw)
    require(isinstance(summary, dict), "Current diagnostic summary is not an object")
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        namespace = {"__name__": "retained_current_diagnostic", "__file__": "/__retained_text__/verify_evidence.py"}
        exec(compile(source, namespace["__file__"], "exec", optimize=0), namespace)
        proof = namespace["verify"](read_bytes=reader)
    finally:
        sys.dont_write_bytecode = previous
    require(isinstance(proof, dict) and proof.get("verified") is True
            and proof.get("index_sha256") == hashlib.sha256(reader.raw_index).hexdigest()
            and proof.get("summary") == summary, "Current diagnostic portable reanalysis differs")
    return summary, contract


def admitted_closure(role, pin, seen=()):
    contract = admitted_contract(role, pin)
    key = (pin["index"], pin["sha256"])
    require(key not in seen, "Cyclic current diagnostic ancestry")
    ancestors = contract.get("ancestors", [])
    require(isinstance(ancestors, list), "Invalid current diagnostic ancestry")
    if contract.get("setup_provenance", {}).get("kind") == "inherited":
        require(ancestors, "Inherited setup requires an explicit admitted ancestor")
    for ancestor in ancestors:
        admitted_closure(role, ancestor, (*seen, key))


def collect_one(evidence, role, pin):
    value, contract = verified_summary(evidence, role, pin)
    result = stress_preparation(value, contract) if role == "current_physical_stress" else cms(value, contract)
    result.update(integrity_verified=True, evidence=pin["index"], index_sha256=pin["sha256"],
                  retained_summary=value, setup_provenance=contract.get("setup_provenance", {"kind": "unspecified"}),
                  failure_scope=contract.get("failure_scope"),
                  ancestry=[collect_one(evidence, role, ancestor) for ancestor in contract.get("ancestors", [])])
    return result


def stress_preparation(value, contract):
    require(contract["stage"] == "prepared", "Current stress execution needs a separately admitted contract")
    fixed = {"scope": "current-source preparation and preflight only", "role": "current-physical-stress-preparation",
             "passed": True, "preparation_qualified": True, "closed_stages": ["prepare", "preflight"],
             "native_state": "pending", "candidate_revision": REVISION, "baseline_revision": BASELINE,
             "candidate_generator_sha256": GENERATOR, "campaign_manifest_sha256": MANIFEST,
             "source_index_sha256": STRESS_SOURCE, "candidate_DSO_sha256": DSO,
             "fresh_CLI_exports": 8, "fresh_full_builds": 8, "mass_preflights": 8,
             "amplitude_evaluations": 0, "physical_stress_qualified": False,
             "pending_native_requests": 24, "performance_eligible": False, "EW_analytic_oracle_qualified": False}
    require(all(value.get(key) == wanted and type(value.get(key)) is type(wanted) for key, wanted in fixed.items()),
            "Current stress preparation scope or identity differs")
    rows = value.get("request_rows")
    require(isinstance(rows, list) and len(rows) == 24, "Current stress preparation omits requests")
    normalized = []
    for row, wanted in zip(rows, expected("current_physical_stress")):
        require(all(row.get(key) == wanted[key] for key in ("case", "label", "source", "backend"))
                and type(row.get("backend")) is int
                and row.get("state") == "not_started" and row.get("numerically_qualified") is False
                and all(row.get(key) is None for key in ("observed_statuses", "incomplete_warnings", "failure")),
                "Current stress preparation request coverage or stage differs")
        normalized.append(dict(wanted, state="unexecuted", execution_started=False,
                               numerically_qualified=False, retained=row))
    return {"status": "prepared", "stage": "prepared", "amplitude_evaluations": 0,
            "scope": "Immutable preparation/preflight snapshot only; later execution is not inferred.",
            "native_execution_coverage": "pending a separate terminal execution capsule",
            "preparation": {"fresh_CLI_exports": 8, "fresh_full_builds": 8, "mass_preflights": 8},
            "request_rows": normalized, "coverage": {"expected_requests": 24, "pending_requests": 24,
                "amplitude_requests_started": 0, "worker_requests_started": 0, "completed_requests": 0},
            "gates": {"preparation_passed": True, "preflight_passed": True,
                      "host_checked": None, "full_stream": None, "agreement": None},
            "broad_qualified": False, "performance_eligible": False}


def phase(rows, counts):
    require(isinstance(rows, list) and len(rows) == 42,
            "CMS phase does not preserve all 42 order outcomes")
    require(isinstance(counts, dict) and set(counts) == COUNTS
            and all(type(n) is int and n >= 0 for n in counts.values()) and sum(counts.values()) == 42,
            "CMS phase counts omit or duplicate order outcomes")
    require(all(isinstance(row, dict) and type(row.get("point")) is int and type(row.get("order")) is int
                and row.get("state") in COUNTS for row in rows)
            and [(row["point"], row["order"]) for row in rows]
                == [(point, order) for point in range(1, 22) for order in (0, 1)],
            "CMS phase point/order coverage differs")
    observed = {name: sum(row["state"] == name for row in rows) for name in COUNTS}
    require(counts == observed, "CMS phase row/count classification differs")
    require(all(row.get("status") is None for row in rows if row["state"] == "unexecuted"),
            "Unexecuted CMS outcome contains a native status")
    require(all(row["state"] == "unchecked" for row in rows if row.get("status") == 120),
            "CMS status 120 was counted as checked")
    return counts


def cms(value, contract):
    require(type(value.get("schema_version")) is int and value["schema_version"] == 1
            and value.get("role") == "current-cms-replay"
            and value.get("closed") is True and value.get("stage") == contract["stage"]
            and contract["stage"] in {"prepared", "native_terminal", "setup_terminal"},
            "CMS capsule stage or schema differs")
    binding = value.get("binding", {})
    require(isinstance(binding, dict)
            and all(binding.get(key) == wanted and type(binding.get(key)) is type(wanted)
                    for key, wanted in CMS_SCIENCE_BINDING.items()),
            "CMS common science binding differs")
    control = contract.get("control_binding")
    require(isinstance(control, dict) and {"proposal_index_sha256", "supervisor_index_sha256"} <= set(control)
            and all(digest(wanted) and binding.get(key) == wanted for key, wanted in control.items()),
            "CMS admitted controller/proposal binding differs")
    require(isinstance(binding.get("geometry_source"), str) and binding["geometry_source"],
            "CMS serialized geometry provenance is absent")
    gates = value.get("gates", {})
    require(gates.get("independent_finite_term_oracle") is False
            and value.get("performance_eligible") is False and value.get("requested_requests") == 6,
            "CMS diagnostic cannot close broad CMS or an independent finite oracle")
    for name in ("amplitude_launch_count", "completed_requests", "worker_launch_count", "physics_loader_launch_count"):
        require(type(value.get(name)) is int and value[name] >= 0, "CMS native-phase launch accounting is absent")
    rows = value.get("requests")
    require(isinstance(rows, list) and len(rows) == 6, "CMS capsule omits requests")
    normalized, started, qualified, unchecked = [], 0, 0, 0
    for row, wanted in zip(rows, expected("current_cms")):
        require(all(row.get(key) == wanted[key] for key in ("id", "source", "backend"))
                and type(row.get("backend")) is int
                and row.get("state") in {"pending", "unexecuted", "failed", "incomplete", "host_checked"}
                and type(row.get("execution_started")) is bool
                and type(row.get("numerically_qualified")) is bool, "CMS request identity or state differs")
        outcomes, counts = row.get("phase_outcomes", {}), row.get("phase_counts", {})
        require(isinstance(outcomes, dict) and isinstance(counts, dict), "CMS phase outcomes/counts are absent")
        warmup, final = phase(outcomes.get("warmup"), counts.get("warmup")), phase(outcomes.get("final"), counts.get("final"))
        if not row["execution_started"]:
            require(warmup["unexecuted"] == final["unexecuted"] == 42
                    and row["numerically_qualified"] is False, "Unstarted CMS request has numerical outcomes")
        if row["numerically_qualified"]:
            require(row["execution_started"] and contract["stage"] == "native_terminal" and row["state"] == "host_checked"
                    and row.get("amplitude_observed") is True
                    and final["accepted"] == 42 and row.get("failure") is None,
                    "Unchecked, incomplete or failed CMS request was numerically qualified")
        started += row["execution_started"]
        qualified += row["numerically_qualified"]
        unchecked += final["unchecked"]
        normalized.append(dict(wanted, state=row["state"], execution_started=row["execution_started"],
                               numerically_qualified=row["numerically_qualified"], retained=row))
    require(qualified <= value["completed_requests"] <= started <= 6
            and started <= value["worker_launch_count"]
            and qualified <= value["amplitude_launch_count"] <= started,
            "CMS completed/started/amplitude request accounting differs")
    if "amplitude_execution_started" in gates:
        require((gates["amplitude_execution_started"] is None or type(gates["amplitude_execution_started"]) is bool)
                and (gates["amplitude_execution_started"] is True if value["amplitude_launch_count"] > 0
                     else gates["amplitude_execution_started"] is not True),
                "CMS amplitude gate differs from explicit native-phase accounting")
    if (gates.get("numerically_qualified") is True or gates.get("host_checked_full21") is True
            or gates.get("agreement_3e6") is True or gates.get("ordinary_1e8") is True):
        require(qualified == 6 and unchecked == 0, "CMS agreement or host gate bypasses failed/unchecked requests")
    stages = value.get("stages", {})
    if contract["stage"] == "prepared":
        require(gates.get("preparation_passed") is True and gates.get("preflight_passed") is True
                and started == value["amplitude_launch_count"] == value["completed_requests"] == 0
                and value["worker_launch_count"] == value["physics_loader_launch_count"] == 0
                and all(row["state"] == "pending" for row in rows)
                and stages.get("run", {}).get("status") == "not_started",
                "CMS preparation capsule contains executed or failed native work")
        status = "prepared"
    elif contract["stage"] == "setup_terminal":
        require(started == value["amplitude_launch_count"] == value["worker_launch_count"]
                == value["physics_loader_launch_count"] == value["completed_requests"] == 0
                and stages.get("run", {}).get("status") == "not_started"
                and any(type(stages.get(name, {}).get("exit_code")) is int
                        and stages[name]["exit_code"] != 0 for name in ("prepare", "preflight")),
                "CMS setup-terminal capsule does not preserve its failed setup stage")
        status = "failed"
    else:
        run = stages.get("run", {})
        require(type(run.get("exit_code")) is int and run.get("status") in {"failed", "terminal", "complete", "passed"},
                "CMS terminal capsule lacks terminal orchestration outcome")
        if "terminal_exit_code" in value:
            require(type(value["terminal_exit_code"]) is int and value["terminal_exit_code"] == run["exit_code"],
                    "CMS terminal exit accounting differs")
        status = "failed" if run["exit_code"] != 0 or value.get("overall_status") == "failed" else (
            "complete" if qualified == 6 else "incomplete")
    if gates.get("native_run_passed") is True:
        require(contract["stage"] == "native_terminal" and stages["run"]["exit_code"] == 0
                and value["completed_requests"] == started == 6,
                "CMS native-run gate bypasses unexecuted or failed orchestration")
    if value.get("primary_policy_agreement") is True:
        require(qualified == 6 and unchecked == 0,
                "CMS primary-policy agreement bypasses failed or unchecked requests")
    if value.get("five_current_reducers_complete") is True:
        require(value["completed_requests"] >= 5 and all(row["execution_started"]
                and row["phase_counts"]["final"]["unexecuted"] == 0 for row in rows if row["source"] == "candidate"),
                "CMS candidate completion gate bypasses missing streams")
    # A launch is not the number of numerical evaluations: bootstrap and repeat
    # warmup observations have separate accounting. No physics worker establishes
    # zero evaluations; launched workers need actual trace accounting instead.
    return {"status": status, "stage": contract["stage"],
            "amplitude_evaluations": 0 if started == value["worker_launch_count"] == value["amplitude_launch_count"] == 0 else None,
            "amplitude_launch_count": value["amplitude_launch_count"],
            "request_rows": normalized, "coverage": {"expected_requests": 6,
                "pending_requests": sum(row["state"] == "pending" for row in rows),
                "amplitude_requests_started": value["amplitude_launch_count"],
                "worker_requests_started": started, "completed_requests": value["completed_requests"],
                "unstarted_worker_requests": 6 - started,
                "unexecuted_amplitude_streams": 6 if value["worker_launch_count"] == 0 else None,
                "host_checked_requests": qualified,
                "amplitude_requests_unobserved": 6 - value["amplitude_launch_count"],
                "unchecked_final_order_rows": unchecked},
            "gates": {**gates, "host_checked_full21": qualified == 6,
                      "primary_policy_agreement": value.get("primary_policy_agreement"),
                      "five_current_reducers_complete": value.get("five_current_reducers_complete")},
            "broad_qualified": False, "performance_eligible": False}


def collect(evidence, selected):
    require(isinstance(selected, dict) and set(selected) <= ROLES, "Unknown current diagnostic selection")
    # Reject any unadmitted role/pin before reading either selected capsule.
    for role, pin in selected.items():
        admitted_closure(role, pin)
    results, packages, notes = {}, {}, []
    for role in sorted(ROLES):
        if role not in selected:
            result = pending(role)
        else:
            result = collect_one(evidence, role, selected[role])
            packages[role] = dict(selected[role])
        results[role] = result
        coverage = result["coverage"]
        text = (f"{result['status']}: {coverage['expected_requests']} expected automatic requests. "
                + ("No closed capsule selected; execution observations remain unknown. " if result["status"] == "pending" else
                   f"This selected closed capsule records {coverage['amplitude_requests_started']} amplitude streams started; {coverage['completed_requests']} completed. ")
                + ("Each stress stream has two saved points. " if role == "current_physical_stress" else
                   "Each CMS final stream requires 21 points and 42 order outcomes; warmup, bootstrap and final phases remain separate. ")
                + "Integrity/preparation/preflight do not establish checked full-stream agreement. Failed, unexecuted and unchecked outcomes remain retained. "
                + "No repaired-geometry EW oracle, broad stress/CMS acceptance or performance qualification follows.")
        if role == "current_physical_stress" and result["status"] == "prepared":
            text += " Eight fresh CLI exports, full builds and mass preflights passed; this immutable preparation/preflight snapshot contains zero amplitude evaluations and marks all 24 native streams unexecuted therein. It does not describe later execution; current numerical stress remains pending a separate terminal execution capsule."
        if role == "current_cms" and result["status"] == "failed" and result.get("amplitude_evaluations") == 0:
            text += (" The native orchestrator failed before any RUN-stage physics worker or amplitude evaluation; successful preparation/preflight does not relabel that failure."
                     if result["stage"] == "native_terminal" else
                     " Setup failed before a native attempt; the unexecuted streams do not become successful numerical evidence.")
        if result["status"] == "failed" and result.get("failure_scope"):
            text += " " + result["failure_scope"]
        if result.get("setup_provenance", {}).get("kind") == "inherited":
            text += " Preparation/preflight are inherited from explicitly indexed prior setup; this capsule does not claim fresh CLI exports or builds."
        if any(ancestor["status"] == "failed" for ancestor in result.get("ancestry", [])):
            text += " Explicit failed ancestor capsules and all their request rows remain retained separately; successor observations do not erase them."
        notes.append({"title": "Current physical stress coverage" if role == "current_physical_stress" else "Current CMS coverage",
                      "status": result["status"], "text": text,
                      "evidence": result["evidence"] or "no closed capsule selected"})
    return {"packages": packages, "current_campaign_diagnostics": results, "notes": notes,
            "main_campaign_modified": False, "performance_acceptance_eligible": False}
