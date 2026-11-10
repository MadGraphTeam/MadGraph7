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
    "current_physical_stress": [{
        "index": "experiments/physical-stress-current-portable-4-preflight-1/evidence-index.json",
        "sha256": "49cb1ec298bbd2f8b942b02c3e81b60b53cd2f14a5cdaa56d0fa93452c175c22",
        "verifier_sha256": "3850e1dcfbd3cecfc2c11eabd090e5ae174b0d0fbad6da11b6d94557c5e72f24",
        "summary": "qualification-summary.json",
        "summary_sha256": "6b5c2450098343b7eac059cb791697817b514fde81959c90414a9980bf9fe4d9",
        "format": "stress", "stage": "prepared",
    }, {
        "index": "experiments/physical-stress-current-portable-4-terminal-2/evidence-index.json",
        "sha256": "6fc2035649ba0387f0d4457784ae883592f13bb7e3b6f9c33b99e038b81f7146",
        "verifier_sha256": "20d1666c837ca09632121181b1a6a95b9d6afd677cb897598be7de80cb79a3e8",
        "summary": "qualification-summary.json",
        "summary_sha256": "267b38106bff2bf5a0bb4ef5d4bdc41081e48bb0c147a63b7bcc7ad38e94de40",
        "format": "stress", "stage": "native_terminal",
        "reader_dependency": {"prefix": "preparation", "pin": {
            "index": "experiments/physical-stress-current-portable-4-preflight-1/evidence-index.json",
            "sha256": "49cb1ec298bbd2f8b942b02c3e81b60b53cd2f14a5cdaa56d0fa93452c175c22"}},
        "ancestors": [{
            "index": "experiments/physical-stress-current-portable-4-preflight-1/evidence-index.json",
            "sha256": "49cb1ec298bbd2f8b942b02c3e81b60b53cd2f14a5cdaa56d0fa93452c175c22"}],
        "setup_provenance": {"kind": "inherited", "scope": "Exact closed preparation/preflight snapshot; no new CLI exports or builds in the terminal overlay."},
        "failure_scope": "The stage ended with a lock timeout before backend 8 launched. Four threshold streams were host checked; five of their six pairwise comparisons fail the regression criterion. One stream was deferred and 19 were not started.",
    }],
    "current_cms": [{
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
    }, {
        "index": "experiments/current-cms-replay-retention-2/evidence-index.json",
        "sha256": "aa39809c5d6cde25367454bb1fd9c3500b9aaa9b14b976fc1ff42bb0783714d3",
        "verifier_sha256": "1cf318978a06d1afedff084d9d1e727a8b25610b82d163b41220a5e5906eb0b8",
        "summary": "summary.json",
        "summary_sha256": "a84cf40930f9ef8d824ab993df6a1d4eb6f0be8694306d828777d6b8a0cb5cce",
        "format": "cms", "stage": "native_terminal",
        "control_binding": {
            "proposal_index_sha256": "59370d67a1ef278fd9de76b660beeac065bea6ebbe05b7cc8619627a98b60068",
            "supervisor_index_sha256": "50112efc92cfab1c90a3147bc2acf0ff6729a9dd7ceec0206973369fc921a934",
            "scientific_proposal_index_sha256": "9c1703b96ec3933b9f8644a4850db2d9742579a109d4fdb7e72d3d776f0eba2e",
            "ancestor_failed1_index_sha256": "7ef5f6a3f7a88449d67d56cb21e80e5ad1e75433e8e066ab07e3abf2744237ed",
            "inherited_checkpoint_sha256": "b3e38892b2cc86b679c9ed6c05bbad0b125a65ba6a67cb41eca6824dabb09949",
            "link_closure_audit_sha256": "db5663ece3fa81217a11119f556a2cffa57d9fa09da7540685632cd1b768463e",
        },
        "ancestors": [{
            "index": "experiments/current-cms-replay-retention-1/evidence-index.json",
            "sha256": "7ef5f6a3f7a88449d67d56cb21e80e5ad1e75433e8e066ab07e3abf2744237ed"}],
        "setup_provenance": {"kind": "inherited", "scope": "Exact source1 preparation/preflight retained by failed attempt1; source2 performs read-only inheritance and no new setup builds or model probes."},
        "retained_setup_provenance": {
            "kind": "inherited",
            "ancestor_index_sha256": "7ef5f6a3f7a88449d67d56cb21e80e5ad1e75433e8e066ab07e3abf2744237ed",
            "inherited_checkpoint_sha256": "b3e38892b2cc86b679c9ed6c05bbad0b125a65ba6a67cb41eca6824dabb09949",
            "inherited_stages": {"prepare": {"exit_code": 0, "session_id": 85858, "source": "attempt1"},
                                 "preflight": {"exit_code": 0, "session_id": 11912, "source": "attempt1"}},
            "read_only_inherit_session": {"exit_code": 0, "session_id": 52331},
            "successor_setup_subprocess_launch_count": 0,
            "successor_mass_probe_launch_count": 0,
            "successor_CMS_model_probe_launch_count": 0,
        },
    }],
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
# Exact public aliases of the pinned stress worker protocol. The retained
# verifier remains authoritative; these checks also reject contradictory summary fields.
STRESS_STATUS_ALIASES = {(2, 1): (1, "DP"), (3, 9): (1, "QP"),
                        (2, 6): (6, "DP"), (3, 8): (6, "QP"), (2, 7): (7, "DP"),
                        (2, 8): (8, "DP"), (3, 2): (8, "QP"),
                        (2, 9): (9, "DP"), (3, 3): (9, "QP")}


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
        if contract.get("reader_dependency"):
            aliases = self.index.get("compressed_text", {})
            require(isinstance(aliases, dict), "Unknown current stress compressed aliases")
            require(all(safe(name) != "preparation" and not name.startswith("preparation/")
                        for name in {*self.files, *aliases}),
                    "Reserved preparation namespace in current stress index")
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


class CompositeReader:
    """Two explicit indexed physical closures; compressed aliases stay internal."""
    def __init__(self, own, preparation):
        self.own, self.preparation = own, preparation
        self.raw_index, self.index = own.raw_index, own.index
        self.paths = frozenset(own.paths | {"preparation/" + name for name in preparation.paths})

    def __call__(self, name):
        safe(name)
        if name.startswith("preparation/"):
            return self.preparation(name[len("preparation/"):])
        return self.own(name)


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
    dependency = contract.get("reader_dependency")
    if dependency:
        prepared_contract = admitted_contract(role, dependency["pin"])
        preparation = IndexedReader(evidence, dependency["pin"], prepared_contract)
        require(reader.index.get("dependencies") == {"preparation": {
            "sibling": str(PurePosixPath(dependency["pin"]["index"]).parent.name),
            "index_sha256": dependency["pin"]["sha256"],
            "verifier_sha256": prepared_contract["verifier_sha256"]}},
            "Current stress dependency declaration differs")
        reader = CompositeReader(reader, preparation)
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
    if dependency:
        require(proof.get("preparation_index_sha256") == dependency["pin"]["sha256"],
                "Current stress proof omitted its exact preparation dependency")
    return summary, contract


def admitted_closure(role, pin, seen=()):
    contract = admitted_contract(role, pin)
    key = (pin["index"], pin["sha256"])
    require(key not in seen, "Cyclic current diagnostic ancestry")
    ancestors = contract.get("ancestors", [])
    require(isinstance(ancestors, list), "Invalid current diagnostic ancestry")
    if contract.get("setup_provenance", {}).get("kind") == "inherited":
        require(ancestors, "Inherited setup requires an explicit admitted ancestor")
    dependency = contract.get("reader_dependency")
    if dependency is not None:
        require(role == "current_physical_stress" and contract.get("stage") == "native_terminal"
                and isinstance(dependency, dict) and set(dependency) == {"prefix", "pin"}
                and dependency["prefix"] == "preparation" and dependency["pin"] in ancestors,
                "Unknown or unbound current diagnostic reader dependency")
        prepared = admitted_contract(role, dependency["pin"])
        require(prepared.get("format") == "stress" and prepared.get("stage") == "prepared"
                and not prepared.get("reader_dependency"), "Current stress dependency is not the preparation capsule")
    for ancestor in ancestors:
        admitted_closure(role, ancestor, (*seen, key))


def collect_one(evidence, role, pin):
    value, contract = verified_summary(evidence, role, pin)
    result = (stress_preparation(value, contract) if contract["stage"] == "prepared" else stress_terminal(value, contract)
              ) if role == "current_physical_stress" else cms(value, contract)
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


def stress_terminal(value, contract):
    """Keep terminal orchestration, checked subsets and comparisons separate."""
    dependency = contract.get("reader_dependency", {}).get("pin", {})
    fixed = {"role": "current-physical-stress-terminal",
             "scope": "four frozen physical stress streams under original strict automatic policy",
             "candidate_revision": REVISION, "candidate_generator_sha256": GENERATOR,
             "candidate_DSO_sha256": DSO, "campaign_manifest_sha256": MANIFEST,
             "source_index_sha256": STRESS_SOURCE, "prepared_dependency_index_sha256": dependency.get("sha256"),
             "requested": 24, "independent_physical_accuracy_qualified": False,
             "EW_analytic_oracle_qualified": False, "performance_eligible": False,
             "warmup_counts_are_observed_only": True}
    require(contract["stage"] == "native_terminal" and digest(dependency.get("sha256"))
            and all(value.get(key) == wanted and type(value.get(key)) is type(wanted) for key, wanted in fixed.items()),
            "Current stress terminal scope or identity differs")
    require(value.get("stage_status") in {"passed", "failed"} and type(value.get("root_exit_code")) is int
            and (value["stage_status"] == "passed") == (value["root_exit_code"] == 0),
            "Current stress terminal orchestration outcome differs")
    rows = value.get("request_rows")
    require(isinstance(rows, list) and len(rows) == 24, "Current stress terminal omits requests")
    states = {"not_started", "deferred_before_launch", "controller_failed", "physical_rejected", "host_checked_complete"}
    normalized, checked, histogram, observed_amplitudes, warmups = [], {}, {}, 0, 0
    for row, wanted in zip(rows, expected("current_physical_stress")):
        require(isinstance(row, dict) and all(row.get(key) == wanted[key] for key in ("case", "label", "source", "backend"))
                and row.get("request_id") == wanted["id"] and type(row.get("backend")) is int
                and row.get("state") in states and type(row.get("expected_final_points")) is int
                and row["expected_final_points"] == 2,
                "Current stress terminal request coverage or identity differs")
        state = row["state"]; accepted = state == "host_checked_complete"
        require(type(row.get("host_checked_complete_stream")) is bool
                and row["host_checked_complete_stream"] == accepted
                and row.get("coefficient_comparison_eligible") is accepted
                and row.get("qualified_final_points") == ([1, 2] if accepted else []),
                "Current stress unchecked or incomplete stream was qualified")
        require(type(row.get("warmup_calls_observed")) is int and row["warmup_calls_observed"] >= 0
                and type(row.get("nonfinite_rows")) is int and row["nonfinite_rows"] >= 0
                and isinstance(row.get("warmup_trace"), dict) and isinstance(row.get("final_trace"), dict)
                and isinstance(row["warmup_trace"].get("rows"), list) and isinstance(row["final_trace"].get("rows"), list),
                "Current stress phase observations are absent")
        warm, final = row["warmup_trace"]["rows"], row["final_trace"]["rows"]
        if accepted:
            require(row.get("warnings") == [] and row["nonfinite_rows"] == 0
                    and not row.get("binding_errors")
                    and all(row.get(name) is None for name in ("parser_failure", "native_failure", "controller_failure", "fatal_infrastructure_failure"))
                    and [(entry.get("point"), entry.get("order")) for entry in final] == [(1, 0), (1, 1), (2, 0), (2, 1)]
                    and row.get("raw_final_points_observed") == [1, 2],
                    "Current stress checked stream contains warnings, failure or missing final outcomes")
            for entry in final:
                require(type(entry.get("status")) is int and entry["status"] != 120
                        and entry.get("alias_backend") == wanted["backend"]
                        and entry.get("alias_precision") in {"DP", "QP"}
                        and entry["status"] // 10 % 10 in (1, 2)
                        and STRESS_STATUS_ALIASES.get((entry["status"] // 100, entry["status"] % 10))
                            == (wanted["backend"], entry["alias_precision"])
                        and isinstance(entry.get("raw_coefficients"), list) and len(entry["raw_coefficients"]) == 4
                        and entry.get("nonfinite_tokens") == [], "Current stress final outcome is unchecked")
                try:
                    numbers = [float(token) for token in [*entry["raw_coefficients"], entry["raw_accuracy"]]]
                except (KeyError, TypeError, ValueError, OverflowError) as error:
                    raise ValueError("Current stress final coefficient/accuracy token differs") from error
                require(all(math.isfinite(number) for number in numbers) and 0 <= numbers[-1] <= 1e-6,
                        "Current stress final coefficient or host accuracy is ineligible")
            checked[wanted["id"]] = row
        if state in {"not_started", "deferred_before_launch"}:
            require(not warm and not final and row["warmup_calls_observed"] == 0
                    and row.get("raw_final_points_observed") == [] and row.get("warnings") is None,
                    "Unstarted current stress request contains numerical observations")
        observed = bool(warm or final)
        observed_amplitudes += observed
        warmups += row["warmup_calls_observed"]
        histogram[state] = histogram.get(state, 0) + 1
        normalized.append(dict(wanted, state=state, execution_started=False if state in {"not_started", "deferred_before_launch"}
                               else (True if observed or state in {"physical_rejected", "host_checked_complete"} else None),
                               amplitude_observed=observed, numerically_qualified=accepted, retained=row))
    require(isinstance(value.get("state_counts"), dict)
            and all(type(number) is int for number in value["state_counts"].values())
            and value["state_counts"] == histogram and type(value.get("host_checked_complete_streams")) is int
            and value["host_checked_complete_streams"] == len(checked)
            and type(value.get("completed_orchestration")) is int
            and len(checked) <= value["completed_orchestration"] <= 24 - histogram.get("not_started", 0) - histogram.get("deferred_before_launch", 0),
            "Current stress terminal state or completion accounting differs")
    comparisons, coverage = value.get("comparisons"), value.get("case_coverage")
    require(isinstance(comparisons, list) and isinstance(coverage, list) and len(coverage) == 4,
            "Current stress comparison coverage is absent")
    wanted_pairs = {(case, left, right) for case in STRESS_CASES
                    for i, left in enumerate([key for key in checked if key.startswith(case + "/")])
                    for right in [key for key in checked if key.startswith(case + "/")][i + 1:]}
    require(all(isinstance(item, dict) and type(item.get("passed")) is bool
                and type(item.get("ordinary_target_passed")) is bool and item.get("tolerance") == 3e-6
                and item.get("ordinary_target_tolerance") == 1e-8 for item in comparisons)
            and len(comparisons) == len(wanted_pairs)
            and {(item.get("case"), item.get("left"), item.get("right")) for item in comparisons} == wanted_pairs,
            "Current stress comparisons omit, duplicate or include unchecked streams")
    for case, observed in zip(STRESS_CASES, coverage):
        count = sum(key.startswith(case + "/") for key in checked)
        pairs = [item for item in comparisons if item["case"] == case]
        subset = all(item["passed"] for item in pairs) if pairs else None
        require(observed == {"case": case, "requested_lanes": 6, "host_checked_lanes": count,
                             "six_lane_coverage": count == 6, "compared_pairs": len(pairs),
                             "compared_subset_agreement": subset,
                             "six_lane_agreement": count == 6 and len(pairs) == 15 and subset is True},
                "Current stress six-lane or subset agreement differs")
    agreement = all(item["six_lane_agreement"] for item in coverage)
    require(value.get("six_lane_all_cases_agree") is agreement,
            "Current stress partial subset was promoted to full agreement")
    return {"status": "failed" if value["root_exit_code"] != 0 else ("complete" if len(checked) == 24 else "incomplete"),
            "stage": "native_terminal", "scope": fixed["scope"], "amplitude_evaluations": None,
            "request_rows": normalized, "coverage": {"expected_requests": 24,
                "pending_requests": histogram.get("not_started", 0) + histogram.get("deferred_before_launch", 0),
                "amplitude_requests_started": observed_amplitudes, "amplitude_observations_are_trace_only": True,
                "worker_requests_started": None, "completed_requests": value["completed_orchestration"],
                "host_checked_requests": len(checked), "qualified_final_points": 2 * len(checked),
                "warmup_calls_observed": warmups, "state_counts": histogram},
            "case_coverage": coverage, "comparisons": comparisons,
            "gates": {"preparation_passed": True, "preflight_passed": True,
                      "native_orchestration_passed": value["root_exit_code"] == 0,
                      "host_checked": len(checked) == 24, "full_stream": len(checked) == 24,
                      "agreement": agreement,
                      "two_point_target_1e8": len(checked) == 24 and all(item["ordinary_target_passed"] for item in comparisons)},
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
    if "retained_setup_provenance" in contract:
        require(value.get("setup_provenance") == contract["retained_setup_provenance"]
                and contract["retained_setup_provenance"].get("kind") == "inherited"
                and any(ancestor["sha256"] == contract["retained_setup_provenance"].get("ancestor_index_sha256")
                        for ancestor in contract.get("ancestors", [])),
                "CMS inherited setup or failed ancestor binding differs")
    gates = value.get("gates", {})
    require(gates.get("independent_finite_term_oracle") is False
            and value.get("performance_eligible") is False and value.get("requested_requests") == 6,
            "CMS diagnostic cannot close broad CMS or an independent finite oracle")
    for name in ("amplitude_launch_count", "completed_requests", "worker_launch_count", "physics_loader_launch_count"):
        require(type(value.get(name)) is int and value[name] >= 0, "CMS native-phase launch accounting is absent")
    rows = value.get("requests")
    require(isinstance(rows, list) and len(rows) == 6, "CMS capsule omits requests")
    normalized, started, qualified, unchecked = [], 0, 0, 0
    phase_totals = {name: {state: 0 for state in COUNTS} for name in ("warmup", "final")}
    for row, wanted in zip(rows, expected("current_cms")):
        require(all(row.get(key) == wanted[key] for key in ("id", "source", "backend"))
                and type(row.get("backend")) is int
                and row.get("state") in {"pending", "unexecuted", "failed", "incomplete", "host_checked", "startup_timeout", "warmup_rejected"}
                and type(row.get("execution_started")) is bool
                and type(row.get("numerically_qualified")) is bool, "CMS request identity or state differs")
        outcomes, counts = row.get("phase_outcomes", {}), row.get("phase_counts", {})
        require(isinstance(outcomes, dict) and isinstance(counts, dict), "CMS phase outcomes/counts are absent")
        warmup, final = phase(outcomes.get("warmup"), counts.get("warmup")), phase(outcomes.get("final"), counts.get("final"))
        for name, values in (("warmup", warmup), ("final", final)):
            for state in COUNTS:
                phase_totals[name][state] += values[state]
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
                "phase_order_outcomes": phase_totals,
                "complete_final_streams": sum(row["phase_counts"]["final"]["unexecuted"] == 0 for row in rows),
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
        if role == "current_physical_stress" and result["stage"] == "native_terminal":
            failures = sum(not item["passed"] for item in result["comparisons"])
            text += (f" {coverage['host_checked_requests']} host-checked two-point streams supply "
                     f"{coverage['qualified_final_points']} qualified final point evaluations and "
                     f"{coverage['warmup_calls_observed']} observed warmup calls; these observations do not count hidden evaluations. "
                     f"{failures} of {len(result['comparisons'])} checked-subset comparisons fail the 3e-6 criterion. "
                     "Complete six-lane coverage and agreement remain separate gates. The retained comparator's ordinary_target_passed field applies the 1e-8 target to the whole two-point stream. "
                     "Deferred and not-started requests remain explicit; the stage outcome is not replaced by a passing subset.")
        if role == "current_cms" and result["status"] == "failed" and result.get("amplitude_evaluations") == 0:
            text += (" The native orchestrator failed before any RUN-stage physics worker or amplitude evaluation; successful preparation/preflight does not relabel that failure."
                     if result["stage"] == "native_terminal" else
                     " Setup failed before a native attempt; the unexecuted streams do not become successful numerical evidence.")
        if role == "current_cms" and result["status"] == "incomplete":
            phases = coverage["phase_order_outcomes"]
            text += (f" Native orchestration completed {coverage['completed_requests']} requests, while "
                     f"{coverage['host_checked_requests']} complete streams were host checked and "
                     f"{phases['final']['unexecuted']} final order outcomes remain unexecuted. "
                     "Individual recorded warmup acceptance does not qualify a request or imply READY/RUN/final traversal. "
                     "Worker launches, observed amplitude entry, bootstrap rows and repeated warmup observations do not determine the hidden amplitude evaluation count.")
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
