"""Closed private diagnostic roles, collected without executing stored programs.

The five roles below describe historical synthetic/private candidates. They do
not qualify the selected campaign, installed bridge, physical truth or timing.
Only the explicitly sealed package identities are admitted. ``diagnostic`` also
accepts a Package-compatible object, allowing small, labelled synthetic tests.
"""
from decimal import Decimal, InvalidOperation, localcontext, ROUND_CEILING
import gzip
import hashlib
import json
import math
from pathlib import PurePosixPath


CLOSED = {
    "rank6_feynman_sampled": ("rank6-feynman-parameter-qualification-1/index.json",
        "c2fe806d2996e7ee758e2a7894318d04388b640126b9f636fd23be99650b90d7"),
    "rank6_feynman_all_components": ("rank6-feynman-parameter-all-components-qualification-1/index.json",
        "14d13b3fc44e01916d6cc826d35b34efb4eccc9c33ada8ba70f5a6efa8b5abeb"),
    "adaptive_fixed_replay": ("adaptive-qp-physical-replay-qualification-1/evidence-index.json",
        "a728fbeb29c64e77b09189c5a97b1a509c71a681aa722dbe3affa1312895c9e9"),
    "adaptive_independent_backend8": ("adaptive-qp-physical-replay-independent-8-audit-1/evidence-index.json",
        "5b9100fc21b6d197c07401d89ec77709d9e37486f32e40f9e1e7a7e537345e32"),
    "adaptive_masked_trap": ("adaptive-qp-checked-trap-proposal-1/qualification-1/evidence-index.json",
        "d60f909d5dd20d749342c446a74b605743aa43556cb6554ba9ea8030519d967e"),
}
ROLES = frozenset(CLOSED)


def default_selection():
    return {role: {"index": "experiments/" + name, "sha256": checksum}
            for role, (name, checksum) in CLOSED.items()}


INDEX_SCHEMAS = {
    "rank6_feynman_sampled": 1,
    "rank6_feynman_all_components": 1,
    "adaptive_fixed_replay": "adaptive-physical-one-native-attempt-additive-offline-correction-v1",
    "adaptive_independent_backend8": "adaptive-native8-independent-fixed-QP-diagnostic-audit-v1",
    "adaptive_masked_trap": "private-checked-trap-one-build-one-run-portable-v1",
}
SCOPES = {
    "rank6_feynman_sampled": "closed independent numerical cross-check of seven synthetic N6/rank6 finite components only",
    "rank6_feynman_all_components": "all210finite Cartesian components of the retained synthetic N6/rank6 fixture108 only",
    "adaptive_fixed_replay": "Native process completion and same-input served-output preservation; original controller failure retained; no physical accuracy, stability, precision-tier or performance qualification",
    "adaptive_masked_trap": "Masked automatic stability/QP coverage passes; combined trapped gate fails; all original outcomes retained; no independent truth, per-call precision or performance qualification",
}
STATES = {"TIRust_QP": [32, 0, 32, 0, 1, 0, 0], "COLLIER": [32, 0, 37, 0, 1, 0, -1]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def safe(name):
    require(isinstance(name, str) and name and "\\" not in name, "Invalid diagnostic path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name,
            "Unsafe diagnostic path")
    return name


def decode(raw):
    def invalid(value):
        raise ValueError("Nonfinite diagnostic JSON: " + value)
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate diagnostic JSON key: " + key)
            value[key] = item
        return value
    def finite_float(value):
        result = float(value)
        require(math.isfinite(result), "Nonfinite diagnostic JSON number")
        return result
    return json.loads(raw, parse_constant=invalid, parse_float=finite_float, object_pairs_hook=pairs)


def number(value):
    require(isinstance(value, str), "Diagnostic decimal must retain its string representation")
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("Invalid diagnostic decimal") from error
    require(result.is_finite() and result >= 0, "Invalid diagnostic residual")
    return result


def upper_bound(value):
    # An unbounded analytic interval is explicitly inconclusive at coarse
    # orders. This sentinel is never admitted as a point estimate or residual.
    return Decimal("Infinity") if value == "Infinity" else number(value)


def display_upper(value):
    """Short conservative display; the primitive retains the exact decimal."""
    with localcontext() as context:
        context.rounding = ROUND_CEILING
        return format(number(value), ".3E").lower()


def equal(value, expected, message):
    # bool is an int subclass; JSON primitive types are part of this contract.
    require(type(value) is type(expected) and value == expected, message)


def fields(value, expected, message):
    require(isinstance(value, dict), message)
    for key, item in expected.items():
        equal(value.get(key), item, message + ": " + key)


def index_scope(package, role):
    value = package.index
    equal(value.get("schema"), INDEX_SCHEMAS[role], "Unknown diagnostic index schema")
    if role in SCOPES:
        equal(value.get("scope"), SCOPES[role], "Diagnostic index scope differs")
    if role == "rank6_feynman_sampled":
        fields(value, {"native_reducer_calls": 0, "additional_physics_evaluations_during_replay": 0,
            "reference_repositories_changed": False, "collier_cache_changed": False}, "Sampled index promoted")
    elif role == "rank6_feynman_all_components":
        fields(value, {"native_reducer_calls_in_experiment": 0, "quadrature_calls_during_replay": 0,
            "old_evidence_changed": False, "collier_cache_changed": False,
            "total_floating_point_error_formally_certified": False}, "Whole-fixture index promoted")
    elif role == "adaptive_independent_backend8":
        fields(value, {"native_executed": False, "truth_or_stability_or_performance_qualified": False},
               "Independent audit index promoted")


class Package:
    """Exact closed-role loader using the report Evidence add/json interface.

    Package-owned bytes and portable payloads are checked in full. Repository
    bindings under benchmarkfolder are also collected. Other source identities
    remain sealed historical provenance; they do not assert current-source
    qualification and are never opened via historical absolute paths.
    """
    def __init__(self, evidence, pin, role):
        require(role in ROLES and isinstance(pin, dict) and set(pin) == {"index", "sha256"},
                "Diagnostic role requires an exact index pin")
        expected_name, expected_sha = CLOSED[role]
        require(pin == {"index": "experiments/" + expected_name, "sha256": expected_sha},
                "Diagnostic selection is not an admitted closed package")
        self.evidence, self.name, self.checksum = evidence, safe(pin["index"]), pin["sha256"]
        self.prefix = str(PurePosixPath(self.name).parent)
        self.index = decode(evidence.add(self.name, self.checksum))
        index_scope(self, role)
        keys = {"files", "schema", "scope"}
        if role == "rank6_feynman_sampled":
            keys |= {"status", "inputs", "native_reducer_calls", "additional_physics_evaluations_during_replay",
                     "reference_repositories_changed", "collier_cache_changed"}
        elif role == "rank6_feynman_all_components":
            keys |= {"status", "inputs", "native_reducer_calls_in_experiment", "quadrature_calls_during_replay",
                     "old_evidence_changed", "collier_cache_changed", "total_floating_point_error_formally_certified"}
        elif role == "adaptive_independent_backend8":
            keys = {"files", "schema", "antecedent_indices", "native_executed", "truth_or_stability_or_performance_qualified"}
        require(set(self.index) == keys, "Unexpected diagnostic index fields")
        records = self.index["files"]
        self.files = {}
        if role.startswith("rank6_"):
            require(isinstance(records, list) and records, "Diagnostic files must be indexed records")
            for record in records:
                require(isinstance(record, dict) and "path" in record, "Malformed diagnostic file record")
                name = safe(record["path"])
                require(name not in self.files, "Duplicate diagnostic file path")
                self.files[name] = {k: v for k, v in record.items() if k != "path"}
        else:
            require(isinstance(records, dict) and records, "Diagnostic files must be an indexed mapping")
            self.files = records
        self._raw = {}
        for name in self.files:
            self.raw(name)
        for record in self.index.get("inputs", []):
            require(isinstance(record, dict) and set(record) <= {"path", "bytes", "sha256", "uncompressed_bytes", "uncompressed_sha256"}
                    and {"path", "bytes", "sha256"} <= set(record), "Malformed diagnostic source binding")
            name = safe(record["path"])
            require(digest(record["sha256"]) and type(record["bytes"]) is int and record["bytes"] >= 0,
                    "Invalid diagnostic source identity")
            if name.startswith("docs/benchmarkfolder/"):
                raw = evidence.add(name.removeprefix("docs/benchmarkfolder/"), record["sha256"])
                require(len(raw) == record["bytes"], "Diagnostic source size differs")
        self.inventory = {}
        if "payload-inventory.json" in self.files:
            self.inventory = self.json("payload-inventory.json")
            require(isinstance(self.inventory, dict) and self.inventory, "Empty diagnostic payload inventory")
            for name in self.inventory:
                self.payload(name)
        if "text-inputs.json.gz" in self.files:
            texts, inventory = self.json("text-inputs.json.gz"), self.json("text-inventory.json")
            require(isinstance(texts, dict) and set(texts) == set(inventory), "Diagnostic text inventory differs")
            for name, identity in inventory.items():
                safe(name)
                require(isinstance(texts[name], str) and set(identity) == {"bytes", "sha256", "origin"},
                        "Malformed diagnostic text payload")
                raw = texts[name].encode("utf-8")
                require(len(raw) == identity["bytes"] and sha(raw) == identity["sha256"],
                        "Diagnostic text identity differs")
        for name, checksum in self.index.get("antecedent_indices", {}).items():
            require(name.startswith("docs/benchmarkfolder/") and digest(checksum), "Invalid diagnostic antecedent")
            evidence.add(safe(name.removeprefix("docs/benchmarkfolder/")), checksum)

    def raw(self, name):
        require(safe(name) in self.files, "Unindexed diagnostic claim input: " + name)
        identity = self.files[name]
        require(isinstance(identity, dict) and {"sha256", "bytes"} <= set(identity)
                and set(identity) <= {"sha256", "bytes", "uncompressed_bytes", "uncompressed_sha256"}
                and digest(identity["sha256"]) and type(identity["bytes"]) is int and identity["bytes"] >= 0,
                "Invalid diagnostic file identity")
        if name not in self._raw:
            raw = self.evidence.add(self.prefix + "/" + name, identity["sha256"])
            require(len(raw) == identity["bytes"], "Diagnostic file size differs")
            require(("uncompressed_bytes" in identity) == ("uncompressed_sha256" in identity),
                    "Incomplete diagnostic compressed identity")
            if "uncompressed_sha256" in identity:
                unpacked = gzip.decompress(raw)
                require(type(identity["uncompressed_bytes"]) is int and digest(identity["uncompressed_sha256"])
                        and len(unpacked) == identity["uncompressed_bytes"]
                        and sha(unpacked) == identity["uncompressed_sha256"], "Diagnostic decompressed identity differs")
            self._raw[name] = raw
        return self._raw[name]

    def json(self, name):
        raw = self.raw(name)
        return decode(gzip.decompress(raw) if name.endswith(".gz") else raw)

    def payload(self, name):
        require(safe(name) in self.inventory, "Unindexed diagnostic text payload")
        item = self.inventory[name]
        require(isinstance(item, dict) and {"archive", "archive_sha256", "bytes", "origin", "sha256"} <= set(item)
                and set(item) <= {"archive", "archive_sha256", "bytes", "origin", "sha256", "mode"}
                and type(item["bytes"]) is int and item["bytes"] >= 0 and digest(item["sha256"])
                and digest(item["archive_sha256"]), "Invalid diagnostic payload identity")
        archive = safe(item["archive"])
        require(archive == "payloads/" + item["sha256"] + ".gz"
                and self.files[archive]["sha256"] == item["archive_sha256"], "Diagnostic payload archive differs")
        raw = gzip.decompress(self.raw(archive))
        require(len(raw) == item["bytes"] and sha(raw) == item["sha256"], "Diagnostic payload content differs")
        return raw

    def payload_json(self, name):
        return decode(self.payload(name))


def comparison_parts(components, count):
    require(isinstance(components, (dict, list)) and len(components) == count, "Diagnostic component coverage differs")
    components = list(components.values()) if isinstance(components, dict) else components
    require(len({row["index"] for row in components}) == count, "Duplicate diagnostic component")
    parts = []
    for row in components:
        require(set(row["parts"]) == {"real", "imaginary"}, "Missing diagnostic component part")
        for part in row["parts"].values():
            for backend in STATES:
                value = part[backend]
                residual = number(value["symmetric_residual"])
                for criterion, threshold in (("3e6", "3e-6"), ("1e8", "1e-8")):
                    equal(value["estimate_agrees_at_" + criterion], residual <= Decimal(threshold),
                          "Diagnostic estimate classification differs")
                interval = value["analytic_error_only_interval"]
                fields(interval, {"excludes_finite_precision_node_weight_arithmetic_error": True}, "Analytic scope promoted")
                lower, upper = number(interval["symmetric_residual_lower"]), upper_bound(interval["symmetric_residual_upper"])
                require(lower <= upper, "Invalid diagnostic analytic interval")
                equal(interval["quadrature_error_only_agreement_at_3e6"], upper <= Decimal("3e-6"), "Analytic agreement differs")
                equal(interval["quadrature_error_only_disagreement_at_3e6"], lower > Decimal("3e-6"), "Analytic disagreement differs")
            parts.append(part)
    return parts


def maximum(parts, backend, field="symmetric_residual"):
    return max((part[backend][field] for part in parts), key=number)


def rank6_sampled(package):
    value, analysis = package.json("summary.json"), package.json("analysis.json")
    fields(value, {"schema": 1, "scope": "seven Cartesian finite components of synthetic N6/R6 fixture108 only",
        "sampled_components": 7, "sampled_real_imaginary_parts": 14, "orders_binary64": [4, 6, 8, 10, 18],
        "order_decimal80": 8, "endpoint": 18, "native_reducer_calls_in_this_experiment": 0,
        "performance_qualified": False, "total_floating_point_error_formally_certified": False}, "Sampled scope differs")
    fields(analysis, {"schema": 1, "finite_truth_assigned": False, "native_reducer_calls": 0,
        "retained_backend_states": STATES}, "Sampled analysis promoted")
    orders = analysis["orders"]
    equal([row["n"] for row in orders], value["orders_binary64"], "Sampled order list differs")
    endpoint = orders[-1]
    expected = {"scalar": (0, [0, 0, 0, 0]), "odd_time": (1, [1, 0, 0, 0]),
        "time_time": (5, [2, 0, 0, 0]), "space_space": (7, [0, 2, 0, 0]),
        "index55": (55, [1, 1, 1, 1]), "index111": (111, [1, 1, 1, 2]), "index161": (161, [3, 1, 1, 1])}
    for row in orders:
        require(set(row["components"]) == set(expected), "Sampled component names differ")
        for name, (index, powers) in expected.items():
            fields(row["components"][name], {"index": index, "powers": powers}, "Sampled component mapping differs")
        comparison_parts(row["components"], 7)
    parts = comparison_parts(endpoint["components"], 7)
    disputed = [part for name in ("index55", "index111", "index161") for part in endpoint["components"][name]["parts"].values()]
    derived = {"tirust_maximum_observed_symmetric_residual": maximum(parts, "TIRust_QP"),
        "tirust_all_observed_parts_agree_at_3e6": all(p["TIRust_QP"]["estimate_agrees_at_3e6"] for p in parts),
        "tirust_all_observed_parts_agree_at_1e8": all(p["TIRust_QP"]["estimate_agrees_at_1e8"] for p in parts),
        "lowrank_convention_checks": endpoint["lowrank_convention_checks_by_estimate"],
        "collier_failed_disputed_parts_at_3e6": sum(not p["COLLIER"]["estimate_agrees_at_3e6"] for p in disputed),
        "disputed_parts": len(disputed), "collier_maximum_observed_symmetric_residual": maximum(disputed, "COLLIER"),
        "tirust_maximum_analytic_error_only_upper_residual": max((p["TIRust_QP"]["analytic_error_only_interval"]["symmetric_residual_upper"] for p in parts), key=number),
        "tirust_all_parts_agree_at_3e6_with_analytic_error_only": all(p["TIRust_QP"]["analytic_error_only_interval"]["quadrature_error_only_agreement_at_3e6"] for p in parts),
        "collier_failed_disputed_parts_with_analytic_error_only": sum(p["COLLIER"]["analytic_error_only_interval"]["quadrature_error_only_disagreement_at_3e6"] for p in disputed),
        "retained_backend_states": analysis["retained_backend_states"]}
    fields(value, derived, "Sampled summary/analysis differs")
    require(derived["tirust_all_observed_parts_agree_at_1e8"] and derived["tirust_all_parts_agree_at_3e6_with_analytic_error_only"]
            and derived["collier_failed_disputed_parts_at_3e6"] == derived["collier_failed_disputed_parts_with_analytic_error_only"] == 5,
            "Sampled closed outcome differs")
    fields(value["lowrank_convention_checks"]["TIRust_QP"], {"all_four_controls_agree_at_1e8": True}, "Sampled controls differ")
    for name, expected_sha in analysis["raw_result_sha256"].items():
        require(sha(gzip.decompress(package.raw("data/" + name + ".json.gz"))) == expected_sha,
                "Sampled raw result binding differs")
    return {"components": 7, "parts": 14, "endpoint": 18, "whole_rank_qualified": False,
        "tirust_maximum_residual": derived["tirust_maximum_observed_symmetric_residual"],
        "tirust_analytic_error_only_upper": derived["tirust_maximum_analytic_error_only_upper_residual"],
        "collier_disputed_part_misses": 5, "physical_accuracy_qualified": False,
        "total_floating_point_error_formally_certified": False, "performance_qualified": False}


def rank6_all_components(package):
    value, analysis = package.json("summary.json"), package.json("analysis.json.gz")
    fields(value, {"schema": 1, "scope": "all 210 Cartesian finite components of synthetic N6/rank6 fixture108",
        "execution_status": "completed estimates; total floating error not formally certified", "failure": None,
        "attempted_order": 18, "unavailable_proposed_orders": [], "retained_backend_states": STATES,
        "native_reducer_calls_in_experiment": 0, "performance_qualified": False,
        "total_floating_point_error_formally_certified": False}, "Whole-fixture scope differs")
    for key in ("schema", "scope", "execution_status", "failure", "attempted_order", "unavailable_proposed_orders",
                "precision", "retained_backend_states", "total_floating_point_error_formally_certified"):
        equal(analysis[key], value[key], "Whole-fixture summary/analysis differs: " + key)
    require(value["precision"]["significand_bits"] == 64, "Whole-fixture arithmetic label differs")
    require([row["n"] for row in analysis["orders"]] == [8, 16, 18]
            and len(value["orders"]) == 3, "Whole-fixture completed order coverage differs")
    for brief, row in zip(value["orders"], analysis["orders"]):
        require(brief == {key: row[key] for key in brief}, "Whole-fixture order summary differs")
        require({c["index"] for c in row["components"]} == set(range(210)), "Whole-fixture index coverage differs")
        parts = comparison_parts(row["components"], 210)
        for backend in STATES:
            summary = row["summary"][backend]
            derived = {"parts": 420, "maximum_symmetric_residual": maximum(parts, backend)}
            for criterion in ("3e6", "1e8"):
                passed = sum(p[backend]["estimate_agrees_at_" + criterion] for p in parts)
                derived.update({"pass_" + criterion: passed, "fail_" + criterion: 420 - passed})
            fields(summary, derived, "Whole-fixture aggregate differs")
            # The 1e-8 analytic enclosure is deliberately weaker than the estimates.
            for criterion, threshold in (("3e6", "3e-6"), ("1e8", "1e-8")):
                intervals = [p[backend]["analytic_error_only_interval"] for p in parts]
                agree = sum(upper_bound(x["symmetric_residual_upper"]) <= Decimal(threshold) for x in intervals)
                disagree = sum(number(x["symmetric_residual_lower"]) > Decimal(threshold) for x in intervals)
                fields(summary, {"analytic_error_only_agree_" + criterion: agree,
                    "analytic_error_only_disagree_" + criterion: disagree,
                    "analytic_error_only_inconclusive_" + criterion: 420 - agree - disagree}, "Whole-fixture analytic counts differ")
    endpoint = value["orders"][-1]["summary"]
    fields(endpoint["TIRust_QP"], {"pass_3e6": 420, "pass_1e8": 420, "analytic_error_only_agree_3e6": 420,
        "analytic_error_only_agree_1e8": 357, "analytic_error_only_inconclusive_1e8": 63}, "Whole-fixture TIRust outcome differs")
    fields(endpoint["COLLIER"], {"pass_3e6": 298, "pass_1e8": 92, "fail_3e6": 122, "fail_1e8": 328}, "Whole-fixture comparator outcome differs")
    return {"components": 210, "parts": 420, "endpoint": 18, "arithmetic_significand_bits": 64,
        "tirust_maximum_residual": endpoint["TIRust_QP"]["maximum_symmetric_residual"],
        "tirust_estimate_passes_1e8": 420, "tirust_analytic_error_only_agree_1e8": 357,
        "tirust_analytic_error_only_inconclusive_1e8": 63, "collier_passes_3e6": 298, "collier_passes_1e8": 92,
        "whole_synthetic_fixture_corroborated": True, "physical_accuracy_qualified": False,
        "total_floating_point_error_formally_certified": False, "performance_qualified": False}


def fixed_replay(package):
    value, receipt = package.json("corrected-analysis.json"), package.json("offline-receipt.json")
    fields(value, {"schema": "adaptive-physical-same-input-diagnostic-v2-offline-classification-only",
        "invariants_passed": True, "failures": [], "observed_native_calls": 400, "refusals_now_served": 37,
        "new_W03": [], "fixed_CTMode4_is_checked": False, "physical_accuracy_qualified": False,
        "stability_or_performance_qualified": False, "actual_per_call_working_precision_available": False,
        "native_execution_performed_by_this_comparison": False, "observed_sums_have_no_recorded_native_omissions": True}, "Fixed replay scope differs")
    fields(receipt, {"original_native_attempt_repeated": False, "original_controller_record_modified": False,
        "native_executed_by_offline_work": False, "offline_python_comparison_exit_code": 0,
        "Born_flat_indices_checked_exactly_at_each_point": [1, 5], "checked_stability_qualified": False,
        "physical_accuracy_qualified": False, "performance_qualified": False,
        "per_call_working_precision_available": False}, "Fixed replay successor promoted")
    refused, served = value["previously_refused_calls"], value["previously_served_calls"]
    require(len(refused) == 37 and len(served) == 363 and len(value["old_W03"]) == 74
            and all(row["old_served_output_bits_preserved"] is True for row in served)
            and all(row["old_status"] == 1 and row["new_status"] == 0 and row["new_errors"] == []
                and row["actual_working_precision_bits"] is None and row["independent_numerical_truth_available"] is False
                and set(row["new_raw_slots"]) == {"RAW_FINITE", "RAW_IR1", "RAW_IR2", "RAW_UV1"} for row in refused),
            "Fixed replay served/refused consistency differs")
    require({(r["point"], r["group"]) for r in refused + served} == {(p, g) for p in (1, 2) for g in range(1, 201)},
            "Fixed replay native call coverage differs")
    require(len(value["Born_bit_checks"]) == 2 and {r["point"] for r in value["Born_bit_checks"]} == {1, 2}
            and all(r["bits_preserved"] is True and r["flat_indices"] == [1, 5] and r["old"] == r["new"] for r in value["Born_bit_checks"]),
            "Fixed replay Born preservation differs")
    require(len(value["point_outputs"]) == 2 and all(
        row[arm]["PUBLIC_STATUS"][0][1] == "120" for row in value["point_outputs"] for arm in ("old", "new")),
        "Fixed replay status is not unchecked 120")
    original = package.payload_json("attempt/execution.json")
    fields(original, {"complete": False, "native_launched": True}, "Original failed controller was rewritten")
    fields(package.payload_json("attempt/analysis.json"), {"failures": [[1, "point_input_or_model_bits"], [2, "point_input_or_model_bits"]]},
           "Original failed classification was rewritten")
    mapping = {"new_console": "attempt/case/stdout.log", "new_journal": "attempt/case/mass-observer.hex",
        "old_console": "original/stdout.log", "old_journal": "original/mass-observer.hex", "parser": "frozen/journal-parser.py"}
    require(set(value["offline_input_sha256"]) == set(mapping) and all(
        sha(package.payload(name)) == value["offline_input_sha256"][key] for key, name in mapping.items()), "Fixed replay raw input binding differs")
    require(value["offline_analyzer_sha256"] == sha(package.raw("offline-successor/analyze.py")), "Fixed replay analyzer binding differs")
    return {"observed_native_calls": 400, "refusals_recovered": 37, "previously_served_bitwise_preserved": 363,
        "old_W03": 74, "new_W03": 0, "public_status": 120, "checked_stability_qualified": False,
        "original_controller_complete": False, "native_retry": False, "physical_accuracy_qualified": False,
        "per_call_working_precision_available": False, "performance_qualified": False}


def independent_backend8(package):
    value = package.json("analysis.json")
    fields(value, {"schema": "adaptive-fixed-QP-independent8-diagnostic-audit-v1", "failures": [],
        "physical_input_and_Born_parity_passed": True, "native_served_counts": {"8": 400, "9": 400},
        "native_integral_input_bits_identical": False, "same_input_independent_truth_qualified": False,
        "checked_stability_qualified": False, "physical_accuracy_qualified": False,
        "per_call_working_precision_available": False, "performance_qualified": False,
        "W03": {"adaptive9": [], "guarded8": []}}, "Independent backend8 scope differs")
    expected_antecedents = {"docs/benchmarkfolder/experiments/" + CLOSED["adaptive_fixed_replay"][0]: CLOSED["adaptive_fixed_replay"][1],
        "docs/benchmarkfolder/experiments/ew-qp-mass-prototype/combined-qualification-1/evidence-index.json": "4f2555b2d4e707e4308bda68bed306a0193b60959f90cccf8e2253bfc65cba73"}
    equal(package.index["antecedent_indices"], expected_antecedents, "Independent audit antecedent binding differs")
    differences = value["native_integral_input_differences"]
    require(len(differences) == 200 and {r["group"] for r in differences} == set(range(1, 201))
            and all(r["point"] == 2 and r["label"] == "NATIVE_SCALE" and r["adaptive9"] != r["backend8"] for r in differences),
            "Independent audit native scale differences were hidden")
    require(len(value["identical_nonselection_card_resource_files"]) == 7 and len(value["residuals"]) == 16,
            "Independent audit output/input coverage differs")
    maxima = {}
    for point in (1, 2):
        rows = [row for row in value["residuals"] if row["point"] == point]
        require(len(rows) == 8 and {r["flat_index"] for r in rows} == set(range(1, 9)), "Independent audit coefficient coverage differs")
        for row in rows:
            residual = number(row["symmetric_relative_difference"])
            equal(row["within_1e_8"], residual <= Decimal("1e-8"), "Independent audit residual classification differs")
            require(row["within_1e_8"] is True and row["within_3e_6"] is True, "Independent diagnostic agreement failed")
        maxima[str(point)] = max((r["symmetric_relative_difference"] for r in rows), key=number)
    return {"physical_points": 2, "calls_per_backend": 400, "native_scale_differences": 200,
        "maximum_residual_by_point": maxima, "same_physical_inputs": True, "identical_native_inputs": False,
        "checked_stability_qualified": False, "physical_accuracy_qualified": False,
        "per_call_working_precision_available": False, "performance_qualified": False}


def masked_trap(package):
    value, audit = package.json("offline-analysis.json"), package.json("independent-audit.json")
    fields(value, {"schema": "checked-trap-one-experiment-offline-outcome-v1",
        "masked_adaptive_checked_outputs_qualified": True, "masked_adaptive_QP_three_pass_coverage": True,
        "combined_checked_trap_coverage_qualified": False, "exact_hardware_exception_at_trap_identified": False,
        "independent_physical_accuracy_qualified": False, "per_call_working_precision_available": False,
        "performance_qualified": False, "native_or_loader_executed_by_reanalysis": False,
        "original_records_modified": False}, "Masked/trap scope differs")
    proof = value["raw_record_proof"]
    fields(proof, {"failures": [], "formerly_served_calls": 489, "formerly_served_outputs_preserved": True,
        "selected_input_Born_bits_match": True, "selected_input_Born_records": 101086}, "Masked raw preservation differs")
    fields(audit, {"masked_checked_QP_pass": True, "combined_trap_gate_pass": False,
        "exact_trap_exception_code_captured": False, "independent_physical_truth_or_performance_qualified": False,
        "formerly_served_preserved": 489, "selected_input_Born_records": 101086,
        "observed_pass_modes": [4, 5, 4], "original_served_by_pass": [163, 163, 163],
        "original_refused_by_pass": [37, 37, 37], "adaptive_served_by_pass": [200, 200, 200],
        "adaptive_refused_by_pass": [0, 0, 0], "check_pass_true_by_pass": [0, 200, 0],
        "cache_enabled_true_by_pass": [200, 200, 200], "ordinary_point_both_status": 219,
        "ordinary_point_accuracy": 5.432914922319511e-13, "threshold_original_status": 420,
        "threshold_original_accuracy": 1, "threshold_adaptive_status": 323,
        "threshold_adaptive_accuracy": 0}, "Independent masked audit differs")
    require(proof["original_served_records_sha256"] == proof["adaptive_served_records_sha256"] == audit["served_output_records_sha256"]
            and digest(audit["served_output_records_sha256"])
            and proof["original_input_Born_sha256"] == proof["adaptive_input_Born_sha256"] == audit["selected_input_Born_sha256"]
            and digest(audit["selected_input_Born_sha256"]), "Masked audit raw digests differ")
    require(len(proof["passes"]) == 3 and len(proof["refusal_transitions"]) == 111, "Masked pass/refusal coverage differs")
    for position, (row, mode, checked) in enumerate(zip(proof["passes"], (4, 5, 4), (0, 200, 0))):
        for arm, served, refused in (("original", 163, 37), ("adaptive", 200, 0)):
            fields(row[arm], {"calls": 200, "mode": mode, "served": served, "refused": refused,
                "cache_enabled_true": 200, "check_pass_true": checked}, "Masked pass policy differs")
    require(all(r["old_status"] == 1 and r["new_status"] == 0 and r["new_errors"] == []
                for r in proof["refusal_transitions"]), "Masked refusal transition differs")
    workers = value["workers"]
    require(set(workers) == {"original-masked", "adaptive-masked", "original-trapped", "adaptive-trapped",
        "fault-invalid", "fault-divide_by_zero", "fault-overflow"}, "Masked/trap worker coverage differs")
    for arm, status, served, refused, warnings in (("original", "420", 489, 111, 2), ("adaptive", "323", 600, 0, 0)):
        worker = workers[arm + "-masked"]
        fields(worker, {"exit_code": 0, "named_control_complete": True, "W03_count": warnings}, "Masked worker failed")
        journal = worker["journal_summary"]
        fields(journal, {"complete": True, "partial": False}, "Masked journal incomplete")
        require(len(journal["points"]) == 2, "Masked point coverage differs")
        ordinary, threshold = journal["points"]
        fields(ordinary, {"point": 1, "public_status": "219", "native_calls": 0,
            "accuracy_bits": [["R8", 1, "3D631D88E3AB0553"], ["R8", 2, "3D631D88E3AB0553"]]}, "Ordinary DP status differs")
        accuracy_word = "0000000000000000" if arm == "adaptive" else "3FF0000000000000"
        fields(threshold, {"point": 2, "public_status": status, "native_calls": 600, "served_calls": served,
            "native_refusals": refused, "pass_modes": [4, 5, 4], "complete_passes": 3, "model_snapshots": 2,
            "rotation_label_available": False, "accuracy_bits": [["R8", 1, accuracy_word], ["R8", 2, accuracy_word]]},
            "Threshold checked coverage differs")
        trapped = workers[arm + "-trapped"]
        fields(trapped, {"exit_code": -8, "named_control_complete": False}, "Trapped worker outcome changed")
        require("journal_summary" not in trapped and "journal_disposition" in trapped
                and trapped["calls"][-1] == {"index": 2, "point": 2, "returned": False, "stage": "WARMUP"},
                "Trapped journal absence or warmup fault was hidden")
    for fault in ("invalid", "divide_by_zero", "overflow"):
        worker = workers["fault-" + fault]
        fields(worker, {"exit_code": -8}, "Fault control did not trap")
        fields(worker["control_assessment"], {"fault_control_demonstrated": True, "physical_qualification": False,
            "policy_failures": []}, "Fault control coverage differs")
    fields(package.json("symbolization.json"), {"exact_hardware_exception_identified": False,
        "instruction_at_both_fault_PCs": "DIVSD %xmm1,%xmm0; bytesf20f5ec1"}, "Trap localization was promoted")
    return {"masked_checked_outputs_qualified": True, "masked_threshold_status": 323, "masked_qp_calls_served": 600,
        "previously_served_bitwise_preserved": 489, "former_refusals_recovered": 111,
        "combined_trap_gate": False, "trapped_workers_SIGFPE": 2, "fault_controls": 3,
        "fault_stage": "DP threshold warmup2", "exact_hardware_exception_identified": False,
        "explicit_rotation_identity_available": False, "physical_accuracy_qualified": False,
        "per_call_working_precision_available": False, "performance_qualified": False}


HANDLERS = {"rank6_feynman_sampled": rank6_sampled, "rank6_feynman_all_components": rank6_all_components,
    "adaptive_fixed_replay": fixed_replay, "adaptive_independent_backend8": independent_backend8,
    "adaptive_masked_trap": masked_trap}


def diagnostic(package, role):
    require(role in ROLES, "Unknown closed diagnostic role")
    try:
        index_scope(package, role)
        return HANDLERS[role](package)
    except (KeyError, TypeError, IndexError, UnicodeError, InvalidOperation) as error:
        raise ValueError("Malformed closed diagnostic package: " + role) from error


def note(role, primitive):
    p = primitive
    suffix = (" This is retained synthetic/private diagnostic evidence, separate from the current campaign's installed bridge. "
              "Physical amplitudes, general stress, production promotion and performance remain unqualified; recorded wall durations are administrative.")
    if role == "rank6_feynman_sampled":
        title = "Rank-six Feynman parameters: seven-component cross-check"
        text = (f"Independent direct integration corroborates {p['components']} sampled finite components ({p['parts']} real/imaginary parts); "
            f"maximum observed TIRust residual at most {display_upper(p['tirust_maximum_residual'])}. Five of six disputed COLLIER parts miss 3e-6, consistent with retained flag -1. "
            f"The analytic-error-only maximum upper residual is at most {display_upper(p['tirust_analytic_error_only_upper'])}. "
            "Seven samples do not qualify all 210 components. Finite node/weight/arithmetic error is checked empirically; no whole-program certificate or COLLIER cache-bug attribution is supplied.")
    elif role == "rank6_feynman_all_components":
        title = "Rank-six Feynman parameters: all-component fixture cross-check"
        text = (f"The later independent integration corroborates all {p['components']} finite components ({p['parts']} parts) of the same synthetic fixture: "
            f"420 point-estimate passes at 3e-6 and 1e-8, maximum TIRust residual at most {display_upper(p['tirust_maximum_residual'])}. "
            "COLLIER passes 298/420 at 3e-6 and 92/420 at 1e-8; original answers and flag -1 remain retained. "
            "Analytic error alone preserves 420 TIRust passes at 3e-6; at 1e-8 it gives 357 agreements and 63 inconclusive parts. "
            "The vector arithmetic has a 64-bit significand. Analytic bounds exclude finite node/weight/arithmetic error; convergence supports a practical cross-check, without a whole-program certificate or cache-bug attribution.")
    elif role == "adaptive_fixed_replay":
        title = "Adaptive fixed-QP private replay"
        text = ("The completed private replay serves 400 calls, recovers 37 former refusals, preserves 363 formerly served outputs bitwise, "
            "and changes W03 lines from 74 to zero. Both public statuses remain unchecked fixed-mode 120. The additive offline classification correction retains "
            "the original controller's complete=false failure and makes no native retry. Successful service establishes neither independent truth nor checked stability or per-call 512-bit execution.")
    elif role == "adaptive_independent_backend8":
        title = "Adaptive replay: independent backend8 audit"
        text = (f"Backend8 final sums agree with the private adaptive replay at maximum symmetric residuals at most {display_upper(p['maximum_residual_by_point']['1'])} (ordinary) "
            f"and {display_upper(p['maximum_residual_by_point']['2'])} (threshold). Both serve 400 calls with the same physical/Born inputs. "
            "All 200 threshold native scales differ: backend8 squares in DP before promotion, backend9 after promotion. This is not identical native input data or checked truth; fixed-mode status120 remains unchecked.")
    else:
        title = "Adaptive masked checks and retained trapped failure"
        text = ("Masked adaptive threshold output passes independent automatic checks with status323: 600 QP calls served, 489 former served outputs preserved "
            "bitwise and 111 former refusals recovered across observed modes [4,5,4]. Ordinary status219 is accepted in DP and adds no QP coverage. "
            "Both original and adaptive trapped workers SIGFPE during threshold DP warmup2; the combined trap gate remains false. "
            "Offline evidence localizes the shared DP Cayley DIVSD instruction; the precise hardware exception remains inferred. Explicit rotation identity and per-call precision remain unavailable.")
    return {"title": title, "text": text + suffix, "proof_role": role, "primitives": p}


def collect(evidence, selected):
    require(isinstance(selected, dict) and selected and set(selected) <= ROLES, "Unknown or empty closed diagnostic selection")
    packages = {role: Package(evidence, pin, role) for role, pin in selected.items()}
    primitives = {role: diagnostic(package, role) for role, package in packages.items()}
    notes = []
    for role, p in primitives.items():
        row = note(role, p)
        row["evidence"] = packages[role].name
        notes.append(row)
    return {"packages": {role: {"index": p.name, "sha256": p.checksum} for role, p in packages.items()},
            "primitives": primitives, "notes": notes, "main_campaign_modified": False,
            "performance_acceptance_eligible": False}
