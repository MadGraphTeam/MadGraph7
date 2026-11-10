"""Closed guarded physical diagnostic role; authenticated portable text replay.

Masked observations and combined qualification gates are distinct. Replay only
reconstructs sealed UTF-8/gzip evidence in /tmp; historical native identities
remain provenance strings. No native artifact, worker, build or renderer runs.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import sysconfig
import tempfile


ROLE = "guarded_adaptive_physical"
ROLES = frozenset({ROLE})
INDEX = "experiments/cayley-guarded-adaptive-physical-qualification-1/evidence-index.json"
INDEX_SHA256 = "715f436c704fe1cf1f7de9c2f15a2ca1b17ba8f7453956196ceb88bc4828cd90"
INDEX_SCHEMA = "guarded-adaptive-physical-portable-evidence-v1"
INDEX_SCOPE = "Closed private two-point masked successes and new trapped failures; no promotion, independent truth or performance acceptance."
PINNED_FILES = {
    "summary.json": "9ad2730d4c924e5f2747eb610d09ee63336437e7811ef151da4b31d1f3c9cf4a",
    "verify.py": "d2bf4e52bcd32994e4c4a5c5c84c0869e655a84936c663385a0e14c60366a85f",
    "analyze.py": "7db9c9adce41ce15431f169b0c835dd1b3db82bc6eda99606a3c2e12c540bd47",
    "localization/verify.py": "b8355a5938f194cb08ee6ec232ef43cf41cd9d99d52d244db516af4305d0010c",
}
LIBRARY_SHA256 = "69722452c234f494f0aa436fa24bbabfeae59100c03b05a1f0b76336cc17886b"
OLD_LIBRARY_SHA256 = "c5de24e6c41a90caf9e34684392d0193588522f26e6bbac646bdf74e5f151cb7"
MAIN_LIBRARY_SHA256 = "eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9"
SOURCE_SHA256 = "7d9a40657b9df702fe1bdd1b3f302363813b1ba4a9514a2bcc88467f371dcd1c"
BINDING_SHA256 = "ee2933ed02f2f2c6ba1b004c0feb951553275c2967f829b5bb0efb894de7c61a"
EXECUTION_SHA256 = "18f7d89c565a97962871ee428e4edc993e7d33c9e5980da96915cff0bf22ba13"
BUILD_SHA256 = "59167eec84a767f0712b6905285a99564458ff0667427de87ce1921534b46632"
LOCALIZATION_SHA256 = "08c221e1a259d0e47595ebac67d04c4471f2efec2713c9afb7afde0d61d31264"
GATE_KEYS = {"QP_path_coverage", "host_stability_acceptance", "masked_trapped_numerical_agreement",
             "numerical_agreement", "per_call_working_precision_qualified", "trap_completion"}
FALSE_FLAGS = {"checked_trap_gate_qualified": False, "forced_qp_trap_gate_qualified": False,
               "independent_physical_truth_qualified": False, "per_call_working_precision_qualified": False,
               "performance_qualified": False}
SUMMARY_FIELDS = {"schema", "execution_sha256", "source_index_sha256", "artifact_binding_sha256", "library_sha256",
    "build_receipt_sha256", "text_inputs", "native_identity_only", "execution_complete", "diagnostic_elapsed_seconds_not_benchmark",
    "workers", "comparisons", "gate_observations", "qualified_gates", "fault_controls_demonstrated",
    "localization_evidence_index_sha256", "trapped_instruction_sites"} | set(FALSE_FLAGS)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def safe(name):
    require(isinstance(name, str) and name and "\\" not in name, "Invalid guarded-physical evidence path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name
            and "__pycache__" not in path.parts, "Unsafe guarded-physical evidence path")
    return name


def decode(raw):
    def invalid(value):
        raise ValueError("Nonfinite guarded-physical JSON: " + value)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate guarded-physical JSON key: " + key)
            result[key] = value
        return result
    def finite(value):
        result = float(value)
        require(math.isfinite(result), "Nonfinite guarded-physical JSON number")
        return result
    return json.loads(raw, parse_constant=invalid, parse_float=finite, object_pairs_hook=pairs)


def fields(value, expected, message):
    require(isinstance(value, dict), message)
    for key, wanted in expected.items():
        actual = value.get(key)
        require(type(actual) is type(wanted) and actual == wanted, message + ": " + key)


def default_selection():
    return {ROLE: {"index": INDEX, "sha256": INDEX_SHA256}}


def trusted_stdlib_path():
    """Frozen imports need stdlib only, without caller/project/site paths."""
    paths = [Path(sysconfig.get_path("stdlib")).resolve(),
             Path(sysconfig.get_path("platstdlib")).resolve(),
             Path(sysconfig.get_config_var("DESTSHARED")).resolve()]
    require(all(path.is_dir() for path in paths), "Trusted Python standard-library paths are unavailable")
    return list(dict.fromkeys(str(path) for path in paths))


class Package:
    """Authenticate the entire closed inventory and independently pin code."""
    def __init__(self, evidence, pin):
        require(isinstance(pin, dict) and set(pin) == {"index", "sha256"}
                and pin == {"index": INDEX, "sha256": INDEX_SHA256}, "Guarded physical role requires its exact closed pin")
        self.evidence, self.name, self.checksum = evidence, safe(pin["index"]), pin["sha256"]
        self.prefix = str(PurePosixPath(self.name).parent)
        self.index_raw = evidence.add(self.name, self.checksum)
        require(sha(self.index_raw) == self.checksum, "Guarded-physical index identity differs")
        self.index = decode(self.index_raw)
        require(isinstance(self.index, dict) and set(self.index) == {"schema", "scope", "files"}, "Unexpected guarded-physical index fields")
        fields(self.index, {"schema": INDEX_SCHEMA, "scope": INDEX_SCOPE}, "Unknown guarded-physical index")
        self.files = self.index["files"]
        require(isinstance(self.files, dict) and set(PINNED_FILES) <= set(self.files), "Incomplete guarded-physical source inventory")
        self._raw = {}
        for name, identity in self.files.items():
            safe(name)
            require(isinstance(identity, dict) and set(identity) == {"bytes", "sha256"}
                    and type(identity["bytes"]) is int and 0 <= identity["bytes"] <= 1048576 and digest(identity["sha256"]),
                    "Invalid guarded-physical payload identity")
            path = PurePosixPath(name)
            require(path.suffix not in {".so", ".a", ".o", ".mod", ".rlib", ".pyc", ".dll", ".dylib"} and ".so." not in path.name,
                    "Guarded physical role accepts text payloads only")
            raw = evidence.add(self.prefix + "/" + name, identity["sha256"])
            require(len(raw) == identity["bytes"] and sha(raw) == identity["sha256"], "Guarded-physical payload bytes differ")
            plain = gzip.decompress(raw) if name.endswith(".gz") else raw
            require(not raw.startswith((b"\x7fELF", b"!<arch>")) and not plain.startswith((b"\x7fELF", b"!<arch>")),
                    "Native guarded-physical payload is forbidden")
            plain.decode("utf-8")
            self._raw[name] = raw
        for name, checksum in PINNED_FILES.items():
            require(sha(self.raw(name)) == checksum, "Guarded-physical code/summary pin differs: " + name)

    def raw(self, name):
        require(safe(name) in self._raw, "Unindexed guarded-physical claim input: " + name)
        return self._raw[name]

    def json(self, name):
        raw = self.raw(name)
        return decode(gzip.decompress(raw) if name.endswith(".gz") else raw)


def replay(package):
    """Run pinned text verification without CLI, bytecode or native execution.

    The verifier imports other sealed proofs through SourceFileLoader, whose
    assertions follow process optimization. Fail closed under Python -O/-OO;
    compiling only the outer source with optimize=0 would not protect imports.
    """
    require(sys.flags.optimize == 0, "Guarded physical replay requires Python without -O/-OO so imported proof assertions remain enabled")
    for name, checksum in PINNED_FILES.items():
        require(sha(package.raw(name)) == checksum, "Untrusted guarded-physical code/summary: " + name)
    try:
        with tempfile.TemporaryDirectory(prefix="report-guarded-physical-", dir="/tmp") as temporary:
            root = Path(temporary)
            entries = {name: package.raw(name) for name in package.files}
            require("evidence-index.json" not in entries, "Guarded-physical index may not index itself")
            entries["evidence-index.json"] = package.index_raw
            for name, raw in entries.items():
                path = root / safe(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            previous_flag, previous_tempdir, previous_path = sys.dont_write_bytecode, tempfile.tempdir, list(sys.path)
            namespace = {"__file__": str(root / "verify.py"), "__name__": "_closed_guarded_physical_verifier_",
                         "__package__": None, "__cached__": None}
            try:
                sys.dont_write_bytecode = True
                # The sealed analyzer itself creates a second temporary tree.
                tempfile.tempdir = "/tmp"
                sys.path[:] = trusted_stdlib_path()
                exec(compile(package.raw("verify.py"), namespace["__file__"], "exec", dont_inherit=True, optimize=0), namespace)
                require(callable(namespace.get("verify")), "Guarded-physical verifier has no entry point")
                result = namespace["verify"]()
            finally:
                sys.dont_write_bytecode = previous_flag
                tempfile.tempdir = previous_tempdir
                sys.path[:] = previous_path
            paths = list(root.rglob("*"))
            require(not any(path.is_symlink() for path in paths)
                    and {path.relative_to(root).as_posix() for path in paths if path.is_file()} == set(entries),
                    "Guarded-physical replay changed its exact text inventory")
            require(all((root / name).read_bytes() == raw for name, raw in entries.items()),
                    "Guarded-physical replay modified retained evidence")
            return result
    except (AssertionError, KeyError, TypeError, IndexError, UnicodeError, OSError, SyntaxError) as error:
        raise ValueError("Closed guarded-physical portable verification failed") from error


def point_view(precision, code, calls, modes, models, accuracy):
    return {"accepted_precision": precision, "public_code": code, "native_calls": calls, "served_calls": calls,
            "native_refusals": 0, "groups": calls, "pass_modes": modes, "model_snapshots": models,
            "accuracy_bits": [["R8", 1, accuracy], ["R8", 2, accuracy]]}


def derive(package, verification):
    summary = package.json("summary.json")
    require(isinstance(summary, dict) and set(summary) == SUMMARY_FIELDS, "Unexpected guarded-physical summary fields")
    fields(summary, {"schema": "guarded-adaptive-physical-closed-summary-v1", "execution_complete": True,
        "execution_sha256": EXECUTION_SHA256, "source_index_sha256": SOURCE_SHA256,
        "artifact_binding_sha256": BINDING_SHA256, "library_sha256": LIBRARY_SHA256,
        "build_receipt_sha256": BUILD_SHA256, "localization_evidence_index_sha256": LOCALIZATION_SHA256,
        "text_inputs": 264, "native_identity_only": 83, "fault_controls_demonstrated": True, **FALSE_FLAGS},
        "Guarded-physical summary promoted or identity differs")
    fields(verification, {"verified": True, "payload_files": len(package.files), "text_inputs": 264,
        "native_identity_only": 83, "execution_complete_as_recorded": True,
        "native_loader_or_process_executed": False}, "Guarded-physical verifier outcome differs")
    expected_offline = {key: summary[key] for key in ("gate_observations", "qualified_gates", "fault_controls_demonstrated", *FALSE_FLAGS)}
    fields(verification, {"offline_result": expected_offline}, "Guarded-physical summary/portable replay differs")
    gates = summary["qualified_gates"]
    require(set(gates) == {"checked", "forced-qp"}, "Guarded-physical gate modes differ")
    for mode, gate in gates.items():
        require(set(gate) == GATE_KEYS, "Guarded-physical gate fields differ")
        fields(gate, {key: False for key in GATE_KEYS}, "Individual masked success promoted to a joint gate")
        observed = summary["gate_observations"][mode]
        fields(observed, {"policy_numerical_agreement": False}, "Masked/trapped agreement was invented")
        for name in ("IEEE_policy_worker_completion", "host_stability_acceptance", "numerical_to_closed_adaptive", "observed_QP_path_coverage"):
            fields(observed[name], {"masked": not (mode == "forced-qp" and name == "host_stability_acceptance"), "trapped": False},
                   "Individual masked/trapped observation differs")
    workers = summary["workers"]
    require(set(workers) == {"checked-masked", "checked-trapped", "forced-qp-masked", "forced-qp-trapped",
                            "fault-invalid", "fault-divide_by_zero", "fault-overflow"}, "Guarded-physical worker coverage differs")
    for mode in ("checked", "forced-qp"):
        for policy in ("masked", "trapped"):
            worker = workers[mode + "-" + policy]
            masked = policy == "masked"
            fields(worker, {"kind": "physical", "mode": mode, "policy": policy, "exit_code": 0 if masked else -8},
                   "Guarded-physical worker outcome differs")
            fields(worker["control_assessment"], {"actual_per_call_precision_available": False, "physical_qualification": False,
                "physical_worker_completed_with_requested_IEEE_policy": masked, "policy_failures": []}, "Guarded-physical control scope differs")
            if masked:
                fields(worker, {"journal_disposition": "complete", "retained_offline_assessment_failure": None}, "Masked journal is incomplete")
            else:
                fields(worker, {"journal_disposition": "absent after interrupted worker; no observed journal or physical qualification",
                    "numerical_assessment": None}, "Trapped journal absence was hidden")
                failure = worker["retained_offline_assessment_failure"]
                fields(failure, {"type": "FileNotFoundError"}, "Missing trapped journal failure was removed")
                require(isinstance(failure["message"], str) and "/" + mode + "-trapped/mass-observer.hex" in failure["message"],
                        "Retained trapped-journal failure names another worker")
    checked = workers["checked-masked"]["numerical_assessment"]
    fields(checked, {"checked_outputs_qualified": True, "errors": [], "independent_physical_accuracy_qualified": False,
        "qp_coverage_observed": True, "rotation_identity_qualified": False}, "Masked checked scope differs")
    require(len(checked["points"]) == 2, "Masked checked point coverage differs")
    for row, point, status, qp in zip(checked["points"], (1, 2), ([2, 1, 9], [3, 2, 3]), (False, True)):
        fields(row, {"point": point, "status": status, "errors": [], "qp_observed": qp,
            "qp_checked_sequence_observed": qp, "rotation_identity_qualified": False}, "Masked checked status differs")
    fields(workers["forced-qp-masked"]["numerical_assessment"], {"errors": [], "fixed_qp_outputs_qualified": True,
        "independent_physical_accuracy_qualified": False, "per_call_working_precision_qualified": False,
        "public120_means_unchecked": True, "stability_checked": False}, "Fixed120 was promoted to checked stability")
    for name in ("invalid", "divide_by_zero", "overflow"):
        worker = workers["fault-" + name]
        fields(worker, {"kind": "fault", "exit_code": -8}, "Fault control did not trap")
        fields(worker["control_assessment"], {"fault_control_demonstrated": True, "physical_qualification": False,
            "policy_failures": []}, "Fault control was promoted to physical qualification")
    comparisons = summary["comparisons"]
    checked_name = "closed-adaptive-checked__checked-masked"
    fixed_name = "closed-adaptive-fixed__forced-qp-masked"
    original_name = "closed-original-checked__checked-masked"
    require(set(comparisons) == {checked_name, fixed_name, original_name}, "Closed comparator coverage differs")
    checked_views = [point_view("DP", 219, 0, [], 0, "3D631D88E3AB0553"),
                     point_view("QP", 323, 600, [4, 5, 4], 2, "0000000000000000")]
    fixed_views = [point_view("unchecked_QP", 120, 200, [4], 1, "BFF0000000000000") for _ in range(2)]
    for name, views in ((checked_name, checked_views), (fixed_name, fixed_views)):
        comparison = comparisons[name]
        fields(comparison, {"numerical_agreement_reference": True, "passed": True, "ordinary_target_passed": True,
            "maximum_relative_error": 0.0, "strict_bitwise_outputs_exact": True}, "Masked adaptive-reference equality differs")
        paths = comparison["path_changes"]
        fields(paths, {"FPE_flags_can_change_rescue_selection": True, "cleaner_flags_do_not_qualify_host_stability": True,
            "path_identity_is_not_a_numerical_acceptance_requirement": True, "per_call_working_precision_qualified": False},
            "Requested/accepted QP path was promoted to per-call precision")
        require(len(paths["points"]) == 2, "Adaptive-reference point coverage differs")
        for row, point, view in zip(paths["points"], (1, 2), views):
            fields(row, {"point": point, "accepted_precision_changed": False, "observed_call_count_changed": False,
                "status_or_accuracy_or_pass_usage_changed": False, "before": view, "after": view}, "Adaptive-reference path or input outcome differs")
    original = comparisons[original_name]
    fields(original, {"numerical_agreement_reference": False, "passed": True, "ordinary_target_passed": False,
        "maximum_relative_error": 0.9241097961391376, "strict_bitwise_outputs_exact": False}, "Original diagnostic was promoted to numerical reference")
    original_points = original["path_changes"]["points"]
    require(len(original_points) == 2, "Original comparator point coverage differs")
    fields(original_points[1]["before"], {"public_code": 420, "accepted_precision": "not_accepted", "native_refusals": 111,
        "served_calls": 489}, "Original threshold failure was hidden")
    fields(original_points[1]["after"], checked_views[1], "Guarded threshold result differs from checked reference")
    sites = summary["trapped_instruction_sites"]
    require(len(sites) == 2 and {row["case"] for row in sites} == {"checked-trapped", "forced-qp-trapped"}, "Trapped localization coverage differs")
    expected_sites = {"checked-trapped": ("0x11fcecb", "divsd %xmm1,%xmm0", "iregi_rust::direct::exec::gram::gram_indicator::<f64>"),
        "forced-qp-trapped": ("0x2523cc3", "divsd %xmm0,%xmm1", "oneloop::native::dilog::series::<numerica::domains::float::multiprecision::Float>")}
    for row in sites:
        address, instruction, symbol = expected_sites[row["case"]]
        fields(row, {"ELF_address": address, "instruction": instruction, "symbol": symbol,
            "not_in_cayley_cancellation_symbol": True, "actual_register_operands_or_exception_subtype_available": False},
            "Trap instruction localization was promoted to hardware operands/subtype")
    manifest = package.json("manifest.json")
    fields(manifest, {"schema": "guarded-physical-portable-text-v1", "native_payloads_retained": False,
        "historical_native_paths_required": False, "native_loader_or_process_executed_by_capture": False}, "Portable capture scope differs")
    identities = manifest["native_identities"]
    require(len(identities) == 83 and all(row["payload_retained"] is False for row in identities.values()), "Native identities became payloads")
    expected_libraries = {
        ".tirust-build/cayley-guarded-adaptive-physical-proposal-2/run1/runtime-guarded/libmadloop_tirust.so": LIBRARY_SHA256,
        ".tirust-build/adaptive-qp-rescue-release-build/attempt1/runtime/libmadloop_tirust_adaptive_private.so": OLD_LIBRARY_SHA256,
        ".tirust-build/qp-mass-general-implementation/combined-qualification/current-runtime/libmadloop_tirust.so": MAIN_LIBRARY_SHA256}
    require(len(set(expected_libraries.values())) == 3 and all(identities[name]["sha256"] == checksum
                for name, checksum in expected_libraries.items()), "Private/predecessor/campaign library identities were conflated")
    audit_failures = []
    for number in (1, 2):
        failure = package.json(f"audit-attempt{number}/failure.json")
        fields(failure, {"schema": f"guarded-portable-audit-harness-failure-v{number}", "exit_code": 1,
            "native_or_loader_executed": False, "numerical_runtime_or_captured_evidence_modified": False},
            "Retained optional verifier-test failure changed scope")
        audit_failures.append(failure)
    return {"execution_complete_transport_only": True, "checked_masked_statuses": [219, 323], "checked_masked_QP_calls": 600,
        "forced_masked_statuses": [120, 120], "forced_masked_QP_calls": 400, "forced_masked_stability_checked": False,
        "masked_adaptive_reference_outputs_bitwise_exact": True, "comparisons": comparisons,
        "qualified_gates": gates, "gate_observations": summary["gate_observations"],
        "trapped_workers_SIGFPE": 2, "trapped_journals_absent": True, "trapped_journal_failures": {
            name: workers[name]["retained_offline_assessment_failure"] for name in ("checked-trapped", "forced-qp-trapped")},
        "trapped_instruction_sites": sites, "fault_controls_demonstrated": 3,
        "library_sha256": LIBRARY_SHA256, "old_private_library_sha256": OLD_LIBRARY_SHA256,
        "campaign_library_sha256": MAIN_LIBRARY_SHA256, "libraries_distinct": True,
        "build_receipt_sha256": BUILD_SHA256, "source_index_sha256": SOURCE_SHA256,
        "retained_optional_verifier_test_failures": audit_failures, "portable_text_replay_passed": True,
        "native_artifact_read_or_loaded_by_collector": False, "production_promoted": False,
        "explicit_rotation_identity_qualified": False, **FALSE_FLAGS}


def note(primitive):
    return {"title": "Guarded adaptive physical replay: masked equality, new trapped failures",
        "text": ("The guarded private bridge preserves both closed adaptive masked references bitwise. Checked mode accepts the ordinary point in DP "
            "(status219) and the threshold point in QP (status323), with 600 QP calls in modes [4,5,4]. Forced masked QP serves 400 calls; "
            "status120 and accuracy -1 remain unchecked. The original-backend comparison remains a nonreference diagnostic: its permissive regression passes "
            "despite maximum relative residual 0.9241097961, nonidentical outputs and the retained threshold status420 with 111 refusals. "
            "Both guarded trapped workers SIGFPE before observer journals; their FileNotFoundError assessments remain retained. All joint numerical, host-stability, "
            "QP-coverage and trap gates remain false. Offline localization identifies other DIVSD sites in DP gram_indicator and multiprecision dilogarithm series; "
            "the actual operands and exact exception subtype remain unknown. The 69722452 guarded library, old c5de24e6 private library and campaign eb3d67fd bridge remain distinct. "
            "Two optional verifier-test policy failures are retained separately from the passing portable replay and physical outcomes. "
            "This two-point private diagnostic establishes neither independent physical truth, explicit rotation identity, per-call working precision, production promotion nor runtime performance. "
            "Recorded execution duration is administrative; precise timing remains a local TODO."),
        "proof_role": ROLE, "primitives": primitive}


def collect(evidence, selected):
    require(isinstance(selected, dict) and set(selected) == ROLES, "Unknown or empty guarded-physical role selection")
    package = Package(evidence, selected[ROLE])
    try:
        primitive = derive(package, replay(package))
    except (KeyError, TypeError, IndexError, UnicodeError) as error:
        raise ValueError("Malformed closed guarded-physical evidence") from error
    row = note(primitive)
    row["evidence"] = package.name
    return {"packages": {ROLE: {"index": package.name, "sha256": package.checksum}},
            "primitives": {ROLE: primitive}, "notes": [row], "main_campaign_modified": False,
            "performance_acceptance_eligible": False}
