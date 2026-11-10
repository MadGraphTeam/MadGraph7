"""Selected implementation claims are portable, scoped and append-only."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


IMPL = module("campaign_report_implementation")
REPORT = module("campaign_report")
REFRESH = module("campaign_report_refresh")


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.suffix == ".gz":
        raw = gzip.compress(raw, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def seal(root, schema, summary, name="proof", extra=None):
    files = {"summary.json": write(root / name / "summary.json", summary)}
    for path, value in (extra or {}).items():
        files[path] = write(root / name / path, value)
    identity = write(root / name / "index.json", {"schema": schema, "files": files})
    return {"index": name + "/index.json", "sha256": identity["sha256"]}


def component():
    return {"schema": "ew-invariant-fixture-retained-reparse-v2", "fixture_status": "complete",
            "fixture_passed": True, "semantic_reparse_matches_execution": True,
            "physical_amplitude": False, "performance_eligible": False,
            "full_adapter_qualified": False, "reducers": False,
            "bulk": [{"config": config, "precision": precision, "pairs": 12, "frames": 24,
                      "failed_pairs": 0, "problems": {}, "caller_input_words_unchanged": True,
                      "halting_and_rounding_unchanged": True, "ieee_state_parity_passed": True,
                      "matrix_policy_passed": True}
                     for config in ("historical-flags", "supplemental-O3") for precision in ("dp", "qp")],
            "traps": {"pairs": 8, "supported": 8, "mismatched": 0,
                      "post_call_evidence_for_trapped_calls": False},
            "mutations": [{"mutation": name, "detected_matrix_failure": True, "failed_pairs": 2}
                          for name in ("no-finite", "no-equality")]}


def ew100():
    return {"schema": "ew-v2-full100-retained-summary-v1", "worker_exit": 0, "worker_reaped": True,
            "actual_loaded_runtime_bound": True, "input_words_exact": True, "protected_inputs_unchanged": True,
            "previously_accepted_DP_answer_and_accuracy_words_exact": True,
            "global_campaign_qualified": False, "performance_eligible": False,
            "physical_points": 100, "public_calls_including_bootstrap": 101,
            "new_accepted_DP_points": 100, "new_accepted_QP_points": 0,
            "old_accepted_DP_points": 82, "old_accepted_QP_points": 18,
            "all_six_reference_comparisons_passed": True,
            "comparisons": {name: {"passed": True, "ordinary_target_passed": True,
                                   "tolerance": 3e-6, "ordinary_target_tolerance": 1e-8}
                            for name in ("baseline-1", "candidate-1", "candidate-6", "candidate-7", "candidate-8", "candidate-9")},
            "W03_lines": 0, "ieee_state_differences": 0}


def test_selected_scoped_primitives_leave_campaign_and_gates_unchanged(tmp_path):
    pins = {"symmetry_components": seal(tmp_path, "ew-invariant-fixture-component-package-v2", component()),
            "symmetry_ew100": seal(tmp_path, "ew-v2-portable-runtime-v1", ew100(), "ew")}
    evidence = REPORT.Evidence(tmp_path)
    supplement = IMPL.collect(evidence, pins)
    assert supplement["primitives"]["symmetry_components"]["component_pairs"] == 48
    assert supplement["primitives"]["symmetry_ew100"]["new_DP"] == 100
    assert supplement["main_campaign_modified"] is False
    assert supplement["performance_acceptance_eligible"] is False
    assert supplement["production_promotion"] == "pending separate receipt"
    refreshed = {"private_diagnostics": [{"title": "Unrelated", "text": "retained"}],
                 "closed_lanes": 66, "final_performance_accepted": False,
                 "old_DP": 82, "performance_status": "deferred to Hua-Sheng"}
    frozen = deepcopy(refreshed)
    IMPL.apply(refreshed, supplement)
    for key in ("closed_lanes", "final_performance_accepted", "old_DP", "performance_status"):
        assert refreshed[key] == frozen[key]
    assert refreshed["private_diagnostics"][0] == frozen["private_diagnostics"][0]
    assert len(refreshed["private_diagnostics"]) == 3
    assert IMPL.collect(REPORT.Evidence(tmp_path, evidence.files), pins) == supplement


@pytest.mark.parametrize("change", [
    lambda d: d.update(physical_amplitude=True),
    lambda d: d.update(full_adapter_qualified=True),
    lambda d: d["bulk"][0].update(caller_input_words_unchanged=False),
    lambda d: d["bulk"][0].update(precision="qp"),
    lambda d: d["traps"].update(mismatched=1),
    lambda d: d.update(mutations=[]),
])
def test_component_scope_and_negative_controls_are_required(tmp_path, change):
    value = component()
    change(value)
    pin = seal(tmp_path, "ew-invariant-fixture-component-package-v2", value)
    with pytest.raises(ValueError):
        IMPL.collect(REPORT.Evidence(tmp_path), {"symmetry_components": pin})


@pytest.mark.parametrize("change", [
    lambda d: d.update(global_campaign_qualified=True),
    lambda d: d.update(performance_eligible=True),
    lambda d: d.update(worker_reaped=False),
    lambda d: d.update(W03_lines=1),
    lambda d: d.update(new_accepted_QP_points=1),
    lambda d: d["comparisons"]["candidate-9"].update(ordinary_target_passed=False),
])
def test_ew_stream_scope_never_implies_campaign_acceptance(tmp_path, change):
    value = ew100()
    change(value)
    pin = seal(tmp_path, "ew-v2-portable-runtime-v1", value)
    with pytest.raises(ValueError):
        IMPL.collect(REPORT.Evidence(tmp_path), {"symmetry_ew100": pin})


def test_all_indexed_bytes_are_bound_before_claims(tmp_path):
    pin = seal(tmp_path, "ew-v2-portable-runtime-v1", ew100(), extra={"retained.json.gz": {"raw": 1}})
    (tmp_path / "proof/retained.json.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        IMPL.collect(REPORT.Evidence(tmp_path), {"symmetry_ew100": pin})


def test_accessor_rejects_unindexed_claim_and_decodes_stored_gzip(tmp_path):
    pin = seal(tmp_path, "unused", {}, extra={"retained.json.gz": {"raw": 1}})
    package = IMPL.Package(REPORT.Evidence(tmp_path), pin)
    assert package.json("retained.json.gz") == {"raw": 1}
    with pytest.raises(ValueError, match="Unindexed"):
        package.json("unbound.json")


@pytest.mark.parametrize("selected", [{}, {"promotion": {}},
    {"symmetry_ew100": {"index": "../escape", "sha256": "0" * 64}},
    {"symmetry_ew100": {"index": "proof/index.json", "sha256": None}}])
def test_unknown_empty_or_unsafe_selection_is_rejected(tmp_path, selected):
    with pytest.raises(ValueError):
        IMPL.collect(REPORT.Evidence(tmp_path), selected)


class LegacyEvidence:
    def json(self, path):
        if path.endswith("adaptive-qp-rescue-private-tests/attempt1/execution.json"):
            return {"complete": True, "test_result": {"failed": 0, "ignored": 0, "passed": 20},
                    "physical_amplitude_qualified": False, "production_promoted": False}
        if path.endswith("born-effective-point-native/ordinary/summary.json"):
            return {"state": "complete", "runs": [None] * 12, "comparisons": []}
        if path.endswith("born-effective-point-native/fixed-qp/exact-summary.json"):
            return {"full_rows": 24, "full_requested_view_exact_matches": 24,
                    "full_C8_coordinate_exact_matches": 24, "records": [{"ret": 120}]}
        if path.endswith("qcd-topology-rescue-higher-precision-native2/audit-result.json"):
            return {"verification": "pass", "group_calls": 232, "runtime_capture_complete": True,
                    "ordinary_parity_exact": True, "physics_campaign_accepted": False, "production_promoted": False}
        if path.endswith("default-stability-current-1/evidence/execution-1/adjudication.json"):
            return {"cases": 8, "default_dp_failures_retained": 4, "tighter_policy_qp_rescues": 4}
        if path.endswith("qcd-n4-target-triage-1/analysis.json"):
            return {"new_native_evaluations": 0, "higher_precision_adjudication": None,
                    "performance_eligible": False, "target_points": []}
        raise AssertionError("Unexpected evidence discovery: " + path)


def test_absent_optional_selection_retains_historical_claims_without_discovery():
    report = {"numerical": {"cases": [{"id": "one", "all_six_complete": True, "regression_pass": True}]}, "stress": []}
    result = REFRESH.supplement(LegacyEvidence(), {"closed_cases": ["one"], "performance_status": "deferred to Hua-Sheng"}, report)
    assert "implementation_supplement" not in result
    assert "did not load it through Fortran" in result["private_diagnostics"][0]["text"]
    assert "No backend 8/9" in result["private_diagnostics"][1]["text"]


def test_apply_replaces_only_explicitly_proved_titles():
    refreshed = {"private_diagnostics": [{"title": "Experimental adaptive-precision bridge", "text": "old"},
                                         {"title": "Installer metadata hardening remains open", "text": "pending"}]}
    proof = {"notes": [{"title": "Experimental adaptive-precision bridge", "text": "new, scoped"}]}
    IMPL.apply(refreshed, proof)
    assert refreshed["private_diagnostics"] == proof["notes"] + [
        {"title": "Installer metadata hardening remains open", "text": "pending"}]


def test_adaptive_profile_label_and_observed_release_library_are_bound(tmp_path):
    pin = seal(tmp_path, 1, {}, extra={"frozen/build1/bindings.json": {
        "build_complete": True, "physics_qualified": False, "production_promoted": False,
        "selected_release_sha256": "a" * 64}})
    package = IMPL.Package(REPORT.Evidence(tmp_path), pin)
    package.index["scope"] = ("retained release standalone tensor runtime; portable Python replay only; "
                              "no amplitude/performance qualification")
    primitive = {"adaptive_library_sha256": "a" * 64}
    IMPL.adaptive_profile(package, "adaptive_release", primitive)
    assert primitive["build_profile"] == "release"
    with pytest.raises(ValueError, match="build-profile"):
        IMPL.adaptive_profile(package, "adaptive_debug", primitive)
    with pytest.raises(ValueError, match="observed library"):
        IMPL.adaptive_profile(package, "adaptive_release", {"adaptive_library_sha256": "b" * 64})


def installer_receipt(root, change=lambda d: None):
    source, tests = {"synthetic": "source"}, {"synthetic": "tests"}
    source_sha = write(root / "installer/tirust_install.py", source)["sha256"]
    tests_sha = write(root / "installer/test_tirust_install.py", tests)["sha256"]
    receipt = {"schema": "tirust-strict-metadata-production-qualification-v1", "promoted": True,
               "pytest_returncode": 0, "passed": 27, "subtests_passed": 37,
               "actual_qualified_prefix_available": True, "actual_qualified_prefix_supports_QP": True,
               "native_build_or_evaluation_performed": False,
               "source": "madgraph/various/tirust_install.py", "tests": "tests/unit_tests/various/test_tirust_install.py",
               "source_sha256": source_sha, "tests_sha256": tests_sha,
               "observation": "synthetic receipt; no raw pytest log captured", "scope": "metadata only"}
    change(receipt)
    return seal(root, "tirust-strict-metadata-promotion-index-v1", {}, "installer",
                {"qualification.json": receipt, "tirust_install.py": source, "test_tirust_install.py": tests})


def test_installer_receipt_replaces_only_old_pending_item(tmp_path):
    pin = installer_receipt(tmp_path)
    supplement = IMPL.collect(REPORT.Evidence(tmp_path), {"installer_promotion": pin})
    assert supplement["strict_installer_promotion"] == "applied; recorded tests passed"
    assert supplement["production_promotion"] == "pending separate receipt"
    assert supplement["main_campaign_modified"] is False
    refreshed = {"private_diagnostics": [{"title": "Installer metadata hardening remains open", "text": "old"}]}
    IMPL.apply(refreshed, supplement)
    assert len(refreshed["private_diagnostics"]) == 1
    assert refreshed["private_diagnostics"][0]["title"] == "Installer metadata hardening"
    assert "no raw pytest log captured" in refreshed["private_diagnostics"][0]["text"]


@pytest.mark.parametrize("change", [
    lambda d: d.update(promoted=False),
    lambda d: d.update(pytest_returncode=False),
    lambda d: d.update(passed=0),
    lambda d: d.update(actual_qualified_prefix_supports_QP=False),
    lambda d: d.update(native_build_or_evaluation_performed=True),
    lambda d: d.update(source_sha256="0" * 64),
])
def test_installer_cannot_close_without_bound_applied_passing_receipt(tmp_path, change):
    with pytest.raises(ValueError):
        IMPL.collect(REPORT.Evidence(tmp_path), {"installer_promotion": installer_receipt(tmp_path, change)})
