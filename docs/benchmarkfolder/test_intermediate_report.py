"""Small synthetic evidence fixtures test report binding, never native code."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


REPORT = module("build_report")
INTERMEDIATE = module("intermediate_report")


def write(root, name, data):
    path = root / name
    path.write_text(json.dumps(data, sort_keys=True) + "\n")
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


@pytest.fixture
def evidence(tmp_path):
    stats = {"geometric_mean_ratio": 1.12, "lower_95": 1.10, "upper_95": 1.14}
    source = {"value": True, "performance_eligible": False, "statistics": stats}
    files = {"source.json": write(tmp_path, "source.json", source),
             "failure.json": write(tmp_path, "failure.json", {"acceptance_eligible": True, "passed": False, "statistics": stats}),
             "installer.json": write(tmp_path, "installer.json", {"source_revision": "committed-source", "native_artifacts": {"libmadloop_tirust.so": {"sha256": "actual-library"}}}),
             "qualification.json": write(tmp_path, "qualification.json", [])}
    claims = [{"id": name, "status": "partial", "scope": "diagnostic only",
               "performance_acceptance_eligible": False, "evidence": ["source.json"],
               "assertions": [{"path": "source.json", "pointer": ["value"], "expected": True}]}
              for name in sorted(INTERMEDIATE.REQUIRED)]
    data = {"schema": INTERMEDIATE.SCHEMA, "kind": "intermediate", "final_campaign_status": "pending",
            "final_campaign_accepted": False, "new_measurements_performed": False,
            "files": files, "qualification": [], "sections": [{"claims": claims}],
            "authoritative_failure": {"case": "qcd_n1_uux_z", "path": "failure.json", "statistics": stats},
            "installer_identity": {"path": "installer.json", "source_revision": "committed-source", "library_sha256": "actual-library"},
            "diagnostic_timings": [{"path": "source.json", "pointer": ["statistics"], "statistics": stats,
                                    "performance_acceptance_eligible": False}]}
    summary = {"gate": {"failed_cases": ["qcd_n1_uux_z"]},
               "cases": [{"id": "qcd_n1_uux_z", "performance": {"passed": False}}]}
    for name in ("intermediate_report.py", "collect_intermediate.py", "intermediate.typ"):
        (tmp_path / name).write_text("fixture")
    return tmp_path, data, summary


def validate(fixture):
    root, data, summary = fixture
    write(root, "intermediate-status.json", data)
    return INTERMEDIATE.validate(REPORT.ReportInputs(root), summary, required=True)


def test_valid_frozen_intermediate_evidence(evidence):
    assert validate(evidence) == evidence[1]


@pytest.mark.parametrize("change", [
    lambda d: d.update(final_campaign_accepted=True),
    lambda d: d.update(final_campaign_status="passed"),
    lambda d: d.update(new_measurements_performed=True),
    lambda d: d["sections"][0]["claims"].pop(),
    lambda d: d["sections"][0]["claims"].append(deepcopy(d["sections"][0]["claims"][0])),
    lambda d: d["sections"][0]["claims"][0].update(scope=""),
    lambda d: d["sections"][0]["claims"][0].update(performance_acceptance_eligible=True),
    lambda d: d["sections"][0]["claims"][0].update(evidence=["unbound.json"]),
    lambda d: d["installer_identity"].update(library_sha256="other-library"),
    lambda d: d["diagnostic_timings"][0].update(performance_acceptance_eligible=True),
    lambda d: d["diagnostic_timings"][0].update(statistics={"geometric_mean_ratio": .80}),
    lambda d: d["files"].update({"../escape.json": {"sha256": "fake", "bytes": 0}}),
])
def test_inflation_omission_identity_and_path_errors_are_rejected(evidence, change):
    change(evidence[1])
    with pytest.raises(ValueError):
        validate(evidence)


def test_changed_raw_evidence_fails_hash_check(evidence):
    (evidence[0] / "source.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate(evidence)


def test_resealed_source_still_must_match_claim(evidence):
    evidence[1]["files"]["source.json"] = write(evidence[0], "source.json", {"value": False})
    with pytest.raises(ValueError, match="Claim differs"):
        validate(evidence)


def test_qualified_failure_cannot_be_hidden(evidence):
    evidence[2]["gate"]["failed_cases"] = []
    with pytest.raises(ValueError, match="omits"):
        validate(evidence)


def test_future_acceptance_requires_explicit_status_refresh(evidence):
    evidence[2]["gate"]["passed"] = True
    with pytest.raises(ValueError, match="pending supplement"):
        validate(evidence)


def test_snapshot_does_not_silently_rewrite_qualification(evidence):
    evidence[1]["qualification"] = [{"status": "pass"}]
    with pytest.raises(ValueError, match="qualification differs"):
        validate(evidence)


def test_missing_supplement_only_allowed_for_legacy_fixture(evidence):
    inputs = REPORT.ReportInputs(evidence[0])
    assert INTERMEDIATE.validate(inputs, evidence[2]) is None
    with pytest.raises(ValueError, match="missing"):
        INTERMEDIATE.validate(inputs, evidence[2], required=True)


def test_preparation_rejects_supplement_without_validator(tmp_path, monkeypatch):
    write(tmp_path, "intermediate-status.json", {})
    monkeypatch.setattr(REPORT, "__file__", str(tmp_path / "build_report.py"))
    with pytest.raises(ValueError, match="validator is missing"):
        REPORT.prepare_report(tmp_path)
