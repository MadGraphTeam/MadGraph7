"""Portable report checks never run an amplitude, compiler or live guard."""
import hashlib
from copy import deepcopy
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


REPORT = module("campaign_report")
PAIRS = module("campaign_report_pairs")
COHORTS = module("campaign_report_cohorts")
LIFECYCLE = module("campaign_report_lifecycle")
QUALIFICATION = module("campaign_report_qualification")


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def test_frozen_snapshot_ignores_future_append_only_runs(tmp_path):
    write(tmp_path, "runs/first/summary.json", {"passed": False})
    initial = REPORT.Evidence(tmp_path)
    initial.tree("runs")
    frozen = REPORT.Evidence(tmp_path, initial.files)
    write(tmp_path, "runs/later/summary.json", {"passed": True})
    assert not frozen.exists("runs/later")
    assert frozen.tree("runs") == ["runs/first/summary.json"]
    virtual = PAIRS.snapshot_paths(frozen)(tmp_path / "runs")
    assert [p.parent.name for p in virtual.glob("*/summary.json")] == ["first"]


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a\\b"])
def test_evidence_rejects_unsafe_paths(tmp_path, name):
    with pytest.raises(ValueError):
        REPORT.Evidence(tmp_path).path(name)


def test_evidence_rejects_changed_raw_or_symlink(tmp_path):
    path = write(tmp_path, "raw.json", {"value": 1})
    initial = REPORT.Evidence(tmp_path)
    initial.add("raw.json")
    frozen = REPORT.Evidence(tmp_path, initial.files)
    path.write_text("changed")
    with pytest.raises(ValueError):
        frozen.add("raw.json")
    (tmp_path / "alias.json").symlink_to(path)
    with pytest.raises(ValueError):
        initial.tree("alias.json")


def test_frozen_parser_reads_require_inventory_and_root(tmp_path):
    write(tmp_path, "bound.json", {"value": 1})
    write(tmp_path, "unbound.json", {"value": 2})
    evidence = REPORT.Evidence(tmp_path)
    evidence.add("bound.json")
    frozen = REPORT.Evidence(tmp_path, evidence.files)
    path = PAIRS.snapshot_paths(frozen)
    assert not path(tmp_path / "unbound.json").is_file()
    with pytest.raises(ValueError, match="unbound"):
        path(tmp_path / "unbound.json").read_text()
    with path(tmp_path / "bound.json").open() as stream:
        assert json.load(stream) == {"value": 1}
    with pytest.raises(ValueError, match="outside"):
        path(tmp_path.parent / "anything").exists()
    with pytest.raises(ValueError, match="cannot write"):
        path(tmp_path / "bound.json").open("w")


@pytest.fixture
def identity():
    binding = {"id": "campaign", "manifest_sha256": "manifest"}
    source = {"revision": "registered", "files_sha256": "source", "files": {"code.f": "code"}}
    native = {"lib/" + name: name + "-hash" for name in ("libmadloop_tirust.so", "libtirust.a", "tirust-config.json")}
    manifest = {"candidate": {"exporter_identity": source, "native": {"files": native}}, "baseline": {"revision": "baseline"}}
    value = {"campaign_binding": binding, "generation": {"campaign_binding": binding,
             "source": "candidate", "source_identity": {**deepcopy(source), "revision": "later-documentation"}},
             "compilation": {"campaign_binding": binding},
             "runtime": {"libraries": {"/retained/" + name[4:]: checksum for name, checksum in native.items()}}}
    return value, manifest, binding


def test_documentary_head_does_not_change_numerical_identity(identity):
    PAIRS.validate_identity(*identity)


@pytest.mark.parametrize("change", [
    lambda d: d["generation"]["source_identity"].update(files_sha256="other"),
    lambda d: d["generation"]["source_identity"]["files"].update({"code.f": "changed"}),
    lambda d: d["runtime"]["libraries"].update({"/retained/libmadloop_tirust.so": "other"}),
    lambda d: d["compilation"].update(campaign_binding=None),
])
def test_source_native_or_campaign_substitution_fails(identity, change):
    change(identity[0])
    with pytest.raises(ValueError):
        PAIRS.validate_identity(*identity)


def test_cohort_card_requires_complete_exact_types(tmp_path):
    path = tmp_path / "MadLoopParams.dat"
    path.write_text("#MLReductionLib\n9\n#Finite\n.true.\n#Tolerance\n1d-6\n")
    expected = {"mlreductionlib": "9", "finite": True, "tolerance": 1e-6}
    assert COHORTS.card_settings(path, expected) == expected
    path.write_text(path.read_text() + "#Finite\n.false.\n")
    with pytest.raises(ValueError):
        COHORTS.card_settings(path, expected)


def test_no_cohorts_or_pairs_remain_pending(tmp_path):
    evidence = REPORT.Evidence(tmp_path)
    config = {"cases": [{"id": "one"}], "backends": [1, 6, 7, 8, 9]}
    cohorts = COHORTS.collect_cohorts(evidence, "campaigns/test", {"id": "test"}, config)
    assert cohorts["status"] == "pending" and len(cohorts["rows"]) == 5
    pairs = PAIRS.collect_pairs(evidence, "campaigns/test", {"id": "test"}, config)
    assert pairs["main_status"] == pairs["regression_status"] == "pending"


def test_main_runtime_pass_alone_cannot_complete_delivery(tmp_path, monkeypatch):
    config = {"cases": [{"id": str(i)} for i in range(11)]}
    card = write(tmp_path, "campaigns/test/cases.json", config)
    manifest = {"id": "test", "schema": "madloop-tirust-campaign-v1",
                "shared_inputs": {"files": {"cases.json": hashlib.sha256(card.read_bytes()).hexdigest()}},
                "candidate": {"exporter_identity": {"revision": "source", "files_sha256": "hash"},
                              "numerical_identity_sha256": "identity"}}
    write(tmp_path, "campaigns/test/manifest.json", manifest)
    class AllMeasured:
        @staticmethod
        def collect_numerical(*args): return {"status": "passed"}
        @staticmethod
        def collect_pairs(*args): return {"status": "passed", "main_status": "passed", "regression_status": "passed"}
        @staticmethod
        def collect_cohorts(*args): return {"status": "passed"}
        @staticmethod
        def collect_lifecycle(*args): return LIFECYCLE.collect_lifecycle(*args)
        @staticmethod
        def collect_statistics(*args): return QUALIFICATION.collect_statistics(*args)
    monkeypatch.setattr(REPORT, "document_module", lambda name: AllMeasured)
    monkeypatch.setattr(REPORT, "implementation", lambda *args: {"status": "passed"})
    stress = [{"title": title, "status": "passed", "required_for_stress_acceptance": True}
              for title in ("EW threshold mass transport", "QCD soft/collinear prototype",
                            "Complex-mass extremes")]
    stress[0]["status"] = "open"
    stress.extend([
        {"title": "Default stability estimator", "status": "open", "required_for_stress_acceptance": False},
        {"title": "QCD n3 ordinary target", "status": "open", "required_for_stress_acceptance": False},
        {"title": "QCD higher precision", "status": "pending", "required_for_stress_acceptance": False}])
    monkeypatch.setattr(REPORT, "stress_limitations", lambda *args: stress)
    observed = REPORT.collect(REPORT.Evidence(tmp_path), "test")
    assert observed["criteria"]["eight_main_runtime_gates"] == "passed"
    assert observed["delivery_complete"] is False
    assert observed["criteria"]["installed_abi_and_fortran_checks"] == "passed"
    assert observed["criteria"]["subprocess_event_cache_lifecycle"] == "pending"
    assert observed["subprocess_event_cache_lifecycle"]["evidence"] is None
    assert observed["criteria"]["statistics_roundtrip"] == "pending"
    assert observed["criteria"]["stress_accuracy"] == "open"
    stress[0]["status"] = "passed"
    observed = REPORT.collect(REPORT.Evidence(tmp_path), "test")
    assert observed["criteria"]["stress_accuracy"] == "passed"
    assert observed["delivery_complete"] is False


def test_fortran_proof_requires_same_registered_production_bytes():
    sources = {"madgraph/loop/code.py": "one", "madgraph/iolibs/helas_call_writers.py": "two"}
    proof = {"source_and_test_sha256": {"madgraph/loop/code.py": "one", "tests/test.py": "test"},
             "protocol_invocations": [{"source_sha256": sources}]}
    manifest = {"candidate": {"exporter_identity": {"files": dict(sources)}}}
    assert REPORT.validate_fortran_proof_identity(proof, manifest) == sources
    manifest["candidate"]["exporter_identity"]["files"]["madgraph/loop/code.py"] = "different"
    with pytest.raises(ValueError, match="registered production"):
        REPORT.validate_fortran_proof_identity(proof, manifest)


def test_fortran_proof_can_bind_manifest_template_identity():
    sources = {"Template/code.f": "one", "madgraph/iolibs/helas_call_writers.py": "two"}
    proof = {"source_and_test_sha256": {}, "protocol_invocations": [{"source_sha256": sources}]}
    manifest = {"candidate": {"exporter_identity": {"files": {"madgraph/iolibs/helas_call_writers.py": "two"}},
                             "numerical_sources": {"Template/code.f": {"sha256": "one", "numerical_sha256": "normalised"}}}}
    assert REPORT.validate_fortran_proof_identity(proof, manifest) == sources


def test_mixed_saved_parser_sources_are_rejected(tmp_path):
    specs = []
    for directory, source in (("first", "old parser"), ("second", "new parser")):
        raw = source.encode()
        path = tmp_path / directory / "protocol-sources/parser.py"
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
        name = directory + "/specification.json"
        write(tmp_path, name, {"protocol_sources": {"parser.py": hashlib.sha256(raw).hexdigest()}})
        specs.append(name)
    with pytest.raises(ValueError, match="different retained parser"):
        PAIRS.validate_protocol_uniformity(REPORT.Evidence(tmp_path), specs)


def test_earlier_qualified_failure_remains_authoritative():
    row = {"role": "main", "performance": {"complete": True, "passed": True},
           "existing_backend_regressions": [{"complete": True, "passed": True}],
           "retained_attempts": [{"label": "tirust-first", "acceptance_eligible": True, "passed": False}]}
    gate = {"passed": True, "failed_cases": []}
    assert PAIRS.acceptance_states([row], gate)["main_status"] == "failed"
    assert PAIRS.acceptance_states([row], gate)["status"] == "failed"
    row["retained_attempts"][0]["label"] = "backend-6"
    assert PAIRS.acceptance_states([row], gate)["regression_status"] == "failed"


def test_control_slowdown_is_descriptive_and_raw_status_survives(tmp_path):
    main = {"role": "main", "performance": {"complete": True, "passed": True},
            "existing_backend_regressions": [{"complete": True, "passed": True}]}
    control = {"id": "control", "role": "control", "status": "failed",
        "correctness": [{"status": "complete", "passed": True}],
        "retained_attempts": [{"label": "tirust-first", "directory": "tirust-first", "selected": True,
                               "acceptance_eligible": True, "passed": False}],
        "authoritative_qualified_failures": ["tirust-first"]}
    write(tmp_path, "campaign/runs/control/tirust-first/summary.json", {
        "acceptance_eligible": True, "minimum_batch_duration_met": True, "passed": False,
        "raw_observations": {"baseline": [{"seconds": .96}, {"seconds": 1.2}],
                             "candidate": [{"seconds": 2.3}, {"seconds": 2.4}]}})
    states = PAIRS.acceptance_states([main, control], {"passed": True, "failed_cases": []})
    assert set(states.values()) == {"passed"}
    detail = PAIRS.timing_details(REPORT.Evidence(tmp_path), "campaign", [control], {"minimum_batch_seconds": 1.})
    assert detail["eligibility_floor_seconds"] == .95
    assert detail["calibration_target_seconds"] == 2.
    assert detail["controls"][0]["raw_summary_status"] == "failed"
    assert detail["controls"][0]["numerical_status"] == "passed"
    assert detail["controls"][0]["runtime_acceptance_applicable"] is False
    assert detail["batch_durations"][0]["minimum_seconds"]["baseline"] == .96


@pytest.mark.parametrize("failed_indices,eligible", [([7], True), ([], False)])
def test_completed_rejected_cohort_is_failed_not_pending(tmp_path, monkeypatch, failed_indices, eligible):
    config = {"cases": [{"id": "one"}], "backends": [9]}
    write(tmp_path, "campaign/experiments/cohorts/one/specification.json",
          {"case": config["cases"][0], "source": "candidate", "backend": 9})
    groups = {name: {"indices": [], "eligible_diagnostic": True} for name in ("dp", "compensated_dp", "qp", "failed")}
    groups["dp"] = {"indices": [1], "eligible_diagnostic": eligible}
    groups["failed"]["indices"] = failed_indices
    monkeypatch.setattr(COHORTS, "reparse", lambda *args: {"status": "diagnostic", "cohorts": groups})
    result = COHORTS.collect_cohorts(REPORT.Evidence(tmp_path), "campaign", {}, config)
    assert result["status"] == result["rows"][0]["status"] == "failed"


def test_baseline_revision_alone_does_not_bind_source_bytes():
    record = {"revision": "baseline", "files": {"source.f": "hash"}, "archive_sha256": "archive"}
    value = {"campaign_binding": None, "generation": {"campaign_binding": None, "source": "baseline",
             "source_identity": {"revision": "baseline", "files_sha256": "changed", "archive_sha256": "archive"}},
             "compilation": {"campaign_binding": None}}
    with pytest.raises(ValueError, match="frozen baseline"):
        PAIRS.validate_identity(value, {"baseline": {"revision": "baseline"}}, None, record)


def test_real_retained_numerical_card_with_terminal_comment(tmp_path):
    directory = HERE / "experiments/cohorts/pilot-qcd-n1-tirust-fpe"
    expected = json.loads((directory / "inputs.json").read_text())["settings"]
    card = directory / "MadLoopParams.dat"
    assert COHORTS.card_settings(card, expected) == expected
    altered = tmp_path / "MadLoopParams.dat"
    altered.write_bytes(card.read_bytes() + b"\n#MLReductionLib\n1\n")
    with pytest.raises(ValueError, match="after numerical-card terminator"):
        COHORTS.card_settings(altered, expected)
