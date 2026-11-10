"""Prevent diagnostic scope or artifact identity from becoming acceptance."""
from copy import deepcopy
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


REPORT = module("campaign_report")
Q = module("campaign_report_qualification")


def read(name):
    return json.loads((HERE / name).read_text())


@pytest.fixture
def manifest():
    return read("campaigns/split-dp-terminal-1/manifest.json")


def test_actual_pinned_packages_and_frozen_replay_preserve_all_limits(manifest):
    evidence = REPORT.Evidence(HERE)
    stats = Q.collect_statistics(evidence, manifest)
    rows = Q.collect_stress(evidence, manifest)
    assert stats["status"] == "passed" and stats["tests"] == 10 and stats["compiled_fortran_subtests"] == 4
    assert len(stats["registered_production_sources"]) == 10
    assert [row["status"] for row in rows] == ["open"] * 5 + ["pending"]
    assert [row["required_for_stress_acceptance"] for row in rows] == [True, True, True, False, True, False]
    assert [row["native_library_sha256"] for row in rows] == [Q.HISTORICAL] * 3 + [Q.CURRENT, None, None]
    assert "unchecked" in rows[0]["text"] and "incomplete" in rows[0]["text"]
    assert "51" in rows[1]["text"] and "status 420" in rows[1]["text"]
    assert "not replayed on current eb3d" in rows[2]["text"]
    assert "same 28" in rows[3]["text"] and "unchecked status 120" in rows[3]["text"]
    assert rows[-1]["evidence"] is None and "512-bit capture is pending" in rows[-1]["text"]
    assert set(Q.PINS).issubset(evidence.files)
    assert not any("higher-precision-runner" in name for name in evidence.files)
    frozen = REPORT.Evidence(HERE, deepcopy(evidence.files))
    assert Q.collect_statistics(frozen, manifest) == stats
    assert Q.collect_stress(frozen, manifest) == rows


@pytest.fixture
def qualified_stress():
    return [{"title": title, "status": "passed", "required_for_stress_acceptance": True}
            for title in ("EW threshold mass transport", "QCD soft/collinear prototype",
                          "Default stability estimator", "Complex-mass extremes")]


def test_optional_target_and_strategy_disclosures_do_not_add_stress_gates(qualified_stress):
    rows = qualified_stress + [
        {"title": "QCD n3 ordinary target", "status": "open", "required_for_stress_acceptance": False},
        {"title": "QCD higher precision", "status": "pending", "required_for_stress_acceptance": False}]
    assert REPORT.stress_acceptance(rows) == "passed"


@pytest.mark.parametrize("status", ["open", "pending", "failed"])
def test_required_unqualified_stress_still_blocks_acceptance(qualified_stress, status):
    qualified_stress[0]["status"] = status
    assert REPORT.stress_acceptance(qualified_stress) == "open"


@pytest.mark.parametrize("mutate", [
    lambda rows: rows.clear(),
    lambda rows: rows.pop(),
    lambda rows: rows[0].pop("required_for_stress_acceptance"),
    lambda rows: rows[0].update(required_for_stress_acceptance=False),
    lambda rows: rows[0].update(required_for_stress_acceptance=1),
    lambda rows: rows.__setitem__(0, deepcopy(rows[1])),
    lambda rows: rows.append({**rows[0], "status": "open", "required_for_stress_acceptance": False}),
])
def test_missing_or_misclassified_required_stress_cannot_pass(qualified_stress, mutate):
    mutate(qualified_stress)
    assert REPORT.stress_acceptance(qualified_stress) == "open"


def test_optional_only_disclosures_cannot_vacuously_pass_stress_acceptance():
    assert REPORT.stress_acceptance([
        {"title": "QCD n3 ordinary target", "status": "passed", "required_for_stress_acceptance": False}]) == "open"


def test_old_frozen_snapshot_does_not_discover_new_qualifications(manifest):
    evidence = REPORT.Evidence(HERE, {})
    assert Q.collect_statistics(evidence, manifest)["status"] == "pending"
    assert all(row["status"] == "pending" for row in Q.collect_stress(evidence, manifest))
    assert not evidence.files


@pytest.fixture
def statistics_args(manifest):
    return [read(Q.STATISTICS + "manifest.json"), read(Q.STATISTICS + "before.json"),
            read(Q.STATISTICS + "results.json"), (HERE / Q.STATISTICS / "pytest-output.txt").read_text(), manifest]


@pytest.mark.parametrize("mutate", [
    lambda a: a[0].update(historical92_pass_invocation_recovered=True),
    lambda a: a[0].update(native_amplitude_or_rust_reducer_executed=True),
    lambda a: a[0].update(compiled_fortran_subtests=0),
    lambda a: a[2].update(returncode=1),
    lambda a: a[2].update(timed_out=True),
    lambda a: a[2]["source_sha256_after"].update({"Template/LO/Source/dsample.f": "changed"}),
    lambda a: a[1]["argv"].__setitem__(2, "0"),
    lambda a: a.__setitem__(3, a[3].replace("10 passed, 4 subtests passed", "10 passed")),
    lambda a: a.__setitem__(3, a[3].replace("test_old_xml_is_unchanged PASSED", "test_old_xml_is_unchanged FAILED")),
    lambda a: a[4]["candidate"]["exporter_identity"]["files"].update({"Template/LO/Source/dsample.f": "changed"}),
])
def test_statistics_cannot_promote_missing_execution_or_other_sources(statistics_args, mutate):
    mutate(statistics_args)
    with pytest.raises(ValueError):
        Q.statistics_scope(*statistics_args)


@pytest.fixture
def ew_args():
    analysis = read(Q.EW + "geometry-audit/analysis.json")
    return [read(Q.EW + "evidence-index.json"), read(Q.EW + "geometry-audit/evidence-index.json"),
            read(Q.EW + "summary.json"), {"geometry_records": [None] * 800, "native_summary": analysis["native_summary"]},
            read(Q.EW + "run.json")]


@pytest.mark.parametrize("mutate", [
    lambda a: a[0].update(physical_accuracy_qualified=True),
    lambda a: a[1].update(threshold_tirust_aggregate_complete=True),
    lambda a: a[1].update(native_execution_performed=True),
    lambda a: a[2]["cases"]["candidate-9"].update(refusals=[0, 0]),
    lambda a: a[3]["native_summary"]["candidate-9"]["points"][1].update(public_status=[["INTEGER", "323", "FFFFF"]]),
    lambda a: a[4]["cases"][0]["loaded_libraries"]["libmadloop_tirust.so"].update(sha256=Q.CURRENT),
])
def test_ew_unchecked_incomplete_historical_capture_cannot_be_promoted(ew_args, mutate):
    mutate(ew_args)
    with pytest.raises(ValueError):
        Q.ew_scope(*ew_args)


@pytest.mark.parametrize("mutate", [
    lambda a, g: a.update(physics_accepted=True),
    lambda a, g: a["normalization"]["topology-c:9"].update(host_code=323),
    lambda a, g: a["normalization"]["topology-c:8"].update(independent_finite_single_truth_available=True),
    lambda a, g: a["normalization"]["topology-b:8"].update(passes_3e_6_double_pole_only=True),
    lambda a, g: g.update(historical_library_sha256=Q.CURRENT),
])
def test_qcd_partial_or_one_pole_results_remain_unqualified(mutate):
    args = [read(Q.QCD + "analysis1/adjudication.json"), read(Q.QCD + "analysis1/guard-source-audit.json")]
    mutate(*args)
    with pytest.raises(ValueError):
        Q.qcd_scope(*args)


@pytest.mark.parametrize("field", ["current_campaign_bridge_replayed_for_this_fixture", "full_physical_amplitude_qualification",
                                   "production_fix_justified_by_this_case_alone", "performance_qualification"])
def test_historical_default_helper_cannot_qualify_current_native_or_physics(manifest, field):
    observations = read(Q.DEFAULT + "observations.json")
    observations[field] = True
    with pytest.raises(ValueError):
        Q.default_scope(observations, manifest)


@pytest.mark.parametrize("mutate", [
    lambda a: a["higher_precision"].update(stability_certified=True),
    lambda a: a.update(fpe_flags_observed=True),
    lambda a: a.update(internal_retries_observed=True),
    lambda a: a["original_comparisons"]["candidate-9"].update(ordinary_target_passed=True),
    lambda a: a["original_comparisons"]["candidate-6"].update(target_missed_points=[]),
])
def test_current_target_diagnosis_retains_misses_and_unchecked_qp_limits(manifest, mutate):
    analysis = read(Q.TRIAGE + "analysis.json")
    mutate(analysis)
    with pytest.raises(ValueError):
        Q.triage_scope(analysis, manifest)


def test_index_and_file_hashes_are_both_required(tmp_path, monkeypatch):
    raw = b"retained bytes\n"
    (tmp_path / "row.txt").write_bytes(raw)
    index = {"schema_version": 1, "files": {"row.txt": {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}}
    path = tmp_path / "index.json"
    path.write_text(json.dumps(index))
    monkeypatch.setitem(Q.PINS, "index.json", hashlib.sha256(path.read_bytes()).hexdigest())
    assert Q.bind_package(REPORT.Evidence(tmp_path), "index.json") == index
    (tmp_path / "row.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash differs"):
        Q.bind_package(REPORT.Evidence(tmp_path), "index.json")
    path.write_text(json.dumps({**index, "schema_version": 2}))
    with pytest.raises(ValueError, match="hash differs"):
        Q.bind_package(REPORT.Evidence(tmp_path), "index.json")


@pytest.mark.parametrize("name", ["../outside", "/machine/path", "./alias", "a\\b"])
def test_portable_index_paths_cannot_escape_report_evidence(name):
    with pytest.raises(ValueError):
        Q.relative(name)


def test_absolute_metadata_references_are_not_opened(tmp_path):
    with pytest.raises(ValueError, match="not portable"):
        Q.bind_references(REPORT.Evidence(tmp_path), {"/recorded/native/library.so": "a" * 64})
