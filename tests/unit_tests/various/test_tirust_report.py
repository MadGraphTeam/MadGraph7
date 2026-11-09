"""Report qualification and campaign provenance require retained evidence."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "tirust_report", ROOT / "docs/benchmarkfolder/build_report.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def passing_summary():
    return {
        "baseline_revision": "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f",
        "candidate_revision": "candidate",
        "gate": {"passed": True, "complete": True,
                 "consistent_candidate_source": True},
        "cases": [{
            "id": str(i), "role": "main", "points": 100,
            "correctness": [{"backend": b, "passed": True, "samples": 7}
                            for b in (1, 6, 7, 8, 9)],
            "performance": {"passed": True, "complete": True,
                            "pairs": 7, "upper_95": 1.04},
            "existing_backend_regressions": [
                {"backend": b, "complete": True, "passed": True}
                for b in (1, 6, 7, 8)],
        } for i in range(8)],
    }


def test_hundred_points_with_seven_comparison_repetitions():
    data = passing_summary()
    assert REPORT.validate_summary(data) is data


@pytest.mark.parametrize("change", [
    lambda row: row.update(points=99),
    lambda row: row["correctness"][0].update(samples=6),
    lambda row: row["correctness"].pop(),
    lambda row: row["performance"].update(upper_95=1.051),
    lambda row: row["existing_backend_regressions"].pop(),
    lambda row: row.update(authoritative_qualified_failures=["tirust-first"]),
    lambda row: row.update(validation_errors=["retained evidence hash differs"]),
])
def test_unqualified_evidence_cannot_produce_passing_headline(change):
    data = passing_summary()
    change(data["cases"][0])
    with pytest.raises(ValueError):
        REPORT.validate_summary(data)


def test_failed_cases_cannot_produce_passing_headline():
    data = passing_summary()
    data["gate"]["failed_cases"] = [data["cases"][0]["id"]]
    with pytest.raises(ValueError):
        REPORT.validate_summary(data)


BASELINE = "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f"
OLD_SOURCE = {
    "revision": "c91cb85f3aa942c90299b3518bb1f195f55334a5",
    "files_sha256": "74116e4bb2ba7abdb24a3dbcdb51cfae6547f05c3e276d3e17f5d5dab2b8d7a5",
}
OLD_TIMING = {"geometric_mean_ratio": 1.1210500947672746,
              "lower_95": 1.1034925280600616, "upper_95": 1.138887018281026}
CASE = {"id": "qcd_n1_uux_z", "process": "u u~ > z [virt=QCD]",
        "model": "loop_sm", "nfinal": 1, "role": "main"}
DOCUMENT_INPUTS = (
    "build_report.py", "benchmark.typ", "style.typ", "poles-timing.typ",
    "implementation-start.json", "style-provenance.json",
    "experiments/numerical/completed-pilots-index.json",
    "experiments/poles-timing/historical-release-paired/analysis-summary.json",
)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def failure_fixture(root, report=".", directory="tirust-first-attempt-3",
                    label="tirust-first", identity=None):
    directory = "runs/" + CASE["id"] + "/" + directory
    files = {}
    payloads = {
        "summary.json": {"label": label, "acceptance_eligible": True,
                         "passed": False, "statistics": dict(OLD_TIMING)},
        "candidate-inputs.json": {"generation": {
            "source_identity": dict(identity or OLD_SOURCE)}},
    }
    for name, value in payloads.items():
        relative = directory + "/" + name
        files[relative] = write_json(root / report / relative, value)
    return {"case": CASE["id"], "label": label, "directory": directory,
            "files": files}


def failure_reference(identifier, failure):
    return {"predecessor_id": identifier, "case": failure["case"],
            "label": failure["label"], "directory": failure["directory"],
            "evidence_sha256": digest_json(failure["files"])}


def seal_campaign(fixture):
    """Update manifest binding after a deliberate, otherwise valid fixture edit."""
    report = fixture["root"] / "campaigns" / fixture["id"]
    checksum = write_json(report / "manifest.json", fixture["manifest"])
    fixture["summary"]["campaign_binding"] = {
        "id": fixture["id"], "manifest_sha256": checksum}
    write_json(report / "summary.json", fixture["summary"])


def campaign_fixture(root, identifier, predecessor, references):
    report = root / "campaigns" / identifier
    cases = {"baseline_revision": BASELINE, "cases": [deepcopy(CASE)]}
    cases_sha = write_json(report / "cases.json", cases)
    identity = {"revision": "new-candidate-" + identifier, "files_sha256": "a" * 64}
    manifest = {
        "schema": "madloop-tirust-campaign-v1", "id": identifier,
        "baseline": {"revision": BASELINE},
        "candidate": {"exporter_identity": identity,
                      "numerical_identity_sha256": "b" * 64},
        "shared_inputs": {"policy_sha256": digest_json(cases),
                          "files": {"cases.json": cases_sha}},
        "predecessor": deepcopy(predecessor),
    }
    summary = {
        "baseline_revision": BASELINE, "candidate_revision": identity["revision"],
        "candidate_revisions": [identity["revision"]],
        "candidate_source_hashes": [identity["files_sha256"]],
        "campaign_numerical_identity_sha256": "b" * 64,
        "cases": deepcopy(cases["cases"]), "gate": {"passed": False},
        "campaign_predecessor": {
            **{key: predecessor[key] for key in ("id", "manifest_sha256", "summary_sha256")},
            "qualified_failures": deepcopy(references)},
    }
    result = {"root": root, "id": identifier, "report": report, "cases": cases,
              "manifest": manifest, "summary": summary}
    seal_campaign(result)
    return result


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path / "report"
    for name in DOCUMENT_INPUTS:
        write_json(root / name, {})
    cases = {"baseline_revision": BASELINE, "cases": [deepcopy(CASE)]}
    write_json(root / "cases.json", cases)
    legacy = {"baseline_revision": BASELINE, "cases": deepcopy(cases["cases"]),
              "candidate_revision": "aggregate-headline-is-not-failure-source",
              "gate": {"passed": False}}
    write_json(root / "summary.json", legacy)
    snapshot = "campaigns/current/inputs/predecessor/summary.json"
    predecessor = {
        "id": "legacy", "report": ".", "manifest_sha256": None,
        "summary_sha256": write_json(root / snapshot, legacy),
        "summary_snapshot": snapshot, "policy_sha256": digest_json(cases),
        "numerical_identity_sha256": None, "ancestor": None,
        "qualified_failures": [failure_fixture(root)], "inherited_discoveries": [],
    }
    failure = predecessor["qualified_failures"][0]
    return campaign_fixture(root, "current", predecessor,
                            [failure_reference("legacy", failure)])


def prepare(fixture):
    return REPORT.prepare_report(fixture["root"], fixture["id"])


@pytest.mark.parametrize("identifier", ["legacy", "../current", "/current", "a/b", "a\\b", ""])
def test_campaign_id_cannot_escape_or_alias_legacy(campaign, identifier):
    with pytest.raises(ValueError, match="Invalid campaign ID"):
        REPORT.prepare_report(campaign["root"], identifier)


def test_campaign_retains_original_failed_timing_and_exact_source(campaign):
    summary, inputs, context, typst = prepare(campaign)
    assert summary == campaign["summary"]
    assert summary["gate"]["passed"] is False
    assert context["campaign_id"] == "current"
    assert context["numerical_identity_sha256"] == "b" * 64
    assert typst == {"summary": "campaigns/current/summary.json",
                     "manifest": "campaigns/current/manifest.json",
                     "context": "campaigns/current/report-context.json"}
    assert context["predecessors"] == [{
        "predecessor_id": "legacy", "id": CASE["id"], "process": CASE["process"],
        "label": "tirust-first", "directory": "runs/qcd_n1_uux_z/tirust-first-attempt-3",
        "revision": OLD_SOURCE["revision"], "source_sha256": OLD_SOURCE["files_sha256"],
        **OLD_TIMING, "passed": False, "acceptance_eligible": True}]
    failure = campaign["manifest"]["predecessor"]["qualified_failures"][0]
    for name, checksum in failure["files"].items():
        assert inputs.files[name] == {
            "sha256": checksum, "bytes": len((campaign["root"] / name).read_bytes())}
    assert set(DOCUMENT_INPUTS).issubset(inputs.files)


@pytest.mark.parametrize("field,value,error", [
    ("campaign_binding", {"id": "current", "manifest_sha256": "0" * 64}, "binding"),
    ("campaign_numerical_identity_sha256", "0" * 64, "binding"),
    ("candidate_revision", "another-source", "candidate source"),
    ("candidate_revisions", ["another-source"], "candidate source"),
    ("candidate_source_hashes", ["0" * 64], "candidate source"),
])
def test_campaign_rejects_changed_binding_or_source(campaign, field, value, error):
    campaign["summary"][field] = value
    write_json(campaign["report"] / "summary.json", campaign["summary"])
    with pytest.raises(ValueError, match=error):
        prepare(campaign)


def test_campaign_rejects_changed_policy_even_with_updated_file_hash(campaign):
    campaign["cases"]["minimum_pairs"] = 1
    checksum = write_json(campaign["report"] / "cases.json", campaign["cases"])
    campaign["manifest"]["shared_inputs"]["files"]["cases.json"] = checksum
    seal_campaign(campaign)
    with pytest.raises(ValueError, match="policy"):
        prepare(campaign)


def test_campaign_rejects_process_different_from_configured_case(campaign):
    campaign["summary"]["cases"][0]["process"] = "u u~ > z g [virt=QCD]"
    seal_campaign(campaign)
    with pytest.raises(ValueError, match="process differs"):
        prepare(campaign)


@pytest.mark.parametrize("name", ["../summary.json", "/summary.json", "inputs/../../x", "inputs\\x"])
def test_campaign_rejects_unsafe_input_paths(campaign, name):
    campaign["manifest"]["shared_inputs"]["files"][name] = "0" * 64
    seal_campaign(campaign)
    with pytest.raises(ValueError, match="Unsafe report input path"):
        prepare(campaign)


@pytest.mark.parametrize("outside", [False, True], ids=["in-root", "out-root"])
@pytest.mark.parametrize("directory", [False, True], ids=["file", "parent-directory"])
def test_campaign_rejects_symlinked_inputs(campaign, outside, directory):
    root, report = campaign["root"], campaign["report"]
    target = (root.parent if outside else root) / "actual"
    target.mkdir()
    checksum = write_json(target / "input.json", {"value": 1})
    alias = report / "alias"
    alias.symlink_to(target if directory else target / "input.json",
                     target_is_directory=directory)
    name = "alias/input.json" if directory else "alias"
    campaign["manifest"]["shared_inputs"]["files"][name] = checksum
    seal_campaign(campaign)
    with pytest.raises(ValueError, match="symlink"):
        prepare(campaign)


def successor_fixture(campaign):
    predecessor = {
        "id": campaign["id"], "report": "campaigns/" + campaign["id"],
        "manifest_sha256": campaign["summary"]["campaign_binding"]["manifest_sha256"],
        "summary_sha256": hashlib.sha256((campaign["report"] / "summary.json").read_bytes()).hexdigest(),
        "policy_sha256": campaign["manifest"]["shared_inputs"]["policy_sha256"],
        "numerical_identity_sha256": "b" * 64,
        "ancestor": deepcopy(campaign["manifest"]["predecessor"]),
        "qualified_failures": [], "inherited_discoveries": [],
    }
    references = campaign["summary"]["campaign_predecessor"]["qualified_failures"]
    return campaign_fixture(campaign["root"], "successor", predecessor, references)


def test_campaign_recurses_through_ancestors_and_pins_each_manifest(campaign):
    successor = successor_fixture(campaign)
    _, inputs, context, _ = prepare(successor)
    assert context["predecessors"][0]["revision"] == OLD_SOURCE["revision"]
    assert {"campaigns/current/manifest.json", "campaigns/current/summary.json",
            "campaigns/current/inputs/predecessor/summary.json"}.issubset(inputs.files)


def test_campaign_rejects_omitted_flattened_ancestor_failure(campaign):
    successor = successor_fixture(campaign)
    successor["summary"]["campaign_predecessor"]["qualified_failures"] = []
    seal_campaign(successor)
    with pytest.raises(ValueError, match="omits or changes retained predecessor failures"):
        prepare(successor)


def add_discovery(campaign):
    failure = failure_fixture(campaign["root"], directory="backend-6", label="backend-6")
    predecessor = campaign["manifest"]["predecessor"]
    record = {"predecessor_id": "legacy", "report": ".",
              "policy_sha256": predecessor["policy_sha256"], "failure": failure}
    path = campaign["report"] / "inputs/predecessor/discoveries" / (digest_json(record) + ".json")
    checksum = write_json(path, record)
    campaign["summary"]["campaign_predecessor"]["qualified_failures"].append(
        failure_reference("legacy", failure))
    seal_campaign(campaign)
    return {"path": path.relative_to(campaign["root"]).as_posix(),
            "sha256": checksum, "record": record}


def test_content_named_discovery_retained_and_inherited_by_successor(campaign):
    discovery = add_discovery(campaign)
    _, inputs, context, _ = prepare(campaign)
    assert {row["label"] for row in context["predecessors"]} == {"tirust-first", "backend-6"}
    assert inputs.files[discovery["path"]]["sha256"] == discovery["sha256"]
    successor = successor_fixture(campaign)
    successor["manifest"]["predecessor"]["inherited_discoveries"] = [discovery]
    seal_campaign(successor)
    assert prepare(successor)[2]["predecessors"] == context["predecessors"]


def test_campaign_rejects_discovery_with_wrong_content_filename(campaign):
    discovery = add_discovery(campaign)
    path = campaign["root"] / discovery["path"]
    path.rename(path.with_name("0" * 64 + ".json"))
    with pytest.raises(ValueError, match="filename/hash differs"):
        prepare(campaign)


@pytest.mark.parametrize("which", ["snapshot", "summary.json", "candidate-inputs.json"])
def test_campaign_rejects_tampered_failure_evidence(campaign, which):
    predecessor = campaign["manifest"]["predecessor"]
    if which == "snapshot":
        name = predecessor["summary_snapshot"]
    else:
        name = predecessor["qualified_failures"][0]["directory"] + "/" + which
    path = campaign["root"] / name
    value = read_json(path)
    value["tampered"] = True
    write_json(path, value)
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare(campaign)


def test_campaign_checks_pinned_raw_evidence_beyond_required_pair(campaign):
    failure = campaign["manifest"]["predecessor"]["qualified_failures"][0]
    raw_name = failure["directory"] + "/raw.json"
    failure["files"][raw_name] = write_json(campaign["root"] / raw_name, {"seconds": 1.0})
    campaign["summary"]["campaign_predecessor"]["qualified_failures"] = [
        failure_reference("legacy", failure)]
    seal_campaign(campaign)
    assert raw_name in prepare(campaign)[1].files
    write_json(campaign["root"] / raw_name, {"seconds": 0.5})
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare(campaign)


def test_legacy_report_needs_no_campaign_manifest(campaign):
    root = campaign["root"]
    assert not (root / "manifest.json").exists()
    summary, inputs, context, typst = REPORT.prepare_report(root)
    assert summary == read_json(root / "summary.json")
    assert context["campaign_id"] is None
    assert context["predecessors"] == []
    assert typst == {"summary": "summary.json", "manifest": "",
                     "context": "report-context.json"}
    assert set(inputs.files) == set(DOCUMENT_INPUTS) | {"summary.json", "cases.json"}


def mock_build(monkeypatch, campaign, output=None, during_compile=None):
    root = campaign['root']
    output = output or root / 'benchmark.pdf'
    monkeypatch.setattr(REPORT, 'ROOT', root)
    monkeypatch.setattr(sys, 'argv', ['build_report.py', '--campaign', campaign['id'],
                                   '--output', str(output)])
    monkeypatch.setattr(REPORT.shutil, 'which', lambda name: '/mock/typst')
    monkeypatch.setattr(REPORT.subprocess, 'check_output',
                        lambda *args, **kwargs: 'typst 0.15.1 (mock)\n')
    calls = []

    def compile_stub(command, **kwargs):
        calls.append(command)
        assert command[command.index('--root') + 1] == str(root)
        assert 'summary=campaigns/current/summary.json' in command
        assert 'manifest=campaigns/current/manifest.json' in command
        assert 'context=campaigns/current/report-context.json' in command
        if during_compile:
            during_compile()
        Path(command[-1]).write_bytes(b'%PDF mocked compilation\n')

    monkeypatch.setattr(REPORT.subprocess, 'run', compile_stub)
    return output, calls


def test_campaign_build_and_verify_only_bind_all_document_inputs(campaign, monkeypatch):
    output, calls = mock_build(monkeypatch, campaign)
    REPORT.main()
    assert len(calls) == 1
    ledger = read_json(campaign['root'] / 'report-build.json')
    context = read_json(campaign['report'] / 'report-context.json')
    assert context['predecessors'][0]['geometric_mean_ratio'] == OLD_TIMING['geometric_mean_ratio']
    assert ledger['campaign'] == 'current'
    assert ledger['typst_inputs']['manifest'] == 'campaigns/current/manifest.json'
    assert 'campaigns/current/report-context.json' in ledger['inputs']
    assert set(DOCUMENT_INPUTS).issubset(ledger['inputs'])
    assert ledger['pdf_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, 'argv', [*sys.argv, '--verify-only'])
    REPORT.main()
    assert len(calls) == 1  # Verification never invokes Typst or a numerical worker.


@pytest.mark.parametrize('tamper', ['context', 'campaign', 'typst-input', 'pdf'])
def test_verify_only_rejects_changed_context_selection_or_pdf(campaign, monkeypatch, tamper):
    output, _ = mock_build(monkeypatch, campaign)
    REPORT.main()
    ledger_path = campaign['root'] / 'report-build.json'
    ledger = read_json(ledger_path)
    if tamper == 'context':
        context = read_json(campaign['report'] / 'report-context.json')
        context['predecessors'] = []
        write_json(campaign['report'] / 'report-context.json', context)
    elif tamper == 'campaign':
        ledger['campaign'] = 'another-campaign'
        write_json(ledger_path, ledger)
    elif tamper == 'typst-input':
        ledger['typst_inputs']['manifest'] = ''
        write_json(ledger_path, ledger)
    else:
        output.write_bytes(b'%PDF other document\n')
    monkeypatch.setattr(sys, 'argv', [*sys.argv, '--verify-only'])
    with pytest.raises(ValueError, match='Report context|Report inputs|PDF differs'):
        REPORT.main()


def test_custom_pdf_keeps_the_canonical_build_ledger(campaign, monkeypatch):
    ledger_path = campaign['root'] / 'report-build.json'
    ledger_path.write_text('canonical ledger must remain unchanged\n')
    output, _ = mock_build(monkeypatch, campaign, campaign['root'] / 'preview.pdf')
    REPORT.main()
    assert output.exists()
    assert output.with_suffix('.build.json').exists()
    assert ledger_path.read_text() == 'canonical ledger must remain unchanged\n'


def test_changed_input_during_compile_preserves_previous_pdf(campaign, monkeypatch):
    root = campaign['root']
    output = root / 'benchmark.pdf'
    output.write_bytes(b'previous report')
    mock_build(monkeypatch, campaign, during_compile=lambda: (root / 'style.typ').write_text('changed'))
    with pytest.raises(ValueError, match='changed during compilation: style.typ'):
        REPORT.main()
    assert output.read_bytes() == b'previous report'
    assert not list(root.glob('.benchmark-*.pdf'))
