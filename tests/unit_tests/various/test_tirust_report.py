"""Report qualification and campaign provenance require retained evidence.

Plain unittest (the CI runners have no pytest): every former parametrization is a
subTest with its own fresh campaign fixture.
"""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


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


def make_campaign(tmp_path):
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


def mock_build(stack, campaign, output=None, during_compile=None):
    root = campaign['root']
    output = output or root / 'benchmark.pdf'
    stack.enter_context(mock.patch.object(REPORT, 'ROOT', root))
    stack.enter_context(mock.patch.object(sys, 'argv', ['build_report.py', '--campaign', campaign['id'],
                                                        '--output', str(output)]))
    stack.enter_context(mock.patch.object(REPORT.shutil, 'which', lambda name: '/mock/typst'))
    stack.enter_context(mock.patch.object(REPORT.subprocess, 'check_output',
                                          lambda *args, **kwargs: 'typst 0.15.1 (mock)\n'))
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

    stack.enter_context(mock.patch.object(REPORT.subprocess, 'run', compile_stub))
    return output, calls


class TIRustReportSummaryTest(unittest.TestCase):
    """Headline validation of the qualification summary."""

    def test_hundred_points_with_seven_comparison_repetitions(self):
        data = passing_summary()
        self.assertIs(REPORT.validate_summary(data), data)

    def test_passing_mixed_head_summary_requires_registered_single_source(self):
        data = passing_summary()
        data.update(candidate_revision=None, candidate_revisions=["doc-a", "doc-b"],
                    candidate_source_hashes=["same-exporter"], campaign_binding={"id": "named"})
        with self.assertRaises(ValueError):
            REPORT.validate_summary(data)
        REPORT.validate_summary(data, registered_identity={"files_sha256": "same-exporter"})
        data["candidate_source_hashes"].append("different-exporter")
        with self.assertRaises(ValueError):
            REPORT.validate_summary(data, registered_identity={"files_sha256": "same-exporter"})

    def test_unqualified_evidence_cannot_produce_passing_headline(self):
        changes = {
            "points": lambda row: row.update(points=99),
            "samples": lambda row: row["correctness"][0].update(samples=6),
            "backend": lambda row: row["correctness"].pop(),
            "timing": lambda row: row["performance"].update(upper_95=1.051),
            "regression": lambda row: row["existing_backend_regressions"].pop(),
            "qualified-failure": lambda row: row.update(authoritative_qualified_failures=["tirust-first"]),
            "validation-error": lambda row: row.update(validation_errors=["retained evidence hash differs"]),
        }
        for name, change in changes.items():
            with self.subTest(change=name):
                data = passing_summary()
                change(data["cases"][0])
                with self.assertRaises(ValueError):
                    REPORT.validate_summary(data)

    def test_failed_cases_cannot_produce_passing_headline(self):
        data = passing_summary()
        data["gate"]["failed_cases"] = [data["cases"][0]["id"]]
        with self.assertRaises(ValueError):
            REPORT.validate_summary(data)


class TIRustReportCampaignTest(unittest.TestCase):
    """Campaign provenance: bindings, retained failures and the build ledger."""

    def campaign(self):
        """A fresh campaign fixture in its own temporary directory."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        return make_campaign(Path(directory.name))

    def test_documentary_export_revisions_preserve_registered_numerical_anchor(self):
        campaign = self.campaign()
        campaign["summary"]["candidate_revision"] = None
        campaign["summary"]["candidate_revisions"] = ["documentation-head-a", "documentation-head-b"]
        write_json(campaign["report"] / "summary.json", campaign["summary"])
        summary, _, context, _ = prepare(campaign)
        self.assertIsNone(summary["candidate_revision"])
        self.assertEqual(context["observed_candidate_revisions"],
                         ["documentation-head-a", "documentation-head-b"])
        self.assertEqual(context["registered_exporter_revision"],
                         campaign["manifest"]["candidate"]["exporter_identity"]["revision"])

    def test_campaign_id_cannot_escape_or_alias_legacy(self):
        for identifier in ["legacy", "../current", "/current", "a/b", "a\\b", ""]:
            with self.subTest(identifier=identifier):
                campaign = self.campaign()
                with self.assertRaisesRegex(ValueError, "Invalid campaign ID"):
                    REPORT.prepare_report(campaign["root"], identifier)

    def test_campaign_retains_original_failed_timing_and_exact_source(self):
        campaign = self.campaign()
        summary, inputs, context, typst = prepare(campaign)
        self.assertEqual(summary, campaign["summary"])
        self.assertIs(summary["gate"]["passed"], False)
        self.assertEqual(context["campaign_id"], "current")
        self.assertEqual(context["numerical_identity_sha256"], "b" * 64)
        self.assertEqual(typst, {"summary": "campaigns/current/summary.json",
                                 "manifest": "campaigns/current/manifest.json",
                                 "context": "campaigns/current/report-context.json"})
        self.assertEqual(context["predecessors"], [{
            "predecessor_id": "legacy", "id": CASE["id"], "process": CASE["process"],
            "label": "tirust-first", "directory": "runs/qcd_n1_uux_z/tirust-first-attempt-3",
            "revision": OLD_SOURCE["revision"], "source_sha256": OLD_SOURCE["files_sha256"],
            **OLD_TIMING, "passed": False, "acceptance_eligible": True}])
        failure = campaign["manifest"]["predecessor"]["qualified_failures"][0]
        for name, checksum in failure["files"].items():
            self.assertEqual(inputs.files[name], {
                "sha256": checksum, "bytes": len((campaign["root"] / name).read_bytes())})
        self.assertTrue(set(DOCUMENT_INPUTS).issubset(inputs.files))

    def test_campaign_rejects_changed_binding_or_source(self):
        cases = [
            ("campaign_binding", {"id": "current", "manifest_sha256": "0" * 64}, "binding"),
            ("campaign_numerical_identity_sha256", "0" * 64, "binding"),
            ("candidate_revision", "another-source", "candidate source"),
            ("candidate_revisions", ["another-source"], "candidate source"),
            ("candidate_source_hashes", ["0" * 64], "candidate source"),
        ]
        for field, value, error in cases:
            with self.subTest(field=field):
                campaign = self.campaign()
                campaign["summary"][field] = value
                write_json(campaign["report"] / "summary.json", campaign["summary"])
                with self.assertRaisesRegex(ValueError, error):
                    prepare(campaign)

    def test_campaign_rejects_changed_policy_even_with_updated_file_hash(self):
        campaign = self.campaign()
        campaign["cases"]["minimum_pairs"] = 1
        checksum = write_json(campaign["report"] / "cases.json", campaign["cases"])
        campaign["manifest"]["shared_inputs"]["files"]["cases.json"] = checksum
        seal_campaign(campaign)
        with self.assertRaisesRegex(ValueError, "policy"):
            prepare(campaign)

    def test_campaign_rejects_process_different_from_configured_case(self):
        campaign = self.campaign()
        campaign["summary"]["cases"][0]["process"] = "u u~ > z g [virt=QCD]"
        seal_campaign(campaign)
        with self.assertRaisesRegex(ValueError, "process differs"):
            prepare(campaign)

    def test_campaign_rejects_unsafe_input_paths(self):
        for name in ["../summary.json", "/summary.json", "inputs/../../x", "inputs\\x"]:
            with self.subTest(name=name):
                campaign = self.campaign()
                campaign["manifest"]["shared_inputs"]["files"][name] = "0" * 64
                seal_campaign(campaign)
                with self.assertRaisesRegex(ValueError, "Unsafe report input path"):
                    prepare(campaign)

    def test_campaign_rejects_symlinked_inputs(self):
        for outside in (False, True):
            for directory in (False, True):
                with self.subTest(outside=outside, directory=directory):
                    campaign = self.campaign()
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
                    with self.assertRaisesRegex(ValueError, "symlink"):
                        prepare(campaign)

    def test_campaign_recurses_through_ancestors_and_pins_each_manifest(self):
        successor = successor_fixture(self.campaign())
        _, inputs, context, _ = prepare(successor)
        self.assertEqual(context["predecessors"][0]["revision"], OLD_SOURCE["revision"])
        self.assertTrue({"campaigns/current/manifest.json", "campaigns/current/summary.json",
                         "campaigns/current/inputs/predecessor/summary.json"}.issubset(inputs.files))

    def test_campaign_rejects_omitted_flattened_ancestor_failure(self):
        successor = successor_fixture(self.campaign())
        successor["summary"]["campaign_predecessor"]["qualified_failures"] = []
        seal_campaign(successor)
        with self.assertRaisesRegex(ValueError, "omits or changes retained predecessor failures"):
            prepare(successor)

    def test_content_named_discovery_retained_and_inherited_by_successor(self):
        campaign = self.campaign()
        discovery = add_discovery(campaign)
        _, inputs, context, _ = prepare(campaign)
        self.assertEqual({row["label"] for row in context["predecessors"]},
                         {"tirust-first", "backend-6"})
        self.assertEqual(inputs.files[discovery["path"]]["sha256"], discovery["sha256"])
        successor = successor_fixture(campaign)
        successor["manifest"]["predecessor"]["inherited_discoveries"] = [discovery]
        seal_campaign(successor)
        self.assertEqual(prepare(successor)[2]["predecessors"], context["predecessors"])

    def test_campaign_rejects_discovery_with_wrong_content_filename(self):
        campaign = self.campaign()
        discovery = add_discovery(campaign)
        path = campaign["root"] / discovery["path"]
        path.rename(path.with_name("0" * 64 + ".json"))
        with self.assertRaisesRegex(ValueError, "filename/hash differs"):
            prepare(campaign)

    def test_campaign_rejects_tampered_failure_evidence(self):
        for which in ["snapshot", "summary.json", "candidate-inputs.json"]:
            with self.subTest(which=which):
                campaign = self.campaign()
                predecessor = campaign["manifest"]["predecessor"]
                if which == "snapshot":
                    name = predecessor["summary_snapshot"]
                else:
                    name = predecessor["qualified_failures"][0]["directory"] + "/" + which
                path = campaign["root"] / name
                value = read_json(path)
                value["tampered"] = True
                write_json(path, value)
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    prepare(campaign)

    def test_campaign_checks_pinned_raw_evidence_beyond_required_pair(self):
        campaign = self.campaign()
        failure = campaign["manifest"]["predecessor"]["qualified_failures"][0]
        raw_name = failure["directory"] + "/raw.json"
        failure["files"][raw_name] = write_json(campaign["root"] / raw_name, {"seconds": 1.0})
        campaign["summary"]["campaign_predecessor"]["qualified_failures"] = [
            failure_reference("legacy", failure)]
        seal_campaign(campaign)
        self.assertIn(raw_name, prepare(campaign)[1].files)
        write_json(campaign["root"] / raw_name, {"seconds": 0.5})
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            prepare(campaign)

    def test_legacy_report_needs_no_campaign_manifest(self):
        root = self.campaign()["root"]
        self.assertFalse((root / "manifest.json").exists())
        summary, inputs, context, typst = REPORT.prepare_report(root)
        self.assertEqual(summary, read_json(root / "summary.json"))
        self.assertIsNone(context["campaign_id"])
        self.assertEqual(context["predecessors"], [])
        self.assertEqual(typst, {"summary": "summary.json", "manifest": "",
                                 "context": "report-context.json"})
        self.assertEqual(set(inputs.files), set(DOCUMENT_INPUTS) | {"summary.json", "cases.json"})

    def test_campaign_build_and_verify_only_bind_all_document_inputs(self):
        campaign = self.campaign()
        with ExitStack() as stack:
            output, calls = mock_build(stack, campaign)
            REPORT.main()
            self.assertEqual(len(calls), 1)
            ledger = read_json(campaign['root'] / 'report-build.json')
            context = read_json(campaign['report'] / 'report-context.json')
            self.assertEqual(context['predecessors'][0]['geometric_mean_ratio'],
                             OLD_TIMING['geometric_mean_ratio'])
            self.assertEqual(ledger['campaign'], 'current')
            self.assertEqual(ledger['typst_inputs']['manifest'], 'campaigns/current/manifest.json')
            self.assertIn('campaigns/current/report-context.json', ledger['inputs'])
            self.assertTrue(set(DOCUMENT_INPUTS).issubset(ledger['inputs']))
            self.assertEqual(ledger['pdf_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
            stack.enter_context(mock.patch.object(sys, 'argv', [*sys.argv, '--verify-only']))
            REPORT.main()
            self.assertEqual(len(calls), 1)  # Verification never invokes Typst or a numerical worker.

    def test_verify_only_rejects_changed_context_selection_or_pdf(self):
        for tamper in ['context', 'campaign', 'typst-input', 'pdf']:
            with self.subTest(tamper=tamper), ExitStack() as stack:
                campaign = self.campaign()
                output, _ = mock_build(stack, campaign)
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
                stack.enter_context(mock.patch.object(sys, 'argv', [*sys.argv, '--verify-only']))
                with self.assertRaisesRegex(ValueError, 'Report context|Report inputs|PDF differs'):
                    REPORT.main()

    def test_custom_pdf_keeps_the_canonical_build_ledger(self):
        campaign = self.campaign()
        ledger_path = campaign['root'] / 'report-build.json'
        ledger_path.write_text('canonical ledger must remain unchanged\n')
        with ExitStack() as stack:
            output, _ = mock_build(stack, campaign, campaign['root'] / 'preview.pdf')
            REPORT.main()
        self.assertTrue(output.exists())
        self.assertTrue(output.with_suffix('.build.json').exists())
        self.assertEqual(ledger_path.read_text(), 'canonical ledger must remain unchanged\n')

    def test_changed_input_during_compile_preserves_previous_pdf(self):
        campaign = self.campaign()
        root = campaign['root']
        output = root / 'benchmark.pdf'
        output.write_bytes(b'previous report')
        with ExitStack() as stack:
            mock_build(stack, campaign,
                       during_compile=lambda: (root / 'style.typ').write_text('changed'))
            with self.assertRaisesRegex(ValueError, 'changed during compilation: style.typ'):
                REPORT.main()
        self.assertEqual(output.read_bytes(), b'previous report')
        self.assertFalse(list(root.glob('.benchmark-*.pdf')))


if __name__ == "__main__":
    unittest.main()
