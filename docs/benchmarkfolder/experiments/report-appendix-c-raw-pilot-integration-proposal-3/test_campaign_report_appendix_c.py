"""Synthetic Appendix C capsules only; no real pilot, native or Typst reads.

Tiny root-only verifiers exercise report admission and relocation without
executing an amplitude or a plotter. Every image and observation is invented.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class MemoryEvidence:
    """In-memory physical files, with the same hash boundary as Evidence.add."""
    def __init__(self, entries):
        self.entries, self.reads, self.files = dict(entries), [], {}

    def add(self, name, expected=None):
        self.reads.append(name)
        raw = self.entries[name]
        if expected is not None and sha(raw) != expected:
            raise ValueError("Synthetic evidence hash differs: " + name)
        self.files[name] = {"sha256": sha(raw), "bytes": len(raw)}
        return raw

    def json(self, name, expected=None):
        return json.loads(self.add(name, expected))


class NoReads:
    def add(self, *args, **kwargs):
        raise AssertionError("An unadmitted or absent capsule was read")

    json = add


def verifier_source(figures=False, fail=False):
    """The invented proof checks roots and inventory; it never executes plotter.py."""
    source = '''import hashlib
import json
from pathlib import Path
import sys

def checked(root):
    root = Path(root)
    assert sys.dont_write_bytecode is True
    assert not list(root.rglob("__pycache__"))
    raw_index = (root / "index.json").read_bytes()
    index = json.loads(raw_index)
    assert {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} == {"index.json", *index["files"]}
    for name, identity in index["files"].items():
        raw = (root / name).read_bytes()
        assert len(raw) == identity["bytes"]
        assert hashlib.sha256(raw).hexdigest() == identity["sha256"]
    return {"verified": True, "index_sha256": hashlib.sha256(raw_index).hexdigest()}
'''
    if figures:
        source += '''
def verify(figure_root, *, data_root, plotter_root):
    assert len({Path(figure_root), Path(data_root), Path(plotter_root)}) == 3
    result = checked(figure_root)
    data, plotter = checked(data_root), checked(plotter_root)
    inputs = json.loads((Path(figure_root) / "proof-input.json").read_bytes())
    assert inputs["data_index_sha256"] == data["index_sha256"]
    assert inputs["plotter_index_sha256"] == plotter["index_sha256"]
    result.update(dependencies={"data_index_sha256": inputs["data_index_sha256"],
                               "plotter_index_sha256": inputs["plotter_index_sha256"]},
                  cases=inputs["cases"], artifacts=inputs["artifacts"])
    return result
'''
    else:
        source += '''
def verify(root=None):
    result = checked(root)
    result["summary"] = json.loads((Path(root) / "summary.json").read_bytes())
    return result
'''
    if fail:
        source = source.replace('    result = checked(root)', '    raise RuntimeError("synthetic root verifier failure")')
    return source.encode()


class Capsule:
    def __init__(self, name, summary, *, figures=False, extra=None, verifier=None):
        self.role = name
        self.prefix = "experiments/synthetic-appendix-c/" + name
        self.summary = summary
        self.verifier_name = "figures.py" if figures else "verify.py"
        self.payloads = {"proof-input.json" if figures else "summary.json": encoded(summary),
                         self.verifier_name: verifier or verifier_source(figures),
                         "unclaimed-retained.txt": b"invented retained payload"}
        self.payloads.update(extra or {})
        self.reseal()

    def reseal(self):
        schemas = {"data": "madloop-raw-stability-capsule-v1", "figures": "madloop-raw-stability-figures-v1",
                   "plotter": "madloop-stability-plotter-source-v1"}
        self.index = {"schema": schemas[self.role], "files": {name: {"sha256": sha(raw), "bytes": len(raw)}
            for name, raw in self.payloads.items()}}
        self.refresh_index()

    def refresh_index(self):
        raw = encoded(self.index)
        self.pin = {"index": self.prefix + "/index.json", "sha256": sha(raw)}
        self.entries = {self.prefix + "/" + name: value for name, value in self.payloads.items()}
        self.entries[self.pin["index"]] = raw


CASES = ("qcd_n2_gg_ttx", "qcd_n3_uux_zgg", "ew_n2_epem_mupmum", "qcd_n4_uux_zzgg")
BACKENDS = (1, 6, 7, 8, 9)
FAILURES = ("coverage_missing", "active_group_failure", "nonfinite_coefficients",
            "nonfinite_results", "floating_point_failure_latch")


def data_summary():
    rows = []
    for case in CASES:
        for backend in BACKENDS:
            rows.append({"case": case, "backend": backend, "result_state": "complete",
                "native_started": True, "error": None, "diagnostic_error": None, "readonly_guard_error": None,
                "intended_points": 100, "available_points": 100, "unavailable_points": 0,
                "zero_points": 0, "infinite_points": 0, "return_code_counts": {"0": 300},
                "bootstrap_return_code_counts": {"0": 1}, "primary_outcome_counts": {"available": 300},
                "group_failure_call_counts": dict.fromkeys(FAILURES, 0),
                "tail_at_1e3": {"n": 100, "known_exceedances": 2, "unavailable": 0,
                    "lower": .02, "upper": .02, "fraction": .02}})
    return {"schema": "madloop-raw-stability-summary-v1", "rows": rows}


def missing_row(row, *, started=False):
    row.update(result_state="unavailable", native_started=started,
        error={"type": "SyntheticFailure", "message": "invented unavailable evaluation"},
        diagnostic_error="invented diagnostic failure", readonly_guard_error="invented readonly refusal",
        available_points=0, unavailable_points=100, zero_points=0, infinite_points=0,
        return_code_counts={"-1": 300} if started else {}, bootstrap_return_code_counts={},
        primary_outcome_counts={"unavailable": 300} if started else {"unexecuted": 300},
        group_failure_call_counts={**dict.fromkeys(FAILURES, 0), "coverage_missing": 300 if started else 0},
        tail_at_1e3={"n": 100, "known_exceedances": 0, "unavailable": 100,
            "lower": 0., "upper": 1., "fraction": None})


class Corpus:
    def __init__(self, summary=None, *, data_verifier=None, figure_verifier=None):
        self.data = Capsule("data", deepcopy(summary) if summary is not None else data_summary(),
            extra={"results/retained-statuses.json": encoded({"all_result_statuses": "invented sidecar"})},
            verifier=data_verifier)
        self.plotter = Capsule("plotter", {"scope": "invented plotter identity"},
            extra={"plotter.py": b'raise AssertionError("The report must never execute the plotter")\n'})
        figure_summary = {"scope": "invented figure proof", "data_index_sha256": self.data.pin["sha256"],
            "plotter_index_sha256": self.plotter.pin["sha256"], "cases": list(CASES), "artifacts": 12}
        extra = {}
        for case in CASES:
            extra["figures/" + case + ".png"] = b"\x89PNG\r\n\x1a\nsynthetic figure " + case.encode()
            extra["figures/" + case + ".pdf"] = b"%PDF-1.4\n% invented export " + case.encode()
            extra["figures/" + case + ".json"] = encoded({"case": case, "unplotted_failures": ["invented"]})
        self.figures = Capsule("figures", figure_summary, figures=True, extra=extra, verifier=figure_verifier)
        self.data_verifier_sha256 = sha(self.data.payloads["verify.py"])
        self.figure_verifier_sha256 = sha(self.figures.payloads["figures.py"])

    @property
    def contract(self):
        return {"data": self.data.pin, "figures": self.figures.pin, "plotter": self.plotter.pin,
            "data_verifier_sha256": self.data_verifier_sha256,
            "figure_verifier_sha256": self.figure_verifier_sha256}

    @property
    def entries(self):
        return {**self.data.entries, **self.figures.entries, **self.plotter.entries}

    @contextmanager
    def admitted(self, helper, evidence=None):
        evidence = evidence or MemoryEvidence(self.entries)
        pin = self.figures.pin
        with patch.object(helper, "ADMITTED", {(pin["index"], pin["sha256"]): self.contract}), \
                patch("subprocess.Popen", side_effect=AssertionError("Native processes are forbidden")), \
                patch("ctypes.CDLL", side_effect=AssertionError("Native loading is forbidden")):
            yield evidence, pin


class SyntheticAppendixCapsules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.C = module("campaign_report_appendix_c")

    def normalize(self, summary):
        return self.C.normalize(summary, {case: {"figure_path": "synthetic/" + case + ".png"} for case in CASES})

    def test_unadmitted_or_nonexact_selection_rejects_before_any_reads(self):
        corpus = Corpus()
        for pin in (corpus.figures.pin, {**corpus.figures.pin, "extra": True},
                    {"index": "../active/evidence-index.json", "sha256": "a" * 64}):
            with self.subTest(pin=pin), patch.object(self.C, "ADMITTED", {}):
                with self.assertRaises(ValueError): self.C.collect(NoReads(), pin)

    def test_four_figures_and_twenty_full_result_records_bind_all_three_closures(self):
        summary = data_summary();missing_row(summary["rows"][0], started=True)
        corpus = Corpus(summary)
        before = deepcopy(summary)
        with corpus.admitted(self.C) as (evidence, pin):
            result = self.C.collect(evidence, pin)
        self.assertEqual([case["id"] for case in result["cases"]], list(CASES))
        self.assertEqual([(lane["case"], lane["backend"]) for case in result["cases"] for lane in case["lanes"]],
            [(case, backend) for case in CASES for backend in BACKENDS])
        for case in result["cases"]:
            self.assertEqual(case["process"], case["id"])
            self.assertEqual(len(case["lanes"]), 5)
            for key, suffix in (("figure_path", ".png"), ("export_pdf_path", ".pdf"), ("figure_data_path", ".json")):
                name = corpus.figures.prefix + "/figures/" + case["id"] + suffix
                self.assertEqual(case[key], name)
                self.assertEqual(evidence.files[name], {"sha256": sha(corpus.entries[name]), "bytes": len(corpus.entries[name])})
        self.assertEqual(set(evidence.files), set(corpus.entries))
        self.assertIn(corpus.data.prefix + "/results/retained-statuses.json", evidence.files)
        self.assertEqual(result["retained_data_summary"], before)
        self.assertNotIn("summary", result["retained_figure_verification"])
        self.assertEqual(result["retained_figure_verification"]["cases"], list(CASES))
        self.assertEqual(result["retained_figure_verification"]["artifacts"], 12)
        self.assertEqual(result["retained_figure_verification"]["dependencies"],
            {"data_index_sha256": corpus.data.pin["sha256"], "plotter_index_sha256": corpus.plotter.pin["sha256"]})
        first = result["cases"][0]["lanes"][0]
        for key in before["rows"][0]: self.assertEqual(first[key], before["rows"][0][key])
        self.assertEqual(json.loads(first["bootstrap_return_code_counts_text"]), first["bootstrap_return_code_counts"])
        for key in ("performance_eligible", "independent_truth_qualified", "observed_actual_backend",
                    "observed_per_result_internal_precision", "main_campaign_modified"):
            self.assertIs(result[key], False)
        self.assertEqual(summary, before)

    def test_missing_duplicate_reordered_case_backend_or_status_evidence_rejects(self):
        for defect in ("summary_schema", "missing", "duplicate", "case", "backend", "order", "state", "error", "histogram", "group_key", "boolean_count", "primary299", "primary301"):
            with self.subTest(defect=defect):
                summary = data_summary();row = summary["rows"][0]
                if defect == "summary_schema": summary["schema"] = "invented-wrong-summary"
                elif defect == "missing": summary["rows"].pop()
                elif defect == "duplicate": summary["rows"][1] = deepcopy(row)
                elif defect == "case": row["case"] = "invented_unrequested_case"
                elif defect == "backend": row["backend"] = 2
                elif defect == "order": summary["rows"].reverse()
                elif defect == "state": del row["result_state"]
                elif defect == "error": del row["readonly_guard_error"]
                elif defect == "histogram": del row["return_code_counts"]
                elif defect == "group_key": del row["group_failure_call_counts"]["coverage_missing"]
                elif defect == "primary299": row["primary_outcome_counts"]["available"] = 299
                elif defect == "primary301": row["primary_outcome_counts"]["available"] = 301
                else: row["primary_outcome_counts"]["available"] = True
                with self.assertRaises(ValueError): self.normalize(summary)

    def test_point_availability_zero_infinite_and_unstarted_bounds_reject(self):
        for defect in ("sum", "negative", "boolean", "intended", "zero_inf", "infinite_tail", "zero_tail", "unstarted", "unstarted_available", "unstarted_unavailable"):
            with self.subTest(defect=defect):
                summary = data_summary();row = summary["rows"][0]
                if defect == "sum": row["available_points"] = 99
                elif defect == "negative": row["zero_points"] = -1
                elif defect == "boolean": row["available_points"] = True
                elif defect == "intended": row["intended_points"] = 99
                elif defect == "zero_inf": row.update(zero_points=51, infinite_points=50)
                elif defect == "infinite_tail": row["infinite_points"] = 10
                elif defect == "zero_tail": row["zero_points"] = 100
                elif defect in {"unstarted_available", "unstarted_unavailable"}:
                    missing_row(row)
                    row["primary_outcome_counts"] = {"available" if defect == "unstarted_available" else "unavailable": 300}
                else: row["native_started"] = False
                with self.assertRaises(ValueError): self.normalize(summary)

    def test_all_zero_and_all_infinite_tails_are_exact_numerical_edges(self):
        for kind, expected in (("zero", 0.), ("infinite", 1.)):
            with self.subTest(kind=kind):
                summary = data_summary();row = summary["rows"][0]
                row[kind + "_points"] = 100
                row["tail_at_1e3"].update(known_exceedances=int(expected * 100), lower=expected, upper=expected, fraction=expected)
                actual = self.normalize(summary)[0]["lanes"][0]
                self.assertEqual(actual[kind + "_points"], 100)
                self.assertEqual(actual["tail_at_1e3"]["fraction"], expected)
                row["tail_at_1e3"]["fraction"] = bool(expected)
                with self.assertRaises(ValueError): self.normalize(summary)

    def test_missing_tail_stays_unknown_and_partial_tail_has_exact_bounds(self):
        summary = data_summary();missing_row(summary["rows"][0])
        partial = summary["rows"][1]
        partial.update(available_points=75, unavailable_points=25)
        partial["tail_at_1e3"].update(known_exceedances=3, unavailable=25, lower=.03, upper=.28, fraction=None)
        cases = self.normalize(summary)
        self.assertEqual(cases[0]["lanes"][0]["tail_at_1e3"],
            {"n": 100, "known_exceedances": 0, "unavailable": 100, "lower": 0., "upper": 1., "fraction": None})
        self.assertEqual(cases[0]["lanes"][0]["primary_outcome_counts"], {"unexecuted": 300})
        self.assertEqual(cases[0]["lanes"][0]["return_code_counts"], {})
        self.assertIsNone(cases[0]["lanes"][1]["tail_at_1e3"]["fraction"])
        self.assertEqual(cases[0]["lanes"][1]["tail_at_1e3"]["lower"], .03)
        self.assertEqual(cases[0]["lanes"][1]["tail_at_1e3"]["upper"], .28)
        for defect in ("n", "known", "unavailable", "lower", "upper", "fraction"):
            with self.subTest(defect=defect):
                bad = deepcopy(summary);tail = bad["rows"][0]["tail_at_1e3"]
                tail[{"n": "n", "known": "known_exceedances", "unavailable": "unavailable",
                      "lower": "lower", "upper": "upper", "fraction": "fraction"}[defect]] = {
                    "n": 99, "known": 1, "unavailable": 99, "lower": .1, "upper": .9, "fraction": 0.}[defect]
                with self.assertRaises(ValueError): self.normalize(bad)

    def test_png_status_index_hash_size_and_verifier_drift_are_hard_errors(self):
        for defect in ("png", "status", "sidecar", "index", "size", "verifier", "data_schema", "figures_schema", "plotter_schema"):
            with self.subTest(defect=defect):
                corpus = Corpus()
                if defect == "size":
                    corpus.figures.index["files"]["figures/" + CASES[0] + ".png"]["bytes"] += 1
                    corpus.figures.refresh_index()
                elif defect == "verifier":
                    corpus.data.payloads["verify.py"] += b"\n# synthetic source drift\n"
                    corpus.data.reseal()
                elif defect.endswith("_schema"):
                    packet = getattr(corpus, defect.removesuffix("_schema"))
                    packet.index["schema"] = "invented-wrong-role-schema"
                    packet.refresh_index()
                evidence = MemoryEvidence(corpus.entries)
                if defect in {"png", "status", "sidecar", "index"}:
                    name = {"png": corpus.figures.prefix + "/figures/" + CASES[0] + ".png",
                        "status": corpus.data.prefix + "/summary.json",
                        "sidecar": corpus.data.prefix + "/results/retained-statuses.json",
                        "index": corpus.data.pin["index"]}[defect]
                    evidence.entries[name] += b"synthetic byte drift"
                with corpus.admitted(self.C, evidence) as (evidence, pin):
                    with self.assertRaises(ValueError): self.C.collect(evidence, pin)

    def test_missing_figure_export_or_result_file_cannot_be_silently_omitted(self):
        for suffix in (".png", ".pdf", ".json"):
            with self.subTest(suffix=suffix):
                corpus = Corpus()
                name = "figures/" + CASES[-1] + suffix
                del corpus.figures.payloads[name];corpus.figures.reseal()
                with corpus.admitted(self.C) as (evidence, pin):
                    with self.assertRaises(ValueError): self.C.collect(evidence, pin)

    def test_unsafe_index_paths_and_actual_evidence_symlinks_reject(self):
        for name in ("../escape.txt", "/absolute.txt", "bad\\alias.txt"):
            with self.subTest(name=name):
                corpus = Corpus()
                corpus.figures.index["files"][name] = {"sha256": "a" * 64, "bytes": 1}
                corpus.figures.refresh_index()
                with corpus.admitted(self.C) as (evidence, pin):
                    with self.assertRaises(ValueError): self.C.collect(evidence, pin)
        report = module("campaign_report")
        corpus = Corpus()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, raw in corpus.entries.items():
                path = root / name;path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(raw)
            image = root / corpus.figures.prefix / "figures" / (CASES[0] + ".png")
            target = image.with_suffix(".physical");image.rename(target);image.symlink_to(target)
            with corpus.admitted(self.C, report.Evidence(root)) as (evidence, pin):
                with self.assertRaisesRegex(ValueError, "symlink"): self.C.collect(evidence, pin)

    def test_figure_verifier_checks_roots_and_returned_dependency_case_artifact_receipt(self):
        for dependency in ("data", "plotter"):
            with self.subTest(dependency=dependency):
                corpus = Corpus()
                corpus.figures.summary[dependency + "_index_sha256"] = "a" * 64
                corpus.figures.payloads["proof-input.json"] = encoded(corpus.figures.summary)
                corpus.figures.reseal()
                with corpus.admitted(self.C) as (evidence, pin):
                    with self.assertRaises(AssertionError): self.C.collect(evidence, pin)
        mutations = {
            "data": 'result["dependencies"]["data_index_sha256"] = "a" * 64',
            "plotter": 'result["dependencies"]["plotter_index_sha256"] = "a" * 64',
            "cases": 'result["cases"] = list(reversed(result["cases"]))',
            "duplicate_case": 'result["cases"][1] = result["cases"][0]',
            "artifacts": 'result["artifacts"] = 11',
            "boolean_artifacts": 'result["artifacts"] = True',
            "float_artifacts": 'result["artifacts"] = 12.0',
            "index": 'result["index_sha256"] = "a" * 64',
            "verified": 'result["verified"] = False'}
        for defect, mutation in mutations.items():
            with self.subTest(receipt=defect):
                source = verifier_source(figures=True).replace(b"    return result\n", ("    " + mutation + "\n    return result\n").encode())
                corpus = Corpus(figure_verifier=source)
                with corpus.admitted(self.C) as (evidence, pin):
                    with self.assertRaises(ValueError): self.C.collect(evidence, pin)

    def test_verifiers_cannot_return_nonfinite_retained_summary_or_receipt(self):
        for role in ("data", "figures"):
            with self.subTest(role=role):
                source = verifier_source(figures=role == "figures")
                target = 'result["summary"]["retained_extra"]' if role == "data" else 'result["retained_extra"]'
                source = source.replace(b"    return result\n", ("    " + target + ' = {"nonfinite": float("nan")}\n    return result\n').encode())
                corpus = Corpus(**{"data_verifier" if role == "data" else "figure_verifier": source})
                with corpus.admitted(self.C) as (evidence, pin):
                    with self.assertRaises(ValueError): self.C.collect(evidence, pin)

    def test_bytecode_flag_restores_after_success_and_verifier_exception(self):
        previous = sys.dont_write_bytecode
        try:
            for fail in (False, True):
                with self.subTest(fail=fail):
                    sys.dont_write_bytecode = False
                    corpus = Corpus(data_verifier=verifier_source(fail=fail))
                    with corpus.admitted(self.C) as (evidence, pin):
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "synthetic root verifier failure"):
                                self.C.collect(evidence, pin)
                        else: self.C.collect(evidence, pin)
                    self.assertIs(sys.dont_write_bytecode, False)
        finally:
            sys.dont_write_bytecode = previous


class SyntheticOptionalRouting(unittest.TestCase):
    def test_report_validate_ledgers_appendix_helper_only_when_present(self):
        report = module("campaign_report")
        binding = {"id": "synthetic", "manifest_sha256": "invented-manifest"}
        ledgers = {}
        for state in ("absent", "null", "present"):
            with self.subTest(state=state):
                refresh = {} if state == "absent" else {"appendix_c_raw_dp": None if state == "null" else {"cases": []}}
                dossier = {"schema": report.SCHEMA, "new_numerical_execution": False,
                    "campaign_binding": binding, "refresh": refresh, "files": {}}
                actual = {key: value for key, value in dossier.items() if key != "files"}
                class RecordingInputs:
                    root = Path("/synthetic-never-opened")
                    def __init__(self): self.added = []
                    def json(self, name):
                        if name != "synthetic-dossier.json": raise AssertionError("Unexpected dossier read")
                        return deepcopy(dossier)
                    def add(self, name, expected=None):
                        self.added.append((name, expected))
                        return b"invented source bytes; no file is opened"
                inputs, evidence = RecordingInputs(), NoReads()
                with patch.object(report, "Evidence", return_value=evidence) as factory, \
                        patch.object(report, "collect", return_value=actual) as collect:
                    validated = report.validate(inputs, "synthetic-dossier.json", {"id": "synthetic"},
                        {"campaign_binding": binding})
                self.assertEqual(validated, dossier)
                factory.assert_called_once_with(inputs.root, {})
                collect.assert_called_once_with(evidence, "synthetic", None)
                ledgers[state] = inputs.added
        self.assertEqual(ledgers["absent"], ledgers["null"])
        self.assertNotIn(("campaign_report_appendix_c.py", None), ledgers["absent"])
        self.assertEqual(ledgers["present"], ledgers["absent"] + [("campaign_report_appendix_c.py", None)])

    def test_current_selection_admits_optional_shape_before_checkpoint_or_evidence_reads(self):
        refresh = module("campaign_report_refresh")
        helper = module("campaign_report_appendix_c")
        selected = {"schema": refresh.CURRENT_SCHEMA, "campaign": "synthetic", "numerical_checkpoint": {},
            "closed_cases": [], "documents": {}, "packages": {}, "implementation_packages": {},
            "historical_archive": {}, "local_handoff_path": "synthetic", "snapshot_date": "synthetic",
            "performance_status": "deferred to Hua-Sheng", "final_performance_accepted": False,
            "current_lifecycle": None,
            "appendix_c_raw_dp": {"index": "experiments/synthetic-unadmitted/evidence-index.json", "sha256": "a" * 64}}
        calls = []
        def sibling(name):
            calls.append(name)
            self.assertEqual(name, "campaign_report_appendix_c", "Checkpoint was accessed before admission")
            return helper
        with patch.object(refresh, "sibling", side_effect=sibling), patch.object(helper, "ADMITTED", {}), \
                patch.object(helper, "require_admitted", wraps=helper.require_admitted) as admitted:
            with self.assertRaisesRegex(ValueError, "not explicitly admitted"):
                refresh.current_selection(NoReads(), selected, {"id": "synthetic"}, {})
        self.assertEqual(calls, ["campaign_report_appendix_c"])
        admitted.assert_called_once_with(selected["appendix_c_raw_dp"])

    def test_absent_and_null_field_preserve_old_output_and_never_load_helper(self):
        refresh = module("campaign_report_refresh")
        report = {"numerical": {"cases": [], "status": "open"}, "criteria": {"stress_accuracy": "open"},
            "delivery_complete": False, "paired": {"status": "deferred"}}
        before = deepcopy(report)
        selected = {"schema": refresh.CURRENT_SCHEMA, "closed_cases": [], "performance_status": "deferred to Hua-Sheng",
            "local_handoff_path": "synthetic-handoff", "historical_archive": {}, "implementation_packages": {}}
        implementation = SimpleNamespace(collect=lambda *args, **kwargs: {}, apply=lambda *args: None)
        calls = []
        def sibling(name):
            calls.append(name)
            self.assertEqual(name, "campaign_report_implementation")
            return implementation
        with patch.object(refresh, "archive", return_value={}), \
                patch.object(refresh, "preserve_historical_disclosures"), \
                patch.object(refresh, "bind_index", return_value={"schema_version": 1, "new_native_evaluations": 0,
                    "files": {"analysis.json": {"sha256": refresh.N3_ANALYSIS_JSON_SHA}}}), \
                patch.object(refresh, "sibling", side_effect=sibling):
            absent = refresh.current_supplement(NoReads(), selected, report)
            null = refresh.current_supplement(NoReads(), {**selected, "appendix_c_raw_dp": None}, report)
        self.assertEqual(absent, null)
        self.assertNotIn("appendix_c_raw_dp", absent)
        self.assertEqual(calls, ["campaign_report_implementation"] * 2)
        self.assertEqual(report, before)
        self.assertIs(absent["final_performance_accepted"], False)

    def test_selected_capsule_adds_its_refresh_field_without_mutating_old_gates(self):
        refresh = module("campaign_report_refresh")
        helper = module("campaign_report_appendix_c")
        report = {"numerical": {"cases": [], "status": "open"}, "criteria": {"stress_accuracy": "open"},
            "delivery_complete": False, "paired": {"status": "deferred"}, "cohorts": {"status": "open"},
            "implementation": {"status": "retained"}, "registered_source_sha256": "synthetic-current-source6"}
        before = deepcopy(report)
        selected = {"schema": refresh.CURRENT_SCHEMA, "closed_cases": [], "performance_status": "deferred to Hua-Sheng",
            "local_handoff_path": "synthetic-handoff", "historical_archive": {}, "implementation_packages": {}}
        implementation = SimpleNamespace(collect=lambda *args, **kwargs: {}, apply=lambda *args: None)
        def sibling(name):
            self.assertIn(name, ("campaign_report_implementation", "campaign_report_appendix_c"))
            return implementation if name == "campaign_report_implementation" else helper
        corpus = Corpus()
        with patch.object(refresh, "archive", return_value={}), \
                patch.object(refresh, "preserve_historical_disclosures"), \
                patch.object(refresh, "bind_index", return_value={"schema_version": 1, "new_native_evaluations": 0,
                    "files": {"analysis.json": {"sha256": refresh.N3_ANALYSIS_JSON_SHA}}}), \
                patch.object(refresh, "sibling", side_effect=sibling), corpus.admitted(helper) as (evidence, pin):
            absent = refresh.current_supplement(NoReads(), selected, report)
            present = refresh.current_supplement(evidence, {**selected, "appendix_c_raw_dp": pin}, report)
        appendix = present.pop("appendix_c_raw_dp")
        self.assertEqual(present, absent)
        self.assertEqual(report, before)
        self.assertEqual(sum(len(case["lanes"]) for case in appendix["cases"]), 20)
        self.assertIs(appendix["main_campaign_modified"], False)
        self.assertIs(appendix["performance_eligible"], False)

    def test_raw_absent_null_and_selected_preserve_existing_retrospective_appendix(self):
        refresh = module("campaign_report_refresh")
        helper = module("campaign_report_appendix_c")
        report = {"numerical": {"cases": [], "status": "open"}, "criteria": {"stress_accuracy": "open"},
            "delivery_complete": False, "paired": {"status": "deferred"}, "cohorts": {"status": "open"},
            "registered_source_sha256": "invented-current-source"}
        before = deepcopy(report)
        retrospective_pin = {"index": "experiments/synthetic-retrospective/index.json", "sha256": "c" * 64}
        checkpoint_pin = {"index": "experiments/synthetic-checkpoint/index.json", "sha256": "d" * 64}
        selected = {"schema": refresh.CURRENT_SCHEMA, "closed_cases": [], "performance_status": "deferred to Hua-Sheng",
            "local_handoff_path": "synthetic-handoff", "historical_archive": {}, "implementation_packages": {},
            "appendix_c_retrospective": retrospective_pin, "numerical_checkpoint": checkpoint_pin}
        retrospective = {"scope": "invented accepted post-rescue retrospective", "cases": [{"id": "invented"}],
            "performance_eligible": False, "independent_truth_qualified": False}
        implementation = SimpleNamespace(collect=lambda *args, **kwargs: {}, apply=lambda *args: None)
        sibling_calls = []
        def sibling(name):
            sibling_calls.append(name)
            self.assertIn(name, ("campaign_report_implementation", "campaign_report_appendix_c"))
            return implementation if name == "campaign_report_implementation" else helper
        corpus = Corpus()
        with patch.object(refresh, "archive", return_value={}), \
                patch.object(refresh, "preserve_historical_disclosures"), \
                patch.object(refresh, "bind_index", return_value={"schema_version": 1, "new_native_evaluations": 0,
                    "files": {"analysis.json": {"sha256": refresh.N3_ANALYSIS_JSON_SHA}}}), \
                patch.object(refresh, "sibling", side_effect=sibling), \
                patch.object(refresh, "retrospective_plots", return_value=retrospective) as previous, \
                corpus.admitted(helper) as (evidence, pin):
            absent = refresh.current_supplement(NoReads(), selected, report)
            null = refresh.current_supplement(NoReads(), {**selected, "appendix_c_raw_dp": None}, report)
            present = refresh.current_supplement(evidence, {**selected, "appendix_c_raw_dp": pin}, report)
        self.assertEqual(previous.call_count, 3)
        for args in previous.call_args_list:
            self.assertEqual(args.args[1:], (retrospective_pin, checkpoint_pin))
        self.assertEqual(sibling_calls.count("campaign_report_appendix_c"), 1)
        self.assertEqual(absent, null)
        appendix = present.pop("appendix_c_raw_dp")
        self.assertEqual(present, absent)
        self.assertEqual(present["appendix_c_retrospective"], retrospective)
        self.assertEqual(report, before)
        self.assertEqual(sum(len(case["lanes"]) for case in appendix["cases"]), 20)
        self.assertIs(appendix["performance_eligible"], False)
        self.assertIs(present["final_performance_accepted"], False)


if __name__ == "__main__":
    unittest.main()

