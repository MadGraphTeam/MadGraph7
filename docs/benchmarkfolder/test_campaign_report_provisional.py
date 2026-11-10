"""Synthetic sealed-package ingestion; no active campaign or numerical library."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


PROVISIONAL = module("campaign_report_provisional")
REPORT = module("campaign_report")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


class Fixture:
    def __init__(self, root):
        self.root = root
        self.base = root / "closed"
        self.case_ids = ["case" + str(i) for i in range(11)]
        self.report = {"registered_source_sha256": "source-old", "numerical_identity_sha256": "numeric-old",
                       "registered_exporter_revision": "exporter-old",
                       "numerical": {"cases": [{"id": c} for c in self.case_ids]}}
        manifest = {"candidate": {"exporter_identity": {"files_sha256": "source-old", "revision": "exporter-old"},
                                   "numerical_identity_sha256": "numeric-old"},
                    "baseline": {"revision": "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f"}}
        self.docs = {"manifest.json": manifest}
        identity = write(self.base / "manifest.json", manifest)
        self.binding = {"id": "synthetic", "manifest_sha256": identity["sha256"]}
        self.report["campaign_binding"] = self.binding
        self.schedule = [{"id": "chain-" + c, "case": c, "kind": "chain", "backend": None,
                          "cap_seconds": 120, "chains": {"baseline": [1], "candidate": [9, 1]}}
                         for c in self.case_ids]
        self.schedule += [{"id": f"standalone-{c}-{b}", "case": c, "kind": "standalone", "backend": b,
                           "cap_seconds": 180, "chains": {"baseline": [1 if b == 9 else b], "candidate": [b]}}
                          for c in self.case_ids for b in PROVISIONAL.BACKENDS]
        self.spec = {"schema": PROVISIONAL.RUNTIME_SCHEMA, "campaign_binding": self.binding,
                     "pairs": 3, "points": 100, "point_subset": None, "cpu_reservation": False,
                     "total_cap_seconds": 1800, "performance_acceptance_eligible": False,
                     "schedule": self.schedule}
        rows = [{"comparison": row, "status": "not_run_budget", "performance_acceptance_eligible": False}
                for row in self.schedule]
        rows[0].update(status="complete", elapsed_seconds_including_cleanup=12.)
        self.closure = {"schema": PROVISIONAL.RUNTIME_SCHEMA, "status": "closed_budget", "rows": rows,
                        "performance_acceptance_eligible": False, "precise_acceptance": "deferred_to_local",
                        "budget_seconds": 1800, "elapsed_seconds": 1800.1, "budget_exceeded": True,
                        "publication_scope": "administrative-after-deadline; outside work cap; no measurement"}
        self.result = {"status": "complete", "comparison": self.schedule[0], "numerical_pass": True,
                       "performance_acceptance_eligible": False,
                       "statistics": {"paired_repetitions": 3, "performance_acceptance_eligible": False,
                                      "paired_ratios": [1., 1., 1.], "geometric_mean_paired_ratio": 1.,
                                      "ratio_range": [1., 1.], "median_seconds_per_point": {"baseline": .01, "candidate": .01}},
                       "timings": {"baseline": [.01] * 3, "candidate": [.01] * 3},
                       "calibration_loops": {"baseline": 1, "candidate": 1},
                       "observations": {"baseline": [{"seconds": 1.}] * 3, "candidate": [{"seconds": 1.}] * 3},
                       "actual_backend_precision_counts": {"baseline": {"1:DP": 300}, "candidate": {"9:QP": 300}},
                       "initialization_and_warmup_seconds": {"baseline": 2., "candidate": 3.}}
        self.docs.update({"specification.json": self.spec, "closure.json": self.closure,
                          "comparison/completion.json": rows[0], "comparison/result.json": self.result,
                          "comparison/baseline-ready.json": {"cold_seconds": 2.},
                          "comparison/candidate-ready.json": {"cold_seconds": 3.}})
        bindings = {row["id"]: {"completion": None} for row in self.schedule}
        bindings[self.schedule[0]["id"]] = {"completion": "comparison/completion.json",
            "result": "comparison/result.json", "ready": {s: f"comparison/{s}-ready.json" for s in ("baseline", "candidate")}}
        self.index = {"schema": PROVISIONAL.SCHEMA, "closed": True, "performance_acceptance_eligible": False,
                      "campaign_binding": self.binding, "paths": {"manifest": "manifest.json",
                      "specification": "specification.json", "closure": "closure.json"},
                      "comparisons": bindings, "replay": {"path": "replay.py", "function": "replay_completed"}}
        self.base.joinpath("replay.py").write_text("# Synthetic wrapper is never a native timing producer.\n")
        self.replay_called = False

    def seal(self):
        files = {name: write(self.base / name, value) for name, value in self.docs.items()}
        raw = (self.base / "replay.py").read_bytes()
        files["replay.py"] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        self.index["files"] = files
        identity = write(self.base / "index.json", self.index)
        return {"index": "closed/index.json", "sha256": identity["sha256"]}

    def replay(self):
        self.replay_called = True
        return {self.schedule[0]["id"]: deepcopy(self.result)} if self.closure["rows"][0]["status"] == "complete" else {}

    def collect(self):
        return PROVISIONAL.collect(REPORT.Evidence(self.root), self.seal(), self.report, replay=self.replay)


def test_closed_projection_preserves_unavailable_rows_and_old_identity(tmp_path):
    f = Fixture(tmp_path)
    result = f.collect()
    assert f.replay_called and len(result["rows"]) == 66
    assert result["rows"][0]["ratio"] == 1.
    assert result["rows"][0]["initialization_seconds"] == {"baseline": 2., "candidate": 3.}
    assert result["rows"][0]["batch_seconds"] == {"baseline": [1.] * 3, "candidate": [1.] * 3}
    assert all(row["ratio"] is None for row in result["rows"][1:])
    assert result["candidate_source_sha256"] == "source-old"
    assert result["precise_acceptance"] == "deferred to Hua-Sheng"
    assert result["performance_acceptance_eligible"] is False


def test_result_left_after_deadline_never_becomes_timing(tmp_path):
    f = Fixture(tmp_path)
    f.closure["rows"][0]["status"] = "unavailable_deadline"
    result = f.collect()
    assert result["rows"][0]["status"] == "unavailable_deadline"
    assert result["rows"][0]["ratio"] is None
    assert result["rows"][0]["accepted_usage"] is None


@pytest.mark.parametrize("mutation", [
    lambda f: f.index.update(closed=False),
    lambda f: f.index.update(performance_acceptance_eligible=True),
    lambda f: f.closure.update(status="running"),
    lambda f: f.spec.update(pairs=2),
    lambda f: f.spec.update(points=99),
    lambda f: f.spec.update(cpu_reservation=True),
    lambda f: f.closure["rows"].pop(),
    lambda f: f.report.update(registered_source_sha256="symmetry-v2"),
    lambda f: f.result.update(numerical_pass=False),
    lambda f: f.result["statistics"].update(upper_95=1.01),
    lambda f: f.result["timings"]["candidate"].pop(),
    lambda f: f.result["timings"]["candidate"].__setitem__(0, 0.),
    lambda f: f.result["initialization_and_warmup_seconds"].update(candidate=9.),
    lambda f: f.result["actual_backend_precision_counts"]["candidate"].update({"9:QP": 299}),
    lambda f: f.index["comparisons"][f.schedule[0]["id"]].update(completion=None),
    lambda f: f.index["comparisons"].pop(f.schedule[-1]["id"]),
    lambda f: f.index["replay"].update(path="../replay.py"),
    lambda f: f.closure["rows"][0].update(elapsed_seconds_including_cleanup=999.),
    lambda f: f.closure["rows"][0].update(elapsed_seconds_including_cleanup=-1.),
    lambda f: f.closure.update(elapsed_seconds=-1.),
    lambda f: f.closure.update(budget_exceeded=False),
])
def test_scope_and_measurement_substitutions_are_rejected(tmp_path, mutation):
    f = Fixture(tmp_path)
    mutation(f)
    with pytest.raises(ValueError):
        f.collect()


def test_outer_corruption_is_rejected_before_frozen_code(tmp_path):
    f = Fixture(tmp_path)
    selected = f.seal()
    (f.base / "comparison/result.json").write_text("{}")
    with pytest.raises(ValueError, match="hash differs"):
        PROVISIONAL.collect(REPORT.Evidence(tmp_path), selected, f.report, replay=f.replay)
    assert not f.replay_called


def test_independently_modified_completion_is_rejected(tmp_path):
    f = Fixture(tmp_path)
    f.docs["comparison/completion.json"] = {**f.closure["rows"][0], "status": "unavailable_deadline"}
    with pytest.raises(ValueError, match="Closure/completion disagreement"):
        f.collect()


def test_compact_retained_accessor_stays_within_logical_allowlist(tmp_path):
    f = Fixture(tmp_path)
    f.seal()
    f.index["retained_files"] = deepcopy(f.index["files"])
    wrapper = f.index["files"]["replay.py"]
    f.index["files"] = {"replay.py": wrapper}
    selected = {"index": "closed/index.json", "sha256": write(f.base / "index.json", f.index)["sha256"]}
    result = PROVISIONAL.collect(REPORT.Evidence(tmp_path), selected, f.report,
                                 replay=f.replay, reader=lambda name: f.docs[name])
    assert result["rows"][0]["ratio"] == 1.
    del f.index["retained_files"]["comparison/candidate-ready.json"]
    selected["sha256"] = write(f.base / "index.json", f.index)["sha256"]
    with pytest.raises(ValueError, match="not indexed"):
        PROVISIONAL.collect(REPORT.Evidence(tmp_path), selected, f.report,
                            replay=f.replay, reader=lambda name: f.docs[name])


def test_compact_inventory_is_loaded_once_for_all_report_reads(tmp_path, monkeypatch):
    f = Fixture(tmp_path)
    selected = f.seal()
    f.index["retained_files"] = deepcopy(f.index["files"])
    f.index["files"] = {"replay.py": f.index["files"]["replay.py"]}
    selected["sha256"] = write(f.base / "index.json", f.index)["sha256"]
    loads = []
    def loader(*, expected_index_sha256):
        loads.append(expected_index_sha256)
        raw = {name: json.dumps(value).encode() for name, value in f.docs.items()}
        raw["replay.py"] = b"# fixture"
        return deepcopy(f.index), raw
    monkeypatch.setattr(PROVISIONAL.runpy, "run_path", lambda path: {
        "_load": loader, "parsed": json.loads, "replay_completed": f.replay})
    result = PROVISIONAL.collect(REPORT.Evidence(tmp_path), selected, f.report)
    assert loads == [selected["sha256"]]
    assert f.replay_called and result["rows"][0]["ratio"] == 1.


@pytest.mark.parametrize("campaign_entrypoint", [False, True])
def test_synthetic_appendix_compiles_without_touching_canonical_report(tmp_path, campaign_entrypoint):
    typst = shutil.which("typst")
    if typst is None:
        pytest.skip("Typst unavailable")
    result = Fixture(tmp_path).collect()
    for name in ("campaign-provisional.typ", "campaign-report.typ", "style.typ"):
        shutil.copyfile(HERE / name, tmp_path / name)
    write(tmp_path / "synthetic.json", {"refresh": {"provisional": result}} if campaign_entrypoint else result)
    importer = ('#import "campaign-report.typ": show-campaign-provisional\n' if campaign_entrypoint
                else '#import "campaign-provisional.typ": show-provisional\n')
    invocation = 'show-campaign-provisional' if campaign_entrypoint else 'show-provisional'
    (tmp_path / "preview.typ").write_text(
        importer +
        '#import "style.typ": report-style\n'
        '#show: body => report-style(body, date: "synthetic layout check", revision: "synthetic")\n'
        '#' + invocation + '(json("synthetic.json"), (), c => c.id)\n')
    outcome = subprocess.run([typst, "compile", str(tmp_path / "preview.typ"), str(tmp_path / "preview.pdf")],
                             capture_output=True, text=True, timeout=60)
    assert outcome.returncode == 0, outcome.stderr
    assert (tmp_path / "preview.pdf").read_bytes().startswith(b"%PDF-")
