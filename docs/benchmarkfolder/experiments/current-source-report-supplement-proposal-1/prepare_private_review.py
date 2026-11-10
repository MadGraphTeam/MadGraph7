"""Prepare a closed indexed text tree and collect the new historical dossier.

Only exact published-ledger and closed-preview inventory paths are read. Existing
data are hard linked read-only; proposed code/archive/selection files replace
their links before writing. This script never renders or launches native work.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DOCS = ROOT / "docs/benchmarkfolder"
SELECTION = "campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-selection-1.json"
DOSSIER = "campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-1.json"
ARCHIVE = "history/report-before-entry-latch-cache-guard-1/PRECISION_STREAM_TODO.md"
PROPOSED = ("build_report.py", "benchmark.typ", "campaign_report_refresh.py",
            "current_source_supplement.py", "current-source-supplement.typ")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def replace(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    path.write_bytes(raw)


def prepare(stage):
    stage = Path(stage).resolve()
    if not stage.is_relative_to(ROOT / ".tirust-build") or stage.exists():
        raise ValueError("Use a fresh explicit stage below this checkout's .tirust-build")
    stage.mkdir(parents=True)
    published = json.loads((DOCS / "report-build.json").read_text())
    if published["pdf_sha256"] != "91ceeae99968df742e7e141467e3bab321394a0512328be7df9be0f5b86d559a":
        raise ValueError("Published historical report anchor changed")
    records = dict(published["inputs"])
    records.update(json.loads((HERE / "preview-inputs.json").read_text()))
    # The old global TODO is now deliberately replaced by a named archive role.
    records.pop("PRECISION_STREAM_TODO.md", None)
    private_prefix = "experiments/current-source-report-supplement-proposal-1/"
    for name, item in records.items():
        if name.startswith(private_prefix) or name in (SELECTION, DOSSIER, ARCHIVE):
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe indexed staging path")
        source = DOCS
        for part in relative.parts:
            source /= part
            if source.is_symlink():
                raise ValueError("Indexed staging source contains a symlink")
        raw = source.read_bytes()
        if source.is_symlink() or digest(raw) != item["sha256"] or len(raw) != item["bytes"]:
            raise ValueError("Indexed staging source differs: " + name)
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)
    for name in PROPOSED:
        replace(stage / name, (HERE / "proposed" / name).read_bytes())
    replace(stage / ARCHIVE, (HERE / "historical-PRECISION_STREAM_TODO.md").read_bytes())
    replace(stage / SELECTION, (HERE / "historical-selection.json").read_bytes())
    # The separate current-source supplement binds the current navigation text.
    replace(stage / "PRECISION_STREAM_TODO.md", (DOCS / "PRECISION_STREAM_TODO.md").read_bytes())
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location("private_historical_report", stage / "campaign_report.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        evidence = module.Evidence(stage)
        result = module.collect(evidence, "invariant-symmetry-1", SELECTION)
        result["files"] = dict(sorted(evidence.files.items()))
    finally:
        sys.dont_write_bytecode = previous
    raw = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    replace(stage / DOSSIER, raw)
    replace(stage / "experiments/current-source-report-supplement-proposal-1/selection.json",
            (HERE / "selection.json").read_bytes())
    (HERE / "historical-dossier.json").write_bytes(raw)
    receipt = {"stage": str(stage), "historical_selection_path": SELECTION,
        "historical_dossier_path": DOSSIER, "historical_dossier_sha256": digest(raw),
        "criteria": result["criteria"], "delivery_complete": result["delivery_complete"],
        "inputs": len(result["files"]), "native_or_typst_execution": False}
    (HERE / "private-collection.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    print(json.dumps(prepare(parser.parse_args().stage), indent=2))
