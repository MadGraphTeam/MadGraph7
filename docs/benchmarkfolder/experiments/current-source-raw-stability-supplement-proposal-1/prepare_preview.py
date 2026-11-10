"""Explicit proposed-file mappings for a read-only private report preparation."""
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DOCS = HERE.parents[1]
SELECTION = "campaigns/entry-latch-cache-guard-1/report-supplements/progress-2/selection.json"
DOSSIER = "campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-1.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = load("raw_slot_build", DOCS / "build_report.py")
HELPER = load("raw_slot_helper", HERE / "proposed/current_source_supplement.py")


def prepare():
    """Do not write the future context, render, or touch canonical inputs."""
    mapped = {SELECTION: HERE / "selection.json", **{
        name: HERE / "proposed" / name for name in
        ("current_source_supplement.py", "current-source-supplement.typ")}}

    class ProposedInputs(BUILD.ReportInputs):
        def add(self, name, expected=None):
            if name not in mapped:
                return super().add(name, expected)
            raw = mapped[name].read_bytes()
            checksum = hashlib.sha256(raw).hexdigest()
            if expected is not None and expected != checksum:
                raise ValueError("Proposed review input hash differs: " + name)
            self.files[name] = {"sha256": checksum, "bytes": len(raw)}
            return raw

    original_spec = importlib.util.spec_from_file_location

    def proposed_spec(name, path, *args, **kwargs):
        if name == "current_source_supplement":
            path = HERE / "proposed/current_source_supplement.py"
        return original_spec(name, path, *args, **kwargs)

    with patch.object(BUILD, "ReportInputs", ProposedInputs), \
            patch.object(importlib.util, "spec_from_file_location", proposed_spec), \
            patch.object(subprocess, "Popen", side_effect=AssertionError("process launch")), \
            patch.object(ctypes, "CDLL", side_effect=AssertionError("foreign call")), \
            patch.object(ctypes, "PyDLL", side_effect=AssertionError("foreign call")):
        summary, inputs, context, typst_inputs = BUILD.prepare_report(
            DOCS, HELPER.HISTORICAL, DOSSIER, SELECTION)
    return summary, inputs, context, typst_inputs


if __name__ == "__main__":
    if sys.flags.optimize:
        raise ValueError("Use unoptimized Python for retained replay")
    summary, inputs, context, typst_inputs = prepare()
    old = json.loads((DOCS / "campaigns/entry-latch-cache-guard-1/report-supplements/progress-1/report-context.json").read_text())
    stripped = json.loads(json.dumps(context))
    raw = stripped["source_supplement"].pop("raw_stability")
    if stripped != old:
        raise ValueError("The optional raw slot changed the previous context")
    receipt = {"verified": True, "existing_context_fields_identical": True,
        "current_closed_cases": context["source_supplement"]["numerical"]["closed_cases"],
        "current_closed_lanes": context["source_supplement"]["numerical"]["closed_lanes"],
        "raw_primary_calls": raw["primary_calls"], "raw_bootstrap_calls": raw["bootstrap_calls"],
        "raw_available": [row["available_points"] for row in raw["summary"]["rows"]],
        "input_count": len(inputs.files), "typst_inputs": typst_inputs,
        "canonical_context_written": False, "rendered": False,
        "native_or_compiler_executions": 0, "process_or_foreign_calls": 0}
    for name, value in (("preview.json", context), ("preview-inputs.json", inputs.files),
                        ("preview-receipt.json", receipt)):
        (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
