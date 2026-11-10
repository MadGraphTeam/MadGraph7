"""Freeze pure source-progress data and report preparation; never render."""
import ctypes
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
SELECTION = "experiments/current-source-report-supplement-proposal-1/selection.json"
DOSSIER = "campaigns/invariant-symmetry-1/report-snapshots/closed11-entry-cache-supplement-1.json"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    stage = Path(json.loads((HERE / "private-collection.json").read_text())["stage"])
    build = module("closed_preview_build", stage / "build_report.py")
    supplement = module("closed_preview_supplement", stage / "current_source_supplement.py")
    with patch.object(subprocess, "Popen", side_effect=RuntimeError("No process launches in report preparation")), \
            patch.object(ctypes, "CDLL", side_effect=RuntimeError("No foreign calls in report preparation")), \
            patch.object(ctypes, "PyDLL", side_effect=RuntimeError("No foreign calls in report preparation")):
        inputs = build.ReportInputs(stage)
        view, context_path = supplement.validate(inputs, SELECTION, "invariant-symmetry-1", DOSSIER)
        write("preview.json", {"source_supplement": view, "context_path": context_path})
        write("preview-inputs.json", inputs.files)
        summary, inventory, context, typst = build.prepare_report(stage, "invariant-symmetry-1", DOSSIER, SELECTION)
    write("prepared-context.json", context)
    write("prepared-inputs.json", inventory.files)
    receipt = {"historical_summary_gate": summary["gate"], "current_closed_cases": view["numerical"]["closed_cases"],
        "current_closed_lanes": view["numerical"]["closed_lanes"], "typst_inputs": typst,
        "historical_delivery_complete": context["delivery_complete"], "current_delivery_complete": view["delivery_complete"],
        "input_count": len(inventory.files), "context_written": False, "rendered": False,
        "process_or_foreign_calls": 0}
    write("prepared-receipt.json", receipt)
    print(json.dumps(receipt, indent=2))
