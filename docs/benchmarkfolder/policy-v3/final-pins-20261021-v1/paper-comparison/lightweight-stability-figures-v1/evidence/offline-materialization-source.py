"""Offline layout of the exact previously validated actual figure arrays.

No native archive, generated process, point generator, or reduction is opened.
The original live verification and independent audits remain separate evidence.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "bin/madgraph").is_file())
FIRST = HERE / "attempt-1"
RECEIPT_SHA256 = "86082074d7284c3d8fa15caa141cbca991959bc05faab4eb2015c7693f2f803c"
AUDITS = {
    "arithmetic": ("light-eight-arithmetic-audit-v1.json", "99e0f508d32c2c668fe0a54a9c2e4cf3361e8915197ad54782137ef4c027baa7"),
    "raw_custody": ("light8-final-raw-custody-ownership-audit-v1.json", "7a99193f3970403891d1e2dd5285843779c8063274dac4966a206ec4489811d4"),
    "composite": ("light-eight-final-independent-composite-audit-v1.json", "64ac6c50cb1712272e11dbca41e797e7389d893df88ca67c1189e8f1bdf14e33")}


def digest(filename):
    return hashlib.sha256(filename.read_bytes()).hexdigest()


def bound(filename):
    return {"path": str(filename.relative_to(ROOT)), "sha256": digest(filename), "bytes": filename.stat().st_size}


def write(filename, value):
    filename.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("actual_light_stability_layout", HERE/"render-print.py")
    layout = importlib.util.module_from_spec(spec); spec.loader.exec_module(layout)
    output = layout.paper.path(args.output)
    layout.paper.require(not output.exists(), "Preserve previous offline layouts")
    first_receipt = FIRST/"render-receipt.json"
    layout.paper.require(digest(first_receipt) == RECEIPT_SHA256, "Original live-verification receipt differs")
    original = json.loads(first_receipt.read_text())
    records = {record["path"]: record for record in original["artifacts"]}

    def verified(name):
        filename = FIRST/name; record = records[str(filename.relative_to(ROOT))]
        layout.paper.require(bound(filename) == record, "Materialized actual figure input changed: "+name)
        return filename

    snapshot_file = verified("validated-snapshot.json")
    snapshot = json.loads(snapshot_file.read_text()); dp = snapshot["stability"]
    layout.paper.require(dp["physical_qualified"] is True and dp["complete"] is True and
                         dp["cohort"] == layout.COHORT and dp["manifest"]["sha256"] == layout.PINS["manifest.json"],
                         "Unexpected actual figure snapshot")
    sources = {"original_live_verification": bound(first_receipt), "validated_snapshot": bound(snapshot_file),
               "layout": bound(HERE/"render-print.py"), "offline_renderer": bound(Path(__file__))}
    audit_records = {}
    for key, (name, sha256) in AUDITS.items():
        filename = ROOT/".benchmark-work/final-release-audit-20261021-v1"/name
        layout.paper.require(digest(filename) == sha256, "Independent audit changed: "+key)
        audit_records[key] = bound(filename)
    arrays = {estimator: verified("figure-data-"+estimator+".json") for estimator in ("combined", "direction", "lorentz")}
    output.mkdir(parents=True)
    os.environ["MPLCONFIGDIR"] = str(ROOT/".ninjars-build/paper-report-mpl")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot
    with pyplot.rc_context({"font.family": "Libertinus Serif", "font.size": 10.3, "text.color": "#1f3b63",
            "axes.labelcolor": "#1f3b63", "axes.titlecolor": "#1f3b63", "mathtext.fontset": "stix",
            "svg.fonttype": "none", "svg.hashsalt": "actual-lightweight-dp-v1"}):
        for estimator, filename in arrays.items():
            prepared = json.loads(filename.read_text())
            shutil.copyfile(filename, output/filename.name)
            for group, pair in layout.PAIRS.items():
                layout.draw(prepared, pair, estimator, output/(estimator+"-"+group), pyplot)
    hardware = {cid: {"N": case["N"], "transformed_observations_per_tool": 3*case["N"],
        "tools": {backend: {"hardware_exception_observations": tool["hardware_exception_observations"],
            "algorithmic_failure_categories": tool["failure_categories"],
            "classification_counts": tool["classification_counts"]} for backend, tool in case["tools"].items()}}
        for cid, case in dp["cases"].items()}
    write(output/"hardware-diagnostics.json", hardware)
    shutil.copyfile(verified("counts.json"), output/"counts.json")
    for key, record in audit_records.items():
        shutil.copyfile(ROOT/record["path"], output/("independent-"+key+"-audit.json"))
    receipt = {"schema": "offline-actual-lightweight-dp-figure-render-v1", "software_fixture": False,
        "scientific_data_changed": False, "native_evaluations": 0, "native_artifacts_opened": 0,
        "source_runtime_qualification": sources["original_live_verification"], "sources": sources,
        "audits": audit_records, "independent_selected_eight_diagnostic_audit": "passed",
        "hardware_scope": "Saved IEEE invalid is observed in all 300 transformed observations on five NRS streams; plain-prefix observations match. It is distinct from zero algorithmic FPE/rejections/nonfinite/MP-comp and is not converted into a numerical error or a zero-hardware claim.",
        "requested_tools": [6, 8], "points_per_case_per_tool": 100, "main_report_changed": False,
        "publication": "actual audited figures; awaiting root visual review and final timing report assembly",
        "matplotlib": matplotlib.__version__, "array_inputs": {name:bound(filename) for name,filename in arrays.items()},
        "artifacts": [bound(filename) for filename in sorted(output.iterdir()) if filename.is_file()]}
    write(output/"render-receipt.json", receipt)
    print(json.dumps({"output": str(output), "figures": 12, "native_artifacts_opened": 0,
                      "independent_selected_eight_diagnostic_audit": "passed"}))


if __name__ == "__main__":
    main()
