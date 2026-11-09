"""Compile the measured report and record its exact document inputs.

Numerical measurements are never launched by this script. Run the benchmark
summarizer first; incomplete evidence is rendered as incomplete, not fabricated.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
BACKENDS = {"1", "6", "7", "8", "9"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_summary(data):
    """Refuse a passing headline when declared numerical evidence is absent."""
    cases = data["cases"]
    if len({row["id"] for row in cases}) != len(cases):
        raise ValueError("Duplicate process rows in report")
    if data["baseline_revision"] != "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f":
        raise ValueError("Report baseline differs from the approved frozen revision")
    if data["gate"]["passed"]:
        main = [row for row in cases if row["role"] == "main"]
        if (not data["gate"]["complete"] or len(main) != 8 or
                not data["gate"].get("consistent_candidate_source") or
                not data.get("candidate_revision")):
            raise ValueError("Passing report requires all eight main cases")
        for row in main:
            checks = row["correctness"]
            timing = row["performance"]
            if (row["points"] is None or row["points"] < 100 or len(checks) != 5 or
                    {str(check["backend"]) for check in checks} != BACKENDS or
                    not all(check["passed"] is True and check["samples"] >= 100
                            for check in checks) or
                    timing["passed"] is not True or not timing["complete"] or
                    timing["pairs"] < 7 or
                    timing["upper_95"] is None or
                    not math.isfinite(timing["upper_95"]) or timing["upper_95"] > 1.05):
                raise ValueError("Unqualified process in passing report: " + row["id"])
            regressions = row.get("existing_backend_regressions", [])
            if (len(regressions) != 4 or
                    {str(check["backend"]) for check in regressions} != BACKENDS - {"9"} or
                    not all(check["complete"] and check["passed"] is True
                            for check in regressions)):
                raise ValueError("Missing existing-backend regression evidence: " + row["id"])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark.pdf")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--font-path", type=Path, action="append", default=[])
    args = parser.parse_args()
    summary_path = ROOT / "summary.json"
    summary = validate_summary(json.loads(summary_path.read_text()))
    source_names = ["benchmark.typ", "style.typ", "summary.json",
                    "implementation-start.json", "style-provenance.json", "cases.json"]
    if summary.get("ratio_figure"):
        figure = (ROOT / summary["ratio_figure"]).resolve()
        if not figure.is_relative_to(ROOT):
            raise ValueError("Report figure must be inside the report directory")
        source_names.append(figure.relative_to(ROOT).as_posix())
    inputs = {name: {"sha256": sha256(ROOT / name),
                     "bytes": (ROOT / name).stat().st_size} for name in source_names}
    if args.verify_only:
        recorded = json.loads((ROOT / "report-build.json").read_text())
        if inputs != recorded["inputs"]:
            raise ValueError("Report inputs differ from the recorded PDF build")
        if sha256(args.output) != recorded["pdf_sha256"]:
            raise ValueError("PDF differs from its recorded build")
        print("Report inputs and PDF hashes verified")
        return
    typst = shutil.which("typst")
    if not typst:
        raise RuntimeError("Source tools/tirust_env.sh or activate the Nix shell")
    version = subprocess.check_output([typst, "--version"], text=True).strip()
    if not version.startswith("typst 0.15.1 "):
        raise RuntimeError("Reproduction requires Typst 0.15.1; found " + version)
    epoch = int(summary.get("source_date_epoch", 1794182400))
    command = [typst, "compile", "--root", str(ROOT),
               "--creation-timestamp", str(epoch)]
    for directory in args.font_path:
        command.extend(["--font-path", str(directory.resolve())])
    command.extend([str(ROOT / "benchmark.typ"), str(args.output.resolve())])
    environment = dict(os.environ, SOURCE_DATE_EPOCH=str(epoch), RAYON_NUM_THREADS="1")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, env=environment, check=True)
    manifest = {"typst": version, "source_date_epoch": epoch,
                "inputs": inputs, "pdf_sha256": sha256(args.output),
                "font_paths": [str(path.resolve()) for path in args.font_path],
                "font_search_environment": environment.get("TYPST_FONT_PATHS", ""),
                "numerical_campaign_rerun": False}
    (ROOT / "report-build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"pdf": str(args.output), "sha256": manifest["pdf_sha256"],
                      "gate": summary["gate"]}, indent=2))


if __name__ == "__main__":
    main()
