#!/usr/bin/env python3
"""Inventory preserved preparation attempts without executing physics code."""
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
ROOT = REPORT.parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path):
    return {"path": str(path.relative_to(REPORT)), "sha256": digest(path),
            "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Create or explicitly refresh the derived index")
    args = parser.parse_args()
    original = HERE / "iregi-preparation-failure/cli-output.txt"
    isolated = HERE / "iregi-preparation-failure/isolated-build-output.txt"
    wrapper = HERE / "iregi-preparation-failure/baseline-makefile"
    text = original.read_text()
    assert "generate u u~ > z [virt=QCD]" in text
    assert "Cannot open module file" in text and "global.mod" in text
    assert ".benchmark-work/tirust/baseline/vendor/IREGI/src" in text
    requested = re.search(r"compile (\S*/\.benchmark-work/tirust/baseline/vendor/IREGI/src)", text).group(1)
    entered = re.search(r"\(cd (\S*/vendor/IREGI/src);make clean", text).group(1)
    assert requested != entered and "/.benchmark-work/" not in entered
    assert "ranlib libiregi.a" in isolated.read_text()
    # The source record below is a retained baseline snapshot, not current HEAD.
    baseline = json.loads((REPORT / "inputs/baseline.json").read_text())
    assert baseline["revision"] == "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f"
    assert digest(wrapper) == baseline["files"]["vendor/IREGI/src/makefile"]
    assert "$(PWD)/vendor/IREGI/src" in wrapper.read_text()
    old_command = HERE / "loop-induced-output-syntax/baseline-li_gg_h.mg5"
    new_command = REPORT / "commands/baseline-li_gg_h.mg5"
    generation_path = REPORT / "inputs/baseline/li_gg_h/generation.json"
    generation = json.loads(generation_path.read_text())
    assert "generate g g > h [noborn=QCD]" in old_command.read_text()
    assert "generate g g > h [sqrvirt=QCD]" in new_command.read_text()
    assert digest(new_command) == generation["command_sha256"]
    assert generation["source_identity"]["revision"] == baseline["revision"]
    value = {"schema": "madloop-inactive-preparation-evidence-v1",
        "scope": "Preparation and protocol history; no numerical or timing acceptance claims",
        "script_sha256": digest(Path(__file__)),
        "records": [
            {"directory": "experiments/iregi-preparation-failure", "classification": "dependency_preparation_failure",
             "observed_failure": "IREGI global.mod missing; nested make entered main checkout instead of baseline",
             "requested_directory": requested, "entered_directory": entered,
             "subsequent_record": "isolated baseline build log reaches ranlib libiregi.a",
             "missing_original_evidence": ["complete environment", "separate exit-status record"],
             "files": [record(original), record(isolated), record(wrapper)],
             "baseline_wrapper": {**record(wrapper), "original_path": "vendor/IREGI/src/makefile",
                                  "revision": baseline["revision"]}},
            {"directory": "experiments/loop-induced-output-syntax", "classification": "superseded_command_only",
             "observed_change": "standalone loop-induced command changed from noborn to sqrvirt",
             "missing_original_evidence": ["initial CLI stdout/stderr", "initial exit status"],
             "files": [record(old_command)],
             "successful_command": record(new_command), "successful_generation": record(generation_path)}]}
    numerical = HERE / "numerical/inactive-preparation-audit.json"
    if numerical.exists():
        value["independent_numerical_preparation_audit"] = record(numerical)
    output = HERE / "inactive-preparation-index.json"
    if args.write:
        output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    else:
        assert json.loads(output.read_text()) == value, "Inactive preparation evidence or index changed"
    print(json.dumps({"index": str(output.relative_to(ROOT)), "preparation_groups": len(value["records"]),
                      "independent_numerical_audit_present": numerical.exists()}))


if __name__ == "__main__":
    main()
