#!/usr/bin/env python3
"""Retain/check the exact compiled sources shared with the earlier campaign."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "madgraph/__init__.py").is_file())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def retain(run_id, write):
    report = HERE / run_id
    result = json.loads((report / "result.json").read_text())
    assert result["complete"]
    earlier = HERE.parent / "qcd3-factor-quiet-v1/generated-sources"
    records = {}
    for variant in ("original", "factor"):
        compiled = json.loads((report / (variant + "-compiled-sources.json")).read_text())
        for previous in sorted((earlier / variant).iterdir()):
            target = report / "generated-sources" / variant / previous.name
            relative = "SubProcesses/P0_uux_zgg/" + previous.name
            if write:
                current = ROOT / ".benchmark-work/experiments/symmetric-factorization"
                current /= run_id + "/" + variant + "/" + relative
                assert current.read_bytes() == previous.read_bytes()
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(current.read_bytes())
            assert target.read_bytes() == previous.read_bytes()
            digest = sha(target)
            assert compiled["files"][relative]["sha256"] == digest
            records[str(target.relative_to(report))] = dict(
                sha256=digest, bytes=target.stat().st_size,
                earlier_source=str(previous.relative_to(ROOT)),
                compiled_manifest=variant + "-compiled-sources.json",
                compiled_manifest_sha256=sha(
                    report / (variant + "-compiled-sources.json")),
                compiled_relative_path=relative)
    record = dict(
        complete=True, all_selected_sources_byte_identical_to_earlier_seven=True,
        files=records, script_sha256=sha(Path(__file__)),
        scope="Exact original/factored polynomial, coefficient batch, driver, "
              "dimensions and make recipe, cross-checked against this run's "
              "compiled-source manifests and the earlier sealed campaign. "
              "All other generated sources are retained by the full manifests.")
    path = report / "source-equivalence.json"
    if write:
        with path.open("x") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
    else:
        assert json.loads(path.read_text()) == record
    print("Exact source equivalence verified:", len(records), "files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    retain(args.run_id, args.write)
