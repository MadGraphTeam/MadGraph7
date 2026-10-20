#!/usr/bin/env python3
"""Export a fresh EW3 process through the CLI with corrected MP generation."""
import json
import shutil
import sys

from probe import HERE, ROOT, WORK, sha

sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark as bench


def main():
    bench.WORK = WORK / "fresh-export"
    bench.REPORT = HERE / "fresh-export"
    if bench.WORK.exists() or bench.REPORT.exists():
        raise RuntimeError("Refusing to overwrite fresh-export evidence")
    config = json.loads((ROOT / "docs/benchmarkfolder/cases.json").read_text())
    case = next(c for c in config["cases"] if c["id"] == "ew-3")
    bench.generate("candidate", case)
    masses = bench.build("candidate", case)
    original = ROOT / ".benchmark-work/outputs/baseline/ew-3"
    target = bench.output_dir("candidate", case)
    assert (original / "Cards/param_card.dat").read_bytes() == (
        target / "Cards/param_card.dat").read_bytes()
    changed = []
    for path in sorted((target / "Source/MODEL").glob("*")):
        if path.suffix not in (".f", ".inc"):
            continue
        previous = original / path.relative_to(target)
        if path.is_file() and previous.exists() and path.read_bytes() != previous.read_bytes():
            changed.append(str(path.relative_to(target)))
            for label, source in (("original", previous), ("corrected", path)):
                retained = bench.REPORT / "model-source" / label / path.name
                retained.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, retained)
    for path in (bench.WORK / "logs").glob("*"):
        destination = bench.REPORT / "logs" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    result = {
        "acceptance_eligible": False,
        "scope": "Fresh CLI export; model helpers and generated MP REAL casts corrected",
        "masses": masses,
        "parameter_card_unchanged": True,
        "parameter_card_sha256": sha(target / "Cards/param_card.dat"),
        "changed_model_sources_against_original": changed,
        "exporter_sha256": sha(ROOT / "madgraph/iolibs/export_v4.py"),
        "parser_sha256": sha(ROOT / "madgraph/iolibs/ufo_expression_parsers.py"),
        "output": str(target.relative_to(ROOT)),
    }
    (bench.REPORT / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
