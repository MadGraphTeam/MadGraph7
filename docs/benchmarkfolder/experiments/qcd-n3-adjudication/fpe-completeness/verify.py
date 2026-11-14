#!/usr/bin/env python3
"""Independently reparse retained FPE/numerical evidence, without native work.

Attempt 1 omitted a parser file from its recorded protocol list. Preserve that
attempt and bind this separate verification to all parser dependencies instead
of retroactively asserting their original execution hashes.
"""
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench
import tirust_benchmark_cohorts as cohort
from tirust_complex_stress import parse_forced_qp
import run as helper


def main():
    report = Path(__file__).resolve().parent
    attempt = report / "attempt-1"
    destination = report / "verification-1"
    destination.mkdir(exist_ok=False)
    sources = {"verify.py": Path(__file__), "instrumentation.py": report / "run.py",
        "original-adjudication.py": report.parent / "run.py",
        "frozen-worker.py": report.parent / "protocol/tirust_benchmark_worker.py",
        **{name: ROOT / "tools" / name for name in (*cohort.PROTOCOLS, "tirust_complex_stress.py")}}
    for name, path in sources.items():
        shutil.copy2(path, destination / name)
    specification = json.loads((attempt / "specification.json").read_text())
    record = json.loads((attempt / "result.json").read_text())
    compilation = json.loads((attempt / "compilation.json").read_text())
    old = json.loads((attempt / "original-result.json").read_text())
    old_compilation = json.loads((attempt / "original-diagnostic-compilation.json").read_text())
    old_inputs = json.loads((attempt / "original-candidate-inputs.json").read_text())
    if any(bench.digest(attempt / name) != checksum for name, checksum in specification["protocols"].items()):
        raise ValueError("Original attempt protocol changed")
    if any(record[key] != value for key, value in specification.items()):
        raise ValueError("Result/specification identity changed")
    original_parent = helper.module(report.parent / "run.py", "verified_original_adjudication")
    pilot, config, case, parent, parent_file, card_sha = original_parent.inputs()
    original_parent.validate_binding(helper.ORIGINAL_REPORT, old, pilot, config, case, parent, parent_file, card_sha)
    point_record = json.loads((attempt / "points.json").read_text())
    if (specification["original_result_sha256"] != bench.digest(attempt / "original-result.json")
            or specification["parent_points_sha256"] != bench.digest(parent_file)
            or specification["points_sha256"] != bench.digest(attempt / "points.dat")
            or point_record != old["point_record"]
            or (attempt / "points.dat").read_bytes() != (report.parent / "points/minimal.dat").read_bytes()
            or (attempt / "MadLoopParams.dat").read_bytes() != (attempt / "original-MadLoopParams.dat").read_bytes()
            or bench.digest(attempt / "original-MadLoopParams.dat") != old["numerical_card_sha256"]
            or bench.digest(attempt / "original-param_card.dat") != card_sha):
        raise ValueError("Physical/numerical input binding changed")
    prefix = old_inputs["compilation"]["worker_render_parameters"]["prefix"]
    driver = helper.instrument((attempt / "original-check_sa.f").read_text(), prefix)
    if driver != (attempt / "check_sa.f").read_text():
        raise ValueError("Diagnostic driver differs from post-call instrumentation")
    changed = [key for key in compilation["sources_before"]
               if compilation["sources_after"].get(key) != compilation["sources_before"][key]]
    if (len(changed) != 1 or not changed[0].endswith("/check_sa.f")
            or set(compilation["sources_after"]) != set(compilation["sources_before"])
            or compilation["sources_after"][changed[0]] != bench.digest(attempt / "check_sa.f")
            or compilation["sources_before"] != old_compilation["sources"]
            or compilation["objects_before"] != old_compilation["numerical_objects_after"]
            or compilation["objects_before"] != compilation["objects_after"]
            or compilation["deps_before"] != old_compilation["link_inputs_after"]
            or compilation["deps_before"] != compilation["deps_after"]
            or not bench.same_compiler_policy(compilation["flags"], old_compilation["effective_compiler_flags"])
            or any(compilation["runtime_before"][key] != record["runtime_after"][key] for key in ("libraries", "archives", "metadata"))):
        raise ValueError("Driver-only numerical provenance changed")
    values = parse_forced_qp(attempt, 9, 4)
    for row in values["rows"]:
        row["point"] = specification["source_indices"][row["point"]-1]
    flags = helper.summarize_flags(attempt / "raw/fpe-calls.dat", 4)
    if values != record["observation"] or any(record[key] != value for key, value in flags.items()):
        raise ValueError("Derived values/flags disagree with raw evidence")
    if values["rows"] != old["observation"]["rows"]:
        raise ValueError("Instrumentation changed numerical rows")
    if (attempt / "raw/values.dat").read_bytes() != (helper.ORIGINAL_REPORT / "raw/values.dat").read_bytes():
        raise ValueError("Instrumented numerical output bytes changed")
    cohort.exclusive_json(destination / "verification.json", {
        "status": "verified", "scope": "Independent retained-evidence reparse; no native execution or accuracy certification",
        "original_parser_execution_hashes": "Not retroactively asserted; initial omission retained",
        "protocols": {name: bench.digest(destination / name) for name in sources},
        "evidence": {str(path.relative_to(attempt)): bench.digest(path) for path in sorted(attempt.rglob("*")) if path.is_file()},
        "flags": flags, "rows_equal_original": True, "values_bytes_equal_original": True})
    print("Verified: all reported flags false; same numerical rows and bytes; driver-only provenance.")


if __name__ == "__main__":
    main()
