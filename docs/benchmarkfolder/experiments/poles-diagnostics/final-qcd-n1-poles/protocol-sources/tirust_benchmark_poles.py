#!/usr/bin/env python3
"""Compare full and poles-only tensor diagnostics in an exported MadLoop process.

This never invokes an incomplete physical amplitude. It clones the exported
process into ignored working space and retains all fixture/output bytes.
"""
import argparse
from decimal import Decimal, localcontext
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import sys

import tirust_benchmark as bench
from tirust_benchmark_compiler import capture_effective_flags, validate_compiler_equality


def fixture_stream(fixtures, max_coefficients, loops, precision):
    selected, omitted = [], []
    modes = (1, 4) if precision == "both" else (1,) if precision == "dp" else (4,)
    for fixture in fixtures:
        n, rank = fixture["n"], fixture["rank"]
        count = math.comb(rank + 4, 4)
        if count > max_coefficients:
            omitted.append({"name": fixture["name"], "reason": "rank exceeds generated process capacity"})
            continue
        if not (1 <= n <= 6 and 0 <= rank <= 6 and len(fixture["shifts"]) == n
                and len(fixture["mass2"]) == n and fixture["mu2"] > 0):
            raise ValueError("Invalid undotted tensor fixture")
        if any(value != 0 for value in fixture["shifts"][0]):
            raise ValueError("Fixture shifts must use the same zero origin as MadLoop")
        for mode in (0, 1, 2):
            for ctmode in modes:
                selected.append({**fixture, "pole_mode": mode, "ctmode": ctmode, "loops": loops})
    lines = [str(len(selected))]
    for fixture in selected:
        lines.append(f'{fixture["n"]} {fixture["rank"]} {fixture["mu2"]} {loops} {fixture["pole_mode"]} {fixture["ctmode"]}')
        lines.extend(" ".join(format(float(value), ".17e") for value in vector) for vector in fixture["shifts"])
        lines.extend(" ".join(format(float(value), ".17e") for value in mass) for mass in fixture["mass2"])
    return "\n".join(lines) + "\n", selected, omitted


def parse_results(text, selected):
    frames = {}
    for line in text.splitlines():
        fields = line.split()
        if fields[0] == "F":
            if len(fields) != 10:
                raise ValueError("Malformed diagnostic frame")
            identifier, ctmode, mode, finite, loops = map(int, fields[1:6])
            key = (identifier, finite)
            if key in frames or not 1 <= identifier <= len(selected):
                raise ValueError("Duplicate or unknown diagnostic frame")
            fixture = selected[identifier - 1]
            if (ctmode, mode, loops) != (fixture["ctmode"], fixture["pole_mode"], fixture["loops"]):
                raise ValueError("Diagnostic fixture metadata changed")
            frames[key] = {"stable": fields[6] == "T", "finite_nan": fields[7] == "T",
                           "count": int(fields[8]), "seconds": str(Decimal(fields[9])), "rows": []}
        elif fields[0] == "C":
            if len(fields) != 12:
                raise ValueError("Malformed diagnostic coefficient")
            identifier, ctmode, mode, finite, coefficient = map(int, fields[1:6])
            frame = frames[(identifier, finite)]
            fixture = selected[identifier - 1]
            if (ctmode, mode) != (fixture["ctmode"], fixture["pole_mode"]) or coefficient != len(frame["rows"]):
                raise ValueError("Unordered or mismatched tensor coefficients")
            frame["rows"].append(fields[6:])
        else:
            raise ValueError("Unknown diagnostic record")
    results = []
    with localcontext() as context:
        context.prec = 80
        for identifier, fixture in enumerate(selected, 1):
            full, partial = frames[(identifier, 1)], frames[(identifier, 0)]
            count = math.comb(fixture["rank"] + 4, 4)
            stable = full["stable"] and partial["stable"]
            if stable and any(value["count"] != count or len(value["rows"]) != count for value in (full, partial)):
                raise ValueError("Missing tensor coefficients")
            finite_mask = partial["finite_nan"] and not full["finite_nan"]
            residuals = []
            threshold = Decimal("1e-10") if fixture["ctmode"] == 1 else Decimal("1e-24")
            if stable:
                for coefficient, (a, b) in enumerate(zip(full["rows"], partial["rows"])):
                    if any(not Decimal(value).is_finite() for value in a):
                        raise ValueError("Nonfinite full tensor result")
                    if not Decimal(b[0]).is_nan() or any(not Decimal(value).is_finite() for value in b[2:]):
                        finite_mask = False
                    for slot in range(2, 6):
                        left, right = Decimal(a[slot]), Decimal(b[slot])
                        if not left.is_finite() or not right.is_finite():
                            raise ValueError("Nonfinite evaluated pole")
                        relative = abs(left - right) / max(Decimal(1), abs(left), abs(right))
                        residuals.append({"coefficient": coefficient, "component": slot,
                                          "scaled_residual": str(relative), "passed": relative <= threshold})
            passed = stable and finite_mask and all(item["passed"] for item in residuals)
            results.append({"fixture": fixture, "status": "passed" if passed else "failed",
                "stable": stable, "finite_unevaluated_mask_valid": finite_mask,
                "threshold": str(threshold), "residuals": residuals,
                "full_seconds": full["seconds"], "poles_only_seconds": partial["seconds"],
                "timing_scope": "diagnostic only; fixed full-then-poles ordering, no performance acceptance"})
    if len(frames) != 2 * len(selected):
        raise ValueError("Missing or extra diagnostic frames")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Existing exported standalone process root")
    parser.add_argument("--fixtures", type=Path, default=bench.ROOT / "vendor/ninjars/tests/poles/tensor_cases.json")
    parser.add_argument("--label", required=True)
    parser.add_argument("--loops", type=int, default=1)
    parser.add_argument("--precision", choices=("dp", "qp", "both"), default="both")
    args = parser.parse_args()
    if Path.cwd() != bench.ROOT:
        parser.error(f"Run from {bench.ROOT}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label) or args.loops < 1:
        parser.error("label must be a simple name and loops must be positive")
    output = args.output.resolve()
    copies = bench.WORK / "pole-diagnostics" / args.label
    report = bench.REPORT / "experiments/poles-diagnostics" / args.label
    copies.parent.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=False)
    try:
        original_compilation = json.loads((output / "compilation.json").read_text())
        if original_compilation.get("numerical_build_from_sources") is not True:
            raise RuntimeError("Run the benchmark build first to bind numerical object provenance")
        if bench.generated_sources(output) != original_compilation["sources"]:
            raise RuntimeError("Original process source differs from its qualified compilation")
        shutil.copytree(output, copies, symlinks=True)
        directories = sorted(copies.glob("SubProcesses/P*_*"))
        if len(directories) != 1:
            raise ValueError("Expected one generated subprocess")
        directory = directories[0]
        prefix = (directory / "proc_prefix.txt").read_text().strip()
        max_coefficients = int(re.search(r"LOOPMAXCOEFS\s*=\s*(\d+)",
            (directory / "loop_max_coefs.inc").read_text(), re.I).group(1))
        raw, selected, omitted = fixture_stream(json.loads(args.fixtures.read_text()), max_coefficients,
                                               args.loops, args.precision)
        if not selected:
            raise ValueError("No fixture fits this process rank; use a higher-rank export")
        fixture_path = report / "fixtures.dat"
        fixture_path.write_text(raw)
        protocol_dir = report / "protocol-sources"
        protocol_dir.mkdir()
        protocols = (Path(__file__), bench.ROOT / "tools/tirust_benchmark_poles.f",
                     bench.ROOT / "tools/tirust_benchmark.py", bench.ROOT / "tools/tirust_benchmark_compiler.py")
        for path in protocols:
            shutil.copy2(path, protocol_dir / path.name)
        bench.dump(report / "specification.json", {"selected": selected, "omitted": omitted,
            "fixture_source": str(args.fixtures), "fixture_source_sha256": bench.digest(args.fixtures),
            "input_sha256": bench.digest(fixture_path), "generated_sources": bench.generated_sources(output),
            "protocol_sources": {p.name: bench.digest(protocol_dir / p.name) for p in protocols}})
        driver = (bench.ROOT / "tools/tirust_benchmark_poles.f").read_text().replace("@PREFIX@", prefix)
        (directory / "tirust_pole_diagnostic.f").write_text(driver)
        (directory / "tirust_pole.mk").write_text("include makefile\n"
            "tirust-pole-check: tirust_pole_diagnostic.o $(PROCESS) $(LIBS)\n"
            "\t$(FC) $(FFLAGS) -o $@ tirust_pole_diagnostic.o $(PROCESS) $(LINKLIBS)\n")
        source_before = bench.generated_sources(copies)
        dependencies = bench.dependency_inputs(directory)
        flags = capture_effective_flags(directory, env=bench.environment())
        validate_compiler_equality(original_compilation["effective_compiler_flags"], flags)
        bench.run(["make", "-f", "tirust_pole.mk", "tirust-pole-check"], directory, report / "build-output.txt")
        if bench.generated_sources(copies) != source_before:
            raise RuntimeError("Diagnostic sources changed during compilation")
        if bench.dependency_inputs(directory) != dependencies:
            raise RuntimeError("Diagnostic dependencies changed during link")
        if not bench.same_compiler_policy(capture_effective_flags(directory, env=bench.environment()), flags):
            raise RuntimeError("Diagnostic compiler flags changed during link")
        candidates = [directory / "param_card.dat"] + [
            directory.joinpath(*([".."] * level), "Cards/param_card.dat") for level in range(6)]
        actual_card = next((path for path in candidates if path.is_file()), None)
        if actual_card is None:
            raise RuntimeError("Missing diagnostic parameter card")
        runtime = bench.runtime_identity(directory, "tirust-pole-check")
        executable_hash = bench.digest(directory / "tirust-pole-check")
        bench.dump(report / "compilation.json", {"generated_sources": source_before,
            "executable_sha256": executable_hash, "link_inputs_before": dependencies,
            "link_inputs_after": bench.dependency_inputs(directory), "effective_compiler_flags": flags,
            "runtime": runtime, "resolved_parameter_card": str(actual_card.resolve()),
            "parameter_card_sha256": bench.digest(actual_card),
            "original_compilation": original_compilation,
            "generation": json.loads((output / "generation.json").read_text())
                if (output / "generation.json").exists() else None})
        try:
            with (bench.ROOT / ".tirust-build/numerical-tests.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                bench.run(["./tirust-pole-check", str(fixture_path)], directory, report / "run-output.txt")
        finally:
            if (directory / "pole-values.dat").exists():
                shutil.copy2(directory / "pole-values.dat", report / "pole-values.dat")
        current_runtime = bench.runtime_identity(directory, "tirust-pole-check")
        if (any(current_runtime[key] != runtime[key] for key in ("libraries", "archives", "metadata"))
                or bench.digest(directory / "tirust-pole-check") != executable_hash
                or bench.generated_sources(copies) != source_before):
            raise RuntimeError("Diagnostic executable, source or dependency bytes changed during evaluation")
        results = parse_results((report / "pole-values.dat").read_text(), selected)
        bench.dump(report / "results.json", {"results": results, "omitted": omitted,
            "passed": all(item["status"] == "passed" for item in results),
            "qualified_all_requested_fixtures": not omitted and all(item["status"] == "passed" for item in results)})
        print(report)
    except BaseException as error:
        bench.dump(report / "failure.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
