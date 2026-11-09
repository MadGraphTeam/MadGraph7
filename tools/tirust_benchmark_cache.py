#!/usr/bin/env python3
"""Qualify caches through an already compiled, generated MadLoop adapter.

All values are compared with uncached calls of the same precision, pole
selection and loop ordering. This is a correctness diagnostic, not a timing.
"""
import argparse
from decimal import Decimal, localcontext
import fcntl
import json
import math
from pathlib import Path
import re
import shutil

import tirust_benchmark as bench
from tirust_benchmark_compiler import capture_effective_flags, validate_compiler_equality

DEFAULT_BUDGET = 268435456
DEFAULT_FIXTURE = {"name": "infrared-triangle-r2", "n": 3, "rank": 2, "mu2": 7,
    "shifts": [[0, 0, 0, 0], [5, 0, 0, 5], [10, 0, 0, 0]],
    "mass2": [[0, 0], [0, 0], [0, 0]]}
AGGREGATE_NAMES = ["subsets_reused", "subsets_computed", "masters_reused", "masters_computed",
    "moments_reused_including_qp_hits", "moments_computed_including_qp_misses", "entries", "bytes", "not_stored"]
QP_NAMES = ["hits", "misses", "entries", "bytes", "not_stored", "bypassed"]


def actions(n):
    records, refs = [], {}

    def add(name, ctmode, mode=0, finite=1, cache=-1, budget=DEFAULT_BUDGET,
            control=0, expectation="values", reference=False):
        key = (ctmode, mode, finite)
        identifier = len(records) + 1
        if reference:
            refs[key] = identifier
        records.append(dict(id=identifier, name=name, ctmode=ctmode, mode=mode, finite=finite,
            cache=cache, budget=budget, control=control, expectation=expectation,
            reference=identifier if reference else refs[key], is_reference=reference))

    for ctmode in (1, 4):
        for mode in (0, 1, 2):
            for finite in (1, 0):
                add(f"reference-ct{ctmode}-mode{mode}-finite{finite}", ctmode, mode, finite,
                    cache=0, control=1, reference=True, expectation="bypass")
    for ctmode in (2, 5):
        for finite in (1, 0):
            add(f"reference-ct{ctmode}-finite{finite}", ctmode, finite=finite,
                cache=0, control=1, reference=True, expectation="bypass")
    add("dp-full-cold", 1, control=1, expectation="dp_miss")
    add("dp-full-hit", 1, expectation="dp_hit")
    add("dp-poles-cold", 1, finite=0, expectation="dp_miss")
    add("dp-poles-hit", 1, finite=0, expectation="dp_hit")
    for mode in (1, 2):
        # DP stores retain unselected pole data; either a hit or a miss is
        # legitimate here. Values must still match the mode-specific reference.
        add(f"dp-poles-mode{mode}", 1, mode, finite=0)
        add(f"dp-full-mode{mode}", 1, mode)
    add("dp-full-after-poles", 1, expectation="dp_hit")
    for finite in (1, 0):
        add(f"dp-check-pass-finite{finite}", 2, finite=finite, expectation="bypass")
    add("dp-cache-disabled", 1, cache=0, expectation="bypass")
    add("dp-topology-excluded", 1, cache=n-1, expectation="bypass")
    add("dp-cache-survives-bypass", 1, expectation="dp_hit")
    for finite in (1, 0):
        for mode in (0, 1, 2):
            add(f"qp-mode{mode}-finite{finite}-cold", 4, mode, finite, expectation="qp_miss")
            add(f"qp-mode{mode}-finite{finite}-hit", 4, mode, finite, expectation="qp_hit")
    add("qp-full-after-poles", 4, expectation="qp_hit")
    add("dp-cache-survives-qp", 1, expectation="dp_hit")
    for finite in (1, 0):
        add(f"qp-check-pass-finite{finite}", 5, finite=finite, expectation="bypass")
    add("qp-cache-disabled", 4, cache=0, expectation="bypass")
    add("qp-topology-excluded", 4, cache=n-1, expectation="bypass")
    add("qp-cache-survives-bypass", 4, expectation="qp_hit")
    add("qp-budget64-refusal", 4, budget=64, control=3, expectation="qp_refused")
    add("qp-budget64-repeat-refusal", 4, budget=64, expectation="qp_refused")
    add("dp-budget0-refusal", 1, budget=0, control=3, expectation="dp_refused")
    add("dp-poles-budget0-refusal", 1, finite=0, budget=0, expectation="dp_refused")
    add("dp-budget-restored", 1, control=3, expectation="dp_miss")
    add("qp-budget-restored", 4, expectation="qp_miss")
    add("dp-cache-cleared", 1, control=1, expectation="dp_miss")
    add("qp-after-clear", 4, expectation="qp_miss")
    add("dp-after-finalize", 1, control=2, expectation="dp_miss")
    add("qp-after-finalize", 4, expectation="qp_miss")
    return records


def fixture_stream(fixture, max_coefficients):
    n, rank = fixture["n"], fixture["rank"]
    if not (2 <= n <= 6 and 0 <= rank <= 6 and math.comb(rank+4, 4) <= max_coefficients
            and len(fixture["shifts"]) == n and len(fixture["mass2"]) == n and fixture["mu2"] > 0
            and all(x == 0 for x in fixture["shifts"][0])):
        raise ValueError("Cache fixture must fit the generated rank and have 2..6 propagators with zero first shift")
    records = actions(n)
    lines = [f'{n} {rank} {fixture["mu2"]} {len(records)}']
    lines.extend(" ".join(format(float(x), ".17e") for x in row) for row in fixture["shifts"])
    lines.extend(" ".join(format(float(x), ".17e") for x in row) for row in fixture["mass2"])
    lines.extend(" ".join(str(row[k]) for k in ("ctmode", "mode", "finite", "cache", "budget", "control"))
                 for row in records)
    return "\n".join(lines) + "\n", records


def parse_results(text, fixture, records):
    frames = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        identifier = int(fields[1])
        if not 1 <= identifier <= len(records):
            raise ValueError("Unknown cache record")
        record = records[identifier-1]
        if fields[0] == "F":
            if len(fields) != 12 or identifier in frames:
                raise ValueError("Malformed or duplicate cache frame")
            if list(map(int, fields[2:8])) != [record[k] for k in
                    ("ctmode", "mode", "finite", "cache", "budget", "control")]:
                raise ValueError("Cache action metadata changed")
            frames[identifier] = {"stable": fields[9] == "T", "count": int(fields[8]),
                "generation_before": int(fields[10]), "generation_after": int(fields[11]), "rows": []}
        elif fields[0] in ("B", "A"):
            if len(fields) != 17 or fields[0] in frames[identifier]:
                raise ValueError("Malformed or duplicate cache statistics")
            frames[identifier][fields[0]] = list(map(int, fields[2:]))
        elif fields[0] == "C":
            if len(fields) != 9 or int(fields[2]) != len(frames[identifier]["rows"]):
                raise ValueError("Unordered cache tensor coefficients")
            frames[identifier]["rows"].append(fields[3:])
        else:
            raise ValueError("Unknown cache diagnostic record")
    if len(frames) != len(records):
        raise ValueError("Missing cache frame")
    results = []
    with localcontext() as context:
        context.prec = 80
        last_generation = -1
        for record in records:
            frame, reference = frames[record["id"]], frames[record["reference"]]
            count = math.comb(fixture["rank"]+4, 4)
            failures = []
            if not frame["stable"] or not reference["stable"]:
                failures.append("adapter returned unstable")
            if frame["count"] != count or len(frame["rows"]) != count:
                failures.append("missing tensor coefficients")
            threshold = Decimal("1e-24" if record["ctmode"] >= 4 else "1e-10")
            maximum = Decimal(0)
            for row, truth in zip(frame["rows"], reference["rows"]):
                values, expected = list(map(Decimal, row)), list(map(Decimal, truth))
                if record["finite"] == 0:
                    if not values[0].is_nan() or values[1] != 0:
                        failures.append("unevaluated finite coefficient lacks NaN+0i mask")
                    values, expected = values[2:], expected[2:]
                for actual, target in zip(values, expected):
                    if not actual.is_finite() or not target.is_finite():
                        failures.append("evaluated tensor coefficient is nonfinite")
                    else:
                        residual = abs(actual-target)/max(Decimal(1), abs(actual), abs(target))
                        maximum = max(maximum, residual)
            if maximum > threshold:
                failures.append("cached tensor differs from same-mode uncached reference")
            before, after = frame["B"], frame["A"]
            delta = [a-b for a,b in zip(after,before)]
            qb, qa, qd = before[9:], after[9:], delta[9:]
            expectation = record["expectation"]
            if min(before+after) < 0 or after[7] > record["budget"] or qa[3] > record["budget"]//2:
                failures.append("cache byte budget exceeded or negative counter")
            if record["control"] in (1, 2, 3) and any(before):
                failures.append("reset did not empty cache statistics")
            if record["control"] == 2 and (frame["generation_before"] != -1 or
                    frame["generation_after"] <= last_generation):
                failures.append("finalize did not reinitialize a new context through the adapter")
            last_generation = frame["generation_after"]
            if record["ctmode"] < 4 and any(qd):
                failures.append("DP operation changed QP cache")
            if record["ctmode"] >= 4:
                # Aggregate counters include QP fields. Subtract them to check
                # that QP operations leave the independent DP stores intact.
                dp_delta = delta[:9]
                for index, qi in ((4,0),(5,1),(6,2),(7,3),(8,4)):
                    dp_delta[index] -= qd[qi]
                if any(dp_delta):
                    failures.append("QP operation changed DP caches")
            if expectation == "bypass":
                expected_qd = [0,0,0,0,0,1] if record["ctmode"] >= 4 else [0]*6
                if any(delta[:9]) or qd != expected_qd:
                    failures.append("disabled/topology/check-pass call changed retained cache")
            elif expectation == "dp_hit":
                if sum(delta[i] for i in (0,2,4)) <= 0 or delta[6] != 0:
                    failures.append("repeated DP call did not reuse retained entries")
            elif expectation == "dp_miss":
                if sum(delta[i] for i in (1,3,5)) <= 0 or delta[6] <= 0:
                    failures.append("cold DP call did not populate cache")
            elif expectation == "dp_refused":
                if delta[8] <= 0 or after[6] != 0 or after[7] != 0:
                    failures.append("zero-budget DP call was stored or refusal unobserved")
            elif expectation == "qp_hit":
                if qd != [1,0,0,0,0,0]:
                    failures.append("repeated QP call did not reuse exactly one tensor")
            elif expectation == "qp_miss":
                if qd[0:3] != [0,1,1] or qd[3] <= 0 or any(qd[4:]):
                    failures.append("cold QP mode did not create exactly one distinct tensor entry")
            elif expectation == "qp_refused":
                if qd != [0,1,0,0,1,0] or qa[2] != 0 or qa[3] != 0:
                    failures.append("small-budget QP call was stored or refusal unobserved")
            results.append({"action": record, "status": "failed" if failures else "passed",
                "failures": sorted(set(failures)), "maximum_scaled_residual": str(maximum),
                "threshold": str(threshold), "aggregate_before": before[:9], "aggregate_after": after[:9],
                "qp_before": qb, "qp_after": qa, "generation_before": frame["generation_before"],
                "generation_after": frame["generation_after"]})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fixture", type=Path, help="JSON object; defaults to a massless infrared triangle of rank 2")
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    if Path.cwd() != bench.ROOT or not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("Run from repository root with a simple label")
    output = args.output.resolve()
    copies = bench.WORK / "cache-diagnostics" / args.label
    report = bench.REPORT / "experiments/poles-diagnostics" / args.label
    copies.parent.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=False)
    try:
        original_compilation = json.loads((output / "compilation.json").read_text())
        if original_compilation.get("numerical_build_from_sources") is not True:
            raise RuntimeError("Run benchmark build first to bind numerical object provenance")
        if bench.generated_sources(output) != original_compilation["sources"]:
            raise RuntimeError("Original source differs from qualified compilation")
        shutil.copytree(output, copies, symlinks=True)
        directories = sorted(copies.glob("SubProcesses/P*_*"))
        if len(directories) != 1:
            raise ValueError("Expected one generated subprocess")
        directory = directories[0]
        prefix = (directory / "proc_prefix.txt").read_text().strip()
        capacity = int(re.search(r"LOOPMAXCOEFS\s*=\s*(\d+)",
            (directory / "loop_max_coefs.inc").read_text(), re.I).group(1))
        fixture = json.loads(args.fixture.read_text()) if args.fixture else DEFAULT_FIXTURE
        raw, records = fixture_stream(fixture, capacity)
        fixture_path = report / "fixtures.dat"
        fixture_path.write_text(raw)
        protocol_dir = report / "protocol-sources"
        protocol_dir.mkdir()
        protocols = [Path(__file__), bench.ROOT / "tools/tirust_benchmark_cache.f",
                     bench.ROOT / "tools/tirust_benchmark.py", bench.ROOT / "tools/tirust_benchmark_compiler.py"]
        for path in protocols:
            shutil.copy2(path, protocol_dir / path.name)
        bench.dump(report / "specification.json", {"fixture": fixture, "actions": records,
            "aggregate_counters": AGGREGATE_NAMES, "qp_counters": QP_NAMES,
            "input_sha256": bench.digest(fixture_path), "generated_sources": bench.generated_sources(output),
            "protocol_sources": {p.name: bench.digest(protocol_dir / p.name) for p in protocols},
            "scope": "generated adapter correctness diagnostics; no performance acceptance",
            "limitation": "Public statistics count caches, not finite evaluations; NaN masking and pole agreement do not alone measure skipped arithmetic."})
        driver = (bench.ROOT / "tools/tirust_benchmark_cache.f").read_text().replace("@PREFIX@", prefix)
        (directory / "tirust_cache_diagnostic.f").write_text(driver)
        (directory / "tirust_cache.mk").write_text("include makefile\n"
            "tirust-cache-check: tirust_cache_diagnostic.o $(PROCESS) $(LIBS)\n"
            "\t$(FC) $(FFLAGS) -o $@ tirust_cache_diagnostic.o $(PROCESS) $(LINKLIBS)\n")
        sources, dependencies = bench.generated_sources(copies), bench.dependency_inputs(directory)
        flags = capture_effective_flags(directory, env=bench.environment())
        validate_compiler_equality(original_compilation["effective_compiler_flags"], flags)
        bench.run(["make", "-f", "tirust_cache.mk", "tirust-cache-check"], directory, report / "build-output.txt")
        if (bench.generated_sources(copies) != sources or bench.dependency_inputs(directory) != dependencies
                or not bench.same_compiler_policy(capture_effective_flags(directory, env=bench.environment()), flags)):
            raise RuntimeError("Diagnostic source/dependency/compiler identity changed during build")
        candidates = [directory / "param_card.dat"] + [directory.joinpath(*([".."] * n), "Cards/param_card.dat") for n in range(6)]
        card = next((p for p in candidates if p.is_file()), None)
        if card is None:
            raise RuntimeError("Missing parameter card")
        runtime = bench.runtime_identity(directory, "tirust-cache-check")
        executable_hash = bench.digest(directory / "tirust-cache-check")
        bench.dump(report / "compilation.json", {"generated_sources": sources,
            "executable_sha256": executable_hash, "link_inputs_before": dependencies,
            "link_inputs_after": bench.dependency_inputs(directory), "effective_compiler_flags": flags,
            "runtime": runtime, "resolved_parameter_card": str(card.resolve()),
            "parameter_card_sha256": bench.digest(card), "original_compilation": original_compilation,
            "generation": json.loads((output / "generation.json").read_text()) if (output / "generation.json").exists() else None})
        try:
            with (bench.ROOT / ".tirust-build/numerical-tests.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                bench.run(["./tirust-cache-check", str(fixture_path)], directory, report / "run-output.txt")
        finally:
            if (directory / "cache-values.dat").exists():
                shutil.copy2(directory / "cache-values.dat", report / "cache-values.dat")
        current_runtime = bench.runtime_identity(directory, "tirust-cache-check")
        if (any(current_runtime[k] != runtime[k] for k in ("libraries", "archives", "metadata"))
                or bench.digest(directory / "tirust-cache-check") != executable_hash
                or bench.generated_sources(copies) != sources):
            raise RuntimeError("Diagnostic executable/source/dependency changed during evaluation")
        results = parse_results((report / "cache-values.dat").read_text(), fixture, records)
        bench.dump(report / "results.json", {"results": results, "passed": all(x["status"] == "passed" for x in results),
            "fixture": fixture, "scope": "correctness diagnostics only; no performance timing"})
        print(report)
        if any(x["status"] != "passed" for x in results):
            raise RuntimeError("Cache diagnostic failed; see results.json")
    except BaseException as error:
        bench.dump(report / "failure.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
