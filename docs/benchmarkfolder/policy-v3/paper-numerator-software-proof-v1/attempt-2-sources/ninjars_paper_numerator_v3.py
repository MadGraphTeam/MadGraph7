#!/usr/bin/env python3
"""Private DP construction diagnostics; not physical amplitudes or gate timing."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ninjars_paper_timing_v3 as full
import ninjars_benchmark_references_v2 as references
from madgraph.various.banner import MadLoopParam

ROOT, bench, workers, policy = full.ROOT, full.bench, full.workers, full.policy
supplement, ownership, pair = full.supplement, full.ownership, full.pair
SCHEMA = "madloop-dp-construction-spec-v1"
RUN = "madloop-dp-construction-run-v1"
SUMMARY = "ninjars-paper-construction-summary-v1"
SOURCES = (Path(__file__).name, *full.SOURCES)
require, read = full.require, full.read


def skip_reduction(source):
    pattern = r"(PARAMETER\s*\(\s*SKIPLOOPEVAL\s*=\s*)\.FALSE\.(\s*\))"
    result, count = re.subn(pattern, r"\1.TRUE.\2", source, flags=re.I)
    require(count == 1, "Exactly one private skip-reduction declaration required")
    return result


def construction_driver(ordinary):
    require(ordinary.count("FORCE_STABILITY_CHECK(.TRUE.)") == 1 and
            ordinary.count("SLOOPMATRIXHEL_THRES") == 3, "Expected reviewed ordinary HEL driver")
    return ordinary.replace("FORCE_STABILITY_CHECK(.TRUE.)", "FORCE_STABILITY_CHECK(.FALSE.)")


def parse_outputs(directory, points, loops):
    result = full.parse_observation(directory, 6, points, loops)
    require(result["timed_status_counts"] == {"110": points * loops} and
            result["status_counts"] == {"110": points}, "Construction must remain direct DP110 on every call")
    require(all(row["accuracy"] == -1. for row in result["rows"]), "Direct DP diagnostics have no stability estimate")
    result.pop("physical_valid"); result.pop("physical_failures")
    return {**result, "diagnostic_complete": True, "physical_acceptance_applicable": False,
        "column_scope": "Born and skipped-reduction counterterm accumulators, not a one-loop amplitude"}


def cards(output):
    return [output / "Cards/MadLoopParams.dat", *(output / "SubProcesses").rglob("MadLoopParams.dat")]


def configure_cards(output):
    paths = cards(output)
    objects = [MadLoopParam(str(path)) for path in paths]
    records = [dict(card) for card in objects]
    normalize = lambda value: {str(k).lower(): v for k, v in value.items()}
    expected = normalize(records[0])
    require(all(normalize(record) == expected for record in records), "Effective ordinary numerical cards differ")
    for path, card in zip(paths, objects):
        if path.is_symlink():
            require(path.resolve().is_relative_to(output), "Refuse shared numerical card mutation")
            contents = path.read_bytes(); path.unlink(); path.write_bytes(contents)
        before = normalize(dict(card))
        require(str(card["MLReductionLib"]) == "6" and card["CTModeRun"] == -1, "Matched ordinary Ninja6 parent required")
        require(card["COLLIERComputeUVpoles"] and card["COLLIERComputeIRpoles"], "Full poles must stay configured")
        card["CTModeRun"] = 1; card["DoubleCheckHelicityFilter"] = False
        card.write(str(path), template=str(ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
        after = normalize(dict(MadLoopParam(str(path))))
        require(after == {**before, "ctmoderun": 1, "doublecheckhelicityfilter": False}, "Unexpected diagnostic card change")
    return expected


def filter_record(output, case):
    table = next((output / "SubProcesses/MadLoop5_resources").glob("*HelConfigs.dat"))
    require(table.read_bytes() == supplement.checked(case["helicity_table"]).read_bytes(), "Changed common helicity table")
    path = table.with_name(table.name.replace("HelConfigs", "HelFilter"))
    weights = list(map(int, path.read_text().split()))
    require(weights[case["helicity_index"] - 1] > 0, "Matched ordinary parent has no positive selected helicity")
    return supplement.bound(path)


def make_spec(parent_result, destination_report, destination_work):
    # The ordinary reader verifies all eight blocks, not a calibration probe.
    full.report_snapshot(parent_result)
    result = policy.verified(read(parent_result), full.RESULT)
    ordinary = full.validate_spec(read(supplement.checked(result["specification"])))
    report = policy.path(result["report"])
    parents = {}
    for cid, case in ordinary["cases"].items():
        output = policy.path(ordinary["work"]) / "workers" / (cid + "-candidate-6") / "outputs/candidate" / cid
        state = read(report / "states" / (cid + ".json"))["6"]
        require(bench.generated_source_manifest(output) == state["compiled_manifest"] and
                full.pair.actual_runtime(output, state["compiled_manifest"], 6) == state["runtime"], "Ordinary parent source/runtime changed")
        directory = next(output.glob("SubProcesses/P*_*"))
        require(workers.digest(directory / "check") == state["executable_sha256"], "Ordinary parent executable changed")
        parents[cid] = {"output": policy.relative(output), "state": state,
            "filter": filter_record(output, case), "build": supplement.bound(report / "builds" / (cid + "-candidate-6") / "build.json")}
    return policy.seal({"schema": SCHEMA, "ordinary_result": supplement.bound(parent_result),
        "ordinary_specification": result["specification"], "parents": parents,
        "report": policy.relative(destination_report), "work": policy.relative(destination_work),
        "cpu": 207, "repetitions": 8, "minimum_seconds": 1., "timeout": 86400.,
        "sources": {name: supplement.bound(ROOT / "tools" / name) for name in SOURCES},
        "scope": "DP construction only; no reduction, normal stability or rescue measured",
        "acceptance_eligible": False, "normal_rescue": False})


def validate_spec(spec):
    policy.verified(spec, SCHEMA)
    require(spec["cpu"] == 207 and spec["repetitions"] == 8 and spec["minimum_seconds"] == 1.
        and 0 < spec["timeout"] <= 86400. and spec["acceptance_eligible"] is False and spec["normal_rescue"] is False,
        "Changed direct diagnostic contract")
    require(set(spec["sources"]) == set(SOURCES), "Incomplete construction sources")
    for name, record in spec["sources"].items():
        require(supplement.checked(record) == ROOT / "tools" / name, "Construction producer/helper changed")
    ordinary = full.validate_spec(read(supplement.checked(spec["ordinary_specification"])))
    require(set(spec["parents"]) == set(ordinary["cases"]), "Missing construction cases")
    return ordinary


def prepare(parent, case, output, report, compile_only=False):
    """Clone a completed ordinary HEL worker; never mutate that parent."""
    original = policy.path(parent["output"])
    require(bench.generated_source_manifest(original) == parent["state"]["compiled_manifest"] and
        policy.library_hashes(original) == parent["state"]["libraries"], "Changed ordinary parent source/libraries")
    # clone_output's cp uses the same registered subprocess proxy as Worker.
    old = references.subprocess
    try:
        references.subprocess = workers.subprocess
        references.clone_output(original, output)
    finally: references.subprocess = old
    directory = next(output.glob("SubProcesses/P*_*"))
    require(workers.digest(directory / "check") == parent["state"]["executable_sha256"], "Copied ordinary executable changed")
    ordinary_cards = configure_cards(output)
    report.mkdir(parents=True, exist_ok=True)
    shutil.copy2(directory / "loop_matrix.f", report / "ordinary-loop_matrix.f")
    (directory / "loop_matrix.f").write_text(skip_reduction((directory / "loop_matrix.f").read_text()))
    (directory / "check_sa.f").write_text(construction_driver((directory / "check_sa.f").read_text()))
    require((output / "Cards/param_card.dat").read_bytes() == supplement.checked(case["origin"]["physical_card"]).read_bytes(), "Changed construction physical card")
    if not compile_only:
        source = supplement.checked(parent["filter"])
        shutil.copy2(source, output / "SubProcesses/MadLoop5_resources" / source.name)
        require(filter_record(output, case)["sha256"] == parent["filter"]["sha256"], "Changed matched filter")
    for name in ("check", "check_sa.o", "loop_matrix.o"): (directory / name).unlink(missing_ok=True)
    bench.run(["make", "-j1", "check"], directory, report / "compile.log")
    state = full.target_state(output, 6, case["origin"]["reducer_runtime"], report / "runtime-assets")
    delta = {name for name in state["compiled_manifest"]["files"] if
        state["compiled_manifest"]["files"][name] != parent["state"]["compiled_manifest"]["files"].get(name)}
    require(delta == {str((directory / name).relative_to(output)) for name in ("check_sa.f", "loop_matrix.f")}, "Unexpected construction source changes")
    metadata = {"parent": parent, "state": state, "ordinary_cards": ordinary_cards,
        "compile_only": compile_only, "ready_for_numerical": not compile_only,
        "physical_helicity": case["physical_helicity"], "helicity_index": case["helicity_index"]}
    bench.dump(report / "build.json", metadata)
    for name in ("loop_matrix.f", "check_sa.f"): shutil.copy2(directory / name, report / name)
    for path in (output / "Cards/param_card.dat", output / "Cards/MadLoopParams.dat"): shutil.copy2(path, report / path.name)
    if not compile_only: shutil.copy2(supplement.checked(parent["filter"]), report / "HelFilter.dat")
    return directory, metadata


def collect(spec_path, handoff, execute=False):
    require(execute, "Explicit construction diagnostic launch flag required")
    spec = read(spec_path); ordinary = validate_spec(spec)
    require(read(handoff) == {"root_authorized": True, "phase": RUN, "cpu": 207,
        "specification_sha256": workers.digest(Path(spec_path))}, "Missing construction resource handoff")
    require(207 in os.sched_getaffinity(0) and all(os.environ.get(k) == "1" for k in pair.THREADS), "QuietCPU207/four single-thread settings required")
    full.report_snapshot(supplement.checked(spec["ordinary_result"]))
    report, work = policy.path(spec["report"]), policy.path(spec["work"])
    require(not report.exists() and not work.exists(), "Preserve old construction attempts")
    report.mkdir(parents=True); work.mkdir(parents=True)
    shutil.copy2(spec_path, report / "input-spec.json"); shutil.copy2(handoff, report / "root-handoff.json")
    for name, source in spec["sources"].items():
        target = report / "sources" / name; target.parent.mkdir(exist_ok=True); shutil.copy2(supplement.checked(source), target)
    status = {"status": "incomplete", "requested": list(ordinary["cases"]), "completed": []}
    bench.dump(report / "status.json", status)
    try:
        for cid, case in ordinary["cases"].items():
            cell = report / "cases" / cid; cell.mkdir(parents=True)
            with ownership.owned_runtime(cell), workers.isolated_harness(work, cell):
                output = work / cid
                directory, metadata = prepare(spec["parents"][cid], case, output, cell)
                points = cell / "points.dat"; shutil.copy2(supplement.checked(case["points"]), points)
                require(metadata["ready_for_numerical"], "Compile probe cannot be measured")
                worker = None
                try:
                    worker = workers.Worker(["taskset", "-c", "207", "./check", str(points)], directory,
                        cell / "worker.log", env=ownership.profile_environment(work), timeout=spec["timeout"])
                    require(worker.points == 100, "Changed construction point stream")
                    loops = 1
                    for r in ("calibration", *range(8)):
                        try: wall = worker.run(loops)
                        finally: raw = pair.retain_raw(cell, case["definition"], ("candidate", 6), "construction", r, directory)
                        row = {**parse_outputs(directory, 100, loops), "repetition": r, "loops": loops,
                            "worker_sequence": worker.sequence, "raw_capture": raw, "command_wall_seconds": wall,
                            "points_sha256": workers.digest(points), "helicity_index": case["helicity_index"],
                            "physical_helicity": case["physical_helicity"], "state": metadata["state"], "acceptance_eligible": False}
                        bench.dump(cell / "data" / (str(r) + ".json"), row)
                        if r == "calibration": loops = max(1, math.ceil(1. / row["seconds"]))
                    require(full.target_state(output, 6, case["origin"]["reducer_runtime"], cell / "runtime-assets") == metadata["state"], "Construction source/runtime/card changed")
                finally:
                    if worker is not None:
                        with pair.defer_interruptions(): worker.close()
            status["completed"].append(cid); bench.dump(report / "status.json", status)
        status["status"] = "complete"; bench.dump(report / "status.json", status)
        inventory = {policy.relative(p): workers.digest(p) for p in report.rglob("*") if p.is_file()}
        result = policy.seal({"schema": RUN, "report": policy.relative(report), "files": inventory,
            "specification": supplement.bound(report / "input-spec.json"), "acceptance_eligible": False})
        bench.dump(report / "result.json", result)
        return report_snapshot(report / "result.json")
    except BaseException as error:
        status["status"] = "failed"; status["error"] = str(error); bench.dump(report / "status.json", status)
        raise


def remaining(total_samples, construction_samples):
    require(len(total_samples) == len(construction_samples) == 8, "Eight separate full/construction blocks required")
    require(all(math.isfinite(x) and x > 0 for x in total_samples + construction_samples), "Invalid full/construction timing")
    rng = random.Random(20261022)
    boot = sorted(statistics.median(rng.choices(total_samples, k=8)) - statistics.median(rng.choices(construction_samples, k=8)) for _ in range(10000))
    return {"median_difference_seconds_per_point": statistics.median(total_samples) - statistics.median(construction_samples),
        "confidence_interval95": [boot[249], boot[9749]], "bootstrap": "independent percentile; separate measurement sessions",
        "label": "Remaining full-evaluation work, including stability/rescue/repeated construction and other path differences; not pure reduction"}


def report_snapshot(filename, expected=None, inputs=None):
    result = policy.verified(read(filename), RUN)
    require(result["acceptance_eligible"] is False, "Construction cannot be promoted to normal timing")
    if inputs is not None: inputs[policy.relative(filename)] = workers.digest(Path(filename))
    for name, digest in result["files"].items():
        require(workers.digest(policy.path(name)) == digest, "Changed construction evidence")
        if inputs is not None: inputs[name] = digest
    spec = read(supplement.checked(result["specification"])); ordinary = validate_spec(spec)
    total = full.report_snapshot(supplement.checked(spec["ordinary_result"]), expected=expected, inputs=inputs)
    report = policy.path(result["report"]); status = read(report / "status.json")
    require(status["status"] == "complete" and status["completed"] == status["requested"] == list(ordinary["cases"]), "Incomplete construction series")
    summary = {"schema": SUMMARY, "acceptance_eligible": False, "source": supplement.bound(filename), "cases": {}}
    for cid, case in ordinary["cases"].items():
        cell = report / "cases" / cid; build = read(cell / "build.json")
        require(build["ready_for_numerical"] and not build["compile_only"] and build["parent"] == spec["parents"][cid], "Unqualified construction parent/probe")
        require(workers.digest(cell / "points.dat") == case["points"]["sha256"] and workers.digest(cell / "HelFilter.dat") == build["parent"]["filter"]["sha256"], "Changed matched points/filter")
        require((cell / "check_sa.f").read_text() == construction_driver((policy.path(read(spec["ordinary_result"]["path"])["report"]) / "builds" / (cid + "-candidate-6") / "worker.f").read_text()), "Changed construction driver")
        before = build["parent"]["state"]["compiled_manifest"]
        after = build["state"]["compiled_manifest"]
        changed = {name for name in set(before["files"]) | set(after["files"]) if before["files"].get(name) != after["files"].get(name)}
        require(len(changed) == 2 and {Path(n).name for n in changed} == {"check_sa.f", "loop_matrix.f"}, "Unexpected construction source delta")
        for name in changed:
            require(workers.digest(cell / Path(name).name) == after["files"][name]["sha256"], "Changed retained construction source")
        matrix = next(n for n in changed if Path(n).name == "loop_matrix.f")
        require(workers.digest(cell / "ordinary-loop_matrix.f") == before["files"][matrix]["sha256"] and
            (cell / "loop_matrix.f").read_text() == skip_reduction((cell / "ordinary-loop_matrix.f").read_text()), "Construction must change exactly one skip flag")
        require(after["files_sha256"] == bench.content_digest(after["files"]) and build["state"]["runtime"] == case["origin"]["reducer_runtime"], "Changed construction manifest/runtime")
        require(policy.reducer_signature({"backend": 6, "compiled_manifest": after, "libraries": build["state"]["libraries"],
            "runtime_metadata": build["state"]["runtime_metadata"]}) == build["state"]["runtime"], "Changed complete construction runtime")
        numerical = {str(k).lower(): v for k,v in dict(MadLoopParam(str(cell / "MadLoopParams.dat"))).items()}
        require(numerical == {**build["ordinary_cards"], "ctmoderun": 1, "doublecheckhelicityfilter": False} and
            workers.digest(cell / "MadLoopParams.dat") == build["state"]["madloop_card_sha256"] and
            workers.digest(cell / "param_card.dat") == build["state"]["parameter_card_sha256"] == case["origin"]["physical_card"]["sha256"], "Changed construction input cards")
        values, calibration = [], None
        for r in ("calibration", *range(8)):
            path = cell / "data" / (str(r) + ".json"); row = read(path)
            require(policy.relative(path) in result["files"] and row["repetition"] == r and row["worker_sequence"] == (1 if r == "calibration" else r+2)
                and row["state"] == build["state"] and row["helicity_index"] == case["helicity_index"] and row["physical_helicity"] == case["physical_helicity"]
                and row["points_sha256"] == case["points"]["sha256"] and row["acceptance_eligible"] is False, "Wrong diagnostic source/input/sequence")
            loops = 1 if r == "calibration" else max(1, math.ceil(1. / calibration["seconds"]))
            require(row["loops"] == loops, "Changed construction count")
            directory = cell / "raw" / (cid + "-candidate-6") / ("construction-" + str(r))
            require(set(row["raw_capture"]) == {*workers.OUTPUTS, "dimensions.dat"}, "Incomplete construction raw")
            for name, record in row["raw_capture"].items(): require(supplement.checked(record) == directory / name, "Calibration or other raw reused")
            actual = parse_outputs(directory, 100, loops)
            require(all(row[k] == v for k, v in actual.items()), "Construction raw disagrees")
            if r == "calibration": calibration = row
            else: values.append(row["seconds_per_point"])
        require({p.name for p in (cell / "data").glob("*.json")} == {str(r) + ".json" for r in ("calibration", *range(8))}, "Undeclared construction attempt")
        residual = {}
        for b in full.BACKENDS:
            paths = total["cases"][cid]["backend"][str(b)]["raw_measurements"]
            residual[str(b)] = remaining([read(supplement.checked(p))["seconds_per_point"] for p in paths], values)
        summary["cases"][cid] = {"scope": "DP construction diagnostic; not physical amplitude/stability/rescue acceptance",
            "timing_repetitions": 8, "points": 100, "physical_helicity": case["physical_helicity"],
            **full.distribution(values), "remaining": residual, "ordinary_physics_status": {str(b): total["cases"][cid]["backend"][str(b)]["status"] for b in full.BACKENDS}}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("write-spec")
    for name in ("parent-result", "report", "work", "output"): make.add_argument("--" + name, required=True, type=Path)
    run = commands.add_parser("collect"); run.add_argument("--spec", required=True, type=Path); run.add_argument("--handoff", required=True, type=Path)
    run.add_argument("--execute-numerical", action="store_true")
    check = commands.add_parser("validate"); check.add_argument("result", type=Path)
    args = parser.parse_args()
    if args.command == "write-spec":
        require(not args.output.exists(), "Preserve prior construction specs")
        bench.dump(args.output, make_spec(args.parent_result, args.report, args.work))
    elif args.command == "collect": print(json.dumps(collect(args.spec, args.handoff, args.execute_numerical), indent=2))
    else: print(json.dumps(report_snapshot(args.result), indent=2))


if __name__ == "__main__":
    def interrupted(signum, _frame): raise SystemExit("Construction interrupted by signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGHUP, interrupted)
    main()
