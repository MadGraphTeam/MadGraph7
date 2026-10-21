#!/usr/bin/env python3
"""Separate, rescue-enabled single-helicity four-tool diagnostic timings.

This is not the full-helicity original-Ninja/NinjaRS performance gate.  Reuse
the unchanged Worker, timer, configuration and owned-child primitives.  Failed
physics remains visible, including its raw elapsed cost; calibration never
becomes a measurement.  No reduction-only stopwatch is inferred here.
"""
import argparse
from collections import Counter
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
import ninjars_benchmark as bench
import ninjars_benchmark_workers as workers
import ninjars_benchmark_pair_v3 as pair
import ninjars_benchmark_policy_v2 as policy
import ninjars_supplement_v2 as supplement
import ninjars_profile_v2 as ownership

ROOT = bench.ROOT
SCHEMA = "madloop-paper-timing-spec-v1"
RESULT = "madloop-paper-timing-run-v1"
SUMMARY = "ninjars-paper-timing-summary-v1"
BACKENDS = (1, 6, 7, 8)
CORE = tuple(x + "-" + str(n) for n in range(1, 5) for x in ("qcd", "ew"))
LOOPS = {"supplement-gg-h": "control-gg-h", "supplement-gg-hh": "control-gg-hh"}
ORDERS = tuple(tuple(a[i:] + a[:i]) for a in (list(BACKENDS), list(reversed(BACKENDS))) for i in range(4))
SOURCES = (Path(__file__).name, *workers.SOURCE_NAMES, Path(pair.__file__).name,
           Path(policy.__file__).name, Path(supplement.__file__).name,
           Path(ownership.__file__).name, "ninjars_benchmark_references_v2.py")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def helicity_worker(source, helicity):
    """Change only the public entry point; retain timer, rows and counters."""
    require(type(helicity) is int and helicity > 0, "Positive fixed helicity index required")
    old = "CALL @PREFIX@SLOOPMATRIX_THRES(P,ANS,-1D0,ACC,CODE)"
    require(source.count(old) == 3 and source.count("      INTEGER NEXTERNAL\n") == 1,
            "Unexpected frozen worker template")
    return source.replace("      INTEGER NEXTERNAL\n", "      INTEGER NEXTERNAL,HELICITY\n"
        "      PARAMETER(HELICITY=%d)\n" % helicity).replace(old,
        "CALL @PREFIX@SLOOPMATRIXHEL_THRES(P,HELICITY,ANS,\n     $        -1D0,ACC,CODE)")


def table_selection(record, candidate_table):
    """Verify the predeclared physical choice, without evaluating amplitudes."""
    table = supplement.checked(record["baseline_helicity_table"])
    weights = list(map(int, supplement.checked(record["baseline_current_filter"]).read_text().split()))
    tuples = [tuple(map(int, line.split())) for line in table.read_text().splitlines() if line.strip()]
    require(len(weights) == len(tuples), "Helicity filter/table length mismatch")
    available = [(value, i + 1) for i, (value, weight) in enumerate(zip(tuples, weights)) if weight > 0]
    require(available, "No positive declared helicity weight")
    physical, index = min(available)
    require(index == record["selected_one_based_index"] and list(physical) == record["selected_physical_tuple"],
            "Helicity differs from deterministic shared choice")
    require(candidate_table.read_bytes() == table.read_bytes(), "Candidate physical helicity table differs")
    return index, list(physical)


def make_spec(pair_path, selection_path, controls_path, report, work):
    parent = pair.validate_spec(read(pair_path))
    selected = read(selection_path)
    require(selected.get("schema") == "metadata-only-shared-helicity-preselection-v1", "Wrong helicity selection authority")
    controls = read(controls_path)
    require(controls.get("complete") is True, "Loop-induced CLI generation is incomplete")
    config = read(supplement.checked(parent["configuration"]))
    rows = {r["case"]: r for r in selected["cases"]}
    require(set(rows) == set(CORE), "Shared helicity selection omits/injects cases")
    cases = {}
    for case in config["cases"]:
        cid = case["id"]
        output = policy.path(parent["output_roots"]["candidate"]) / cid
        table = next((output / "SubProcesses/MadLoop5_resources").glob("*HelConfigs.dat"))
        index, physical = table_selection(rows[cid], table)
        cases[cid] = {"definition": case, "output": policy.relative(output),
            "origin": parent["origins"][cid]["candidate"], "points": parent["points"][cid],
            "helicity_index": index, "physical_helicity": physical, "helicity_table": supplement.bound(table)}
    for cid, alias in LOOPS.items():
        row = controls["cases"][alias]
        require(row["status"] == "complete", "Loop-induced export is incomplete")
        output = policy.path(row["output"])
        retained = Path(controls_path).parent / "inputs/candidate" / alias
        table = next((output / "SubProcesses/MadLoop5_resources").glob("*HelConfigs.dat"))
        tuples = [list(map(int, line.split())) for line in table.read_text().splitlines() if line.strip()]
        physical = [-1, -1] + [0] * row["case"]["nfinal"]
        require(tuples.count(physical) == 1, "Missing unique all-minus scalar helicity")
        cases[cid] = {"definition": {**row["case"], "id": cid}, "actual_generation_case": alias,
            "output": policy.relative(output), "origin": {"generation": supplement.bound(retained / "generation.json"),
                "compilation": supplement.bound(retained / "compilation.json"),
                "compiled_manifest": supplement.bound(retained / "compiled-sources.json"),
                "physical_card": supplement.bound(retained / "param_card.dat"),
                "reducer_runtime": parent["candidate_runtime_signature"]},
            "points": supplement.bound(ROOT / "docs/benchmarkfolder/policy-v2/final-supplement-generation-v1/points" / (cid + ".dat")),
            "helicity_index": tuples.index(physical) + 1, "physical_helicity": physical,
            "helicity_table": supplement.bound(table)}
    return policy.seal({"schema": SCHEMA, "parent_pair_spec": supplement.bound(pair_path),
        "selection": supplement.bound(selection_path), "controls_generation": supplement.bound(controls_path),
        "candidate_cohort": parent["candidate_cohort"], "cases": cases,
        "backends": list(BACKENDS), "orders": [list(x) for x in ORDERS], "repetitions": 8,
        "cpu": 207, "minimum_seconds": 1., "timeout": 86400.,
        "report": policy.relative(report), "work": policy.relative(work),
        "sources": {name: supplement.bound(ROOT / "tools" / name) for name in SOURCES},
        "color_mode": "summed; generated initial-color normalization retained",
        "poles": {"uv": True, "ir": True}, "normal_rescue": True,
        "acceptance_eligible": False, "numerator_dp": {"status": "pending_separate_diagnostic_authority"}})


def validate_spec(spec):
    policy.verified(spec, SCHEMA)
    require(spec["backends"] == list(BACKENDS) and spec["orders"] == [list(x) for x in ORDERS]
        and spec["repetitions"] == 8 and spec["cpu"] == 207, "Changed four-tool balanced schedule")
    require(spec["poles"] == {"uv": True, "ir": True} and spec["normal_rescue"] is True
        and spec["acceptance_eligible"] is False, "Changed pole/rescue/diagnostic scope")
    require(spec["minimum_seconds"] == 1. and 0 < spec["timeout"] <= 86400., "Changed calibration or timeout")
    require(set(spec["cases"]) == set(CORE) | set(LOOPS), "Four-tool report requires all ten declared cases")
    require(set(spec["sources"]) == set(SOURCES), "Missing producer/payload source")
    for name, record in spec["sources"].items():
        require(supplement.checked(record) == ROOT / "tools" / name, "Unexpected source path")
    return spec


def preflight(spec):
    validate_spec(spec)
    parent = pair.validate_spec(read(supplement.checked(spec["parent_pair_spec"])))
    require(spec["candidate_cohort"] == parent["candidate_cohort"], "Mixed final candidate cohorts")
    correction = policy.validate_correction(supplement.checked(parent["correction"]), live=True)
    config = read(supplement.checked(parent["configuration"]))
    pair.preflight(parent, config, correction)
    proof = policy.validate_proof(supplement.checked(parent["runtime_qualification"]),
                                  supplement.checked(parent["qualification_role_policy"]))
    proven = policy.proof_runtime_signatures(proof["entries"])
    selection = {r["case"]: r for r in read(supplement.checked(spec["selection"]))["cases"]}
    controls = read(supplement.checked(spec["controls_generation"]))
    for cid, case in spec["cases"].items():
        output, origin = policy.path(case["output"]), case["origin"]
        generation, compilation, manifest = (read(supplement.checked(origin[k]))
            for k in ("generation", "compilation", "compiled_manifest"))
        identity = pair.validate_origin_records("candidate", output, generation, compilation, manifest)
        cohort = spec["candidate_cohort"]
        require(identity["revision"] == cohort["generation_revision"] and
            identity["tree_state_sha256"] == cohort["generator_identity_sha256"] and
            identity.get("dirty_patch_sha256") is None and not identity["tree_state"]["untracked"], "Changed actual CLI cohort")
        require((output / "Cards/param_card.dat").read_bytes() == supplement.checked(origin["physical_card"]).read_bytes(), "Changed physical card")
        points = supplement.checked(case["points"])
        require(int(points.read_text().splitlines()[0]) == 100, "Exactly100 retained points required")
        table = supplement.checked(case["helicity_table"])
        tuples = [list(map(int, line.split())) for line in table.read_text().splitlines() if line.strip()]
        require(type(case["helicity_index"]) is int and 1 <= case["helicity_index"] <= len(tuples), "Invalid fixed helicity index")
        require(table.is_relative_to(output) and tuples[case["helicity_index"] - 1] == case["physical_helicity"], "Changed physical helicity/index")
        if cid in CORE:
            require(case["definition"] == next(x for x in config["cases"] if x["id"] == cid), "Changed physical process definition")
            require(origin == parent["origins"][cid]["candidate"] and case["points"] == parent["points"][cid], "Changed core origin/point authority")
            require(table_selection(selection[cid], table) == (case["helicity_index"], case["physical_helicity"]), "Changed common helicity")
        else:
            row = controls["cases"][LOOPS[cid]]
            require(row["generation"] == generation and row["compilation"] == compilation and
                    policy.path(row["output"]) == output and case["definition"] == {**row["case"], "id": cid} and
                    case["physical_helicity"] == [-1, -1] + [0] * case["definition"]["nfinal"], "Changed loop-induced alias/source/helicity")
        for backend in BACKENDS:
            actual = pair.actual_runtime(output, manifest, backend)
            require(actual == origin["reducer_runtime"] == parent["candidate_runtime_signature"], "Changed full actual runtime")
            group = proven["candidate_corrected/" + str(backend)][actual["identity_sha256"]]
            require(group["signature"] == actual, "Runtime lacks exact representative qualification")
    return config


def parse_observation(directory, backend, points, loops):
    """Strict structure/counts; failed physics is data, never a successful cell."""
    require(type(loops) is int and loops > 0, "Invalid traversal count")
    timing = list(map(float, (directory / "timing.dat").read_text().split()))
    require(len(timing) == 2 and all(math.isfinite(x) and x > 0 for x in timing), "Invalid elapsed time")
    dimensions = list(map(int, (directory / "dimensions.dat").read_text().split()))
    require(len(dimensions) == 3, "Invalid dimensions")
    ndim, nso, nborn = dimensions
    require(nso >= 0 and 0 <= nborn <= nso and ndim == max(nso, nborn), "Incomplete physical dimensions")
    rows, counts, statuses, failures = [], Counter(), {}, []
    for line in (directory / "values.dat").read_text().splitlines():
        f = line.split(); require(len(f) == 8, "Malformed raw numerical row")
        row = {"point": int(f[0]), "order": int(f[1]), "status": int(f[2]),
               "coefficients": list(map(float, f[3:7])), "accuracy": float(f[7])}
        require(all(math.isfinite(x) for x in row["coefficients"] + [row["accuracy"]]), "Nonfinite raw numerical row")
        require(statuses.setdefault(row["point"], row["status"]) == row["status"], "Split-order statuses disagree")
        if row["order"] == 0: counts[str(row["status"])] += 1
        rows.append(row)
    require([(r["point"], r["order"]) for r in rows] == [(p, o) for p in range(1, points+1) for o in range(nso+1)], "Missing/duplicated/reordered physical rows")
    timed = {}
    for line in (directory / "timed_status.dat").read_text().splitlines():
        code, count = map(int, line.split())
        require(0 <= code <= 499 and count > 0 and str(code) not in timed, "Invalid status histogram")
        timed[str(code)] = count
    require(sum(timed.values()) == loops * points and set(counts) <= set(timed), "Timed status count mismatch")
    for code in sorted(set(counts) | set(timed)):
        try: bench.validate_status(int(code), backend)
        except (RuntimeError, ValueError) as error: failures.append({"status": int(code), "error": str(error)})
    if any(row["accuracy"] < 0 for row in rows): failures.append({"error": "Missing normal-mode stability estimate"})
    result = {"cold_seconds": timing[0], "seconds": timing[1], "seconds_per_point": timing[1]/(loops*points),
        "rows": rows, "status_counts": dict(counts), "timed_status_counts": timed,
        "answer_dimensions": {"allocated": ndim, "loop_orders": nso, "born_orders": nborn},
        "physical_valid": not failures, "physical_failures": failures}
    if not failures:
        require(workers.parse_outputs(directory, backend, points, loops) ==
            {k: v for k, v in result.items() if k not in ("physical_valid", "physical_failures")}, "Ordinary parser disagreement")
    return result


def prepare_target(case, backend, config, work, report):
    definition = case["definition"]
    original = policy.path(case["output"])
    with workers.isolated_harness(work, report), supplement.input_roots({"candidate": policy.relative(original.parent)}, work,
            {definition["id"]: original.name}):
        job, directory, metadata = workers.prepare("candidate", definition, backend, config, work / "workers", report, clean=True)
    retained = report / "builds" / (definition["id"] + "-candidate-" + str(backend))
    for name in ("build.json", "compiled-sources.json", "worker.f"):
        shutil.copy2(retained / name, retained / ("ordinary-" + name))
    # Render the reviewed template independently of the ordinary prepared driver.
    old = (ROOT / "tools/ninjars_benchmark_worker.f").read_text()
    source = helicity_worker(old, case["helicity_index"])
    rendered = (directory / "check_sa.f").read_text()
    prefix = re.search(r"CALL\s+(\w*)GET_ANSWER_DIMENSION", rendered, re.I).group(1)
    nborn = re.search(r"PARAMETER\(NSQSO_BORN=(\d+)\)", rendered).group(1)
    source = source.replace("@PREFIX@", prefix).replace("@NEXTERNAL@", str(definition["nfinal"]+2)).replace("@NBORN@", nborn)
    (directory / "check_sa.f").write_text(source)
    (directory / "check_sa.o").unlink(missing_ok=True)
    (directory / "check").unlink(missing_ok=True)
    relink_seconds = bench.run(["make", "-j1", "check"], directory, report / "logs" / (definition["id"] + "-hel-" + str(backend) + ".log"))
    output = job / "outputs/candidate" / definition["id"]
    manifest = bench.generated_source_manifest(output)
    require(pair.actual_runtime(output, manifest, backend) == case["origin"]["reducer_runtime"], "Private helicity build changed runtime")
    metadata = {**metadata, "diagnostic_scheme": SCHEMA, "helicity_index": case["helicity_index"],
        "ordinary_worker_compilation_seconds": metadata["compilation_seconds"],
        "helicity_driver_relink_seconds": relink_seconds,
        "compilation_seconds": metadata["compilation_seconds"] + relink_seconds,
        "physical_helicity": case["physical_helicity"], "driver_sha256": workers.digest(directory / "check_sa.f"),
        "executable_sha256": workers.digest(directory / "check"), "generated_sources_sha256": manifest["files_sha256"]}
    bench.dump(retained / "build.json", metadata); bench.dump(retained / "compiled-sources.json", manifest)
    shutil.copy2(directory / "check_sa.f", retained / "worker.f")
    return job, directory, metadata


def target_state(output, backend, expected, assets):
    manifest = bench.generated_source_manifest(output)
    runtime = pair.actual_runtime(output, manifest, backend)
    require(runtime == expected, "Runtime changed during diagnostic")
    directory = next(output.glob("SubProcesses/P*_*"))
    card = output / "Cards/param_card.dat"
    workers.validate_param_card(directory, card.read_bytes())
    numerical = output / "Cards/MadLoopParams.dat"
    require(workers.digest(directory / "MadLoop5_resources/MadLoopParams.dat") == workers.digest(numerical), "Effective numerical card differs")
    metadata = {}
    assets.mkdir(exist_ok=True)
    for name, path in policy.runtime_metadata_files(output).items():
        destination = assets / workers.digest(path)
        if not destination.exists(): shutil.copy2(path, destination)
        require(workers.digest(destination) == workers.digest(path), "Changed retained runtime interface")
        metadata[name] = supplement.bound(destination)
    return {"compiled_manifest": manifest, "libraries": policy.library_hashes(output), "runtime": runtime,
        "runtime_metadata": metadata,
        "executable_sha256": workers.digest(directory / "check"), "parameter_card_sha256": workers.digest(card),
        "madloop_card_sha256": workers.digest(numerical)}


def run_case(case, config, spec, work, report, dispatch):
    cid = case["definition"]["id"]
    points = report / "points" / (cid + ".dat")
    jobs, instances, before, calibrations = {}, {}, {}, {}
    current, phase, repetition = None, "prepare", None
    try:
        for backend in BACKENDS:
            current = backend
            dispatch.event(case["definition"], phase, "start", ("candidate", backend))
            jobs[backend] = prepare_target(case, backend, config, work, report)
            dispatch.event(case["definition"], phase, "complete", ("candidate", backend))
        def observe(backend, loops, phase, repetition):
            worker = instances[backend]
            index = dispatch.event(case["definition"], phase, "start", ("candidate", backend), repetition, loops=loops, worker_sequence=worker.sequence + 1)
            try: wall = worker.run(loops)
            finally: raw = pair.retain_raw(report, case["definition"], ("candidate", backend), phase, repetition, worker.directory)
            data = parse_observation(worker.directory, backend, 100, loops)
            require(data["answer_dimensions"] == {"allocated": worker.ndim, "loop_orders": worker.nso, "born_orders": worker.nborn}, "READY dimensions changed")
            metadata = jobs[backend][2]
            result = {**metadata, **data, "case": cid, "backend": backend, "source": "candidate", "cpu": spec["cpu"],
                "points_sha256": workers.digest(points), "helicity_index": case["helicity_index"], "physical_helicity": case["physical_helicity"],
                "repetition": repetition, "phase": phase, "loops": loops, "worker_sequence": worker.sequence,
                "dispatch_index": index, "raw_capture": raw, "command_wall_seconds": wall, "acceptance_eligible": False}
            if phase == "measurement" and result["physical_valid"] and calibrations[backend]["physical_valid"]:
                bench.validate_repetition(result, calibrations[backend], config)
            bench.dump(report / "data" / (cid + "-" + str(backend) + "-" + str(repetition) + ".json"), result)
            dispatch.event(case["definition"], phase, "complete", ("candidate", backend), repetition, loops=loops, physical_valid=result["physical_valid"])
            return result
        for backend in BACKENDS:
            current, phase = backend, "initialize"
            job, directory, metadata = jobs[backend]
            output = job / "outputs/candidate" / cid
            before[backend] = target_state(output, backend, case["origin"]["reducer_runtime"], report / "runtime-assets")
            workers.validate_param_card(directory, supplement.checked(case["origin"]["physical_card"]).read_bytes())
            dispatch.event(case["definition"], phase, "start", ("candidate", backend))
            instances[backend] = workers.Worker(["taskset", "-c", str(spec["cpu"]), "./check", str(points)], directory,
                report / "logs" / (cid + "-" + str(backend) + ".log"), env=ownership.profile_environment(job), timeout=spec["timeout"])
            require(instances[backend].points == 100, "Changed fixed stream length")
            dispatch.event(case["definition"], phase, "complete", ("candidate", backend), worker_sequence=instances[backend].sequence)
            phase, repetition = "calibration", "calibration"
            calibrations[backend] = observe(backend, 1, phase, repetition)
        settings = [supplement.comparable_numerical_settings(x) for x in calibrations.values()]
        require(all(x == settings[0] for x in settings), "Four tools have unequal numerical settings")
        loops = max(1, *(math.ceil(spec["minimum_seconds"] / x["seconds"]) for x in calibrations.values()))
        bench.dump(report / "schedules" / (cid + ".json"), {"common_loops": loops, "orders": spec["orders"], "calibrations_are_measurements": False})
        for repetition, order in enumerate(spec["orders"]):
            for backend in order:
                current, phase = backend, "measurement"
                observe(backend, loops, phase, repetition)
        for backend, (job, _, _) in jobs.items():
            require(target_state(job / "outputs/candidate" / cid, backend, case["origin"]["reducer_runtime"], report / "runtime-assets") == before[backend], "Source/library/runtime changed during timing")
        bench.dump(report / "states" / (cid + ".json"), {str(k): v for k, v in before.items()})
    except BaseException as error:
        bench.dump(report / "failures" / (cid + ".json"), {"case": cid, "backend": current, "phase": phase,
            "repetition": repetition, "error": str(error), "type": type(error).__name__, "acceptance_eligible": False})
        raise
    finally:
        errors = []
        with pair.defer_interruptions():
            for backend, worker in reversed(list(instances.items())):
                try: worker.close()
                except BaseException as error: errors.append(str(error))
        if errors:
            bench.dump(report / "failures" / (cid + "-cleanup.json"), {"errors": errors})
            if sys.exc_info()[0] is None: raise RuntimeError("Worker cleanup failed: " + "; ".join(errors))


def collect(spec_path, handoff_path, execute=False):
    require(execute, "Explicit numerical launch flag required")
    spec = validate_spec(read(spec_path))
    require(read(handoff_path) == {"root_authorized": True, "phase": RESULT, "cpu": 207,
        "specification_sha256": workers.digest(Path(spec_path))}, "Missing exact root quiet-resource handoff")
    require(207 in os.sched_getaffinity(0) and all(os.environ.get(k) == "1" for k in pair.THREADS), "CPU207/four single-thread settings required")
    config = preflight(spec)
    config = {**config, "cases": [x["definition"] for x in spec["cases"].values()]}
    report, work = policy.path(spec["report"]), policy.path(spec["work"])
    require(not report.exists() and not work.exists(), "New attempt namespace required; no automatic retries")
    report.mkdir(parents=True); work.mkdir(parents=True)
    (report / "points").mkdir(); (report / "sources").mkdir()
    for name, record in spec["sources"].items(): shutil.copy2(supplement.checked(record), report / "sources" / name)
    shutil.copy2(spec_path, report / "input-spec.json"); shutil.copy2(handoff_path, report / "root-handoff.json")
    for cid, case in spec["cases"].items(): shutil.copy2(supplement.checked(case["points"]), report / "points" / (cid + ".dat"))
    protocol = {"schema": RESULT, "specification": supplement.bound(report / "input-spec.json"), "configuration": config,
        "thread_environment": {k: os.environ[k] for k in pair.THREADS}, "normal_rescue": True, "acceptance_eligible": False,
        "timer": "unchanged Fortran Worker full selected-helicity evaluation, including stability/rescue and final-traversal copying",
        "numerator_dp": spec["numerator_dp"], "remaining": {"status": "pending; not a pure-reduction timer"}}
    bench.dump(report / "protocol.json", protocol)
    status = {"schema": RESULT, "status": "incomplete", "requested_cases": list(spec["cases"]), "completed_cases": []}
    bench.dump(report / "run-status.json", status)
    dispatch = pair.Dispatch(report)
    try:
        for cid, case in spec["cases"].items():
            with ownership.owned_runtime(report), workers.isolated_harness(work, report):
                run_case(case, config, spec, work, report, dispatch)
            status["completed_cases"].append(cid); bench.dump(report / "run-status.json", status)
        preflight(spec)
        status["status"] = "complete"; bench.dump(report / "run-status.json", status)
        inventory = {policy.relative(p): workers.digest(p) for p in report.rglob("*") if p.is_file()}
        result = policy.seal({"schema": RESULT, "report": policy.relative(report), "run_files_sha256": inventory,
            "acceptance_eligible": False, "status": "complete", "specification": supplement.bound(report / "input-spec.json")})
        bench.dump(report / "result.json", result)
        return report_snapshot(report / "result.json")
    except BaseException:
        status["status"] = "failed"; bench.dump(report / "run-status.json", status)
        raise


def distribution(values):
    require(len(values) == 8 and all(math.isfinite(x) and x > 0 for x in values), "Eight actual timing repetitions required")
    rng = random.Random(20261021)
    boots = sorted(statistics.median(rng.choices(values, k=8)) for _ in range(10000))
    return {"median_seconds_per_point": statistics.median(values), "confidence_interval95": [boots[249], boots[9749]],
        "dispersion": {"minimum": min(values), "maximum": max(values), "median_absolute_deviation": statistics.median(abs(x-statistics.median(values)) for x in values)}}


def check_build(case, backend, report, state):
    """Bind the private executable/card/source to exactly one driver change."""
    cid = case["definition"]["id"]
    retained = report / "builds" / (cid + "-candidate-" + str(backend))
    build, compiled = read(retained / "build.json"), read(retained / "compiled-sources.json")
    original = read(supplement.checked(case["origin"]["compiled_manifest"]))
    changed = {name for name in set(compiled["files"]) | set(original["files"])
               if compiled["files"].get(name) != original["files"].get(name)}
    require(len(changed) == 1 and next(iter(changed)).endswith("/check_sa.f"), "Private build changes more than the driver")
    worker = (retained / "worker.f").read_text()
    prefix = re.search(r"CALL\s+(\w*)GET_ANSWER_DIMENSION", worker, re.I).group(1)
    nborn = re.search(r"PARAMETER\(NSQSO_BORN=(\d+)\)", worker).group(1)
    expected = helicity_worker((ROOT / "tools/ninjars_benchmark_worker.f").read_text(), case["helicity_index"])
    expected = expected.replace("@PREFIX@", prefix).replace("@NEXTERNAL@", str(case["definition"]["nfinal"]+2)).replace("@NBORN@", nborn)
    require(worker == expected and compiled["files"][next(iter(changed))]["sha256"] == workers.digest(retained / "worker.f")
        and build["driver_sha256"] == workers.digest(retained / "worker.f"), "Private driver differs from reviewed HEL-only transformation")
    require(state["compiled_manifest"] == compiled and state["runtime"] == case["origin"]["reducer_runtime"] and
        compiled["files_sha256"] == bench.content_digest(compiled["files"]), "Changed source/runtime snapshot")
    require(policy.reducer_signature({"backend": backend, "compiled_manifest": compiled,
        "libraries": state["libraries"], "runtime_metadata": state["runtime_metadata"]}) == state["runtime"], "Incomplete full runtime descriptor")
    for name, field in (("param_card.dat", "parameter_card_sha256"), ("MadLoopParams.dat", "madloop_card_sha256")):
        require(workers.digest(retained / name) == build[field] == state[field], "Changed retained input card")
    require(build["executable_sha256"] == state["executable_sha256"] and build["generated_sources_sha256"] == compiled["files_sha256"], "Changed executable/source build identity")
    require(build["parameter_card_sha256"] == case["origin"]["physical_card"]["sha256"], "Retained physical card differs from origin")
    return build


def report_snapshot(filename, expected=None, inputs=None):
    """Reparse bound raw blocks; no calibration-only or failed-physics promotion."""
    result = policy.verified(read(filename), RESULT)
    require(result["status"] == "complete" and result["acceptance_eligible"] is False, "Incomplete/incorrectly promoted diagnostic")
    if inputs is not None: inputs[policy.relative(filename)] = workers.digest(Path(filename))
    for name, digest in result["run_files_sha256"].items():
        require(workers.digest(policy.path(name)) == digest, "Changed diagnostic run evidence")
        if inputs is not None: inputs[name] = digest
    report = policy.path(result["report"])
    spec = validate_spec(read(supplement.checked(result["specification"])))
    if expected is not None: require(spec["candidate_cohort"]["runtime_identity_sha256"] == expected, "Wrong expected runtime")
    status = read(report / "run-status.json")
    require(status["status"] == "complete" and status["completed_cases"] == status["requested_cases"] == list(spec["cases"]), "Missing diagnostic cases")
    protocol = read(report / "protocol.json")
    require(protocol["schema"] == RESULT and protocol["specification"] == result["specification"] and
        protocol["thread_environment"] == {k: "1" for k in pair.THREADS} and
        protocol["normal_rescue"] is True and protocol["acceptance_eligible"] is False, "Changed diagnostic protocol/thread policy")
    for name, source in spec["sources"].items():
        require(workers.digest(report / "sources" / name) == source["sha256"], "Retained producer/payload changed")
    dispatch = [json.loads(line) for line in (report / "dispatch.jsonl").read_text().splitlines()]
    require([x["index"] for x in dispatch] == list(range(len(dispatch))), "Broken dispatch ledger")
    summary = {"schema": SUMMARY, "acceptance_eligible": False, "source": supplement.bound(filename),
        "specification": result["specification"], "candidate_cohort": spec["candidate_cohort"], "cases": {},
        "numerator_dp": spec["numerator_dp"], "remaining": {"status": "pending_separate_DP_construction_measurement", "label": "remaining full-evaluation work includes stability/rescue; not pure reduction"}}
    for cid, case in spec["cases"].items():
        schedule = read(report / "schedules" / (cid + ".json"))
        require(workers.digest(report / "points" / (cid + ".dat")) == case["points"]["sha256"], "Changed retained point stream")
        require(schedule["orders"] == spec["orders"] and schedule["calibrations_are_measurements"] is False and type(schedule["common_loops"]) is int and schedule["common_loops"] > 0, "Changed common sample schedule")
        cell = {"points": 100, "helicity": {"index": case["helicity_index"], "physical_tuple": case["physical_helicity"]},
            "color_mode": spec["color_mode"], "poles": spec["poles"], "normal_rescue": True, "backend": {}}
        valid_groups, calibration_times, comparable_settings = {}, [], []
        states = read(report / "states" / (cid + ".json"))
        starts = [x for x in dispatch if x["case"] == cid and x["event"] == "start" and x["phase"] in ("calibration", "measurement")]
        required = [("calibration", "calibration", b, 1) for b in BACKENDS] + [("measurement", r, b, schedule["common_loops"]) for r, order in enumerate(ORDERS) for b in order]
        require([(x["phase"], x["repetition"], x["backend"], x["loops"]) for x in starts] == required, "Changed all-four actual measurement dispatch order")
        for backend in BACKENDS:
            build = check_build(case, backend, report, states[str(backend)])
            observations = []
            for repetition in ("calibration", *range(8)):
                path = report / "data" / (cid + "-" + str(backend) + "-" + str(repetition) + ".json")
                require(policy.relative(path) in result["run_files_sha256"], "Unbound measurement/calibration")
                row = read(path); loops = 1 if repetition == "calibration" else schedule["common_loops"]
                require(row["case"] == cid and row["source"] == "candidate" and row["backend"] == backend and row["repetition"] == repetition and
                    row["phase"] == ("calibration" if repetition == "calibration" else "measurement") and row["loops"] == loops and
                    row["helicity_index"] == case["helicity_index"] and row["physical_helicity"] == case["physical_helicity"] and
                    row["points_sha256"] == case["points"]["sha256"] and row["cpu"] == 207 and row["acceptance_eligible"] is False,
                    "Wrong source/input/helicity/phase/counting contract")
                for key in ("parameter_card_sha256", "madloop_card_sha256", "executable_sha256", "generated_sources_sha256", "driver_sha256", "shared_numerical_settings"):
                    require(row[key] == build[key], "Measurement build/card/settings identity changed")
                supplement.comparable_numerical_settings(row)
                event = dispatch[row["dispatch_index"]]
                require(event in starts and event["backend"] == backend and event["repetition"] == repetition and
                    event["worker_sequence"] == row["worker_sequence"] and event["loops"] == loops, "Observation not bound to actual dispatch")
                require(row["worker_sequence"] == (1 if repetition == "calibration" else repetition + 2), "Unexpected extra/replacement worker command")
                completions = [x for x in dispatch if x["case"] == cid and x["phase"] == row["phase"] and
                    x["event"] == "complete" and x["backend"] == backend and x["repetition"] == repetition]
                require(len(completions) == 1 and completions[0]["physical_valid"] == row["physical_valid"], "Missing/contradictory command completion")
                require(set(row["raw_capture"]) == {*workers.OUTPUTS, "dimensions.dat"}, "Incomplete raw command capture")
                expected_directory = report / "raw" / (cid + "-candidate-" + str(backend)) / (row["phase"] + "-" + str(repetition))
                for name, record in row["raw_capture"].items():
                    require(supplement.checked(record) == expected_directory / name and
                        policy.relative(expected_directory / name) in result["run_files_sha256"], "Calibration/reused/unbound raw block cannot become a measurement")
                directory = supplement.checked(row["raw_capture"]["timing.dat"]).parent
                for raw in row["raw_capture"].values(): supplement.checked(raw)
                raw = parse_observation(directory, backend, 100, loops)
                require(all(row[key] == value for key, value in raw.items()), "Raw observation differs from retained summary")
                observations.append((path, row))
            measured = observations[1:]
            calibration_times.append(observations[0][1]["seconds"])
            comparable_settings.append(supplement.comparable_numerical_settings(observations[0][1]))
            for _, row in measured:
                if row["physical_valid"] and observations[0][1]["physical_valid"]:
                    bench.validate_repetition(row, observations[0][1], protocol["configuration"])
            counts = Counter()
            for _, row in measured: counts.update(row["timed_status_counts"])
            good = all(row["physical_valid"] for _, row in observations)
            totals = {"dp": 0, "qp": 0, "failed": 0}
            for code, count in counts.items():
                try: bench.validate_status(int(code), backend)
                except (ValueError, RuntimeError): totals["failed"] += count
                else: totals["dp" if int(code)//100 == 2 else "qp"] += count
            cell["backend"][str(backend)] = {"status": "valid_observed_physics" if good else "failed_physics",
                "calibration": supplement.bound(observations[0][0]), "timing_repetitions": 8,
                **distribution([row["seconds_per_point"] for _, row in measured]), "timed_status_counts": dict(counts), **totals,
                "evaluations": 8 * schedule["common_loops"] * 100, "raw_measurements": [supplement.bound(path) for path, _ in measured]}
            if good: valid_groups[str(backend)] = [dict(row, point=row["point"] + repetition*100)
                for repetition, (_, measurement) in enumerate(measured) for row in measurement["rows"]]
        cell["comparison_of_valid_backends"] = policy.compare_rows(valid_groups) if valid_groups else {"passed": False, "pairs": 0, "failures": []}
        cell["all_four_physics_valid"] = len(valid_groups) == 4 and cell["comparison_of_valid_backends"]["passed"]
        require(schedule["common_loops"] == max(1, *(math.ceil(spec["minimum_seconds"] / x) for x in calibration_times)), "Common loop count differs from actual calibrations")
        require(all(x == comparable_settings[0] for x in comparable_settings), "Different four-tool numerical settings")
        summary["cases"][cid] = cell
    expected_data = {cid + "-" + str(b) + "-" + str(r) + ".json" for cid in spec["cases"] for b in BACKENDS for r in ("calibration", *range(8))}
    require({p.name for p in (report / "data").glob("*.json")} == expected_data, "Undeclared, replaced or missing raw timing attempt")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("write-spec")
    for name in ("pair-spec", "selection", "controls", "report", "work", "output"): make.add_argument("--" + name, required=True, type=Path)
    run = commands.add_parser("collect")
    run.add_argument("--spec", required=True, type=Path); run.add_argument("--handoff", required=True, type=Path)
    run.add_argument("--execute-numerical", action="store_true")
    check = commands.add_parser("validate"); check.add_argument("result", type=Path)
    args = parser.parse_args()
    if args.command == "write-spec":
        require(not args.output.exists(), "Preserve old specifications")
        spec = make_spec(args.pair_spec, args.selection, args.controls, args.report, args.work)
        preflight(spec); bench.dump(args.output, spec); print(json.dumps(supplement.bound(args.output)))
    elif args.command == "collect": print(json.dumps(collect(args.spec, args.handoff, args.execute_numerical), indent=2))
    else: print(json.dumps(report_snapshot(args.result), indent=2))


if __name__ == "__main__":
    def interrupted(signum, _frame): raise SystemExit("Paper timing interrupted by signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGHUP, interrupted)
    main()
