#!/usr/bin/env python3
"""Held wrapper around the real MG7 profile command; no replacement timer."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "bin/madgraph").is_file() and (p / "madgraph").is_dir())
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark as bench
import ninjars_benchmark_workers as workers
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_pair_v3 as pair
import ninjars_profile_v2 as ownership
from madgraph.various.banner import MadLoopParam

PHASE = "mg7-profile-cli-ninja-nrs-v1"
PRODUCER_ROLE = "ninjars-mg7-profile-cli-v3"
LABELS = {6: "Ninja", 8: "NinjaRS"}
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def require(value, message):
    if not value: raise ValueError(message)


def bound(path):
    path = policy.path(path)
    return {"path": policy.relative(path), "sha256": workers.digest(path), "bytes": path.stat().st_size}


def read(path):
    return json.loads(Path(path).read_text())


def checked(record):
    path = policy.path(record["path"])
    require(bound(path) == record, "Changed profile CLI input")
    return path


def rounded(value):
    # output_timings uses %.3g. Preserve its quantization rather than inventing precision.
    uncertainty = 0. if value == 0 else .5 * 10. ** (math.floor(math.log10(abs(value))) - 2)
    return {"milliseconds": value, "printed_rounding_bound_ms": uncertainty}


def declared_scope(spec):
    repetitions = spec.get("repetitions", 7)
    points = spec.get("initial_stability_points", 100)
    require(type(repetitions) is int and repetitions >= 3,
        "Declared CLI repetitions must be an integer of at least3")
    require(type(points) is int and points in (20, 100),
        "Declared initial CLI stability sample must be20 or100")
    return {"repetitions": repetitions, "initial_stability_points": points}


def check_declared_item(item, scope):
    repetition = item["repetition"]
    require(item["backend"] in LABELS and type(repetition) is int and
        0 <= repetition < scope["repetitions"] and type(item["npoints"]) is int and
        item["npoints"] == (scope["initial_stability_points"] if repetition == 0 else 0),
        "Changed CLI tool/repetition/cached-sample schedule")


def parse_profile(text, backend):
    require(backend in LABELS, "Only Ninja6/NinjaRS8 are in this study")
    labels = re.findall(r"^= Reduction tool tested\.+\s+(.+?)\s*$", text, re.M)
    require(labels == [LABELS[backend]], "Wrong, missing or repeated CLI backend label")
    result = {"schema": "mg7-profile-cli-stage-timing-v1", "backend": backend,
        "acceptance_eligible": False, "timed_status_counts_available": False,
        "scope": "Stock direct-DP MG7 profile; process wall time minus boot wall time per adaptive PS count",
        "reduction_boundary": "MG7 CLI full-minus-SKIPLOOPEVAL estimate; masters/callbacks/Laurent assembly included; no direct reduction stopwatch"}
    for heading, key in (("Helicity sum", "helicity_sum"), ("One helicity", "single_helicity")):
        pattern = (r"^= " + heading + r" time / PSpoint =+\s+(" + NUMBER + r")ms\s*\n"
            r"\|= Coefs\. computation time\.+\s+(" + NUMBER + r")ms \([-+]?\d+%\)\s*\n"
            r"\|= Loop evaluation time\.+\s+(" + NUMBER + r")ms \([-+]?\d+%\)")
        matches = re.findall(pattern, text, re.M)
        require(len(matches) == 1, "Missing, repeated or incompatible CLI timing section: " + heading)
        values = list(map(float, matches[0]))
        require(all(math.isfinite(x) for x in values) and values[0] > 0, "Nonfinite/nonpositive CLI total")
        records = list(map(rounded, values))
        require(abs(values[0] - values[1] - values[2]) <= sum(r["printed_rounding_bound_ms"] for r in records) + 1e-12,
            "CLI timing partition disagrees beyond printed rounding")
        result[key] = dict(zip(("total", "construction", "reduction_estimate"), records))
    counts = re.findall(r"^\|= Number of loop groups\.+\s+(\S+)\s*$", text, re.M)
    require(len(counts) == 1, "Missing or repeated loop-group count")
    result["reducible_loop_groups"] = None if counts[0] == "NA" else int(counts[0])
    require(result["reducible_loop_groups"] is None or result["reducible_loop_groups"] >= 0, "Invalid loop groups")
    return result


def command_card(case, backend, npoints, verify_inputs=True, source_root=None):
    require(backend in LABELS and type(npoints) is int and npoints in (0, 20, 100),
        "Only declared two-tool/first20-or100-or-cached0 profile scope")
    require(all("\n" not in str(case[key]) for key in ("model", "process")), "Invalid command-card definition")
    def location(record):
        if verify_inputs: return checked(record)
        path = Path(record["path"])
        require(not path.is_absolute() and ".." not in path.parts, "Invalid declared CLI input path")
        return Path(source_root or ROOT) / path
    physical = location(case["physical_card"])
    return "\n".join(("set automatic_html_opening False", "set auto_update 0", "set nb_core 1",
        "set loop_optimized_output True", "set output_dependencies external", "set complex_mass_scheme False",
        "set ninja " + str(location(case["ninja_archive"]).parent),
        "set ninjars " + str(location(case["ninjars_archive"]).parent),
        "import model " + str(case["model"]),
        "check profile -reuse %d %s %s --reduction=%d --seed=%d --energy=1000 --split_orders=-1" %
            (npoints, physical, case["process"], backend, case["seed"]), "quit", ""))


class PrimitivePickle(pickle.Unpickler):
    def find_class(self, module, name):
        raise ValueError("Stability cache must contain primitive data only")


def cached_stability(path, expected_points=100):
    require(type(expected_points) is int and expected_points in (20, 100),
        "Expected CLI stability sample must be20 or100")
    data = PrimitivePickle(io.BytesIO(Path(path).read_bytes())).load()
    require(isinstance(data, dict) and isinstance(data.get("DP_stability"), list) and
        len(data["DP_stability"]) == expected_points and isinstance(data.get("QP_stability"), list) and
        len(data["QP_stability"]) == expected_points,
        "CLI cache is not the declared first%d DP/QP stability sample" % expected_points)
    return {"cache": bound(path), "dp_points": expected_points,
        "qp_points": sum(bool(x) for x in data["QP_stability"]),
        "scope": "CLI self-consistency sample; ordinary momenta not retained by stock pickle; not matched-point CDF authority"}


def same_stage_scope(left, right):
    keys = ("physical_card_sha256", "timing_energy_source", "helicity_index", "physical_helicity",
        "helicity_table_sha256", "helicity_filter_sha256", "selected_helicity_weight", "loop_optimized_output",
        "direct_dp", "common_madloop_settings", "compiler_settings_sha256", "actual_split_orders_target")
    return all(key in left and key in right and left[key] is not None and left[key] == right[key] for key in keys)


def metadata_text(record):
    data = record["text"].encode("utf-8")
    require(len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"], "Changed timing metadata text custody")
    return record["text"]


def metadata_runtime(metadata, stage):
    if "runtime_files" in stage: return stage["runtime_files"]
    require(stage.get("runtime_files_reference") == "runtime_files", "Unknown static timing-runtime reference")
    return metadata["runtime_files"]


def stage_sources(metadata, stage, snapshot_root=None):
    sources = {}
    for name, field in (("check_sa.f", "check_sa"), ("loop_matrix.f", "loop_matrix")):
        record = stage["source_snapshots"][name]
        original = Path(record["path"])
        relative = original.relative_to(Path(metadata["export_dir"]) / "SavedRunInfos")
        require(".." not in relative.parts, "Invalid timing snapshot path")
        path = (Path(snapshot_root) / relative) if snapshot_root is not None else original
        data = path.read_bytes()
        require(len(data) == record["bytes"] and hashlib.sha256(data).hexdigest() == record["sha256"] == stage[field]["sha256"],
            "Changed timing stage source snapshot")
        sources[name] = data.decode()
    driver = sources["check_sa.f"]
    flat = "\n".join(line[6:] for line in driver.splitlines() if len(line) > 6 and line[0] not in "cC*!")
    count = re.search(r"NPSPOINTS\s*=\s*(\d+)", flat, re.I)
    readps = re.search(r"READPS\s*=\s*(\.TRUE\.|\.FALSE\.)", flat, re.I)
    helicity = re.search(r"SLOOPMATRIXHEL_THRES\s*\(\s*P\s*,\s*([+-]?\d+)", flat, re.I)
    require(count and int(count[1]) == stage["actual_npoints"] and readps and (readps[1].upper() == ".TRUE.") == stage["read_ps"] and
        (int(helicity[1]) if helicity else -1) == stage["helicity_one_based_index"], "Timing driver disagrees with stage metadata")
    targets = [int(value) for value in re.findall(r"SET_COUPLINGORDERS_TARGET\s*\(\s*([+-]?\d+)\s*\)", flat, re.I)]
    require(all(value == -1 for value in targets) and (targets[0] if targets else None) == stage.get("split_orders_target"),
        "Timing driver changed default/all coupling-order target")
    for flag in ("SKIPLOOPEVAL", "BOOTANDSTOP"):
        values = re.findall(r"\b" + flag + r"\s*=\s*(\.TRUE\.|\.FALSE\.)", sources["loop_matrix.f"], re.I)
        require(values and len(set(v.upper() for v in values)) == 1 and
            (values[0].upper() == ".TRUE.") == stage["flags"][flag], "Timing loop flags disagree with stage source")
    energy = [line.strip() for line in driver.splitlines() if re.search(r"\bSQRTS\s*=", line, re.I)]
    require(energy == stage["energy_assignment_lines"], "Changed actual timing energy source")
    return energy


def metadata_timing(metadata, backend, printed, snapshot_root=None):
    """Validate the generic CLI observer's raw timers, without rerunning physics."""
    require(metadata["schema"] == "madloop-profile-timing-metadata-v1" and
        metadata["reduction_backends"] == [backend] and metadata["backend_labels"] == [LABELS[backend]],
        "Wrong metadata schema/actual timing backend")
    selected = metadata["selected_timing_helicity"]
    index, physical = selected["one_based_index"], selected["physical_tuple"]
    require(type(index) is int and index > 0 and isinstance(physical, list) and physical and
        all(type(x) is int for x in physical), "Missing actual timing physical helicity")
    require(len(selected["filter_tokens"]) == selected["total_count"] and index <= selected["total_count"] and
        int(selected["filter_tokens"][index-1]) > -10000, "Changed stock timing helicity selection")
    table = [list(map(int,line.split())) for line in metadata_text(selected["helicity_configurations"]).splitlines() if line.strip()]
    require(len(table) == selected["total_count"] and table[index-1] == physical and
        metadata_text(selected["filter"]).split() == selected["filter_tokens"], "Timing table/filter text disagrees with physical selection")
    require(metadata["requested_split_orders"] == -1, "Changed requested coupling-order scope")
    stages = metadata["stages"]
    names = [s["name"] for s in stages]
    expected = ["boot", "polarized_full", "unpolarized_full", "unpolarized_skip", "polarized_skip"]
    require(names == expected or names == ["estimate", *expected], "Missing/reordered CLI timing stages")
    stages = {s["name"]: s for s in stages}
    boot = stages["boot"]["wall_seconds"]
    require(isinstance(boot, (int,float)) and math.isfinite(boot) and boot > 0, "Invalid raw boot timer")
    stage_sources(metadata, stages["boot"], snapshot_root)
    require(stages["boot"]["flags"] == {"SKIPLOOPEVAL": False, "BOOTANDSTOP": True}, "Changed boot measurement flags")
    timing, fields, scope = metadata["timing_result"], {}, None
    base = stages["unpolarized_full"]["actual_npoints"]
    require(type(base) is int and base >= 50, "Invalid actual adaptive count")
    for name, key in (("polarized_full", "run_polarized_total"), ("unpolarized_full", "run_unpolarized_total"),
            ("unpolarized_skip", "run_unpolarized_coefs"), ("polarized_skip", "run_polarized_coefs")):
        stage = stages[name]; polarized = name.startswith("polarized")
        energy = stage_sources(metadata, stage, snapshot_root)
        count = base * (2 if polarized else 1)
        require(stage["actual_npoints"] == count and stage["read_ps"] is False and
            stage["helicity_one_based_index"] == (index if polarized else -1) and
            stage["flags"] == {"SKIPLOOPEVAL": name.endswith("skip"), "BOOTANDSTOP": False}, "Changed actual CLI stage count/flags/helicity")
        wall = stage["wall_seconds"]
        require(isinstance(wall, (int,float)) and math.isfinite(wall) and wall > 0 and stage["boot_wall_seconds"] == boot,
            "Invalid raw CLI stage timer")
        value = (wall-boot)/count
        require(stage["derived_seconds_per_point"] == timing[key] == value, "Raw CLI timer/derived value mismatch")
        parameters = {k.lower():v for k,v in stage["effective_madloop_parameters"].items()}
        require(metadata_text(stage["parameter_card"]) == metadata_text(stage["subprocess_parameter_card"]), "Different effective physical cards")
        card_text = metadata_text(stage["effective_madloop_card"])
        require(all(metadata_text(alias) == card_text for alias in stage["madloop_card_aliases"]), "Different effective numerical card aliases")
        with io.StringIO(card_text) as stream:
            parsed_card = {k.lower():v for k,v in dict(MadLoopParam(stream)).items()}
        require(parsed_card == parameters, "Timing effective card text disagrees with captured settings")
        require(str(parameters.pop("mlreductionlib")) == str(backend) and parameters["ctmoderun"] == parameters["ctmodeinit"] == 1,
            "CLI stage is not the actual selected direct-DP backend")
        candidate = {"physical_card_sha256": stage["parameter_card"]["sha256"],
            "timing_energy_source": energy or None, "helicity_index": index,
            "physical_helicity": physical, "loop_optimized_output": True, "direct_dp": True,
            "helicity_table_sha256": selected["helicity_configurations"]["sha256"],
            "helicity_filter_sha256": selected["filter"]["sha256"],
            "selected_helicity_weight": int(selected["filter_tokens"][index-1]),
            "common_madloop_settings": parameters,
            "actual_split_orders_target": stage.get("split_orders_target") if stage.get("split_orders_target") is not None else "default-all",
            "compiler_settings_sha256": metadata_runtime(metadata, stage).get("Source/make_opts", {}).get("sha256")}
        require(scope is None or candidate == scope, "CLI stage physical/card/energy settings drift")
        scope = candidate; fields[key] = value
    require(timing["Booting_time"] == boot, "Changed exact returned boot timing")
    for name, total, coefs in (("single_helicity", "run_polarized_total", "run_polarized_coefs"),
            ("helicity_sum", "run_unpolarized_total", "run_unpolarized_coefs")):
        values = {"total": fields[total]*1000, "construction": fields[coefs]*1000,
            "reduction_estimate": (fields[total]-fields[coefs])*1000}
        for field, value in values.items():
            record = printed[name][field]
            require(abs(value-record["milliseconds"]) <= record["printed_rounding_bound_ms"] + 1e-10,
                "Raw metadata disagrees with printed CLI timing/units")
        fields[name] = {"seconds_per_point": fields[total], "construction_seconds_per_point": fields[coefs],
            "reduction_estimate_seconds_per_point": fields[total]-fields[coefs]}
    return {"raw_timing": fields, "stage_scope": scope, "scope_complete": same_stage_scope(scope, scope),
        "adaptive_base_points": base, "metadata_stage_count": len(names),
        "selected_helicity_weight": int(selected["filter_tokens"][index-1])}


def execute_one(spec_path, index, handoff_path, execute=False):
    require(execute, "Numerical CLI execution is held without explicit launch flag")
    spec = read(spec_path); item = spec["schedule"][index]
    require(spec["phase"] == PHASE and item["backend"] in LABELS, "Wrong profile campaign/tool")
    require(read(handoff_path) == {"root_authorized": True, "phase": PHASE, "cpu": 207,
        "specification_sha256": workers.digest(Path(spec_path))}, "Missing exact root CLI handoff")
    require(spec.get("ready") is True, "Final CLI metadata/source freeze and review are still held")
    require(207 in os.sched_getaffinity(0), "Quiet CPU207 affinity is required")
    for record in spec["sources"].values(): checked(record)
    scope = declared_scope(spec)
    case = spec["cases"][item["case"]]
    require(re.fullmatch(r"[A-Za-z0-9_-]+", item["case"]), "Invalid declared case/repetition")
    check_declared_item(item, scope)
    for key in ("physical_card", "ninja_archive", "ninjars_archive"): checked(case[key])
    work = policy.path(spec["work"]) / item["case"] / str(item["backend"])
    report = policy.path(spec["report"]) / item["case"] / str(item["backend"]) / str(item["repetition"])
    require(not report.exists(), "Preserve every prior CLI attempt")
    require((not work.exists()) if item["repetition"] == 0 else work.is_dir(), "Wrong fresh/reused CLI output state")
    work.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True)
    before = list(work.glob("SAVEDTMP_CHECK_*/SubProcesses/P*_*"))
    previous = None
    if item["npoints"] == 0:
        require(len(before) == 1, "Expected one prior CLI output")
        cache = list(before[0].glob("SavedStabilityRun_%s_0.pkl" % LABELS[item["backend"]]))
        require(len(cache) == 1, "No declared initial cached CLI sample")
        previous = cached_stability(cache[0], scope["initial_stability_points"])
    card = report / "commands.mg5"; card.write_text(command_card(case, item["backend"], item["npoints"]))
    shutil.copy2(spec_path, report / "input-specification.json")
    shutil.copy2(handoff_path, report / "root-handoff.json")
    shutil.copy2(__file__, report / "profile_cli.py")
    for old in work.glob("profile_optimized_*.log"): old.unlink()
    for old in work.glob("SAVEDTMP_CHECK_*/SavedRunInfos/profile_timing_metadata.json"): old.unlink()
    result = {"phase": PHASE, "complete": False, "acceptance_eligible": False,
        "producer_role": PRODUCER_ROLE, "producer_source": bound(Path(__file__)),
        "specification": bound(spec_path), "command_card": bound(card), "item": item,
        "declared_scope": scope,
        "source_root": str(ROOT),
        "work": policy.relative(work), "paired_ratio_eligible": False,
        "stage_scope": None, "stage_scope_limitation": "Await exact timing-stage metadata; final profile output alone cannot prove prior selected timing helicity"}
    try:
        with ownership.owned_runtime(report):
            environment = ownership.profile_environment(work)
            command = ["taskset", "-c", "207", str(ROOT / "bin/madgraph"), "--debug", "--plain", str(card)]
            with (report / "stdout.log").open("w") as log:
                process = workers.subprocess.Popen(command, cwd=work, env=environment,
                    stdout=log, stderr=workers.subprocess.STDOUT, start_new_session=True)
                require(process.wait(timeout=spec["timeout_seconds"]) == 0, "Actual MG7 CLI failed")
            logs = list(work.glob("profile_optimized_*.log")); require(len(logs) == 1, "CLI did not produce exactly one fresh profile log")
            shutil.copy2(logs[0], report / "profile.log")
            parsed = parse_profile((report / "profile.log").read_text(), item["backend"])
            outputs = list(work.glob("SAVEDTMP_CHECK_*/SubProcesses/P*_*")); require(len(outputs) == 1, "Wrong actual generated output count")
            directory = outputs[0]; output = directory.parents[1]
            timing_metadata_path = output / "SavedRunInfos/profile_timing_metadata.json"
            shutil.copy2(timing_metadata_path, report / "timing-metadata.json")
            shutil.copytree(output / "SavedRunInfos", report / "timing-stage-inputs")
            timing_metadata = read(report / "timing-metadata.json")
            raw_timing = metadata_timing(timing_metadata, item["backend"], parsed, report / "timing-stage-inputs")
            require(raw_timing["stage_scope"]["physical_card_sha256"] == case["physical_card"]["sha256"] and
                workers.digest(output / "Cards/param_card.dat") == case["physical_card"]["sha256"], "Actual CLI physical card differs from retained input")
            cache = list(directory.glob("SavedStabilityRun_%s_0.pkl" % LABELS[item["backend"]]))
            require(len(cache) == 1, "Missing actual CLI stability sample")
            cached = cached_stability(cache[0], scope["initial_stability_points"])
            require(previous is None or previous == cached, "profile0 changed cached stability sample")
            shutil.copy2(cache[0], report / cache[0].name)
            for path in (output / "Cards/param_card.dat", directory / "MadLoop5_resources/MadLoopParams.dat", directory / "check_sa.f", directory / "loop_matrix.f"):
                shutil.copy2(path, report / path.name)
            stdout = (report / "stdout.log").read_text()
            counts = re.findall(r"Checking timing for process .+? with (\d+) PS points\.", stdout)
            require(len(counts) == 1 and int(counts[0]) >= 50, "Missing actual adaptive CLI timing count")
            result.update(timing=parsed, adaptive_base_points=int(counts[0]),
                single_helicity_points=2*int(counts[0]), helicity_sum_points=int(counts[0]),
                stability=cached, new_stability_points=scope["initial_stability_points"] if previous is None else 0,
                generated_sources=bench.generated_source_manifest(output), libraries=policy.library_hashes(output),
                executable=bound(directory / "check"), thread_environment={k: environment[k] for k in pair.THREADS})
            selected_archive = "libninja.a" if item["backend"] == 6 else "libninjars.a"
            expected_archive = case["ninja_archive" if item["backend"] == 6 else "ninjars_archive"]["sha256"]
            require(result["libraries"].get(selected_archive) == expected_archive, "Actual CLI selected reducer archive mismatch")
            require(all(any(Path(name).name == selected_archive and record.get("sha256") == expected_archive
                for name, record in metadata_runtime(timing_metadata, stage).items()) for stage in timing_metadata["stages"]),
                "Timing stage used an unbound reducer archive")
            require(raw_timing["adaptive_base_points"] == int(counts[0]), "Actual adaptive count disagrees with CLI log")
            result.update(raw_timing=raw_timing["raw_timing"], stage_scope=raw_timing["stage_scope"],
                stage_scope_complete=raw_timing["scope_complete"], selected_helicity_weight=raw_timing["selected_helicity_weight"],
                stage_scope_limitation=None if raw_timing["scope_complete"] else "Missing actual timing scope; this cell is incomparable",
                timing_metadata=bound(report / "timing-metadata.json"))
            for record in spec["sources"].values(): checked(record)
            result["complete"] = True
    except BaseException as error:
        result["complete"] = False
        result["error"] = repr(error)
        raise
    finally:
        # All stdout/source/cache files already present remain after any failure.
        for path in (*work.glob("profile_optimized_*.log"), *work.glob("stability_optimized_*.log"),
                *work.glob("SAVEDTMP_CHECK_*/SavedRunInfos/profile_timing_metadata.json"),
                *work.glob("SAVEDTMP_CHECK_*/SubProcesses/P*_*/SavedStabilityRun_*.pkl")):
            target = report / "retained-work" / path.relative_to(work)
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        result["files"] = {str(p.relative_to(report)): bound(p) for p in report.rglob("*") if p.is_file()}
        bench.dump(report / "result.json", result)
    return result


def failure_snapshot(filename, expected=None, inputs=None):
    """Retain a failed attempt's custody and error, without interpreting timings."""
    filename = policy.path(filename); report = filename.parent; result = read(filename)
    require(result["phase"] == PHASE and result["complete"] is False and
        result["acceptance_eligible"] is False and isinstance(result.get("error"), str) and result["error"],
        "Not a retained failed CLI attempt")
    if inputs is not None: inputs[policy.relative(filename)] = workers.digest(filename)
    for name, record in result["files"].items():
        require(policy.path(record["path"]) == report / name, "CLI failure inventory escaped its command")
        checked(record)
        if inputs is not None: inputs[record["path"]] = record["sha256"]
    for name in ("input-specification.json", "root-handoff.json", "profile_cli.py", "commands.mg5"):
        require(name in result["files"], "Missing retained CLI failure authority/input")
    spec = read(report / "input-specification.json")
    require(workers.digest(report / "input-specification.json") == result["specification"]["sha256"] and
        spec["phase"] == PHASE and spec["ready"] is True and result["item"] in spec["schedule"],
        "Mixed CLI failure specification or undeclared attempted cell")
    item = result["item"]; backend = item["backend"]; case = spec["cases"][item["case"]]
    scope = declared_scope(spec); check_declared_item(item, scope)
    require(result.get("declared_scope", scope) == scope, "Changed declared CLI scope")
    require(read(report / "root-handoff.json") == {"root_authorized": True, "phase": PHASE, "cpu": 207,
        "specification_sha256": result["specification"]["sha256"]}, "Wrong retained CLI failure handoff")
    require(result["producer_role"] == PRODUCER_ROLE and result["producer_source"] in spec["sources"].values() and
        workers.digest(report / "profile_cli.py") == result["producer_source"]["sha256"], "Changed failed CLI producer role/source")
    require((report / "commands.mg5").read_text() == command_card(case, backend, item["npoints"],
        verify_inputs=False, source_root=result["source_root"]), "Failed CLI command differs from its declared cell")
    declared = {"ninja_archive": case["ninja_archive"]["sha256"], "ninjars_archive": case["ninjars_archive"]["sha256"],
        "process_checks": spec["sources"]["madgraph/various/process_checks.py"]["sha256"]}
    require(expected is None or declared == expected, "Failed CLI source/runtime pins differ from selected cohort")
    return {"schema": "retained-mg7-cli-profile-failure-v1", "status": "failed", "error": result["error"],
        "source": bound(filename), "specification": result["specification"], "declared_specification": spec, "item": item,
        "producer_source": result["producer_source"], "runtime_pins": declared, "raw_files": result["files"],
        "declared_scope": scope}


def report_snapshot(filename, expected=None, inputs=None):
    """Read-only authority for one actual CLI command, never a timing gate."""
    filename = policy.path(filename); report = filename.parent; result = read(filename)
    require(result["phase"] == PHASE and result["complete"] is True and "error" not in result and
        result["acceptance_eligible"] is False, "Incomplete/failed/non-diagnostic CLI result")
    if inputs is not None: inputs[policy.relative(filename)] = workers.digest(filename)
    for name, record in result["files"].items():
        require(policy.path(record["path"]) == report / name, "CLI result inventory escaped its command")
        checked(record)
        if inputs is not None: inputs[record["path"]] = record["sha256"]
    for name in ("input-specification.json", "root-handoff.json", "profile_cli.py", "commands.mg5", "profile.log", "timing-metadata.json"):
        require(name in result["files"], "Missing actual CLI authority/input/raw artifact")
    spec = read(report / "input-specification.json")
    require(workers.digest(report / "input-specification.json") == result["specification"]["sha256"] and
        spec["phase"] == PHASE and spec["ready"] is True and result["item"] in spec["schedule"], "Mixed CLI specification or undeclared item")
    item = result["item"]; backend = item["backend"]; case = spec["cases"][item["case"]]
    scope = declared_scope(spec); check_declared_item(item, scope)
    require(result.get("declared_scope", scope) == scope, "Changed declared CLI scope")
    require(read(report / "root-handoff.json") == {"root_authorized": True, "phase": PHASE, "cpu": 207,
        "specification_sha256": result["specification"]["sha256"]}, "Wrong retained root CLI handoff")
    require(result["producer_role"] == PRODUCER_ROLE and result["producer_source"] in spec["sources"].values() and
        workers.digest(report / "profile_cli.py") == result["producer_source"]["sha256"], "Changed actual CLI producer role/source")
    require((report / "commands.mg5").read_text() == command_card(case, backend, item["npoints"],
        verify_inputs=False, source_root=result["source_root"]), "Actual CLI command card differs from declared physical inputs")
    declared = {"ninja_archive": case["ninja_archive"]["sha256"], "ninjars_archive": case["ninjars_archive"]["sha256"],
        "process_checks": spec["sources"]["madgraph/various/process_checks.py"]["sha256"]}
    require(expected is None or declared == expected, "CLI source/runtime pins differ from selected cohort")
    printed = parse_profile((report / "profile.log").read_text(), backend)
    metadata = read(report / "timing-metadata.json")
    actual = metadata_timing(metadata, backend, printed, report / "timing-stage-inputs")
    require(result["timing"] == printed and result["raw_timing"] == actual["raw_timing"] and
        result["stage_scope"] == actual["stage_scope"] and result["stage_scope_complete"] == actual["scope_complete"] and
        result["selected_helicity_weight"] == actual["selected_helicity_weight"] and
        result["adaptive_base_points"] == actual["adaptive_base_points"] and
        result["single_helicity_points"] == 2*actual["adaptive_base_points"] and result["helicity_sum_points"] == actual["adaptive_base_points"],
        "CLI raw metadata/result disagrees")
    require(actual["stage_scope"]["physical_card_sha256"] == case["physical_card"]["sha256"], "Changed actual physical card")
    archive = "libninja.a" if backend == 6 else "libninjars.a"; archive_sha = declared["ninja_archive" if backend == 6 else "ninjars_archive"]
    require(result["libraries"].get(archive) == archive_sha and all(any(Path(name).name == archive and record.get("sha256") == archive_sha
        for name, record in metadata_runtime(metadata, stage).items()) for stage in metadata["stages"]), "Changed actual selected CLI runtime")
    # Snapshot files must belong to the complete transitive inventory, not only match a free-standing hash.
    for stage in metadata["stages"]:
        for record in stage["source_snapshots"].values():
            relative = Path(record["path"]).relative_to(Path(metadata["export_dir"]) / "SavedRunInfos")
            name = str(Path("timing-stage-inputs") / relative)
            require(name in result["files"] and result["files"][name]["sha256"] == record["sha256"], "Untracked timing source snapshot")
    cache_path = report / ("SavedStabilityRun_%s_0.pkl" % LABELS[backend])
    require(cache_path.name in result["files"], "Missing retained actual CLI stability cache")
    cache = cached_stability(cache_path, scope["initial_stability_points"])
    require(result["new_stability_points"] == (scope["initial_stability_points"] if item["repetition"] == 0 else 0) and
        cache["cache"]["sha256"] == result["stability"]["cache"]["sha256"] and
        all(cache[k] == result["stability"][k] for k in ("dp_points","qp_points","scope")), "Changed/counterfeit cached stability sample")
    return {"schema": "validated-mg7-cli-profile-result-v1", "acceptance_eligible": False,
        "source": bound(filename), "specification": result["specification"], "declared_specification": spec,
        "item": item, **actual, "timing": printed, "runtime_pins": declared, "declared_scope": scope,
        "stage_scope_complete": actual["scope_complete"], "stage_point_counts": {stage["name"]:stage["actual_npoints"] for stage in metadata["stages"]},
        "timing_metadata": bound(report / "timing-metadata.json"),
        "runtime_files": metadata_runtime(metadata, metadata["stages"][0]), "generated_sources": result["generated_sources"],
        "stability": cache, "new_stability_points": result["new_stability_points"], "timed_status_counts_available": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--handoff", required=True, type=Path); parser.add_argument("--execute-numerical", action="store_true")
    args = parser.parse_args(); execute_one(args.spec, args.index, args.handoff, args.execute_numerical)
