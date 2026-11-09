#!/usr/bin/env python3
"""Same-input fixed-QP diagnostics; never performance or stability certification.

Run from the MadGraph root after the campaign owner releases its timing window.
Only an isolated protocol driver is recompiled; numerical objects and libraries
remain unchanged. Numerical cards and filters are restored after each worker,
including failed workers. Evidence is append-only.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import fcntl
import itertools
import importlib.util
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench
from tirust_complex_stress import parse_forced_qp

REPORT = Path(__file__).resolve().parent
FROZEN_PROTOCOL = REPORT / "protocol"
PILOT = REPORT.parent / "numerical/strict-policy-qcd-n3"
CASE_ID = "qcd_n3_uux_zgg"
MINIMAL = [39, 60, 91, 93]


def inputs():
    pilot = json.loads((PILOT / "results.json").read_text())
    config = json.loads((PILOT / "cases.json").read_text())
    specification = json.loads((PILOT / "specification.json").read_text())
    if bench.digest(PILOT / "cases.json") != specification["configuration_sha256"]:
        raise ValueError("Original pilot configuration changed")
    if config["madloop_parameters"]["MLStabThres"] != 1e-6:
        raise ValueError("Expected the frozen strict pilot")
    case = next(c for c in config["cases"] if c["id"] == CASE_ID)
    if case != pilot["case"] or any(r["status"] != "complete" for r in pilot["results"]):
        raise ValueError("Incomplete or different pilot")
    metadata_path = bench.REPORT / "points" / (CASE_ID + ".json")
    metadata = json.loads(metadata_path.read_text())
    point_path = metadata_path.with_suffix(".dat")
    if bench.digest(point_path) != metadata["sha256"]:
        raise ValueError("Original physical points changed")
    from tirust_benchmark_worker import parse_outputs
    cards = set()
    for result in pilot["results"]:
        run = PILOT / f"{result['source']}-{result['backend']}"
        observed = parse_outputs(run / "raw", [result["backend"]], config["points_per_case"], 1)
        if observed["rows"] != result["observation"]["rows"]:
            raise ValueError("Pilot rows differ from raw evidence")
        record = json.loads((run / (result["source"] + "-inputs.json")).read_text())
        if record["points_sha256"] != metadata["sha256"] or record["generation"]["case"] != case:
            raise ValueError("Pilot physical input identity differs")
        cards.add(record["param_card_sha256"])
    if len(cards) != 1:
        raise ValueError("Pilot runs have different physical cards")
    return pilot, config, case, metadata, point_path, cards.pop()


def target_misses(pilot):
    """Derive the complete ordinary-target miss set without changing its gate."""
    misses = set()
    for result in pilot["results"]:
        for row in result["comparison"]["residuals"]:
            if row["order"] != 0:
                continue
            error, pole = row["symmetric_relative_error"], row["pole_to_finite"]
            if (error is None or error >= 1e-8) and (pole is None or pole >= 1e-8):
                misses.add(row["point"])
    return sorted(misses)


def frozen_file(path, data):
    """Do not silently replace a previous point/configuration snapshot."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Frozen evidence changed: {path}")
    else:
        path.write_bytes(data)


def freeze_subset(group):
    pilot, config, case, metadata, parent, card_sha = inputs()
    indices = MINIMAL if group == "minimal" else target_misses(pilot)
    if not indices:
        raise ValueError("No selected points")
    lines = parent.read_text().splitlines(keepends=True)
    nexternal = case["nfinal"] + 2
    if int(lines[0]) != metadata["inputs"]["count"] or len(lines) != 1 + int(lines[0]) * nexternal:
        raise ValueError("Unexpected original point-file layout")
    # Copy original decimal records verbatim; do not round or regenerate inputs.
    content = str(len(indices)) + "\n" + "".join(
        "".join(lines[1 + (i-1)*nexternal:1 + i*nexternal]) for i in indices)
    path = REPORT / "points" / (group + ".dat")
    frozen_file(path, content.encode())
    selected = [metadata["points"][i-1] for i in indices]
    for point in selected:
        total = [sum(v[k] for v in point["momenta"][:2]) for k in range(4)]
        if total != [1000., 0., 0., 0.]:
            raise ValueError("Unexpected incoming invariant/renormalization scale")
    record = {"inputs": dict(metadata["inputs"], count=len(indices)), "points": selected,
              "sha256": bench.digest(path), "parent_sha256": metadata["sha256"],
              "source_point_indices": indices, "parameter_card_sha256": card_sha,
              "mu_r_gev": 1000., "mu_r_squared_gev2": 1000000.,
              "scope": "Exact original physical inputs; fixed-QP numerical diagnostic only"}
    frozen_file(path.with_suffix(".json"), (json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
    frozen_file(REPORT / "original-policy-cases.json", (PILOT / "cases.json").read_bytes())
    return pilot, config, case, path, record


def snapshot_mutable(output):
    cards = {output / "Cards/MadLoopParams.dat", *(output / "SubProcesses").rglob("MadLoopParams.dat")}
    filters = {p for pattern in ("*HelFilter.dat", "*LoopFilter.dat")
               for p in (output / "SubProcesses").rglob(pattern)}
    return {p.resolve(): p.read_bytes() for p in cards | filters}


def restore_mutable(output, saved):
    for pattern in ("*HelFilter.dat", "*LoopFilter.dat"):
        for path in (output / "SubProcesses").rglob(pattern):
            if path.resolve() not in saved:
                path.unlink()
    for path, content in saved.items():
        path.write_bytes(content)
    if any(path.read_bytes() != content for path, content in saved.items()):
        raise RuntimeError("Could not restore numerical cards/filters")


def isolated_copy(source, case, group, backend, initialization, report):
    """Copy generated code and mutable files while retaining pinned libraries."""
    original = bench.output_dir(source, case)
    workspace = ROOT / ".tirust-build/qcd-n3-adjudication" / group / f"{backend}-init-{initialization}-unchecked"
    output = workspace / "outputs" / source / case["id"]
    def ignored(directory, names):
        return [name for name in names if name in ("build-history", "worker-raw")
                or stat.S_ISFIFO((Path(directory) / name).lstat().st_mode)]
    shutil.copytree(original, output, symlinks=True, ignore=ignored)
    # These archives are inputs to an already linked executable, not writable
    # outputs of this driver-only compilation. Preserve their canonical
    # original identities so start_worker's link-time checks remain exact.
    for archive in (output / "lib").glob("*.a"):
        if not archive.is_symlink():
            archive.unlink()
            archive.symlink_to((original / "lib" / archive.name).resolve())
    compilation = json.loads((original / "compilation.json").read_text())
    if bench.generated_sources(output) != compilation["sources"]:
        raise ValueError("Isolated source copy differs from compiled source inventory")
    subdirs = sorted(output.glob("SubProcesses/P*_*"))
    if len(subdirs) != 1 or bench.digest(subdirs[0] / "check") != compilation["executable_sha256"]:
        raise ValueError("Isolated executable differs from original source-built executable")
    dependencies = bench.dependency_inputs(subdirs[0])
    if dependencies != compilation["link_inputs_before"] or dependencies != compilation["link_inputs_after"]:
        raise ValueError("Isolated dependency aliases differ from original link inputs")
    bench.dump(report / "isolation.json", {"original": str(original), "copy": str(output),
        "compilation_sha256": bench.digest(original / "compilation.json"),
        "executable_sha256": compilation["executable_sha256"],
        "source_inventory_unchanged": True, "link_inputs_unchanged": True,
        "mutations": "Isolated driver-only relink plus numerical cards and filter files"})
    return workspace


def start_unchecked_worker(stack, source, case, backend, config, cpu, report, timeout, points):
    """Relink only an isolated driver with forced checks disabled for fixed QP."""
    from tirust_benchmark_compiler import capture_effective_flags
    worker_spec = importlib.util.spec_from_file_location(
        "tirust_qp_diagnostic_worker", FROZEN_PROTOCOL / "tirust_benchmark_worker.py")
    worker_module = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker_module)
    Worker = worker_module.Worker
    output = bench.output_dir(source, case)
    directory = bench.subprocess_dir(source, case)
    compilation = json.loads((output / "compilation.json").read_text())
    original_sources = bench.generated_sources(output)
    bench.verify_worker_template(compilation, FROZEN_PROTOCOL / "tirust_benchmark_worker.f")
    if original_sources != compilation["sources"]:
        raise ValueError("Original numerical sources changed")
    flags = capture_effective_flags(directory, env=bench.environment())
    if not bench.same_compiler_policy(flags, compilation["effective_compiler_flags"]):
        raise ValueError("Isolated effective compiler policy changed")
    objects_before = {str(path.relative_to(output)): bench.digest(path)
        for path in output.rglob("*.o") if path.name != "check_sa.o"}
    dependencies = bench.dependency_inputs(directory)
    template = (FROZEN_PROTOCOL / "tirust_benchmark_worker.f").read_text()
    if template.count("FORCE_STABILITY_CHECK(.TRUE.)") != 1:
        raise ValueError("Unexpected original worker stability-check call")
    unchecked = template.replace("FORCE_STABILITY_CHECK(.TRUE.)", "FORCE_STABILITY_CHECK(.FALSE.)")
    driver = bench.render_worker(unchecked, compilation["worker_render_parameters"])
    (report / "unchecked-worker.f").write_text(unchecked)
    (report / "check_sa.f").write_text(driver)
    (directory / "check_sa.f").write_text(driver)
    command = ["make", "SHELL=" + shutil.which("bash"), "-j1", "check"]
    env = bench.environment()
    env["PWD"] = str(directory)
    with (report / "driver-build.log").open("w") as stream:
        subprocess.run(command, cwd=directory, env=env, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=180)
    objects_after = {str(path.relative_to(output)): bench.digest(path)
        for path in output.rglob("*.o") if path.name != "check_sa.o"}
    if objects_after != objects_before or bench.dependency_inputs(directory) != dependencies:
        raise ValueError("Diagnostic link changed numerical objects or libraries")
    sources = bench.generated_sources(output)
    changed = [key for key in sources if sources[key] != original_sources.get(key)]
    expected_driver = str((directory / "check_sa.f").relative_to(output))
    if set(sources) != set(original_sources) or changed != [expected_driver]:
        raise ValueError("Only the diagnostic driver may change")
    flags_after = capture_effective_flags(directory, env=bench.environment())
    if not bench.same_compiler_policy(flags, flags_after):
        raise ValueError("Diagnostic effective compiler policy changed")
    diagnostic = {"scope": "Only isolated driver stability-forcing flag changed; numerical objects unchanged",
        "command": command, "executable_sha256": bench.digest(directory / "check"), "sources": sources,
        "driver_sha256": bench.digest(directory / "check_sa.f"),
        "template_sha256": bench.digest(report / "unchecked-worker.f"),
        "numerical_objects_before": objects_before, "numerical_objects_after": objects_after,
        "link_inputs_before": dependencies, "link_inputs_after": bench.dependency_inputs(directory),
        "effective_compiler_flags": flags_after}
    bench.dump(report / "diagnostic-compilation.json", diagnostic)
    settings = bench.configure(source, case, [backend], config)
    generation = json.loads((output / "generation.json").read_text())
    card = output / "Cards/param_card.dat"
    card_hash = bench.digest(card)
    if card_hash != generation["parameter_card_sha256"]:
        raise ValueError("Diagnostic physical card differs from source generation")
    candidates = [directory / "param_card.dat"] + [directory.joinpath(*([".."]*level), "Cards/param_card.dat")
                                                   for level in range(6)]
    resolved = next((path for path in candidates if path.is_file()), None)
    if resolved is None or bench.digest(resolved) != card_hash:
        raise ValueError("Fortran resolves a different diagnostic physical card")
    metadata = json.loads(points.with_suffix(".json").read_text())
    if bench.digest(points) != metadata["sha256"] or metadata["inputs"]["count"] != config["points_per_case"]:
        raise ValueError("Diagnostic points changed")
    if metadata["inputs"]["incoming_masses"] + metadata["inputs"]["masses"] != compilation["masses"]:
        raise ValueError("Diagnostic masses differ from compiled process")
    bench.dump(report / (source + "-inputs.json"), {"settings": settings, "points_sha256": bench.digest(points),
        "param_card_sha256": card_hash, "resolved_parameter_card": str(resolved.resolve()),
        "runtime": bench.runtime_identity(directory), "compilation": compilation,
        "diagnostic_compilation": diagnostic, "generation": generation})
    worker = Worker(["taskset", "-c", str(cpu), "./check", str(points)], directory,
                    report / (source + "-output.txt"), env=bench.environment(), timeout=timeout)
    stack.callback(worker.close)
    return worker, settings


def run(group, backend, cpu, timeout, initialization):
    pilot, config, case, points, point_record = freeze_subset(group)
    source = "baseline" if backend == 1 else "candidate"
    report = REPORT / "runs" / group / f"{source}-{backend}-init-{initialization}-unchecked"
    report.mkdir(parents=True, exist_ok=False)
    config = copy.deepcopy(config)
    config["points_per_case"] = point_record["inputs"]["count"]
    init_mode = 1 if initialization == "dp" else 4
    config["madloop_parameters"].update(CTModeInit=init_mode, CTModeRun=4)
    bench.dump(report / "configuration.json", config)
    protocol_names = ("tirust_benchmark.py", "tirust_benchmark_worker.py", "tirust_benchmark_worker.f",
                      "tirust_benchmark_compiler.py", "tirust_complex_stress.py")
    manifest = json.loads((FROZEN_PROTOCOL / "manifest.json").read_text())
    if any(bench.digest(FROZEN_PROTOCOL / name) != value for name, value in manifest.items()):
        raise ValueError("Frozen worker protocol changed")
    for name in protocol_names:
        origin = FROZEN_PROTOCOL if name in manifest else ROOT / "tools"
        shutil.copy2(origin / name, report / name)
    shutil.copy2(Path(__file__), report / "run.py")
    specification = {"case": case, "source": source, "backend": backend, "group": group,
        "initialization_mode": init_mode,
        "performance_eligible": False, "stability_certified": False,
        "scope": "Forced-QP coefficient diagnostic; fixed mode has no independent stability certification",
        "pilot_results_sha256": bench.digest(PILOT / "results.json"),
        "point_record": point_record, "configuration_sha256": bench.digest(report / "configuration.json"),
        "protocols": {name: bench.digest(report / name) for name in (*protocol_names, "run.py")}}
    bench.dump(report / "specification.json", specification)
    original_workspace = bench.WORK
    try:
        with ExitStack() as stack:
            if backend == 9:
                lock = stack.enter_context((ROOT / ".tirust-build/numerical-tests.lock").open("a"))
                print("Waiting for exclusive numerical-tests lock", flush=True)
                fcntl.flock(lock, fcntl.LOCK_EX)
                print("Acquired numerical-tests lock", flush=True)
            # Sole CT/Ninja dispatch never evaluates Symbolica. Isolated mutable
            # files allow these diagnostics to avoid blocking the Symbolica queue.
            bench.WORK = isolated_copy(source, case, group, backend, initialization, report)
            output = bench.output_dir(source, case)
            directory = bench.subprocess_dir(source, case)
            saved = snapshot_mutable(output)
            # LIFO callbacks close the worker, restore state, then release lock.
            stack.callback(restore_mutable, output, saved)
            worker, settings = start_unchecked_worker(stack, source, case, backend, config,
                                                      cpu, report, timeout, points)
            shutil.copy2(output / "Cards/MadLoopParams.dat", report / "MadLoopParams.dat")
            shutil.copy2(output / "Cards/param_card.dat", report / "param_card.dat")
            saved_inputs = json.loads((report / (source + "-inputs.json")).read_text())
            if saved_inputs["param_card_sha256"] != point_record["parameter_card_sha256"]:
                raise ValueError("Physical card differs from the original pilot")
            if (settings["ctmoderun"] != 4 or settings["ctmodeinit"] != init_mode
                    or settings["mlreductionlib"] != str(backend)):
                raise ValueError("Fixed-QP mode was not applied")
            try:
                worker.run(1)
            finally:
                bench.retain_observation(worker.directory, report / "raw")
            result = parse_forced_qp(report, backend, config["points_per_case"])
        runtime_after = bench.runtime_identity(directory)
        if any(runtime_after[key] != saved_inputs["runtime"][key]
               for key in ("libraries", "archives", "metadata")):
            raise ValueError("Libraries changed during the diagnostic")
        compilation = saved_inputs["diagnostic_compilation"]
        if bench.digest(directory / "check") != compilation["executable_sha256"]:
            raise ValueError("Executable changed during the diagnostic")
        if bench.generated_sources(output) != compilation["sources"]:
            raise ValueError("Generated source changed during the diagnostic")
        for row in result["rows"]:
            row["point"] = point_record["source_point_indices"][row["point"]-1]
        bench.dump(report / "result.json", {**specification, "status": "diagnostic", "settings": settings,
            "numerical_card_sha256": bench.digest(report / "MadLoopParams.dat"),
            "runtime_after": runtime_after, "observation": result})
        print(json.dumps({"source": source, "backend": backend, "group": group,
                          "status": "diagnostic", "points": config["points_per_case"]}), flush=True)
    except BaseException as error:
        bench.dump(report / "failure.json", {**specification, "status": "failed", "error": repr(error)})
        raise
    finally:
        bench.WORK = original_workspace


def validate_binding(report, record, pilot, original_config, case, parent, parent_file, card_sha):
    """Revalidate physical inputs and fixed-QP scope before any comparison."""
    group = record["group"]
    if group not in ("minimal", "target-misses"):
        raise ValueError("Unknown diagnostic subset")
    point_file = REPORT / "points" / (group + ".dat")
    metadata = json.loads(point_file.with_suffix(".json").read_text())
    indices = MINIMAL if group == "minimal" else target_misses(pilot)
    if (record["point_record"] != metadata or metadata["source_point_indices"] != indices
            or metadata["parent_sha256"] != parent["sha256"]
            or metadata["points"] != [parent["points"][i-1] for i in indices]
            or bench.digest(point_file) != metadata["sha256"]
            or metadata["parameter_card_sha256"] != card_sha
            or record["pilot_results_sha256"] != bench.digest(PILOT / "results.json")):
        raise ValueError("Diagnostic subset/parent/pilot identity changed")
    lines = parent_file.read_text().splitlines(keepends=True)
    n = case["nfinal"] + 2
    exact = str(len(indices)) + "\n" + "".join("".join(lines[1+(i-1)*n:1+i*n]) for i in indices)
    if point_file.read_text() != exact:
        raise ValueError("Diagnostic subset does not preserve original decimal inputs")
    configuration = json.loads((report / "configuration.json").read_text())
    expected = copy.deepcopy(original_config)
    expected["points_per_case"] = len(indices)
    if record["initialization_mode"] not in (1, 4):
        raise ValueError("Unknown initialization mode")
    expected["madloop_parameters"].update(CTModeInit=record["initialization_mode"], CTModeRun=4)
    saved_inputs = json.loads((report / (record["source"] + "-inputs.json")).read_text())
    source = "baseline" if record["backend"] == 1 else "candidate"
    if (record["backend"] not in (1, 6, 9) or record["source"] != source
            or record["case"] != case or saved_inputs["generation"]["case"] != case
            or configuration != expected
            or bench.digest(report / "configuration.json") != record["configuration_sha256"]
            or bench.digest(report / "param_card.dat") != card_sha
            or saved_inputs["param_card_sha256"] != card_sha
            or saved_inputs["generation"]["parameter_card_sha256"] != card_sha
            or saved_inputs["points_sha256"] != metadata["sha256"]
            or saved_inputs["settings"] != record["settings"]
            or record["settings"]["mlreductionlib"] != str(record["backend"])
            or record["settings"]["ctmoderun"] != 4
            or record["settings"]["ctmodeinit"] != record["initialization_mode"]
            or bench.digest(report / "MadLoopParams.dat") != record["numerical_card_sha256"]):
        raise ValueError("Diagnostic card/configuration/source binding changed")
    if any(bench.digest(report / name) != digest for name, digest in record["protocols"].items()):
        raise ValueError("Retained diagnostic protocol changed")
    if any(record["runtime_after"][key] != saved_inputs["runtime"][key]
           for key in ("libraries", "archives", "metadata")):
        raise ValueError("Diagnostic libraries changed during evaluation")
    original = saved_inputs["compilation"]
    diagnostic = saved_inputs["diagnostic_compilation"]
    bench.verify_worker_template(original, report / "tirust_benchmark_worker.f")
    template = (report / "tirust_benchmark_worker.f").read_text()
    if template.count("FORCE_STABILITY_CHECK(.TRUE.)") != 1:
        raise ValueError("Original worker stability flag changed")
    expected_template = template.replace("FORCE_STABILITY_CHECK(.TRUE.)", "FORCE_STABILITY_CHECK(.FALSE.)")
    expected_driver = bench.render_worker(expected_template, original["worker_render_parameters"])
    source_keys = set(diagnostic["sources"])
    changed = [key for key in source_keys if diagnostic["sources"][key] != original["sources"].get(key)]
    if (diagnostic != json.loads((report / "diagnostic-compilation.json").read_text())
            or (report / "unchecked-worker.f").read_text() != expected_template
            or (report / "check_sa.f").read_text() != expected_driver
            or bench.digest(report / "unchecked-worker.f") != diagnostic["template_sha256"]
            or bench.digest(report / "check_sa.f") != diagnostic["driver_sha256"]
            or source_keys != set(original["sources"])
            or len(changed) != 1 or not changed[0].endswith("/check_sa.f")
            or diagnostic["sources"][changed[0]] != diagnostic["driver_sha256"]
            or diagnostic["numerical_objects_before"] != diagnostic["numerical_objects_after"]
            or diagnostic["link_inputs_before"] != diagnostic["link_inputs_after"]
            or diagnostic["link_inputs_before"] != original["link_inputs_before"]
            or diagnostic["link_inputs_before"] != original["link_inputs_after"]
            or not bench.same_compiler_policy(diagnostic["effective_compiler_flags"],
                                               original["effective_compiler_flags"])):
        raise ValueError("Diagnostic driver-only compilation binding changed")


def summarize():
    pilot, config, case, parent, parent_file, card_sha = inputs()
    complete, failures, comparisons, pilot_comparisons = {}, {}, [], []
    for report in sorted((REPORT / "runs").glob("*/*")):
        label = "/".join(report.parts[-2:])
        if (report / "failure.json").exists():
            failures[label] = json.loads((report / "failure.json").read_text())
            continue
        if not (report / "result.json").exists():
            continue
        record = json.loads((report / "result.json").read_text())
        points = record["point_record"]
        validate_binding(report, record, pilot, config, case, parent, parent_file, card_sha)
        actual = parse_forced_qp(report, record["backend"], points["inputs"]["count"])
        for row in actual["rows"]:
            row["point"] = points["source_point_indices"][row["point"]-1]
        if actual != record["observation"]:
            raise ValueError("Diagnostic summary differs from raw rows")
        complete[label] = record
    for (left_name, left), (right_name, right) in itertools.combinations(complete.items(), 2):
        if left["point_record"]["sha256"] != right["point_record"]["sha256"]:
            continue
        comparisons.append({"reference": left_name, "candidate": right_name,
            **bench.compare_rows(left["observation"]["rows"], right["observation"]["rows"])})
    for label, record in complete.items():
        if record["backend"] != 1:
            continue
        selected = set(record["point_record"]["source_point_indices"])
        for run_record in pilot["results"]:
            rows = [row for row in run_record["observation"]["rows"] if row["point"] in selected]
            pilot_comparisons.append({"qp_reference": label,
                "pilot": f"{run_record['source']}-{run_record['backend']}",
                **bench.compare_rows(record["observation"]["rows"], rows)})
    result = {"scope": "Independent fixed-QP adjudication; original target misses remain retained",
              "performance_eligible": False, "stability_certified": False,
              "original_target_miss_points": target_misses(pilot), "complete": sorted(complete),
              "failures": failures, "qp_comparisons": comparisons,
              "pilot_against_cuttools_qp": pilot_comparisons}
    bench.dump(REPORT / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run", "summarize"))
    parser.add_argument("--group", choices=("minimal", "target-misses"), default="minimal")
    parser.add_argument("--backend", type=int, choices=(1, 6, 9))
    parser.add_argument("--initialization", choices=("dp", "qp"), default="dp")
    parser.add_argument("--cpu", type=int, default=min(bench.os.sched_getaffinity(0)))
    parser.add_argument("--timeout", type=float, default=300.)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error(f"Run from {ROOT}")
    if args.cpu not in bench.os.sched_getaffinity(0):
        parser.error("CPU is outside allowed affinity")
    if args.phase == "prepare":
        _, _, _, points, record = freeze_subset(args.group)
        print(json.dumps({"points": str(points), "indices": record["source_point_indices"]}))
    elif args.phase == "run":
        if args.backend is None:
            parser.error("run requires --backend")
        run(args.group, args.backend, args.cpu, args.timeout, args.initialization)
    else:
        summary = summarize()
        print(json.dumps({key: summary[key] for key in ("complete", "failures", "scope")}))


if __name__ == "__main__":
    main()
