#!/usr/bin/env python3
"""Isolated final-source rotations, mass/scale rebinding and singular controls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
sys.path[:0] = [str(ROOT / "tools"), str(ROOT)]
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_references_v2 as adapter
import ninjars_benchmark_workers as workers
import controls_driver as controls

REPORT = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/policy-v2/final-variant-controls-both-precision"
ORIGINAL_REPORT = ROOT / "docs/benchmarkfolder"
GENERATION = ORIGINAL_REPORT / "policy-v2/generation-abi3-terminal-fixed"
PARENTS = ROOT / ".benchmark-work/policy-v2/generation-abi3-terminal-fixed/outputs/candidate"
bench.WORK, bench.REPORT = WORK, REPORT
controls.CONFIG, controls.VARIANTS = REPORT / "controls.json", REPORT / "control-variants.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    if path.exists():
        raise RuntimeError("Refuse to overwrite retained evidence: " + str(path))
    bench.dump(path, value)


def verify_parents(prepared):
    for name, record in prepared["parents"].items():
        parent = PARENTS / name
        if (bench.generated_source_manifest(parent) != record["sources"]
                or policy.library_hashes(parent) != record["libraries"]
                or sha(parent / "Cards/param_card.dat") != record["physical_card"]):
            raise RuntimeError("Parent source, runtime or physical card changed")


def writable_clone(output):
    """Reject external aliases before the reused build/configuration mutates."""
    output = output.resolve()
    paths = [output / name for name in (
        "Source", "SubProcesses", "Cards", "compilation.json",
        "compiled-sources.json", "compilation-history", "Cards/MadLoopParams.dat")]
    for directory in output.glob("SubProcesses/P*_*"):
        paths.extend(directory / name for name in (
            "check", "check_sa.f", "check_sa.original", "dimensions.dat", "masses.dat",
            "timing.dat", "values.dat", "timed_status.dat"))
        # Imported module symlinks (e.g. CutTools mpmodule.mod) are read-only
        # inputs. Only locally declared MODULEs can be written by this build.
        local_modules = {
            name.lower() for source in directory.iterdir()
            if source.is_file() and source.suffix.lower() in (".f", ".f90", ".inc")
            for name in re.findall(
                r"(?im)^\s*MODULE\s+(?!PROCEDURE\b|FUNCTION\b|SUBROUTINE\b)(\w+)",
                source.read_text())}
        paths.extend(p for p in directory.rglob("*") if
                     p.suffix in (".o", ".smod") or
                     (p.suffix == ".mod" and
                      (not p.is_symlink() or p.stem.lower() in local_modules)))
    paths.extend((output / "SubProcesses").rglob("MadLoopParams.dat"))
    paths.extend((output / "compilation-history").rglob("*"))
    paths.extend((output / "SubProcesses").rglob("*HelFilter.dat"))
    paths.extend((output / "SubProcesses").rglob("*LoopFilter.dat"))
    if any(not path.resolve().is_relative_to(output) for path in paths):
        raise RuntimeError("Writable generated path escapes its isolated clone")


def clone(parent, destination):
    output = adapter.clone_output(parent, destination)
    writable_clone(output)
    return output


def prepare():
    if WORK.exists() or (REPORT / "prepared.json").exists():
        raise RuntimeError("Use a new control namespace")
    summary = policy.read(GENERATION / "summary.json")
    if not summary["complete"] or len(summary["cases"]) != 13:
        raise RuntimeError("Complete final CLI generation is required")
    WORK.mkdir(parents=True)
    (REPORT / "points").mkdir()
    shutil.copy2(ORIGINAL_REPORT / "policy-v2/controls.json", controls.CONFIG)
    config = policy.read(controls.CONFIG)
    prepared = {
        "schema": "final-variant-controls-both-precision-preparation-v1",
        "generation": policy.bound_file(GENERATION / "summary.json"),
        "generator_state": bench.generator_tree_state(),
        "sources": {name: policy.bound_file(ROOT / "tools" / name)
                    for name in ("ninjars_controls.py", "ninjars_benchmark.py",
                                 "ninjars_benchmark_driver.f")},
        "parents": {}, "variants": {}, "acceptance_eligible": False,
        "script_sha256": sha(Path(__file__)),
        "control_driver": policy.bound_file(REPORT / "controls_driver.py"),
    }
    for name in ("control-gg-hh", "control-forced-qp"):
        parent = PARENTS / name
        prepared["parents"][name] = {
            "sources": bench.generated_source_manifest(parent),
            "libraries": policy.library_hashes(parent),
            "physical_card": sha(parent / "Cards/param_card.dat"),
            "generation": policy.read(parent / "generation.json")}
        clone(parent, WORK / "outputs/candidate" / name)
        for suffix in (".dat", ".json"):
            shutil.copy2(ORIGINAL_REPORT / "points" / (name + suffix),
                         REPORT / "points" / (name + suffix))
    # Reuse the previously exercised transformations and Fortran state-update
    # driver. Exact retained point bytes are checked below, not regenerated seeds.
    original_clone = controls.clone_output
    controls.clone_output = lambda parent, case: clone(
        bench.output_dir("candidate", parent), bench.output_dir("candidate", case))
    try:
        controls.prepare()
    finally:
        controls.clone_output = original_clone
    for case in policy.read(controls.VARIANTS)["cases"]:
        name = case["id"]
        points = REPORT / "points" / (name + ".dat")
        if points.read_bytes() != (ORIGINAL_REPORT / "points" / points.name).read_bytes():
            raise RuntimeError("Transformed input stream changed: " + name)
        output = bench.output_dir("candidate", case)
        directory = bench.subprocess_dir("candidate", case)
        manifest = bench.generated_source_manifest(output)
        parent = prepared["parents"][case["parent_control"]]
        changes = policy.delta(parent["sources"]["files"], manifest["files"])
        allowed = {str((directory / "check_sa.f").relative_to(output))}
        if not changes.issubset(allowed):
            raise RuntimeError("Control changed numerical process source: " + repr(changes))
        if policy.library_hashes(output) != parent["libraries"]:
            raise RuntimeError("Control changed numerical libraries")
        if sha(output / "Cards/param_card.dat") != parent["physical_card"]:
            raise RuntimeError("Control changed the physical card")
        driver = directory / "check_sa.f"
        if name.startswith("control-rebinding"):
            if driver.read_text().count("CALL CONTROL_STATE(") != 4:
                raise RuntimeError("State updates do not cover every driver call")
        retained = REPORT / "drivers" / (name + ".f")
        retained.parent.mkdir(exist_ok=True)
        shutil.copy2(driver, retained)
        prepared["variants"][name] = {
            "case": case, "sources": manifest,
            "libraries": policy.library_hashes(output),
            "points": policy.bound_file(points), "driver": policy.bound_file(retained),
            "executable_sha256": sha(directory / "check"),
            "compilation": policy.read(output / "compilation.json")}
    verify_parents(prepared)
    dump(REPORT / "prepared.json", policy.seal(prepared))
    print("Prepared four isolated controls; original point bytes and sources verified.")


def run(selected, cpu):
    prepared = policy.verified(policy.read(REPORT / "prepared.json"),
                               "final-variant-controls-both-precision-preparation-v1")
    if sha(Path(__file__)) != prepared["script_sha256"]:
        raise RuntimeError("Control script changed after preparation")
    policy.check_file(prepared["control_driver"])
    for record in prepared["sources"].values():
        policy.check_file(record)
    verify_parents(prepared)
    config = policy.read(controls.VARIANTS)
    definitions = {case["id"]: case for case in config["cases"]}
    if not selected or not set(selected).issubset(definitions) or len(set(selected)) != len(selected):
        raise RuntimeError("Unknown, empty or duplicate control selection")
    for name in selected:
        case = definitions[name]
        directory = bench.subprocess_dir("candidate", case)
        identity = prepared["variants"][name]
        backends = (8,) if name.endswith("-fresh") else (
            (1, 6, 8) if name == "control-near-singular" else (1, 6, 7, 8))
        results = {}
        for backend in backends:
            path = REPORT / "data" / f"candidate-{name}-{backend}-0.json"
            raw = REPORT / "raw" / name / str(backend)
            if path.exists() or raw.exists():
                raise RuntimeError("Preserve existing control attempt")
            writable_clone(bench.output_dir("candidate", case))
            if (sha(directory / "check") != identity["executable_sha256"]
                    or bench.generated_source_manifest(bench.output_dir("candidate", case)) != identity["sources"]
                    or policy.library_hashes(bench.output_dir("candidate", case)) != identity["libraries"]):
                raise RuntimeError("Prepared control source/runtime changed")
            raw.mkdir(parents=True)
            for filename in (*workers.OUTPUTS, "dimensions.dat"):
                (directory / filename).unlink(missing_ok=True)
            try:
                results[f"candidate-{backend}"] = controls.measure_one(case, backend, config, cpu)
            finally:
                for filename in (*workers.OUTPUTS, "dimensions.dat"):
                    if (directory / filename).is_file():
                        shutil.copy2(directory / filename, raw / filename)
                log = WORK / "logs" / f"run-candidate-{name}-{backend}-0.log"
                if log.is_file():
                    shutil.copy2(log, raw / "run.log")
            if (sha(directory / "check") != identity["executable_sha256"]
                    or bench.generated_source_manifest(bench.output_dir("candidate", case)) != identity["sources"]
                    or policy.library_hashes(bench.output_dir("candidate", case)) != identity["libraries"]):
                raise RuntimeError("Control source/runtime changed during evaluation")
        if len(results) > 1:
            if not controls.compare_records(name, results, config,
                                           require_rescue=name == "control-near-singular"):
                raise RuntimeError("Control comparison failed: " + name)
        if name.endswith("-fresh"):
            retained = policy.read(REPORT / "data/candidate-control-rebinding-8-0.json")
            if not controls.compare_records("rebinding-context-reset",
                    {"retained-8": retained, "fresh-8": results["candidate-8"]}, config):
                raise RuntimeError("Reused and fresh NinjaRS contexts disagree")
        verify_parents(prepared)
        print(name, "completed", flush=True)


def compare_rotation(binding_path):
    """Compare to the separately collected final unrotated ggHH worker."""
    correction = policy.validate_correction(ORIGINAL_REPORT / "policy-v2/model-correction.json")
    binding = policy.checked_bindings(binding_path, correction)
    prepared = policy.verified(policy.read(REPORT / "prepared.json"),
                               "final-variant-controls-both-precision-preparation-v1")
    parent = prepared["parents"]["control-gg-hh"]
    config = policy.read(controls.VARIANTS)
    if (REPORT / "rotation-reference.json").exists():
        raise RuntimeError("Preserve the previous rotation comparison")
    references = {}
    for backend in (1, 6, 7, 8):
        key = f"control-gg-hh-candidate-{backend}"
        target = binding["targets"][key]
        if (target["role"] != "candidate_corrected"
                or target["generation_source_identity"] != parent["generation"]["source_identity"]
                or target["libraries"] != parent["libraries"]):
            raise RuntimeError("Unrotated reference has another source/runtime")
        changes = policy.delta(parent["sources"]["files"], target["compiled_manifest"]["files"])
        drivers = {name for name in parent["sources"]["files"]
                   if name.startswith("SubProcesses/P") and name.endswith("/check_sa.f")}
        if not changes.issubset(drivers):
            raise RuntimeError("Unrotated reference changes numerical process source")
        record_path = policy.path(binding["report"]) / "data" / (key + "-0.json")
        if binding["run_files_sha256"].get(policy.relative(record_path)) != sha(record_path):
            raise RuntimeError("Unrotated measurement is not bound")
        original = policy.read(record_path)
        rotated = policy.read(REPORT / "data" / f"candidate-control-rotation-{backend}-0.json")
        if original["parameter_card_sha256"] != rotated["parameter_card_sha256"]:
            raise RuntimeError("Rotation comparison changes the physical card")
        if not controls.compare_records(f"rotation-invariance-{backend}",
                {"original": original, "rotated": rotated}, config):
            raise RuntimeError("Rotation invariance failed")
        references[str(backend)] = policy.bound_file(record_path)
    dump(REPORT / "rotation-reference.json", {
        "binding": policy.bound_file(binding_path), "references": references,
        "passed": True, "acceptance_eligible": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "compare-rotation"))
    parser.add_argument("--cases", default="control-rotation,control-rebinding,control-rebinding-fresh,control-near-singular")
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--binding", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "run":
        run(args.cases.split(","), args.cpu)
    else:
        if args.binding is None:
            parser.error("compare-rotation requires --binding")
        compare_rotation(args.binding)
