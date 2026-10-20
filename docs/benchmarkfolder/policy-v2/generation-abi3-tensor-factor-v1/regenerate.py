#!/usr/bin/env python3
"""Fresh CLI exports/builds for policy v2; never evaluate a loop amplitude."""
import concurrent.futures
import datetime
import difflib
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import traceback

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
OLD_REPORT = ROOT / "docs/benchmarkfolder"
OLD_WORK = ROOT / ".benchmark-work"
REPORT = Path(__file__).resolve().parent
WORK = OLD_WORK / "policy-v2/generation-abi3-tensor-factor-v1"
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import ninjars_benchmark as bench
from madgraph.various.ninjars_install import read_config

bench.WORK = WORK
bench.REPORT = REPORT
assert bench.ROOT == ROOT and bench.BASELINE == OLD_WORK / "original"
LOCK = threading.Lock()
SUMMARY = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return str(path.relative_to(ROOT))


def file_record(path):
    return {"path": relative(path), "sha256": digest(path),
            "bytes": path.stat().st_size}


def files_manifest(directory):
    return {str(p.relative_to(directory)): {"sha256": digest(p),
                                           "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file()}


def dependency_manifest():
    roots = [ROOT / "HEPTools/ninjars", bench.BASELINE / "HEPTools/ninja/lib",
             bench.BASELINE / "HEPTools/collier",
             ROOT / "vendor/CutTools/includects",
             bench.BASELINE / "vendor/CutTools/includects"]
    files = {}
    for directory in roots:
        if not directory.is_dir():
            continue
        for p in sorted(directory.rglob("*")):
            if p.is_file() and (p.suffix.lower() in
                               (".a", ".so", ".mod", ".h", ".json", ".txt", ".lock", ".toml")
                               or ".so." in p.name):
                files[relative(p)] = file_record(p)
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "vendor/ninjars"], cwd=ROOT).decode().split("\0")
    for name in filter(None, tracked):
        p = ROOT / name
        if p.is_file():
            files[name] = file_record(p)
    return files


def update(case_id, record):
    with LOCK:
        SUMMARY["cases"][case_id] = record
        bench.dump(REPORT / "status.json", SUMMARY)
    print(case_id, record["status"], flush=True)


def model_manifest(output):
    return {str(p.relative_to(output)): {"sha256": digest(p),
                                        "bytes": p.stat().st_size}
            for p in sorted((output / "Source/MODEL").rglob("*"))
            if p.is_file() and p.suffix.lower() in (".f", ".inc")}


def copy_logs(case_id):
    for stage in ("generate", "compile", "masses"):
        source = WORK / "logs" / f"{stage}-candidate-{case_id}.log"
        if source.exists():
            target = REPORT / "logs" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def process(case, config, frozen_state, preserved_original):
    case_id = case["id"]
    output = bench.output_dir("candidate", case)
    original = OLD_WORK / "outputs/baseline" / case_id
    record = {"status": "started", "case": case, "output": relative(output),
              "original_output": relative(original)}
    update(case_id, record)
    try:
        if bench.content_digest(bench.generator_tree_state()) != frozen_state:
            raise RuntimeError("Generator source changed since this series started")
        if output.exists():
            raise RuntimeError("Refuse to overwrite an existing v2 export")
        before_identity = bench.capture_generation_identity("candidate")
        record["identity_before_export"] = before_identity
        bench.generate("candidate", case)
        generated = bench.generated_source_manifest(output)
        bench.dump(REPORT / "manifests" / f"{case_id}-generated.json", generated)
        card = output / "Cards/param_card.dat"
        original_card = original / "Cards/param_card.dat"
        record["generated_default_parameter_card_sha256"] = digest(card)
        if card.read_bytes() != original_card.read_bytes():
            raise RuntimeError("Fresh parameter card differs from retained original")
        # Copy exact retained physical inputs, even though equality was verified.
        shutil.copy2(original_card, card)
        points = OLD_REPORT / "points" / f"{case_id}.dat"
        point_meta = points.with_suffix(".json")
        for p in (points, point_meta):
            target = REPORT / "points" / p.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        record["physical_inputs"] = {"param_card": file_record(card),
                                     "points": file_record(points),
                                     "point_metadata": file_record(point_meta)}
        before, after = model_manifest(original), model_manifest(output)
        if before.keys() != after.keys():
            raise RuntimeError("Generated model source inventory changed")
        changed = sorted(name for name in before if before[name] != after[name])
        allowed = {"Source/MODEL/model_functions.f",
                   "Source/MODEL/mp_intparam_definition.inc"}
        if not set(changed).issubset(allowed) or not changed:
            raise RuntimeError(f"Unexpected model correction delta: {changed}")
        correction_dir = REPORT / "model-deltas" / case_id
        retained = {}
        for name in changed:
            old, new = original / name, output / name
            a, b = correction_dir / "before" / name, correction_dir / "after" / name
            for source, target in ((old, a), (new, b)):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            patch = correction_dir / (Path(name).name + ".patch")
            patch.write_text("".join(difflib.unified_diff(
                old.read_text().splitlines(True), new.read_text().splitlines(True),
                fromfile="original/" + name, tofile="candidate/" + name)))
            retained[name] = {"before": file_record(a), "after": file_record(b),
                              "patch": file_record(patch)}
        delta = {"case": case_id, "before": before, "after": after,
                 "changed_files": changed, "retained": retained,
                 "scope": "Complete Source/MODEL .f/.inc inventory; exact source bytes."}
        bench.dump(correction_dir / "manifest.json", delta)
        record["model_delta"] = relative(correction_dir / "manifest.json")
        record["changed_model_files"] = changed
        record["generation"] = json.loads((output / "generation.json").read_text())
        update(case_id, {**record, "status": "building"})
        masses = bench.build("candidate", case)
        saved = json.loads(point_meta.read_text())
        if (saved.get("masses") != masses or saved["points"] != config["points_per_case"]
                or saved["sha256"] != digest(points)):
            raise RuntimeError("Retained kinematics metadata disagrees with generated masses")
        directory = bench.subprocess_dir("candidate", case)
        compilation = json.loads((output / "compilation.json").read_text())
        if compilation["build_kind"] != "clean-subprocess":
            raise RuntimeError("Fresh subprocess build unexpectedly reused object files")
        imports = subprocess.check_output(
            ["readelf", "-Ws", str(directory / "check")], text=True)
        quad_imports = [line.strip() for line in imports.splitlines()
                        if any(symbol in line for symbol in
                               ("__addtf3@", "__multf3@", "__gttf2@"))]
        if not any("GCC_" in line for line in quad_imports):
            raise RuntimeError("Executable does not import versioned GNU quad runtime")
        record.update(status="complete", compilation=compilation, masses=masses,
                      dimensions=(directory / "dimensions.dat").read_text(),
                      quad_runtime_imports=quad_imports,
                      loop_amplitude_evaluations=0,
                      initialized_symbolica_workers=0)
        if bench.generated_source_manifest(original) != preserved_original[case_id]:
            raise RuntimeError("Existing original generated sources changed")
        if bench.content_digest(bench.generator_tree_state()) != frozen_state:
            raise RuntimeError("Generator changed before the build completed")
    except Exception as error:
        record.update(status="failed", error=repr(error))
        target = REPORT / "logs" / f"failure-{case_id}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(traceback.format_exc())
        record["failure_log"] = relative(target)
    finally:
        copy_logs(case_id)
        update(case_id, record)


def main():
    os.chdir(ROOT)
    if (REPORT / "status.json").exists():
        raise RuntimeError("Refuse to overwrite this generation series")
    configs = [json.loads((OLD_REPORT / name).read_text())
               for name in ("cases.json", "controls.json")]
    cases = [(case, config) for config in configs for case in config["cases"]]
    if len({case["id"] for case, _ in cases}) != len(cases):
        raise RuntimeError("Duplicate core/control id")
    for name in ("cases.json", "controls.json"):
        shutil.copy2(OLD_REPORT / name, REPORT / name)
    # Final collection starts from committed numerical sources. Historical
    # experiments retain their own dirty-source identities separately.
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--",
                    *bench.GENERATOR_PATHS, "vendor/ninjars"], cwd=ROOT, check=True)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--",
         *bench.GENERATOR_PATHS, "vendor/ninjars"], cwd=ROOT, text=True)
    if untracked.strip():
        raise RuntimeError("Uncommitted numerical source additions remain")
    initial_state = bench.generator_tree_state()
    frozen_state = bench.content_digest(initial_state)
    libraries = dependency_manifest()
    installation = read_config(str(ROOT / "HEPTools/ninjars/lib"))
    if installation["abi_version"] != 3 or not installation["quad_precision"]:
        raise RuntimeError("Installed NinjaRS lacks genuine QP support")
    subprocess.run([sys.executable, "tools/ninjars_setup_references.py",
                    "--verify-model-only"], cwd=ROOT, check=True)
    model_path = bench.BASELINE / "models/loop_qcd_qed_sm_Gmu"
    ufo_files = {p.name: file_record(p) for p in sorted(model_path.glob("*.py"))}
    originals = {case["id"]: bench.generated_source_manifest(
        OLD_WORK / "outputs/baseline" / case["id"]) for case, _ in cases}
    retained = {name: files_manifest(OLD_REPORT / name)
                for name in ("inputs", "data", "points")}
    protected_candidates = {case["id"]: bench.generated_source_manifest(
        OLD_WORK / "outputs/candidate" / case["id"]) for case, _ in cases}
    SUMMARY.update(schema_version=1,
                   started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   git_revision=subprocess.check_output(
                       ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   generator_tree_state_sha256=frozen_state,
                   generator_tree_state=initial_state,
                   installation=installation, dependencies=libraries,
                   ufo_sources=ufo_files, jobs=2,
                   cases={case["id"]: {"status": "pending", "case": case}
                          for case, _ in cases},
                   scope="Actual CLI generation and clean subprocess builds only; --masses exits before loop evaluation.")
    bench.dump(REPORT / "status.json", SUMMARY)
    bench.dump(REPORT / "protected-inputs-before.json",
               {"original_sources": originals, "candidate_sources": protected_candidates,
                "retained_report_inputs": retained})
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(process, case, config, frozen_state, originals)
                   for case, config in cases]
        for future in concurrent.futures.as_completed(futures):
            future.result()
    unchanged = all(files_manifest(OLD_REPORT / name) == value
                    for name, value in retained.items())
    unchanged &= all(bench.generated_source_manifest(
        OLD_WORK / "outputs/candidate" / case_id) == value
        for case_id, value in protected_candidates.items())
    unchanged &= all(bench.generated_source_manifest(
        OLD_WORK / "outputs/baseline" / case_id) == value
        for case_id, value in originals.items())
    dependencies_unchanged = dependency_manifest() == libraries
    SUMMARY.update(complete=all(record["status"] == "complete"
                               for record in SUMMARY["cases"].values()),
                   preserved_existing_inputs=unchanged,
                   dependencies_unchanged=dependencies_unchanged,
                   script_sha256=digest(Path(__file__)),
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    bench.dump(REPORT / "status.json", SUMMARY)
    if not SUMMARY["complete"] or not unchanged or not dependencies_unchanged:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
