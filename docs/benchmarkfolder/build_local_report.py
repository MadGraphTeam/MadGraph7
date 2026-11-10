#!/usr/bin/env python3
"""Render a verified local measurement without touching the canonical report.

Run from the repository root after the local numerical preparation completes:
  python3 -B docs/benchmarkfolder/build_local_report.py \
      --measurement hs-local-1 --render-label render-1

Use --data-only to freeze the dossier and its hash ledger without invoking Typst.
Output is append-only: local-measurements/ID/render-N/benchmark.pdf. This command
does not install, compile a process, evaluate an amplitude, or register a local
measurement. It invokes the existing retained-data parsers and summarizer in the
launcher's verified private context. Complete numerical preparation is required;
pending, failed and ineligible timing remains visible. No overall qualification
claim follows from this report. oneloopmaster bd703d09a78f7849a6b86a07096e6f222fc01193
is pinned by the full Cargo Git dependency list verified by the local launcher.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark_local as local

bench = local.bench
SCHEMA = "madloop-tirust-local-report-v1"
BACKENDS = (1, 6, 7, 8, 9)
LABELS = ("tirust-first", *(f"backend-{i}" for i in BACKENDS))
ONELOOP_REVISION = "bd703d09a78f7849a6b86a07096e6f222fc01193"


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON in retained report input: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def record(path):
    return {"sha256": bench.digest(path), "bytes": Path(path).stat().st_size}


def tree(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob("*")):
        local.within(root, path.relative_to(root))
        if path.is_file() and "__pycache__" not in path.parts:
            files[path.relative_to(root).as_posix()] = record(path)
    return files


def fonts(environment):
    directories = [Path(name) for name in environment.get("TYPST_FONT_PATHS", "").split(os.pathsep) if name]
    files = {str(path.resolve()): record(path) for directory in directories for path in sorted(directory.rglob("*"))
             if path.is_file() and path.suffix.lower() in (".ttf", ".otf", ".ttc")}
    if not files:
        raise ValueError("Set TYPST_FONT_PATHS to Libertinus and DejaVu font directories; source tools/tirust_env.sh")
    return files


def required_speed(case, label):
    return case["role"] == "main" and label in ("tirust-first", "backend-1", "backend-6", "backend-7", "backend-8")


def numerical_rows(config):
    rows = []
    for case in config["cases"]:
        directory = bench.REPORT / "experiments/numerical" / ("ordinary-" + case["id"])
        lanes = []
        for source, backend in [("baseline", 1), *(("candidate", value) for value in BACKENDS)]:
            result = read(directory / f"{source}-{backend}" / "result.json")
            comparison, observation = result["comparison"], result["observation"]
            lanes.append({"source": source, "backend": backend, "status": result["status"],
                          "regression_pass": comparison["passed"],
                          "target_pass": comparison["ordinary_target_passed"],
                          "max_relative_error": comparison["maximum_relative_error"],
                          "backend_precision_counts": observation["backend_precision_counts"],
                          "cold_seconds": observation["cold_seconds"]})
        rows.append({**case, "points": 100, "regression_pass": all(r["regression_pass"] for r in lanes),
                     "ordinary_target_passed": all(r["target_pass"] for r in lanes), "lanes": lanes})
    return {"status": "complete", "measurement_phase": "steady-warmup",
            "performance_eligible": False, "cases": rows}


def attempt(case, directory, config):
    """Every attempt stays visible, even absent or invalid summary JSON."""
    guessed = re.sub(r"-attempt-[0-9]+$", "", directory.name)
    label = guessed
    result = {"case": case["id"], "label": label, "directory": directory.name,
              "status": "pending", "selected": False, "acceptance_eligible": None,
              "passed": None, "core_passed": None, "ratio": None, "lower_95": None,
              "upper_95": None, "pairs": None, "baseline_seconds_per_point": None,
              "candidate_seconds_per_point": None, "numerical_pass": None, "usage": {}, "error": None}
    if (directory / "failure.json").exists():
        result.update(status="failed", error=read(directory / "failure.json").get("error", "Retained measurement failure"))
    elif (directory / "summary.json").exists():
        try:
            declared = read(directory / "summary.json")
            label = declared["label"]
            if label not in LABELS:
                raise ValueError("Unrecognized local timing label")
            value, _ = bench.validate_retained_result(directory, label, case, config)
            stats = value["statistics"]
            eligible = value["acceptance_eligible"]
            status = ("failed" if not value["numerical_pass"] else "timing-ineligible" if not eligible
                      else "passed" if required_speed(case, label) and value["passed"]
                      else "failed" if required_speed(case, label) else "descriptive")
            result.update(label=label, status=status, acceptance_eligible=eligible,
                passed=value["passed"] if required_speed(case, label) else None,
                core_passed=value["passed"], ratio=stats["geometric_mean_ratio"],
                lower_95=stats["lower_95"], upper_95=stats["upper_95"], pairs=stats["pairs"],
                baseline_seconds_per_point=statistics.median(value["timings"]["baseline"]),
                candidate_seconds_per_point=statistics.median(value["timings"]["candidate"]),
                numerical_pass=value["numerical_pass"],
                usage={source: bench.aggregate_usage(value["raw_observations"][source]) for source in ("baseline", "candidate")})
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            result.update(status="invalid-evidence", error=str(error))
    if label in LABELS:
        result["selected"] = directory == bench.selected_run_directory(directory.parent, label)
    else:
        result.update(status="invalid-evidence", error=result["error"] or "Unrecognized local attempt directory")
    result["speed_required"] = required_speed(case, label)
    return result


def timing_rows(config, core):
    rows, attempts = [], []
    core_cases = {case["id"]: case for case in core["cases"]}
    for case in config["cases"]:
        base = bench.REPORT / "runs" / case["id"]
        existing = {}
        for directory in sorted(base.iterdir()) if base.exists() else []:
            if not directory.is_dir() or directory.name == "retry-requests":
                continue
            local.within(base, directory.name)
            item = attempt(case, directory, config)
            attempts.append(item)
            existing[directory.name] = item
        comparisons = []
        for label in LABELS:
            selected = bench.selected_run_directory(base, label)
            item = copy.deepcopy(existing.get(selected.name) or attempt(case, selected, config))
            item["selected"] = True
            error = core_cases[case["id"]].get("validation_errors", {}).get(label)
            if error:
                item.update(status="invalid-evidence", error=error, acceptance_eligible=False, passed=None,
                            ratio=None, lower_95=None, upper_95=None,
                            baseline_seconds_per_point=None, candidate_seconds_per_point=None)
            comparisons.append(item)
        rows.append({**case, "comparisons": comparisons,
                     "standalone_correctness": core_cases[case["id"]]["correctness"]})
    required = [item for case in rows if case["role"] == "main"
                for item in case["comparisons"] if item["speed_required"]]
    main_comparisons = [item for case in rows if case["role"] == "main" for item in case["comparisons"]]
    standalone = [item for case in rows if case["role"] == "main" for item in case["standalone_correctness"]]
    retained_failure = any(item["speed_required"] and item["acceptance_eligible"] is True
                           and item["passed"] is False for item in attempts)
    failed = (retained_failure or any(item["status"] in ("failed", "invalid-evidence", "timing-ineligible") for item in main_comparisons)
              or any(item.get("passed") is False or item.get("status") == "failed" for item in standalone))
    complete = (bool(main_comparisons) and all(item["acceptance_eligible"] is True for item in main_comparisons)
                and bool(standalone) and all(item["status"] == "complete" for item in standalone))
    passed = (complete and not failed and all(item["passed"] is True for item in required)
              and all(item["passed"] is True for item in standalone)
              and core["gate"]["consistent_candidate_source"] is True)
    return {"cases": rows, "attempts": attempts,
            "main_gate": {"status": "passed" if passed else "failed" if failed or complete else "pending",
                          "complete": complete, "passed": passed,
                          "scope": "main rescue chains, existing-backend regressions, timed standalone correctness; controls descriptive"}}


def historical_rows(value, measurement_root):
    result = []
    for item in value["origin"]["qualified_failures"]:
        failure = item["failure"]
        directory = item["report"] + "/" + failure["directory"] if item["report"] else failure["directory"]
        origin = measurement_root / "origin"
        summary = read(local.within(origin, directory + "/summary.json"))
        if summary.get("acceptance_eligible") is not True or summary.get("passed") is not False:
            raise ValueError("Canonical qualified failure was reclassified")
        stats = summary.get("statistics", {})
        result.append({"case": failure["case"], "label": failure["label"],
                       "predecessor_id": item["predecessor_id"], "directory": directory,
                       "ratio": stats.get("geometric_mean_ratio"), "lower_95": stats.get("lower_95"),
                       "upper_95": stats.get("upper_95"),
                       "evidence_sha256": bench.content_digest(failure["files"])})
    return result


def collect(measurement, value, measurement_root):
    proof = local.verify_numerical(value["cpu"])
    seal = measurement_root / "numerical-prerequisites.json"
    if not seal.is_file() or read(seal) != proof:
        raise ValueError("Complete numerical prerequisite seal is missing or changed")
    config = bench.load_config()
    core = bench.summarize(config)
    canonical = value["origin"]["manifest"]
    recipe = copy.deepcopy(value["native_recipe"])
    dependencies = recipe["git_dependencies"]
    if not any(item["name"] == "oneloop" and item["source"].endswith("#" + ONELOOP_REVISION) for item in dependencies):
        raise ValueError("Local report lacks the pinned oneloopmaster Git revision")
    patch_path = bench.ROOT / "vendor/tirust/patches/manifest.json"
    expected = value["source_identity"]["files"]["vendor/tirust/patches/manifest.json"]
    if bench.digest(patch_path) != expected:
        raise ValueError("Local report patch provenance differs from registered source")
    patch = read(patch_path)
    recipe.update(tirust_revision=patch["upstream_revision"], patch_sha256=patch["patch_sha256"])
    dossier = {"schema": SCHEMA, "id": measurement, "date": core["date"], "profile": "formal-local-7pairs",
        "overall_goal_qualified": False, "historical_results_replaced": False,
        "canonical": {"campaign": value["origin"]["campaign"],
                      "manifest_sha256": value["origin"]["manifest_sha256"],
                      "source_revision": value["source_identity"]["revision"],
                      "source_files_sha256": value["source_identity"]["files_sha256"],
                      "numerical_identity_sha256": canonical["candidate"]["numerical_identity_sha256"]},
        "deployment": {"native_files": value["local_deployment"]["native"]["files"],
                       "source_binding": value["native_source_binding"],
                       "host": value["registry_identity"]["host"], "cpu": value["cpu"], "recipe": recipe},
        "policy": {"points_per_case": config["points_per_case"], "repetitions": config["repetitions"],
                   "minimum_batch_seconds": config["minimum_batch_seconds"],
                   "performance_upper_ratio": config["performance_upper_ratio"],
                   "relative_tolerance": config["relative_tolerance"], "ordinary_target": 1e-8,
                   "oneloopmaster_revision": ONELOOP_REVISION},
        "numerical": numerical_rows(config), "timing": timing_rows(config, core),
        "historical_failures": historical_rows(value, measurement_root),
        "provenance": {"measurement_sha256": bench.digest(measurement_root / "measurement.json"),
                       "numerical_prerequisites_sha256": bench.digest(seal),
                       "core_summary_sha256": bench.digest(bench.REPORT / "summary.json")},
        "definitions": {"full100": "Complete final steady warmup stream; numerical evidence, no warmed timing",
                        "timed_standalone": "Separate cross-backend checks from timed standalone observations",
                        "usage": "Final accepted amplitude backend/precision; internal compensated retries are not enumerated",
                        "controls": "All control runtime ratios and standalone backend-9 ratios are descriptive",
                        "initialization": "cold_seconds includes initialization and warmup; excludes startup/SETPARA"}}
    return dossier, core


def build(measurement, render_label, data_only=False):
    if not re.fullmatch(r"render-[1-9][0-9]*", render_label):
        raise ValueError("Use an append-only render label such as render-1")
    measurement_root, _ = local.roots(measurement)
    destination = local.within(measurement_root, render_label)
    if destination.exists():
        raise ValueError("Render directories are append-only; choose a new render-N")
    with (measurement_root / ".execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with local.isolated(measurement) as value:
            destination.mkdir()
            ledger = {"schema": "madloop-tirust-local-report-build-v1", "measurement": measurement,
                      "render_label": render_label, "created_utc": bench.utc_now(), "status": "running",
                      "overall_goal_qualified": False, "pdf": None, "typst": None}
            try:
                dossier, core = collect(measurement, value, measurement_root)
                bench.dump(destination / "dossier.json", dossier)
                bench.dump(destination / "core-summary.json", core)
                for name in ("measurement.json", "numerical-prerequisites.json"):
                    shutil.copy2(measurement_root / name, destination / name)
                for name in ("local-report.typ", "style.typ", "build_local_report.py"):
                    shutil.copy2(HERE / name, destination / name)
                protocol_dir = destination / "protocol-sources"
                protocol_dir.mkdir()
                for name in local.PROTOCOLS:
                    shutil.copy2(bench.ROOT / "tools" / name, protocol_dir / name)
                inputs = {}
                for folder in ("origin", "report/inputs", "report/points", "report/experiments/numerical", "report/runs", "commands"):
                    for name, item in tree(measurement_root / folder).items():
                        inputs[folder + "/" + name] = item
                ledger.update(inputs=inputs, protocols=tree(protocol_dir),
                    document_inputs={name: record(destination / name) for name in
                        ("dossier.json", "core-summary.json", "measurement.json", "numerical-prerequisites.json",
                         "local-report.typ", "style.typ", "build_local_report.py")})
                if not data_only:
                    executable = shutil.which("typst")
                    if executable is None:
                        raise ValueError("Typst is unavailable; source tools/tirust_env.sh or use --data-only")
                    command = [executable, "compile", "--root", str(destination), "--input", "dossier=dossier.json",
                               "--ignore-system-fonts", "--ignore-embedded-fonts", "local-report.typ", "benchmark.pdf"]
                    env = dict(os.environ, SOURCE_DATE_EPOCH=str(core["source_date_epoch"]))
                    ledger["typst"] = {"executable": str(Path(executable).resolve()),
                        "executable_sha256": bench.digest(executable),
                        "version": subprocess.check_output([executable, "--version"], text=True).strip(),
                        "command": command, "source_date_epoch": core["source_date_epoch"],
                        "font_paths": env.get("TYPST_FONT_PATHS"), "font_files": fonts(env)}
                    with (destination / "typst.log").open("w") as log:
                        subprocess.run(command, cwd=destination, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                    if "unknown font family" in (destination / "typst.log").read_text().lower():
                        raise ValueError("Typst could not resolve a requested report font")
                    ledger["typst"]["log"] = record(destination / "typst.log")
                    ledger["pdf"] = record(destination / "benchmark.pdf")
                for name, item in inputs.items():
                    if record(local.within(measurement_root, name)) != item:
                        raise ValueError("Retained local report inputs changed while rendering")
                for name, item in ledger["document_inputs"].items():
                    if record(destination / name) != item:
                        raise ValueError("Frozen document inputs changed while rendering")
                if tree(protocol_dir) != ledger["protocols"]:
                    raise ValueError("Frozen protocol inputs changed while rendering")
                bench.campaign_guard("local-report-after")
                ledger["status"] = "data-only" if data_only else "rendered"
                ledger["finished_utc"] = bench.utc_now()
                bench.dump(destination / "build.json", ledger)
            except BaseException as error:
                ledger.update(status="failed", error=repr(error), finished_utc=bench.utc_now())
                bench.dump(destination / "build.json", ledger)
                raise
    return {"measurement": measurement, "directory": str(destination), "status": ledger["status"],
            "pdf": str(destination / "benchmark.pdf") if ledger["pdf"] else None,
            "ledger_sha256": bench.digest(destination / "build.json"), "overall_goal_qualified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--measurement", required=True)
    parser.add_argument("--render-label", required=True)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if Path.cwd() != bench.ROOT:
        parser.error(f"Run from {bench.ROOT}")
    print(json.dumps(build(args.measurement, args.render_label, args.data_only), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
