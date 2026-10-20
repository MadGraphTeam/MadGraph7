#!/usr/bin/env python3
"""Render a validated policy-v2 dataset without overwriting historical reports."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers
import ninjars_profile_v2 as profiler

ROOT = bench.ROOT
REPORT = ROOT / "docs/benchmarkfolder"
BACKENDS = (1, 6, 7, 8)
# Keep PDF metadata reproducible outside the repository's Nix environment too.
# The actual snapshot date is recorded separately in generated_utc.
PDF_CREATION_TIMESTAMP = 315532800
BASELINE_REVISION = "27147b3300f556a75661a886e92bdbe00558ee25"


def compare(groups, pairs=None):
    """Recompute every comparison independently of the publication summary."""
    keys = list(groups)
    if not keys:
        raise ValueError("Empty numerical cohort")
    stream = [(r["point"], r["order"]) for r in groups[keys[0]]]
    if not stream or len(stream) != len(set(stream)):
        raise ValueError("Empty or duplicated point/order stream")
    scales = defaultdict(float)
    for rows in groups.values():
        if [(r["point"], r["order"]) for r in rows] != stream:
            raise ValueError("Mismatched point/order streams")
        for row in rows:
            if len(row["coefficients"]) != 4 or not all(
                    math.isfinite(x) for x in row["coefficients"]):
                raise ValueError("Invalid Born/Laurent coefficients")
            scales[row["point"]] = max(scales[row["point"]],
                                       *map(abs, row["coefficients"]))
    pairs = list(itertools.combinations(keys, 2) if pairs is None else pairs)
    failures, worst, checks = [], 0., 0
    for left, right in pairs:
        for a, b in zip(groups[left], groups[right]):
            for coefficient, (x, y) in enumerate(zip(a["coefficients"], b["coefficients"])):
                bound = 1e-7 * max(abs(x), abs(y)) + 1e-10 * scales[a["point"]]
                error = abs(x-y)
                fraction = error/bound if bound else 0.
                worst = max(worst, fraction)
                checks += 1
                if error > bound:
                    failures.append({"targets": [left, right], "point": a["point"],
                                     "order": a["order"], "coefficient": coefficient,
                                     "values": [x, y], "tolerance": bound})
    return {"passed": not failures, "pairs": len(pairs), "failures": failures,
            "status": "failed" if failures else "passed",
            "points": len({p for p, _ in stream}),
            "orders": len({o for _, o in stream}), "comparisons": checks,
            "max_tolerance_fraction": worst}


def estimate_distribution(values):
    """Describe MadLoop's estimates without interpreting them as true errors.

    The 95th percentile is the nearest-rank order statistic. In particular,
    an exact zero estimate stays zero; it is never converted to infinite digits
    of demonstrated accuracy.
    """
    values = sorted(values)
    if not values or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
                         for x in values):
        raise ValueError("Invalid or missing MadLoop stability estimate")
    return {"count": len(values), "minimum": values[0],
            "median": statistics.median(values),
            "p95_nearest_rank": values[math.ceil(.95 * len(values))-1],
            "maximum": values[-1], "zero_count": values.count(0.)}


def stability_snapshot(groups):
    """Separate observed disagreements, reported estimates and actual rescues.

    These finite-stream comparisons have no analytic truth reference. Each
    target is compared with every *other* corrected target, using the common
    point scale from all corrected targets and orders. Status counts describe
    one evaluation per point, while estimates retain every point/order slot.
    """
    compare(groups)  # Validate aligned, finite and nonduplicated coefficient rows.
    if len(groups) < 2:
        raise ValueError("Stability comparison needs independent target streams")
    scales = defaultdict(float)
    for rows in groups.values():
        for row in rows:
            scales[row["point"]] = max(scales[row["point"]],
                                       *map(abs, row["coefficients"]))
    targets = {}
    for name, rows in groups.items():
        role, backend_text = name.rsplit("/", 1)
        backend = int(backend_text)
        if role not in policy.ROLES or backend not in policy.ROLES[role]:
            raise ValueError("Unknown stability source role/backend")
        if role == "original":
            raise ValueError("Uncorrected original cannot define corrected stability")
        statuses, by_point = Counter(), {}
        for row in rows:
            bench.validate_status(row["status"], backend)
            previous = by_point.setdefault(row["point"], row["status"])
            if previous != row["status"]:
                raise ValueError("Point status differs between split-order slots")
            if row["order"] == 0:
                statuses[str(row["status"])] += 1
        if sum(statuses.values()) != len(by_point):
            raise ValueError("Missing total-order stability slot")
        all_estimates = estimate_distribution([row["accuracy"] for row in rows])
        total_estimates = estimate_distribution([row["accuracy"] for row in rows
                                                 if row["order"] == 0])
        comparisons = failures = 0
        maximum = absolute = scaled = 0.
        worst = None
        for other, reference in groups.items():
            if other == name:
                continue
            for a, b in zip(rows, reference):
                for coefficient, (x, y) in enumerate(zip(a["coefficients"], b["coefficients"])):
                    error = abs(x-y)
                    bound = 1e-7 * max(abs(x), abs(y)) + 1e-10 * scales[a["point"]]
                    fraction = error / bound if bound else 0.
                    point_scaled = error / scales[a["point"]] if scales[a["point"]] else 0.
                    comparisons += 1
                    failures += error > bound
                    absolute = max(absolute, error)
                    scaled = max(scaled, point_scaled)
                    if worst is None or fraction > maximum:
                        maximum = fraction
                        worst = {"reference": other, "point": a["point"],
                                 "order": a["order"], "coefficient": coefficient,
                                 "values": [x, y], "absolute_error": error,
                                 "tolerance": bound, "tolerance_fraction": fraction}
        targets[name] = {
            "estimated_accuracy": {"interpretation": "MadLoop estimate; not an observed error",
                                   "all_orders": all_estimates, "total_order": total_estimates},
            "observed_discrepancy": {
                "interpretation": "Pairwise corrected-cohort discrepancy; not an analytic error bound",
                "reference_targets": [other for other in groups if other != name],
                "comparisons": comparisons, "failures": failures,
                "max_tolerance_fraction": maximum, "max_absolute_error": absolute,
                "max_point_scaled_error": scaled, "worst": worst},
            "status_counts": dict(sorted(statuses.items())), "points": len(by_point),
            "dp": sum(v for k, v in statuses.items() if int(k)//100 == 2),
            "qp": sum(v for k, v in statuses.items() if int(k)//100 == 3),
        }
    return {"status": "measured", "estimate_is_observed_error": False,
            "comparison_scope": "all corrected-cohort targets, every point/order/coefficient",
            "targets": targets}


def summary(runs, points, minimum):
    """Summarize already structurally validated records, retaining real usage."""
    if not runs:
        return {"status": "pending", "repetitions": 0, "median_us": None,
                "cold_seconds": None, "dp": None, "qp": None, "timed_calls": None,
                "status_counts": {}, "point_status_counts": {}, "cpus": [],
                "minimum_us": None, "maximum_us": None, "mad_us": None, "ratio": None}
    if [r["repetition"] for r in runs] != list(range(len(runs))) or len(runs) < minimum:
        raise ValueError("Incomplete timing/reference repetitions")
    values, cold, counts, point_counts = [], [], Counter(), Counter()
    for run in runs:
        for key in ("seconds_per_point", "cold_seconds"):
            if not math.isfinite(run[key]) or run[key] <= 0:
                raise ValueError("Invalid measured time")
        if sum(run["timed_status_counts"].values()) != run["loops"] * points:
            raise ValueError("Incomplete timed status histogram")
        for code, count in run["timed_status_counts"].items():
            if type(count) is not int or count < 0:
                raise ValueError("Invalid timed status count")
            bench.validate_status(int(code), run["backend"])
        numerical_counts = Counter(str(row["status"]) for row in run["rows"] if row["order"] == 0)
        if sum(numerical_counts.values()) != points or dict(numerical_counts) != run["status_counts"]:
            raise ValueError("Point status histogram differs from numerical rows")
        for code in numerical_counts:
            bench.validate_status(int(code), run["backend"])
        values.append(run["seconds_per_point"] * 1e6)
        cold.append(run["cold_seconds"])
        counts.update(run["timed_status_counts"])
        point_counts.update(numerical_counts)
    median = statistics.median(values)
    return {"status": "measured", "repetitions": len(runs), "median_us": median,
            "cold_seconds": statistics.median(cold),
            "dp": sum(v for k, v in counts.items() if int(k)//100 == 2),
            "qp": sum(v for k, v in counts.items() if int(k)//100 == 3),
            "timed_calls": sum(counts.values()), "status_counts": dict(sorted(counts.items())),
            "point_status_counts": dict(sorted(point_counts.items())),
            "cpus": sorted({r["cpu"] for r in runs}), "minimum_us": min(values),
            "maximum_us": max(values),
            "mad_us": statistics.median(abs(v-median) for v in values), "ratio": None,
            "executable_sha256": sorted({r["executable_sha256"] for r in runs})}


def compare_recorded(actual, recorded):
    """Publication booleans and mismatch records must agree with raw data."""
    if {k: actual[k] for k in ("passed", "pairs", "failures")} != recorded:
        raise ValueError("Stale or forged numerical dataset summary")


def generation_for_build(record, build):
    """A historical model-correction export is not a later measured export."""
    if record.get("source_identity") != build["generation_source_identity"]:
        return None
    if record.get("parameter_card_sha256") != build["parameter_card_sha256"]:
        raise ValueError("Generation parameter card differs from measured build")
    seconds = record.get("seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Invalid generation duration")
    return record


def profile_snapshot(specification, dataset, bindings, correction, inputs):
    """Check diagnostic samples against the exact final measured worker."""
    if specification is None:
        return []

    def retain(filename, checksum=None):
        filename = policy.path(filename)
        actual = workers.digest(filename)
        if checksum is not None and checksum != actual:
            raise ValueError("Changed profile evidence: " + policy.relative(filename))
        inputs[policy.relative(filename)] = actual
        return filename

    def bound(record):
        return retain(record["path"], record["sha256"])

    spec = policy.verified(policy.read(retain(specification)), "profile-report-evidence-v2")
    profiles, seen = [], set()
    for entry in spec["profiles"]:
        filename = bound(entry["profile"])
        if filename in seen:
            raise ValueError("Duplicate profile")
        seen.add(filename)
        record = policy.read(filename)
        if (record.get("schema") != "madloop-sampled-profile-v2"
                or record.get("complete") is not True
                or record.get("acceptance_eligible") is not False
                or record.get("every_physical_row_exact") is not True
                or record.get("original_worker_unchanged") is not True):
            raise ValueError("Incomplete or acceptance-labelled diagnostic profile")
        measurement = bound(record["measurement"])
        if dataset["measurements_sha256"].get(policy.relative(measurement)) != workers.digest(measurement):
            raise ValueError("Profile does not refer to this dataset's measurement")
        reference = policy.read(measurement)
        key = f"{reference['case']}-{reference['source']}-{reference['backend']}"
        if key != record["target"] or reference["repetition"] < 0:
            raise ValueError("Profile target differs from measured target")
        source_binding = policy.checked_bindings(bound(record["binding"]), correction)
        target = source_binding["targets"][key]
        matches = [b["targets"][key] for b in bindings
                   if key in b["targets"] and b["targets"][key]["role"] == target["role"]]
        fields = ("role", "case", "backend", "compiled_manifest", "generation_source_identity",
                  "libraries", "executable_sha256", "model_identity_sha256")
        if (len(matches) != 1 or any(target[field] != matches[0][field] for field in fields)
                or policy.reducer_signature(target) != policy.reducer_signature(matches[0])):
            raise ValueError("Profile source/runtime differs from this dataset")
        if (source_binding["run_files_sha256"].get(policy.relative(measurement)) != workers.digest(measurement)
                or record["executable_sha256"] != target["executable_sha256"]
                or record["generation_source_identity"] != target["generation_source_identity"]
                or record["compiled_sources_sha256"] != target["compiled_manifest"]["files_sha256"]
                or record["libraries"] != target["libraries"]
                or record["reducer_runtime"] != policy.reducer_signature(target)):
            raise ValueError("Profile worker identity changed")
        directory = filename.parent
        retain(directory / "ninjars_profile_v2.py", record["script_sha256"])
        points = bound(record["points"])
        retain(points, reference["points_sha256"])
        for name, field in (("param_card.dat", "parameter_card_sha256"),
                            ("MadLoopParams.dat", "madloop_card_sha256")):
            if record[field] != reference[field]:
                raise ValueError("Profile card differs from measured card")
            retain(directory / name, reference[field])
        point_count = int(points.read_text().split()[0])
        if point_count != len({row["point"] for row in reference["rows"]}):
            raise ValueError("Profile point stream is incomplete")
        evaluations, usage = 0, Counter()
        if not record["blocks"]:
            raise ValueError("Profile has no complete blocks")
        for index, block in enumerate(record["blocks"]):
            block_dir = directory / f"block-{index}"
            for name in (*workers.OUTPUTS, "dimensions.dat"):
                retain(block_dir / name)
            count = sum(block["timed_status_counts"].values())
            if count <= 0 or count % point_count:
                raise ValueError("Incomplete profile stream traversal")
            parsed = workers.parse_outputs(block_dir, target["backend"], point_count, count // point_count)
            if parsed != block or parsed["rows"] != reference["rows"]:
                raise ValueError("Profile physical rows, accuracy or status changed")
            evaluations += count
            usage.update(parsed["timed_status_counts"])
        partition = record["partition"]
        retain(bound(entry["raw_perf"]), partition["raw_perf_sha256"])
        stacks = bound(partition["stacks"])
        counts = Counter()
        for block in stacks.read_text().split("\n\n"):
            lines = [line for line in block.splitlines()
                     if line.strip() and not line.startswith("#")]
            if lines:
                counts[profiler.category(lines)] += 1
        samples = sum(counts.values())
        categories = {name: {"samples": count, "percent": 100 * count / samples}
                      for name, count in sorted(counts.items())} if samples else {}
        if samples <= 0 or samples != partition["samples"] or categories != partition["categories"]:
            raise ValueError("Profile sample classification changed")
        profiles.append({
            "case": target["case"], "role": target["role"], "backend": target["backend"],
            "profile": policy.relative(filename), "points": point_count,
            "evaluations": evaluations, "blocks": len(record["blocks"]), "cpu": record["cpu"],
            "samples": samples, "categories": categories, "status_counts": dict(usage),
            "executable_sha256": target["executable_sha256"],
            "reducer_runtime_identity": target["reducer_runtime"]["identity_sha256"],
        })
    return profiles


def candidate_runtime_signature(bindings):
    signatures = {}
    for binding in bindings:
        for item in binding["targets"].values():
            if item["role"] == "candidate_corrected" and item["backend"] == 8:
                signature = policy.reducer_signature(item)
                signatures[signature["identity_sha256"]] = signature
    if len(signatures) != 1:
        raise ValueError("Supplement requires one bound candidate generator and NinjaRS runtime")
    return next(iter(signatures.values()))


def supplement_expectations(dataset, bindings):
    """Require the supplement to evaluate the same final candidate as the core."""
    candidates = [item for binding in bindings for item in binding["targets"].values()
                  if item["role"] == "candidate_corrected"]
    generators = {item["generation_source_identity"].get("tree_state_sha256")
                  for item in candidates}
    runtime = candidate_runtime_signature(bindings)
    if len(generators) != 1 or None in generators:
        raise ValueError("Supplement requires one bound candidate generator and NinjaRS runtime")
    return {"core_dataset_identity_sha256": dataset["identity_sha256"],
            "literal_baseline_revision": BASELINE_REVISION,
            "candidate_generator_identity_sha256": next(iter(generators)),
            "candidate_runtime_identity_sha256": runtime["identity_sha256"]}


def supplement_snapshot(specification, dataset, bindings, inputs):
    if specification is None:
        return {"status": "pending", "complete_campaign_passed": False,
                "cases": [], "threshold_grid": None, "precision_controls": None}
    # The supplement collector is deliberately separate from the frozen core
    # protocol. Its reader must revalidate every raw binding and recompute all
    # conclusions before any supplemental result reaches the report.
    import ninjars_supplement_v2 as supplement
    expected = supplement_expectations(dataset, bindings)
    result = supplement.report_snapshot(specification, expected, inputs,
                                        runtime_signature=candidate_runtime_signature(bindings))
    if (result.get("schema") != "madloop-final-supplement-v1"
            or result.get("status") not in ("passed", "failed", "unresolved", "pending")
            or type(result.get("complete_campaign_passed")) is not bool
            or any(result.get(key) != value for key, value in expected.items())):
        raise ValueError("Invalid or differently bound final supplement")
    if result["complete_campaign_passed"] and result["status"] != "passed":
        raise ValueError("Contradictory supplement campaign conclusion")
    inputs[policy.relative(specification)] = workers.digest(policy.path(specification))
    return result


def stress_snapshot(specification, dataset, bindings, inputs):
    """Additional fixed two-to-four correctness controls; timings are diagnostic."""
    if specification is None:
        return {"status": "pending", "complete_campaign_passed": False, "cases": []}
    import ninjars_stress_v2 as stress
    expected = supplement_expectations(dataset, bindings)
    result = stress.report_snapshot(specification, expected, inputs,
                                   runtime_signature=candidate_runtime_signature(bindings))
    if (result.get("schema") != "madloop-final-stress-v1"
            or result.get("status") not in ("passed", "failed", "unresolved", "pending")
            or type(result.get("complete_campaign_passed")) is not bool
            or any(result.get(key) != value for key, value in expected.items())):
        raise ValueError("Invalid or differently bound final stress evidence")
    if result["complete_campaign_passed"] and result["status"] != "passed":
        raise ValueError("Contradictory stress campaign conclusion")
    inputs[policy.relative(specification)] = workers.digest(policy.path(specification))
    return result


def runtime_provenance_snapshot(specification, dataset, bindings, inputs):
    """Read runtime metadata from measured targets, never historical setup pins.

    Dependency revisions additionally need retained installed Cargo files and a
    successful installation record whose archive hash is the measured archive.
    A live checkout's current Cargo.toml cannot identify an older linked library.
    """
    candidates = [item for binding in bindings for item in binding["targets"].values()
                  if item["role"] == "candidate_corrected" and item["backend"] == 8]
    configs, archives = {}, set()
    source_roles = defaultdict(dict)
    for binding in bindings:
        for item in binding["targets"].values():
            identity = item.get("generation_source_identity")
            if identity is not None:
                source = {key: identity[key] for key in
                          ("revision", "tree_state_sha256", "dirty_patch_sha256", "scope")
                          if key in identity}
                source_roles[item["role"]][bench.content_digest(source)] = source
    for item in candidates:
        archives.update(value for name, value in item.get("libraries", {}).items()
                        if Path(name).name == "libninjars.a")
        for name, record in item.get("runtime_metadata", {}).items():
            if Path(name).name != "ninjars-config.json":
                continue
            filename = policy.check_file(record)
            inputs[policy.relative(filename)] = workers.digest(filename)
            configs[record["sha256"]] = policy.read(filename)
    versions = {value.get("abi_version") for value in configs.values()}
    if len(versions) > 1 or (versions and versions != {3}):
        raise ValueError("Measured NinjaRS runtime is not consistently ABI 3")
    result = {"status": "pending", "abi_version": next(iter(versions)) if versions else None,
              "dependencies": {"ninjars": None, "oneloopmaster": None, "symbolica": None},
              "archive_sha256": sorted(archives),
              "config_sha256": sorted(configs),
              "source_roles": {role: list(values.values())
                               for role, values in sorted(source_roles.items())},
              "manifest": None}
    if specification is None:
        return result

    def retain(record):
        filename = policy.check_file(record)
        inputs[policy.relative(filename)] = workers.digest(filename)
        return filename

    filename = policy.path(specification)
    inputs[policy.relative(filename)] = workers.digest(filename)
    record = policy.verified(policy.read(filename), "madloop-final-runtime-provenance-v1")
    expected = supplement_expectations(dataset, bindings)
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Runtime provenance differs from measured source/runtime cohort")
    if len(archives) != 1 or len(configs) != 1 or versions != {3}:
        raise ValueError("Runtime provenance needs one measured NinjaRS archive and config")
    installation = policy.read(retain(record["installation"]))
    cargo_file, lock_file, config_file = [retain(record[key]) for key in
                                          ("cargo_toml", "cargo_lock", "installed_config")]
    cargo, lock = [tomllib.loads(filename.read_text()) for filename in (cargo_file, lock_file)]
    config = policy.read(config_file)
    if (installation.get("passed") is not True or installation.get("exit_code") != 0
            or installation.get("compiled_smoke_passed") is not True
            or installation.get("abi_version") != 3 or config.get("abi_version") != 3):
        raise ValueError("Runtime provenance installation did not pass ABI-3 smoke checks")
    expected_files = {"Cargo.toml": workers.digest(cargo_file),
                      "Cargo.lock": workers.digest(lock_file),
                      "lib/ninjars-config.json": workers.digest(config_file),
                      "lib/libninjars.a": next(iter(archives))}
    if (any(installation.get("files", {}).get(key) != value
            for key, value in expected_files.items())
            or workers.digest(config_file) not in configs):
        raise ValueError("Installed dependency metadata is not bound to the measured archive")
    if config.get("local_sources"):
        raise ValueError("Final pinned-dependency provenance cannot hide local source overrides")
    revisions = {}
    for name, crate in (("ninjars", "ninjars"), ("oneloopmaster", "oneloop"),
                        ("symbolica", "symbolica")):
        packages = [package for package in lock.get("package", []) if package["name"] == crate]
        if len(packages) != 1:
            raise ValueError("Ambiguous installed dependency: " + crate)
        source = packages[0].get("source", "")
        revision = source.rsplit("#", 1)[-1]
        if not source.startswith("git+") or not re.fullmatch("[0-9a-f]{40}", revision):
            raise ValueError("Installed dependency lacks a full pinned Git revision: " + crate)
        revisions[name] = revision
    direct = cargo.get("dependencies", {})
    patches = cargo.get("patch", {}).get("crates-io", {})
    if (any(direct.get(name, {}).get("rev") != revisions["ninjars"]
            for name in ("ninjars", "ninjars-symbolica"))
            or any(patches.get(name, {}).get("rev") != revisions["symbolica"]
                   for name in ("symbolica", "numerica", "graphica"))
            or record.get("dependencies") != revisions):
        raise ValueError("Declared dependency revisions differ from installed Cargo lockfile")
    result.update(status="passed", dependencies=revisions, manifest=policy.relative(filename),
                  identity_sha256=record["identity_sha256"],
                  installation_revision=installation.get("revision"))
    return result


def regression_snapshot(specification, bindings, inputs):
    """Attach the independently checked MG7 suites with their actual scope/skips."""
    if specification is None:
        return {"status": "pending", "suites": [], "tests_run": 0, "skipped": 0}
    generators = {}
    for binding in bindings:
        for item in binding["targets"].values():
            if item["role"] == "candidate_corrected":
                files = item["generation_source_identity"]["tree_state"]["tracked"]
                generators[bench.content_digest(files)] = files
    if len(generators) != 1:
        raise ValueError("Regression evidence requires one final candidate generator")
    filename = policy.path(specification)
    verifier = filename.parent / "verify_validation.py"
    # The retained verifier is part of the report implementation, with its hash
    # included in both the input and renderer manifests. It checks the raw
    # unittest logs, reviewed fixtures and relevant production source hashes.
    for source in (filename, verifier):
        inputs[policy.relative(source)] = workers.digest(source)
    module_spec = importlib.util.spec_from_file_location("ninjars_final_regressions_v1", verifier)
    module = importlib.util.module_from_spec(module_spec)
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        module_spec.loader.exec_module(module)
        result = module.report_snapshot(filename, next(iter(generators.values())), inputs)
    finally:
        sys.dont_write_bytecode = previous_bytecode
    if (result.get("status") not in ("passed", "failed", "pending")
            or not isinstance(result.get("suites"), list)):
        raise ValueError("Invalid final MG7 regression summary")
    return result


def build_snapshot(dataset_path, timestamp=None, generation_inputs=None, profile_spec=None,
                   supplement_spec=None, runtime_provenance_spec=None, validation_spec=None,
                   stress_spec=None):
    dataset_path = policy.path(dataset_path)
    dataset = policy.validate_dataset(dataset_path)
    inputs = {}

    def read(filename):
        filename = policy.path(filename)
        raw = filename.read_bytes()
        inputs[str(filename.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    read(dataset_path)
    rules = read(policy.check_file(dataset["policy"]))
    config = read(policy.check_file(rules["configuration"]))
    correction = read(policy.check_file(rules["correction"]))
    proof = read(policy.check_file(dataset["representative_proof"]))
    bindings = [read(policy.check_file(dataset[key]))
                for key in ("timing_bindings", "reference_bindings")]
    provenance = read(REPORT / "provenance.json")
    # Historical diagnostics are tied to their old data and must not be shown as
    # findings of this independently validated dataset.
    provenance.pop("outstanding_findings", None)
    # Setup provenance predates ABI 3 and later dependency updates. Only the
    # measured archive's bound installed metadata may supply current pins/ABI.
    for key in ("ninjars_revision", "oneloopmaster_revision", "symbolica_revision",
                "ninjars_abi_version", "accepted_core_series", "validation_status"):
        provenance.pop(key, None)
    if provenance["baseline_revision"] != BASELINE_REVISION:
        raise ValueError("Report provenance changed the literal original timing baseline")
    groups = defaultdict(list)
    for filename, checksum in dataset["measurements_sha256"].items():
        record = read(filename)
        if inputs[filename] != checksum:
            raise ValueError("Measurement changed after dataset validation")
        matches = [b for b in bindings if policy.path(filename).is_relative_to(policy.path(b["report"]))]
        if len(matches) != 1:
            raise ValueError("Ambiguous timing/reference source role")
        target = matches[0]["targets"][f"{record['case']}-{record['source']}-{record['backend']}"]
        role = target["role"]
        if record["backend"] not in policy.ROLES[role]:
            raise ValueError("Backend is invalid for source role")
        groups[record["case"], role, record["backend"]].append(record)
    for runs in groups.values():
        runs.sort(key=lambda r: r["repetition"])
    cases = []
    for case in config["cases"]:
        case_id = case["id"]
        timings = {
            source: {str(b): summary(groups.get((case_id, role, b), []),
                                    config["points_per_case"], config["repetitions"])
                     for b in BACKENDS}
            for source, role in (("baseline", "original"), ("candidate", "candidate_corrected"))
        }
        references = {str(b): summary(groups.get((case_id, "original_helper_corrected", b), []),
                                     config["points_per_case"], 2) for b in (1, 6, 7)}
        agreement = {"status": "pending", "points": None, "orders": None,
                     "comparisons": None, "max_tolerance_fraction": None}
        original = {"status": "pending", "failures": [], "pairs": 0}
        performance = {"status": "pending", "runtime_ratio": None,
                       "ratio_confidence_interval_95": None}
        stability = {"status": "pending", "targets": {}}
        point_hash = parameter_hash = None
        if case_id in dataset["completed_cases"]:
            rows = {f"{role}/{b}": groups[case_id, role, b][0]["rows"]
                    for role, backends in policy.ROLES.items() for b in backends}
            corrected = {k: v for k, v in rows.items() if not k.startswith("original/")}
            agreement = compare(corrected)
            stability = stability_snapshot(corrected)
            pairs = [(f"original/{b}", f"original_helper_corrected/{b}") for b in (1, 6, 7)]
            pairs += [(f"original/{b}", f"candidate_corrected/{b}") for b in (1, 6, 7)]
            pairs += [("original/6", "candidate_corrected/8")]
            original = compare(rows, pairs)
            recorded = dataset["cases"][case_id]
            compare_recorded(agreement, recorded["corrected_agreement"])
            compare_recorded(original, recorded["original_diagnostics"])
            performance = bench.runtime_statistics(
                [r["seconds_per_point"] for r in groups[case_id, "original", 6]],
                [r["seconds_per_point"] for r in groups[case_id, "candidate_corrected", 8]],
                config["max_runtime_ratio"], config["seed"])
            if performance != recorded["performance"]:
                raise ValueError("Stale performance conclusion")
            first = groups[case_id, "candidate_corrected", 8][0]
            point_hash, parameter_hash = first["points_sha256"], first["parameter_card_sha256"]
            for item in timings["candidate"].values():
                item["ratio"] = item["median_us"] / timings["baseline"]["6"]["median_us"]
        preparation = {source: {"generation": None, "compilation": None}
                       for source in ("baseline", "candidate")}
        for source in preparation:
            key = f"{case_id}-{source}-6"
            if key in bindings[0]["targets"]:
                build = read(policy.check_file(bindings[0]["targets"][key]["build"]))
                preparation[source]["compilation"] = build["origin_compilation"]
                c = correction["cases"][case_id]
                record = (c["original_generation"] if source == "baseline" else
                          read(policy.check_file(c["generation_records"][2])))
                retained = (policy.path(generation_inputs) if generation_inputs is not None else
                            policy.path(bindings[0]["report"]) / "inputs") / source / case_id
                if source == "candidate" and (
                        generation_inputs is not None or (retained / "generation.json").is_file()):
                    record = read(retained / "generation.json")
                    manifest = read(retained / "compiled-sources.json")
                    if (bench.content_digest(manifest["files"]) != manifest["files_sha256"] or
                            manifest["files_sha256"] != build["original_generated_sources_sha256"]):
                        raise ValueError("Generation source manifest differs from measured build")
                    if generation_for_build(record, build) is None:
                        raise ValueError("Generation identity differs from measured build")
                preparation[source]["generation"] = generation_for_build(record, build)
        cases.append(dict(case, timings=timings, corrected_reference_usage=references,
                          agreement=agreement, original_diagnostics=original,
                          performance=performance, stability=stability, preparation=preparation,
                          point_hash=point_hash, parameter_hash=parameter_hash))
    counts = {"cases": len(cases)}
    for gate, name in (("agreement", "correctness"), ("performance", "performance")):
        for status in ("passed", "failed", "unresolved"):
            counts[f"{name}_{status}"] = sum(c[gate]["status"] == status for c in cases)
    counts["measured_candidate_cells"] = sum(
        t["status"] == "measured" for c in cases for t in c["timings"]["candidate"].values())
    counts["original_mismatch_records"] = sum(len(c["original_diagnostics"]["failures"]) for c in cases)
    profiles = profile_snapshot(profile_spec, dataset, bindings, correction, inputs)
    supplement = supplement_snapshot(supplement_spec, dataset, bindings, inputs)
    stress = stress_snapshot(stress_spec, dataset, bindings, inputs)
    runtime_provenance = runtime_provenance_snapshot(runtime_provenance_spec, dataset, bindings, inputs)
    regressions = regression_snapshot(validation_spec, bindings, inputs)
    complete = (dataset["all_cases_passed"] and supplement["complete_campaign_passed"]
                and stress["complete_campaign_passed"]
                and runtime_provenance["status"] == "passed" and regressions["status"] == "passed")
    failed = (counts["correctness_failed"] or counts["performance_failed"]
              or supplement["status"] == "failed" or stress["status"] == "failed"
              or regressions["status"] == "failed")
    unresolved = (counts["performance_unresolved"] or supplement["status"] == "unresolved"
                  or stress["status"] == "unresolved")
    campaign_status = "passed" if complete else "failed" if failed else "unresolved" if unresolved else "pending"
    return {"schema_version": 2, "generated_utc": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d · %H:%M UTC"),
            "dataset_identity_sha256": dataset["identity_sha256"],
            "policy_identity_sha256": rules["identity_sha256"],
            "correction_identity_sha256": correction["identity_sha256"],
            "representative_proof_identity_sha256": proof["identity_sha256"],
            "dataset_path": str(dataset_path.relative_to(ROOT)),
            "config": config, "provenance": provenance, "cases": cases, "summary": counts,
            "profiles": profiles,
            "supplement": supplement,
            "stress": stress,
            "stress_specification": policy.relative(stress_spec) if stress_spec else None,
            "runtime_provenance": runtime_provenance,
            "regressions": regressions,
            "validation_specification": policy.relative(validation_spec) if validation_spec else None,
            "runtime_provenance_specification": (policy.relative(runtime_provenance_spec)
                                                 if runtime_provenance_spec else None),
            "supplement_specification": policy.relative(supplement_spec) if supplement_spec else None,
            "complete_campaign_passed": complete, "campaign_status": campaign_status,
            "profile_specification": policy.relative(profile_spec) if profile_spec else None,
            "generation_inputs": policy.relative(generation_inputs) if generation_inputs else None,
            "all_cases_passed": dataset["all_cases_passed"],
            "host_cpu": provenance.get("host_cpu", provenance.get("host", {}).get("Model name", "Not recorded")),
            "inputs_sha256": dict(sorted(inputs.items()))}


def render(dataset_path, destination, timestamp=None, compile_pdf=False,
           generation_inputs=None, profile_spec=None, supplement_spec=None,
           runtime_provenance_spec=None, validation_spec=None, stress_spec=None):
    destination = policy.path(destination)
    if destination.exists():
        raise ValueError("Report destination exists; preserve the previous report")
    snapshot = build_snapshot(dataset_path, timestamp, generation_inputs, profile_spec, supplement_spec,
                              runtime_provenance_spec, validation_spec, stress_spec)
    template = REPORT / "policy-v2/template/benchmark.typ"
    style = REPORT / "style.typ"
    render_sources = [Path(__file__).resolve(), Path(profiler.__file__).resolve(), template, style]
    if supplement_spec is not None or stress_spec is not None:
        import ninjars_supplement_v2 as supplement
        render_sources.append(Path(supplement.__file__).resolve())
    if supplement_spec is not None:
        import ninjars_precision_v2 as precision
        render_sources.append(Path(precision.__file__).resolve())
    if stress_spec is not None:
        import ninjars_stress_v2 as stress
        render_sources.append(Path(stress.__file__).resolve())
    if validation_spec is not None:
        render_sources.append(policy.path(validation_spec).parent / "verify_validation.py")
    snapshot["render_sources_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in render_sources}
    compile_command = ["typst", "compile", "--creation-timestamp",
                       str(PDF_CREATION_TIMESTAMP), "benchmark.typ", "benchmark.pdf"]
    snapshot["pdf_build"] = {"command": compile_command,
                             "creation_timestamp": PDF_CREATION_TIMESTAMP}
    destination.mkdir(parents=True)
    bench.dump(destination / "report-data.json", snapshot)
    shutil.copy2(template, destination / "benchmark.typ")
    shutil.copy2(style, destination / "style.typ")
    if compile_pdf:
        subprocess.run(compile_command, cwd=destination, check=True)
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timestamp")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--generation-inputs", type=Path,
                        help="retained final CLI inputs root (candidate/CASE/generation.json "
                             "and compiled-sources.json), verified against measured builds")
    parser.add_argument("--profile-spec", type=Path,
                        help="sealed diagnostic profile/raw-perf manifest, bound to this dataset")
    parser.add_argument("--supplement", type=Path,
                        help="validated final loop-induced, threshold and precision supplement; "
                             "must match the exact core candidate generator and NinjaRS runtime")
    parser.add_argument("--runtime-provenance", type=Path,
                        help="sealed final installed Cargo/config/installation provenance, "
                             "bound to the exact measured NinjaRS archive")
    parser.add_argument("--validation-spec", type=Path,
                        help="retained MG7 regression manifest and verifier; tested production "
                             "source hashes must match the final measured generator")
    parser.add_argument("--stress", type=Path,
                        help="validated additional two-to-four correctness/rotation/scale controls; "
                             "times are diagnostic and must match the exact core source/runtime")
    args = parser.parse_args()
    output = args.output or args.dataset.parent / "report"
    snapshot = render(args.dataset, output, args.timestamp, args.compile,
                      args.generation_inputs, args.profile_spec, args.supplement,
                      args.runtime_provenance, args.validation_spec, args.stress)
    print(json.dumps({"report": str(output), "summary": snapshot["summary"],
                      "all_cases_passed": snapshot["all_cases_passed"],
                      "complete_campaign_passed": snapshot["complete_campaign_passed"]}, indent=2))


if __name__ == "__main__":
    main()
