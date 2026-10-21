#!/usr/bin/env python3
"""Frozen-input loop-induced and threshold supplement, outside the core harness.

The preparation commands do not evaluate a loop amplitude.  Collection reuses
the unchanged persistent-worker protocol; its explicit launch flag is intended
for orchestration after the final numerical source/runtime freeze.
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_references_v2 as references
import ninjars_benchmark_workers as workers

ROOT = bench.ROOT
PREREG = ROOT / "docs/benchmarkfolder/policy-v2/final-supplement-plan-v1"
DEFAULT_REPORT = ROOT / "docs/benchmarkfolder/policy-v2/final-supplement-generation-v1"
DEFAULT_WORK = ROOT / ".benchmark-work/policy-v2/final-supplement-generation-v1"
SCHEMA = "madloop-final-supplement-v1"
MODEL_FILES = policy.MODEL_FILES
ROLES = {"original": (1, 6, 7), "candidate_corrected": (1, 6, 7, 8),
         "original_helper_corrected": (1, 6, 7)}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(path):
    path = policy.path(path)
    return {"path": policy.relative(path), "sha256": digest(path),
            "bytes": path.stat().st_size}


def checked(record, inputs=None):
    path = policy.path(record["path"])
    if digest(path) != record["sha256"] or ("bytes" in record and path.stat().st_size != record["bytes"]):
        raise ValueError("Changed supplement evidence: " + str(path))
    if inputs is not None:
        inputs[policy.relative(path)] = record["sha256"]
    return path


def frozen_configuration():
    manifest = read(PREREG / "manifest.json")
    for name, item in manifest["files"].items():
        if digest(PREREG / name) != item["sha256"] or (PREREG / name).stat().st_size != item["bytes"]:
            raise ValueError("Changed supplement preregistration")
    plan = read(PREREG / "configuration.json")
    if digest(ROOT / plan["policy"]["source_configuration"]) != plan["policy"]["source_configuration_sha256"]:
        raise ValueError("Changed control policy")
    return plan


def loop_configuration():
    plan = frozen_configuration()
    return {"schema_version": 1, "seed": plan["loop_induced"]["seed"],
            "points_per_case": plan["loop_induced"]["points_per_case"],
            "repetitions": plan["loop_induced"]["repetitions"],
            "energy_gev": plan["loop_induced"]["energy_gev"],
            "relative_tolerance": plan["policy"]["relative_tolerance"],
            "scale_tolerance": plan["policy"]["scale_tolerance"],
            "max_runtime_ratio": plan["loop_induced"]["max_runtime_ratio"],
            "madloop_parameters": plan["policy"]["madloop_parameters"],
            "cases": [{k: case[k] for k in ("id", "model", "process", "nfinal")}
                      for case in plan["loop_induced"]["cases"]]}


def validate_configuration(config, threshold=False):
    expected = loop_configuration()
    if threshold:
        expected["points_per_case"] = 12
        expected["repetitions"] = 2
        expected["cases"] = [{**expected["cases"][1], "id": "supplement-gg-hh-threshold"}]
    if config != expected:
        raise ValueError("Supplement input configuration differs from fixed plan")


@contextmanager
def harness(work, report):
    old_work, old_report = bench.WORK, bench.REPORT
    bench.WORK, bench.REPORT = Path(work), Path(report)
    try:
        yield
    finally:
        bench.WORK, bench.REPORT = old_work, old_report


def preparation(report=DEFAULT_REPORT, work=DEFAULT_WORK):
    report, work = policy.path(report), policy.path(work)
    if report.exists() or work.exists():
        raise RuntimeError("Preparation destination exists; use a new namespace")
    report.mkdir(parents=True)
    work.mkdir(parents=True)
    (report / "sources").mkdir()
    shutil.copy2(Path(__file__), report / "sources/preparation.py")
    config = loop_configuration()
    bench.dump(report / "cases.json", config)
    bench.dump(report / "preparation.json", {
        "schema": "supplement-preparation-v1", "configuration": bound(report / "cases.json"),
        "preregistration": bound(PREREG / "configuration.json"),
        "baseline_revision": bench.BASELINE_REVISION,
        "script": bound(report / "sources/preparation.py"), "work": policy.relative(work),
        "report": policy.relative(report), "final_candidate_bound": False,
        "loop_evaluations": 0, "initialized_symbolica_workers": 0})
    return report


def baseline_integrity():
    """Compare literal git-archive members, respecting .gitattributes exports."""
    count, tree = 0, {}
    process = subprocess.Popen(["git", "archive", bench.BASELINE_REVISION], cwd=ROOT,
                               stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                if not (member.isfile() or member.issym()):
                    continue
                raw = member.linkname.encode() if member.issym() else archive.extractfile(member).read()
                path = bench.BASELINE / member.name
                actual = os.readlink(path).encode() if member.issym() else path.read_bytes()
                if actual != raw:
                    raise ValueError("Literal baseline archive changed: " + member.name)
                tree[member.name] = hashlib.sha256(raw).hexdigest()
                count += 1
    finally:
        process.stdout.close()
        code = process.wait()
    if code:
        raise ValueError("Original git archive verification failed")
    return {"revision": bench.BASELINE_REVISION, "files": count,
            "files_sha256": bench.content_digest(tree)}


def point_metadata(path, masses, nfinal):
    raw = Path(path).read_bytes()
    lines = raw.decode().splitlines()
    count = int(lines[0])
    values = [list(map(float, line.split())) for line in lines[1:]]
    nexternal = nfinal + 2
    if len(values) != count * nexternal or len(masses) != nexternal or any(len(p) != 4 for p in values):
        raise ValueError("Invalid supplemental momentum stream")
    events = [values[i*nexternal:(i+1)*nexternal] for i in range(count)]
    residual = shell = 0.0
    invariants = []
    for event in events:
        scale = max(p[0] for p in event)
        if scale <= 0 or any(not math.isfinite(x) for p in event for x in p):
            raise ValueError("Nonphysical supplemental point")
        r = max(abs(sum(p[j] for p in event[:2])-sum(p[j] for p in event[2:])) for j in range(4)) / scale
        s = max(abs(p[0]**2-sum(x*x for x in p[1:])-m*m) for p, m in zip(event, masses)) / scale**2
        if r > 1e-10 or s > 1e-10:
            raise ValueError("Off-shell or nonconserving supplemental point")
        residual, shell = max(residual, r), max(shell, s)
        total = [event[0][j]+event[1][j] for j in range(4)]
        invariants.append(total[0]**2-sum(x*x for x in total[1:]))
    return {"points": count, "external_masses": masses, "sha256": hashlib.sha256(raw).hexdigest(),
            "serialized_distinct_configurations": len({tuple(tuple(p) for p in e) for e in events}),
            "physical_distinct_invariant_s": 1 if nfinal == 1 else None,
            "serialized_invariant_s_min": min(invariants), "serialized_invariant_s_max": max(invariants),
            "invariant_interpretation": "2->1 has one constrained physical s; storage roundoff is not new phase space" if nfinal == 1 else "RAMBO orientations at fixed center-of-mass energy",
            "max_relative_conservation_residual": residual,
            "max_relative_mass_shell_residual": shell,
            "storage": "binary64 values serialized as 17 digits after decimal in exponential notation"}


def threshold_points(top_mass, higgs_mass):
    plan = frozen_configuration()["gg_hh_threshold"]
    if not all(math.isfinite(m) and m > 0 for m in (top_mass, higgs_mass)):
        raise ValueError("Invalid final-card top/Higgs masses")
    points, grid = [], []
    for delta in plan["delta"]:
        energy = 2 * top_mass * (1 + float(delta))
        if energy <= 2 * higgs_mass:
            raise ValueError("Fixed threshold plan is incompatible with the physical card")
        e = energy/2
        p = math.sqrt(e*e-higgs_mass*higgs_mass)
        for cosine in plan["cos_theta"]:
            c = float(cosine)
            x, z = p*math.sqrt(1-c*c), p*c
            points.append([[e, 0., 0., e], [e, 0., 0., -e],
                           [e, x, 0., z], [e, -x, 0., -z]])
            grid.append({"point": len(points), "delta": delta, "cos_theta": cosine,
                         "sqrt_s_gev": energy, "azimuth": "0"})
    return points, grid


def generate(source, report=DEFAULT_REPORT, work=DEFAULT_WORK):
    report, work = policy.path(report), policy.path(work)
    config = read(report / "cases.json")
    validate_configuration(config)
    if source not in ("baseline", "candidate"):
        raise ValueError("Invalid generator role")
    # Candidate is intentionally deferred until root declares source/runtime
    # final; the ordinary bench capture binds exact clean numerical inputs.
    if source == "candidate":
        subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *bench.GENERATOR_PATHS, "vendor/ninjars"], cwd=ROOT, check=True)
    status_path = report / (source + "-generation.json")
    if status_path.exists():
        raise RuntimeError("Do not overwrite previous supplemental generation")
    status = {"status": "started", "source": source, "cases": {},
              "loop_evaluations": 0, "initialized_symbolica_workers": 0,
              "affinity": sorted(os.sched_getaffinity(0))}
    bench.dump(status_path, status)
    try:
        with harness(work, report):
            for case in config["cases"]:
                bench.generate(source, case)
                masses = bench.build(source, case)
                output = bench.output_dir(source, case)
                if source == "baseline":
                    points = bench.make_points(case, masses, config)
                    bench.dump(points.with_suffix(".physical.json"), point_metadata(points, masses, case["nfinal"]))
                else:
                    points = report / "points" / (case["id"] + ".dat")
                    if read(points.with_suffix(".json"))["masses"] != masses:
                        raise ValueError("Candidate external masses differ from original")
                    baseline = bench.output_dir("baseline", case)
                    if (baseline / "Cards/param_card.dat").read_bytes() != (output / "Cards/param_card.dat").read_bytes():
                        raise ValueError("Candidate physical card differs from literal original")
                manifest = bench.generated_source_manifest(output)
                target = report / "manifests" / f"{case['id']}-{source}.json"
                bench.dump(target, manifest)
                status["cases"][case["id"]] = {
                    "output": policy.relative(output), "generated_manifest": bound(target),
                    "generation": bound(output / "generation.json"),
                    "compilation": bound(output / "compilation.json"), "masses": masses,
                    "point_stream": bound(points), "physical_metadata": bound(points.with_suffix(".physical.json"))}
                bench.dump(status_path, status)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        bench.dump(status_path, status)
        raise
    status["status"] = "complete"
    bench.dump(status_path, status)
    return status


def generation_anchors(report=DEFAULT_REPORT, work=DEFAULT_WORK):
    report, work = policy.path(report), policy.path(work)
    integrity = read(report / "baseline-integrity.json")
    if integrity["files"] != 6009 or integrity["revision"] != bench.BASELINE_REVISION:
        raise ValueError("Complete original source verification is required")
    result = {}
    for source, role in (("baseline", "original"), ("candidate", "candidate_corrected")):
        status_file = report / (source + "-generation.json")
        if not status_file.exists():
            continue
        status = read(status_file)
        if status["status"] != "complete":
            raise ValueError("Incomplete generated supplement source")
        result[source] = {}
        for case in loop_configuration()["cases"]:
            output = work / "outputs" / source / case["id"]
            for stage in ("generate", "compile", "masses"):
                original_log = work / "logs" / f"{stage}-{source}-{case['id']}.log"
                retained_log = report / "logs" / original_log.name
                retained_log.parent.mkdir(exist_ok=True)
                if retained_log.exists() and digest(retained_log) != digest(original_log):
                    raise ValueError("Retained supplemental generation/build log changed")
                shutil.copy2(original_log, retained_log)
            compilation = read(output / "compilation.json")
            manifest = bench.generated_source_manifest(output)
            if manifest["files_sha256"] != compilation["generated_sources_sha256"]:
                raise ValueError("Supplement sources changed after compilation")
            record = policy.seal({"schema": "supplement-generation-anchor-v1", "role": role,
                                  "case": case, "compiled_manifest": manifest,
                                  "generation": read(output / "generation.json"), "compilation": compilation,
                                  "parameter_card_sha256": digest(output / "Cards/param_card.dat"),
                                  "cli_command": bound(report / "commands" / f"{source}-{case['id']}.mg5"),
                                  "cli_log": bound(report / "logs" / f"generate-{source}-{case['id']}.log"),
                                  "baseline_integrity": integrity if source == "baseline" else None})
            target = report / "source-proofs/with-cli-v2" / source / (case["id"] + ".json")
            if target.exists() and read(target) != record:
                raise ValueError("Refuse to replace an existing supplemental source anchor")
            bench.dump(target, record)
            result[source][case["id"]] = bound(target)
    return result


def collection_specs(report=DEFAULT_REPORT, work=DEFAULT_WORK, control_cpu=2):
    report, work = policy.path(report), policy.path(work)
    anchors = generation_anchors(report, work)
    if set(anchors) != {"baseline", "candidate"}:
        raise ValueError("Final candidate CLI generation must precede collection specs")
    helpers = read(report / "helpers/summary.json")
    helper_proofs = {cid: item["source_proof"] for cid, item in helpers["cases"].items()}
    config = read(report / "cases.json")
    points = {c["id"]: bound(report / "points" / (c["id"] + ".dat")) for c in config["cases"]}
    threshold_points = {"supplement-gg-hh-threshold": bound(report / "points/supplement-gg-hh-threshold.dat")}
    common = {"timeout": 86400., "configuration": bound(report / "cases.json"), "points": points}
    targets = [
        ("loop-timing", {"baseline": "original", "candidate": "candidate_corrected"}, 207, 7, True, False),
        ("loop-reference", {"baseline": "original_helper_corrected"}, control_cpu, 2, False, False),
        ("threshold-candidate", {"candidate": "candidate_corrected"}, control_cpu, 2, False, True),
        ("threshold-reference", {"baseline": "original_helper_corrected"}, control_cpu, 2, False, True)]
    results = {}
    for label, roles, cpu, repetitions, timing, threshold in targets:
        spec = {**common, "role_map": roles, "cpu": cpu, "repetitions": repetitions,
                "timing": timing, "threshold": threshold, "run_id": "final-supplement-"+label+"-v1",
                "report": policy.relative(report / "runs" / label),
                "work": policy.relative(work / "runs" / label), "output_roots": {}, "source_proofs": {}}
        for source, role in roles.items():
            spec["output_roots"][source] = policy.relative(work / "helpers/outputs/baseline" if role == "original_helper_corrected" else work / "outputs" / source)
            spec["source_proofs"][source] = helper_proofs if role == "original_helper_corrected" else anchors[source]
        if threshold:
            spec.update(configuration=bound(report / "threshold.json"), points=threshold_points,
                        case_aliases={"supplement-gg-hh-threshold": "supplement-gg-hh"})
        path = report / "collection-specs" / (label + ".json")
        if path.exists():
            raise ValueError("Collection spec already exists")
        bench.dump(path, spec)
        results[label] = bound(path)
    return results


def verify_preparation(report=DEFAULT_REPORT, work=DEFAULT_WORK, live=False):
    """Reparse retained literal/helper sources, exact fixed streams and grid."""
    report, work = policy.path(report), policy.path(work)
    config = read(report / "cases.json")
    validate_configuration(config)
    integrity = read(report / "baseline-integrity.json")
    if integrity["revision"] != bench.BASELINE_REVISION or integrity["files"] != 6009:
        raise ValueError("Incomplete original archive verification")
    if live and baseline_integrity() != integrity:
        raise ValueError("Literal original source changed since supplement generation")
    source_status = read(report / "baseline-generation.json")
    if source_status["status"] != "complete" or source_status["loop_evaluations"] or source_status["initialized_symbolica_workers"]:
        raise ValueError("Baseline preparation did not complete without amplitude evaluation")
    helpers = policy.verified(read(report / "helpers/summary.json"), "supplement-helper-preparation-v1")
    correction = policy.validate_correction(ROOT / "docs/benchmarkfolder/policy-v2/model-correction.json")
    for case in config["cases"]:
        cid = case["id"]
        saved = source_status["cases"][cid]
        original = read(checked(saved["generated_manifest"]))
        compilation = read(report / "inputs/baseline" / cid / "compilation.json")
        if original["files_sha256"] != compilation["generated_sources_sha256"]:
            raise ValueError("Retained literal source manifest differs from build")
        points = checked(saved["point_stream"])
        physical = point_metadata(points, saved["masses"], case["nfinal"])
        if physical != read(checked(saved["physical_metadata"])) or physical["points"] != 100:
            raise ValueError("Retained fixed phase-space stream differs from metadata")
        if live and bench.generated_source_manifest(work / "outputs/baseline" / cid) != original:
            raise ValueError("Original generated sources changed after supplement preparation")
        helper = helpers["cases"][cid]
        proof = policy.verified(read(checked(helper["source_proof"])), "supplement-helper-correction-v1")
        correction_case = correction["cases"][proof["source_control_case"]]
        if proof["original_manifest"] != original or proof["correction_identity_sha256"] != correction["identity_sha256"]:
            raise ValueError("Helper reference is not derived from this literal original")
        policy.verify_delta(original, proof["corrected_manifest"],
                            {name: correction_case["fresh_after_manifest"]["files"].get(name) for name in MODEL_FILES})
        for name, record in proof["postimages"].items():
            if checked(record).read_bytes() != checked(correction_case["postimages"][name]).read_bytes():
                raise ValueError("Helper numerical postimage differs from core correction")
        rebuild = policy.verified(helper["rebuild"], "forced-model-rebuild-v2")
        if (rebuild["sources_sha256"] != proof["corrected_manifest"]["files_sha256"] or
                rebuild["prebuild_model_objects"] or rebuild["prebuild_model_archive_exists"] or
                rebuild["prelink_executable_exists"] or not rebuild["model_objects"] or
                {k: v for k, v in rebuild["libraries_before"].items() if k != "libmodel.a"} !=
                {k: v for k, v in rebuild["libraries_after"].items() if k != "libmodel.a"}):
            raise ValueError("Helper source/build proof differs")
        if live and bench.generated_source_manifest(policy.path(helper["output"])) != proof["corrected_manifest"]:
            raise ValueError("Private helper source changed after preparation")
    threshold_path = report / "points/supplement-gg-hh-threshold.dat"
    metadata_path = threshold_path.with_suffix(".bound.json")
    metadata = read(metadata_path if metadata_path.exists() else threshold_path.with_suffix(".json"))
    events, grid = threshold_points(metadata["top_mass_gev"], metadata["higgs_mass_gev"])
    expected = str(len(events))+"\n"+"".join(" ".join(f"{x:.17e}" for x in p)+"\n" for event in events for p in event)
    if threshold_path.read_text() != expected or metadata["grid"] != grid:
        raise ValueError("Threshold points differ from preregistered Cartesian grid")
    checked(metadata["parameter_card"])
    physical = point_metadata(threshold_path, [0., 0., metadata["higgs_mass_gev"], metadata["higgs_mass_gev"]], 2)
    if any(metadata[key] != value for key, value in physical.items()):
        raise ValueError("Threshold point metadata differs from serialized kinematics")
    return {"status": "passed", "baseline_files": 6009, "literal_cli_exports": 2,
            "private_helper_rebuilds": 2, "seeded_points": 200, "threshold_points": 12,
            "gg_h_boosted_configurations": 100, "gg_h_physical_distinct_invariant_s": 1,
            "loop_amplitude_evaluations": 0, "initialized_symbolica_workers": 0,
            "live_source_checks": live}


def prepare_threshold(report=DEFAULT_REPORT, work=DEFAULT_WORK):
    from madgraph.models.check_param_card import ParamCard
    report, work = policy.path(report), policy.path(work)
    config = read(report / "cases.json")
    case = config["cases"][1]
    output = work / "outputs/baseline" / case["id"]
    card = output / "Cards/param_card.dat"
    param = ParamCard(str(card))
    top, higgs = param["mass"].get((6,)).value, param["mass"].get((25,)).value
    events, grid = threshold_points(float(top), float(higgs))
    path = report / "points/supplement-gg-hh-threshold.dat"
    if path.exists():
        raise RuntimeError("Threshold points are already bound")
    path.write_text(str(len(events)) + "\n" + "".join(" ".join(f"{x:.17e}" for x in p) + "\n" for event in events for p in event))
    metadata = point_metadata(path, [0., 0., float(higgs), float(higgs)], 2)
    metadata.update(grid=grid, parameter_card=bound(card), top_mass_gev=top,
                    higgs_mass_gev=higgs, width_policy="Unchanged zero-width physical card")
    bench.dump(path.with_suffix(".json"), metadata)
    portable = {**metadata, "parameter_card": bound(report / "inputs/baseline" / case["id"] / "param_card.dat")}
    bench.dump(path.with_suffix(".bound.json"), portable)
    threshold = copy.deepcopy(config)
    threshold.update(points_per_case=12, repetitions=2,
                     cases=[{**case, "id": "supplement-gg-hh-threshold"}])
    bench.dump(report / "threshold.json", threshold)
    return metadata


def prepare_helpers(report=DEFAULT_REPORT, work=DEFAULT_WORK):
    """Copy only the core policy's exact MODEL postimages into private clones."""
    report, work = policy.path(report), policy.path(work)
    correction_path = ROOT / "docs/benchmarkfolder/policy-v2/model-correction.json"
    correction = policy.validate_correction(correction_path)
    config = read(report / "cases.json")
    helper_work, helper_report = work / "helpers", report / "helpers"
    if helper_work.exists() or helper_report.exists():
        raise RuntimeError("Helper clone namespace already exists")
    helper_work.mkdir()
    helper_report.mkdir()
    results = {}
    for case in config["cases"]:
        control_id = case["id"].replace("supplement-", "control-", 1)
        proof = correction["cases"][control_id]
        original = work / "outputs/baseline" / case["id"]
        output = helper_work / "outputs/baseline" / case["id"]
        before = bench.generated_source_manifest(original)
        references.clone_output(original, output)
        retained = helper_report / "postimages" / case["id"]
        for name in MODEL_FILES:
            record = proof["postimages"][name]
            if record is None:
                continue
            raw = checked(record).read_bytes()
            target = output / name
            references.local_write(target, output)
            target.write_bytes(raw)
            saved = retained / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(raw)
        after = bench.generated_source_manifest(output)
        policy.verify_delta(before, after, {name: proof["fresh_after_manifest"]["files"].get(name) for name in MODEL_FILES})
        source_proof = policy.seal({
            "schema": "supplement-helper-correction-v1", "case": case["id"],
            "source_control_case": control_id, "correction": bound(correction_path),
            "correction_identity_sha256": correction["identity_sha256"],
            "original_output": policy.relative(original), "helper_output": policy.relative(output),
            "original_manifest": before, "corrected_manifest": after,
            "physical_card_sha256": digest(original / "Cards/param_card.dat"),
            "postimages": {name: bound(retained / name) for name in MODEL_FILES if (retained / name).exists()}})
        bench.dump(helper_report / (case["id"] + "-source.json"), source_proof)
        with harness(helper_work, helper_report):
            rebuild = references.force_rebuild(output, case, correction["identity_sha256"], helper_work, helper_report)
        if bench.generated_source_manifest(original) != before:
            raise ValueError("Helper rebuild altered literal original generated output")
        results[case["id"]] = {"source_proof": bound(helper_report / (case["id"] + "-source.json")),
                               "rebuild": rebuild, "output": policy.relative(output)}
    record = policy.seal({"schema": "supplement-helper-preparation-v1", "cases": results,
                          "loop_evaluations": 0, "initialized_symbolica_workers": 0})
    bench.dump(helper_report / "summary.json", record)
    return record


def distribution(values):
    if not values or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Invalid estimated-accuracy population")
    ordered = sorted(values)
    return {"count": len(values), "minimum": ordered[0], "median": statistics.median(values),
            "maximum": ordered[-1], "p90": ordered[math.ceil(.9*len(values))-1],
            "zero_count": values.count(0.), "interpretation": "MadLoop internal estimate; not independently measured precision"}


def comparable_numerical_settings(record):
    """Apply the core policy's backend-specific QP annotation rule.

    The selector is derived metadata: CutTools, Ninja and NinjaRS require their
    own QP integrand; Collier has no corresponding option. Validate that
    requirement before omitting only this annotation from cross-backend equality.
    All physical/numerical card settings remain in the comparison unchanged.
    """
    numerical = record["shared_numerical_settings"]
    if (numerical.get("collierrequiredaccuracy") != 1e-12 or
            numerical.get("mlstabthres") != 1e-9 or
            numerical.get("colliercomputeuvpoles") is not True or
            numerical.get("colliercomputeirpoles") is not True or
            (record["backend"] != 7 and numerical.get("selected_backend_qp_integrand") is not True)):
        raise ValueError("Measurements do not use required full precision/pole/settings policy")
    return {key: value for key, value in numerical.items()
            if key != "selected_backend_qp_integrand"}


def compare_rows(groups, pairs=None):
    keys = list(groups)
    if not keys:
        raise ValueError("Empty numerical supplement cohort")
    stream = [(r["point"], r["order"]) for r in groups[keys[0]]]
    if not stream or len(set(stream)) != len(stream):
        raise ValueError("Empty or duplicated point/order stream")
    scales = defaultdict(float)
    for rows in groups.values():
        if [(r["point"], r["order"]) for r in rows] != stream:
            raise ValueError("Mismatched supplement point/order streams")
        for row in rows:
            if len(row["coefficients"]) != 4 or not all(math.isfinite(x) for x in row["coefficients"]):
                raise ValueError("Nonfinite physical supplement coefficients")
            scales[row["point"]] = max(scales[row["point"]], *map(abs, row["coefficients"]))
    pairs = list(itertools.combinations(keys, 2) if pairs is None else pairs)
    failures, checks, worst, largest_scaled = [], 0, 0., 0.
    per_point = {p: {"maximum_tolerance_fraction": 0., "maximum_point_scaled_absolute_difference": 0.} for p, _ in stream}
    for left, right in pairs:
        for a, b in zip(groups[left], groups[right]):
            for coefficient, (x, y) in enumerate(zip(a["coefficients"], b["coefficients"])):
                tolerance = 1e-7*max(abs(x), abs(y)) + 1e-10*scales[a["point"]]
                error = abs(x-y)
                fraction = error/tolerance if tolerance else 0.
                scaled = error/scales[a["point"]] if scales[a["point"]] else 0.
                worst, largest_scaled = max(worst, fraction), max(largest_scaled, scaled)
                target = per_point[a["point"]]
                target["maximum_tolerance_fraction"] = max(target["maximum_tolerance_fraction"], fraction)
                target["maximum_point_scaled_absolute_difference"] = max(target["maximum_point_scaled_absolute_difference"], scaled)
                checks += 1
                if error > tolerance:
                    failures.append({"targets": [left, right], "point": a["point"], "order": a["order"],
                                     "coefficient": coefficient, "values": [x, y], "tolerance": tolerance})
    return {"passed": not failures, "status": "failed" if failures else "passed", "pairs": len(pairs),
            "points": len({p for p, _ in stream}), "orders": len({o for _, o in stream}),
            "comparisons": checks, "max_tolerance_fraction": worst,
            "maximum_point_scaled_absolute_difference": largest_scaled, "failures": failures,
            "per_point": {str(p): item for p, item in sorted(per_point.items())}}


def stability_summary(groups):
    """Separate estimates, observed discrepancies, and status/rescue use."""
    references = [name for name in groups if name.startswith("original_helper_corrected/")]
    if not references:
        raise ValueError("Independent backend references are missing")
    targets = {}
    for name, rows in groups.items():
        peers = [peer for peer in references if peer != name]
        discrepancy = compare_rows(groups, [(name, peer) for peer in peers])
        totals = [r for r in rows if r["order"] == 0]
        counts = Counter(str(r["status"]) for r in totals)
        targets[name] = {
            "estimated_accuracy": distribution([r["accuracy"] for r in totals]),
            "estimated_accuracy_by_order": {str(order): distribution([r["accuracy"] for r in rows if r["order"] == order])
                                             for order in sorted({r["order"] for r in rows})},
            "observed_discrepancy": {**discrepancy,
                "interpretation": "Observed cross-backend coefficient discrepancy; the references are not an exact high-precision amplitude oracle"},
            "status_counts": dict(sorted(counts.items())),
            "dp": sum(count for code, count in counts.items() if int(code)//100 == 2),
            "qp": sum(count for code, count in counts.items() if int(code)//100 == 3),
            "ninjars_genuine_qp": sum(count for code, count in counts.items() if int(code)//100 == 3 and int(code)%10 == 2) if name.endswith("/8") else None,
            "ninjars_accepted_full_qp_322": counts.get("322", 0) if name.endswith("/8") else None,
            "per_point": [{"point": r["point"], "status": r["status"], "H": r["status"]//100,
                           "T": r["status"]//10 % 10, "U": r["status"] % 10,
                           "estimated_accuracy": r["accuracy"],
                           **discrepancy["per_point"][str(r["point"])]} for r in totals]}
    return {"targets": targets,
            "coverage": "Fixed deterministic streams/grid only; no probabilistic instability-rate claim",
            "independent_high_precision_controls": "Reported separately under precision_controls; not inferred from MadLoop accuracy or cross-backend agreement"}


@contextmanager
def input_roots(roots, sentinel, aliases=None):
    original = bench.output_dir
    roots = {source: policy.path(root) for source, root in roots.items()}
    aliases = aliases or {}
    def resolve(source, case):
        if bench.WORK == sentinel:
            return roots[source] / aliases.get(case["id"], case["id"])
        return original(source, case)
    bench.output_dir = resolve
    try:
        yield
    finally:
        bench.output_dir = original


def bind_collected(report, work, role_map, config, spec):
    """Capture numerical runtime plus source/build bindings while outputs exist."""
    targets = {}
    assets = report / "runtime-assets"
    assets.mkdir()
    protocol = read(report / "protocol.json")
    sources = ("baseline", "candidate") if protocol["sources"] == "both" else (protocol["sources"],)
    for case in config["cases"]:
        for source in sources:
            role = role_map[source]
            for backend in protocol["backends"]:
                if source == "baseline" and backend == 8:
                    continue
                if backend not in ROLES[role]:
                    raise ValueError("Invalid supplement source role/backend")
                key = f"{case['id']}-{source}-{backend}"
                output = work / "workers" / key / "outputs" / source / case["id"]
                retained = report / "builds" / key
                build = read(retained / "build.json")
                manifest = read(retained / "compiled-sources.json")
                if bench.generated_source_manifest(output) != manifest:
                    raise ValueError("Worker generated source changed after collection")
                executable = next(output.glob("SubProcesses/P*_*/check"))
                if digest(executable) != build["executable_sha256"]:
                    raise ValueError("Worker executable changed after collection")
                metadata = {}
                for name, path in policy.runtime_metadata_files(output).items():
                    destination = assets / digest(path)
                    if not destination.exists():
                        shutil.copy2(path, destination)
                    metadata[name] = bound(destination)
                item = {"role": role, "case": case["id"], "source": source, "backend": backend,
                        "output": policy.relative(output), "build": bound(retained / "build.json"),
                        "compiled_manifest": manifest, "libraries": policy.library_hashes(output),
                        "runtime_metadata": metadata, "generation_source_identity": build["generation_source_identity"],
                        "executable_sha256": build["executable_sha256"],
                        "generated_sources_sha256": manifest["files_sha256"]}
                source_case = spec.get("case_aliases", {}).get(case["id"], case["id"])
                origin = policy.path(spec["output_roots"][source]) / source_case
                anchor = report / "source-anchors" / key
                anchor.mkdir(parents=True)
                for name in ("compiled-sources.json", "generation.json", "compilation.json"):
                    shutil.copy2(origin / name, anchor / name)
                shutil.copy2(origin / "Cards/param_card.dat", anchor / "param_card.dat")
                item["origin_manifest"] = bound(anchor / "compiled-sources.json")
                item["origin_generation"] = bound(anchor / "generation.json")
                item["origin_compilation"] = bound(anchor / "compilation.json")
                item["physical_card"] = bound(anchor / "param_card.dat")
                proof = checked(spec["source_proofs"][source][source_case])
                shutil.copy2(proof, anchor / "source-proof.json")
                item["source_proof"] = bound(anchor / "source-proof.json")
                item["reducer_runtime"] = policy.reducer_signature(item)
                targets[key] = item
    record = policy.seal({"schema": "supplement-source-bindings-v1", "report": policy.relative(report),
                          "role_map": role_map, "targets": targets})
    bench.dump(report / "source-bindings.json", record)
    return record


def collect(specification):
    """Adapter only: all physics evaluation uses unchanged workers.run_case."""
    specification = policy.path(specification)
    spec = read(specification)
    config = read(checked(spec["configuration"]))
    threshold = bool(spec.get("threshold", False))
    validate_configuration(config, threshold=threshold)
    roles = spec["role_map"]
    timing = bool(spec["timing"])
    if timing:
        if threshold or roles != {"baseline": "original", "candidate": "candidate_corrected"}:
            raise ValueError("Timing denominator/roles must match fixed supplement")
        if spec["cpu"] != 207 or spec["repetitions"] != 7:
            raise ValueError("Final supplement timing requires CPU207 and seven repetitions")
        backends = (1, 6, 7, 8)
    else:
        if spec["repetitions"] != 2 or roles not in (
                {"baseline": "original_helper_corrected"}, {"candidate": "candidate_corrected"}):
            raise ValueError("Correctness needs complete two-command candidate/reference roles")
        backends = (1, 6, 7) if "baseline" in roles else (1, 6, 7, 8)
    cpu = spec["cpu"]
    if cpu not in os.sched_getaffinity(0):
        raise ValueError("Unavailable supplement CPU")
    report, work = policy.path(spec["report"]), policy.path(spec["work"])
    if report.exists() or work.exists():
        raise RuntimeError("Collection namespace already exists")
    report.mkdir(parents=True)
    work.mkdir(parents=True)
    for name in ("sources", "points"):
        (report / name).mkdir()
    for name in workers.SOURCE_NAMES:
        shutil.copy2(ROOT / "tools" / name, report / "sources" / name)
    shutil.copy2(Path(__file__), report / "sources" / Path(__file__).name)
    shutil.copy2(specification, report / "input-spec.json")
    for case in config["cases"]:
        item = spec["points"][case["id"]]
        shutil.copy2(checked(item), report / "points" / (case["id"] + ".dat"))
    descriptor = workers.protocol_descriptor()
    sources = tuple(roles)
    bench.dump(report / "protocol.json", {
        "schema": workers.SCHEME, "configuration": config, "protocol_descriptor": descriptor,
        "protocol_identity_sha256": bench.content_digest(descriptor), "cpu": cpu,
        "repetitions": spec["repetitions"], "sources": "both" if len(sources) == 2 else sources[0],
        "backends": list(backends), "fresh_parity": False, "acceptance_eligible": False,
        "input_adapter": bound(report / "sources" / Path(__file__).name),
        "input_spec": bound(report / "input-spec.json"), "role_map": roles,
        "timing_acceptance_sample": timing,
        "initialization_policy": "Unchanged persistent-worker-v1: initialize and warm once",
        "ordering": "Unchanged run_case: adjacent original6/candidate8, reverse on odd repetitions"})
    status = {"status": "incomplete", "requested_cases": [c["id"] for c in config["cases"]],
              "completed_cases": [], "acceptance_eligible": False}
    bench.dump(report / "run-status.json", status)
    args = SimpleNamespace(cpu=cpu, repetitions=spec["repetitions"],
                           minimum_seconds=1. if timing else 1e-12,
                           timeout=float(spec.get("timeout", 86400.)),
                           fresh_parity=False, clean_build=True)
    sentinel = work / "input-root-sentinel"
    try:
        with harness(sentinel, report), input_roots(spec["output_roots"], sentinel, spec.get("case_aliases")):
            for case in config["cases"]:
                workers.run_case(case, sources, backends, config, args,
                                 work / "workers", report, spec["run_id"])
                status["completed_cases"].append(case["id"])
                bench.dump(report / "run-status.json", status)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        bench.dump(report / "run-status.json", status)
        raise
    status["status"] = "complete"
    bench.dump(report / "run-status.json", status)
    workers.checked_series(report, minimum=7 if timing else 2, require_core=timing)
    return bind_collected(report, work, roles, config, spec)


def checked_bindings(filename, inputs=None):
    data = policy.verified(read(filename), "supplement-source-bindings-v1")
    report = policy.path(data["report"])
    protocol, records = workers.checked_series(report, minimum=2)
    checked(bound(filename), inputs)
    if protocol["role_map"] != data["role_map"]:
        raise ValueError("Supplement roles differ from collected protocol")
    for key, target in data["targets"].items():
        build = read(checked(target["build"], inputs))
        manifest = target["compiled_manifest"]
        if (bench.content_digest(manifest["files"]) != manifest["files_sha256"] or
                manifest["files_sha256"] != build["generated_sources_sha256"] or
                target["executable_sha256"] != build["executable_sha256"] or
                target["generation_source_identity"] != build["generation_source_identity"]):
            raise ValueError("Invalid supplement source/build identity")
        if target["role"] != data["role_map"][target["source"]] or target["backend"] not in ROLES[target["role"]]:
            raise ValueError("Invalid bound supplement role/backend")
        if policy.reducer_signature(target) != target["reducer_runtime"]:
            raise ValueError("Changed supplement reduction runtime")
        for item in target["runtime_metadata"].values():
            checked(item, inputs)
        identity = target["generation_source_identity"]
        origin = read(checked(target["origin_manifest"], inputs))
        generation = read(checked(target["origin_generation"], inputs))
        compilation = read(checked(target["origin_compilation"], inputs))
        physical_card = checked(target["physical_card"], inputs)
        source_proof = read(checked(target["source_proof"], inputs))
        if (bench.content_digest(origin["files"]) != origin["files_sha256"] or
                origin["files_sha256"] != build["original_generated_sources_sha256"] or
                origin["files_sha256"] != compilation["generated_sources_sha256"] or
                compilation != build["origin_compilation"] or
                generation["source_identity"] != identity or
                digest(physical_card) != build["parameter_card_sha256"] or
                generation["parameter_card_sha256"] != build["parameter_card_sha256"]):
            raise ValueError("Supplement input-source anchor differs from worker build")
        if target["role"] == "original_helper_corrected":
            policy.verified(source_proof, "supplement-helper-correction-v1")
            correction = policy.validate_correction(checked(source_proof["correction"], inputs))
            proof = correction["cases"][source_proof["source_control_case"]]
            if (source_proof["corrected_manifest"] != origin or
                    source_proof["correction_identity_sha256"] != correction["identity_sha256"] or
                    source_proof["physical_card_sha256"] != digest(physical_card)):
                raise ValueError("Helper reference differs from exact original MODEL correction")
            policy.verify_delta(source_proof["original_manifest"], origin,
                                {name: proof["fresh_after_manifest"]["files"].get(name) for name in MODEL_FILES})
        else:
            policy.verified(source_proof, "supplement-generation-anchor-v1")
            command = checked(source_proof["cli_command"], inputs).read_text()
            log = checked(source_proof["cli_log"], inputs).read_text()
            case_definition = source_proof["case"]
            if (f"generate {case_definition['process']}\n" not in command or
                    f"import model {case_definition['model']}\n" not in command or
                    "Error detected in sub-command" in log or "interrupted in sub-command" in log):
                raise ValueError("Missing or failed actual-CLI supplement generation")
            if (source_proof["role"] != target["role"] or source_proof["compiled_manifest"] != origin or
                    source_proof["generation"] != generation or source_proof["compilation"] != compilation or
                    source_proof["parameter_card_sha256"] != digest(physical_card)):
                raise ValueError("Generated source differs from supplemental CLI anchor")
            if target["role"] == "original":
                if source_proof["baseline_integrity"]["revision"] != bench.BASELINE_REVISION or source_proof["baseline_integrity"]["files"] != 6009:
                    raise ValueError("Literal supplement source lacks complete original archive proof")
        if target["role"] in ("original", "original_helper_corrected"):
            policy.original_identity(identity)
        elif (identity.get("dirty_patch_sha256") is not None or
              identity.get("tree_state_sha256") != bench.content_digest(identity["tree_state"]) or
              identity["tree_state"]["untracked"]):
            raise ValueError("Supplement candidate generator is not a clean bound source")
    for (_, source, backend, _), (path, record) in records.items():
        checked(bound(path), inputs)
        key = f"{record['case']}-{source}-{backend}"
        target = data["targets"][key]
        if (record["generation_source_identity"] != target["generation_source_identity"] or
                record["executable_sha256"] != target["executable_sha256"]):
            raise ValueError("Numerical stream differs from supplement source binding")
        points = report / "points" / (record["case"] + ".dat")
        checked(bound(points), inputs)
        if digest(points) != record["points_sha256"]:
            raise ValueError("Retained supplemental momentum bytes differ from measurements")
    return data, protocol, records


def validate_supplement(filename, inputs=None):
    """Validate complete raw-bound inputs; recompute publication in snapshot()."""
    filename = policy.path(filename)
    data = policy.verified(read(filename), SCHEMA)
    checked(bound(filename), inputs)
    frozen_configuration()
    if data["literal_baseline_revision"] != bench.BASELINE_REVISION:
        raise ValueError("Supplement literal baseline differs from approved revision")
    collections = []
    for entry in data["source_groups"]:
        source, protocol, records = checked_bindings(checked(entry["bindings"], inputs), inputs)
        validate_configuration(protocol["configuration"], threshold=entry["kind"] == "threshold")
        if entry["kind"] not in ("loop-induced", "threshold"):
            raise ValueError("Unknown supplement evidence kind")
        if entry["kind"] == "loop-induced" and "original" in source["role_map"].values():
            if (protocol["cpu"] != 207 or protocol["repetitions"] != 7 or
                    source["role_map"] != {"baseline": "original", "candidate": "candidate_corrected"}):
                raise ValueError("Supplement timing roles/CPU/repetitions differ from fixed plan")
        for target in source["targets"].values():
            if target["role"] == "candidate_corrected":
                if target["generation_source_identity"]["tree_state_sha256"] != data["candidate_generator_identity_sha256"]:
                    raise ValueError("Mixed candidate generators in supplement")
                if target["backend"] == 8 and target["reducer_runtime"]["identity_sha256"] != data["candidate_runtime_identity_sha256"]:
                    raise ValueError("Mixed NinjaRS runtimes in supplement")
        collections.append((entry, source, protocol, records))
    return data, collections


def assemble_supplement(report, expected, destination, precision_proof=None):
    """Bind completed/failed subcampaigns without inventing missing results."""
    report, destination = policy.path(report), policy.path(destination)
    if destination.exists():
        raise ValueError("Supplement dataset exists; choose a new immutable snapshot")
    required = {"core_dataset_identity_sha256", "literal_baseline_revision",
                "candidate_generator_identity_sha256", "candidate_runtime_identity_sha256"}
    if set(expected) != required or any(not value for value in expected.values()):
        raise ValueError("All final core cohort identities are required")
    source_groups, failed_groups = [], []
    for name, kind in (("loop-timing", "loop-induced"), ("loop-reference", "loop-induced"),
                       ("threshold-candidate", "threshold"), ("threshold-reference", "threshold")):
        directory = report / "runs" / name
        if (directory / "source-bindings.json").exists():
            source_groups.append({"kind": kind, "bindings": bound(directory / "source-bindings.json")})
        elif (directory / "run-status.json").exists():
            rejected = [bound(p) for p in sorted((directory / "rejected").glob("*.json"))]
            failed_groups.append({"kind": kind, "run_status": bound(directory / "run-status.json"), "rejected": rejected})
    data = policy.seal({"schema": SCHEMA, **expected, "source_groups": source_groups,
                        "failed_groups": failed_groups,
                        "threshold_points_metadata": bound(report / "points/supplement-gg-hh-threshold.bound.json"),
                        "point_metadata": {case["id"]: bound(report / "points" / (case["id"] + ".physical.json")) for case in loop_configuration()["cases"]},
                        "precision_controls": bound(precision_proof) if precision_proof else {"status": "pending", "working_bits": 256, "convergence_bits": 384, "checks": []}})
    bench.dump(destination, data)
    return report_snapshot(destination, expected)


def report_snapshot(filename, expected, inputs=None, runtime_signature=None):
    data, collections = validate_supplement(filename, inputs)
    for key, value in expected.items():
        if value is None or data.get(key) != value:
            raise ValueError("Supplement does not belong to final core cohort: " + key)
    # Lazy import avoids a module cycle when the renderer imports this adapter.
    import ninjars_report_v2 as report
    groups, threshold = {}, {}
    for entry, source, protocol, records in collections:
        destination = threshold if entry["kind"] == "threshold" else groups
        for (case, source_name, backend, repetition), (_, row) in records.items():
            role = source["role_map"][source_name]
            destination.setdefault((case, role, backend), []).append(row)
    for table in (groups, threshold):
        for runs in table.values():
            runs.sort(key=lambda r: r["repetition"])
            if [r["repetition"] for r in runs] != list(range(len(runs))):
                raise ValueError("Duplicate or missing supplement repetitions")
        for case_id in {key[0] for key in table}:
            records = [record for (case, _, _), runs in table.items() if case == case_id for record in runs]
            for key in ("points_sha256", "parameter_card_sha256", "shared_numerical_settings", "answer_dimensions", "protocol_identity_sha256"):
                if len({bench.content_digest(comparable_numerical_settings(record)
                                            if key == "shared_numerical_settings" else record[key])
                        for record in records}) != 1:
                    raise ValueError("Supplement roles use different physical inputs/settings: "+key)
    cases = []
    config = loop_configuration()
    for case in config["cases"]:
        cid = case["id"]
        timings = {label: {str(b): report.summary(groups.get((cid, role, b), []), 100, 7)
                          for b in (1, 6, 7, 8)}
                   for label, role in (("baseline", "original"), ("candidate", "candidate_corrected"))}
        corrected_rows = {f"{role}/{b}": runs[0]["rows"] for (c, role, b), runs in groups.items()
                          if c == cid and role != "original"}
        required = {f"{role}/{b}" for role in ("candidate_corrected", "original_helper_corrected") for b in ROLES[role]}
        agreement = report.compare(corrected_rows) if set(corrected_rows) == required else {
            "status": "pending", "points": None, "orders": None, "comparisons": None,
            "max_tolerance_fraction": None, "pairs": 0, "failures": []}
        baseline, candidate = groups.get((cid, "original", 6), []), groups.get((cid, "candidate_corrected", 8), [])
        performance = bench.runtime_statistics([r["seconds_per_point"] for r in baseline],
                                              [r["seconds_per_point"] for r in candidate], 1.05, config["seed"]) if len(baseline) == len(candidate) == 7 else {"status": "pending"}
        stable = report.stability_snapshot(corrected_rows) if set(corrected_rows) == required else {"status": "pending"}
        if timings["baseline"]["6"]["median_us"] is not None:
            for timing in timings["candidate"].values():
                if timing["median_us"] is not None:
                    timing["ratio"] = timing["median_us"]/timings["baseline"]["6"]["median_us"]
        all_rows = {f"{role}/{b}": runs[0]["rows"] for (c, role, b), runs in groups.items() if c == cid}
        pairs = [(f"original/{b}", f"original_helper_corrected/{b}") for b in (1, 6, 7)]
        pairs += [(f"original/{b}", f"candidate_corrected/{b}") for b in (1, 6, 7)]
        pairs += [("original/6", "candidate_corrected/8")]
        original = report.compare(all_rows, pairs) if all(left in all_rows and right in all_rows for left, right in pairs) else {"status": "pending", "pairs": 0, "failures": []}
        for rows in corrected_rows.values():
            if any(row["coefficients"][0] != 0. for row in rows):
                raise ValueError("Loop-induced supplement contains a nonzero Born coefficient")
        cases.append({**case, "loop_induced": True, "born_verified_zero": bool(corrected_rows), "points": 100, "timings": timings,
                      "agreement": agreement, "performance": performance, "stability": stable,
                      "original_diagnostics": original,
                      "kinematics": read(checked(data["point_metadata"][cid], inputs)),
                      "corrected_reference_usage": {str(b): report.summary(groups.get((cid, "original_helper_corrected", b), []), 100, 2) for b in (1, 6, 7)}})
    threshold_rows = {f"{role}/{b}": runs[0]["rows"] for (_, role, b), runs in threshold.items()}
    grid = {"agreement": {"status": "pending"}, "stability": {"status": "pending"}, "point_summaries": []}
    required = {f"{role}/{b}" for role in ("candidate_corrected", "original_helper_corrected") for b in ROLES[role]}
    if set(threshold_rows) == required:
        grid["agreement"] = report.compare(threshold_rows)
        grid["stability"] = report.stability_snapshot(threshold_rows)
        metadata = read(checked(data["threshold_points_metadata"], inputs))
        details = compare_rows(threshold_rows)["per_point"]
        for point in metadata["grid"]:
            number = point["point"]
            grid["point_summaries"].append({**point, "max_tolerance_fraction": details[str(number)]["maximum_tolerance_fraction"],
                "estimated_accuracy": {target: next(r["accuracy"] for r in rows if r["point"] == number and r["order"] == 0) for target, rows in threshold_rows.items()},
                "status_counts": {target: str(next(r["status"] for r in rows if r["point"] == number and r["order"] == 0)) for target, rows in threshold_rows.items()}})
    precision = data.get("precision_controls", {"status": "pending", "working_bits": 256, "convergence_bits": 384, "checks": []})
    if "path" in precision and "sha256" in precision:
        import ninjars_precision_v2 as precision_adapter
        precision = precision_adapter.validate(checked(precision, inputs), data["candidate_runtime_identity_sha256"], inputs, runtime_signature)
    elif precision.get("status") != "pending":
        raise ValueError("A supplied precision pass label cannot replace raw installed-archive proof")
    statuses = [c[g]["status"] for c in cases for g in ("agreement", "performance")]
    statuses += [grid["agreement"]["status"], precision["status"]]
    failures = []
    for item in data.get("failed_groups", []):
        failure = read(checked(item["run_status"], inputs))
        if failure["status"] == "complete":
            raise ValueError("Complete supplement collection is missing its source bindings")
        failures.append({"kind": item["kind"], "status": failure,
                         "rejected": [read(checked(record, inputs)) for record in item["rejected"]]})
        statuses.append("failed" if failure["status"] == "failed" else "unresolved")
    status = "failed" if "failed" in statuses else "passed" if all(s == "passed" for s in statuses) else "unresolved"
    return {"schema": SCHEMA, **expected, "dataset_identity_sha256": data["identity_sha256"],
            "status": status, "complete_campaign_passed": status == "passed", "cases": cases,
            "threshold_grid": grid, "precision_controls": precision, "source_groups": data["source_groups"],
            "failed_groups": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "generate-baseline", "generate-candidate", "threshold", "helpers", "anchors", "specs", "verify-preparation", "collect", "assemble"))
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--execute-numerical", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--precision-proof", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        result = preparation(args.report, args.work)
    elif args.action.startswith("generate-"):
        result = generate(args.action.removeprefix("generate-"), args.report, args.work)
    elif args.action == "helpers":
        result = prepare_helpers(args.report, args.work)
    elif args.action == "anchors":
        result = generation_anchors(args.report, args.work)
    elif args.action == "specs":
        result = collection_specs(args.report, args.work)
    elif args.action == "verify-preparation":
        result = verify_preparation(args.report, args.work, args.live)
    elif args.action == "assemble":
        if not args.expected or not args.output:
            parser.error("Assembly requires --expected final core identities and --output immutable dataset")
        result = assemble_supplement(args.report, read(args.expected), args.output, args.precision_proof)
    elif args.action == "collect":
        if not args.spec or not args.execute_numerical:
            parser.error("Collection requires an explicit spec and --execute-numerical handoff")
        result = collect(args.spec)
    else:
        result = prepare_threshold(args.report, args.work)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
