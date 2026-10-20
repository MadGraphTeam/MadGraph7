#!/usr/bin/env python3
"""Isolated generated-process numerical controls for the NinjaRS integration."""
import argparse
import collections
import copy
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil

import ninjars_benchmark as bench


CONFIG = bench.REPORT / "controls.json"
VARIANTS = bench.REPORT / "control-variants.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_points(path, nexternal):
    lines = path.read_text().splitlines()
    count = int(lines[0])
    values = [[float(x) for x in line.split()] for line in lines[1:]]
    if len(values) != count * nexternal:
        raise RuntimeError("Incomplete control point stream")
    return [values[i*nexternal:(i+1)*nexternal] for i in range(count)]


def write_points(case, events, provenance):
    masses = [float(x) for x in (
        bench.subprocess_dir("candidate", case) / "masses.dat").read_text().split()]
    largest_residual = largest_shell = 0.
    for event in events:
        scale = max(p[0] for p in event)
        residual = max(abs(sum(p[k] for p in event[:2]) -
                           sum(p[k] for p in event[2:])) for k in range(4))
        shell = max(abs(p[0]**2-sum(x*x for x in p[1:])-mass**2)
                    for p, mass in zip(event, masses))
        if (not all(math.isfinite(x) for p in event for x in p) or
                residual > 1e-10*scale or shell > 1e-10*scale*scale):
            raise RuntimeError(f"Invalid transformed kinematics: {case['id']}")
        largest_residual = max(largest_residual, residual/scale)
        largest_shell = max(largest_shell, shell/(scale*scale))
    provenance.update(max_relative_conservation_residual=largest_residual,
                      max_relative_mass_shell_residual=largest_shell,
                      external_masses=masses)
    path = bench.REPORT / "points" / (case["id"] + ".dat")
    path.write_text(str(len(events)) + "\n" + "".join(
        " ".join(f"{x:.17e}" for x in momentum) + "\n"
        for event in events for momentum in event))
    bench.dump(path.with_suffix(".json"), {
        "points": len(events), "sha256": digest(path), "provenance": provenance})


def clone_output(parent, case):
    source = bench.output_dir("candidate", parent)
    target = bench.output_dir("candidate", case)
    if not (source / "compilation.json").exists():
        raise RuntimeError(f"Parent control not compiled: {source}")
    if not target.exists():
        shutil.copytree(source, target, symlinks=True)
    return target


def build_rebinding_driver(case, fresh):
    directory = bench.subprocess_dir("candidate", case)
    path = directory / "check_sa.f"
    text = path.read_text()
    state_indices = iter(("1", "1+MOD(K-1,NPOINTS)", "K", "K"))
    text = re.sub(r"CALL UPDATE_AS_PARAM\(\)",
                  lambda _: f"CALL CONTROL_STATE({next(state_indices)})", text)
    factors = [1., 1.125, .875, 1., 1.125, .875, 1., 1.125, .875, 1.]
    scales = [1., 2., .5, 1., .5, 2., 1., 2., .5, 1.]
    text += """
      SUBROUTINE CONTROL_STATE(INDEX)
      USE MNINJARS, ONLY: NINJARS_FINALIZE
      IMPLICIT NONE
      INTEGER INDEX
      INCLUDE 'coupl.inc'
      INCLUDE 'mp_coupl.inc'
      REAL*8 INITIAL_MT, MASS_FACTOR(10), SCALE_FACTOR(10)
      LOGICAL FIRST
      DATA FIRST/.TRUE./
      SAVE FIRST,INITIAL_MT
      DATA MASS_FACTOR/1D0,1.125D0,.875D0,1D0,1.125D0,
     $ .875D0,1D0,1.125D0,.875D0,1D0/
      DATA SCALE_FACTOR/1D0,2D0,.5D0,1D0,.5D0,2D0,
     $ 1D0,2D0,.5D0,1D0/
      IF(FIRST) THEN
        INITIAL_MT=MDL_MT
        FIRST=.FALSE.
      ENDIF
      MDL_MT=INITIAL_MT*MASS_FACTOR(INDEX)
      MU_R=MU_R*SCALE_FACTOR(INDEX)
      MP__MDL_MT=REAL(MDL_MT,KIND=16)
      MP__MU_R=CMPLX(REAL(MU_R,KIND=16),0.0_16,KIND=16)
      CALL COUP()
"""
    if fresh:
        text += "      CALL NINJARS_FINALIZE()\n"
    text += "      END\n"
    path.write_text(text)
    seconds = bench.run(["make", "-j4", "check"], directory,
                        bench.WORK / "logs" / f"compile-candidate-{case['id']}-driver.log")
    output = bench.output_dir("candidate", case)
    metadata = json.loads((output / "compilation.json").read_text())
    source_manifest = bench.generated_source_manifest(output)
    bench.dump(output / "compiled-sources.json", source_manifest)
    metadata.update(seconds=seconds, driver_sha256=digest(path),
                    executable_sha256=digest(directory / "check"),
                    generated_sources_sha256=source_manifest["files_sha256"],
                    source_manifest_recording="recorded-after-control-driver-rebuild")
    bench.dump(output / "compilation.json", metadata)
    bench.retain_build_inputs("candidate", case)
    return {"top_mass_factors": factors, "renormalization_scale_factors": scales,
            "reset_ninjars_context_before_each_evaluation": fresh,
            "model_updates": "DP and MP independent top mass and renormalization scale assigned before full generated COUP()"}


def prepare():
    config = json.loads(CONFIG.read_text())
    parents = {c["id"]: c for c in config["cases"]}
    variants = []
    for name, parent_id in (
        ("control-rotation", "control-gg-hh"),
        ("control-rebinding", "control-gg-hh"),
        ("control-rebinding-fresh", "control-gg-hh"),
        ("control-near-singular", "control-forced-qp"),
    ):
        parent = parents[parent_id]
        case = copy.deepcopy(parent)
        case["id"] = name
        case["parent_control"] = parent_id
        case["madloop_parameters"] = {"NRotations_DP": 2, "NRotations_QP": 2}
        if name == "control-near-singular":
            case["madloop_parameters"]["MLStabThres"] = 1e-20
        clone_output(parent, case)
        bench.build("candidate", case)
        source_points = bench.REPORT / "points" / (parent_id + ".dat")
        events = read_points(source_points, case["nfinal"] + 2)
        provenance = {"parent_points": str(source_points.relative_to(bench.ROOT)),
                      "parent_points_sha256": digest(source_points)}
        if name == "control-rotation":
            # Two exact rational orthogonal matrices, rounded only on storage.
            for event in events:
                for p in event:
                    x, y, z = p[1:]
                    x, y = (3*x-4*y)/5, (4*x+3*y)/5
                    y, z = (5*y-12*z)/13, (12*y+5*z)/13
                    p[1:] = [x, y, z]
            provenance["rotation"] = "R_x(cos=5/13,sin=12/13) R_z(cos=3/5,sin=4/5)"
        elif name.startswith("control-rebinding"):
            events = [copy.deepcopy(events[0]) for _ in events]
            provenance.update(build_rebinding_driver(case, name.endswith("-fresh")))
        else:
            masses = [float(x) for x in (
                bench.subprocess_dir("candidate", case) / "masses.dat").read_text().split()]
            energy = config["energy_gev"]
            momentum = math.sqrt((energy/2)**2 - masses[2]**2)
            angles = [1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1e-3, 7e-4, 5e-4, 3e-4, 1e-4]
            events = []
            for angle in angles:
                px = momentum*math.sin(angle)*.6
                py = momentum*math.sin(angle)*.8
                pz = momentum*math.cos(angle)
                events.append([[energy/2, 0., 0., energy/2],
                               [energy/2, 0., 0., -energy/2],
                               [energy/2, px, py, pz], [energy/2, -px, -py, -pz]])
            provenance.update(scattering_angles_radians=angles, energy_gev=energy,
                              external_masses=masses)
        write_points(case, events, provenance)
        variants.append(case)
    variant_config = {**config, "cases": variants}
    bench.dump(VARIANTS, variant_config)


def compare_records(label, records, config, require_rescue=False):
    if not label.startswith("rotation-invariance"):
        inputs = {(r["points_sha256"], r["parameter_card_sha256"])
                  for r in records.values()}
        if len(inputs) != 1:
            raise RuntimeError(f"{label}: reducer inputs differ")
    rows = {name: data["rows"] for name, data in records.items()}
    keys = [[(r["point"], r["order"]) for r in value] for value in rows.values()]
    if any(key != keys[0] for key in keys):
        raise RuntimeError("Different control output ordering")
    scales = {}
    for output in rows.values():
        for row in output:
            scales[row["point"]] = max(scales.get(row["point"], 0.),
                                      *(abs(x) for x in row["coefficients"]))
    failures = []
    comparisons = 0
    largest_tolerance_fraction = 0.
    largest_scaled_difference = 0.
    validation_failures = {
        name: data["measurement_validation_error"]
        for name, data in records.items() if "measurement_validation_error" in data}
    for left, right in itertools.combinations(rows, 2):
        for a, b in zip(rows[left], rows[right]):
            for coefficient, (x, y) in enumerate(zip(a["coefficients"], b["coefficients"])):
                tolerance = config["relative_tolerance"]*max(abs(x), abs(y))
                tolerance += config["scale_tolerance"]*scales[a["point"]]
                difference = abs(x-y)
                comparisons += 1
                if tolerance:
                    largest_tolerance_fraction = max(
                        largest_tolerance_fraction, difference/tolerance)
                if scales[a["point"]]:
                    largest_scaled_difference = max(
                        largest_scaled_difference, difference/scales[a["point"]])
                if not math.isfinite(x+y) or difference > tolerance:
                    failures.append({"point": a["point"], "order": a["order"],
                        "coefficient": coefficient, "sources": [left, right],
                        "values": [x, y], "tolerance": tolerance})
    rescue = any(row["status"] in (312, 322) for name, data in records.items()
                 if name.endswith("-8") for row in data["rows"])
    full_qp_rescue = any(row["status"] == 322 for name, data in records.items()
                        if name.endswith("-8") for row in data["rows"])
    if require_rescue and not rescue:
        failures.append({"required_status_hundreds_units": [3, 2],
                         "reason": "No actual NinjaRS rescue occurred"})
    summary = {"passed": not failures and not validation_failures, "failures": failures,
               "numerical_agreement_passed": not any("values" in f for f in failures),
               "measurement_validation_failures": validation_failures,
               "coefficient_comparisons": comparisons,
               "maximum_tolerance_fraction": largest_tolerance_fraction,
               "maximum_point_scaled_absolute_difference": largest_scaled_difference,
               "observed_ninjars_rescue": rescue,
               "observed_ninjars_full_qp_rescue_322": full_qp_rescue,
               "status_counts": {name: data["status_counts"] for name, data in records.items()},
               "relative_tolerance": config["relative_tolerance"],
               "scale_tolerance": config["scale_tolerance"]}
    bench.dump(bench.REPORT / "data" / f"agreement-{label}.json", summary)
    return summary["passed"]


def measure_one(case, backend, config, cpu):
    """Retain rejected stability results as failures, never successful timings."""
    points = bench.REPORT / "points" / (case["id"] + ".dat")
    try:
        return bench.measure("candidate", case, backend, points, 1, 0, cpu, config=config)
    except RuntimeError as error:
        directory = bench.subprocess_dir("candidate", case)
        # Only a fully written numerical stream can be compared diagnostically.
        # Other failures remain hard errors.
        if "returned unexpected status" not in str(error):
            raise
        output = bench.output_dir("candidate", case)
        rows = []
        for line in (directory / "values.dat").read_text().splitlines():
            fields = line.split()
            if len(fields) != 8:
                raise
            rows.append({"point": int(fields[0]), "order": int(fields[1]),
                         "status": int(fields[2]),
                         "coefficients": [float(x) for x in fields[3:7]],
                         "accuracy": float(fields[7])})
        diagnostics = bench.REPORT / "diagnostics" / case["id"] / str(backend)
        diagnostics.mkdir(parents=True, exist_ok=True)
        log = bench.WORK / "logs" / f"run-candidate-{case['id']}-{backend}-0.log"
        for path in [directory / name for name in
                     ("values.dat", "timing.dat", "timed_status.dat", "dimensions.dat")] + [log]:
            shutil.copy2(path, diagnostics / path.name)
        result = {
            "source": "candidate", "case": case["id"], "backend": backend,
            "repetition": 0, "loops": 1, "cpu": cpu,
            "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "measurement_validation_error": str(error),
            "diagnostics": str(diagnostics.relative_to(bench.ROOT)),
            "points_sha256": digest(points),
            "parameter_card_sha256": digest(output / "Cards/param_card.dat"),
            "madloop_card_sha256": digest(output / "Cards/MadLoopParams.dat"),
            "executable_sha256": digest(directory / "check"),
            "status_counts": dict(collections.Counter(str(r["status"]) for r in rows
                                                      if r["order"] == 0)),
            "rows": rows}
        bench.dump(bench.REPORT / "data" / f"candidate-{case['id']}-{backend}-0.json", result)
        print("REJECTED", case["id"], backend, str(error), flush=True)
        return result


def measure(config, selected, cpu):
    failed = []
    for case in config["cases"]:
        if selected and case["id"] not in selected:
            continue
        backends = (8,) if case["id"].endswith("-fresh") else (
            (1, 6, 8) if case["id"] in ("control-forced-qp", "control-near-singular")
            else (1, 6, 7, 8))
        records = {}
        for backend in backends:
            print("control", case["id"], backend, flush=True)
            result = measure_one(case, backend, config, cpu)
            records[f"candidate-{backend}"] = result
        if len(records) > 1:
            if not compare_records(case["id"], records, config,
                                   require_rescue=case["id"] == "control-forced-qp"):
                failed.append(case["id"])
        if case["id"] == "control-forced-qp":
            # Keep the three-backend strict failure visible while separately
            # testing successful own rescue against successful CutTools QP.
            compare_records("control-own-rescue",
                            {k: v for k, v in records.items() if k != "candidate-6"},
                            config, require_rescue=True)
        if case["id"] == "control-rotation":
            for backend in backends:
                original = json.loads((bench.REPORT / "data" /
                    f"candidate-control-gg-hh-{backend}-0.json").read_text())
                label = f"rotation-invariance-{backend}"
                if not compare_records(label,
                    {"original": original, "rotated": records[f"candidate-{backend}"]}, config):
                    failed.append(label)
        if case["id"].endswith("-fresh"):
            retained = json.loads((bench.REPORT / "data" /
                "candidate-control-rebinding-8-0.json").read_text())
            if not compare_records("rebinding-context-reset",
                            {"retained-8": retained, "fresh-8": records["candidate-8"]}, config):
                failed.append("rebinding-context-reset")
    if failed:
        raise RuntimeError(f"Control requirements failed: {', '.join(failed)}")


def summarize():
    """Summarize retained decisions without running a numerical process."""
    required = [
        "control-gg-h", "control-gg-hh", "control-mixed", "control-complex",
        "control-rotation", "control-rebinding", "rebinding-context-reset",
        "control-near-singular", "control-own-rescue",
        *[f"rotation-invariance-{b}" for b in (1, 6, 7, 8)]]
    data = bench.REPORT / "data"
    controls = {}
    for label in required + ["control-forced-qp"]:
        path = data / f"agreement-{label}.json"
        controls[label] = {**json.loads(path.read_text()),
                           "evidence": str(path.relative_to(bench.REPORT)),
                           "evidence_sha256": digest(path)}
    legacy = {}
    for label in required[:4]:
        path = data / f"legacy-agreement-{label}.json"
        legacy[label] = {**json.loads(path.read_text()),
                         "evidence": str(path.relative_to(bench.REPORT)),
                         "evidence_sha256": digest(path)}
    bench.dump(data / "generated-controls-summary.json", {
        "schema_version": 1,
        "required_numerical_controls_passed": all(controls[x]["passed"] for x in required),
        "legacy_regression_passed": all(x["passed"] for x in legacy.values()),
        "strict_three_backend_diagnostic_passed": controls["control-forced-qp"]["passed"],
        "timings_are_performance_acceptance_data": False,
        "status_semantics": {
            "312": "Successful NinjaRS MP reduction with DP integrand",
            "322": "Successful NinjaRS MP reduction with QP integrand",
            "328": "Successful original Ninja QP reduction with QP integrand",
            "329": "Successful CutTools QP reduction with QP integrand",
            "420": "QP evaluation failed the requested stability threshold"},
        "controls": controls,
        "untouched_baseline_regression": legacy})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "measure", "variants", "summarize"))
    parser.add_argument("--cases", default="")
    parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "summarize":
        summarize()
    else:
        config = json.loads((VARIANTS if args.action == "variants" else CONFIG).read_text())
        measure(config, set(filter(None, args.cases.split(","))), args.cpu)


if __name__ == "__main__":
    main()
