#!/usr/bin/env python3
"""Physical stress points for a registered campaign (diagnostic, never timing).

Each run copies one built campaign export, replaces only its driver with
tirust_stress_driver.f (warm-up on the case's ordinary points, then every stress
point of points/<case>-stress.dat evaluated once; refusals are recorded, not
fatal), relinks it, and runs it once:

  run --case C --lane L --mode M   L: baseline-1, candidate-{1,6,7,8,9}, chain-9671
                                   M: auto (CTModeRun=-1, stability checked)
                                      qp   (CTModeRun=CTModeInit=4, unchecked QP)
  summarize                        reference values and TIRust classification

Reference: forced-QP CutTools and Ninja (independent reductions) when they agree
at the campaign tolerance. Otherwise, at least two stability-checked values from
independent reducers that agree; a single reducer is never its own reference
("unverified"). Without a reference, an accepted value is still WRONG when it lies
far outside the spread of all independent values (forced QP of 1, 6 and 8 and the
checked values of 1, 6, 7 and 8; see consensus_check). Independently of any
reference, the IR double pole of every
value is checked: its ratio to the Born is kinematics independent, so it must
equal the case's ratio at ordinary points (from the ordinary matrix, or the
case's ordinary stress anchors).

TIRust (alone or as first backend of the 9|6|7|1 chain) may refuse a stress
point. Blocking outcomes are an accepted value that disagrees with the reference
or has the wrong double pole ("WRONG"), and one whose error exceeds the lane's
MLStabThres although its claimed accuracy does not ("INACCURATE": MadLoop's
stability test was fooled).

Outputs: <campaign report>/experiments/stress/<case>/<lane>-<mode>/result.json
         <campaign report>/experiments/stress/stress-summary.{json,md}
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark as bench  # noqa: E402
import tirust_benchmark_cohorts as cohort  # noqa: E402
from tirust_benchmark_worker import validate_status  # noqa: E402

DRIVER = ROOT / "tools/tirust_stress_driver.f"
LANES = {"baseline-1": ("baseline", [1]), **{f"candidate-{b}": ("candidate", [b]) for b in (1, 6, 7, 8, 9)},
         "chain-9671": ("candidate", [9, 6, 7, 1])}
QP_BACKENDS = (1, 6, 8, 9)
# Stress-only cases outside the registered ordinary matrix (built by tirust_cms_stress.py);
# their warm-up and stress points live in <report>/stress-points/.
EXTRA_CASES = {"cms_ew_n2_epem_mupmum": {"id": "cms_ew_n2_epem_mupmum", "model": "loop_qcd_qed_sm_Gmu",
                                         "process": "e+ e- > mu+ mu- [virt=QED]", "nfinal": 2,
                                         "role": "complex-mass-stress", "complex_mass_scheme": True}}
REFERENCES = ("candidate-1-qp", "candidate-6-qp")
# Without a reference, an accepted value must still lie within the spread of these and of ACCEPTED_REFERENCES.
INDEPENDENT_QP = ("candidate-1-qp", "candidate-6-qp", "candidate-8-qp")
# Weaker fallback: stability-checked values accepted by independent (non-TIRust) reducers.
ACCEPTED_REFERENCES = ("candidate-1-auto", "candidate-6-auto", "candidate-7-auto", "candidate-8-auto")


def route_hooks(prefix, enabled):
    if not enabled:
        return {}
    getter = f"      CALL {prefix}GET_TIRUST_DP_ROUTE_STATISTICS("
    return {"DECLARATIONS": ("      INTEGER*8 STRESS_RB(20),STRESS_RA(20)\n"
                             "      INTEGER STRESS_RSB,STRESS_RSA"),
            "OPEN": "      OPEN(79,FILE='stress-routes.dat',STATUS='REPLACE')",
            "BEFORE": getter + "STRESS_RB,STRESS_RSB)",
            "WRITE": (getter + "STRESS_RA,STRESS_RSA)\n"
                      "      WRITE(79,'(3I12,20I20)') K,STRESS_RSB,STRESS_RSA,\n"
                      "     $ STRESS_RA-STRESS_RB\n"
                      "      FLUSH(79)"),
            "CLOSE": "      CLOSE(79)"}


def render(template, parameters, numerical_source, routes=False):
    prefix = parameters["prefix"]
    active = "\n".join(line for line in numerical_source.splitlines() if not line or line[0] not in "Cc*!")
    common = f"COMMON/{prefix}FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION".upper()
    if common not in "".join(active.upper().split()).replace("$", ""):
        raise ValueError("Generated numerical source lacks the exact reduction FPE flag layout")
    text = template.replace("@FPE_COMMON@", f"      COMMON/{prefix}FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,\n"
                                            "     $ FPE_IN_QP_REDUCTION")
    for name, code in route_hooks(prefix, routes).items():
        anchor = f"C@STRESS_ROUTE_{name}@"
        if anchor not in text:
            raise ValueError(f"Stress driver lacks route anchor {anchor}")
        text = text.replace(anchor, code)
    return bench.render_worker(text, parameters)


def parse_values(path, ndim, partial=False):
    """Rows of stress-values.dat; with ``partial`` an incomplete trailing point is dropped."""
    rows = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if len(fields) != 11:
            raise ValueError("Malformed stress value row")
        point, order, code = map(int, fields[:3])
        numbers = [float(x) for x in fields[3:8]]
        valid, fpe_dp, fpe_qp = map(int, fields[8:])
        rows.append({"point": point, "order": order, "status": code,
                     "coefficients": [x if math.isfinite(x) else None for x in numbers[:4]],
                     "accuracy": numbers[4] if math.isfinite(numbers[4]) else None,
                     "finite": bool(valid), "fpe_dp": bool(fpe_dp), "fpe_qp": bool(fpe_qp)})
    points = sorted({r["point"] for r in rows})
    if partial and points and sum(1 for r in rows if r["point"] == points[-1]) < ndim + 1:
        rows = [r for r in rows if r["point"] != points[-1]]
        points = points[:-1]
    if [(r["point"], r["order"]) for r in rows] != [(p, j) for p in points for j in range(ndim + 1)]:
        raise ValueError("Stress value rows are missing or reordered")
    return rows


def parse_routes(path):
    routes = {}
    for line in Path(path).read_text().splitlines():
        values = list(map(int, line.split()))
        if len(values) != 23:
            raise ValueError("Malformed stress route row")
        routes[values[0]] = {"status_before": values[1], "status_after": values[2], "delta": values[3:]}
    return routes


OUTPUTS = ("warm.dat", "stress-values.dat", "stress-routes.dat", "stress-timing.dat", "dimensions.dat")


def split_points(path):
    """Header count and the per-point line blocks of a points file."""
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    count = int(lines[0])
    if count <= 0 or (len(lines) - 1) % count:
        raise ValueError(f"Malformed points file {path}")
    legs = (len(lines) - 1) // count
    return [lines[1 + i * legs:1 + (i + 1) * legs] for i in range(count)]


def run_driver(campaign, config, case, lane, cpu, points, report, workspace, options, checked=True,
               timeout=7200., extra=None):
    """Clone the campaign export for one lane, render the stress driver and evaluate every point.

    Some reducers end the whole program with a Fortran STOP on a pathological point
    (CutTools: momentum-conservation flag CT692) and exit 0, or never return (CutTools
    on the most forward complex-mass small-Gram point). The driver prints STRESS-DONE only after its
    last point; when it is missing, or an invocation exceeds `timeout` seconds, the
    next point is recorded as stopping the process and the remaining points run in a
    fresh process.
    """
    source, chain = LANES[lane]
    warm = (extra or {}).get("warm") or bench.REPORT / "points" / f"{case['id']}.dat"
    report.mkdir(parents=True, exist_ok=False)
    if workspace.exists():
        shutil.rmtree(workspace)
    output = workspace / "outputs" / source / case["id"]
    env = bench.environment()
    record = {"campaign": campaign, "case": case, "lane": lane, "chain": chain, "source": source, "cpu": cpu,
              "created_utc": bench.utc_now(), "performance_eligible": False,
              "points_sha256": bench.digest(points), "warm_points_sha256": bench.digest(warm),
              "driver_sha256": bench.digest(DRIVER), "runner_sha256": bench.digest(Path(__file__)),
              **{k: v for k, v in (extra or {}).items() if k != "warm"}}
    blocks = split_points(points)
    try:
        with ExitStack() as stack:
            if 8 in chain:
                # Unlicensed Symbolica (NinjaRS) allows one restricted instance per lock directory.
                # TIRust's scoped OEM activation does not take that lock, so TIRust-only runs overlap.
                lock = stack.enter_context((ROOT / ".tirust-build/numerical-tests.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX)
            directory, _ = cohort.build_clone(bench.output_dir(source, case), output, report, env,
                                              routes=9 in chain, template=DRIVER, renderer=render)
            record["settings"] = cohort.configure_clone(workspace, source, case, chain, options)
            if checked:
                bench.validate_worker_modes(record["settings"])
            record["runtime"] = bench.runtime_identity(directory)
            remaining = list(range(1, len(blocks) + 1))
            rows, routes, stops, invocations = [], {}, [], 0
            while remaining:
                invocations += 1
                subset = report / f"points-run{invocations}.dat"
                subset.write_text(f"{len(remaining)}\n" + "".join("\n".join(blocks[i - 1]) + "\n" for i in remaining))
                for name in OUTPUTS:
                    (directory / name).unlink(missing_ok=True)
                command = ["taskset", "-c", str(cpu), "./check", str(warm), str(subset),
                           "checked" if checked else "unchecked"]
                log_path = report / f"driver-run{invocations}.log"
                with log_path.open("w") as log:
                    try:
                        completed = subprocess.run(command, cwd=directory, env=env, stdout=log,
                                                   stderr=subprocess.STDOUT, timeout=timeout)
                        returncode = completed.returncode
                    except subprocess.TimeoutExpired:
                        returncode = "timeout"
                for name in OUTPUTS:
                    if (directory / name).is_file():
                        shutil.copy2(directory / name, report / (name if invocations == 1 else f"run{invocations}-{name}"))
                ndim = int((directory / "dimensions.dat").read_text().split()[0])
                if invocations == 1:
                    record["warm_calls"] = len((directory / "warm.dat").read_text().splitlines())
                done = parse_values(directory / "stress-values.dat", ndim, partial=True) \
                    if (directory / "stress-values.dat").is_file() else []
                for row in done:
                    row["point"] = remaining[row["point"] - 1]
                rows += done
                if (directory / "stress-routes.dat").is_file():
                    complete = {row["point"] for row in done}
                    for k, route in parse_routes(directory / "stress-routes.dat").items():
                        if remaining[k - 1] in complete:
                            routes[remaining[k - 1]] = route
                finished = sorted({row["point"] for row in done})
                if returncode == 0 and "STRESS-DONE" in log_path.read_text():
                    break
                culprit = remaining[len(finished)] if len(finished) < len(remaining) else None
                tail = log_path.read_text(errors="replace").splitlines()[-3:]
                stops.append({"point": culprit, "exit_code": returncode,
                              "reason": "timeout" if returncode == "timeout" else "stop", "log_tail": tail})
                if culprit is None:
                    break
                remaining = remaining[len(finished) + 1:]
            rows.sort(key=lambda r: (r["point"], r["order"]))
            record.update(rows=rows, routes=routes, stopped_points=stops, invocations=invocations,
                          status="complete")
    except subprocess.TimeoutExpired as error:
        record.update(status="timeout", error=repr(error))
    except Exception as error:
        record.update(status="failed", error=repr(error))
    record["finished_utc"] = bench.utc_now()
    bench.dump(report / "result.json", record)
    return record


def run(campaign, case_id, lane, mode, cpu, timeout=3600.):
    from tirust_benchmark_campaign import activate_campaign
    activate_campaign(bench, campaign)
    config = bench.load_config()
    case = next((c for c in config["cases"] if c["id"] == case_id), EXTRA_CASES.get(case_id))
    if case is None:
        raise ValueError(f"Unknown case {case_id}")
    source, chain = LANES[lane]
    if mode not in ("auto", "qp") or (mode == "qp" and (len(chain) != 1 or chain[0] not in QP_BACKENDS)):
        raise ValueError("Forced QP needs a single QP-capable backend (1, 6, 8 or 9)")
    point_dir = bench.REPORT / ("stress-points" if case_id in EXTRA_CASES else "points")
    stress = point_dir / f"{case_id}-stress.dat"
    metadata = json.loads(stress.with_suffix(".json").read_text())
    if bench.digest(stress) != metadata["sha256"]:
        raise ValueError("Stress point stream changed")
    label = f"{lane}-{mode}"
    options = json.loads(json.dumps(config))
    if mode == "qp":
        options["madloop_parameters"].update(CTModeRun=4, CTModeInit=4)
    record = run_driver(campaign, config, case, lane, cpu, stress, bench.REPORT / "experiments/stress" / case_id / label,
                        bench.WORK / "stress" / case_id / label, options, checked=mode == "auto", timeout=timeout,
                        extra={"mode": mode, "warm": point_dir / f"{case_id}.dat",
                               "stress_points": [{"index": i + 1, "id": p["id"], "kind": p.get("kind")}
                                                 for i, p in enumerate(metadata["points"])]})
    print(json.dumps({"case": case_id, "lane": label, "status": record["status"],
                      "stops": [s["point"] for s in record.get("stopped_points", [])],
                      "codes": sorted({r["status"] for r in record.get("rows", [])})}), flush=True)
    return record


def point_rows(result, point):
    return [r for r in result.get("rows", []) if r["point"] == point]


def accepted_backend(rows, chain):
    """The backend a checked automatic result was accepted from, or None (refused/failed)."""
    if not rows or any(not r["finite"] or r["fpe_dp"] or r["fpe_qp"] for r in rows):
        return None
    try:
        decoded = validate_status(rows[0]["status"], chain)
    except ValueError:
        return None
    return decoded["backend"], decoded["precision"]


def comparable(rows):
    return rows and all(r["finite"] and None not in r["coefficients"] for r in rows)


def compare(reference, candidate, tolerance):
    left = [{"point": r["point"], "order": r["order"], "coefficients": r["coefficients"]} for r in reference]
    right = [{"point": r["point"], "order": r["order"], "coefficients": r["coefficients"]} for r in candidate]
    result = bench.compare_rows(left, right, tolerance)
    return {"passed": result["passed"], "maximum_relative_error": result["maximum_relative_error"],
            "ordinary_target_passed": result["ordinary_target_passed"],
            "madloop_error": madloop_error(reference, candidate)}


def madloop_error(reference, candidate):
    """The candidate's actual error in the measure of MadLoop's accuracy estimate (compute_accuracy):
    per order, the summed deviations of finite part and poles over a third of their summed magnitude."""
    worst = 0.0
    for left, right in zip(reference, candidate):
        deviation = sum(abs(a - b) for a, b in zip(left["coefficients"][1:], right["coefficients"][1:]))
        scale = abs(sum(left["coefficients"][1:])) / 3.0
        worst = max(worst, deviation / scale if scale else deviation)
    return worst


def summarize(campaign):
    from tirust_benchmark_campaign import activate_campaign
    activate_campaign(bench, campaign)
    config = bench.load_config()
    tolerance = config["relative_tolerance"]
    base = bench.REPORT / "experiments/stress"
    summary = {"campaign": campaign, "tolerance": tolerance, "references": list(REFERENCES), "points": []}
    blocking = {"tirust": 0, "fallback": 0}
    for case_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        results = {p.parent.name: json.loads(p.read_text()) for p in case_dir.glob("*/result.json")}
        if not results:
            continue
        points = next(iter(results.values()))["stress_points"]
        ratios = case_pole_ratios(case_dir.name, results, points, tolerance)
        for point in points:
            index = point["index"]
            entry = {"case": case_dir.name, **point, "lanes": {}, "double_pole_over_born": ratios}
            refs = [point_rows(results[r], index) for r in REFERENCES
                    if r in results and comparable(point_rows(results[r], index))]
            accepted = [(label, point_rows(results[label], index)) for label in ACCEPTED_REFERENCES
                        if label in results and results[label]["status"] == "complete"
                        and accepted_backend(point_rows(results[label], index), results[label]["chain"])]
            reference, entry["reference"], agreement = select_reference(refs, accepted, ratios, tolerance)
            if agreement is not None:
                entry["reference_agreement"] = agreement
            independents = independent_results(results, index, ratios, tolerance) if reference is None else []
            for label, result in sorted(results.items()):
                rows = point_rows(result, index)
                lane = {"status": result["status"], "codes": sorted({r["status"] for r in rows}),
                        "accuracy": rows[0]["accuracy"] if rows else None,
                        "stopped": next((("hang" if stop.get("reason") == "timeout" else "STOP")
                                         for stop in result.get("stopped_points", []) if stop["point"] == index), None)}
                if result["mode"] == "auto":
                    accepted_lane = accepted_backend(rows, result["chain"]) if result["status"] == "complete" else None
                    lane["accepted"] = list(accepted_lane) if accepted_lane else None
                else:
                    lane["accepted"] = ["unchecked-qp"] if comparable(rows) else None
                if lane["accepted"] and reference is not None and comparable(rows):
                    lane["vs_reference"] = compare(reference, rows, tolerance)
                elif lane["accepted"] and result["mode"] == "auto" and comparable(rows):
                    consensus = consensus_check(rows, independents, lane["accepted"][0],
                                                (result.get("settings") or {}).get("mlstabthres"), tolerance)
                    if consensus is not None:
                        lane["vs_independent"] = consensus
                if lane["accepted"]:
                    poles = pole_check(rows, ratios, tolerance)
                    if poles is not None:
                        lane["pole_check"] = poles
                if result["mode"] == "auto":
                    threshold = (result.get("settings") or {}).get("mlstabthres")
                    lane["mlstabthres"] = threshold
                    lane["classification"] = classify(lane, result["chain"], reference, threshold)
                    if 9 in result["chain"] and blocking_outcome(lane["classification"]):
                        blocking["tirust" if lane["accepted"][0] == 9 else "fallback"] += 1
                if 9 in result["chain"]:
                    route = result.get("routes", {}).get(str(index)) or result.get("routes", {}).get(index)
                    lane["route_delta"] = route["delta"] if route else None
                entry["lanes"][label] = lane
            summary["points"].append(entry)
    summary["blocking_wrong_accepted"] = blocking["tirust"] + blocking["fallback"]
    summary["blocking_by_source"] = blocking
    (base / "stress-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (base / "stress-summary.md").write_text(markdown(summary))
    print(markdown(summary))
    return summary


def universal_ratios(rows, tolerance):
    """Double pole / Born per order, if it is the same at every row (IR poles are kinematics independent)."""
    values = {}
    for row in rows:
        born, double = row["coefficients"][0], row["coefficients"][3]
        if born and double is not None:
            values.setdefault(row["order"], []).append(double / born)
    if not values:
        return None  # no Born (loop-induced) or no rows
    ratios = {}
    for order, ratio in values.items():
        median = sorted(ratio)[len(ratio) // 2]
        if max(deviation(x, median) for x in ratio) > tolerance:
            return None
        ratios[order] = median
    return ratios


def deviation(value, reference):
    return abs(value - reference) / abs(reference) if reference else abs(value)


def case_pole_ratios(case_id, results, points, tolerance):
    """The case's double-pole/Born ratios at ordinary points: ordinary-matrix baseline, else stress anchors."""
    observation = bench.REPORT / "experiments/numerical" / f"ordinary-{case_id}" / "baseline-1/raw/observation.json"
    if observation.is_file():
        rows = json.loads(observation.read_text())["rows"]
    else:
        ordinary = {p["index"] for p in points if p.get("kind") == "ordinary"}
        rows = [r for r in results.get(REFERENCES[0], {}).get("rows", []) if r["point"] in ordinary]
    ratios = universal_ratios(rows, tolerance)
    return {str(k): v for k, v in ratios.items()} if ratios else None


def pole_check(rows, ratios, tolerance):
    """Compare a value's double pole/Born with the case's ordinary-point ratio; None when not applicable."""
    if not ratios or not comparable(rows):
        return None
    deviations = [deviation(r["coefficients"][3] / r["coefficients"][0], ratios[str(r["order"])])
                  for r in rows if r["coefficients"][0] and str(r["order"]) in ratios]
    if not deviations:
        return None
    return {"passed": max(deviations) <= tolerance, "maximum_relative_deviation": max(deviations)}


def select_reference(qp_refs, accepted, ratios, tolerance):
    """(reference rows or None, label, QP agreement or None).

    qp_refs: comparable forced-QP rows of REFERENCES; accepted: [(lane label, rows)] of
    stability-checked independent reducers. A reference must pass the double-pole check,
    and the fallback needs two agreeing reducers: one reducer cannot verify itself.
    """
    agreement = None
    sound = lambda rows: (pole_check(rows, ratios, tolerance) or {"passed": True})["passed"]
    if len(qp_refs) == len(REFERENCES):
        agreement = compare(qp_refs[0], qp_refs[1], tolerance)
        if agreement["passed"]:
            if sound(qp_refs[0]):
                return qp_refs[0], "QP CutTools = QP Ninja", agreement
            return None, "QP CutTools = QP Ninja, wrong double pole", agreement
    backend = lambda label: label.split("-")[1]
    wrong = [label for label, rows in accepted if not sound(rows)]
    accepted = [(label, rows) for label, rows in accepted if sound(rows)]
    suffix = f"; wrong double pole: {', '.join(backend(label) for label in wrong)}" if wrong else ""
    if len(accepted) >= 2 and all(compare(accepted[0][1], rows, tolerance)["passed"] for _, rows in accepted[1:]):
        return accepted[0][1], "checked: " + " = ".join(backend(label) for label, _ in accepted) + suffix, agreement
    if len(accepted) == 1:
        return None, f"unverified: {backend(accepted[0][0])}" + suffix, agreement
    if accepted:
        return None, "checked results disagree" + suffix, agreement
    return None, "unavailable or disputed" + suffix, agreement


def lane_backend(label):
    """The single backend of a lane label ("candidate-8-qp" -> 8), None for the chain."""
    parts = label.split("-")
    return int(parts[1]) if len(parts) > 2 and parts[1] in ("1", "6", "7", "8", "9") else None


def independent_results(results, index, ratios, tolerance):
    """[(backend, rows)] of comparable non-TIRust values at a point: forced QP of 1, 6 and 8, and the
    stability-checked values of ACCEPTED_REFERENCES. Values with a wrong double pole are left out."""
    found = []
    for label in INDEPENDENT_QP + ACCEPTED_REFERENCES:
        result = results.get(label)
        if not result or result["status"] != "complete":
            continue
        rows = point_rows(result, index)
        if not comparable(rows):
            continue
        if result["mode"] == "auto" and not accepted_backend(rows, result["chain"]):
            continue
        if not (pole_check(rows, ratios, tolerance) or {"passed": True})["passed"]:
            continue
        found.append((lane_backend(label), rows))
    return found


def consensus_check(rows, independents, own_backend, threshold, tolerance):
    """Without a reference: is the value far outside the spread of the independent results?

    Uses the independent values of other reducers only (a reducer cannot vouch for itself).
    The value is outside when even the nearest of them is more than max(MLStabThres or the
    tolerance, 10 x their largest pairwise difference) away, in MadLoop's accuracy measure.
    None if fewer than two such values exist.
    """
    others = [r for backend, r in independents if backend != own_backend]
    if len(others) < 2:
        return None
    spread = max(madloop_error(a, b) for i, a in enumerate(others) for b in others[i + 1:])
    distance = min(madloop_error(r, rows) for r in others)
    bound = max(threshold or tolerance, 10.0 * spread)
    return {"count": len(others), "spread": spread, "minimum_madloop_error": distance, "bound": bound,
            "outside": distance > bound}


def classify(lane, chain, reference, threshold=None):
    """TIRust may refuse; any accepted or chain-rescued value must be right and as accurate as claimed.

    WRONG: wrong double pole, disagrees with the reference, or (without a reference) lies far
    outside the spread of every independent reducer's value. INACCURATE: agrees within the
    campaign tolerance, but its actual error (in MadLoop's accuracy measure) exceeds the lane's
    MLStabThres (`threshold`) while its claimed accuracy does not, so MadLoop's stability test
    accepted it on a false estimate.
    """
    if lane["status"] != "complete":
        return "driver-" + lane["status"]
    if not lane["accepted"]:
        verb = "refused" if chain == [9] else "unrescued"
        return verb + (f" (process {lane['stopped']})" if lane.get("stopped") else "")
    backend = lane["accepted"][0]
    label = "accepted" if backend == 9 else f"rescued-by-{backend}" if 9 in chain else f"accepted-by-{backend}"
    if lane.get("pole_check") and not lane["pole_check"]["passed"]:
        return label + "-WRONG"
    if reference is None or "vs_reference" not in lane:
        if (lane.get("vs_independent") or {}).get("outside"):
            return label + "-WRONG"
        return label + "-unreferenced"
    if not lane["vs_reference"]["passed"]:
        return label + "-WRONG"
    error, accuracy = lane["vs_reference"].get("madloop_error"), lane.get("accuracy")
    if threshold and error is not None and error > threshold and accuracy is not None and 0 <= accuracy <= threshold:
        return label + "-INACCURATE"
    return label + "-agrees"


def blocking_outcome(classification):
    return classification.endswith(("-WRONG", "-INACCURATE"))


def markdown(summary):
    lines = ["| Case | Point | Kind | Reference | TIRust (9) | Chain 9|6|7|1 | TIRust forced QP vs ref |",
             "|---|---|---|---|---|---|---|"]
    for entry in summary["points"]:
        lanes = entry["lanes"]
        tirust = lanes.get("candidate-9-auto", {})
        chain = lanes.get("chain-9671-auto", {})
        qp = lanes.get("candidate-9-qp", {})
        qp_text = ("agrees" if qp.get("vs_reference", {}).get("passed") else
                   "differs" if "vs_reference" in qp else "n/a")
        if "vs_reference" in qp:
            qp_text += f" ({qp['vs_reference']['maximum_relative_error']:.1e})"
        lines.append(f"| {entry['case']} | {entry['id']} | {entry.get('kind')} | {entry['reference']} | "
                     f"{tirust.get('classification', 'pending')} {tirust.get('codes', '')} | "
                     f"{chain.get('classification', 'pending')} {chain.get('codes', '')} | {qp_text} |")
    lines.append("")
    by_source = summary.get("blocking_by_source", {})
    lines.append(f"Blocking wrong-accepted outcomes: {summary['blocking_wrong_accepted']}"
                 + (f" (accepted from TIRust: {by_source.get('tirust', 0)}, from a fallback reducer of the chain: "
                    f"{by_source.get('fallback', 0)})" if by_source else ""))
    details = {"tirust": [], "other": []}
    for entry in summary["points"]:
        for label, lane in sorted(entry["lanes"].items()):
            if not blocking_outcome(lane.get("classification", "")):
                continue
            text = f"- {entry['case']} {entry['id']}, {label} {lane['codes']}: {lane['classification']}"
            if lane.get("pole_check") and not lane["pole_check"]["passed"]:
                text += f"; double pole/Born off by {lane['pole_check']['maximum_relative_deviation']:.2e}"
            if "vs_reference" in lane:
                text += f"; {lane['vs_reference']['maximum_relative_error']:.2e} from the reference"
                if lane["classification"].endswith("-INACCURATE"):
                    text += (f", actual error {lane['vs_reference']['madloop_error']:.1e} in MadLoop's accuracy"
                             f" measure, above MLStabThres {lane['mlstabthres']:.0e}")
            elif (lane.get("vs_independent") or {}).get("outside"):
                consensus = lane["vs_independent"]
                text += (f"; {consensus['minimum_madloop_error']:.1e} from the nearest of {consensus['count']}"
                         f" independent values in MadLoop's accuracy measure (their largest difference"
                         f" {consensus['spread']:.1e})")
            if lane.get("accuracy") is not None:
                text += f"; claimed accuracy {lane['accuracy']:.1e}"
            details["tirust" if label.startswith(("candidate-9", "chain-9671")) else "other"].append(text)
    for key, title in (("tirust", "Blocking outcomes"), ("other", "Wrong values accepted by reducers without TIRust")):
        if details[key]:
            lines += ["", f"{title}:", ""] + details[key]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "summarize"))
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--case")
    parser.add_argument("--lane", choices=sorted(LANES))
    parser.add_argument("--mode", choices=("auto", "qp"), default="auto")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--timeout", type=float, default=3600., help="per driver invocation, seconds")
    args = parser.parse_args(argv)
    if Path.cwd() != bench.ROOT:
        parser.error(f"Run from {bench.ROOT}")
    if args.command == "run":
        if not (args.case and args.lane and args.cpu is not None):
            parser.error("run needs --case, --lane and --cpu")
        result = run(args.campaign, args.case, args.lane, args.mode, args.cpu, args.timeout)
        return 0 if result["status"] == "complete" else 1
    summary = summarize(args.campaign)
    return 1 if summary["blocking_wrong_accepted"] else 0


if __name__ == "__main__":
    sys.exit(main())
