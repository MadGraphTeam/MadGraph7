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
at the campaign tolerance. TIRust (alone or as first backend of the 9|6|7|1
chain) may refuse a stress point, but an accepted value must agree with the
reference: "wrong-accepted" is the only blocking outcome.

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
                      "     $ STRESS_RA-STRESS_RB"),
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


def parse_values(path, ndim):
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
    (CutTools: momentum-conservation flag CT692) and exit 0. The driver prints
    STRESS-DONE only after its last point; when it is missing, the next point is
    recorded as stopping the process and the remaining points run in a fresh process.
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
                    completed = subprocess.run(command, cwd=directory, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, timeout=timeout)
                for name in OUTPUTS:
                    if (directory / name).is_file():
                        shutil.copy2(directory / name, report / (name if invocations == 1 else f"run{invocations}-{name}"))
                ndim = int((directory / "dimensions.dat").read_text().split()[0])
                if invocations == 1:
                    record["warm_calls"] = len((directory / "warm.dat").read_text().splitlines())
                done = parse_values(directory / "stress-values.dat", ndim) if (directory / "stress-values.dat").is_file() else []
                for row in done:
                    row["point"] = remaining[row["point"] - 1]
                rows += done
                if (directory / "stress-routes.dat").is_file():
                    for k, route in parse_routes(directory / "stress-routes.dat").items():
                        routes[remaining[k - 1]] = route
                finished = sorted({row["point"] for row in done})
                if completed.returncode == 0 and "STRESS-DONE" in log_path.read_text():
                    break
                culprit = remaining[len(finished)] if len(finished) < len(remaining) else None
                tail = log_path.read_text().splitlines()[-3:]
                stops.append({"point": culprit, "exit_code": completed.returncode, "log_tail": tail})
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


def run(campaign, case_id, lane, mode, cpu, timeout=7200.):
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
            "ordinary_target_passed": result["ordinary_target_passed"]}


def summarize(campaign):
    from tirust_benchmark_campaign import activate_campaign
    activate_campaign(bench, campaign)
    config = bench.load_config()
    tolerance = config["relative_tolerance"]
    base = bench.REPORT / "experiments/stress"
    summary = {"campaign": campaign, "tolerance": tolerance, "references": list(REFERENCES), "points": []}
    blocking = 0
    for case_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        results = {p.parent.name: json.loads(p.read_text()) for p in case_dir.glob("*/result.json")}
        if not results:
            continue
        points = next(iter(results.values()))["stress_points"]
        for point in points:
            index = point["index"]
            entry = {"case": case_dir.name, **point, "lanes": {}}
            refs = [point_rows(results[r], index) for r in REFERENCES
                    if r in results and comparable(point_rows(results[r], index))]
            reference = None
            if len(refs) == len(REFERENCES):
                agreement = compare(refs[0], refs[1], tolerance)
                entry["reference_agreement"] = agreement
                reference = refs[0] if agreement["passed"] else None
            entry["reference"] = "QP CutTools = QP Ninja" if reference else "unavailable or disputed"
            if reference is None:
                accepted = [(label, point_rows(results[label], index)) for label in ACCEPTED_REFERENCES
                            if label in results and results[label]["status"] == "complete"
                            and accepted_backend(point_rows(results[label], index), results[label]["chain"])]
                if accepted and all(compare(accepted[0][1], rows, tolerance)["passed"] for _, rows in accepted[1:]):
                    reference = accepted[0][1]
                    entry["reference"] = "checked: " + " = ".join(label.split("-")[1] for label, _ in accepted)
                elif accepted:
                    entry["reference"] = "checked results disagree"
            for label, result in sorted(results.items()):
                rows = point_rows(result, index)
                lane = {"status": result["status"], "codes": sorted({r["status"] for r in rows}),
                        "accuracy": rows[0]["accuracy"] if rows else None,
                        "stopped": any(stop["point"] == index for stop in result.get("stopped_points", []))}
                if result["mode"] == "auto":
                    accepted = accepted_backend(rows, result["chain"]) if result["status"] == "complete" else None
                    lane["accepted"] = list(accepted) if accepted else None
                else:
                    lane["accepted"] = ["unchecked-qp"] if comparable(rows) else None
                if lane["accepted"] and reference is not None and comparable(rows):
                    lane["vs_reference"] = compare(reference, rows, tolerance)
                if 9 in result["chain"]:
                    route = result.get("routes", {}).get(str(index)) or result.get("routes", {}).get(index)
                    lane["route_delta"] = route["delta"] if route else None
                    if result["mode"] == "auto":
                        lane["classification"] = classify(lane, result["chain"], reference)
                        blocking += lane["classification"].startswith("WRONG")
                entry["lanes"][label] = lane
            summary["points"].append(entry)
    summary["blocking_wrong_accepted"] = blocking
    (base / "stress-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (base / "stress-summary.md").write_text(markdown(summary))
    print(markdown(summary))
    return summary


def classify(lane, chain, reference):
    """TIRust may refuse; any accepted or chain-rescued value must match the reference."""
    if lane["status"] != "complete":
        return "driver-" + lane["status"]
    if not lane["accepted"]:
        verb = "refused" if chain == [9] else "unrescued"
        return verb + (" (process STOP)" if lane.get("stopped") else "")
    backend = lane["accepted"][0]
    label = "accepted" if backend == 9 else f"rescued-by-{backend}"
    if reference is None or "vs_reference" not in lane:
        return label + "-unreferenced"
    return label + ("-agrees" if lane["vs_reference"]["passed"] else "-WRONG")


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
    lines.append(f"Blocking wrong-accepted outcomes: {summary['blocking_wrong_accepted']}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "summarize"))
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--case")
    parser.add_argument("--lane", choices=sorted(LANES))
    parser.add_argument("--mode", choices=("auto", "qp"), default="auto")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--timeout", type=float, default=7200.)
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
