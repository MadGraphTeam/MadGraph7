#!/usr/bin/env python3
"""Appendix-C-style stability study (arXiv:1604.01363) for a registered campaign.

The plotted quantity is MadLoop's internal stability estimate (loop-direction and
Lorentz-rotation tests, the returned ACC), not an independent error. Two variants
run every reducer on the same frozen cut-qualified points:

  raw      CTModeRun=-1 with MLStabThres=1e30: every point stays in the first
           (DP) evaluation; no precision rescue or next-library fallback occurs.
           A reduction failure keeps ACC=1 with the DP failure latch set and is
           reported as unavailable, never as a successful sample.
  rescued  the campaign's natural policy (MLStabThres from cases.json); accepted
           DP or QP estimates.

Native numerical caches are disabled (TIRustGlobalCache=0, COLLIERGlobalCache=0)
as in the earlier pilot. TIRust's "raw DP" includes its internal double-double
and MPFR tensor tiers, which take binary64 inputs; per-point bridge routes record
which tier was used.

  prepare                       freeze the pilot's 100-point samples into the campaign
  run --case C --lane L --variant V --cpu N
  analyse                       data + figures under <report>/stability/
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
import tirust_stress as stress  # noqa: E402

PILOT = ROOT / "docs/benchmarkfolder/stability_appendix_c/raw-dp-pilot-5-1/points"
CASES = ("qcd_n2_gg_ttx", "qcd_n3_uux_zgg", "qcd_n4_uux_zzgg", "ew_n2_epem_mupmum")
REDUCERS = (1, 6, 7, 8, 9)
NAMES = {1: "CutTools", 6: "Ninja", 7: "COLLIER", 8: "NinjaRS", 9: "TIRust"}
VARIANTS = ("raw", "rescued")
RAW_THRESHOLD = 1e30


def activate(campaign):
    from tirust_benchmark_campaign import activate_campaign
    activate_campaign(bench, campaign)
    return bench.load_config()


def prepare(campaign):
    activate(campaign)
    target = bench.REPORT / "stability/points"
    target.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        source = PILOT / f"{case}.dat"
        destination = target / f"{case}.dat"
        if destination.exists():
            if bench.digest(destination) != bench.digest(source):
                raise ValueError(f"Frozen stability points differ for {case}")
            continue
        shutil.copy2(source, destination)
        count = int(destination.read_text().split("\n", 1)[0])
        bench.dump(destination.with_suffix(".json"), {
            "source": str(source.relative_to(ROOT)), "source_metadata": str(source.with_suffix(".json").relative_to(ROOT)),
            "sha256": bench.digest(destination), "count": count,
            "points": [{"id": f"stability-{i:04d}"} for i in range(1, count + 1)],
            "scope": "Cut-qualified 1 TeV points of the earlier raw-DP pilot (pT > 50 GeV, Delta R > 0.5)"})


# Shared driver core (resumes after a reducer's Fortran STOP); see tirust_stress.run_driver.
run_driver = stress.run_driver


def variant_options(config, variant):
    options = json.loads(json.dumps(config))
    options["madloop_parameters"]["COLLIERGlobalCache"] = 0
    options["candidate_parameters"]["TIRustGlobalCache"] = 0
    if variant == "raw":
        options["madloop_parameters"]["MLStabThres"] = RAW_THRESHOLD
    return options


def run(campaign, case_id, lane, variant, cpu, timeout=7200.):
    config = activate(campaign)
    case = next(c for c in config["cases"] if c["id"] == case_id)
    points = bench.REPORT / "stability/points" / f"{case_id}.dat"
    metadata = json.loads(points.with_suffix(".json").read_text())
    if bench.digest(points) != metadata["sha256"]:
        raise ValueError("Frozen stability points changed")
    report = bench.REPORT / "experiments/stability" / variant / case_id / lane
    workspace = bench.WORK / "stability" / variant / case_id / lane
    record = run_driver(campaign, config, case, lane, cpu, points, report, workspace,
                        variant_options(config, variant), timeout=timeout)
    print(json.dumps({"case": case_id, "lane": lane, "variant": variant, "status": record["status"]}), flush=True)
    return record


def point_outcome(rows, variant, backend, route=None):
    """Total-answer (order 0) estimate of one point, or the reason it is unavailable."""
    if not rows:
        return {"available": False, "reason": "missing"}
    row = rows[0]
    if any(not r["finite"] for r in rows):
        return {"available": False, "reason": "nonfinite", "status": row["status"]}
    if any(r["fpe_dp"] or r["fpe_qp"] for r in rows):
        return {"available": False, "reason": "reduction-failure-latch", "status": row["status"]}
    h, t, u = row["status"] // 100, row["status"] // 10 % 10, row["status"] % 10
    accuracy = row["accuracy"]
    if accuracy is None or accuracy < 0 or t not in (1, 2):
        return {"available": False, "reason": "unchecked-or-initialization", "status": row["status"]}
    if variant == "raw" and h != 2:
        return {"available": False, "reason": "not-a-dp-result", "status": row["status"]}
    if variant == "rescued" and h not in (2, 3):
        return {"available": False, "reason": "unstable-after-rescue", "status": row["status"],
                "accuracy": accuracy}
    outcome = {"available": True, "accuracy": accuracy, "status": row["status"],
               "precision": "QP" if h == 3 else "DP"}
    if backend == 9 and route is not None:
        category = cohort.route_category("dp", route)
        outcome["tirust_route"] = category if h == 2 else "qp"
    return outcome


def analyse(campaign):
    activate(campaign)
    base = bench.REPORT / "experiments/stability"
    data = {"campaign": campaign, "quantity": "MadLoop internal stability estimate (ACC) of the total answer",
            "variants": {}}
    for variant in VARIANTS:
        cases = {}
        for case in CASES:
            reducers = {}
            for backend in REDUCERS:
                path = base / variant / case / f"candidate-{backend}" / "result.json"
                if not path.is_file():
                    continue
                result = json.loads(path.read_text())
                points = json.loads((bench.REPORT / "stability/points" / f"{case}.json").read_text())["count"]
                outcomes = []
                for index in range(1, points + 1):
                    rows = [r for r in result.get("rows", []) if r["point"] == index]
                    routes = result.get("routes", {})
                    route = routes.get(str(index)) or routes.get(index)
                    outcome = point_outcome(rows, variant, backend, route) if result["status"] == "complete" \
                        else {"available": False, "reason": "driver-" + result["status"]}
                    outcomes.append(outcome)
                reducers[str(backend)] = {"name": NAMES[backend], "points": points, "outcomes": outcomes,
                                          "available": sum(o["available"] for o in outcomes),
                                          "reasons": count_reasons(outcomes)}
            if reducers:
                cases[case] = reducers
        data["variants"][variant] = cases
    out = bench.REPORT / "stability"
    out.mkdir(parents=True, exist_ok=True)
    (out / "stability-data.json").write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    rows = table(data)
    (out / "stability-table.json").write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    (out / "stability-table.md").write_text(markdown(rows))
    policy = bench.load_config()["madloop_parameters"]["MLStabThres"]
    data["figures"] = figures(data, out, policy)
    return data


def count_reasons(outcomes):
    reasons = {}
    for outcome in outcomes:
        key = outcome.get("reason") or outcome.get("tirust_route") or outcome.get("precision")
        reasons[key] = reasons.get(key, 0) + 1
    return reasons


def tail(outcomes, threshold):
    """(known exceedances k, unavailable m, total) for count(acc > threshold)."""
    known = sum(1 for o in outcomes if o["available"] and o["accuracy"] > threshold)
    missing = sum(1 for o in outcomes if not o["available"])
    return known, missing, len(outcomes)


# Validated categorical palette (fixed reducer order; dataviz reference palette,
# light mode on white); line styles are the secondary encoding for print and CVD.
STYLE = {1: ("#2a78d6", "-"), 6: ("#eb6834", (0, (5, 2))), 7: ("#1baf7a", (0, (6, 2, 1, 2))),
         8: ("#eda100", (0, (1, 1.2))), 9: ("#e87ba4", (0, (8, 2, 2, 2, 2, 2)))}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
LABELS = {"qcd_n2_gg_ttx": "QCD  g g → t t̄", "qcd_n3_uux_zgg": "QCD  u ū → Z g g",
          "qcd_n4_uux_zzgg": "QCD  u ū → Z Z g g", "ew_n2_epem_mupmum": "EW  e⁺e⁻ → μ⁺μ⁻"}
TABLE_THRESHOLDS = (1e-3, 1e-6, 1e-9)


def grid():
    return [10 ** (-16 + i / 20) for i in range(16 * 20 + 1)]


def figures(data, out, policy_threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": .6})
    thresholds = grid()
    written = []
    for variant, cases in data["variants"].items():
        if not cases:
            continue
        fig = plt.figure(figsize=(7.4, 8.2))
        outer = fig.add_gridspec(2, 2, hspace=.42, wspace=.28, top=.87, bottom=.07, left=.09, right=.98)
        handles = {}
        for slot, case in enumerate(c for c in CASES if c in cases):
            inner = outer[slot // 2, slot % 2].subgridspec(2, 1, height_ratios=(3, 1.2), hspace=.06)
            top, ratio = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            reducers = cases[case]
            ninja = reducers.get("6")
            for backend in REDUCERS:
                entry = reducers.get(str(backend))
                if entry is None:
                    continue
                color, dash = STYLE[backend]
                n = entry["points"]
                known = [tail(entry["outcomes"], t)[0] / n for t in thresholds]
                upper = [sum(tail(entry["outcomes"], t)[:2]) / n for t in thresholds]
                label = f"{entry['name']} ({entry['available']}/{n} available)"
                line, = top.step(thresholds, [k if k > 0 else float("nan") for k in known], where="post",
                                 color=color, linestyle=dash, linewidth=1.4, label=label)
                handles.setdefault(backend, line)
                if any(u > k for u, k in zip(upper, known)):
                    top.fill_between(thresholds, known, upper, step="post", color=color, alpha=.18, linewidth=0)
                if ninja and backend != 6 and entry["available"] == n and ninja["available"] == ninja["points"]:
                    base = [tail(ninja["outcomes"], t)[0] / ninja["points"] for t in thresholds]
                    values = [k / b if b > 0 else float("nan") for k, b in zip(known, base)]
                    ratio.step(thresholds, values, where="post", color=color, linestyle=dash, linewidth=1.2)
            for axis in (top, ratio):
                axis.set_xscale("log")
                axis.set_xlim(1e-16, 1)
                axis.grid(True, color=GRID, linewidth=.5)
                axis.axvline(policy_threshold, color=MUTED, linewidth=.7, linestyle=":")
                axis.axvline(1e-3, color=MUTED, linewidth=.7, linestyle="--")
            top.set_yscale("log")
            top.set_ylim(.8 / 100, 1.25)
            top.axhline(1 / 100, color=MUTED, linewidth=.5, linestyle=(0, (2, 2)))
            top.set_title(LABELS.get(case, case), color=INK, fontsize=9, loc="left")
            top.set_ylabel("fraction with estimate > Δ")
            top.tick_params(labelbottom=False)
            ratio.axhline(1, color=MUTED, linewidth=.6)
            ratio.set_yscale("log")
            ratio.set_ylim(1e-2, 1e2)
            ratio.set_ylabel("ratio to Ninja")
            ratio.set_xlabel("Δ (MadLoop stability estimate)")
        fig.legend([handles[b] for b in REDUCERS if b in handles],
                   [NAMES[b] for b in REDUCERS if b in handles], loc="upper center", ncol=5, frameon=False,
                   bbox_to_anchor=(.5, .99), fontsize=8.5, handlelength=3.2)
        title = {"raw": "Raw double precision (no rescue; TIRust includes its internal DD/MPFR tensor tiers)",
                 "rescued": f"After automatic rescue (MLStabThres = {policy_threshold:g})"}[variant]
        fig.text(.5, .945, title, ha="center", color=INK, fontsize=9.5)
        fig.text(.5, .924, "Shaded: unknown tail from unavailable points.  Dotted line: campaign threshold; "
                 "dashed: 10⁻³; horizontal dashes: 1/100 resolution.", ha="center", color=MUTED, fontsize=7.2)
        for suffix in ("png", "pdf"):
            path = out / f"stability-{variant}.{suffix}"
            fig.savefig(path, dpi=200, metadata={"CreationDate": None} if suffix == "pdf" else None)
            written.append(path.name)
        plt.close(fig)
    return written


def table(data):
    rows = []
    for variant, cases in data["variants"].items():
        for case, reducers in cases.items():
            for backend, entry in sorted(reducers.items(), key=lambda kv: int(kv[0])):
                bounds = {}
                for threshold in TABLE_THRESHOLDS:
                    known, missing, total = tail(entry["outcomes"], threshold)
                    bounds[f"{threshold:g}"] = [known / total, (known + missing) / total]
                rows.append({"variant": variant, "case": case, "reducer": entry["name"], "points": entry["points"],
                             "available": entry["available"], "reasons": entry["reasons"], "tail": bounds})
    return rows


def markdown(rows):
    lines = ["| Variant | Process | Reducer | Available | > 1e-3 | > 1e-6 | > 1e-9 | Breakdown |", "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        cells = []
        for key in ("0.001", "1e-06", "1e-09"):
            low, high = row["tail"][key]
            cells.append(f"{low:.2f}" if low == high else f"[{low:.2f}, {high:.2f}]")
        reasons = ", ".join(f"{k}: {v}" for k, v in sorted(row["reasons"].items()))
        lines.append(f"| {row['variant']} | {row['case']} | {row['reducer']} | {row['available']}/{row['points']} | "
                     + " | ".join(cells) + f" | {reasons} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("prepare", "run", "analyse"))
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--lane", choices=[f"candidate-{b}" for b in REDUCERS])
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--timeout", type=float, default=7200.)
    args = parser.parse_args(argv)
    if Path.cwd() != bench.ROOT:
        parser.error(f"Run from {bench.ROOT}")
    if args.command == "prepare":
        prepare(args.campaign)
    elif args.command == "run":
        if not (args.case and args.lane and args.variant and args.cpu is not None):
            parser.error("run needs --case, --lane, --variant and --cpu")
        return 0 if run(args.campaign, args.case, args.lane, args.variant, args.cpu, args.timeout)["status"] == "complete" else 1
    else:
        data = analyse(args.campaign)
        for variant, cases in data["variants"].items():
            for case, reducers in cases.items():
                print(variant, case, {r["name"]: (r["available"], r["reasons"]) for r in reducers.values()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
