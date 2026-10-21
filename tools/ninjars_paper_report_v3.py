"""Paper-analogue tables and DP-only figures, separate from acceptance timing.

No model, pickle, compiler or numerical engine is executed. Production readers
must validate their raw observations before this module consumes their summaries.
"""
import argparse
from collections import Counter
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
TOOLS = {"1": "CutTools", "6": "Ninja", "7": "COLLIER", "8": "NinjaRS"}
COLORS = {"1": "#356b92", "6": "#c78320", "7": "#8064a2", "8": "#2e8b57"}
SCHEMA = "ninjars-paper-comparison-report-v1"


class Inputs(dict):
    """Support the two concrete producer readers' retained-file interfaces."""
    def add(self, filename):
        retain(filename, self)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(filename):
    result = Path(filename)
    result = (result if result.is_absolute() else ROOT / result).resolve()
    require(result.is_relative_to(ROOT), "Paper report input/output must stay inside the repository")
    return result


def retain(filename, inputs, expected=None):
    filename = path(filename)
    raw = filename.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        require(actual == expected["sha256"] and len(raw) == expected["bytes"],
                "Changed generated-source inventory input: " + str(filename.relative_to(ROOT)))
    inputs[str(filename.relative_to(ROOT))] = actual
    return raw


def parameters(source, names=None):
    """Read literal integer PARAMETER values; never evaluate Fortran text."""
    active = []
    for line in source.splitlines():
        if line and line[0] in "cC*!":
            continue
        active.append(line[6:] if len(line) >= 6 else line)
    values = {}
    for body in re.findall(r"\bPARAMETER\s*\(([^)]+)\)", " ".join(active), re.I):
        for name, number in re.findall(r"\b([A-Z][A-Z0-9_]*)\s*=\s*([+-]?\d+)\s*(?=,|$)", body, re.I):
            name, number = name.upper(), int(number)
            if names is not None and name not in names:
                continue
            require(name not in values or values[name] == number, "Contradictory generated integer parameter " + name)
            values[name] = number
    return values


def process_inventory(output, generation, compilation, compiled_manifest, generation_log=None, inputs=None):
    """Extract bound generated counts without loading matrix-element pickles."""
    inputs = {} if inputs is None else inputs
    output = path(output)
    generated = json.loads(retain(generation, inputs))
    compiled = json.loads(retain(compilation, inputs))
    manifest = json.loads(retain(compiled_manifest, inputs))
    encoded = json.dumps(manifest["files"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    require(hashlib.sha256(encoded).hexdigest() == manifest["files_sha256"], "Generated manifest identity differs")
    require(compiled["generated_sources_sha256"] == manifest["files_sha256"], "Compilation source identity differs")
    directories = sorted(output.glob("SubProcesses/P*_*"))
    require(len(directories) == 1, "Process counts need one explicitly identified subprocess")
    directory = directories[0]

    def source(filename):
        relative = str(filename.relative_to(output))
        require(relative in manifest["files"], "Count source is absent from the compiled manifest")
        return retain(filename, inputs, manifest["files"][relative]).decode()

    matrix = directory / "loop_matrix.f"
    if not matrix.is_file():
        return {"status": "pending", "reason": "Generated loop_matrix.f is unavailable", "inputs_sha256": dict(inputs)}
    values = parameters(source(matrix), {"NLOOPS", "NLOOPGROUPS", "NBORNAMPS", "NCOMB"})
    diagram_ids = []
    for filename in sorted(directory.glob("coef_construction_*.f")):
        diagram_ids.extend(map(int, re.findall(r"Coefficient construction for loop diagram with ID\s+(\d+)", source(filename))))
    ids = sorted(set(diagram_ids))
    log_counts = None
    if generation_log is not None:
        log = retain(generation_log, inputs).decode()
        pattern = r"Contributing diagrams generated:\s*(\d+) Born,\s*(\d+)(?:\(\+(\d+)\))? loops,\s*(\d+) R2,\s*(\d+) UV"
        matches = re.findall(pattern, re.sub(r"\x1b\[[0-9;]*m", "", log))
        if len(matches) == 1:
            born, loops, equivalent, r2, uv = matches[0]
            log_counts = {"born": int(born), "loop_classes": int(loops), "identified_equivalent_copies": int(equivalent or 0),
                          "r2": int(r2), "uv": int(uv)}
            require(not ids or len(ids) == log_counts["loop_classes"], "Generated diagram IDs and CLI loop count disagree")
    coefficients = directory / "loop_max_coefs.inc"
    maximum_rank = None
    if coefficients.is_file():
        count = parameters(source(coefficients), {"LOOPMAXCOEFS"}).get("LOOPMAXCOEFS")
        ranks = [rank for rank in range(21) if math.comb(rank + 4, 4) == count]
        maximum_rank = ranks[0] if len(ranks) == 1 else None
    seconds = {}
    for key, record in (("generation_cli_seconds", generated), ("compilation_seconds", compiled)):
        value = record.get("seconds")
        require(value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0), "Invalid recorded wall time")
        seconds[key] = value
    return {"status": "derived", **seconds, "loop_diagram_classes": len(ids) if ids else
            log_counts["loop_classes"] if log_counts is not None else None,
            "identified_equivalent_copies": log_counts["identified_equivalent_copies"] if log_counts else None,
            "helas_loop_amplitudes": values.get("NLOOPS"), "reducible_loop_groups": values.get("NLOOPGROUPS"),
            "maximum_tensor_rank": maximum_rank, "born_amplitudes": values.get("NBORNAMPS"),
            "helicity_configurations": values.get("NCOMB"), "cli_diagram_counts": log_counts,
            "diagram_ids": ids, "definition": "Numerically distinct loop-diagram classes after MG7 identification; equivalent flavor copies enter by multiplier. UV/R2 excluded. NLOOPS counts HELAS loop amplitudes and is not a diagram count.",
            "rank_definition": "Exact inverse of LOOPMAXCOEFS = binomial(rank+4,4), for a four-dimensional symmetric polynomial; unknown counts stay missing.",
            "timing_scope": "Recorded generation CLI and compilation wall times; initialization is measured separately.",
            "inputs_sha256": dict(sorted(inputs.items()))}


def wilson95(k, n):
    require(type(n) is int and type(k) is int and n > 0 and 0 <= k <= n, "Invalid empirical count")
    z = 1.959963984540054
    p, denominator = k / n, 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [max(0., center-radius), min(1., center+radius)]


def tail(observations, thresholds):
    """Keep attempted failures/unbounded values in every finite-threshold tail."""
    require(observations, "No DP observations")
    ids, finite, failures, categories = [], [], 0, Counter()
    for row in observations:
        identity = row["point"]
        require(type(identity) is int or (isinstance(identity, str) and identity), "Invalid point identity")
        require(identity not in ids, "Duplicate DP point")
        ids.append(identity)
        classification, delta = row["classification"], row["delta"]
        require(isinstance(classification, str) and classification, "Missing DP classification")
        categories[classification] += 1
        if classification in ("finite", "all_zero"):
            require(type(delta) in (int, float) and math.isfinite(delta) and delta >= 0,
                    "Finite DP spread must be nonnegative and finite")
            require(classification != "all_zero" or delta == 0, "An all-zero observation has nonzero spread")
            finite.append(delta)
        else:
            require(delta is None, "Failed/unbounded DP spread must be null")
            failures += 1
    thresholds = list(thresholds)
    require(thresholds and all(type(t) in (int, float) and math.isfinite(t) and t > 0 for t in thresholds) and
            thresholds == sorted(set(thresholds)), "Logarithmic thresholds must be positive, finite and increasing")
    n, rows = len(ids), []
    upper = -math.expm1(math.log(.05)/n)
    for threshold in thresholds:
        k = failures + sum(delta > threshold for delta in finite)
        rows.append({"threshold": threshold, "k": k, "N": n, "fraction": k/n,
                     "wilson_interval95": wilson95(k, n), "zero_event_upper95": upper if k == 0 else None})
    return {"N": n, "point_ids": ids, "failure_or_unbounded": failures, "categories": dict(categories),
            "empirical_resolution": 1/n, "rows": rows,
            "interval_scope": "Binomial reference; population coverage requires an independent random sample. Fixed/boosted/targeted streams remain descriptive."}


def ninja_ratios(numerator, denominator):
    require(numerator["point_ids"] == denominator["point_ids"], "Ninja ratios require matched point identities and order")
    require([r["threshold"] for r in numerator["rows"]] == [r["threshold"] for r in denominator["rows"]],
            "Ninja ratios require matching thresholds")
    rows = []
    for a, b in zip(numerator["rows"], denominator["rows"]):
        require(a["N"] == b["N"], "Ninja ratios require the same complete denominator")
        value = a["k"]/b["k"] if b["k"] else None
        state = "finite" if b["k"] else "undefined_0_over_0" if a["k"] == 0 else "unbounded_positive_over_0"
        rows.append({"threshold": a["threshold"], "ratio": value, "classification": state,
                     "numerator_k": a["k"], "ninja_k": b["k"], "N": b["N"]})
    return rows


def figure_data(cases, thresholds, estimator="combined"):
    """Prepare plotting arrays; production summaries are revalidated upstream."""
    prepared = {}
    declared = list(thresholds)
    # Use the actual jump positions, rather than smoothing a coarsely sampled CDF.
    tail([{"point": 0, "classification": "all_zero", "delta": 0}], declared)
    for cid, case in cases.items():
        grid = sorted(set(declared + [row["delta"] for tool in case.get("tools", {}).values()
            for row in tool.get("scores", {}).get(estimator, [])
            if row.get("classification") == "finite" and type(row.get("delta")) in (int, float)
            and math.isfinite(row["delta"]) and declared[0] < row["delta"] < declared[-1]]))
        tools = {}
        for backend in TOOLS:
            tool = case.get("tools", {}).get(backend)
            if tool is None or tool.get("status") in ("pending", "unavailable"):
                tools[backend] = {"status": "pending" if tool is None else tool["status"], "tail": None, "ratio": None}
                continue
            observations = tool["scores"][estimator]
            result = tail(observations, grid)
            require(result["N"] == case["N"], "DP curve omitted a planned observation")
            tools[backend] = {"status": tool["status"], "tail": result, "ratio": None}
        ninja = tools["6"]["tail"]
        for value in tools.values():
            if value["tail"] is not None and ninja is not None:
                value["ratio"] = ninja_ratios(value["tail"], ninja)
        prepared[cid] = {"process": case.get("process", cid), "N": case["N"],
                         "group": case.get("group", "unspecified"), "declared_thresholds": declared,
                         "sample_kind": case.get("sample_kind", "fixed pilot stream"),
                         "helicity": case.get("helicity", "not supplied"), "tools": tools}
    return prepared


def plot_stability(cases, thresholds, destination, estimator="combined", title="DP-only self-consistency",
                   software_fixture=False):
    """Write vector plots with zero-tail bounds and real Ninja-denominator gaps."""
    destination = path(destination)
    require(not destination.exists(), "Preserve existing figures; choose a new destination")
    data = figure_data(cases, thresholds, estimator)
    require(data, "No stability cases to plot")
    cache = ROOT / ".ninjars-build/paper-report-mpl"
    cache.mkdir(parents=True, exist_ok=True)
    previous_cache = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str(cache)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        settings = {"font.family": "serif", "font.serif": ["Libertinus Serif", "DejaVu Serif"],
                    "font.size": 8, "text.color": "#203c60", "axes.labelcolor": "#203c60",
                    "axes.titlecolor": "#203c60", "svg.fonttype": "none", "svg.hashsalt": "madloop-paper-v3"}
        with plt.rc_context(settings):
            columns, rows = min(2, len(data)), math.ceil(len(data)/2)
            fig = plt.figure(figsize=(5.5*columns, 4.3*rows + .7))
            outer = fig.add_gridspec(rows, columns, hspace=.48, wspace=.25)
            for index, (cid, case) in enumerate(data.items()):
                inner = outer[index//columns, index%columns].subgridspec(2, 1, height_ratios=(3, 1), hspace=.08)
                top = fig.add_subplot(inner[0]); ratio = fig.add_subplot(inner[1], sharex=top)
                for backend, label in TOOLS.items():
                    tool = case["tools"][backend]
                    if tool["tail"] is None:
                        top.plot([], [], color=COLORS[backend], label=label + " (" + tool["status"] + ")")
                        continue
                    observed = tool["tail"]
                    x = [r["threshold"] for r in observed["rows"]]
                    y = [r["fraction"] if r["k"] else math.nan for r in observed["rows"]]
                    top.step(x, y, where="post", color=COLORS[backend], label=f"{label}: {observed['failure_or_unbounded']} failed/unbounded")
                    zeros = [r for r in observed["rows"] if r["k"] == 0]
                    if zeros:
                        first = zeros[0]
                        top.scatter([first["threshold"]], [first["zero_event_upper95"]], marker="v", facecolors="none",
                                    edgecolors=COLORS[backend], zorder=5)
                    if tool["ratio"] is not None:
                        ratio.step(x, [r["ratio"] if r["ratio"] is not None else math.nan for r in tool["ratio"]],
                                   where="post", color=COLORS[backend])
                top.set_xscale("log"); top.set_yscale("log"); ratio.set_xscale("log")
                top.set_ylim(min(.5/case["N"], .005), 1.15)
                top.axhline(1/case["N"], color="#808080", linewidth=.7, linestyle=":")
                top.set_xlim(thresholds[0], thresholds[-1]); top.tick_params(labelbottom=False)
                top.set_title(f"{case['process']} · N={case['N']} · {case['helicity']}", fontsize=9)
                top.set_ylabel("tail fraction (failures included)"); top.legend(fontsize=6.8, loc="upper right")
                ratio.axhline(1, color="#808080", linewidth=.7, linestyle="--")
                ratio.set_ylabel("/ Ninja"); ratio.set_xlabel("relative spread threshold Δ")
                ratio.set_ylim(bottom=0)
                for axis in (top, ratio):
                    axis.grid(True, which="major", linewidth=.4, color="#d7dfe7")
                    axis.spines[["top", "right"]].set_visible(False)
                ratio.text(.02, .94, "gaps: Ninja 0/N", transform=ratio.transAxes, va="top", fontsize=6.5)
            suffix = " · SOFTWARE FIXTURE, no physics" if software_fixture else ""
            fig.suptitle(title + suffix, fontsize=12, color="#203c60", y=.99)
            fig.text(.5, .015, "Step counts use fixed N; no smoothing or epsilon. Open triangles: zero events, exact one-sided 95% binomial upper reference.\n"
                     "Resolution is 1/N. Fixed/boosted/targeted samples are descriptive; ratio panels are linear and have no independent-sample CI.",
                     ha="center", fontsize=7)
            fig.subplots_adjust(top=.90, bottom=.14)
            destination.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(destination, format="svg", metadata={"Date": None, "Creator": "MadLoop NinjaRS paper comparison"})
            plt.close(fig)
    finally:
        if previous_cache is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = previous_cache
    return {"schema": "ninjars-dp-stability-figure-v1", "estimator": estimator, "cases": data,
            "svg_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "software_fixture": software_fixture,
            "qualification_eligible": False}


def component(filename, module_name, schema, expected, inputs):
    retain(filename, inputs)
    reader = importlib.import_module(module_name)
    value = reader.report_snapshot(path(filename), expected=expected, inputs=inputs)
    require(value.get("schema") == schema, "Wrong paper-comparison producer summary")
    inputs[str(path(reader.__file__).relative_to(ROOT))] = hashlib.sha256(Path(reader.__file__).read_bytes()).hexdigest()
    return value


def table_rows(timing, inventories):
    """Preserve physical rejection/status counts alongside measured elapsed cost."""
    cases = {} if timing is None else timing["cases"]
    return {cid: {"generated": inventories.get(cid, {"status": "pending"}),
                  "single_helicity": cases.get(cid, {"status": "pending"}),
                  "numerator_dp": timing.get("numerator_dp", {"status": "pending"}) if timing else {"status": "pending"},
                  "remaining": timing.get("remaining", {"status": "pending"}) if timing else {"status": "pending"},
                  "scope": "Single-helicity color-summed/rescue-on totals; failed-physics cost remains failed. Separate from full-helicity acceptance timing."}
            for cid in sorted(set(cases) | set(inventories))}


def plot_groups(snapshot, destination, software_fixture=False):
    """Separate QCD/EW/loop/stress figures; never substitute accepted rescue data."""
    require(snapshot["schema"] == SCHEMA, "Wrong paper report snapshot")
    dp = snapshot["stability"]
    require(dp is not None and dp.get("stream") == "forced_dp_self_consistency" and
            dp.get("rescue_enabled") is False, "No actual DP-only summary")
    require(software_fixture or dp.get("physical_qualified") is True,
            "Physical figures require the DP producer's qualified runtime/process-parity authority")
    groups = {}
    for cid, case in dp["cases"].items():
        group = case.get("group", "unspecified")
        require(re.fullmatch(r"[A-Za-z0-9_-]+", group) is not None, "Unsafe figure group name")
        groups.setdefault(group, {})[cid] = case
    return {group: plot_stability(cases, dp["thresholds"], path(destination) / (group + ".svg"),
                software_fixture=software_fixture) for group, cases in groups.items()}


def build_snapshot(timing=None, stability=None, expected=None, inventory=None):
    inputs = Inputs()
    timed = component(timing, "ninjars_paper_timing_v3", "ninjars-paper-timing-summary-v1",
                      expected["runtime_identity_sha256"] if expected else None, inputs) if timing else None
    dp = component(stability, "ninjars_dp_stability", "ninjars-dp-stability-summary-v1",
                   {"cohort": expected} if expected else None, inputs) if stability else None
    if dp is not None:
        require(dp.get("stream") == "forced_dp_self_consistency" and dp.get("rescue_enabled") is False,
                "DP-only figures require the actual unrescued DP stream")
    inventories = {}
    if inventory is not None:
        definitions = json.loads(retain(inventory, inputs))
        for cid, definition in definitions.items():
            require(isinstance(cid, str) and isinstance(definition, dict), "Invalid process-inventory selection")
            inventories[cid] = process_inventory(**definition, inputs=inputs)
    return {"schema": SCHEMA, "timing": timed, "stability": dp, "tables": table_rows(timed, inventories),
            "inputs_sha256": dict(sorted(inputs.items())),
            "qualification_eligible": False, "scope": "Paper-analogue diagnostic components; the complete full-helicity/rescue-on acceptance gate remains separate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", type=Path)
    parser.add_argument("--stability", type=Path)
    parser.add_argument("--expected", type=Path)
    parser.add_argument("--inventory", type=Path, help="Case-to-generated-file selection for static process counts")
    parser.add_argument("--figures", type=Path, help="New destination directory for DP-only SVG groups")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = path(args.output)
    require(not target.exists(), "Preserve prior paper report snapshots")
    expected = json.loads(path(args.expected).read_text()) if args.expected else None
    snapshot = build_snapshot(args.timing, args.stability, expected, args.inventory)
    if args.figures:
        require(not path(args.figures).exists(), "Preserve prior figure directories")
        snapshot["figures"] = plot_groups(snapshot, args.figures)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
