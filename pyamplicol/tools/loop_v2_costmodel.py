#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Loop-v2 per-point cost model and the gate-G0 recalibration of DESIGN Table 1a.

The design's recipe (docs/development/loop_v2/DESIGN.md §1.3) predicts the
time per point of every process from schedule-B dressing work ``D`` (complex
multiply-adds per point, structured and dense kernel constants) and the
NinjaRS stage ``R``:

    t = [D * quartic / throughput + R * check] * outer

Gate G0 (§5.4, §14) replaces the assumed kernel throughput by the throughput
measured by ``rust/crates/rusticol-core/benches/loop_v2_kernels.rs``. For each
benchmarked node shape the design-model work of that node is divided by the
time of the fastest kernel form allowed at that end of the band:

- lower end: structured constants (fused 200 X, split vertex 300), any form;
- upper end: dense constants (fused 320 X, split vertex 1,300), dense forms.

Shapes are weighted by the design's dressing-cost profile over input rank
(§5.4) into one harmonic-mean throughput per lane count; the upper end uses
the slowest shape instead. Per-node and per-edge-call dispatch from the same
benchmark is added explicitly per trie node and edge, which replaces the
recipe's 45 % allowance.

Per REQUIREMENTS_ADDENDUM.md §D the speed gate times pyAmpliCol in
finite-only mode with stability checks disabled, so the gate columns use
``check = 1``; the columns with the reduction-direction check use the
design's re-test factor (1.75-2.0). G0 passes when every gated QCD process is
within its gate (1.3 x best production provider) with x1.3 headroom at the
upper end of the band.

``--trie-shapes`` evaluates the same model on real trie node shapes (a JSON
list of ``{"rows_out", "x", "rank_in", "count"}`` nodes) instead of the
design's counted work.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import comb
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProcessRow:
    process: str
    colour: str
    families: int
    skeletons: int
    trie_nodes: int
    dress_structured: float
    dress_dense: float
    ninjars_ms: tuple[float, float]


# Schedule-B counts and dressing work in the providers' configuration
# (loop_sm-no_b_mass) and the NinjaRS stage with cached masters, from the
# design's generator models: workspace/tmp/engine-v2/revision/mb0_all.json
# (counts, D) and revision/ninjars_stage.py (R). DESIGN Table 1a.
PROCESSES: tuple[ProcessRow, ...] = (
    ProcessRow("gg>gg", "FC", 24, 63, 102, 510036, 908340, (0.138, 0.157)),
    ProcessRow("gg>gg", "LC", 12, 21, 38, 187992, 307140, (0.069, 0.078)),
    ProcessRow(
        "gg>gggg", "FC", 1100, 9180, 7619, 133788120, 237521080, (43.51, 51.209)
    ),
    ProcessRow("gg>gggg", "LC", 550, 3060, 2723, 46987260, 79393460, (21.755, 25.604)),
    ProcessRow(
        "gg>ggggg",
        "FC",
        9414,
        122040,
        82738,
        2585755000,
        4593989160,
        (854.041, 1001.659),
    ),
    ProcessRow(
        "gg>ggggg", "LC", 4707, 40680, 29155, 894864640, 1533038320, (427.021, 500.83)
    ),
    ProcessRow(
        "uu~>ggggg", "FC", 5258, 91440, 56861, 857766200, 1550332060, (216.02, 293.678)
    ),
    ProcessRow(
        "uu~>ggggg", "LC", 2712, 13560, 15526, 156409740, 262336700, (91.466, 123.932)
    ),
    ProcessRow(
        "uu~>e-e+ggg", "FC", 463, 3990, 2774, 21045804, 36361136, (9.307, 12.815)
    ),
    ProcessRow("uu~>e-e+ggg", "LC", 164, 306, 666, 2567660, 4054812, (1.47, 2.081)),
    ProcessRow("uu~>e-e+dd~g", "FC", 242, 2484, 1375, 4890412, 8711848, (1.821, 3.272)),
    ProcessRow("uu~>e-e+dd~g", "LC", 114, 144, 419, 940000, 1441400, (0.503, 0.907)),
    ProcessRow("gg>tt~gg", "FC", 1086, 6888, 5659, 49495768, 85320548, (19.18, 26.878)),
    ProcessRow("gg>tt~gg", "LC", 375, 1224, 1702, 13022052, 21675864, (5.217, 7.16)),
    ProcessRow(
        "gg>tt~e-e+g", "FC", 579, 3990, 2840, 23869840, 40824140, (10.182, 14.067)
    ),
    ProcessRow("gg>tt~e-e+g", "LC", 164, 306, 666, 3589988, 5852332, (1.47, 2.081)),
)

# Best production-mode provider per gated QCD process on this host, s/pt
# (DESIGN Table 1b): the speed gate is 1.3 x the lower value.
PROVIDERS: Mapping[str, tuple[str, float, float]] = {
    "gg>gggg": ("OpenLoops", 2.92, 2.96),
    "uu~>e-e+dd~g": ("Recola", 0.0347, 0.0350),
    "uu~>e-e+ggg": ("Recola", 0.246, 0.259),
    "gg>tt~gg": ("OpenLoops", 0.487, 0.492),
    "gg>tt~e-e+g": ("OpenLoops", 0.425, 0.433),
}
GATE_FACTOR = 1.3
G0_HEADROOM = 1.3

# Recipe factors (DESIGN §1.3) that G0 does not measure: quartic segments and
# ghost loops on top of D; trees, seeds, counterterms and checks on top of all.
QUARTIC = (1.20, 1.35)
OUTER = (1.05, 1.10)
DESIGN_THROUGHPUT = (1.8e9, 1.0e9)
RETEST = {"none": (1.0, 1.0), "reduction-direction": (1.75, 2.0)}

# Share of the 7-leg dressing cost per input rank 0..6 (DESIGN §5.4).
RANK_PROFILE = (0.10, 0.19, 0.19, 0.19, 0.18, 0.11, 0.03)

STRUCTURED = (200, 300)
DENSE = (320, 1300)
DENSE_FORMS = frozenset({"split-dense", "fused-dense", "fused-blocked"})


def node_work(
    rows_out: int, rank_in: int, x: int, constants: tuple[int, int], seeded: bool = True
) -> int:
    """Complex MACs per lane of one trie node in the design's cost model.

    ``x`` is the number of attached block rows summed over the node's edges.
    A ``seeded`` rank-0 node reads leaf seeds proportional to
    delta(alpha, beta) (fused 80 X, split 4 X + vertex); otherwise the input
    is a full object, as in the kernel benchmark.
    """
    fused, vertex = constants
    if rank_in == 0 and seeded:
        return rows_out * min(80 * x, 4 * x + vertex)
    return rows_out * comb(rank_in + 4, 4) * min(fused * x, 64 * x + vertex)


@dataclass(frozen=True, slots=True)
class Calibration:
    """Measured kernel model at one lane count."""

    lanes: int
    throughput: tuple[float, float]  # design CMAC/s at the (lower, upper) end
    per_shape: tuple[dict[str, object], ...]
    per_node_seconds: float
    per_edge_call_seconds: float


def _gluon_nodes(records: Iterable[Mapping[str, object]], lanes: int):
    shapes: dict[tuple[int, int, int], dict[str, float]] = {}
    for record in records:
        if record["kind"] != "node" or record["lanes"] != lanes:
            continue
        shape = record["shape"]
        assert isinstance(shape, Mapping)
        if shape["line"] != "gluon":
            continue
        key = (int(shape["rows_out"]), int(shape["x"]), int(shape["rank_in"]))
        shapes.setdefault(key, {})[str(record["form"])] = float(
            record["seconds_per_node"]  # type: ignore[arg-type]
        )
    return shapes


def calibrate(report: Mapping[str, object], lanes: int) -> Calibration:
    records = report["records"]
    assert isinstance(records, list)
    shapes = _gluon_nodes(records, lanes)
    if not shapes:
        raise ValueError(f"no gluon node records at {lanes} lanes")
    per_shape = []
    by_rank: dict[int, tuple[float, float]] = {}
    for (rows_out, x, rank_in), times in sorted(shapes.items(), key=lambda i: i[0][2]):
        best_form = min(times, key=times.__getitem__)
        best_dense = min((f for f in times if f in DENSE_FORMS), key=times.__getitem__)
        low = (
            node_work(rows_out, rank_in, x, STRUCTURED, False)
            * lanes
            / times[best_form]
        )
        high = node_work(rows_out, rank_in, x, DENSE, False) * lanes / times[best_dense]
        by_rank[rank_in] = (low, high)
        per_shape.append(
            {
                "rows_out": rows_out,
                "x": x,
                "rank_in": rank_in,
                "best_form": best_form,
                "best_dense_form": best_dense,
                "structured_design_cmac_per_s": low,
                "dense_design_cmac_per_s": high,
                "seconds": dict(sorted(times.items())),
            }
        )
    weights = {
        rank: share for rank, share in enumerate(RANK_PROFILE) if rank in by_rank
    }
    total = sum(weights.values())
    low = total / sum(share / by_rank[rank][0] for rank, share in weights.items())
    high = min(value[1] for value in by_rank.values())
    dispatch = [r for r in records if r["kind"] == "dispatch" and r["lanes"] == lanes]
    per_node = max(float(r["per_node_seconds"]) for r in dispatch)
    per_edge = max(float(r["per_edge_call_seconds"]) for r in dispatch)
    return Calibration(lanes, (low, high), tuple(per_shape), per_node, per_edge)


def lane_counts(report: Mapping[str, object]) -> list[int]:
    records = report["records"]
    assert isinstance(records, list)
    return sorted({int(r["lanes"]) for r in records if r["kind"] == "node"})


def recommended_lanes(calibrations: Sequence[Calibration]) -> Calibration:
    """Highest lower-end throughput; a smaller lane count within 5 % wins."""
    best = max(c.throughput[0] for c in calibrations)
    return min(
        (c for c in calibrations if c.throughput[0] >= 0.95 * best),
        key=lambda c: c.lanes,
    )


def dispatch_seconds(calibration: Calibration, nodes: int, edges: int) -> float:
    """Per-point dispatch: every node and edge call is shared by the lanes."""
    return (
        nodes * calibration.per_node_seconds + edges * calibration.per_edge_call_seconds
    ) / calibration.lanes


def predict(
    row: ProcessRow, calibration: Calibration | None, check: str
) -> tuple[float, float]:
    """(lower, upper) seconds per point; ``calibration=None`` is the design recipe."""
    retest = RETEST[check]
    r_lo, r_hi = (value / 1e3 for value in row.ninjars_ms)
    if calibration is None:
        throughput = DESIGN_THROUGHPUT
        overhead = 0.0
    else:
        throughput = calibration.throughput
        # A trie has one parent edge per node plus one seed edge per skeleton.
        overhead = dispatch_seconds(
            calibration, row.trie_nodes, row.trie_nodes + row.skeletons
        )
    low = (
        row.dress_structured * QUARTIC[0] / throughput[0] + overhead + r_lo * retest[0]
    ) * OUTER[0]
    high = (
        row.dress_dense * QUARTIC[1] / throughput[1] + overhead + r_hi * retest[1]
    ) * OUTER[1]
    return low, high


def gate(process: str) -> float | None:
    provider = PROVIDERS.get(process)
    return None if provider is None else GATE_FACTOR * provider[1]


def g0_rows(calibration: Calibration) -> list[dict[str, object]]:
    rows = []
    for row in PROCESSES:
        limit = gate(row.process)
        if limit is None or row.colour != "FC":
            continue
        low, high = predict(row, calibration, "none")
        rows.append(
            {
                "process": row.process,
                "target_seconds": (low, high),
                "gate_seconds": limit,
                "headroom": limit / high,
                "pass": limit / high >= G0_HEADROOM,
            }
        )
    return rows


def trie_shape_work(nodes: Iterable[Mapping[str, int]]) -> tuple[int, int, int]:
    """(structured D, dense D, node count) of real trie node shapes."""
    structured = dense = count = 0
    for node in nodes:
        n = int(node.get("count", 1))
        args = (int(node["rows_out"]), int(node["rank_in"]), int(node["x"]))
        structured += n * node_work(*args, STRUCTURED)
        dense += n * node_work(*args, DENSE)
        count += n
    return structured, dense, count


def _band(values: tuple[float, float]) -> str:
    return f"{values[0]:.3g}-{values[1]:.3g}"


def report_text(
    label: str, features: object, calibrations: Sequence[Calibration]
) -> tuple[str, bool]:
    chosen = recommended_lanes(calibrations)
    lines = [f"== {label}: target features {features}"]
    lines.append(
        "lanes  design-CMAC/s lower (rank-weighted)  upper (slowest shape)  "
        "per node  per edge call"
    )
    for c in calibrations:
        lines.append(
            f"{c.lanes:5d}  {c.throughput[0] / 1e9:30.2f} G"
            f"  {c.throughput[1] / 1e9:19.2f} G"
            f"  {c.per_node_seconds * 1e9:7.0f} ns"
            f"  {c.per_edge_call_seconds * 1e9:10.1f} ns"
        )
    lines.append(f"recommended lanes: {chosen.lanes}")
    lines.append("fastest form per node shape at the recommended lane count:")
    for shape in chosen.per_shape:
        lines.append(
            f"  rows={shape['rows_out']:3d} X={shape['x']:3d} r_in={shape['rank_in']}:"
            f" {shape['best_form']:16s} (dense: {shape['best_dense_form']})"
        )
    lines.append("")
    lines.append(
        f"{'process':14s} col  design recipe s/pt     recalibrated, no checks   "
        "with reduction-direction check"
    )
    for row in PROCESSES:
        design = predict(row, None, "reduction-direction")
        gate_mode = predict(row, chosen, "none")
        checked = predict(row, chosen, "reduction-direction")
        lines.append(
            f"{row.process:14s} {row.colour}  {_band(design):22s}"
            f" {_band(gate_mode):25s} {_band(checked)}"
        )
    lines.append("")
    passed = True
    for row in g0_rows(chosen):
        verdict = "pass" if row["pass"] else "FAIL"
        passed &= bool(row["pass"])
        lines.append(
            f"G0 {row['process']:14s} target {_band(row['target_seconds'])} s"
            f"  gate {row['gate_seconds']:.3g} s"
            f"  headroom x{row['headroom']:.2f}  {verdict}"
        )
    lines.append(
        f"G0 {'PASS' if passed else 'FAIL'}"
        f" (headroom >= x{G0_HEADROOM} at the upper end)"
    )
    return "\n".join(lines), passed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "benchmarks",
        nargs="*",
        type=Path,
        help="JSON reports of benches/loop_v2_kernels.rs (--json)",
    )
    parser.add_argument("--json", type=Path, help="write the recalibration as JSON")
    parser.add_argument(
        "--trie-shapes",
        type=Path,
        help="JSON node shapes of a real trie: predicts its dressing time",
    )
    args = parser.parse_args(argv)
    if not args.benchmarks:
        print(f"{'process':14s} col  design recipe s/pt (with re-test)")
        for row in PROCESSES:
            band = _band(predict(row, None, "reduction-direction"))
            print(f"{row.process:14s} {row.colour}  {band}")
        return 0
    status = 0
    summary = []
    for path in args.benchmarks:
        report = json.loads(path.read_text())
        features = report.get("target_features")
        calibrations = [calibrate(report, lanes) for lanes in lane_counts(report)]
        chosen = recommended_lanes(calibrations)
        text, passed = report_text(str(path), features, calibrations)
        print(text)
        status |= 0 if passed else 1
        entry: dict[str, object] = {
            "benchmark": str(path),
            "target_features": features,
            "calibrations": [
                {
                    "lanes": c.lanes,
                    "throughput_design_cmac_per_s": c.throughput,
                    "per_node_seconds": c.per_node_seconds,
                    "per_edge_call_seconds": c.per_edge_call_seconds,
                    "shapes": c.per_shape,
                }
                for c in calibrations
            ],
            "recommended_lanes": chosen.lanes,
            "targets": [
                {
                    "process": row.process,
                    "colour": row.colour,
                    "design_recipe": predict(row, None, "reduction-direction"),
                    "no_checks": predict(row, chosen, "none"),
                    "reduction_direction_check": predict(
                        row, chosen, "reduction-direction"
                    ),
                }
                for row in PROCESSES
            ],
            "g0": g0_rows(chosen),
        }
        if args.trie_shapes is not None:
            structured, dense, count = trie_shape_work(
                json.loads(args.trie_shapes.read_text())
            )
            seconds = (
                structured / chosen.throughput[0],
                dense / chosen.throughput[1],
            )
            print(
                f"trie shapes {args.trie_shapes}: {count} nodes, D = {structured:.3g}"
                f"-{dense:.3g} CMAC/pt, dressing {_band(seconds)} s/pt"
                f" at {chosen.lanes} lanes"
            )
            entry["trie_shapes"] = {
                "nodes": count,
                "dress": (structured, dense),
                "seconds": seconds,
            }
        summary.append(entry)
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
