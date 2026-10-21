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

The probe shapes are weighted by the design's dressing-cost profile over
input rank (§5.4) into one harmonic-mean throughput per lane count; the upper
end uses the slowest shape instead. The benchmark's per-node and
per-edge-call dispatch is added per trie node and edge, but it times empty
calls and is only a lower bound: real-trie gathers and cache misses are not
measured until the real-trie half of G0 and G1. The upper end therefore keeps
the recipe's allowance (45 % of the throughput, ``--allowance``) for them.

All reports are evaluated at one lane count, the one the executor will use:
``--lanes``, or by default the count that holds up across every report
(``robust_lanes``; with a single report its fastest count).

Per REQUIREMENTS_ADDENDUM.md §D the speed gate times pyAmpliCol in
finite-only mode with stability checks disabled, so the gate columns use
``check = 1``; the columns with the reduction-direction check use the
design's re-test factor (1.75-2.0). G0 passes when every gated QCD process is
within its gate (1.3 x best production provider) with x1.3 headroom at the
upper end of the band.

``--trie-shapes`` evaluates the same model on real trie node shapes (a JSON
list of ``{"rows_out", "x", "rank_in", "count"}`` nodes, or the records of
``pyamplicol.loops.v2.schedules.trie_shapes``, whose ``x_rows`` is ``x``)
instead of the design's counted work.

``--export-programs DIR`` writes the schedule-B programs that the Python
reference executor runs for the G0 processes (``PROGRAM_PROCESSES``) for the
benchmark's real-trie runs.

The fastest kernel form is reported for every benchmarked node shape,
including the ``small-x`` set (at most two attached block rows) that does not
enter the calibration. Leaf-seeded rank-0 nodes have no kernel yet, so their
work (``node_work(seeded=True)``) is a model, not a measurement.
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
# Share of the upper-end throughput kept back for dispatch, real-trie gathers
# and cache misses (DESIGN §1.3, §5.4: 1.8 G probe -> 1.0 G recipe).
ALLOWANCE = 0.45
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


def _node_times(
    records: Iterable[Mapping[str, object]],
    lanes: int,
    lines: frozenset[str] = frozenset({"gluon"}),
    sets: frozenset[str] = frozenset({"probe"}),
) -> dict[tuple[str, str, int, int, int], dict[str, float]]:
    """Seconds per node and form, keyed by (set, line, rows_out, x, rank_in)."""
    shapes: dict[tuple[str, str, int, int, int], dict[str, float]] = {}
    for record in records:
        if record["kind"] != "node" or record["lanes"] != lanes:
            continue
        shape = record["shape"]
        assert isinstance(shape, Mapping)
        shape_set = str(shape.get("set", "probe"))
        if shape["line"] not in lines or shape_set not in sets:
            continue
        key = (
            shape_set,
            str(shape["line"]),
            int(shape["rows_out"]),
            int(shape["x"]),
            int(shape["rank_in"]),
        )
        shapes.setdefault(key, {})[str(record["form"])] = float(
            record["seconds_per_node"]  # type: ignore[arg-type]
        )
    return shapes


def fastest_forms(report: Mapping[str, object], lanes: int) -> list[dict[str, object]]:
    """Fastest kernel form of every benchmarked node shape (all forms compute
    the same node, so their times compare directly)."""
    records = report["records"]
    assert isinstance(records, list)
    shapes = _node_times(
        records,
        lanes,
        frozenset({"gluon", "ghost"}),
        frozenset({"probe", "small-x"}),
    )
    rows = []
    for (shape_set, line, rows_out, x, rank_in), times in sorted(shapes.items()):
        best = min(times, key=times.__getitem__)
        rows.append(
            {
                "set": shape_set,
                "line": line,
                "rows_out": rows_out,
                "x": x,
                "rank_in": rank_in,
                "best_form": best,
                "slowdown": {
                    form: seconds / times[best]
                    for form, seconds in sorted(times.items())
                },
            }
        )
    return rows


def poly_layout_ratio(report: Mapping[str, object], lanes: int) -> float | None:
    """Median rate of ``poly::apply_segment`` (Complex lanes) over the
    lane-block structured fused kernel on the same gluon probe shapes."""
    records = report["records"]
    assert isinstance(records, list)
    fused = {
        _shape_key(r["shape"]): float(r["gcmac_per_s"])
        for r in records
        if r["kind"] == "node"
        and r["lanes"] == lanes
        and r["form"] == "fused-structured"
    }
    ratios = sorted(
        float(r["gcmac_per_s"]) / fused[_shape_key(r["shape"])]
        for r in records
        if r["kind"] == "poly-layout"
        and r["lanes"] == lanes
        and _shape_key(r["shape"]) in fused
    )
    return ratios[len(ratios) // 2] if ratios else None


def _shape_key(shape: object) -> tuple[object, ...]:
    assert isinstance(shape, Mapping)
    return (
        shape.get("set", "probe"),
        shape["line"],
        shape["rows_out"],
        shape["x"],
        shape["rank_in"],
    )


def calibrate(report: Mapping[str, object], lanes: int) -> Calibration:
    records = report["records"]
    assert isinstance(records, list)
    shapes = _node_times(records, lanes)
    if not shapes:
        raise ValueError(f"no gluon probe node records at {lanes} lanes")
    per_shape = []
    by_rank: dict[int, tuple[float, float]] = {}
    for (_, _, rows_out, x, rank_in), times in sorted(
        shapes.items(), key=lambda i: i[0][4]
    ):
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


def gate_lanes(reports: Sequence[Sequence[Calibration]]) -> int:
    """Default lane count of the gate: robust across reports, else the fastest."""
    if len(reports) == 1:
        return recommended_lanes(reports[0]).lanes
    return robust_lanes(reports)


def robust_lanes(reports: Sequence[Sequence[Calibration]]) -> int:
    """Lane count that stays closest to each report's best lower-end throughput.

    With reports from several code generations (baseline, AVX2, AVX-512) this
    is the lane count whose worst ratio to the best of its report is highest.
    """
    common = set.intersection(*({c.lanes for c in report} for report in reports))

    def worst(lanes: int) -> float:
        return min(
            next(c for c in report if c.lanes == lanes).throughput[0]
            / max(c.throughput[0] for c in report)
            for report in reports
        )

    return max(sorted(common), key=worst)


def dispatch_seconds(calibration: Calibration, nodes: int, edges: int) -> float:
    """Per-point dispatch: every node and edge call is shared by the lanes."""
    return (
        nodes * calibration.per_node_seconds + edges * calibration.per_edge_call_seconds
    ) / calibration.lanes


def predict(
    row: ProcessRow,
    calibration: Calibration | None,
    check: str,
    allowance: float = ALLOWANCE,
) -> tuple[float, float]:
    """(lower, upper) seconds per point; ``calibration=None`` is the design recipe.

    The upper end runs at ``1 - allowance`` of the measured throughput.
    """
    retest = RETEST[check]
    r_lo, r_hi = (value / 1e3 for value in row.ninjars_ms)
    if calibration is None:
        throughput = DESIGN_THROUGHPUT
        overhead = 0.0
    else:
        throughput = (
            calibration.throughput[0],
            calibration.throughput[1] * (1.0 - allowance),
        )
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


def g0_rows(
    calibration: Calibration, allowance: float = ALLOWANCE
) -> list[dict[str, object]]:
    rows = []
    for row in PROCESSES:
        limit = gate(row.process)
        if limit is None or row.colour != "FC":
            continue
        low, high = predict(row, calibration, "none", allowance)
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
        x = node["x"] if "x" in node else node["x_rows"]
        args = (int(node["rows_out"]), int(node["rank_in"]), int(x))
        structured += n * node_work(*args, STRUCTURED)
        dense += n * node_work(*args, DENSE)
        count += n
    return structured, dense, count


# ----------------------------------------------------------- real trie programs

# Processes whose real schedule-B programs gate G0 runs (DESIGN §5.4, WP0.3):
# the loop_sm-no_b_mass configuration of tools/loop_v2_operation_counts.py.
PROGRAM_PROCESSES: tuple[dict[str, object], ...] = (
    {
        "process": "gg>gggg",
        "incoming": ["G", "G"],
        "outgoing": ["G", "G", "G", "G"],
        "born_orders": {"QCD": 4},
        "colour": ("LC", "FC"),
    },
    {
        "process": "uu~>e-e+dd~g",
        "incoming": ["u", "u~"],
        "outgoing": ["e-", "e+", "d", "d~", "G"],
        "born_orders": {"QCD": 3, "QED": 2},
        "colour": ("FC",),
    },
    {
        "process": "uu~>e-e+ggg",
        "incoming": ["u", "u~"],
        "outgoing": ["e-", "e+", "G", "G", "G"],
        "born_orders": {"QCD": 3, "QED": 2},
        "colour": ("FC",),
    },
    {
        "process": "gg>tt~gg",
        "incoming": ["G", "G"],
        "outgoing": ["t", "t~", "G", "G"],
        "born_orders": {"QCD": 4},
        "colour": ("FC",),
    },
    {
        "process": "gg>tt~e-e+g",
        "incoming": ["G", "G"],
        "outgoing": ["t", "t~", "e-", "e+", "G"],
        "born_orders": {"QCD": 3, "QED": 2},
        "colour": ("FC",),
    },
)


def _operation_counts_tool():
    """``tools/loop_v2_operation_counts.py``: the WP1.5 configuration."""

    import importlib.util

    path = Path(__file__).resolve().with_name("loop_v2_operation_counts.py")
    spec = importlib.util.spec_from_file_location("_loop_v2_operation_counts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def program_plan(
    prepared: Mapping[str, object], entry: Mapping[str, object], colour: str
):
    """(plan, kinematics, seeds) of one process in the configuration of
    ``tools/loop_v2_operation_counts.py``: catalogue kernels, synthetic block
    chiralities, cost-optimal frames, inclusive seeds (one column); LC keeps
    the skeletons without a closed quark loop (all-gluon face rule)."""

    from pyamplicol.loops.v2 import reference_executor as rx
    from pyamplicol.loops.v2.families import (
        LoopOrderSelection,
        ModelBlockTable,
        generate_families,
    )
    from pyamplicol.loops.v2.schedules import SkeletonView, build_schedules, cheapest
    from pyamplicol.loops.v2.skeletons import PlanarWords

    data, catalogue = prepared["data"], prepared["catalogue"]
    born = dict(entry["born_orders"])  # type: ignore[arg-type]
    loop = {**born, "QCD": born.get("QCD", 0) + 2}
    selection = LoopOrderSelection(
        frozenset({"QCD"}), frozenset({tuple(sorted(loop.items()))})
    )
    table = ModelBlockTable(
        data, entry["incoming"], entry["outgoing"], cap=selection.cap
    )
    families = generate_families(table, catalogue, selection)
    legs = families.legs
    singlets = sum(1 << leg.index for leg in legs if abs(leg.colour) == 1)
    letters = [1 << leg.index for leg in legs if abs(leg.colour) != 1]
    words = PlanarWords(len(legs), letters + ([singlets] if singlets else []))
    particles = {particle["name"]: particle for particle in data["particles"]}

    def helicities(leg) -> tuple[int, ...]:
        particle = particles[leg.field]
        conserved = particle["spin"] == 2 and leg.mass.massless
        return (1,) if conserved and particle["pdg_code"] > 0 else (-1, 1)

    blocks = rx.SyntheticBlockRows(
        [helicities(leg) for leg in legs],
        chiralities=rx.block_chiralities(prepared["kernels"]),
    )
    kinematics = rx.CatalogueKinematics(
        prepared["kernels"],
        prepared["compiled"].ir.vertex_terms,
        data,
        blocks,
        (),
        rx.KernelConstants({}, rx.Carrier.exact()),
    )
    seeds = rx.SyntheticSeeds(blocks.row_set((1 << len(legs)) - 1), values=False)

    def no_quark_loop(skeleton) -> bool:
        return not any(
            abs(catalogue.species(name).colour) == 3 for name in skeleton.species
        )

    view = SkeletonView(
        catalogue, words, keep=no_quark_loop if colour == "LC" else None
    )
    plan = build_schedules(
        families,
        catalogue,
        words,
        policy=cheapest(lambda mask: len(blocks.row_set(mask))),
        view=view,
    )
    return plan, kinematics, seeds


@dataclass(frozen=True, slots=True)
class _Object:
    """An open-loop object of the program: a seed or the output of a step."""

    seed: int  # seed id, or -1
    step: int  # step index, or -1
    rows: tuple[object, ...]
    d_alpha: int
    d_beta: int
    rank: int  # the reference executor's structural rank
    storage_rank: int  # the rank the object is stored at (>= rank)

    @property
    def diagonal(self) -> bool:
        return self.seed >= 0


class _ProgramBuilder:
    """Walks schedule B exactly as ``ReferenceExecutor.run_b`` and records the
    program: steps (one per dressed trie node, one per cut species of a family
    accumulator), their edges with row links, and the kernel structures."""

    def __init__(self, plan, kinematics, seeds):
        self.plan = plan
        self.kinematics = kinematics
        self.seeds = seeds
        self.contents: dict[object, int] = {}
        self.content_records: list[dict[str, object]] = []
        self.groups: dict[object, int] = {}
        self.group_records: list[dict[str, object]] = []
        self.currents: dict[tuple[int, int], int] = {}
        self.factors: dict[tuple[int, int], int] = {}
        self.factor_records: list[list[object]] = []
        self.vertices: dict[tuple[int, int], int] = {}
        self.seed_ids: dict[object, int] = {}
        self.seed_rows: list[int] = []
        self.steps: list[dict[str, object]] = []
        self.consumed: set[int] = set()
        self.work: dict[str, int] = {}

    def _count(self, kind: str, value: int) -> None:
        self.work[kind] = self.work.get(kind, 0) + value

    def _content(self, content) -> tuple[int, object]:
        shape = self.kinematics.shape(content)
        found = self.contents.get(content)
        if found is not None:
            return found, shape
        index = self.contents[content] = len(self.content_records)
        split = []
        for part in shape.split:
            group = self.groups.get(part.key)
            if group is None:
                group = self.groups[part.key] = len(self.group_records)
                self.group_records.append(
                    {
                        "width": part.width,
                        "d_in": shape.loop_in,
                        "d_out": shape.loop_out,
                        "entries": [[b, nu, b2, *e] for b, nu, b2, e in part.entries],
                    }
                )
            split.append([group, [shape.rows.index(row) for row in part.rows]])
        self.content_records.append(
            {
                "d_in": shape.loop_in,
                "d_out": shape.loop_out,
                "degree": shape.degree,
                "rows": len(shape.rows),
                "fused": [[b, b2, *e] for b, b2, e in shape.fused],
                "split": split,
            }
        )
        return index, shape

    def _id(self, table: dict, key) -> int:
        return table.setdefault(key, len(table))

    def _factor(self, content: int, shape, shift: int) -> int:
        """X(J) of a content at a frame position: its block currents times the
        vertex tensors of its split parts at that position."""

        found = self.factors.get((content, shift))
        if found is None:
            vertices = [
                self._id(self.vertices, (self.groups[part.key], shift))
                for part in shape.split
            ]
            found = self.factors[(content, shift)] = len(self.factor_records)
            self.factor_records.append([content, vertices])
        return found

    def run(self) -> None:
        for family, (key, schedule) in enumerate(self.plan.schedules.items()):
            self._family(family, key, schedule.trie)
        steps = {i for i, step in enumerate(self.steps) if not step["trace"]}
        if steps != self.consumed:
            raise AssertionError("every open-loop object feeds exactly one edge")

    def _family(self, family: int, key, trie) -> None:
        before = [0]
        for mask, _mass in trie.frame.apply(key):
            before.append(before[-1] | mask)
        full = before[-1]
        leaves = []
        for skeleton, ring in trie.leaves:
            contents = [self.plan.view.content(segment.label) for segment in ring]
            seed_key, rows = self.seeds.seed(skeleton, contents)
            seed = self.seed_ids.get(seed_key)
            if seed is None:
                seed = self.seed_ids[seed_key] = len(self.seed_rows)
                self.seed_rows.append(len(rows))
            dimension = self.kinematics.shape(contents[0]).loop_in
            leaves.append(_Object(seed, -1, tuple(rows), dimension, dimension, 0, 0))
        values: dict[int, _Object] = {}
        for index, node in enumerate(trie.nodes):
            edges = [
                (
                    self.plan.view.content(segment.label),
                    leaves[child] if node.depth == 1 else values.pop(child),
                )
                for segment, child in node.edges
            ]
            step = {
                "family": family,
                "depth": node.depth,
                "shift": before[node.depth - 1],
                "rank_in": node.rank_in,
            }
            if node.depth < len(trie.family):
                values[index] = self._dress(
                    edges, out_mask=full ^ before[node.depth], trace=False, **step
                )
                continue
            # Accumulator: each cut species closes apart (reference executor).
            cuts: dict[str, list] = {}
            for content, source in edges:
                cuts.setdefault(content.loop_out, []).append((content, source))
            for species in sorted(cuts):
                self._dress(cuts[species], out_mask=0, trace=True, **step)

    def _dress(self, edges, *, out_mask, trace, family, depth, shift, rank_in):
        from pyamplicol.loops.v2.reference_executor import restrict

        shapes = [self._content(content)[1] for content, _source in edges]
        (d_alpha,) = {source.d_alpha for _content, source in edges}
        (d_out,) = {shape.loop_out for shape in shapes}
        (diagonal,) = {source.diagonal for _content, source in edges}
        degree = max(shape.degree for shape in shapes)
        storage_in = max(
            [rank_in] + [source.storage_rank for _content, source in edges]
        )
        c_in = comb(rank_in + 4, 4)
        per_entry = 1 if diagonal else d_alpha * c_in
        work = {"fused": 0, "split-accumulate": 0, "split-vertex": 0}
        targets: dict[object, int] = {}
        groups: dict[object, dict[str, object]] = {}
        records = []
        for (content, source), shape in zip(edges, shapes, strict=True):
            if shape.loop_in != source.d_beta:
                raise AssertionError("a segment does not fit its open-loop current")
            index, _shape = self._content(content)
            block_index = {row: i for i, row in enumerate(shape.rows)}
            pairs = []
            for position, row in enumerate(source.rows):
                block = block_index.get(restrict(row, content.mask))
                if block is not None:
                    target = targets.setdefault(restrict(row, out_mask), len(targets))
                    pairs.append((target, position, block))
            work["fused"] += len(pairs) * len(shape.fused) * per_entry
            width_scale = 1 if diagonal else d_alpha * source.d_beta * c_in
            split = []
            for k, part in enumerate(shape.split):
                rows = {block_index[row] for row in part.rows}
                links = [pair for pair in pairs if pair[2] in rows]
                group = groups.get(part.key)
                if group is None:
                    number = self.groups[part.key]
                    group = groups[part.key] = {
                        "slot": len(groups),
                        "group": number,
                        "vertex": self._id(self.vertices, (number, shift)),
                        "targets": set(),
                    }
                group["targets"].update(t for t, _s, _b in links)  # type: ignore[attr-defined]
                work["split-accumulate"] += len(links) * part.width * width_scale
                split.append(
                    {
                        "current": self._id(self.currents, (index, k)),
                        "group": group["slot"],
                        "links": [i for link in links for i in link],
                    }
                )
            if not source.diagonal:
                self.consumed.add(source.step)
            records.append(
                {
                    "seed": source.seed,
                    "child": source.step,
                    "content": index,
                    "factor": self._factor(index, shape, shift),
                    "links": [i for pair in pairs for i in pair],
                    "split": split,
                }
            )
        step_groups = []
        for group in groups.values():
            targets_of = sorted(group["targets"])  # type: ignore[call-overload]
            entries = len(self.group_records[group["group"]]["entries"])  # type: ignore[index, arg-type]
            work["split-vertex"] += entries * per_entry * len(targets_of)
            step_groups.append(
                {
                    "group": group["group"],
                    "vertex": group["vertex"],
                    "targets": targets_of,
                }
            )
        kind = "seed" if diagonal else "step"
        split_work = work["split-accumulate"] + work["split-vertex"]
        for form, value in work.items():
            self._count(f"{form}/{kind}", value)
        self._count(f"best/{kind}", min(work["fused"], split_work))
        self._count("dress-steps", 1)
        self._count("edges", len(edges))
        if trace:
            if set(targets) - {()}:
                raise AssertionError("a closure with undressed legs")
            self._count(
                "closure", d_alpha * comb(rank_in + degree + 4, 4) * len(targets)
            )
        self.steps.append(
            {
                "family": family,
                "depth": depth,
                "seeded": diagonal,
                "trace": trace,
                "rows": len(targets),
                "d_alpha": d_alpha,
                "d_out": d_out,
                "degree": degree,
                "rank_in": rank_in,
                "storage_rank_in": storage_in,
                "work": {"fused": work["fused"], "split": split_work},
                "edges": records,
                "groups": step_groups,
            }
        )
        return _Object(
            -1,
            len(self.steps) - 1,
            tuple(targets),
            d_alpha,
            d_out,
            rank_in + degree,
            storage_in + degree,
        )

    def program(self, process: str, colour: str) -> dict[str, object]:
        content_of = {value: key for key, value in self.currents.items()}
        return {
            "process": process,
            "colour": colour,
            "families": len(self.plan.schedules),
            "skeletons": sum(len(s.trie.leaves) for s in self.plan.schedules.values()),
            "trie_nodes": sum(len(s.trie.nodes) for s in self.plan.schedules.values()),
            "work": dict(sorted(self.work.items())),
            "contents": self.content_records,
            "groups": self.group_records,
            "currents": [list(content_of[i]) for i in range(len(content_of))],
            "factors": self.factor_records,
            "vertices": [group for group, _shift in self.vertices],
            "seeds": self.seed_rows,
            "steps": self.steps,
        }


def trie_program(plan, kinematics, seeds, process: str, colour: str):
    """The schedule-B program the reference executor runs, for the kernel
    benchmark: per step its edges (child step or seed, content, row links per
    kernel form) and the per-form work the reference executor counts."""

    builder = _ProgramBuilder(plan, kinematics, seeds)
    builder.run()
    return builder.program(process, colour)


def program_file_name(process: str, colour: str) -> str:
    """``gg>tt~gg``, FC -> ``gg_ttxgg-FC.json``."""

    return f"{process.replace('>', '_').replace('~', 'x')}-{colour}.json"


def export_programs(
    output: Path, ufo: Path, names: Sequence[str] | None = None
) -> list[Path]:
    """Write ``<process>-<colour>.json`` programs of ``PROGRAM_PROCESSES``."""

    import tempfile

    counts = _operation_counts_tool()
    output.mkdir(parents=True, exist_ok=True)
    written = []
    with tempfile.TemporaryDirectory() as scratch:
        prepared = counts.prepare(counts.compile_loop_sm(ufo, Path(scratch)))
        for entry in PROGRAM_PROCESSES:
            if names and entry["process"] not in names:
                continue
            for colour in entry["colour"]:  # type: ignore[union-attr]
                plan, kinematics, seeds = program_plan(prepared, entry, colour)
                program = trie_program(
                    plan, kinematics, seeds, str(entry["process"]), colour
                )
                path = output / program_file_name(str(entry["process"]), colour)
                path.write_text(json.dumps(program, separators=(",", ":")))
                written.append(path)
                print(
                    f"{path}: {program['families']} families,"
                    f" {len(program['steps'])} steps, work {program['work']}"  # type: ignore[arg-type]
                )
    return written


def _band(values: tuple[float, float]) -> str:
    return f"{values[0]:.3g}-{values[1]:.3g}"


def report_text(
    label: str,
    report: Mapping[str, object],
    calibrations: Sequence[Calibration],
    lanes: int,
    allowance: float,
) -> tuple[str, bool]:
    chosen = next(c for c in calibrations if c.lanes == lanes)
    lines = [f"== {label}: target features {report.get('target_features')}"]
    lines.append(
        "lanes  design-CMAC/s lower (rank-weighted)  upper (slowest shape)  "
        "dispatch lower bounds: per node  per edge call"
    )
    for c in calibrations:
        lines.append(
            f"{c.lanes:5d}  {c.throughput[0] / 1e9:30.2f} G"
            f"  {c.throughput[1] / 1e9:19.2f} G"
            f"  {c.per_node_seconds * 1e9:22.0f} ns"
            f"  {c.per_edge_call_seconds * 1e9:10.1f} ns"
        )
    lines.append(
        "fastest lanes for this code generation:"
        f" {recommended_lanes(calibrations).lanes}; gate evaluated at {lanes} lanes"
    )
    lines.append(
        f"fastest form per node shape at {lanes} lanes (slowdown of the others):"
    )
    for shape in fastest_forms(report, lanes):
        others = ", ".join(
            f"{form} x{ratio:.2f}"
            for form, ratio in shape["slowdown"].items()  # type: ignore[union-attr]
            if form != shape["best_form"]
        )
        lines.append(
            f"  {shape['set']:7s} {shape['line']:5s} rows={shape['rows_out']:3d}"
            f" X={shape['x']:3d} r_in={shape['rank_in']}: {shape['best_form']:16s}"
            f" ({others})"
        )
    ratio = poly_layout_ratio(report, lanes)
    if ratio is not None:
        lines.append(
            f"poly.rs apply_segment (Complex lanes) / lane-block fused-structured rate:"
            f" x{ratio:.2f} (median over the gluon probe shapes)"
        )
    lines.append("")
    lines.append(
        f"{'process':14s} col  design recipe s/pt     recalibrated, no checks   "
        "with reduction-direction check"
    )
    for row in PROCESSES:
        design = predict(row, None, "reduction-direction")
        gate_mode = predict(row, chosen, "none", allowance)
        checked = predict(row, chosen, "reduction-direction", allowance)
        lines.append(
            f"{row.process:14s} {row.colour}  {_band(design):22s}"
            f" {_band(gate_mode):25s} {_band(checked)}"
        )
    lines.append("")
    passed = True
    for row in g0_rows(chosen, allowance):
        verdict = "pass" if row["pass"] else "FAIL"
        passed &= bool(row["pass"])
        lines.append(
            f"G0 {row['process']:14s} target {_band(row['target_seconds'])} s"
            f"  gate {row['gate_seconds']:.3g} s"
            f"  headroom x{row['headroom']:.2f}  {verdict}"
        )
    lines.append(
        f"G0 {'PASS' if passed else 'FAIL'} at {lanes} lanes, allowance"
        f" {allowance:.0%} (headroom >= x{G0_HEADROOM} at the upper end)"
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
        "--lanes",
        type=int,
        help="lane count of the gate (default: robust across all reports)",
    )
    parser.add_argument(
        "--allowance",
        type=float,
        default=ALLOWANCE,
        help="share of the upper-end throughput kept for dispatch and gathers",
    )
    parser.add_argument(
        "--trie-shapes",
        type=Path,
        help="JSON node shapes of a real trie: predicts its dressing time",
    )
    parser.add_argument(
        "--export-programs",
        type=Path,
        metavar="DIR",
        help="write the real schedule-B programs of the G0 processes and exit",
    )
    parser.add_argument(
        "--processes",
        help="comma-separated subset of the G0 processes to export",
    )
    parser.add_argument(
        "--ufo",
        type=Path,
        help="loop_sm UFO directory (default: as tools/loop_v2_operation_counts.py)",
    )
    args = parser.parse_args(argv)
    if args.export_programs is not None:
        names = args.processes.split(",") if args.processes else None
        ufo = args.ufo or _operation_counts_tool().UFO
        export_programs(args.export_programs, ufo, names)
        return 0
    if not args.benchmarks:
        print(f"{'process':14s} col  design recipe s/pt (with re-test)")
        for row in PROCESSES:
            band = _band(predict(row, None, "reduction-direction"))
            print(f"{row.process:14s} {row.colour}  {band}")
        return 0
    reports = [json.loads(path.read_text()) for path in args.benchmarks]
    per_report = [
        [calibrate(report, lanes) for lanes in lane_counts(report)]
        for report in reports
    ]
    lanes = args.lanes if args.lanes is not None else gate_lanes(per_report)
    status = 0
    summary = []
    for path, report, calibrations in zip(
        args.benchmarks, reports, per_report, strict=True
    ):
        if lanes not in {c.lanes for c in calibrations}:
            parser.error(f"{path} has no records at {lanes} lanes")
        chosen = next(c for c in calibrations if c.lanes == lanes)
        text, passed = report_text(
            str(path), report, calibrations, lanes, args.allowance
        )
        print(text)
        status |= 0 if passed else 1
        entry: dict[str, object] = {
            "benchmark": str(path),
            "target_features": report.get("target_features"),
            "calibrations": [
                {
                    "lanes": c.lanes,
                    "throughput_design_cmac_per_s": c.throughput,
                    "per_node_seconds_lower_bound": c.per_node_seconds,
                    "per_edge_call_seconds_lower_bound": c.per_edge_call_seconds,
                    "shapes": c.per_shape,
                }
                for c in calibrations
            ],
            "fastest_lanes": recommended_lanes(calibrations).lanes,
            "gate_lanes": lanes,
            "allowance": args.allowance,
            "fastest_forms": fastest_forms(report, lanes),
            "poly_layout_ratio": poly_layout_ratio(report, lanes),
            "targets": [
                {
                    "process": row.process,
                    "colour": row.colour,
                    "design_recipe": predict(row, None, "reduction-direction"),
                    "no_checks": predict(row, chosen, "none", args.allowance),
                    "reduction_direction_check": predict(
                        row, chosen, "reduction-direction", args.allowance
                    ),
                }
                for row in PROCESSES
            ],
            "g0": g0_rows(chosen, args.allowance),
        }
        if args.trie_shapes is not None:
            structured, dense, count = trie_shape_work(
                json.loads(args.trie_shapes.read_text())
            )
            seconds = (
                structured / chosen.throughput[0],
                dense / (chosen.throughput[1] * (1.0 - args.allowance)),
            )
            print(
                f"trie shapes {args.trie_shapes}: {count} nodes, D = {structured:.3g}"
                f"-{dense:.3g} CMAC/pt, dressing {_band(seconds)} s/pt"
                f" at {lanes} lanes"
            )
            entry["trie_shapes"] = {
                "nodes": count,
                "dress": (structured, dense),
                "seconds": seconds,
            }
        summary.append(entry)
    if len(args.benchmarks) > 1:
        print(f"lane count robust across all reports: {robust_lanes(per_report)}")
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    return status


if __name__ == "__main__":
    sys.exit(main())
