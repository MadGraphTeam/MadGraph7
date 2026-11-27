#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Loop-v2 per-point cost model and the gate-G0 recalibration of DESIGN Table 1a.

The design's recipe (docs/development/loop_v2/DESIGN.md §1.3) predicts the
time per point of every process from schedule-B dressing work ``D`` (complex
multiply-adds per point, structured and dense kernel constants) and the
NinjaRS stage ``R``:

    t = [D * quartic / throughput + R * check] * outer

Gate G0 (§5.4, §14) measures the dressing term on real tries:

1. ``--export-programs DIR`` writes, for every process of
   ``PROGRAM_PROCESSES``, the schedule-B program that the Python reference
   executor runs (``ReferenceExecutor.run_b`` on the families, schedules and
   kernel catalogue of ``pyamplicol.loops.v2``, in the configuration of
   ``tools/loop_v2_operation_counts.py``): every dressing step with its
   edges, child objects, catalogue kernel entries and helicity row links.
2. ``rust/crates/rusticol-core/benches/loop_v2_kernels.rs --program FILE``
   runs each program with the Rust kernels in every form, a walk without
   kernels (dispatch), the per-step fastest form (oracle) and a fitted
   per-step form rule, at 8, 16 and 32 lanes; it first checks that every
   form gives the same family numerators.
3. This tool reads the bench reports: each measured program replaces the
   modelled dressing term of its Table 1a row (quartic segments, ghost loops
   and the fused-factor builds are part of the program), unmeasured rows
   scale from their class by Appendix B work, and the NinjaRS stage and outer
   factor are added as in the recipe.

All reports are evaluated at one lane count, the one the executor will use:
``--lanes``, or by default the count whose time stays closest to the fastest
in every program of every report (``program_lanes``). The executor's kernel
form rule (``selection_rules``) is one time model per form, fitted on the
per-step times of ``RULE_CALIBRATION`` and checked against the per-step
oracle of every program.

Per REQUIREMENTS_ADDENDUM.md §D the speed gate times pyAmpliCol in
finite-only mode with stability checks disabled, so the gate columns use
``check = 1``; the columns with the reduction-direction check use the
design's re-test factor (1.75-2.0). G0 passes when every gated QCD process is
within its gate (1.3 x best production provider) with x1.3 headroom at the
upper end of the band.

Gate G1 (WP2.7) measures the whole evaluation end to end
(``tools/loop_benchmark``); ``--g1 G1_JSON`` sets its measured terms (dressing,
NinjaRS stage, the rest of the native loop stage, everything outside it, the
kernel throughput) beside the recipe and G0, which recalibrates the recipe's
NinjaRS and outer factors (``g1_recalibration``).

The bench's synthetic probe shapes are still reported (throughput per lane
count, fastest form per shape, the recipe at the probe's throughput with the
design's allowance for dispatch), but carry no verdict.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import comb, exp, log
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


# ----------------------------------------------------- real-program calibration

# The executor's kernel-form selection: the bench's fitted per-step rule.
GATE_POLICY = "rule"
# A larger lane count must beat the smallest one by more than this.
LANE_TOLERANCE = 0.05
# Table 1a rows without a program scale from the measured row of their class
# by their Appendix B structured work (same colour where measured, else FC).
PROGRAM_CLASS = {
    "gg>gg": "gg>gggg",
    "gg>ggggg": "gg>gggg",
    "uu~>ggggg": "uu~>e-e+ggg",
}


@dataclass(frozen=True, slots=True)
class ProgramTiming:
    """One run of a real program in the bench (one lane count, one policy)."""

    process: str
    colour: str
    lanes: int
    policy: str
    seconds: float  # per point, fused-factor build included
    build_seconds: float  # per point
    reference_best: int  # CMAC/pt counted by the reference executor (best form)

    @property
    def throughput(self) -> float:
        """Reference-counted CMAC/s: the unit of the design's work D."""
        return self.reference_best / self.seconds


def program_timings(report: Mapping[str, object]) -> list[ProgramTiming]:
    records = report["records"]
    assert isinstance(records, list)
    out = []
    for record in records:
        if record["kind"] != "program":
            continue
        work = record["reference_work"]
        lanes = int(record["lanes"])
        out.append(
            ProgramTiming(
                str(record["process"]),
                str(record["colour"]),
                lanes,
                str(record["policy"]),
                float(record["seconds_per_point"]),
                float(record["build_seconds"]) / lanes,
                int(work["best/seed"]) + int(work["best/step"]),
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class ProgramCalibration:
    """Measured dressing time per point of every program at one lane count."""

    lanes: int
    policy: str
    measured: Mapping[tuple[str, str], ProgramTiming]


def program_calibration(
    report: Mapping[str, object], lanes: int, policy: str = GATE_POLICY
) -> ProgramCalibration:
    measured = {
        (t.process, t.colour): t
        for t in program_timings(report)
        if t.lanes == lanes and t.policy == policy
    }
    if not measured:
        raise ValueError(f"no {policy!r} program records at {lanes} lanes")
    return ProgramCalibration(lanes, policy, measured)


def program_lanes(
    reports: Sequence[Mapping[str, object]],
    policy: str = GATE_POLICY,
    tolerance: float = LANE_TOLERANCE,
) -> int:
    """The executor's lane count: the smallest count whose geometric-mean
    time over the programs is within ``tolerance`` of the fastest count in
    every report (code generation). Fewer lanes halve the live objects and
    fused factors per batch, so a larger count must be measurably faster."""
    common: set[int] | None = None
    within: list[set[int]] = []
    for report in reports:
        runs: dict[tuple[str, str], dict[int, float]] = {}
        for t in program_timings(report):
            if t.policy == policy:
                runs.setdefault((t.process, t.colour), {})[t.lanes] = t.seconds
        counts = set.intersection(*(set(by_lanes) for by_lanes in runs.values()))
        common = counts if common is None else common & counts
        mean = {
            lanes: exp(
                sum(log(by_lanes[lanes]) for by_lanes in runs.values()) / len(runs)
            )
            for lanes in counts
        }
        best = min(mean.values()) if mean else 0.0
        within.append(
            {lanes for lanes, t in mean.items() if t <= (1 + tolerance) * best}
        )
    if not common:
        raise ValueError(f"no lane count has {policy!r} records in every report")
    good = set.intersection(*within) & common
    if not good:
        raise ValueError("no lane count is within the tolerance in every report")
    return min(good)


def _process_row(process: str, colour: str) -> ProcessRow | None:
    return next(
        (r for r in PROCESSES if (r.process, r.colour) == (process, colour)), None
    )


def dressing_seconds(
    row: ProcessRow, calibration: ProgramCalibration
) -> tuple[float, float]:
    """(lower, upper) dressing time per point.

    A measured program is its own value. Other rows scale by their Appendix B
    structured work from the measured row of their class (lower end) and from
    the measured row with the most time per unit of that work (upper end).
    """
    measured = calibration.measured.get((row.process, row.colour))
    if measured is not None:
        return measured.seconds, measured.seconds
    per_work = {}
    for key, timing in calibration.measured.items():
        base_row = _process_row(*key)
        if base_row is not None:
            per_work[key] = timing.seconds / base_row.dress_structured
    if not per_work:
        raise ValueError("no measured program has an Appendix B row")
    worst = max(per_work.values())
    base = PROGRAM_CLASS.get(row.process, row.process)
    own = per_work.get((base, row.colour), per_work.get((base, "FC"), worst))
    return own * row.dress_structured, max(own, worst) * row.dress_structured


def predict_measured(
    row: ProcessRow, calibration: ProgramCalibration, check: str
) -> tuple[float, float]:
    """(lower, upper) seconds per point from measured dressing times.

    The measured programs already contain quartic segments, ghost loops and
    fused-factor builds, so only the NinjaRS stage (with ``check``) and the
    outer factor (trees, seeds, counterterms) are added.
    """
    retest = RETEST[check]
    r_lo, r_hi = (value / 1e3 for value in row.ninjars_ms)
    d_lo, d_hi = dressing_seconds(row, calibration)
    return (d_lo + r_lo * retest[0]) * OUTER[0], (d_hi + r_hi * retest[1]) * OUTER[1]


def g0_rows(calibration: ProgramCalibration) -> list[dict[str, object]]:
    """Gated QCD processes: upper end of the finite-only target (no checks)
    against 1.3 x the best production provider, with x1.3 headroom."""
    rows = []
    for row in PROCESSES:
        limit = gate(row.process)
        if limit is None or row.colour != "FC":
            continue
        low, high = predict_measured(row, calibration, "none")
        rows.append(
            {
                "process": row.process,
                "measured": (row.process, row.colour) in calibration.measured,
                "target_seconds": (low, high),
                "gate_seconds": limit,
                "headroom": limit / high,
                "pass": limit / high >= G0_HEADROOM,
            }
        )
    return rows


# Kernel forms the executor chooses from per step (the dense split and fused
# forms are never faster than their structured counterparts).
RULE_FORMS = ("fused-structured", "split-structured", "fused-blocked")
# The program whose per-step times calibrate the executor's selection rule.
RULE_CALIBRATION = ("gg>gggg", "FC")


@dataclass(frozen=True, slots=True)
class StepModel:
    """Time of one step in one kernel form: ``per_step + per_iteration *
    iterations + per_cmac * work * lanes`` (the bench's ``StepModel``)."""

    per_step: float
    per_iteration: float
    per_cmac: float

    def seconds(self, cost: Sequence[int], lanes: int) -> float:
        work, iterations = cost
        return (
            self.per_step
            + self.per_iteration * iterations
            + self.per_cmac * work * lanes
        )


def _least_squares(rows, active: list[int]) -> list[float]:
    n = len(active)
    a = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for x, t in rows:
        for i, ki in enumerate(active):
            for j, kj in enumerate(active):
                a[i][j] += x[ki] * x[kj]
            b[i] += x[ki] * t
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        b[col], b[pivot] = b[pivot], b[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            for k in range(col, n):
                a[row][k] -= factor * a[col][k]
            b[row] -= factor * b[col]
    solution = [0.0] * n
    for row in reversed(range(n)):
        rest = sum(a[row][k] * solution[k] for k in range(row + 1, n))
        solution[row] = (b[row] - rest) / a[row][row]
    out = [0.0, 0.0, 0.0]
    for k, value in zip(active, solution, strict=True):
        out[k] = value
    return out


def fit_step_model(
    costs: Sequence[Sequence[int]], own: Sequence[float], lanes: int
) -> StepModel:
    """Non-negative least squares (a coefficient that would be negative is
    dropped), as the bench fits its self-calibrated rule."""
    rows = [
        ((1.0, float(iterations), float(work * lanes)), t)
        for (work, iterations), t in zip(costs, own, strict=True)
    ]
    active = [0, 1, 2]
    while True:
        x = _least_squares(rows, active)
        negative = [k for k in active if x[k] < 0.0]
        if not negative:
            return StepModel(*x)
        active.remove(negative[0])


def program_steps(
    report: Mapping[str, object], lanes: int
) -> dict[tuple[str, str], Mapping[str, object]]:
    records = report["records"]
    assert isinstance(records, list)
    return {
        (str(r["process"]), str(r["colour"])): r
        for r in records
        if r["kind"] == "program-steps" and r["lanes"] == lanes
    }


def selection_rules(
    report: Mapping[str, object],
    lanes: int,
    calibration: tuple[str, str] = RULE_CALIBRATION,
) -> dict[str, object]:
    """The executor's kernel-form rule and how close it comes to the oracle.

    The rule picks, per step, the form of ``RULE_FORMS`` with the least
    modelled time, with one ``StepModel`` per form fitted on the per-step
    times of the ``calibration`` program. Every program then sums the
    measured own step times of the chosen forms; ``oracle`` is the per-step
    fastest form of all five, ``counted`` the form with less counted work.
    """
    steps = program_steps(report, lanes)
    if calibration not in steps:
        raise ValueError(f"no per-step records of {calibration} at {lanes} lanes")
    forms = steps[calibration]["forms"]
    assert isinstance(forms, Mapping)
    models = {
        form: fit_step_model(forms[form]["cost"], forms[form]["own_seconds"], lanes)
        for form in RULE_FORMS
    }
    programs = []
    for (process, colour), record in sorted(steps.items()):
        forms = record["forms"]
        fields = list(record["fields"])
        fused, split = fields.index("reference_fused"), fields.index("reference_split")
        own = {form: forms[form]["own_seconds"] for form in forms}
        count = len(record["steps"])
        oracle = sum(min(own[f][s] for f in own) for s in range(count))
        rule = sum(
            own[
                min(
                    RULE_FORMS,
                    key=lambda f: models[f].seconds(forms[f]["cost"][s], lanes),
                )
            ][s]
            for s in range(count)
        )
        counted = sum(
            own[
                "split-structured" if step[split] <= step[fused] else "fused-structured"
            ][s]
            for s, step in enumerate(record["steps"])
        )
        single = min(sum(own[f]) for f in own)
        programs.append(
            {
                "process": process,
                "colour": colour,
                "oracle_seconds_per_point": oracle / lanes,
                "rule_over_oracle": rule / oracle,
                "counted_over_oracle": counted / oracle,
                "best_single_form_over_oracle": single / oracle,
            }
        )
    return {
        "lanes": lanes,
        "calibration": list(calibration),
        "models": {
            form: {
                "per_step_seconds": m.per_step,
                "per_iteration_seconds": m.per_iteration,
                "gcmac_per_s": 1.0 / m.per_cmac / 1e9 if m.per_cmac else None,
            }
            for form, m in models.items()
        },
        "programs": programs,
    }


def program_summary(report: Mapping[str, object], lanes: int) -> list[dict]:
    """Per program at ``lanes``: seconds per point of every policy, the walk
    without kernels (dispatch), and the fitted selection rule."""
    records = report["records"]
    assert isinstance(records, list)
    out: dict[tuple[str, str], dict[str, object]] = {}
    for record in records:
        if record["lanes"] != lanes or record["kind"] not in {
            "program",
            "program-walk",
        }:
            continue
        key = (str(record["process"]), str(record["colour"]))
        entry = out.setdefault(
            key,
            {
                "process": key[0],
                "colour": key[1],
                "steps": record["steps"],
                "edges": record["edges"],
                "links": record["links"],
                "seconds": {},
            },
        )
        entry["seconds"][record["policy"]] = record["seconds_per_point"]  # type: ignore[index]
        if record["kind"] == "program-walk":
            walk = float(record["walk_seconds"])
            entry["walk_ns_per_step"] = walk / int(record["steps"]) * 1e9
            entry["walk_ns_per_edge"] = walk / int(record["edges"]) * 1e9
        details = record.get("details", {})
        if record["policy"] == GATE_POLICY:
            entry["rule_fits"] = details.get("fits")
            entry["steps_per_form"] = details.get("steps_per_form")
            entry["oracle_agreement"] = details.get("oracle_agreement", 0.0)
            entry["build_seconds_per_point"] = float(record["build_seconds"]) / lanes
            entry["reference_gcmac_per_s"] = record["reference_best_gcmac_per_s"]
    return list(out.values())


def _band(values: tuple[float, float]) -> str:
    return f"{values[0]:.3g}-{values[1]:.3g}"


def probe_text(
    label: str,
    report: Mapping[str, object],
    calibrations: Sequence[Calibration],
    lanes: int,
    allowance: float,
) -> str:
    """The synthetic probe: throughput per lane count, fastest form per shape
    and the design recipe at the probe's throughput (no verdict)."""
    chosen = next(c for c in calibrations if c.lanes == lanes)
    lines = [
        f"== {label} (synthetic probe): target features {report.get('target_features')}"
    ]
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
        "fastest probe lanes for this code generation:"
        f" {recommended_lanes(calibrations).lanes}"
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
    lines.append(
        f"{'process':14s} col  design recipe s/pt     probe recipe, no checks"
        f" (allowance {allowance:.0%})"
    )
    for row in PROCESSES:
        design = predict(row, None, "reduction-direction")
        probe = predict(row, chosen, "none", allowance)
        lines.append(
            f"{row.process:14s} {row.colour}  {_band(design):22s} {_band(probe)}"
        )
    return "\n".join(lines)


PROGRAM_COLUMNS = (
    GATE_POLICY,
    "counted",
    "oracle",
    "split-structured",
    "fused-structured",
    "fused-blocked",
)


def program_text(
    label: str, report: Mapping[str, object], lanes: int
) -> tuple[str, bool]:
    calibration = program_calibration(report, lanes)
    lines = [
        f"== {label} (real tries): target features {report.get('target_features')}"
    ]
    counts = sorted({t.lanes for t in program_timings(report)})
    lines.append(
        f"{'program':17s} lanes "
        + " ".join(f"{p:>16s}" for p in PROGRAM_COLUMNS)
        + "   (ms/pt; rule: G ref-CMAC/s)"
    )
    for count in counts:
        for entry in program_summary(report, count):
            seconds = entry["seconds"]
            cells = " ".join(
                f"{seconds[p] * 1e3:16.4g}" if p in seconds else f"{'-':>16s}"  # type: ignore[operator]
                for p in PROGRAM_COLUMNS
            )
            rate = entry.get("reference_gcmac_per_s")
            lines.append(
                f"{entry['process'] + ' ' + entry['colour']:17s} {count:5d} {cells}"
                + (f"   {rate:.2f}" if rate is not None else "")
            )
    lines.append(f"gate evaluated at {lanes} lanes, policy {GATE_POLICY!r}")
    for entry in program_summary(report, lanes):
        fits = ", ".join(
            f"{fit['form']}: {fit['per_step_seconds'] * 1e9:.0f} ns/step"
            f" + {fit['per_iteration_seconds'] * 1e9:.1f} ns/iteration"
            f" + work at {fit['gcmac_per_s']:.2f} G/s"
            for fit in entry.get("rule_fits") or ()
        )
        lines.append(
            f"  {entry['process']} {entry['colour']}: {entry['steps']} steps,"
            f" {entry['edges']} edges, {entry['links']} links; walk without kernels"
            f" {entry.get('walk_ns_per_step', float('nan')):.0f} ns/step"
            f" ({entry.get('walk_ns_per_edge', float('nan')):.0f} ns/edge);"
            f" factor build {entry.get('build_seconds_per_point', 0.0) * 1e3:.3f}"
            " ms/pt;"
            f" rule = oracle on {entry.get('oracle_agreement', 0.0):.0%} of steps,"
            f" forms {entry.get('steps_per_form')}"
        )
        lines.append(f"    self-calibrated rule: {fits}")
    steps = program_steps(report, lanes)
    if RULE_CALIBRATION in steps:
        rules = selection_rules(report, lanes)
        models = ", ".join(
            f"{form}: {m['per_step_seconds'] * 1e9:.0f} ns/step"
            f" + {m['per_iteration_seconds'] * 1e9:.1f} ns/iteration"
            f" + work at {m['gcmac_per_s'] or float('inf'):.2f} G/s"
            for form, m in rules["models"].items()  # type: ignore[union-attr]
        )
        lines.append(
            f"executor rule (step models fitted on {' '.join(RULE_CALIBRATION)}):"
            f" {models}"
        )
        for entry in rules["programs"]:  # type: ignore[union-attr]
            lines.append(
                f"  {entry['process']} {entry['colour']}: own step time over the"
                f" oracle: rule x{entry['rule_over_oracle']:.3f},"
                f" counted x{entry['counted_over_oracle']:.3f},"
                f" best single form x{entry['best_single_form_over_oracle']:.3f}"
            )
    lines.append(
        f"{'process':14s} col  design recipe s/pt     dressing s/pt          "
        "measured, no checks       with reduction-direction check"
    )
    for row in PROCESSES:
        design = predict(row, None, "reduction-direction")
        dressing = dressing_seconds(row, calibration)
        gate_mode = predict_measured(row, calibration, "none")
        checked = predict_measured(row, calibration, "reduction-direction")
        tag = "*" if (row.process, row.colour) in calibration.measured else " "
        lines.append(
            f"{row.process:14s} {row.colour}  {_band(design):22s}"
            f" {_band(dressing):21s}{tag}"
            f" {_band(gate_mode):25s} {_band(checked)}"
        )
    lines.append(
        "(* measured program; others scale from their class by Appendix B work)"
    )
    passed = True
    for row in g0_rows(calibration):
        verdict = "pass" if row["pass"] else "FAIL"
        passed &= bool(row["pass"])
        lines.append(
            f"G0 {row['process']:14s} target {_band(row['target_seconds'])} s"
            f"  gate {row['gate_seconds']:.3g} s"
            f"  headroom x{row['headroom']:.2f}  {verdict}"
            + ("" if row["measured"] else "  (extrapolated)")
        )
    lines.append(
        f"G0 {'PASS' if passed else 'FAIL'} at {lanes} lanes on real tries"
        f" (headroom >= x{G0_HEADROOM} at the upper end)"
    )
    return "\n".join(lines), passed


# ------------------------------------------------------------- gate G1

# G1 benchmark keys (tools/loop_benchmark) of the processes with a design row.
G1_PROCESSES = {
    "gg_gg": "gg>gg",
    "gg_gggg": "gg>gggg",
    "gg_ttxgg": "gg>tt~gg",
    "gg_ggggg": "gg>ggggg",
    "uux_ggggg": "uu~>ggggg",
    "uux_emepggg": "uu~>e-e+ggg",
    "uux_emepddxg": "uu~>e-e+dd~g",
    "gg_ttxemepg_qcd": "gg>tt~e-e+g",
}
G1_COLOUR = {"full": "FC", "lc": "LC", "nlc": "NLC"}
# Gate G0's measured dressing per point (DESIGN §5.4, rule, 16 lanes,
# production code generation, fused-factor builds included), seconds.
G0_DRESSING = {
    ("gg>gggg", "FC"): 0.171,
    ("gg>gggg", "LC"): 0.0453,
    ("gg>tt~gg", "FC"): 0.0546,
}
G0_THROUGHPUT = (1.29e9, 1.37e9)
NATIVE_LOOP_STAGES = ("seeds", "factors", "dressing", "scatter", "reduction")


def g1_recalibration(document: Mapping[str, object]) -> list[dict[str, object]]:
    """Measured G1 terms against the design recipe (DESIGN §1.3, §5.4).

    For every pyAmpliCol configuration of a G1 document (``python -m
    tools.loop_benchmark gate``) at its largest measured batch: the dressing
    term (dressing plus fused-factor builds), the NinjaRS stage (reduction,
    masters included), the rest of the native loop stage (seeds, scatter),
    everything outside the native loop stage (trees, Born, counterterms,
    assembly) as the recipe's outer factor, and the kernel throughput.
    Where the design has a row (``PROCESSES``) the recipe's finite-only band,
    its NinjaRS stage model R and gate G0's measured dressing are set beside
    the measurement, with the recalibrated factors measured / modelled.
    """

    rows: list[dict[str, object]] = []
    for process in document["processes"]:  # type: ignore[index]
        for name, config in process["pyamplicol"].items():
            batches = config["batches"]
            if not batches:
                continue
            size = max(batches, key=int)
            batch = batches[size]
            stages = batch["stages_per_point"]
            dressing = stages.get("dressing", 0.0) + stages.get("factors", 0.0)
            reduction = stages.get("reduction", 0.0)
            native = sum(stages.get(stage, 0.0) for stage in NATIVE_LOOP_STAGES)
            total = batch["median_seconds_per_point"]
            row: dict[str, object] = {
                "process": process["key"],
                "configuration": name,
                "batch": int(size),
                "total": total,
                "dressing": dressing,
                "reduction": reduction,
                "masters": stages.get("reduction: masters"),
                "native_loop_rest": native - dressing - reduction,
                "outside_native_loop": total - native,
                "outer_factor": total / (dressing + reduction)
                if dressing + reduction
                else None,
                "kernel_cmac_per_second": (batch.get("throughput") or {}).get(
                    "kernel_cmac_per_second"
                ),
            }
            design = G1_PROCESSES.get(process["key"])
            colour = G1_COLOUR.get(config["accuracy"])
            model = _process_row(design, colour) if design and colour else None
            if model is not None:
                low, high = predict(model, None, "none")
                r_lo, r_hi = (value / 1e3 for value in model.ninjars_ms)
                row["design"] = {
                    "recipe_finite_only": [low, high],
                    "ninjars_model": [r_lo, r_hi],
                    "dressing_model": [
                        model.dress_structured * QUARTIC[0] / DESIGN_THROUGHPUT[0],
                        model.dress_dense * QUARTIC[1] / DESIGN_THROUGHPUT[1],
                    ],
                    "g0_dressing": G0_DRESSING.get((design, colour)),
                    "total_over_recipe": [total / high, total / low],
                    "reduction_over_model": [reduction / r_hi, reduction / r_lo],
                }
            rows.append(row)
    return rows


# The G1 benchmark key whose measured outer factor stands for a design row's
# class (same colour where measured, else full colour) when G1 did not time
# the row itself: pure gluon from g g > g g g g, quark lines with gluons from
# u u~ > g g g g, V+jets from u u~ > e- e+ g g, top pairs from g g > t t~ g g.
G1_OUTER_CLASS = {
    "gg>gg": "gg_gg",
    "gg>gggg": "gg_gggg",
    "gg>ggggg": "gg_gggg",
    "uu~>ggggg": "uux_gggg",
    "uu~>e-e+ggg": "uux_emepgg",
    "uu~>e-e+dd~g": "uux_emepgg",
    "gg>tt~gg": "gg_ttxgg",
    "gg>tt~e-e+g": "gg_ttxgg",
}
# The NinjaRS stage with masters measured against the model R (G1 Table of
# 2026-10-24, six legs and fewer): the band used when a document has no
# measured design row (``g1_reduction_band``).
G1_REDUCTION_OVER_MODEL = (0.97, 1.39)


def g1_reduction_band(rows: Sequence[Mapping[str, object]]) -> tuple[float, float]:
    """The measured NinjaRS stage over the model R, lowest and highest over
    the full-colour native-input rows with a design row (``g1_recalibration``)."""

    values = [
        value
        for row in rows
        if row["configuration"] == "native/full" and row.get("design")
        for value in row["design"]["reduction_over_model"]  # type: ignore[index]
    ]
    if not values:
        return G1_REDUCTION_OVER_MODEL
    return (min(values), max(values))


def g1_estimates(document: Mapping[str, object]) -> list[dict[str, object]]:
    """Every design row (DESIGN Table 1a) re-estimated with G1's measured
    terms, finite only, at G1's largest batch with native inputs.

    The dressing term is the recipe's (its throughput holds: G1 measures
    G0's rate on the real plans), the NinjaRS term the model R times the
    measured band ``G1_REDUCTION_OVER_MODEL``, and the outer factor (trees,
    Born and counterterm inputs, seeds, scatter, glue) the measured
    total / (dressing + reduction) of the row's class representative
    (``G1_OUTER_CLASS``; the same colour where measured, else full colour,
    marked ``inferred``); a row that G1 timed itself takes its own measured
    outer factor and carries its measured total. The gate limit is G1's
    binding limit at the largest batch where G1 has one, else 1.3 x the best
    production provider of DESIGN Table 1b."""

    outer: dict[tuple[str, str], float] = {}
    measured: dict[tuple[str, str], float] = {}
    limits: dict[str, float] = {}
    for process in document["processes"]:  # type: ignore[index]
        config = process["pyamplicol"].get("native/full")
        if not config or not config["batches"]:
            continue
        verdict = config["batches"][max(config["batches"], key=int)]["gate"]
        if "binding_limit" in verdict:
            limits[process["key"]] = verdict["binding_limit"]
    recalibration = g1_recalibration(document)
    band = g1_reduction_band(recalibration)
    for row in recalibration:
        name = row["configuration"]
        if not str(name).startswith("native/") or str(name).count("/") != 1:
            continue
        colour = G1_COLOUR[str(name).split("/")[1]]
        if row["outer_factor"] is not None:
            outer[str(row["process"]), colour] = float(row["outer_factor"])  # type: ignore[arg-type]
        measured[str(row["process"]), colour] = float(row["total"])  # type: ignore[arg-type]
    out: list[dict[str, object]] = []
    for row in PROCESSES:
        own = next((k for k, v in G1_PROCESSES.items() if v == row.process), None)
        key = G1_OUTER_CLASS.get(row.process)
        if own is not None and ((own, row.colour) in outer or (own, "FC") in outer):
            key = own
        if key is None:
            continue
        factor = outer.get((key, row.colour))
        inferred = factor is None
        if factor is None:
            factor = outer.get((key, "FC"))
        if factor is None:
            continue
        r_lo, r_hi = (value / 1e3 for value in row.ninjars_ms)
        low = (
            row.dress_structured * QUARTIC[0] / DESIGN_THROUGHPUT[0] + r_lo * band[0]
        ) * factor
        high = (
            row.dress_dense * QUARTIC[1] / DESIGN_THROUGHPUT[1] + r_hi * band[1]
        ) * factor
        recipe = predict(row, None, "none")
        limit = None
        if row.colour == "FC":
            limit = limits.get(own) if own else None
            limit = gate(row.process) if limit is None else limit
        out.append(
            {
                "process": row.process,
                "label": " ".join(re.findall(r"e[-+]|[a-z]~?|>", row.process)),
                "colour": row.colour,
                "recipe_finite_only": list(recipe),
                "outer_class": key,
                "outer_factor": factor,
                "outer_inferred": inferred,
                "ninjars_over_model": list(band),
                "estimate": [low, high],
                "measured": measured.get((own, row.colour)) if own else None,
                "gate": limit,
                "headroom": None if limit is None else limit / high,
            }
        )
    return out


def g1_estimates_text(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "process       col  recipe s/pt (finite)  outer (class)        "
        "G1 estimate s/pt     measured  gate  headroom"
    ]
    for row in rows:
        recipe, estimate = row["recipe_finite_only"], row["estimate"]
        measured, limit = row["measured"], row["gate"]
        lines.append(
            f"{row['process']:13s} {row['colour']:3s}  "
            f"{recipe[0]:.4g}-{recipe[1]:.4g}".ljust(42)  # type: ignore[index]
            + f"{row['outer_factor']:.2f} ({row['outer_class']}"  # type: ignore[str-format]
            + (", FC)" if row["outer_inferred"] else ")")
            + f"  {estimate[0]:.4g}-{estimate[1]:.4g}"  # type: ignore[index]
            + ("" if measured is None else f"  {measured:.4g}")
            + ("" if limit is None else f"  {limit:.4g}  {row['headroom']:.2f}")
        )
    return "\n".join(lines)


def g1_text(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "process     configuration     B    total ms  dressing  reduction  "
        "rest  outside  outer  G CMAC/s  recipe ms (finite only)  total/recipe"
    ]
    for row in rows:
        design = row.get("design") or {}
        recipe = design.get("recipe_finite_only")
        ratio = design.get("total_over_recipe")
        throughput = row["kernel_cmac_per_second"]
        lines.append(
            f"{row['process']:11s} {row['configuration']:16s} {row['batch']:4d} "
            f"{row['total'] * 1e3:10.3f} {row['dressing'] * 1e3:9.3f} "
            f"{row['reduction'] * 1e3:9.3f} {row['native_loop_rest'] * 1e3:6.3f} "
            f"{row['outside_native_loop'] * 1e3:8.3f} "
            f"{row['outer_factor'] or float('nan'):6.2f} "
            f"{(throughput or float('nan')) / 1e9:9.2f}  "
            + (
                f"{recipe[0] * 1e3:8.2f}-{recipe[1] * 1e3:8.2f}         "
                f"{ratio[0]:.2f}-{ratio[1]:.2f}"
                if recipe
                else "—"
            )
        )
    return "\n".join(lines)


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
        help="probe recipe: share of the throughput kept for dispatch and gathers",
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
    parser.add_argument(
        "--g1",
        type=Path,
        metavar="G1_JSON",
        help="gate G1 document (tools/loop_benchmark gate): measured terms "
        "against the recipe",
    )
    args = parser.parse_args(argv)
    if args.g1 is not None:
        document = json.loads(args.g1.read_text())
        rows = g1_recalibration(document)
        estimates = g1_estimates(document)
        print(g1_text(rows))
        print()
        print(g1_estimates_text(estimates))
        if args.json is not None:
            args.json.write_text(
                json.dumps({"terms": rows, "estimates": estimates}, indent=1) + "\n"
            )
        return 0
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
    with_programs = [bool(program_timings(report)) for report in reports]
    if not all(with_programs):
        missing = [
            str(p)
            for p, ok in zip(args.benchmarks, with_programs, strict=True)
            if not ok
        ]
        print(f"G0 needs real-program records (--program); missing in {missing}")
    lanes = args.lanes
    if lanes is None and all(with_programs):
        lanes = program_lanes(reports)
    summary = []
    passed = all(with_programs)
    for path, report, has_programs in zip(
        args.benchmarks, reports, with_programs, strict=True
    ):
        entry: dict[str, object] = {
            "benchmark": str(path),
            "target_features": report.get("target_features"),
        }
        probe_lanes = lane_counts(report)
        if probe_lanes:
            calibrations = [calibrate(report, count) for count in probe_lanes]
            shown = (
                lanes if lanes in probe_lanes else recommended_lanes(calibrations).lanes
            )
            print(probe_text(str(path), report, calibrations, shown, args.allowance))
            chosen = next(c for c in calibrations if c.lanes == shown)
            entry["probe"] = {
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
                "lanes": shown,
                "allowance": args.allowance,
                "fastest_forms": fastest_forms(report, shown),
                "poly_layout_ratio": poly_layout_ratio(report, shown),
                "targets": [
                    {
                        "process": row.process,
                        "colour": row.colour,
                        "design_recipe": predict(row, None, "reduction-direction"),
                        "probe_no_checks": predict(row, chosen, "none", args.allowance),
                    }
                    for row in PROCESSES
                ],
            }
        if has_programs and lanes is not None:
            if lanes not in {t.lanes for t in program_timings(report)}:
                parser.error(f"{path} has no program records at {lanes} lanes")
            text, ok = program_text(str(path), report, lanes)
            print(text)
            passed &= ok
            calibration = program_calibration(report, lanes)
            entry["gate_lanes"] = lanes
            entry["programs"] = {
                str(count): program_summary(report, count)
                for count in sorted({t.lanes for t in program_timings(report)})
            }
            entry["targets"] = [
                {
                    "process": row.process,
                    "colour": row.colour,
                    "measured": (row.process, row.colour) in calibration.measured,
                    "design_recipe": predict(row, None, "reduction-direction"),
                    "dressing": dressing_seconds(row, calibration),
                    "no_checks": predict_measured(row, calibration, "none"),
                    "reduction_direction_check": predict_measured(
                        row, calibration, "reduction-direction"
                    ),
                }
                for row in PROCESSES
            ]
            entry["g0"] = g0_rows(calibration)
            if RULE_CALIBRATION in program_steps(report, lanes):
                entry["selection_rule"] = selection_rules(report, lanes)
        summary.append(entry)
    if lanes is not None and len(reports) > 1:
        print(f"lane count robust across all reports: {lanes}")
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=1) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
