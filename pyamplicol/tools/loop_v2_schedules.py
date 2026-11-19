#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Schedule B against schedule M on generated slices (WP2.8; data for gate G1).

``export`` generates the slice of each requested process and colour accuracy
(loop_sm with MB = 0, QCD corrections; LC and NLC with the committed QCD
colour continuation), builds schedule M's planar state graph
(``loops.v2.merged.merged_graph``) and writes plan v4 with both schedules,
``<key>-<accuracy>.json``, plus ``<key>-<accuracy>.summary.json`` (graph
counts, the families left on schedule B and why, the squared loop masses at
the default parameters with MT = 173, generation seconds).
Processes with a photon keep photon exchange only and use Furry's certified
zeros for the C-odd closed quark loops, as the native executor tests do,
until the electroweak signs (WP3.2) and the model certifier (WP3.5) land.

``measure`` runs the complete native evaluation of such plans with both
schedules through the extension's ``V2Executor``: dressing, fused factors
and split inputs, seeds, scatter and the NinjaRS reduction (binary64, poles
off, no stability check: the speed-gate protocol), on a batch of
phase-space points (RAMBO at sqrt(s) = 1 TeV with the calibration cuts,
momenta rounded to 17 digits as binary64 inputs are, conservation exact;
mu_R = 1 TeV; the summary's masses) with random block rows, Born
components and kernel constants. The cost of the dressing and the reduction
does not depend on these values; the trees, Born amplitudes and counterterm
scalars that a real point needs are still Python (WP3.1) and are not timed.
Per plan, thread count and schedule it records the best of ``--reps`` runs
per point and stage, the reduction lanes, the failed points and the reals
per lane of every live store.

``report`` reads the output of ``measure`` and of the native bench

    cargo bench -p rusticol-core --features loop-v2 --bench loop_v2_schedules -- \\
        --plan DIR/gg_gggg-FC.json ... --json bench.json

(dressing side only, with the modelled kernel work of the chosen forms and
the share of the seed injection in it) and prints the adoption table: M/B
time and, for ``measure`` rows, M/B with the reduction included, the memory
per lane of the live stores, and the design's adoption rule (§1.6 item 5): M
is adopted for a class when it is at least 15 % faster with the reduction
included (``measure``), or on the dressing side (bench rows).

Usage:
    python tools/loop_v2_schedules.py export --output DIR [--process KEY ...]
        [--accuracy fc nlc lc] [--chunk-depth 1] [--ufo PATH]
    python tools/loop_v2_schedules.py measure PLAN.json [...] --json OUT.json
        [--threads 1 8] [--points 16] [--lanes 16] [--reps 3]
    python tools/loop_v2_schedules.py report RESULTS.json [...]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.loop_v2_operation_counts import UFO, compile_loop_sm  # noqa: E402

# key: (process, Born orders, ignored particles)
PROCESSES = {
    "uux_ddxg": ("u u~ > d d~ g", {"QCD": 3, "QED": 0}, ()),
    "gg_ggg": ("g g > g g g", {"QCD": 3, "QED": 0}, ()),
    "uux_ggg": ("u u~ > g g g", {"QCD": 3, "QED": 0}, ()),
    "gg_gggg": ("g g > g g g g", {"QCD": 4, "QED": 0}, ()),
    "uux_gggg": ("u u~ > g g g g", {"QCD": 4, "QED": 0}, ()),
    "uux_emepgg": ("u u~ > e- e+ g g", {"QCD": 2, "QED": 2}, (23,)),
    "uux_emepggg": ("u u~ > e- e+ g g g", {"QCD": 3, "QED": 2}, (23,)),
    "uux_emepddxg": ("u u~ > e- e+ d d~ g", {"QCD": 3, "QED": 2}, (23,)),
}
ACCURACIES = {"fc": "full", "nlc": "nlc", "lc": "lc"}
PARAMETERS = {"MT": "173"}
ADOPTION_GAIN = 0.15
SCALE_SQUARED = "1000000"
# The dressing side of a native evaluation: everything but the reduction
# (the same NinjaRS work for both schedules).
DRESSING_SIDE = ("dressing", "factors", "seeds", "scatter")
NATIVE_STAGES = (*DRESSING_SIDE, "reduction")


def _furry_zeros(generate):
    """``generate_counterterm_sites`` with Furry's certified zeros for the
    C-odd closed quark loops with one photon and two gluons."""

    from pyamplicol.loops.v2 import ct_sites

    def with_zeros(families, table, rules, data, selection, **options):
        zeros = [
            ct_sites.CertifiedZero(
                signature,
                species,
                "Furry: C-odd closed quark loop (one photon, two gluons)",
            )
            for signature, sets in ct_sites.divergent_witnesses(families, data).items()
            if signature.fields == ("A", "G", "G")
            for species in sets
            if all(1 <= pdg <= 6 for pdg in species)
        ]
        return generate(families, table, rules, data, selection, zeros=zeros, **options)

    return with_zeros


def generate(compiled: Any, native: Any, key: str, accuracy: str) -> Any:
    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loop_color_declaration import (
        resolve_loop_color_declaration,
    )

    process, born, ignored = PROCESSES[key]
    loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
    declaration = None
    if accuracy != "full":
        # The model's declaration: for loop_sm the shipped one (MQ-NC); the
        # example QCD subset does not cover the g g g A R2 the photon rows
        # reach.
        declaration = resolve_loop_color_declaration(compiled.raw_loop_model)
    request = SliceRequest(
        process,
        born,
        loop,
        fixed_parameters={"MB": "0"},
        particle_veto=ignored,
        accuracy=accuracy,
        colour_declaration=declaration,
    )
    original = assemble.generate_counterterm_sites
    if born.get("QED"):
        assemble.generate_counterterm_sites = _furry_zeros(original)
    try:
        return LoopSlice.generate(compiled, request, native)
    finally:
        assemble.generate_counterterm_sites = original


def export(options: argparse.Namespace) -> int:
    from pyamplicol.loops.native import load_loop_native
    from pyamplicol.loops.v2.assemble import Precision, _mass_values
    from pyamplicol.loops.v2.merged import merged_graph
    from pyamplicol.loops.v2.plan_export import export_plan

    options.output.mkdir(parents=True, exist_ok=True)
    native = load_loop_native()
    with tempfile.TemporaryDirectory() as scratch:
        compiled = compile_loop_sm(options.ufo, Path(scratch))
        for key in options.process:
            for label in options.accuracy:
                name = f"{key}-{label.upper()}"
                start = time.perf_counter()
                slice_ = generate(compiled, native, key, ACCURACIES[label])
                generated = time.perf_counter() - start
                start = time.perf_counter()
                graph = merged_graph(slice_, chunk_depth=options.chunk_depth)
                graphed = time.perf_counter() - start
                start = time.perf_counter()
                exported = export_plan(slice_, forward=False, merged=graph)
                written = time.perf_counter() - start
                (options.output / f"{name}.json").write_text(exported.json())
                squared = _mass_values(
                    slice_.compiled,
                    PARAMETERS,
                    exported.masses,
                    Precision.double().working_digits,
                )
                summary = {
                    "process": PROCESSES[key][0],
                    "accuracy": ACCURACIES[label],
                    "chunk_depth": options.chunk_depth,
                    "families": len(slice_.plan.schedules),
                    "b_trie_nodes": sum(
                        len(s.trie.nodes) for s in slice_.plan.schedules.values()
                    ),
                    "merged": graph.counts(),
                    "fallback": sorted(set(graph.fallback.values())),
                    "fallback_families": len(graph.fallback),
                    # Per plan mass index: (parameter, squared mass) at
                    # PARAMETERS.
                    "masses": [
                        [None if mass.massless else mass.mass, str(squared[mass])]
                        for mass in exported.masses
                    ],
                    "seconds": {
                        "slice": generated,
                        "merged_graph": graphed,
                        "export": written,
                    },
                }
                (options.output / f"{name}.summary.json").write_text(
                    json.dumps(summary, indent=1) + "\n"
                )
                print(name, json.dumps(summary), flush=True)
    return 0


# --------------------------------------------------------------------- measure


def _point(outgoing: int, rng: random.Random) -> list[list[Fraction]]:
    """A 2 -> ``outgoing`` RAMBO point at sqrt(s) = 1 TeV (binary64) with its
    final-state components rounded to 17 significant digits, as a user's
    binary64 momenta are: the last leg's transverse momentum and the
    incoming momenta along +-z make momentum conservation exact."""

    from tools.loop_v2_stability_calibration import ENERGY, _passes_cuts, _rambo

    while True:
        floats = _rambo(outgoing, ENERGY, rng)
        if _passes_cuts(floats):
            break
    final = [[Fraction(f"{x:.17g}") for x in p] for p in floats]
    for mu in (1, 2):
        final[-1][mu] = -sum(p[mu] for p in final[:-1])
    energy = sum(p[0] for p in final)
    z = sum(p[3] for p in final)
    zero = Fraction(0)
    first = [(energy + z) / 2, zero, zero, (energy + z) / 2]
    second = [(energy - z) / 2, zero, zero, -(energy - z) / 2]
    return [first, second, *final]


def _inputs(program: dict, masses: list[str], points: int, seed: int) -> dict:
    """Batch inputs of the extension's ``V2Executor`` (binary64): points in
    the all-incoming convention (``_point``) and random carrier values
    (module docstring)."""

    rng = random.Random(seed)

    def value() -> list[float]:
        return [rng.uniform(-1, 1), rng.uniform(-1, 1)]

    momenta = []
    for _ in range(points):
        point = _point(program["legs"] - 2, rng)
        incoming = [*point[:2], *[[-x for x in p] for p in point[2:]]]
        if any(sum(p[mu] for p in incoming) != 0 for mu in range(4)):
            raise AssertionError("a point that does not conserve momentum")
        momenta.append([_text(x) for p in incoming for x in p])
    block = sum(program["block_rows"])
    born = program["rows"] * program["born_components"]
    return {
        "momenta": momenta,
        "scales": [SCALE_SQUARED] * points,
        "masses": masses,
        "constants": [value() for _ in range(program["constants"])],
        "block_rows": [[value() for _ in range(block)] for _ in range(points)],
        "born": [[value() for _ in range(born)] for _ in range(points)],
    }


def _text(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def measure(options: argparse.Namespace) -> int:
    from pyamplicol.loops.native import load_loop_native

    native = load_loop_native()
    records = []
    for path in options.plans:
        text = path.read_text()
        program = json.loads(text)
        summary_path = path.with_name(f"{path.stem}.summary.json")
        summary = json.loads(summary_path.read_text())
        masses = [squared for _name, squared in summary["masses"]]
        if len(masses) != program["masses"]:
            raise ValueError(f"{summary_path}: masses do not match the plan")
        merged = program["merged"] or {"families": [], "states": [], "chunks": []}
        inputs = _inputs(program, masses, options.points, options.seed)
        for threads in options.threads:
            executor = native.V2Executor(
                text, "double", None, lanes=options.lanes, threads=threads
            )
            record = {
                "kind": "measure",
                "plan": str(path),
                "process": summary["process"],
                "accuracy": summary["accuracy"],
                "families": len(program["families"]),
                "merged_families": len(merged["families"]),
                "m_states": len(merged["states"]),
                "m_chunks": len(merged["chunks"]),
                "points": options.points,
                "lanes": options.lanes,
                "threads": threads,
                "chunk_families": options.chunk_families,
                "part_chunks": options.part_chunks,
            }
            best: dict[str, tuple] = {}
            # One untimed run per schedule, then the schedules alternate.
            for rep in range(options.reps + 1):
                for schedule in ("B", "M"):
                    start = time.perf_counter()
                    output = executor.evaluate(
                        inputs,
                        poles="none",
                        check=False,
                        schedule=schedule,
                        chunk_families=options.chunk_families,
                        part_chunks=options.part_chunks,
                    )
                    seconds = (time.perf_counter() - start) / options.points
                    report = output["report"]
                    native_seconds = sum(
                        report["seconds"].get(stage, 0.0) for stage in NATIVE_STAGES
                    )
                    if rep and (
                        schedule not in best or native_seconds < best[schedule][0]
                    ):
                        best[schedule] = native_seconds, seconds, output
            for schedule, (native_seconds, seconds, output) in best.items():
                report = output["report"]
                record[schedule] = {
                    "seconds_per_point": native_seconds / options.points,
                    "wall_seconds_per_point": seconds,
                    "stages_per_point": {
                        stage: value / options.points
                        for stage, value in report["seconds"].items()
                    },
                    "counts": report["counts"],
                    "reals_per_lane": report["reals_per_lane"],
                    "real_bytes": report["real_bytes"],
                    "failed_points": sum(1 for f in output["failures"] if f),
                }
                stages = record[schedule]["stages_per_point"]
                print(
                    f"{path.stem} t={threads} {schedule}: "
                    f"{record[schedule]['seconds_per_point'] * 1e3:.2f} ms/pt native "
                    f"(reduction {stages.get('reduction', 0) * 1e3:.2f}, dressing "
                    f"{stages.get('dressing', 0) * 1e3:.2f}; wall {seconds * 1e3:.2f}),"
                    f" failed points {record[schedule]['failed_points']}",
                    flush=True,
                )
            records.append(record)
            if options.json:
                options.json.write_text(json.dumps(records, indent=1) + "\n")
    return 0


# --------------------------------------------------------------------- report


def _kb(reals: int, real_bytes: int) -> str:
    return f"{reals * real_bytes / 1024:.0f}"


def _stores(run: dict) -> tuple[int, int, int]:
    """Reals per lane of a run's chunk inputs, live open-loop objects (one
    family walk or one M chunk, per thread) and family numerators (one
    bucket chunk, or M's accumulators plus B's bucket chunk of the families M
    leaves)."""

    reals = run["reals_per_lane"]
    inputs = reals.get("chunk inputs (factors, vertex tensors, currents)", 0)
    objects = max(
        reals.get("open objects (one family walk)", 0),
        reals.get("open objects (one merged chunk)", 0),
    )
    numerators = reals.get("family numerators (one bucket chunk)", 0) + reals.get(
        "merged family accumulators", 0
    )
    return inputs, objects, numerators


def _dressing_side(run: dict) -> float:
    stages = run["stages_per_point"]
    return sum(stages.get(stage, 0.0) for stage in DRESSING_SIDE)


def report(options: argparse.Namespace) -> int:
    """The adoption table of ``measure`` and bench runs (module docstring)."""

    rows = []
    for path in options.results:
        rows.extend(json.loads(Path(path).read_text()))
    print(
        "| plan | threads | families (M) | B ms/pt | M ms/pt | M/B | "
        "dressing side B, M ms/pt (M/B) | reduction B, M ms/pt | M/B work "
        "| kB per lane B (inputs, objects, numerators) | kB per lane M | adopt M |"
    )
    print("|---" * 12 + "|")
    for row in rows:
        b, m = row["B"], row["M"]
        real = b["real_bytes"]
        memory = {
            name: ", ".join(_kb(x, real) for x in _stores(run))
            for name, run in (("B", b), ("M", m))
        }
        side = _dressing_side(b), _dressing_side(m)
        if row.get("kind") == "measure":
            total = b["seconds_per_point"], m["seconds_per_point"]
            reduction = (
                f"{b['stages_per_point'].get('reduction', 0) * 1e3:.2f}, "
                f"{m['stages_per_point'].get('reduction', 0) * 1e3:.2f}"
            )
            work = "—"
            failed = b["failed_points"] + m["failed_points"]
            if failed:
                reduction += f" ({failed} failed points)"
        else:
            total = side
            reduction = "not run"
            work = f"{row['work_per_lane']['M'] / row['work_per_lane']['B']:.3f}"
        ratio = total[1] / total[0]
        adopt = "yes" if ratio <= 1 - ADOPTION_GAIN else "no"
        print(
            f"| {Path(row['plan']).stem} | {row['threads']} "
            f"| {row['families']} ({row['merged_families']}) "
            f"| {total[0] * 1e3:.2f} | {total[1] * 1e3:.2f} | {ratio:.3f} "
            f"| {side[0] * 1e3:.2f}, {side[1] * 1e3:.2f} ({side[1] / side[0]:.3f}) "
            f"| {reduction} | {work} "
            f"| {memory['B']} | {memory['M']} | {adopt} |"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    commands = parser.add_subparsers(dest="command", required=True)
    writer = commands.add_parser("export", help="write plans with both schedules")
    writer.add_argument("--output", type=Path, required=True)
    writer.add_argument(
        "--process", nargs="+", default=list(PROCESSES), choices=list(PROCESSES)
    )
    writer.add_argument(
        "--accuracy", nargs="+", default=list(ACCURACIES), choices=list(ACCURACIES)
    )
    writer.add_argument("--chunk-depth", type=int, default=1)
    writer.add_argument("--ufo", type=Path, default=UFO)
    runner = commands.add_parser("measure", help="B and M with the reduction")
    runner.add_argument("plans", nargs="+", type=Path)
    runner.add_argument("--json", type=Path)
    runner.add_argument("--threads", nargs="+", type=int, default=[1, 8])
    runner.add_argument("--points", type=int, default=16)
    runner.add_argument("--lanes", type=int, default=16)
    runner.add_argument("--reps", type=int, default=3)
    runner.add_argument("--seed", type=int, default=20261022)
    runner.add_argument(
        "--chunk-families",
        type=int,
        help="families per streamed bucket chunk of B (default: the executor's)",
    )
    runner.add_argument(
        "--part-chunks",
        type=int,
        help="M chunks per streamed part (default: the executor's)",
    )
    reader = commands.add_parser("report", help="the adoption table")
    reader.add_argument("results", nargs="+", type=Path)
    options = parser.parse_args(argv)
    if options.command == "export":
        return export(options)
    if options.command == "measure":
        return measure(options)
    return report(options)


if __name__ == "__main__":
    sys.exit(main())
