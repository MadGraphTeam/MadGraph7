#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Schedule B against schedule M on generated slices (WP2.8; data for gate G1).

``export`` generates the slice of each requested process and colour accuracy
(loop_sm with MB = 0, QCD corrections; LC and NLC with the committed QCD
colour continuation), builds schedule M's planar state graph
(``loops.v2.merged.merged_graph``) and writes plan v4 with both schedules,
``<key>-<accuracy>.json``, plus ``<key>-<accuracy>.summary.json`` (graph
counts, the families left on schedule B and why, generation seconds).
Processes with a photon keep photon exchange only and use Furry's certified
zeros for the C-odd closed quark loops, as the native executor tests do,
until the electroweak signs (WP3.2) and the model certifier (WP3.5) land.

``report`` reads the output of the native bench

    cargo bench -p rusticol-core --features loop-v2 --bench loop_v2_schedules -- \\
        --plan DIR/gg_gggg-FC.json ... --json bench.json

and prints, per plan, the dressing-side time per point (dressing, fused
factors and split inputs, seeds, scatter; the reduction is the same for both
schedules and is not run) and modelled kernel work of B and M, their ratios,
the reals per lane of every live store, and the design's adoption rule
(§1.6 item 5): M is adopted for a class when it is at least 15 % faster.

Usage:
    python tools/loop_v2_schedules.py export --output DIR [--process KEY ...]
        [--accuracy fc lc] [--chunk-depth 1] [--ufo PATH]
    python tools/loop_v2_schedules.py report BENCH.json [...]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.loop_v2_operation_counts import UFO, compile_loop_sm  # noqa: E402

# key: (process, Born orders, ignored particles)
PROCESSES = {
    "gg_ggg": ("g g > g g g", {"QCD": 3, "QED": 0}, ()),
    "uux_ggg": ("u u~ > g g g", {"QCD": 3, "QED": 0}, ()),
    "gg_gggg": ("g g > g g g g", {"QCD": 4, "QED": 0}, ()),
    "uux_emepggg": ("u u~ > e- e+ g g g", {"QCD": 3, "QED": 2}, (23,)),
}
ACCURACIES = {"fc": "full", "nlc": "nlc", "lc": "lc"}
ADOPTION_GAIN = 0.15


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
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import legacy_capture

    process, born, ignored = PROCESSES[key]
    loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
    declaration = None
    if accuracy != "full":
        declaration = LoopColorDeclaration.from_dict(
            json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
        )
    request = SliceRequest(
        process,
        born,
        loop,
        fixed_parameters={"MB": "0"},
        ignored_particle_ids=ignored,
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


def _mb(reals: int, real_bytes: int) -> str:
    return f"{reals * real_bytes / 2**20:.2f}"


def report(options: argparse.Namespace) -> int:
    rows = []
    for path in options.bench:
        rows.extend(json.loads(Path(path).read_text()))
    header = (
        "| plan | families (M) | B steps / M states | B ms/pt | M ms/pt | M/B time "
        "| M/B work | B dressing | M dressing | memory per lane, MB (B / M) | adopt M |"
    )
    print(header)
    print("|---" * 11 + "|")
    for row in rows:
        b, m = row["B"], row["M"]
        time_ratio = m["seconds_per_point"] / b["seconds_per_point"]
        work = row["work_per_lane"]
        work_ratio = work["M"] / work["B"]
        real = b["real_bytes"]
        b_memory = sum(b["reals_per_lane"].values())
        m_memory = sum(m["reals_per_lane"].values())
        adopt = "yes" if time_ratio <= 1 - ADOPTION_GAIN else "no"
        print(
            f"| {Path(row['plan']).stem} | {row['families']} ({row['merged_families']}) "
            f"| {row['b_steps']} / {row['m_states']} "
            f"| {b['seconds_per_point'] * 1e3:.2f} | {m['seconds_per_point'] * 1e3:.2f} "
            f"| {time_ratio:.3f} | {work_ratio:.3f} "
            f"| {b['stages_per_point'].get('dressing', 0) * 1e3:.2f} "
            f"| {m['stages_per_point'].get('dressing', 0) * 1e3:.2f} "
            f"| {_mb(b_memory, real)} / {_mb(m_memory, real)} | {adopt} |"
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
        "--accuracy", nargs="+", default=["fc", "lc"], choices=list(ACCURACIES)
    )
    writer.add_argument("--chunk-depth", type=int, default=1)
    writer.add_argument("--ufo", type=Path, default=UFO)
    reader = commands.add_parser("report", help="the adoption table of bench runs")
    reader.add_argument("bench", nargs="+", type=Path)
    options = parser.parse_args(argv)
    return export(options) if options.command == "export" else report(options)


if __name__ == "__main__":
    sys.exit(main())
