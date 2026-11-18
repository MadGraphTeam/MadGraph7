#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Exact operation counts of the v2 schedules F, B and M for gate G1 (WP1.5).

The Python reference executor (``pyamplicol.loops.v2.reference_executor``)
walks every schedule without numerics and counts the complex multiply-adds a
numeric run performs per point and observable column, for both kernel forms
of every dressing step (DESIGN.md §5.2): ``fused`` (X(J) per block row) and
``split`` (``split-accumulate`` T = sum O x J, then ``split-vertex`` once per
vertex group); ``best`` takes the cheaper form per step. ``seed`` steps start
from a Born seed (rank 0, diagonal in the cut index), ``step`` from an open-
loop object. ``closure`` traces the cut index (F also contracts the seeds
there) and ``frame-map`` re-expands F/M closures in the family frame; G1
multiplies these counts by G0's measured kernel throughput (DESIGN.md §2.2).

Configuration (the providers', ``loop_sm-no_b_mass``): loop_sm with MB = 0,
QCD corrections, kernels of the WP1.1 catalogue (their structural entries),
colour-ordered contents (``PlanarWords``; the colour-singlet legs form one
letter), helicity rows two per leg except one for a massless fermion whose
incoming field is a particle (helicity conservation, as Appendix B), B frames
cost-optimal per family by the default cost model. Every ghost and quark-loop
orientation is its own skeleton (no C-parity folding). LC keeps the skeletons
without a closed quark loop (the all-gluon face rule, DESIGN.md §7.5).

Block chiralities are synthetic (``SyntheticBlockRows``): each helicity row of
a massless-quark block has one Weyl chirality, the rows cycling through
them, as a tree-engine block does; which row has which chirality is not the
tree engine's. Kernel entry counts per chirality are equal for the two Weyl
chiralities, so the counts depend only on how many rows each has.

Schedule M is admissible at LC and, in full colour, only for classes with a
loop-level colour certificate (DESIGN.md §2.2); none exists yet (WP2.4), so
``M`` in FC records B's fallback. ``M-uncertified`` records what M would
count once WP2.4 certifies the class (gluon and ghost rings, one quark line);
it is not admissible for G1 today. Several quark lines (uu~ -> e-e+dd~g) are
a generic class that stays on B, so they have no such record.

Usage:
    python tools/loop_v2_operation_counts.py [--output PATH] [--ufo PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # the ``tools`` package, run as a script
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "tests" / "fixtures" / "loop" / "v2" / "operation_counts.json"
# As the tests' loop_sm_ufo input: the environment variable, else the
# checkout's MadGraph copy.
UFO = Path(
    os.environ.get(
        "PYAMPLICOL_NLO_UFO_REFERENCE",
        ROOT / "FOR_REFERENCE_ONLY" / "MG5_aMC_v3_7_0" / "models" / "loop_sm",
    )
)

PROCESSES = (
    {
        "process": "g g > g g g g",
        "incoming": ["G", "G"],
        "outgoing": ["G", "G", "G", "G"],
        "born_orders": {"QCD": 4},
        "colour": ["LC", "FC"],
    },
    {
        "process": "u u~ > e- e+ d d~ g",
        "incoming": ["u", "u~"],
        "outgoing": ["e-", "e+", "d", "d~", "G"],
        "born_orders": {"QCD": 3, "QED": 2},
        "colour": ["FC"],
    },
    {
        "process": "u u~ > e- e+ g g g",
        "incoming": ["u", "u~"],
        "outgoing": ["e-", "e+", "G", "G", "G"],
        "born_orders": {"QCD": 3, "QED": 2},
        "colour": ["FC"],
    },
)
SCHEDULES = ("B", "M", "F")


def compile_loop_sm(ufo: Path, cache: Path) -> Any:
    """loop_sm compiled with the checked scalar-arity view (as the tests)."""

    from pyamplicol.models.loading import compile_model_source, load_compiled_model
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    model = compile_model_source(ufo, cache_dir=cache / "model-cache")
    model = replace(
        model,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            model.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    return load_compiled_model(model.write(cache / "loop-sm.json"))


def prepare(compiled: Any) -> dict[str, Any]:
    """The request-scoped native model, its family catalogue and the kernels."""

    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.v2.catalogue import build_loop_kernel_catalogue
    from pyamplicol.loops.v2.families import ModelLoopCatalogue

    structure = prepare_loop_request_model(
        compiled,
        born_order_bounds={"QCD": 5, "QED": 2},
        loop_order_bounds={"QCD": 7, "QED": 2},
        fixed_parameters={"MB": "0"},
    ).structure
    data = json.loads(structure.native.to_json())
    return {
        "compiled": compiled,
        "data": data,
        "catalogue": ModelLoopCatalogue(
            data, species_orders=structure.perturbative_species_orders
        ),
        "kernels": build_loop_kernel_catalogue(compiled),
    }


def count_process(prepared: Mapping[str, Any], entry: Mapping[str, Any]) -> dict:
    """Counts of every schedule at every colour accuracy of one process."""

    from pyamplicol.loops.v2 import reference_executor as rx
    from pyamplicol.loops.v2.families import (
        LoopOrderSelection,
        ModelBlockTable,
        generate_families,
    )
    from pyamplicol.loops.v2.schedules import SkeletonView, build_schedules, cheapest
    from pyamplicol.loops.v2.skeletons import PlanarWords

    data, catalogue = prepared["data"], prepared["catalogue"]
    born = entry["born_orders"]
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
    colour = PlanarWords(len(legs), letters + ([singlets] if singlets else []))
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
    full = (1 << len(legs)) - 1
    seeds = rx.SyntheticSeeds(blocks.row_set(full), values=False)
    executor = rx.ReferenceExecutor(kinematics, seeds, numeric=False)

    def rows(mask: int) -> int:
        return len(blocks.row_set(mask))

    def no_quark_loop(skeleton) -> bool:
        return not any(
            abs(catalogue.species(name).colour) == 3 for name in skeleton.species
        )

    quark_lines = sum(abs(leg.colour) == 3 for leg in legs) // 2
    out: dict[str, Any] = {}
    for accuracy in entry["colour"]:
        view = SkeletonView(
            catalogue, colour, keep=no_quark_loop if accuracy == "LC" else None
        )
        plan = build_schedules(
            families, catalogue, colour, policy=cheapest(rows), view=view
        )
        record: dict[str, Any] = {
            "families": len(plan.schedules),
            "skeletons": len(rx.plan_skeletons(plan)),
            "trie_nodes": sum(len(s.trie.nodes) for s in plan.schedules.values()),
        }
        colour_accuracy = "lc" if accuracy == "LC" else "full"
        for schedule in SCHEDULES:
            record[schedule] = _counts(executor.run(plan, schedule, colour_accuracy))
        if accuracy == "FC" and quark_lines <= 1:
            record["M-uncertified"] = _counts(
                executor.run(plan, "M", colour_accuracy, certified=lambda s: True)
            )
        out[accuracy] = record
    return out


def _counts(execution: Any) -> dict[str, Any]:
    if execution.fallback:
        return {"fallback": execution.fallback}
    return dict(sorted(execution.counts.items()))


def operation_counts(prepared: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "description": (__doc__ or "").split("\n\nUsage:")[0],
        "units": "complex multiply-adds per point and observable column",
        "processes": [
            {
                "process": entry["process"],
                "born_orders": entry["born_orders"],
                "counts": count_process(prepared, entry),
            }
            for entry in PROCESSES
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--ufo", type=Path, default=UFO)
    options = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as scratch:
        prepared = prepare(compile_loop_sm(options.ufo, Path(scratch)))
        payload = operation_counts(prepared)
    options.output.write_text(json.dumps(payload, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
