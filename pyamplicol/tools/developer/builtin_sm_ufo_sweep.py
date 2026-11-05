#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""The built-in SM against the UFO sm: process sets, points and references.

The 2 -> 2 processes come from the built-in particle table: every pair of
initial and final particles that conserves electric charge, baryon and
lepton-family number (the built-in parser's quantum numbers) and whose four
colour representations contain a singlet. They fall into classes under
crossing, charge-parity conjugation and the relabellings of particles that
map the built-in vertex table onto itself (the lepton families and the two
light quark generations); ``class_representatives`` keeps the first member of
each class.

The reference is the UFO ``sm`` of MG5_aMC 3.7.0 with its ``no_masses``
restriction (massless b and tau, as built in), compiled lane, binary64, at
two RAMBO points per process (``on_shell_points``); ``None`` marks a process
without a tree amplitude. Writing the committed reference::

    python tools/developer/builtin_sm_ufo_sweep.py reference \\
        --madgraph "$WORKSPACE/reference/sources/MG5_aMC_v3_7_0" \\
        --work /tmp/sweep --output tests/fixtures/tree/builtin-sm-ufo-sm.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

from pyamplicol.generation.phase_space import massive_rambo_final_state
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.builtin.process_catalog import (
    ANTI_PARTICLE,
    PDGS,
    selection_metadata,
)
from pyamplicol.models.builtin.process_ir import build_process_ir

LANES = ("recurrence", "compiled", "eager", "on-the-fly")
SQRT_S = 700.0
SEEDS = (9001, 9002)
# Decays and 2 -> 3 processes besides the generated 1 -> 2 decays: one per
# interaction family (gauge, Yukawa, Higgs self-coupling, QCD with
# electroweak) and both quark-line electroweak currents.
SAMPLE = (
    "z > e+ e- a",
    "w+ > u d~ g",
    "t > b e+ ve",
    "h > e+ e- z",
    "e+ e- > a a a",
    "e+ e- > u u~ g",
    "u d~ > e+ ve g",
    "u u~ > d d~ a",
    "e+ e- > z h a",
    "g g > t t~ h",
    "h h > h h h",
    "w+ w- > z h",
)
# Electroweak exchange between quark lines with its explicit coupling orders
# (the minimal orders keep QCD alone).
ORDERED = (
    "u u~ > d d~ [QCD=0 QED=2]",
    "u d > u d [QCD=0 QED=2]",
    "t t~ > b b~ [QCD=0 QED=2]",
)


def _outgoing(process: str) -> tuple[str, ...]:
    initial, final = (side.split() for side in process.split(">"))
    return (*(ANTI_PARTICLE[name] for name in initial), *final)


def _conserved(outgoing: Sequence[str]) -> bool:
    metadata = [selection_metadata(name) for name in outgoing]
    if sum(item.charge3 for item in metadata) or sum(item.family for item in metadata):
        return False
    colours = [item.color_rep for item in metadata]
    coloured = [rep for rep in colours if rep != 1]
    triality = sum(1 if rep == 3 else -1 if rep == -3 else 0 for rep in colours)
    return len(coloured) != 1 and triality % 3 == 0


def two_to_two_processes() -> tuple[str, ...]:
    """Every 2 -> 2 process of the built-in particle table (module docstring)."""

    names = sorted(PDGS, key=lambda name: (abs(int(PDGS[name])), int(PDGS[name])))
    processes = []
    for initial in itertools.combinations_with_replacement(names, 2):
        for final in itertools.combinations_with_replacement(names, 2):
            process = f"{' '.join(initial)} > {' '.join(final)}"
            if _conserved(_outgoing(process)):
                processes.append(process)
    return tuple(processes)


def one_to_two_decays() -> tuple[str, ...]:
    """The kinematically open 1 -> 2 decays of every built-in particle with a
    width, under the quantum numbers of ``two_to_two_processes``."""

    model = BuiltinSMModel()
    names = sorted(PDGS, key=lambda name: (abs(int(PDGS[name])), int(PDGS[name])))
    decays = []
    for parent in names:
        mass = model.mass(int(PDGS[parent]))
        if model.width(int(PDGS[parent])) == 0.0:
            continue
        for final in itertools.combinations_with_replacement(names, 2):
            process = f"{parent} > {' '.join(final)}"
            if _conserved(_outgoing(process)) and mass > sum(
                model.mass(int(PDGS[name])) for name in final
            ):
                decays.append(process)
    return tuple(decays)


def vertex_table_relabellings(model: BuiltinSMModel | None = None) -> tuple[dict, ...]:
    """Particle relabellings that map the built-in vertex table onto itself.

    Candidates permute external particles with identical properties (spin,
    colour, charge, mass, width, weak quantum numbers), antiparticles along.
    """

    model = model or BuiltinSMModel()
    external = sorted({abs(int(pdg)) for pdg in PDGS.values()} & set(model.particles))
    groups: dict[tuple, list[int]] = {}
    for pdg in external:
        particle = model.particles[pdg]
        key = (
            particle.spin,
            particle.color_rep,
            particle.charge,
            particle.mass,
            particle.width,
            particle.weak_isospin,
            particle.weak_hypercharge,
            particle.anti_pdg == pdg,
        )
        groups.setdefault(key, []).append(pdg)
    table = {(v.kind, tuple(v.particles), tuple(v.coupling)) for v in model.vertices}
    relabellings = []
    for images in itertools.product(
        *(itertools.permutations(members) for members in groups.values())
    ):
        mapping: dict[int, int] = {}
        for members, image in zip(groups.values(), images, strict=True):
            for source, target in zip(members, image, strict=True):
                mapping[source] = target
                if model.particles[source].anti_pdg != source:
                    mapping[-source] = -target

        def relabel(pdg: int, mapping: dict[int, int] = mapping) -> int:
            return mapping.get(pdg, pdg)

        if {
            (kind, tuple(relabel(pdg) for pdg in particles), coupling)
            for kind, particles, coupling in table
        } == table:
            relabellings.append(mapping)
    return tuple(relabellings)


def class_representatives(processes: Iterable[str]) -> tuple[str, ...]:
    """The first process of each class under crossing, charge-parity
    conjugation and ``vertex_table_relabellings``."""

    name_by_pdg = {int(pdg): name for name, pdg in PDGS.items()}
    relabellings = vertex_table_relabellings()
    order = {name: index for index, name in enumerate(sorted(PDGS))}
    seen: set[tuple[str, ...]] = set()
    representatives = []
    for process in processes:
        outgoing = [int(PDGS[name]) for name in _outgoing(process)]
        keys = []
        for mapping in relabellings:
            for conjugate in (False, True):
                names = (name_by_pdg[mapping.get(pdg, pdg)] for pdg in outgoing)
                if conjugate:
                    names = (ANTI_PARTICLE[name] for name in names)
                keys.append(tuple(sorted(names, key=order.__getitem__)))
        key = min(keys)
        if key not in seen:
            seen.add(key)
            representatives.append(process)
    return tuple(representatives)


def relabelling_representatives(processes: Iterable[str]) -> tuple[str, ...]:
    """The first process of each orbit of ``vertex_table_relabellings``: the
    vertex table is invariant under them, so the members of an orbit run the
    same recursion on relabelled particles."""

    name_by_pdg = {int(pdg): name for name, pdg in PDGS.items()}
    relabellings = vertex_table_relabellings()
    order = {name: index for index, name in enumerate(sorted(PDGS))}
    seen: set[tuple[tuple[str, ...], ...]] = set()
    representatives = []
    for process in processes:
        sides = [side.split() for side in process.split(">")]
        key = min(
            tuple(
                tuple(
                    sorted(
                        (
                            name_by_pdg[mapping.get(int(PDGS[name]), int(PDGS[name]))]
                            for name in side
                        ),
                        key=order.__getitem__,
                    )
                )
                for side in sides
            )
            for mapping in relabellings
        )
        if key not in seen:
            seen.add(key)
            representatives.append(process)
    return tuple(representatives)


def on_shell_points(process: str, seeds: Sequence[int] = SEEDS) -> list:
    """RAMBO final states with the built-in masses: beams along z at
    ``SQRT_S`` or a decaying particle at rest."""

    model = BuiltinSMModel()
    ir = build_process_ir(process)
    initial = [model.mass(pdg) for pdg in ir.initial_pdgs]
    final_masses = [model.mass(pdg) for pdg in ir.final_pdgs]
    root = initial[0] if len(initial) == 1 else SQRT_S
    points = []
    for seed in seeds:
        final = massive_rambo_final_state(
            len(final_masses), sqrt_s=root, masses=final_masses, seed=seed
        )
        if len(initial) == 1:
            beams = [[root, 0.0, 0.0, 0.0]]
        else:
            m1, m2 = initial
            s = root * root
            p = math.sqrt((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)) / (2 * root)
            beams = [
                [math.hypot(p, m1), 0.0, 0.0, p],
                [math.hypot(p, m2), 0.0, 0.0, -p],
            ]
        points.append([*beams, *(list(momentum) for momentum in final)])
    return points


def split_entry(entry: str) -> tuple[str, dict[str, int] | None]:
    """A reference entry: the process and its explicit coupling orders."""

    process, _, orders = entry.partition("[")
    if not orders:
        return process.strip(), None
    return process.strip(), {
        name: int(value)
        for name, value in (item.split("=") for item in orders.rstrip("]").split())
    }


def run_config(execution_mode: str, orders: dict[str, int] | None = None):
    from pyamplicol.config import (
        ColorConfig,
        EvaluatorConfig,
        EvaluatorOptimizationConfig,
        GenerationConfig,
        GenerationRelationDiscoveryConfig,
        GenerationValidationConfig,
        ModelConfig,
        ProcessConfig,
        RunConfig,
    )

    return RunConfig(
        action="generate",
        model=ModelConfig(cache=False),
        color=ColorConfig(accuracy="full"),
        process=(
            ProcessConfig()
            if orders is None
            else ProcessConfig(
                coupling_order_policy="explicit", max_coupling_orders=orders
            )
        ),
        generation=GenerationConfig(
            workers=1,
            emit_api_bundle=False,
            relation_discovery=GenerationRelationDiscoveryConfig(mode="off"),
            validation=GenerationValidationConfig(
                enabled=False, samples=1, seed=7, post_build_validation=False
            ),
        ),
        evaluator=EvaluatorConfig(
            execution_mode=execution_mode,
            optimization=EvaluatorOptimizationConfig(cores=1),
        ),
    )


def evaluate(entry: str, model, execution_mode: str, artifact: Path):
    """|M|^2 of a reference entry at ``on_shell_points``, or ``None`` when
    generation is refused with a ``GenerationError`` (no tree amplitude)."""

    from pyamplicol import Generator, Runtime
    from pyamplicol.api.errors import GenerationError

    process, orders = split_entry(entry)
    try:
        Generator(run_config(execution_mode, orders)).generate(
            [process], artifact, model=model, mode="replace"
        )
    except GenerationError:
        return None
    runtime = Runtime.load(artifact, process=process)
    return [value.real for value in runtime.evaluate(on_shell_points(process))]


def reference_processes() -> tuple[str, ...]:
    """The default-tier entries: the 2 -> 2 class representatives, every
    1 -> 2 decay, ``SAMPLE`` and ``ORDERED``."""

    return (
        *class_representatives(two_to_two_processes()),
        *one_to_two_decays(),
        *SAMPLE,
        *ORDERED,
    )


def nightly_processes() -> tuple[str, ...]:
    """Every relabelling representative of the 2 -> 2 processes and decays,
    ``SAMPLE`` and ``ORDERED``."""

    return (
        *relabelling_representatives(two_to_two_processes()),
        *relabelling_representatives(one_to_two_decays()),
        *SAMPLE,
        *ORDERED,
    )


def _reference_worker(arguments: tuple[str, str, str]):
    from pyamplicol import ModelSource

    entry, bundle, artifact = arguments
    return entry, evaluate(
        entry, ModelSource.from_path(bundle), "compiled", Path(artifact)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=("reference",))
    parser.add_argument("--madgraph", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()

    from pyamplicol import ModelSource
    from pyamplicol.config import EvaluatorConfig

    args.work.mkdir(parents=True, exist_ok=True)
    bundle = args.work / "sm-no_masses.pyamplicol-model"
    if not bundle.exists():
        ModelSource.from_path(
            args.madgraph / "models/sm", restriction="no_masses"
        ).compile(
            prepared_output=bundle,
            cache_dir=args.work / "cache",
            evaluator=EvaluatorConfig(),
        )
    from concurrent.futures import ProcessPoolExecutor

    entries = reference_processes()
    jobs = [
        (entry, str(bundle), str(args.work / f"artifact-{index}"))
        for index, entry in enumerate(entries)
    ]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        values = dict(pool.map(_reference_worker, jobs))
    values = {entry: values[entry] for entry in entries}
    args.output.write_text(
        json.dumps(
            {
                "source": (
                    "MG5_aMC 3.7.0 models/sm, restriction no_masses, compiled "
                    "lane, binary64; tools/developer/builtin_sm_ufo_sweep.py"
                ),
                "sqrt_s": SQRT_S,
                "seeds": list(SEEDS),
                "processes": values,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
