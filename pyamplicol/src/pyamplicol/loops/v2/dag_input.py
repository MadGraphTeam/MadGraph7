# SPDX-License-Identifier: 0BSD
"""Native loop-request DAG builder: input tables, handle and stage digests.

WP3.1 package A (``RUST-MIGRATION.md`` §3.A, increments A1-A2). The Python
oracle is the tree engine's loop-request mode,
``GenericDAGCompiler(loop_blocks=True).compile`` followed by
``loops.v2.blocks._blocks``; ``rusticol_core::loop_v2::dag`` reproduces both.

**Input tables** (``dag_input``). The Rust builder never sees the model. It
reads Θ(model) tables that this module fills from the oracle's own
predicates, so every entry is what the Python recursion computes for the same
key: per particle the colour roles of ``ColorEngine`` (loop-request mode; for
a process without coloured legs also its colour sectors),
dimensions, admission, auxiliary kind and fermion-ordering contract; per
vertex its usability (ignored ids, ``ColorEngine.vertex_allowed``, duplicate
orientations), coupling orders, colour weight, absorbing sides, lowering
rule, evaluation equivalence and the quantum flows of every input chirality
pair (``allowed_quantum_flows``); ``vertices_accepting`` and
``_right_particles_by_left``; the order limits; the sources of every leg; and
the loop-aware reachability as its admitted leg sets plus the block
predicate. Floats travel as IEEE-754 bit patterns. Only compiled UFO models
are covered; anything else raises ``NativeDagUnsupported`` (rule 2).

**Columns** (``encode_dag`` / ``NativeDag.decode``). A DAG travels as one
byte container: ``b"PYAMDAG1"``, a little-endian u64 header length, a JSON
header (schema ``pyamplicol.loop-v2.dag-columns/1``: interned tables of
current species ``(pdg, auxiliary kind, quantum-number flow)``, flavour
flows, colour states, coupling orders and vertices ``(kind, particles,
backend, full-network flag)``, then the column list ``[name, dtype,
count]``) and the little-endian columns: per current ``c_species``,
``c_mask``, ``c_momentum``, ``c_ancestry``, ``c_chirality``, ``c_spin``,
``c_flavour``, ``c_colour``, ``c_orders``, ``c_dimension``, ``c_source``,
``c_leg`` (-1: none), ``c_helicity`` (``-2**31``: none), the words as
``c_word_offsets`` (n + 1) and ``c_words``; ``sources``; per interaction
``i_vertex``, ``i_left``, ``i_right``, ``i_result``, ``i_group`` (-1: none)
and the float pairs ``i_coupling``, ``i_colour_weight``, ``i_factor``.
``NativeDag.decode`` rebuilds the ``GenericDAG`` (its colour plan is the one
the compiler builds) and is counted (``decode_count``, rule 9).

**Stage digests** (rule 6). The DAG part is ``V2Dag.digest``: SHA-256 over
the schema tag ``pyamplicol.loop-v2.dag-digest/1\\0``, the current,
interaction and source counts (u64), then one flat record per current
(pdg, auxiliary kind, quantum numbers, mask, momentum mask, word, ancestry,
chirality, spin state, flavour flow, colour state, coupling orders,
dimension, source flag, leg, helicity), the source ids, and one record per
interaction (vertex kind and particles, inputs, result, coupling and colour
weight bits, backend, full-network flag, evaluation group, factor bits),
resolved through the tables so interning order does not enter.
``blocks_digest`` does the same for ``Block`` records (and the clone lines of
the CT program). S1' (loop-block DAG + blocks) and S1'' (CT DAG + blocks +
clone lines) are ``stage_digest`` of those parts; the Python side gets its
DAG digest by encoding its ``GenericDAG`` (``NativeDag.from_dag``).
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
from array import array
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from types import SimpleNamespace
from typing import Any

from pyamplicol.color.plan_types import GenericColorPlan
from pyamplicol.generation.dag_compiler import (
    AttachableLoopBlocks,
    LoopBlockReachability,
)
from pyamplicol.generation.dag_types import (
    ColorState,
    CurrentIndex,
    CurrentNode,
    GenericDAG,
    InteractionNode,
)
from pyamplicol.models.contracts import U1_SUBTRACTION_AUXILIARY_KIND
from pyamplicol.processes.ir import CanonicalProcessIR

from .blocks import (
    Block,
    BlockColourFlow,
    BlockKey,
    BlockRow,
    RequestedLoopBlocks,
)

# Rule 1: the one flag of the native loop-request DAG stage (A4). On, the
# loop-block DAG of ``blocks.build_tree_blocks`` and the counterterm program
# of ``ct_closure.CountertermBlocks.build`` are the native builder's (their
# Ward-kernel, coloured-width and U(1)-leakage queries native too); off, the
# tree engine's loop-request mode, which stays the oracle. Under the flag a
# request the native builder does not cover (``NativeDagUnsupported``)
# takes the tree engine, and ``TreeBlocks.dag_fallback`` names the feature
# (rule 2's listed fallbacks: ``NATIVE_FALLBACKS``).
NATIVE_DAG = True

# The features the native builder refuses (``NativeDagUnsupported``), for
# which ``build_tree_blocks`` and ``CountertermBlocks.build`` fall back to the
# tree engine under ``NATIVE_DAG`` (RUST-MIGRATION §2 rule 2): no control of
# the Python stage needs one; each is tested in
# ``tests/unit/test_loop_v2_native_dag.py``.
NATIVE_FALLBACKS = (
    "a model other than a compiled UFO model",
    "a leg without an outgoing particle id",
    "a tuple (multi-component) spin state of a vertex or a source",
    "a flavour flow outside the recurrence operations (constant, append "
    "left, append right), or one changing with the input chiralities",
    "a vertex setting quantum numbers other than its result particle's",
    "a NaN coupling or colour weight",
    "source colour states with basis keys",
    "more than 64 source states (helicity ancestry bits)",
    "a loop-block reachability other than the three of the slice",
)

INPUT_SCHEMA = "pyamplicol.loop-v2.dag-input/1"
BLOCK_INPUT_SCHEMA = "pyamplicol.loop-v2.dag-block-input/1"
COLUMNS_SCHEMA = "pyamplicol.loop-v2.dag-columns/1"
_MAGIC = b"PYAMDAG1"
_BLOCKS_DIGEST_SCHEMA = b"pyamplicol.loop-v2.blocks-digest/1\0"
_NO_LEG = -1
_NO_HELICITY = -(2**31)
_NO_GROUP = -1
_TYPECODES = {
    "u8": "B",
    "i8": "b",
    "i32": "i",
    "u32": "I",
    "i64": "q",
    "u64": "Q",
    "f64": "d",
}
_MODEL_ERRORS = (KeyError, NotImplementedError, TypeError, ValueError)


class NativeDagUnsupported(ValueError):
    """A feature the native DAG builder does not cover (rule 2: fail closed)."""


_DECODES = 0
_REBUILDS = 0


def decode_count() -> int:
    """How many native DAGs were decoded into Python objects (rule 9)."""

    return _DECODES


def rebuild_count() -> int:
    """How many released native DAGs were rebuilt from their tables."""

    return _REBUILDS


_NATIVE: list[Any] = []


def _native() -> Any:
    """The staged loop core, verified once per process (each
    ``load_loop_native`` re-verifies the native build, 0.2-0.4 s)."""

    if not _NATIVE:
        from pyamplicol.loops.native import load_loop_native

        _NATIVE.append(load_loop_native())
    return _NATIVE[0]


def _call(operation: Any, *args: Any) -> Any:
    try:
        return operation(*args)
    except ValueError as error:
        message = str(error)
        if message.startswith("unsupported: "):
            raise NativeDagUnsupported(message.removeprefix("unsupported: ")) from None
        raise


def _pdg(leg: Any) -> int:
    if leg.outgoing_pdg is None:
        raise NativeDagUnsupported("legs without an outgoing particle id")
    return int(leg.outgoing_pdg)


def _bits(value: float) -> int:
    return int(struct.unpack("<Q", struct.pack("<d", float(value)))[0])


def _pair_bits(pair: Sequence[float]) -> list[int]:
    return [_bits(pair[0]), _bits(pair[1])]


# ------------------------------------------------------------------ input tables


def _require_compiled_ufo(model: Any) -> None:
    from pyamplicol.models.external import CompiledUFOModel

    if not isinstance(model, CompiledUFOModel):
        raise NativeDagUnsupported(
            f"model type {type(model).__qualname__} (the native DAG tables cover "
            "compiled UFO models)"
        )


def _loop_engine(process: CanonicalProcessIR, model: Any) -> Any:
    """The oracle's loop-request colour engine.

    In loop-request mode with coloured legs ``ColorEngine`` reads no colour
    sector: its predicates depend on the legs and the model only, so it is
    built without Born sectors. Without coloured legs it orders labels and
    makes source states from the colour plan's sectors (one singlet sector),
    so it gets the plan ``GenericDAGCompiler.compile`` builds.
    """

    from pyamplicol.color.plan import build_color_plan
    from pyamplicol.generation.dag_color import ColorEngine

    plan = GenericColorPlan(process=process, color_accuracy="full", sectors=())
    engine = ColorEngine(plan, model, loop_blocks=True)
    if engine._shared_lc_coloured_labels:
        return engine
    plan = build_color_plan(
        process,
        color_accuracy="full",
        max_sectors=None,
        reference_color_order=None,
        fold_trace_reflections=False,
    )
    return ColorEngine(plan, model, loop_blocks=True)


def _sector_rows(process: CanonicalProcessIR, engine: Any) -> list[dict[str, Any]]:
    """The colour sectors ``ColorEngine.ordered_combination_labels`` reads in
    loop-request mode, i.e. for processes without coloured legs: per sector
    its intermediate words and singlet labels."""

    from pyamplicol.generation.dag_ordering import _sector_intermediate_order_words

    if engine._shared_lc_coloured_labels:
        return []
    return [
        {
            "id": int(sector.id),
            "words": [
                [int(label) for label in word]
                for word in _sector_intermediate_order_words(
                    sector, include_compatible_traversals=False, process=process
                )
            ],
            "singlet_labels": sorted(int(label) for label in sector.singlet_labels),
        }
        for sector in engine.color_plan.sectors
    ]


def _known(operation: Any, *args: Any) -> Any:
    try:
        return operation(*args)
    except _MODEL_ERRORS:
        return None


def _particle_row(model: Any, engine: Any, pdg: int) -> dict[str, Any]:
    from pyamplicol.generation.dag_ordering import _known_color_representation
    from pyamplicol.generation.fermion_ordering import (
        _fermion_rep_requires_exterior_sign,
    )

    contract: list[Any] | None = None
    contract_error: str | None = None
    try:
        fermion = bool(model.is_fermion(pdg))
        orientation = (
            str(model.source_orientation(pdg)) if fermion else "self-conjugate"
        )
        owned = (
            _fermion_rep_requires_exterior_sign(model.color_rep(pdg), False)
            if fermion
            else False
        )
        contract = [fermion, orientation, owned]
    except _MODEL_ERRORS as error:
        contract_error = str(error)
    return {
        "pdg": pdg,
        "rep": _known_color_representation(model, pdg),
        "has_colour": bool(engine._particle_has_colour(pdg)),
        "fla": pdg in engine._fundamental_line_auxiliary_ids,
        "loop_rep": engine._loop_representation(pdg),
        "physical_adjoint": bool(engine._is_physical_adjoint(pdg)),
        "u1_singlet": bool(engine._is_u1_projection_singlet(pdg)),
        "auxiliary": model.auxiliary_kind(pdg),
        "quantum_numbers": [list(pair) for pair in model.quantum_number_flow(pdg)],
        "dimension": [int(model.current_dimension(pdg, c)) for c in (-1, 0, 1)],
        "allowed": [
            bool(model.current_allowed(SimpleNamespace(particle_id=pdg, chirality=c)))
            for c in (-1, 0, 1)
        ],
        "sign_contract": contract,
        "sign_contract_error": contract_error,
        "block_rep": _known(model.color_rep, pdg),
        "block_fermion": _known(model.is_fermion, pdg),
    }


def _flavour_operation(model: Any, vertex: Any, left: int, right: int) -> str | None:
    """The flavour-flow rule of ``allowed_quantum_flows`` for this vertex,
    read off two probes with sentinel flows (``None``: no flow)."""

    result = vertex.particles[2]

    def probe(left_flow: tuple[int, ...], right_flow: tuple[int, ...]) -> set[Any]:
        flows = model.allowed_quantum_flows(
            vertex,
            SimpleNamespace(
                particle_id=vertex.particles[0],
                chirality=left,
                flavour_flow=left_flow,
                coupling_orders=(),
            ),
            SimpleNamespace(
                particle_id=vertex.particles[1],
                chirality=right,
                flavour_flow=right_flow,
                coupling_orders=(),
            ),
        )
        return {flow.flavour_flow for flow in flows}

    first = probe((-101, -102), (-103,))
    if not first:
        return None
    rules = {
        "constant": ({(result,)}, {(result,)}),
        "left": ({(-101, -102, result)}, {(-101, result)}),
        "right": ({(-103, result)}, {(-103, result)}),
    }
    second = probe((-101, result), (-103, result))
    for name, (expected_first, expected_second) in rules.items():
        if first == expected_first and second == expected_second:
            return name
    raise NativeDagUnsupported(
        f"the flavour flow of vertex {vertex.kind} is not one of the recurrence "
        "operations (constant, append left, append right)"
    )


def _vertex_usable(
    model: Any,
    engine: Any,
    vertex: Any,
    ignored_particles: frozenset[int],
    ignored_kinds: frozenset[int],
) -> bool:
    return (
        vertex.kind not in ignored_kinds
        and vertex.particles[2] not in ignored_particles
        and bool(engine.vertex_allowed(vertex))
        and not model.skip_duplicate_vertex_orientation(vertex)
    )


def _vertex_row(
    model: Any,
    engine: Any,
    vertex: Any,
    *,
    usable: bool,
    coupling_identities: dict[Any, int],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "kind": int(vertex.kind),
        "particles": [int(p) for p in vertex.particles],
        "usable": usable,
        "orders": [],
        "colour_weight": None,
        "colour_weight_error": None,
        "absorbing": None,
        "backend": "",
        "full_tensor_network_ready": False,
        "class": "",
        "factor": _pair_bits((1.0, 0.0)),
        "swap": False,
        "exchange": None,
        "flows": [],
        "flavour": "constant",
    }
    if not usable:
        return row
    absorbing = engine._loop_block_absorbing_sides(vertex)
    found = _model_vertex_row(model, vertex)
    row.update(found.fields)
    row["absorbing"] = None if absorbing is None else [bool(side) for side in absorbing]
    row["flows"] = [
        [
            left,
            right,
            [
                [
                    *entry,
                    coupling_identities.setdefault(identity, len(coupling_identities)),
                ]
                for entry, identity in entries
            ],
        ]
        for left, right, entries in found.flows
    ]
    return row


@dataclass(frozen=True)
class _ModelVertexRow:
    """The model-only part of a usable vertex row (``_vertex_row``): no
    colour engine and no process enters it. ``flows`` hold, per input
    chirality pair, ``(chirality, spin state, coupling bits)`` with the
    runtime coupling identity each entry is interned by."""

    fields: Mapping[str, Any]
    flows: tuple[tuple[int, int, tuple[tuple[tuple[int, ...], Any], ...]], ...]


# The model-only vertex rows of the most recent models (the Θ(model) input
# tables are per model, not per process): ``(id(compiled), runtime
# parameters) -> (compiled, rows)``, the compiled model kept so that its id
# is not reused while the entry lives. A ``CompiledUFOModel`` is its compiled
# model and runtime parameters. Most of a row is ``allowed_quantum_flows``'
# symbolic Weyl-projection probe, which a model instance caches only for
# itself (a lowered model's instance, ``CountertermLoweredModel.tree_model``,
# comes with them decided).
_MODEL_ROWS: dict[
    tuple[int, str], tuple[Any, dict[tuple[int, tuple[int, ...]], Any]]
] = {}
_MODEL_ROWS_KEPT = 4


def _model_rows(model: Any) -> dict[tuple[int, tuple[int, ...]], Any]:
    compiled = getattr(model, "compiled", None)
    if compiled is None:
        return {}
    key = (
        id(compiled),
        repr(sorted(getattr(model, "_runtime_parameters", {}).items())),
    )
    found = _MODEL_ROWS.pop(key, None)
    if found is None or found[0] is not compiled:
        found = (compiled, {})
    # Most recently used last; the oldest entries leave first.
    _MODEL_ROWS[key] = found
    while len(_MODEL_ROWS) > _MODEL_ROWS_KEPT:
        del _MODEL_ROWS[next(iter(_MODEL_ROWS))]
    return found[1]


def _model_vertex_row(model: Any, vertex: Any) -> _ModelVertexRow:
    rows = _model_rows(model)
    key = (int(vertex.kind), tuple(int(p) for p in vertex.particles))
    found = rows.get(key)
    if found is None:
        found = rows[key] = _compute_model_vertex_row(model, vertex)
    return found


def _compute_model_vertex_row(model: Any, vertex: Any) -> _ModelVertexRow:
    from pyamplicol.generation.dag_equivalence import (
        _kernel_equivalence,
        _runtime_coupling_identity,
    )

    fields: dict[str, Any] = {
        "colour_weight": None,
        "colour_weight_error": None,
    }
    try:
        fields["colour_weight"] = _pair_bits(
            model.vertex_color_weight(vertex, color_accuracy="full")
        )
    except _MODEL_ERRORS as error:
        fields["colour_weight_error"] = str(error)
    rule = model.vertex_lowering_rule(vertex.kind)
    equivalence = _kernel_equivalence(model, vertex.kind, {})
    fields.update(
        orders=[
            [str(name).upper(), int(value)]
            for name, value in model.vertex_coupling_orders(vertex)
        ],
        backend=rule.backend,
        full_tensor_network_ready=bool(rule.full_tensor_network_ready),
        factor=_pair_bits(equivalence.factor),
        swap=tuple(equivalence.input_order) == (1, 0),
        exchange=(
            None
            if equivalence.input_exchange_factor is None
            else _pair_bits(equivalence.input_exchange_factor)
        ),
        **{"class": equivalence.class_id},
    )

    def chiralities(pdg: int) -> tuple[int, ...]:
        return (-1, 0, 1) if model.is_chiral_eligible(pdg) else (0,)

    result = vertex.particles[2]
    quantum_numbers = model.quantum_number_flow(result)
    operations = set()
    flows = []
    for left in chiralities(vertex.particles[0]):
        for right in chiralities(vertex.particles[1]):
            stub = SimpleNamespace(
                particle_id=vertex.particles[0],
                chirality=left,
                flavour_flow=(vertex.particles[0],),
                coupling_orders=(),
            )
            other = SimpleNamespace(
                particle_id=vertex.particles[1],
                chirality=right,
                flavour_flow=(vertex.particles[1],),
                coupling_orders=(),
            )
            entries = []
            for flow in model.allowed_quantum_flows(vertex, stub, other):
                if not isinstance(flow.spin_state, int):
                    raise NativeDagUnsupported(
                        f"tuple spin state {flow.spin_state!r} of vertex {vertex.kind}"
                    )
                if flow.quantum_number_flow != quantum_numbers:
                    raise NativeDagUnsupported(
                        f"vertex {vertex.kind} sets quantum numbers other than its "
                        "result particle's"
                    )
                if any(math.isnan(part) for part in flow.coupling):
                    raise NativeDagUnsupported(
                        f"a NaN coupling of vertex {vertex.kind} (its runtime "
                        "coupling identity has no order)"
                    )
                identity = _runtime_coupling_identity(
                    model,
                    vertex_kind=vertex.kind,
                    vertex_particles=vertex.particles,
                    coupling=flow.coupling,
                )
                entries.append(
                    (
                        (
                            int(flow.chirality),
                            int(flow.spin_state),
                            *_pair_bits(flow.coupling),
                        ),
                        identity,
                    )
                )
            flows.append((left, right, tuple(entries)))
            operation = _flavour_operation(model, vertex, left, right)
            if operation is not None:
                operations.add(operation)
    if len(operations) > 1:
        raise NativeDagUnsupported(
            f"vertex {vertex.kind} changes its flavour rule with the input chiralities"
        )
    fields["flavour"] = operations.pop() if operations else "constant"
    return _ModelVertexRow(fields, tuple(flows))


def _order_bounded(
    orders: Iterable[tuple[str, int]], limits: Mapping[str, int]
) -> bool:
    """Whether a vertex of these coupling orders can enter a current within
    ``limits`` (normalised as the recursion's): orders only add up along a
    tree, so one vertex beyond a limit puts every current holding it beyond
    too (a vertex with a negative order bounds nothing)."""

    values = {str(name).upper(): int(value) for name, value in orders}
    if any(value < 0 for value in values.values()):
        return True
    return all(values.get(name, 0) <= limit for name, limit in limits.items())


def _reachable_vertices(
    process: CanonicalProcessIR,
    model: Any,
    engine: Any,
    rights: Mapping[int, Sequence[int]],
    limits: Mapping[str, int],
    ignored_particles: frozenset[int],
    ignored_kinds: frozenset[int],
) -> tuple[list[tuple[int, int, list[tuple[Any, bool]]]], dict[int, tuple[int, ...]]]:
    """``vertices_accepting`` restricted to the species the recursion can
    reach, and ``_right_particles_by_left`` restricted to them.

    A species is reachable when it is a leg's or the result of a usable
    vertex within the order limits whose inputs are reachable (a fixed
    point over species: a superset of every current's particle, whatever
    its leg set). The native recursion looks up only pairs of its
    currents' particles (a missing pair accepts nothing), so the tables of
    the other species are never read; the oracle pays their model probes
    only for the vertices it reaches, and so does this table.
    """

    usable: dict[tuple[int, tuple[int, ...]], bool] = {}

    def admitted(vertex: Any) -> bool:
        key = (int(vertex.kind), tuple(int(p) for p in vertex.particles))
        found = usable.get(key)
        if found is None:
            found = usable[key] = _vertex_usable(
                model, engine, vertex, ignored_particles, ignored_kinds
            )
        return found

    reached = {_pdg(leg) for leg in process.legs}
    accepted: dict[tuple[int, int], tuple[Any, ...]] = {}
    grown = True
    while grown:
        grown = False
        for left in sorted(reached & set(rights)):
            for right in rights[left]:
                if right not in reached or (left, right) in accepted:
                    continue
                vertices = tuple(
                    model.vertices_accepting(left, right, color_accuracy="full")
                )
                accepted[(left, right)] = vertices
                for vertex in vertices:
                    if (
                        admitted(vertex)
                        and _order_bounded(model.vertex_coupling_orders(vertex), limits)
                        and int(vertex.particles[2]) not in reached
                    ):
                        reached.add(int(vertex.particles[2]))
                        grown = True
    rows = []
    for left in sorted(rights):
        if left not in reached:
            continue
        for right in rights[left]:
            if right not in reached:
                continue
            rows.append(
                (
                    int(left),
                    int(right),
                    [
                        (vertex, admitted(vertex))
                        for vertex in accepted[(left, right)]
                        if _order_bounded(model.vertex_coupling_orders(vertex), limits)
                    ],
                )
            )
    restricted = {
        int(left): tuple(int(r) for r in rights[left] if r in reached)
        for left in sorted(rights)
        if left in reached
    }
    return rows, restricted


def _submasks(full: int) -> list[int]:
    masks = []
    mask = full
    while mask:
        masks.append(mask)
        mask = (mask - 1) & full
    return sorted(masks)


def _reachability_tables(
    process: CanonicalProcessIR,
    model: Any,
    reachability: LoopBlockReachability | None,
    *,
    ignored_particle_ids: Iterable[int],
    ignored_vertex_kinds: Iterable[int],
) -> tuple[list[int], dict[str, Any]]:
    if reachability is None:
        reachability = AttachableLoopBlocks.for_process(
            process,
            model,
            ignored_particle_ids=ignored_particle_ids,
            ignored_vertex_kinds=ignored_vertex_kinds,
        )
    full = 0
    for leg in process.legs:
        full |= 1 << (leg.label - 1)
    masks = [mask for mask in _submasks(full) if reachability.admits_mask(mask)]
    rule: dict[str, Any]
    if isinstance(reachability, AttachableLoopBlocks):
        rule = {
            "kind": "particles",
            "particles": sorted(reachability.attachable_particle_ids),
        }
    elif isinstance(reachability, RequestedLoopBlocks):
        rule = {
            "kind": "requests",
            "requests": sorted(
                [int(mask), int(pdg), [[str(n), int(v)] for n, v in orders]]
                for mask, pdg, orders in reachability.requests
            ),
        }
    elif (
        type(reachability).__module__ == "pyamplicol.loops.v2.ct_closure"
        and type(reachability).__qualname__ == "RootReachability"
    ):
        rule = {"kind": "mask"}
    else:
        raise NativeDagUnsupported(
            f"loop-block reachability {type(reachability).__qualname__}"
        )
    return masks, rule


def dag_input(
    process: CanonicalProcessIR,
    model: Any,
    *,
    max_coupling_orders: Mapping[str, int] | None = None,
    ignored_particle_ids: Iterable[int] = (),
    ignored_vertex_kinds: Iterable[int] = (),
    reachability: LoopBlockReachability | None = None,
) -> str:
    """The JSON input tables of ``V2Dag.build`` for the loop-request DAG of
    ``GenericDAGCompiler(model=model, loop_blocks=True, ...).compile(process)``."""

    from pyamplicol.generation.dag_ordering import _known_color_representation
    from pyamplicol.generation.dag_reachability import (
        _normalize_coupling_order_limits,
        _right_particles_by_left,
    )

    _require_compiled_ufo(model)
    if process.color_accuracy != "full":
        raise ValueError(
            "loop-request blocks are exact colour-flow currents; compile the "
            "full-colour process"
        )
    if any(leg.outgoing_pdg is None for leg in process.legs):
        raise NativeDagUnsupported("legs without an outgoing particle id")
    ignored_particles = frozenset(int(pdg) for pdg in ignored_particle_ids)
    ignored_kinds = frozenset(int(kind) for kind in ignored_vertex_kinds)
    engine = _loop_engine(process, model)
    limits = _normalize_coupling_order_limits(max_coupling_orders)
    accepted, rights = _reachable_vertices(
        process,
        model,
        engine,
        _right_particles_by_left(model, color_accuracy="full"),
        limits,
        ignored_particles,
        ignored_kinds,
    )
    vertex_rows: dict[tuple[int, tuple[int, ...]], int] = {}
    vertices: list[tuple[Any, bool]] = []
    accepting = []
    for left, right, found in accepted:
        rows = []
        for vertex, usable in found:
            key = (int(vertex.kind), tuple(int(p) for p in vertex.particles))
            if key not in vertex_rows:
                vertex_rows[key] = len(vertices)
                vertices.append((vertex, usable))
            rows.append(vertex_rows[key])
        accepting.append([left, right, rows])
    pdgs = sorted(
        {int(pdg) for pdg in model.particles}
        | {int(p) for vertex, _usable in vertices for p in vertex.particles}
        | {_pdg(leg) for leg in process.legs}
    )
    coupling_identities: dict[Any, int] = {}
    masks, rule = _reachability_tables(
        process,
        model,
        reachability,
        ignored_particle_ids=ignored_particles,
        ignored_vertex_kinds=ignored_kinds,
    )
    colours: list[list[Any]] = []
    sources = []
    for leg in process.legs:
        pdg = _pdg(leg)
        source = model._source_ir(pdg)
        states = []
        for declared in source.states:
            state = source.crossing.apply(declared) if leg.is_initial else declared
            if not isinstance(state.spin_state, int):
                raise NativeDagUnsupported(
                    f"tuple source spin state {state.spin_state!r} of particle {pdg}"
                )
            states.append(
                [int(state.chirality), int(state.helicity), int(state.spin_state)]
            )
        ids = []
        for colour in engine.source_states_for_leg(leg):
            if colour.basis_key or colour.traces or colour.lines:
                raise NativeDagUnsupported("source colour states with basis keys")
            ids.append(len(colours))
            colours.append([int(colour.sector_id), list(colour.line_groups)])
        sources.append([int(leg.label), states, ids])
    payload = {
        "schema": INPUT_SCHEMA,
        "accuracy": "full",
        "legs": [
            {
                "label": int(leg.label),
                "pdg": _pdg(leg),
                "initial": bool(leg.is_initial),
                "statistics": str(leg.statistics),
                "colour_role": str(leg.color_role),
                "rep": _known_color_representation(model, _pdg(leg)),
            }
            for leg in process.legs
        ],
        "particles": [_particle_row(model, engine, pdg) for pdg in pdgs],
        "vertices": [
            _vertex_row(
                model,
                engine,
                vertex,
                usable=usable,
                coupling_identities=coupling_identities,
            )
            for vertex, usable in vertices
        ],
        "accepting": accepting,
        "right_particles": [[left, list(rights[left])] for left in sorted(rights)],
        "order_limits": sorted([name, int(value)] for name, value in limits.items()),
        "ignored_particles": sorted(ignored_particles),
        "sources": sources,
        "colours": colours,
        "admitted_masks": masks,
        "block_rule": rule,
        "early_sources_only": any(
            _pdg(leg) in ignored_particles for leg in process.legs
        ),
        "coupling_identity_ranks": _ranks(coupling_identities),
        "sectors": _sector_rows(process, engine),
    }
    return json.dumps(payload, separators=(",", ":"))


def _ranks(ids: Mapping[Any, int]) -> list[int]:
    """Rank of every interned key (by id) in Python's sort order of the keys
    (the order of the reuse derivation's evaluation keys)."""

    ranks = [0] * len(ids)
    for rank, key in enumerate(sorted(ids)):
        ranks[ids[key]] = rank
    return ranks


def block_input(
    process: CanonicalProcessIR,
    model: Any,
    clone_fields: Mapping[int, int] | None = None,
) -> str:
    """The JSON tables of ``V2Dag.blocks`` / ``coloured_propagated``;
    ``clone_fields`` maps the clone particles of a CT program to the fields
    their lines are read as (``ct_closure._clone_lines``)."""

    label_reps = []
    endpoints = []
    for leg in process.legs:
        if leg.outgoing_pdg is None:
            continue
        pdg = _pdg(leg)
        rep = _known(model.color_rep, pdg)
        label_reps.append([int(leg.label), None if rep is None else int(rep)])
        if rep is not None and abs(int(rep)) == 3 and model.is_fermion(pdg):
            endpoints.append(int(leg.label))
    particles = []
    for pdg in sorted(int(p) for p in model.particles):
        rep = _known(model.color_rep, pdg)
        fermion = _known(model.is_fermion, pdg)
        particles.append(
            [
                pdg,
                None if rep is None else int(rep),
                None if fermion is None else bool(fermion),
            ]
        )
    return json.dumps(
        {
            "schema": BLOCK_INPUT_SCHEMA,
            "label_reps": label_reps,
            "particles": particles,
            "endpoints": endpoints,
            "u1_subtraction_kind": U1_SUBTRACTION_AUXILIARY_KIND,
            "clones": sorted(
                [int(clone), int(field)]
                for clone, field in (clone_fields or {}).items()
            ),
        },
        separators=(",", ":"),
    )


# ------------------------------------------------------------------ columns


def _column(values: Iterable[Any], dtype: str) -> array[Any]:
    column = array(_TYPECODES[dtype], values)
    if sys.byteorder != "little":
        column.byteswap()
    return column


def encode_dag(dag: GenericDAG) -> bytes:
    """``dag`` as a columnar container (the oracle side of the digests);
    ``NativeDagUnsupported`` for a field outside its column type (masks and
    the helicity ancestry are u64)."""

    species: dict[Any, int] = {}
    flavours: dict[tuple[int, ...], int] = {}
    colours: dict[ColorState, int] = {}
    orders: dict[tuple[tuple[str, int], ...], int] = {}
    vertices: dict[tuple[Any, ...], int] = {}

    def intern(table: dict[Any, int], key: Any) -> int:
        found = table.get(key)
        if found is None:
            found = table[key] = len(table)
        return found

    columns: dict[str, list[Any]] = {
        name: []
        for name in (
            "c_species",
            "c_mask",
            "c_momentum",
            "c_ancestry",
            "c_chirality",
            "c_spin",
            "c_flavour",
            "c_colour",
            "c_orders",
            "c_dimension",
            "c_source",
            "c_leg",
            "c_helicity",
            "c_words",
        )
    }
    offsets = [0]
    for position, current in enumerate(dag.currents):
        index = current.index
        if current.id != position:
            raise ValueError("DAG current ids are not their positions")
        if not isinstance(index.spin_state, int):
            raise NativeDagUnsupported(f"tuple spin state {index.spin_state!r}")
        columns["c_species"].append(
            intern(
                species,
                (
                    int(index.particle_id),
                    index.auxiliary_kind,
                    tuple(tuple(pair) for pair in index.quantum_number_flow),
                ),
            )
        )
        columns["c_mask"].append(index.external_mask)
        columns["c_momentum"].append(index.momentum_mask)
        columns["c_ancestry"].append(index.helicity_ancestry)
        columns["c_chirality"].append(index.chirality)
        columns["c_spin"].append(index.spin_state)
        columns["c_flavour"].append(intern(flavours, tuple(index.flavour_flow)))
        columns["c_colour"].append(intern(colours, index.color_state))
        columns["c_orders"].append(intern(orders, tuple(index.coupling_orders)))
        columns["c_dimension"].append(current.dimension)
        columns["c_source"].append(int(current.is_source))
        columns["c_leg"].append(
            _NO_LEG if current.source_leg_label is None else current.source_leg_label
        )
        columns["c_helicity"].append(
            _NO_HELICITY if current.source_helicity is None else current.source_helicity
        )
        columns["c_words"].extend(index.ordered_external_labels)
        offsets.append(len(columns["c_words"]))
    interaction_columns: dict[str, list[Any]] = {
        "i_vertex": [],
        "i_left": [],
        "i_right": [],
        "i_result": [],
        "i_group": [],
        "i_coupling": [],
        "i_colour_weight": [],
        "i_factor": [],
    }
    for position, interaction in enumerate(dag.interactions):
        if interaction.id != position:
            raise ValueError("DAG interaction ids are not their positions")
        interaction_columns["i_vertex"].append(
            intern(
                vertices,
                (
                    int(interaction.vertex_kind),
                    tuple(int(p) for p in interaction.vertex_particles),
                    interaction.lowering_backend,
                    bool(interaction.full_tensor_network_ready),
                ),
            )
        )
        interaction_columns["i_left"].append(interaction.left_id)
        interaction_columns["i_right"].append(interaction.right_id)
        interaction_columns["i_result"].append(interaction.result_id)
        interaction_columns["i_group"].append(
            _NO_GROUP
            if interaction.evaluation_group_id is None
            else interaction.evaluation_group_id
        )
        interaction_columns["i_coupling"].extend(interaction.coupling)
        interaction_columns["i_colour_weight"].extend(interaction.color_weight)
        interaction_columns["i_factor"].extend(interaction.evaluation_factor)
    dtypes = {
        "c_species": "u32",
        "c_mask": "u64",
        "c_momentum": "u64",
        "c_ancestry": "u64",
        "c_chirality": "i8",
        "c_spin": "i32",
        "c_flavour": "u32",
        "c_colour": "u32",
        "c_orders": "u32",
        "c_dimension": "i32",
        "c_source": "u8",
        "c_leg": "i32",
        "c_helicity": "i32",
        "c_word_offsets": "u32",
        "c_words": "u8",
        "sources": "u32",
        "i_vertex": "u32",
        "i_left": "u32",
        "i_right": "u32",
        "i_result": "u32",
        "i_group": "i64",
        "i_coupling": "f64",
        "i_colour_weight": "f64",
        "i_factor": "f64",
    }
    values = {
        **columns,
        "c_word_offsets": offsets,
        "sources": list(dag.sources),
        **interaction_columns,
    }
    header_columns = []
    payload = bytearray()
    for name, dtype in dtypes.items():
        try:
            column = _column(values[name], dtype)
        except OverflowError as error:
            # E.g. the helicity ancestry of a full-colour tree DAG wider than
            # 64 bits: the container (and the native DAG) holds u64 masks.
            raise NativeDagUnsupported(
                f"{name} values outside the native {dtype} column"
            ) from error
        header_columns.append([name, dtype, len(column)])
        payload += column.tobytes()
    header = {
        "schema": COLUMNS_SCHEMA,
        "tables": {
            "species": [
                {
                    "pdg": pdg,
                    "auxiliary": auxiliary,
                    "quantum_numbers": [list(pair) for pair in quantum_numbers],
                }
                for pdg, auxiliary, quantum_numbers in species
            ],
            "flavours": [list(flow) for flow in flavours],
            "colours": [
                {
                    "accuracy": state.accuracy,
                    "sector_id": state.sector_id,
                    "line_groups": list(state.line_groups),
                    "basis_key": list(state.basis_key),
                    "traces": [list(trace) for trace in state.traces],
                    "lines": [list(line) for line in state.lines],
                }
                for state in colours
            ],
            "orders": [[list(pair) for pair in row] for row in orders],
            "vertices": [
                {
                    "kind": kind,
                    "particles": list(particles),
                    "backend": backend,
                    "full_tensor_network_ready": ready,
                }
                for kind, particles, backend, ready in vertices
            ],
        },
        "columns": header_columns,
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return _MAGIC + struct.pack("<Q", len(encoded)) + encoded + bytes(payload)


def _read_container(data: bytes) -> tuple[dict[str, Any], dict[str, array[Any]]]:
    if data[:8] != _MAGIC:
        raise ValueError("not a native DAG container")
    (length,) = struct.unpack_from("<Q", data, 8)
    header = json.loads(data[16 : 16 + length])
    if header["schema"] != COLUMNS_SCHEMA:
        raise ValueError(f"native DAG container schema {header['schema']!r}")
    view = memoryview(data)[16 + length :]
    columns = {}
    for name, dtype, count in header["columns"]:
        column = array(_TYPECODES[dtype])
        size = column.itemsize * count
        column.frombytes(view[:size])
        if sys.byteorder != "little":
            column.byteswap()
        columns[name] = column
        view = view[size:]
    return header["tables"], columns


def _labels(mask: int) -> tuple[int, ...]:
    return tuple(bit + 1 for bit in range(mask.bit_length()) if mask >> bit & 1)


def _colour_state(row: Mapping[str, Any]) -> ColorState:
    return ColorState(
        accuracy=row["accuracy"],
        sector_id=row["sector_id"],
        line_groups=tuple(row["line_groups"]),
        basis_key=tuple(row["basis_key"]),
        traces=tuple(tuple(trace) for trace in row["traces"]),
        lines=tuple(tuple(line) for line in row["lines"]),
    )


def _shared_pairs(column: array[Any]) -> list[tuple[float, float]]:
    """The ``(re, im)`` pairs of an ``f64`` column, equal bit patterns as one
    tuple (the compiled DAG shares a vertex's coupling over its
    interactions; signed zeros and NaN payloads stay distinct)."""

    bits: array[int] = array("Q")
    bits.frombytes(column.tobytes())
    if sys.byteorder != "little":
        bits.byteswap()
    table: dict[tuple[int, int], tuple[float, float]] = {}
    out = []
    for k in range(0, len(column) - 1, 2):
        key = (bits[k], bits[k + 1])
        found = table.get(key)
        if found is None:
            found = table[key] = (column[k], column[k + 1])
        out.append(found)
    return out


def _decode(
    data: bytes, process: CanonicalProcessIR, color_plan: GenericColorPlan
) -> GenericDAG:
    tables, c = _read_container(data)
    species = [
        (
            row["pdg"],
            row["auxiliary"],
            tuple(tuple(pair) for pair in row["quantum_numbers"]),
        )
        for row in tables["species"]
    ]
    flavours = [tuple(flow) for flow in tables["flavours"]]
    colours = [_colour_state(row) for row in tables["colours"]]
    orders = [tuple(tuple(pair) for pair in row) for row in tables["orders"]]
    vertices = [
        (
            row["kind"],
            tuple(row["particles"]),
            row["backend"],
            row["full_tensor_network_ready"],
        )
        for row in tables["vertices"]
    ]
    offsets = c["c_word_offsets"]
    words = c["c_words"]
    # Equal values are one object, as in the compiled DAG (whose ids,
    # labels and couplings are shared by the nodes that read them): the
    # current ids, the label tuples of a leg set and the words.
    ids = list(range(max(len(c["c_species"]), len(c["i_vertex"]))))
    labels: dict[int, tuple[int, ...]] = {}
    masks: dict[int, int] = {}
    shared_words: dict[tuple[int, ...], tuple[int, ...]] = {}

    def mask_of(value: int) -> int:
        return masks.setdefault(value, value)

    currents = []
    for i in range(len(c["c_species"])):
        pdg, auxiliary, quantum_numbers = species[c["c_species"][i]]
        mask = mask_of(c["c_mask"][i])
        found = labels.get(mask)
        if found is None:
            found = labels[mask] = _labels(mask)
        word = tuple(words[offsets[i] : offsets[i + 1]])
        leg = c["c_leg"][i]
        helicity = c["c_helicity"][i]
        currents.append(
            CurrentNode(
                id=ids[i],
                index=CurrentIndex(
                    particle_id=pdg,
                    external_mask=mask,
                    external_labels=found,
                    ordered_external_labels=shared_words.setdefault(word, word),
                    helicity_ancestry=c["c_ancestry"][i],
                    chirality=c["c_chirality"][i],
                    spin_state=c["c_spin"][i],
                    flavour_flow=flavours[c["c_flavour"][i]],
                    quantum_number_flow=quantum_numbers,
                    color_state=colours[c["c_colour"][i]],
                    momentum_mask=mask_of(c["c_momentum"][i]),
                    coupling_orders=orders[c["c_orders"][i]],
                    auxiliary_kind=auxiliary,
                ),
                dimension=c["c_dimension"][i],
                is_source=bool(c["c_source"][i]),
                source_leg_label=None if leg == _NO_LEG else leg,
                source_helicity=None if helicity == _NO_HELICITY else helicity,
            )
        )
    couplings = _shared_pairs(c["i_coupling"])
    colour_weights = _shared_pairs(c["i_colour_weight"])
    factors = _shared_pairs(c["i_factor"])
    interactions = []
    for k in range(len(c["i_vertex"])):
        kind, particles, backend, ready = vertices[c["i_vertex"][k]]
        group = c["i_group"][k]
        interactions.append(
            InteractionNode(
                id=ids[k],
                vertex_kind=kind,
                vertex_particles=particles,
                left_id=ids[c["i_left"][k]],
                right_id=ids[c["i_right"][k]],
                result_id=ids[c["i_result"][k]],
                coupling=couplings[k],
                color_weight=colour_weights[k],
                lowering_backend=backend,
                full_tensor_network_ready=ready,
                evaluation_group_id=(
                    None
                    if group == _NO_GROUP
                    else ids[group]
                    if group < len(ids)
                    else group
                ),
                evaluation_factor=factors[k],
            )
        )
    return GenericDAG(
        process=process,
        color_plan=color_plan,
        currents=tuple(currents),
        sources=tuple(ids[source] for source in c["sources"]),
        interactions=tuple(interactions),
        amplitude_roots=(),
        helicity_coverage="complete",
        color_coverage="selected" if color_plan.truncated else "complete",
    )


# ------------------------------------------------------------------ handle


# The ``GenericDAG`` fields a materialized ``NativeDag`` holds itself.
_DAG_FIELDS = tuple(field.name for field in fields(GenericDAG))


class NativeDag:
    """A native columnar loop-request DAG and its queries.

    ``decode`` builds the ``GenericDAG`` of Python objects (oracles, the
    per-point Python path) and is counted (``decode_count``). As the
    ``blocks_dag`` of ``blocks.TreeBlocks`` the handle stands in for that
    ``GenericDAG``: reading any other of its attributes (``currents``,
    ``interactions``, ``sources``, ``color_plan``, ...) decodes it once
    (``materialize``) and reads it there, so consumers that need Python
    objects keep working while generation decodes nothing (rule 9).

    A DAG the builder made keeps its input tables (``build_input``):
    ``release`` then drops the native columns, and a later query rebuilds
    them from the tables (deterministic: the same columns,
    ``rebuild_count``), so generation does not hold them to its end. The
    queries generation asks (blocks, propagated species, U(1) leakage) are
    answered once and kept.
    """

    def __init__(
        self,
        handle: Any,
        process: CanonicalProcessIR,
        model: Any,
        clone_fields: Mapping[int, int] | None = None,
        *,
        build_input: str | None = None,
    ) -> None:
        self._handle = handle
        self.process = process
        self._model = model
        self._clone_fields = dict(clone_fields or {})
        self._block_input: str | None = None
        self._decoded: GenericDAG | None = None
        self._build_input = build_input
        self._blocks: tuple[tuple[Block, ...], dict[int, int]] | None = None
        self._answers: dict[Any, Any] = {}

    def _native_handle(self) -> Any:
        global _REBUILDS
        if self._handle is None:
            if self._build_input is None:
                raise ValueError("a released native DAG without its input tables")
            _REBUILDS += 1
            self._handle = _call(_native().V2Dag.build, self._build_input)
        return self._handle

    def release(self) -> None:
        """Drop the native columns of a DAG the builder made (its input
        tables rebuild them on the next query that needs them)."""

        if self._build_input is not None:
            self._handle = None

    @property
    def released(self) -> bool:
        return self._handle is None

    def _answer(self, key: Any, work: Any) -> Any:
        found = self._answers.get(key)
        if found is None:
            found = self._answers[key] = work()
        return found

    def __getattr__(self, name: str) -> Any:
        # Only attributes the handle does not have: the GenericDAG's.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.materialize(), name)

    @property  # type: ignore[misc]
    def __class__(self) -> type:
        # The stand-in passes the consumers' ``isinstance(dag, GenericDAG)``
        # (e.g. ``StageCompilationInput`` of the per-point rows), which reads
        # no attribute; ``type(dag)`` stays ``NativeDag``, which the native
        # dispatches test first. Equality with a ``GenericDAG`` compares the
        # decoded fields.
        return GenericDAG

    def __reduce_ex__(self, protocol: Any) -> Any:
        # ``pickle`` and ``copy`` take the decoded ``GenericDAG`` (counted):
        # a native handle does not cross a process.
        return self.materialize().__reduce_ex__(protocol)

    def materialize(self) -> GenericDAG:
        """The decoded ``GenericDAG``, decoded on the first call only (a
        released DAG is rebuilt for it and released again)."""

        if self._decoded is None:
            released = self.released
            data = self.columns()
            if released:
                # The rebuilt columns are not held while Python decodes.
                self.release()
            decoded = self._decode_columns(data)
            self._decoded = decoded
            # The decoded fields become the handle's own attributes, so the
            # consumers' reads in their loops (``dag.currents[i]``, the
            # runtime schema's and the export's) do not go through
            # ``__getattr__`` (0.7 M calls in the u u~ > g g g g export).
            for name in _DAG_FIELDS:
                self.__dict__.setdefault(name, getattr(decoded, name))
        return self._decoded

    @property
    def decoded(self) -> bool:
        """Whether ``materialize`` has decoded this DAG."""

        return self._decoded is not None

    @classmethod
    def from_dag(cls, dag: GenericDAG, model: Any) -> NativeDag:
        """The columnar form of a Python ``GenericDAG`` (its stage digest)."""

        handle = _call(_native().V2Dag.from_columns, encode_dag(dag))
        return cls(handle, dag.process, model)

    def counts(self) -> tuple[int, int, int]:
        """``(currents, interactions, sources)``."""

        return self._answer(  # type: ignore[no-any-return]
            "counts", lambda: tuple(self._native_handle().counts())
        )

    def columns(self) -> bytes:
        return bytes(self._native_handle().columns())

    def digest(self) -> str:
        """Flat-record SHA-256 of the DAG (the DAG part of S1'/S1'')."""

        return self._answer(  # type: ignore[no-any-return]
            "digest", lambda: str(self._native_handle().digest())
        )

    def _tables(self) -> str:
        if self._block_input is None:
            self._block_input = block_input(
                self.process, self._model, self._clone_fields
            )
        return self._block_input

    def blocks(self, kernel_variant: str) -> tuple[Block, ...]:
        """``loops.v2.blocks._blocks(dag, model, kernel_variant)``; the blocks
        of a CT program's clone particles are keyed by the fields their lines
        are read as (``ct_closure.CountertermBlocks.build``)."""

        return self._blocks_and_clones(kernel_variant)[0]

    def _blocks_and_clones(
        self, kernel_variant: str
    ) -> tuple[tuple[Block, ...], dict[int, int]]:
        """The blocks and, for the clone blocks, block id -> clone particle."""

        from dataclasses import replace

        from .blocks import _checked_kernel_variant

        variant = _checked_kernel_variant("plain", kernel_variant)  # type: ignore[arg-type]
        if self._blocks is not None:
            # The variant only labels the blocks (the DAG does not depend on
            # it, ``TreeBlocks.with_kernels``).
            found, clone_map = self._blocks
            if found and found[0].kernel_variant != variant:
                found = tuple(replace(block, kernel_variant=variant) for block in found)
                self._blocks = (found, clone_map)
            return found, dict(clone_map)
        out: list[Block] = []
        clones: dict[int, int] = {}
        raw, assignments = _call(self._native_handle().blocks, self._tables())
        # Equal values are built once: the rows' helicity assignments, the
        # colour states and orders repeat over the blocks of a leg set.
        helicities = [tuple(tuple(pair) for pair in row) for row in assignments]
        shared: dict[Any, Any] = {}

        def one(value: Any) -> Any:
            return shared.setdefault(value, value)

        for block in raw:
            if block["clone"] is not None:
                clones[len(out)] = int(block["clone"])
            colour = block["colour"]
            key = BlockKey(
                external_mask=block["mask"],
                particle_id=block["pdg"],
                word=tuple(block["word"]),
                coupling_orders=one(tuple(tuple(pair) for pair in block["orders"])),
                color_state=one(
                    ColorState(
                        accuracy=colour[0],
                        sector_id=colour[1],
                        line_groups=tuple(colour[2]),
                        basis_key=tuple(colour[3]),
                        traces=tuple(tuple(trace) for trace in colour[4]),
                        lines=tuple(tuple(line) for line in colour[5]),
                    )
                ),
                chirality=block["chirality"],
                spin_state=block["spin_state"],
                flavour_flow=one(tuple(block["flavour"])),
                quantum_number_flow=one(
                    tuple(tuple(pair) for pair in block["quantum_numbers"])
                ),
                auxiliary_kind=block["auxiliary"],
            )
            out.append(
                Block(
                    id=len(out),
                    key=key,
                    colour_flow=BlockColourFlow(
                        strings=tuple(tuple(string) for string in block["strings"]),
                        incoming_open=block["incoming_open"],
                        outgoing_open=block["outgoing_open"],
                        singlet_labels=tuple(block["singlet_labels"]),
                        sign=block["colour_sign"],
                        nc_grade=block["nc_grade"],
                        traces=tuple(tuple(trace) for trace in block["traces"]),
                    ),
                    block_class="plain",
                    kernel_variant=variant,
                    rows=tuple(
                        BlockRow(helicities[assignment], current_id)
                        for assignment, current_id in block["rows"]
                    ),
                    connection_sign=block["connection_sign"],
                    pairing_sign=block["pairing_sign"],
                )
            )
        self._blocks = (tuple(out), clones)
        return tuple(out), dict(clones)

    def u1_leaking(
        self,
        kinds: Iterable[int],
        generator_kinds: Iterable[int],
        adjoint: Iterable[int],
        fundamental_mask: int,
        rest_mask: int,
    ) -> frozenset[int]:
        """The vertex kinds the DAG scan of ``ct_closure.u1_leaking_structures``
        reports (see ``u1_leaking_structures``)."""

        arguments = (
            tuple(sorted(int(k) for k in kinds)),
            tuple(sorted(int(k) for k in generator_kinds)),
            tuple(sorted(int(p) for p in adjoint)),
            int(fundamental_mask),
            int(rest_mask),
        )
        return self._answer(  # type: ignore[no-any-return]
            ("u1", arguments),
            lambda: frozenset(
                self._native_handle().u1_leaking(
                    *map(list, arguments[:3]), *arguments[3:]
                )
            ),
        )

    def coloured_propagated(self) -> frozenset[int]:
        """Coloured particle ids that non-source, non-auxiliary currents
        propagate (the DAG scan of ``blocks.ward_breaking_fields``)."""

        return self._answer(  # type: ignore[no-any-return]
            "coloured",
            lambda: frozenset(
                _call(self._native_handle().coloured_propagated, self._tables())
            ),
        )

    def propagated_species(self) -> tuple[tuple[int, bool], ...]:
        """Distinct ``(particle id, amputated)`` of the non-source,
        non-auxiliary currents, sorted; ``amputated``: the current's
        momentum leaves exactly one leg out. The current scans of
        ``blocks.ward_breaking_fields`` (every particle) and
        ``artifact.coloured_widths`` (the particles of currents that are not
        amputated), which test colours, widths and propagators themselves."""

        full = (1 << len(self.process.legs)) - 1
        return self._answer(  # type: ignore[no-any-return]
            "species",
            lambda: tuple(
                (int(pdg), bool(amputated))
                for pdg, amputated in self._native_handle().propagated_species(full)
            ),
        )

    def decode(self) -> GenericDAG:
        """A new ``GenericDAG`` of Python objects (counted)."""

        return self._decode_columns(self.columns())

    def _decode_columns(self, data: bytes) -> GenericDAG:
        global _DECODES
        from pyamplicol.color.plan import build_color_plan

        _DECODES += 1
        plan = build_color_plan(
            self.process,
            color_accuracy="full",
            max_sectors=None,
            reference_color_order=None,
            fold_trace_reflections=False,
        )
        return _decode(data, self.process, plan)


def native_blocks(
    process: CanonicalProcessIR,
    model: Any,
    *,
    max_coupling_orders: Mapping[str, int] | None = None,
    ignored_particle_ids: Iterable[int] = (),
    ignored_vertex_kinds: Iterable[int] = (),
    reachability: LoopBlockReachability | None = None,
) -> NativeDag:
    """The native twin of ``GenericDAGCompiler(model=model,
    max_coupling_orders=..., ignored_particle_ids=..., ignored_vertex_kinds=...,
    loop_blocks=True, loop_block_reachability=reachability).compile(process)``."""

    tables = dag_input(
        process,
        model,
        max_coupling_orders=max_coupling_orders,
        ignored_particle_ids=ignored_particle_ids,
        ignored_vertex_kinds=ignored_vertex_kinds,
        reachability=reachability,
    )
    return NativeDag(
        _call(_native().V2Dag.build, tables), process, model, build_input=tables
    )


@dataclass(frozen=True)
class NativeCountertermBlocks:
    """The native twin of ``ct_closure.CountertermBlocks``: the CT program's
    loop-request DAG (``dag``, a ``NativeDag``), its blocks (clone blocks
    keyed by their fields) and clone lines (block id -> ``CloneLine``)."""

    recursion: Any
    dag: NativeDag
    blocks: tuple[Block, ...]
    clones: Mapping[int, Any]
    kernel_variant: str


def native_counterterm_blocks(
    recursion: Any,
    process: CanonicalProcessIR,
    *,
    ignored_particle_ids: Iterable[int] = (),
) -> NativeCountertermBlocks:
    """The native twin of ``ct_closure.CountertermBlocks.build(recursion,
    process, ignored_particle_ids=...)``: the loop-request recursion of the
    CT-lowered model (``recursion.model.lowered``) over the leg sets without
    leg 1 except all other legs (``RootReachability``), within the program's
    order limits and without the vetoed particles (DESIGN §6.1 items 1-6 and
    8), grouped into blocks with the clone lines of ``_clone_lines``."""

    from dataclasses import replace

    from pyamplicol.models.prepared_catalog_builder import model_tree_kernel_variant

    from .ct_closure import RootReachability, _clone_lines

    # As ``CountertermBlocks.build``: the lowered model's Weyl-projection
    # support comes decided from the lowering cache (most of the input
    # tables' cost otherwise).
    model = recursion.model.lowered.tree_model()
    full = (1 << len(process.legs)) - 1
    process = replace(process, color_accuracy="full")
    lines = _clone_lines(recursion)
    tables = dag_input(
        process,
        model,
        max_coupling_orders=recursion.coupling_order_limits(),
        ignored_particle_ids=tuple(ignored_particle_ids),
        reachability=RootReachability(full),
    )
    dag = NativeDag(
        _call(_native().V2Dag.build, tables),
        process,
        model,
        {clone: line.field_id for clone, line in lines.items()},
        build_input=tables,
    )
    variant = model_tree_kernel_variant(model)
    blocks, clone_particles = dag._blocks_and_clones(variant)
    return NativeCountertermBlocks(
        recursion,
        dag,
        blocks,
        {block_id: lines[clone] for block_id, clone in clone_particles.items()},
        variant,
    )


def u1_leaking_structures(program: NativeCountertermBlocks) -> frozenset[Any]:
    """``ct_closure.u1_leaking_structures`` on a native CT program (its
    model-level gates and kinds-to-structures map; the DAG scan is native,
    ``NativeDag.u1_leaking``)."""

    from .ct_closure import _u1_leaking

    return _u1_leaking(program.recursion, program.dag)


# ------------------------------------------------------------------ reuse oracle

REUSE_SCHEMA = "pyamplicol.loop-v2.dag-reuse/1"


def reuse_tables(dag: GenericDAG, model: Any) -> str:
    """The JSON tables of ``V2Dag.reuse`` for ``dag``: the evaluation
    equivalence of every vertex kind it uses (``_kernel_equivalence``) and the
    rank, in Python's order, of every interaction's runtime coupling identity
    (``_runtime_coupling_identity``)."""

    from pyamplicol.generation.dag_equivalence import (
        _kernel_equivalence,
        _runtime_coupling_identity,
    )

    cache: dict[int, Any] = {}
    kernels = []
    for kind in sorted({int(item.vertex_kind) for item in dag.interactions}):
        equivalence = _kernel_equivalence(model, kind, cache)
        kernels.append(
            [
                kind,
                equivalence.class_id,
                *_pair_bits(equivalence.factor),
                tuple(equivalence.input_order) == (1, 0),
                None
                if equivalence.input_exchange_factor is None
                else _pair_bits(equivalence.input_exchange_factor),
            ]
        )
    identities: dict[Any, int] = {}
    known: dict[tuple[Any, ...], int] = {}
    per_interaction = []
    for item in dag.interactions:
        key = (int(item.vertex_kind), tuple(item.vertex_particles), item.coupling)
        found = known.get(key)
        if found is None:
            if any(math.isnan(part) for part in item.coupling):
                raise NativeDagUnsupported("a NaN coupling")
            identity = _runtime_coupling_identity(
                model,
                vertex_kind=item.vertex_kind,
                vertex_particles=item.vertex_particles,
                coupling=item.coupling,
            )
            found = known[key] = identities.setdefault(identity, len(identities))
        per_interaction.append(found)
    ranks = _ranks(identities)
    return json.dumps(
        {
            "schema": REUSE_SCHEMA,
            "kernels": kernels,
            "identity_ranks": [ranks[identity] for identity in per_interaction],
        },
        separators=(",", ":"),
    )


def native_reuse(
    dag: GenericDAG, model: Any, *, restricted: bool = True
) -> tuple[tuple[Any, ...], tuple[InteractionNode, ...]]:
    """The native value-equivalence derivation of a Python ``dag``
    (``_derive_current_value_equivalences``; ``restricted``: of the currents
    that share their contract, as ``assign_recursive_current_evaluation_reuse``
    runs it) and the interactions regrouped through it
    (``_rewrite_interaction_evaluation_reuse``): the oracle of A3's port.
    ``dag`` must fit the columns (``encode_dag``; otherwise
    ``NativeDagUnsupported``)."""

    from pyamplicol.generation.dag_equivalence import _CurrentValueEquivalence

    handle = _call(_native().V2Dag.from_columns, encode_dag(dag))
    records, rewritten = _call(handle.reuse, reuse_tables(dag, model), restricted)
    equivalences = tuple(
        _CurrentValueEquivalence(
            int(representative),
            (_float(real), _float(imaginary)),
        )
        for representative, real, imaginary in records
    )
    decoded = _decode(bytes(rewritten.columns()), dag.process, dag.color_plan)
    return equivalences, decoded.interactions


def _float(bits: int) -> float:
    return float(struct.unpack("<d", struct.pack("<Q", int(bits)))[0])


# ------------------------------------------------------------------ digests


def _put_str(out: bytearray, value: str) -> None:
    data = value.encode()
    out += struct.pack("<I", len(data)) + data


def _put_labels(out: bytearray, labels: Sequence[int]) -> None:
    out += struct.pack("<I", len(labels)) + bytes(labels)


def _put_orders(out: bytearray, orders: Sequence[tuple[str, int]]) -> None:
    out += struct.pack("<I", len(orders))
    for name, value in orders:
        _put_str(out, str(name))
        out += struct.pack("<q", int(value))


def blocks_digest(
    blocks: Sequence[Block], clones: Mapping[int, Any] | None = None
) -> str:
    """Flat-record SHA-256 of ``blocks`` (and the CT program's clone lines,
    ``CountertermBlocks.clones``: block id -> ``CloneLine``)."""

    hash_ = hashlib.sha256(_BLOCKS_DIGEST_SCHEMA)
    hash_.update(struct.pack("<Q", len(blocks)))
    for block in blocks:
        out = bytearray()
        key = block.key
        out += struct.pack("<IQq", block.id, key.external_mask, key.particle_id)
        _put_labels(out, key.word)
        _put_orders(out, key.coupling_orders)
        state = key.color_state
        _put_str(out, state.accuracy)
        out += struct.pack("<qI", state.sector_id, len(state.line_groups))
        out += b"".join(struct.pack("<q", group) for group in state.line_groups)
        out += struct.pack("<I", len(state.basis_key))
        for basis in state.basis_key:
            _put_str(out, basis)
        for parts in (state.traces, state.lines):
            out += struct.pack("<I", len(parts))
            for labels in parts:
                _put_labels(out, labels)
        if not isinstance(key.spin_state, int):
            raise NativeDagUnsupported(f"tuple spin state {key.spin_state!r}")
        out += struct.pack("<bi", key.chirality, key.spin_state)
        out += struct.pack("<I", len(key.flavour_flow))
        out += b"".join(struct.pack("<q", value) for value in key.flavour_flow)
        out += struct.pack("<I", len(key.quantum_number_flow))
        for name, value in key.quantum_number_flow:
            _put_str(out, name)
            _put_str(out, value)
        out += b"\x00" if key.auxiliary_kind is None else b"\x01"
        if key.auxiliary_kind is not None:
            _put_str(out, key.auxiliary_kind)
        flow = block.colour_flow
        out += struct.pack("<I", len(flow.strings))
        for string in flow.strings:
            _put_labels(out, string)
        out += struct.pack("<BB", flow.incoming_open, flow.outgoing_open)
        _put_labels(out, flow.singlet_labels)
        out += struct.pack("<b", flow.sign)
        out += (
            b"\x00"
            if flow.nc_grade is None
            else b"\x01" + struct.pack("<q", flow.nc_grade)
        )
        out += struct.pack("<I", len(flow.traces))
        for trace in flow.traces:
            _put_labels(out, trace)
        _put_str(out, block.block_class)
        _put_str(out, block.kernel_variant)
        out += struct.pack("<I", len(block.rows))
        for row in block.rows:
            out += struct.pack("<I", len(row.helicities))
            out += b"".join(
                struct.pack("<Bi", label, helicity)
                for label, helicity in row.helicities
            )
            out += struct.pack("<I", row.current_id)
        out += struct.pack("<bb", block.connection_sign, block.pairing_sign)
        hash_.update(out)
    clone_rows = sorted((clones or {}).items())
    hash_.update(struct.pack("<Q", len(clone_rows)))
    for block_id, line in clone_rows:
        out = bytearray(struct.pack("<Iq", block_id, line.field_id))
        _put_orders(out, line.half)
        component, power, grade = line.slot
        _put_str(out, str(component))
        out += struct.pack("<qq", power, grade)
        out += b"\x00" if line.factor is None else b"\x01"
        if line.factor is not None:
            _put_str(out, line.factor)
        hash_.update(out)
    return hash_.hexdigest()


def stage_digest(stage: str, dag_digest: str, blocks_digest_: str) -> str:
    """S1' (``stage="S1'"``) or S1'' (``"S1''"``) of a DAG and its blocks."""

    if stage not in ("S1'", "S1''"):
        raise ValueError(f"unknown stage {stage!r}")
    return hashlib.sha256(
        stage.encode()
        + b"\0"
        + bytes.fromhex(dag_digest)
        + bytes.fromhex(blocks_digest_)
    ).hexdigest()


__all__ = [
    "BLOCK_INPUT_SCHEMA",
    "COLUMNS_SCHEMA",
    "INPUT_SCHEMA",
    "NATIVE_DAG",
    "NATIVE_FALLBACKS",
    "REUSE_SCHEMA",
    "NativeCountertermBlocks",
    "NativeDag",
    "NativeDagUnsupported",
    "block_input",
    "blocks_digest",
    "dag_input",
    "decode_count",
    "encode_dag",
    "native_blocks",
    "native_counterterm_blocks",
    "native_reuse",
    "rebuild_count",
    "reuse_tables",
    "stage_digest",
    "u1_leaking_structures",
]
