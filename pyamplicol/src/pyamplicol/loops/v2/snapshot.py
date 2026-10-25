# SPDX-License-Identifier: 0BSD
"""Stage digests of the v2 generator (WP3.1 I0, structural equality).

The native generator stages must reproduce the Python stages exactly. A stage
digest is the SHA-256 of a canonical encoding of the stage's objects, in the
generator's own numbering (ids are assigned first-seen, so a native stage
equals the Python stage iff every record does, in the same order):

* ``S1`` families: keys, ranks, closures, the prefix graph, the logs of
  scaleless and rejected families, the certified-scaleless families with
  their closures (the poles companion) and the merged closed-loop flavours;
* ``S4`` schedules: the view's contents and, per family, the trie (frame,
  leaves, nodes), the re-test relabelling and the master keys;
* ``S5`` seeds: the seed table (classes, orders, exact cells) and each
  skeleton's class combination;
* ``S8`` the exported plan v4 (``forward=False``);
* ``L3`` the complete virtual (every component and sector, the Born and the
  IR poles) at given points, binary64 rounded to 12 significant digits: it
  covers what the plan does not (counterterm sites, external fields).

The stages of the loop-request DAG builder (WP3.1 A, ``RUST-MIGRATION.md``
§2 rule 6) are streaming SHA-256 digests over flat records (no dataclass
recursion), so that seven legs cost a fraction of their stage:

* ``S1'`` the loop-block DAG and its blocks (``loop_blocks_digest``);
* ``S1''`` the counterterm program's DAG, its blocks and clone lines
  (``counterterm_program_digest``);
* ``S2`` the Born and counterterm root-closure terms
  (``root_closure_digest``);
* ``P`` the native interaction programs and their saved kernels
  (``programs_digest``);
* ``P-ct`` the counterterm program's interaction program and saved kernels
  in the native lane's export (``counterterm_trees_digest``).

The DAG part of ``S1'``/``S1''`` is ``dag_input.NativeDag.digest`` (every
current and interaction field, floats by their bits; a Python ``GenericDAG``
is encoded into the same columns first), the blocks part
``dag_input.blocks_digest``, so the native and the Python stage give the
same digest exactly when they build equal objects.

The encoding is independent of the hash seed: sets are sorted by their
encoded items and dataclasses are encoded by their compared fields. Digests
are a test tool; artifacts never carry them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from typing import Any


def canonical(value: Any) -> Any:
    """A JSON-encodable form of ``value`` independent of the hash seed."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, (Fraction, Decimal)):
        return str(value)
    if isinstance(value, complex):
        return [repr(value.real), repr(value.imag)]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return [
            type(value).__name__,
            *(
                canonical(getattr(value, f.name))
                for f in dataclasses.fields(value)
                if f.compare
            ),
        ]
    if isinstance(value, Mapping):
        items = [[canonical(k), canonical(v)] for k, v in value.items()]
        return ["map", *sorted(items, key=_text)]
    if isinstance(value, (set, frozenset)):
        return ["set", *sorted((canonical(v) for v in value), key=_text)]
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    return ["repr", type(value).__name__, repr(value)]


def _text(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"))


def digest(records: Iterable[Any]) -> str:
    """SHA-256 of a sequence of records, each encoded canonically."""

    hasher = hashlib.sha256()
    for record in records:
        hasher.update(_text(canonical(record)).encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


def families_records(families: Any) -> Iterable[Any]:
    """S1: every record of a ``FamilySet``."""

    yield "legs", families.legs
    for key, family in families.families.items():
        yield "family", key, family.rank, family.closures
    graph = families.prefixes
    for state in range(len(graph)):
        yield (
            "state",
            graph.parent[state],
            graph.position[state],
            graph.species[state],
            graph.depth[state],
        )
    yield "scaleless", families.scaleless
    for key, family in families.scaleless_families.items():
        yield "scaleless-family", key, family.rank, family.closures
    yield "rejected", families.rejected
    yield "merged", families.merged


def schedules_records(plan: Any) -> Iterable[Any]:
    """S4: the view's contents and every family's schedule-B data."""

    for label, content in enumerate(plan.view._contents):
        yield "content", label, content
    for key, schedule in plan.schedules.items():
        trie = schedule.trie
        yield "family", key, trie.frame, trie.rank, schedule.relabelling
        for skeleton, ring in trie.leaves:
            yield "leaf", skeleton, [segment.label for segment in ring]
        for node in trie.nodes:
            yield (
                "node",
                node.depth,
                node.orders,
                [segment.label for segment in node.suffix],
                node.rank_in,
                [(segment.label, child) for segment, child in node.edges],
            )
        yield "masters", schedule.masters


def seeds_records(slice_: Any) -> Iterable[Any]:
    """S5: the seed table and each skeleton's class combination."""

    table = slice_.table
    yield "basis", table.basis
    for c, (seed, orders, cells) in enumerate(
        zip(table.classes, table.orders, table.cells, strict=True)
    ):
        yield "class", c, seed, orders, cells
    yield "formal_nc", table.formal_nc
    yield "certificates", table.certificates
    for skeleton, combination in slice_.class_of.items():
        yield "skeleton", skeleton, combination


def stage_digests(slice_: Any, *, plan: bool = True) -> dict[str, str]:
    """S1, S4, S5 and (with ``plan``) S8 of a ``LoopSlice``."""

    out = {
        "S1": digest(families_records(slice_.families)),
        "S4": digest(schedules_records(slice_.plan)),
        "S5": digest(seeds_records(slice_)),
    }
    if plan:
        from .plan_export import export_plan

        text = export_plan(slice_, forward=False).json()
        out["S8"] = hashlib.sha256(text.encode()).hexdigest()
    return out


def _rounded(value: Any) -> str:
    return f"{float(value):.11e}"


def virtual_digest(
    slice_: Any,
    points: Sequence[Any],
    scales: Sequence[Any],
    parameters: Mapping[str, str],
) -> str:
    """L3: the complete virtual at ``points`` (binary64, 12 digits)."""

    def records():
        for point, scale in zip(points, scales, strict=True):
            result = slice_.evaluate(point, scale, parameters=parameters)
            for sector, value in sorted(result.born.items()):
                yield "born", sector, _rounded(value)
            for component, sectors in sorted(result.components.items()):
                for sector, laurent in sorted(sectors.items()):
                    yield (
                        "component",
                        component,
                        sector,
                        [_rounded(x) for x in laurent.coefficients],
                    )
            for sector, poles in sorted((result.ir_poles or {}).items()):
                yield "poles", sector, [_rounded(x) for x in poles]

    return digest(records())


# ------------------------------------------------------------ DAG stages (A)


def _dag_digest(dag: Any) -> str:
    from .dag_input import NativeDag

    native = dag if isinstance(dag, NativeDag) else NativeDag.from_dag(dag, None)
    return native.digest()


def loop_blocks_digest(blocks: Any) -> str:
    """S1': the loop-block DAG of ``blocks`` (``blocks.TreeBlocks``) and its
    blocks; a native DAG is not decoded."""

    from .dag_input import blocks_digest, stage_digest

    return stage_digest(
        "S1'", _dag_digest(blocks.blocks_dag), blocks_digest(blocks.blocks)
    )


def counterterm_program_digest(program: Any) -> str:
    """S1'': the counterterm program's DAG, blocks and clone lines
    (``ct_closure.CountertermBlocks``); a native DAG is not decoded."""

    from .dag_input import blocks_digest, stage_digest

    tree = program.blocks
    return stage_digest(
        "S1''",
        _dag_digest(tree.blocks_dag),
        blocks_digest(tree.blocks, program.clones),
    )


def _site_term_record(term: Any, colours: dict[int, str]) -> list[Any]:
    """A root-closure term (``ct_sites.SiteTerm`` or
    ``assemble.WholeRuleTerm``) as one flat record: blocks by id (their
    contents are S1'/S1''), colour tensors by their canonical repr."""

    colour = colours.get(id(term.colour))
    if colour is None:
        colour = colours[id(term.colour)] = repr(term.colour)
    blocks = [int(block.id) for block in term.blocks]
    if hasattr(term, "binding"):
        entry = term.entry
        return [
            "site",
            str(entry.collection),
            int(entry.vertex_id),
            int(entry.color_index),
            int(entry.lorentz_index),
            entry.loop_group_index,
            int(term.binding.kernel_id),
            blocks,
            [int(leg) for leg in term.kernel_legs],
            int(term.sign),
            colour,
        ]
    return [
        "whole",
        blocks,
        int(term.sign),
        colour,
        repr(term.pairings),
        repr(term.grades),
        [str(name) for name in term.parameters],
    ]


def root_closure_records(slice_: Any) -> Iterable[Any]:
    """S2: the Born root-closure terms by amplitude order, then the
    counterterm program's root terms (``CountertermClosure.terms``)."""

    colours: dict[int, str] = {}
    for orders, terms in sorted(slice_.born_terms.items()):
        yield ["born-order", [list(pair) for pair in orders], len(terms)]
        for term in terms:
            yield _site_term_record(term, colours)
    program = slice_.counterterm_program
    if program is None or program.closure is None:
        return
    yield ["ct-terms", len(program.closure.terms)]
    for item in program.closure.terms:
        yield [
            "ct",
            item.root,
            item.component,
            int(item.power),
            [list(pair) for pair in item.orders],
            None if item.coupling is None else list(item.coupling),
            int(item.multiplicity),
            item.scale,
            int(item.grade),
            _site_term_record(item.term, colours),
        ]


def root_closure_digest(slice_: Any) -> str:
    """S2 of a ``LoopSlice`` (``root_closure_records``)."""

    hasher = hashlib.sha256(b"pyamplicol.loop-v2.S2/1\0")
    for record in root_closure_records(slice_):
        hasher.update(json.dumps(record, separators=(",", ":")).encode())
        hasher.update(b"\n")
    return hasher.hexdigest()


def programs_digest(trees: Any) -> str:
    """P: the native interaction programs and saved kernels of
    ``native_trees.NativeTrees`` (``export_plan(..., trees=True).trees``):
    its program record, its layout fields, then every saved evaluator's
    bytes in order (the vertex and propagator kernels, the parameters')."""

    hasher = hashlib.sha256(b"pyamplicol.loop-v2.P/1\0")
    hasher.update(trees.json().encode())
    hasher.update(
        json.dumps(
            canonical(
                [
                    trees.couplings,
                    trees.born_orders,
                    trees.squared_orders,
                    trees.masses,
                    trees.components,
                    trees.counterterm_outputs,
                ]
            ),
            separators=(",", ":"),
        ).encode()
    )
    for kernel in (*trees.kernels, trees.parameters):
        hasher.update(len(kernel).to_bytes(8, "little"))
        hasher.update(bytes(kernel))
    return hasher.hexdigest()


def counterterm_trees_digest(trees: Any) -> str:
    """P-ct: the counterterm program's interaction program and saved kernels
    of ``native_trees._CountertermTrees`` (``_CountertermTrees.of``, A5): the
    SHA-256 of its ``record`` as compact JSON, then of each saved kernel's
    SHA-256 in order. Untagged, so it equals the P-ct digests the stage
    records of ``tools/loop_v2_dag.py`` hold since A5."""

    hasher = hashlib.sha256()
    hasher.update(json.dumps(trees.record, separators=(",", ":")).encode())
    for saved in trees.saved:
        hasher.update(hashlib.sha256(saved).digest())
    return hasher.hexdigest()
