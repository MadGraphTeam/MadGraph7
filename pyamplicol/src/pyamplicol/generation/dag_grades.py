# SPDX-License-Identifier: 0BSD
"""N_c grades of loop-request currents (design §3.2 item 4, §3.3).

The tree engine writes the ``-1/N_c`` of the SU(N_c) Fierz identity as a
U(1)-subtraction auxiliary current with the numeric weight of N_c = 3. A
current whose trees produce k such auxiliaries has the *N_c grade* ``-k``:
its formal-N_c value is its value times ``N_c^-k 3^k``, which the loop seeds
apply per block (``loops.v2.colour.block_colour_network``). A block (the
currents of one ``CurrentIndex`` up to the helicity ancestry, its rows) is
exact in N_c only if all its trees have one grade. Identical-flavour lines
break that from u u~ > u u~ g upward: one colour word is reached through a
gluon (grade 0) and through the U(1)-subtraction piece of the
Fermi-exchanged pairing (grade -1), for different helicities of the lines'
legs or for the same ones; a photon and a U(1) piece joining the same lines
do the same.

``split_u1_grades`` makes every block of a loop-request DAG exact in N_c.
The grade enters the identity (``ColorState.nc_grade``) of exactly the
currents of a block whose trees have several grades:

* a current whose own derivations mix grades becomes one current per grade,
  each fed by the derivations of that grade, and so do its consumers (a
  current fed by a mixed one mixes as well);
* a current of one grade in such a block (the block mixes across its
  helicity rows) only takes its grade as a label.

Every other current, interaction and id order is unchanged, so a DAG whose
blocks have one grade each is returned as it is (and one without
U(1)-subtraction auxiliaries is not even scanned). The sum of a split
current's copies is its value, so full colour at N_c = 3 is unchanged up to
the summation order. The native builder (``rusticol-core``
``loop_v2::dag::grades``) runs the same pass.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from pyamplicol.models.contracts import U1_SUBTRACTION_AUXILIARY_KIND

from .dag_types import CurrentIndex, CurrentNode, GenericDAG, InteractionNode


def u1_grade_sets(dag: GenericDAG) -> list[frozenset[int]]:
    """The N_c grades of every current's derivations (sources: {0}; empty
    for a current no interaction derives).

    Interactions are read by increasing size of their result's leg mask (a
    stable order: interactions of one mask, e.g. a two-point counterterm
    insertion, keep the compiler's order), so every input's grades are
    complete before it is read. This is the one grade rule of the Python
    lanes: ``loops.v2.blocks.u1_grades`` is its single-grade view.
    """

    currents = dag.currents
    u1 = [
        current.index.auxiliary_kind == U1_SUBTRACTION_AUXILIARY_KIND
        for current in currents
    ]
    found: list[set[int]] = [
        {0} if current.is_source else set() for current in currents
    ]
    for interaction in sorted(
        dag.interactions,
        key=lambda row: currents[row.result_id].index.external_mask.bit_count(),
    ):
        left, right = found[interaction.left_id], found[interaction.right_id]
        if not left or not right:
            raise ValueError(
                "a loop-request DAG interaction reads a current derived after it"
            )
        shift = u1[interaction.result_id]
        found[interaction.result_id].update(a + b - shift for a in left for b in right)
    return [frozenset(grades) for grades in found]


def _block_key(index: CurrentIndex) -> tuple[object, ...]:
    """The current's identity without its helicity ancestry: its block
    (``loops.v2.blocks.BlockKey``; the momentum mask of a loop-request
    current is its leg mask)."""

    return (
        index.particle_id,
        index.external_mask,
        index.chirality,
        index.spin_state,
        index.flavour_flow,
        index.quantum_number_flow,
        index.color_state,
        index.coupling_orders,
        index.auxiliary_kind,
        index.ordered_external_labels,
    )


def split_u1_grades(dag: GenericDAG) -> GenericDAG:
    """``dag`` with every block of several N_c grades graded (module
    docstring).

    The copies of a split current take its place, by decreasing grade; the
    interactions of a split result are emitted in its place, one per grade
    of each split input (decreasing, left before right). Interactions keep
    their colour weights, couplings and evaluation records; the evaluation
    reuse that follows in the loop-request compilation regroups them.
    """

    currents = dag.currents
    if not any(
        current.index.auxiliary_kind == U1_SUBTRACTION_AUXILIARY_KIND
        for current in currents
    ):
        return dag
    if dag.amplitude_roots:
        raise ValueError("the N_c grade split takes loop-request DAGs (no roots)")
    grade_sets = u1_grade_sets(dag)
    keys = [_block_key(current.index) for current in currents]
    block_grades: dict[tuple[object, ...], set[int]] = defaultdict(set)
    for key, grades in zip(keys, grade_sets, strict=True):
        block_grades[key].update(grades)
    graded = [len(block_grades[key]) > 1 for key in keys]
    if not any(graded):
        return dag
    u1 = [
        current.index.auxiliary_kind == U1_SUBTRACTION_AUXILIARY_KIND
        for current in currents
    ]
    # old id -> ((grade, new id), ...) by decreasing grade.
    copies: list[tuple[tuple[int, int], ...]] = []
    out_currents: list[CurrentNode] = []
    for current, grades, label in zip(currents, grade_sets, graded, strict=True):
        if not label:
            # One grade (none for a current that no interaction derives and
            # that is not a source: nothing reads it; the native pass keeps
            # it too).
            new = len(out_currents)
            copies.append(tuple((grade, new) for grade in grades))
            out_currents.append(
                current if current.id == new else replace(current, id=new)
            )
            continue
        ids = []
        for grade in sorted(grades, reverse=True):
            new = len(out_currents)
            index = current.index
            out_currents.append(
                replace(
                    current,
                    id=new,
                    index=replace(
                        index,
                        color_state=replace(index.color_state, nc_grade=grade),
                    ),
                )
            )
            ids.append((grade, new))
        copies.append(tuple(ids))
    out_interactions: list[InteractionNode] = []
    for interaction in dag.interactions:
        targets = dict(copies[interaction.result_id])
        shift = u1[interaction.result_id]
        for left_grade, left in copies[interaction.left_id]:
            for right_grade, right in copies[interaction.right_id]:
                new = len(out_interactions)
                target = targets[left_grade + right_grade - shift]
                if new == interaction.id and (left, right, target) == (
                    interaction.left_id,
                    interaction.right_id,
                    interaction.result_id,
                ):
                    out_interactions.append(interaction)
                    continue
                out_interactions.append(
                    replace(
                        interaction,
                        id=new,
                        left_id=left,
                        right_id=right,
                        result_id=target,
                    )
                )
    sources = tuple(copies[source][0][1] for source in dag.sources)
    return replace(
        dag,
        currents=tuple(out_currents),
        sources=sources,
        interactions=tuple(out_interactions),
    )


__all__ = ["split_u1_grades", "u1_grade_sets"]
