# SPDX-License-Identifier: 0BSD
"""N_c grades of loop-request currents (design §3.2 item 4, §3.3).

The tree engine writes the ``-1/N_c`` of the SU(N_c) Fierz identity as a
U(1)-subtraction auxiliary current with the numeric weight of N_c = 3. A
current whose trees produce k such auxiliaries has the *N_c grade* ``-k``:
its formal-N_c value is its value times ``N_c^-k 3^k``, which the loop seeds
apply per block (``loops.v2.colour.block_colour_network``). A current
derived both with and without such a piece has no single grade, and its
value is not exact in N_c: from u u~ > u u~ g upward, identical-flavour lines
reach one colour word through a gluon (grade 0) and through the
U(1)-subtraction piece of the Fermi-exchanged pairing (grade -1); a photon
and a U(1) piece joining the same lines do the same.

``split_u1_grades`` makes every current of a loop-request DAG exact in N_c:
a current whose derivations mix grades becomes one current per grade, each
fed by the derivations of that grade, and so do its consumers (a current
fed by a mixed one mixes as well). The split copies carry their grade in
their identity (``ColorState.nc_grade``); every other current, interaction
and id order is unchanged, so a DAG without mixed currents is returned as it
is. The sum of the copies' values is the mixed current's value, so full
colour at N_c = 3 is unchanged up to the summation order. The native builder
(``rusticol-core`` ``loop_v2::dag::grades``) runs the same pass.
"""

from __future__ import annotations

from dataclasses import replace

from pyamplicol.models.contracts import U1_SUBTRACTION_AUXILIARY_KIND

from .dag_types import CurrentNode, GenericDAG, InteractionNode


def u1_grade_sets(dag: GenericDAG) -> list[frozenset[int]]:
    """The N_c grades of every current's derivations (sources: {0}).

    Interactions are in increasing result-subset order (the compiler emits
    them so), so every input's grades are complete before it is read.
    """

    currents = dag.currents
    u1 = [
        current.index.auxiliary_kind == U1_SUBTRACTION_AUXILIARY_KIND
        for current in currents
    ]
    found: list[set[int]] = [
        {0} if current.is_source else set() for current in currents
    ]
    for interaction in dag.interactions:
        left, right = found[interaction.left_id], found[interaction.right_id]
        if not left or not right:
            raise ValueError(
                "a loop-request DAG interaction reads a current derived after it"
            )
        shift = u1[interaction.result_id]
        found[interaction.result_id].update(a + b - shift for a in left for b in right)
    return [frozenset(grades) for grades in found]


def split_u1_grades(dag: GenericDAG) -> GenericDAG:
    """``dag`` with every current of mixed N_c grade split per grade
    (module docstring).

    The copies of a current take its place, by decreasing grade; the
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
    if all(len(grades) == 1 for grades in grade_sets):
        return dag
    u1 = [
        current.index.auxiliary_kind == U1_SUBTRACTION_AUXILIARY_KIND
        for current in currents
    ]
    # old id -> ((grade, new id), ...) by decreasing grade.
    copies: list[tuple[tuple[int, int], ...]] = []
    out_currents: list[CurrentNode] = []
    for current, grades in zip(currents, grade_sets, strict=True):
        if len(grades) == 1:
            new = len(out_currents)
            copies.append(((next(iter(grades)), new),))
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
        result = copies[interaction.result_id]
        shift = u1[interaction.result_id]
        targets = dict(result)
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
