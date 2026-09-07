# SPDX-License-Identifier: 0BSD
"""Generated-basis compatibility and directed exact correlator matrix tests."""

import json
from dataclasses import replace
from fractions import Fraction

import pytest

from pyamplicol.color import correlator_matrices
from pyamplicol.color.connections import (
    ColorLeg,
    EmitGluon,
    ExactColorCoefficient,
    SplitGluon,
)
from pyamplicol.color.contraction_factors import exact_color_contraction_factor
from pyamplicol.color.correlator_matrices import (
    ColorCorrelator,
    ColorCorrelatorMatrixEntry,
    build_color_correlator_matrices,
    build_color_correlator_matrix,
    process_color_legs,
    sector_color_tensor,
)
from pyamplicol.color.plan import build_color_plan
from pyamplicol.models.builtin.process_ir import build_process_ir


def _plan(expression, *, accuracy="full"):
    return build_color_plan(
        build_process_ir(expression, color_accuracy=accuracy), color_accuracy=accuracy
    )


def _entries(matrix):
    return {
        (entry.left_sector_id, entry.right_sector_id): entry.weight
        for entry in matrix.entries
    }


@pytest.mark.parametrize(
    "expression",
    (
        "e- e+ > a a",
        "g g > g g",
        "d d~ > z g g",
        "d d~ > u u~",
        "d d~ > u u~ g g",
        "d d~ > d d~ g",
        "d d~ > u u~ s s~ g",
    ),
)
def test_identity_matches_every_existing_full_metric_entry(expression):
    plan = _plan(expression)
    matrix = build_color_correlator_matrix(plan, ColorCorrelator("born"))
    entries = _entries(matrix)
    assert matrix.order == 0
    assert matrix.sector_ids == tuple(sector.id for sector in plan.sectors)
    for left in plan.sectors:
        for right in plan.sectors:
            expected = exact_color_contraction_factor(
                plan, left, right, accuracy="full"
            )
            assert entries.get(
                (left.id, right.id), ExactColorCoefficient()
            ) == ExactColorCoefficient(expected)


def test_sector_adapter_adds_pairing_phase_and_ignores_traversal_replicas():
    plan = _plan("d d~ > u u~")
    terms = tuple(sector_color_tensor(sector, plan.process) for sector in plan.sectors)
    assert {term.coefficient.real for term in terms} == {Fraction(-1), Fraction(1)}
    assert len(set(terms)) == 2
    matrix = build_color_correlator_matrix(plan, ColorCorrelator("born"))
    assert any(entry.weight.real == -3 for entry in matrix.entries)


def test_process_roles_are_already_crossed_and_include_singlets():
    process = _plan("d d~ > z").process
    assert process_color_legs(process) == (
        ColorLeg(1, -3),
        ColorLeg(2, 3),
        ColorLeg(3, 1),
    )
    invalid = replace(
        process,
        legs=(replace(process.legs[0], color_role="inclusive"), *process.legs[1:]),
    )
    with pytest.raises(ValueError, match="concrete"):
        process_color_legs(invalid)


def test_dipole_convenience_and_n3lo_emitted_pair_normalization():
    plan = _plan("d d~ > z")
    pair = ColorCorrelator(
        "pair",
        (
            EmitGluon(2, -1),
            SplitGluon(-1, -2, -3),
            EmitGluon(-2, -4),
        ),
        (
            EmitGluon(2, -1),
            SplitGluon(-1, -2, -3),
            EmitGluon(-2, -4),
        ),
    )
    born, dipole, triple = build_color_correlator_matrices(
        plan,
        (
            ColorCorrelator("born"),
            ColorCorrelator.dipole("B12", 1, 2),
            pair,
        ),
    )
    assert born.entries[0].weight == ExactColorCoefficient(3)
    assert dipole.entries[0].weight == ExactColorCoefficient(-4)
    assert triple.order == 3
    assert triple.entries[0].weight == ExactColorCoefficient(Fraction(8, 3))
    assert ColorLeg(-2, 3) in triple.output_legs
    assert ColorLeg(-3, -3) in triple.output_legs
    assert all(leg.label != -1 for leg in triple.output_legs)
    assert ColorCorrelator.dipole("B12", 1, 2).to_json_dict() == {
        "id": "B12",
        "bra": [{"kind": "emit-gluon", "emitter": 1, "emitted": -1}],
        "ket": [{"kind": "emit-gluon", "emitter": 2, "emitted": -1}],
    }
    assert pair.to_json_dict()["bra"][1] == {
        "kind": "split-gluon",
        "parent": -1,
        "quark": -2,
        "antiquark": -3,
    }


def test_ordered_connections_store_both_triangles_without_hermiticity_assumptions():
    plan = _plan("d d~ > g g")
    request = ColorCorrelator(
        "ordered",
        (EmitGluon(2, -1), EmitGluon(1, -2)),
        (EmitGluon(3, -2), EmitGluon(1, -1)),
    )
    matrix, adjoint = build_color_correlator_matrices(
        plan,
        (
            request,
            ColorCorrelator("adjoint", request.ket, request.bra),
        ),
    )
    entries = _entries(matrix)
    assert entries == {
        (0, 0): ExactColorCoefficient(Fraction(-16, 3)),
        (0, 1): ExactColorCoefficient(Fraction(-16, 3)),
        (1, 0): ExactColorCoefficient(Fraction(20, 3)),
        (1, 1): ExactColorCoefficient(Fraction(2, 3)),
    }
    assert _entries(adjoint) == {
        (j, i): value.conjugate() for (i, j), value in entries.items()
    }
    amplitudes = (ExactColorCoefficient(1), ExactColorCoefficient(0, 1))
    value = ExactColorCoefficient()
    for entry in matrix.entries:
        value = (
            value
            + amplitudes[entry.left_sector_id].conjugate()
            * entry.weight
            * amplitudes[entry.right_sector_id]
        )
    assert value == ExactColorCoefficient(Fraction(-14, 3), -12)


def test_insertions_can_revive_zero_ordinary_metric_entries():
    plan = _plan("d d~ > u u~ g")
    born, inserted = build_color_correlator_matrices(
        plan,
        (
            ColorCorrelator("born"),
            ColorCorrelator.dipole("B13", 1, 3),
        ),
    )
    assert (0, 2) not in _entries(born)
    assert _entries(inserted)[0, 2] == ExactColorCoefficient(-4)


def test_selected_owner_ids_preserve_requested_coordinates():
    plan = _plan("g g > g g")
    matrix = build_color_correlator_matrix(
        plan, ColorCorrelator("born"), sector_ids=(5, 0)
    )
    assert matrix.sector_ids == (5, 0)
    assert {
        (entry.left_sector_id, entry.right_sector_id) for entry in matrix.entries
    } <= {(5, 5), (5, 0), (0, 5), (0, 0)}


def test_graph_images_are_reused_across_requested_matrices(monkeypatch):
    plan = _plan("d d~ > g g")
    original = correlator_matrices.apply_color_connection
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[1].emissions)
        return original(*args, **kwargs)

    monkeypatch.setattr(correlator_matrices, "apply_color_connection", counted)
    build_color_correlator_matrices(
        plan,
        (
            ColorCorrelator("born"),
            ColorCorrelator.dipole("B13", 1, 3),
            ColorCorrelator.dipole("B31", 3, 1),
        ),
    )
    assert len(calls) == 3 * len(plan.sectors)


def test_json_is_exact_primitive_data_and_preserves_imaginary_weights():
    matrix = build_color_correlator_matrix(_plan("d d~ > z"), ColorCorrelator("born"))
    complex_entry = ColorCorrelatorMatrixEntry(
        0, 0, ExactColorCoefficient(Fraction(2**70, 3), Fraction(-2, 7))
    )
    matrix = replace(matrix, entries=(complex_entry,))
    payload = matrix.to_json_dict()
    assert json.loads(json.dumps(payload, allow_nan=False)) == payload
    assert payload["storage"] == "sparse-directed"
    assert payload["includes_color_factor"] is True
    assert payload["entries"] == [
        {
            "left_sector_id": 0,
            "right_sector_id": 0,
            "weight": {"real": [str(2**70), "3"], "imag": ["-2", "7"]},
        }
    ]


@pytest.mark.parametrize("id", ("", "  ", None, 3))
def test_correlator_ids_must_be_nonempty_strings(id):
    with pytest.raises(ValueError, match="ID"):
        ColorCorrelator(id)


def test_duplicate_request_ids_and_incompatible_outputs_are_rejected():
    plan = _plan("d d~ > z")
    with pytest.raises(ValueError, match="unique"):
        build_color_correlator_matrices(
            plan, (ColorCorrelator("x"), ColorCorrelator("x"))
        )
    with pytest.raises(ValueError, match="incompatible"):
        build_color_correlator_matrix(
            plan, ColorCorrelator("x", (EmitGluon(1, -1),), (EmitGluon(1, -2),))
        )
    with pytest.raises(ValueError, match="same number"):
        ColorCorrelator("x", (EmitGluon(1, -1),))
    with pytest.raises(ValueError, match="active coloured"):
        build_color_correlator_matrix(plan, ColorCorrelator.dipole("singlet", 1, 3))


@pytest.mark.parametrize("accuracy", ("lc", "nlc"))
def test_non_full_plans_are_rejected(accuracy):
    with pytest.raises(ValueError, match="full-colour"):
        build_color_correlator_matrix(
            _plan("g g > g g", accuracy=accuracy), ColorCorrelator("born")
        )


@pytest.mark.parametrize(
    "change", ({"truncated": True}, {"trace_reflections_folded": True}, {"sectors": ()})
)
def test_incomplete_or_folded_plans_are_rejected(change):
    with pytest.raises(ValueError, match="complete unfolded"):
        build_color_correlator_matrix(
            replace(_plan("g g > g g"), **change), ColorCorrelator("born")
        )


@pytest.mark.parametrize("ids", ((), (0, 0), (999,), (True,)))
def test_invalid_selected_sector_ids_are_rejected(ids):
    with pytest.raises(ValueError, match="sector IDs"):
        build_color_correlator_matrix(
            _plan("g g > g g"), ColorCorrelator("born"), sector_ids=ids
        )


def test_empty_correlator_request_does_not_touch_the_ordinary_lc_path():
    assert build_color_correlator_matrices(_plan("g g > g g", accuracy="lc"), ()) == ()
