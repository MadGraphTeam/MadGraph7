# SPDX-License-Identifier: 0BSD
"""Saved normalized integral control; this fixture is not a scattering model."""

import io
import json
from decimal import Decimal, localcontext

import pytest


def _bubble_plan(destination, *, rank=4):
    from symbolica import E, Expression, S

    from pyamplicol.generation.evaluator_container import (
        PacbinMemberKind,
        PacbinMemberSource,
        write_pacbin_atomic,
    )
    from pyamplicol.loops.physical import _saved
    from pyamplicol.loops.physical_process import _reduction_numerators
    from pyamplicol.loops.portable import _PLAN_MEMBER

    payloads, references = {}, {}

    def state(width, expressions):
        parameters = [S(f"portable_rank_control::x{i}") for i in range(width)]
        values = expressions(parameters) if callable(expressions) else expressions
        payload = _saved(
            [
                value if isinstance(value, Expression) else E(str(value))
                for value in values
            ],
            parameters,
        )
        if payload not in references:
            name = f"evaluators/exact-{len(references)}.state"
            references[payload] = name
            payloads[name] = payload
        return references[payload]

    def group(schedule, columns):
        return {
            "schedules": [schedule],
            "input_columns": [columns],
            "representatives": [0],
            "representative_for": [0],
        }

    powers = (
        [[4 - i, i, 0, 0, 0, 0] for i in range(5)]
        if rank == 4
        else [[rank, 0, 0, 0, 0, 0]]
    )
    coefficients = [1, -4, 6, -4, 1] if rank == 4 else [1]
    numerator = _reduction_numerators(powers, [f"c{i}" for i in range(len(powers))])
    # The only loop schedule returns a; its family projector supplies the raw
    # monomial coefficients of a*(q0-q1)^4. All effective-rule blocks are zero.
    data = {
        "schema": 3,
        "r2_strategy": "model",
        "dimensional_numerators": False,
        "external_pdgs": [9000001, 9000001],
        "state_legs": [
            {
                "index": i,
                "kind": "scalar",
                "incoming": i == 0,
                "massless": True,
                "offset": i,
            }
            for i in range(2)
        ],
        "helicities": [[0, 0]],
        "initial_average": 1,
        "model_inputs": ["a"],
        "model_count": 1,
        "model_binding": {"kind": "legacy"},
        "bound_positions": {"a": 10},
        "order_layout": {
            "born": [{"order": [["QCD", 2]], "pairs": [[[["QCD", 1]], [["QCD", 1]]]]}],
            "virtual": [
                {
                    "order": [["QCD", 4]],
                    "pairs": [{"born": [["QCD", 1]], "loop": [["QCD", 3]]}],
                }
            ],
            "correction_names": ["QCD"],
        },
        "schedules": [
            {
                "states": [state(0, [1])],
                "input_slots": [[]],
                "parameter_count": 0,
                "output_slots": [0],
            },
            {
                "states": [state(1, lambda x: x)],
                "input_slots": [[0]],
                "parameter_count": 1,
                "output_slots": [1],
            },
        ],
        "born": group(0, []),
        "loops": group(1, [10]),
        "families": [
            {
                "index": 0,
                "graph_indices": [0],
                "sector_indices": [0],
                "powers": powers,
                "propagators": 2,
                "mass_columns": [],
                "kinematics": {
                    "context": "raw",
                    "mass_output_count": 2,
                    "on_shell": [],
                },
                "exact_state": state(8, [0, 0, 0, 0, 0, 1, 0, 0, 0, -1, -1, 0]),
                "mass_state": state(0, [0, 0]),
                "projector_state": state(2, lambda x: [c * x[1] for c in coefficients]),
                "numerators": numerator,
                "rational_numerators": ["0"] * 3,
            }
        ],
        "bare_evaluators": {
            "model_evaluator": state(1, lambda x: x),
            "born_projector": state(1, [1]),
            "postprocess": state(3, lambda x: x),
            "zero_poles": state(0, [0] * 3),
            "sum_order_values": state(4, lambda x: x),
        },
        "currents": [],
        "fields": [],
        "model_evaluators": {
            "projector": state(1, [0] * 9),
            "assemble": state(
                12,
                lambda x: (
                    [x[i] + x[i + 3] + x[i + 6] + x[i + 9] for i in range(3)]
                    + [x[i + 3] + x[i + 9] for i in range(3)]
                ),
            ),
            "sum_order_values": state(19, lambda x: x),
        },
    }
    members = [
        PacbinMemberSource(
            _PLAN_MEMBER,
            PacbinMemberKind.LOOP_RUNTIME_PLAN,
            io.BytesIO(json.dumps(data).encode()),
        )
    ]
    members.extend(
        PacbinMemberSource(
            name, PacbinMemberKind.SYMBOLICA_EXACT_STATE, io.BytesIO(payload)
        )
        for name, payload in payloads.items()
    )
    write_pacbin_atomic(destination, members)
    return destination


@pytest.fixture(scope="module")
def bubble_plans(tmp_path_factory):
    symbolica = pytest.importorskip("symbolica")
    pytest.importorskip("_loop_reduction")
    if not symbolica.is_licensed():
        pytest.skip("licensed exact evaluator preparation required")
    root = tmp_path_factory.mktemp("portable-rank")
    return _bubble_plan(root / "rank4.pacbin"), _bubble_plan(
        root / "rank5.pacbin", rank=5
    )


@pytest.mark.parametrize(
    "arithmetic,precision,engine,tolerance",
    [
        ("double", None, "fast", "2e-12"),
        ("double-double", None, "generic", "2e-25"),
        ("arbitrary", 80, "generic", "1e-60"),
    ],
)
def test_saved_rank_four_bubble_uses_backend_domain(
    bubble_plans, arithmetic, precision, engine, tolerance
):
    from _loop_reduction import NativeLoopProgram, PortableLoopPlan

    valid, unsupported = bubble_plans
    # Cold loading does not impose a guessed reducer-domain ceiling. Only the
    # selected reducer binding can reject the actual rank-five polynomial.
    for path in bubble_plans:
        assert (
            json.loads(PortableLoopPlan(str(path)).metadata_json())["families"][0][
                "propagators"
            ]
            == 2
        )
    options = dict(arithmetic=arithmetic, precision=precision, reduction_engine=engine)
    runtime = NativeLoopProgram(str(valid), **options)
    momenta = [[["1", "0", "0", "1"], ["1", "0", "0", "1"]]] * 2
    result = runtime.evaluate(momenta, ["1", "4"], ["-2"])
    assert result.point_ids == [0, 1]
    assert result.statuses == [[(0, "success")], [(1, "success")]]
    values = result.component_batches[0].to_decimal()
    # Same independent Feynman-parameter identity as integration_rank_n2.py:
    # B(5-eps,1-eps)/B(1-eps,1-eps)=1/5+29eps/300+O(eps²).
    with localcontext() as context:
        context.prec = 110
        for row, scale in enumerate((Decimal(1), Decimal(4))):
            expected = (Decimal(149) / 300 + scale.ln() / 5, Decimal(1) / 5, Decimal(0))
            for pair, value in zip(
                values[3 * row : 3 * row + 3], expected, strict=True
            ):
                assert abs(Decimal(pair[0]) + 2 * value) < Decimal(tolerance)
                assert abs(Decimal(pair[1])) < Decimal(tolerance)
    with pytest.raises(ValueError, match="rank"):
        NativeLoopProgram(str(unsupported), **options)
