# SPDX-License-Identifier: 0BSD
"""Opt-in whole-result agreement and genuine compact physical rescue."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.loops.physical_counterterms import (
    distinct_flavour_model_virtual_program,
    drell_yan_model_virtual_program,
)
from pyamplicol.loops.stability import (
    ArithmeticSpec,
    NativeModelStabilityProgram,
    NumericalVerificationError,
)


@pytest.fixture(scope="module")
def source():
    symbolica = pytest.importorskip("symbolica")
    pytest.importorskip("_loop_reduction")
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not symbolica.is_licensed() or not path:
        pytest.skip("licensed Symbolica and loop_sm reference required")
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.models.loop_import import load_raw_loop_model

    return build_loop_model(load_raw_loop_model(path))


def points(name):
    path = Path(__file__).resolve().parents[2] / "tools/loop_reference" / name
    return np.asarray(json.loads(path.read_text())["points"], dtype=str)


@pytest.mark.parametrize(
    "requested,higher,rtol",
    [
        (
            ArithmeticSpec("double"),
            (ArithmeticSpec("double-double"), ArithmeticSpec("arbitrary", 70)),
            "1e-10",
        ),
        (
            ArithmeticSpec("double-double"),
            (ArithmeticSpec("arbitrary", 60), ArithmeticSpec("arbitrary", 90)),
            "1e-24",
        ),
        (
            ArithmeticSpec("arbitrary", 50),
            (ArithmeticSpec("arbitrary", 80), ArithmeticSpec("arbitrary", 110)),
            "1e-40",
        ),
    ],
)
def test_verification_is_opt_in_and_preserves_requested_carrier(
    source, requested, higher, rtol
):
    model = drell_yan_model_virtual_program(source, **requested.options)
    program = NativeModelStabilityProgram(
        model, higher, relative_tolerance=rtol, absolute_tolerance="1e-60"
    )
    data = points("drell_yan_qcd.json")[:2]
    exact_model = ["0.1180000000000000000000000000001", "132.507"]
    ordinary = program.evaluate_native(
        data, ["1000000", "2000000"], model_values=exact_model, point_ids=[71, 3]
    )
    assert len(program._views) == 1
    assert all(
        d.accuracy == "not-established" and d.attempts == (requested,)
        for d in ordinary.diagnostics
    )
    checked = program.evaluate_native(
        data,
        ["1000000", "2000000"],
        model_values=exact_model,
        point_ids=[71, 3],
        verify=True,
    )
    assert checked.point_ids == checked.requested_point_ids == (71, 3)
    assert len(program._views) == 2
    for d in checked.diagnostics:
        assert (
            d.status == "success"
            and d.accuracy == "empirical-cross-precision-agreement"
        )
        assert d.output_arithmetic == d.evaluated_arithmetic == requested
        assert d.verification_pair == (requested, higher[0])
        assert len(d.reduction_statuses) == 2
    for actual, expected in zip(
        [checked.born, *checked.component_batches, checked.totals],
        [ordinary.born, *ordinary.component_batches, ordinary.totals],
        strict=True,
    ):
        assert actual.precision_bits == expected.precision_bits
        assert actual.agreement_rows(expected, "0", "1e-150") == [(71, True), (3, True)]
    empty = program.evaluate_native(data[:0], [], model_values=exact_model, verify=True)
    assert empty.point_ids == () and empty.diagnostics == () and empty.totals.rows == 0


def test_output_rounding_bound_cannot_be_certified_by_high_precision_alone(source):
    program = NativeModelStabilityProgram(
        drell_yan_model_virtual_program(source),
        [ArithmeticSpec("double-double"), ArithmeticSpec("arbitrary", 70)],
        relative_tolerance="1e-25",
        absolute_tolerance="1e-60",
    )
    data = points("drell_yan_qcd.json")[:1]
    kwargs = dict(model_values=["0.118", "132.507"], verify=True, point_ids=[9])
    partial = program.evaluate_native(data, ["1000000"], allow_partial=True, **kwargs)
    assert partial.point_ids == () and partial.born.rows == 0
    d = partial.diagnostics[0]
    assert d.status == "failed" and d.accuracy == "not-established"
    assert "output tolerance" in d.detail
    assert d.verification_pair is None and d.evaluated_arithmetic is None
    with pytest.raises(NumericalVerificationError) as error:
        program.evaluate_native(data, ["1000000"], **kwargs)
    assert error.value.result.requested_point_ids == (9,)
    assert error.value.result.point_ids == ()


def test_real_near_forward_reduction_failure_rescues_only_original_failing_point(
    source,
):
    program = NativeModelStabilityProgram(
        distinct_flavour_model_virtual_program(source),
        [
            ArithmeticSpec("double-double"),
            ArithmeticSpec("arbitrary", 70),
            ArithmeticSpec("arbitrary", 100),
        ],
        relative_tolerance="1e-8",
        absolute_tolerance="1e-12",
    )
    # Exact massless Pythagorean kinematics, close to forward scattering.
    # This is an empirical precision-stability control, not a reference value.
    near = np.asarray(
        [
            ["5000000000.5", "0", "0", "5000000000.5"],
            ["5000000000.5", "0", "0", "-5000000000.5"],
            ["5000000000.5", "100000", "0", "4999999999.5"],
            ["5000000000.5", "-100000", "0", "-4999999999.5"],
        ]
    )
    data = np.stack([points("uudd_qcd.json")[0], near])
    kwargs = dict(model_values=["0.118", "173"], point_ids=[11, 93])
    partial = program.evaluate_native(
        data, ["1000000"] * 2, allow_partial=True, **kwargs
    )
    assert partial.point_ids == (11,)
    assert partial.diagnostics[1].status == "failed"
    assert any(
        "DegenerateGram" in failure.detail
        for _, failure in partial.diagnostics[1].failures
    )
    checked = program.evaluate_native(data, ["1000000"] * 2, verify=True, **kwargs)
    assert checked.point_ids == (11, 93)
    normal, rescued = checked.diagnostics
    assert normal.attempts == (
        ArithmeticSpec("double"),
        ArithmeticSpec("double-double"),
    )
    assert rescued.status == "rescued"
    assert rescued.verification_pair == (
        ArithmeticSpec("arbitrary", 70),
        ArithmeticSpec("arbitrary", 100),
    )
    assert rescued.evaluated_arithmetic == ArithmeticSpec("arbitrary", 100)
    assert rescued.output_arithmetic == ArithmeticSpec("double")
    assert len(rescued.failures) == 2
    assert all(
        buffer.row_ids == [11, 93]
        and buffer.precision_bits == 53
        and not buffer.nonfinite_rows()
        for buffer in [checked.born, *checked.component_batches, checked.totals]
    )
    assert checked.totals.take_rows([0]).agreement_rows(
        partial.totals, "0", "1e-150"
    ) == [(11, True)]
    # Reuse with a different exact model tuple: no cached rounded parameter state.
    again = program.evaluate_native(
        data[:1],
        ["2000000"],
        model_values=["0.120000000000000000000001", "180"],
        verify=True,
        point_ids=[11],
    )
    assert again.diagnostics[0].status == "success"
    assert again.born.agreement_rows(checked.born.take_rows([0]), "1e-8", "0") == [
        (11, False)
    ]


def test_verification_configuration_rejects_non_increasing_precision(source):
    model = drell_yan_model_virtual_program(source)
    with pytest.raises(ValueError, match="strictly increase"):
        NativeModelStabilityProgram(
            model,
            [ArithmeticSpec("double")],
            relative_tolerance="1e-8",
            absolute_tolerance="0",
        )
    with pytest.raises(ValueError, match="tolerances"):
        NativeModelStabilityProgram(
            model, [], relative_tolerance="0", absolute_tolerance="0"
        )
    program = NativeModelStabilityProgram(
        model, [], relative_tolerance="1e-8", absolute_tolerance="0"
    )
    with pytest.raises(ValueError, match="at least two"):
        program.evaluate_native(
            points("drell_yan_qcd.json")[:1],
            ["1000000"],
            model_values=["0.118", "132.507"],
            verify=True,
        )


def test_original_layout_is_validated_before_any_retry_gather(source):
    program = NativeModelStabilityProgram(
        drell_yan_model_virtual_program(source),
        [],
        relative_tolerance="1e-8",
        absolute_tolerance="0",
    )
    data = points("drell_yan_qcd.json")[:2]
    for ids in ([9], [9, 9], [-1, 2], [1.0, 2], [1, 2, 3]):
        with pytest.raises(ValueError, match="distinct nonnegative"):
            program.evaluate_native(
                data, ["1000000"] * 2, model_values=["0.118", "132.507"], point_ids=ids
            )
    for scales in (["1000000"], ["1000000"] * 3):
        with pytest.raises(ValueError, match="scale shape"):
            program.evaluate_native(data, scales, model_values=["0.118", "132.507"])
    with pytest.raises(ValueError, match="four components"):
        program.evaluate_native(
            data[:, :, :3], ["1000000"] * 2, model_values=["0.118", "132.507"]
        )


@pytest.mark.parametrize("arithmetic", ["double", "double-double"])
def test_partial_preparation_isolates_finite_original_state_overflow(
    source, arithmetic
):
    from pyamplicol.loops.numerical_lanes import LaneFailures

    model = drell_yan_model_virtual_program(source, arithmetic=arithmetic)
    ordinary = points("drell_yan_qcd.json")[:1]
    huge = np.asarray(
        [
            ["5e400", "0", "0", "5e400"],
            ["5e400", "0", "0", "-5e400"],
            ["5e400", "0", "0", "5e400"],
            ["5e400", "0", "0", "-5e400"],
        ]
    )
    mixed = np.stack([ordinary[0], huge])
    result = model.evaluate_ordered_partial_native(
        mixed, ["1000000"] * 2, model_values=["0.118", "132.507"], point_ids=[7, 91]
    )
    assert result.requested_point_ids == (7, 91) and result.batch.point_ids == [7]
    assert {failure.point_id for failure in result.failures} == {91}
    assert all(
        failure.stage == "external state preparation" for failure in result.failures
    )
    assert result.totals.nonfinite_rows() == []
    with pytest.raises(ValueError, match="external state lane"):
        model.evaluate_ordered_native(
            mixed, ["1000000"] * 2, model_values=["0.118", "132.507"]
        )
    empty = model.evaluate_ordered_partial_native(
        mixed[1:], ["1000000"], model_values=["0.118", "132.507"], point_ids=[91]
    )
    assert empty.batch.point_ids == [] and empty.totals.rows == 0
    # A malformed original scale is validated even if its state would be evicted.
    with pytest.raises(ValueError, match="finite decimal"):
        model.evaluate_ordered_partial_native(
            mixed,
            ["1000000", "NaN"],
            model_values=["0.118", "132.507"],
            point_ids=[7, 91],
        )
    invalid = mixed.copy()
    invalid[1, 0, 0] = "NaN"
    with pytest.raises(ValueError, match="finite decimal"):
        model.evaluate_ordered_partial_native(
            invalid, ["1000000"] * 2, model_values=["0.118", "132.507"]
        )
    if arithmetic == "double":
        # Original decimal inputs recover finite native states/currents in AP.
        # This assertion deliberately makes no full reduction accuracy claim.
        precise = drell_yan_model_virtual_program(
            source, arithmetic="arbitrary", precision=80
        )
        failures = LaneFailures()
        currents = precise.bare.prepare_currents_native(
            mixed,
            ["1000000"] * 2,
            model_values=["0.118", "132.507"],
            point_ids=[7, 91],
            _failures=failures,
        )
        assert failures.entries == [] and currents.batch.point_ids == [7, 91]
        assert currents.born.nonfinite_rows() == []
        assert all(value.nonfinite_rows() == [] for value in currents.loops)
