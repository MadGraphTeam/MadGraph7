# SPDX-License-Identifier: 0BSD
"""Complete retained-owner controls, without regenerating process graphs.

The retained-session runner installs copied exact prepared records and an
unchanged physical reference in RETAINED_CONTROL. Ordinary test discovery skips
this expensive control when no producer session is explicitly supplied.
"""

import numpy as np
import pytest

from pyamplicol.api.virtual_results import VirtualArithmeticSpec, VirtualStabilityPolicy
from pyamplicol.loops.stability import ArithmeticSpec, NativeModelStabilityProgram

RETAINED_CONTROL = None


@pytest.fixture(scope="module")
def retained():
    if RETAINED_CONTROL is None:
        pytest.skip("supply an actual retained complete source owner")
    return RETAINED_CONTROL


def _buffers(result):
    return (result.born, *result.component_batches, result.totals)


def _compare(result, reference, ids, rtol, atol):
    assert len(_buffers(result)) == len(_buffers(reference))
    for actual, expected in zip(_buffers(result), _buffers(reference), strict=True):
        expected = expected.take_rows(list(range(len(ids))), row_ids=ids)
        assert actual.agreement_rows(expected, rtol, atol) == [
            (point, True) for point in ids
        ]


@pytest.mark.parametrize(
    "arithmetic,precision,rtol,atol",
    [
        ("double", None, "1e-8", "1e-12"),
        ("double-double", None, "1e-22", "1e-26"),
        ("arbitrary", 80, "1e-58", "1e-62"),
    ],
)
def test_complete_real_mass_source_fields_preserve_partial_results(
    retained, arithmetic, precision, rtol, atol
):
    program = retained["ordinary"].bind(
        backend=retained["backend"],
        arithmetic=arithmetic,
        precision=precision,
        parameters=retained["parameters"],
    )
    assert program.external_fields and program.field_currents
    result = program.evaluate_ordered_partial_native(
        retained["points"], retained["scales"], point_ids=retained["ids"]
    )
    assert not result.failures
    assert result.requested_point_ids == tuple(retained["ids"])
    _compare(result, retained["reference"], retained["ids"], rtol, atol)
    retained["views"][arithmetic] = program


def test_complete_global_source_rescue_replays_exact_snapshot_and_components(retained):
    policy = VirtualStabilityPolicy(
        (
            VirtualArithmeticSpec("double-double"),
            VirtualArithmeticSpec("arbitrary", 60),
            VirtualArithmeticSpec("arbitrary", 80),
        ),
        "1e-8",
        "1e-12",
    )
    program = retained["guarded"].bind(
        backend=retained["backend"],
        arithmetic="double",
        parameters=retained["parameters"],
        stability=policy,
    )
    owner = program.bare.model_state
    snapshot = owner.snapshot()
    assert snapshot.binding_failure_stage == "real model input type"
    partial = program.evaluate_ordered_partial_native(
        retained["points"], retained["scales"], point_ids=retained["ids"]
    )
    assert partial.batch.point_ids == []
    assert {f.point_id for f in partial.failures} == set(retained["ids"])
    assert all(f.stage == "real model input type" for f in partial.failures)
    assert all(value.rows == 0 for value in _buffers(partial))
    stability = NativeModelStabilityProgram(
        program,
        [
            ArithmeticSpec("double-double"),
            ArithmeticSpec("arbitrary", 60),
            ArithmeticSpec("arbitrary", 80),
        ],
        relative_tolerance=policy.relative_tolerance,
        absolute_tolerance=policy.absolute_tolerance,
    )
    prepare = program.bare.prepare_currents_native
    captured = []

    def update_after_capture(*args, **kwargs):
        captured.append(kwargs["model_snapshot"])
        owner.update({"aS": "0.12"})
        return prepare(*args, **kwargs)

    program.bare.prepare_currents_native = update_after_capture
    try:
        result = stability.evaluate_native(
            retained["points"],
            retained["scales"],
            point_ids=retained["ids"],
            verify=True,
        )
    finally:
        program.bare.prepare_currents_native = prepare
    assert result.point_ids == tuple(retained["ids"])
    _compare(result, retained["reference"], retained["ids"], "1e-8", "1e-12")
    for diagnostic in result.diagnostics:
        assert diagnostic.status == "rescued"
        assert diagnostic.verification_pair == (
            ArithmeticSpec("arbitrary", 60),
            ArithmeticSpec("arbitrary", 80),
        )
        assert len(diagnostic.attempts) == 4
        assert [failure.stage for _, failure in diagnostic.failures] == [
            "real model input type"
        ] * 2
    assert captured == [snapshot]
    assert owner.snapshot().revision == snapshot.revision + 1
    assert dict(owner.snapshot().overrides)["aS"] == ("0.12", "0")
    for view in stability._views.values():
        assert view.bare.model_state.prepared is owner.prepared
    updated_reference = retained["views"]["arbitrary"]
    updated_reference.bare.model_state.update({"aS": "0.12"})
    expected = updated_reference.evaluate_ordered_partial_native(
        retained["points"][:1],
        retained["scales"][:1],
        point_ids=retained["ids"][:1],
    )
    next_result = stability.evaluate_native(
        retained["points"][:1],
        retained["scales"][:1],
        point_ids=retained["ids"][:1],
        verify=True,
    )
    _compare(next_result, expected, retained["ids"][:1], "1e-8", "1e-12")
    empty = stability.evaluate_native(
        retained["points"][:0], [], point_ids=[], verify=True
    )
    assert empty.point_ids == () and empty.diagnostics == ()
    bad = np.array(retained["points"], copy=True, dtype=object)
    bad[0, 0, 0] = "not-a-number"
    with pytest.raises(ValueError):
        stability.evaluate_native(
            bad, retained["scales"], point_ids=retained["ids"], verify=True
        )
    with pytest.raises(ValueError):
        program.rebind(arithmetic="arbitrary", precision=80)
    replacement = VirtualStabilityPolicy(
        (VirtualArithmeticSpec("arbitrary", 100),), "1e-20", "1e-30"
    )
    higher = program.rebind(arithmetic="arbitrary", precision=80, stability=replacement)
    assert higher.bare.model_state.stability is replacement
    assert program.bare.model_state.stability is policy
    retained["rescued"] = result
    retained["guarded_view"] = program


def test_late_source_field_column_failure_keeps_other_original_points(retained):
    program = retained["views"]["double"]
    scale_view = program.source_scale_view
    original = scale_view._evaluator
    backend = retained["backend"]
    failed = retained["ids"][1]
    column = 2 * program.field_currents[0].source_scalar_column

    class FailedFieldColumn:
        def evaluate_batch(self, values):
            result = original.evaluate_batch(values)
            valid = [i for i, point in enumerate(result.row_ids) if point != failed]
            bad = backend.NativeBatch.from_exact_decimal(
                [("1e400", "0")], 1, 1, row_ids=[failed]
            )
            selected = backend.NativeBatch.merge_rows(
                result.row_ids,
                [
                    result.take_columns([column]).take_rows(valid),
                    bad,
                ],
            )
            return backend.NativeBatch.concat_columns(
                [
                    selected if i == column else result.take_columns([i])
                    for i in range(result.columns)
                ]
            )

    scale_view._evaluator = FailedFieldColumn()
    try:
        result = program.evaluate_ordered_partial_native(
            retained["points"], retained["scales"], point_ids=retained["ids"]
        )
    finally:
        scale_view._evaluator = original
    assert result.batch.point_ids == retained["ids"][:1]
    assert {failure.point_id for failure in result.failures} == {failed}
    assert {failure.stage for failure in result.failures} == {
        "scale-dependent scalar values and active diagnostics"
    }
    _compare(result, retained["reference"], retained["ids"][:1], "1e-8", "1e-12")


@pytest.mark.parametrize(
    "arithmetic,precision,rtol,atol",
    [
        ("double", None, "1e-8", "1e-12"),
        ("double-double", None, "1e-22", "1e-26"),
        ("arbitrary", 80, "1e-58", "1e-62"),
    ],
)
def test_dependency_massive_state_failure_compacts_whole_point(
    retained, arithmetic, precision, rtol, atol
):
    leg = retained.get("massive_vector_leg")
    if leg is None:
        pytest.skip("requires the retained massive-vector control")
    program = retained["ordinary"].bind(
        backend=retained["backend"],
        arithmetic=arithmetic,
        precision=precision,
        parameters=retained["parameters"],
    )
    assert program.bare.state_layout.legs[leg].helicities == (-1, 0, 1)
    points = np.array(retained["points"], dtype=object, copy=True)
    # FeynKit deliberately has no longitudinal rest-state convention. This
    # diagnostic point exercises that real dependency failure before reduction;
    # it is not another physical scattering/reference configuration.
    points[1, leg, 1:] = "0"
    result = program.evaluate_ordered_partial_native(
        points, retained["scales"], point_ids=retained["ids"]
    )
    assert result.batch.point_ids == retained["ids"][:1]
    assert {failure.point_id for failure in result.failures} == {retained["ids"][1]}
    assert {failure.stage for failure in result.failures} == {
        "external state preparation"
    }
    assert all("longitudinal" in failure.detail.lower() for failure in result.failures)
    _compare(result, retained["reference"], retained["ids"][:1], rtol, atol)
