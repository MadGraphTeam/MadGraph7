# SPDX-License-Identifier: 0BSD
"""Projected weights survive graph-free transport and exact arithmetic rebinding."""

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.loop_fast


@pytest.fixture(scope="module")
def source(loop_sm_ufo, loop_native):
    symbolica = pytest.importorskip("symbolica")
    path = str(loop_sm_ufo)
    if not symbolica.is_licensed():
        pytest.skip("licensed loop model required")
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.models.loop_import import load_raw_loop_model

    return build_loop_model(load_raw_loop_model(path))


@pytest.mark.parametrize("accuracy", ["full", "lc", "nlc"])
def test_projected_model_transport(
    loop_native, source, accuracy, tmp_path, monkeypatch
):
    native = loop_native

    from pyamplicol.loops import physical_counterterms, physical_process
    from pyamplicol.loops.physical_counterterms import drell_yan_model_virtual_program
    from pyamplicol.loops.portable import PortableLoopPlan, write_loop_pacbin

    original = drell_yan_model_virtual_program(
        source, formal_color=True, color_accuracy=accuracy, reduction_engine="fast"
    )
    path = tmp_path / "projected.pacbin"
    write_loop_pacbin(original, path, resolved=True)
    plan = PortableLoopPlan.open(path)
    description = json.loads(plan._native.metadata_json())["color_projection"]
    assert description == {
        "accuracy": accuracy,
        "born_degrees": list(original.color_projection.born_degrees),
        "virtual_degrees": list(original.color_projection.virtual_degrees),
    }
    fixture = (
        Path(__file__).resolve().parents[2] / "tools/loop_reference/drell_yan_qcd.json"
    )
    points = np.asarray(json.loads(fixture.read_text())["points"], dtype=str)[:1]

    def forbidden(*args, **kwargs):
        raise AssertionError("retained projected view repeated cold construction")

    for owner, name in (
        (physical_process.NativeBareProcessProgram, "__init__"),
        (physical_counterterms.NativeModelVirtualProgram, "__init__"),
        (physical_process, "_saved"),
        (physical_counterterms, "_saved"),
        (physical_counterterms, "_saved_scalar_pairs"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    for arithmetic, precision, engine, tolerance, bits in (
        ("double", None, "fast", "1e-13", 30),
        ("double-double", None, "generic", "1e-27", 70),
        ("arbitrary", 80, "generic", "1e-68", 180),
    ):
        options = dict(
            arithmetic=arithmetic, precision=precision, reduction_engine=engine
        )
        expected_view = original.rebind(**options)
        view = plan.bind(**options)
        assert expected_view.bare._formal_color_projected
        for candidate in (expected_view, view, view.rebind(**options)):
            assert candidate.color_accuracy == candidate.bare.color_accuracy == accuracy
            assert (
                candidate.color_projection.born_degrees
                == original.color_projection.born_degrees
            )
            assert (
                candidate.color_projection.virtual_degrees
                == original.color_projection.virtual_degrees
            )
        expected = expected_view.evaluate_ordered_native(
            points, ["1000000"], model_values=["0.118", "132.507"]
        )
        actual = view.evaluate_ordered_native(
            points, ["1000000"], model_values=["0.118", "132.507"], point_ids=[71]
        )
        shared = native.NativeLoopProgram(str(path), **options)
        inclusive = shared.evaluate(
            points.tolist(), ["1000000"], model_values=["0.118", "132.507"]
        )
        resolved = shared.evaluate(
            points.tolist(),
            ["1000000"],
            model_values=["0.118", "132.507"],
            resolved=True,
        )
        for summed, individual in zip(
            (inclusive.born, *inclusive.component_batches),
            (resolved.born, *resolved.component_batches),
            strict=True,
        ):
            with localcontext() as context:
                context.prec = 100
                values = individual.to_decimal()
                for column, reference in enumerate(summed.to_decimal()):
                    for part in range(2):
                        total = sum(
                            Decimal(values[row * individual.columns + column][part])
                            for row in range(individual.rows)
                        )
                        expected_part = Decimal(reference[part])
                        assert abs(total - expected_part) <= Decimal(tolerance) * max(
                            abs(expected_part), Decimal(1)
                        )
        for left, right in zip(
            (actual.born, *actual.component_batches),
            (expected.born, *expected.component_batches),
            strict=True,
        ):
            assert all(
                ok
                for _, ok in left.agreement_rows(
                    right.take_rows([0], [71]), tolerance, tolerance
                )
            )
        with localcontext() as context:
            context.prec = 100
            alpha = str(Decimal("0.118") + Decimal(2) ** -bits)
        changed = view.evaluate_ordered_native(
            points, ["1000000"], model_values=[alpha, "132.507"]
        )
        reference = expected_view.evaluate_ordered_native(
            points, ["1000000"], model_values=[alpha, "132.507"]
        )
        assert changed.virtual.to_decimal() != actual.virtual.to_decimal()
        assert all(
            ok
            for _, ok in changed.virtual.agreement_rows(
                reference.virtual, tolerance, tolerance
            )
        )
        assert (
            view.evaluate_ordered_native(
                points[:0], [], model_values=[alpha, "132.507"]
            ).virtual.rows
            == 0
        )
