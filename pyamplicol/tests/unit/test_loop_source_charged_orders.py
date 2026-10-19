# SPDX-License-Identifier: 0BSD
"""Full-source charged R2 enters QCD corrections to mixed SM Born orders."""

import json
import os
from decimal import Decimal, localcontext
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def charged_plan(tmp_path_factory):
    symbolica = pytest.importorskip("symbolica")
    pytest.importorskip("_loop_reduction")
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not path or not symbolica.is_licensed():
        pytest.skip("licensed source generation and loop_sm reference required")
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.model_compatibility import prepare_loop_sm_scalar_declarations
    from pyamplicol.loops.portable import write_loop_pacbin
    from pyamplicol.models.loop_import import load_raw_loop_model

    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/uudd_mixed_sm_qcd.json"
    )
    reference = json.loads(fixture.read_text())
    # Full Wolfenstein CKM matches the provider. The default restriction instead
    # removes off-diagonal CKM entries and is a different physical input.
    raw = prepare_loop_sm_scalar_declarations(
        load_raw_loop_model(path, restriction="full")
    ).derived
    prepared = prepare_loop_request_model(
        raw,
        born_order_bounds={"QCD": (0, 2), "QED": (0, 2)},
        loop_order_bounds={"QCD": (0, 4), "QED": (0, 2)},
        fixed_parameters=reference["fixed_parameters"],
    )
    program = prepared.compile(
        (2, -2),
        (1, -1),
        selected_corrections=reference["selected_corrections"],
        parameters=reference["parameters"],
        coefficient_engine="jit",
        reduction_engine="fast",
    )
    layout = program.virtual.bare.order_layout
    assert {item.order for item in layout.born} == {
        (("QCD", 4),),
        (("QCD", 2), ("QED", 2)),
        (("QED", 4),),
    }
    mixed = next(
        item for item in layout.virtual if item.order == (("QCD", 4), ("QED", 2))
    )
    assert {(pair.born, pair.loop) for pair in mixed.pairs} == {
        ((("QCD", 2),), (("QCD", 2), ("QED", 2))),
        ((("QED", 2),), (("QCD", 4),)),
    }
    assert layout.correction_names == ("QCD",)
    destination = tmp_path_factory.mktemp("charged-source") / "loop.pacbin"
    write_loop_pacbin(program.virtual, destination, resolved=True)
    return destination, reference


def _close(a, b, tolerance):
    a, b = Decimal(str(a)), Decimal(str(b))
    assert abs(a - b) <= Decimal(tolerance) * max(abs(b), Decimal("1e-10")), (a, b)


@pytest.mark.parametrize(
    "arithmetic,precision,reducer,tolerance,bit",
    [
        ("double", None, "fast", "5e-11", 20),
        ("double-double", None, "generic", "2e-25", 70),
        ("arbitrary", 70, "generic", "1e-56", 170),
    ],
)
def test_saved_charged_orders_reference_resolved_and_exact_inputs(
    charged_plan, monkeypatch, arithmetic, precision, reducer, tolerance, bit
):
    from _loop_reduction import NativeLoopProgram

    from .test_loop_source_portable import _forbid_source

    path, reference = charged_plan
    _forbid_source(monkeypatch)
    runtime = NativeLoopProgram(
        str(path), arithmetic=arithmetic, precision=precision, reduction_engine=reducer
    )
    metadata = json.loads(runtime.metadata_json())
    axes = metadata["order_layout"]
    assert axes["correction_names"] == ["QCD"]
    nborn, nvirtual, nhelicity = len(axes["born"]), len(axes["virtual"]), 16
    points = reference["points"]
    scales = [reference["mu_r2"]] * len(points)
    inclusive = runtime.evaluate(points, scales, point_ids=[91, 7, 30])
    resolved = runtime.evaluate(points, scales, point_ids=[91, 7, 30], resolved=True)
    assert inclusive.point_ids == [91, 7, 30]
    assert resolved.point_ids == [91, 7, 30]
    assert resolved.born.row_ids == list(range(3 * nhelicity))
    for result in (inclusive, resolved):
        assert all(
            status == "success"
            for statuses in result.statuses
            for _, status in statuses
        )
        assert result.r2_strategy == "model"
    born = inclusive.born.to_decimal()
    virtual = inclusive.component_batches[0].to_decimal()
    born_h = resolved.born.to_decimal()
    virtual_h = resolved.component_batches[0].to_decimal()
    assert len(born) == len(points) * nborn
    assert len(virtual) == len(points) * nvirtual * 3
    assert all(Decimal(z[1]) == 0 for z in (*born, *virtual, *born_h, *virtual_h))
    with localcontext() as context:
        context.prec = 110
        for p, row in enumerate(reference["rows"]):
            for kind, data, data_h, width, provider_key in (
                ("born", born, born_h, 1, "born_squared_orders"),
                ("virtual", virtual, virtual_h, 3, "virtual_squared_orders"),
            ):
                sectors = axes[kind]
                for s, sector in enumerate(sectors):
                    order = dict(sector["order"])
                    key = next(
                        k
                        for k, value in reference[provider_key].items()
                        if {name: power for name, power in value.items() if power}
                        == order
                    )
                    expected = (
                        [row["born"][key]]
                        if kind == "born"
                        else [
                            row["full"][key][pole]
                            for pole in ("finite", "single_pole", "double_pole")
                        ]
                    )
                    for k in range(width):
                        index = (p * len(sectors) + s) * width + k
                        # The independent provider is binary64. All carriers use
                        # its common tolerance; this is not a 60-digit oracle.
                        _close(data[index][0], expected[k], "5e-10")
                        summed = sum(
                            Decimal(
                                data_h[
                                    ((p * nhelicity + h) * len(sectors) + s) * width + k
                                ][0]
                            )
                            for h in range(nhelicity)
                        )
                        _close(summed, data[index][0], tolerance)
        # A below-binary64 input increment must survive the saved source owner
        # and affect a known QCD Born sector in both higher arithmetic modes.
        old_alpha = Decimal(reference["parameters"]["aS"])
        alpha = old_alpha * (1 + Decimal(2) ** -bit)
        runtime.source_state.update({"aS": [str(alpha), "0"]})
        changed = runtime.evaluate(points[:1], scales[:1]).born.to_decimal()
        pure = next(
            i
            for i, sector in enumerate(axes["born"])
            if dict(sector["order"]) == {"QCD": 4}
        )
        ratio = Decimal(changed[pure][0]) / Decimal(born[pure][0])
        assert ratio != 1
        _close(ratio, (alpha / old_alpha) ** 2, tolerance)
