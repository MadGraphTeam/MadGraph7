# SPDX-License-Identifier: 0BSD
"""Full-source real-mass Gmu bare control; no UV/R2 completion claim."""

from __future__ import annotations

import json
import os
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def source():
    s = pytest.importorskip("symbolica")
    if not s.is_licensed():
        pytest.skip("source graph generation requires licensed Symbolica")
    pytest.importorskip("_loop_reduction")
    path = os.environ.get("PYAMPLICOL_EW_UFO_REFERENCE")
    if not path:
        pytest.skip("set PYAMPLICOL_EW_UFO_REFERENCE to the full Gmu UFO")
    from pyamplicol.loops.covariant_model import build_covariant_structure
    from pyamplicol.models.loop_import import load_raw_loop_model

    zeros = (
        "aEW0M1",
        "MD",
        "MU",
        "MS",
        "MC",
        "MB",
        "Me",
        "MM",
        "MTA",
        "lamWS",
        "AWS",
        "rhoWS",
        "etaWS",
        "ymdo",
        "ymup",
        "yms",
        "ymc",
        "ymb",
        "yme",
        "ymm",
        "ymtau",
    )
    ones = ("epsUV", "epsIR", "LogSwitch", "GfRedefinitionChoice")
    return build_covariant_structure(
        load_raw_loop_model(path, restriction="full"),
        compatibility="gmu_particle_anti",
        fixed_parameters={**dict.fromkeys(zeros, "0"), **dict.fromkeys(ones, "1")},
        covariant_multiplets={
            23: [23, 250, 9000002, -9000002],
            24: [24, 251, 9000003, -9000004],
            -24: [-24, -251, 9000004, -9000003],
        },
    )


@pytest.fixture(scope="module")
def points():
    path = Path(__file__).resolve().parents[2] / "tools/loop_reference/uudd_qcd.json"
    return np.asarray(json.loads(path.read_text())["points"], dtype=str)


def _program(source, arithmetic="double", precision=None, prepared=None):
    from symbolica import E

    from pyamplicol.loops.physical_source import (
        source_bare_program,
        source_model_program,
    )

    factory = source_bare_program if prepared is None else source_model_program
    arguments = (source,) if prepared is None else (source, prepared)
    return factory(
        *arguments,
        ("e-", "e+"),
        ("m-", "m+"),
        selected_corrections=["QED"],
        state_kinds=("u", "v_bar", "u_bar", "v"),
        initial_average=E("4"),
        born_order_bounds={"QCD": 0, "QED": 2},
        loop_order_bounds={"QCD": 0, "QED": 4},
        parameters={
            "Gf": "0.0000116639",
            "aS": "0.118",
            "MU_R": "1000",
            "MT": "173.3",
            "MZ": "91.188",
            "MW": "80.419",
            "MH": "125",
            "ymt": "173.3",
            "WT": "0",
            "WZ": "0",
            "WW": "0",
            "WH": "0",
        },
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine="fast" if arithmetic == "double" else "generic",
        coefficient_engine="jit" if arithmetic == "double" else "interpreted",
    )


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance,bit",
    [
        ("double", None, "2e-12", 0),
        ("double-double", None, "5e-27", 80),
        ("arbitrary", 80, "2e-68", 180),
    ],
)
def test_full_gmu_bare_native_source_all_arithmetic(
    source, points, arithmetic, precision, tolerance, bit
):
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo

    original = source.native.to_json()
    program = _program(source, arithmetic, precision)
    bare = program.bare
    assert bare.model_evaluator is None
    assert bare.vertex_spinor_frame == "native-physical"
    assert all(r.vertex_spinor_frame == "native-physical" for r in bare.recurrences)
    assert len(bare.process.born) == 2 and len(bare.loop_graphs) == 98
    assert source.native.to_json() == original
    result = program.evaluate_decimal(points, ["1000000"] * 3)
    # Independently generated MadLoop full-Gmu N4-only component, real masses,
    # official zero restrictions and Cepsilon normalization. This reference is
    # binary64; the universal pole/rebinding below is the high-precision gate.
    expected = np.asarray(
        [
            [
                0.027479671300195206,
                0.001199173114959527,
                -0.00033794549692768073,
                -0.00013202400437376525,
            ],
            [
                0.016886440248949166,
                -0.00007649313933557234,
                -0.00006162255972291822,
                -0.00008112962622186653,
            ],
            [
                0.006066986982776142,
                -0.00012907198618978188,
                -0.000029798587460658466,
                -0.000029148380531899732,
            ],
        ]
    )
    actual = np.concatenate([result.born[:, None, :], result.virtual], axis=1)
    np.testing.assert_allclose(actual[:, :, 0].astype(float), expected, rtol=5e-11)
    assert "no-R2-no-UV" in result.convention
    assert np.max(np.abs(actual[:, :, 1].astype(float))) < 1e-16
    alpha = parse_exact_ufo(
        "2**(1/2)*0.0000116639*80.419**2*(1-80.419**2/91.188**2)/cmath.pi"
    )
    ratio = (
        Expression.evaluator_multiple(
            [-4 * alpha / (2 * E("pi"))], [], n_cores=1, jit_compile=False
        )
        .evaluate_complex_with_prec([], 100)[0]
        .to_decimal_tuple()[0]
    )
    with localcontext() as context:
        context.prec = 110
        for born, virtual in zip(result.born, result.virtual, strict=True):
            assert abs(Decimal(virtual[2, 0]) / Decimal(born[0]) - ratio) < Decimal(
                tolerance
            )
        # Rebinding is global and atomic; exact source helper functions stay
        # owned by the source program. Beyond-binary64 bits must affect outputs.
        factor = Decimal(2) if not bit else 1 + Decimal(2) ** -bit
        original_snapshot = program.state.snapshot()
        program.update_parameters({"Gf": str(Decimal("0.0000116639") * factor)})
        rebound = program.evaluate_decimal(points[:1], ["1000000"])
        assert program.state.revision == original_snapshot.revision + 1
        assert Decimal(rebound.born[0, 0]) != Decimal(result.born[0, 0])
        assert abs(
            Decimal(rebound.born[0, 0]) / Decimal(result.born[0, 0]) - factor**2
        ) < Decimal(tolerance)
        for index in range(3):
            assert (
                abs(
                    Decimal(rebound.virtual[0, index, 0])
                    / Decimal(result.virtual[0, index, 0])
                    - factor**3
                )
                < Decimal(tolerance) * 20
            )
        # A retained immutable snapshot remains valid after a global update.
        scale = bare.backend.NativeBatch.from_decimal(
            [("1000000", "0")], 1, 1, row_ids=[19], **bare.options
        )
        prior = bare.model_scale_view.evaluate_native(scale, snapshot=original_snapshot)
        assert prior.row_ids == [19]
        assert prior.to_decimal() == original_snapshot.values.to_decimal()
    empty = program.evaluate_decimal(points[:0], [])
    assert empty.born.shape == (0, 2) and empty.virtual.shape == (0, 3, 2)
    with pytest.raises(ValueError, match="named inputs"):
        bare.prepare_currents_native(points[:1], ["1000000"], model_values=[])
    with pytest.raises(ValueError, match="fixed during generation"):
        program.update_parameters({"Me": "1"})
    from pyamplicol.loops.physical_counterterms import NativeModelVirtualProgram

    with pytest.raises(NotImplementedError, match="matching certified vertex frame"):
        NativeModelVirtualProgram(bare, (), source_bindings=())
    if arithmetic == "double":
        _check_pointwise_source_bindings(program, points)


def _check_pointwise_source_bindings(program, points):
    """Exercise ordinary native binding with an explicitly scale-dependent output."""
    from pyamplicol.loops.physical_process import NativeBareProcessProgram
    from pyamplicol.loops.scalar_bindings import (
        NativeScalarBindings,
        NativeScaleBindings,
    )

    original = program.bare
    outputs = (*program.state.plan.outputs, ("MU_R", 0))
    state = NativeScalarBindings(
        program.structure.scalar_bindings(outputs),
        parameters=program.state.overrides,
        backend=original.backend,
    )
    indices = dict(
        zip(original.model_names, original.model_output_indices, strict=True)
    )
    # The extra output probes the same cold source→native input boundary without
    # modifying the physical source's Feynman rules or numerical couplings.
    indices["source_scale_probe::value"] = len(outputs) - 1
    bare = NativeBareProcessProgram(
        original.process,
        state_kinds=original.state_kinds,
        initial_average=original.initial_average,
        model_state=state,
        model_output_indices=indices,
        model_scale_view=NativeScaleBindings(state),
        vertex_spinor_frame=original.vertex_spinor_frame,
        reduction_engine="fast",
        coefficient_engine="jit",
        backend=original.backend,
    )
    currents = bare.prepare_currents_native(points[[0, 0]], ["1000000", "4000000"])
    position = bare.model_positions["source_scale_probe::value"]
    assert [
        Decimal(pair[0])
        for pair in currents.model.take_columns([position]).to_decimal()
    ] == [1000, 2000]
    bound = currents.batch.bound.take_columns(
        [bare.bound_positions["source_scale_probe::value"]]
    )
    assert [Decimal(pair[0]) for pair in bound.to_decimal()] == [1000] * 16 + [
        2000
    ] * 16
    assert currents.batch.model_snapshot is state.snapshot()


def test_source_dimensional_frame_is_not_inferred(source):
    from symbolica import E

    from pyamplicol.loops.physical_source import source_bare_program

    with pytest.raises(NotImplementedError, match="dimensional source"):
        source_bare_program(
            source,
            ("e-", "e+"),
            ("m-", "m+"),
            selected_corrections=["QED"],
            state_kinds=("u", "v_bar", "u_bar", "v"),
            initial_average=E("4"),
            dimensional_numerators=True,
        )


@pytest.fixture(scope="module")
def counterterms(source):
    from pyamplicol.loops.counterterms import prepare_counterterm_model
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.model_electroweak import (
        neutral_lepton_counterterm_validations,
    )

    ir = build_loop_model(source.raw, scalar_program=source.scalars)
    prepared = prepare_counterterm_model(
        ir,
        source.native,
        primitive_validations=neutral_lepton_counterterm_validations(ir),
        vertex_spinor_frame="ufo",
    )
    assert prepared.source_entries_complete and not prepared.unavailable_entries
    return prepared


def test_source_model_rejects_pruned_or_unavailable_counterterm_collections():
    from types import SimpleNamespace

    from pyamplicol.loops.physical_source import source_model_program

    for complete, unavailable in ((False, ()), (True, (("missing",),))):
        prepared = SimpleNamespace(
            source_entries_complete=complete, unavailable_entries=unavailable
        )
        with pytest.raises(ValueError, match="complete source CT collection"):
            source_model_program(None, prepared, (), ())


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance,bit",
    [
        ("double", None, "2e-11", 0),
        ("double-double", None, "2e-26", 80),
        ("arbitrary", 80, "2e-67", 180),
    ],
)
def test_full_source_model_components_scales_and_snapshot(
    source, counterterms, points, arithmetic, precision, tolerance, bit
):
    from dataclasses import replace

    from pyamplicol.loops.physical_counterterms import NativeModelVirtualProgram

    program = _program(source, arithmetic, precision, counterterms)
    model, bare = program.virtual, program.source_bare.bare
    assert len(program.matched.insertions) == 59
    assert not model.field_currents
    assert model.source_scale_view.global_state is program.source_bare.state
    assert all(current.coefficient_evaluator is None for current in model.currents)
    assert all(item.rule.vertex_spinor_frame == "ufo" for item in model.insertions)
    result = program.evaluate_decimal(points, ["1000000"] * 3)
    expected = {
        "uv": [
            [0.0006974628709385066, -0.0001501768968413483, 0],
            [0.0005026410170483123, -0.00008329131552308456, 0],
            [0.0006608485463514334, 0.000026484252229958202, 0],
        ],
        "r2": [
            [-0.00007942496245631837, 0, 0],
            [-0.000050256519953281034, 0, 0],
            [-0.000037496173851166774, 0, 0],
        ],
        "virtual": [
            [0.001817211023441716, -0.00048812239376902906, -0.00013202400437376525],
            [0.00037589135775946, -0.00014491387524600278, -0.00008112962622186653],
            [0.0004942803863104847, -0.000003314335230700264, -0.000029148380531899732],
        ],
        "bare": [
            [0.001199173114959527, -0.00033794549692768073, -0.00013202400437376525],
            [-0.00007649313933557234, -0.00006162255972291822, -0.00008112962622186653],
            [
                -0.00012907198618978188,
                -0.000029798587460658466,
                -0.000029148380531899732,
            ],
        ],
    }
    for name, values in expected.items():
        np.testing.assert_allclose(
            getattr(result, name)[:, :, 0].astype(float), values, rtol=5e-11, atol=1e-17
        )
    scales = ["250000", "2250000", "4000000"]
    changed = program.evaluate_decimal(points, scales)
    # Independent native MadLoop calls update MU_R and source EW COUP3 per event.
    expected_scales = {
        "uv": [
            [0.0009056522562001524, -0.0001501768968413483, 0],
            [0.00043509757254223266, -0.00008329131552308456, 0],
            [0.0006975635158762997, 0.000026484252229958202, 0],
        ],
        "bare": [
            [0.001540802390094379, -0.00015492136413186242, -0.00013202400437376525],
            [-0.00015314047315447595, -0.00012741302505656633, -0.00008112962622186653],
            [-0.0001983904545106025, -0.00007020682302780786, -0.000029148380531899732],
        ],
        "virtual": [
            [0.002367029683838213, -0.0003050982609732107, -0.00013202400437376525],
            [0.00023170057943447685, -0.0002107043405796509, -0.00008112962622186653],
            [0.0004616768875145301, -0.00004372257079784966, -0.000029148380531899732],
        ],
        "r2": expected["r2"],
    }
    for name, values in expected_scales.items():
        np.testing.assert_allclose(
            getattr(changed, name)[:, :, 0].astype(float),
            values,
            rtol=5e-11,
            atol=1e-17,
        )
    batch = bare.prepare_currents_native(points[:1], scales[:1]).batch
    before = [
        value.to_decimal()
        for value in model.evaluate_counterterm_currents_native(batch)
    ]
    with localcontext() as context:
        context.prec = 110
        factor = Decimal(2) if not bit else 1 + Decimal(2) ** -bit
        program.update_parameters({"Gf": str(Decimal("0.0000116639") * factor)})
        assert batch.model_snapshot is not program.source_bare.state.snapshot()
        after = [
            value.to_decimal()
            for value in model.evaluate_counterterm_currents_native(batch)
        ]
        assert before == after  # CTs retain exactly the bare call's model revision.
        rebound = program.evaluate_decimal(points[:1], scales[:1])
        for name in ("bare", "uv", "r2", "virtual"):
            target, actual = getattr(changed, name), getattr(rebound, name)
            assert Decimal(actual[0, 0, 0]) != Decimal(target[0, 0, 0])
            assert abs(
                Decimal(actual[0, 0, 0]) / Decimal(target[0, 0, 0]) - factor**3
            ) < Decimal(tolerance)
    empty = program.evaluate_decimal(points[:0], [])
    assert empty.virtual.shape == (0, 3, 2)
    first = model.insertions[0]
    wrong = replace(first, rule=replace(first.rule, vertex_spinor_frame=None))
    with pytest.raises(NotImplementedError, match="matching certified vertex frame"):
        NativeModelVirtualProgram(
            bare, (wrong, *model.insertions[1:]), source_bindings=()
        )
    with pytest.raises(NotImplementedError, match="anonymous scalar binding"):
        NativeModelVirtualProgram(
            bare, model.insertions, source_bindings=(), external_fields=(object(),)
        )
