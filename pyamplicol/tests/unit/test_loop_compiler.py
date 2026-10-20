# SPDX-License-Identifier: 0BSD
"""Generic source preparation against independent complete component controls."""

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.loops.compiler import prepare_loop_request_model
from pyamplicol.models.loading import CompiledModel

pytestmark = pytest.mark.loop_physics


def _compiled(raw):
    # Only the existing raw sidecar and declared compile options belong to the
    # loop ingress. Deliberately incompatible normalized defaults must not leak.
    return CompiledModel(
        source={"options": {"simplify": True}},
        producer={},
        model={},
        ir=None,
        parameter_defaults={"aS": (1000.0, 0.0)},
        capabilities={},
        issues=(),
        phase_timings={},
        conversion_seconds=0,
        raw_loop_model=raw,
    )


def test_compiler_requires_captured_source_not_normalized_tree_defaults():
    with pytest.raises(NotImplementedError, match="raw NLO sidecar"):
        prepare_loop_request_model(_compiled(None))
    with pytest.raises(TypeError, match="CompiledModel or RawLoopModel"):
        prepare_loop_request_model({"model": "arbitrary display name"})


@pytest.fixture(scope="module")
def prepared(loop_sm_ufo, loop_sm_compiled_model, loop_native):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("source generation requires licensed Symbolica")
    path = str(loop_sm_ufo)
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_import import load_raw_loop_model

    raw = load_raw_loop_model(path)
    # The actual compiled tree IR supplies the source particle contracts; its
    # normalized aS default is made deliberately incompatible and must not leak.
    compiled = load_compiled_model(loop_sm_compiled_model)
    compiled = replace(
        compiled,
        raw_loop_model=raw,
        parameter_defaults={**compiled.parameter_defaults, "aS": (1000.0, 0.0)},
        _serialized_path=None,
        _prepared_bundle=None,
    )
    return prepare_loop_request_model(
        compiled,
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
        fixed_parameters={"MB": "0"},
    )


def test_generic_scope_and_exact_tensor_capabilities(prepared):
    from pyamplicol.loops.model_catalogue import certified_source_counterterm_bases

    assert prepared.compiled_model.raw_loop_model is prepared.original
    assert prepared.model.scalar_program is prepared.structure.scalars
    assert prepared.structure.scalars.library.raw is prepared.declarations.derived
    with pytest.raises(ValueError, match="bounds differ"):
        prepared.compile(
            (2, -2),
            (1, -1),
            selected_corrections=["QCD"],
            loop_order_bounds={"QCD": 4, "QED": 2},
        )
    with pytest.raises(NotImplementedError, match="dimensional source"):
        prepared.compile(
            (2, -2), (1, -1), selected_corrections=["QCD"], dimensional_numerators=True
        )
    entry = next(e for e in prepared.model.kernels if e.vertex_name == "V_R2GUU")
    certified = {
        v.source_key for v in certified_source_counterterm_bases(prepared.model)
    }
    assert entry.key in certified
    # Keeping a known name/key never certifies a changed actual tensor.
    altered = replace(entry, lorentz="Gamma(3,2,1)+Identity(2,1)*P(3,1)")
    model = replace(
        prepared.model,
        kernels=tuple(
            altered if item.key == entry.key else item
            for item in prepared.model.kernels
        ),
    )
    assert entry.key not in {
        v.source_key for v in certified_source_counterterm_bases(model)
    }


def _exact_born_uv(point, scale, alpha_s, mass):
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo as parse

    p = [[parse(value) for value in leg] for leg in point]

    def square(v):
        return v[0] ** 2 - sum((x * x for x in v[1:]), E("0"))

    s, t, u = [
        square([a + sign * b for a, b in zip(p[0], p[leg], strict=True)])
        for leg, sign in ((1, 1), (2, -1), (3, -1))
    ]
    alpha = parse(alpha_s)
    born = E("4/9") * (4 * E("pi") * alpha) ** 2 * (t * t + u * u) / s**2
    finite = born * alpha / (3 * E("pi")) * (parse(scale) / parse(mass) ** 2).log()
    single = -7 * born * alpha / (2 * E("pi"))
    values = Expression.evaluator_multiple(
        [born, finite, single], [], n_cores=1, jit_compile=False
    ).evaluate_complex_with_prec([], 110)
    return [value.to_decimal_tuple()[0] for value in values]


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, "3e-11"),
        ("double-double", None, "3e-27"),
        ("arbitrary", 60, "3e-53"),
    ],
)
def test_complete_source_qcd_without_process_dispatch(
    loop_reference_path, prepared, arithmetic, precision, tolerance
):
    reference = str(loop_reference_path("uudd-component-control"))
    reference = json.loads(Path(reference).read_text())
    points = np.asarray(reference["fixture"]["points"], dtype=str)
    program = prepared.compile(
        (2, -2),
        (1, -1),
        selected_corrections=["QCD"],
        parameters={"aS": "0.118", "MT": "173", "MU_R": "1000"},
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine="fast" if arithmetic == "double" else "generic",
        coefficient_engine="jit" if arithmetic == "double" else "interpreted",
    )
    assert program.source_bare.external_request.initial_average == 36
    assert len(program.source_bare.bare.loop_graphs) == 14
    assert len(program.matched.insertions) == 23
    assert not program.virtual.external_fields
    result = program.evaluate_decimal(points, ["1000000"] * 3)
    for index, row in enumerate(reference["rows"]):
        for name, key in (
            ("bare", "bare"),
            ("uv", "uv"),
            ("r2", "r2"),
            ("virtual", "full"),
        ):
            actual = getattr(result, name)[index].astype(float)
            np.testing.assert_allclose(
                actual[:, 0],
                [row[key][p] for p in ("finite", "single_pole", "double_pole")],
                rtol=4e-11,
                atol=1e-14,
            )
            assert np.max(np.abs(actual[:, 1])) < 1e-14
    # Genuine carrier check: exact source parameter bits and per-point scale
    # bind through the same source owner; no binary64 reference is used here.
    alpha, mass, scale = "0.100000000000000000000000001", "190", "2100000"
    program.update_parameters({"aS": alpha, "MT": mass})
    rebound = program.evaluate_decimal(points[1:2], [scale])
    expected = _exact_born_uv(points[1], scale, alpha, mass)
    with localcontext() as context:
        context.prec = 120
        for actual, target in zip(
            [rebound.born[0, 0], rebound.uv[0, 0, 0], rebound.uv[0, 1, 0]],
            expected,
            strict=True,
        ):
            assert abs(Decimal(actual) - target) <= Decimal(tolerance) * abs(target)
    with pytest.raises(ValueError, match="fixed during generation"):
        program.update_parameters({"MB": "4.7"})


def test_same_compiler_handles_complete_neutral_source(loop_gmu_ufo, loop_native):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("source generation requires licensed Symbolica")
    path = str(loop_gmu_ufo)
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
    prepared = prepare_loop_request_model(
        load_raw_loop_model(path, restriction="full"),
        born_order_bounds={"QCD": 0, "QED": 2},
        loop_order_bounds={"QCD": 0, "QED": 4},
        compatibility="gmu_particle_anti",
        fixed_parameters={
            **dict.fromkeys(zeros, "0"),
            **dict.fromkeys(
                ("epsUV", "epsIR", "LogSwitch", "GfRedefinitionChoice"), "1"
            ),
        },
        covariant_multiplets={
            23: [23, 250, 9000002, -9000002],
            24: [24, 251, 9000003, -9000004],
            -24: [-24, -251, 9000004, -9000003],
        },
    )
    program = prepared.compile(
        (11, -11),
        (13, -13),
        selected_corrections=["QED"],
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
        reduction_engine="fast",
        coefficient_engine="jit",
    )
    assert program.source_bare.external_request.initial_average == 4
    assert len(program.source_bare.bare.loop_graphs) == 98
    assert len(program.matched.insertions) == 59
    result = program.evaluate_decimal(
        np.asarray(
            [
                [
                    [500, 0, 0, 500],
                    [500, 0, 0, -500],
                    [500, 300, 0, 400],
                    [500, -300, 0, -400],
                ]
            ],
            dtype=str,
        ),
        ["1000000"],
    )
    # Unmodified independent MadLoop source controls, binary64 reference only.
    for name, expected in {
        "bare": (
            0.001199173114959527,
            -0.00033794549692768073,
            -0.00013202400437376525,
        ),
        "uv": (0.0006974628709385066, -0.0001501768968413483, 0),
        "r2": (-0.00007942496245631837, 0, 0),
        "virtual": (
            0.001817211023441716,
            -0.00048812239376902906,
            -0.00013202400437376525,
        ),
    }.items():
        np.testing.assert_allclose(
            getattr(result, name)[0, :, 0].astype(float),
            expected,
            rtol=5e-11,
            atol=1e-17,
        )


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, "3e-11"),
        ("double-double", None, "3e-25"),
        ("arbitrary", 60, "3e-50"),
    ],
)
def test_source_particle_fields_complete_all_arithmetic(
    loop_reference_path, prepared, arithmetic, precision, tolerance
):
    reference_path = str(loop_reference_path("gluon-scattering-component-control"))
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo as parse

    reference = json.loads(Path(reference_path).read_text())
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/gluon_scattering_qcd.json"
    )
    points = np.asarray(json.loads(fixture.read_text())["points"], dtype=str)
    program = prepared.compile(
        (21, 21),
        (21, 21),
        selected_corrections=["QCD"],
        parameters={"aS": "0.118", "MT": "173", "MU_R": "1000"},
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine="fast" if arithmetic == "double" else "generic",
        coefficient_engine="jit" if arithmetic == "double" else "interpreted",
    )
    model, bare = program.virtual, program.source_bare.bare
    assert program.source_bare.external_request.initial_average == 512
    assert len(bare.loop_graphs) == 186 and len(model.external_fields) == 48
    assert len(model.insertions) == 119
    assert len(model.source_scale_view.plan.anonymous_definitions) == 3
    assert model.source_scale_view.global_state is program.source_bare.state
    assert all(
        c.coefficient_evaluator is None and c.source_scalar_column is not None
        for c in model.field_currents
    )
    result = program.evaluate_decimal(points, ["1000000"] * 3)
    for index, row in enumerate(reference["rows"]):
        for name, key in (
            ("bare", "bare"),
            ("uv", "uv"),
            ("r2", "r2"),
            ("virtual", "full"),
            ("uv_local", "uv_vertex"),
            ("uv_external_fields", "wf"),
        ):
            np.testing.assert_allclose(
                getattr(result, name)[index, :, 0].astype(float),
                [row[key][p] for p in ("finite", "single_pole", "double_pole")],
                rtol=5e-11,
                atol=1e-10,
            )
    # Field factors must use the same captured owner revision as the Born ket.
    batch = bare.prepare_currents_native(points[:1], ["900000"]).batch
    old = [
        value.to_decimal()
        for value in model.evaluate_counterterm_currents_native(batch)
    ]
    alpha, mass, scale = (
        "0.100000000000000000000000001",
        "190.000000000000000000000001",
        "2100000",
    )
    program.update_parameters({"aS": alpha, "MT": mass})
    assert old == [
        value.to_decimal()
        for value in model.evaluate_counterterm_currents_native(batch)
    ]
    rebound = program.evaluate_decimal(points[:1], [scale])
    # Independent heavy-quark field/decoupling ratios in the same declared
    # scheme. Native dependency algebra supplies the high-precision constants.
    p = [[parse(value) for value in leg] for leg in points[0]]

    def square(v):
        return v[0] ** 2 - sum((x * x for x in v[1:]), E("0"))

    s, t, u = [
        square([a + sign * b for a, b in zip(p[0], p[leg], strict=True)])
        for leg, sign in ((1, 1), (2, -1), (3, -1))
    ]
    coupling = parse(alpha)
    born = (
        E("9/4")
        * (4 * E("pi") * coupling) ** 2
        * (3 - t * u / s**2 - u * s / t**2 - s * t / u**2)
    )
    finite = coupling / (3 * E("pi")) * (parse(scale) / parse(mass) ** 2).log()
    targets = Expression.evaluator_multiple(
        [
            born,
            born * finite,
            -7 * born * coupling / (2 * E("pi")),
            -2 * born * finite,
            -2 * born * coupling / (3 * E("pi")),
        ],
        [],
        n_cores=1,
        jit_compile=False,
    ).evaluate_complex_with_prec([], 110)
    actual = [
        rebound.born[0],
        rebound.uv_local[0, 0],
        rebound.uv_local[0, 1],
        rebound.uv_external_fields[0, 0],
        rebound.uv_external_fields[0, 1],
    ]
    with localcontext() as context:
        context.prec = 120
        for value, target in zip(actual, targets, strict=True):
            expected = target.to_decimal_tuple()[0]
            assert abs(Decimal(value[0]) - expected) < abs(expected) * Decimal(
                tolerance
            )
            assert abs(Decimal(value[1])) < abs(expected) * Decimal(tolerance)
    empty = program.evaluate_decimal(points[:0], [])
    assert empty.virtual.shape == (0, 3, 2)
    # Real-mass source programs in the native physical frame admit partial
    # evaluation (e0988180); one good point keeps its opaque ID and values.
    partial = model.evaluate_ordered_partial_native(
        points[:1], ["1000000"], point_ids=[71]
    )
    ordinary = model.evaluate_ordered_native(points[:1], ["1000000"])
    assert partial.failures == () and partial.batch.point_ids == [71]
    assert partial.totals.agreement_rows(
        ordinary.totals.take_rows([0], row_ids=[71]), "1e-13", "1e-100"
    ) == [(71, True)]


@pytest.mark.parametrize("gluons", [False, True], ids=["uudd", "gg-fields"])
def test_complete_source_retained_views_without_producer_work(
    prepared, monkeypatch, gluons, loop_reference
):
    from pyamplicol.loops import (
        physical_counterterms,
        physical_process,
        physical_source,
    )
    from pyamplicol.loops.color_bilinears import PhysicalColorSpace
    from pyamplicol.loops.kernels import CoefficientKernel, PreparedCoefficientDAG
    from pyamplicol.loops.kinematics import LoopKinematicPlan
    from pyamplicol.loops.scalar_bindings import ScalarBindingPlan
    from pyamplicol.loops.scalar_frontend import ScalarFunctionLibrary

    reference = loop_reference(
        "gluon-scattering-component-control" if gluons else "uudd-component-control"
    )
    fixture = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference"
        / ("gluon_scattering_qcd.json" if gluons else "uudd_qcd.json")
    )
    points = np.asarray(json.loads(fixture.read_text())["points"], dtype=str)
    program = prepared.compile(
        (21, 21) if gluons else (2, -2),
        (21, 21) if gluons else (1, -1),
        selected_corrections=["QCD"],
        parameters={"aS": "0.118", "MT": "173", "MU_R": "1000"},
        reduction_engine="fast",
        coefficient_engine="jit",
    )
    record = program.prepared
    assert record.source_scale.owner is record.bare.source_scalars
    assert record.bare.source_scale.owner is record.bare.source_scalars
    assert all(
        current.coefficient_state is None
        for current in (*record.currents, *record.fields)
    )
    assert bool(record.fields) == gluons
    from types import SimpleNamespace

    from pyamplicol.loops.prepared_process import (
        PreparedBareProcess,
        PreparedModelProcess,
    )

    # Since d53d871c retained views keep exact colour projections; source
    # bindings with dimensional numerators remain unsupported, projected or not.
    for override in (
        {"dimensional_numerators": True},
        {"dimensional_numerators": True, "_formal_color_projected": True},
    ):
        altered = SimpleNamespace(**(vars(program.virtual.bare) | override))
        with pytest.raises(NotImplementedError, match="dimensional numerators"):
            PreparedBareProcess.from_program(altered)
    other_owner = record.bare.source_scalars.bind()
    other_scale = record.source_scale.bind(other_owner)
    altered = SimpleNamespace(
        **(vars(program.virtual) | {"source_scale_view": other_scale})
    )
    with pytest.raises(ValueError, match="different scalar owner"):
        PreparedModelProcess.from_program(altered)
    baseline = program.evaluate_decimal(points, ["1000000"] * 3)
    saved = baseline.virtual.copy()

    def forbidden(*args, **kwargs):
        raise AssertionError("source arithmetic rebind repeated producer work")

    for owner, name in (
        (physical_source, "source_bare_program"),
        (physical_source, "source_model_program"),
        (physical_process, "build_color_loop_recurrence"),
        (physical_process, "build_color_tree_recurrence"),
        (physical_process, "build_loop_kinematics"),
        (physical_counterterms, "build_color_tree_recurrence"),
        (physical_counterterms, "_saved"),
        (physical_counterterms, "_saved_scalar_pairs"),
        (physical_process, "_saved"),
        (PhysicalColorSpace, "overlap"),
        (PreparedCoefficientDAG, "from_dag"),
        (CoefficientKernel, "evaluator_bytes"),
        (LoopKinematicPlan, "evaluator_bytes"),
        (ScalarBindingPlan, "evaluator_bytes"),
        (ScalarFunctionLibrary, "lower"),
        (ScalarFunctionLibrary, "lower_value"),
        (ScalarFunctionLibrary, "evaluator"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    for arithmetic, precision, tolerance in (
        ("double", None, "3e-11"),
        ("double-double", None, "3e-25"),
        ("arbitrary", 60, "3e-50"),
    ):
        view = program.rebind(
            arithmetic=arithmetic,
            precision=precision,
            reduction_engine="fast" if arithmetic == "double" else "generic",
            coefficient_engine="jit" if arithmetic == "double" else "interpreted",
        )
        state, bare, model = view.source_bare.state, view.source_bare.bare, view.virtual
        assert view.prepared is record and bare.prepared is record.bare
        assert state is bare.model_state is model.source_scale_view.global_state
        assert bare.model_scale_view.global_state is state
        assert state is not program.source_bare.state
        assert bare.families[0].reducer is not program.virtual.bare.families[0].reducer
        assert state.overrides == program.source_bare.state.overrides
        # Ordinary execution does not require optional producer inspection refs.
        state.plan = bare.model_scale_view.plan = model.source_scale_view.plan = None
        actual = view.evaluate_decimal(points, ["1000000"] * 3)
        pairs = [("bare", "bare"), ("uv", "uv"), ("r2", "r2"), ("virtual", "full")]
        if gluons:
            pairs += [("uv_local", "uv_vertex"), ("uv_external_fields", "wf")]
        for index, row in enumerate(reference["rows"]):
            for name, key in pairs:
                np.testing.assert_allclose(
                    getattr(actual, name)[index, :, 0].astype(float),
                    [row[key][p] for p in ("finite", "single_pole", "double_pole")],
                    rtol=5e-11,
                    atol=1e-10 if gluons else 1e-14,
                )
        batch = bare.prepare_currents_native(points[:1], ["900000"]).batch
        retained = tuple(
            value.to_decimal()
            for value in model.evaluate_counterterm_currents_native(batch)
        )
        alpha, mass, scale = "0.100000000000000000000000001", "190", "2100000"
        view.update_parameters({"aS": alpha, "MT": mass})
        assert (
            tuple(
                value.to_decimal()
                for value in model.evaluate_counterterm_currents_native(batch)
            )
            == retained
        )
        changed = view.evaluate_decimal(points[:1], [scale])
        if not gluons:
            targets = _exact_born_uv(points[0], scale, alpha, mass)
            observed = [changed.born[0, 0], changed.uv[0, 0, 0], changed.uv[0, 1, 0]]
        else:
            from symbolica import E, Expression

            from pyamplicol.loops.model_expressions import parse_exact_ufo as parse

            # Independent field half-Z relation; exact source input strings enter
            # both the analytic control and saved native scalar view.
            ratio = parse(alpha) / (3 * E("pi"))
            expected = Expression.evaluator_multiple(
                [-2 * ratio * (parse(scale) / parse(mass) ** 2).log(), -2 * ratio],
                [],
                n_cores=1,
                jit_compile=False,
            ).evaluate_complex_with_prec([], 100)
            with localcontext() as context:
                context.prec = 110
                targets = [
                    Decimal(changed.born[0, 0]) * x.to_decimal_tuple()[0]
                    for x in expected
                ]
            observed = [
                changed.uv_external_fields[0, 0, 0],
                changed.uv_external_fields[0, 1, 0],
            ]
        with localcontext() as context:
            context.prec = 110
            for value, target in zip(observed, targets, strict=True):
                assert abs(Decimal(value) - target) < abs(target) * Decimal(tolerance)
        assert view.evaluate_decimal(points[:0], []).virtual.shape == (0, 3, 2)
        assert program.source_bare.state.overrides["aS"] == ("0.118", "0")
    np.testing.assert_array_equal(baseline.virtual, saved)
    # Facade rebind captures exact overrides once, including bits absent from its
    # original double scalar buffers. The bare facade follows the same contract.
    program.update_parameters({"aS": alpha, "MT": mass})
    bare_view = program.source_bare.rebind(arithmetic="arbitrary", precision=60)
    assert bare_view.state.overrides["aS"] == (alpha, "0")
    assert bare_view.state.prepared is record.bare.source_scalars
    if not gluons:
        rebound = bare_view.evaluate_decimal(points[:1], [scale])
        with localcontext() as context:
            context.prec = 110
            expected = _exact_born_uv(points[0], scale, alpha, mass)[0]
            assert abs(Decimal(rebound.born[0, 0]) - expected) < abs(
                expected
            ) * Decimal("3e-53")
