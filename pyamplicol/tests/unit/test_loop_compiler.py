# SPDX-License-Identifier: 0BSD
"""Generic source preparation against independent complete component controls."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.loops.compiler import prepare_loop_request_model
from pyamplicol.models.loading import CompiledModel


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
def prepared():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("source generation requires licensed Symbolica")
    pytest.importorskip("_loop_reduction")
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not path:
        pytest.skip("set PYAMPLICOL_NLO_UFO_REFERENCE to loop_sm")
    from pyamplicol.models.loop_import import load_raw_loop_model

    raw = load_raw_loop_model(path)
    return prepare_loop_request_model(
        _compiled(raw),
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
    prepared, arithmetic, precision, tolerance
):
    reference = os.environ.get("PYAMPLICOL_UUDD_COMPONENT_REFERENCE")
    if not reference:
        pytest.skip("set PYAMPLICOL_UUDD_COMPONENT_REFERENCE to the MadLoop capture")
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


def test_same_compiler_handles_complete_neutral_source():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("source generation requires licensed Symbolica")
    pytest.importorskip("_loop_reduction")
    path = os.environ.get("PYAMPLICOL_EW_UFO_REFERENCE")
    if not path:
        pytest.skip("set PYAMPLICOL_EW_UFO_REFERENCE to the full Gmu UFO")
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
