# SPDX-License-Identifier: 0BSD
"""Physical fixed-Nf projections include source scalar and tensor color factors."""

from __future__ import annotations

import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.loops.physical_counterterms import (
    NativeModelVirtualProgram,
    distinct_flavour_model_virtual_program,
    drell_yan_model_virtual_program,
    top_pair_model_virtual_program,
)

pytestmark = pytest.mark.loop_nightly


@pytest.fixture(scope="module")
def source(loop_sm_ufo, loop_native):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native color controls require licensed Symbolica")
    path = str(loop_sm_ufo)
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.models.loop_import import load_raw_loop_model

    return build_loop_model(load_raw_loop_model(path))


def _point(name):
    path = Path(__file__).resolve().parents[2] / f"tools/loop_reference/{name}.json"
    return np.asarray(
        json.loads(path.read_text(), parse_int=str, parse_float=str)["points"],
        dtype=str,
    )[:1]


def _close(left, right, tolerance, ratio=Decimal(1)):
    with localcontext() as context:
        context.prec = 120
        if isinstance(ratio, tuple):
            ratio = Decimal(ratio[0]) / Decimal(ratio[1])
        for a, b in zip(
            np.asarray(left).reshape(-1, 2),
            np.asarray(right).reshape(-1, 2),
            strict=True,
        ):
            actual = tuple(map(Decimal, a))
            expected = tuple(Decimal(value) * ratio for value in b)
            # Roundoff in the imaginary part scales with its complex value,
            # including when the exact physical result is real.
            scale = max(*map(abs, expected), Decimal("1e-10"))
            assert all(
                abs(x - y) < scale * tolerance
                for x, y in zip(actual, expected, strict=True)
            )


@pytest.mark.parametrize("first_accuracy", ["dense", "lc"])
def test_model_color_projection_cannot_mutate_shared_bare(source, first_accuracy):
    from symbolica import S

    first = drell_yan_model_virtual_program(
        source,
        formal_color=first_accuracy != "dense",
        color_accuracy="full" if first_accuracy == "dense" else first_accuracy,
    )
    original_metric = first.bare.loop_metric
    with pytest.raises(ValueError, match="construct a fresh bare program"):
        NativeModelVirtualProgram(
            first.bare,
            first.insertions,
            source_bindings=first.source_bindings,
            formal_nc=None if first_accuracy == "lc" else S("color_reuse::Nc"),
            color_accuracy="full" if first_accuracy == "lc" else "lc",
        )
    with pytest.raises(ValueError, match="already bound"):
        first.bare.set_color_metrics(first.bare.born_metric, original_metric)
    assert first.bare.loop_metric == original_metric


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, Decimal("3e-14")),
        ("double-double", None, Decimal("3e-29")),
        ("arbitrary", 60, Decimal("3e-57")),
    ],
)
def test_conditional_laurent_collection_keeps_runtime_branches(
    loop_native, arithmetic, precision, tolerance
):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native conditional evaluator requires licensed Symbolica")
    from symbolica import E, S

    from pyamplicol.loops.physical import _saved
    from pyamplicol.loops.physical_color import expand_color_conditionals

    native = loop_native
    nc, mass, scale, other = S(
        "if_color_test::Nc",
        "if_color_test::mass",
        "if_color_test::scale",
        "if_color_test::other",
    )
    conditional = S("symbolica::if")
    expression = conditional(
        mass,
        (nc**2 - 1)
        / (2 * nc)
        * ((mass**2 / scale).log() + conditional(other, nc**2, 1 / nc)),
        1 / nc + nc**2,
    )
    rewritten = expand_color_conditionals(expression, nc)
    assert all(
        nc not in coefficient.get_all_symbols()
        for _, coefficient in rewritten.expand().coefficient_list(nc)
    )
    evaluator = native.CoefficientEvaluator(
        _saved([expression, rewritten], [nc, mass, scale, other]),
        arithmetic=arithmetic,
        precision=precision,
    )
    inputs = native.NativeBatch.from_decimal(
        [
            [value, "0"]
            for row in [
                ("3", "0", "4", "0"),
                ("3", "2", "16", "0"),
                ("4", "3", "5", "2"),
            ]
            for value in row
        ],
        3,
        4,
        arithmetic=arithmetic,
        precision=precision,
    )
    results = np.asarray(
        evaluator.evaluate_batch(inputs).to_decimal(), dtype=object
    ).reshape(3, 2, 2)
    _close(results[:, 0], results[:, 1], tolerance)
    with pytest.raises(NotImplementedError, match="Nc-dependent conditions"):
        expand_color_conditionals(conditional(nc, mass, E("0")), nc)
    with pytest.raises(NotImplementedError, match="unsupported functional"):
        expand_color_conditionals(conditional(mass, nc.log(), E("0")), nc)


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, Decimal("3e-11")),
        ("double-double", None, Decimal("3e-26")),
        ("arbitrary", 50, Decimal("3e-43")),
    ],
)
def test_drell_yan_projection_includes_model_r2(
    source, arithmetic, precision, tolerance
):
    results = {}
    for accuracy in ("dense", "full", "lc", "nlc"):
        program = drell_yan_model_virtual_program(
            source,
            formal_color=accuracy != "dense",
            color_accuracy="full" if accuracy == "dense" else accuracy,
            arithmetic=arithmetic,
            precision=precision,
        )
        results[accuracy] = program.evaluate_decimal(
            _point("drell_yan_qcd"), ["1000000"], model_values=["0.118", "132.507"]
        )
    full = results["full"]
    assert (full.born_color_degree, full.virtual_color_degree) == (1, 2)
    for name in ("born", "bare", "r2", "virtual"):
        _close(getattr(full, name), getattr(results["dense"], name), tolerance)
        _close(getattr(results["nlc"], name), getattr(full, name), tolerance)
        _close(
            getattr(results["lc"], name),
            getattr(full, name),
            tolerance,
            Decimal(1) if name == "born" else Decimal(9) / 8,
        )


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, Decimal("3e-11")),
        ("double-double", None, Decimal("3e-26")),
        ("arbitrary", 50, Decimal("3e-43")),
    ],
)
def test_fixed_nf_odd_suppression_and_shared_virtual_degree(
    source, arithmetic, precision, tolerance
):
    results = {}
    for accuracy in ("dense", "full", "lc", "nlc"):
        program = distinct_flavour_model_virtual_program(
            source,
            formal_color=accuracy != "dense",
            color_accuracy="full" if accuracy == "dense" else accuracy,
            arithmetic=arithmetic,
            precision=precision,
        )
        results[accuracy] = program.evaluate_decimal(
            _point("uudd_qcd"), ["1000000"], model_values=["0.118", "173"]
        )
    full, lc, nlc = results["full"], results["lc"], results["nlc"]
    assert (full.born_color_degree, full.virtual_color_degree) == (2, 3)
    for name in ("born", "bare", "uv", "r2", "virtual"):
        _close(getattr(full, name), getattr(results["dense"], name), tolerance)
    _close(lc.born, full.born, tolerance, Decimal(9) / 8)
    _close(nlc.born, full.born, tolerance)
    # CF*(Nc^2-1) gives Nc^3 - 2*Nc + 1/Nc, so a single common P=3
    # determines the universal double pole independently of its numerical value.
    _close(lc.virtual[:, 2], full.virtual[:, 2], tolerance, Decimal(81) / 64)
    _close(nlc.virtual[:, 2], full.virtual[:, 2], tolerance, Decimal(63) / 64)
    # Top decoupling is degree2 at fixed Nf: absent at LC, present at NLC.
    assert all(Decimal(value) == 0 for value in lc.uv[:, 0].flat)
    _close(nlc.uv[:, 0], full.uv[:, 0], tolerance, Decimal(9) / 8)
    # With six explicit species beta0=11*Nc/3-4; its degree2 quark term
    # is an odd suppression and must be retained by the NLC rule P-2.
    _close(lc.uv[:, 1], full.uv[:, 1], tolerance, (99, 56))
    _close(nlc.uv[:, 1], full.uv[:, 1], tolerance, (13, 14))


@pytest.mark.parametrize(
    "arithmetic,precision,tolerance",
    [
        ("double", None, Decimal("3e-11")),
        ("double-double", None, Decimal("3e-26")),
        ("arbitrary", 50, Decimal("3e-43")),
    ],
)
def test_top_field_color_is_projected_with_all_virtual_sectors(
    source, arithmetic, precision, tolerance
):
    results = {}
    for accuracy in ("dense", "full", "lc", "nlc"):
        program = top_pair_model_virtual_program(
            source,
            formal_color=accuracy != "dense",
            color_accuracy="full" if accuracy == "dense" else accuracy,
            arithmetic=arithmetic,
            precision=precision,
        )
        results[accuracy] = program.evaluate_decimal(
            _point("top_pair_qcd"), ["1000000"], model_values=["0.118", "173"]
        )
    full, lc, nlc = results["full"], results["lc"], results["nlc"]
    assert (full.born_color_degree, full.virtual_color_degree) == (2, 3)
    for name in ("born", "bare", "uv", "r2", "virtual", "uv_external_fields"):
        _close(getattr(full, name), getattr(results["dense"], name), tolerance)
    _close(lc.uv_external_fields, full.uv_external_fields, tolerance, Decimal(81) / 64)
    _close(nlc.uv_external_fields, full.uv_external_fields, tolerance, Decimal(63) / 64)
    assert all(Decimal(value) == 0 for value in lc.uv_local[:, 0].flat)
    _close(nlc.uv_local[:, 0], full.uv_local[:, 0], tolerance, Decimal(9) / 8)


@pytest.mark.parametrize(
    "arithmetic,precision,reducer,coefficients,tolerance",
    [
        ("double", None, "fast", "jit", Decimal("3e-10")),
        ("double-double", None, "generic", "interpreted", Decimal("3e-24")),
        ("arbitrary", 50, "generic", "interpreted", Decimal("3e-40")),
    ],
)
@pytest.mark.deferred_direct_r2
def test_gluon_direct_rectangular_projection_and_fixed_nf(
    loop_reference_path, source, arithmetic, precision, reducer, coefficients, tolerance
):
    from symbolica import E, Expression

    from pyamplicol.loops.physical_counterterms import (
        gluon_scattering_model_virtual_program,
    )

    path = (
        Path(__file__).resolve().parents[2]
        / "tools/loop_reference/gluon_scattering_qcd.json"
    )
    points = np.asarray(
        json.loads(path.read_text(), parse_int=str, parse_float=str)["points"],
        dtype=str,
    )
    results = {}
    for accuracy in ("full", "lc", "nlc"):
        program = gluon_scattering_model_virtual_program(
            source,
            r2_strategy="direct",
            formal_color=True,
            color_accuracy=accuracy,
            arithmetic=arithmetic,
            precision=precision,
            reduction_engine=reducer,
            coefficient_engine=coefficients,
        )
        assert len(program.bare.born_colors) == 6
        assert any(current.term_count == 3 for current in program.currents)
        assert {current.color_grade for current in program.currents} == {0, 1}
        assert {current.color_grade for current in program.field_currents} == {0}
        results[accuracy] = program.evaluate_decimal(
            points, ["1000000"] * len(points), model_values=["0.118", "173"]
        )
    full, lc, nlc = (results[key] for key in ("full", "lc", "nlc"))
    assert all(
        (result.born_color_degree, result.virtual_color_degree) == (4, 5)
        for result in results.values()
    )
    _close(lc.born, full.born, tolerance, (9, 8))
    _close(nlc.born, full.born, tolerance)
    # The four external adjoints give -4*CA*alpha_s/(2*pi). Born has
    # Nc^4-Nc^2; its virtual double pole has Nc^5-Nc^3, with one common P=5.
    ratio = (
        Expression.evaluator_multiple(
            [-6 * E("59/500") / E("pi")], [], n_cores=1, jit_compile=False
        )
        .evaluate_complex_with_prec([], 110)[0]
        .to_decimal_tuple()[0]
    )
    for result in results.values():
        _close(result.virtual[:, 2], result.born, tolerance, ratio)
    _close(lc.virtual[:, 2], full.virtual[:, 2], tolerance, (9, 8))
    _close(nlc.virtual[:, 2], full.virtual[:, 2], tolerance)
    # Fixed-Nf quark UV/fields carry Nc^4 and Nc^2. The odd suppression Nc^4
    # survives NLC(P-2=3), while LC drops it; source TF is never scaled as Nc.
    assert all(Decimal(value) == 0 for value in lc.uv_local[:, 0].flat)
    assert all(Decimal(value) == 0 for value in lc.uv_external_fields.flat)
    _close(nlc.uv_local[:, 0], full.uv_local[:, 0], tolerance, (9, 8))
    _close(nlc.uv_external_fields, full.uv_external_fields, tolerance, (9, 8))
    reference_path = str(loop_reference_path("gluon-scattering-component-control"))
    reference = json.loads(Path(reference_path).read_text())
    for name, key in (
        ("bare", "bare"),
        ("r2", "r2"),
        ("uv", "uv"),
        ("uv_local", "uv_vertex"),
        ("uv_external_fields", "wf"),
        ("virtual", "full"),
    ):
        np.testing.assert_allclose(
            getattr(full, name)[:, :, 0].astype(float),
            [
                [row[key][p] for p in ("finite", "single_pole", "double_pole")]
                for row in reference["rows"]
            ],
            rtol=5e-11,
            atol=1e-10,
        )
