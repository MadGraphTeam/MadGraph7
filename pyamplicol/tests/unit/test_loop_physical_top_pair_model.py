# SPDX-License-Identifier: 0BSD
"""Model-rule top pairs with internal massive tops against independent MadLoop.

``g g > t t~`` exercises massive internal top propagators, the model's top
mass counterterm and the R2 slash-plus-mass two-point. ``u u~ > t t~`` adds the
top-loop gluon self-energy. References are binary64 MadLoop values on exactly
on-shell points; higher-precision carriers additionally check exact identities.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

_POLES = ("finite", "single_pole", "double_pole")
_COMPONENTS = (
    ("bare", "bare"),
    ("uv", "uv"),
    ("r2", "r2"),
    ("virtual", "full"),
    ("uv_local", "uv_vertex"),
    ("uv_external_fields", "wf"),
)
# Exact infrared double-pole coefficients in units of alpha_s/(2 pi) * Born:
# -2 C_A for two external gluons, -2 C_F for two massless quarks.
_DOUBLE_POLE = {"gg": "-6", "uu": "-8/3"}
_INCOMING = {"gg": (21, 21), "uu": (2, -2)}
_CARRIERS = [
    ("double", None, "2e-12"),
    ("double-double", None, "3e-25"),
    ("arbitrary", 80, "1e-60"),
]


@pytest.fixture(scope="module")
def reference():
    data = json.loads(
        (
            Path(__file__).resolve().parents[1] / "fixtures/loop/top-pair-model.json"
        ).read_text()
    )
    with localcontext() as context:
        context.prec = 60
        for point in data["points"]:
            legs = [[Decimal(x) for x in leg] for leg in point]
            for index, leg in enumerate(legs):
                assert leg[0] ** 2 - sum(x * x for x in leg[1:]) == (
                    0 if index < 2 else Decimal(173) ** 2
                )
            for mu in range(4):
                assert legs[0][mu] + legs[1][mu] == legs[2][mu] + legs[3][mu]
    return data


@pytest.fixture(scope="module")
def prepared(reference):
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("licensed native generation required")
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not path:
        pytest.skip("set PYAMPLICOL_NLO_UFO_REFERENCE to loop_sm")
    from pyamplicol.loops.compiler import prepare_loop_request_model
    from pyamplicol.loops.model_compatibility import prepare_loop_sm_scalar_declarations
    from pyamplicol.models.loop_import import load_raw_loop_model

    view = prepare_loop_sm_scalar_declarations(load_raw_loop_model(path))
    return prepare_loop_request_model(
        view.derived,
        born_order_bounds={"QCD": 2, "QED": 0},
        loop_order_bounds={"QCD": 4, "QED": 0},
        fixed_parameters=reference["fixed_parameters"],
    )


@pytest.fixture(scope="module")
def programs(prepared, reference):
    cache = {}

    def program(key):
        if key not in cache:
            cache[key] = prepared.compile(
                _INCOMING[key],
                (6, -6),
                selected_corrections=["QCD"],
                parameters=reference["parameters"],
                arithmetic="double",
                reduction_engine="fast",
                coefficient_engine="jit",
            )
        return cache[key]

    return program


def _carrier(program, arithmetic, precision):
    if arithmetic == "double":
        return program
    return program.rebind(arithmetic=arithmetic, precision=precision)


def _inputs(reference):
    points = np.asarray(reference["points"], dtype=str)
    return points, [str(int(mu) ** 2) for mu in reference["mu_r"]]


def _compare(result, rows, components):
    for index, row in enumerate(rows):
        born = row["born"]
        np.testing.assert_allclose(float(result.born[index, 0]), born, rtol=2e-12)
        for name, key in components:
            np.testing.assert_allclose(
                getattr(result, name)[index, :, 0].astype(float),
                [row["components"][key][pole] for pole in _POLES],
                rtol=1e-10,
                atol=abs(born) * 1e-12,
                err_msg=f"{key} point {index}",
            )


def _check_exact_identities(result, key, tolerance):
    from symbolica import E, Expression

    factor = (
        Expression.evaluator_multiple(
            [E(_DOUBLE_POLE[key]) * E("118/1000") / (2 * E("pi"))],
            [],
            n_cores=1,
            jit_compile=False,
        )
        .evaluate_complex_with_prec([], 110)[0]
        .to_decimal_tuple()[0]
    )
    with localcontext() as context:
        context.prec = 110
        threshold = Decimal(tolerance)
        for index in range(result.born.shape[0]):
            born = Decimal(result.born[index, 0])
            expected = born * factor
            assert (
                abs(Decimal(result.virtual[index, 2, 0]) - expected)
                < abs(expected) * threshold
            )
            for pole in range(3):
                total = sum(
                    Decimal(getattr(result, name)[index, pole, 0])
                    for name in ("bare", "uv", "r2")
                )
                assert (
                    abs(Decimal(result.virtual[index, pole, 0]) - total)
                    < abs(born) * threshold
                )


@pytest.mark.parametrize("arithmetic,precision,tolerance", _CARRIERS)
def test_quark_top_pair_with_top_loop_self_energy(
    programs, reference, arithmetic, precision, tolerance
):
    program = _carrier(programs("uu"), arithmetic, precision)
    bare = program.source_bare.bare
    assert len(bare.process.born) == 1 and len(bare.loop_graphs) == 14
    points, scales = _inputs(reference)
    result = program.evaluate_decimal(points, scales)
    assert all(status == "success" for row in result.statuses for _, status in row)
    _compare(result, reference["rows"]["uu"], _COMPONENTS)
    _check_exact_identities(result, "uu", tolerance)


def _witness_virtual(program, prepared, select=None):
    """Counterterm assembly over its own loop witnesses only (diagnostic).

    The actual matched source insertions keep their admitted witness loops; all
    other loop graphs are omitted, so the bare component is incomplete and is
    not compared. ``select`` restricts the insertion set without changing any.
    """
    from pyamplicol.loops.physical_counterterms import NativeModelVirtualProgram
    from pyamplicol.loops.physical_process import NativeBareProcessProgram
    from pyamplicol.loops.wavefunction import match_external_field_counterterms

    bare = program.source_bare.bare
    insertions = tuple(program.matched.insertions)
    witnesses = set().union(*(item.witness_loop_ids for item in insertions))
    loops = tuple(graph for graph in bare.process.loops if graph.id in witnesses)
    assert 0 < len(loops) < len(bare.process.loops)
    filtered = NativeBareProcessProgram(
        replace(bare.process, loops=loops),
        state_kinds=bare.state_kinds,
        initial_average=bare.initial_average,
        model_state=bare.model_state,
        model_output_indices=dict(
            zip(bare.model_names, bare.model_output_indices, strict=True)
        ),
        model_scale_view=bare.model_scale_view,
        vertex_spinor_frame=bare.vertex_spinor_frame,
        backend=bare.backend,
        **bare.options,
        reduction_engine=bare.reduction_engine,
        coefficient_engine=bare.coefficient_engine,
    )
    process = filtered.process
    if select is None:
        fields = match_external_field_counterterms(
            prepared.counterterms.source,
            process,
            selected_corrections=process.orders.correction_names,
            allowed_amplitude_orders={
                pair.loop for sector in process.orders.sectors for pair in sector.pairs
            },
        ).insertions
    else:
        insertions, fields = tuple(filter(select, insertions)), ()
    return NativeModelVirtualProgram(
        filtered, insertions, source_bindings=(), external_fields=fields
    )


def _is_mass_counterterm(item):
    return item.rule.kind == "UVmass"


def _is_r2_fermion_two_point(item):
    pdgs = item.rule.particle_pdgs
    return item.rule.kind == "R2" and len(pdgs) == 2 and pdgs[0] == -pdgs[1]


@pytest.mark.parametrize("arithmetic,precision,tolerance", _CARRIERS)
def test_gluon_top_pair_counterterms_with_internal_massive_top(
    programs, prepared, reference, arithmetic, precision, tolerance
):
    program = programs("gg")
    bare = program.source_bare.bare
    assert len(bare.process.born) == 3 and len(bare.loop_graphs) == 45
    kinds = {item.rule.kind for item in program.matched.insertions}
    assert {"UVmass", "R2", "UVloop"} <= kinds
    assert all(
        item.rule.has_validated_tensor_basis for item in program.matched.insertions
    )
    points, scales = _inputs(reference)
    rows = reference["rows"]["gg"]
    virtual = _carrier(_witness_virtual(program, prepared), arithmetic, precision)
    result = virtual.evaluate_decimal(points, scales)
    assert all(status == "success" for row in result.statuses for _, status in row)
    _compare(result, rows, _COMPONENTS[1:3] + _COMPONENTS[4:])
    if arithmetic != "double":
        return
    # The Identity(1,2) insertions alone: top mass UV counterterm and the R2
    # slash-plus-mass two-point, each against its isolated MadLoop selection.
    for select, name, key in (
        (_is_mass_counterterm, "uv", "uv_mass"),
        (_is_r2_fermion_two_point, "r2", "r2_quark_two_point"),
    ):
        isolated = _witness_virtual(program, prepared, select).evaluate_decimal(
            points, scales
        )
        _compare(isolated, rows, ((name, key),))
        assert all(
            Decimal(value) == 0
            for value in isolated.uv_external_fields[:, :, 0].reshape(-1)
        )


@pytest.mark.parametrize("arithmetic,precision,tolerance", _CARRIERS)
def test_gluon_top_pair_complete(programs, reference, arithmetic, precision, tolerance):
    program = _carrier(programs("gg"), arithmetic, precision)
    points, scales = _inputs(reference)
    try:
        result = program.evaluate_decimal(points, scales)
    except ArithmeticError as error:
        # oneloopmaster 09b2c74 returns a non-finite D0 for the two exactly
        # on-shell infrared boxes with adjacent internal top masses (box11 is
        # reached only after its chart exchange). The published fix,
        # alphal00p/oneloopmaster#10 (cd945f4), passes this test in all three
        # carriers; remove this branch with the pin update.
        if not re.search(r"loop reduction failed.*MasterFailed", str(error)):
            raise
        pytest.xfail("oneloopmaster D0 adjacent-mass IR chart (oneloopmaster#10)")
    assert all(status == "success" for row in result.statuses for _, status in row)
    _compare(result, reference["rows"]["gg"], _COMPONENTS)
    _check_exact_identities(result, "gg", tolerance)
