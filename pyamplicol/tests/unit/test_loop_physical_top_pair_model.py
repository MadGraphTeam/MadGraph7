# SPDX-License-Identifier: 0BSD
"""Model-rule top pairs with internal massive tops against independent MadLoop.

``g g > t t~`` exercises massive internal top propagators, the model's top
mass counterterm and the R2 slash-plus-mass two-point. ``u u~ > t t~`` adds the
top-loop gluon self-energy. ``e+ e- > t t~ h`` adds the top Yukawa and Goldstone
(FFS) vertices and their QCD counterterms. References are binary64 MadLoop
values on exactly on-shell points; higher-precision carriers additionally check
exact identities.
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
_QCD = ({"QCD": 2, "QED": 0}, {"QCD": 4, "QED": 0})
# fixture, incoming, outgoing, external masses, Born/loop order bounds and the
# exact infrared double pole in units of alpha_s/(2 pi) * Born: -2 C_A for two
# external gluons, -2 C_F for two massless quarks, none for massive emitters.
_PROCESSES = {
    "gg": ("top-pair-model.json", (21, 21), (6, -6), (0, 0, 173, 173), _QCD, "-6"),
    "uu": ("top-pair-model.json", (2, -2), (6, -6), (0, 0, 173, 173), _QCD, "-8/3"),
    "ee": (
        "top-pair-higgs-model.json",
        (-11, 11),
        (6, -6, 25),
        (0, 0, 173, 173, 125),
        ({"QCD": 0, "QED": 3}, {"QCD": 2, "QED": 3}),
        "0",
    ),
}
_CARRIERS = [
    ("double", None, "2e-12"),
    ("double-double", None, "3e-25"),
    ("arbitrary", 80, "1e-60"),
]


@pytest.fixture(scope="module")
def reference():
    cache = {}

    def load(key):
        name, _incoming, _outgoing, masses, _orders, _double = _PROCESSES[key]
        if name not in cache:
            data = json.loads(
                (
                    Path(__file__).resolve().parents[1] / "fixtures/loop" / name
                ).read_text()
            )
            with localcontext() as context:
                context.prec = 60
                for point in data["points"]:
                    legs = [[Decimal(x) for x in leg] for leg in point]
                    for leg, mass in zip(legs, masses, strict=True):
                        assert leg[0] ** 2 - sum(x * x for x in leg[1:]) == mass**2
                    for mu in range(4):
                        assert legs[0][mu] + legs[1][mu] == sum(
                            leg[mu] for leg in legs[2:]
                        )
            cache[name] = data
        return cache[name]

    return load


@pytest.fixture(scope="module")
def programs(reference):
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
    models, cache = {}, {}

    def program(key):
        if key not in cache:
            _name, incoming, outgoing, _masses, orders, _double = _PROCESSES[key]
            data = reference(key)
            bounds = json.dumps(orders, sort_keys=True)
            if bounds not in models:
                models[bounds] = prepare_loop_request_model(
                    view.derived,
                    born_order_bounds=orders[0],
                    loop_order_bounds=orders[1],
                    fixed_parameters=data["fixed_parameters"],
                )
            prepared = models[bounds]
            cache[key] = (
                prepared,
                prepared.compile(
                    incoming,
                    outgoing,
                    selected_corrections=["QCD"],
                    parameters=data["parameters"],
                    arithmetic="double",
                    reduction_engine="fast",
                    coefficient_engine="jit",
                ),
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
            [E(_PROCESSES[key][-1]) * E("118/1000") / (2 * E("pi"))],
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
                < (abs(expected) or abs(born)) * threshold
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
    _prepared, program = programs("uu")
    program = _carrier(program, arithmetic, precision)
    bare = program.source_bare.bare
    assert len(bare.process.born) == 1 and len(bare.loop_graphs) == 14
    data = reference("uu")
    result = program.evaluate_decimal(*_inputs(data))
    assert all(status == "success" for row in result.statuses for _, status in row)
    _compare(result, data["rows"]["uu"], _COMPONENTS)
    _check_exact_identities(result, "uu", tolerance)


def _witness_virtual(program, prepared, select=None):
    """Counterterm assembly over one smallest loop witness per insertion.

    The actual matched source insertions keep an admitted witness loop; all
    other loop graphs are omitted, so the bare component is incomplete and is
    not compared. ``select`` restricts the insertion set without changing any.
    """
    from pyamplicol.loops.physical_counterterms import NativeModelVirtualProgram
    from pyamplicol.loops.physical_process import NativeBareProcessProgram
    from pyamplicol.loops.wavefunction import match_external_field_counterterms

    bare = program.source_bare.bare
    size = {
        bare.loop_graphs[index].id: len(
            bare.process.families[family.index].denominators
        )
        for family in bare.families
        for index in family.graph_indices
    }
    insertions = tuple(program.matched.insertions)
    witnesses = {
        min(item.witness_loop_ids, key=lambda graph: (size[graph], graph))
        for item in insertions
        if item.witness_loop_ids
    }
    loops = tuple(graph for graph in bare.process.loops if graph.id in witnesses)
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
@pytest.mark.parametrize("key", ["gg", "ee"])
def test_counterterms_with_internal_massive_top(
    programs, reference, key, arithmetic, precision, tolerance
):
    prepared, program = programs(key)
    kinds = {item.rule.kind for item in program.matched.insertions}
    assert {"UVmass", "R2", "UVloop"} <= kinds
    assert all(
        item.rule.has_validated_tensor_basis for item in program.matched.insertions
    )
    data = reference(key)
    points, scales = _inputs(data)
    rows = data["rows"][key]
    virtual = _carrier(_witness_virtual(program, prepared), arithmetic, precision)
    result = virtual.evaluate_decimal(points, scales)
    assert all(status == "success" for row in result.statuses for _, status in row)
    _compare(result, rows, _COMPONENTS[1:3] + _COMPONENTS[4:])
    if arithmetic != "double":
        return
    # The Identity(1,2) insertions alone: top mass UV counterterm and the R2
    # slash-plus-mass two-point, each against its isolated MadLoop selection.
    for select, name, component in (
        (_is_mass_counterterm, "uv", "uv_mass"),
        (_is_r2_fermion_two_point, "r2", "r2_quark_two_point"),
    ):
        isolated = _witness_virtual(program, prepared, select).evaluate_decimal(
            points, scales
        )
        _compare(isolated, rows, ((name, component),))
        assert all(
            Decimal(value) == 0
            for value in isolated.uv_external_fields[:, :, 0].reshape(-1)
        )


_KNOWN_MASTER_FAILURES = {
    # oneloopmaster 09b2c74 returns a non-finite D0 for the two exactly on-shell
    # infrared boxes with adjacent internal top masses (box11 is reached only
    # after its chart exchange). The published fix alphal00p/oneloopmaster#10
    # (cd945f4) passes this test in all three carriers.
    "gg": "oneloopmaster D0 adjacent-mass IR chart (oneloopmaster#10)",
    # The soft-divergent box family with internal masses (m, m, 0, m) and
    # on-shell top legs returns MasterFailed at the third point (also with #10).
    # NOT attributed yet: at the other two points the bare loop is finite but
    # disagrees with MadLoop (it lacks part of the soft single pole), with both
    # reduction engines. Once the master failure is gone this test compares and
    # therefore reports that discrepancy instead of hiding it.
    "ee": "MasterFailed for the soft (m,m,0,m) box; bare also unattributed",
}


@pytest.mark.parametrize("arithmetic,precision,tolerance", _CARRIERS)
@pytest.mark.parametrize("key", ["gg", "ee"])
def test_complete_with_internal_massive_top(
    programs, reference, key, arithmetic, precision, tolerance
):
    _prepared, program = programs(key)
    program = _carrier(program, arithmetic, precision)
    data = reference(key)
    try:
        result = program.evaluate_decimal(*_inputs(data))
    except ArithmeticError as error:
        # Only the identified dependency failure is reported as expected; any
        # other failure, or a finite wrong result, fails normally.
        if not re.search(r"loop reduction failed.*MasterFailed", str(error)):
            raise
        pytest.xfail(_KNOWN_MASTER_FAILURES[key])
    assert all(status == "success" for row in result.statuses for _, status in row)
    _compare(result, data["rows"][key], _COMPONENTS)
    _check_exact_identities(result, key, tolerance)
