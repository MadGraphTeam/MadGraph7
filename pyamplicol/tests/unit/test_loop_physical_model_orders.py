# SPDX-License-Identifier: 0BSD
"""Source-complete QCD corrections to the mixed gluon/photon Born catalogue."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from pyamplicol.loops.orders import BornLoopPair, compile_order_plan
from pyamplicol.loops.physical_counterterms import NativeModelVirtualProgram
from pyamplicol.loops.physical_process import NativeBareProcessProgram


@pytest.fixture(scope="module")
def control():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("native generation requires licensed Symbolica")
    backend = pytest.importorskip("_loop_reduction")
    path = os.environ.get("PYAMPLICOL_NLO_UFO_REFERENCE")
    if not path:
        pytest.skip("set PYAMPLICOL_NLO_UFO_REFERENCE to loop_sm")
    from symbolica import E, S
    from symbolica.community import hepkit as h

    from pyamplicol.loops.counterterms import (
        match_counterterms,
        prepare_counterterm_model,
    )
    from pyamplicol.loops.graphs import build_loop_process
    from pyamplicol.loops.model import build_loop_model
    from pyamplicol.loops.model_correspondence import (
        CountertermPrimitiveValidation,
        OrdinaryBasisRepresentative,
    )
    from pyamplicol.loops.wavefunction import match_external_field_counterterms
    from pyamplicol.models.loop_import import load_raw_loop_model

    source = build_loop_model(load_raw_loop_model(path))
    model = h.Model.qcd_qed()
    representatives, validations = [], []
    native_rules = {
        (-1, 1, 21): "V_74",
        (-2, 2, 21): "V_135",
        (-6, 6, 21): "V_137",
        (-1, 1, 22): "V_71",
        (-2, 2, 22): "V_132",
        (-6, 6, 22): "V_134",
    }
    for entry in source.kernels:
        pdgs = tuple(
            source.raw.collections["particles"][i]["fields"]["pdg_code"]
            for i in entry.particle_ids
        )
        if entry.kind == "tree" and pdgs in native_rules:
            representatives.append(
                OrdinaryBasisRepresentative(
                    entry.key,
                    native_rules[pdgs],
                    0,
                    0,
                    "native-qcd-qed-vector-current-control",
                )
            )
        if (
            entry.kind != "tree"
            and pdgs == (21, 21)
            and entry.color == "Tr(1,2)"
            and entry.lorentz
            in {"P(-1,1)*P(-1,1)*Metric(1,2)", "P(1,1)*P(2,1)", "Metric(1,2)"}
        ):
            validations.append(
                CountertermPrimitiveValidation(
                    entry.key,
                    entry.color,
                    entry.lorentz,
                    pdgs,
                    (3, 3),
                    (8, 8),
                    "local-gluon-two-point-aloha-control",
                )
            )
    prepared = prepare_counterterm_model(
        source,
        model,
        ordinary_basis_representatives=representatives,
        primitive_validations=validations,
    )
    alpha, inverse_alpha, top = S(
        "mixed_model::alpha_s", "mixed_model::aEWM1", "mixed_model::mt"
    )
    strong, electromagnetic = (
        (4 * E("pi") * alpha) ** E("1/2"),
        (4 * E("pi") / inverse_alpha) ** E("1/2"),
    )
    expressions = {f"UFO::{name}": E("0") for name in ("MB", "MC", "Me", "MM", "MTA")}
    expressions["UFO::MT"] = top
    for record in json.loads(model.to_json())["couplings"]:
        coupling = model.coupling(record["name"])
        expressions[coupling.symbol.get_name()] = coupling.expression.replace(
            S("UFO::G"), strong
        ).replace(S("UFO::ee"), electromagnetic)
    bindings = [
        (S("UFO::G"), strong),
        (S("UFO::ee"), electromagnetic),
        (S("UFO::MT"), top),
        (S("UFO::MB"), E("0")),
        (S("UFO::MC"), E("0")),
        (S("UFO::Ncol"), E("3")),
        (S("UFO::CA"), E("3")),
        (S("UFO::CF"), E("4/3")),
        (S("UFO::TF"), E("1/2")),
        (S("UFO::lhv"), E("1")),
    ]

    def setup(outgoing):
        process = build_loop_process(
            prepared.process(["u", "u~"], outgoing),
            ordinary_model=model,
            selected_corrections=["QCD"],
            perturbative_orders={"QCD": 1},
        )
        matched = match_counterterms(
            prepared,
            prepared.process(["u", "u~"], outgoing, counterterms=True),
            process,
            selected_corrections=["QCD"],
        )
        fields = match_external_field_counterterms(
            source,
            process,
            selected_corrections=["QCD"],
            allowed_amplitude_orders={
                p.loop for s in process.orders.sectors for p in s.pairs
            },
        )
        return process, matched.insertions, fields.insertions

    return backend, setup, expressions, (alpha, inverse_alpha, top), bindings


def _program(
    control, catalogue, *, process=None, insertions=None, fields=None, **options
):
    from symbolica import E, S

    backend, _, expressions, inputs, bindings = control
    source_process, source_insertions, source_fields = catalogue
    bare = NativeBareProcessProgram(
        source_process if process is None else process,
        state_kinds=("u", "v_bar", "u_bar", "v"),
        initial_average=E("36"),
        model_expressions=expressions,
        model_inputs=inputs,
        backend=backend,
        **options,
    )
    return NativeModelVirtualProgram(
        bare,
        source_insertions if insertions is None else insertions,
        source_bindings=(
            *bindings,
            (S("UFO::MU_R"), bare.scale_squared_symbol ** E("1/2")),
        ),
        external_fields=source_fields if fields is None else fields,
    )


@pytest.fixture(scope="module")
def catalogue(control):
    return control[1](["d", "d~"])


def _points(name="uudd_qcd.json"):
    path = Path(__file__).resolve().parents[2] / "tools/loop_reference" / name
    return np.asarray(json.loads(path.read_text())["points"], dtype=str)


def _order(qcd, qed):
    return tuple((name, power) for name, power in (("QCD", qcd), ("QED", qed)) if power)


@pytest.mark.parametrize(
    "arithmetic,precision,engine",
    [
        ("double", None, "fast"),
        ("double-double", None, "generic"),
        ("arbitrary", 60, "generic"),
    ],
)
def test_complete_mixed_order_axes_and_components(
    control, catalogue, arithmetic, precision, engine
):
    program = _program(
        control,
        catalogue,
        arithmetic=arithmetic,
        precision=precision,
        reduction_engine=engine,
    )
    assert len(program.bare.loop_graphs) == 16
    points = _points()
    result = program.evaluate_ordered_decimal(
        points, ["1000000"] * 3, model_values=["0.118", "132.507", "173"]
    )
    assert result.born.shape == (3, 3, 2)
    assert result.virtual.shape == (3, 3, 3, 2)
    assert result.orders.correction_names == ("QCD",)
    assert set(result.virtual_orders) == {_order(6, 0), _order(4, 2), _order(2, 4)}
    assert all(Decimal(x) == 0 for x in result.born_order(_order(2, 2)).flat)
    assert all(Decimal(x) == 0 for x in result.uv_external_fields.flat)
    assert all(status == "success" for row in result.statuses for _, status in row)
    assert len(result.channels) == len(result.statuses[0])
    with pytest.raises(NotImplementedError, match="single-sector model"):
        program.evaluate_decimal(
            points, ["1000000"] * 3, model_values=["0.118", "132.507", "173"]
        )
    with pytest.raises(ValueError, match="absent"):
        result.virtual_order(_order(0, 6))
    with localcontext() as context:
        context.prec = 100
        tolerance = Decimal(
            {"double": "1e-12", "double-double": "1e-29", "arbitrary": "1e-56"}[
                arithmetic
            ]
        )
        for name in ("virtual", "bare", "uv", "r2", "uv_local", "uv_external_fields"):
            values = getattr(result, name)
            assert not values.flags.writeable
            summed = getattr(result.sum_orders(), name)
            for point in range(3):
                for pole in range(3):
                    target = sum(Decimal(x) for x in values[point, :, pole, 0])
                    assert abs(Decimal(summed[point, pole, 0]) - target) < tolerance
        for index in np.ndindex(result.virtual.shape):
            target = (
                Decimal(result.bare[index])
                + Decimal(result.uv[index])
                + Decimal(result.r2[index])
            )
            assert abs(Decimal(result.virtual[index]) - target) < tolerance
    path = os.environ.get("PYAMPLICOL_MIXED_UUDD_QCD_REFERENCE")
    if path:
        reference = json.loads(Path(path).read_text())
        for order, index in (
            (_order(6, 0), "1"),
            (_order(4, 2), "2"),
            (_order(2, 4), "3"),
        ):
            for name, key in (
                ("virtual", "full"),
                ("bare", "bare"),
                ("r2", "r2"),
                ("uv", "uv"),
                ("uv_local", "uv_local"),
                ("uv_external_fields", "fields"),
            ):
                expected = [
                    [
                        row[key][index][pole]
                        for pole in ("finite", "single_pole", "double_pole")
                    ]
                    for row in reference["rows"]
                ]
                sector = result.virtual_orders.index(order)
                np.testing.assert_allclose(
                    getattr(result, name)[:, sector, :, 0].astype(float),
                    expected,
                    rtol=3e-10,
                    atol=3e-14,
                )
    # The photon Born has two independently corrected quark currents. Its
    # complete sector is twice the massless form factor, an all-precision oracle.
    scales = ["900000", "2100000"]
    alpha, inverse = "0.100000000000000000000000001", "132.507000000000000000000000003"
    rebound = program.evaluate_ordered_decimal(
        points[1:], scales, model_values=[alpha, inverse, "173"]
    )
    tolerance = Decimal(
        {"double": "3e-11", "double-double": "3e-26", "arbitrary": "3e-51"}[arithmetic]
    )
    with localcontext() as context:
        context.prec = 120
        for i, (point, scale) in enumerate(zip(points[1:], scales, strict=True)):
            target = _photon_targets(point, scale, alpha, inverse)
            actual = [
                rebound.born_order(_order(0, 4))[i, 0],
                *rebound.virtual_order(_order(2, 4))[i, :, 0],
                rebound.r2[i, rebound.virtual_orders.index(_order(2, 4)), 0, 0],
            ]
            for value, expected in zip(actual, target, strict=True):
                assert abs(Decimal(value) - expected) < abs(expected) * tolerance
    empty = program.evaluate_ordered_decimal(
        points[:0], [], model_values=["0.118", "132.507", "173"]
    )
    assert empty.virtual.shape == (0, 3, 3, 2) and empty.sum_orders().virtual.shape == (
        0,
        3,
        2,
    )


def _photon_targets(point, scale, alpha_s, inverse_alpha):
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo as exact

    p = [[exact(value) for value in leg] for leg in point]

    def square(v):
        return v[0] ** 2 - sum((x * x for x in v[1:]), E("0"))

    s, t, u = [
        square([a + sign * b for a, b in zip(p[0], p[leg], strict=True)])
        for leg, sign in ((1, 1), (2, -1), (3, -1))
    ]
    born = (
        E("8/81") * (4 * E("pi") / exact(inverse_alpha)) ** 2 * (t * t + u * u) / s**2
    )
    ratio, log = exact(alpha_s) * E("4/3") / (2 * E("pi")), (exact(scale) / s).log()
    targets = [
        born,
        2 * born * ratio * (E("pi") ** 2 - 8 - 3 * log - log**2),
        2 * born * ratio * (-3 - 2 * log),
        -4 * born * ratio,
        -4 * born * ratio,
    ]
    return [
        v.to_decimal_tuple()[0]
        for v in Expression.evaluator_multiple(
            targets, [], n_cores=1, jit_compile=False
        ).evaluate_complex_with_prec([], 110)
    ]


def test_directed_complete_mixed_orders_and_complex_effective_cross_terms(
    control, catalogue
):
    from symbolica import E

    from pyamplicol.color.connections import ColorConnection, EmitGluon
    from pyamplicol.loops.model_expressions import LaurentExpression
    from pyamplicol.loops.physical_correlations import directed_color_legs
    from pyamplicol.loops.physical_model_correlations import NativeDirectedModelProgram

    # A deliberate complex multiplier on effective rules is an algebra control,
    # not a claim about an SM renormalization convention. The actual source QCD
    # completeness is checked separately by uudd/top identity histories.
    original, insertions, fields = catalogue
    modified = tuple(
        replace(
            item,
            rule=replace(
                item.rule,
                couplings=tuple(
                    (
                        name,
                        replace(
                            entry,
                            coupling=replace(
                                entry.coupling,
                                value=LaurentExpression(
                                    entry.coupling.value.expression * E("1+1i")
                                ),
                            ),
                        ),
                    )
                    for name, entry in item.rule.couplings
                ),
            ),
        )
        for item in insertions
    )
    model = _program(control, catalogue, insertions=modified, reduction_engine="fast")
    points, scales, inputs = (
        _points()[:2],
        ["1000000", "900000"],
        ["0.118", "132.507", "173"],
    )
    baseline = model.evaluate_ordered_decimal(points, scales, model_values=inputs)

    def histories(bare):
        legs = directed_color_legs(bare)
        identity = ColorConnection(legs)
        left = ColorConnection(legs, (EmitGluon(0, -1), EmitGluon(0, -2)))
        right = ColorConnection(legs, (EmitGluon(1, -1), EmitGluon(2, -2)))
        return ((identity, identity), (left, right), (right, left))

    directed = NativeDirectedModelProgram(model, histories(model.bare))
    result = directed.evaluate_decimal(points, scales, model_values=inputs)
    with localcontext() as context:
        context.prec = 80
        for component in (
            "born",
            "virtual",
            "bare",
            "uv",
            "r2",
            "uv_local",
            "uv_external_fields",
        ):
            actual = getattr(result, component)
            expected = getattr(baseline, component)
            for value, target in zip(actual[:, 0].flat, expected.flat, strict=True):
                assert abs(Decimal(value) - Decimal(target)) < Decimal("4e-11")
            for index in np.ndindex(actual[:, 1].shape):
                sign = 1 if index[-1] == 0 else -1
                assert abs(
                    Decimal(actual[:, 1][index]) - sign * Decimal(actual[:, 2][index])
                ) < Decimal("4e-11")
        assert max(
            abs(Decimal(value)) for value in result.r2[:, 1, :, :, 1].flat
        ) > Decimal("1e-5")
    # Isolate the two Born origins of S42; their directed effective terms must
    # recombine using the same exact masks as the bare reductions.
    isolated = []
    for born_order, loop_order in (
        (_order(2, 0), _order(2, 2)),
        (_order(0, 2), _order(4, 0)),
    ):
        plan = compile_order_plan(
            selected_corrections=["QCD"],
            perturbative_orders={"QCD": 1},
            born_orders=original.orders.born_orders,
            contributions=original.orders.contributions,
            allowed_pairs=[BornLoopPair(born_order, loop_order)],
        )
        selected = _program(
            control,
            catalogue,
            process=replace(original, orders=plan),
            insertions=tuple(i for i in modified if i.graph.order == loop_order),
            fields=tuple(i for i in fields if i.order == loop_order),
            reduction_engine="fast",
        )
        selected = NativeDirectedModelProgram(selected, histories(selected.bare))
        isolated.append(selected.evaluate_decimal(points, scales, model_values=inputs))
    sector = tuple(item.order for item in result.orders.virtual).index(_order(4, 2))
    with localcontext() as context:
        context.prec = 80
        for component in (
            "virtual",
            "bare",
            "uv",
            "r2",
            "uv_local",
            "uv_external_fields",
            "first",
            "second",
        ):
            combined = getattr(result, component)[:, :, sector]
            left, right = (getattr(item, component)[:, :, 0] for item in isolated)
            for value, a, b in zip(combined.flat, left.flat, right.flat, strict=True):
                assert abs(Decimal(value) - Decimal(a) - Decimal(b)) < Decimal("4e-11")


def test_complete_mixed_sector_retains_both_explicit_born_origins(control, catalogue):
    path = os.environ.get("PYAMPLICOL_MIXED_UUDD_QCD_REFERENCE")
    if not path:
        pytest.skip(
            "set PYAMPLICOL_MIXED_UUDD_QCD_REFERENCE for isolated Born-source control"
        )
    reference = json.loads(Path(path).read_text())
    original, insertions, _ = catalogue
    values = []
    for source, born_order, loop_order in (
        ("B20", _order(2, 0), _order(2, 2)),
        ("B02", _order(0, 2), _order(4, 0)),
    ):
        plan = compile_order_plan(
            selected_corrections=["QCD"],
            perturbative_orders={"QCD": 1},
            born_orders=original.orders.born_orders,
            contributions=original.orders.contributions,
            allowed_pairs=[BornLoopPair(born_order, loop_order)],
        )
        program = _program(
            control,
            catalogue,
            process=replace(original, orders=plan),
            insertions=tuple(i for i in insertions if i.graph.order == loop_order),
            reduction_engine="fast",
        )
        result = program.evaluate_ordered_decimal(
            _points(), ["1000000"] * 3, model_values=["0.118", "132.507", "173"]
        )
        assert result.virtual_orders == (_order(4, 2),)
        expected = [
            [
                row["by_born_order"][source]["2"][pole]
                for pole in ("finite", "single_pole", "double_pole")
            ]
            for row in reference["rows"]
        ]
        actual = result.virtual_order(_order(4, 2))[:, :, 0].astype(float)
        np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=3e-14)
        values.append(actual)
        if source == "B20":
            # Colour orthogonality makes this structural pair identically zero;
            # it remains present rather than disappearing from the catalogue.
            np.testing.assert_allclose(actual, 0, atol=3e-14)
    assert np.max(np.abs(values[1])) > 1e-6


def test_counterterm_witness_and_source_correction_are_not_order_inference(
    control, catalogue
):
    program = _program(control, catalogue, reduction_engine="fast")
    item = next(i for i in program.insertions if i.rule.kind != "UVtree")
    for changed, message in (
        (replace(item, witness_loop_ids=("unadmitted-graph",)), "loop graph witness"),
        (
            replace(item, rule=replace(item.rule, perturbation_type="QED")),
            "selected set",
        ),
        (
            replace(
                item,
                rule=replace(
                    item.rule, kind="UVtree", perturbation_type=None, order=_order(0, 2)
                ),
            ),
            "UVtree",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            NativeModelVirtualProgram(
                program.bare, [changed], source_bindings=program.source_bindings
            )


def test_external_fields_keep_right_born_graph_and_left_pair_mask(control):
    from symbolica import E, Expression

    from pyamplicol.loops.model_expressions import parse_exact_ufo as exact

    catalogue = control[1](["t", "t~"])
    program = _program(control, catalogue, arithmetic="arbitrary", precision=60)
    assert len(program.external_fields) == 4
    points, scales = _points("top_pair_qcd.json")[:2], ["900000", "2100000"]
    alpha = "0.100000000000000000000000001"
    result = program.evaluate_ordered_decimal(
        points, scales, model_values=[alpha, "132.507", "173"]
    )
    factors = [
        v.to_decimal_tuple()[0]
        for v in Expression.evaluator_multiple(
            [
                -exact(alpha)
                / (2 * E("pi"))
                * E("4/3")
                * (4 + 3 * (exact(scale) / E("173^2")).log())
                for scale in scales
            ],
            [],
            n_cores=1,
            jit_compile=False,
        ).evaluate_complex_with_prec([], 110)
    ]
    pole = (
        Expression.evaluator_multiple(
            [-exact(alpha) / (2 * E("pi")) * 4], [], n_cores=1, jit_compile=False
        )
        .evaluate_complex_with_prec([], 110)[0]
        .to_decimal_tuple()[0]
    )
    with localcontext() as context:
        context.prec = 120
        for born_order, virtual_order in (
            (_order(4, 0), _order(6, 0)),
            (_order(0, 4), _order(2, 4)),
        ):
            sector = result.virtual_orders.index(virtual_order)
            for i, factor in enumerate(factors):
                born = Decimal(result.born_order(born_order)[i, 0])
                for epsilon, coefficient in ((0, factor), (1, pole)):
                    expected = born * coefficient
                    assert abs(
                        Decimal(result.uv_external_fields[i, sector, epsilon, 0])
                        - expected
                    ) < abs(expected) * Decimal("3e-53")
    # Retain only the photon graph's physical field insertions, without changing
    # any bare graph, current or pair. They cannot renormalize the QCD Born ket.
    photon_fields = tuple(
        i for i in program.external_fields if i.born.order == _order(0, 2)
    )
    isolated = NativeModelVirtualProgram(
        program.bare,
        program.insertions,
        source_bindings=program.source_bindings,
        external_fields=photon_fields,
    )
    filtered = isolated.evaluate_ordered_decimal(
        points, scales, model_values=[alpha, "132.507", "173"]
    )
    for order in (_order(6, 0), _order(4, 2)):
        assert all(
            Decimal(v) == 0
            for v in filtered.uv_external_fields[
                :, filtered.virtual_orders.index(order)
            ].flat
        )
    index = filtered.virtual_orders.index(_order(2, 4))
    np.testing.assert_array_equal(
        filtered.uv_external_fields[:, index], result.uv_external_fields[:, index]
    )
