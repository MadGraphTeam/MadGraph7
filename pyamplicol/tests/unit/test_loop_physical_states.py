# SPDX-License-Identifier: 0BSD
"""Variable-width native states, exact Born controls and scalar loop poles."""

import json
from decimal import Decimal, localcontext

import numpy as np
import pytest

from pyamplicol.loops.graphs import build_loop_process
from pyamplicol.loops.physical_process import NativeBareProcessProgram
from pyamplicol.loops.physical_states import ExternalStateLayout


@pytest.fixture(scope="module")
def dependencies():
    symbolica = pytest.importorskip("symbolica")
    if not symbolica.is_licensed():
        pytest.skip("licensed dependency graph generation required")
    from symbolica.community import hepkit

    from pyamplicol.loops.native import load_loop_native

    return symbolica, hepkit, load_loop_native()


@pytest.fixture(scope="module")
def catalogues(dependencies):
    s, h, _ = dependencies
    scalar = h.Model.phi4()
    payload = json.loads(h.Model.standard_model().to_json())
    # An isolated scalar-vector contact model, not an SM process prediction.
    payload["vertex_rules"] = [
        rule
        for rule in payload["vertex_rules"]
        if sorted(rule["particles"]) == ["H", "H", "Z", "Z"]
    ]
    vector = h.Model.from_json(json.dumps(payload))
    result = {}
    for name, model, particles, order, kinds, average in (
        ("scalar", scalar, ["phi", "phi"], "SCALAR", ("scalar",) * 4, 2),
        (
            "mixed",
            vector,
            ["H", "Z"],
            "QED",
            ("scalar", "epsilon", "scalar", "epsilon_bar"),
            3,
        ),
    ):
        process = build_loop_process(
            model.process(particles, particles),
            selected_corrections=[order],
            perturbative_orders={order: 1},
        )
        expressions = {
            model.particle(particle).mass.get_name(): s.E("5") for particle in particles
        }
        expressions.update(
            {
                model.coupling(coupling).symbol.get_name(): s.E("1i")
                * s.S("state_test::coupling")
                * (-1 if name == "scalar" else 1)
                for rule in model.vertex_rules
                for row in rule.couplings
                for coupling in row
                if coupling is not None
            }
        )
        result[name] = process, expressions, kinds, s.E(str(average))
    return result


def _program(dependencies, catalogues, name, arithmetic="double"):
    s, _, backend = dependencies
    process, expressions, kinds, average = catalogues[name]
    return NativeBareProcessProgram(
        process,
        state_kinds=kinds,
        initial_average=average,
        model_expressions=expressions,
        model_inputs=(s.S("state_test::coupling"),),
        backend=backend,
        arithmetic=arithmetic,
        precision=70 if arithmetic == "arbitrary" else None,
        reduction_engine="fast" if arithmetic == "double" else "generic",
    )


def _points():
    return np.asarray(
        [
            [
                ["13", "0", "0", "12"],
                ["13", "0", "0", "-12"],
                ["13", "7.2", "0", "9.6"],
                ["13", "-7.2", "0", "-9.6"],
            ],
            [
                ["13", "0", "0", "12"],
                ["13", "0", "0", "-12"],
                ["13", "12", "0", "0"],
                ["13", "-12", "0", "0"],
            ],
        ],
        dtype=object,
    )


@pytest.mark.parametrize("arithmetic", ("double", "double-double", "arbitrary"))
@pytest.mark.parametrize("name", ("scalar", "mixed"))
def test_native_scalar_and_massive_vector_processes(
    dependencies, catalogues, arithmetic, name
):
    s, _, _ = dependencies
    program = _program(dependencies, catalogues, name, arithmetic)
    assert program.initial_spin_average == (1 if name == "scalar" else 3)
    assert len(program.helicities) == (1 if name == "scalar" else 9)
    assert program.state_layout.components == (4 if name == "scalar" else 10)
    assert len(program.external_states) == (0 if name == "scalar" else 2)
    points = _points()
    coupling = "0.700000000000000000000000001"
    result = program.evaluate_decimal(points, ["676", "700"], model_values=[coupling])
    tolerance = Decimal(
        {"double": "3e-13", "double-double": "3e-27", "arbitrary": "3e-61"}[arithmetic]
    )
    with localcontext() as context:
        context.prec = 100
        for row, dot in enumerate((Decimal("53.8"), Decimal("169"))):
            # Scalar unit states; vector sum from physical completeness,
            # averaged over all three incoming polarizations.
            expected = Decimal(coupling) ** 2 * (
                Decimal("0.5")
                if name == "scalar"
                else (2 + dot**2 / Decimal(5) ** 4) / 3
            )
            assert abs(Decimal(result.born[row, 0]) - expected) < expected * tolerance
            assert abs(Decimal(result.born[row, 1])) < tolerance
        if name == "scalar":
            from pyamplicol.loops.model_expressions import parse_exact_ufo

            # Three equal symmetry-1/2 bubbles, each with I2 pole +1.
            expected = -3 * parse_exact_ufo(coupling) ** 3 / (32 * s.E("pi") ** 2)
            value = s.Expression.evaluator_multiple(
                [expected], [], n_cores=1, jit_compile=False
            ).evaluate_complex_with_prec([], 100)[0]
            expected = value.to_decimal_tuple()[0]
            for row in result.virtual:
                assert abs(Decimal(row[1, 0]) - expected) < abs(expected) * tolerance
                assert Decimal(row[2, 0]) == 0
    assert all(status == "success" for row in result.statuses for _, status in row)
    empty = program.evaluate_decimal(points[:0], [], model_values=[coupling])
    assert empty.born.shape == (0, 2) and empty.virtual.shape == (0, 3, 2)
    # Models own normalization conventions; layout metadata does not constrain
    # additional color or symmetry factors by an integer divisibility guess.
    assert program.initial_average == catalogues[name][3]


def test_layout_named_fixed_zero_ports_and_spin_kind_checks(dependencies, catalogues):
    s, _, _ = dependencies
    process, expressions, kinds, _ = catalogues["mixed"]
    layout = ExternalStateLayout.from_graph(process.born[0], kinds, expressions)
    assert [leg.offset for leg in layout.legs] == [0, 1, 5, 6]
    assert layout.initial_spin_average == 3
    vector_mass = next(
        edge.particle.mass
        for edge in process.born[0].native.external_edges
        if edge.particle.spin == 3
    )
    zero = ExternalStateLayout.from_graph(
        process.born[0], kinds, {**expressions, vector_mass.get_name(): s.E("0")}
    )
    assert zero.initial_spin_average == 2 and len(zero.helicities) == 4
    adjustable = ExternalStateLayout.from_graph(
        process.born[0], kinds, {vector_mass.get_name(): s.S("state_test::mass")}
    )
    assert adjustable.initial_spin_average == 3
    with pytest.raises(NotImplementedError, match="compatible"):
        ExternalStateLayout.from_graph(process.born[0], ("u",) * 4, expressions)
    with pytest.raises(ValueError, match="missing"):
        layout.state_positions(())
    with pytest.raises(ValueError, match="disagree"):
        layout.state_positions(((0, (s.S("state_test::unwanted"),)),))


def test_literal_source_column_packing_around_scalar_slots(dependencies, catalogues):
    # Exercise only the column-packing helper; the raw control lacks the
    # model-owned SourceIR required by the source plan.
    from pyamplicol.loops.spin_sources import (
        NativeSpinSourcePlan,
        replace_vector_sources,
    )

    program = _program(dependencies, catalogues, "mixed")
    native = program.backend.NativeBatch
    ids = [91, 7]
    markers = native.from_decimal(
        [(str(i), "0") for i in range(20)], 2, 10, row_ids=ids
    )
    source = native.from_decimal(
        [(str(i), str(-i)) for i in range(101, 109)], 2, 4, row_ids=[8, 4]
    )
    values = replace_vector_sources(
        markers,
        {2: source},
        point_rows=[1, 0],
        lane_ids=ids,
        state_layout=program.state_layout,
        backend=program.backend,
    )
    original, replaced = markers.to_decimal(), values.to_decimal()
    assert values.row_ids == ids
    for row in range(2):
        assert replaced[10 * row] == original[10 * row]
        assert (
            replaced[10 * row + 5 : 10 * (row + 1)]
            == original[10 * row + 5 : 10 * (row + 1)]
        )
        assert (
            replaced[10 * row + 1 : 10 * row + 5]
            == source.to_decimal()[4 * (1 - row) : 4 * (2 - row)]
        )
    with pytest.raises(ValueError, match="model-owned SourceIR"):
        NativeSpinSourcePlan(program, ((2,),))


@pytest.mark.parametrize("arithmetic", ("double", "double-double", "arbitrary"))
def test_mixed_state_partial_failure_keeps_original_point_ids(
    dependencies, catalogues, arithmetic
):
    program = _program(dependencies, catalogues, "mixed", arithmetic)
    points = _points()
    # Dependency has no longitudinal rest-state convention. Keep this explicit;
    # a failed helicity removes its entire original point, not a spin sum term.
    rest = np.asarray([["5", "0", "0", "0"]] * 4, dtype=object)
    inputs = np.stack([points[0], rest, points[1]])
    result = program.evaluate_ordered_partial_native(
        inputs, ["676"] * 3, model_values=["0.7"], point_ids=[41, 99, 72]
    )
    assert result.requested_point_ids == (41, 99, 72)
    assert result.batch.point_ids == [41, 72]
    assert result.batch.lane_ids == [41 * 9 + i for i in range(9)] + [
        72 * 9 + i for i in range(9)
    ]
    assert result.failures and {failure.point_id for failure in result.failures} == {99}
    assert all(
        failure.stage == "external state preparation" for failure in result.failures
    )
    independent = program.evaluate_ordered_native(
        points, ["676"] * 2, model_values=["0.7"]
    )
    assert result.born.to_decimal() == independent.born.to_decimal()
    assert result.virtual.to_decimal() == independent.virtual.to_decimal()
    empty = program.evaluate_ordered_partial_native(
        inputs[1:2], ["676"], model_values=["0.7"], point_ids=[99]
    )
    assert empty.batch.point_ids == [] and empty.batch.bound.rows == 0
    assert empty.born.rows == empty.virtual.rows == 0
    assert {failure.point_id for failure in empty.failures} == {99}
