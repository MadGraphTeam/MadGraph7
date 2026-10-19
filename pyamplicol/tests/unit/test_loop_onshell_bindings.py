# SPDX-License-Identifier: 0BSD
"""Physical mass-shell bindings are exact model identities, not tolerances."""

import os
from decimal import Decimal, localcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.loops.graphs import build_loop_process
from pyamplicol.loops.kinematics import build_loop_kinematics
from pyamplicol.loops.physical_process import NativeBareProcessProgram, _saved
from pyamplicol.loops.recurrence import (
    build_open_loop_recurrence,
    momentum_component_library,
)

from .test_loop_kinematics import native

__all__ = ["native"]


@pytest.fixture(scope="module")
def massive_process(native):
    _, h, _ = native
    return build_loop_process(
        h.Model.qcd_qed().process(["u", "u~"], ["t", "t~"]),
        selected_corrections=["QCD"],
        perturbative_orders={"QCD": 1},
    )


@pytest.fixture(scope="module")
def massive_recurrences(massive_process):
    graph = next(g for g in massive_process.loops if set(g.loop_species) == {6, 21})
    return tuple(
        build_open_loop_recurrence(graph, routing=routing)
        for routing in ("graph", "family")
    )


def test_physical_bindings_include_dependent_leg_and_reversed_routing(
    native, massive_recurrences
):
    s, _, _ = native
    for recurrence in massive_recurrences:
        raw = build_loop_kinematics(recurrence)
        physical = build_loop_kinematics(recurrence, physical=True)
        assert raw.kinematic_context == "raw" and not raw.on_shell_bindings
        assert physical.kinematic_context == "physical"
        # Physical assumptions do not alter the raw exact polynomial oracle.
        assert physical.exact_input_expressions == raw.exact_input_expressions
        library = momentum_component_library(
            recurrence.graph,
            (recurrence.loop_components,),
            recurrence.momentum_parameters,
        )
        external_squares = {}
        dependent = set(recurrence.graph.native.loop_momentum_basis.dependent_externals)
        dependent_squares = []
        for edge in recurrence.graph.native.external_edges:
            vector = edge.momentum_expression(dimension=4).to_tensor(library)
            square = vector.dot(vector).to_tensor()[0].expand()
            external_squares[square] = edge.particle.mass**2
            if edge.id in dependent:
                dependent_squares.append(square)
        assert dependent_squares
        assignments = dict(physical.on_shell_bindings)
        assert any(physical.s_matrix[i] in dependent_squares for i in assignments)
        n = physical.propagators
        for entry, column in assignments.items():
            assert (
                physical.mass_expressions[column]
                == external_squares[physical.s_matrix[entry]]
            )
            assert assignments[(entry % n) * n + entry // n] == column
        # The massive external top shell reuses an existing propagator output.
        assert any(
            physical.mass_expressions[c] != s.E("0") for c in assignments.values()
        )
        assert all(c < n for c in assignments.values())


@pytest.mark.parametrize(
    "arithmetic,precision",
    [
        ("double", None),
        ("double-double", None),
        ("arbitrary", 80),
    ],
)
def test_exact_mass_updates_share_carrier_and_keep_raw_offshell_oracle(
    native, massive_recurrences, arithmetic, precision
):
    backend = pytest.importorskip("_loop_reduction")
    plan = build_loop_kinematics(massive_recurrences[1], physical=True)
    options = dict(arithmetic=arithmetic, precision=precision)
    parameters = tuple(
        sorted(
            {
                symbol
                for mass in plan.mass_expressions
                for symbol in mass.get_all_symbols(False)
            },
            key=lambda x: x.to_canonical_string(),
        )
    )
    assert len(parameters) == 1
    family = SimpleNamespace(
        kinematics=plan,
        exact_evaluator=backend.ExactPolynomialEvaluator(plan.evaluator_bytes()),
        masses=backend.CoefficientEvaluator(
            _saved(plan.mass_expressions, parameters), **options
        ),
        mass_columns=(0,),
    )
    owner = SimpleNamespace(backend=backend, options=options)
    # Deliberately off-shell components establish that raw evaluation remains raw;
    # this is an invariant-binding unit control, not a valid scattering event.
    points = [
        ["500", "0", "0", "500"],
        ["500", "0", "0", "-500"],
        ["500", "300.00000000001", "0", "400"],
        ["500", "-300", "0", "-400"],
    ]
    pairs = [(value, "0") for point in points for value in point]
    exact = family.exact_evaluator.evaluate_decimal(pairs, 1, row_ids=[91], **options)
    raw_matrix = exact.take_columns(
        list(range(4 * plan.propagators, 4 * plan.propagators + plan.propagators**2))
    )
    values = []
    for mass in ("173", "190.00000000000000000000000001"):
        model = backend.NativeBatch.from_decimal([(mass, "0")], 1, 1, **options)
        currents = SimpleNamespace(
            model=model,
            momenta_pairs=pairs,
            batch=SimpleNamespace(size=1, point_ids=[91]),
        )
        _, masses, matrix = NativeBareProcessProgram.family_kinematics_native(
            owner,
            currents,
            family,
        )
        m = masses.to_decimal()
        actual, raw = matrix.to_decimal(), raw_matrix.to_decimal()
        assert matrix.row_ids == [91]
        for entry, column in plan.on_shell_bindings:
            assert actual[entry] == m[column]
            assert Decimal(actual[entry][0]) != Decimal(raw[entry][0])
        unbound = set(range(plan.propagators**2)) - dict(plan.on_shell_bindings).keys()
        assert all(actual[i] == raw[i] for i in unbound)
        values.append(actual)
    assert values[0] != values[1]
    if arithmetic == "arbitrary":
        with localcontext() as context:
            context.prec = 110
            column = plan.on_shell_bindings[0][0]
            expected = Decimal("190.00000000000000000000000001") ** 2
            assert abs(Decimal(values[1][column][0]) - expected) < Decimal("1e-70")


def test_equal_current_values_do_not_merge_distinct_mass_parameters(
    native, massive_process
):
    s, _, _ = native
    backend = pytest.importorskip("_loop_reduction")
    graph = next(g for g in massive_process.loops if set(g.loop_species) == {2, 6, 21})
    recurrence = build_open_loop_recurrence(graph, routing="family")
    plan = build_loop_kinematics(
        recurrence,
        physical=True,
        model_expressions={
            "UFO::MU": s.S("mass_binding_test::up"),
            "UFO::MT": s.S("mass_binding_test::top"),
        },
    )
    entries = {}
    for entry, column in plan.on_shell_bindings:
        entries.setdefault(plan.mass_expressions[column], (entry, column))
    assert s.S("UFO::MU") ** 2 in entries and s.S("UFO::MT") ** 2 in entries
    up, top = entries[s.S("UFO::MU") ** 2], entries[s.S("UFO::MT") ** 2]
    assert up[1] != top[1]
    parameters = (s.S("UFO::MU"), s.S("UFO::MT"))
    evaluator = backend.CoefficientEvaluator(
        _saved(plan.mass_expressions, parameters), arithmetic="arbitrary", precision=80
    )
    equal = backend.NativeBatch.from_decimal(
        [("173", "0"), ("173", "0")], 1, 2, arithmetic="arbitrary", precision=80
    )
    changed = backend.NativeBatch.from_decimal(
        [("2", "0"), ("173", "0")], 1, 2, arithmetic="arbitrary", precision=80
    )
    before, after = (
        evaluator.evaluate_batch(value).to_decimal() for value in (equal, changed)
    )
    assert before[up[1]] == before[top[1]]
    assert after[up[1]] != after[top[1]]


@pytest.mark.parametrize(
    "kind,message",
    [
        ("raw", "raw kinematics"),
        ("asymmetric", "not symmetric"),
        ("duplicate", "duplicate"),
        ("column", "matrix assignment"),
        ("outputs", "shape/reference"),
    ],
)
def test_native_loader_rejects_malformed_kinematic_bindings(tmp_path, kind, message):
    backend = pytest.importorskip("_loop_reduction")
    source = os.environ.get("PYAMPLICOL_RESOLVED_SOURCE_PACBIN")
    if not source:
        pytest.skip("set PYAMPLICOL_RESOLVED_SOURCE_PACBIN to a retained control")
    from .test_loop_portable import _rewrite

    def mutate(data):
        family = next(f for f in data["families"] if f["propagators"] > 1)
        n = family["propagators"]
        descriptor = {
            "context": "physical",
            "mass_output_count": n,
            "on_shell": [[1, 0], [n, 0]],
        }
        family["kinematics"] = descriptor
        if kind == "raw":
            descriptor["context"] = "raw"
        elif kind == "asymmetric":
            descriptor["on_shell"].pop()
        elif kind == "duplicate":
            descriptor["on_shell"].append([1, 0])
        elif kind == "column":
            descriptor["on_shell"][0][1] = n
        else:
            descriptor["mass_output_count"] = n + 1

    destination = tmp_path / "bad-kinematics.pacbin"
    _rewrite(Path(source), destination, mutate)
    with pytest.raises(ValueError, match=message):
        backend.PortableLoopPlan(str(destination))
