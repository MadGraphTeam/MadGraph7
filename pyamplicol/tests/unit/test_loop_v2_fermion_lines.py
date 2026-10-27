# SPDX-License-Identifier: 0BSD
"""One-loop QCD virtuals with several coloured fermion lines (package
fermion-lines; design §3.3 "Fermion sign", "Coloured fermion lines").

The ring sign of design §3.3 has no cap on the number of coloured fermion
lines: the colour flow owns their ordering, identical flavours included. The
references (``tests/fixtures/loop/required``, ``tools/loop_reference``) are
MadLoop in quadruple precision at every point (with Recola and, where an
installed library has the process, OpenLoops), three exact on-shell points
each, colour-summed:

* u u~ > d d~ s s~ (three distinct quark lines);
* u u~ > u u~ d d~ (two identical-flavour lines and a third line);
* u u~ > u u~ u u~ (three identical-flavour lines: two of them can pair the
  legs of one counterterm current in two ways, whose colour-connection
  parities differ; ``generation.fermion_ordering.apply_colour_connection_signs``);
* u u~ > u u~ e+ e- (two identical-flavour quark lines and a lepton line).

The virtual of the production route (counterterms from the recursion, native
DAG, native executor with Python inputs) in double-double equals MadLoop's
Born and every Laurent coefficient within 1e-12, strictly relative per
coefficient, and its poles equal Catani's within 1e-12. LC, NLC and full
colour of several quark lines are consistent: the projected slices are the
sums of their kept N_c powers of the exact colour cells, and all powers sum
to full colour, identical flavours included (their blocks of one colour
word and different U(1)-subtraction grades are separate blocks,
``generation.dag_grades``), in binary64 and double-double. The 2->4 requests
run in the nightly tier; the physics tier keeps a 2->3 control (u u~ > u u~ g,
two identical-flavour lines).
"""

from __future__ import annotations

import argparse
import cmath
import json
from fractions import Fraction
from pathlib import Path

import pytest

REQUIRED = Path(__file__).resolve().parents[1] / "fixtures" / "loop" / "required"
LAURENT = ("finite", "single_pole", "double_pole")
TOLERANCE = 1e-12
# ``name -> points`` of the provider comparison; the 2->4 requests are nightly.
CASES = {
    "uux_uuxg": (0,),
    "uux_ddxssx": (0, 1, 2),
    "uux_uuxddx": (0, 1, 2),
    "uux_uuxuux": (0, 1, 2),
    "uux_uuxepem": (0, 1, 2),
}
PHYSICS = frozenset({"uux_uuxg"})
# The references this package added (``tools/loop_reference``, 2026-10-26).
ADDED = ("uux_ddxssx", "uux_uuxddx", "uux_uuxuux", "uux_uuxepem")


def _reference(name: str) -> dict:
    return json.loads((REQUIRED / f"{name}.json").read_text())


def _request(compiled, reference: dict, **changes):
    """The production request of a required fixture
    (``tools/loop_v2_generate._fixture_request``) with ``changes``."""

    import dataclasses

    from tools import loop_v2_generate

    args = argparse.Namespace(compatibility=None, scale_parameter=None)
    request, ignored = loop_v2_generate._fixture_request(reference, compiled, args)
    assert not ignored, ignored
    return dataclasses.replace(request, **changes)


def _relative(a: float, b: float) -> float:
    """|a - b| / max(|a|, |b|): strictly relative, per coefficient."""

    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0 else abs(a - b) / scale


@pytest.mark.loop_fast
@pytest.mark.parametrize("name", ADDED)
def test_reference_scope_states_the_recorded_provider_arithmetic(name):
    """The scope of a reference states each provider's arithmetic as its run
    recorded it (``references.provider_arithmetic``): MadLoop evaluated every
    point of these references in quadruple precision (return code 3xx)."""

    from tools.loop_reference import references

    reference = _reference(name)
    results = {
        provider: {"rows": [row["providers"][provider] for row in reference["rows"]]}
        for provider in reference["providers"]
    }
    statuses = {row["providers"]["madloop"]["status"] for row in reference["rows"]}
    assert all(status // 100 == 3 for status in statuses), statuses
    arithmetic = references.provider_arithmetic(results)
    assert "MadLoop in quadruple precision at every point" in arithmetic
    assert arithmetic in reference["scope"]


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.loop_physics
            if name in PHYSICS
            else pytest.mark.loop_nightly,
        )
        for name in CASES
    ],
)
def test_virtual_equals_madloop_and_catani(loop_sm_compiled_model, loop_native, name):
    """Double-double virtual of the production route against MadLoop's
    quadruple-precision values (1e-12 per coefficient) and Catani's poles."""

    from pyamplicol.loops.v2 import dag_input
    from pyamplicol.loops.v2.assemble import LoopSlice, Precision
    from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan
    from pyamplicol.models.loading import load_compiled_model

    assert dag_input.NATIVE_DAG
    reference = _reference(name)
    compiled = load_compiled_model(loop_sm_compiled_model)
    slice_ = LoopSlice.generate(compiled, _request(compiled, reference), loop_native)
    assert slice_.request.counterterms == "recursion"
    assert slice_.counterterm_program is not None
    executor = NativeExecutor(
        slice_, forward=False, exported=export_plan(slice_, forward=False)
    )
    for index in CASES[name]:
        point = reference["points"][index]
        scale = str(Fraction(reference["mu_r"][index]) ** 2)
        (result,) = executor.evaluate(
            [point],
            [scale],
            parameters=reference["parameters"],
            precision=Precision.double_double(),
        )
        born, *virtual = (float(x) for x in result.totals[:4])
        madloop = reference["rows"][index]["providers"]["madloop"]
        assert _relative(born, madloop["born"]) <= TOLERANCE, (index, born)
        for key, value in zip(LAURENT, virtual, strict=True):
            error = _relative(value, madloop[key])
            assert error <= TOLERANCE, (index, key, value, madloop[key], error)
        assert result.ir_poles, index
        for sector, (single, double) in result.ir_poles.items():
            laurent = result.virtual(sector)
            size = max(abs(float(single)), abs(float(double)))
            assert abs(float(laurent.single_pole - single)) <= TOLERANCE * size
            assert abs(float(laurent.double_pole - double)) <= TOLERANCE * size


# LC and NLC as formal-N_c projections (design §7.4-§7.6). Identical-flavour
# lines reach one colour word through a gluon and through the
# U(1)-subtraction piece of the Fermi-exchanged pairing (from u u~ > u u~ g
# up); the loop-request compilation makes those separate blocks of one N_c
# grade each (``generation.dag_grades``, design §3.3), so the projections are
# exact for them as for distinct flavours. ``name -> (process, Born QCD, QED,
# leg masses, exact point per carrier)`` of the requests without a provider
# fixture. The first three exact points of g g > u u~ u u~ are outside the
# binary64 scope of the O1 rule (full-colour estimates 9.5e-9, 1.1e-9 and
# 4.1e-8 above its 1e-9 cap; true binary64 errors 6.6e-9, 3.5e-10 and 7.6e-8
# against double-double), so binary64 takes the fourth (estimate 1.9e-11,
# true error 1.0e-12) and double-double the first.
PROJECTION_SLICES = {
    "gg_uuxuux": ("g g > u u~ u u~", 4, 0, (0,) * 6, {"double": 3, "double-double": 0})
}
PROJECTION_PHYSICS = frozenset({"uux_uuxg"})
PROJECTION_NAMES = (
    "uux_ddxssx",
    "uux_uuxg",
    "uux_uuxddx",
    "uux_uuxuux",
    "uux_uuxepem",
    "gg_uuxuux",
)
# Binary64 (Python reference path) and double-double (native executor): the
# parts over the N_c powers sum to full colour and the LC (NLC) slice is the
# sum of its kept powers within 1e-12 and 1e-24 of each sector's largest
# Laurent coefficient. In binary64 a value whose own stability estimate E
# exceeds 1e-12 is held to E (capped at 1e-9; the O1 rule, HANDOFF §0.5),
# which the double-double check then holds to 1e-24: u u~ > u u~ u u~ has E =
# 3.7e-11 at its first point (its parts are up to six times its full-colour
# coefficients and cancel).
PROJECTION_CARRIERS = {"double": 1e-12, "double-double": 1e-24}
O1_CAP = 1e-9


def _projection_request(compiled, name: str, accuracy: str, carrier: str):
    """The request of ``name`` at ``accuracy`` with the colour declaration of
    the formal-N_c continuation (the counterterms' N_c grades), and the
    ``carrier``'s point (a fixture's first), scale and parameters."""

    from pyamplicol.loops.v2.assemble import SliceRequest
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import exact_points, legacy_capture

    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )
    colour = {
        "accuracy": accuracy,
        "colour_declaration": declaration,
        "colour_contraction": "direct",
    }
    if name in PROJECTION_SLICES:
        process, qcd, qed, masses, points = PROJECTION_SLICES[name]
        request = SliceRequest(
            process,
            {"QCD": qcd, "QED": qed},
            {"QCD": qcd + 2, "QED": qed},
            fixed_parameters=legacy_capture.FIXED_PARAMETERS,
            **colour,
        )
        index = points[carrier]
        point = exact_points.generate(list(masses), count=index + 1)[index]
        point = [[exact_points.decimal_string(x) for x in p] for p in point]
        return request, point, "1000000", {"aS": "0.118"}
    reference = _reference(name)
    point, scale = reference["points"][0], str(Fraction(reference["mu_r"][0]) ** 2)
    return (
        _request(compiled, reference, **colour),
        point,
        scale,
        reference["parameters"],
    )


@pytest.mark.parametrize("carrier", list(PROJECTION_CARRIERS))
@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.loop_physics
            if name in PROJECTION_PHYSICS
            else pytest.mark.loop_nightly,
        )
        for name in PROJECTION_NAMES
    ],
)
def test_colour_projections_are_formal_nc_projections(
    loop_sm_compiled_model, loop_native, monkeypatch, name, carrier
):
    """LC + NLC + ... = FC with several quark lines, identical flavours
    included: the LC (NLC) slice equals the N_c^P (N_c^P, N_c^(P-1),
    N_c^(P-2)) part of the exact colour cells of every skeleton (an unpruned
    LC slice evaluated one N_c power at a time), P the sector's leading
    degree, above which every part vanishes; the parts over every power sum
    to the full-colour result (design §7.4-§7.6), Born and every component
    and Laurent coefficient, at the first point."""

    from decimal import Decimal, localcontext

    from pyamplicol.loops.v2 import assemble, colour
    from pyamplicol.loops.v2.assemble import LoopSlice, Precision
    from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan
    from pyamplicol.loops.v2.stability import Rotation, StabilityChecks
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    tolerance = Decimal(str(PROJECTION_CARRIERS[carrier]))
    _request0, point, scale, parameters = _projection_request(
        compiled, name, "lc", carrier
    )

    def build(accuracy, prune=True):
        request = _projection_request(compiled, name, accuracy, carrier)[0]
        return LoopSlice.generate(compiled, request, loop_native, prune=prune)

    def evaluate(slice_, stability=None):
        if carrier == "double":
            return slice_.evaluate(
                point,
                scale,
                parameters=parameters,
                precision=Precision.double(),
                stability=stability,
            )
        executor = NativeExecutor(
            slice_, forward=False, exported=export_plan(slice_, forward=False)
        )
        (result,) = executor.evaluate(
            [point],
            [scale],
            parameters=parameters,
            precision=Precision.double_double(),
        )
        return result

    unpruned, full = build("lc", prune=False), build("full")
    degrees = unpruned.degrees()

    def at(power):
        def kept(exponent, _degree, _accuracy):
            return exponent == power

        with monkeypatch.context() as patch:
            patch.setattr(colour, "kept", kept)
            patch.setattr(assemble, "kept", kept)
            return evaluate(unpruned)

    top = max(
        value
        for kind in degrees.values()
        for value in kind.values()
        if value is not None
    )
    lowest = top - 10
    parts = {power: at(power) for power in range(lowest, top + 1)}
    if carrier == "double":
        exact = evaluate(
            full, StabilityChecks((Rotation.about("z", "3/5", "4/5"),), True)
        )
        estimate = exact.stability.relative
        assert estimate < O1_CAP, estimate
        tolerance = max(tolerance, Decimal(str(estimate)))
    else:
        exact = evaluate(full)

    def number(value):
        return Decimal(str(value))

    def coefficients(result, component, sector):
        found = result.components[component].get(sector)
        return None if found is None else [number(x) for x in found.coefficients]

    with localcontext() as context:
        context.prec = 60
        # Below the lowest power evaluated every part vanishes: one power
        # lower is zero in the Born and in every component.
        below = at(lowest - 1)
        assert not any(number(x) for x in below.born.values())
        assert not any(
            number(c)
            for sectors in below.components.values()
            for laurent in sectors.values()
            for c in laurent.coefficients
        )
        total_born = {
            order: sum(number(part.born.get(order, 0)) for part in parts.values())
            for order in exact.born
        }
        for order, value in exact.born.items():
            assert all(
                number(parts[p].born.get(order, 0)) == 0
                for p in parts
                if p > degrees["born"][order]
            ), order
            total = total_born[order]
            assert abs(total - number(value)) <= tolerance * abs(total), order
        for accuracy, depth in (("lc", 0), ("nlc", 2)):
            mine = evaluate(build(accuracy))
            for order, value in mine.born.items():
                degree = degrees["born"][order]
                expected = sum(
                    number(parts[p].born.get(order, 0))
                    for p in range(degree - depth, degree + 1)
                )
                error = abs(number(value) - expected)
                assert error <= tolerance * abs(expected), (accuracy, order, error)
            for component, sectors in exact.components.items():
                for sector, laurent in sectors.items():
                    values = [number(x) for x in laurent.coefficients]
                    size = max(map(abs, values)) or Decimal(1)
                    totals = [Decimal(0)] * len(values)
                    kept_values = [Decimal(0)] * len(values)
                    degree = degrees["virtual"][sector]
                    for p, part in parts.items():
                        found = coefficients(part, component, sector)
                        if found is None:
                            continue
                        if p > degree:
                            assert not any(found), (component, sector, p)
                        for k, x in enumerate(found):
                            totals[k] += x
                            if degree - depth <= p <= degree:
                                kept_values[k] += x
                    for k, (a, b) in enumerate(zip(totals, values, strict=True)):
                        assert abs(a - b) <= tolerance * size, (component, sector, k)
                    ours = coefficients(mine, component, sector) or [Decimal(0)] * len(
                        values
                    )
                    for k, (a, b) in enumerate(zip(ours, kept_values, strict=True)):
                        assert abs(a - b) <= tolerance * size, (
                            component,
                            sector,
                            accuracy,
                            k,
                        )


# The N_c dependence of the tree currents, independently of the grade
# bookkeeping of ``generation.dag_grades``: every tree producing k
# U(1)-subtraction auxiliaries carries the numeric -1/N_c of their colour
# weight k times, so with that weight doubled (each interaction whose result
# is the auxiliary, before the fold and the evaluation reuse) the rows of a
# block of grade g are 2^(-g) times its rows (an N_c scan of the trees). A
# block that mixes grades scales by no power of 2: the unsplit DAG of every
# process below has such blocks, so the scan sees a missing or wrong grade.
# Binary64 (scaling by 2 is exact, measured bitwise) at the first exact
# point with its last leg rescaled, so that no block current is on shell;
# loop-request blocks and the CT program; the 2->4 requests run in the
# nightly tier.
GRADE_SCAN_NAMES = ("uux_uuxg", "uux_uuxuux", "uux_uuxepem", "gg_uuxuux")
GRADE_SCAN_TOLERANCE = 1e-14


def _doubled_u1_weights(dag):
    """``dag`` with the colour weight of every interaction producing the
    U(1)-subtraction auxiliary doubled."""

    from dataclasses import replace

    from pyamplicol.models.contracts import U1_SUBTRACTION_AUXILIARY_KIND

    auxiliaries = {
        current.id
        for current in dag.currents
        if current.index.auxiliary_kind == U1_SUBTRACTION_AUXILIARY_KIND
    }
    return replace(
        dag,
        interactions=tuple(
            replace(
                interaction,
                color_weight=tuple(2 * x for x in interaction.color_weight),
            )
            if interaction.result_id in auxiliaries
            else interaction
            for interaction in dag.interactions
        ),
    )


def _block_values(tree, rows):
    """Block key -> (N_c grade, the block's row values, flattened)."""

    found = {}
    for block in tree.blocks:
        if block.key.auxiliary_kind is not None:
            continue
        variant = "source" if block.is_source else "propagated"
        values = []
        for row in block.rows:
            values.extend(rows.get((row.current_id, variant), ()))
        found[block.key] = (block.colour_flow.nc_grade, values)
    return found


def _scaling_powers(plain, doubled):
    """Block key -> (grade, the power k with doubled = 2^k plain within
    ``GRADE_SCAN_TOLERANCE`` of the block's largest row, or None) over the
    blocks with nonzero rows."""

    assert plain.keys() == doubled.keys()
    found = {}
    for key, (grade, values) in plain.items():
        other_grade, others = doubled[key]
        assert other_grade == grade and len(others) == len(values), key
        assert all(map(cmath.isfinite, (*values, *others))), key
        size = max((abs(x) for x in values), default=0.0)
        if size == 0:
            assert not any(others), key
            continue
        power = next(
            (
                k
                for k in range(len(key.word) + 1)
                if max(abs(b - 2**k * a) for a, b in zip(values, others, strict=True))
                <= GRADE_SCAN_TOLERANCE * 2**k * size
            ),
            None,
        )
        found[key] = (grade, power)
    return found


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.loop_physics
            if name in PROJECTION_PHYSICS
            else pytest.mark.loop_nightly,
        )
        for name in GRADE_SCAN_NAMES
    ],
)
def test_graded_blocks_scale_as_their_u1_pieces(
    loop_sm_compiled_model, monkeypatch, name
):
    """Every block of the grade split (loop-request blocks and the CT
    program, as leading and next-to-leading colour compile them) has rows
    proportional to (U(1)-subtraction weight)^(-grade), and the same DAGs
    without the split have blocks that mix grades (comment above)."""

    from decimal import Decimal

    from pyamplicol.generation import dag_compiler
    from pyamplicol.loops.v2 import dag_input
    from pyamplicol.loops.v2.assemble import (
        Precision,
        TreeRows,
        _Numbers,
        prepare_request_model,
    )
    from pyamplicol.loops.v2.blocks import (
        build_tree_blocks,
        vetoed_particle_ids,
        with_ward_kernels,
    )
    from pyamplicol.loops.v2.ct_closure import (
        counterterm_program_blocks,
        program_propagated_currents,
    )
    from pyamplicol.loops.v2.families import LoopOrderSelection, add_orders
    from pyamplicol.models.external import CompiledUFOModel
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.processes.model import build_model_process_ir

    compiled = load_compiled_model(loop_sm_compiled_model)
    request, point, _scale, parameters = _projection_request(
        compiled, name, "full", "double-double"
    )
    # Massless legs, the last one rescaled: no momentum conservation, so no
    # block current is on shell (the leg sets of all legs but one would be).
    *point, last = [tuple(Decimal(x) for x in momentum) for momentum in point]
    point = [*point, tuple(Decimal(3) / 2 * x for x in last)]
    numbers = _Numbers(Precision.double())
    process = build_model_process_ir(
        request.process, compiled.ir, color_accuracy="full"
    )
    prepared, continuation, fixed, _scale, _alpha = prepare_request_model(
        compiled, request
    )
    data = json.loads(prepared.structure.native.to_json())
    selection = LoopOrderSelection(
        frozenset(request.corrections),
        frozenset(add_orders(tuple(o.items())) for o in request.exact_loop_orders),
    )
    fold = dag_compiler.apply_colour_connection_signs

    def build(u1_grades, doubled):
        """(loop-request blocks, CT program) of the Python lane, their rows
        at the point."""

        with monkeypatch.context() as patch:
            if doubled:
                patch.setattr(
                    dag_compiler,
                    "apply_colour_connection_signs",
                    lambda dag, model: fold(_doubled_u1_weights(dag), model),
                )
            with dag_input.native_flags(dag=False):
                tree = build_tree_blocks(
                    process,
                    CompiledUFOModel(compiled),
                    born_coupling_orders=dict(request.born_orders),
                    block_coupling_orders=request.loop_order_bounds,
                    ignored_particle_ids=vetoed_particle_ids(request.particle_veto),
                    u1_grades=u1_grades,
                )
                warded, kernels = with_ward_kernels(tree, compiled, fixed)
                program = counterterm_program_blocks(
                    kernels,
                    prepared,
                    request.process,
                    data,
                    selection,
                    warded.born_orders,
                    veto=request.particle_veto,
                    fixed_parameters=fixed,
                    declaration=request.colour_declaration,
                    continuation=continuation,
                    scope=(
                        tuple(sorted(request.born_orders.items())),
                        tuple(sorted(request.loop_order_bounds.items())),
                        request.compatibility,
                    ),
                    u1_grades=u1_grades,
                )
        rows = TreeRows(tree.blocks_dag, compiled).evaluate(point, parameters, numbers)
        dag = program.blocks.blocks_dag
        # As ``ct_closure.CountertermProgram.rows`` evaluates the program.
        program_rows = TreeRows(
            dag,
            program.recursion.model.lowered.compiled,
            jit=False,
            propagated_currents=program_propagated_currents(dag),
        ).evaluate(point, parameters, numbers)
        return _block_values(tree, rows), _block_values(program.blocks, program_rows)

    for u1_grades in (True, False):
        (blocks, ct), (blocks2, ct2) = build(u1_grades, False), build(u1_grades, True)
        for plain, doubled in ((blocks, blocks2), (ct, ct2)):
            powers = _scaling_powers(plain, doubled)
            grades = {grade for grade, _power in powers.values()}
            if u1_grades:
                assert None not in grades and len(grades) > 1, grades
                assert all(power == -grade for grade, power in powers.values()), sorted(
                    (key.word, grade, power)
                    for key, (grade, power) in powers.items()
                    if power != -grade
                )
            else:
                mixed = [power for grade, power in powers.values() if grade is None]
                assert mixed and not any(power is not None for power in mixed)
