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
# leg masses)`` of the requests without a provider fixture (an exact point).
PROJECTION_SLICES = {"gg_uuxuux": ("g g > u u~ u u~", 4, 0, (0,) * 6)}
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


def _projection_request(compiled, name: str, accuracy: str):
    """The request of ``name`` at ``accuracy`` with the colour declaration of
    the formal-N_c continuation (the counterterms' N_c grades), and its
    first point, scale and parameters."""

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
        process, qcd, qed, masses = PROJECTION_SLICES[name]
        request = SliceRequest(
            process,
            {"QCD": qcd, "QED": qed},
            {"QCD": qcd + 2, "QED": qed},
            fixed_parameters=legacy_capture.FIXED_PARAMETERS,
            **colour,
        )
        (point,) = exact_points.generate(list(masses), count=1)
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
    _request0, point, scale, parameters = _projection_request(compiled, name, "lc")

    def build(accuracy, prune=True):
        request = _projection_request(compiled, name, accuracy)[0]
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
