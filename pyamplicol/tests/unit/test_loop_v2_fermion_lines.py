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
colour of three distinct lines are consistent: the projected slices are the
sums of their kept N_c powers of the exact colour cells, and all powers sum
to full colour; identical-flavour lines are refused at LC and NLC (their
blocks mix U(1)-subtraction grades; strict xfail). The
2->4 requests run in the nightly tier; the physics tier keeps a 2->3 control
(u u~ > u u~ g, two identical-flavour lines).
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


# Identical-flavour lines mix U(1)-subtraction grades inside one block (a
# current derived as quark a + gluon (b~, b) and as the U(1)-subtraction
# auxiliary (a, b~) + quark b, the Fermi-exchanged pairing; from u u~ > u u~ g
# up), so their tree currents are not N_c-exact and LC/NLC are refused
# (``SeedTable.seeds``, design §3.3 "Coloured fermion lines"): open.
_MIXED_GRADES = pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="identical-flavour blocks mix U(1)-subtraction grades (not N_c-exact)",
)


@pytest.mark.loop_nightly
@pytest.mark.parametrize(
    "name", ["uux_ddxssx", pytest.param("uux_uuxuux", marks=_MIXED_GRADES)]
)
@pytest.mark.parametrize("accuracy", ["lc", "nlc"])
def test_colour_projections_are_formal_nc_projections(
    loop_sm_compiled_model, loop_native, monkeypatch, name, accuracy
):
    """LC + NLC + ... = FC with three quark lines: the LC (NLC) slice equals
    the N_c^P (N_c^P, N_c^(P-1), N_c^(P-2)) part of the exact colour cells of
    every skeleton (an unpruned LC slice evaluated one N_c power at a time),
    P the sector's leading degree, above which every part vanishes; the
    parts over every power sum to the full-colour result (design §7.4-§7.6),
    Born and every component, binary64 at the fixture's first point."""

    from pyamplicol.loops.v2 import assemble, colour
    from pyamplicol.loops.v2.assemble import LoopSlice
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import legacy_capture

    reference = _reference(name)
    compiled = load_compiled_model(loop_sm_compiled_model)
    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )

    def build(accuracy, prune=True):
        request = _request(
            compiled,
            reference,
            accuracy=accuracy,
            colour_declaration=declaration,
            colour_contraction="direct",
        )
        return LoopSlice.generate(compiled, request, loop_native, prune=prune)

    projected, unpruned, full = build(accuracy), build("lc", prune=False), build("full")
    parameters = reference["parameters"]
    point, scale = reference["points"][0], str(Fraction(reference["mu_r"][0]) ** 2)
    degrees = unpruned.degrees()

    def at(power):
        def kept(exponent, _degree, _accuracy):
            return exponent == power

        with monkeypatch.context() as patch:
            patch.setattr(colour, "kept", kept)
            patch.setattr(assemble, "kept", kept)
            return unpruned.evaluate(point, scale, parameters=parameters)

    top = max(
        value
        for kind in degrees.values()
        for value in kind.values()
        if value is not None
    )
    lowest = top - 10
    parts = {power: at(power) for power in range(lowest, top + 1)}

    def coefficients(result, component, sector):
        found = result.components[component].get(sector)
        return None if found is None else [float(x) for x in found.coefficients]

    # Below the lowest power evaluated every part vanishes: one power lower is
    # zero in the Born and in every component.
    below = at(lowest - 1)
    assert not any(float(x) for x in below.born.values())
    assert not any(
        float(c)
        for sectors in below.components.values()
        for laurent in sectors.values()
        for c in laurent.coefficients
    )
    depth = {"lc": 0, "nlc": 2}[accuracy]
    mine = projected.evaluate(point, scale, parameters=parameters)
    exact = full.evaluate(point, scale, parameters=parameters)
    for order, value in mine.born.items():
        degree = degrees["born"][order]
        assert all(
            float(parts[p].born.get(order, 0)) == 0 for p in parts if p > degree
        ), order
        expected = sum(
            float(parts[p].born.get(order, 0))
            for p in range(degree - depth, degree + 1)
        )
        assert abs(float(value) - expected) <= TOLERANCE * abs(expected), order
        total = sum(float(part.born.get(order, 0)) for part in parts.values())
        assert abs(total - float(exact.born[order])) <= TOLERANCE * abs(total)
    for component, sectors in exact.components.items():
        for sector, laurent in sectors.items():
            values = [float(x) for x in laurent.coefficients]
            size = max(map(abs, values)) or 1.0
            totals = [0.0] * len(values)
            kept_values = [0.0] * len(values)
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
            for a, b in zip(totals, values, strict=True):
                assert abs(a - b) <= TOLERANCE * size, (component, sector)
            ours = coefficients(mine, component, sector) or [0.0] * len(values)
            for a, b in zip(ours, kept_values, strict=True):
                assert abs(a - b) <= TOLERANCE * size, (component, sector, accuracy)
