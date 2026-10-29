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
``generation.dag_grades``), in binary64 and double-double. The grades
themselves are checked independently by an N_c scan of the tree currents:
every block scales as its U(1)-subtraction pieces. Independently of the
grades altogether, LC and NLC are the kept coefficients of full colour as a
function of N_c (full colour has no grade split; ``loop_nc_scan`` evaluates
it at eleven N_c). The 2->4 requests run in
the nightly tier; the physics tier keeps a 2->3 control (u u~ > u u~ g, two
identical-flavour lines).
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
# u u~ > d d~ s s~, u u~ > u u~ d d~ and u u~ > u u~ u u~ are the M-QCD
# campaign's rows A19, A20 and A21 (``test_loop_v2_mqcd_nightly``: the same
# double-double comparison with MadLoop's quadruple precision within 1e-12
# per sector and coefficient and the poles against Catani within 1e-24, plus
# binary64 under the O1 rule and the estimates' tenfold bracketing; PLAN
# §3.2 MQ-VALIDATE item 3).
CASES = {
    "uux_uuxg": (0,),
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


# The user's one-order-of-magnitude margin of single-point estimate tests
# (2026-10-23; DESIGN §10.4, ``test_loop_v2_stability.ESTIMATE_MARGIN``).
ESTIMATE_MARGIN = 10


@pytest.mark.loop_physics
def test_recommended_estimate_brackets_the_uux_uuxepem_double_pole(
    loop_sm_compiled_model, loop_native
):
    """Package mq-stab, the '22x case': u u~ > u u~ e+ e- at the fixture's
    points 0 and 2, binary64 against our double-double. With a rotation about
    the beam axis (z by (3/5, 4/5): both incoming momenta and every energy
    and p_z stay bitwise the same) and the cached reduction check, the double
    pole's estimate was 1.2e-13 and 7.6e-14 for errors of 5.6e-13 and 1.7e-12
    (4.7x and 22.8x); a rotation moving the beams brackets them (0.5x). The
    masters are not the cause: every rotation keeps their exact invariants.
    The recommended checks (``stability.RECOMMENDED_CHECKS``: a generic
    rotation and the rescaled reduction check) bracket every Laurent
    coefficient within ``ESTIMATE_MARGIN``."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision
    from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan
    from pyamplicol.loops.v2.stability import RECOMMENDED_CHECKS
    from pyamplicol.models.loading import load_compiled_model

    reference = _reference("uux_uuxepem")
    compiled = load_compiled_model(loop_sm_compiled_model)
    slice_ = LoopSlice.generate(compiled, _request(compiled, reference), loop_native)
    executor = NativeExecutor(
        slice_, forward=False, exported=export_plan(slice_, forward=False)
    )
    for index in (0, 2):
        point = [reference["points"][index]]
        scale = [str(Fraction(reference["mu_r"][index]) ** 2)]
        parameters = reference["parameters"]
        (truth,) = executor.evaluate(
            point, scale, parameters=parameters, precision=Precision.double_double()
        )
        (checked,) = executor.evaluate(
            point,
            scale,
            parameters=parameters,
            precision=Precision.double(),
            stability=RECOMMENDED_CHECKS,
        )
        estimate = checked.stability
        for sector, laurent in truth.components["virtual"].items():
            values = checked.components["virtual"][sector].coefficients
            bounds = estimate.absolute[sector].coefficients
            for key, value, exact, bound in zip(
                LAURENT, values, laurent.coefficients, bounds, strict=True
            ):
                error = abs(complex(value) - complex(exact))
                assert 0 < error <= ESTIMATE_MARGIN * bound, (
                    index, sector, key, error, bound
                )


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
# against double-double), so binary64 takes the fourth and double-double the
# first. The fourth is not within 1e-12 either: its full-colour estimate is
# 1.87e-11, and there the binary64 sum over the N_c powers is within 1.33e-12
# of full colour and the NLC within 1.93e-12 of its kept powers (double-double
# at the same point: 9.1e-29 and 1.2e-28, so this is binary64 conditioning),
# which holds only under the O1 rule (``PROJECTION_O1`` below).
PROJECTION_SLICES = {
    "gg_uuxuux": ("g g > u u~ u u~", 4, 0, (0,) * 6, {"double": 3, "double-double": 0}),
    # Another flavour and a crossing with identical quarks on both sides
    # (the grade scan below).
    "ddx_ddxg": ("d d~ > d d~ g", 3, 0, (0,) * 5, {"double": 0, "double-double": 0}),
    "ug_uuux": ("u g > u u u~", 3, 0, (0,) * 5, {"double": 0, "double-double": 0}),
}
# The default (physics) tier runs the native double-double parts check of
# u u~ > u u~ g; its binary64 one (the Python reference path, which the
# legacy and artifact LC/NLC controls of the same process also cover at the
# physics tier) and every other id are nightly.
PROJECTION_PHYSICS = frozenset({("uux_uuxg", "double-double")})
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
# Laurent coefficient. In binary64 the ids of ``PROJECTION_O1``, whose point
# is not within 1e-12, are held to their own stability estimate E (capped at
# 1e-9; the O1 rule, HANDOFF §0.5), which the double-double check then holds
# to 1e-24; every other binary64 id is held strictly to 1e-12 (u u~ > d d~
# s s~ included, as at the base, where E = 1.28e-11). A fixture's first point
# is used unless ``PROJECTION_POINTS`` names another for a carrier: the first
# point of u u~ > u u~ u u~ is outside binary64's 1e-12 (full-colour estimate
# 3.9e-11, true binary64 error 1.95e-12 against double-double; its parts are
# up to six times its full-colour coefficients and cancel), so binary64
# takes its second (estimate 4.0e-12, true error 2.7e-13).
PROJECTION_CARRIERS = {"double": 1e-12, "double-double": 1e-24}
PROJECTION_POINTS = {"uux_uuxuux": {"double": 1}}
PROJECTION_O1 = frozenset({"gg_uuxuux"})
O1_CAP = 1e-9


def _projection_request(compiled, name: str, accuracy: str, carrier: str):
    """The request of ``name`` at ``accuracy`` with the colour declaration of
    the formal-N_c continuation (the counterterms' N_c grades), and the
    ``carrier``'s point (a fixture's first unless ``PROJECTION_POINTS``),
    scale and parameters."""

    from pyamplicol.loops.v2.assemble import SliceRequest
    from pyamplicol.models.loop_color_declaration import (
        resolve_loop_color_declaration,
    )
    from tools.loop_reference import exact_points, legacy_capture

    # The shipped loop_sm declaration (MQ-NC): u u~ > u u~ e+ e- reaches the
    # closed-quark-loop R2 of g g Z, which the example QCD subset
    # (``legacy_capture.COLOUR_DECLARATION``) does not cover.
    declaration = resolve_loop_color_declaration(compiled.raw_loop_model)
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
    index = PROJECTION_POINTS.get(name, {}).get(carrier, 0)
    point = reference["points"][index]
    scale = str(Fraction(reference["mu_r"][index]) ** 2)
    return (
        _request(compiled, reference, **colour),
        point,
        scale,
        reference["parameters"],
    )


class _PowerParts:
    """The N_c-power parts of ``name`` at its ``carrier`` point: ``parts[p]``
    is the unpruned LC slice evaluated keeping N_c power p only (``at``), p
    from ``top`` (the largest leading degree) down to ``lowest`` = top - 10;
    binary64 on the Python reference path, double-double in the native
    executor (``evaluate``)."""

    def __init__(self, compiled, loop_native, monkeypatch, name, carrier):
        self.compiled, self.native, self.name = compiled, loop_native, name
        self.monkeypatch, self.carrier = monkeypatch, carrier
        _request0, self.point, self.scale, self.parameters = _projection_request(
            compiled, name, "lc", carrier
        )
        self.unpruned = self.build("lc", prune=False)
        self.degrees = self.unpruned.degrees()
        self.top = max(
            value
            for kind in self.degrees.values()
            for value in kind.values()
            if value is not None
        )
        self.lowest = self.top - 10
        self.parts = {p: self.at(p) for p in range(self.lowest, self.top + 1)}

    def build(self, accuracy, prune=True):
        from pyamplicol.loops.v2.assemble import LoopSlice

        request, *_rest = _projection_request(
            self.compiled, self.name, accuracy, self.carrier
        )
        return LoopSlice.generate(self.compiled, request, self.native, prune=prune)

    def evaluate(self, slice_, stability=None):
        from pyamplicol.loops.v2.assemble import Precision
        from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan

        if self.carrier == "double":
            return slice_.evaluate(
                self.point,
                self.scale,
                parameters=self.parameters,
                precision=Precision.double(),
                stability=stability,
            )
        executor = NativeExecutor(
            slice_, forward=False, exported=export_plan(slice_, forward=False)
        )
        (result,) = executor.evaluate(
            [self.point],
            [self.scale],
            parameters=self.parameters,
            precision=Precision.double_double(),
        )
        return result

    def at(self, power):
        from pyamplicol.loops.v2 import assemble, colour

        def kept(exponent, _degree, _accuracy):
            return exponent == power

        with self.monkeypatch.context() as patch:
            patch.setattr(colour, "kept", kept)
            patch.setattr(assemble, "kept", kept)
            return self.evaluate(self.unpruned)


@pytest.mark.parametrize(
    ("name", "carrier"),
    [
        pytest.param(
            name,
            carrier,
            marks=pytest.mark.loop_physics
            if (name, carrier) in PROJECTION_PHYSICS
            else pytest.mark.loop_nightly,
        )
        for name in PROJECTION_NAMES
        for carrier in PROJECTION_CARRIERS
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

    from pyamplicol.loops.v2.stability import Rotation, StabilityChecks
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    tolerance = Decimal(str(PROJECTION_CARRIERS[carrier]))
    projection = _PowerParts(compiled, loop_native, monkeypatch, name, carrier)
    build, evaluate, at = projection.build, projection.evaluate, projection.at
    full, degrees = build("full"), projection.degrees
    lowest = projection.lowest
    parts = projection.parts
    if carrier == "double" and name in PROJECTION_O1:
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


# An independent formal-N_c projection: full colour as a function of N_c.
# Full colour has no grade split, and ``loop_nc_scan`` evaluates it with
# N_c = N wherever the engine substitutes N_c = 3 and 1/N for the trees'
# U(1)-subtraction weight, so FC(N) = sum_p c_p N^p comes without the grade
# bookkeeping LC and NLC rest on. The parts of the projection test above are
# c_p 3^p. FC(N) at eleven N (one per power of the parts; 1 and powers of two
# keep 1/N exact in binary64, more of them below 3 as the parts reach N_c^-6
# but only N_c^4: the exact fit loses about three digits) fixes every
# coefficient: in binary64 FC(N) equals sum_p part_p (N/3)^p within 1e-12 of
# the largest term (the O1 rule above; FC(N) itself is no scale: at N = 1 it
# cancels to rounding), and at 32 digits (FC(N) on the Python reference path,
# whose arbitrary precision is the only carrier that projects in Python; the
# parts and the LC and NLC slices in double-double) the coefficients solved
# exactly from FC(N) equal the parts, and their kept powers the LC and NLC
# slices, within 1e-24 of the largest part. One process per N.
NC_SCAN_NAMES = ("uux_uuxg", "uux_uuxuux")
NC_SCAN_VALUES = tuple(
    Fraction(n)
    for n in ("3", "2", "4", "1", "8", "1/2", "16", "1/4", "1/8", "1/16", "1/32")
)
NC_SCAN_PRECISION = {"double": "double", "double-double": "arbitrary:32"}


def _fc_at_nc(model, name, carrier, values, precision, cache):
    """FC of ``name`` at the ``carrier``'s point and each N_c of ``values``
    (``loop_nc_scan``, one process each, concurrently): N -> (rescaled
    interactions, entries)."""

    import os
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    root = Path(__file__).resolve().parents[2]

    def run(index, n):
        environment = dict(os.environ, PYAMPLICOL_CACHE_DIR=str(cache / f"nc-{index}"))
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "tests.unit.loop_nc_scan",
                str(model),
                name,
                carrier,
                str(n),
                precision,
            ],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        assert process.returncode == 0, (n, process.stderr[-4000:])
        found = json.loads(process.stdout.strip().splitlines()[-1])
        return found["rescaled"], _entries(found)

    with ThreadPoolExecutor(len(values)) as pool:
        return dict(zip(values, pool.map(run, range(len(values)), values), strict=True))


def _entries(result) -> dict:
    """(born, order) and (component, sector, k) -> exact value, from a
    ``PointResult`` or its ``loop_nc_scan`` JSON (orders and sectors by repr)."""

    from decimal import Decimal

    def exact(value):
        return Fraction(Decimal(str(value)))

    if isinstance(result, dict):
        born, components = result["born"], result["components"]
    else:
        born = {repr(k): v for k, v in result.born.items()}
        components = {
            c: {repr(s): laurent.coefficients for s, laurent in sectors.items()}
            for c, sectors in result.components.items()
        }
    out = {("born", order, 0): exact(v) for order, v in born.items()}
    for component, sectors in components.items():
        for sector, coefficients in sectors.items():
            for k, c in enumerate(coefficients):
                out[component, sector, k] = exact(c)
    return out


@pytest.mark.loop_nightly
@pytest.mark.parametrize("carrier", list(NC_SCAN_PRECISION))
@pytest.mark.parametrize("name", NC_SCAN_NAMES)
def test_colour_projections_are_coefficients_of_an_nc_scan_of_full_colour(
    loop_sm_compiled_model, loop_native, monkeypatch, tmp_path, name, carrier
):
    """LC and NLC of identical-flavour lines equal the N_c^P (N_c^P..N_c^(P-2))
    coefficients of full colour as a function of N_c (comment above): Born
    and every component and Laurent coefficient at the first point."""

    from pyamplicol.loops.v2.stability import Rotation, StabilityChecks
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    projection = _PowerParts(compiled, loop_native, monkeypatch, name, carrier)
    parts = {p: _entries(part) for p, part in projection.parts.items()}
    powers = sorted(parts)
    assert len(NC_SCAN_VALUES) == len(powers)
    scan = _fc_at_nc(
        loop_sm_compiled_model,
        name,
        carrier,
        NC_SCAN_VALUES,
        NC_SCAN_PRECISION[carrier],
        tmp_path,
    )
    assert all(scan[n][0] for n in NC_SCAN_VALUES if n != 3), scan.keys()
    keys = set().union(*parts.values(), *(entries for _r, entries in scan.values()))

    def sector(key):
        return key[:2]

    if carrier == "double":
        full = projection.build("full")
        estimate = projection.evaluate(
            full, StabilityChecks((Rotation.about("z", "3/5", "4/5"),), True)
        ).stability.relative
        assert estimate < O1_CAP, estimate
        tolerance = Fraction(max(PROJECTION_CARRIERS[carrier], estimate))
        for n, (_rescaled, value) in scan.items():
            terms = {
                key: [parts[p].get(key, 0) * (n / 3) ** p for p in powers]
                for key in keys
            }
            size = {}
            for key, row in terms.items():
                size[sector(key)] = max(size.get(sector(key), 0), *map(abs, row))
            for key, row in terms.items():
                error = abs(value.get(key, 0) - sum(row))
                assert error <= tolerance * (size[sector(key)] or 1), (n, key)
        return
    tolerance = Fraction(PROJECTION_CARRIERS[carrier])
    accuracies = {
        accuracy: _entries(projection.evaluate(projection.build(accuracy)))
        for accuracy in ("lc", "nlc")
    }
    degrees = {("born", repr(k)): v for k, v in projection.degrees["born"].items()}
    degrees.update(
        {
            (component, repr(k)): v
            for component in {key[0] for key in keys if key[0] != "born"}
            for k, v in projection.degrees["virtual"].items()
        }
    )
    matrix = [[(n / 3) ** p for p in powers] for n in NC_SCAN_VALUES]
    size = {}
    for key in keys:
        largest = max(abs(parts[p].get(key, 0)) for p in powers)
        size[sector(key)] = max(size.get(sector(key), 0), largest)
    for key in keys:
        fitted = _solve(matrix, [scan[n][1].get(key, 0) for n in NC_SCAN_VALUES])
        scale = size[sector(key)] or 1
        for p, c in zip(powers, fitted, strict=True):
            assert abs(c - parts[p].get(key, 0)) <= tolerance * scale, (key, p)
        degree = degrees[sector(key)]
        for accuracy, depth in (("lc", 0), ("nlc", 2)):
            kept = sum(
                c
                for p, c in zip(powers, fitted, strict=True)
                if degree - depth <= p <= degree
            )
            error = abs(accuracies[accuracy].get(key, 0) - kept)
            assert error <= tolerance * scale, (accuracy, key)


def _solve(matrix, rhs):
    """The exact solution of the square linear system ``matrix x = rhs``."""

    rows = [[*row, b] for row, b in zip(matrix, rhs, strict=True)]
    size = len(rows)
    for column in range(size):
        pivot = next(r for r in range(column, size) if rows[r][column])
        rows[column], rows[pivot] = rows[pivot], rows[column]
        for r in range(size):
            if r != column and rows[r][column]:
                factor = rows[r][column] / rows[column][column]
                rows[r] = [
                    x - factor * y for x, y in zip(rows[r], rows[column], strict=True)
                ]
    return [rows[r][size] / rows[r][r] for r in range(size)]


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
# loop-request blocks and the CT program; another flavour (d d~ > d d~ g), a
# crossing (u g > u u u~) and the 2->4 requests run in the nightly tier.
GRADE_SCAN_NAMES = (
    "uux_uuxg",
    "ddx_ddxg",
    "ug_uuux",
    "uux_uuxuux",
    "uux_uuxepem",
    "gg_uuxuux",
)
GRADE_SCAN_TOLERANCE = 1e-14
GRADE_SCAN_PHYSICS = frozenset({"uux_uuxg"})


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
            if name in GRADE_SCAN_PHYSICS
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
        resolve_request,
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
    # The model's strong correction (``resolve_request``: SliceRequest
    # names no correction by default).
    request = resolve_request(compiled, request)
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


# The legacy engine's own formal-N_c projections (its colour algebra,
# ``SUNColorSpace(N_c)``, independent of the grade split) of three
# identical-flavour lines, at the first point of the provider fixture: the
# projection-only records of ``legacy_capture.PROJECTION_CONTROLS``
# (``<name>.<accuracy>.json``; the legacy program of one accuracy takes over
# an hour, so only these are captured). u u~ > u u~ g's projections are in
# its full legacy capture (``test_loop_v2_slice.py``, ``test_loop_v2_artifact.py``).
LEGACY = Path(__file__).resolve().parents[1] / "fixtures" / "loop" / "legacy"
LEGACY_PROJECTIONS = (("uux_uuxuux", "lc"),)


@pytest.mark.loop_nightly
@pytest.mark.parametrize("carrier", list(PROJECTION_CARRIERS))
@pytest.mark.parametrize(("name", "accuracy"), LEGACY_PROJECTIONS)
def test_identical_flavour_projections_equal_legacy(
    loop_sm_compiled_model, loop_native, name, accuracy, carrier
):
    """The production slice's LC (NLC) equals the legacy projection: leading
    degrees, Born and every component and Laurent coefficient against its
    200-bit values, binary64 (Python reference path) within 1e-12 and
    double-double (native executor) within 1e-24 of each component's largest
    coefficient (the Born's own size for the Born)."""

    from decimal import Decimal, localcontext

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
    from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan
    from pyamplicol.models.loading import load_compiled_model
    from pyamplicol.models.loop_color_declaration import LoopColorDeclaration
    from tools.loop_reference import legacy_capture

    fixture = json.loads((LEGACY / f"{name}.{accuracy}.json").read_text())
    projection = fixture["colour_projections"][accuracy]
    assert projection["colour_declaration"] == str(
        legacy_capture.COLOUR_DECLARATION.relative_to(legacy_capture.ROOT)
    )
    control = legacy_capture.PROJECTION_CONTROLS[name]
    declaration = LoopColorDeclaration.from_dict(
        json.loads(legacy_capture.COLOUR_DECLARATION.read_text())
    )
    slice_ = LoopSlice.generate(
        load_compiled_model(loop_sm_compiled_model),
        SliceRequest(
            control.process,
            control.born_order_bounds,
            control.loop_order_bounds,
            fixed_parameters=legacy_capture.FIXED_PARAMETERS,
            accuracy=accuracy,
            colour_declaration=declaration,
        ),
        loop_native,
    )
    ((born_degree,), (virtual_degree,)) = (
        tuple(slice_.degrees()[kind].values()) for kind in ("born", "virtual")
    )
    assert (born_degree, virtual_degree) == (
        projection["born_degree"],
        projection["virtual_degree"],
    )
    (point,), (scale,) = fixture["points"], fixture["scales_squared"]
    parameters = {k: v for k, v in fixture["parameters"].items() if k != "MU_R"}
    if carrier == "double":
        result = slice_.evaluate(
            point, scale, parameters=parameters, precision=Precision.double()
        )
    else:
        executor = NativeExecutor(
            slice_, forward=False, exported=export_plan(slice_, forward=False)
        )
        (result,) = executor.evaluate(
            [point],
            [scale],
            parameters=parameters,
            precision=Precision.double_double(),
        )
    reference = projection["results"]["arbitrary-200"]
    tolerance = Decimal(str(PROJECTION_CARRIERS[carrier]))
    with localcontext() as context:
        context.prec = 60
        (((born, _imaginary),),) = reference["born"]
        born = Decimal(born)
        (value,) = result.born.values()
        assert abs(Decimal(str(value)) - born) <= tolerance * abs(born)
        for component in legacy_capture.COMPONENTS:
            ((coefficients,),) = reference[component]
            expected = [Decimal(real) for real, _imaginary in coefficients]
            (laurent,) = result.components[component].values()
            size = max(map(abs, expected)) or abs(born)
            for k, (a, b) in enumerate(
                zip(laurent.coefficients, expected, strict=True)
            ):
                error = abs(Decimal(str(a)) - b)
                assert error <= tolerance * size, (component, k, error / size)
