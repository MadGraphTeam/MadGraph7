# SPDX-License-Identifier: 0BSD
"""The native v2 executor against the Python reference executor (WP2.1).

``plan_export.export_plan`` writes plan v4 of a generated slice and the
native ``V2Executor`` runs it: seeds, fused factors and split vertex tensors,
schedule B per family on 16 SIMD lanes and family threads with each step in
the kernel form of gate G0's rule, schedule F as its oracle, closures and
the bucket-streamed NinjaRS reduction. Checks:

* family numerators equal the Python reference executor's (schedule B, same
  trie frames): within 1e-13 of each column's largest coefficient in
  binary64 (every fixture point) and to the working precision in arbitrary
  precision, on the slice processes and gg -> ggg, uu~ -> ggg, uu~ -> e-e+gg,
  uu~ -> uu~g and uu~ -> tt~ (open top lines through the loop, read with the
  flipped mass), with every step fused, split where it can, or as the rule
  chooses;
* the native rational carrier reproduces the Python exact run bit for bit
  (identical kernels, seeds and leaf weights) in both kernel forms, and
  schedule F equals B;
* the native evaluation (reduction included) reproduces the slice's complete
  virtual in binary64, finite-only evaluations equal the finite parts with
  absent poles, and the native reduction-direction check gives the slice's
  estimate (native carriers and lane chunks: ``test_loop_v2_stability.py``);
* the run reports its memory per lane and its stage timings (development
  numbers, not benchmark rows).
"""

from __future__ import annotations

import json
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from pyamplicol.config import PoleSelection
from pyamplicol.loops.v2 import reference_executor as rx
from pyamplicol.loops.v2.assemble import (
    COMPONENTS,
    LoopSlice,
    Precision,
    SectorSeeds,
    SliceRequest,
    _BlockRows,
    _fraction,
    _Numbers,
)
from pyamplicol.loops.v2.plan_export import (
    PLAN_SCHEMA,
    NativeExecutor,
    compare_numerators,
    export_plan,
)
from pyamplicol.loops.v2.stability import Rotation, StabilityChecks

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop"
PARAMETERS = {"aS": "0.118", "MT": "173"}
FIXED = {"MB": "0"}
DOUBLE = Precision.double()
ARBITRARY = Precision.arbitrary(60)
DOUBLE_TOLERANCE = 1e-13
# Both executors carry the working digits (60 + 20 guard digits).
ARBITRARY_TOLERANCE = 1e-70
# A family whose 60-digit numerator is below ARBITRARY_NOISE of the point's
# numerator scale vanishes exactly (e.g. Furry's C-odd quark loops): its
# size is the carrier's rounding noise, so it is compared on the point's
# scale only.
ARBITRARY_NOISE = 1e-60
FORMS = ("rule", "fused", "split")


def _points(source: str) -> tuple[list, list[str]]:
    data = json.loads((FIXTURES / source).read_text(), parse_float=str)
    points = data["points"]
    mu = data.get("mu_r", ["1000"] * len(points))
    if isinstance(mu, str):
        mu = [mu] * len(points)
    return points, [format(Decimal(str(m)) ** 2, "f") for m in mu]


def _reference_points(name: str) -> tuple[list, list[str]]:
    data = json.loads(
        (ROOT / "tools" / "loop_reference" / f"{name}.json").read_text(),
        parse_float=str,
    )
    points = data["points"]
    return points, ["1000000"] * len(points)


# process, Born orders, loop orders, points
PROCESSES = {
    "gg_gg": (
        "g g > g g",
        {"QCD": 2, "QED": 0},
        lambda: _reference_points("gluon_scattering_qcd"),
    ),
    "uux_ddxg": (
        "u u~ > d d~ g",
        {"QCD": 3, "QED": 0},
        lambda: _points("required/uux_ddxg.json"),
    ),
    "uux_uux": (
        "u u~ > u u~",
        {"QCD": 2, "QED": 0},
        lambda: _points("legacy/uux_uux.json"),
    ),
    "uux_uuxg": (
        "u u~ > u u~ g",
        {"QCD": 3, "QED": 0},
        lambda: _points("legacy/uux_uuxg.json"),
    ),
    # Open massive quark lines through the loop: the tree reading flips the
    # top mass in their segments (``reference_executor.mass_flips``).
    "uux_ttx": (
        "u u~ > t t~",
        {"QCD": 2, "QED": 0},
        lambda: _points("legacy/tt.json"),
    ),
    # The massless five-point momenta of the u u~ -> ggg reference.
    "gg_ggg": (
        "g g > g g g",
        {"QCD": 3, "QED": 0},
        lambda: _reference_points("quark_three_gluon_qcd"),
    ),
    "uux_ggg": (
        "u u~ > g g g",
        {"QCD": 3, "QED": 0},
        lambda: _reference_points("quark_three_gluon_qcd"),
    ),
    "uux_emepgg": (
        "u u~ > e- e+ g g",
        {"QCD": 2, "QED": 2},
        lambda: _points("dilepton-two-gluon.json"),
    ),
}
# Photon exchange only: its closed quark loops with one photon and two gluons
# are exactly vanishing families (Furry, ``CERTIFIED_ZEROS``); the Z's axial
# coupling would keep them (``test_loop_v2_axial``).
IGNORED = {"uux_emepgg": (23,)}
SLICE = ("gg_gg", "uux_ddxg", "uux_uux", "uux_uuxg", "uux_ttx")
EXTRA = ("gg_ggg", "uux_ggg", "uux_emepgg")


@pytest.fixture(scope="module")
def slices(loop_sm_compiled_model, loop_native):
    from pyamplicol.models.loading import load_compiled_model

    compiled = load_compiled_model(loop_sm_compiled_model)
    built: dict[str, tuple[LoopSlice, NativeExecutor]] = {}

    def build(name: str) -> tuple[LoopSlice, NativeExecutor]:
        if name not in built:
            process, born, _points = PROCESSES[name]
            loop = {k: v + (2 if k == "QCD" else 0) for k, v in born.items()}
            request = SliceRequest(
                process,
                born,
                loop,
                fixed_parameters=FIXED,
                ignored_particle_ids=IGNORED.get(name, ()),
            )
            slice_ = LoopSlice.generate(compiled, request, loop_native)
            built[name] = slice_, NativeExecutor(slice_)
        return built[name]

    return build


def _energy(point) -> float:
    """sqrt(s): the O2 weight of degree-r coefficients is sqrt(s)^r."""

    return float(Fraction(point[0][0]) + Fraction(point[1][0]))


def _point(name: str, index: int = 0):
    points, scales = PROCESSES[name][2]()
    return points[index], scales[index]


def _with_forms(executor: NativeExecutor, forms: str) -> NativeExecutor:
    """The same exported plan with another kernel-form policy."""

    if forms == executor.forms:
        return executor
    return NativeExecutor(executor.slice, exported=executor.exported, forms=forms)


# ----------------------------------------------------------------- plan


@pytest.mark.loop_physics
def test_plan_v4_round_trips_through_the_native_loader(slices):
    """The exported plan loads natively; its structure matches the slice."""

    slice_, executor = slices("gg_gg")
    exported = executor.exported
    program = exported.program
    assert program["schema"] == PLAN_SCHEMA
    assert len(program["families"]) == len(slice_.plan.schedules)
    assert executor.native("double").families == len(slice_.plan.schedules)
    leaves = sum(len(s.trie.leaves) for s in slice_.plan.schedules.values())
    assert sum(len(f["leaves"]) for f in program["families"]) == leaves
    # Every factor is read; every step output feeds one edge or is traced.
    read = {
        e["factor"] for f in program["families"] for s in f["steps"] for e in s["edges"]
    }
    read |= {state["factor"] for state in program["forward"]["states"]}
    assert read == set(range(len(program["factors"])))
    assert export_plan(slice_).json() == exported.json()


# ----------------------------------------------------------------- numerators


def _assert_numerators(native, reference, point, tolerance, vanishing, label):
    """Deviation on the point's numerator scale for every family, and on each
    family's own scale except for the ``vanishing`` families."""

    energy = _energy(point)
    deviation = compare_numerators(native, reference, energy=energy, scale="point")
    assert deviation <= tolerance, (label, "point", deviation)
    alone = {k: v for k, v in reference.items() if k not in vanishing}
    deviation = compare_numerators(
        {key: native[key] for key in alone}, alone, energy=energy
    )
    assert deviation <= tolerance, (label, "family", deviation)


def _vanishing(arbitrary, point) -> set:
    """Families whose 60-digit reference numerator is below ARBITRARY_NOISE
    of the point's numerator scale: they vanish exactly."""

    energy = _energy(point)
    floor = ARBITRARY_NOISE * _point_size(arbitrary, energy)
    return {k for k, v in arbitrary.items() if _size(v, energy) <= floor}


def _size(numerator, energy) -> float:
    from pyamplicol.loops.v2.plan_export import _degree

    return max(
        abs(complex(v)) * energy ** _degree(k)
        for column in numerator.columns
        for k, v in enumerate(column)
    )


def _point_size(numerators, energy) -> float:
    return max(_size(value, energy) for value in numerators.values())


# Processes with exactly vanishing families (Furry), certified at 60 digits.
CERTIFIED_ZEROS = ("uux_emepgg",)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", (*SLICE, *EXTRA))
def test_numerators_equal_the_reference_in_binary64(slices, name):
    """Schedules B (every kernel-form policy) and F natively equal the Python
    reference executor's family numerators (one column per sector, the B
    tries' frames) in binary64."""

    slice_, executor = slices(name)
    point, scale = _point(name)
    vanishing: set = set()
    if name in CERTIFIED_ZEROS:
        arbitrary, _ = slice_.family_numerators(
            point, scale, parameters=PARAMETERS, precision=ARBITRARY
        )
        vanishing = _vanishing(arbitrary.numerators, point)
        assert vanishing, "the certified zeros of the process are gone"
    for schedule in ("B", "F"):
        reference, _ = slice_.family_numerators(
            point, scale, parameters=PARAMETERS, precision=DOUBLE, schedule=schedule
        )
        for forms in FORMS if schedule == "B" else ("rule",):
            (native,) = _with_forms(executor, forms).family_numerators(
                [point],
                [scale],
                parameters=PARAMETERS,
                precision=DOUBLE,
                schedule=schedule,
            )
            _assert_numerators(
                native,
                reference.numerators,
                point,
                DOUBLE_TOLERANCE,
                vanishing,
                (schedule, forms),
            )
    # The other fixture points, in one batch, in the rule's forms.
    points, scales = PROCESSES[name][2]()
    natives = executor.family_numerators(
        points[1:], scales[1:], parameters=PARAMETERS, precision=DOUBLE
    )
    for index, (native, point, scale) in enumerate(
        zip(natives, points[1:], scales[1:], strict=True), start=1
    ):
        reference, _ = slice_.family_numerators(
            point, scale, parameters=PARAMETERS, precision=DOUBLE
        )
        _assert_numerators(
            native, reference.numerators, point, DOUBLE_TOLERANCE, vanishing, index
        )


# gg -> ggg in arbitrary precision: schedule B in the rule's forms only (the
# Python 60-digit reference runs for minutes per schedule; F and the split
# forms are covered by the other processes).
SLOW_ARBITRARY = ("gg_ggg",)


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    "name",
    (
        *SLICE,
        "uux_ggg",
        "uux_emepgg",
        *(pytest.param(n, marks=pytest.mark.loop_nightly) for n in SLOW_ARBITRARY),
    ),
)
def test_numerators_equal_the_reference_in_arbitrary_precision(slices, name):
    slice_, executor = slices(name)
    point, scale = _point(name)
    runs = {"B": ("rule", "split"), "F": ("rule",)}
    if name in SLOW_ARBITRARY:
        runs = {"B": ("rule",)}
    for schedule, forms_of_schedule in runs.items():
        reference, _ = slice_.family_numerators(
            point, scale, parameters=PARAMETERS, precision=ARBITRARY, schedule=schedule
        )
        vanishing = _vanishing(reference.numerators, point)
        assert bool(vanishing) == (name in CERTIFIED_ZEROS), sorted(vanishing)[:3]
        for forms in forms_of_schedule:
            (native,) = _with_forms(executor, forms).family_numerators(
                [point],
                [scale],
                parameters=PARAMETERS,
                precision=ARBITRARY,
                schedule=schedule,
            )
            _assert_numerators(
                native,
                reference.numerators,
                point,
                ARBITRARY_TOLERANCE,
                vanishing,
                (schedule, forms),
            )


@pytest.mark.loop_physics
def test_kernel_forms_follow_the_g0_rule(slices):
    """Every step can run fused; ``split`` splits every step that can, and
    the rule picks a subset of those (gate G0's step-time model)."""

    for name in ("gg_gg", "uux_ddxg"):
        _slice, executor = slices(name)
        forms = {f: _with_forms(executor, f).native("double").forms for f in FORMS}
        steps = sum(forms["fused"].values())
        assert forms["fused"] == {"fused": steps, "split": 0}
        assert sum(forms["rule"].values()) == sum(forms["split"].values()) == steps
        assert forms["split"]["split"] > 0
        assert forms["rule"]["split"] <= forms["split"]["split"]


class _ExactNumbers:
    """Exact rational stand-in of ``assemble._Numbers`` for the seeds."""

    double = False
    zero = rx.Complex(Fraction(0), Fraction(0))

    @staticmethod
    def real(value):
        return Fraction(value)

    @staticmethod
    def pair(real, imag=0):
        return rx.Complex(Fraction(real), Fraction(imag))


def _exact_reference(slice_: LoopSlice, point, scale):
    """The Python exact run on the binary64 tree rows and Born amplitudes
    read as exact rationals, with exact seeds."""

    precision = DOUBLE
    exact_point = tuple(tuple(_fraction(x) for x in p) for p in point)
    with _Numbers(precision).scope():
        evaluation = slice_._point(
            exact_point,
            _fraction(scale),
            dict(PARAMETERS),
            precision,
            PoleSelection.ALL,
        )
    carrier = rx.Carrier.exact(precision.working_digits)
    constants = rx.KernelConstants(evaluation.constants.parameters, carrier)
    born = {
        row: {
            key: {grade: rx.exact(value) for grade, value in grades.items()}
            for key, grades in amplitudes.items()
        }
        for row, amplitudes in evaluation.amplitudes.items()
    }
    accuracy = slice_.request.accuracy
    degrees = slice_.degrees()["virtual"] if accuracy != "full" else {}
    seeds = SectorSeeds(
        slice_.table,
        slice_.class_of,
        born,
        [pairs for _sector, pairs in slice_.sectors],
        _ExactNumbers(),
        accuracy,
        [degrees.get(sector) for sector, _pairs in slice_.sectors],
    )
    kinematics = rx.CatalogueKinematics(
        slice_.catalogue,
        slice_.compiled.ir.vertex_terms,
        slice_.data,
        _BlockRows(slice_, evaluation.rows),
        evaluation.incoming,
        constants,
        reading=slice_.reading,
    )
    executor = rx.ReferenceExecutor(
        kinematics,
        seeds,
        momenta=evaluation.incoming,
        layout=slice_.layout,
        carrier=carrier,
        weight=slice_.weights,
    )
    return executor.run_b(slice_.plan).numerators


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["gg_gg", "uux_uux", "uux_ddxg"])
def test_exact_numerators_equal_the_reference_exactly(slices, name):
    """In exact arithmetic B (in both kernel forms) and F natively equal the
    Python run bit for bit: kernels, vertex tensors, seeds, leaf weights,
    closures and frame maps agree."""

    slice_, executor = slices(name)
    point, scale = _point(name)
    reference = _exact_reference(slice_, point, scale)
    for schedule, forms in (("B", "fused"), ("B", "split"), ("F", "rule")):
        (native,) = _with_forms(executor, forms).family_numerators(
            [point], [scale], parameters=PARAMETERS, schedule=schedule, exact=True
        )
        assert native.keys() == reference.keys()
        for key, mine in native.items():
            theirs = reference[key]
            for x, y in zip(mine.columns, theirs.columns, strict=True):
                width = max(len(x), len(y))
                left = [rx.exact(v) for v in x] + [0] * (width - len(x))
                right = [rx.exact(v) for v in y] + [0] * (width - len(y))
                assert left == right, (schedule, forms, key)


# ----------------------------------------------------------------- evaluation


def _assert_close(native, reference, tolerance):
    assert native.born == reference.born
    for component in COMPONENTS:
        for sector, value in reference.components[component].items():
            mine = native.components[component][sector]
            scale = max(abs(complex(x)) for x in value.coefficients) or 1.0
            for a, b in zip(mine.coefficients, value.coefficients, strict=True):
                assert abs(complex(a) - complex(b)) <= tolerance * scale, component


# The dilepton fixture's decimal momenta are massless to 17 digits only, which
# the slice's massless Catani pole check refuses: compared finite-only.
FINITE_ONLY = ("uux_emepgg",)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", (*SLICE, "uux_ggg", *FINITE_ONLY))
def test_native_virtual_equals_the_slice(slices, name):
    """Dressing and reduction natively give the slice's complete virtual
    (Born and counterterms are shared), all three Laurent coefficients, on
    the slice processes and two 2 -> 3 processes."""

    slice_, executor = slices(name)
    points, scales = PROCESSES[name][2]()
    poles = "none" if name in FINITE_ONLY else "all"
    natives = executor.evaluate(points, scales, parameters=PARAMETERS, poles=poles)
    for native, point, scale in zip(natives, points, scales, strict=True):
        reference = slice_.evaluate(point, scale, parameters=PARAMETERS, poles=poles)
        _assert_close(native, reference, 1e-12)
    report = executor.last_report
    assert report["lanes"] == 16
    assert report["reals_per_lane"]["open objects (one family walk)"] > 0


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["gg_gg", "uux_ddxg", "uux_ggg"])
def test_native_finite_only_equals_the_finite_part(slices, name):
    """poles="none": one artifact, finite parts bitwise equal, poles absent."""

    _slice, executor = slices(name)
    points, scales = PROCESSES[name][2]()
    full = executor.evaluate(points, scales, parameters=PARAMETERS)
    finite = executor.evaluate(points, scales, parameters=PARAMETERS, poles="none")
    for a, b in zip(finite, full, strict=True):
        assert a.ir_poles is None and len(a.totals) == 7
        for component in COMPONENTS:
            for sector, value in a.components[component].items():
                assert value.single_pole is None and value.double_pole is None
                assert value.finite == b.components[component][sector].finite
        for lane, value in a.families.items():
            assert value.coefficients == b.families[lane].coefficients[:1]


@pytest.mark.loop_physics
def test_native_stability_estimates_equal_the_slice(slices):
    """The native re-test and rotation checks give the slice's estimates."""

    slice_, executor = slices("uux_ddxg")
    point, scale = _point("uux_ddxg")
    checks = StabilityChecks((Rotation.about("z", "3/5", "4/5"),), True)
    (native,) = executor.evaluate(
        [point], [scale], parameters=PARAMETERS, stability=checks
    )
    reference = slice_.evaluate(point, scale, parameters=PARAMETERS, stability=checks)
    mine, theirs = native.stability, reference.stability
    assert mine is not None and theirs is not None
    assert mine.relative == pytest.approx(theirs.relative, rel=1e-6)
    assert mine.coefficients == pytest.approx(theirs.coefficients, rel=1e-6)
    (plain,) = executor.evaluate([point], [scale], parameters=PARAMETERS)
    assert plain.stability is None


@pytest.mark.loop_physics
def test_native_bare_equals_the_assembled_bare(slices):
    """The executor's own bare virtual (2 Re(i/16pi^2 avg sum Omega)) equals
    the slice assembly of its family integrals."""

    slice_, executor = slices("gg_gg")
    points, scales = PROCESSES["gg_gg"][2]()
    results = executor.evaluate(points, scales, parameters=PARAMETERS)
    for result, columns in zip(results, executor.last_bare, strict=True):
        for (sector, _pairs), column in zip(slice_.sectors, columns, strict=True):
            value = result.components["bare"][sector]
            scale = max(abs(complex(x)) for x in value.coefficients)
            for a, b in zip(value.coefficients, column, strict=True):
                assert abs(complex(a) - complex(*map(float, b))) <= 1e-13 * scale


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", ["gg_gg", "uux_ddxg"])
def test_double_double_numerators_equal_the_reference(slices, name):
    """The DoubleFloat carrier on 200-bit inputs equals the arbitrary-
    precision reference to its own precision (2^-104 ~ 5e-32)."""

    slice_, executor = slices(name)
    point, scale = _point(name)
    reference, _ = slice_.family_numerators(
        point, scale, parameters=PARAMETERS, precision=ARBITRARY
    )
    (native,) = executor.family_numerators(
        [point],
        [scale],
        parameters=PARAMETERS,
        precision=ARBITRARY,
        double_double=True,
    )
    deviation = compare_numerators(native, reference.numerators, energy=_energy(point))
    assert deviation <= 1e-28, deviation
