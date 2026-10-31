# SPDX-License-Identifier: 0BSD
"""The known complex-mass master defects on the masters of the CM rows
(M-SM MS-MASTERS, m-sm/PLAN.md §3.2; DESIGN risk 13).

``tools/loop_benchmark/master_census.py`` enumerates the scalar masters a
plan's reductions reach (every subset of at most four propagators of every
family, at the executor's exact invariants and the bound plan masses) and
counts the signatures of the defects recorded upstream (oneloopmaster
``5b710437`` STATUS "Open C0/D0 limitations", NinjaRS ``41a4c00`` V9):

* C0(0, 0, 0; m_c, 0, m_r) (every route) and C0 at vanishing invariants with
  a complex mass (interpreted ``Float``);
* unequal-mass bubbles with a small momentum in every component;
* a real negative squared mass;
* complex-mass boxes on the ``Float`` route, checked at row level as AP(40)
  against AP(60) and double-double;
* |Im m^2| / Re m^2 < 1e-150, where oneloopmaster's SymJIT masters fail (on
  no reducer's route); checked at row level against the zero-width value.

None bites on CM01-CM10 at their widths (the census of all nine rows; AP(40)
against double-double on CM01-CM04, CM07, CM08 and CM10, AP(32) against
MadLoop CMS on CM05 and CM09 (MS-CMS); DESIGN risk 13). Below 1e-150 the
double-double C0 turned non-finite on CM01, so double-double refuses such a
width (``complex-mass-master``); binary64 and arbitrary precision evaluate
it. CM10 runs in the default physics tier, CM01 (the G_mu UFO, ZZ / WW boxes
with complex masses) nightly; ``tools/loop_benchmark/master_census_rows.py``
reruns any row's census and carrier comparison.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import pytest

from tools.loop_benchmark.master_census import (
    Point,
    Ring,
    bubble_loss,
    census,
    slice_census,
)
from tools.loop_benchmark.master_census_rows import (
    GMU_PARAMETERS,
    _difference,
    _values,
)

REPOSITORY = Path(__file__).resolve().parents[2]
FIXTURES = REPOSITORY / "tests/fixtures/loop"


def _p(*components) -> list[Fraction]:
    return [Fraction(x) for x in components]


@pytest.mark.loop_fast
def test_the_census_reads_each_defect_signature():
    """Synthetic rings: the collinear three-leg triangle reaches
    C0(0, 0, 0; m_c, 0, m_r); a generic 2 -> 2 triangle does not; a soft
    unequal-mass bubble, a real negative and a nearly real complex mass are
    read; the masters are counted once."""

    complex_mass = (Fraction(29929), Fraction(-519, 2))
    collinear = Point([_p(1, 0, 0, 1), _p(2, 0, 0, 2), _p(-3, 0, 0, -3)], [0, 0, 0])
    triangle = Ring((1, 3, 7), (0, 1, 2))
    found = census([triangle], [complex_mass, 0, 400], [collinear])
    assert found.masters == {"A0": 3, "B": 3, "C0": 1}
    assert found.complex_masters == {"A0": 1, "B": 2, "C0": 1}
    assert (found.c0_zero_invariants, found.c0_complex_zero_real) == (1, 1)
    assert found.bites()["c0-complex-zero-real"]
    found = census([triangle], [complex_mass, 400, 400], [collinear])
    assert (found.c0_zero_invariants, found.c0_complex_zero_real) == (1, 0)

    generic = Point(
        [_p(5, 0, 0, 5), _p(5, 0, 0, -5), _p(-5, -3, 0, -4), _p(-5, 3, 0, 4)],
        [0, 0, 0, 0],
    )
    found = census([Ring((1, 3, 15), (0, 1, 2))], [complex_mass, 0, 400], [generic])
    assert found.c0_zero_invariants == 0
    assert not any(found.bites().values())

    a = Fraction(1, 1000)
    soft = Point([_p(5 * a, 0, 0, 3 * a), _p(-5 * a, 0, 0, -3 * a)], [16 * a * a] * 2)
    bubble = Ring((1, 3), (0, 1))
    found = census([bubble], [complex_mass, (Fraction(6400), Fraction(-168))], [soft])
    assert found.small_bubble_ratio == pytest.approx(5e-3 / math.sqrt(29929 + 259.5))
    assert found.bites()["small-momentum-bubble"]
    assert bubble_loss(found.small_bubble_ratio, double_double=True) < 1e-5
    found = census([bubble], [complex_mass, complex_mass], [soft])
    assert found.small_bubble_ratio is None

    mz2 = Fraction(91188, 1000) ** 2
    found = census([], [-1, (mz2, -mz2 / 10**160), (mz2, -mz2 / 10**140)], [generic])
    assert (found.negative_real_masses, found.tiny_widths) == ([0], [1])


def _real(value) -> Decimal:
    part = value.real if hasattr(value, "real") else value
    return Decimal(repr(part)) if isinstance(part, float) else Decimal(str(part))


def _carriers(executor, points, scales, parameters, coupling: float) -> dict:
    """AP(40), double-double and binary64 against AP(60) at every point."""

    from pyamplicol.loops.v2.assemble import Precision

    runs = {
        name: [
            _values(result)
            for result in executor.evaluate(
                points, scales, parameters=parameters, precision=precision
            )
        ]
        for name, precision in (
            ("ap60", Precision.arbitrary(60)),
            ("ap40", Precision.arbitrary(40)),
            ("dd", Precision.double_double()),
            ("double", Precision.double()),
        )
    }
    return {
        name: max(
            _difference(mine, theirs, coupling)
            for mine, theirs in zip(runs[name], runs["ap60"], strict=True)
        )
        for name in ("ap40", "dd", "double")
    }


@pytest.mark.loop_physics
def test_cm10_masters_avoid_the_known_complex_mass_defects(
    loop_sm_ufo, loop_native, tmp_path
):
    """CM10 (u d~ > w+ b b~ QCD=0 [QCD] on loop_sm under the complex-mass
    scheme, Gamma_t > 0; ``required/msm/udx_wpbbx_cms.json``): at its four
    points no master reaches a defect signature; at two of them the
    complex-top boxes on the ``Float`` route agree, AP(40) = AP(60) to 1e-30
    (measured 1e-54) and double-double = AP(60) to 1e-20 (6e-26); binary64
    to 1e-8 (8e-10)."""

    from pyamplicol.loops.model_cms import complex_mass_model
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.models.loading import compile_model_source
    from tools.loop_reference.model_compatibility import (
        prepare_loop_sm_scalar_declarations,
    )

    reference = json.loads((FIXTURES / "required/msm/udx_wpbbx_cms.json").read_text())
    compiled = compile_model_source(
        loop_sm_ufo, restriction=reference["restriction"], cache_dir=tmp_path
    )
    compiled = replace(
        compiled,
        raw_loop_model=prepare_loop_sm_scalar_declarations(
            compiled.raw_loop_model
        ).derived,
        _serialized_path=None,
        _prepared_bundle=None,
    )
    fixed = reference["fixed_parameters"]
    model, _scheme = complex_mass_model(
        compiled, fixed_parameters=fixed, cache_dir=tmp_path
    )
    fixed = {k: v for k, v in fixed.items() if k in model.parameter_defaults}
    slice_ = LoopSlice.generate(
        model,
        SliceRequest(
            "u d~ > w+ b b~",
            reference["born_orders"],
            reference["loop_orders"],
            fixed_parameters=fixed,
        ),
        loop_native,
    )
    executor = NativeExecutor(slice_, inputs="native")
    parameters = {
        k: v
        for k, v in reference["parameters"].items()
        if k in model.parameter_defaults and k not in fixed
    }
    points = reference["points"]
    scales = [str(Fraction(mu) ** 2) for mu in reference["mu_r"]]
    found = slice_census(executor, points, scales, parameters)
    assert not any(found.bites().values()), found.record()
    assert found.complex_masters["D0"] > 0
    assert found.small_bubble_ratio > 0.1
    agreement = _carriers(
        executor, points[:2], scales[:2], parameters, float(parameters["aS"])
    )
    assert agreement["ap40"] <= 1e-30, agreement
    assert agreement["dd"] <= 1e-20, agreement
    assert agreement["double"] <= 1e-8, agreement


def _madloop_rows(block) -> list[dict[str, list[Decimal]]]:
    """A ``cms-ufo-standard-gmu.json`` block's MadLoop Born and full virtual
    (finite, 1/eps, 1/eps^2) per point, keyed as ``_totals`` keys a result."""

    return [
        {"born": [Decimal(born)], "virtual": [Decimal(x) for x in full]}
        for born, full in zip(block["born"], block["full"], strict=True)
    ]


def _totals(result) -> dict[str, list[Decimal]]:
    """A result's Born and complete virtual (``PointResult.totals``)."""

    born, *virtual = (_real(x) for x in result.totals[:4])
    return {"born": [born], "virtual": virtual}


@pytest.mark.loop_nightly
def test_cm01_masters_avoid_the_known_complex_mass_defects(
    loop_gmu_ufo, loop_native, tmp_path
):
    """CM01 (e- e+ > mu- mu+ [QED] on the G_mu UFO under the complex-mass
    scheme, widths free at run time; ``cms-ufo-standard-gmu.json``): at its
    four points no master reaches a defect signature, and its ZZ / WW boxes
    with complex masses agree on the ``Float`` route, AP(40) = AP(60) to
    1e-30 (measured 1e-57), double-double = AP(60) to 1e-20 (2e-30).

    Widths changed at run time (``TreeProgram::bind``): at zero widths the
    binary64 SymJIT parameter program took the other side of the
    electroweak UV counterterm couplings' branch cuts (finite part 3.9e-5
    against MadLoop's 1.8e-3), a -0 imaginary part from SymJIT's negation;
    the translation now makes every cut argument's zero imaginary part +0
    (``symjit_plane::canonical_cut_row``). Every carrier equals MadLoop's
    zero-width virtual (to 1e-8), as at widths of 1e-100 and 1e-152 of the
    masses in binary64 and arbitrary precision; double-double, whose C0
    turned non-finite below 1e-150, refuses 1e-152 with
    ``complex-mass-master``.

    Some widths zero (``required/msm/cm01_zero_widths.json``): at WZ = WT = 0
    with WW != 0 the G_mu UFO's MZ^4 - 4 MT^2 MZ^2 reaches its square root
    through cw^2 = MW^2 / MZ^2, so its imaginary part, zero in exact
    arithmetic, came out as a rounding residue of either sign in every
    carrier (arbitrary precision flipped with its digits, MadLoop's double
    and quadruple precision differ at three of 54 fuzz rows); such an
    argument now reads its real projection when its widths are zero
    (``ct_recursion._RealArguments.zero_width_cases``). Binary64,
    double-double and AP(32) equal MadLoop CMS there (to 1e-8), and equal
    each other where MadLoop's two precisions differ, taking the side of one
    of them. Generation about 5 min, 10 GB."""

    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from tests.unit.test_loop_v2_cms import _gmu_cms_slice

    reference = json.loads((FIXTURES / "cms-ufo-standard-gmu.json").read_text())
    _model, _scheme, slice_ = _gmu_cms_slice(
        loop_gmu_ufo,
        loop_native,
        tmp_path,
        "e- e+ > m- m+",
        {"QED": 2},
        ("QED",),
        fixed={"MB": "0", "MTA": "0"},
    )
    executor = NativeExecutor(slice_, inputs="native")
    block = reference["finite"]
    parameters = {**GMU_PARAMETERS, **block["widths"]}
    points = block["points"]
    scales = ["1000000"] * len(points)
    found = slice_census(executor, points, scales, parameters)
    assert not any(found.bites().values()), found.record()
    assert found.complex_masters["D0"] > 0
    agreement = _carriers(executor, points, scales, parameters, 1 / 132.0)
    assert agreement["ap40"] <= 1e-30, agreement
    assert agreement["dd"] <= 1e-20, agreement
    assert agreement["double"] <= 1e-8, agreement

    masses = {"WZ": "MZ", "WW": "MW", "WT": "MT", "WH": "MH"}
    zero_block = reference["zero"]
    madloop = _madloop_rows(zero_block)
    points = zero_block["points"]
    scales = ["1000000"] * len(points)

    def evaluate(widths, precision):
        return [
            _totals(result)
            for result in executor.evaluate(
                points,
                scales,
                parameters={**GMU_PARAMETERS, **widths},
                precision=precision,
            )
        ]

    zero = dict.fromkeys(masses, "0")
    small = {width: f"{GMU_PARAMETERS[mass]}e-100" for width, mass in masses.items()}
    tiny = {width: f"{GMU_PARAMETERS[mass]}e-152" for width, mass in masses.items()}
    for widths, precision in (
        (zero, Precision.double()),
        (zero, Precision.double_double()),
        (zero, Precision.arbitrary(32)),
        (small, Precision.double_double()),
        (small, Precision.double()),
        (tiny, Precision.double()),
        (tiny, Precision.arbitrary(32)),
    ):
        difference = max(
            _difference(mine, theirs, 1 / 132.0)
            for mine, theirs in zip(evaluate(widths, precision), madloop, strict=True)
        )
        assert difference <= 1e-8, (precision.arithmetic, widths, difference)
    with pytest.raises(ValueError, match="complex-mass-master: the complex pole mass"):
        evaluate(tiny, Precision.double_double())
    found = slice_census(executor, points, scales, {**GMU_PARAMETERS, **tiny})
    assert found.bites()["tiny-width"], found.record()

    # Some widths zero: every carrier equals MadLoop CMS (whose double and
    # quadruple precision agree there), and where MadLoop's two precisions
    # take different sides of a cut the carriers agree with each other and
    # with one of them.
    partial = json.loads((FIXTURES / "required/msm/cm01_zero_widths.json").read_text())
    physical = {**GMU_PARAMETERS, **block["widths"]}
    carriers = (Precision.double(), Precision.double_double(), Precision.arbitrary(32))

    def virtual(row, precision):
        (result,) = executor.evaluate(
            [partial["point"]],
            [str(Fraction(row["mu_r"]) ** 2)],
            parameters={**physical, **row["parameters"]},
            precision=precision,
        )
        return _totals(result)

    for row in partial["rows"]:
        madloop = {
            "born": [Decimal(row["born"])],
            "virtual": [Decimal(x) for x in row["full"]],
        }
        for precision in carriers:
            difference = _difference(virtual(row, precision), madloop, 1 / 132.0)
            assert difference <= 1e-8, (row["row"], precision.arithmetic, difference)
    for tag, row in partial["madloop_double_and_quadruple_differ"].items():
        finite = [virtual(row, precision)["virtual"][0] for precision in carriers]
        tolerance = Decimal("1e-8") * abs(finite[2])
        assert max(abs(x - finite[2]) for x in finite) <= tolerance, tag
        assert any(
            abs(finite[2] - Decimal(row[key])) <= tolerance
            for key in ("finite_double", "finite_quadruple")
        ), (tag, finite)


@pytest.mark.loop_nightly
def test_cm05_native_inputs_equal_madloop_cms(loop_gmu_ufo, loop_native, tmp_path):
    """CM05 (u u~ > e- e+ g [QCD QED] on the G_mu UFO under the complex-mass
    scheme, ``required/msm/uux_emepg_cms.json``) with native point inputs:
    every squared-order sector and their sum equal MadLoop CMS at the four
    points in binary64 to 1e-8, and the pole split holds (``_check_cms_split``).
    Not a master defect: one output of its binary64 parameter program was NaN
    in SymJIT's optimised AVX/AVX-512 code (2.27.3 through v228 ``c0ccfa4``;
    a block of equal lanes, as a batch
    runs it), right in its scalar code, at optimisation level 0 and in the
    exact evaluator; the plane check now runs every probe row in a block of
    its own and, on a disagreement, compiles the program again at level 0
    (``loop_v2::trees::compile_plane_application_checked``; upstream report
    ``upstream/symjit-simd-complex-times-alias``). Before, the finite part was NaN at
    every point (``uv_local``). Generation about 3 min, 6 GB."""

    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from tests.unit.test_loop_v2_cms import (
        _check_cms_split,
        _check_sectors,
        _gmu_cms_slice,
    )

    reference = json.loads((FIXTURES / "required/msm/uux_emepg_cms.json").read_text())
    model, _scheme, slice_ = _gmu_cms_slice(
        loop_gmu_ufo,
        loop_native,
        tmp_path,
        "u u~ > e- e+ g",
        reference["born_orders"],
        ("QCD", "QED"),
        restriction=reference["restriction"],
        fixed=reference["fixed_parameters"],
    )
    native = NativeExecutor(slice_, forward=False, inputs="native")
    _check_sectors(
        slice_,
        reference,
        model,
        (("double", 1e-8),),
        range(len(reference["points"])),
        executor=native,
    )
    _check_cms_split(slice_, reference, model, lanes=("native",))
