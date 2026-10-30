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
* |Im m^2| / Re m^2 < 1e-150, which only oneloopmaster's SymJIT masters
  mishandle (the reducers take its native and interpreted routes); checked at
  row level against the zero-width value.

None bites on CM01-CM10 (the census of all nine rows: MS-MASTERS report), so
no master signature is refused. CM10 runs in the default physics tier, CM01
(the G_mu UFO, ZZ / WW boxes with complex masses) nightly.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from decimal import Decimal, localcontext
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


def _values(result) -> dict[str, list[Decimal]]:
    """The Born and the virtual's Laurent coefficients per sector."""

    values = {"born": [sum(_real(v) for v in result.born.values())]}
    for sector, laurent in result.components["virtual"].items():
        values[str(sorted(dict(sector).items()))] = [
            _real(x) for x in laurent.coefficients
        ]
    return values


def _difference(mine, theirs, coupling: float) -> float:
    """The largest coefficient difference in the campaign metric: relative,
    against max(|value|, |Born| coupling / 2 pi)."""

    with localcontext(prec=90):
        flux = abs(theirs["born"][0]) * Decimal(coupling) / Decimal(2 * math.pi)
        return float(
            max(
                abs(a - b) / max(abs(b), flux)
                for key, values in theirs.items()
                for a, b in zip(mine[key], values, strict=True)
            )
        )


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
    points no master reaches a defect signature; the complex-top boxes on the
    ``Float`` route agree, AP(40) = AP(60) to 1e-30 (measured 1e-54) and
    double-double = AP(60) to 1e-20 (6e-26); binary64 to 1e-8 (8e-10)."""

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
    agreement = _carriers(executor, points, scales, parameters, float(parameters["aS"]))
    assert agreement["ap40"] <= 1e-30, agreement
    assert agreement["dd"] <= 1e-20, agreement
    assert agreement["double"] <= 1e-8, agreement


GMU_PARAMETERS = {
    "Gf": "0.0000116639",
    "aS": "0.118",
    "MT": "173.3",
    "MZ": "91.188",
    "MW": "80.419",
    "MH": "125",
}


@pytest.mark.loop_nightly
def test_cm01_masters_avoid_the_known_complex_mass_defects(
    loop_gmu_ufo, loop_native, tmp_path
):
    """CM01 (e- e+ > mu- mu+ [QED] on the G_mu UFO under the complex-mass
    scheme, every width; ``cms-ufo-standard-gmu.json``): at its four points no
    master reaches a defect signature; its ZZ / WW boxes with complex masses
    agree on the ``Float`` route, AP(40) = AP(60) and double-double = AP(60);
    widths of 1e-152 of their masses (|Im m^2| / Re m^2 below SymJIT's 1e-150)
    give the zero-width virtual in binary64 and double-double. Generation
    about 5 min, 10 GB."""

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
    tiny = {width: f"{GMU_PARAMETERS[mass]}e-152" for width, mass in masses.items()}
    zero = dict.fromkeys(masses, "0")
    points = reference["zero"]["points"]
    scales = ["1000000"] * len(points)
    found = slice_census(executor, points, scales, {**GMU_PARAMETERS, **tiny})
    assert found.tiny_widths, found.record()
    for precision, tolerance in (
        (Precision.double_double(), 1e-20),
        (Precision.double(), 1e-12),
    ):
        runs = [
            [
                _values(result)
                for result in executor.evaluate(
                    points,
                    scales,
                    parameters={**GMU_PARAMETERS, **widths},
                    precision=precision,
                )
            ]
            for widths in (tiny, zero)
        ]
        difference = max(
            _difference(a, b, 1 / 132.0) for a, b in zip(*runs, strict=True)
        )
        assert difference <= tolerance, (precision.arithmetic, difference)
