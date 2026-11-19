# SPDX-License-Identifier: 0BSD
"""The electroweak ladder, joint selection, carriers and conventions (WP3.2 EWI-7).

Nightly (``loop_nightly``; one id per process, each generation a few
minutes on the G_mu UFO, ``loop_qcd_qed_sm_Gmu_py3``, real on-shell masses,
widths zero, 't Hooft-Feynman gauge):

* **the ladder** against its three-provider fixtures (MadLoop, Recola,
  OpenLoops; ``tests/fixtures/loop/required``): g g > t t~ EW and
  u u~ > e- e+ g EW at all three points here (Born and every Laurent
  coefficient, Python point inputs; the native-input acceptance of the rungs
  is WP3.2 EWI-N's), g g > t t~ e- e+ EW at all three points in
  ``test_loop_v2_ew_leptons.test_top_pair_and_leptons_equal_three_providers``
  (double-double: binary64 loses 2e-10 of its finite part at point 0); the
  poles of the two rungs here also against the QCD and QED Catani operators;
* **joint selection** (EW-08, EW-32, EW-33): g g > t t~ generated once with
  QCD and QED corrections and both loop orders, whose alpha_s^3 sector
  equals ``gg_ttx.json`` (loop_sm's providers: the QCD virtual of g g > t t~
  depends on no electroweak input) and whose alpha_s^2 alpha sector equals
  ``gg_ttx_ew.json``, the sectors summing to the totals, and every sector
  and component equal to the separate requests' (QCD corrections only, and
  the electroweak request of ``gg_ttx_ew``);
* **the DD and AP rung** (EW-38): e- e+ > mu- mu+ (full G_mu corrections)
  and g g > t t~ EW at point 0 in binary64, double-double and arbitrary
  precision: each carrier's deviation from the arbitrary-precision value
  within ten times its own stability estimate (the O1 rule: binary64 within
  1e-12 unless its estimate is larger, double-double within 1e-24), all
  three against the providers.

Mixed-order partials (EW-32, review M11) on u u~ > d d~: the bare
photon-only joint QCD + QED selection against MadLoop's
``mixed-uudd-bare-control.json`` per squared order (here); the photon-only
QCD control is ``test_loop_v2_artifact`` (``mixed-uudd-qcd``) and the QCD
corrections of the full Standard-Model Born (gluon, photon, Z and W
exchange; loop_sm with the Wolfenstein CKM, ``mixed-uudd-sm-qcd-control``)
are the M-QCD campaign's row G01 (``tools/loop_benchmark/mqcd_manifest.py``).
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_physics

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/loop"
LAURENT = ("finite", "single_pole", "double_pole")
# The ladder against every provider, in the fixtures' metric (|a - b| over
# max(|a|, |b|, |Born| alpha / 2 pi)). The fixtures' own cross-provider
# tolerance is 2e-8; measured on this engine (binary64, 2026-10-28): at most
# 5.7e-14 (g g > t t~) and 1.0e-12 (u u~ > e- e+ g) against any provider at
# any point, so 1e-11 holds every provider to its own rounding.
LADDER_TOLERANCE = 1e-11
# Catani's poles against the virtual's, relative to the larger pole.
CATANI_TOLERANCE = 1e-12
# The O1 rule (HANDOFF §0.5): binary64 observables to 1e-12 unless their own
# stability estimate is larger; DoubleFloat to 1e-24; every carrier's error
# within ten times its estimate.
O1_BINARY64 = 1e-12
O1_DOUBLE_DOUBLE = 1e-24
ESTIMATE_MARGIN = 10
AP_DIGITS = 40
ZERO_WIDTHS = {"WT": "0", "WW": "0", "WZ": "0", "WH": "0"}
G_MU_FIXED = {"MB": "0", "ymb": "0", "MTA": "0", "ymtau": "0", **ZERO_WIDTHS}


_SIBLINGS: dict[str, object] = {}


def _sibling(name: str):
    """A test module of this directory (pytest imports test modules by path,
    ``--import-mode=importlib``), loaded once under a private name."""

    import importlib.util
    import sys

    if name not in _SIBLINGS:
        path = Path(__file__).with_name(f"{name}.py")
        spec = importlib.util.spec_from_file_location(f"_ew_validate_{name}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SIBLINGS[name] = module
    return _SIBLINGS[name]


def _metric(got: float, want: float, floor: float) -> float:
    scale = max(abs(got), abs(want), floor)
    return abs(got - want) / scale if scale else 0.0


def _gmu(loop_gmu_ufo):
    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(loop_gmu_ufo, require_supported=False)


def _fixture_slice(name, compiled, loop_native):
    """The slice of a required electroweak fixture's request
    (``tools/loop_v2_generate._fixture_request``: QCD and QED corrections,
    the fixture's orders and fixed parameters the model has) and the
    fixture."""

    import argparse

    from pyamplicol.loops.v2.assemble import LoopSlice
    from tools import loop_v2_generate

    reference = json.loads((FIXTURES / "required" / f"{name}.json").read_text())
    request, _ignored = loop_v2_generate._fixture_request(
        reference, compiled, argparse.Namespace()
    )
    return reference, LoopSlice.generate(compiled, request, loop_native)


def _executor(slice_):
    from pyamplicol.loops.v2.plan_export import NativeExecutor, export_plan

    return NativeExecutor(
        slice_, forward=False, exported=export_plan(slice_, forward=False)
    )


def _assert_native_inputs(slice_, reference, index=0) -> float:
    """The native point inputs (WP3.2 EWI-N) equal the Python inputs on the
    same exported plan at row ``index``: every component per sector within
    1e-12 of the sector's virtual (binary64; the lanes are not bitwise
    equal, DESIGN §12.1 item 4). Returns the worst deviation."""

    from pyamplicol.loops.v2.plan_export import NativeExecutor

    native = NativeExecutor(slice_, forward=False, inputs="native")
    python = NativeExecutor(slice_, forward=False, exported=native.exported)
    mine = _evaluate(native, reference, index)
    theirs = _evaluate(python, reference, index)
    worst = 0.0
    for component, sectors in theirs.components.items():
        for sector, laurent in sectors.items():
            virtual = theirs.components["virtual"][sector].coefficients
            size = max(abs(float(x)) for x in virtual)
            for x, y in zip(
                mine.components[component][sector].coefficients,
                laurent.coefficients,
                strict=True,
            ):
                worst = max(worst, abs(float(x) - float(y)) / size)
    assert worst <= 1e-12, worst
    return worst


def _evaluate(executor, reference, index, precision=None, stability=None):
    (result,) = executor.evaluate(
        [reference["points"][index]],
        [str(Fraction(reference["mu_r"][index]) ** 2)],
        parameters=dict(reference["parameters"]),
        precision=precision,
        stability=stability,
    )
    return result


def _assert_providers(
    result, reference, index, tolerance, providers=None
) -> dict[str, float]:
    """Born and the virtual's Laurent coefficients (summed over sectors)
    against every provider of the fixture's row ``index`` (or those named
    in ``providers``); the worst metric per provider."""

    alpha = float(reference["derived_parameters"]["alpha"])
    born, *virtual = (float(x) for x in result.totals[:4])
    worst = {}
    for provider, values in reference["rows"][index]["providers"].items():
        if providers is not None and provider not in providers:
            continue
        floor = abs(values["born"]) * alpha / (2 * math.pi)
        errors = [_metric(born, values["born"], 0.0)]
        errors += [
            _metric(value, values[key], floor)
            for value, key in zip(virtual, LAURENT, strict=True)
        ]
        worst[provider] = max(errors)
        assert worst[provider] <= tolerance, (provider, index, errors)
    return worst


def _assert_catani(result, tolerance=None) -> float:
    """The virtual's single and double pole per sector against the Catani
    operators' prediction (QCD with alpha_s/2pi, QED with the loops' alpha;
    ``poles.sector_sum``): every sector of the request is predicted.
    ``tolerance`` defaults to ``CATANI_TOLERANCE``."""

    assert result.ir_poles, "no Catani prediction"
    worst = 0.0
    for sector, virtual in result.components["virtual"].items():
        single, double = (complex(x) for x in result.ir_poles[sector])
        scale = max(abs(single), abs(double))
        assert scale > 0, sector
        for got, want in (
            (complex(virtual.single_pole), single),
            (complex(virtual.double_pole), double),
        ):
            worst = max(worst, abs(got - want) / scale)
    assert worst <= (CATANI_TOLERANCE if tolerance is None else tolerance), worst
    return worst


# ------------------------------------------------------------------ ladder


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", ["gg_ttx_ew", "uux_emepg_ew"])
def test_electroweak_ladder_equals_three_providers(name, loop_gmu_ufo, loop_native):
    """A ladder rung (PLAN wp32 §3.2 EWI-7) against MadLoop, Recola and
    OpenLoops at all three fixture points: Born and every Laurent
    coefficient of the full NLO EW virtual (QCD and QED corrections, the
    mixed QCD-EW loops of the selected orders included) through the native
    executor in binary64, and the poles of every sector against Catani;
    the native point inputs (WP3.2 EWI-N) equal the Python inputs at point
    0. g g > t t~: about 4 min, 5 GB; u u~ > e- e+ g: about 6 min, 8 GB (a
    colour-singlet lepton line through the boxes, the Z on quark loops)."""

    compiled = _gmu(loop_gmu_ufo)
    reference, slice_ = _fixture_slice(name, compiled, loop_native)
    assert slice_.compiled.ir.gauge == "feynman"
    executor = _executor(slice_)
    worst: dict[str, float] = {}
    catani = 0.0
    for index in range(len(reference["points"])):
        result = _evaluate(executor, reference, index)
        for provider, error in _assert_providers(
            result, reference, index, LADDER_TOLERANCE
        ).items():
            worst[provider] = max(worst.get(provider, 0.0), error)
        catani = max(catani, _assert_catani(result))
    assert set(worst) == {"madloop", "recola", "openloops"}
    native = _assert_native_inputs(slice_, reference)
    print(
        name,
        {k: f"{v:.1e}" for k, v in worst.items()},
        f"catani {catani:.1e}",
        f"native - python inputs {native:.1e}",
    )


# --------------------------------------------------------- joint selection


def _top_pair_request(loop_orders, corrections):
    from pyamplicol.loops.v2.assemble import SliceRequest

    return SliceRequest(
        "g g > t t~",
        {"QCD": 2, "QED": 0},
        loop_orders,
        corrections=corrections,
        fixed_parameters=G_MU_FIXED,
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )


# Separate requests may round a sector's sums in another order than the
# joint one (more families share a reduction lane's rank and the
# counterterm program more terms): measured at most 2 units of the last
# place of the sector's largest coefficient (bare alpha_s^3 finite part at
# point 2; R2 of alpha_s^2 alpha at points 1 and 2), every other sector and
# component bitwise.
JOINT_ULPS = 4


@pytest.mark.loop_nightly
def test_joint_top_pair_selection_equals_its_parts(loop_gmu_ufo, loop_native):
    """g g > t t~ generated once with QCD and QED corrections and both loop
    orders (alpha_s^2 x alpha_s^2 and alpha_s alpha), no squared-order
    restriction: its two sectors are the separate requests' (the QCD
    corrections alone; the electroweak request of ``gg_ttx_ew``) per sector
    and component (to JOINT_ULPS units of the last place), they sum to the
    totals, the alpha_s^3 sector equals ``gg_ttx.json`` (MadLoop, Recola and
    OpenLoops on loop_sm: same momenta and mu_R; the QCD virtual of g g >
    t t~ depends on no electroweak input) and the alpha_s^2 alpha sector
    ``gg_ttx_ew.json``, Born included, at the three points. About 7 min."""

    from pyamplicol.loops.v2.assemble import LoopSlice

    compiled = _gmu(loop_gmu_ufo)
    qcd_fixture = json.loads((FIXTURES / "required/gg_ttx.json").read_text())
    ew_fixture = json.loads((FIXTURES / "required/gg_ttx_ew.json").read_text())
    for key in ("points", "mu_r"):
        assert qcd_fixture[key] == ew_fixture[key], key
    qcd, ew = (("QCD", 6),), (("QCD", 4), ("QED", 2))
    requests = {
        "joint": _top_pair_request(
            ({"QCD": 4, "QED": 0}, {"QCD": 2, "QED": 2}), ("QCD", "QED")
        ),
        qcd: _top_pair_request({"QCD": 4, "QED": 0}, ("QCD",)),
        ew: _top_pair_request({"QCD": 2, "QED": 2}, ("QCD", "QED")),
    }
    executors = {
        name: _executor(LoopSlice.generate(compiled, request, loop_native))
        for name, request in requests.items()
    }
    worst = {"ulps": 0.0, "gg_ttx": 0.0, "gg_ttx_ew": 0.0}
    for index in range(len(ew_fixture["points"])):
        results = {
            name: _evaluate(executor, ew_fixture, index)
            for name, executor in executors.items()
        }
        joint = results["joint"]
        assert set(joint.components["virtual"]) == {qcd, ew}
        for sector in (qcd, ew):
            part = results[sector]
            assert set(part.components["virtual"]) == {sector}
            assert joint.born == part.born
            for name, sectors in joint.components.items():
                mine = [complex(x) for x in sectors[sector].coefficients]
                theirs = [
                    complex(x) for x in part.components[name][sector].coefficients
                ]
                unit = max(abs(x) for x in theirs) * 2.0**-52
                for a, b in zip(mine, theirs, strict=True):
                    ulps = abs(a - b) / unit if unit else abs(a - b)
                    worst["ulps"] = max(worst["ulps"], ulps)
                    assert ulps <= JOINT_ULPS, (index, sector, name, mine, theirs)
        # The sectors sum to the totals (Born, then the virtual's Laurent
        # coefficients).
        virtual = joint.components["virtual"]
        for offset, key in enumerate(LAURENT, start=1):
            total = sum(complex(getattr(virtual[s], key)) for s in (qcd, ew))
            assert abs(total - complex(joint.totals[offset])) <= 1e-15 * max(
                abs(total), 1e-300
            )
        # Each sector against its fixture.
        for sector, fixture, name in (
            (qcd, qcd_fixture, "gg_ttx"),
            (ew, ew_fixture, "gg_ttx_ew"),
        ):
            coupling = float(
                fixture["derived_parameters"]["alpha"]
                if fixture["correction"] == "EW"
                else fixture["parameters"]["aS"]
            )
            born = float(complex(sum(joint.born.values())).real)
            laurent = joint.components["virtual"][sector]
            for provider, values in fixture["rows"][index]["providers"].items():
                floor = abs(values["born"]) * coupling / (2 * math.pi)
                errors = [_metric(born, values["born"], 0.0)]
                errors += [
                    _metric(
                        float(complex(getattr(laurent, key)).real), values[key], floor
                    )
                    for key in LAURENT
                ]
                worst[name] = max(worst[name], *errors)
                assert max(errors) <= LADDER_TOLERANCE, (name, provider, index, errors)
    print({k: f"{v:.1e}" for k, v in worst.items()})


# The photon-only veto of MadLoop's mixed-order controls (Z, W, H and the
# Goldstones; their ghosts follow from ``ew_policy.gauge_closed_veto``).
PHOTON_ONLY = (23, 24, 25, 250, 251)


@pytest.mark.loop_nightly
def test_photon_only_joint_bare_selection_of_uudd_equals_madloop(
    loop_qcd_qed_sm_ufo, loop_native, monkeypatch
):
    """The bare joint selection and the mixed-order sectors (EW-08, EW-32;
    PLAN wp32 §3.2 EWI-7 "mixed-uudd"): u u~ > d d~ generated once with QCD
    and QED corrections and its three loop orders on loop_qcd_qed_sm, the
    Z, W, H and Goldstones vetoed, against MadLoop's photon-only control
    (``mixed-uudd-bare-control.json``: bare N4 + R1 loops, no UV, R2 or
    field terms) at its three points: the Born per squared order and the bare
    component of the four sectors QCD^6 to QED^6 to 1e-12. The veto's
    counterterm guard (``veto-unlabelled-counterterms``: this model's
    electroweak counterterms do not declare their loop content) is lifted
    here since only the bare loops are compared. The W ghost loops of the
    photon self-energy go with the W (``ew_policy.gauge_closed_veto``); with
    them the QED^6 single pole misses by 1.2 %. About 1 min."""

    from pyamplicol.loops.v2 import assemble
    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import compile_model_source

    reference = json.loads((FIXTURES / "mixed-uudd-bare-control.json").read_text())
    points = json.loads((ROOT / "tools/loop_reference/uudd_qcd.json").read_text())
    born_key = {_orders_key(v): k for k, v in reference["born_orders"].items()}
    loop_key = {_orders_key(v): k for k, v in reference["loop_orders"].items()}
    monkeypatch.setattr(
        assemble, "require_labelled_counterterms_for_veto", lambda *_args: None
    )
    compiled = compile_model_source(loop_qcd_qed_sm_ufo, require_supported=False)
    slice_ = LoopSlice.generate(
        compiled,
        SliceRequest(
            "u u~ > d d~",
            {"QCD": 2, "QED": 2},
            ({"QCD": 4, "QED": 0}, {"QCD": 2, "QED": 2}, {"QCD": 0, "QED": 4}),
            corrections=("QCD", "QED"),
            fixed_parameters={**ZERO_WIDTHS, "MTA": "0", "ymtau": "0"},
            particle_veto=PHOTON_ONLY,
            compatibility="gmu_particle_anti",
            scale_parameter="MU_R",
        ),
        loop_native,
    )
    assert set(slice_.request.particle_veto) > set(PHOTON_ONLY)  # the ghosts
    parameters = {
        "aS": reference["parameters"]["alpha_s"],
        "aEWM1": reference["parameters"]["alpha_inverse"],
        "MT": reference["parameters"]["MT"],
    }
    mu_r2 = str(Fraction(reference["parameters"]["mu_R"]) ** 2)
    worst = 0.0
    for point, row in zip(points["points"], reference["rows"], strict=True):
        result = slice_.evaluate(
            [[str(x) for x in leg] for leg in point], mu_r2, parameters=parameters
        )
        largest = max(abs(value) for value in row["born"].values())
        assert {_orders_key(dict(k)) for k in result.born} <= set(born_key)
        for order, key in born_key.items():
            found = complex(result.born.get(tuple(order), 0)).real
            assert abs(found - row["born"][key]) <= 1e-12 * largest, (order, found)
        bare = result.components["bare"]
        assert {_orders_key(dict(k)) for k in bare} == set(loop_key)
        for sector, laurent in bare.items():
            want = row["bare"][loop_key[_orders_key(dict(sector))]]
            wanted = [want[k] for k in LAURENT]
            scale = max(map(abs, wanted))
            for found, target in zip(laurent.coefficients, wanted, strict=True):
                error = abs(complex(found).real - target) / scale
                worst = max(worst, error)
                assert error <= 1e-12, (sector, laurent.coefficients, wanted)
    print(f"bare sectors {worst:.1e}")


def _orders_key(orders) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((name, int(power)) for name, power in orders.items() if power))


def test_vetoed_vector_bosons_take_their_ghosts(loop_qcd_qed_sm_ufo):
    """A particle_veto of a vector boson removes its Faddeev-Popov ghosts
    (``ew_policy.gauge_closed_veto``, applied by ``resolve_request``): the
    ghost of the vectors of its colour, |charge| and mass parameter, read
    from the particle records (ghA of the photon, ghZ of the Z, ghWp and
    ghWm of the W, ghG of the gluon); a veto without vectors is unchanged."""

    from pyamplicol.loops.v2.assemble import SliceRequest, resolve_request
    from pyamplicol.loops.v2.ew_policy import gauge_closed_veto, gauge_ghost_partners
    from pyamplicol.models.loading import compile_model_source

    compiled = compile_model_source(loop_qcd_qed_sm_ufo, require_supported=False)
    codes = {
        str(r["name"]): abs(int(r["pdg_code"])) for r in compiled.model["particles"]
    }
    ghosts = {name: codes[name] for name in ("ghA", "ghZ", "ghWp", "ghWm", "ghG")}
    assert gauge_ghost_partners(compiled) == {
        ghosts["ghA"]: frozenset({codes["A"]}),
        ghosts["ghZ"]: frozenset({codes["Z"]}),
        ghosts["ghWp"]: frozenset({codes["W+"]}),
        ghosts["ghWm"]: frozenset({codes["W+"]}),
        ghosts["ghG"]: frozenset({codes["G"]}),
    }
    assert gauge_closed_veto(compiled, PHOTON_ONLY) == tuple(
        sorted({*PHOTON_ONLY, ghosts["ghZ"], ghosts["ghWp"], ghosts["ghWm"]})
    )
    assert gauge_closed_veto(compiled, (codes["A"],)) == tuple(
        sorted((codes["A"], ghosts["ghA"]))
    )
    assert gauge_closed_veto(compiled, (6, 5)) == (5, 6)
    request = SliceRequest(
        "e- e+ > mu- mu+",
        {"QCD": 0, "QED": 2},
        {"QCD": 0, "QED": 4},
        corrections=("QED",),
        particle_veto=(codes["Z"],),
    )
    assert resolve_request(compiled, request).particle_veto == (
        codes["Z"],
        ghosts["ghZ"],
    )


# ------------------------------------------------------------ DD / AP rung

# e- e+ > mu- mu+ with full electroweak corrections (QED^4 loops over QED^2
# Borns): the MadLoop G_mu control's first row (``test_loop_v2_ew_leptons``).
MUMU_CONTROL = "electroweak-mumu-control"
MUMU_PARAMETERS = {
    "Gf": "0.0000116639",
    "MW": "80.419",
    "MZ": "91.188",
    "MH": "125",
    "MT": "173.3",
    "ymt": "173.3",
    "aS": "0.118",
}
# alpha_Gmu at the control's inputs (the floor of the fixtures' metric).
ALPHA_GMU = 0.0075467722613403515


def _carriers(executor, point, scale, parameters):
    """(binary64, double-double, arbitrary-precision) results at one point,
    each with its stability estimate: one exact rotation and the rescaled
    reduction-direction check (the calibrated configuration, ``stability``)."""

    from pyamplicol.loops.v2.assemble import Precision
    from pyamplicol.loops.v2.stability import Rotation, StabilityChecks

    checks = StabilityChecks(
        rotations=(Rotation.about("z", "3/5", "4/5"),),
        reduction_direction=True,
        rescaled=True,
    )
    out = []
    for precision in (
        None,
        Precision.double_double(),
        Precision.arbitrary(AP_DIGITS),
    ):
        (result,) = executor.evaluate(
            [point],
            [scale],
            parameters=parameters,
            precision=precision,
            stability=checks,
        )
        assert result.stability is not None
        out.append(result)
    return out


def _deviation(result, truth) -> tuple[float, float]:
    """(relative deviation of the Born, largest deviation of the virtual's
    Laurent coefficients over its largest one: the estimate's metric,
    ``StabilityEstimate.relative``) of ``result`` from ``truth``, in
    Decimal arithmetic (single-sector processes)."""

    from decimal import Decimal

    def value(x):
        return Decimal(str(complex(x).real)) if isinstance(x, float) else Decimal(x)

    born, *laurent = (value(x) for x in result.totals[:4])
    true_born, *true_laurent = (Decimal(x) for x in truth.totals[:4])
    scale = max(abs(x) for x in true_laurent)
    return (
        float(abs(born - true_born) / abs(true_born)),
        float(
            max(abs(a - b) for a, b in zip(laurent, true_laurent, strict=True)) / scale
        ),
    )


def _assert_carriers(results) -> dict[str, float]:
    """The O1 rule on binary64 and double-double against the arbitrary
    precision result: each within ten times its own estimate, binary64
    within 1e-12 (or its estimate, capped at 1e-9, when that is larger) and
    double-double within 1e-24."""

    binary64, double_double, arbitrary = results
    assert len(arbitrary.components["virtual"]) == 1  # one sector
    report = {}
    for name, result, bound in (
        ("f64", binary64, O1_BINARY64),
        ("dd", double_double, O1_DOUBLE_DOUBLE),
    ):
        estimate = result.stability.relative
        if name == "f64" and estimate > O1_BINARY64:
            bound = min(estimate, 1e-9)
        born, laurent = _deviation(result, arbitrary)
        assert born <= bound, (name, born)
        assert laurent <= bound, (name, laurent, bound)
        assert laurent <= ESTIMATE_MARGIN * estimate, (name, laurent, estimate)
        report[name] = laurent
        report[f"{name}_estimate"] = estimate
    return report


@pytest.mark.loop_nightly
def test_carriers_of_electroweak_mumu_agree_within_their_estimates(
    loop_gmu_ufo, loop_native
):
    """The DD/AP rung on e- e+ > mu- mu+ with full electroweak corrections
    (G_mu, 't Hooft-Feynman gauge; two colour-singlet lines through the
    boxes): binary64, double-double and arbitrary precision (40 digits) at
    the control's first point, end to end through the native executor; each
    carrier within ten times its stability estimate of the arbitrary
    precision value and the O1 bounds, and every carrier equal to MadLoop
    (``electroweak-mumu-control.json``) to 1e-12 in the fixtures' metric.
    About 5 min (generation 4 min)."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest

    compiled = _gmu(loop_gmu_ufo)
    request = SliceRequest(
        "e- e+ > m- m+",
        {"QED": 2, "QCD": 0},
        {"QED": 4, "QCD": 0},
        corrections=("QED",),
        fixed_parameters=G_MU_FIXED,
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )
    executor = _executor(LoopSlice.generate(compiled, request, loop_native))
    control = json.loads((FIXTURES / f"{MUMU_CONTROL}.json").read_text())
    points = json.loads((ROOT / "tools/loop_reference/uudd_qcd.json").read_text())
    point = [[str(x) for x in leg] for leg in points["points"][0]]
    row = control["rows"][0]
    results = _carriers(executor, point, "1000000", MUMU_PARAMETERS)
    report = _assert_carriers(results)
    madloop_born = row["born"]["1"]
    floor = abs(madloop_born) * ALPHA_GMU / (2 * math.pi)
    for result in results:
        born, *virtual = (float(complex(x).real) for x in result.totals[:4])
        assert _metric(born, madloop_born, 0.0) <= 1e-12
        for value, key in zip(virtual, LAURENT, strict=True):
            assert _metric(value, row["full"]["1"][key], floor) <= 1e-12, key
    print({k: f"{v:.1e}" for k, v in report.items()})


@pytest.mark.loop_nightly
def test_carriers_of_electroweak_top_pair_agree_within_their_estimates(
    loop_gmu_ufo, loop_native
):
    """The DD/AP rung on g g > t t~ EW (``gg_ttx_ew``, QCD and QED
    corrections) at point 0: binary64, double-double and arbitrary precision
    within ten times their estimates of the arbitrary-precision value and the
    O1 bounds, and every carrier against the three providers to the ladder
    tolerance. About 4 min."""

    compiled = _gmu(loop_gmu_ufo)
    reference, slice_ = _fixture_slice("gg_ttx_ew", compiled, loop_native)
    executor = _executor(slice_)
    results = _carriers(
        executor,
        reference["points"][0],
        str(Fraction(reference["mu_r"][0]) ** 2),
        dict(reference["parameters"]),
    )
    report = _assert_carriers(results)
    for result in results:
        _assert_providers(result, reference, 0, LADDER_TOLERANCE)
    print({k: f"{v:.1e}" for k, v in report.items()})


# ------------------------------------------------------- convention metadata


def _scheme_slice(compiled, corrections, photons=None, fields=("e-", "e+"), fixed=None):
    """The fields of a slice ``ew_policy.scheme_conventions`` reads, without
    a generation: a request with ``corrections`` on ``compiled`` (no loop
    species facts: a QED correction alone makes it electroweak) and external
    legs of incoming ``fields`` (``ew_policy.slice_photon_tags``), with
    ``G_MU_FIXED`` and ``fixed`` fixed."""

    from types import SimpleNamespace

    from pyamplicol.loops.v2.assemble import SliceRequest
    from pyamplicol.loops.v2.poles import pole_regulators

    request = SliceRequest(
        "e- e+ > m- m+",
        {"QED": 2, "QCD": 0},
        {"QED": 4, "QCD": 0},
        corrections=corrections,
        fixed_parameters={**G_MU_FIXED, **(fixed or {})},
    )
    return SimpleNamespace(
        compiled=compiled,
        request=request,
        # As ``LoopSlice.pole_regulators`` (no preparation-fixed parameters).
        pole_regulators=pole_regulators(
            compiled.ir.parameters,
            request.fixed_parameters,
            compiled.raw_loop_model,
        ),
        prepared=SimpleNamespace(
            structure=SimpleNamespace(perturbative_species_orders={})
        ),
        families=SimpleNamespace(
            legs=tuple(
                SimpleNamespace(index=index, field=field)
                for index, field in enumerate(fields)
            )
        ),
        external_photons=photons,
    )


def test_conventions_record_the_models_schemes(loop_gmu_ufo, loop_qcd_qed_sm_ufo):
    """``ew_policy.scheme_conventions`` (WP3.2 EWI-7; EW-14, EW-18) on both
    electroweak input-scheme UFOs: HV with the model's lambda_HV (``lhv``,
    found by its TeX name), the gauge, real on-shell masses, the input
    scheme from the SLHA codes the loops' alpha depends on (G_mu: G_F,
    M_W, M_Z; loop_qcd_qed_sm: 1/alpha at SMINPUTS 1) and the external-
    photon convention; a request that is not electroweak records none."""

    from types import SimpleNamespace

    from pyamplicol.loops.v2.ew_policy import scheme_conventions
    from pyamplicol.models.loading import compile_model_source

    gmu = _gmu(loop_gmu_ufo)
    record = scheme_conventions(_scheme_slice(gmu, ("QED",)))
    assert record["regularisation"] == "HV" and record["lambda_HV"] == "1.0"
    assert record["gauge"] == gmu.ir.gauge
    assert record["mass_scheme"] == "real-on-shell"
    assert record["ew_input_scheme"] == "G_mu"
    assert record["alpha_inputs"]["Gf"] == "SMINPUTS 2"
    assert record["alpha_inputs"]["MW"] == "MASS 24"
    assert record["alpha_inputs"]["MZ"] == "MASS 23"
    assert record["external_photons"] == "none"
    assert "generation-complete" in record["gamma5"]
    # MS-POLES: the counterterms' poles split by the model's regulators, by
    # the remainder once one is fixed.
    assert record["pole_split"] == "regulators"
    assert record["pole_regulators"] == {"uv": "epsUV", "ir": "epsIR"}
    fixed = scheme_conventions(_scheme_slice(gmu, ("QED",), fixed={"epsIR": "1"}))
    assert fixed["pole_split"] == "remainder" and "pole_regulators" not in fixed
    # MS-PHOTON: the photons' coupling per leg, with the tagged (alpha(0))
    # and untagged (the loops' coupling) legs.
    legs = ("e-", "e+", "A", "A")
    photons = scheme_conventions(
        _scheme_slice(gmu, ("QED",), SimpleNamespace(legs=(3, 4)), legs)
    )
    assert photons["external_photons"] == "alpha(0)"
    assert photons["tagged_photon_legs"] == [3, 4]
    assert photons["untagged_photon_legs"] == []
    assert photons["external_photons_option"] == "final"
    mixed = scheme_conventions(
        _scheme_slice(gmu, ("QED",), SimpleNamespace(legs=(4,)), legs)
    )
    assert mixed["external_photons"] == "mixed"
    assert (mixed["tagged_photon_legs"], mixed["untagged_photon_legs"]) == ([4], [3])
    untagged = scheme_conventions(_scheme_slice(gmu, ("QED",), None, legs))
    assert untagged["external_photons"] == "loop-coupling"
    assert untagged["untagged_photon_legs"] == [3, 4]
    assert scheme_conventions(_scheme_slice(gmu, ("QCD",))) is None
    alpha = compile_model_source(loop_qcd_qed_sm_ufo, require_supported=False)
    record = scheme_conventions(_scheme_slice(alpha, ("QED",)))
    assert record["ew_input_scheme"] == "alpha"
    assert record["alpha_inputs"] == {"aEWM1": "SMINPUTS 1"}
    assert record["lambda_HV"] == "1.0"


def test_conventions_are_stored_and_inspected_only_where_recorded():
    """The plumbing of the record (WP3.2 EWI-7): the conventions live in the
    process's generation record (the physics metadata keeps the schema the
    native admission reads), ``recorded_conventions`` finds them there for
    ``Runtime.inspect`` and the artifact inspection (None where none is
    recorded, so other artifacts keep their bytes), the inspection lists
    them as (field, value) pairs and its summary shows a "conventions"
    row."""

    from dataclasses import replace

    from pyamplicol.artifacts.inspection import _conventions
    from pyamplicol.generation.loop_artifact import recorded_conventions

    record = {
        "regularisation": "HV",
        "lambda_HV": "1.0",
        "alpha_inputs": {"MZ": "MASS 23", "Gf": "SMINPUTS 2"},
    }
    assert _conventions(None, "m") == ()
    assert _conventions(record, "m") == (
        ("regularisation", "HV"),
        ("lambda_HV", "1.0"),
        ("alpha_inputs", "Gf=SMINPUTS 2, MZ=MASS 23"),
    )
    extensions = {
        "generation": {
            "concrete_processes": [
                {"id": "qcd", "filters": {"loop_engine": "v2"}},
                {"id": "ew", "filters": {"conventions": record}},
            ]
        }
    }
    assert recorded_conventions(extensions, "ew") == record
    assert recorded_conventions(extensions, "qcd") is None
    assert recorded_conventions(extensions, "other") is None
    assert recorded_conventions({}, "ew") is None
    from dataclasses import asdict

    from pyamplicol.artifacts.inspection import ArtifactLoopInspection
    from pyamplicol.reporting.summary import _loop_inspection_rows

    loop = ArtifactLoopInspection(
        engine="v2",
        runtime_capability="loop-v2",
        scheme_status="declared-not-certified",
        r2_strategy="model",
        arithmetic=("double",),
        born_orders=("QED^4",),
        sectors=(("QED^6", (("QED^4", "QED^4"),)),),
        correlations=False,
        resolved=True,
        runtime_bytes=1,
    )
    assert "conventions" not in dict(_loop_inspection_rows(asdict(loop)))
    loop = replace(loop, conventions=_conventions(record, "m"))
    rows = dict(_loop_inspection_rows(asdict(loop)))
    assert rows["conventions"] == (
        "regularisation HV; lambda_HV 1.0; alpha_inputs Gf=SMINPUTS 2, MZ=MASS 23"
    )


@pytest.mark.loop_nightly
@pytest.mark.parametrize("ufo", ["loop_gmu_ufo", "loop_qcd_qed_sm_ufo"])
def test_both_input_schemes_generate_electroweak_mumu(ufo, request, loop_native):
    """EW-14: e- e+ > mu- mu+ with full electroweak corrections generates on
    both input-scheme UFOs (G_mu; alpha(M_Z), loop_qcd_qed_sm), each
    recording its scheme (``ew_policy.scheme_conventions`` on the generated
    slice: 't Hooft-Feynman gauge, its input scheme), and its poles equal
    the QED Catani operator's at the point of the MadLoop controls (the
    loops' alpha of each scheme). G_mu about 4 min, loop_qcd_qed_sm 1 min."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.loops.v2.ew_policy import scheme_conventions
    from pyamplicol.models.loading import compile_model_source

    compiled = compile_model_source(
        request.getfixturevalue(ufo), require_supported=False
    )
    externals = {record.name for record in compiled.ir.parameters}
    slice_ = LoopSlice.generate(
        compiled,
        SliceRequest(
            "e- e+ > m- m+",
            {"QED": 2, "QCD": 0},
            {"QED": 4, "QCD": 0},
            corrections=("QED",),
            fixed_parameters={k: v for k, v in G_MU_FIXED.items() if k in externals},
            compatibility="gmu_particle_anti",
            scale_parameter="MU_R",
        ),
        loop_native,
    )
    record = scheme_conventions(slice_)
    assert record["gauge"] == "feynman"
    assert record["ew_input_scheme"] == ("G_mu" if ufo == "loop_gmu_ufo" else "alpha")
    points = json.loads((ROOT / "tools/loop_reference/uudd_qcd.json").read_text())
    point = [[str(x) for x in leg] for leg in points["points"][0]]
    result = slice_.evaluate(point, "1000000", parameters={"aS": "0.118"})
    _assert_catani(result)
    print(ufo, record)


@pytest.mark.loop_nightly
def test_stored_electroweak_artifact_lists_its_conventions(
    loop_qcd_qed_sm_ufo, loop_native, tmp_path
):
    """The record end to end (WP3.2 EWI-7): ``Generator.generate`` of
    e- e+ > mu- mu+ with full electroweak corrections writes the conventions
    into the process's generation record and its metadata, and
    ``Runtime.inspect`` and the artifact inspection show them (stored with
    native point inputs since WP3.2 EWI-N continuation 1; a strict xfail
    before). About 2 min."""

    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.artifacts import load_manifest
    from pyamplicol.artifacts.inspection import inspect_artifact
    from pyamplicol.config import resolve_config

    _PUBLIC_CARD = _sibling("test_loop_v2_ew_leptons")._PUBLIC_CARD
    output = tmp_path / "eemm"
    card = tmp_path / "eemm.toml"
    card.write_text(
        _PUBLIC_CARD.format(
            model=json.dumps(str(loop_qcd_qed_sm_ufo)),
            output=json.dumps(str(output)),
            veto="",
        )
    )
    processes = ProcessSet((ProcessRequest.parse("e- e+ > m- m+", name="eemm"),))
    Generator(resolve_config(card)).generate(processes, output)
    manifest = load_manifest(output, verify_payloads=True)
    (record,) = manifest.extensions["generation"]["concrete_processes"]
    conventions = record["filters"]["conventions"]
    assert conventions["ew_input_scheme"] == "alpha"
    info = Runtime.load(output, process="eemm").inspect()
    assert info["conventions"] == conventions
    (process,) = inspect_artifact(output).processes
    assert dict(process.loop.conventions)["ew_input_scheme"] == "alpha"


# ------------------------------------------- ghost vertices, listed refusals


def test_ghost_vertices_are_read_on_their_own_legs_with_field_signs(loop_gmu_ufo):
    """The W ghost vertices (WP3.2 EWI-7, routed items (ii), (iii)): every
    ghost rule is read on its own legs (``conventions.
    conjugate_vertex_rules``: the ghost Lagrangian has no Hermitian partner,
    the antiparticles' rule [ghWm ghA~ W+] differentiates the other ghost
    leg), with the product of the field signs of its boson legs. The
    conjugate readings of the boson vertices not closed under conjugation
    tie the charged fields together consistently with the closed vertices
    (W G H, W G G0, W G A and W G Z all give s(W-) s(G+) = -1; no
    contradiction, so the ghost-G0 vertices keep -1); the free sign per
    charge-conjugate family is fixed to +1, and every charge-balanced ghost
    loop is independent of it. Measured consequence (nightly): W G ghost
    loops with the other sign miss Catani by 6e-2 on g b > t t~ b."""

    from pyamplicol.loops.v2 import catalogue
    from pyamplicol.loops.v2.assemble import SliceRequest
    from pyamplicol.loops.v2.conventions import conjugate_vertex_rules
    from pyamplicol.loops.v2.ew_policy import feynman_gauge_model
    from pyamplicol.models.external import CompiledUFOModel

    compiled = feynman_gauge_model(
        _gmu(loop_gmu_ufo),
        SliceRequest(
            "e- e+ > m- m+",
            {"QED": 2, "QCD": 0},
            {"QED": 4, "QCD": 0},
            corrections=("QED",),
            fixed_parameters=G_MU_FIXED,
        ),
    )
    builder = catalogue._CatalogueBuilder(compiled, CompiledUFOModel(compiled))
    closed, signs = builder._signs(), builder._signs(conjugates=True)
    for solver in (closed, signs):
        assert solver.product(["W+", "W-"]) == 1
        assert solver.product(["G+", "G-"]) == 1
        assert solver.product(["G0"]) == -1
    # The closed vertices leave the charged families apart; the conjugate
    # readings tie them, with one free sign per family.
    assert closed.product(["W-", "G+"]) is None
    assert signs.product(["W-", "G+"]) == signs.product(["W+", "G-"]) == -1
    assert signs.product(["W-"]) is None
    rules = conjugate_vertex_rules(
        {
            "particles": [
                {"name": r.name, "antiname": r.antiname, "spin": r.spin}
                for r in compiled.ir.particles
            ],
            "couplings": [],
            "vertex_rules": [
                {
                    "name": term.vertex,
                    "particles": list(term.particles),
                    "couplings": [],
                    "color_structures": ["1"],
                }
                for term in compiled.ir.vertex_terms
            ],
        }
    )
    ghosts = {r.name for r in compiled.ir.particles if r.ghost_number}
    w_ghosts = 0
    for term in compiled.ir.vertex_terms:
        if not ghosts.intersection(term.particles):
            continue
        assert rules[term.vertex] == term.vertex
        bosons = [name for name in term.particles if name not in ghosts]
        if sorted(term.particles) == sorted(
            next(r.antiname for r in compiled.ir.particles if r.name == name)
            for name in term.particles
        ):
            assert builder.reading_sign(term) == closed.product(bosons)
        else:
            assert builder.reading_sign(term) == signs.product(bosons, gauge=True)
        if {"W+", "W-"} & set(bosons):
            w_ghosts += 1
    assert w_ghosts == 8  # the A, Z and W ghosts' vertices with a W leg


@pytest.mark.loop_nightly
def test_w_ghost_loops_keep_the_poles_of_charged_drell_yan_equal_to_catani(
    loop_gmu_ufo, loop_native
):
    """u d~ > e+ ve with full electroweak corrections (refused before WP3.2
    EWI-7 at the W ghost vertex [ghA ghWm~ W-] of the W self-energy's ghost
    loops): its single and double poles equal the QED Catani operator's to
    1e-12 at two points and two scales, so the ghost loops, UV divergent,
    are renormalised with the model's W counterterms. About 5 min."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest

    compiled = _gmu(loop_gmu_ufo)
    slice_ = LoopSlice.generate(
        compiled,
        SliceRequest(
            "u d~ > e+ ve",
            {"QCD": 0, "QED": 2},
            {"QCD": 0, "QED": 4},
            corrections=("QED",),
            fixed_parameters=G_MU_FIXED,
            compatibility="gmu_particle_anti",
            scale_parameter="MU_R",
        ),
        loop_native,
    )
    ghosts = {r.name for r in slice_.compiled.ir.particles if r.ghost_number}
    assert any(ghosts & species for species in _family_species(slice_))
    points = (
        [
            ["500", "0", "0", "500"],
            ["500", "0", "0", "-500"],
            ["500", "240", "320", "300"],
            ["500", "-240", "-320", "-300"],
        ],
        [
            ["300", "0", "0", "300"],
            ["300", "0", "0", "-300"],
            ["300", "0", "180", "240"],
            ["300", "0", "-180", "-240"],
        ],
    )
    worst = 0.0
    for point in points:
        for scale in ("1000000", "10000"):
            result = slice_.evaluate(
                point, scale, parameters={"aS": "0.118", "MT": "173"}
            )
            worst = max(worst, _assert_catani(result))
    print(f"catani {worst:.1e}")


def _family_species(slice_):
    """The loop species (names) of every family routing of ``slice_``."""

    out = []
    for family in slice_.families.families.values():
        for closure in family.closures:
            _cut, positions = slice_.families.routing(closure)
            out.append({position.vertex.loop_out for position in positions})
    return out


# Root-mixing rungs in binary64: their poles cancel to 2e-10 of the Born's
# alpha/2pi scale (the vanishing QED double pole of massive emitters).
ROOT_MIXING_TOLERANCE = 1e-8


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", ["ttx_ttx_ew", "ttx_zh_ew"])
def test_goldstone_side_root_mixing_of_top_pair_initial_states_equals_recola(
    name, loop_gmu_ufo, loop_native, loop_reference
):
    """WP3.2 EWI-7 routed item (v): t t~ > t t~ and t t~ > z h with
    electroweak corrections reach Goldstone-vector mixing lines (G0-Z,
    G0-A) that end at the root on their Goldstone side (the top Yukawa of
    leg 1), closed through the root vertex's promoted entries
    (``ct_sites.promoted_root_entry``: the Goldstone leg takes the line's
    vector current contracted with its momentum, as the vertex copies inside
    the blocks do; refused before). Against Recola (``required/<name>``) at
    its three points: Born and every Laurent coefficient, and the poles
    against Catani (binary64, native executor: at most 2e-10, the vanishing
    QED double pole of t t~ > z h in binary64 cancellations; held to 1e-8,
    the user's precision ruling 2026-10-29 ~10:20). MadLoop's Born, poles,
    bare part (virtual - ct) and UV counterterms (ct - r2, the mixings
    included; ``goldstone-mixing.json``) agree too, while its R2 exceeds
    ours by exactly its finite-part difference from Recola (up to 2.5 % and
    7.7 % of the finite part): MadLoop omits R2_G0tt_b (its exports call
    R2_G0TT, R2_HTT and R2_HTT_B and never R2_G0TT_B; slice-status
    DECISIONS 2026-10-28 ~08:10 (1)). The native point inputs, the promoted
    root terms in the native closure, equal the Python inputs at point 0
    (WP3.2 EWI-N; t t~ > t t~: 8.7e-15 of the virtual). About 6 min each."""

    compiled = _gmu(loop_gmu_ufo)
    reference, slice_ = _fixture_slice(name, compiled, loop_native)
    components = loop_reference("goldstone-mixing")["processes"][name]
    assert components["points"] == reference["points"]
    executor = _executor(slice_)
    worst = {"recola": 0.0, "madloop": 0.0, "catani": 0.0}
    for index in range(len(reference["points"])):
        result = _evaluate(executor, reference, index)
        recola = _assert_providers(
            result, reference, index, ROOT_MIXING_TOLERANCE, providers=("recola",)
        )
        worst["recola"] = max(worst["recola"], recola["recola"])
        worst["catani"] = max(
            worst["catani"], _assert_catani(result, ROOT_MIXING_TOLERANCE)
        )
        (sector,) = result.components["virtual"]

        def laurent(*names, result=result, sector=sector):
            return [
                sum(
                    float(complex(result.components[n][sector].coefficients[i]).real)
                    for n in names
                )
                for i in range(3)
            ]

        row = components["rows"][index]
        providers = reference["rows"][index]["providers"]
        virtual = [row["virtual"][k] for k in LAURENT]
        scale = max(map(abs, virtual))
        born = float(complex(sum(result.born.values())).real)
        assert born == pytest.approx(row["born"], rel=1e-12)
        expected = {
            "bare": [virtual[i] - row["ct"][k] for i, k in enumerate(LAURENT)],
            "uv": [row["ct"][k] - row["r2"][k] for k in LAURENT],
        }
        found = {
            "bare": laurent("bare"),
            "uv": laurent("uv_local", "uv_external_fields"),
        }
        for name, want in expected.items():
            for got, target in zip(found[name], want, strict=True):
                error = abs(got - target) / scale
                worst["madloop"] = max(worst["madloop"], error)
                assert error <= ROOT_MIXING_TOLERANCE, (index, name, found[name], want)
        for i in (1, 2):
            error = abs(laurent("virtual")[i] - virtual[i]) / scale
            assert error <= ROOT_MIXING_TOLERANCE, (index, i, error)
        # MadLoop's R2 exceeds ours by its finite-part difference from Recola.
        omitted = providers["madloop"]["finite"] - providers["recola"]["finite"]
        assert abs(omitted) > 1e-4 * abs(virtual[0])
        r2 = row["r2"]["finite"] - laurent("r2")[0]
        assert abs(r2 - omitted) <= ROOT_MIXING_TOLERANCE * scale, (index, r2, omitted)
        # The mixings: 0.3 % to 5.6 % of MadLoop's UV single pole.
        assert abs(row["mixing"]["single_pole"]) > 1e-3 * abs(row["ct"]["single_pole"])
    # The promoted root terms in the native closure (WP3.2 EWI-N inputs).
    worst["native - python inputs"] = _assert_native_inputs(slice_, reference)
    print({k: f"{v:.1e}" for k, v in worst.items()})


@pytest.mark.loop_nightly
def test_single_top_s_channel_equals_madloop_and_recola(loop_gmu_ufo, loop_native):
    """WP3.2 EWI-7 routed item (iv): u d~ > t b~ with electroweak corrections
    and a massless b (a W line joining the massive top and the massless b,
    whose Born closes at a vertex since m-qcd MQ-5F; refused before) against
    MadLoop and Recola (``required/udx_tbx_ew``, which agree to 4e-13) at
    its three points: Born and every Laurent coefficient through the native
    executor in binary64 within the ladder's tolerance, the poles Catani's;
    the native point inputs (WP3.2 EWI-N) equal the Python inputs at point
    0 (2.5e-15 of the virtual). About 6 min."""

    compiled = _gmu(loop_gmu_ufo)
    reference, slice_ = _fixture_slice("udx_tbx_ew", compiled, loop_native)
    executor = _executor(slice_)
    worst: dict[str, float] = {}
    catani = 0.0
    for index in range(len(reference["points"])):
        result = _evaluate(executor, reference, index)
        for provider, error in _assert_providers(
            result, reference, index, LADDER_TOLERANCE
        ).items():
            worst[provider] = max(worst.get(provider, 0.0), error)
        catani = max(catani, _assert_catani(result))
    assert set(worst) == {"madloop", "recola"}
    native = _assert_native_inputs(slice_, reference)
    print(
        {k: f"{v:.1e}" for k, v in worst.items()},
        f"catani {catani:.1e}",
        f"native - python inputs {native:.1e}",
    )


# ------------------------------------------------------- reflection audit

# A species name after every other: the mirror of a routing read as sorting
# after it, so ``families._Recursion._close`` keeps that routing.
_LAST = "\U0010ffff"


def _mumu_families(loop_qcd_qed_sm_ufo, loop_native, monkeypatch, other):
    """Every family's integral (40 digits) of e- e+ > mu- mu+ with full
    electroweak corrections on loop_qcd_qed_sm, with the Python family
    recursion keeping, of every N <= 2 routing and its mirror (the routing
    read backwards from position 1, ``_Recursion._reflected``), the one that
    sorts first (the default) or, with ``other``, the one that sorts last."""

    from pyamplicol.loops.v2 import families, generator_input
    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
    from pyamplicol.models.loading import compile_model_source

    monkeypatch.setattr(generator_input, "NATIVE_STAGES", frozenset({"schedules"}))
    if other:
        close, reflected = families._Recursion._close, families._Recursion._reflected

        def close_other(self, cut, positions, admitted, rejected):
            if len(positions) <= 2:
                mirror = reflected(self, positions)
                if mirror > (cut, positions):
                    return  # the default's reading: its mirror is kept
                if mirror < (cut, positions):
                    families._Recursion._reflected = lambda s, p: (_LAST, p)
                    try:
                        return close(self, cut, positions, admitted, rejected)
                    finally:
                        families._Recursion._reflected = reflected
            return close(self, cut, positions, admitted, rejected)

        monkeypatch.setattr(families._Recursion, "_close", close_other)
    compiled = compile_model_source(loop_qcd_qed_sm_ufo, require_supported=False)
    slice_ = LoopSlice.generate(
        compiled,
        SliceRequest(
            "e- e+ > m- m+",
            {"QED": 2, "QCD": 0},
            {"QED": 4, "QCD": 0},
            corrections=("QED",),
            fixed_parameters=ZERO_WIDTHS,
            compatibility="gmu_particle_anti",
            scale_parameter="MU_R",
        ),
        loop_native,
    )
    monkeypatch.undo()
    point = [["500", "0", "0", "500"], ["500", "0", "0", "-500"]]
    point += [["500", "240", "320", "300"], ["500", "-240", "-320", "-300"]]
    result = slice_.evaluate(
        point,
        "1000000",
        parameters={"aS": "0.118", "MT": "173"},
        precision=Precision.arbitrary(AP_DIGITS),
    )
    return slice_, result


def _family_deviations(default, other):
    """Per family (key, loop order, sector): the largest deviation of the
    other reading's Laurent coefficients from the default's, relative to the
    default's largest one."""

    out = {}
    for lane, laurent in default.families.items():
        mine = [complex(x) for x in laurent.coefficients]
        theirs = [complex(x) for x in other.families[lane].coefficients]
        scale = max(abs(x) for x in mine) or 1.0
        out[lane] = max(abs(a - b) for a, b in zip(mine, theirs, strict=True)) / scale
    return out


@pytest.mark.loop_nightly
def test_one_and_two_point_loops_do_not_depend_on_their_reading(
    loop_qcd_qed_sm_ufo, loop_native, monkeypatch
):
    """The N <= 2 reflection identification (EW-21, review minor 10;
    ``families`` conventions): a one- or two-point ring and its mirror are
    one loop diagram, so the family recursion keeps one of them (the one
    that sorts first; weight 1/2 when the ring is its own mirror, a real
    field's seagull). On e- e+ > mu- mu+ with full electroweak corrections
    (the gamma-Z, Z-Z and gamma-gamma fermion bubbles of the s-channel line
    with the Z's chiral couplings; W, Goldstone and ghost bubbles; the H,
    G0, W+- and G+- seagulls of the Z and photon lines) every one- and
    two-point family's integral is the same, to 1e-30 of its size at 40
    digits, whichever reading is kept: the identification is exact (a
    bubble's C-odd V.A part, odd under the reflection, integrates to zero,
    epsilon^{mu nu rho sigma} p_rho p_sigma = 0, and the readings agree
    family by family). N >= 3 rings keep both orientations (one reading
    direction, min(B_2) < min(B_N)), so chiral loops of three or more
    points never rely on it. About 3 min."""

    _slice, default = _mumu_families(
        loop_qcd_qed_sm_ufo, loop_native, monkeypatch, other=False
    )
    _slice, other = _mumu_families(
        loop_qcd_qed_sm_ufo, loop_native, monkeypatch, other=True
    )
    assert set(default.families) == set(other.families)
    rings = [lane for lane in default.families if len(lane[0]) <= 2]
    # The s-channel bubbles: the two blocks are the initial and the final
    # lepton pair; the seagulls attach both to one vertex.
    assert any({mask for mask, _mass in lane[0]} == {3, 12} for lane in rings)
    assert any(len(lane[0]) == 1 for lane in rings)
    deviations = _family_deviations(default, other)
    worst = max(deviations[lane] for lane in rings)
    assert worst <= 1e-30, worst
    print(f"{len(rings)} one- and two-point families, worst {worst:.1e}")


# ---------------------------------------------------------- [U2] for EW


def test_doublet_breaking_vetoes_of_electroweak_corrections_are_refused(
    loop_gmu_ufo,
):
    """The [U2] anomaly guard for electroweak corrections (WP3.2 EWI-7,
    "check and complete"): a particle_veto that breaks an anomaly-free
    multiplet (the b of the third quark doublet) removes a species
    perturbative in the electromagnetic correction, so on the G_mu UFO,
    whose electroweak UV counterterms declare no loop content, the request
    is refused in the model preparation, before any tree, with
    ``veto-unlabelled-counterterms``; QCD corrections with the same veto
    pass this preparation (their anomalous Levi-Civita cells are the
    counterterm guard's, ``ct_recursion`` "Anomalous vetoes",
    ``test_loop_v2_ct_cancellation``). About 40 s."""

    from pyamplicol.loops.v2.assemble import SliceRequest, prepare_request_model
    from pyamplicol.loops.v2.refusals import LoopRequestUnsupported

    compiled = _gmu(loop_gmu_ufo)

    def request(corrections, loop_orders):
        return SliceRequest(
            "g g > t t~",
            {"QCD": 2, "QED": 0},
            loop_orders,
            corrections=corrections,
            particle_veto=(5,),
            fixed_parameters=G_MU_FIXED,
            compatibility="gmu_particle_anti",
            scale_parameter="MU_R",
        )

    with pytest.raises(LoopRequestUnsupported) as refused:
        prepare_request_model(compiled, request(("QCD", "QED"), {"QCD": 2, "QED": 2}))
    assert refused.value.key == "veto-unlabelled-counterterms"
    assert "removes the species [5]" in str(refused.value)
    prepare_request_model(compiled, request(("QCD",), {"QCD": 4, "QED": 0}))
