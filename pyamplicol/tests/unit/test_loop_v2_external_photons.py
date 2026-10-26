# SPDX-License-Identifier: 0BSD
"""On-shell external photons of electroweak requests (WP3.2 EWI-6b; the
user's decision [U9] of 2026-10-25): every on-shell external photon couples
with alpha(0), the loops with alpha_Gmu.

MG5_aMC applies the convention outside the UFO (1804.10017 §2;
``export_fks.write_rescale_a0gmu_file``, the G_mu model branch): the Born and
the virtual are multiplied by r = (alpha(0) / alpha_Gmu)^n for n photons, and
the virtual of the electromagnetic correction gains -2 n DGmua0 Born, with
DGmua0 = delta Z_e^Gmu - delta Z_e^alpha(0) the UFO's eCoup_DGmua0_UV_EW. The
engine takes r into every output's normalisation
(``_PointEvaluation.average``) and the conversion as an external-field
factor of each photon leg (``assemble._alpha0_field_rules``).

References (``tests/fixtures/loop/required/uux_ag_ew.json``,
``uux_aa_ew.json``; [U6](c), MadLoop standalone on loop_qcd_qed_sm_Gmu,
real masses, widths zero, the EW sector of u u~ > a g and u u~ > a a): MadLoop
evaluates the pure G_mu virtual, converted here with MG5_aMC's formula and
MadLoop's own DGmua0 (``madloop-ct-parameters.json``, MU_R = 1000 GeV, the
references' scale). With alpha(0) the photons' gamma_gamma leaves the IR
poles (Catani's operator adds none), the conversion's pole is IR
(``epsIR``) and the UV poles still cancel.
"""

from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.loops.v2.assemble import Precision, _Numbers, _PointEvaluation
from pyamplicol.loops.v2.ew_policy import ExternalPhotons

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/loop"
TOLERANCE = 1e-12
ALPHA0_INVERSE = "137.035999"
# alpha_Gmu at the references' inputs (their derived alpha).
ALPHA_GMU = "0.00754677226134035151919702202521"


class _Scale(_PointEvaluation):
    """What ``_PointEvaluation.average`` reads of a point and its slice."""

    def __init__(self, photons, parameters, precision, average=36):
        self.slice = SimpleNamespace(
            normalisation=SimpleNamespace(average=average),
            external_photons=photons,
            compiled=SimpleNamespace(parameter_defaults={}),
            request=SimpleNamespace(process="u u~ > a a"),
        )
        self.parameters = parameters
        self.numbers = _Numbers(precision)

    def _qed_alpha(self):
        return self.numbers.real(Fraction(ALPHA_GMU))


def _photons(*legs):
    return ExternalPhotons(legs, "aEW0M1", "eCoup_DGmua0_UV_EW", "QED")


@pytest.mark.loop_fast
@pytest.mark.parametrize("precision", ["double", "arbitrary"])
def test_each_photon_scales_the_normalisation_by_alpha0_over_alpha(precision):
    """average / (alpha(0) / alpha)^n: one factor per on-shell external
    photon, in the carrier; no photon leaves the slice's average (an int)."""

    precision = Precision.double() if precision == "double" else Precision.arbitrary(40)
    parameters = {"aEW0M1": ALPHA0_INVERSE}
    ratio = Fraction(1) / Fraction(ALPHA0_INVERSE) / Fraction(ALPHA_GMU)
    for legs in ((3,), (3, 4)):
        scale = _Scale(_photons(*legs), parameters, precision)
        with scale.numbers.scope():
            got = scale.average
        want = Fraction(36) / ratio ** len(legs)
        assert abs(Fraction(got) - want) <= Fraction(1, 10**14) * want
    assert _Scale(None, parameters, precision).average == 36


@pytest.mark.loop_fast
def test_alpha0_must_be_set():
    with pytest.raises(ValueError, match=r"set aEW0M1 \(1/alpha\(0\)"):
        _ = _Scale(_photons(3), {"aEW0M1": "0"}, Precision.double()).average


# ------------------------------------------------------------------ engine


def _reference(name):
    return json.loads((FIXTURES / "required" / f"{name}.json").read_text())


def _conversion():
    """MadLoop's eCoup_DGmua0_UV_EW (1/eps and finite, binary64) at MU_R =
    1000 GeV on the G_mu UFO."""

    values = json.loads((FIXTURES / "madloop-ct-parameters.json").read_text())
    found = values["models"]["gmu"]
    assert found["card"]["MU_R"] == "1000"
    delta = found["values"]["eCoup_DGmua0_UV_EW"]
    return float(delta["-1"]["double"][0]), float(delta["0"]["double"][0])


@pytest.fixture(scope="module")
def gmu_a0(loop_gmu_ufo, tmp_path_factory):
    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(
        loop_gmu_ufo,
        restriction="a0",
        require_supported=False,
        cache_dir=tmp_path_factory.mktemp("gmu-a0"),
    )


@pytest.fixture(scope="module")
def photon_slices(gmu_a0, loop_native):
    """The references' requests (``tools/loop_v2_generate._fixture_request``:
    QCD and QED corrections, the EW sector), generated once."""

    import argparse

    from pyamplicol.loops.v2.assemble import LoopSlice
    from tools import loop_v2_generate

    built = {}

    def build(name):
        if name not in built:
            reference = _reference(name)
            request, _ignored = loop_v2_generate._fixture_request(
                reference, gmu_a0, argparse.Namespace()
            )
            built[name] = reference, LoopSlice.generate(gmu_a0, request, loop_native)
        return built[name]

    return build


@pytest.mark.loop_nightly
@pytest.mark.parametrize(("name", "photons"), [("uux_ag_ew", 1), ("uux_aa_ew", 2)])
def test_alpha0_virtual_equals_madloop_with_mg5_conversion(
    photon_slices, name, photons
):
    """Every point of the reference: Born and the virtual's three Laurent
    coefficients equal MadLoop's G_mu values converted to alpha(0) (MG5_aMC's
    formula, MadLoop's DGmua0) to 1e-12 (the fixtures' metric); the
    virtual's poles are Catani's with no gamma_gamma, and the bare and
    counterterm UV poles cancel."""

    reference, slice_ = photon_slices(name)
    assert slice_.external_photons is not None
    assert len(slice_.external_photons.legs) == photons
    pole, finite = _conversion()
    alpha = float(Fraction(ALPHA_GMU))
    r = (1 / float(ALPHA0_INVERSE) / alpha) ** photons
    parameters = {**reference["parameters"], "aEW0M1": ALPHA0_INVERSE}
    for point, mu, row in zip(
        reference["points"], reference["mu_r"], reference["rows"], strict=True
    ):
        assert mu == "1000"
        madloop = row["providers"]["madloop"]
        born = madloop["born"]
        want = [
            r * born,
            r * (madloop["finite"] - 2 * photons * finite * born),
            r * (madloop["single_pole"] - 2 * photons * pole * born),
            r * madloop["double_pole"],
        ]
        result = slice_.evaluate(point, str(Fraction(mu) ** 2), parameters=parameters)
        got = [float(x) for x in result.totals[:4]]
        floor = abs(want[0]) * alpha / (2 * math.pi)
        for k, (mine, theirs) in enumerate(zip(got, want, strict=True)):
            error = abs(mine - theirs) / max(abs(mine), abs(theirs), floor if k else 0)
            assert error <= TOLERANCE, (k, mine, theirs, error)
        ((sector, (single, double)),) = result.ir_poles.items()
        virtual = result.components["virtual"][sector]
        scale = max(abs(float(single)), abs(float(double)))
        assert abs(float(virtual.single_pole) - float(single)) <= TOLERANCE * scale
        assert abs(float(virtual.double_pole) - float(double)) <= TOLERANCE * scale
    uv = slice_.evaluate(
        reference["points"][0], "1000000", parameters=parameters, poles="uv"
    )
    for sector, bare in uv.components["bare"].items():
        counterterms = uv.components["uv"][sector]
        for index in (1, 2):
            total = float(bare.coefficients[index]) + float(
                counterterms.coefficients[index]
            )
            assert abs(total) <= 1e-10 * max(abs(float(bare.coefficients[1])), 1e-300)


@pytest.mark.loop_nightly
def test_external_photon_ward_identity(gmu_a0, loop_native):
    """photon k.M = 0 (PLAN EWI-6b oracle; O10's asymmetric Ward mode): u u~
    > a g EW with alpha(0) for the photon, epsilon_3 -> c p_3 on the loop
    side (blocks, counterterm currents, field insertions) for c = 1 and
    c = i: every renormalised component of the interference with the
    physical Born vanishes per Laurent coefficient, in binary64 within
    1e-11 of the physical scale plus 1e-12 of the run's lanes' absolute sum
    (measured 2.1e-12 of the physical scale; the alpha(0) field term 6e-17),
    while a literal complex vector (non-gauge control) gives every part at
    least 1e-2 of it. Nightly: generation about 7 min."""

    import argparse

    from pyamplicol.loops.v2.assemble import COMPONENTS
    from pyamplicol.loops.v2.conventions import LOOP_FACTOR_DENOMINATOR
    from pyamplicol.loops.v2.fields import FIELD_COMPONENT
    from pyamplicol.loops.v2.observables import ObservableSlice
    from tools import loop_v2_generate

    reference = _reference("uux_ag_ew")
    request, _ignored = loop_v2_generate._fixture_request(
        reference, gmu_a0, argparse.Namespace()
    )
    observables = ObservableSlice.generate(gmu_a0, request, loop_native, ward=True)
    assert observables.slice.external_photons.legs == (3,)
    parameters = {**reference["parameters"], "aEW0M1": ALPHA0_INVERSE}
    point, scale = reference["points"][0], str(Fraction(reference["mu_r"][0]) ** 2)
    physical = observables.slice.evaluate(point, scale, parameters=parameters)
    size = [
        max(
            abs(complex(laurent.coefficients[k]))
            for name in COMPONENTS
            for laurent in physical.components[name].values()
        )
        for k in range(3)
    ]
    average = float(observables.slice.normalisation.average)
    parts = ("virtual", "bare", "r2", "uv_local", FIELD_COMPONENT)
    for factor in ((1, 0), (0, 1)):
        (run,) = observables.evaluate(
            point, scale, ward=(3, factor), parameters=parameters
        ).values()
        lanes = [
            2
            / (LOOP_FACTOR_DENOMINATOR * math.pi**2 * average)
            * sum(abs(complex(lane.coefficients[k])) for lane in run.families.values())
            for k in range(3)
        ]
        for part in parts:
            for sector, laurent in run.components[part].items():
                for k, value in enumerate(laurent.coefficients):
                    bound = 1e-11 * size[k] + 1e-12 * lanes[k]
                    assert abs(complex(value)) <= bound, (factor, part, sector, k)
    vector = (("0.1", "0.2"), ("0.8", "-0.3"), ("0.4", "1"), ("-0.6", "0.5"))
    (control,) = observables.evaluate(
        point, scale, ward=(3, vector), parameters=parameters
    ).values()
    for part in parts:
        largest = max(
            abs(complex(value)) / size[k]
            for laurent in control.components[part].values()
            for k, value in enumerate(laurent.coefficients)
        )
        assert largest > 1e-2, (part, largest)
