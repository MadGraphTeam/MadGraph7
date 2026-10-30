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
factor of each photon leg (``assemble._alpha0_field_rules``). The native
point inputs (WP3.2 EWI-N) take r in the photons' sources instead: each
source of a photon leg times sqrt(alpha(0)/alpha), a parameter-program
output, so the average stays the slice's.

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
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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


def _madloop_alpha0(row, photons):
    """MadLoop's G_mu Born and virtual of a reference row converted to
    alpha(0) (MG5_aMC's formula, MadLoop's DGmua0): Born, finite part, single
    and double pole."""

    pole, finite = _conversion()
    r = (1 / float(ALPHA0_INVERSE) / float(Fraction(ALPHA_GMU))) ** photons
    madloop = row["providers"]["madloop"]
    born = madloop["born"]
    return [
        r * born,
        r * (madloop["finite"] - 2 * photons * finite * born),
        r * (madloop["single_pole"] - 2 * photons * pole * born),
        r * madloop["double_pole"],
    ]


@pytest.mark.loop_nightly
@pytest.mark.parametrize(("name", "photons"), [("uux_ag_ew", 1), ("uux_aa_ew", 2)])
def test_alpha0_native_inputs_equal_madloop_and_the_python_inputs(
    photon_slices, name, photons
):
    """WP3.2 EWI-N: the native point inputs take the convention. The tree
    program scales every source of the photon legs by sqrt(alpha(0)/alpha),
    in the trees and the counterterm program (``loop_v2::trees``), from the
    parameter program's output alpha(0)/alpha (``native_trees``' ``photons``
    record, on the inputs 1/alpha(0) and the photon coupling's), and the
    conversion is a field insertion of the photon legs. At every reference
    point Born and the virtual's three Laurent coefficients equal MadLoop's
    converted values to 1e-12 (the fixtures' metric), and Born and every
    component equal the Python inputs' on the same exported plan within
    1e-12 of the sector's virtual (binary64; measured 1.1e-15 and 3.4e-15).
    Shares the generation with the test above (module fixture)."""

    from pyamplicol.loops.v2.plan_export import NativeExecutor

    reference, slice_ = photon_slices(name)
    native = NativeExecutor(slice_, forward=False, inputs="native")
    record = native.exported.trees.program["parameters"]
    ratio, legs = record["photons"]
    assert legs == list(slice_.external_photons.legs) and ratio in record["positive"]
    assert "aEW0M1" in record["names"]
    python = NativeExecutor(slice_, forward=False, exported=native.exported)
    parameters = {**reference["parameters"], "aEW0M1": ALPHA0_INVERSE}
    alpha = float(Fraction(ALPHA_GMU))
    for point, mu, row in zip(
        reference["points"], reference["mu_r"], reference["rows"], strict=True
    ):
        want = _madloop_alpha0(row, photons)
        scale = str(Fraction(mu) ** 2)
        (mine,) = native.evaluate([point], [scale], parameters=parameters)
        (theirs,) = python.evaluate([point], [scale], parameters=parameters)
        got = [float(x) for x in mine.totals[:4]]
        floor = abs(want[0]) * alpha / (2 * math.pi)
        for k, (value, expected) in enumerate(zip(got, want, strict=True)):
            error = abs(value - expected) / max(
                abs(value), abs(expected), floor if k else 0
            )
            assert error <= TOLERANCE, (k, value, expected, error)
        assert mine.born.keys() == theirs.born.keys() and theirs.born
        for order, value in theirs.born.items():
            assert abs(float(mine.born[order]) - float(value)) <= TOLERANCE * abs(
                float(value)
            )
        assert mine.components.keys() == theirs.components.keys()
        for component, sectors in theirs.components.items():
            for sector, laurent in sectors.items():
                size = max(
                    abs(float(x))
                    for x in theirs.components["virtual"][sector].coefficients
                )
                for x, y in zip(
                    mine.components[component][sector].coefficients,
                    laurent.coefficients,
                    strict=True,
                ):
                    assert abs(float(x) - float(y)) <= TOLERANCE * size, (
                        component,
                        sector,
                    )
    # Without 1/alpha(0) the native binder refuses the point, naming it.
    with pytest.raises(ValueError, match=r"set the model's 1/alpha\(0\) parameter"):
        native.evaluate(
            [reference["points"][0]],
            [str(Fraction(reference["mu_r"][0]) ** 2)],
            parameters={**reference["parameters"], "aEW0M1": "0"},
        )


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
    least 1e-2 of it; the same with native point inputs, whose control
    equals the Python lane's. Nightly: generation about 2-7 min."""

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

    def vanishes(run, factor):
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

    for factor in ((1, 0), (0, 1)):
        (run,) = observables.evaluate(
            point, scale, ward=(3, factor), parameters=parameters
        ).values()
        vanishes(run, factor)
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
    # The native inputs (WP3.2 EWI-N): the photon's loop-side vector takes
    # the photons' source scale sqrt(alpha(0)/alpha) as its physical source
    # does, so the native Ward run vanishes within the same bound (measured
    # 8.3e-12 of the physical scale) and the control equals the Python
    # lane's (measured 5.4e-15 of it).
    from pyamplicol.loops.v2.plan_export import NativeExecutor

    native = NativeExecutor(observables.slice, forward=False, inputs="native")
    for factor in ((1, 0), (0, 1)):
        (run,) = native.evaluate(
            [point], [scale], ward=(3, factor), parameters=parameters
        )
        vanishes(run, factor)
    (mine,) = native.evaluate([point], [scale], ward=(3, vector), parameters=parameters)
    for part in parts:
        for sector, laurent in control.components[part].items():
            for k, (x, y) in enumerate(
                zip(
                    mine.components[part][sector].coefficients,
                    laurent.coefficients,
                    strict=True,
                )
            ):
                assert abs(complex(x) - complex(y)) <= 1e-12 * size[k], (part, k)


# ------------------------------------------------------------------ per leg
#
# MS-PHOTON (m-sm/PLAN.md §3.2, Q1 (a)): the alpha(0) convention per tagged
# leg (``SliceRequest.external_photons``, MG5_aMC's ``!a!``; default
# "final"). The rows' references are raw MadLoop G_mu (MadLoop standalone
# converts no photon) times MG5_aMC's formula over the tagged legs: Born and
# virtual times (alpha(0)/alpha_Gmu)^n_tag, the virtual of the
# electromagnetic correction minus 2 n_tag DGmua0 Born; untagged photons are
# raw MadLoop, with -gamma_gamma/eps each in the IR poles (rule 4 per leg).
# DGmua0 at a reference point's MU_R is MadLoop's 1000 GeV value shifted by
# its IR pole's logarithm, pole x ln(MU_R^2 / 1000^2): the UFO's
# eCoup_DGmua0_UV_EW depends on MU_R only through it (checked on the UFO's
# own expressions at 500, 700 and 1400 GeV: 2e-15).


@pytest.mark.loop_fast
def test_photon_tag_option_values():
    from pyamplicol.loops.v2.assemble import SliceRequest
    from pyamplicol.loops.v2.ew_policy import photon_tag_option

    assert photon_tag_option("final") == "final"
    assert photon_tag_option([3, 1]) == (1, 3)
    assert photon_tag_option(()) == ()
    for bad in ("initial", [0], [1, 1], [True], 3, b"all", {"1": 1}):
        with pytest.raises(ValueError, match="external_photons"):
            photon_tag_option(bad)
    request = SliceRequest("a u > a u", {"QED": 2}, {"QCD": 2, "QED": 2})
    assert request.external_photons == "final"
    listed = SliceRequest(
        "a u > a u", {"QED": 2}, {"QCD": 2, "QED": 2}, external_photons=[3, 1]
    )
    assert listed.external_photons == (1, 3)


@pytest.mark.loop_fast
def test_request_records_name_non_default_tags_only():
    """The request record names the tags only when they are not the default
    "final": the records of every other request keep their bytes, and a
    record written before the field existed restores the default."""

    from pyamplicol.loops.v2.artifact import request_from_record, request_record
    from pyamplicol.loops.v2.assemble import SliceRequest

    plain = SliceRequest("u u~ > a g", {"QCD": 1, "QED": 1}, {"QCD": 3, "QED": 1})
    record = request_record(plain)
    assert not {"external_photons", "alpha0_conversion", "alpha0_inverse"} & set(record)
    named = SliceRequest(
        "u u~ > a g",
        {"QCD": 1, "QED": 1},
        {"QCD": 3, "QED": 1},
        alpha0_conversion="dZe_conv",
        alpha0_inverse="ainv0",
    )
    record = json.loads(json.dumps(request_record(named)))
    assert (record["alpha0_conversion"], record["alpha0_inverse"]) == (
        "dZe_conv",
        "ainv0",
    )
    assert request_from_record(record) == named
    record = request_record(plain)
    assert request_from_record(json.loads(json.dumps(record))) == plain
    for option in ("all", "none", (1, 3)):
        tagged = SliceRequest(
            "a u > a u", {"QED": 2}, {"QCD": 2, "QED": 2}, external_photons=option
        )
        record = json.loads(json.dumps(request_record(tagged)))
        assert record["external_photons"] == (
            option if isinstance(option, str) else list(option)
        )
        assert request_from_record(record) == tagged


@pytest.mark.loop_fast
def test_loop_external_photons_configuration():
    """``loop.external_photons`` in TOML, overrides and Python: a choice or a
    list of 1-based leg labels (initial legs allowed), "final" by default."""

    from pyamplicol.config.errors import ConfigurationError
    from pyamplicol.config.models import LoopConfig
    from pyamplicol.config.registry import FIELD_REGISTRY
    from pyamplicol.config.resolver import parse_override

    assert FIELD_REGISTRY["loop.external_photons"].default == "final"
    assert FIELD_REGISTRY["loop.alpha0_conversion_parameter"].default == (
        "eCoup_DGmua0_UV_EW"
    )
    assert FIELD_REGISTRY["loop.alpha0_inverse_parameter"].default is None
    assert LoopConfig().external_photons == "final"
    assert LoopConfig(external_photons=[3, 1]).external_photons == (1, 3)
    assert parse_override("loop.external_photons=none").value == "none"
    assert parse_override("loop.external_photons=all").value == "all"
    assert parse_override("loop.external_photons=[1, 3]").value == (1, 3)
    for bad in ("initial", "[1.5]", "3"):
        with pytest.raises(ConfigurationError, match=r"loop\.external_photons"):
            LoopConfig(
                external_photons=parse_override(f"loop.external_photons={bad}").value
            )
    with pytest.raises(ConfigurationError, match=r"loop\.external_photons"):
        LoopConfig(external_photons=[0])


@pytest.mark.loop_fast
def test_photon_refusals_are_listed():
    from pyamplicol.loops.v2.ew_policy import ExternalPhotonError
    from pyamplicol.loops.v2.refusals import (
        LoopRequestUnsupported,
        listed_refusal_keys,
    )

    assert {"external-photon-scheme", "external-photon-legs"} <= set(
        listed_refusal_keys()
    )
    error = ExternalPhotonError("external-photon-legs", "x")
    assert isinstance(error, LoopRequestUnsupported) and error.key == (
        "external-photon-legs"
    )


def _leg(label, pdg, side):
    return SimpleNamespace(label=label, pdg=pdg, side=side)


def _photon_model():
    def record(name, pdg, spin, charge=0.0, mass="ZERO", color=1):
        return SimpleNamespace(
            name=name,
            pdg_code=pdg,
            spin=spin,
            color=color,
            charge=charge,
            mass=mass,
            ghost_number=0,
            goldstoneboson=False,
        )

    return SimpleNamespace(
        ir=SimpleNamespace(
            particles=(
                record("a", 22, 3),
                record("u", 2, 2, 2 / 3, color=3),
                record("Z", 23, 3, mass="MZ"),
            ),
            parameters=(SimpleNamespace(name="MZ", nature="external", value="91"),),
        )
    )


@pytest.mark.loop_fast
@pytest.mark.parametrize(
    ("option", "tagged", "untagged"),
    [
        ("final", (3,), (1,)),
        ("all", (1, 3), ()),
        ("none", (), (1, 3)),
        ((1,), (1,), (3,)),
        ((1, 3), (1, 3), ()),
    ],
)
def test_photon_tags_per_leg(option, tagged, untagged):
    """a u > a u: "final" tags the final photon (leg 3) and leaves the
    initial one (a PDF photon) untagged; a list tags exactly its legs."""

    from pyamplicol.loops.v2 import ew_policy

    process = SimpleNamespace(
        legs=(_leg(1, 22, "initial"), _leg(2, 2, "initial"), _leg(3, 22, "final"),
              _leg(4, 2, "final"))
    )  # fmt: skip
    request = SimpleNamespace(process="a u > a u", external_photons=option)
    found = ew_policy.photon_tags(_photon_model(), request, process, {})
    assert found == (tagged, untagged)


@pytest.mark.loop_fast
def test_listed_legs_must_be_on_shell_photons():
    from pyamplicol.loops.v2 import ew_policy

    process = SimpleNamespace(
        legs=(_leg(1, 22, "initial"), _leg(2, 2, "initial"), _leg(3, 23, "final"),
              _leg(4, 2, "final"))
    )  # fmt: skip
    for option, named in (((2,), "2 (u, initial)"), ((3,), "3 (Z, final)"),
                          ((1, 7), "7 (the process has 4 legs)")):  # fmt: skip
        request = SimpleNamespace(process="a u > z u", external_photons=option)
        with pytest.raises(ew_policy.ExternalPhotonError) as refused:
            ew_policy.photon_tags(_photon_model(), request, process, {})
        assert refused.value.key == "external-photon-legs"
        assert named in str(refused.value)
        assert "on-shell photons: legs [1]" in str(refused.value)


# Measured (binary64, points 0 and 1): MadLoop x the formula within
# 1.9e-12 (LL08, 2->3; <= 6.4e-13 at 2->2), rule 4 within 1.5e-12, native =
# Python inputs within 9.8e-13 (LL08; <= 1.8e-13 at 2->2).
ROW_TOLERANCE = 1e-10
ROW_NATIVE_TOLERANCE = 1e-11
_GMU_OPTIONS = {"compatibility": "gmu_particle_anti", "scale_parameter": "MU_R"}


@dataclass(frozen=True)
class _PhotonRow:
    """A photon row of m-sm/PLAN.md §4.2: its MadLoop fixtures (each serving
    the sectors it lists, at shared points), process, corrections, Born
    amplitude orders, model (``gmu`` = G0, ``gmu_a0`` = Ga) and tags."""

    fixtures: tuple[str, ...]
    process: str
    corrections: tuple[str, ...]
    born: tuple[tuple[str, int], ...]
    model: str
    option: Any
    tagged: tuple[int, ...]
    untagged: tuple[int, ...]

    @property
    def loop_orders(self) -> tuple[dict[str, int], ...]:
        born = dict(self.born)
        return tuple({**born, name: born.get(name, 0) + 2} for name in self.corrections)


PHOTON_ROWS = {
    # Two initial (PDF) photons, untagged by default: raw MadLoop.
    "PI01": _PhotonRow(("msm/aa_emep_ew",), "a a > e- e+", ("QED",),
                       (("QCD", 0), ("QED", 2)), "gmu", "final", (), (1, 2)),
    # Its "all" twin on Ga: both photons tagged (one MadLoop job serves both).
    "PI01-all": _PhotonRow(("msm/aa_emep_ew",), "a a > e- e+", ("QED",),
                           (("QCD", 0), ("QED", 2)), "gmu_a0", "all", (1, 2), ()),
    "PI02": _PhotonRow(("msm/aa_wpwm_ew",), "a a > w+ w-", ("QED",),
                       (("QCD", 0), ("QED", 2)), "gmu", "final", (), (1, 2)),
    "PI03": _PhotonRow(("msm/ag_uux_ew",), "a g > u u~", ("QCD", "QED"),
                       (("QCD", 1), ("QED", 1)), "gmu", "final", (), (1,)),
    # The initial photon untagged, the final one tagged.
    "PI04": _PhotonRow(("msm/au_au_ew_a0",), "a u > a u", ("QCD", "QED"),
                       (("QCD", 0), ("QED", 2)), "gmu_a0", "final", (3,), (1,)),
    "VJ03": _PhotonRow(("uux_ag_ew", "msm/uux_ag_ew_qcd"), "u u~ > a g",
                       ("QCD", "QED"), (("QCD", 1), ("QED", 1)), "gmu_a0", "final",
                       (3,), ()),
    "VV05": _PhotonRow(("msm/udx_wpa_ew_a0",), "u d~ > w+ a", ("QCD", "QED"),
                       (("QCD", 0), ("QED", 2)), "gmu_a0", "final", (4,), ()),
    "VV06": _PhotonRow(("uux_aa_ew", "msm/uux_aa_ew_qcd"), "u u~ > a a",
                       ("QCD", "QED"), (("QCD", 0), ("QED", 2)), "gmu_a0", "final",
                       (3, 4), ()),
    "LL08": _PhotonRow(("msm/emep_mummpa_ew_a0",), "e- e+ > m- m+ a", ("QED",),
                       (("QCD", 0), ("QED", 3)), "gmu_a0", "final", (5,), ()),
}  # fmt: skip


def _sectors(reference, row):
    """(Born per squared order, MadLoop virtual (finite, single, double) per
    squared order) of a fixture row: its ``sectors`` record, else the
    fixture's one sector (an older fixture: Born orders twice, plus its loop
    orders)."""

    def key(orders):
        return tuple(sorted((k, int(v)) for k, v in orders.items() if int(v)))

    madloop = row["providers"]["madloop"]
    if "sectors" in row:
        born = {
            key({"QCD": s["QCD"], "QED": s["QED"]}): s["born"]
            for s in row["sectors"]["born"]
        }
        virtual = {
            key({"QCD": s["QCD"], "QED": s["QED"]}): (
                s["finite"],
                s["single_pole"],
                s["double_pole"],
            )
            for s in row["sectors"]["loop"]
        }
        return born, virtual
    orders = reference["born_orders"]
    loops = reference["loop_orders"]
    born = {key({k: 2 * v for k, v in orders.items()}): madloop["born"]}
    sector = key({k: orders.get(k, 0) + loops.get(k, 0) for k in {*orders, *loops}})
    return born, {
        sector: (madloop["finite"], madloop["single_pole"], madloop["double_pole"])
    }


def _converted(row: _PhotonRow, born, virtual, mu):
    """MG5_aMC's formula over the row's tagged legs on raw MadLoop values."""

    n = len(row.tagged)
    if not n:
        return born, virtual
    pole, finite = _conversion()
    finite += pole * math.log((float(mu) / 1000) ** 2)
    r = (1 / float(ALPHA0_INVERSE) / float(Fraction(ALPHA_GMU))) ** n
    total = sum(born.values())
    (born_order,) = born
    corrected = tuple(
        sorted({**dict(born_order), "QED": dict(born_order).get("QED", 0) + 2}.items())
    )
    out = {}
    for sector, (fin, single, double) in virtual.items():
        if sector == corrected:
            fin -= 2 * n * finite * total
            single -= 2 * n * pole * total
        out[sector] = (r * fin, r * single, r * double)
    return {k: r * v for k, v in born.items()}, out


@pytest.fixture(scope="module")
def photon_models(request, gmu_a0):
    def model(name):
        return gmu_a0 if name == "gmu_a0" else request.getfixturevalue("gmu_default")

    return model


@pytest.fixture(scope="module")
def gmu_default(loop_gmu_ufo, tmp_path_factory):
    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(
        loop_gmu_ufo,
        require_supported=False,
        cache_dir=tmp_path_factory.mktemp("gmu-default"),
    )


@pytest.mark.loop_nightly
@pytest.mark.parametrize("name", sorted(PHOTON_ROWS))
def test_photon_rows_equal_madloop_with_the_formula_over_tagged_legs(
    photon_models, loop_native, name
):
    """m-sm/PLAN.md MS-PHOTON acceptance (PI01-PI04, PI01's "all" twin,
    VJ03, VV05, VV06, LL08): at the first two reference points the Born and
    every sector's Laurent coefficients equal raw MadLoop times MG5_aMC's
    formula over the tagged legs (binary64, the fixtures' metric, 1e-10), with
    Python and with native point inputs; rule 4 per leg: every sector is
    predicted and its poles are Catani's, an untagged photon adding
    -gamma_gamma/eps (a tagged one nothing); the native binder scales the
    tagged legs only (none without tags, so no 1/alpha(0) input), and native
    = Python inputs within 1e-11 of each sector's virtual. Nightly: one EW
    2->2 or 2->3 generation (about 1-8 min)."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from tools import loop_v2_generate

    row = PHOTON_ROWS[name]
    compiled = photon_models(row.model)
    references = [_reference(fixture) for fixture in row.fixtures]
    first = references[0]
    externals = loop_v2_generate._external_parameters(compiled)
    fixed = {
        k: v for k, v in first.get("fixed_parameters", {}).items() if k in externals
    }
    request = SliceRequest(
        row.process,
        dict(row.born),
        row.loop_orders,
        corrections=row.corrections,
        fixed_parameters=fixed,
        external_photons=row.option,
        **_GMU_OPTIONS,
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    photons = slice_.external_photons
    assert (() if photons is None else photons.legs) == row.tagged
    from pyamplicol.loops.v2.ew_policy import scheme_conventions

    conventions = scheme_conventions(slice_)
    assert conventions["tagged_photon_legs"] == list(row.tagged)
    assert conventions["untagged_photon_legs"] == list(row.untagged)
    parameters = {k: v for k, v in first["parameters"].items() if k in externals}
    if row.tagged:
        parameters["aEW0M1"] = ALPHA0_INVERSE
    native = NativeExecutor(slice_, forward=False, inputs="native")
    record = native.exported.trees.program["parameters"]
    if row.tagged:
        assert record["photons"][1] == list(row.tagged)
    else:
        assert record.get("photons") is None and "aEW0M1" not in record["names"]
    alpha = float(Fraction(ALPHA_GMU))
    worst = {"python": 0.0, "native": 0.0, "rule 4": 0.0, "native-python": 0.0}
    for index in (0, 1):
        point = first["points"][index]
        mu = first["mu_r"][index]
        born, virtual = {}, {}
        for reference in references:
            assert reference["points"][index] == point
            assert reference["mu_r"][index] == mu
            b, v = _sectors(reference, reference["rows"][index])
            born.update(b)
            virtual.update(v)
        born, virtual = _converted(row, born, virtual, mu)
        scale = str(Fraction(mu) ** 2)
        python = slice_.evaluate(point, scale, parameters=parameters)
        (mine,) = native.evaluate([point], [scale], parameters=parameters)
        floor = sum(abs(x) for x in born.values()) * alpha / (2 * math.pi)
        for lane, result in (("python", python), ("native", mine)):
            assert set(result.born) == set(born)
            for order, value in born.items():
                got = float(result.born[order])
                error = abs(got - value) / abs(value)
                worst[lane] = max(worst[lane], error)
                assert error <= ROW_TOLERANCE, (order, got)
            assert set(result.components["virtual"]) == set(virtual)
            for sector, want in virtual.items():
                got = [
                    float(x) for x in result.components["virtual"][sector].coefficients
                ]
                for k, (x, y) in enumerate(zip(got, want, strict=True)):
                    error = abs(x - y) / max(abs(x), abs(y), floor)
                    worst[lane] = max(worst[lane], error)
                    assert error <= ROW_TOLERANCE, (index, sector, k, x, y, error)
        # Rule 4 per leg: every sector predicted, poles = Catani's.
        assert set(python.ir_poles) == set(virtual)
        for sector, (single, double) in python.ir_poles.items():
            laurent = python.components["virtual"][sector].coefficients
            size = max(abs(float(single)), abs(float(double)), floor)
            for got, want in ((laurent[1], single), (laurent[2], double)):
                error = abs(float(got) - float(want)) / size
                worst["rule 4"] = max(worst["rule 4"], error)
                assert error <= ROW_TOLERANCE, (sector, got, want)
        for component, sectors in python.components.items():
            for sector, laurent in sectors.items():
                size = max(
                    abs(float(x))
                    for x in python.components["virtual"][sector].coefficients
                )
                for x, y in zip(
                    mine.components[component][sector].coefficients,
                    laurent.coefficients,
                    strict=True,
                ):
                    error = abs(float(x) - float(y)) / size
                    worst["native-python"] = max(worst["native-python"], error)
                    assert error <= ROW_NATIVE_TOLERANCE, (component, sector)
    print(f"{name}: worst errors {worst}")


@pytest.mark.loop_fast
def test_an_ambiguous_photon_is_refused():
    """Genericity audit 2 G2-m6: a model with a second massless, neutral,
    colourless vector (a massless Z') makes the photon ambiguous, so an
    electroweak request with such an external leg is refused (listed,
    ``external-photon-scheme``) whether the leg is tagged or not; the model's
    one photon passes this check."""

    from pyamplicol.loops.v2 import ew_policy

    model = _photon_model()
    process = SimpleNamespace(
        legs=(_leg(1, 22, "initial"), _leg(2, 2, "initial"), _leg(3, 22, "final"),
              _leg(4, 2, "final"))
    )  # fmt: skip
    assert ew_policy.photon_species(model, {}) == {22}
    dark = SimpleNamespace(**{**vars(model.ir.particles[0]), "name": "zp"})
    dark.pdg_code = 9000032
    ambiguous = SimpleNamespace(
        ir=SimpleNamespace(
            particles=(*model.ir.particles, dark), parameters=model.ir.parameters
        )
    )
    assert ew_policy.photon_species(ambiguous, {}) == {22, 9000032}
    for option in ("final", "none"):
        request = SimpleNamespace(
            process="a u > a u",
            external_photons=option,
            corrections=("QCD", "QED"),
            particle_veto=(),
        )
        with pytest.raises(ew_policy.ExternalPhotonError) as refused:
            ew_policy.external_photon_scheme(ambiguous, request, process, {}, {})
        assert refused.value.key == "external-photon-scheme"
        assert "|PDG| [22, 9000032]" in str(refused.value)


@pytest.mark.loop_nightly
def test_untagged_photon_request_on_the_alpha_mz_model_generates(
    loop_qcd_qed_sm_ufo, loop_native, tmp_path_factory
):
    """MS-PHOTON: loop_qcd_qed_sm (alpha(M_Z) input, no alpha(0) parameter
    and no conversion counterterm, where MG5_aMC refuses tagged photons)
    generates u u~ > a g [QCD QED] with its photon untagged
    (``external_photons = "none"``): the IR poles of both sectors equal
    Catani's, the QED one with the untagged photon's -gamma_gamma/eps, the
    UV poles of the bare loops and the counterterms cancel, and the native
    point inputs (no photon scaling) equal the Python inputs within 1e-12 of
    each sector's virtual. The default tag ("final") is refused on this model
    (``external-photon-scheme``, naming "none" as the remedy). Nightly: one
    EW 2->2 generation (about 2-4 min)."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.loops.v2.ew_policy import ExternalPhotonError
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.models.loading import compile_model_source

    compiled = compile_model_source(
        loop_qcd_qed_sm_ufo,
        require_supported=False,
        cache_dir=tmp_path_factory.mktemp("qcd-qed"),
    )
    fixed = {"MB": "0", "ymb": "0", "WT": "0", "WW": "0", "WZ": "0", "WH": "0"}
    options = {"corrections": ("QCD", "QED"), "fixed_parameters": fixed}
    loops = ({"QCD": 3, "QED": 1}, {"QCD": 1, "QED": 3})
    born = {"QCD": 1, "QED": 1}
    with pytest.raises(ExternalPhotonError) as refused:
        LoopSlice.generate(
            compiled,
            SliceRequest("u u~ > a g", born, loops, **options, **_GMU_OPTIONS),
            loop_native,
        )
    assert refused.value.key == "external-photon-scheme"
    assert 'loop.external_photons = "none"' in str(refused.value)
    request = SliceRequest(
        "u u~ > a g", born, loops, external_photons="none", **options, **_GMU_OPTIONS
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    assert slice_.external_photons is None
    reference = _reference("uux_ag_ew")
    point, scale = reference["points"][0], str(Fraction(reference["mu_r"][0]) ** 2)
    result = slice_.evaluate(point, scale)
    sectors = set(result.components["virtual"])
    assert len(sectors) == 2 and set(result.ir_poles) == sectors
    for sector, (single, double) in result.ir_poles.items():
        laurent = result.components["virtual"][sector].coefficients
        size = max(abs(float(single)), abs(float(double)))
        assert abs(float(laurent[1]) - float(single)) <= 1e-10 * size, sector
        assert abs(float(laurent[2]) - float(double)) <= 1e-10 * size, sector
    uv = slice_.evaluate(point, scale, poles="uv")
    for sector, bare in uv.components["bare"].items():
        counterterms = uv.components["uv"][sector]
        for index in (1, 2):
            total = float(bare.coefficients[index]) + float(
                counterterms.coefficients[index]
            )
            assert abs(total) <= 1e-10 * max(abs(float(bare.coefficients[1])), 1e-300)
    native = NativeExecutor(slice_, forward=False, inputs="native")
    assert native.exported.trees.program["parameters"].get("photons") is None
    (mine,) = native.evaluate([point], [scale])
    for sector, laurent in result.components["virtual"].items():
        size = max(abs(float(x)) for x in laurent.coefficients)
        for x, y in zip(
            mine.components["virtual"][sector].coefficients,
            laurent.coefficients,
            strict=True,
        ):
            assert abs(float(x) - float(y)) <= 1e-12 * size, sector
