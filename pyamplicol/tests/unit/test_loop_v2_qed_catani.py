# SPDX-License-Identifier: 0BSD
"""QED Catani operator of the v2 virtual (WP3.2 EWI-6; ``loops.v2.poles``).

Operator-level checks: no electroweak request generates before the
counterterm lowering of WP3.2 EWI-3, so the engine's operator
(``_PointEvaluation._ir_poles`` and ``_qed_poles``) runs at each fixture
point on a stand-in point with the model's legs, the fixture's Born and its
parameters (``_OperatorPoint``):

* with the G_mu UFO's charges, masses and alpha (``poles.qed_coupling``:
  alpha_Gmu), the operator equals the IR poles of every provider of the four
  three-provider electroweak rungs (MadLoop, Recola, OpenLoops; g g > t t~,
  u u~ > e- e+ g, g g > t t~ e- e+ and g g > t t~ e- e+ g at O(alpha): massive
  pairs, massive-massless and massless pairs) and of the MadLoop G_mu
  controls e- e+ > mu- mu+ (two sets of scales) and e- e+ > u u~. These
  Borns have one QCD order, so the cross-order QCD operator dresses no Born
  interference (``poles.dressed_born``) and the QED operator is the whole
  prediction;
* the operator's parts: exact UFO charges, all-incoming charges (physical
  charges of outgoing legs are refused: they do not add to zero), the
  massless closed form, a massive pair, the external-photon conventions
  (a tagged photon, alpha(0), the user's decision [U9], adds no gamma_gamma;
  an untagged one, MS-PHOTON, adds -gamma_gamma/eps) and no prediction for a
  massless charged vector;
* the per-correction structure: the Born interferences each correction
  dresses (cross orders included), the sum over the corrections, and no
  prediction for a correction without an operator, never a partial sum (the
  silent 0 a joint QCD + QED request got before);
* alpha of the photon couplings in binary64 and at 40 digits: alpha_Gmu
  (the fixtures' derived value) on the G_mu UFO, 1/aEWM1 on loop_sm.

The G_mu control with a massive muon (``electroweak-massive-mumu-control
.json``) is not a reference of the operator: its single pole exceeds the
massive operator by 3.00000 (alpha/2pi) Born at its three points (to 1e-5).
That is the IR pole by which the on-shell field renormalisation of a
massive muon pair differs from a massless one, -(alpha/2pi) 3/eps Born. Its
UFO puts the lepton field renormalisation into the vertex counterterms and
writes it for massless leptons; the control changes only the muon mass in
the restriction card.
"""

from __future__ import annotations

import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyamplicol.loops.v2.assemble import Precision, _Numbers, _PointEvaluation
from pyamplicol.loops.v2.ew_policy import CorrectionOrders
from pyamplicol.loops.v2.families import ExternalLeg, MassId, add_orders, mass_name
from pyamplicol.loops.v2.poles import (
    EXTERNAL_PHOTON_CONVENTION,
    coupling_value,
    dressed_born,
    exact_charge,
    photon_gamma,
    qed_catani_poles,
    qed_coupling,
    qed_gamma,
    sector_sum,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/loop"
# The controls' phase-space points (``phase_space_fixture``).
CONTROL_POINTS = ROOT / "tools/loop_reference/uudd_qcd.json"
REQUIRED = ("gg_ttx_ew", "uux_emepg_ew", "gg_ttxemep_ew", "gg_ttxemepg_ew")
# The fixtures' metric: |a - b| / max(|a|, |b|, |Born| alpha / 2pi), against
# the primary provider (MadLoop); every validated provider is held to the
# fixture's own cross-provider tolerance (``tolerance``, 2e-8: Recola's single
# poles of the 2 -> 4 and 2 -> 5 rungs differ from MadLoop's by up to 1e-9).
PRIMARY_TOLERANCE = 1e-11
CONTROL_TOLERANCE = 1e-12
# The G_mu inputs of the MadLoop controls (their ``parameters``).
CONTROL_PARAMETERS = {
    "Gf": "0.0000116639",
    "MW": "80.419",
    "MZ": "91.188",
    "MH": "125",
    "MT": "173.3",
}
ZERO_WIDTHS = {"WT": "0", "WW": "0", "WZ": "0", "WH": "0"}
# alpha_Gmu at these inputs (the fixtures' derived value), for the metric.
ALPHA_GMU = 0.0075467722613403515


@pytest.fixture(scope="module")
def gmu(loop_gmu_ufo):
    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(loop_gmu_ufo, require_supported=False)


def _is_photon(compiled):
    from pyamplicol.loops.v2.ew_policy import is_photon

    parameters = {record.name: record for record in compiled.ir.parameters}
    return lambda record: is_photon(record, parameters, {})


class _OperatorPoint(_PointEvaluation):
    """What ``_PointEvaluation._ir_poles`` reads of a point and its slice:
    the model's external legs (the incoming field of each, outgoing legs
    crossed), all-incoming momenta, mu_R^2, the point's parameters and the
    Born of one squared Born order at one sector."""

    def __init__(
        self,
        compiled,
        pdgs,
        point,
        scale2,
        parameters,
        born,
        sector,
        *,
        corrections=("QED",),
        precision=None,
        photons=None,
    ):
        records = {int(record.pdg_code): record for record in compiled.ir.particles}
        legs, incoming = [], []
        for index, (pdg, momentum) in enumerate(zip(pdgs, point, strict=True)):
            initial = index < 2
            record = records[pdg] if initial else records.get(-pdg, records[pdg])
            legs.append(
                ExternalLeg(
                    index,
                    str(record.name),
                    MassId(mass_name(str(record.mass)), mass_name(str(record.width))),
                    int(record.color),
                )
            )
            exact = [Fraction(str(x)) for x in momentum]
            incoming.append(exact if initial else [-x for x in exact])
        names = {record.name for record in compiled.ir.parameters}
        # Every charged species is perturbative in QED (the untagged
        # photons' gamma_gamma reads them, ``_photon_gamma``).
        species = {
            abs(int(record.pdg_code)): ("QED",)
            for record in compiled.ir.particles
            if float(record.charge)
        }
        self.slice = SimpleNamespace(
            compiled=compiled,
            request=SimpleNamespace(
                corrections=corrections,
                fixed_parameters=dict(ZERO_WIDTHS),
                particle_veto=(),
            ),
            prepared=SimpleNamespace(
                structure=SimpleNamespace(perturbative_species_orders=species)
            ),
            families=SimpleNamespace(legs=tuple(legs)),
            sectors=((sector, frozenset()),),
            external_photons=photons,
            _cache={},
        )
        self.numbers = _Numbers(precision or Precision.double())
        self.parameters = {k: v for k, v in parameters.items() if k in names}
        self.incoming = incoming
        self.scale = Fraction(str(scale2))
        self.born = born

    def _predicted_sectors(self):
        # Operator-level points: their one sector is complete.
        return frozenset(sector for sector, _pairs in self.slice.sectors)


def _orders(orders) -> tuple:
    return add_orders(tuple(orders.items()))


def _metric(got: float, want: float, floor: float) -> float:
    return abs(got - want) / max(abs(got), abs(want), floor)


@pytest.mark.loop_physics
@pytest.mark.parametrize("name", REQUIRED)
def test_qed_operator_equals_the_providers_poles_of_the_electroweak_rungs(gmu, name):
    """Every point and every provider of a three-provider electroweak rung:
    the engine's operator (QED only: the Born has one QCD order, which the
    QCD correction of the sector cannot dress) against each provider's
    single and double pole, the primary one (MadLoop) to 1e-11."""

    fixture = json.loads((FIXTURES / "required" / f"{name}.json").read_text())
    born, loop = _orders(fixture["born_orders"]), _orders(fixture["loop_orders"])
    born_order, sector = add_orders(born, born), add_orders(born, loop)
    parameters = {**fixture["parameters"], **fixture["fixed_parameters"]}
    alpha = float(Fraction(fixture["derived_parameters"]["alpha"]))
    worst: dict[str, float] = {}
    for point, mu, row in zip(
        fixture["points"], fixture["mu_r"], fixture["rows"], strict=True
    ):
        born = {born_order: row["born"]}
        assert dressed_born(born, sector, "QCD", 0.0) == 0.0
        evaluation = _OperatorPoint(
            gmu,
            fixture["pdgs"],
            point,
            Fraction(str(mu)) ** 2,
            parameters,
            born,
            sector,
            corrections=("QCD", "QED"),
        )
        # The QCD operator's share is the zero dressed Born above.
        evaluation._qcd_poles = lambda: {sector: (0.0, 0.0)}
        ((got_sector, (single, double)),) = evaluation._ir_poles().items()
        assert got_sector == sector
        floor = abs(row["born"]) * alpha / (2 * math.pi)
        for provider in fixture["validated_providers"]:
            values = row["providers"][provider]
            tolerance = (
                PRIMARY_TOLERANCE
                if provider == fixture["primary_provider"]
                else fixture["tolerance"]
            )
            for got, key in ((single, "single_pole"), (double, "double_pole")):
                error = _metric(got, values[key], floor)
                worst[provider] = max(worst.get(provider, 0.0), error)
                assert error <= tolerance, (provider, key, got, values[key])
    print(name, {provider: f"{error:.1e}" for provider, error in worst.items()})


@pytest.mark.loop_physics
@pytest.mark.parametrize(
    ("control", "pdgs"),
    (
        ("electroweak-mumu-control", (11, -11, 13, -13)),
        ("electroweak-mumu-scale-control", (11, -11, 13, -13)),
        ("electroweak-eeuu-control", (11, -11, 2, -2)),
    ),
)
def test_qed_operator_equals_the_madloop_gmu_controls(gmu, control, pdgs):
    """The renormalised virtual's poles of the MadLoop G_mu controls (full
    electroweak, Feynman gauge) at their points (``uudd_qcd.json``; the
    fourth e- e+ > u u~ row repeats the first point at mu_R = 1500 GeV) and
    scales. The double pole is -4 (alpha/2pi) Born (four legs of charge
    +-1 or the quarks' charges), the single pole carries the scale."""

    fixture = json.loads((FIXTURES / f"{control}.json").read_text())
    points = json.loads(CONTROL_POINTS.read_text())["points"]
    scales = fixture.get("mu_r2") or [
        r.get("mu_r2", "1000000") for r in fixture["rows"]
    ]
    (born_order,) = (_orders(o) for o in fixture["born_orders"].values())
    (sector,) = (_orders(o) for o in fixture["loop_orders"].values())
    for index, row in enumerate(fixture["rows"]):
        born = row["born"]["1"]
        full = row["full"]["1"]
        evaluation = _OperatorPoint(
            gmu,
            pdgs,
            points[index % len(points)],
            scales[index],
            CONTROL_PARAMETERS,
            {born_order: born},
            sector,
        )
        ((single, double),) = evaluation._ir_poles().values()
        floor = abs(born) * ALPHA_GMU / (2 * math.pi)
        assert _metric(single, full["single_pole"], floor) <= CONTROL_TOLERANCE
        assert _metric(double, full["double_pole"], floor) <= CONTROL_TOLERANCE


@pytest.mark.loop_physics
def test_alpha_is_the_coupling_of_the_photon_vertices(gmu, loop_sm_compiled_model):
    """alpha read from the photon couplings, no parameter name assumed: on
    the G_mu UFO alpha_Gmu = sqrt(2) G_F M_W^2 (1 - M_W^2/M_Z^2) / pi at the
    fixtures' inputs (their derived alpha, 32 digits) in binary64 and at 40
    digits; on loop_sm (alpha(M_Z) scheme) 1/aEWM1."""

    from pyamplicol.loops.v2 import reference_executor as rx
    from pyamplicol.models.loading import load_compiled_model

    fixture = json.loads((FIXTURES / "required/gg_ttx_ew.json").read_text())
    expected = Decimal(fixture["derived_parameters"]["alpha"])
    coupling = qed_coupling(gmu, _is_photon(gmu))
    assert coupling is not None
    assert sorted(name for _symbol, name in coupling[1]) == [
        "Gf",
        "GfRedefinitionChoice",
        "MW",
        "MZ",
    ]
    values = {n: Fraction(str(v[0])) for n, v in gmu.parameter_defaults.items()}
    values.update({k: Fraction(v) for k, v in fixture["parameters"].items()})
    double = coupling_value(coupling, values, rx.Carrier.f64())
    assert abs(double - float(expected)) <= 2e-16 * float(expected)
    with localcontext(prec=40):
        precise = coupling_value(coupling, values, rx.Carrier.ap(40))
        # The fixture's value has 30 significant digits.
        assert abs(precise - expected) <= Decimal("1e-29") * expected
    loop_sm = load_compiled_model(loop_sm_compiled_model)
    coupling = qed_coupling(loop_sm, _is_photon(loop_sm))
    values = {n: Fraction(str(v[0])) for n, v in loop_sm.parameter_defaults.items()}
    alpha = coupling_value(coupling, values, rx.Carrier.f64())
    assert abs(alpha - 1 / float(values["aEWM1"])) <= 2e-16 * alpha


# ------------------------------------------------------------- operator parts


def _poles(charges, momenta, masses, born=1.0, **options):
    return qed_catani_poles(
        charges,
        momenta,
        masses,
        born,
        gamma=options.pop("gamma", lambda j: qed_gamma(2, charges[j])),
        mu2=options.pop("mu2", 1.0),
        log=math.log,
        real=float,
        sqrt=math.sqrt,
        **options,
    )


Q = Fraction
# e- e+ > mu- mu+ (all incoming: the outgoing mu- enters as a mu+).
MOMENTA = {
    1: (Q(1), Q(0), Q(0), Q(1)),
    2: (Q(1), Q(0), Q(0), Q(-1)),
    3: (Q(-1), Q(0), Q(3, 5), Q(4, 5)),
    4: (Q(-1), Q(0), Q(-3, 5), Q(-4, 5)),
}
MASSLESS = dict.fromkeys(MOMENTA, Q(0))


def _dot(a, b) -> Fraction:
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


@pytest.mark.loop_fast
def test_qed_operator_parts():
    """Exact charges; the massless closed form -sum_j gamma_j + sum_{j != k}
    Q_j Q_k log(mu^2 / s_jk) (single) and -sum_j Q_j^2 (double); physical
    instead of all-incoming charges are refused; a massive pair from a
    neutral initial state (massive legs soft only: -2 Q^2 (1 + ln(rho)/v),
    no double pole); a massless charged vector has no gamma (no
    prediction)."""

    assert exact_charge(2 / 3) == Q(2, 3)
    assert exact_charge(-0.3333333333333333) == Q(-1, 3)
    assert exact_charge("-1") == -1
    with pytest.raises(ValueError, match="not a small rational"):
        exact_charge(math.pi)
    charges = {1: Q(-1), 2: Q(1), 3: Q(1), 4: Q(-1)}
    mu2 = 7.0
    single, double = _poles(charges, MOMENTA, MASSLESS, born=2.0, mu2=mu2)
    expected = -sum(float(qed_gamma(2, q)) for q in charges.values())
    for j in charges:
        for k in charges:
            if j != k:
                invariant = abs(2 * float(_dot(MOMENTA[j], MOMENTA[k])))
                expected += float(charges[j] * charges[k]) * math.log(mu2 / invariant)
    assert abs(single - 2.0 * expected) <= 1e-14 * abs(single)
    assert double == -2.0 * 4
    # u d~ > e+ ve with the physical charges of the outgoing legs.
    with pytest.raises(ValueError, match="do not add to zero"):
        _poles({1: Q(2, 3), 2: Q(1, 3), 3: Q(1)}, MOMENTA, MASSLESS)
    # g g > t t~: only the massive pair is charged (all incoming: t~ enters
    # as a t, the outgoing t~ as a t).
    pair = {3: Q(-2, 3), 4: Q(2, 3)}
    mass2 = Q(1, 10)
    single, double = _poles(pair, MOMENTA, {3: mass2, 4: mass2})
    dot = _dot(MOMENTA[3], MOMENTA[4])
    ratio = float(mass2 * mass2 / (dot * dot))
    v = math.sqrt(1 - ratio)
    log_rho = math.log(ratio / (1 + v) ** 2) / 2
    assert double == 0
    expected = -2 * float(pair[3] ** 2) * (1 + log_rho / v)
    assert abs(single - expected) <= 1e-14 * abs(expected)
    assert qed_gamma(1, Q(1)) == 2 and qed_gamma(2, Q(2, 3)) == Q(2, 3)
    assert qed_gamma(3, Q(1)) is None
    assert _poles(charges, MOMENTA, MASSLESS, gamma=lambda j: None) is None


@pytest.mark.loop_fast
def test_external_photon_conventions():
    """alpha(0) per on-shell photon (production, the user's decision [U9])
    adds nothing: delta Z_e^alpha(0) + 1/2 delta Z_AA has no light-fermion
    pole; with the loops' coupling each photon adds -gamma_gamma/eps Born,
    gamma_gamma = -2/3 sum_f N_c^f Q_f^2 (five light quarks and three
    leptons: -40/9). The engine passes its untagged photons under the
    loops' coupling (MS-PHOTON, ``test_untagged_external_photons_add_gamma_
    gamma``)."""

    assert EXTERNAL_PHOTON_CONVENTION == "alpha0"
    light = [(Q(2, 3), 3)] * 2 + [(Q(-1, 3), 3)] * 3 + [(Q(-1), 1)] * 3
    assert photon_gamma(light) == Q(-40, 9)
    # e- e+ > a a: the photons are neutral legs 3 and 4.
    charges = {1: Q(-1), 2: Q(1)}
    plain = _poles(charges, MOMENTA, MASSLESS, born=3.0)
    assert _poles(charges, MOMENTA, MASSLESS, born=3.0, photons=2) == plain
    single, double = _poles(
        charges,
        MOMENTA,
        MASSLESS,
        born=3.0,
        photons=2,
        gamma_photon=photon_gamma(light),
        convention="loop-coupling",
    )
    assert double == plain[1]
    assert abs(single - (plain[0] + 2 * 40 / 9 * 3.0)) <= 1e-14 * abs(single)
    with pytest.raises(ValueError, match="needs gamma_gamma"):
        _poles(charges, MOMENTA, MASSLESS, photons=1, convention="loop-coupling")
    with pytest.raises(ValueError, match="unknown external-photon convention"):
        _poles(charges, MOMENTA, MASSLESS, convention="alpha-mz")


@pytest.mark.loop_physics
def test_untagged_external_photons_add_gamma_gamma(gmu):
    """u u~ > a g at QCD + QED orders (MS-PHOTON, per leg): with the photon
    untagged (a slice whose ``external_photons`` does not list it) the
    engine's QED operator predicts the poles of the loops' coupling, the
    tagged (alpha(0)) prediction minus gamma_gamma/eps Born per photon,
    gamma_gamma = -2/3 sum_f N_c^f Q_f^2 over the massless charged fermions
    (the default card: five quarks and three leptons, -40/9); the joint
    request is predicted (no partial sum)."""

    from pyamplicol.loops.v2.ew_policy import ExternalPhotons

    point = [
        [1, 0, 0, 1],
        [1, 0, 0, -1],
        [1, 0, Q(3, 5), Q(4, 5)],
        [1, 0, Q(-3, 5), Q(-4, 5)],
    ]
    sector = (("QCD", 2), ("QED", 4))

    def evaluation(photons):
        found = _OperatorPoint(
            gmu,
            (2, -2, 22, 21),
            point,
            1,
            CONTROL_PARAMETERS,
            {(("QCD", 2), ("QED", 2)): 1.0},
            sector,
            corrections=("QCD", "QED"),
            photons=photons,
        )
        found._qcd_poles = lambda: {sector: (1.0, 1.0)}
        return found

    tagged = evaluation(ExternalPhotons((3,), "aEW0M1", "eCoup_DGmua0_UV_EW", "QED"))
    untagged = evaluation(None)
    alpha = untagged._qed_alpha()
    with_alpha0 = tagged._qed_poles()[sector]
    single, double = untagged._qed_poles()[sector]
    assert double == with_alpha0[1]
    shift = 40 / 9 * alpha / (2 * math.pi)
    assert abs(single - (with_alpha0[0] + shift)) <= 1e-14 * abs(single)
    assert untagged._ir_poles()[sector] == (1.0 + single, 1.0 + double)


@pytest.mark.loop_physics
def test_charged_massive_legs_must_be_stable_for_the_qed_operator(gmu):
    """Catani's operators describe stable emitters: with QED corrections an
    external W (colourless, charged) with a width that is not zero gives no
    prediction, as a coloured one does with QCD; QCD corrections do not
    dress it."""

    point = [[1, 0, 0, 1], [1, 0, 0, -1], [1, 0, Q(3, 5), Q(4, 5)]]
    point.append([1, 0, Q(-3, 5), Q(-4, 5)])
    sector = (("QCD", 2), ("QED", 4))

    def stable(corrections, width):
        return _OperatorPoint(
            gmu,
            (2, -1, 24, 21),
            point,
            1,
            {**CONTROL_PARAMETERS, "WW": width},
            {(("QCD", 2), ("QED", 2)): 1.0},
            sector,
            corrections=corrections,
        )._stable_coloured_legs()

    assert not stable(("QCD", "QED"), "2.085")
    assert stable(("QCD", "QED"), "0")
    assert stable(("QCD",), "2.085")


@pytest.mark.loop_fast
def test_per_correction_sectors_and_sums():
    """Each correction dresses the Born interferences whose squared order
    plus its two powers is the sector (cross orders included); a joint
    request's prediction is the sum of its corrections' operators; a
    correction without an operator, or an operator without a prediction,
    gives none (``_PointEvaluation._ir_poles``)."""

    qcd4, mixed, qed4 = (("QCD", 4),), (("QCD", 2), ("QED", 2)), (("QED", 4),)
    born = {qcd4: 1.0, mixed: 10.0, qed4: 100.0}
    upper = (("QCD", 2), ("QED", 4))
    lower = (("QCD", 4), ("QED", 2))
    assert dressed_born(born, upper, "QED", 0.0) == 10.0
    assert dressed_born(born, upper, "QCD", 0.0) == 100.0
    assert dressed_born(born, lower, "QED", 0.0) == 1.0
    assert dressed_born(born, lower, "QCD", 0.0) == 10.0
    assert dressed_born(born, (("QCD", 6),), "QED", 0.0) == 0.0
    assert (
        dressed_born({mixed: 10.0, (("QCD", 1), ("QED", 3)): 5.0}, upper, "QED", 0.0)
        == 10.0
    )

    one = {upper: (1.0, 2.0), lower: (3.0, 4.0)}
    other = {upper: (10.0, 20.0), lower: (30.0, 40.0)}
    assert sector_sum([one]) == one
    assert sector_sum([one, other]) == {upper: (11.0, 22.0), lower: (33.0, 44.0)}
    with pytest.raises(ValueError, match="different sectors"):
        sector_sum([one, {upper: (0.0, 0.0)}])

    class Point(_PointEvaluation):
        def __init__(self, corrections, qcd, qed):
            self.slice = SimpleNamespace(
                request=SimpleNamespace(corrections=corrections)
            )
            self._qcd, self._qed = qcd, qed

        def _correction_orders(self):
            return CorrectionOrders("QCD", "QED")

        def _predicted_sectors(self):
            return frozenset({upper, lower})

        def _qcd_poles(self):
            return self._qcd

        def _qed_poles(self):
            return self._qed

    assert Point(("QCD",), one, None)._ir_poles() == one
    assert Point(("QED",), None, other)._ir_poles() == other
    assert Point(("QCD", "QED"), one, other)._ir_poles() == sector_sum([one, other])
    assert Point(("QCD", "QED"), one, None)._ir_poles() is None
    assert Point(("QCD", "QED"), None, other)._ir_poles() is None
    assert Point(("QCD", "QNP"), one, other)._ir_poles() is None


# ---------------------------------------------------------------- engine


@pytest.mark.loop_nightly
def test_joint_request_predicts_the_poles_of_every_sector(
    loop_qcd_qed_sm_ufo, loop_native
):
    """g g > g g with QCD and QED corrections on loop_qcd_qed_sm (the joint
    request that generates today: no electroweak counterterm enters its
    trees, no width propagates in them): one Python point in binary64. Its
    QCD^6 sector's IR poles equal the prediction, the QCD operator alone
    (the joint request's prediction there was 0 before WP3.2 EWI-0b/EWI-6);
    the QCD^4 QED^2 sector (no one-loop amplitude of order QCD^2 QED^2) and
    its prediction are 0, and the QED operator adds nothing (no charged
    leg). Nightly: 2 min cold (generation 104 s at host load 200)."""

    from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
    from pyamplicol.models.loading import compile_model_source

    compiled = compile_model_source(loop_qcd_qed_sm_ufo, require_supported=False)
    request = SliceRequest(
        "g g > g g",
        {"QCD": 2, "QED": 0},
        ({"QCD": 4, "QED": 0}, {"QCD": 2, "QED": 2}),
        corrections=("QCD", "QED"),
        fixed_parameters={"MB": "0"},
        compatibility="gmu_particle_anti",
        scale_parameter="MU_R",
    )
    slice_ = LoopSlice.generate(compiled, request, loop_native)
    assert slice_.alpha_s_parameter == "aS"
    point = [[500, 0, 0, 500], [500, 0, 0, -500], [500, 300, 0, 400]]
    point.append([500, -300, 0, -400])
    result = slice_.evaluate(point, "1000000", parameters={"aS": "0.118", "MT": "173"})
    qcd = (("QCD", 6),)
    assert set(result.ir_poles) == {qcd, (("QCD", 4), ("QED", 2))}
    for sector, virtual in result.components["virtual"].items():
        single, double = (float(x) for x in result.ir_poles[sector])
        if sector != qcd:
            assert (single, double) == (0.0, 0.0)
            assert virtual.single_pole == virtual.double_pole == 0
            continue
        scale = max(abs(single), abs(double))
        assert scale > 0
        assert abs(complex(virtual.single_pole) - single) <= 1e-12 * scale
        assert abs(complex(virtual.double_pole) - double) <= 1e-12 * scale
