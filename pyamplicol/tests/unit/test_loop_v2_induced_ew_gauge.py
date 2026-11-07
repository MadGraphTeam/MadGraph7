# SPDX-License-Identifier: 0BSD
"""Gauge checks of loop-induced amplitudes with massive external vectors
(M-LIT LIT-EW, m-lit PLAN §2.a.7 "massive external vectors", §4 L4 /
LI-W1; RULED Q7, Q12):

* the Goldstone-equivalence probe (``loops.v2.goldstone``) in
  't Hooft-Feynman gauge: k.A(V) = eta M_V A(G) per (loop order, helicity
  row) on the amplitude basis, within ten times the two sides' own
  estimates, with one phase eta of unit modulus over every row. On the
  complete [QED] a a > z z it holds with eta = -i, the convention the
  tree-level Slavnov-Taylor identities of ``test_loop_v2_ew_gauge.py`` fix
  for an outgoing Z, while the Goldstone side is far from vanishing; a
  flipped G0 t t~ coupling breaks it; on h > z a both sides vanish (the
  photon's transversality); the generation-time probes run it on one leg
  per massive species at zero widths and record it;
* the both-gauges equality of the QCD-only g g > h z and g g > w+ w- at
  zero widths (their trees take unitary gauge; s-channel Z* / G0* with the
  t-b mass splitting differ by gauge term by term): unitary = 't Hooft-
  Feynman gauge, and both equal MadLoop.
"""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_physics

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop" / "required" / "li"
LAX = 1e-5
# Unitary = Feynman gauge: one amplitude computed through different
# families (G0/G+- exchange vs the k k / M^2 term); binary64 rounding of the
# triangle-box cancellations only.
GAUGES_DOUBLE = 1e-9
EW_OPTIONS = {
    "compatibility": "gmu_particle_anti",
    "scale_parameter": "MU_R",
    "external_photons": "none",
}


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _process(fixture: dict) -> str:
    text = fixture["madloop_process"].split("[")[0]
    return " ".join(token for token in text.split() if "=" not in token)


@pytest.fixture(scope="module")
def gmu(loop_gmu_ufo, tmp_path_factory):
    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(
        loop_gmu_ufo,
        require_supported=False,
        cache_dir=tmp_path_factory.mktemp("li-ew-gauge") / "model-cache",
    )


def _ew_request(compiled, name, **fields):
    """The fixture's [QED] loop-induced request (zero widths)."""

    from pyamplicol.loops.v2.assemble import SliceRequest

    fixture = _fixture(name)
    orders = dict(fixture["loop_orders"])
    fixed = {
        key: value
        for key, value in fixture["scheme_data"]["fixed_parameters"].items()
        if key in compiled.parameter_defaults
    }
    return SliceRequest(
        _process(fixture),
        orders,
        orders,
        contribution="loop-induced",
        corrections=("QED",),
        fixed_parameters=fixed,
        **{**EW_OPTIONS, **fields},
    )


def _point(fixture: dict, index: int = 0):
    return fixture["points"][index], Fraction(fixture["mu_r"][index]) ** 2


class _Probes:
    """The Goldstone probe slices of a fixture row (the vector's Ward slice
    and the Goldstone's, both on the full tree kernels), generated once."""

    def __init__(self, compiled, native):
        self.compiled, self.native = compiled, native
        self.found: dict = {}

    def sides(self, name: str):
        found = self.found.get(name)
        if found is None:
            from pyamplicol.loops.v2 import goldstone
            from pyamplicol.loops.v2.plan_export import NativeExecutor
            from pyamplicol.loops.v2.ward import ward_slice

            request = _ew_request(self.compiled, name)
            vector = ward_slice(self.compiled, request, self.native)
            ((leg, partner),) = goldstone.massive_vector_legs(vector).items()
            side = goldstone.goldstone_slice(
                self.compiled, request, self.native, vector, leg, partner
            )
            found = self.found[name] = (
                leg,
                vector,
                side,
                NativeExecutor(vector, forward=False, inputs="native"),
                NativeExecutor(side, forward=False, inputs="native"),
            )
        return found

    def probe(self, name: str, *, precision=None, index: int = 0):
        """The probe of the fixture's massive vector leg at point ``index``."""

        from pyamplicol.loops.v2 import goldstone
        from pyamplicol.loops.v2.assemble import Precision

        leg, vector, _side, executor, partner = self.sides(name)
        fixture = _fixture(name)
        point, scale = _point(fixture, index)
        external = vector._external_parameters()
        parameters = {
            key: value
            for key, value in fixture["scheme_data"]["parameters"].items()
            if key in external and key not in vector.fixed_parameters
        }
        return goldstone.probe(
            executor,
            partner,
            point,
            scale,
            leg,
            parameters=parameters,
            precision=precision or Precision.double(),
        )


@pytest.fixture(scope="module")
def probes(gmu, loop_native):
    return _Probes(gmu, loop_native)


# --------------------------------------------------- Goldstone equivalence


def test_goldstone_partners_are_structural(gmu):
    """Each massive vector's Goldstone: the scalar flagged goldstoneboson of
    its charge, colour and mass parameter (Z -> G0, W+- -> G+-), by the
    model's records, never by name."""

    from pyamplicol.loops.v2.goldstone import goldstone_partners

    assert goldstone_partners(gmu) == {23: 250, 24: 251, -24: -251}


def test_goldstone_equivalence_holds_on_a_a_to_z_z(probes):
    """a a > z z [QED] in 't Hooft-Feynman gauge at zero widths: with
    epsilon_3 -> p_3 / E_3 the loop amplitude equals eta (M_Z / E_3) times
    a a > G0 z per row within ten times the estimates, eta = -i (the
    tree-level convention of an outgoing Z in this model), in binary64 and
    double-double; the Goldstone side is far above its estimate (the
    identity is not vacuous here)."""

    from pyamplicol.loops.v2.assemble import Precision

    leg, vector, side, _executor, _partner = probes.sides("li_aa_zz")
    assert str(vector.compiled.ir.gauge) == "feynman"
    assert str(side.request.process) == "A A > G0 Z" and leg == 3
    for precision in (Precision.double(), Precision.double_double()):
        result = probes.probe("li_aa_zz", precision=precision)
        assert result.leg == 3 and result.passed, result.ratios
        assert result.goldstone_ratio > 1e6
        re, im = result.phase
        assert abs(re) <= 1e-9 and abs(im + 1) <= 1e-9


def test_goldstone_probe_discriminates(probes, monkeypatch):
    """The probe fails by far when either side of the identity is wrong: a
    non-gauge vector in place of p_3 / E_3 on the vector side (the Ward
    probe's control), or a Goldstone side off by 1 % in modulus (the fitted
    phase cannot absorb it: eta has unit modulus)."""

    from pyamplicol.loops.v2 import goldstone, ward

    control = (("1", "0"), ("0", "0"), ("1", "0"), ("0", "0"))
    with monkeypatch.context() as patch:
        patch.setattr(ward, "gauge_ward", lambda leg, factor, point: (leg, control))
        result = probes.probe("li_aa_zz")
    assert not result.passed and result.worst > 1e6
    exact = goldstone._mass_over_energy
    with monkeypatch.context() as patch:
        patch.setattr(
            goldstone,
            "_mass_over_energy",
            lambda point, leg, digits: exact(point, leg, digits) * 101 / 100,
        )
        result = probes.probe("li_aa_zz")
    assert not result.passed and result.worst > 1e6


def test_goldstone_equivalence_on_h_to_z_a_vanishes_on_both_sides(probes):
    """h > z a: k_Z . A(h -> Z gamma) vanishes by the photon's
    transversality, and so does h > G0 a: both sides are zero within their
    estimates, the probe passes with no phase to read."""

    result = probes.probe("li_h_za")
    assert result.passed and result.phase is None
    assert result.goldstone_ratio <= 10


def test_generation_probes_run_the_goldstone_probe(gmu, loop_native):
    """m-lit Q12 / Q7: the generation-time probes of a a > z z [QED] at the
    model's default widths zero the widths (outside the complex-mass
    scheme), run the finiteness probe, the Ward probe of the photon species
    and the Goldstone-equivalence probe of the Z species, all within ten
    times their estimates in double-double, and record the Goldstone phase."""

    from pyamplicol.loops.v2 import induced
    from pyamplicol.loops.v2.assemble import LoopSlice

    request = _ew_request(gmu, "li_aa_zz")
    fixed = {
        key: value
        for key, value in request.fixed_parameters.items()
        if not key.startswith("W")
    }
    slice_ = LoopSlice.generate(
        gmu, replace(request, fixed_parameters=fixed), loop_native
    )
    candidates = [
        [[float(x) for x in p] for p in point]
        for point in _fixture("li_aa_zz")["points"]
    ]
    record = induced.generation_probes(slice_, gmu, loop_native, candidates)
    assert record["widths"] == "zero"
    assert {"WZ", "WW", "WT", "WH"} <= set(record["zeroed_widths"])
    assert record["finiteness"]["passed"] and record["ward"]["passed"]
    assert record["ward"]["legs"] == [1]
    goldstone = record["goldstone"]
    assert goldstone["legs"] == [3] and goldstone["passed"]
    re, im = goldstone["phases"]["3"]
    assert abs(re) <= 1e-9 and abs(im + 1) <= 1e-9
    assert "failed" not in record


# ------------------------------------------------------------ both gauges


@pytest.fixture(scope="module")
def loop_sm_no_b_mass(loop_sm_ufo, tmp_path_factory):
    from tools.loop_benchmark.mqcd import compile_restriction

    return compile_restriction(
        loop_sm_ufo, "no_b_mass", tmp_path_factory.mktemp("li-ew-sm") / "no_b_mass"
    )


@pytest.mark.parametrize("name", ["li_gg_hz", "li_gg_wpwm"])
def test_qcd_induced_massive_vectors_agree_in_both_gauges(
    loop_sm_no_b_mass, loop_native, name
):
    """LI-25 / LI-22 (m-lit PLAN §2.a.7): the QCD-only g g > h z and
    g g > w+ w- keep unitary-gauge trees; at zero widths (the fixture's
    scheme; fixed tree widths would differ between the gauges at
    O(M Gamma / s)) their |A|^2 in unitary and in 't Hooft-Feynman gauge
    (G0 / G+- exchange instead of the k k / M^2 term) agree in binary64 at
    every point within 1e-9, and both equal MadLoop within the lax rule."""

    from pyamplicol.loops.v2.assemble import LoopSlice, Precision, SliceRequest
    from pyamplicol.loops.v2.plan_export import NativeExecutor
    from pyamplicol.models.loading import with_model_gauge

    fixture = _fixture(name)
    orders = dict(fixture["loop_orders"])
    totals = {}
    for gauge, compiled in (
        ("unitary", loop_sm_no_b_mass),
        ("feynman", with_model_gauge(loop_sm_no_b_mass, "feynman")),
    ):
        fixed = {
            key: value
            for key, value in fixture["scheme_data"]["fixed_parameters"].items()
            if key in compiled.parameter_defaults
        }
        slice_ = LoopSlice.generate(
            compiled,
            SliceRequest(
                _process(fixture),
                orders,
                orders,
                contribution="loop-induced",
                fixed_parameters=fixed,
            ),
            loop_native,
        )
        assert str(slice_.compiled.ir.gauge) == gauge
        external = slice_._external_parameters()
        parameters = {
            key: value
            for key, value in fixture["scheme_data"]["parameters"].items()
            if key in external and key not in slice_.fixed_parameters
        }
        results = NativeExecutor(slice_, forward=False, inputs="native").evaluate(
            fixture["points"],
            [Fraction(mu) ** 2 for mu in fixture["mu_r"]],
            parameters=parameters,
            precision=Precision.double(),
        )
        totals[gauge] = [float(result.totals[1]) for result in results]
    for index, (unitary, feynman) in enumerate(
        zip(totals["unitary"], totals["feynman"], strict=True)
    ):
        assert abs(feynman / unitary - 1) <= GAUGES_DOUBLE, index
        madloop = fixture["rows"][index]["providers"]["madloop"]["value"]
        assert abs(unitary / madloop - 1) <= LAX, index
