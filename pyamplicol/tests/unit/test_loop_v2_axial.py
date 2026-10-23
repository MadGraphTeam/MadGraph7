# SPDX-License-Identifier: 0BSD
"""Chiral loop lines and closed-fermion certified zeros (WP3.1 I-AX).

The tree reading's fermion sign (``conventions``) holds for every Lorentz
structure, so the Z's chiral couplings run on loop quark lines:

* Drell-Yan with the photon and the Z: the virtual over the Born is the
  universal massless quark form factor, as for the photon alone (an open
  quark line through the loop with the Z vertex on it);
* u u~ > e- e+ g g with the photon and the Z against MadLoop's instrumented
  components (``dilepton-two-gluon.json``): the bare loops include closed
  quark loops with the Z's axial coupling (reversing the top's moves the
  finite part by 6e-4 to 7e-3), and the counterterms include the
  Levi-Civita R2 rules (``test_loop_v2_levi_civita``).

Closed quark loops with one photon and two gluons need no counterterm rule:
the generator certifies them (Furry), as it certifies a closed loop with one
gluon (traceless colour); a Z keeps its axial part, which R2_GGZ covers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pyamplicol.loops.v2.assemble import LoopSlice, SliceRequest
from pyamplicol.loops.v2.ct_sites import Signature, closed_fermion_zeros
from pyamplicol.loops.v2.families import BlockCurrent
from tools.loop_reference import legacy_capture

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "loop"
C_F = 4 / 3


def _laurent(result, component: str) -> list[float]:
    total = [0.0, 0.0, 0.0]
    for laurent in result.components[component].values():
        total = [a + float(b) for a, b in zip(total, laurent.coefficients, strict=True)]
    return total


@pytest.mark.loop_physics
def test_drell_yan_with_the_z_is_the_massless_form_factor(
    loop_sm_compiled_model, loop_native
):
    """V/B = (alpha_s/2pi) C_F (-2/eps^2 - (3 + 2L)/eps - 8 + pi^2 - 3L - L^2),
    L = log(mu^2/s), with the chiral Z vertex on the loop's quark line."""

    from pyamplicol.models.loading import load_compiled_model

    control = legacy_capture.CONTROLS["dy"]
    fixture = json.loads((FIXTURES / "legacy" / "dy.json").read_text())
    slice_ = LoopSlice.generate(
        load_compiled_model(loop_sm_compiled_model),
        SliceRequest(
            control.process.removesuffix(" / z"),
            control.born_order_bounds,
            control.loop_order_bounds,
            fixed_parameters=legacy_capture.FIXED_PARAMETERS,
        ),
        loop_native,
    )
    a = 0.118 / (2 * math.pi) * C_F
    for point, scale in zip(fixture["points"], fixture["scales_squared"], strict=True):
        result = slice_.evaluate(
            point, scale, parameters={"aS": "0.118", "aEWM1": "132.507"}
        )
        s = (float(point[0][0]) + float(point[1][0])) ** 2
        log = math.log(float(scale) / s)
        expected = [
            a * (math.pi**2 - 8 - 3 * log - log**2),
            a * (-3 - 2 * log),
            -2 * a,
        ]
        born = sum(float(x) for x in result.born.values())
        ratio = [x / born for x in _laurent(result, "virtual")]
        for mine, theirs in zip(ratio, expected, strict=True):
            assert abs(mine - theirs) <= 1e-12 * max(map(abs, expected)), (
                ratio,
                expected,
            )


@pytest.mark.loop_physics
def test_dilepton_two_gluons_with_the_z_equals_madloop(
    loop_sm_compiled_model, loop_native
):
    """Born, bare, UV, R2 and the virtual against MadLoop's components."""

    from pyamplicol.models.loading import load_compiled_model

    fixture = json.loads((FIXTURES / "dilepton-two-gluon.json").read_text())
    born, loop = {"QCD": 2, "QED": 2}, {"QCD": 4, "QED": 2}
    slice_ = LoopSlice.generate(
        load_compiled_model(loop_sm_compiled_model),
        SliceRequest(
            "u u~ > e- e+ g g",
            born,
            loop,
            fixed_parameters=fixture["fixed_parameters"],
        ),
        loop_native,
    )
    certified = {c.certificate for c in slice_.sites.coverage if c.certificate}
    assert any(text.startswith("Furry") for text in certified), certified
    parameters = {k: v for k, v in fixture["parameters"].items() if k != "MU_R"}
    for index in range(len(fixture["points"])):
        result = slice_.evaluate(
            fixture["points"][index],
            str(int(fixture["mu_r"][index]) ** 2),
            parameters=parameters,
        )
        row = fixture["rows"][index]
        mine_born = sum(float(x) for x in result.born.values())
        assert abs(mine_born - row["born"]) <= 1e-12 * abs(row["born"])
        floor = abs(row["born"]) * 0.118 / (2 * math.pi)
        for mine, theirs in (
            ("bare", "bare"),
            ("uv", "uv"),
            ("r2", "r2"),
            ("virtual", "full"),
        ):
            reference = row["components"][theirs]
            values = _laurent(result, mine)
            for value, key in zip(
                values, ("finite", "single_pole", "double_pole"), strict=True
            ):
                scale = max(abs(value), abs(reference[key]), floor)
                assert abs(value - reference[key]) <= 1e-10 * scale, (
                    index,
                    mine,
                    key,
                    value,
                    reference[key],
                )


def _model(z_structures: tuple[str, ...]):
    spinors = "idx(1,1),idx(1,2)"
    lorentz = {
        "FFV1": f"Gamma(idx(1,3),{spinors})",
        "FFV2": "Gamma(idx(1,3),idx(1,2),dummy(1))*ProjM(dummy(1),idx(1,1))",
    }

    def particle(name, anti, pdg, spin, colour):
        return {
            "name": name,
            "antiname": anti,
            "pdg_code": pdg,
            "spin": spin,
            "color": colour,
            "mass": "ZERO",
        }

    return {
        "particles": [
            particle("u", "u~", 2, 2, 3),
            particle("u~", "u", -2, 2, -3),
            particle("g", "g", 21, 3, 8),
            particle("a", "a", 22, 3, 1),
            particle("z", "z", 23, 3, 1),
        ],
        "lorentz_structures": [
            {"name": name, "structure": structure}
            for name, structure in lorentz.items()
        ],
        "vertex_rules": [
            {
                "name": "QQG",
                "particles": ["u~", "u", "g"],
                "lorentz_structures": ["FFV1"],
            },
            {
                "name": "QQA",
                "particles": ["u~", "u", "a"],
                "lorentz_structures": ["FFV1"],
            },
            {
                "name": "QQZ",
                "particles": ["u~", "u", "z"],
                "lorentz_structures": list(z_structures),
            },
        ],
    }


def _signature(*fields: str) -> Signature:
    blocks = tuple(
        BlockCurrent(1 << index, field, ()) for index, field in enumerate(fields)
    )
    return Signature(tuple(sorted(blocks)), ())


@pytest.mark.loop_fast
def test_closed_fermion_zeros():
    """Furry: an odd number of vector couplings with a reversal-symmetric
    colour trace; traceless colour: one gluon. A chiral vertex, an even
    count, three gluons or a quark-gluon loop are not certified."""

    witnesses: dict[Signature, set] = {}
    for fields, species in (
        (("a", "g", "g"), (2,)),
        (("a", "g"), (2,)),
        (("a", "a", "a"), (2,)),
        (("z", "g", "g"), (2,)),
        (("a", "a", "g", "g"), (2,)),
        (("g", "g", "g"), (2,)),
        (("a", "g", "g"), (2, 21)),
    ):
        witnesses.setdefault(_signature(*fields), set()).add(species)
    found = {
        (zero.signature.fields, zero.species): zero.certificate.split(":")[0]
        for zero in closed_fermion_zeros(witnesses, _model(("FFV2",)))
    }
    assert found == {
        (("a", "g", "g"), (2,)): "Furry",
        (("a", "a", "a"), (2,)): "Furry",
        (("a", "g"), (2,)): "colour",
    }
    # A pure vector Z coupling is certified like the photon's.
    vector = closed_fermion_zeros(witnesses, _model(("FFV1",)))
    assert (("g", "g", "z"), (2,)) in {(z.signature.fields, z.species) for z in vector}
