# SPDX-License-Identifier: 0BSD
"""The cuts of the drawn phase-space points (``[evaluation.phase_space]
pt_min`` / ``delta_r_min``; ``generation.phase_space_cuts``)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from pyamplicol.cli.flat_points import draw_flat_points
from pyamplicol.cli.parser import parse_cli
from pyamplicol.config import ConfigurationError
from pyamplicol.config.models import EvaluationPhaseSpaceConfig
from pyamplicol.generation.phase_space import massive_rambo_final_state
from pyamplicol.generation.phase_space_cuts import (
    CutsError,
    PhaseSpaceCuts,
    attempt_seed,
    draw_accepted,
    is_cut_particle,
)

# spin (2 s + 1), colour representation, charge.
PARTICLES = {
    21: (3, 8, 0.0),
    1: (2, 3, -1 / 3),
    6: (2, 3, 2 / 3),
    11: (2, 1, -1.0),
    12: (2, 1, 0.0),
    22: (3, 1, 0.0),
    25: (1, 1, 0.0),
}
TOP = 173.0


def _artifact(root, expression, pdgs, masses, sqrt_s=1000.0):
    final = massive_rambo_final_state(len(masses), sqrt_s=sqrt_s, masses=masses, seed=1)
    beam = sqrt_s / 2
    momenta = [(beam, 0, 0, beam), (beam, 0, 0, -beam), *final]
    if len(expression.split(">")[0].split()) == 1:
        momenta = [(sqrt_s, 0, 0, 0), *final]
    process = root / "processes" / "p"
    process.mkdir(parents=True)
    (root / "model").mkdir()
    (root / "artifact.json").write_text(
        json.dumps({"processes": [{"id": "p", "expression": expression}]})
    )
    (process / "validation-momenta.json").write_text(
        json.dumps(
            {
                "available": True,
                "points": [
                    [
                        {"pdg": pdg, "momentum": [repr(float(x)) for x in leg]}
                        for pdg, leg in zip(pdgs, momenta, strict=True)
                    ]
                ],
            }
        )
    )
    (root / "model" / "compiled-model.json").write_text(
        json.dumps(
            {
                "ir": {
                    "particles": [
                        {
                            "pdg_code": sign * pdg,
                            "spin": spin,
                            "color": color,
                            "charge": charge,
                        }
                        for pdg, (spin, color, charge) in PARTICLES.items()
                        for sign in (1, -1)
                    ]
                }
            }
        )
    )
    return root


def _pt(p):
    return math.hypot(p[1], p[2])


def _distance(p, q):
    dphi = abs(math.atan2(p[2], p[1]) - math.atan2(q[2], q[1]))
    dphi = min(dphi, 2 * math.pi - dphi)
    eta = [0.5 * math.log((_mag(x) + x[3]) / (_mag(x) - x[3])) for x in (p, q)]
    return math.hypot(eta[0] - eta[1], dphi)


def _mag(p):
    return math.sqrt(p[1] ** 2 + p[2] ** 2 + p[3] ** 2)


def test_which_particles_are_cut():
    cut = {
        pdg: is_cut_particle(spin=s, color=c, charge=q)
        for pdg, (s, c, q) in PARTICLES.items()
    }
    assert cut == {21: True, 1: True, 6: True, 11: True, 12: False, 22: True, 25: False}


def test_cut_points(tmp_path):
    """Every point passes on its massless charged or coloured legs only; a
    point depends on its own seed alone and is the uncut one if that
    passes."""

    artifact = _artifact(
        tmp_path / "a",
        "g g > t t~ g g e- ve",
        [21, 21, 6, -6, 21, 21, 11, -12],
        [TOP, TOP, 0.0, 0.0, 0.0, 0.0],
    )
    cuts = PhaseSpaceCuts(25.0, 0.7)
    arguments = {"process": None, "sqrt_s": 1000.0}
    flat = draw_flat_points(artifact, points=200, seed=5, cuts=cuts, **arguments)
    assert flat.cut_legs == (4, 5, 6)
    free = draw_flat_points(artifact, points=200, seed=5, **arguments)
    assert free.cut_legs == ()
    kept = 0
    soft = 0
    for point, uncut in zip(flat.points, free.points, strict=True):
        legs = [point[i] for i in flat.cut_legs]
        assert all(_pt(p) > 25.0 for p in legs)
        assert all(
            _distance(legs[i], legs[j]) > 0.7 for i in range(3) for j in range(i)
        )
        soft += min(_pt(point[i]) for i in (2, 3, 7)) <= 25.0
        old = [uncut[i] for i in flat.cut_legs]
        passes = all(_pt(p) > 25.0 for p in old) and all(
            _distance(old[i], old[j]) > 0.7 for i in range(3) for j in range(i)
        )
        assert (point == uncut) is passes
        kept += passes
        total = np.sum(point[2:], axis=0)
        assert total == pytest.approx([1000.0, 0, 0, 0], abs=1e-9)
    assert 0 < kept < 200
    assert soft  # the tops and the neutrino are not cut
    one = draw_flat_points(artifact, points=1, seed=5 + 137, cuts=cuts, **arguments)
    assert one.points[0] == flat.points[137]


def test_each_cut_alone(tmp_path):
    artifact = _artifact(tmp_path, "g g > g g g", [21] * 5, [0.0] * 3)
    draw = {"process": None, "points": 100, "seed": 1, "sqrt_s": 1000.0}
    for cuts in (PhaseSpaceCuts(pt_min=100.0), PhaseSpaceCuts(delta_r_min=1.0)):
        points = draw_flat_points(artifact, cuts=cuts, **draw).points
        pt = min(_pt(p) for point in points for p in point[2:])
        distance = min(
            _distance(point[i], point[j])
            for point in points
            for i in range(2, 5)
            for j in range(2, i)
        )
        assert (pt > 100.0) is (cuts.pt_min is not None)
        assert (distance > 1.0) is (cuts.delta_r_min is not None)


def test_impossible_cuts_are_refused(tmp_path):
    artifact = _artifact(tmp_path / "s", "g g > g g g", [21] * 5, [0.0] * 3)
    draw = {"process": None, "points": 1, "seed": 1, "sqrt_s": 1000.0}
    with pytest.raises(ConfigurationError, match="cannot be met"):
        draw_flat_points(artifact, cuts=PhaseSpaceCuts(pt_min=400.0), **draw)
    with pytest.raises(ConfigurationError, match="cut_attempts"):
        draw_flat_points(
            artifact, cuts=PhaseSpaceCuts(delta_r_min=50.0, attempts=20), **draw
        )
    decay = _artifact(tmp_path / "d", "t > d e-", [6, 1, 11], [0.0, 0.0], sqrt_s=TOP)
    with pytest.raises(ConfigurationError, match="decay"):
        draw_flat_points(
            decay,
            process=None,
            points=1,
            seed=1,
            sqrt_s=None,
            cuts=PhaseSpaceCuts(pt_min=1.0),
        )


def test_a_leg_along_the_beam_fails_and_attempts_are_counted():
    beam = [[(1.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, -1.0), (2.0, 2.0, 0.0, 0.0)]]
    assert not PhaseSpaceCuts(pt_min=0.0).accepted(beam, (0, 1, 2))[0]
    assert attempt_seed(3, 0) == 3
    assert len({attempt_seed(3, k) for k in range(50)}) == 50
    with pytest.raises(CutsError):
        draw_accepted(
            [1, 2],
            lambda s: [beam[0]] * len(s),
            lambda p: p,
            PhaseSpaceCuts(pt_min=1.0, attempts=3),
            (0, 1),
        )


def test_settings_and_flags(tmp_path):
    assert not EvaluationPhaseSpaceConfig().cuts.active
    for name, value in (("pt_min", -1.0), ("delta_r_min", float("inf"))):
        with pytest.raises(ConfigurationError, match=name):
            EvaluationPhaseSpaceConfig(**{name: value})
    card = tmp_path / "card.toml"
    card.write_text('schema_version = 1\naction = "generate"\n')
    for command in ("evaluate", "profile", "check"):
        run = parse_cli(
            (command, str(tmp_path), "--pt-min", "25", "--delta-r-min", "0.7")
        ).resolve()
        space = run.effective.evaluation.phase_space
        assert (space.pt_min, space.delta_r_min) == (25.0, 0.7)
        assert space.cuts == PhaseSpaceCuts(25.0, 0.7)
