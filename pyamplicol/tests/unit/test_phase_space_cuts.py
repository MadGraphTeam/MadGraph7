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
from pyamplicol.generation.phase_space import (
    flat_final_states,
    massive_rambo_final_state,
)
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
    assert not PhaseSpaceCuts(pt_min=1e-9).accepted(beam, (0, 1, 2))[0]
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
    with pytest.raises(ConfigurationError, match="cut_attempts"):
        EvaluationPhaseSpaceConfig(cut_attempts=0)


def test_a_configuration_without_cuts_states_none_of_their_keys():
    """The configuration bytes a cut-free artifact stores (requested and
    effective; an append compares them) have no key of the cuts."""

    from pyamplicol.config import config_to_dict, resolve_config
    from pyamplicol.generation.artifact_writer import _toml_bytes

    run = resolve_config({"schema_version": 1, "action": "generate"})
    for config in (run.requested, run.effective):
        text = _toml_bytes(config_to_dict(config)).decode()
        table = text.split("[evaluation.phase_space]\n")[1].split("\n[")[0]
        assert table.split("\n") == ["points = 2", "seed = 1", ""]


def test_zero_is_no_cut(tmp_path):
    none = PhaseSpaceCuts(0.0, 0.0)
    assert not none.active and none.described() == []
    assert none.record() == {"pt_min": None, "delta_r_min": None}
    assert PhaseSpaceCuts(0.0, 0.7).described() == ["Delta R > 0.7"]
    artifact = _artifact(tmp_path / "s", "g g > g g g", [21] * 5, [0.0] * 3)
    draw = {"process": None, "points": 5, "seed": 1, "sqrt_s": 1000.0}
    assert draw_flat_points(artifact, cuts=none, **draw) == draw_flat_points(
        artifact, **draw
    )
    assert (
        draw_flat_points(artifact, cuts=PhaseSpaceCuts(0.0, 0.7), **draw).points
        == draw_flat_points(
            artifact, cuts=PhaseSpaceCuts(delta_r_min=0.7), **draw
        ).points
    )
    decay = _artifact(tmp_path / "d", "t > d e-", [6, 1, 11], [0.0, 0.0], sqrt_s=TOP)
    drawn = draw_flat_points(
        decay, process=None, points=1, seed=1, sqrt_s=None, cuts=none
    )
    assert len(drawn.points) == 1


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        (None, (25.0, 0.7, 5000)),
        (PhaseSpaceCuts(), (25.0, 0.7, 5000)),
        (PhaseSpaceCuts(pt_min=40.0), (40.0, 0.7, 5000)),
        (PhaseSpaceCuts(delta_r_min=1.0, attempts=9), (25.0, 1.0, 9)),
        (PhaseSpaceCuts(pt_min=0.0), (0.0, 0.7, 5000)),
        (PhaseSpaceCuts(0.0, 0.0), (0.0, 0.0, 5000)),
    ],
)
def test_check_overrides_the_recorded_cuts_one_by_one(given, expected, monkeypatch):
    """Each cut given to ``check`` replaces its own recorded one only; a 0
    switches it off (the cuts that reach the generation service's check)."""

    from types import SimpleNamespace

    from pyamplicol.api import services
    from pyamplicol.config import resolve_config

    seen = {}

    class Reached(Exception):
        pass

    def operation(_processes, **options):
        seen.update(options)
        raise Reached

    monkeypatch.setattr(
        services.Generator,
        "_implementation",
        lambda _self: SimpleNamespace(check=operation),
    )
    monkeypatch.setattr(
        services.Generator, "_resolve_generation_resources", lambda _self: None
    )
    run = resolve_config(
        {
            "schema_version": 1,
            "action": "check",
            "process": {"entries": [{"expression": "g g > g g", "name": "p"}]},
            "evaluation": {
                "phase_space": {
                    "pt_min": 25.0,
                    "delta_r_min": 0.7,
                    "cut_attempts": 5000,
                }
            },
        }
    )
    with pytest.raises(Reached):
        services.check(run, cuts=given)
    assert seen["cuts"] == PhaseSpaceCuts(*expected)


def test_check_options_left_out_or_zero_reach_the_check(tmp_path, monkeypatch):
    """The command line distinguishes a cut left out from one switched off."""

    from pyamplicol.api import services
    from pyamplicol.cli.handlers import DefaultCliServices
    from pyamplicol.reporting import CallbackProgressSink

    artifact = _artifact(tmp_path, "g g > g g g", [21] * 5, [0.0] * 3)
    seen = []
    monkeypatch.setattr(
        services, "check", lambda _target, **options: seen.append(options["cuts"])
    )
    for flags in ((), ("--pt-min", "0"), ("--delta-r-min", "1")):
        run = parse_cli(("check", str(artifact), *flags)).resolve()
        DefaultCliServices().check(
            run.effective, CallbackProgressSink(lambda _event: None)
        )
    assert seen == [
        PhaseSpaceCuts(),
        PhaseSpaceCuts(pt_min=0.0),
        PhaseSpaceCuts(delta_r_min=1.0),
    ]


def test_a_retried_point_is_a_function_of_its_own_seed(tmp_path):
    """Cuts most first attempts fail: every point, retried or not, is the one
    drawn alone, whatever the other points are, and no two are the same."""

    artifact = _artifact(tmp_path, "g g > g g g g", [21] * 6, [0.0] * 4)
    cuts = PhaseSpaceCuts(60.0, 1.2)
    arguments = {"process": None, "sqrt_s": 1000.0, "cuts": cuts}
    points = draw_flat_points(artifact, points=60, seed=1, **arguments).points
    free = draw_flat_points(
        artifact, points=60, seed=1, process=None, sqrt_s=1000.0
    ).points
    retried = [i for i in range(60) if points[i] != free[i]]
    assert len(retried) > 20
    assert len(set(points)) == 60
    for i in range(60):
        alone = draw_flat_points(artifact, points=1, seed=1 + i, **arguments)
        assert alone.points[0] == points[i]
    rest = draw_flat_points(artifact, points=40, seed=21, **arguments)
    assert rest.points == points[20:]


def test_retried_points_have_the_cut_distribution():
    """The accepted points are distributed as the uncut ones that pass
    (two-sample Kolmogorov-Smirnov distances, fixed seeds)."""

    cuts = PhaseSpaceCuts(40.0, 0.9)
    legs = range(4)

    def draw(seeds):
        return flat_final_states([0.0] * 4, sqrt_s=1000.0, seeds=seeds)

    def momenta(state):
        return state.momenta

    drawn = draw_accepted(list(range(4000)), draw, momenta, cuts, legs)
    points = np.array([state.momenta for state in drawn])
    uncut = np.array([s.momenta for s in draw(list(range(10**6, 10**6 + 24000)))])
    passing = uncut[cuts.accepted(uncut, legs)]

    def observables(sample):
        pair = sample[:, 0] + sample[:, 1]
        pt = np.hypot(sample[:, :, 1], sample[:, :, 2])
        return (
            pt.min(axis=1),
            pt.max(axis=1),
            sample[:, 0, 0],
            sample[:, 3, 3] / sample[:, 3, 0],
            pair[:, 0] ** 2 - (pair[:, 1:] ** 2).sum(axis=1),
        )

    first = np.array([state.momenta for state in draw(list(range(4000)))])
    again = ~cuts.accepted(first, legs)
    assert 1000 < again.sum() < 3000
    for sample in (points, points[again]):
        scale = math.sqrt(1 / len(sample) + 1 / len(passing))
        for x, y in zip(observables(sample), observables(passing), strict=True):
            grid = np.concatenate([x, y])
            x, y = np.sort(x), np.sort(y)
            distance = np.abs(
                np.searchsorted(x, grid, side="right") / len(x)
                - np.searchsorted(y, grid, side="right") / len(y)
            ).max()
            # 2.0: a chance of 2 exp(-2 x 2.0^2) = 7e-4 for one observable.
            assert distance < 2.0 * scale


def test_check_of_an_artifact_takes_every_option(tmp_path, monkeypatch):
    """``check ARTIFACT`` regenerates from the artifact's own configuration:
    the command's settings reach the check as arguments."""

    from pyamplicol.api import services
    from pyamplicol.cli.handlers import DefaultCliServices
    from pyamplicol.reporting import CallbackProgressSink

    artifact = _artifact(tmp_path, "g g > g g g", [21] * 5, [0.0] * 3)
    run = parse_cli(
        (
            *("check", str(artifact), "--points", "4", "--seed", "7"),
            *("--sqrt-s", "777", "--arbitrary-digits", "50"),
            *("--pt-min", "25", "--delta-r-min", "0.7"),
            *("--kinematics-improvement", "off"),
            "--kinematics-improvement-binary64",
        )
    ).resolve()
    seen = {}

    def check(target, **options):
        seen.update(options, target=target)

    monkeypatch.setattr(services, "check", check)
    DefaultCliServices().check(run.effective, CallbackProgressSink(lambda _event: None))
    assert seen.pop("target") == artifact
    expected = {
        "points": 4,
        "seed": 7,
        "precision": "auto",
        "sqrt_s": 777.0,
        "arbitrary_digits": 50,
        "cuts": PhaseSpaceCuts(25.0, 0.7),
        "kinematics_improvement": "off",
        "kinematics_improvement_binary64": True,
    }
    assert {key: seen[key] for key in expected} == expected


def test_check_points_are_cut_on_the_model_s_legs():
    """``check``'s own points (``loop_service._validation_momenta``): the cuts
    act on the massless charged or coloured final legs, a point that passes
    is the uncut one, decays and impossible cuts are refused."""

    from types import SimpleNamespace

    from pyamplicol.api.errors import GenerationError
    from pyamplicol.generation.loop_service import _validation_momenta

    masses = {6: TOP}
    particles = [
        SimpleNamespace(pdg_code=sign * pdg, spin=spin, color=color, charge=charge)
        for pdg, (spin, color, charge) in PARTICLES.items()
        for sign in (1, -1)
    ]
    context = SimpleNamespace(
        resolved=SimpleNamespace(
            model=SimpleNamespace(mass=lambda pdg: masses.get(abs(pdg), 0.0)),
            compiled=SimpleNamespace(ir=SimpleNamespace(particles=particles)),
        )
    )

    def entry(initial, final):
        return SimpleNamespace(
            process_ir=SimpleNamespace(
                process="p", initial_pdgs=initial, final_pdgs=final
            ),
            request=SimpleNamespace(name="p"),
        )

    process = entry((21, 21), (6, -6, 21, 21, 11, -12))
    arguments = {"seed": 5, "count": 60, "unavailable": "none", "sqrt_s": 1500.0}
    cuts = PhaseSpaceCuts(60.0, 1.0)
    points = _validation_momenta(context, process, cuts=cuts, **arguments)
    free = _validation_momenta(context, process, **arguments)
    assert (
        _validation_momenta(context, process, cuts=PhaseSpaceCuts(), **arguments)
        == free
    )
    kept = 0
    soft = 0
    for point, uncut in zip(points, free, strict=True):
        legs = [point[i] for i in (4, 5, 6)]
        assert all(_pt(p) > 60.0 for p in legs)
        assert all(
            _distance(legs[i], legs[j]) > 1.0 for i in range(3) for j in range(i)
        )
        soft += min(_pt(point[i]) for i in (2, 3, 7)) <= 60.0
        kept += point == uncut
    assert 0 < kept < 60 and soft
    with pytest.raises(GenerationError, match="cannot be met"):
        _validation_momenta(
            context, process, cuts=PhaseSpaceCuts(pt_min=600.0), **arguments
        )
    with pytest.raises(GenerationError, match="decay"):
        _validation_momenta(
            context, entry((6,), (1, 11)), cuts=cuts, **{**arguments, "sqrt_s": None}
        )


def test_the_runtime_setup_names_the_cuts():
    from pyamplicol.loop_profile import _runtime_setup

    cuts = PhaseSpaceCuts(25.0, 0.7)
    assert "(seed 3, pT > 25, Delta R > 0.7)" in _runtime_setup(False, 3, cuts=cuts)
    plain = _runtime_setup(False, 3)
    assert "(seed 3)" in plain
    assert _runtime_setup(False, 3, cuts=PhaseSpaceCuts()) == plain


@pytest.mark.parametrize("estimate_only", [False, True])
def test_the_stability_tables_name_the_cuts(estimate_only):
    """In the percentile table's title, and nowhere else (no line added)."""

    from types import SimpleNamespace

    from pyamplicol.reporting import loop_profile

    prettytable = pytest.importorskip("prettytable")

    def tables(cuts):
        setup = {
            "target": 1e-5,
            "points": 64,
            "sqrt_s": 1000.0,
            "seed": 3,
            "ground_truth": {},
        }
        report = SimpleNamespace(
            stability=(
                SimpleNamespace(
                    backend="ninjars",
                    refusal="refused",
                    points=0,
                    percentiles=(),
                    estimate_only=estimate_only,
                ),
            ),
            stability_setup={**setup, **cuts},
            ground_truth=None,
            default_backend="ninjars",
            failure_causes=(),
        )
        return loop_profile._stability_tables(report, prettytable, False)

    def text(blocks):
        return " ".join("\n".join(blocks).replace("|", " ").split())

    cut = text(tables({"cuts": {"pt_min": 25.0, "delta_r_min": 0.7, "legs": [2, 3]}}))
    plain = tables({"cuts": {"pt_min": None, "delta_r_min": None, "legs": []}})
    assert plain == tables({})
    assert "seed 3)" in text(plain)
    assert "seed 3, pT > 25, Delta R > 0.7)" in cut
    assert cut.count("Delta R") == 1
