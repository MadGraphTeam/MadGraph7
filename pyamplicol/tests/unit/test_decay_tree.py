# SPDX-License-Identifier: 0BSD
"""Tree-level 1 -> n decays (M-FINAL MF-DEC-1).

A process has one initial particle (a decay, MG5_aMC's ``t > b w+``) or two;
decay chains and ``$`` stay refused (``madloop-syntax``); the one-loop decay
is refused at the loop admission (``decay-process``, until MF-DEC-2). Tree
decays equal MG5_aMC 3.7.0 standalone (``tools/loop_reference/decay_tree_mg5.py``,
fixtures ``tests/fixtures/loop/required/dec/tree_*.json``) at three exact
rest-frame points with the same parameter card, and stay invariant under the
stability rotations (spatial: the rest frame is kept) and under boosts.
"""

from __future__ import annotations

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DEC = ROOT / "tests/fixtures/loop/required/dec"
REFERENCE = ROOT / "tools/loop_reference"


def _toolkit():
    if str(REFERENCE) not in sys.path:
        sys.path.insert(0, str(REFERENCE))
    import decay_points

    return decay_points


# ------------------------------------------------------------ process model


@pytest.mark.parametrize(
    "expression", ["t > b w+", "z > e+ e- g", "h > b b~", "2*g > g g g", "w+ > j j"]
)
def test_decays_are_processes(expression):
    from pyamplicol.api import ProcessRequest

    assert ProcessRequest.parse(expression).expression == expression


@pytest.mark.parametrize(
    ("expression", "found"),
    [
        ("t > b w+, w+ > e+ ve", "a decay chain (',')"),
        ("g g > t t~, (t > b w+)", "a decay chain (',')"),
        ("t > b w+ $ z", "an s-channel exclusion ('$')"),
        ("t > b w+ @1", "a process tag ('@')"),
        ("t > b w+ / z", "a particle exclusion ('/')"),
    ],
)
def test_decay_chains_and_mg5_only_syntax_are_refused_typed(expression, found):
    from pyamplicol.api import ProcessRequest
    from pyamplicol.processes.refusals import ProcessRequestUnsupported

    with pytest.raises(ProcessRequestUnsupported) as refused:
        ProcessRequest.parse(expression)
    assert refused.value.key == "madloop-syntax"
    assert found in str(refused.value)


def test_external_leg_counts_are_one_or_two_initial_particles():
    from pyamplicol.processes.model import require_external_leg_counts

    for initial, final in ((1, 2), (1, 5), (2, 1), (2, 6)):
        require_external_leg_counts(initial, final)
    for initial, final in ((1, 1), (0, 3), (3, 2), (2, 0)):
        with pytest.raises(ValueError, match="pyAmpliCol processes"):
            require_external_leg_counts(initial, final)


def test_one_loop_decays_are_refused_at_the_loop_admission():
    """``decay-process`` left the process parser (tree decays are served) for
    the loop admission, with MF-DEC-2 as its destination; the criterion is
    the number of initial legs."""

    from pyamplicol.loops.v2.refusals import (
        LOOP_ADMISSION_REFUSALS,
        LoopRequestUnsupported,
        require_scattering,
    )
    from pyamplicol.processes.refusals import listed_process_refusal_keys

    assert "decay-process" not in listed_process_refusal_keys()
    (refusal,) = [r for r in LOOP_ADMISSION_REFUSALS if r.key == "decay-process"]
    assert "MF-DEC-2" in refusal.owner
    require_scattering(2, "'p' (u u~ > d d~ g)")
    with pytest.raises(LoopRequestUnsupported) as refused:
        require_scattering(1, "'p' (t > b w+)")
    assert refused.value.key == "decay-process"
    assert str(refused.value).startswith("decay-process: 'p' (t > b w+)")


# --------------------------------------------------------------- kinematics


def test_rational_sphere_points_follow_legendre():
    import random

    decay_points = _toolkit()
    chooser = random.Random(1)
    for n in (0, 1, 2, 3, 6, 2427600, 10**40 + 3):
        a, b, c = decay_points.three_squares(n, chooser)
        assert a * a + b * b + c * c == n
    for n in (7, 28, 1234567, 2**61 - 1):  # 4^a (8b + 7)
        assert decay_points.three_squares(n, chooser) is None


@pytest.mark.parametrize(
    "masses",
    [
        ("173", "0", "80.419"),  # massless + massive: rational |p|
        ("125", "4.7", "4.7"),  # massive pair: |p|^2 a sum of three squares
        ("173", "4.7", "80.419", "0"),  # 1 -> 3 with a massive pair last
        ("91.188", "0", "0", "0", "0"),
    ],
)
def test_decay_points_are_exact_rest_frame_points(masses):
    decay_points = _toolkit()
    masses = [Fraction(m) for m in masses]
    points = decay_points.generate(masses, 4)
    assert len({tuple(x for p in point for x in p) for point in points}) == 4
    for point in points:
        decay_points.check(point, masses)  # exact shells and conservation
        assert point[0] == [masses[0], 0, 0, 0]
        printed = decay_points.decimal_point(point)
        total = [sum(Fraction(p[mu]) for p in printed[1:]) for mu in range(4)]
        assert total == [Fraction(x) for x in printed[0]]


def test_decay_streams_are_exact_rest_frame_points_starting_with_the_campaign():
    from tools.loop_benchmark import streams

    decay_points = _toolkit()
    _, required_spec = streams._toolkit()
    keys = streams.decay_keys()
    assert len(keys) == 8
    for key in keys:
        stream = streams.load(key)
        masses = [
            Fraction(m)
            for m in required_spec.masses(
                required_spec.load_process(streams.DECAY_SPEC, key)
            )
        ]
        points = [streams.exact(stream, i) for i in range(len(stream["points"]))]
        assert len(points) == streams.STREAM_LENGTH
        assert len({tuple(x for p in pt for x in p) for pt in points}) == len(points)
        for point in points:
            decay_points.check(point, masses)
            assert point[0] == [masses[0], 0, 0, 0]
        assert streams.decay_campaign_agreement(key, stream) in (True, None)
        # frozen: the construction is deterministic
        assert streams.make_decay_stream(key, 4)["points"] == stream["points"][:4]


# ------------------------------------------------------------------- physics

# Rows checked in the default tier: the decaying fermion at rest (the rest
# spinor) at 1 -> 2 and 1 -> 3 on one compiled model; the others nightly.
DEFAULT_ROWS = ("t_bwp", "t_bwpg")
ROWS = sorted(path.stem.removeprefix("tree_") for path in DEC.glob("tree_*.json"))


def _rows():
    return [
        pytest.param(
            row,
            marks=pytest.mark.loop_physics
            if row in DEFAULT_ROWS
            else pytest.mark.loop_nightly,
        )
        for row in ROWS
    ]


@pytest.fixture(scope="module")
def prepared_loop_sm(tmp_path_factory):
    """The shipped loop_sm's prepared kernel pack per restriction (compiled
    once per module, ~2 min each)."""

    from pyamplicol.api import ModelSource
    from pyamplicol.config.models import EvaluatorConfig

    root = tmp_path_factory.mktemp("decay-models")
    packs = {}

    def get(restriction):
        if restriction not in packs:
            output = root / f"loop_sm-{restriction}.pyamplicol-model"
            ModelSource.from_path("loop_sm", restriction=restriction).compile(
                prepared_output=output, evaluator=EvaluatorConfig()
            )
            packs[restriction] = output
        return packs[restriction]

    return get


def _boost(momentum, beta):
    """``momentum`` boosted by velocity ``beta`` (a 3-vector)."""

    b2 = sum(x * x for x in beta)
    gamma = 1 / math.sqrt(1 - b2)
    bp = sum(b * p for b, p in zip(beta, momentum[1:], strict=True))
    energy = gamma * (momentum[0] + bp)
    factor = (gamma - 1) * bp / b2 + gamma * momentum[0]
    return [energy, *(p + factor * b for p, b in zip(momentum[1:], beta, strict=True))]


@pytest.mark.parametrize("row", _rows())
def test_decay_trees_equal_mg5_standalone(row, prepared_loop_sm, tmp_path):
    """Full colour on the shipped loop_sm at the MG5 card's parameters (the
    restriction's own zeros kept): = MG5 standalone to rounding (measured
    <= 1.2e-13); the decaying particle is incoming; a stability rotation and
    a boost leave the value unchanged."""

    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.loops.v2.stability import default_rotation

    fixture = json.loads((DEC / f"tree_{row}.json").read_text())
    pack = prepared_loop_sm(fixture["pyamplicol_model"]["restriction"])
    output = tmp_path / row
    card = tmp_path / f"{row}.toml"
    card.write_text(
        f"""
action = "generate"
[model]
source = "{pack}"
[process]
entries = [{{expression = "{fixture["process"]}", name = "p"}}]
[color]
accuracy = "full"
[generation]
output = "{output}"
emit_api_bundle = false
"""
    )
    Generator(resolve_config(card)).generate(
        ProcessSet((ProcessRequest.parse(fixture["process"], name="p"),)), output
    )
    runtime = Runtime.load(output, process="p")
    states = [particle.state for particle in runtime.physics.external_particles]
    assert states == ["incoming"] + ["outgoing"] * (len(fixture["pdgs"]) - 1)
    declared = {p.name: p for p in runtime.physics.model_parameters}
    values = fixture["parameters"]
    for key, value in values.items():
        if key in declared and not declared[key].mutable:
            assert declared[key].default_real == float(value), (row, key)
    runtime.set_model_parameters(
        {k: float(v) for k, v in values.items() if k in declared and declared[k].mutable}
    )
    for point, reference in zip(fixture["points"], fixture["values"], strict=True):
        momenta = [[float(x) for x in leg] for leg in point]
        (value,) = runtime.evaluate([momenta])
        assert abs(complex(value).real - reference) <= 1e-12 * abs(reference), row
    exact = [[Fraction(x) for x in leg] for leg in fixture["exact_points"][0]]
    rotated = default_rotation(0).apply(exact)
    assert [float(x) for x in rotated[0]] == [float(exact[0][0]), 0.0, 0.0, 0.0]
    boosted = [_boost([float(x) for x in leg], (0.3, -0.2, 0.5)) for leg in exact]
    (reference,) = runtime.evaluate([[[float(x) for x in leg] for leg in exact]])
    for momenta in ([[float(x) for x in leg] for leg in rotated], boosted):
        (value,) = runtime.evaluate([momenta])
        assert abs(complex(value - reference)) <= 1e-12 * abs(reference), row
