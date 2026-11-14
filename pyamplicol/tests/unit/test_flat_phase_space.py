# SPDX-License-Identifier: 0BSD
"""The flat phase-space generator (``generation.phase_space.flat_final_state``,
RAMBO on diet, arXiv:1308.2922; native ``rusticol_core::phase_space``):
flatness against the analytic energy distribution of flat phase space and
against classic RAMBO's pair invariants, unit
massless weights, the massive weight's mean against the exact phase-space
volume ratio, mass shells and momentum conservation in binary64,
double-double and arbitrary precision, the same point in every carrier,
reproducibility, and no exactly planar 2 -> 3 point."""

from __future__ import annotations

import math
import random
from decimal import Decimal, localcontext
from fractions import Fraction

import pytest

from pyamplicol.generation.phase_space import flat_final_state, flat_final_states

POINTS = 10_000
# The Kolmogorov-Smirnov critical value at 0.1 % (sqrt(N) D_N), so a
# correct generator fails one check in a thousand seeds; the seeds are fixed.
KS_CRITICAL = 1.95


def _dot(p, q):
    return p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]


def _ks(samples, cdf) -> float:
    ordered = sorted(samples)
    n = len(ordered)
    return math.sqrt(n) * max(
        max(abs((i + 1) / n - cdf(x)), abs(i / n - cdf(x)))
        for i, x in enumerate(ordered)
    )


def _ks2(a, b) -> float:
    """The two-sample Kolmogorov-Smirnov statistic sqrt(n m / (n + m)) D."""

    a, b = sorted(a), sorted(b)
    i = j = 0
    distance = 0.0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            i += 1
        else:
            j += 1
        distance = max(distance, abs(i / len(a) - j / len(b)))
    return distance * math.sqrt(len(a) * len(b) / (len(a) + len(b)))


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_massless_points_follow_flat_phase_space(n):
    """n flat massless particles over 10^4 points: the energy fraction
    x = 2 E / sqrt(s) of the first and of the last particle has the flat
    phase-space density (n - 1) (n - 2) x (1 - x)^(n - 3) (d^3p / E times
    the massless (n - 1)-body volume ~ (1 - x)^(n - 3)), and the invariants
    of the first and of the last pair are distributed as those of classic
    RAMBO (Kleiss, Stirling, Ellis: an independent flat construction,
    ``massive_rambo_final_state`` at zero masses); unit weights."""

    from pyamplicol.generation.phase_space import massive_rambo_final_state

    states = flat_final_states([0.0] * n, sqrt_s=1000.0, seeds=range(POINTS))
    assert all(state.weight == 1.0 for state in states)
    b = n - 2

    def cdf(x):
        x = min(max(x, 0.0), 1.0)
        return 1.0 - (1.0 - x) ** b * (1.0 + b * x)

    for leg in (0, n - 1):
        fractions = [2 * st.momenta[leg][0] / 1000.0 for st in states]
        assert _ks(fractions, cdf) < KS_CRITICAL, (n, leg)
    classic = [
        massive_rambo_final_state(n, sqrt_s=1000.0, masses=[0.0] * n, seed=10**6 + k)
        for k in range(POINTS)
    ]
    for i, j in ((0, 1), (n - 2, n - 1)):
        ours = [2 * _dot(st.momenta[i], st.momenta[j]) for st in states]
        theirs = [2 * _dot(p[i], p[j]) for p in classic]
        assert _ks2(ours, theirs) < KS_CRITICAL, (n, i, j)


def test_two_body_angles_are_flat_in_cos_theta_and_phi():
    states = flat_final_states([0.0, 0.0], sqrt_s=500.0, seeds=range(POINTS))
    cosines = [st.momenta[0][3] / st.momenta[0][0] for st in states]
    phis = [math.atan2(st.momenta[0][2], st.momenta[0][1]) for st in states]
    assert _ks(cosines, lambda c: (min(max(c, -1.0), 1.0) + 1) / 2) < KS_CRITICAL
    assert _ks(phis, lambda p: (p + math.pi) / (2 * math.pi)) < KS_CRITICAL


def _three_body_volume_ratio(m, root_s):
    """The massive to massless three-body phase-space volume ratio: the
    Dalitz area over s^2 / 2 (by the midpoint rule on s_12)."""

    s = root_s * root_s
    low, high = (m[0] + m[1]) ** 2, (root_s - m[2]) ** 2
    steps = 20_000
    area = 0.0
    for k in range(steps):
        s12 = low + (k + 0.5) * (high - low) / steps
        e2 = (s12 - m[0] ** 2 + m[1] ** 2) / (2 * math.sqrt(s12))
        e3 = (s - s12 - m[2] ** 2) / (2 * math.sqrt(s12))
        p2 = math.sqrt(max(e2 * e2 - m[1] ** 2, 0.0))
        p3 = math.sqrt(max(e3 * e3 - m[2] ** 2, 0.0))
        area += 4 * p2 * p3 * (high - low) / steps
    return area / (s * s / 2)


@pytest.mark.parametrize(
    "masses", [(173.0, 173.0, 0.0), (91.1876, 91.1876, 0.0), (125.0, 0.0, 0.0)]
)
def test_massive_weight_mean_is_the_phase_space_volume_ratio(masses):
    """The mean of the massive mapping's weight is the ratio of the massive
    to the massless phase-space volume (t t~ g, z z g, h g g at 1 TeV),
    within four standard errors (the weight of one point may exceed 1);
    every point on its shells."""

    states = flat_final_states(list(masses), sqrt_s=1000.0, seeds=range(POINTS))
    weights = [st.weight for st in states]
    mean = sum(weights) / len(weights)
    spread = math.sqrt(sum((w - mean) ** 2 for w in weights) / (len(weights) - 1))
    expected = _three_body_volume_ratio(masses, 1000.0)
    assert abs(mean - expected) < 4 * spread / math.sqrt(len(weights)) + 1e-6
    # A Jacobian: positive and finite (its mean, the volume ratio, is < 1).
    assert all(0.0 < w < math.inf for w in weights) and mean < 1.0
    for st in states[:200]:
        for leg, m in zip(st.momenta, masses, strict=True):
            assert abs(_dot(leg, leg) - m * m) <= 1e-12 * leg[0] ** 2


def test_two_body_massive_weight_is_exact():
    m1, m2, root_s = 173.0, 91.0, 600.0
    s = root_s * root_s
    lam = (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)
    state = flat_final_state([m1, m2], sqrt_s=root_s, seed=4)
    assert state.weight == pytest.approx(math.sqrt(lam) / s, rel=1e-14)


def _residuals(momenta, masses, root_s):
    scale = Fraction(root_s)
    shell = max(
        abs(_dot(p, p) - Fraction(m) ** 2) / scale**2
        for p, m in zip(momenta, masses, strict=True)
    )
    total = [sum(p[mu] for p in momenta) for mu in range(4)]
    conservation = max(
        abs(total[0] - scale) / scale, *(abs(x) / scale for x in total[1:])
    )
    return float(shell), float(conservation)


@pytest.mark.parametrize(
    "masses",
    [
        ("0", "0", "0", "0", "0"),
        ("173", "173", "0"),
        ("91.1876", "91.1876", "0"),
        ("125", "0", "0"),
    ],
)
def test_every_carrier_is_on_shell_and_conserving_and_draws_the_same_point(masses):
    """binary64, double-double and arbitrary precision (60 digits): shells
    and conservation at each carrier's rounding, and the binary64 and
    double-double points equal the arbitrary-precision one to their
    rounding (one carrier-independent random stream)."""

    exact = [Fraction(m) for m in masses]
    for seed in range(10):
        b64 = flat_final_state(exact, sqrt_s=2000, seed=seed)
        dd = flat_final_state(exact, sqrt_s=2000, seed=seed, arithmetic="double-double")
        mp = flat_final_state(
            exact, sqrt_s=2000, seed=seed, arithmetic="arbitrary", digits=60
        )
        legs64 = [[Fraction(x) for x in leg] for leg in b64.momenta]
        for legs, bound in ((legs64, 1e-14), (dd.momenta, 1e-29), (mp.momenta, 1e-57)):
            shell, conservation = _residuals(legs, exact, 2000)
            assert shell < bound and conservation < bound, (masses, seed, bound)
        for legs, bound in ((legs64, 1e-13), (dd.momenta, 1e-28)):
            for leg, ref in zip(legs, mp.momenta, strict=True):
                assert all(
                    abs(float((x - y) / 2000)) < bound
                    for x, y in zip(leg, ref, strict=True)
                ), (masses, seed)
        assert abs(float((Fraction(b64.weight) - mp.weight) / mp.weight)) < 1e-13
        # The binary64 binding of exact inputs and of floats is one generator.
        assert b64 == flat_final_state(
            [float(m) for m in exact], sqrt_s=2000.0, seed=seed
        )


def test_a_seed_is_reproducible_and_a_stream_draws_new_points():
    first = flat_final_state([0.0] * 4, sqrt_s=1000.0, seed=17)
    assert first == flat_final_state([0.0] * 4, sqrt_s=1000.0, seed=17)
    assert first != flat_final_state([0.0] * 4, sqrt_s=1000.0, seed=18)
    rng = random.Random(3)
    a = flat_final_state([0.0] * 4, sqrt_s=1000.0, seed=rng)
    b = flat_final_state([0.0] * 4, sqrt_s=1000.0, seed=rng)
    assert a != b
    rng = random.Random(3)
    assert flat_final_state([0.0] * 4, sqrt_s=1000.0, seed=rng) == a


def test_refusals():
    with pytest.raises(ValueError, match="at least two"):
        flat_final_state([0.0], sqrt_s=10.0, seed=1)
    with pytest.raises(ValueError, match="threshold"):
        flat_final_state([6.0, 6.0], sqrt_s=10.0, seed=1)
    with pytest.raises(ValueError, match="negative"):
        flat_final_state([-1.0, 0.0], sqrt_s=10.0, seed=1)
    with pytest.raises(ValueError, match=r"2\^64"):
        flat_final_state([0.0, 0.0], sqrt_s=10.0, seed=2**64)


def _epsilon(a, b, c, d):
    """The Levi-Civita contraction eps(a, b, c, d) (its square is minus the
    four-dimensional Gram determinant): zero exactly for a planar point."""

    rows = [list(a), list(b), list(c), list(d)]
    total = 0
    for perm in _PERMUTATIONS:
        sign = perm[0]
        term = sign
        for row, column in enumerate(perm[1]):
            term *= rows[row][column]
        total += term
    return total


def _permutations():
    import itertools

    out = []
    for perm in itertools.permutations(range(4)):
        inversions = sum(
            1 for i in range(4) for j in range(i + 1, 4) if perm[i] > perm[j]
        )
        out.append(((-1) ** inversions, perm))
    return out


_PERMUTATIONS = _permutations()


def test_no_two_to_three_point_is_planar():
    """g g > x0 g g at sqrt(s) = 1 TeV (the 27-of-200 planar points of the
    former reference construction): over 10^4 points the Levi-Civita
    contraction of the beams and two final momenta, in exact arithmetic on
    the binary64 components, is far from zero relative to s^2 (no
    configuration is special; its smallest value is set by the sample
    size, not by the rounding)."""

    states = flat_final_states([125.0, 0.0, 0.0], sqrt_s=1000.0, seeds=range(POINTS))
    beams = ((500.0, 0.0, 0.0, 500.0), (500.0, 0.0, 0.0, -500.0))
    smallest = min(
        abs(_epsilon(*beams, st.momenta[0], st.momenta[1])) / 1000.0**4 for st in states
    )
    assert smallest > 1e-12


def test_supplied_points_are_not_planar():
    """The reference toolkit's points (``supplied_points.generate``, regular
    family) come from the generator: none of 200 g g > x0 g g points at
    1 TeV is planar, not even to the printed precision (the former
    construction put about one in seven in a plane), each exactly conserving
    after printing."""

    from tools.loop_reference import supplied_points

    masses = [Fraction(0), Fraction(0), Fraction(125), Fraction(0), Fraction(0)]
    with localcontext() as context:
        context.prec = supplied_points.CONSTRUCTION_DIGITS
        points, scales, _ = supplied_points.generate(
            masses, count=200, sqrt_s_values=[Fraction(1000)] * 200
        )
        printed = [
            supplied_points.quantise(p, s) for p, s in zip(points, scales, strict=True)
        ]
    with localcontext() as context:
        context.prec = 200
        for point in printed:
            # Far from planar: the 30-digit printing of an exactly planar
            # 60-digit point (the former construction) leaves ~1e-28 of s^2.
            eps = abs(_epsilon(point[0], point[1], point[2], point[3]))
            assert eps / Decimal(1000) ** 4 > Decimal("1e-12")
            for mu in range(4):
                assert point[0][mu] + point[1][mu] == sum(p[mu] for p in point[2:])
            shell, _ = supplied_points.measure(point, masses)
            assert shell < Decimal("1e-27")


_DRAW = """
import json, sys
from pyamplicol.generation.phase_space import flat_final_states
states = flat_final_states([0.0, 0.0, 173.0, 173.0], sqrt_s=1500.0, seeds=range(64))
print(json.dumps([[list(leg) for leg in s.momenta] + [s.weight] for s in states]))
"""


def test_binary64_points_need_no_symbolica_licensing(tmp_path):
    """The binary64 generator performs no Symbolica state access (the
    licence-free binary64 guarantee of docs/user/symbolica-and-licensing.md):
    concurrent processes without ``SYMBOLICA_LICENSE`` and with the package
    key disabled draw their points while another process holds the user's
    restricted Symbolica permit, bitwise as this (licensed) process."""

    import importlib.util
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "loop_artifact_portability", root / "tools" / "loop_artifact_portability.py"
    )
    assert spec is not None and spec.loader is not None
    portability = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portability)
    expected = [
        [list(leg) for leg in s.momenta] + [s.weight]
        for s in flat_final_states(
            [0.0, 0.0, 173.0, 173.0], sqrt_s=1500.0, seeds=range(64)
        )
    ]
    lock = tmp_path / "symbolica-locks"
    lock.mkdir()
    lock.chmod(0o700)
    base = {k: v for k, v in os.environ.items() if k != "SYMBOLICA_LICENSE"}
    with portability.RestrictedPermit(lock_dir=str(lock)) as permit:
        assert permit.record["verified"], permit.record
        env = permit.environment({**base, "PYAMPLICOL_NATIVE_LIBRARY_KEY": "off"})
        running = [
            subprocess.Popen(
                [sys.executable, "-c", _DRAW],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(3)
        ]
        outputs = [process.communicate(timeout=300) for process in running]
    for process, (stdout, stderr) in zip(running, outputs, strict=True):
        assert process.returncode == 0, stderr[-3000:]
        assert "restricted" not in stderr.lower(), stderr[-3000:]
        assert json.loads(stdout) == expected
