#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Deterministic on-shell phase-space points for any masses, built at 60 digits.

``exact_points.py`` builds exact terminating decimals, which exist only for
some mass patterns (one massive subsystem, massless beams). This generator
covers every other pattern: several massive subsystems with different masses
(t t~ h, t t~ b b~, t w-, z h), massive beams (b b~ > h), irrational masses (a
W mass derived at tree level) and the limit families of MASTER_PLAN §5.4.

Construction. Every momentum is computed at ``CONSTRUCTION_DIGITS`` (60)
significant digits: beams along +-z in the centre-of-mass frame (massive
beams allowed); the final state (and the generic part of the limit families)
is drawn flat in phase space by pyAmpliCol's one generator
(``pyamplicol.generation.phase_space.flat_final_state``: RAMBO on diet,
arXiv:1308.2922, constructed natively in arbitrary precision at
``CONSTRUCTION_DIGITS`` digits, one 64-bit seed per candidate from the
fixed pseudo-random sequence), so no configuration is special: until
2026-11-14 the final state came from sequential two-body decays with
directions from a few rational cosines and eight Pythagorean azimuths, which
put about one 2->3 point in seven exactly in a plane (zero four-dimensional
Gram determinant); the committed fixtures keep the points of that
construction.
Every component is then rounded to a common absolute quantum of
``PRINTED_DIGITS`` (30) significant digits of sqrt(s), and the last final-state
particle absorbs the rounding residual, so momentum conservation holds exactly
in the printed decimals and every mass shell holds to about 1e-29 relative
(recorded per point). Parameters come from a fixed pseudo-random sequence
(``random.Random(SEED)``), so the points depend only on the masses, the family
and its options.

Families (``--family``):

* ``regular``: generic points, sqrt(s) from a fixed list above threshold, cut
  ``2|p_i.p_j| >= cut*s`` on massless pairs and resonance exclusion on final
  particle-antiparticle pairs (as exact_points.py);
* ``threshold``: a 2->2 final pair of equal masses at velocity beta (one point
  per ``--values`` entry, e.g. 1e-1 1e-2 1e-3), fixed scattering direction;
* ``soft``: the last final-state particle (massless) with energy fraction
  E/sqrt(s) = value, the rest generic;
* ``collinear``: the last two final-state particles (massless) at opening
  angle theta = value, the rest generic;
* ``radiation-zero``: a 2->2 process with one massless final particle (the
  photon) at the polar cosine ``--zero-cosine`` relative to beam 1 (e.g. -1/3 for
  u d~ > w+ a), plus one regular neighbour at ``--neighbour-cosine``;
* ``rest``: a 2->1 process at sqrt(s) = the final mass, the beam axis turned
  by a different rational rotation per point.

``--sqrt-s v1 v2 ...`` (regular family) builds one point per listed sqrt(s), in
order (repeats allowed), without the resonance exclusion: points at or near a
resonance (sqrt(s) = M_Z, M_W) or at one high energy.

``--rotate x`` turns the whole event (beams included) about the x axis by the
rational angle cos = 3/5, sin = 4/5 (beams off the z axis). ``--half-scale-point``
appends a copy of point 1 at half its scale.
"""

import argparse
import itertools
import json
import random
import sys
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction as F
from pathlib import Path

CONSTRUCTION_DIGITS = 60
PRINTED_DIGITS = 30
SEED = 20261028
MAX_TRIALS = 200000  # candidates per point before an unreachable configuration is refused
_PYTHAGOREAN = [(F(3, 5), F(4, 5)), (F(7, 25), F(24, 25)), (F(117, 125), F(44, 125)),
                (F(4, 5), F(3, 5)), (F(527, 625), F(336, 625)), (F(-3, 5), F(4, 5)),
                (F(-7, 25), F(-24, 25)), (F(4, 5), F(-3, 5))]
_COSINES = [F(n, 20) for n in range(-17, 18) if n]
_SQRT_S = [F(x) for x in (800, 1000, 1200, 1500, 1750, 2000, 2500, 3000, 3500, 4000)]


def D(x):
    """Exact Decimal of a Fraction or decimal string at the construction precision."""
    if isinstance(x, F):
        return Decimal(x.numerator) / Decimal(x.denominator)
    return Decimal(x)


def dot(p, q):
    return p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]


def direction(cos_theta, azimuth):
    """Unit vector from a rational polar cosine and a Pythagorean (cos, sin) azimuth."""
    c = D(cos_theta)
    s = (1 - c * c).sqrt()
    return [s * D(azimuth[0]), s * D(azimuth[1]), c]


def two_body(mass, m1, m2, n):
    """Momenta of a two-body decay at rest: particle 1 along n, particle 2 opposite."""
    m, a, b = D(mass), D(m1), D(m2)
    lam = (m * m - (a + b) ** 2) * (m * m - (a - b) ** 2)
    if lam < 0:
        raise ValueError("below threshold")
    p = lam.sqrt() / (2 * m)
    e1, e2 = (a * a + p * p).sqrt(), (b * b + p * p).sqrt()
    return [e1, *(p * x for x in n)], [e2, *(-p * x for x in n)]


def boost(q, frame):
    """Boost q from the rest frame of ``frame`` (a timelike momentum) to its frame."""
    mass = dot(frame, frame).sqrt()
    gamma = frame[0] / mass
    beta = [x / frame[0] for x in frame[1:]]
    b2 = sum(x * x for x in beta)
    bq = sum(b * x for b, x in zip(beta, q[1:], strict=False))
    factor = ((gamma - 1) * bq / b2 if b2 else 0) + gamma * q[0]
    return [gamma * (q[0] + bq), *(x + factor * b for x, b in zip(q[1:], beta, strict=False))]


def beams(sqrt_s, m1, m2):
    s, a, b = D(sqrt_s) ** 2, D(m1), D(m2)
    lam = (s - (a + b) ** 2) * (s - (a - b) ** 2)
    p = lam.sqrt() / (2 * D(sqrt_s))
    e1, e2 = (s + a * a - b * b) / (2 * D(sqrt_s)), (s + b * b - a * a) / (2 * D(sqrt_s))
    return [e1, Decimal(0), Decimal(0), p], [e2, Decimal(0), Decimal(0), -p]


def sequential(sqrt_s, masses, chooser, remainder_masses=None, directions=None):
    """Final-state momenta in the centre-of-mass frame: particle k is emitted
    from the remaining system, whose mass is drawn between threshold and the
    parent mass (or taken from ``remainder_masses``)."""
    masses = [D(m) for m in masses]
    parent = [D(sqrt_s), Decimal(0), Decimal(0), Decimal(0)]
    out = []
    for k in range(len(masses) - 1):
        rest = masses[k + 1:]
        parent_mass = dot(parent, parent).sqrt()
        if remainder_masses is not None and remainder_masses[k] is not None:
            m_rest = D(remainder_masses[k])
        elif len(rest) == 1:
            m_rest = rest[0]
        else:
            low, high = sum(rest), parent_mass - masses[k]
            m_rest = low + D(chooser.choice([F(n, 10) for n in range(2, 9)])) * (high - low)
        n = (directions[k] if directions is not None and directions[k] is not None
             else direction(chooser.choice(_COSINES), chooser.choice(_PYTHAGOREAN)))
        emitted, system = two_body(parent_mass, masses[k], m_rest, n)
        out.append(boost(emitted, parent))
        parent = boost(system, parent)
    out.append(parent)
    return out


def flat(sqrt_s, masses, chooser):
    """A flat final state (``flat_final_state``) at ``sqrt_s`` in the
    centre-of-mass frame as Decimal momenta at the construction precision,
    from one 64-bit seed of ``chooser``."""
    from pyamplicol.generation.phase_space import flat_final_state

    point = flat_final_state([F(D(m)) for m in masses], sqrt_s=F(D(sqrt_s)), seed=chooser.getrandbits(64),
                             arithmetic="arbitrary", digits=CONSTRUCTION_DIGITS)
    return [[D(x) for x in leg] for leg in point.momenta]


def flat_system(system, masses, chooser):
    """A flat final state of ``masses`` in the rest frame of the timelike
    ``system`` momentum, boosted to the frame where ``system`` is given."""
    if len(masses) == 1:
        return [system]
    return [boost(q, system) for q in flat(dot(system, system).sqrt(), masses, chooser)]


def sin_cos(theta):
    """sin and cos of a Decimal angle by their Taylor series (small angles)."""
    sin, cos, term, k = Decimal(0), Decimal(0), Decimal(1), 0
    while True:
        if k % 2 == 0:
            cos += term if k % 4 == 0 else -term
        else:
            sin += term if k % 4 == 1 else -term
        k += 1
        term = term * theta / k
        if abs(term) < Decimal(10) ** (-2 * CONSTRUCTION_DIGITS):
            return sin, cos


def collinear_pair(sqrt_s, final, theta, chooser, z=F(2, 5)):
    """The last two (massless) particles at opening angle ``theta`` in the
    centre-of-mass frame, with energies z E and (1 - z) E of their pair: the
    pair system (mass^2 = 2 z (1 - z) E^2 (1 - cos theta)) recoils against the
    other particles, which decay sequentially."""
    rest = final[:-2]
    sin_t, cos_t = sin_cos(theta)
    c = 2 * D(z) * (1 - D(z)) * (1 - cos_t)
    root_s = D(sqrt_s)
    s = root_s * root_s
    if len(rest) == 1:
        r = D(rest[0])
    else:
        low = sum(D(m) for m in rest)
        r = low + D(chooser.choice([F(n, 10) for n in range(2, 9)])) * (root_s - low)
    energy = (s - r * r) / (root_s + (s - c * (s - r * r)).sqrt())
    mass = (c * energy * energy).sqrt()
    # The pair recoils against the rest isotropically (a flat two-body state).
    pair, system = flat(root_s, [mass, r], chooser)
    momentum = (pair[0] ** 2 - mass * mass).sqrt()
    n = [x / momentum for x in pair[1:]]
    e4, e5 = D(z) * pair[0], (1 - D(z)) * pair[0]
    sin_a, sin_b = e5 * sin_t / momentum, e4 * sin_t / momentum
    cos_a, cos_b = (1 - sin_a ** 2).sqrt(), (1 - sin_b ** 2).sqrt()
    # A unit vector orthogonal to n, in an isotropic orientation about n:
    # the part of an isotropic direction (a flat massless pair's) orthogonal
    # to n.
    helper = flat(root_s, [0, 0], chooser)[0]
    helper = [x / helper[0] for x in helper[1:]]
    along = sum(a * b for a, b in zip(helper, n, strict=False))
    e = [a - along * b for a, b in zip(helper, n, strict=False)]
    norm = sum(x * x for x in e).sqrt()
    e = [x / norm for x in e]
    p4 = [e4, *(e4 * (cos_a * a + sin_a * b) for a, b in zip(n, e, strict=False))]
    p5 = [e5, *(e5 * (cos_b * a - sin_b * b) for a, b in zip(n, e, strict=False))]
    others = flat_system(system, rest, chooser)
    return [*others, p4, p5]


def rotate_x(point):
    c, s = D(F(3, 5)), D(F(4, 5))
    return [[p[0], p[1], c * p[2] - s * p[3], s * p[2] + c * p[3]] for p in point]


def rotation(index):
    """A deterministic rational rotation (product of two Pythagorean rotations)."""
    (c1, s1), (c2, s2) = _PYTHAGOREAN[index % len(_PYTHAGOREAN)], _PYTHAGOREAN[(3 * index + 1) % len(_PYTHAGOREAN)]
    c1, s1, c2, s2 = map(D, (c1, s1, c2, s2))
    rx = ((1, 0, 0), (0, c1, -s1), (0, s1, c1))
    ry = ((c2, 0, s2), (0, 1, 0), (-s2, 0, c2))
    m = [[sum(D(rx[i][k]) * D(ry[k][j]) for k in range(3)) for j in range(3)] for i in range(3)]
    return lambda p: [p[0], *(sum(m[i][k] * p[1 + k] for k in range(3)) for i in range(3))]


def quantise(point, sqrt_s, incoming=2):
    """Round every component to the common quantum; the last final particle
    takes the residual so momentum conservation is exact in the printed decimals."""
    exponent = D(sqrt_s).adjusted() - PRINTED_DIGITS + 1
    quantum = Decimal(1).scaleb(exponent)
    rounded = [[x.quantize(quantum, rounding=ROUND_HALF_EVEN) for x in p] for p in point]
    total_in = [sum(p[mu] for p in rounded[:incoming]) for mu in range(4)]
    others = [sum(p[mu] for p in rounded[incoming:-1]) for mu in range(4)]
    rounded[-1] = [total_in[mu] - others[mu] for mu in range(4)]
    return rounded


def madloop_qp_point(point, masses, incoming=2):
    """The point MadLoop evaluates in quadruple precision for ``point``.

    MadLoop reads binary64 momenta (correctly rounded) and, in quadruple
    precision with ImprovePSPoint = 2, restores on-shellness and momentum
    conservation by its ORIG method (``MP_ORIG_IMPROVE_PS_POINT_PRECISION``
    of the export's improve_ps.f): the final-state E, px, py are kept and pz is
    put on shell; the two massless beams along +-z absorb the imbalance. This
    replica uses the same formulas at the construction precision, so a limit
    point and MadLoop's quadruple-precision evaluation describe the same
    kinematics to ~1e-33 instead of the ~1e-16 of the binary64 input. Only for
    massless beams along the z axis (MadLoop's preconditions for ORIG)."""
    p = [[Decimal(float(x)) for x in leg] for leg in point]
    if incoming != 2 or any(D(m) for m in masses[:2]):
        raise ValueError("the ORIG projection needs two massless beams")
    ref = sum(abs(leg[0]) for leg in p)
    if any(abs(p[i][j]) / ref > Decimal("1e-15") for i in (0, 1) for j in (1, 2)):
        raise ValueError("the ORIG projection needs beams along the z axis")
    new = [list(leg) for leg in p]
    for i in range(incoming, len(p)):
        e, x, y, z = p[i]
        new[i][3] = (abs(e * e - x * x - y * y - D(masses[i]) ** 2)).sqrt().copy_sign(z)
    pt = [sum(new[i][mu] for i in range(incoming, len(p))) for mu in range(4)]
    if p[0][3] > 0:
        a, b = 0, 1
    elif p[1][3] > 0:
        a, b = 1, 0
    else:
        raise ValueError("no beam along +z")
    discr = abs(-pt[0] ** 2 + pt[1] ** 2 + pt[2] ** 2 + pt[3] ** 2)
    d0 = 2 * (pt[0] - pt[3]) * (pt[0] + pt[3])
    shift_e1 = (pt[0] * (-2 * p[a][0] * pt[0] + pt[0] ** 2 + pt[1] ** 2 + pt[2] ** 2)
                + (2 * p[a][0] - pt[0]) * pt[3] ** 2 + pt[3] * discr) / d0
    shift_e2 = -(pt[0] * (2 * p[b][0] * pt[0] - pt[0] ** 2 + pt[1] ** 2 + pt[2] ** 2)
                 + (-2 * p[b][0] + pt[0]) * pt[3] ** 2 + pt[3] * discr) / d0
    d1 = 2 * (pt[0] ** 2 - pt[3] ** 2)
    shift_z1 = (-2 * p[a][3] * (pt[0] ** 2 - pt[3] ** 2) + pt[3] * (pt[0] ** 2 + pt[1] ** 2 + pt[2] ** 2 - pt[3] ** 2)
                + pt[0] * discr) / d1
    shift_z2 = -(2 * p[b][3] * (pt[0] ** 2 - pt[3] ** 2) + pt[3] * (-pt[0] ** 2 + pt[1] ** 2 + pt[2] ** 2 + pt[3] ** 2)
                 + pt[0] * discr) / d1
    new[a][0], new[a][3] = p[a][0] + shift_e1, p[a][3] + shift_z1
    new[b][0], new[b][3] = p[b][0] + shift_e2, p[b][3] + shift_z2
    new[b][1], new[b][2] = p[b][1], p[b][2]
    for j in (1, 2):
        new[a][j] = sum(p[i][j] for i in range(incoming, len(p))) - p[b][j]
    return new


def print_digits(point, sqrt_s):
    """Round to the printed quantum without moving the residual (a projected
    point is already conserving to the construction precision)."""
    exponent = D(sqrt_s).adjusted() - PRINTED_DIGITS + 1
    quantum = Decimal(1).scaleb(exponent)
    return [[x.quantize(quantum, rounding=ROUND_HALF_EVEN) for x in leg] for leg in point]


def text(x):
    """Exact decimal text of ``x`` (no rounding: ``format`` of a Decimal keeps
    every digit, whatever the context precision; ``normalize`` would round to
    the context's 28 digits outside the construction context)."""
    t = format(x, "f")
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    return "0" if t in ("", "-0") else t


def passes_cuts(point, masses, cut, resonances, pdgs, incoming=2):
    s = dot([a + b for a, b in zip(point[0], point[1], strict=False)], [a + b for a, b in zip(point[0], point[1], strict=False)])
    for (i, p), (j, q) in itertools.combinations(enumerate(point), 2):
        if not masses[i] and not masses[j] and 2 * abs(dot(p, q)) < D(cut) * s:
            return False
    final = point[incoming:]
    for (i, p), (j, q) in itertools.combinations(enumerate(final), 2):
        if pdgs is not None and pdgs[incoming + i] != -pdgs[incoming + j]:
            continue
        pair = [x + y for x, y in zip(p, q, strict=False)]
        pair_s = dot(pair, pair)
        if any(abs(pair_s - D(r) ** 2) < D(r) ** 2 / 10 for r in resonances):
            return False
    return True


def generate(masses, family="regular", count=3, values=(), cut=F(1, 1000), resonances=(), pdgs=None,
             rotate=None, zero_cosine=None, neighbour_cosine=None, incoming=2, sqrt_s_values=None):
    """Return (points, sqrt_s list, labels) as Decimal momenta [[E, px, py, pz], ...]."""
    if incoming != 2:
        raise ValueError("only 2->n scattering is implemented")
    chooser = random.Random(SEED)
    initial, final = masses[:incoming], masses[incoming:]
    threshold = max(sum(D(m) for m in initial), sum(D(m) for m in final))
    points, scales, labels = [], [], []

    def finish(final_momenta, sqrt_s, label, turn=None):
        b1, b2 = beams(sqrt_s, *initial)
        point = [b1, b2, *final_momenta]
        if turn is not None:
            point = [turn(p) for p in point]
        if rotate == "x":
            point = rotate_x(point)
        points.append(point)
        scales.append(sqrt_s)
        labels.append(label)

    if family == "rest":
        if len(final) != 1:
            raise ValueError("family 'rest' is a 2->1 process")
        sqrt_s = F(final[0])
        for index in range(count):
            finish([[D(sqrt_s), Decimal(0), Decimal(0), Decimal(0)]], sqrt_s, f"rest frame, beam rotation {index}",
                   rotation(index))
        return points, scales, labels
    if family == "threshold":
        if len(final) != 2 or final[0] != final[1] or not final[0]:
            raise ValueError("family 'threshold' needs a final pair of equal nonzero masses")
        n = direction(F(3, 5), _PYTHAGOREAN[1])
        for beta in values:
            with localcontext() as context:
                context.prec = CONSTRUCTION_DIGITS
                sqrt_s_d = 2 * D(final[0]) / (1 - D(beta) ** 2).sqrt()
            # sqrt(s) is irrational: the quantisation below rounds it with the momenta.
            pair = two_body(sqrt_s_d, final[0], final[1], n)
            finish(list(pair), F(sqrt_s_d), f"threshold beta={beta}")
        return points, scales, labels
    if family == "radiation-zero":
        if len(final) != 2 or (final[0] and final[1]):
            raise ValueError("family 'radiation-zero' needs a 2->2 process with one massless final particle")
        photon = 0 if not final[0] else 1
        sqrt_s_list = [s for s in _SQRT_S if D(s) > 2 * threshold][:count]
        cosines = [(zero_cosine, "radiation zero")] * len(sqrt_s_list) + [(neighbour_cosine, "regular neighbour")]
        for index, (cosine, label) in enumerate(cosines):
            sqrt_s = sqrt_s_list[min(index, len(sqrt_s_list) - 1)]
            n = direction(cosine, _PYTHAGOREAN[index % len(_PYTHAGOREAN)])
            # two_body puts final[0] along its direction: the massless particle along n.
            first = n if photon == 0 else [-x for x in n]
            pair = two_body(sqrt_s, final[0], final[1], first)
            finish(list(pair), sqrt_s, f"{label} cos(beam1, massless)={cosine}")
        return points, scales, labels

    scales_ok = [s for s in _SQRT_S if D(s) > threshold * D(F(13, 10)) and D(s) > threshold + 50]
    if sqrt_s_values:
        if family != "regular":
            raise ValueError("--sqrt-s applies to the regular family only")
        if any(D(s) <= threshold for s in sqrt_s_values):
            raise ValueError(f"a listed sqrt(s) is not above the threshold {threshold}")
        scales_ok, count = list(sqrt_s_values), len(sqrt_s_values)
    if not scales_ok:
        raise ValueError("threshold above the sqrt(s) list")
    # Configuration errors are refused before the search (inside it a ValueError
    # only rejects one candidate), and the search itself is bounded.
    if family not in ("regular", "soft", "collinear"):
        raise ValueError(f"unknown family {family!r}")
    if family == "soft" and final[-1]:
        raise ValueError("soft family needs a massless last particle")
    if family == "collinear" and (len(final) < 2 or final[-1] or final[-2]):
        raise ValueError("collinear family needs two massless last particles")
    if family in ("soft", "collinear") and not values:
        raise ValueError(f"family {family!r} needs --values")
    if family == "soft":
        rest_threshold = sum(D(m) for m in final[:-1])
        for target in values:
            # The recoiling system's mass sqrt(s - 2 E sqrt(s)) must exceed its threshold
            # at one sqrt(s) of the list at least.
            if not any(D(s) ** 2 * (1 - 2 * D(target)) > rest_threshold ** 2 for s in scales_ok):
                raise ValueError(f"soft energy fraction {target} leaves no room for the other particles")
    if family == "regular" and count > len(scales_ok):
        raise ValueError(f"{count} regular points need {count} distinct sqrt(s) values; "
                         f"only {len(scales_ok)} lie above threshold")
    targets = list(values) if family in ("soft", "collinear") else [None] * count
    for index, target in enumerate(targets):
        for trial in itertools.count(1):
            if trial > MAX_TRIALS:
                raise ValueError(f"no {family} point found in {MAX_TRIALS} candidates "
                                 f"(target {target}); the configuration is unreachable")
            if trial % 10000 == 0:
                print(f"supplied_points: {trial} candidates tried", file=sys.stderr)
            sqrt_s = sqrt_s_values[index] if sqrt_s_values else chooser.choice(scales_ok)
            try:
                if family == "soft":
                    # The soft particle recoils isotropically against the
                    # rest (a flat two-body state), which decays flat.
                    energy = D(target) * D(sqrt_s)
                    m_rest = (D(sqrt_s) ** 2 - 2 * D(sqrt_s) * energy).sqrt()
                    if m_rest <= sum(D(m) for m in final[:-1]):
                        raise ValueError("below threshold")
                    soft, system = flat(sqrt_s, [0, m_rest], chooser)
                    final_momenta = [*flat_system(system, final[:-1], chooser), soft]
                elif family == "collinear":
                    final_momenta = collinear_pair(sqrt_s, final, D(target), chooser)
                else:
                    final_momenta = flat(sqrt_s, final, chooser)
            except ValueError:
                continue
            b1, b2 = beams(sqrt_s, *initial)
            if family == "regular" and not passes_cuts([b1, b2, *final_momenta], masses, cut, resonances, pdgs):
                continue
            if sqrt_s in scales and family == "regular" and not sqrt_s_values:
                continue
            finish(final_momenta, sqrt_s, family if target is None else f"{family} {target}")
            break
    return points, scales, labels


def measure(point, masses, incoming=2):
    """Largest |p^2 - m^2| / E^2 and largest momentum-conservation defect / E."""
    shell = max(abs(dot(p, p) - D(m) ** 2) / p[0] ** 2 for p, m in zip(point, masses, strict=False))
    scale = max(p[0] for p in point)
    conservation = max(abs(sum(p[mu] for p in point[:incoming]) - sum(p[mu] for p in point[incoming:]))
                       for mu in range(4)) / scale
    return shell, conservation


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--process", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", default="regular",
                        choices=("regular", "threshold", "soft", "collinear", "radiation-zero", "rest"))
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--values", nargs="*", default=[], help="beta / energy fractions / angles of a limit family")
    parser.add_argument("--zero-cosine", default="-1/3")
    parser.add_argument("--neighbour-cosine", default="-1/5")
    parser.add_argument("--rotate", choices=("x",))
    parser.add_argument("--sqrt-s", nargs="+", help="regular family: one point per listed sqrt(s), no resonance "
                        "exclusion")
    parser.add_argument("--mu-r", nargs="+", default=["1000", "700", "1400"])
    parser.add_argument("--cut", default="1/1000", help="minimal 2|p_i.p_j|/s of massless pairs (regular family)")
    parser.add_argument("--half-scale-point", action="store_true")
    parser.add_argument("--madloop-qp-point", action="store_true",
                        help="publish the point MadLoop evaluates in quadruple precision (madloop_qp_point); "
                        "use with a MadLoop accuracy target that forces its quadruple-precision evaluation")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import required_spec
    from exact_points import decimal_string

    spec = required_spec.load_process(args.spec or required_spec.DEFAULT_SPEC, args.process)
    masses = required_spec.masses(spec)
    values = {**spec["scheme_data"]["parameters"], **spec["scheme_data"]["fixed_parameters"]}
    resonances = [values[n] for n in ("MZ", "MW", "MH", "MT") if n in values and F(values[n])]
    if "MW" not in values:
        resonances.append(required_spec.derived_mw(spec["scheme_data"]))
    if args.sqrt_s:
        resonances = []
    with localcontext() as context:
        context.prec = CONSTRUCTION_DIGITS
        try:
            points, scales, labels = generate(
                masses, args.family, args.count, [F(v) for v in args.values], F(args.cut),
                [F(r) for r in resonances], spec["pdgs"], args.rotate, F(args.zero_cosine), F(args.neighbour_cosine),
                sqrt_s_values=[F(v) for v in args.sqrt_s] if args.sqrt_s else None)
        except ValueError as error:
            raise SystemExit(f"supplied_points: {args.process}, family {args.family}: {error}") from None
        printed = [quantise(p, s) for p, s in zip(points, scales, strict=False)]
        if args.madloop_qp_point:
            printed = [print_digits(madloop_qp_point(p, masses), s) for p, s in zip(printed, scales, strict=False)]
        defects = [measure(p, masses) for p in printed]
    if args.half_scale_point:
        printed.append(printed[0])
        labels.append(labels[0] + " (mu_R = mu_0/2)")
        defects.append(defects[0])
    mu_r = list(itertools.islice(itertools.cycle(args.mu_r), len(points)))
    if args.half_scale_point:
        mu_r.append(decimal_string(F(mu_r[0]) / 2))
    report = {
        "process_name": spec["name"],
        "pdgs": spec["pdgs"],
        "masses": [decimal_string(m) for m in masses],
        "construction": (
            f"tools/loop_reference/supplied_points.py: flat phase-space construction (pyAmpliCol's "
            f"flat_final_state, RAMBO on diet) at {CONSTRUCTION_DIGITS} digits, printed on a common quantum of {PRINTED_DIGITS} significant digits "
            "of sqrt(s); momentum conservation exact in the printed decimals (the last final particle takes "
            "the rounding residual), mass shells to the recorded defects"),
        "family": args.family,
        "family_values": args.values,
        "rotation": "about x by cos=3/5, sin=4/5 (beams off the z axis)" if args.rotate == "x" else None,
        "madloop_qp_point": (
            "each point is MadLoop's quadruple-precision point of its binary64 rounding (ImprovePSPoint=2, "
            "ORIG method of improve_ps.f, replicated at the construction precision)" if args.madloop_qp_point
            else None),
        "point_labels": labels,
        "cuts": {"min_2pipj_over_s": args.cut, "resonance_exclusion":
                 "none (--sqrt-s)" if args.sqrt_s else
                 "|s_ij-M^2| >= M^2/10 for final particle-antiparticle pairs (regular family)",
                 "resonances": [str(r) for r in resonances]},
        "sqrt_s": [text(D(s)) for s in scales] + ([text(D(scales[0]))] if args.half_scale_point else []),
        "points": [[[text(x) for x in leg] for leg in point] for point in printed],
        "mu_r": mu_r,
        "precision": {
            "construction_digits": CONSTRUCTION_DIGITS, "printed_digits": PRINTED_DIGITS,
            "max_relative_mass_shell_defect": [f"{d[0]:.3e}" for d in defects],
            "max_relative_momentum_conservation_defect": [f"{d[1]:.3e}" for d in defects]},
    }
    if args.half_scale_point:
        report["scale_variation"] = {"point": len(printed), "copy_of_point": 1, "mu_r_factor": "1/2"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
