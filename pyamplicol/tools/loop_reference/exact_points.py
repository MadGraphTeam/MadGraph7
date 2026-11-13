#!/usr/bin/env python3
# SPDX-License-Identifier: 0BSD
"""Deterministic exact on-shell 2->n phase-space points with terminating decimals.

Every coordinate is an exact rational whose denominator contains only the
primes 2 and 5, so it is written as a finite decimal string. Mass shells and
momentum conservation hold exactly in rational arithmetic and are asserted.

Construction (generalizes the sequential-decay fixtures of the earlier
seven-leg controls). A seed subsystem is built in its rest frame: an
equal-mass pair, a massive particle with a massless partner, or two massless
particles. Momenta use the rapidity-like parameterization
``E = m (k + 1/k)/2``, ``|p| = m (k - 1/k)/2`` with ``k = 2^a 5^b``, so energies
and momenta terminate. The remaining massless particles are added one at a
time: the current system of mass ``M`` recoils against a massless particle
with ``|p| = M (j - 1/j)/2``; the combined mass is ``M j`` and the boost back
uses ``gamma = (j + 1/j)/2``, ``gamma beta = (j - 1/j)/2``. Rational rotations
with Pythagorean entries (denominators 5^n) remove planar configurations.
Incoming massless beams are along +-z in the centre-of-mass frame.

Representable final states: all massive particles must form equal-mass pairs
or each have a massless partner, and there must be at most one seed carrying
massive particles. Other mass patterns need simultaneous rational solutions of
several mass shells and are rejected explicitly; supply such points directly.

Candidates follow a fixed parameter sequence and are kept only if every
external pair invariant satisfies ``2|p_i.p_j| >= cut * s`` and every final
particle-antiparticle pair invariant stays away from resonance masses. Results are
therefore deterministic for given masses, cuts and target scale.
"""

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction as F
import itertools
import json
import random
import sys
from pathlib import Path

# Rotations about the x, y, z axes with terminating cos/sin.
_PYTHAGOREAN = [(F(3, 5), F(4, 5)), (F(7, 25), F(24, 25)), (F(117, 125), F(44, 125)),
                (F(4, 5), F(3, 5)), (F(527, 625), F(336, 625))]
# Terminating rapidity-like parameters 2^a 5^b > 1: small ones for massless
# additions, a wider range for the seed (it sets the scale of massive pairs).
_TERMINATING = sorted({F(2) ** a * F(5) ** b for a in range(-6, 7) for b in range(-4, 4)})
_K_ADD = [k for k in _TERMINATING if F(6, 5) <= k <= 4]
_K_SEED = [k for k in _TERMINATING if F(6, 5) <= k <= 40]


def terminating(value):
    d = value.denominator
    for prime in (2, 5):
        while d % prime == 0:
            d //= prime
    return d == 1


def decimal_string(value):
    if not terminating(value):
        raise ValueError(f"{value} has no terminating decimal expansion")
    with localcontext() as context:
        context.prec = 400
        text = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    if F(text) != value:
        raise AssertionError("decimal conversion lost exactness")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _rotation(axis, cs):
    c, s = cs
    if axis == 0:
        return ((1, 0, 0), (0, c, -s), (0, s, c))
    if axis == 1:
        return ((c, 0, s), (0, 1, 0), (-s, 0, c))
    return ((c, -s, 0), (s, c, 0), (0, 0, 1))


def _matmul(a, b):
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def _rotate(matrix, p):
    return [p[0], *(sum(matrix[i][k] * p[1 + k] for k in range(3)) for i in range(3))]


def _boost(p, n, gamma, gamma_beta):
    """Boost p (rest frame of a system) into a frame where the system moves along n."""
    pn = sum(a * b for a, b in zip(p[1:], n))
    energy = gamma * p[0] + gamma_beta * pn
    shift = (gamma - 1) * pn + gamma_beta * p[0]
    return [energy, *(p[1 + i] + shift * n[i] for i in range(3))]


def _pair(mass, k, n):
    energy, momentum = mass * (k + 1 / k) / 2, mass * (k - 1 / k) / 2
    return [energy, *(momentum * x for x in n)], [energy, *(-momentum * x for x in n)]


def _dot(p, q):
    return p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]


def plan(masses):
    """Order final-state indices into a seed and a list of massless additions."""
    massive = [i for i, m in enumerate(masses) if m]
    massless = [i for i, m in enumerate(masses) if not m]
    seeds = []
    by_mass = {}
    for i in massive:
        by_mass.setdefault(masses[i], []).append(i)
    for mass, members in sorted(by_mass.items()):
        while len(members) >= 2:
            seeds.append(("pair", members.pop(0), members.pop(0)))
        if members:
            if not massless:
                raise ValueError("unpaired massive particle without massless partner is not representable")
            seeds.append(("massive+massless", members.pop(0), massless.pop(0)))
    if len(seeds) > 1:
        raise ValueError(
            "several independent massive subsystems need simultaneous rational mass shells; "
            "this exact terminating construction does not cover them"
        )
    if not seeds:
        if len(massless) < 2:
            raise ValueError("need at least two final-state particles")
        seeds.append(("massless", massless.pop(0), massless.pop(0)))
    return seeds[0], massless


def construct(masses, parameters, seed_scale):
    """Build one exact final state in the centre-of-mass frame."""
    (kind, a, b), additions = plan(masses)
    params = iter(parameters)
    rotations = itertools.cycle(range(len(_PYTHAGOREAN) * 3))

    def next_rotation():
        r = next(rotations)
        r2 = next(rotations)
        return _matmul(_rotation(r % 3, _PYTHAGOREAN[r // 3]), _rotation((r2 + 1) % 3, _PYTHAGOREAN[r2 // 3]))

    z = (F(0), F(0), F(1))
    n = _rotate(next_rotation(), [0, *z])[1:]
    momenta = {}
    k = next(params)
    if kind == "pair":
        momenta[a], momenta[b] = _pair(masses[a], k, n)
        mass = masses[a] * (k + 1 / k)
    elif kind == "massive+massless":
        heavy, _ = _pair(masses[a], k, n)
        momentum = masses[a] * (k - 1 / k) / 2
        momenta[a] = heavy
        momenta[b] = [momentum, *(-momentum * x for x in n)]
        mass = masses[a] * k
    else:
        if not additions:
            # A massless 2->2 final state has no addition to vary, so the otherwise
            # unused seed parameter sets its scale and scattering axis. Larger final
            # states keep their points unchanged.
            index = _K_SEED.index(k)
            n = _rotate(_matmul(_rotation(index % 3, _PYTHAGOREAN[index // 3 % 5]),
                                _rotation((index + 1) % 3, _PYTHAGOREAN[index % 5])), [0, *z])[1:]
            seed_scale *= k
        energy = seed_scale / 2
        momenta[a] = [energy, *(energy * x for x in n)]
        momenta[b] = [energy, *(-energy * x for x in n)]
        mass = seed_scale
    for index in additions:
        j = next(params)
        direction = _rotate(next_rotation(), [0, *z])[1:]
        matrix = next_rotation()  # rotate the existing system in its rest frame
        gamma, gamma_beta = (j + 1 / j) / 2, (j - 1 / j) / 2
        for key in momenta:
            momenta[key] = _boost(_rotate(matrix, momenta[key]), direction, gamma, gamma_beta)
        momentum = mass * gamma_beta
        momenta[index] = [momentum, *(-momentum * x for x in direction)]
        mass *= j
    return [momenta[i] for i in range(len(masses))], mass


def generate(masses, count=3, cut=F(1, 1000), resonances=(), scale_window=(F(800), F(4000)),
             incoming=2, pdgs=None):
    """Return ``count`` deterministic exact points ``[[E, px, py, pz], ...]``."""
    if incoming != 2:
        raise ValueError("only 2->n scattering with massless beams is implemented")
    final_masses = [F(m) for m in masses[incoming:]]
    if any(F(m) for m in masses[:incoming]):
        raise ValueError("massive initial states are not implemented")
    (kind, a, _), additions = plan(final_masses)
    if kind != "massless":
        # The seed mass is at most m (k + 1/k) and each massless addition
        # multiplies it by j <= 4: refuse a window no candidate can reach.
        reachable = final_masses[a] * (_K_SEED[-1] + 1 / _K_SEED[-1]) * F(_K_ADD[-1]) ** len(additions)
        if reachable < scale_window[0]:
            raise ValueError(
                f"masses {[str(m) for m in final_masses]} cannot reach sqrt(s) >= {scale_window[0]} with "
                "terminating parameters; use supplied_points.py")
    accepted = []
    seen_scales = set()
    chooser = random.Random(20261020)
    for trial in itertools.count(1):
        # No trial limit: report progress instead, so a too-strict cut is
        # visible without silently capping the search.
        if trial % 100000 == 0:
            print(f"exact_points: {trial} candidates tried, {len(accepted)} accepted", file=sys.stderr)
        parameters = [chooser.choice(_K_SEED)] + [chooser.choice(_K_ADD) for _ in additions]
        seed_scale = F(100)
        try:
            final, sqrt_s = construct(final_masses, parameters, seed_scale)
        except ZeroDivisionError:
            continue
        if not any(final_masses):
            # Massless final states may be rescaled exactly into the window.
            while sqrt_s < scale_window[0]:
                final = [[2 * x for x in p] for p in final]
                sqrt_s *= 2
            while sqrt_s > scale_window[1]:
                final = [[x / 2 for x in p] for p in final]
                sqrt_s /= 2
        if not scale_window[0] <= sqrt_s <= scale_window[1] or sqrt_s in seen_scales:
            continue
        half = sqrt_s / 2
        point = [[half, F(0), F(0), half], [half, F(0), F(0), -half], *final]
        s = sqrt_s ** 2
        if any(2 * abs(_dot(p, q)) < cut * s for p, q in itertools.combinations(point, 2)):
            continue
        near = False
        for (i, p), (j, q) in itertools.combinations(enumerate(final), 2):
            if pdgs is not None and pdgs[incoming + i] != -pdgs[incoming + j]:
                continue
            pair_s = _dot([x + y for x, y in zip(p, q)], [x + y for x, y in zip(p, q)])
            if any(abs(pair_s - F(r) ** 2) < F(r) ** 2 / 10 for r in resonances):
                near = True
        if near:
            continue
        check(point, [F(m) for m in masses], incoming)
        accepted.append(point)
        seen_scales.add(sqrt_s)
        if len(accepted) == count:
            return accepted


def check(point, masses, incoming=2):
    for p, m in zip(point, masses, strict=True):
        if _dot(p, p) != m * m or p[0] <= 0:
            raise AssertionError("mass shell violated")
    for mu in range(4):
        if sum(p[mu] for p in point[:incoming]) != sum(p[mu] for p in point[incoming:]):
            raise AssertionError("momentum conservation violated")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--process", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--mu-r", nargs="+", default=["1000", "700", "1400"])
    parser.add_argument("--cut", default="1/1000", help="minimal 2|p_i.p_j|/s")
    parser.add_argument("--half-scale-point", action="store_true",
                        help="append a copy of point 1 at half its scale (mu_R = mu_0/2)")
    parser.add_argument("--scale-window", nargs=2, default=["800", "4000"], metavar=("LOW", "HIGH"),
                        help="sqrt(s) window of the points (default 800 4000)")
    args = parser.parse_args()
    import required_spec

    spec = required_spec.load_process(args.spec or required_spec.DEFAULT_SPEC, args.process)
    masses = required_spec.masses(spec)
    values = {**spec["scheme_data"]["parameters"], **spec["scheme_data"]["fixed_parameters"]}
    resonances = [values[name] for name in ("MZ", "MW", "MH", "MT") if name in values and F(values[name])]
    window = (F(args.scale_window[0]), F(args.scale_window[1]))
    points = generate(masses, args.count, F(args.cut), resonances, scale_window=window, pdgs=spec["pdgs"])
    mu_r = list(itertools.islice(itertools.cycle(args.mu_r), args.count))
    if args.half_scale_point:
        points.append(points[0])
        mu_r.append(decimal_string(F(mu_r[0]) / 2))
    report = {
        "process_name": spec["name"],
        "pdgs": spec["pdgs"],
        "masses": [decimal_string(m) for m in masses],
        "construction": "tools/loop_reference/exact_points.py: deterministic exact terminating-decimal "
        "sequential construction; mass shells and momentum conservation exact",
        **({"scale_window": args.scale_window} if window != (F(800), F(4000)) else {}),
        "cuts": {"min_2pipj_over_s": args.cut, "resonance_exclusion": "|s_ij-M^2| >= M^2/10 for final particle-antiparticle pairs",
                 "resonances": resonances},
        "sqrt_s": [decimal_string(p[0][0] + p[1][0]) for p in points],
        "points": [[[decimal_string(x) for x in leg] for leg in point] for point in points],
        "mu_r": mu_r,
    }
    if args.half_scale_point:
        report["scale_variation"] = {"point": len(points), "copy_of_point": 1, "mu_r_factor": "1/2"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
