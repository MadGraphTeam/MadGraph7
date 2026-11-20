# SPDX-License-Identifier: 0BSD
"""Phase-space points.

``flat_final_state`` is the product's flat phase-space generator: every
random phase-space point (``evaluate`` and ``profile`` without a momenta
file, the loop profile's stability points, the reference and benchmark
tools) is drawn by it.

``massive_rambo_final_state`` is the artifact validation-point convention
(classic RAMBO of Kleiss, Stirling and Ellis with the massive rescaling, no
weight): the generation's validation and probe points of every artifact are
drawn by it and stored, digested and compared, so it is kept unchanged. It
is not a sampler; use ``flat_final_state`` for random points.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from .._internal.physics.types import FourMomentum, NativeEvaluationError

_SEED_BOUND = 2**64


@dataclass(frozen=True, slots=True)
class FlatFinalState:
    """A final state of ``flat_final_state``: its ``momenta`` (E, px, py,
    pz) in the rest frame of the total momentum (sqrt(s), 0, 0, 0) and its
    phase-space ``weight`` relative to the flat massless phase space of the
    same multiplicity at the same sqrt(s) (exactly 1 when every mass is
    zero; for massive final states the Jacobian of the mass mapping, whose
    mean is the ratio of the massive to the massless phase-space volume).
    Floats in binary64; in double-double and arbitrary precision the exact
    ``Fraction`` of every carrier value."""

    momenta: tuple[tuple[Any, Any, Any, Any], ...]
    weight: Any


def _native() -> Any:
    from pyamplicol.loops.native import load_loop_native

    return load_loop_native()


def _seed(seed: int | random.Random) -> int:
    if isinstance(seed, random.Random):
        return seed.getrandbits(64)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("a phase-space seed is an integer or a random.Random")
    if not 0 <= seed < _SEED_BOUND:
        raise ValueError("a phase-space seed is an integer in [0, 2^64)")
    return seed


def _exact_text(value: Any) -> str:
    return str(Fraction(str(value)) if isinstance(value, float) else Fraction(value))


def flat_final_states(
    masses: Sequence[Any],
    *,
    sqrt_s: Any,
    seeds: Sequence[int | random.Random],
    arithmetic: str = "double",
    digits: int | None = None,
) -> list[FlatFinalState]:
    """One flat final state per seed of ``seeds`` (``flat_final_state``):
    one native call for the whole list."""

    native = _native()
    if arithmetic == "double":
        drawn = native.flat_phase_space_f64(
            [float(m) for m in masses], float(sqrt_s), [_seed(s) for s in seeds]
        )
        return [
            FlatFinalState(tuple(tuple(leg) for leg in legs), weight)
            for legs, weight in drawn
        ]
    drawn = native.flat_phase_space(
        [_exact_text(m) for m in masses],
        _exact_text(sqrt_s),
        [_seed(s) for s in seeds],
        arithmetic,
        digits,
    )
    return [
        FlatFinalState(
            tuple(tuple(Fraction(x) for x in leg) for leg in legs), Fraction(weight)
        )
        for legs, weight in drawn
    ]


def flat_final_state(
    masses: Sequence[Any],
    *,
    sqrt_s: Any,
    seed: int | random.Random,
    arithmetic: str = "double",
    digits: int | None = None,
) -> FlatFinalState:
    """A final state of ``len(masses)`` >= 2 particles flat in Lorentz-
    invariant phase space at total momentum (``sqrt_s``, 0, 0, 0): RAMBO on
    diet (S. Plaetzer, arXiv:1308.2922), drawn natively
    (``rusticol_core::phase_space``, the one implementation; its module
    documentation states the construction): massless splittings with the
    paper's intermediate-mass equation and isotropic directions (unit
    weight), massive final states through the paper's mass mapping with
    its Jacobian as ``weight`` (reported, not applied).

    ``arithmetic`` is the carrier the point is constructed in: "double"
    (binary64 floats), "double-double" or "arbitrary" at ``digits`` decimal
    digits (exact ``Fraction`` values of the carrier's components); every
    mass shell and the momentum conservation hold to that carrier's
    rounding. ``seed`` is an integer in [0, 2^64) or a ``random.Random``
    (one 64-bit draw from it). The random numbers are drawn from the seed
    independently of the carrier, so a seed gives the same point in every
    carrier up to rounding, and binary64 points are reproducible bit for bit
    (no transcendental function is evaluated). A continuous distribution:
    no planar, collinear or soft configuration is drawn with nonzero
    probability."""

    (point,) = flat_final_states(
        masses, sqrt_s=sqrt_s, seeds=[seed], arithmetic=arithmetic, digits=digits
    )
    return point


def massive_rambo_final_state(
    multiplicity: int,
    *,
    sqrt_s: float,
    masses: Sequence[float],
    seed: int,
) -> tuple[FourMomentum, ...]:
    """Generate a deterministic massive RAMBO final state: the artifact
    validation-point convention (module docstring), kept unchanged because
    artifacts store and digest its points.

    It returns momenta in the centre-of-mass frame with total four-momentum
    ``(sqrt_s, 0, 0, 0)``: classic massless RAMBO rescaled to the masses (no
    weight). Random phase-space points come from ``flat_final_state``.
    """

    return _massive_rambo_final_state(
        multiplicity,
        sqrt_s=sqrt_s,
        masses=masses,
        rng=np.random.default_rng(seed),
    )


def _massive_rambo_final_state(
    multiplicity: int,
    *,
    sqrt_s: float,
    masses: Sequence[float],
    rng: np.random.Generator,
) -> tuple[FourMomentum, ...]:
    if multiplicity != len(masses):
        raise NativeEvaluationError("RAMBO mass list length mismatch")
    massless = _massless_rambo_final_state(multiplicity, sqrt_s=sqrt_s, rng=rng)
    spatial_norms = tuple(
        math.sqrt(momentum[1] ** 2 + momentum[2] ** 2 + momentum[3] ** 2)
        for momentum in massless
    )
    scale = _massive_spatial_scale(spatial_norms, masses, sqrt_s=sqrt_s)
    final: list[FourMomentum] = []
    for momentum, mass, spatial_norm in zip(
        massless,
        masses,
        spatial_norms,
        strict=True,
    ):
        px = scale * momentum[1]
        py = scale * momentum[2]
        pz = scale * momentum[3]
        energy = math.sqrt(mass * mass + (scale * spatial_norm) ** 2)
        final.append((energy, px, py, pz))
    return tuple(final)


def _massless_rambo_final_state(
    multiplicity: int,
    *,
    sqrt_s: float,
    rng: np.random.Generator,
) -> tuple[FourMomentum, ...]:
    if multiplicity < 2:
        raise NativeEvaluationError("RAMBO needs at least two final particles")
    raw: list[FourMomentum] = []
    for _ in range(multiplicity):
        r = rng.random(4)
        costheta = 2.0 * r[0] - 1.0
        sintheta = math.sqrt(max(0.0, 1.0 - costheta * costheta))
        phi = 2.0 * math.pi * r[1]
        energy = -math.log(max(r[2] * r[3], np.finfo(float).tiny))
        raw.append(
            (
                energy,
                energy * sintheta * math.cos(phi),
                energy * sintheta * math.sin(phi),
                energy * costheta,
            )
        )

    total = _sum_momenta(raw)
    invariant = _minkowski_square(total)
    if invariant <= 0.0:
        raise NativeEvaluationError("RAMBO generated a non-timelike total momentum")
    mass = math.sqrt(invariant)
    beta_to_rest = (
        -total[1] / total[0],
        -total[2] / total[0],
        -total[3] / total[0],
    )
    scale = sqrt_s / mass
    return tuple(
        (
            scale * boosted[0],
            scale * boosted[1],
            scale * boosted[2],
            scale * boosted[3],
        )
        for boosted in (_boost_from_rest(momentum, beta_to_rest) for momentum in raw)
    )


def _boost_from_rest(
    momentum: FourMomentum, beta: tuple[float, float, float]
) -> FourMomentum:
    beta2 = beta[0] ** 2 + beta[1] ** 2 + beta[2] ** 2
    if beta2 == 0.0:
        return momentum
    if beta2 >= 1.0:
        raise NativeEvaluationError("invalid canonical boost with beta >= 1")
    gamma = 1.0 / math.sqrt(1.0 - beta2)
    beta_dot_p = beta[0] * momentum[1] + beta[1] * momentum[2] + beta[2] * momentum[3]
    spatial_factor = ((gamma - 1.0) * beta_dot_p / beta2) + gamma * momentum[0]
    return (
        gamma * (momentum[0] + beta_dot_p),
        momentum[1] + spatial_factor * beta[0],
        momentum[2] + spatial_factor * beta[1],
        momentum[3] + spatial_factor * beta[2],
    )


def _massive_spatial_scale(
    spatial_norms: Sequence[float],
    masses: Sequence[float],
    *,
    sqrt_s: float,
) -> float:
    if sum(masses) >= sqrt_s:
        raise NativeEvaluationError("massive RAMBO point is below threshold")
    low = 0.0
    high = 1.0
    while _massive_energy_sum(spatial_norms, masses, high) < sqrt_s:
        high *= 2.0
    for _ in range(128):
        mid = 0.5 * (low + high)
        if _massive_energy_sum(spatial_norms, masses, mid) > sqrt_s:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def _massive_energy_sum(
    spatial_norms: Sequence[float],
    masses: Sequence[float],
    scale: float,
) -> float:
    return sum(
        math.sqrt(mass * mass + (scale * spatial_norm) ** 2)
        for spatial_norm, mass in zip(spatial_norms, masses, strict=True)
    )


def _sum_momenta(momenta: Sequence[FourMomentum]) -> FourMomentum:
    total = (0.0, 0.0, 0.0, 0.0)
    for momentum in momenta:
        total = (
            total[0] + momentum[0],
            total[1] + momentum[1],
            total[2] + momentum[2],
            total[3] + momentum[3],
        )
    return total


def _minkowski_square(momentum: FourMomentum) -> float:
    return (
        momentum[0] * momentum[0]
        - momentum[1] * momentum[1]
        - momentum[2] * momentum[2]
        - momentum[3] * momentum[3]
    )


__all__ = [
    "FlatFinalState",
    "flat_final_state",
    "flat_final_states",
    "massive_rambo_final_state",
]
