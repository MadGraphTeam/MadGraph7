# SPDX-License-Identifier: 0BSD
"""Cuts of the drawn phase-space points (``[evaluation.phase_space] pt_min``,
``delta_r_min``), applied by rejection.

The cuts act on the massless final-state legs an infrared singularity or a
detector can see: the coloured ones (partons), the electrically charged ones
(charged leptons) and the neutral colour-singlet vectors (photons). Massive
legs and massless neutral fermions or scalars (neutrinos) are not cut. Which
leg is what comes from the model's quantum numbers, never from a list of
particles.

A point is the first of its own attempts that passes: attempt 0 is the
point's seed, so a point that passes the cuts is the point drawn without
them, and attempt k > 0 a seed derived from the point's seed and k. A point
therefore depends on its own seed alone, not on how many are drawn.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

_T = TypeVar("_T")

# The attempts of a point when the settings name none.
DEFAULT_ATTEMPTS = 100_000


class CutsError(ValueError):
    """Cuts that no point of the process can pass, or that none did."""


@dataclass(frozen=True, slots=True)
class PhaseSpaceCuts:
    """``pt_min``: every cut leg's transverse momentum exceeds it;
    ``delta_r_min``: the distance (d eta^2 + d phi^2)^(1/2) of every pair of
    cut legs does. None or 0: no such cut. ``attempts``: the attempts per
    point before the cuts are reported as too tight (None:
    ``DEFAULT_ATTEMPTS``)."""

    pt_min: float | None = None
    delta_r_min: float | None = None
    attempts: int | None = None

    @property
    def active(self) -> bool:
        return bool(self.pt_min) or bool(self.delta_r_min)

    def over(self, recorded: PhaseSpaceCuts) -> PhaseSpaceCuts:
        """These cuts, each value not given here (None) taken from
        ``recorded``; a 0 given switches the recorded cut off."""

        return PhaseSpaceCuts(
            recorded.pt_min if self.pt_min is None else self.pt_min,
            recorded.delta_r_min if self.delta_r_min is None else self.delta_r_min,
            recorded.attempts if self.attempts is None else self.attempts,
        )

    def record(self) -> dict[str, float | None]:
        return {
            "pt_min": self.pt_min or None,
            "delta_r_min": self.delta_r_min or None,
        }

    def described(self) -> list[str]:
        """The cuts set, in words (none: empty)."""

        return [
            f"{name} > {value:g}"
            for name, value in (("pT", self.pt_min), ("Delta R", self.delta_r_min))
            if value
        ]

    def accepted(self, momenta: Any, legs: Sequence[int]) -> np.ndarray:
        """Which of ``momenta`` (point, leg, (E, px, py, pz); beams along
        +-z) pass the cuts on their ``legs``."""

        points = np.asarray(momenta, dtype=np.float64)[:, list(legs), :]
        px, py, pz = (points[..., i] for i in range(1, 4))
        ok = np.ones(len(points), dtype=bool)
        pt = np.hypot(px, py)
        if self.pt_min:
            ok &= (pt > self.pt_min).all(axis=1)
        if self.delta_r_min:
            with np.errstate(divide="ignore", invalid="ignore"):
                # Along the beam the pseudorapidity is infinite: a pair of
                # such legs has no distance (nan) and fails.
                eta = np.arcsinh(pz / pt)
                phi = np.arctan2(py, px)
                for i in range(len(legs)):
                    for j in range(i):
                        dphi = np.abs(phi[:, i] - phi[:, j])
                        dphi = np.minimum(dphi, 2.0 * np.pi - dphi)
                        distance = np.hypot(eta[:, i] - eta[:, j], dphi)
                        ok &= distance > self.delta_r_min
        return ok

    def require_possible(self, legs: int, sqrt_s: float) -> None:
        """Refuse cuts no point can pass: the cut legs' transverse momenta
        sum to at most their energies, so to less than ``sqrt_s``."""

        if self.pt_min and legs and legs * self.pt_min >= sqrt_s:
            raise CutsError(
                f"evaluation.phase_space.pt_min = {self.pt_min} on {legs} "
                f"final-state leg(s) cannot be met at sqrt(s) = {sqrt_s}"
            )


def attempt_seed(seed: int, attempt: int) -> int:
    """The seed of a point's ``attempt`` (0: the point's own)."""

    if attempt == 0:
        return seed
    digest = hashlib.blake2b(f"{seed}:{attempt}".encode("ascii"), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


def is_cut_particle(*, spin: int, color: int, charge: float) -> bool:
    """Whether the cuts act on a massless leg of these quantum numbers (the
    compiled model's: ``spin`` 2 s + 1, ``color`` the representation):
    coloured, electrically charged or a vector."""

    return abs(color) != 1 or charge != 0.0 or spin == 3


def draw_accepted(
    seeds: Sequence[int],
    draw: Callable[[list[int]], Sequence[_T]],
    momenta: Callable[[_T], Any],
    cuts: PhaseSpaceCuts,
    legs: Sequence[int],
) -> list[_T]:
    """The point of every seed: ``draw(attempt seeds)`` gives a candidate per
    seed, the first whose ``momenta(candidate)`` pass the cuts on ``legs``
    is the point."""

    found: dict[int, _T] = {}
    pending = list(range(len(seeds)))
    attempts = DEFAULT_ATTEMPTS if cuts.attempts is None else cuts.attempts
    for attempt in range(attempts):
        if not pending:
            break
        drawn = draw([attempt_seed(seeds[i], attempt) for i in pending])
        ok = cuts.accepted([momenta(candidate) for candidate in drawn], legs)
        found.update(
            (i, candidate)
            for i, candidate, keep in zip(pending, drawn, ok, strict=True)
            if keep
        )
        pending = [i for i, keep in zip(pending, ok, strict=True) if not keep]
    if pending:
        raise CutsError(
            f"no phase-space point within the cuts ({cuts.record()}) in "
            f"{attempts} attempts (seed {seeds[pending[0]]}); loosen them or "
            "raise evaluation.phase_space.cut_attempts"
        )
    return [found[i] for i in range(len(seeds))]


__all__ = [
    "DEFAULT_ATTEMPTS",
    "CutsError",
    "PhaseSpaceCuts",
    "attempt_seed",
    "draw_accepted",
    "is_cut_particle",
]
