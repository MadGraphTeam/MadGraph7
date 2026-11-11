# SPDX-License-Identifier: 0BSD
"""Flat (RAMBO) phase-space points for an artifact's process.

``pyamplicol profile --stability`` (and any command that needs points when no
momenta file is given) samples the process's phase space uniformly with the
product's RAMBO generator (``generation.phase_space.massive_rambo_final_state``):
two beams along +-z in the centre-of-mass frame at ``sqrt_s``, the final state
flat. The external masses are read from the artifact's own validation point
(``processes/<id>/validation-momenta.json``, on shell in the model's masses at
generation): m^2 = E^2 - |p|^2 in exact arithmetic of the stored decimal
texts, zero when it lies within the rounding of those binary64 components
(``external_masses``). The default energy is the validation point's own
sqrt(s), the generation convention of the process.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

# binary64's unit roundoff.
_UNIT_ROUNDOFF = Fraction(1, 2**53)


@dataclass(frozen=True, slots=True)
class ProcessKinematics:
    """What a flat point of a process needs: the PDG codes of its legs, the
    number of incoming ones, their masses and the generation's sqrt(s)."""

    pdgs: tuple[int, ...]
    incoming: int
    masses: tuple[float, ...]
    sqrt_s: float


def external_masses(point: Sequence[Sequence[object]]) -> tuple[float, ...]:
    """The masses of a stored on-shell point's legs (four-momenta as decimal
    texts or numbers). Each m^2 = E^2 - |p|^2 is formed exactly from the
    stored binary64 values; a mass whose square lies within the rounding of
    those values (each of E^2 and |p|^2 is known to a few units of binary64's
    roundoff: |m^2| <= 8 u (E^2 + |p|^2)) is zero."""

    masses = []
    for momentum in point:
        e, x, y, z = (Fraction(str(component)) for component in momentum)
        energy2, spatial2 = e * e, x * x + y * y + z * z
        square = energy2 - spatial2
        if abs(square) <= 8 * _UNIT_ROUNDOFF * (energy2 + spatial2):
            masses.append(0.0)
        elif square < 0:
            raise ValueError(f"the stored point has a spacelike leg ({momentum!r})")
        else:
            masses.append(math.sqrt(float(square)))
    return tuple(masses)


def artifact_kinematics(
    artifact: os.PathLike[str] | str, process_id: str
) -> ProcessKinematics:
    """The kinematics of ``process_id`` in an artifact directory, from its
    stored validation point."""

    path = (
        Path(os.fspath(artifact)) / "processes" / process_id / "validation-momenta.json"
    )
    try:
        record = json.loads(path.read_text())
    except OSError as error:
        raise ValueError(
            f"flat points need the artifact's validation point ({path}): {error}"
        ) from error
    points = record.get("points") or []
    if not record.get("available", True) or not points:
        raise ValueError(
            f"{process_id}: the artifact stores no validation point "
            f"({record.get('error')}); give the points with --momenta"
        )
    point = points[0]
    momenta = [entry["momentum"] for entry in point]
    pdgs = tuple(int(entry["pdg"]) for entry in point)
    masses = external_masses(momenta)
    incoming = _incoming_count(momenta)
    total = [
        sum(Fraction(str(momenta[i][k])) for i in range(incoming)) for k in range(4)
    ]
    s = total[0] ** 2 - total[1] ** 2 - total[2] ** 2 - total[3] ** 2
    return ProcessKinematics(pdgs, incoming, masses, math.sqrt(float(s)))


def _incoming_count(momenta: Sequence[Sequence[object]]) -> int:
    """The number of incoming legs of a stored point: the leading legs whose
    momentum sum balances the remaining ones (one for a decay, two for a
    scattering)."""

    vectors = [[Fraction(str(c)) for c in momentum] for momentum in momenta]
    best = None
    for incoming in (1, 2):
        if incoming >= len(vectors):
            break
        imbalance = max(
            abs(
                sum(v[k] for v in vectors[:incoming])
                - sum(v[k] for v in vectors[incoming:])
            )
            for k in range(4)
        )
        if best is None or imbalance < best[0]:
            best = (imbalance, incoming)
    if best is None:
        raise ValueError("a point needs at least two legs")
    return best[1]


def flat_points(
    kinematics: ProcessKinematics,
    count: int,
    *,
    seed: int,
    sqrt_s: float | None = None,
) -> list[list[list[float]]]:
    """``count`` flat points (binary64 four-momenta per leg in the order of
    the process) at ``sqrt_s`` (default: the generation's), point k from the
    seed ``seed + k`` (deterministic, independent of ``count``)."""

    from pyamplicol.generation.phase_space import massive_rambo_final_state

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("flat points need a positive point count")
    incoming = kinematics.incoming
    initial = kinematics.masses[:incoming]
    final = kinematics.masses[incoming:]
    if incoming == 1:
        if sqrt_s is not None:
            raise ValueError("a decay has no free energy: sqrt(s) is its mass")
        energy = initial[0]
    else:
        energy = kinematics.sqrt_s if sqrt_s is None else float(sqrt_s)
    if not (
        math.isfinite(energy)
        and energy >= sum(initial)
        and energy > sum(final)
        and (incoming == 1 or energy > sum(initial))
    ):
        raise ValueError(
            f"sqrt(s) = {energy!r} must exceed the initial ({sum(initial)!r}) and "
            f"final ({sum(final)!r}) thresholds"
        )
    if incoming == 1:
        beams = [[energy, 0.0, 0.0, 0.0]]
    else:
        m1, m2 = initial
        e1 = (energy * energy + m1 * m1 - m2 * m2) / (2.0 * energy)
        e2 = energy - e1
        pz = math.sqrt(max(e1 * e1 - m1 * m1, 0.0))
        beams = [[e1, 0.0, 0.0, pz], [e2, 0.0, 0.0, -pz]]
    points = []
    for k in range(count):
        outgoing = massive_rambo_final_state(
            len(final), sqrt_s=energy, masses=final, seed=seed + k
        )
        points.append([list(beam) for beam in beams] + [list(p) for p in outgoing])
    return points


__all__ = [
    "ProcessKinematics",
    "artifact_kinematics",
    "external_masses",
    "flat_points",
]
