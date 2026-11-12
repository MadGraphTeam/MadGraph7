# SPDX-License-Identifier: 0BSD
"""Flat phase-space points for ``evaluate`` and ``profile`` without a
momenta file (``[evaluation.phase_space]``).

The points are RAMBO-flat (``generation.phase_space``, the product's flat
generator): a 2 -> n scattering in its centre-of-mass frame with the beams
along +-z, a 1 -> n decay in the rest frame of the decaying particle, a
2 -> 1 process at its single point. Point i is drawn from seed + i, so a
point does not depend on how many are drawn.

Nothing is read from the model: the external masses are those of the
artifact's own validation point (``processes/<id>/validation-momenta.json``,
written at generation with the generation's masses), m^2 = E^2 - |p|^2 in
exact decimal arithmetic on its stored digits. A square below the rounding
of the stored binary64 components (64 epsilon E^2) is a massless leg. The
default energy is the validation point's (its initial state's invariant
mass), so it is above threshold for any process the artifact holds.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path

from pyamplicol.config import ConfigurationError

# The stored components are binary64 values: a squared mass below this many
# roundings of E^2 is a massless leg.
_MASSLESS_ROUNDINGS = 64


@dataclass(frozen=True, slots=True)
class FlatPoints:
    """``points`` (per point the four-momenta of the legs, in the artifact's
    leg order), drawn at ``sqrt_s`` from ``seed`` with ``masses``."""

    points: tuple[tuple[tuple[float, float, float, float], ...], ...]
    sqrt_s: float
    seed: int
    masses: tuple[float, ...]
    initial: int
    process_id: str

    def as_inputs(self, *, exact: bool) -> list[list[list[object]]]:
        """The points as evaluation inputs: decimal numbers (exact, a loop
        runtime's inputs) or floats."""

        convert = (lambda x: Decimal(repr(x))) if exact else float
        return [
            [[convert(component) for component in momentum] for momentum in point]
            for point in self.points
        ]

    def mu_r2(self) -> list[str]:
        """Per point the squared invariant mass of its initial state: the
        default renormalisation scale (as ``profile`` and ``check``)."""

        return [repr(self.sqrt_s * self.sqrt_s) for _ in self.points]


def _process_id(artifact: Path, process: str | None) -> str:
    try:
        manifest = json.loads((artifact / "artifact.json").read_text())
    except (OSError, ValueError) as exc:
        raise ConfigurationError(f"cannot read the artifact {artifact}: {exc}") from exc
    ids = [str(entry["id"]) for entry in manifest.get("processes", ())]
    if process is None:
        default = manifest.get("default_process_id")
        if default is not None:
            return str(default)
        if len(ids) == 1:
            return ids[0]
        raise ConfigurationError(
            f"the artifact {artifact} holds several processes ({', '.join(ids)}); "
            "select one with evaluation.process (--process) for its flat points"
        )
    for entry in manifest.get("processes", ()):
        if process == entry["id"] or process in entry.get("aliases", ()):
            return str(entry["id"])
    return process


def _payload_text(artifact: Path, relative: str) -> str:
    """A declared payload's text, stored as a file or packed in payloads.pacbin
    (``generation.pack_payloads``)."""

    path = artifact / relative
    if path.is_file():
        return path.read_text()
    from pyamplicol.api.errors import ArtifactError
    from pyamplicol.artifacts.manifest import ArtifactPayloads, load_manifest

    try:
        payloads = ArtifactPayloads(load_manifest(artifact, verify_payloads=False))
        return payloads.read_text(relative)
    except ArtifactError as exc:
        raise OSError(str(exc)) from exc


def _validation_point(artifact: Path, process_id: str) -> list[tuple[Decimal, ...]]:
    path = artifact / "processes" / process_id / "validation-momenta.json"
    try:
        record = json.loads(
            _payload_text(artifact, path.relative_to(artifact).as_posix())
        )
    except (OSError, ValueError) as exc:
        raise ConfigurationError(
            f"no momenta file was given and the artifact has no readable "
            f"validation point to take the external masses from ({path}: {exc}); "
            "give evaluation.momenta (--momenta)"
        ) from exc
    if not record.get("available") or not record.get("points"):
        raise ConfigurationError(
            f"no momenta file was given and the artifact's validation point is "
            f"unavailable ({record.get('error')}); give evaluation.momenta"
        )
    return [
        tuple(Decimal(str(value)) for value in entry["momentum"])
        for entry in record["points"][0]
    ]


def _mass(momentum: tuple[Decimal, ...]) -> float:
    with localcontext() as context:
        context.prec = 60
        energy, px, py, pz = momentum
        square = energy * energy - px * px - py * py - pz * pz
        rounding = _MASSLESS_ROUNDINGS * Decimal(sys.float_info.epsilon) * energy**2
        if abs(square) <= rounding:
            return 0.0
        if square < 0:
            raise ConfigurationError(
                "the artifact's validation point has a spacelike external leg"
            )
        return float(square.sqrt())


def _initial_count(artifact: Path, process_id: str, legs: int) -> int:
    physics = artifact / "processes" / process_id / "physics.json"
    try:
        record = json.loads(
            _payload_text(artifact, physics.relative_to(artifact).as_posix())
        )
        states = record.get("state_legs") or record.get("metadata", {}).get(
            "state_legs"
        )
        if states:
            return sum(1 for state in states if str(state).startswith("in"))
    except (OSError, ValueError):
        pass
    # The validation point's leg order: initial legs first; their energies
    # sum to the final ones'.
    raise ConfigurationError(
        f"cannot read the initial legs of {process_id!r} from {physics}; "
        f"give evaluation.momenta ({legs} legs per point)"
    )


def draw_flat_points(
    artifact: Path,
    *,
    process: str | None,
    points: int,
    seed: int,
    sqrt_s: float | None,
) -> FlatPoints:
    """``points`` flat phase-space points of the artifact's process
    (``process``: an id or alias, None the default process) from ``seed``
    at ``sqrt_s`` (None: the validation point's energy; a decay or a 2 -> 1
    process takes its mass and refuses another value)."""

    from pyamplicol.generation.phase_space import massive_rambo_final_state

    process_id = _process_id(artifact, process)
    validation = _validation_point(artifact, process_id)
    masses = tuple(_mass(momentum) for momentum in validation)
    initial = _initial_count_from_point(artifact, process_id, validation)
    final = len(validation) - initial
    incoming = validation[:initial]
    with localcontext() as context:
        context.prec = 60
        total = [sum(momentum[i] for momentum in incoming) for i in range(4)]
        invariant = total[0] ** 2 - total[1] ** 2 - total[2] ** 2 - total[3] ** 2
        own = float(invariant.sqrt())
    fixed_energy = initial == 1 or final == 1
    if sqrt_s is None:
        energy = masses[0] if initial == 1 else own
    elif fixed_energy:
        expected = masses[0] if initial == 1 else masses[-1]
        if not math.isclose(sqrt_s, expected, rel_tol=1e-12):
            raise ConfigurationError(
                f"evaluation.phase_space.sqrt_s = {sqrt_s} for {process_id!r}, "
                f"whose energy is fixed by its masses ({expected}); remove it"
            )
        energy = expected
    else:
        energy = float(sqrt_s)
    if initial == 2 and energy <= masses[0] + masses[1]:
        raise ConfigurationError(
            f"evaluation.phase_space.sqrt_s = {energy} is below the threshold "
            f"of the initial state ({masses[0]} + {masses[1]})"
        )
    if final > 1 and energy <= sum(masses[initial:]):
        raise ConfigurationError(
            f"evaluation.phase_space.sqrt_s = {energy} is below the threshold "
            f"of the final state (sum of masses {sum(masses[initial:])})"
        )
    drawn = []
    for index in range(points):
        if initial == 1:
            legs = [(energy, 0.0, 0.0, 0.0)]
        else:
            m1, m2 = masses[0], masses[1]
            e1 = (energy * energy + m1 * m1 - m2 * m2) / (2.0 * energy)
            e2 = energy - e1
            momentum = math.sqrt(max(0.0, e1 * e1 - m1 * m1))
            legs = [(e1, 0.0, 0.0, momentum), (e2, 0.0, 0.0, -momentum)]
        if final == 1:
            legs.append((energy, 0.0, 0.0, 0.0))
        else:
            legs.extend(
                massive_rambo_final_state(
                    final,
                    sqrt_s=energy,
                    masses=masses[initial:],
                    seed=seed + index,
                )
            )
        drawn.append(tuple(tuple(float(x) for x in leg) for leg in legs))
    return FlatPoints(
        tuple(drawn), float(energy), int(seed), masses, initial, process_id
    )


def _initial_count_from_point(
    artifact: Path, process_id: str, validation: list[tuple[Decimal, ...]]
) -> int:
    """The number of initial legs (1 or 2): from the artifact's manifest
    (the process expression's arrow), else the physics record."""

    try:
        manifest = json.loads((artifact / "artifact.json").read_text())
        for entry in manifest.get("processes", ()):
            if str(entry["id"]) == process_id:
                expression = str(entry["expression"])
                if ">" in expression:
                    return len(expression.split(">", 1)[0].split())
    except (OSError, ValueError, KeyError):
        pass
    return _initial_count(artifact, process_id, len(validation))


__all__ = ["FlatPoints", "draw_flat_points"]
