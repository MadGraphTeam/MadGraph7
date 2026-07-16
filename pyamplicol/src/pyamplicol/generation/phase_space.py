# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from .._internal.physics.types import (
    ExternalMomentum,
    FourMomentum,
    NativeEvaluationError,
)
from ..models import BuiltinSMModel
from ..processes.ir import build_process_ir


def _legacy_rambo_z_gluon_point(
    process: str,
    model: BuiltinSMModel,
    *,
    gluon_count: int,
    sqrt_s: float,
    seed: int,
) -> tuple[ExternalMomentum, ...]:
    """Generate a legacy deterministic RAMBO-style point for q q~ -> Z + n g.

    The massless construction follows the standard flat RAMBO mapping of
    Kleiss, Stirling and Ellis, CPC 40 (1986).  Massive final particles are
    then handled by a common spatial rescaling, as in the usual RAMBO massive
    extension: the centre-of-mass directions are kept fixed while the common
    scale is chosen so that the final-state energies add up to ``sqrt_s``.
    """

    if gluon_count < 1:
        raise NativeEvaluationError("RAMBO Z-gluon points need at least one gluon")
    z_mass = model.mass(23)
    if sqrt_s <= z_mass:
        raise NativeEvaluationError("sqrt(s) must be above the Z mass")

    pdgs = _physical_pdgs(process)
    expected = gluon_count + 3
    if len(pdgs) != expected:
        raise NativeEvaluationError(
            f"expected {expected} external particles for q q~ -> Z + {gluon_count} g"
        )

    final_pdgs = (*((21,) * gluon_count), 23)
    final_masses = tuple(0.0 if pdg == 21 else model.mass(pdg) for pdg in final_pdgs)
    if sum(final_masses) >= sqrt_s:
        raise NativeEvaluationError("final-state masses exceed sqrt(s)")

    rng = np.random.default_rng(seed)
    final_momenta = _massive_rambo_final_state(
        len(final_masses),
        sqrt_s=sqrt_s,
        masses=final_masses,
        rng=rng,
    )
    beam_energy = 0.5 * sqrt_s
    return (
        ExternalMomentum(pdgs[0], (beam_energy, 0.0, 0.0, beam_energy)),
        ExternalMomentum(pdgs[1], (beam_energy, 0.0, 0.0, -beam_energy)),
        *(
            ExternalMomentum(pdg, momentum)
            for pdg, momentum in zip(final_pdgs, final_momenta, strict=True)
        ),
    )


def massive_rambo_final_state(
    multiplicity: int,
    *,
    sqrt_s: float,
    masses: Sequence[float],
    seed: int,
) -> tuple[FourMomentum, ...]:
    """Generate a deterministic massive RAMBO final state.

    This process-independent helper is the public version of the RAMBO
    construction used by the generic DAG artifact writer.  It returns momenta
    in the centre-of-mass frame with total four-momentum
    ``(sqrt_s, 0, 0, 0)``.
    """

    return _massive_rambo_final_state(
        multiplicity,
        sqrt_s=sqrt_s,
        masses=masses,
        rng=np.random.default_rng(seed),
    )


def generic_validation_point(
    process: str,
    *,
    model: BuiltinSMModel | None = None,
    sqrt_s: float | None = None,
    seed: int = 101,
) -> tuple[ExternalMomentum, ...]:
    """Return deterministic process-generic two-beam validation kinematics."""

    model = model or BuiltinSMModel()
    ir = build_process_ir(process)
    initial_pdgs = tuple(int(pdg) for pdg in ir.initial_pdgs)
    final_pdgs = tuple(int(pdg) for pdg in ir.final_pdgs)
    if len(initial_pdgs) != 2:
        raise NativeEvaluationError(
            "generic validation momenta currently require a two-body initial state"
        )
    if not final_pdgs:
        raise NativeEvaluationError(
            "generic validation momenta require at least one final-state particle"
        )
    final_masses = tuple(float(model.mass(pdg)) for pdg in final_pdgs)
    threshold = sum(final_masses)
    if sqrt_s is None:
        sqrt_s = threshold if len(final_pdgs) == 1 else max(1000.0, threshold + 100.0)
    if sqrt_s < threshold:
        raise NativeEvaluationError("sqrt(s) is below the final-state mass threshold")
    if len(final_pdgs) == 1:
        if threshold <= 0.0:
            raise NativeEvaluationError(
                "no finite centre-of-mass validation point exists for one "
                "massless final state"
            )
        final_momenta = ((float(sqrt_s), 0.0, 0.0, 0.0),)
    else:
        final_momenta = massive_rambo_final_state(
            len(final_pdgs),
            sqrt_s=float(sqrt_s),
            masses=final_masses,
            seed=seed,
        )
    beam_energy = 0.5 * float(sqrt_s)
    return (
        ExternalMomentum(initial_pdgs[0], (beam_energy, 0.0, 0.0, beam_energy)),
        ExternalMomentum(initial_pdgs[1], (beam_energy, 0.0, 0.0, -beam_energy)),
        *(
            ExternalMomentum(pdg, momentum)
            for pdg, momentum in zip(final_pdgs, final_momenta, strict=True)
        ),
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


def _physical_pdgs(process: str) -> tuple[int, ...]:
    ir = build_process_ir(process)
    return (*ir.initial_pdgs, *ir.final_pdgs)


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


__all__ = ["generic_validation_point", "massive_rambo_final_state"]
