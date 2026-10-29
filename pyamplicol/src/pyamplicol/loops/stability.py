# SPDX-License-Identifier: 0BSD
"""Optional empirical cross-precision verification of complete native controls.

Agreement between evaluations is evidence under explicit componentwise bounds,
not an independent certification of NinjaRS or the physical model. Ordinary
physical evaluators do not enable this additional work implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ArithmeticSpec:
    arithmetic: str
    precision: int | None = None

    @property
    def options(self):
        return {"arithmetic": self.arithmetic, "precision": self.precision}


def _validate_precision_sequence(
    backend, specs, relative_tolerance, absolute_tolerance
):
    previous_bits = 0
    for spec in specs:
        empty = backend.NativeBatch.from_decimal(
            [], 0, 1, arithmetic=spec.arithmetic, precision=spec.precision
        )
        if empty.precision_bits <= previous_bits:
            raise ValueError("verification arithmetic precision must strictly increase")
        previous_bits = empty.precision_bits
        empty.agreement_rows(empty, relative_tolerance, absolute_tolerance)


def validate_source_stability(backend, arithmetic, precision, policy):
    """Validate the existing public policy before a producer source is bound."""
    if policy is None:
        return
    from pyamplicol.api.virtual_results import VirtualStabilityPolicy

    if not isinstance(policy, VirtualStabilityPolicy):
        raise TypeError("source stability requires a VirtualStabilityPolicy")
    _validate_precision_sequence(
        backend,
        (ArithmeticSpec(arithmetic, precision), *policy.higher),
        policy.relative_tolerance,
        policy.absolute_tolerance,
    )
