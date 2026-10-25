# SPDX-License-Identifier: 0BSD
"""Compile explicit gauge/Goldstone partner contracts from normalized UFO IR.

A UFO-declared Goldstone is paired with the unique propagating massive vector
of its colour, quantum numbers and resolved mass. Its disposition follows the
compiled model's gauge (``contracts.ModelGauge``):

* ``unitary`` (the default): the vector's standard propagator is the
  unitary-gauge one and absorbs the Goldstone (``absorbed``), whose vertices
  leave the tree model;
* ``feynman`` ('t Hooft-Feynman gauge, xi = 1): the vector propagates with
  -i g / (p^2 - M^2 + i M W) and the Goldstone is an explicit field of the
  tree model, with the vector's mass (``explicit``, naming the vector). Every
  massive vector with a standard propagator then needs exactly one Goldstone
  partner: without it the Feynman-gauge propagator misses the vector's
  longitudinal mode, so the compile is refused.

A vector with a model-supplied propagator keeps it in both gauges, and its
Goldstone stays a field of the tree model (``model-supplied``). A Goldstone
without a massive partner is ``explicit`` with no vector.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .contracts import (
    MODEL_GAUGES,
    CompiledGoldstonePartnerRecord,
    CompiledParameterRecord,
    CompiledParticleRecord,
    CompiledPropagatorRecord,
    ModelGauge,
    _goldstone_vector_quantum_contract_matches,
    _resolved_mass_expression,
)


def compile_goldstone_partner_records(
    particles: Sequence[CompiledParticleRecord],
    parameters: Sequence[CompiledParameterRecord],
    propagators: Sequence[CompiledPropagatorRecord],
    *,
    gauge: ModelGauge = "unitary",
) -> tuple[CompiledGoldstonePartnerRecord, ...]:
    """Resolve every declared Goldstone once, before executable model loading."""

    if gauge not in MODEL_GAUGES:
        raise ValueError(f"unsupported model gauge {gauge!r}")
    parameter_by_name: Mapping[str, CompiledParameterRecord] = {
        parameter.name: parameter for parameter in parameters
    }
    propagator_by_particle = {
        propagator.particle: propagator for propagator in propagators
    }
    vectors = tuple(
        particle
        for particle in particles
        if particle.spin == 3
        and particle.wavefunction_family == "vector"
        and particle.propagating
        and not particle.goldstoneboson
    )
    records: list[CompiledGoldstonePartnerRecord] = []
    for goldstone in particles:
        if not goldstone.goldstoneboson:
            continue
        if goldstone.spin != 1 or not goldstone.propagating:
            raise ValueError(
                f"Goldstone {goldstone.name!r} must be a propagating scalar"
            )
        candidates = tuple(
            vector
            for vector in vectors
            if _goldstone_vector_quantum_contract_matches(
                goldstone,
                vector,
                parameter_by_name,
            )
            and _resolved_mass_expression(vector, parameter_by_name) != "0"
        )
        if len(candidates) > 1:
            raise ValueError(
                f"Goldstone {goldstone.name!r} ambiguously matches vectors "
                f"{[vector.name for vector in candidates]!r}"
            )
        mass_expression = _resolved_mass_expression(
            goldstone,
            parameter_by_name,
        )
        if not candidates:
            records.append(
                CompiledGoldstonePartnerRecord(
                    goldstone=goldstone.name,
                    vector=None,
                    policy="explicit",
                    mass_expression=mass_expression,
                )
            )
            continue
        vector = candidates[0]
        propagator = propagator_by_particle.get(vector.name)
        custom = propagator is not None and propagator.custom
        records.append(
            CompiledGoldstonePartnerRecord(
                goldstone=goldstone.name,
                vector=vector.name,
                policy=(
                    "model-supplied"
                    if custom
                    else "explicit"
                    if gauge == "feynman"
                    else "absorbed"
                ),
                mass_expression=mass_expression,
            )
        )
    if gauge == "feynman":
        _require_feynman_partners(
            vectors, records, parameter_by_name, propagator_by_particle
        )
    return tuple(records)


def _require_feynman_partners(
    vectors: Sequence[CompiledParticleRecord],
    records: Sequence[CompiledGoldstonePartnerRecord],
    parameters: Mapping[str, CompiledParameterRecord],
    propagators: Mapping[str, CompiledPropagatorRecord],
) -> None:
    """Every massive vector with a standard propagator has exactly one
    explicit Goldstone partner. A charge-conjugate pair is one field: a vector
    without a record of its own is paired through its antiparticle (a UFO
    whose antiparticle records drop the Goldstone flag, as the G- of the
    loop_qcd_qed_sm models; that G- is then an ordinary propagating scalar,
    which is what an explicit Goldstone is)."""

    partners: dict[str, list[str]] = {}
    for record in records:
        if record.policy == "explicit" and record.vector is not None:
            partners.setdefault(record.vector, []).append(record.goldstone)
    for vector in vectors:
        propagator = propagators.get(vector.name)
        if propagator is not None and propagator.custom:
            continue
        if _resolved_mass_expression(vector, parameters) == "0":
            continue
        found = partners.get(vector.name) or partners.get(vector.antiname, [])
        if len(found) != 1:
            raise ValueError(
                f"'t Hooft-Feynman gauge needs exactly one Goldstone partner of "
                f"the massive vector {vector.name!r}, found {sorted(found)!r}"
            )


__all__ = ["compile_goldstone_partner_records"]
