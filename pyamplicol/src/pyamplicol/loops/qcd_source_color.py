# SPDX-License-Identifier: 0BSD
"""Explicit prepared loop_sm four-gluon continuation, without a model certificate."""

from __future__ import annotations

from dataclasses import replace

from .counterterms import prepare_counterterm_model
from .model import CovariantKernelCatalogue
from .qcd_r2_color import prepare_qcd_four_gluon_r2
from .source_color import SourceColorContinuation


def prepare_loop_sm_four_gluon_color(prepared, continuation):
    """Apply checked whole-rule input and the mandatory source ``lhv=1`` tie.

    ``G`` and ``lhv`` are explicit inputs of this preparation profile. The model
    display name has no effect. Every selected four-adjoint-vector R2 group must
    match the independent complete HV tensor at Nc=3, with actual loop species
    and alternatives retained. Other caller-declared kernels are unchanged and
    gain no certificate from this operation.
    """
    from symbolica import E, S

    if not isinstance(continuation, SourceColorContinuation):
        raise TypeError("four-gluon preparation requires source colour model data")
    if continuation.model is not prepared.model:
        raise ValueError("four-gluon continuation and prepared source owners differ")
    if continuation.scheme.dimensional_regularization != "HV":
        raise NotImplementedError("prepared four-gluon QCD continuation requires HV")
    ties = dict(continuation.scheme_parameter_values)
    if "lhv" in ties and ties["lhv"] != E("1"):
        raise ValueError("prepared four-gluon QCD continuation requires lhv=1")
    ties["lhv"] = E("1")
    tied = replace(continuation, scheme_parameter_values=ties)
    # This proof uses the same fixed map as all ordinary and CT scalar views.
    # An adjustable external lhv cannot pass merely because its default is one.
    tied.require_whole_rules((), prepared.structure.scalar_bindings((("lhv", 0),)))
    catalogue = CovariantKernelCatalogue(prepared.model, prepared.structure.native)
    particles = prepared.model.raw.collections["particles"]
    replacements = list(tied.whole_rule_continuations)
    keys = set(tied.kernel_keys)
    for rule in prepared.counterterms.rules.values():
        first = rule.entries[0]
        fields = tuple(particles[index]["fields"] for index in first.particle_ids)
        if (
            rule.kind != "R2"
            or len(fields) != 4
            or any(
                (particle["spin"], particle["color"]) != (3, 8) for particle in fields
            )
        ):
            continue
        if rule.whole_rule_continuation is not None:
            raise ValueError("four-gluon source group is already replaced")
        gauge = first.loop_particles and all(
            alternative == (first.particle_ids[0],)
            for alternative in first.loop_particles
        )
        replacement = prepare_qcd_four_gluon_r2(
            prepared.model,
            catalogue,
            rule.entries,
            tied.nc,
            gauge_coupling=S("UFO::G"),
            contribution="gauge" if gauge else "fundamental-dirac",
            scheme="HV",
        )
        replacements.append(replacement)
        keys.update(entry.key for entry in replacement.source_entries)
    completed = replace(
        tied, kernel_keys=frozenset(keys), whole_rule_continuations=tuple(replacements)
    )
    original = prepared.counterterms
    # Do not discard an alternative preparation path's representatives.
    if any(
        evidence is not None
        for rule in original.rules.values()
        for evidence in (*rule.basis_correspondences, *rule.tensor_representatives)
    ):
        raise NotImplementedError(
            "four-gluon source preparation requires raw UFO rules"
        )
    selected_keys = {
        entry.key for rule in original.rules.values() for entry in rule.entries
    } | set(original.unavailable_entries)
    rebuilt = prepare_counterterm_model(
        prepared.model,
        original.ordinary_model,
        entries=tuple(
            entry for entry in prepared.model.kernels if entry.key in selected_keys
        ),
        primitive_validations=tuple(
            evidence
            for rule in original.rules.values()
            for evidence in rule.primitive_validations
            if evidence is not None
        ),
        whole_rule_continuations=completed.whole_rule_continuations,
        vertex_spinor_frame=original.vertex_spinor_frame,
    )
    completed.require_whole_rules(
        rebuilt.rules.values(), prepared.structure.scalar_bindings((("lhv", 0),))
    )
    return replace(prepared, counterterms=rebuilt), completed
