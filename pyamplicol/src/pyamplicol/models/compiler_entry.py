# SPDX-License-Identifier: 0BSD
"""Entry point for compiling external UFO model IR."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from typing import Any

from .._internal.physics.symbols import ModelSymbolRegistry, symbols
from . import compiler_symbolica as _sym
from .compiler_auxiliary_components import reduce_contact_auxiliary_components
from .compiler_color_flow import (
    compile_lc_color_transition_terms,
    synthesize_fundamental_fierz_auxiliaries,
    synthesize_u1_projections,
)
from .compiler_contact_trees import (
    _compile_color_singlet_contact_trees,
    _compile_heft_colored_contact_trees,
    _deduplicate_contact_partials,
    _record_heft_contact_orbit_certificates,
)
from .compiler_contacts import (
    _build_contact_decomposition_proof,
    _contact_final_component_expressions,
    _contact_term_has_literal_color_singlet,
    _fuse_contact_finals,
    _record_contact_decomposition_proofs,
    _validated_contact_decomposition_proof,
    contact_pair_symmetry,
    projected_contact_partial_components,
)
from .compiler_contractions import compile_contraction_records
from .compiler_gauge import close_goldstone_flags, compile_goldstone_partner_records
from .compiler_kernels import (
    _canonical_product_text,
    _fuse_oriented_kernels,
    _is_compile_time_zero_parameter,
    _is_single_structure_constant,
    _lc_color_normalization_power,
    _oriented_component_expressions,
    _remap_kernel_symbols,
    _renamed_kernel_text,
    _replace_expression_symbols,
    _spin_axis_labels,
)
from .compiler_records import (
    _coupling,
    _mappings,
    _order,
    _parameter,
    _particle,
    _propagator,
    _resolve_coupling_records,
    _resolve_parameter_records,
    _sequence,
)
from .compiler_tensor_ordering import compile_tensor_ordering_metadata
from .contact_decomposition import (
    CompiledContactDecompositionSplit,
    CompiledContactOrbitStep,
    CompiledContactOrientationProof,
    compiled_contact_orbit_step,
)
from .contracts import (
    TREE_KERNEL_VARIANTS,
    CompiledModelIR,
    CompiledOrientedKernel,
    CompiledParameterRecord,
    CompiledParticleRecord,
    CompiledPropagatorRecord,
    CompiledVertexTerm,
    ModelGauge,
    TreeKernelVariant,
    compiled_particle_component_dimension,
)
from .tensors import (
    classify_trilinear_color_expression,
    normalize_color_expression,
    normalize_lorentz_expression,
    project_trilinear_color_expression,
)

_SUPPORTED_TRILINEAR_COLOR_PROJECTIONS = frozenset(
    {
        "singlet",
        "color-identity",
        "fundamental-generator",
        "adjoint-structure-constant",
    }
)
# The d-symbol couples to the U(1) part of a colour-flow gluon, which only
# fermion lines subtract: it is lowered only on the vertices whose gluon legs
# are projected onto SU(N) (``u1_projected_vertices``, the counterterm
# vertices of the counterterm closure), never on an ordinary model vertex.
_U1_PROJECTED_TRILINEAR_COLOR_PROJECTIONS = frozenset({"adjoint-symmetric-invariant"})


def compile_ufo_model_ir(
    model: Mapping[str, object],
    *,
    tree_kernel_variant: TreeKernelVariant = "transverse",
    u1_projected_vertices: frozenset[str] = frozenset(),
    gauge: ModelGauge = "unitary",
) -> CompiledModelIR:
    """Compile one UFO payload; ``tree_kernel_variant`` selects YM lowering.

    The variant changes only the component expressions of oriented kernels
    proven eligible for the transverse massless Yang-Mills replacement. Kernel
    kinds, particles and every other record are identical in both variants.

    ``gauge`` is the gauge of the massive vectors (``contracts.ModelGauge``):
    it decides the Goldstone partner records (``compiler_gauge``) and is
    recorded in the IR; every other record is the same in both gauges (the
    oriented kernels of Goldstone vertices exist in both; a unitary-gauge tree
    model leaves out those of absorbed Goldstones).

    ``u1_projected_vertices`` names vertex rules whose colour-flow stages get
    the U(1) projection of their gluon legs
    (``compiler_color_flow.synthesize_u1_projections``; the counterterm
    vertices of the counterterm closure, ``models.loading``).
    """

    if tree_kernel_variant not in TREE_KERNEL_VARIANTS:
        raise ValueError(f"unsupported tree kernel variant {tree_kernel_variant!r}")
    _sym._ensure_symbolica()
    model_name = str(model.get("name", "unnamed-model"))
    model_symbols = symbols.model(model_name)
    particles = close_goldstone_flags(
        tuple(_particle(item) for item in _mappings(model.get("particles")))
    )
    particle_by_name = {particle.name: particle for particle in particles}
    lorentz_by_name = {
        str(item["name"]): item for item in _mappings(model.get("lorentz_structures"))
    }
    parameter_records = tuple(
        _parameter(item, model_symbols=model_symbols)
        for item in _mappings(model.get("parameters"))
    )
    parameter_records = _resolve_parameter_records(parameter_records, model_symbols)
    coupling_records = tuple(
        _coupling(item, model_symbols=model_symbols)
        for item in _mappings(model.get("couplings"))
    )
    coupling_records = _resolve_coupling_records(
        coupling_records,
        parameter_records,
        model_symbols,
    )
    coupling_by_name = {coupling.name: coupling for coupling in coupling_records}
    terms: list[CompiledVertexTerm] = []
    for vertex in _mappings(model.get("vertex_rules")):
        particle_names = tuple(str(value) for value in _sequence(vertex["particles"]))
        try:
            vertex_particles = tuple(particle_by_name[name] for name in particle_names)
        except KeyError as exc:
            raise ValueError(
                f"vertex {vertex.get('name')} refers to unknown particle {exc.args[0]}"
            ) from exc
        colors = tuple(particle.color for particle in vertex_particles)
        color_sources = tuple(
            str(value) for value in _sequence(vertex["color_structures"])
        )
        lorentz_names = tuple(
            str(value) for value in _sequence(vertex["lorentz_structures"])
        )
        coupling_matrix = _sequence(vertex["couplings"])
        if len(coupling_matrix) != len(color_sources):
            raise ValueError(
                f"vertex {vertex.get('name')} coupling rows do not match "
                "color structures"
            )
        normalized_colors = tuple(
            normalize_color_expression(source, colors) for source in color_sources
        )
        normalized_lorentz = []
        for name in lorentz_names:
            try:
                lorentz = lorentz_by_name[name]
            except KeyError as exc:
                raise ValueError(
                    f"vertex {vertex.get('name')} refers to unknown Lorentz "
                    f"structure {name}"
                ) from exc
            normalized_lorentz.append(
                normalize_lorentz_expression(
                    str(lorentz["structure"]),
                    tuple(int(value) for value in _sequence(lorentz["spins"])),
                    model_symbols=model_symbols,
                )
            )
        for color_index, row_value in enumerate(coupling_matrix):
            row = _sequence(row_value)
            if len(row) != len(lorentz_names):
                raise ValueError(
                    f"vertex {vertex.get('name')} coupling columns do not match "
                    "Lorentz structures"
                )
            for lorentz_index, coupling_value in enumerate(row):
                if coupling_value is None:
                    continue
                coupling_name = str(coupling_value)
                try:
                    coupling = coupling_by_name[coupling_name]
                except KeyError as exc:
                    raise ValueError(
                        f"vertex {vertex.get('name')} refers to unknown coupling "
                        f"{coupling_name}"
                    ) from exc
                source_lorentz = lorentz_by_name[lorentz_names[lorentz_index]]
                terms.append(
                    CompiledVertexTerm(
                        id=len(terms),
                        vertex=str(vertex["name"]),
                        particles=particle_names,
                        color_index=color_index,
                        lorentz_index=lorentz_index,
                        color_source=color_sources[color_index],
                        color_expression=normalized_colors[color_index].expression,
                        lorentz_name=lorentz_names[lorentz_index],
                        lorentz_source=str(source_lorentz["structure"]),
                        lorentz_expression=normalized_lorentz[lorentz_index].expression,
                        coupling=coupling.name,
                        coupling_expression=coupling.resolved_expression,
                        coupling_orders=coupling.orders,
                        lc_color_normalization_power=(
                            _lc_color_normalization_power(color_sources[color_index])
                        ),
                    )
                )
    propagators = tuple(
        _propagator(item, particles) for item in _mappings(model.get("propagators"))
    )
    terms = list(
        _record_contact_decomposition_proofs(
            terms,
            particles,
            model_symbols=model_symbols,
        )
    )
    terms = list(
        _record_heft_contact_orbit_certificates(
            terms,
            particles,
            model_symbols=model_symbols,
        )
    )
    oriented_kernels = _compile_oriented_kernels(
        terms,
        particles,
        parameter_records,
        propagators,
        model_symbols,
        transverse_massless_yang_mills=tree_kernel_variant == "transverse",
    )
    contact_particles, contact_kernels = _compile_four_point_contact_kernels(
        terms,
        particles,
        start_kind=len(oriented_kernels),
        model_symbols=model_symbols,
    )
    contact_particles, contact_kernels = _deduplicate_contact_partials(
        contact_particles,
        contact_kernels,
        terms,
        model_symbols=model_symbols,
    )
    contact_kernels = _fuse_contact_finals(
        contact_kernels,
        terms,
        model_symbols=model_symbols,
    )
    particles = (*particles, *contact_particles)
    oriented_kernels = (*oriented_kernels, *contact_kernels)
    tree_start_kind = max((kernel.kind for kernel in oriented_kernels), default=-1) + 1
    tree_particles, tree_kernels = _compile_color_singlet_contact_trees(
        terms,
        particles,
        start_kind=tree_start_kind,
        model_symbols=model_symbols,
    )
    particles = (*particles, *tree_particles)
    oriented_kernels = (*oriented_kernels, *tree_kernels)
    heft_start_kind = max((kernel.kind for kernel in oriented_kernels), default=-1) + 1
    heft_particles, heft_kernels = _compile_heft_colored_contact_trees(
        terms,
        particles,
        start_kind=heft_start_kind,
        model_symbols=model_symbols,
    )
    particles = (*particles, *heft_particles)
    oriented_kernels = (*oriented_kernels, *heft_kernels)
    particles, oriented_kernels = reduce_contact_auxiliary_components(
        particles,
        oriented_kernels,
        auxiliary_particles=(*tree_particles, *heft_particles),
        model_symbols=model_symbols,
    )
    (
        annotated_terms,
        oriented_kernels,
        tensor_orderings,
        current_orderings,
    ) = compile_tensor_ordering_metadata(
        terms,
        particles,
        oriented_kernels,
        parameter_records,
        propagators,
    )
    terms = list(annotated_terms)
    oriented_kernels = _annotate_oriented_kernel_color_projections(
        oriented_kernels,
        particles,
        terms,
        u1_projected_vertices=frozenset(u1_projected_vertices),
    )
    particles, oriented_kernels, propagators = synthesize_fundamental_fierz_auxiliaries(
        particles,
        oriented_kernels,
        propagators,
        model_symbols=model_symbols,
    )
    if u1_projected_vertices:
        particles, oriented_kernels = synthesize_u1_projections(
            particles,
            oriented_kernels,
            projected_vertices=frozenset(u1_projected_vertices),
            model_symbols=model_symbols,
        )
    (
        annotated_terms,
        oriented_kernels,
        tensor_orderings,
        current_orderings,
    ) = compile_tensor_ordering_metadata(
        terms,
        particles,
        oriented_kernels,
        parameter_records,
        propagators,
    )
    terms = list(annotated_terms)
    oriented_kernels = _annotate_oriented_kernel_evaluation_equivalence(
        oriented_kernels,
        particles,
        terms,
        model_symbols,
    )
    direct_contractions, closure_contractions = compile_contraction_records(
        particles,
        parameter_records,
        propagators,
    )
    goldstone_partners = compile_goldstone_partner_records(
        particles,
        parameter_records,
        propagators,
        gauge=gauge,
    )
    return CompiledModelIR(
        name=model_name,
        orders=tuple(_order(item) for item in _mappings(model.get("orders"))),
        parameters=parameter_records,
        particles=particles,
        couplings=coupling_records,
        propagators=propagators,
        vertex_terms=tuple(terms),
        oriented_kernels=oriented_kernels,
        direct_contractions=direct_contractions,
        closure_contractions=closure_contractions,
        tensor_orderings=tensor_orderings,
        current_orderings=current_orderings,
        goldstone_partners=goldstone_partners,
        tree_kernel_variant=tree_kernel_variant,
        gauge=gauge,
    )


def _annotate_oriented_kernel_color_projections(
    kernels: Sequence[CompiledOrientedKernel],
    particles: Sequence[CompiledParticleRecord],
    terms: Sequence[CompiledVertexTerm],
    *,
    u1_projected_vertices: frozenset[str] = frozenset(),
) -> tuple[CompiledOrientedKernel, ...]:
    particle_by_name = {particle.name: particle for particle in particles}
    term_by_id = {term.id: term for term in terms}
    catalog = _contact_split_catalog(term_by_id)
    annotated: list[CompiledOrientedKernel] = []
    for kernel in kernels:
        term = term_by_id.get(kernel.term_id)
        oriented_representations = tuple(
            particle_by_name[name].color for name in kernel.particles
        )
        if term is not None and term.valence == 3 and kernel.vertex == term.vertex:
            representations = tuple(
                particle_by_name[name].color for name in term.particles
            )
            certified_structure, coefficient = project_trilinear_color_expression(
                term.color_expression,
                representations,
            )
            if (
                certified_structure not in _SUPPORTED_TRILINEAR_COLOR_PROJECTIONS
                and not (
                    certified_structure in _U1_PROJECTED_TRILINEAR_COLOR_PROJECTIONS
                    and term.vertex.split("::", 1)[0] in u1_projected_vertices
                )
            ):
                raise ValueError(
                    f"vertex {term.vertex!r} has unsupported trilinear color "
                    f"tensor {term.color_source!r}; pyAmpliCol could not prove it "
                    "as a singlet, identity, fundamental generator, or adjoint "
                    "structure constant"
                )
            structure, _ = classify_trilinear_color_expression(
                kernel.color_expression,
                kernel.color_source,
                oriented_representations,
                allow_source_fallback=True,
            )
            if structure not in {certified_structure, "generic-tensor"}:
                raise ValueError(
                    f"vertex {term.vertex!r} changed color-tensor family while "
                    "being oriented"
                )
            if structure == "generic-tensor":
                # Numeric prefactors and the dual representation carried by an
                # oriented result state can hide the family from the cheap
                # oriented recognizer.  The original trilinear term and the
                # compiler-owned source-leg permutation already certify it.
                structure = certified_structure
            if structure == "adjoint-structure-constant":
                # The projection is f in the term's leg order. An orientation
                # permutes the legs and f is totally antisymmetric, so its
                # colour-flow sign is the parity of the permutation (this
                # matters only for vertices whose f legs are distinct species).
                coefficient *= _permutation_parity(kernel.source_particle_legs)
            recurrence_structure = certified_structure
            recurrence_proof_source = "exact-trilinear-symbolica-projection-v1"
            recurrence_provenance = (("vertex-term-id", str(term.id)),)
            tensor_role_representations = tuple(
                representations[source_leg]
                for source_leg in kernel.source_particle_legs
            )
        else:
            # Contact fragments are emitted only after compiler_contacts has
            # serialized a complete decomposition proof. Their compact source
            # is compiler-owned, so textual recognition is safe at this point.
            structure, coefficient = classify_trilinear_color_expression(
                kernel.color_expression,
                kernel.color_source,
                oriented_representations,
                allow_source_fallback=True,
            )
            if structure == "generic-tensor":
                raise ValueError(
                    f"compiler-generated kernel {kernel.vertex!r} has no proven "
                    "color-flow projection"
                )
            contact_contract = _contact_recurrence_color_contract(
                kernel,
                term_by_id,
                catalog,
            )
            if contact_contract is None:
                recurrence_structure = structure
                recurrence_proof_source = "compiler-generated-color-projection-v1"
                recurrence_provenance = ()
            else:
                (
                    recurrence_structure,
                    recurrence_proof_source,
                    recurrence_provenance,
                ) = contact_contract
            tensor_role_representations = (
                oriented_representations[0],
                oriented_representations[1],
                _dual_color_representation(oriented_representations[2]),
            )
        projection = {
            "color_projection_coefficient": (
                float(coefficient.real),
                float(coefficient.imag),
            ),
            "lc_recurrence_color_rule_kind": recurrence_structure,
        }
        recurrence_kernel = replace(
            kernel,
            color_projection_structure=recurrence_structure,
            **projection,
        )
        annotated.append(
            replace(
                kernel,
                color_projection_structure=structure,
                lc_color_transition_terms=compile_lc_color_transition_terms(
                    recurrence_kernel,
                    oriented_representations,
                    proof_source=recurrence_proof_source,
                    provenance=recurrence_provenance,
                    tensor_role_representations=tensor_role_representations,
                ),
                **projection,
            )
        )
    return tuple(annotated)


def _permutation_parity(permutation: Sequence[int]) -> int:
    inversions = sum(
        1
        for index, value in enumerate(permutation)
        for later in permutation[index + 1 :]
        if value > later
    )
    return -1 if inversions % 2 else 1


def _dual_color_representation(representation: int) -> int:
    return -representation if abs(representation) == 3 else representation


# A contact split of the proof catalog: (catalog position, term, proof, split).
_CatalogSplit = tuple[int, CompiledVertexTerm, Any, Any]


def _contact_split_catalog(
    term_by_id: Mapping[int, CompiledVertexTerm],
) -> tuple[dict[str, list[_CatalogSplit]], dict[int, list[_CatalogSplit]]]:
    """The splits of every proven contact decomposition of ``term_by_id``,
    in catalog order (terms, then their splits), by the auxiliary each split
    owns and by its term (``_contact_recurrence_color_contract``)."""

    by_auxiliary: dict[str, list[_CatalogSplit]] = {}
    by_term: dict[int, list[_CatalogSplit]] = {}
    position = 0
    for term in term_by_id.values():
        proof = term.contact_decomposition_proof
        if proof is None or proof.status != "proven":
            continue
        for split in proof.splits:
            entry = (position, term, proof, split)
            position += 1
            auxiliary_name = f"__pyamplicol_contact_{term.id}_r{split.result_leg}"
            by_auxiliary.setdefault(auxiliary_name, []).append(entry)
            by_term.setdefault(term.id, []).append(entry)
    return by_auxiliary, by_term


def _contact_recurrence_color_contract(
    kernel: CompiledOrientedKernel,
    term_by_id: dict[int, CompiledVertexTerm],
    catalog: tuple[dict[str, list[_CatalogSplit]], dict[int, list[_CatalogSplit]]]
    | None = None,
) -> tuple[str, str, tuple[tuple[str, str], ...]] | None:
    is_partial = kernel.vertex.endswith("::contact-partial")
    is_final = kernel.vertex.endswith(("::contact-final", "::contact-final-fused"))
    if not is_partial and not is_final:
        # Balanced contact-tree lowering has its own compiler-owned singlet
        # auxiliaries and is not described by the four-point split proof.
        # Its trilinear color projection is authenticated by the generic
        # compiler-generated contract below.
        return None

    stage = "partial" if is_partial else "final"
    by_auxiliary, by_term = (
        _contact_split_catalog(term_by_id) if catalog is None else catalog
    )
    fused_term_ids = set(kernel.term_ids or (kernel.term_id,))
    # Contact-partial deduplication and final fusion may retain an auxiliary
    # owned by a term that is no longer the fused kernel's primary term ID.
    # Match the authenticated auxiliary name and source-leg orientation against
    # the complete proof catalog instead of trusting that incidental owner.
    owned = {
        entry[0]: entry
        for name in (
            (kernel.particles[2],) if stage == "partial" else kernel.particles[:2]
        )
        for entry in by_auxiliary.get(name, ())
    }
    owner_matches = [owned[position][1:] for position in sorted(owned)]
    relevant = (
        owned
        if stage == "partial"
        else {
            entry[0]: entry
            for term_id in fused_term_ids
            for entry in by_term.get(term_id, ())
        }
    )
    orientation_matches = []
    for position in sorted(relevant):
        _position, term, proof, split = relevant[position]
        for orientation in split.orientations:
            if orientation.stage != stage:
                continue
            expected_source_legs = (
                orientation.input_legs[0],
                orientation.input_legs[1],
                -1 if stage == "partial" else split.result_leg,
            )
            if kernel.source_particle_legs == expected_source_legs:
                orientation_matches.append((term, proof, split, orientation))
    if len(owner_matches) != 1 or not orientation_matches:
        raise ValueError(
            f"compiler-generated contact kernel {kernel.vertex!r} has "
            f"{len(owner_matches)} recurrence auxiliary owners and "
            f"{len(orientation_matches)} exact stage orientations"
        )
    term, proof, split = owner_matches[0]
    # Deduplicated partials may serve splits of different kinds whose stage
    # tensors agree (a closing identity of two adjoint legs, say); a final
    # takes its structure from its own terms' splits, not the auxiliary's
    # owner.
    candidates = [match[2] for match in orientation_matches]
    if stage == "partial":
        candidates.append(split)
    recurrence_structure = _contact_stage_structure(candidates[0], stage)
    if any(
        _contact_stage_structure(candidate, stage) != recurrence_structure
        for candidate in candidates
    ):
        raise ValueError(
            f"compiler-generated contact kernel {kernel.vertex!r} fuses "
            "incompatible recurrence color decompositions"
        )
    return (
        recurrence_structure,
        f"{proof.algorithm}-recurrence-{stage}-v{proof.algorithm_version}",
        (
            ("contact-decomposition-kind", split.decomposition_kind),
            ("contact-result-leg", str(split.result_leg)),
            ("contact-stage", stage),
            ("vertex-term-id", str(term.id)),
        ),
    )


def _contact_stage_structure(split: Any, stage: str) -> str:
    """The recurrence colour structure of one stage of a contact split."""

    kind = split.decomposition_kind
    if kind in ("two-structure-constants", "trilinear-identity", "identity-singlets"):
        # Each stage carries its own tensor: a structure constant or a
        # d-symbol (counterterm contacts, ``ct_color_flow``), an identity, or
        # nothing (a pair of colour singlets).
        source = (
            split.outer_color_source
            if stage == "partial"
            else (split.final_color_source)
        )
        if "Identity(" in source:
            return "color-identity"
        if "d(" in source:
            return "adjoint-symmetric-invariant"
        if "f(" in source:
            return "adjoint-structure-constant"
        return "singlet"
    return {
        "literal-color-singlet": "singlet",
        "two-identities": "color-identity",
    }[kind]


# Kernel-equivalence proofs of this process by the sha256 of their key
# (``_annotate_oriented_kernel_evaluation_equivalence``): a proof is a pure
# function of its key, so a compile reuses the proofs of earlier compiles,
# e.g. a counterterm-lowered model those of its base model's vertices and
# of the lowered models of other processes. Cleared beyond the bound.
_KERNEL_PROOFS: dict[str, tuple[Any, ...]] = {}
_KERNEL_PROOF_BOUND = 1 << 16
# The proof keys and the proofs a compile lacks are formed natively, in
# parallel (``rusticol_core::model_proofs`` through the loop extension's
# ``model_kernel_equivalence``; WP3.2 EWI-3R): byte-identical to the key loop
# of ``_annotate_oriented_kernel_evaluation_equivalence`` and to
# ``_kernel_equivalence_proof``, which stay the oracle
# (``native_kernel_proofs(False)`` runs them).
NATIVE_KERNEL_PROOFS = True


@contextmanager
def native_kernel_proofs(enabled: bool) -> Iterator[None]:
    """``NATIVE_KERNEL_PROOFS`` set to ``enabled`` for the block and restored
    after it: ``native_kernel_proofs(False)`` is the oracle lane."""

    global NATIVE_KERNEL_PROOFS
    before = NATIVE_KERNEL_PROOFS
    NATIVE_KERNEL_PROOFS = bool(enabled)
    try:
        yield
    finally:
        NATIVE_KERNEL_PROOFS = before


def _annotate_oriented_kernel_evaluation_equivalence(
    kernels: Sequence[CompiledOrientedKernel],
    particles: Sequence[CompiledParticleRecord],
    terms: Sequence[CompiledVertexTerm],
    model_symbols: ModelSymbolRegistry,
) -> tuple[CompiledOrientedKernel, ...]:
    """Prove exact signed/permuted kernel relations from lowered expressions.

    The comparison uses the concrete component expressions after resolving
    generated coupling aliases.  It is therefore independent of how a UFO
    author spelled or ordered the source Lorentz/color structures.  Only exact
    Symbolica canonical equality up to an overall sign and input exchange is
    recorded; kernels that do not pass that test remain in distinct classes.
    """

    particle_by_name = {particle.name: particle for particle in particles}
    term_by_id = {term.id: term for term in terms}

    def derived_terms(kernel: CompiledOrientedKernel) -> list[int]:
        return [
            term_id
            for name in kernel.runtime_parameters
            if name.startswith("derived_coupling_")
            for term_id in (int(name.rsplit("_", 1)[1]),)
            if term_id in term_by_id
        ]

    def dimensions_of(kernel: CompiledOrientedKernel) -> tuple[int, int, int]:
        return tuple(
            compiled_particle_component_dimension(particle_by_name[name])
            for name in kernel.particles[:3]
        )

    if NATIVE_KERNEL_PROOFS:
        from pyamplicol.loops.native import load_loop_native

        derived_names: dict[int, str] = {}

        def derived(term_id: int) -> tuple[str, str]:
            if term_id not in derived_names:
                derived_names[term_id] = symbols.derived_coupling(
                    model_symbols.model_name, term_id
                ).get_name()
            return derived_names[term_id], term_by_id[term_id].coupling_expression

        # The model's kernel symbol namespace (``kernel_component``).
        prefix = model_symbols.kernel_component(0, "left", 0).get_name()
        keys, found = load_loop_native().model_kernel_equivalence(
            [
                (
                    kernel.kind,
                    kernel.coupling_expression,
                    int(kernel.lc_color_normalization_power or 0),
                    list(kernel.component_expressions),
                    # As the oracle's dict: one replacement per symbol.
                    [
                        derived(term_id)
                        for term_id in dict.fromkeys(derived_terms(kernel))
                    ],
                    dimensions_of(kernel),
                )
                for kernel in kernels
            ],
            prefix.rsplit("::", 1)[0] + "::",
            set(_KERNEL_PROOFS),
        )
        proofs = {
            key: (
                class_digest,
                (float(sign), 0.0),
                tuple(input_order),
                None if exchange is None else (float(exchange), 0.0),
            )
            for key, (class_digest, sign, input_order, exchange) in found
        }
    else:
        keys, proofs = _kernel_keys_and_proofs(
            kernels, term_by_id, derived_terms, dimensions_of, model_symbols
        )
    needed = {key: _KERNEL_PROOFS.get(key) or proofs[key] for key in keys}
    if len(_KERNEL_PROOFS) + len(proofs) > _KERNEL_PROOF_BOUND:
        _KERNEL_PROOFS.clear()
    _KERNEL_PROOFS.update(proofs)
    # Each kernel's factor relative to the first factor of its class.
    reference_factor_by_class: dict[str, complex] = {}
    normalized: list[CompiledOrientedKernel] = []
    for kernel, key in zip(kernels, keys, strict=True):
        class_digest, factor, input_order, input_exchange_factor = needed[key]
        evaluation_class = f"symbolica-sha256:{class_digest}"
        class_factor_complex = complex(*factor)
        reference_factor = reference_factor_by_class.setdefault(
            evaluation_class,
            class_factor_complex,
        )
        relative_factor = class_factor_complex / reference_factor
        normalized.append(
            replace(
                kernel,
                evaluation_class=evaluation_class,
                evaluation_factor=(
                    float(relative_factor.real),
                    float(relative_factor.imag),
                ),
                evaluation_input_order=input_order,
                evaluation_input_exchange_factor=input_exchange_factor,
                evaluation_equivalence_verified=True,
            )
        )
    return tuple(normalized)


def _kernel_keys_and_proofs(
    kernels: Sequence[CompiledOrientedKernel],
    term_by_id: Mapping[int, CompiledVertexTerm],
    derived_terms: Any,
    dimensions_of: Any,
    model_symbols: ModelSymbolRegistry,
) -> tuple[list[str], dict[str, tuple[Any, ...]]]:
    """The oracle of ``model_kernel_equivalence``: per kernel its proof key,
    and the proofs (``_kernel_equivalence_proof``) of the distinct keys the
    process's memo (``_KERNEL_PROOFS``) lacks."""

    parsed: dict[int, _sym.Expression] = {}

    def term_coupling(term_id: int) -> _sym.Expression:
        if term_id not in parsed:
            parsed[term_id] = _sym.E(term_by_id[term_id].coupling_expression)
        return parsed[term_id]

    keys: list[str] = []
    proofs: dict[str, tuple[Any, ...]] = {}
    for kernel in kernels:
        derived_couplings = {
            symbols.derived_coupling(model_symbols.model_name, term_id): term_coupling(
                term_id
            )
            for term_id in derived_terms(kernel)
        }
        coupling = _replace_expression_symbols(
            _sym.E(kernel.coupling_expression),
            derived_couplings,
        )
        # The algebraic colour-flow normalization is evaluated by the kernel,
        # so it must also participate in proofs of shared kernel evaluations.
        if kernel.lc_color_normalization_power:
            coupling *= _sym.E(f"2^(-{kernel.lc_color_normalization_power}/2)")
        dimensions = dimensions_of(kernel)
        # The kernel's components in the canonical kernel symbols, with the
        # coupling aliases resolved, per input order: with the dimensions and
        # the coupling they determine the proof below exactly, so kernels
        # with equal ones (the vertex copies of counterterm clone lines,
        # ``ct_line_insertions``) share one proof.
        remapped = {
            input_order: tuple(
                _replace_expression_symbols(
                    _remap_kernel_symbols(
                        _sym.E(component),
                        old_kind=kernel.kind,
                        new_kind=0,
                        model_symbols=model_symbols,
                        swap_sides=swap_sides,
                    ),
                    derived_couplings,
                )
                for component in kernel.component_expressions
            )
            for input_order, swap_sides in (((0, 1), False), ((1, 0), True))
        }
        key = hashlib.sha256(
            json.dumps(
                [
                    dimensions,
                    coupling.to_canonical_string(),
                    *(
                        [component.to_canonical_string() for component in components]
                        for components in remapped.values()
                    ),
                ],
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        keys.append(key)
        if key not in _KERNEL_PROOFS and key not in proofs:
            proofs[key] = _kernel_equivalence_proof(remapped, coupling, dimensions)
    return keys, proofs


def _kernel_equivalence_proof(
    remapped: Mapping[tuple[int, int], tuple[_sym.Expression, ...]],
    coupling: _sym.Expression,
    dimensions: tuple[int, int, int],
) -> tuple[str, tuple[float, float], tuple[int, int], tuple[float, float] | None]:
    """(class digest, factor, input order, input exchange factor) of one
    oriented kernel (``_annotate_oriented_kernel_evaluation_equivalence``)
    from its ``remapped`` components per input order, its ``coupling`` and
    the component ``dimensions`` of its two inputs and its output.

    The class is the least signature (input dimensions and canonical
    component strings) over the input orders and overall signs. The
    canonical strings of large expanded components dominate the cost, so
    they are formed lazily: the signatures share their prefix up to the
    first component, and a JSON string token is never a proper prefix of
    another, so the first component whose tokens differ decides the order
    of two signatures exactly. Only the candidates tied so far get their
    next component's string, and the least signature is then formed whole,
    with ties kept in candidate order as ``min`` keeps them: the result is
    the eager comparison of every complete signature."""

    oriented: dict[tuple[int, int], tuple[_sym.Expression, ...]] = {
        input_order: tuple((component * coupling).expand() for component in components)
        for input_order, components in remapped.items()
    }
    candidates = [(input_order, sign) for input_order in oriented for sign in (1, -1)]
    strings: dict[tuple[tuple[int, int], int, int], str] = {}

    def component_string(candidate: tuple[tuple[int, int], int], index: int) -> str:
        input_order, sign = candidate
        key = (input_order, sign, index)
        if key not in strings:
            # Horner/factor collection optimizes evaluation, but is not a
            # unique algebraic normal form: mirrored kernels can retain
            # different factorizations of the same coupling polynomial.
            # Proof identities must use expanded expressions, while the
            # stored runtime components remain optimized and unchanged.
            strings[key] = (
                (sign * oriented[input_order][index]).expand().to_canonical_string()
            )
        return strings[key]

    def signature(candidate: tuple[tuple[int, int], int]) -> str:
        input_order, _sign = candidate
        return json.dumps(
            {
                "input_dimensions": [
                    dimensions[input_order[0]],
                    dimensions[input_order[1]],
                ],
                "output_dimension": dimensions[2],
                "components": [
                    component_string(candidate, index)
                    for index in range(len(oriented[input_order]))
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    tied = candidates
    counts = {len(components) for components in oriented.values()}
    if len(counts) == 1:
        for index in range(counts.pop()):
            if len(tied) == 1:
                break
            tokens = {
                candidate: json.dumps(component_string(candidate, index))
                for candidate in tied
            }
            least = min(tokens.values())
            tied = [candidate for candidate in tied if tokens[candidate] == least]
    winner = min(tied, key=signature)
    input_order, sign = winner
    class_digest = hashlib.sha256(signature(winner).encode("utf-8")).hexdigest()
    factor = (float(sign), 0.0)
    input_exchange_factor: tuple[float, float] | None = None
    if dimensions[0] == dimensions[1]:
        direct_components = oriented[(0, 1)]
        swapped_components = oriented[(1, 0)]
        zero = _sym.E("0")
        for exchange_sign in (1, -1):
            # A structural zero test: no canonical string of a non-zero
            # difference is formed.
            if all(
                (swapped - exchange_sign * direct).expand() == zero
                for direct, swapped in zip(
                    direct_components,
                    swapped_components,
                    strict=True,
                )
            ):
                input_exchange_factor = (float(exchange_sign), 0.0)
                break
    return class_digest, factor, input_order, input_exchange_factor


def _compile_oriented_kernels(
    terms: Sequence[CompiledVertexTerm],
    particles: Sequence[CompiledParticleRecord],
    parameters: Sequence[CompiledParameterRecord],
    propagators: Sequence[CompiledPropagatorRecord],
    model_symbols: ModelSymbolRegistry,
    *,
    transverse_massless_yang_mills: bool,
) -> tuple[CompiledOrientedKernel, ...]:
    particle_by_name = {particle.name: particle for particle in particles}
    parameter_by_name = {parameter.name: parameter for parameter in parameters}
    propagator_by_name = {propagator.name: propagator for propagator in propagators}
    external_symbols = {
        parameter.name: model_symbols.symbol(parameter.name)
        for parameter in parameters
        if parameter.nature == "external"
    }
    kernels: list[CompiledOrientedKernel] = []
    # The components of an oriented Lorentz structure, by (structure, spins,
    # legs, transverse variant), with the kernel kind they were formed for:
    # the network runs on canonical symbols whose order does not depend on
    # the kind (``_execute_dense_network``), so another kernel's components
    # are these with the kernel symbols renamed (the vertex copies of
    # counterterm clone lines and every repeated structure).
    oriented_memo: dict[tuple[object, ...], tuple[int, Any]] = {}
    for term in terms:
        if term.valence != 3:
            continue
        coupling_symbols = set(_sym.E(term.coupling_expression).get_all_symbols(False))
        runtime_parameters = tuple(
            sorted(
                name
                for name, symbol in external_symbols.items()
                if symbol in coupling_symbols
            )
        )
        oriented_result_particles: set[str] = set()
        for result_leg in range(3):
            result_source_name = term.particles[result_leg]
            if result_source_name in oriented_result_particles:
                continue
            oriented_result_particles.add(result_source_name)
            input_legs = tuple(leg for leg in range(3) if leg != result_leg)
            input_orders = (
                (input_legs,)
                if term.particles[input_legs[0]] == term.particles[input_legs[1]]
                else (input_legs, tuple(reversed(input_legs)))
            )
            for left_leg, right_leg in input_orders:
                result_source = particle_by_name[term.particles[result_leg]]
                try:
                    result_name = particle_by_name[result_source.antiname].name
                except KeyError as exc:
                    raise ValueError(
                        f"vertex {term.vertex} particle {result_source.name} refers to "
                        f"absent antiparticle {result_source.antiname}"
                    ) from exc
                transverse = (
                    transverse_massless_yang_mills
                    and _term_supports_transverse_massless_yang_mills(
                        term,
                        particle_by_name,
                        parameter_by_name,
                        propagator_by_name,
                    )
                )
                memo_key = (
                    term.lorentz_expression,
                    tuple(particle_by_name[name].spin for name in term.particles),
                    left_leg,
                    right_leg,
                    result_leg,
                    transverse,
                )
                found = oriented_memo.get(memo_key)
                if found is None:
                    ordered_components = _oriented_component_expressions(
                        term,
                        particle_by_name,
                        left_leg=left_leg,
                        right_leg=right_leg,
                        result_leg=result_leg,
                        kind=len(kernels),
                        model_symbols=model_symbols,
                        use_transverse_massless_yang_mills=transverse,
                    )
                    oriented_memo[memo_key] = (len(kernels), ordered_components)
                else:
                    formed, ordered_components = found
                    ordered_components = replace(
                        ordered_components,
                        values=tuple(
                            _renamed_kernel_text(
                                value,
                                old_kind=formed,
                                new_kind=len(kernels),
                                model_symbols=model_symbols,
                            )
                            for value in ordered_components.values
                        ),
                    )
                kernels.append(
                    CompiledOrientedKernel(
                        kind=len(kernels),
                        term_id=term.id,
                        vertex=term.vertex,
                        particles=(
                            term.particles[left_leg],
                            term.particles[right_leg],
                            result_name,
                        ),
                        source_particle_legs=(left_leg, right_leg, result_leg),
                        component_expressions=tuple(
                            str(component) for component in ordered_components.values
                        ),
                        coupling_expression=term.coupling_expression,
                        coupling_orders=term.coupling_orders,
                        runtime_parameters=runtime_parameters,
                        color_source=term.color_source,
                        color_expression=term.color_expression,
                        lc_color_normalization_power=(
                            term.lc_color_normalization_power
                        ),
                        term_ids=(term.id,),
                        output_ordering_id=ordered_components.ordering_id,
                    )
                )
    return _fuse_oriented_kernels(
        kernels,
        model_name=model_symbols.model_name,
    )


def _term_supports_transverse_massless_yang_mills(
    term: CompiledVertexTerm,
    particles: Mapping[str, CompiledParticleRecord],
    parameters: Mapping[str, CompiledParameterRecord],
    propagators: Mapping[str, CompiledPropagatorRecord],
) -> bool:
    """Prove the field/propagator contract needed by transverse YM lowering."""

    if not _is_single_structure_constant(term.color_expression):
        return False
    for name in term.particles:
        particle = particles[name]
        propagator = (
            None
            if particle.propagator is None
            else propagators.get(particle.propagator)
        )
        if not (
            particle.spin == 3
            and particle.color == 8
            and particle.self_conjugate is True
            and _is_compile_time_zero_parameter(particle.mass, parameters)
            and particle.propagating
            and particle.ghost_number == 0
            and propagator is not None
            and propagator.particle == particle.name
            and not propagator.custom
        ):
            return False
    return True


def _compile_four_point_contact_kernels(
    terms: Sequence[CompiledVertexTerm],
    particles: Sequence[CompiledParticleRecord],
    *,
    start_kind: int,
    model_symbols: ModelSymbolRegistry,
) -> tuple[tuple[CompiledParticleRecord, ...], tuple[CompiledOrientedKernel, ...]]:
    """Lower momentum-independent four-point tensors through dense auxiliaries."""

    particle_by_name = {particle.name: particle for particle in particles}
    used_pdgs = {abs(particle.pdg_code) for particle in particles}
    next_pdg = max(9_000_000, max(used_pdgs, default=0) + 1)
    auxiliary_particles: list[CompiledParticleRecord] = []
    kernels: list[CompiledOrientedKernel] = []

    def allocate_pdg() -> int:
        nonlocal next_pdg
        while next_pdg in used_pdgs:
            next_pdg += 1
        result = next_pdg
        used_pdgs.add(result)
        next_pdg += 1
        return result

    for term in terms:
        if term.valence != 4:
            continue
        if "ufo_momentum_" in term.lorentz_expression:
            continue
        source_particles = tuple(particle_by_name[name] for name in term.particles)
        proof = _validated_contact_decomposition_proof(term)
        if (
            proof is None
            and term.contact_decomposition_proof is None
            and _contact_term_has_literal_color_singlet(term)
            and all(particle.color == 1 for particle in source_particles)
        ):
            # Preserve direct construction of uncolored test/compiler terms.
            # UFO compilation always serializes this trivial proof beforehand.
            proof = _build_contact_decomposition_proof(
                term,
                source_particles,
                particle_by_name,
                model_symbols=model_symbols,
            )
        if proof is None:
            # Leave the complete term unlowered so model preflight reports a
            # structured unsupported-contact-color-lowering error. Partial
            # orientation output would incorrectly make the term appear valid.
            continue
        for split in proof.splits:
            result_leg = split.result_leg
            source_result = source_particles[result_leg]
            remaining_leg = split.remaining_leg
            open_legs = split.open_legs
            component_axis_order = tuple(
                label
                for leg in open_legs
                for label in _spin_axis_labels(source_particles[leg].spin, leg + 1)
            )
            if component_axis_order != split.component_axis_order:
                raise ValueError(
                    f"contact decomposition component order mismatch for term "
                    f"{term.id} result leg {result_leg}"
                )
            auxiliary_name = f"__pyamplicol_contact_{term.id}_r{result_leg}"
            representative_indices = split.component_basis_order
            component_expansion = split.component_expansion
            auxiliary_dimension = len(representative_indices)
            auxiliary = CompiledParticleRecord(
                name=auxiliary_name,
                antiname=auxiliary_name,
                pdg_code=allocate_pdg(),
                spin=-1,
                color=split.auxiliary_color,
                mass="ZERO",
                width="ZERO",
                charge=0.0,
                quantum_numbers=(("electric_charge", "0"),),
                ghost_number=0,
                propagating=False,
                goldstoneboson=False,
                propagator=None,
                component_dimension=auxiliary_dimension,
                auxiliary_kind=(
                    f"ufo-contact:{term.id}:result-{result_leg}:"
                    + ",".join(str(leg) for leg in open_legs)
                ),
            )
            auxiliary_particles.append(auxiliary)

            partial_orientations = tuple(
                item for item in split.orientations if item.stage == "partial"
            )
            for orientation in partial_orientations:
                left_leg, right_leg = orientation.input_legs
                kind = start_kind + len(kernels)
                components = projected_contact_partial_components(
                    term,
                    particle_by_name,
                    left_leg=left_leg,
                    right_leg=right_leg,
                    open_legs=open_legs,
                    kind=kind,
                    model_symbols=model_symbols,
                    symmetry=contact_pair_symmetry(
                        source_particles, split.pair_legs, split.outer_color_source
                    ),
                )
                kernels.append(
                    CompiledOrientedKernel(
                        kind=kind,
                        term_id=term.id,
                        vertex=f"{term.vertex}::contact-partial",
                        particles=(
                            source_particles[left_leg].name,
                            source_particles[right_leg].name,
                            auxiliary.name,
                        ),
                        source_particle_legs=(left_leg, right_leg, -1),
                        component_expressions=tuple(
                            _canonical_product_text(
                                components[index],
                                orientation.scalar_prefactor,
                                kind=kind,
                                model_symbols=model_symbols,
                            )
                            for index in representative_indices
                        ),
                        coupling_expression="1",
                        coupling_orders=(),
                        runtime_parameters=(),
                        color_source="1",
                        color_expression="1",
                        lc_color_normalization_power=0,
                        term_ids=(),
                        contact_orbit_steps=_compiled_contact_orbit_steps(
                            term,
                            split,
                            orientation,
                        ),
                    )
                )

            result_name = particle_by_name[source_result.antiname].name
            final_orientations = tuple(
                item for item in split.orientations if item.stage == "final"
            )
            for orientation in final_orientations:
                auxiliary_on_left = orientation.input_legs[0] == -1
                left_name, right_name = (
                    (auxiliary.name, source_particles[remaining_leg].name)
                    if auxiliary_on_left
                    else (source_particles[remaining_leg].name, auxiliary.name)
                )
                kind = start_kind + len(kernels)
                derived_coupling = symbols.derived_coupling(
                    model_symbols.model_name,
                    term.id,
                )
                final_prefactor = (
                    _sym.E(orientation.scalar_prefactor) * derived_coupling
                )
                combined_color_source = (
                    f"{split.outer_color_source}*{split.final_color_source}"
                    if split.decomposition_kind == "two-structure-constants"
                    # The identity only passes the summed index on.
                    else split.outer_color_source
                    if split.decomposition_kind == "trilinear-identity"
                    # The pair closed into a singlet; the result takes the
                    # remaining leg's colour with weight 1 (recurrence:
                    # ``_contact_recurrence_color_contract``).
                    else "1"
                    if split.decomposition_kind
                    in ("two-identities", "identity-singlets")
                    else term.color_source
                )
                kernels.append(
                    CompiledOrientedKernel(
                        kind=kind,
                        term_id=term.id,
                        vertex=f"{term.vertex}::contact-final",
                        particles=(left_name, right_name, result_name),
                        source_particle_legs=(
                            -1 if auxiliary_on_left else remaining_leg,
                            remaining_leg if auxiliary_on_left else -1,
                            result_leg,
                        ),
                        component_expressions=tuple(
                            (_sym.E(component) * final_prefactor).to_canonical_string()
                            for component in _contact_final_component_expressions(
                                source_particles,
                                auxiliary,
                                open_legs=open_legs,
                                remaining_leg=remaining_leg,
                                result_leg=result_leg,
                                kind=kind,
                                auxiliary_on_left=auxiliary_on_left,
                                component_expansion=component_expansion,
                                model_symbols=model_symbols,
                            )
                        ),
                        coupling_expression="1",
                        coupling_orders=term.coupling_orders,
                        runtime_parameters=(f"derived_coupling_{term.id}",),
                        color_source=combined_color_source,
                        color_expression=combined_color_source,
                        lc_color_normalization_power=(
                            split.outer_color_normalization_power
                            + split.final_color_normalization_power
                        ),
                        term_ids=(term.id,),
                        contact_orbit_steps=_compiled_contact_orbit_steps(
                            term,
                            split,
                            orientation,
                        ),
                    )
                )
    return tuple(auxiliary_particles), tuple(kernels)


def _compiled_contact_orbit_steps(
    term: CompiledVertexTerm,
    split: CompiledContactDecompositionSplit,
    orientation: CompiledContactOrientationProof,
) -> tuple[CompiledContactOrbitStep, ...]:
    certificate = term.contact_orbit_certificate
    if certificate is None:
        return ()
    return (compiled_contact_orbit_step(certificate, split, orientation),)
