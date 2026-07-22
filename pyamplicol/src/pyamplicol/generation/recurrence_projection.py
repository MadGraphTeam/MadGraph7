# SPDX-License-Identifier: 0BSD
"""Project proven process semantics into recurrence-builder logical records.

The projector is deliberately upstream of generic DAG construction.  It maps
only canonical process IR, a complete LC colour plan, an optional exact replay
proof, and a prepared recurrence-template catalog into the compact logical ABI
consumed by :mod:`pyamplicol.generation.recurrence_columnar`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from ..color.plan import GenericColorPlan, LCColorTopologyReplayPlan
from ..models.recurrence_template import (
    CurrentStateTemplateV1,
    EvaluatorBindingV1,
    RecurrenceTemplateCatalog,
    SourceTemplateV1,
)
from ..processes.ir import CanonicalProcessIR, ProcessLegIR
from .recurrence_columnar import (
    ExactComplexRationalV1,
    RecurrenceBuilderLogicalInputV1,
    RecurrenceCouplingLimitV1,
    RecurrenceExternalLegV1,
    RecurrenceLCFlowLayout,
    RecurrenceLCOpenStringV1,
    RecurrenceNormalizationV1,
    RecurrenceParameterProjectionV1,
    RecurrencePhysicalLCSectorV1,
    RecurrencePublicLCFlowV1,
    RecurrenceReplayPartitionV1,
    RecurrenceReplayTargetV1,
    RecurrenceSelectedSourceCoverageV1,
    RecurrenceSemanticDigestV1,
    RecurrenceSemanticTemplateReferenceV1,
    RecurrenceSourceStateV1,
)

_TEMPLATE_SECTIONS: Final = (
    ("parameter", "parameters"),
    ("current-state", "current_states"),
    ("source", "sources"),
    ("quantum-flow", "quantum_flows"),
    ("transition", "transitions"),
    ("propagator", "propagators"),
    ("closure", "closures"),
    ("color-contraction", "color_contractions"),
    ("symmetry-proof", "symmetry_proofs"),
)
_CLOSURE_ANCHOR_PROOF_ALGORITHM: Final = "canonical-lc-closure-anchor-v1"


class RecurrenceProjectionError(ValueError):
    """The supplied process semantics cannot be projected exactly."""


@dataclass(frozen=True, slots=True)
class RecurrenceGenerationSliceV1:
    """Explicit generation-time coverage retained by recurrence v1."""

    selected_public_flow_ids: tuple[int, ...] | None = None
    selected_source_helicities: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        if self.selected_public_flow_ids is not None:
            flows = tuple(sorted(int(value) for value in self.selected_public_flow_ids))
            if not flows or len(flows) != len(set(flows)):
                raise RecurrenceProjectionError(
                    "selected recurrence public flows must be nonempty and unique"
                )
            object.__setattr__(self, "selected_public_flow_ids", flows)
        if self.selected_source_helicities is not None:
            selected = tuple(
                sorted(
                    (int(label), int(helicity))
                    for label, helicity in self.selected_source_helicities
                )
            )
            labels = tuple(label for label, _ in selected)
            if not selected or len(labels) != len(set(labels)):
                raise RecurrenceProjectionError(
                    "selected recurrence source labels must be nonempty and unique"
                )
            object.__setattr__(self, "selected_source_helicities", selected)


def project_recurrence_process_v1(
    process: CanonicalProcessIR,
    color_plan: GenericColorPlan,
    template_catalog: RecurrenceTemplateCatalog,
    *,
    layout: RecurrenceLCFlowLayout,
    normalization: RecurrenceNormalizationV1,
    topology_replay: LCColorTopologyReplayPlan | None = None,
    generation_slice: RecurrenceGenerationSliceV1 | None = None,
    coupling_order_limits: Mapping[str, int] | None = None,
    process_support_mask: int = 1,
) -> RecurrenceBuilderLogicalInputV1:
    """Return deterministic recurrence-builder records for one LC process.

    Normalization is supplied as an exact, model-owned contract because it is
    process dependent and intentionally absent from the process-independent
    recurrence template catalog.
    """

    _validate_projection_roots(process, color_plan, template_catalog, layout)
    if not isinstance(normalization, RecurrenceNormalizationV1):
        raise TypeError("recurrence projection requires exact normalization v1")
    if (
        isinstance(process_support_mask, bool)
        or not isinstance(process_support_mask, int)
        or process_support_mask <= 0
    ):
        raise RecurrenceProjectionError(
            "recurrence process support mask must be a positive integer bitmap"
        )

    selection = generation_slice or RecurrenceGenerationSliceV1()
    template_ids, template_references = _project_template_references(template_catalog)
    external_legs, selected_sources = _project_external_legs(
        process,
        template_catalog,
        template_ids,
        selection.selected_source_helicities,
        process_support_mask,
    )
    physical_sectors, public_flows = _project_physical_sectors(
        color_plan,
        process,
        external_legs,
        process_support_mask,
    )
    selected_flows = selection.selected_public_flow_ids
    if selected_flows is not None:
        available_flow_ids = {flow.flow_id for flow in public_flows}
        unknown = tuple(sorted(set(selected_flows) - available_flow_ids))
        if unknown:
            raise RecurrenceProjectionError(
                f"generation slice references unknown public LC flows {unknown!r}"
            )
        if layout == "all-flow-union":
            raise RecurrenceProjectionError(
                "all-flow-union recurrence must retain every public LC flow"
            )
    replay_partitions = _project_replay_partitions(
        process,
        color_plan,
        physical_sectors,
        topology_replay,
        layout,
    )

    return RecurrenceBuilderLogicalInputV1(
        process_id=process.key,
        layout=layout,
        semantic_digests=(
            RecurrenceSemanticDigestV1("process", _digest(process.to_json_dict())),
            RecurrenceSemanticDigestV1(
                "model-catalog", template_catalog.header.compiled_model_digest
            ),
            RecurrenceSemanticDigestV1(
                "prepared-catalog", template_catalog.catalog_digest
            ),
            RecurrenceSemanticDigestV1(
                "color-plan", _digest(color_plan.to_json_dict())
            ),
        ),
        external_legs=external_legs,
        physical_sectors=physical_sectors,
        public_flows=public_flows,
        semantic_template_references=template_references,
        normalization=normalization,
        replay_partitions=replay_partitions,
        selected_public_flow_ids=selected_flows,
        selected_source_coverage=selected_sources,
        coupling_limits=_project_coupling_limits(coupling_order_limits),
        parameter_projection=_project_parameters(template_catalog, template_ids),
        process_support_mask=process_support_mask,
    )


def _validate_projection_roots(
    process: CanonicalProcessIR,
    color_plan: GenericColorPlan,
    template_catalog: RecurrenceTemplateCatalog,
    layout: str,
) -> None:
    if not isinstance(process, CanonicalProcessIR):
        raise TypeError("recurrence projection requires CanonicalProcessIR")
    if not isinstance(color_plan, GenericColorPlan):
        raise TypeError("recurrence projection requires GenericColorPlan")
    if not isinstance(template_catalog, RecurrenceTemplateCatalog):
        raise TypeError("recurrence projection requires RecurrenceTemplateCatalog")
    if layout not in {"topology-replay", "all-flow-union"}:
        raise RecurrenceProjectionError(
            f"unsupported recurrence LC flow layout {layout!r}"
        )
    if process.color_accuracy != "lc" or color_plan.color_accuracy != "lc":
        raise RecurrenceProjectionError(
            "recurrence process projection v1 supports LC only; use compiled or "
            "eager for NLC/full"
        )
    if color_plan.process != process:
        raise RecurrenceProjectionError(
            "recurrence color plan does not belong to the supplied process IR"
        )
    if color_plan.truncated:
        raise RecurrenceProjectionError(
            "recurrence projection rejects truncated color plans"
        )
    if not color_plan.sectors:
        raise RecurrenceProjectionError(
            "recurrence projection requires physical LC sectors"
        )
    sector_ids = tuple(sorted(int(sector.id) for sector in color_plan.sectors))
    if sector_ids != tuple(range(len(sector_ids))):
        raise RecurrenceProjectionError(
            "recurrence projection requires dense physical LC sector IDs"
        )


def _project_template_references(
    catalog: RecurrenceTemplateCatalog,
) -> tuple[
    dict[tuple[str, str], int],
    tuple[RecurrenceSemanticTemplateReferenceV1, ...],
]:
    binding_by_template: dict[str, EvaluatorBindingV1] = {}
    for binding in catalog.evaluator_bindings:
        for semantic_id in binding.semantic_template_ids:
            previous = binding_by_template.setdefault(semantic_id, binding)
            if previous != binding:
                raise RecurrenceProjectionError(
                    f"semantic template {semantic_id!r} has multiple callables"
                )

    numeric_ids: dict[tuple[str, str], int] = {}
    references: list[RecurrenceSemanticTemplateReferenceV1] = []
    for kind, attribute in _TEMPLATE_SECTIONS:
        records = tuple(
            sorted(getattr(catalog, attribute), key=lambda item: item.template_id)
        )
        for numeric_id, record in enumerate(records):
            key = (kind, record.template_id)
            numeric_ids[key] = numeric_id
            binding = binding_by_template.get(record.template_id)
            prepared_kernel_id = None if binding is None else binding.prepared_kernel_id
            references.append(
                RecurrenceSemanticTemplateReferenceV1(
                    kind=kind,
                    template_id=numeric_id,
                    semantic_digest=record.semantic_digest,
                    prepared_kernel_id=prepared_kernel_id,
                )
            )
    return numeric_ids, tuple(references)


def _project_external_legs(
    process: CanonicalProcessIR,
    catalog: RecurrenceTemplateCatalog,
    template_ids: Mapping[tuple[str, str], int],
    selected_helicities: tuple[tuple[int, int], ...] | None,
    support_mask: int,
) -> tuple[
    tuple[RecurrenceExternalLegV1, ...],
    tuple[RecurrenceSelectedSourceCoverageV1, ...] | None,
]:
    selected = dict(selected_helicities or ())
    known_labels = {int(leg.label) for leg in process.legs}
    unknown_labels = tuple(sorted(set(selected) - known_labels))
    if unknown_labels:
        raise RecurrenceProjectionError(
            f"generation slice references unknown source labels {unknown_labels!r}"
        )

    states_by_id = {record.template_id: record for record in catalog.current_states}
    source_rows: dict[
        int,
        list[tuple[SourceTemplateV1, CurrentStateTemplateV1]],
    ] = {}
    for source in catalog.sources:
        try:
            current = states_by_id[source.state_template_id]
        except KeyError as exc:
            raise RecurrenceProjectionError(
                f"source template {source.template_id!r} references a missing state"
            ) from exc
        source_rows.setdefault(int(current.particle_id), []).append((source, current))

    external: list[RecurrenceExternalLegV1] = []
    selected_coverage: list[RecurrenceSelectedSourceCoverageV1] = []
    for source_slot, leg in enumerate(process.legs):
        is_fermionic = _is_fermionic_process_leg(leg)
        if leg.pdg is None or leg.outgoing_pdg is None:
            raise RecurrenceProjectionError(
                f"external leg {leg.label} has no concrete physical/outgoing PDG"
            )
        candidates = source_rows.get(int(leg.outgoing_pdg), ())
        projected: list[
            tuple[
                SourceTemplateV1,
                CurrentStateTemplateV1,
                int,
                int,
                int,
                int,
                ExactComplexRationalV1,
            ]
        ] = []
        for source, current in candidates:
            if source.wavefunction_family != leg.wavefunction_family:
                continue
            helicity = int(source.helicity)
            chirality = int(current.chirality)
            spin_state = int(source.spin_state)
            crossing = _crossing_contract(source)
            momentum_sign = 1
            phase = ExactComplexRationalV1(1)
            if leg.is_initial:
                helicity *= crossing[0]
                chirality *= crossing[1]
                spin_state *= crossing[2]
                momentum_sign = crossing[3]
                phase = crossing[4]
            projected.append(
                (
                    source,
                    current,
                    helicity,
                    chirality,
                    spin_state,
                    momentum_sign,
                    phase,
                )
            )
        projected.sort(
            key=lambda item: (
                item[2],
                item[3],
                item[4],
                item[1].template_id,
                item[0].template_id,
            )
        )
        if not projected:
            raise RecurrenceProjectionError(
                "recurrence template catalog has no supported source semantics for "
                f"external leg {leg.label} (outgoing PDG {leg.outgoing_pdg})"
            )

        source_states = tuple(
            RecurrenceSourceStateV1(
                state_index=index,
                public_helicity=helicity,
                chirality=chirality,
                spin_state=spin_state,
                current_state_template_id=_template_id(
                    template_ids,
                    "current-state",
                    current.template_id,
                ),
                source_template_id=_template_id(
                    template_ids, "source", source.template_id
                ),
                momentum_sign=momentum_sign,
                crossing_phase=phase,
            )
            for index, (
                source,
                current,
                helicity,
                chirality,
                spin_state,
                momentum_sign,
                phase,
            ) in enumerate(projected)
        )
        external.append(
            RecurrenceExternalLegV1(
                source_slot=source_slot,
                public_label=int(leg.label),
                physical_pdg=int(leg.pdg),
                outgoing_pdg=int(leg.outgoing_pdg),
                is_initial=bool(leg.is_initial),
                is_fermionic=is_fermionic,
                source_states=source_states,
                momentum_mask=1 << source_slot,
                support_mask=support_mask,
            )
        )
        if leg.label in selected:
            requested = selected[int(leg.label)]
            retained = tuple(
                index
                for index, (
                    _source,
                    _current,
                    helicity,
                    _chirality,
                    _spin_state,
                    _momentum_sign,
                    _phase,
                ) in enumerate(projected)
                if helicity == requested
            )
            if not retained:
                available = tuple(sorted({item[2] for item in projected}))
                raise RecurrenceProjectionError(
                    f"selected helicity {requested} is unavailable for source "
                    f"label {leg.label}; available={available!r}"
                )
            selected_coverage.append(
                RecurrenceSelectedSourceCoverageV1(source_slot, retained)
            )

    return (
        tuple(external),
        None if selected_helicities is None else tuple(selected_coverage),
    )


def _is_fermionic_process_leg(leg: ProcessLegIR) -> bool:
    concrete_statistics = {"auxiliary", "boson", "fermion", "ghost"}
    concrete_families = {
        "auxiliary",
        "fermion",
        "ghost",
        "scalar",
        "spin2",
        "vector",
    }
    if (
        leg.statistics not in concrete_statistics
        or leg.wavefunction_family not in concrete_families
    ):
        raise RecurrenceProjectionError(
            "recurrence closure-anchor classification requires concrete external "
            f"statistics and spin metadata for leg {leg.label}"
        )
    statistics_is_fermionic = leg.statistics == "fermion"
    family_is_fermionic = leg.wavefunction_family == "fermion"
    if statistics_is_fermionic != family_is_fermionic:
        raise RecurrenceProjectionError(
            "recurrence closure-anchor classification found inconsistent "
            f"statistics and spin metadata for leg {leg.label}"
        )
    return statistics_is_fermionic


def _crossing_contract(
    source: SourceTemplateV1,
) -> tuple[int, int, int, int, ExactComplexRationalV1]:
    try:
        payload = json.loads(source.crossing)
        helicity = payload["helicity_factor"]
        chirality = payload["chirality_factor"]
        spin_state = payload["spin_state_factor"]
        momentum_transform = payload["momentum_transform"]
        phase = payload["phase"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RecurrenceProjectionError(
            f"source template {source.template_id!r} has malformed crossing semantics"
        ) from exc
    if (
        helicity not in {-1, 1}
        or chirality not in {-1, 1}
        or spin_state not in {-1, 1}
        or momentum_transform not in {"identity", "negate-four-momentum"}
        or not isinstance(phase, Mapping)
    ):
        raise RecurrenceProjectionError(
            f"source template {source.template_id!r} has unsupported crossing factors"
        )
    try:
        exact_phase = ExactComplexRationalV1(
            int(phase["real_numerator"]),
            int(phase["real_denominator"]),
            int(phase["imag_numerator"]),
            int(phase["imag_denominator"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RecurrenceProjectionError(
            f"source template {source.template_id!r} has malformed crossing phase"
        ) from exc
    momentum_sign = -1 if momentum_transform == "negate-four-momentum" else 1
    return int(helicity), int(chirality), int(spin_state), momentum_sign, exact_phase


def _project_physical_sectors(
    color_plan: GenericColorPlan,
    process: CanonicalProcessIR,
    external_legs: tuple[RecurrenceExternalLegV1, ...],
    support_mask: int,
) -> tuple[
    tuple[RecurrencePhysicalLCSectorV1, ...],
    tuple[RecurrencePublicLCFlowV1, ...],
]:
    slot_by_label = {
        int(leg.label): source_slot for source_slot, leg in enumerate(process.legs)
    }

    def slots(labels: Sequence[int], context: str) -> tuple[int, ...]:
        try:
            return tuple(slot_by_label[int(label)] for label in labels)
        except KeyError as exc:
            raise RecurrenceProjectionError(
                f"{context} references unknown external label {exc.args[0]}"
            ) from exc

    result: list[RecurrencePhysicalLCSectorV1] = []
    public_flows: list[RecurrencePublicLCFlowV1] = []
    identity_permutation = tuple(range(len(process.legs)))
    for sector in sorted(color_plan.sectors, key=lambda item: item.id):
        words = sector.color_words
        if not words:
            raise RecurrenceProjectionError(
                f"LC sector {sector.id} has no physical color word"
            )
        word = sector.word_labels or words[0]
        public_id = (
            "flow:" + ",".join(str(int(label)) for label in word)
            if word
            else "flow:singlet"
        )
        open_strings = tuple(
            RecurrenceLCOpenStringV1(
                fundamental_source_slot=slots(
                    (line.fundamental_label,), "LC fundamental endpoint"
                )[0],
                antifundamental_source_slot=slots(
                    (line.antifundamental_label,), "LC antifundamental endpoint"
                )[0],
                adjoint_source_slots=slots(
                    line.adjoint_labels, "LC open-string adjoints"
                ),
                singlet_source_slots=slots(
                    line.singlet_labels, "LC open-string singlets"
                ),
            )
            for line in sector.open_color_lines
        )
        trace_source_slots = slots(sector.trace_labels, "LC trace")
        singlet_source_slots = slots(sector.singlet_labels, "LC singlets")
        word_source_slots = slots(word, "LC color word")
        closure_source_slot, closure_proof_digest = _closure_anchor_contract(
            sector_id=int(sector.id),
            sector_kind=sector.kind,
            word_source_slots=word_source_slots,
            singlet_source_slots=singlet_source_slots,
            external_legs=external_legs,
        )
        result.append(
            RecurrencePhysicalLCSectorV1(
                sector_id=int(sector.id),
                public_id=public_id,
                kind=sector.kind,
                closure_source_slot=closure_source_slot,
                closure_proof_algorithm=_CLOSURE_ANCHOR_PROOF_ALGORITHM,
                closure_proof_digest=closure_proof_digest,
                open_strings=open_strings,
                trace_source_slots=trace_source_slots,
                singlet_source_slots=singlet_source_slots,
                word_source_slots=word_source_slots,
                support_mask=support_mask,
            )
        )
        public_flows.append(
            RecurrencePublicLCFlowV1(
                flow_id=len(public_flows),
                public_id=public_id,
                construction_sector_id=int(sector.id),
                word_source_slots=slots(word, "LC public color word"),
                source_slot_permutation=identity_permutation,
            )
        )
        if not (
            color_plan.trace_reflections_folded
            and sector.kind == "single-trace"
            and len(word) > 2
        ):
            continue
        reflected = (word[0], *reversed(word[1:]))
        if reflected == tuple(word):
            continue
        label_mapping = {
            int(left): int(right) for left, right in zip(word, reflected, strict=True)
        }
        complete_mapping = {
            int(leg.label): label_mapping.get(int(leg.label), int(leg.label))
            for leg in process.legs
        }
        permutation = tuple(
            slot_by_label[complete_mapping[int(leg.label)]] for leg in process.legs
        )
        public_flows.append(
            RecurrencePublicLCFlowV1(
                flow_id=len(public_flows),
                public_id="flow:" + ",".join(str(int(label)) for label in reflected),
                construction_sector_id=int(sector.id),
                word_source_slots=slots(reflected, "reflected LC public color word"),
                source_slot_permutation=permutation,
            )
        )
    return tuple(result), tuple(public_flows)


def _closure_anchor_contract(
    *,
    sector_id: int,
    sector_kind: str,
    word_source_slots: tuple[int, ...],
    singlet_source_slots: tuple[int, ...],
    external_legs: tuple[RecurrenceExternalLegV1, ...],
) -> tuple[int, str]:
    external_source_count = len(external_legs)
    if external_source_count <= 0:
        raise RecurrenceProjectionError(
            "a physical LC sector requires at least one external source"
        )
    if word_source_slots:
        policy = "terminal-colour-word-endpoint"
        closure_source_slot = word_source_slots[-1]
    else:
        if sector_kind != "singlet" or tuple(sorted(singlet_source_slots)) != tuple(
            range(external_source_count)
        ):
            raise RecurrenceProjectionError(
                "an LC sector without a colour word must be an all-singlet sector"
            )
        fermionic_source_slots = tuple(
            leg.source_slot for leg in external_legs if leg.is_fermionic
        )
        if fermionic_source_slots:
            policy = "minimum-fermionic-source-slot"
            closure_source_slot = min(fermionic_source_slots)
        else:
            policy = "minimum-source-slot"
            closure_source_slot = min(leg.source_slot for leg in external_legs)
    proof_digest = _digest(
        {
            "algorithm": _CLOSURE_ANCHOR_PROOF_ALGORITHM,
            "closure_source_slot": closure_source_slot,
            "external_source_count": external_source_count,
            "fermionic_source_slots": tuple(
                leg.source_slot for leg in external_legs if leg.is_fermionic
            ),
            "policy": policy,
            "sector_id": sector_id,
            "sector_kind": sector_kind,
            "singlet_source_slots": singlet_source_slots,
            "word_source_slots": word_source_slots,
        }
    )
    return closure_source_slot, proof_digest


def _project_replay_partitions(
    process: CanonicalProcessIR,
    color_plan: GenericColorPlan,
    physical_sectors: tuple[RecurrencePhysicalLCSectorV1, ...],
    replay: LCColorTopologyReplayPlan | None,
    layout: RecurrenceLCFlowLayout,
) -> tuple[RecurrenceReplayPartitionV1, ...]:
    if layout == "all-flow-union":
        return ()
    if replay is None:
        return ()
    physical_ids = tuple(sorted(int(sector.id) for sector in color_plan.sectors))
    if replay.physical_sector_ids != physical_ids:
        raise RecurrenceProjectionError(
            "topology replay proof does not cover this physical LC plan"
        )

    source_slot_by_label = {
        int(leg.label): source_slot for source_slot, leg in enumerate(process.legs)
    }
    labels = set(source_slot_by_label)
    physical_sector_by_id = {sector.sector_id: sector for sector in physical_sectors}
    partitions: list[RecurrenceReplayPartitionV1] = []
    for partition in replay.partitions:
        if partition.proof_algorithm is None or partition.proof_digest is None:
            raise RecurrenceProjectionError(
                "topology replay partition lacks an exact proof certificate"
            )
        targets: list[RecurrenceReplayTargetV1] = []
        representative_anchor = physical_sector_by_id[
            int(partition.representative_sector_id)
        ].closure_source_slot
        for sector_id, raw_mapping, weight, sign in zip(
            partition.active_sector_ids,
            partition.label_permutations,
            partition.weights,
            partition.signs,
            strict=True,
        ):
            mapping = {int(left): int(right) for left, right in raw_mapping}
            if not set(mapping).issubset(labels) or not set(mapping.values()).issubset(
                labels
            ):
                raise RecurrenceProjectionError(
                    "topology replay maps an unknown external label"
                )
            complete = {label: mapping.get(label, label) for label in labels}
            if set(complete.values()) != labels:
                raise RecurrenceProjectionError(
                    "topology replay external mapping is not a bijection"
                )
            permutation = tuple(
                source_slot_by_label[complete[int(leg.label)]] for leg in process.legs
            )
            sector = color_plan.sector(int(sector_id))
            if sector is None:
                raise RecurrenceProjectionError(
                    f"topology replay references missing LC sector {sector_id}"
                )
            expected_weight = (
                2.0
                if color_plan.trace_reflections_folded
                and sector.kind == "single-trace"
                and len(sector.trace_labels) > 2
                else 1.0
            )
            if not math.isclose(
                float(weight),
                expected_weight,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RecurrenceProjectionError(
                    "topology replay carries an unsupported non-public-flow "
                    f"multiplicity {weight!r} for sector {sector_id}"
                )
            target_anchor = physical_sector_by_id[int(sector_id)].closure_source_slot
            if permutation[representative_anchor] != target_anchor:
                raise RecurrenceProjectionError(
                    "topology replay does not map the representative closure "
                    "anchor onto the target sector anchor"
                )
            targets.append(
                RecurrenceReplayTargetV1(
                    sector_id=int(sector_id),
                    external_permutation=permutation,
                    source_slot_permutation=permutation,
                    # Rust derives and certifies source-state and closure
                    # bijections from this exact slot permutation.
                    amplitude_phase=ExactComplexRationalV1(1),
                    fermion_sign=int(sign),
                )
            )
        partitions.append(
            RecurrenceReplayPartitionV1(
                representative_sector_id=int(partition.representative_sector_id),
                materialized_sector_id=int(partition.materialized_sector_id),
                proof_algorithm=partition.proof_algorithm,
                proof_digest=partition.proof_digest,
                targets=tuple(targets),
            )
        )
    return tuple(partitions)


def _project_coupling_limits(
    limits: Mapping[str, int] | None,
) -> tuple[RecurrenceCouplingLimitV1, ...]:
    normalized = tuple(
        sorted(
            (str(raw_name).upper(), raw_maximum)
            for raw_name, raw_maximum in (limits or {}).items()
        )
    )
    names = tuple(name for name, _ in normalized)
    if len(names) != len(set(names)):
        raise RecurrenceProjectionError(
            "coupling-order names must be unique after canonicalization"
        )
    result: list[RecurrenceCouplingLimitV1] = []
    for name, raw_maximum in normalized:
        if not name:
            raise RecurrenceProjectionError("coupling-order names must be nonempty")
        if isinstance(raw_maximum, bool) or not isinstance(raw_maximum, int):
            raise RecurrenceProjectionError(
                f"coupling-order limit {name!r} must be an integer"
            )
        if raw_maximum < 0:
            raise RecurrenceProjectionError(
                f"coupling-order limit {name!r} must be nonnegative"
            )
        result.append(RecurrenceCouplingLimitV1(name, 0, raw_maximum))
    return tuple(result)


def _project_parameters(
    catalog: RecurrenceTemplateCatalog,
    template_ids: Mapping[tuple[str, str], int],
) -> tuple[RecurrenceParameterProjectionV1, ...]:
    mutable = tuple(
        sorted(
            (parameter for parameter in catalog.parameters if parameter.mutable),
            key=lambda item: (item.name, item.template_id),
        )
    )
    names = tuple(parameter.name for parameter in mutable)
    if len(names) != len(set(names)):
        raise RecurrenceProjectionError(
            "recurrence catalog exposes duplicate mutable parameter names"
        )
    result: list[RecurrenceParameterProjectionV1] = []
    for parameter in mutable:
        components = (0,) if parameter.value_type == "real" else (0, 1)
        for component in components:
            result.append(
                RecurrenceParameterProjectionV1(
                    runtime_slot=len(result),
                    runtime_name=parameter.name,
                    parameter_template_id=_template_id(
                        template_ids, "parameter", parameter.template_id
                    ),
                    prepared_parameter_id=parameter.prepared_parameter_id,
                    component=component,
                )
            )
    return tuple(result)


def _template_id(
    template_ids: Mapping[tuple[str, str], int],
    kind: str,
    semantic_id: str,
) -> int:
    try:
        return template_ids[(kind, semantic_id)]
    except KeyError as exc:
        raise RecurrenceProjectionError(
            f"missing {kind} semantic template {semantic_id!r}"
        ) from exc


def _digest(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise RecurrenceProjectionError(
            "recurrence projection input is not canonical JSON"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "RecurrenceGenerationSliceV1",
    "RecurrenceProjectionError",
    "project_recurrence_process_v1",
]
