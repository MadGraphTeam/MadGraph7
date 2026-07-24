// SPDX-License-Identifier: 0BSD

//! Deterministic lowering from a validated semantic recurrence program to the
//! direct-arena layout.
//!
//! This is deliberately a lowering-only slice. It creates immutable row-group
//! descriptors and direct-arena rows; it does not construct or invoke a
//! prepared runtime backend.

use std::collections::{BTreeMap, BTreeSet};

use super::arena::{DirectArenaLayout, recurrence_direct_arena_layout};
use super::direct_plan::{
    DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION, DIRECT_NONE_U32,
    DirectAmplitudeDestinationDescriptor, DirectClosureRow, DirectContributionRow,
    DirectCurrentDescriptor, DirectDestinationOperation, DirectExecutorRole, DirectFinalizationRow,
    DirectMomentumFormDescriptor, DirectMomentumTerm, DirectNodeKind, DirectRecurrencePlan,
    DirectRecurrencePlanParts, DirectReplayTargetDescriptor, DirectResolvedHelicityDescriptor,
    DirectResolvedSourceSelection, DirectRowGroupDescriptor, DirectSelectorDomainDescriptor,
    DirectSourceDispatchVariantDescriptor, DirectSourceEmbeddingRow, DirectSourceProjectionRow,
    DirectSourceRow, DirectSourceStateAssignment,
};
use super::layout::RuntimeSourceVariantBinding;
use super::template::{
    EvaluatorContractKind, MISSING_U32, OwnedRecurrenceTemplateInput, RuntimeHelicityVariantRow,
    ValidatedRecurrenceTemplateInput,
};
use super::{
    CanonicalMomentumLinearForm, CurrentSourceBinding, ExactComplexRational, RecurrenceCurrent,
    RecurrenceNodeKind, RecurrenceProgram, RecurrenceStrategy, SemanticDigest,
};
use crate::{RusticolError, RusticolResult};

const UNIVERSAL_SELECTOR_DOMAIN_ID: u32 = 0;
const DIRECT_ROW_FLAGS_NONE: u32 = 0;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("recurrence direct lowering: {}", message.into()))
}

/// Runtime sizing recorded in the immutable direct plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectRecurrenceRuntimeOptions {
    pub point_tile_size: u32,
    pub workspace_mib: u32,
}

impl DirectRecurrenceRuntimeOptions {
    pub fn new(point_tile_size: u32, workspace_mib: u32) -> RusticolResult<Self> {
        if point_tile_size == 0 {
            return Err(invalid("point tile size must be positive"));
        }
        if workspace_mib == 0 {
            return Err(invalid("workspace MiB must be positive"));
        }
        Ok(Self {
            point_tile_size,
            workspace_mib,
        })
    }
}

/// Stable semantic key exported by prepared direct-template-v1.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PreparedDirectExecutorKey {
    Evaluator {
        role: DirectExecutorRole,
        evaluator_binding_id: u32,
    },
    IdentityFinalizer,
}

impl PreparedDirectExecutorKey {
    const fn role(self) -> DirectExecutorRole {
        match self {
            Self::Evaluator { role, .. } => role,
            Self::IdentityFinalizer => DirectExecutorRole::Finalization,
        }
    }
}

/// One authenticated prepared direct-template mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedDirectExecutorBinding {
    pub key: PreparedDirectExecutorKey,
    pub direct_executor_id: u32,
}

impl PreparedDirectExecutorBinding {
    pub const fn evaluator(
        role: DirectExecutorRole,
        evaluator_binding_id: u32,
        direct_executor_id: u32,
    ) -> Self {
        Self {
            key: PreparedDirectExecutorKey::Evaluator {
                role,
                evaluator_binding_id,
            },
            direct_executor_id,
        }
    }

    pub const fn identity_finalizer(direct_executor_id: u32) -> Self {
        Self {
            key: PreparedDirectExecutorKey::IdentityFinalizer,
            direct_executor_id,
        }
    }
}

/// Authenticated model-level direct executor resolver.
///
/// IDs are the globally stable dense IDs owned by the complete loaded
/// direct-template catalog. Lowering never compacts or renumbers a
/// process-specific subset.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedDirectExecutorCatalog {
    direct_template_catalog_digest: SemanticDigest,
    bindings: BTreeMap<PreparedDirectExecutorKey, u32>,
    executor_roles: Box<[DirectExecutorRole]>,
}

impl PreparedDirectExecutorCatalog {
    pub fn new(
        direct_template_catalog_digest: SemanticDigest,
        bindings: Vec<PreparedDirectExecutorBinding>,
    ) -> RusticolResult<Self> {
        if bindings.is_empty() {
            return Err(invalid(
                "prepared direct-template executor catalog must not be empty",
            ));
        }
        let mut by_key = BTreeMap::new();
        let mut roles_by_id = BTreeMap::new();
        for binding in bindings {
            if matches!(
                binding.key,
                PreparedDirectExecutorKey::Evaluator {
                    evaluator_binding_id: MISSING_U32,
                    ..
                }
            ) {
                return Err(invalid(
                    "prepared direct-template mapping uses the missing evaluator-binding sentinel",
                ));
            }
            if by_key
                .insert(binding.key, binding.direct_executor_id)
                .is_some()
            {
                return Err(invalid(format!(
                    "prepared direct-template mapping repeats key {:?}",
                    binding.key
                )));
            }
            let role = binding.key.role();
            if let Some(previous) = roles_by_id.insert(binding.direct_executor_id, role)
                && previous != role
            {
                return Err(invalid(format!(
                    "prepared direct executor {} is assigned both {previous:?} and {role:?} roles",
                    binding.direct_executor_id
                )));
            }
        }
        for (expected, actual) in roles_by_id.keys().copied().enumerate() {
            if actual != u32_len("prepared direct executor", expected)? {
                return Err(invalid(format!(
                    "prepared direct executor IDs are not dense zero-based at {actual}"
                )));
            }
        }
        let executor_roles = roles_by_id
            .into_values()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Self {
            direct_template_catalog_digest,
            bindings: by_key,
            executor_roles,
        })
    }

    pub const fn direct_template_catalog_digest(&self) -> SemanticDigest {
        self.direct_template_catalog_digest
    }

    pub fn direct_executor_count(&self) -> u32 {
        self.executor_roles.len() as u32
    }

    pub fn resolve_evaluator(
        &self,
        role: DirectExecutorRole,
        evaluator_binding_id: u32,
    ) -> RusticolResult<u32> {
        let key = PreparedDirectExecutorKey::Evaluator {
            role,
            evaluator_binding_id,
        };
        if let Some(id) = self.bindings.get(&key).copied() {
            return Ok(id);
        }
        if self.bindings.keys().any(|key| {
            matches!(
                key,
                PreparedDirectExecutorKey::Evaluator {
                    evaluator_binding_id: candidate,
                    ..
                } if *candidate == evaluator_binding_id
            )
        }) {
            return Err(invalid(format!(
                "prepared direct-template mapping for evaluator binding {evaluator_binding_id} has a role mismatch; expected {role:?}"
            )));
        }
        Err(invalid(format!(
            "prepared direct-template catalog has no {role:?} mapping for evaluator binding {evaluator_binding_id}"
        )))
    }

    pub fn resolve_identity_finalizer(&self) -> RusticolResult<u32> {
        self.bindings
            .get(&PreparedDirectExecutorKey::IdentityFinalizer)
            .copied()
            .ok_or_else(|| {
                invalid(
                    "prepared direct-template catalog has no generic identity-finalizer binding",
                )
            })
    }
}

#[derive(Clone, Copy, Debug)]
struct SourceDraft {
    stage: u16,
    executor_id: u32,
    semantic_current_id: u32,
    row: DirectSourceRow,
}

#[derive(Clone, Copy, Debug)]
struct ContributionDraft {
    stage: u16,
    executor_id: u32,
    semantic_contribution_id: u32,
    semantic_result_current_id: u32,
    row: DirectContributionRow,
}

#[derive(Clone, Copy, Debug)]
struct FinalizationDraft {
    stage: u16,
    executor_id: u32,
    semantic_current_id: u32,
    row: DirectFinalizationRow,
}

#[derive(Clone, Copy, Debug)]
struct ClosureDraft {
    stage: u16,
    executor_id: u32,
    semantic_closure_id: u32,
    amplitude_destination_id: u32,
    row: DirectClosureRow,
}

#[derive(Clone, Copy, Debug)]
struct DirectParents {
    parent0_component_base: u32,
    parent1_component_base_or_sentinel: u32,
    parent0_momentum_form_id: u32,
    parent1_momentum_form_id_or_sentinel: u32,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RuntimeSourceChoice {
    pub(super) source_state_index: u32,
    pub(super) public_helicity: i32,
    pub(super) dispatch_variant_id: u32,
}

#[derive(Debug, Default)]
struct LoweredRuntimeSources {
    variants: Vec<DirectSourceDispatchVariantDescriptor>,
    embeddings: Vec<DirectSourceEmbeddingRow>,
    projections: Vec<DirectSourceProjectionRow>,
    choices_by_source: Vec<Vec<RuntimeSourceChoice>>,
}

/// Lower one validated recurrence program to validated plan-v2 parts.
///
/// `semantic_digest` authenticates the process semantic plan.
/// `prepared_pack_digest` must authenticate the prepared pack whose evaluator
/// bindings are present in `templates`. `direct_template_catalog_digest`
/// authenticates `direct_executors`.
pub fn lower_recurrence_direct_v2(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct_executors: &PreparedDirectExecutorCatalog,
    semantic_digest: SemanticDigest,
    prepared_pack_digest: SemanticDigest,
    direct_template_catalog_digest: SemanticDigest,
    runtime_options: DirectRecurrenceRuntimeOptions,
) -> RusticolResult<DirectRecurrencePlanParts> {
    DirectRecurrencePlan::new(build_direct_parts(
        program,
        templates,
        direct_executors,
        semantic_digest,
        prepared_pack_digest,
        direct_template_catalog_digest,
        runtime_options,
    )?)
    .map(DirectRecurrencePlan::into_parts)
}

/// Lower one validated recurrence program directly to an immutable plan.
pub fn lower_recurrence_direct_plan_v2(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct_executors: &PreparedDirectExecutorCatalog,
    semantic_digest: SemanticDigest,
    prepared_pack_digest: SemanticDigest,
    direct_template_catalog_digest: SemanticDigest,
    runtime_options: DirectRecurrenceRuntimeOptions,
) -> RusticolResult<DirectRecurrencePlan> {
    DirectRecurrencePlan::new(build_direct_parts(
        program,
        templates,
        direct_executors,
        semantic_digest,
        prepared_pack_digest,
        direct_template_catalog_digest,
        runtime_options,
    )?)
}

fn build_direct_parts(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct_executors: &PreparedDirectExecutorCatalog,
    semantic_digest: SemanticDigest,
    prepared_pack_digest: SemanticDigest,
    direct_template_catalog_digest: SemanticDigest,
    runtime_options: DirectRecurrenceRuntimeOptions,
) -> RusticolResult<DirectRecurrencePlanParts> {
    program.validate()?;
    validate_lowering_boundary(
        program,
        templates,
        direct_executors,
        prepared_pack_digest,
        direct_template_catalog_digest,
        runtime_options,
    )?;

    let input = templates.input();
    let external_source_count = external_source_count(program)?;
    let component_counts = component_counts(program, templates)?;
    let arena = recurrence_direct_arena_layout(program, &component_counts)?;
    arena.validate()?;

    let (momentum_ids, momentum_forms, momentum_terms) =
        intern_momentum_forms(program, external_source_count)?;
    let (mut exact_factors, mut exact_factor_ids) = intern_exact_factors(program)?;
    let mut closure_factor_blocks = BTreeMap::new();

    let mut source_drafts = Vec::new();
    let mut runtime_sources = LoweredRuntimeSources {
        choices_by_source: vec![Vec::new(); external_source_count as usize],
        ..LoweredRuntimeSources::default()
    };
    for current in program
        .currents()
        .iter()
        .filter(|current| current.is_source())
    {
        let assignment = arena_assignment(&arena, current.id())?;
        let source_slot = only_source_slot(current)?;
        match (program.strategy(), current.key().source_binding()) {
            (
                RecurrenceStrategy::TopologyReplay,
                CurrentSourceBinding::FixedTemplate(source_template_id),
            ) => {
                let source_factor = current.source_exact_factor().ok_or_else(|| {
                    invalid(format!(
                        "topology-replay source current {} has no exact source factor",
                        current.id()
                    ))
                })?;
                let source = input
                    .sources
                    .get(*source_template_id as usize)
                    .ok_or_else(|| {
                        invalid(format!(
                            "source current {} references absent source template {source_template_id}",
                            current.id()
                        ))
                    })?;
                let current_state =
                    state_row(templates, current.key().current_state_template_id())?;
                let source_state = state_row(templates, source.state_template_id)?;
                if current_state.dimension != source_state.dimension {
                    return Err(invalid(format!(
                        "source current {} and source template {source_template_id} have different component dimensions",
                        current.id()
                    )));
                }
                let evaluator_binding_id = validated_evaluator_binding_id(
                    templates,
                    source.evaluator_binding_id,
                    EvaluatorContractKind::Source,
                    "source",
                )?;
                let executor_id = direct_executors
                    .resolve_evaluator(DirectExecutorRole::Source, evaluator_binding_id)?;
                source_drafts.push(SourceDraft {
                    stage: current_stage(current)?,
                    executor_id,
                    semantic_current_id: current.id(),
                    row: DirectSourceRow {
                        source_slot,
                        destination_component_base: assignment.component_base,
                        momentum_form_id: momentum_id(&momentum_ids, current.key().momentum())?,
                        source_template_or_dispatch_domain: *source_template_id,
                        spin_state_class: current.key().spin_state_class(),
                        exact_factor_id: exact_factor_id(&exact_factor_ids, source_factor)?,
                        selector_domain_id: UNIVERSAL_SELECTOR_DOMAIN_ID,
                    },
                });
            }
            (
                RecurrenceStrategy::AllFlowUnion,
                CurrentSourceBinding::RuntimeDispatch {
                    domain,
                    source_template_ids,
                    variant_bindings,
                },
            ) => {
                lower_union_source(
                    current,
                    source_slot,
                    *domain,
                    source_template_ids,
                    variant_bindings,
                    templates,
                    direct_executors,
                    &arena,
                    &momentum_ids,
                    &mut exact_factors,
                    &mut exact_factor_ids,
                    &mut source_drafts,
                    &mut runtime_sources,
                )?;
            }
            (RecurrenceStrategy::TopologyReplay, _) => {
                return Err(invalid(format!(
                    "topology-replay source current {} does not use one fixed source template",
                    current.id()
                )));
            }
            (RecurrenceStrategy::AllFlowUnion, _) => {
                return Err(invalid(format!(
                    "all-flow-union source current {} does not use runtime dispatch",
                    current.id()
                )));
            }
        }
    }

    let mut contribution_drafts = Vec::with_capacity(program.contributions().len());
    for contribution in program.contributions() {
        validate_direct_arity(
            "contribution",
            contribution.id(),
            contribution.parent_current_ids().len(),
        )?;
        let result = program
            .currents()
            .get(contribution.result_current_id() as usize)
            .ok_or_else(|| invalid("contribution result current is absent"))?;
        let transition_id = contribution.key().transition_template_id();
        let transition = input
            .transitions
            .get(transition_id as usize)
            .ok_or_else(|| {
                invalid(format!(
                    "contribution {} references absent transition template {transition_id}",
                    contribution.id()
                ))
            })?;
        let expected_parent_states = template_u32_sequence(
            templates,
            transition.input_state_sequence_id,
            "transition input states",
        )?;
        let canonical_input_order = template_u32_sequence(
            templates,
            transition.canonical_input_order_sequence_id,
            "transition canonical input order",
        )?;
        let actual_parent_states =
            parent_state_template_ids(program, contribution.parent_current_ids())?;
        if !parent_state_contract_matches(
            expected_parent_states,
            canonical_input_order,
            transition.input_exchange_factor_id != MISSING_U32,
            &actual_parent_states,
        )? || contribution.key().parent_state_template_ids() != actual_parent_states
        {
            return Err(invalid(format!(
                "contribution {} parent-state contract does not match transition {transition_id}",
                contribution.id()
            )));
        }
        if transition.result_state_template_id != result.key().current_state_template_id()
            || transition.result_state_template_id != contribution.key().result_state_template_id()
        {
            return Err(invalid(format!(
                "contribution {} result-state contract does not match transition {transition_id}",
                contribution.id()
            )));
        }
        if transition.quantum_flow_template_id != contribution.key().quantum_flow_witness_id() {
            return Err(invalid(format!(
                "contribution {} quantum-flow witness does not match transition {transition_id}",
                contribution.id()
            )));
        }
        validate_coupling_digest(templates, contribution)?;
        if transition.output_projection_string_id != contribution.key().output_projection_id() {
            return Err(invalid(format!(
                "contribution {} output projection does not match transition {transition_id}",
                contribution.id()
            )));
        }
        let color_witness = contribution.key().color_witness_term_id();
        if color_witness.color_contraction_template_id() != transition.color_contraction_template_id
        {
            return Err(invalid(format!(
                "contribution {} color witness does not belong to transition {transition_id}",
                contribution.id()
            )));
        }
        let contraction = input
            .color_contractions
            .get(transition.color_contraction_template_id as usize)
            .ok_or_else(|| invalid("transition color contraction is absent"))?;
        if u64::from(color_witness.witness_ordinal()) >= contraction.witness_count {
            return Err(invalid(format!(
                "contribution {} color-witness ordinal is out of bounds",
                contribution.id()
            )));
        }
        validate_parent_momentum_contract(program, contribution)?;
        let parents = direct_parents(
            program,
            contribution.parent_current_ids(),
            &arena,
            &momentum_ids,
        )?;
        let evaluator_binding_id = validated_evaluator_binding_id(
            templates,
            transition.evaluator_binding_id,
            EvaluatorContractKind::Vertex,
            "contribution",
        )?;
        let executor_id = direct_executors
            .resolve_evaluator(DirectExecutorRole::Contribution, evaluator_binding_id)?;
        contribution_drafts.push(ContributionDraft {
            stage: current_stage(result)?,
            executor_id,
            semantic_contribution_id: contribution.id(),
            semantic_result_current_id: result.id(),
            row: DirectContributionRow {
                parent0_component_base: parents.parent0_component_base,
                parent1_component_base_or_sentinel: parents.parent1_component_base_or_sentinel,
                parent0_momentum_form_id: parents.parent0_momentum_form_id,
                parent1_momentum_form_id_or_sentinel: parents.parent1_momentum_form_id_or_sentinel,
                destination_component_base: arena_assignment(&arena, result.id())?.component_base,
                exact_factor_id: exact_factor_id(&exact_factor_ids, contribution.exact_factor())?,
                selector_domain_id: UNIVERSAL_SELECTOR_DOMAIN_ID,
                flags: DIRECT_ROW_FLAGS_NONE,
            },
        });
    }

    let mut finalization_drafts = Vec::with_capacity(program.finalizations().len());
    for finalization in program.finalizations() {
        let current = program
            .currents()
            .get(finalization.current_id() as usize)
            .ok_or_else(|| invalid("finalization current is absent"))?;
        if finalization.propagator_template_id().is_none()
            && finalization.exact_factor() == ExactComplexRational::ONE
        {
            continue;
        }
        let executor_id = if let Some(propagator_template_id) =
            finalization.propagator_template_id()
        {
            let propagator = input
                .propagators
                .get(propagator_template_id as usize)
                .ok_or_else(|| {
                    invalid(format!(
                        "finalization {} references absent propagator template {propagator_template_id}",
                        finalization.id()
                    ))
                })?;
            if propagator.applies_propagator == 0
                || propagator.state_template_id != current.key().current_state_template_id()
            {
                return Err(invalid(format!(
                    "finalization {} does not match an active propagator for current {}",
                    finalization.id(),
                    current.id()
                )));
            }
            let evaluator_binding_id = validated_evaluator_binding_id(
                templates,
                propagator.evaluator_binding_id,
                EvaluatorContractKind::Propagator,
                "finalization",
            )?;
            direct_executors
                .resolve_evaluator(DirectExecutorRole::Finalization, evaluator_binding_id)?
        } else {
            direct_executors.resolve_identity_finalizer()?
        };
        let assignment = arena_assignment(&arena, current.id())?;
        finalization_drafts.push(FinalizationDraft {
            stage: current_stage(current)?,
            executor_id,
            semantic_current_id: current.id(),
            row: DirectFinalizationRow {
                component_base: assignment.component_base,
                component_count: u16::try_from(assignment.component_count).map_err(|_| {
                    invalid(format!(
                        "current {} component count exceeds u16",
                        current.id()
                    ))
                })?,
                momentum_form_id: momentum_id(&momentum_ids, current.key().momentum())?,
                exact_factor_id: exact_factor_id(&exact_factor_ids, finalization.exact_factor())?,
                selector_domain_id: UNIVERSAL_SELECTOR_DOMAIN_ID,
                flags: DIRECT_ROW_FLAGS_NONE,
            },
        });
    }

    let closure_stage = program
        .currents()
        .iter()
        .map(current_stage)
        .collect::<RusticolResult<Vec<_>>>()?
        .into_iter()
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| invalid("closure stage exceeds u16"))?;
    let mut closure_drafts = Vec::with_capacity(program.closure_terms().len());
    for closure in program.closure_terms() {
        validate_direct_arity("closure", closure.id(), closure.parent_current_ids().len())?;
        let closure_template_id = closure.closure_template_id();
        let template = input
            .closures
            .get(closure_template_id as usize)
            .ok_or_else(|| {
                invalid(format!(
                    "closure {} references absent closure template {closure_template_id}",
                    closure.id()
                ))
            })?;
        let expected_parent_states = template_u32_sequence(
            templates,
            template.input_state_sequence_id,
            "closure input states",
        )?;
        let canonical_input_order = template_u32_sequence(
            templates,
            template.canonical_input_order_sequence_id,
            "closure canonical input order",
        )?;
        let actual_parent_states =
            parent_state_template_ids(program, closure.parent_current_ids())?;
        if !parent_state_contract_matches(
            expected_parent_states,
            canonical_input_order,
            template.input_exchange_factor_id != MISSING_U32,
            &actual_parent_states,
        )? {
            return Err(invalid(format!(
                "closure {} parent-state contract does not match template {closure_template_id}",
                closure.id()
            )));
        }
        let eligible_flows = template_u32_sequence(
            templates,
            template.eligible_quantum_flow_sequence_id,
            "closure eligible quantum flows",
        )?;
        match closure.quantum_flow_template_id() {
            Some(flow_id) if eligible_flows.binary_search(&flow_id).is_ok() => {}
            Some(flow_id) => {
                return Err(invalid(format!(
                    "closure {} quantum-flow witness {flow_id} is not eligible for template {closure_template_id}",
                    closure.id()
                )));
            }
            None if eligible_flows.is_empty() => {}
            None => {
                return Err(invalid(format!(
                    "closure {} omits the quantum-flow witness required by template {closure_template_id}",
                    closure.id()
                )));
            }
        }
        let evaluator_binding_id = validated_evaluator_binding_id(
            templates,
            template.evaluator_binding_id,
            EvaluatorContractKind::Closure,
            "closure",
        )?;
        let executor_id = direct_executors
            .resolve_evaluator(DirectExecutorRole::Closure, evaluator_binding_id)?;
        let parents = direct_parents(program, closure.parent_current_ids(), &arena, &momentum_ids)?;
        let component_coefficients =
            templates.closure_component_coefficients(closure_template_id)?;
        let parent0_id = closure.parent_current_ids()[0] as usize;
        let component_count = *component_counts
            .get(parent0_id)
            .ok_or_else(|| invalid("closure parent 0 component count is absent"))?;
        if let Some(parent1_id) = closure.parent_current_ids().get(1) {
            let parent1_component_count = *component_counts
                .get(*parent1_id as usize)
                .ok_or_else(|| invalid("closure parent 1 component count is absent"))?;
            if parent1_component_count != component_count {
                return Err(invalid(format!(
                    "closure {} parent component counts differ",
                    closure.id()
                )));
            }
        }
        if component_coefficients.len() != component_count as usize {
            return Err(invalid(format!(
                "closure {} has {} component coefficients for {component_count} current components",
                closure.id(),
                component_coefficients.len()
            )));
        }
        let (component_factor_start, component_count) = intern_closure_factor_block(
            &component_coefficients,
            &mut exact_factors,
            &mut closure_factor_blocks,
        )?;
        closure_drafts.push(ClosureDraft {
            stage: closure_stage,
            executor_id,
            semantic_closure_id: closure.id(),
            amplitude_destination_id: closure.target_destination_id(),
            row: DirectClosureRow {
                parent0_component_base: parents.parent0_component_base,
                parent1_component_base_or_sentinel: parents.parent1_component_base_or_sentinel,
                parent0_momentum_form_id: parents.parent0_momentum_form_id,
                parent1_momentum_form_id_or_sentinel: parents.parent1_momentum_form_id_or_sentinel,
                amplitude_destination_id: closure.target_destination_id(),
                exact_factor_id: exact_factor_id(&exact_factor_ids, closure.exact_factor())?,
                component_factor_start,
                component_count,
                selector_domain_id: UNIVERSAL_SELECTOR_DOMAIN_ID,
                flags: DIRECT_ROW_FLAGS_NONE,
            },
        });
    }

    source_drafts.sort_by_key(|draft| (draft.stage, draft.executor_id, draft.semantic_current_id));
    contribution_drafts.sort_by_key(|draft| {
        (
            draft.stage,
            draft.executor_id,
            draft.semantic_contribution_id,
        )
    });
    let mut initialized_currents = BTreeSet::new();
    for draft in &mut contribution_drafts {
        if initialized_currents.insert(draft.semantic_result_current_id) {
            draft.row.flags |= DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION;
        }
    }
    finalization_drafts
        .sort_by_key(|draft| (draft.stage, draft.executor_id, draft.semantic_current_id));
    // Destination ownership is primary. Each destination retains exactly one
    // contiguous closure range; executor changes create additional row groups.
    closure_drafts.sort_by_key(|draft| {
        (
            draft.amplitude_destination_id,
            draft.stage,
            draft.executor_id,
            draft.semantic_closure_id,
        )
    });

    let mut source_row_by_current = vec![DIRECT_NONE_U32; program.currents().len()];
    for (row_index, draft) in source_drafts.iter().enumerate() {
        source_row_by_current[draft.semantic_current_id as usize] =
            u32_len("source row", row_index)?;
    }
    let mut finalization_row_by_current = vec![DIRECT_NONE_U32; program.currents().len()];
    for (row_index, draft) in finalization_drafts.iter().enumerate() {
        finalization_row_by_current[draft.semantic_current_id as usize] =
            u32_len("finalization row", row_index)?;
    }

    let currents = program
        .currents()
        .iter()
        .map(|current| {
            let assignment = arena_assignment(&arena, current.id())?;
            Ok(DirectCurrentDescriptor {
                semantic_current_id: current.id(),
                node_kind: match current.key().node_kind() {
                    RecurrenceNodeKind::Source => DirectNodeKind::Source,
                    RecurrenceNodeKind::Current => DirectNodeKind::Current,
                },
                state_template_id: current.key().current_state_template_id(),
                component_base: assignment.component_base,
                component_count: u16::try_from(assignment.component_count).map_err(|_| {
                    invalid(format!(
                        "current {} component count exceeds u16",
                        current.id()
                    ))
                })?,
                momentum_form_id: momentum_id(&momentum_ids, current.key().momentum())?,
                stage: current_stage(current)?,
                selector_domain_id: UNIVERSAL_SELECTOR_DOMAIN_ID,
                first_use: direct_event(assignment.first_use, "arena first use")?,
                last_use: direct_event(assignment.last_use, "arena last use")?,
                source_row_or_sentinel: source_row_by_current[current.id() as usize],
                finalization_row_or_sentinel: finalization_row_by_current[current.id() as usize],
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let mut row_groups = Vec::new();
    append_row_groups(
        source_drafts
            .iter()
            .map(|draft| (draft.stage, draft.executor_id)),
        DirectExecutorRole::Source,
        DirectDestinationOperation::Initialize,
        &mut row_groups,
    )?;
    append_row_groups(
        contribution_drafts
            .iter()
            .map(|draft| (draft.stage, draft.executor_id)),
        DirectExecutorRole::Contribution,
        DirectDestinationOperation::Add,
        &mut row_groups,
    )?;
    append_row_groups(
        finalization_drafts
            .iter()
            .map(|draft| (draft.stage, draft.executor_id)),
        DirectExecutorRole::Finalization,
        DirectDestinationOperation::FinalizeInPlace,
        &mut row_groups,
    )?;
    append_row_groups(
        closure_drafts
            .iter()
            .map(|draft| (draft.stage, draft.executor_id)),
        DirectExecutorRole::Closure,
        DirectDestinationOperation::ClosureAdd,
        &mut row_groups,
    )?;
    row_groups.sort_by_key(|row_group| (row_group.stage, row_group.role, row_group.row_start));

    let mut source_permutations = Vec::new();
    let mut replay_targets = Vec::with_capacity(program.replay_targets().len());
    for target in program.replay_targets() {
        let source_permutation_start = u64::try_from(source_permutations.len())
            .map_err(|_| invalid("source permutation table exceeds u64"))?;
        source_permutations.extend_from_slice(target.source_slot_permutation());
        replay_targets.push(DirectReplayTargetDescriptor {
            public_flow_id: target.target_sector_id(),
            representative_id: target.materialized_sector_id(),
            source_permutation_start,
            source_permutation_count: u32_len(
                "replay source permutation",
                target.source_slot_permutation().len(),
            )?,
            phase_exact_factor_id: exact_factor_id(&exact_factor_ids, target.amplitude_factor())?,
            // RecurrenceProgram stores one exact row per public target and has
            // no separate orbit-multiplicity field.
            multiplicity: 1,
            selector_domain_id: UNIVERSAL_SELECTOR_DOMAIN_ID,
        });
    }

    let amplitude_destinations = lower_amplitude_destinations(program, &closure_drafts)?;
    let (
        resolved_helicities,
        source_state_assignments,
        resolved_source_selections,
        public_helicities,
    ) = match program.strategy() {
        RecurrenceStrategy::TopologyReplay => {
            let (descriptors, source_states, public_helicities) =
                lower_resolved_helicities(program)?;
            (descriptors, source_states, Vec::new(), public_helicities)
        }
        RecurrenceStrategy::AllFlowUnion => lower_union_resolved_helicities(
            &runtime_sources.choices_by_source,
            program.retained_helicity_count(),
        )?,
    };

    let state_template_count = u32_len("state template", input.current_states.len())?;
    let source_template_count = u32_len("source template", input.sources.len())?;
    let direct_executor_count = direct_executors.direct_executor_count();
    if state_template_count == 0 || source_template_count == 0 || direct_executor_count == 0 {
        return Err(invalid(
            "direct lowering requires nonempty state, source, and executor catalogs",
        ));
    }

    Ok(DirectRecurrencePlanParts {
        strategy: program.strategy(),
        semantic_digest,
        prepared_pack_digest,
        direct_template_catalog_digest,
        point_tile_size: runtime_options.point_tile_size,
        workspace_mib: runtime_options.workspace_mib,
        current_arena_components: arena.component_count(),
        physical_sector_count: program.physical_sector_count(),
        retained_helicity_count: program.retained_helicity_count(),
        amplitude_destination_count: u32_len(
            "amplitude destination",
            program.amplitude_destinations().len(),
        )?,
        parameter_value_count: templates.summary().parameter_count,
        external_source_count,
        state_template_count,
        source_template_count,
        source_template_or_dispatch_count: match program.strategy() {
            RecurrenceStrategy::TopologyReplay => source_template_count,
            RecurrenceStrategy::AllFlowUnion => templates.summary().runtime_helicity_contract_count,
        },
        runtime_helicity_contract_count: match program.strategy() {
            RecurrenceStrategy::TopologyReplay => 0,
            RecurrenceStrategy::AllFlowUnion => templates.summary().runtime_helicity_contract_count,
        },
        runtime_helicity_variant_count: match program.strategy() {
            RecurrenceStrategy::TopologyReplay => 0,
            RecurrenceStrategy::AllFlowUnion => u32_len(
                "runtime-helicity variant",
                input.runtime_helicity_variants.len(),
            )?,
        },
        direct_executor_count,
        currents,
        sources: source_drafts.iter().map(|draft| draft.row).collect(),
        contributions: contribution_drafts.iter().map(|draft| draft.row).collect(),
        finalizations: finalization_drafts.iter().map(|draft| draft.row).collect(),
        closures: closure_drafts.iter().map(|draft| draft.row).collect(),
        row_groups,
        momentum_forms,
        momentum_terms,
        // The compact topology-replay program is already liveness-pruned.
        // Until process selector-domain rows are added to RecurrenceProgram,
        // every retained direct row belongs to one universal domain.
        selector_domains: vec![DirectSelectorDomainDescriptor {
            word_start: 0,
            word_count: 1,
        }],
        selector_words: vec![u64::MAX],
        replay_targets,
        source_permutations,
        amplitude_destinations,
        resolved_helicities,
        source_state_assignments,
        source_dispatch_variants: runtime_sources.variants,
        source_embeddings: runtime_sources.embeddings,
        source_projections: runtime_sources.projections,
        resolved_source_selections,
        public_helicities,
        exact_factors,
    })
}

fn validate_lowering_boundary(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    direct_executors: &PreparedDirectExecutorCatalog,
    prepared_pack_digest: SemanticDigest,
    direct_template_catalog_digest: SemanticDigest,
    runtime_options: DirectRecurrenceRuntimeOptions,
) -> RusticolResult<()> {
    if runtime_options.point_tile_size == 0 || runtime_options.workspace_mib == 0 {
        return Err(invalid(
            "point tile size and workspace MiB must both be positive",
        ));
    }
    let summary = templates.summary();
    if prepared_pack_digest != summary.prepared_kernel_pack_digest {
        return Err(invalid(format!(
            "prepared-pack digest {prepared_pack_digest} does not match validated template pack {}",
            summary.prepared_kernel_pack_digest
        )));
    }
    if direct_template_catalog_digest != direct_executors.direct_template_catalog_digest() {
        return Err(invalid(format!(
            "direct-template catalog digest {direct_template_catalog_digest} does not match resolver catalog {}",
            direct_executors.direct_template_catalog_digest()
        )));
    }
    validate_prepared_parameter_catalog(templates)?;
    validate_direct_executor_catalog(templates, direct_executors)?;
    for current in program.currents() {
        if current.key().catalog_digest() != summary.catalog_digest {
            return Err(invalid(format!(
                "current {} template-catalog digest does not match the validated catalog",
                current.id()
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn lower_union_source(
    current: &RecurrenceCurrent,
    source_slot: u32,
    dispatch_domain: u32,
    advertised_source_templates: &[u32],
    variant_bindings: &[RuntimeSourceVariantBinding],
    templates: &ValidatedRecurrenceTemplateInput,
    direct_executors: &PreparedDirectExecutorCatalog,
    arena: &DirectArenaLayout,
    momentum_ids: &BTreeMap<CanonicalMomentumLinearForm, u32>,
    exact_factors: &mut Vec<ExactComplexRational>,
    exact_factor_ids: &mut BTreeMap<ExactComplexRational, u32>,
    source_drafts: &mut Vec<SourceDraft>,
    runtime_sources: &mut LoweredRuntimeSources,
) -> RusticolResult<()> {
    if current.source_exact_factor().is_some() {
        return Err(invalid(format!(
            "all-flow-union source current {} must take its factor from runtime dispatch",
            current.id()
        )));
    }
    if variant_bindings.is_empty() {
        return Err(invalid(format!(
            "all-flow-union source current {} has no runtime variants",
            current.id()
        )));
    }
    let input = templates.input();
    let contract = input
        .runtime_helicity_contracts
        .get(dispatch_domain as usize)
        .ok_or_else(|| {
            invalid(format!(
                "source current {} references absent runtime-helicity contract {dispatch_domain}",
                current.id()
            ))
        })?;
    if contract.id != dispatch_domain {
        return Err(invalid(format!(
            "runtime-helicity contract {dispatch_domain} has noncanonical ID {}",
            contract.id
        )));
    }
    if contract.full_state_template_id != current.key().current_state_template_id() {
        return Err(invalid(format!(
            "source current {} state does not match runtime-helicity contract {dispatch_domain}",
            current.id()
        )));
    }
    let contract_variant_range = contract.variant_range.as_usize_range(
        input.runtime_helicity_variants.len(),
        "runtime-helicity contract variants",
    )?;
    let assignment = arena_assignment(arena, current.id())?;
    let full_state = state_row(templates, contract.full_state_template_id)?;
    if assignment.component_count != u32::from(full_state.dimension) {
        return Err(invalid(format!(
            "source current {} arena width does not match runtime-helicity full state",
            current.id()
        )));
    }
    let source_row_id = u32_len("source row", source_drafts.len())?;
    let source_choices = runtime_sources
        .choices_by_source
        .get_mut(source_slot as usize)
        .ok_or_else(|| invalid(format!("source slot {source_slot} is out of range")))?;
    if !source_choices.is_empty() {
        return Err(invalid(format!(
            "all-flow-union source slot {source_slot} has more than one dispatch current"
        )));
    }
    source_drafts.push(SourceDraft {
        stage: current_stage(current)?,
        executor_id: DIRECT_NONE_U32,
        semantic_current_id: current.id(),
        row: DirectSourceRow {
            source_slot,
            destination_component_base: assignment.component_base,
            momentum_form_id: momentum_id(momentum_ids, current.key().momentum())?,
            source_template_or_dispatch_domain: dispatch_domain,
            spin_state_class: current.key().spin_state_class(),
            exact_factor_id: intern_direct_factor(
                ExactComplexRational::ONE,
                exact_factors,
                exact_factor_ids,
            )?,
            selector_domain_id: UNIVERSAL_SELECTOR_DOMAIN_ID,
        },
    });

    for binding in variant_bindings.iter().copied() {
        if advertised_source_templates
            .binary_search(&binding.source_template_id())
            .is_err()
        {
            return Err(invalid(format!(
                "source current {} runtime variant {} is absent from its advertised source-template catalog",
                current.id(),
                binding.runtime_variant_id()
            )));
        }
        let variant_index = usize::try_from(binding.runtime_variant_id())
            .map_err(|_| invalid("runtime-helicity variant ID exceeds usize"))?;
        if !contract_variant_range.contains(&variant_index) {
            return Err(invalid(format!(
                "source current {} runtime variant {} is outside contract {dispatch_domain}",
                current.id(),
                binding.runtime_variant_id()
            )));
        }
        let variant = input
            .runtime_helicity_variants
            .get(variant_index)
            .ok_or_else(|| invalid("runtime-helicity variant is absent"))?;
        if variant.id != binding.runtime_variant_id()
            || variant.contract_id != dispatch_domain
            || variant.source_template_id != binding.source_template_id()
            || variant.source_state_template_id != binding.source_state_template_id()
        {
            return Err(invalid(format!(
                "source current {} runtime variant {} disagrees with its prepared contract",
                current.id(),
                binding.runtime_variant_id()
            )));
        }
        let source = input
            .sources
            .get(binding.source_template_id() as usize)
            .ok_or_else(|| invalid("runtime source template is absent"))?;
        if source.id != binding.source_template_id()
            || source.state_template_id != binding.source_state_template_id()
        {
            return Err(invalid(format!(
                "runtime source template {} has a stale state contract",
                binding.source_template_id()
            )));
        }
        state_row(templates, binding.crossed_state_template_id())?;
        let evaluator_binding_id = validated_evaluator_binding_id(
            templates,
            source.evaluator_binding_id,
            EvaluatorContractKind::Source,
            "runtime source",
        )?;
        let direct_executor_id =
            direct_executors.resolve_evaluator(DirectExecutorRole::Source, evaluator_binding_id)?;

        let effective_variant = effective_runtime_helicity_variant(
            input,
            contract_variant_range.clone(),
            binding.crossed_state_template_id(),
        )?;
        let embedding_start = u64::try_from(runtime_sources.embeddings.len())
            .map_err(|_| invalid("runtime source embeddings exceed u64"))?;
        let embedding_range = effective_variant.embedding_range.as_usize_range(
            input.runtime_helicity_embeddings.len(),
            "runtime source embeddings",
        )?;
        for embedding in &input.runtime_helicity_embeddings[embedding_range] {
            if embedding.variant_id != effective_variant.id {
                return Err(invalid(format!(
                    "runtime variant {} owns a stale embedding row",
                    effective_variant.id
                )));
            }
            let factor = template_exact_factor(templates, embedding.factor_id)?;
            runtime_sources.embeddings.push(DirectSourceEmbeddingRow {
                full_component: embedding.full_component,
                source_component_or_sentinel: if embedding.source_component == MISSING_U32 {
                    DIRECT_NONE_U32
                } else {
                    embedding.source_component
                },
                exact_factor_id: intern_direct_factor(factor, exact_factors, exact_factor_ids)?,
            });
        }

        let projection_start = u64::try_from(runtime_sources.projections.len())
            .map_err(|_| invalid("runtime source projections exceed u64"))?;
        let projection_range = effective_variant.projection_range.as_usize_range(
            input.runtime_helicity_projections.len(),
            "runtime source projections",
        )?;
        for projection in &input.runtime_helicity_projections[projection_range] {
            if projection.variant_id != effective_variant.id {
                return Err(invalid(format!(
                    "runtime variant {} owns a stale projection row",
                    effective_variant.id
                )));
            }
            runtime_sources.projections.push(DirectSourceProjectionRow {
                source_component: projection.source_component,
                full_component: projection.full_component,
            });
        }

        let dispatch_variant_id = u32_len(
            "lowered source dispatch variant",
            runtime_sources.variants.len(),
        )?;
        runtime_sources
            .variants
            .push(DirectSourceDispatchVariantDescriptor {
                embedding_start,
                projection_start,
                source_row_id,
                dispatch_domain_id: dispatch_domain,
                runtime_variant_id: binding.runtime_variant_id(),
                source_state_index: binding.source_state_index(),
                source_template_id: binding.source_template_id(),
                source_state_template_id: binding.source_state_template_id(),
                crossed_state_template_id: binding.crossed_state_template_id(),
                crossed_spin_state_class: binding.crossed_spin_state_class(),
                direct_executor_id,
                crossing_exact_factor_id: intern_direct_factor(
                    binding.crossing_factor(),
                    exact_factors,
                    exact_factor_ids,
                )?,
                embedding_count: u32_len(
                    "runtime source embedding",
                    runtime_sources.embeddings.len()
                        - usize::try_from(embedding_start)
                            .map_err(|_| invalid("embedding start exceeds usize"))?,
                )?,
                projection_count: u32_len(
                    "runtime source projection",
                    runtime_sources.projections.len()
                        - usize::try_from(projection_start)
                            .map_err(|_| invalid("projection start exceeds usize"))?,
                )?,
            });
        source_choices.push(RuntimeSourceChoice {
            source_state_index: binding.source_state_index(),
            public_helicity: binding.public_helicity(),
            dispatch_variant_id,
        });
    }
    source_choices.sort_by_key(|choice| choice.source_state_index);
    if source_choices
        .windows(2)
        .any(|pair| pair[0].source_state_index == pair[1].source_state_index)
    {
        return Err(invalid(format!(
            "all-flow-union source slot {source_slot} repeats a source-state variant"
        )));
    }
    Ok(())
}

pub(super) fn effective_runtime_helicity_variant(
    input: &OwnedRecurrenceTemplateInput,
    contract_variant_range: std::ops::Range<usize>,
    crossed_state_template_id: u32,
) -> RusticolResult<RuntimeHelicityVariantRow> {
    let candidates = input.runtime_helicity_variants[contract_variant_range]
        .iter()
        .copied()
        .filter(|variant| variant.source_state_template_id == crossed_state_template_id)
        .collect::<Vec<_>>();
    let Some(representative) = candidates.first().copied() else {
        return Err(invalid(format!(
            "runtime-helicity contract has no embedding for crossed state {crossed_state_template_id}"
        )));
    };
    for candidate in candidates.iter().copied().skip(1) {
        if !runtime_helicity_variants_share_embedding(input, representative, candidate)? {
            return Err(invalid(format!(
                "runtime-helicity contract gives crossed state {crossed_state_template_id} incompatible embeddings"
            )));
        }
    }
    Ok(representative)
}

fn runtime_helicity_variants_share_embedding(
    input: &OwnedRecurrenceTemplateInput,
    left: RuntimeHelicityVariantRow,
    right: RuntimeHelicityVariantRow,
) -> RusticolResult<bool> {
    let left_embeddings = left.embedding_range.as_usize_range(
        input.runtime_helicity_embeddings.len(),
        "left runtime source embeddings",
    )?;
    let right_embeddings = right.embedding_range.as_usize_range(
        input.runtime_helicity_embeddings.len(),
        "right runtime source embeddings",
    )?;
    let left_embeddings = &input.runtime_helicity_embeddings[left_embeddings];
    let right_embeddings = &input.runtime_helicity_embeddings[right_embeddings];
    if left_embeddings.len() != right_embeddings.len()
        || !left_embeddings
            .iter()
            .zip(right_embeddings)
            .all(|(left, right)| {
                left.full_component == right.full_component
                    && left.source_component == right.source_component
                    && left.factor_id == right.factor_id
            })
    {
        return Ok(false);
    }

    let left_projections = left.projection_range.as_usize_range(
        input.runtime_helicity_projections.len(),
        "left runtime source projections",
    )?;
    let right_projections = right.projection_range.as_usize_range(
        input.runtime_helicity_projections.len(),
        "right runtime source projections",
    )?;
    let left_projections = &input.runtime_helicity_projections[left_projections];
    let right_projections = &input.runtime_helicity_projections[right_projections];
    Ok(left_projections.len() == right_projections.len()
        && left_projections
            .iter()
            .zip(right_projections)
            .all(|(left, right)| {
                left.source_component == right.source_component
                    && left.full_component == right.full_component
            }))
}

fn validate_prepared_parameter_catalog(
    templates: &ValidatedRecurrenceTemplateInput,
) -> RusticolResult<()> {
    let input = templates.input();
    // Prepared IDs address the model-wide parameter arena. A process may
    // reference only a sparse subset of those slots, especially for UFO
    // models, so only bounded uniqueness is required here.
    let prepared_parameter_count = input.parameters.len();
    let mut seen = vec![false; prepared_parameter_count];
    for parameter in &input.parameters {
        if parameter.prepared_parameter_id == MISSING_U32 {
            continue;
        }
        let slot = usize::try_from(parameter.prepared_parameter_id)
            .map_err(|_| invalid("prepared parameter slot exceeds usize"))?;
        let present = seen.get_mut(slot).ok_or_else(|| {
            invalid(format!(
                "parameter template {} prepared runtime slot {} is out of bounds {prepared_parameter_count}",
                parameter.id, parameter.prepared_parameter_id
            ))
        })?;
        if std::mem::replace(present, true) {
            return Err(invalid(format!(
                "prepared runtime parameter slot {} is assigned more than once",
                parameter.prepared_parameter_id
            )));
        }
    }
    Ok(())
}

fn validate_direct_executor_catalog(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &PreparedDirectExecutorCatalog,
) -> RusticolResult<()> {
    let evaluators = &templates.input().evaluator_bindings;
    for key in catalog.bindings.keys().copied() {
        let PreparedDirectExecutorKey::Evaluator {
            role,
            evaluator_binding_id,
        } = key
        else {
            continue;
        };
        let evaluator = evaluators
            .get(evaluator_binding_id as usize)
            .ok_or_else(|| {
                invalid(format!(
                    "prepared direct-template mapping references absent evaluator binding {evaluator_binding_id}"
                ))
            })?;
        let expected_role = direct_role_for_contract(EvaluatorContractKind::try_from(
            evaluator.contract_kind,
        )?)?
        .ok_or_else(|| {
            invalid(format!(
                "model-parameter evaluator binding {evaluator_binding_id} cannot be a direct row executor"
            ))
        })?;
        if role != expected_role {
            return Err(invalid(format!(
                "prepared direct-template mapping for evaluator binding {evaluator_binding_id} has role {role:?}, expected {expected_role:?}"
            )));
        }
    }
    for evaluator in evaluators {
        let Some(role) =
            direct_role_for_contract(EvaluatorContractKind::try_from(evaluator.contract_kind)?)?
        else {
            continue;
        };
        catalog.resolve_evaluator(role, evaluator.id)?;
    }
    Ok(())
}

fn direct_role_for_contract(
    contract: EvaluatorContractKind,
) -> RusticolResult<Option<DirectExecutorRole>> {
    Ok(match contract {
        EvaluatorContractKind::Source => Some(DirectExecutorRole::Source),
        EvaluatorContractKind::Vertex => Some(DirectExecutorRole::Contribution),
        EvaluatorContractKind::Propagator => Some(DirectExecutorRole::Finalization),
        EvaluatorContractKind::Closure => Some(DirectExecutorRole::Closure),
        EvaluatorContractKind::ModelParameter => None,
    })
}

fn component_counts(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
) -> RusticolResult<Vec<u32>> {
    program
        .currents()
        .iter()
        .map(|current| {
            let dimension =
                state_row(templates, current.key().current_state_template_id())?.dimension;
            u16::try_from(dimension).map_err(|_| {
                invalid(format!(
                    "current {} state dimension {dimension} exceeds the direct-row u16 component domain",
                    current.id()
                ))
            })?;
            Ok(dimension)
        })
        .collect()
}

fn state_row(
    templates: &ValidatedRecurrenceTemplateInput,
    state_template_id: u32,
) -> RusticolResult<&super::template::CurrentStateRow> {
    templates
        .input()
        .current_states
        .get(state_template_id as usize)
        .ok_or_else(|| {
            invalid(format!(
                "state template {state_template_id} is absent from the validated catalog"
            ))
        })
}

fn validated_evaluator_binding_id(
    templates: &ValidatedRecurrenceTemplateInput,
    evaluator_binding_id: u32,
    expected_contract: EvaluatorContractKind,
    role: &str,
) -> RusticolResult<u32> {
    if evaluator_binding_id == MISSING_U32 {
        return Err(invalid(format!(
            "{role} has no prepared evaluator binding to use as a provisional direct executor"
        )));
    }
    let binding = templates
        .input()
        .evaluator_bindings
        .get(evaluator_binding_id as usize)
        .ok_or_else(|| {
            invalid(format!(
                "{role} references absent evaluator binding {evaluator_binding_id}"
            ))
        })?;
    if binding.id != evaluator_binding_id
        || EvaluatorContractKind::try_from(binding.contract_kind)? != expected_contract
    {
        return Err(invalid(format!(
            "{role} evaluator binding {evaluator_binding_id} has the wrong direct role"
        )));
    }

    Ok(evaluator_binding_id)
}

fn template_u32_sequence<'a>(
    templates: &'a ValidatedRecurrenceTemplateInput,
    sequence_id: u32,
    label: &str,
) -> RusticolResult<&'a [u32]> {
    let input = templates.input();
    let range = input
        .u32_sequence_ranges
        .get(sequence_id as usize)
        .ok_or_else(|| invalid(format!("{label} sequence {sequence_id} is absent")))?;
    if range.id != sequence_id {
        return Err(invalid(format!(
            "{label} sequence {sequence_id} has noncanonical ID {}",
            range.id
        )));
    }
    let range = range
        .range
        .as_usize_range(input.u32_sequence_values.len(), label)?;
    Ok(&input.u32_sequence_values[range])
}

fn validate_coupling_digest(
    templates: &ValidatedRecurrenceTemplateInput,
    contribution: &super::RecurrenceContribution,
) -> RusticolResult<()> {
    let flow_id = contribution.key().quantum_flow_witness_id();
    let flow = templates
        .input()
        .quantum_flows
        .get(flow_id as usize)
        .ok_or_else(|| {
            invalid(format!(
                "contribution {} references absent quantum-flow witness {flow_id}",
                contribution.id()
            ))
        })?;
    let digest = templates
        .input()
        .digest_catalog
        .get(flow.semantic_digest_id as usize)
        .ok_or_else(|| invalid("quantum-flow semantic digest is absent"))?;
    let digest = SemanticDigest::new(digest.value)?;
    if digest != contribution.key().runtime_coupling_binding_digest() {
        return Err(invalid(format!(
            "contribution {} runtime coupling digest does not match quantum flow {flow_id}",
            contribution.id()
        )));
    }
    Ok(())
}

fn parent_state_template_ids(
    program: &RecurrenceProgram,
    parent_ids: &[u32],
) -> RusticolResult<Vec<u32>> {
    parent_ids
        .iter()
        .map(|&parent_id| {
            program
                .currents()
                .get(parent_id as usize)
                .map(|parent| parent.key().current_state_template_id())
                .ok_or_else(|| invalid(format!("parent current {parent_id} is absent")))
        })
        .collect()
}

fn parent_state_contract_matches(
    concrete_states: &[u32],
    canonical_input_order: &[u32],
    exchange_is_proven: bool,
    actual_states: &[u32],
) -> RusticolResult<bool> {
    let [left, right] = concrete_states else {
        return Err(invalid("direct parent-state contract is not binary"));
    };
    let canonical = match canonical_input_order {
        [0, 1] => [*left, *right],
        [1, 0] => [*right, *left],
        _ => {
            return Err(invalid(
                "direct canonical input order is not a binary permutation",
            ));
        }
    };
    Ok(actual_states == canonical
        || (exchange_is_proven && actual_states == [canonical[1], canonical[0]]))
}

fn validate_parent_momentum_contract(
    program: &RecurrenceProgram,
    contribution: &super::RecurrenceContribution,
) -> RusticolResult<()> {
    if contribution.key().parent_momenta().len() != contribution.parent_current_ids().len() {
        return Err(invalid(format!(
            "contribution {} parent-momentum arity is inconsistent",
            contribution.id()
        )));
    }
    for (ordinal, (&parent_id, expected)) in contribution
        .parent_current_ids()
        .iter()
        .zip(contribution.key().parent_momenta())
        .enumerate()
    {
        let parent = program
            .currents()
            .get(parent_id as usize)
            .ok_or_else(|| invalid(format!("parent current {parent_id} is absent")))?;
        if parent.key().momentum() != expected {
            return Err(invalid(format!(
                "contribution {} parent {ordinal} momentum does not match current {parent_id}",
                contribution.id()
            )));
        }
    }
    Ok(())
}

fn validate_direct_arity(label: &str, id: u32, arity: usize) -> RusticolResult<()> {
    if !(1..=2).contains(&arity) {
        return Err(invalid(format!(
            "{label} {id} has unsupported parent arity {arity}; direct plan-v2 supports only one or two parents"
        )));
    }
    Ok(())
}

fn direct_parents(
    program: &RecurrenceProgram,
    parent_ids: &[u32],
    arena: &DirectArenaLayout,
    momentum_ids: &BTreeMap<CanonicalMomentumLinearForm, u32>,
) -> RusticolResult<DirectParents> {
    validate_direct_arity("direct row", 0, parent_ids.len())?;
    let parent0 = program
        .currents()
        .get(parent_ids[0] as usize)
        .ok_or_else(|| invalid("direct parent 0 current is absent"))?;
    let parent0_assignment = arena_assignment(arena, parent0.id())?;
    let (parent1_component_base_or_sentinel, parent1_momentum_form_id_or_sentinel) =
        if let Some(&parent1_id) = parent_ids.get(1) {
            let parent1 = program
                .currents()
                .get(parent1_id as usize)
                .ok_or_else(|| invalid("direct parent 1 current is absent"))?;
            (
                arena_assignment(arena, parent1.id())?.component_base,
                momentum_id(momentum_ids, parent1.key().momentum())?,
            )
        } else {
            (DIRECT_NONE_U32, DIRECT_NONE_U32)
        };
    Ok(DirectParents {
        parent0_component_base: parent0_assignment.component_base,
        parent1_component_base_or_sentinel,
        parent0_momentum_form_id: momentum_id(momentum_ids, parent0.key().momentum())?,
        parent1_momentum_form_id_or_sentinel,
    })
}

fn external_source_count(program: &RecurrenceProgram) -> RusticolResult<u32> {
    let slots = program
        .currents()
        .iter()
        .filter(|current| current.is_source())
        .map(only_source_slot)
        .collect::<RusticolResult<BTreeSet<_>>>()?;
    let count = slots
        .iter()
        .next_back()
        .copied()
        .and_then(|slot| slot.checked_add(1))
        .ok_or_else(|| invalid("semantic program contains no source currents"))?;
    if slots.iter().copied().ne(0..count) {
        return Err(invalid(
            "source currents do not cover a contiguous external-source domain",
        ));
    }
    Ok(count)
}

fn only_source_slot(current: &RecurrenceCurrent) -> RusticolResult<u32> {
    let [source_slot] = current.key().support_source_slots() else {
        return Err(invalid(format!(
            "source current {} does not cover exactly one source slot",
            current.id()
        )));
    };
    Ok(*source_slot)
}

fn current_stage(current: &RecurrenceCurrent) -> RusticolResult<u16> {
    let stage = current
        .key()
        .support_source_slots()
        .len()
        .checked_sub(1)
        .ok_or_else(|| invalid(format!("current {} has empty source support", current.id())))?;
    u16::try_from(stage).map_err(|_| invalid(format!("current {} stage exceeds u16", current.id())))
}

fn direct_event(event: u64, label: &str) -> RusticolResult<u32> {
    let event = event
        .checked_sub(1)
        .ok_or_else(|| invalid(format!("{label} precedes direct stage zero")))?;
    u32::try_from(event).map_err(|_| invalid(format!("{label} exceeds u32")))
}

fn arena_assignment(
    arena: &DirectArenaLayout,
    current_id: u32,
) -> RusticolResult<super::DirectArenaAssignment> {
    arena.assignment(current_id).ok_or_else(|| {
        invalid(format!(
            "arena assignment for semantic current {current_id} is absent"
        ))
    })
}

fn intern_momentum_forms(
    program: &RecurrenceProgram,
    external_source_count: u32,
) -> RusticolResult<(
    BTreeMap<CanonicalMomentumLinearForm, u32>,
    Vec<DirectMomentumFormDescriptor>,
    Vec<DirectMomentumTerm>,
)> {
    let forms = program
        .currents()
        .iter()
        .map(|current| current.key().momentum().clone())
        .collect::<BTreeSet<_>>();
    let mut ids = BTreeMap::new();
    let mut descriptors = Vec::with_capacity(forms.len());
    let mut terms = Vec::new();
    for form in forms {
        for term in form.terms() {
            if term.source_slot >= external_source_count {
                return Err(invalid(format!(
                    "momentum form references source slot {} outside the source domain",
                    term.source_slot
                )));
            }
        }
        let id = u32_len("momentum form", descriptors.len())?;
        let term_start =
            u64::try_from(terms.len()).map_err(|_| invalid("momentum term table exceeds u64"))?;
        terms.extend(form.terms().iter().map(|term| DirectMomentumTerm {
            source_slot: term.source_slot,
            coefficient: term.coefficient,
        }));
        descriptors.push(DirectMomentumFormDescriptor {
            term_start,
            term_count: u32_len("momentum term", form.terms().len())?,
        });
        ids.insert(form, id);
    }
    Ok((ids, descriptors, terms))
}

fn momentum_id(
    ids: &BTreeMap<CanonicalMomentumLinearForm, u32>,
    form: &CanonicalMomentumLinearForm,
) -> RusticolResult<u32> {
    ids.get(form)
        .copied()
        .ok_or_else(|| invalid("semantic momentum form was not interned"))
}

fn intern_exact_factors(
    program: &RecurrenceProgram,
) -> RusticolResult<(
    Vec<ExactComplexRational>,
    BTreeMap<ExactComplexRational, u32>,
)> {
    let mut unique = BTreeSet::from([ExactComplexRational::ONE]);
    unique.extend(
        program
            .currents()
            .iter()
            .filter_map(RecurrenceCurrent::source_exact_factor),
    );
    unique.extend(
        program
            .contributions()
            .iter()
            .map(super::RecurrenceContribution::exact_factor),
    );
    unique.extend(
        program
            .finalizations()
            .iter()
            .map(super::RecurrenceFinalization::exact_factor),
    );
    unique.extend(
        program
            .closure_terms()
            .iter()
            .map(super::RecurrenceClosureTerm::exact_factor),
    );
    unique.extend(
        program
            .replay_targets()
            .iter()
            .map(super::RecurrenceReplayTarget::amplitude_factor),
    );

    let mut factors = vec![ExactComplexRational::ONE];
    factors.extend(
        unique
            .into_iter()
            .filter(|factor| *factor != ExactComplexRational::ONE),
    );
    let ids = factors
        .iter()
        .copied()
        .enumerate()
        .map(|(id, factor)| Ok((factor, u32_len("exact factor", id)?)))
        .collect::<RusticolResult<BTreeMap<_, _>>>()?;
    Ok((factors, ids))
}

fn exact_factor_id(
    ids: &BTreeMap<ExactComplexRational, u32>,
    factor: ExactComplexRational,
) -> RusticolResult<u32> {
    ids.get(&factor)
        .copied()
        .ok_or_else(|| invalid("semantic exact factor was not interned"))
}

fn intern_direct_factor(
    factor: ExactComplexRational,
    factors: &mut Vec<ExactComplexRational>,
    ids: &mut BTreeMap<ExactComplexRational, u32>,
) -> RusticolResult<u32> {
    if let Some(id) = ids.get(&factor).copied() {
        return Ok(id);
    }
    let id = u32_len("exact factor", factors.len())?;
    factors.push(factor);
    ids.insert(factor, id);
    Ok(id)
}

fn template_exact_factor(
    templates: &ValidatedRecurrenceTemplateInput,
    factor_id: u32,
) -> RusticolResult<ExactComplexRational> {
    let input = templates.input();
    let row = input
        .exact_factors
        .get(factor_id as usize)
        .ok_or_else(|| invalid(format!("template exact factor {factor_id} is absent")))?;
    if row.id != factor_id {
        return Err(invalid(format!(
            "template exact factor {factor_id} has noncanonical ID {}",
            row.id
        )));
    }
    ExactComplexRational::parse_parts(
        template_string(input, row.real_numerator_string_id, "exact real numerator")?,
        template_string(
            input,
            row.real_denominator_string_id,
            "exact real denominator",
        )?,
        template_string(
            input,
            row.imag_numerator_string_id,
            "exact imaginary numerator",
        )?,
        template_string(
            input,
            row.imag_denominator_string_id,
            "exact imaginary denominator",
        )?,
    )
    .map_err(|error| {
        invalid(format!(
            "template exact factor {factor_id} is invalid: {error}"
        ))
    })
}

fn template_string<'a>(
    input: &'a super::template::OwnedRecurrenceTemplateInput,
    string_id: u32,
    label: &str,
) -> RusticolResult<&'a str> {
    let range = input
        .string_ranges
        .get(string_id as usize)
        .ok_or_else(|| invalid(format!("{label} string {string_id} is absent")))?;
    let bytes = &input.string_bytes[range.as_usize_range(input.string_bytes.len(), label)?];
    std::str::from_utf8(bytes)
        .map_err(|error| invalid(format!("{label} string {string_id} is not UTF-8: {error}")))
}

fn intern_closure_factor_block(
    coefficients: &[ExactComplexRational],
    factors: &mut Vec<ExactComplexRational>,
    blocks: &mut BTreeMap<Vec<ExactComplexRational>, (u32, u16)>,
) -> RusticolResult<(u32, u16)> {
    if let Some(block) = blocks.get(coefficients) {
        return Ok(*block);
    }
    let start = u32_len("closure component-factor start", factors.len())?;
    let count = u16::try_from(coefficients.len())
        .map_err(|_| invalid("closure component count exceeds u16"))?;
    if count == 0 {
        return Err(invalid("closure component-factor block is empty"));
    }
    factors.extend_from_slice(coefficients);
    let block = (start, count);
    blocks.insert(coefficients.to_vec(), block);
    Ok(block)
}

fn lower_amplitude_destinations(
    program: &RecurrenceProgram,
    closures: &[ClosureDraft],
) -> RusticolResult<Vec<DirectAmplitudeDestinationDescriptor>> {
    let mut next_closure = 0usize;
    let mut result = Vec::with_capacity(program.amplitude_destinations().len());
    for destination in program.amplitude_destinations().iter().copied() {
        let start = next_closure;
        while next_closure < closures.len()
            && closures[next_closure].amplitude_destination_id == destination.id()
        {
            next_closure += 1;
        }
        let count = next_closure - start;
        if count as u64 != destination.closure_range().count {
            return Err(invalid(format!(
                "amplitude destination {} lowered {count} closure rows, expected {}",
                destination.id(),
                destination.closure_range().count
            )));
        }
        result.push(DirectAmplitudeDestinationDescriptor {
            closure_row_start: u64::try_from(start)
                .map_err(|_| invalid("closure row start exceeds u64"))?,
            id: destination.id(),
            target_sector_id: destination.target_sector_id(),
            target_helicity_id_or_sentinel: destination
                .target_helicity_id()
                .unwrap_or(DIRECT_NONE_U32),
            closure_row_count: u32_len("amplitude-destination closure row", count)?,
            selector_domain_id: UNIVERSAL_SELECTOR_DOMAIN_ID,
        });
    }
    if next_closure != closures.len() {
        return Err(invalid(
            "closure rows are not fully owned by amplitude destinations",
        ));
    }
    Ok(result)
}

fn lower_resolved_helicities(
    program: &RecurrenceProgram,
) -> RusticolResult<(
    Vec<DirectResolvedHelicityDescriptor>,
    Vec<DirectSourceStateAssignment>,
    Vec<i32>,
)> {
    let mut descriptors = Vec::with_capacity(program.resolved_helicities().len());
    let mut source_states = Vec::new();
    let mut public_helicities = Vec::new();
    for helicity in program.resolved_helicities() {
        let source_state_start = u64::try_from(source_states.len())
            .map_err(|_| invalid("resolved-helicity source states exceed u64"))?;
        let public_helicity_start = u64::try_from(public_helicities.len())
            .map_err(|_| invalid("resolved-helicity public values exceed u64"))?;
        source_states.extend(helicity.source_states().iter().map(|assignment| {
            DirectSourceStateAssignment {
                source_slot: assignment.source_slot(),
                state_index: assignment.state_index(),
            }
        }));
        public_helicities.extend_from_slice(helicity.public_helicities());
        descriptors.push(DirectResolvedHelicityDescriptor {
            source_state_start,
            source_selection_start: 0,
            public_helicity_start,
            id: helicity.id(),
            source_state_count: u32_len(
                "resolved-helicity source state",
                helicity.source_states().len(),
            )?,
            source_selection_count: 0,
            public_helicity_count: u32_len(
                "resolved-helicity public value",
                helicity.public_helicities().len(),
            )?,
            selector_domain_id: UNIVERSAL_SELECTOR_DOMAIN_ID,
        });
    }
    Ok((descriptors, source_states, public_helicities))
}

pub(super) fn lower_union_resolved_helicities(
    choices_by_source: &[Vec<RuntimeSourceChoice>],
    retained_helicity_count: u64,
) -> RusticolResult<(
    Vec<DirectResolvedHelicityDescriptor>,
    Vec<DirectSourceStateAssignment>,
    Vec<DirectResolvedSourceSelection>,
    Vec<i32>,
)> {
    if choices_by_source.is_empty() || choices_by_source.iter().any(Vec::is_empty) {
        return Err(invalid(
            "all-flow-union runtime source choices do not cover every external source",
        ));
    }
    let cartesian_count = choices_by_source.iter().try_fold(1_u64, |count, choices| {
        count
            .checked_mul(
                u64::try_from(choices.len())
                    .map_err(|_| invalid("runtime source-choice count exceeds u64"))?,
            )
            .ok_or_else(|| invalid("resolved-helicity Cartesian product exceeds u64"))
    })?;
    if cartesian_count != retained_helicity_count {
        return Err(invalid(format!(
            "all-flow-union runtime source choices span {cartesian_count} helicities, expected {retained_helicity_count}"
        )));
    }
    let descriptor_capacity = usize::try_from(cartesian_count)
        .map_err(|_| invalid("resolved-helicity count exceeds usize"))?;
    let mut descriptors = Vec::with_capacity(descriptor_capacity);
    let mut source_states = Vec::with_capacity(
        descriptor_capacity
            .checked_mul(choices_by_source.len())
            .ok_or_else(|| invalid("resolved source-state capacity exceeds usize"))?,
    );
    let mut source_selections = Vec::with_capacity(source_states.capacity());
    let mut public_helicities = Vec::with_capacity(source_states.capacity());
    let mut indices = vec![0usize; choices_by_source.len()];

    for id in 0..descriptor_capacity {
        let source_state_start = u64::try_from(source_states.len())
            .map_err(|_| invalid("resolved source states exceed u64"))?;
        let source_selection_start = u64::try_from(source_selections.len())
            .map_err(|_| invalid("resolved source selections exceed u64"))?;
        let public_helicity_start = u64::try_from(public_helicities.len())
            .map_err(|_| invalid("resolved public helicities exceed u64"))?;
        for (source_slot, (&choice_index, choices)) in
            indices.iter().zip(choices_by_source).enumerate()
        {
            let choice = choices[choice_index];
            let source_slot = u32_len("resolved source slot", source_slot)?;
            source_states.push(DirectSourceStateAssignment {
                source_slot,
                state_index: choice.source_state_index,
            });
            source_selections.push(DirectResolvedSourceSelection {
                source_slot,
                dispatch_variant_id: choice.dispatch_variant_id,
            });
            public_helicities.push(choice.public_helicity);
        }
        let source_count = u32_len("resolved source", choices_by_source.len())?;
        descriptors.push(DirectResolvedHelicityDescriptor {
            source_state_start,
            source_selection_start,
            public_helicity_start,
            id: u32_len("resolved helicity", id)?,
            source_state_count: source_count,
            source_selection_count: source_count,
            public_helicity_count: source_count,
            selector_domain_id: UNIVERSAL_SELECTOR_DOMAIN_ID,
        });

        for source_slot in (0..indices.len()).rev() {
            indices[source_slot] += 1;
            if indices[source_slot] < choices_by_source[source_slot].len() {
                break;
            }
            indices[source_slot] = 0;
        }
    }
    Ok((
        descriptors,
        source_states,
        source_selections,
        public_helicities,
    ))
}

fn append_row_groups(
    keys: impl IntoIterator<Item = (u16, u32)>,
    role: DirectExecutorRole,
    destination_operation: DirectDestinationOperation,
    row_groups: &mut Vec<DirectRowGroupDescriptor>,
) -> RusticolResult<()> {
    let keys = keys.into_iter().collect::<Vec<_>>();
    let mut start = 0usize;
    while start < keys.len() {
        let (stage, direct_executor_id) = keys[start];
        let mut stop = start + 1;
        while stop < keys.len() && keys[stop] == (stage, direct_executor_id) {
            stop += 1;
        }
        row_groups.push(DirectRowGroupDescriptor {
            stage,
            role,
            destination_operation,
            direct_executor_id,
            row_start: u64::try_from(start)
                .map_err(|_| invalid("direct row_group row start exceeds u64"))?,
            row_count: u32_len("direct row_group row", stop - start)?,
        });
        start = stop;
    }
    Ok(())
}

fn u32_len(label: &str, value: usize) -> RusticolResult<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("{label} count exceeds u32")))
}

#[cfg(test)]
#[path = "direct_lowering_tests.rs"]
mod tests;
