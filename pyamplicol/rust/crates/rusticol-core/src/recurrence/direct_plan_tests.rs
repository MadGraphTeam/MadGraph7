// SPDX-License-Identifier: 0BSD

use super::*;
use crate::recurrence::direct_plan::{
    DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION, DirectResolvedSourceSelection,
    DirectSourceDispatchVariantDescriptor, DirectSourceEmbeddingRow, DirectSourceProjectionRow,
};

pub(crate) fn valid_parts() -> DirectRecurrencePlanParts {
    DirectRecurrencePlanParts {
        strategy: RecurrenceStrategy::TopologyReplay,
        semantic_digest: SemanticDigest::new([0x11; 32]).unwrap(),
        prepared_pack_digest: SemanticDigest::new([0x22; 32]).unwrap(),
        direct_template_catalog_digest: SemanticDigest::new([0x33; 32]).unwrap(),
        point_tile_size: 1024,
        workspace_mib: 256,
        current_arena_components: 3,
        physical_sector_count: 1,
        retained_helicity_count: 1,
        amplitude_destination_count: 1,
        parameter_value_count: 1,
        external_source_count: 1,
        state_template_count: 2,
        source_template_count: 8,
        source_template_or_dispatch_count: 8,
        runtime_helicity_contract_count: 0,
        runtime_helicity_variant_count: 0,
        direct_executor_count: 4,
        currents: vec![
            DirectCurrentDescriptor {
                semantic_current_id: 0,
                node_kind: DirectNodeKind::Source,
                state_template_id: 0,
                component_base: 0,
                component_count: 2,
                momentum_form_id: 0,
                stage: 0,
                selector_domain_id: 0,
                first_use: 0,
                last_use: 1,
                source_row_or_sentinel: 0,
                finalization_row_or_sentinel: DIRECT_NONE_U32,
            },
            DirectCurrentDescriptor {
                semantic_current_id: 1,
                node_kind: DirectNodeKind::Current,
                state_template_id: 1,
                component_base: 2,
                component_count: 1,
                momentum_form_id: 0,
                stage: 1,
                selector_domain_id: 0,
                first_use: 1,
                last_use: 2,
                source_row_or_sentinel: DIRECT_NONE_U32,
                finalization_row_or_sentinel: 0,
            },
        ],
        sources: vec![DirectSourceRow {
            source_slot: 0,
            destination_component_base: 0,
            momentum_form_id: 0,
            source_template_or_dispatch_domain: 7,
            spin_state_class: -1,
            exact_factor_id: 0,
            selector_domain_id: 0,
        }],
        contributions: vec![DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: DIRECT_NONE_U32,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
            destination_component_base: 2,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
        }],
        finalizations: vec![DirectFinalizationRow {
            component_base: 2,
            component_count: 1,
            momentum_form_id: 0,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        }],
        closures: vec![DirectClosureRow {
            parent0_component_base: 2,
            parent1_component_base_or_sentinel: DIRECT_NONE_U32,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
            amplitude_destination_id: 0,
            exact_factor_id: 0,
            component_factor_start: 0,
            component_count: 1,
            selector_domain_id: 0,
            flags: 0,
        }],
        row_groups: vec![
            DirectRowGroupDescriptor {
                stage: 0,
                role: DirectExecutorRole::Source,
                destination_operation: DirectDestinationOperation::Initialize,
                direct_executor_id: 0,
                row_start: 0,
                row_count: 1,
            },
            DirectRowGroupDescriptor {
                stage: 1,
                role: DirectExecutorRole::Contribution,
                destination_operation: DirectDestinationOperation::Add,
                direct_executor_id: 1,
                row_start: 0,
                row_count: 1,
            },
            DirectRowGroupDescriptor {
                stage: 1,
                role: DirectExecutorRole::Finalization,
                destination_operation: DirectDestinationOperation::FinalizeInPlace,
                direct_executor_id: 2,
                row_start: 0,
                row_count: 1,
            },
            DirectRowGroupDescriptor {
                stage: 2,
                role: DirectExecutorRole::Closure,
                destination_operation: DirectDestinationOperation::ClosureAdd,
                direct_executor_id: 3,
                row_start: 0,
                row_count: 1,
            },
        ],
        momentum_forms: vec![DirectMomentumFormDescriptor {
            term_start: 0,
            term_count: 1,
        }],
        momentum_terms: vec![DirectMomentumTerm {
            source_slot: 0,
            coefficient: 1,
        }],
        selector_domains: vec![DirectSelectorDomainDescriptor {
            word_start: 0,
            word_count: 1,
        }],
        selector_words: vec![1],
        replay_targets: vec![DirectReplayTargetDescriptor {
            public_flow_id: 0,
            representative_id: 0,
            source_permutation_start: 0,
            source_permutation_count: 1,
            phase_exact_factor_id: 0,
            multiplicity: 1,
            selector_domain_id: 0,
        }],
        source_permutations: vec![0],
        amplitude_destinations: vec![DirectAmplitudeDestinationDescriptor {
            closure_row_start: 0,
            id: 0,
            target_sector_id: 0,
            target_helicity_id_or_sentinel: 0,
            closure_row_count: 1,
            selector_domain_id: 0,
        }],
        resolved_helicities: vec![DirectResolvedHelicityDescriptor {
            source_state_start: 0,
            source_selection_start: 0,
            public_helicity_start: 0,
            id: 0,
            source_state_count: 1,
            source_selection_count: 0,
            public_helicity_count: 1,
            selector_domain_id: 0,
        }],
        source_state_assignments: vec![DirectSourceStateAssignment {
            source_slot: 0,
            state_index: 0,
        }],
        source_dispatch_variants: vec![],
        source_embeddings: vec![],
        source_projections: vec![],
        resolved_source_selections: vec![],
        public_helicities: vec![-1],
        exact_factors: vec![ExactComplexRational::ONE],
    }
}

pub(crate) fn valid_plan() -> DirectRecurrencePlan {
    DirectRecurrencePlan::new(valid_parts()).unwrap()
}

#[test]
fn direct_descriptor_native_layouts_are_fixed_width() {
    assert_eq!(std::mem::size_of::<DirectCurrentDescriptor>(), 48);
    assert_eq!(std::mem::size_of::<DirectSourceRow>(), 28);
    assert_eq!(std::mem::size_of::<DirectContributionRow>(), 32);
    assert_eq!(std::mem::size_of::<DirectFinalizationRow>(), 24);
    assert_eq!(std::mem::size_of::<DirectClosureRow>(), 40);
    assert_eq!(std::mem::size_of::<DirectRowGroupDescriptor>(), 32);
    assert_eq!(std::mem::size_of::<DirectMomentumFormDescriptor>(), 16);
    assert_eq!(std::mem::size_of::<DirectMomentumTerm>(), 8);
    assert_eq!(std::mem::size_of::<DirectSelectorDomainDescriptor>(), 16);
    assert_eq!(std::mem::size_of::<DirectReplayTargetDescriptor>(), 32);
    assert_eq!(
        std::mem::size_of::<DirectAmplitudeDestinationDescriptor>(),
        32
    );
    assert_eq!(std::mem::size_of::<DirectResolvedHelicityDescriptor>(), 48);
    assert_eq!(std::mem::size_of::<DirectSourceStateAssignment>(), 8);
    assert_eq!(
        std::mem::size_of::<DirectSourceDispatchVariantDescriptor>(),
        64
    );
    assert_eq!(std::mem::size_of::<DirectSourceEmbeddingRow>(), 12);
    assert_eq!(std::mem::size_of::<DirectSourceProjectionRow>(), 8);
    assert_eq!(std::mem::size_of::<DirectResolvedSourceSelection>(), 8);
}

#[test]
fn direct_plan_validates_and_has_a_stable_nonzero_layout_digest() {
    let first = valid_plan();
    let second = valid_plan();
    assert_eq!(first, second);
    assert_eq!(
        first.runtime_layout_digest(),
        second.runtime_layout_digest()
    );
    assert_ne!(first.runtime_layout_digest().as_bytes(), &[0; 32]);
}

#[test]
fn direct_plan_accepts_an_elided_identity_finalization() {
    let mut parts = valid_parts();
    parts.currents[1].finalization_row_or_sentinel = DIRECT_NONE_U32;
    parts.finalizations.clear();
    parts
        .row_groups
        .retain(|group| group.role != DirectExecutorRole::Finalization);
    DirectRecurrencePlan::new(parts).unwrap();
}

#[test]
fn direct_plan_canonicalizes_a_missing_initialization_marker() {
    let mut parts = valid_parts();
    parts.contributions[0].flags = 0;
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    assert_eq!(
        plan.contributions()[0].flags,
        DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION
    );
}

#[test]
fn direct_plan_rejects_multiple_initialization_markers_for_one_current() {
    let mut parts = valid_parts();
    parts.contributions.push(parts.contributions[0]);
    parts.row_groups[1].row_count = 2;
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("exactly one"));
}

#[test]
fn direct_plan_rejects_invalid_arena_and_optional_parent_references() {
    let mut parts = valid_parts();
    parts.contributions[0].destination_component_base = 99;
    assert!(
        DirectRecurrencePlan::new(parts)
            .unwrap_err()
            .to_string()
            .contains("arena base")
    );

    let mut parts = valid_parts();
    parts.contributions[0].parent1_component_base_or_sentinel = 0;
    assert!(
        DirectRecurrencePlan::new(parts)
            .unwrap_err()
            .to_string()
            .contains("mismatched optional")
    );
}

#[test]
fn direct_plan_rejects_overlapping_live_arena_ranges() {
    let mut parts = valid_parts();
    parts.currents[1].component_base = 1;
    parts.contributions[0].destination_component_base = 1;
    parts.finalizations[0].component_base = 1;
    parts.closures[0].parent0_component_base = 1;
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("shared by live currents"));
}

#[test]
fn direct_plan_rejects_row_group_gaps_and_role_operation_mismatch() {
    let mut parts = valid_parts();
    parts.row_groups[1].row_start = 1;
    assert!(
        DirectRecurrencePlan::new(parts)
            .unwrap_err()
            .to_string()
            .contains("row partition")
    );

    let mut parts = valid_parts();
    parts.row_groups[1].destination_operation = DirectDestinationOperation::Initialize;
    assert!(
        DirectRecurrencePlan::new(parts)
            .unwrap_err()
            .to_string()
            .contains("incompatible")
    );
}

#[test]
fn direct_plan_rejects_non_permutation_replay_mappings() {
    let mut parts = valid_parts();
    parts.external_source_count = 2;
    parts.replay_targets[0].source_permutation_count = 2;
    parts.source_permutations = vec![0, 0];
    let error = DirectRecurrencePlan::new(parts).unwrap_err();
    assert!(error.to_string().contains("not a permutation"));
}

#[test]
fn direct_plan_authenticates_resolved_helicity_and_destination_contracts() {
    let mut parts = valid_parts();
    parts.amplitude_destinations[0].target_sector_id = 1;
    assert!(
        DirectRecurrencePlan::new(parts)
            .unwrap_err()
            .to_string()
            .contains("physical sector")
    );

    let mut parts = valid_parts();
    parts.resolved_helicities[0].source_state_count = 0;
    assert!(
        DirectRecurrencePlan::new(parts)
            .unwrap_err()
            .to_string()
            .contains("source state")
    );

    let mut parts = valid_parts();
    parts.amplitude_destinations[0].target_helicity_id_or_sentinel = 1;
    assert!(
        DirectRecurrencePlan::new(parts)
            .unwrap_err()
            .to_string()
            .contains("resolved helicity")
    );
}

#[test]
fn changing_physical_layout_changes_only_the_layout_digest() {
    let first = valid_plan();
    let mut parts = valid_parts();
    parts.point_tile_size = 128;
    let second = DirectRecurrencePlan::new(parts).unwrap();
    assert_eq!(first.semantic_digest(), second.semantic_digest());
    assert_ne!(
        first.runtime_layout_digest(),
        second.runtime_layout_digest()
    );
}
