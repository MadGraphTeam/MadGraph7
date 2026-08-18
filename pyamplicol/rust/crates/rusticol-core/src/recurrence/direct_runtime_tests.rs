// SPDX-License-Identifier: 0BSD

use crate::recurrence::direct_backend::{
    DIRECT_STATUS_OK, DirectArenaView, DirectContributionExecutionMetadata,
    DirectContributionExecutor, DirectContributionFanoutBatch,
    DirectContributionFanoutExecutorHandle, DirectContributionFanoutProgram,
    DirectExecutionCounters, DirectExecutorCatalog, DirectExecutorHandle, DirectFactorView,
    DirectFinalizationExecutor, DirectInteractionBundleBatch, DirectInteractionExecutorContexts,
    DirectInteractionIntrinsicKind, DirectInteractionOperands, DirectInteractionVectorParent,
    DirectMomentumView, DirectParameterView, DirectSourceExecutor, DirectWorkspace,
    begin_direct_current_observation, execute_direct_plan,
    execute_direct_plan_unprofiled_with_fanout, take_direct_current_observation,
};
use crate::recurrence::direct_plan::{
    DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION, DirectClosureRow, DirectContributionRow,
    DirectFinalizationRow, DirectMomentumTerm, DirectRecurrencePlan, DirectReplayTargetDescriptor,
    DirectSourceRow, DirectSourceStateAssignment,
};
use crate::recurrence::direct_runtime::{
    DIRECT_RUNTIME_ARENA_ALIGNMENT, DirectRecurrenceExecutionRuntime,
    DirectRuntimeActivityCounters, replay_cache_split_complex_scalar_count,
};
use crate::recurrence::exact::{ExactComplexRational, ExactRational};
#[allow(unused_imports)]
use crate::recurrence::{
    ClosureExecutionProofGroupV2, ClosureProofContributionV2, ClosureProofMetadataV2,
    DIRECT_NONE_U32, DirectAmplitudeDestinationDescriptor, DirectCurrentDescriptor,
    DirectDestinationOperation, DirectExecutorRole, DirectMomentumFormDescriptor, DirectNodeKind,
    DirectResolvedHelicityDescriptor, DirectRowGroupDescriptor, DirectSelectorDomainDescriptor,
    RecurrenceStrategy, SemanticDigest, closure_component_factor_digest_v2,
};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};

const STATUS_BOUNDS: i32 = 2;

fn direct_executor_handles() -> Vec<DirectExecutorHandle> {
    vec![
        DirectExecutorHandle::Source {
            call: fill_sources as DirectSourceExecutor,
            context: std::ptr::null(),
        },
        DirectExecutorHandle::Contribution {
            call: accumulate_contributions as DirectContributionExecutor,
            context: std::ptr::null(),
        },
        DirectExecutorHandle::Finalization {
            call: finalize_currents as DirectFinalizationExecutor,
            context: std::ptr::null(),
        },
        DirectExecutorHandle::Closure {
            call: accumulate_closures,
            context: std::ptr::null(),
        },
    ]
}

unsafe extern "C" fn fill_sources(
    _context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    _parameters: DirectParameterView,
    _factors: DirectFactorView,
    rows: *const DirectSourceRow,
    row_count: u32,
    point_count: u32,
) -> i32 {
    let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
    for row in rows {
        for point in 0..point_count as usize {
            let source = row.momentum_form_id as usize
                * momenta.lorentz_component_count as usize
                * momenta.point_stride as usize
                + point;
            let destination =
                row.destination_component_base as usize * arena.point_stride as usize + point;
            if source >= momenta.scalar_len as usize
                || destination >= arena.current_scalar_len as usize
            {
                return STATUS_BOUNDS;
            }
            unsafe {
                *arena.current_re.add(destination) = *momenta.values.add(source);
                *arena.current_im.add(destination) = 0.0;
            }
        }
    }
    DIRECT_STATUS_OK
}

unsafe extern "C" fn accumulate_contributions(
    _context: *const c_void,
    arena: DirectArenaView,
    _momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectContributionRow,
    row_count: u32,
    point_count: u32,
) -> i32 {
    let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
    for row in rows {
        if row.exact_factor_id >= factors.value_count || parameters.value_count == 0 {
            return STATUS_BOUNDS;
        }
        let factor_re = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
        let factor_im = unsafe { *factors.values_im.add(row.exact_factor_id as usize) };
        let parameter_re = unsafe { *parameters.values_re };
        let parameter_im = unsafe { *parameters.values_im };
        let scale_re = factor_re * parameter_re - factor_im * parameter_im;
        let scale_im = factor_re * parameter_im + factor_im * parameter_re;
        for point in 0..point_count as usize {
            let source = row.parent0_component_base as usize * arena.point_stride as usize + point;
            let destination =
                row.destination_component_base as usize * arena.point_stride as usize + point;
            if source >= arena.current_scalar_len as usize
                || destination >= arena.current_scalar_len as usize
            {
                return STATUS_BOUNDS;
            }
            let source_re = unsafe { *arena.current_re.add(source) };
            let source_im = unsafe { *arena.current_im.add(source) };
            let value_re = source_re * scale_re - source_im * scale_im;
            let value_im = source_re * scale_im + source_im * scale_re;
            unsafe {
                if row.flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION != 0 {
                    *arena.current_re.add(destination) = value_re;
                    *arena.current_im.add(destination) = value_im;
                } else {
                    *arena.current_re.add(destination) += value_re;
                    *arena.current_im.add(destination) += value_im;
                }
            }
        }
    }
    DIRECT_STATUS_OK
}

#[derive(Default)]
struct ContributionCallCensus {
    calls: AtomicU64,
    rows: AtomicU64,
}

#[derive(Default)]
struct InteractionDispatchCensus {
    calls: AtomicU64,
}

unsafe extern "C" fn counted_accumulate_contributions(
    context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectContributionRow,
    row_count: u32,
    point_count: u32,
) -> i32 {
    if context.is_null() {
        return STATUS_BOUNDS;
    }
    let census = unsafe { &*context.cast::<ContributionCallCensus>() };
    census.calls.fetch_add(1, Ordering::Relaxed);
    census
        .rows
        .fetch_add(u64::from(row_count), Ordering::Relaxed);
    unsafe {
        accumulate_contributions(
            std::ptr::null(),
            arena,
            momenta,
            parameters,
            factors,
            rows,
            row_count,
            point_count,
        )
    }
}

unsafe fn unexpected_single_point_fanout(
    _context: *const c_void,
    _arena: DirectArenaView,
    _momenta: DirectMomentumView,
    _parameters: DirectParameterView,
    _factors: DirectFactorView,
    _batch: DirectContributionFanoutBatch<'_>,
) -> crate::RusticolResult<()> {
    panic!("single-point contribution fanout bypassed the interaction action")
}

unsafe fn inspect_nonidentity_interaction_dispatch(
    contexts: DirectInteractionExecutorContexts,
    _arena: DirectArenaView,
    _momenta: DirectMomentumView,
    _parameters: DirectParameterView,
    _factors: DirectFactorView,
    batch: DirectInteractionBundleBatch<'_>,
) -> crate::RusticolResult<()> {
    assert!(!contexts.color.is_null());
    assert_eq!(contexts.color, contexts.antisymmetric_tensor_vector);
    assert_eq!(batch.anchors.len(), 1);
    assert_eq!(batch.outputs.len(), 2);
    assert!(batch.atv_before_color);

    let anchor = batch.anchors[0];
    assert_eq!(anchor.output_start, 0);
    assert_eq!(anchor.output_end, 2);
    assert!(matches!(
        anchor.operands,
        DirectInteractionOperands::One(operand)
            if operand.tensor_component_base == 1
                && operand.vector_parent == DirectInteractionVectorParent::Parent0
    ));
    assert_eq!(
        batch
            .outputs
            .iter()
            .map(|output| output.destination_component_base)
            .collect::<Vec<_>>(),
        [3, 2]
    );
    assert!(batch.outputs.iter().all(|output| output.initialize));

    let census = unsafe { &*contexts.color.cast::<InteractionDispatchCensus>() };
    census.calls.fetch_add(1, Ordering::Relaxed);
    Ok(())
}

unsafe extern "C" fn finalize_currents(
    _context: *const c_void,
    arena: DirectArenaView,
    _momenta: DirectMomentumView,
    _parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectFinalizationRow,
    row_count: u32,
    point_count: u32,
) -> i32 {
    let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
    for row in rows {
        if row.exact_factor_id >= factors.value_count {
            return STATUS_BOUNDS;
        }
        let factor = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
        for component in 0..usize::from(row.component_count) {
            for point in 0..point_count as usize {
                let destination =
                    (row.component_base as usize + component) * arena.point_stride as usize + point;
                if destination >= arena.current_scalar_len as usize {
                    return STATUS_BOUNDS;
                }
                unsafe {
                    *arena.current_re.add(destination) *= factor;
                    *arena.current_im.add(destination) *= factor;
                }
            }
        }
    }
    DIRECT_STATUS_OK
}

unsafe extern "C" fn accumulate_closures(
    _context: *const c_void,
    arena: DirectArenaView,
    _momenta: DirectMomentumView,
    _parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectClosureRow,
    row_count: u32,
    point_count: u32,
) -> i32 {
    let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
    for row in rows {
        if row.exact_factor_id >= factors.value_count {
            return STATUS_BOUNDS;
        }
        let factor = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
        for point in 0..point_count as usize {
            let source = row.parent0_component_base as usize * arena.point_stride as usize + point;
            let destination =
                row.amplitude_destination_id as usize * arena.point_stride as usize + point;
            if source >= arena.current_scalar_len as usize
                || destination >= arena.amplitude_scalar_len as usize
            {
                return STATUS_BOUNDS;
            }
            unsafe {
                *arena.amplitude_re.add(destination) += *arena.current_re.add(source) * factor;
                *arena.amplitude_im.add(destination) += *arena.current_im.add(source) * factor;
            }
        }
    }
    DIRECT_STATUS_OK
}

fn rational(numerator: i128, denominator: i128) -> ExactComplexRational {
    complex_rational(numerator, denominator, 0, 1)
}

fn complex_rational(
    real_numerator: i128,
    real_denominator: i128,
    imag_numerator: i128,
    imag_denominator: i128,
) -> ExactComplexRational {
    ExactComplexRational::new(
        ExactRational::new(real_numerator, real_denominator).unwrap(),
        ExactRational::new(imag_numerator, imag_denominator).unwrap(),
    )
}

fn replace_single_closure_proof_factor(
    parts: &mut crate::recurrence::DirectRecurrencePlanParts,
    exact_factor: ExactComplexRational,
    runtime_parent_ids: Option<Vec<Option<u32>>>,
) {
    let contribution = &parts.closure_proofs.contributions()[0];
    let contribution = ClosureProofContributionV2::new(
        contribution.id(),
        contribution.target_sector_id(),
        contribution.target_destination_id(),
        contribution.target_helicity_id(),
        contribution.closure_template_id(),
        contribution.closure_template_semantic_digest(),
        contribution.quantum_flow_template_id(),
        contribution.construction_parent_builder_ids().to_vec(),
        runtime_parent_ids
            .unwrap_or_else(|| contribution.construction_parent_runtime_ids().to_vec()),
        contribution.construction_parent_semantic_digests().to_vec(),
        contribution.construction_parent_color_digests().to_vec(),
        contribution.construction_parent_permutation().to_vec(),
        contribution.reconstruction_parent_permutation().to_vec(),
        contribution.evaluator_parent_permutation().to_vec(),
        contribution.color_witness_term_id(),
        contribution.color_witness_proof_digest(),
        contribution.three_line_certificate_id(),
        contribution.pairing_certificate_ids().to_vec(),
        contribution.reflection_certificate_id(),
        exact_factor,
        contribution.multiplicity(),
    )
    .unwrap();
    let group = &parts.closure_proofs.groups()[0];
    let closure = &parts.closures[group.emitted_direct_closure_row_id().unwrap() as usize];
    let component_start = closure.component_factor_start as usize;
    let component_end = component_start + closure.component_count as usize;
    let component_factor_digest =
        closure_component_factor_digest_v2(&parts.exact_factors[component_start..component_end])
            .unwrap();
    let group = ClosureExecutionProofGroupV2::new_with_candidate_selector_domain(
        group.id(),
        group.emitted_runtime_closure_term_id(),
        group.emitted_direct_closure_row_id(),
        group.contribution_range(),
        exact_factor,
        component_factor_digest,
        group.candidate_selector_domain_digest(),
        group.selector_domain_digest(),
    )
    .unwrap();
    parts.closure_proofs = ClosureProofMetadataV2::new_with_three_line_certificates(
        vec![contribution],
        vec![group],
        parts.closure_proofs.reflection_certificates().to_vec(),
        parts
            .closure_proofs
            .three_line_traversal_certificates()
            .to_vec(),
    )
    .unwrap();
}

#[cfg(any())]
fn synthetic_plan_and_executors() -> (DirectRecurrencePlan, DirectExecutorCatalog) {
    let plan = DirectRecurrencePlan::new(DirectRecurrencePlanParts {
        strategy: RecurrenceStrategy::TopologyReplay,
        semantic_digest: SemanticDigest::new([0x11; 32]).unwrap(),
        prepared_pack_digest: SemanticDigest::new([0x22; 32]).unwrap(),
        direct_template_catalog_digest: SemanticDigest::new([0x33; 32]).unwrap(),
        point_tile_size: 4,
        workspace_mib: 1,
        current_arena_components: 2,
        physical_sector_count: 2,
        retained_helicity_count: 2,
        amplitude_destination_count: 1,
        parameter_value_count: 1,
        external_source_count: 2,
        state_template_count: 2,
        source_template_or_dispatch_count: 1,
        direct_executor_count: 4,
        currents: vec![
            DirectCurrentDescriptor {
                semantic_current_id: 0,
                node_kind: DirectNodeKind::Source,
                state_template_id: 0,
                component_base: 0,
                component_count: 1,
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
                component_base: 1,
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
            source_template_or_dispatch_domain: 0,
            spin_state_class: 1,
            exact_factor_id: 3,
            selector_domain_id: 0,
        }],
        contributions: vec![DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: DIRECT_NONE_U32,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
            destination_component_base: 1,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        }],
        finalizations: vec![DirectFinalizationRow {
            component_base: 1,
            component_count: 1,
            momentum_form_id: 0,
            exact_factor_id: 1,
            selector_domain_id: 0,
            flags: 0,
        }],
        closures: vec![DirectClosureRow {
            parent0_component_base: 1,
            parent1_component_base_or_sentinel: DIRECT_NONE_U32,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
            amplitude_destination_id: 0,
            exact_factor_id: 2,
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
            term_count: 2,
        }],
        momentum_terms: vec![
            DirectMomentumTerm {
                source_slot: 0,
                coefficient: 1,
            },
            DirectMomentumTerm {
                source_slot: 1,
                coefficient: 2,
            },
        ],
        selector_domains: vec![DirectSelectorDomainDescriptor {
            word_start: 0,
            word_count: 1,
        }],
        selector_words: vec![1],
        replay_targets: vec![
            DirectReplayTargetDescriptor {
                public_flow_id: 0,
                representative_id: 0,
                source_permutation_start: 0,
                source_permutation_count: 2,
                helicity_map_start: 0,
                helicity_map_count: 1,
                phase_exact_factor_id: 3,
                multiplicity: 1,
                selector_domain_id: 0,
            },
            DirectReplayTargetDescriptor {
                public_flow_id: 1,
                representative_id: 0,
                source_permutation_start: 2,
                source_permutation_count: 2,
                helicity_map_start: 1,
                helicity_map_count: 1,
                phase_exact_factor_id: 4,
                multiplicity: 2,
                selector_domain_id: 0,
            },
        ],
        source_permutations: vec![0, 1, 1, 0],
        replay_momentum_signs: vec![1, 1, 1, 1],
        replay_helicity_map: vec![0, 0],
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
            public_helicity_start: 0,
            id: 0,
            source_state_count: 2,
            public_helicity_count: 2,
            selector_domain_id: 0,
        }],
        source_state_assignments: vec![
            DirectSourceStateAssignment {
                source_slot: 0,
                state_index: 0,
            },
            DirectSourceStateAssignment {
                source_slot: 1,
                state_index: 0,
            },
        ],
        public_helicities: vec![-1, 1],
        exact_factors: vec![
            rational(2, 1),
            rational(1, 2),
            rational(-1, 1),
            ExactComplexRational::ONE,
            complex_rational(0, 1, 1, 1),
        ],
    })
    .unwrap();
    let executors = DirectExecutorCatalog::new(
        &plan,
        plan.direct_template_catalog_digest(),
        direct_executor_handles(),
    )
    .unwrap();
    (plan, executors)
}

fn synthetic_plan_and_executors() -> (DirectRecurrencePlan, DirectExecutorCatalog) {
    let mut parts = crate::recurrence::direct_plan::tests::valid_parts();
    parts.point_tile_size = 4;
    parts.current_arena_components = 2;
    parts.physical_sector_count = 2;
    parts.retained_helicity_count = 2;
    parts.external_source_count = 2;
    parts.currents[0].component_count = 1;
    parts.currents[1].component_base = 1;
    parts.sources[0].destination_component_base = 0;
    parts.contributions[0].destination_component_base = 1;
    parts.contributions[0].exact_factor_id = 0;
    parts.finalizations[0].component_base = 1;
    parts.finalizations[0].exact_factor_id = 1;
    parts.closures[0].parent0_component_base = 1;
    parts.closures[0].exact_factor_id = 2;
    parts.momentum_forms[0].term_count = 2;
    parts.momentum_terms.push(DirectMomentumTerm {
        source_slot: 1,
        coefficient: 2,
    });
    parts.replay_targets = vec![
        DirectReplayTargetDescriptor {
            public_flow_id: 0,
            representative_id: 0,
            source_permutation_start: 0,
            source_permutation_count: 2,
            helicity_map_start: 0,
            helicity_map_count: 1,
            phase_exact_factor_id: 3,
            multiplicity: 1,
            selector_domain_id: 0,
        },
        DirectReplayTargetDescriptor {
            public_flow_id: 1,
            representative_id: 0,
            source_permutation_start: 2,
            source_permutation_count: 2,
            helicity_map_start: 1,
            helicity_map_count: 1,
            phase_exact_factor_id: 4,
            multiplicity: 2,
            selector_domain_id: 0,
        },
    ];
    parts.source_permutations = vec![0, 1, 1, 0];
    parts.replay_momentum_signs = vec![1, 1, 1, 1];
    parts.replay_helicity_map = vec![0, 0];
    parts.resolved_helicities[0].source_state_count = 2;
    parts.resolved_helicities[0].public_helicity_count = 2;
    parts.source_state_assignments = vec![
        DirectSourceStateAssignment {
            source_slot: 0,
            state_index: 0,
        },
        DirectSourceStateAssignment {
            source_slot: 1,
            state_index: 0,
        },
    ];
    parts.public_helicities = vec![-1, 1];
    parts.exact_factors = vec![
        rational(2, 1),
        rational(1, 2),
        rational(-1, 1),
        ExactComplexRational::ONE,
        complex_rational(0, 1, 1, 1),
    ];
    replace_single_closure_proof_factor(&mut parts, rational(-1, 1), None);
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    let executors = DirectExecutorCatalog::new(
        &plan,
        plan.direct_template_catalog_digest(),
        direct_executor_handles(),
    )
    .unwrap();
    (plan, executors)
}

fn synthetic_runtime_with_lorentz(
    lorentz_component_count: u16,
) -> DirectRecurrenceExecutionRuntime {
    let (plan, executors) = synthetic_plan_and_executors();
    let mut runtime =
        DirectRecurrenceExecutionRuntime::new(plan, executors, lorentz_component_count).unwrap();
    runtime.set_parameters(&[3.0], &[1.0]).unwrap();
    runtime
}

fn synthetic_runtime() -> DirectRecurrenceExecutionRuntime {
    synthetic_runtime_with_lorentz(1)
}

fn fanout_test_plan() -> DirectRecurrencePlan {
    let mut parts = crate::recurrence::direct_plan::tests::valid_parts();
    parts.point_tile_size = 8;
    parts.exact_factors = vec![
        ExactComplexRational::ONE,
        rational(-1, 1),
        complex_rational(0, 1, 1, 1),
        complex_rational(0, 1, -1, 1),
        rational(5, 7),
        complex_rational(0, 1, 2, 5),
        complex_rational(11, 13, -7, 17),
        complex_rational(10_000, 1, 3, 1),
        complex_rational(-10_000, 1, -3, 1),
    ];
    let prototype = parts.contributions[0];
    parts.contributions = (0_u32..=8)
        .flat_map(|exact_factor_id| [exact_factor_id; 2])
        .enumerate()
        .map(|(row_index, exact_factor_id)| DirectContributionRow {
            exact_factor_id,
            flags: u32::from(row_index == 0) * DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
            ..prototype
        })
        .collect();
    parts
        .row_groups
        .iter_mut()
        .find(|group| group.role == DirectExecutorRole::Contribution)
        .unwrap()
        .row_count = parts.contributions.len() as u32;
    DirectRecurrencePlan::new(parts).unwrap()
}

fn fanout_test_catalog(
    plan: &DirectRecurrencePlan,
    census: &ContributionCallCensus,
    exact_factor_is_kernel_input: bool,
) -> DirectExecutorCatalog {
    let mut handles = direct_executor_handles();
    handles[1] = DirectExecutorHandle::Contribution {
        call: counted_accumulate_contributions,
        context: census as *const ContributionCallCensus as *const c_void,
    };
    DirectExecutorCatalog::new_sparse_with_metadata(
        plan,
        plan.direct_template_catalog_digest(),
        handles.into_iter().map(Some).collect(),
        vec![
            None,
            Some(
                DirectContributionExecutionMetadata::new(1, exact_factor_is_kernel_input).unwrap(),
            ),
            None,
            None,
        ],
    )
    .unwrap()
}

fn interaction_dispatch_test_plan() -> DirectRecurrencePlan {
    let mut parts = crate::recurrence::direct_plan::tests::valid_parts();
    parts.point_tile_size = 4;
    parts.current_arena_components = 4;
    parts.direct_executor_count = 5;

    let source = parts.currents[0];
    let current = parts.currents[1];
    parts.currents[0].component_count = 1;
    parts.currents[1] = DirectCurrentDescriptor {
        semantic_current_id: 2,
        component_base: 2,
        component_count: 1,
        last_use: 1,
        ..current
    };
    parts.currents.insert(
        1,
        DirectCurrentDescriptor {
            semantic_current_id: 1,
            component_base: 1,
            component_count: 1,
            source_row_or_sentinel: 1,
            last_use: 2,
            ..source
        },
    );
    parts.currents.push(DirectCurrentDescriptor {
        semantic_current_id: 3,
        component_base: 3,
        component_count: 1,
        last_use: 1,
        finalization_row_or_sentinel: DIRECT_NONE_U32,
        ..current
    });

    let source_row = parts.sources[0];
    parts.sources.push(DirectSourceRow {
        destination_component_base: 1,
        ..source_row
    });

    let contribution = parts.contributions[0];
    parts.contributions = vec![
        DirectContributionRow {
            parent0_component_base: 1,
            parent1_component_base_or_sentinel: 0,
            parent1_momentum_form_id_or_sentinel: 0,
            destination_component_base: 2,
            flags: 0,
            ..contribution
        },
        DirectContributionRow {
            parent0_component_base: 1,
            parent1_component_base_or_sentinel: 0,
            parent1_momentum_form_id_or_sentinel: 0,
            destination_component_base: 3,
            flags: 0,
            ..contribution
        },
        DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: 1,
            parent1_momentum_form_id_or_sentinel: 0,
            destination_component_base: 3,
            flags: 0,
            ..contribution
        },
        DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: 1,
            parent1_momentum_form_id_or_sentinel: 0,
            destination_component_base: 2,
            flags: 0,
            ..contribution
        },
    ];
    parts.finalizations[0].component_base = 2;
    parts.closures[0].parent0_component_base = 1;
    parts.row_groups[0].row_count = 2;
    parts.row_groups[1].row_count = 2;
    parts.row_groups.insert(
        2,
        DirectRowGroupDescriptor {
            direct_executor_id: 4,
            row_start: 2,
            row_count: 2,
            ..parts.row_groups[1]
        },
    );
    DirectRecurrencePlan::new(parts).unwrap()
}

fn interaction_dispatch_test_catalog(
    plan: &DirectRecurrencePlan,
    ordinary: &ContributionCallCensus,
    interaction: &InteractionDispatchCensus,
) -> DirectExecutorCatalog {
    let ordinary_context = ordinary as *const ContributionCallCensus as *const c_void;
    let interaction_context = interaction as *const InteractionDispatchCensus as *const c_void;
    let mut handles = direct_executor_handles();
    handles[1] = DirectExecutorHandle::Contribution {
        call: counted_accumulate_contributions,
        context: ordinary_context,
    };
    handles.push(DirectExecutorHandle::Contribution {
        call: counted_accumulate_contributions,
        context: ordinary_context,
    });
    let mut metadata = vec![None; 5];
    metadata[1] = Some(DirectContributionExecutionMetadata::new(1, false).unwrap());
    metadata[4] = Some(DirectContributionExecutionMetadata::new(1, false).unwrap());
    let mut fanout = vec![None; 5];
    fanout[1] = Some(DirectContributionFanoutExecutorHandle {
        call: unexpected_single_point_fanout,
        context: interaction_context,
        destination_component_count: 1,
        parent_component_counts: [1, 1],
        requires_two_momenta: false,
        required_parameter_count: 0,
        interaction_kind: DirectInteractionIntrinsicKind::AntisymmetricTensorVector,
        interaction_call: None,
    });
    fanout[4] = Some(DirectContributionFanoutExecutorHandle {
        call: unexpected_single_point_fanout,
        context: interaction_context,
        destination_component_count: 1,
        parent_component_counts: [1, 1],
        requires_two_momenta: false,
        required_parameter_count: 0,
        interaction_kind: DirectInteractionIntrinsicKind::ColorOrderedThreeVector,
        interaction_call: Some(inspect_nonidentity_interaction_dispatch),
    });
    DirectExecutorCatalog::new_sparse_with_metadata_and_fanout(
        plan,
        plan.direct_template_catalog_digest(),
        handles.into_iter().map(Some).collect(),
        metadata,
        fanout,
    )
    .unwrap()
}

fn direct_baseline(
    plan: &DirectRecurrencePlan,
    executors: &DirectExecutorCatalog,
    point_stride: u32,
    point_count: u32,
    momenta: &[f64],
    parameter: (f64, f64),
) -> Vec<(f64, f64)> {
    let mut current_re =
        vec![0.0; plan.current_arena_components() as usize * point_stride as usize];
    let mut current_im = vec![0.0; current_re.len()];
    let mut amplitude_re =
        vec![0.0; plan.amplitude_destination_count() as usize * point_stride as usize];
    let mut amplitude_im = vec![0.0; amplitude_re.len()];
    let factors_re = plan
        .exact_factors()
        .iter()
        .map(|factor| factor.real().numerator() as f64 / factor.real().denominator() as f64)
        .collect::<Vec<_>>();
    let factors_im = plan
        .exact_factors()
        .iter()
        .map(|factor| factor.imag().numerator() as f64 / factor.imag().denominator() as f64)
        .collect::<Vec<_>>();
    let mut workspace = DirectWorkspace {
        current_re: &mut current_re,
        current_im: &mut current_im,
        amplitude_re: &mut amplitude_re,
        amplitude_im: &mut amplitude_im,
        momenta,
        momentum_form_count: plan.momentum_forms().len() as u32,
        lorentz_component_count: 1,
        parameters_re: &[parameter.0],
        parameters_im: &[parameter.1],
        factors_re: &factors_re,
        factors_im: &factors_im,
        point_stride,
    };
    let mut counters = DirectExecutionCounters::default();
    execute_direct_plan(plan, executors, &mut workspace, point_count, &mut counters).unwrap();
    assert_eq!(
        counters.contribution_rows,
        plan.contributions().len() as u64
    );
    amplitude_re[..point_count as usize]
        .iter()
        .copied()
        .zip(amplitude_im[..point_count as usize].iter().copied())
        .collect()
}

fn assert_scale_relative_complex_parity(baseline: &[(f64, f64)], candidate: &[(f64, f64)]) {
    assert_eq!(baseline.len(), candidate.len());
    for (point, (&expected, &observed)) in baseline.iter().zip(candidate).enumerate() {
        let error = (expected.0 - observed.0).hypot(expected.1 - observed.1);
        let scale = expected
            .0
            .hypot(expected.1)
            .max(observed.0.hypot(observed.1))
            .max(1.0);
        assert!(
            error <= 1.0e-10 * scale,
            "point {point} differs by {error:e} at scale {scale:e}: expected {expected:?}, observed {observed:?}"
        );
    }
}

#[test]
fn sparse_executor_catalog_resolves_arbitrary_referenced_ids_and_keeps_unused_holes() {
    let mut parts = crate::recurrence::direct_plan::tests::valid_parts();
    parts.direct_executor_count = 8;
    let selected_ids = [6_u32, 1, 7, 3];
    for (descriptor, executor_id) in parts.row_groups.iter_mut().zip(selected_ids) {
        descriptor.direct_executor_id = executor_id;
    }
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    let dense_handles = direct_executor_handles();
    let mut sparse_handles = vec![None; plan.direct_executor_count() as usize];
    for (handle, executor_id) in dense_handles.into_iter().zip(selected_ids) {
        sparse_handles[executor_id as usize] = Some(handle);
    }

    DirectExecutorCatalog::new_sparse(
        &plan,
        plan.direct_template_catalog_digest(),
        sparse_handles.clone(),
    )
    .unwrap();

    sparse_handles[selected_ids[2] as usize] = None;
    let error = DirectExecutorCatalog::new_sparse(
        &plan,
        plan.direct_template_catalog_digest(),
        sparse_handles,
    )
    .err()
    .unwrap();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Evaluation);
    assert!(error.to_string().contains("executor 7 is not loaded"));
}

#[test]
fn interaction_dispatch_uses_nonidentity_bijection_and_multipoint_falls_back() {
    let plan = interaction_dispatch_test_plan();
    let ordinary = ContributionCallCensus::default();
    let interaction = InteractionDispatchCensus::default();
    let catalog = interaction_dispatch_test_catalog(&plan, &ordinary, &interaction);
    let mut runtime = DirectRecurrenceExecutionRuntime::new(plan, catalog, 1).unwrap();
    runtime.set_parameters(&[1.0], &[0.0]).unwrap();

    runtime.execute_tile(1).unwrap();
    assert_eq!(interaction.calls.load(Ordering::Relaxed), 1);
    assert_eq!(ordinary.calls.load(Ordering::Relaxed), 0);

    runtime.execute_tile(2).unwrap();
    assert_eq!(interaction.calls.load(Ordering::Relaxed), 1);
    assert_eq!(ordinary.calls.load(Ordering::Relaxed), 2);
    assert_eq!(ordinary.rows.load(Ordering::Relaxed), 2);
}

#[test]
fn output_only_contribution_fanout_matches_direct_under_complex_cancellation() {
    let plan = fanout_test_plan();
    let point_stride = 8;
    let point_count = 7;
    let momenta = [0.125, -0.875, 1.75, -3.5, 0.0625, 7.25, -11.0, 0.0];
    let parameter = (0.375, -1.125);

    let factor_norm_sum = plan
        .exact_factors()
        .iter()
        .skip(1)
        .map(|factor| {
            let real = factor.real().numerator() as f64 / factor.real().denominator() as f64;
            let imaginary = factor.imag().numerator() as f64 / factor.imag().denominator() as f64;
            real.hypot(imaginary)
        })
        .sum::<f64>();
    let factor_sum = plan
        .exact_factors()
        .iter()
        .skip(1)
        .fold((0.0_f64, 0.0_f64), |sum, factor| {
            (
                sum.0 + factor.real().numerator() as f64 / factor.real().denominator() as f64,
                sum.1 + factor.imag().numerator() as f64 / factor.imag().denominator() as f64,
            )
        });
    assert!(factor_norm_sum > 1_000.0 * factor_sum.0.hypot(factor_sum.1));

    let baseline_census = ContributionCallCensus::default();
    let baseline_catalog = fanout_test_catalog(&plan, &baseline_census, false);
    let baseline = direct_baseline(
        &plan,
        &baseline_catalog,
        point_stride,
        point_count,
        &momenta,
        parameter,
    );
    assert_eq!(baseline_census.calls.load(Ordering::Relaxed), 1);
    assert_eq!(baseline_census.rows.load(Ordering::Relaxed), 18);

    let optimized_census = ContributionCallCensus::default();
    let optimized_catalog = fanout_test_catalog(&plan, &optimized_census, false);
    let fanout = DirectContributionFanoutProgram::build(&plan, &optimized_catalog).unwrap();
    assert_eq!(fanout.row_counts(), (18, 1));
    assert_eq!(fanout.scatter_bundle_counts(), (18, 9));
    assert_eq!(fanout.scratch_component_count(), 1);
    assert!(fanout.needs_unit_factor());
    let mut runtime =
        DirectRecurrenceExecutionRuntime::new(plan.clone(), optimized_catalog, 1).unwrap();
    runtime
        .set_parameters(&[parameter.0], &[parameter.1])
        .unwrap();
    runtime
        .momentum_plane_mut(0, 0)
        .unwrap()
        .copy_from_slice(&momenta);
    assert_eq!(runtime.factors_mut().0.len(), plan.exact_factors().len());
    assert_eq!(
        runtime.current_arenas().0.len(),
        plan.current_arena_components() as usize * runtime.point_stride() as usize
    );
    let storage = storage_identity(&mut runtime);
    let output = runtime.execute_tile(point_count).unwrap();
    let candidate = output
        .destination_re(0)
        .unwrap()
        .iter()
        .copied()
        .zip(output.destination_im(0).unwrap().iter().copied())
        .collect::<Vec<_>>();
    assert_scale_relative_complex_parity(&baseline, &candidate);
    assert_eq!(optimized_census.calls.load(Ordering::Relaxed), 1);
    assert_eq!(optimized_census.rows.load(Ordering::Relaxed), 1);

    runtime.execute_tile(point_count).unwrap();
    assert_eq!(storage_identity(&mut runtime), storage);
    assert_eq!(optimized_census.calls.load(Ordering::Relaxed), 2);
    assert_eq!(optimized_census.rows.load(Ordering::Relaxed), 2);

    let tail_count = 5;
    let tail_baseline = direct_baseline(
        &plan,
        &baseline_catalog,
        point_stride,
        tail_count,
        &momenta,
        parameter,
    );
    let tail_output = runtime.execute_tile(tail_count).unwrap();
    let tail_candidate = tail_output
        .destination_re(0)
        .unwrap()
        .iter()
        .copied()
        .zip(tail_output.destination_im(0).unwrap().iter().copied())
        .collect::<Vec<_>>();
    assert_scale_relative_complex_parity(&tail_baseline, &tail_candidate);
    assert_eq!(storage_identity(&mut runtime), storage);
}

#[test]
fn full_fanout_execution_overwrites_poisoned_destination_without_stage_clear() {
    let plan = fanout_test_plan();
    let point_stride = 8_u32;
    let point_count = 7_u32;
    let momenta = [0.125, -0.875, 1.75, -3.5, 0.0625, 7.25, -11.0, 0.0];
    let parameter = (0.375, -1.125);
    let census = ContributionCallCensus::default();
    let catalog = fanout_test_catalog(&plan, &census, false);
    let fanout = DirectContributionFanoutProgram::build(&plan, &catalog).unwrap();
    let current_plane_count = plan.current_arena_components() + fanout.scratch_component_count();
    let mut current_re = vec![f64::NAN; current_plane_count as usize * point_stride as usize];
    let mut current_im = vec![f64::NAN; current_re.len()];
    let mut amplitude_re = vec![0.0; point_stride as usize];
    let mut amplitude_im = vec![0.0; point_stride as usize];
    let mut factors_re = plan
        .exact_factors()
        .iter()
        .map(|factor| factor.real().numerator() as f64 / factor.real().denominator() as f64)
        .collect::<Vec<_>>();
    let mut factors_im = plan
        .exact_factors()
        .iter()
        .map(|factor| factor.imag().numerator() as f64 / factor.imag().denominator() as f64)
        .collect::<Vec<_>>();
    factors_re.push(1.0);
    factors_im.push(0.0);

    let execute = |current_re: &mut [f64],
                   current_im: &mut [f64],
                   amplitude_re: &mut [f64],
                   amplitude_im: &mut [f64]| {
        let mut workspace = DirectWorkspace {
            current_re,
            current_im,
            amplitude_re,
            amplitude_im,
            momenta: &momenta,
            momentum_form_count: plan.momentum_forms().len() as u32,
            lorentz_component_count: 1,
            parameters_re: &[parameter.0],
            parameters_im: &[parameter.1],
            factors_re: &factors_re,
            factors_im: &factors_im,
            point_stride,
        };
        execute_direct_plan_unprofiled_with_fanout(
            &plan,
            &catalog,
            &fanout,
            &mut workspace,
            point_count,
        )
        .unwrap();
    };

    execute(
        &mut current_re,
        &mut current_im,
        &mut amplitude_re,
        &mut amplitude_im,
    );
    let first = amplitude_re[..point_count as usize]
        .iter()
        .copied()
        .zip(amplitude_im[..point_count as usize].iter().copied())
        .collect::<Vec<_>>();
    current_re.fill(f64::NAN);
    current_im.fill(f64::NAN);
    amplitude_re.fill(0.0);
    amplitude_im.fill(0.0);
    execute(
        &mut current_re,
        &mut current_im,
        &mut amplitude_re,
        &mut amplitude_im,
    );
    let second = amplitude_re[..point_count as usize]
        .iter()
        .copied()
        .zip(amplitude_im[..point_count as usize].iter().copied())
        .collect::<Vec<_>>();

    assert!(
        second
            .iter()
            .all(|(real, imaginary)| real.is_finite() && imaginary.is_finite())
    );
    assert_scale_relative_complex_parity(&first, &second);
}

#[test]
fn factor_consuming_contribution_fanout_is_fail_closed() {
    let plan = fanout_test_plan();
    let census = ContributionCallCensus::default();
    let catalog = fanout_test_catalog(&plan, &census, true);
    let fanout = DirectContributionFanoutProgram::build(&plan, &catalog).unwrap();
    assert_eq!(fanout.row_counts(), (18, 9));
    assert_eq!(fanout.scatter_bundle_counts(), (18, 9));
    assert_eq!(fanout.scratch_component_count(), 9);
    assert!(!fanout.needs_unit_factor());

    let mut bad_metadata = vec![None; plan.direct_executor_count() as usize];
    bad_metadata[1] = Some(DirectContributionExecutionMetadata::new(2, false).unwrap());
    let error = DirectExecutorCatalog::new_sparse_with_metadata(
        &plan,
        plan.direct_template_catalog_digest(),
        direct_executor_handles().into_iter().map(Some).collect(),
        bad_metadata,
    )
    .and_then(|catalog| DirectContributionFanoutProgram::build(&plan, &catalog))
    .err()
    .unwrap();
    assert!(
        error
            .to_string()
            .contains("declares 2 destination components")
    );
}

fn output_only_physical_fanout_census(plan: &DirectRecurrencePlan) -> (u64, u64) {
    let mut logical = 0_u64;
    let mut evaluated = 0_u64;
    for descriptor in plan.row_groups().iter().filter(|descriptor| {
        descriptor.role == DirectExecutorRole::Contribution
            && descriptor.direct_executor_id != DIRECT_NONE_U32
    }) {
        let start = descriptor.row_start as usize;
        let end = start + descriptor.row_count as usize;
        let rows = &plan.contributions()[start..end];
        logical += rows.len() as u64;
        let mut classes = std::collections::BTreeSet::new();
        for row in rows {
            if row.flags & crate::recurrence::DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE != 0 {
                evaluated += 1;
                continue;
            }
            classes.insert((
                row.selector_domain_id,
                row.parent0_component_base,
                row.parent1_component_base_or_sentinel,
                row.parent0_momentum_form_id,
                row.parent1_momentum_form_id_or_sentinel,
            ));
        }
        evaluated += classes.len() as u64;
    }
    (logical, evaluated)
}

#[test]
#[ignore = "release census requires PYAMPLICOL_FANOUT_N7_SCHEDULE and PYAMPLICOL_FANOUT_N8_SCHEDULE"]
fn production_gluon_artifacts_certify_the_expected_fanout_reduction() {
    for (label, variable, minimum_reduction) in [
        ("N7", "PYAMPLICOL_FANOUT_N7_SCHEDULE", 0.47),
        ("N8", "PYAMPLICOL_FANOUT_N8_SCHEDULE", 0.49),
    ] {
        let path = std::env::var(variable).unwrap_or_else(|_| panic!("{variable} is required"));
        let plan = crate::recurrence::load_recurrence_direct_plan_pacbin(path).unwrap();
        let (logical, evaluated) = output_only_physical_fanout_census(&plan);
        let reduction = 1.0 - evaluated as f64 / logical as f64;
        eprintln!(
            "{label} output-only physical fanout: logical={logical}, evaluated={evaluated}, reduction={reduction:.6}",
        );
        assert!(
            reduction >= minimum_reduction,
            "{label} fanout reduction {reduction:.6} is below {minimum_reduction:.6}"
        );
    }
}

#[test]
fn direct_plan_rejects_a_row_group_executor_outside_the_catalog_domain() {
    let mut parts = crate::recurrence::direct_plan::tests::valid_parts();
    parts.row_groups[0].direct_executor_id = parts.direct_executor_count;
    let error = DirectRecurrencePlan::new(parts).err().unwrap();
    assert_eq!(error.kind(), crate::RusticolErrorKind::InvalidArgument);
}

#[test]
fn low_footprint_runtime_retains_the_requested_point_tile() {
    let (plan, executors) = synthetic_plan_and_executors();
    assert_eq!(plan.point_tile_size(), 4);
    let mut runtime = DirectRecurrenceExecutionRuntime::new(plan, executors, 1).unwrap();
    assert_eq!(runtime.point_tile_size(), 4);
    assert_eq!(runtime.point_stride(), 8);
    assert_eq!(runtime.momenta_mut().len(), 4);
    assert_eq!(runtime.physical_momenta_mut().len(), 8);
}

#[test]
fn replay_cache_footprint_uses_authenticated_active_selector_work() {
    let mut parts = crate::recurrence::direct_plan::tests::valid_parts();
    parts.current_arena_components = 128;
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    let persisted_split_scalars =
        2 * (plan.current_arena_components() + plan.amplitude_destination_count()) as usize;

    assert_eq!(persisted_split_scalars, 258);
    assert_eq!(
        replay_cache_split_complex_scalar_count(&plan, persisted_split_scalars, 0).unwrap(),
        8
    );
}

#[test]
fn legacy_flat_momentum_access_round_trips_every_padded_plane_in_place() {
    let (plan, executors) = synthetic_plan_and_executors();
    let mut runtime = DirectRecurrenceExecutionRuntime::new(plan, executors, 4).unwrap();
    let tile_capacity = runtime.point_tile_size() as usize;
    let point_stride = runtime.point_stride() as usize;
    assert_eq!(tile_capacity, 4);
    assert_eq!(point_stride, 8);

    let compact = runtime.momenta_mut();
    for plane in 0..4 {
        for point in 0..tile_capacity {
            compact[plane * tile_capacity + point] = (plane * 10 + point) as f64;
        }
    }
    let physical = runtime.physical_momenta_mut();
    for plane in 0..4 {
        for point in 0..tile_capacity {
            assert_eq!(
                physical[plane * point_stride + point],
                (plane * 10 + point) as f64
            );
        }
    }

    let compact = runtime.momenta_mut();
    for plane in 0..4 {
        for point in 0..tile_capacity {
            assert_eq!(
                compact[plane * tile_capacity + point],
                (plane * 10 + point) as f64
            );
        }
    }
}

#[test]
fn high_footprint_runtime_uses_a_power_of_two_cache_tile() {
    let mut parts = crate::recurrence::direct_plan::tests::valid_parts();
    parts.strategy = RecurrenceStrategy::ContractedColorUnion;
    parts.point_tile_size = 1024;
    parts.workspace_mib = 256;
    parts.current_arena_components = 4_000;
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    assert_eq!(plan.point_tile_size(), 1024);
    let executors = DirectExecutorCatalog::new(
        &plan,
        plan.direct_template_catalog_digest(),
        direct_executor_handles(),
    )
    .unwrap();
    let runtime = DirectRecurrenceExecutionRuntime::new(plan, executors, 4).unwrap();
    assert_eq!(runtime.point_tile_size(), 64);
}

#[test]
fn cache_target_never_rejects_a_point_that_fits_the_workspace_limit() {
    let mut parts = crate::recurrence::direct_plan::tests::valid_parts();
    parts.point_tile_size = 1024;
    parts.workspace_mib = 64;
    parts.current_arena_components = 262_144;
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    let executors = DirectExecutorCatalog::new(
        &plan,
        plan.direct_template_catalog_digest(),
        direct_executor_handles(),
    )
    .unwrap();
    let runtime = DirectRecurrenceExecutionRuntime::new(plan, executors, 4).unwrap();
    assert_eq!(runtime.point_tile_size(), 8);
    assert_eq!(runtime.point_stride(), 8);
}

#[test]
fn hard_workspace_budget_counts_the_minimum_aligned_physical_pitch() {
    let mut parts = crate::recurrence::direct_plan::tests::valid_parts();
    parts.point_tile_size = 1024;
    parts.workspace_mib = 8;
    parts.current_arena_components = 262_144;
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    let executors = DirectExecutorCatalog::new(
        &plan,
        plan.direct_template_catalog_digest(),
        direct_executor_handles(),
    )
    .unwrap();
    let error = DirectRecurrenceExecutionRuntime::new(plan, executors, 4)
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("minimum aligned Direct-Arena pitch")
    );
}

fn storage_identity(runtime: &mut DirectRecurrenceExecutionRuntime) -> ([usize; 9], [usize; 9]) {
    let ((current_re_pointer, current_re_len), (current_im_pointer, current_im_len)) = {
        let (current_re, current_im) = runtime.current_arenas();
        (
            (current_re.as_ptr() as usize, current_re.len()),
            (current_im.as_ptr() as usize, current_im.len()),
        )
    };
    let ((amplitude_re_pointer, amplitude_re_len), (amplitude_im_pointer, amplitude_im_len)) = {
        let (amplitude_re, amplitude_im) = runtime.amplitude_arenas();
        (
            (amplitude_re.as_ptr() as usize, amplitude_re.len()),
            (amplitude_im.as_ptr() as usize, amplitude_im.len()),
        )
    };
    let (momenta_pointer, momenta_len) = {
        let values = runtime.momenta_mut();
        (values.as_ptr() as usize, values.len())
    };
    let ((parameters_re_pointer, parameters_re_len), (parameters_im_pointer, parameters_im_len)) = {
        let (parameters_re, parameters_im) = runtime.parameters_mut();
        (
            (parameters_re.as_ptr() as usize, parameters_re.len()),
            (parameters_im.as_ptr() as usize, parameters_im.len()),
        )
    };
    let ((factors_re_pointer, factors_re_len), (factors_im_pointer, factors_im_len)) = {
        let (factors_re, factors_im) = runtime.factors_mut();
        (
            (factors_re.as_ptr() as usize, factors_re.len()),
            (factors_im.as_ptr() as usize, factors_im.len()),
        )
    };
    (
        [
            current_re_pointer,
            current_im_pointer,
            amplitude_re_pointer,
            amplitude_im_pointer,
            momenta_pointer,
            parameters_re_pointer,
            parameters_im_pointer,
            factors_re_pointer,
            factors_im_pointer,
        ],
        [
            current_re_len,
            current_im_len,
            amplitude_re_len,
            amplitude_im_len,
            momenta_len,
            parameters_re_len,
            parameters_im_len,
            factors_re_len,
            factors_im_len,
        ],
    )
}

fn external_two_point_momenta() -> [f64; 16] {
    [
        1.0, 10.0, 100.0, 1000.0, 4.0, 40.0, 400.0, 4000.0, 2.0, 20.0, 200.0, 2000.0, 5.0, 50.0,
        500.0, 5000.0,
    ]
}

fn external_three_point_momenta() -> [f64; 24] {
    [
        1.0, 10.0, 100.0, 1000.0, 4.0, 40.0, 400.0, 4000.0, 2.0, 20.0, 200.0, 2000.0, 5.0, 50.0,
        500.0, 5000.0, 7.0, 70.0, 700.0, 7000.0, 8.0, 80.0, 800.0, 8000.0,
    ]
}

#[test]
fn warmed_tiles_reuse_stable_aligned_storage_and_return_correct_borrowed_outputs() {
    let mut runtime = synthetic_runtime();
    let identity = storage_identity(&mut runtime);
    assert!(
        identity
            .0
            .iter()
            .all(|pointer| pointer.is_multiple_of(DIRECT_RUNTIME_ARENA_ALIGNMENT))
    );

    for _ in 0..8 {
        runtime
            .momentum_plane_mut(0, 0)
            .unwrap()
            .copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let output = runtime.execute_tile(4).unwrap();
        assert_eq!(
            output.destination_re(0).unwrap(),
            &[-3.0, -6.0, -9.0, -12.0]
        );
        assert_eq!(output.destination_im(0).unwrap(), &[-1.0, -2.0, -3.0, -4.0]);
        assert_eq!(output.destination_re(1), None);
        assert_eq!(storage_identity(&mut runtime), identity);
    }

    let (current_re, current_im) = runtime.current_arenas();
    assert_eq!(&current_re[0..4], &[1.0, 2.0, 3.0, 4.0]);
    let point_stride = runtime.point_stride() as usize;
    assert_eq!(
        &current_re[point_stride..point_stride + 4],
        &[3.0, 6.0, 9.0, 12.0]
    );
    assert_eq!(&current_im[0..4], &[0.0; 4]);
    assert_eq!(
        &current_im[point_stride..point_stride + 4],
        &[1.0, 2.0, 3.0, 4.0]
    );
    let counters = runtime.counters();
    assert_eq!(counters.source_calls, 8);
    assert_eq!(counters.source_rows, 8);
    assert_eq!(counters.contribution_calls, 8);
    assert_eq!(counters.contribution_rows, 8);
    assert_eq!(counters.finalization_calls, 8);
    assert_eq!(counters.finalization_rows, 8);
    assert_eq!(counters.closure_calls, 8);
    assert_eq!(counters.closure_rows, 8);
    assert_eq!(counters.packed_input_bytes, 0);
    assert_eq!(counters.packed_output_bytes, 0);
    assert_eq!(counters.scatter_bytes, 0);
    let allocation_counters = runtime.allocation_counters();
    assert_eq!(allocation_counters.allocation_requests, 9);
    assert!(allocation_counters.requested_bytes != 0);
    let traffic = runtime.traffic_counters();
    assert_eq!(traffic.calls, 32);
    assert_eq!(traffic.rows, 32);
    assert_eq!(traffic.points, 128);
    traffic.validate_direct().unwrap();
}

#[test]
fn elided_identity_finalizer_remains_correct_across_repeated_evaluations() {
    let mut parts = crate::recurrence::direct_plan::tests::valid_parts();
    parts.point_tile_size = 2;
    parts.currents[1].finalization_row_or_sentinel = DIRECT_NONE_U32;
    parts.finalizations.clear();
    parts
        .row_groups
        .retain(|group| group.role != DirectExecutorRole::Finalization);
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    let executors = DirectExecutorCatalog::new(
        &plan,
        plan.direct_template_catalog_digest(),
        direct_executor_handles(),
    )
    .unwrap();
    let mut runtime = DirectRecurrenceExecutionRuntime::new(plan, executors, 1).unwrap();
    runtime.set_parameters(&[1.0], &[0.0]).unwrap();

    for values in [[2.0, 3.0], [5.0, 7.0], [11.0, 13.0]] {
        runtime
            .momentum_plane_mut(0, 0)
            .unwrap()
            .copy_from_slice(&values);
        let output = runtime.execute_tile(2).unwrap();
        assert_eq!(output.destination_re(0).unwrap(), values.as_slice());
        assert_eq!(output.destination_im(0).unwrap(), &[0.0, 0.0]);
    }
    assert_eq!(runtime.counters().finalization_calls, 0);
    assert_eq!(runtime.counters().finalization_rows, 0);
}

#[test]
fn a_reused_source_slot_is_cleared_before_the_later_current_accumulates() {
    let mut parts = crate::recurrence::direct_plan::tests::valid_parts();
    parts.point_tile_size = 1;
    parts.current_arena_components = 2;
    parts.currents[0].component_count = 1;
    parts.currents[0].last_use = 1;
    parts.currents[1].component_base = 1;
    parts.currents[1].first_use = 1;
    parts.currents[1].last_use = 2;
    parts.currents.push(DirectCurrentDescriptor {
        semantic_current_id: 2,
        node_kind: DirectNodeKind::Current,
        state_template_id: 1,
        component_base: 0,
        component_count: 1,
        momentum_form_id: 0,
        stage: 2,
        selector_domain_id: 0,
        first_use: 2,
        last_use: 3,
        source_row_or_sentinel: DIRECT_NONE_U32,
        finalization_row_or_sentinel: 1,
    });
    parts.contributions[0].destination_component_base = 1;
    parts.contributions.push(DirectContributionRow {
        parent0_component_base: 1,
        parent1_component_base_or_sentinel: DIRECT_NONE_U32,
        parent0_momentum_form_id: 0,
        parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
        destination_component_base: 0,
        exact_factor_id: 0,
        selector_domain_id: 0,
        flags: DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
    });
    parts.finalizations[0].component_base = 1;
    parts.finalizations.push(DirectFinalizationRow {
        component_base: 0,
        component_count: 1,
        momentum_form_id: 0,
        exact_factor_id: 0,
        selector_domain_id: 0,
        flags: 0,
    });
    parts.closures[0].parent0_component_base = 0;
    replace_single_closure_proof_factor(&mut parts, ExactComplexRational::ONE, Some(vec![Some(2)]));
    parts.row_groups = vec![
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
            role: DirectExecutorRole::Contribution,
            destination_operation: DirectDestinationOperation::Add,
            direct_executor_id: 1,
            row_start: 1,
            row_count: 1,
        },
        DirectRowGroupDescriptor {
            stage: 2,
            role: DirectExecutorRole::Finalization,
            destination_operation: DirectDestinationOperation::FinalizeInPlace,
            direct_executor_id: 2,
            row_start: 1,
            row_count: 1,
        },
        DirectRowGroupDescriptor {
            stage: 3,
            role: DirectExecutorRole::Closure,
            destination_operation: DirectDestinationOperation::ClosureAdd,
            direct_executor_id: 3,
            row_start: 0,
            row_count: 1,
        },
    ];
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    let executors = DirectExecutorCatalog::new(
        &plan,
        plan.direct_template_catalog_digest(),
        direct_executor_handles(),
    )
    .unwrap();
    let mut runtime = DirectRecurrenceExecutionRuntime::new(plan, executors, 1).unwrap();
    runtime.set_parameters(&[3.0], &[1.0]).unwrap();
    runtime.momentum_plane_mut(0, 0).unwrap()[0] = 2.0;

    let output = runtime.execute_tile(1).unwrap();
    assert_eq!(output.destination_re(0).unwrap(), &[16.0]);
    assert_eq!(output.destination_im(0).unwrap(), &[12.0]);
}

#[test]
fn external_momentum_fill_resolves_forms_once_and_clears_newly_inactive_tails() {
    let mut runtime = synthetic_runtime_with_lorentz(4);
    let flow_zero = runtime.prepare_replay_selector(0).unwrap();
    let flow_one = runtime.prepare_replay_selector(1).unwrap();
    let identity = storage_identity(&mut runtime);

    assert!(
        runtime
            .fill_momenta_from_external(&flow_zero, 3, &external_two_point_momenta())
            .unwrap_err()
            .to_string()
            .contains("expected 24")
    );
    runtime
        .fill_momenta_from_external(&flow_zero, 3, &external_three_point_momenta())
        .unwrap();
    let point_stride = runtime.point_tile_size() as usize;
    assert_eq!(runtime.momenta_mut().len(), 4 * point_stride);
    assert_eq!(&runtime.momenta_mut()[0..4], &[9.0, 12.0, 23.0, 0.0]);
    assert_eq!(
        &runtime.momenta_mut()[point_stride..point_stride + 4],
        &[90.0, 120.0, 230.0, 0.0]
    );
    assert_eq!(
        &runtime.momenta_mut()[2 * point_stride..2 * point_stride + 4],
        &[900.0, 1200.0, 2300.0, 0.0]
    );
    assert_eq!(
        &runtime.momenta_mut()[3 * point_stride..3 * point_stride + 4],
        &[9000.0, 12000.0, 23000.0, 0.0]
    );

    runtime
        .fill_momenta_from_external(&flow_one, 2, &external_two_point_momenta())
        .unwrap();
    assert_eq!(&runtime.momenta_mut()[0..4], &[6.0, 9.0, 0.0, 0.0]);
    for plane in 0..4 {
        assert_eq!(runtime.momenta_mut()[plane * point_stride + 2], 0.0);
    }
    assert_eq!(storage_identity(&mut runtime), identity);
    assert_eq!(
        runtime.activity_counters(),
        DirectRuntimeActivityCounters {
            momentum_fill_calls: 2,
            momentum_forms_filled: 2,
            momentum_terms_filled: 4,
            momentum_scalar_values_filled: 20,
            ..DirectRuntimeActivityCounters::default()
        }
    );

    let mut scalar_runtime = synthetic_runtime();
    let scalar_selector = scalar_runtime.prepare_replay_selector(0).unwrap();
    assert!(
        scalar_runtime
            .fill_momenta_from_external(&scalar_selector, 2, &external_two_point_momenta(),)
            .unwrap_err()
            .to_string()
            .contains("requires 4 Lorentz components")
    );
}

#[test]
fn prepared_replay_selectors_cover_both_physical_flows_without_regeneration() {
    let mut runtime = synthetic_runtime_with_lorentz(4);
    let flow_zero = runtime.prepare_replay_selector(0).unwrap();
    let flow_one = runtime.prepare_replay_selector(1).unwrap();
    assert_eq!(flow_zero.mapped_external_source_slot(0), Some(0));
    assert_eq!(flow_zero.mapped_external_source_slot(1), Some(1));
    assert_eq!(flow_one.mapped_external_source_slot(0), Some(1));
    assert_eq!(flow_one.mapped_external_source_slot(1), Some(0));
    assert_eq!(flow_zero.phase(), (1.0, 0.0));
    assert_eq!(flow_one.phase(), (0.0, 1.0));
    assert_eq!(flow_one.multiplicity(), 2);
    let identity = storage_identity(&mut runtime);

    let output = runtime
        .execute_replay_tile_from_external(&flow_zero, 2, &external_two_point_momenta())
        .unwrap();
    assert_eq!(output.public_flow_id(), Some(0));
    assert_eq!(output.representative_flow_id(), Some(0));
    assert_eq!(
        output.selected_destination_ids().collect::<Vec<_>>(),
        vec![0]
    );
    assert_eq!(output.destination_re(0).unwrap(), &[-27.0, -36.0]);
    assert_eq!(output.destination_im(0).unwrap(), &[-9.0, -12.0]);

    runtime
        .fill_momenta_from_external(&flow_zero, 2, &external_two_point_momenta())
        .unwrap();
    assert!(
        runtime
            .execute_replay_tile(&flow_one, 2)
            .unwrap_err()
            .to_string()
            .contains("were not filled for this selector")
    );

    let output = runtime
        .execute_replay_tile_from_external(&flow_one, 2, &external_two_point_momenta())
        .unwrap();
    assert_eq!(output.public_flow_id(), Some(1));
    assert_eq!(output.representative_flow_id(), Some(0));
    assert_eq!(output.destination_re(0).unwrap(), &[12.0, 18.0]);
    assert_eq!(output.destination_im(0).unwrap(), &[-36.0, -54.0]);
    assert_eq!(storage_identity(&mut runtime), identity);

    let direct = runtime.counters();
    assert_eq!(direct.source_calls, 2);
    assert_eq!(direct.contribution_calls, 2);
    assert_eq!(direct.finalization_calls, 2);
    assert_eq!(direct.closure_calls, 2);
    assert_eq!(direct.packed_input_bytes, 0);
    assert_eq!(direct.packed_output_bytes, 0);
    assert_eq!(direct.scatter_bytes, 0);
    let activity = runtime.activity_counters();
    assert_eq!(activity.momentum_fill_calls, 3);
    assert_eq!(activity.schedule_executions, 2);
    assert_eq!(activity.replay_schedule_executions, 2);
    assert_eq!(activity.replay_output_values_scaled, 4);
}

#[test]
fn replay_observation_captures_representative_amplitude_before_single_public_scaling() {
    let mut runtime = synthetic_runtime_with_lorentz(4);
    let selector = runtime.prepare_replay_selector(1).unwrap();
    begin_direct_current_observation(runtime.plan(), Some(selector.representative_flow_id()), 2)
        .unwrap();

    let output = runtime
        .execute_replay_tile_from_external(&selector, 2, &external_two_point_momenta())
        .unwrap();
    let public = output
        .destination_re(0)
        .unwrap()
        .iter()
        .copied()
        .zip(output.destination_im(0).unwrap().iter().copied())
        .collect::<Vec<_>>();
    let observation = take_direct_current_observation().unwrap();
    let representative = &observation.amplitudes_before_replay[&0];

    assert_eq!(representative, &[(-18.0, -6.0), (-27.0, -9.0)]);
    let phase = selector.phase();
    let multiplicity = f64::from(selector.multiplicity());
    let expected_public = representative
        .iter()
        .map(|&(value_re, value_im)| {
            let scale_re = phase.0 * multiplicity;
            let scale_im = phase.1 * multiplicity;
            (
                value_re * scale_re - value_im * scale_im,
                value_re * scale_im + value_im * scale_re,
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(public, expected_public);
    assert_eq!(public, [(12.0, -36.0), (18.0, -54.0)]);
}

#[test]
fn replay_selector_executes_only_its_dependency_closed_rows() {
    let (base_plan, _) = synthetic_plan_and_executors();
    let mut parts = base_plan.into_parts();
    parts.current_arena_components = 3;
    parts.selector_domains.push(DirectSelectorDomainDescriptor {
        word_start: 1,
        word_count: 1,
    });
    parts.selector_words.push(2);

    let mut inactive_current = parts.currents[0];
    inactive_current.semantic_current_id = 2;
    inactive_current.component_base = 2;
    inactive_current.selector_domain_id = 1;
    inactive_current.source_row_or_sentinel = 1;
    parts.currents.push(inactive_current);

    let mut inactive_source = parts.sources[0];
    inactive_source.destination_component_base = 2;
    inactive_source.selector_domain_id = 1;
    parts.sources.push(inactive_source);
    parts
        .row_groups
        .iter_mut()
        .find(|group| group.role == DirectExecutorRole::Source)
        .unwrap()
        .row_count = 2;

    let plan = DirectRecurrencePlan::new(parts).unwrap();
    let executors = DirectExecutorCatalog::new(
        &plan,
        plan.direct_template_catalog_digest(),
        direct_executor_handles(),
    )
    .unwrap();
    let mut runtime = DirectRecurrenceExecutionRuntime::new(plan, executors, 4).unwrap();
    runtime.set_parameters(&[3.0], &[1.0]).unwrap();
    let selector = runtime.prepare_replay_selector(0).unwrap();

    assert_eq!(selector.selected_row_group_count(), 4);
    assert_eq!(selector.selected_row_count(), 4);
    let output = runtime
        .execute_replay_tile_from_external(&selector, 2, &external_two_point_momenta())
        .unwrap();
    assert_eq!(output.destination_re(0).unwrap(), &[-27.0, -36.0]);
    assert_eq!(runtime.counters().source_rows, 1);
}

#[test]
fn tile_execution_clears_only_active_additive_regions() {
    let mut runtime = synthetic_runtime();
    runtime
        .momentum_plane_mut(0, 0)
        .unwrap()
        .copy_from_slice(&[2.0, 4.0, 100.0, 200.0]);
    runtime.execute_tile(4).unwrap();
    let point_stride = runtime.point_stride() as usize;
    let (prior_current_re_tail, prior_current_im_tail) = {
        let (current_re, current_im) = runtime.current_arenas();
        (
            current_re[point_stride + 2..point_stride + 4].to_vec(),
            current_im[point_stride + 2..point_stride + 4].to_vec(),
        )
    };
    let (prior_amplitude_re_tail, prior_amplitude_im_tail) = {
        let (amplitude_re, amplitude_im) = runtime.amplitude_arenas();
        (amplitude_re[2..4].to_vec(), amplitude_im[2..4].to_vec())
    };

    let output = runtime.execute_tile(2).unwrap();
    assert_eq!(output.destination_re(0).unwrap(), &[-6.0, -12.0]);
    assert_eq!(output.destination_im(0).unwrap(), &[-2.0, -4.0]);
    assert_eq!(&output.storage_re()[2..4], prior_amplitude_re_tail);
    assert_eq!(&output.storage_im()[2..4], prior_amplitude_im_tail);
    let (current_re, current_im) = runtime.current_arenas();
    assert_eq!(&current_re[point_stride..point_stride + 2], &[6.0, 12.0]);
    assert_eq!(
        &current_re[point_stride + 2..point_stride + 4],
        prior_current_re_tail
    );
    assert_eq!(
        &current_im[point_stride + 2..point_stride + 4],
        prior_current_im_tail
    );
}

#[test]
fn tile_bounds_fail_before_direct_execution() {
    let mut runtime = synthetic_runtime();
    assert!(
        runtime
            .execute_tile(0)
            .unwrap_err()
            .to_string()
            .contains("must be positive")
    );
    assert!(
        runtime
            .execute_tile(5)
            .unwrap_err()
            .to_string()
            .contains("exceeds point tile size 4")
    );
    assert!(
        runtime
            .set_parameters(&[], &[])
            .unwrap_err()
            .to_string()
            .contains("expected 1")
    );
    assert!(
        runtime
            .set_parameters(&[1.0], &[0.0, 0.0])
            .unwrap_err()
            .to_string()
            .contains("expected 1")
    );
    assert!(runtime.outputs().is_none());
    assert_eq!(runtime.counters(), DirectExecutionCounters::default());

    let (plan, executors) = synthetic_plan_and_executors();
    let error = DirectRecurrenceExecutionRuntime::new(plan, executors, 0)
        .err()
        .unwrap();
    assert!(error.to_string().contains("Lorentz component count"));
}

#[test]
fn runtime_parameter_storage_is_sized_from_the_authenticated_plan() {
    let (plan, executors) = synthetic_plan_and_executors();
    assert_eq!(plan.parameter_value_count(), 1);
    let mut runtime = DirectRecurrenceExecutionRuntime::new(plan, executors, 1).unwrap();
    let (parameters_re, parameters_im) = runtime.parameters_mut();
    assert_eq!(parameters_re.len(), 1);
    assert_eq!(parameters_im.len(), 1);
}

#[test]
fn runtime_clamps_the_effective_tile_to_workspace_and_rejects_an_oversized_point() {
    let mut parts = crate::recurrence::direct_plan::tests::valid_parts();
    parts.point_tile_size = 1024;
    parts.workspace_mib = 1;
    parts.current_arena_components = 256;
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    let executors = DirectExecutorCatalog::new(
        &plan,
        plan.direct_template_catalog_digest(),
        direct_executor_handles(),
    )
    .unwrap();
    let runtime = DirectRecurrenceExecutionRuntime::new(plan, executors, 4).unwrap();
    let per_point_bytes = (256 * 2 + 2 + 4) * std::mem::size_of::<f64>();
    let expected_tile = ((1024 * 1024 / per_point_bytes) / 8 * 8) as u32;
    assert_eq!(runtime.point_tile_size(), expected_tile);
    assert_eq!(runtime.point_stride(), expected_tile);
    assert!(runtime.point_tile_size() < 1024);

    let mut parts = crate::recurrence::direct_plan::tests::valid_parts();
    parts.point_tile_size = 1024;
    parts.workspace_mib = 1;
    parts.current_arena_components = 70_000;
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    let executors = DirectExecutorCatalog::new(
        &plan,
        plan.direct_template_catalog_digest(),
        direct_executor_handles(),
    )
    .unwrap();
    let error = DirectRecurrenceExecutionRuntime::new(plan, executors, 4)
        .err()
        .unwrap();
    assert!(error.to_string().contains("one point requires"));
}

#[test]
fn runtime_source_has_no_eager_or_scatter_execution_route() {
    let source = include_str!("direct_runtime.rs");
    for forbidden in [
        "EagerKernelInput",
        "EagerKernelBackend",
        "EagerKernelCall",
        "evaluate_batch",
        "Attachment",
    ] {
        assert!(
            !source.contains(forbidden),
            "direct runtime must not contain {forbidden}"
        );
    }
}
