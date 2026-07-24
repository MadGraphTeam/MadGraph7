// SPDX-License-Identifier: 0BSD

use crate::recurrence::direct_backend::{
    DIRECT_STATUS_OK, DirectArenaView, DirectContributionExecutor, DirectExecutionCounters,
    DirectExecutorCatalog, DirectExecutorHandle, DirectFactorView, DirectFinalizationExecutor,
    DirectMomentumView, DirectParameterView, DirectSourceExecutor,
};
use crate::recurrence::direct_plan::{
    DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION, DirectClosureRow, DirectContributionRow,
    DirectFinalizationRow, DirectMomentumTerm, DirectRecurrencePlan, DirectReplayTargetDescriptor,
    DirectSourceRow, DirectSourceStateAssignment,
};
use crate::recurrence::direct_runtime::{
    DIRECT_RUNTIME_ARENA_ALIGNMENT, DirectRecurrenceExecutionRuntime, DirectRuntimeActivityCounters,
};
use crate::recurrence::exact::{ExactComplexRational, ExactRational};
#[allow(unused_imports)]
use crate::recurrence::{
    DIRECT_NONE_U32, DirectAmplitudeDestinationDescriptor, DirectCurrentDescriptor,
    DirectDestinationOperation, DirectExecutorRole, DirectMomentumFormDescriptor, DirectNodeKind,
    DirectResolvedHelicityDescriptor, DirectRowGroupDescriptor, DirectSelectorDomainDescriptor,
    RecurrenceStrategy, SemanticDigest,
};
use std::ffi::c_void;

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
            unsafe {
                *arena.current_re.add(destination) += source_re * scale_re - source_im * scale_im;
                *arena.current_im.add(destination) += source_re * scale_im + source_im * scale_re;
            }
        }
    }
    DIRECT_STATUS_OK
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
                phase_exact_factor_id: 3,
                multiplicity: 1,
                selector_domain_id: 0,
            },
            DirectReplayTargetDescriptor {
                public_flow_id: 1,
                representative_id: 0,
                source_permutation_start: 2,
                source_permutation_count: 2,
                phase_exact_factor_id: 4,
                multiplicity: 2,
                selector_domain_id: 0,
            },
        ],
        source_permutations: vec![0, 1, 1, 0],
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
            phase_exact_factor_id: 3,
            multiplicity: 1,
            selector_domain_id: 0,
        },
        DirectReplayTargetDescriptor {
            public_flow_id: 1,
            representative_id: 0,
            source_permutation_start: 2,
            source_permutation_count: 2,
            phase_exact_factor_id: 4,
            multiplicity: 2,
            selector_domain_id: 0,
        },
    ];
    parts.source_permutations = vec![0, 1, 1, 0];
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

#[test]
fn low_footprint_runtime_retains_the_requested_point_tile() {
    let (plan, executors) = synthetic_plan_and_executors();
    assert_eq!(plan.point_tile_size(), 4);
    let runtime = DirectRecurrenceExecutionRuntime::new(plan, executors, 1).unwrap();
    assert_eq!(runtime.point_tile_size(), 4);
}

#[test]
fn high_footprint_runtime_uses_a_power_of_two_cache_tile() {
    let mut parts = crate::recurrence::direct_plan::tests::valid_parts();
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
    parts.workspace_mib = 8;
    parts.current_arena_components = 262_144;
    let plan = DirectRecurrencePlan::new(parts).unwrap();
    let executors = DirectExecutorCatalog::new(
        &plan,
        plan.direct_template_catalog_digest(),
        direct_executor_handles(),
    )
    .unwrap();
    let runtime = DirectRecurrenceExecutionRuntime::new(plan, executors, 4).unwrap();
    assert_eq!(runtime.point_tile_size(), 1);
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
    assert_eq!(&current_re[4..8], &[3.0, 6.0, 9.0, 12.0]);
    assert_eq!(&current_im[0..4], &[0.0; 4]);
    assert_eq!(&current_im[4..8], &[1.0, 2.0, 3.0, 4.0]);
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
    assert_eq!(&runtime.momenta_mut()[0..4], &[9.0, 12.0, 23.0, 0.0]);
    assert_eq!(&runtime.momenta_mut()[4..8], &[90.0, 120.0, 230.0, 0.0]);
    assert_eq!(&runtime.momenta_mut()[8..12], &[900.0, 1200.0, 2300.0, 0.0]);
    assert_eq!(
        &runtime.momenta_mut()[12..16],
        &[9000.0, 12000.0, 23000.0, 0.0]
    );

    runtime
        .fill_momenta_from_external(&flow_one, 2, &external_two_point_momenta())
        .unwrap();
    assert_eq!(&runtime.momenta_mut()[0..4], &[6.0, 9.0, 0.0, 0.0]);
    for plane in 0..4 {
        assert_eq!(runtime.momenta_mut()[plane * 4 + 2], 0.0);
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
fn tile_execution_clears_only_active_additive_regions() {
    let mut runtime = synthetic_runtime();
    runtime
        .momentum_plane_mut(0, 0)
        .unwrap()
        .copy_from_slice(&[2.0, 4.0, 100.0, 200.0]);
    runtime.execute_tile(4).unwrap();
    let (prior_current_re_tail, prior_current_im_tail) = {
        let (current_re, current_im) = runtime.current_arenas();
        (current_re[6..8].to_vec(), current_im[6..8].to_vec())
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
    assert_eq!(&current_re[4..6], &[6.0, 12.0]);
    assert_eq!(&current_re[6..8], prior_current_re_tail);
    assert_eq!(&current_im[6..8], prior_current_im_tail);
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
    let per_point_bytes = (256 * 2 + 1 * 2 + 1 * 4) * std::mem::size_of::<f64>();
    let expected_tile = (1024 * 1024 / per_point_bytes) as u32;
    assert_eq!(runtime.point_tile_size(), expected_tile);
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
