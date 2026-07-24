// SPDX-License-Identifier: 0BSD

use rusticol_core::RusticolError;
use rusticol_core::recurrence::direct_backend::{
    DIRECT_STATUS_OK, DirectArenaView, DirectExecutorCatalog, DirectExecutorHandle,
    DirectFactorView, DirectMomentumView, DirectParameterView,
};
use rusticol_core::recurrence::direct_runtime::DirectRecurrenceExecutionRuntime;
use rusticol_core::recurrence::{
    DIRECT_NONE_U32, DirectAmplitudeDestinationDescriptor, DirectClosureRow, DirectContributionRow,
    DirectCurrentDescriptor, DirectDestinationOperation, DirectExecutorRole, DirectFinalizationRow,
    DirectMomentumFormDescriptor, DirectMomentumTerm, DirectNodeKind, DirectRecurrencePlan,
    DirectRecurrencePlanParts, DirectReplayTargetDescriptor, DirectResolvedHelicityDescriptor,
    DirectRowGroupDescriptor, DirectSelectorDomainDescriptor, DirectSourceRow,
    DirectSourceStateAssignment, ExactComplexRational, RecurrenceStrategy, SemanticDigest,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ffi::c_void;

thread_local! {
    static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
    static ALLOCATED_BYTES: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_allocation(new_size);
        unsafe { System.realloc(pointer, layout, new_size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

fn count_allocation(bytes: usize) {
    let tracking = TRACK_ALLOCATIONS.try_with(Cell::get).unwrap_or(false);
    if tracking {
        let _ = ALLOCATION_COUNT.try_with(|count| count.set(count.get() + 1));
        let _ = ALLOCATED_BYTES.try_with(|total| total.set(total.get().saturating_add(bytes)));
    }
}

fn count_allocations<T>(function: impl FnOnce() -> T) -> (T, usize, usize) {
    ALLOCATION_COUNT.with(|count| count.set(0));
    ALLOCATED_BYTES.with(|total| total.set(0));
    TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
    let result = function();
    TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
    let count = ALLOCATION_COUNT.with(Cell::get);
    let bytes = ALLOCATED_BYTES.with(Cell::get);
    (result, count, bytes)
}

const STATUS_BOUNDS: i32 = 2;
const POINT_COUNT: u32 = 4;
const EXPECTED_RE: [f64; 4] = [3.0, 6.0, 9.0, 12.0];
const EXPECTED_IM: [f64; 4] = [1.0, 2.0, 3.0, 4.0];

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

fn topology_replay_runtime() -> DirectRecurrenceExecutionRuntime {
    let catalog_digest = SemanticDigest::new([0x33; 32]).unwrap();
    let plan = DirectRecurrencePlan::new(DirectRecurrencePlanParts {
        strategy: RecurrenceStrategy::TopologyReplay,
        semantic_digest: SemanticDigest::new([0x11; 32]).unwrap(),
        prepared_pack_digest: SemanticDigest::new([0x22; 32]).unwrap(),
        direct_template_catalog_digest: catalog_digest,
        point_tile_size: POINT_COUNT,
        workspace_mib: 1,
        current_arena_components: 2,
        physical_sector_count: 1,
        retained_helicity_count: 1,
        amplitude_destination_count: 1,
        parameter_value_count: 1,
        external_source_count: 1,
        state_template_count: 2,
        source_template_count: 1,
        source_template_or_dispatch_count: 1,
        runtime_helicity_contract_count: 0,
        runtime_helicity_variant_count: 0,
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
            spin_state_class: -1,
            exact_factor_id: 0,
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
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        }],
        closures: vec![DirectClosureRow {
            parent0_component_base: 1,
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
    })
    .unwrap();

    let executors = DirectExecutorCatalog::new(
        &plan,
        catalog_digest,
        vec![
            DirectExecutorHandle::Source {
                call: fill_sources,
                context: std::ptr::null(),
            },
            DirectExecutorHandle::Contribution {
                call: accumulate_contributions,
                context: std::ptr::null(),
            },
            DirectExecutorHandle::Finalization {
                call: finalize_currents,
                context: std::ptr::null(),
            },
            DirectExecutorHandle::Closure {
                call: accumulate_closures,
                context: std::ptr::null(),
            },
        ],
    )
    .unwrap();
    let mut runtime = DirectRecurrenceExecutionRuntime::new(plan, executors, 4).unwrap();
    runtime.set_parameters(&[3.0], &[1.0]).unwrap();
    runtime
}

fn external_momenta() -> [f64; 16] {
    [
        1.0, 10.0, 100.0, 1000.0, 2.0, 20.0, 200.0, 2000.0, 3.0, 30.0, 300.0, 3000.0, 4.0, 40.0,
        400.0, 4000.0,
    ]
}

#[test]
fn warmed_topology_replay_tiles_allocate_zero_heap_bytes_and_remain_correct() {
    let mut runtime = topology_replay_runtime();
    let selector = runtime.prepare_replay_selector(0).unwrap();
    let momenta = external_momenta();

    let warm_output = runtime
        .execute_replay_tile_from_external(&selector, POINT_COUNT, &momenta)
        .unwrap();
    assert_eq!(warm_output.destination_re(0).unwrap(), EXPECTED_RE);
    assert_eq!(warm_output.destination_im(0).unwrap(), EXPECTED_IM);

    let (result, allocation_count, allocated_bytes) = count_allocations(|| {
        let mut observed_re = [0.0; POINT_COUNT as usize];
        let mut observed_im = [0.0; POINT_COUNT as usize];
        for _ in 0..32 {
            let output =
                runtime.execute_replay_tile_from_external(&selector, POINT_COUNT, &momenta)?;
            observed_re.copy_from_slice(output.destination_re(0).ok_or_else(|| {
                RusticolError::internal("missing direct recurrence real destination")
            })?);
            observed_im.copy_from_slice(output.destination_im(0).ok_or_else(|| {
                RusticolError::internal("missing direct recurrence imaginary destination")
            })?);
        }
        Ok::<_, RusticolError>((observed_re, observed_im))
    });

    let (observed_re, observed_im) = result.unwrap();
    assert_eq!(observed_re, EXPECTED_RE);
    assert_eq!(observed_im, EXPECTED_IM);
    assert_eq!(allocation_count, 0, "warmed recurrence loop allocated");
    assert_eq!(allocated_bytes, 0, "warmed recurrence loop allocated bytes");
}
