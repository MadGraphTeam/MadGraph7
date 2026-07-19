// SPDX-License-Identifier: 0BSD

use rusticol_core::{
    EagerAttachmentRow, EagerClosureRow, EagerComplex64, EagerCouplingRow, EagerDirectClosureSpec,
    EagerExecutionPlan, EagerExecutionRuntime, EagerFinalizationRow, EagerInvocationRow,
    EagerKernelBackend, EagerKernelCall, EagerKernelInput, EagerKernelRole, EagerKernelSpec,
    EagerPlanDefinition, EagerPlanDimensions, EagerPlanPayloads, EagerReductionEntry,
    EagerReductionGroup, EagerRuntimeOptions, EagerStagePayload, MISSING_U32, RusticolError,
    RusticolErrorKind, RusticolResult,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static TRACK_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
    static ALLOCATION_COUNT: Cell<usize> = const { Cell::new(0) };
}

struct CountingAllocator;

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAllocator = CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count_allocation();
        unsafe { System.realloc(pointer, layout, new_size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}

fn count_allocation() {
    let tracking = TRACK_ALLOCATIONS.try_with(Cell::get).unwrap_or(false);
    if tracking {
        let _ = ALLOCATION_COUNT.try_with(|count| count.set(count.get() + 1));
    }
}

fn count_allocations<T>(function: impl FnOnce() -> T) -> (T, usize) {
    ALLOCATION_COUNT.with(|count| count.set(0));
    TRACK_ALLOCATIONS.with(|tracking| tracking.set(true));
    let result = function();
    TRACK_ALLOCATIONS.with(|tracking| tracking.set(false));
    let count = ALLOCATION_COUNT.with(Cell::get);
    (result, count)
}

#[derive(Default)]
struct MockBackend {
    calls: [usize; 3],
    max_lanes: [usize; 3],
}

impl EagerKernelBackend for MockBackend {
    fn evaluate_batch(&mut self, call: EagerKernelCall<'_>) -> RusticolResult<()> {
        let kernel = usize::try_from(call.kernel_id)
            .map_err(|_| RusticolError::evaluation("mock kernel id does not fit usize"))?;
        if kernel >= self.calls.len() {
            return Err(RusticolError::evaluation("unknown mock eager kernel"));
        }
        self.calls[kernel] += 1;
        self.max_lanes[kernel] = self.max_lanes[kernel].max(call.lane_count);
        match call.kernel_id {
            0 => {
                assert_eq!(call.input_component_count, 7);
                assert_eq!(call.output_component_count, 1);
                for lane in 0..call.lane_count {
                    let row = lane * call.input_component_count;
                    let right = call.inputs[row];
                    let left = call.inputs[row + 1];
                    let left_momentum = call.inputs[row + 2];
                    let right_momentum = call.inputs[row + 3];
                    let coupling_real = call.inputs[row + 4];
                    let coupling_imag = call.inputs[row + 5];
                    let parameter = call.inputs[row + 6];
                    call.outputs[lane] = (left + right + left_momentum + right_momentum)
                        * coupling_real
                        + coupling_imag
                        + parameter;
                }
            }
            1 => {
                assert_eq!(call.input_component_count, 3);
                assert_eq!(call.output_component_count, 1);
                for lane in 0..call.lane_count {
                    let row = lane * call.input_component_count;
                    let momentum = call.inputs[row];
                    let current = call.inputs[row + 1];
                    let parameter = call.inputs[row + 2];
                    call.outputs[lane] = current * momentum + parameter;
                }
            }
            2 => {
                assert_eq!(call.input_component_count, 5);
                assert_eq!(call.output_component_count, 1);
                for lane in 0..call.lane_count {
                    let row = lane * call.input_component_count;
                    let coupling_imag = call.inputs[row];
                    let right = call.inputs[row + 1];
                    let coupling_real = call.inputs[row + 2];
                    let left = call.inputs[row + 3];
                    let parameter = call.inputs[row + 4];
                    call.outputs[lane] = left * right * coupling_real + coupling_imag + parameter;
                }
            }
            _ => unreachable!(),
        }
        Ok(())
    }
}

fn c64(real: f64) -> EagerComplex64 {
    EagerComplex64::new(real, 0.0)
}

fn definition() -> EagerPlanDefinition {
    EagerPlanDefinition {
        dimensions: EagerPlanDimensions {
            value_slot_component_counts: vec![1, 1, 1, 1, 1, 1],
            momentum_slot_component_counts: vec![1],
            current_component_counts: vec![1, 1],
            parameter_count: 1,
            amplitude_count: 2,
        },
        kernels: vec![
            EagerKernelSpec {
                kernel_id: 0,
                role: EagerKernelRole::Vertex,
                inputs: vec![
                    EagerKernelInput::SecondCurrentComponent(0),
                    EagerKernelInput::FirstCurrentComponent(0),
                    EagerKernelInput::FirstMomentumComponent(0),
                    EagerKernelInput::SecondMomentumComponent(0),
                    EagerKernelInput::CouplingReal,
                    EagerKernelInput::CouplingImag,
                    EagerKernelInput::ModelParameter(0),
                ],
                output_component_count: 1,
            },
            EagerKernelSpec {
                kernel_id: 1,
                role: EagerKernelRole::Finalization,
                inputs: vec![
                    EagerKernelInput::FirstMomentumComponent(0),
                    EagerKernelInput::FirstCurrentComponent(0),
                    EagerKernelInput::ModelParameter(0),
                ],
                output_component_count: 1,
            },
            EagerKernelSpec {
                kernel_id: 2,
                role: EagerKernelRole::Closure,
                inputs: vec![
                    EagerKernelInput::CouplingImag,
                    EagerKernelInput::SecondCurrentComponent(0),
                    EagerKernelInput::CouplingReal,
                    EagerKernelInput::FirstCurrentComponent(0),
                    EagerKernelInput::ModelParameter(0),
                ],
                output_component_count: 1,
            },
        ],
        direct_closures: vec![EagerDirectClosureSpec {
            closure_index: 1,
            coefficients: vec![c64(3.0)],
        }],
        reduction_groups: vec![
            EagerReductionGroup {
                amplitude_indices: vec![0],
            },
            EagerReductionGroup {
                amplitude_indices: vec![1],
            },
        ],
        reduction_entries: vec![
            EagerReductionEntry {
                left_group_index: 0,
                right_group_index: 0,
                coefficient: c64(1.0),
            },
            EagerReductionEntry {
                left_group_index: 1,
                right_group_index: 1,
                coefficient: c64(0.5),
            },
        ],
    }
}

fn invocation_rows() -> Vec<EagerInvocationRow> {
    vec![
        EagerInvocationRow {
            kernel_id: 0,
            left_value_slot_id: 0,
            right_value_slot_id: 1,
            left_momentum_slot_id: 0,
            right_momentum_slot_id: 0,
            coupling_slot_id: 0,
            attachment_start: 0,
            attachment_count: 1,
        },
        EagerInvocationRow {
            kernel_id: 0,
            left_value_slot_id: 1,
            right_value_slot_id: 0,
            left_momentum_slot_id: 0,
            right_momentum_slot_id: 0,
            coupling_slot_id: 0,
            attachment_start: 1,
            attachment_count: 1,
        },
    ]
}

fn attachment_rows() -> Vec<EagerAttachmentRow> {
    vec![
        EagerAttachmentRow {
            result_current_id: 0,
            factor_real: 1.0,
            factor_imag: 0.0,
        },
        EagerAttachmentRow {
            result_current_id: 1,
            factor_real: -1.0,
            factor_imag: 0.0,
        },
    ]
}

fn finalization_rows() -> Vec<EagerFinalizationRow> {
    vec![
        EagerFinalizationRow {
            kernel_id: 1,
            current_id: 0,
            unpropagated_value_slot_id: 2,
            propagated_value_slot_id: 3,
            momentum_slot_id: 0,
        },
        EagerFinalizationRow {
            kernel_id: 1,
            current_id: 1,
            unpropagated_value_slot_id: 4,
            propagated_value_slot_id: 5,
            momentum_slot_id: 0,
        },
    ]
}

fn closure_rows() -> Vec<EagerClosureRow> {
    vec![
        EagerClosureRow {
            kernel_id: 2,
            left_value_slot_id: 3,
            right_value_slot_id: 0,
            amplitude_index: 0,
            coupling_slot_id: 0,
            factor_real: 1.0,
            factor_imag: 0.0,
        },
        EagerClosureRow {
            kernel_id: MISSING_U32,
            left_value_slot_id: 4,
            right_value_slot_id: 1,
            amplitude_index: 1,
            coupling_slot_id: MISSING_U32,
            factor_real: 2.0,
            factor_imag: 0.0,
        },
    ]
}

fn build_plan_with(
    definition: EagerPlanDefinition,
    invocations: &[EagerInvocationRow],
    attachments: &[EagerAttachmentRow],
    finalizations: &[EagerFinalizationRow],
    closures: &[EagerClosureRow],
) -> RusticolResult<EagerExecutionPlan> {
    let coupling_bytes = EagerCouplingRow::encode_table(&[EagerCouplingRow {
        real_parameter_id: 0,
        imag_parameter_id: MISSING_U32,
        constant_real: 99.0,
        constant_imag: 0.5,
    }])?;
    let invocation_bytes = EagerInvocationRow::encode_table(invocations)?;
    let attachment_bytes = EagerAttachmentRow::encode_table(attachments)?;
    let finalization_bytes = EagerFinalizationRow::encode_table(finalizations)?;
    let closure_bytes = EagerClosureRow::encode_table(closures)?;
    let stage = EagerStagePayload {
        stage_index: 1,
        invocations: &invocation_bytes,
        attachments: &attachment_bytes,
        finalizations: &finalization_bytes,
    };
    EagerExecutionPlan::from_payloads(
        definition,
        EagerPlanPayloads {
            couplings: &coupling_bytes,
            stages: &[stage],
            closures: &closure_bytes,
        },
    )
}

fn build_runtime(options: EagerRuntimeOptions) -> EagerExecutionRuntime {
    let plan = build_plan_with(
        definition(),
        &invocation_rows(),
        &attachment_rows(),
        &finalization_rows(),
        &closure_rows(),
    )
    .unwrap();
    EagerExecutionRuntime::new(plan, options).unwrap()
}

fn inputs(point_count: usize) -> (Vec<EagerComplex64>, Vec<f64>) {
    let mut values = vec![c64(0.0); 6 * point_count];
    let mut momenta = vec![0.0; point_count];
    for point in 0..point_count {
        values[point] = c64(point as f64 + 1.0);
        values[point_count + point] = c64(10.0 + point as f64);
        momenta[point] = 0.5 + point as f64 * 0.25;
    }
    (values, momenta)
}

fn expected(point: usize, momenta: &[f64]) -> (EagerComplex64, EagerComplex64, f64) {
    let left = point as f64 + 1.0;
    let right = 10.0 + point as f64;
    let momentum = momenta[point];
    let current = (left + right + 2.0 * momentum) * 2.0 + 0.5 + 2.0;
    let propagated = current * momentum + 2.0;
    let amplitude_0 = propagated * left * 2.0 + 0.5 + 2.0;
    let amplitude_1 = -6.0 * current * right;
    let reduced = amplitude_0 * amplitude_0 + 0.5 * amplitude_1 * amplitude_1;
    (c64(amplitude_0), c64(amplitude_1), reduced)
}

#[test]
fn executes_packetized_stages_finalization_closures_and_reduction() {
    let point_count = 5;
    let mut runtime = build_runtime(EagerRuntimeOptions {
        point_tile_size: 2,
        workspace_bytes: 4096,
    });
    let (values, momenta) = inputs(point_count);
    let mut amplitudes = vec![c64(0.0); 2 * point_count];
    let mut reduced = vec![0.0; point_count];
    let mut backend = MockBackend::default();

    runtime
        .evaluate_into(
            &mut backend,
            point_count,
            &values,
            &momenta,
            &[c64(2.0)],
            &mut amplitudes,
            &mut reduced,
        )
        .unwrap();

    for point in 0..point_count {
        let (amplitude_0, amplitude_1, expected_reduced) = expected(point, &momenta);
        assert_eq!(amplitudes[point], amplitude_0);
        assert_eq!(amplitudes[point_count + point], amplitude_1);
        assert_eq!(reduced[point], expected_reduced);
    }
    assert_eq!(backend.calls, [3, 3, 3]);
    assert_eq!(backend.max_lanes, [4, 4, 2]);
    assert_eq!(runtime.effective_point_tile_size(), 2);
    assert_eq!(runtime.plan().invocation_count(), 2);
    assert_eq!(runtime.plan().attachment_count(), 2);
    assert_eq!(runtime.plan().closure_count(), 2);
    assert_eq!(runtime.plan().reduction_group_count(), 2);
    assert_eq!(runtime.plan().reduction_entry_count(), 2);
    assert_eq!(runtime.plan().stage_indices().collect::<Vec<_>>(), [1]);
}

#[test]
fn reduces_coherent_amplitudes_through_compact_groups() {
    let point_count = 3;
    let mut definition = definition();
    definition.reduction_groups = vec![EagerReductionGroup {
        amplitude_indices: vec![0, 1],
    }];
    definition.reduction_entries = vec![EagerReductionEntry {
        left_group_index: 0,
        right_group_index: 0,
        coefficient: c64(0.25),
    }];
    let plan = build_plan_with(
        definition,
        &invocation_rows(),
        &attachment_rows(),
        &finalization_rows(),
        &closure_rows(),
    )
    .unwrap();
    let mut runtime = EagerExecutionRuntime::new(
        plan,
        EagerRuntimeOptions {
            point_tile_size: 2,
            workspace_bytes: 4096,
        },
    )
    .unwrap();
    let (values, momenta) = inputs(point_count);
    let mut amplitudes = vec![c64(0.0); 2 * point_count];
    let mut reduced = vec![0.0; point_count];
    runtime
        .evaluate_into(
            &mut MockBackend::default(),
            point_count,
            &values,
            &momenta,
            &[c64(2.0)],
            &mut amplitudes,
            &mut reduced,
        )
        .unwrap();

    for point in 0..point_count {
        let (left, right, _) = expected(point, &momenta);
        assert_eq!(reduced[point], 0.25 * (left + right).norm_sqr());
    }
}

#[test]
fn workspace_budget_reduces_tile_and_splits_packets() {
    let runtime = build_runtime(EagerRuntimeOptions {
        point_tile_size: 8,
        workspace_bytes: 400,
    });
    assert_eq!(runtime.effective_point_tile_size(), 1);
    assert_eq!(runtime.packet_count(), 4);
    assert!(runtime.workspace_bytes() <= 400);

    let plan = build_plan_with(
        definition(),
        &invocation_rows(),
        &attachment_rows(),
        &finalization_rows(),
        &closure_rows(),
    )
    .unwrap();
    let error = EagerExecutionRuntime::new(
        plan,
        EagerRuntimeOptions {
            point_tile_size: 8,
            workspace_bytes: 250,
        },
    )
    .unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::InvalidArgument);
}

#[test]
fn warmed_evaluation_performs_no_allocations() {
    let point_count = 4;
    let mut runtime = build_runtime(EagerRuntimeOptions {
        point_tile_size: point_count,
        workspace_bytes: 4096,
    });
    let (values, momenta) = inputs(point_count);
    let parameters = [c64(2.0)];
    let mut amplitudes = vec![c64(0.0); 2 * point_count];
    let mut reduced = vec![0.0; point_count];
    let mut backend = MockBackend::default();
    runtime
        .evaluate_into(
            &mut backend,
            point_count,
            &values,
            &momenta,
            &parameters,
            &mut amplitudes,
            &mut reduced,
        )
        .unwrap();

    let (result, allocation_count) = count_allocations(|| {
        runtime.evaluate_into(
            &mut backend,
            point_count,
            &values,
            &momenta,
            &parameters,
            &mut amplitudes,
            &mut reduced,
        )
    });
    result.unwrap();
    assert_eq!(allocation_count, 0);
}

#[test]
fn malformed_attachment_ranges_are_artifact_errors() {
    let mut invocations = invocation_rows();
    invocations[0].attachment_start = u64::MAX;
    let error = build_plan_with(
        definition(),
        &invocations,
        &attachment_rows(),
        &finalization_rows(),
        &closure_rows(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("attachment"));
}

#[test]
fn unknown_attachment_targets_are_artifact_errors() {
    let mut attachments = attachment_rows();
    attachments[1].result_current_id = 99;
    let error = build_plan_with(
        definition(),
        &invocation_rows(),
        &attachments,
        &finalization_rows(),
        &closure_rows(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("unknown slot"));
}

#[test]
fn duplicate_finalization_is_an_artifact_error() {
    let mut finalizations = finalization_rows();
    finalizations[1].current_id = 0;
    let error = build_plan_with(
        definition(),
        &invocation_rows(),
        &attachment_rows(),
        &finalizations,
        &closure_rows(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("more than once"));
}

#[test]
fn kernel_roles_and_finalization_outputs_are_validated() {
    let mut wrong_role = definition();
    wrong_role.kernels[0].role = EagerKernelRole::Closure;
    let error = build_plan_with(
        wrong_role,
        &invocation_rows(),
        &attachment_rows(),
        &finalization_rows(),
        &closure_rows(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("expected Vertex"));

    let mut aliased = finalization_rows();
    aliased[0].propagated_value_slot_id = aliased[0].unpropagated_value_slot_id;
    let error = build_plan_with(
        definition(),
        &invocation_rows(),
        &attachment_rows(),
        &aliased,
        &closure_rows(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("aliases"));
}

#[test]
fn direct_closure_requires_exact_coefficients() {
    let mut missing = definition();
    missing.direct_closures.clear();
    let error = build_plan_with(
        missing,
        &invocation_rows(),
        &attachment_rows(),
        &finalization_rows(),
        &closure_rows(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("lacks contraction coefficients"));

    let mut wrong_width = definition();
    wrong_width.direct_closures[0].coefficients.push(c64(1.0));
    let error = build_plan_with(
        wrong_width,
        &invocation_rows(),
        &attachment_rows(),
        &finalization_rows(),
        &closure_rows(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("component widths"));
}

#[test]
fn kernel_input_components_are_validated_against_invocation_slots() {
    let mut definition = definition();
    definition.kernels[0].inputs[0] = EagerKernelInput::SecondCurrentComponent(1);
    let error = build_plan_with(
        definition,
        &invocation_rows(),
        &attachment_rows(),
        &finalization_rows(),
        &closure_rows(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("SecondCurrentComponent(1)"));
    assert!(error.to_string().contains("outside 0..1"));
}

#[test]
fn kernel_input_roles_reject_unavailable_sources() {
    let mut definition = definition();
    definition.kernels[1]
        .inputs
        .push(EagerKernelInput::CouplingReal);
    let error = build_plan_with(
        definition,
        &invocation_rows(),
        &attachment_rows(),
        &finalization_rows(),
        &closure_rows(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("Finalization"));
    assert!(error.to_string().contains("CouplingReal"));
}

#[test]
fn kernel_input_descriptors_are_unique_and_parameter_bounded() {
    let mut duplicate = definition();
    duplicate.kernels[2]
        .inputs
        .push(EagerKernelInput::CouplingImag);
    let error = build_plan_with(
        duplicate,
        &invocation_rows(),
        &attachment_rows(),
        &finalization_rows(),
        &closure_rows(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("repeats input descriptor"));

    let mut out_of_bounds = definition();
    out_of_bounds.kernels[0]
        .inputs
        .push(EagerKernelInput::ModelParameter(1));
    let error = build_plan_with(
        out_of_bounds,
        &invocation_rows(),
        &attachment_rows(),
        &finalization_rows(),
        &closure_rows(),
    )
    .unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("model parameter"));
    assert!(error.to_string().contains("outside 0..1"));
}

#[test]
fn invalid_execution_buffers_return_errors_without_panicking() {
    let mut runtime = build_runtime(EagerRuntimeOptions {
        point_tile_size: 2,
        workspace_bytes: 4096,
    });
    let mut backend = MockBackend::default();
    let mut amplitudes = vec![c64(0.0); 2];
    let mut reduced = vec![0.0];
    let error = runtime
        .evaluate_into(
            &mut backend,
            1,
            &[],
            &[1.0],
            &[c64(2.0)],
            &mut amplitudes,
            &mut reduced,
        )
        .unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::InvalidArgument);
    assert!(error.to_string().contains("initial value buffer"));
}
