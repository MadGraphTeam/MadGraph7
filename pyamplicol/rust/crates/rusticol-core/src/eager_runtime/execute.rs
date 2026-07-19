// SPDX-License-Identifier: 0BSD

use super::plan::{
    ComponentRange, EagerExecutionPlan, EagerStagePlan, ScheduledAttachment, ScheduledClosure,
    ScheduledDirectClosure, ScheduledFinalization, ScheduledInvocation,
};
use super::runtime::{EagerWorkspace, KernelPacket, PacketRole, StageSchedule};
use super::{
    EagerComplex64, EagerKernelBackend, EagerKernelCall, EagerKernelInput, EagerKernelSpec,
};
use crate::{RusticolError, RusticolResult};
use std::collections::BTreeMap;

pub(super) fn initialize_tile(
    plan: &EagerExecutionPlan,
    workspace: &mut EagerWorkspace,
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    initial_values: &[EagerComplex64],
) {
    workspace.values.fill(EagerComplex64::new(0.0, 0.0));
    workspace.currents.fill(EagerComplex64::new(0.0, 0.0));
    workspace.amplitudes.fill(EagerComplex64::new(0.0, 0.0));
    workspace
        .reduction_groups
        .fill(EagerComplex64::new(0.0, 0.0));
    workspace.reduced.fill(0.0);
    for component in 0..plan.values.component_count {
        let source = component * point_count + tile_start;
        let target = component * workspace.tile_capacity;
        workspace.values[target..target + tile_points]
            .copy_from_slice(&initial_values[source..source + tile_points]);
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn execute_stage<B: EagerKernelBackend>(
    stage: &EagerStagePlan,
    schedule: &StageSchedule,
    kernels: &BTreeMap<u32, EagerKernelSpec>,
    workspace: &mut EagerWorkspace,
    backend: &mut B,
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    momenta: &[f64],
    model_parameters: &[EagerComplex64],
) -> RusticolResult<()> {
    for packet in &schedule.invocation_packets {
        debug_assert_eq!(packet.role, PacketRole::Invocation);
        let items = &stage.invocations[packet.item_range.clone()];
        let lane_count = items.len() * tile_points;
        let input_len = packet.input_components * lane_count;
        let output_len = packet.output_components * lane_count;
        let (inputs, outputs) = packet_slices(&mut workspace.packet, input_len, output_len)?;
        let kernel = kernels.get(&packet.kernel_id).ok_or_else(|| {
            RusticolError::internal(format!(
                "eager schedule lost invocation kernel {}",
                packet.kernel_id
            ))
        })?;
        gather_invocations(
            items,
            &kernel.inputs,
            inputs,
            lane_count,
            point_count,
            tile_start,
            tile_points,
            workspace.tile_capacity,
            &workspace.values,
            momenta,
            &workspace.couplings,
            model_parameters,
        );
        backend.evaluate_batch(EagerKernelCall {
            kernel_id: packet.kernel_id,
            lane_count,
            input_component_count: packet.input_components,
            output_component_count: packet.output_components,
            inputs,
            outputs,
        })?;
        scatter_invocations(
            items,
            &stage.attachments,
            outputs,
            lane_count,
            tile_points,
            workspace.tile_capacity,
            &mut workspace.currents,
        );
    }

    for item in &stage.finalization_copies {
        copy_component_range(
            &workspace.currents,
            item.current,
            &mut workspace.values,
            item.unpropagated,
            workspace.tile_capacity,
            tile_points,
        );
    }
    for packet in &schedule.finalization_packets {
        debug_assert_eq!(packet.role, PacketRole::Finalization);
        let items = &stage.finalizations[packet.item_range.clone()];
        let lane_count = items.len() * tile_points;
        let input_len = packet.input_components * lane_count;
        let output_len = packet.output_components * lane_count;
        let (inputs, outputs) = packet_slices(&mut workspace.packet, input_len, output_len)?;
        let kernel = kernels.get(&packet.kernel_id).ok_or_else(|| {
            RusticolError::internal(format!(
                "eager schedule lost finalization kernel {}",
                packet.kernel_id
            ))
        })?;
        gather_finalizations(
            items,
            &kernel.inputs,
            inputs,
            lane_count,
            point_count,
            tile_start,
            tile_points,
            workspace.tile_capacity,
            &workspace.currents,
            momenta,
            model_parameters,
        );
        backend.evaluate_batch(EagerKernelCall {
            kernel_id: packet.kernel_id,
            lane_count,
            input_component_count: packet.input_components,
            output_component_count: packet.output_components,
            inputs,
            outputs,
        })?;
        scatter_finalizations(
            items,
            outputs,
            lane_count,
            tile_points,
            workspace.tile_capacity,
            &mut workspace.values,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gather_invocations(
    items: &[ScheduledInvocation],
    descriptors: &[EagerKernelInput],
    inputs: &mut [EagerComplex64],
    lane_count: usize,
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    tile_capacity: usize,
    values: &[EagerComplex64],
    momenta: &[f64],
    couplings: &[EagerComplex64],
    model_parameters: &[EagerComplex64],
) {
    for (item_index, item) in items.iter().enumerate() {
        let lane_start = item_index * tile_points;
        let coupling = couplings[item.row.coupling_slot_id as usize];
        for (target_component, descriptor) in descriptors.iter().copied().enumerate() {
            match descriptor {
                EagerKernelInput::FirstCurrentComponent(component) => gather_complex_component(
                    values,
                    item.left_values,
                    component,
                    tile_capacity,
                    0,
                    inputs,
                    target_component,
                    lane_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::SecondCurrentComponent(component) => gather_complex_component(
                    values,
                    item.right_values,
                    component,
                    tile_capacity,
                    0,
                    inputs,
                    target_component,
                    lane_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::FirstMomentumComponent(component) => gather_real_component(
                    momenta,
                    item.left_momenta,
                    component,
                    point_count,
                    tile_start,
                    inputs,
                    target_component,
                    lane_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::SecondMomentumComponent(component) => gather_real_component(
                    momenta,
                    item.right_momenta,
                    component,
                    point_count,
                    tile_start,
                    inputs,
                    target_component,
                    lane_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::CouplingReal => fill_packet_component(
                    inputs,
                    target_component,
                    lane_count,
                    lane_start,
                    tile_points,
                    EagerComplex64::new(coupling.re, 0.0),
                ),
                EagerKernelInput::CouplingImag => fill_packet_component(
                    inputs,
                    target_component,
                    lane_count,
                    lane_start,
                    tile_points,
                    EagerComplex64::new(coupling.im, 0.0),
                ),
                EagerKernelInput::ModelParameter(parameter) => fill_packet_component(
                    inputs,
                    target_component,
                    lane_count,
                    lane_start,
                    tile_points,
                    model_parameters[parameter as usize],
                ),
            }
        }
    }
}

fn scatter_invocations(
    items: &[ScheduledInvocation],
    attachments: &[ScheduledAttachment],
    outputs: &[EagerComplex64],
    lane_count: usize,
    tile_points: usize,
    tile_capacity: usize,
    currents: &mut [EagerComplex64],
) {
    let output_components = outputs.len() / lane_count;
    for (item_index, item) in items.iter().enumerate() {
        let lane_start = item_index * tile_points;
        for attachment in &attachments[item.attachment_range.clone()] {
            let factor =
                EagerComplex64::new(attachment.row.factor_real, attachment.row.factor_imag);
            debug_assert_eq!(attachment.current.len, output_components);
            for component in 0..output_components {
                let source = component * lane_count + lane_start;
                let target = (attachment.current.start + component) * tile_capacity;
                for point in 0..tile_points {
                    currents[target + point] += factor * outputs[source + point];
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn gather_finalizations(
    items: &[ScheduledFinalization],
    descriptors: &[EagerKernelInput],
    inputs: &mut [EagerComplex64],
    lane_count: usize,
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    tile_capacity: usize,
    currents: &[EagerComplex64],
    momenta: &[f64],
    model_parameters: &[EagerComplex64],
) {
    for (item_index, item) in items.iter().enumerate() {
        let lane_start = item_index * tile_points;
        for (target_component, descriptor) in descriptors.iter().copied().enumerate() {
            match descriptor {
                EagerKernelInput::FirstCurrentComponent(component) => gather_complex_component(
                    currents,
                    item.current,
                    component,
                    tile_capacity,
                    0,
                    inputs,
                    target_component,
                    lane_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::FirstMomentumComponent(component) => gather_real_component(
                    momenta,
                    item.momentum,
                    component,
                    point_count,
                    tile_start,
                    inputs,
                    target_component,
                    lane_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::ModelParameter(parameter) => fill_packet_component(
                    inputs,
                    target_component,
                    lane_count,
                    lane_start,
                    tile_points,
                    model_parameters[parameter as usize],
                ),
                _ => unreachable!("validated eager finalization descriptor"),
            }
        }
    }
}

fn scatter_finalizations(
    items: &[ScheduledFinalization],
    outputs: &[EagerComplex64],
    lane_count: usize,
    tile_points: usize,
    tile_capacity: usize,
    values: &mut [EagerComplex64],
) -> RusticolResult<()> {
    let output_components = outputs.len() / lane_count;
    for (item_index, item) in items.iter().enumerate() {
        let lane_start = item_index * tile_points;
        let Some(target) = item.propagated else {
            return Err(RusticolError::internal(
                "eager finalization schedule lost its propagated output",
            ));
        };
        debug_assert_eq!(target.len, output_components);
        for component in 0..output_components {
            let source = component * lane_count + lane_start;
            let target_start = (target.start + component) * tile_capacity;
            values[target_start..target_start + tile_points]
                .copy_from_slice(&outputs[source..source + tile_points]);
        }
    }
    Ok(())
}

pub(super) fn execute_closures<B: EagerKernelBackend>(
    plan: &EagerExecutionPlan,
    packets: &[KernelPacket],
    workspace: &mut EagerWorkspace,
    backend: &mut B,
    tile_points: usize,
    model_parameters: &[EagerComplex64],
) -> RusticolResult<()> {
    for packet in packets {
        debug_assert_eq!(packet.role, PacketRole::Closure);
        let items = &plan.closures[packet.item_range.clone()];
        let lane_count = items.len() * tile_points;
        let input_len = packet.input_components * lane_count;
        let output_len = packet.output_components * lane_count;
        let (inputs, outputs) = packet_slices(&mut workspace.packet, input_len, output_len)?;
        let kernel = plan.kernels.get(&packet.kernel_id).ok_or_else(|| {
            RusticolError::internal(format!(
                "eager schedule lost closure kernel {}",
                packet.kernel_id
            ))
        })?;
        gather_closures(
            items,
            &kernel.inputs,
            inputs,
            lane_count,
            tile_points,
            workspace.tile_capacity,
            &workspace.values,
            &workspace.couplings,
            model_parameters,
        );
        backend.evaluate_batch(EagerKernelCall {
            kernel_id: packet.kernel_id,
            lane_count,
            input_component_count: packet.input_components,
            output_component_count: packet.output_components,
            inputs,
            outputs,
        })?;
        scatter_closures(
            items,
            outputs,
            tile_points,
            workspace.tile_capacity,
            &mut workspace.amplitudes,
        );
    }
    execute_direct_closures(
        &plan.direct_closures,
        &workspace.values,
        workspace.tile_capacity,
        tile_points,
        &mut workspace.amplitudes,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn gather_closures(
    items: &[ScheduledClosure],
    descriptors: &[EagerKernelInput],
    inputs: &mut [EagerComplex64],
    lane_count: usize,
    tile_points: usize,
    tile_capacity: usize,
    values: &[EagerComplex64],
    couplings: &[EagerComplex64],
    model_parameters: &[EagerComplex64],
) {
    for (item_index, item) in items.iter().enumerate() {
        let lane_start = item_index * tile_points;
        let coupling = couplings[item.row.coupling_slot_id as usize];
        for (target_component, descriptor) in descriptors.iter().copied().enumerate() {
            match descriptor {
                EagerKernelInput::FirstCurrentComponent(component) => gather_complex_component(
                    values,
                    item.left_values,
                    component,
                    tile_capacity,
                    0,
                    inputs,
                    target_component,
                    lane_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::SecondCurrentComponent(component) => gather_complex_component(
                    values,
                    item.right_values,
                    component,
                    tile_capacity,
                    0,
                    inputs,
                    target_component,
                    lane_count,
                    lane_start,
                    tile_points,
                ),
                EagerKernelInput::CouplingReal => fill_packet_component(
                    inputs,
                    target_component,
                    lane_count,
                    lane_start,
                    tile_points,
                    EagerComplex64::new(coupling.re, 0.0),
                ),
                EagerKernelInput::CouplingImag => fill_packet_component(
                    inputs,
                    target_component,
                    lane_count,
                    lane_start,
                    tile_points,
                    EagerComplex64::new(coupling.im, 0.0),
                ),
                EagerKernelInput::ModelParameter(parameter) => fill_packet_component(
                    inputs,
                    target_component,
                    lane_count,
                    lane_start,
                    tile_points,
                    model_parameters[parameter as usize],
                ),
                _ => unreachable!("validated eager closure descriptor"),
            }
        }
    }
}

fn scatter_closures(
    items: &[ScheduledClosure],
    outputs: &[EagerComplex64],
    tile_points: usize,
    tile_capacity: usize,
    amplitudes: &mut [EagerComplex64],
) {
    for (item_index, item) in items.iter().enumerate() {
        let lane_start = item_index * tile_points;
        let factor = EagerComplex64::new(item.row.factor_real, item.row.factor_imag);
        let target = item.row.amplitude_index as usize * tile_capacity;
        for point in 0..tile_points {
            amplitudes[target + point] += factor * outputs[lane_start + point];
        }
    }
}

fn execute_direct_closures(
    closures: &[ScheduledDirectClosure],
    values: &[EagerComplex64],
    tile_capacity: usize,
    tile_points: usize,
    amplitudes: &mut [EagerComplex64],
) {
    for closure in closures {
        let factor = EagerComplex64::new(closure.row.factor_real, closure.row.factor_imag);
        let target = closure.row.amplitude_index as usize * tile_capacity;
        for point in 0..tile_points {
            let mut contraction = EagerComplex64::new(0.0, 0.0);
            for (component, coefficient) in closure.coefficients.iter().enumerate() {
                let left = values[(closure.left_values.start + component) * tile_capacity + point];
                let right =
                    values[(closure.right_values.start + component) * tile_capacity + point];
                contraction += *coefficient * left * right;
            }
            amplitudes[target + point] += factor * contraction;
        }
    }
}

pub(super) fn reduce_tile(
    plan: &EagerExecutionPlan,
    workspace: &mut EagerWorkspace,
    tile_points: usize,
) {
    for (group_index, group) in plan.reduction_groups.iter().enumerate() {
        let target = group_index * workspace.tile_capacity;
        for amplitude_index in &group.amplitude_indices {
            let source = *amplitude_index as usize * workspace.tile_capacity;
            for point in 0..tile_points {
                workspace.reduction_groups[target + point] += workspace.amplitudes[source + point];
            }
        }
    }
    for entry in &plan.reduction_entries {
        let left = entry.left_group_index as usize * workspace.tile_capacity;
        let right = entry.right_group_index as usize * workspace.tile_capacity;
        for point in 0..tile_points {
            let product = workspace.reduction_groups[left + point]
                * workspace.reduction_groups[right + point].conj();
            workspace.reduced[point] += (entry.coefficient * product).re;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn copy_tile_results(
    plan: &EagerExecutionPlan,
    workspace: &EagerWorkspace,
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
    amplitudes: &mut [EagerComplex64],
    reduced: &mut [f64],
) {
    for amplitude in 0..plan.amplitude_count {
        let source = amplitude * workspace.tile_capacity;
        let target = amplitude * point_count + tile_start;
        amplitudes[target..target + tile_points]
            .copy_from_slice(&workspace.amplitudes[source..source + tile_points]);
    }
    reduced[tile_start..tile_start + tile_points]
        .copy_from_slice(&workspace.reduced[..tile_points]);
}

fn packet_slices(
    packet: &mut [EagerComplex64],
    input_len: usize,
    output_len: usize,
) -> RusticolResult<(&mut [EagerComplex64], &mut [EagerComplex64])> {
    let total = input_len
        .checked_add(output_len)
        .ok_or_else(|| RusticolError::internal("eager active packet size overflows"))?;
    if total > packet.len() {
        return Err(RusticolError::internal(format!(
            "eager active packet needs {total} elements, workspace has {}",
            packet.len()
        )));
    }
    let active = &mut packet[..total];
    Ok(active.split_at_mut(input_len))
}

#[allow(clippy::too_many_arguments)]
fn gather_complex_component(
    source: &[EagerComplex64],
    range: ComponentRange,
    component: u32,
    source_stride: usize,
    source_point_start: usize,
    target: &mut [EagerComplex64],
    target_component: usize,
    target_stride: usize,
    target_point_start: usize,
    point_count: usize,
) {
    let component = component as usize;
    debug_assert!(component < range.len);
    let source_start = (range.start + component) * source_stride + source_point_start;
    let target_start = target_component * target_stride + target_point_start;
    target[target_start..target_start + point_count]
        .copy_from_slice(&source[source_start..source_start + point_count]);
}

#[allow(clippy::too_many_arguments)]
fn gather_real_component(
    source: &[f64],
    range: ComponentRange,
    component: u32,
    source_stride: usize,
    source_point_start: usize,
    target: &mut [EagerComplex64],
    target_component: usize,
    target_stride: usize,
    target_point_start: usize,
    point_count: usize,
) {
    let component = component as usize;
    debug_assert!(component < range.len);
    let source_start = (range.start + component) * source_stride + source_point_start;
    let target_start = target_component * target_stride + target_point_start;
    for point in 0..point_count {
        target[target_start + point] = EagerComplex64::new(source[source_start + point], 0.0);
    }
}

fn fill_packet_component(
    target: &mut [EagerComplex64],
    target_component: usize,
    target_stride: usize,
    target_point_start: usize,
    point_count: usize,
    value: EagerComplex64,
) {
    let start = target_component * target_stride + target_point_start;
    target[start..start + point_count].fill(value);
}

fn copy_component_range(
    source: &[EagerComplex64],
    source_range: ComponentRange,
    target: &mut [EagerComplex64],
    target_range: ComponentRange,
    stride: usize,
    point_count: usize,
) {
    debug_assert_eq!(source_range.len, target_range.len);
    for component in 0..source_range.len {
        let source_start = (source_range.start + component) * stride;
        let target_start = (target_range.start + component) * stride;
        target[target_start..target_start + point_count]
            .copy_from_slice(&source[source_start..source_start + point_count]);
    }
}
