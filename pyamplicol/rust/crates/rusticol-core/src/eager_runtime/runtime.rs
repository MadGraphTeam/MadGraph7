// SPDX-License-Identifier: 0BSD

use super::execute::{
    copy_tile_results, execute_closures, execute_stage, initialize_tile, reduce_tile,
};
use super::plan::{EagerExecutionPlan, complex_is_finite};
use super::{EagerComplex64, EagerKernelBackend, EagerKernelSpec, EagerRuntimeOptions};
use crate::{EagerCouplingRow, MISSING_U32, RusticolError, RusticolResult};
use std::cmp::min;
use std::collections::BTreeMap;
use std::mem::size_of;
use std::ops::Range;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PacketRole {
    Invocation,
    Finalization,
    Closure,
}

#[derive(Clone, Debug)]
pub(super) struct KernelPacket {
    pub(super) role: PacketRole,
    pub(super) kernel_id: u32,
    pub(super) item_range: Range<usize>,
    pub(super) input_components: usize,
    pub(super) output_components: usize,
}

#[derive(Clone, Debug)]
pub(super) struct StageSchedule {
    pub(super) invocation_packets: Vec<KernelPacket>,
    pub(super) finalization_packets: Vec<KernelPacket>,
}

#[derive(Clone, Debug)]
struct ExecutionSchedule {
    stages: Vec<StageSchedule>,
    closure_packets: Vec<KernelPacket>,
    packet_buffer_len: usize,
}

#[derive(Debug)]
pub(super) struct EagerWorkspace {
    pub(super) tile_capacity: usize,
    pub(super) values: Vec<EagerComplex64>,
    pub(super) currents: Vec<EagerComplex64>,
    pub(super) amplitudes: Vec<EagerComplex64>,
    pub(super) reduction_groups: Vec<EagerComplex64>,
    pub(super) couplings: Vec<EagerComplex64>,
    pub(super) reduced: Vec<f64>,
    pub(super) packet: Vec<EagerComplex64>,
}

#[derive(Debug)]
pub struct EagerExecutionRuntime {
    plan: EagerExecutionPlan,
    schedule: ExecutionSchedule,
    workspace: EagerWorkspace,
    workspace_bytes: usize,
}

impl EagerExecutionRuntime {
    pub fn new(plan: EagerExecutionPlan, options: EagerRuntimeOptions) -> RusticolResult<Self> {
        if options.point_tile_size == 0 {
            return Err(RusticolError::invalid_argument(
                "eager point tile size must be positive",
            ));
        }
        let complex_bytes = size_of::<EagerComplex64>();
        let static_bytes = plan
            .couplings
            .len()
            .checked_mul(complex_bytes)
            .ok_or_else(|| RusticolError::invalid_argument("eager coupling workspace overflows"))?;
        let persistent_complex_components = plan
            .values
            .component_count
            .checked_add(plan.currents.component_count)
            .and_then(|value| value.checked_add(plan.amplitude_count))
            .and_then(|value| value.checked_add(plan.reduction_groups.len()))
            .ok_or_else(|| {
                RusticolError::invalid_argument("eager persistent workspace overflows")
            })?;
        let persistent_bytes_per_point = persistent_complex_components
            .checked_mul(complex_bytes)
            .and_then(|value| value.checked_add(size_of::<f64>()))
            .ok_or_else(|| RusticolError::invalid_argument("eager point workspace overflows"))?;
        let largest_kernel_io = used_kernel_io_components(&plan)?;
        let minimum_bytes_per_point = largest_kernel_io
            .checked_mul(complex_bytes)
            .and_then(|value| value.checked_add(persistent_bytes_per_point))
            .ok_or_else(|| RusticolError::invalid_argument("eager packet workspace overflows"))?;
        let dynamic_bytes = options
            .workspace_bytes
            .checked_sub(static_bytes)
            .ok_or_else(|| {
                RusticolError::invalid_argument("eager workspace cannot hold coupling values")
            })?;
        let maximum_tile = dynamic_bytes
            .checked_div(minimum_bytes_per_point)
            .unwrap_or(0);
        if maximum_tile == 0 {
            return Err(RusticolError::invalid_argument(format!(
                "eager workspace needs at least {} bytes for one point",
                static_bytes + minimum_bytes_per_point
            )));
        }
        let tile_capacity = min(options.point_tile_size, maximum_tile);
        let persistent_bytes = persistent_bytes_per_point
            .checked_mul(tile_capacity)
            .ok_or_else(|| {
                RusticolError::invalid_argument("eager persistent workspace overflows")
            })?;
        let packet_budget = dynamic_bytes.checked_sub(persistent_bytes).ok_or_else(|| {
            RusticolError::internal("eager workspace accounting lost its packet budget")
        })?;
        let schedule = build_schedule(&plan, tile_capacity, packet_budget)?;
        let workspace = allocate_workspace(&plan, tile_capacity, schedule.packet_buffer_len)?;
        let workspace_bytes = static_bytes
            .checked_add(persistent_bytes)
            .and_then(|value| {
                schedule
                    .packet_buffer_len
                    .checked_mul(complex_bytes)
                    .and_then(|packet| value.checked_add(packet))
            })
            .ok_or_else(|| RusticolError::invalid_argument("eager workspace size overflows"))?;
        if workspace_bytes > options.workspace_bytes {
            return Err(RusticolError::internal(
                "eager scheduler exceeded its workspace budget",
            ));
        }
        Ok(Self {
            plan,
            schedule,
            workspace,
            workspace_bytes,
        })
    }

    pub fn plan(&self) -> &EagerExecutionPlan {
        &self.plan
    }

    pub fn effective_point_tile_size(&self) -> usize {
        self.workspace.tile_capacity
    }

    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    pub fn packet_count(&self) -> usize {
        self.schedule
            .stages
            .iter()
            .map(|stage| stage.invocation_packets.len() + stage.finalization_packets.len())
            .sum::<usize>()
            + self.schedule.closure_packets.len()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_into<B: EagerKernelBackend>(
        &mut self,
        backend: &mut B,
        point_count: usize,
        initial_values: &[EagerComplex64],
        momenta: &[f64],
        model_parameters: &[EagerComplex64],
        amplitudes: &mut [EagerComplex64],
        reduced: &mut [f64],
    ) -> RusticolResult<()> {
        validate_execution_buffers(
            &self.plan,
            point_count,
            initial_values,
            momenta,
            model_parameters,
            amplitudes,
            reduced,
        )?;
        if model_parameters
            .iter()
            .any(|value| !complex_is_finite(*value))
        {
            return Err(RusticolError::invalid_argument(
                "eager model parameters must be finite",
            ));
        }
        resolve_couplings(
            &self.plan.couplings,
            model_parameters,
            &mut self.workspace.couplings,
        );
        let tile_capacity = self.workspace.tile_capacity;
        let mut tile_start = 0usize;
        while tile_start < point_count {
            let tile_points = min(tile_capacity, point_count - tile_start);
            initialize_tile(
                &self.plan,
                &mut self.workspace,
                point_count,
                tile_start,
                tile_points,
                initial_values,
            );
            for (stage, schedule) in self.plan.stages.iter().zip(&self.schedule.stages) {
                execute_stage(
                    stage,
                    schedule,
                    &self.plan.kernels,
                    &mut self.workspace,
                    backend,
                    point_count,
                    tile_start,
                    tile_points,
                    momenta,
                    model_parameters,
                )?;
            }
            execute_closures(
                &self.plan,
                &self.schedule.closure_packets,
                &mut self.workspace,
                backend,
                tile_points,
                model_parameters,
            )?;
            reduce_tile(&self.plan, &mut self.workspace, tile_points);
            copy_tile_results(
                &self.plan,
                &self.workspace,
                point_count,
                tile_start,
                tile_points,
                amplitudes,
                reduced,
            );
            tile_start += tile_points;
        }
        Ok(())
    }
}

fn used_kernel_io_components(plan: &EagerExecutionPlan) -> RusticolResult<usize> {
    let mut largest = 0usize;
    let mut record = |kernel_id: u32| -> RusticolResult<()> {
        let kernel = plan.kernels.get(&kernel_id).ok_or_else(|| {
            RusticolError::internal(format!("eager schedule lost kernel {kernel_id}"))
        })?;
        let count = usize::try_from(kernel.output_component_count)
            .ok()
            .and_then(|output| kernel.inputs.len().checked_add(output))
            .ok_or_else(|| RusticolError::artifact("eager kernel width overflows usize"))?;
        largest = largest.max(count);
        Ok(())
    };
    for stage in &plan.stages {
        for item in &stage.invocations {
            record(item.row.kernel_id)?;
        }
        for item in &stage.finalizations {
            record(item.row.kernel_id)?;
        }
    }
    for item in &plan.closures {
        record(item.row.kernel_id)?;
    }
    Ok(largest.max(1))
}

fn build_schedule(
    plan: &EagerExecutionPlan,
    tile_capacity: usize,
    packet_budget: usize,
) -> RusticolResult<ExecutionSchedule> {
    let mut stages = Vec::new();
    stages
        .try_reserve_exact(plan.stages.len())
        .map_err(|error| {
            RusticolError::invalid_argument(format!("could not reserve eager schedule: {error}"))
        })?;
    let mut packet_buffer_len = 0usize;
    for stage in &plan.stages {
        let invocation_packets = packetize(
            PacketRole::Invocation,
            &stage
                .invocations
                .iter()
                .map(|item| item.row.kernel_id)
                .collect::<Vec<_>>(),
            &plan.kernels,
            tile_capacity,
            packet_budget,
            &mut packet_buffer_len,
        )?;
        let finalization_ids = stage
            .finalizations
            .iter()
            .map(|item| item.row.kernel_id)
            .collect::<Vec<_>>();
        let finalization_packets = packetize(
            PacketRole::Finalization,
            &finalization_ids,
            &plan.kernels,
            tile_capacity,
            packet_budget,
            &mut packet_buffer_len,
        )?;
        stages.push(StageSchedule {
            invocation_packets,
            finalization_packets,
        });
    }
    let closure_ids = plan
        .closures
        .iter()
        .map(|item| item.row.kernel_id)
        .collect::<Vec<_>>();
    let closure_packets = packetize(
        PacketRole::Closure,
        &closure_ids,
        &plan.kernels,
        tile_capacity,
        packet_budget,
        &mut packet_buffer_len,
    )?;
    Ok(ExecutionSchedule {
        stages,
        closure_packets,
        packet_buffer_len,
    })
}

fn packetize(
    role: PacketRole,
    kernel_ids: &[u32],
    kernels: &BTreeMap<u32, EagerKernelSpec>,
    tile_capacity: usize,
    packet_budget: usize,
    packet_buffer_len: &mut usize,
) -> RusticolResult<Vec<KernelPacket>> {
    // Plan loading stably sorts each role by kernel id. Packetization therefore
    // scans contiguous equal-kernel runs and never needs a hot-loop dispatch map.
    let mut packets = Vec::new();
    let mut start = 0usize;
    while start < kernel_ids.len() {
        let kernel_id = kernel_ids[start];
        let mut run_end = start + 1;
        while run_end < kernel_ids.len() && kernel_ids[run_end] == kernel_id {
            run_end += 1;
        }
        let kernel = kernels.get(&kernel_id).ok_or_else(|| {
            RusticolError::internal(format!("eager schedule lost kernel {kernel_id}"))
        })?;
        let input_components = kernel.inputs.len();
        let output_components = usize::try_from(kernel.output_component_count)
            .map_err(|_| RusticolError::artifact("eager kernel output width does not fit usize"))?;
        let bytes_per_invocation = input_components
            .checked_add(output_components)
            .and_then(|value| value.checked_mul(tile_capacity))
            .and_then(|value| value.checked_mul(size_of::<EagerComplex64>()))
            .ok_or_else(|| RusticolError::invalid_argument("eager packet size overflows"))?;
        let capacity = packet_budget.checked_div(bytes_per_invocation).unwrap_or(0);
        if capacity == 0 {
            return Err(RusticolError::invalid_argument(format!(
                "eager workspace cannot hold one kernel {kernel_id} packet"
            )));
        }
        let mut packet_start = start;
        while packet_start < run_end {
            let packet_end = min(run_end, packet_start + capacity);
            let invocation_count = packet_end - packet_start;
            let elements = input_components
                .checked_add(output_components)
                .and_then(|value| value.checked_mul(invocation_count))
                .and_then(|value| value.checked_mul(tile_capacity))
                .ok_or_else(|| RusticolError::invalid_argument("eager packet size overflows"))?;
            *packet_buffer_len = (*packet_buffer_len).max(elements);
            packets.push(KernelPacket {
                role,
                kernel_id,
                item_range: packet_start..packet_end,
                input_components,
                output_components,
            });
            packet_start = packet_end;
        }
        start = run_end;
    }
    Ok(packets)
}

fn allocate_workspace(
    plan: &EagerExecutionPlan,
    tile_capacity: usize,
    packet_buffer_len: usize,
) -> RusticolResult<EagerWorkspace> {
    Ok(EagerWorkspace {
        tile_capacity,
        values: zeroed_complex_workspace(plan.values.component_count, tile_capacity, "values")?,
        currents: zeroed_complex_workspace(
            plan.currents.component_count,
            tile_capacity,
            "currents",
        )?,
        amplitudes: zeroed_complex_workspace(plan.amplitude_count, tile_capacity, "amplitudes")?,
        reduction_groups: zeroed_complex_workspace(
            plan.reduction_groups.len(),
            tile_capacity,
            "reduction groups",
        )?,
        couplings: zeroed_complex_values(plan.couplings.len(), "couplings")?,
        reduced: zeroed_real_values(tile_capacity, "reduction")?,
        packet: zeroed_complex_values(packet_buffer_len, "packet")?,
    })
}

fn zeroed_complex_workspace(
    component_count: usize,
    tile_capacity: usize,
    name: &str,
) -> RusticolResult<Vec<EagerComplex64>> {
    let len = component_count.checked_mul(tile_capacity).ok_or_else(|| {
        RusticolError::invalid_argument(format!("eager {name} workspace overflows"))
    })?;
    zeroed_complex_values(len, name)
}

fn zeroed_complex_values(len: usize, name: &str) -> RusticolResult<Vec<EagerComplex64>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        RusticolError::invalid_argument(format!("could not reserve eager {name}: {error}"))
    })?;
    values.resize(len, EagerComplex64::new(0.0, 0.0));
    Ok(values)
}

fn zeroed_real_values(len: usize, name: &str) -> RusticolResult<Vec<f64>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        RusticolError::invalid_argument(format!("could not reserve eager {name}: {error}"))
    })?;
    values.resize(len, 0.0);
    Ok(values)
}

#[allow(clippy::too_many_arguments)]
fn validate_execution_buffers(
    plan: &EagerExecutionPlan,
    point_count: usize,
    initial_values: &[EagerComplex64],
    momenta: &[f64],
    model_parameters: &[EagerComplex64],
    amplitudes: &[EagerComplex64],
    reduced: &[f64],
) -> RusticolResult<()> {
    validate_buffer_len(
        "initial value",
        initial_values.len(),
        plan.values.component_count,
        point_count,
    )?;
    validate_buffer_len(
        "momentum",
        momenta.len(),
        plan.momenta.component_count,
        point_count,
    )?;
    if model_parameters.len() != plan.parameter_count {
        return Err(RusticolError::invalid_argument(format!(
            "eager model parameter buffer has length {}, expected {}",
            model_parameters.len(),
            plan.parameter_count
        )));
    }
    validate_buffer_len(
        "amplitude output",
        amplitudes.len(),
        plan.amplitude_count,
        point_count,
    )?;
    if reduced.len() != point_count {
        return Err(RusticolError::invalid_argument(format!(
            "eager reduced output has length {}, expected {point_count}",
            reduced.len()
        )));
    }
    Ok(())
}

fn validate_buffer_len(
    name: &str,
    actual: usize,
    component_count: usize,
    point_count: usize,
) -> RusticolResult<()> {
    let expected = component_count.checked_mul(point_count).ok_or_else(|| {
        RusticolError::invalid_argument(format!("eager {name} buffer length overflows"))
    })?;
    if actual != expected {
        return Err(RusticolError::invalid_argument(format!(
            "eager {name} buffer has length {actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn resolve_couplings(
    rows: &[EagerCouplingRow],
    parameters: &[EagerComplex64],
    output: &mut [EagerComplex64],
) {
    for (row, value) in rows.iter().zip(output) {
        let real = if row.real_parameter_id == MISSING_U32 {
            row.constant_real
        } else {
            parameters[row.real_parameter_id as usize].re
        };
        let imag = if row.imag_parameter_id == MISSING_U32 {
            row.constant_imag
        } else {
            parameters[row.imag_parameter_id as usize].re
        };
        *value = EagerComplex64::new(real, imag);
    }
}
