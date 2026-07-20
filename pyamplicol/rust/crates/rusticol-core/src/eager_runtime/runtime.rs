// SPDX-License-Identifier: 0BSD

use super::execute::{
    copy_selected_tile_amplitudes, copy_tile_amplitudes, copy_tile_results, execute_closures,
    execute_stage, initialize_tile, reduce_tile,
};
use super::plan::{
    ClosureExecutionRows, ComponentRange, EagerExecutionPlan, EagerStagePlan, ScheduledClosure,
    ScheduledDirectClosure, ScheduledFinalization, complex_is_finite,
};
use super::profile::{execute_closures_profiled, execute_stage_profiled};
use super::{
    EagerComplex64, EagerExecutionProfile, EagerKernelBackend, EagerKernelSpec, EagerRuntimeOptions,
};
use crate::{EagerCouplingRow, MISSING_U32, RusticolError, RusticolResult};
use std::cmp::min;
use std::collections::BTreeMap;
use std::mem::size_of;
use std::ops::Range;
use std::time::Instant;

// `workspace_bytes` is a hard upper bound, not a target packet size. Keeping
// one gather/evaluate/scatter packet cache-sized avoids turning otherwise
// linear eager execution into a memory-bandwidth workload when many
// invocations share one prepared kernel.
const MAX_PACKET_BUFFER_BYTES: usize = 4 * 1024 * 1024;
const MAX_PERSISTENT_TILE_BYTES: usize = 16 * 1024 * 1024;
const PREFERRED_TILE_ALIGNMENT: usize = 8;

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
    pub(super) independent_block_size: usize,
    pub(super) item_range: Range<usize>,
    pub(super) input_components: usize,
    pub(super) output_components: usize,
    pub(super) linear_finalization: Option<LinearFinalizationPacket>,
}

#[derive(Clone, Debug)]
pub(super) struct LinearFinalizationPacket {
    pub(super) momentum_groups: Vec<ComponentRange>,
    pub(super) item_group_indices: Vec<usize>,
    pub(super) current_components: Vec<u32>,
}

impl LinearFinalizationPacket {
    pub(super) fn lane_multiplier(&self) -> usize {
        self.momentum_groups.len() * self.current_components.len()
    }
}

#[derive(Clone, Debug)]
pub(super) struct StageSchedule {
    pub(super) invocation_packets: Vec<KernelPacket>,
    pub(super) finalization_packets: Vec<KernelPacket>,
}

#[derive(Clone, Debug)]
pub(super) struct ExecutionSchedule {
    pub(super) stages: Vec<StageSchedule>,
    pub(super) closure_packets: Vec<KernelPacket>,
    packet_buffer_len: usize,
}

#[derive(Clone, Debug)]
struct SelectedExecution {
    active_groups: Vec<u32>,
    active_amplitude_indices: Vec<usize>,
    stages: Vec<EagerStagePlan>,
    closures: Vec<ScheduledClosure>,
    direct_closures: Vec<ScheduledDirectClosure>,
    zero_amplitude_indices: Vec<usize>,
    active_zero_amplitude_indices: Vec<usize>,
    schedule: ExecutionSchedule,
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
    packet_budget: usize,
    selected: Option<SelectedExecution>,
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
            .checked_add(plan.current_workspace_component_count())
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
        let workspace_maximum_tile = dynamic_bytes
            .checked_div(minimum_bytes_per_point)
            .unwrap_or(0);
        if workspace_maximum_tile == 0 {
            return Err(RusticolError::invalid_argument(format!(
                "eager workspace needs at least {} bytes for one point",
                static_bytes + minimum_bytes_per_point
            )));
        }
        let tile_capacity = effective_tile_capacity(
            options.point_tile_size,
            workspace_maximum_tile,
            persistent_bytes_per_point,
        );
        let persistent_bytes = persistent_bytes_per_point
            .checked_mul(tile_capacity)
            .ok_or_else(|| {
                RusticolError::invalid_argument("eager persistent workspace overflows")
            })?;
        let packet_budget = dynamic_bytes
            .checked_sub(persistent_bytes)
            .ok_or_else(|| {
                RusticolError::internal("eager workspace accounting lost its packet budget")
            })?
            .min(MAX_PACKET_BUFFER_BYTES);
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
            packet_budget,
            selected: None,
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

    pub fn selector_group_ids(&self) -> Option<Vec<u32>> {
        let domains = self.plan.selector_domains.as_ref()?;
        Some(domains.group_ids.clone())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_selected_amplitudes_into<B: EagerKernelBackend>(
        &mut self,
        backend: &mut B,
        active_groups: &[u32],
        point_count: usize,
        initial_values: &[EagerComplex64],
        momenta: &[f64],
        model_parameters: &[EagerComplex64],
        amplitudes: &mut [EagerComplex64],
    ) -> RusticolResult<()> {
        self.evaluate_selected_amplitudes_impl(
            backend,
            active_groups,
            point_count,
            initial_values,
            momenta,
            model_parameters,
            amplitudes,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_selected_active_amplitudes_into<B: EagerKernelBackend>(
        &mut self,
        backend: &mut B,
        active_groups: &[u32],
        point_count: usize,
        initial_values: &[EagerComplex64],
        momenta: &[f64],
        model_parameters: &[EagerComplex64],
        amplitudes: &mut [EagerComplex64],
    ) -> RusticolResult<()> {
        self.evaluate_selected_amplitudes_impl(
            backend,
            active_groups,
            point_count,
            initial_values,
            momenta,
            model_parameters,
            amplitudes,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_selected_amplitudes_impl<B: EagerKernelBackend>(
        &mut self,
        backend: &mut B,
        active_groups: &[u32],
        point_count: usize,
        initial_values: &[EagerComplex64],
        momenta: &[f64],
        model_parameters: &[EagerComplex64],
        amplitudes: &mut [EagerComplex64],
        active_amplitudes_only: bool,
    ) -> RusticolResult<()> {
        validate_selected_execution_buffers(
            &self.plan,
            point_count,
            initial_values,
            momenta,
            model_parameters,
            amplitudes,
        )?;
        validate_model_parameters(model_parameters)?;
        self.prepare_selected_execution(active_groups)?;
        let (plan, full_schedule, selected, workspace) = (
            &self.plan,
            &self.schedule,
            &self.selected,
            &mut self.workspace,
        );
        resolve_couplings(&plan.couplings, model_parameters, &mut workspace.couplings);
        let (stages, closures, direct_closures, zero_amplitudes, schedule) =
            selected_execution_rows(
                plan,
                full_schedule,
                selected.as_ref(),
                active_amplitudes_only,
            );
        let active_amplitude_indices = selected
            .as_ref()
            .map(|selected| selected.active_amplitude_indices.as_slice());
        let tile_capacity = workspace.tile_capacity;
        let mut tile_start = 0usize;
        while tile_start < point_count {
            let tile_points = min(tile_capacity, point_count - tile_start);
            initialize_tile(
                plan,
                workspace,
                point_count,
                tile_start,
                tile_points,
                initial_values,
                zero_amplitudes,
            );
            for (stage, stage_schedule) in stages.iter().zip(&schedule.stages) {
                execute_stage(
                    stage,
                    stage_schedule,
                    &plan.kernels,
                    workspace,
                    backend,
                    point_count,
                    tile_start,
                    tile_points,
                    momenta,
                    model_parameters,
                )?;
            }
            execute_closures(
                ClosureExecutionRows {
                    closures,
                    direct_closures,
                    kernels: &plan.kernels,
                },
                &schedule.closure_packets,
                workspace,
                backend,
                tile_points,
                model_parameters,
            )?;
            if active_amplitudes_only {
                copy_selected_tile_amplitudes(
                    active_amplitude_indices.unwrap_or(&[]),
                    workspace,
                    point_count,
                    tile_start,
                    tile_points,
                    amplitudes,
                );
            } else {
                copy_tile_amplitudes(
                    plan,
                    workspace,
                    point_count,
                    tile_start,
                    tile_points,
                    amplitudes,
                );
            }
            tile_start += tile_points;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_selected_amplitudes_profile_into<B: EagerKernelBackend>(
        &mut self,
        backend: &mut B,
        active_groups: &[u32],
        point_count: usize,
        initial_values: &[EagerComplex64],
        momenta: &[f64],
        model_parameters: &[EagerComplex64],
        amplitudes: &mut [EagerComplex64],
    ) -> RusticolResult<EagerExecutionProfile> {
        validate_selected_execution_buffers(
            &self.plan,
            point_count,
            initial_values,
            momenta,
            model_parameters,
            amplitudes,
        )?;
        validate_model_parameters(model_parameters)?;
        let total_started = Instant::now();
        self.prepare_selected_execution(active_groups)?;
        let (plan, full_schedule, selected, workspace) = (
            &self.plan,
            &self.schedule,
            &self.selected,
            &mut self.workspace,
        );
        let initialize_started = Instant::now();
        resolve_couplings(&plan.couplings, model_parameters, &mut workspace.couplings);
        let mut profile = EagerExecutionProfile {
            initialize: initialize_started.elapsed(),
            ..EagerExecutionProfile::default()
        };
        let (stages, closures, direct_closures, zero_amplitudes, schedule) =
            selected_execution_rows(plan, full_schedule, selected.as_ref(), true);
        let active_amplitude_indices = selected
            .as_ref()
            .map(|selected| selected.active_amplitude_indices.as_slice());
        let tile_capacity = workspace.tile_capacity;
        let mut tile_start = 0usize;
        while tile_start < point_count {
            let tile_points = min(tile_capacity, point_count - tile_start);
            let initialize_started = Instant::now();
            initialize_tile(
                plan,
                workspace,
                point_count,
                tile_start,
                tile_points,
                initial_values,
                zero_amplitudes,
            );
            profile.initialize += initialize_started.elapsed();
            for (stage, stage_schedule) in stages.iter().zip(&schedule.stages) {
                execute_stage_profiled(
                    stage,
                    stage_schedule,
                    &plan.kernels,
                    workspace,
                    backend,
                    point_count,
                    tile_start,
                    tile_points,
                    momenta,
                    model_parameters,
                    &mut profile,
                )?;
            }
            execute_closures_profiled(
                ClosureExecutionRows {
                    closures,
                    direct_closures,
                    kernels: &plan.kernels,
                },
                &schedule.closure_packets,
                workspace,
                backend,
                tile_points,
                model_parameters,
                &mut profile,
            )?;
            let copy_started = Instant::now();
            copy_selected_tile_amplitudes(
                active_amplitude_indices.unwrap_or(&[]),
                workspace,
                point_count,
                tile_start,
                tile_points,
                amplitudes,
            );
            profile.copy_out += copy_started.elapsed();
            tile_start += tile_points;
        }
        profile.total = total_started.elapsed();
        Ok(profile)
    }

    fn prepare_selected_execution(&mut self, active_groups: &[u32]) -> RusticolResult<()> {
        if active_groups.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(RusticolError::invalid_argument(
                "eager active selector groups must be sorted and unique",
            ));
        }
        if self.plan.selector_domains.is_none() {
            self.selected = None;
            return Ok(());
        }
        let selector_domains = self
            .plan
            .selector_domains
            .as_ref()
            .expect("selector domains were checked above");
        if let Some(unknown) = active_groups
            .iter()
            .find(|group| selector_domains.group_ids.binary_search(group).is_err())
        {
            return Err(RusticolError::invalid_argument(format!(
                "eager selector references unknown coherent group {unknown}"
            )));
        }
        if self
            .selected
            .as_ref()
            .is_some_and(|selected| selected.active_groups == active_groups)
        {
            return Ok(());
        }
        self.selected = Some(build_selected_execution(
            &self.plan,
            active_groups,
            self.workspace.tile_capacity,
            self.packet_budget,
        )?);
        Ok(())
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
                &self.plan.zero_amplitude_indices,
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
                ClosureExecutionRows {
                    closures: &self.plan.closures,
                    direct_closures: &self.plan.direct_closures,
                    kernels: &self.plan.kernels,
                },
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

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_profile_into<B: EagerKernelBackend>(
        &mut self,
        backend: &mut B,
        point_count: usize,
        initial_values: &[EagerComplex64],
        momenta: &[f64],
        model_parameters: &[EagerComplex64],
        amplitudes: &mut [EagerComplex64],
        reduced: &mut [f64],
    ) -> RusticolResult<EagerExecutionProfile> {
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

        let total_started = Instant::now();
        let initialize_started = Instant::now();
        resolve_couplings(
            &self.plan.couplings,
            model_parameters,
            &mut self.workspace.couplings,
        );
        let mut profile = EagerExecutionProfile {
            initialize: initialize_started.elapsed(),
            ..EagerExecutionProfile::default()
        };
        let tile_capacity = self.workspace.tile_capacity;
        let mut tile_start = 0usize;
        while tile_start < point_count {
            let tile_points = min(tile_capacity, point_count - tile_start);

            let initialize_started = Instant::now();
            initialize_tile(
                &self.plan,
                &mut self.workspace,
                point_count,
                tile_start,
                tile_points,
                initial_values,
                &self.plan.zero_amplitude_indices,
            );
            profile.initialize += initialize_started.elapsed();

            for (stage, schedule) in self.plan.stages.iter().zip(&self.schedule.stages) {
                execute_stage_profiled(
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
                    &mut profile,
                )?;
            }
            execute_closures_profiled(
                ClosureExecutionRows {
                    closures: &self.plan.closures,
                    direct_closures: &self.plan.direct_closures,
                    kernels: &self.plan.kernels,
                },
                &self.schedule.closure_packets,
                &mut self.workspace,
                backend,
                tile_points,
                model_parameters,
                &mut profile,
            )?;

            let reduction_started = Instant::now();
            reduce_tile(&self.plan, &mut self.workspace, tile_points);
            profile.reduction += reduction_started.elapsed();

            let copy_started = Instant::now();
            copy_tile_results(
                &self.plan,
                &self.workspace,
                point_count,
                tile_start,
                tile_points,
                amplitudes,
                reduced,
            );
            profile.copy_out += copy_started.elapsed();
            tile_start += tile_points;
        }
        profile.total = total_started.elapsed();
        debug_assert!(profile.accounted() <= profile.total);
        Ok(profile)
    }
}

fn effective_tile_capacity(
    requested_tile: usize,
    workspace_maximum_tile: usize,
    persistent_bytes_per_point: usize,
) -> usize {
    let cache_maximum_tile = (MAX_PERSISTENT_TILE_BYTES / persistent_bytes_per_point).max(1);
    let maximum_tile = workspace_maximum_tile.min(cache_maximum_tile);
    let mut tile_capacity = min(requested_tile, maximum_tile);
    if tile_capacity < requested_tile && tile_capacity >= PREFERRED_TILE_ALIGNMENT {
        tile_capacity -= tile_capacity % PREFERRED_TILE_ALIGNMENT;
    }
    tile_capacity
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
    build_schedule_for_rows(
        &plan.kernels,
        &plan.stages,
        &plan.closures,
        tile_capacity,
        packet_budget,
    )
}

fn build_schedule_for_rows(
    kernels: &BTreeMap<u32, EagerKernelSpec>,
    stage_rows: &[EagerStagePlan],
    closure_rows: &[ScheduledClosure],
    tile_capacity: usize,
    packet_budget: usize,
) -> RusticolResult<ExecutionSchedule> {
    let mut stages = Vec::new();
    stages
        .try_reserve_exact(stage_rows.len())
        .map_err(|error| {
            RusticolError::invalid_argument(format!("could not reserve eager schedule: {error}"))
        })?;
    let mut packet_buffer_len = 0usize;
    for stage in stage_rows {
        let invocation_packets = packetize(
            PacketRole::Invocation,
            &stage
                .invocations
                .iter()
                .map(|item| item.row.kernel_id)
                .collect::<Vec<_>>(),
            kernels,
            tile_capacity,
            packet_budget,
            &mut packet_buffer_len,
        )?;
        let finalization_ids = stage
            .finalizations
            .iter()
            .map(|item| item.row.kernel_id)
            .collect::<Vec<_>>();
        let mut finalization_packets = packetize(
            PacketRole::Finalization,
            &finalization_ids,
            kernels,
            tile_capacity,
            packet_budget,
            &mut packet_buffer_len,
        )?;
        configure_linear_finalization_packets(
            &mut finalization_packets,
            &stage.finalizations,
            kernels,
        )?;
        stages.push(StageSchedule {
            invocation_packets,
            finalization_packets,
        });
    }
    let closure_ids = closure_rows
        .iter()
        .map(|item| item.row.kernel_id)
        .collect::<Vec<_>>();
    let closure_packets = packetize(
        PacketRole::Closure,
        &closure_ids,
        kernels,
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

fn build_selected_execution(
    plan: &EagerExecutionPlan,
    active_groups: &[u32],
    tile_capacity: usize,
    packet_budget: usize,
) -> RusticolResult<SelectedExecution> {
    let domains = plan.selector_domains.as_ref().ok_or_else(|| {
        RusticolError::internal("cannot build a selected eager plan without selector domains")
    })?;
    let domain_active = domains
        .memberships
        .iter()
        .map(|members| {
            members
                .iter()
                .any(|group| active_groups.binary_search(group).is_ok())
        })
        .collect::<Vec<_>>();
    let active = |domain_id: Option<u32>| -> RusticolResult<bool> {
        let Some(domain_id) = domain_id else {
            return Ok(true);
        };
        domain_active
            .get(domain_id as usize)
            .copied()
            .ok_or_else(|| RusticolError::internal("eager selector domain id escaped validation"))
    };

    let mut stages = Vec::with_capacity(plan.stages.len());
    for stage in &plan.stages {
        let mut invocations = Vec::new();
        let mut attachments = Vec::new();
        for invocation in &stage.invocations {
            if !active(invocation.selector_domain_id)? {
                continue;
            }
            let attachment_start = attachments.len();
            for attachment in &stage.attachments[invocation.attachment_range.clone()] {
                if active(attachment.selector_domain_id)? {
                    attachments.push(*attachment);
                }
            }
            let attachment_count = attachments.len() - attachment_start;
            if attachment_count == 0 {
                return Err(RusticolError::internal(
                    "active eager invocation has no active attachment",
                ));
            }
            let mut invocation = invocation.clone();
            invocation.attachment_range = attachment_start..attachment_start + attachment_count;
            invocations.push(invocation);
        }
        let mut finalization_copies = Vec::new();
        for item in &stage.finalization_copies {
            if active(item.selector_domain_id)? {
                finalization_copies.push(*item);
            }
        }
        let mut finalizations = Vec::new();
        for item in &stage.finalizations {
            if active(item.selector_domain_id)? {
                finalizations.push(item.clone());
            }
        }
        stages.push(EagerStagePlan {
            stage_index: stage.stage_index,
            current_component_count: stage.current_component_count,
            invocations,
            attachments,
            finalization_copies,
            finalizations,
            zero_current_ranges: Vec::new(),
        });
    }
    let mut closures = Vec::new();
    for item in &plan.closures {
        if active(item.selector_domain_id)? {
            closures.push(item.clone());
        }
    }
    let mut direct_closures = Vec::new();
    for item in &plan.direct_closures {
        if active(item.selector_domain_id)? {
            direct_closures.push(item.clone());
        }
    }
    for stage in &mut stages {
        stage.zero_current_ranges = super::plan::mark_initial_current_writes(
            &stage.invocations,
            &mut stage.attachments,
            &stage.finalizations,
        );
    }
    let zero_amplitude_indices = super::plan::mark_initial_amplitude_writes(
        &mut closures,
        &mut direct_closures,
        plan.amplitude_count,
    );
    let mut active_amplitude_indices = plan
        .reduction_groups
        .iter()
        .filter(|group| {
            active_groups
                .binary_search(&group.coherent_group_id)
                .is_ok()
        })
        .flat_map(|group| group.amplitude_indices.iter().copied())
        .map(|index| index as usize)
        .collect::<Vec<_>>();
    active_amplitude_indices.sort_unstable();
    active_amplitude_indices.dedup();
    let active_zero_amplitude_indices = zero_amplitude_indices
        .iter()
        .copied()
        .filter(|index| active_amplitude_indices.binary_search(index).is_ok())
        .collect::<Vec<_>>();
    let schedule = build_schedule_for_rows(
        &plan.kernels,
        &stages,
        &closures,
        tile_capacity,
        packet_budget,
    )?;
    Ok(SelectedExecution {
        active_groups: active_groups.to_vec(),
        active_amplitude_indices,
        stages,
        closures,
        direct_closures,
        zero_amplitude_indices,
        active_zero_amplitude_indices,
        schedule,
    })
}

fn selected_execution_rows<'a>(
    plan: &'a EagerExecutionPlan,
    full_schedule: &'a ExecutionSchedule,
    selected: Option<&'a SelectedExecution>,
    active_amplitudes_only: bool,
) -> (
    &'a [EagerStagePlan],
    &'a [ScheduledClosure],
    &'a [ScheduledDirectClosure],
    &'a [usize],
    &'a ExecutionSchedule,
) {
    if let Some(selected) = selected {
        return (
            &selected.stages,
            &selected.closures,
            &selected.direct_closures,
            if active_amplitudes_only {
                &selected.active_zero_amplitude_indices
            } else {
                &selected.zero_amplitude_indices
            },
            &selected.schedule,
        );
    }
    (
        &plan.stages,
        &plan.closures,
        &plan.direct_closures,
        &plan.zero_amplitude_indices,
        full_schedule,
    )
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
        let requested_block_size = if role == PacketRole::Invocation {
            usize::try_from(kernel.independent_block_size)
                .map_err(|_| RusticolError::artifact("eager block size does not fit usize"))?
        } else {
            1
        };
        let block_size = if requested_block_size > 1 && capacity >= requested_block_size {
            requested_block_size
        } else {
            1
        };
        let blocked_end = if block_size > 1 {
            start + ((run_end - start) / block_size) * block_size
        } else {
            start
        };
        let mut packet_start = start;
        let blocked_capacity = (capacity / block_size) * block_size;
        while packet_start < blocked_end {
            let packet_end = min(blocked_end, packet_start + blocked_capacity);
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
                independent_block_size: block_size,
                item_range: packet_start..packet_end,
                input_components,
                output_components,
                linear_finalization: None,
            });
            packet_start = packet_end;
        }
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
                independent_block_size: 1,
                item_range: packet_start..packet_end,
                input_components,
                output_components,
                linear_finalization: None,
            });
            packet_start = packet_end;
        }
        start = run_end;
    }
    Ok(packets)
}

fn configure_linear_finalization_packets(
    packets: &mut [KernelPacket],
    rows: &[ScheduledFinalization],
    kernels: &BTreeMap<u32, EagerKernelSpec>,
) -> RusticolResult<()> {
    for packet in packets {
        let items = &rows[packet.item_range.clone()];
        let kernel = kernels.get(&packet.kernel_id).ok_or_else(|| {
            RusticolError::internal(format!(
                "eager schedule lost finalization kernel {}",
                packet.kernel_id
            ))
        })?;
        if !kernel.homogeneous_linear_first_current {
            continue;
        }
        let current_components = kernel
            .inputs
            .iter()
            .filter_map(|input| match input {
                super::EagerKernelInput::FirstCurrentComponent(component) => Some(*component),
                _ => None,
            })
            .collect::<Vec<_>>();
        if current_components.is_empty() {
            continue;
        }

        let mut momentum_groups = Vec::<ComponentRange>::new();
        let mut item_group_indices = Vec::with_capacity(items.len());
        for item in items {
            let group_index = momentum_groups
                .iter()
                .position(|momentum| *momentum == item.momentum)
                .unwrap_or_else(|| {
                    momentum_groups.push(item.momentum);
                    momentum_groups.len() - 1
                });
            item_group_indices.push(group_index);
        }
        let lane_multiplier = momentum_groups
            .len()
            .checked_mul(current_components.len())
            .ok_or_else(|| {
                RusticolError::invalid_argument("eager linear finalization overflows")
            })?;
        // Basis evaluation is useful only when shared momentum dependence
        // reduces the number of backend lanes. Otherwise retain the direct
        // prepared-kernel path and its simpler scatter.
        if lane_multiplier >= items.len() {
            continue;
        }
        packet.linear_finalization = Some(LinearFinalizationPacket {
            momentum_groups,
            item_group_indices,
            current_components,
        });
    }
    Ok(())
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
            plan.current_workspace_component_count(),
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

fn validate_selected_execution_buffers(
    plan: &EagerExecutionPlan,
    point_count: usize,
    initial_values: &[EagerComplex64],
    momenta: &[f64],
    model_parameters: &[EagerComplex64],
    amplitudes: &[EagerComplex64],
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
    )
}

fn validate_model_parameters(parameters: &[EagerComplex64]) -> RusticolResult<()> {
    if parameters.iter().any(|value| !complex_is_finite(*value)) {
        return Err(RusticolError::invalid_argument(
            "eager model parameters must be finite",
        ));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        EagerAttachmentRow, EagerClosureRow, EagerCouplingRow, EagerDirectClosureSpec,
        EagerFinalizationRow, EagerInvocationRow, EagerKernelInput, EagerKernelRole,
        EagerPlanDefinition, EagerPlanDimensions, EagerPlanPayloads, EagerReductionEntry,
        EagerReductionGroup, EagerStagePayload, MISSING_U32,
    };

    #[derive(Default)]
    struct AddBackend;

    impl EagerKernelBackend for AddBackend {
        fn evaluate_batch(
            &mut self,
            call: super::super::EagerKernelCall<'_>,
        ) -> RusticolResult<()> {
            assert_eq!(call.kernel_id, 7);
            assert_eq!(call.input_component_count, 3);
            assert_eq!(call.output_component_count, 1);
            for lane in 0..call.lane_count {
                let row = lane * call.input_component_count;
                call.outputs[lane] = call.inputs[row] + call.inputs[row + 1] + call.inputs[row + 2];
            }
            Ok(())
        }
    }

    fn profiled_runtime() -> EagerExecutionRuntime {
        let definition = EagerPlanDefinition {
            dimensions: EagerPlanDimensions {
                value_slot_component_counts: vec![1, 1, 1],
                momentum_slot_component_counts: vec![1],
                current_component_counts: vec![1],
                parameter_count: 0,
                amplitude_count: 1,
            },
            kernels: vec![EagerKernelSpec {
                kernel_id: 7,
                role: EagerKernelRole::Vertex,
                inputs: vec![
                    EagerKernelInput::FirstCurrentComponent(0),
                    EagerKernelInput::SecondCurrentComponent(0),
                    EagerKernelInput::FirstMomentumComponent(0),
                ],
                output_component_count: 1,
                homogeneous_linear_first_current: false,
                independent_block_size: 1,
            }],
            direct_closures: vec![EagerDirectClosureSpec {
                closure_index: 0,
                coefficients: vec![EagerComplex64::new(1.0, 0.0)],
            }],
            reduction_groups: vec![EagerReductionGroup {
                coherent_group_id: 0,
                amplitude_indices: vec![0],
            }],
            reduction_entries: vec![EagerReductionEntry {
                left_group_index: 0,
                right_group_index: 0,
                coefficient: EagerComplex64::new(1.0, 0.0),
            }],
        };
        let couplings = EagerCouplingRow::encode_table(&[EagerCouplingRow {
            real_parameter_id: MISSING_U32,
            imag_parameter_id: MISSING_U32,
            constant_real: 1.0,
            constant_imag: 0.0,
        }])
        .expect("coupling table");
        let invocations = EagerInvocationRow::encode_table(&[EagerInvocationRow {
            kernel_id: 7,
            left_value_slot_id: 0,
            right_value_slot_id: 1,
            left_momentum_slot_id: 0,
            right_momentum_slot_id: 0,
            coupling_slot_id: 0,
            output_factor_source: crate::EAGER_OUTPUT_FACTOR_NONE,
            attachment_start: 0,
            attachment_count: 1,
        }])
        .expect("invocation table");
        let attachments = EagerAttachmentRow::encode_table(&[EagerAttachmentRow {
            result_current_id: 0,
            factor_real: 1.0,
            factor_imag: 0.0,
        }])
        .expect("attachment table");
        let finalizations = EagerFinalizationRow::encode_table(&[EagerFinalizationRow {
            kernel_id: MISSING_U32,
            current_id: 0,
            unpropagated_value_slot_id: 2,
            propagated_value_slot_id: MISSING_U32,
            momentum_slot_id: 0,
        }])
        .expect("finalization table");
        let closures = EagerClosureRow::encode_table(&[EagerClosureRow {
            kernel_id: MISSING_U32,
            left_value_slot_id: 2,
            right_value_slot_id: 0,
            amplitude_index: 0,
            coupling_slot_id: MISSING_U32,
            output_factor_source: crate::EAGER_OUTPUT_FACTOR_NONE,
            factor_real: 1.0,
            factor_imag: 0.0,
        }])
        .expect("closure table");
        let stage = EagerStagePayload {
            stage_index: 1,
            invocations: &invocations,
            attachments: &attachments,
            finalizations: &finalizations,
        };
        let plan = EagerExecutionPlan::from_payloads(
            definition,
            EagerPlanPayloads {
                couplings: &couplings,
                stages: &[stage],
                closures: &closures,
                selector_domains: None,
            },
        )
        .expect("profile plan");
        EagerExecutionRuntime::new(
            plan,
            EagerRuntimeOptions {
                point_tile_size: 2,
                workspace_bytes: 4096,
            },
        )
        .expect("profile runtime")
    }

    #[test]
    fn profiled_execution_matches_hot_path_and_accounts_for_every_phase() {
        let point_count = 4;
        let initial_values = [
            EagerComplex64::new(1.0, 0.0),
            EagerComplex64::new(2.0, 0.0),
            EagerComplex64::new(3.0, 0.0),
            EagerComplex64::new(4.0, 0.0),
            EagerComplex64::new(10.0, 0.0),
            EagerComplex64::new(20.0, 0.0),
            EagerComplex64::new(30.0, 0.0),
            EagerComplex64::new(40.0, 0.0),
            EagerComplex64::new(0.0, 0.0),
            EagerComplex64::new(0.0, 0.0),
            EagerComplex64::new(0.0, 0.0),
            EagerComplex64::new(0.0, 0.0),
        ];
        let momenta = [0.5, 1.0, 1.5, 2.0];
        let mut hot = profiled_runtime();
        let mut profiled = profiled_runtime();
        let mut hot_amplitudes = vec![EagerComplex64::new(0.0, 0.0); point_count];
        let mut hot_reduced = vec![0.0; point_count];
        let mut profiled_amplitudes = hot_amplitudes.clone();
        let mut profiled_reduced = hot_reduced.clone();

        hot.evaluate_into(
            &mut AddBackend,
            point_count,
            &initial_values,
            &momenta,
            &[],
            &mut hot_amplitudes,
            &mut hot_reduced,
        )
        .expect("hot eager execution");
        let profile = profiled
            .evaluate_profile_into(
                &mut AddBackend,
                point_count,
                &initial_values,
                &momenta,
                &[],
                &mut profiled_amplitudes,
                &mut profiled_reduced,
            )
            .expect("profile eager execution");

        assert_eq!(profiled_amplitudes, hot_amplitudes);
        assert_eq!(profiled_reduced, hot_reduced);
        assert!(!profile.initialize.is_zero());
        assert!(!profile.gather.is_zero());
        assert!(!profile.kernel_call.is_zero());
        assert!(!profile.invocation_scatter.is_zero());
        assert!(!profile.finalization.is_zero());
        assert!(!profile.closure.is_zero());
        assert!(!profile.reduction.is_zero());
        assert!(!profile.copy_out.is_zero());
        assert!(profile.accounted() <= profile.total);
    }

    #[test]
    fn effective_tile_uses_requested_size_as_an_upper_bound() {
        assert_eq!(effective_tile_capacity(32, 4096, 63_240), 32);
        assert_eq!(effective_tile_capacity(1024, 4096, 63_240), 264);
        assert_eq!(effective_tile_capacity(1024, 4096, usize::MAX), 1);
    }

    #[test]
    fn shared_linear_finalization_requires_an_exact_kernel_proof() {
        let rows = [0_u32, 1_u32].map(|current_id| ScheduledFinalization {
            row: EagerFinalizationRow {
                kernel_id: 7,
                current_id,
                unpropagated_value_slot_id: MISSING_U32,
                propagated_value_slot_id: current_id,
                momentum_slot_id: 0,
            },
            current: ComponentRange {
                start: current_id as usize,
                len: 1,
            },
            propagated: Some(ComponentRange {
                start: current_id as usize,
                len: 1,
            }),
            momentum: ComponentRange { start: 0, len: 1 },
            selector_domain_id: None,
        });
        let packet = KernelPacket {
            role: PacketRole::Finalization,
            kernel_id: 7,
            independent_block_size: 1,
            item_range: 0..2,
            input_components: 2,
            output_components: 1,
            linear_finalization: None,
        };
        let kernel = |proved| EagerKernelSpec {
            kernel_id: 7,
            role: EagerKernelRole::Finalization,
            inputs: vec![
                EagerKernelInput::FirstCurrentComponent(0),
                EagerKernelInput::FirstMomentumComponent(0),
            ],
            output_component_count: 1,
            homogeneous_linear_first_current: proved,
            independent_block_size: 1,
        };

        let mut packets = vec![packet.clone()];
        configure_linear_finalization_packets(
            &mut packets,
            &rows,
            &BTreeMap::from([(7, kernel(false))]),
        )
        .unwrap();
        assert!(packets[0].linear_finalization.is_none());

        configure_linear_finalization_packets(
            &mut packets,
            &rows,
            &BTreeMap::from([(7, kernel(true))]),
        )
        .unwrap();
        assert_eq!(
            packets[0]
                .linear_finalization
                .as_ref()
                .unwrap()
                .lane_multiplier(),
            1
        );
    }
}
