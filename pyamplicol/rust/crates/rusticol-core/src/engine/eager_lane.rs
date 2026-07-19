// SPDX-License-Identifier: 0BSD

use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct EagerParameterProjectionEntry {
    pub(super) prepared_index: usize,
    pub(super) runtime_real_index: usize,
    pub(super) runtime_imaginary_index: Option<usize>,
}

#[derive(Debug)]
pub(super) struct EagerParameterProjection {
    pub(super) parameter_count: usize,
    pub(super) entries: Vec<EagerParameterProjectionEntry>,
}

pub(super) struct EagerNativeRuntime {
    scheduler: crate::EagerExecutionRuntime,
    backend: PreparedEvaluatorBackend,
    backend_name: String,
    parameter_projection: EagerParameterProjection,
    raw_sum_groups: Vec<RawSumGroup>,
    color_contraction: Option<ColorContractionRuntime>,
    initial_values: Vec<crate::EagerComplex64>,
    momenta: Vec<f64>,
    model_parameters: Vec<crate::EagerComplex64>,
    amplitudes: Vec<crate::EagerComplex64>,
    color_group_scratch: Vec<crate::EagerComplex64>,
    reduced: Vec<f64>,
    selected_groups: Vec<u32>,
}

impl EagerNativeRuntime {
    pub(super) fn new(
        scheduler: crate::EagerExecutionRuntime,
        backend: PreparedEvaluatorBackend,
        backend_name: String,
        parameter_projection: EagerParameterProjection,
        raw_sum_groups: Vec<RawSumGroup>,
        color_contraction: Option<ColorContractionRuntime>,
    ) -> Self {
        Self {
            scheduler,
            backend,
            backend_name,
            parameter_projection,
            raw_sum_groups,
            color_contraction,
            initial_values: Vec::new(),
            momenta: Vec::new(),
            model_parameters: Vec::new(),
            amplitudes: Vec::new(),
            color_group_scratch: Vec::new(),
            reduced: Vec::new(),
            selected_groups: Vec::new(),
        }
    }

    pub(super) fn backend_name(&self) -> &str {
        &self.backend_name
    }

    pub(super) fn effective_point_tile_size(&self) -> usize {
        self.scheduler.effective_point_tile_size()
    }

    pub(super) fn workspace_bytes(&self) -> usize {
        self.scheduler.workspace_bytes()
    }

    pub(super) fn run_f64(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "eager evaluation requires at least one point",
            ));
        }
        let total_start = Instant::now();
        let point_count = batch.len();
        let state_len = point_count
            .checked_mul(common.parameter_count)
            .ok_or_else(|| RusticolError::invalid_argument("eager source state overflows"))?;
        common
            .state_scratch_f64
            .resize(state_len, Complex::new(0.0, 0.0));
        common.state_scratch_f64.fill(Complex::new(0.0, 0.0));

        let source_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state = &mut common.state_scratch_f64
                [row * common.parameter_count..(row + 1) * common.parameter_count];
            ExecutionRuntime::fill_sources_row(
                &common.sources,
                common.external_count,
                &common.particle_masses,
                row_state,
                point,
            )?;
        }
        let source_fill_s = source_start.elapsed().as_secs_f64();

        let momentum_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state = &mut common.state_scratch_f64
                [row * common.parameter_count..(row + 1) * common.parameter_count];
            ExecutionRuntime::fill_momenta_row(
                &common.momentum_slots,
                common.value_parameter_count,
                common.external_count,
                &common.external_is_initial,
                row_state,
                point,
            )?;
        }
        let momentum_setup_s = momentum_start.elapsed().as_secs_f64();

        transpose_state_components(
            &common.state_scratch_f64,
            point_count,
            common.parameter_count,
            0,
            common.value_parameter_count,
            &mut self.initial_values,
        )?;
        transpose_real_state_components(
            &common.state_scratch_f64,
            point_count,
            common.parameter_count,
            common.value_parameter_count,
            common.momentum_parameter_count,
            &mut self.momenta,
        )?;
        project_model_parameters(
            &self.parameter_projection,
            &common.model_parameter_values_f64,
            &mut self.model_parameters,
        )?;
        let amplitude_len = point_count
            .checked_mul(self.scheduler.plan().amplitude_count())
            .ok_or_else(|| RusticolError::invalid_argument("eager amplitudes overflow"))?;
        self.amplitudes
            .resize(amplitude_len, crate::EagerComplex64::new(0.0, 0.0));
        self.reduced.resize(point_count, 0.0);

        let execute_start = Instant::now();
        self.scheduler.evaluate_into(
            &mut self.backend,
            point_count,
            &self.initial_values,
            &self.momenta,
            &self.model_parameters,
            &mut self.amplitudes,
            &mut self.reduced,
        )?;
        let execute_s = execute_start.elapsed().as_secs_f64();
        for value in &mut self.reduced {
            *value *= common.normalization_factor;
        }
        let values = self.reduced.clone();
        Ok((
            values,
            RuntimeProfile {
                source_fill_s,
                momentum_setup_s,
                stage_evaluator_call_s: execute_s,
                stage_evaluator_s: execute_s,
                total_s: total_start.elapsed().as_secs_f64(),
                ..RuntimeProfile::default()
            },
        ))
    }

    pub(super) fn run_f64_profile(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "eager evaluation requires at least one point",
            ));
        }
        let total_start = Instant::now();
        let point_count = batch.len();
        let state_len = point_count
            .checked_mul(common.parameter_count)
            .ok_or_else(|| RusticolError::invalid_argument("eager source state overflows"))?;
        common
            .state_scratch_f64
            .resize(state_len, Complex::new(0.0, 0.0));
        common.state_scratch_f64.fill(Complex::new(0.0, 0.0));

        let source_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state = &mut common.state_scratch_f64
                [row * common.parameter_count..(row + 1) * common.parameter_count];
            ExecutionRuntime::fill_sources_row(
                &common.sources,
                common.external_count,
                &common.particle_masses,
                row_state,
                point,
            )?;
        }
        let source_fill_s = source_start.elapsed().as_secs_f64();

        let momentum_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state = &mut common.state_scratch_f64
                [row * common.parameter_count..(row + 1) * common.parameter_count];
            ExecutionRuntime::fill_momenta_row(
                &common.momentum_slots,
                common.value_parameter_count,
                common.external_count,
                &common.external_is_initial,
                row_state,
                point,
            )?;
        }
        let momentum_setup_s = momentum_start.elapsed().as_secs_f64();

        transpose_state_components(
            &common.state_scratch_f64,
            point_count,
            common.parameter_count,
            0,
            common.value_parameter_count,
            &mut self.initial_values,
        )?;
        transpose_real_state_components(
            &common.state_scratch_f64,
            point_count,
            common.parameter_count,
            common.value_parameter_count,
            common.momentum_parameter_count,
            &mut self.momenta,
        )?;
        project_model_parameters(
            &self.parameter_projection,
            &common.model_parameter_values_f64,
            &mut self.model_parameters,
        )?;
        let amplitude_len = point_count
            .checked_mul(self.scheduler.plan().amplitude_count())
            .ok_or_else(|| RusticolError::invalid_argument("eager amplitudes overflow"))?;
        self.amplitudes
            .resize(amplitude_len, crate::EagerComplex64::new(0.0, 0.0));
        self.reduced.resize(point_count, 0.0);

        let eager = self.scheduler.evaluate_profile_into(
            &mut self.backend,
            point_count,
            &self.initial_values,
            &self.momenta,
            &self.model_parameters,
            &mut self.amplitudes,
            &mut self.reduced,
        )?;
        for value in &mut self.reduced {
            *value *= common.normalization_factor;
        }
        let values = self.reduced.clone();
        Ok((
            values,
            RuntimeProfile {
                source_fill_s,
                momentum_setup_s,
                stage_evaluator_call_s: eager.total_s,
                stage_evaluator_s: eager.total_s,
                reduction_s: eager.reduction_s,
                total_s: total_start.elapsed().as_secs_f64(),
                eager_initialize_s: eager.initialize_s,
                eager_gather_s: eager.gather_s,
                eager_kernel_call_s: eager.kernel_call_s,
                eager_scatter_finalization_s: eager.invocation_scatter_s + eager.finalization_s,
                eager_closure_s: eager.closure_s,
                eager_reduction_s: eager.reduction_s,
                eager_copy_out_s: eager.copy_out_s,
                ..RuntimeProfile::default()
            },
        ))
    }

    pub(super) fn run_resolved_f64(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "eager evaluation requires at least one point",
            ));
        }
        let total_start = Instant::now();
        let point_count = batch.len();
        let state_len = point_count
            .checked_mul(common.parameter_count)
            .ok_or_else(|| RusticolError::invalid_argument("eager source state overflows"))?;
        common
            .state_scratch_f64
            .resize(state_len, Complex::new(0.0, 0.0));
        common.state_scratch_f64.fill(Complex::new(0.0, 0.0));

        let source_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state = &mut common.state_scratch_f64
                [row * common.parameter_count..(row + 1) * common.parameter_count];
            ExecutionRuntime::fill_sources_row(
                &common.sources,
                common.external_count,
                &common.particle_masses,
                row_state,
                point,
            )?;
        }
        let source_fill_s = source_start.elapsed().as_secs_f64();

        let momentum_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state = &mut common.state_scratch_f64
                [row * common.parameter_count..(row + 1) * common.parameter_count];
            ExecutionRuntime::fill_momenta_row(
                &common.momentum_slots,
                common.value_parameter_count,
                common.external_count,
                &common.external_is_initial,
                row_state,
                point,
            )?;
        }
        let momentum_setup_s = momentum_start.elapsed().as_secs_f64();

        transpose_state_components(
            &common.state_scratch_f64,
            point_count,
            common.parameter_count,
            0,
            common.value_parameter_count,
            &mut self.initial_values,
        )?;
        transpose_real_state_components(
            &common.state_scratch_f64,
            point_count,
            common.parameter_count,
            common.value_parameter_count,
            common.momentum_parameter_count,
            &mut self.momenta,
        )?;
        project_model_parameters(
            &self.parameter_projection,
            &common.model_parameter_values_f64,
            &mut self.model_parameters,
        )?;
        let amplitude_len = point_count
            .checked_mul(self.scheduler.plan().amplitude_count())
            .ok_or_else(|| RusticolError::invalid_argument("eager amplitudes overflow"))?;
        self.amplitudes
            .resize(amplitude_len, crate::EagerComplex64::new(0.0, 0.0));
        self.reduced.resize(point_count, 0.0);

        let physics = common.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact("resolved eager evaluation requires physics metadata")
        })?;
        let has_selected_groups = if self.scheduler.plan().has_selector_domains()
            && (selected_helicity_ids.is_some() || selected_color_ids.is_some())
        {
            fill_selected_eager_group_ids(
                &self.raw_sum_groups,
                physics,
                selected_helicity_ids,
                selected_color_ids,
                &mut self.selected_groups,
            )?;
            true
        } else {
            false
        };
        let execute_start = Instant::now();
        if has_selected_groups {
            self.scheduler.evaluate_selected_amplitudes_into(
                &mut self.backend,
                &self.selected_groups,
                point_count,
                &self.initial_values,
                &self.momenta,
                &self.model_parameters,
                &mut self.amplitudes,
            )?;
        } else {
            self.scheduler.evaluate_into(
                &mut self.backend,
                point_count,
                &self.initial_values,
                &self.momenta,
                &self.model_parameters,
                &mut self.amplitudes,
                &mut self.reduced,
            )?;
        }
        let execute_s = execute_start.elapsed().as_secs_f64();
        let reduction_start = Instant::now();
        let resolved = reduce_eager_amplitudes_resolved(
            &self.amplitudes,
            point_count,
            &self.raw_sum_groups,
            self.color_contraction.as_mut(),
            &mut self.color_group_scratch,
            physics,
            common.normalization_factor,
            selected_helicity_ids,
            selected_color_ids,
        )?;
        let reduction_s = reduction_start.elapsed().as_secs_f64();
        Ok((
            resolved,
            RuntimeProfile {
                source_fill_s,
                momentum_setup_s,
                stage_evaluator_call_s: execute_s,
                stage_evaluator_s: execute_s,
                reduction_s,
                total_s: total_start.elapsed().as_secs_f64(),
                ..RuntimeProfile::default()
            },
        ))
    }

    pub(super) fn run_resolved_f64_profile(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "eager evaluation requires at least one point",
            ));
        }
        let total_start = Instant::now();
        let point_count = batch.len();
        let state_len = point_count
            .checked_mul(common.parameter_count)
            .ok_or_else(|| RusticolError::invalid_argument("eager source state overflows"))?;
        common
            .state_scratch_f64
            .resize(state_len, Complex::new(0.0, 0.0));
        common.state_scratch_f64.fill(Complex::new(0.0, 0.0));

        let source_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state = &mut common.state_scratch_f64
                [row * common.parameter_count..(row + 1) * common.parameter_count];
            ExecutionRuntime::fill_sources_row(
                &common.sources,
                common.external_count,
                &common.particle_masses,
                row_state,
                point,
            )?;
        }
        let source_fill_s = source_start.elapsed().as_secs_f64();

        let momentum_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state = &mut common.state_scratch_f64
                [row * common.parameter_count..(row + 1) * common.parameter_count];
            ExecutionRuntime::fill_momenta_row(
                &common.momentum_slots,
                common.value_parameter_count,
                common.external_count,
                &common.external_is_initial,
                row_state,
                point,
            )?;
        }
        let momentum_setup_s = momentum_start.elapsed().as_secs_f64();

        transpose_state_components(
            &common.state_scratch_f64,
            point_count,
            common.parameter_count,
            0,
            common.value_parameter_count,
            &mut self.initial_values,
        )?;
        transpose_real_state_components(
            &common.state_scratch_f64,
            point_count,
            common.parameter_count,
            common.value_parameter_count,
            common.momentum_parameter_count,
            &mut self.momenta,
        )?;
        project_model_parameters(
            &self.parameter_projection,
            &common.model_parameter_values_f64,
            &mut self.model_parameters,
        )?;
        let amplitude_len = point_count
            .checked_mul(self.scheduler.plan().amplitude_count())
            .ok_or_else(|| RusticolError::invalid_argument("eager amplitudes overflow"))?;
        self.amplitudes
            .resize(amplitude_len, crate::EagerComplex64::new(0.0, 0.0));
        self.reduced.resize(point_count, 0.0);

        let physics = common.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact("resolved eager evaluation requires physics metadata")
        })?;
        let has_selected_groups = if self.scheduler.plan().has_selector_domains()
            && (selected_helicity_ids.is_some() || selected_color_ids.is_some())
        {
            fill_selected_eager_group_ids(
                &self.raw_sum_groups,
                physics,
                selected_helicity_ids,
                selected_color_ids,
                &mut self.selected_groups,
            )?;
            true
        } else {
            false
        };
        let eager = if has_selected_groups {
            self.scheduler.evaluate_selected_amplitudes_profile_into(
                &mut self.backend,
                &self.selected_groups,
                point_count,
                &self.initial_values,
                &self.momenta,
                &self.model_parameters,
                &mut self.amplitudes,
            )?
        } else {
            self.scheduler.evaluate_profile_into(
                &mut self.backend,
                point_count,
                &self.initial_values,
                &self.momenta,
                &self.model_parameters,
                &mut self.amplitudes,
                &mut self.reduced,
            )?
        };
        let reduction_start = Instant::now();
        let resolved = reduce_eager_amplitudes_resolved(
            &self.amplitudes,
            point_count,
            &self.raw_sum_groups,
            self.color_contraction.as_mut(),
            &mut self.color_group_scratch,
            physics,
            common.normalization_factor,
            selected_helicity_ids,
            selected_color_ids,
        )?;
        let resolved_reduction_s = reduction_start.elapsed().as_secs_f64();
        let reduction_s = eager.reduction_s + resolved_reduction_s;
        Ok((
            resolved,
            RuntimeProfile {
                source_fill_s,
                momentum_setup_s,
                stage_evaluator_call_s: eager.total_s,
                stage_evaluator_s: eager.total_s,
                reduction_s,
                total_s: total_start.elapsed().as_secs_f64(),
                eager_initialize_s: eager.initialize_s,
                eager_gather_s: eager.gather_s,
                eager_kernel_call_s: eager.kernel_call_s,
                eager_scatter_finalization_s: eager.invocation_scatter_s + eager.finalization_s,
                eager_closure_s: eager.closure_s,
                eager_reduction_s: reduction_s,
                eager_copy_out_s: eager.copy_out_s,
                ..RuntimeProfile::default()
            },
        ))
    }
}

fn fill_selected_eager_group_ids(
    groups: &[RawSumGroup],
    physics: &PhysicsRuntime,
    selected_helicity_ids: Option<&BTreeSet<String>>,
    selected_color_ids: Option<&BTreeSet<String>>,
    active: &mut Vec<u32>,
) -> RusticolResult<()> {
    let intersects = |members: &[String], selected: Option<&BTreeSet<String>>| {
        selected.is_none_or(|selected| members.iter().any(|id| selected.contains(id)))
    };
    active.clear();
    active.try_reserve_exact(groups.len()).map_err(|error| {
        RusticolError::invalid_argument(format!(
            "could not reserve eager active selector groups: {error}"
        ))
    })?;
    for group in groups {
        let reduction = physics
            .reduction_by_group_id
            .get(&group.id)
            .ok_or_else(|| {
                RusticolError::integrity(format!(
                    "resolved eager metadata is missing coherent group {}",
                    group.id
                ))
            })?;
        if intersects(&reduction.physical_helicity_ids, selected_helicity_ids)
            && intersects(&reduction.physical_color_ids, selected_color_ids)
        {
            active.push(u32::try_from(group.id).map_err(|_| {
                RusticolError::integrity(format!(
                    "eager coherent group {} does not fit the selector-domain ABI",
                    group.id
                ))
            })?);
        }
    }
    active.sort_unstable();
    active.dedup();
    Ok(())
}

fn project_model_parameters(
    projection: &EagerParameterProjection,
    runtime_parameters: &[f64],
    output: &mut Vec<crate::EagerComplex64>,
) -> RusticolResult<()> {
    output.resize(
        projection.parameter_count,
        crate::EagerComplex64::new(0.0, 0.0),
    );
    output.fill(crate::EagerComplex64::new(0.0, 0.0));
    for entry in &projection.entries {
        let real = runtime_parameters
            .get(entry.runtime_real_index)
            .copied()
            .ok_or_else(|| {
                RusticolError::integrity("eager parameter projection has an invalid real slot")
            })?;
        let imaginary = entry
            .runtime_imaginary_index
            .map(|index| {
                runtime_parameters.get(index).copied().ok_or_else(|| {
                    RusticolError::integrity(
                        "eager parameter projection has an invalid imaginary slot",
                    )
                })
            })
            .transpose()?
            .unwrap_or(0.0);
        output[entry.prepared_index] = crate::EagerComplex64::new(real, imaginary);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn reduce_eager_amplitudes_resolved(
    amplitudes: &[crate::EagerComplex64],
    point_count: usize,
    groups: &[RawSumGroup],
    contraction: Option<&mut ColorContractionRuntime>,
    color_group_scratch: &mut Vec<crate::EagerComplex64>,
    physics: &PhysicsRuntime,
    normalization_factor: f64,
    selected_helicity_ids: Option<&BTreeSet<String>>,
    selected_color_ids: Option<&BTreeSet<String>>,
) -> RusticolResult<ResolvedValues<f64>> {
    let helicity_indices = physics.selected_helicity_indices(selected_helicity_ids)?;
    let color_indices = physics.selected_color_indices(selected_color_ids)?;
    let selected_component_count = helicity_indices
        .len()
        .checked_mul(color_indices.len())
        .ok_or_else(|| RusticolError::invalid_argument("resolved eager shape overflows"))?;
    let mut values = vec![0.0; point_count * selected_component_count];
    let amplitude_count = groups
        .iter()
        .flat_map(|group| group.indices.iter().copied())
        .max()
        .map(|value| value + 1)
        .unwrap_or(0);
    if amplitudes.len() != point_count * amplitude_count {
        return Err(RusticolError::integrity(format!(
            "eager amplitude buffer has length {}, expected {}",
            amplitudes.len(),
            point_count * amplitude_count,
        )));
    }
    let helicity_position = selected_position_map(
        physics.manifest.helicities.len(),
        &helicity_indices,
        "helicity",
    )?;
    let color_position = selected_position_map(
        physics.manifest.color_components.len(),
        &color_indices,
        "color",
    )?;

    if let Some(contraction) = contraction {
        if !physics.has_contracted_color_axis()
            || physics.manifest.color_components.len() != 1
            || contraction.group_count != groups.len()
        {
            return Err(RusticolError::integrity(
                "resolved eager color contraction does not match physics metadata",
            ));
        }
        let Some(selected_color_position) = color_position[0] else {
            return Ok(ResolvedValues {
                values,
                point_count,
                helicity_indices,
                color_indices,
            });
        };
        color_group_scratch.resize(
            point_count * groups.len(),
            crate::EagerComplex64::new(0.0, 0.0),
        );
        for (group_index, group) in groups.iter().enumerate() {
            for point in 0..point_count {
                let sum = group
                    .indices
                    .iter()
                    .fold(crate::EagerComplex64::new(0.0, 0.0), |sum, index| {
                        sum + amplitudes[*index * point_count + point]
                    });
                color_group_scratch[group_index * point_count + point] = sum;
            }
        }
        for entry in &contraction.entries {
            let left_group = &groups[entry.left_group_index];
            let right_group = &groups[entry.right_group_index];
            let left_reduction = physics
                .reduction_by_group_id
                .get(&left_group.id)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "resolved eager metadata is missing coherent group {}",
                        left_group.id
                    ))
                })?;
            let right_reduction = physics
                .reduction_by_group_id
                .get(&right_group.id)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "resolved eager metadata is missing coherent group {}",
                        right_group.id
                    ))
                })?;
            if left_reduction.physical_helicity_ids != right_reduction.physical_helicity_ids {
                return Err(RusticolError::integrity(
                    "resolved eager color contraction mixes distinct physical helicities",
                ));
            }
            let left_weights = physics.normalized_helicity_weights(left_reduction)?;
            let right_weights = physics.normalized_helicity_weights(right_reduction)?;
            if left_weights.len() != right_weights.len()
                || left_weights.iter().zip(&right_weights).any(
                    |((left_index, left_weight), (right_index, right_weight))| {
                        left_index != right_index || left_weight.to_bits() != right_weight.to_bits()
                    },
                )
            {
                return Err(RusticolError::integrity(
                    "resolved eager color groups have inconsistent helicity weights",
                ));
            }
            let selected_weights = left_weights
                .into_iter()
                .filter_map(|(index, weight)| {
                    helicity_position[index].map(|position| (position, weight))
                })
                .collect::<Vec<_>>();
            if selected_weights.is_empty() {
                continue;
            }
            for point in 0..point_count {
                let left = color_group_scratch[entry.left_group_index * point_count + point];
                let right = color_group_scratch[entry.right_group_index * point_count + point];
                let product = left * right.conj();
                let contribution = normalization_factor
                    * entry.symmetry_factor
                    * (entry.weight_re * product.re - entry.weight_im * product.im);
                for (helicity_position, weight) in &selected_weights {
                    let index = point * selected_component_count
                        + *helicity_position * color_indices.len()
                        + selected_color_position;
                    values[index] += contribution * *weight;
                }
            }
        }
    } else {
        for group in groups {
            let reduction = physics
                .reduction_by_group_id
                .get(&group.id)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "resolved eager metadata is missing coherent group {}",
                        group.id
                    ))
                })?;
            let selected_members = physics
                .normalized_member_weights(reduction)?
                .into_iter()
                .filter_map(|(helicity_index, color_index, weight)| {
                    Some((
                        helicity_position[helicity_index]?,
                        color_position[color_index]?,
                        weight,
                    ))
                })
                .collect::<Vec<_>>();
            if selected_members.is_empty() {
                continue;
            }
            for point in 0..point_count {
                let sum = group
                    .indices
                    .iter()
                    .fold(crate::EagerComplex64::new(0.0, 0.0), |sum, index| {
                        sum + amplitudes[*index * point_count + point]
                    });
                let contribution = normalization_factor
                    * group.all_sector_weight
                    * (sum.re * sum.re + sum.im * sum.im);
                for (helicity_position, color_position, weight) in &selected_members {
                    let index = point * selected_component_count
                        + *helicity_position * color_indices.len()
                        + *color_position;
                    values[index] += contribution * *weight;
                }
            }
        }
    }

    Ok(ResolvedValues {
        values,
        point_count,
        helicity_indices,
        color_indices,
    })
}

fn selected_position_map(
    physical_count: usize,
    selected: &[usize],
    kind: &str,
) -> RusticolResult<Vec<Option<usize>>> {
    let mut positions = vec![None; physical_count];
    for (position, index) in selected.iter().copied().enumerate() {
        let slot = positions.get_mut(index).ok_or_else(|| {
            RusticolError::integrity(format!("resolved eager {kind} index is out of range"))
        })?;
        *slot = Some(position);
    }
    Ok(positions)
}

fn transpose_state_components(
    state: &[Complex<f64>],
    point_count: usize,
    row_width: usize,
    component_start: usize,
    component_count: usize,
    output: &mut Vec<crate::EagerComplex64>,
) -> RusticolResult<()> {
    let output_len = point_count
        .checked_mul(component_count)
        .ok_or_else(|| RusticolError::invalid_argument("eager value input overflows"))?;
    output.resize(output_len, crate::EagerComplex64::new(0.0, 0.0));
    for component in 0..component_count {
        for point in 0..point_count {
            let value = state[point * row_width + component_start + component];
            output[component * point_count + point] =
                crate::EagerComplex64::new(value.re, value.im);
        }
    }
    Ok(())
}

fn transpose_real_state_components(
    state: &[Complex<f64>],
    point_count: usize,
    row_width: usize,
    component_start: usize,
    component_count: usize,
    output: &mut Vec<f64>,
) -> RusticolResult<()> {
    let output_len = point_count
        .checked_mul(component_count)
        .ok_or_else(|| RusticolError::invalid_argument("eager momentum input overflows"))?;
    output.resize(output_len, 0.0);
    for component in 0..component_count {
        for point in 0..point_count {
            let value = state[point * row_width + component_start + component];
            if value.im != 0.0 {
                return Err(RusticolError::internal(
                    "eager momentum setup produced an imaginary component",
                ));
            }
            output[component * point_count + point] = value.re;
        }
    }
    Ok(())
}
