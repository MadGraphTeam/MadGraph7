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
    source_schedule: Option<EagerSourceSchedule>,
    source_wavefunction_scratch: Vec<Complex<f64>>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct EagerSourceKey {
    source_kind: String,
    leg_label: usize,
    family: u8,
    dimension: usize,
    pdg_label: i32,
    anti_pdg_label: i32,
    orientation: u8,
    momentum_transform: u8,
    phase_real: u64,
    phase_imaginary: u64,
    source_helicity: i32,
    chirality: i32,
}

#[derive(Debug)]
struct EagerSourceClass {
    representative_index: usize,
    target_component_starts: Vec<usize>,
    dimension: usize,
}

#[derive(Debug)]
struct EagerSourceSchedule {
    classes: Vec<EagerSourceClass>,
    maximum_dimension: usize,
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
            source_schedule: None,
            source_wavefunction_scratch: Vec::new(),
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
        let (source_fill_s, momentum_setup_s) = prepare_eager_inputs(
            common,
            batch,
            &mut self.initial_values,
            &mut self.momenta,
            &mut self.source_schedule,
            &mut self.source_wavefunction_scratch,
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
        let (source_fill_s, momentum_setup_s) = prepare_eager_inputs(
            common,
            batch,
            &mut self.initial_values,
            &mut self.momenta,
            &mut self.source_schedule,
            &mut self.source_wavefunction_scratch,
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
        validate_eager_profile_accounting(&eager)?;
        for value in &mut self.reduced {
            *value *= common.normalization_factor;
        }
        let values = self.reduced.clone();
        Ok((
            values,
            RuntimeProfile {
                source_fill_s,
                momentum_setup_s,
                stage_evaluator_call_s: eager.total.as_secs_f64(),
                stage_evaluator_s: eager.total.as_secs_f64(),
                reduction_s: eager.reduction.as_secs_f64(),
                total_s: total_start.elapsed().as_secs_f64(),
                eager_initialize_s: eager.initialize.as_secs_f64(),
                eager_gather_s: eager.gather.as_secs_f64(),
                eager_kernel_call_s: eager.kernel_call.as_secs_f64(),
                eager_invocation_scatter_s: eager.invocation_scatter.as_secs_f64(),
                eager_finalization_s: eager.finalization.as_secs_f64(),
                eager_scatter_finalization_s: (eager.invocation_scatter + eager.finalization)
                    .as_secs_f64(),
                eager_closure_s: eager.closure.as_secs_f64(),
                eager_reduction_s: eager.reduction.as_secs_f64(),
                eager_copy_out_s: eager.copy_out.as_secs_f64(),
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
        let (source_fill_s, momentum_setup_s) = prepare_eager_inputs(
            common,
            batch,
            &mut self.initial_values,
            &mut self.momenta,
            &mut self.source_schedule,
            &mut self.source_wavefunction_scratch,
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
            self.scheduler.evaluate_selected_active_amplitudes_into(
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
        let (source_fill_s, momentum_setup_s) = prepare_eager_inputs(
            common,
            batch,
            &mut self.initial_values,
            &mut self.momenta,
            &mut self.source_schedule,
            &mut self.source_wavefunction_scratch,
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
        validate_eager_profile_accounting(&eager)?;
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
        let reduction_s = eager.reduction.as_secs_f64() + resolved_reduction_s;
        Ok((
            resolved,
            RuntimeProfile {
                source_fill_s,
                momentum_setup_s,
                stage_evaluator_call_s: eager.total.as_secs_f64(),
                stage_evaluator_s: eager.total.as_secs_f64(),
                reduction_s,
                total_s: total_start.elapsed().as_secs_f64(),
                eager_initialize_s: eager.initialize.as_secs_f64(),
                eager_gather_s: eager.gather.as_secs_f64(),
                eager_kernel_call_s: eager.kernel_call.as_secs_f64(),
                eager_invocation_scatter_s: eager.invocation_scatter.as_secs_f64(),
                eager_finalization_s: eager.finalization.as_secs_f64(),
                eager_scatter_finalization_s: (eager.invocation_scatter + eager.finalization)
                    .as_secs_f64(),
                eager_closure_s: eager.closure.as_secs_f64(),
                eager_reduction_s: reduction_s,
                eager_copy_out_s: eager.copy_out.as_secs_f64(),
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

fn prepare_eager_inputs(
    common: &mut ExecutionRuntime,
    batch: &[Vec<[f64; 4]>],
    initial_values: &mut Vec<crate::EagerComplex64>,
    momenta: &mut Vec<f64>,
    source_schedule: &mut Option<EagerSourceSchedule>,
    source_wavefunction_scratch: &mut Vec<Complex<f64>>,
) -> RusticolResult<(f64, f64)> {
    let point_count = batch.len();
    let value_len = point_count
        .checked_mul(common.value_parameter_count)
        .ok_or_else(|| RusticolError::invalid_argument("eager value input overflows"))?;
    let momentum_len = point_count
        .checked_mul(common.momentum_parameter_count)
        .ok_or_else(|| RusticolError::invalid_argument("eager momentum input overflows"))?;
    initial_values.resize(value_len, crate::EagerComplex64::new(0.0, 0.0));
    momenta.resize(momentum_len, 0.0);

    if source_schedule.is_none() {
        *source_schedule = Some(build_eager_source_schedule(
            &common.sources,
            common.value_parameter_count,
        )?);
    }
    let source_schedule = source_schedule
        .as_ref()
        .expect("eager source schedule was initialized above");
    source_wavefunction_scratch.resize(source_schedule.maximum_dimension, Complex::new(0.0, 0.0));

    let source_started = Instant::now();
    for (point_index, point) in batch.iter().enumerate() {
        for class in &source_schedule.classes {
            let source = &common.sources[class.representative_index];
            let wavefunction = &mut source_wavefunction_scratch[..class.dimension];
            ExecutionRuntime::write_source_wavefunction(
                source,
                common.external_count,
                &common.particle_masses,
                point,
                wavefunction,
            )?;
            for target_start in &class.target_component_starts {
                for (component, value) in wavefunction.iter().enumerate() {
                    initial_values[(target_start + component) * point_count + point_index] =
                        crate::EagerComplex64::new(value.re, value.im);
                }
            }
        }
    }
    let source_fill_s = source_started.elapsed().as_secs_f64();

    let momentum_started = Instant::now();
    for (point_index, point) in batch.iter().enumerate() {
        for slot in &common.momentum_slots {
            let start = slot.component_start;
            let stop = slot.component_stop;
            if stop > common.momentum_parameter_count || stop < start || stop - start != 4 {
                return Err(RusticolError::invalid_argument(format!(
                    "generic momentum slot {} has an invalid component range",
                    slot.momentum_slot_id
                )));
            }
            let mut momentum = [0.0; 4];
            for label in &slot.external_labels {
                let index = label.checked_sub(1).ok_or_else(|| {
                    RusticolError::invalid_argument("generic momentum labels are one-based")
                })?;
                if index >= common.external_count || index >= common.external_is_initial.len() {
                    return Err(RusticolError::invalid_argument(format!(
                        "generic momentum slot {} refers to unknown external label {}",
                        slot.momentum_slot_id, label
                    )));
                }
                let sign = if common.external_is_initial[index] {
                    -1.0
                } else {
                    1.0
                };
                for (output, input) in momentum.iter_mut().zip(&point[index]) {
                    *output += sign * input;
                }
            }
            for (component, value) in momentum.into_iter().enumerate() {
                momenta[(start + component) * point_count + point_index] = value;
            }
        }
    }
    Ok((source_fill_s, momentum_started.elapsed().as_secs_f64()))
}

fn build_eager_source_schedule(
    sources: &[GenericSourceRecordManifest],
    value_component_count: usize,
) -> RusticolResult<EagerSourceSchedule> {
    let mut class_by_key: BTreeMap<EagerSourceKey, usize> = BTreeMap::new();
    let mut classes: Vec<EagerSourceClass> = Vec::new();
    let mut maximum_dimension = 0usize;
    for (source_index, source) in sources.iter().enumerate() {
        let start = source.value_slot.component_start;
        let stop = source.value_slot.component_stop;
        if stop > value_component_count || stop < start || stop - start != source.dimension {
            return Err(RusticolError::artifact(format!(
                "eager source {} has an invalid value-slot range",
                source.source_id
            )));
        }
        let key = eager_source_key(source);
        if let Some(class_index) = class_by_key.get(&key).copied() {
            let class = &mut classes[class_index];
            if class.dimension != source.dimension {
                return Err(RusticolError::internal(
                    "equivalent eager sources have inconsistent dimensions",
                ));
            }
            class.target_component_starts.push(start);
        } else {
            let class_index = classes.len();
            class_by_key.insert(key, class_index);
            classes.push(EagerSourceClass {
                representative_index: source_index,
                target_component_starts: vec![start],
                dimension: source.dimension,
            });
        }
        maximum_dimension = maximum_dimension.max(source.dimension);
    }
    Ok(EagerSourceSchedule {
        classes,
        maximum_dimension,
    })
}

fn eager_source_key(source: &GenericSourceRecordManifest) -> EagerSourceKey {
    EagerSourceKey {
        source_kind: source.source_kind.clone(),
        leg_label: source.leg_label,
        family: match source.source_ir.wavefunction_family {
            GenericWavefunctionFamilyManifest::Scalar => 0,
            GenericWavefunctionFamilyManifest::Fermion => 1,
            GenericWavefunctionFamilyManifest::Vector => 2,
            GenericWavefunctionFamilyManifest::Spin2 => 3,
            GenericWavefunctionFamilyManifest::Ghost => 4,
            GenericWavefunctionFamilyManifest::Auxiliary => 5,
        },
        dimension: source.source_ir.component_dimension,
        pdg_label: source.source_ir.identity.pdg_label,
        anti_pdg_label: source.source_ir.identity.anti_pdg_label,
        orientation: match source.source_ir.identity.orientation {
            GenericSourceOrientationManifest::Particle => 0,
            GenericSourceOrientationManifest::Antiparticle => 1,
            GenericSourceOrientationManifest::SelfConjugate => 2,
        },
        momentum_transform: match source.applied_crossing.momentum_transform {
            GenericMomentumTransformManifest::Identity => 0,
            GenericMomentumTransformManifest::NegateFourMomentum => 1,
        },
        phase_real: source.applied_crossing.phase[0].to_bits(),
        phase_imaginary: source.applied_crossing.phase[1].to_bits(),
        source_helicity: source.source_helicity,
        chirality: source.chirality,
    }
}

fn validate_eager_profile_accounting(
    profile: &crate::eager_runtime::EagerExecutionProfile,
) -> RusticolResult<()> {
    let accounted = profile.accounted();
    let tolerance = Duration::from_nanos(1);
    if accounted > profile.total.saturating_add(tolerance) {
        return Err(RusticolError::internal(format!(
            "eager runtime profile phases account for {accounted_s:.9e}s, exceeding the aggregate {total_s:.9e}s: {profile:?}",
            accounted_s = accounted.as_secs_f64(),
            total_s = profile.total.as_secs_f64(),
        )));
    }
    Ok(())
}

#[inline(always)]
#[cfg(test)]
fn store_complex_row_component_major(
    row: &[Complex<f64>],
    point_index: usize,
    point_count: usize,
    output: &mut [crate::EagerComplex64],
) {
    debug_assert!(point_index < point_count);
    debug_assert_eq!(output.len(), row.len() * point_count);
    for (component, value) in row.iter().enumerate() {
        output[component * point_count + point_index] =
            crate::EagerComplex64::new(value.re, value.im);
    }
}

#[inline(always)]
#[cfg(test)]
fn store_real_row_component_major(
    row: &[Complex<f64>],
    point_index: usize,
    point_count: usize,
    output: &mut [f64],
) -> RusticolResult<()> {
    debug_assert!(point_index < point_count);
    debug_assert_eq!(output.len(), row.len() * point_count);
    for (component, value) in row.iter().enumerate() {
        if value.im != 0.0 {
            return Err(RusticolError::internal(
                "eager momentum setup produced an imaginary component",
            ));
        }
        output[component * point_count + point_index] = value.re;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eager_input_rows_are_stored_component_major() {
        let point_count = 2;
        let mut complex = vec![crate::EagerComplex64::new(0.0, 0.0); 6];
        let mut real = vec![0.0; 6];
        for (point, values) in [
            [
                Complex::new(1.0, -1.0),
                Complex::new(2.0, -2.0),
                Complex::new(3.0, -3.0),
            ],
            [
                Complex::new(4.0, -4.0),
                Complex::new(5.0, -5.0),
                Complex::new(6.0, -6.0),
            ],
        ]
        .iter()
        .enumerate()
        {
            store_complex_row_component_major(values, point, point_count, &mut complex);
            let real_values = values.map(|value| Complex::new(value.re, 0.0));
            store_real_row_component_major(&real_values, point, point_count, &mut real).unwrap();
        }
        assert_eq!(
            complex,
            [
                crate::EagerComplex64::new(1.0, -1.0),
                crate::EagerComplex64::new(4.0, -4.0),
                crate::EagerComplex64::new(2.0, -2.0),
                crate::EagerComplex64::new(5.0, -5.0),
                crate::EagerComplex64::new(3.0, -3.0),
                crate::EagerComplex64::new(6.0, -6.0),
            ]
        );
        assert_eq!(real, [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn eager_real_input_rejects_imaginary_components() {
        let error = store_real_row_component_major(&[Complex::new(1.0, 1.0)], 0, 1, &mut [0.0])
            .unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Internal);
    }

    #[test]
    fn eager_profile_rejects_impossible_phase_totals() {
        let valid = crate::eager_runtime::EagerExecutionProfile {
            initialize: Duration::from_millis(2),
            gather: Duration::from_millis(3),
            kernel_call: Duration::from_millis(4),
            total: Duration::from_millis(10),
            ..crate::eager_runtime::EagerExecutionProfile::default()
        };
        validate_eager_profile_accounting(&valid).unwrap();

        let impossible = crate::eager_runtime::EagerExecutionProfile {
            invocation_scatter: Duration::from_millis(20),
            total: Duration::from_millis(10),
            ..crate::eager_runtime::EagerExecutionProfile::default()
        };
        let error = validate_eager_profile_accounting(&impossible).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Internal);
        assert!(error.to_string().contains("exceeding the aggregate"));
    }
}
