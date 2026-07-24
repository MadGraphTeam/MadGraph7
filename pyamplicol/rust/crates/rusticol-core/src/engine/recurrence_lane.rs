// SPDX-License-Identifier: 0BSD

//! Native adapter from public runtime inputs to Direct-Arena recurrence.

use super::recurrence_backend::NativeRecurrenceDirectExecutorOwners;
use super::*;
use crate::recurrence::direct_backend::DirectExecutorCatalog;
use crate::recurrence::direct_runtime::{
    DirectRecurrenceExecutionRuntime, DirectReplaySelectorPlan, DirectUnionHelicitySelectorPlan,
};
use crate::recurrence::{DIRECT_NONE_U32, DirectRecurrencePlan, RecurrenceStrategy};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub(super) struct RecurrenceParameterProjectionEntry {
    pub(super) runtime_slot: usize,
    pub(super) prepared_slot: usize,
    pub(super) component: u8,
}

/// One process-local Direct-Arena runtime.
///
/// `_backend_owner` keeps every immutable source/SymJIT context addressed by
/// the scheduler's lightweight function handles alive. All point-dependent
/// storage belongs to `scheduler` or the fixed-size external-momentum scratch.
pub(super) struct RecurrenceNativeRuntime {
    scheduler: DirectRecurrenceExecutionRuntime,
    _backend_owner: NativeRecurrenceDirectExecutorOwners,
    backend_name: String,
    selectors: RecurrenceNativeSelectors,
    parameter_defaults: Vec<crate::EagerComplex64>,
    parameter_projection: Vec<RecurrenceParameterProjectionEntry>,
    external_source_count: usize,
    external_momenta: Vec<f64>,
}

enum RecurrenceNativeSelectors {
    TopologyReplay {
        replay_selectors: Vec<DirectReplaySelectorPlan>,
        destination_physics_helicity_by_flow: Vec<Vec<usize>>,
    },
    AllFlowUnion {
        helicity_selectors_by_physics: Vec<Option<DirectUnionHelicitySelectorPlan>>,
        destination_by_public_flow: Vec<u32>,
    },
}

impl RecurrenceNativeRuntime {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        plan: DirectRecurrencePlan,
        executors: DirectExecutorCatalog,
        backend_owner: NativeRecurrenceDirectExecutorOwners,
        parameter_defaults: Vec<crate::EagerComplex64>,
        parameter_projection: Vec<RecurrenceParameterProjectionEntry>,
        public_flow_ids: Vec<u32>,
        direct_helicity_to_physics: Vec<usize>,
    ) -> RusticolResult<Self> {
        let parameter_count = usize::try_from(plan.parameter_value_count())
            .map_err(|_| RusticolError::artifact("recurrence parameter count exceeds usize"))?;
        if parameter_defaults.len() != parameter_count {
            return Err(RusticolError::integrity(format!(
                "recurrence prepared defaults have length {}, expected {parameter_count}",
                parameter_defaults.len()
            )));
        }
        for entry in &parameter_projection {
            if entry.prepared_slot >= parameter_count || entry.component > 1 {
                return Err(RusticolError::artifact(
                    "recurrence parameter projection is outside the prepared layout",
                ));
            }
        }
        if direct_helicity_to_physics.len() != plan.resolved_helicities().len() {
            return Err(RusticolError::integrity(
                "recurrence resolved-helicity mapping does not cover the direct plan",
            ));
        }
        let backend_name = backend_owner.summary().backend.clone();
        if public_flow_ids.is_empty() {
            return Err(RusticolError::integrity(
                "recurrence public color-flow mapping is empty",
            ));
        }
        let strategy = plan.strategy();
        let scheduler = match strategy {
            RecurrenceStrategy::TopologyReplay => {
                DirectRecurrenceExecutionRuntime::new(plan, executors, 4)?
            }
            RecurrenceStrategy::AllFlowUnion => {
                let dispatch = backend_owner.union_source_dispatch()?;
                DirectRecurrenceExecutionRuntime::new_with_union_source_dispatch(
                    plan, executors, 4, dispatch,
                )?
            }
        };
        let selectors = match strategy {
            RecurrenceStrategy::TopologyReplay => {
                let replay_selectors = public_flow_ids
                    .into_iter()
                    .map(|public_flow_id| scheduler.prepare_replay_selector(public_flow_id))
                    .collect::<RusticolResult<Vec<_>>>()?;
                let destination_physics_helicity_by_flow = replay_destination_helicity_maps(
                    scheduler.plan(),
                    &replay_selectors,
                    &direct_helicity_to_physics,
                )?;
                RecurrenceNativeSelectors::TopologyReplay {
                    replay_selectors,
                    destination_physics_helicity_by_flow,
                }
            }
            RecurrenceStrategy::AllFlowUnion => {
                let physics_helicity_count = direct_helicity_to_physics
                    .iter()
                    .copied()
                    .max()
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "all-flow-union recurrence has no public helicities",
                        )
                    })?;
                let mut helicity_selectors_by_physics = vec![None; physics_helicity_count];
                for (direct_id, physics_id) in
                    direct_helicity_to_physics.iter().copied().enumerate()
                {
                    let selector = scheduler.prepare_union_helicity_selector(
                        u32::try_from(direct_id).map_err(|_| {
                            RusticolError::artifact("recurrence direct helicity ID exceeds u32")
                        })?,
                    )?;
                    if helicity_selectors_by_physics[physics_id]
                        .replace(selector)
                        .is_some()
                    {
                        return Err(RusticolError::integrity(
                            "all-flow-union recurrence repeats a public helicity",
                        ));
                    }
                }
                if helicity_selectors_by_physics.iter().any(Option::is_none) {
                    return Err(RusticolError::integrity(
                        "all-flow-union recurrence does not cover the public helicity axis",
                    ));
                }
                let destination_by_public_flow =
                    union_destination_ids(scheduler.plan(), &public_flow_ids)?;
                RecurrenceNativeSelectors::AllFlowUnion {
                    helicity_selectors_by_physics,
                    destination_by_public_flow,
                }
            }
        };
        let external_source_count = usize::try_from(scheduler.plan().external_source_count())
            .map_err(|_| RusticolError::artifact("recurrence source count exceeds usize"))?;
        let scratch_len = scheduler
            .point_tile_size()
            .try_into()
            .ok()
            .and_then(|points: usize| points.checked_mul(external_source_count))
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| {
                RusticolError::artifact("recurrence external-momentum workspace overflows usize")
            })?;
        let external_momenta = vec![0.0; scratch_len];

        Ok(Self {
            scheduler,
            _backend_owner: backend_owner,
            backend_name,
            selectors,
            parameter_defaults,
            parameter_projection,
            external_source_count,
            external_momenta,
        })
    }

    pub(super) fn backend_name(&self) -> &str {
        &self.backend_name
    }

    pub(super) fn effective_point_tile_size(&self) -> usize {
        self.scheduler.point_tile_size() as usize
    }

    pub(super) fn run_f64(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        self.run_f64_with_global_selectors(common, batch, None, None)
    }

    pub(super) fn run_f64_with_global_selectors(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        if matches!(
            self.selectors,
            RecurrenceNativeSelectors::AllFlowUnion { .. }
        ) {
            return self.run_union_f64_with_global_selectors(
                common,
                batch,
                selected_helicities,
                selected_colors,
            );
        }
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "recurrence evaluation requires at least one point",
            ));
        }
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("recurrence execution requires physics metadata")
        })?;
        let helicity_indices = physics.selected_helicity_indices(selected_helicities)?;
        let color_indices = physics.selected_color_indices(selected_colors)?;
        self.validate_public_axes(&physics, &helicity_indices, &color_indices)?;

        let mut values = vec![0.0; batch.len()];
        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let mut momentum_setup = parameter_started.elapsed();
        let mut direct_execution = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        for color_index in color_indices {
            if !physics.color_is_computed(color_index) {
                continue;
            }
            let color_weight = physics.manifest.color_components[color_index].coefficient();
            let mut tile_start = 0usize;
            while tile_start < batch.len() {
                let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
                let flatten_started = Instant::now();
                let point_count = self.flatten_external_tile(&batch[tile_start..tile_stop])?;
                momentum_setup += flatten_started.elapsed();
                let input_len = point_count
                    .checked_mul(self.external_source_count)
                    .and_then(|values| values.checked_mul(4))
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "recurrence external-momentum tile length overflows",
                        )
                    })?;
                let execute_started = Instant::now();
                let replay_selector = match &self.selectors {
                    RecurrenceNativeSelectors::TopologyReplay {
                        replay_selectors, ..
                    } => replay_selectors.get(color_index).ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence replay selector is outside the public color axis",
                        )
                    })?,
                    RecurrenceNativeSelectors::AllFlowUnion { .. } => unreachable!(),
                };
                let output = self.scheduler.execute_replay_tile_from_external(
                    replay_selector,
                    u32::try_from(point_count).map_err(|_| {
                        RusticolError::invalid_argument(
                            "recurrence point tile exceeds the native u32 ABI",
                        )
                    })?,
                    &self.external_momenta[..input_len],
                )?;
                direct_execution += execute_started.elapsed();

                let reduction_started = Instant::now();
                let destination_physics_helicity = match &self.selectors {
                    RecurrenceNativeSelectors::TopologyReplay {
                        destination_physics_helicity_by_flow,
                        ..
                    } => destination_physics_helicity_by_flow
                        .get(color_index)
                        .ok_or_else(|| {
                            RusticolError::integrity(
                                "recurrence replay flow has no destination-helicity mapping",
                            )
                        })?,
                    RecurrenceNativeSelectors::AllFlowUnion { .. } => unreachable!(),
                };
                for destination_id in output.selected_destination_ids() {
                    let helicity_index = *destination_physics_helicity
                        .get(destination_id as usize)
                        .ok_or_else(|| {
                            RusticolError::integrity(
                                "recurrence destination-helicity mapping is incomplete",
                            )
                        })?;
                    if helicity_indices.binary_search(&helicity_index).is_err() {
                        continue;
                    }
                    let helicity = &physics.manifest.helicities[helicity_index];
                    if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0
                    {
                        continue;
                    }
                    let values_re = output.destination_re(destination_id).ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence selected amplitude destination is absent",
                        )
                    })?;
                    let values_im = output.destination_im(destination_id).ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence selected amplitude destination is absent",
                        )
                    })?;
                    let weight = helicity.coefficient * color_weight * common.normalization_factor;
                    for point in 0..point_count {
                        values[tile_start + point] += weight
                            * values_re[point]
                                .mul_add(values_re[point], values_im[point] * values_im[point]);
                    }
                }
                reduction += reduction_started.elapsed();
                tile_start = tile_stop;
            }
        }

        Ok((
            values,
            direct_profile(
                total_started.elapsed(),
                momentum_setup,
                direct_execution,
                reduction,
            ),
        ))
    }

    pub(super) fn run_resolved_f64(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if matches!(
            self.selectors,
            RecurrenceNativeSelectors::AllFlowUnion { .. }
        ) {
            return self.run_union_resolved_f64(
                common,
                batch,
                selected_helicities,
                selected_colors,
            );
        }
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "resolved recurrence evaluation requires at least one point",
            ));
        }
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("resolved recurrence execution requires physics metadata")
        })?;
        let helicity_indices = physics.selected_helicity_indices(selected_helicities)?;
        let color_indices = physics.selected_color_indices(selected_colors)?;
        self.validate_public_axes(&physics, &helicity_indices, &color_indices)?;
        let component_count = helicity_indices
            .len()
            .checked_mul(color_indices.len())
            .ok_or_else(|| {
                RusticolError::invalid_argument("recurrence resolved shape overflows")
            })?;
        let mut values = vec![
            0.0;
            batch.len().checked_mul(component_count).ok_or_else(|| {
                RusticolError::invalid_argument("recurrence resolved output overflows")
            })?
        ];
        let mut helicity_position = vec![None; physics.manifest.helicities.len()];
        for (position, index) in helicity_indices.iter().copied().enumerate() {
            helicity_position[index] = Some(position);
        }

        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let mut momentum_setup = parameter_started.elapsed();
        let mut direct_execution = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        for (color_position, color_index) in color_indices.iter().copied().enumerate() {
            if !physics.color_is_computed(color_index) {
                continue;
            }
            let color_weight = physics.manifest.color_components[color_index].coefficient();
            let mut tile_start = 0usize;
            while tile_start < batch.len() {
                let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
                let flatten_started = Instant::now();
                let point_count = self.flatten_external_tile(&batch[tile_start..tile_stop])?;
                momentum_setup += flatten_started.elapsed();
                let input_len = point_count
                    .checked_mul(self.external_source_count)
                    .and_then(|count| count.checked_mul(4))
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "recurrence external-momentum tile length overflows",
                        )
                    })?;
                let execute_started = Instant::now();
                let replay_selector = match &self.selectors {
                    RecurrenceNativeSelectors::TopologyReplay {
                        replay_selectors, ..
                    } => replay_selectors.get(color_index).ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence replay selector is outside the public color axis",
                        )
                    })?,
                    RecurrenceNativeSelectors::AllFlowUnion { .. } => unreachable!(),
                };
                let output = self.scheduler.execute_replay_tile_from_external(
                    replay_selector,
                    u32::try_from(point_count).map_err(|_| {
                        RusticolError::invalid_argument(
                            "recurrence point tile exceeds the native u32 ABI",
                        )
                    })?,
                    &self.external_momenta[..input_len],
                )?;
                direct_execution += execute_started.elapsed();

                let reduction_started = Instant::now();
                let destination_physics_helicity = match &self.selectors {
                    RecurrenceNativeSelectors::TopologyReplay {
                        destination_physics_helicity_by_flow,
                        ..
                    } => destination_physics_helicity_by_flow
                        .get(color_index)
                        .ok_or_else(|| {
                            RusticolError::integrity(
                                "recurrence replay flow has no destination-helicity mapping",
                            )
                        })?,
                    RecurrenceNativeSelectors::AllFlowUnion { .. } => unreachable!(),
                };
                for destination_id in output.selected_destination_ids() {
                    let helicity_index = *destination_physics_helicity
                        .get(destination_id as usize)
                        .ok_or_else(|| {
                            RusticolError::integrity(
                                "recurrence destination-helicity mapping is incomplete",
                            )
                        })?;
                    let Some(helicity_position) = helicity_position[helicity_index] else {
                        continue;
                    };
                    let helicity = &physics.manifest.helicities[helicity_index];
                    if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0
                    {
                        continue;
                    }
                    let values_re = output.destination_re(destination_id).ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence selected amplitude destination is absent",
                        )
                    })?;
                    let values_im = output.destination_im(destination_id).ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence selected amplitude destination is absent",
                        )
                    })?;
                    let weight = helicity.coefficient * color_weight * common.normalization_factor;
                    for point in 0..point_count {
                        let target = (tile_start + point) * component_count
                            + helicity_position * color_indices.len()
                            + color_position;
                        values[target] += weight
                            * values_re[point]
                                .mul_add(values_re[point], values_im[point] * values_im[point]);
                    }
                }
                reduction += reduction_started.elapsed();
                tile_start = tile_stop;
            }
        }

        Ok((
            ResolvedValues {
                values,
                point_count: batch.len(),
                helicity_indices,
                color_indices,
            },
            direct_profile(
                total_started.elapsed(),
                momentum_setup,
                direct_execution,
                reduction,
            ),
        ))
    }

    fn run_union_f64_with_global_selectors(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "recurrence evaluation requires at least one point",
            ));
        }
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("recurrence execution requires physics metadata")
        })?;
        let helicity_indices = physics.selected_helicity_indices(selected_helicities)?;
        let color_indices = physics.selected_color_indices(selected_colors)?;
        self.validate_public_axes(&physics, &helicity_indices, &color_indices)?;
        let color_destinations = color_indices
            .iter()
            .copied()
            .map(|color_index| Ok((color_index, self.union_destination_id(color_index)?)))
            .collect::<RusticolResult<Vec<_>>>()?;

        let mut values = vec![0.0; batch.len()];
        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let mut momentum_setup = parameter_started.elapsed();
        let mut direct_execution = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        for helicity_index in helicity_indices {
            let helicity = &physics.manifest.helicities[helicity_index];
            if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0 {
                continue;
            }
            let selector = self.union_helicity_selector(helicity_index)?;
            let mut tile_start = 0usize;
            while tile_start < batch.len() {
                let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
                let flatten_started = Instant::now();
                let point_count = self.flatten_external_tile(&batch[tile_start..tile_stop])?;
                momentum_setup += flatten_started.elapsed();
                let input_len = point_count
                    .checked_mul(self.external_source_count)
                    .and_then(|count| count.checked_mul(4))
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "recurrence external-momentum tile length overflows",
                        )
                    })?;
                let execute_started = Instant::now();
                let output = self.scheduler.execute_union_tile_from_external(
                    &selector,
                    u32::try_from(point_count).map_err(|_| {
                        RusticolError::invalid_argument(
                            "recurrence point tile exceeds the native u32 ABI",
                        )
                    })?,
                    &self.external_momenta[..input_len],
                )?;
                direct_execution += execute_started.elapsed();

                let reduction_started = Instant::now();
                for (color_index, destination_id) in color_destinations.iter().copied() {
                    if !physics.color_is_computed(color_index) {
                        continue;
                    }
                    let values_re = output.destination_re(destination_id).ok_or_else(|| {
                        RusticolError::integrity("all-flow-union amplitude destination is absent")
                    })?;
                    let values_im = output.destination_im(destination_id).ok_or_else(|| {
                        RusticolError::integrity("all-flow-union amplitude destination is absent")
                    })?;
                    let color_weight = physics.manifest.color_components[color_index].coefficient();
                    let weight = helicity.coefficient * color_weight * common.normalization_factor;
                    for point in 0..point_count {
                        values[tile_start + point] += weight
                            * values_re[point]
                                .mul_add(values_re[point], values_im[point] * values_im[point]);
                    }
                }
                reduction += reduction_started.elapsed();
                tile_start = tile_stop;
            }
        }

        Ok((
            values,
            direct_profile(
                total_started.elapsed(),
                momentum_setup,
                direct_execution,
                reduction,
            ),
        ))
    }

    fn run_union_resolved_f64(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "resolved recurrence evaluation requires at least one point",
            ));
        }
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("resolved recurrence execution requires physics metadata")
        })?;
        let helicity_indices = physics.selected_helicity_indices(selected_helicities)?;
        let color_indices = physics.selected_color_indices(selected_colors)?;
        self.validate_public_axes(&physics, &helicity_indices, &color_indices)?;
        let color_destinations = color_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(color_position, color_index)| {
                Ok((
                    color_position,
                    color_index,
                    self.union_destination_id(color_index)?,
                ))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let component_count = helicity_indices
            .len()
            .checked_mul(color_indices.len())
            .ok_or_else(|| {
                RusticolError::invalid_argument("recurrence resolved shape overflows")
            })?;
        let mut values = vec![
            0.0;
            batch.len().checked_mul(component_count).ok_or_else(|| {
                RusticolError::invalid_argument("recurrence resolved output overflows")
            })?
        ];

        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let mut momentum_setup = parameter_started.elapsed();
        let mut direct_execution = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        for (helicity_position, helicity_index) in helicity_indices.iter().copied().enumerate() {
            let helicity = &physics.manifest.helicities[helicity_index];
            if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0 {
                continue;
            }
            let selector = self.union_helicity_selector(helicity_index)?;
            let mut tile_start = 0usize;
            while tile_start < batch.len() {
                let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
                let flatten_started = Instant::now();
                let point_count = self.flatten_external_tile(&batch[tile_start..tile_stop])?;
                momentum_setup += flatten_started.elapsed();
                let input_len = point_count
                    .checked_mul(self.external_source_count)
                    .and_then(|count| count.checked_mul(4))
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "recurrence external-momentum tile length overflows",
                        )
                    })?;
                let execute_started = Instant::now();
                let output = self.scheduler.execute_union_tile_from_external(
                    &selector,
                    u32::try_from(point_count).map_err(|_| {
                        RusticolError::invalid_argument(
                            "recurrence point tile exceeds the native u32 ABI",
                        )
                    })?,
                    &self.external_momenta[..input_len],
                )?;
                direct_execution += execute_started.elapsed();

                let reduction_started = Instant::now();
                for (color_position, color_index, destination_id) in
                    color_destinations.iter().copied()
                {
                    if !physics.color_is_computed(color_index) {
                        continue;
                    }
                    let values_re = output.destination_re(destination_id).ok_or_else(|| {
                        RusticolError::integrity("all-flow-union amplitude destination is absent")
                    })?;
                    let values_im = output.destination_im(destination_id).ok_or_else(|| {
                        RusticolError::integrity("all-flow-union amplitude destination is absent")
                    })?;
                    let color_weight = physics.manifest.color_components[color_index].coefficient();
                    let weight = helicity.coefficient * color_weight * common.normalization_factor;
                    for point in 0..point_count {
                        let target = (tile_start + point) * component_count
                            + helicity_position * color_indices.len()
                            + color_position;
                        values[target] += weight
                            * values_re[point]
                                .mul_add(values_re[point], values_im[point] * values_im[point]);
                    }
                }
                reduction += reduction_started.elapsed();
                tile_start = tile_stop;
            }
        }

        Ok((
            ResolvedValues {
                values,
                point_count: batch.len(),
                helicity_indices,
                color_indices,
            },
            direct_profile(
                total_started.elapsed(),
                momentum_setup,
                direct_execution,
                reduction,
            ),
        ))
    }

    fn union_helicity_selector(
        &self,
        physics_helicity_index: usize,
    ) -> RusticolResult<DirectUnionHelicitySelectorPlan> {
        match &self.selectors {
            RecurrenceNativeSelectors::AllFlowUnion {
                helicity_selectors_by_physics,
                ..
            } => helicity_selectors_by_physics
                .get(physics_helicity_index)
                .copied()
                .flatten()
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "all-flow-union helicity is outside retained public coverage",
                    )
                }),
            RecurrenceNativeSelectors::TopologyReplay { .. } => Err(RusticolError::integrity(
                "union selector requested from topology replay",
            )),
        }
    }

    fn union_destination_id(&self, color_index: usize) -> RusticolResult<u32> {
        match &self.selectors {
            RecurrenceNativeSelectors::AllFlowUnion {
                destination_by_public_flow,
                ..
            } => destination_by_public_flow
                .get(color_index)
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "all-flow-union color is outside retained public coverage",
                    )
                }),
            RecurrenceNativeSelectors::TopologyReplay { .. } => Err(RusticolError::integrity(
                "union destination requested from topology replay",
            )),
        }
    }

    fn prepare_parameters(&mut self, common: &ExecutionRuntime) -> RusticolResult<()> {
        let (parameters_re, parameters_im) = self.scheduler.parameters_mut();
        if parameters_re.len() != self.parameter_defaults.len()
            || parameters_im.len() != self.parameter_defaults.len()
        {
            return Err(RusticolError::integrity(
                "recurrence prepared parameter workspace has the wrong size",
            ));
        }
        for ((real, imaginary), default) in parameters_re
            .iter_mut()
            .zip(parameters_im.iter_mut())
            .zip(&self.parameter_defaults)
        {
            *real = default.re;
            *imaginary = default.im;
        }
        for entry in &self.parameter_projection {
            let value = common
                .model_parameter_values_f64
                .get(entry.runtime_slot)
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "recurrence runtime parameter projection is out of range",
                    )
                })?;
            if entry.component == 0 {
                parameters_re[entry.prepared_slot] = value;
            } else {
                parameters_im[entry.prepared_slot] = value;
            }
        }
        Ok(())
    }

    fn flatten_external_tile(&mut self, batch: &[Vec<[f64; 4]>]) -> RusticolResult<usize> {
        let point_count = batch.len();
        if point_count > self.effective_point_tile_size() {
            return Err(RusticolError::invalid_argument(
                "recurrence point tile exceeds its persistent workspace",
            ));
        }
        for (point_index, point) in batch.iter().enumerate() {
            if point.len() != self.external_source_count {
                return Err(RusticolError::invalid_argument(format!(
                    "recurrence point has {} external momenta, expected {}",
                    point.len(),
                    self.external_source_count
                )));
            }
            for (source_slot, momentum) in point.iter().enumerate() {
                let start =
                    (point_index * self.external_source_count + source_slot) * momentum.len();
                self.external_momenta[start..start + 4].copy_from_slice(momentum);
            }
        }
        Ok(point_count)
    }

    fn validate_public_axes(
        &self,
        physics: &PhysicsRuntime,
        helicity_indices: &[usize],
        color_indices: &[usize],
    ) -> RusticolResult<()> {
        let (color_count, helicity_count) = match &self.selectors {
            RecurrenceNativeSelectors::TopologyReplay {
                replay_selectors, ..
            } => (replay_selectors.len(), physics.manifest.helicities.len()),
            RecurrenceNativeSelectors::AllFlowUnion {
                helicity_selectors_by_physics,
                destination_by_public_flow,
            } => (
                destination_by_public_flow.len(),
                helicity_selectors_by_physics.len(),
            ),
        };
        if color_count != physics.manifest.color_components.len()
            || helicity_count != physics.manifest.helicities.len()
        {
            return Err(RusticolError::integrity(
                "recurrence selectors do not cover the public physics axes",
            ));
        }
        if color_indices.iter().any(|index| *index >= color_count)
            || helicity_indices
                .iter()
                .any(|index| *index >= helicity_count)
        {
            return Err(RusticolError::integrity(
                "recurrence selector mapping is outside the public physics axes",
            ));
        }
        Ok(())
    }
}

fn union_destination_ids(
    plan: &DirectRecurrencePlan,
    public_flow_ids: &[u32],
) -> RusticolResult<Vec<u32>> {
    if plan.strategy() != RecurrenceStrategy::AllFlowUnion {
        return Err(RusticolError::integrity(
            "union destination mapping requires an all-flow-union plan",
        ));
    }
    let mut destination_by_sector = BTreeMap::new();
    for destination in plan.amplitude_destinations() {
        if destination.target_helicity_id_or_sentinel != DIRECT_NONE_U32 {
            return Err(RusticolError::integrity(
                "all-flow-union amplitude destination fixes a numerical helicity",
            ));
        }
        if destination_by_sector
            .insert(destination.target_sector_id, destination.id)
            .is_some()
        {
            return Err(RusticolError::integrity(
                "all-flow-union repeats a physical-flow destination",
            ));
        }
    }
    let result = public_flow_ids
        .iter()
        .map(|sector_id| {
            destination_by_sector
                .get(sector_id)
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "all-flow-union public flow has no amplitude destination",
                    )
                })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    if result.len() != destination_by_sector.len() {
        return Err(RusticolError::integrity(
            "all-flow-union amplitude destinations do not match the public flow axis",
        ));
    }
    Ok(result)
}

fn replay_destination_helicity_maps(
    plan: &DirectRecurrencePlan,
    replay_selectors: &[DirectReplaySelectorPlan],
    direct_helicity_to_physics: &[usize],
) -> RusticolResult<Vec<Vec<usize>>> {
    let source_count = usize::try_from(plan.external_source_count())
        .map_err(|_| RusticolError::artifact("recurrence source count exceeds usize"))?;
    let mut physics_by_public_helicity = BTreeMap::<Vec<i32>, usize>::new();
    for descriptor in plan.resolved_helicities() {
        let start = usize::try_from(descriptor.public_helicity_start).map_err(|_| {
            RusticolError::integrity("recurrence public-helicity range start exceeds usize")
        })?;
        let count = usize::try_from(descriptor.public_helicity_count).map_err(|_| {
            RusticolError::integrity("recurrence public-helicity count exceeds usize")
        })?;
        if count != source_count {
            return Err(RusticolError::integrity(format!(
                "recurrence resolved helicity {} covers {count} public sources, expected {source_count}",
                descriptor.id
            )));
        }
        let end = start.checked_add(count).ok_or_else(|| {
            RusticolError::integrity("recurrence public-helicity range overflows usize")
        })?;
        let public_helicities = plan.public_helicities().get(start..end).ok_or_else(|| {
            RusticolError::integrity("recurrence public-helicity range is out of bounds")
        })?;
        let physics_index = direct_helicity_to_physics
            .get(descriptor.id as usize)
            .copied()
            .ok_or_else(|| {
                RusticolError::integrity("recurrence direct helicity has no public physics mapping")
            })?;
        if physics_by_public_helicity
            .insert(public_helicities.to_vec(), physics_index)
            .is_some()
        {
            return Err(RusticolError::integrity(
                "recurrence direct plan contains duplicate public helicity assignments",
            ));
        }
    }

    let mut result = Vec::with_capacity(replay_selectors.len());
    let mut public = vec![0_i32; source_count];
    let mut assigned = vec![false; source_count];
    for selector in replay_selectors {
        let mut direct_to_physics = vec![usize::MAX; plan.resolved_helicities().len()];
        for descriptor in plan.resolved_helicities() {
            let start = usize::try_from(descriptor.public_helicity_start).map_err(|_| {
                RusticolError::integrity("recurrence public-helicity range start exceeds usize")
            })?;
            let end = start.checked_add(source_count).ok_or_else(|| {
                RusticolError::integrity("recurrence public-helicity range overflows usize")
            })?;
            let representative = plan.public_helicities().get(start..end).ok_or_else(|| {
                RusticolError::integrity("recurrence public-helicity range is out of bounds")
            })?;
            map_representative_helicity_to_public(
                representative,
                selector.source_permutation(),
                &mut public,
                &mut assigned,
            )?;
            direct_to_physics[descriptor.id as usize] = physics_by_public_helicity
                .get(public.as_slice())
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "recurrence replay flow {} maps a representative helicity outside retained public coverage",
                        selector.public_flow_id()
                    ))
                })?;
        }

        let mut destinations = Vec::with_capacity(plan.amplitude_destinations().len());
        for destination in plan.amplitude_destinations() {
            if destination.target_helicity_id_or_sentinel == DIRECT_NONE_U32 {
                return Err(RusticolError::integrity(
                    "topology-replay amplitude destination lacks a resolved helicity",
                ));
            }
            let physics_index = direct_to_physics
                .get(destination.target_helicity_id_or_sentinel as usize)
                .copied()
                .filter(|index| *index != usize::MAX)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "recurrence amplitude destination helicity is not publicly mapped",
                    )
                })?;
            destinations.push(physics_index);
        }
        result.push(destinations);
    }
    Ok(result)
}

fn map_representative_helicity_to_public(
    representative: &[i32],
    representative_to_public: &[u32],
    public: &mut [i32],
    assigned: &mut [bool],
) -> RusticolResult<()> {
    let source_count = representative.len();
    if representative_to_public.len() != source_count
        || public.len() != source_count
        || assigned.len() != source_count
    {
        return Err(RusticolError::integrity(
            "recurrence replay permutation has inconsistent source coverage",
        ));
    }
    assigned.fill(false);
    for (representative_slot, helicity) in representative.iter().copied().enumerate() {
        let public_slot = usize::try_from(representative_to_public[representative_slot])
            .map_err(|_| RusticolError::integrity("recurrence replay source slot exceeds usize"))?;
        if public_slot >= source_count || assigned[public_slot] {
            return Err(RusticolError::integrity(
                "recurrence replay permutation is not bijective",
            ));
        }
        public[public_slot] = helicity;
        assigned[public_slot] = true;
    }
    if assigned.contains(&false) {
        return Err(RusticolError::integrity(
            "recurrence replay permutation does not cover every source",
        ));
    }
    Ok(())
}

fn direct_profile(
    total: Duration,
    momentum_setup: Duration,
    direct_execution: Duration,
    reduction: Duration,
) -> RuntimeProfile {
    RuntimeProfile {
        momentum_setup_s: profile_duration_seconds(momentum_setup),
        stage_evaluator_call_s: profile_duration_seconds(direct_execution),
        stage_evaluator_s: profile_duration_seconds(direct_execution),
        reduction_s: profile_duration_seconds(reduction),
        total_s: profile_duration_seconds(total),
        ..RuntimeProfile::default()
    }
}

#[cfg(test)]
mod tests {
    use super::map_representative_helicity_to_public;

    #[test]
    fn replay_helicity_mapping_uses_representative_to_public_direction() {
        let mut public = vec![0; 3];
        let mut assigned = vec![false; 3];

        map_representative_helicity_to_public(
            &[10, 20, 30],
            &[1, 2, 0],
            &mut public,
            &mut assigned,
        )
        .expect("three-cycle replay permutation should be valid");

        assert_eq!(public, vec![30, 10, 20]);
        assert_eq!(assigned, vec![true, true, true]);
    }
}
