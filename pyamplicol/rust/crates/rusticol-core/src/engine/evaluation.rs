// SPDX-License-Identifier: 0BSD

use super::*;

impl ExecutionRuntime {
    pub(super) fn set_lc_sector_selector(&mut self, sector_id: Option<i64>) -> Option<f64> {
        let index = *self
            .model_parameter_name_to_index
            .get(LC_SECTOR_SELECTOR_PARAMETER)?;
        let previous = *self.model_parameter_values_f64.get(index)?;
        if let Some(value) = self.model_parameter_values_f64.get_mut(index) {
            *value = sector_id.map(|id| id as f64).unwrap_or(-1.0);
        }
        Some(previous)
    }

    pub(super) fn restore_lc_sector_selector(&mut self, previous: Option<f64>) {
        let Some(previous) = previous else {
            return;
        };
        let Some(index) = self
            .model_parameter_name_to_index
            .get(LC_SECTOR_SELECTOR_PARAMETER)
            .copied()
        else {
            return;
        };
        if let Some(value) = self.model_parameter_values_f64.get_mut(index) {
            *value = previous;
        }
    }

    #[allow(dead_code)] // Allocating compatibility wrapper around the into lane.
    pub(super) fn run_f64_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
    ) -> RusticolResult<Vec<f64>> {
        let mut output = vec![0.0; batch.point_count()];
        self.run_f64_into_unprofiled(batch, &mut output)?;
        Ok(output)
    }

    pub(super) fn run_f64_into_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        self.run_f64_selected_into_unprofiled(batch, None, None, output)
    }

    pub(super) fn run_f64_selected_into_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        if output.len() != batch.point_count() {
            return Err(RusticolError::invalid_argument(format!(
                "evaluation output has length {}, expected {}",
                output.len(),
                batch.point_count()
            )));
        }
        if selected_helicity_ids.is_none()
            && let Some(sum_runtime) = self.helicity_sum_runtime.as_mut()
        {
            return sum_runtime.run_f64_selected_into_unprofiled(
                batch,
                None,
                selected_color_ids,
                output,
            );
        }
        if self.lc_topology_replay_enabled {
            if selected_color_ids.is_some_and(|ids| ids.len() == 1)
                && self.try_run_f64_with_single_lc_topology_replay_into_unprofiled(
                    batch,
                    selected_helicity_ids,
                    selected_color_ids,
                    output,
                )?
            {
                return Ok(());
            }
            let resolved = self.run_resolved_f64_with_lc_topology_replay_unprofiled(
                batch,
                selected_helicity_ids,
                selected_color_ids,
            )?;
            return write_resolved_f64_totals(&resolved, output);
        }
        if self.has_compiled_helicity_execution_plan() {
            if self.try_run_f64_with_helicity_recurrence_into_unprofiled(
                batch,
                selected_helicity_ids,
                selected_color_ids,
                output,
            )? {
                return Ok(());
            }
            let resolved = self.run_resolved_f64_with_helicity_recurrence_unprofiled(
                batch,
                selected_helicity_ids,
                selected_color_ids,
                None,
            )?;
            return write_resolved_f64_totals(&resolved, output);
        }
        if selected_helicity_ids.is_some() || selected_color_ids.is_some() {
            let resolved =
                self.run_resolved_f64_unprofiled(batch, selected_helicity_ids, selected_color_ids)?;
            return write_resolved_f64_totals(&resolved, output);
        }
        let previous = self.set_lc_sector_selector(None);
        let result = self.run_f64_materialized_selected_into_unprofiled(batch, None, output);
        self.restore_lc_sector_selector(previous);
        result
    }

    pub(super) fn run_resolved_f64_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<ResolvedValues<f64>> {
        if selected_helicity_ids.is_none()
            && let Some(sum_runtime) = self.helicity_sum_runtime.as_mut()
        {
            return sum_runtime.run_resolved_f64_unprofiled(batch, None, selected_color_ids);
        }
        if self.lc_topology_replay_enabled {
            return self.run_resolved_f64_with_lc_topology_replay_unprofiled(
                batch,
                selected_helicity_ids,
                selected_color_ids,
            );
        }
        if self.has_compiled_helicity_execution_plan() {
            return self.run_resolved_f64_with_helicity_recurrence_unprofiled(
                batch,
                selected_helicity_ids,
                selected_color_ids,
                None,
            );
        }
        let physics = self.physics.clone().ok_or_else(|| {
            RusticolError::invalid_argument(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        let selected_color_lane = if selected_color_ids.is_some_and(|ids| ids.len() == 1) {
            self.lc_materialized_sector_ids_for_color_ids(
                physics.as_ref(),
                selected_color_ids.expect("singleton selection checked"),
            )?
            .into_iter()
            .next()
        } else {
            None
        };
        if let Some(sector_id) = selected_color_lane
            && let Some(selector_runtime) = self.color_selector_runtimes.get_mut(&sector_id)
        {
            return selector_runtime.run_resolved_f64_unprofiled(
                batch,
                selected_helicity_ids,
                selected_color_ids,
            );
        }
        self.run_f64_materialized_selected_for_resolved_unprofiled(batch, None)?;
        self.amplitude_stage
            .as_mut()
            .expect("generic amplitude stage checked")
            .reduce_scratch_f64_resolved(
                batch.point_count(),
                &physics,
                self.normalization_factor,
                selected_helicity_ids,
                selected_color_ids,
            )
    }

    fn run_resolved_f64_with_lc_topology_replay_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<ResolvedValues<f64>> {
        let physics = self.physics.clone().ok_or_else(|| {
            RusticolError::invalid_argument(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        let replay_plan = self.cached_lc_resolved_replay_plan(&physics)?;
        let selection = self.cached_lc_resolved_replay_selection(
            &physics,
            replay_plan.as_ref(),
            selected_helicity_ids,
            selected_color_ids,
        )?;
        let n_points = batch.point_count();
        let component_count = selection.helicity_indices.len() * selection.color_indices.len();
        let mut full_values = vec![0.0; n_points * component_count];
        let mappings = self.lc_topology_replay_mappings.clone();
        let mappings_per_chunk = replay_mappings_per_expanded_batch(n_points);
        for source_group in &selection.source_groups {
            let materialized_sector_ids =
                self.lc_materialized_sector_ids_for_color_ids(&physics, &source_group.color_ids)?;
            for chunk_start in (0..source_group.mapping_indices.len()).step_by(mappings_per_chunk) {
                let chunk_end = usize::min(
                    chunk_start + mappings_per_chunk,
                    source_group.mapping_indices.len(),
                );
                let mapping_chunk = source_group.mapping_indices[chunk_start..chunk_end]
                    .iter()
                    .map(|mapping_index| mappings[*mapping_index].clone())
                    .collect::<Vec<_>>();
                let entry_chunk = &source_group.entries[chunk_start..chunk_end];
                let expanded_batch = if mapping_chunk.len() == 1 && mapping_chunk[0].is_empty() {
                    None
                } else {
                    Some(apply_lc_topology_label_permutations_from_view(
                        batch,
                        self.external_count,
                        &mapping_chunk,
                    )?)
                };
                let evaluation_view = if let Some(expanded_batch) = expanded_batch.as_deref() {
                    F64MomentumBatchView::from_nested(expanded_batch, self.external_count)?
                } else {
                    batch
                };
                let selected_lane_id = (materialized_sector_ids.len() == 1)
                    .then(|| materialized_sector_ids.iter().next().copied())
                    .flatten();
                let materialized = if self.has_compiled_helicity_execution_plan()
                    && selected_helicity_ids.is_some()
                {
                    self.run_resolved_f64_with_helicity_recurrence_unprofiled(
                        evaluation_view,
                        Some(&source_group.helicity_ids),
                        Some(&source_group.color_ids),
                        Some(&materialized_sector_ids),
                    )?
                } else if let Some(selector_runtime) = selected_lane_id
                    .and_then(|sector_id| self.color_selector_runtimes.get_mut(&sector_id))
                {
                    selector_runtime.run_resolved_f64_unprofiled(
                        evaluation_view,
                        Some(&source_group.helicity_ids),
                        Some(&source_group.color_ids),
                    )?
                } else {
                    self.run_f64_materialized_selected_for_resolved_unprofiled(
                        evaluation_view,
                        Some(&materialized_sector_ids),
                    )?;
                    self.amplitude_stage
                        .as_mut()
                        .expect("generic amplitude stage checked")
                        .reduce_scratch_f64_resolved(
                            evaluation_view.point_count(),
                            &physics,
                            self.normalization_factor,
                            Some(&source_group.helicity_ids),
                            Some(&source_group.color_ids),
                        )?
                };
                accumulate_selected_lc_replay_resolved_f64(
                    &mut full_values,
                    n_points,
                    &materialized,
                    entry_chunk,
                    source_group.source_component_count,
                    component_count,
                )?;
            }
        }
        Ok(ResolvedValues {
            values: full_values,
            point_count: n_points,
            helicity_indices: selection.helicity_indices.clone(),
            color_indices: selection.color_indices.clone(),
        })
    }

    /// Execute the dominant complete-layout selector shape without resolved
    /// batch tensors: one requested public flow, replayed by one topology
    /// mapping through one fused-helicity materialized-flow lane.
    ///
    /// Multiple mappings or recursive execution plans retain the resolved
    /// fallback because their target cells must survive across evaluations to
    /// preserve exact accumulation order.
    fn try_run_f64_with_single_lc_topology_replay_into_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<bool> {
        let physics = self.physics.clone().ok_or_else(|| {
            RusticolError::invalid_argument(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        let replay_plan = self.cached_lc_resolved_replay_plan(&physics)?;
        let selection = self.cached_lc_resolved_replay_selection(
            &physics,
            replay_plan.as_ref(),
            selected_helicity_ids,
            selected_color_ids,
        )?;
        let [source_group] = selection.source_groups.as_slice() else {
            return Ok(false);
        };
        let [mapping_index] = source_group.mapping_indices.as_slice() else {
            return Ok(false);
        };
        let [replay_entry] = source_group.entries.as_slice() else {
            return Ok(false);
        };
        let materialized_sector_ids =
            self.lc_materialized_sector_ids_for_color_ids(&physics, &source_group.color_ids)?;
        if materialized_sector_ids.len() != 1 {
            return Ok(false);
        }
        let materialized_sector_id = *materialized_sector_ids
            .iter()
            .next()
            .expect("singleton materialized sector checked");
        let Some(selector_runtime) = self.color_selector_runtimes.get(&materialized_sector_id)
        else {
            return Ok(false);
        };
        if selector_runtime.helicity_sum_runtime.is_some()
            || selector_runtime.lc_topology_replay_enabled
            || selector_runtime.has_compiled_helicity_execution_plan()
            || selector_runtime
                .amplitude_stage
                .as_ref()
                .is_none_or(|amplitude| amplitude.color_contraction.is_some())
        {
            return Ok(false);
        }

        let target_component_count = selection
            .helicity_indices
            .len()
            .checked_mul(selection.color_indices.len())
            .ok_or_else(|| RusticolError::invalid_argument("resolved shape overflows usize"))?;
        let mapping = self
            .lc_topology_replay_mappings
            .get(*mapping_index)
            .cloned()
            .ok_or_else(|| {
                RusticolError::integrity(
                    "LC topology replay selection references an unknown mapping",
                )
            })?;
        let expanded_batch = if mapping.is_empty() {
            None
        } else {
            Some(apply_lc_topology_label_permutations_from_view(
                batch,
                self.external_count,
                std::slice::from_ref(&mapping),
            )?)
        };
        let evaluation_view = if let Some(expanded_batch) = expanded_batch.as_deref() {
            F64MomentumBatchView::from_nested(expanded_batch, self.external_count)?
        } else {
            batch
        };
        self.color_selector_runtimes
            .get_mut(&materialized_sector_id)
            .expect("selector lane checked above")
            .run_f64_routed_materialized_into_unprofiled(
                evaluation_view,
                Some(&source_group.helicity_ids),
                Some(&source_group.color_ids),
                replay_entry,
                source_group.source_component_count,
                target_component_count,
                output,
            )?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_f64_routed_materialized_into_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        replay_entry: &LcResolvedReplayEntry,
        source_component_count: usize,
        target_component_count: usize,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        if self.helicity_sum_runtime.is_some()
            || self.lc_topology_replay_enabled
            || self.has_compiled_helicity_execution_plan()
        {
            return Err(RusticolError::integrity(
                "routed direct totals require a materialized leaf execution lane",
            ));
        }
        let physics = self.physics.clone().ok_or_else(|| {
            RusticolError::invalid_argument(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        self.run_f64_materialized_selected_for_resolved_unprofiled(batch, None)?;
        self.amplitude_stage
            .as_mut()
            .expect("materialized leaf amplitude stage checked")
            .reduce_scratch_f64_routed_totals_into(
                batch.point_count(),
                &physics,
                self.normalization_factor,
                selected_helicity_ids,
                selected_color_ids,
                replay_entry,
                source_component_count,
                target_component_count,
                output,
            )
    }

    #[allow(dead_code)] // Allocating compatibility wrapper around the into lane.
    fn run_f64_materialized_selected_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_color_sector_ids: Option<&BTreeSet<i64>>,
    ) -> RusticolResult<Vec<f64>> {
        let mut output = vec![0.0; batch.point_count()];
        self.run_f64_materialized_selected_into_unprofiled(
            batch,
            selected_color_sector_ids,
            &mut output,
        )?;
        Ok(output)
    }

    fn run_f64_materialized_selected_for_resolved_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_color_sector_ids: Option<&BTreeSet<i64>>,
    ) -> RusticolResult<()> {
        self.evaluate_f64_materialized_selected_unprofiled(batch, selected_color_sector_ids)
    }

    fn run_f64_materialized_selected_into_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_color_sector_ids: Option<&BTreeSet<i64>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let n_points = batch.point_count();
        if output.len() != n_points {
            return Err(RusticolError::invalid_argument(format!(
                "evaluation output has length {}, expected {n_points}",
                output.len()
            )));
        }
        self.evaluate_f64_materialized_selected_unprofiled(batch, selected_color_sector_ids)?;
        self.amplitude_stage
            .as_mut()
            .expect("generic amplitude stage checked")
            .reduce_scratch_f64_into_selected_slice(n_points, output, selected_color_sector_ids)?;
        for value in output {
            *value *= self.normalization_factor;
        }
        Ok(())
    }

    fn evaluate_f64_materialized_selected_unprofiled(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_color_sector_ids: Option<&BTreeSet<i64>>,
    ) -> RusticolResult<()> {
        if self.stages.is_none() || self.amplitude_stage.is_none() {
            return Err(self.execution_unavailable_error());
        }
        let n_points = batch.point_count();
        let color_schedule = selected_color_sector_ids
            .map(|sector_ids| self.compiled_color_selector_schedule(sector_ids))
            .transpose()?
            .flatten();
        let state_len = n_points * self.parameter_count;
        self.state_scratch_f64.resize(state_len, c64(0.0, 0.0));
        if self.state_scratch_f64_requires_clear {
            self.state_scratch_f64.fill(c64(0.0, 0.0));
        }
        self.state_scratch_f64_requires_clear = true;
        let state = &mut self.state_scratch_f64;
        let model_parameter_start = self.value_parameter_count + self.momentum_parameter_count;
        for row in 0..n_points {
            let point = batch.point(row);
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_sources_row(
                &self.sources,
                self.external_count,
                &self.particle_masses,
                row_state,
                &point,
            )?;
        }
        for row in 0..n_points {
            let point = batch.point(row);
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_momenta_row(
                &self.momentum_slots,
                self.value_parameter_count,
                self.external_count,
                &self.external_is_initial,
                row_state,
                &point,
            )?;
        }
        for row in 0..n_points {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_model_parameters_row(
                model_parameter_start,
                &self.model_parameter_values_f64,
                row_state,
            )?;
        }
        for (stage_index, stage) in self
            .stages
            .as_mut()
            .expect("generic stages checked")
            .iter_mut()
            .enumerate()
        {
            if let Some(schedule) = color_schedule.as_ref() {
                stage.evaluate_active_chunks_f64_into_state(
                    n_points,
                    self.parameter_count,
                    state.as_mut_slice(),
                    &schedule.active_stage_chunk_indices[stage_index],
                )?;
            } else {
                stage.evaluate_f64_into_state(
                    n_points,
                    self.parameter_count,
                    state.as_mut_slice(),
                )?;
            }
        }
        if let Some(schedule) = color_schedule.as_ref() {
            self.amplitude_stage
                .as_mut()
                .expect("generic amplitude stage checked")
                .evaluate_active_chunks_f64_into_scratch(
                    n_points,
                    state.as_slice(),
                    &schedule.active_amplitude_chunk_indices,
                )?;
        } else {
            self.amplitude_stage
                .as_mut()
                .expect("generic amplitude stage checked")
                .evaluate_f64_into_scratch(n_points, state.as_slice())?;
        }
        self.state_scratch_f64_requires_clear = color_schedule.is_some();
        Ok(())
    }

    pub(super) fn run_f64(
        &mut self,
        batch: &[Vec<[f64; 4]>],
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        if let Some(sum_runtime) = self.helicity_sum_runtime.as_mut() {
            return sum_runtime.run_f64(batch);
        }
        if self.lc_topology_replay_enabled {
            return self.run_f64_with_lc_topology_replay(batch);
        }
        if self.has_compiled_helicity_execution_plan() {
            let (resolved, mut profile) =
                self.run_resolved_f64_with_helicity_recurrence(batch, None, None, None)?;
            let component_count = resolved.helicity_indices.len() * resolved.color_indices.len();
            let materialization_start = Instant::now();
            let values: Vec<f64> = resolved
                .values
                .chunks_exact(component_count)
                .map(|components| components.iter().sum())
                .collect();
            profile.total_materialization_s += materialization_start.elapsed().as_secs_f64();
            profile.total_materialized_value_count += values.len() as u64;
            return Ok((values, profile));
        }
        self.run_f64_selected(batch, None)
    }

    pub(super) fn run_resolved_f64(
        &mut self,
        batch: &[Vec<[f64; 4]>],
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if selected_helicity_ids.is_none()
            && let Some(sum_runtime) = self.helicity_sum_runtime.as_mut()
        {
            return sum_runtime.run_resolved_f64(batch, None, selected_color_ids);
        }
        if self.lc_topology_replay_enabled {
            // A target-flow helicity generally maps to a different source
            // helicity in the materialized topology.  Always compose runtime
            // helicity selection through the replay plan, even when a nested
            // selector lane can evaluate the primary topology directly.
            return self.run_resolved_f64_with_lc_topology_replay(
                batch,
                selected_helicity_ids,
                selected_color_ids,
            );
        }
        if self.has_compiled_helicity_execution_plan() {
            return self.run_resolved_f64_with_helicity_recurrence(
                batch,
                selected_helicity_ids,
                selected_color_ids,
                None,
            );
        }
        let physics = self.physics.clone().ok_or_else(|| {
            RusticolError::invalid_argument(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        let selected_color_lane = if selected_color_ids.is_some_and(|ids| ids.len() == 1) {
            self.lc_materialized_sector_ids_for_color_ids(
                physics.as_ref(),
                selected_color_ids.expect("singleton selection checked"),
            )?
            .into_iter()
            .next()
        } else {
            None
        };
        if let Some(sector_id) = selected_color_lane
            && let Some(selector_runtime) = self.color_selector_runtimes.get_mut(&sector_id)
        {
            return selector_runtime.run_resolved_f64(
                batch,
                selected_helicity_ids,
                selected_color_ids,
            );
        }
        let (_summed, mut profile) = self.run_f64_materialized(batch)?;
        let resolved_reduction_start = Instant::now();
        let resolved = self
            .amplitude_stage
            .as_mut()
            .expect("generic amplitude stage checked")
            .reduce_scratch_f64_resolved(
                batch.len(),
                &physics,
                self.normalization_factor,
                selected_helicity_ids,
                selected_color_ids,
            )?;
        let resolved_reduction_s = resolved_reduction_start.elapsed().as_secs_f64();
        profile.reduction_s += resolved_reduction_s;
        profile.resolved_reduction_materialization_s += resolved_reduction_s;
        profile.resolved_materialized_component_count += resolved.values.len() as u64;
        Ok((resolved, profile))
    }

    fn run_resolved_f64_with_lc_topology_replay(
        &mut self,
        batch: &[Vec<[f64; 4]>],
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        let physics = self.physics.clone().ok_or_else(|| {
            RusticolError::invalid_argument(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        let replay_plan = self.cached_lc_resolved_replay_plan(&physics)?;
        let selection = self.cached_lc_resolved_replay_selection(
            &physics,
            replay_plan.as_ref(),
            selected_helicity_ids,
            selected_color_ids,
        )?;
        let total_start = Instant::now();
        let initial_orchestration_start = Instant::now();
        let n_points = batch.len();
        let component_count = selection.helicity_indices.len() * selection.color_indices.len();
        let mut full_values = vec![0.0; n_points * component_count];
        let mut sector_profiles = Vec::new();
        let mut replay_reduction_elapsed = Duration::ZERO;
        let mappings = self.lc_topology_replay_mappings.clone();
        let mappings_per_chunk = replay_mappings_per_expanded_batch(n_points);
        let mut replay_orchestration_elapsed = initial_orchestration_start.elapsed();
        for source_group in &selection.source_groups {
            let group_orchestration_start = Instant::now();
            let materialized_sector_ids =
                self.lc_materialized_sector_ids_for_color_ids(&physics, &source_group.color_ids)?;
            replay_orchestration_elapsed += group_orchestration_start.elapsed();
            for chunk_start in (0..source_group.mapping_indices.len()).step_by(mappings_per_chunk) {
                let chunk_orchestration_start = Instant::now();
                let chunk_end = usize::min(
                    chunk_start + mappings_per_chunk,
                    source_group.mapping_indices.len(),
                );
                let mapping_chunk = source_group.mapping_indices[chunk_start..chunk_end]
                    .iter()
                    .map(|mapping_index| mappings[*mapping_index].clone())
                    .collect::<Vec<_>>();
                let entry_chunk = &source_group.entries[chunk_start..chunk_end];
                let expanded_batch = if mapping_chunk.len() == 1 && mapping_chunk[0].is_empty() {
                    None
                } else {
                    Some(apply_lc_topology_label_permutations(
                        batch,
                        self.external_count,
                        &mapping_chunk,
                    )?)
                };
                let evaluation_batch = expanded_batch.as_deref().unwrap_or(batch);
                let selected_lane_id = (materialized_sector_ids.len() == 1)
                    .then(|| materialized_sector_ids.iter().next().copied())
                    .flatten();
                replay_orchestration_elapsed += chunk_orchestration_start.elapsed();
                let (materialized, sector_profile) = if self.has_compiled_helicity_execution_plan()
                    && selected_helicity_ids.is_some()
                {
                    self.run_resolved_f64_with_helicity_recurrence(
                        evaluation_batch,
                        Some(&source_group.helicity_ids),
                        Some(&source_group.color_ids),
                        Some(&materialized_sector_ids),
                    )?
                } else if let Some(selector_runtime) = selected_lane_id
                    .and_then(|sector_id| self.color_selector_runtimes.get_mut(&sector_id))
                {
                    selector_runtime.run_resolved_f64(
                        evaluation_batch,
                        Some(&source_group.helicity_ids),
                        Some(&source_group.color_ids),
                    )?
                } else {
                    let (_expanded_totals, mut sector_profile) = self
                        .run_f64_materialized_selected(
                            evaluation_batch,
                            Some(&materialized_sector_ids),
                        )?;
                    let resolved_reduction_start = Instant::now();
                    let materialized = self
                        .amplitude_stage
                        .as_mut()
                        .expect("generic amplitude stage checked")
                        .reduce_scratch_f64_resolved(
                            evaluation_batch.len(),
                            &physics,
                            self.normalization_factor,
                            Some(&source_group.helicity_ids),
                            Some(&source_group.color_ids),
                        )?;
                    let resolved_reduction_s = resolved_reduction_start.elapsed().as_secs_f64();
                    sector_profile.reduction_s += resolved_reduction_s;
                    sector_profile.resolved_reduction_materialization_s += resolved_reduction_s;
                    sector_profile.resolved_materialized_component_count +=
                        materialized.values.len() as u64;
                    (materialized, sector_profile)
                };
                let reduction_start = Instant::now();
                accumulate_selected_lc_replay_resolved_f64(
                    &mut full_values,
                    n_points,
                    &materialized,
                    entry_chunk,
                    source_group.source_component_count,
                    component_count,
                )?;
                replay_reduction_elapsed += reduction_start.elapsed();
                let post_reduction_orchestration_start = Instant::now();
                sector_profiles.push(sector_profile);
                replay_orchestration_elapsed += post_reduction_orchestration_start.elapsed();
            }
        }
        let profile_fold_start = Instant::now();
        let mut profile = RuntimeProfile::default();
        for sector_profile in &sector_profiles {
            profile.add_sector(sector_profile);
        }
        replay_orchestration_elapsed += profile_fold_start.elapsed();
        profile.orchestration_s += profile_duration_seconds(replay_orchestration_elapsed);
        profile.reduction_s += profile_duration_seconds(replay_reduction_elapsed);
        profile.total_s = total_start.elapsed().as_secs_f64();
        Ok((
            ResolvedValues {
                values: full_values,
                point_count: n_points,
                helicity_indices: selection.helicity_indices.clone(),
                color_indices: selection.color_indices.clone(),
            },
            profile,
        ))
    }

    pub(super) fn run_f64_selected(
        &mut self,
        batch: &[Vec<[f64; 4]>],
        selected_color_sector_ids: Option<&BTreeSet<i64>>,
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        if let Some(sum_runtime) = self.helicity_sum_runtime.as_mut() {
            return sum_runtime.run_f64_selected(batch, selected_color_sector_ids);
        }
        if self.lc_topology_replay_enabled {
            if selected_color_sector_ids.is_some() {
                return Err(RusticolError::invalid_argument(
                    "LC color-sector runtime selection is not available for topology-replay artifacts",
                ));
            }
            return self.run_f64_with_lc_topology_replay(batch);
        }
        let Some(selected) = selected_color_sector_ids else {
            let previous = self.set_lc_sector_selector(None);
            let result = self.run_f64_materialized_selected(batch, None);
            self.restore_lc_sector_selector(previous);
            return result;
        };
        if selected.is_empty() {
            return Err(RusticolError::invalid_argument(
                "LC color-sector runtime selection requires at least one sector id",
            ));
        }
        if selected.len() == 1
            && let Some(selector_runtime) = selected
                .iter()
                .next()
                .and_then(|sector_id| self.color_selector_runtimes.get_mut(sector_id))
        {
            return selector_runtime.run_f64_materialized(batch);
        }
        if !self
            .model_parameter_name_to_index
            .contains_key(LC_SECTOR_SELECTOR_PARAMETER)
        {
            return self.run_f64_materialized_selected(batch, Some(selected));
        }
        let total_start = Instant::now();
        let n_points = batch.len();
        let mut values = vec![0.0; n_points];
        let mut sector_profiles = Vec::with_capacity(selected.len());
        let previous = self.set_lc_sector_selector(None);
        let result = (|| {
            for sector_id in selected {
                self.set_lc_sector_selector(Some(*sector_id));
                let mut singleton = BTreeSet::new();
                singleton.insert(*sector_id);
                let (sector_values, sector_profile) =
                    self.run_f64_materialized_selected(batch, Some(&singleton))?;
                for (value, sector_value) in values.iter_mut().zip(sector_values) {
                    *value += sector_value;
                }
                sector_profiles.push(sector_profile);
            }
            let mut profile = RuntimeProfile::default();
            for sector_profile in &sector_profiles {
                profile.add_sector(sector_profile);
            }
            profile.total_s = total_start.elapsed().as_secs_f64();
            Ok((values, profile))
        })();
        self.restore_lc_sector_selector(previous);
        result
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn run_double(
        &mut self,
        batch: &[Vec<[DoubleFloat; 4]>],
    ) -> RusticolResult<(Vec<DoubleFloat>, RuntimeProfile)> {
        if let Some(sum_runtime) = self.helicity_sum_runtime.as_mut() {
            return sum_runtime.run_double(batch);
        }
        if self.lc_topology_replay_enabled {
            return self.run_generic_with_lc_topology_replay(batch, None);
        }
        if self.has_compiled_helicity_execution_plan() {
            return self.run_generic_with_helicity_recurrence(batch, None);
        }
        self.run_generic_materialized(batch, None)
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn run_float(
        &mut self,
        batch: &[Vec<[Float; 4]>],
        binary_precision: u32,
    ) -> RusticolResult<(Vec<Float>, RuntimeProfile)> {
        if let Some(sum_runtime) = self.helicity_sum_runtime.as_mut() {
            return sum_runtime.run_float(batch, binary_precision);
        }
        if self.lc_topology_replay_enabled {
            return self.run_generic_with_lc_topology_replay(batch, Some(binary_precision));
        }
        if self.has_compiled_helicity_execution_plan() {
            return self.run_generic_with_helicity_recurrence(batch, Some(binary_precision));
        }
        self.run_generic_materialized(batch, Some(binary_precision))
    }

    pub(super) fn run_f64_with_lc_topology_replay(
        &mut self,
        batch: &[Vec<[f64; 4]>],
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        let (resolved, profile) =
            self.run_resolved_f64_with_lc_topology_replay(batch, None, None)?;
        let component_count = resolved.helicity_indices.len() * resolved.color_indices.len();
        let mut profile = profile;
        let materialization_start = Instant::now();
        let values: Vec<f64> = resolved
            .values
            .chunks_exact(component_count)
            .map(|components| components.iter().sum())
            .collect();
        profile.total_materialization_s += materialization_start.elapsed().as_secs_f64();
        profile.total_materialized_value_count += values.len() as u64;
        Ok((values, profile))
    }

    pub(super) fn run_f64_materialized(
        &mut self,
        batch: &[Vec<[f64; 4]>],
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        self.run_f64_materialized_selected(batch, None)
    }

    pub(super) fn run_f64_materialized_selected(
        &mut self,
        batch: &[Vec<[f64; 4]>],
        selected_color_sector_ids: Option<&BTreeSet<i64>>,
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        if self.stages.is_none() || self.amplitude_stage.is_none() {
            return Err(self.execution_unavailable_error());
        }
        let total_start = Instant::now();
        let n_points = batch.len();
        let state_prepare_start = Instant::now();
        let color_schedule = selected_color_sector_ids
            .map(|sector_ids| self.compiled_color_selector_schedule(sector_ids))
            .transpose()?
            .flatten();
        let state_len = n_points * self.parameter_count;
        let state_capacity = self.state_scratch_f64.capacity();
        self.state_scratch_f64.resize(state_len, c64(0.0, 0.0));
        let state_prepare_elapsed = state_prepare_start.elapsed();
        let mut scratch_reallocation_count =
            u64::from(self.state_scratch_f64.capacity() != state_capacity);
        let clear_start = Instant::now();
        let mut state_clear_component_count = 0;
        if self.state_scratch_f64_requires_clear {
            self.state_scratch_f64.fill(c64(0.0, 0.0));
            state_clear_component_count = state_len as u64;
        }
        let state_clear_elapsed = clear_start.elapsed();
        // Keep this conservative while execution is in progress: an evaluator
        // error can leave a partially written state. A complete schedule clears
        // the flag below; selector-pruned schedules deliberately leave it set.
        self.state_scratch_f64_requires_clear = true;
        let state = &mut self.state_scratch_f64;
        let sources = &self.sources;
        let momentum_slots = &self.momentum_slots;
        let external_count = self.external_count;
        let external_is_initial = &self.external_is_initial;
        let particle_masses = &self.particle_masses;
        let value_parameter_count = self.value_parameter_count;
        let model_parameter_start = self.value_parameter_count + self.momentum_parameter_count;
        let model_parameter_values = &self.model_parameter_values_f64;
        let source_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_sources_row(sources, external_count, particle_masses, row_state, point)?;
        }
        let source_fill_elapsed = source_start.elapsed();

        let momentum_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_momenta_row(
                momentum_slots,
                value_parameter_count,
                external_count,
                external_is_initial,
                row_state,
                point,
            )?;
        }
        let momentum_setup_elapsed = momentum_start.elapsed();

        let model_parameter_start_time = Instant::now();
        for row in 0..n_points {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_model_parameters_row(
                model_parameter_start,
                model_parameter_values,
                row_state,
            )?;
        }
        let model_parameter_setup_elapsed = model_parameter_start_time.elapsed();
        let mut stage_profiles = Vec::new();
        let mut stage_input_pack_by_stage_s = Vec::new();
        let mut stage_leaf_input_pack_by_stage_s = Vec::new();
        let mut stage_evaluator_call_by_stage_s = Vec::new();
        let mut stage_backend_call_by_stage_s = Vec::new();
        let mut stage_evaluator_output_gather_by_stage_s = Vec::new();
        let mut stage_output_assign_by_stage_s = Vec::new();
        for (stage_index, stage) in self
            .stages
            .as_mut()
            .expect("generic stages checked")
            .iter_mut()
            .enumerate()
        {
            let stage_profile = if let Some(schedule) = color_schedule.as_ref() {
                stage.evaluate_active_chunks_f64_into_state_profile(
                    n_points,
                    self.parameter_count,
                    state.as_mut_slice(),
                    &schedule.active_stage_chunk_indices[stage_index],
                )?
            } else {
                stage.evaluate_f64_into_state_profile(
                    n_points,
                    self.parameter_count,
                    state.as_mut_slice(),
                )?
            };
            stage_input_pack_by_stage_s.push(stage_profile.input_pack_s);
            stage_leaf_input_pack_by_stage_s.push(stage_profile.evaluator.leaf_input_pack_s);
            stage_evaluator_call_by_stage_s.push(stage_profile.evaluator.legacy_evaluator_call_s);
            stage_backend_call_by_stage_s.push(stage_profile.evaluator.evaluator_call_s);
            stage_evaluator_output_gather_by_stage_s.push(stage_profile.evaluator.output_gather_s);
            stage_output_assign_by_stage_s.push(stage_profile.output_assign_s);
            stage_profiles.push(stage_profile);
        }

        let amplitude_profile = if let Some(schedule) = color_schedule.as_ref() {
            self.amplitude_stage
                .as_mut()
                .expect("generic amplitude stage checked")
                .evaluate_active_chunks_f64_into_scratch_profile(
                    n_points,
                    state.as_slice(),
                    &schedule.active_amplitude_chunk_indices,
                )?
        } else {
            self.amplitude_stage
                .as_mut()
                .expect("generic amplitude stage checked")
                .evaluate_f64_into_scratch_profile(n_points, state.as_slice())?
        };
        let amplitude_input_pack_s = amplitude_profile.input_pack_s;
        let amplitude_evaluator_call_s =
            amplitude_profile.evaluator.legacy_evaluator_call_s + amplitude_profile.output_remap_s;
        let amplitude_evaluator_s = amplitude_input_pack_s + amplitude_evaluator_call_s;

        let reduction_start = Instant::now();
        let values_capacity = self.values_scratch_f64.capacity();
        self.amplitude_stage
            .as_mut()
            .expect("generic amplitude stage checked")
            .reduce_scratch_f64_into_selected(
                n_points,
                &mut self.values_scratch_f64,
                selected_color_sector_ids,
            )?;
        for value in &mut self.values_scratch_f64 {
            *value *= self.normalization_factor;
        }
        let reduction_s = reduction_start.elapsed().as_secs_f64();
        scratch_reallocation_count +=
            u64::from(self.values_scratch_f64.capacity() != values_capacity);
        self.state_scratch_f64_requires_clear = color_schedule.is_some();
        let stage_input_pack_s = stage_input_pack_by_stage_s.iter().sum::<f64>();
        let stage_evaluator_call_s = stage_evaluator_call_by_stage_s.iter().sum::<f64>();
        let stage_evaluator_s = stage_input_pack_s + stage_evaluator_call_s;
        let output_assign_s = stage_output_assign_by_stage_s.iter().sum::<f64>();
        let final_output_copy_start = Instant::now();
        let values = self.values_scratch_f64.clone();
        let final_output_copy_s = final_output_copy_start.elapsed().as_secs_f64();
        Ok((
            values,
            RuntimeProfile {
                source_fill_s: profile_duration_seconds(source_fill_elapsed),
                state_prepare_s: profile_duration_seconds(state_prepare_elapsed),
                state_clear_s: profile_duration_seconds(state_clear_elapsed),
                momentum_input_setup_s: profile_duration_seconds(momentum_setup_elapsed),
                momentum_setup_s: profile_duration_seconds(
                    momentum_setup_elapsed + model_parameter_setup_elapsed,
                ),
                model_parameter_setup_s: profile_duration_seconds(model_parameter_setup_elapsed),
                stage_input_pack_s,
                stage_leaf_input_pack_s: stage_profiles
                    .iter()
                    .map(|profile| profile.evaluator.leaf_input_pack_s)
                    .sum(),
                stage_evaluator_call_s,
                stage_backend_call_s: stage_profiles
                    .iter()
                    .map(|profile| profile.evaluator.evaluator_call_s)
                    .sum(),
                stage_evaluator_output_gather_s: stage_profiles
                    .iter()
                    .map(|profile| profile.evaluator.output_gather_s)
                    .sum(),
                stage_evaluator_s,
                output_assign_s,
                amplitude_input_pack_s,
                amplitude_leaf_input_pack_s: amplitude_profile.evaluator.leaf_input_pack_s,
                amplitude_evaluator_call_s,
                amplitude_backend_call_s: amplitude_profile.evaluator.evaluator_call_s,
                amplitude_evaluator_output_gather_s: amplitude_profile.evaluator.output_gather_s,
                amplitude_output_remap_s: amplitude_profile.output_remap_s,
                amplitude_evaluator_s,
                reduction_s,
                total_s: total_start.elapsed().as_secs_f64(),
                stage_input_pack_by_stage_s,
                stage_leaf_input_pack_by_stage_s,
                stage_evaluator_call_by_stage_s,
                stage_backend_call_by_stage_s,
                stage_evaluator_output_gather_by_stage_s,
                stage_output_assign_by_stage_s,
                state_component_count: state_len as u64,
                state_clear_component_count,
                source_component_count: (n_points * self.source_count) as u64,
                momentum_component_count: (n_points * self.momentum_parameter_count) as u64,
                model_parameter_component_count: (n_points * model_parameter_values.len()) as u64,
                stage_input_copy_component_count: stage_profiles
                    .iter()
                    .map(|profile| profile.input_copy_component_count)
                    .sum(),
                stage_leaf_input_copy_component_count: stage_profiles
                    .iter()
                    .map(|profile| profile.evaluator.leaf_input_copy_component_count)
                    .sum(),
                stage_evaluator_output_gather_component_count: stage_profiles
                    .iter()
                    .map(|profile| profile.evaluator.output_gather_component_count)
                    .sum(),
                stage_output_assign_component_count: stage_profiles
                    .iter()
                    .map(|profile| profile.output_assign_component_count)
                    .sum(),
                amplitude_input_copy_component_count: amplitude_profile.input_copy_component_count,
                amplitude_leaf_input_copy_component_count: amplitude_profile
                    .evaluator
                    .leaf_input_copy_component_count,
                amplitude_evaluator_output_gather_component_count: amplitude_profile
                    .evaluator
                    .output_gather_component_count,
                amplitude_output_remap_component_count: amplitude_profile
                    .output_remap_component_count,
                evaluator_backend_call_count: stage_profiles
                    .iter()
                    .map(|profile| profile.evaluator.backend_call_count)
                    .sum::<u64>()
                    + amplitude_profile.evaluator.backend_call_count,
                reduction_input_component_count: (n_points
                    * self
                        .amplitude_stage
                        .as_ref()
                        .expect("generic amplitude stage checked")
                        .output_length) as u64,
                total_materialized_value_count: n_points as u64,
                final_output_copy_s,
                final_output_copy_value_count: self.values_scratch_f64.len() as u64,
                scratch_reallocation_count: scratch_reallocation_count
                    + stage_profiles
                        .iter()
                        .map(|profile| {
                            profile.scratch_reallocation_count
                                + profile.evaluator.scratch_reallocation_count
                        })
                        .sum::<u64>()
                    + amplitude_profile.scratch_reallocation_count
                    + amplitude_profile.evaluator.scratch_reallocation_count,
                ..RuntimeProfile::default()
            },
        ))
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn run_generic_with_lc_topology_replay<T>(
        &mut self,
        batch: &[Vec<[T; 4]>],
        binary_precision: Option<u32>,
    ) -> RusticolResult<(Vec<T>, RuntimeProfile)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let (resolved, profile) =
            self.run_resolved_generic_with_lc_topology_replay(batch, binary_precision, None, None)?;
        let component_count = resolved.helicity_indices.len() * resolved.color_indices.len();
        let values = resolved
            .values
            .chunks_exact(component_count)
            .map(|components| {
                components
                    .iter()
                    .cloned()
                    .fold(T::new_zero(), |sum, value| sum + value)
            })
            .collect();
        Ok((values, profile))
    }

    #[cfg(feature = "symbolica-runtime")]
    fn run_generic_with_helicity_recurrence<T>(
        &mut self,
        batch: &[Vec<[T; 4]>],
        binary_precision: Option<u32>,
    ) -> RusticolResult<(Vec<T>, RuntimeProfile)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let (resolved, profile) = self.run_resolved_generic_with_helicity_recurrence(
            batch,
            binary_precision,
            None,
            None,
            true,
        )?;
        let component_count = resolved.helicity_indices.len() * resolved.color_indices.len();
        let values = resolved
            .values
            .chunks_exact(component_count)
            .map(|components| {
                components
                    .iter()
                    .cloned()
                    .fold(T::new_zero(), |sum, value| sum + value)
            })
            .collect();
        Ok((values, profile))
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn run_generic_materialized<T>(
        &mut self,
        batch: &[Vec<[T; 4]>],
        binary_precision: Option<u32>,
    ) -> RusticolResult<(Vec<T>, RuntimeProfile)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        if self.stages.is_none() || self.amplitude_stage.is_none() {
            return Err(self.execution_unavailable_error());
        }
        let total_start = Instant::now();
        let n_points = batch.len();
        let mut state = vec![complex_zero::<T>(); n_points * self.parameter_count];
        let sources = &self.sources;
        let momentum_slots = &self.momentum_slots;
        let external_count = self.external_count;
        let external_is_initial = &self.external_is_initial;
        let particle_masses = &self.particle_masses;
        let value_parameter_count = self.value_parameter_count;
        let model_parameter_start = self.value_parameter_count + self.momentum_parameter_count;
        let model_parameter_values = &self.model_parameter_values_f64;

        let source_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_sources_row_generic(
                sources,
                external_count,
                particle_masses,
                row_state,
                point,
            )?;
        }
        let source_fill_elapsed = source_start.elapsed();

        let momentum_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_momenta_row_generic(
                momentum_slots,
                value_parameter_count,
                external_count,
                external_is_initial,
                row_state,
                point,
            )?;
        }
        let momentum_setup_elapsed = momentum_start.elapsed();

        let model_parameter_start_time = Instant::now();
        for row in 0..n_points {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_model_parameters_row_generic(
                model_parameter_start,
                model_parameter_values,
                row_state,
            )?;
        }
        let model_parameter_setup_elapsed = model_parameter_start_time.elapsed();

        let mut stage_input_pack_s = 0.0;
        let mut stage_evaluator_call_s = 0.0;
        let mut stage_evaluator_s = 0.0;
        let mut output_assign_s = 0.0;
        let mut stage_input_pack_by_stage_s = Vec::new();
        let mut stage_evaluator_call_by_stage_s = Vec::new();
        let mut stage_output_assign_by_stage_s = Vec::new();
        for stage in self.stages.as_mut().expect("generic stages checked") {
            let (pack_s, eval_s, assign_s) = stage.evaluate_generic_into_state(
                n_points,
                self.parameter_count,
                state.as_mut_slice(),
                binary_precision,
            )?;
            stage_input_pack_s += pack_s;
            stage_evaluator_call_s += eval_s;
            stage_evaluator_s += pack_s + eval_s;
            output_assign_s += assign_s;
            stage_input_pack_by_stage_s.push(pack_s);
            stage_evaluator_call_by_stage_s.push(eval_s);
            stage_output_assign_by_stage_s.push(assign_s);
        }

        let (raw_sums, amplitude_input_pack_s, amplitude_evaluator_call_s) = self
            .amplitude_stage
            .as_mut()
            .expect("generic amplitude stage checked")
            .evaluate_raw_sums_generic(n_points, state.as_slice(), binary_precision)?;
        let amplitude_evaluator_s = amplitude_input_pack_s + amplitude_evaluator_call_s;

        let reduction_start = Instant::now();
        let factor = T::from(self.normalization_factor);
        let values = raw_sums
            .into_iter()
            .map(|value| value * factor.clone())
            .collect::<Vec<_>>();
        let reduction_s = reduction_start.elapsed().as_secs_f64();
        Ok((
            values,
            RuntimeProfile {
                source_fill_s: profile_duration_seconds(source_fill_elapsed),
                momentum_input_setup_s: profile_duration_seconds(momentum_setup_elapsed),
                momentum_setup_s: profile_duration_seconds(
                    momentum_setup_elapsed + model_parameter_setup_elapsed,
                ),
                model_parameter_setup_s: profile_duration_seconds(model_parameter_setup_elapsed),
                stage_input_pack_s,
                stage_evaluator_call_s,
                stage_evaluator_s,
                output_assign_s,
                amplitude_input_pack_s,
                amplitude_evaluator_call_s,
                amplitude_evaluator_s,
                reduction_s,
                total_s: total_start.elapsed().as_secs_f64(),
                stage_input_pack_by_stage_s,
                stage_evaluator_call_by_stage_s,
                stage_output_assign_by_stage_s,
                ..RuntimeProfile::default()
            },
        ))
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn run_resolved_generic<T>(
        &mut self,
        batch: &[Vec<[T; 4]>],
        binary_precision: Option<u32>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<T>, RuntimeProfile)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        if selected_helicity_ids.is_none()
            && let Some(sum_runtime) = self.helicity_sum_runtime.as_mut()
        {
            return sum_runtime.run_resolved_generic(
                batch,
                binary_precision,
                None,
                selected_color_ids,
            );
        }
        if self.lc_topology_replay_enabled {
            return self.run_resolved_generic_with_lc_topology_replay(
                batch,
                binary_precision,
                selected_helicity_ids,
                selected_color_ids,
            );
        }
        if self.has_compiled_helicity_execution_plan() {
            return self.run_resolved_generic_with_helicity_recurrence(
                batch,
                binary_precision,
                selected_helicity_ids,
                selected_color_ids,
                true,
            );
        }
        self.run_resolved_generic_materialized(
            batch,
            binary_precision,
            selected_helicity_ids,
            selected_color_ids,
        )
    }

    #[cfg(feature = "symbolica-runtime")]
    fn run_resolved_generic_with_lc_topology_replay<T>(
        &mut self,
        batch: &[Vec<[T; 4]>],
        binary_precision: Option<u32>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<T>, RuntimeProfile)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let physics = self.physics.clone().ok_or_else(|| {
            RusticolError::invalid_argument(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        let replay_plan = self.cached_lc_resolved_replay_plan(&physics)?;
        let selection = self.cached_lc_resolved_replay_selection(
            &physics,
            replay_plan.as_ref(),
            selected_helicity_ids,
            selected_color_ids,
        )?;
        let total_start = Instant::now();
        let initial_orchestration_start = Instant::now();
        let n_points = batch.len();
        let component_count = selection.helicity_indices.len() * selection.color_indices.len();
        let mut full_values = vec![T::new_zero(); n_points * component_count];
        let mut sector_profiles = Vec::new();
        let mut replay_reduction_elapsed = Duration::ZERO;
        let mappings = self.lc_topology_replay_mappings.clone();
        let mappings_per_chunk = replay_mappings_per_expanded_batch(n_points);
        let mut replay_orchestration_elapsed = initial_orchestration_start.elapsed();
        for source_group in &selection.source_groups {
            for chunk_start in (0..source_group.mapping_indices.len()).step_by(mappings_per_chunk) {
                let chunk_orchestration_start = Instant::now();
                let chunk_end = usize::min(
                    chunk_start + mappings_per_chunk,
                    source_group.mapping_indices.len(),
                );
                let mapping_chunk = source_group.mapping_indices[chunk_start..chunk_end]
                    .iter()
                    .map(|mapping_index| mappings[*mapping_index].clone())
                    .collect::<Vec<_>>();
                let entry_chunk = &source_group.entries[chunk_start..chunk_end];
                let expanded_batch = if mapping_chunk.len() == 1 && mapping_chunk[0].is_empty() {
                    None
                } else {
                    Some(apply_lc_topology_label_permutations_generic(
                        batch,
                        self.external_count,
                        &mapping_chunk,
                    )?)
                };
                let evaluation_batch = expanded_batch.as_deref().unwrap_or(batch);
                replay_orchestration_elapsed += chunk_orchestration_start.elapsed();
                let (materialized, sector_profile) = if self.has_compiled_helicity_execution_plan()
                    && selected_helicity_ids.is_some()
                {
                    self.run_resolved_generic_with_helicity_recurrence(
                        evaluation_batch,
                        binary_precision,
                        Some(&source_group.helicity_ids),
                        Some(&source_group.color_ids),
                        false,
                    )?
                } else {
                    self.run_resolved_generic_materialized(
                        evaluation_batch,
                        binary_precision,
                        Some(&source_group.helicity_ids),
                        Some(&source_group.color_ids),
                    )?
                };
                let reduction_start = Instant::now();
                accumulate_selected_lc_replay_resolved_generic(
                    &mut full_values,
                    n_points,
                    &materialized,
                    entry_chunk,
                    source_group.source_component_count,
                    component_count,
                )?;
                replay_reduction_elapsed += reduction_start.elapsed();
                let post_reduction_orchestration_start = Instant::now();
                sector_profiles.push(sector_profile);
                replay_orchestration_elapsed += post_reduction_orchestration_start.elapsed();
            }
        }
        let profile_fold_start = Instant::now();
        let mut profile = RuntimeProfile::default();
        for sector_profile in &sector_profiles {
            profile.add_sector(sector_profile);
        }
        replay_orchestration_elapsed += profile_fold_start.elapsed();
        profile.orchestration_s += profile_duration_seconds(replay_orchestration_elapsed);
        profile.reduction_s += profile_duration_seconds(replay_reduction_elapsed);
        profile.total_s = total_start.elapsed().as_secs_f64();
        Ok((
            ResolvedValues {
                values: full_values,
                point_count: n_points,
                helicity_indices: selection.helicity_indices.clone(),
                color_indices: selection.color_indices.clone(),
            },
            profile,
        ))
    }

    #[cfg(feature = "symbolica-runtime")]
    fn run_resolved_generic_materialized<T>(
        &mut self,
        batch: &[Vec<[T; 4]>],
        binary_precision: Option<u32>,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<T>, RuntimeProfile)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        if self.stages.is_none() || self.amplitude_stage.is_none() {
            return Err(self.execution_unavailable_error());
        }
        let physics = self.physics.clone().ok_or_else(|| {
            RusticolError::invalid_argument(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        let total_start = Instant::now();
        let n_points = batch.len();
        let mut state = vec![complex_zero::<T>(); n_points * self.parameter_count];
        let sources = &self.sources;
        let momentum_slots = &self.momentum_slots;
        let external_count = self.external_count;
        let external_is_initial = &self.external_is_initial;
        let particle_masses = &self.particle_masses;
        let value_parameter_count = self.value_parameter_count;
        let model_parameter_start = self.value_parameter_count + self.momentum_parameter_count;
        let model_parameter_values = &self.model_parameter_values_f64;

        let source_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_sources_row_generic(
                sources,
                external_count,
                particle_masses,
                row_state,
                point,
            )?;
        }
        let source_fill_elapsed = source_start.elapsed();

        let momentum_start = Instant::now();
        for (row, point) in batch.iter().enumerate() {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_momenta_row_generic(
                momentum_slots,
                value_parameter_count,
                external_count,
                external_is_initial,
                row_state,
                point,
            )?;
        }
        let momentum_setup_elapsed = momentum_start.elapsed();

        let model_parameter_start_time = Instant::now();
        for row in 0..n_points {
            let row_state =
                &mut state[row * self.parameter_count..(row + 1) * self.parameter_count];
            Self::fill_model_parameters_row_generic(
                model_parameter_start,
                model_parameter_values,
                row_state,
            )?;
        }
        let model_parameter_setup_elapsed = model_parameter_start_time.elapsed();

        let mut stage_input_pack_s = 0.0;
        let mut stage_evaluator_call_s = 0.0;
        let mut stage_evaluator_s = 0.0;
        let mut output_assign_s = 0.0;
        let mut stage_input_pack_by_stage_s = Vec::new();
        let mut stage_evaluator_call_by_stage_s = Vec::new();
        let mut stage_output_assign_by_stage_s = Vec::new();
        for stage in self.stages.as_mut().expect("generic stages checked") {
            let (pack_s, eval_s, assign_s) = stage.evaluate_generic_into_state(
                n_points,
                self.parameter_count,
                state.as_mut_slice(),
                binary_precision,
            )?;
            stage_input_pack_s += pack_s;
            stage_evaluator_call_s += eval_s;
            stage_evaluator_s += pack_s + eval_s;
            output_assign_s += assign_s;
            stage_input_pack_by_stage_s.push(pack_s);
            stage_evaluator_call_by_stage_s.push(eval_s);
            stage_output_assign_by_stage_s.push(assign_s);
        }

        let reduction_start = Instant::now();
        let (resolved, amplitude_input_pack_s, amplitude_evaluator_call_s) = self
            .amplitude_stage
            .as_mut()
            .expect("generic amplitude stage checked")
            .evaluate_resolved_generic(
                n_points,
                state.as_slice(),
                binary_precision,
                &physics,
                self.normalization_factor,
                selected_helicity_ids,
                selected_color_ids,
            )?;
        let reduction_s = reduction_start.elapsed().as_secs_f64();
        let amplitude_evaluator_s = amplitude_input_pack_s + amplitude_evaluator_call_s;
        Ok((
            resolved,
            RuntimeProfile {
                source_fill_s: profile_duration_seconds(source_fill_elapsed),
                momentum_input_setup_s: profile_duration_seconds(momentum_setup_elapsed),
                momentum_setup_s: profile_duration_seconds(
                    momentum_setup_elapsed + model_parameter_setup_elapsed,
                ),
                model_parameter_setup_s: profile_duration_seconds(model_parameter_setup_elapsed),
                stage_input_pack_s,
                stage_evaluator_call_s,
                stage_evaluator_s,
                output_assign_s,
                amplitude_input_pack_s,
                amplitude_evaluator_call_s,
                amplitude_evaluator_s,
                reduction_s,
                total_s: total_start.elapsed().as_secs_f64(),
                stage_input_pack_by_stage_s,
                stage_evaluator_call_by_stage_s,
                stage_output_assign_by_stage_s,
                ..RuntimeProfile::default()
            },
        ))
    }
}

pub(super) fn write_resolved_f64_totals(
    resolved: &ResolvedValues<f64>,
    output: &mut [f64],
) -> RusticolResult<()> {
    if output.len() != resolved.point_count {
        return Err(RusticolError::invalid_argument(format!(
            "evaluation output has length {}, expected {}",
            output.len(),
            resolved.point_count
        )));
    }
    let component_count = resolved.helicity_indices.len() * resolved.color_indices.len();
    if component_count == 0 {
        output.fill(0.0);
        return Ok(());
    }
    if resolved.values.len() != resolved.point_count * component_count {
        return Err(RusticolError::integrity(
            "resolved evaluation has an inconsistent component buffer length",
        ));
    }
    for (target, components) in output
        .iter_mut()
        .zip(resolved.values.chunks_exact(component_count))
    {
        *target = components.iter().sum();
    }
    Ok(())
}

pub(super) fn accumulate_selected_lc_replay_resolved_f64(
    target: &mut [f64],
    point_count: usize,
    materialized: &ResolvedValues<f64>,
    replay_entries: &[LcResolvedReplayEntry],
    source_component_count: usize,
    target_component_count: usize,
) -> RusticolResult<()> {
    validate_materialized_replay_shape(
        materialized,
        point_count,
        replay_entries.len(),
        source_component_count,
    )?;
    if target.len() != point_count * target_component_count {
        return Err(RusticolError::invalid_argument(
            "LC topology replay target has an inconsistent resolved shape",
        ));
    }
    for (entry_index, entry) in replay_entries.iter().enumerate() {
        for point_index in 0..point_count {
            let source_row = (entry_index * point_count + point_index) * source_component_count;
            let target_row = point_index * target_component_count;
            for route in &entry.routes {
                if route.source_index >= source_component_count
                    || route.target_index >= target_component_count
                {
                    return Err(RusticolError::integrity(
                        "LC topology replay selected route is out of bounds",
                    ));
                }
                target[target_row + route.target_index] +=
                    route.weight * materialized.values[source_row + route.source_index];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn accumulate_lc_replay_resolved_f64(
    target: &mut [f64],
    point_count: usize,
    materialized: &ResolvedValues<f64>,
    replay_entries: &[LcResolvedReplayEntry],
    component_count: usize,
) -> RusticolResult<()> {
    accumulate_selected_lc_replay_resolved_f64(
        target,
        point_count,
        materialized,
        replay_entries,
        component_count,
        component_count,
    )
}

#[cfg(feature = "symbolica-runtime")]
fn accumulate_selected_lc_replay_resolved_generic<T>(
    target: &mut [T],
    point_count: usize,
    materialized: &ResolvedValues<T>,
    replay_entries: &[LcResolvedReplayEntry],
    source_component_count: usize,
    target_component_count: usize,
) -> RusticolResult<()>
where
    T: RusticolHighPrecisionNumber,
    Complex<T>: Real + EvaluationDomain,
{
    validate_materialized_replay_shape(
        materialized,
        point_count,
        replay_entries.len(),
        source_component_count,
    )?;
    if target.len() != point_count * target_component_count {
        return Err(RusticolError::invalid_argument(
            "LC topology replay target has an inconsistent resolved shape",
        ));
    }
    for (entry_index, entry) in replay_entries.iter().enumerate() {
        for point_index in 0..point_count {
            let source_row = (entry_index * point_count + point_index) * source_component_count;
            let target_row = point_index * target_component_count;
            for route in &entry.routes {
                if route.source_index >= source_component_count
                    || route.target_index >= target_component_count
                {
                    return Err(RusticolError::integrity(
                        "LC topology replay selected route is out of bounds",
                    ));
                }
                target[target_row + route.target_index] +=
                    materialized.values[source_row + route.source_index].clone()
                        * T::from(route.weight);
            }
        }
    }
    Ok(())
}

fn validate_materialized_replay_shape<T>(
    materialized: &ResolvedValues<T>,
    point_count: usize,
    replay_count: usize,
    component_count: usize,
) -> RusticolResult<()> {
    let expected_points = point_count.checked_mul(replay_count).ok_or_else(|| {
        RusticolError::invalid_argument("LC topology replay point count overflowed")
    })?;
    let expected_values = expected_points
        .checked_mul(component_count)
        .ok_or_else(|| {
            RusticolError::invalid_argument("LC topology replay resolved shape overflowed")
        })?;
    if materialized.point_count != expected_points
        || materialized.values.len() != expected_values
        || materialized.helicity_indices.len() * materialized.color_indices.len() != component_count
    {
        return Err(RusticolError::invalid_argument(
            "materialized LC topology replay result has an inconsistent resolved shape",
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn select_resolved_values<T: Clone>(
    full_values: Vec<T>,
    point_count: usize,
    physics: &PhysicsRuntime,
    selected_helicity_ids: Option<&BTreeSet<String>>,
    selected_color_ids: Option<&BTreeSet<String>>,
) -> RusticolResult<ResolvedValues<T>> {
    let helicity_count = physics.manifest.helicities.len();
    let color_count = physics.manifest.color_components.len();
    if full_values.len() != point_count * helicity_count * color_count {
        return Err(RusticolError::invalid_argument(
            "resolved LC topology replay result has an inconsistent public shape",
        ));
    }
    let helicity_indices = physics.selected_helicity_indices(selected_helicity_ids)?;
    let color_indices = physics.selected_color_indices(selected_color_ids)?;
    let mut values = Vec::with_capacity(point_count * helicity_indices.len() * color_indices.len());
    for point_index in 0..point_count {
        for helicity_index in &helicity_indices {
            for color_index in &color_indices {
                values.push(
                    full_values[(point_index * helicity_count + *helicity_index) * color_count
                        + *color_index]
                        .clone(),
                );
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

#[cfg(test)]
mod selected_replay_tests {
    use super::*;

    #[test]
    fn accumulates_selected_routes_without_materializing_public_axis() {
        let materialized = ResolvedValues {
            values: vec![
                0.0, 2.0, 0.0, 0.0, // mapping 0, point 0
                0.0, 3.0, 0.0, 0.0, // mapping 0, point 1
                0.0, 0.0, 0.0, 5.0, // mapping 1, point 0
                0.0, 0.0, 0.0, 7.0, // mapping 1, point 1
            ],
            point_count: 4,
            helicity_indices: vec![0, 1],
            color_indices: vec![0, 1],
        };
        let entries = vec![
            LcResolvedReplayEntry {
                routes: vec![LcResolvedReplayRoute {
                    source_index: 1,
                    target_index: 0,
                    weight: 1.0,
                }],
            },
            LcResolvedReplayEntry {
                routes: vec![LcResolvedReplayRoute {
                    source_index: 3,
                    target_index: 0,
                    weight: 2.0,
                }],
            },
        ];
        let mut selected = vec![0.0; 2];
        accumulate_selected_lc_replay_resolved_f64(&mut selected, 2, &materialized, &entries, 4, 1)
            .expect("selected replay accumulation");
        assert_eq!(selected, vec![12.0, 17.0]);
    }

    #[test]
    fn accumulates_two_replay_groups_and_one_residual_exactly_once() {
        let materialized = ResolvedValues {
            values: vec![
                10.0, 0.0, 20.0, 0.0, 30.0, // identity mapping
                11.0, 0.0, 21.0, 0.0, 31.0, // shared swap mapping
            ],
            point_count: 2,
            helicity_indices: vec![0],
            color_indices: vec![0, 1, 2, 3, 4],
        };
        let entries = vec![
            LcResolvedReplayEntry {
                routes: vec![
                    LcResolvedReplayRoute {
                        source_index: 0,
                        target_index: 0,
                        weight: 1.0,
                    },
                    LcResolvedReplayRoute {
                        source_index: 2,
                        target_index: 2,
                        weight: 1.0,
                    },
                    LcResolvedReplayRoute {
                        source_index: 4,
                        target_index: 4,
                        weight: 1.0,
                    },
                ],
            },
            LcResolvedReplayEntry {
                routes: vec![
                    LcResolvedReplayRoute {
                        source_index: 0,
                        target_index: 1,
                        weight: 1.0,
                    },
                    LcResolvedReplayRoute {
                        source_index: 2,
                        target_index: 3,
                        weight: 1.0,
                    },
                ],
            },
        ];
        let mut target = vec![0.0; 5];

        accumulate_selected_lc_replay_resolved_f64(&mut target, 1, &materialized, &entries, 5, 5)
            .unwrap();

        assert_eq!(target, vec![10.0, 11.0, 20.0, 21.0, 30.0]);
    }
}
