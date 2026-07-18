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

    pub(super) fn run_f64(
        &mut self,
        batch: &[Vec<[f64; 4]>],
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        self.run_f64_selected(batch, None)
    }

    pub(super) fn run_resolved_f64(
        &mut self,
        batch: &[Vec<[f64; 4]>],
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if self.lc_topology_replay_enabled {
            return self.run_resolved_f64_with_lc_topology_replay(
                batch,
                selected_helicity_ids,
                selected_color_ids,
            );
        }
        let physics = self.physics.clone().ok_or_else(|| {
            RusticolError::invalid_argument(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        let (_summed, profile) = self.run_f64_materialized(batch)?;
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
        let replay_plan = physics.lc_resolved_replay_plan(
            &self.lc_topology_replay_public_mappings,
            &self.lc_topology_replay_weights,
        )?;
        let total_start = Instant::now();
        let n_points = batch.len();
        let component_count = replay_plan.helicity_count * replay_plan.color_count;
        let mut full_values = vec![0.0; n_points * component_count];
        let mut profile = RuntimeProfile::default();
        let mappings = self.lc_topology_replay_mappings.clone();
        let mappings_per_chunk = replay_mappings_per_expanded_batch(n_points);
        for chunk_start in (0..mappings.len()).step_by(mappings_per_chunk) {
            let chunk_end = usize::min(chunk_start + mappings_per_chunk, mappings.len());
            let mapping_chunk = &mappings[chunk_start..chunk_end];
            let expanded_batch =
                apply_lc_topology_label_permutations(batch, self.external_count, mapping_chunk)?;
            let (_expanded_totals, sector_profile) = self.run_f64_materialized(&expanded_batch)?;
            let reduction_start = Instant::now();
            let materialized = self
                .amplitude_stage
                .as_mut()
                .expect("generic amplitude stage checked")
                .reduce_scratch_f64_resolved(
                    expanded_batch.len(),
                    &physics,
                    self.normalization_factor,
                    None,
                    None,
                )?;
            profile.add_sector(&sector_profile);
            accumulate_lc_replay_resolved_f64(
                &mut full_values,
                n_points,
                &materialized,
                &replay_plan.entries[chunk_start..chunk_end],
                component_count,
            )?;
            profile.reduction_s += reduction_start.elapsed().as_secs_f64();
        }
        profile.total_s = total_start.elapsed().as_secs_f64();
        let resolved = select_resolved_values(
            full_values,
            n_points,
            &physics,
            selected_helicity_ids,
            selected_color_ids,
        )?;
        Ok((resolved, profile))
    }

    pub(super) fn run_f64_selected(
        &mut self,
        batch: &[Vec<[f64; 4]>],
        selected_color_sector_ids: Option<&BTreeSet<i64>>,
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
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
        if !self
            .model_parameter_name_to_index
            .contains_key(LC_SECTOR_SELECTOR_PARAMETER)
        {
            return self.run_f64_materialized_selected(batch, Some(selected));
        }
        let total_start = Instant::now();
        let n_points = batch.len();
        let mut values = vec![0.0; n_points];
        let mut profile = RuntimeProfile::default();
        let previous = self.set_lc_sector_selector(None);
        for sector_id in selected {
            self.set_lc_sector_selector(Some(*sector_id));
            let mut singleton = BTreeSet::new();
            singleton.insert(*sector_id);
            let (sector_values, sector_profile) =
                self.run_f64_materialized_selected(batch, Some(&singleton))?;
            for (value, sector_value) in values.iter_mut().zip(sector_values) {
                *value += sector_value;
            }
            profile.add_sector(&sector_profile);
        }
        self.restore_lc_sector_selector(previous);
        profile.total_s = total_start.elapsed().as_secs_f64();
        Ok((values, profile))
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn run_double(
        &mut self,
        batch: &[Vec<[DoubleFloat; 4]>],
    ) -> RusticolResult<(Vec<DoubleFloat>, RuntimeProfile)> {
        if self.lc_topology_replay_enabled {
            return self.run_generic_with_lc_topology_replay(batch, None);
        }
        self.run_generic_materialized(batch, None)
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(super) fn run_float(
        &mut self,
        batch: &[Vec<[Float; 4]>],
        binary_precision: u32,
    ) -> RusticolResult<(Vec<Float>, RuntimeProfile)> {
        if self.lc_topology_replay_enabled {
            return self.run_generic_with_lc_topology_replay(batch, Some(binary_precision));
        }
        self.run_generic_materialized(batch, Some(binary_precision))
    }

    pub(super) fn run_f64_with_lc_topology_replay(
        &mut self,
        batch: &[Vec<[f64; 4]>],
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        let total_start = Instant::now();
        let n_points = batch.len();
        let mut values = vec![0.0; n_points];
        let mut profile = RuntimeProfile::default();
        let mappings = self.lc_topology_replay_mappings.clone();
        let weights = self.lc_topology_replay_weights.clone();
        let mappings_per_chunk = replay_mappings_per_expanded_batch(n_points);
        for chunk_start in (0..mappings.len()).step_by(mappings_per_chunk) {
            let chunk_end = usize::min(chunk_start + mappings_per_chunk, mappings.len());
            let mapping_chunk = &mappings[chunk_start..chunk_end];
            let weight_chunk = &weights[chunk_start..chunk_end];
            let expanded_batch =
                apply_lc_topology_label_permutations(batch, self.external_count, mapping_chunk)?;
            let (expanded_values, sector_profile) = self.run_f64_materialized(&expanded_batch)?;
            for (mapping_index, weight) in weight_chunk.iter().copied().enumerate() {
                let offset = mapping_index * n_points;
                for point_index in 0..n_points {
                    values[point_index] += weight * expanded_values[offset + point_index];
                }
            }
            profile.add_sector(&sector_profile);
        }
        profile.total_s = total_start.elapsed().as_secs_f64();
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
        let state_len = n_points * self.parameter_count;
        // Sources, momenta, model parameters, and every generated stage output
        // are overwritten on each call. Slots that are never generated stay at
        // their initialization value, so clearing the full state would only
        // rewrite previously initialized memory between evaluations.
        self.state_scratch_f64.resize(state_len, c64(0.0, 0.0));
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
        let source_fill_s = source_start.elapsed().as_secs_f64();

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
        let momentum_setup_s = momentum_start.elapsed().as_secs_f64();

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
        let model_parameter_setup_s = model_parameter_start_time.elapsed().as_secs_f64();

        let mut stage_input_pack_by_stage_s = Vec::new();
        let mut stage_evaluator_call_by_stage_s = Vec::new();
        let mut stage_output_assign_by_stage_s = Vec::new();
        for stage in self.stages.as_mut().expect("generic stages checked") {
            let (pack_s, eval_s, assign_s) = stage.evaluate_f64_into_state(
                n_points,
                self.parameter_count,
                state.as_mut_slice(),
            )?;
            stage_input_pack_by_stage_s.push(pack_s);
            stage_evaluator_call_by_stage_s.push(eval_s);
            stage_output_assign_by_stage_s.push(assign_s);
        }

        let (amplitude_input_pack_s, amplitude_evaluator_call_s) = self
            .amplitude_stage
            .as_mut()
            .expect("generic amplitude stage checked")
            .evaluate_f64_into_scratch(n_points, state.as_slice())?;
        let amplitude_evaluator_s = amplitude_input_pack_s + amplitude_evaluator_call_s;

        let reduction_start = Instant::now();
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
        let stage_input_pack_s = stage_input_pack_by_stage_s.iter().sum::<f64>();
        let stage_evaluator_call_s = stage_evaluator_call_by_stage_s.iter().sum::<f64>();
        let stage_evaluator_s = stage_input_pack_s + stage_evaluator_call_s;
        let output_assign_s = stage_output_assign_by_stage_s.iter().sum::<f64>();
        Ok((
            self.values_scratch_f64.clone(),
            RuntimeProfile {
                source_fill_s,
                momentum_setup_s: momentum_setup_s + model_parameter_setup_s,
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
        let total_start = Instant::now();
        let n_points = batch.len();
        let mut values = vec![T::new_zero(); n_points];
        let mut profile = RuntimeProfile::default();
        let mappings = self.lc_topology_replay_mappings.clone();
        let weights = self.lc_topology_replay_weights.clone();
        let mappings_per_chunk = replay_mappings_per_expanded_batch(n_points);
        for chunk_start in (0..mappings.len()).step_by(mappings_per_chunk) {
            let chunk_end = usize::min(chunk_start + mappings_per_chunk, mappings.len());
            let mapping_chunk = &mappings[chunk_start..chunk_end];
            let weight_chunk = &weights[chunk_start..chunk_end];
            let expanded_batch = apply_lc_topology_label_permutations_generic(
                batch,
                self.external_count,
                mapping_chunk,
            )?;
            let (expanded_values, sector_profile) =
                self.run_generic_materialized(&expanded_batch, binary_precision)?;
            for (mapping_index, weight) in weight_chunk.iter().copied().enumerate() {
                let weight = T::from(weight);
                let offset = mapping_index * n_points;
                for point_index in 0..n_points {
                    values[point_index] +=
                        weight.clone() * expanded_values[offset + point_index].clone();
                }
            }
            profile.add_sector(&sector_profile);
        }
        profile.total_s = total_start.elapsed().as_secs_f64();
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
        let source_fill_s = source_start.elapsed().as_secs_f64();

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
        let momentum_setup_s = momentum_start.elapsed().as_secs_f64();

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
        let model_parameter_setup_s = model_parameter_start_time.elapsed().as_secs_f64();

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
                source_fill_s,
                momentum_setup_s: momentum_setup_s + model_parameter_setup_s,
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
        if self.lc_topology_replay_enabled {
            return self.run_resolved_generic_with_lc_topology_replay(
                batch,
                binary_precision,
                selected_helicity_ids,
                selected_color_ids,
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
        let replay_plan = physics.lc_resolved_replay_plan(
            &self.lc_topology_replay_public_mappings,
            &self.lc_topology_replay_weights,
        )?;
        let total_start = Instant::now();
        let n_points = batch.len();
        let component_count = replay_plan.helicity_count * replay_plan.color_count;
        let mut full_values = vec![T::new_zero(); n_points * component_count];
        let mut profile = RuntimeProfile::default();
        let mappings = self.lc_topology_replay_mappings.clone();
        let mappings_per_chunk = replay_mappings_per_expanded_batch(n_points);
        for chunk_start in (0..mappings.len()).step_by(mappings_per_chunk) {
            let chunk_end = usize::min(chunk_start + mappings_per_chunk, mappings.len());
            let mapping_chunk = &mappings[chunk_start..chunk_end];
            let expanded_batch = apply_lc_topology_label_permutations_generic(
                batch,
                self.external_count,
                mapping_chunk,
            )?;
            let (materialized, sector_profile) = self.run_resolved_generic_materialized(
                &expanded_batch,
                binary_precision,
                None,
                None,
            )?;
            profile.add_sector(&sector_profile);
            let reduction_start = Instant::now();
            accumulate_lc_replay_resolved_generic(
                &mut full_values,
                n_points,
                &materialized,
                &replay_plan.entries[chunk_start..chunk_end],
                component_count,
            )?;
            profile.reduction_s += reduction_start.elapsed().as_secs_f64();
        }
        profile.total_s = total_start.elapsed().as_secs_f64();
        let resolved = select_resolved_values(
            full_values,
            n_points,
            &physics,
            selected_helicity_ids,
            selected_color_ids,
        )?;
        Ok((resolved, profile))
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
        let source_fill_s = source_start.elapsed().as_secs_f64();

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
        let momentum_setup_s = momentum_start.elapsed().as_secs_f64();

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
        let model_parameter_setup_s = model_parameter_start_time.elapsed().as_secs_f64();

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
                source_fill_s,
                momentum_setup_s: momentum_setup_s + model_parameter_setup_s,
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
            },
        ))
    }
}

pub(super) fn accumulate_lc_replay_resolved_f64(
    target: &mut [f64],
    point_count: usize,
    materialized: &ResolvedValues<f64>,
    replay_entries: &[LcResolvedReplayEntry],
    component_count: usize,
) -> RusticolResult<()> {
    validate_materialized_replay_shape(
        materialized,
        point_count,
        replay_entries.len(),
        component_count,
    )?;
    if target.len() != point_count * component_count {
        return Err(RusticolError::invalid_argument(
            "LC topology replay target has an inconsistent resolved shape",
        ));
    }
    for (entry_index, entry) in replay_entries.iter().enumerate() {
        for point_index in 0..point_count {
            let source_row = (entry_index * point_count + point_index) * component_count;
            let target_row = point_index * component_count;
            for route in &entry.routes {
                target[target_row + route.target_index] +=
                    route.weight * materialized.values[source_row + route.source_index];
            }
        }
    }
    Ok(())
}

#[cfg(feature = "symbolica-runtime")]
fn accumulate_lc_replay_resolved_generic<T>(
    target: &mut [T],
    point_count: usize,
    materialized: &ResolvedValues<T>,
    replay_entries: &[LcResolvedReplayEntry],
    component_count: usize,
) -> RusticolResult<()>
where
    T: RusticolHighPrecisionNumber,
    Complex<T>: Real + EvaluationDomain,
{
    validate_materialized_replay_shape(
        materialized,
        point_count,
        replay_entries.len(),
        component_count,
    )?;
    if target.len() != point_count * component_count {
        return Err(RusticolError::invalid_argument(
            "LC topology replay target has an inconsistent resolved shape",
        ));
    }
    for (entry_index, entry) in replay_entries.iter().enumerate() {
        for point_index in 0..point_count {
            let source_row = (entry_index * point_count + point_index) * component_count;
            let target_row = point_index * component_count;
            for route in &entry.routes {
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
