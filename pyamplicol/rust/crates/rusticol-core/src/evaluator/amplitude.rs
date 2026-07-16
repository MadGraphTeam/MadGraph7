// SPDX-License-Identifier: 0BSD

use super::super::*;
use super::*;

impl AmplitudeRuntime {
    pub(crate) fn load(
        amplitude_stage: &GenericAmplitudeStageManifest,
        stage: &GenericSerializedStageEvaluatorManifest,
        root: &Path,
    ) -> RusticolResult<Self> {
        if stage.stage_kind != "amplitude-roots" {
            return Err(RusticolError::invalid_argument(
                "generic amplitude runtime expected an amplitude-roots stage",
            ));
        }
        let raw_sum_weights = amplitude_stage
            .roots
            .iter()
            .map(|root| root.helicity_weight)
            .collect::<Vec<_>>();
        let raw_sum_all_sector_weights = amplitude_stage
            .roots
            .iter()
            .map(|root| root.all_sector_weight.unwrap_or(root.helicity_weight))
            .collect::<Vec<_>>();
        let raw_sum_color_sector_ids = amplitude_stage
            .roots
            .iter()
            .map(|root| root.color_sector_id)
            .collect::<Vec<_>>();
        let raw_sum_group_ids = amplitude_stage
            .roots
            .iter()
            .map(generic_root_group_id)
            .collect::<RusticolResult<Vec<_>>>()?;
        let has_coherent_groups = raw_sum_group_ids.iter().any(Option::is_some);
        let raw_sum_groups = if has_coherent_groups {
            build_raw_sum_groups(
                amplitude_stage.output_count,
                &raw_sum_weights,
                &raw_sum_all_sector_weights,
                &raw_sum_group_ids,
                &raw_sum_color_sector_ids,
            )?
        } else {
            Vec::new()
        };
        let color_contraction = build_color_contraction_runtime(
            amplitude_stage.color_contraction.as_ref(),
            &raw_sum_groups,
        )?;
        let (input_components, input_spans) =
            if stage.parameter_layout == "stage-local-value-momentum" {
                let mut map = vec![0usize; stage.parameter_count];
                for component in &stage.input_components {
                    map[component.parameter_index] = component.global_component;
                }
                let spans = contiguous_input_spans(&map);
                (Some(map), spans)
            } else {
                (None, Vec::new())
            };
        Ok(Self {
            output_length: amplitude_stage.output_count,
            raw_sum_weights,
            raw_sum_all_sector_weights,
            raw_sum_color_sector_ids,
            raw_sum_groups,
            has_coherent_groups,
            color_contraction,
            input_components,
            input_spans,
            parameter_scratch_f64: Vec::new(),
            output_scratch_f64: Vec::new(),
            parameter_scratch_native2: Vec::new(),
            output_scratch_native2: Vec::new(),
            evaluator: EvaluatorGroup::load(&stage.evaluator, root)?,
        })
    }

    pub(crate) fn evaluate_f64_into_scratch(
        &mut self,
        batch_size: usize,
        state: &[Complex<f64>],
    ) -> RusticolResult<(f64, f64)> {
        if self.evaluator.supports_native2() {
            let global_parameter_count = state.len().checked_div(batch_size).ok_or_else(|| {
                RusticolError::invalid_argument("generic amplitude batch size is zero")
            })?;
            let pack_start = Instant::now();
            pack_native2_parameters(
                batch_size,
                global_parameter_count,
                self.input_components.as_deref(),
                state,
                &mut self.parameter_scratch_native2,
            )?;
            let input_pack_s = pack_start.elapsed().as_secs_f64();

            let eval_start = Instant::now();
            self.evaluator.evaluate_native2_into(
                batch_size.div_ceil(2),
                &self.parameter_scratch_native2,
                &mut self.output_scratch_native2,
            )?;
            self.output_scratch_f64
                .resize(batch_size * self.evaluator.output_len, c64(0.0, 0.0));
            for row in 0..batch_size {
                let native_row = row / 2 * self.evaluator.output_len;
                let output_row = row * self.evaluator.output_len;
                let lane = row % 2;
                for column in 0..self.evaluator.output_len {
                    let value = self.output_scratch_native2[native_row + column];
                    self.output_scratch_f64[output_row + column] =
                        c64(value.re.as_array()[lane], value.im.as_array()[lane]);
                }
            }
            return Ok((input_pack_s, eval_start.elapsed().as_secs_f64()));
        }
        if let Some(input_components) = self.input_components.as_ref() {
            let local_parameter_count = input_components.len();
            let global_parameter_count = state.len().checked_div(batch_size).ok_or_else(|| {
                RusticolError::invalid_argument("generic amplitude batch size is zero")
            })?;
            let pack_start = Instant::now();
            self.parameter_scratch_f64
                .resize(batch_size * local_parameter_count, c64(0.0, 0.0));
            for row in 0..batch_size {
                let row_state = row * global_parameter_count;
                let row_params = row * local_parameter_count;
                if self.input_spans.is_empty() {
                    for (local_index, global_index) in input_components.iter().enumerate() {
                        self.parameter_scratch_f64[row_params + local_index] =
                            state[row_state + *global_index];
                    }
                } else {
                    for (local_start, global_start, len) in &self.input_spans {
                        let target_start = row_params + *local_start;
                        let source_start = row_state + *global_start;
                        self.parameter_scratch_f64[target_start..target_start + *len]
                            .copy_from_slice(&state[source_start..source_start + *len]);
                    }
                }
            }
            let input_pack_s = pack_start.elapsed().as_secs_f64();
            let eval_start = Instant::now();
            self.evaluator.evaluate_batch_into(
                batch_size,
                &self.parameter_scratch_f64,
                &mut self.output_scratch_f64,
            )?;
            return Ok((input_pack_s, eval_start.elapsed().as_secs_f64()));
        }
        let eval_start = Instant::now();
        self.evaluator
            .evaluate_batch_into(batch_size, state, &mut self.output_scratch_f64)?;
        Ok((0.0, eval_start.elapsed().as_secs_f64()))
    }

    pub(crate) fn reduce_scratch_f64_into_selected(
        &mut self,
        batch_size: usize,
        raw_sums: &mut Vec<f64>,
        selected_color_sector_ids: Option<&BTreeSet<i64>>,
    ) -> RusticolResult<()> {
        let amplitudes = &self.output_scratch_f64;
        if amplitudes.len() != batch_size * self.output_length {
            return Err(RusticolError::invalid_argument(format!(
                "generic amplitude output buffer has length {}, expected {}",
                amplitudes.len(),
                batch_size * self.output_length
            )));
        }
        raw_sums.clear();
        raw_sums.resize(batch_size, 0.0);
        if let Some(contraction) = self.color_contraction.as_mut() {
            if selected_color_sector_ids.is_some() {
                return Err(RusticolError::invalid_argument(
                    "LC color-sector runtime selection is only supported for leading-colour diagonal artifacts",
                ));
            }
            if self.raw_sum_groups.len() != contraction.group_count {
                return Err(RusticolError::invalid_argument(
                    "colour contraction group count does not match coherent groups",
                ));
            }
            contraction
                .group_scratch_f64
                .resize(batch_size * contraction.group_count, c64(0.0, 0.0));
            for (row, raw_sum) in raw_sums.iter_mut().enumerate() {
                let row_offset = row * self.output_length;
                let group_row = row * contraction.group_count;
                for (group_index, group) in self.raw_sum_groups.iter().enumerate() {
                    let mut sum = c64(0.0, 0.0);
                    for index in &group.indices {
                        sum += amplitudes[row_offset + *index];
                    }
                    contraction.group_scratch_f64[group_row + group_index] = sum;
                }
                for entry in &contraction.entries {
                    let left = contraction.group_scratch_f64[group_row + entry.left_group_index];
                    let right = contraction.group_scratch_f64[group_row + entry.right_group_index];
                    let product = left * right.conj();
                    *raw_sum += entry.symmetry_factor
                        * (entry.weight_re * product.re - entry.weight_im * product.im);
                }
            }
            return Ok(());
        }
        for (row, raw_sum) in raw_sums.iter_mut().enumerate() {
            let row_offset = row * self.output_length;
            if self.has_coherent_groups {
                for group in &self.raw_sum_groups {
                    if !raw_sum_group_is_selected(group, selected_color_sector_ids) {
                        continue;
                    }
                    let mut sum = c64(0.0, 0.0);
                    for index in &group.indices {
                        sum += amplitudes[row_offset + *index];
                    }
                    let weight = if selected_color_sector_ids.is_none() {
                        group.all_sector_weight
                    } else {
                        group.weight
                    };
                    *raw_sum += weight * (sum.re * sum.re + sum.im * sum.im);
                }
                continue;
            }
            for index in 0..self.output_length {
                if !raw_sum_index_is_selected(
                    self.raw_sum_color_sector_ids.get(index).copied().flatten(),
                    selected_color_sector_ids,
                ) {
                    continue;
                }
                let value = amplitudes[row_offset + index];
                let weight = if selected_color_sector_ids.is_none() {
                    self.raw_sum_all_sector_weights[index]
                } else {
                    self.raw_sum_weights[index]
                };
                *raw_sum += weight * (value.re * value.re + value.im * value.im);
            }
        }
        Ok(())
    }

    pub(crate) fn reduce_scratch_f64_resolved(
        &mut self,
        batch_size: usize,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<ResolvedValues<f64>> {
        let amplitudes = &self.output_scratch_f64;
        if amplitudes.len() != batch_size * self.output_length {
            return Err(RusticolError::invalid_argument(format!(
                "generic amplitude output buffer has length {}, expected {}",
                amplitudes.len(),
                batch_size * self.output_length
            )));
        }
        let helicity_count = physics.manifest.helicities.len();
        let color_count = physics.manifest.color_components.len();
        let mut full_values = vec![0.0; batch_size * helicity_count * color_count];

        if let Some(contraction) = self.color_contraction.as_mut() {
            if color_count != 1 || !physics.has_contracted_color_axis() {
                return Err(RusticolError::invalid_argument(
                    "resolved NLC/full evaluation requires one contracted color component",
                ));
            }
            if self.raw_sum_groups.len() != contraction.group_count {
                return Err(RusticolError::invalid_argument(
                    "colour contraction group count does not match coherent groups",
                ));
            }
            contraction
                .group_scratch_f64
                .resize(batch_size * contraction.group_count, c64(0.0, 0.0));
            for row in 0..batch_size {
                let row_offset = row * self.output_length;
                let group_row = row * contraction.group_count;
                for (group_index, group) in self.raw_sum_groups.iter().enumerate() {
                    let mut sum = c64(0.0, 0.0);
                    for index in &group.indices {
                        sum += amplitudes[row_offset + *index];
                    }
                    contraction.group_scratch_f64[group_row + group_index] = sum;
                }
                for entry in &contraction.entries {
                    let left_group = &self.raw_sum_groups[entry.left_group_index];
                    let right_group = &self.raw_sum_groups[entry.right_group_index];
                    let left_reduction = physics
                        .reduction_by_group_id
                        .get(&left_group.id)
                        .ok_or_else(|| {
                            RusticolError::invalid_argument(format!(
                                "resolved metadata is missing coherent group {}",
                                left_group.id
                            ))
                        })?;
                    let right_reduction = physics
                        .reduction_by_group_id
                        .get(&right_group.id)
                        .ok_or_else(|| {
                            RusticolError::invalid_argument(format!(
                                "resolved metadata is missing coherent group {}",
                                right_group.id
                            ))
                        })?;
                    if left_reduction.physical_helicity_ids != right_reduction.physical_helicity_ids
                    {
                        return Err(RusticolError::invalid_argument(
                            "colour contraction mixed distinct physical helicities",
                        ));
                    }
                    let left_weights = physics.normalized_helicity_weights(left_reduction)?;
                    let right_weights = physics.normalized_helicity_weights(right_reduction)?;
                    if left_weights.len() != right_weights.len()
                        || left_weights.iter().zip(&right_weights).any(
                            |((left_index, left_weight), (right_index, right_weight))| {
                                left_index != right_index
                                    || left_weight.to_bits() != right_weight.to_bits()
                            },
                        )
                    {
                        return Err(RusticolError::invalid_argument(
                            "colour contraction groups have inconsistent helicity weights",
                        ));
                    }
                    let left = contraction.group_scratch_f64[group_row + entry.left_group_index];
                    let right = contraction.group_scratch_f64[group_row + entry.right_group_index];
                    let product = left * right.conj();
                    let contribution = normalization_factor
                        * entry.symmetry_factor
                        * (entry.weight_re * product.re - entry.weight_im * product.im);
                    for (helicity_index, weight) in left_weights {
                        full_values[(row * helicity_count + helicity_index) * color_count] +=
                            contribution * weight;
                    }
                }
            }
        } else {
            if !self.has_coherent_groups {
                return Err(RusticolError::invalid_argument(
                    "resolved evaluation requires coherent amplitude-group metadata",
                ));
            }
            for row in 0..batch_size {
                let row_offset = row * self.output_length;
                for group in &self.raw_sum_groups {
                    let reduction =
                        physics
                            .reduction_by_group_id
                            .get(&group.id)
                            .ok_or_else(|| {
                                RusticolError::invalid_argument(format!(
                                    "resolved metadata is missing coherent group {}",
                                    group.id
                                ))
                            })?;
                    let mut sum = c64(0.0, 0.0);
                    for index in &group.indices {
                        sum += amplitudes[row_offset + *index];
                    }
                    let contribution = normalization_factor
                        * group.all_sector_weight
                        * (sum.re * sum.re + sum.im * sum.im);
                    for (helicity_index, color_index, weight) in
                        physics.normalized_member_weights(reduction)?
                    {
                        full_values[(row * helicity_count + helicity_index) * color_count
                            + color_index] += contribution * weight;
                    }
                }
            }
        }

        let helicity_indices = physics.selected_helicity_indices(selected_helicity_ids)?;
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        let mut values =
            Vec::with_capacity(batch_size * helicity_indices.len() * color_indices.len());
        for row in 0..batch_size {
            for helicity_index in &helicity_indices {
                for color_index in &color_indices {
                    values.push(
                        full_values
                            [(row * helicity_count + *helicity_index) * color_count + *color_index],
                    );
                }
            }
        }
        Ok(ResolvedValues {
            values,
            point_count: batch_size,
            helicity_indices,
            color_indices,
        })
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn evaluate_outputs_generic<T>(
        &mut self,
        batch_size: usize,
        state: &[Complex<T>],
        binary_precision: Option<u32>,
    ) -> RusticolResult<(Vec<Complex<T>>, f64, f64)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let mut input_pack_s = 0.0;
        let (evaluated, evaluator_call_s) = if let Some(input_components) =
            self.input_components.as_ref()
        {
            let local_parameter_count = input_components.len();
            let global_parameter_count = state.len().checked_div(batch_size).ok_or_else(|| {
                RusticolError::invalid_argument("generic amplitude batch size is zero")
            })?;
            let pack_start = Instant::now();
            let mut parameter_scratch =
                vec![complex_zero::<T>(); batch_size * local_parameter_count];
            for row in 0..batch_size {
                let row_state = row * global_parameter_count;
                let row_params = row * local_parameter_count;
                if self.input_spans.is_empty() {
                    for (local_index, global_index) in input_components.iter().enumerate() {
                        parameter_scratch[row_params + local_index] =
                            state[row_state + *global_index].clone();
                    }
                } else {
                    for (local_start, global_start, len) in &self.input_spans {
                        let target_start = row_params + *local_start;
                        let source_start = row_state + *global_start;
                        parameter_scratch[target_start..target_start + *len]
                            .clone_from_slice(&state[source_start..source_start + *len]);
                    }
                }
            }
            input_pack_s = pack_start.elapsed().as_secs_f64();
            let eval_start = Instant::now();
            let evaluated = self.evaluator.evaluate_batch_generic(
                batch_size,
                &parameter_scratch,
                binary_precision,
            )?;
            (evaluated, eval_start.elapsed().as_secs_f64())
        } else {
            let eval_start = Instant::now();
            let evaluated =
                self.evaluator
                    .evaluate_batch_generic(batch_size, state, binary_precision)?;
            (evaluated, eval_start.elapsed().as_secs_f64())
        };
        if evaluated.len() != batch_size * self.output_length {
            return Err(RusticolError::invalid_argument(format!(
                "generic amplitude output buffer has length {}, expected {}",
                evaluated.len(),
                batch_size * self.output_length
            )));
        }
        Ok((evaluated, input_pack_s, evaluator_call_s))
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn evaluate_raw_sums_generic<T>(
        &mut self,
        batch_size: usize,
        state: &[Complex<T>],
        binary_precision: Option<u32>,
    ) -> RusticolResult<(Vec<T>, f64, f64)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let (evaluated, input_pack_s, evaluator_call_s) =
            self.evaluate_outputs_generic(batch_size, state, binary_precision)?;
        let mut raw_sums = vec![T::new_zero(); batch_size];
        if let Some(contraction) = self.color_contraction.as_ref() {
            if self.raw_sum_groups.len() != contraction.group_count {
                return Err(RusticolError::invalid_argument(
                    "colour contraction group count does not match coherent groups",
                ));
            }
            let mut group_values = vec![complex_zero::<T>(); batch_size * contraction.group_count];
            for (row, raw_sum) in raw_sums.iter_mut().enumerate() {
                let row_offset = row * self.output_length;
                let group_row = row * contraction.group_count;
                for (group_index, group) in self.raw_sum_groups.iter().enumerate() {
                    let mut sum = complex_zero::<T>();
                    for index in &group.indices {
                        sum.re += evaluated[row_offset + *index].re.clone();
                        sum.im += evaluated[row_offset + *index].im.clone();
                    }
                    group_values[group_row + group_index] = sum;
                }
                for entry in &contraction.entries {
                    let left = &group_values[group_row + entry.left_group_index];
                    let right = &group_values[group_row + entry.right_group_index];
                    let product_re =
                        left.re.clone() * right.re.clone() + left.im.clone() * right.im.clone();
                    let product_im =
                        left.im.clone() * right.re.clone() - left.re.clone() * right.im.clone();
                    *raw_sum += T::from(entry.symmetry_factor)
                        * (T::from(entry.weight_re) * product_re
                            - T::from(entry.weight_im) * product_im);
                }
            }
            return Ok((raw_sums, input_pack_s, evaluator_call_s));
        }
        for (row, raw_sum) in raw_sums.iter_mut().enumerate() {
            let row_offset = row * self.output_length;
            if self.has_coherent_groups {
                for group in &self.raw_sum_groups {
                    let mut sum_re = T::new_zero();
                    let mut sum_im = T::new_zero();
                    for index in &group.indices {
                        let value = &evaluated[row_offset + *index];
                        sum_re += value.re.clone();
                        sum_im += value.im.clone();
                    }
                    *raw_sum += T::from(group.all_sector_weight)
                        * (sum_re.clone() * sum_re + sum_im.clone() * sum_im);
                }
                continue;
            }
            for index in 0..self.output_length {
                let value = &evaluated[row_offset + index];
                *raw_sum += T::from(self.raw_sum_all_sector_weights[index])
                    * (value.re.clone() * value.re.clone() + value.im.clone() * value.im.clone());
            }
        }
        Ok((raw_sums, input_pack_s, evaluator_call_s))
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn evaluate_resolved_generic<T>(
        &mut self,
        batch_size: usize,
        state: &[Complex<T>],
        binary_precision: Option<u32>,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<T>, f64, f64)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let (evaluated, input_pack_s, evaluator_call_s) =
            self.evaluate_outputs_generic(batch_size, state, binary_precision)?;
        let helicity_count = physics.manifest.helicities.len();
        let color_count = physics.manifest.color_components.len();
        let mut full_values = vec![T::new_zero(); batch_size * helicity_count * color_count];
        if let Some(contraction) = self.color_contraction.as_ref() {
            if color_count != 1 || !physics.has_contracted_color_axis() {
                return Err(RusticolError::invalid_argument(
                    "resolved NLC/full evaluation requires one contracted color component",
                ));
            }
            let mut group_values = vec![complex_zero::<T>(); batch_size * contraction.group_count];
            for row in 0..batch_size {
                let row_offset = row * self.output_length;
                let group_row = row * contraction.group_count;
                for (group_index, group) in self.raw_sum_groups.iter().enumerate() {
                    let mut sum = complex_zero::<T>();
                    for index in &group.indices {
                        sum.re += evaluated[row_offset + *index].re.clone();
                        sum.im += evaluated[row_offset + *index].im.clone();
                    }
                    group_values[group_row + group_index] = sum;
                }
                for entry in &contraction.entries {
                    let left_group = &self.raw_sum_groups[entry.left_group_index];
                    let right_group = &self.raw_sum_groups[entry.right_group_index];
                    let left_reduction = physics
                        .reduction_by_group_id
                        .get(&left_group.id)
                        .ok_or_else(|| {
                            RusticolError::invalid_argument(format!(
                                "resolved metadata is missing coherent group {}",
                                left_group.id
                            ))
                        })?;
                    let right_reduction = physics
                        .reduction_by_group_id
                        .get(&right_group.id)
                        .ok_or_else(|| {
                            RusticolError::invalid_argument(format!(
                                "resolved metadata is missing coherent group {}",
                                right_group.id
                            ))
                        })?;
                    if left_reduction.physical_helicity_ids != right_reduction.physical_helicity_ids
                    {
                        return Err(RusticolError::invalid_argument(
                            "colour contraction mixed distinct physical helicities",
                        ));
                    }
                    let left_weights = physics.normalized_helicity_weights(left_reduction)?;
                    let right_weights = physics.normalized_helicity_weights(right_reduction)?;
                    if left_weights.len() != right_weights.len()
                        || left_weights.iter().zip(&right_weights).any(
                            |((left_index, left_weight), (right_index, right_weight))| {
                                left_index != right_index
                                    || left_weight.to_bits() != right_weight.to_bits()
                            },
                        )
                    {
                        return Err(RusticolError::invalid_argument(
                            "colour contraction groups have inconsistent helicity weights",
                        ));
                    }
                    let left = &group_values[group_row + entry.left_group_index];
                    let right = &group_values[group_row + entry.right_group_index];
                    let product_re =
                        left.re.clone() * right.re.clone() + left.im.clone() * right.im.clone();
                    let product_im =
                        left.im.clone() * right.re.clone() - left.re.clone() * right.im.clone();
                    let coefficient = normalization_factor * entry.symmetry_factor;
                    let contribution = T::from(coefficient)
                        * (T::from(entry.weight_re) * product_re
                            - T::from(entry.weight_im) * product_im);
                    for (helicity_index, weight) in left_weights {
                        full_values[(row * helicity_count + helicity_index) * color_count] +=
                            contribution.clone() * T::from(weight);
                    }
                }
            }
        } else {
            for row in 0..batch_size {
                let row_offset = row * self.output_length;
                for group in &self.raw_sum_groups {
                    let reduction =
                        physics
                            .reduction_by_group_id
                            .get(&group.id)
                            .ok_or_else(|| {
                                RusticolError::invalid_argument(format!(
                                    "resolved metadata is missing coherent group {}",
                                    group.id
                                ))
                            })?;
                    let mut sum_re = T::new_zero();
                    let mut sum_im = T::new_zero();
                    for index in &group.indices {
                        sum_re += evaluated[row_offset + *index].re.clone();
                        sum_im += evaluated[row_offset + *index].im.clone();
                    }
                    let contribution = T::from(normalization_factor * group.all_sector_weight)
                        * (sum_re.clone() * sum_re + sum_im.clone() * sum_im);
                    for (helicity_index, color_index, weight) in
                        physics.normalized_member_weights(reduction)?
                    {
                        full_values[(row * helicity_count + helicity_index) * color_count
                            + color_index] += contribution.clone() * T::from(weight);
                    }
                }
            }
        }
        let helicity_indices = physics.selected_helicity_indices(selected_helicity_ids)?;
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        let mut values =
            Vec::with_capacity(batch_size * helicity_indices.len() * color_indices.len());
        for row in 0..batch_size {
            for helicity_index in &helicity_indices {
                for color_index in &color_indices {
                    values.push(
                        full_values
                            [(row * helicity_count + *helicity_index) * color_count + *color_index]
                            .clone(),
                    );
                }
            }
        }
        Ok((
            ResolvedValues {
                values,
                point_count: batch_size,
                helicity_indices,
                color_indices,
            },
            input_pack_s,
            evaluator_call_s,
        ))
    }
}

pub(crate) fn build_raw_sum_groups(
    output_length: usize,
    weights: &[f64],
    all_sector_weights: &[f64],
    group_ids: &[Option<i64>],
    color_sector_ids: &[Option<i64>],
) -> RusticolResult<Vec<RawSumGroup>> {
    if weights.len() != output_length
        || all_sector_weights.len() != output_length
        || group_ids.len() != output_length
        || color_sector_ids.len() != output_length
    {
        return Err(RusticolError::invalid_argument(
            "raw-sum group metadata length does not match amplitude outputs",
        ));
    }
    let mut grouped: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    let mut groups = Vec::new();
    for index in 0..output_length {
        if let Some(group_id) = group_ids[index] {
            grouped.entry(group_id).or_default().push(index);
        } else {
            groups.push(RawSumGroup {
                id: index as i64,
                indices: vec![index],
                weight: weights[index],
                all_sector_weight: all_sector_weights[index],
                sector_ids: color_sector_ids[index].into_iter().collect(),
            });
        }
    }
    for (group_id, indices) in grouped {
        let weight = weights[indices[0]];
        let all_sector_weight = all_sector_weights[indices[0]];
        if indices
            .iter()
            .any(|index| (weights[*index] - weight).abs() > 0.0)
        {
            return Err(RusticolError::invalid_argument(format!(
                "coherent amplitude group {group_id} has inconsistent raw-sum weights"
            )));
        }
        if indices
            .iter()
            .any(|index| (all_sector_weights[*index] - all_sector_weight).abs() > 0.0)
        {
            return Err(RusticolError::invalid_argument(format!(
                "coherent amplitude group {group_id} has inconsistent all-sector raw-sum weights"
            )));
        }
        groups.push(RawSumGroup {
            id: group_id,
            sector_ids: unique_color_sector_ids(&indices, color_sector_ids),
            indices,
            weight,
            all_sector_weight,
        });
    }
    Ok(groups)
}

pub(crate) fn unique_color_sector_ids(
    indices: &[usize],
    color_sector_ids: &[Option<i64>],
) -> Vec<i64> {
    indices
        .iter()
        .filter_map(|index| color_sector_ids.get(*index).copied().flatten())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

pub(crate) fn raw_sum_group_is_selected(
    group: &RawSumGroup,
    selected_color_sector_ids: Option<&BTreeSet<i64>>,
) -> bool {
    let Some(selected) = selected_color_sector_ids else {
        return true;
    };
    group
        .sector_ids
        .iter()
        .any(|sector_id| selected.contains(sector_id))
}

pub(crate) fn raw_sum_index_is_selected(
    sector_id: Option<i64>,
    selected_color_sector_ids: Option<&BTreeSet<i64>>,
) -> bool {
    let Some(selected) = selected_color_sector_ids else {
        return true;
    };
    sector_id
        .map(|value| selected.contains(&value))
        .unwrap_or(false)
}

pub(crate) fn build_color_contraction_runtime(
    manifest: Option<&GenericColorContractionManifest>,
    groups: &[RawSumGroup],
) -> RusticolResult<Option<ColorContractionRuntime>> {
    let Some(manifest) = manifest else {
        return Ok(None);
    };
    if !manifest.supported {
        return Err(RusticolError::invalid_argument(format!(
            "generic colour contraction is unsupported: {}",
            manifest
                .reason
                .as_deref()
                .unwrap_or("no diagnostic was provided")
        )));
    }
    let group_index_by_id = groups
        .iter()
        .enumerate()
        .map(|(index, group)| (group.id, index))
        .collect::<BTreeMap<_, _>>();
    if group_index_by_id.len() != manifest.group_count {
        return Err(RusticolError::invalid_argument(format!(
            "colour contraction declares {} groups but runtime has {} coherent groups",
            manifest.group_count,
            group_index_by_id.len()
        )));
    }
    let mut entries = Vec::with_capacity(manifest.entries.len());
    for entry in &manifest.entries {
        let left_group_index = *group_index_by_id.get(&entry.left_group_id).ok_or_else(|| {
            RusticolError::invalid_argument(format!(
                "colour contraction references unknown left group {}",
                entry.left_group_id
            ))
        })?;
        let right_group_index = *group_index_by_id
            .get(&entry.right_group_id)
            .ok_or_else(|| {
                RusticolError::invalid_argument(format!(
                    "colour contraction references unknown right group {}",
                    entry.right_group_id
                ))
            })?;
        let weight_re = entry.weight.first().copied().unwrap_or(0.0);
        let weight_im = entry.weight.get(1).copied().unwrap_or(0.0);
        entries.push(ColorContractionEntry {
            left_group_index,
            right_group_index,
            weight_re,
            weight_im,
            symmetry_factor: entry.symmetry_factor,
        });
    }
    Ok(Some(ColorContractionRuntime {
        group_count: manifest.group_count,
        entries,
        group_scratch_f64: Vec::new(),
    }))
}

pub(crate) fn generic_root_group_id(
    root: &GenericAmplitudeRootManifest,
) -> RusticolResult<Option<i64>> {
    let Some(value) = root.coherent_group_id.as_ref() else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    if let Some(group_id) = value.as_i64() {
        return Ok(Some(group_id));
    }
    if let Some(group_id) = value.as_u64() {
        return i64::try_from(group_id)
            .map(Some)
            .map_err(|_| RusticolError::invalid_argument("generic coherent group id exceeds i64"));
    }
    if let Some(text) = value.as_str() {
        return text.parse::<i64>().map(Some).map_err(|err| {
            RusticolError::invalid_argument(format!(
                "could not parse generic coherent group id {text:?}: {err}"
            ))
        });
    }
    Err(RusticolError::invalid_argument(format!(
        "generic coherent group id for root {} must be an integer or string",
        root.root_id
    )))
}
