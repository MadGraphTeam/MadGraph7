// SPDX-License-Identifier: 0BSD

use super::super::*;
use super::*;

#[inline(always)]
fn sum_real_hermitian_products(left: &[Complex<f64>], right: &[Complex<f64>]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    let mut sums = [0.0; 4];
    let mut index = 0;
    while index + 4 <= left.len() {
        for lane in 0..4 {
            let left_value = left[index + lane];
            let right_value = right[index + lane];
            sums[lane] += left_value.re * right_value.re + left_value.im * right_value.im;
        }
        index += 4;
    }
    while index < left.len() {
        let left_value = left[index];
        let right_value = right[index];
        sums[0] += left_value.re * right_value.re + left_value.im * right_value.im;
        index += 1;
    }
    (sums[0] + sums[1]) + (sums[2] + sums[3])
}

#[inline(always)]
fn sum_hermitian_products(left: &[Complex<f64>], right: &[Complex<f64>]) -> Complex<f64> {
    debug_assert_eq!(left.len(), right.len());
    let mut sum = c64(0.0, 0.0);
    for (left_value, right_value) in left.iter().zip(right) {
        sum.re += left_value.re * right_value.re + left_value.im * right_value.im;
        sum.im += left_value.im * right_value.re - left_value.re * right_value.im;
    }
    sum
}

fn remap_amplitude_outputs(
    batch_size: usize,
    evaluator_output_order: &[usize],
    evaluator_outputs: &[Complex<f64>],
    canonical_outputs: &mut Vec<Complex<f64>>,
) -> RusticolResult<()> {
    let output_length = evaluator_output_order.len();
    let expected = batch_size
        .checked_mul(output_length)
        .ok_or_else(|| RusticolError::invalid_argument("amplitude output length overflows"))?;
    if evaluator_outputs.len() != expected {
        return Err(RusticolError::integrity(
            "amplitude evaluator output mapping has an inconsistent buffer length",
        ));
    }
    canonical_outputs.clear();
    canonical_outputs.resize(expected, c64(0.0, 0.0));
    for row in 0..batch_size {
        let row_offset = row * output_length;
        for (evaluator_output, canonical_output) in
            evaluator_output_order.iter().copied().enumerate()
        {
            canonical_outputs[row_offset + canonical_output] =
                evaluator_outputs[row_offset + evaluator_output];
        }
    }
    Ok(())
}

impl AmplitudeRuntime {
    pub(crate) fn load(
        amplitude_stage: &GenericAmplitudeStageManifest,
        stage: &GenericSerializedStageEvaluatorManifest,
        payloads: &EvaluatorPayloadStore,
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
        let mut evaluator_output_order = vec![usize::MAX; stage.output_length];
        let mut canonical_outputs = vec![false; stage.output_length];
        for slot in &stage.output_slots {
            let output_len = slot
                .output_stop
                .checked_sub(slot.output_start)
                .ok_or_else(|| {
                    RusticolError::artifact("amplitude evaluator has an invalid output range")
                })?;
            let component_len = slot
                .component_stop
                .checked_sub(slot.component_start)
                .ok_or_else(|| {
                    RusticolError::artifact("amplitude evaluator has an invalid component range")
                })?;
            if output_len != component_len {
                return Err(RusticolError::artifact(
                    "amplitude evaluator output and component ranges disagree",
                ));
            }
            for offset in 0..output_len {
                let evaluator_output = slot.output_start + offset;
                let canonical_output = slot.component_start + offset;
                if evaluator_output >= stage.output_length
                    || canonical_output >= stage.output_length
                    || evaluator_output_order[evaluator_output] != usize::MAX
                    || canonical_outputs[canonical_output]
                {
                    return Err(RusticolError::artifact(
                        "amplitude evaluator output mapping is not a permutation",
                    ));
                }
                evaluator_output_order[evaluator_output] = canonical_output;
                canonical_outputs[canonical_output] = true;
            }
        }
        if evaluator_output_order.contains(&usize::MAX)
            || canonical_outputs.iter().any(|covered| !covered)
        {
            return Err(RusticolError::artifact(
                "amplitude evaluator output mapping is incomplete",
            ));
        }
        let evaluator_output_order = evaluator_output_order
            .iter()
            .enumerate()
            .any(|(index, output)| index != *output)
            .then_some(evaluator_output_order);
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
            evaluator_output_scratch_f64: Vec::new(),
            output_scratch_f64: Vec::new(),
            resolved_source_row_scratch_f64: Vec::new(),
            resolved_target_row_scratch_f64: Vec::new(),
            evaluator_output_order,
            evaluator: EvaluatorGroup::load_from_store(&stage.evaluator, payloads)?,
        })
    }

    pub(crate) fn evaluate_f64_into_scratch(
        &mut self,
        batch_size: usize,
        state: &[Complex<f64>],
    ) -> RusticolResult<()> {
        let evaluator_params = if let Some(input_components) = self.input_components.as_ref() {
            let local_parameter_count = input_components.len();
            let global_parameter_count = state.len().checked_div(batch_size).ok_or_else(|| {
                RusticolError::invalid_argument("generic amplitude batch size is zero")
            })?;
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
            self.parameter_scratch_f64.as_slice()
        } else {
            state
        };
        if let Some(order) = self.evaluator_output_order.as_deref() {
            self.evaluator.evaluate_batch_into(
                batch_size,
                evaluator_params,
                &mut self.evaluator_output_scratch_f64,
            )?;
            remap_amplitude_outputs(
                batch_size,
                order,
                &self.evaluator_output_scratch_f64,
                &mut self.output_scratch_f64,
            )
        } else {
            self.evaluator.evaluate_batch_into(
                batch_size,
                evaluator_params,
                &mut self.output_scratch_f64,
            )
        }
    }

    pub(crate) fn evaluate_f64_into_scratch_profile(
        &mut self,
        batch_size: usize,
        state: &[Complex<f64>],
    ) -> RusticolResult<AmplitudeEvaluationProfile> {
        let mut input_pack_elapsed = Duration::ZERO;
        let input_copy_component_count;
        let mut scratch_reallocation_count = 0;
        let evaluator;
        let evaluator_params;
        if let Some(input_components) = self.input_components.as_ref() {
            let local_parameter_count = input_components.len();
            let global_parameter_count = state.len().checked_div(batch_size).ok_or_else(|| {
                RusticolError::invalid_argument("generic amplitude batch size is zero")
            })?;
            let pack_start = Instant::now();
            let capacity = self.parameter_scratch_f64.capacity();
            self.parameter_scratch_f64
                .resize(batch_size * local_parameter_count, c64(0.0, 0.0));
            scratch_reallocation_count +=
                u64::from(self.parameter_scratch_f64.capacity() != capacity);
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
            // Generated evaluators may use the platform floating-point ABI
            // aggressively. Keep timing state in integer-backed Duration
            // values until the generated call has returned.
            input_pack_elapsed = pack_start.elapsed();
            input_copy_component_count = batch_size * local_parameter_count;
            evaluator_params = self.parameter_scratch_f64.as_slice();
        } else {
            input_copy_component_count = 0;
            evaluator_params = state;
        }
        let mut output_remap_elapsed = Duration::ZERO;
        if let Some(order) = self.evaluator_output_order.as_deref() {
            evaluator = self.evaluator.evaluate_batch_into_profile(
                batch_size,
                evaluator_params,
                &mut self.evaluator_output_scratch_f64,
                false,
            )?;
            let remap_start = Instant::now();
            remap_amplitude_outputs(
                batch_size,
                order,
                &self.evaluator_output_scratch_f64,
                &mut self.output_scratch_f64,
            )?;
            output_remap_elapsed = remap_start.elapsed();
        } else {
            evaluator = self.evaluator.evaluate_batch_into_profile(
                batch_size,
                evaluator_params,
                &mut self.output_scratch_f64,
                false,
            )?;
        }
        let input_pack_s = profile_duration_seconds(input_pack_elapsed);
        Ok(AmplitudeEvaluationProfile {
            input_pack_s,
            evaluator,
            output_remap_s: profile_duration_seconds(output_remap_elapsed),
            input_copy_component_count: input_copy_component_count as u64,
            output_remap_component_count: if self.evaluator_output_order.is_some() {
                (batch_size * self.output_length) as u64
            } else {
                0
            },
            scratch_reallocation_count,
        })
    }

    pub(crate) fn evaluate_active_chunks_f64_into_scratch(
        &mut self,
        batch_size: usize,
        state: &[Complex<f64>],
        active_chunk_indices: &[usize],
    ) -> RusticolResult<()> {
        if let Some(order) = self.evaluator_output_order.as_deref() {
            self.evaluator.evaluate_selected_chunks_f64_into_output(
                batch_size,
                state,
                self.input_components.as_deref(),
                &self.input_spans,
                &mut self.evaluator_output_scratch_f64,
                active_chunk_indices,
            )?;
            remap_amplitude_outputs(
                batch_size,
                order,
                &self.evaluator_output_scratch_f64,
                &mut self.output_scratch_f64,
            )
        } else {
            self.evaluator.evaluate_selected_chunks_f64_into_output(
                batch_size,
                state,
                self.input_components.as_deref(),
                &self.input_spans,
                &mut self.output_scratch_f64,
                active_chunk_indices,
            )
        }
    }

    pub(crate) fn evaluate_active_chunks_f64_into_scratch_profile(
        &mut self,
        batch_size: usize,
        state: &[Complex<f64>],
        active_chunk_indices: &[usize],
    ) -> RusticolResult<AmplitudeEvaluationProfile> {
        let evaluator;
        let mut output_remap_elapsed = Duration::ZERO;
        if let Some(order) = self.evaluator_output_order.as_deref() {
            evaluator = self
                .evaluator
                .evaluate_selected_chunks_f64_into_output_profile(
                    batch_size,
                    state,
                    self.input_components.as_deref(),
                    &self.input_spans,
                    &mut self.evaluator_output_scratch_f64,
                    active_chunk_indices,
                )?;
            let remap_start = Instant::now();
            remap_amplitude_outputs(
                batch_size,
                order,
                &self.evaluator_output_scratch_f64,
                &mut self.output_scratch_f64,
            )?;
            output_remap_elapsed = remap_start.elapsed();
        } else {
            evaluator = self
                .evaluator
                .evaluate_selected_chunks_f64_into_output_profile(
                    batch_size,
                    state,
                    self.input_components.as_deref(),
                    &self.input_spans,
                    &mut self.output_scratch_f64,
                    active_chunk_indices,
                )?;
        }
        Ok(AmplitudeEvaluationProfile {
            input_pack_s: evaluator.leaf_input_pack_s,
            output_remap_s: profile_duration_seconds(output_remap_elapsed),
            // Active chunks compose the amplitude input and leaf maps, so
            // the evaluator leaf gather is the only actual input copy.
            input_copy_component_count: 0,
            output_remap_component_count: if self.evaluator_output_order.is_some() {
                (batch_size * self.output_length) as u64
            } else {
                0
            },
            scratch_reallocation_count: 0,
            evaluator,
        })
    }

    pub(crate) fn reduce_scratch_f64_into_selected(
        &mut self,
        batch_size: usize,
        raw_sums: &mut Vec<f64>,
        selected_color_sector_ids: Option<&BTreeSet<i64>>,
    ) -> RusticolResult<()> {
        raw_sums.clear();
        raw_sums.resize(batch_size, 0.0);
        self.reduce_scratch_f64_into_selected_slice(batch_size, raw_sums, selected_color_sector_ids)
    }

    pub(crate) fn reduce_scratch_f64_into_selected_slice(
        &mut self,
        batch_size: usize,
        raw_sums: &mut [f64],
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
        if raw_sums.len() != batch_size {
            return Err(RusticolError::invalid_argument(format!(
                "generic reduction output has length {}, expected {batch_size}",
                raw_sums.len()
            )));
        }
        raw_sums.fill(0.0);
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
            if let Some(repeated_block) = contraction.repeated_block.as_ref() {
                contraction
                    .group_scratch_f64
                    .resize(contraction.group_count, c64(0.0, 0.0));
                for (row, raw_sum) in raw_sums.iter_mut().enumerate() {
                    let row_offset = row * self.output_length;
                    if let Some(output_indices) = repeated_block.singleton_output_indices.as_deref()
                    {
                        for (target, output_index) in
                            contraction.group_scratch_f64.iter_mut().zip(output_indices)
                        {
                            *target = amplitudes[row_offset + *output_index];
                        }
                    } else {
                        for (target, group_index) in contraction
                            .group_scratch_f64
                            .iter_mut()
                            .zip(&repeated_block.component_group_indices)
                        {
                            let mut sum = c64(0.0, 0.0);
                            for output_index in &self.raw_sum_groups[*group_index].indices {
                                sum += amplitudes[row_offset + *output_index];
                            }
                            *target = sum;
                        }
                    }

                    let component_count = repeated_block.component_count;
                    if repeated_block.all_weights_real {
                        for entry in &repeated_block.entries {
                            let left_start = entry.left_group_index * component_count;
                            let right_start = entry.right_group_index * component_count;
                            let product_re = sum_real_hermitian_products(
                                &contraction.group_scratch_f64
                                    [left_start..left_start + component_count],
                                &contraction.group_scratch_f64
                                    [right_start..right_start + component_count],
                            );
                            *raw_sum += entry.symmetry_factor * (entry.weight_re * product_re);
                        }
                    } else {
                        for entry in &repeated_block.entries {
                            let left_start = entry.left_group_index * component_count;
                            let right_start = entry.right_group_index * component_count;
                            let product = sum_hermitian_products(
                                &contraction.group_scratch_f64
                                    [left_start..left_start + component_count],
                                &contraction.group_scratch_f64
                                    [right_start..right_start + component_count],
                            );
                            *raw_sum += entry.symmetry_factor
                                * (entry.weight_re * product.re - entry.weight_im * product.im);
                        }
                    }
                }
                return Ok(());
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
        let helicity_indices = physics.selected_helicity_indices(selected_helicity_ids)?;
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        let mut selected_helicity_positions = vec![None; helicity_count];
        for (position, index) in helicity_indices.iter().copied().enumerate() {
            selected_helicity_positions[index] = Some(position);
        }
        let mut selected_color_positions = vec![None; color_count];
        for (position, index) in color_indices.iter().copied().enumerate() {
            selected_color_positions[index] = Some(position);
        }
        let selected_component_count = helicity_indices
            .len()
            .checked_mul(color_indices.len())
            .ok_or_else(|| RusticolError::invalid_argument("resolved shape overflows usize"))?;
        let mut values = vec![0.0; batch_size * selected_component_count];

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
            let selected_contracted_color_position = selected_color_positions[0];
            let mut selected_helicity_weights_by_entry =
                Vec::with_capacity(contraction.entries.len());
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
                if left_reduction.physical_helicity_ids != right_reduction.physical_helicity_ids {
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
                selected_helicity_weights_by_entry.push(
                    left_weights
                        .into_iter()
                        .filter_map(|(helicity_index, weight)| {
                            selected_helicity_positions[helicity_index]
                                .map(|position| (position, weight))
                        })
                        .collect::<Vec<_>>(),
                );
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
                for (entry, left_weights) in contraction
                    .entries
                    .iter()
                    .zip(&selected_helicity_weights_by_entry)
                {
                    let left = contraction.group_scratch_f64[group_row + entry.left_group_index];
                    let right = contraction.group_scratch_f64[group_row + entry.right_group_index];
                    let product = left * right.conj();
                    let contribution = normalization_factor
                        * entry.symmetry_factor
                        * (entry.weight_re * product.re - entry.weight_im * product.im);
                    let Some(color_position) = selected_contracted_color_position else {
                        continue;
                    };
                    for (helicity_position, weight) in left_weights {
                        values[row * selected_component_count
                            + *helicity_position * color_indices.len()
                            + color_position] += contribution * weight;
                    }
                }
            }
        } else {
            if !self.has_coherent_groups {
                return Err(RusticolError::invalid_argument(
                    "resolved evaluation requires coherent amplitude-group metadata",
                ));
            }
            let mut selected_member_weights_by_group =
                Vec::with_capacity(self.raw_sum_groups.len());
            for group in &self.raw_sum_groups {
                let reduction = physics
                    .reduction_by_group_id
                    .get(&group.id)
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(format!(
                            "resolved metadata is missing coherent group {}",
                            group.id
                        ))
                    })?;
                selected_member_weights_by_group.push(
                    physics
                        .normalized_member_weights(reduction)?
                        .into_iter()
                        .filter_map(|(helicity_index, color_index, weight)| {
                            Some((
                                selected_helicity_positions[helicity_index]?,
                                selected_color_positions[color_index]?,
                                weight,
                            ))
                        })
                        .collect::<Vec<_>>(),
                );
            }
            for row in 0..batch_size {
                let row_offset = row * self.output_length;
                for (group, member_weights) in self
                    .raw_sum_groups
                    .iter()
                    .zip(&selected_member_weights_by_group)
                {
                    let mut sum = c64(0.0, 0.0);
                    for index in &group.indices {
                        sum += amplitudes[row_offset + *index];
                    }
                    let contribution = normalization_factor
                        * group.all_sector_weight
                        * (sum.re * sum.re + sum.im * sum.im);
                    for (helicity_position, color_position, weight) in member_weights {
                        values[row * selected_component_count
                            + *helicity_position * color_indices.len()
                            + *color_position] += contribution * weight;
                    }
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

    /// Reduce one topology-replay source mapping directly into public totals.
    ///
    /// The source and target component buffers are row-sized persistent
    /// scratch.  Replaying the routes before summing the target row preserves
    /// the resolved contract's component accumulation and H-major/C-minor
    /// summation order without materializing either batch-wide tensor.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduce_scratch_f64_routed_totals_into(
        &mut self,
        batch_size: usize,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        selected_helicity_ids: Option<&BTreeSet<String>>,
        selected_color_ids: Option<&BTreeSet<String>>,
        replay_entry: &LcResolvedReplayEntry,
        source_component_count: usize,
        target_component_count: usize,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let amplitudes = &self.output_scratch_f64;
        if amplitudes.len() != batch_size * self.output_length {
            return Err(RusticolError::invalid_argument(format!(
                "generic amplitude output buffer has length {}, expected {}",
                amplitudes.len(),
                batch_size * self.output_length
            )));
        }
        if output.len() != batch_size {
            return Err(RusticolError::invalid_argument(format!(
                "routed reduction output has length {}, expected {batch_size}",
                output.len()
            )));
        }
        if self.color_contraction.is_some() {
            return Err(RusticolError::invalid_argument(
                "LC topology replay does not support contracted color reduction",
            ));
        }
        if !self.has_coherent_groups {
            return Err(RusticolError::invalid_argument(
                "resolved evaluation requires coherent amplitude-group metadata",
            ));
        }

        let helicity_count = physics.manifest.helicities.len();
        let color_count = physics.manifest.color_components.len();
        let helicity_indices = physics.selected_helicity_indices(selected_helicity_ids)?;
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        let expected_source_component_count = helicity_indices
            .len()
            .checked_mul(color_indices.len())
            .ok_or_else(|| RusticolError::invalid_argument("resolved shape overflows usize"))?;
        if source_component_count != expected_source_component_count {
            return Err(RusticolError::integrity(format!(
                "topology replay source has {source_component_count} components, expected {expected_source_component_count}"
            )));
        }

        let mut selected_helicity_positions = vec![None; helicity_count];
        for (position, index) in helicity_indices.iter().copied().enumerate() {
            selected_helicity_positions[index] = Some(position);
        }
        let mut selected_color_positions = vec![None; color_count];
        for (position, index) in color_indices.iter().copied().enumerate() {
            selected_color_positions[index] = Some(position);
        }
        let mut selected_member_weights_by_group = Vec::with_capacity(self.raw_sum_groups.len());
        for group in &self.raw_sum_groups {
            let reduction = physics
                .reduction_by_group_id
                .get(&group.id)
                .ok_or_else(|| {
                    RusticolError::invalid_argument(format!(
                        "resolved metadata is missing coherent group {}",
                        group.id
                    ))
                })?;
            selected_member_weights_by_group.push(
                physics
                    .normalized_member_weights(reduction)?
                    .into_iter()
                    .filter_map(|(helicity_index, color_index, weight)| {
                        Some((
                            selected_helicity_positions[helicity_index]?,
                            selected_color_positions[color_index]?,
                            weight,
                        ))
                    })
                    .collect::<Vec<_>>(),
            );
        }

        self.resolved_source_row_scratch_f64
            .resize(source_component_count, 0.0);
        self.resolved_target_row_scratch_f64
            .resize(target_component_count, 0.0);
        for (row, target_total) in output.iter_mut().enumerate() {
            let source_row = &mut self.resolved_source_row_scratch_f64;
            source_row.fill(0.0);
            let row_offset = row * self.output_length;
            for (group, member_weights) in self
                .raw_sum_groups
                .iter()
                .zip(&selected_member_weights_by_group)
            {
                let mut sum = c64(0.0, 0.0);
                for index in &group.indices {
                    sum += amplitudes[row_offset + *index];
                }
                let contribution = normalization_factor
                    * group.all_sector_weight
                    * (sum.re * sum.re + sum.im * sum.im);
                for (helicity_position, color_position, weight) in member_weights {
                    source_row[*helicity_position * color_indices.len() + *color_position] +=
                        contribution * weight;
                }
            }

            let target_row = &mut self.resolved_target_row_scratch_f64;
            target_row.fill(0.0);
            for route in &replay_entry.routes {
                if route.source_index >= source_component_count
                    || route.target_index >= target_component_count
                {
                    return Err(RusticolError::integrity(
                        "LC topology replay selected route is out of bounds",
                    ));
                }
                target_row[route.target_index] += route.weight * source_row[route.source_index];
            }
            *target_total = target_row.iter().sum();
        }
        Ok(())
    }

    /// Reduce amplitudes evaluated for one physical runtime-helicity domain.
    ///
    /// The ordinary resolved reducer distributes one representative amplitude
    /// over every proof-equivalent physical helicity.  A materialized
    /// recurrence artifact instead reevaluates that representative with the
    /// requested source states, so the selected helicity receives the full
    /// contribution.  Colour symmetry weights remain distributed over their
    /// physical components exactly as in the ordinary resolved contract.
    pub(crate) fn reduce_scratch_f64_for_materialized_helicity(
        &mut self,
        batch_size: usize,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        helicity_index: usize,
        root_factors: &[Option<Complex<f64>>],
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
        if root_factors.len() != self.output_length {
            return Err(RusticolError::integrity(format!(
                "helicity recurrence route table has {} roots, expected {}",
                root_factors.len(),
                self.output_length
            )));
        }
        let helicity = physics
            .manifest
            .helicities
            .get(helicity_index)
            .ok_or_else(|| {
                RusticolError::selector(format!(
                    "runtime helicity index {helicity_index} is out of range"
                ))
            })?;
        let color_count = physics.manifest.color_components.len();
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        let mut full_values = vec![0.0; batch_size * color_count];

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
            let mut group_active = vec![false; contraction.group_count];
            for (group_index, group) in self.raw_sum_groups.iter().enumerate() {
                let reduction = physics
                    .reduction_by_group_id
                    .get(&group.id)
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(format!(
                            "resolved metadata is missing coherent group {}",
                            group.id
                        ))
                    })?;
                group_active[group_index] = reduction
                    .physical_helicity_ids
                    .iter()
                    .any(|id| id == &helicity.id)
                    && group
                        .indices
                        .iter()
                        .any(|index| root_factors[*index].is_some());
            }
            for (row, full_value) in full_values.iter_mut().enumerate().take(batch_size) {
                let row_offset = row * self.output_length;
                let group_row = row * contraction.group_count;
                for (group_index, group) in self.raw_sum_groups.iter().enumerate() {
                    let mut sum = c64(0.0, 0.0);
                    if group_active[group_index] {
                        for index in &group.indices {
                            if let Some(factor) = root_factors[*index] {
                                sum += amplitudes[row_offset + *index] * factor;
                            }
                        }
                    }
                    contraction.group_scratch_f64[group_row + group_index] = sum;
                }
                for entry in &contraction.entries {
                    if !group_active[entry.left_group_index]
                        || !group_active[entry.right_group_index]
                    {
                        continue;
                    }
                    let left = contraction.group_scratch_f64[group_row + entry.left_group_index];
                    let right = contraction.group_scratch_f64[group_row + entry.right_group_index];
                    let product = left * right.conj();
                    *full_value += normalization_factor
                        * entry.symmetry_factor
                        * (entry.weight_re * product.re - entry.weight_im * product.im);
                }
            }
        } else {
            if !self.has_coherent_groups {
                return Err(RusticolError::invalid_argument(
                    "materialized helicity reduction requires coherent amplitude-group metadata",
                ));
            }
            let mut active_groups = Vec::new();
            for (group_index, group) in self.raw_sum_groups.iter().enumerate() {
                let reduction = physics
                    .reduction_by_group_id
                    .get(&group.id)
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(format!(
                            "resolved metadata is missing coherent group {}",
                            group.id
                        ))
                    })?;
                if !reduction
                    .physical_helicity_ids
                    .iter()
                    .any(|id| id == &helicity.id)
                    || !group
                        .indices
                        .iter()
                        .any(|index| root_factors[*index].is_some())
                {
                    continue;
                }
                let mut color_weights = reduction
                    .physical_color_ids
                    .iter()
                    .map(|id| {
                        let index = *physics.color_index_by_id.get(id).ok_or_else(|| {
                            RusticolError::artifact(format!(
                                "resolved reduction group {} references unknown color {id:?}",
                                group.id
                            ))
                        })?;
                        Ok((
                            index,
                            physics.manifest.color_components[index].coefficient(),
                        ))
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                let total_color_weight =
                    color_weights.iter().map(|(_, weight)| *weight).sum::<f64>();
                if !total_color_weight.is_finite() || total_color_weight <= 0.0 {
                    return Err(RusticolError::artifact(format!(
                        "resolved reduction group {} has no positive color weight",
                        group.id
                    )));
                }
                for (_, weight) in &mut color_weights {
                    *weight /= total_color_weight;
                }
                active_groups.push((group_index, color_weights));
            }
            for row in 0..batch_size {
                let row_offset = row * self.output_length;
                let color_row = row * color_count;
                for (group_index, color_weights) in &active_groups {
                    let group = &self.raw_sum_groups[*group_index];
                    let mut sum = c64(0.0, 0.0);
                    for index in &group.indices {
                        if let Some(factor) = root_factors[*index] {
                            sum += amplitudes[row_offset + *index] * factor;
                        }
                    }
                    let contribution = normalization_factor
                        * group.all_sector_weight
                        * (sum.re * sum.re + sum.im * sum.im);
                    for (color_index, weight) in color_weights {
                        full_values[color_row + *color_index] += contribution * *weight;
                    }
                }
            }
        }

        let mut values = Vec::with_capacity(batch_size * color_indices.len());
        for row in 0..batch_size {
            for color_index in &color_indices {
                values.push(full_values[row * color_count + *color_index]);
            }
        }
        Ok(ResolvedValues {
            values,
            point_count: batch_size,
            helicity_indices: vec![helicity_index],
            color_indices,
        })
    }

    /// Add one materialized physical-helicity row to caller-owned totals.
    ///
    /// Colour cells are accumulated in the same group order as the resolved
    /// reducer, then folded into each point total in manifest colour order.
    /// Keeping only one colour row preserves exact summation semantics while
    /// avoiding a batch-by-colour resolved allocation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn reduce_scratch_f64_for_materialized_helicity_add_into(
        &mut self,
        batch_size: usize,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        helicity_index: usize,
        root_factors: &[Option<Complex<f64>>],
        selected_color_ids: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        let amplitudes = &self.output_scratch_f64;
        if amplitudes.len() != batch_size * self.output_length {
            return Err(RusticolError::invalid_argument(format!(
                "generic amplitude output buffer has length {}, expected {}",
                amplitudes.len(),
                batch_size * self.output_length
            )));
        }
        if output.len() != batch_size {
            return Err(RusticolError::invalid_argument(format!(
                "materialized helicity output has length {}, expected {batch_size}",
                output.len()
            )));
        }
        if root_factors.len() != self.output_length {
            return Err(RusticolError::integrity(format!(
                "helicity recurrence route table has {} roots, expected {}",
                root_factors.len(),
                self.output_length
            )));
        }
        if self.color_contraction.is_some() {
            return Err(RusticolError::invalid_argument(
                "direct materialized-helicity totals require diagonal LC color",
            ));
        }
        if !self.has_coherent_groups {
            return Err(RusticolError::invalid_argument(
                "materialized helicity reduction requires coherent amplitude-group metadata",
            ));
        }
        let helicity = physics
            .manifest
            .helicities
            .get(helicity_index)
            .ok_or_else(|| {
                RusticolError::selector(format!(
                    "runtime helicity index {helicity_index} is out of range"
                ))
            })?;
        let color_count = physics.manifest.color_components.len();
        let color_indices = physics.selected_color_indices(selected_color_ids)?;

        let mut active_groups = Vec::new();
        for (group_index, group) in self.raw_sum_groups.iter().enumerate() {
            let reduction = physics
                .reduction_by_group_id
                .get(&group.id)
                .ok_or_else(|| {
                    RusticolError::invalid_argument(format!(
                        "resolved metadata is missing coherent group {}",
                        group.id
                    ))
                })?;
            if !reduction
                .physical_helicity_ids
                .iter()
                .any(|id| id == &helicity.id)
                || !group
                    .indices
                    .iter()
                    .any(|index| root_factors[*index].is_some())
            {
                continue;
            }
            let mut color_weights = reduction
                .physical_color_ids
                .iter()
                .map(|id| {
                    let index = *physics.color_index_by_id.get(id).ok_or_else(|| {
                        RusticolError::artifact(format!(
                            "resolved reduction group {} references unknown color {id:?}",
                            group.id
                        ))
                    })?;
                    Ok((
                        index,
                        physics.manifest.color_components[index].coefficient(),
                    ))
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            let total_color_weight = color_weights.iter().map(|(_, weight)| *weight).sum::<f64>();
            if !total_color_weight.is_finite() || total_color_weight <= 0.0 {
                return Err(RusticolError::artifact(format!(
                    "resolved reduction group {} has no positive color weight",
                    group.id
                )));
            }
            for (_, weight) in &mut color_weights {
                *weight /= total_color_weight;
            }
            active_groups.push((group_index, color_weights));
        }

        self.resolved_source_row_scratch_f64
            .resize(color_count, 0.0);
        for (row, target_total) in output.iter_mut().enumerate() {
            let color_row = &mut self.resolved_source_row_scratch_f64;
            color_row.fill(0.0);
            let row_offset = row * self.output_length;
            for (group_index, color_weights) in &active_groups {
                let group = &self.raw_sum_groups[*group_index];
                let mut sum = c64(0.0, 0.0);
                for index in &group.indices {
                    if let Some(factor) = root_factors[*index] {
                        sum += amplitudes[row_offset + *index] * factor;
                    }
                }
                let contribution = normalization_factor
                    * group.all_sector_weight
                    * (sum.re * sum.re + sum.im * sum.im);
                for (color_index, weight) in color_weights {
                    color_row[*color_index] += contribution * *weight;
                }
            }
            for color_index in &color_indices {
                *target_total += color_row[*color_index];
            }
        }
        Ok(())
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn evaluate_materialized_helicity_generic<T>(
        &mut self,
        batch_size: usize,
        state: &[Complex<T>],
        binary_precision: Option<u32>,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        helicity_index: usize,
        root_factors: &[Option<Complex<f64>>],
        selected_color_ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<T>, f64, f64, f64)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let (amplitudes, input_pack_s, evaluator_call_s) =
            self.evaluate_outputs_generic(batch_size, state, binary_precision)?;
        let reduction_start = Instant::now();
        if root_factors.len() != self.output_length {
            return Err(RusticolError::integrity(format!(
                "helicity recurrence route table has {} roots, expected {}",
                root_factors.len(),
                self.output_length
            )));
        }
        let helicity = physics
            .manifest
            .helicities
            .get(helicity_index)
            .ok_or_else(|| {
                RusticolError::selector(format!(
                    "runtime helicity index {helicity_index} is out of range"
                ))
            })?;
        let color_count = physics.manifest.color_components.len();
        let color_indices = physics.selected_color_indices(selected_color_ids)?;
        let mut full_values = vec![T::new_zero(); batch_size * color_count];

        if let Some(contraction) = self.color_contraction.as_ref() {
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
            let mut group_active = vec![false; contraction.group_count];
            for (group_index, group) in self.raw_sum_groups.iter().enumerate() {
                let reduction = physics
                    .reduction_by_group_id
                    .get(&group.id)
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(format!(
                            "resolved metadata is missing coherent group {}",
                            group.id
                        ))
                    })?;
                group_active[group_index] = reduction
                    .physical_helicity_ids
                    .iter()
                    .any(|id| id == &helicity.id)
                    && group
                        .indices
                        .iter()
                        .any(|index| root_factors[*index].is_some());
            }
            let mut group_values = vec![complex_zero::<T>(); batch_size * contraction.group_count];
            for row in 0..batch_size {
                let row_offset = row * self.output_length;
                let group_row = row * contraction.group_count;
                for (group_index, group) in self.raw_sum_groups.iter().enumerate() {
                    if !group_active[group_index] {
                        continue;
                    }
                    let mut sum = complex_zero::<T>();
                    for index in &group.indices {
                        let Some(factor) = root_factors[*index] else {
                            continue;
                        };
                        sum += amplitudes[row_offset + *index].clone()
                            * c_generic(T::from(factor.re), T::from(factor.im));
                    }
                    group_values[group_row + group_index] = sum;
                }
                for entry in &contraction.entries {
                    if !group_active[entry.left_group_index]
                        || !group_active[entry.right_group_index]
                    {
                        continue;
                    }
                    let left = &group_values[group_row + entry.left_group_index];
                    let right = &group_values[group_row + entry.right_group_index];
                    let product_re =
                        left.re.clone() * right.re.clone() + left.im.clone() * right.im.clone();
                    let product_im =
                        left.im.clone() * right.re.clone() - left.re.clone() * right.im.clone();
                    full_values[row] += T::from(normalization_factor * entry.symmetry_factor)
                        * (T::from(entry.weight_re) * product_re
                            - T::from(entry.weight_im) * product_im);
                }
            }
        } else {
            if !self.has_coherent_groups {
                return Err(RusticolError::invalid_argument(
                    "materialized helicity reduction requires coherent amplitude-group metadata",
                ));
            }
            for row in 0..batch_size {
                let row_offset = row * self.output_length;
                let color_row = row * color_count;
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
                    if !reduction
                        .physical_helicity_ids
                        .iter()
                        .any(|id| id == &helicity.id)
                    {
                        continue;
                    }
                    let mut sum = complex_zero::<T>();
                    for index in &group.indices {
                        let Some(factor) = root_factors[*index] else {
                            continue;
                        };
                        sum += amplitudes[row_offset + *index].clone()
                            * c_generic(T::from(factor.re), T::from(factor.im));
                    }
                    let color_weights = reduction
                        .physical_color_ids
                        .iter()
                        .map(|id| {
                            let index = *physics.color_index_by_id.get(id).ok_or_else(|| {
                                RusticolError::artifact(format!(
                                    "resolved reduction group {} references unknown color {id:?}",
                                    group.id
                                ))
                            })?;
                            Ok((
                                index,
                                physics.manifest.color_components[index].coefficient(),
                            ))
                        })
                        .collect::<RusticolResult<Vec<_>>>()?;
                    let total_color_weight =
                        color_weights.iter().map(|(_, weight)| *weight).sum::<f64>();
                    if !total_color_weight.is_finite() || total_color_weight <= 0.0 {
                        return Err(RusticolError::artifact(format!(
                            "resolved reduction group {} has no positive color weight",
                            group.id
                        )));
                    }
                    let contribution = T::from(normalization_factor * group.all_sector_weight)
                        * (sum.re.clone() * sum.re + sum.im.clone() * sum.im);
                    for (color_index, weight) in color_weights {
                        full_values[color_row + color_index] +=
                            contribution.clone() * T::from(weight / total_color_weight);
                    }
                }
            }
        }

        let mut values = Vec::with_capacity(batch_size * color_indices.len());
        for row in 0..batch_size {
            for color_index in &color_indices {
                values.push(full_values[row * color_count + *color_index].clone());
            }
        }
        Ok((
            ResolvedValues {
                values,
                point_count: batch_size,
                helicity_indices: vec![helicity_index],
                color_indices,
            },
            input_pack_s,
            evaluator_call_s,
            reduction_start.elapsed().as_secs_f64(),
        ))
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
        let mut input_pack_elapsed = Duration::ZERO;
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
            input_pack_elapsed = pack_start.elapsed();
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
        Ok((
            evaluated,
            profile_duration_seconds(input_pack_elapsed),
            evaluator_call_s,
        ))
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
    Ok(Some(ColorContractionRuntime::new(groups, entries)))
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
