// SPDX-License-Identifier: 0BSD

use super::super::*;
use super::*;

impl EvaluatorGroup {
    pub(crate) fn load(manifest: &EvaluatorManifest, root: &Path) -> RusticolResult<Self> {
        let payloads = EvaluatorPayloadStore::directory(root);
        Self::load_from_store(manifest, &payloads)
    }

    pub(crate) fn load_from_store(
        manifest: &EvaluatorManifest,
        payloads: &EvaluatorPayloadStore,
    ) -> RusticolResult<Self> {
        ensure_evaluator_capabilities_supported(manifest)?;
        let (input_len, _) = manifest.io_len()?;
        let mut evaluators = Vec::new();
        let mut input_mappings = Vec::new();
        flatten_evaluators_with_mappings(
            manifest,
            payloads,
            None,
            input_len,
            &mut evaluators,
            &mut input_mappings,
        )?;
        let input_mapping_spans = input_mappings
            .iter()
            .map(|mapping| {
                mapping
                    .as_deref()
                    .map(contiguous_input_spans)
                    .unwrap_or_default()
            })
            .collect();
        let output_len = evaluators.iter().map(|e| e.output_len).sum();
        Ok(Self {
            evaluators,
            input_len,
            input_mappings,
            input_mapping_spans,
            output_len,
            chunk_parameter_scratch_f64: Vec::new(),
            chunk_scratch_f64: Vec::new(),
        })
    }

    pub(crate) fn uses_simd_jit(&self) -> bool {
        self.evaluators
            .iter()
            .any(|evaluator| match &evaluator.eval {
                #[cfg(feature = "f64-symjit")]
                F64Evaluator::SymjitApplication(_) => true,
                #[cfg(feature = "symbolica-runtime")]
                F64Evaluator::Jit(_) => true,
                #[cfg(feature = "f64-compiled")]
                F64Evaluator::Compiled(_) => false,
            })
    }

    pub(crate) fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
    ) -> RusticolResult<Vec<Complex<f64>>> {
        let mut out = Vec::new();
        self.evaluate_batch_into(batch_size, params, &mut out)?;
        Ok(out)
    }

    pub(crate) fn evaluate_single_row(
        &mut self,
        params: &[Complex<f64>],
    ) -> RusticolResult<Vec<Complex<f64>>> {
        self.evaluate_batch(1, params)
    }

    pub(crate) fn output_chunk_lengths(&self) -> Vec<usize> {
        self.evaluators
            .iter()
            .map(|evaluator| evaluator.output_len)
            .collect()
    }

    pub(crate) fn is_chunked(&self) -> bool {
        self.evaluators.len() > 1
    }

    pub(crate) fn evaluate_chunks_f64_into_state(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        state_parameter_count: usize,
        state: &mut [Complex<f64>],
        chunk_outputs: &[Vec<(usize, usize)>],
        chunk_output_spans: &[Vec<(usize, usize, usize)>],
    ) -> RusticolResult<(f64, f64)> {
        if chunk_outputs.len() != self.evaluators.len()
            || chunk_output_spans.len() != self.evaluators.len()
        {
            return Err(RusticolError::invalid_argument(
                "stage chunk-output layout does not match evaluator chunks",
            ));
        }
        let expected_state_len = batch_size
            .checked_mul(state_parameter_count)
            .ok_or_else(|| RusticolError::invalid_argument("stage state length overflows usize"))?;
        if state.len() != expected_state_len {
            return Err(RusticolError::invalid_argument(format!(
                "stage state has length {}, expected {expected_state_len}",
                state.len()
            )));
        }

        // Keep timing state out of floating-point registers while generated
        // evaluators execute. Some native evaluator ABIs use the full SIMD
        // register file, whereas Duration remains integer-backed.
        let mut evaluator_elapsed = Duration::ZERO;
        let mut assign_elapsed = Duration::ZERO;
        if params.len() != batch_size * self.input_len {
            return Err(RusticolError::invalid_argument(format!(
                "parameter buffer has length {}, expected {}",
                params.len(),
                batch_size * self.input_len
            )));
        }
        for ((((evaluator, input_mapping), input_spans), outputs), spans) in self
            .evaluators
            .iter_mut()
            .zip(&self.input_mappings)
            .zip(&self.input_mapping_spans)
            .zip(chunk_outputs)
            .zip(chunk_output_spans)
        {
            let evaluator_params = mapped_f64_parameters(
                params,
                batch_size,
                self.input_len,
                input_mapping.as_deref(),
                input_spans,
                &mut self.chunk_parameter_scratch_f64,
            );
            validate_leaf_parameter_length(evaluator, batch_size, evaluator_params)?;
            self.chunk_scratch_f64
                .resize(batch_size * evaluator.output_len, c64(0.0, 0.0));
            let eval_start = Instant::now();
            evaluator.evaluate_f64_batch_unpadded(
                batch_size,
                evaluator_params,
                &mut self.chunk_scratch_f64,
            )?;
            evaluator_elapsed += eval_start.elapsed();

            let assign_start = Instant::now();
            for row in 0..batch_size {
                let row_state = row * state_parameter_count;
                let row_eval = row * evaluator.output_len;
                if spans.is_empty() {
                    for (column, state_offset) in outputs {
                        state[row_state + *state_offset] =
                            self.chunk_scratch_f64[row_eval + *column];
                    }
                } else {
                    for (column_start, state_offset_start, len) in spans {
                        let source_start = row_eval + *column_start;
                        let target_start = row_state + *state_offset_start;
                        state[target_start..target_start + *len].copy_from_slice(
                            &self.chunk_scratch_f64[source_start..source_start + *len],
                        );
                    }
                }
            }
            assign_elapsed += assign_start.elapsed();
        }
        Ok((
            evaluator_elapsed.as_secs_f64(),
            assign_elapsed.as_secs_f64(),
        ))
    }

    // The explicit slices avoid constructing a descriptor in this selector hot path.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_selected_chunks_f64_into_state(
        &mut self,
        batch_size: usize,
        state_parameter_count: usize,
        state: &mut [Complex<f64>],
        parent_input_components: Option<&[usize]>,
        parent_input_spans: &[(usize, usize, usize)],
        chunk_outputs: &[Vec<(usize, usize)>],
        chunk_output_spans: &[Vec<(usize, usize, usize)>],
        active_chunk_indices: &[usize],
    ) -> RusticolResult<(f64, f64, f64)> {
        if chunk_outputs.len() != self.evaluators.len()
            || chunk_output_spans.len() != self.evaluators.len()
        {
            return Err(RusticolError::invalid_argument(
                "stage chunk-output layout does not match evaluator chunks",
            ));
        }
        validate_active_chunk_indices(active_chunk_indices, self.evaluators.len())?;
        let expected_state_len = batch_size
            .checked_mul(state_parameter_count)
            .ok_or_else(|| RusticolError::invalid_argument("stage state length overflows usize"))?;
        if state.len() != expected_state_len {
            return Err(RusticolError::invalid_argument(format!(
                "stage state has length {}, expected {expected_state_len}",
                state.len()
            )));
        }
        match parent_input_components {
            Some(components) => {
                if components.len() != self.input_len {
                    return Err(RusticolError::invalid_argument(format!(
                        "stage input-component map has length {}, expected {}",
                        components.len(),
                        self.input_len
                    )));
                }
                if components
                    .iter()
                    .any(|component| *component >= state_parameter_count)
                {
                    return Err(RusticolError::invalid_argument(
                        "stage input-component map references state outside the parameter layout",
                    ));
                }
            }
            None if self.input_len != state_parameter_count => {
                return Err(RusticolError::invalid_argument(format!(
                    "evaluator input length {} does not match stage state parameter count {state_parameter_count}",
                    self.input_len
                )));
            }
            None => {}
        }

        let mut input_pack_elapsed = Duration::ZERO;
        let mut evaluator_elapsed = Duration::ZERO;
        let mut assign_elapsed = Duration::ZERO;
        for &chunk_index in active_chunk_indices {
            let input_pack_start = Instant::now();
            let evaluator_params = selected_chunk_f64_parameters(
                state,
                batch_size,
                state_parameter_count,
                self.input_len,
                parent_input_components,
                parent_input_spans,
                self.input_mappings[chunk_index].as_deref(),
                &self.input_mapping_spans[chunk_index],
                &mut self.chunk_parameter_scratch_f64,
            )?;
            input_pack_elapsed += input_pack_start.elapsed();

            let evaluator = &mut self.evaluators[chunk_index];
            validate_leaf_parameter_length(evaluator, batch_size, evaluator_params)?;
            let scratch_len = batch_size
                .checked_mul(evaluator.output_len)
                .ok_or_else(|| {
                    RusticolError::invalid_argument("stage chunk output length overflows usize")
                })?;
            self.chunk_scratch_f64.resize(scratch_len, c64(0.0, 0.0));
            let eval_start = Instant::now();
            evaluator.evaluate_f64_batch_unpadded(
                batch_size,
                evaluator_params,
                &mut self.chunk_scratch_f64,
            )?;
            evaluator_elapsed += eval_start.elapsed();

            let assign_start = Instant::now();
            let outputs = &chunk_outputs[chunk_index];
            let spans = &chunk_output_spans[chunk_index];
            for row in 0..batch_size {
                let row_state = row * state_parameter_count;
                let row_eval = row * evaluator.output_len;
                if spans.is_empty() {
                    for (column, state_offset) in outputs {
                        state[row_state + *state_offset] =
                            self.chunk_scratch_f64[row_eval + *column];
                    }
                } else {
                    for (column_start, state_offset_start, len) in spans {
                        let source_start = row_eval + *column_start;
                        let target_start = row_state + *state_offset_start;
                        state[target_start..target_start + *len].copy_from_slice(
                            &self.chunk_scratch_f64[source_start..source_start + *len],
                        );
                    }
                }
            }
            assign_elapsed += assign_start.elapsed();
        }
        Ok((
            input_pack_elapsed.as_secs_f64(),
            evaluator_elapsed.as_secs_f64(),
            assign_elapsed.as_secs_f64(),
        ))
    }

    pub(crate) fn evaluate_selected_chunks_f64_into_output(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        parent_input_components: Option<&[usize]>,
        parent_input_spans: &[(usize, usize, usize)],
        out: &mut Vec<Complex<f64>>,
        active_chunk_indices: &[usize],
    ) -> RusticolResult<(f64, f64)> {
        validate_active_chunk_indices(active_chunk_indices, self.evaluators.len())?;
        if batch_size == 0 || params.len() % batch_size != 0 {
            return Err(RusticolError::invalid_argument(
                "amplitude evaluator parameter buffer has an inconsistent batch size",
            ));
        }
        let state_parameter_count = params.len() / batch_size;
        match parent_input_components {
            Some(components) => {
                if components.len() != self.input_len
                    || components
                        .iter()
                        .any(|component| *component >= state_parameter_count)
                {
                    return Err(RusticolError::invalid_argument(
                        "amplitude input-component map is inconsistent with runtime state",
                    ));
                }
            }
            None if self.input_len != state_parameter_count => {
                return Err(RusticolError::invalid_argument(format!(
                    "evaluator input length {} does not match amplitude state parameter count {state_parameter_count}",
                    self.input_len
                )));
            }
            None => {}
        }
        let output_len = batch_size
            .checked_mul(self.output_len)
            .ok_or_else(|| RusticolError::invalid_argument("amplitude output length overflows"))?;
        out.resize(output_len, c64(0.0, 0.0));
        out.fill(c64(0.0, 0.0));

        let mut chunk_output_starts = Vec::with_capacity(self.evaluators.len());
        let mut output_start = 0usize;
        for evaluator in &self.evaluators {
            chunk_output_starts.push(output_start);
            output_start = output_start
                .checked_add(evaluator.output_len)
                .ok_or_else(|| {
                    RusticolError::invalid_argument("amplitude chunk output range overflows")
                })?;
        }
        if output_start != self.output_len {
            return Err(RusticolError::integrity(
                "amplitude evaluator chunk outputs do not cover the declared output length",
            ));
        }

        let mut input_pack_elapsed = Duration::ZERO;
        let mut evaluator_elapsed = Duration::ZERO;
        for &chunk_index in active_chunk_indices {
            let input_pack_start = Instant::now();
            let evaluator_params = selected_chunk_f64_parameters(
                params,
                batch_size,
                state_parameter_count,
                self.input_len,
                parent_input_components,
                parent_input_spans,
                self.input_mappings[chunk_index].as_deref(),
                &self.input_mapping_spans[chunk_index],
                &mut self.chunk_parameter_scratch_f64,
            )?;
            input_pack_elapsed += input_pack_start.elapsed();

            let evaluator = &mut self.evaluators[chunk_index];
            validate_leaf_parameter_length(evaluator, batch_size, evaluator_params)?;
            self.chunk_scratch_f64.resize(
                batch_size
                    .checked_mul(evaluator.output_len)
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "amplitude chunk output length overflows usize",
                        )
                    })?,
                c64(0.0, 0.0),
            );
            let eval_start = Instant::now();
            evaluator.evaluate_f64_batch_unpadded(
                batch_size,
                evaluator_params,
                &mut self.chunk_scratch_f64,
            )?;
            let chunk_output_start = chunk_output_starts[chunk_index];
            for row in 0..batch_size {
                let source_start = row * evaluator.output_len;
                let target_start = row * self.output_len + chunk_output_start;
                out[target_start..target_start + evaluator.output_len].copy_from_slice(
                    &self.chunk_scratch_f64[source_start..source_start + evaluator.output_len],
                );
            }
            evaluator_elapsed += eval_start.elapsed();
        }
        Ok((
            input_pack_elapsed.as_secs_f64(),
            evaluator_elapsed.as_secs_f64(),
        ))
    }

    pub(crate) fn evaluate_batch_into(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut Vec<Complex<f64>>,
    ) -> RusticolResult<()> {
        self.evaluate_batch_into_with_tail_policy(batch_size, params, out, false)
    }

    pub(crate) fn evaluate_batch_into_padded_simd_tail(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut Vec<Complex<f64>>,
    ) -> RusticolResult<()> {
        self.evaluate_batch_into_with_tail_policy(batch_size, params, out, true)
    }

    fn evaluate_batch_into_with_tail_policy(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut Vec<Complex<f64>>,
        pad_incomplete_simd_tail: bool,
    ) -> RusticolResult<()> {
        let expected_output_len = batch_size * self.output_len;
        if out.len() != expected_output_len {
            out.resize(expected_output_len, c64(0.0, 0.0));
        }
        if params.len() != batch_size * self.input_len {
            return Err(RusticolError::invalid_argument(format!(
                "parameter buffer has length {}, expected {}",
                params.len(),
                batch_size * self.input_len
            )));
        }
        if self.evaluators.len() == 1 {
            let evaluator = &mut self.evaluators[0];
            let evaluator_params = mapped_f64_parameters(
                params,
                batch_size,
                self.input_len,
                self.input_mappings[0].as_deref(),
                &self.input_mapping_spans[0],
                &mut self.chunk_parameter_scratch_f64,
            );
            validate_leaf_parameter_length(evaluator, batch_size, evaluator_params)?;
            if pad_incomplete_simd_tail {
                evaluator.evaluate_f64_batch(batch_size, evaluator_params, out)?;
            } else {
                evaluator.evaluate_f64_batch_unpadded(batch_size, evaluator_params, out)?;
            }
            return Ok(());
        }
        let mut output_offset = 0;
        for ((evaluator, input_mapping), input_spans) in self
            .evaluators
            .iter_mut()
            .zip(&self.input_mappings)
            .zip(&self.input_mapping_spans)
        {
            let evaluator_params = mapped_f64_parameters(
                params,
                batch_size,
                self.input_len,
                input_mapping.as_deref(),
                input_spans,
                &mut self.chunk_parameter_scratch_f64,
            );
            validate_leaf_parameter_length(evaluator, batch_size, evaluator_params)?;
            self.chunk_scratch_f64
                .resize(batch_size * evaluator.output_len, c64(0.0, 0.0));
            if pad_incomplete_simd_tail {
                evaluator.evaluate_f64_batch(
                    batch_size,
                    evaluator_params,
                    &mut self.chunk_scratch_f64,
                )?;
            } else {
                evaluator.evaluate_f64_batch_unpadded(
                    batch_size,
                    evaluator_params,
                    &mut self.chunk_scratch_f64,
                )?;
            }
            for row in 0..batch_size {
                let src = row * evaluator.output_len;
                let dst = row * self.output_len + output_offset;
                out[dst..dst + evaluator.output_len]
                    .copy_from_slice(&self.chunk_scratch_f64[src..src + evaluator.output_len]);
            }
            output_offset += evaluator.output_len;
        }
        Ok(())
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn evaluate_batch_generic<T>(
        &mut self,
        batch_size: usize,
        params: &[Complex<T>],
        binary_precision: Option<u32>,
    ) -> RusticolResult<Vec<Complex<T>>>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let mut out = vec![complex_zero::<T>(); batch_size * self.output_len];
        if params.len() != batch_size * self.input_len {
            return Err(RusticolError::invalid_argument(format!(
                "parameter buffer has length {}, expected {}",
                params.len(),
                batch_size * self.input_len
            )));
        }
        let mut output_offset = 0;
        for (evaluator, input_mapping) in self.evaluators.iter_mut().zip(&self.input_mappings) {
            let mapped_params = input_mapping.as_ref().map(|indices| {
                let mut mapped = Vec::with_capacity(batch_size * indices.len());
                for row in 0..batch_size {
                    let row_start = row * self.input_len;
                    mapped.extend(
                        indices
                            .iter()
                            .map(|index| params[row_start + *index].clone()),
                    );
                }
                mapped
            });
            let evaluator_params = mapped_params.as_deref().unwrap_or(params);
            let mut chunk_out = vec![complex_zero::<T>(); batch_size * evaluator.output_len];
            for row in 0..batch_size {
                let in_start = row * evaluator.input_len;
                let out_start = row * evaluator.output_len;
                T::evaluate_loaded(
                    evaluator,
                    &evaluator_params[in_start..in_start + evaluator.input_len],
                    &mut chunk_out[out_start..out_start + evaluator.output_len],
                    binary_precision,
                )?;
            }
            for row in 0..batch_size {
                let src = row * evaluator.output_len;
                let dst = row * self.output_len + output_offset;
                out[dst..dst + evaluator.output_len]
                    .clone_from_slice(&chunk_out[src..src + evaluator.output_len]);
            }
            output_offset += evaluator.output_len;
        }
        Ok(out)
    }
}

fn mapped_f64_parameters<'a>(
    params: &'a [Complex<f64>],
    batch_size: usize,
    parent_input_len: usize,
    input_mapping: Option<&[usize]>,
    input_spans: &[(usize, usize, usize)],
    scratch: &'a mut Vec<Complex<f64>>,
) -> &'a [Complex<f64>] {
    let Some(indices) = input_mapping else {
        return params;
    };
    scratch.resize(batch_size * indices.len(), c64(0.0, 0.0));
    for row in 0..batch_size {
        let source_start = row * parent_input_len;
        let target_start = row * indices.len();
        if input_spans.is_empty() {
            for (local_index, parent_index) in indices.iter().enumerate() {
                scratch[target_start + local_index] = params[source_start + *parent_index];
            }
        } else {
            for (local_start, parent_start, len) in input_spans {
                let target = target_start + *local_start;
                let source = source_start + *parent_start;
                scratch[target..target + *len].copy_from_slice(&params[source..source + *len]);
            }
        }
    }
    scratch
}

#[allow(clippy::too_many_arguments, dead_code)]
fn selected_chunk_f64_parameters<'a>(
    state: &'a [Complex<f64>],
    batch_size: usize,
    state_parameter_count: usize,
    parent_input_len: usize,
    parent_input_components: Option<&[usize]>,
    parent_input_spans: &[(usize, usize, usize)],
    input_mapping: Option<&[usize]>,
    input_mapping_spans: &[(usize, usize, usize)],
    scratch: &'a mut Vec<Complex<f64>>,
) -> RusticolResult<&'a [Complex<f64>]> {
    let Some(parent_components) = parent_input_components else {
        return Ok(mapped_f64_parameters(
            state,
            batch_size,
            state_parameter_count,
            input_mapping,
            input_mapping_spans,
            scratch,
        ));
    };
    let mapped_input_len = input_mapping.map_or(parent_input_len, <[usize]>::len);
    let scratch_len = batch_size.checked_mul(mapped_input_len).ok_or_else(|| {
        RusticolError::invalid_argument("selected stage input length overflows usize")
    })?;
    scratch.resize(scratch_len, c64(0.0, 0.0));
    for row in 0..batch_size {
        let source_start = row * state_parameter_count;
        let target_start = row * mapped_input_len;
        if let Some(indices) = input_mapping {
            for (local_index, parent_index) in indices.iter().copied().enumerate() {
                scratch[target_start + local_index] =
                    state[source_start + parent_components[parent_index]];
            }
        } else if parent_input_spans.is_empty() {
            for (local_index, global_index) in parent_components.iter().copied().enumerate() {
                scratch[target_start + local_index] = state[source_start + global_index];
            }
        } else {
            for (local_start, global_start, len) in parent_input_spans {
                let target = target_start + *local_start;
                let source = source_start + *global_start;
                scratch[target..target + *len].copy_from_slice(&state[source..source + *len]);
            }
        }
    }
    Ok(scratch)
}

#[allow(dead_code)]
fn validate_active_chunk_indices(
    active_chunk_indices: &[usize],
    chunk_count: usize,
) -> RusticolResult<()> {
    let mut previous = None;
    for &chunk_index in active_chunk_indices {
        if chunk_index >= chunk_count {
            return Err(RusticolError::invalid_argument(format!(
                "active evaluator chunk index {chunk_index} is outside chunk count {chunk_count}",
            )));
        }
        if previous.is_some_and(|previous| previous >= chunk_index) {
            return Err(RusticolError::invalid_argument(
                "active evaluator chunk indices must be strictly increasing",
            ));
        }
        previous = Some(chunk_index);
    }
    Ok(())
}

fn validate_leaf_parameter_length(
    evaluator: &LoadedEvaluator,
    batch_size: usize,
    params: &[Complex<f64>],
) -> RusticolResult<()> {
    if params.len() != batch_size * evaluator.input_len {
        return Err(RusticolError::artifact(format!(
            "mapped evaluator parameter buffer has length {}, expected {}",
            params.len(),
            batch_size * evaluator.input_len
        )));
    }
    Ok(())
}

impl LoadedEvaluator {
    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn exact_evaluator(
        &mut self,
    ) -> RusticolResult<&ExpressionEvaluator<Complex<Rational>>> {
        if self.exact_eval.is_none() {
            let source = self.exact_eval_source.as_ref().ok_or_else(|| {
                RusticolError::invalid_argument(
                    "high-precision evaluation requires an evaluator-state artifact, but this process artifact has no evaluator_state_path for one or more chunks",
                )
            })?;
            self.exact_eval = Some(load_evaluator_state_source(source)?.1);
        }
        Ok(self
            .exact_eval
            .as_ref()
            .expect("exact evaluator initialized from its state path"))
    }

    pub(crate) fn evaluate_f64_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        match &mut self.eval {
            #[cfg(feature = "f64-symjit")]
            F64Evaluator::SymjitApplication(eval) => eval.evaluate_batch(batch_size, params, out),
            #[cfg(feature = "f64-compiled")]
            F64Evaluator::Compiled(eval) => eval.evaluate_batch(batch_size, params, out),
            #[cfg(feature = "symbolica-runtime")]
            F64Evaluator::Jit(eval) => {
                if params.len() != batch_size * self.input_len {
                    return Err(RusticolError::invalid_argument(format!(
                        "parameter buffer has length {}, expected {}",
                        params.len(),
                        batch_size * self.input_len
                    )));
                }
                if out.len() != batch_size * self.output_len {
                    return Err(RusticolError::invalid_argument(format!(
                        "output buffer has length {}, expected {}",
                        out.len(),
                        batch_size * self.output_len
                    )));
                }
                eval.evaluate_batch(batch_size, params, out)
                    .map_err(RusticolError::evaluation)
            }
        }
    }

    fn evaluate_f64_batch_unpadded(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> RusticolResult<()> {
        match &mut self.eval {
            #[cfg(feature = "f64-symjit")]
            F64Evaluator::SymjitApplication(eval) => {
                eval.evaluate_batch_unpadded(batch_size, params, out)
            }
            #[cfg(feature = "f64-compiled")]
            F64Evaluator::Compiled(eval) => eval.evaluate_batch(batch_size, params, out),
            #[cfg(feature = "symbolica-runtime")]
            F64Evaluator::Jit(eval) => {
                if params.len() != batch_size * self.input_len {
                    return Err(RusticolError::invalid_argument(format!(
                        "parameter buffer has length {}, expected {}",
                        params.len(),
                        batch_size * self.input_len
                    )));
                }
                if out.len() != batch_size * self.output_len {
                    return Err(RusticolError::invalid_argument(format!(
                        "output buffer has length {}, expected {}",
                        out.len(),
                        batch_size * self.output_len
                    )));
                }
                eval.evaluate_batch(batch_size, params, out)
                    .map_err(RusticolError::evaluation)
            }
        }
    }
}

fn flatten_evaluators_with_mappings(
    manifest: &EvaluatorManifest,
    payloads: &EvaluatorPayloadStore,
    inherited_mapping: Option<&[usize]>,
    root_input_len: usize,
    output: &mut Vec<LoadedEvaluator>,
    input_mappings: &mut Vec<Option<Vec<usize>>>,
) -> RusticolResult<()> {
    if let EvaluatorManifest::Chunked {
        input_len,
        chunk_input_indices,
        chunks,
        ..
    } = manifest
    {
        manifest.io_len()?;
        match (input_len, chunk_input_indices) {
            (None, None) => {
                for chunk in chunks {
                    flatten_evaluators_with_mappings(
                        chunk,
                        payloads,
                        inherited_mapping,
                        root_input_len,
                        output,
                        input_mappings,
                    )?;
                }
            }
            (Some(_), Some(chunk_indices)) => {
                for (chunk, indices) in chunks.iter().zip(chunk_indices) {
                    let composed =
                        compose_input_mapping(inherited_mapping, indices, root_input_len);
                    flatten_evaluators_with_mappings(
                        chunk,
                        payloads,
                        composed.as_deref(),
                        root_input_len,
                        output,
                        input_mappings,
                    )?;
                }
            }
            _ => unreachable!("chunk input metadata was validated above"),
        }
        return Ok(());
    }

    let (leaf_input_len, _) = manifest.io_len()?;
    let mapping = inherited_mapping.map(ToOwned::to_owned);
    if let Some(indices) = mapping.as_ref()
        && indices.len() != leaf_input_len
    {
        return Err(RusticolError::artifact(
            "flattened evaluator input map does not match leaf input length",
        ));
    }
    let normalized = mapping.and_then(|indices| {
        if indices.len() == root_input_len
            && indices
                .iter()
                .enumerate()
                .all(|(expected, index)| expected == *index)
        {
            None
        } else {
            Some(indices)
        }
    });
    flatten_evaluators_from_store(manifest, payloads, output)?;
    input_mappings.push(normalized);
    Ok(())
}

fn compose_input_mapping(
    inherited_mapping: Option<&[usize]>,
    child_indices: &[usize],
    root_input_len: usize,
) -> Option<Vec<usize>> {
    let composed = match inherited_mapping {
        Some(parent_indices) => child_indices
            .iter()
            .map(|index| parent_indices[*index])
            .collect::<Vec<_>>(),
        None => child_indices.to_vec(),
    };
    if composed.len() == root_input_len
        && composed
            .iter()
            .enumerate()
            .all(|(expected, index)| expected == *index)
    {
        None
    } else {
        Some(composed)
    }
}

pub(crate) fn flatten_evaluators_from_store(
    manifest: &EvaluatorManifest,
    payloads: &EvaluatorPayloadStore,
    output: &mut Vec<LoadedEvaluator>,
) -> RusticolResult<()> {
    match manifest {
        EvaluatorManifest::SymjitApplication {
            runtime_capability,
            application_path,
            application_abi,
            input_len,
            output_len,
            element_layout,
            batch_layout,
            compiler_type,
            translation_mode,
            optimization_level,
            word_bits,
            endianness,
            required_defuns,
            evaluator_state_path,
            evaluator_state_runtime_capability,
        } => {
            #[cfg(feature = "f64-symjit")]
            {
                let application_source = payloads.source(application_path)?;
                let application_bytes = application_source.read()?;
                let eval = SymjitApplicationEvaluator::load_bytes(
                    application_bytes.as_ref(),
                    PathBuf::from(application_source.display_name()),
                    SymjitApplicationMetadata {
                        runtime_capability,
                        application_abi,
                        input_len: *input_len,
                        output_len: *output_len,
                        element_layout,
                        batch_layout,
                        compiler_type,
                        translation_mode,
                        optimization_level: *optimization_level,
                        word_bits: *word_bits,
                        endianness,
                        required_defuns,
                    },
                )?;
                #[cfg(feature = "symbolica-runtime")]
                let exact_eval_source = match evaluator_state_path {
                    Some(state_path) => {
                        if evaluator_state_runtime_capability.as_deref()
                            != Some(SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY)
                        {
                            return Err(RusticolError::compatibility(format!(
                                "SymJIT evaluator state {:?} does not declare capability {:?}",
                                state_path, SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY
                            )));
                        }
                        Some(payloads.source(state_path)?)
                    }
                    None => None,
                };
                #[cfg(not(feature = "symbolica-runtime"))]
                let _ = (evaluator_state_path, evaluator_state_runtime_capability);
                output.push(LoadedEvaluator {
                    eval: F64Evaluator::SymjitApplication(eval),
                    #[cfg(feature = "symbolica-runtime")]
                    exact_eval: None,
                    #[cfg(feature = "symbolica-runtime")]
                    exact_eval_source,
                    #[cfg(feature = "symbolica-runtime")]
                    double_eval: None,
                    #[cfg(feature = "symbolica-runtime")]
                    arb_eval: None,
                    input_len: *input_len,
                    output_len: *output_len,
                });
                Ok(())
            }
            #[cfg(not(feature = "f64-symjit"))]
            {
                let _ = (
                    application_path,
                    application_abi,
                    input_len,
                    output_len,
                    element_layout,
                    batch_layout,
                    compiler_type,
                    translation_mode,
                    optimization_level,
                    word_bits,
                    endianness,
                    required_defuns,
                    evaluator_state_path,
                    evaluator_state_runtime_capability,
                );
                Err(unsupported_capability(runtime_capability))
            }
        }
        EvaluatorManifest::Jit {
            runtime_capability,
            input_len,
            output_len,
            evaluator_state_path,
        } => {
            #[cfg(feature = "symbolica-runtime")]
            {
                let _ = runtime_capability;
                let (jit_settings, exact_eval, jit_eval) =
                    load_evaluator_state_source(&payloads.source(evaluator_state_path)?)?;
                let eval = F64Evaluator::Jit(match jit_eval {
                    Some(eval) => eval,
                    None => exact_eval
                        .jit_compile::<Complex<f64>>(jit_settings)
                        .map_err(|err| {
                            RusticolError::evaluation(format!(
                                "could not compile scalar JIT evaluator from {}: {err}",
                                evaluator_state_path
                            ))
                        })?,
                });
                output.push(LoadedEvaluator {
                    eval,
                    exact_eval: Some(exact_eval),
                    exact_eval_source: None,
                    double_eval: None,
                    arb_eval: None,
                    input_len: *input_len,
                    output_len: *output_len,
                });
                Ok(())
            }
            #[cfg(not(feature = "symbolica-runtime"))]
            {
                let _ = (input_len, output_len, evaluator_state_path);
                Err(unsupported_capability(runtime_capability))
            }
        }
        EvaluatorManifest::CompiledComplex {
            runtime_capability,
            function_name,
            input_len,
            output_len,
            library_path,
            evaluator_state_path,
            number_type,
        } => {
            #[cfg(feature = "f64-compiled")]
            {
                let _ = runtime_capability;
                if number_type != "complex" {
                    return Err(RusticolError::invalid_argument(format!(
                        "rusticol currently supports compiled complex evaluators, got {number_type}"
                    )));
                }
                let library = payloads.physical_path(library_path)?;
                let eval = CompiledComplexF64Evaluator::load(
                    &library,
                    function_name,
                    *input_len,
                    *output_len,
                )?;
                #[cfg(feature = "symbolica-runtime")]
                let exact_eval_source = evaluator_state_path
                    .as_deref()
                    .map(|state_path| payloads.source(state_path))
                    .transpose()?;
                #[cfg(not(feature = "symbolica-runtime"))]
                let _ = evaluator_state_path;
                output.push(LoadedEvaluator {
                    eval: F64Evaluator::Compiled(eval),
                    #[cfg(feature = "symbolica-runtime")]
                    exact_eval: None,
                    #[cfg(feature = "symbolica-runtime")]
                    exact_eval_source,
                    #[cfg(feature = "symbolica-runtime")]
                    double_eval: None,
                    #[cfg(feature = "symbolica-runtime")]
                    arb_eval: None,
                    input_len: *input_len,
                    output_len: *output_len,
                });
                Ok(())
            }
            #[cfg(not(feature = "f64-compiled"))]
            {
                let _ = (
                    function_name,
                    input_len,
                    output_len,
                    library_path,
                    evaluator_state_path,
                    number_type,
                );
                Err(unsupported_capability(runtime_capability))
            }
        }
        EvaluatorManifest::Chunked { chunks, .. } => {
            for chunk in chunks {
                flatten_evaluators_from_store(chunk, payloads, output)?;
            }
            Ok(())
        }
    }
}

pub(crate) fn ensure_evaluator_capabilities_supported(
    manifest: &EvaluatorManifest,
) -> RusticolResult<()> {
    let declared = evaluator_runtime_capabilities(manifest)?;
    ensure_runtime_capabilities_supported(declared.iter().map(String::as_str))
}

pub(crate) fn evaluator_runtime_capabilities(
    manifest: &EvaluatorManifest,
) -> RusticolResult<BTreeSet<String>> {
    let mut capabilities = BTreeSet::new();
    collect_evaluator_capabilities(manifest, &mut capabilities)?;
    Ok(capabilities)
}

fn collect_evaluator_capabilities(
    manifest: &EvaluatorManifest,
    output: &mut BTreeSet<String>,
) -> RusticolResult<()> {
    match manifest {
        EvaluatorManifest::SymjitApplication {
            runtime_capability, ..
        } => validate_and_insert_capability(
            runtime_capability,
            &[SYMJIT_APPLICATION_RUNTIME_CAPABILITY],
            output,
        ),
        EvaluatorManifest::Jit {
            runtime_capability, ..
        } => validate_and_insert_capability(
            runtime_capability,
            &[SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY],
            output,
        ),
        EvaluatorManifest::CompiledComplex {
            runtime_capability, ..
        } => validate_and_insert_capability(
            runtime_capability,
            &[
                SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY,
                SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY,
            ],
            output,
        ),
        EvaluatorManifest::Chunked {
            required_runtime_capabilities,
            chunks,
            ..
        } => {
            let mut actual = BTreeSet::new();
            for chunk in chunks {
                collect_evaluator_capabilities(chunk, &mut actual)?;
            }
            let declared = required_runtime_capabilities
                .iter()
                .cloned()
                .collect::<BTreeSet<_>>();
            if declared.len() != required_runtime_capabilities.len() || declared != actual {
                return Err(RusticolError::integrity(format!(
                    "chunked evaluator capabilities {:?} do not match child capabilities {:?}",
                    required_runtime_capabilities, actual
                )));
            }
            output.extend(actual);
            Ok(())
        }
    }
}

fn validate_and_insert_capability(
    capability: &str,
    expected: &[&str],
    output: &mut BTreeSet<String>,
) -> RusticolResult<()> {
    if !expected.contains(&capability) {
        return Err(RusticolError::compatibility(format!(
            "evaluator declares runtime capability {capability:?}, expected one of {expected:?}"
        )));
    }
    output.insert(capability.to_string());
    Ok(())
}

#[cfg(any(
    not(feature = "f64-compiled"),
    not(feature = "f64-symjit"),
    not(feature = "symbolica-runtime")
))]
fn unsupported_capability(capability: &str) -> RusticolError {
    RusticolError::unsupported_runtime_capability(
        capability,
        format!(
            "this Rusticol build supports {:?}",
            supported_runtime_capabilities()
        ),
    )
}

#[cfg(feature = "symbolica-runtime")]
fn load_evaluator_state_source(
    source: &EvaluatorPayloadSource,
) -> RusticolResult<(
    JITCompilationSettings,
    ExpressionEvaluator<Complex<Rational>>,
    Option<JITCompiledEvaluator<Complex<f64>>>,
)> {
    let bytes = source.read()?;
    load_evaluator_state_bytes(bytes.as_ref(), &source.display_name())
}

#[cfg(feature = "symbolica-runtime")]
fn load_evaluator_state_bytes(
    bytes: &[u8],
    display_name: &str,
) -> RusticolResult<(
    JITCompilationSettings,
    ExpressionEvaluator<Complex<Rational>>,
    Option<JITCompiledEvaluator<Complex<f64>>>,
)> {
    type SavedEvaluator = (
        bool,
        JITCompilationSettings,
        ExpressionEvaluator<Complex<Rational>>,
        Option<JITCompiledEvaluator<f64>>,
        Option<JITCompiledEvaluator<Complex<f64>>>,
    );

    let ((_, settings, evaluator, _, jit_complex), consumed) =
        bincode::decode_from_slice::<SavedEvaluator, _>(bytes, bincode::config::standard())
            .map_err(|error| {
                RusticolError::compatibility(format!(
                    "evaluator state {} does not match Symbolica serialization ABI {}: {error}; regenerate the schema-v3 artifact",
                    display_name,
                    crate::SYMBOLICA_SERIALIZATION_ABI
                ))
            })?;
    ensure_evaluator_state_consumed(display_name, bytes.len(), consumed)?;
    Ok((settings, evaluator, jit_complex))
}

#[cfg(feature = "symbolica-runtime")]
fn ensure_evaluator_state_consumed(
    display_name: &str,
    encoded_len: usize,
    consumed: usize,
) -> RusticolResult<()> {
    if consumed == encoded_len {
        return Ok(());
    }
    Err(RusticolError::integrity(format!(
        "evaluator state {} contains {} trailing bytes",
        display_name,
        encoded_len - consumed
    )))
}
