// SPDX-License-Identifier: 0BSD

use super::super::*;

impl StageRuntime {
    pub(crate) fn load(
        stage: &GenericSerializedStageEvaluatorManifest,
        root: &Path,
    ) -> RusticolResult<Self> {
        let evaluator = EvaluatorGroup::load(&stage.evaluator, root)?;
        let mut outputs = Vec::new();
        for slot in &stage.output_slots {
            let output_len = slot
                .output_stop
                .checked_sub(slot.output_start)
                .ok_or_else(|| {
                    RusticolError::invalid_argument(format!(
                        "generic stage {} has an invalid output slot range",
                        stage.evaluator_label
                    ))
                })?;
            let component_len = slot
                .component_stop
                .checked_sub(slot.component_start)
                .ok_or_else(|| {
                    RusticolError::invalid_argument(format!(
                        "generic stage {} has an invalid component slot range",
                        stage.evaluator_label
                    ))
                })?;
            if output_len != component_len {
                return Err(RusticolError::invalid_argument(format!(
                    "generic stage {} output slot length does not match component length",
                    stage.evaluator_label
                )));
            }
            for component in 0..component_len {
                outputs.push((
                    slot.output_start + component,
                    slot.component_start + component,
                ));
            }
        }
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
        let output_spans = contiguous_output_spans(&outputs);
        Ok(Self {
            outputs,
            output_spans,
            input_components,
            input_spans,
            parameter_scratch_f64: Vec::new(),
            output_scratch_f64: Vec::new(),
            evaluator,
        })
    }

    pub(crate) fn evaluate_f64_into_state(
        &mut self,
        batch_size: usize,
        parameter_count: usize,
        state: &mut [Complex<f64>],
    ) -> RusticolResult<(f64, f64, f64)> {
        let mut input_pack_s = 0.0;
        let eval_start;
        if let Some(input_components) = self.input_components.as_ref() {
            let local_parameter_count = input_components.len();
            let pack_start = Instant::now();
            self.parameter_scratch_f64
                .resize(batch_size * local_parameter_count, c64(0.0, 0.0));
            for row in 0..batch_size {
                let row_state = row * parameter_count;
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
            input_pack_s = pack_start.elapsed().as_secs_f64();
            eval_start = Instant::now();
            self.evaluator.evaluate_batch_into(
                batch_size,
                &self.parameter_scratch_f64,
                &mut self.output_scratch_f64,
            )?;
        } else {
            eval_start = Instant::now();
            self.evaluator
                .evaluate_batch_into(batch_size, state, &mut self.output_scratch_f64)?;
        }
        let evaluator_s = eval_start.elapsed().as_secs_f64();

        let assign_start = Instant::now();
        for row in 0..batch_size {
            let row_state = row * parameter_count;
            let row_eval = row * self.evaluator.output_len;
            if self.output_spans.is_empty() {
                for (column, state_offset) in &self.outputs {
                    state[row_state + *state_offset] = self.output_scratch_f64[row_eval + *column];
                }
            } else {
                for (column_start, state_offset_start, len) in &self.output_spans {
                    let source_start = row_eval + *column_start;
                    let target_start = row_state + *state_offset_start;
                    state[target_start..target_start + *len].copy_from_slice(
                        &self.output_scratch_f64[source_start..source_start + *len],
                    );
                }
            }
        }
        Ok((
            input_pack_s,
            evaluator_s,
            assign_start.elapsed().as_secs_f64(),
        ))
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn evaluate_generic_into_state<T>(
        &mut self,
        batch_size: usize,
        parameter_count: usize,
        state: &mut [Complex<T>],
        binary_precision: Option<u32>,
    ) -> RusticolResult<(f64, f64, f64)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let mut input_pack_s = 0.0;
        let (evaluated, evaluator_s) =
            if let Some(input_components) = self.input_components.as_ref() {
                let local_parameter_count = input_components.len();
                let pack_start = Instant::now();
                let mut parameter_scratch =
                    vec![complex_zero::<T>(); batch_size * local_parameter_count];
                for row in 0..batch_size {
                    let row_state = row * parameter_count;
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

        self.assign_generic_outputs(
            batch_size,
            parameter_count,
            state,
            evaluated,
            input_pack_s,
            evaluator_s,
        )
    }

    #[cfg(feature = "symbolica-runtime")]
    pub(crate) fn assign_generic_outputs<T>(
        &self,
        batch_size: usize,
        parameter_count: usize,
        state: &mut [Complex<T>],
        evaluated: Vec<Complex<T>>,
        input_pack_s: f64,
        evaluator_s: f64,
    ) -> RusticolResult<(f64, f64, f64)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let assign_start = Instant::now();
        for row in 0..batch_size {
            let row_state = row * parameter_count;
            let row_eval = row * self.evaluator.output_len;
            if self.output_spans.is_empty() {
                for (column, state_offset) in &self.outputs {
                    state[row_state + *state_offset] = evaluated[row_eval + *column].clone();
                }
            } else {
                for (column_start, state_offset_start, len) in &self.output_spans {
                    let source_start = row_eval + *column_start;
                    let target_start = row_state + *state_offset_start;
                    state[target_start..target_start + *len]
                        .clone_from_slice(&evaluated[source_start..source_start + *len]);
                }
            }
        }
        Ok((
            input_pack_s,
            evaluator_s,
            assign_start.elapsed().as_secs_f64(),
        ))
    }
}

pub(crate) fn contiguous_input_spans(input_components: &[usize]) -> Vec<(usize, usize, usize)> {
    if input_components.is_empty() {
        return Vec::new();
    }
    let mut spans = Vec::new();
    let mut local_start = 0usize;
    let mut global_start = input_components[0];
    let mut previous_local = local_start;
    let mut previous_global = global_start;
    let mut len = 1usize;
    for (local, global) in input_components.iter().copied().enumerate().skip(1) {
        if local == previous_local + 1 && global == previous_global + 1 {
            previous_local = local;
            previous_global = global;
            len += 1;
            continue;
        }
        spans.push((local_start, global_start, len));
        local_start = local;
        global_start = global;
        previous_local = local;
        previous_global = global;
        len = 1;
    }
    spans.push((local_start, global_start, len));
    if spans.len() >= input_components.len() {
        Vec::new()
    } else {
        spans
    }
}

pub(crate) fn contiguous_output_spans(outputs: &[(usize, usize)]) -> Vec<(usize, usize, usize)> {
    if outputs.is_empty() {
        return Vec::new();
    }
    let mut spans = Vec::new();
    let (mut output_start, mut state_start) = outputs[0];
    let mut previous_output = output_start;
    let mut previous_state = state_start;
    let mut len = 1usize;
    for (output, state) in outputs.iter().copied().skip(1) {
        if output == previous_output + 1 && state == previous_state + 1 {
            previous_output = output;
            previous_state = state;
            len += 1;
            continue;
        }
        spans.push((output_start, state_start, len));
        output_start = output;
        state_start = state;
        previous_output = output;
        previous_state = state;
        len = 1;
    }
    spans.push((output_start, state_start, len));
    if spans.len() >= outputs.len() {
        Vec::new()
    } else {
        spans
    }
}
