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
            parameter_scratch_native2: Vec::new(),
            output_scratch_native2: Vec::new(),
            evaluator,
        })
    }

    pub(crate) fn evaluate_f64_into_state(
        &mut self,
        batch_size: usize,
        parameter_count: usize,
        state: &mut [Complex<f64>],
    ) -> RusticolResult<(f64, f64, f64)> {
        if self.evaluator.supports_native2() {
            return self.evaluate_f64_native2_into_state(batch_size, parameter_count, state);
        }
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

    fn evaluate_f64_native2_into_state(
        &mut self,
        batch_size: usize,
        parameter_count: usize,
        state: &mut [Complex<f64>],
    ) -> RusticolResult<(f64, f64, f64)> {
        let pack_start = Instant::now();
        pack_native2_parameters(
            batch_size,
            parameter_count,
            self.input_components.as_deref(),
            state,
            &mut self.parameter_scratch_native2,
        )?;
        let input_pack_s = pack_start.elapsed().as_secs_f64();

        let evaluator_start = Instant::now();
        self.evaluator.evaluate_native2_into(
            batch_size.div_ceil(2),
            &self.parameter_scratch_native2,
            &mut self.output_scratch_native2,
        )?;
        let evaluator_s = evaluator_start.elapsed().as_secs_f64();

        let assign_start = Instant::now();
        let output_len = self.evaluator.output_len;
        for row in 0..batch_size {
            let state_row = row * parameter_count;
            let native_row = row / 2 * output_len;
            let lane = row % 2;
            for (output_column, state_offset) in &self.outputs {
                let value = self.output_scratch_native2[native_row + *output_column];
                state[state_row + *state_offset] =
                    c64(value.re.as_array()[lane], value.im.as_array()[lane]);
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

pub(crate) fn pack_native2_parameters(
    batch_size: usize,
    global_parameter_count: usize,
    input_components: Option<&[usize]>,
    state: &[Complex<f64>],
    target: &mut Vec<Complex<wide::f64x2>>,
) -> RusticolResult<()> {
    if batch_size == 0 {
        return Err(RusticolError::invalid_argument(
            "native two-lane evaluation requires a non-empty batch",
        ));
    }
    if state.len() != batch_size * global_parameter_count {
        return Err(RusticolError::invalid_argument(format!(
            "state buffer has length {}, expected {}",
            state.len(),
            batch_size * global_parameter_count
        )));
    }
    let local_parameter_count = input_components
        .map(|components| components.len())
        .unwrap_or(global_parameter_count);
    target.resize(
        batch_size.div_ceil(2) * local_parameter_count,
        Complex::new(wide::f64x2::ZERO, wide::f64x2::ZERO),
    );
    for native_row in 0..batch_size.div_ceil(2) {
        let first_row = native_row * 2;
        let second_row = usize::min(first_row + 1, batch_size - 1);
        let target_row = native_row * local_parameter_count;
        for local_index in 0..local_parameter_count {
            let global_index = input_components
                .map(|components| components[local_index])
                .unwrap_or(local_index);
            let first = state[first_row * global_parameter_count + global_index];
            let second = state[second_row * global_parameter_count + global_index];
            target[target_row + local_index] = Complex::new(
                wide::f64x2::new([first.re, second.re]),
                wide::f64x2::new([first.im, second.im]),
            );
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native2_parameter_packing_duplicates_the_final_row_of_an_odd_batch() {
        let state = vec![
            c64(1.0, 10.0),
            c64(2.0, 20.0),
            c64(3.0, 30.0),
            c64(4.0, 40.0),
            c64(5.0, 50.0),
            c64(6.0, 60.0),
        ];
        let mut packed = Vec::new();

        pack_native2_parameters(3, 2, None, &state, &mut packed).unwrap();

        assert_eq!(packed.len(), 4);
        assert_eq!(packed[0].re.as_array(), &[1.0, 3.0]);
        assert_eq!(packed[0].im.as_array(), &[10.0, 30.0]);
        assert_eq!(packed[1].re.as_array(), &[2.0, 4.0]);
        assert_eq!(packed[1].im.as_array(), &[20.0, 40.0]);
        assert_eq!(packed[2].re.as_array(), &[5.0, 5.0]);
        assert_eq!(packed[2].im.as_array(), &[50.0, 50.0]);
        assert_eq!(packed[3].re.as_array(), &[6.0, 6.0]);
        assert_eq!(packed[3].im.as_array(), &[60.0, 60.0]);
    }
}
