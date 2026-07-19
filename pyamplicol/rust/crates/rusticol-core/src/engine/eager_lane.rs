// SPDX-License-Identifier: 0BSD

use super::*;

pub(super) struct EagerNativeRuntime {
    scheduler: crate::EagerExecutionRuntime,
    backend: PreparedEvaluatorBackend,
    backend_name: String,
    has_derived_parameter_kernel: bool,
    initial_values: Vec<Complex<f64>>,
    momenta: Vec<f64>,
    model_parameters: Vec<Complex<f64>>,
    amplitudes: Vec<Complex<f64>>,
    reduced: Vec<f64>,
}

impl EagerNativeRuntime {
    pub(super) fn new(
        scheduler: crate::EagerExecutionRuntime,
        backend: PreparedEvaluatorBackend,
        backend_name: String,
        has_derived_parameter_kernel: bool,
    ) -> Self {
        Self {
            scheduler,
            backend,
            backend_name,
            has_derived_parameter_kernel,
            initial_values: Vec::new(),
            momenta: Vec::new(),
            model_parameters: Vec::new(),
            amplitudes: Vec::new(),
            reduced: Vec::new(),
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

    pub(super) fn supports_parameter_updates(&self) -> bool {
        !self.has_derived_parameter_kernel
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
        self.model_parameters.clear();
        self.model_parameters.extend(
            common
                .model_parameter_values_f64
                .iter()
                .copied()
                .map(|value| Complex::new(value, 0.0)),
        );
        let amplitude_len = point_count
            .checked_mul(self.scheduler.plan().amplitude_count())
            .ok_or_else(|| RusticolError::invalid_argument("eager amplitudes overflow"))?;
        self.amplitudes
            .resize(amplitude_len, Complex::new(0.0, 0.0));
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
}

fn transpose_state_components(
    state: &[Complex<f64>],
    point_count: usize,
    row_width: usize,
    component_start: usize,
    component_count: usize,
    output: &mut Vec<Complex<f64>>,
) -> RusticolResult<()> {
    let output_len = point_count
        .checked_mul(component_count)
        .ok_or_else(|| RusticolError::invalid_argument("eager value input overflows"))?;
    output.resize(output_len, Complex::new(0.0, 0.0));
    for component in 0..component_count {
        for point in 0..point_count {
            output[component * point_count + point] =
                state[point * row_width + component_start + component];
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
