// SPDX-License-Identifier: 0BSD

use super::*;

pub(super) fn parse_complex_parameter_overrides(
    text: &str,
    path: &Path,
) -> RusticolResult<BTreeMap<String, (f64, f64)>> {
    let value = serde_json::from_str::<Value>(text).map_err(|err| {
        RusticolError::invalid_argument(format!(
            "could not parse model-parameter JSON {}: {err}",
            path.display()
        ))
    })?;
    let object = value.as_object().ok_or_else(|| {
        RusticolError::invalid_argument(format!(
            "model-parameter JSON {} must contain an object",
            path.display()
        ))
    })?;
    let mut overrides = BTreeMap::new();
    for (name, value) in object {
        let components = value.as_array().ok_or_else(|| {
            RusticolError::invalid_argument(format!(
                "model-parameter JSON entry {name:?} must be [real, imaginary]"
            ))
        })?;
        if components.len() != 2 {
            return Err(RusticolError::invalid_argument(format!(
                "model-parameter JSON entry {name:?} must have exactly two components"
            )));
        }
        let real = components[0].as_f64().ok_or_else(|| {
            RusticolError::invalid_argument(format!(
                "model-parameter JSON entry {name:?} has a non-numeric real component"
            ))
        })?;
        let imaginary = components[1].as_f64().ok_or_else(|| {
            RusticolError::invalid_argument(format!(
                "model-parameter JSON entry {name:?} has a non-numeric imaginary component"
            ))
        })?;
        if !real.is_finite() || !imaginary.is_finite() {
            return Err(RusticolError::invalid_argument(format!(
                "model-parameter JSON entry {name:?} must contain finite values"
            )));
        }
        overrides.insert(name.clone(), (real, imaginary));
    }
    Ok(overrides)
}

impl ExecutionRuntime {
    pub(super) fn apply_model_parameter_overrides(
        &mut self,
        overrides: &BTreeMap<String, (f64, f64)>,
    ) -> RusticolResult<()> {
        let mut proposed = self.model_parameter_values_f64.clone();
        for (name, (real, imaginary)) in overrides {
            let Some(slots) = self.model_parameter_runtime_slots.get(name).copied() else {
                return Err(RusticolError::invalid_argument(format!(
                    "model-parameter override {name:?} is not used by process {}",
                    self.process
                )));
            };
            if slots.real >= self.model_parameter_values_f64.len()
                || !real.is_finite()
                || !imaginary.is_finite()
            {
                return Err(RusticolError::invalid_argument(format!(
                    "model-parameter override {name:?} has invalid value [{real}, {imaginary}]",
                )));
            }
            proposed[slots.real] = *real;
            if let Some(index) = slots.imaginary {
                if index >= self.model_parameter_values_f64.len() {
                    return Err(RusticolError::invalid_argument(format!(
                        "model-parameter override {name:?} has an invalid imaginary slot"
                    )));
                }
                proposed[index] = *imaginary;
            } else if *imaginary != 0.0 {
                return Err(RusticolError::invalid_argument(format!(
                    "real model parameter {name:?} cannot receive a nonzero imaginary component"
                )));
            }
        }
        let previous_values = std::mem::replace(&mut self.model_parameter_values_f64, proposed);
        let previous_masses = self.particle_masses.clone();
        let previous_normalization = self.normalization_factor;
        if let Err(error) = self.refresh_derived_model_parameters() {
            self.model_parameter_values_f64 = previous_values;
            self.particle_masses = previous_masses;
            self.normalization_factor = previous_normalization;
            return Err(error);
        }
        self.refresh_particle_mass_parameters();
        self.refresh_normalization_factor();
        Ok(())
    }

    pub(super) fn refresh_derived_model_parameters(&mut self) -> RusticolResult<()> {
        let Some(runtime) = self.model_parameter_evaluator.as_mut() else {
            return Ok(());
        };
        let parameters = runtime
            .input_parameter_indices
            .iter()
            .map(|index| {
                self.model_parameter_values_f64
                    .get(*index)
                    .copied()
                    .map(|value| c64(value, 0.0))
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(format!(
                            "model-parameter evaluator input index {index} is out of range"
                        ))
                    })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let evaluated = runtime.evaluator.evaluate_single_row(&parameters)?;
        for output in &runtime.outputs {
            let value = evaluated.get(output.output_index).ok_or_else(|| {
                RusticolError::invalid_argument(format!(
                    "model-parameter evaluator output {} for {:?} is absent",
                    output.output_index, output.runtime_name
                ))
            })?;
            let Some(real) = self
                .model_parameter_values_f64
                .get_mut(output.real_parameter_index)
            else {
                return Err(RusticolError::invalid_argument(format!(
                    "derived model-parameter real slot {} is out of range",
                    output.real_parameter_index
                )));
            };
            *real = value.re;
            let Some(imaginary) = self
                .model_parameter_values_f64
                .get_mut(output.imag_parameter_index)
            else {
                return Err(RusticolError::invalid_argument(format!(
                    "derived model-parameter imaginary slot {} is out of range",
                    output.imag_parameter_index
                )));
            };
            *imaginary = value.im;
        }
        Ok(())
    }

    pub(super) fn refresh_particle_mass_parameters(&mut self) {
        for parameter in &self.model_parameters {
            if parameter.kind == "particle_mass"
                && let Some(pdg) = parameter.pdg
                && let Some(value) = self
                    .model_parameter_values_f64
                    .get(parameter.parameter_index)
            {
                self.particle_masses.insert(pdg, *value);
            }
        }
        for (pdg, name) in &self.particle_mass_parameter_names {
            let Some(slots) = self.model_parameter_runtime_slots.get(name) else {
                continue;
            };
            if let Some(value) = self.model_parameter_values_f64.get(slots.real) {
                self.particle_masses.insert(*pdg, *value);
            }
        }
    }

    pub(super) fn refresh_normalization_factor(&mut self) {
        let alpha_s = self
            .model_parameter_name_to_index
            .get("normalization.alpha_s_me_check")
            .and_then(|index| self.model_parameter_values_f64.get(*index))
            .copied();
        let alpha_ew = self
            .model_parameter_name_to_index
            .get("normalization.alpha_ew")
            .and_then(|index| self.model_parameter_values_f64.get(*index))
            .copied();
        let mut global_coupling_factor = 1.0;
        if self.normalization_qcd_coupling_power > 0 {
            let Some(alpha_s) = alpha_s else {
                return;
            };
            global_coupling_factor *= (4.0 * std::f64::consts::PI * alpha_s)
                .powi(self.normalization_qcd_coupling_power as i32);
        }
        if self.normalization_electroweak_coupling_power > 0 {
            let Some(alpha_ew) = alpha_ew else {
                return;
            };
            global_coupling_factor *= (2.0 * 4.0 * std::f64::consts::PI * alpha_ew)
                .powi(self.normalization_electroweak_coupling_power as i32);
        }
        self.normalization_factor = self.normalization_color_factor * global_coupling_factor
            / (self.normalization_average_factor * self.normalization_identical_factor);
    }
}
