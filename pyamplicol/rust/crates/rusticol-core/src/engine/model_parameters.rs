// SPDX-License-Identifier: 0BSD

use super::*;
#[cfg(feature = "symbolica-runtime")]
use symbolica::{atom::Atom, atom::AtomCore, parser::ParseSettings};

struct RuntimeParameterStateSnapshot {
    values: Vec<f64>,
    particle_masses: BTreeMap<i32, f64>,
    normalization_factor: f64,
    helicity_sum: Option<Box<RuntimeParameterStateSnapshot>>,
    helicity_selectors: Vec<RuntimeParameterStateSnapshot>,
    color_selectors: BTreeMap<i64, RuntimeParameterStateSnapshot>,
}

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
        let (real, imaginary) = if let Some(real) = value.as_f64() {
            (real, 0.0)
        } else {
            let components = value.as_array().ok_or_else(|| {
                RusticolError::invalid_argument(format!(
                    "model-parameter JSON entry {name:?} must be a real number or [real, imaginary]"
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
            (real, imaginary)
        };
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
    pub(super) fn current_model_parameter_value(&self, name: &str) -> RusticolResult<(f64, f64)> {
        let mut real = None;
        let mut imaginary = 0.0;
        for parameter in &self.model_parameters {
            if parameter.runtime_name.as_deref().unwrap_or(&parameter.name) != name {
                continue;
            }
            let value = self
                .model_parameter_values_f64
                .get(parameter.parameter_index)
                .copied()
                .ok_or_else(|| RusticolError::integrity("model-parameter value slot is absent"))?;
            match parameter.complex_component.as_deref() {
                Some("imag") => imaginary = value,
                Some("real") | None => real = Some(value),
                Some(_) => return Err(RusticolError::integrity("invalid parameter component")),
            }
        }
        real.map(|value| (value, imaginary)).ok_or_else(|| {
            RusticolError::model_parameter(format!("model parameter {name:?} has no runtime value"))
        })
    }

    pub(super) fn apply_model_parameter_overrides(
        &mut self,
        overrides: &BTreeMap<String, (f64, f64)>,
    ) -> RusticolResult<()> {
        let snapshot = self.parameter_state_snapshot();
        if let Err(error) = self.apply_model_parameter_overrides_recursive(overrides) {
            self.restore_parameter_state(snapshot);
            return Err(error);
        }
        Ok(())
    }

    fn apply_model_parameter_overrides_recursive(
        &mut self,
        overrides: &BTreeMap<String, (f64, f64)>,
    ) -> RusticolResult<()> {
        self.apply_model_parameter_overrides_local(overrides)?;
        if let Some(sum_runtime) = self.helicity_sum_runtime.as_mut() {
            sum_runtime.apply_model_parameter_overrides_recursive(overrides)?;
        }
        for selector_runtime in &mut self.helicity_selector_runtimes {
            selector_runtime.apply_model_parameter_overrides_recursive(overrides)?;
        }
        for selector_runtime in self.color_selector_runtimes.values_mut() {
            selector_runtime.apply_model_parameter_overrides_recursive(overrides)?;
        }
        Ok(())
    }

    fn parameter_state_snapshot(&self) -> RuntimeParameterStateSnapshot {
        RuntimeParameterStateSnapshot {
            values: self.model_parameter_values_f64.clone(),
            particle_masses: self.particle_masses.clone(),
            normalization_factor: self.normalization_factor,
            helicity_sum: self
                .helicity_sum_runtime
                .as_deref()
                .map(ExecutionRuntime::parameter_state_snapshot)
                .map(Box::new),
            helicity_selectors: self
                .helicity_selector_runtimes
                .iter()
                .map(|runtime| runtime.parameter_state_snapshot())
                .collect(),
            color_selectors: self
                .color_selector_runtimes
                .iter()
                .map(|(sector_id, runtime)| (*sector_id, runtime.parameter_state_snapshot()))
                .collect(),
        }
    }

    fn restore_parameter_state(&mut self, snapshot: RuntimeParameterStateSnapshot) {
        self.model_parameter_values_f64 = snapshot.values;
        self.particle_masses = snapshot.particle_masses;
        self.normalization_factor = snapshot.normalization_factor;
        if let (Some(runtime), Some(snapshot)) =
            (self.helicity_sum_runtime.as_mut(), snapshot.helicity_sum)
        {
            runtime.restore_parameter_state(*snapshot);
        }
        for (runtime, lane_snapshot) in self
            .helicity_selector_runtimes
            .iter_mut()
            .zip(snapshot.helicity_selectors)
        {
            runtime.restore_parameter_state(lane_snapshot);
        }
        for (sector_id, lane_snapshot) in snapshot.color_selectors {
            if let Some(runtime) = self.color_selector_runtimes.get_mut(&sector_id) {
                runtime.restore_parameter_state(lane_snapshot);
            }
        }
    }

    fn apply_model_parameter_overrides_local(
        &mut self,
        overrides: &BTreeMap<String, (f64, f64)>,
    ) -> RusticolResult<()> {
        let mut proposed = self.model_parameter_values_f64.clone();
        for (name, (real, imaginary)) in overrides {
            if self.model_parameters.iter().any(|parameter| {
                parameter.kind == "derived_parameter_component"
                    && parameter.runtime_name.as_deref().unwrap_or(&parameter.name) == name
            }) {
                return Err(RusticolError::invalid_argument(format!(
                    "derived model parameter {name:?} is immutable",
                )));
            }
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
        refresh_derived_model_parameter_values(
            self.model_parameter_evaluator.as_mut(),
            &mut proposed,
        )?;
        let proposed_masses = self.particle_mass_parameters_for(&proposed);
        validate_particle_mass_class_stability(&self.particle_masses, &proposed_masses)?;
        self.model_parameter_values_f64 = proposed;
        self.particle_masses = proposed_masses;
        self.refresh_normalization_factor();
        Ok(())
    }

    pub(super) fn refresh_derived_model_parameters(&mut self) -> RusticolResult<()> {
        refresh_derived_model_parameter_values(
            self.model_parameter_evaluator.as_mut(),
            &mut self.model_parameter_values_f64,
        )
    }

    /// The binary64 mass of particle `pdg` in the runtime that evaluates.
    pub(super) fn current_particle_mass(&self, pdg: i32) -> RusticolResult<f64> {
        if let Some(sum_runtime) = self.helicity_sum_runtime.as_deref() {
            return sum_runtime.current_particle_mass(pdg);
        }
        self.particle_masses.get(&pdg).copied().ok_or_else(|| {
            RusticolError::invalid_argument(format!("particle {pdg} has no mass in this runtime"))
        })
    }

    fn particle_mass_parameters_for(&self, values: &[f64]) -> BTreeMap<i32, f64> {
        let mut particle_masses = self.particle_masses.clone();
        for (pdg, slot) in self.particle_mass_slots() {
            if let Some(value) = values.get(slot) {
                particle_masses.insert(pdg, *value);
            }
        }
        particle_masses
    }

    /// (PDG code, model-parameter slot) of every mass that follows a model
    /// parameter, in the order in which they apply.
    fn particle_mass_slots(&self) -> Vec<(i32, usize)> {
        let mut slots = Vec::new();
        for parameter in &self.model_parameters {
            if parameter.kind == "particle_mass"
                && let Some(pdg) = parameter.pdg
            {
                slots.push((pdg, parameter.parameter_index));
            }
        }
        for (pdg, name) in &self.particle_mass_parameter_names {
            let Some(slots_of_name) = self.model_parameter_runtime_slots.get(name) else {
                continue;
            };
            slots.push((*pdg, slots_of_name.real));
        }
        slots
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

pub(super) fn refresh_derived_model_parameter_values(
    runtime: Option<&mut ModelParameterEvaluatorRuntime>,
    values: &mut [f64],
) -> RusticolResult<()> {
    let Some(runtime) = runtime else {
        return Ok(());
    };
    let parameters = runtime
        .input_parameter_indices
        .iter()
        .map(|index| {
            values
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
        let Some(real) = values.get_mut(output.real_parameter_index) else {
            return Err(RusticolError::invalid_argument(format!(
                "derived model-parameter real slot {} is out of range",
                output.real_parameter_index
            )));
        };
        *real = value.re;
        if let Some(index) = output.imag_parameter_index {
            let imaginary = values.get_mut(index).ok_or_else(|| {
                RusticolError::invalid_argument(format!(
                    "derived model-parameter imaginary slot {index} is out of range"
                ))
            })?;
            *imaginary = value.im;
        }
    }
    Ok(())
}

/// Guard bits of the derived-parameter evaluation above the working precision.
#[cfg(feature = "symbolica-runtime")]
const DERIVED_PARAMETER_GUARD_BITS: u32 = 64;

/// The compiled model's definitions of the derived parameters, by runtime
/// name: `resolved_expression` of every internal parameter and
/// `coupling_expression` of every vertex term as `derived_coupling_{id}`
/// (Symbolica canonical strings), with the model's symbol namespace.
#[cfg(feature = "symbolica-runtime")]
pub(super) struct CompiledModelDefinitions {
    namespace: String,
    definitions: BTreeMap<String, String>,
}

#[cfg(feature = "symbolica-runtime")]
impl CompiledModelDefinitions {
    pub(super) fn load(artifact: &VerifiedArtifact) -> RusticolResult<Self> {
        let unavailable = |reason: &str| {
            RusticolError::unsupported_precision(format!(
                "derived model parameters at the working precision need the artifact's compiled model: {reason}"
            ))
        };
        let records: Vec<_> = artifact
            .manifest()
            .payloads
            .iter()
            .filter(|payload| payload.role == PayloadRole::CompiledModel)
            .collect();
        let [record] = records.as_slice() else {
            return Err(unavailable(
                "the artifact has no single compiled-model payload",
            ));
        };
        let model: Value = serde_json::from_slice(&artifact.read_payload(&record.path)?)
            .map_err(|error| RusticolError::artifact(format!("compiled model: {error}")))?;
        let ir = &model["ir"];
        let name = ir["name"]
            .as_str()
            .ok_or_else(|| unavailable("the compiled model has no name"))?;
        let text = |value: &Value, what: &str| {
            value
                .as_str()
                .map(str::to_owned)
                .ok_or_else(|| RusticolError::artifact(format!("compiled model: invalid {what}")))
        };
        let mut definitions = BTreeMap::new();
        for parameter in ir["parameters"].as_array().into_iter().flatten() {
            if parameter["nature"].as_str() == Some("internal") {
                definitions.insert(
                    text(&parameter["name"], "parameter name")?,
                    text(&parameter["resolved_expression"], "resolved expression")?,
                );
            }
        }
        for term in ir["vertex_terms"].as_array().into_iter().flatten() {
            let id = match &term["id"] {
                Value::String(id) => id.clone(),
                Value::Number(id) => id.to_string(),
                _ => {
                    return Err(RusticolError::artifact(
                        "compiled model: invalid vertex-term id",
                    ));
                }
            };
            definitions.insert(
                format!("derived_coupling_{id}"),
                text(&term["coupling_expression"], "coupling expression")?,
            );
        }
        Ok(Self {
            namespace: format!("model_{}", safe_symbol_part(name.trim())),
            definitions,
        })
    }
}

/// `symbols._safe_text_part`: a Symbolica-safe name part, `readable_digest`
/// (the first 12 hex digits of the text's sha256) unless already an
/// identifier.
#[cfg(feature = "symbolica-runtime")]
fn safe_symbol_part(text: &str) -> String {
    let identifier = text
        .chars()
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && text
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_');
    if identifier {
        return text.to_owned();
    }
    // re.sub(r"[^A-Za-z0-9_]+", "_", text): one `_` per run of other characters.
    let mut readable = String::new();
    let mut in_run = false;
    for character in text.chars() {
        if character.is_ascii_alphanumeric() || character == '_' {
            readable.push(character);
            in_run = false;
        } else if !in_run {
            readable.push('_');
            in_run = true;
        }
    }
    let mut readable = readable.trim_matches('_').to_owned();
    if readable.is_empty() {
        readable = "unnamed".into();
    }
    if readable.starts_with(|first: char| first.is_ascii_digit()) {
        readable.insert_str(0, "n_");
    }
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(text.as_bytes());
    let hex: String = digest
        .iter()
        .take(6)
        .map(|byte| format!("{byte:02x}"))
        .collect();
    format!("{readable}_{hex}")
}

/// `symbols._safe_text_path`: every `::`-separated part made safe.
#[cfg(feature = "symbolica-runtime")]
fn safe_symbol_path(text: &str) -> String {
    text.split("::")
        .filter(|part| !part.is_empty())
        .map(|part| safe_symbol_part(part.trim()))
        .collect::<Vec<_>>()
        .join("::")
}

/// (real slot, imaginary slot, value) of one derived model parameter.
#[cfg(feature = "symbolica-runtime")]
type DerivedParameterValue = (usize, Option<usize>, Complex<Float>);

/// The derived model parameters of one execution runtime at the working
/// precision, for the double-double and arbitrary-precision paths.
///
/// The artifact's binary64 evaluator lowers Pi to its binary64 value when it
/// is built (`compiler_records._replace_evaluator_constants`), so every
/// derived parameter (UFO `G = 2 sqrt(Pi aS)`, the `GC_*` couplings, derived
/// masses) would carry binary64 accuracy. The compiled model keeps the exact
/// definitions; they are evaluated here from the external parameters'
/// binary64 values, promoted exactly, as the Python exact executor does
/// (`symbolica_exact._derive_model_parameters`).
#[cfg(feature = "symbolica-runtime")]
pub(super) struct ExactModelParameterDerivation {
    /// Each external parameter's symbol with its real and imaginary slots.
    inputs: Vec<(Atom, Option<usize>, Option<usize>)>,
    /// Each derived output's definition with its real and imaginary slots.
    outputs: Vec<(Atom, usize, Option<usize>)>,
}

#[cfg(feature = "symbolica-runtime")]
impl ExactModelParameterDerivation {
    fn new(
        model: &CompiledModelDefinitions,
        parameters: &[GenericRuntimeModelParameterManifest],
        outputs: &[GenericDerivedParameterOutputManifest],
    ) -> RusticolResult<Self> {
        let _unlock = crate::licence::unlock();
        // Model expressions may call pyAmpliCol's scalar predicates.
        crate::source_scalar::register_scalar_primitives();
        let parse = |text: &str| {
            Atom::parse(text, "python", ParseSettings::default()).map_err(|error| {
                RusticolError::artifact(format!(
                    "compiled model: cannot parse a parameter definition: {error}"
                ))
            })
        };
        let mut slots: Vec<(String, [Option<usize>; 2])> = Vec::new();
        for parameter in parameters.iter().filter(|parameter| {
            matches!(
                parameter.kind.as_str(),
                "external_parameter" | "external_parameter_component"
            )
        }) {
            let name = parameter.runtime_name.as_deref().unwrap_or(&parameter.name);
            let component = usize::from(parameter.complex_component.as_deref() == Some("imag"));
            let position = match slots.iter().position(|(known, _)| known == name) {
                Some(position) => position,
                None => {
                    slots.push((name.to_owned(), [None, None]));
                    slots.len() - 1
                }
            };
            slots[position].1[component] = Some(parameter.parameter_index);
        }
        let inputs = slots
            .into_iter()
            .map(|(name, [real, imag])| {
                let symbol = parse(&format!("{}::{}", model.namespace, safe_symbol_path(&name)))?;
                Ok((symbol, real, imag))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let outputs = outputs
            .iter()
            .map(|output| {
                let definition = model.definitions.get(&output.runtime_name).ok_or_else(|| {
                    RusticolError::unsupported_precision(format!(
                        "derived model parameter {:?} has no exact definition in the compiled model",
                        output.runtime_name
                    ))
                })?;
                Ok((
                    parse(definition)?,
                    output.real_parameter_index,
                    output.imag_parameter_index,
                ))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        Ok(Self { inputs, outputs })
    }

    /// (real slot, imaginary slot, value at `bits` bits) of every derived output.
    fn evaluate(&self, values: &[f64], bits: u32) -> RusticolResult<Vec<DerivedParameterValue>> {
        let _unlock = crate::licence::unlock();
        let component = |index: Option<usize>| -> RusticolResult<Float> {
            let value = match index {
                Some(index) => *values.get(index).ok_or_else(|| {
                    RusticolError::invalid_argument(format!(
                        "model-parameter slot {index} is out of range"
                    ))
                })?,
                None => 0.0,
            };
            Ok(Float::with_val(bits, value))
        };
        let mut map = std::collections::HashMap::new();
        for (symbol, real, imag) in &self.inputs {
            map.insert(
                symbol.clone(),
                Complex::new(component(*real)?, component(*imag)?),
            );
        }
        self.outputs
            .iter()
            .map(|(definition, real, imag)| {
                let value = definition.evaluate_with_prec(&map, bits).map_err(|error| {
                    RusticolError::invalid_argument(format!(
                        "derived model parameter at the working precision: {error}"
                    ))
                })?;
                Ok((*real, *imag, value))
            })
            .collect()
    }
}

#[cfg(feature = "symbolica-runtime")]
impl ExecutionRuntime {
    /// Whether a derived-parameter evaluator of this runtime or a nested one
    /// lacks its exact definitions.
    pub(super) fn needs_exact_model_parameters(&self) -> bool {
        self.model_parameter_evaluator
            .as_ref()
            .is_some_and(|evaluator| evaluator.exact.is_none())
            || self
                .helicity_sum_runtime
                .as_deref()
                .is_some_and(Self::needs_exact_model_parameters)
            || self
                .helicity_selector_runtimes
                .iter()
                .any(|runtime| runtime.needs_exact_model_parameters())
            || self
                .color_selector_runtimes
                .values()
                .any(|runtime| runtime.needs_exact_model_parameters())
    }

    pub(super) fn attach_exact_model_parameters(
        &mut self,
        model: &CompiledModelDefinitions,
    ) -> RusticolResult<()> {
        if let Some(evaluator) = self.model_parameter_evaluator.as_mut()
            && evaluator.exact.is_none()
        {
            evaluator.exact = Some(Arc::new(ExactModelParameterDerivation::new(
                model,
                &self.model_parameters,
                &evaluator.outputs,
            )?));
        }
        if let Some(runtime) = self.helicity_sum_runtime.as_mut() {
            runtime.attach_exact_model_parameters(model)?;
        }
        for runtime in &mut self.helicity_selector_runtimes {
            runtime.attach_exact_model_parameters(model)?;
        }
        for runtime in self.color_selector_runtimes.values_mut() {
            runtime.attach_exact_model_parameters(model)?;
        }
        Ok(())
    }

    /// The model-parameter values and particle masses of the generic
    /// (double-double and arbitrary-precision) paths: binary64 values promoted
    /// exactly, the derived parameters, and the masses that follow them, at
    /// the working precision ([`ExactModelParameterDerivation`]).
    pub(super) fn generic_model_parameters<T>(
        &self,
        binary_precision: Option<u32>,
    ) -> RusticolResult<(Vec<T>, BTreeMap<i32, T>)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        let mut values: Vec<T> = self
            .model_parameter_values_f64
            .iter()
            .map(|value| T::from_f64_constant(*value, binary_precision))
            .collect();
        let mut masses = masses_at_precision::<T>(&self.particle_masses, binary_precision);
        let Some(evaluator) = self.model_parameter_evaluator.as_ref() else {
            return Ok((values, masses));
        };
        let exact = evaluator.exact.as_ref().ok_or_else(|| {
            RusticolError::unsupported_precision(
                "derived model parameters at the working precision are not loaded for this runtime",
            )
        })?;
        let bits = binary_precision.unwrap_or(106) + DERIVED_PARAMETER_GUARD_BITS;
        let mut derived = BTreeSet::new();
        for (real, imag, value) in exact.evaluate(&self.model_parameter_values_f64, bits)? {
            for (slot, part) in
                std::iter::once((real, &value.re)).chain(imag.map(|imag| (imag, &value.im)))
            {
                let target = values.get_mut(slot).ok_or_else(|| {
                    RusticolError::invalid_argument(format!(
                        "derived model-parameter slot {slot} is out of range"
                    ))
                })?;
                *target = T::from_float(part, binary_precision);
                derived.insert(slot);
            }
        }
        for (pdg, slot) in self.particle_mass_slots() {
            if derived.contains(&slot) {
                masses.insert(pdg, values[slot].clone());
            }
        }
        Ok((values, masses))
    }

    /// The (real, imaginary) value of one logical model parameter in the
    /// generic paths at `binary_precision` (`None`: double-double), taken
    /// from [`Self::generic_model_parameters`] of the runtime that evaluates.
    pub(super) fn generic_model_parameter_value<T>(
        &self,
        name: &str,
        binary_precision: Option<u32>,
    ) -> RusticolResult<(T, T)>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        if let Some(sum_runtime) = self.helicity_sum_runtime.as_deref() {
            return sum_runtime.generic_model_parameter_value(name, binary_precision);
        }
        let (values, _) = self.generic_model_parameters::<T>(binary_precision)?;
        let mut real = None;
        let mut imaginary = T::from_f64_constant(0.0, binary_precision);
        for parameter in &self.model_parameters {
            if parameter.runtime_name.as_deref().unwrap_or(&parameter.name) != name {
                continue;
            }
            let value = values
                .get(parameter.parameter_index)
                .cloned()
                .ok_or_else(|| RusticolError::integrity("model-parameter value slot is absent"))?;
            match parameter.complex_component.as_deref() {
                Some("imag") => imaginary = value,
                Some("real") | None => real = Some(value),
                Some(_) => return Err(RusticolError::integrity("invalid parameter component")),
            }
        }
        real.map(|value| (value, imaginary)).ok_or_else(|| {
            RusticolError::model_parameter(format!("model parameter {name:?} has no runtime value"))
        })
    }

    /// The mass of particle `pdg` that the generic paths give its external
    /// wavefunctions at `binary_precision` (`None`: double-double), from
    /// [`Self::generic_model_parameters`] of the runtime that evaluates.
    pub(super) fn generic_particle_mass<T>(
        &self,
        pdg: i32,
        binary_precision: Option<u32>,
    ) -> RusticolResult<T>
    where
        T: RusticolHighPrecisionNumber,
        Complex<T>: Real + EvaluationDomain,
    {
        if let Some(sum_runtime) = self.helicity_sum_runtime.as_deref() {
            return sum_runtime.generic_particle_mass(pdg, binary_precision);
        }
        let (_, mut masses) = self.generic_model_parameters::<T>(binary_precision)?;
        masses.remove(&pdg).ok_or_else(|| {
            RusticolError::invalid_argument(format!("particle {pdg} has no mass in this runtime"))
        })
    }
}

#[cfg(feature = "symbolica-runtime")]
impl NativeRuntime {
    /// Attaches the compiled model's exact definitions to every derived-parameter
    /// evaluator of this runtime (once; a no-op without derived parameters, as
    /// for the built-in SM), before a double-double or arbitrary-precision call.
    pub(super) fn ensure_exact_model_parameters(&mut self) -> RusticolResult<()> {
        if !self.runtime.needs_exact_model_parameters() {
            return Ok(());
        }
        let artifact = VerifiedArtifact::open(&self.root)?;
        if artifact.manifest().artifact_id != self.artifact_id {
            return Err(RusticolError::integrity(
                "process output changed since the runtime was loaded; reload the runtime before high-precision evaluation",
            ));
        }
        let model = CompiledModelDefinitions::load(&artifact)?;
        self.runtime.attach_exact_model_parameters(&model)
    }
}

fn validate_particle_mass_class_stability(
    previous: &BTreeMap<i32, f64>,
    proposed: &BTreeMap<i32, f64>,
) -> RusticolResult<()> {
    for (pdg, previous_mass) in previous {
        let Some(proposed_mass) = proposed.get(pdg) else {
            continue;
        };
        if (*previous_mass == 0.0) != (*proposed_mass == 0.0) {
            return Err(RusticolError::invalid_argument(format!(
                "model-parameter update changes particle {pdg} from a {} to a {} mass class; regenerate the process artifact for that mass class",
                if *previous_mass == 0.0 {
                    "massless"
                } else {
                    "massive"
                },
                if *proposed_mass == 0.0 {
                    "massless"
                } else {
                    "massive"
                },
            )));
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "symbolica-runtime"))]
mod exact_parameter_tests {
    use super::{safe_symbol_part, safe_symbol_path};

    #[test]
    fn symbol_names_follow_the_python_model_registry() {
        // Expected values from pyamplicol._internal.physics.symbols.
        for (text, part, path) in [
            ("sm", "sm", "sm"),
            ("aS", "aS", "aS"),
            (
                "loop_sm-no_b_mass",
                "loop_sm_no_b_mass_b773d6c57928",
                "loop_sm_no_b_mass_b773d6c57928",
            ),
            ("a_-b", "a__b_56f89b8480c1", "a__b_56f89b8480c1"),
            ("2HDM", "n_2HDM_1cb6a6a0ecf2", "n_2HDM_1cb6a6a0ecf2"),
            ("--", "unnamed_d8156bae0c42", "unnamed_d8156bae0c42"),
            ("\u{dc}", "unnamed_913d7314efae", "unnamed_913d7314efae"),
            (
                "MDL::x::y-z",
                "MDL_x_y_z_3a79e3a86ef8",
                "MDL::x::y_z_2fd2bd70492c",
            ),
        ] {
            assert_eq!(safe_symbol_part(text), part, "{text:?}");
            assert_eq!(safe_symbol_path(text), path, "{text:?}");
        }
    }
}

#[cfg(test)]
mod parameter_json_tests {
    use super::parse_complex_parameter_overrides;
    use std::path::Path;

    #[test]
    fn parameter_json_accepts_real_scalars_and_complex_pairs() {
        let parsed = parse_complex_parameter_overrides(
            r#"{"aS": 0.117, "complex": [2.5, -0.25]}"#,
            Path::new("parameters.json"),
        )
        .expect("parameter card should parse");

        assert_eq!(parsed.get("aS"), Some(&(0.117, 0.0)));
        assert_eq!(parsed.get("complex"), Some(&(2.5, -0.25)));
    }

    #[test]
    fn parameter_json_rejects_non_numeric_values() {
        let error =
            parse_complex_parameter_overrides(r#"{"aS": "0.117"}"#, Path::new("parameters.json"))
                .expect_err("string parameter must fail");

        assert!(
            error
                .to_string()
                .contains("must be a real number or [real, imaginary]")
        );
    }
}
