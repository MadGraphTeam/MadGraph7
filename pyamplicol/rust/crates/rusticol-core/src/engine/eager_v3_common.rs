// SPDX-License-Identifier: 0BSD

//! Checked construction of the common f64 state used by eager plan-v3.
//!
//! This adapter deliberately reads the compact typed plan, retained columns,
//! and canonical IR directly. It does not recreate the expanded
//! `runtime_schema` representation.

use super::eager_v3_decode::{
    DecodedEagerPrimitiveColumn, DecodedEagerRetainedTable, DecodedEagerRuntimeV3,
};
use super::eager_v3_manifest::EagerV3ExecutionManifest;
use super::*;
use serde::Deserialize;
use serde_json::{Value, json};

const MISSING_I32: i32 = i32::MIN;

/// Validated fields that differ between eager artifacts while the compiled
/// lane state remains empty.
pub(super) struct EagerV3CommonParts {
    process: String,
    key: String,
    color_accuracy: String,
    external_pdg_order: Vec<i32>,
    external_count: usize,
    parameter_count: usize,
    value_parameter_count: usize,
    momentum_parameter_count: usize,
    current_count: usize,
    source_count: usize,
    interaction_count: usize,
    stage_count: usize,
    amplitude_output_count: usize,
    sources: Vec<GenericSourceRecordManifest>,
    momentum_slots: Vec<GenericMomentumSlotManifest>,
    external_is_initial: Vec<bool>,
    particle_masses: BTreeMap<i32, f64>,
    particle_mass_parameter_names: BTreeMap<i32, String>,
    normalization_factor: f64,
    normalization_color_factor: f64,
    normalization_average_factor: f64,
    normalization_identical_factor: f64,
    normalization_qcd_coupling_power: usize,
    normalization_electroweak_coupling_power: usize,
    model_parameters: Vec<GenericRuntimeModelParameterManifest>,
    model_parameter_name_to_index: BTreeMap<String, usize>,
    model_parameter_runtime_slots: BTreeMap<String, RuntimeParameterSlots>,
    model_parameter_values_f64: Vec<f64>,
    physics: Arc<PhysicsRuntime>,
}

/// One-line integration entry point for the plan-v3 loader.
pub(super) fn build_eager_v3_common_runtime(
    decoded: &DecodedEagerRuntimeV3,
    manifest: &EagerV3ExecutionManifest,
    physics: ProcessPhysicsV1,
) -> RusticolResult<ExecutionRuntime> {
    EagerV3CommonParts::checked(decoded, manifest, physics).map(EagerV3CommonParts::into_runtime)
}

impl EagerV3CommonParts {
    fn checked(
        decoded: &DecodedEagerRuntimeV3,
        manifest: &EagerV3ExecutionManifest,
        physics: ProcessPhysicsV1,
    ) -> RusticolResult<Self> {
        validate_identity(decoded, manifest, &physics)?;
        let retained = RetainedTables::new(&decoded.retained_tables);
        let metadata = retained.table("metadata")?;
        let normalization_ir_id = retained.scalar_u32(metadata, "normalization_ir_id")?;
        let normalization: CanonicalNormalization =
            canonical_ir(decoded, normalization_ir_id, "runtime normalization")?;
        normalization.validate()?;

        let model_parameters = build_model_parameters(decoded, &retained)?;
        validate_physics_parameters(&model_parameters, &physics)?;
        let model_parameter_values_f64 = model_parameters
            .iter()
            .map(|parameter| parameter.default)
            .collect::<Vec<_>>();
        let model_parameter_name_to_index = model_parameters
            .iter()
            .map(|parameter| (parameter.name.clone(), parameter.parameter_index))
            .collect::<BTreeMap<_, _>>();
        if model_parameter_name_to_index.len() != model_parameters.len() {
            return Err(integrity("eager model-parameter names are not unique"));
        }
        let model_parameter_runtime_slots = runtime_parameter_slots(&model_parameters)?;
        let (mut particle_masses, mut particle_mass_parameter_names) =
            direct_particle_masses(&model_parameters)?;

        let momentum_slots = build_momentum_slots(decoded, physics.external_particles.len())?;
        let sources = build_sources(
            decoded,
            &retained,
            &physics,
            &model_parameter_runtime_slots,
            &model_parameter_values_f64,
            &mut particle_masses,
            &mut particle_mass_parameter_names,
        )?;
        let value_parameter_count = component_count(&decoded.values, "value")?;
        let momentum_parameter_count = component_count(&decoded.momenta, "momentum")?;
        let parameter_count = value_parameter_count
            .checked_add(momentum_parameter_count)
            .and_then(|count| count.checked_add(model_parameters.len()))
            .ok_or_else(|| RusticolError::artifact("eager common parameter count overflows"))?;

        // Contracted-color reductions already carry the color contraction.
        let color_factor = if physics.color_accuracy.as_str() == "lc" {
            normalization.color_factor
        } else {
            1.0
        };
        let normalization_factor = color_factor * normalization.global_coupling_factor
            / (normalization.average_factor * normalization.identical_factor);
        if !normalization_factor.is_finite() {
            return Err(integrity("eager runtime normalization is not finite"));
        }
        let physics = Arc::new(PhysicsRuntime::new(physics)?);

        Ok(Self {
            process: manifest.process.clone(),
            key: manifest.key.clone(),
            color_accuracy: manifest.color_accuracy.clone(),
            external_pdg_order: manifest.external_pdg_order.clone(),
            external_count: manifest.external_pdg_order.len(),
            parameter_count,
            value_parameter_count,
            momentum_parameter_count,
            current_count: decoded.currents.len(),
            source_count: decoded.sources.len(),
            interaction_count: usize_count(
                manifest.dag_summary.interaction_count,
                "interaction count",
            )?,
            stage_count: decoded.stages.len(),
            amplitude_output_count: usize::try_from(decoded.dimensions.amplitude_count)
                .map_err(|_| RusticolError::artifact("eager amplitude count exceeds usize"))?,
            sources,
            momentum_slots,
            external_is_initial: physics
                .manifest
                .external_particles
                .iter()
                .map(|particle| particle.role == crate::ParticleRole::Initial)
                .collect(),
            particle_masses,
            particle_mass_parameter_names,
            normalization_factor,
            normalization_color_factor: color_factor,
            normalization_average_factor: normalization.average_factor,
            normalization_identical_factor: normalization.identical_factor,
            normalization_qcd_coupling_power: normalization.qcd_coupling_power,
            normalization_electroweak_coupling_power: normalization.electroweak_coupling_power,
            model_parameters,
            model_parameter_name_to_index,
            model_parameter_runtime_slots,
            model_parameter_values_f64,
            physics,
        })
    }

    fn into_runtime(self) -> ExecutionRuntime {
        ExecutionRuntime {
            process: self.process,
            key: self.key,
            color_accuracy: self.color_accuracy,
            external_pdg_order: self.external_pdg_order,
            external_count: self.external_count,
            parameter_count: self.parameter_count,
            value_parameter_count: self.value_parameter_count,
            momentum_parameter_count: self.momentum_parameter_count,
            current_count: self.current_count,
            source_count: self.source_count,
            interaction_count: self.interaction_count,
            stage_count: self.stage_count,
            amplitude_output_count: self.amplitude_output_count,
            lc_topology_replay_enabled: false,
            lc_topology_replay_mappings: Arc::new(Vec::new()),
            lc_topology_replay_public_mappings: Vec::new(),
            lc_topology_replay_routes: Vec::new(),
            lc_topology_replay_materialized_sector_ids: BTreeSet::new(),
            lc_resolved_replay_plan: None,
            lc_resolved_replay_selection_cache: None,
            helicity_recurrence: None,
            compiled_helicity_execution_plan: None,
            compiled_color_execution_plan: None,
            helicity_sum_runtime: None,
            helicity_selector_runtimes: Vec::new(),
            helicity_selector_runtime_schedule_modes: Vec::new(),
            helicity_selector_lane_by_domain: BTreeMap::new(),
            color_selector_runtimes: BTreeMap::new(),
            runtime_unavailable_message: None,
            sources: self.sources,
            momentum_slots: self.momentum_slots,
            external_is_initial: self.external_is_initial,
            particle_masses: self.particle_masses,
            particle_mass_parameter_names: self.particle_mass_parameter_names,
            normalization_factor: self.normalization_factor,
            normalization_color_factor: self.normalization_color_factor,
            normalization_average_factor: self.normalization_average_factor,
            normalization_identical_factor: self.normalization_identical_factor,
            normalization_qcd_coupling_power: self.normalization_qcd_coupling_power,
            normalization_electroweak_coupling_power: self.normalization_electroweak_coupling_power,
            model_parameters: self.model_parameters,
            model_parameter_name_to_index: self.model_parameter_name_to_index,
            model_parameter_runtime_slots: self.model_parameter_runtime_slots,
            model_parameter_values_f64: self.model_parameter_values_f64,
            model_parameter_evaluator: None,
            physics_reduction_override: None,
            physics: Some(self.physics),
            stages: None,
            amplitude_stage: None,
            state_scratch_f64: Vec::new(),
            state_scratch_f64_requires_clear: false,
            values_scratch_f64: Vec::new(),
        }
    }
}

fn validate_identity(
    decoded: &DecodedEagerRuntimeV3,
    manifest: &EagerV3ExecutionManifest,
    physics: &ProcessPhysicsV1,
) -> RusticolResult<()> {
    physics.validate()?;
    if decoded.process_key.as_ref() != manifest.key
        || physics.process_id != manifest.key
        || physics.process != manifest.process
        || physics.color_accuracy.as_str() != manifest.color_accuracy
        || physics.external_particles.len() != manifest.external_pdg_order.len()
        || physics
            .external_particles
            .iter()
            .map(|particle| particle.pdg)
            .ne(manifest.external_pdg_order.iter().copied())
    {
        return Err(integrity(
            "eager plan-v3 identity does not match its manifest and resolved physics",
        ));
    }
    if decoded.currents.len() != usize_count(manifest.dag_summary.current_count, "current count")?
        || decoded.sources.len() != usize_count(manifest.dag_summary.source_count, "source count")?
        || decoded.closures.is_empty()
    {
        return Err(integrity(
            "eager plan-v3 decoded counts do not match its DAG summary",
        ));
    }
    Ok(())
}

fn build_model_parameters(
    decoded: &DecodedEagerRuntimeV3,
    retained: &RetainedTables<'_>,
) -> RusticolResult<Vec<GenericRuntimeModelParameterManifest>> {
    let table = retained.table("model_parameters")?;
    let pdgs = retained.i32_column(table, "pdg", 1)?;
    if pdgs.len() != decoded.parameters.len() {
        return Err(integrity(
            "retained model-parameter PDGs have the wrong row count",
        ));
    }
    decoded
        .parameters
        .iter()
        .zip(pdgs)
        .map(|(row, pdg)| {
            let index = usize::try_from(row.parameter_id)
                .map_err(|_| RusticolError::artifact("parameter ID exceeds usize"))?;
            let factor = decoded
                .exact_factors
                .get(row.default_factor_id as usize)
                .ok_or_else(|| integrity("parameter default factor is absent"))?;
            let default = f64::from_bits(factor.real_bits);
            let imaginary = f64::from_bits(factor.imaginary_bits);
            if !default.is_finite() || imaginary != 0.0 {
                return Err(integrity(
                    "eager model-parameter default is not finite and real",
                ));
            }
            Ok(GenericRuntimeModelParameterManifest {
                name: string(decoded, row.name_string_id, "parameter name")?.to_owned(),
                kind: string(decoded, row.kind_string_id, "parameter kind")?.to_owned(),
                parameter_index: index,
                default,
                pdg: (*pdg != MISSING_I32).then_some(*pdg),
                runtime_name: optional_string(decoded, row.runtime_name_string_id)?
                    .map(str::to_owned),
                complex_component: match row.complex_component {
                    -1 => None,
                    0 => Some("real".to_owned()),
                    1 => Some("imag".to_owned()),
                    _ => return Err(integrity("invalid eager parameter complex component")),
                },
            })
        })
        .collect()
}

fn runtime_parameter_slots(
    parameters: &[GenericRuntimeModelParameterManifest],
) -> RusticolResult<BTreeMap<String, RuntimeParameterSlots>> {
    let mut result = BTreeMap::new();
    let mut complex = BTreeMap::<String, (Option<usize>, Option<usize>)>::new();
    for parameter in parameters {
        if parameter.kind == "derived_parameter_component" {
            continue;
        }
        if let Some(name) = &parameter.runtime_name {
            let slots = complex.entry(name.clone()).or_default();
            let slot = match parameter.complex_component.as_deref() {
                Some("real") => &mut slots.0,
                Some("imag") => &mut slots.1,
                _ => {
                    return Err(integrity(
                        "complex eager parameter lacks component metadata",
                    ));
                }
            };
            if slot.replace(parameter.parameter_index).is_some() {
                return Err(integrity("duplicate eager complex parameter component"));
            }
        } else if result
            .insert(
                parameter.name.clone(),
                RuntimeParameterSlots {
                    real: parameter.parameter_index,
                    imaginary: None,
                },
            )
            .is_some()
        {
            return Err(integrity("duplicate eager scalar parameter"));
        }
    }
    for (name, (real, imaginary)) in complex {
        let real = real.ok_or_else(|| integrity("complex eager parameter lacks a real slot"))?;
        if result
            .insert(name, RuntimeParameterSlots { real, imaginary })
            .is_some()
        {
            return Err(integrity("conflicting eager logical parameter"));
        }
    }
    Ok(result)
}

fn direct_particle_masses(
    parameters: &[GenericRuntimeModelParameterManifest],
) -> RusticolResult<(BTreeMap<i32, f64>, BTreeMap<i32, String>)> {
    let mut masses = BTreeMap::new();
    let mut names = BTreeMap::new();
    for parameter in parameters {
        if parameter.kind != "particle_mass" {
            continue;
        }
        let pdg = parameter
            .pdg
            .ok_or_else(|| integrity("particle-mass parameter lacks a retained PDG"))?;
        if parameter.default < 0.0
            || masses.insert(pdg, parameter.default).is_some()
            || names.insert(pdg, parameter.name.clone()).is_some()
        {
            return Err(integrity("invalid or duplicate particle-mass parameter"));
        }
    }
    Ok((masses, names))
}

#[allow(clippy::too_many_arguments)]
fn build_sources(
    decoded: &DecodedEagerRuntimeV3,
    retained: &RetainedTables<'_>,
    physics: &ProcessPhysicsV1,
    runtime_slots: &BTreeMap<String, RuntimeParameterSlots>,
    parameter_values: &[f64],
    particle_masses: &mut BTreeMap<i32, f64>,
    mass_names: &mut BTreeMap<i32, String>,
) -> RusticolResult<Vec<GenericSourceRecordManifest>> {
    let currents = retained.table("currents")?;
    let particle_ids = retained.i32_column(currents, "particle_id", 1)?;
    let source_labels = retained.u32_column(currents, "source_leg_label", 1)?;
    let source_helicities = retained.i32_column(currents, "source_helicity", 1)?;
    let chiralities = retained.i32_column(currents, "chirality", 1)?;
    let spin_state_ids = retained.u32_column(currents, "spin_state_sequence_id", 1)?;
    let ancestry_ids = retained.u32_column(currents, "helicity_ancestry_bitset_id", 1)?;
    for column in [
        particle_ids.len(),
        source_labels.len(),
        source_helicities.len(),
        chiralities.len(),
        spin_state_ids.len(),
        ancestry_ids.len(),
    ] {
        if column != decoded.currents.len() {
            return Err(integrity(
                "retained current metadata has the wrong row count",
            ));
        }
    }

    let mut source_parameter_start = 0usize;
    let mut source_ir_by_momentum_slot = BTreeMap::new();
    let mut source_ir_by_canonical_id = BTreeMap::new();
    decoded
        .sources
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let current_index = row.current_id as usize;
            let current = decoded
                .currents
                .get(current_index)
                .ok_or_else(|| integrity("eager source current is absent"))?;
            let value = decoded
                .values
                .get(row.value_slot_id as usize)
                .ok_or_else(|| integrity("eager source value slot is absent"))?;
            let particle = physics
                .external_particles
                .get(row.input_momentum_slot as usize)
                .ok_or_else(|| integrity("eager source external particle is absent"))?;
            let source_ir: GenericSourceIrManifest =
                canonical_ir(decoded, row.source_ir_id, "source IR")?;
            let applied_crossing: GenericCrossingIrManifest =
                canonical_ir(decoded, row.crossing_ir_id, "source crossing IR")?;
            let declared = source_ir
                .states
                .get(row.declared_state_index as usize)
                .ok_or_else(|| integrity("eager source declared state is absent"))?;
            let state = declared
                .transformed(&applied_crossing)
                .map_err(|message| integrity(format!("invalid eager source state: {message}")))?;
            let crossing_factor = decoded
                .exact_factors
                .get(row.crossing_factor_id as usize)
                .ok_or_else(|| integrity("eager source crossing factor is absent"))?;
            if crossing_factor.real_bits != applied_crossing.phase[0].to_bits()
                || crossing_factor.imaginary_bits != applied_crossing.phase[1].to_bits()
            {
                return Err(integrity(
                    "eager source crossing factor disagrees with its IR",
                ));
            }
            let expected_outgoing = match particle.role {
                crate::ParticleRole::Initial => source_ir.identity.anti_pdg_label,
                crate::ParticleRole::Final => source_ir.identity.pdg_label,
            };
            if particle.label != row.external_label as usize
                || particle.momentum_slot != row.input_momentum_slot as usize
                || particle.pdg != expected_outgoing
                || particle_ids[current_index] != source_ir.identity.pdg_label
                || source_labels[current_index] != row.external_label
                || source_helicities[current_index] != state.helicity
                || chiralities[current_index] != state.chirality
                || current.component_count as usize != source_ir.component_dimension
                || value.component_count != current.component_count
                || value.current_id != row.current_id
            {
                return Err(integrity(
                    "eager source metadata is internally inconsistent",
                ));
            }
            let retained_spin = i32_sequence(decoded, spin_state_ids[current_index])?;
            let expected_spin = match &state.spin_state {
                GenericSourceSpinStateManifest::Scalar(value) => vec![*value],
                GenericSourceSpinStateManifest::Components(values) => values.clone(),
            };
            if retained_spin != expected_spin {
                return Err(integrity(
                    "eager source spin state disagrees with retained metadata",
                ));
            }
            bind_source_mass(
                &source_ir,
                runtime_slots,
                parameter_values,
                particle_masses,
                mass_names,
            )?;
            let dimension = source_ir.component_dimension;
            let current_start = usize_count(current.component_start, "source current start")?;
            let value_start = usize_count(value.component_start, "source value start")?;
            let source_parameter_stop = source_parameter_start
                .checked_add(dimension)
                .ok_or_else(|| RusticolError::artifact("source parameter layout overflows"))?;
            let side = match particle.role {
                crate::ParticleRole::Initial => "initial",
                crate::ParticleRole::Final => "final",
            };
            let source = GenericSourceRecordManifest {
                source_id: index,
                current_id: current_index,
                current_component_start: current_start,
                current_component_stop: current_start + dimension,
                value_slot: GenericValueSlotRefManifest {
                    value_slot_id: row.value_slot_id as usize,
                    current_id: current_index,
                    variant: "source".to_owned(),
                    component_start: value_start,
                    component_stop: value_start + dimension,
                    dimension,
                },
                source_parameter_start,
                source_parameter_stop,
                leg_label: row.external_label as usize,
                input_momentum_slot: row.input_momentum_slot as usize,
                side: side.to_owned(),
                crossing: applied_crossing
                    .momentum_transform
                    .legacy_projection()
                    .to_owned(),
                physical_pdg: particle.pdg,
                outgoing_pdg: source_ir.identity.pdg_label,
                particle_id: source_ir.identity.pdg_label,
                anti_particle_id: source_ir.identity.anti_pdg_label,
                source_kind: "external-wavefunction".to_owned(),
                wavefunction_kind: source_ir.wavefunction_family.as_str().to_owned(),
                source_orientation: source_ir.identity.orientation,
                source_basis: source_ir.basis.clone(),
                source_ir,
                applied_crossing,
                source_helicity: state.helicity,
                chirality: state.chirality,
                spin_state: spin_state_value(&state.spin_state),
                dimension,
                helicity_ancestry: Value::from(bitset_u64(decoded, ancestry_ids[current_index])?),
                color_state: json!({"accuracy": manifest_color_accuracy(physics)}),
            };
            super::validation::validate_source_wavefunction_metadata(index, &source)?;
            super::validation::validate_consistent_source_ir(
                &mut source_ir_by_momentum_slot,
                &mut source_ir_by_canonical_id,
                index,
                &source,
            )?;
            source_parameter_start = source_parameter_stop;
            Ok(source)
        })
        .collect()
}

fn bind_source_mass(
    source_ir: &GenericSourceIrManifest,
    runtime_slots: &BTreeMap<String, RuntimeParameterSlots>,
    parameter_values: &[f64],
    particle_masses: &mut BTreeMap<i32, f64>,
    mass_names: &mut BTreeMap<i32, String>,
) -> RusticolResult<()> {
    let Some(name) = source_ir.mass_parameter.as_ref() else {
        return Ok(());
    };
    let slots = runtime_slots.get(name).ok_or_else(|| {
        integrity(format!(
            "massive eager source references unbound mass parameter {name:?}"
        ))
    })?;
    if slots.imaginary.is_some() {
        return Err(integrity("eager source mass parameter must be real"));
    }
    let mass = *parameter_values
        .get(slots.real)
        .ok_or_else(|| integrity("eager source mass parameter slot is absent"))?;
    if !mass.is_finite() || mass < 0.0 {
        return Err(integrity("eager source mass parameter is invalid"));
    }
    for pdg in [
        source_ir.identity.pdg_label,
        source_ir.identity.anti_pdg_label,
    ] {
        if particle_masses
            .insert(pdg, mass)
            .is_some_and(|previous| previous.to_bits() != mass.to_bits())
            || mass_names
                .insert(pdg, name.clone())
                .is_some_and(|previous| previous != *name)
        {
            return Err(integrity("eager source has conflicting mass bindings"));
        }
    }
    Ok(())
}

fn build_momentum_slots(
    decoded: &DecodedEagerRuntimeV3,
    external_count: usize,
) -> RusticolResult<Vec<GenericMomentumSlotManifest>> {
    decoded
        .momenta
        .iter()
        .map(|row| {
            if row.component_count != 4 {
                return Err(integrity(
                    "eager momentum slot does not contain four components",
                ));
            }
            let external_labels = bitset_labels(decoded, row.bitset_id, external_count)?;
            if external_labels.is_empty() {
                return Err(integrity("eager momentum slot has an empty external mask"));
            }
            let start = usize_count(row.component_start, "momentum component start")?;
            Ok(GenericMomentumSlotManifest {
                momentum_slot_id: row.momentum_slot_id as usize,
                momentum_mask: bitset_u64(decoded, row.bitset_id)?,
                external_labels,
                component_start: start,
                component_stop: start + 4,
                real_valued: true,
            })
        })
        .collect()
}

fn validate_physics_parameters(
    runtime: &[GenericRuntimeModelParameterManifest],
    physics: &ProcessPhysicsV1,
) -> RusticolResult<()> {
    let mut projected = BTreeMap::<String, (crate::ParameterKind, f64, f64)>::new();
    for parameter in runtime {
        let name = parameter
            .runtime_name
            .as_ref()
            .unwrap_or(&parameter.name)
            .clone();
        let kind = public_parameter_kind(&parameter.kind)?;
        let entry = projected.entry(name).or_insert((kind, 0.0, 0.0));
        if entry.0 != kind {
            return Err(integrity("eager logical parameter has inconsistent kinds"));
        }
        match parameter.complex_component.as_deref() {
            Some("imag") => entry.2 = parameter.default,
            _ => entry.1 = parameter.default,
        }
    }
    if projected.len() != physics.model_parameters.len() {
        return Err(integrity(
            "eager runtime and resolved physics parameter counts disagree",
        ));
    }
    for parameter in &physics.model_parameters {
        let Some((kind, real, imaginary)) = projected.get(&parameter.name) else {
            return Err(integrity(
                "resolved physics parameter is absent from eager runtime",
            ));
        };
        if *kind != parameter.kind
            || real.to_bits() != parameter.default_real.to_bits()
            || imaginary.to_bits() != parameter.default_imaginary.to_bits()
        {
            return Err(integrity(
                "eager runtime parameter disagrees with resolved physics",
            ));
        }
    }
    Ok(())
}

fn public_parameter_kind(kind: &str) -> RusticolResult<crate::ParameterKind> {
    match kind {
        "normalization" => Ok(crate::ParameterKind::Normalization),
        "particle_mass" => Ok(crate::ParameterKind::Mass),
        "particle_width" => Ok(crate::ParameterKind::Width),
        "coupling_component" => Ok(crate::ParameterKind::Coupling),
        "external_parameter" | "external_parameter_component" => Ok(crate::ParameterKind::External),
        "derived_parameter_component" => Ok(crate::ParameterKind::Derived),
        _ => Err(RusticolError::compatibility(format!(
            "unsupported eager model-parameter kind {kind:?}"
        ))),
    }
}

#[derive(Deserialize)]
struct CanonicalNormalization {
    #[serde(default = "one")]
    color_factor: f64,
    #[serde(default = "one")]
    global_coupling_factor: f64,
    #[serde(default = "one")]
    average_factor: f64,
    #[serde(default = "one")]
    identical_factor: f64,
    #[serde(default)]
    qcd_coupling_power: usize,
    #[serde(default)]
    electroweak_coupling_power: usize,
}

impl CanonicalNormalization {
    fn validate(&self) -> RusticolResult<()> {
        if !self.color_factor.is_finite()
            || !self.global_coupling_factor.is_finite()
            || !self.average_factor.is_finite()
            || !self.identical_factor.is_finite()
            || self.average_factor <= 0.0
            || self.identical_factor <= 0.0
        {
            return Err(integrity("invalid eager runtime normalization IR"));
        }
        Ok(())
    }
}

fn one() -> f64 {
    1.0
}

struct RetainedTables<'a> {
    tables: &'a [DecodedEagerRetainedTable],
}

impl<'a> RetainedTables<'a> {
    fn new(tables: &'a [DecodedEagerRetainedTable]) -> Self {
        Self { tables }
    }

    fn table(&self, name: &str) -> RusticolResult<&'a DecodedEagerRetainedTable> {
        let mut matches = self
            .tables
            .iter()
            .filter(|table| table.name.as_ref() == name);
        let table = matches
            .next()
            .ok_or_else(|| integrity(format!("retained eager table {name:?} is absent")))?;
        if matches.next().is_some() {
            return Err(integrity(format!(
                "retained eager table {name:?} is duplicated"
            )));
        }
        Ok(table)
    }

    fn scalar_u32(&self, table: &DecodedEagerRetainedTable, name: &str) -> RusticolResult<u32> {
        let values = self.u32_column(table, name, 1)?;
        if table.row_count != 1 || values.len() != 1 {
            return Err(integrity(format!(
                "retained eager scalar {}.{name} has the wrong shape",
                table.name
            )));
        }
        Ok(values[0])
    }

    fn u32_column(
        &self,
        table: &'a DecodedEagerRetainedTable,
        name: &str,
        elements_per_row: u32,
    ) -> RusticolResult<&'a [u32]> {
        let column = retained_column(table, name, elements_per_row)?;
        match &column.values {
            DecodedEagerPrimitiveColumn::U32(values) => Ok(values),
            _ => Err(integrity(format!(
                "retained eager column {}.{name} is not u32",
                table.name
            ))),
        }
    }

    fn i32_column(
        &self,
        table: &'a DecodedEagerRetainedTable,
        name: &str,
        elements_per_row: u32,
    ) -> RusticolResult<&'a [i32]> {
        let column = retained_column(table, name, elements_per_row)?;
        match &column.values {
            DecodedEagerPrimitiveColumn::I32(values) => Ok(values),
            _ => Err(integrity(format!(
                "retained eager column {}.{name} is not i32",
                table.name
            ))),
        }
    }
}

fn retained_column<'a>(
    table: &'a DecodedEagerRetainedTable,
    name: &str,
    elements_per_row: u32,
) -> RusticolResult<&'a super::eager_v3_decode::DecodedEagerRetainedColumn> {
    let mut matches = table
        .columns
        .iter()
        .filter(|column| column.name.as_ref() == name);
    let column = matches.next().ok_or_else(|| {
        integrity(format!(
            "retained eager column {}.{name} is absent",
            table.name
        ))
    })?;
    if matches.next().is_some() || column.elements_per_row != elements_per_row {
        return Err(integrity(format!(
            "retained eager column {}.{name} has a non-canonical shape",
            table.name
        )));
    }
    Ok(column)
}

fn canonical_ir<T: for<'de> Deserialize<'de>>(
    decoded: &DecodedEagerRuntimeV3,
    id: u32,
    context: &str,
) -> RusticolResult<T> {
    let payload = decoded
        .exact_ir
        .get(id as usize)
        .ok_or_else(|| integrity(format!("{context} references absent canonical IR {id}")))?;
    serde_json::from_str(payload).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse eager {context} canonical IR {id}: {error}"
        ))
    })
}

fn string<'a>(
    decoded: &'a DecodedEagerRuntimeV3,
    id: u32,
    context: &str,
) -> RusticolResult<&'a str> {
    decoded
        .strings
        .get(id as usize)
        .map(AsRef::as_ref)
        .ok_or_else(|| integrity(format!("{context} string {id} is absent")))
}

fn optional_string(decoded: &DecodedEagerRuntimeV3, id: u32) -> RusticolResult<Option<&str>> {
    if id == crate::MISSING_U32 {
        Ok(None)
    } else {
        string(decoded, id, "optional parameter").map(Some)
    }
}

fn i32_sequence(decoded: &DecodedEagerRuntimeV3, id: u32) -> RusticolResult<Vec<i32>> {
    let range = decoded
        .i32_sequence_ranges
        .get(id as usize)
        .ok_or_else(|| integrity("retained current references absent i32 sequence"))?;
    checked_catalog_range(
        &decoded.i32_sequence_values,
        range.start,
        range.count,
        "i32 sequence",
    )
    .map(<[i32]>::to_vec)
}

fn bitset_words(decoded: &DecodedEagerRuntimeV3, id: u32) -> RusticolResult<&[u64]> {
    let range = decoded
        .bitset_ranges
        .get(id as usize)
        .ok_or_else(|| integrity("eager metadata references absent bitset"))?;
    checked_catalog_range(&decoded.bitset_words, range.start, range.count, "bitset")
}

fn bitset_u64(decoded: &DecodedEagerRuntimeV3, id: u32) -> RusticolResult<u64> {
    let words = bitset_words(decoded, id)?;
    if words.iter().skip(1).any(|word| *word != 0) {
        return Err(RusticolError::compatibility(
            "common eager metadata cannot project a bitset wider than 64 bits",
        ));
    }
    Ok(words.first().copied().unwrap_or(0))
}

fn bitset_labels(
    decoded: &DecodedEagerRuntimeV3,
    id: u32,
    external_count: usize,
) -> RusticolResult<Vec<usize>> {
    let mut labels = Vec::new();
    for (word_index, word) in bitset_words(decoded, id)?.iter().copied().enumerate() {
        for bit in 0..64 {
            if word & (1_u64 << bit) == 0 {
                continue;
            }
            let index = word_index
                .checked_mul(64)
                .and_then(|start| start.checked_add(bit))
                .ok_or_else(|| RusticolError::artifact("eager momentum label overflows"))?;
            if index >= external_count {
                return Err(integrity(
                    "eager momentum mask exceeds the external-particle count",
                ));
            }
            labels.push(index + 1);
        }
    }
    Ok(labels)
}

fn checked_catalog_range<'a, T>(
    values: &'a [T],
    start: u64,
    count: u64,
    context: &str,
) -> RusticolResult<&'a [T]> {
    let start = usize_count(start, context)?;
    let count = usize_count(count, context)?;
    let stop = start
        .checked_add(count)
        .ok_or_else(|| RusticolError::artifact(format!("eager {context} range overflows")))?;
    values
        .get(start..stop)
        .ok_or_else(|| integrity(format!("eager {context} range is out of bounds")))
}

fn component_count<T>(rows: &[T], context: &str) -> RusticolResult<usize>
where
    T: ComponentRow,
{
    rows.iter().try_fold(0usize, |total, row| {
        total
            .checked_add(row.component_count() as usize)
            .ok_or_else(|| RusticolError::artifact(format!("eager {context} layout overflows")))
    })
}

trait ComponentRow {
    fn component_count(&self) -> u32;
}

impl ComponentRow for crate::eager_lowering_v3::EagerPlanValueRow {
    fn component_count(&self) -> u32 {
        self.component_count
    }
}

impl ComponentRow for crate::eager_lowering_v3::EagerPlanMomentumRow {
    fn component_count(&self) -> u32 {
        self.component_count
    }
}

fn spin_state_value(state: &GenericSourceSpinStateManifest) -> Value {
    match state {
        GenericSourceSpinStateManifest::Scalar(value) => Value::from(*value),
        GenericSourceSpinStateManifest::Components(values) => json!(values),
    }
}

fn manifest_color_accuracy(physics: &ProcessPhysicsV1) -> &'static str {
    match physics.color_accuracy {
        crate::ColorAccuracy::Lc => "lc",
        crate::ColorAccuracy::Nlc => "nlc",
        crate::ColorAccuracy::Full => "full",
    }
}

fn usize_count(value: u64, context: &str) -> RusticolResult<usize> {
    usize::try_from(value)
        .map_err(|_| RusticolError::artifact(format!("eager {context} exceeds usize")))
}

fn integrity(message: impl Into<String>) -> RusticolError {
    RusticolError::integrity(message.into())
}
