// SPDX-License-Identifier: 0BSD

use super::*;

pub(super) fn build_lc_topology_replay_mappings(
    replay: Option<&LcTopologyReplayManifest>,
) -> RusticolResult<LcTopologyReplayData> {
    let Some(replay) = replay else {
        return Ok((Vec::new(), Vec::new()));
    };
    if !replay.enabled {
        return Ok((Vec::new(), Vec::new()));
    }
    if replay.mode != "external-label-permutation" {
        return Err(RusticolError::invalid_argument(format!(
            "unsupported LC topology replay mode {:?}",
            replay.mode
        )));
    }
    let mut mappings = Vec::new();
    let mut weights = Vec::new();
    for group in &replay.groups {
        if group.materialized_sector_id != group.representative_sector_id {
            return Err(RusticolError::invalid_argument(
                "LC topology replay currently requires the materialized sector to be the representative sector",
            ));
        }
        if group.sector_permutations.is_empty() {
            return Err(RusticolError::invalid_argument(
                "enabled LC topology replay group contains no sector permutations",
            ));
        }
        if !group.active_sector_ids.is_empty() {
            if !group
                .active_sector_ids
                .contains(&group.representative_sector_id)
            {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay active sector ids do not include the representative sector",
                ));
            }
            let mut seen_active = BTreeSet::new();
            for sector_id in &group.active_sector_ids {
                if !seen_active.insert(*sector_id) {
                    return Err(RusticolError::invalid_argument(
                        "LC topology replay active sector ids contain duplicates",
                    ));
                }
            }
        }
        let mut seen_permutations = BTreeSet::new();
        for permutation in &group.sector_permutations {
            if !group.active_sector_ids.is_empty()
                && !group.active_sector_ids.contains(&permutation.sector_id)
            {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay sector permutation is not listed in active sector ids",
                ));
            }
            if !seen_permutations.insert(permutation.sector_id) {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay sector permutations contain duplicate sector ids",
                ));
            }
            if !permutation.weight.is_finite() || permutation.weight <= 0.0 {
                return Err(RusticolError::invalid_argument(
                    "LC topology replay sector permutation weights must be positive finite numbers",
                ));
            }
            let mut mapping = Vec::new();
            for item in &permutation.label_permutation {
                if item.representative_label == 0 || item.sector_label == 0 {
                    return Err(RusticolError::invalid_argument(
                        "LC topology replay label permutations must use one-based labels",
                    ));
                }
                mapping.push((item.representative_label - 1, item.sector_label - 1));
            }
            mappings.push(mapping);
            weights.push(permutation.weight);
        }
    }
    if mappings.len() != replay.replayed_sector_count {
        return Err(RusticolError::invalid_argument(format!(
            "LC topology replay declares {} sectors but contains {} permutations",
            replay.replayed_sector_count,
            mappings.len()
        )));
    }
    Ok((mappings, weights))
}

fn build_runtime_parameter_slots(
    parameters: &[GenericRuntimeModelParameterManifest],
) -> RusticolResult<BTreeMap<String, RuntimeParameterSlots>> {
    let mut direct = BTreeMap::new();
    let mut complex_components: BTreeMap<String, (Option<usize>, Option<usize>)> = BTreeMap::new();
    for parameter in parameters {
        if parameter.kind == "derived_parameter_component" {
            continue;
        }
        if let Some(runtime_name) = &parameter.runtime_name {
            let slots = complex_components
                .entry(runtime_name.clone())
                .or_insert((None, None));
            match parameter.complex_component.as_deref() {
                Some("real") if slots.0.replace(parameter.parameter_index).is_none() => {}
                Some("imag") if slots.1.replace(parameter.parameter_index).is_none() => {}
                Some(component) => {
                    return Err(RusticolError::invalid_argument(format!(
                        "runtime model parameter {runtime_name:?} has duplicate or invalid component {component:?}"
                    )));
                }
                None => {
                    return Err(RusticolError::invalid_argument(format!(
                        "runtime model parameter {runtime_name:?} is missing component metadata"
                    )));
                }
            }
        } else if direct
            .insert(
                parameter.name.clone(),
                RuntimeParameterSlots {
                    real: parameter.parameter_index,
                    imaginary: None,
                },
            )
            .is_some()
        {
            return Err(RusticolError::invalid_argument(format!(
                "duplicate runtime model parameter name {:?}",
                parameter.name
            )));
        }
    }
    for (name, (real, imaginary)) in complex_components {
        let (Some(real), Some(imaginary)) = (real, imaginary) else {
            return Err(RusticolError::invalid_argument(format!(
                "runtime model parameter {name:?} requires real and imaginary slots"
            )));
        };
        if direct
            .insert(
                name.clone(),
                RuntimeParameterSlots {
                    real,
                    imaginary: Some(imaginary),
                },
            )
            .is_some()
        {
            return Err(RusticolError::invalid_argument(format!(
                "duplicate runtime model parameter name {name:?}"
            )));
        }
    }
    Ok(direct)
}

impl ExecutionRuntime {
    pub(super) fn from_manifest(manifest: ExecutionManifest) -> RusticolResult<Self> {
        let topology_replay = manifest.compiled.lc_topology_replay.as_ref();
        let (topology_replay_mappings, topology_replay_weights) =
            build_lc_topology_replay_mappings(topology_replay)?;
        let topology_replay_public_mappings = topology_replay_mappings.clone();
        let external_is_initial = manifest
            .runtime_schema
            .external_particles
            .iter()
            .map(|particle| particle.role == "initial")
            .collect::<Vec<_>>();
        let particle_masses = manifest
            .runtime_schema
            .model
            .as_ref()
            .map(|model| {
                model
                    .particles
                    .iter()
                    .map(|particle| (particle.pdg, particle.mass))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        let particle_mass_parameter_names = manifest
            .runtime_schema
            .model
            .as_ref()
            .map(|model| {
                model
                    .particles
                    .iter()
                    .filter_map(|particle| {
                        particle
                            .mass_parameter
                            .as_ref()
                            .map(|name| (particle.pdg, name.clone()))
                    })
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        let mut model_parameters = manifest.runtime_schema.model_parameters.clone();
        model_parameters.sort_by_key(|parameter| parameter.parameter_index);
        let model_parameter_values_f64 = model_parameters
            .iter()
            .map(|parameter| parameter.default)
            .collect::<Vec<_>>();
        let model_parameter_name_to_index = model_parameters
            .iter()
            .map(|parameter| (parameter.name.clone(), parameter.parameter_index))
            .collect::<BTreeMap<_, _>>();
        let model_parameter_runtime_slots = build_runtime_parameter_slots(&model_parameters)?;
        let color_factor_in_contraction = manifest
            .runtime_schema
            .amplitude_stage
            .color_contraction
            .as_ref()
            .map(|contraction| contraction.supported && contraction.includes_color_factor)
            .unwrap_or(false);
        let (
            normalization_factor,
            normalization_color_factor,
            normalization_average_factor,
            normalization_identical_factor,
            normalization_qcd_coupling_power,
            normalization_electroweak_coupling_power,
        ) = manifest
            .runtime_schema
            .normalization
            .as_ref()
            .map(|normalization| {
                let color_factor = if color_factor_in_contraction {
                    1.0
                } else {
                    normalization.color_factor
                };
                (
                    color_factor * normalization.global_coupling_factor
                        / (normalization.average_factor * normalization.identical_factor),
                    color_factor,
                    normalization.average_factor,
                    normalization.identical_factor,
                    normalization.qcd_coupling_power,
                    normalization.electroweak_coupling_power,
                )
            })
            .unwrap_or((1.0, 1.0, 1.0, 1.0, 0, 0));
        Ok(Self {
            process: manifest.process,
            key: manifest.key,
            color_accuracy: manifest.color_accuracy,
            external_count: manifest.external_pdg_order.len(),
            external_pdg_order: manifest.external_pdg_order,
            parameter_count: manifest
                .runtime_schema
                .parameter_layout
                .value_component_count
                + manifest
                    .runtime_schema
                    .parameter_layout
                    .momentum_parameter_count
                + manifest
                    .runtime_schema
                    .parameter_layout
                    .model_parameter_count,
            value_parameter_count: manifest
                .runtime_schema
                .parameter_layout
                .value_component_count,
            momentum_parameter_count: manifest
                .runtime_schema
                .parameter_layout
                .momentum_parameter_count,
            current_count: manifest.dag_summary.current_count,
            source_count: manifest.dag_summary.source_count,
            interaction_count: manifest.dag_summary.interaction_count,
            stage_count: manifest.runtime_schema.stages.len(),
            amplitude_output_count: manifest.runtime_schema.amplitude_stage.output_count,
            lc_topology_replay_enabled: !topology_replay_mappings.is_empty(),
            lc_topology_replay_mappings: topology_replay_mappings,
            lc_topology_replay_public_mappings: topology_replay_public_mappings,
            lc_topology_replay_weights: topology_replay_weights,
            runtime_unavailable_message: manifest.compiled.runtime_unavailable_message,
            sources: manifest.runtime_schema.source_fill.sources,
            momentum_slots: manifest.runtime_schema.momentum_slots,
            external_is_initial,
            particle_masses,
            particle_mass_parameter_names,
            normalization_factor,
            normalization_color_factor,
            normalization_average_factor,
            normalization_identical_factor,
            normalization_qcd_coupling_power,
            normalization_electroweak_coupling_power,
            model_parameters,
            model_parameter_name_to_index,
            model_parameter_runtime_slots,
            model_parameter_values_f64,
            model_parameter_evaluator: None,
            physics: None,
            stages: None,
            amplitude_stage: None,
            state_scratch_f64: Vec::new(),
            values_scratch_f64: Vec::new(),
        })
    }

    pub(super) fn load_from_manifest(
        manifest: ExecutionManifest,
        root: &Path,
    ) -> RusticolResult<Self> {
        ensure_execution_capabilities_supported(&manifest)?;
        let stage_evaluators = manifest.compiled.stage_evaluators.clone();
        let model_parameter_evaluator = manifest.compiled.model_parameter_evaluator.clone();
        let amplitude_stage_manifest = manifest.runtime_schema.amplitude_stage.clone();
        let mut runtime = Self::from_manifest(manifest)?;
        if let Some(manifest) = model_parameter_evaluator {
            if manifest.kind != "generic-model-parameter-evaluator" {
                return Err(RusticolError::invalid_argument(format!(
                    "unsupported model-parameter evaluator kind {:?}",
                    manifest.kind
                )));
            }
            runtime.model_parameter_evaluator = Some(ModelParameterEvaluatorRuntime {
                input_parameter_indices: manifest.input_parameter_indices,
                outputs: manifest.outputs,
                evaluator: EvaluatorGroup::load(&manifest.evaluator, root)?,
            });
            runtime.refresh_derived_model_parameters()?;
        }
        if let Some(stage_evaluators) = stage_evaluators {
            let stages = stage_evaluators
                .stages
                .iter()
                .map(|stage| StageRuntime::load(stage, root))
                .collect::<RusticolResult<Vec<_>>>()?;
            runtime.stages = Some(stages);
            runtime.amplitude_stage = Some(AmplitudeRuntime::load(
                &amplitude_stage_manifest,
                &stage_evaluators.amplitude_stage,
                root,
            )?);
        }
        Ok(runtime)
    }

    pub(super) fn execution_unavailable_error(&self) -> RusticolError {
        if self.stages.is_none() || self.amplitude_stage.is_none() {
            return RusticolError::invalid_argument(format!(
                "generic DAG schema-v3 artifact for {} loaded successfully, but it has no \
                 serialized generic stage evaluators",
                self.process
            ));
        }
        let detail = self.runtime_unavailable_message.as_deref().unwrap_or(
            "generic source filling and staged evaluator execution from schema-v3 metadata remain pending",
        );
        RusticolError::invalid_argument(format!(
            "generic DAG schema-v3 artifact for {} loaded successfully, but generic Rusticol \
             execution is not wired yet; {detail}",
            self.process,
        ))
    }
}

fn ensure_execution_capabilities_supported(manifest: &ExecutionManifest) -> RusticolResult<()> {
    let mut execution_capabilities = BTreeSet::new();
    if let Some(parameters) = &manifest.compiled.model_parameter_evaluator {
        ensure_evaluator_capabilities_supported(&parameters.evaluator)?;
        let actual = evaluator_runtime_capabilities(&parameters.evaluator)?;
        validate_declared_capabilities(
            &parameters.required_runtime_capabilities,
            &actual,
            "model-parameter evaluator",
        )?;
        execution_capabilities.extend(actual);
    }
    if let Some(stages) = &manifest.compiled.stage_evaluators {
        let mut stage_capabilities = BTreeSet::new();
        for stage in &stages.stages {
            ensure_evaluator_capabilities_supported(&stage.evaluator)?;
            stage_capabilities.extend(evaluator_runtime_capabilities(&stage.evaluator)?);
        }
        ensure_evaluator_capabilities_supported(&stages.amplitude_stage.evaluator)?;
        stage_capabilities.extend(evaluator_runtime_capabilities(
            &stages.amplitude_stage.evaluator,
        )?);
        validate_declared_capabilities(
            &stages.required_runtime_capabilities,
            &stage_capabilities,
            "stage evaluator set",
        )?;
        execution_capabilities.extend(stage_capabilities);
    }
    validate_declared_capabilities(
        &manifest.required_runtime_capabilities,
        &execution_capabilities,
        "execution manifest",
    )?;
    Ok(())
}

fn validate_declared_capabilities(
    declared: &[String],
    actual: &BTreeSet<String>,
    context: &str,
) -> RusticolResult<()> {
    let declared_set = declared.iter().cloned().collect::<BTreeSet<_>>();
    if declared_set.len() != declared.len() || &declared_set != actual {
        return Err(RusticolError::integrity(format!(
            "{context} capabilities {declared:?} do not match evaluator capabilities {actual:?}"
        )));
    }
    Ok(())
}
