// SPDX-License-Identifier: 0BSD

use super::*;

pub(super) fn load_verified_evaluator(
    artifact: &VerifiedArtifact,
    selection: &crate::ArtifactSelection,
) -> RusticolResult<(ExecutionManifest, PathBuf)> {
    let root_manifest_path = &artifact.manifest().runtime.evaluator_manifest_path;
    if artifact.payload(root_manifest_path)?.role != PayloadRole::EvaluatorManifest {
        return Err(RusticolError::security(format!(
            "runtime evaluator path {root_manifest_path:?} is not an evaluator-manifest payload"
        )));
    }
    let bytes = artifact.read_payload(root_manifest_path)?;
    let header: ExecutionManifestHeader = serde_json::from_slice(&bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse evaluator manifest {root_manifest_path:?}: {error}"
        ))
    })?;
    if matches!(header.schema_version, 1 | 2) {
        return Err(RusticolError::compatibility(format!(
            "internal evaluator schema v{} is unsupported and unsafe to migrate; regenerate the artifact with `pyamplicol generate`",
            header.schema_version
        )));
    }
    if header.schema_version != PROCESS_ARTIFACT_SCHEMA_VERSION {
        return Err(RusticolError::compatibility(format!(
            "unsupported internal evaluator schema {}; this runtime requires schema v{}",
            header.schema_version, PROCESS_ARTIFACT_SCHEMA_VERSION
        )));
    }
    if header.kind == "pyamplicol-runtime-execution" {
        if artifact.manifest().processes.len() != 1 {
            return Err(RusticolError::integrity(
                "a direct execution manifest cannot represent multiple outer processes",
            ));
        }
        let manifest = parse_execution_payload(&bytes, root_manifest_path)?;
        let outer = &artifact.manifest().processes[0];
        if manifest.key != outer.id
            || manifest.process != outer.expression
            || manifest.color_accuracy != outer.color_accuracy
            || manifest.external_pdg_order != outer.external_pdgs
        {
            return Err(RusticolError::integrity(format!(
                "execution manifest {root_manifest_path:?} does not match outer process {:?}",
                outer.id
            )));
        }
        validate_capability_list_match(
            &artifact.manifest().runtime.required_runtime_capabilities,
            &manifest.required_runtime_capabilities,
            "outer runtime and direct execution manifest",
        )?;
        validate_capability_list_match(
            &outer.required_runtime_capabilities,
            &manifest.required_runtime_capabilities,
            "outer process and direct execution manifest",
        )?;
        let path = artifact.payload_path(root_manifest_path)?;
        let root = path.parent().ok_or_else(|| {
            RusticolError::artifact("evaluator manifest has no containing directory")
        })?;
        validate_evaluator_payload_references(artifact, root, &manifest)?;
        return Ok((manifest, root.to_path_buf()));
    }
    if header.kind != "pyamplicol-runtime-execution-set" {
        return Err(RusticolError::compatibility(format!(
            "unsupported internal evaluator manifest kind {:?}",
            header.kind
        )));
    }
    let process_set: ExecutionSetManifest = serde_json::from_slice(&bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse evaluator process-set manifest {root_manifest_path:?}: {error}"
        ))
    })?;
    if process_set.schema_version != PROCESS_ARTIFACT_SCHEMA_VERSION
        || process_set.kind != "pyamplicol-runtime-execution-set"
        || process_set.processes.is_empty()
    {
        return Err(RusticolError::compatibility(
            "invalid schema-v3 execution-set manifest; regenerate the artifact",
        ));
    }
    let mut process_ids = BTreeSet::new();
    let mut manifest_paths = BTreeSet::new();
    let mut entry_capabilities = BTreeSet::new();
    for entry in &process_set.processes {
        if !process_ids.insert(entry.process_id.as_str()) {
            return Err(RusticolError::integrity(format!(
                "execution-set manifest contains duplicate process id {:?}",
                entry.process_id
            )));
        }
        let path = execution_manifest_path(root_manifest_path, &entry.manifest_path)?;
        if !manifest_paths.insert(path.to_ascii_lowercase()) {
            return Err(RusticolError::integrity(format!(
                "execution-set manifest reuses manifest path {path:?}"
            )));
        }
        if artifact.payload(&path)?.role != PayloadRole::EvaluatorManifest {
            return Err(RusticolError::security(format!(
                "internal evaluator path {path:?} is not an evaluator-manifest payload"
            )));
        }
        let capabilities = entry
            .required_runtime_capabilities
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if capabilities.len() != entry.required_runtime_capabilities.len() {
            return Err(RusticolError::integrity(format!(
                "execution-set entry {:?} contains duplicate runtime capabilities",
                entry.process_id
            )));
        }
        entry_capabilities.extend(capabilities);
    }
    validate_capability_set_match(
        &process_set.required_runtime_capabilities,
        &entry_capabilities,
        "execution-set manifest",
    )?;
    validate_capability_list_match(
        &artifact.manifest().runtime.required_runtime_capabilities,
        &process_set.required_runtime_capabilities,
        "outer runtime and execution-set manifest",
    )?;
    let outer_process_ids = artifact
        .manifest()
        .processes
        .iter()
        .map(|process| process.id.as_str())
        .collect::<BTreeSet<_>>();
    if process_ids != outer_process_ids {
        return Err(RusticolError::integrity(
            "execution-set process ids do not exactly match the outer schema-v3 manifest",
        ));
    }

    let mut selected = None;
    for entry in &process_set.processes {
        let manifest_path = execution_manifest_path(root_manifest_path, &entry.manifest_path)?;
        let bytes = artifact.read_payload(&manifest_path)?;
        let manifest = parse_execution_payload(&bytes, &manifest_path)?;
        let outer = artifact
            .manifest()
            .processes
            .iter()
            .find(|process| process.id == entry.process_id)
            .expect("execution-set ids were matched to outer processes");
        validate_capability_list_match(
            &outer.required_runtime_capabilities,
            &entry.required_runtime_capabilities,
            &format!(
                "outer process {:?} and execution-set entry",
                entry.process_id
            ),
        )?;
        if manifest.key != outer.id
            || manifest.process != outer.expression
            || manifest.color_accuracy != outer.color_accuracy
            || manifest.external_pdg_order != outer.external_pdgs
        {
            return Err(RusticolError::integrity(format!(
                "execution manifest {manifest_path:?} does not match outer process {:?}",
                outer.id
            )));
        }
        validate_capability_list_match(
            &entry.required_runtime_capabilities,
            &manifest.required_runtime_capabilities,
            &format!("execution-set entry {:?}", entry.process_id),
        )?;
        let path = artifact.payload_path(&manifest_path)?;
        let evaluator_root = path.parent().ok_or_else(|| {
            RusticolError::artifact("evaluator manifest has no containing directory")
        })?;
        validate_evaluator_payload_references(artifact, evaluator_root, &manifest)?;
        if entry.process_id == selection.process.id {
            selected = Some((manifest, evaluator_root.to_path_buf()));
        }
    }
    selected.ok_or_else(|| {
        RusticolError::integrity(format!(
            "execution set does not contain selected outer process {:?}",
            selection.process.id
        ))
    })
}

fn validate_capability_list_match(
    left: &[String],
    right: &[String],
    context: &str,
) -> RusticolResult<()> {
    let right_set = right.iter().cloned().collect::<BTreeSet<_>>();
    if right_set.len() != right.len() {
        return Err(RusticolError::integrity(format!(
            "{context} contains duplicate runtime capabilities"
        )));
    }
    validate_capability_set_match(left, &right_set, context)
}

fn validate_capability_set_match(
    declared: &[String],
    actual: &BTreeSet<String>,
    context: &str,
) -> RusticolResult<()> {
    let declared_set = declared.iter().cloned().collect::<BTreeSet<_>>();
    if declared_set.len() != declared.len() || &declared_set != actual {
        return Err(RusticolError::integrity(format!(
            "{context} runtime capabilities {declared:?} do not match {actual:?}"
        )));
    }
    Ok(())
}

fn parse_execution_payload(bytes: &[u8], path: &str) -> RusticolResult<ExecutionManifest> {
    let header: ExecutionManifestHeader = serde_json::from_slice(bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse evaluator manifest {path:?}: {error}"
        ))
    })?;
    if matches!(header.schema_version, 1 | 2) {
        return Err(RusticolError::compatibility(format!(
            "internal evaluator schema v{} is unsupported and unsafe to migrate; regenerate the artifact with `pyamplicol generate`",
            header.schema_version
        )));
    }
    if header.schema_version != PROCESS_ARTIFACT_SCHEMA_VERSION {
        return Err(RusticolError::compatibility(format!(
            "unsupported internal evaluator schema {}; this runtime requires schema v{}",
            header.schema_version, PROCESS_ARTIFACT_SCHEMA_VERSION
        )));
    }
    if header.kind != "pyamplicol-runtime-execution" {
        return Err(RusticolError::compatibility(format!(
            "unsupported internal evaluator manifest kind {:?}",
            header.kind
        )));
    }
    serde_json::from_slice(bytes).map_err(|error| execution_manifest_parse_error(path, error))
}

pub(super) fn execution_manifest_parse_error(
    path: &str,
    error: serde_json::Error,
) -> RusticolError {
    let detail = error.to_string();
    if ["source_ir", "applied_crossing", "source_basis"]
        .iter()
        .any(|field| detail.contains(&format!("missing field `{field}`")))
    {
        return RusticolError::compatibility(format!(
            "schema-v3 evaluator manifest {path:?} predates typed source metadata; regenerate the artifact with `pyamplicol generate`"
        ));
    }
    if detail.contains("missing field `contraction_ir`") {
        return RusticolError::compatibility(format!(
            "schema-v3 evaluator manifest {path:?} predates typed amplitude-contraction metadata; regenerate the artifact with `pyamplicol generate`"
        ));
    }
    RusticolError::serialization(format!(
        "could not parse schema-v3 evaluator manifest {path:?}: {detail}"
    ))
}

pub(super) fn execution_manifest_path(
    parent_manifest: &str,
    entry_path: &str,
) -> RusticolResult<String> {
    let base = Path::new(parent_manifest)
        .parent()
        .unwrap_or_else(|| Path::new(""));
    let entry = confined_internal_path(entry_path, "evaluator process path")?;
    let path = base.join(entry);
    path.to_str()
        .map(str::to_string)
        .ok_or_else(|| RusticolError::security("evaluator process path is not valid UTF-8"))
}

pub(super) fn validate_evaluator_payload_references(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &ExecutionManifest,
) -> RusticolResult<()> {
    let relative_root = evaluator_root.strip_prefix(artifact.root()).map_err(|_| {
        RusticolError::security("evaluator root escapes the verified artifact root")
    })?;
    if let Some(parameters) = &manifest.compiled.model_parameter_evaluator {
        validate_evaluator_reference(artifact, relative_root, &parameters.evaluator)?;
    }
    if let Some(stages) = &manifest.compiled.stage_evaluators {
        for stage in &stages.stages {
            validate_evaluator_reference(artifact, relative_root, &stage.evaluator)?;
        }
        validate_evaluator_reference(artifact, relative_root, &stages.amplitude_stage.evaluator)?;
    }
    Ok(())
}

pub(super) fn validate_evaluator_reference(
    artifact: &VerifiedArtifact,
    relative_root: &Path,
    evaluator: &EvaluatorManifest,
) -> RusticolResult<()> {
    match evaluator {
        EvaluatorManifest::SymjitApplication {
            application_path,
            evaluator_state_path,
            ..
        } => {
            validate_evaluator_state_path(artifact, relative_root, application_path)?;
            if let Some(path) = evaluator_state_path {
                validate_evaluator_state_path(artifact, relative_root, path)?;
            }
            Ok(())
        }
        EvaluatorManifest::Jit {
            evaluator_state_path,
            ..
        } => validate_evaluator_state_path(artifact, relative_root, evaluator_state_path),
        EvaluatorManifest::CompiledComplex {
            library_path,
            evaluator_state_path,
            ..
        } => {
            validate_evaluator_state_path(artifact, relative_root, library_path)?;
            if let Some(path) = evaluator_state_path {
                validate_evaluator_state_path(artifact, relative_root, path)?;
            }
            Ok(())
        }
        EvaluatorManifest::Chunked { chunks, .. } => {
            for chunk in chunks {
                validate_evaluator_reference(artifact, relative_root, chunk)?;
            }
            Ok(())
        }
    }
}

pub(super) fn validate_evaluator_state_path(
    artifact: &VerifiedArtifact,
    relative_root: &Path,
    value: &str,
) -> RusticolResult<()> {
    let path = relative_root.join(confined_internal_path(value, "evaluator-state path")?);
    let path = path
        .to_str()
        .ok_or_else(|| RusticolError::security("evaluator-state path is not valid UTF-8"))?;
    let payload = artifact.payload(path)?;
    if payload.role != PayloadRole::EvaluatorState {
        return Err(RusticolError::security(format!(
            "evaluator reference {path:?} has payload role {:?}, expected evaluator-state",
            payload.role
        )));
    }
    artifact.payload_path(path)?;
    Ok(())
}

pub(super) fn confined_internal_path<'a>(
    value: &'a str,
    description: &str,
) -> RusticolResult<&'a Path> {
    let path = Path::new(value);
    if value.is_empty()
        || value.contains('\\')
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(RusticolError::security(format!(
            "{description} {value:?} is not a confined relative path"
        )));
    }
    Ok(path)
}

pub(super) fn parse_reduction_group_id(value: &str) -> RusticolResult<i64> {
    value
        .parse::<i64>()
        .or_else(|_| value.rsplit(':').next().unwrap_or(value).parse::<i64>())
        .map_err(|_| {
            RusticolError::artifact(format!(
                "reduction group id {value:?} does not contain the evaluator's numeric group id"
            ))
        })
}

pub(super) fn apply_final_state_alias_metadata(
    mut physics: ProcessPhysicsV1,
    alias: &crate::ProcessAlias,
) -> RusticolResult<ProcessPhysicsV1> {
    physics.validate()?;
    let permutation = &alias.external_permutation;
    let particle_count = physics.external_particles.len();
    if permutation.len() != particle_count
        || alias.external_pdgs.len() != particle_count
        || permutation.iter().copied().collect::<BTreeSet<_>>()
            != (0..particle_count).collect::<BTreeSet<_>>()
    {
        return Err(RusticolError::integrity(format!(
            "alias {:?} does not define a complete particle permutation and PDG order",
            alias.id
        )));
    }
    if particle_count < 2 || permutation[0] != 0 || permutation[1] != 1 {
        return Err(RusticolError::compatibility(format!(
            "alias {:?} is not a final-state permutation; crossing reuse is unsupported",
            alias.id
        )));
    }
    let mut expected_alias_pdgs = vec![0; particle_count];
    for (representative_index, alias_index) in permutation.iter().copied().enumerate() {
        expected_alias_pdgs[alias_index] = physics.external_particles[representative_index].pdg;
    }
    if alias.external_pdgs != expected_alias_pdgs {
        return Err(RusticolError::integrity(format!(
            "alias {:?} PDG order {:?} does not match the representative physics payload and permutation; expected {:?}",
            alias.id, alias.external_pdgs, expected_alias_pdgs
        )));
    }

    let mut particles = vec![None; particle_count];
    for (representative_index, alias_index) in permutation.iter().copied().enumerate() {
        let mut particle = physics.external_particles[representative_index].clone();
        particle.index = alias_index;
        particle.label = alias_index + 1;
        particle.momentum_slot = alias_index;
        particles[alias_index] = Some(particle);
    }
    physics.external_particles = particles
        .into_iter()
        .map(|particle| particle.expect("validated permutation is complete"))
        .collect();

    let mut helicity_id_map = BTreeMap::new();
    let mut helicity_updates = Vec::with_capacity(physics.helicities.len());
    for helicity in &physics.helicities {
        let mut alias_values = vec![0; particle_count];
        for (representative_index, alias_index) in permutation.iter().copied().enumerate() {
            alias_values[alias_index] = helicity.values[representative_index];
        }
        let alias_id = canonical_helicity_id(&alias_values);
        helicity_id_map.insert(helicity.id.clone(), alias_id.clone());
        helicity_updates.push((alias_values, alias_id));
    }
    for (helicity, (values, id)) in physics.helicities.iter_mut().zip(helicity_updates) {
        let representative_id = remapped_alias_id(
            &helicity_id_map,
            &helicity.representative_id,
            "helicity representative",
        )?;
        helicity.values = values;
        helicity.id = id;
        helicity.representative_id = representative_id;
    }

    let mut color_id_map = BTreeMap::new();
    let mut color_updates = Vec::with_capacity(physics.color_components.len());
    for color in &physics.color_components {
        match color {
            PhysicsColorComponentV1::LcFlow(flow) => {
                let alias_word = flow
                    .word
                    .iter()
                    .map(|label| {
                        let representative_index = label.checked_sub(1).ok_or_else(|| {
                            RusticolError::artifact("LC color-flow labels must be one-based")
                        })?;
                        let alias_index =
                            permutation.get(representative_index).ok_or_else(|| {
                                RusticolError::artifact(
                                    "LC color-flow label exceeds alias permutation",
                                )
                            })?;
                        Ok(alias_index + 1)
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                let alias_id = canonical_color_id(&alias_word);
                color_id_map.insert(flow.id.clone(), alias_id.clone());
                color_updates.push((Some(alias_word), alias_id));
            }
            PhysicsColorComponentV1::ContractedColor(color) => {
                color_id_map.insert(color.id.clone(), color.id.clone());
                color_updates.push((None, color.id.clone()));
            }
        }
    }
    for (color, (word, id)) in physics.color_components.iter_mut().zip(color_updates) {
        match color {
            PhysicsColorComponentV1::LcFlow(flow) => {
                let representative_id = remapped_alias_id(
                    &color_id_map,
                    &flow.representative_id,
                    "color representative",
                )?;
                flow.word = word.expect("LC color update includes a word");
                flow.id = id;
                flow.representative_id = representative_id;
            }
            PhysicsColorComponentV1::ContractedColor(color) => {
                if word.is_some() {
                    return Err(RusticolError::integrity(
                        "contracted color alias update unexpectedly contains an LC word",
                    ));
                }
                color.id = id;
            }
        }
    }

    for group in &mut physics.reduction.groups {
        group.representative_helicity_id = remapped_alias_id(
            &helicity_id_map,
            &group.representative_helicity_id,
            "reduction representative helicity",
        )?;
        group.representative_color_id = remapped_alias_id(
            &color_id_map,
            &group.representative_color_id,
            "reduction representative color",
        )?;
        for id in &mut group.physical_helicity_ids {
            *id = remapped_alias_id(&helicity_id_map, id, "reduction physical helicity")?;
        }
        for id in &mut group.physical_color_ids {
            *id = remapped_alias_id(&color_id_map, id, "reduction physical color")?;
        }
    }

    physics.process_id = alias.id.clone();
    physics.process = alias.expression.clone();
    if physics
        .external_particles
        .iter()
        .map(|particle| particle.pdg)
        .ne(alias.external_pdgs.iter().copied())
    {
        return Err(RusticolError::integrity(format!(
            "alias {:?} external particle metadata does not match its public PDG order",
            alias.id
        )));
    }
    physics.validate()?;
    Ok(physics)
}

fn canonical_helicity_id(values: &[i32]) -> String {
    format!(
        "h:{}",
        values
            .iter()
            .map(|value| format!("{value:+}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn canonical_color_id(word: &[usize]) -> String {
    if word.is_empty() {
        return "flow:singlet".to_string();
    }
    format!(
        "flow:{}",
        word.iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",")
    )
}

fn remapped_alias_id(
    ids: &BTreeMap<String, String>,
    representative_id: &str,
    description: &str,
) -> RusticolResult<String> {
    ids.get(representative_id).cloned().ok_or_else(|| {
        RusticolError::integrity(format!(
            "{description} {representative_id:?} is absent from alias remapping"
        ))
    })
}

pub(super) fn selector_set(
    ids: Option<&[String]>,
    kind: &str,
) -> Result<Option<BTreeSet<String>>, RusticolError> {
    let Some(ids) = ids else {
        return Ok(None);
    };
    if ids.is_empty() {
        return Err(RusticolError::selector(format!(
            "resolved {kind} selection must not be empty"
        )));
    }
    let selected = ids.iter().cloned().collect::<BTreeSet<_>>();
    if selected.len() != ids.len() {
        return Err(RusticolError::selector(format!(
            "resolved {kind} selection contains duplicate ids"
        )));
    }
    Ok(Some(selected))
}
