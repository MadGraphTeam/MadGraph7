// SPDX-License-Identifier: 0BSD

//! Authenticated construction boundary for the private LC on-the-fly lane.
//!
//! Loading stops at the compact process seed and model-wide prepared catalogs.
//! It never opens or constructs a process-wide [`crate::recurrence::DirectRecurrencePlan`]
//! and it never materializes dense helicity/color axes. The existing runtime
//! adapter owns public selector decoding after this boundary.

use super::eager_manifest::PreparedKernelPackManifest;
use super::on_the_fly_lane::OnTheFlyNativeRuntime;
use super::on_the_fly_manifest::{
    ON_THE_FLY_PROCESS_SEED_MEMBER, OnTheFlyColorCoverage, OnTheFlyExecutionManifest,
};
use super::on_the_fly_public_metadata::{
    OnTheFlyPublicMetadataV1, parse_on_the_fly_public_metadata,
};
use super::on_the_fly_selectors::{
    OnTheFlyCompactSelectorAdapterV1, OnTheFlyLcColorCoverageV1, OnTheFlyLcSelectorPolicyV1,
};
use super::recurrence_backend::NativeRecurrencePreparedExecutorPool;
use super::recurrence_lane::PreparedParameterProjectionEntry;
use super::recurrence_load::{
    build_on_the_fly_source_domains, decode_sha256, runtime_parameter_slots,
    validate_recurrence_prepared_pack_outer_target,
};
use super::*;
use crate::pacbin::{PacbinMemberKind, PacbinReader};
use crate::recurrence::on_the_fly::decode_on_the_fly_process_seed_v1;
use crate::recurrence::template_json::project_recurrence_template_catalog_json_v1;

pub(super) struct LoadedOnTheFlyRuntime {
    pub(super) common: ExecutionRuntime,
    pub(super) lane: OnTheFlyNativeRuntime,
    pub(super) selectors: OnTheFlyCompactSelectorAdapterV1,
    pub(super) metadata_selectors: OnTheFlyCompactSelectorAdapterV1,
    pub(super) public_metadata: OnTheFlyPublicMetadataV1,
}

pub(super) fn load_on_the_fly_native_runtime(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &OnTheFlyExecutionManifest,
    selection: &crate::ArtifactSelection,
) -> RusticolResult<LoadedOnTheFlyRuntime> {
    let public_metadata = load_public_metadata(artifact, manifest, &selection.process)?;
    let seed = load_process_seed(artifact, evaluator_root, manifest)?;
    if seed.identity() != manifest.runtime_metadata.process_seed_identity {
        return Err(RusticolError::integrity(
            "on-the-fly execution metadata does not identify its decoded process seed",
        ));
    }
    let (pack_bytes, pack, payload_root) = load_prepared_pack(artifact, manifest)?;
    let raw_templates = pack.recurrence_template.as_ref().ok_or_else(|| {
        RusticolError::compatibility(
            "on-the-fly execution requires the prepared recurrence template catalog",
        )
    })?;
    let templates = project_recurrence_template_catalog_json_v1(raw_templates)?.validate()?;
    let summary = templates.summary();
    if seed.template_catalog_digest() != summary.catalog_digest
        || seed.model_digest() != summary.compiled_model_digest
        || seed.prepared_pack_digest() != summary.prepared_kernel_pack_digest
    {
        return Err(RusticolError::integrity(
            "on-the-fly process seed disagrees with the authenticated recurrence template catalog",
        ));
    }

    let payloads = artifact.evaluator_payload_store(&payload_root)?;
    let pool = NativeRecurrencePreparedExecutorPool::load_from_store(
        &pack_bytes,
        &payloads,
        &seed.prepared_pack_digest().to_string(),
        &seed.direct_catalog_digest().to_string(),
    )?;
    let direct_catalog = pool.prepared_direct_catalog()?;
    if direct_catalog.direct_template_catalog_digest() != seed.direct_catalog_digest() {
        return Err(RusticolError::integrity(
            "on-the-fly prepared Direct-Arena catalog disagrees with the compact process seed",
        ));
    }
    let source_domains = build_on_the_fly_source_domains(&seed, &templates)?;
    let sources = pool.bind_source_domains(source_domains)?;
    let resolver = pool.into_on_the_fly_resolver(sources);

    let (mut common, defaults, projection) = build_common_runtime(manifest)?;
    common.model_parameter_evaluator =
        super::eager_load::load_prepared_model_parameter_evaluator_for_runtime(
            &pack,
            &common.model_parameters,
            &payloads,
        )?;
    common.refresh_derived_model_parameters()?;

    let metadata_selectors =
        OnTheFlyCompactSelectorAdapterV1::from_seed(&seed, selector_policy(manifest))?;
    manifest.selector_policy.selector_census.validate_against(
        metadata_selectors.helicity_count(),
        metadata_selectors.color_count(),
    )?;
    let selectors = metadata_selectors
        .clone()
        .with_public_permutation(&selection.external_permutation)?;
    let lane = OnTheFlyNativeRuntime::new(
        templates,
        direct_catalog,
        seed,
        resolver,
        defaults,
        projection,
        &common.model_parameter_values_f64,
    )?;
    Ok(LoadedOnTheFlyRuntime {
        common,
        lane,
        selectors,
        metadata_selectors,
        public_metadata,
    })
}

fn load_public_metadata(
    artifact: &VerifiedArtifact,
    manifest: &OnTheFlyExecutionManifest,
    outer: &crate::ArtifactProcess,
) -> RusticolResult<OnTheFlyPublicMetadataV1> {
    let record = artifact.payload(&outer.physics_path)?;
    if record.role != PayloadRole::RuntimePhysics
        || record.media_type != "application/json"
        || record.executable
        || record.process_id.as_deref() != Some(manifest.key.as_str())
    {
        return Err(RusticolError::integrity(
            "on-the-fly compact public metadata has the wrong authenticated payload role",
        ));
    }
    let bytes = artifact.read_payload(&outer.physics_path)?;
    parse_on_the_fly_public_metadata(&bytes, &outer.physics_path, outer, manifest)
}

fn selector_policy(manifest: &OnTheFlyExecutionManifest) -> OnTheFlyLcSelectorPolicyV1 {
    let color_coverage = match manifest.selector_policy.color_coverage {
        OnTheFlyColorCoverage::Complete => OnTheFlyLcColorCoverageV1::Complete,
    };
    OnTheFlyLcSelectorPolicyV1 {
        color_coverage,
        reference_color_word: manifest
            .selector_policy
            .reference_color_word
            .clone()
            .map(Vec::into_boxed_slice),
        trace_reflections_folded: manifest.selector_policy.trace_reflections_folded,
    }
}

fn load_process_seed(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &OnTheFlyExecutionManifest,
) -> RusticolResult<crate::recurrence::on_the_fly::OnTheFlyProcessSeedV1> {
    let container = &manifest.runtime_container;
    let payloads = artifact.evaluator_payload_store(evaluator_root)?;
    let logical_path = payloads.logical_path(&container.path)?;
    let record = artifact.payload(&logical_path)?;
    if record.role != PayloadRole::EvaluatorState
        || record.media_type != "application/octet-stream"
        || record.executable
        || record.process_id.as_deref() != Some(manifest.key.as_str())
    {
        return Err(RusticolError::integrity(
            "on-the-fly runtime container has the wrong authenticated payload role",
        ));
    }
    let path = artifact.root().join(&logical_path);
    let file = artifact.open_payload_file(&logical_path)?;
    let expected_sha = decode_sha256(&record.sha256)?;
    let reader = PacbinReader::open_file_with_sha256(file, &path, &expected_sha)?;
    if reader.members().len() != 1 {
        return Err(RusticolError::integrity(
            "on-the-fly runtime container must contain exactly its compact process seed",
        ));
    }
    let member = reader.member(ON_THE_FLY_PROCESS_SEED_MEMBER)?;
    if member.logical_path() != container.seed_member_path
        || member.kind() != PacbinMemberKind::OnTheFlyProcessSeed
    {
        return Err(RusticolError::integrity(
            "on-the-fly runtime container does not contain the canonical process-seed member",
        ));
    }
    decode_on_the_fly_process_seed_v1(reader.member_bytes(ON_THE_FLY_PROCESS_SEED_MEMBER)?)
}

fn load_prepared_pack(
    artifact: &VerifiedArtifact,
    manifest: &OnTheFlyExecutionManifest,
) -> RusticolResult<(Vec<u8>, PreparedKernelPackManifest, PathBuf)> {
    let path = confined_internal_path(
        &manifest.kernel_pack.manifest_path,
        "on-the-fly prepared kernel-pack manifest path",
    )?;
    let logical = path.to_str().ok_or_else(|| {
        RusticolError::security("on-the-fly prepared kernel-pack path is not UTF-8")
    })?;
    if artifact.payload(logical)?.role != PayloadRole::EvaluatorManifest {
        return Err(RusticolError::security(
            "on-the-fly prepared kernel-pack manifest has the wrong payload role",
        ));
    }
    let bytes = artifact.read_payload(logical)?;
    let pack: PreparedKernelPackManifest = serde_json::from_slice(&bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse on-the-fly prepared kernel pack: {error}"
        ))
    })?;
    pack.validate()?;
    validate_recurrence_prepared_pack_outer_target(&artifact.manifest().producer.target, &pack)?;
    let payload_root = artifact.root().join(confined_internal_path(
        &manifest.kernel_pack.payload_root,
        "on-the-fly prepared kernel payload root",
    )?);
    Ok((bytes, pack, payload_root))
}

fn build_common_runtime(
    manifest: &OnTheFlyExecutionManifest,
) -> RusticolResult<(
    ExecutionRuntime,
    Vec<crate::EagerComplex64>,
    Vec<PreparedParameterProjectionEntry>,
)> {
    let metadata = &manifest.runtime_metadata;
    let model_parameters = metadata
        .runtime_parameters
        .iter()
        .map(|parameter| GenericRuntimeModelParameterManifest {
            name: parameter.name.clone(),
            kind: parameter.kind.clone(),
            parameter_index: parameter.parameter_index as usize,
            default: parameter.default,
            pdg: None,
            runtime_name: parameter.runtime_name.clone(),
            complex_component: parameter.complex_component.clone(),
        })
        .collect::<Vec<_>>();
    let model_parameter_runtime_slots = runtime_parameter_slots(&model_parameters)?;
    let model_parameter_values_f64 = model_parameters
        .iter()
        .map(|parameter| parameter.default)
        .collect::<Vec<_>>();
    let model_parameter_name_to_index = model_parameters
        .iter()
        .map(|parameter| (parameter.name.clone(), parameter.parameter_index))
        .collect::<BTreeMap<_, _>>();
    let defaults = metadata
        .prepared_parameter_defaults
        .iter()
        .map(|[real, imaginary]| crate::EagerComplex64::new(*real, *imaginary))
        .collect::<Vec<_>>();
    let projection = metadata
        .parameter_projection
        .iter()
        .filter_map(|row| row.prepared_parameter_id.map(|prepared| (row, prepared)))
        .map(|(row, prepared)| {
            Ok(PreparedParameterProjectionEntry {
                runtime_slot: usize::try_from(row.runtime_slot).map_err(|_| {
                    RusticolError::artifact("on-the-fly runtime parameter slot exceeds usize")
                })?,
                prepared_slot: usize::try_from(prepared).map_err(|_| {
                    RusticolError::artifact("on-the-fly prepared parameter slot exceeds usize")
                })?,
                component: u8::try_from(row.component).map_err(|_| {
                    RusticolError::artifact("on-the-fly parameter component exceeds u8")
                })?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let normalization = &metadata.normalization;
    if !normalization.couplings_in_stage_evaluators {
        return Err(RusticolError::compatibility(
            "on-the-fly execution requires local vertex couplings in prepared kernel calls",
        ));
    }
    let normalization_factor = normalization.color_factor * normalization.global_coupling_factor
        / (normalization.average_factor * normalization.identical_factor);
    if !normalization_factor.is_finite() {
        return Err(RusticolError::integrity(
            "on-the-fly runtime normalization is not finite",
        ));
    }
    let particle_masses = metadata
        .particle_masses
        .iter()
        .map(|row| (row.outgoing_pdg, row.mass))
        .collect::<BTreeMap<_, _>>();
    let external_count = metadata.external_legs.len();
    let common = ExecutionRuntime {
        process: manifest.process.clone(),
        key: manifest.key.clone(),
        color_accuracy: manifest.color_accuracy.clone(),
        external_pdg_order: manifest.external_pdg_order.clone(),
        external_count,
        parameter_count: model_parameters.len(),
        value_parameter_count: 0,
        momentum_parameter_count: 0,
        current_count: 0,
        source_count: external_count,
        interaction_count: 0,
        stage_count: 0,
        amplitude_output_count: 0,
        lc_topology_replay_enabled: false,
        lc_topology_replay_mappings: Arc::new(Vec::new()),
        lc_topology_replay_public_mappings: Vec::new(),
        lc_topology_replay_routes: Vec::new(),
        lc_topology_replay_materialized_sector_ids: BTreeSet::new(),
        lc_resolved_replay_plan: None,
        lc_resolved_replay_selection_cache: None,
        lc_replay_flat_momenta_scratch: Vec::new(),
        lc_replay_target_components_scratch: Vec::new(),
        color_topology_replay_enabled: false,
        color_topology_replay_mappings: Arc::new(Vec::new()),
        color_replay_flat_momenta_scratch: Vec::new(),
        helicity_recurrence: None,
        compiled_helicity_execution_plan: None,
        compiled_color_execution_plan: None,
        compiled_direct_runtime: None,
        compiled_direct_color_schedules: BTreeMap::new(),
        compiled_direct_helicity_schedules: BTreeMap::new(),
        helicity_sum_runtime: None,
        helicity_selector_runtimes: Vec::new(),
        helicity_selector_runtime_schedule_modes: Vec::new(),
        helicity_selector_lane_by_domain: BTreeMap::new(),
        color_selector_runtimes: BTreeMap::new(),
        runtime_unavailable_message: None,
        sources: Vec::new(),
        momentum_slots: Vec::new(),
        external_is_initial: metadata
            .external_legs
            .iter()
            .map(|leg| leg.is_initial)
            .collect(),
        particle_masses,
        particle_mass_parameter_names: BTreeMap::new(),
        normalization_factor,
        normalization_color_factor: normalization.color_factor,
        normalization_average_factor: normalization.average_factor,
        normalization_identical_factor: normalization.identical_factor,
        normalization_qcd_coupling_power: normalization.qcd_coupling_power.unwrap_or(0) as usize,
        normalization_electroweak_coupling_power: normalization
            .electroweak_coupling_power
            .unwrap_or(0) as usize,
        model_parameters,
        model_parameter_name_to_index,
        model_parameter_runtime_slots,
        model_parameter_values_f64,
        model_parameter_evaluator: None,
        physics_reduction_override: None,
        physics: None,
        stages: None,
        amplitude_stage: None,
        state_scratch_f64: Vec::new(),
        state_scratch_f64_requires_clear: false,
        values_scratch_f64: Vec::new(),
    };
    Ok((common, defaults, projection))
}
