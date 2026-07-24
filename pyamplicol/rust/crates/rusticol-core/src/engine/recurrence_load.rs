// SPDX-License-Identifier: 0BSD

//! Native loading of compact recurrence artifacts.

use super::eager_manifest::PreparedKernelPackManifest;
use super::evaluator::recurrence_source_direct::{
    DirectSourceDispatchDomainSpec, DirectSourceOrientation, DirectSourceTemplateSpec,
    DirectSourceWavefunctionFamily,
};
use super::recurrence_backend::NativeRecurrenceDirectExecutorBackend;
use super::recurrence_manifest::*;
use super::*;
use crate::pacbin::{PacbinMemberKind, PacbinReader};
use crate::recurrence::{
    DirectRecurrencePlan, RECURRENCE_DIRECT_PLAN_MEMBER, decode_recurrence_direct_plan_v2,
};

pub(super) struct LoadedRecurrenceRuntime {
    pub(super) common: ExecutionRuntime,
    pub(super) lane: RecurrenceNativeRuntime,
}

pub(super) fn load_recurrence_native_runtime(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &RecurrenceExecutionManifest,
    physics: &ProcessPhysicsV1,
) -> RusticolResult<LoadedRecurrenceRuntime> {
    let (pack_bytes, pack, payload_root) = load_prepared_pack(artifact, manifest)?;
    let plan = load_plan(artifact, evaluator_root, manifest)?;
    let (mut common, parameter_defaults, parameter_projection, source_domains) =
        build_common_runtime(&plan, manifest, physics)?;
    let loaded_backend = NativeRecurrenceDirectExecutorBackend::load_from_verified_artifact(
        &pack_bytes,
        artifact,
        &payload_root,
        &plan,
        &manifest.prepared_kernel_pack_digest,
        &manifest.direct_template_catalog_digest,
        source_domains,
    )?;
    let (executors, backend_owners) = loaded_backend.into_parts();
    let kernel_payloads = artifact.evaluator_payload_store(&payload_root)?;
    common.model_parameter_evaluator =
        super::eager_load::load_prepared_model_parameter_evaluator_for_runtime(
            &pack,
            &common.model_parameters,
            &kernel_payloads,
        )?;
    common.refresh_derived_model_parameters()?;
    let public_flow_ids = public_flow_ids(&plan, &manifest.runtime_metadata, physics)?;
    let direct_helicity_to_physics = direct_helicity_to_physics(&plan, physics)?;
    let lane = RecurrenceNativeRuntime::new(
        plan,
        executors,
        backend_owners,
        parameter_defaults,
        parameter_projection,
        public_flow_ids,
        direct_helicity_to_physics,
    )?;
    Ok(LoadedRecurrenceRuntime { common, lane })
}

pub(super) fn load_recurrence_exact_sections(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &RecurrenceExecutionManifest,
    physics: &ProcessPhysicsV1,
) -> RusticolResult<NativeRecurrenceExactSections> {
    let (_pack_bytes, pack, _payload_root) = load_prepared_pack(artifact, manifest)?;
    let plan = load_plan(artifact, evaluator_root, manifest)?;
    let direct = pack.recurrence_direct_template_catalog(
        &manifest.prepared_kernel_pack_digest,
        &manifest.direct_template_catalog_digest,
    )?;
    if direct.catalog_digest != plan.direct_template_catalog_digest().to_string() {
        return Err(RusticolError::integrity(
            "prepared Direct-Arena catalog digest does not match the recurrence plan",
        ));
    }
    let public_flow_ids = public_flow_ids(&plan, &manifest.runtime_metadata, physics)?;
    let exact_factors = plan
        .exact_factors()
        .iter()
        .map(|factor| NativeRecurrenceExactFactor {
            real_numerator: factor.real().numerator().to_string(),
            real_denominator: factor.real().denominator().to_string(),
            imaginary_numerator: factor.imag().numerator().to_string(),
            imaginary_denominator: factor.imag().denominator().to_string(),
        })
        .collect();
    let executors = direct
        .templates
        .into_iter()
        .map(|template| NativeRecurrenceExactExecutor {
            direct_executor_id: template.direct_executor_id,
            role: template.role,
            destination_operation: template.destination_operation,
            parent_component_counts: template.parent_component_counts,
            destination_component_count: template.destination_component_count,
            momentum_operand_count: template.momentum_operand_count,
            prepared_kernel_id: template.payload_binding.prepared_kernel_id,
            runtime_template: template.payload_binding.runtime_template,
        })
        .collect();
    Ok(NativeRecurrenceExactSections {
        process_id: manifest.key.clone(),
        strategy: plan.strategy().as_str().to_string(),
        semantic_digest: plan.semantic_digest().to_string(),
        runtime_layout_digest: plan.runtime_layout_digest().to_string(),
        current_arena_components: plan.current_arena_components(),
        amplitude_destination_count: plan.amplitude_destination_count(),
        parameter_value_count: plan.parameter_value_count(),
        external_source_count: plan.external_source_count(),
        currents: plan.currents().to_vec(),
        sources: plan.sources().to_vec(),
        contributions: plan.contributions().to_vec(),
        finalizations: plan.finalizations().to_vec(),
        closures: plan.closures().to_vec(),
        row_groups: plan.row_groups().to_vec(),
        momentum_forms: plan.momentum_forms().to_vec(),
        momentum_terms: plan.momentum_terms().to_vec(),
        replay_targets: plan.replay_targets().to_vec(),
        source_permutations: plan.source_permutations().to_vec(),
        amplitude_destinations: plan.amplitude_destinations().to_vec(),
        resolved_helicities: plan.resolved_helicities().to_vec(),
        public_helicities: plan.public_helicities().to_vec(),
        source_state_assignments: plan.source_state_assignments().to_vec(),
        source_dispatch_variants: plan.source_dispatch_variants().to_vec(),
        source_embeddings: plan.source_embeddings().to_vec(),
        source_projections: plan.source_projections().to_vec(),
        resolved_source_selections: plan.resolved_source_selections().to_vec(),
        exact_factors,
        public_flow_ids,
        executors,
    })
}

fn public_flow_ids(
    plan: &DirectRecurrencePlan,
    metadata: &RecurrenceRuntimeMetadata,
    physics: &ProcessPhysicsV1,
) -> RusticolResult<Vec<u32>> {
    if metadata.public_color_flows.len() != physics.color_components.len() {
        return Err(RusticolError::integrity(
            "recurrence public color-flow bindings do not cover the physics axis",
        ));
    }
    let available = match plan.strategy() {
        crate::recurrence::RecurrenceStrategy::TopologyReplay => {
            let available = plan
                .replay_targets()
                .iter()
                .map(|target| target.public_flow_id)
                .collect::<BTreeSet<_>>();
            if available.len() != plan.replay_targets().len() {
                return Err(RusticolError::integrity(
                    "recurrence direct plan repeats a public replay target",
                ));
            }
            available
        }
        crate::recurrence::RecurrenceStrategy::AllFlowUnion => {
            let available = plan
                .amplitude_destinations()
                .iter()
                .map(|destination| destination.target_sector_id)
                .collect::<BTreeSet<_>>();
            if available.len() != plan.amplitude_destinations().len() {
                return Err(RusticolError::integrity(
                    "all-flow-union direct plan repeats a physical-flow destination",
                ));
            }
            available
        }
    };
    let mut seen = BTreeSet::new();
    let result = metadata
        .public_color_flows
        .iter()
        .zip(&physics.color_components)
        .map(|(binding, component)| {
            if binding.public_id != component.id() {
                return Err(RusticolError::integrity(
                    "recurrence public color-flow binding order disagrees with physics.json",
                ));
            }
            if !matches!(component, PhysicsColorComponentV1::LcFlow(_)) {
                return Err(RusticolError::integrity(
                    "recurrence public color-flow binding references a non-LC component",
                ));
            }
            if binding.target_sector_id >= plan.physical_sector_count()
                || !available.contains(&binding.target_sector_id)
                || !seen.insert(binding.target_sector_id)
            {
                return Err(RusticolError::integrity(
                    "recurrence public color-flow target is absent or repeated in the direct plan",
                ));
            }
            Ok(binding.target_sector_id)
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    if result.len() != available.len() {
        return Err(RusticolError::integrity(
            "recurrence direct-plan flow destinations do not match the public color-flow axis",
        ));
    }
    Ok(result)
}

fn direct_helicity_to_physics(
    plan: &DirectRecurrencePlan,
    physics: &ProcessPhysicsV1,
) -> RusticolResult<Vec<usize>> {
    let mut result = Vec::with_capacity(plan.resolved_helicities().len());
    let mut seen_physics = BTreeSet::new();
    for (expected_id, descriptor) in plan.resolved_helicities().iter().enumerate() {
        if descriptor.id as usize != expected_id {
            return Err(RusticolError::integrity(
                "recurrence resolved-helicity IDs are not dense and ordered",
            ));
        }
        let start = usize::try_from(descriptor.public_helicity_start).map_err(|_| {
            RusticolError::artifact("recurrence public-helicity offset exceeds usize")
        })?;
        let count = usize::try_from(descriptor.public_helicity_count).map_err(|_| {
            RusticolError::artifact("recurrence public-helicity count exceeds usize")
        })?;
        let stop = start
            .checked_add(count)
            .ok_or_else(|| RusticolError::artifact("recurrence public-helicity range overflows"))?;
        let values = plan.public_helicities().get(start..stop).ok_or_else(|| {
            RusticolError::integrity(
                "recurrence resolved-helicity public vector is outside the direct plan",
            )
        })?;
        if values.len() != physics.external_particles.len() {
            return Err(RusticolError::integrity(
                "recurrence resolved-helicity width disagrees with the physics axis",
            ));
        }
        let matches = physics
            .helicities
            .iter()
            .enumerate()
            .filter_map(|(index, helicity)| (helicity.values == values).then_some(index))
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(RusticolError::integrity(format!(
                "recurrence resolved helicity {expected_id} maps to {} physics helicities",
                matches.len()
            )));
        }
        let physics_index = matches[0];
        if !seen_physics.insert(physics_index) {
            return Err(RusticolError::integrity(
                "recurrence resolved helicities repeat a physics helicity",
            ));
        }
        result.push(physics_index);
    }
    Ok(result)
}

fn load_prepared_pack(
    artifact: &VerifiedArtifact,
    manifest: &RecurrenceExecutionManifest,
) -> RusticolResult<(Vec<u8>, PreparedKernelPackManifest, PathBuf)> {
    let manifest_path = confined_internal_path(
        &manifest.kernel_pack.manifest_path,
        "recurrence prepared kernel-pack manifest path",
    )?;
    let manifest_path = manifest_path.to_str().ok_or_else(|| {
        RusticolError::security("recurrence kernel-pack manifest path is not valid UTF-8")
    })?;
    if artifact.payload(manifest_path)?.role != PayloadRole::EvaluatorManifest {
        return Err(RusticolError::security(
            "recurrence kernel-pack manifest is not an evaluator-manifest payload",
        ));
    }
    let bytes = artifact.read_payload(manifest_path)?;
    let pack: PreparedKernelPackManifest = serde_json::from_slice(&bytes).map_err(|error| {
        RusticolError::serialization(format!(
            "could not parse recurrence prepared kernel pack: {error}"
        ))
    })?;
    pack.validate()?;
    let payload_root = artifact.root().join(confined_internal_path(
        &manifest.kernel_pack.payload_root,
        "recurrence prepared kernel payload root",
    )?);
    Ok((bytes, pack, payload_root))
}

fn load_plan(
    artifact: &VerifiedArtifact,
    evaluator_root: &Path,
    manifest: &RecurrenceExecutionManifest,
) -> RusticolResult<DirectRecurrencePlan> {
    let container = &manifest.plan.runtime_container;
    let path = evaluator_root.join(&container.path);
    let relative = path.strip_prefix(artifact.root()).map_err(|_| {
        RusticolError::security("recurrence runtime container escapes the artifact root")
    })?;
    let logical_path = relative
        .components()
        .map(|component| match component {
            std::path::Component::Normal(part) => part.to_str().ok_or_else(|| {
                RusticolError::security("recurrence runtime container path is not valid UTF-8")
            }),
            _ => Err(RusticolError::security(
                "recurrence runtime container path is not canonical",
            )),
        })
        .collect::<RusticolResult<Vec<_>>>()?
        .join("/");
    let payload = artifact.payload(&logical_path)?;
    if payload.role != PayloadRole::EvaluatorState
        || payload.media_type != "application/octet-stream"
        || payload.process_id.as_deref() != Some(manifest.key.as_str())
        || payload.executable
        || payload.size_bytes != container.size_bytes
        || payload.sha256 != container.sha256
    {
        return Err(RusticolError::integrity(
            "recurrence runtime container disagrees with its authenticated payload",
        ));
    }

    let reader = PacbinReader::open(&path)?;
    let index = reader.index();
    if index.file_size() != container.size_bytes
        || reader.container_size() as u64 != container.size_bytes
        || index.index_sha256().as_slice() != decode_sha256(&container.index_sha256)?.as_slice()
        || index.members().len() as u64 != container.member_count
        || index
            .members()
            .iter()
            .map(|member| member.length())
            .sum::<u64>()
            != container.unpacked_size_bytes
    {
        return Err(RusticolError::integrity(
            "recurrence runtime PACBIN metadata disagrees with execution.json",
        ));
    }
    let member = reader.member(RECURRENCE_DIRECT_PLAN_MEMBER)?;
    if member.kind() != PacbinMemberKind::RecurrenceDirectPlan {
        return Err(RusticolError::compatibility(
            "recurrence runtime PACBIN contains an incompatible plan member",
        ));
    }
    let bytes = reader.member_bytes(RECURRENCE_DIRECT_PLAN_MEMBER)?;
    let plan = decode_recurrence_direct_plan_v2(bytes)?;
    if plan.prepared_pack_digest().to_string() != manifest.prepared_kernel_pack_digest
        || plan.direct_template_catalog_digest().to_string()
            != manifest.direct_template_catalog_digest
    {
        return Err(RusticolError::integrity(
            "direct recurrence plan authentication digests disagree with execution.json",
        ));
    }
    Ok(plan)
}

fn build_common_runtime(
    plan: &DirectRecurrencePlan,
    manifest: &RecurrenceExecutionManifest,
    physics: &ProcessPhysicsV1,
) -> RusticolResult<(
    ExecutionRuntime,
    Vec<crate::EagerComplex64>,
    Vec<RecurrenceParameterProjectionEntry>,
    Vec<DirectSourceDispatchDomainSpec>,
)> {
    let metadata = &manifest.runtime_metadata;
    let runtime_parameters = metadata
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
    let model_parameter_runtime_slots = runtime_parameter_slots(&runtime_parameters)?;
    let model_parameter_values_f64 = runtime_parameters
        .iter()
        .map(|parameter| parameter.default)
        .collect::<Vec<_>>();
    let model_parameter_name_to_index = runtime_parameters
        .iter()
        .map(|parameter| (parameter.name.clone(), parameter.parameter_index))
        .collect::<BTreeMap<_, _>>();

    let parameter_defaults = metadata
        .prepared_parameter_defaults
        .iter()
        .map(|[real, imaginary]| crate::EagerComplex64::new(*real, *imaginary))
        .collect::<Vec<_>>();
    if parameter_defaults.len()
        != usize::try_from(plan.parameter_value_count())
            .map_err(|_| RusticolError::artifact("recurrence parameter count exceeds usize"))?
    {
        return Err(RusticolError::integrity(
            "recurrence prepared defaults do not match the direct plan",
        ));
    }
    let parameter_projection = metadata
        .parameter_projection
        .iter()
        .filter_map(|row| {
            row.prepared_parameter_id
                .map(|prepared_slot| (row, prepared_slot))
        })
        .map(|(row, prepared_slot)| {
            Ok(RecurrenceParameterProjectionEntry {
                runtime_slot: usize::try_from(row.runtime_slot).map_err(|_| {
                    RusticolError::artifact("recurrence runtime parameter slot exceeds usize")
                })?,
                prepared_slot: usize::try_from(prepared_slot).map_err(|_| {
                    RusticolError::artifact("recurrence prepared parameter slot exceeds usize")
                })?,
                component: u8::try_from(row.component).map_err(|_| {
                    RusticolError::artifact("recurrence parameter component exceeds u8")
                })?,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;

    let mut particle_masses = metadata
        .particle_masses
        .iter()
        .map(|row| (row.outgoing_pdg, row.mass))
        .collect::<BTreeMap<_, _>>();
    let mut particle_mass_parameter_names = BTreeMap::new();
    for source in &metadata.source_templates {
        let Some(name) = source.source_ir.mass_parameter.as_ref() else {
            continue;
        };
        for pdg in [
            source.source_ir.identity.pdg_label,
            source.source_ir.identity.anti_pdg_label,
        ] {
            particle_mass_parameter_names.insert(pdg, name.clone());
            if let Some(slots) = model_parameter_runtime_slots.get(name)
                && let Some(mass) = model_parameter_values_f64.get(slots.real)
            {
                particle_masses.insert(pdg, *mass);
            }
        }
    }

    let source_domains =
        build_direct_source_domains(plan, metadata, &model_parameter_runtime_slots)?;
    let normalization = &metadata.normalization;
    if !normalization.couplings_in_stage_evaluators {
        return Err(RusticolError::compatibility(
            "recurrence execution requires local vertex couplings in prepared kernel calls",
        ));
    }
    let normalization_factor = normalization.color_factor * normalization.global_coupling_factor
        / (normalization.average_factor * normalization.identical_factor);
    if !normalization_factor.is_finite() {
        return Err(RusticolError::integrity(
            "recurrence runtime normalization is not finite",
        ));
    }
    let external_count = manifest.external_pdg_order.len();
    if plan.external_source_count() as usize != external_count
        || metadata.external_legs.len() != external_count
    {
        return Err(RusticolError::integrity(
            "recurrence direct plan external-source count disagrees with process metadata",
        ));
    }
    let external_is_initial = physics
        .external_particles
        .iter()
        .map(|particle| particle.role == crate::ParticleRole::Initial)
        .collect();
    let source_count = plan.sources().len();
    let current_count = plan.currents().len();
    let interaction_count = plan.contributions().len();
    let stage_count = plan
        .row_groups()
        .iter()
        .map(|group| usize::from(group.stage))
        .max()
        .and_then(|stage| stage.checked_add(1))
        .ok_or_else(|| RusticolError::integrity("recurrence direct plan has no row groups"))?;
    let amplitude_output_count = plan.amplitude_destinations().len();
    let common = ExecutionRuntime {
        process: manifest.process.clone(),
        key: manifest.key.clone(),
        color_accuracy: manifest.color_accuracy.clone(),
        external_pdg_order: manifest.external_pdg_order.clone(),
        external_count,
        parameter_count: runtime_parameters.len(),
        value_parameter_count: 0,
        momentum_parameter_count: 0,
        current_count,
        source_count,
        interaction_count,
        stage_count,
        amplitude_output_count,
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
        // Direct-Arena source execution owns typed SourceIR dispatch domains.
        // The legacy source records are intentionally not reconstructed.
        sources: Vec::new(),
        momentum_slots: Vec::new(),
        external_is_initial,
        particle_masses,
        particle_mass_parameter_names,
        normalization_factor,
        normalization_color_factor: normalization.color_factor,
        normalization_average_factor: normalization.average_factor,
        normalization_identical_factor: normalization.identical_factor,
        normalization_qcd_coupling_power: normalization.qcd_coupling_power.unwrap_or(0) as usize,
        normalization_electroweak_coupling_power: normalization
            .electroweak_coupling_power
            .unwrap_or(0) as usize,
        model_parameters: runtime_parameters,
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
    Ok((
        common,
        parameter_defaults,
        parameter_projection,
        source_domains,
    ))
}

fn build_direct_source_domains(
    plan: &DirectRecurrencePlan,
    metadata: &RecurrenceRuntimeMetadata,
    runtime_parameter_slots: &BTreeMap<String, RuntimeParameterSlots>,
) -> RusticolResult<Vec<DirectSourceDispatchDomainSpec>> {
    let templates = metadata
        .source_templates
        .iter()
        .map(|source| (source.source_template_id, source))
        .collect::<BTreeMap<_, _>>();
    let legs = metadata
        .external_legs
        .iter()
        .map(|leg| (leg.source_slot, leg))
        .collect::<BTreeMap<_, _>>();
    let domain_count = usize::try_from(plan.source_template_or_dispatch_count())
        .map_err(|_| RusticolError::artifact("recurrence source-domain count exceeds usize"))?;
    let mut variants = vec![BTreeMap::<i32, DirectSourceTemplateSpec>::new(); domain_count];

    match plan.strategy() {
        crate::recurrence::RecurrenceStrategy::TopologyReplay => {
            for row in plan.sources() {
                let source = templates
                    .get(&row.source_template_or_dispatch_domain)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence direct plan references an absent SourceIR template",
                        )
                    })?;
                let leg = legs.get(&row.source_slot).ok_or_else(|| {
                    RusticolError::integrity(
                        "recurrence direct source references an absent external leg",
                    )
                })?;
                let spec = direct_source_template_spec(
                    source,
                    leg,
                    metadata,
                    runtime_parameter_slots,
                    plan.parameter_value_count(),
                )?;
                if row.spin_state_class != spec.spin_state_class {
                    return Err(RusticolError::integrity(
                        "recurrence direct source spin class disagrees with crossed SourceIR metadata",
                    ));
                }
                insert_source_domain_variant(
                    &mut variants,
                    row.source_template_or_dispatch_domain,
                    row.spin_state_class,
                    spec,
                )?;
            }
        }
        crate::recurrence::RecurrenceStrategy::AllFlowUnion => {
            for variant in plan.source_dispatch_variants() {
                let row = plan
                    .sources()
                    .get(variant.source_row_id as usize)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence source variant references an absent source row",
                        )
                    })?;
                let source = templates.get(&variant.source_template_id).ok_or_else(|| {
                    RusticolError::integrity(
                        "recurrence source variant references an absent SourceIR template",
                    )
                })?;
                let leg = legs.get(&row.source_slot).ok_or_else(|| {
                    RusticolError::integrity(
                        "recurrence source variant references an absent external leg",
                    )
                })?;
                let spec = direct_source_template_spec(
                    source,
                    leg,
                    metadata,
                    runtime_parameter_slots,
                    plan.parameter_value_count(),
                )?;
                if row.source_template_or_dispatch_domain != variant.dispatch_domain_id
                    || spec.spin_state_class != variant.crossed_spin_state_class
                {
                    return Err(RusticolError::integrity(
                        "recurrence union source variant disagrees with its dispatch domain or crossed SourceIR state",
                    ));
                }
                insert_source_domain_variant(
                    &mut variants,
                    variant.dispatch_domain_id,
                    variant.crossed_spin_state_class,
                    spec,
                )?;
            }
        }
    }

    let inert = variants
        .iter()
        .flat_map(BTreeMap::values)
        .next()
        .copied()
        .ok_or_else(|| RusticolError::integrity("recurrence direct plan has no source rows"))?;
    Ok(variants
        .into_iter()
        .map(|domain| DirectSourceDispatchDomainSpec {
            // Prepared source-template IDs are model-global and may be sparse
            // for one process. Unreferenced slots remain inert but preserve the
            // stable IDs addressed by DirectSourceRow.
            variants: if domain.is_empty() {
                vec![inert]
            } else {
                domain.into_values().collect()
            },
        })
        .collect())
}

fn insert_source_domain_variant(
    domains: &mut [BTreeMap<i32, DirectSourceTemplateSpec>],
    domain_id: u32,
    spin_state_class: i32,
    spec: DirectSourceTemplateSpec,
) -> RusticolResult<()> {
    let domain = domains.get_mut(domain_id as usize).ok_or_else(|| {
        RusticolError::integrity("recurrence direct source-domain ID is out of bounds")
    })?;
    if let Some(previous) = domain.insert(spin_state_class, spec)
        && previous != spec
    {
        return Err(RusticolError::integrity(
            "recurrence source domain maps one spin class to different SourceIR semantics",
        ));
    }
    Ok(())
}

fn direct_source_template_spec(
    source: &RecurrenceSourceTemplate,
    leg: &RecurrenceExternalLeg,
    metadata: &RecurrenceRuntimeMetadata,
    runtime_parameter_slots: &BTreeMap<String, RuntimeParameterSlots>,
    parameter_count: u32,
) -> RusticolResult<DirectSourceTemplateSpec> {
    if source.source_ir.identity.pdg_label != leg.outgoing_pdg {
        return Err(RusticolError::integrity(
            "recurrence SourceIR particle disagrees with its external leg",
        ));
    }
    let crossing = if leg.is_initial {
        &source.crossing
    } else {
        &IDENTITY_RECURRENCE_CROSSING
    };
    let helicity = source
        .helicity
        .checked_mul(crossing.helicity_factor)
        .ok_or_else(|| RusticolError::integrity("recurrence source helicity overflows"))?;
    let chirality = source
        .chirality
        .checked_mul(crossing.chirality_factor)
        .ok_or_else(|| RusticolError::integrity("recurrence source chirality overflows"))?;
    let spin_state_class = source
        .spin_state
        .checked_mul(crossing.spin_state_factor)
        .ok_or_else(|| RusticolError::integrity("recurrence source spin state overflows"))?;
    let family = direct_source_family(&source.source_ir)?;
    let orientation = match source.source_ir.identity.orientation {
        RecurrenceSourceOrientation::Particle => DirectSourceOrientation::Particle,
        RecurrenceSourceOrientation::Antiparticle => DirectSourceOrientation::Antiparticle,
        RecurrenceSourceOrientation::SelfConjugate => DirectSourceOrientation::SelfConjugate,
    };
    let mass_parameter_index = source
        .source_ir
        .mass_parameter
        .as_deref()
        .map(|name| {
            prepared_parameter_index(
                name,
                runtime_parameter_slots,
                &metadata.parameter_projection,
                parameter_count,
            )
        })
        .transpose()?;
    if mass_parameter_index.is_none()
        && metadata
            .particle_masses
            .iter()
            .find(|mass| mass.outgoing_pdg == leg.outgoing_pdg)
            .is_some_and(|mass| mass.mass != 0.0)
    {
        return Err(RusticolError::compatibility(
            "massive recurrence SourceIR has no prepared mass-parameter slot",
        ));
    }
    Ok(DirectSourceTemplateSpec {
        spin_state_class,
        family,
        orientation,
        helicity,
        chirality,
        mass_parameter_index,
    })
}

const IDENTITY_RECURRENCE_CROSSING: RecurrenceGenericCrossingIr = RecurrenceGenericCrossingIr {
    momentum_transform: RecurrenceMomentumTransform::Identity,
    helicity_factor: 1,
    chirality_factor: 1,
    spin_state_factor: 1,
    phase: [1.0, 0.0],
};

fn direct_source_family(
    source: &RecurrenceGenericSourceIr,
) -> RusticolResult<DirectSourceWavefunctionFamily> {
    match (
        source.statistics,
        source.wavefunction_family,
        source.component_dimension,
    ) {
        (RecurrenceParticleStatistics::Boson, RecurrenceWavefunctionFamily::Scalar, 1) => {
            Ok(DirectSourceWavefunctionFamily::Scalar)
        }
        (RecurrenceParticleStatistics::Fermion, RecurrenceWavefunctionFamily::Fermion, 2) => {
            Ok(DirectSourceWavefunctionFamily::WeylFermion)
        }
        (RecurrenceParticleStatistics::Fermion, RecurrenceWavefunctionFamily::Fermion, 4) => {
            Ok(DirectSourceWavefunctionFamily::DiracFermion)
        }
        (RecurrenceParticleStatistics::Boson, RecurrenceWavefunctionFamily::Vector, 4) => {
            Ok(DirectSourceWavefunctionFamily::Vector)
        }
        (RecurrenceParticleStatistics::Boson, RecurrenceWavefunctionFamily::Spin2, 16) => {
            Ok(DirectSourceWavefunctionFamily::Spin2)
        }
        (_, RecurrenceWavefunctionFamily::Ghost, _)
        | (_, RecurrenceWavefunctionFamily::Auxiliary, _) => Err(RusticolError::compatibility(
            "Direct-Arena recurrence does not yet support ghost or auxiliary external sources",
        )),
        _ => Err(RusticolError::integrity(
            "recurrence SourceIR statistics, family, and component dimension are incompatible",
        )),
    }
}

fn prepared_parameter_index(
    name: &str,
    runtime_parameter_slots: &BTreeMap<String, RuntimeParameterSlots>,
    projection: &[RecurrenceParameterProjection],
    parameter_count: u32,
) -> RusticolResult<u32> {
    let runtime_slot = runtime_parameter_slots
        .get(name)
        .ok_or_else(|| {
            RusticolError::integrity(format!(
                "recurrence source mass parameter {name:?} has no runtime projection"
            ))
        })?
        .real;
    let row = projection
        .iter()
        .find(|row| row.runtime_slot as usize == runtime_slot && row.component == 0)
        .ok_or_else(|| {
            RusticolError::integrity(
                "recurrence source mass parameter has no real prepared projection",
            )
        })?;
    if row.runtime_name != name {
        return Err(RusticolError::integrity(
            "recurrence source mass projection is not its real component",
        ));
    }
    let prepared = row.prepared_parameter_id.ok_or_else(|| {
        RusticolError::compatibility(format!(
            "recurrence source mass parameter {name:?} has no prepared parameter slot"
        ))
    })?;
    if prepared >= parameter_count {
        return Err(RusticolError::integrity(
            "recurrence source mass prepared slot exceeds the direct plan",
        ));
    }
    Ok(prepared)
}

fn runtime_parameter_slots(
    parameters: &[GenericRuntimeModelParameterManifest],
) -> RusticolResult<BTreeMap<String, RuntimeParameterSlots>> {
    let mut result = BTreeMap::new();
    let mut complex = BTreeMap::<String, (Option<usize>, Option<usize>)>::new();
    for parameter in parameters {
        if let Some(name) = &parameter.runtime_name {
            let slots = complex.entry(name.clone()).or_default();
            let target = match parameter.complex_component.as_deref() {
                Some("real") => &mut slots.0,
                Some("imag") => &mut slots.1,
                _ => {
                    return Err(RusticolError::integrity(
                        "recurrence complex parameter lacks component metadata",
                    ));
                }
            };
            if target.replace(parameter.parameter_index).is_some() {
                return Err(RusticolError::integrity(
                    "recurrence complex parameter repeats a component",
                ));
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
            return Err(RusticolError::integrity(
                "recurrence runtime parameter names are not unique",
            ));
        }
    }
    for (name, (real, imaginary)) in complex {
        let real = real.ok_or_else(|| {
            RusticolError::integrity("recurrence complex parameter lacks a real component")
        })?;
        if result
            .insert(name, RuntimeParameterSlots { real, imaginary })
            .is_some()
        {
            return Err(RusticolError::integrity(
                "recurrence runtime parameter names conflict",
            ));
        }
    }
    Ok(result)
}

fn decode_sha256(value: &str) -> RusticolResult<[u8; 32]> {
    if value.len() != 64 {
        return Err(RusticolError::integrity(
            "recurrence SHA-256 has an invalid encoded length",
        ));
    }
    let mut output = [0u8; 32];
    for (index, byte) in output.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16).map_err(|_| {
            RusticolError::integrity("recurrence SHA-256 is not lowercase hexadecimal")
        })?;
    }
    Ok(output)
}
