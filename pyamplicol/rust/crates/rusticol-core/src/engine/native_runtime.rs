// SPDX-License-Identifier: 0BSD

use super::*;

struct PointSelectorProfileCounts {
    gather_point_count: usize,
    input_bytes_per_point: usize,
    scatter_value_count: usize,
}

type ProfiledPreparedF64Batch = (Vec<Vec<[f64; 4]>>, Duration, Duration);

fn attach_point_selector_profile(
    profile: &mut NativeRuntimeProfile,
    plan: &PointSelectorPlanProfile,
    planner: Duration,
    gather: Duration,
    scatter: Duration,
    counts: PointSelectorProfileCounts,
) {
    profile.selector_planner_s = profile_duration_seconds(planner);
    profile.selector_gather_s = profile_duration_seconds(gather);
    profile.selector_scatter_s = profile_duration_seconds(scatter);
    profile.selector_plan_kind = plan.kind.to_string();
    profile.selector_group_sizes.clone_from(&plan.group_sizes);
    profile.selector_reordered_point_count = plan.reordered_point_count;
    profile.selector_simd_lane_width = plan.simd_lane_width;
    profile.selector_simd_occupancy = plan.simd_occupancy;
    profile.selector_gather_point_count = counts.gather_point_count as u64;
    profile.selector_gather_bytes =
        (counts.gather_point_count * counts.input_bytes_per_point) as u64;
    profile.selector_scatter_value_count = counts.scatter_value_count as u64;
}

fn execution_uses_simd_jit(runtime: &ExecutionRuntime) -> bool {
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    if runtime.compiled_direct_runtime.is_some() {
        return true;
    }
    runtime
        .stages
        .as_ref()
        .is_some_and(|stages| stages.iter().any(|stage| stage.evaluator.uses_simd_jit()))
        || runtime
            .amplitude_stage
            .as_ref()
            .and_then(|stage| stage.evaluator.as_ref())
            .is_some_and(EvaluatorGroup::uses_simd_jit)
        || runtime
            .helicity_sum_runtime
            .as_deref()
            .is_some_and(execution_uses_simd_jit)
        || runtime
            .helicity_selector_runtimes
            .iter()
            .any(|lane| execution_uses_simd_jit(lane))
        || runtime
            .color_selector_runtimes
            .values()
            .any(|lane| execution_uses_simd_jit(lane))
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
#[derive(Clone, Copy, Debug, Default)]
struct CompiledDirectProfileSnapshot {
    engine_count: usize,
    source_fill_bytes: u64,
    momentum_fill_bytes: u64,
    parameter_fill_bytes: u64,
    amplitude_clear_bytes: u64,
    backend_call_count: u64,
    boundary_input_bytes: u64,
    boundary_current_output_bytes: u64,
    boundary_amplitude_output_bytes: u64,
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn compiled_direct_profile_snapshot(
    runtime: &ExecutionRuntime,
) -> RusticolResult<CompiledDirectProfileSnapshot> {
    let mut snapshot = CompiledDirectProfileSnapshot::default();
    if let Some(direct) = runtime.compiled_direct_runtime.as_ref() {
        let traffic = direct.traffic();
        traffic.leaf.validate_direct()?;
        if traffic.boundary_input_bytes
            | traffic.boundary_current_output_bytes
            | traffic.boundary_amplitude_output_bytes
            != 0
        {
            return Err(RusticolError::integrity(
                "production compiled Direct-Arena profile observed developer boundary traffic",
            ));
        }
        snapshot.engine_count += 1;
        snapshot.source_fill_bytes = traffic.source_fill_bytes;
        snapshot.momentum_fill_bytes = traffic.momentum_fill_bytes;
        snapshot.parameter_fill_bytes = traffic.parameter_fill_bytes;
        snapshot.amplitude_clear_bytes = traffic.amplitude_clear_bytes;
        snapshot.backend_call_count = traffic.leaf.calls;
        snapshot.boundary_input_bytes = traffic.boundary_input_bytes;
        snapshot.boundary_current_output_bytes = traffic.boundary_current_output_bytes;
        snapshot.boundary_amplitude_output_bytes = traffic.boundary_amplitude_output_bytes;
    }
    let children = runtime
        .helicity_sum_runtime
        .iter()
        .map(Box::as_ref)
        .chain(runtime.helicity_selector_runtimes.iter().map(Box::as_ref))
        .chain(runtime.color_selector_runtimes.values().map(Box::as_ref));
    for child in children {
        let child = compiled_direct_profile_snapshot(child)?;
        snapshot.engine_count += child.engine_count;
        snapshot.source_fill_bytes = snapshot
            .source_fill_bytes
            .saturating_add(child.source_fill_bytes);
        snapshot.momentum_fill_bytes = snapshot
            .momentum_fill_bytes
            .saturating_add(child.momentum_fill_bytes);
        snapshot.parameter_fill_bytes = snapshot
            .parameter_fill_bytes
            .saturating_add(child.parameter_fill_bytes);
        snapshot.amplitude_clear_bytes = snapshot
            .amplitude_clear_bytes
            .saturating_add(child.amplitude_clear_bytes);
        snapshot.backend_call_count = snapshot
            .backend_call_count
            .saturating_add(child.backend_call_count);
        snapshot.boundary_input_bytes = snapshot
            .boundary_input_bytes
            .saturating_add(child.boundary_input_bytes);
        snapshot.boundary_current_output_bytes = snapshot
            .boundary_current_output_bytes
            .saturating_add(child.boundary_current_output_bytes);
        snapshot.boundary_amplitude_output_bytes = snapshot
            .boundary_amplitude_output_bytes
            .saturating_add(child.boundary_amplitude_output_bytes);
    }
    Ok(snapshot)
}

fn ensure_selected_runtime_capabilities_supported(capabilities: &[String]) -> RusticolResult<()> {
    if capabilities.iter().any(|capability| {
        capability == EAGER_DAG_RUNTIME_CAPABILITY
            || capability == EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY
    }) {
        return Err(RusticolError::compatibility(
            "legacy eager plan-v2 artifacts are no longer executable; regenerate the artifact \
             with the current `pyamplicol generate` before loading it",
        ));
    }
    ensure_runtime_capabilities_supported(capabilities.iter().map(String::as_str))
}

impl NativeRuntime {
    pub const ABI_VERSION: u32 = crate::C_ABI_VERSION;

    /// Content identity of the authenticated artifact manifest that supplied
    /// this in-memory runtime.
    pub fn artifact_id(&self) -> &str {
        &self.artifact_id
    }

    /// Load only the authenticated physical reduction groups of an eager
    /// plan-v3 artifact.
    ///
    /// Unlike the exact-section bridge, this does not materialize exact
    /// invocations, attachments, finalizations, or closures that are unrelated
    /// to reduction authentication.
    #[doc(hidden)]
    pub fn load_eager_reduction_groups(
        artifact_path: impl AsRef<Path>,
        process_id: &str,
    ) -> Result<Value, RusticolError> {
        let artifact = VerifiedArtifact::open_with_manifest_preflight(artifact_path, |manifest| {
            let selection = manifest.select_process(Some(process_id))?;
            ensure_selected_runtime_capabilities_supported(
                &selection.process.required_runtime_capabilities,
            )
        })?;
        let selection = artifact.select_process(Some(process_id))?;
        if selection.alias.is_some() {
            return Err(RusticolError::invalid_argument(
                "compact eager reduction groups must be requested by representative process ID",
            ));
        }
        let (manifest, evaluator_root) = load_verified_evaluator(&artifact, &selection)?;
        let physics_bytes = artifact.read_payload(&selection.process.physics_path)?;
        let mut physics =
            ProcessPhysicsV1::from_json(&physics_bytes, &selection.process.physics_path)?;
        if physics.process_id != selection.process.id
            || physics.process != selection.process.expression
            || physics.color_accuracy.as_str() != selection.process.color_accuracy
            || physics
                .external_particles
                .iter()
                .map(|particle| particle.pdg)
                .ne(selection.process.external_pdgs.iter().copied())
        {
            return Err(RusticolError::integrity(format!(
                "runtime physics payload {:?} does not match process {:?}",
                selection.process.physics_path, selection.process.id
            )));
        }
        match manifest {
            LoadedExecutionManifest::Compiled(_) => Err(RusticolError::compatibility(
                "compact reduction-group loading requires an eager plan-v3 artifact",
            )),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::EagerV3(manifest) => {
                super::eager_v3_load::load_eager_v3_reduction_groups(
                    &artifact,
                    &evaluator_root,
                    &manifest,
                    &mut physics,
                )
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::Recurrence(_) => Err(RusticolError::compatibility(
                "recurrence reduction-group loading is unavailable through the eager bridge",
            )),
        }
    }

    /// Load only the exact-required sections of an eager plan-v3 artifact.
    ///
    /// This authenticates the artifact and compact runtime container but does
    /// not instantiate the prepared f64 backend. It is intentionally separate
    /// from the normal runtime hot path.
    pub fn load_eager_exact_sections(
        artifact_path: impl AsRef<Path>,
        process_id: &str,
    ) -> Result<NativeEagerExactSections, RusticolError> {
        let artifact = VerifiedArtifact::open_with_manifest_preflight(artifact_path, |manifest| {
            let selection = manifest.select_process(Some(process_id))?;
            ensure_selected_runtime_capabilities_supported(
                &selection.process.required_runtime_capabilities,
            )
        })?;
        let selection = artifact.select_process(Some(process_id))?;
        if selection.alias.is_some() {
            return Err(RusticolError::invalid_argument(
                "compact eager exact sections must be requested by representative process ID",
            ));
        }
        let (manifest, evaluator_root) = load_verified_evaluator(&artifact, &selection)?;
        let physics_bytes = artifact.read_payload(&selection.process.physics_path)?;
        let mut physics =
            ProcessPhysicsV1::from_json(&physics_bytes, &selection.process.physics_path)?;
        if physics.process_id != selection.process.id
            || physics.process != selection.process.expression
            || physics.color_accuracy.as_str() != selection.process.color_accuracy
            || physics
                .external_particles
                .iter()
                .map(|particle| particle.pdg)
                .ne(selection.process.external_pdgs.iter().copied())
        {
            return Err(RusticolError::integrity(format!(
                "runtime physics payload {:?} does not match process {:?}",
                selection.process.physics_path, selection.process.id
            )));
        }
        match manifest {
            LoadedExecutionManifest::Compiled(_) => Err(RusticolError::compatibility(
                "compact exact-section loading requires an eager plan-v3 artifact",
            )),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::EagerV3(manifest) => {
                super::eager_v3_load::load_eager_v3_exact_sections(
                    &artifact,
                    &evaluator_root,
                    &manifest,
                    &mut physics,
                )
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::Recurrence(_) => Err(RusticolError::compatibility(
                "recurrence exact-section loading is not available in the initial f64 runtime slice",
            )),
        }
    }

    /// Load the immutable exact-execution sections of an authenticated
    /// topology-replay recurrence artifact without instantiating its f64
    /// Direct-Arena backend.
    pub fn load_recurrence_exact_sections(
        artifact_path: impl AsRef<Path>,
        process_id: &str,
    ) -> Result<NativeRecurrenceExactSections, RusticolError> {
        let artifact = VerifiedArtifact::open_with_manifest_preflight(artifact_path, |manifest| {
            let selection = manifest.select_process(Some(process_id))?;
            ensure_selected_runtime_capabilities_supported(
                &selection.process.required_runtime_capabilities,
            )
        })?;
        let selection = artifact.select_process(Some(process_id))?;
        if selection.alias.is_some() {
            return Err(RusticolError::invalid_argument(
                "compact recurrence exact sections must be requested by representative process ID",
            ));
        }
        let (manifest, evaluator_root) = load_verified_evaluator(&artifact, &selection)?;
        let physics_bytes = artifact.read_payload(&selection.process.physics_path)?;
        let physics = ProcessPhysicsV1::from_json(&physics_bytes, &selection.process.physics_path)?;
        if physics.process_id != selection.process.id
            || physics.process != selection.process.expression
            || physics.color_accuracy.as_str() != selection.process.color_accuracy
            || physics
                .external_particles
                .iter()
                .map(|particle| particle.pdg)
                .ne(selection.process.external_pdgs.iter().copied())
        {
            return Err(RusticolError::integrity(format!(
                "runtime physics payload {:?} does not match process {:?}",
                selection.process.physics_path, selection.process.id
            )));
        }
        match manifest {
            LoadedExecutionManifest::Compiled(_) => Err(RusticolError::compatibility(
                "compact recurrence exact-section loading requires a recurrence artifact",
            )),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::EagerV3(_) => Err(RusticolError::compatibility(
                "compact recurrence exact-section loading requires a recurrence artifact",
            )),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            LoadedExecutionManifest::Recurrence(manifest) => {
                load_recurrence_exact_sections(&artifact, &evaluator_root, &manifest, &physics)
            }
        }
    }

    pub fn load(
        artifact_path: impl AsRef<Path>,
        process_id: Option<&str>,
        model_parameters_path: Option<&Path>,
    ) -> Result<Self, RusticolError> {
        let artifact = VerifiedArtifact::open_with_manifest_preflight(artifact_path, |manifest| {
            let selection = manifest.select_process(process_id)?;
            ensure_selected_runtime_capabilities_supported(
                &selection.process.required_runtime_capabilities,
            )
        })?;
        let artifact_id = artifact.manifest().artifact_id.clone();
        let selection = artifact.select_process(process_id)?;
        let (manifest, evaluator_root) = load_verified_evaluator(&artifact, &selection)?;
        let physics_bytes = artifact.read_payload(&selection.process.physics_path)?;
        let mut physics_v1 =
            ProcessPhysicsV1::from_json(&physics_bytes, &selection.process.physics_path)?;
        if physics_v1.process_id != selection.process.id
            || physics_v1.process != selection.process.expression
            || physics_v1.color_accuracy.as_str() != selection.process.color_accuracy
            || physics_v1
                .external_particles
                .iter()
                .map(|particle| particle.pdg)
                .ne(selection.process.external_pdgs.iter().copied())
        {
            return Err(RusticolError::integrity(format!(
                "runtime physics payload {:?} does not match process {:?}",
                selection.process.physics_path, selection.process.id
            )));
        }
        let (representative_process, representative_key, mut runtime, execution_lane) =
            match manifest {
                LoadedExecutionManifest::Compiled(manifest) => {
                    super::eager_v3_load::reject_native_reduction_groups_for_compiled(&physics_v1)?;
                    let representative_process = manifest.process.clone();
                    let representative_key = manifest.key.clone();
                    let evaluator_payloads = artifact.evaluator_payload_store(&evaluator_root)?;
                    let runtime =
                        load_execution_manifest_with_store(*manifest, &evaluator_payloads)?;
                    (
                        representative_process,
                        representative_key,
                        runtime,
                        NativeExecutionLane::Compiled,
                    )
                }
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                LoadedExecutionManifest::EagerV3(manifest) => {
                    let representative_process = manifest.process.clone();
                    let representative_key = manifest.key.clone();
                    let loaded = super::eager_v3_load::load_eager_v3_native_runtime(
                        &artifact,
                        &evaluator_root,
                        &manifest,
                        &mut physics_v1,
                    )?;
                    (
                        representative_process,
                        representative_key,
                        loaded.common,
                        NativeExecutionLane::Eager(Box::new(loaded.lane)),
                    )
                }
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                LoadedExecutionManifest::Recurrence(manifest) => {
                    let representative_process = manifest.process.clone();
                    let representative_key = manifest.key.clone();
                    let loaded = load_recurrence_native_runtime(
                        &artifact,
                        &evaluator_root,
                        &manifest,
                        &physics_v1,
                    )?;
                    (
                        representative_process,
                        representative_key,
                        loaded.common,
                        NativeExecutionLane::Recurrence(Box::new(loaded.lane)),
                    )
                }
            };
        let process = selection
            .alias
            .as_ref()
            .map(|alias| alias.expression.clone())
            .unwrap_or_else(|| representative_process.clone());
        let process_key = selection.requested_id.clone();
        let input_crossing_map = if let Some(alias) = &selection.alias {
            let representative_physics = physics_v1.clone();
            let alias_physics =
                apply_final_state_alias_metadata(physics_v1, alias).map_err(|error| {
                    RusticolError::with_kind(
                        error.kind(),
                        format!("could not remap alias physics metadata: {error}"),
                    )
                })?;
            let helicity_id_map = representative_physics
                .helicities
                .iter()
                .zip(&alias_physics.helicities)
                .map(|(representative, public)| (representative.id.clone(), public.id.clone()))
                .collect::<BTreeMap<_, _>>();
            let color_id_map = representative_physics
                .color_components
                .iter()
                .zip(&alias_physics.color_components)
                .map(|(representative, public)| {
                    (representative.id().to_string(), public.id().to_string())
                })
                .collect::<BTreeMap<_, _>>();
            runtime
                .remap_lc_topology_replay_public_labels(&alias.external_permutation)
                .map_err(|error| {
                    RusticolError::with_kind(
                        error.kind(),
                        format!("could not remap alias execution metadata: {error}"),
                    )
                })?;
            runtime.remap_physics_reduction_overrides(&helicity_id_map, &color_id_map)?;
            physics_v1 = alias_physics;
            runtime.set_external_pdg_order_recursive(&alias.external_pdgs);
            Some(
                alias
                    .external_permutation
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(target_index, source_index)| InputCrossingMapEntry {
                        target_index,
                        source_index,
                        sign: 1.0,
                    })
                    .collect(),
            )
        } else {
            None
        };
        let input_crossing_map =
            prevalidate_input_crossing_lookup(runtime.external_count, input_crossing_map)?;
        runtime.attach_physics(Arc::new(PhysicsRuntime::new(physics_v1.clone())?))?;
        if matches!(&execution_lane, NativeExecutionLane::Compiled) {
            runtime.initialize_compiled_helicity_execution_plan(
                selection
                    .alias
                    .as_ref()
                    .map(|alias| alias.external_permutation.as_slice()),
            )?;
        }
        let selector_simd_lane_width = {
            let uses_simd_jit = match &execution_lane {
                NativeExecutionLane::Compiled => execution_uses_simd_jit(&runtime),
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                NativeExecutionLane::Eager(runtime) => runtime.backend_name() == "jit",
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                NativeExecutionLane::Recurrence(runtime) => runtime.backend_name() == "jit",
            };
            if uses_simd_jit {
                super::evaluator::native_f64_simd_lane_width()
            } else {
                1
            }
        };
        let mut loaded = Self {
            root: artifact.root().to_path_buf(),
            artifact_id,
            runtime,
            execution_lane,
            process,
            process_key,
            input_crossing_map,
            final_state_permutation_alias_of: selection.alias.as_ref().map(|_| representative_key),
            physics_v1,
            warnings_muted: false,
            warned_kinds: BTreeSet::new(),
            pending_warnings: Vec::new(),
            point_selector_scratch: PointSelectorExecutionScratch::default(),
            selector_simd_lane_width,
        };
        if let Some(path) = model_parameters_path {
            loaded.set_model_parameters_json(path)?;
        }
        Ok(loaded)
    }

    pub fn metadata(&self) -> NativeRuntimeMetadata {
        let (
            execution_mode,
            prepared_backend,
            eager_effective_point_tile_size,
            eager_workspace_bytes,
        ) = match &self.execution_lane {
            NativeExecutionLane::Compiled => ("compiled", None, None, None),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Eager(runtime) => (
                "eager",
                Some(runtime.backend_name().to_string()),
                Some(runtime.effective_point_tile_size()),
                Some(runtime.workspace_bytes()),
            ),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Recurrence(runtime) => (
                "recurrence",
                Some(runtime.backend_name().to_string()),
                Some(runtime.effective_point_tile_size()),
                None,
            ),
        };
        NativeRuntimeMetadata {
            abi_version: Self::ABI_VERSION,
            schema_version: PROCESS_ARTIFACT_SCHEMA_VERSION,
            execution_mode: execution_mode.to_string(),
            prepared_backend,
            eager_effective_point_tile_size,
            eager_workspace_bytes,
            process: self.process.clone(),
            process_key: self.process_key.clone(),
            representative_process: self.runtime.process.clone(),
            representative_process_key: self.runtime.key.clone(),
            final_state_permutation_alias_of: self.final_state_permutation_alias_of.clone(),
            color_accuracy: self.runtime.color_accuracy.clone(),
            external_pdg_order: self.runtime.external_pdg_order.clone(),
            external_count: self.runtime.external_count,
            current_count: self.runtime.current_count,
            source_count: self.runtime.source_count,
            interaction_count: self.runtime.interaction_count,
            stage_count: self.runtime.stage_count,
            amplitude_output_count: self.runtime.amplitude_output_count,
        }
    }

    fn selector_simd_lane_width(&self) -> usize {
        self.selector_simd_lane_width
    }

    pub fn metadata_json(&self) -> Result<String, RusticolError> {
        serde_json::to_string(&self.metadata()).map_err(|error| {
            RusticolError::serialization(format!("could not serialize runtime metadata: {error}"))
        })
    }

    pub fn physics_json(&self) -> Result<String, RusticolError> {
        serde_json::to_string(&self.physics_v1).map_err(|error| {
            RusticolError::serialization(format!("could not serialize physics metadata: {error}"))
        })
    }

    /// Return the validated mutable state needed by the lazy Python
    /// high-precision executor.
    ///
    /// The values have already passed Rusticol's atomic parameter update and
    /// derived-parameter refresh logic. They are intentionally exposed only as
    /// an internal bridge; f64 evaluation continues to execute entirely in the
    /// Python-independent core.
    pub fn exact_runtime_state_json(&self) -> Result<String, RusticolError> {
        serde_json::to_string(&serde_json::json!({
            "model_parameter_values": self.runtime.model_parameter_values_f64,
            "normalization_factor": self.runtime.normalization_factor,
        }))
        .map_err(|error| {
            RusticolError::serialization(format!(
                "could not serialize exact-runtime state: {error}"
            ))
        })
    }

    pub fn process_physics(&self) -> &ProcessPhysicsV1 {
        &self.physics_v1
    }

    pub fn external_count(&self) -> usize {
        self.runtime.external_count
    }

    pub fn external_particles(&self) -> Result<Vec<NativeExternalParticle>, RusticolError> {
        Ok(self
            .physics_v1
            .external_particles
            .iter()
            .map(|item| NativeExternalParticle {
                label: item.label,
                index: item.index,
                side: match item.role {
                    crate::ParticleRole::Initial => "initial",
                    crate::ParticleRole::Final => "final",
                }
                .to_string(),
                role: match item.role {
                    crate::ParticleRole::Initial => "initial",
                    crate::ParticleRole::Final => "final",
                }
                .to_string(),
                particle: item.particle.clone(),
                outgoing_particle: item.particle.clone(),
                pdg: item.pdg,
                outgoing_pdg: item.pdg,
                particle_class: String::new(),
                momentum_slot: item.momentum_slot,
            })
            .collect())
    }

    pub fn helicities(&self) -> Result<Vec<NativeHelicityConfiguration>, RusticolError> {
        Ok(self
            .physics_v1
            .helicities
            .iter()
            .map(|item| NativeHelicityConfiguration {
                id: item.id.clone(),
                index: item.index,
                helicities: item.values.clone(),
                representative_id: item.representative_id.clone(),
                computed: item.computed,
                structural_zero: item.structural_zero,
                coefficient: item.coefficient,
            })
            .collect())
    }

    pub fn color_components(&self) -> Result<Vec<NativeColorComponent>, RusticolError> {
        Ok(self
            .physics_v1
            .color_components
            .iter()
            .map(|item| match item {
                PhysicsColorComponentV1::LcFlow(flow) => NativeColorComponent {
                    id: flow.id.clone(),
                    index: flow.index,
                    kind: "lc-flow".to_string(),
                    word: flow.word.clone(),
                    representative_id: flow.representative_id.clone(),
                    computed: flow.computed,
                    coefficient: flow.coefficient,
                },
                PhysicsColorComponentV1::ContractedColor(color) => NativeColorComponent {
                    id: color.id.clone(),
                    index: color.index,
                    kind: "contracted-color".to_string(),
                    word: Vec::new(),
                    representative_id: color.id.clone(),
                    computed: true,
                    coefficient: 1.0,
                },
            })
            .collect())
    }

    pub fn model_parameters(&self) -> Result<Vec<NativeModelParameter>, RusticolError> {
        Ok(self
            .physics_v1
            .model_parameters
            .iter()
            .enumerate()
            .map(|(parameter_index, item)| NativeModelParameter {
                name: item.name.clone(),
                kind: format!("{:?}", item.kind).to_ascii_lowercase(),
                parameter_index,
                default: item.default_real,
                default_imaginary: item.default_imaginary,
                mutable: item.mutable,
            })
            .collect())
    }

    pub fn helicity_ids(&self) -> Result<Vec<String>, RusticolError> {
        Ok(self
            .physics_v1
            .helicities
            .iter()
            .map(|item| item.id.clone())
            .collect())
    }

    pub fn color_ids(&self) -> Result<Vec<String>, RusticolError> {
        Ok(self
            .physics_v1
            .color_components
            .iter()
            .map(|item| item.id().to_string())
            .collect())
    }

    pub fn resolved_shape(
        &self,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<(usize, usize), RusticolError> {
        self.validate_selector_capabilities(helicity_ids.is_some(), color_ids.is_some())?;
        let selected_helicities = selector_set(helicity_ids, "helicity")?;
        let selected_colors = selector_set(color_ids, "color component")?;
        let physics = self.runtime.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        let helicity_count = physics
            .selected_helicity_indices(selected_helicities.as_ref())
            .map_err(|error| RusticolError::selector(error.to_string()))?
            .len();
        let color_count = physics
            .selected_color_indices(selected_colors.as_ref())
            .map_err(|error| RusticolError::selector(error.to_string()))?
            .len();
        Ok((helicity_count, color_count))
    }

    /// Resolve and retain batch-global recurrence selectors once.
    ///
    /// The returned handle is tied to this artifact and representative
    /// process. Handle creation may allocate; repeated evaluation through the
    /// prepared-plan API does not rebuild string selector sets.
    pub fn prepare_recurrence_selector_plan(
        &self,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> RusticolResult<NativeRecurrenceSelectorPlan> {
        let NativeExecutionLane::Recurrence(runtime) = &self.execution_lane else {
            return Err(RusticolError::invalid_argument(
                "recurrence selector plans require a recurrence artifact",
            ));
        };
        self.validate_selector_capabilities(helicity_ids.is_some(), color_ids.is_some())?;
        let selected_helicities = selector_set(helicity_ids, "helicity")?;
        let selected_colors = selector_set(color_ids, "color component")?;
        let physics = self.runtime.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact("recurrence execution requires physics metadata")
        })?;
        runtime.validate_global_selectors(
            physics,
            selected_helicities.as_ref(),
            selected_colors.as_ref(),
        )?;
        Ok(NativeRecurrenceSelectorPlan {
            artifact_root: self.root.clone(),
            process_key: self.process_key.clone(),
            selected_helicities,
            selected_colors,
        })
    }

    pub fn evaluate_f64(
        &mut self,
        momenta: &[f64],
        point_count: usize,
    ) -> Result<Vec<f64>, RusticolError> {
        self.evaluate_f64_with_selectors(momenta, point_count, None, None, None, None)
    }

    pub fn evaluate_f64_into(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        self.evaluate_f64_into_with_selectors(momenta, point_count, None, None, None, None, output)
    }

    /// Evaluate recurrence totals with a previously resolved selector plan.
    pub fn evaluate_f64_into_with_recurrence_selector_plan(
        &mut self,
        plan: &NativeRecurrenceSelectorPlan,
        momenta: &[f64],
        point_count: usize,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        if !matches!(&self.execution_lane, NativeExecutionLane::Recurrence(_)) {
            return Err(RusticolError::invalid_argument(
                "recurrence selector plans require a recurrence artifact",
            ));
        }
        if plan.artifact_root != self.root || plan.process_key != self.process_key {
            return Err(RusticolError::selector(
                "recurrence selector plan belongs to a different artifact or process",
            ));
        }
        validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        if output.len() != point_count {
            return Err(RusticolError::invalid_argument(format!(
                "evaluation output has length {}, expected {point_count}",
                output.len()
            )));
        }
        let crossing_lookup = std::mem::take(&mut self.input_crossing_map);
        let result = (|| {
            let batch = F64MomentumBatchView::from_contiguous_prevalidated(
                momenta,
                point_count,
                self.runtime.external_count,
                crossing_lookup.as_deref(),
            )?;
            if plan.selected_helicities.is_some() || plan.selected_colors.is_some() {
                self.run_selected_f64_batch_into(
                    batch,
                    plan.selected_helicities.as_ref(),
                    plan.selected_colors.as_ref(),
                    output,
                )
            } else {
                self.run_f64_batch_into(batch, output)
            }
        })();
        self.input_crossing_map = crossing_lookup;
        result
    }

    /// Evaluate one total per point with optional global or per-point selectors.
    ///
    /// Global selectors retain the existing subset semantics. Per-point
    /// selectors are resolved physical-axis indices and contain exactly one
    /// selector for every input point. The two forms are mutually exclusive on
    /// the same axis. An omitted axis is summed over all components retained by
    /// the artifact.
    pub fn evaluate_f64_with_selectors(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
        helicity_by_point: Option<&[u32]>,
        color_by_point: Option<&[u32]>,
    ) -> Result<Vec<f64>, RusticolError> {
        validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        let mut output = vec![0.0; point_count];
        self.evaluate_f64_into_with_selectors(
            momenta,
            point_count,
            helicity_ids,
            color_ids,
            helicity_by_point,
            color_by_point,
            &mut output,
        )?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_f64_into_with_selectors(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
        helicity_by_point: Option<&[u32]>,
        color_by_point: Option<&[u32]>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        if output.len() != point_count {
            return Err(RusticolError::invalid_argument(format!(
                "evaluation output has length {}, expected {point_count}",
                output.len()
            )));
        }
        if helicity_ids.is_some() && helicity_by_point.is_some() {
            return Err(RusticolError::selector(
                "helicities and helicity_by_point are mutually exclusive",
            ));
        }
        if color_ids.is_some() && color_by_point.is_some() {
            return Err(RusticolError::selector(
                "color_flows and color_flow_by_point are mutually exclusive",
            ));
        }
        self.validate_selector_capabilities(
            helicity_ids.is_some() || helicity_by_point.is_some(),
            color_ids.is_some() || color_by_point.is_some(),
        )?;
        let physics = self.runtime.physics.clone().ok_or_else(|| {
            RusticolError::artifact(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        let crossing_lookup = std::mem::take(&mut self.input_crossing_map);
        let mut selector_scratch = std::mem::take(&mut self.point_selector_scratch);
        let result = (|| {
            if helicity_by_point.is_some() || color_by_point.is_some() {
                selector_scratch.prepare_singletons(&physics);
            }
            let PointSelectorExecutionScratch {
                planner,
                gathered_batch,
                partition_totals,
                output_totals,
                helicity_selector_sets,
                color_selector_sets,
                helicity_singletons,
                color_singletons,
            } = &mut selector_scratch;
            let selected_helicities = helicity_selector_sets.resolve(helicity_ids, "helicity")?;
            let selected_colors = color_selector_sets.resolve(color_ids, "color component")?;
            let batch = F64MomentumBatchView::from_contiguous_prevalidated(
                momenta,
                point_count,
                self.runtime.external_count,
                crossing_lookup.as_deref(),
            )?;
            let plan = planner.build(
                point_count,
                helicity_by_point,
                color_by_point,
                physics.manifest.helicities.len(),
                physics.manifest.color_components.len(),
            )?;
            if plan == PointSelectorPlan::None {
                if selected_helicities.is_some() || selected_colors.is_some() {
                    return self.run_selected_f64_batch_into(
                        batch,
                        selected_helicities,
                        selected_colors,
                        output,
                    );
                }
                return self.run_f64_batch_into(batch, output);
            }

            self.record_resolved_warnings(helicity_ids, color_ids)?;
            output_totals.resize(point_count, 0.0);
            output_totals.fill(0.0);
            if let PointSelectorPlan::Homogeneous(key) = plan {
                let point_helicities = key.helicity_index.map(|index| &helicity_singletons[index]);
                let point_colors = key.color_index.map(|index| &color_singletons[index]);
                let effective_helicities = point_helicities.or(selected_helicities);
                let effective_colors = point_colors.or(selected_colors);
                self.run_selected_f64_batch_into(
                    batch,
                    effective_helicities,
                    effective_colors,
                    output_totals,
                )?;
                output.copy_from_slice(output_totals);
                return Ok(());
            }

            let partition_count = planner.partitions().len();
            for partition_index in 0..partition_count {
                let partition = planner.partitions()[partition_index];
                let point_helicities = partition
                    .key
                    .helicity_index
                    .map(|index| &helicity_singletons[index]);
                let point_colors = partition
                    .key
                    .color_index
                    .map(|index| &color_singletons[index]);
                let effective_helicities = point_helicities.or(selected_helicities);
                let effective_colors = point_colors.or(selected_colors);
                match partition.rows {
                    PointSelectorRows::Contiguous { start, end } => {
                        self.run_selected_f64_batch_into(
                            batch.subview(start, end)?,
                            effective_helicities,
                            effective_colors,
                            &mut output_totals[start..end],
                        )?;
                    }
                    rows @ PointSelectorRows::Gathered { .. } => {
                        let point_indices = planner.gathered_rows(rows);
                        let gathered_batch =
                            fill_gathered_batch_from_view(gathered_batch, batch, point_indices)?;
                        partition_totals.resize(partition.rows.len(), 0.0);
                        partition_totals.fill(0.0);
                        self.run_selected_f64_batch_into(
                            F64MomentumBatchView::from_nested(
                                gathered_batch,
                                self.runtime.external_count,
                            )?,
                            effective_helicities,
                            effective_colors,
                            partition_totals,
                        )?;
                        scatter_partition_totals(
                            output_totals,
                            partition_totals,
                            partition.rows,
                            planner,
                        );
                    }
                }
            }
            output.copy_from_slice(output_totals);
            Ok(())
        })();
        self.point_selector_scratch = selector_scratch;
        self.input_crossing_map = crossing_lookup;
        result
    }

    fn run_f64_batch_into(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        match &mut self.execution_lane {
            NativeExecutionLane::Compiled => self.runtime.run_f64_into_unprofiled(batch, output),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Eager(runtime) => {
                runtime.run_f64_view_into_unprofiled(&mut self.runtime, batch, output)
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Recurrence(runtime) => {
                runtime.run_f64_view_into_unprofiled(&mut self.runtime, batch, None, None, output)
            }
        }
    }

    fn run_selected_f64_batch_into(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        match &mut self.execution_lane {
            NativeExecutionLane::Compiled => self.runtime.run_f64_selected_into_unprofiled(
                batch,
                selected_helicities,
                selected_colors,
                output,
            ),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Eager(runtime) => runtime.run_f64_view_selected_into_unprofiled(
                &mut self.runtime,
                batch,
                selected_helicities,
                selected_colors,
                output,
            ),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Recurrence(runtime) => runtime.run_f64_view_into_unprofiled(
                &mut self.runtime,
                batch,
                selected_helicities,
                selected_colors,
                output,
            ),
        }
    }

    fn run_resolved_f64_batch(
        &mut self,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> Result<ResolvedValues<f64>, RusticolError> {
        match &mut self.execution_lane {
            NativeExecutionLane::Compiled => self.runtime.run_resolved_f64_unprofiled(
                batch,
                selected_helicities,
                selected_colors,
            ),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Eager(runtime) => runtime.run_resolved_f64_view_unprofiled(
                &mut self.runtime,
                batch,
                selected_helicities,
                selected_colors,
            ),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Recurrence(runtime) => {
                let nested = batch.materialize_nested();
                runtime
                    .run_resolved_f64(
                        &mut self.runtime,
                        &nested,
                        selected_helicities,
                        selected_colors,
                    )
                    .map(|(resolved, _profile)| resolved)
            }
        }
    }

    fn run_selected_f64_batch_profile(
        &mut self,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> Result<(Vec<f64>, RuntimeProfile), RusticolError> {
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        if matches!(&self.execution_lane, NativeExecutionLane::Compiled) {
            let before = compiled_direct_profile_snapshot(&self.runtime)?;
            if before.engine_count != 0 {
                let total_start = Instant::now();
                let mut values = vec![0.0; batch.len()];
                self.runtime.run_f64_selected_into_unprofiled(
                    F64MomentumBatchView::from_nested(batch, self.runtime.external_count)?,
                    selected_helicities,
                    selected_colors,
                    &mut values,
                )?;
                let total_s = total_start.elapsed().as_secs_f64();
                let after = compiled_direct_profile_snapshot(&self.runtime)?;
                let f64_bytes = std::mem::size_of::<f64>() as u64;
                let complex_bytes = 2 * f64_bytes;
                return Ok((
                    values,
                    RuntimeProfile {
                        // The direct hot path is deliberately uninstrumented.
                        // Until its three coarse phase clocks are added, keep
                        // the complete measured envelope in orchestration so
                        // top-level accounting remains exact rather than
                        // manufacturing dense-stage timing fields.
                        orchestration_s: total_s,
                        total_s,
                        source_component_count: after
                            .source_fill_bytes
                            .saturating_sub(before.source_fill_bytes)
                            / complex_bytes,
                        momentum_component_count: after
                            .momentum_fill_bytes
                            .saturating_sub(before.momentum_fill_bytes)
                            / f64_bytes,
                        model_parameter_component_count: after
                            .parameter_fill_bytes
                            .saturating_sub(before.parameter_fill_bytes)
                            / complex_bytes,
                        state_clear_component_count: after
                            .amplitude_clear_bytes
                            .saturating_sub(before.amplitude_clear_bytes)
                            / complex_bytes,
                        evaluator_backend_call_count: after
                            .backend_call_count
                            .saturating_sub(before.backend_call_count),
                        compiled_direct_arena_engine_count: after.engine_count as u64,
                        compiled_direct_arena_call_count: after
                            .backend_call_count
                            .saturating_sub(before.backend_call_count),
                        compiled_direct_arena_boundary_input_bytes: after.boundary_input_bytes,
                        compiled_direct_arena_boundary_current_output_bytes: after
                            .boundary_current_output_bytes,
                        compiled_direct_arena_boundary_amplitude_output_bytes: after
                            .boundary_amplitude_output_bytes,
                        total_materialized_value_count: batch.len() as u64,
                        ..RuntimeProfile::default()
                    },
                ));
            }
        }
        match &mut self.execution_lane {
            NativeExecutionLane::Compiled => {
                self.runtime
                    .run_f64_selected_totals(batch, selected_helicities, selected_colors)
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Eager(runtime) => {
                let (resolved, mut profile) = runtime.run_resolved_f64_profile(
                    &mut self.runtime,
                    batch,
                    selected_helicities,
                    selected_colors,
                )?;
                let materialization_start = Instant::now();
                let values = resolved_f64_totals(&resolved)?;
                profile.total_materialization_s += materialization_start.elapsed().as_secs_f64();
                profile.total_materialized_value_count += values.len() as u64;
                Ok((values, profile))
            }
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Recurrence(runtime) => {
                let (resolved, mut profile) = runtime.run_resolved_f64(
                    &mut self.runtime,
                    batch,
                    selected_helicities,
                    selected_colors,
                )?;
                let materialization_start = Instant::now();
                let values = resolved_f64_totals(&resolved)?;
                profile.total_materialization_s += materialization_start.elapsed().as_secs_f64();
                profile.total_materialized_value_count += values.len() as u64;
                Ok((values, profile))
            }
        }
    }

    pub fn benchmark_f64_wall_time(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        repetitions: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<f64, RusticolError> {
        self.benchmark_f64_wall_time_with_selectors(
            momenta,
            point_count,
            repetitions,
            helicity_ids,
            color_ids,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn benchmark_f64_wall_time_with_selectors(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        repetitions: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
        helicity_by_point: Option<&[u32]>,
        color_by_point: Option<&[u32]>,
    ) -> Result<f64, RusticolError> {
        if repetitions == 0 {
            return Err(RusticolError::invalid_argument(
                "benchmark repetitions must be positive",
            ));
        }
        validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        let mut output = vec![0.0; point_count];
        let started = Instant::now();
        for _ in 0..repetitions {
            self.evaluate_f64_into_with_selectors(
                momenta,
                point_count,
                helicity_ids,
                color_ids,
                helicity_by_point,
                color_by_point,
                &mut output,
            )?;
            std::hint::black_box(&output);
        }
        Ok(started.elapsed().as_secs_f64())
    }

    /// Profile the warmed Direct-Arena timing boundary used by
    /// [`Self::benchmark_f64_wall_time`] for batch-global selectors.
    ///
    /// The borrowed flat input and the caller-sized output are prepared once
    /// outside the repeated region. Every measured repetition then uses the
    /// allocation-free `evaluate_f64_into_with_selectors` path. This is kept
    /// separate from the public diagnostic profiler below, whose materialized
    /// input and owned result remain useful when inspecting legacy lanes.
    #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
    pub fn evaluate_f64_arena_profile_repeated(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        repetitions: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<NativeProfiledEvaluation, RusticolError> {
        if repetitions == 0 {
            return Err(RusticolError::invalid_argument(
                "arena profile repetitions must be positive",
            ));
        }
        validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        let compiled_lane = match &self.execution_lane {
            NativeExecutionLane::Compiled => {
                let snapshot = compiled_direct_profile_snapshot(&self.runtime)?;
                if snapshot.engine_count == 0 {
                    return Err(RusticolError::compatibility(
                        "warmed Arena profiling requires a compiled Direct-Arena artifact",
                    ));
                }
                true
            }
            NativeExecutionLane::Eager(_) => false,
            NativeExecutionLane::Recurrence(_) => {
                return Err(RusticolError::compatibility(
                    "warmed Arena profiling is available only for eager and compiled execution",
                ));
            }
        };
        let measured_points = point_count.checked_mul(repetitions).ok_or_else(|| {
            RusticolError::invalid_argument("arena profile point count overflowed")
        })?;
        let measured_input_components =
            momenta.len().checked_mul(repetitions).ok_or_else(|| {
                RusticolError::invalid_argument(
                    "arena profile native input component count overflowed",
                )
            })?;
        let measured_points_u64 = u64::try_from(measured_points).map_err(|_| {
            RusticolError::invalid_argument("arena profile point count does not fit in u64")
        })?;
        let measured_input_components_u64 =
            u64::try_from(measured_input_components).map_err(|_| {
                RusticolError::invalid_argument(
                    "arena profile native input component count does not fit in u64",
                )
            })?;
        let repetitions_u64 = u64::try_from(repetitions).map_err(|_| {
            RusticolError::invalid_argument("arena profile repetition count does not fit in u64")
        })?;

        // The result allocation is deliberately outside the measured repeated
        // boundary, matching the native headline timer. Prime the exact
        // selector and workspace route once before the counter snapshot so the
        // measured region is independently warmed even when this private
        // operation is called directly.
        let mut values = vec![0.0; point_count];
        self.evaluate_f64_into_with_selectors(
            momenta,
            point_count,
            helicity_ids,
            color_ids,
            None,
            None,
            &mut values,
        )?;
        let compiled_before = if compiled_lane {
            Some(compiled_direct_profile_snapshot(&self.runtime)?)
        } else {
            None
        };
        let started = Instant::now();
        for _ in 0..repetitions {
            std::hint::black_box(&mut values);
            self.evaluate_f64_into_with_selectors(
                momenta,
                point_count,
                helicity_ids,
                color_ids,
                None,
                None,
                &mut values,
            )?;
            std::hint::black_box(&values);
        }
        let total_s = started.elapsed().as_secs_f64();
        let mut runtime_profile = RuntimeProfile {
            // The warmed Direct-Arena boundary is deliberately free of phase
            // clocks. Keep its complete measured envelope in orchestration.
            // Callers must treat leaf-evaluator phase timing as unavailable,
            // rather than interpreting the zero-valued phase fields as
            // measurements below clock resolution.
            orchestration_s: total_s,
            total_s,
            total_materialized_value_count: measured_points_u64,
            ..RuntimeProfile::default()
        };
        if let Some(before) = compiled_before {
            let after = compiled_direct_profile_snapshot(&self.runtime)?;
            let f64_bytes = std::mem::size_of::<f64>() as u64;
            let complex_bytes = 2 * f64_bytes;
            runtime_profile.source_component_count = after
                .source_fill_bytes
                .saturating_sub(before.source_fill_bytes)
                / complex_bytes;
            runtime_profile.momentum_component_count = after
                .momentum_fill_bytes
                .saturating_sub(before.momentum_fill_bytes)
                / f64_bytes;
            runtime_profile.model_parameter_component_count = after
                .parameter_fill_bytes
                .saturating_sub(before.parameter_fill_bytes)
                / complex_bytes;
            runtime_profile.state_clear_component_count = after
                .amplitude_clear_bytes
                .saturating_sub(before.amplitude_clear_bytes)
                / complex_bytes;
            runtime_profile.evaluator_backend_call_count = after
                .backend_call_count
                .saturating_sub(before.backend_call_count);
            runtime_profile.compiled_direct_arena_engine_count = u64::try_from(before.engine_count)
                .ok()
                .and_then(|count| count.checked_mul(repetitions_u64))
                .ok_or_else(|| {
                    RusticolError::invalid_argument("compiled Direct-Arena engine count overflowed")
                })?;
            runtime_profile.compiled_direct_arena_call_count =
                runtime_profile.evaluator_backend_call_count;
            runtime_profile.compiled_direct_arena_boundary_input_bytes = after
                .boundary_input_bytes
                .saturating_sub(before.boundary_input_bytes);
            runtime_profile.compiled_direct_arena_boundary_current_output_bytes = after
                .boundary_current_output_bytes
                .saturating_sub(before.boundary_current_output_bytes);
            runtime_profile.compiled_direct_arena_boundary_amplitude_output_bytes = after
                .boundary_amplitude_output_bytes
                .saturating_sub(before.boundary_amplitude_output_bytes);
        }
        let mut profile: NativeRuntimeProfile = runtime_profile.into();
        profile.native_input_component_count = measured_input_components_u64;
        profile.native_input_pack_bytes = 0;
        profile.native_input_crossing_bytes = 0;
        profile.native_input_container_allocation_count = 0;
        profile.native_output_allocation_count = 0;
        self.validate_profile_accounting(&profile)?;
        Ok(NativeProfiledEvaluation { values, profile })
    }

    #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
    pub fn evaluate_f64_arena_profile_repeated(
        &mut self,
        _momenta: &[f64],
        _point_count: usize,
        _repetitions: usize,
        _helicity_ids: Option<&[String]>,
        _color_ids: Option<&[String]>,
    ) -> Result<NativeProfiledEvaluation, RusticolError> {
        Err(RusticolError::compatibility(
            "warmed Arena profiling requires the f64-compiled or f64-symjit feature",
        ))
    }

    pub fn evaluate_f64_profile(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<NativeProfiledEvaluation, RusticolError> {
        let total_start = Instant::now();
        let (batch, native_input_pack_elapsed, native_input_crossing_elapsed) =
            self.prepare_f64_batch_profile(momenta, point_count)?;
        let compiled_direct_profile = {
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            {
                matches!(&self.execution_lane, NativeExecutionLane::Compiled)
                    && compiled_direct_profile_snapshot(&self.runtime)?.engine_count != 0
            }
            #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
            {
                false
            }
        };
        let (values, profile) = if compiled_direct_profile {
            if helicity_ids.is_some() || color_ids.is_some() {
                self.validate_selector_capabilities(helicity_ids.is_some(), color_ids.is_some())?;
                self.record_resolved_warnings(helicity_ids, color_ids)?;
            }
            let selected_helicities = selector_set(helicity_ids, "helicity")?;
            let selected_colors = selector_set(color_ids, "color component")?;
            self.run_selected_f64_batch_profile(
                &batch,
                selected_helicities.as_ref(),
                selected_colors.as_ref(),
            )?
        } else if helicity_ids.is_some() || color_ids.is_some() {
            self.validate_selector_capabilities(helicity_ids.is_some(), color_ids.is_some())?;
            self.record_resolved_warnings(helicity_ids, color_ids)?;
            let selected_helicities = selector_set(helicity_ids, "helicity")?;
            let selected_colors = selector_set(color_ids, "color component")?;
            let (values, profile) = match &mut self.execution_lane {
                NativeExecutionLane::Compiled => self.runtime.run_f64_selected_totals(
                    &batch,
                    selected_helicities.as_ref(),
                    selected_colors.as_ref(),
                )?,
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                NativeExecutionLane::Eager(runtime) => {
                    let (resolved, mut profile) = runtime.run_resolved_f64_profile(
                        &mut self.runtime,
                        &batch,
                        selected_helicities.as_ref(),
                        selected_colors.as_ref(),
                    )?;
                    let materialization_start = Instant::now();
                    let values = resolved_f64_totals(&resolved)?;
                    profile.total_materialization_s +=
                        materialization_start.elapsed().as_secs_f64();
                    profile.total_materialized_value_count += point_count as u64;
                    (values, profile)
                }
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                NativeExecutionLane::Recurrence(runtime) => {
                    let (resolved, mut profile) = runtime.run_resolved_f64(
                        &mut self.runtime,
                        &batch,
                        selected_helicities.as_ref(),
                        selected_colors.as_ref(),
                    )?;
                    let materialization_start = Instant::now();
                    let values = resolved_f64_totals(&resolved)?;
                    profile.total_materialization_s +=
                        materialization_start.elapsed().as_secs_f64();
                    profile.total_materialized_value_count += point_count as u64;
                    (values, profile)
                }
            };
            (values, profile)
        } else {
            match &mut self.execution_lane {
                NativeExecutionLane::Compiled => self.runtime.run_f64(&batch)?,
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                NativeExecutionLane::Eager(runtime) => {
                    runtime.run_f64_profile(&mut self.runtime, &batch)?
                }
                #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
                NativeExecutionLane::Recurrence(runtime) => {
                    runtime.run_f64(&mut self.runtime, &batch)?
                }
            }
        };
        let mut profile: NativeRuntimeProfile = profile.into();
        profile.native_input_pack_s = profile_duration_seconds(native_input_pack_elapsed);
        profile.native_input_crossing_s = profile_duration_seconds(native_input_crossing_elapsed);
        profile.native_input_component_count = momenta.len() as u64;
        profile.native_input_pack_bytes = std::mem::size_of_val(momenta) as u64;
        if self.input_crossing_map.is_some() {
            profile.native_input_crossing_bytes = profile.native_input_pack_bytes;
        }
        profile.native_input_container_allocation_count =
            (point_count + 1 + usize::from(self.input_crossing_map.is_some()) * (point_count + 2))
                as u64;
        profile.native_output_allocation_count = 1;
        profile.total_s = total_start.elapsed().as_secs_f64();
        self.validate_profile_accounting(&profile)?;
        Ok(NativeProfiledEvaluation { values, profile })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_f64_profile_with_selectors(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
        helicity_by_point: Option<&[u32]>,
        color_by_point: Option<&[u32]>,
    ) -> Result<NativeProfiledEvaluation, RusticolError> {
        if helicity_by_point.is_none() && color_by_point.is_none() {
            return self.evaluate_f64_profile(momenta, point_count, helicity_ids, color_ids);
        }
        if helicity_ids.is_some() && helicity_by_point.is_some() {
            return Err(RusticolError::selector(
                "helicities and helicity_by_point are mutually exclusive",
            ));
        }
        if color_ids.is_some() && color_by_point.is_some() {
            return Err(RusticolError::selector(
                "color_flows and color_flow_by_point are mutually exclusive",
            ));
        }
        self.validate_selector_capabilities(
            helicity_ids.is_some() || helicity_by_point.is_some(),
            color_ids.is_some() || color_by_point.is_some(),
        )?;
        let total_start = Instant::now();
        let selected_helicities = selector_set(helicity_ids, "helicity")?;
        let selected_colors = selector_set(color_ids, "color component")?;
        let (batch, native_input_pack_elapsed, native_input_crossing_elapsed) =
            self.prepare_f64_batch_profile(momenta, point_count)?;
        let physics = self.runtime.physics.clone().ok_or_else(|| {
            RusticolError::artifact(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        let selector_simd_lane_width = self.selector_simd_lane_width();
        let mut selector_scratch = std::mem::take(&mut self.point_selector_scratch);
        let result = (|| {
            let planner_started = Instant::now();
            let plan = selector_scratch.planner.build(
                point_count,
                helicity_by_point,
                color_by_point,
                physics.manifest.helicities.len(),
                physics.manifest.color_components.len(),
            )?;
            let plan_profile =
                selector_scratch
                    .planner
                    .profile(plan, point_count, selector_simd_lane_width);
            let planner = planner_started.elapsed();
            self.record_resolved_warnings(helicity_ids, color_ids)?;

            if let PointSelectorPlan::Homogeneous(key) = plan {
                let point_helicities = key
                    .helicity_index
                    .map(|index| BTreeSet::from([physics.manifest.helicities[index].id.clone()]));
                let point_colors = key.color_index.map(|index| {
                    BTreeSet::from([physics.manifest.color_components[index].id().to_string()])
                });
                let effective_helicities =
                    point_helicities.as_ref().or(selected_helicities.as_ref());
                let effective_colors = point_colors.as_ref().or(selected_colors.as_ref());
                let (values, runtime_profile) = self.run_selected_f64_batch_profile(
                    &batch,
                    effective_helicities,
                    effective_colors,
                )?;
                let mut profile: NativeRuntimeProfile = runtime_profile.into();
                attach_point_selector_profile(
                    &mut profile,
                    &plan_profile,
                    planner,
                    Duration::ZERO,
                    Duration::ZERO,
                    PointSelectorProfileCounts {
                        gather_point_count: 0,
                        input_bytes_per_point: self.runtime.external_count
                            * 4
                            * std::mem::size_of::<f64>(),
                        scatter_value_count: 0,
                    },
                );
                profile.native_input_pack_s = profile_duration_seconds(native_input_pack_elapsed);
                profile.native_input_crossing_s =
                    profile_duration_seconds(native_input_crossing_elapsed);
                profile.native_input_component_count = momenta.len() as u64;
                profile.native_input_pack_bytes = std::mem::size_of_val(momenta) as u64;
                if self.input_crossing_map.is_some() {
                    profile.native_input_crossing_bytes = profile.native_input_pack_bytes;
                }
                profile.native_input_container_allocation_count = (point_count
                    + 1
                    + usize::from(self.input_crossing_map.is_some()) * (point_count + 2))
                    as u64;
                profile.native_output_allocation_count = 1;
                profile.total_s = total_start.elapsed().as_secs_f64();
                self.validate_profile_accounting(&profile)?;
                return Ok(NativeProfiledEvaluation { values, profile });
            }

            let mut values = vec![0.0; point_count];
            let mut partition_profiles = Vec::new();
            let mut gather = Duration::ZERO;
            let mut scatter = Duration::ZERO;
            let mut gather_point_count = 0usize;
            let partition_count = selector_scratch.planner.partitions().len();
            for partition_index in 0..partition_count {
                let partition = selector_scratch.planner.partitions()[partition_index];
                let point_helicities = partition
                    .key
                    .helicity_index
                    .map(|index| BTreeSet::from([physics.manifest.helicities[index].id.clone()]));
                let point_colors = partition.key.color_index.map(|index| {
                    BTreeSet::from([physics.manifest.color_components[index].id().to_string()])
                });
                let effective_helicities =
                    point_helicities.as_ref().or(selected_helicities.as_ref());
                let effective_colors = point_colors.as_ref().or(selected_colors.as_ref());
                let (partition_totals, partition_profile) = match partition.rows {
                    PointSelectorRows::Contiguous { start, end } => self
                        .run_selected_f64_batch_profile(
                            &batch[start..end],
                            effective_helicities,
                            effective_colors,
                        )?,
                    rows @ PointSelectorRows::Gathered { .. } => {
                        let gather_started = Instant::now();
                        let point_indices = selector_scratch.planner.gathered_rows(rows);
                        gather_point_count += point_indices.len();
                        let gathered_batch = fill_gathered_batch(
                            &mut selector_scratch.gathered_batch,
                            &batch,
                            point_indices,
                        );
                        gather += gather_started.elapsed();
                        self.run_selected_f64_batch_profile(
                            gathered_batch,
                            effective_helicities,
                            effective_colors,
                        )?
                    }
                };
                if partition_totals.len() != partition.rows.len() {
                    return Err(RusticolError::integrity(
                        "per-point selector partition returned the wrong number of values",
                    ));
                }
                let scatter_started = Instant::now();
                scatter_partition_totals(
                    &mut values,
                    &partition_totals,
                    partition.rows,
                    &selector_scratch.planner,
                );
                scatter += scatter_started.elapsed();
                partition_profiles.push(partition_profile);
            }
            let mut runtime_profile = RuntimeProfile::default();
            for partition_profile in &partition_profiles {
                runtime_profile.add_sector(partition_profile);
            }
            let mut profile: NativeRuntimeProfile = runtime_profile.into();
            attach_point_selector_profile(
                &mut profile,
                &plan_profile,
                planner,
                gather,
                scatter,
                PointSelectorProfileCounts {
                    gather_point_count,
                    input_bytes_per_point: self.runtime.external_count
                        * 4
                        * std::mem::size_of::<f64>(),
                    scatter_value_count: point_count,
                },
            );
            profile.native_input_pack_s = profile_duration_seconds(native_input_pack_elapsed);
            profile.native_input_crossing_s =
                profile_duration_seconds(native_input_crossing_elapsed);
            profile.native_input_component_count = momenta.len() as u64;
            profile.native_input_pack_bytes = std::mem::size_of_val(momenta) as u64;
            if self.input_crossing_map.is_some() {
                profile.native_input_crossing_bytes = profile.native_input_pack_bytes;
            }
            profile.native_input_container_allocation_count = (point_count
                + 1
                + usize::from(self.input_crossing_map.is_some()) * (point_count + 2))
                as u64;
            profile.native_output_allocation_count = 1;
            profile.total_s = total_start.elapsed().as_secs_f64();
            self.validate_profile_accounting(&profile)?;
            Ok(NativeProfiledEvaluation { values, profile })
        })();
        self.point_selector_scratch = selector_scratch;
        result
    }

    pub fn evaluate_f64_profile_repeated(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        repetitions: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<NativeProfiledEvaluation, RusticolError> {
        self.evaluate_f64_profile_repeated_with_selectors(
            momenta,
            point_count,
            repetitions,
            helicity_ids,
            color_ids,
            None,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_f64_profile_repeated_with_selectors(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        repetitions: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
        helicity_by_point: Option<&[u32]>,
        color_by_point: Option<&[u32]>,
    ) -> Result<NativeProfiledEvaluation, RusticolError> {
        if repetitions == 0 {
            return Err(RusticolError::invalid_argument(
                "profile repetitions must be positive",
            ));
        }
        let started = Instant::now();
        let mut profile: Option<Box<NativeRuntimeProfile>> = None;
        let mut values = Vec::new();
        for _ in 0..repetitions {
            // The saved evaluator payload may use the complete AArch64
            // floating-point register file. Keep the aggregate behind an
            // opaque heap boundary so no accumulated f64 survives the next
            // generated call in a register.
            std::hint::black_box(&mut profile);
            let profiled = self.evaluate_f64_profile_with_selectors(
                momenta,
                point_count,
                helicity_ids,
                color_ids,
                helicity_by_point,
                color_by_point,
            )?;
            std::hint::black_box(&profiled.values);
            values = profiled.values;
            if let Some(aggregate) = profile.as_deref_mut() {
                aggregate.accumulate(&profiled.profile);
            } else {
                profile = Some(Box::new(profiled.profile));
            }
        }
        let mut profile = *profile.expect("positive repetitions checked");
        profile.total_s = started.elapsed().as_secs_f64();
        self.validate_profile_accounting(&profile)?;
        Ok(NativeProfiledEvaluation { values, profile })
    }

    fn validate_profile_accounting(&self, profile: &NativeRuntimeProfile) -> RusticolResult<()> {
        match &self.execution_lane {
            NativeExecutionLane::Compiled => profile.validate_compiled_top_level_accounting(),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Eager(_) => profile.validate_eager_top_level_accounting(),
            #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
            NativeExecutionLane::Recurrence(_) => {
                profile.validate_recurrence_top_level_accounting()
            }
        }
    }

    pub fn evaluate_resolved_f64(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<NativeResolvedEvaluation, RusticolError> {
        self.validate_selector_capabilities(helicity_ids.is_some(), color_ids.is_some())?;
        self.record_resolved_warnings(helicity_ids, color_ids)?;
        validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        let physics = self.runtime.physics.clone().ok_or_else(|| {
            RusticolError::artifact(
                "schema-v3 artifact is missing resolved physics metadata; regenerate it with pyAmpliCol 0.1.0 or newer",
            )
        })?;
        let crossing_lookup = std::mem::take(&mut self.input_crossing_map);
        let mut selector_scratch = std::mem::take(&mut self.point_selector_scratch);
        let result = (|| {
            physics.validate_helicity_id_slice(helicity_ids)?;
            physics.validate_color_id_slice(color_ids)?;
            let selected_helicities = selector_scratch
                .helicity_selector_sets
                .resolve(helicity_ids, "helicity")?;
            let selected_colors = selector_scratch
                .color_selector_sets
                .resolve(color_ids, "color component")?;
            let batch = F64MomentumBatchView::from_contiguous_prevalidated(
                momenta,
                point_count,
                self.runtime.external_count,
                crossing_lookup.as_deref(),
            )?;
            self.run_resolved_f64_batch(batch, selected_helicities, selected_colors)
        })();
        self.point_selector_scratch = selector_scratch;
        self.input_crossing_map = crossing_lookup;
        let resolved = result?;
        let helicity_ids = resolved
            .helicity_indices
            .iter()
            .map(|index| physics.manifest.helicities[*index].id.clone())
            .collect();
        let color_ids = resolved
            .color_indices
            .iter()
            .map(|index| physics.manifest.color_components[*index].id().to_string())
            .collect();
        Ok(NativeResolvedEvaluation {
            values: resolved.values,
            point_count: resolved.point_count,
            helicity_ids,
            color_ids,
        })
    }

    #[cfg(feature = "symbolica-runtime")]
    pub fn evaluate_with_precision(
        &mut self,
        momenta: &[String],
        point_count: usize,
        decimal_digits: u32,
    ) -> Result<NativeDecimalEvaluation, RusticolError> {
        if decimal_digits == 0 {
            return Err(RusticolError::unsupported_precision(
                "precision must be a positive number of decimal digits",
            ));
        }
        if decimal_digits == 16 {
            let values = momenta
                .iter()
                .map(|value| {
                    value.parse::<f64>().map_err(|error| {
                        RusticolError::invalid_argument(format!(
                            "could not parse f64 momentum component {value:?}: {error}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let values = self.evaluate_f64(&values, point_count)?;
            return Ok(NativeDecimalEvaluation {
                values: format_decimal_values(values, decimal_digits),
                decimal_digits,
            });
        }
        if self.execution_lane.is_eager() {
            return Err(eager_parity_pending("higher-precision evaluation"));
        }
        if self.execution_lane.is_recurrence() {
            return Err(recurrence_parity_pending("higher-precision evaluation"));
        }
        if decimal_digits == 32 {
            let batch = self.prepare_double_batch(momenta, point_count)?;
            let (values, _profile) = self.runtime.run_double(&batch)?;
            return Ok(NativeDecimalEvaluation {
                values: format_decimal_values(values, decimal_digits),
                decimal_digits,
            });
        }
        let binary_precision = decimal_digits_to_bits(decimal_digits);
        let batch = self.prepare_float_batch(momenta, point_count, binary_precision)?;
        let (values, _profile) = self.runtime.run_float(&batch, binary_precision)?;
        Ok(NativeDecimalEvaluation {
            values: format_decimal_values(values, decimal_digits),
            decimal_digits,
        })
    }

    #[cfg(feature = "symbolica-runtime")]
    pub fn evaluate_resolved_with_precision(
        &mut self,
        momenta: &[String],
        point_count: usize,
        decimal_digits: u32,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<NativeDecimalResolvedEvaluation, RusticolError> {
        if decimal_digits == 0 {
            return Err(RusticolError::unsupported_precision(
                "precision must be a positive number of decimal digits",
            ));
        }
        self.validate_selector_capabilities(helicity_ids.is_some(), color_ids.is_some())?;
        if self.execution_lane.is_eager() {
            return Err(eager_parity_pending("resolved higher-precision evaluation"));
        }
        if self.execution_lane.is_recurrence() {
            return Err(recurrence_parity_pending(
                "resolved higher-precision evaluation",
            ));
        }
        self.record_resolved_warnings(helicity_ids, color_ids)?;
        let selected_helicities = selector_set(helicity_ids, "helicity")?;
        let selected_colors = selector_set(color_ids, "color component")?;
        if decimal_digits == 16 {
            let values = momenta
                .iter()
                .map(|value| {
                    value.parse::<f64>().map_err(|error| {
                        RusticolError::invalid_argument(format!(
                            "could not parse f64 momentum component {value:?}: {error}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let resolved =
                self.evaluate_resolved_f64(&values, point_count, helicity_ids, color_ids)?;
            let totals = resolved.totals();
            return Ok(NativeDecimalResolvedEvaluation {
                values: format_decimal_values(resolved.values, decimal_digits),
                totals: format_decimal_values(totals, decimal_digits),
                point_count: resolved.point_count,
                helicity_ids: resolved.helicity_ids,
                color_ids: resolved.color_ids,
                decimal_digits,
            });
        }
        let physics =
            self.runtime.physics.clone().ok_or_else(|| {
                RusticolError::artifact("resolved physics metadata is unavailable")
            })?;
        if decimal_digits == 32 {
            let batch = self.prepare_double_batch(momenta, point_count)?;
            let (resolved, _profile) = self.runtime.run_resolved_generic(
                &batch,
                None,
                selected_helicities.as_ref(),
                selected_colors.as_ref(),
            )?;
            return decimal_resolved_evaluation(resolved, &physics.manifest, decimal_digits);
        }
        let binary_precision = decimal_digits_to_bits(decimal_digits);
        let batch = self.prepare_float_batch(momenta, point_count, binary_precision)?;
        let (resolved, _profile) = self.runtime.run_resolved_generic(
            &batch,
            Some(binary_precision),
            selected_helicities.as_ref(),
            selected_colors.as_ref(),
        )?;
        decimal_resolved_evaluation(resolved, &physics.manifest, decimal_digits)
    }

    pub fn set_model_parameters(
        &mut self,
        values: &BTreeMap<String, (f64, f64)>,
    ) -> Result<(), RusticolError> {
        for name in values.keys() {
            let parameter = self
                .physics_v1
                .model_parameters
                .iter()
                .find(|parameter| parameter.name == *name)
                .ok_or_else(|| {
                    RusticolError::model_parameter(format!(
                        "model parameter {name:?} is not declared by process {}",
                        self.process
                    ))
                })?;
            if !parameter.mutable {
                return Err(RusticolError::model_parameter(format!(
                    "model parameter {name:?} is derived or immutable"
                )));
            }
        }
        self.runtime
            .apply_model_parameter_overrides(values)
            .map_err(|error| RusticolError::model_parameter(error.to_string()))
    }

    pub fn set_model_parameter(
        &mut self,
        name: &str,
        real: f64,
        imaginary: f64,
    ) -> Result<(), RusticolError> {
        self.set_model_parameters(&BTreeMap::from([(name.to_string(), (real, imaginary))]))
    }

    pub fn set_model_parameters_json(&mut self, path: &Path) -> Result<(), RusticolError> {
        let text = fs::read_to_string(path).map_err(|error| {
            RusticolError::model_parameter(format!(
                "could not read model-parameter JSON {}: {error}",
                path.display()
            ))
        })?;
        let overrides = parse_complex_parameter_overrides(&text, path)
            .map_err(|error| RusticolError::model_parameter(error.to_string()))?;
        self.set_model_parameters(&overrides)
    }

    pub fn mute_warnings(&mut self) {
        self.warnings_muted = true;
    }

    pub fn unmute_warnings(&mut self) {
        self.warnings_muted = false;
    }

    pub fn take_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.pending_warnings)
    }

    pub fn pending_warnings_json(&self) -> Result<String, RusticolError> {
        serde_json::to_string(&self.pending_warnings).map_err(|error| {
            RusticolError::serialization(format!("could not serialize warnings: {error}"))
        })
    }

    pub fn clear_pending_warnings(&mut self) {
        self.pending_warnings.clear();
    }

    pub fn take_warnings_json(&mut self) -> Result<String, RusticolError> {
        serde_json::to_string(&self.take_warnings()).map_err(|error| {
            RusticolError::serialization(format!("could not serialize warnings: {error}"))
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn prepare_f64_batch_profile(
        &self,
        momenta: &[f64],
        point_count: usize,
    ) -> Result<ProfiledPreparedF64Batch, RusticolError> {
        let pack_start = Instant::now();
        let batch = self.prepare_uncrossed_f64_batch(momenta, point_count)?;
        let pack_elapsed = pack_start.elapsed();
        if self.input_crossing_map.is_none() {
            return Ok((batch, pack_elapsed, Duration::ZERO));
        }
        let crossing_start = Instant::now();
        let batch = apply_input_crossing_map(
            batch,
            self.runtime.external_count,
            self.input_crossing_map.as_deref(),
        )?;
        Ok((batch, pack_elapsed, crossing_start.elapsed()))
    }

    fn prepare_uncrossed_f64_batch(
        &self,
        momenta: &[f64],
        point_count: usize,
    ) -> Result<Vec<Vec<[f64; 4]>>, RusticolError> {
        if point_count == 0 {
            return Err(RusticolError::invalid_argument(
                "point_count must be positive",
            ));
        }
        let values_per_point = self
            .runtime
            .external_count
            .checked_mul(4)
            .ok_or_else(|| RusticolError::invalid_argument("momentum shape overflow"))?;
        let expected = point_count
            .checked_mul(values_per_point)
            .ok_or_else(|| RusticolError::invalid_argument("momentum shape overflow"))?;
        if momenta.len() != expected {
            return Err(RusticolError::invalid_argument(format!(
                "momenta contain {} values, expected {expected} for shape ({point_count}, {}, 4)",
                momenta.len(),
                self.runtime.external_count
            )));
        }
        let mut batch = Vec::with_capacity(point_count);
        for point_values in momenta.chunks_exact(values_per_point) {
            let point = point_values
                .chunks_exact(4)
                .map(|components| [components[0], components[1], components[2], components[3]])
                .collect();
            batch.push(point);
        }
        Ok(batch)
    }

    #[cfg(feature = "symbolica-runtime")]
    fn prepare_double_batch(
        &self,
        momenta: &[String],
        point_count: usize,
    ) -> RusticolResult<Vec<Vec<[DoubleFloat; 4]>>> {
        let floats = self.prepare_float_batch(momenta, point_count, 106)?;
        Ok(floats
            .into_iter()
            .map(|point| {
                point
                    .into_iter()
                    .map(|leg| {
                        [
                            leg[0].to_double_float(),
                            leg[1].to_double_float(),
                            leg[2].to_double_float(),
                            leg[3].to_double_float(),
                        ]
                    })
                    .collect()
            })
            .collect())
    }

    #[cfg(feature = "symbolica-runtime")]
    fn prepare_float_batch(
        &self,
        momenta: &[String],
        point_count: usize,
        binary_precision: u32,
    ) -> RusticolResult<Vec<Vec<[Float; 4]>>> {
        let values_per_point =
            validate_flat_momentum_shape(momenta.len(), point_count, self.runtime.external_count)?;
        let mut batch = Vec::with_capacity(point_count);
        for point_values in momenta.chunks_exact(values_per_point) {
            let mut point = Vec::with_capacity(self.runtime.external_count);
            for components in point_values.chunks_exact(4) {
                let values = components
                    .iter()
                    .map(|value| {
                        Float::parse(value, Some(binary_precision)).map_err(|error| {
                            RusticolError::invalid_argument(format!(
                                "could not parse high-precision momentum component {value:?}: {error}"
                            ))
                        })
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                point.push([
                    values[0].clone(),
                    values[1].clone(),
                    values[2].clone(),
                    values[3].clone(),
                ]);
            }
            batch.push(point);
        }
        apply_input_crossing_map_generic(
            &batch,
            self.runtime.external_count,
            self.input_crossing_map.as_deref(),
        )
    }

    pub(super) fn record_resolved_warnings(
        &mut self,
        helicity_ids: Option<&[String]>,
        color_ids: Option<&[String]>,
    ) -> Result<(), RusticolError> {
        if self.warnings_muted {
            return Ok(());
        }
        let physics = self.runtime.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact("resolved evaluation requires regenerated physics metadata")
        })?;
        let mut warnings = Vec::new();
        if physics.manifest.coverage.helicities != "complete" {
            warnings.push((
                "incomplete-helicity-coverage",
                "resolved evaluation contains only the helicities represented by this artifact",
            ));
        }
        if physics.manifest.color_accuracy == crate::ColorAccuracy::Lc
            && physics.manifest.coverage.color != "complete"
        {
            warnings.push((
                "incomplete-color-coverage",
                "resolved evaluation contains only the color components represented by this artifact",
            ));
        }
        let reduction_only_helicity = helicity_ids.is_some_and(|ids| {
            ids.iter().any(|id| {
                physics
                    .helicity_index_by_id
                    .get(id)
                    .and_then(|index| physics.manifest.helicities.get(*index))
                    .is_some_and(|item| !item.computed)
            })
        });
        let reduction_only_color = color_ids.is_some_and(|ids| {
            ids.iter().any(|id| {
                physics
                    .color_index_by_id
                    .get(id)
                    .is_some_and(|index| !physics.color_is_computed(*index))
            })
        });
        if reduction_only_helicity || reduction_only_color {
            warnings.push((
                "reduction-only-selection",
                "the selected resolved component reuses an exact symmetry representative",
            ));
        }
        for (kind, message) in warnings {
            if self.warned_kinds.insert(kind.to_string()) {
                self.pending_warnings.push(message.to_string());
            }
        }
        Ok(())
    }

    fn validate_selector_capabilities(
        &self,
        helicity_requested: bool,
        color_requested: bool,
    ) -> Result<(), RusticolError> {
        if helicity_requested && !self.physics_v1.selectors.helicity {
            return Err(RusticolError::selector(
                "this artifact does not support physical helicity selection",
            ));
        }
        if color_requested {
            if self.runtime.color_accuracy != "lc" {
                return Err(RusticolError::selector(
                    "LC color-flow selection is unavailable for NLC/full artifacts; their resolved color axis is contracted",
                ));
            }
            if !self.physics_v1.selectors.color_flow {
                return Err(RusticolError::selector(
                    "this artifact does not support physical color-flow selection",
                ));
            }
        }
        Ok(())
    }
}

fn validate_flat_momentum_shape(
    value_count: usize,
    point_count: usize,
    external_count: usize,
) -> RusticolResult<usize> {
    if point_count == 0 {
        return Err(RusticolError::invalid_argument(
            "point_count must be positive",
        ));
    }
    let values_per_point = external_count
        .checked_mul(4)
        .ok_or_else(|| RusticolError::invalid_argument("momentum shape overflow"))?;
    let expected = point_count
        .checked_mul(values_per_point)
        .ok_or_else(|| RusticolError::invalid_argument("momentum shape overflow"))?;
    if value_count != expected {
        return Err(RusticolError::invalid_argument(format!(
            "momenta contain {value_count} values, expected {expected} for shape ({point_count}, {external_count}, 4)"
        )));
    }
    Ok(values_per_point)
}

#[cfg(feature = "symbolica-runtime")]
fn format_decimal_values<T: std::fmt::LowerExp>(
    values: Vec<T>,
    decimal_digits: u32,
) -> Vec<String> {
    let digits = decimal_digits as usize;
    values
        .into_iter()
        .map(|value| format!("{value:.digits$e}"))
        .collect()
}

#[cfg(feature = "symbolica-runtime")]
fn decimal_resolved_evaluation<T>(
    resolved: ResolvedValues<T>,
    physics: &ProcessPhysicsV1,
    decimal_digits: u32,
) -> RusticolResult<NativeDecimalResolvedEvaluation>
where
    T: RusticolHighPrecisionNumber + std::fmt::LowerExp,
    Complex<T>: Real + EvaluationDomain,
{
    let component_count = resolved.helicity_indices.len() * resolved.color_indices.len();
    if component_count == 0 {
        return Err(RusticolError::internal(
            "resolved evaluation produced an empty component axis",
        ));
    }
    let mut totals = Vec::with_capacity(resolved.point_count);
    for point in resolved.values.chunks(component_count) {
        let mut total = T::new_zero();
        for value in point {
            total += value.clone();
        }
        totals.push(total);
    }
    let helicity_ids = resolved
        .helicity_indices
        .iter()
        .map(|index| physics.helicities[*index].id.clone())
        .collect();
    let color_ids = resolved
        .color_indices
        .iter()
        .map(|index| physics.color_components[*index].id().to_string())
        .collect();
    Ok(NativeDecimalResolvedEvaluation {
        values: format_decimal_values(resolved.values, decimal_digits),
        totals: format_decimal_values(totals, decimal_digits),
        point_count: resolved.point_count,
        helicity_ids,
        color_ids,
        decimal_digits,
    })
}

#[cfg(feature = "symbolica-runtime")]
fn eager_parity_pending(feature: &str) -> RusticolError {
    RusticolError::unsupported_runtime_capability(
        EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY,
        format!(
            "eager {feature} is not provided by the native f64 Direct-Arena ABI; \
             use the public Python Runtime exact-evaluation path for precision above 16 digits"
        ),
    )
}

#[cfg(feature = "symbolica-runtime")]
fn recurrence_parity_pending(feature: &str) -> RusticolError {
    RusticolError::unsupported_runtime_capability(
        RECURRENCE_RUNTIME_CAPABILITY,
        format!(
            "recurrence {feature} is not available in the initial f64 runtime slice; use compiled execution for this operation"
        ),
    )
}

#[cfg(test)]
mod runtime_capability_cutover_tests {
    use super::*;

    #[test]
    fn retired_eager_v2_capabilities_fail_with_regeneration_guidance() {
        for capability in [
            EAGER_DAG_RUNTIME_CAPABILITY,
            EAGER_LC_TOPOLOGY_REPLAY_RUNTIME_CAPABILITY,
        ] {
            let error = ensure_selected_runtime_capabilities_supported(&[capability.to_string()])
                .expect_err("retired eager capability must fail before generic preflight");
            assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
            assert!(error.to_string().contains("legacy eager plan-v2"));
            assert!(error.to_string().contains("regenerate"));
        }
    }
}
