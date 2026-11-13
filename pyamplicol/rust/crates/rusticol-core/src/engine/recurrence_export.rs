// SPDX-License-Identifier: 0BSD

//! Cold, static export of the resolved batched LC recurrence program.
//!
//! This reads the same process binding and source domains as the loader and
//! resolves them at the runtime's current model parameters. It never evaluates
//! a phase-space point and adds no state to the ordinary execution path.

use super::recurrence_export_executors::{export_executors, export_source_templates};
use super::recurrence_process_pack::{ProcessDirectExecutorBinding, ProcessDirectExecutorPack};
use super::*;
use crate::engine::evaluator::recurrence_source_direct::DirectSourceDispatchDomainSpec;
use crate::recurrence::direct_backend::{DirectCurrentClearRange, DirectStageClearRanges};
use crate::recurrence::direct_export::{
    DirectExportInstruction, DirectExportKernelBinding, DirectExportSource, ExportComplex,
    export_direct_body,
};
use crate::recurrence::direct_runtime::DirectReplaySelectorPlan;
use crate::recurrence::{DirectNodeKind, DirectRecurrencePlan, RecurrenceStrategy};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde_json::json;

/// The versioned, self-contained result of [`NativeRuntime::export_recurrence`].
#[derive(Clone, Debug, Serialize)]
pub struct RecurrenceExport {
    pub schema_version: u32,
    pub rusticol_version: String,
    pub artifact_id: String,
    /// This schema describes the tiled f64 path, including its explicit FMAs.
    pub execution_semantics: String,
    pub parameters: Vec<RecurrenceExportParameter>,
    pub programs: Vec<RecurrenceExportProgram>,
    pub kernels: Vec<RecurrenceExportKernel>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RecurrenceExportParameter {
    pub name: String,
    pub value: ExportComplex,
}

#[derive(Clone, Debug, Serialize)]
pub struct RecurrenceExportKernel {
    pub sha256: String,
    /// Base64-encoded unmodified saved `.plane.symjit` application bytes.
    pub application_base64: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RecurrenceExportLeg {
    pub pdg: i32,
    pub name: String,
    pub role: crate::ParticleRole,
}

#[derive(Clone, Debug, Serialize)]
pub struct RecurrenceExportCrossing {
    pub program_leg: usize,
    pub caller_leg: usize,
    pub sign: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RecurrenceExportMomentumTerm {
    pub leg: u32,
    pub coefficient: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RecurrenceExportCurrent {
    pub semantic_id: u32,
    /// Instructions address these physical component offsets. Ranges may be
    /// reused by semantic currents with disjoint lifetimes.
    pub current: u32,
    pub component_count: u16,
    pub physical_type: Option<String>,
    pub initial_state: String,
    pub stage: u16,
    pub first_use: u32,
    pub last_use: u32,
}

#[derive(Clone, Debug, Serialize)]
pub struct RecurrenceExportHelicity {
    pub id: String,
    pub helicities: Vec<i32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RecurrenceExportAlphaS {
    pub parameter: String,
    pub value: ExportComplex,
    /// Present only for a homogeneous QCD order. Mixed-order programs have
    /// no single alpha_s exponent.
    pub squared_amplitude_power: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RecurrenceExportReduction {
    pub helicity_output: usize,
    pub amplitude: u32,
    /// First perform the non-fused complex multiply by this resolved replay
    /// scale, even for (1,0). Then compute fma(re,re,im*im), multiply by weight,
    /// and add to the output in the serialized reduction order.
    pub amplitude_scale: ExportComplex,
    /// Union execution skips replay entirely, rather than multiplying by one.
    pub apply_amplitude_scale: bool,
    pub weight: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RecurrenceExportRun {
    /// Source variants in the encounter order of `program.body` source slots.
    pub sources: Vec<DirectExportSource>,
    pub reduction: Vec<RecurrenceExportReduction>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RecurrenceExportProgram {
    pub process_key: String,
    pub process: String,
    pub representative_process_key: String,
    pub representative_colour_id: String,
    pub colour_id: String,
    pub external_count: usize,
    pub legs: Vec<RecurrenceExportLeg>,
    pub caller_legs: Vec<RecurrenceExportLeg>,
    pub crossing: Option<Vec<RecurrenceExportCrossing>>,
    /// Ordered sums, each starting at positive zero. Momentum terms refer to
    /// program legs; apply `crossing` to caller momenta before these sums.
    pub momentum_forms: Vec<Vec<RecurrenceExportMomentumTerm>>,
    pub currents: Vec<RecurrenceExportCurrent>,
    pub current_component_count: u32,
    pub amplitude_count: u32,
    pub scales: Vec<ExportComplex>,
    pub factors: Vec<ExportComplex>,
    pub helicity_outputs: Vec<RecurrenceExportHelicity>,
    pub body: Vec<DirectExportInstruction>,
    pub kernel_bindings: Vec<DirectExportKernelBinding>,
    /// Execute the shared body and each run's reduction in order. Runs may
    /// reuse one arena: their clears and initializing writes define live data.
    /// Initialize the output vector to positive zero once before the first run.
    pub runs: Vec<RecurrenceExportRun>,
    /// Apply once, after every run's reductions, only if numerator differs
    /// from denominator: `(value * (numerator as f64)) / (denominator as f64)`.
    /// Do not precompute their ratio: its rounding can differ.
    pub group_factor: NativeRationalFactor,
    pub alpha_s: Option<RecurrenceExportAlphaS>,
}

struct ExportContext {
    artifact: VerifiedArtifact,
    executors: ProcessDirectExecutorPack,
    domains: Vec<DirectSourceDispatchDomainSpec>,
    payload_root: PathBuf,
    legs: Vec<RecurrenceExportLeg>,
    colour_ids: Vec<String>,
    alpha_s_parameter: Option<String>,
}

impl NativeRuntime {
    /// Export every requested LC colour of this selected process at its current
    /// parameters, without evaluating momenta. `None` selects all colours.
    ///
    /// Process-group members preserve their public identity, crossing and exact
    /// group factor. Export other provider processes using separately selected
    /// runtimes and concatenate their `programs`, deduplicating kernels by SHA.
    pub fn export_recurrence(
        &mut self,
        colours: Option<&[String]>,
    ) -> RusticolResult<RecurrenceExport> {
        if !matches!(self.execution_lane, NativeExecutionLane::Recurrence(_)) {
            return Err(RusticolError::compatibility(
                "recurrence export requires an LC recurrence artifact",
            ));
        }
        let context = self.recurrence_export_context()?;
        let physics = self.physics_v1.get()?.clone();
        if physics.color_accuracy.as_str() != "lc" {
            return Err(RusticolError::compatibility(
                "recurrence export requires LC colour accuracy",
            ));
        }
        let requested = colours.map(|values| values.iter().cloned().collect::<BTreeSet<_>>());
        if let Some(requested) = &requested {
            for colour in requested {
                if !physics
                    .color_components
                    .iter()
                    .any(|component| component.id() == colour)
                {
                    return Err(RusticolError::selector(format!(
                        "unknown recurrence export colour {colour:?} for process {:?} ({}); \
                         colour-flow IDs are per process, this process has: {available}",
                        self.process_key, self.process
                    )));
                }
            }
                    // Colour-flow IDs name leg permutations of one process,
                    // so the same ID can be absent from another process.
                    let available = physics
                        .color_components
                        .iter()
                        .map(|component| component.id())
                        .collect::<Vec<_>>()
                        .join(", ");
        }
        let parameters = self
            .runtime
            .model_parameter_runtime_slots
            .keys()
            .map(|name| {
                let (re, im) = self.runtime.current_model_parameter_value(name)?;
                Ok(RecurrenceExportParameter {
                    name: name.clone(),
                    value: ExportComplex { re, im },
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let group_factor = self.process_group_factor();
        let crossing: Option<Vec<RecurrenceExportCrossing>> =
            self.input_crossing_map.as_ref().map(|entries| {
                entries
                    .iter()
                    .map(|entry| RecurrenceExportCrossing {
                        program_leg: entry.target_index,
                        caller_leg: entry.source_index,
                        sign: entry.sign,
                    })
                    .collect()
            });
        let NativeExecutionLane::Recurrence(lane) = &mut self.execution_lane else {
            unreachable!()
        };
        lane.prepare_parameters(&self.runtime)?;
        let (re, im) = lane.scheduler.parameters_mut();
        let scales = split_values(re, im);
        let (re, im) = lane.scheduler.factors_mut();
        let factors = split_values(re, im);
        let plan = lane.scheduler.plan();
        let executors = export_executors(&context.executors.descriptors, &scales)?;
        let alpha_s_parameter =
            resolve_alpha_s_parameter(context.alpha_s_parameter.as_deref(), &parameters);
        let alpha_s = export_alpha_s(
            &parameters,
            plan.amplitude_orders(),
            self.runtime.normalization_qcd_coupling_power,
            alpha_s_parameter.as_deref(),
        );
        let runtime_physics = self.runtime.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact("recurrence export requires reduction metadata")
        })?;
        let caller_legs = export_legs(&physics.external_particles);
        let helicity_outputs: Vec<_> = physics
            .helicities
            .iter()
            .map(|helicity| RecurrenceExportHelicity {
                id: helicity.id.clone(),
                helicities: helicity.values.clone(),
            })
            .collect();
        let currents = export_currents(plan);
        let mut programs = Vec::new();
        for (colour_index, colour) in physics.color_components.iter().enumerate() {
            if requested
                .as_ref()
                .is_some_and(|ids| !ids.contains(colour.id()))
            {
                continue;
            }
            let mut program = RecurrenceExportProgram {
                process_key: self.process_key.clone(),
                process: self.process.clone(),
                representative_process_key: self.runtime.key.clone(),
                representative_colour_id: context.colour_ids[colour_index].clone(),
                colour_id: colour.id().to_owned(),
                external_count: self.runtime.external_count,
                legs: context.legs.clone(),
                caller_legs: caller_legs.clone(),
                crossing: crossing.clone(),
                momentum_forms: Vec::new(),
                currents: currents.clone(),
                current_component_count: plan.current_arena_components()
                    + lane.scheduler.export_fanout().scratch_component_count(),
                amplitude_count: plan.amplitude_destination_count(),
                scales: scales.clone(),
                factors: factors.clone(),
                helicity_outputs: helicity_outputs.clone(),
                body: Vec::new(),
                kernel_bindings: Vec::new(),
                runs: Vec::new(),
                group_factor,
                alpha_s: alpha_s.clone(),
            };
            match &lane.selectors {
                RecurrenceNativeSelectors::TopologyReplay {
                    replay_selectors,
                    direct_helicity_to_physics,
                } => {
                    let selector = &replay_selectors[colour_index];
                    program.momentum_forms = export_momentum_forms(plan, Some(selector))?;
                    if runtime_physics.color_is_computed(colour_index) {
                        let mut sources =
                            export_source_templates(&context.domains, plan, &scales, None)?;
                        for source in sources.values_mut() {
                            source.leg = selector
                                .mapped_external_source_slot(source.leg)
                                .ok_or_else(|| {
                                    RusticolError::integrity(
                                        "export source leg permutation is incomplete",
                                    )
                                })?;
                        }
                        let clears =
                            export_stage_clears(plan, Some(selector.representative_flow_id()))?;
                        let (body, bindings, source_rows) = export_direct_body(
                            plan,
                            selector.selected_row_groups(),
                            &clears,
                            &executors,
                            Some(lane.scheduler.export_fanout()),
                        )?;
                        program.body = body;
                        program.kernel_bindings = bindings;
                        let mut reduction = Vec::new();
                        let (re, im) = selector.amplitude_scale();
                        let amplitude_scale = ExportComplex { re, im };
                        for destination in
                            plan.amplitude_destinations().iter().filter(|destination| {
                                destination.target_sector_id == selector.representative_flow_id()
                            })
                        {
                            let direct_id = destination.target_helicity_id_or_sentinel as usize;
                            let mapped =
                                *selector.helicity_map().get(direct_id).ok_or_else(|| {
                                    RusticolError::integrity(
                                        "export replay helicity map is incomplete",
                                    )
                                })?;
                            let physics_index = *direct_helicity_to_physics
                                .get(mapped as usize)
                                .ok_or_else(|| {
                                    RusticolError::integrity(
                                        "export replay helicity index is absent",
                                    )
                                })?;
                            let helicity = &runtime_physics.manifest.helicities[physics_index];
                            if !helicity.computed
                                || helicity.structural_zero
                                || helicity.coefficient == 0.0
                            {
                                continue;
                            }
                            for &output in
                                &runtime_physics.helicity_members_by_representative[physics_index]
                            {
                                let weight = runtime_physics.manifest.helicities[output]
                                    .coefficient
                                    * colour.coefficient()
                                    * self.runtime.normalization_factor;
                                reduction.push(RecurrenceExportReduction {
                                    helicity_output: output,
                                    amplitude: destination.id,
                                    amplitude_scale,
                                    apply_amplitude_scale: true,
                                    weight,
                                });
                            }
                        }
                        program.runs.push(RecurrenceExportRun {
                            sources: ordered_sources(&sources, &source_rows)?,
                            reduction,
                        });
                    }
                }
                RecurrenceNativeSelectors::AllFlowUnion {
                    helicity_selectors_by_physics,
                    destination_by_public_flow,
                } => {
                    program.momentum_forms = export_momentum_forms(plan, None)?;
                    let mut source_rows = Vec::new();
                    if runtime_physics.color_is_computed(colour_index) {
                        for (output, helicity) in
                            runtime_physics.manifest.helicities.iter().enumerate()
                        {
                            if !helicity.computed
                                || helicity.structural_zero
                                || helicity.coefficient == 0.0
                            {
                                continue;
                            }
                            let selector = helicity_selectors_by_physics[output]
                                .as_ref()
                                .ok_or_else(|| {
                                    RusticolError::integrity(
                                        "export union helicity selector is absent",
                                    )
                                })?;
                            let sources = export_source_templates(
                                &context.domains,
                                plan,
                                &scales,
                                Some(selector.resolved_helicity_id()),
                            )?;
                            if program.runs.is_empty() {
                                let clears = export_stage_clears(plan, None)?;
                                let (body, bindings, rows) = export_direct_body(
                                    plan,
                                    plan.row_groups(),
                                    &clears,
                                    &executors,
                                    Some(lane.scheduler.export_fanout()),
                                )?;
                                program.body = body;
                                program.kernel_bindings = bindings;
                                source_rows = rows;
                            }
                            let weight = helicity.coefficient
                                * colour.coefficient()
                                * self.runtime.normalization_factor;
                            program.runs.push(RecurrenceExportRun {
                                sources: ordered_sources(&sources, &source_rows)?,
                                reduction: vec![RecurrenceExportReduction {
                                    helicity_output: output,
                                    amplitude: destination_by_public_flow[colour_index],
                                    amplitude_scale: ExportComplex::ONE,
                                    apply_amplitude_scale: false,
                                    weight,
                                }],
                            });
                        }
                    }
                }
                RecurrenceNativeSelectors::ContractedColorUnion { .. } => {
                    return Err(RusticolError::compatibility(
                        "recurrence export requires LC colour flows",
                    ));
                }
            }
            refine_current_metadata(&mut program);
            programs.push(program);
        }
        let payloads = context
            .artifact
            .evaluator_payload_store(&context.payload_root)?;
        let mut kernels = BTreeMap::new();
        for descriptor in &context.executors.descriptors {
            if let ProcessDirectExecutorBinding::Jit(binding) = &descriptor.binding {
                let sha256: String = binding
                    .source_application_sha256
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect();
                if !kernels.contains_key(&sha256) {
                    let source = payloads.source(&binding.source_application_path)?;
                    let application = source.read()?;
                    kernels.insert(
                        sha256.clone(),
                        RecurrenceExportKernel {
                            sha256,
                            application_base64: BASE64.encode(application.as_ref()),
                        },
                    );
                }
            }
        }
        Ok(RecurrenceExport {
            schema_version: 2,
            rusticol_version: env!("CARGO_PKG_VERSION").to_owned(),
            artifact_id: self.artifact_id.clone(),
            execution_semantics: "rusticol-batched-f64-v1".to_owned(),
            parameters,
            programs,
            kernels: kernels.into_values().collect(),
        })
    }

    pub fn export_recurrence_json(&mut self, colours: Option<&[String]>) -> RusticolResult<String> {
        serde_json::to_string(&self.export_recurrence(colours)?).map_err(|error| {
            RusticolError::serialization(format!("could not serialize recurrence export: {error}"))
        })
    }

    fn recurrence_export_context(&self) -> RusticolResult<ExportContext> {
        if let Some(bootstrap) = VerifiedArtifact::try_open_recurrence_bootstrap(
            &self.root,
            Some(&self.representative_process_id),
        )? {
            let crate::artifact::RecurrenceBootstrapArtifact {
                mut artifact,
                selection,
                ready_execution,
                evaluator_payload_container,
                physics,
            } = bootstrap;
            if artifact.manifest().artifact_id != self.artifact_id {
                return Err(RusticolError::integrity(
                    "recurrence artifact changed since the runtime was loaded",
                ));
            }
            let evaluator_root = artifact
                .root()
                .join("processes")
                .join(&selection.process.id);
            let loaded = super::recurrence_load::load_ready_plan(
                &artifact,
                &evaluator_root,
                &ready_execution,
                &ready_execution.primary_plan,
                super::recurrence_load::RecurrenceReadyPlanRoleV1::Primary,
            )?;
            let domains = super::recurrence_load::decode_ready_source_domains(
                &ready_execution.runtime.source_domains,
                &loaded.plan,
            )?;
            artifact.install_evaluator_payload_container(&evaluator_payload_container)?;
            let payload_root = artifact.root().join(confined_internal_path(
                &ready_execution.kernel_pack.payload_root,
                "recurrence export payload root",
            )?);
            let legs = export_legs(&physics.external_particles);
            let colour_ids = ready_execution.runtime.public_flow_public_ids.clone();
            let alpha_s_parameter = load_alpha_s_parameter(&artifact, &selection.process.id)?;
            return Ok(ExportContext {
                artifact,
                executors: loaded.process_executors,
                domains,
                payload_root,
                legs,
                colour_ids,
                alpha_s_parameter,
            });
        }
        let artifact = VerifiedArtifact::open(&self.root)?;
        if artifact.manifest().artifact_id != self.artifact_id {
            return Err(RusticolError::integrity(
                "recurrence artifact changed since the runtime was loaded",
            ));
        }
        let selection = artifact.select_process(Some(&self.representative_process_id))?;
        let (manifest, evaluator_root) = load_verified_evaluator(&artifact, &selection)?;
        let LoadedExecutionManifest::Recurrence(manifest) = manifest else {
            return Err(RusticolError::compatibility(
                "recurrence export requires a recurrence artifact",
            ));
        };
        let loaded = super::recurrence_load::load_plan_summary(
            &artifact,
            &evaluator_root,
            &manifest,
            &manifest.plan,
        )?;
        let domains = super::recurrence_load::build_direct_source_domains(
            &loaded.plan,
            &manifest.runtime_metadata,
            &self.runtime.model_parameter_runtime_slots,
        )?;
        let payload_root = artifact.root().join(confined_internal_path(
            &manifest.kernel_pack.payload_root,
            "recurrence export payload root",
        )?);
        let physics = ProcessPhysicsV1::from_json(
            &artifact.read_payload(&selection.process.physics_path)?,
            &selection.process.physics_path,
        )?;
        let legs = export_legs(&physics.external_particles);
        let colour_ids = physics
            .color_components
            .iter()
            .map(|colour| colour.id().to_owned())
            .collect();
        let alpha_s_parameter = load_alpha_s_parameter(&artifact, &selection.process.id)?;
        Ok(ExportContext {
            artifact,
            executors: loaded.process_executors,
            domains,
            payload_root,
            legs,
            colour_ids,
            alpha_s_parameter,
        })
    }
}

fn split_values(re: &[f64], im: &[f64]) -> Vec<ExportComplex> {
    re.iter()
        .zip(im)
        .map(|(&re, &im)| ExportComplex { re, im })
        .collect()
}

fn export_legs(particles: &[crate::ExternalParticle]) -> Vec<RecurrenceExportLeg> {
    particles
        .iter()
        .map(|particle| RecurrenceExportLeg {
            pdg: particle.pdg,
            name: particle.particle.clone(),
            role: particle.role,
        })
        .collect()
}

fn ordered_sources(
    sources: &BTreeMap<u32, DirectExportSource>,
    rows: &[u32],
) -> RusticolResult<Vec<DirectExportSource>> {
    rows.iter()
        .map(|row| {
            sources
                .get(row)
                .cloned()
                .ok_or_else(|| RusticolError::integrity("export source slot has no variant"))
        })
        .collect()
}

fn refine_current_metadata(program: &mut RecurrenceExportProgram) {
    let mut source_families = BTreeMap::new();
    for source in program.runs.iter().flat_map(|run| &run.sources) {
        source_families
            .entry(source.current)
            .or_insert_with(|| source.family.clone());
    }
    let mut ranges: Vec<_> = program
        .body
        .iter()
        .filter_map(|instruction| match instruction {
            DirectExportInstruction::Clear {
                current,
                component_count,
            } => Some((*current, current + component_count)),
            _ => None,
        })
        .collect();
    ranges.sort_unstable();
    // Merge overlapping physical spans once. A predecessor lookup then finds
    // the only possible covering clear, including aliased semantic currents.
    let mut cleared: Vec<(u32, u32)> = Vec::new();
    for (base, end) in ranges {
        if let Some((_, previous_end)) = cleared.last_mut()
            && base <= *previous_end
        {
            *previous_end = (*previous_end).max(end);
            continue;
        }
        cleared.push((base, end));
    }
    for current in &mut program.currents {
        if current.initial_state == "source" {
            current.physical_type = source_families.get(&current.current).cloned();
        } else if cleared[..cleared.partition_point(|(base, _)| *base <= current.current)]
            .last()
            .is_some_and(|(_, end)| current.current + u32::from(current.component_count) <= *end)
        {
            current.initial_state = "zero".to_owned();
        }
    }
}

fn export_currents(plan: &DirectRecurrencePlan) -> Vec<RecurrenceExportCurrent> {
    plan.currents()
        .iter()
        .map(|current| RecurrenceExportCurrent {
            semantic_id: current.semantic_current_id,
            current: current.component_base,
            component_count: current.component_count,
            physical_type: None,
            initial_state: if current.node_kind == DirectNodeKind::Source {
                "source"
            } else {
                "first_instruction"
            }
            .to_owned(),
            stage: current.stage,
            first_use: current.first_use,
            last_use: current.last_use,
        })
        .collect()
}

fn export_momentum_forms(
    plan: &DirectRecurrencePlan,
    replay: Option<&DirectReplaySelectorPlan>,
) -> RusticolResult<Vec<Vec<RecurrenceExportMomentumTerm>>> {
    plan.momentum_forms()
        .iter()
        .map(|form| {
            plan.momentum_terms()
                [form.term_start as usize..form.term_start as usize + form.term_count as usize]
                .iter()
                .map(|term| {
                    let (leg, sign) = if let Some(replay) = replay {
                        (
                            replay
                                .mapped_external_source_slot(term.source_slot)
                                .ok_or_else(|| {
                                    RusticolError::integrity(
                                        "export source permutation is incomplete",
                                    )
                                })?,
                            replay
                                .mapped_external_source_sign(term.source_slot)
                                .ok_or_else(|| {
                                    RusticolError::integrity(
                                        "export momentum-sign map is incomplete",
                                    )
                                })?,
                        )
                    } else {
                        (term.source_slot, 1)
                    };
                    Ok(RecurrenceExportMomentumTerm {
                        leg,
                        coefficient: f64::from(term.coefficient) * f64::from(sign),
                    })
                })
                .collect()
        })
        .collect()
}

fn export_stage_clears(
    plan: &DirectRecurrencePlan,
    sector: Option<u32>,
) -> RusticolResult<Vec<DirectStageClearRanges>> {
    if sector.is_none() && plan.strategy() != RecurrenceStrategy::AllFlowUnion {
        return Ok(Vec::new());
    }
    let mut stages = BTreeMap::<u16, Vec<DirectCurrentClearRange>>::new();
    for current in plan.currents() {
        if current.node_kind != DirectNodeKind::Current {
            continue;
        }
        if let Some(sector) = sector
            && !plan.selector_domain_contains(current.selector_domain_id, sector)?
        {
            continue;
        }
        stages
            .entry(current.stage)
            .or_default()
            .push(DirectCurrentClearRange {
                component_base: current.component_base,
                component_count: u32::from(current.component_count),
            });
    }
    Ok(stages
        .into_iter()
        .map(|(stage, ranges)| DirectStageClearRanges {
            stage,
            ranges: ranges.into_boxed_slice(),
        })
        .collect())
}

fn declared_alpha_s_parameter(model: &Value) -> Option<String> {
    let ir = model
        .get("ir")
        .or_else(|| model.get("model"))
        .unwrap_or(model);
    if let Some(name) = [
        "/extensions/alpha_s_parameter",
        "/model/extensions/alpha_s_parameter",
        "/ir/extensions/alpha_s_parameter",
    ]
    .into_iter()
    .find_map(|path| model.pointer(path).and_then(Value::as_str))
    .filter(|name| !name.is_empty())
    {
        return Some(name.to_owned());
    }
    let mut candidates = ir
        .get("parameters")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|parameter| {
            parameter["nature"].as_str() == Some("external")
                && parameter["lhablock"]
                    .as_str()
                    .is_some_and(|block| block.eq_ignore_ascii_case("SMINPUTS"))
                && parameter["lhacode"] == json!([3])
        })
        .filter_map(|parameter| parameter["name"].as_str());
    let candidate = candidates.next()?;
    candidates.next().is_none().then(|| candidate.to_owned())
}

fn load_alpha_s_parameter(
    artifact: &VerifiedArtifact,
    process_key: &str,
) -> RusticolResult<Option<String>> {
    if let Some(payload) = artifact
        .manifest()
        .payloads
        .iter()
        .find(|payload| payload.role == PayloadRole::CompiledModel)
    {
        let model: Value = serde_json::from_slice(&artifact.read_payload(&payload.path)?)
            .map_err(|error| RusticolError::artifact(format!("compiled model: {error}")))?;
        if let Some(name) = declared_alpha_s_parameter(&model) {
            return Ok(Some(name));
        }
    }
    for payload in &artifact.manifest().payloads {
        if payload.role != PayloadRole::SdkMetadata
            || !payload.path.ends_with("/umami/metadata.json")
        {
            continue;
        }
        let provider: Value = serde_json::from_slice(&artifact.read_payload(&payload.path)?)
            .map_err(|error| RusticolError::artifact(format!("UMAMI metadata: {error}")))?;
        let process = provider["runtime_processes"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|process| process["id"].as_str() == Some(process_key));
        if let Some(name) = process
            .and_then(|process| process["alpha_s_parameter"].as_str())
            .or_else(|| {
                provider
                    .pointer("/provider/alpha_s_parameter")
                    .and_then(Value::as_str)
            })
        {
            return Ok(Some(name.to_owned()));
        }
    }
    Ok(None)
}

/// Prefer a model or provider declaration that names a runtime parameter.
/// Otherwise use the runtime's own normalization contract whenever the runtime
/// defines its parameter: built-in models, compiled or not, apply
/// `(4 pi alpha_s)^p` through it (see `refresh_normalization_factor`). This
/// reads the runtime's contract, not a guess at an imported model's names.
fn resolve_alpha_s_parameter(
    declared: Option<&str>,
    parameters: &[RecurrenceExportParameter],
) -> Option<String> {
    let defined = |name: &str| parameters.iter().any(|parameter| parameter.name == name);
    declared
        .filter(|name| defined(name))
        .or_else(|| {
            defined(super::model_parameters::NORMALIZATION_ALPHA_S_PARAMETER)
                .then_some(super::model_parameters::NORMALIZATION_ALPHA_S_PARAMETER)
        })
        .map(str::to_owned)
}

fn export_alpha_s(
    parameters: &[RecurrenceExportParameter],
    orders: &crate::recurrence::DirectAmplitudeOrders,
    normalization_power: usize,
    parameter_name: Option<&str>,
) -> Option<RecurrenceExportAlphaS> {
    let name = parameter_name?;
    let parameter = parameters.iter().find(|parameter| parameter.name == name)?;
    let qcd = orders.names().iter().position(|name| name == "QCD");
    let mut powers = (0..orders.component_count()).map(|index| {
        qcd.map_or(0, |position| {
            orders.component(index).expect("validated order component")[position]
        })
    });
    let first = powers.next();
    let homogeneous = first.filter(|first| powers.all(|power| power == *first));
    // The semantic QCD order already includes couplings factored into the
    // global builtin normalization. Adding normalization_power double-counts
    // them. Without order metadata only the explicit normalization is known.
    let squared_amplitude_power = if orders.names().is_empty() {
        u32::try_from(normalization_power)
            .ok()
            .filter(|power| *power > 0)
    } else {
        homogeneous
    };
    Some(RecurrenceExportAlphaS {
        parameter: parameter.name.clone(),
        value: parameter.value,
        squared_amplitude_power,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_s_export_does_not_double_count_factored_builtin_couplings() {
        let mut parts = crate::recurrence::valid_direct_plan_parts_fixture();
        parts.amplitude_orders.names = vec!["QCD".to_owned()];
        parts.amplitude_orders.powers = vec![3];
        let plan = DirectRecurrencePlan::new(parts).unwrap();
        let parameters = [RecurrenceExportParameter {
            name: "normalization.alpha_s_me_check".to_owned(),
            value: ExportComplex { re: 0.118, im: 0.0 },
        }];
        let exported = export_alpha_s(
            &parameters,
            plan.amplitude_orders(),
            3,
            Some("normalization.alpha_s_me_check"),
        )
        .unwrap();
        assert_eq!(exported.squared_amplitude_power, Some(3));
        assert_eq!(exported.value, parameters[0].value);

        let without_orders = crate::recurrence::valid_direct_plan_fixture();
        assert_eq!(
            export_alpha_s(
                &parameters,
                without_orders.amplitude_orders(),
                3,
                Some("normalization.alpha_s_me_check")
            )
            .unwrap()
            .squared_amplitude_power,
            Some(3),
        );
    }

    #[test]
    fn alpha_s_uses_declared_custom_parameter_and_keeps_mixed_order_identity() {
        let model = json!({ "ir": { "parameters": [
            { "name": "custom_strong_input", "nature": "external", "lhablock": "sminputs", "lhacode": [3] },
            { "name": "aS", "nature": "external", "lhablock": "UNRELATED", "lhacode": [1] }
        ]}});
        let name = declared_alpha_s_parameter(&model).unwrap();
        assert_eq!(name, "custom_strong_input");
        let parameters = [RecurrenceExportParameter {
            name: name.clone(),
            value: ExportComplex { re: 0.117, im: 0.0 },
        }];
        let orders = crate::recurrence::DirectAmplitudeOrders {
            names: vec!["QCD".to_owned()],
            component_count: 2,
            powers: vec![2, 3],
            closure_components: vec![0, 1],
        };
        let exported = export_alpha_s(&parameters, &orders, 0, Some(&name)).unwrap();
        assert_eq!(exported.parameter, name);
        assert_eq!(exported.value, parameters[0].value);
        assert_eq!(exported.squared_amplitude_power, None);
        assert!(export_alpha_s(&parameters, &orders, 0, None).is_none());

        let explicit = json!({ "extensions": { "alpha_s_parameter": "declared_strong" }, "ir": { "parameters": [] } });
        assert_eq!(
            declared_alpha_s_parameter(&explicit).as_deref(),
            Some("declared_strong")
        );
        let nested = json!({
            "model": { "extensions": { "alpha_s_parameter": "nested_strong" } },
            "ir": { "parameters": [{ "name": "nested_strong", "nature": "external", "lhablock": "CUSTOM", "lhacode": [7] }] }
        });
        assert_eq!(
            declared_alpha_s_parameter(&nested).as_deref(),
            Some("nested_strong")
        );
        let name_only = json!({ "ir": { "parameters": [{ "name": "aS", "nature": "external" }] } });
        assert_eq!(declared_alpha_s_parameter(&name_only), None);
    }

    #[test]
    fn alpha_s_falls_back_to_the_runtime_normalization_contract() {
        let parameter = |name: &str| RecurrenceExportParameter {
            name: name.to_owned(),
            value: ExportComplex { re: 0.118, im: 0.0 },
        };
        let normalization = super::super::model_parameters::NORMALIZATION_ALPHA_S_PARAMETER;
        // A compiled built-in model declares nothing, but its runtime applies
        // alpha_s through the normalization parameter.
        let builtin = [parameter("particle.23.mass"), parameter(normalization)];
        assert_eq!(
            resolve_alpha_s_parameter(None, &builtin).as_deref(),
            Some(normalization)
        );
        // A declaration naming a runtime parameter wins; one naming no runtime
        // parameter cannot hide the contract.
        let ufo = [parameter("aS"), parameter("MZ")];
        assert_eq!(
            resolve_alpha_s_parameter(Some("aS"), &ufo).as_deref(),
            Some("aS")
        );
        assert_eq!(
            resolve_alpha_s_parameter(Some("alpha_s"), &builtin).as_deref(),
            Some(normalization)
        );
        assert_eq!(resolve_alpha_s_parameter(Some("alpha_s"), &ufo), None);
        assert_eq!(resolve_alpha_s_parameter(None, &ufo), None);
    }
}
