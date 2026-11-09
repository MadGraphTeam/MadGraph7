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
    DirectExportInstruction, ExportComplex, export_direct_body,
};
use crate::recurrence::direct_runtime::DirectReplaySelectorPlan;
use crate::recurrence::{DirectNodeKind, DirectRecurrencePlan, RecurrenceStrategy};

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
    /// Unmodified saved `.plane.symjit` application bytes.
    pub application: Vec<u8>,
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
    pub body: Vec<DirectExportInstruction>,
    pub reduction: Vec<RecurrenceExportReduction>,
}

#[derive(Clone, Debug, Serialize)]
pub struct RecurrenceExportProgram {
    pub process_key: String,
    pub process: String,
    pub representative_process_key: String,
    pub colour_id: String,
    pub external_count: usize,
    pub external_pdgs: Vec<i32>,
    pub caller_external_pdgs: Vec<i32>,
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
    /// Each run has its own arena lifetime. Initialize the output vector to
    /// positive zero; execute runs, then their reductions in the given order.
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
    external_pdgs: Vec<i32>,
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
                        "unknown recurrence export colour {colour:?}"
                    )));
                }
            }
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
        let alpha_s = export_alpha_s(
            &parameters,
            plan,
            self.runtime.normalization_qcd_coupling_power,
        );
        let runtime_physics = self.runtime.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact("recurrence export requires reduction metadata")
        })?;
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
                colour_id: colour.id().to_owned(),
                external_count: self.runtime.external_count,
                external_pdgs: context.external_pdgs.clone(),
                caller_external_pdgs: physics
                    .external_particles
                    .iter()
                    .map(|particle| particle.pdg)
                    .collect(),
                crossing: crossing.clone(),
                momentum_forms: Vec::new(),
                currents: export_currents(plan),
                current_component_count: plan.current_arena_components()
                    + lane.scheduler.export_fanout().scratch_component_count(),
                amplitude_count: plan.amplitude_destination_count(),
                scales: scales.clone(),
                factors: factors.clone(),
                helicity_outputs: physics
                    .helicities
                    .iter()
                    .map(|helicity| RecurrenceExportHelicity {
                        id: helicity.id.clone(),
                        helicities: helicity.values.clone(),
                    })
                    .collect(),
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
                        let body = export_direct_body(
                            plan,
                            selector.selected_row_groups(),
                            &clears,
                            &executors,
                            &sources,
                            Some(lane.scheduler.export_fanout()),
                        )?;
                        let mut reduction = Vec::new();
                        let (phase_re, phase_im) = selector.phase();
                        let amplitude_scale = ExportComplex {
                            re: phase_re * f64::from(selector.multiplicity()),
                            im: phase_im * f64::from(selector.multiplicity()),
                        };
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
                        program.runs.push(RecurrenceExportRun { body, reduction });
                    }
                }
                RecurrenceNativeSelectors::AllFlowUnion {
                    helicity_selectors_by_physics,
                    destination_by_public_flow,
                } => {
                    program.momentum_forms = export_momentum_forms(plan, None)?;
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
                            let clears = export_stage_clears(plan, None)?;
                            let body = export_direct_body(
                                plan,
                                plan.row_groups(),
                                &clears,
                                &executors,
                                &sources,
                                Some(lane.scheduler.export_fanout()),
                            )?;
                            let weight = helicity.coefficient
                                * colour.coefficient()
                                * self.runtime.normalization_factor;
                            program.runs.push(RecurrenceExportRun {
                                body,
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
                    let application = payloads
                        .source(&binding.source_application_path)?
                        .read()?
                        .into_owned();
                    kernels.insert(
                        sha256.clone(),
                        RecurrenceExportKernel {
                            sha256,
                            application,
                        },
                    );
                }
            }
        }
        Ok(RecurrenceExport {
            schema_version: 1,
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
                ..
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
            let external_pdgs = ready_execution.external_pdg_order.clone();
            return Ok(ExportContext {
                artifact,
                executors: loaded.process_executors,
                domains,
                payload_root,
                external_pdgs,
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
        let external_pdgs = selection.process.external_pdgs.clone();
        Ok(ExportContext {
            artifact,
            executors: loaded.process_executors,
            domains,
            payload_root,
            external_pdgs,
        })
    }
}

fn split_values(re: &[f64], im: &[f64]) -> Vec<ExportComplex> {
    re.iter()
        .zip(im)
        .map(|(&re, &im)| ExportComplex { re, im })
        .collect()
}

fn refine_current_metadata(program: &mut RecurrenceExportProgram) {
    let mut source_families = BTreeMap::new();
    let mut clear_ends = BTreeMap::<u32, u32>::new();
    for instruction in program.runs.iter().flat_map(|run| &run.body) {
        match instruction {
            DirectExportInstruction::Source { source } => {
                source_families
                    .entry(source.current)
                    .or_insert_with(|| source.family.clone());
            }
            DirectExportInstruction::Clear {
                current,
                component_count,
            } => {
                let end = current + component_count;
                clear_ends
                    .entry(*current)
                    .and_modify(|previous| *previous = (*previous).max(end))
                    .or_insert(end);
            }
            _ => {}
        }
    }
    // Merge overlapping physical spans once. A predecessor lookup then finds
    // the only possible covering clear, including aliased semantic currents.
    let mut cleared = BTreeMap::<u32, u32>::new();
    for (base, end) in clear_ends {
        if let Some((_, previous_end)) = cleared.last_key_value() {
            if base <= *previous_end {
                let previous_end = cleared.last_entry().expect("nonempty clear map").into_mut();
                *previous_end = (*previous_end).max(end);
                continue;
            }
        }
        cleared.insert(base, end);
    }
    for current in &mut program.currents {
        if current.initial_state == "source" {
            current.physical_type = source_families.get(&current.current).cloned();
        } else if cleared
            .range(..=current.current)
            .next_back()
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
        if let Some(sector) = sector {
            if !plan.selector_domain_contains(current.selector_domain_id, sector)? {
                continue;
            }
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

fn export_alpha_s(
    parameters: &[RecurrenceExportParameter],
    plan: &DirectRecurrencePlan,
    normalization_power: usize,
) -> Option<RecurrenceExportAlphaS> {
    let parameter = parameters.iter().find(|parameter| {
        matches!(
            parameter.name.as_str(),
            "aS" | "alpha_s" | "alphas" | "normalization.alpha_s_me_check"
        )
    })?;
    let orders = plan.amplitude_orders();
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
    let squared_amplitude_power = if qcd.is_some() {
        homogeneous
    } else if orders.names().is_empty() {
        u32::try_from(normalization_power).ok()
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
        let exported = export_alpha_s(&parameters, &plan, 3).unwrap();
        assert_eq!(exported.squared_amplitude_power, Some(3));
        assert_eq!(exported.value, parameters[0].value);

        let without_orders = crate::recurrence::valid_direct_plan_fixture();
        assert_eq!(
            export_alpha_s(&parameters, &without_orders, 3)
                .unwrap()
                .squared_amplitude_power,
            Some(3),
        );
    }
}
