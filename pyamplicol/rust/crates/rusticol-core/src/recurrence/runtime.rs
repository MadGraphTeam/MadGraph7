// SPDX-License-Identifier: 0BSD

//! Numeric execution of compact recurrence schedules.
//!
//! Recurrence rows remain compact. Within one dependency stage, calls sharing
//! a prepared kernel are packed as `(edge, point)` lanes, all contributions are
//! accumulated, and each current is finalized exactly once.

use std::collections::BTreeMap;
use std::ops::Range;

use num_complex::Complex;

use super::construct::TemplateCatalog;
use super::template::{MISSING_U32, ValidatedRecurrenceTemplateInput};
use super::{
    CanonicalMomentumLinearForm, CurrentSourceBinding, ExactComplexRational, RecurrenceProgram,
    RecurrenceStrategy,
};
use crate::{
    EagerComplex64, EagerKernelBackend, EagerKernelCall, EagerKernelInput, EagerKernelRole,
    EagerKernelSpec, RusticolError, RusticolResult,
};

const OUTPUT_FACTOR_NONE: u8 = 0;
const OUTPUT_FACTOR_COUPLING_REAL: u8 = 1;
const OUTPUT_FACTOR_COUPLING_IMAG: u8 = 2;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ComponentRange {
    start: usize,
    len: usize,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct InvocationKey {
    kernel_id: u32,
    transition_id: u32,
    quantum_flow_id: u32,
    parent_current_ids: Box<[u32]>,
}

#[derive(Clone, Debug)]
struct KernelSite {
    kernel_id: u32,
    parent_current_ids: Box<[u32]>,
    parent_momentum_ids: Box<[u32]>,
    coupling: EagerComplex64,
}

trait HasKernelSite {
    fn kernel_site(&self) -> &KernelSite;
}

#[derive(Clone, Copy, Debug)]
struct Attachment {
    result_current_id: u32,
    factor: EagerComplex64,
}

#[derive(Clone, Debug)]
struct Invocation {
    site: KernelSite,
    output_factor_source: u8,
    attachments: Range<usize>,
}

impl HasKernelSite for Invocation {
    fn kernel_site(&self) -> &KernelSite {
        &self.site
    }
}

#[derive(Clone, Debug)]
struct KernelPacket {
    kernel_spec_index: usize,
    calls: Range<usize>,
}

#[derive(Clone, Debug)]
struct FinalizationCall {
    site: KernelSite,
    current_id: u32,
    factor: EagerComplex64,
}

impl HasKernelSite for FinalizationCall {
    fn kernel_site(&self) -> &KernelSite {
        &self.site
    }
}

#[derive(Clone, Copy, Debug)]
struct IdentityFinalization {
    current_id: u32,
    factor: EagerComplex64,
}

#[derive(Clone, Debug, Default)]
struct StagePlan {
    invocations: Vec<Invocation>,
    attachments: Vec<Attachment>,
    invocation_packets: Vec<KernelPacket>,
    finalizations: Vec<FinalizationCall>,
    finalization_packets: Vec<KernelPacket>,
    identity_finalizations: Vec<IdentityFinalization>,
}

#[derive(Clone, Debug)]
struct PreparedClosureCall {
    site: KernelSite,
    target_sector_id: u32,
    factor: EagerComplex64,
    output_factor_source: u8,
}

impl HasKernelSite for PreparedClosureCall {
    fn kernel_site(&self) -> &KernelSite {
        &self.site
    }
}

#[derive(Clone, Debug)]
struct DirectClosureCall {
    parent_current_ids: [u32; 2],
    target_sector_id: u32,
    coefficients: Box<[EagerComplex64]>,
    factor: EagerComplex64,
}

#[derive(Clone, Copy, Debug)]
struct SourceCopy {
    current_id: u32,
    source_slot: u32,
    source_template_id: u32,
    source_component_start: usize,
    factor: EagerComplex64,
}

/// Stable source-buffer ownership needed by selector and source-fill planning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecurrenceSourceLayout {
    pub current_id: u32,
    pub source_slot: u32,
    pub source_template_id: u32,
    pub component_start: usize,
    pub component_count: usize,
}

/// Immutable microprogram built once after composite authentication.
#[derive(Clone, Debug)]
pub struct RecurrenceExecutionPlan {
    program: RecurrenceProgram,
    kernels: Box<[EagerKernelSpec]>,
    current_components: Box<[ComponentRange]>,
    current_momenta: Box<[CanonicalMomentumLinearForm]>,
    sources: Box<[SourceCopy]>,
    stages: Box<[StagePlan]>,
    prepared_closures: Box<[PreparedClosureCall]>,
    prepared_closure_packets: Box<[KernelPacket]>,
    direct_closures: Box<[DirectClosureCall]>,
    source_component_count: usize,
    current_component_count: usize,
    source_slot_count: usize,
    parameter_count: usize,
    sector_count: usize,
    maximum_packet_input_width: usize,
    maximum_packet_output_width: usize,
}

impl RecurrenceExecutionPlan {
    pub fn new(
        program: RecurrenceProgram,
        template: &ValidatedRecurrenceTemplateInput,
        mut kernels: Vec<EagerKernelSpec>,
    ) -> RusticolResult<Self> {
        if program.strategy() != RecurrenceStrategy::TopologyReplay {
            return Err(RusticolError::compatibility(
                "the first recurrence runtime slice supports topology-replay only",
            ));
        }
        program.validate()?;
        kernels.sort_by_key(|kernel| kernel.kernel_id);
        if kernels
            .windows(2)
            .any(|pair| pair[0].kernel_id == pair[1].kernel_id)
        {
            return Err(invalid("recurrence prepared kernel IDs are not unique"));
        }
        let kernel_indices = kernels
            .iter()
            .enumerate()
            .map(|(index, kernel)| (kernel.kernel_id, index))
            .collect::<BTreeMap<_, _>>();
        let input = template.input();
        let catalog = TemplateCatalog::new(input)?;
        let parameter_count = template.summary().parameter_count as usize;

        let mut current_components = Vec::with_capacity(program.currents().len());
        let mut current_component_count = 0usize;
        let mut source_slot_count = 0usize;
        for current in program.currents() {
            let state = input
                .current_states
                .get(current.key().current_state_template_id() as usize)
                .ok_or_else(|| invalid("recurrence current state is absent"))?;
            let len = state.dimension as usize;
            current_components.push(ComponentRange {
                start: current_component_count,
                len,
            });
            current_component_count = current_component_count
                .checked_add(len)
                .ok_or_else(|| invalid("recurrence current components overflow"))?;
            for term in current.key().momentum().terms() {
                source_slot_count = source_slot_count.max(term.source_slot as usize + 1);
            }
        }

        let mut source_component_count = 0usize;
        let mut sources = Vec::new();
        for current in program
            .currents()
            .iter()
            .filter(|current| current.is_source())
        {
            let len = current_components[current.id() as usize].len;
            let source_slot = current.key().support_source_slots()[0];
            let CurrentSourceBinding::FixedTemplate(source_template_id) =
                current.key().source_binding()
            else {
                return Err(invalid(
                    "topology-replay source lacks a fixed source template",
                ));
            };
            sources.push(SourceCopy {
                current_id: current.id(),
                source_slot,
                source_template_id,
                source_component_start: source_component_count,
                factor: exact_complex(
                    current
                        .source_exact_factor()
                        .ok_or_else(|| invalid("topology-replay source factor is absent"))?,
                ),
            });
            source_component_count = source_component_count
                .checked_add(len)
                .ok_or_else(|| invalid("recurrence source components overflow"))?;
        }

        let stage_count = program
            .currents()
            .iter()
            .map(|current| current.key().support_source_slots().len())
            .max()
            .unwrap_or(0);
        let mut stage_groups =
            vec![BTreeMap::<InvocationKey, Vec<Attachment>>::new(); stage_count + 1];
        for contribution in program.contributions() {
            let transition = *input
                .transitions
                .get(contribution.key().transition_template_id() as usize)
                .ok_or_else(|| invalid("recurrence transition is absent"))?;
            if contribution.key().output_projection_id() != transition.output_projection_string_id {
                return Err(invalid(
                    "recurrence contribution output projection does not match its transition",
                ));
            }
            let evaluator = input
                .evaluator_bindings
                .get(transition.evaluator_binding_id as usize)
                .ok_or_else(|| invalid("recurrence transition evaluator is absent"))?;
            if evaluator.prepared_kernel_id == MISSING_U32 {
                return Err(invalid("recurrence transition lacks a prepared kernel"));
            }
            let kernel = required_kernel(
                &kernels,
                &kernel_indices,
                evaluator.prepared_kernel_id,
                EagerKernelRole::Vertex,
            )?;
            validate_kernel_site(
                kernel,
                contribution.parent_current_ids(),
                &current_components,
                parameter_count,
                current_components[contribution.result_current_id() as usize].len,
            )?;
            validate_output_factor_source(transition.output_factor_source)?;
            let stage = program.currents()[contribution.result_current_id() as usize]
                .key()
                .support_source_slots()
                .len();
            if contribution.parent_current_ids().iter().any(|parent_id| {
                program.currents()[*parent_id as usize]
                    .key()
                    .support_source_slots()
                    .len()
                    >= stage
            }) {
                return Err(invalid(
                    "recurrence contribution parent is not in an earlier dependency stage",
                ));
            }
            stage_groups[stage]
                .entry(InvocationKey {
                    kernel_id: evaluator.prepared_kernel_id,
                    transition_id: transition.id,
                    quantum_flow_id: contribution.key().quantum_flow_witness_id(),
                    parent_current_ids: contribution.parent_current_ids().into(),
                })
                .or_default()
                .push(Attachment {
                    result_current_id: contribution.result_current_id(),
                    factor: exact_complex(contribution.exact_factor()),
                });
        }

        let mut stages = vec![StagePlan::default(); stage_count + 1];
        for (stage_index, groups) in stage_groups.into_iter().enumerate() {
            let stage = &mut stages[stage_index];
            for (key, attachments) in groups {
                let transition = input.transitions[key.transition_id as usize];
                let quantum = input.quantum_flows[key.quantum_flow_id as usize];
                let start = stage.attachments.len();
                stage.attachments.extend(attachments);
                let stop = stage.attachments.len();
                stage.invocations.push(Invocation {
                    site: KernelSite {
                        kernel_id: key.kernel_id,
                        parent_momentum_ids: key.parent_current_ids.clone(),
                        parent_current_ids: key.parent_current_ids,
                        coupling: exact_complex(catalog.factor(
                            quantum.exact_coupling_factor_id,
                            "recurrence transition coupling",
                        )?),
                    },
                    output_factor_source: transition.output_factor_source,
                    attachments: start..stop,
                });
            }
            stage.invocation_packets = packetize(&stage.invocations, &kernel_indices)?;
        }

        for finalization in program.finalizations() {
            let current = &program.currents()[finalization.current_id() as usize];
            let stage = current.key().support_source_slots().len();
            let factor = exact_complex(finalization.exact_factor());
            let Some(propagator_id) = finalization.propagator_template_id() else {
                stages[stage]
                    .identity_finalizations
                    .push(IdentityFinalization {
                        current_id: current.id(),
                        factor,
                    });
                continue;
            };
            let propagator = input
                .propagators
                .get(propagator_id as usize)
                .ok_or_else(|| invalid("recurrence propagator is absent"))?;
            let evaluator = input
                .evaluator_bindings
                .get(propagator.evaluator_binding_id as usize)
                .ok_or_else(|| invalid("recurrence propagator evaluator is absent"))?;
            if evaluator.prepared_kernel_id == MISSING_U32 {
                return Err(invalid(
                    "active recurrence propagator lacks a prepared kernel",
                ));
            }
            let kernel = required_kernel(
                &kernels,
                &kernel_indices,
                evaluator.prepared_kernel_id,
                EagerKernelRole::Finalization,
            )?;
            validate_kernel_site(
                kernel,
                &[current.id()],
                &current_components,
                parameter_count,
                current_components[current.id() as usize].len,
            )?;
            stages[stage].finalizations.push(FinalizationCall {
                site: KernelSite {
                    kernel_id: evaluator.prepared_kernel_id,
                    parent_current_ids: vec![current.id()].into_boxed_slice(),
                    parent_momentum_ids: vec![current.id()].into_boxed_slice(),
                    coupling: EagerComplex64::new(1.0, 0.0),
                },
                current_id: current.id(),
                factor,
            });
        }
        for stage in &mut stages {
            stage.finalizations.sort_by_key(|call| call.site.kernel_id);
            stage.finalization_packets = packetize(&stage.finalizations, &kernel_indices)?;
        }

        let mut prepared_closures = Vec::new();
        let mut direct_closures = Vec::new();
        for term in program.closure_terms() {
            let closure = input
                .closures
                .get(term.closure_template_id() as usize)
                .ok_or_else(|| invalid("recurrence closure is absent"))?;
            let evaluator = input
                .evaluator_bindings
                .get(closure.evaluator_binding_id as usize)
                .ok_or_else(|| invalid("recurrence closure evaluator is absent"))?;
            let coupling = if let Some(quantum_id) = term.quantum_flow_template_id() {
                exact_complex(catalog.factor(
                    input.quantum_flows[quantum_id as usize].exact_coupling_factor_id,
                    "recurrence closure coupling",
                )?)
            } else {
                EagerComplex64::new(1.0, 0.0)
            };
            if evaluator.prepared_kernel_id != MISSING_U32 {
                let kernel = required_kernel(
                    &kernels,
                    &kernel_indices,
                    evaluator.prepared_kernel_id,
                    EagerKernelRole::Closure,
                )?;
                validate_kernel_site(
                    kernel,
                    term.parent_current_ids(),
                    &current_components,
                    parameter_count,
                    1,
                )?;
                validate_output_factor_source(closure.output_factor_source)?;
                prepared_closures.push(PreparedClosureCall {
                    site: KernelSite {
                        kernel_id: evaluator.prepared_kernel_id,
                        parent_current_ids: term.parent_current_ids().into(),
                        parent_momentum_ids: term.parent_current_ids().into(),
                        coupling,
                    },
                    target_sector_id: term.target_sector_id(),
                    factor: exact_complex(term.exact_factor()),
                    output_factor_source: closure.output_factor_source,
                });
            } else {
                let parent_current_ids: [u32; 2] = term
                    .parent_current_ids()
                    .try_into()
                    .map_err(|_| invalid("direct recurrence closure requires two parents"))?;
                let coefficient_ids = catalog.u32_sequence(
                    closure.component_coefficient_sequence_id,
                    "direct recurrence closure coefficients",
                )?;
                let coefficients = coefficient_ids
                    .iter()
                    .map(|id| {
                        catalog
                            .factor(*id, "direct recurrence closure coefficient")
                            .map(exact_complex)
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                let left = current_components[parent_current_ids[0] as usize].len;
                let right = current_components[parent_current_ids[1] as usize].len;
                if coefficients.len() != left || left != right {
                    return Err(invalid(
                        "direct recurrence closure coefficients do not match parent dimensions",
                    ));
                }
                direct_closures.push(DirectClosureCall {
                    parent_current_ids,
                    target_sector_id: term.target_sector_id(),
                    coefficients: coefficients.into_boxed_slice(),
                    factor: exact_complex(term.exact_factor()),
                });
            }
        }
        prepared_closures.sort_by_key(|call| call.site.kernel_id);
        let prepared_closure_packets = packetize(&prepared_closures, &kernel_indices)?;

        let mut maximum_packet_input_width = 0usize;
        let mut maximum_packet_output_width = 0usize;
        for stage in &stages {
            update_packet_widths(
                &stage.invocation_packets,
                &kernels,
                &mut maximum_packet_input_width,
                &mut maximum_packet_output_width,
            )?;
            update_packet_widths(
                &stage.finalization_packets,
                &kernels,
                &mut maximum_packet_input_width,
                &mut maximum_packet_output_width,
            )?;
        }
        update_packet_widths(
            &prepared_closure_packets,
            &kernels,
            &mut maximum_packet_input_width,
            &mut maximum_packet_output_width,
        )?;

        Ok(Self {
            current_momenta: program
                .currents()
                .iter()
                .map(|current| current.key().momentum().clone())
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            sector_count: program.target_sector_closure_ranges().len(),
            parameter_count,
            program,
            kernels: kernels.into_boxed_slice(),
            current_components: current_components.into_boxed_slice(),
            sources: sources.into_boxed_slice(),
            stages: stages.into_boxed_slice(),
            prepared_closures: prepared_closures.into_boxed_slice(),
            prepared_closure_packets: prepared_closure_packets.into_boxed_slice(),
            direct_closures: direct_closures.into_boxed_slice(),
            source_component_count,
            current_component_count,
            source_slot_count,
            maximum_packet_input_width,
            maximum_packet_output_width,
        })
    }

    pub const fn program(&self) -> &RecurrenceProgram {
        &self.program
    }

    pub const fn source_component_count(&self) -> usize {
        self.source_component_count
    }

    pub const fn source_slot_count(&self) -> usize {
        self.source_slot_count
    }

    pub const fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    pub const fn sector_count(&self) -> usize {
        self.sector_count
    }

    pub fn source_layout(&self) -> Vec<RecurrenceSourceLayout> {
        self.sources
            .iter()
            .map(|source| RecurrenceSourceLayout {
                current_id: source.current_id,
                source_slot: source.source_slot,
                source_template_id: source.source_template_id,
                component_start: source.source_component_start,
                component_count: self.current_components[source.current_id as usize].len,
            })
            .collect()
    }
}

/// Fixed-capacity workspace reused by every warmed evaluation.
pub struct RecurrenceExecutionRuntime {
    plan: RecurrenceExecutionPlan,
    tile_capacity: usize,
    currents: Vec<EagerComplex64>,
    current_momenta: Vec<f64>,
    kernel_inputs: Vec<EagerComplex64>,
    kernel_outputs: Vec<EagerComplex64>,
    sector_amplitudes: Vec<EagerComplex64>,
}

impl RecurrenceExecutionRuntime {
    pub fn new(plan: RecurrenceExecutionPlan, tile_capacity: usize) -> RusticolResult<Self> {
        if tile_capacity == 0 {
            return Err(invalid("recurrence tile size must be positive"));
        }
        Ok(Self {
            currents: zero_complex(
                plan.current_component_count
                    .checked_mul(tile_capacity)
                    .ok_or_else(|| invalid("recurrence current workspace overflows"))?,
            ),
            current_momenta: vec![
                0.0;
                plan.current_momenta
                    .len()
                    .checked_mul(4)
                    .and_then(|value| value.checked_mul(tile_capacity))
                    .ok_or_else(|| invalid(
                        "recurrence momentum workspace overflows"
                    ))?
            ],
            kernel_inputs: zero_complex(
                plan.maximum_packet_input_width
                    .checked_mul(tile_capacity)
                    .ok_or_else(|| invalid("recurrence input workspace overflows"))?,
            ),
            kernel_outputs: zero_complex(
                plan.maximum_packet_output_width
                    .checked_mul(tile_capacity)
                    .ok_or_else(|| invalid("recurrence output workspace overflows"))?,
            ),
            sector_amplitudes: zero_complex(
                plan.sector_count
                    .checked_mul(tile_capacity)
                    .ok_or_else(|| invalid("recurrence sector workspace overflows"))?,
            ),
            plan,
            tile_capacity,
        })
    }

    pub const fn plan(&self) -> &RecurrenceExecutionPlan {
        &self.plan
    }

    /// Evaluate one resolved helicity in sector-major, point-contiguous order.
    ///
    /// `source_values` contains every materialized source-state slot, but only
    /// the slots selected for this helicity may be nonzero. Selector planning
    /// owns that sparse fill and is deliberately outside this numeric core.
    pub fn evaluate_one_helicity_amplitudes_into<B: EagerKernelBackend>(
        &mut self,
        backend: &mut B,
        point_count: usize,
        source_values: &[EagerComplex64],
        external_momenta: &[f64],
        model_parameters: &[EagerComplex64],
        output: &mut [EagerComplex64],
    ) -> RusticolResult<()> {
        if point_count == 0 {
            return Err(invalid("recurrence evaluation requires at least one point"));
        }
        check_matrix(
            source_values.len(),
            self.plan.source_component_count,
            point_count,
            "source values",
        )?;
        check_matrix(
            external_momenta.len(),
            self.plan.source_slot_count * 4,
            point_count,
            "external momenta",
        )?;
        if model_parameters.len() < self.plan.parameter_count {
            return Err(invalid("recurrence model-parameter input is too short"));
        }
        check_matrix(
            output.len(),
            self.plan.sector_count,
            point_count,
            "sector output",
        )?;

        for tile_start in (0..point_count).step_by(self.tile_capacity) {
            let tile_points = usize::min(self.tile_capacity, point_count - tile_start);
            self.currents.fill(EagerComplex64::new(0.0, 0.0));
            self.sector_amplitudes.fill(EagerComplex64::new(0.0, 0.0));
            fill_sources(
                &self.plan,
                self.tile_capacity,
                &mut self.currents,
                source_values,
                point_count,
                tile_start,
                tile_points,
            );
            fill_momenta(
                &self.plan,
                self.tile_capacity,
                &mut self.current_momenta,
                external_momenta,
                point_count,
                tile_start,
                tile_points,
            );
            for stage_index in 2..self.plan.stages.len() {
                execute_stage(
                    &self.plan,
                    self.tile_capacity,
                    &mut self.currents,
                    &self.current_momenta,
                    &mut self.kernel_inputs,
                    &mut self.kernel_outputs,
                    backend,
                    stage_index,
                    tile_points,
                    model_parameters,
                )?;
            }
            execute_closures(
                &self.plan,
                self.tile_capacity,
                &self.currents,
                &self.current_momenta,
                &mut self.kernel_inputs,
                &mut self.kernel_outputs,
                &mut self.sector_amplitudes,
                backend,
                tile_points,
                model_parameters,
            )?;
            for sector in 0..self.plan.sector_count {
                for point in 0..tile_points {
                    output[sector * point_count + tile_start + point] =
                        self.sector_amplitudes[sector * self.tile_capacity + point];
                }
            }
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_stage<B: EagerKernelBackend>(
    plan: &RecurrenceExecutionPlan,
    tile_capacity: usize,
    currents: &mut [EagerComplex64],
    momenta: &[f64],
    inputs: &mut [EagerComplex64],
    outputs: &mut [EagerComplex64],
    backend: &mut B,
    stage_index: usize,
    tile_points: usize,
    model_parameters: &[EagerComplex64],
) -> RusticolResult<()> {
    let stage = &plan.stages[stage_index];
    for packet in &stage.invocation_packets {
        let spec = &plan.kernels[packet.kernel_spec_index];
        let calls = &stage.invocations[packet.calls.clone()];
        call_kernel(
            plan,
            tile_capacity,
            currents,
            momenta,
            inputs,
            outputs,
            backend,
            spec,
            calls,
            tile_points,
            model_parameters,
        )?;
        let output_count = spec.output_component_count as usize;
        for (call_index, call) in calls.iter().enumerate() {
            let output_scale = output_factor_scale(call.output_factor_source, call.site.coupling);
            for attachment in &stage.attachments[call.attachments.clone()] {
                let target = plan.current_components[attachment.result_current_id as usize];
                for point in 0..tile_points {
                    let lane = call_index * tile_points + point;
                    for component in 0..output_count {
                        currents[(target.start + component) * tile_capacity + point] += attachment
                            .factor
                            * output_scale
                            * outputs[lane * output_count + component];
                    }
                }
            }
        }
    }

    for row in &stage.identity_finalizations {
        let target = plan.current_components[row.current_id as usize];
        for component in 0..target.len {
            for point in 0..tile_points {
                currents[(target.start + component) * tile_capacity + point] *= row.factor;
            }
        }
    }
    for packet in &stage.finalization_packets {
        let spec = &plan.kernels[packet.kernel_spec_index];
        let calls = &stage.finalizations[packet.calls.clone()];
        call_kernel(
            plan,
            tile_capacity,
            currents,
            momenta,
            inputs,
            outputs,
            backend,
            spec,
            calls,
            tile_points,
            model_parameters,
        )?;
        let output_count = spec.output_component_count as usize;
        for (call_index, call) in calls.iter().enumerate() {
            let target = plan.current_components[call.current_id as usize];
            for point in 0..tile_points {
                let lane = call_index * tile_points + point;
                for component in 0..output_count {
                    currents[(target.start + component) * tile_capacity + point] =
                        call.factor * outputs[lane * output_count + component];
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_closures<B: EagerKernelBackend>(
    plan: &RecurrenceExecutionPlan,
    tile_capacity: usize,
    currents: &[EagerComplex64],
    momenta: &[f64],
    inputs: &mut [EagerComplex64],
    outputs: &mut [EagerComplex64],
    sectors: &mut [EagerComplex64],
    backend: &mut B,
    tile_points: usize,
    model_parameters: &[EagerComplex64],
) -> RusticolResult<()> {
    for packet in plan.prepared_closure_packets.iter() {
        let spec = &plan.kernels[packet.kernel_spec_index];
        let calls = &plan.prepared_closures[packet.calls.clone()];
        call_kernel(
            plan,
            tile_capacity,
            currents,
            momenta,
            inputs,
            outputs,
            backend,
            spec,
            calls,
            tile_points,
            model_parameters,
        )?;
        for (call_index, call) in calls.iter().enumerate() {
            let scale =
                call.factor * output_factor_scale(call.output_factor_source, call.site.coupling);
            for point in 0..tile_points {
                sectors[call.target_sector_id as usize * tile_capacity + point] +=
                    scale * outputs[call_index * tile_points + point];
            }
        }
    }
    for call in plan.direct_closures.iter() {
        let left = plan.current_components[call.parent_current_ids[0] as usize];
        let right = plan.current_components[call.parent_current_ids[1] as usize];
        for point in 0..tile_points {
            let mut value = EagerComplex64::new(0.0, 0.0);
            for component in 0..call.coefficients.len() {
                value += call.coefficients[component]
                    * currents[(left.start + component) * tile_capacity + point]
                    * currents[(right.start + component) * tile_capacity + point];
            }
            sectors[call.target_sector_id as usize * tile_capacity + point] += call.factor * value;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn call_kernel<B: EagerKernelBackend, T: HasKernelSite>(
    plan: &RecurrenceExecutionPlan,
    tile_capacity: usize,
    currents: &[EagerComplex64],
    momenta: &[f64],
    inputs: &mut [EagerComplex64],
    outputs: &mut [EagerComplex64],
    backend: &mut B,
    spec: &EagerKernelSpec,
    calls: &[T],
    tile_points: usize,
    model_parameters: &[EagerComplex64],
) -> RusticolResult<()> {
    let lane_count = calls
        .len()
        .checked_mul(tile_points)
        .ok_or_else(|| invalid("recurrence lane count overflows"))?;
    let input_count = spec.inputs.len();
    let output_count = spec.output_component_count as usize;
    let input_len = lane_count
        .checked_mul(input_count)
        .ok_or_else(|| invalid("recurrence input packet overflows"))?;
    let output_len = lane_count
        .checked_mul(output_count)
        .ok_or_else(|| invalid("recurrence output packet overflows"))?;
    if input_len > inputs.len() || output_len > outputs.len() {
        return Err(RusticolError::internal(
            "recurrence packet exceeds its preallocated workspace",
        ));
    }
    for (call_index, call) in calls.iter().enumerate() {
        let site = call.kernel_site();
        for point in 0..tile_points {
            let lane = call_index * tile_points + point;
            for (input_index, descriptor) in spec.inputs.iter().enumerate() {
                inputs[lane * input_count + input_index] = match *descriptor {
                    EagerKernelInput::FirstCurrentComponent(component) => current_value(
                        plan,
                        currents,
                        tile_capacity,
                        site.parent_current_ids[0],
                        component as usize,
                        point,
                    ),
                    EagerKernelInput::SecondCurrentComponent(component) => current_value(
                        plan,
                        currents,
                        tile_capacity,
                        site.parent_current_ids[1],
                        component as usize,
                        point,
                    ),
                    EagerKernelInput::FirstMomentumComponent(component) => momentum_value(
                        momenta,
                        tile_capacity,
                        site.parent_momentum_ids[0],
                        component as usize,
                        point,
                    ),
                    EagerKernelInput::SecondMomentumComponent(component) => momentum_value(
                        momenta,
                        tile_capacity,
                        site.parent_momentum_ids[1],
                        component as usize,
                        point,
                    ),
                    EagerKernelInput::CouplingReal => EagerComplex64::new(site.coupling.re, 0.0),
                    EagerKernelInput::CouplingImag => EagerComplex64::new(site.coupling.im, 0.0),
                    EagerKernelInput::ModelParameter(index) => *model_parameters
                        .get(index as usize)
                        .ok_or_else(|| invalid("recurrence model parameter is absent"))?,
                };
            }
        }
    }
    backend.evaluate_batch(EagerKernelCall {
        kernel_id: spec.kernel_id,
        independent_block_size: 1,
        lane_count,
        input_component_count: input_count,
        output_component_count: output_count,
        inputs: &inputs[..input_len],
        outputs: &mut outputs[..output_len],
    })
}

fn fill_sources(
    plan: &RecurrenceExecutionPlan,
    tile_capacity: usize,
    currents: &mut [EagerComplex64],
    sources: &[EagerComplex64],
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
) {
    for source in plan.sources.iter() {
        let target = plan.current_components[source.current_id as usize];
        for component in 0..target.len {
            for point in 0..tile_points {
                currents[(target.start + component) * tile_capacity + point] = source.factor
                    * sources[(source.source_component_start + component) * point_count
                        + tile_start
                        + point];
            }
        }
    }
}

fn fill_momenta(
    plan: &RecurrenceExecutionPlan,
    tile_capacity: usize,
    current_momenta: &mut [f64],
    external_momenta: &[f64],
    point_count: usize,
    tile_start: usize,
    tile_points: usize,
) {
    for (momentum_id, form) in plan.current_momenta.iter().enumerate() {
        for component in 0..4 {
            for point in 0..tile_points {
                let mut value = 0.0;
                for term in form.terms() {
                    value += f64::from(term.coefficient)
                        * external_momenta[(term.source_slot as usize * 4 + component)
                            * point_count
                            + tile_start
                            + point];
                }
                current_momenta[(momentum_id * 4 + component) * tile_capacity + point] = value;
            }
        }
    }
}

fn current_value(
    plan: &RecurrenceExecutionPlan,
    values: &[EagerComplex64],
    tile_capacity: usize,
    current_id: u32,
    component: usize,
    point: usize,
) -> EagerComplex64 {
    let range = plan.current_components[current_id as usize];
    values[(range.start + component) * tile_capacity + point]
}

fn momentum_value(
    values: &[f64],
    tile_capacity: usize,
    momentum_id: u32,
    component: usize,
    point: usize,
) -> EagerComplex64 {
    EagerComplex64::new(
        values[(momentum_id as usize * 4 + component) * tile_capacity + point],
        0.0,
    )
}

fn packetize<T: HasKernelSite>(
    calls: &[T],
    kernel_indices: &BTreeMap<u32, usize>,
) -> RusticolResult<Vec<KernelPacket>> {
    let mut packets = Vec::new();
    let mut start = 0usize;
    while start < calls.len() {
        let kernel_id = calls[start].kernel_site().kernel_id;
        let mut stop = start + 1;
        while stop < calls.len() && calls[stop].kernel_site().kernel_id == kernel_id {
            stop += 1;
        }
        packets.push(KernelPacket {
            kernel_spec_index: *kernel_indices
                .get(&kernel_id)
                .ok_or_else(|| invalid(format!("recurrence kernel {kernel_id} is absent")))?,
            calls: start..stop,
        });
        start = stop;
    }
    Ok(packets)
}

fn required_kernel<'a>(
    kernels: &'a [EagerKernelSpec],
    indices: &BTreeMap<u32, usize>,
    kernel_id: u32,
    role: EagerKernelRole,
) -> RusticolResult<&'a EagerKernelSpec> {
    let kernel = indices
        .get(&kernel_id)
        .and_then(|index| kernels.get(*index))
        .ok_or_else(|| invalid(format!("recurrence kernel {kernel_id} is absent")))?;
    if kernel.role != role {
        return Err(invalid(format!(
            "recurrence kernel {kernel_id} has role {:?}, expected {role:?}",
            kernel.role
        )));
    }
    Ok(kernel)
}

fn validate_kernel_site(
    kernel: &EagerKernelSpec,
    parents: &[u32],
    ranges: &[ComponentRange],
    parameter_count: usize,
    output_count: usize,
) -> RusticolResult<()> {
    if kernel.output_component_count as usize != output_count {
        return Err(invalid(format!(
            "recurrence kernel {} outputs {}, expected {output_count}",
            kernel.kernel_id, kernel.output_component_count
        )));
    }
    for descriptor in &kernel.inputs {
        match *descriptor {
            EagerKernelInput::FirstCurrentComponent(component) => {
                validate_component(parents, 0, component, ranges)?
            }
            EagerKernelInput::SecondCurrentComponent(component) => {
                validate_component(parents, 1, component, ranges)?
            }
            EagerKernelInput::FirstMomentumComponent(component) => {
                if parents.is_empty() || component >= 4 {
                    return Err(invalid("recurrence first momentum input is invalid"));
                }
            }
            EagerKernelInput::SecondMomentumComponent(component) => {
                if parents.len() < 2 || component >= 4 {
                    return Err(invalid("recurrence second momentum input is invalid"));
                }
            }
            EagerKernelInput::ModelParameter(index) if index as usize >= parameter_count => {
                return Err(invalid(format!(
                    "recurrence kernel {} references absent model parameter {index}",
                    kernel.kernel_id
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_component(
    parents: &[u32],
    parent_index: usize,
    component: u32,
    ranges: &[ComponentRange],
) -> RusticolResult<()> {
    let parent = *parents
        .get(parent_index)
        .ok_or_else(|| invalid("recurrence kernel parent is absent"))?;
    let range = ranges
        .get(parent as usize)
        .ok_or_else(|| invalid("recurrence parent current is absent"))?;
    if component as usize >= range.len {
        return Err(invalid(format!(
            "recurrence parent {parent} component {component} exceeds dimension {}",
            range.len
        )));
    }
    Ok(())
}

fn update_packet_widths(
    packets: &[KernelPacket],
    kernels: &[EagerKernelSpec],
    maximum_input: &mut usize,
    maximum_output: &mut usize,
) -> RusticolResult<()> {
    for packet in packets {
        let kernel = kernels
            .get(packet.kernel_spec_index)
            .ok_or_else(|| invalid("recurrence packet kernel is absent"))?;
        *maximum_input = (*maximum_input).max(
            packet
                .calls
                .len()
                .checked_mul(kernel.inputs.len())
                .ok_or_else(|| invalid("recurrence packet input width overflows"))?,
        );
        *maximum_output = (*maximum_output).max(
            packet
                .calls
                .len()
                .checked_mul(kernel.output_component_count as usize)
                .ok_or_else(|| invalid("recurrence packet output width overflows"))?,
        );
    }
    Ok(())
}

fn validate_output_factor_source(value: u8) -> RusticolResult<()> {
    match value {
        OUTPUT_FACTOR_NONE | OUTPUT_FACTOR_COUPLING_REAL | OUTPUT_FACTOR_COUPLING_IMAG => Ok(()),
        _ => Err(invalid(format!(
            "unsupported recurrence output-factor source {value}"
        ))),
    }
}

fn output_factor_scale(source: u8, coupling: EagerComplex64) -> f64 {
    match source {
        OUTPUT_FACTOR_NONE => 1.0,
        OUTPUT_FACTOR_COUPLING_REAL => coupling.re,
        OUTPUT_FACTOR_COUPLING_IMAG => coupling.im,
        _ => unreachable!("validated recurrence output-factor source"),
    }
}

fn exact_complex(value: ExactComplexRational) -> Complex<f64> {
    Complex::new(
        value.real().numerator() as f64 / value.real().denominator() as f64,
        value.imag().numerator() as f64 / value.imag().denominator() as f64,
    )
}

fn zero_complex(len: usize) -> Vec<EagerComplex64> {
    vec![EagerComplex64::new(0.0, 0.0); len]
}

fn check_matrix(actual: usize, rows: usize, points: usize, label: &str) -> RusticolResult<()> {
    let expected = rows
        .checked_mul(points)
        .ok_or_else(|| invalid(format!("recurrence {label} length overflows")))?;
    if actual != expected {
        return Err(invalid(format!(
            "recurrence {label} has length {actual}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::{CheckedTableRange, MomentumTerm};

    #[derive(Default)]
    struct AddBackend {
        calls: Vec<usize>,
    }

    impl EagerKernelBackend for AddBackend {
        fn evaluate_batch(&mut self, call: EagerKernelCall<'_>) -> RusticolResult<()> {
            assert_eq!(call.kernel_id, 7);
            assert_eq!(call.independent_block_size, 1);
            assert_eq!(call.input_component_count, 2);
            assert_eq!(call.output_component_count, 1);
            self.calls.push(call.lane_count);
            for lane in 0..call.lane_count {
                call.outputs[lane] = call.inputs[lane * 2] + call.inputs[lane * 2 + 1];
            }
            Ok(())
        }
    }

    fn momentum(source_slot: u32) -> CanonicalMomentumLinearForm {
        CanonicalMomentumLinearForm::new(vec![MomentumTerm {
            source_slot,
            coefficient: 1,
        }])
        .expect("test momentum is canonical")
    }

    fn synthetic_plan() -> RecurrenceExecutionPlan {
        let kernel = EagerKernelSpec {
            kernel_id: 7,
            role: EagerKernelRole::Vertex,
            inputs: vec![
                EagerKernelInput::FirstCurrentComponent(0),
                EagerKernelInput::SecondCurrentComponent(0),
            ],
            output_component_count: 1,
            homogeneous_linear_first_current: false,
            independent_block_size: 1,
        };
        let first_invocation = Invocation {
            site: KernelSite {
                kernel_id: 7,
                parent_current_ids: vec![0, 1].into_boxed_slice(),
                parent_momentum_ids: vec![0, 1].into_boxed_slice(),
                coupling: EagerComplex64::new(1.0, 0.0),
            },
            output_factor_source: OUTPUT_FACTOR_NONE,
            attachments: 0..1,
        };
        let second_invocation = Invocation {
            site: KernelSite {
                kernel_id: 7,
                parent_current_ids: vec![0, 0].into_boxed_slice(),
                parent_momentum_ids: vec![0, 0].into_boxed_slice(),
                coupling: EagerComplex64::new(1.0, 0.0),
            },
            output_factor_source: OUTPUT_FACTOR_NONE,
            attachments: 1..2,
        };
        let mut stages = vec![StagePlan::default(); 3];
        stages[2] = StagePlan {
            invocations: vec![first_invocation, second_invocation],
            attachments: vec![
                Attachment {
                    result_current_id: 2,
                    factor: EagerComplex64::new(1.0, 0.0),
                },
                Attachment {
                    result_current_id: 3,
                    factor: EagerComplex64::new(1.0, 0.0),
                },
            ],
            invocation_packets: vec![KernelPacket {
                kernel_spec_index: 0,
                calls: 0..2,
            }],
            finalizations: vec![],
            finalization_packets: vec![],
            identity_finalizations: vec![
                IdentityFinalization {
                    current_id: 2,
                    factor: EagerComplex64::new(1.0, 0.0),
                },
                IdentityFinalization {
                    current_id: 3,
                    factor: EagerComplex64::new(1.0, 0.0),
                },
            ],
        };
        RecurrenceExecutionPlan {
            program: RecurrenceProgram::new(
                RecurrenceStrategy::TopologyReplay,
                vec![],
                vec![],
                vec![],
                vec![],
                vec![CheckedTableRange::new(0, 0)],
                vec![],
            )
            .expect("empty semantic program is valid for an executor unit test"),
            kernels: vec![kernel].into_boxed_slice(),
            current_components: vec![
                ComponentRange { start: 0, len: 1 },
                ComponentRange { start: 1, len: 1 },
                ComponentRange { start: 2, len: 1 },
                ComponentRange { start: 3, len: 1 },
            ]
            .into_boxed_slice(),
            current_momenta: vec![momentum(0), momentum(1), momentum(0), momentum(0)]
                .into_boxed_slice(),
            sources: vec![
                SourceCopy {
                    current_id: 0,
                    source_slot: 0,
                    source_template_id: 0,
                    source_component_start: 0,
                    factor: EagerComplex64::new(1.0, 0.0),
                },
                SourceCopy {
                    current_id: 1,
                    source_slot: 1,
                    source_template_id: 1,
                    source_component_start: 1,
                    factor: EagerComplex64::new(1.0, 0.0),
                },
            ]
            .into_boxed_slice(),
            stages: stages.into_boxed_slice(),
            prepared_closures: Box::new([]),
            prepared_closure_packets: Box::new([]),
            direct_closures: vec![DirectClosureCall {
                parent_current_ids: [2, 3],
                target_sector_id: 0,
                coefficients: vec![EagerComplex64::new(1.0, 0.0)].into_boxed_slice(),
                factor: EagerComplex64::new(1.0, 0.0),
            }]
            .into_boxed_slice(),
            source_component_count: 2,
            current_component_count: 4,
            source_slot_count: 2,
            parameter_count: 0,
            sector_count: 1,
            maximum_packet_input_width: 4,
            maximum_packet_output_width: 2,
        }
    }

    #[test]
    fn evaluates_tiled_compact_recurrence_without_per_edge_backend_calls() {
        let mut runtime = RecurrenceExecutionRuntime::new(synthetic_plan(), 2)
            .expect("runtime workspace is valid");
        let mut backend = AddBackend::default();
        let sources = [
            EagerComplex64::new(1.0, 0.0),
            EagerComplex64::new(2.0, 0.0),
            EagerComplex64::new(3.0, 0.0),
            EagerComplex64::new(4.0, 0.0),
            EagerComplex64::new(5.0, 0.0),
            EagerComplex64::new(6.0, 0.0),
        ];
        let momenta = vec![0.0; 2 * 4 * 3];
        let mut output = vec![EagerComplex64::new(0.0, 0.0); 3];
        runtime
            .evaluate_one_helicity_amplitudes_into(
                &mut backend,
                3,
                &sources,
                &momenta,
                &[],
                &mut output,
            )
            .expect("synthetic recurrence evaluates");

        assert_eq!(backend.calls, vec![4, 2]);
        assert_eq!(
            output,
            vec![
                EagerComplex64::new(10.0, 0.0),
                EagerComplex64::new(28.0, 0.0),
                EagerComplex64::new(54.0, 0.0),
            ]
        );
    }
}
