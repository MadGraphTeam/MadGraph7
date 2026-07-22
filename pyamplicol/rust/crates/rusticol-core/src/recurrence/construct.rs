// SPDX-License-Identifier: 0BSD

//! Compact model-generic recurrence construction.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::process::{
    OwnedRecurrenceProcessInput, ProcessLCSectorKind, ProcessPhysicalLCSectorRow,
};
use super::template::{
    ClosureRow, LCColorTransitionWitnessRow, OwnedRecurrenceTemplateInput, QuantumFlowRow,
    SourceRow, TransitionRow,
};
use super::{
    AuthenticatedRecurrenceBuilderInput, CanonicalMomentumLinearForm, CheckedTableRange,
    ContributionKey, CurrentCoreKey, CurrentHelicityIdentity, CurrentSourceBinding,
    DynamicLCColorState, DynamicLCColorStateInterner, ExactComplexRational, LCColorComponent,
    LCColorComponentKind, LCColorComponentOperation, LCColorComponentRole, LCColorParentPort,
    LCColorPortWiring, LCColorSourceSeed, LCColorSourceSeedOperation, LCColorTransitionWitness,
    LCColorWitnessTermId, MomentumTerm, RecurrenceAmplitudeDestination, RecurrenceClosureTerm,
    RecurrenceContribution, RecurrenceCurrent, RecurrenceFinalization, RecurrenceNodeKind,
    RecurrenceProgram, RecurrenceReplayTarget, RecurrenceResolvedHelicity, RecurrenceStrategy,
    SemanticDigest, SourceStateAssignment,
};
use crate::{RusticolError, RusticolResult};

const MISSING_U32: u32 = u32::MAX;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PendingContributionKey {
    parent_current_ids: Box<[u32]>,
    key: ContributionKey,
}

#[derive(Clone, Debug)]
struct PendingCurrent {
    key: CurrentCoreKey,
    source_exact_factor: Option<ExactComplexRational>,
    contributions: BTreeMap<PendingContributionKey, ExactComplexRational>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PendingClosureKey {
    target_sector_id: u32,
    complete_source_states: Box<[SourceStateAssignment]>,
    closure_template_id: u32,
    quantum_flow_template_id: Option<u32>,
    parent_current_ids: Box<[u32]>,
}

#[derive(Clone, Debug, Default)]
struct StageConstructionDiagnostics {
    target_size: usize,
    parent_pair_count: usize,
    state_order_count: usize,
    quantum_match_count: usize,
    coupling_match_count: usize,
    color_shape_match_count: usize,
    color_result_count: usize,
    contribution_count: usize,
}

pub(super) struct TemplateCatalog<'a> {
    input: &'a OwnedRecurrenceTemplateInput,
    strings: Vec<&'a str>,
    digests: Vec<SemanticDigest>,
    factors: Vec<ExactComplexRational>,
    coupling_names: Vec<&'a str>,
}

impl<'a> TemplateCatalog<'a> {
    pub(super) fn new(input: &'a OwnedRecurrenceTemplateInput) -> RusticolResult<Self> {
        let strings = decode_strings(
            &input.string_ranges,
            &input.string_bytes,
            "recurrence template string",
        )?;
        let digests = input
            .digest_catalog
            .iter()
            .map(|row| SemanticDigest::new(row.value))
            .collect::<RusticolResult<Vec<_>>>()?;
        let factors = decode_template_factors(input, &strings)?;
        let coupling_names = input
            .coupling_order_terms
            .iter()
            .map(|row| required_string(&strings, row.name_string_id, "coupling-order name"))
            .collect::<RusticolResult<BTreeSet<_>>>()?
            .into_iter()
            .collect();
        Ok(Self {
            input,
            strings,
            digests,
            factors,
            coupling_names,
        })
    }

    fn string(&self, id: u32, label: &str) -> RusticolResult<&'a str> {
        required_string(&self.strings, id, label)
    }

    fn digest(&self, id: u32, label: &str) -> RusticolResult<SemanticDigest> {
        self.digests
            .get(id as usize)
            .copied()
            .ok_or_else(|| invalid(format!("{label} digest {id} is absent")))
    }

    pub(super) fn factor(&self, id: u32, label: &str) -> RusticolResult<ExactComplexRational> {
        self.factors
            .get(id as usize)
            .copied()
            .ok_or_else(|| invalid(format!("{label} factor {id} is absent")))
    }

    pub(super) fn u32_sequence(&self, id: u32, label: &str) -> RusticolResult<&'a [u32]> {
        indexed_sequence(
            &self.input.u32_sequence_ranges,
            &self.input.u32_sequence_values,
            id,
            label,
        )
    }

    fn i32_sequence(&self, id: u32, label: &str) -> RusticolResult<&'a [i32]> {
        indexed_sequence(
            &self.input.i32_sequence_ranges,
            &self.input.i32_sequence_values,
            id,
            label,
        )
    }

    fn flavour_flow(&self, id: u32, label: &str) -> RusticolResult<&'a [i32]> {
        indexed_sequence(
            &self.input.flavour_flow_ranges,
            &self.input.flavour_flow_values,
            id,
            label,
        )
    }

    fn source_seed(&self, row: SourceRow) -> RusticolResult<LCColorSourceSeed> {
        let operation = LCColorSourceSeedOperation::try_from(row.lc_color_seed_operation)?;
        let kind = (row.lc_color_seed_component_kind != u8::MAX)
            .then(|| LCColorComponentKind::try_from(row.lc_color_seed_component_kind))
            .transpose()?;
        LCColorSourceSeed::new(
            operation,
            row.lc_color_seed_shape_string_id,
            kind,
            LCColorComponentRole::try_from(row.lc_color_seed_component_role)?,
            self.digest(row.lc_color_seed_proof_digest_id, "source color proof")?,
        )
    }

    fn witness(
        &self,
        row: LCColorTransitionWitnessRow,
    ) -> RusticolResult<LCColorTransitionWitness> {
        let permutation = match row.input_permutation {
            0 => [0, 1],
            1 => [1, 0],
            value => return Err(invalid(format!("invalid LC witness permutation {value}"))),
        };
        let kind = (row.result_component_kind != u8::MAX)
            .then(|| LCColorComponentKind::try_from(row.result_component_kind))
            .transpose()?;
        let shape =
            (row.result_shape_string_id != MISSING_U32).then_some(row.result_shape_string_id);
        let pairing_values = self.u32_sequence(
            row.input_port_pairing_sequence_id,
            "LC witness input-port pairings",
        )?;
        if pairing_values.len() % 4 != 0 {
            return Err(invalid(
                "LC witness input-port pairing sequence is not divisible by four",
            ));
        }
        let input_pairings = pairing_values
            .chunks_exact(4)
            .map(|chunk| {
                Ok([
                    LCColorParentPort::new(
                        u8::try_from(chunk[0])
                            .map_err(|_| invalid("LC witness parent index exceeds u8"))?,
                        u8::try_from(chunk[1])
                            .map_err(|_| invalid("LC witness local port index exceeds u8"))?,
                    )?,
                    LCColorParentPort::new(
                        u8::try_from(chunk[2])
                            .map_err(|_| invalid("LC witness parent index exceeds u8"))?,
                        u8::try_from(chunk[3])
                            .map_err(|_| invalid("LC witness local port index exceeds u8"))?,
                    )?,
                ])
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let binding_values = self.u32_sequence(
            row.result_port_binding_sequence_id,
            "LC witness result-port bindings",
        )?;
        if binding_values.len() % 2 != 0 {
            return Err(invalid(
                "LC witness result-port binding sequence is not divisible by two",
            ));
        }
        let result_port_bindings = binding_values
            .chunks_exact(2)
            .map(|chunk| {
                LCColorParentPort::new(
                    u8::try_from(chunk[0])
                        .map_err(|_| invalid("LC witness parent index exceeds u8"))?,
                    u8::try_from(chunk[1])
                        .map_err(|_| invalid("LC witness local port index exceeds u8"))?,
                )
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let component_parent_order = match row.input_permutation {
            0 => [0, 1],
            1 => [1, 0],
            value => return Err(invalid(format!("invalid LC witness permutation {value}"))),
        };
        LCColorTransitionWitness::new(
            permutation,
            row.reverse_parent_mask,
            LCColorComponentOperation::try_from(row.component_operation)?,
            kind,
            LCColorComponentRole::try_from(row.result_component_role)?,
            shape,
            LCColorPortWiring::new(component_parent_order, input_pairings, result_port_bindings)?,
            self.factor(row.exact_factor_id, "LC witness")?,
            self.digest(row.proof_digest_id, "LC witness proof")?,
        )
    }

    fn witness_rows(
        &self,
        color_contraction_id: u32,
    ) -> RusticolResult<&'a [LCColorTransitionWitnessRow]> {
        let row = self
            .input
            .color_contractions
            .get(color_contraction_id as usize)
            .ok_or_else(|| invalid("color-contraction template is absent"))?;
        let range = CheckedTableRange::new(row.witness_start, row.witness_count).as_usize_range(
            self.input.lc_color_transition_witnesses.len(),
            "LC witnesses",
        )?;
        Ok(&self.input.lc_color_transition_witnesses[range])
    }

    fn coupling_orders(&self, set_id: u32) -> RusticolResult<Vec<u32>> {
        let range = self
            .input
            .coupling_order_ranges
            .get(set_id as usize)
            .ok_or_else(|| invalid(format!("coupling-order set {set_id} is absent")))?
            .range
            .as_usize_range(self.input.coupling_order_terms.len(), "coupling-order set")?;
        let mut result = vec![0_u32; self.coupling_names.len()];
        for term in &self.input.coupling_order_terms[range] {
            let name = self.string(term.name_string_id, "coupling-order term")?;
            let index = self
                .coupling_names
                .binary_search(&name)
                .map_err(|_| invalid("coupling-order name disappeared"))?;
            result[index] = term.power;
        }
        Ok(result)
    }
}

struct ProcessCatalog<'a> {
    input: &'a OwnedRecurrenceProcessInput,
    strings: Vec<&'a str>,
    factors: Vec<ExactComplexRational>,
}

impl<'a> ProcessCatalog<'a> {
    fn new(input: &'a OwnedRecurrenceProcessInput) -> RusticolResult<Self> {
        let strings = decode_strings(
            &input.string_ranges,
            &input.string_bytes,
            "recurrence process string",
        )?;
        let factors = decode_process_factors(input, &strings)?;
        Ok(Self {
            input,
            strings,
            factors,
        })
    }

    fn string(&self, id: u32, label: &str) -> RusticolResult<&'a str> {
        required_string(&self.strings, id, label)
    }

    fn factor(&self, id: u32, label: &str) -> RusticolResult<ExactComplexRational> {
        self.factors
            .get(id as usize)
            .copied()
            .ok_or_else(|| invalid(format!("{label} factor {id} is absent")))
    }

    fn u32_sequence(&self, id: u32, label: &str) -> RusticolResult<&'a [u32]> {
        let range = self
            .input
            .u32_sequence_ranges
            .get(id as usize)
            .copied()
            .ok_or_else(|| invalid(format!("{label} sequence {id} is absent")))?;
        let range = range.as_usize_range(self.input.u32_sequence_values.len(), label)?;
        Ok(&self.input.u32_sequence_values[range])
    }

    fn public_helicities(
        &self,
        source_states: &[SourceStateAssignment],
    ) -> RusticolResult<Vec<i32>> {
        if source_states.len() != self.input.external_legs.len() {
            return Err(invalid(
                "resolved helicity does not cover every external source slot",
            ));
        }
        source_states
            .iter()
            .copied()
            .enumerate()
            .map(|(source_slot, assignment)| {
                if assignment.source_slot() as usize != source_slot {
                    return Err(invalid(
                        "resolved-helicity source-state ancestry is not canonical",
                    ));
                }
                let leg = self.input.external_legs[source_slot];
                if u64::from(assignment.state_index()) >= leg.source_state_range.count {
                    return Err(invalid(format!(
                        "resolved helicity references absent state {} for source slot {source_slot}",
                        assignment.state_index()
                    )));
                }
                let row_index = leg
                    .source_state_range
                    .start
                    .checked_add(u64::from(assignment.state_index()))
                    .ok_or_else(|| invalid("resolved-helicity source-state index overflows"))?;
                let row_index = usize::try_from(row_index)
                    .map_err(|_| invalid("resolved-helicity source-state index exceeds usize"))?;
                let state = self
                    .input
                    .source_states
                    .get(row_index)
                    .ok_or_else(|| invalid("resolved-helicity source state is absent"))?;
                if state.source_slot as usize != source_slot
                    || state.state_index != assignment.state_index()
                {
                    return Err(invalid(
                        "resolved-helicity source-state catalog is inconsistent",
                    ));
                }
                Ok(state.public_helicity)
            })
            .collect()
    }
}

pub(super) fn build_recurrence_program(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
) -> RusticolResult<RecurrenceProgram> {
    let strategy = authenticated.process().summary().strategy();
    let catalog_digest = authenticated.template().summary().catalog_digest;
    let process_input = authenticated.process().input();
    let template_input = authenticated.template().input();
    let process_catalog = ProcessCatalog::new(process_input)?;
    let retained_helicity_count = retained_helicity_count(process_input)?;
    let template_catalog = TemplateCatalog::new(template_input)?;
    let coupling_limits = coupling_limits(&process_catalog, &template_catalog)?;
    let propagators = propagator_by_state(template_input)?;
    let replay_targets = build_replay_targets(strategy, process_input, &process_catalog)?;
    let materialized_sectors = materialized_sector_ids(strategy, process_input, &replay_targets);
    let mut color_states = DynamicLCColorStateInterner::default();
    let mut currents = Vec::<PendingCurrent>::new();
    let mut current_ids = BTreeMap::<CurrentCoreKey, u32>::new();
    let mut currents_by_size = vec![Vec::<u32>::new(); process_input.external_legs.len()];

    build_sources(
        strategy,
        catalog_digest,
        process_input,
        &process_catalog,
        template_input,
        &template_catalog,
        &mut color_states,
        &mut currents,
        &mut current_ids,
        &mut currents_by_size,
    )?;
    let stage_diagnostics = build_internal_currents(
        catalog_digest,
        process_input,
        template_input,
        &template_catalog,
        &coupling_limits,
        &propagators,
        &mut color_states,
        &mut currents,
        &mut current_ids,
        &mut currents_by_size,
    )?;
    let closures = build_closures(
        process_input,
        &process_catalog,
        template_input,
        &template_catalog,
        &color_states,
        &currents,
        &materialized_sectors,
        &stage_diagnostics,
    )?;
    finish_program(
        strategy,
        &process_catalog,
        color_states.into_states(),
        currents,
        closures,
        replay_targets,
        retained_helicity_count,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_sources(
    strategy: RecurrenceStrategy,
    catalog_digest: SemanticDigest,
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    template: &OwnedRecurrenceTemplateInput,
    template_catalog: &TemplateCatalog<'_>,
    color_states: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
    currents_by_size: &mut [Vec<u32>],
) -> RusticolResult<()> {
    let zero_orders = vec![0_u32; template_catalog.coupling_names.len()];
    for leg in &process.external_legs {
        let range = leg
            .source_state_range
            .as_usize_range(process.source_states.len(), "recurrence source-state range")?;
        let retained_state_indices = retained_source_state_indices(process, leg.source_slot)?;
        for state_index in retained_state_indices {
            let process_state = process
                .source_states
                .get(range.start + state_index as usize)
                .ok_or_else(|| invalid("retained recurrence source state is absent"))?;
            let source = *template
                .sources
                .get(process_state.source_template_id as usize)
                .ok_or_else(|| invalid("source template is absent"))?;
            validate_crossed_source_state(leg.is_initial != 0, process_state, source, template)?;
            let dynamic_state = template_catalog.source_seed(source)?.instantiate(
                leg.source_slot,
                template
                    .current_states
                    .get(source.state_template_id as usize)
                    .ok_or_else(|| invalid("source current-state template is absent"))?
                    .color_representation,
            )?;
            let color_id = color_states.intern(dynamic_state)?;
            let helicity_identity = match strategy {
                RecurrenceStrategy::TopologyReplay => CurrentHelicityIdentity::topology_replay(
                    process_state.spin_state,
                    vec![SourceStateAssignment::new(
                        leg.source_slot,
                        process_state.state_index,
                    )],
                )?,
                RecurrenceStrategy::AllFlowUnion => {
                    CurrentHelicityIdentity::all_flow_union(process_state.spin_state)
                }
            };
            let source_binding = match strategy {
                RecurrenceStrategy::TopologyReplay => {
                    CurrentSourceBinding::FixedTemplate(process_state.source_template_id)
                }
                RecurrenceStrategy::AllFlowUnion => {
                    CurrentSourceBinding::RuntimeDispatchDomain(leg.source_slot)
                }
            };
            let key = CurrentCoreKey::new(
                catalog_digest,
                RecurrenceNodeKind::Source,
                process_state.current_state_template_id,
                color_id,
                vec![leg.source_slot],
                CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                    source_slot: leg.source_slot,
                    coefficient: process_state.momentum_sign,
                }])?,
                helicity_identity,
                template_catalog
                    .flavour_flow(source.flavour_flow_id, "source flavour flow")?
                    .to_vec(),
                source.quantum_number_flow_id,
                zero_orders.clone(),
                source_binding,
                None,
            )?;
            let source_factor = match strategy {
                RecurrenceStrategy::TopologyReplay => Some(process_catalog.factor(
                    process_state.crossing_phase_factor_id,
                    "source crossing phase",
                )?),
                RecurrenceStrategy::AllFlowUnion => None,
            };
            if let Some(existing) = current_ids.get(&key).copied() {
                if currents[existing as usize].source_exact_factor != source_factor {
                    return Err(invalid(
                        "equivalent source currents have different exact factors",
                    ));
                }
                continue;
            }
            let id = u32::try_from(currents.len())
                .map_err(|_| invalid("recurrence current count exceeds u32"))?;
            current_ids.insert(key.clone(), id);
            currents.push(PendingCurrent {
                key,
                source_exact_factor: source_factor,
                contributions: BTreeMap::new(),
            });
            currents_by_size[0].push(id);
        }
    }
    Ok(())
}

fn validate_crossed_source_state(
    is_initial: bool,
    process_state: &super::process::ProcessSourceStateRow,
    source: SourceRow,
    template: &OwnedRecurrenceTemplateInput,
) -> RusticolResult<()> {
    let canonical = template
        .current_states
        .get(source.state_template_id as usize)
        .ok_or_else(|| invalid("source canonical current-state template is absent"))?;
    let effective = template
        .current_states
        .get(process_state.current_state_template_id as usize)
        .ok_or_else(|| invalid("source effective current-state template is absent"))?;
    if effective.chirality != process_state.chirality {
        return Err(invalid(format!(
            "process source chirality {} does not match effective current-state chirality {}",
            process_state.chirality, effective.chirality,
        )));
    }
    let compatible = canonical.particle_id == effective.particle_id
        && canonical.anti_particle_id == effective.anti_particle_id
        && canonical.species_string_id == effective.species_string_id
        && canonical.orientation == effective.orientation
        && canonical.statistics == effective.statistics
        && canonical.color_representation == effective.color_representation
        && canonical.basis_string_id == effective.basis_string_id
        && canonical.tensor_ordering_sequence_id == effective.tensor_ordering_sequence_id
        && canonical.dimension == effective.dimension
        && canonical.lc_color_shape_string_id == effective.lc_color_shape_string_id
        && canonical.auxiliary_kind_string_id == effective.auxiliary_kind_string_id
        && canonical.mass_parameter_id == effective.mass_parameter_id
        && canonical.width_parameter_id == effective.width_parameter_id;
    if !compatible || (!is_initial && canonical.id != effective.id) {
        return Err(invalid(format!(
            "source template {} and effective current-state template {} are not crossing-compatible",
            process_state.source_template_id, process_state.current_state_template_id,
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_internal_currents(
    catalog_digest: SemanticDigest,
    process: &OwnedRecurrenceProcessInput,
    template: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    coupling_limits: &[Option<u32>],
    propagators: &BTreeMap<u32, Option<u32>>,
    color_states: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
    currents_by_size: &mut [Vec<u32>],
) -> RusticolResult<Vec<StageConstructionDiagnostics>> {
    let mut diagnostics = Vec::new();
    for target_size in 2..process.external_legs.len() {
        let mut parent_pairs = Vec::new();
        for left_size in 1..target_size {
            let right_size = target_size - left_size;
            for &left_id in &currents_by_size[left_size - 1] {
                for &right_id in &currents_by_size[right_size - 1] {
                    if left_id >= right_id {
                        continue;
                    }
                    if disjoint_support(
                        currents[left_id as usize].key.support_source_slots(),
                        currents[right_id as usize].key.support_source_slots(),
                    ) {
                        parent_pairs.push((left_id, right_id));
                    }
                }
            }
        }
        parent_pairs.sort_unstable();
        parent_pairs.dedup();

        let mut stage = StageConstructionDiagnostics {
            target_size,
            parent_pair_count: parent_pairs.len(),
            ..StageConstructionDiagnostics::default()
        };

        for (left_id, right_id) in parent_pairs {
            for transition in &template.transitions {
                let input_states = catalog.u32_sequence(
                    transition.input_state_sequence_id,
                    "transition input states",
                )?;
                if input_states.len() != 2 {
                    return Err(invalid(
                        "recurrence v1 requires binary prepared transitions",
                    ));
                }
                let left_state = currents[left_id as usize].key.current_state_template_id();
                let right_state = currents[right_id as usize].key.current_state_template_id();
                let mut orders = Vec::new();
                if input_states == [left_state, right_state] {
                    orders.push([left_id, right_id]);
                }
                if input_states == [right_state, left_state] && left_state != right_state {
                    orders.push([right_id, left_id]);
                }
                stage.state_order_count += orders.len();
                for parent_ids in orders {
                    add_transition_contributions(
                        catalog_digest,
                        *transition,
                        parent_ids,
                        target_size + 1 < process.external_legs.len(),
                        template,
                        catalog,
                        coupling_limits,
                        propagators,
                        color_states,
                        currents,
                        current_ids,
                        &mut currents_by_size[target_size - 1],
                        &mut stage,
                    )?;
                }
            }
        }
        diagnostics.push(stage);
    }
    Ok(diagnostics)
}

#[allow(clippy::too_many_arguments)]
fn add_transition_contributions(
    catalog_digest: SemanticDigest,
    transition: TransitionRow,
    parent_ids: [u32; 2],
    propagate_result: bool,
    template: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    coupling_limits: &[Option<u32>],
    propagators: &BTreeMap<u32, Option<u32>>,
    color_states: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
    target_bucket: &mut Vec<u32>,
    diagnostics: &mut StageConstructionDiagnostics,
) -> RusticolResult<()> {
    let parent_keys = [
        currents[parent_ids[0] as usize].key.clone(),
        currents[parent_ids[1] as usize].key.clone(),
    ];
    let parents = [&parent_keys[0], &parent_keys[1]];
    let quantum = *template
        .quantum_flows
        .get(transition.quantum_flow_template_id as usize)
        .ok_or_else(|| invalid("transition quantum-flow template is absent"))?;
    if !quantum_flow_matches(quantum, &parents, catalog)? {
        return Ok(());
    }
    diagnostics.quantum_match_count += 1;
    let Some(coupling_orders) = combined_coupling_orders(
        parents[0].coupling_orders(),
        parents[1].coupling_orders(),
        &catalog.coupling_orders(transition.coupling_order_set_id)?,
        coupling_limits,
    )?
    else {
        return Ok(());
    };
    diagnostics.coupling_match_count += 1;
    let contraction = *template
        .color_contractions
        .get(transition.color_contraction_template_id as usize)
        .ok_or_else(|| invalid("transition color contraction is absent"))?;
    authenticate_runtime_coupling(
        catalog,
        quantum,
        transition.binding_coupling_factor_id,
        "transition",
    )?;
    let (evaluator_parent_ids, exchange_factor) = canonical_evaluator_parents(
        parent_ids,
        catalog.u32_sequence(
            transition.canonical_input_order_sequence_id,
            "transition canonical input order",
        )?,
        transition.input_exchange_factor_id,
        catalog,
        "transition",
    )?;
    let base_factor = multiply_factors(&[
        catalog.factor(transition.exact_factor_id, "transition exact")?,
        exchange_factor,
        catalog.factor(contraction.exact_coefficient_factor_id, "color contraction")?,
    ])?;

    for witness_row in catalog.witness_rows(transition.color_contraction_template_id)? {
        let left_color = color_states
            .get(parents[0].dynamic_lc_color_state_id())
            .ok_or_else(|| invalid("left dynamic color state disappeared"))?;
        let right_color = color_states
            .get(parents[1].dynamic_lc_color_state_id())
            .ok_or_else(|| invalid("right dynamic color state disappeared"))?;
        if witness_row.left_shape_string_id != left_color.output_color_shape_id()
            || witness_row.right_shape_string_id != right_color.output_color_shape_id()
        {
            continue;
        }
        diagnostics.color_shape_match_count += 1;
        let witness = catalog.witness(*witness_row)?;
        let Some(result_color) = witness.apply(left_color, right_color)? else {
            continue;
        };
        diagnostics.color_result_count += 1;
        let result_color_id = color_states.intern(result_color)?;
        let support = merged_support(
            parents[0].support_source_slots(),
            parents[1].support_source_slots(),
        )?;
        let helicity_identity = merged_helicity_identity(
            parents[0].helicity_identity(),
            parents[1].helicity_identity(),
            quantum.result_spin_state,
        )?;
        let result_flavour_flow = quantum_flow_result_flavour(quantum, &parents, catalog)?;
        let key = CurrentCoreKey::new(
            catalog_digest,
            RecurrenceNodeKind::Current,
            transition.result_state_template_id,
            result_color_id,
            support,
            merged_momentum(parents[0].momentum(), parents[1].momentum())?,
            helicity_identity,
            result_flavour_flow,
            quantum.result_quantum_number_flow_id,
            coupling_orders.clone(),
            CurrentSourceBinding::None,
            if propagate_result {
                propagators
                    .get(&transition.result_state_template_id)
                    .copied()
                    .flatten()
            } else {
                None
            },
        )?;
        let result_id = if let Some(id) = current_ids.get(&key).copied() {
            id
        } else {
            let id = u32::try_from(currents.len())
                .map_err(|_| invalid("recurrence current count exceeds u32"))?;
            current_ids.insert(key.clone(), id);
            currents.push(PendingCurrent {
                key,
                source_exact_factor: None,
                contributions: BTreeMap::new(),
            });
            target_bucket.push(id);
            id
        };
        let contribution_key = ContributionKey::new(
            transition.id,
            evaluator_parent_ids.to_vec(),
            evaluator_parent_ids
                .iter()
                .map(|id| currents[*id as usize].key.current_state_template_id())
                .collect(),
            evaluator_parent_ids
                .iter()
                .map(|id| currents[*id as usize].key.momentum().clone())
                .collect(),
            transition.result_state_template_id,
            quantum.id,
            LCColorWitnessTermId::new(
                transition.color_contraction_template_id,
                witness_row.ordinal,
            ),
            catalog.digest(quantum.semantic_digest_id, "quantum-flow semantic")?,
            transition.output_projection_string_id,
        )?;
        let pending_key = PendingContributionKey {
            parent_current_ids: evaluator_parent_ids.into(),
            key: contribution_key,
        };
        let factor = base_factor.checked_mul(witness.exact_factor())?;
        aggregate_factor(
            currents[result_id as usize]
                .contributions
                .entry(pending_key)
                .or_insert(ExactComplexRational::ZERO),
            factor,
        )?;
        diagnostics.contribution_count += 1;
    }
    Ok(())
}

fn build_closures(
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    template: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    color_states: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    materialized_sectors: &BTreeSet<u32>,
    stage_diagnostics: &[StageConstructionDiagnostics],
) -> RusticolResult<BTreeMap<PendingClosureKey, ExactComplexRational>> {
    let full_support = (0..process.external_legs.len() as u32).collect::<Vec<_>>();
    let mut result = BTreeMap::new();
    for sector_id in materialized_sectors.iter().copied() {
        let sector = process.physical_lc_sectors[sector_id as usize];
        let complement = full_support
            .iter()
            .copied()
            .filter(|slot| *slot != sector.closure_source_slot)
            .collect::<Vec<_>>();
        let anchor_ids = currents
            .iter()
            .enumerate()
            .filter(|(_, current)| {
                current.key.node_kind() == RecurrenceNodeKind::Source
                    && current.key.support_source_slots() == [sector.closure_source_slot]
            })
            .map(|(id, _)| id as u32)
            .collect::<Vec<_>>();
        let complement_ids = currents
            .iter()
            .enumerate()
            .filter(|(_, current)| current.key.support_source_slots() == complement)
            .map(|(id, _)| id as u32)
            .collect::<Vec<_>>();
        let anchor_count = anchor_ids.len();
        let complement_count = complement_ids.len();
        let mut state_matched_attempts = 0usize;
        let mut closure_color_attempts = BTreeSet::new();
        for anchor_id in anchor_ids {
            for &complement_id in &complement_ids {
                for closure in &template.closures {
                    let input_states = catalog
                        .u32_sequence(closure.input_state_sequence_id, "closure input states")?;
                    if input_states.len() != 2 {
                        return Err(invalid("recurrence v1 requires binary prepared closures"));
                    }
                    let anchor_state = currents[anchor_id as usize].key.current_state_template_id();
                    let complement_state = currents[complement_id as usize]
                        .key
                        .current_state_template_id();
                    let mut orders = Vec::new();
                    if input_states == [complement_state, anchor_state] {
                        orders.push([complement_id, anchor_id]);
                    }
                    if input_states == [anchor_state, complement_state]
                        && anchor_state != complement_state
                    {
                        orders.push([anchor_id, complement_id]);
                    }
                    state_matched_attempts = state_matched_attempts
                        .checked_add(orders.len())
                        .ok_or_else(|| invalid("closure-attempt count exceeds usize"))?;
                    for parent_ids in orders {
                        add_closure_terms(
                            sector,
                            *closure,
                            parent_ids,
                            process,
                            process_catalog,
                            template,
                            catalog,
                            color_states,
                            currents,
                            &mut result,
                            &mut closure_color_attempts,
                        )?;
                    }
                }
            }
        }
        if !result.keys().any(|key| key.target_sector_id == sector_id) {
            let mut support_histogram = BTreeMap::<usize, usize>::new();
            let mut support_signatures = BTreeSet::new();
            for current in currents {
                *support_histogram
                    .entry(current.key.support_source_slots().len())
                    .or_default() += 1;
                support_signatures.insert((
                    current.key.support_source_slots().len(),
                    current.key.current_state_template_id(),
                    current.key.spin_state_class(),
                    current.key.flavour_flow().to_vec(),
                    current.key.quantum_number_flow_id(),
                    current.key.coupling_orders().to_vec(),
                ));
            }
            return Err(invalid(format!(
                "recurrence builder found no exact closure for physical LC sector {sector_id} \
                 (anchors={anchor_count}, complement_currents={complement_count}, \
                 state_matched_attempts={state_matched_attempts}, \
                 currents_by_support_size={support_histogram:?}, \
                 stage_diagnostics={stage_diagnostics:?}, \
                 expected_color_components={:?}, \
                 closure_color_attempts={closure_color_attempts:?}, \
                 support_signatures={support_signatures:?})",
                expected_sector_components(sector, process, process_catalog)?,
            )));
        }
    }
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn add_closure_terms(
    sector: ProcessPhysicalLCSectorRow,
    closure: ClosureRow,
    parent_ids: [u32; 2],
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    template: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    color_states: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    result: &mut BTreeMap<PendingClosureKey, ExactComplexRational>,
    closure_color_attempts: &mut BTreeSet<Vec<(LCColorComponentKind, Vec<u32>)>>,
) -> RusticolResult<()> {
    let parents = [
        &currents[parent_ids[0] as usize].key,
        &currents[parent_ids[1] as usize].key,
    ];
    let contraction = template
        .color_contractions
        .get(closure.color_contraction_template_id as usize)
        .copied()
        .ok_or_else(|| invalid("closure color contraction is absent"))?;
    let eligible = catalog.u32_sequence(
        closure.eligible_quantum_flow_sequence_id,
        "closure eligible quantum flows",
    )?;
    let quantum_flows = if eligible.is_empty() {
        vec![None]
    } else {
        let mut flows = Vec::new();
        for quantum_id in eligible {
            let quantum = *template
                .quantum_flows
                .get(*quantum_id as usize)
                .ok_or_else(|| invalid("closure quantum flow is absent"))?;
            if quantum_flow_matches(quantum, &parents, catalog)? {
                flows.push(Some(quantum));
            }
        }
        flows
    };
    for quantum in quantum_flows {
        if let Some(quantum) = quantum {
            authenticate_runtime_coupling(
                catalog,
                quantum,
                closure.binding_coupling_factor_id,
                "closure",
            )?;
        }
        let (evaluator_parent_ids, exchange_factor) = canonical_evaluator_parents(
            parent_ids,
            catalog.u32_sequence(
                closure.canonical_input_order_sequence_id,
                "closure canonical input order",
            )?,
            closure.input_exchange_factor_id,
            catalog,
            "closure",
        )?;
        let base_factor = multiply_factors(&[
            catalog.factor(closure.exact_factor_id, "closure exact")?,
            exchange_factor,
            catalog.factor(contraction.exact_coefficient_factor_id, "closure color")?,
        ])?;
        for witness_row in catalog.witness_rows(closure.color_contraction_template_id)? {
            let left = color_states
                .get(parents[0].dynamic_lc_color_state_id())
                .ok_or_else(|| invalid("closure left color state disappeared"))?;
            let right = color_states
                .get(parents[1].dynamic_lc_color_state_id())
                .ok_or_else(|| invalid("closure right color state disappeared"))?;
            if witness_row.left_shape_string_id != left.output_color_shape_id()
                || witness_row.right_shape_string_id != right.output_color_shape_id()
            {
                continue;
            }
            let witness = catalog.witness(*witness_row)?;
            let closed = witness.closed_components(left, right)?;
            closure_color_attempts.insert(
                closed
                    .iter()
                    .map(|component| (component.kind(), component.source_slots().to_vec()))
                    .collect(),
            );
            if !closed_components_match_sector(&closed, sector, process, process_catalog, template)?
            {
                continue;
            }
            let key = PendingClosureKey {
                target_sector_id: sector.sector_id,
                complete_source_states: complete_closure_source_states(
                    parents,
                    process.external_legs.len(),
                )?,
                closure_template_id: closure.id,
                quantum_flow_template_id: quantum.map(|row| row.id),
                parent_current_ids: evaluator_parent_ids.into(),
            };
            let factor = base_factor.checked_mul(witness.exact_factor())?;
            aggregate_factor(
                result.entry(key).or_insert(ExactComplexRational::ZERO),
                factor,
            )?;
        }
    }
    Ok(())
}

fn finish_program(
    strategy: RecurrenceStrategy,
    process_catalog: &ProcessCatalog<'_>,
    dynamic_color_states: Vec<DynamicLCColorState>,
    pending: Vec<PendingCurrent>,
    pending_closures: BTreeMap<PendingClosureKey, ExactComplexRational>,
    replay_targets: Vec<RecurrenceReplayTarget>,
    retained_helicity_count: u64,
) -> RusticolResult<RecurrenceProgram> {
    let sector_count = process_catalog.input.physical_lc_sectors.len();
    let helicity_keys = pending_closures
        .iter()
        .filter(|(key, factor)| !key.complete_source_states.is_empty() && !factor.is_zero())
        .map(|(key, _)| key)
        .map(|key| key.complete_source_states.clone())
        .collect::<BTreeSet<_>>();
    let helicity_ids = helicity_keys
        .iter()
        .cloned()
        .enumerate()
        .map(|(id, key)| (key, id as u32))
        .collect::<BTreeMap<_, _>>();
    let resolved_helicities = helicity_keys
        .into_iter()
        .enumerate()
        .map(|(id, source_states)| {
            let public_helicities = process_catalog.public_helicities(&source_states)?;
            RecurrenceResolvedHelicity::new(id as u32, source_states.into(), public_helicities)
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let mut live = BTreeSet::new();
    let mut queue = VecDeque::new();
    for (key, factor) in &pending_closures {
        if factor.is_zero() {
            continue;
        }
        for parent in key.parent_current_ids.iter().copied() {
            if live.insert(parent) {
                queue.push_back(parent);
            }
        }
    }
    while let Some(current_id) = queue.pop_front() {
        for (contribution, factor) in &pending[current_id as usize].contributions {
            if factor.is_zero() {
                continue;
            }
            for parent in contribution.parent_current_ids.iter().copied() {
                if live.insert(parent) {
                    queue.push_back(parent);
                }
            }
        }
    }
    let remap = live
        .iter()
        .copied()
        .enumerate()
        .map(|(new, old)| (old, new as u32))
        .collect::<BTreeMap<_, _>>();
    let mut currents = Vec::with_capacity(live.len());
    let mut contributions = Vec::new();
    let mut finalizations = Vec::new();
    for old_id in live.iter().copied() {
        let pending_current = &pending[old_id as usize];
        let start = contributions.len() as u64;
        for (pending_key, factor) in &pending_current.contributions {
            if factor.is_zero() {
                continue;
            }
            let parent_ids = pending_key
                .parent_current_ids
                .iter()
                .map(|id| {
                    remap
                        .get(id)
                        .copied()
                        .ok_or_else(|| invalid("live parent is absent"))
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            let old_key = &pending_key.key;
            let key = ContributionKey::new(
                old_key.transition_template_id(),
                parent_ids.clone(),
                old_key.parent_state_template_ids().to_vec(),
                old_key.parent_momenta().to_vec(),
                old_key.result_state_template_id(),
                old_key.quantum_flow_witness_id(),
                old_key.color_witness_term_id(),
                old_key.runtime_coupling_binding_digest(),
                old_key.output_projection_id(),
            )?;
            contributions.push(RecurrenceContribution::new(
                contributions.len() as u32,
                remap[&old_id],
                parent_ids,
                key,
                *factor,
            )?);
        }
        let count = contributions.len() as u64 - start;
        let finalization_id = if pending_current.key.node_kind() == RecurrenceNodeKind::Current {
            let id = finalizations.len() as u32;
            finalizations.push(RecurrenceFinalization::new(
                id,
                remap[&old_id],
                pending_current.key.propagator_template_id(),
                ExactComplexRational::ONE,
            )?);
            Some(id)
        } else {
            None
        };
        currents.push(RecurrenceCurrent::new(
            remap[&old_id],
            pending_current.key.clone(),
            pending_current.source_exact_factor,
            CheckedTableRange::new(start, count),
            finalization_id,
        )?);
    }

    let destination_keys = pending_closures
        .iter()
        .filter(|(_, factor)| !factor.is_zero())
        .map(|(key, _)| (key.target_sector_id, key.complete_source_states.clone()))
        .collect::<BTreeSet<_>>();
    let mut closure_terms = Vec::new();
    let mut amplitude_destinations = Vec::with_capacity(destination_keys.len());
    for (destination_id, (sector_id, source_states)) in destination_keys.into_iter().enumerate() {
        let start = closure_terms.len() as u64;
        for (key, factor) in pending_closures.iter().filter(|(key, factor)| {
            key.target_sector_id == sector_id
                && key.complete_source_states == source_states
                && !factor.is_zero()
        }) {
            let parents = key
                .parent_current_ids
                .iter()
                .map(|id| {
                    remap
                        .get(id)
                        .copied()
                        .ok_or_else(|| invalid("closure parent is absent"))
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            closure_terms.push(RecurrenceClosureTerm::new(
                closure_terms.len() as u32,
                destination_id as u32,
                key.closure_template_id,
                key.quantum_flow_template_id,
                parents,
                *factor,
            )?);
        }
        let target_helicity_id = if source_states.is_empty() {
            None
        } else {
            Some(
                *helicity_ids
                    .get(&source_states)
                    .ok_or_else(|| invalid("resolved-helicity destination disappeared"))?,
            )
        };
        amplitude_destinations.push(RecurrenceAmplitudeDestination::new(
            destination_id as u32,
            sector_id,
            target_helicity_id,
            CheckedTableRange::new(start, closure_terms.len() as u64 - start),
        )?);
    }
    RecurrenceProgram::new(
        strategy,
        u32::try_from(sector_count).map_err(|_| invalid("physical sector count exceeds u32"))?,
        retained_helicity_count,
        dynamic_color_states,
        currents,
        contributions,
        finalizations,
        replay_targets,
        resolved_helicities,
        amplitude_destinations,
        closure_terms,
    )
}

fn retained_helicity_count(process: &OwnedRecurrenceProcessInput) -> RusticolResult<u64> {
    process.external_legs.iter().try_fold(1_u64, |count, leg| {
        let retained =
            u64::try_from(retained_source_state_indices(process, leg.source_slot)?.len())
                .map_err(|_| invalid("retained source-state count exceeds u64"))?;
        count
            .checked_mul(retained)
            .ok_or_else(|| invalid("retained public-helicity count exceeds u64"))
    })
}

fn retained_source_state_indices(
    process: &OwnedRecurrenceProcessInput,
    source_slot: u32,
) -> RusticolResult<Vec<u32>> {
    let leg = process
        .external_legs
        .get(source_slot as usize)
        .ok_or_else(|| invalid("recurrence source slot is absent"))?;
    if !process.header[0].selected_source_mode()? {
        return (0..leg.source_state_range.count)
            .map(|index| {
                u32::try_from(index)
                    .map_err(|_| invalid("source-state index exceeds the u32 ID domain"))
            })
            .collect();
    }
    let retained = process
        .selected_source_coverage
        .iter()
        .filter(|row| row.source_slot == source_slot)
        .map(|row| row.source_state_index)
        .collect::<Vec<_>>();
    if retained.is_empty() {
        return Err(invalid(format!(
            "generation-selected recurrence coverage has no state for source slot {source_slot}"
        )));
    }
    Ok(retained)
}

fn complete_closure_source_states(
    parents: [&CurrentCoreKey; 2],
    source_count: usize,
) -> RusticolResult<Box<[SourceStateAssignment]>> {
    match (
        parents[0].helicity_identity(),
        parents[1].helicity_identity(),
    ) {
        (
            CurrentHelicityIdentity::TopologyReplay {
                local_source_states: left,
                ..
            },
            CurrentHelicityIdentity::TopologyReplay {
                local_source_states: right,
                ..
            },
        ) => {
            let mut result = left.iter().chain(right.iter()).copied().collect::<Vec<_>>();
            result.sort_unstable();
            if result.len() != source_count {
                return Err(invalid(
                    "topology-replay closure ancestry does not cover every external source",
                ));
            }
            for (source_slot, assignment) in result.iter().copied().enumerate() {
                if assignment.source_slot() as usize != source_slot {
                    return Err(invalid(
                        "topology-replay closure ancestry is incomplete or overlapping",
                    ));
                }
            }
            Ok(result.into_boxed_slice())
        }
        (
            CurrentHelicityIdentity::AllFlowUnion { .. },
            CurrentHelicityIdentity::AllFlowUnion { .. },
        ) => Ok(Box::new([])),
        _ => Err(invalid(
            "closure parents use incompatible recurrence helicity strategies",
        )),
    }
}

fn quantum_flow_matches(
    quantum: QuantumFlowRow,
    parents: &[&CurrentCoreKey; 2],
    catalog: &TemplateCatalog<'_>,
) -> RusticolResult<bool> {
    let states = catalog.u32_sequence(quantum.input_state_sequence_id, "quantum input states")?;
    let spins = catalog.i32_sequence(quantum.input_spin_sequence_id, "quantum input spins")?;
    let flavours = catalog.u32_sequence(
        quantum.input_flavour_sequence_id,
        "quantum input flavour flows",
    )?;
    let quantum_numbers = catalog.u32_sequence(
        quantum.input_quantum_sequence_id,
        "quantum input number flows",
    )?;
    if states.len() != 2 || spins.len() != 2 || flavours.len() != 2 || quantum_numbers.len() != 2 {
        return Err(invalid(
            "recurrence v1 requires binary quantum-flow contracts",
        ));
    }
    for index in 0..2 {
        // The authenticated model contract proves that branch admission is
        // independent of accumulated flavour and quantum-number ancestry.
        // Those template columns describe the seed probe used to certify the
        // branch, while state and spin remain the actual admission keys.
        let _ = catalog.flavour_flow(flavours[index], "quantum parent flavour")?;
        let _ = quantum_numbers[index];
        if states[index] != parents[index].current_state_template_id()
            || spins[index] != parents[index].spin_state_class()
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn quantum_flow_result_flavour(
    quantum: QuantumFlowRow,
    parents: &[&CurrentCoreKey; 2],
    catalog: &TemplateCatalog<'_>,
) -> RusticolResult<Vec<i32>> {
    let operation = catalog.string(
        quantum.flavour_flow_operation_string_id,
        "quantum-flow flavour operation",
    )?;
    let static_result = catalog.flavour_flow(
        quantum.result_flavour_flow_id,
        "quantum-flow result flavour",
    )?;
    let result_particle = *static_result
        .last()
        .ok_or_else(|| invalid("quantum-flow result flavour ancestry is empty"))?;

    let append_result = |parent: &CurrentCoreKey| {
        let mut result = parent.flavour_flow().to_vec();
        if result.last().copied() != Some(result_particle) {
            result.push(result_particle);
        }
        result
    };

    match operation {
        "constant-result" => Ok(static_result.to_vec()),
        "append-left-result" => Ok(append_result(parents[0])),
        "append-right-result" => Ok(append_result(parents[1])),
        "concat-left-right-result" => {
            let mut result = Vec::with_capacity(
                parents[0]
                    .flavour_flow()
                    .len()
                    .saturating_add(parents[1].flavour_flow().len())
                    .saturating_add(1),
            );
            result.extend_from_slice(parents[0].flavour_flow());
            result.extend_from_slice(parents[1].flavour_flow());
            result.push(result_particle);
            Ok(result)
        }
        value => Err(invalid(format!(
            "unsupported quantum-flow flavour operation {value:?}"
        ))),
    }
}

fn expected_sector_components(
    sector: ProcessPhysicalLCSectorRow,
    process: &OwnedRecurrenceProcessInput,
    catalog: &ProcessCatalog<'_>,
) -> RusticolResult<Vec<LCColorComponent>> {
    let mut result = Vec::new();
    match sector.kind()? {
        ProcessLCSectorKind::Singlet => {}
        ProcessLCSectorKind::SingleTrace => result.push(LCColorComponent::new(
            LCColorComponentKind::Trace,
            catalog
                .u32_sequence(sector.trace_sequence_id, "physical LC trace")?
                .to_vec(),
        )?),
        ProcessLCSectorKind::OpenLines => {
            let range = sector
                .open_string_range
                .as_usize_range(process.lc_open_strings.len(), "physical LC open strings")?;
            for row in &process.lc_open_strings[range] {
                let mut word = vec![row.fundamental_source_slot];
                word.extend_from_slice(
                    catalog.u32_sequence(
                        row.adjoint_sequence_id,
                        "physical LC open-string adjoints",
                    )?,
                );
                word.push(row.antifundamental_source_slot);
                result.push(LCColorComponent::new(
                    LCColorComponentKind::OpenString,
                    word,
                )?);
            }
        }
    }
    Ok(result)
}

fn closed_components_match_sector(
    closed: &[LCColorComponent],
    sector: ProcessPhysicalLCSectorRow,
    process: &OwnedRecurrenceProcessInput,
    catalog: &ProcessCatalog<'_>,
    _template: &OwnedRecurrenceTemplateInput,
) -> RusticolResult<bool> {
    let expected = expected_sector_components(sector, process, catalog)?;
    if sector.kind()? != ProcessLCSectorKind::OpenLines {
        return Ok(closed == expected);
    }
    if closed.len() != expected.len() {
        return Ok(false);
    }

    // Independent open strings form an unordered physical forest at closure,
    // even though their construction order remains part of every partial-current
    // identity. Match the exact line blocks without canonicalizing the states so
    // alternative multi-line closure partners survive as distinct terms.
    let mut matched = vec![false; expected.len()];
    for component in closed {
        let Some(index) = expected.iter().enumerate().find_map(|(index, candidate)| {
            (!matched[index] && candidate == component).then_some(index)
        }) else {
            return Ok(false);
        };
        matched[index] = true;
    }
    Ok(true)
}

fn materialized_sector_ids(
    strategy: RecurrenceStrategy,
    process: &OwnedRecurrenceProcessInput,
    replay_targets: &[RecurrenceReplayTarget],
) -> BTreeSet<u32> {
    match strategy {
        RecurrenceStrategy::AllFlowUnion => process
            .physical_lc_sectors
            .iter()
            .map(|sector| sector.sector_id)
            .collect(),
        RecurrenceStrategy::TopologyReplay => replay_targets
            .iter()
            .map(RecurrenceReplayTarget::materialized_sector_id)
            .collect(),
    }
}

fn build_replay_targets(
    strategy: RecurrenceStrategy,
    process: &OwnedRecurrenceProcessInput,
    catalog: &ProcessCatalog<'_>,
) -> RusticolResult<Vec<RecurrenceReplayTarget>> {
    if strategy == RecurrenceStrategy::AllFlowUnion {
        return Ok(Vec::new());
    }
    let retained_sectors = retained_physical_sector_ids(process)?;
    let mut rows = Vec::<(u32, u32, Vec<u32>, ExactComplexRational)>::new();
    let mut covered = BTreeSet::new();
    for partition in &process.replay_partitions {
        let range = partition
            .target_range
            .as_usize_range(process.replay_targets.len(), "recurrence replay targets")?;
        for target in &process.replay_targets[range] {
            if !retained_sectors.contains(&target.sector_id) {
                continue;
            }
            let mut factor = catalog.factor(
                target.amplitude_phase_factor_id,
                "recurrence replay amplitude phase",
            )?;
            if target.fermion_sign == -1 {
                factor = factor.checked_neg()?;
            }
            rows.push((
                partition.materialized_sector_id,
                target.sector_id,
                catalog
                    .u32_sequence(
                        target.source_slot_permutation_sequence_id,
                        "recurrence replay source permutation",
                    )?
                    .to_vec(),
                factor,
            ));
            covered.insert(target.sector_id);
        }
    }
    let identity = (0..process.external_legs.len())
        .map(|slot| u32::try_from(slot).map_err(|_| invalid("source-slot count exceeds u32")))
        .collect::<RusticolResult<Vec<_>>>()?;
    for sector_id in retained_sectors.difference(&covered).copied() {
        rows.push((
            sector_id,
            sector_id,
            identity.clone(),
            ExactComplexRational::ONE,
        ));
    }
    rows.sort_by_key(|(_, target_sector_id, _, _)| *target_sector_id);
    rows.into_iter()
        .enumerate()
        .map(|(id, (materialized, target, permutation, factor))| {
            RecurrenceReplayTarget::new(
                u32::try_from(id).map_err(|_| invalid("replay-target count exceeds u32"))?,
                materialized,
                target,
                permutation,
                factor,
            )
        })
        .collect()
}

fn retained_physical_sector_ids(
    process: &OwnedRecurrenceProcessInput,
) -> RusticolResult<BTreeSet<u32>> {
    if !process.header[0].selected_flow_mode()? {
        return Ok(process
            .physical_lc_sectors
            .iter()
            .map(|sector| sector.sector_id)
            .collect());
    }
    process
        .selected_public_flow_coverage
        .iter()
        .map(|selected| {
            process
                .public_lc_flows
                .get(selected.flow_id as usize)
                .map(|flow| flow.construction_sector_id)
                .ok_or_else(|| invalid("selected public LC flow is absent"))
        })
        .collect()
}

fn propagator_by_state(
    template: &OwnedRecurrenceTemplateInput,
) -> RusticolResult<BTreeMap<u32, Option<u32>>> {
    let mut result = BTreeMap::new();
    for row in &template.propagators {
        let value = Some(row.id);
        if result.insert(row.state_template_id, value).is_some() {
            return Err(invalid(format!(
                "current-state template {} has multiple propagators",
                row.state_template_id
            )));
        }
    }
    Ok(result)
}

fn coupling_limits(
    process: &ProcessCatalog<'_>,
    template: &TemplateCatalog<'_>,
) -> RusticolResult<Vec<Option<u32>>> {
    let mut limits = vec![None; template.coupling_names.len()];
    for row in &process.input.coupling_limits {
        let name = process.string(row.name_string_id, "process coupling limit")?;
        if let Ok(index) = template.coupling_names.binary_search(&name) {
            limits[index] = Some(row.maximum);
        }
    }
    Ok(limits)
}

fn combined_coupling_orders(
    left: &[u32],
    right: &[u32],
    local: &[u32],
    limits: &[Option<u32>],
) -> RusticolResult<Option<Vec<u32>>> {
    if left.len() != right.len() || left.len() != local.len() || left.len() != limits.len() {
        return Err(invalid(
            "coupling-order vectors have inconsistent dimensions",
        ));
    }
    let mut result = Vec::with_capacity(left.len());
    for index in 0..left.len() {
        let value = left[index]
            .checked_add(right[index])
            .and_then(|value| value.checked_add(local[index]))
            .ok_or_else(|| invalid("coupling order exceeds u32"))?;
        if limits[index].is_some_and(|maximum| value > maximum) {
            return Ok(None);
        }
        result.push(value);
    }
    Ok(Some(result))
}

fn merged_helicity_identity(
    left: &CurrentHelicityIdentity,
    right: &CurrentHelicityIdentity,
    result_spin: i32,
) -> RusticolResult<CurrentHelicityIdentity> {
    match (left, right) {
        (
            CurrentHelicityIdentity::TopologyReplay {
                local_source_states: left,
                ..
            },
            CurrentHelicityIdentity::TopologyReplay {
                local_source_states: right,
                ..
            },
        ) => {
            let mut values = left.iter().chain(right.iter()).copied().collect::<Vec<_>>();
            values.sort_unstable();
            CurrentHelicityIdentity::topology_replay(result_spin, values)
        }
        (
            CurrentHelicityIdentity::AllFlowUnion { .. },
            CurrentHelicityIdentity::AllFlowUnion { .. },
        ) => Ok(CurrentHelicityIdentity::all_flow_union(result_spin)),
        _ => Err(invalid(
            "cannot merge recurrence helicity identities from different strategies",
        )),
    }
}

fn merged_support(left: &[u32], right: &[u32]) -> RusticolResult<Vec<u32>> {
    if !disjoint_support(left, right) {
        return Err(invalid(
            "recurrence parents have overlapping source support",
        ));
    }
    let mut result = left.iter().chain(right).copied().collect::<Vec<_>>();
    result.sort_unstable();
    Ok(result)
}

fn disjoint_support(left: &[u32], right: &[u32]) -> bool {
    left.iter().all(|slot| right.binary_search(slot).is_err())
}

fn merged_momentum(
    left: &CanonicalMomentumLinearForm,
    right: &CanonicalMomentumLinearForm,
) -> RusticolResult<CanonicalMomentumLinearForm> {
    let mut terms = left
        .terms()
        .iter()
        .chain(right.terms())
        .copied()
        .collect::<Vec<_>>();
    terms.sort_unstable_by_key(|term| term.source_slot);
    CanonicalMomentumLinearForm::new(terms)
}

fn authenticate_runtime_coupling(
    catalog: &TemplateCatalog<'_>,
    quantum: QuantumFlowRow,
    binding_coupling_factor_id: u32,
    label: &str,
) -> RusticolResult<()> {
    let quantum_coupling = catalog.factor(
        quantum.exact_coupling_factor_id,
        &format!("{label} quantum-flow coupling"),
    )?;
    let binding_coupling = catalog.factor(
        binding_coupling_factor_id,
        &format!("{label} binding coupling"),
    )?;
    if quantum_coupling != binding_coupling {
        return Err(invalid(format!(
            "{label} binding coupling does not match its quantum-flow coupling witness"
        )));
    }
    Ok(())
}

fn canonical_evaluator_parents(
    concrete_parent_ids: [u32; 2],
    canonical_input_order: &[u32],
    input_exchange_factor_id: u32,
    catalog: &TemplateCatalog<'_>,
    label: &str,
) -> RusticolResult<([u32; 2], ExactComplexRational)> {
    let mut ordered = match canonical_input_order {
        [0, 1] => concrete_parent_ids,
        [1, 0] => [concrete_parent_ids[1], concrete_parent_ids[0]],
        _ => {
            return Err(invalid(format!(
                "{label} canonical input order is not a binary permutation"
            )));
        }
    };
    let mut factor = ExactComplexRational::ONE;
    if input_exchange_factor_id != MISSING_U32 && ordered[1] < ordered[0] {
        ordered.swap(0, 1);
        factor = catalog.factor(input_exchange_factor_id, &format!("{label} input-exchange"))?;
    }
    Ok((ordered, factor))
}

fn multiply_factors(values: &[ExactComplexRational]) -> RusticolResult<ExactComplexRational> {
    values
        .iter()
        .copied()
        .try_fold(ExactComplexRational::ONE, ExactComplexRational::checked_mul)
}

fn aggregate_factor(
    target: &mut ExactComplexRational,
    value: ExactComplexRational,
) -> RusticolResult<()> {
    *target = target.checked_add(value)?;
    Ok(())
}

fn decode_strings<'a>(
    ranges: &[CheckedTableRange],
    bytes: &'a [u8],
    label: &str,
) -> RusticolResult<Vec<&'a str>> {
    ranges
        .iter()
        .copied()
        .enumerate()
        .map(|(index, range)| {
            let range = range.as_usize_range(bytes.len(), &format!("{label} {index}"))?;
            std::str::from_utf8(&bytes[range])
                .map_err(|error| invalid(format!("{label} {index} is not UTF-8: {error}")))
        })
        .collect()
}

fn required_string<'a>(strings: &[&'a str], id: u32, label: &str) -> RusticolResult<&'a str> {
    strings
        .get(id as usize)
        .copied()
        .ok_or_else(|| invalid(format!("{label} string {id} is absent")))
}

fn indexed_sequence<'a, T>(
    ranges: &[super::template::IndexedRangeRow],
    values: &'a [T],
    id: u32,
    label: &str,
) -> RusticolResult<&'a [T]> {
    let range = ranges
        .get(id as usize)
        .ok_or_else(|| invalid(format!("{label} sequence {id} is absent")))?
        .range
        .as_usize_range(values.len(), label)?;
    Ok(&values[range])
}

fn decode_template_factors(
    input: &OwnedRecurrenceTemplateInput,
    strings: &[&str],
) -> RusticolResult<Vec<ExactComplexRational>> {
    input
        .exact_factors
        .iter()
        .map(|row| {
            ExactComplexRational::parse_parts(
                required_string(strings, row.real_numerator_string_id, "factor numerator")?,
                required_string(
                    strings,
                    row.real_denominator_string_id,
                    "factor denominator",
                )?,
                required_string(strings, row.imag_numerator_string_id, "factor numerator")?,
                required_string(
                    strings,
                    row.imag_denominator_string_id,
                    "factor denominator",
                )?,
            )
        })
        .collect()
}

fn decode_process_factors(
    input: &OwnedRecurrenceProcessInput,
    strings: &[&str],
) -> RusticolResult<Vec<ExactComplexRational>> {
    input
        .exact_factors
        .iter()
        .map(|row| {
            ExactComplexRational::parse_parts(
                required_string(strings, row.real_numerator_string_id, "factor numerator")?,
                required_string(
                    strings,
                    row.real_denominator_string_id,
                    "factor denominator",
                )?,
                required_string(strings, row.imag_numerator_string_id, "factor numerator")?,
                required_string(
                    strings,
                    row.imag_denominator_string_id,
                    "factor denominator",
                )?,
            )
        })
        .collect()
}
