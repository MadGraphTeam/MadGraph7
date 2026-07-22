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
    LCColorComponentKind, LCColorComponentOperation, LCColorComponentRole, LCColorSourceSeed,
    LCColorSourceSeedOperation, LCColorTransitionWitness, LCColorWitnessTermId, MomentumTerm,
    RecurrenceClosureTerm, RecurrenceContribution, RecurrenceCurrent, RecurrenceFinalization,
    RecurrenceNodeKind, RecurrenceProgram, RecurrenceStrategy, SemanticDigest,
    SourceStateAssignment,
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
    closure_template_id: u32,
    quantum_flow_template_id: Option<u32>,
    parent_current_ids: Box<[u32]>,
}

struct TemplateCatalog<'a> {
    input: &'a OwnedRecurrenceTemplateInput,
    strings: Vec<&'a str>,
    digests: Vec<SemanticDigest>,
    factors: Vec<ExactComplexRational>,
    coupling_names: Vec<&'a str>,
}

impl<'a> TemplateCatalog<'a> {
    fn new(input: &'a OwnedRecurrenceTemplateInput) -> RusticolResult<Self> {
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

    fn factor(&self, id: u32, label: &str) -> RusticolResult<ExactComplexRational> {
        self.factors
            .get(id as usize)
            .copied()
            .ok_or_else(|| invalid(format!("{label} factor {id} is absent")))
    }

    fn u32_sequence(&self, id: u32, label: &str) -> RusticolResult<&'a [u32]> {
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
        LCColorTransitionWitness::new(
            permutation,
            row.reverse_parent_mask,
            LCColorComponentOperation::try_from(row.component_operation)?,
            kind,
            LCColorComponentRole::try_from(row.result_component_role)?,
            shape,
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
}

pub(super) fn build_recurrence_program(
    authenticated: AuthenticatedRecurrenceBuilderInput,
) -> RusticolResult<RecurrenceProgram> {
    let strategy = authenticated.process().summary().strategy();
    let catalog_digest = authenticated.template().summary().catalog_digest;
    let (process, template) = authenticated.into_parts();
    let process_input = process.input();
    let template_input = template.input();
    let process_catalog = ProcessCatalog::new(process_input)?;
    let template_catalog = TemplateCatalog::new(template_input)?;
    let coupling_limits = coupling_limits(&process_catalog, &template_catalog)?;
    let propagators = propagator_by_state(template_input)?;
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
    build_internal_currents(
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
        strategy,
        process_input,
        &process_catalog,
        template_input,
        &template_catalog,
        &color_states,
        &currents,
    )?;
    finish_program(
        strategy,
        color_states.into_states(),
        currents,
        closures,
        process_input.physical_lc_sectors.len(),
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
        for process_state in &process.source_states[range] {
            let source = *template
                .sources
                .get(process_state.source_template_id as usize)
                .ok_or_else(|| invalid("source template is absent"))?;
            if source.state_template_id != process_state.current_state_template_id {
                return Err(invalid(format!(
                    "source template {} targets current-state template {}, but the process source row targets {}",
                    process_state.source_template_id,
                    source.state_template_id,
                    process_state.current_state_template_id,
                )));
            }
            let dynamic_state = template_catalog
                .source_seed(source)?
                .instantiate(leg.source_slot)?;
            let color_id = color_states.intern(dynamic_state)?;
            let helicity_identity = match strategy {
                RecurrenceStrategy::TopologyReplay => CurrentHelicityIdentity::topology_replay(
                    source.spin_state,
                    vec![SourceStateAssignment::new(
                        leg.source_slot,
                        process_state.state_index,
                    )],
                )?,
                RecurrenceStrategy::AllFlowUnion => {
                    CurrentHelicityIdentity::all_flow_union(source.spin_state)
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
) -> RusticolResult<()> {
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
                for parent_ids in orders {
                    add_transition_contributions(
                        catalog_digest,
                        *transition,
                        parent_ids,
                        template,
                        catalog,
                        coupling_limits,
                        propagators,
                        color_states,
                        currents,
                        current_ids,
                        &mut currents_by_size[target_size - 1],
                    )?;
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn add_transition_contributions(
    catalog_digest: SemanticDigest,
    transition: TransitionRow,
    parent_ids: [u32; 2],
    template: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    coupling_limits: &[Option<u32>],
    propagators: &BTreeMap<u32, Option<u32>>,
    color_states: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
    target_bucket: &mut Vec<u32>,
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
    let Some(coupling_orders) = combined_coupling_orders(
        parents[0].coupling_orders(),
        parents[1].coupling_orders(),
        &catalog.coupling_orders(transition.coupling_order_set_id)?,
        coupling_limits,
    )?
    else {
        return Ok(());
    };
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
        let witness = catalog.witness(*witness_row)?;
        let Some(result_color) = witness.apply(left_color, right_color)? else {
            continue;
        };
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
        let key = CurrentCoreKey::new(
            catalog_digest,
            RecurrenceNodeKind::Current,
            transition.result_state_template_id,
            result_color_id,
            support,
            merged_momentum(parents[0].momentum(), parents[1].momentum())?,
            helicity_identity,
            catalog
                .flavour_flow(quantum.result_flavour_flow_id, "result flavour flow")?
                .to_vec(),
            quantum.result_quantum_number_flow_id,
            coupling_orders.clone(),
            CurrentSourceBinding::None,
            propagators
                .get(&transition.result_state_template_id)
                .copied()
                .flatten(),
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
    }
    Ok(())
}

fn build_closures(
    strategy: RecurrenceStrategy,
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    template: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    color_states: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
) -> RusticolResult<BTreeMap<PendingClosureKey, ExactComplexRational>> {
    let materialized_sectors = materialized_sector_ids(strategy, process);
    let full_support = (0..process.external_legs.len() as u32).collect::<Vec<_>>();
    let mut result = BTreeMap::new();
    for sector_id in materialized_sectors {
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
                        )?;
                    }
                }
            }
        }
        if !result.keys().any(|key| key.target_sector_id == sector_id) {
            let mut support_histogram = BTreeMap::<usize, usize>::new();
            for current in currents {
                *support_histogram
                    .entry(current.key.support_source_slots().len())
                    .or_default() += 1;
            }
            return Err(invalid(format!(
                "recurrence builder found no exact closure for physical LC sector {sector_id} \
                 (anchors={anchor_count}, complement_currents={complement_count}, \
                 state_matched_attempts={state_matched_attempts}, \
                 currents_by_support_size={support_histogram:?})"
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
            let mut closed = witness.closed_components(left, right)?;
            closed.sort_unstable();
            if closed != expected_sector_components(sector, process, process_catalog)? {
                continue;
            }
            let key = PendingClosureKey {
                target_sector_id: sector.sector_id,
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
    dynamic_color_states: Vec<DynamicLCColorState>,
    pending: Vec<PendingCurrent>,
    pending_closures: BTreeMap<PendingClosureKey, ExactComplexRational>,
    sector_count: usize,
) -> RusticolResult<RecurrenceProgram> {
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

    let mut closure_terms = Vec::new();
    let mut closure_ranges = Vec::with_capacity(sector_count);
    for sector_id in 0..sector_count as u32 {
        let start = closure_terms.len() as u64;
        for (key, factor) in pending_closures
            .iter()
            .filter(|(key, factor)| key.target_sector_id == sector_id && !factor.is_zero())
        {
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
                sector_id,
                key.closure_template_id,
                key.quantum_flow_template_id,
                parents,
                *factor,
            )?);
        }
        closure_ranges.push(CheckedTableRange::new(
            start,
            closure_terms.len() as u64 - start,
        ));
    }
    RecurrenceProgram::new(
        strategy,
        dynamic_color_states,
        currents,
        contributions,
        finalizations,
        closure_ranges,
        closure_terms,
    )
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
        if states[index] != parents[index].current_state_template_id()
            || spins[index] != parents[index].spin_state_class()
            || catalog.flavour_flow(flavours[index], "quantum parent flavour")?
                != parents[index].flavour_flow()
            || quantum_numbers[index] != parents[index].quantum_number_flow_id()
        {
            return Ok(false);
        }
    }
    Ok(true)
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
    result.sort_unstable();
    Ok(result)
}

fn materialized_sector_ids(
    strategy: RecurrenceStrategy,
    process: &OwnedRecurrenceProcessInput,
) -> BTreeSet<u32> {
    match strategy {
        RecurrenceStrategy::AllFlowUnion => process
            .physical_lc_sectors
            .iter()
            .map(|sector| sector.sector_id)
            .collect(),
        RecurrenceStrategy::TopologyReplay if !process.replay_partitions.is_empty() => process
            .replay_partitions
            .iter()
            .map(|partition| partition.materialized_sector_id)
            .collect(),
        RecurrenceStrategy::TopologyReplay => process
            .physical_lc_sectors
            .iter()
            .map(|sector| sector.sector_id)
            .collect(),
    }
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
