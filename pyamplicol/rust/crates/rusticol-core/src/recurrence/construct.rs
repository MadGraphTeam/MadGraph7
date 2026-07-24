// SPDX-License-Identifier: 0BSD

//! Compact model-generic recurrence construction.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::{Duration, Instant};

use super::layout::RuntimeSourceVariantBinding;
use super::process::{
    OwnedRecurrenceProcessInput, ProcessLCSectorKind, ProcessPhysicalLCSectorRow,
    ProcessSourceStateRow,
};
use super::template::{
    ClosureRow, LCColorTransitionWitnessRow, OutputFactorSource, OwnedRecurrenceTemplateInput,
    QuantumFlowRow, RuntimeHelicityContractRow, RuntimeHelicityVariantRow, SourceRow,
    TransitionRow,
};
use super::{
    AuthenticatedRecurrenceBuilderInput, CanonicalMomentumLinearForm, CheckedTableRange,
    ContributionKey, CurrentCoreKey, CurrentHelicityIdentity, CurrentSourceBinding,
    DynamicLCColorState, DynamicLCColorStateInterner, ExactComplexRational, ExactRational,
    LCColorComponent, LCColorComponentKind, LCColorComponentOperation, LCColorComponentRole,
    LCColorParentPort, LCColorPortWiring, LCColorSourceSeed, LCColorSourceSeedOperation,
    LCColorTransitionWitness, LCColorWitnessTermId, MomentumTerm, RecurrenceAmplitudeDestination,
    RecurrenceClosureTerm, RecurrenceContribution, RecurrenceCurrent, RecurrenceFinalization,
    RecurrenceNodeKind, RecurrenceProgram, RecurrenceReplayTarget, RecurrenceResolvedHelicity,
    RecurrenceStrategy, SemanticDigest, SourceStateAssignment,
};
use crate::{RusticolError, RusticolResult};

const MISSING_U32: u32 = u32::MAX;
const PROGRESS_PAIR_INTERVAL: usize = 16_384;
const PROGRESS_TIME_INTERVAL: Duration = Duration::from_millis(250);
const PURE_MASSLESS_ADJOINT_HELICITY_SUPPORT_ROLE: &str =
    "helicity-support:pure-massless-adjoint-tree-v1";

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

/// One rate-limited snapshot of compact recurrence construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceBuildProgress {
    pub phase: &'static str,
    pub phase_index: usize,
    pub phase_total: usize,
    pub stage_index: Option<usize>,
    pub stage_total: usize,
    pub subset_size: Option<usize>,
    pub candidate_parent_pair_count: usize,
    pub candidate_parent_pair_total: Option<usize>,
    pub current_count: usize,
    pub contribution_count: usize,
    pub dynamic_color_state_count: usize,
    pub color_target_prune_count: usize,
}

impl RecurrenceBuildProgress {
    #[allow(clippy::too_many_arguments)]
    fn snapshot(
        phase: &'static str,
        phase_index: usize,
        phase_total: usize,
        stage_index: Option<usize>,
        stage_total: usize,
        subset_size: Option<usize>,
        candidate_parent_pair_count: usize,
        candidate_parent_pair_total: Option<usize>,
        current_count: usize,
        contribution_count: usize,
        dynamic_color_state_count: usize,
        color_target_prune_count: usize,
    ) -> Self {
        Self {
            phase,
            phase_index,
            phase_total,
            stage_index,
            stage_total,
            subset_size,
            candidate_parent_pair_count,
            candidate_parent_pair_total,
            current_count,
            contribution_count,
            dynamic_color_state_count,
            color_target_prune_count,
        }
    }
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HelicitySupportRule {
    None,
    PureMasslessAdjointTree,
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
    candidate_parent_pair_count: usize,
    parent_pair_count: usize,
    transition_index_hit_count: usize,
    transition_candidate_count: usize,
    state_order_count: usize,
    quantum_match_count: usize,
    coupling_match_count: usize,
    color_shape_match_count: usize,
    color_result_count: usize,
    color_target_prune_count: usize,
    contribution_count: usize,
}

#[derive(Clone, Copy, Debug)]
struct IndexedTransition {
    row: TransitionRow,
    input_states: [u32; 2],
}

impl IndexedTransition {
    fn parent_ids(
        self,
        left_state: u32,
        right_state: u32,
        left_id: u32,
        right_id: u32,
    ) -> RusticolResult<[u32; 2]> {
        if self.input_states == [left_state, right_state] {
            return Ok([left_id, right_id]);
        }
        if left_state != right_state && self.input_states == [right_state, left_state] {
            return Ok([right_id, left_id]);
        }
        Err(invalid("recurrence transition state index is inconsistent"))
    }
}

#[derive(Debug, Default)]
struct TransitionStateIndex {
    rows_by_state_pair: BTreeMap<(u32, u32), Vec<IndexedTransition>>,
}

impl TransitionStateIndex {
    fn new(
        template: &OwnedRecurrenceTemplateInput,
        catalog: &TemplateCatalog<'_>,
    ) -> RusticolResult<Self> {
        let mut result = Self::default();
        for transition in &template.transitions {
            let input_states = catalog.u32_sequence(
                transition.input_state_sequence_id,
                "transition input states",
            )?;
            let input_states: [u32; 2] = input_states
                .try_into()
                .map_err(|_| invalid("recurrence v1 requires binary prepared transitions"))?;
            result.insert(*transition, input_states);
        }
        Ok(result)
    }

    fn insert(&mut self, row: TransitionRow, input_states: [u32; 2]) {
        self.rows_by_state_pair
            .entry(canonical_state_pair(input_states))
            .or_default()
            .push(IndexedTransition { row, input_states });
    }

    fn rows(&self, left_state: u32, right_state: u32) -> &[IndexedTransition] {
        self.rows_by_state_pair
            .get(&canonical_state_pair([left_state, right_state]))
            .map(Vec::as_slice)
            .unwrap_or_default()
    }
}

fn canonical_state_pair([left, right]: [u32; 2]) -> (u32, u32) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

/// Necessary physical-sector compatibility for a partial LC color forest.
///
/// Recurrence witnesses can join or explicitly reverse ordered components, but
/// they never split or internally permute an existing component. A partial
/// component must therefore occur as an oriented contiguous word in at least
/// one materialized representative sector. Reversed public flows are handled
/// by certified replay mappings; accepting their partial words here would
/// materialize an unphased second current instead of an exact replay alias.
/// This cheap forward filter prevents the builder from interning color words
/// that exact closure reachability would discard later; final backward
/// liveness remains authoritative.
#[derive(Debug)]
struct MaterializedColorTargets {
    sectors: Vec<Vec<LCColorComponent>>,
}

impl MaterializedColorTargets {
    fn new(
        materialized_sector_ids: &BTreeSet<u32>,
        process: &OwnedRecurrenceProcessInput,
        catalog: &ProcessCatalog<'_>,
    ) -> RusticolResult<Self> {
        let sectors = materialized_sector_ids
            .iter()
            .copied()
            .map(|sector_id| {
                let sector = process
                    .physical_lc_sectors
                    .get(sector_id as usize)
                    .copied()
                    .ok_or_else(|| invalid("materialized LC sector is absent"))?;
                expected_sector_components(sector, process, catalog)
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        if sectors.is_empty() {
            return Err(invalid(
                "recurrence construction has no materialized LC sector",
            ));
        }
        Ok(Self { sectors })
    }

    fn accepts(&self, state: &DynamicLCColorState) -> bool {
        if state.components().is_empty() {
            return true;
        }
        self.sectors.iter().any(|sector| {
            state.components().iter().all(|partial| {
                sector
                    .iter()
                    .any(|target| component_can_embed(partial, target))
            })
        })
    }
}

fn component_can_embed(partial: &LCColorComponent, target: &LCColorComponent) -> bool {
    if partial.kind() == LCColorComponentKind::Trace {
        return target.kind() == LCColorComponentKind::Trace
            && partial.source_slots().len() == target.source_slots().len()
            && cyclic_word_contains(target.source_slots(), partial.source_slots());
    }
    match target.kind() {
        LCColorComponentKind::Trace => {
            cyclic_word_contains(target.source_slots(), partial.source_slots())
        }
        LCColorComponentKind::OpenString | LCColorComponentKind::AdjointSegment => {
            linear_word_contains(target.source_slots(), partial.source_slots())
        }
    }
}

fn linear_word_contains(target: &[u32], partial: &[u32]) -> bool {
    !partial.is_empty()
        && partial.len() <= target.len()
        && target
            .windows(partial.len())
            .any(|window| window == partial)
}

fn cyclic_word_contains(target: &[u32], partial: &[u32]) -> bool {
    !partial.is_empty()
        && partial.len() <= target.len()
        && (0..target.len()).any(|start| {
            (0..partial.len())
                .all(|offset| target[(start + offset) % target.len()] == partial[offset])
        })
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
    build_recurrence_program_with_progress(authenticated, &mut |_| Ok(()))
}

pub(super) fn build_recurrence_program_with_progress(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    progress: &mut dyn FnMut(RecurrenceBuildProgress) -> RusticolResult<()>,
) -> RusticolResult<RecurrenceProgram> {
    let strategy = authenticated.process().summary().strategy();
    let catalog_digest = authenticated.template().summary().catalog_digest;
    let process_input = authenticated.process().input();
    let template_input = authenticated.template().input();
    let process_catalog = ProcessCatalog::new(process_input)?;
    let helicity_support_rule = helicity_support_rule(authenticated)?;
    let retained_helicity_count = retained_helicity_count(process_input)?;
    let template_catalog = TemplateCatalog::new(template_input)?;
    let coupling_limits = coupling_limits(&process_catalog, &template_catalog)?;
    let propagators = propagator_by_state(template_input)?;
    let replay_targets = build_replay_targets(strategy, process_input, &process_catalog)?;
    let materialized_sectors = materialized_sector_ids(strategy, process_input, &replay_targets);
    let color_targets =
        MaterializedColorTargets::new(&materialized_sectors, process_input, &process_catalog)?;
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
    let stage_total = process_input.external_legs.len().saturating_sub(2);
    let phase_total = stage_total.saturating_add(3);
    progress(RecurrenceBuildProgress::snapshot(
        "source construction",
        1,
        phase_total,
        None,
        stage_total,
        Some(1),
        0,
        None,
        currents.len(),
        0,
        color_states.len(),
        0,
    ))?;
    let stage_diagnostics = build_internal_currents(
        catalog_digest,
        process_input,
        template_input,
        &template_catalog,
        &coupling_limits,
        &propagators,
        &color_targets,
        &mut color_states,
        &mut currents,
        &mut current_ids,
        &mut currents_by_size,
        phase_total,
        progress,
    )?;
    let contribution_count = stage_diagnostics
        .iter()
        .map(|stage| stage.contribution_count)
        .try_fold(0usize, |total, count| {
            total
                .checked_add(count)
                .ok_or_else(|| invalid("recurrence contribution count exceeds usize"))
        })?;
    let color_target_prune_count = stage_diagnostics
        .iter()
        .map(|stage| stage.color_target_prune_count)
        .try_fold(0usize, |total, count| {
            total
                .checked_add(count)
                .ok_or_else(|| invalid("recurrence color-target prune count exceeds usize"))
        })?;
    progress(RecurrenceBuildProgress::snapshot(
        "amplitude closure",
        stage_total.saturating_add(2),
        phase_total,
        None,
        stage_total,
        None,
        0,
        None,
        currents.len(),
        contribution_count,
        color_states.len(),
        color_target_prune_count,
    ))?;
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
    progress(RecurrenceBuildProgress::snapshot(
        "schedule finalization",
        phase_total,
        phase_total,
        None,
        stage_total,
        None,
        0,
        None,
        currents.len(),
        contribution_count,
        color_states.len(),
        color_target_prune_count,
    ))?;
    finish_program(
        strategy,
        &process_catalog,
        color_states.into_states(),
        currents,
        closures,
        replay_targets,
        retained_helicity_count,
        helicity_support_rule,
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
        let retained_states = retained_state_indices
            .into_iter()
            .map(|state_index| {
                process
                    .source_states
                    .get(range.start + state_index as usize)
                    .ok_or_else(|| invalid("retained recurrence source state is absent"))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        for process_state in &retained_states {
            let source = *template
                .sources
                .get(process_state.source_template_id as usize)
                .ok_or_else(|| invalid("source template is absent"))?;
            validate_crossed_source_state(leg.is_initial != 0, process_state, source, template)?;
        }

        match strategy {
            RecurrenceStrategy::TopologyReplay => {
                for process_state in retained_states {
                    let source = *template
                        .sources
                        .get(process_state.source_template_id as usize)
                        .ok_or_else(|| invalid("source template is absent"))?;
                    let color_id = source_color_state_id(
                        leg.source_slot,
                        source,
                        template,
                        template_catalog,
                        color_states,
                    )?;
                    insert_source_current(
                        catalog_digest,
                        process_state.current_state_template_id,
                        color_id,
                        leg.source_slot,
                        process_state.momentum_sign,
                        CurrentHelicityIdentity::topology_replay(
                            process_state.spin_state,
                            vec![SourceStateAssignment::new(
                                leg.source_slot,
                                process_state.state_index,
                            )],
                        )?,
                        source,
                        CurrentSourceBinding::FixedTemplate(process_state.source_template_id),
                        Some(process_catalog.factor(
                            process_state.crossing_phase_factor_id,
                            "source crossing phase",
                        )?),
                        &zero_orders,
                        template_catalog,
                        currents,
                        current_ids,
                        currents_by_size,
                    )?;
                }
            }
            RecurrenceStrategy::AllFlowUnion => {
                let groups = union_source_dispatch_groups(
                    &retained_states,
                    process_catalog,
                    template,
                    template_catalog,
                )?;
                for group in groups {
                    let source = group.representative_source;
                    let color_id = source_color_state_id(
                        leg.source_slot,
                        source,
                        template,
                        template_catalog,
                        color_states,
                    )?;
                    insert_source_current(
                        catalog_digest,
                        group.full_state_template_id,
                        color_id,
                        leg.source_slot,
                        group.momentum_sign,
                        CurrentHelicityIdentity::all_flow_union(group.full_spin_state_class),
                        source,
                        CurrentSourceBinding::runtime_dispatch_with_variants(
                            group.contract_id,
                            group.variants,
                        )?,
                        None,
                        &zero_orders,
                        template_catalog,
                        currents,
                        current_ids,
                        currents_by_size,
                    )?;
                }
            }
        }
    }
    Ok(())
}

fn source_color_state_id(
    source_slot: u32,
    source: SourceRow,
    template: &OwnedRecurrenceTemplateInput,
    template_catalog: &TemplateCatalog<'_>,
    color_states: &mut DynamicLCColorStateInterner,
) -> RusticolResult<super::DynamicLCColorStateId> {
    let dynamic_state = template_catalog.source_seed(source)?.instantiate(
        source_slot,
        template
            .current_states
            .get(source.state_template_id as usize)
            .ok_or_else(|| invalid("source current-state template is absent"))?
            .color_representation,
    )?;
    color_states.intern(dynamic_state)
}

#[derive(Debug)]
struct UnionSourceDispatchGroup {
    contract_id: u32,
    full_state_template_id: u32,
    full_spin_state_class: i32,
    momentum_sign: i32,
    representative_source: SourceRow,
    variants: Vec<RuntimeSourceVariantBinding>,
}

fn union_source_dispatch_groups(
    retained_states: &[&ProcessSourceStateRow],
    process_catalog: &ProcessCatalog<'_>,
    template: &OwnedRecurrenceTemplateInput,
    template_catalog: &TemplateCatalog<'_>,
) -> RusticolResult<Vec<UnionSourceDispatchGroup>> {
    let mut grouped = BTreeMap::<
        u32,
        (
            RuntimeHelicityContractRow,
            Vec<(ProcessSourceStateRow, RuntimeHelicityVariantRow)>,
        ),
    >::new();
    for process_state in retained_states {
        let (contract, variant) =
            runtime_helicity_variant_for_source(template, process_state.source_template_id)?;
        if variant.source_state_template_id
            != template.sources[process_state.source_template_id as usize].state_template_id
        {
            return Err(invalid(format!(
                "runtime-helicity variant {} source-state contract is stale",
                variant.id
            )));
        }
        grouped
            .entry(contract.id)
            .or_insert_with(|| (contract, Vec::new()))
            .1
            .push((**process_state, variant));
    }
    if grouped.is_empty() {
        return Err(invalid(
            "all-flow-union source has no certified runtime-helicity dispatch domain",
        ));
    }

    grouped
        .into_values()
        .map(|(contract, mut entries)| {
            entries.sort_by_key(|(state, _)| state.state_index);
            if entries
                .windows(2)
                .any(|pair| pair[0].0.state_index == pair[1].0.state_index)
            {
                return Err(invalid(format!(
                    "runtime-helicity contract {} has an ambiguous process source-state mapping",
                    contract.id
                )));
            }
            let full_state = template
                .current_states
                .get(contract.full_state_template_id as usize)
                .ok_or_else(|| invalid("runtime-helicity full state is absent"))?;
            let representative_state = entries[0].0;
            let representative_source =
                template.sources[representative_state.source_template_id as usize];
            let momentum_sign = representative_state.momentum_sign;
            for (process_state, variant) in &entries {
                let source = template.sources[process_state.source_template_id as usize];
                validate_union_source_family(
                    representative_source,
                    source,
                    contract,
                    *variant,
                    *process_state,
                    full_state,
                    template,
                    template_catalog,
                )?;
                if process_state.momentum_sign != momentum_sign {
                    return Err(invalid(format!(
                        "runtime-helicity contract {} mixes source momentum signs",
                        contract.id
                    )));
                }
            }
            let full_spin_state_class =
                union_full_spin_state_class(contract, &entries, template, template_catalog)?;
            let variants = entries
                .into_iter()
                .map(|(process_state, variant)| {
                    RuntimeSourceVariantBinding::new(
                        process_state.state_index,
                        process_state.public_helicity,
                        variant.id,
                        variant.source_template_id,
                        variant.source_state_template_id,
                        process_state.current_state_template_id,
                        process_state.spin_state,
                        process_catalog.factor(
                            process_state.crossing_phase_factor_id,
                            "runtime source crossing phase",
                        )?,
                    )
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            Ok(UnionSourceDispatchGroup {
                contract_id: contract.id,
                full_state_template_id: contract.full_state_template_id,
                full_spin_state_class,
                momentum_sign,
                representative_source,
                variants,
            })
        })
        .collect()
}

fn runtime_helicity_variant_for_source(
    template: &OwnedRecurrenceTemplateInput,
    source_template_id: u32,
) -> RusticolResult<(RuntimeHelicityContractRow, RuntimeHelicityVariantRow)> {
    let mut found = None;
    for contract in template.runtime_helicity_contracts.iter().copied() {
        let range = contract.variant_range.as_usize_range(
            template.runtime_helicity_variants.len(),
            "runtime-helicity variants",
        )?;
        for variant in template.runtime_helicity_variants[range].iter().copied() {
            if variant.source_template_id != source_template_id {
                continue;
            }
            if found.replace((contract, variant)).is_some() {
                return Err(invalid(format!(
                    "source template {source_template_id} belongs to ambiguous runtime-helicity variants"
                )));
            }
        }
    }
    found.ok_or_else(|| {
        invalid(format!(
            "source template {source_template_id} has no certified runtime-helicity variant"
        ))
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_union_source_family(
    representative: SourceRow,
    source: SourceRow,
    contract: RuntimeHelicityContractRow,
    variant: RuntimeHelicityVariantRow,
    process_state: ProcessSourceStateRow,
    full_state: &super::template::CurrentStateRow,
    template: &OwnedRecurrenceTemplateInput,
    template_catalog: &TemplateCatalog<'_>,
) -> RusticolResult<()> {
    if variant.contract_id != contract.id
        || variant.source_template_id != source.id
        || variant.source_state_template_id != source.state_template_id
    {
        return Err(invalid(format!(
            "runtime-helicity variant {} is inconsistent with source template {}",
            variant.id, source.id
        )));
    }
    let effective = template
        .current_states
        .get(process_state.current_state_template_id as usize)
        .ok_or_else(|| invalid("runtime source effective state is absent"))?;
    let full_state_is_compatible = full_state.particle_id == effective.particle_id
        && full_state.anti_particle_id == effective.anti_particle_id
        && full_state.species_string_id == effective.species_string_id
        && full_state.orientation == effective.orientation
        && full_state.statistics == effective.statistics
        && full_state.color_representation == effective.color_representation
        && full_state.lc_color_shape_string_id == effective.lc_color_shape_string_id
        && full_state.auxiliary_kind_string_id == effective.auxiliary_kind_string_id
        && full_state.mass_parameter_id == effective.mass_parameter_id
        && full_state.width_parameter_id == effective.width_parameter_id;
    if !full_state_is_compatible {
        return Err(invalid(format!(
            "runtime-helicity full state {} is not crossing-compatible with process state {}",
            contract.full_state_template_id, process_state.current_state_template_id
        )));
    }
    if !same_union_source_dispatch_semantics(representative, source)
        || template_catalog.flavour_flow(
            representative.flavour_flow_id,
            "runtime source flavour flow",
        )? != template_catalog
            .flavour_flow(source.flavour_flow_id, "runtime source flavour flow")?
    {
        return Err(invalid(format!(
            "runtime-helicity contract {} mixes incompatible source semantics",
            contract.id
        )));
    }
    Ok(())
}

fn same_union_source_dispatch_semantics(left: SourceRow, right: SourceRow) -> bool {
    left.crossing_string_id == right.crossing_string_id
        && left.wavefunction_family_string_id == right.wavefunction_family_string_id
        && left.flavour_flow_id == right.flavour_flow_id
        && left.quantum_number_flow_id == right.quantum_number_flow_id
        && left.lc_color_seed_operation == right.lc_color_seed_operation
        && left.lc_color_seed_shape_string_id == right.lc_color_seed_shape_string_id
        && left.lc_color_seed_component_kind == right.lc_color_seed_component_kind
        && left.lc_color_seed_component_role == right.lc_color_seed_component_role
        && left.lc_color_seed_provenance_sequence_id == right.lc_color_seed_provenance_sequence_id
        && left.mass_parameter_id == right.mass_parameter_id
        && left.width_parameter_id == right.width_parameter_id
}

fn union_full_spin_state_class(
    contract: RuntimeHelicityContractRow,
    entries: &[(ProcessSourceStateRow, RuntimeHelicityVariantRow)],
    template: &OwnedRecurrenceTemplateInput,
    _catalog: &TemplateCatalog<'_>,
) -> RusticolResult<i32> {
    let mut result_spins = BTreeSet::new();
    for flow in &template.quantum_flows {
        if flow.result_state_template_id == contract.full_state_template_id {
            result_spins.insert(flow.result_spin_state);
        }
    }
    if result_spins.is_empty() {
        result_spins.extend(entries.iter().map(|(state, _)| state.spin_state));
    }
    if result_spins.len() != 1 {
        return Err(invalid(format!(
            "runtime-helicity contract {} has ambiguous full-state spin classes {:?}",
            contract.id, result_spins
        )));
    }
    Ok(*result_spins.iter().next().expect("checked nonempty"))
}

#[allow(clippy::too_many_arguments)]
fn insert_source_current(
    catalog_digest: SemanticDigest,
    current_state_template_id: u32,
    color_id: super::DynamicLCColorStateId,
    source_slot: u32,
    momentum_sign: i32,
    helicity_identity: CurrentHelicityIdentity,
    source: SourceRow,
    source_binding: CurrentSourceBinding,
    source_factor: Option<ExactComplexRational>,
    zero_orders: &[u32],
    template_catalog: &TemplateCatalog<'_>,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
    currents_by_size: &mut [Vec<u32>],
) -> RusticolResult<()> {
    let key = CurrentCoreKey::new(
        catalog_digest,
        RecurrenceNodeKind::Source,
        current_state_template_id,
        color_id,
        vec![source_slot],
        CanonicalMomentumLinearForm::new(vec![MomentumTerm {
            source_slot,
            coefficient: momentum_sign,
        }])?,
        helicity_identity,
        template_catalog
            .flavour_flow(source.flavour_flow_id, "source flavour flow")?
            .to_vec(),
        source.quantum_number_flow_id,
        zero_orders.to_vec(),
        source_binding,
        None,
    )?;
    if let Some(existing) = current_ids.get(&key).copied() {
        if currents[existing as usize].source_exact_factor != source_factor {
            return Err(invalid(
                "equivalent source currents have different exact factors",
            ));
        }
        return Ok(());
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

fn parent_pairs_for_target(
    target_size: usize,
    prior_currents_by_size: &[Vec<u32>],
) -> impl Iterator<Item = [u32; 2]> + '_ {
    (1..=target_size / 2).flat_map(move |left_size| {
        let right_size = target_size - left_size;
        let same_size = left_size == right_size;
        prior_currents_by_size[left_size - 1]
            .iter()
            .copied()
            .flat_map(move |left_id| {
                prior_currents_by_size[right_size - 1]
                    .iter()
                    .copied()
                    .filter_map(move |right_id| match left_id.cmp(&right_id) {
                        std::cmp::Ordering::Less => Some([left_id, right_id]),
                        std::cmp::Ordering::Equal => None,
                        std::cmp::Ordering::Greater if same_size => None,
                        std::cmp::Ordering::Greater => Some([right_id, left_id]),
                    })
            })
    })
}

fn parent_pair_total_for_target(
    target_size: usize,
    prior_currents_by_size: &[Vec<u32>],
) -> RusticolResult<usize> {
    (1..=target_size / 2).try_fold(0usize, |total, left_size| {
        let right_size = target_size - left_size;
        let left_count = prior_currents_by_size[left_size - 1].len();
        let right_count = prior_currents_by_size[right_size - 1].len();
        let pair_count = if left_size == right_size {
            left_count
                .checked_mul(left_count.saturating_sub(1))
                .and_then(|value| value.checked_div(2))
        } else {
            left_count.checked_mul(right_count)
        }
        .ok_or_else(|| invalid("recurrence parent-pair total exceeds usize"))?;
        total
            .checked_add(pair_count)
            .ok_or_else(|| invalid("recurrence parent-pair total exceeds usize"))
    })
}

fn checked_diagnostic_add(counter: &mut usize, amount: usize, label: &str) -> RusticolResult<()> {
    *counter = counter
        .checked_add(amount)
        .ok_or_else(|| invalid(format!("{label} exceeds usize")))?;
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
    color_targets: &MaterializedColorTargets,
    color_states: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
    currents_by_size: &mut [Vec<u32>],
    phase_total: usize,
    progress: &mut dyn FnMut(RecurrenceBuildProgress) -> RusticolResult<()>,
) -> RusticolResult<Vec<StageConstructionDiagnostics>> {
    let transition_index = TransitionStateIndex::new(template, catalog)?;
    let mut diagnostics = Vec::new();
    let stage_total = process.external_legs.len().saturating_sub(2);
    let mut completed_contribution_count = 0usize;
    let mut completed_color_target_prune_count = 0usize;
    for target_size in 2..process.external_legs.len() {
        let mut stage = StageConstructionDiagnostics {
            target_size,
            ..StageConstructionDiagnostics::default()
        };
        let (prior_buckets, target_and_later) = currents_by_size.split_at_mut(target_size - 1);
        debug_assert!(prior_buckets.iter().flatten().copied().is_sorted());
        let target_bucket = &mut target_and_later[0];
        let candidate_parent_pair_total = parent_pair_total_for_target(target_size, prior_buckets)?;
        let stage_index = target_size - 1;
        progress(RecurrenceBuildProgress::snapshot(
            "recurrence stage",
            stage_index.saturating_add(1),
            phase_total,
            Some(stage_index),
            stage_total,
            Some(target_size),
            0,
            Some(candidate_parent_pair_total),
            currents.len(),
            completed_contribution_count,
            color_states.len(),
            completed_color_target_prune_count,
        ))?;
        let mut last_progress = Instant::now();
        for [left_id, right_id] in parent_pairs_for_target(target_size, prior_buckets) {
            checked_diagnostic_add(
                &mut stage.candidate_parent_pair_count,
                1,
                "recurrence candidate parent-pair count",
            )?;
            if stage.candidate_parent_pair_count % PROGRESS_PAIR_INTERVAL == 0
                && last_progress.elapsed() >= PROGRESS_TIME_INTERVAL
            {
                progress(RecurrenceBuildProgress::snapshot(
                    "recurrence stage",
                    stage_index.saturating_add(1),
                    phase_total,
                    Some(stage_index),
                    stage_total,
                    Some(target_size),
                    stage.candidate_parent_pair_count,
                    Some(candidate_parent_pair_total),
                    currents.len(),
                    completed_contribution_count.saturating_add(stage.contribution_count),
                    color_states.len(),
                    completed_color_target_prune_count
                        .saturating_add(stage.color_target_prune_count),
                ))?;
                last_progress = Instant::now();
            }
            if !disjoint_support(
                currents[left_id as usize].key.support_source_slots(),
                currents[right_id as usize].key.support_source_slots(),
            ) {
                continue;
            }
            checked_diagnostic_add(
                &mut stage.parent_pair_count,
                1,
                "recurrence parent-pair count",
            )?;
            let left_state = currents[left_id as usize].key.current_state_template_id();
            let right_state = currents[right_id as usize].key.current_state_template_id();
            let indexed_transitions = transition_index.rows(left_state, right_state);
            if !indexed_transitions.is_empty() {
                checked_diagnostic_add(
                    &mut stage.transition_index_hit_count,
                    1,
                    "recurrence transition-index hit count",
                )?;
            }
            checked_diagnostic_add(
                &mut stage.transition_candidate_count,
                indexed_transitions.len(),
                "recurrence transition-candidate count",
            )?;
            for indexed in indexed_transitions {
                checked_diagnostic_add(
                    &mut stage.state_order_count,
                    1,
                    "recurrence state-order count",
                )?;
                add_transition_contributions(
                    catalog_digest,
                    indexed.row,
                    indexed.parent_ids(left_state, right_state, left_id, right_id)?,
                    target_size + 1 < process.external_legs.len(),
                    template,
                    catalog,
                    coupling_limits,
                    propagators,
                    color_targets,
                    color_states,
                    currents,
                    current_ids,
                    target_bucket,
                    &mut stage,
                )?;
            }
        }
        debug_assert_eq!(stage.target_size, target_size);
        debug_assert_eq!(stage.transition_candidate_count, stage.state_order_count);
        completed_contribution_count = completed_contribution_count
            .checked_add(stage.contribution_count)
            .ok_or_else(|| invalid("recurrence contribution count exceeds usize"))?;
        completed_color_target_prune_count = completed_color_target_prune_count
            .checked_add(stage.color_target_prune_count)
            .ok_or_else(|| invalid("recurrence color-target prune count exceeds usize"))?;
        progress(RecurrenceBuildProgress::snapshot(
            "recurrence stage",
            stage_index.saturating_add(1),
            phase_total,
            Some(stage_index),
            stage_total,
            Some(target_size),
            stage.candidate_parent_pair_count,
            Some(candidate_parent_pair_total),
            currents.len(),
            completed_contribution_count,
            color_states.len(),
            completed_color_target_prune_count,
        ))?;
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
    color_targets: &MaterializedColorTargets,
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
    checked_diagnostic_add(
        &mut diagnostics.quantum_match_count,
        1,
        "recurrence quantum-match count",
    )?;
    let Some(coupling_orders) = combined_coupling_orders(
        parents[0].coupling_orders(),
        parents[1].coupling_orders(),
        &catalog.coupling_orders(transition.coupling_order_set_id)?,
        coupling_limits,
    )?
    else {
        return Ok(());
    };
    checked_diagnostic_add(
        &mut diagnostics.coupling_match_count,
        1,
        "recurrence coupling-match count",
    )?;
    let contraction = *template
        .color_contractions
        .get(transition.color_contraction_template_id as usize)
        .ok_or_else(|| invalid("transition color contraction is absent"))?;
    let binding_coupling = authenticate_runtime_coupling(
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
        output_factor_from_binding(
            binding_coupling,
            transition.output_factor_source,
            "transition",
        )?,
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
        checked_diagnostic_add(
            &mut diagnostics.color_shape_match_count,
            1,
            "recurrence color-shape-match count",
        )?;
        let witness = catalog.witness(*witness_row)?;
        let Some(result_color) = witness.apply(left_color, right_color)? else {
            continue;
        };
        checked_diagnostic_add(
            &mut diagnostics.color_result_count,
            1,
            "recurrence color-result count",
        )?;
        if !color_targets.accepts(&result_color) {
            checked_diagnostic_add(
                &mut diagnostics.color_target_prune_count,
                1,
                "recurrence color-target prune count",
            )?;
            continue;
        }
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
        checked_diagnostic_add(
            &mut diagnostics.contribution_count,
            1,
            "recurrence contribution-attempt count",
        )?;
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
        let binding_coupling = if let Some(quantum) = quantum {
            authenticate_runtime_coupling(
                catalog,
                quantum,
                closure.binding_coupling_factor_id,
                "closure",
            )?
        } else {
            catalog.factor(
                closure.binding_coupling_factor_id,
                "closure binding coupling",
            )?
        };
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
            output_factor_from_binding(binding_coupling, closure.output_factor_source, "closure")?,
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
    mut pending_closures: BTreeMap<PendingClosureKey, ExactComplexRational>,
    replay_targets: Vec<RecurrenceReplayTarget>,
    retained_helicity_count: u64,
    helicity_support_rule: HelicitySupportRule,
) -> RusticolResult<RecurrenceProgram> {
    if helicity_support_rule != HelicitySupportRule::None {
        let mut retained = BTreeMap::new();
        for (key, factor) in pending_closures {
            if closure_helicity_is_supported(
                helicity_support_rule,
                process_catalog,
                &key.complete_source_states,
            )? {
                retained.insert(key, factor);
            }
        }
        pending_closures = retained;
    }
    let sector_count = match strategy {
        RecurrenceStrategy::TopologyReplay => process_catalog
            .input
            .physical_lc_sectors
            .len()
            .max(replay_targets.len()),
        RecurrenceStrategy::AllFlowUnion => process_catalog.input.physical_lc_sectors.len(),
    };
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

fn helicity_support_rule(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
) -> RusticolResult<HelicitySupportRule> {
    let extensions = authenticated
        .process()
        .semantic_identity()
        .extension_digests();
    let mut rule = HelicitySupportRule::None;
    for role in extensions
        .keys()
        .filter(|role| role.starts_with("helicity-support:"))
    {
        let candidate = match role.as_str() {
            PURE_MASSLESS_ADJOINT_HELICITY_SUPPORT_ROLE => {
                HelicitySupportRule::PureMasslessAdjointTree
            }
            _ => {
                return Err(invalid(format!(
                    "unsupported recurrence helicity-support proof {role:?}"
                )));
            }
        };
        if rule != HelicitySupportRule::None {
            return Err(invalid(
                "recurrence process carries more than one helicity-support proof",
            ));
        }
        rule = candidate;
    }
    Ok(rule)
}

fn closure_helicity_is_supported(
    rule: HelicitySupportRule,
    process_catalog: &ProcessCatalog<'_>,
    source_states: &[SourceStateAssignment],
) -> RusticolResult<bool> {
    if rule == HelicitySupportRule::None || source_states.is_empty() {
        return Ok(true);
    }
    let public_helicities = process_catalog.public_helicities(source_states)?;
    let mut positive = 0usize;
    let mut negative = 0usize;
    for (leg, helicity) in process_catalog
        .input
        .external_legs
        .iter()
        .zip(public_helicities)
    {
        let physical = if leg.is_initial == 1 {
            -helicity
        } else {
            helicity
        };
        match physical {
            1 => positive += 1,
            -1 => negative += 1,
            value => {
                return Err(invalid(format!(
                    "pure-massless-adjoint helicity proof received unsupported helicity {value}"
                )));
            }
        }
    }
    Ok(positive >= 2 && negative >= 2)
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
    let retained_flows = retained_public_flows(process)?;
    let retained_sectors = retained_flows
        .iter()
        .map(|flow| flow.construction_sector_id)
        .collect::<BTreeSet<_>>();
    let mut base_by_sector = BTreeMap::<u32, (u32, Vec<u32>, ExactComplexRational)>::new();
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
            let previous = base_by_sector.insert(
                target.sector_id,
                (
                    partition.materialized_sector_id,
                    catalog
                        .u32_sequence(
                            target.source_slot_permutation_sequence_id,
                            "recurrence replay source permutation",
                        )?
                        .to_vec(),
                    factor,
                ),
            );
            if previous.is_some() {
                return Err(invalid(format!(
                    "construction sector {} has multiple recurrence replay targets",
                    target.sector_id
                )));
            }
        }
    }
    let identity = (0..process.external_legs.len())
        .map(|slot| u32::try_from(slot).map_err(|_| invalid("source-slot count exceeds u32")))
        .collect::<RusticolResult<Vec<_>>>()?;
    for sector_id in retained_sectors {
        base_by_sector
            .entry(sector_id)
            .or_insert_with(|| (sector_id, identity.clone(), ExactComplexRational::ONE));
    }
    retained_flows
        .into_iter()
        .enumerate()
        .map(|(target_sector_id, flow)| {
            let (materialized, construction_permutation, factor) = base_by_sector
                .get(&flow.construction_sector_id)
                .ok_or_else(|| invalid("public flow construction replay target is absent"))?;
            let public_permutation = catalog.u32_sequence(
                flow.source_slot_permutation_sequence_id,
                "public LC flow source permutation",
            )?;
            if construction_permutation.len() != public_permutation.len() {
                return Err(invalid(
                    "public LC flow and construction replay permutations have different sizes",
                ));
            }
            let permutation =
                compose_gather_permutations(construction_permutation, public_permutation)?;
            RecurrenceReplayTarget::new(
                u32::try_from(target_sector_id)
                    .map_err(|_| invalid("replay-target count exceeds u32"))?,
                *materialized,
                u32::try_from(target_sector_id)
                    .map_err(|_| invalid("public-flow target ID exceeds u32"))?,
                permutation,
                *factor,
            )
        })
        .collect()
}

fn compose_gather_permutations(
    representative_to_construction: &[u32],
    construction_to_public: &[u32],
) -> RusticolResult<Vec<u32>> {
    if representative_to_construction.len() != construction_to_public.len() {
        return Err(invalid(
            "public LC flow and construction replay permutations have different sizes",
        ));
    }
    representative_to_construction
        .iter()
        .map(|construction_slot| {
            construction_to_public
                .get(*construction_slot as usize)
                .copied()
                .ok_or_else(|| invalid("construction replay permutation is out of range"))
        })
        .collect()
}

fn retained_public_flows(
    process: &OwnedRecurrenceProcessInput,
) -> RusticolResult<Vec<&super::process::ProcessPublicLCFlowRow>> {
    if !process.header[0].selected_flow_mode()? {
        return Ok(process.public_lc_flows.iter().collect());
    }
    let selected = process
        .selected_public_flow_coverage
        .iter()
        .map(|row| row.flow_id)
        .collect::<BTreeSet<_>>();
    Ok(process
        .public_lc_flows
        .iter()
        .filter(|flow| selected.contains(&flow.flow_id))
        .collect())
}

fn propagator_by_state(
    template: &OwnedRecurrenceTemplateInput,
) -> RusticolResult<BTreeMap<u32, Option<u32>>> {
    let mut result = BTreeMap::new();
    for row in &template.propagators {
        let value = (row.applies_propagator != 0).then_some(row.id);
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
) -> RusticolResult<ExactComplexRational> {
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
    Ok(binding_coupling)
}

fn output_factor_from_binding(
    binding_coupling: ExactComplexRational,
    output_factor_source: u8,
    label: &str,
) -> RusticolResult<ExactComplexRational> {
    let component = match OutputFactorSource::try_from(output_factor_source)? {
        OutputFactorSource::None => return Ok(ExactComplexRational::ONE),
        OutputFactorSource::CouplingReal => binding_coupling.real(),
        OutputFactorSource::CouplingImag => binding_coupling.imag(),
    };
    if component.is_zero() {
        return Err(invalid(format!(
            "{label} selects a zero binding-coupling component"
        )));
    }
    Ok(ExactComplexRational::new(component, ExactRational::ZERO))
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

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use super::super::process::{ProcessExternalLegRow, ProcessPhysicalLCSectorRow};
    use super::super::template::{
        ColorContractionRow, DigestCatalogRow, ExactFactorRow, IndexedRangeRow,
        LCColorTransitionWitnessRow, PropagatorRow, QuantumFlowRow,
    };
    use super::*;

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).expect("test digest must be nonzero")
    }

    fn transition(id: u32) -> TransitionRow {
        TransitionRow {
            id,
            template_string_id: 0,
            input_state_sequence_id: 0,
            result_state_template_id: 0,
            quantum_flow_template_id: 0,
            evaluator_binding_id: 0,
            canonical_input_order_sequence_id: 0,
            momentum_convention_sequence_id: 0,
            coupling_parameter_sequence_id: 0,
            coupling_order_set_id: 0,
            color_contraction_template_id: 0,
            binding_coupling_factor_id: 0,
            exact_factor_id: 0,
            output_factor_source: 0,
            equivalence_class_string_id: 0,
            input_exchange_factor_id: MISSING_U32,
            output_projection_string_id: 0,
            semantic_digest_id: 0,
        }
    }

    fn source(id: u32) -> SourceRow {
        SourceRow {
            id,
            template_string_id: 1,
            state_template_id: id,
            crossing_string_id: 2,
            wavefunction_family_string_id: 3,
            helicity: if id == 0 { -1 } else { 1 },
            spin_state: if id == 0 { -1 } else { 1 },
            flavour_flow_id: 4,
            quantum_number_flow_id: 5,
            lc_color_seed_operation: 6,
            lc_color_seed_shape_string_id: 7,
            lc_color_seed_component_kind: 8,
            lc_color_seed_component_role: 9,
            lc_color_seed_proof_digest_id: 10 + id,
            lc_color_seed_provenance_sequence_id: 11,
            wavefunction_expression_digest_id: 12 + id,
            evaluator_binding_id: 13 + id,
            mass_parameter_id: 14,
            width_parameter_id: 15,
            semantic_digest_id: 16 + id,
        }
    }

    #[test]
    fn union_source_dispatch_ignores_proof_instance_identity_only() {
        let left = source(0);
        let right = source(1);
        assert_ne!(
            left.lc_color_seed_proof_digest_id,
            right.lc_color_seed_proof_digest_id
        );
        assert!(same_union_source_dispatch_semantics(left, right));

        let mut incompatible = right;
        incompatible.crossing_string_id += 1;
        assert!(!same_union_source_dispatch_semantics(left, incompatible));

        let mut incompatible = right;
        incompatible.lc_color_seed_provenance_sequence_id += 1;
        assert!(!same_union_source_dispatch_semantics(left, incompatible));
    }

    fn buffered_parent_pairs(target_size: usize, currents_by_size: &[Vec<u32>]) -> Vec<[u32; 2]> {
        let mut pairs = Vec::new();
        for left_size in 1..target_size {
            let right_size = target_size - left_size;
            for &left_id in &currents_by_size[left_size - 1] {
                for &right_id in &currents_by_size[right_size - 1] {
                    if left_id < right_id {
                        pairs.push([left_id, right_id]);
                    }
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        pairs
    }

    #[test]
    fn streamed_parent_pairs_match_buffered_reference_order() {
        let fixtures = [
            (2, vec![vec![0, 1, 2, 3]]),
            (3, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7, 8, 9]]),
            (
                4,
                vec![
                    vec![0, 1, 2, 3],
                    vec![4, 5, 6, 7, 8, 9],
                    vec![10, 11, 12, 13],
                ],
            ),
        ];
        for (target_size, currents_by_size) in fixtures {
            assert_eq!(
                parent_pairs_for_target(target_size, &currents_by_size).collect::<Vec<_>>(),
                buffered_parent_pairs(target_size, &currents_by_size),
                "support size {target_size}",
            );
        }
    }

    #[test]
    fn streamed_parent_pair_totals_match_iteration() {
        for (target_size, currents_by_size) in [
            (2, vec![vec![0, 1, 2, 3]]),
            (3, vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7, 8, 9]]),
            (
                4,
                vec![
                    vec![0, 1, 2, 3],
                    vec![4, 5, 6, 7, 8, 9],
                    vec![10, 11, 12, 13],
                ],
            ),
        ] {
            assert_eq!(
                parent_pair_total_for_target(target_size, &currents_by_size).unwrap(),
                parent_pairs_for_target(target_size, &currents_by_size).count(),
                "support size {target_size}",
            );
        }
    }

    #[test]
    fn state_index_preserves_interleaved_reference_transition_order() {
        let rows = [
            (transition(0), [1, 2]),
            (transition(1), [2, 1]),
            (transition(2), [1, 2]),
            (transition(3), [3, 4]),
            (transition(4), [2, 1]),
        ];
        let mut index = TransitionStateIndex::default();
        for (row, input_states) in rows {
            index.insert(row, input_states);
        }

        let actual = index
            .rows(1, 2)
            .iter()
            .map(|indexed| Ok((indexed.row.id, indexed.parent_ids(1, 2, 10, 20)?)))
            .collect::<RusticolResult<Vec<_>>>()
            .unwrap();
        let reference = rows
            .iter()
            .flat_map(|(row, input_states)| {
                let mut applications = Vec::new();
                if *input_states == [1, 2] {
                    applications.push((row.id, [10, 20]));
                }
                if *input_states == [2, 1] {
                    applications.push((row.id, [20, 10]));
                }
                applications
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, reference);
        assert_eq!(
            actual,
            [(0, [10, 20]), (1, [20, 10]), (2, [10, 20]), (4, [20, 10])]
        );
    }

    fn encoded_strings(values: &[&str]) -> (Vec<CheckedTableRange>, Vec<u8>) {
        let mut ranges = Vec::new();
        let mut bytes = Vec::new();
        for value in values {
            ranges.push(CheckedTableRange::new(
                bytes.len() as u64,
                value.len() as u64,
            ));
            bytes.extend_from_slice(value.as_bytes());
        }
        (ranges, bytes)
    }

    fn indexed_u32_sequences(sequences: &[&[u32]]) -> (Vec<IndexedRangeRow>, Vec<u32>) {
        let mut ranges = Vec::new();
        let mut values = Vec::new();
        for (id, sequence) in sequences.iter().enumerate() {
            ranges.push(IndexedRangeRow {
                id: id as u32,
                range: CheckedTableRange::new(values.len() as u64, sequence.len() as u64),
            });
            values.extend_from_slice(sequence);
        }
        (ranges, values)
    }

    fn scalar_reference_template() -> OwnedRecurrenceTemplateInput {
        const EMPTY: u32 = 0;
        const STATE_PAIR: u32 = 1;
        const CANONICAL_ORDER: u32 = 2;
        const PARENT_FLAVOURS: u32 = 3;
        const PARENT_QUANTUM_NUMBERS: u32 = 4;

        let (string_ranges, string_bytes) = encoded_strings(&["0", "1", "constant-result"]);
        let (u32_sequence_ranges, u32_sequence_values) =
            indexed_u32_sequences(&[&[], &[0, 0], &[0, 1], &[0, 0], &[0, 0]]);
        OwnedRecurrenceTemplateInput {
            input_abi: "scalar-reference-template-v1".to_owned(),
            catalog_digest: digest(1),
            compiled_model_digest: digest(2),
            prepared_kernel_pack_digest: digest(3),
            catalog_header: vec![],
            coupling_order_ranges: vec![IndexedRangeRow {
                id: 0,
                range: CheckedTableRange::new(0, 0),
            }],
            coupling_order_terms: vec![],
            current_states: vec![],
            digest_catalog: vec![DigestCatalogRow {
                id: 0,
                value: [4; 32],
            }],
            evaluator_bindings: vec![],
            exact_factors: vec![ExactFactorRow {
                id: 0,
                real_numerator_string_id: 1,
                real_denominator_string_id: 1,
                imag_numerator_string_id: 0,
                imag_denominator_string_id: 1,
            }],
            flavour_flow_ranges: vec![IndexedRangeRow {
                id: 0,
                range: CheckedTableRange::new(0, 1),
            }],
            flavour_flow_values: vec![0],
            i32_sequence_ranges: vec![IndexedRangeRow {
                id: 0,
                range: CheckedTableRange::new(0, 2),
            }],
            i32_sequence_values: vec![0, 0],
            parameters: vec![],
            propagators: vec![],
            quantum_flows: vec![QuantumFlowRow {
                id: 0,
                template_string_id: 0,
                input_state_sequence_id: STATE_PAIR,
                input_spin_sequence_id: 0,
                input_flavour_sequence_id: PARENT_FLAVOURS,
                input_quantum_sequence_id: PARENT_QUANTUM_NUMBERS,
                flavour_flow_operation_string_id: 2,
                quantum_number_flow_operation_string_id: 2,
                coupling_order_set_id: 0,
                result_state_template_id: 0,
                result_spin_state: 0,
                result_flavour_flow_id: 0,
                result_quantum_number_flow_id: 0,
                exact_coupling_factor_id: 0,
                predicate_digest_id: 0,
                semantic_digest_id: 0,
            }],
            quantum_number_flow_ranges: vec![],
            quantum_number_flow_terms: vec![],
            runtime_helicity_contracts: vec![],
            runtime_helicity_variants: vec![],
            runtime_helicity_embeddings: vec![],
            runtime_helicity_projections: vec![],
            sources: vec![],
            string_ranges,
            string_bytes,
            symmetry_proofs: vec![],
            transitions: vec![TransitionRow {
                input_state_sequence_id: STATE_PAIR,
                canonical_input_order_sequence_id: CANONICAL_ORDER,
                input_exchange_factor_id: MISSING_U32,
                ..transition(0)
            }],
            closures: vec![ClosureRow {
                id: 0,
                template_string_id: 0,
                input_state_sequence_id: STATE_PAIR,
                result_state_template_id: MISSING_U32,
                evaluator_binding_id: 0,
                canonical_input_order_sequence_id: CANONICAL_ORDER,
                coupling_parameter_sequence_id: EMPTY,
                coupling_order_set_id: 0,
                eligible_quantum_flow_sequence_id: EMPTY,
                color_contraction_template_id: 1,
                binding_coupling_factor_id: 0,
                exact_factor_id: 0,
                output_factor_source: 0,
                equivalence_class_string_id: 0,
                input_exchange_factor_id: MISSING_U32,
                projection_string_id: 0,
                component_coefficient_sequence_id: EMPTY,
                chirality_relation_string_id: 0,
                metric_signature_string_id: 0,
                semantic_digest_id: 0,
            }],
            color_contractions: vec![
                ColorContractionRow {
                    id: 0,
                    template_string_id: 0,
                    rule_kind_string_id: 0,
                    input_representation_sequence_id: EMPTY,
                    has_output_representation: 1,
                    output_representation: 1,
                    ordered_open_string_arity: 0,
                    exact_coefficient_factor_id: 0,
                    witness_start: 0,
                    witness_count: 1,
                    nc_term_start: 0,
                    nc_term_count: 0,
                    expression_digest_id: 0,
                    semantic_digest_id: 0,
                },
                ColorContractionRow {
                    id: 1,
                    template_string_id: 0,
                    rule_kind_string_id: 0,
                    input_representation_sequence_id: EMPTY,
                    has_output_representation: 0,
                    output_representation: 0,
                    ordered_open_string_arity: 0,
                    exact_coefficient_factor_id: 0,
                    witness_start: 1,
                    witness_count: 1,
                    nc_term_start: 0,
                    nc_term_count: 0,
                    expression_digest_id: 0,
                    semantic_digest_id: 0,
                },
            ],
            lc_color_transition_witnesses: vec![
                LCColorTransitionWitnessRow {
                    color_contraction_id: 0,
                    ordinal: 0,
                    left_shape_string_id: 0,
                    right_shape_string_id: 0,
                    input_permutation: 0,
                    reverse_parent_mask: 0,
                    component_operation: LCColorComponentOperation::Empty as u8,
                    result_component_kind: u8::MAX,
                    result_component_role: LCColorComponentRole::None as u8,
                    result_shape_string_id: 0,
                    exact_factor_id: 0,
                    proof_digest_id: 0,
                    input_port_pairing_sequence_id: EMPTY,
                    result_port_binding_sequence_id: EMPTY,
                    provenance_sequence_id: EMPTY,
                },
                LCColorTransitionWitnessRow {
                    color_contraction_id: 1,
                    ordinal: 0,
                    left_shape_string_id: 0,
                    right_shape_string_id: 0,
                    input_permutation: 0,
                    reverse_parent_mask: 0,
                    component_operation: LCColorComponentOperation::Close as u8,
                    result_component_kind: u8::MAX,
                    result_component_role: LCColorComponentRole::None as u8,
                    result_shape_string_id: MISSING_U32,
                    exact_factor_id: 0,
                    proof_digest_id: 0,
                    input_port_pairing_sequence_id: EMPTY,
                    result_port_binding_sequence_id: EMPTY,
                    provenance_sequence_id: EMPTY,
                },
            ],
            color_nc_terms: vec![],
            u32_sequence_ranges,
            u32_sequence_values,
        }
    }

    fn scalar_reference_process(external_count: usize) -> OwnedRecurrenceProcessInput {
        OwnedRecurrenceProcessInput {
            input_abi: "scalar-reference-process-v1".to_owned(),
            declared_input_digest: digest(5),
            fermion_pairing: None,
            bitset_ranges: vec![],
            bitset_words: vec![],
            coupling_limits: vec![],
            digest_catalog: vec![],
            exact_factors: vec![],
            external_legs: (0..external_count)
                .map(|slot| ProcessExternalLegRow {
                    source_slot: slot as u32,
                    public_label: slot as u32 + 1,
                    physical_pdg: 0,
                    outgoing_pdg: 0,
                    is_initial: 0,
                    is_fermionic: 0,
                    source_state_range: CheckedTableRange::new(0, 0),
                    momentum_mask_id: 0,
                    support_mask_id: 0,
                })
                .collect(),
            header: vec![],
            header_digests: vec![],
            lc_open_strings: vec![],
            normalization: vec![],
            parameter_projection: vec![],
            physical_lc_sectors: vec![ProcessPhysicalLCSectorRow {
                sector_id: 0,
                public_id_string_id: 0,
                kind: ProcessLCSectorKind::Singlet as u8,
                closure_source_slot: 0,
                closure_proof_algorithm_string_id: 0,
                closure_proof_digest_id: 0,
                open_string_range: CheckedTableRange::new(0, 0),
                trace_sequence_id: 0,
                singlet_sequence_id: 0,
                word_sequence_id: 0,
                support_mask_id: 0,
            }],
            public_lc_flows: vec![],
            replay_partitions: vec![],
            replay_targets: vec![],
            selected_public_flow_coverage: vec![],
            selected_source_coverage: vec![],
            semantic_template_references: vec![],
            source_states: vec![],
            string_ranges: vec![],
            string_bytes: vec![],
            u32_sequence_ranges: vec![CheckedTableRange::new(0, 0)],
            u32_sequence_values: vec![],
        }
    }

    fn scalar_reference_program(
        external_count: usize,
    ) -> (
        RecurrenceProgram,
        Vec<StageConstructionDiagnostics>,
        (usize, usize, usize),
    ) {
        let template = scalar_reference_template();
        let template_catalog = TemplateCatalog::new(&template).unwrap();
        let process = scalar_reference_process(external_count);
        let process_catalog = ProcessCatalog::new(&process).unwrap();
        let color_targets =
            MaterializedColorTargets::new(&BTreeSet::from([0]), &process, &process_catalog)
                .unwrap();
        let mut color_states = DynamicLCColorStateInterner::default();
        let color_id = color_states
            .intern(DynamicLCColorState::new(0, None, vec![]).unwrap())
            .unwrap();
        let mut currents = Vec::new();
        let mut current_ids = BTreeMap::new();
        let mut currents_by_size = vec![Vec::new(); external_count];
        for slot in 0..external_count as u32 {
            let key = CurrentCoreKey::new(
                template.catalog_digest,
                RecurrenceNodeKind::Source,
                0,
                color_id,
                vec![slot],
                CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                    source_slot: slot,
                    coefficient: 1,
                }])
                .unwrap(),
                CurrentHelicityIdentity::all_flow_union(0),
                vec![0],
                0,
                vec![],
                CurrentSourceBinding::runtime_dispatch(slot, vec![slot]).unwrap(),
                None,
            )
            .unwrap();
            let id = currents.len() as u32;
            assert!(current_ids.insert(key.clone(), id).is_none());
            currents.push(PendingCurrent {
                key,
                source_exact_factor: None,
                contributions: BTreeMap::new(),
            });
            currents_by_size[0].push(id);
        }

        let stage_diagnostics = build_internal_currents(
            template.catalog_digest,
            &process,
            &template,
            &template_catalog,
            &[],
            &BTreeMap::new(),
            &color_targets,
            &mut color_states,
            &mut currents,
            &mut current_ids,
            &mut currents_by_size,
            external_count.saturating_add(1),
            &mut |_| Ok(()),
        )
        .unwrap();
        let closures = build_closures(
            &process,
            &process_catalog,
            &template,
            &template_catalog,
            &color_states,
            &currents,
            &BTreeSet::from([0]),
            &stage_diagnostics,
        )
        .unwrap();
        let constructed_counts = (
            currents.len(),
            currents
                .iter()
                .map(|current| current.contributions.len())
                .sum(),
            closures.len(),
        );
        let program = finish_program(
            RecurrenceStrategy::AllFlowUnion,
            &process_catalog,
            color_states.into_states(),
            currents,
            closures,
            vec![],
            1,
            HelicitySupportRule::None,
        )
        .unwrap();
        (program, stage_diagnostics, constructed_counts)
    }

    #[test]
    fn materialized_color_targets_keep_only_embeddable_ordered_words() {
        let target_open =
            LCColorComponent::new(LCColorComponentKind::OpenString, vec![0, 2, 3, 4, 1]).unwrap();
        let targets = MaterializedColorTargets {
            sectors: vec![vec![target_open]],
        };
        let state = |slots: Vec<u32>| {
            DynamicLCColorState::new(
                0,
                Some(0),
                vec![LCColorComponent::new(LCColorComponentKind::AdjointSegment, slots).unwrap()],
            )
            .unwrap()
        };

        assert!(targets.accepts(&state(vec![2, 3, 4])));
        assert!(!targets.accepts(&state(vec![4, 3, 2])));
        assert!(!targets.accepts(&state(vec![2, 4])));
        assert!(!targets.accepts(&state(vec![0, 3])));
        assert!(targets.accepts(&DynamicLCColorState::new(0, None, vec![]).unwrap()));
    }

    #[test]
    fn materialized_color_targets_accept_cyclic_trace_segments() {
        let target_trace =
            LCColorComponent::new(LCColorComponentKind::Trace, vec![1, 2, 3, 4]).unwrap();
        let targets = MaterializedColorTargets {
            sectors: vec![vec![target_trace]],
        };
        let state = |kind, slots: Vec<u32>| {
            DynamicLCColorState::new(
                0,
                (kind != LCColorComponentKind::Trace).then_some(0),
                vec![LCColorComponent::new(kind, slots).unwrap()],
            )
            .unwrap()
        };

        assert!(targets.accepts(&state(LCColorComponentKind::AdjointSegment, vec![4, 1, 2],)));
        assert!(!targets.accepts(&state(LCColorComponentKind::AdjointSegment, vec![2, 1, 4],)));
        assert!(!targets.accepts(&state(LCColorComponentKind::AdjointSegment, vec![1, 3],)));
        assert!(!targets.accepts(&state(LCColorComponentKind::Trace, vec![4, 3, 2, 1],)));
        assert!(!targets.accepts(&state(LCColorComponentKind::Trace, vec![1, 2, 3],)));
    }

    #[test]
    fn scalar_reference_fixtures_keep_structural_schedule_counts() {
        // Frozen from the buffered parent-pair and full transition-scan constructor.
        for (name, external_count, expected_constructed, expected_schedule) in [
            ("three-point", 3, (6, 3, 1), (4, 1, 1)),
            ("four-point", 4, (14, 18, 1), (8, 6, 1)),
        ] {
            let (program, _, constructed_counts) = scalar_reference_program(external_count);
            assert_eq!(
                constructed_counts, expected_constructed,
                "{name} scalar construction fixture",
            );
            assert_eq!(
                (
                    program.currents().len(),
                    program.contributions().len(),
                    program.closure_terms().len(),
                ),
                expected_schedule,
                "{name} scalar reference fixture",
            );
        }
    }

    #[test]
    fn scalar_reference_fixture_reports_streaming_selectivity() {
        let (_, diagnostics, _) = scalar_reference_program(4);
        assert_eq!(diagnostics.len(), 2);
        assert_eq!(
            diagnostics
                .iter()
                .map(|stage| (
                    stage.target_size,
                    stage.candidate_parent_pair_count,
                    stage.parent_pair_count,
                    stage.transition_index_hit_count,
                    stage.transition_candidate_count,
                    stage.contribution_count,
                ))
                .collect::<Vec<_>>(),
            [(2, 6, 6, 6, 6, 6), (3, 24, 12, 12, 12, 12)],
        );
    }

    #[test]
    fn non_propagating_states_use_identity_finalization() {
        fn propagator(id: u32, state: u32, applies: bool) -> PropagatorRow {
            PropagatorRow {
                id,
                template_string_id: 0,
                state_template_id: state,
                applies_propagator: u8::from(applies),
                evaluator_binding_id: if applies { 0 } else { MISSING_U32 },
                numerator_expression_digest_id: 0,
                denominator_expression_digest_id: 0,
                mass_parameter_id: MISSING_U32,
                width_parameter_id: MISSING_U32,
                gauge_string_id: 0,
                linearity_proof_template_id: MISSING_U32,
                semantic_digest_id: 0,
            }
        }

        let mut template = scalar_reference_template();
        template.propagators = vec![propagator(0, 4, false), propagator(1, 7, true)];
        assert_eq!(
            propagator_by_state(&template).unwrap(),
            BTreeMap::from([(4, None), (7, Some(1))]),
        );
    }

    #[test]
    fn output_factor_uses_the_declared_binding_coupling_component() {
        let coupling = ExactComplexRational::new(
            ExactRational::new(2, 3).unwrap(),
            ExactRational::new(-5, 7).unwrap(),
        );
        assert_eq!(
            output_factor_from_binding(
                coupling,
                OutputFactorSource::None as u8,
                "test transition",
            )
            .unwrap(),
            ExactComplexRational::ONE,
        );
        assert_eq!(
            output_factor_from_binding(
                coupling,
                OutputFactorSource::CouplingReal as u8,
                "test transition",
            )
            .unwrap(),
            ExactComplexRational::new(ExactRational::new(2, 3).unwrap(), ExactRational::ZERO,),
        );
        assert_eq!(
            output_factor_from_binding(
                coupling,
                OutputFactorSource::CouplingImag as u8,
                "test transition",
            )
            .unwrap(),
            ExactComplexRational::new(ExactRational::new(-5, 7).unwrap(), ExactRational::ZERO,),
        );
    }

    #[test]
    fn composes_construction_and_public_gather_permutations() {
        let representative_to_construction = [0, 2, 3, 1];
        let construction_to_public = [0, 3, 1, 2];
        assert_eq!(
            compose_gather_permutations(&representative_to_construction, &construction_to_public,)
                .unwrap(),
            [0, 1, 2, 3]
        );
    }
}
