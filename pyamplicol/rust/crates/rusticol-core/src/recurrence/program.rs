// SPDX-License-Identifier: 0BSD

//! Compact, immutable recurrence schedule produced by the recurrence builder.
//!
//! The owned vectors are build-time storage. Evaluation consumes only packed
//! slices and ranges, so traversing a validated program requires no maps or
//! heap allocation.

use std::collections::BTreeSet;

use super::{
    CheckedTableRange, ContributionKey, CurrentCoreKey, DynamicLCColorState, ExactComplexRational,
    RecurrenceNodeKind, RecurrenceStrategy, SourceStateAssignment,
};
use crate::{RusticolError, RusticolResult};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

fn checked_table_len(label: &str, length: usize) -> RusticolResult<u64> {
    u64::try_from(length).map_err(|_| invalid(format!("{label} length {length} exceeds u64")))
}

/// One source or propagated current in topological execution order.
///
/// For a source, the key's dynamic color-state ID refers to the materialized
/// result of a compiler-owned source-color seed. This schedule never infers a
/// seed from particle identity, representation, or source position.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceCurrent {
    id: u32,
    key: CurrentCoreKey,
    source_exact_factor: Option<ExactComplexRational>,
    contribution_range: CheckedTableRange,
    finalization_id: Option<u32>,
}

impl RecurrenceCurrent {
    pub fn new(
        id: u32,
        key: CurrentCoreKey,
        source_exact_factor: Option<ExactComplexRational>,
        contribution_range: CheckedTableRange,
        finalization_id: Option<u32>,
    ) -> RusticolResult<Self> {
        contribution_range.end("recurrence current contribution")?;
        if source_exact_factor.is_some_and(ExactComplexRational::is_zero) {
            return Err(invalid("recurrence source factor must not be zero"));
        }
        Ok(Self {
            id,
            key,
            source_exact_factor,
            contribution_range,
            finalization_id,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub const fn key(&self) -> &CurrentCoreKey {
        &self.key
    }

    pub const fn contribution_range(&self) -> CheckedTableRange {
        self.contribution_range
    }

    pub const fn source_exact_factor(&self) -> Option<ExactComplexRational> {
        self.source_exact_factor
    }

    pub const fn fan_in(&self) -> u64 {
        self.contribution_range.count
    }

    pub const fn finalization_id(&self) -> Option<u32> {
        self.finalization_id
    }

    pub const fn is_source(&self) -> bool {
        matches!(self.key.node_kind(), RecurrenceNodeKind::Source)
    }
}

/// One exact contribution accumulated into one result current.
///
/// The key's color-witness term owns the compiler-certified result-component
/// role (active, passive, or absent). The program only stores its materialized
/// result current and does not derive that role from parent ordering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceContribution {
    id: u32,
    result_current_id: u32,
    parent_current_ids: Box<[u32]>,
    key: ContributionKey,
    exact_factor: ExactComplexRational,
}

impl RecurrenceContribution {
    pub fn new(
        id: u32,
        result_current_id: u32,
        parent_current_ids: Vec<u32>,
        key: ContributionKey,
        exact_factor: ExactComplexRational,
    ) -> RusticolResult<Self> {
        if parent_current_ids.is_empty() {
            return Err(invalid(
                "recurrence contribution requires at least one parent current",
            ));
        }
        u32::try_from(parent_current_ids.len())
            .map_err(|_| invalid("recurrence contribution parent count exceeds u32"))?;
        if exact_factor.is_zero() {
            return Err(invalid("recurrence contribution factor must not be zero"));
        }
        Ok(Self {
            id,
            result_current_id,
            parent_current_ids: parent_current_ids.into_boxed_slice(),
            key,
            exact_factor,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub const fn result_current_id(&self) -> u32 {
        self.result_current_id
    }

    pub fn parent_current_ids(&self) -> &[u32] {
        &self.parent_current_ids
    }

    pub const fn key(&self) -> &ContributionKey {
        &self.key
    }

    pub const fn exact_factor(&self) -> ExactComplexRational {
        self.exact_factor
    }
}

/// Exactly one propagation/finalization operation for one non-source current.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceFinalization {
    id: u32,
    current_id: u32,
    propagator_template_id: Option<u32>,
    exact_factor: ExactComplexRational,
}

/// One complete physical helicity retained by a topology-replay schedule.
///
/// Currents retain only their local source-state ancestry. This catalog names
/// the complete ancestry reached by closure terms so amplitudes from distinct
/// helicities are never added coherently before squaring.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceResolvedHelicity {
    id: u32,
    source_states: Box<[SourceStateAssignment]>,
    public_helicities: Box<[i32]>,
}

/// One live amplitude destination and its packed closure-term range.
///
/// Destinations are sparse: topology replay stores only live combinations of
/// a materialized representative sector and a resolved helicity, never their
/// Cartesian product.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecurrenceAmplitudeDestination {
    id: u32,
    target_sector_id: u32,
    target_helicity_id: Option<u32>,
    closure_range: CheckedTableRange,
}

/// One exact topology-replay route from a materialized recurrence to a public sector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceReplayTarget {
    id: u32,
    materialized_sector_id: u32,
    target_sector_id: u32,
    source_slot_permutation: Box<[u32]>,
    amplitude_factor: ExactComplexRational,
}

impl RecurrenceReplayTarget {
    pub fn new(
        id: u32,
        materialized_sector_id: u32,
        target_sector_id: u32,
        source_slot_permutation: Vec<u32>,
        amplitude_factor: ExactComplexRational,
    ) -> RusticolResult<Self> {
        if source_slot_permutation.is_empty() {
            return Err(invalid(
                "recurrence replay target requires a source permutation",
            ));
        }
        if amplitude_factor.is_zero() {
            return Err(invalid("recurrence replay target factor must not be zero"));
        }
        Ok(Self {
            id,
            materialized_sector_id,
            target_sector_id,
            source_slot_permutation: source_slot_permutation.into_boxed_slice(),
            amplitude_factor,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub const fn materialized_sector_id(&self) -> u32 {
        self.materialized_sector_id
    }

    pub const fn target_sector_id(&self) -> u32 {
        self.target_sector_id
    }

    pub fn source_slot_permutation(&self) -> &[u32] {
        &self.source_slot_permutation
    }

    pub const fn amplitude_factor(&self) -> ExactComplexRational {
        self.amplitude_factor
    }
}

impl RecurrenceAmplitudeDestination {
    pub fn new(
        id: u32,
        target_sector_id: u32,
        target_helicity_id: Option<u32>,
        closure_range: CheckedTableRange,
    ) -> RusticolResult<Self> {
        closure_range.end("recurrence amplitude destination")?;
        if closure_range.count == 0 {
            return Err(invalid(
                "recurrence amplitude destination requires closure terms",
            ));
        }
        Ok(Self {
            id,
            target_sector_id,
            target_helicity_id,
            closure_range,
        })
    }

    pub const fn id(self) -> u32 {
        self.id
    }

    pub const fn target_sector_id(self) -> u32 {
        self.target_sector_id
    }

    pub const fn target_helicity_id(self) -> Option<u32> {
        self.target_helicity_id
    }

    pub const fn closure_range(self) -> CheckedTableRange {
        self.closure_range
    }
}

impl RecurrenceResolvedHelicity {
    pub fn new(
        id: u32,
        source_states: Vec<SourceStateAssignment>,
        public_helicities: Vec<i32>,
    ) -> RusticolResult<Self> {
        if source_states.is_empty() || source_states.len() != public_helicities.len() {
            return Err(invalid(
                "resolved helicity requires one public value per source-state assignment",
            ));
        }
        for (source_slot, assignment) in source_states.iter().copied().enumerate() {
            if assignment.source_slot() as usize != source_slot {
                return Err(invalid(
                    "resolved-helicity source states must cover every source slot in order",
                ));
            }
        }
        Ok(Self {
            id,
            source_states: source_states.into_boxed_slice(),
            public_helicities: public_helicities.into_boxed_slice(),
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub fn source_states(&self) -> &[SourceStateAssignment] {
        &self.source_states
    }

    pub fn public_helicities(&self) -> &[i32] {
        &self.public_helicities
    }
}

impl RecurrenceFinalization {
    pub fn new(
        id: u32,
        current_id: u32,
        propagator_template_id: Option<u32>,
        exact_factor: ExactComplexRational,
    ) -> RusticolResult<Self> {
        if exact_factor.is_zero() {
            return Err(invalid("recurrence finalization factor must not be zero"));
        }
        Ok(Self {
            id,
            current_id,
            propagator_template_id,
            exact_factor,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub const fn current_id(&self) -> u32 {
        self.current_id
    }

    pub const fn propagator_template_id(&self) -> Option<u32> {
        self.propagator_template_id
    }

    pub const fn exact_factor(&self) -> ExactComplexRational {
        self.exact_factor
    }
}

/// One exact signed term contributing to one physical LC target sector.
///
/// The closure template owns the result-component kind. `target_sector_id` is
/// only the reduction destination selected after validating the physical
/// sector's compiler-owned closure anchor; it is not itself an anchor or an
/// instruction for reconstructing one.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceClosureTerm {
    id: u32,
    target_destination_id: u32,
    closure_template_id: u32,
    quantum_flow_template_id: Option<u32>,
    parent_current_ids: Box<[u32]>,
    exact_factor: ExactComplexRational,
}

impl RecurrenceClosureTerm {
    pub fn new(
        id: u32,
        target_destination_id: u32,
        closure_template_id: u32,
        quantum_flow_template_id: Option<u32>,
        parent_current_ids: Vec<u32>,
        exact_factor: ExactComplexRational,
    ) -> RusticolResult<Self> {
        if parent_current_ids.is_empty() {
            return Err(invalid(
                "recurrence closure term requires at least one parent current",
            ));
        }
        u32::try_from(parent_current_ids.len())
            .map_err(|_| invalid("recurrence closure parent count exceeds u32"))?;
        if exact_factor.is_zero() {
            return Err(invalid("recurrence closure term factor must not be zero"));
        }
        Ok(Self {
            id,
            target_destination_id,
            closure_template_id,
            quantum_flow_template_id,
            parent_current_ids: parent_current_ids.into_boxed_slice(),
            exact_factor,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub const fn target_destination_id(&self) -> u32 {
        self.target_destination_id
    }

    pub const fn closure_template_id(&self) -> u32 {
        self.closure_template_id
    }

    /// Prepared quantum-flow witness for a kernel closure.
    ///
    /// Direct Rusticol closure templates have no vertex coupling and therefore
    /// carry `None`.
    pub const fn quantum_flow_template_id(&self) -> Option<u32> {
        self.quantum_flow_template_id
    }

    pub fn parent_current_ids(&self) -> &[u32] {
        &self.parent_current_ids
    }

    pub const fn exact_factor(&self) -> ExactComplexRational {
        self.exact_factor
    }
}

/// Validated compact recurrence program ready for serialization or execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceProgram {
    strategy: RecurrenceStrategy,
    physical_sector_count: u32,
    retained_helicity_count: u64,
    dynamic_color_states: Box<[DynamicLCColorState]>,
    currents: Box<[RecurrenceCurrent]>,
    contributions: Box<[RecurrenceContribution]>,
    finalizations: Box<[RecurrenceFinalization]>,
    replay_targets: Box<[RecurrenceReplayTarget]>,
    resolved_helicities: Box<[RecurrenceResolvedHelicity]>,
    amplitude_destinations: Box<[RecurrenceAmplitudeDestination]>,
    closure_terms: Box<[RecurrenceClosureTerm]>,
}

impl RecurrenceProgram {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        strategy: RecurrenceStrategy,
        physical_sector_count: u32,
        retained_helicity_count: u64,
        dynamic_color_states: Vec<DynamicLCColorState>,
        currents: Vec<RecurrenceCurrent>,
        contributions: Vec<RecurrenceContribution>,
        finalizations: Vec<RecurrenceFinalization>,
        replay_targets: Vec<RecurrenceReplayTarget>,
        resolved_helicities: Vec<RecurrenceResolvedHelicity>,
        amplitude_destinations: Vec<RecurrenceAmplitudeDestination>,
        closure_terms: Vec<RecurrenceClosureTerm>,
    ) -> RusticolResult<Self> {
        let program = Self {
            strategy,
            physical_sector_count,
            retained_helicity_count,
            dynamic_color_states: dynamic_color_states.into_boxed_slice(),
            currents: currents.into_boxed_slice(),
            contributions: contributions.into_boxed_slice(),
            finalizations: finalizations.into_boxed_slice(),
            replay_targets: replay_targets.into_boxed_slice(),
            resolved_helicities: resolved_helicities.into_boxed_slice(),
            amplitude_destinations: amplitude_destinations.into_boxed_slice(),
            closure_terms: closure_terms.into_boxed_slice(),
        };
        program.validate()?;
        Ok(program)
    }

    pub const fn strategy(&self) -> RecurrenceStrategy {
        self.strategy
    }

    pub const fn physical_sector_count(&self) -> u32 {
        self.physical_sector_count
    }

    pub const fn retained_helicity_count(&self) -> u64 {
        self.retained_helicity_count
    }

    pub fn dynamic_color_states(&self) -> &[DynamicLCColorState] {
        &self.dynamic_color_states
    }

    pub fn currents(&self) -> &[RecurrenceCurrent] {
        &self.currents
    }

    pub fn contributions(&self) -> &[RecurrenceContribution] {
        &self.contributions
    }

    pub fn finalizations(&self) -> &[RecurrenceFinalization] {
        &self.finalizations
    }

    pub fn replay_targets(&self) -> &[RecurrenceReplayTarget] {
        &self.replay_targets
    }

    pub fn resolved_helicities(&self) -> &[RecurrenceResolvedHelicity] {
        &self.resolved_helicities
    }

    pub fn amplitude_destinations(&self) -> &[RecurrenceAmplitudeDestination] {
        &self.amplitude_destinations
    }

    pub fn closure_terms(&self) -> &[RecurrenceClosureTerm] {
        &self.closure_terms
    }

    pub fn current_range(&self) -> CheckedTableRange {
        CheckedTableRange::new(0, self.currents.len() as u64)
    }

    pub fn contribution_range(&self) -> CheckedTableRange {
        CheckedTableRange::new(0, self.contributions.len() as u64)
    }

    pub fn closure_term_range(&self) -> CheckedTableRange {
        CheckedTableRange::new(0, self.closure_terms.len() as u64)
    }

    pub fn closure_range_for_destination(&self, destination_id: u32) -> Option<CheckedTableRange> {
        self.amplitude_destinations
            .get(destination_id as usize)
            .map(|destination| destination.closure_range())
    }

    pub fn validate(&self) -> RusticolResult<()> {
        let current_count = checked_table_len("recurrence current", self.currents.len())?;
        let color_state_count = checked_table_len(
            "recurrence dynamic LC color state",
            self.dynamic_color_states.len(),
        )?;
        let contribution_count =
            checked_table_len("recurrence contribution", self.contributions.len())?;
        let finalization_count =
            checked_table_len("recurrence finalization", self.finalizations.len())?;
        let replay_target_count =
            checked_table_len("recurrence replay target", self.replay_targets.len())?;
        let closure_count = checked_table_len("recurrence closure term", self.closure_terms.len())?;
        let helicity_count = checked_table_len(
            "recurrence resolved helicity",
            self.resolved_helicities.len(),
        )?;
        let destination_count = checked_table_len(
            "recurrence amplitude destination",
            self.amplitude_destinations.len(),
        )?;

        if self.physical_sector_count == 0 {
            return Err(invalid("recurrence requires physical LC sectors"));
        }
        if self.retained_helicity_count == 0 {
            return Err(invalid("recurrence retains no public helicities"));
        }

        u32::try_from(current_count)
            .map_err(|_| invalid("recurrence current count exceeds the u32 ID domain"))?;
        u32::try_from(color_state_count)
            .map_err(|_| invalid("recurrence dynamic color-state count exceeds u32"))?;
        u32::try_from(contribution_count)
            .map_err(|_| invalid("recurrence contribution count exceeds the u32 ID domain"))?;
        u32::try_from(finalization_count)
            .map_err(|_| invalid("recurrence finalization count exceeds the u32 ID domain"))?;
        u32::try_from(replay_target_count)
            .map_err(|_| invalid("recurrence replay-target count exceeds the u32 ID domain"))?;
        u32::try_from(closure_count)
            .map_err(|_| invalid("recurrence closure count exceeds the u32 ID domain"))?;
        u32::try_from(helicity_count)
            .map_err(|_| invalid("recurrence resolved-helicity count exceeds u32"))?;
        u32::try_from(destination_count)
            .map_err(|_| invalid("recurrence amplitude-destination count exceeds u32"))?;

        let mut next_contribution = 0u64;
        let mut fan_in_sum = 0u64;
        let mut non_source_count = 0u64;
        for (index, current) in self.currents.iter().enumerate() {
            let expected_id = index as u32;
            if current.id != expected_id {
                return Err(invalid(format!(
                    "recurrence current row {index} has non-dense id {}",
                    current.id
                )));
            }
            if current.key.helicity_identity().strategy() != self.strategy {
                return Err(invalid(format!(
                    "recurrence current {expected_id} uses strategy {} in a {} program",
                    current.key.helicity_identity().strategy(),
                    self.strategy
                )));
            }
            if u64::from(current.key.dynamic_lc_color_state_id().get()) >= color_state_count {
                return Err(invalid(format!(
                    "recurrence current {expected_id} references unknown dynamic LC color state {}",
                    current.key.dynamic_lc_color_state_id().get()
                )));
            }
            if current.contribution_range.start != next_contribution {
                return Err(invalid(format!(
                    "recurrence current {expected_id} contribution range starts at {}, expected packed offset {next_contribution}",
                    current.contribution_range.start
                )));
            }
            let range = current.contribution_range.as_usize_range(
                self.contributions.len(),
                &format!("recurrence current {expected_id} contribution"),
            )?;
            next_contribution = current
                .contribution_range
                .end("recurrence current contribution")?;
            fan_in_sum = fan_in_sum
                .checked_add(current.fan_in())
                .ok_or_else(|| invalid("recurrence fan-in sum exceeds u64"))?;

            if current.is_source() {
                if !range.is_empty() || current.finalization_id.is_some() {
                    return Err(invalid(format!(
                        "source current {expected_id} must not have contributions or finalization"
                    )));
                }
                match self.strategy {
                    RecurrenceStrategy::TopologyReplay if current.source_exact_factor.is_none() => {
                        return Err(invalid(format!(
                            "topology-replay source current {expected_id} requires an exact source factor"
                        )));
                    }
                    RecurrenceStrategy::AllFlowUnion if current.source_exact_factor.is_some() => {
                        return Err(invalid(format!(
                            "all-flow-union source current {expected_id} must take its factor from runtime dispatch"
                        )));
                    }
                    _ => {}
                }
            } else {
                if current.source_exact_factor.is_some() {
                    return Err(invalid(format!(
                        "non-source current {expected_id} cannot carry a source factor"
                    )));
                }
                non_source_count += 1;
                if range.is_empty() {
                    return Err(invalid(format!(
                        "non-source current {expected_id} requires at least one contribution"
                    )));
                }
                let finalization_id = current.finalization_id.ok_or_else(|| {
                    invalid(format!(
                        "non-source current {expected_id} requires exactly one finalization"
                    ))
                })?;
                let finalization = self
                    .finalizations
                    .get(finalization_id as usize)
                    .ok_or_else(|| {
                        invalid(format!(
                            "current {expected_id} references unknown finalization {finalization_id}"
                        ))
                    })?;
                if finalization.current_id != expected_id {
                    return Err(invalid(format!(
                        "current {expected_id} references finalization {finalization_id} owned by current {}",
                        finalization.current_id
                    )));
                }
            }

            for contribution in &self.contributions[range] {
                if contribution.result_current_id != expected_id {
                    return Err(invalid(format!(
                        "contribution {} is packed under current {expected_id} but targets current {}",
                        contribution.id, contribution.result_current_id
                    )));
                }
            }
        }
        if next_contribution != contribution_count || fan_in_sum != contribution_count {
            return Err(invalid(format!(
                "stored contribution count {contribution_count} does not equal packed fan-in sum {fan_in_sum}"
            )));
        }

        for (index, contribution) in self.contributions.iter().enumerate() {
            let expected_id = index as u32;
            if contribution.id != expected_id {
                return Err(invalid(format!(
                    "recurrence contribution row {index} has non-dense id {}",
                    contribution.id
                )));
            }
            let result = self
                .currents
                .get(contribution.result_current_id as usize)
                .ok_or_else(|| {
                    invalid(format!(
                        "contribution {expected_id} references unknown result current {}",
                        contribution.result_current_id
                    ))
                })?;
            if contribution.parent_current_ids.len()
                != contribution.key.parent_state_template_ids().len()
            {
                return Err(invalid(format!(
                    "contribution {expected_id} has {} parent current IDs but {} semantic parent states",
                    contribution.parent_current_ids.len(),
                    contribution.key.parent_state_template_ids().len()
                )));
            }
            if contribution.key.result_state_template_id() != result.key.current_state_template_id()
            {
                return Err(invalid(format!(
                    "contribution {expected_id} result-state template {} does not match current {} state template {}",
                    contribution.key.result_state_template_id(),
                    contribution.result_current_id,
                    result.key.current_state_template_id()
                )));
            }
            for (parent_ordinal, parent_id) in
                contribution.parent_current_ids.iter().copied().enumerate()
            {
                if parent_id >= contribution.result_current_id {
                    return Err(invalid(format!(
                        "contribution {expected_id} parent {parent_ordinal} current {parent_id} does not precede result current {}",
                        contribution.result_current_id
                    )));
                }
                let parent = &self.currents[parent_id as usize];
                let expected_state = contribution.key.parent_state_template_ids()[parent_ordinal];
                if parent.key.current_state_template_id() != expected_state {
                    return Err(invalid(format!(
                        "contribution {expected_id} parent {parent_ordinal} current {parent_id} has state template {}, expected {expected_state}",
                        parent.key.current_state_template_id()
                    )));
                }
            }
            if contribution.exact_factor.is_zero() {
                return Err(invalid(format!(
                    "recurrence contribution {expected_id} has zero exact factor"
                )));
            }
        }

        if finalization_count != non_source_count {
            return Err(invalid(format!(
                "recurrence has {finalization_count} finalizations for {non_source_count} non-source currents"
            )));
        }
        for (index, finalization) in self.finalizations.iter().enumerate() {
            let expected_id = index as u32;
            if finalization.id != expected_id {
                return Err(invalid(format!(
                    "recurrence finalization row {index} has non-dense id {}",
                    finalization.id
                )));
            }
            let current = self
                .currents
                .get(finalization.current_id as usize)
                .ok_or_else(|| {
                    invalid(format!(
                        "finalization {expected_id} references unknown current {}",
                        finalization.current_id
                    ))
                })?;
            if current.is_source() {
                return Err(invalid(format!(
                    "finalization {expected_id} references source current {}",
                    finalization.current_id
                )));
            }
            if current.finalization_id != Some(expected_id) {
                return Err(invalid(format!(
                    "finalization {expected_id} is not the unique finalization recorded by current {}",
                    finalization.current_id
                )));
            }
            if finalization.propagator_template_id != current.key.propagator_template_id() {
                return Err(invalid(format!(
                    "finalization {expected_id} propagator template {:?} does not match current {} template {:?}",
                    finalization.propagator_template_id,
                    finalization.current_id,
                    current.key.propagator_template_id()
                )));
            }
            if finalization.exact_factor.is_zero() {
                return Err(invalid(format!(
                    "recurrence finalization {expected_id} has zero exact factor"
                )));
            }
        }

        let source_slot_count = self
            .currents
            .iter()
            .flat_map(|current| current.key.support_source_slots().iter().copied())
            .max()
            .map_or(0usize, |slot| slot as usize + 1);
        let mut replayed_sectors = BTreeSet::new();
        let mut materialized_sectors = BTreeSet::new();
        for (index, target) in self.replay_targets.iter().enumerate() {
            let expected_id = index as u32;
            if target.id != expected_id {
                return Err(invalid(format!(
                    "recurrence replay-target row {index} has non-dense id {}",
                    target.id
                )));
            }
            if target.materialized_sector_id >= self.physical_sector_count
                || target.target_sector_id >= self.physical_sector_count
            {
                return Err(invalid(format!(
                    "recurrence replay target {expected_id} references an unknown physical sector"
                )));
            }
            if !replayed_sectors.insert(target.target_sector_id) {
                return Err(invalid(format!(
                    "physical sector {} has multiple recurrence replay targets",
                    target.target_sector_id
                )));
            }
            materialized_sectors.insert(target.materialized_sector_id);
            if target.source_slot_permutation.len() != source_slot_count {
                return Err(invalid(format!(
                    "recurrence replay target {expected_id} source permutation has length {}, expected {source_slot_count}",
                    target.source_slot_permutation.len()
                )));
            }
            let mut permutation = target.source_slot_permutation.to_vec();
            permutation.sort_unstable();
            if permutation
                .iter()
                .copied()
                .enumerate()
                .any(|(slot, value)| value as usize != slot)
            {
                return Err(invalid(format!(
                    "recurrence replay target {expected_id} source mapping is not a permutation"
                )));
            }
        }

        match self.strategy {
            RecurrenceStrategy::TopologyReplay if self.replay_targets.is_empty() => {
                return Err(invalid(
                    "topology-replay recurrence requires replay targets",
                ));
            }
            RecurrenceStrategy::AllFlowUnion if !self.replay_targets.is_empty() => {
                return Err(invalid(
                    "all-flow-union recurrence must not carry topology-replay targets",
                ));
            }
            _ => {}
        }
        for destination in &self.amplitude_destinations {
            if self.strategy == RecurrenceStrategy::TopologyReplay
                && !materialized_sectors.contains(&destination.target_sector_id)
            {
                return Err(invalid(format!(
                    "topology-replay amplitude destination {} targets non-materialized sector {}",
                    destination.id, destination.target_sector_id
                )));
            }
        }
        for sector in materialized_sectors {
            if !self
                .amplitude_destinations
                .iter()
                .any(|destination| destination.target_sector_id == sector)
            {
                return Err(invalid(format!(
                    "recurrence replay materialized sector {sector} has no amplitude destination"
                )));
            }
        }

        match self.strategy {
            RecurrenceStrategy::TopologyReplay if self.resolved_helicities.is_empty() => {
                return Err(invalid(
                    "topology-replay recurrence requires resolved-helicity destinations",
                ));
            }
            RecurrenceStrategy::AllFlowUnion if !self.resolved_helicities.is_empty() => {
                return Err(invalid(
                    "all-flow-union recurrence must select source helicity at runtime",
                ));
            }
            _ => {}
        }
        if helicity_count > self.retained_helicity_count {
            return Err(invalid(format!(
                "recurrence has {helicity_count} active helicities but retains only {} public assignments",
                self.retained_helicity_count
            )));
        }
        for (index, helicity) in self.resolved_helicities.iter().enumerate() {
            if helicity.id != index as u32 {
                return Err(invalid(format!(
                    "resolved-helicity row {index} has non-dense id {}",
                    helicity.id
                )));
            }
            if helicity.source_states.len() != helicity.public_helicities.len() {
                return Err(invalid(format!(
                    "resolved helicity {index} has inconsistent source-state and public-helicity dimensions"
                )));
            }
            for (source_slot, assignment) in helicity.source_states.iter().copied().enumerate() {
                if assignment.source_slot() as usize != source_slot {
                    return Err(invalid(format!(
                        "resolved helicity {index} does not cover source slot {source_slot} canonically"
                    )));
                }
            }
        }

        let mut next_closure = 0u64;
        for (destination_index, destination) in
            self.amplitude_destinations.iter().copied().enumerate()
        {
            let destination_id = destination_index as u32;
            if destination.id != destination_id {
                return Err(invalid(format!(
                    "amplitude destination row {destination_index} has non-dense id {}",
                    destination.id
                )));
            }
            if destination.target_sector_id >= self.physical_sector_count {
                return Err(invalid(format!(
                    "amplitude destination {destination_id} references unknown physical sector {}",
                    destination.target_sector_id
                )));
            }
            match (self.strategy, destination.target_helicity_id) {
                (RecurrenceStrategy::TopologyReplay, Some(helicity_id))
                    if u64::from(helicity_id) < helicity_count => {}
                (RecurrenceStrategy::TopologyReplay, Some(helicity_id)) => {
                    return Err(invalid(format!(
                        "amplitude destination {destination_id} references unknown resolved helicity {helicity_id}"
                    )));
                }
                (RecurrenceStrategy::TopologyReplay, None) => {
                    return Err(invalid(format!(
                        "topology-replay amplitude destination {destination_id} lacks a helicity"
                    )));
                }
                (RecurrenceStrategy::AllFlowUnion, None) => {}
                (RecurrenceStrategy::AllFlowUnion, Some(_)) => {
                    return Err(invalid(format!(
                        "all-flow-union amplitude destination {destination_id} fixes a helicity"
                    )));
                }
            }
            let range = destination.closure_range;
            if range.start != next_closure {
                return Err(invalid(format!(
                    "amplitude destination {destination_id} closure range starts at {}, expected packed offset {next_closure}",
                    range.start
                )));
            }
            let term_range = range.as_usize_range(
                self.closure_terms.len(),
                &format!("amplitude destination {destination_id} closure"),
            )?;
            next_closure = range.end("amplitude-destination closure")?;
            for term in &self.closure_terms[term_range] {
                if term.target_destination_id != destination_id {
                    return Err(invalid(format!(
                        "closure term {} is packed under destination {destination_id} but points to destination {}",
                        term.id, term.target_destination_id
                    )));
                }
            }
        }
        if next_closure != closure_count {
            return Err(invalid(format!(
                "packed amplitude-destination ranges cover {next_closure} of {closure_count} terms"
            )));
        }

        for (index, term) in self.closure_terms.iter().enumerate() {
            let expected_id = index as u32;
            if term.id != expected_id {
                return Err(invalid(format!(
                    "recurrence closure row {index} has non-dense id {}",
                    term.id
                )));
            }
            let destination = self
                .amplitude_destinations
                .get(term.target_destination_id as usize)
                .ok_or_else(|| {
                    invalid(format!(
                        "closure term {expected_id} references unknown amplitude destination {}",
                        term.target_destination_id
                    ))
                })?;
            if term.exact_factor.is_zero() {
                return Err(invalid(format!(
                    "recurrence closure term {expected_id} has zero exact factor"
                )));
            }
            for parent_id in term.parent_current_ids.iter().copied() {
                if u64::from(parent_id) >= current_count {
                    return Err(invalid(format!(
                        "closure term {expected_id} references unknown parent current {parent_id}"
                    )));
                }
            }
            if let Some(helicity_id) = destination.target_helicity_id {
                let expected = self.resolved_helicities[helicity_id as usize].source_states();
                let actual = closure_parent_source_states(term, &self.currents)?;
                if actual != expected {
                    return Err(invalid(format!(
                        "closure term {expected_id} ancestry does not match amplitude destination {}",
                        destination.id
                    )));
                }
            }
        }

        Ok(())
    }
}

fn closure_parent_source_states(
    term: &RecurrenceClosureTerm,
    currents: &[RecurrenceCurrent],
) -> RusticolResult<Vec<SourceStateAssignment>> {
    let mut states = Vec::new();
    for parent_id in term.parent_current_ids.iter().copied() {
        let parent = currents
            .get(parent_id as usize)
            .ok_or_else(|| invalid("closure parent current is absent"))?;
        states.extend_from_slice(parent.key.helicity_identity().local_source_states());
    }
    states.sort_unstable();
    if states
        .windows(2)
        .any(|pair| pair[0].source_slot() == pair[1].source_slot())
    {
        return Err(invalid("closure parent helicity ancestries overlap"));
    }
    Ok(states)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::{
        CanonicalMomentumLinearForm, CurrentHelicityIdentity, CurrentSourceBinding,
        DynamicLCColorStateId, LCColorWitnessTermId, MomentumTerm, SemanticDigest,
        SourceStateAssignment,
    };

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).unwrap()
    }

    fn momentum() -> CanonicalMomentumLinearForm {
        CanonicalMomentumLinearForm::new(vec![MomentumTerm {
            source_slot: 0,
            coefficient: 1,
        }])
        .unwrap()
    }

    fn source_key() -> CurrentCoreKey {
        CurrentCoreKey::new(
            digest(1),
            RecurrenceNodeKind::Source,
            0,
            DynamicLCColorStateId::from_interner(0),
            vec![0],
            momentum(),
            CurrentHelicityIdentity::topology_replay(-1, vec![SourceStateAssignment::new(0, 0)])
                .unwrap(),
            vec![],
            0,
            vec![],
            CurrentSourceBinding::FixedTemplate(0),
            None,
        )
        .unwrap()
    }

    fn propagated_key() -> CurrentCoreKey {
        CurrentCoreKey::new(
            digest(1),
            RecurrenceNodeKind::Current,
            1,
            DynamicLCColorStateId::from_interner(0),
            vec![0],
            momentum(),
            CurrentHelicityIdentity::topology_replay(-1, vec![SourceStateAssignment::new(0, 0)])
                .unwrap(),
            vec![],
            0,
            vec![],
            CurrentSourceBinding::None,
            Some(0),
        )
        .unwrap()
    }

    fn contribution(parent_id: u32) -> RecurrenceContribution {
        RecurrenceContribution::new(
            0,
            1,
            vec![parent_id],
            ContributionKey::new(
                0,
                vec![0],
                vec![0],
                vec![momentum()],
                1,
                0,
                LCColorWitnessTermId::new(0, 0),
                digest(2),
                0,
            )
            .unwrap(),
            ExactComplexRational::ONE,
        )
        .unwrap()
    }

    fn resolved_helicity() -> RecurrenceResolvedHelicity {
        RecurrenceResolvedHelicity::new(0, vec![SourceStateAssignment::new(0, 0)], vec![-1])
            .unwrap()
    }

    fn identity_replay_target() -> RecurrenceReplayTarget {
        RecurrenceReplayTarget::new(0, 0, 0, vec![0], ExactComplexRational::ONE).unwrap()
    }

    fn valid_program() -> RecurrenceProgram {
        RecurrenceProgram::new(
            RecurrenceStrategy::TopologyReplay,
            1,
            1,
            vec![DynamicLCColorState::new(0, None, vec![]).unwrap()],
            vec![
                RecurrenceCurrent::new(
                    0,
                    source_key(),
                    Some(ExactComplexRational::ONE),
                    CheckedTableRange::new(0, 0),
                    None,
                )
                .unwrap(),
                RecurrenceCurrent::new(
                    1,
                    propagated_key(),
                    None,
                    CheckedTableRange::new(0, 1),
                    Some(0),
                )
                .unwrap(),
            ],
            vec![contribution(0)],
            vec![RecurrenceFinalization::new(0, 1, Some(0), ExactComplexRational::ONE).unwrap()],
            vec![identity_replay_target()],
            vec![resolved_helicity()],
            vec![
                RecurrenceAmplitudeDestination::new(0, 0, Some(0), CheckedTableRange::new(0, 1))
                    .unwrap(),
            ],
            vec![
                RecurrenceClosureTerm::new(0, 0, 0, None, vec![1], ExactComplexRational::ONE)
                    .unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn validates_one_compact_recurrence_program() {
        let program = valid_program();
        assert_eq!(program.current_range(), CheckedTableRange::new(0, 2));
        assert_eq!(program.contribution_range(), CheckedTableRange::new(0, 1));
        assert_eq!(program.closure_term_range(), CheckedTableRange::new(0, 1));
        assert_eq!(
            program.closure_range_for_destination(0),
            Some(CheckedTableRange::new(0, 1))
        );
        assert!(program.validate().is_ok());
    }

    #[test]
    fn rejects_a_parent_that_does_not_precede_its_result() {
        let error = RecurrenceProgram::new(
            RecurrenceStrategy::TopologyReplay,
            1,
            1,
            vec![DynamicLCColorState::new(0, None, vec![]).unwrap()],
            vec![
                RecurrenceCurrent::new(
                    0,
                    source_key(),
                    Some(ExactComplexRational::ONE),
                    CheckedTableRange::new(0, 0),
                    None,
                )
                .unwrap(),
                RecurrenceCurrent::new(
                    1,
                    propagated_key(),
                    None,
                    CheckedTableRange::new(0, 1),
                    Some(0),
                )
                .unwrap(),
            ],
            vec![contribution(1)],
            vec![RecurrenceFinalization::new(0, 1, Some(0), ExactComplexRational::ONE).unwrap()],
            vec![identity_replay_target()],
            vec![resolved_helicity()],
            vec![
                RecurrenceAmplitudeDestination::new(0, 0, Some(0), CheckedTableRange::new(0, 1))
                    .unwrap(),
            ],
            vec![
                RecurrenceClosureTerm::new(0, 0, 0, None, vec![1], ExactComplexRational::ONE)
                    .unwrap(),
            ],
        )
        .unwrap_err();
        assert!(error.message().contains("does not precede result current"));
    }

    #[test]
    fn rejects_a_closure_destination_with_mismatched_source_ancestry() {
        let error = RecurrenceProgram::new(
            RecurrenceStrategy::TopologyReplay,
            1,
            2,
            vec![DynamicLCColorState::new(0, None, vec![]).unwrap()],
            vec![
                RecurrenceCurrent::new(
                    0,
                    source_key(),
                    Some(ExactComplexRational::ONE),
                    CheckedTableRange::new(0, 0),
                    None,
                )
                .unwrap(),
                RecurrenceCurrent::new(
                    1,
                    propagated_key(),
                    None,
                    CheckedTableRange::new(0, 1),
                    Some(0),
                )
                .unwrap(),
            ],
            vec![contribution(0)],
            vec![RecurrenceFinalization::new(0, 1, Some(0), ExactComplexRational::ONE).unwrap()],
            vec![identity_replay_target()],
            vec![
                RecurrenceResolvedHelicity::new(0, vec![SourceStateAssignment::new(0, 1)], vec![1])
                    .unwrap(),
            ],
            vec![
                RecurrenceAmplitudeDestination::new(0, 0, Some(0), CheckedTableRange::new(0, 1))
                    .unwrap(),
            ],
            vec![
                RecurrenceClosureTerm::new(0, 0, 0, None, vec![1], ExactComplexRational::ONE)
                    .unwrap(),
            ],
        )
        .unwrap_err();
        assert!(error.message().contains("ancestry does not match"));
    }
}
