// SPDX-License-Identifier: 0BSD

//! Compact, immutable recurrence schedule produced by the recurrence builder.
//!
//! The owned vectors are build-time storage. Evaluation consumes only packed
//! slices and ranges, so traversing a validated program requires no maps or
//! heap allocation.

use super::{
    CheckedTableRange, ContributionKey, CurrentCoreKey, DynamicLCColorState, ExactComplexRational,
    RecurrenceNodeKind, RecurrenceStrategy,
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
    target_sector_id: u32,
    closure_template_id: u32,
    quantum_flow_template_id: Option<u32>,
    parent_current_ids: Box<[u32]>,
    exact_factor: ExactComplexRational,
}

impl RecurrenceClosureTerm {
    pub fn new(
        id: u32,
        target_sector_id: u32,
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
            target_sector_id,
            closure_template_id,
            quantum_flow_template_id,
            parent_current_ids: parent_current_ids.into_boxed_slice(),
            exact_factor,
        })
    }

    pub const fn id(&self) -> u32 {
        self.id
    }

    pub const fn target_sector_id(&self) -> u32 {
        self.target_sector_id
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
    dynamic_color_states: Box<[DynamicLCColorState]>,
    currents: Box<[RecurrenceCurrent]>,
    contributions: Box<[RecurrenceContribution]>,
    finalizations: Box<[RecurrenceFinalization]>,
    target_sector_closure_ranges: Box<[CheckedTableRange]>,
    closure_terms: Box<[RecurrenceClosureTerm]>,
}

impl RecurrenceProgram {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        strategy: RecurrenceStrategy,
        dynamic_color_states: Vec<DynamicLCColorState>,
        currents: Vec<RecurrenceCurrent>,
        contributions: Vec<RecurrenceContribution>,
        finalizations: Vec<RecurrenceFinalization>,
        target_sector_closure_ranges: Vec<CheckedTableRange>,
        closure_terms: Vec<RecurrenceClosureTerm>,
    ) -> RusticolResult<Self> {
        let program = Self {
            strategy,
            dynamic_color_states: dynamic_color_states.into_boxed_slice(),
            currents: currents.into_boxed_slice(),
            contributions: contributions.into_boxed_slice(),
            finalizations: finalizations.into_boxed_slice(),
            target_sector_closure_ranges: target_sector_closure_ranges.into_boxed_slice(),
            closure_terms: closure_terms.into_boxed_slice(),
        };
        program.validate()?;
        Ok(program)
    }

    pub const fn strategy(&self) -> RecurrenceStrategy {
        self.strategy
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

    pub fn target_sector_closure_ranges(&self) -> &[CheckedTableRange] {
        &self.target_sector_closure_ranges
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

    pub fn closure_range_for_sector(&self, sector_id: u32) -> Option<CheckedTableRange> {
        self.target_sector_closure_ranges
            .get(sector_id as usize)
            .copied()
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
        let closure_count = checked_table_len("recurrence closure term", self.closure_terms.len())?;
        let sector_count = checked_table_len(
            "recurrence target sector",
            self.target_sector_closure_ranges.len(),
        )?;

        u32::try_from(current_count)
            .map_err(|_| invalid("recurrence current count exceeds the u32 ID domain"))?;
        u32::try_from(color_state_count)
            .map_err(|_| invalid("recurrence dynamic color-state count exceeds u32"))?;
        u32::try_from(contribution_count)
            .map_err(|_| invalid("recurrence contribution count exceeds the u32 ID domain"))?;
        u32::try_from(finalization_count)
            .map_err(|_| invalid("recurrence finalization count exceeds the u32 ID domain"))?;
        u32::try_from(closure_count)
            .map_err(|_| invalid("recurrence closure count exceeds the u32 ID domain"))?;
        u32::try_from(sector_count)
            .map_err(|_| invalid("recurrence target-sector count exceeds the u32 ID domain"))?;

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

        let mut next_closure = 0u64;
        for (sector_index, range) in self
            .target_sector_closure_ranges
            .iter()
            .copied()
            .enumerate()
        {
            let sector_id = sector_index as u32;
            if range.start != next_closure {
                return Err(invalid(format!(
                    "target sector {sector_id} closure range starts at {}, expected packed offset {next_closure}",
                    range.start
                )));
            }
            let term_range = range.as_usize_range(
                self.closure_terms.len(),
                &format!("target sector {sector_id} closure"),
            )?;
            next_closure = range.end("target-sector closure")?;
            for term in &self.closure_terms[term_range] {
                if term.target_sector_id != sector_id {
                    return Err(invalid(format!(
                        "closure term {} is packed under target sector {sector_id} but points to sector {}",
                        term.id, term.target_sector_id
                    )));
                }
            }
        }
        if next_closure != closure_count {
            return Err(invalid(format!(
                "packed target-sector closure ranges cover {next_closure} of {closure_count} terms"
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
            if u64::from(term.target_sector_id) >= sector_count {
                return Err(invalid(format!(
                    "closure term {expected_id} references unknown target sector {}",
                    term.target_sector_id
                )));
            }
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
        }

        Ok(())
    }
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

    fn valid_program() -> RecurrenceProgram {
        RecurrenceProgram::new(
            RecurrenceStrategy::TopologyReplay,
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
            vec![CheckedTableRange::new(0, 1)],
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
            program.closure_range_for_sector(0),
            Some(CheckedTableRange::new(0, 1))
        );
        assert!(program.validate().is_ok());
    }

    #[test]
    fn rejects_a_parent_that_does_not_precede_its_result() {
        let error = RecurrenceProgram::new(
            RecurrenceStrategy::TopologyReplay,
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
            vec![CheckedTableRange::new(0, 1)],
            vec![
                RecurrenceClosureTerm::new(0, 0, 0, None, vec![1], ExactComplexRational::ONE)
                    .unwrap(),
            ],
        )
        .unwrap_err();
        assert!(error.message().contains("does not precede result current"));
    }
}
