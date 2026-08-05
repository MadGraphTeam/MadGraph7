// SPDX-License-Identifier: 0BSD

//! Exact owner selection for compiler-certified scalar contact orbits.
//!
//! The helper is deliberately process-blind. It forgets only binary
//! traversal direction and the concrete evaluator-binding row while retaining
//! every physical, selector, color, coupling, and numerical evaluator witness.
//! Owner selection uses fallibly reserved vector scratch: it does not clone
//! current/application graphs or grow map/set nodes per transition.

use std::{cmp::Ordering, mem::size_of};

use super::{
    CurrentCoreKey, ExactComplexRational, LCColorWitnessTermId, SemanticDigest,
    template::ContactOrbitStage,
};
use crate::{RusticolError, RusticolResult};

fn integrity(message: impl Into<String>) -> RusticolError {
    RusticolError::integrity(message)
}

fn allocation(message: impl Into<String>) -> RusticolError {
    RusticolError::internal(message)
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct ContactOrbitCoveredLegs {
    len: u8,
    values: [u32; 2],
}

impl ContactOrbitCoveredLegs {
    pub(super) fn new(values: &[u32]) -> RusticolResult<Self> {
        let len = u8::try_from(values.len())
            .map_err(|_| integrity("contact-orbit covered-leg count exceeds u8"))?;
        if !(1..=2).contains(&len) {
            return Err(integrity(
                "contact-orbit covered-leg count must be one or two",
            ));
        }
        let mut canonical = [0_u32; 2];
        canonical[..values.len()].copy_from_slice(values);
        canonical[..values.len()].sort_unstable();
        if values.len() == 2 && canonical[0] == canonical[1] {
            return Err(integrity("contact-orbit covered legs are not unique"));
        }
        Ok(Self {
            len,
            values: canonical,
        })
    }

    fn values(&self) -> &[u32] {
        &self.values[..usize::from(self.len)]
    }

    const fn len(self) -> usize {
        self.len as usize
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct ContactOrbitStepProof {
    pub(super) certificate_semantic_digest: SemanticDigest,
    pub(super) step_semantic_digest: SemanticDigest,
    pub(super) stage: ContactOrbitStage,
    pub(super) result_leg: u32,
    pub(super) physical_leg_equivalence_classes: [u32; 4],
    pub(super) left_covered_legs: ContactOrbitCoveredLegs,
    pub(super) right_covered_legs: ContactOrbitCoveredLegs,
    pub(super) source_particle_legs: [i32; 3],
    pub(super) certificate_reconstruction_factor: ExactComplexRational,
    pub(super) step_reconstruction_factor: ExactComplexRational,
    pub(super) evaluator_class: Box<str>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct ContactOrbitApplicationWitness {
    pub(super) quantum_semantic_digest: SemanticDigest,
    pub(super) color_contraction_semantic_digest: SemanticDigest,
    pub(super) color_witness_term_id: LCColorWitnessTermId,
    pub(super) color_witness_proof_digest: SemanticDigest,
    pub(super) coupling_orders: Box<[u32]>,
    pub(super) binding_coupling: ExactComplexRational,
    pub(super) transition_exact_factor: ExactComplexRational,
    pub(super) color_exact_factor: ExactComplexRational,
    pub(super) witness_exact_factor: ExactComplexRational,
    pub(super) input_exchange_factor: Option<ExactComplexRational>,
    pub(super) output_factor_source: u8,
    pub(super) evaluator_callable_signature: SemanticDigest,
    pub(super) evaluator_exact_expression_digests: Box<[SemanticDigest]>,
    pub(super) output_projection_id: u32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PhysicalLegAssignments {
    len: u8,
    values: [(u32, u32); 2],
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ContactOrbitParentAssignment<'a> {
    current: &'a CurrentCoreKey,
    physical_leg_assignments: PhysicalLegAssignments,
    source_particle_leg: i32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ContactOrbitOwnerGroupKey<'a> {
    certificate_semantic_digest: SemanticDigest,
    stage: ContactOrbitStage,
    result_leg_equivalence_class: u32,
    destination: &'a CurrentCoreKey,
    parents: [ContactOrbitParentAssignment<'a>; 2],
    output_source_particle_leg: i32,
    evaluator_class: &'a str,
    application: &'a ContactOrbitApplicationWitness,
    certificate_reconstruction_factor: ExactComplexRational,
    step_reconstruction_factor: ExactComplexRational,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ContactOrbitOwnerRank {
    oriented_covered_legs: [ContactOrbitCoveredLegs; 2],
    oriented_source_particle_legs: [i32; 3],
    step_semantic_digest: SemanticDigest,
    transition_semantic_digest: SemanticDigest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ContactOrbitOwnerCandidate<'a> {
    group: ContactOrbitOwnerGroupKey<'a>,
    rank: ContactOrbitOwnerRank,
}

fn canonical_assignment<'a>(
    current: &'a CurrentCoreKey,
    covered_legs: ContactOrbitCoveredLegs,
    equivalence_classes: &[u32; 4],
    source_particle_leg: i32,
) -> RusticolResult<ContactOrbitParentAssignment<'a>> {
    if covered_legs.len() != current.support_source_slots().len() {
        return Err(integrity(
            "contact-orbit covered-leg cardinality differs from parent support",
        ));
    }
    let mut assignments = [(0_u32, 0_u32); 2];
    for (index, leg) in covered_legs.values().iter().copied().enumerate() {
        let equivalence = *equivalence_classes
            .get(leg as usize)
            .ok_or_else(|| integrity("contact-orbit covered leg is outside arity"))?;
        assignments[index] = (leg, equivalence);
    }
    assignments[..covered_legs.len()].sort_unstable();
    Ok(ContactOrbitParentAssignment {
        current,
        physical_leg_assignments: PhysicalLegAssignments {
            len: covered_legs.len,
            values: assignments,
        },
        source_particle_leg,
    })
}

fn validate_step(step: &ContactOrbitStepProof) -> RusticolResult<()> {
    if step.certificate_reconstruction_factor != ExactComplexRational::ONE
        || step.step_reconstruction_factor != ExactComplexRational::ONE
    {
        return Err(integrity(
            "contact-orbit owner requires exact unit reconstruction",
        ));
    }
    let result_index = usize::try_from(step.result_leg)
        .ok()
        .filter(|index| *index < 4)
        .ok_or_else(|| integrity("contact-orbit result leg is outside arity"))?;
    let mut covered = [false; 4];
    for leg in step
        .left_covered_legs
        .values()
        .iter()
        .chain(step.right_covered_legs.values())
        .copied()
    {
        let index = usize::try_from(leg)
            .ok()
            .filter(|index| *index < 4)
            .ok_or_else(|| integrity("contact-orbit covered leg is outside arity"))?;
        if index == result_index || covered[index] {
            return Err(integrity("contact-orbit covered-leg partition is invalid"));
        }
        covered[index] = true;
    }
    match step.stage {
        ContactOrbitStage::Partial => {
            if step.left_covered_legs.len() != 1
                || step.right_covered_legs.len() != 1
                || step.source_particle_legs
                    != [
                        step.left_covered_legs.values()[0] as i32,
                        step.right_covered_legs.values()[0] as i32,
                        -1,
                    ]
            {
                return Err(integrity(
                    "contact-orbit partial owner lineage is inconsistent",
                ));
            }
        }
        ContactOrbitStage::Final => {
            let expected_sources = [
                if step.left_covered_legs.len() == 2 {
                    -1
                } else {
                    step.left_covered_legs.values()[0] as i32
                },
                if step.right_covered_legs.len() == 2 {
                    -1
                } else {
                    step.right_covered_legs.values()[0] as i32
                },
                step.result_leg as i32,
            ];
            if !matches!(
                (step.left_covered_legs.len(), step.right_covered_legs.len()),
                (1, 2) | (2, 1)
            ) || step.source_particle_legs != expected_sources
                || covered
                    .iter()
                    .enumerate()
                    .any(|(index, value)| *value != (index != result_index))
            {
                return Err(integrity(
                    "contact-orbit final owner lineage is inconsistent",
                ));
            }
        }
    }
    Ok(())
}

fn validate_support_union(
    destination: &CurrentCoreKey,
    parents: [&CurrentCoreKey; 2],
) -> RusticolResult<()> {
    let left = parents[0].support_source_slots();
    let right = parents[1].support_source_slots();
    let destination_support = destination.support_source_slots();
    if left
        .len()
        .checked_add(right.len())
        .filter(|combined| *combined == destination_support.len())
        .is_none()
    {
        return Err(integrity(
            "contact-orbit parent supports do not reproduce the destination",
        ));
    }
    let (mut left_index, mut right_index) = (0_usize, 0_usize);
    for expected in destination_support.iter().copied() {
        let next = match (left.get(left_index), right.get(right_index)) {
            (Some(left_value), Some(right_value)) => match left_value.cmp(right_value) {
                Ordering::Less => {
                    left_index += 1;
                    *left_value
                }
                Ordering::Greater => {
                    right_index += 1;
                    *right_value
                }
                Ordering::Equal => {
                    return Err(integrity("contact-orbit parent supports are not disjoint"));
                }
            },
            (Some(left_value), None) => {
                left_index += 1;
                *left_value
            }
            (None, Some(right_value)) => {
                right_index += 1;
                *right_value
            }
            (None, None) => {
                return Err(integrity(
                    "contact-orbit parent supports do not reproduce the destination",
                ));
            }
        };
        if next != expected {
            return Err(integrity(
                "contact-orbit parent supports do not reproduce the destination",
            ));
        }
    }
    Ok(())
}

pub(super) fn contact_orbit_owner_candidate<'a>(
    step: Option<&'a ContactOrbitStepProof>,
    destination: &'a CurrentCoreKey,
    parents: [&'a CurrentCoreKey; 2],
    application: &'a ContactOrbitApplicationWitness,
    transition_semantic_digest: SemanticDigest,
) -> RusticolResult<Option<ContactOrbitOwnerCandidate<'a>>> {
    let Some(step) = step else {
        return Ok(None);
    };
    validate_step(step)?;
    validate_support_union(destination, parents)?;

    let mut parent_assignments = [
        canonical_assignment(
            parents[0],
            step.left_covered_legs,
            &step.physical_leg_equivalence_classes,
            step.source_particle_legs[0],
        )?,
        canonical_assignment(
            parents[1],
            step.right_covered_legs,
            &step.physical_leg_equivalence_classes,
            step.source_particle_legs[1],
        )?,
    ];
    parent_assignments.sort_unstable();
    let result_leg_equivalence_class =
        step.physical_leg_equivalence_classes[step.result_leg as usize];
    Ok(Some(ContactOrbitOwnerCandidate {
        group: ContactOrbitOwnerGroupKey {
            certificate_semantic_digest: step.certificate_semantic_digest,
            stage: step.stage,
            result_leg_equivalence_class,
            destination,
            parents: parent_assignments,
            output_source_particle_leg: step.source_particle_legs[2],
            evaluator_class: &step.evaluator_class,
            application,
            certificate_reconstruction_factor: step.certificate_reconstruction_factor,
            step_reconstruction_factor: step.step_reconstruction_factor,
        },
        rank: ContactOrbitOwnerRank {
            oriented_covered_legs: [step.left_covered_legs, step.right_covered_legs],
            oriented_source_particle_legs: step.source_particle_legs,
            step_semantic_digest: step.step_semantic_digest,
            transition_semantic_digest,
        },
    }))
}

fn compare_items<T: Ord>(
    left: &(T, Option<ContactOrbitOwnerCandidate<'_>>),
    right: &(T, Option<ContactOrbitOwnerCandidate<'_>>),
) -> Ordering {
    match (&left.1, &right.1) {
        (None, None) => left.0.cmp(&right.0),
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(left_candidate), Some(right_candidate)) => left_candidate
            .group
            .cmp(&right_candidate.group)
            .then_with(|| left_candidate.rank.cmp(&right_candidate.rank))
            .then_with(|| left.0.cmp(&right.0)),
    }
}

/// Select one exact owner per certified group without mutating caller state.
///
/// The exact-size iterator is copied into one fallibly reserved scratch vector,
/// then sorted in place. The returned token vector is separately reserved
/// before any selection, so every allocation failure occurs before caller-owned
/// construction state can be replaced.
pub(super) fn selected_contact_orbit_owner_tokens<'a, T, I>(items: I) -> RusticolResult<Vec<T>>
where
    T: Ord,
    I: IntoIterator<Item = (T, Option<ContactOrbitOwnerCandidate<'a>>)>,
    I::IntoIter: ExactSizeIterator,
{
    let iterator = items.into_iter();
    let item_count = iterator.len();
    item_count
        .checked_mul(size_of::<(T, Option<ContactOrbitOwnerCandidate<'a>>)>())
        .ok_or_else(|| integrity("contact-orbit owner scratch byte size overflows usize"))?;
    let mut scratch = Vec::new();
    scratch.try_reserve_exact(item_count).map_err(|error| {
        allocation(format!(
            "contact-orbit owner scratch allocation failed: {error}"
        ))
    })?;
    for item in iterator {
        if scratch.len() == item_count {
            return Err(integrity(
                "contact-orbit exact-size owner iterator exceeded its declared length",
            ));
        }
        scratch.push(item);
    }
    if scratch.len() != item_count {
        return Err(integrity(
            "contact-orbit exact-size owner iterator changed length",
        ));
    }

    let mut selected = Vec::new();
    selected.try_reserve_exact(item_count).map_err(|error| {
        allocation(format!(
            "contact-orbit selected-owner allocation failed: {error}"
        ))
    })?;
    scratch.sort_unstable_by(compare_items);
    let mut iterator = scratch.into_iter().peekable();
    while let Some((token, candidate)) = iterator.next() {
        let Some(candidate) = candidate else {
            selected.push(token);
            continue;
        };
        while let Some((next_token, Some(next_candidate))) = iterator.peek() {
            if next_candidate.group != candidate.group {
                break;
            }
            if next_candidate.rank == candidate.rank && next_token != &token {
                return Err(integrity(
                    "contact-orbit owner candidates have a conflicting exact rank",
                ));
            }
            iterator.next();
        }
        selected.push(token);
    }
    selected.sort_unstable();
    if selected.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(integrity(
            "contact-orbit owner selection produced duplicate tokens",
        ));
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::{
        CanonicalMomentumLinearForm, CurrentHelicityIdentity, CurrentSourceBinding,
        DynamicLCColorStateId, ExactRational, MomentumTerm, RecurrenceNodeKind,
    };

    fn digest(value: u8) -> SemanticDigest {
        SemanticDigest::new([value; 32]).unwrap()
    }

    fn exact(value: i128) -> ExactComplexRational {
        ExactComplexRational::new(ExactRational::new(value, 1).unwrap(), ExactRational::ZERO)
    }

    fn current(state: u32, color: u32, support: &[u32]) -> CurrentCoreKey {
        CurrentCoreKey::new(
            digest(1),
            RecurrenceNodeKind::Current,
            state,
            DynamicLCColorStateId::from_interner(color),
            support.to_vec(),
            CanonicalMomentumLinearForm::new(
                support
                    .iter()
                    .copied()
                    .map(|source_slot| MomentumTerm {
                        source_slot,
                        coefficient: 1,
                    })
                    .collect(),
            )
            .unwrap(),
            CurrentHelicityIdentity::all_flow_union(0),
            vec![state as i32],
            0,
            vec![0],
            CurrentSourceBinding::None,
            None,
        )
        .unwrap()
    }

    fn application() -> ContactOrbitApplicationWitness {
        ContactOrbitApplicationWitness {
            quantum_semantic_digest: digest(2),
            color_contraction_semantic_digest: digest(3),
            color_witness_term_id: LCColorWitnessTermId::new(4, 0),
            color_witness_proof_digest: digest(5),
            coupling_orders: vec![1].into_boxed_slice(),
            binding_coupling: ExactComplexRational::ONE,
            transition_exact_factor: ExactComplexRational::ONE,
            color_exact_factor: ExactComplexRational::ONE,
            witness_exact_factor: ExactComplexRational::ONE,
            input_exchange_factor: None,
            output_factor_source: 0,
            evaluator_callable_signature: digest(6),
            evaluator_exact_expression_digests: vec![digest(7)].into_boxed_slice(),
            output_projection_id: 8,
        }
    }

    fn partial_step(left: u32, right: u32, result: u32, digest_byte: u8) -> ContactOrbitStepProof {
        ContactOrbitStepProof {
            certificate_semantic_digest: digest(9),
            step_semantic_digest: digest(digest_byte),
            stage: ContactOrbitStage::Partial,
            result_leg: result,
            physical_leg_equivalence_classes: [0, 0, 0, 0],
            left_covered_legs: ContactOrbitCoveredLegs::new(&[left]).unwrap(),
            right_covered_legs: ContactOrbitCoveredLegs::new(&[right]).unwrap(),
            source_particle_legs: [left as i32, right as i32, -1],
            certificate_reconstruction_factor: ExactComplexRational::ONE,
            step_reconstruction_factor: ExactComplexRational::ONE,
            evaluator_class: "constant-scalar-contact-v1".into(),
        }
    }

    fn candidate<'a>(
        step: &'a ContactOrbitStepProof,
        destination: &'a CurrentCoreKey,
        parents: [&'a CurrentCoreKey; 2],
        application: &'a ContactOrbitApplicationWitness,
        transition_digest: u8,
    ) -> ContactOrbitOwnerCandidate<'a> {
        contact_orbit_owner_candidate(
            Some(step),
            destination,
            parents,
            application,
            digest(transition_digest),
        )
        .unwrap()
        .unwrap()
    }

    #[test]
    fn owner_group_forgets_only_left_right_traversal() {
        let left = current(1, 1, &[10]);
        let right = current(1, 1, &[11]);
        let destination = current(2, 2, &[10, 11]);
        let application = application();
        let forward_step = partial_step(0, 1, 2, 10);
        let reverse_step = partial_step(1, 0, 2, 11);
        let forward = candidate(
            &forward_step,
            &destination,
            [&left, &right],
            &application,
            12,
        );
        let reverse = candidate(
            &reverse_step,
            &destination,
            [&right, &left],
            &application,
            13,
        );

        assert_eq!(forward.group, reverse.group);
        assert_ne!(forward.rank, reverse.rank);
        assert_eq!(
            selected_contact_orbit_owner_tokens([(1_u32, Some(reverse)), (0_u32, Some(forward)),])
                .unwrap(),
            vec![0],
        );
        assert_eq!(
            selected_contact_orbit_owner_tokens([(0_u32, Some(forward)), (1_u32, Some(reverse)),])
                .unwrap(),
            vec![0],
        );
    }

    #[test]
    fn same_core_four_scalar_channels_keep_three_physical_pairs() {
        let left = current(1, 1, &[10]);
        let right = current(1, 1, &[11]);
        let destination = current(2, 2, &[10, 11]);
        let application = application();
        let forward_steps = [
            partial_step(0, 1, 2, 20),
            partial_step(0, 2, 1, 21),
            partial_step(0, 3, 1, 22),
        ];
        let reverse_steps = [
            partial_step(1, 0, 2, 30),
            partial_step(2, 0, 1, 31),
            partial_step(3, 0, 1, 32),
        ];
        let candidates = [
            (
                0_u32,
                Some(candidate(
                    &forward_steps[0],
                    &destination,
                    [&left, &right],
                    &application,
                    40,
                )),
            ),
            (
                1_u32,
                Some(candidate(
                    &reverse_steps[0],
                    &destination,
                    [&right, &left],
                    &application,
                    41,
                )),
            ),
            (
                2_u32,
                Some(candidate(
                    &forward_steps[1],
                    &destination,
                    [&left, &right],
                    &application,
                    42,
                )),
            ),
            (
                3_u32,
                Some(candidate(
                    &reverse_steps[1],
                    &destination,
                    [&right, &left],
                    &application,
                    43,
                )),
            ),
            (
                4_u32,
                Some(candidate(
                    &forward_steps[2],
                    &destination,
                    [&left, &right],
                    &application,
                    44,
                )),
            ),
            (
                5_u32,
                Some(candidate(
                    &reverse_steps[2],
                    &destination,
                    [&right, &left],
                    &application,
                    45,
                )),
            ),
        ];
        assert_eq!(
            candidates
                .iter()
                .map(|(_, value)| value.unwrap().group)
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3,
        );
        assert_eq!(
            selected_contact_orbit_owner_tokens(candidates).unwrap(),
            vec![0, 2, 4],
        );
    }

    #[test]
    fn owner_group_binds_every_physical_and_evaluator_witness() {
        let left = current(1, 1, &[10]);
        let right = current(1, 1, &[11]);
        let destination = current(2, 2, &[10, 11]);
        let step = partial_step(0, 1, 2, 10);
        let base_application = application();
        let base = candidate(&step, &destination, [&left, &right], &base_application, 12);

        let mut variants = Vec::new();
        let mut value = base_application.clone();
        value.quantum_semantic_digest = digest(20);
        variants.push(value);
        let mut value = base_application.clone();
        value.color_contraction_semantic_digest = digest(21);
        variants.push(value);
        let mut value = base_application.clone();
        value.color_witness_term_id = LCColorWitnessTermId::new(4, 1);
        variants.push(value);
        let mut value = base_application.clone();
        value.color_witness_proof_digest = digest(22);
        variants.push(value);
        let mut value = base_application.clone();
        value.coupling_orders = vec![2].into_boxed_slice();
        variants.push(value);
        let mut value = base_application.clone();
        value.binding_coupling = exact(2);
        variants.push(value);
        let mut value = base_application.clone();
        value.transition_exact_factor = exact(2);
        variants.push(value);
        let mut value = base_application.clone();
        value.color_exact_factor = exact(2);
        variants.push(value);
        let mut value = base_application.clone();
        value.witness_exact_factor = exact(2);
        variants.push(value);
        let mut value = base_application.clone();
        value.input_exchange_factor = Some(exact(-1));
        variants.push(value);
        let mut value = base_application.clone();
        value.output_factor_source = 1;
        variants.push(value);
        let mut value = base_application.clone();
        value.evaluator_callable_signature = digest(23);
        variants.push(value);
        let mut value = base_application.clone();
        value.evaluator_exact_expression_digests = vec![digest(24)].into_boxed_slice();
        variants.push(value);
        let mut value = base_application.clone();
        value.output_projection_id = 9;
        variants.push(value);

        for (index, variant) in variants.iter().enumerate() {
            assert_ne!(
                base.group,
                candidate(&step, &destination, [&left, &right], variant, 12).group,
                "application witness variant {index} coalesced",
            );
        }

        let mut certificate = step.clone();
        certificate.certificate_semantic_digest = digest(30);
        assert_ne!(
            base.group,
            candidate(
                &certificate,
                &destination,
                [&left, &right],
                &base_application,
                12,
            )
            .group,
        );
        let mut equivalence = step.clone();
        equivalence.physical_leg_equivalence_classes[step.result_leg as usize] = 1;
        assert_ne!(
            base.group,
            candidate(
                &equivalence,
                &destination,
                [&left, &right],
                &base_application,
                12,
            )
            .group,
        );
        let mut evaluator_class = step.clone();
        evaluator_class.evaluator_class = "different-contact-class".into();
        assert_ne!(
            base.group,
            candidate(
                &evaluator_class,
                &destination,
                [&left, &right],
                &base_application,
                12,
            )
            .group,
        );

        let mut invalid_reconstruction = step.clone();
        invalid_reconstruction.certificate_reconstruction_factor = exact(2);
        assert!(
            contact_orbit_owner_candidate(
                Some(&invalid_reconstruction),
                &destination,
                [&left, &right],
                &base_application,
                digest(12),
            )
            .unwrap_err()
            .to_string()
            .contains("exact unit reconstruction")
        );
        invalid_reconstruction.certificate_reconstruction_factor = ExactComplexRational::ONE;
        invalid_reconstruction.step_reconstruction_factor = exact(2);
        assert!(
            contact_orbit_owner_candidate(
                Some(&invalid_reconstruction),
                &destination,
                [&left, &right],
                &base_application,
                digest(12),
            )
            .unwrap_err()
            .to_string()
            .contains("exact unit reconstruction")
        );

        let mut changed_stage = base;
        changed_stage.group.stage = ContactOrbitStage::Final;
        assert_ne!(base.group, changed_stage.group);
        let mut changed_output_source = base;
        changed_output_source.group.output_source_particle_leg = 2;
        assert_ne!(base.group, changed_output_source.group);
    }

    #[test]
    fn passthrough_rank_conflict_duplicate_token_and_capacity_fail_closed() {
        assert_eq!(
            selected_contact_orbit_owner_tokens([(7_u32, None)]).unwrap(),
            vec![7],
        );

        let left = current(1, 1, &[10]);
        let right = current(1, 1, &[11]);
        let destination = current(2, 2, &[10, 11]);
        let application = application();
        let step = partial_step(0, 1, 2, 10);
        let duplicate = candidate(&step, &destination, [&left, &right], &application, 12);
        let error = selected_contact_orbit_owner_tokens([
            (0_u32, Some(duplicate)),
            (1_u32, Some(duplicate)),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("conflicting exact rank"));

        let different_group_step = partial_step(0, 2, 1, 11);
        let different_group = candidate(
            &different_group_step,
            &destination,
            [&left, &right],
            &application,
            13,
        );
        let error = selected_contact_orbit_owner_tokens([
            (0_u32, Some(duplicate)),
            (0_u32, Some(different_group)),
        ])
        .unwrap_err();
        assert!(error.to_string().contains("duplicate tokens"));

        struct ImpossibleExactSize<'a> {
            remaining: usize,
            marker: std::marker::PhantomData<ContactOrbitOwnerCandidate<'a>>,
        }
        impl<'a> Iterator for ImpossibleExactSize<'a> {
            type Item = (u64, Option<ContactOrbitOwnerCandidate<'a>>);

            fn next(&mut self) -> Option<Self::Item> {
                None
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (self.remaining, Some(self.remaining))
            }
        }
        impl ExactSizeIterator for ImpossibleExactSize<'_> {
            fn len(&self) -> usize {
                self.remaining
            }
        }
        let error = selected_contact_orbit_owner_tokens(ImpossibleExactSize {
            remaining: usize::MAX,
            marker: std::marker::PhantomData,
        })
        .unwrap_err();
        assert!(error.to_string().contains("byte size overflows usize"));

        struct UnderreportedExactSize {
            emitted: u8,
        }
        impl Iterator for UnderreportedExactSize {
            type Item = (u64, Option<ContactOrbitOwnerCandidate<'static>>);

            fn next(&mut self) -> Option<Self::Item> {
                if self.emitted == 2 {
                    return None;
                }
                let token = u64::from(self.emitted);
                self.emitted += 1;
                Some((token, None))
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                (1, Some(1))
            }
        }
        impl ExactSizeIterator for UnderreportedExactSize {
            fn len(&self) -> usize {
                1
            }
        }
        let error =
            selected_contact_orbit_owner_tokens(UnderreportedExactSize { emitted: 0 }).unwrap_err();
        assert!(error.to_string().contains("exceeded its declared length"));
    }
}
