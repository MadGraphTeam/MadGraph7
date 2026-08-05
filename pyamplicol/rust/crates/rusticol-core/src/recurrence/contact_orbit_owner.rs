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
    construct::TemplateCatalog,
    template::{
        ContactOrbitStage, EvaluatorCallableKind, EvaluatorContractKind,
        LCColorTransitionWitnessRow, OwnedRecurrenceTemplateInput, TransitionRow,
    },
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
    pub(super) evaluator_class: String,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct ContactOrbitApplicationWitness {
    pub(super) transition_equivalence_class: String,
    pub(super) quantum_semantic_digest: SemanticDigest,
    pub(super) color_contraction_semantic_digest: SemanticDigest,
    pub(super) color_witness_term_id: LCColorWitnessTermId,
    pub(super) color_witness_proof_digest: SemanticDigest,
    pub(super) coupling_orders: Vec<u32>,
    pub(super) coupling_parameter_ids: Vec<u32>,
    pub(super) momentum_convention: Vec<u32>,
    pub(super) binding_coupling: ExactComplexRational,
    pub(super) transition_exact_factor: ExactComplexRational,
    pub(super) color_exact_factor: ExactComplexRational,
    pub(super) witness_exact_factor: ExactComplexRational,
    pub(super) input_exchange_factor: Option<ExactComplexRational>,
    pub(super) output_factor_source: u8,
    pub(super) evaluator_contract_kind: u8,
    pub(super) evaluator_callable_kind: u8,
    pub(super) evaluator_callable_signature: SemanticDigest,
    pub(super) evaluator_input_layout: Vec<u32>,
    pub(super) evaluator_output_layout: Vec<u32>,
    pub(super) evaluator_exact_expression_digests: Vec<SemanticDigest>,
    pub(super) evaluator_runtime_template: Option<String>,
    pub(super) output_projection_id: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PreparedContactOrbitTransition {
    step: ContactOrbitStepProof,
    transition_semantic_digest: SemanticDigest,
    applications: Vec<ContactOrbitApplicationWitness>,
}

fn try_copy_u32(values: &[u32], label: &str) -> RusticolResult<Vec<u32>> {
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(values.len())
        .map_err(|error| allocation(format!("{label} allocation failed: {error}")))?;
    copied.extend_from_slice(values);
    Ok(copied)
}

fn try_copy_digests(
    catalog: &TemplateCatalog<'_>,
    digest_ids: &[u32],
    label: &str,
) -> RusticolResult<Vec<SemanticDigest>> {
    let mut digests = Vec::new();
    digests
        .try_reserve_exact(digest_ids.len())
        .map_err(|error| allocation(format!("{label} allocation failed: {error}")))?;
    for digest_id in digest_ids {
        digests.push(catalog.digest(*digest_id, label)?);
    }
    Ok(digests)
}

fn try_clone_digests(
    values: &[SemanticDigest],
    label: &str,
) -> RusticolResult<Vec<SemanticDigest>> {
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(values.len())
        .map_err(|error| allocation(format!("{label} allocation failed: {error}")))?;
    copied.extend_from_slice(values);
    Ok(copied)
}

fn try_copy_string(value: &str, label: &str) -> RusticolResult<String> {
    let mut copied = String::new();
    copied
        .try_reserve_exact(value.len())
        .map_err(|error| allocation(format!("{label} allocation failed: {error}")))?;
    copied.push_str(value);
    Ok(copied)
}

fn strict_contact_orbit_step(
    input: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    transition: TransitionRow,
) -> RusticolResult<Option<ContactOrbitStepProof>> {
    let step_ids = catalog.u32_sequence(
        transition.contact_orbit_step_sequence_id,
        "transition contact-orbit steps",
    )?;
    let step_digest_ids = catalog.u32_sequence(
        transition.contact_orbit_step_semantic_digest_sequence_id,
        "transition contact-orbit step digests",
    )?;
    if step_ids.len() != step_digest_ids.len() {
        return Err(integrity(
            "transition contact-orbit step bindings are incomplete",
        ));
    }
    match step_ids {
        [] => return Ok(None),
        [_] => {}
        _ => {
            return Err(integrity(
                "certified transition must bind exactly one contact-orbit step",
            ));
        }
    }
    let step_id = step_ids[0];
    let step = input
        .contact_orbit_steps
        .get(step_id as usize)
        .copied()
        .filter(|step| step.id == step_id)
        .ok_or_else(|| integrity("transition contact-orbit step is absent"))?;
    if step.semantic_digest_id != step_digest_ids[0] {
        return Err(integrity(
            "transition contact-orbit step digest binding is stale",
        ));
    }
    let certificate = input
        .contact_orbit_certificates
        .get(step.certificate_id as usize)
        .copied()
        .filter(|certificate| certificate.id == step.certificate_id)
        .ok_or_else(|| integrity("contact-orbit certificate is absent"))?;
    let physical_leg_equivalence_classes: [u32; 4] = catalog
        .u32_sequence(
            certificate.physical_leg_equivalence_sequence_id,
            "contact-orbit physical-leg equivalence classes",
        )?
        .try_into()
        .map_err(|_| integrity("contact-orbit certificate does not describe four legs"))?;
    let source_particle_legs: [i32; 3] = catalog
        .i32_sequence(
            step.source_particle_leg_sequence_id,
            "contact-orbit source-particle legs",
        )?
        .try_into()
        .map_err(|_| integrity("contact-orbit step does not describe three source legs"))?;
    let proof = ContactOrbitStepProof {
        certificate_semantic_digest: catalog.digest(
            certificate.semantic_digest_id,
            "contact-orbit certificate semantic",
        )?,
        step_semantic_digest: catalog
            .digest(step.semantic_digest_id, "contact-orbit step semantic")?,
        stage: ContactOrbitStage::try_from(step.stage)?,
        result_leg: u32::from(step.result_leg),
        physical_leg_equivalence_classes,
        left_covered_legs: ContactOrbitCoveredLegs::new(catalog.u32_sequence(
            step.left_covered_leg_sequence_id,
            "contact-orbit left covered legs",
        )?)?,
        right_covered_legs: ContactOrbitCoveredLegs::new(catalog.u32_sequence(
            step.right_covered_leg_sequence_id,
            "contact-orbit right covered legs",
        )?)?,
        source_particle_legs,
        certificate_reconstruction_factor: catalog.factor(
            certificate.reconstruction_factor_id,
            "contact-orbit certificate reconstruction",
        )?,
        step_reconstruction_factor: catalog.factor(
            step.reconstruction_factor_id,
            "contact-orbit step reconstruction",
        )?,
        evaluator_class: try_copy_string(
            catalog.string(
                certificate.evaluator_class_string_id,
                "contact-orbit evaluator class",
            )?,
            "contact-orbit evaluator class",
        )?,
    };
    validate_step(&proof)?;
    Ok(Some(proof))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_contact_orbit_transition(
    input: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    transition: TransitionRow,
    quantum_semantic_digest: SemanticDigest,
    local_coupling_orders: &[u32],
    binding_coupling: ExactComplexRational,
    transition_exact_factor: ExactComplexRational,
    contraction_exact_factor: ExactComplexRational,
    input_exchange_factor: Option<ExactComplexRational>,
    witness_rows: &[LCColorTransitionWitnessRow],
) -> RusticolResult<Option<PreparedContactOrbitTransition>> {
    let Some(step) = strict_contact_orbit_step(input, catalog, transition)? else {
        return Ok(None);
    };
    let contraction = input
        .color_contractions
        .get(transition.color_contraction_template_id as usize)
        .copied()
        .filter(|row| row.id == transition.color_contraction_template_id)
        .ok_or_else(|| integrity("contact-orbit color contraction is absent"))?;
    let binding = input
        .evaluator_bindings
        .get(transition.evaluator_binding_id as usize)
        .copied()
        .filter(|row| row.id == transition.evaluator_binding_id)
        .ok_or_else(|| integrity("contact-orbit evaluator binding is absent"))?;
    if EvaluatorContractKind::try_from(binding.contract_kind)? != EvaluatorContractKind::Vertex {
        return Err(integrity("contact-orbit evaluator binding is not a vertex"));
    }
    let _ = EvaluatorCallableKind::try_from(binding.callable_kind)?;
    let evaluator_input_layout = try_copy_u32(
        catalog.u32_sequence(
            binding.input_layout_sequence_id,
            "contact-orbit evaluator input layout",
        )?,
        "contact-orbit evaluator input layout",
    )?;
    let evaluator_output_layout = try_copy_u32(
        catalog.u32_sequence(
            binding.output_layout_sequence_id,
            "contact-orbit evaluator output layout",
        )?,
        "contact-orbit evaluator output layout",
    )?;
    let evaluator_exact_expression_digests = try_copy_digests(
        catalog,
        catalog.u32_sequence(
            binding.exact_expression_digest_sequence_id,
            "contact-orbit evaluator exact expressions",
        )?,
        "contact-orbit evaluator exact expression",
    )?;
    let coupling_orders =
        try_copy_u32(local_coupling_orders, "contact-orbit local coupling orders")?;
    let coupling_parameter_ids = try_copy_u32(
        catalog.u32_sequence(
            transition.coupling_parameter_sequence_id,
            "contact-orbit coupling parameters",
        )?,
        "contact-orbit coupling parameters",
    )?;
    let momentum_convention = try_copy_u32(
        catalog.u32_sequence(
            transition.momentum_convention_sequence_id,
            "contact-orbit momentum convention",
        )?,
        "contact-orbit momentum convention",
    )?;
    let transition_equivalence_class = try_copy_string(
        catalog.string(
            transition.equivalence_class_string_id,
            "contact-orbit transition equivalence class",
        )?,
        "contact-orbit transition equivalence class",
    )?;
    let evaluator_runtime_template = if binding.runtime_template_string_id == u32::MAX {
        None
    } else {
        Some(try_copy_string(
            catalog.string(
                binding.runtime_template_string_id,
                "contact-orbit evaluator runtime template",
            )?,
            "contact-orbit evaluator runtime template",
        )?)
    };
    let color_contraction_semantic_digest = catalog.digest(
        contraction.semantic_digest_id,
        "contact-orbit color-contraction semantic",
    )?;
    let evaluator_callable_signature = catalog.digest(
        binding.callable_signature_digest_id,
        "contact-orbit evaluator callable signature",
    )?;
    let mut applications = Vec::new();
    applications
        .try_reserve_exact(witness_rows.len())
        .map_err(|error| {
            allocation(format!(
                "contact-orbit application allocation failed: {error}"
            ))
        })?;
    for witness in witness_rows {
        if witness.color_contraction_id != contraction.id {
            return Err(integrity(
                "contact-orbit witness belongs to another color contraction",
            ));
        }
        applications.push(ContactOrbitApplicationWitness {
            transition_equivalence_class: try_copy_string(
                &transition_equivalence_class,
                "contact-orbit transition equivalence class",
            )?,
            quantum_semantic_digest,
            color_contraction_semantic_digest,
            color_witness_term_id: LCColorWitnessTermId::new(contraction.id, witness.ordinal),
            color_witness_proof_digest: catalog
                .digest(witness.proof_digest_id, "contact-orbit color witness proof")?,
            coupling_orders: try_copy_u32(&coupling_orders, "contact-orbit local coupling orders")?,
            coupling_parameter_ids: try_copy_u32(
                &coupling_parameter_ids,
                "contact-orbit coupling parameters",
            )?,
            momentum_convention: try_copy_u32(
                &momentum_convention,
                "contact-orbit momentum convention",
            )?,
            binding_coupling,
            transition_exact_factor,
            color_exact_factor: contraction_exact_factor,
            witness_exact_factor: catalog
                .factor(witness.exact_factor_id, "contact-orbit color witness")?,
            input_exchange_factor,
            output_factor_source: transition.output_factor_source,
            evaluator_contract_kind: binding.contract_kind,
            evaluator_callable_kind: binding.callable_kind,
            evaluator_callable_signature,
            evaluator_input_layout: try_copy_u32(
                &evaluator_input_layout,
                "contact-orbit evaluator input layout",
            )?,
            evaluator_output_layout: try_copy_u32(
                &evaluator_output_layout,
                "contact-orbit evaluator output layout",
            )?,
            evaluator_exact_expression_digests: try_clone_digests(
                &evaluator_exact_expression_digests,
                "contact-orbit evaluator exact expressions",
            )?,
            evaluator_runtime_template: evaluator_runtime_template
                .as_deref()
                .map(|value| try_copy_string(value, "contact-orbit evaluator runtime template"))
                .transpose()?,
            output_projection_id: transition.output_projection_string_id,
        });
    }
    Ok(Some(PreparedContactOrbitTransition {
        step,
        transition_semantic_digest: catalog.digest(
            transition.semantic_digest_id,
            "contact-orbit transition semantic",
        )?,
        applications,
    }))
}

impl PreparedContactOrbitTransition {
    pub(super) fn owner_candidate<'a>(
        &'a self,
        destination: &'a CurrentCoreKey,
        parents: [&'a CurrentCoreKey; 2],
        color_witness_term_id: LCColorWitnessTermId,
    ) -> RusticolResult<ContactOrbitOwnerCandidate<'a>> {
        let mut matches = self
            .applications
            .iter()
            .filter(|application| application.color_witness_term_id == color_witness_term_id);
        let application = matches
            .next()
            .ok_or_else(|| integrity("certified contact-orbit color witness is absent"))?;
        if matches.next().is_some() {
            return Err(integrity(
                "certified contact-orbit color witness is ambiguous",
            ));
        }
        contact_orbit_owner_candidate(
            Some(&self.step),
            destination,
            parents,
            application,
            self.transition_semantic_digest,
        )?
        .ok_or_else(|| integrity("certified contact-orbit owner candidate disappeared"))
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PhysicalLegAssignments {
    len: u8,
    values: [(u32, u32); 2],
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ContactOrbitPhysicalAssignment {
    physical_leg_assignments: PhysicalLegAssignments,
    source_particle_leg: i32,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ContactOrbitOwnerGroupKey<'a> {
    certificate_semantic_digest: SemanticDigest,
    stage: ContactOrbitStage,
    result_leg_equivalence_class: u32,
    destination: &'a CurrentCoreKey,
    parents: [&'a CurrentCoreKey; 2],
    physical_assignments: [ContactOrbitPhysicalAssignment; 2],
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

fn canonical_physical_assignment(
    covered_legs: ContactOrbitCoveredLegs,
    equivalence_classes: &[u32; 4],
    source_particle_leg: i32,
) -> RusticolResult<ContactOrbitPhysicalAssignment> {
    let mut assignments = [(0_u32, 0_u32); 2];
    for (index, leg) in covered_legs.values().iter().copied().enumerate() {
        let equivalence = *equivalence_classes
            .get(leg as usize)
            .ok_or_else(|| integrity("contact-orbit covered leg is outside arity"))?;
        assignments[index] = (leg, equivalence);
    }
    assignments[..covered_legs.len()].sort_unstable();
    Ok(ContactOrbitPhysicalAssignment {
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
    let mut parent_support_lengths = [
        parents[0].support_source_slots().len(),
        parents[1].support_source_slots().len(),
    ];
    parent_support_lengths.sort_unstable();
    let mut covered_lengths = [step.left_covered_legs.len(), step.right_covered_legs.len()];
    covered_lengths.sort_unstable();
    if parent_support_lengths != covered_lengths {
        return Err(integrity(
            "contact-orbit covered-leg cardinalities differ from parent supports",
        ));
    }
    let mut physical_assignments = [
        canonical_physical_assignment(
            step.left_covered_legs,
            &step.physical_leg_equivalence_classes,
            step.source_particle_legs[0],
        )?,
        canonical_physical_assignment(
            step.right_covered_legs,
            &step.physical_leg_equivalence_classes,
            step.source_particle_legs[1],
        )?,
    ];
    physical_assignments.sort_unstable();
    let mut canonical_parents = parents;
    canonical_parents.sort_unstable();
    let result_leg_equivalence_class =
        step.physical_leg_equivalence_classes[step.result_leg as usize];
    Ok(Some(ContactOrbitOwnerCandidate {
        group: ContactOrbitOwnerGroupKey {
            certificate_semantic_digest: step.certificate_semantic_digest,
            stage: step.stage,
            result_leg_equivalence_class,
            destination,
            parents: canonical_parents,
            physical_assignments,
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ContactOrbitTestBinding {
    None,
    One,
    Two,
    MissingStep,
    DigestMismatch,
}

#[cfg(test)]
pub(super) fn contact_orbit_test_template(
    binding: ContactOrbitTestBinding,
) -> OwnedRecurrenceTemplateInput {
    use super::{
        template::{ContactOrbitCertificateRow, ContactOrbitStepRow, DigestCatalogRow},
        validated_template_fixture,
    };

    fn string_id(input: &OwnedRecurrenceTemplateInput, value: &str) -> u32 {
        input
            .string_ranges
            .iter()
            .position(|range| {
                let range = range
                    .as_usize_range(input.string_bytes.len(), "test string")
                    .unwrap();
                input.string_bytes[range] == *value.as_bytes()
            })
            .unwrap() as u32
    }

    fn u32_sequence_id(input: &OwnedRecurrenceTemplateInput, values: &[u32]) -> u32 {
        input
            .u32_sequence_ranges
            .iter()
            .position(|row| {
                let range = row
                    .range
                    .as_usize_range(input.u32_sequence_values.len(), "test u32 sequence")
                    .unwrap();
                &input.u32_sequence_values[range] == values
            })
            .unwrap() as u32
    }

    fn i32_sequence_id(input: &OwnedRecurrenceTemplateInput, values: &[i32]) -> u32 {
        input
            .i32_sequence_ranges
            .iter()
            .position(|row| {
                let range = row
                    .range
                    .as_usize_range(input.i32_sequence_values.len(), "test i32 sequence")
                    .unwrap();
                &input.i32_sequence_values[range] == values
            })
            .unwrap() as u32
    }

    fn append_digest(input: &mut OwnedRecurrenceTemplateInput, byte: u8) -> u32 {
        let id = input.digest_catalog.len() as u32;
        input.digest_catalog.push(DigestCatalogRow {
            id,
            value: [byte; 32],
        });
        id
    }

    let mut input = validated_template_fixture().into_input();
    if binding == ContactOrbitTestBinding::None {
        return input;
    }

    let algorithm = string_id(&input, "compiler-certified-contact-orbit");
    let evaluator_class = string_id(
        &input,
        "constant-scalar-literal-singlet-self-conjugate-boson",
    );
    let certificate_template = string_id(&input, "any");
    let first_step_template = string_id(&input, "basis");
    let second_step_template = string_id(&input, "component");
    let particle = string_id(&input, "scalar");
    let vertex = particle;
    let particles = u32_sequence_id(&input, &[particle, particle, particle, particle]);
    let equivalence = u32_sequence_id(&input, &[0, 0, 0, 0]);
    let left_first = u32_sequence_id(&input, &[0]);
    let right_first = u32_sequence_id(&input, &[1]);
    let left_second = right_first;
    let right_second = left_first;
    let source_first = i32_sequence_id(&input, &[0, 1, -1]);
    let source_second = i32_sequence_id(&input, &[1, 0, -1]);
    let certificate_digest = append_digest(&mut input, 26);
    let first_step_digest = append_digest(&mut input, 27);
    let second_step_digest = append_digest(&mut input, 28);
    input.contact_orbit_certificates = vec![ContactOrbitCertificateRow {
        id: 0,
        template_string_id: certificate_template,
        algorithm_string_id: algorithm,
        algorithm_version: 1,
        term_id: 0,
        vertex_string_id: vertex,
        particle_string_sequence_id: particles,
        evaluator_class_string_id: evaluator_class,
        physical_leg_equivalence_sequence_id: equivalence,
        reconstruction_factor_id: 0,
        semantic_digest_id: certificate_digest,
    }];
    input.contact_orbit_steps = vec![
        ContactOrbitStepRow {
            id: 0,
            template_string_id: first_step_template,
            certificate_id: 0,
            stage: ContactOrbitStage::Partial as u8,
            result_leg: 2,
            left_covered_leg_sequence_id: left_first,
            right_covered_leg_sequence_id: right_first,
            source_particle_leg_sequence_id: source_first,
            reconstruction_factor_id: 0,
            semantic_digest_id: first_step_digest,
        },
        ContactOrbitStepRow {
            id: 1,
            template_string_id: second_step_template,
            certificate_id: 0,
            stage: ContactOrbitStage::Partial as u8,
            result_leg: 2,
            left_covered_leg_sequence_id: left_second,
            right_covered_leg_sequence_id: right_second,
            source_particle_leg_sequence_id: source_second,
            reconstruction_factor_id: 0,
            semantic_digest_id: second_step_digest,
        },
    ];
    input.catalog_header[0].contact_orbit_certificate_count = 1;
    input.catalog_header[0].contact_orbit_step_count = 2;
    let (step_ids, digest_ids) = match binding {
        ContactOrbitTestBinding::None => unreachable!(),
        ContactOrbitTestBinding::One => (vec![0], vec![first_step_digest]),
        ContactOrbitTestBinding::Two => (vec![0, 1], vec![first_step_digest, second_step_digest]),
        ContactOrbitTestBinding::MissingStep => (vec![2], vec![first_step_digest]),
        ContactOrbitTestBinding::DigestMismatch => (vec![0], vec![second_step_digest]),
    };
    input.transitions[0].contact_orbit_step_sequence_id = u32_sequence_id(&input, &step_ids);
    input.transitions[0].contact_orbit_step_semantic_digest_sequence_id =
        u32_sequence_id(&input, &digest_ids);
    input
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
            transition_equivalence_class: "constant-scalar-contact".into(),
            quantum_semantic_digest: digest(2),
            color_contraction_semantic_digest: digest(3),
            color_witness_term_id: LCColorWitnessTermId::new(4, 0),
            color_witness_proof_digest: digest(5),
            coupling_orders: vec![1],
            coupling_parameter_ids: vec![2],
            momentum_convention: vec![0, 1],
            binding_coupling: ExactComplexRational::ONE,
            transition_exact_factor: ExactComplexRational::ONE,
            color_exact_factor: ExactComplexRational::ONE,
            witness_exact_factor: ExactComplexRational::ONE,
            input_exchange_factor: None,
            output_factor_source: 0,
            evaluator_contract_kind: EvaluatorContractKind::Vertex as u8,
            evaluator_callable_kind: EvaluatorCallableKind::PreparedKernel as u8,
            evaluator_callable_signature: digest(6),
            evaluator_input_layout: vec![10, 11],
            evaluator_output_layout: vec![12],
            evaluator_exact_expression_digests: vec![digest(7)],
            evaluator_runtime_template: None,
            output_projection_id: 8,
        }
    }

    fn prepare_test_contact_orbit(
        input: &OwnedRecurrenceTemplateInput,
    ) -> RusticolResult<Option<PreparedContactOrbitTransition>> {
        let catalog = TemplateCatalog::new(input)?;
        let transition = input.transitions[0];
        let quantum = input.quantum_flows[transition.quantum_flow_template_id as usize];
        let contraction =
            input.color_contractions[transition.color_contraction_template_id as usize];
        prepare_contact_orbit_transition(
            input,
            &catalog,
            transition,
            catalog.digest(quantum.semantic_digest_id, "test quantum semantic")?,
            &catalog.coupling_orders(transition.coupling_order_set_id)?,
            catalog.factor(
                transition.binding_coupling_factor_id,
                "test binding coupling",
            )?,
            catalog.factor(transition.exact_factor_id, "test transition exact")?,
            catalog.factor(
                contraction.exact_coefficient_factor_id,
                "test contraction exact",
            )?,
            None,
            catalog.witness_rows(transition.color_contraction_template_id)?,
        )
    }

    #[test]
    fn strict_prepared_contact_orbit_decode_is_zero_or_one_and_fail_closed() {
        let none = contact_orbit_test_template(ContactOrbitTestBinding::None);
        assert!(prepare_test_contact_orbit(&none).unwrap().is_none());

        let one = contact_orbit_test_template(ContactOrbitTestBinding::One)
            .validate()
            .unwrap();
        assert!(prepare_test_contact_orbit(one.input()).unwrap().is_some());

        let two = contact_orbit_test_template(ContactOrbitTestBinding::Two)
            .validate()
            .unwrap();
        assert!(
            prepare_test_contact_orbit(two.input())
                .unwrap_err()
                .to_string()
                .contains("exactly one contact-orbit step")
        );

        let missing = contact_orbit_test_template(ContactOrbitTestBinding::MissingStep);
        assert!(
            prepare_test_contact_orbit(&missing)
                .unwrap_err()
                .to_string()
                .contains("contact-orbit step is absent")
        );

        let mismatch = contact_orbit_test_template(ContactOrbitTestBinding::DigestMismatch);
        assert!(
            prepare_test_contact_orbit(&mismatch)
                .unwrap_err()
                .to_string()
                .contains("digest binding is stale")
        );
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

    fn final_step(
        left: &[u32],
        right: &[u32],
        result: u32,
        digest_byte: u8,
        equivalence_classes: [u32; 4],
    ) -> ContactOrbitStepProof {
        ContactOrbitStepProof {
            certificate_semantic_digest: digest(9),
            step_semantic_digest: digest(digest_byte),
            stage: ContactOrbitStage::Final,
            result_leg: result,
            physical_leg_equivalence_classes: equivalence_classes,
            left_covered_legs: ContactOrbitCoveredLegs::new(left).unwrap(),
            right_covered_legs: ContactOrbitCoveredLegs::new(right).unwrap(),
            source_particle_legs: [
                if left.len() == 2 { -1 } else { left[0] as i32 },
                if right.len() == 2 {
                    -1
                } else {
                    right[0] as i32
                },
                result as i32,
            ],
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
            [&left, &right],
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
                    [&left, &right],
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
                    [&left, &right],
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
                    [&left, &right],
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

        let single = current(3, 3, &[10]);
        let pair = current(4, 4, &[11, 12]);
        let final_destination = current(5, 5, &[10, 11, 12]);
        let final_forward = [
            final_step(&[0], &[1, 2], 3, 60, [0, 0, 0, 0]),
            final_step(&[0], &[1, 3], 2, 61, [0, 0, 0, 0]),
            final_step(&[0], &[2, 3], 1, 62, [0, 0, 0, 0]),
        ];
        let final_reverse = [
            final_step(&[1, 2], &[0], 3, 70, [0, 0, 0, 0]),
            final_step(&[1, 3], &[0], 2, 71, [0, 0, 0, 0]),
            final_step(&[2, 3], &[0], 1, 72, [0, 0, 0, 0]),
        ];
        let final_candidates = (0..3)
            .flat_map(|index| {
                [
                    (
                        2 * index,
                        Some(candidate(
                            &final_forward[index as usize],
                            &final_destination,
                            [&single, &pair],
                            &application,
                            80 + index as u8,
                        )),
                    ),
                    (
                        2 * index + 1,
                        Some(candidate(
                            &final_reverse[index as usize],
                            &final_destination,
                            [&single, &pair],
                            &application,
                            90 + index as u8,
                        )),
                    ),
                ]
            })
            .collect::<Vec<_>>();
        assert_eq!(
            selected_contact_orbit_owner_tokens(final_candidates.into_iter()).unwrap(),
            vec![0, 2, 4],
        );
    }

    #[test]
    fn same_order_singleton_rows_collapse_partial_and_final_scalar_0012_aliases() {
        let scalar0_left = current(10, 1, &[10]);
        let scalar0_right = current(10, 1, &[11]);
        let partial_destination = current(20, 2, &[10, 11]);
        let application = application();
        let mut partial_forward = partial_step(0, 1, 2, 50);
        partial_forward.physical_leg_equivalence_classes = [0, 0, 1, 2];
        let mut partial_reverse = partial_step(1, 0, 2, 51);
        partial_reverse.physical_leg_equivalence_classes = [0, 0, 1, 2];
        let partial_forward = candidate(
            &partial_forward,
            &partial_destination,
            [&scalar0_left, &scalar0_right],
            &application,
            52,
        );
        let partial_reverse = candidate(
            &partial_reverse,
            &partial_destination,
            [&scalar0_left, &scalar0_right],
            &application,
            53,
        );
        assert_eq!(partial_forward.group, partial_reverse.group);
        assert_eq!(
            selected_contact_orbit_owner_tokens([
                (0_u32, Some(partial_forward)),
                (1_u32, Some(partial_reverse)),
            ])
            .unwrap(),
            vec![0],
        );
        assert_eq!(
            selected_contact_orbit_owner_tokens([
                (1_u32, Some(partial_reverse)),
                (0_u32, Some(partial_forward)),
            ])
            .unwrap(),
            vec![0],
        );

        let pair = current(20, 2, &[10, 11]);
        let scalar1 = current(30, 3, &[12]);
        let final_destination = current(40, 4, &[10, 11, 12]);
        let final_forward = final_step(&[0, 1], &[2], 3, 54, [0, 0, 1, 2]);
        let final_reverse = final_step(&[2], &[0, 1], 3, 55, [0, 0, 1, 2]);
        let final_forward = candidate(
            &final_forward,
            &final_destination,
            [&pair, &scalar1],
            &application,
            56,
        );
        let final_reverse = candidate(
            &final_reverse,
            &final_destination,
            [&pair, &scalar1],
            &application,
            57,
        );
        assert_eq!(final_forward.group, final_reverse.group);
        assert_eq!(
            selected_contact_orbit_owner_tokens([
                (0_u32, Some(final_forward)),
                (1_u32, Some(final_reverse)),
            ])
            .unwrap(),
            vec![1],
        );
        assert_eq!(
            selected_contact_orbit_owner_tokens([
                (1_u32, Some(final_reverse)),
                (0_u32, Some(final_forward)),
            ])
            .unwrap(),
            vec![1],
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
        value.coupling_orders = vec![2];
        variants.push(value);
        let mut value = base_application.clone();
        value.coupling_parameter_ids = vec![3];
        variants.push(value);
        let mut value = base_application.clone();
        value.momentum_convention = vec![1, 0];
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
        value.transition_equivalence_class = "different-equivalence".into();
        variants.push(value);
        let mut value = base_application.clone();
        value.evaluator_contract_kind = EvaluatorContractKind::Closure as u8;
        variants.push(value);
        let mut value = base_application.clone();
        value.evaluator_callable_kind = EvaluatorCallableKind::RusticolTemplate as u8;
        variants.push(value);
        let mut value = base_application.clone();
        value.evaluator_callable_signature = digest(23);
        variants.push(value);
        let mut value = base_application.clone();
        value.evaluator_input_layout = vec![13, 14];
        variants.push(value);
        let mut value = base_application.clone();
        value.evaluator_output_layout = vec![15];
        variants.push(value);
        let mut value = base_application.clone();
        value.evaluator_exact_expression_digests = vec![digest(24)];
        variants.push(value);
        let mut value = base_application.clone();
        value.evaluator_runtime_template = Some("different-runtime-template".into());
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
