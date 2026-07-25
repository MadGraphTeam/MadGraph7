// SPDX-License-Identifier: 0BSD

//! Compact model-generic recurrence construction.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use super::layout::RuntimeSourceVariantBinding;
use super::process::{
    FermionPairingRuleRow, OwnedRecurrenceProcessInput, ProcessLCSectorKind,
    ProcessPhysicalLCSectorRow, ProcessSourceStateRow, ValidatedFermionPairingCatalog,
};
use super::program::closure_candidate_identity_digest_v1;
use super::template::{
    ClosureRow, LCColorTransitionWitnessRow, OutputFactorSource, OwnedRecurrenceTemplateInput,
    QuantumFlowRow, RuntimeHelicityContractRow, RuntimeHelicityVariantRow, SourceRow,
    TransitionRow,
};
use super::{
    AuthenticatedRecurrenceBuilderInput, CanonicalMomentumLinearForm, CheckedTableRange,
    ClosureCandidateDomainCertificateV1, ClosureExecutionProofGroupV2, ClosureProofContributionV2,
    ClosureProofMetadataV2, ContributionKey, CurrentCoreKey, CurrentHelicityIdentity,
    CurrentSourceBinding, DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS, DynamicLCColorState,
    DynamicLCColorStateId, DynamicLCColorStateInterner, ExactComplexRational, ExactRational,
    LCColorComponent, LCColorComponentKind, LCColorComponentOperation, LCColorComponentRole,
    LCColorParentPort, LCColorPortWiring, LCColorSourceSeed, LCColorSourceSeedOperation,
    LCColorTransitionWitness, LCColorWitnessTermId, MomentumTerm, RecurrenceAmplitudeDestination,
    RecurrenceClosureTerm, RecurrenceContribution, RecurrenceCurrent, RecurrenceFinalization,
    RecurrenceNodeKind, RecurrenceProgram, RecurrenceReplayTarget, RecurrenceResolvedHelicity,
    RecurrenceStrategy, ReflectionCertificateV1, SemanticDigest, SourceStateAssignment,
    ThreeLineTraversalCertificateV1, ThreeLineTraversalKindV1, closure_component_factor_digest_v2,
    closure_selector_domain_digest_v2,
};
use crate::{RusticolError, RusticolResult};

const MISSING_U32: u32 = u32::MAX;
const PROGRESS_PAIR_INTERVAL: usize = 16_384;
const PROGRESS_TIME_INTERVAL: Duration = Duration::from_millis(250);
const PURE_MASSLESS_ADJOINT_HELICITY_SUPPORT_ROLE: &str =
    "helicity-support:pure-massless-adjoint-tree-v1";
const REFLECTION_PROOF_ALGORITHM_ID: u32 = 1;
const THREE_LINE_DIRECT_CERTIFICATE_ID: u32 = 0;
const THREE_LINE_PARTNER_CERTIFICATE_ID: u32 = 1;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

fn hash_exact_factor(hash: &mut Sha256, factor: ExactComplexRational) {
    for rational in [factor.real(), factor.imag()] {
        hash.update(rational.numerator().to_le_bytes());
        hash.update(rational.denominator().to_le_bytes());
    }
}

fn semantic_digest_from_u32_fields(
    domain: &[u8],
    values: impl IntoIterator<Item = u32>,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(domain);
    for value in values {
        hash.update(value.to_le_bytes());
    }
    SemanticDigest::new(hash.finalize().into())
}

fn hash_digest_sequence(hash: &mut Sha256, values: &[SemanticDigest]) -> RusticolResult<()> {
    hash.update(
        u64::try_from(values.len())
            .map_err(|_| invalid("reflection proof digest count exceeds u64"))?
            .to_le_bytes(),
    );
    for value in values {
        hash.update(value.as_bytes());
    }
    Ok(())
}

fn dynamic_color_identity_digest(color: &DynamicLCColorState) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-dynamic-lc-color-identity-v1\0");
    hash.update(color.output_color_shape_id().to_le_bytes());
    hash.update(
        color
            .active_component_index()
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    hash.update(
        u64::try_from(color.components().len())
            .map_err(|_| invalid("dynamic LC color component count exceeds u64"))?
            .to_le_bytes(),
    );
    for component in color.components() {
        hash.update([component.kind() as u8]);
        hash.update(
            u64::try_from(component.source_slots().len())
                .map_err(|_| invalid("dynamic LC color word length exceeds u64"))?
                .to_le_bytes(),
        );
        for source_slot in component.source_slots() {
            hash.update(source_slot.to_le_bytes());
        }
    }
    hash.update(
        u64::try_from(color.result_port_bindings().len())
            .map_err(|_| invalid("dynamic LC color port count exceeds u64"))?
            .to_le_bytes(),
    );
    for binding in color.result_port_bindings() {
        hash.update(binding.component_index().to_le_bytes());
        hash.update([binding.endpoint() as u8]);
    }
    SemanticDigest::new(hash.finalize().into())
}

fn source_reflection_proof_root(
    source_slot: u32,
    source_seed_proof: SemanticDigest,
    result_color_identity: SemanticDigest,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-source-color-reflection-root-v1\0");
    hash.update(source_slot.to_le_bytes());
    hash.update(source_seed_proof.as_bytes());
    hash.update(result_color_identity.as_bytes());
    SemanticDigest::new(hash.finalize().into())
}

fn transition_reflection_proof_digest(
    phase: ExactComplexRational,
    semantic_digests: &[SemanticDigest],
    witness_digests: &[SemanticDigest],
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-transition-reflection-proof-v1\0");
    hash_exact_factor(&mut hash, phase);
    hash_digest_sequence(&mut hash, semantic_digests)?;
    hash_digest_sequence(&mut hash, witness_digests)?;
    SemanticDigest::new(hash.finalize().into())
}

fn current_reflection_proof_digest(
    phase: ExactComplexRational,
    lineage_roots: &[SemanticDigest],
    result_color_identity: SemanticDigest,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-current-reflection-proof-v1\0");
    hash_exact_factor(&mut hash, phase);
    hash_digest_sequence(&mut hash, lineage_roots)?;
    hash.update(result_color_identity.as_bytes());
    SemanticDigest::new(hash.finalize().into())
}

fn pending_reflection_certificate_digest(
    canonical_old_current_id: u32,
    reflected_old_current_id: u32,
    canonical: &CurrentReflectionProof,
    reflected: &CurrentReflectionProof,
    source_permutation: &[u32],
    fixed_point: bool,
    orbit_size: u32,
) -> RusticolResult<SemanticDigest> {
    pending_reflection_certificate_fields_digest(
        canonical_old_current_id,
        reflected_old_current_id,
        canonical.result_color_identity(),
        reflected.result_color_identity(),
        canonical.phase(),
        reflected.phase(),
        canonical.proof_digest(),
        reflected.proof_digest(),
        source_permutation,
        fixed_point,
        orbit_size,
    )
}

#[allow(clippy::too_many_arguments)]
fn pending_reflection_certificate_fields_digest(
    canonical_old_current_id: u32,
    reflected_old_current_id: u32,
    canonical_color_identity: SemanticDigest,
    reflected_color_identity: SemanticDigest,
    canonical_phase: ExactComplexRational,
    reflected_phase: ExactComplexRational,
    canonical_lineage_digest: SemanticDigest,
    reflected_lineage_digest: SemanticDigest,
    source_permutation: &[u32],
    fixed_point: bool,
    orbit_size: u32,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-pending-reflection-certificate-v1\0");
    hash.update(canonical_old_current_id.to_le_bytes());
    hash.update(reflected_old_current_id.to_le_bytes());
    hash.update(canonical_color_identity.as_bytes());
    hash.update(reflected_color_identity.as_bytes());
    hash_exact_factor(&mut hash, canonical_phase);
    hash_exact_factor(&mut hash, reflected_phase);
    hash.update(canonical_lineage_digest.as_bytes());
    hash.update(reflected_lineage_digest.as_bytes());
    hash.update(
        u64::try_from(source_permutation.len())
            .map_err(|_| invalid("reflection source permutation length exceeds u64"))?
            .to_le_bytes(),
    );
    for source_slot in source_permutation {
        hash.update(source_slot.to_le_bytes());
    }
    hash.update([u8::from(fixed_point)]);
    hash.update(orbit_size.to_le_bytes());
    SemanticDigest::new(hash.finalize().into())
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
    realized_pairing_rule_ids: BTreeSet<u32>,
    reflection: CurrentReflection,
    reflection_certificate_id: Option<u32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CurrentReflectionProof {
    phase: ExactComplexRational,
    lineage_roots: Box<[SemanticDigest]>,
    result_color_identity: SemanticDigest,
    proof_digest: SemanticDigest,
}

impl CurrentReflectionProof {
    fn new(
        phase: ExactComplexRational,
        lineage_roots: impl IntoIterator<Item = SemanticDigest>,
        result_color_identity: SemanticDigest,
    ) -> RusticolResult<Self> {
        if phase.is_zero() {
            return Err(invalid("current reflection proof has zero phase"));
        }
        let mut lineage_roots = lineage_roots.into_iter().collect::<Vec<_>>();
        lineage_roots.sort_unstable();
        lineage_roots.dedup();
        if lineage_roots.is_empty() {
            return Err(invalid("current reflection proof has empty lineage"));
        }
        let proof_digest =
            current_reflection_proof_digest(phase, &lineage_roots, result_color_identity)?;
        Ok(Self {
            phase,
            lineage_roots: lineage_roots.into_boxed_slice(),
            result_color_identity,
            proof_digest,
        })
    }

    const fn phase(&self) -> ExactComplexRational {
        self.phase
    }

    fn lineage_roots(&self) -> &[SemanticDigest] {
        &self.lineage_roots
    }

    const fn result_color_identity(&self) -> SemanticDigest {
        self.result_color_identity
    }

    const fn proof_digest(&self) -> SemanticDigest {
        self.proof_digest
    }

    fn merged_with(&self, other: &Self) -> RusticolResult<Option<Self>> {
        if self.phase != other.phase || self.result_color_identity != other.result_color_identity {
            return Ok(None);
        }
        Self::new(
            self.phase,
            self.lineage_roots
                .iter()
                .copied()
                .chain(other.lineage_roots.iter().copied()),
            self.result_color_identity,
        )
        .map(Some)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum CurrentReflection {
    Unavailable,
    Proven(CurrentReflectionProof),
}

impl CurrentReflection {
    fn phase(&self) -> Option<ExactComplexRational> {
        match self {
            Self::Unavailable => None,
            Self::Proven(proof) => Some(proof.phase()),
        }
    }

    fn proof(&self) -> Option<&CurrentReflectionProof> {
        match self {
            Self::Unavailable => None,
            Self::Proven(proof) => Some(proof),
        }
    }

    fn include(&mut self, candidate: Option<CurrentReflectionProof>) -> RusticolResult<()> {
        let merged = match (&*self, candidate) {
            (Self::Proven(existing), Some(candidate)) => existing.merged_with(&candidate)?,
            _ => None,
        };
        match merged {
            Some(proof) => *self = Self::Proven(proof),
            None => *self = Self::Unavailable,
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TransitionReflectionProof {
    phase: ExactComplexRational,
    semantic_digests: Box<[SemanticDigest]>,
    witness_digests: Box<[SemanticDigest]>,
    proof_digest: SemanticDigest,
}

impl TransitionReflectionProof {
    fn new(
        phase: ExactComplexRational,
        semantic_digests: impl IntoIterator<Item = SemanticDigest>,
        witness_digests: impl IntoIterator<Item = SemanticDigest>,
    ) -> RusticolResult<Self> {
        if phase.is_zero() {
            return Err(invalid("transition reflection proof has zero phase"));
        }
        let mut semantic_digests = semantic_digests.into_iter().collect::<Vec<_>>();
        semantic_digests.sort_unstable();
        semantic_digests.dedup();
        let mut witness_digests = witness_digests.into_iter().collect::<Vec<_>>();
        witness_digests.sort_unstable();
        witness_digests.dedup();
        if semantic_digests.is_empty() || witness_digests.is_empty() {
            return Err(invalid(
                "transition reflection proof requires semantic and witness digests",
            ));
        }
        let proof_digest =
            transition_reflection_proof_digest(phase, &semantic_digests, &witness_digests)?;
        Ok(Self {
            phase,
            semantic_digests: semantic_digests.into_boxed_slice(),
            witness_digests: witness_digests.into_boxed_slice(),
            proof_digest,
        })
    }

    const fn phase(&self) -> ExactComplexRational {
        self.phase
    }

    fn lineage_roots(&self) -> impl Iterator<Item = SemanticDigest> + '_ {
        self.semantic_digests
            .iter()
            .chain(self.witness_digests.iter())
            .copied()
            .chain(std::iter::once(self.proof_digest))
    }

    fn merged_with(&self, other: &Self) -> RusticolResult<Option<Self>> {
        if self.phase != other.phase {
            return Ok(None);
        }
        Self::new(
            self.phase,
            self.semantic_digests
                .iter()
                .copied()
                .chain(other.semantic_digests.iter().copied()),
            self.witness_digests
                .iter()
                .copied()
                .chain(other.witness_digests.iter().copied()),
        )
        .map(Some)
    }
}

/// Cold proof record emitted before folded stage-local current IDs are compacted.
///
/// This deliberately remains local until the orchestrator-owned program proof
/// API accepts reciprocal reflection orbits.
#[derive(Clone, Debug, Eq, PartialEq)]
struct PendingReflectionCertificate {
    id: u32,
    canonical_old_current_id: u32,
    reflected_old_current_id: u32,
    canonical_color_identity: SemanticDigest,
    reflected_color_identity: SemanticDigest,
    canonical_phase: ExactComplexRational,
    reflected_phase: ExactComplexRational,
    canonical_lineage_digest: SemanticDigest,
    reflected_lineage_digest: SemanticDigest,
    source_permutation: Box<[u32]>,
    fixed_point: bool,
    orbit_size: u32,
    proof_digest: SemanticDigest,
}

impl PendingReflectionCertificate {
    // The pair constructor authenticates both orbit members and both color
    // witnesses explicitly; retaining the parallel arguments makes swaps visible.
    #[allow(clippy::too_many_arguments)]
    fn reciprocal_pair(
        id: u32,
        canonical_old_current_id: u32,
        reflected_old_current_id: u32,
        canonical: &CurrentReflectionProof,
        reflected: &CurrentReflectionProof,
        canonical_color: &DynamicLCColorState,
        reflected_color: &DynamicLCColorState,
        source_count: usize,
    ) -> RusticolResult<Self> {
        if canonical_old_current_id == reflected_old_current_id
            || canonical.result_color_identity() == reflected.result_color_identity()
        {
            return Err(invalid(
                "folded reflection certificate requires two distinct orbit members",
            ));
        }
        if canonical.phase().checked_mul(reflected.phase())? != ExactComplexRational::ONE {
            return Err(invalid(
                "folded reflection certificate phases are not reciprocal",
            ));
        }
        let source_permutation =
            reflection_source_permutation(canonical_color, reflected_color, source_count)?;
        let proof_digest = pending_reflection_certificate_digest(
            canonical_old_current_id,
            reflected_old_current_id,
            canonical,
            reflected,
            &source_permutation,
            false,
            2,
        )?;
        Ok(Self {
            id,
            canonical_old_current_id,
            reflected_old_current_id,
            canonical_color_identity: canonical.result_color_identity(),
            reflected_color_identity: reflected.result_color_identity(),
            canonical_phase: canonical.phase(),
            reflected_phase: reflected.phase(),
            canonical_lineage_digest: canonical.proof_digest(),
            reflected_lineage_digest: reflected.proof_digest(),
            source_permutation: source_permutation.into_boxed_slice(),
            fixed_point: false,
            orbit_size: 2,
            proof_digest,
        })
    }
}

fn reflection_source_permutation(
    canonical_color: &DynamicLCColorState,
    reflected_color: &DynamicLCColorState,
    source_count: usize,
) -> RusticolResult<Vec<u32>> {
    let canonical = canonical_color
        .pure_adjoint_word()
        .ok_or_else(|| invalid("canonical reflection color has no adjoint word"))?;
    let reflected = reflected_color
        .pure_adjoint_word()
        .ok_or_else(|| invalid("reflected reflection color has no adjoint word"))?;
    if canonical.len() != reflected.len() {
        return Err(invalid(
            "reflection color words have inconsistent source counts",
        ));
    }
    let mut permutation = (0..source_count)
        .map(|slot| u32::try_from(slot).map_err(|_| invalid("reflection source count exceeds u32")))
        .collect::<RusticolResult<Vec<_>>>()?;
    for (source, target) in canonical.iter().copied().zip(reflected.iter().copied()) {
        let destination = permutation
            .get_mut(source as usize)
            .ok_or_else(|| invalid("reflection word references an absent source slot"))?;
        *destination = target;
    }
    let mut ordered = permutation.clone();
    ordered.sort_unstable();
    if ordered
        != (0..source_count)
            .map(|slot| u32::try_from(slot).expect("source count checked above"))
            .collect::<Vec<_>>()
    {
        return Err(invalid(
            "reflection source mapping is not a complete permutation",
        ));
    }
    Ok(permutation)
}

fn current_color_for_index<'a>(
    color_states: &'a DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    current_index: usize,
) -> RusticolResult<&'a DynamicLCColorState> {
    let current = currents
        .get(current_index)
        .ok_or_else(|| invalid("reflection current index is absent"))?;
    color_states
        .get(current.key.dynamic_lc_color_state_id())
        .ok_or_else(|| invalid("reflection current color disappeared"))
}

fn validate_pending_reflection_certificates(
    certificates: &[PendingReflectionCertificate],
) -> RusticolResult<()> {
    for (index, certificate) in certificates.iter().enumerate() {
        if certificate.id != index as u32 {
            return Err(invalid(format!(
                "pending reflection certificate row {index} has non-dense id {}",
                certificate.id
            )));
        }
        if certificate.canonical_old_current_id == certificate.reflected_old_current_id
            || certificate.canonical_color_identity == certificate.reflected_color_identity
        {
            return Err(invalid(format!(
                "pending reflection certificate {index} does not identify two orbit members"
            )));
        }
        if certificate.fixed_point || certificate.orbit_size != 2 {
            return Err(invalid(format!(
                "pending folded reflection certificate {index} is not a reciprocal two-cycle"
            )));
        }
        if certificate
            .canonical_phase
            .checked_mul(certificate.reflected_phase)?
            != ExactComplexRational::ONE
        {
            return Err(invalid(format!(
                "pending reflection certificate {index} phases are not reciprocal"
            )));
        }
        let expected = pending_reflection_certificate_fields_digest(
            certificate.canonical_old_current_id,
            certificate.reflected_old_current_id,
            certificate.canonical_color_identity,
            certificate.reflected_color_identity,
            certificate.canonical_phase,
            certificate.reflected_phase,
            certificate.canonical_lineage_digest,
            certificate.reflected_lineage_digest,
            &certificate.source_permutation,
            certificate.fixed_point,
            certificate.orbit_size,
        )?;
        if expected != certificate.proof_digest {
            return Err(invalid(format!(
                "pending reflection certificate {index} proof digest is stale"
            )));
        }
    }
    Ok(())
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

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PendingThreeLineTraversalCertificate {
    sector_id: u32,
    kind: u8,
    sink_block_ordinal: u32,
    reference_block_order: Box<[u32]>,
    witness_block_order: Box<[u32]>,
    block_permutation: Box<[u32]>,
    reference_source_order: Box<[u32]>,
    witness_source_order: Box<[u32]>,
    source_position_permutation: Box<[u32]>,
    closure_anchor_source_slot: u32,
    pairing_rule_id: u32,
    proof_digest: SemanticDigest,
}

/// One compiler/model-certified closure witness before equal runtime rows are
/// combined. These records are cold proof data and never enter the numeric
/// Direct-Arena loop.
#[derive(Clone, Debug, Eq, PartialEq)]
struct PendingClosureProofContribution {
    construction_parent_ids: [u32; 2],
    construction_parent_permutation: [u32; 2],
    reconstruction_parent_permutation: [u32; 2],
    evaluator_parent_permutation: [u32; 2],
    closure_template_semantic_digest: SemanticDigest,
    color_witness_term_id: u32,
    color_witness_proof_digest: SemanticDigest,
    three_line_certificate: Option<PendingThreeLineTraversalCertificate>,
    pairing_certificate_ids: Box<[u32]>,
    reflection_certificate_id: Option<u32>,
    exact_factor: ExactComplexRational,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PendingClosureGroup {
    contributions: Vec<PendingClosureProofContribution>,
    exact_factor: ExactComplexRational,
}

impl Default for PendingClosureGroup {
    fn default() -> Self {
        Self {
            contributions: Vec::new(),
            exact_factor: ExactComplexRational::ZERO,
        }
    }
}

impl PendingClosureGroup {
    fn include(&mut self, contribution: PendingClosureProofContribution) -> RusticolResult<()> {
        aggregate_factor(&mut self.exact_factor, contribution.exact_factor)?;
        self.contributions.push(contribution);
        Ok(())
    }
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
                .map_err(|_| invalid("direct recurrence requires binary prepared transitions"))?;
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

#[derive(Debug, Default)]
struct TransitionReflectionIndex {
    proofs_by_transition: BTreeMap<u32, TransitionReflectionProof>,
}

impl TransitionReflectionIndex {
    fn new(
        template: &OwnedRecurrenceTemplateInput,
        catalog: &TemplateCatalog<'_>,
    ) -> RusticolResult<Self> {
        let transitions_by_subject = template
            .transitions
            .iter()
            .map(|row| (row.template_string_id, row.id))
            .collect::<BTreeMap<_, _>>();
        let mut result = Self::default();
        for proof in &template.symmetry_proofs {
            if catalog.string(proof.proof_algorithm_string_id, "symmetry proof algorithm")?
                != "canonical-current-word-reversal-v1"
            {
                continue;
            }
            let subjects = catalog.u32_sequence(
                proof.subject_template_sequence_id,
                "current-word reversal proof subjects",
            )?;
            let [subject] = subjects else {
                return Err(invalid(
                    "current-word reversal proof must reference one transition",
                ));
            };
            let transition_id = transitions_by_subject
                .get(subject)
                .copied()
                .ok_or_else(|| {
                    invalid("current-word reversal proof subject is not a transition")
                })?;
            if catalog.u32_sequence(
                proof.input_permutation_sequence_id,
                "current-word reversal proof permutation",
            )? != [1, 0]
            {
                return Err(invalid(
                    "current-word reversal proof must exchange two transition inputs",
                ));
            }
            let phase =
                catalog.factor(proof.exact_phase_factor_id, "current-word reversal proof")?;
            let candidate = TransitionReflectionProof::new(
                phase,
                [catalog.digest(
                    proof.semantic_digest_id,
                    "current-word reversal semantic proof",
                )?],
                [catalog.digest(
                    proof.witness_digest_id,
                    "current-word reversal witness proof",
                )?],
            )?;
            match result.proofs_by_transition.entry(transition_id) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(candidate);
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let Some(merged) = entry.get().merged_with(&candidate)? else {
                        return Err(invalid(
                            "transition has conflicting current-word reversal phases",
                        ));
                    };
                    entry.insert(merged);
                }
            }
        }
        Ok(result)
    }

    fn proof(&self, transition_id: u32) -> Option<&TransitionReflectionProof> {
        self.proofs_by_transition.get(&transition_id)
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
/// one materialized representative sector. Pure-adjoint reversed words are
/// admitted transiently for one construction stage so reflection proofs can be
/// finalized over the complete fan-in. A stage-final reconciliation then
/// removes only exactly certified aliases. This cheap forward filter prevents
/// the builder from interning unrelated color words; final backward liveness
/// remains authoritative.
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

    fn accepts_up_to_reflection(&self, state: &DynamicLCColorState) -> RusticolResult<bool> {
        if self.accepts(state) {
            return Ok(true);
        }
        if state.pure_adjoint_word().is_none() {
            return Ok(false);
        }
        Ok(self.accepts(&state.reversed()?))
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

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct StructuralState {
    state_template_id: u32,
    spin_state_class: i32,
}

impl StructuralState {
    const fn new(state_template_id: u32, spin_state_class: i32) -> Self {
        Self {
            state_template_id,
            spin_state_class,
        }
    }

    fn from_current(current: &PendingCurrent) -> Self {
        Self::new(
            current.key.current_state_template_id(),
            current.key.spin_state_class(),
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct StructuralTransition {
    parents: [StructuralState; 2],
    result: StructuralState,
}

fn structural_state_matches(required: StructuralState, actual: StructuralState) -> bool {
    required.state_template_id == actual.state_template_id
        && (required.spin_state_class == actual.spin_state_class
            || actual.spin_state_class == DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS)
}

fn structural_parent_states_match(
    required: [StructuralState; 2],
    actual: [StructuralState; 2],
) -> bool {
    structural_state_matches(required[0], actual[0])
        && structural_state_matches(required[1], actual[1])
}

/// A cheap closure-rooted state/spin grammar used before materializing full
/// current, color, helicity-ancestry, and contribution objects.
///
/// The forward half records only structurally feasible `(support, state,
/// spin)` tuples. The backward half starts from every model-certified physical
/// closure and retains the tuples that can reach one of those roots. This is a
/// conservative exact filter: color, coupling, flavour, pairing, and proof
/// contracts remain authenticated by normal construction, while a full
/// current is never interned for a state/spin branch that cannot close.
#[derive(Debug)]
struct StructuralDemandIndex {
    demanded: BTreeSet<(Vec<u32>, StructuralState)>,
}

impl StructuralDemandIndex {
    fn new(
        process: &OwnedRecurrenceProcessInput,
        template: &OwnedRecurrenceTemplateInput,
        catalog: &TemplateCatalog<'_>,
        materialized_sectors: &BTreeSet<u32>,
        source_currents: &[PendingCurrent],
    ) -> RusticolResult<Self> {
        let transitions = structural_transitions(template, catalog)?;
        let mut feasible = BTreeMap::<Vec<u32>, BTreeSet<StructuralState>>::new();
        for current in source_currents
            .iter()
            .filter(|current| current.key.node_kind() == RecurrenceNodeKind::Source)
        {
            feasible
                .entry(current.key.support_source_slots().to_vec())
                .or_default()
                .insert(StructuralState::from_current(current));
        }
        if feasible.is_empty() {
            return Err(invalid(
                "recurrence structural-demand construction has no source states",
            ));
        }

        let source_count = process.external_legs.len();
        for target_size in 2..source_count {
            let prior = feasible
                .iter()
                .filter(|(support, _)| support.len() < target_size)
                .map(|(support, states)| {
                    (support.clone(), states.iter().copied().collect::<Vec<_>>())
                })
                .collect::<Vec<_>>();
            let mut additions = BTreeMap::<Vec<u32>, BTreeSet<StructuralState>>::new();
            for left_index in 0..prior.len() {
                for right_index in left_index + 1..prior.len() {
                    let (left_support, left_states) = &prior[left_index];
                    let (right_support, right_states) = &prior[right_index];
                    if left_support.len() + right_support.len() != target_size
                        || !disjoint_support(left_support, right_support)
                    {
                        continue;
                    }
                    let support = merged_support(left_support, right_support)?;
                    for left in left_states.iter().copied() {
                        for right in right_states.iter().copied() {
                            for transition in &transitions {
                                if structural_parent_states_match(transition.parents, [left, right])
                                    || structural_parent_states_match(
                                        transition.parents,
                                        [right, left],
                                    )
                                {
                                    additions
                                        .entry(support.clone())
                                        .or_default()
                                        .insert(transition.result);
                                }
                            }
                        }
                    }
                }
            }
            for (support, states) in additions {
                feasible.entry(support).or_default().extend(states);
            }
        }

        let full_support = (0..source_count as u32).collect::<Vec<_>>();
        let mut demanded = BTreeSet::<(Vec<u32>, StructuralState)>::new();
        for sector_id in materialized_sectors.iter().copied() {
            let sector = process
                .physical_lc_sectors
                .get(sector_id as usize)
                .copied()
                .ok_or_else(|| invalid("structural-demand LC sector is absent"))?;
            let anchor_support = vec![sector.closure_source_slot];
            let complement_support = full_support
                .iter()
                .copied()
                .filter(|slot| *slot != sector.closure_source_slot)
                .collect::<Vec<_>>();
            let anchor_states = feasible
                .get(&anchor_support)
                .ok_or_else(|| invalid("structural-demand closure anchor is infeasible"))?;
            let complement_states = feasible
                .get(&complement_support)
                .ok_or_else(|| invalid("structural-demand closure complement is infeasible"))?;
            let mut sector_has_root = false;
            for anchor in anchor_states.iter().copied() {
                for complement in complement_states.iter().copied() {
                    for closure in &template.closures {
                        let closure_states = catalog.u32_sequence(
                            closure.input_state_sequence_id,
                            "structural-demand closure input states",
                        )?;
                        if closure_states.len() != 2 {
                            return Err(invalid(
                                "direct recurrence requires binary prepared closures",
                            ));
                        }
                        let ordered = if closure_states
                            == [complement.state_template_id, anchor.state_template_id]
                        {
                            Some([complement, anchor])
                        } else if closure_states
                            == [anchor.state_template_id, complement.state_template_id]
                            && anchor.state_template_id != complement.state_template_id
                        {
                            Some([anchor, complement])
                        } else {
                            None
                        };
                        let Some(ordered) = ordered else {
                            continue;
                        };
                        if structural_closure_admits(*closure, ordered, template, catalog)? {
                            demanded.insert((complement_support.clone(), complement));
                            sector_has_root = true;
                        }
                    }
                }
            }
            if !sector_has_root {
                return Err(invalid(format!(
                    "recurrence structural-demand grammar found no closure root for LC sector {sector_id}"
                )));
            }
        }

        for target_size in (2..source_count).rev() {
            let targets = demanded
                .iter()
                .filter(|(support, _)| support.len() == target_size)
                .cloned()
                .collect::<Vec<_>>();
            for (target_support, target_state) in targets {
                let feasible_rows = feasible.iter().collect::<Vec<_>>();
                for left_index in 0..feasible_rows.len() {
                    for right_index in left_index + 1..feasible_rows.len() {
                        let (left_support, left_states) = feasible_rows[left_index];
                        let (right_support, right_states) = feasible_rows[right_index];
                        if left_support.len() + right_support.len() != target_size
                            || !disjoint_support(left_support, right_support)
                            || merged_support(left_support, right_support)? != target_support
                        {
                            continue;
                        }
                        for left in left_states.iter().copied() {
                            for right in right_states.iter().copied() {
                                for transition in transitions
                                    .iter()
                                    .filter(|transition| transition.result == target_state)
                                {
                                    if structural_parent_states_match(
                                        transition.parents,
                                        [left, right],
                                    ) || structural_parent_states_match(
                                        transition.parents,
                                        [right, left],
                                    ) {
                                        demanded.insert((left_support.clone(), left));
                                        demanded.insert((right_support.clone(), right));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(Self { demanded })
    }

    fn accepts(
        &self,
        support_source_slots: &[u32],
        state_template_id: u32,
        spin_state_class: i32,
    ) -> bool {
        self.demanded.contains(&(
            support_source_slots.to_vec(),
            StructuralState::new(state_template_id, spin_state_class),
        ))
    }
}

fn structural_transitions(
    template: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
) -> RusticolResult<Vec<StructuralTransition>> {
    let transitions = template
        .transitions
        .iter()
        .map(|transition| {
            let quantum = template
                .quantum_flows
                .get(transition.quantum_flow_template_id as usize)
                .copied()
                .ok_or_else(|| invalid("structural-demand quantum flow is absent"))?;
            let states =
                catalog.u32_sequence(quantum.input_state_sequence_id, "quantum input states")?;
            let spins =
                catalog.i32_sequence(quantum.input_spin_sequence_id, "quantum input spins")?;
            if states.len() != 2 || spins.len() != 2 {
                return Err(invalid(
                    "direct recurrence requires binary quantum-flow contracts",
                ));
            }
            if quantum.result_state_template_id != transition.result_state_template_id {
                return Err(invalid(
                    "structural-demand transition and quantum-flow result states differ",
                ));
            }
            Ok(StructuralTransition {
                parents: [
                    StructuralState::new(states[0], spins[0]),
                    StructuralState::new(states[1], spins[1]),
                ],
                result: StructuralState::new(
                    quantum.result_state_template_id,
                    quantum.result_spin_state,
                ),
            })
        })
        .collect::<RusticolResult<BTreeSet<_>>>()?;
    Ok(transitions.into_iter().collect())
}

fn structural_closure_admits(
    closure: ClosureRow,
    ordered_parents: [StructuralState; 2],
    template: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
) -> RusticolResult<bool> {
    let eligible = catalog.u32_sequence(
        closure.eligible_quantum_flow_sequence_id,
        "structural-demand closure quantum flows",
    )?;
    if eligible.is_empty() {
        return Ok(true);
    }
    for quantum_id in eligible {
        let quantum = template
            .quantum_flows
            .get(*quantum_id as usize)
            .copied()
            .ok_or_else(|| invalid("structural-demand closure quantum flow is absent"))?;
        let states =
            catalog.u32_sequence(quantum.input_state_sequence_id, "quantum input states")?;
        let spins = catalog.i32_sequence(quantum.input_spin_sequence_id, "quantum input spins")?;
        if states.len() != 2 || spins.len() != 2 {
            return Err(invalid(
                "direct recurrence requires binary quantum-flow contracts",
            ));
        }
        if structural_parent_states_match(
            [
                StructuralState::new(states[0], spins[0]),
                StructuralState::new(states[1], spins[1]),
            ],
            ordered_parents,
        ) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn compatible_pairing_rules_for_current(
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    template: &OwnedRecurrenceTemplateInput,
    current_state_template_id: u32,
    support_source_slots: &[u32],
) -> RusticolResult<BTreeSet<u32>> {
    let Some(pairing_catalog) = pairing_catalog else {
        return Ok(BTreeSet::new());
    };
    let state = template
        .current_states
        .get(current_state_template_id as usize)
        .ok_or_else(|| invalid("pairing support references an absent current state"))?;
    let carries_colored_fermion_line = state.statistics == 1 && state.color_representation != 1;
    let support = support_source_slots
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let mut result = BTreeSet::new();
    for rule in pairing_catalog.rules().iter().copied() {
        let crossing_count = pairing_catalog
            .endpoint_pairings(rule)?
            .iter()
            .filter(|pair| {
                support.contains(&pair.fundamental_source_slot)
                    != support.contains(&pair.antifundamental_source_slot)
            })
            .count();
        let compatible = if carries_colored_fermion_line {
            crossing_count == 1
        } else {
            crossing_count == 0
        };
        if compatible {
            result.insert(rule.rule_id);
        }
    }
    Ok(result)
}

fn realized_pairing_rules_for_transition(
    compatible_result_rules: BTreeSet<u32>,
    parent_rules: [&BTreeSet<u32>; 2],
) -> BTreeSet<u32> {
    compatible_result_rules
        .into_iter()
        .filter(|rule_id| parent_rules[0].contains(rule_id) && parent_rules[1].contains(rule_id))
        .collect()
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
    let pairing_catalog = authenticated.process().fermion_pairing_catalog();
    let template_input = authenticated.template().input();
    let process_catalog = ProcessCatalog::new(process_input)?;
    let helicity_support_rule = helicity_support_rule(authenticated)?;
    let retained_helicity_count = retained_helicity_count(process_input)?;
    let template_catalog = TemplateCatalog::new(template_input)?;
    let coupling_limits = coupling_limits(&process_catalog, &template_catalog)?;
    let propagators = propagator_by_state(template_input)?;
    let transition_reflections = TransitionReflectionIndex::new(template_input, &template_catalog)?;
    let replay_targets = build_replay_targets(strategy, process_input, &process_catalog)?;
    let materialized_sectors = materialized_sector_ids(strategy, process_input, &replay_targets);
    let color_targets =
        MaterializedColorTargets::new(&materialized_sectors, process_input, &process_catalog)?;
    let mut color_states = DynamicLCColorStateInterner::default();
    let mut currents = Vec::<PendingCurrent>::new();
    let mut current_ids = BTreeMap::<CurrentCoreKey, u32>::new();
    let mut currents_by_size = vec![Vec::<u32>::new(); process_input.external_legs.len()];
    let mut reflection_certificates = Vec::<PendingReflectionCertificate>::new();

    build_sources(
        strategy,
        catalog_digest,
        process_input,
        &process_catalog,
        pairing_catalog,
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
    let structural_demands = StructuralDemandIndex::new(
        process_input,
        template_input,
        &template_catalog,
        &materialized_sectors,
        &currents,
    )?;
    let stage_diagnostics = build_internal_currents(
        catalog_digest,
        process_input,
        pairing_catalog,
        template_input,
        &template_catalog,
        &transition_reflections,
        &coupling_limits,
        &propagators,
        &color_targets,
        &structural_demands,
        &mut color_states,
        &mut currents,
        &mut current_ids,
        &mut currents_by_size,
        &mut reflection_certificates,
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
        strategy,
        process_input,
        &process_catalog,
        pairing_catalog,
        template_input,
        &template_catalog,
        &color_states,
        &currents,
        &materialized_sectors,
        &stage_diagnostics,
        &reflection_certificates,
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
        reflection_certificates,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_sources(
    strategy: RecurrenceStrategy,
    catalog_digest: SemanticDigest,
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
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
            RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::ContractedColorUnion => {
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
                    let reflection = source_reflection(
                        color_states,
                        color_id,
                        leg.source_slot,
                        template_catalog.source_seed(source)?.proof_digest(),
                    )?;
                    let helicity_identity = match strategy {
                        RecurrenceStrategy::TopologyReplay => {
                            CurrentHelicityIdentity::topology_replay(
                                process_state.spin_state,
                                vec![SourceStateAssignment::new(
                                    leg.source_slot,
                                    process_state.state_index,
                                )],
                            )?
                        }
                        RecurrenceStrategy::ContractedColorUnion => {
                            CurrentHelicityIdentity::contracted_color_union(
                                process_state.spin_state,
                                vec![SourceStateAssignment::new(
                                    leg.source_slot,
                                    process_state.state_index,
                                )],
                            )?
                        }
                        RecurrenceStrategy::AllFlowUnion => unreachable!(),
                    };
                    insert_source_current(
                        catalog_digest,
                        process_state.current_state_template_id,
                        color_id,
                        leg.source_slot,
                        process_state.momentum_sign,
                        helicity_identity,
                        source,
                        reflection,
                        CurrentSourceBinding::FixedTemplate(process_state.source_template_id),
                        Some(process_catalog.factor(
                            process_state.crossing_phase_factor_id,
                            "source crossing phase",
                        )?),
                        &zero_orders,
                        pairing_catalog,
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
                    let reflection = source_reflection(
                        color_states,
                        color_id,
                        leg.source_slot,
                        template_catalog.source_seed(source)?.proof_digest(),
                    )?;
                    insert_source_current(
                        catalog_digest,
                        group.full_state_template_id,
                        color_id,
                        leg.source_slot,
                        group.momentum_sign,
                        CurrentHelicityIdentity::all_flow_union(group.full_spin_state_class),
                        source,
                        reflection,
                        CurrentSourceBinding::runtime_dispatch_with_variants(
                            group.contract_id,
                            group.variants,
                        )?,
                        None,
                        &zero_orders,
                        pairing_catalog,
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
    union_spin_state_class(contract.id, &result_spins)
}

fn union_spin_state_class(contract_id: u32, result_spins: &BTreeSet<i32>) -> RusticolResult<i32> {
    if result_spins.is_empty() {
        return Err(invalid(format!(
            "runtime-helicity contract {} has no full-state spin class",
            contract_id
        )));
    }
    if result_spins.contains(&DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS) {
        return Err(invalid(format!(
            "runtime-helicity contract {} contains the reserved dynamic source-spin sentinel",
            contract_id
        )));
    }
    Ok(if result_spins.len() == 1 {
        *result_spins.iter().next().expect("checked nonempty")
    } else {
        // One all-flow source current stores the full runtime-helicity vector.
        // Its selected chiral embedding is populated at execution time, so a
        // static source spin class would incorrectly discard the other
        // certified branch during recurrence construction.
        DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS
    })
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
    reflection: CurrentReflection,
    source_binding: CurrentSourceBinding,
    source_factor: Option<ExactComplexRational>,
    zero_orders: &[u32],
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
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
    let realized_pairing_rule_ids = compatible_pairing_rules_for_current(
        pairing_catalog,
        template_catalog.input,
        current_state_template_id,
        &[source_slot],
    )?;
    if let Some(existing) = current_ids.get(&key).copied() {
        if currents[existing as usize].source_exact_factor != source_factor {
            return Err(invalid(
                "equivalent source currents have different exact factors",
            ));
        }
        currents[existing as usize]
            .reflection
            .include(reflection.proof().cloned())?;
        currents[existing as usize]
            .realized_pairing_rule_ids
            .extend(realized_pairing_rule_ids);
        return Ok(());
    }
    let id = u32::try_from(currents.len())
        .map_err(|_| invalid("recurrence current count exceeds u32"))?;
    current_ids.insert(key.clone(), id);
    currents.push(PendingCurrent {
        key,
        source_exact_factor: source_factor,
        contributions: BTreeMap::new(),
        realized_pairing_rule_ids,
        reflection,
        reflection_certificate_id: None,
    });
    currents_by_size[0].push(id);
    Ok(())
}

fn source_reflection(
    color_states: &DynamicLCColorStateInterner,
    color_id: super::DynamicLCColorStateId,
    source_slot: u32,
    source_seed_proof: SemanticDigest,
) -> RusticolResult<CurrentReflection> {
    let color = color_states
        .get(color_id)
        .ok_or_else(|| invalid("source dynamic color state disappeared"))?;
    Ok(if color.pure_adjoint_word().is_some() {
        let color_identity = dynamic_color_identity_digest(color)?;
        let root = source_reflection_proof_root(source_slot, source_seed_proof, color_identity)?;
        CurrentReflection::Proven(CurrentReflectionProof::new(
            ExactComplexRational::ONE,
            [root],
            color_identity,
        )?)
    } else {
        CurrentReflection::Unavailable
    })
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
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    template: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    transition_reflections: &TransitionReflectionIndex,
    coupling_limits: &[Option<u32>],
    propagators: &BTreeMap<u32, Option<u32>>,
    color_targets: &MaterializedColorTargets,
    structural_demands: &StructuralDemandIndex,
    color_states: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
    currents_by_size: &mut [Vec<u32>],
    reflection_certificates: &mut Vec<PendingReflectionCertificate>,
    phase_total: usize,
    progress: &mut dyn FnMut(RecurrenceBuildProgress) -> RusticolResult<()>,
) -> RusticolResult<Vec<StageConstructionDiagnostics>> {
    let transition_index = TransitionStateIndex::new(template, catalog)?;
    let mut diagnostics = Vec::new();
    let stage_total = process.external_legs.len().saturating_sub(2);
    let mut completed_contribution_count = 0usize;
    let mut completed_color_target_prune_count = 0usize;
    for target_size in 2..process.external_legs.len() {
        let stage_current_start = currents.len();
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
                    pairing_catalog,
                    template,
                    catalog,
                    transition_reflections,
                    coupling_limits,
                    propagators,
                    color_targets,
                    structural_demands,
                    color_states,
                    currents,
                    current_ids,
                    target_bucket,
                    &mut stage,
                )?;
            }
        }
        reconcile_stage_reflections(
            stage_current_start,
            color_states,
            currents,
            current_ids,
            target_bucket,
            reflection_certificates,
            process.external_legs.len(),
        )?;
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
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    template: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    transition_reflections: &TransitionReflectionIndex,
    coupling_limits: &[Option<u32>],
    propagators: &BTreeMap<u32, Option<u32>>,
    color_targets: &MaterializedColorTargets,
    structural_demands: &StructuralDemandIndex,
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
    let support = merged_support(
        parents[0].support_source_slots(),
        parents[1].support_source_slots(),
    )?;
    let helicity_identity = merged_helicity_identity(
        parents[0].helicity_identity(),
        parents[1].helicity_identity(),
        quantum.result_spin_state,
    )?;
    if !structural_demands.accepts(
        &support,
        transition.result_state_template_id,
        helicity_identity.spin_state_class(),
    ) {
        return Ok(());
    }
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
    let parent_reflections = [
        currents[parent_ids[0] as usize].reflection.clone(),
        currents[parent_ids[1] as usize].reflection.clone(),
    ];
    let parent_colors = [
        color_states
            .get(parents[0].dynamic_lc_color_state_id())
            .ok_or_else(|| invalid("left dynamic color state disappeared"))?
            .clone(),
        color_states
            .get(parents[1].dynamic_lc_color_state_id())
            .ok_or_else(|| invalid("right dynamic color state disappeared"))?
            .clone(),
    ];
    let reversal_masks = current_reversal_masks(&parent_colors, &parent_reflections);
    let local_reflection_proof = transition_reflections.proof(transition.id);

    for witness_row in catalog.witness_rows(transition.color_contraction_template_id)? {
        if witness_row.left_shape_string_id != parent_colors[0].output_color_shape_id()
            || witness_row.right_shape_string_id != parent_colors[1].output_color_shape_id()
        {
            continue;
        }
        checked_diagnostic_add(
            &mut diagnostics.color_shape_match_count,
            1,
            "recurrence color-shape-match count",
        )?;
        let witness = catalog.witness(*witness_row)?;
        for reversal_mask in reversal_masks.iter().copied() {
            let mut variant_colors = parent_colors.clone();
            let mut reversal_factor = ExactComplexRational::ONE;
            for index in 0..2 {
                if reversal_mask & (1 << index) == 0 {
                    continue;
                }
                variant_colors[index] = variant_colors[index].reversed()?;
                reversal_factor = reversal_factor.checked_mul(
                    parent_reflections[index]
                        .phase()
                        .expect("reversal mask requires a proven parent phase"),
                )?;
            }
            let Some(result_color) = witness.apply(&variant_colors[0], &variant_colors[1])? else {
                continue;
            };
            checked_diagnostic_add(
                &mut diagnostics.color_result_count,
                1,
                "recurrence color-result count",
            )?;
            let result_reflection = current_reflection_candidate(
                &result_color,
                &parent_reflections,
                local_reflection_proof,
            )?;
            if !color_targets.accepts_up_to_reflection(&result_color)? {
                checked_diagnostic_add(
                    &mut diagnostics.color_target_prune_count,
                    1,
                    "recurrence color-target prune count",
                )?;
                continue;
            }
            let result_color_id = color_states.intern(result_color)?;
            let result_flavour_flow = quantum_flow_result_flavour(quantum, &parents, catalog)?;
            let key = CurrentCoreKey::new(
                catalog_digest,
                RecurrenceNodeKind::Current,
                transition.result_state_template_id,
                result_color_id,
                support.clone(),
                merged_momentum(parents[0].momentum(), parents[1].momentum())?,
                helicity_identity.clone(),
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
            let realized_pairing_rule_ids = realized_pairing_rules_for_transition(
                compatible_pairing_rules_for_current(
                    pairing_catalog,
                    template,
                    transition.result_state_template_id,
                    key.support_source_slots(),
                )?,
                [
                    &currents[parent_ids[0] as usize].realized_pairing_rule_ids,
                    &currents[parent_ids[1] as usize].realized_pairing_rule_ids,
                ],
            );
            let result_id = if let Some(id) = current_ids.get(&key).copied() {
                currents[id as usize]
                    .reflection
                    .include(result_reflection)?;
                currents[id as usize]
                    .realized_pairing_rule_ids
                    .extend(realized_pairing_rule_ids);
                id
            } else {
                let id = u32::try_from(currents.len())
                    .map_err(|_| invalid("recurrence current count exceeds u32"))?;
                current_ids.insert(key.clone(), id);
                currents.push(PendingCurrent {
                    key,
                    source_exact_factor: None,
                    contributions: BTreeMap::new(),
                    realized_pairing_rule_ids,
                    reflection: result_reflection
                        .map_or(CurrentReflection::Unavailable, CurrentReflection::Proven),
                    reflection_certificate_id: None,
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
            let factor = base_factor
                .checked_mul(witness.exact_factor())?
                .checked_mul(reversal_factor)?;
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
    }
    Ok(())
}

fn current_key_with_dynamic_color(
    key: &CurrentCoreKey,
    dynamic_lc_color_state_id: DynamicLCColorStateId,
) -> RusticolResult<CurrentCoreKey> {
    CurrentCoreKey::new(
        key.catalog_digest(),
        key.node_kind(),
        key.current_state_template_id(),
        dynamic_lc_color_state_id,
        key.support_source_slots().to_vec(),
        key.momentum().clone(),
        key.helicity_identity().clone(),
        key.flavour_flow().to_vec(),
        key.quantum_number_flow_id(),
        key.coupling_orders().to_vec(),
        key.source_binding().clone(),
        key.propagator_template_id(),
    )
}

fn reciprocal_reflection_proof(
    left: &CurrentReflection,
    right: &CurrentReflection,
) -> RusticolResult<bool> {
    let (Some(left), Some(right)) = (left.phase(), right.phase()) else {
        return Ok(false);
    };
    Ok(left.checked_mul(right)? == ExactComplexRational::ONE)
}

/// Finalize pure-adjoint reflection aliases after every contribution to the
/// stage's currents has been seen.
///
/// A contribution that lacks a reflection proof downgrades the whole current.
/// Both orientations are then retained as exact residual states. Only a pair
/// with reciprocal, complete fan-in proofs is compacted to its canonical
/// orientation. Stage currents cannot depend on one another, so compacting the
/// stage tail cannot invalidate any stored parent current ID.
fn reconcile_stage_reflections(
    stage_start: usize,
    color_states: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
    target_bucket: &mut Vec<u32>,
    reflection_certificates: &mut Vec<PendingReflectionCertificate>,
    source_count: usize,
) -> RusticolResult<()> {
    if stage_start > currents.len() {
        return Err(invalid(
            "recurrence reflection stage starts beyond current storage",
        ));
    }
    let stage_end = currents.len();
    let mut visited = vec![false; stage_end - stage_start];
    let mut prune = vec![false; stage_end - stage_start];

    for current_index in stage_start..stage_end {
        let local_index = current_index - stage_start;
        if visited[local_index] {
            continue;
        }
        let key = currents[current_index].key.clone();
        let color = color_states
            .get(key.dynamic_lc_color_state_id())
            .ok_or_else(|| invalid("reflection current color state disappeared"))?
            .clone();
        let Some(word) = color.pure_adjoint_word() else {
            continue;
        };
        if word.len() < 2 {
            continue;
        }
        let canonical = pure_adjoint_word_is_canonical(word);
        let reversed_color_id = color_states.intern(color.reversed()?)?;
        if reversed_color_id == key.dynamic_lc_color_state_id() {
            currents[current_index].reflection = CurrentReflection::Unavailable;
            continue;
        }
        let reversed_key = current_key_with_dynamic_color(&key, reversed_color_id)?;
        let Some(reversed_id) = current_ids.get(&reversed_key).copied() else {
            currents[current_index].reflection = CurrentReflection::Unavailable;
            continue;
        };
        let reversed_index = reversed_id as usize;
        if !(stage_start..stage_end).contains(&reversed_index) {
            currents[current_index].reflection = CurrentReflection::Unavailable;
            continue;
        }
        let reversed_local_index = reversed_index - stage_start;
        visited[local_index] = true;
        visited[reversed_local_index] = true;

        let reversed_color = color_states
            .get(reversed_color_id)
            .ok_or_else(|| invalid("reversed reflection color state disappeared"))?;
        let Some(reversed_word) = reversed_color.pure_adjoint_word() else {
            currents[current_index].reflection = CurrentReflection::Unavailable;
            currents[reversed_index].reflection = CurrentReflection::Unavailable;
            continue;
        };
        let reversed_canonical = pure_adjoint_word_is_canonical(reversed_word);
        let proof_is_complete = reciprocal_reflection_proof(
            &currents[current_index].reflection,
            &currents[reversed_index].reflection,
        )?;
        if !proof_is_complete || canonical == reversed_canonical {
            currents[current_index].reflection = CurrentReflection::Unavailable;
            currents[reversed_index].reflection = CurrentReflection::Unavailable;
            continue;
        }
        let (canonical_index, reflected_index, pruned_local_index) = if canonical {
            (current_index, reversed_index, reversed_local_index)
        } else {
            (reversed_index, current_index, local_index)
        };
        let canonical_proof = currents[canonical_index]
            .reflection
            .proof()
            .cloned()
            .ok_or_else(|| invalid("canonical reflection proof disappeared"))?;
        let reflected_proof = currents[reflected_index]
            .reflection
            .proof()
            .cloned()
            .ok_or_else(|| invalid("reflected reflection proof disappeared"))?;
        for (label, current_index, proof) in [
            ("canonical", canonical_index, &canonical_proof),
            ("reflected", reflected_index, &reflected_proof),
        ] {
            let current_color = color_states
                .get(currents[current_index].key.dynamic_lc_color_state_id())
                .ok_or_else(|| invalid(format!("{label} reflection color disappeared")))?;
            if dynamic_color_identity_digest(current_color)? != proof.result_color_identity() {
                return Err(invalid(format!(
                    "{label} reflection proof carries a stale dynamic color identity"
                )));
            }
        }
        let certificate_id = u32::try_from(reflection_certificates.len())
            .map_err(|_| invalid("pending reflection certificate count exceeds u32"))?;
        reflection_certificates.push(PendingReflectionCertificate::reciprocal_pair(
            certificate_id,
            u32::try_from(canonical_index)
                .map_err(|_| invalid("canonical reflection current ID exceeds u32"))?,
            u32::try_from(reflected_index)
                .map_err(|_| invalid("reflected reflection current ID exceeds u32"))?,
            &canonical_proof,
            &reflected_proof,
            current_color_for_index(color_states, currents, canonical_index)?,
            current_color_for_index(color_states, currents, reflected_index)?,
            source_count,
        )?);
        currents[canonical_index].reflection_certificate_id = Some(certificate_id);
        prune[pruned_local_index] = true;
    }

    for current in &currents[stage_start..] {
        current_ids.remove(&current.key);
    }
    let stage_currents = currents.split_off(stage_start);
    target_bucket.clear();
    for (local_index, current) in stage_currents.into_iter().enumerate() {
        if prune[local_index] {
            continue;
        }
        if current
            .contributions
            .keys()
            .flat_map(|contribution| contribution.parent_current_ids.iter().copied())
            .any(|parent_id| parent_id as usize >= stage_start)
        {
            return Err(invalid(
                "recurrence stage current depends on another current in the same stage",
            ));
        }
        let current_id = u32::try_from(currents.len())
            .map_err(|_| invalid("recurrence current count exceeds u32"))?;
        if current_ids
            .insert(current.key.clone(), current_id)
            .is_some()
        {
            return Err(invalid(
                "recurrence reflection reconciliation produced a duplicate current",
            ));
        }
        target_bucket.push(current_id);
        currents.push(current);
    }
    Ok(())
}

fn current_reversal_masks(
    colors: &[DynamicLCColorState; 2],
    reflections: &[CurrentReflection; 2],
) -> Vec<u8> {
    let mut masks = vec![0_u8];
    for index in 0..2 {
        if reflections[index].phase().is_none()
            || colors[index]
                .pure_adjoint_word()
                .is_none_or(|word| word.len() < 2)
        {
            continue;
        }
        let bit = 1_u8 << index;
        masks.extend(masks.clone().into_iter().map(|mask| mask | bit));
    }
    masks
}

fn current_reflection_candidate(
    result_color: &DynamicLCColorState,
    parent_reflections: &[CurrentReflection; 2],
    local_proof: Option<&TransitionReflectionProof>,
) -> RusticolResult<Option<CurrentReflectionProof>> {
    if result_color.pure_adjoint_word().is_none() {
        return Ok(None);
    }
    let Some(local_proof) = local_proof else {
        return Ok(None);
    };
    let Some(left) = parent_reflections[0].proof() else {
        return Ok(None);
    };
    let Some(right) = parent_reflections[1].proof() else {
        return Ok(None);
    };
    let phase = left
        .phase()
        .checked_mul(right.phase())?
        .checked_mul(local_proof.phase())?;
    let color_identity = dynamic_color_identity_digest(result_color)?;
    CurrentReflectionProof::new(
        phase,
        left.lineage_roots()
            .iter()
            .copied()
            .chain(right.lineage_roots().iter().copied())
            .chain(local_proof.lineage_roots()),
        color_identity,
    )
    .map(Some)
}

fn pure_adjoint_word_is_canonical(word: &[u32]) -> bool {
    if word.len() < 2 {
        return true;
    }
    let (minimum_index, _) = word
        .iter()
        .enumerate()
        .min_by_key(|(_, slot)| *slot)
        .expect("nonempty pure-adjoint word");
    let (maximum_index, _) = word
        .iter()
        .enumerate()
        .max_by_key(|(_, slot)| *slot)
        .expect("nonempty pure-adjoint word");
    minimum_index < maximum_index
}

fn two_parent_permutation(
    construction_parent_ids: [u32; 2],
    ordered_parent_ids: [u32; 2],
    label: &str,
) -> RusticolResult<[u32; 2]> {
    if construction_parent_ids[0] == construction_parent_ids[1] {
        return Err(invalid(format!(
            "{label} cannot derive a permutation from duplicate parent IDs"
        )));
    }
    let mut result = [0_u32; 2];
    for (target, parent_id) in ordered_parent_ids.into_iter().enumerate() {
        let source = construction_parent_ids
            .iter()
            .position(|candidate| *candidate == parent_id)
            .ok_or_else(|| invalid(format!("{label} references a foreign parent ID")))?;
        result[target] =
            u32::try_from(source).map_err(|_| invalid(format!("{label} index exceeds u32")))?;
    }
    if result[0] == result[1] {
        return Err(invalid(format!("{label} is not a permutation")));
    }
    Ok(result)
}

fn three_line_traversal_certificate(
    closed: &[LCColorComponent],
    sector: ProcessPhysicalLCSectorRow,
    process: &OwnedRecurrenceProcessInput,
    catalog: &ProcessCatalog<'_>,
    pairing_rule_id: Option<u32>,
) -> RusticolResult<Option<PendingThreeLineTraversalCertificate>> {
    if sector.kind()? != ProcessLCSectorKind::OpenLines || sector.open_string_range.count != 3 {
        return Ok(None);
    }
    let pairing_rule_id = pairing_rule_id
        .ok_or_else(|| invalid("three-line traversal lacks a physical fermion pairing rule"))?;
    let expected = expected_sector_components(sector, process, catalog)?;
    if expected.len() != 3 || closed.len() != 3 {
        return Err(invalid(
            "three-line closure does not contain exactly three complete line blocks",
        ));
    }
    let blocks = expected
        .iter()
        .map(|component| component.source_slots().to_vec())
        .collect::<Vec<_>>();
    let sector_word = catalog.u32_sequence(sector.word_sequence_id, "three-line sector word")?;
    let mut reference_order = Vec::<u32>::with_capacity(3);
    let mut used_reference = [false; 3];
    let mut offset = 0usize;
    while offset < sector_word.len() {
        let matches = blocks
            .iter()
            .enumerate()
            .filter_map(|(index, block)| {
                (!used_reference[index] && sector_word[offset..].starts_with(block))
                    .then_some(index)
            })
            .collect::<Vec<_>>();
        let [block_index] = matches.as_slice() else {
            return Err(invalid(
                "three-line sector word is not an unambiguous concatenation of its line blocks",
            ));
        };
        used_reference[*block_index] = true;
        reference_order.push(
            u32::try_from(*block_index)
                .map_err(|_| invalid("three-line block index exceeds u32"))?,
        );
        offset = offset
            .checked_add(blocks[*block_index].len())
            .ok_or_else(|| invalid("three-line sector word offset overflows"))?;
    }
    if reference_order.len() != 3 || offset != sector_word.len() {
        return Err(invalid(
            "three-line sector word does not cover exactly three line blocks",
        ));
    }
    let mut used = [false; 3];
    let mut witness_order = Vec::<u32>::with_capacity(3);
    for component in closed {
        let index = expected
            .iter()
            .enumerate()
            .find_map(|(index, candidate)| {
                (!used[index] && candidate == component).then_some(index)
            })
            .ok_or_else(|| invalid("three-line closure contains a foreign line block"))?;
        used[index] = true;
        witness_order
            .push(u32::try_from(index).map_err(|_| invalid("three-line index exceeds u32"))?);
    }
    let sink_candidates = expected
        .iter()
        .enumerate()
        .filter_map(|(index, component)| {
            (component.source_slots().last() == Some(&sector.closure_source_slot)).then_some(index)
        })
        .collect::<Vec<_>>();
    let [sink_index] = sink_candidates.as_slice() else {
        return Err(invalid(
            "three-line closure anchor does not identify exactly one line block",
        ));
    };
    let sink_index =
        u32::try_from(*sink_index).map_err(|_| invalid("three-line sink index exceeds u32"))?;
    let rotate_sink_last = |order: &mut Vec<u32>| -> RusticolResult<()> {
        let sink_position = order
            .iter()
            .position(|index| *index == sink_index)
            .ok_or_else(|| invalid("three-line closure traversal omits its anchor block"))?;
        let order_len = order.len();
        order.rotate_left((sink_position + 1) % order_len);
        Ok(())
    };
    rotate_sink_last(&mut reference_order)?;
    rotate_sink_last(&mut witness_order)?;
    let kind = if witness_order == reference_order {
        THREE_LINE_DIRECT_CERTIFICATE_ID as u8
    } else {
        let mut partner = reference_order.clone();
        partner.swap(0, 1);
        if witness_order != partner {
            return Err(invalid(
                "three-line closure traversal is neither the direct nor partner cyclic obligation",
            ));
        }
        THREE_LINE_PARTNER_CERTIFICATE_ID as u8
    };
    let block_permutation = witness_order
        .iter()
        .map(|block| {
            reference_order
                .iter()
                .position(|candidate| candidate == block)
                .map(|position| {
                    u32::try_from(position)
                        .map_err(|_| invalid("three-line block permutation exceeds u32"))
                })
                .transpose()?
                .ok_or_else(|| invalid("three-line witness references a foreign block"))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let flatten = |order: &[u32]| -> RusticolResult<Vec<u32>> {
        let mut sources = Vec::new();
        for block_index in order {
            sources.extend_from_slice(
                blocks
                    .get(*block_index as usize)
                    .ok_or_else(|| invalid("three-line block order is out of range"))?,
            );
        }
        Ok(sources)
    };
    let reference_source_order = flatten(&reference_order)?;
    let witness_source_order = flatten(&witness_order)?;
    let source_position_permutation = witness_source_order
        .iter()
        .map(|source| {
            reference_source_order
                .iter()
                .position(|candidate| candidate == source)
                .map(|position| {
                    u32::try_from(position)
                        .map_err(|_| invalid("three-line source permutation exceeds u32"))
                })
                .transpose()?
                .ok_or_else(|| invalid("three-line witness references a foreign source"))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-recurrence-three-line-traversal-v1\0");
    hash.update(sector.sector_id.to_le_bytes());
    hash.update([kind]);
    hash.update(sink_index.to_le_bytes());
    for values in [
        reference_order.as_slice(),
        witness_order.as_slice(),
        block_permutation.as_slice(),
        reference_source_order.as_slice(),
        witness_source_order.as_slice(),
        source_position_permutation.as_slice(),
    ] {
        hash.update(
            u64::try_from(values.len())
                .map_err(|_| invalid("three-line certificate sequence length exceeds u64"))?
                .to_le_bytes(),
        );
        for value in values {
            hash.update(value.to_le_bytes());
        }
    }
    hash.update(sector.closure_source_slot.to_le_bytes());
    hash.update(pairing_rule_id.to_le_bytes());
    let proof_digest = SemanticDigest::new(hash.finalize().into())?;
    Ok(Some(PendingThreeLineTraversalCertificate {
        sector_id: sector.sector_id,
        kind,
        sink_block_ordinal: sink_index,
        reference_block_order: reference_order.into_boxed_slice(),
        witness_block_order: witness_order.into_boxed_slice(),
        block_permutation: block_permutation.into_boxed_slice(),
        reference_source_order: reference_source_order.into_boxed_slice(),
        witness_source_order: witness_source_order.into_boxed_slice(),
        source_position_permutation: source_position_permutation.into_boxed_slice(),
        closure_anchor_source_slot: sector.closure_source_slot,
        pairing_rule_id,
        proof_digest,
    }))
}

fn closure_pairing_certificate_ids(
    currents: &[PendingCurrent],
    parent_ids: [u32; 2],
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
) -> RusticolResult<Vec<u32>> {
    let left = currents
        .get(parent_ids[0] as usize)
        .ok_or_else(|| invalid("closure pairing proof left parent is absent"))?;
    let right = currents
        .get(parent_ids[1] as usize)
        .ok_or_else(|| invalid("closure pairing proof right parent is absent"))?;
    let realized = left
        .realized_pairing_rule_ids
        .intersection(&right.realized_pairing_rule_ids)
        .copied()
        .collect::<BTreeSet<_>>();
    match pairing_catalog {
        None if realized.is_empty() => Ok(Vec::new()),
        None => Err(invalid(
            "closure realizes fermion pairings without a pairing catalog",
        )),
        Some(_) if realized.len() == 1 => Ok(realized.into_iter().collect()),
        Some(_) => Err(invalid(format!(
            "closure parent lineage realizes {} fermion pairing rules, expected exactly one",
            realized.len()
        ))),
    }
}

fn pairing_rule_for_certificate(
    certificate_ids: &[u32],
    catalog: Option<ValidatedFermionPairingCatalog<'_>>,
) -> RusticolResult<Option<FermionPairingRuleRow>> {
    let Some(catalog) = catalog else {
        if !certificate_ids.is_empty() {
            return Err(invalid(
                "closure pairing certificate has no pairing catalog",
            ));
        }
        return Ok(None);
    };
    let [rule_id] = certificate_ids else {
        return Err(invalid(
            "closure requires exactly one fermion pairing certificate",
        ));
    };
    catalog
        .rules()
        .iter()
        .copied()
        .find(|rule| rule.rule_id == *rule_id)
        .map(Some)
        .ok_or_else(|| invalid(format!("closure references absent pairing rule {rule_id}")))
}

fn pairing_reconstruction_factor(_rule: Option<FermionPairingRuleRow>) -> ExactComplexRational {
    // The canonical closure/input ordering already carries the fermionic
    // exchange sign. The pairing rule authenticates which Wick lineage was
    // realized; multiplying its parity here would apply that sign twice.
    ExactComplexRational::ONE
}

// Closure certification deliberately receives each authenticated catalog and
// proof table separately so that no unchecked aggregate can cross this boundary.
#[allow(clippy::too_many_arguments)]
fn closure_reflection_certificate_id(
    sector: ProcessPhysicalLCSectorRow,
    closed: &[LCColorComponent],
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    color_states: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    parent_ids: [u32; 2],
    certificates: &[PendingReflectionCertificate],
) -> RusticolResult<Option<u32>> {
    let mut certified_parents = Vec::new();
    for parent_id in parent_ids {
        let current = currents
            .get(parent_id as usize)
            .ok_or_else(|| invalid("closure reflection parent is absent"))?;
        let Some(certificate_id) = current.reflection_certificate_id else {
            continue;
        };
        let certificate = certificates
            .get(certificate_id as usize)
            .filter(|certificate| certificate.id == certificate_id)
            .ok_or_else(|| invalid("closure reflection certificate is absent"))?;
        let proof = current
            .reflection
            .proof()
            .ok_or_else(|| invalid("certified closure reflection parent has no lineage proof"))?;
        if proof.proof_digest() != certificate.canonical_lineage_digest {
            return Err(invalid(
                "closure reflection parent lineage does not match its certificate",
            ));
        }
        let color = color_states
            .get(current.key.dynamic_lc_color_state_id())
            .ok_or_else(|| invalid("closure reflection parent color is absent"))?;
        if dynamic_color_identity_digest(color)? != certificate.canonical_color_identity {
            return Err(invalid(
                "closure reflection parent color does not match its certificate",
            ));
        }
        certified_parents.push(certificate_id);
    }
    match certified_parents.len() {
        0 => return Ok(None),
        1 => {}
        _ => {
            return Err(invalid(
                "one closure references multiple folded reflection orbits",
            ));
        }
    }
    if sector.kind()? != ProcessLCSectorKind::SingleTrace {
        return Ok(None);
    }
    let construction_word =
        process_catalog.u32_sequence(sector.word_sequence_id, "reflection construction word")?;
    if construction_word.len() <= 2 {
        return Ok(None);
    }
    let [closed_trace] = closed else {
        return Ok(None);
    };
    if closed_trace.kind() != LCColorComponentKind::Trace
        || closed_trace.source_slots().len() != construction_word.len()
    {
        return Ok(None);
    }
    let certificate_id = certified_parents[0];
    let certificate = certificates
        .get(certificate_id as usize)
        .filter(|certificate| certificate.id == certificate_id)
        .ok_or_else(|| invalid("closure reflection certificate is absent"))?;
    if certificate.fixed_point
        || certificate.orbit_size != 2
        || certificate.canonical_color_identity == certificate.reflected_color_identity
    {
        return Err(invalid(
            "closure reflection certificate does not describe a reciprocal two-cycle",
        ));
    }
    let mapped_word = construction_word
        .iter()
        .map(|source_slot| {
            certificate
                .source_permutation
                .get(*source_slot as usize)
                .copied()
                .ok_or_else(|| {
                    invalid("closure reflection permutation does not cover its trace word")
                })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    if cyclic_words_equal(construction_word, &mapped_word) {
        return Err(invalid(
            "closure reflection maps a trace to a cyclic fixed point",
        ));
    }
    let mut matching_public_flows = Vec::new();
    for flow in retained_public_flows(process)?
        .into_iter()
        .filter(|flow| flow.construction_sector_id == sector.sector_id)
    {
        let word =
            process_catalog.u32_sequence(flow.word_sequence_id, "reflected public LC flow")?;
        if cyclic_words_equal(word, &mapped_word) {
            matching_public_flows.push(flow.flow_id);
        }
    }
    match matching_public_flows.as_slice() {
        [] => Ok(None),
        [_] => Ok(Some(certificate_id)),
        _ => Err(invalid(
            "closure reflection maps to multiple retained public LC flows",
        )),
    }
}

fn cyclic_words_equal(left: &[u32], right: &[u32]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    if left.is_empty() {
        return true;
    }
    (0..left.len()).any(|offset| {
        left.iter()
            .enumerate()
            .all(|(index, value)| *value == right[(index + offset) % right.len()])
    })
}

fn materialize_reflection_certificates(
    certificates: &[PendingReflectionCertificate],
) -> RusticolResult<Vec<ReflectionCertificateV1>> {
    certificates
        .iter()
        .map(|certificate| {
            ReflectionCertificateV1::new(
                certificate.id,
                certificate.canonical_color_identity,
                certificate.reflected_color_identity,
                certificate.source_permutation.to_vec(),
                certificate.canonical_phase,
                certificate.fixed_point,
                certificate.orbit_size,
                REFLECTION_PROOF_ALGORITHM_ID,
                certificate.proof_digest,
            )
        })
        .collect()
}

fn materialize_three_line_certificates(
    certificate_ids: &BTreeMap<PendingThreeLineTraversalCertificate, u32>,
) -> RusticolResult<Vec<ThreeLineTraversalCertificateV1>> {
    let mut rows = certificate_ids
        .iter()
        .map(|(certificate, id)| {
            ThreeLineTraversalCertificateV1::new(
                *id,
                certificate.sector_id,
                ThreeLineTraversalKindV1::try_from(u32::from(certificate.kind))?,
                certificate.sink_block_ordinal,
                certificate.reference_block_order.to_vec(),
                certificate.witness_block_order.to_vec(),
                certificate.block_permutation.to_vec(),
                certificate.reference_source_order.to_vec(),
                certificate.witness_source_order.to_vec(),
                certificate.source_position_permutation.to_vec(),
                certificate.closure_anchor_source_slot,
                certificate.pairing_rule_id,
                certificate.proof_digest,
            )
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    rows.sort_unstable_by_key(ThreeLineTraversalCertificateV1::id);
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
fn build_closures(
    strategy: RecurrenceStrategy,
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    template: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    color_states: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    materialized_sectors: &BTreeSet<u32>,
    stage_diagnostics: &[StageConstructionDiagnostics],
    reflection_certificates: &[PendingReflectionCertificate],
) -> RusticolResult<BTreeMap<PendingClosureKey, PendingClosureGroup>> {
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
                        return Err(invalid(
                            "direct recurrence requires binary prepared closures",
                        ));
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
                            strategy,
                            sector,
                            *closure,
                            parent_ids,
                            process,
                            process_catalog,
                            template,
                            catalog,
                            color_states,
                            currents,
                            pairing_catalog,
                            reflection_certificates,
                            &mut result,
                            &mut closure_color_attempts,
                        )?;
                    }
                }
            }
        }
        if !result.keys().any(|key| key.target_sector_id == sector_id) {
            if strategy == RecurrenceStrategy::ContractedColorUnion {
                continue;
            }
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
    validate_pending_closure_obligations(&result, pairing_catalog, process, process_catalog)?;
    Ok(result)
}

fn validate_pending_closure_obligations(
    closures: &BTreeMap<PendingClosureKey, PendingClosureGroup>,
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
) -> RusticolResult<()> {
    let Some(pairing_catalog) = pairing_catalog else {
        return Ok(());
    };
    let destinations = closures
        .keys()
        .map(|key| (key.target_sector_id, key.complete_source_states.clone()))
        .collect::<BTreeSet<_>>();
    for (sector_id, source_states) in destinations {
        let sector = *process
            .physical_lc_sectors
            .get(sector_id as usize)
            .ok_or_else(|| invalid("closure obligation sector is absent"))?;
        let rows = closures
            .iter()
            .filter(|(key, _)| {
                key.target_sector_id == sector_id && key.complete_source_states == source_states
            })
            .flat_map(|(_, group)| group.contributions.iter())
            .collect::<Vec<_>>();
        let realized_pairings = rows
            .iter()
            .flat_map(|row| row.pairing_certificate_ids.iter().copied())
            .collect::<BTreeSet<_>>();
        if realized_pairings.is_empty() {
            return Err(invalid(format!(
                "closure destination for sector {sector_id} realizes no fermion pairing rule"
            )));
        }
        for pairing_id in &realized_pairings {
            if pairing_catalog
                .rules()
                .iter()
                .all(|rule| rule.rule_id != *pairing_id)
            {
                return Err(invalid(format!(
                    "closure destination for sector {sector_id} references absent pairing rule {pairing_id}"
                )));
            }
        }
        if sector.kind()? == ProcessLCSectorKind::OpenLines && sector.open_string_range.count == 3 {
            for pairing_id in realized_pairings.iter().copied() {
                let matching_rows = rows
                    .iter()
                    .filter(|row| row.pairing_certificate_ids.contains(&pairing_id))
                    .collect::<Vec<_>>();
                if matching_rows
                    .iter()
                    .any(|row| row.three_line_certificate.is_none())
                {
                    return Err(invalid(format!(
                        "three-line closure sector {sector_id} pairing {pairing_id} has an uncertified traversal"
                    )));
                }
                let traversal_kinds = matching_rows
                    .into_iter()
                    .map(|row| {
                        let certificate = row
                            .three_line_certificate
                            .as_ref()
                            .expect("presence checked above");
                        if certificate.sector_id != sector_id
                            || certificate.pairing_rule_id != pairing_id
                        {
                            return Err(invalid(format!(
                                "three-line closure sector {sector_id} pairing {pairing_id} \
                                 has a traversal certificate for sector {} pairing {}",
                                certificate.sector_id, certificate.pairing_rule_id,
                            )));
                        }
                        match u32::from(certificate.kind) {
                            THREE_LINE_DIRECT_CERTIFICATE_ID
                            | THREE_LINE_PARTNER_CERTIFICATE_ID => Ok(u32::from(certificate.kind)),
                            kind => Err(invalid(format!(
                                "three-line closure sector {sector_id} pairing {pairing_id} \
                                 has invalid traversal kind {kind}"
                            ))),
                        }
                    })
                    .collect::<RusticolResult<BTreeSet<_>>>()?;
                if traversal_kinds.is_empty() {
                    return Err(invalid(format!(
                        "three-line closure sector {sector_id} pairing {pairing_id} \
                         realizes no certified traversal"
                    )));
                }
            }
        }
        let expected_components = expected_sector_components(sector, process, process_catalog)?;
        if expected_components.len() != sector.open_string_range.count as usize
            && sector.kind()? == ProcessLCSectorKind::OpenLines
        {
            return Err(invalid(
                "closure obligation open-line component count is inconsistent",
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn add_closure_terms(
    strategy: RecurrenceStrategy,
    sector: ProcessPhysicalLCSectorRow,
    closure: ClosureRow,
    parent_ids: [u32; 2],
    process: &OwnedRecurrenceProcessInput,
    process_catalog: &ProcessCatalog<'_>,
    template: &OwnedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    color_states: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    pairing_catalog: Option<ValidatedFermionPairingCatalog<'_>>,
    reflection_certificates: &[PendingReflectionCertificate],
    result: &mut BTreeMap<PendingClosureKey, PendingClosureGroup>,
    closure_color_attempts: &mut BTreeSet<Vec<(LCColorComponentKind, Vec<u32>)>>,
) -> RusticolResult<()> {
    let parents = [
        &currents[parent_ids[0] as usize].key,
        &currents[parent_ids[1] as usize].key,
    ];
    let pairing_certificate_ids =
        closure_pairing_certificate_ids(currents, parent_ids, pairing_catalog)?;
    let pairing_rule = pairing_rule_for_certificate(&pairing_certificate_ids, pairing_catalog)?;
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
        let evaluator_parent_permutation =
            two_parent_permutation(parent_ids, evaluator_parent_ids, "closure evaluator order")?;
        let base_factor = multiply_factors(&[
            catalog.factor(closure.exact_factor_id, "closure exact")?,
            exchange_factor,
            catalog.factor(contraction.exact_coefficient_factor_id, "closure color")?,
            output_factor_from_binding(binding_coupling, closure.output_factor_source, "closure")?,
            pairing_reconstruction_factor(pairing_rule),
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
            if !closed_components_match_sector(
                strategy,
                &closed,
                sector,
                process,
                process_catalog,
                template,
            )? {
                continue;
            }
            if pairing_catalog.is_some() && pairing_certificate_ids.is_empty() {
                return Err(invalid(format!(
                    "closure witness {} for sector {} has no exactly realized fermion pairing",
                    witness_row.ordinal, sector.sector_id
                )));
            }
            let reconstruction_parent_permutation = match witness_row.input_permutation {
                0 => [0, 1],
                1 => [1, 0],
                value => {
                    return Err(invalid(format!(
                        "closure color witness has invalid parent permutation {value}"
                    )));
                }
            };
            let color_witness_term_id = contraction
                .witness_start
                .checked_add(u64::from(witness_row.ordinal))
                .ok_or_else(|| invalid("closure color-witness term ID overflows"))?;
            let color_witness_term_id = u32::try_from(color_witness_term_id)
                .map_err(|_| invalid("closure color-witness term ID exceeds u32"))?;
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
            result
                .entry(key)
                .or_default()
                .include(PendingClosureProofContribution {
                    construction_parent_ids: parent_ids,
                    construction_parent_permutation: [0, 1],
                    reconstruction_parent_permutation,
                    evaluator_parent_permutation,
                    closure_template_semantic_digest: catalog
                        .digest(closure.semantic_digest_id, "closure semantic")?,
                    color_witness_term_id,
                    color_witness_proof_digest: witness.proof_digest(),
                    three_line_certificate: three_line_traversal_certificate(
                        &closed,
                        sector,
                        process,
                        process_catalog,
                        pairing_rule.map(|rule| rule.rule_id),
                    )?,
                    pairing_certificate_ids: pairing_certificate_ids.clone().into_boxed_slice(),
                    reflection_certificate_id: closure_reflection_certificate_id(
                        sector,
                        &closed,
                        process,
                        process_catalog,
                        color_states,
                        currents,
                        parent_ids,
                        reflection_certificates,
                    )?,
                    exact_factor: factor,
                })?;
        }
    }
    Ok(())
}

fn closure_selector_proof_words(
    sector_id: u32,
    target_helicity_id: Option<u32>,
    source_states: &[SourceStateAssignment],
) -> RusticolResult<Vec<u64>> {
    let mut words = Vec::with_capacity(
        3usize
            .checked_add(source_states.len().saturating_mul(2))
            .ok_or_else(|| invalid("closure selector-proof word count overflows usize"))?,
    );
    words.push(u64::from(sector_id));
    words.push(u64::from(target_helicity_id.unwrap_or(MISSING_U32)));
    words.push(
        u64::try_from(source_states.len())
            .map_err(|_| invalid("closure selector source-state count exceeds u64"))?,
    );
    for assignment in source_states {
        words.push(u64::from(assignment.source_slot()));
        words.push(u64::from(assignment.state_index()));
    }
    Ok(words)
}

fn pending_closure_candidate_identity_digest(
    key: &PendingClosureKey,
    contribution: &PendingClosureProofContribution,
    target_helicity_id: Option<u32>,
    pending: &[PendingCurrent],
    dynamic_color_states: &[DynamicLCColorState],
    reflection_certificates: &[PendingReflectionCertificate],
) -> RusticolResult<SemanticDigest> {
    let selector_words = closure_selector_proof_words(
        key.target_sector_id,
        target_helicity_id,
        &key.complete_source_states,
    )?;
    let selector_domain_digest = closure_selector_domain_digest_v2(&selector_words)?;
    let mut parent_semantic_digests =
        Vec::with_capacity(contribution.construction_parent_ids.len());
    let mut parent_color_digests = Vec::with_capacity(contribution.construction_parent_ids.len());
    for old_id in contribution.construction_parent_ids {
        let current = pending
            .get(old_id as usize)
            .ok_or_else(|| invalid("closure candidate source parent is absent"))?;
        parent_semantic_digests.push(semantic_digest_from_u32_fields(
            b"recurrence-closure-parent-current-v2\0",
            [
                old_id,
                current.key.current_state_template_id(),
                current.key.dynamic_lc_color_state_id().get(),
            ],
        )?);
        parent_color_digests.push(dynamic_color_identity_digest(
            dynamic_color_states
                .get(current.key.dynamic_lc_color_state_id().get() as usize)
                .ok_or_else(|| invalid("closure candidate parent color state is absent"))?,
        )?);
    }
    let reflection_proof_digest = contribution
        .reflection_certificate_id
        .map(|certificate_id| {
            reflection_certificates
                .get(certificate_id as usize)
                .map(|certificate| certificate.proof_digest)
                .ok_or_else(|| {
                    invalid(format!(
                        "closure candidate references absent reflection certificate {certificate_id}"
                    ))
                })
        })
        .transpose()?;
    closure_candidate_identity_digest_v1(
        selector_domain_digest,
        key.target_sector_id,
        key.closure_template_id,
        contribution.closure_template_semantic_digest,
        key.quantum_flow_template_id,
        &contribution.construction_parent_ids,
        &parent_semantic_digests,
        &parent_color_digests,
        &contribution.construction_parent_permutation,
        &contribution.reconstruction_parent_permutation,
        &contribution.evaluator_parent_permutation,
        contribution.color_witness_term_id,
        contribution.color_witness_proof_digest,
        contribution
            .three_line_certificate
            .as_ref()
            .map(|certificate| certificate.proof_digest),
        &contribution.pairing_certificate_ids,
        reflection_proof_digest,
        contribution.exact_factor,
        1,
    )
}

#[allow(clippy::too_many_arguments)]
fn append_closure_proof_group(
    key: &PendingClosureKey,
    group: &PendingClosureGroup,
    pending: &[PendingCurrent],
    dynamic_color_states: &[DynamicLCColorState],
    remap: &BTreeMap<u32, u32>,
    target_destination_id: Option<u32>,
    target_helicity_id: Option<u32>,
    runtime_term_id: Option<u32>,
    three_line_certificate_ids: &BTreeMap<PendingThreeLineTraversalCertificate, u32>,
    closure_proof_contributions: &mut Vec<ClosureProofContributionV2>,
    closure_proof_groups: &mut Vec<ClosureExecutionProofGroupV2>,
) -> RusticolResult<()> {
    if group.exact_factor.is_zero() != runtime_term_id.is_none() {
        return Err(invalid(
            "closure proof runtime binding does not match its exact aggregate factor",
        ));
    }
    if runtime_term_id.is_some() && target_destination_id.is_none() {
        return Err(invalid(
            "nonzero closure proof group has no amplitude destination",
        ));
    }
    let proof_start = u64::try_from(closure_proof_contributions.len())
        .map_err(|_| invalid("closure proof contribution count exceeds u64"))?;
    let mut component_factors = Vec::with_capacity(group.contributions.len());
    for contribution in &group.contributions {
        let builder_parent_ids = contribution.construction_parent_ids.to_vec();
        let runtime_parent_ids = contribution
            .construction_parent_ids
            .iter()
            .map(|old_id| remap.get(old_id).copied())
            .collect::<Vec<_>>();
        if runtime_term_id.is_some() && runtime_parent_ids.iter().any(Option::is_none) {
            return Err(invalid(
                "nonzero closure proof contribution has a dead runtime parent",
            ));
        }
        let mut parent_semantic_digests =
            Vec::with_capacity(contribution.construction_parent_ids.len());
        let mut parent_color_digests =
            Vec::with_capacity(contribution.construction_parent_ids.len());
        for old_id in contribution.construction_parent_ids {
            let current = pending
                .get(old_id as usize)
                .ok_or_else(|| invalid("closure proof source parent is absent"))?;
            parent_semantic_digests.push(semantic_digest_from_u32_fields(
                b"recurrence-closure-parent-current-v2\0",
                [
                    old_id,
                    current.key.current_state_template_id(),
                    current.key.dynamic_lc_color_state_id().get(),
                ],
            )?);
            parent_color_digests.push(dynamic_color_identity_digest(
                dynamic_color_states
                    .get(current.key.dynamic_lc_color_state_id().get() as usize)
                    .ok_or_else(|| invalid("closure proof parent color state is absent"))?,
            )?);
        }
        let proof_id = u32::try_from(closure_proof_contributions.len())
            .map_err(|_| invalid("closure proof contribution count exceeds u32"))?;
        closure_proof_contributions.push(ClosureProofContributionV2::new(
            proof_id,
            key.target_sector_id,
            target_destination_id,
            target_helicity_id,
            key.closure_template_id,
            contribution.closure_template_semantic_digest,
            key.quantum_flow_template_id,
            builder_parent_ids,
            runtime_parent_ids,
            parent_semantic_digests,
            parent_color_digests,
            contribution.construction_parent_permutation.to_vec(),
            contribution.reconstruction_parent_permutation.to_vec(),
            contribution.evaluator_parent_permutation.to_vec(),
            contribution.color_witness_term_id,
            contribution.color_witness_proof_digest,
            contribution
                .three_line_certificate
                .as_ref()
                .map(|certificate| three_line_certificate_ids[certificate]),
            contribution.pairing_certificate_ids.to_vec(),
            contribution.reflection_certificate_id,
            contribution.exact_factor,
            1,
        )?);
        component_factors.push(contribution.exact_factor);
    }
    let proof_count = u64::try_from(closure_proof_contributions.len())
        .map_err(|_| invalid("closure proof contribution count exceeds u64"))?
        .checked_sub(proof_start)
        .ok_or_else(|| invalid("closure proof contribution range underflows"))?;
    let selector_words = closure_selector_proof_words(
        key.target_sector_id,
        target_helicity_id,
        &key.complete_source_states,
    )?;
    closure_proof_groups.push(ClosureExecutionProofGroupV2::new(
        u32::try_from(closure_proof_groups.len())
            .map_err(|_| invalid("closure proof group count exceeds u32"))?,
        runtime_term_id,
        None,
        CheckedTableRange::new(proof_start, proof_count),
        group.exact_factor,
        closure_component_factor_digest_v2(&component_factors)?,
        closure_selector_domain_digest_v2(&selector_words)?,
    )?);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn finish_program(
    strategy: RecurrenceStrategy,
    process_catalog: &ProcessCatalog<'_>,
    dynamic_color_states: Vec<DynamicLCColorState>,
    pending: Vec<PendingCurrent>,
    mut pending_closures: BTreeMap<PendingClosureKey, PendingClosureGroup>,
    replay_targets: Vec<RecurrenceReplayTarget>,
    retained_helicity_count: u64,
    helicity_support_rule: HelicitySupportRule,
    reflection_certificates: Vec<PendingReflectionCertificate>,
) -> RusticolResult<RecurrenceProgram> {
    validate_pending_reflection_certificates(&reflection_certificates)?;

    if helicity_support_rule != HelicitySupportRule::None {
        let mut retained = BTreeMap::new();
        for (key, group) in pending_closures {
            if closure_helicity_is_supported(
                helicity_support_rule,
                process_catalog,
                &key.complete_source_states,
            )? {
                retained.insert(key, group);
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
        RecurrenceStrategy::AllFlowUnion | RecurrenceStrategy::ContractedColorUnion => {
            process_catalog.input.physical_lc_sectors.len()
        }
    };
    let helicity_keys = pending_closures
        .iter()
        .filter(|(key, group)| {
            !key.complete_source_states.is_empty() && !group.exact_factor.is_zero()
        })
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
    for (key, group) in &pending_closures {
        if group.exact_factor.is_zero() {
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
        .filter(|(_, group)| !group.exact_factor.is_zero())
        .map(|(key, _)| (key.target_sector_id, key.complete_source_states.clone()))
        .collect::<BTreeSet<_>>();
    let mut closure_terms = Vec::new();
    let mut closure_proof_contributions = Vec::new();
    let mut closure_proof_groups = Vec::new();
    let pending_three_line_certificates = pending_closures
        .values()
        .flat_map(|group| group.contributions.iter())
        .filter_map(|contribution| contribution.three_line_certificate.clone())
        .collect::<BTreeSet<_>>();
    let three_line_certificate_ids = pending_three_line_certificates
        .iter()
        .cloned()
        .enumerate()
        .map(|(id, certificate)| {
            u32::try_from(id)
                .map(|id| (certificate, id))
                .map_err(|_| invalid("three-line certificate count exceeds u32"))
        })
        .collect::<RusticolResult<BTreeMap<_, _>>>()?;
    let mut accepted_candidate_identity_digests = Vec::new();
    for (key, group) in &pending_closures {
        let target_helicity_id = if key.complete_source_states.is_empty() {
            None
        } else {
            helicity_ids.get(&key.complete_source_states).copied()
        };
        for contribution in &group.contributions {
            accepted_candidate_identity_digests.push(pending_closure_candidate_identity_digest(
                key,
                contribution,
                target_helicity_id,
                &pending,
                &dynamic_color_states,
                &reflection_certificates,
            )?);
        }
    }
    let candidate_domain_certificate = ClosureCandidateDomainCertificateV1::from_identity_digests(
        accepted_candidate_identity_digests,
    )?;
    let mut amplitude_destinations = Vec::with_capacity(destination_keys.len());
    let mut destination_ids = BTreeMap::new();
    for (destination_id, (sector_id, source_states)) in destination_keys.into_iter().enumerate() {
        let destination_id = u32::try_from(destination_id)
            .map_err(|_| invalid("closure destination ID exceeds u32"))?;
        destination_ids.insert((sector_id, source_states.clone()), destination_id);
        let start = closure_terms.len() as u64;
        for (key, group) in pending_closures.iter().filter(|(key, group)| {
            key.target_sector_id == sector_id
                && key.complete_source_states == source_states
                && !group.exact_factor.is_zero()
        }) {
            let target_helicity_id = if source_states.is_empty() {
                None
            } else {
                Some(
                    *helicity_ids
                        .get(&source_states)
                        .ok_or_else(|| invalid("resolved-helicity destination disappeared"))?,
                )
            };
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
            let runtime_term_id = u32::try_from(closure_terms.len())
                .map_err(|_| invalid("runtime closure-term count exceeds u32"))?;
            closure_terms.push(RecurrenceClosureTerm::new(
                runtime_term_id,
                destination_id,
                key.closure_template_id,
                key.quantum_flow_template_id,
                parents,
                group.exact_factor,
            )?);
            append_closure_proof_group(
                key,
                group,
                &pending,
                &dynamic_color_states,
                &remap,
                Some(destination_id),
                target_helicity_id,
                Some(runtime_term_id),
                &three_line_certificate_ids,
                &mut closure_proof_contributions,
                &mut closure_proof_groups,
            )?;
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
            destination_id,
            sector_id,
            target_helicity_id,
            CheckedTableRange::new(start, closure_terms.len() as u64 - start),
        )?);
    }
    for (key, group) in pending_closures
        .iter()
        .filter(|(_, group)| group.exact_factor.is_zero())
    {
        let target_destination_id = destination_ids
            .get(&(key.target_sector_id, key.complete_source_states.clone()))
            .copied();
        let target_helicity_id = if key.complete_source_states.is_empty() {
            None
        } else {
            helicity_ids.get(&key.complete_source_states).copied()
        };
        append_closure_proof_group(
            key,
            group,
            &pending,
            &dynamic_color_states,
            &remap,
            target_destination_id,
            target_helicity_id,
            None,
            &three_line_certificate_ids,
            &mut closure_proof_contributions,
            &mut closure_proof_groups,
        )?;
    }
    let closure_proofs =
        ClosureProofMetadataV2::new_with_three_line_certificates_and_candidate_domain(
            closure_proof_contributions,
            closure_proof_groups,
            materialize_reflection_certificates(&reflection_certificates)?,
            materialize_three_line_certificates(&three_line_certificate_ids)?,
            candidate_domain_certificate,
        )?;
    RecurrenceProgram::new_with_closure_proofs(
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
        closure_proofs,
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
        (
            CurrentHelicityIdentity::ContractedColorUnion {
                local_source_states: left,
                ..
            },
            CurrentHelicityIdentity::ContractedColorUnion {
                local_source_states: right,
                ..
            },
        ) => {
            let mut result = left.iter().chain(right.iter()).copied().collect::<Vec<_>>();
            result.sort_unstable();
            if result.len() != source_count {
                return Err(invalid(
                    "contracted-color closure ancestry does not cover every external source",
                ));
            }
            for (source_slot, assignment) in result.iter().copied().enumerate() {
                if assignment.source_slot() as usize != source_slot {
                    return Err(invalid(
                        "contracted-color closure ancestry is incomplete or overlapping",
                    ));
                }
            }
            Ok(result.into_boxed_slice())
        }
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
            "direct recurrence requires binary quantum-flow contracts",
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
            || !quantum_parent_spin_matches(spins[index], parents[index])
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn quantum_parent_spin_matches(required_spin: i32, parent: &CurrentCoreKey) -> bool {
    required_spin == parent.spin_state_class()
        || (parent.node_kind() == RecurrenceNodeKind::Source
            && parent.helicity_identity().strategy() == RecurrenceStrategy::AllFlowUnion
            && parent.spin_state_class() == DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS)
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

    // The prepared recurrence contract has already proved that transition
    // admission and kernel selection are independent of accumulated flavour
    // ancestry.  In the runtime-helicity union, retaining the construction
    // history would nevertheless split one physical current into many
    // numerically accumulated copies.  Canonicalize it to the result species,
    // matching the compact all-flow recurrence identity.
    if parents
        .iter()
        .all(|parent| parent.helicity_identity().strategy() == RecurrenceStrategy::AllFlowUnion)
    {
        return Ok(vec![result_particle]);
    }

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
    strategy: RecurrenceStrategy,
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
    if !unordered_color_components_match(closed, &expected) {
        return Ok(false);
    }

    if strategy == RecurrenceStrategy::ContractedColorUnion {
        // A permutation of complete open strings is the same product of color
        // tensors. Accumulate the full coherent amplitude in one deterministic
        // owner instead of duplicating or partitioning it across ordering
        // aliases.
        for candidate in process
            .physical_lc_sectors
            .iter()
            .take(sector.sector_id as usize)
        {
            if candidate.kind()? != ProcessLCSectorKind::OpenLines {
                continue;
            }
            let candidate_components = expected_sector_components(*candidate, process, catalog)?;
            if unordered_color_components_match(&expected, &candidate_components) {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn unordered_color_components_match(left: &[LCColorComponent], right: &[LCColorComponent]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    let mut matched = vec![false; right.len()];
    for component in left {
        let Some(index) = right.iter().enumerate().find_map(|(index, candidate)| {
            (!matched[index] && candidate == component).then_some(index)
        }) else {
            return false;
        };
        matched[index] = true;
    }
    true
}

fn materialized_sector_ids(
    strategy: RecurrenceStrategy,
    process: &OwnedRecurrenceProcessInput,
    replay_targets: &[RecurrenceReplayTarget],
) -> BTreeSet<u32> {
    match strategy {
        RecurrenceStrategy::AllFlowUnion | RecurrenceStrategy::ContractedColorUnion => process
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
    if !strategy.uses_topology_replay_targets() {
        return Ok(Vec::new());
    }
    let source_momentum_signs = external_source_momentum_signs(process)?;
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
            let replay_momentum_signs =
                replay_momentum_signs(&source_momentum_signs, &permutation)?;
            RecurrenceReplayTarget::new(
                u32::try_from(target_sector_id)
                    .map_err(|_| invalid("replay-target count exceeds u32"))?,
                *materialized,
                u32::try_from(target_sector_id)
                    .map_err(|_| invalid("public-flow target ID exceeds u32"))?,
                permutation,
                replay_momentum_signs,
                *factor,
            )
        })
        .collect()
}

fn external_source_momentum_signs(
    process: &OwnedRecurrenceProcessInput,
) -> RusticolResult<Vec<i32>> {
    process
        .external_legs
        .iter()
        .enumerate()
        .map(|(source_slot, leg)| {
            let range = leg.source_state_range.as_usize_range(
                process.source_states.len(),
                "external source momentum signs",
            )?;
            let mut signs = process.source_states[range]
                .iter()
                .map(|state| {
                    if state.source_slot as usize != source_slot
                        || !matches!(state.momentum_sign, -1 | 1)
                    {
                        return Err(invalid(format!(
                            "source slot {source_slot} has an invalid momentum-sign contract"
                        )));
                    }
                    Ok(state.momentum_sign)
                })
                .collect::<RusticolResult<BTreeSet<_>>>()?;
            if signs.len() != 1 {
                return Err(invalid(format!(
                    "source slot {source_slot} has helicity-dependent momentum crossing"
                )));
            }
            signs
                .pop_first()
                .ok_or_else(|| invalid(format!("source slot {source_slot} has no source states")))
        })
        .collect()
}

fn replay_momentum_signs(
    source_momentum_signs: &[i32],
    representative_to_public: &[u32],
) -> RusticolResult<Vec<i32>> {
    representative_to_public
        .iter()
        .copied()
        .enumerate()
        .map(|(representative_slot, public_slot)| {
            let representative_sign = source_momentum_signs
                .get(representative_slot)
                .copied()
                .filter(|sign| matches!(sign, -1 | 1))
                .ok_or_else(|| invalid("representative source momentum sign is absent"))?;
            let public_sign = source_momentum_signs
                .get(public_slot as usize)
                .copied()
                .filter(|sign| matches!(sign, -1 | 1))
                .ok_or_else(|| invalid("public source momentum sign is absent"))?;
            Ok(representative_sign * public_sign)
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
        (
            CurrentHelicityIdentity::ContractedColorUnion {
                local_source_states: left,
                ..
            },
            CurrentHelicityIdentity::ContractedColorUnion {
                local_source_states: right,
                ..
            },
        ) => {
            let mut values = left.iter().chain(right.iter()).copied().collect::<Vec<_>>();
            values.sort_unstable();
            CurrentHelicityIdentity::contracted_color_union(result_spin, values)
        }
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
    use super::super::{LCColorEndpoint, LCColorPortBinding};
    use super::*;

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).expect("test digest must be nonzero")
    }

    fn proven_reflection(
        color: &DynamicLCColorState,
        phase: ExactComplexRational,
        root: u8,
    ) -> CurrentReflection {
        CurrentReflection::Proven(
            CurrentReflectionProof::new(
                phase,
                [digest(root)],
                dynamic_color_identity_digest(color).unwrap(),
            )
            .unwrap(),
        )
    }

    fn proven_interned_reflection(
        colors: &DynamicLCColorStateInterner,
        color_id: DynamicLCColorStateId,
        phase: ExactComplexRational,
        root: u8,
    ) -> CurrentReflection {
        proven_reflection(colors.get(color_id).unwrap(), phase, root)
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

    #[test]
    fn all_flow_union_source_spin_class_is_dynamic_for_chiral_variants() {
        assert_eq!(
            union_spin_state_class(25, &BTreeSet::from([-1, 1])).unwrap(),
            DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS
        );
        assert_eq!(
            union_spin_state_class(25, &BTreeSet::from([-1])).unwrap(),
            -1
        );
        assert!(
            union_spin_state_class(25, &BTreeSet::from([DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS]),)
                .is_err()
        );

        let mut colors = DynamicLCColorStateInterner::default();
        let color_id = colors
            .intern(DynamicLCColorState::new(0, None, vec![]).unwrap())
            .unwrap();
        let source = CurrentCoreKey::new(
            digest(202),
            RecurrenceNodeKind::Source,
            17,
            color_id,
            vec![0],
            CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                source_slot: 0,
                coefficient: 1,
            }])
            .unwrap(),
            CurrentHelicityIdentity::all_flow_union(DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS),
            vec![0],
            0,
            vec![],
            CurrentSourceBinding::runtime_dispatch(25, vec![0, 1]).unwrap(),
            None,
        )
        .unwrap();

        assert!(quantum_parent_spin_matches(-1, &source));
        assert!(quantum_parent_spin_matches(1, &source));
        assert!(structural_parent_states_match(
            [StructuralState::new(17, -1), StructuralState::new(19, 0),],
            [
                StructuralState::new(17, DYNAMIC_UNION_SOURCE_SPIN_STATE_CLASS),
                StructuralState::new(19, 0),
            ],
        ));
    }

    #[test]
    fn pure_adjoint_reflection_masks_require_exact_parent_proofs() {
        let left = DynamicLCColorState::new_port_wired(
            1,
            vec![
                LCColorPortBinding::new(0, LCColorEndpoint::Back),
                LCColorPortBinding::new(0, LCColorEndpoint::Front),
            ],
            vec![LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![1, 4]).unwrap()],
        )
        .unwrap();
        let right = DynamicLCColorState::new_port_wired(
            1,
            vec![
                LCColorPortBinding::new(0, LCColorEndpoint::Back),
                LCColorPortBinding::new(0, LCColorEndpoint::Front),
            ],
            vec![LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![2, 3]).unwrap()],
        )
        .unwrap();
        let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
        let reflections = [
            proven_reflection(&left, minus_one, 40),
            proven_reflection(&right, minus_one, 41),
        ];

        assert_eq!(
            current_reversal_masks(&[left.clone(), right.clone()], &reflections),
            [0, 1, 2, 3],
        );
        let partial_reflections = [
            proven_reflection(&left, minus_one, 42),
            CurrentReflection::Unavailable,
        ];
        assert_eq!(
            current_reversal_masks(&[left, right], &partial_reflections),
            [0, 1],
        );
    }

    #[test]
    fn source_reflection_lineage_is_deterministic_and_color_bound() {
        let mut colors = DynamicLCColorStateInterner::default();
        let first_color = colors
            .intern(
                DynamicLCColorState::new_port_wired(
                    1,
                    vec![
                        LCColorPortBinding::new(0, LCColorEndpoint::Back),
                        LCColorPortBinding::new(0, LCColorEndpoint::Front),
                    ],
                    vec![
                        LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![3])
                            .unwrap(),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        let second_color = colors
            .intern(
                DynamicLCColorState::new_port_wired(
                    2,
                    vec![
                        LCColorPortBinding::new(0, LCColorEndpoint::Back),
                        LCColorPortBinding::new(0, LCColorEndpoint::Front),
                    ],
                    vec![
                        LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![3])
                            .unwrap(),
                    ],
                )
                .unwrap(),
            )
            .unwrap();
        let seed = digest(39);
        let first = source_reflection(&colors, first_color, 3, seed)
            .unwrap()
            .proof()
            .unwrap()
            .clone();
        let repeated = source_reflection(&colors, first_color, 3, seed)
            .unwrap()
            .proof()
            .unwrap()
            .clone();
        let changed_slot = source_reflection(&colors, first_color, 4, seed)
            .unwrap()
            .proof()
            .unwrap()
            .clone();
        let changed_color = source_reflection(&colors, second_color, 3, seed)
            .unwrap()
            .proof()
            .unwrap()
            .clone();

        assert_eq!(first, repeated);
        assert_ne!(first.proof_digest(), changed_slot.proof_digest());
        assert_ne!(first.proof_digest(), changed_color.proof_digest());
        assert_ne!(
            first.result_color_identity(),
            changed_color.result_color_identity()
        );
    }

    #[test]
    fn pure_adjoint_result_reflection_is_phase_composed_and_canonical() {
        let result = DynamicLCColorState::new_port_wired(
            1,
            vec![
                LCColorPortBinding::new(0, LCColorEndpoint::Back),
                LCColorPortBinding::new(0, LCColorEndpoint::Front),
            ],
            vec![
                LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![1, 3, 2]).unwrap(),
            ],
        )
        .unwrap();
        let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
        let left = proven_reflection(&result, minus_one, 43);
        let right = proven_reflection(&result, minus_one, 44);
        let local = TransitionReflectionProof::new(minus_one, [digest(45)], [digest(46)]).unwrap();
        let proof = current_reflection_candidate(&result, &[left, right], Some(&local))
            .unwrap()
            .unwrap();
        assert_eq!(proof.phase(), minus_one);
        assert_eq!(
            proof.result_color_identity(),
            dynamic_color_identity_digest(&result).unwrap()
        );
        assert!(proof.lineage_roots().contains(&digest(43)));
        assert!(proof.lineage_roots().contains(&digest(44)));
        assert!(proof.lineage_roots().contains(&digest(45)));
        assert!(proof.lineage_roots().contains(&digest(46)));
        assert!(proof.lineage_roots().contains(&local.proof_digest));
        assert!(pure_adjoint_word_is_canonical(
            result.pure_adjoint_word().unwrap()
        ));
        assert!(!pure_adjoint_word_is_canonical(&[3, 2, 1]));
    }

    fn reflection_test_current(
        color_id: DynamicLCColorStateId,
        reflection: CurrentReflection,
    ) -> PendingCurrent {
        PendingCurrent {
            key: CurrentCoreKey::new(
                digest(201),
                RecurrenceNodeKind::Current,
                0,
                color_id,
                vec![0, 1, 2],
                CanonicalMomentumLinearForm::new(vec![
                    MomentumTerm {
                        source_slot: 0,
                        coefficient: 1,
                    },
                    MomentumTerm {
                        source_slot: 1,
                        coefficient: 1,
                    },
                    MomentumTerm {
                        source_slot: 2,
                        coefficient: 1,
                    },
                ])
                .unwrap(),
                CurrentHelicityIdentity::all_flow_union(0),
                vec![0],
                0,
                vec![],
                CurrentSourceBinding::None,
                None,
            )
            .unwrap(),
            source_exact_factor: None,
            contributions: BTreeMap::new(),
            realized_pairing_rule_ids: BTreeSet::new(),
            reflection,
            reflection_certificate_id: None,
        }
    }

    fn reflection_test_states() -> (
        DynamicLCColorStateInterner,
        DynamicLCColorStateId,
        DynamicLCColorStateId,
    ) {
        let canonical = DynamicLCColorState::new_port_wired(
            1,
            vec![
                LCColorPortBinding::new(0, LCColorEndpoint::Back),
                LCColorPortBinding::new(0, LCColorEndpoint::Front),
            ],
            vec![
                LCColorComponent::new(LCColorComponentKind::AdjointSegment, vec![1, 3, 2]).unwrap(),
            ],
        )
        .unwrap();
        let reversed = canonical.reversed().unwrap();
        let mut states = DynamicLCColorStateInterner::default();
        let canonical_id = states.intern(canonical).unwrap();
        let reversed_id = states.intern(reversed).unwrap();
        (states, canonical_id, reversed_id)
    }

    #[test]
    fn late_unproved_reflection_keeps_both_stage_orientations() {
        let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
        let (mut color_states, canonical_id, reversed_id) = reflection_test_states();
        let mut late_downgrade =
            proven_interned_reflection(&color_states, reversed_id, minus_one, 51);
        late_downgrade.include(None).unwrap();
        let mut currents = vec![
            reflection_test_current(
                canonical_id,
                proven_interned_reflection(&color_states, canonical_id, minus_one, 50),
            ),
            reflection_test_current(reversed_id, late_downgrade),
        ];
        let mut current_ids = currents
            .iter()
            .enumerate()
            .map(|(id, current)| (current.key.clone(), id as u32))
            .collect::<BTreeMap<_, _>>();
        let mut target_bucket = vec![0, 1];
        let mut certificates = Vec::new();

        reconcile_stage_reflections(
            0,
            &mut color_states,
            &mut currents,
            &mut current_ids,
            &mut target_bucket,
            &mut certificates,
            4,
        )
        .unwrap();

        assert_eq!(currents.len(), 2);
        assert_eq!(target_bucket, [0, 1]);
        assert_eq!(current_ids.len(), 2);
        assert!(certificates.is_empty());
        assert!(
            currents
                .iter()
                .all(|current| current.reflection == CurrentReflection::Unavailable)
        );
    }

    #[test]
    fn late_conflicting_reflection_keeps_both_stage_orientations() {
        let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
        let (mut color_states, canonical_id, reversed_id) = reflection_test_states();
        let mut late_downgrade =
            proven_interned_reflection(&color_states, reversed_id, minus_one, 53);
        late_downgrade
            .include(Some(
                CurrentReflectionProof::new(minus_one, [digest(54)], digest(55)).unwrap(),
            ))
            .unwrap();
        let mut currents = vec![
            reflection_test_current(
                canonical_id,
                proven_interned_reflection(&color_states, canonical_id, minus_one, 52),
            ),
            reflection_test_current(reversed_id, late_downgrade),
        ];
        let mut current_ids = currents
            .iter()
            .enumerate()
            .map(|(id, current)| (current.key.clone(), id as u32))
            .collect::<BTreeMap<_, _>>();
        let mut target_bucket = vec![0, 1];
        let mut certificates = Vec::new();

        reconcile_stage_reflections(
            0,
            &mut color_states,
            &mut currents,
            &mut current_ids,
            &mut target_bucket,
            &mut certificates,
            4,
        )
        .unwrap();

        assert_eq!(currents.len(), 2);
        assert_eq!(target_bucket, [0, 1]);
        assert_eq!(current_ids.len(), 2);
        assert!(certificates.is_empty());
        assert!(
            currents
                .iter()
                .all(|current| current.reflection == CurrentReflection::Unavailable)
        );
    }

    #[test]
    fn nonreciprocal_reflection_proofs_keep_both_stage_orientations() {
        let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
        let (mut color_states, canonical_id, reversed_id) = reflection_test_states();
        let mut currents = vec![
            reflection_test_current(
                canonical_id,
                proven_interned_reflection(&color_states, canonical_id, minus_one, 56),
            ),
            reflection_test_current(
                reversed_id,
                proven_interned_reflection(
                    &color_states,
                    reversed_id,
                    ExactComplexRational::ONE,
                    57,
                ),
            ),
        ];
        let mut current_ids = currents
            .iter()
            .enumerate()
            .map(|(id, current)| (current.key.clone(), id as u32))
            .collect::<BTreeMap<_, _>>();
        let mut target_bucket = vec![0, 1];
        let mut certificates = Vec::new();

        reconcile_stage_reflections(
            0,
            &mut color_states,
            &mut currents,
            &mut current_ids,
            &mut target_bucket,
            &mut certificates,
            4,
        )
        .unwrap();

        assert_eq!(currents.len(), 2);
        assert_eq!(target_bucket, [0, 1]);
        assert_eq!(current_ids.len(), 2);
        assert!(certificates.is_empty());
        assert!(
            currents
                .iter()
                .all(|current| current.reflection == CurrentReflection::Unavailable)
        );
    }

    #[test]
    fn complete_reciprocal_reflection_prunes_only_noncanonical_orientation() {
        let minus_one = ExactComplexRational::ONE.checked_neg().unwrap();
        let (mut color_states, canonical_id, reversed_id) = reflection_test_states();
        let mut currents = vec![
            reflection_test_current(
                canonical_id,
                proven_interned_reflection(&color_states, canonical_id, minus_one, 58),
            ),
            reflection_test_current(
                reversed_id,
                proven_interned_reflection(&color_states, reversed_id, minus_one, 59),
            ),
        ];
        let mut current_ids = currents
            .iter()
            .enumerate()
            .map(|(id, current)| (current.key.clone(), id as u32))
            .collect::<BTreeMap<_, _>>();
        let mut target_bucket = vec![0, 1];
        let mut certificates = Vec::new();

        reconcile_stage_reflections(
            0,
            &mut color_states,
            &mut currents,
            &mut current_ids,
            &mut target_bucket,
            &mut certificates,
            4,
        )
        .unwrap();

        assert_eq!(currents.len(), 1);
        assert_eq!(target_bucket, [0]);
        assert_eq!(current_ids.len(), 1);
        assert_eq!(certificates.len(), 1);
        let certificate = &certificates[0];
        assert_eq!(certificate.id, 0);
        assert_eq!(certificate.canonical_old_current_id, 0);
        assert_eq!(certificate.reflected_old_current_id, 1);
        assert_eq!(certificate.canonical_phase, minus_one);
        assert_eq!(certificate.reflected_phase, minus_one);
        assert_eq!(certificate.source_permutation.as_ref(), [0, 2, 1, 3]);
        assert!(!certificate.fixed_point);
        assert_eq!(certificate.orbit_size, 2);
        validate_pending_reflection_certificates(&certificates).unwrap();
        let mut stale_lineage = certificates.clone();
        stale_lineage[0].canonical_lineage_digest = digest(60);
        assert!(validate_pending_reflection_certificates(&stale_lineage).is_err());
        assert_eq!(currents[0].reflection.phase(), Some(minus_one));
        let color = color_states
            .get(currents[0].key.dynamic_lc_color_state_id())
            .unwrap();
        assert!(pure_adjoint_word_is_canonical(
            color.pure_adjoint_word().unwrap()
        ));
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
                realized_pairing_rule_ids: BTreeSet::new(),
                reflection: CurrentReflection::Unavailable,
                reflection_certificate_id: None,
            });
            currents_by_size[0].push(id);
        }

        let mut reflection_certificates = Vec::new();
        let structural_demands = StructuralDemandIndex::new(
            &process,
            &template,
            &template_catalog,
            &BTreeSet::from([0]),
            &currents,
        )
        .unwrap();
        let stage_diagnostics = build_internal_currents(
            template.catalog_digest,
            &process,
            None,
            &template,
            &template_catalog,
            &TransitionReflectionIndex::new(&template, &template_catalog).unwrap(),
            &[],
            &BTreeMap::new(),
            &color_targets,
            &structural_demands,
            &mut color_states,
            &mut currents,
            &mut current_ids,
            &mut currents_by_size,
            &mut reflection_certificates,
            external_count.saturating_add(1),
            &mut |_| Ok(()),
        )
        .unwrap();
        let closures = build_closures(
            RecurrenceStrategy::AllFlowUnion,
            &process,
            &process_catalog,
            None,
            &template,
            &template_catalog,
            &color_states,
            &currents,
            &BTreeSet::from([0]),
            &stage_diagnostics,
            &reflection_certificates,
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
            reflection_certificates,
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
        let adjoint_state = |slots: Vec<u32>| {
            DynamicLCColorState::new_port_wired(
                0,
                vec![
                    LCColorPortBinding::new(0, LCColorEndpoint::Back),
                    LCColorPortBinding::new(0, LCColorEndpoint::Front),
                ],
                vec![LCColorComponent::new(LCColorComponentKind::AdjointSegment, slots).unwrap()],
            )
            .unwrap()
        };

        assert!(targets.accepts(&state(vec![2, 3, 4])));
        assert!(!targets.accepts(&state(vec![4, 3, 2])));
        assert!(
            targets
                .accepts_up_to_reflection(&adjoint_state(vec![4, 3, 2]))
                .unwrap()
        );
        assert!(!targets.accepts(&state(vec![2, 4])));
        assert!(
            !targets
                .accepts_up_to_reflection(&state(vec![2, 4]))
                .unwrap()
        );
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
        // Closure-rooted admission materializes no current or contribution
        // outside the final exact schedule for this scalar grammar.
        for (name, external_count, expected_constructed, expected_schedule) in [
            ("three-point", 3, (4, 1, 1), (4, 1, 1)),
            ("four-point", 4, (8, 6, 1), (8, 6, 1)),
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
            [(2, 6, 6, 6, 6, 3), (3, 12, 6, 6, 6, 3)],
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

    #[test]
    fn replay_momentum_signs_cross_initial_and_final_sources() {
        assert_eq!(
            replay_momentum_signs(&[-1, -1, 1, 1], &[0, 3, 2, 1]).unwrap(),
            [1, -1, 1, -1]
        );
        assert_eq!(
            replay_momentum_signs(&[-1, -1, 1, 1], &[0, 1, 3, 2]).unwrap(),
            [1, 1, 1, 1]
        );
    }
}
