// SPDX-License-Identifier: 0BSD

use super::source_seed::validate_permutation;
use super::*;

/// One selected source state in a decoded query.  Source slots are always in
/// construction order after applying the seed's authenticated gather map.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct OnTheFlySelectedSourceV1 {
    pub(crate) source_slot: u32,
    pub(crate) state_index: u32,
    pub(crate) public_helicity: i32,
}

/// O(k) proof for one selected permutation of an identical-species pairing
/// class.  No factorial-size pairing catalog or numeric permutation rank is
/// retained.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlyPairingClassProofV1 {
    pub(crate) pairing_class_digest: SemanticDigest,
    /// Public antifundamental slots, in canonical fundamental-endpoint order.
    pub(crate) selected_antifundamental_public_slots: Box<[u32]>,
    /// Lehmer digits for the same selected sequence.
    pub(crate) reference_to_selected_ranks: Box<[u32]>,
}

impl OnTheFlyPairingClassProofV1 {
    pub(crate) fn new(
        pairing_class_digest: SemanticDigest,
        selected_antifundamental_public_slots: Vec<u32>,
        reference_to_selected_ranks: Vec<u32>,
    ) -> Self {
        Self {
            pairing_class_digest,
            selected_antifundamental_public_slots: selected_antifundamental_public_slots
                .into_boxed_slice(),
            reference_to_selected_ranks: reference_to_selected_ranks.into_boxed_slice(),
        }
    }
}

/// One already-decoded public LC selector.  Slots use public external order;
/// [`DecodedLcQueryV1::new`] authenticates and maps them to construction order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum OnTheFlyLcSelectorV1 {
    Singlet,
    SingleTrace {
        public_word: Box<[u32]>,
    },
    OpenLines {
        public_blocks: Box<[Box<[u32]>]>,
        pairing_proofs: Box<[OnTheFlyPairingClassProofV1]>,
    },
}

impl OnTheFlyLcSelectorV1 {
    pub(crate) fn single_trace(public_word: Vec<u32>) -> Self {
        Self::SingleTrace {
            public_word: public_word.into_boxed_slice(),
        }
    }

    pub(crate) fn open_lines(
        public_blocks: Vec<Vec<u32>>,
        pairing_proofs: Vec<OnTheFlyPairingClassProofV1>,
    ) -> Self {
        Self::OpenLines {
            public_blocks: public_blocks
                .into_iter()
                .map(Vec::into_boxed_slice)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            pairing_proofs: pairing_proofs.into_boxed_slice(),
        }
    }
}

/// One exact physical LC selector.  It contains only the selected structural
/// values; selector identity is their digest and never a process-wide flow ID.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DecodedLcQueryV1 {
    pub(super) seed_digest: SemanticDigest,
    pub(super) external_permutation: Box<[u32]>,
    pub(super) selected_sources: Box<[OnTheFlySelectedSourceV1]>,
    pub(super) target_components: Box<[LCColorComponent]>,
    pub(super) closure_anchor_slot: u32,
    pub(super) pairing_endpoint_pairs: Box<[[u32; 2]]>,
    pub(super) pairing_proof_digest: Option<SemanticDigest>,
    pub(super) pairing_source_slot_permutation: Box<[u32]>,
    pub(super) pairing_source_lineage: Box<[u32]>,
    pub(super) pairing_fermion_parity: i32,
    pub(super) selector_digest: SemanticDigest,
    pub(super) semantic_digest: SemanticDigest,
}

impl DecodedLcQueryV1 {
    pub(crate) fn new(
        seed: &OnTheFlyProcessSeedV1,
        external_permutation: Vec<u32>,
        public_helicities: &[i32],
        selector: OnTheFlyLcSelectorV1,
    ) -> RusticolResult<Self> {
        if external_permutation.as_slice() != seed.external_permutation.as_ref() {
            return Err(integrity(
                "query external permutation differs from its compact seed",
            ));
        }
        validate_permutation(
            &external_permutation,
            seed.source_anchors.len(),
            "query external permutation",
        )?;
        if public_helicities.len() != seed.source_anchors.len() {
            return Err(invalid(
                "public-helicity selector does not cover every external slot exactly once",
            ));
        }
        let inverse_permutation = inverse_permutation(&external_permutation)?;
        let selected_sources = selected_sources(seed, public_helicities)?;
        let decoded = decode_selector(seed, &inverse_permutation, selector)?;
        let seed_digest = seed.semantic_digest();
        let mut query = Self {
            seed_digest,
            external_permutation: external_permutation.into_boxed_slice(),
            selected_sources: selected_sources.into_boxed_slice(),
            target_components: decoded.target_components.into_boxed_slice(),
            closure_anchor_slot: decoded.closure_anchor_slot,
            pairing_endpoint_pairs: decoded.pairing_endpoint_pairs.into_boxed_slice(),
            pairing_proof_digest: decoded.pairing_proof_digest,
            pairing_source_slot_permutation: decoded
                .pairing_source_slot_permutation
                .into_boxed_slice(),
            pairing_source_lineage: decoded.pairing_source_lineage.into_boxed_slice(),
            pairing_fermion_parity: decoded.pairing_fermion_parity,
            selector_digest: decoded.selector_digest,
            semantic_digest: SemanticDigest::new([1; 32])?,
        };
        query.semantic_digest = query.compute_digest()?;
        Ok(query)
    }

    fn compute_digest(&self) -> RusticolResult<SemanticDigest> {
        let mut hash = Sha256::new();
        hash.update(b"pyamplicol-on-the-fly-decoded-lc-query-v2\0");
        hash_digest(&mut hash, self.seed_digest);
        hash_digest(&mut hash, self.selector_digest);
        hash_len(
            &mut hash,
            self.external_permutation.len(),
            "query external permutation",
        )?;
        for source_slot in &self.external_permutation {
            hash.update(source_slot.to_le_bytes());
        }
        hash.update(self.closure_anchor_slot.to_le_bytes());
        match self.pairing_proof_digest {
            None => hash.update([0]),
            Some(digest) => {
                hash.update([1]);
                hash_digest(&mut hash, digest);
            }
        }
        hash.update(self.pairing_fermion_parity.to_le_bytes());
        hash_len(
            &mut hash,
            self.pairing_endpoint_pairs.len(),
            "query pairing endpoint pairs",
        )?;
        for pair in &self.pairing_endpoint_pairs {
            hash.update(pair[0].to_le_bytes());
            hash.update(pair[1].to_le_bytes());
        }
        hash_len(
            &mut hash,
            self.pairing_source_slot_permutation.len(),
            "query pairing source permutation",
        )?;
        for source_slot in &self.pairing_source_slot_permutation {
            hash.update(source_slot.to_le_bytes());
        }
        hash_len(
            &mut hash,
            self.pairing_source_lineage.len(),
            "query pairing source lineage",
        )?;
        for lineage in &self.pairing_source_lineage {
            hash.update(lineage.to_le_bytes());
        }
        hash_len(&mut hash, self.selected_sources.len(), "selected sources")?;
        for selected in &self.selected_sources {
            hash.update(selected.source_slot.to_le_bytes());
            hash.update(selected.state_index.to_le_bytes());
            hash.update(selected.public_helicity.to_le_bytes());
        }
        hash_len(&mut hash, self.target_components.len(), "target components")?;
        for component in &self.target_components {
            hash.update([component.kind() as u8]);
            hash_len(&mut hash, component.source_slots().len(), "component word")?;
            for slot in component.source_slots() {
                hash.update(slot.to_le_bytes());
            }
        }
        final_digest(hash)
    }

    pub(crate) const fn semantic_digest(&self) -> SemanticDigest {
        self.semantic_digest
    }

    pub(super) fn selected_pairing_compatible(
        &self,
        support_source_slots: &[u32],
        carries_colored_fermion_line: bool,
    ) -> bool {
        let crossing_count = self
            .pairing_endpoint_pairs
            .iter()
            .filter(|pair| {
                support_source_slots.binary_search(&pair[0]).is_ok()
                    != support_source_slots.binary_search(&pair[1]).is_ok()
            })
            .count();
        crossing_count == usize::from(carries_colored_fermion_line)
    }
}

struct DecodedSelector {
    target_components: Vec<LCColorComponent>,
    closure_anchor_slot: u32,
    pairing_endpoint_pairs: Vec<[u32; 2]>,
    pairing_proof_digest: Option<SemanticDigest>,
    pairing_source_slot_permutation: Vec<u32>,
    pairing_source_lineage: Vec<u32>,
    pairing_fermion_parity: i32,
    selector_digest: SemanticDigest,
}

fn selected_sources(
    seed: &OnTheFlyProcessSeedV1,
    public_helicities: &[i32],
) -> RusticolResult<Vec<OnTheFlySelectedSourceV1>> {
    seed.source_anchors
        .iter()
        .map(|anchor| {
            let public_slot = *seed
                .external_permutation
                .get(anchor.source_slot as usize)
                .ok_or_else(|| integrity("external permutation source slot is absent"))?;
            let public_helicity = public_helicities[public_slot as usize];
            let matches = anchor
                .states
                .iter()
                .filter(|state| state.public_helicity == public_helicity)
                .collect::<Vec<_>>();
            let [state] = matches.as_slice() else {
                return Err(invalid(format!(
                    "source slot {} has {} states for public helicity {public_helicity}, expected exactly one",
                    anchor.source_slot,
                    matches.len(),
                )));
            };
            Ok(OnTheFlySelectedSourceV1 {
                source_slot: anchor.source_slot,
                state_index: state.state_index,
                public_helicity,
            })
        })
        .collect()
}

fn inverse_permutation(permutation: &[u32]) -> RusticolResult<Vec<u32>> {
    let mut inverse = vec![MISSING_U32; permutation.len()];
    for (construction_slot, public_slot) in permutation.iter().copied().enumerate() {
        let entry = inverse
            .get_mut(public_slot as usize)
            .ok_or_else(|| invalid("external permutation public slot is out of bounds"))?;
        if *entry != MISSING_U32 {
            return Err(invalid("external permutation repeats a public slot"));
        }
        *entry = checked_u32(construction_slot, "construction source slot")?;
    }
    if inverse.contains(&MISSING_U32) {
        return Err(invalid("external permutation omits a public slot"));
    }
    Ok(inverse)
}

fn map_public_word(word: &[u32], inverse: &[u32], label: &str) -> RusticolResult<Vec<u32>> {
    if word.is_empty() {
        return Err(invalid(format!("{label} is empty")));
    }
    word.iter()
        .map(|public_slot| {
            inverse
                .get(*public_slot as usize)
                .copied()
                .ok_or_else(|| invalid(format!("{label} public slot is out of bounds")))
        })
        .collect()
}

fn colored_source_slots(seed: &OnTheFlyProcessSeedV1) -> BTreeSet<u32> {
    seed.source_anchors
        .iter()
        .filter(|anchor| anchor.color_role.is_colored())
        .map(|anchor| anchor.source_slot)
        .collect()
}

fn require_exact_colored_coverage(
    seed: &OnTheFlyProcessSeedV1,
    words: &[Vec<u32>],
) -> RusticolResult<()> {
    let mut observed = BTreeSet::new();
    for slot in words.iter().flatten().copied() {
        if !observed.insert(slot) {
            return Err(invalid("decoded selector repeats a colored source slot"));
        }
    }
    if observed != colored_source_slots(seed) {
        return Err(invalid(
            "decoded selector does not cover every colored source exactly once",
        ));
    }
    Ok(())
}

fn decode_selector(
    seed: &OnTheFlyProcessSeedV1,
    inverse: &[u32],
    selector: OnTheFlyLcSelectorV1,
) -> RusticolResult<DecodedSelector> {
    match selector {
        OnTheFlyLcSelectorV1::Singlet => decode_singlet(seed),
        OnTheFlyLcSelectorV1::SingleTrace { public_word } => {
            decode_single_trace(seed, inverse, &public_word)
        }
        OnTheFlyLcSelectorV1::OpenLines {
            public_blocks,
            pairing_proofs,
        } => decode_open_lines(seed, inverse, &public_blocks, &pairing_proofs),
    }
}

fn decode_singlet(seed: &OnTheFlyProcessSeedV1) -> RusticolResult<DecodedSelector> {
    if seed
        .source_anchors
        .iter()
        .any(|anchor| anchor.color_role.is_colored())
    {
        return Err(invalid(
            "singlet selector is invalid for a process with colored external sources",
        ));
    }
    if !seed.pairing_classes.is_empty() {
        return Err(integrity(
            "singlet seed unexpectedly contains pairing classes",
        ));
    }
    let closure_anchor_slot = seed
        .source_anchors
        .iter()
        .find(|anchor| anchor.is_fermionic)
        .unwrap_or(&seed.source_anchors[0])
        .source_slot;
    let selector_digest = selector_digest(&[], &[], None)?;
    Ok(DecodedSelector {
        target_components: Vec::new(),
        closure_anchor_slot,
        pairing_endpoint_pairs: Vec::new(),
        pairing_proof_digest: None,
        pairing_source_slot_permutation: identity_permutation(seed.source_anchors.len())?,
        pairing_source_lineage: vec![MISSING_U32; seed.source_anchors.len()],
        pairing_fermion_parity: 1,
        selector_digest,
    })
}

fn decode_single_trace(
    seed: &OnTheFlyProcessSeedV1,
    inverse: &[u32],
    public_word: &[u32],
) -> RusticolResult<DecodedSelector> {
    if !seed.pairing_classes.is_empty() {
        return Err(invalid(
            "single-trace selector is invalid for a seed with open-line pairing classes",
        ));
    }
    let word = map_public_word(public_word, inverse, "single-trace word")?;
    require_exact_colored_coverage(seed, std::slice::from_ref(&word))?;
    if word.iter().any(|slot| {
        seed.source_anchors[*slot as usize].color_role != OnTheFlyExternalColorRoleV1::Adjoint
    }) {
        return Err(invalid(
            "single-trace selector contains a non-adjoint external source",
        ));
    }
    let closure_anchor_slot = *word
        .last()
        .ok_or_else(|| invalid("single-trace word is empty"))?;
    let component = LCColorComponent::new(LCColorComponentKind::Trace, word)?;
    let selector_digest = selector_digest(std::slice::from_ref(&component), &[], None)?;
    Ok(DecodedSelector {
        target_components: vec![component],
        closure_anchor_slot,
        pairing_endpoint_pairs: Vec::new(),
        pairing_proof_digest: None,
        pairing_source_slot_permutation: identity_permutation(seed.source_anchors.len())?,
        pairing_source_lineage: vec![MISSING_U32; seed.source_anchors.len()],
        pairing_fermion_parity: 1,
        selector_digest,
    })
}

fn decode_open_lines(
    seed: &OnTheFlyProcessSeedV1,
    inverse: &[u32],
    public_blocks: &[Box<[u32]>],
    pairing_proofs: &[OnTheFlyPairingClassProofV1],
) -> RusticolResult<DecodedSelector> {
    if seed.pairing_classes.is_empty() || public_blocks.is_empty() {
        return Err(invalid(
            "open-line selector requires pairing classes and nonempty blocks",
        ));
    }
    let mut blocks = public_blocks
        .iter()
        .map(|word| map_public_word(word, inverse, "open-line block"))
        .collect::<RusticolResult<Vec<_>>>()?;
    require_exact_colored_coverage(seed, &blocks)?;
    for block in &blocks {
        if block.len() < 2
            || seed.source_anchors[block[0] as usize].color_role
                != OnTheFlyExternalColorRoleV1::Fundamental
            || seed.source_anchors[*block.last().expect("nonempty") as usize].color_role
                != OnTheFlyExternalColorRoleV1::Antifundamental
            || block[1..block.len() - 1].iter().any(|slot| {
                seed.source_anchors[*slot as usize].color_role
                    != OnTheFlyExternalColorRoleV1::Adjoint
            })
        {
            return Err(invalid(
                "open-line block must be fundamental, zero or more adjoints, antifundamental",
            ));
        }
    }

    let (endpoint_pairs, proof_digest, source_permutation, source_lineage, parity) =
        validate_pairing_proofs(seed, inverse, &blocks, pairing_proofs)?;
    let minimum_colored_slot = colored_source_slots(seed)
        .into_iter()
        .next()
        .ok_or_else(|| invalid("open-line selector has no colored source"))?;
    let first_block = blocks
        .iter()
        .position(|block| block.contains(&minimum_colored_slot))
        .ok_or_else(|| integrity("minimum colored source is absent from open-line blocks"))?;
    blocks.rotate_left(first_block);
    let closure_anchor_slot = *blocks
        .last()
        .and_then(|block| block.last())
        .ok_or_else(|| invalid("open-line selector has no closure anchor"))?;
    let mut components = blocks
        .into_iter()
        .map(|block| LCColorComponent::new(LCColorComponentKind::OpenString, block))
        .collect::<RusticolResult<Vec<_>>>()?;
    components.sort_unstable();
    let selector_digest = selector_digest(&components, &endpoint_pairs, Some(proof_digest))?;
    Ok(DecodedSelector {
        target_components: components,
        closure_anchor_slot,
        pairing_endpoint_pairs: endpoint_pairs,
        pairing_proof_digest: Some(proof_digest),
        pairing_source_slot_permutation: source_permutation,
        pairing_source_lineage: source_lineage,
        pairing_fermion_parity: parity,
        selector_digest,
    })
}

type PairingValidation = (Vec<[u32; 2]>, SemanticDigest, Vec<u32>, Vec<u32>, i32);

fn validate_pairing_proofs(
    seed: &OnTheFlyProcessSeedV1,
    inverse: &[u32],
    blocks: &[Vec<u32>],
    pairing_proofs: &[OnTheFlyPairingClassProofV1],
) -> RusticolResult<PairingValidation> {
    let mut proof_by_digest = BTreeMap::new();
    for proof in pairing_proofs {
        if proof_by_digest
            .insert(proof.pairing_class_digest, proof)
            .is_some()
        {
            return Err(invalid("pairing proof repeats a class digest"));
        }
    }
    if proof_by_digest.len() != seed.pairing_classes.len() {
        return Err(invalid(
            "pairing proofs do not cover every compact pairing class exactly once",
        ));
    }
    let selected_by_fundamental = blocks
        .iter()
        .map(|block| (block[0], *block.last().expect("validated nonempty")))
        .collect::<BTreeMap<_, _>>();
    if selected_by_fundamental.len() != blocks.len() {
        return Err(invalid("open-line blocks repeat a fundamental endpoint"));
    }

    let mut endpoint_pairs = Vec::new();
    let mut source_permutation = identity_permutation(seed.source_anchors.len())?;
    let mut source_lineage = vec![MISSING_U32; seed.source_anchors.len()];
    let mut parity = 1_i32;
    let mut proof_hash = Sha256::new();
    proof_hash.update(b"pyamplicol-on-the-fly-pairing-proof-v1\0");
    for pairing_class in &seed.pairing_classes {
        let proof = proof_by_digest
            .remove(&pairing_class.semantic_digest)
            .ok_or_else(|| invalid("pairing class proof digest is absent"))?;
        let selected_antifundamentals = proof
            .selected_antifundamental_public_slots
            .iter()
            .map(|public_slot| {
                inverse
                    .get(*public_slot as usize)
                    .copied()
                    .ok_or_else(|| invalid("pairing proof public slot is out of bounds"))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let derived_antifundamentals = pairing_class
            .fundamental_endpoints
            .iter()
            .map(|endpoint| {
                selected_by_fundamental
                    .get(&endpoint.source_slot)
                    .copied()
                    .ok_or_else(|| invalid("pairing class fundamental has no open-line block"))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        if selected_antifundamentals != derived_antifundamentals {
            return Err(integrity(
                "pairing proof selected endpoints differ from the open-line blocks",
            ));
        }
        let reference = pairing_class
            .antifundamental_endpoints
            .iter()
            .map(|endpoint| endpoint.source_slot)
            .collect::<Vec<_>>();
        let digits = lehmer_digits_for_selected(&reference, &selected_antifundamentals)?;
        if digits.as_slice() != proof.reference_to_selected_ranks.as_ref() {
            return Err(integrity("pairing proof Lehmer digits are invalid"));
        }
        if digits.iter().map(|digit| u64::from(*digit)).sum::<u64>() % 2 == 1 {
            parity = -parity;
        }
        hash_digest(&mut proof_hash, pairing_class.semantic_digest);
        hash_len(
            &mut proof_hash,
            selected_antifundamentals.len(),
            "selected pairing",
        )?;
        for (reference_slot, selected_slot) in reference
            .iter()
            .copied()
            .zip(selected_antifundamentals.iter().copied())
        {
            source_permutation[reference_slot as usize] = selected_slot;
            proof_hash.update(reference_slot.to_le_bytes());
            proof_hash.update(selected_slot.to_le_bytes());
        }
        for digit in &digits {
            proof_hash.update(digit.to_le_bytes());
        }
        endpoint_pairs.extend(
            pairing_class
                .fundamental_endpoints
                .iter()
                .map(|endpoint| endpoint.source_slot)
                .zip(selected_antifundamentals)
                .map(|(fundamental, antifundamental)| [fundamental, antifundamental]),
        );
    }
    if !proof_by_digest.is_empty() {
        return Err(invalid("pairing proof contains an unknown class digest"));
    }
    endpoint_pairs.sort_unstable();
    for (line_id, pair) in endpoint_pairs.iter().copied().enumerate() {
        let line_id = checked_u32(line_id, "pairing line ID")?;
        for source_slot in pair {
            let entry = source_lineage
                .get_mut(source_slot as usize)
                .ok_or_else(|| integrity("pairing endpoint is outside the source domain"))?;
            if *entry != MISSING_U32 {
                return Err(integrity("pairing endpoint belongs to multiple lines"));
            }
            *entry = line_id;
        }
    }
    validate_permutation(
        &source_permutation,
        seed.source_anchors.len(),
        "pairing source permutation",
    )?;
    proof_hash.update(parity.to_le_bytes());
    Ok((
        endpoint_pairs,
        final_digest(proof_hash)?,
        source_permutation,
        source_lineage,
        parity,
    ))
}

fn lehmer_digits_for_selected(reference: &[u32], selected: &[u32]) -> RusticolResult<Vec<u32>> {
    if reference.len() != selected.len() {
        return Err(invalid("selected pairing length differs from its class"));
    }
    let mut remaining = reference.to_vec();
    let mut digits = Vec::new();
    digits
        .try_reserve_exact(selected.len())
        .map_err(|error| invalid(format!("Lehmer digit allocation failed: {error}")))?;
    for selected_slot in selected {
        let position = remaining
            .iter()
            .position(|candidate| candidate == selected_slot)
            .ok_or_else(|| invalid("selected pairing is not a class permutation"))?;
        digits.push(checked_u32(position, "Lehmer digit")?);
        remaining.remove(position);
    }
    if !remaining.is_empty() {
        return Err(invalid("selected pairing omits a class endpoint"));
    }
    for (index, digit) in digits.iter().copied().enumerate() {
        let remaining_len = digits.len() - index;
        if digit as usize >= remaining_len {
            return Err(integrity("Lehmer digit exceeds its remaining class domain"));
        }
    }
    Ok(digits)
}

fn identity_permutation(len: usize) -> RusticolResult<Vec<u32>> {
    (0..len)
        .map(|value| checked_u32(value, "identity source permutation"))
        .collect()
}

fn selector_digest(
    components: &[LCColorComponent],
    endpoint_pairs: &[[u32; 2]],
    pairing_proof_digest: Option<SemanticDigest>,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-on-the-fly-structural-selector-v1\0");
    hash_len(&mut hash, components.len(), "selector components")?;
    for component in components {
        hash.update([component.kind() as u8]);
        hash_len(&mut hash, component.source_slots().len(), "selector word")?;
        for source_slot in component.source_slots() {
            hash.update(source_slot.to_le_bytes());
        }
    }
    hash_len(&mut hash, endpoint_pairs.len(), "selector endpoint pairs")?;
    for pair in endpoint_pairs {
        hash.update(pair[0].to_le_bytes());
        hash.update(pair[1].to_le_bytes());
    }
    match pairing_proof_digest {
        None => hash.update([0]),
        Some(digest) => {
            hash.update([1]);
            hash_digest(&mut hash, digest);
        }
    }
    final_digest(hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).unwrap()
    }

    fn anchor(
        source_slot: u32,
        role: OnTheFlyExternalColorRoleV1,
        contract: Option<SemanticDigest>,
    ) -> OnTheFlySourceAnchorV1 {
        let fermionic = role.is_pairing_endpoint();
        let family = if fermionic {
            OnTheFlySourceWavefunctionFamilyV1::DiracFermion
        } else {
            OnTheFlySourceWavefunctionFamilyV1::Vector
        };
        let orientation = match role {
            OnTheFlyExternalColorRoleV1::Fundamental => OnTheFlySourceOrientationV1::Particle,
            OnTheFlyExternalColorRoleV1::Antifundamental => {
                OnTheFlySourceOrientationV1::Antiparticle
            }
            OnTheFlyExternalColorRoleV1::Singlet | OnTheFlyExternalColorRoleV1::Adjoint => {
                OnTheFlySourceOrientationV1::SelfConjugate
            }
        };
        let state = OnTheFlySourceStateV1::new(
            0,
            1,
            1,
            source_slot,
            source_slot,
            digest(20 + source_slot as u8),
            digest(40 + source_slot as u8),
            1,
            ExactComplexRational::ONE,
            1,
            0,
            vec![source_slot as i32 + 1],
            source_slot,
            digest(60 + source_slot as u8),
            family,
            orientation,
            None,
        )
        .unwrap();
        OnTheFlySourceAnchorV1::new(
            source_slot,
            100 + source_slot,
            role,
            fermionic,
            contract,
            vec![state],
        )
        .unwrap()
    }

    fn seed(
        roles: &[OnTheFlyExternalColorRoleV1],
        external_permutation: Vec<u32>,
    ) -> OnTheFlyProcessSeedV1 {
        let anchors = roles
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, role)| {
                let contract = role.is_pairing_endpoint().then(|| digest(100 + slot as u8));
                anchor(slot as u32, role, contract)
            })
            .collect::<Vec<_>>();
        let fundamental = anchors
            .iter()
            .filter(|anchor| anchor.color_role == OnTheFlyExternalColorRoleV1::Fundamental)
            .map(|anchor| OnTheFlyPairingEndpointV1 {
                source_slot: anchor.source_slot,
                source_contract_digest: anchor.pairing_source_contract_digest.unwrap(),
            })
            .collect::<Vec<_>>();
        let antifundamental = anchors
            .iter()
            .filter(|anchor| anchor.color_role == OnTheFlyExternalColorRoleV1::Antifundamental)
            .map(|anchor| OnTheFlyPairingEndpointV1 {
                source_slot: anchor.source_slot,
                source_contract_digest: anchor.pairing_source_contract_digest.unwrap(),
            })
            .collect::<Vec<_>>();
        let classes = if fundamental.is_empty() {
            Vec::new()
        } else {
            vec![
                OnTheFlyPairingClassV1::new("q", digest(88), fundamental, antifundamental).unwrap(),
            ]
        };
        OnTheFlyProcessSeedV1::new(
            digest(1),
            digest(2),
            digest(3),
            digest(4),
            digest(5),
            digest(6),
            "raw amplitude",
            ExactComplexRational::ONE,
            anchors,
            external_permutation,
            vec![Some(2)],
            classes,
        )
        .unwrap()
    }

    fn permutations(values: &[u32]) -> Vec<Vec<u32>> {
        fn visit(values: &mut Vec<u32>, at: usize, output: &mut Vec<Vec<u32>>) {
            if at == values.len() {
                output.push(values.clone());
                return;
            }
            for index in at..values.len() {
                values.swap(at, index);
                visit(values, at + 1, output);
                values.swap(at, index);
            }
        }
        let mut values = values.to_vec();
        let mut output = Vec::new();
        visit(&mut values, 0, &mut output);
        output
    }

    fn inversion_parity(values: &[u32]) -> i32 {
        let inversions = values
            .iter()
            .enumerate()
            .map(|(left, value)| {
                values[left + 1..]
                    .iter()
                    .filter(|right| value > *right)
                    .count()
            })
            .sum::<usize>();
        if inversions % 2 == 0 { 1 } else { -1 }
    }

    #[test]
    fn lehmer_proof_is_bounded_and_has_exact_permutation_parity() {
        for size in 0..=7_u32 {
            let reference = (0..size).collect::<Vec<_>>();
            for selected in permutations(&reference) {
                let digits = lehmer_digits_for_selected(&reference, &selected).unwrap();
                assert_eq!(digits.len(), selected.len());
                assert!(digits.iter().enumerate().all(|(index, digit)| {
                    (*digit as usize) < digits.len().saturating_sub(index)
                }));
                let parity = if digits.iter().map(|digit| u64::from(*digit)).sum::<u64>() % 2 == 0 {
                    1
                } else {
                    -1
                };
                assert_eq!(parity, inversion_parity(&selected));
            }
        }
    }

    #[test]
    fn lehmer_proof_rejects_duplicates_omissions_and_foreign_endpoints() {
        assert!(lehmer_digits_for_selected(&[3, 7, 11], &[3, 3, 11]).is_err());
        assert!(lehmer_digits_for_selected(&[3, 7, 11], &[3, 7]).is_err());
        assert!(lehmer_digits_for_selected(&[3, 7, 11], &[3, 7, 13]).is_err());
    }

    #[test]
    fn lehmer_proof_has_no_numeric_rank_ceiling() {
        let reference = (0..64_u32).collect::<Vec<_>>();
        let selected = reference.iter().rev().copied().collect::<Vec<_>>();
        let digits = lehmer_digits_for_selected(&reference, &selected).unwrap();
        assert_eq!(digits, (0..64_u32).rev().collect::<Vec<_>>());
        assert_eq!(
            digits.iter().map(|digit| u64::from(*digit)).sum::<u64>() % 2,
            0
        );
    }

    #[test]
    fn inverse_permutation_binds_public_and_construction_slots_exactly() {
        assert_eq!(inverse_permutation(&[2, 0, 3, 1]).unwrap(), [1, 3, 0, 2]);
        assert!(inverse_permutation(&[0, 0]).is_err());
        assert!(inverse_permutation(&[0, 2]).is_err());
    }

    #[test]
    fn selected_pairing_compatibility_tracks_only_the_requested_pairing() {
        let query = DecodedLcQueryV1 {
            seed_digest: SemanticDigest::new([1; 32]).unwrap(),
            external_permutation: vec![0, 1, 2, 3].into_boxed_slice(),
            selected_sources: Box::new([]),
            target_components: Box::new([]),
            closure_anchor_slot: 0,
            pairing_endpoint_pairs: vec![[0, 2], [1, 3]].into_boxed_slice(),
            pairing_proof_digest: None,
            pairing_source_slot_permutation: vec![0, 1, 2, 3].into_boxed_slice(),
            pairing_source_lineage: vec![0, 1, 0, 1].into_boxed_slice(),
            pairing_fermion_parity: 1,
            selector_digest: SemanticDigest::new([2; 32]).unwrap(),
            semantic_digest: SemanticDigest::new([3; 32]).unwrap(),
        };
        assert!(query.selected_pairing_compatible(&[0], true));
        assert!(query.selected_pairing_compatible(&[0, 2], false));
        assert!(!query.selected_pairing_compatible(&[0, 1], true));
    }

    #[test]
    fn selector_domains_are_derived_from_external_roles() {
        let singlet_seed = seed(
            &[
                OnTheFlyExternalColorRoleV1::Singlet,
                OnTheFlyExternalColorRoleV1::Singlet,
            ],
            vec![1, 0],
        );
        let singlet = DecodedLcQueryV1::new(
            &singlet_seed,
            vec![1, 0],
            &[1, 1],
            OnTheFlyLcSelectorV1::Singlet,
        )
        .unwrap();
        assert!(singlet.target_components.is_empty());
        assert_eq!(singlet.closure_anchor_slot, 0);

        let trace_seed = seed(
            &[
                OnTheFlyExternalColorRoleV1::Adjoint,
                OnTheFlyExternalColorRoleV1::Adjoint,
            ],
            vec![1, 0],
        );
        assert!(
            DecodedLcQueryV1::new(
                &trace_seed,
                vec![1, 0],
                &[1, 1],
                OnTheFlyLcSelectorV1::Singlet,
            )
            .is_err()
        );
        let trace = DecodedLcQueryV1::new(
            &trace_seed,
            vec![1, 0],
            &[1, 1],
            OnTheFlyLcSelectorV1::single_trace(vec![1, 0]),
        )
        .unwrap();
        assert_eq!(trace.target_components[0].source_slots(), [0, 1]);
        assert_eq!(trace.closure_anchor_slot, 1);
    }

    #[test]
    fn open_line_proof_binds_blocks_permutation_and_parity() {
        let open_seed = seed(
            &[
                OnTheFlyExternalColorRoleV1::Fundamental,
                OnTheFlyExternalColorRoleV1::Adjoint,
                OnTheFlyExternalColorRoleV1::Antifundamental,
            ],
            vec![2, 0, 1],
        );
        let class_digest = open_seed.pairing_classes[0].semantic_digest;
        let query = DecodedLcQueryV1::new(
            &open_seed,
            vec![2, 0, 1],
            &[1, 1, 1],
            OnTheFlyLcSelectorV1::open_lines(
                vec![vec![2, 0, 1]],
                vec![OnTheFlyPairingClassProofV1::new(
                    class_digest,
                    vec![1],
                    vec![0],
                )],
            ),
        )
        .unwrap();
        assert_eq!(query.target_components[0].source_slots(), [0, 1, 2]);
        assert_eq!(query.pairing_endpoint_pairs.as_ref(), [[0, 2]]);
        assert_eq!(query.pairing_source_slot_permutation.as_ref(), [0, 1, 2]);
        assert_eq!(query.pairing_source_lineage.as_ref(), [0, MISSING_U32, 0]);
        assert_eq!(query.pairing_fermion_parity, 1);

        assert!(
            DecodedLcQueryV1::new(
                &open_seed,
                vec![2, 0, 1],
                &[1, 1, 1],
                OnTheFlyLcSelectorV1::open_lines(
                    vec![vec![2, 0, 1]],
                    vec![OnTheFlyPairingClassProofV1::new(
                        class_digest,
                        vec![1],
                        vec![1],
                    )],
                ),
            )
            .is_err()
        );
    }
}
