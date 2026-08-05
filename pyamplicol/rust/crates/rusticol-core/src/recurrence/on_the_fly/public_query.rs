// SPDX-License-Identifier: 0BSD

use super::*;

/// One selected source state in a decoded query.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct OnTheFlySelectedSourceV1 {
    pub(crate) source_slot: u32,
    pub(crate) state_index: u32,
    pub(crate) public_helicity: i32,
}

/// One exact physical LC selector, already decoded before construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DecodedLcQueryV1 {
    pub(super) seed_digest: SemanticDigest,
    pub(super) public_flow_id: u32,
    pub(super) selected_sources: Box<[OnTheFlySelectedSourceV1]>,
    pub(super) target_components: Box<[LCColorComponent]>,
    pub(super) closure_anchor_slot: u32,
    pub(super) pairing_rule_id: Option<u32>,
    pub(super) pairing_proof_digest: Option<SemanticDigest>,
    pub(super) pairing_source_slot_permutation: Box<[u32]>,
    pub(super) pairing_source_lineage: Box<[u32]>,
    pub(super) pairing_fermion_parity: i32,
    pub(super) selector_digest: SemanticDigest,
    pub(super) semantic_digest: SemanticDigest,
}

impl DecodedLcQueryV1 {
    pub(crate) fn from_authenticated_public_flow(
        seed: &OnTheFlyProcessSeedV1,
        public_flow_id: u32,
        public_helicities: &[i32],
    ) -> RusticolResult<Self> {
        if public_helicities.len() != seed.source_anchors.len() {
            return Err(invalid(
                "public-helicity selector does not cover every source slot exactly once",
            ));
        }
        let flow = seed
            .public_flows
            .binary_search_by_key(&public_flow_id, |flow| flow.flow_id)
            .ok()
            .map(|index| &seed.public_flows[index])
            .ok_or_else(|| invalid("requested public flow is not retained by the process"))?;
        let mut selected_sources = Vec::new();
        selected_sources
            .try_reserve_exact(public_helicities.len())
            .map_err(|error| invalid(format!("selected-source allocation failed: {error}")))?;
        for (source_slot, public_helicity) in public_helicities.iter().copied().enumerate() {
            let anchor = &seed.source_anchors[source_slot];
            let matches = anchor
                .states
                .iter()
                .filter(|state| state.public_helicity == public_helicity)
                .collect::<Vec<_>>();
            let [state] = matches.as_slice() else {
                return Err(invalid(format!(
                    "source slot {source_slot} has {} states for public helicity {public_helicity}, expected exactly one",
                    matches.len(),
                )));
            };
            selected_sources.push(OnTheFlySelectedSourceV1 {
                source_slot: checked_u32(source_slot, "selected source slot")?,
                state_index: state.state_index,
                public_helicity,
            });
        }
        let mut colored_slots = BTreeSet::new();
        for component in &flow.target_components {
            for source_slot in component.source_slots() {
                if *source_slot as usize >= selected_sources.len()
                    || !colored_slots.insert(*source_slot)
                {
                    return Err(invalid(
                        "decoded LC components repeat or exceed the source-slot domain",
                    ));
                }
            }
        }

        let seed_digest = seed.semantic_digest();
        let mut query = Self {
            seed_digest,
            public_flow_id,
            selected_sources: selected_sources.into_boxed_slice(),
            target_components: flow.target_components.clone(),
            closure_anchor_slot: flow.closure_anchor_slot,
            pairing_rule_id: flow.pairing_rule_id,
            pairing_proof_digest: flow.pairing_proof_digest,
            pairing_source_slot_permutation: flow.pairing_source_slot_permutation.clone(),
            pairing_source_lineage: flow.pairing_source_lineage.clone(),
            pairing_fermion_parity: flow.pairing_fermion_parity,
            selector_digest: flow.semantic_digest,
            // Replaced immediately below by the digest over every field.
            semantic_digest: SemanticDigest::new([1; 32])?,
        };
        query.semantic_digest = query.compute_digest()?;
        Ok(query)
    }

    fn compute_digest(&self) -> RusticolResult<SemanticDigest> {
        let mut hash = Sha256::new();
        hash.update(ON_THE_FLY_QUERY_DOMAIN);
        hash_digest(&mut hash, self.seed_digest);
        hash_digest(&mut hash, self.selector_digest);
        hash.update(self.public_flow_id.to_le_bytes());
        hash.update(self.closure_anchor_slot.to_le_bytes());
        hash.update(self.pairing_rule_id.unwrap_or(MISSING_U32).to_le_bytes());
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
}
