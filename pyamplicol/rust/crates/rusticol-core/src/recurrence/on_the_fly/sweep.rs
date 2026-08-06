// SPDX-License-Identifier: 0BSD

use super::source_seed::validate_permutation;
use super::*;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct PendingContributionKey {
    pub(super) parent_current_ids: [u32; 2],
    pub(super) key: ContributionKey,
}

#[derive(Debug)]
pub(super) struct PendingCurrent {
    pub(super) key: CurrentCoreKey,
    pub(super) source_factor: Option<ExactComplexRational>,
    pub(super) contributions: BTreeMap<PendingContributionKey, ExactComplexRational>,
    /// Canonical proof-only lineage alternatives.  As in the established
    /// builder's realized-rule set, exact contribution factors remain owned
    /// by `contributions`; lineage does not multiply or normalize amplitudes.
    pub(super) pairing_lineages: Vec<PendingPairingLineage>,
    pub(super) stage: u32,
}

#[derive(Clone, Copy, Debug)]
struct PreparedContactOrbitLocation<'a> {
    transition_id: u32,
    contact_orbit: Option<&'a PreparedContactOrbitTransition>,
}

#[derive(Debug, Default)]
struct PreparedContactOrbitIndex<'a> {
    locations: Vec<PreparedContactOrbitLocation<'a>>,
}

fn reserve_on_the_fly_contact_locations<T>(
    locations: &mut Vec<T>,
    count: usize,
) -> RusticolResult<()> {
    locations.try_reserve_exact(count).map_err(|error| {
        invalid(format!(
            "contact-orbit transition-index allocation failed: {error}"
        ))
    })
}

impl<'a> PreparedContactOrbitIndex<'a> {
    fn new(transitions: &'a BTreeMap<(u32, u32), Vec<PreparedTransition>>) -> RusticolResult<Self> {
        if !transitions
            .values()
            .flatten()
            .any(|prepared| prepared.contact_orbit.is_some())
        {
            return Ok(Self::default());
        }
        let transition_count = transitions.values().try_fold(0usize, |count, rows| {
            count
                .checked_add(rows.len())
                .ok_or_else(|| invalid("contact-orbit transition count exceeds usize"))
        })?;
        let mut locations = Vec::new();
        reserve_on_the_fly_contact_locations(&mut locations, transition_count)?;
        for prepared in transitions.values().flatten() {
            locations.push(PreparedContactOrbitLocation {
                transition_id: prepared.row.id,
                contact_orbit: prepared.contact_orbit.as_ref(),
            });
        }
        locations.sort_unstable_by_key(|location| location.transition_id);
        if locations
            .windows(2)
            .any(|rows| rows[0].transition_id == rows[1].transition_id)
        {
            return Err(integrity(
                "contact-orbit transition index contains duplicate IDs",
            ));
        }
        Ok(Self { locations })
    }

    fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }

    fn get(&self, transition_id: u32) -> Option<&'a PreparedContactOrbitTransition> {
        self.locations
            .binary_search_by_key(&transition_id, |location| location.transition_id)
            .ok()
            .and_then(|index| self.locations.get(index))
            .and_then(|location| location.contact_orbit)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct OnTheFlyContactContributionToken {
    destination_id: usize,
    contribution_ordinal: usize,
}

#[derive(Debug)]
struct OnTheFlyContactOwnerPlan {
    stage_current_start: usize,
    selected_tokens: Vec<OnTheFlyContactContributionToken>,
    staged_contribution_count: usize,
    retained_contribution_count: usize,
}

impl OnTheFlyContactOwnerPlan {
    /// Apply a completely selected plan without allocating or failing.
    fn commit(self, currents: &mut [PendingCurrent]) {
        let mut selected_index = 0usize;
        let mut observed_count = 0usize;
        let mut retained_count = 0usize;
        for (destination_id, current) in currents
            .iter_mut()
            .enumerate()
            .skip(self.stage_current_start)
        {
            let mut contribution_ordinal = 0usize;
            current.contributions.retain(|_, _| {
                let token = OnTheFlyContactContributionToken {
                    destination_id,
                    contribution_ordinal,
                };
                contribution_ordinal += 1;
                observed_count += 1;
                let selected = self.selected_tokens.get(selected_index) == Some(&token);
                if selected {
                    selected_index += 1;
                    retained_count += 1;
                }
                selected
            });
        }
        debug_assert_eq!(observed_count, self.staged_contribution_count);
        debug_assert_eq!(retained_count, self.retained_contribution_count);
        debug_assert_eq!(selected_index, self.selected_tokens.len());
    }
}

fn reserve_on_the_fly_contact_candidates<T>(
    candidates: &mut Vec<T>,
    count: usize,
) -> RusticolResult<()> {
    candidates.try_reserve_exact(count).map_err(|error| {
        invalid(format!(
            "contact-orbit candidate allocation failed: {error}"
        ))
    })
}

fn plan_on_the_fly_contact_orbit_owners(
    stage_current_start: usize,
    contact_orbits: &PreparedContactOrbitIndex<'_>,
    currents: &[PendingCurrent],
) -> RusticolResult<Option<OnTheFlyContactOwnerPlan>> {
    plan_on_the_fly_contact_orbit_owners_with_resolver(
        stage_current_start,
        currents,
        |transition_id| contact_orbits.get(transition_id),
    )
}

fn plan_on_the_fly_contact_orbit_owners_with_resolver<'a, F>(
    stage_current_start: usize,
    currents: &'a [PendingCurrent],
    mut contact_orbit: F,
) -> RusticolResult<Option<OnTheFlyContactOwnerPlan>>
where
    F: FnMut(u32) -> Option<&'a PreparedContactOrbitTransition>,
{
    if stage_current_start > currents.len() {
        return Err(invalid(
            "contact-orbit stage current boundary exceeds current storage",
        ));
    }
    let has_certified_contact = currents[stage_current_start..].iter().any(|current| {
        current
            .contributions
            .keys()
            .any(|pending| contact_orbit(pending.key.transition_template_id()).is_some())
    });
    if !has_certified_contact {
        return Ok(None);
    }
    let staged_contribution_count =
        currents[stage_current_start..]
            .iter()
            .try_fold(0usize, |count, current| {
                count
                    .checked_add(current.contributions.len())
                    .ok_or_else(|| invalid("contact-orbit staged contribution count exceeds usize"))
            })?;
    let mut candidates = Vec::new();
    reserve_on_the_fly_contact_candidates(&mut candidates, staged_contribution_count)?;
    for (destination_id, current) in currents.iter().enumerate().skip(stage_current_start) {
        for (contribution_ordinal, pending) in current.contributions.keys().enumerate() {
            let token = OnTheFlyContactContributionToken {
                destination_id,
                contribution_ordinal,
            };
            let candidate = contact_orbit(pending.key.transition_template_id())
                .map(|contact_orbit| {
                    let left = currents
                        .get(pending.parent_current_ids[0] as usize)
                        .map(|parent| &parent.key)
                        .ok_or_else(|| integrity("contact-orbit left parent is absent"))?;
                    let right = currents
                        .get(pending.parent_current_ids[1] as usize)
                        .map(|parent| &parent.key)
                        .ok_or_else(|| integrity("contact-orbit right parent is absent"))?;
                    contact_orbit.owner_candidate(
                        &current.key,
                        [left, right],
                        pending.key.color_witness_term_id(),
                    )
                })
                .transpose()?;
            candidates.push((token, candidate));
        }
    }
    if candidates.len() != staged_contribution_count {
        return Err(integrity(
            "contact-orbit staged contribution snapshot changed length",
        ));
    }
    let selected_tokens = selected_contact_orbit_owner_tokens(candidates.into_iter())?;
    let retained_contribution_count = selected_tokens.len();
    if retained_contribution_count > staged_contribution_count {
        return Err(integrity(
            "contact-orbit retained contribution count exceeds snapshot",
        ));
    }
    Ok(Some(OnTheFlyContactOwnerPlan {
        stage_current_start,
        selected_tokens,
        staged_contribution_count,
        retained_contribution_count,
    }))
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct PendingPairingLineage {
    pub(super) completed_pairs: Vec<[u32; 2]>,
    pub(super) unmatched_endpoint: Option<u32>,
}

impl PendingPairingLineage {
    pub(super) fn source(seed: &OnTheFlyProcessSeedV1, source_slot: u32) -> Self {
        let unmatched_endpoint = seed.source_anchors[source_slot as usize]
            .color_role
            .is_pairing_endpoint()
            .then_some(source_slot);
        Self {
            completed_pairs: Vec::new(),
            unmatched_endpoint,
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct PendingClosureKey {
    pub(super) closure_template_id: u32,
    pub(super) quantum_flow_template_id: Option<u32>,
    pub(super) parent_current_ids: [u32; 2],
    pub(super) color_witness_term_id: LCColorWitnessTermId,
}

#[derive(Debug)]
pub(super) struct PendingClosure {
    pub(super) key: PendingClosureKey,
    pub(super) factor: ExactComplexRational,
    pub(super) component_coefficients: Box<[ExactComplexRational]>,
    pub(super) pairing_lineages: Vec<PendingPairingLineage>,
}

/// Query-local Wick-lineage proof selected only after physical closure and
/// complete-rectangle color-alias projection have identified a canonical
/// root-bearing representative.
#[derive(Debug, Eq, PartialEq)]
pub(super) struct ResolvedPairingOwnerV1 {
    pub(super) endpoint_pairs: Vec<[u32; 2]>,
    pub(super) proof_digest: Option<SemanticDigest>,
    pub(super) source_slot_permutation: Vec<u32>,
    pub(super) source_lineage: Vec<u32>,
    pub(super) fermion_parity: i32,
}

fn checked_pairing_capacity(left: usize, right: usize, extra: usize) -> RusticolResult<usize> {
    left.checked_add(right)
        .and_then(|combined| combined.checked_add(extra))
        .ok_or_else(|| invalid("pairing-lineage length exceeds usize"))
}

fn try_copy_pairing_pairs(source: &[[u32; 2]]) -> RusticolResult<Vec<[u32; 2]>> {
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(source.len())
        .map_err(|error| invalid(format!("pairing-lineage allocation failed: {error}")))?;
    copied.extend_from_slice(source);
    Ok(copied)
}

fn try_clone_pairing_lineage(
    source: &PendingPairingLineage,
) -> RusticolResult<PendingPairingLineage> {
    Ok(PendingPairingLineage {
        completed_pairs: try_copy_pairing_pairs(&source.completed_pairs)?,
        unmatched_endpoint: source.unmatched_endpoint,
    })
}

pub(super) fn try_clone_pairing_lineages(
    source: &[PendingPairingLineage],
) -> RusticolResult<Vec<PendingPairingLineage>> {
    let mut copied = Vec::new();
    copied
        .try_reserve_exact(source.len())
        .map_err(|error| invalid(format!("pairing-lineage set allocation failed: {error}")))?;
    for lineage in source {
        copied.push(try_clone_pairing_lineage(lineage)?);
    }
    Ok(copied)
}

fn pairing_endpoint_class(
    seed: &OnTheFlyProcessSeedV1,
    source_slot: u32,
) -> RusticolResult<(OnTheFlyExternalColorRoleV1, SemanticDigest)> {
    let anchor = seed
        .source_anchors
        .get(source_slot as usize)
        .ok_or_else(|| integrity("pairing-lineage endpoint is outside the source domain"))?;
    let contract = anchor
        .pairing_source_contract_digest
        .ok_or_else(|| integrity("pairing-lineage endpoint lacks its source contract"))?;
    let class = seed
        .pairing_classes
        .iter()
        .find(|class| {
            class
                .fundamental_endpoints
                .iter()
                .chain(class.antifundamental_endpoints.iter())
                .any(|endpoint| endpoint.source_slot == source_slot)
        })
        .ok_or_else(|| integrity("pairing-lineage endpoint lacks its compact class"))?;
    let class_endpoint = class
        .fundamental_endpoints
        .iter()
        .chain(class.antifundamental_endpoints.iter())
        .find(|endpoint| endpoint.source_slot == source_slot)
        .ok_or_else(|| integrity("pairing-lineage endpoint disappeared from its class"))?;
    if class_endpoint.source_contract_digest != contract {
        return Err(integrity(
            "pairing-lineage endpoint source contract differs from its class",
        ));
    }
    Ok((anchor.color_role, class.semantic_digest))
}

fn close_pairing_endpoints(
    seed: &OnTheFlyProcessSeedV1,
    left: u32,
    right: u32,
) -> RusticolResult<Option<[u32; 2]>> {
    let (left_role, left_class) = pairing_endpoint_class(seed, left)?;
    let (right_role, right_class) = pairing_endpoint_class(seed, right)?;
    if left_class != right_class {
        return Ok(None);
    }
    Ok(match (left_role, right_role) {
        (
            OnTheFlyExternalColorRoleV1::Fundamental,
            OnTheFlyExternalColorRoleV1::Antifundamental,
        ) => Some([left, right]),
        (
            OnTheFlyExternalColorRoleV1::Antifundamental,
            OnTheFlyExternalColorRoleV1::Fundamental,
        ) => Some([right, left]),
        _ => None,
    })
}

fn combine_pairing_lineage(
    seed: &OnTheFlyProcessSeedV1,
    left: &PendingPairingLineage,
    right: &PendingPairingLineage,
    carries_colored_fermion_line: bool,
) -> RusticolResult<Option<PendingPairingLineage>> {
    let possible_new_pair = usize::from(
        left.unmatched_endpoint.is_some()
            && right.unmatched_endpoint.is_some()
            && !carries_colored_fermion_line,
    );
    let capacity = checked_pairing_capacity(
        left.completed_pairs.len(),
        right.completed_pairs.len(),
        possible_new_pair,
    )?;
    let mut completed_pairs = Vec::new();
    completed_pairs
        .try_reserve_exact(capacity)
        .map_err(|error| invalid(format!("pairing-lineage allocation failed: {error}")))?;
    completed_pairs.extend_from_slice(&left.completed_pairs);
    completed_pairs.extend_from_slice(&right.completed_pairs);
    let unmatched_endpoint = match (
        left.unmatched_endpoint,
        right.unmatched_endpoint,
        carries_colored_fermion_line,
    ) {
        (Some(endpoint), None, true) | (None, Some(endpoint), true) => Some(endpoint),
        (None, None, false) => None,
        (Some(left), Some(right), false) => {
            let Some(pair) = close_pairing_endpoints(seed, left, right)? else {
                return Ok(None);
            };
            completed_pairs.push(pair);
            None
        }
        _ => return Ok(None),
    };
    completed_pairs.sort_unstable();
    if completed_pairs.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(integrity(
            "pairing lineage repeats a completed endpoint pair",
        ));
    }
    Ok(Some(PendingPairingLineage {
        completed_pairs,
        unmatched_endpoint,
    }))
}

pub(super) fn combine_pairing_lineage_sets(
    seed: &OnTheFlyProcessSeedV1,
    left: &[PendingPairingLineage],
    right: &[PendingPairingLineage],
    carries_colored_fermion_line: bool,
) -> RusticolResult<Vec<PendingPairingLineage>> {
    let capacity = left
        .len()
        .checked_mul(right.len())
        .ok_or_else(|| invalid("pairing-lineage product exceeds usize"))?;
    let mut result = Vec::new();
    result
        .try_reserve_exact(capacity)
        .map_err(|error| invalid(format!("pairing-lineage allocation failed: {error}")))?;
    for left in left {
        for right in right {
            if let Some(lineage) =
                combine_pairing_lineage(seed, left, right, carries_colored_fermion_line)?
            {
                result.push(lineage);
            }
        }
    }
    result.sort_unstable();
    result.dedup();
    Ok(result)
}

pub(super) fn extend_pairing_lineages(
    target: &mut Vec<PendingPairingLineage>,
    source: &[PendingPairingLineage],
) -> RusticolResult<()> {
    let mut additions = try_clone_pairing_lineages(source)?;
    target
        .try_reserve(additions.len())
        .map_err(|error| invalid(format!("pairing-lineage merge allocation failed: {error}")))?;
    target.append(&mut additions);
    target.sort_unstable();
    target.dedup();
    Ok(())
}

pub(super) fn complete_pairing_lineage(
    seed: &OnTheFlyProcessSeedV1,
    lineage: &PendingPairingLineage,
) -> RusticolResult<bool> {
    if lineage.unmatched_endpoint.is_some() {
        return Ok(false);
    }
    let expected_endpoint_count = seed
        .source_anchors
        .iter()
        .filter(|anchor| anchor.color_role.is_pairing_endpoint())
        .count();
    let observed_endpoint_count = lineage
        .completed_pairs
        .len()
        .checked_mul(2)
        .ok_or_else(|| invalid("pairing endpoint count exceeds usize"))?;
    if observed_endpoint_count != expected_endpoint_count {
        return Ok(false);
    }
    for (index, pair) in lineage.completed_pairs.iter().copied().enumerate() {
        if close_pairing_endpoints(seed, pair[0], pair[1])? != Some(pair) {
            return Ok(false);
        }
        if lineage.completed_pairs[..index]
            .iter()
            .flatten()
            .any(|endpoint| *endpoint == pair[0] || *endpoint == pair[1])
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn identity_source_permutation(source_count: usize) -> RusticolResult<Vec<u32>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(source_count)
        .map_err(|error| invalid(format!("pairing identity allocation failed: {error}")))?;
    for value in 0..source_count {
        result.push(checked_u32(value, "pairing identity source slot")?);
    }
    Ok(result)
}

fn lehmer_digits(reference: &[u32], selected: &[u32]) -> RusticolResult<Vec<u32>> {
    if reference.len() != selected.len() {
        return Err(invalid(
            "discovered pairing length differs from its compact class",
        ));
    }
    let mut remaining = Vec::new();
    remaining
        .try_reserve_exact(reference.len())
        .map_err(|error| invalid(format!("pairing permutation allocation failed: {error}")))?;
    remaining.extend_from_slice(reference);
    let mut digits = Vec::new();
    digits
        .try_reserve_exact(selected.len())
        .map_err(|error| invalid(format!("pairing Lehmer allocation failed: {error}")))?;
    for selected_slot in selected {
        let position = remaining
            .iter()
            .position(|candidate| candidate == selected_slot)
            .ok_or_else(|| {
                invalid("discovered pairing is not a permutation of its compact class")
            })?;
        digits.push(checked_u32(position, "pairing Lehmer digit")?);
        remaining.remove(position);
    }
    if !remaining.is_empty() {
        return Err(invalid("discovered pairing omits a compact-class endpoint"));
    }
    Ok(digits)
}

pub(super) fn resolve_projected_pairing_owner(
    seed: &OnTheFlyProcessSeedV1,
    closures: &[PendingClosure],
) -> RusticolResult<ResolvedPairingOwnerV1> {
    let mut source_slot_permutation = identity_source_permutation(seed.source_anchors.len())?;
    let mut source_lineage = Vec::new();
    source_lineage
        .try_reserve_exact(seed.source_anchors.len())
        .map_err(|error| invalid(format!("pairing source-lineage allocation failed: {error}")))?;
    source_lineage.resize(seed.source_anchors.len(), MISSING_U32);
    if seed.pairing_classes.is_empty() {
        if closures.iter().any(|closure| {
            closure.pairing_lineages.as_slice()
                != [PendingPairingLineage {
                    completed_pairs: Vec::new(),
                    unmatched_endpoint: None,
                }]
        }) {
            return Err(integrity(
                "pairing-free query retained a nontrivial closure lineage",
            ));
        }
        return Ok(ResolvedPairingOwnerV1 {
            endpoint_pairs: Vec::new(),
            proof_digest: None,
            source_slot_permutation,
            source_lineage,
            fermion_parity: 1,
        });
    }
    let owner = unique_projected_pairing_owner(closures)?;
    if !complete_pairing_lineage(seed, owner)? {
        return Err(integrity(
            "canonical projected closure has an incomplete Wick lineage",
        ));
    }
    let endpoint_pairs = try_copy_pairing_pairs(&owner.completed_pairs)?;
    if endpoint_pairs.iter().enumerate().any(|(index, pair)| {
        endpoint_pairs[..index]
            .iter()
            .any(|other| other[0] == pair[0])
    }) {
        return Err(integrity(
            "canonical projected closure repeats a fundamental endpoint",
        ));
    }
    let mut parity = 1_i32;
    let mut proof_hash = Sha256::new();
    proof_hash.update(b"pyamplicol-on-the-fly-discovered-pairing-owner-v1\0");
    for pairing_class in &seed.pairing_classes {
        let mut reference = Vec::new();
        reference
            .try_reserve_exact(pairing_class.antifundamental_endpoints.len())
            .map_err(|error| invalid(format!("pairing reference allocation failed: {error}")))?;
        reference.extend(
            pairing_class
                .antifundamental_endpoints
                .iter()
                .map(|endpoint| endpoint.source_slot),
        );
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(pairing_class.fundamental_endpoints.len())
            .map_err(|error| invalid(format!("selected pairing allocation failed: {error}")))?;
        for endpoint in &pairing_class.fundamental_endpoints {
            let selected_slot = endpoint_pairs
                .iter()
                .find(|pair| pair[0] == endpoint.source_slot)
                .map(|pair| pair[1])
                .ok_or_else(|| {
                    integrity("canonical Wick lineage omits a pairing-class fundamental")
                })?;
            selected.push(selected_slot);
        }
        let digits = lehmer_digits(&reference, &selected)?;
        if digits.iter().map(|digit| u64::from(*digit)).sum::<u64>() % 2 == 1 {
            parity = -parity;
        }
        hash_digest(&mut proof_hash, pairing_class.semantic_digest);
        hash_len(&mut proof_hash, selected.len(), "discovered pairing class")?;
        for (reference_slot, selected_slot) in
            reference.iter().copied().zip(selected.iter().copied())
        {
            source_slot_permutation[reference_slot as usize] = selected_slot;
            proof_hash.update(reference_slot.to_le_bytes());
            proof_hash.update(selected_slot.to_le_bytes());
        }
        for digit in digits {
            proof_hash.update(digit.to_le_bytes());
        }
    }
    validate_permutation(
        &source_slot_permutation,
        seed.source_anchors.len(),
        "discovered pairing source permutation",
    )?;
    for (line_id, pair) in endpoint_pairs.iter().copied().enumerate() {
        let line_id = checked_u32(line_id, "discovered pairing line ID")?;
        for source_slot in pair {
            let entry = source_lineage
                .get_mut(source_slot as usize)
                .ok_or_else(|| integrity("discovered pairing endpoint is out of range"))?;
            if *entry != MISSING_U32 {
                return Err(integrity(
                    "discovered pairing endpoint belongs to multiple lines",
                ));
            }
            *entry = line_id;
        }
    }
    proof_hash.update(parity.to_le_bytes());
    Ok(ResolvedPairingOwnerV1 {
        endpoint_pairs,
        proof_digest: Some(final_digest(proof_hash)?),
        source_slot_permutation,
        source_lineage,
        fermion_parity: parity,
    })
}

fn unique_projected_pairing_owner(
    closures: &[PendingClosure],
) -> RusticolResult<&PendingPairingLineage> {
    let mut owner = None;
    for closure in closures {
        let [lineage] = closure.pairing_lineages.as_slice() else {
            return Err(integrity(format!(
                "canonical projected closure has {} Wick lineages, expected exactly one",
                closure.pairing_lineages.len(),
            )));
        };
        if owner.is_some_and(|previous| previous != lineage) {
            return Err(integrity(
                "canonical projected closures disagree across Wick lineages",
            ));
        }
        owner = Some(lineage);
    }
    owner.ok_or_else(|| integrity("canonical projected closure has no Wick lineage"))
}

pub(super) fn supports_are_disjoint(left: &[u32], right: &[u32]) -> bool {
    let mut left_index = 0usize;
    let mut right_index = 0usize;
    while left_index < left.len() && right_index < right.len() {
        match left[left_index].cmp(&right[right_index]) {
            std::cmp::Ordering::Less => left_index += 1,
            std::cmp::Ordering::Greater => right_index += 1,
            std::cmp::Ordering::Equal => return false,
        }
    }
    true
}

pub(super) fn merge_disjoint_support(left: &[u32], right: &[u32]) -> RusticolResult<Vec<u32>> {
    if !supports_are_disjoint(left, right) {
        return Err(invalid(
            "query-local parents have overlapping source support",
        ));
    }
    let mut result = left.iter().chain(right).copied().collect::<Vec<_>>();
    result.sort_unstable();
    Ok(result)
}

fn selected_source_state<'a>(
    seed: &'a OnTheFlyProcessSeedV1,
    selected: OnTheFlySelectedSourceV1,
) -> RusticolResult<(&'a OnTheFlySourceAnchorV1, &'a OnTheFlySourceStateV1)> {
    let anchor = seed
        .source_anchors
        .get(selected.source_slot as usize)
        .ok_or_else(|| invalid("selected source slot is absent from the compact seed"))?;
    let state = anchor.selected(selected.state_index, selected.public_helicity)?;
    Ok((anchor, state))
}

pub(super) fn validate_seed_against_templates<'a>(
    templates: &'a ValidatedRecurrenceTemplateInput,
    seed: &OnTheFlyProcessSeedV1,
) -> RusticolResult<TemplateCatalog<'a>> {
    let summary = templates.summary();
    if summary.catalog_digest != seed.template_catalog_digest
        || summary.compiled_model_digest != seed.model_digest
        || summary.prepared_kernel_pack_digest != seed.prepared_pack_digest
    {
        return Err(integrity(
            "compact seed belongs to a different model/template/prepared catalog",
        ));
    }
    let catalog = TemplateCatalog::new(templates.input())?;
    if catalog.coupling_order_dimension() != seed.explicit_coupling_limits().len() {
        return Err(integrity(
            "compact coupling-limit dimension differs from the template catalog",
        ));
    }
    Ok(catalog)
}

pub(super) fn validate_source_contract(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    anchor: &OnTheFlySourceAnchorV1,
    state: &OnTheFlySourceStateV1,
) -> RusticolResult<(SourceRow, crate::recurrence::template::CurrentStateRow)> {
    let input = templates.input();
    let source = *input
        .sources
        .get(state.source_template_id as usize)
        .ok_or_else(|| integrity("compact source template is absent"))?;
    let current_state = *input
        .current_states
        .get(state.current_state_template_id as usize)
        .ok_or_else(|| integrity("compact current-state template is absent"))?;
    if source.id != state.source_template_id
        || current_state.id != state.current_state_template_id
        || source.flavour_flow_id as usize >= input.flavour_flow_ranges.len()
        || source.quantum_number_flow_id != state.quantum_number_flow_id
        || current_state.chirality != state.chirality
    {
        return Err(integrity("compact source-state contract is stale"));
    }
    if catalog.flavour_flow(source.flavour_flow_id, "source flavour flow")?
        != state.flavour_flow.as_ref()
        || catalog.digest(source.semantic_digest_id, "source semantic")?
            != state.source_semantic_digest
        || catalog.digest(current_state.semantic_digest_id, "current-state semantic")?
            != state.current_state_semantic_digest
        || catalog.source_seed(source)?.proof_digest() != state.color_seed_proof_digest
    {
        return Err(integrity(
            "compact source-state semantic evidence differs from the template catalog",
        ));
    }
    validate_crossed_source_state(
        anchor.is_initial,
        &ProcessSourceStateRow {
            source_slot: anchor.source_slot,
            state_index: state.state_index,
            public_helicity: state.public_helicity,
            chirality: state.chirality,
            spin_state: state.spin_state,
            current_state_template_id: state.current_state_template_id,
            source_template_id: state.source_template_id,
            momentum_sign: state.momentum_sign,
            // The helper does not consume the process-owned factor ID; the
            // compact seed authenticates the exact crossing phase directly.
            crossing_phase_factor_id: 0,
        },
        source,
        input,
    )?;
    let binding = input
        .evaluator_bindings
        .get(source.evaluator_binding_id as usize)
        .ok_or_else(|| integrity("source evaluator binding is absent"))?;
    if binding.id != source.evaluator_binding_id
        || EvaluatorContractKind::try_from(binding.contract_kind)? != EvaluatorContractKind::Source
    {
        return Err(integrity("source evaluator binding has the wrong role"));
    }
    Ok((source, current_state))
}

fn query_target_matches(mut closed: Vec<LCColorComponent>, query: &DecodedLcQueryV1) -> bool {
    closed.sort_unstable();
    closed.as_slice() == query.target_components.as_ref()
}

pub(super) fn insert_selected_sources(
    grammar: &PreparedOnTheFlyGrammarV1,
    seed: &OnTheFlyProcessSeedV1,
    coupling_limits: &[Option<u32>],
    query: &DecodedLcQueryV1,
    colors: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
) -> RusticolResult<()> {
    if coupling_limits.len() != seed.explicit_coupling_limits().len() {
        return Err(integrity(
            "resolved coupling-limit dimension differs from the compact seed",
        ));
    }
    let zero_orders = vec![0_u32; coupling_limits.len()];
    for selected in query.selected_sources.iter().copied() {
        let (_, state) = selected_source_state(seed, selected)?;
        let contract = grammar
            .sources
            .get(&(state.source_template_id, state.current_state_template_id))
            .ok_or_else(|| integrity("selected source has no prepared source contract"))?;
        let color = contract.color_seed.instantiate(
            selected.source_slot,
            contract.current_state.color_representation,
        )?;
        let color_id = colors.intern(color)?;
        let key = CurrentCoreKey::new(
            seed.template_catalog_digest,
            RecurrenceNodeKind::Source,
            state.current_state_template_id,
            color_id,
            vec![selected.source_slot],
            CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                source_slot: selected.source_slot,
                coefficient: state.momentum_sign,
            }])?,
            CurrentHelicityIdentity::topology_replay(
                state.spin_state,
                vec![SourceStateAssignment::new(
                    selected.source_slot,
                    selected.state_index,
                )],
            )?,
            state.flavour_flow.to_vec(),
            state.quantum_number_flow_id,
            zero_orders.clone(),
            CurrentSourceBinding::FixedTemplate(state.source_template_id),
            None,
        )?;
        let id = checked_u32(currents.len(), "query-local current count")?;
        if current_ids.insert(key.clone(), id).is_some() {
            return Err(integrity(
                "selected source construction produced a duplicate current",
            ));
        }
        currents.push(PendingCurrent {
            key,
            source_factor: Some(state.crossing_phase),
            contributions: BTreeMap::new(),
            pairing_lineages: vec![PendingPairingLineage::source(seed, selected.source_slot)],
            stage: 0,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn include_transition(
    templates: &ValidatedRecurrenceTemplateInput,
    prepared: &PreparedTransition,
    concrete_parent_ids: [u32; 2],
    source_count: usize,
    seed: &OnTheFlyProcessSeedV1,
    coupling_limits: &[Option<u32>],
    propagators: &BTreeMap<u32, Option<u32>>,
    colors: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
) -> RusticolResult<()> {
    let parents = [
        &currents[concrete_parent_ids[0] as usize].key,
        &currents[concrete_parent_ids[1] as usize].key,
    ];
    if !(0..2).all(|index| {
        prepared.input_states[index] == parents[index].current_state_template_id()
            && quantum_parent_spin_matches(prepared.input_spins[index], parents[index])
    }) {
        return Ok(());
    }
    if let Some(contact_orbit) = prepared.contact_orbit.as_ref()
        && !contact_orbit.accepts_parent_domain(parents)?
    {
        return Ok(());
    }
    let Some(coupling_orders) = combined_coupling_orders(
        parents[0].coupling_orders(),
        parents[1].coupling_orders(),
        &prepared.local_orders,
        coupling_limits,
    )?
    else {
        return Ok(());
    };
    let support = merge_disjoint_support(
        parents[0].support_source_slots(),
        parents[1].support_source_slots(),
    )?;
    if support.len() >= source_count {
        return Ok(());
    }
    let helicity = merged_helicity_identity(
        parents[0].helicity_identity(),
        parents[1].helicity_identity(),
        prepared.quantum.result_spin_state,
    )?;
    let flavour = prepared.flavour.apply(parents);
    let momentum = merged_momentum(parents[0].momentum(), parents[1].momentum())?;
    let parent_colors = [
        colors
            .get(parents[0].dynamic_lc_color_state_id())
            .ok_or_else(|| integrity("left query-local color state disappeared"))?
            .clone(),
        colors
            .get(parents[1].dynamic_lc_color_state_id())
            .ok_or_else(|| integrity("right query-local color state disappeared"))?
            .clone(),
    ];
    let (evaluator_parent_ids, exchange_factor) = prepared.evaluator_parents(concrete_parent_ids);
    let result_state = templates
        .input()
        .current_states
        .get(prepared.row.result_state_template_id as usize)
        .ok_or_else(|| integrity("pairing-lineage result state is absent"))?;
    let carries_colored_fermion_line =
        result_state.statistics == 1 && result_state.color_representation != 1;
    let pairing_lineages = combine_pairing_lineage_sets(
        seed,
        &currents[concrete_parent_ids[0] as usize].pairing_lineages,
        &currents[concrete_parent_ids[1] as usize].pairing_lineages,
        carries_colored_fermion_line,
    )?;
    if pairing_lineages.is_empty() {
        return Ok(());
    }
    for prepared_witness in &prepared.witnesses {
        if prepared_witness.row.left_shape_string_id != parent_colors[0].output_color_shape_id()
            || prepared_witness.row.right_shape_string_id
                != parent_colors[1].output_color_shape_id()
        {
            continue;
        }
        let Some(result_color) = prepared_witness
            .witness
            .apply(&parent_colors[0], &parent_colors[1])?
        else {
            continue;
        };
        let color_id = colors.intern(result_color)?;
        let propagator_template_id = if support.len() + 1 < source_count {
            propagators
                .get(&prepared.row.result_state_template_id)
                .copied()
                .flatten()
        } else {
            None
        };
        let key = CurrentCoreKey::new(
            seed.template_catalog_digest,
            RecurrenceNodeKind::Current,
            prepared.row.result_state_template_id,
            color_id,
            support.clone(),
            momentum.clone(),
            helicity.clone(),
            flavour.clone(),
            prepared.quantum.result_quantum_number_flow_id,
            coupling_orders.clone(),
            CurrentSourceBinding::None,
            propagator_template_id,
        )?;
        let result_id = if let Some(id) = current_ids.get(&key).copied() {
            id
        } else {
            let id = checked_u32(currents.len(), "query-local current count")?;
            let map_key = key.clone();
            let pending = PendingCurrent {
                key,
                source_factor: None,
                contributions: BTreeMap::new(),
                pairing_lineages: try_clone_pairing_lineages(&pairing_lineages)?,
                stage: checked_u32(support.len() - 1, "query-local current stage")?,
            };
            current_ids.insert(map_key, id);
            currents.push(pending);
            id
        };
        extend_pairing_lineages(
            &mut currents[result_id as usize].pairing_lineages,
            &pairing_lineages,
        )?;
        let contribution_key = ContributionKey::new(
            prepared.row.id,
            evaluator_parent_ids.to_vec(),
            evaluator_parent_ids
                .iter()
                .map(|id| currents[*id as usize].key.current_state_template_id())
                .collect(),
            evaluator_parent_ids
                .iter()
                .map(|id| currents[*id as usize].key.momentum().clone())
                .collect(),
            prepared.row.result_state_template_id,
            prepared.quantum.id,
            LCColorWitnessTermId::new(
                prepared.row.color_contraction_template_id,
                prepared_witness.row.ordinal,
            ),
            prepared.quantum_semantic_digest,
            prepared.row.output_projection_string_id,
        )?;
        let pending_key = PendingContributionKey {
            parent_current_ids: evaluator_parent_ids,
            key: contribution_key,
        };
        let output_factor = prepared.output_factor()?;
        let witness_factor = prepared_witness.witness.exact_factor();
        let factor = multiply_factors(&[
            prepared.base_factor,
            output_factor,
            exchange_factor,
            witness_factor,
        ])?;
        let aggregate_factor_after = {
            let aggregate = currents[result_id as usize]
                .contributions
                .entry(pending_key)
                .or_insert(ExactComplexRational::ZERO);
            aggregate_factor(aggregate, factor)?;
            *aggregate
        };
        #[cfg(feature = "on-the-fly-test-support")]
        'transition_diagnostic: {
            if !crate::recurrence::diagnostic::transition_diagnostic_observation_active() {
                break 'transition_diagnostic;
            }
            use crate::recurrence::diagnostic::{
                ConstructionTransitionDiagnosticRowV1, observe_transition_diagnostic,
            };

            let digest = |current_id: u32| -> RusticolResult<SemanticDigest> {
                let current = currents
                    .get(current_id as usize)
                    .ok_or_else(|| integrity("diagnostic current is absent"))?;
                let color = colors
                    .get(current.key.dynamic_lc_color_state_id())
                    .ok_or_else(|| integrity("diagnostic current color is absent"))?;
                super::trace::hash_current_key(&current.key, color)
            };
            let result_color = colors
                .get(currents[result_id as usize].key.dynamic_lc_color_state_id())
                .ok_or_else(|| integrity("diagnostic result color is absent"))?;
            observe_transition_diagnostic(ConstructionTransitionDiagnosticRowV1 {
                materialized_sector_id: None,
                output_current_digest: digest(result_id)?,
                ordered_parent_digests: [
                    digest(evaluator_parent_ids[0])?,
                    digest(evaluator_parent_ids[1])?,
                ],
                transition_template_id: prepared.row.id,
                transition_semantic_digest: prepared.transition_semantic_digest,
                evaluator_binding_semantic_digest: prepared.evaluator_binding_digest,
                result_state_template_id: prepared.row.result_state_template_id,
                quantum_flow_witness_id: prepared.quantum.id,
                quantum_semantic_digest: prepared.quantum_semantic_digest,
                color_contraction_template_id: prepared.row.color_contraction_template_id,
                color_witness_ordinal: prepared_witness.row.ordinal,
                color_witness_proof_digest: prepared_witness.witness.proof_digest(),
                output_projection_id: prepared.row.output_projection_string_id,
                transition_factor: prepared.transition_factor,
                contraction_factor: prepared.contraction_factor,
                output_factor,
                exchange_factor,
                witness_factor,
                reversal_mask: 0,
                reversal_factor: ExactComplexRational::ONE,
                candidate_factor: factor,
                aggregate_factor_after,
                parent_reflection_proof_digests: [None, None],
                parent_reflection_phases: [None, None],
                local_reflection_proof_digest: None,
                local_reflection_phase: None,
                result_reflection_proof_digest: None,
                result_reflection_phase: None,
                output_color_orientation: format!("{result_color:?}"),
            });
        }
    }
    Ok(())
}

pub(super) fn build_forward_currents(
    templates: &ValidatedRecurrenceTemplateInput,
    transitions: &BTreeMap<(u32, u32), Vec<PreparedTransition>>,
    seed: &OnTheFlyProcessSeedV1,
    coupling_limits: &[Option<u32>],
    propagators: &BTreeMap<u32, Option<u32>>,
    colors: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
) -> RusticolResult<()> {
    let source_count = seed.source_anchors.len();
    let contact_orbits = PreparedContactOrbitIndex::new(transitions)?;
    for target_size in 2..source_count {
        let stage_current_start = currents.len();
        let eligible = currents
            .iter()
            .enumerate()
            .filter(|(_, current)| current.key.support_source_slots().len() < target_size)
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        for (left_offset, left_index) in eligible.iter().copied().enumerate() {
            for right_index in eligible.iter().copied().skip(left_offset + 1) {
                let left = &currents[left_index].key;
                let right = &currents[right_index].key;
                if left.support_source_slots().len() + right.support_source_slots().len()
                    != target_size
                    || !supports_are_disjoint(
                        left.support_source_slots(),
                        right.support_source_slots(),
                    )
                {
                    continue;
                }
                let left_state = left.current_state_template_id();
                let right_state = right.current_state_template_id();
                let Some(rows) = transitions.get(&canonical_state_pair(left_state, right_state))
                else {
                    continue;
                };
                let left_id = checked_u32(left_index, "query-local parent ID")?;
                let right_id = checked_u32(right_index, "query-local parent ID")?;
                for prepared in rows {
                    let Some(parent_ids) =
                        prepared.parent_ids(left_state, right_state, left_id, right_id)
                    else {
                        continue;
                    };
                    include_transition(
                        templates,
                        prepared,
                        parent_ids,
                        source_count,
                        seed,
                        coupling_limits,
                        propagators,
                        colors,
                        currents,
                        current_ids,
                    )?;
                }
            }
        }
        if !contact_orbits.is_empty()
            && let Some(plan) = plan_on_the_fly_contact_orbit_owners(
                stage_current_start,
                &contact_orbits,
                currents,
            )?
        {
            plan.commit(currents);
        }
    }
    Ok(())
}

fn closure_quantum_matches(
    quantum: &PreparedClosureQuantum,
    parents: [&CurrentCoreKey; 2],
) -> bool {
    match (quantum.input_states, quantum.input_spins) {
        (None, None) => true,
        (Some(states), Some(spins)) => (0..2).all(|index| {
            states[index] == parents[index].current_state_template_id()
                && quantum_parent_spin_matches(spins[index], parents[index])
        }),
        _ => false,
    }
}

pub(super) fn build_selected_closures(
    closures: &BTreeMap<(u32, u32), Vec<PreparedClosure>>,
    seed: &OnTheFlyProcessSeedV1,
    query: &DecodedLcQueryV1,
    colors: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
) -> RusticolResult<Option<Vec<PendingClosure>>> {
    let anchor_support = [query.closure_anchor_slot];
    let complement_support = (0..query.selected_sources.len() as u32)
        .filter(|slot| *slot != query.closure_anchor_slot)
        .collect::<Vec<_>>();
    let anchor_ids = currents
        .iter()
        .enumerate()
        .filter(|(_, current)| {
            current.key.node_kind() == RecurrenceNodeKind::Source
                && current.key.support_source_slots() == anchor_support
        })
        .map(|(id, _)| checked_u32(id, "closure anchor current ID"))
        .collect::<RusticolResult<Vec<_>>>()?;
    if anchor_ids.len() != 1 {
        return Err(invalid(
            "decoded closure anchor does not identify exactly one source",
        ));
    }
    let complement_ids = currents
        .iter()
        .enumerate()
        .filter(|(_, current)| current.key.support_source_slots() == complement_support)
        .map(|(id, _)| checked_u32(id, "closure complement current ID"))
        .collect::<RusticolResult<Vec<_>>>()?;

    let mut retained = BTreeMap::<PendingClosureKey, PendingClosure>::new();
    let anchor_id = anchor_ids[0];
    let anchor_state = currents[anchor_id as usize].key.current_state_template_id();
    for complement_id in complement_ids {
        let complement_state = currents[complement_id as usize]
            .key
            .current_state_template_id();
        let Some(rows) = closures.get(&canonical_state_pair(anchor_state, complement_state)) else {
            continue;
        };
        for closure in rows {
            let Some(concrete_parent_ids) =
                closure.parent_ids(anchor_state, complement_state, anchor_id, complement_id)
            else {
                continue;
            };
            let parents = [
                &currents[concrete_parent_ids[0] as usize].key,
                &currents[concrete_parent_ids[1] as usize].key,
            ];
            let pairing_lineages = combine_pairing_lineage_sets(
                // A physical closure is bosonic and must close the final
                // unmatched fundamental/antifundamental pair, if any.
                // Completed pairs remain query-local proof state only.
                seed,
                &currents[concrete_parent_ids[0] as usize].pairing_lineages,
                &currents[concrete_parent_ids[1] as usize].pairing_lineages,
                false,
            )?;
            let mut complete_lineages = Vec::new();
            complete_lineages
                .try_reserve_exact(pairing_lineages.len())
                .map_err(|error| {
                    invalid(format!(
                        "complete pairing-lineage allocation failed: {error}"
                    ))
                })?;
            for lineage in pairing_lineages {
                if complete_pairing_lineage(seed, &lineage)? {
                    complete_lineages.push(lineage);
                }
            }
            let pairing_lineages = complete_lineages;
            if pairing_lineages.is_empty() {
                continue;
            }
            let parent_colors = [
                colors
                    .get(parents[0].dynamic_lc_color_state_id())
                    .ok_or_else(|| integrity("closure left color state disappeared"))?,
                colors
                    .get(parents[1].dynamic_lc_color_state_id())
                    .ok_or_else(|| integrity("closure right color state disappeared"))?,
            ];
            let (evaluator_parent_ids, exchange_factor) =
                closure.evaluator_parents(concrete_parent_ids);
            for quantum in closure
                .quantum_flows
                .iter()
                .filter(|quantum| closure_quantum_matches(quantum, parents))
            {
                for witness in &closure.witnesses {
                    if witness.row.left_shape_string_id != parent_colors[0].output_color_shape_id()
                        || witness.row.right_shape_string_id
                            != parent_colors[1].output_color_shape_id()
                    {
                        continue;
                    }
                    let closed = witness
                        .witness
                        .closed_components(parent_colors[0], parent_colors[1])?;
                    if !query_target_matches(closed, query) {
                        continue;
                    }
                    let key = PendingClosureKey {
                        closure_template_id: closure.row.id,
                        quantum_flow_template_id: quantum.row.map(|row| row.id),
                        parent_current_ids: evaluator_parent_ids,
                        color_witness_term_id: LCColorWitnessTermId::new(
                            closure.row.color_contraction_template_id,
                            witness.row.ordinal,
                        ),
                    };
                    let factor = multiply_factors(&[
                        closure.base_factor,
                        quantum.output_factor()?,
                        exchange_factor,
                        witness.witness.exact_factor(),
                    ])?;
                    let coefficients = closure.component_coefficients.clone();
                    match retained.entry(key.clone()) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert(PendingClosure {
                                key,
                                factor,
                                component_coefficients: coefficients,
                                pairing_lineages: try_clone_pairing_lineages(&pairing_lineages)?,
                            });
                        }
                        std::collections::btree_map::Entry::Occupied(mut entry) => {
                            if entry.get().component_coefficients != coefficients {
                                return Err(integrity(
                                    "equal closure identities have different component coefficients",
                                ));
                            }
                            aggregate_factor(&mut entry.get_mut().factor, factor)?;
                            extend_pairing_lineages(
                                &mut entry.get_mut().pairing_lineages,
                                &pairing_lineages,
                            )?;
                        }
                    }
                }
            }
        }
    }
    retained.retain(|_, closure| !closure.factor.is_zero());
    if retained.is_empty() {
        // The process-global immutable grammar proved this closure domain
        // complete before it entered the runtime cache.
        return Ok(None);
    }
    Ok(Some(retained.into_values().collect()))
}

pub(super) fn validate_prepared_closure_domain(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    closures: &BTreeMap<(u32, u32), Vec<PreparedClosure>>,
) -> RusticolResult<()> {
    let expected = &templates.input().closures;
    let observed_count = closures.values().try_fold(0usize, |count, rows| {
        count
            .checked_add(rows.len())
            .ok_or_else(|| invalid("prepared closure-domain row count exceeds usize"))
    })?;
    if observed_count != expected.len() {
        return Err(integrity(format!(
            "prepared closure domain has {observed_count} rows, expected {}",
            expected.len()
        )));
    }

    let mut observed = vec![false; expected.len()];
    for (state_pair, rows) in closures {
        for closure in rows {
            let expected_row = expected
                .get(closure.row.id as usize)
                .filter(|row| row.id == closure.row.id)
                .ok_or_else(|| integrity("prepared closure domain contains an unknown row"))?;
            if closure.row != *expected_row {
                return Err(integrity(
                    "prepared closure row differs from its authenticated template",
                ));
            }
            let expected_states: [u32; 2] = catalog
                .u32_sequence(expected_row.input_state_sequence_id, "closure input states")?
                .try_into()
                .map_err(|_| integrity("authenticated closure input-state pair is not binary"))?;
            if closure.input_states != expected_states
                || *state_pair != canonical_state_pair(expected_states[0], expected_states[1])
            {
                return Err(integrity(
                    "prepared closure parent-state pair differs from its authenticated template",
                ));
            }
            let seen = observed
                .get_mut(closure.row.id as usize)
                .ok_or_else(|| integrity("prepared closure row ID exceeds its domain"))?;
            if std::mem::replace(seen, true) {
                return Err(integrity(
                    "prepared closure domain repeats an authenticated row",
                ));
            }
        }
    }
    if observed.iter().any(|seen| !seen) {
        return Err(integrity(
            "prepared closure domain omits an authenticated row",
        ));
    }
    Ok(())
}

pub(super) fn live_current_ids(
    currents: &[PendingCurrent],
    closures: &[PendingClosure],
) -> RusticolResult<BTreeSet<u32>> {
    let mut live = BTreeSet::new();
    let mut queue = VecDeque::new();
    for closure in closures {
        for parent in closure.key.parent_current_ids {
            if live.insert(parent) {
                queue.push_back(parent);
            }
        }
    }
    while let Some(current_id) = queue.pop_front() {
        let current = currents
            .get(current_id as usize)
            .ok_or_else(|| integrity("liveness queue references an absent current"))?;
        for (contribution, factor) in &current.contributions {
            if factor.is_zero() {
                continue;
            }
            for parent in contribution.parent_current_ids {
                if live.insert(parent) {
                    queue.push_back(parent);
                }
            }
        }
    }
    Ok(live)
}

#[cfg(test)]
mod tests {
    use crate::recurrence::DynamicLCColorStateId;
    use crate::recurrence::contact_orbit_owner::{
        ContactOrbitStepProof, ContactOrbitTestBinding, contact_orbit_application_for_test,
        contact_orbit_test_template, final_contact_orbit_step_for_test,
        partial_contact_orbit_step_for_test, prepared_contact_orbit_transition_for_test,
    };

    use super::*;

    fn contact_digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).expect("test digest must be nonzero")
    }

    fn contact_current(
        node_kind: RecurrenceNodeKind,
        state: u32,
        color: u32,
        support: &[u32],
    ) -> PendingCurrent {
        contact_current_with_source_template(
            node_kind,
            state,
            color,
            support,
            support.first().copied().unwrap_or_default(),
        )
    }

    fn contact_current_with_source_template(
        node_kind: RecurrenceNodeKind,
        state: u32,
        color: u32,
        support: &[u32],
        source_template_id: u32,
    ) -> PendingCurrent {
        let key = CurrentCoreKey::new(
            contact_digest(230),
            node_kind,
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
            CurrentHelicityIdentity::topology_replay(
                0,
                support
                    .iter()
                    .copied()
                    .map(|source_slot| SourceStateAssignment::new(source_slot, 0))
                    .collect(),
            )
            .unwrap(),
            vec![state as i32],
            0,
            vec![0],
            if node_kind == RecurrenceNodeKind::Source {
                CurrentSourceBinding::FixedTemplate(source_template_id)
            } else {
                CurrentSourceBinding::None
            },
            None,
        )
        .unwrap();
        PendingCurrent {
            key,
            source_factor: (node_kind == RecurrenceNodeKind::Source)
                .then_some(ExactComplexRational::ONE),
            contributions: BTreeMap::new(),
            pairing_lineages: vec![PendingPairingLineage {
                completed_pairs: Vec::new(),
                unmatched_endpoint: None,
            }],
            stage: u32::try_from(support.len().saturating_sub(1)).unwrap(),
        }
    }

    fn contact_transition(
        step: ContactOrbitStepProof,
        input_state_template_ids: [u32; 2],
        transition_digest: u8,
    ) -> PreparedContactOrbitTransition {
        contact_transition_with_witness(
            step,
            input_state_template_ids,
            transition_digest,
            LCColorWitnessTermId::new(4, 0),
        )
    }

    fn contact_transition_with_witness(
        step: ContactOrbitStepProof,
        input_state_template_ids: [u32; 2],
        transition_digest: u8,
        color_witness_term_id: LCColorWitnessTermId,
    ) -> PreparedContactOrbitTransition {
        let mut application = contact_orbit_application_for_test();
        application.color_witness_term_id = color_witness_term_id;
        prepared_contact_orbit_transition_for_test(
            step,
            input_state_template_ids,
            contact_digest(transition_digest),
            application,
        )
    }

    fn add_contact_contribution(
        currents: &mut [PendingCurrent],
        destination: u32,
        transition: u32,
        parents: [u32; 2],
    ) {
        let parent_keys = parents.map(|id| &currents[id as usize].key);
        let destination_key = &currents[destination as usize].key;
        let key = ContributionKey::new(
            transition,
            parents.to_vec(),
            parent_keys
                .iter()
                .map(|key| key.current_state_template_id())
                .collect(),
            parent_keys
                .iter()
                .map(|key| key.momentum().clone())
                .collect(),
            destination_key.current_state_template_id(),
            0,
            LCColorWitnessTermId::new(4, 0),
            contact_digest(2),
            8,
        )
        .unwrap();
        assert!(
            currents[destination as usize]
                .contributions
                .insert(
                    PendingContributionKey {
                        parent_current_ids: parents,
                        key,
                    },
                    ExactComplexRational::ONE,
                )
                .is_none()
        );
    }

    fn contact_transition_ids(current: &PendingCurrent) -> Vec<u32> {
        current
            .contributions
            .keys()
            .map(|pending| pending.key.transition_template_id())
            .collect()
    }

    fn pending_contact_counts(currents: &[PendingCurrent]) -> (usize, usize) {
        (
            currents.len(),
            currents
                .iter()
                .map(|current| current.contributions.len())
                .sum(),
        )
    }

    fn scalar_contact_seed(source_count: u32) -> OnTheFlyProcessSeedV1 {
        let state = OnTheFlySourceStateV1::new(
            0,
            0,
            0,
            0,
            0,
            contact_digest(231),
            contact_digest(232),
            1,
            ExactComplexRational::ONE,
            0,
            0,
            vec![1],
            0,
            contact_digest(233),
            OnTheFlySourceWavefunctionFamilyV1::Scalar,
            OnTheFlySourceOrientationV1::SelfConjugate,
            None,
        )
        .unwrap();
        let anchors = (0_u32..source_count)
            .map(|source_slot| {
                OnTheFlySourceAnchorV1::new(
                    source_slot,
                    source_slot,
                    false,
                    OnTheFlyExternalColorRoleV1::Singlet,
                    false,
                    None,
                    vec![state.clone()],
                )
                .unwrap()
            })
            .collect();
        OnTheFlyProcessSeedV1::new(
            contact_digest(234),
            contact_digest(235),
            contact_digest(230),
            contact_digest(236),
            contact_digest(237),
            contact_digest(238),
            "raw-amplitude-contact-test",
            ExactComplexRational::ONE,
            anchors,
            (0_u32..source_count).collect(),
            OnTheFlyCouplingOrderPolicyV1::Explicit,
            vec![1],
            vec![Some(0)],
            Vec::new(),
        )
        .unwrap()
    }

    fn production_contact_barrier_case(
        reverse_transition_order: bool,
        include_ordinary_scalar_transition: bool,
    ) -> (Vec<(Vec<u32>, u32, Vec<u32>)>, (usize, usize), bool) {
        let mut template_input = contact_orbit_test_template(ContactOrbitTestBinding::One);
        let mut intermediate_state = template_input.current_states[0];
        intermediate_state.id = 1;
        let mut append_canonical_string = |value: &[u8]| {
            let previous_string = template_input.string_ranges.last().unwrap();
            let previous_range = previous_string
                .as_usize_range(template_input.string_bytes.len(), "test string")
                .unwrap();
            assert!(
                &template_input.string_bytes[previous_range] < value,
                "the intermediate-state fixture must extend the canonical string catalog"
            );
            let id = u32::try_from(template_input.string_ranges.len()).unwrap();
            template_input
                .string_ranges
                .push(crate::recurrence::CheckedTableRange::new(
                    u64::try_from(template_input.string_bytes.len()).unwrap(),
                    u64::try_from(value.len()).unwrap(),
                ));
            template_input.string_bytes.extend_from_slice(value);
            id
        };
        let intermediate_propagator_template_id =
            append_canonical_string(b"zz-on-the-fly-intermediate-propagator");
        let intermediate_state_template_id =
            append_canonical_string(b"zz-on-the-fly-intermediate-state");
        intermediate_state.template_string_id = intermediate_state_template_id;
        let intermediate_digest_id = u32::try_from(template_input.digest_catalog.len()).unwrap();
        template_input
            .digest_catalog
            .push(crate::recurrence::template::DigestCatalogRow {
                id: intermediate_digest_id,
                value: [240; 32],
            });
        intermediate_state.semantic_digest_id = intermediate_digest_id;
        template_input.current_states.push(intermediate_state);
        let intermediate_propagator_digest_id =
            u32::try_from(template_input.digest_catalog.len()).unwrap();
        template_input
            .digest_catalog
            .push(crate::recurrence::template::DigestCatalogRow {
                id: intermediate_propagator_digest_id,
                value: [241; 32],
            });
        let mut intermediate_propagator = template_input.propagators[0];
        intermediate_propagator.id = 1;
        intermediate_propagator.template_string_id = intermediate_propagator_template_id;
        intermediate_propagator.state_template_id = 1;
        intermediate_propagator.applies_propagator = 0;
        intermediate_propagator.evaluator_binding_id = MISSING_U32;
        intermediate_propagator.numerator_expression_digest_id = MISSING_U32;
        intermediate_propagator.denominator_expression_digest_id = MISSING_U32;
        intermediate_propagator.semantic_digest_id = intermediate_propagator_digest_id;
        template_input.propagators.push(intermediate_propagator);
        template_input.catalog_header[0].current_state_count = 2;
        template_input.catalog_header[0].propagator_count = 2;
        let templates = template_input.validate().unwrap();
        let catalog = TemplateCatalog::new(templates.input()).unwrap();
        let prepared = prepared_transitions(&templates, &catalog).unwrap();
        let base = prepared.values().next().unwrap()[0].clone();
        let partial_steps = [
            partial_contact_orbit_step_for_test(0, 1, 2, 20, [0, 0, 0, 0]),
            partial_contact_orbit_step_for_test(1, 0, 2, 30, [0, 0, 0, 0]),
            partial_contact_orbit_step_for_test(0, 2, 1, 21, [0, 0, 0, 0]),
            partial_contact_orbit_step_for_test(2, 0, 1, 31, [0, 0, 0, 0]),
            partial_contact_orbit_step_for_test(0, 3, 1, 22, [0, 0, 0, 0]),
            partial_contact_orbit_step_for_test(3, 0, 1, 32, [0, 0, 0, 0]),
        ];
        let final_steps = [
            final_contact_orbit_step_for_test(&[0], &[1, 2], 3, 60, [0, 0, 0, 0]),
            final_contact_orbit_step_for_test(&[1, 2], &[0], 3, 70, [0, 0, 0, 0]),
            final_contact_orbit_step_for_test(&[0], &[1, 3], 2, 61, [0, 0, 0, 0]),
            final_contact_orbit_step_for_test(&[1, 3], &[0], 2, 71, [0, 0, 0, 0]),
            final_contact_orbit_step_for_test(&[0], &[2, 3], 1, 62, [0, 0, 0, 0]),
            final_contact_orbit_step_for_test(&[2, 3], &[0], 1, 72, [0, 0, 0, 0]),
        ];
        let transition_order = if reverse_transition_order {
            (0_usize..partial_steps.len()).rev().collect::<Vec<_>>()
        } else {
            (0_usize..partial_steps.len()).collect::<Vec<_>>()
        };
        let build_row = |index: usize,
                         id_offset: u32,
                         step: ContactOrbitStepProof,
                         input_states: [u32; 2],
                         result_state: u32| {
            let id = id_offset + u32::try_from(index).unwrap();
            let digest_byte = 40 + u8::try_from(id).unwrap();
            let mut row = base.clone();
            row.row.id = id;
            row.input_states = input_states;
            row.row.result_state_template_id = result_state;
            row.quantum.result_state_template_id = result_state;
            row.local_orders = vec![0].into_boxed_slice();
            row.transition_semantic_digest = contact_digest(digest_byte);
            row.contact_orbit = Some(contact_transition_with_witness(
                step,
                input_states,
                digest_byte,
                LCColorWitnessTermId::new(
                    row.row.color_contraction_template_id,
                    row.witnesses[0].row.ordinal,
                ),
            ));
            row
        };
        let mut partial_rows = transition_order
            .iter()
            .copied()
            .map(|index| build_row(index, 0, partial_steps[index].clone(), [0, 0], 1))
            .collect::<Vec<_>>();
        let final_rows = transition_order
            .into_iter()
            .map(|index| {
                let input_states = if index % 2 == 0 { [0, 1] } else { [1, 0] };
                build_row(index, 6, final_steps[index].clone(), input_states, 1)
            })
            .collect::<Vec<_>>();
        if include_ordinary_scalar_transition {
            let mut ordinary = base.clone();
            ordinary.row.id = 12;
            ordinary.input_states = [0, 0];
            ordinary.row.result_state_template_id = 0;
            ordinary.quantum.result_state_template_id = 0;
            ordinary.local_orders = vec![0].into_boxed_slice();
            ordinary.transition_semantic_digest = contact_digest(120);
            ordinary.contact_orbit = None;
            partial_rows.push(ordinary);
        }
        let transitions = BTreeMap::from([((0, 0), partial_rows), ((0, 1), final_rows)]);
        let propagators = propagator_by_state(&templates).unwrap();
        let seed = scalar_contact_seed(4);
        let mut colors = DynamicLCColorStateInterner::default();
        let color_id = colors
            .intern(
                DynamicLCColorState::new(
                    base.witnesses[0].row.left_shape_string_id,
                    None,
                    Vec::new(),
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(color_id, DynamicLCColorStateId::from_interner(0));
        let mut currents = (0_u32..4)
            .map(|source_slot| {
                contact_current_with_source_template(
                    RecurrenceNodeKind::Source,
                    0,
                    0,
                    &[source_slot],
                    0,
                )
            })
            .collect::<Vec<_>>();
        let mut current_ids = currents
            .iter()
            .enumerate()
            .map(|(id, current)| (current.key.clone(), u32::try_from(id).unwrap()))
            .collect::<BTreeMap<_, _>>();

        build_forward_currents(
            &templates,
            &transitions,
            &seed,
            seed.explicit_coupling_limits(),
            &propagators,
            &mut colors,
            &mut currents,
            &mut current_ids,
        )
        .unwrap();

        assert!(
            currents[4..].iter().all(
                |current| current.contributions.len() == if current.stage == 1 { 1 } else { 3 }
            ),
            "certified pair destinations retain one owner while final triples retain one owner per source assignment",
        );
        assert!(
            currents[4..].iter().all(|current| {
                current.contributions.keys().all(|contribution| {
                    contribution.parent_current_ids.iter().all(|parent_id| {
                        let parent = &currents[*parent_id as usize];
                        parent.source_factor.is_some() || !parent.contributions.is_empty()
                    })
                })
            }),
            "a selected contact transition must not reference an empty parent",
        );

        let staged = currents[4..]
            .iter()
            .map(|current| {
                (
                    current.key.support_source_slots().to_vec(),
                    current.stage,
                    contact_transition_ids(current),
                )
            })
            .collect();
        let retained_partial_is_consumed = currents[10..].iter().all(|current| {
            current.contributions.keys().all(|contribution| {
                let mut partial_parents = contribution
                    .parent_current_ids
                    .iter()
                    .copied()
                    .filter(|parent| (4..10).contains(&usize::try_from(*parent).unwrap()));
                let Some(partial_parent) = partial_parents.next() else {
                    return false;
                };
                partial_parents.next().is_none()
                    && currents[partial_parent as usize]
                        .key
                        .current_state_template_id()
                        == 1
                    && contribution
                        .parent_current_ids
                        .iter()
                        .copied()
                        .any(|parent| {
                            parent != partial_parent
                                && currents[parent as usize].key.current_state_template_id() == 0
                        })
            })
        });
        (
            staged,
            pending_contact_counts(&currents),
            retained_partial_is_consumed,
        )
    }

    fn commit_contact_plan(
        currents: &mut [PendingCurrent],
        contacts: &BTreeMap<u32, PreparedContactOrbitTransition>,
        stage_current_start: usize,
    ) -> RusticolResult<()> {
        if let Some(plan) = plan_on_the_fly_contact_orbit_owners_with_resolver(
            stage_current_start,
            currents,
            |transition_id| contacts.get(&transition_id),
        )? {
            plan.commit(currents);
        }
        Ok(())
    }

    fn contact_0000_case(final_stage: bool, reverse_insertion: bool) -> (Vec<u32>, (usize, usize)) {
        let (mut currents, steps, parent_ids, destination_id) = if final_stage {
            (
                vec![
                    contact_current(RecurrenceNodeKind::Source, 10, 0, &[10]),
                    contact_current(RecurrenceNodeKind::Source, 10, 1, &[11]),
                    contact_current(RecurrenceNodeKind::Source, 10, 2, &[12]),
                    contact_current(RecurrenceNodeKind::Current, 11, 3, &[11, 12]),
                    contact_current(RecurrenceNodeKind::Current, 11, 4, &[10, 12]),
                    contact_current(RecurrenceNodeKind::Current, 11, 5, &[10, 11]),
                    contact_current(RecurrenceNodeKind::Current, 20, 6, &[10, 11, 12]),
                ],
                vec![
                    final_contact_orbit_step_for_test(&[1, 2], &[3], 0, 60, [0, 0, 0, 0]),
                    final_contact_orbit_step_for_test(&[3], &[1, 2], 0, 70, [0, 0, 0, 0]),
                    final_contact_orbit_step_for_test(&[1, 2], &[3], 0, 61, [0, 0, 0, 0]),
                    final_contact_orbit_step_for_test(&[3], &[1, 2], 0, 71, [0, 0, 0, 0]),
                    final_contact_orbit_step_for_test(&[1, 2], &[3], 0, 62, [0, 0, 0, 0]),
                    final_contact_orbit_step_for_test(&[3], &[1, 2], 0, 72, [0, 0, 0, 0]),
                ],
                vec![[3, 0], [0, 3], [4, 1], [1, 4], [5, 2], [2, 5]],
                6_u32,
            )
        } else {
            (
                vec![
                    contact_current(RecurrenceNodeKind::Source, 10, 0, &[10]),
                    contact_current(RecurrenceNodeKind::Source, 10, 1, &[11]),
                    contact_current(RecurrenceNodeKind::Current, 20, 2, &[10, 11]),
                ],
                vec![
                    partial_contact_orbit_step_for_test(0, 1, 2, 20, [0, 0, 0, 0]),
                    partial_contact_orbit_step_for_test(1, 0, 2, 30, [0, 0, 0, 0]),
                    partial_contact_orbit_step_for_test(0, 2, 1, 21, [0, 0, 0, 0]),
                    partial_contact_orbit_step_for_test(2, 0, 1, 31, [0, 0, 0, 0]),
                    partial_contact_orbit_step_for_test(0, 3, 1, 22, [0, 0, 0, 0]),
                    partial_contact_orbit_step_for_test(3, 0, 1, 32, [0, 0, 0, 0]),
                ],
                vec![[0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]],
                2_u32,
            )
        };
        let contacts = steps
            .into_iter()
            .enumerate()
            .map(|(transition, step)| {
                let transition = u32::try_from(transition).unwrap();
                let parents = parent_ids[transition as usize];
                (
                    transition,
                    contact_transition(
                        step,
                        parents.map(|parent| {
                            currents[parent as usize].key.current_state_template_id()
                        }),
                        40 + u8::try_from(transition).unwrap(),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let order = if reverse_insertion {
            (0_u32..6).rev().collect::<Vec<_>>()
        } else {
            (0_u32..6).collect::<Vec<_>>()
        };
        for transition in order {
            add_contact_contribution(
                &mut currents,
                destination_id,
                transition,
                parent_ids[transition as usize],
            );
        }
        commit_contact_plan(&mut currents, &contacts, destination_id as usize).unwrap();
        (
            contact_transition_ids(&currents[destination_id as usize]),
            pending_contact_counts(&currents),
        )
    }

    fn closure(lineages: Vec<PendingPairingLineage>) -> PendingClosure {
        PendingClosure {
            key: PendingClosureKey {
                closure_template_id: 0,
                quantum_flow_template_id: None,
                parent_current_ids: [0, 1],
                color_witness_term_id: LCColorWitnessTermId::new(0, 0),
            },
            factor: ExactComplexRational::ONE,
            component_coefficients: vec![ExactComplexRational::ONE].into_boxed_slice(),
            pairing_lineages: lineages,
        }
    }

    fn lineage(pair: [u32; 2]) -> PendingPairingLineage {
        PendingPairingLineage {
            completed_pairs: vec![pair],
            unmatched_endpoint: None,
        }
    }

    #[test]
    fn contact_0000_fan_in_keeps_exact_assignment_multiplicity_deterministically() {
        let partial_forward = contact_0000_case(false, false);
        let partial_reverse = contact_0000_case(false, true);
        assert_eq!(partial_forward, (vec![0], (3, 1)));
        assert_eq!(partial_reverse, partial_forward);

        let final_forward = contact_0000_case(true, false);
        let final_reverse = contact_0000_case(true, true);
        assert_eq!(final_reverse, final_forward);
        assert_eq!(final_forward.0.len(), 3);
        assert_eq!(final_forward.1, (7, 3));
    }

    #[test]
    fn forward_sweep_barrier_prunes_complete_contact_fan_in_before_next_stage() {
        let forward = production_contact_barrier_case(false, false);
        let reverse = production_contact_barrier_case(true, false);
        assert_eq!(reverse, forward);
        let (staged, counts, retained_partial_is_consumed) = forward;
        assert_eq!(
            staged
                .iter()
                .map(|(support, stage, _)| (support.clone(), *stage))
                .collect::<Vec<_>>(),
            vec![
                (vec![0, 1], 1),
                (vec![0, 2], 1),
                (vec![0, 3], 1),
                (vec![1, 2], 1),
                (vec![1, 3], 1),
                (vec![2, 3], 1),
                (vec![0, 1, 2], 2),
                (vec![0, 1, 3], 2),
                (vec![0, 2, 3], 2),
                (vec![1, 2, 3], 2),
            ],
        );
        assert!(staged.iter().all(|(_, stage, transition_ids)| {
            transition_ids.len() == if *stage == 1 { 1 } else { 3 }
        }));
        assert_eq!(counts, (14, 18));
        assert!(retained_partial_is_consumed);
    }

    #[test]
    fn forward_sweep_skips_contact_partial_outside_its_certified_parent_domain() {
        let forward = production_contact_barrier_case(false, true);
        let reverse = production_contact_barrier_case(true, true);
        assert_eq!(reverse, forward);
        let (staged, _, _) = forward;

        assert!(
            staged
                .iter()
                .filter(|(_, stage, _)| *stage == 1)
                .any(|(_, _, transition_ids)| {
                    transition_ids
                        .iter()
                        .any(|transition_id| *transition_id < 6)
                }),
            "one-plus-one source parents must retain a certified contact partial",
        );
        assert!(
            staged
                .iter()
                .filter(|(_, stage, _)| *stage == 1)
                .any(|(_, _, transition_ids)| transition_ids.as_slice() == [12]),
            "the ordinary scalar transition must construct a composite scalar parent",
        );
        assert!(
            staged
                .iter()
                .filter(|(_, stage, _)| *stage == 2)
                .flat_map(|(_, _, transition_ids)| transition_ids)
                .all(|transition_id| *transition_id >= 6),
            "a certified one-plus-one contact partial must not be reapplied to a composite scalar parent",
        );
        assert!(
            staged
                .iter()
                .filter(|(_, stage, _)| *stage == 2)
                .flat_map(|(_, _, transition_ids)| transition_ids)
                .any(|transition_id| (6..12).contains(transition_id)),
            "a one-plus-two certified contact final must remain available",
        );
    }

    #[test]
    fn contact_0012_partial_and_final_fan_in_each_keep_one_owner() {
        let mut partial_currents = vec![
            contact_current(RecurrenceNodeKind::Source, 10, 0, &[10]),
            contact_current(RecurrenceNodeKind::Source, 10, 1, &[11]),
            contact_current(RecurrenceNodeKind::Current, 20, 2, &[10, 11]),
        ];
        let partial_contacts = BTreeMap::from([
            (
                10,
                contact_transition(
                    partial_contact_orbit_step_for_test(0, 1, 2, 50, [0, 0, 1, 2]),
                    [10, 10],
                    52,
                ),
            ),
            (
                11,
                contact_transition(
                    partial_contact_orbit_step_for_test(1, 0, 2, 51, [0, 0, 1, 2]),
                    [10, 10],
                    53,
                ),
            ),
        ]);
        add_contact_contribution(&mut partial_currents, 2, 11, [0, 1]);
        add_contact_contribution(&mut partial_currents, 2, 10, [0, 1]);
        commit_contact_plan(&mut partial_currents, &partial_contacts, 2).unwrap();
        assert_eq!(contact_transition_ids(&partial_currents[2]), [10]);
        assert_eq!(pending_contact_counts(&partial_currents), (3, 1));

        let mut final_currents = vec![
            contact_current(RecurrenceNodeKind::Current, 20, 0, &[10, 11]),
            contact_current(RecurrenceNodeKind::Source, 30, 1, &[12]),
            contact_current(RecurrenceNodeKind::Current, 40, 2, &[10, 11, 12]),
        ];
        let final_contacts = BTreeMap::from([
            (
                20,
                contact_transition(
                    final_contact_orbit_step_for_test(&[0, 1], &[2], 3, 54, [0, 0, 1, 2]),
                    [20, 30],
                    56,
                ),
            ),
            (
                21,
                contact_transition(
                    final_contact_orbit_step_for_test(&[2], &[0, 1], 3, 55, [0, 0, 1, 2]),
                    [30, 20],
                    57,
                ),
            ),
        ]);
        add_contact_contribution(&mut final_currents, 2, 20, [0, 1]);
        add_contact_contribution(&mut final_currents, 2, 21, [1, 0]);
        commit_contact_plan(&mut final_currents, &final_contacts, 2).unwrap();
        assert_eq!(contact_transition_ids(&final_currents[2]), [21]);
        assert_eq!(pending_contact_counts(&final_currents), (3, 1));
    }

    #[test]
    fn contact_owner_planning_rolls_back_conflict_decode_and_allocation_errors() {
        let mut currents = vec![
            contact_current(RecurrenceNodeKind::Source, 10, 0, &[10]),
            contact_current(RecurrenceNodeKind::Source, 10, 1, &[11]),
            contact_current(RecurrenceNodeKind::Current, 20, 2, &[10, 11]),
        ];
        let duplicate_step = partial_contact_orbit_step_for_test(0, 1, 2, 60, [0, 0, 0, 0]);
        let contacts = BTreeMap::from([
            (30, contact_transition(duplicate_step.clone(), [10, 10], 61)),
            (31, contact_transition(duplicate_step, [10, 10], 61)),
        ]);
        add_contact_contribution(&mut currents, 2, 30, [0, 1]);
        add_contact_contribution(&mut currents, 2, 31, [0, 1]);
        let before = currents[2].contributions.clone();
        assert!(
            plan_on_the_fly_contact_orbit_owners_with_resolver(2, &currents, |transition_id| {
                contacts.get(&transition_id)
            })
            .unwrap_err()
            .to_string()
            .contains("conflicting exact rank")
        );
        assert_eq!(currents[2].contributions, before);

        let mut malformed = currents;
        let (pending, factor) = malformed[2]
            .contributions
            .first_key_value()
            .map(|(pending, factor)| (pending.clone(), *factor))
            .unwrap();
        malformed[2].contributions.remove(&pending);
        malformed[2].contributions.insert(
            PendingContributionKey {
                parent_current_ids: [u32::MAX, pending.parent_current_ids[1]],
                key: pending.key,
            },
            factor,
        );
        let malformed_before = malformed[2].contributions.clone();
        assert!(
            plan_on_the_fly_contact_orbit_owners_with_resolver(2, &malformed, |transition_id| {
                contacts.get(&transition_id)
            },)
            .unwrap_err()
            .to_string()
            .contains("left parent is absent")
        );
        assert_eq!(malformed[2].contributions, malformed_before);

        let mut candidate_reservation = Vec::<OnTheFlyContactContributionToken>::new();
        assert!(
            reserve_on_the_fly_contact_candidates(&mut candidate_reservation, usize::MAX)
                .unwrap_err()
                .to_string()
                .contains("candidate allocation failed")
        );
        assert!(candidate_reservation.is_empty());
        let mut location_reservation = Vec::<u8>::new();
        assert!(
            reserve_on_the_fly_contact_locations(&mut location_reservation, usize::MAX)
                .unwrap_err()
                .to_string()
                .contains("transition-index allocation failed")
        );
        assert!(location_reservation.is_empty());
    }

    #[test]
    fn contact_index_and_uncertified_controls_use_production_lookup_and_fast_path() {
        let none = contact_orbit_test_template(ContactOrbitTestBinding::None)
            .validate()
            .unwrap();
        let none_catalog = TemplateCatalog::new(none.input()).unwrap();
        let none_transitions = prepared_transitions(&none, &none_catalog).unwrap();
        assert!(
            PreparedContactOrbitIndex::new(&none_transitions)
                .unwrap()
                .is_empty()
        );

        let one = contact_orbit_test_template(ContactOrbitTestBinding::One)
            .validate()
            .unwrap();
        let one_catalog = TemplateCatalog::new(one.input()).unwrap();
        let one_transitions = prepared_transitions(&one, &one_catalog).unwrap();
        let index = PreparedContactOrbitIndex::new(&one_transitions).unwrap();
        assert!(!index.is_empty());
        assert!(index.get(0).is_some());
        assert!(index.get(u32::MAX).is_none());

        let mut currents = vec![
            contact_current(RecurrenceNodeKind::Source, 10, 0, &[10]),
            contact_current(RecurrenceNodeKind::Source, 11, 1, &[11]),
            contact_current(RecurrenceNodeKind::Current, 20, 2, &[10, 11]),
        ];
        // Distinct uncertified rows represent ordinary V3, vector, fermion,
        // and QCD controls. No model-level class is inspected by this path.
        for transition in 70..74 {
            add_contact_contribution(&mut currents, 2, transition, [0, 1]);
        }
        let before = currents[2].contributions.clone();
        assert!(
            plan_on_the_fly_contact_orbit_owners(
                2,
                &PreparedContactOrbitIndex::default(),
                &currents
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(currents[2].contributions, before);
        assert_eq!(pending_contact_counts(&currents), (3, 4));
    }

    #[test]
    fn pairing_lineage_capacity_is_checked_without_an_arbitrary_limit() {
        assert_eq!(
            checked_pairing_capacity(1 << 20, 1 << 20, 1).unwrap(),
            2_097_153
        );
        assert!(checked_pairing_capacity(usize::MAX, 1, 0).is_err());
    }

    #[test]
    fn large_pairing_lineage_clone_is_fallible_and_exact() {
        let pair_count = 16_384usize;
        let mut completed_pairs = Vec::new();
        completed_pairs.try_reserve_exact(pair_count).unwrap();
        for index in 0..pair_count {
            completed_pairs.push([index as u32, (index + pair_count) as u32]);
        }
        let lineage = PendingPairingLineage {
            completed_pairs,
            unmatched_endpoint: None,
        };
        let copied = try_clone_pairing_lineage(&lineage).unwrap();
        assert_eq!(copied, lineage);
    }

    #[test]
    fn canonical_projected_pairing_owner_fails_closed_on_ambiguous_lineage() {
        assert!(
            unique_projected_pairing_owner(&[closure(vec![lineage([0, 1]), lineage([0, 3])])])
                .is_err()
        );
        assert!(
            unique_projected_pairing_owner(&[
                closure(vec![lineage([0, 1])]),
                closure(vec![lineage([0, 3])]),
            ])
            .is_err()
        );
    }
}
