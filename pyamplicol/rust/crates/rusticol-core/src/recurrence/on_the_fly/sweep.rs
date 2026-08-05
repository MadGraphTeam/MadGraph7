// SPDX-License-Identifier: 0BSD

use super::source_seed::validate_permutation;
use super::*;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct PendingContributionKey {
    pub(super) parent_current_ids: [u32; 2],
    pub(super) key: ContributionKey,
}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) struct PendingPairingLineage {
    pub(super) completed_pairs: Box<[[u32; 2]]>,
    pub(super) unmatched_endpoint: Option<u32>,
}

impl PendingPairingLineage {
    fn source(seed: &OnTheFlyProcessSeedV1, source_slot: u32) -> Self {
        let unmatched_endpoint = seed.source_anchors[source_slot as usize]
            .color_role
            .is_pairing_endpoint()
            .then_some(source_slot);
        Self {
            completed_pairs: Box::new([]),
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

#[derive(Clone, Debug)]
pub(super) struct PendingClosure {
    pub(super) key: PendingClosureKey,
    pub(super) factor: ExactComplexRational,
    pub(super) component_coefficients: Box<[ExactComplexRational]>,
    pub(super) pairing_lineages: Vec<PendingPairingLineage>,
}

/// Query-local Wick-lineage proof selected only after physical closure and
/// complete-rectangle color-alias projection have identified a canonical
/// root-bearing representative.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ResolvedPairingOwnerV1 {
    pub(super) endpoint_pairs: Box<[[u32; 2]]>,
    pub(super) proof_digest: Option<SemanticDigest>,
    pub(super) source_slot_permutation: Box<[u32]>,
    pub(super) source_lineage: Box<[u32]>,
    pub(super) fermion_parity: i32,
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
    let mut completed_pairs = left
        .completed_pairs
        .iter()
        .chain(right.completed_pairs.iter())
        .copied()
        .collect::<Vec<_>>();
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
        completed_pairs: completed_pairs.into_boxed_slice(),
        unmatched_endpoint,
    }))
}

fn combine_pairing_lineage_sets(
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
    target
        .try_reserve(source.len())
        .map_err(|error| invalid(format!("pairing-lineage merge allocation failed: {error}")))?;
    target.extend_from_slice(source);
    target.sort_unstable();
    target.dedup();
    Ok(())
}

fn complete_pairing_lineage(
    seed: &OnTheFlyProcessSeedV1,
    lineage: &PendingPairingLineage,
) -> RusticolResult<bool> {
    if lineage.unmatched_endpoint.is_some() {
        return Ok(false);
    }
    let expected = seed
        .source_anchors
        .iter()
        .filter(|anchor| anchor.color_role.is_pairing_endpoint())
        .map(|anchor| anchor.source_slot)
        .collect::<BTreeSet<_>>();
    let mut observed = BTreeSet::new();
    for pair in lineage.completed_pairs.iter().copied() {
        if close_pairing_endpoints(seed, pair[0], pair[1])? != Some(pair)
            || !observed.insert(pair[0])
            || !observed.insert(pair[1])
        {
            return Ok(false);
        }
    }
    Ok(observed == expected)
}

fn identity_source_permutation(source_count: usize) -> RusticolResult<Vec<u32>> {
    (0..source_count)
        .map(|value| checked_u32(value, "pairing identity source slot"))
        .collect()
}

fn lehmer_digits(reference: &[u32], selected: &[u32]) -> RusticolResult<Vec<u32>> {
    if reference.len() != selected.len() {
        return Err(invalid(
            "discovered pairing length differs from its compact class",
        ));
    }
    let mut remaining = reference.to_vec();
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
    let mut source_lineage = vec![MISSING_U32; seed.source_anchors.len()];
    if seed.pairing_classes.is_empty() {
        if closures.iter().any(|closure| {
            closure.pairing_lineages.as_slice()
                != [PendingPairingLineage {
                    completed_pairs: Box::new([]),
                    unmatched_endpoint: None,
                }]
        }) {
            return Err(integrity(
                "pairing-free query retained a nontrivial closure lineage",
            ));
        }
        return Ok(ResolvedPairingOwnerV1 {
            endpoint_pairs: Box::new([]),
            proof_digest: None,
            source_slot_permutation: source_slot_permutation.into_boxed_slice(),
            source_lineage: source_lineage.into_boxed_slice(),
            fermion_parity: 1,
        });
    }
    let owner = unique_projected_pairing_owner(closures)?;
    if !complete_pairing_lineage(seed, &owner)? {
        return Err(integrity(
            "canonical projected closure has an incomplete Wick lineage",
        ));
    }
    let endpoint_pairs = owner.completed_pairs.to_vec();
    let pairing_by_fundamental = endpoint_pairs
        .iter()
        .copied()
        .map(|[fundamental, antifundamental]| (fundamental, antifundamental))
        .collect::<BTreeMap<_, _>>();
    if pairing_by_fundamental.len() != endpoint_pairs.len() {
        return Err(integrity(
            "canonical projected closure repeats a fundamental endpoint",
        ));
    }
    let mut parity = 1_i32;
    let mut proof_hash = Sha256::new();
    proof_hash.update(b"pyamplicol-on-the-fly-discovered-pairing-owner-v1\0");
    for pairing_class in &seed.pairing_classes {
        let reference = pairing_class
            .antifundamental_endpoints
            .iter()
            .map(|endpoint| endpoint.source_slot)
            .collect::<Vec<_>>();
        let selected = pairing_class
            .fundamental_endpoints
            .iter()
            .map(|endpoint| {
                pairing_by_fundamental
                    .get(&endpoint.source_slot)
                    .copied()
                    .ok_or_else(|| {
                        integrity("canonical Wick lineage omits a pairing-class fundamental")
                    })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
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
        endpoint_pairs: endpoint_pairs.into_boxed_slice(),
        proof_digest: Some(final_digest(proof_hash)?),
        source_slot_permutation: source_slot_permutation.into_boxed_slice(),
        source_lineage: source_lineage.into_boxed_slice(),
        fermion_parity: parity,
    })
}

fn unique_projected_pairing_owner(
    closures: &[PendingClosure],
) -> RusticolResult<PendingPairingLineage> {
    let owners = closures
        .iter()
        .map(|closure| {
            let [lineage] = closure.pairing_lineages.as_slice() else {
                return Err(integrity(format!(
                    "canonical projected closure has {} Wick lineages, expected exactly one",
                    closure.pairing_lineages.len(),
                )));
            };
            Ok(lineage.clone())
        })
        .collect::<RusticolResult<BTreeSet<_>>>()?;
    if owners.len() != 1 {
        return Err(integrity(format!(
            "canonical projected closures disagree across {} Wick lineages",
            owners.len(),
        )));
    }
    let owner = owners
        .iter()
        .next()
        .ok_or_else(|| integrity("canonical projected closure has no Wick lineage"))?;
    Ok(owner.clone())
}

fn supports_are_disjoint(left: &[u32], right: &[u32]) -> bool {
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

fn merge_disjoint_support(left: &[u32], right: &[u32]) -> RusticolResult<Vec<u32>> {
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
    if catalog.coupling_order_dimension() != seed.coupling_limits.len() {
        return Err(integrity(
            "compact coupling-limit dimension differs from the template catalog",
        ));
    }
    Ok(catalog)
}

fn validate_source_contract(
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
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    seed: &OnTheFlyProcessSeedV1,
    query: &DecodedLcQueryV1,
    colors: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
) -> RusticolResult<()> {
    let zero_orders = vec![0_u32; seed.coupling_limits.len()];
    for selected in query.selected_sources.iter().copied() {
        let (anchor, state) = selected_source_state(seed, selected)?;
        let (source, current_state) = validate_source_contract(templates, catalog, anchor, state)?;
        let color = catalog
            .source_seed(source)?
            .instantiate(selected.source_slot, current_state.color_representation)?;
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
    let Some(coupling_orders) = combined_coupling_orders(
        parents[0].coupling_orders(),
        parents[1].coupling_orders(),
        &prepared.local_orders,
        &seed.coupling_limits,
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
            current_ids.insert(key.clone(), id);
            currents.push(PendingCurrent {
                key,
                source_factor: None,
                contributions: BTreeMap::new(),
                pairing_lineages: pairing_lineages.clone(),
                stage: checked_u32(support.len() - 1, "query-local current stage")?,
            });
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
        let factor = multiply_factors(&[
            prepared.base_factor,
            prepared.output_factor()?,
            exchange_factor,
            prepared_witness.witness.exact_factor(),
        ])?;
        let aggregate = currents[result_id as usize]
            .contributions
            .entry(pending_key)
            .or_insert(ExactComplexRational::ZERO);
        aggregate_factor(aggregate, factor)?;
    }
    Ok(())
}

pub(super) fn build_forward_currents(
    templates: &ValidatedRecurrenceTemplateInput,
    transitions: &BTreeMap<(u32, u32), Vec<PreparedTransition>>,
    seed: &OnTheFlyProcessSeedV1,
    propagators: &BTreeMap<u32, Option<u32>>,
    colors: &mut DynamicLCColorStateInterner,
    currents: &mut Vec<PendingCurrent>,
    current_ids: &mut BTreeMap<CurrentCoreKey, u32>,
) -> RusticolResult<()> {
    let source_count = seed.source_anchors.len();
    for target_size in 2..source_count {
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
                        propagators,
                        colors,
                        currents,
                        current_ids,
                    )?;
                }
            }
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
    templates: &ValidatedRecurrenceTemplateInput,
    closures: &BTreeMap<(u32, u32), Vec<PreparedClosure>>,
    seed: &OnTheFlyProcessSeedV1,
    query: &DecodedLcQueryV1,
    colors: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
) -> RusticolResult<Vec<PendingClosure>> {
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
                    let coefficients = templates
                        .closure_component_coefficients(closure.row.id)?
                        .into_boxed_slice();
                    match retained.entry(key.clone()) {
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            entry.insert(PendingClosure {
                                key,
                                factor,
                                component_coefficients: coefficients,
                                pairing_lineages: pairing_lineages.clone(),
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
        return Err(invalid(
            "query-local construction found no exact closure for the decoded LC selector",
        ));
    }
    Ok(retained.into_values().collect())
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
    use super::*;

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
            completed_pairs: vec![pair].into_boxed_slice(),
            unmatched_endpoint: None,
        }
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
