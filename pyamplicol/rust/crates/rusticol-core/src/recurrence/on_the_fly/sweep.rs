// SPDX-License-Identifier: 0BSD

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
    pub(super) selected_pairing_compatible: bool,
    pub(super) stage: u16,
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
        || source.state_template_id != state.current_state_template_id
        || current_state.id != state.current_state_template_id
        || source.helicity != state.source_helicity
        || source.spin_state != state.spin_state
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

fn selected_pairing_compatible_for_current(
    query: &DecodedLcQueryV1,
    templates: &ValidatedRecurrenceTemplateInput,
    current_state_template_id: u32,
    support_source_slots: &[u32],
) -> RusticolResult<bool> {
    let state = templates
        .input()
        .current_states
        .get(current_state_template_id as usize)
        .ok_or_else(|| integrity("pairing support current state is absent"))?;
    let carries_colored_fermion_line = state.statistics == 1 && state.color_representation != 1;
    Ok(query.selected_pairing_compatible(support_source_slots, carries_colored_fermion_line))
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
        let (_anchor, state) = selected_source_state(seed, selected)?;
        let (source, current_state) = validate_source_contract(templates, catalog, state)?;
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
            selected_pairing_compatible: selected_pairing_compatible_for_current(
                query,
                templates,
                state.current_state_template_id,
                &[selected.source_slot],
            )?,
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
    query: &DecodedLcQueryV1,
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
    let selected_pairing_compatible = selected_pairing_compatible_for_current(
        query,
        templates,
        prepared.row.result_state_template_id,
        &support,
    )? && currents[concrete_parent_ids[0] as usize]
        .selected_pairing_compatible
        && currents[concrete_parent_ids[1] as usize].selected_pairing_compatible;
    if !selected_pairing_compatible {
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
                selected_pairing_compatible,
                stage: u16::try_from(support.len() - 1)
                    .map_err(|_| invalid("query-local current stage exceeds u16"))?,
            });
            id
        };
        currents[result_id as usize].selected_pairing_compatible |= selected_pairing_compatible;
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
    query: &DecodedLcQueryV1,
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
                        query,
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
            if !currents[concrete_parent_ids[0] as usize].selected_pairing_compatible
                || !currents[concrete_parent_ids[1] as usize].selected_pairing_compatible
            {
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
                        quantum.output_factor,
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
                            });
                        }
                        std::collections::btree_map::Entry::Occupied(mut entry) => {
                            if entry.get().component_coefficients != coefficients {
                                return Err(integrity(
                                    "equal closure identities have different component coefficients",
                                ));
                            }
                            aggregate_factor(&mut entry.get_mut().factor, factor)?;
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
