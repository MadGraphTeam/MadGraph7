// SPDX-License-Identifier: 0BSD

//! Selector-independent resolution of the process coupling-order policy.
//!
//! The public minimal policy is process-global.  Resolving it from one chosen
//! helicity or colour selector can accidentally promote a higher-order branch
//! when the true leading-order amplitude vanishes only for that selector.  The
//! cold sweep below therefore erases public helicity labels while retaining
//! every structural contract that controls reachability: source support,
//! current state and spin, flavour/quantum ancestry, exact LC colour state,
//! and fermion-pairing lineage.  It never constructs a public colour-flow
//! table, a recurrence DAG, or executable contributions.

use std::collections::BTreeMap;

use super::public_query::physical_lc_selector_closure_anchor;
use super::*;
use crate::recurrence::contact_orbit_owner::ContactOrbitParentTopologyDomain;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct OnTheFlyCouplingPolicyCensusV1 {
    pub(crate) source_topology_count: u64,
    pub(crate) retained_topology_count: u64,
    pub(crate) transition_witness_attempt_count: u64,
    pub(crate) closure_witness_attempt_count: u64,
    pub(crate) viable_total_order_count: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlyResolvedCouplingPolicyV1 {
    seed_digest: SemanticDigest,
    effective_limits: Box<[Option<u32>]>,
    census: OnTheFlyCouplingPolicyCensusV1,
}

impl OnTheFlyResolvedCouplingPolicyV1 {
    pub(crate) const fn seed_digest(&self) -> SemanticDigest {
        self.seed_digest
    }

    pub(crate) fn effective_limits(&self) -> &[Option<u32>] {
        &self.effective_limits
    }

    pub(crate) const fn census(&self) -> OnTheFlyCouplingPolicyCensusV1 {
        self.census
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct GlobalTopologyKey {
    support: Box<[u32]>,
    current_state_template_id: u32,
    spin_state: i32,
    flavour_flow: Box<[i32]>,
    quantum_number_flow_id: u32,
    color: DynamicLCColorState,
    pairing_lineage: PendingPairingLineage,
}

#[derive(Clone, Debug)]
struct GlobalTopologyNode {
    key: GlobalTopologyKey,
    /// Componentwise Pareto-minimal order vectors reaching this exact state.
    orders: Vec<Box<[u32]>>,
}

fn checked_increment(value: &mut u64, label: &str) -> RusticolResult<()> {
    *value = value
        .checked_add(1)
        .ok_or_else(|| invalid(format!("{label} exceeds u64")))?;
    Ok(())
}

fn weakly_dominates(left: &[u32], right: &[u32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(left, right)| left <= right)
}

fn insert_pareto_order(orders: &mut Vec<Box<[u32]>>, candidate: Vec<u32>) -> bool {
    if orders
        .iter()
        .any(|existing| weakly_dominates(existing, &candidate))
    {
        return false;
    }
    orders.retain(|existing| !weakly_dominates(&candidate, existing));
    orders.push(candidate.into_boxed_slice());
    orders.sort_unstable();
    true
}

fn insert_topology_order(
    nodes: &mut Vec<GlobalTopologyNode>,
    node_ids: &mut BTreeMap<GlobalTopologyKey, usize>,
    key: GlobalTopologyKey,
    orders: Vec<u32>,
) -> RusticolResult<()> {
    if let Some(id) = node_ids.get(&key).copied() {
        insert_pareto_order(&mut nodes[id].orders, orders);
        return Ok(());
    }
    let id = nodes.len();
    nodes
        .try_reserve(1)
        .map_err(|error| invalid(format!("global topology allocation failed: {error}")))?;
    let mut retained_orders = Vec::new();
    retained_orders
        .try_reserve_exact(1)
        .map_err(|error| invalid(format!("global coupling-order allocation failed: {error}")))?;
    retained_orders.push(orders.into_boxed_slice());
    node_ids.insert(key.clone(), id);
    nodes.push(GlobalTopologyNode {
        key,
        orders: retained_orders,
    });
    Ok(())
}

fn ordered_transition_parents<'a>(
    prepared: &PreparedTransition,
    left: &'a GlobalTopologyKey,
    right: &'a GlobalTopologyKey,
) -> Option<[&'a GlobalTopologyKey; 2]> {
    prepared
        .parent_ids(
            left.current_state_template_id,
            right.current_state_template_id,
            0,
            1,
        )
        .map(|ids| {
            let parents = [left, right];
            [parents[ids[0] as usize], parents[ids[1] as usize]]
        })
}

fn ordered_closure_parents<'a>(
    prepared: &PreparedClosure,
    left: &'a GlobalTopologyKey,
    right: &'a GlobalTopologyKey,
) -> Option<[&'a GlobalTopologyKey; 2]> {
    prepared
        .parent_ids(
            left.current_state_template_id,
            right.current_state_template_id,
            0,
            1,
        )
        .map(|ids| {
            let parents = [left, right];
            [parents[ids[0] as usize], parents[ids[1] as usize]]
        })
}

fn topology_quantum_matches(
    states: [u32; 2],
    spins: [i32; 2],
    parents: [&GlobalTopologyKey; 2],
) -> bool {
    (0..2).all(|index| {
        states[index] == parents[index].current_state_template_id
            && spins[index] == parents[index].spin_state
    })
}

fn contact_orbit_topology_domain(
    parent: &GlobalTopologyKey,
) -> ContactOrbitParentTopologyDomain<'_> {
    if parent.support.len() == 1 {
        ContactOrbitParentTopologyDomain::source(parent.current_state_template_id, &parent.support)
    } else {
        ContactOrbitParentTopologyDomain::current(parent.current_state_template_id, &parent.support)
    }
}

fn closure_quantum_matches_topology(
    quantum: &PreparedClosureQuantum,
    parents: [&GlobalTopologyKey; 2],
) -> bool {
    match (quantum.input_states, quantum.input_spins) {
        (None, None) => true,
        (Some(states), Some(spins)) => topology_quantum_matches(states, spins, parents),
        _ => false,
    }
}

fn insert_all_sources(
    seed: &OnTheFlyProcessSeedV1,
    grammar: &PreparedOnTheFlyGrammarV1,
    nodes: &mut Vec<GlobalTopologyNode>,
    node_ids: &mut BTreeMap<GlobalTopologyKey, usize>,
) -> RusticolResult<u64> {
    let zero_orders = vec![0_u32; seed.explicit_coupling_limits().len()];
    let mut source_topology_count = 0_u64;
    for anchor in &seed.source_anchors {
        for state in &anchor.states {
            let contract = grammar
                .sources
                .get(&(state.source_template_id, state.current_state_template_id))
                .ok_or_else(|| integrity("global source has no prepared source contract"))?;
            let color = contract.color_seed.instantiate(
                anchor.source_slot,
                contract.current_state.color_representation,
            )?;
            let key = GlobalTopologyKey {
                support: vec![anchor.source_slot].into_boxed_slice(),
                current_state_template_id: state.current_state_template_id,
                spin_state: state.spin_state,
                flavour_flow: state.flavour_flow.clone(),
                quantum_number_flow_id: state.quantum_number_flow_id,
                color,
                pairing_lineage: PendingPairingLineage::source(seed, anchor.source_slot),
            };
            let previous = nodes.len();
            insert_topology_order(nodes, node_ids, key, zero_orders.clone())?;
            if nodes.len() != previous {
                checked_increment(&mut source_topology_count, "source topology count")?;
            }
        }
    }
    Ok(source_topology_count)
}

#[allow(clippy::too_many_arguments)]
fn apply_transition(
    templates: &ValidatedRecurrenceTemplateInput,
    prepared: &PreparedTransition,
    left: &GlobalTopologyNode,
    right: &GlobalTopologyNode,
    source_count: usize,
    seed: &OnTheFlyProcessSeedV1,
    hard_limits: &[Option<u32>],
    nodes: &mut Vec<GlobalTopologyNode>,
    node_ids: &mut BTreeMap<GlobalTopologyKey, usize>,
    census: &mut OnTheFlyCouplingPolicyCensusV1,
) -> RusticolResult<()> {
    let Some(parents) = ordered_transition_parents(prepared, &left.key, &right.key) else {
        return Ok(());
    };
    if !topology_quantum_matches(prepared.input_states, prepared.input_spins, parents) {
        return Ok(());
    }
    if let Some(contact_orbit) = prepared.contact_orbit.as_ref() {
        if !contact_orbit.accepts_parent_topology_domain([
            contact_orbit_topology_domain(parents[0]),
            contact_orbit_topology_domain(parents[1]),
        ])? {
            return Ok(());
        }
    }
    // Contact-orbit certificates choose one amplitude owner among the
    // admitted equivalent contributions. Retaining all certified owner
    // orientations cannot introduce a distinct coupling-order vector.
    let _ = prepared.output_factor()?;
    let support = merge_disjoint_support(&parents[0].support, &parents[1].support)?;
    if support.len() >= source_count {
        return Ok(());
    }
    let result_state = templates
        .input()
        .current_states
        .get(prepared.row.result_state_template_id as usize)
        .ok_or_else(|| integrity("global topology result state is absent"))?;
    let carries_colored_fermion_line =
        result_state.statistics == 1 && result_state.color_representation != 1;
    let pairing_lineages = combine_pairing_lineage_sets(
        // Each topology key carries one exact lineage alternative.  Keeping
        // it in the key prevents an invalid endpoint pairing from borrowing
        // the coupling order of another alternative.
        //
        // No public selector enters this operation.
        seed,
        std::slice::from_ref(&parents[0].pairing_lineage),
        std::slice::from_ref(&parents[1].pairing_lineage),
        carries_colored_fermion_line,
    )?;
    if pairing_lineages.is_empty() {
        return Ok(());
    }
    let flavour = prepared
        .flavour
        .apply_flows(&parents[0].flavour_flow, &parents[1].flavour_flow);
    for witness in &prepared.witnesses {
        if witness.row.left_shape_string_id != parents[0].color.output_color_shape_id()
            || witness.row.right_shape_string_id != parents[1].color.output_color_shape_id()
        {
            continue;
        }
        checked_increment(
            &mut census.transition_witness_attempt_count,
            "global transition witness-attempt count",
        )?;
        let Some(color) = witness
            .witness
            .apply(&parents[0].color, &parents[1].color)?
        else {
            continue;
        };
        for lineage in &pairing_lineages {
            let key = GlobalTopologyKey {
                support: support.clone().into_boxed_slice(),
                current_state_template_id: prepared.row.result_state_template_id,
                spin_state: prepared.quantum.result_spin_state,
                flavour_flow: flavour.clone().into_boxed_slice(),
                quantum_number_flow_id: prepared.quantum.result_quantum_number_flow_id,
                color: color.clone(),
                pairing_lineage: lineage.clone(),
            };
            for left_orders in &left.orders {
                for right_orders in &right.orders {
                    let Some(orders) = combined_coupling_orders(
                        left_orders,
                        right_orders,
                        &prepared.local_orders,
                        hard_limits,
                    )?
                    else {
                        continue;
                    };
                    insert_topology_order(nodes, node_ids, key.clone(), orders)?;
                }
            }
        }
    }
    Ok(())
}

fn build_global_topologies(
    templates: &ValidatedRecurrenceTemplateInput,
    seed: &OnTheFlyProcessSeedV1,
    grammar: &PreparedOnTheFlyGrammarV1,
    census: &mut OnTheFlyCouplingPolicyCensusV1,
) -> RusticolResult<Vec<GlobalTopologyNode>> {
    let source_count = seed.source_anchors.len();
    let hard_limits = seed.explicit_coupling_limits();
    let mut nodes = Vec::new();
    let mut node_ids = BTreeMap::new();
    census.source_topology_count = insert_all_sources(seed, grammar, &mut nodes, &mut node_ids)?;

    for target_size in 2..source_count {
        // New nodes produced at this stage have support `target_size`, so a
        // stable snapshot of smaller parents is complete for the stage.
        let parents = nodes
            .iter()
            .filter(|node| node.key.support.len() < target_size)
            .cloned()
            .collect::<Vec<_>>();
        for (left_index, left) in parents.iter().enumerate() {
            for right in parents.iter().skip(left_index + 1) {
                if left.key.support.len() + right.key.support.len() != target_size
                    || !supports_are_disjoint(&left.key.support, &right.key.support)
                {
                    continue;
                }
                let Some(rows) = grammar.transitions.get(&canonical_state_pair(
                    left.key.current_state_template_id,
                    right.key.current_state_template_id,
                )) else {
                    continue;
                };
                for prepared in rows {
                    apply_transition(
                        templates,
                        prepared,
                        left,
                        right,
                        source_count,
                        seed,
                        hard_limits,
                        &mut nodes,
                        &mut node_ids,
                        census,
                    )?;
                }
            }
        }
    }
    census.retained_topology_count = u64::try_from(nodes.len())
        .map_err(|_| invalid("global retained-topology count exceeds u64"))?;
    Ok(nodes)
}

fn closure_has_canonical_anchor(
    left: &GlobalTopologyKey,
    right: &GlobalTopologyKey,
    anchor: u32,
    source_count: usize,
) -> bool {
    (left.support.as_ref() == [anchor] && right.support.len() + 1 == source_count)
        || (right.support.as_ref() == [anchor] && left.support.len() + 1 == source_count)
}

fn collect_viable_total_orders(
    seed: &OnTheFlyProcessSeedV1,
    closures: &BTreeMap<(u32, u32), Vec<PreparedClosure>>,
    nodes: &[GlobalTopologyNode],
    census: &mut OnTheFlyCouplingPolicyCensusV1,
) -> RusticolResult<Vec<Box<[u32]>>> {
    let source_count = seed.source_anchors.len();
    let mut totals = Vec::new();
    for (left_index, left) in nodes.iter().enumerate() {
        for right in nodes.iter().skip(left_index + 1) {
            if left.key.support.len() + right.key.support.len() != source_count
                || !supports_are_disjoint(&left.key.support, &right.key.support)
            {
                continue;
            }
            let Some(rows) = closures.get(&canonical_state_pair(
                left.key.current_state_template_id,
                right.key.current_state_template_id,
            )) else {
                continue;
            };
            for closure in rows {
                let Some(parents) = ordered_closure_parents(closure, &left.key, &right.key) else {
                    continue;
                };
                let pairing_lineages = combine_pairing_lineage_sets(
                    seed,
                    std::slice::from_ref(&parents[0].pairing_lineage),
                    std::slice::from_ref(&parents[1].pairing_lineage),
                    false,
                )?;
                if !pairing_lineages
                    .iter()
                    .try_fold(false, |complete, lineage| {
                        Ok::<_, RusticolError>(complete || complete_pairing_lineage(seed, lineage)?)
                    })?
                {
                    continue;
                }
                for quantum in closure
                    .quantum_flows
                    .iter()
                    .filter(|quantum| closure_quantum_matches_topology(quantum, parents))
                {
                    let _ = quantum.output_factor()?;
                    for witness in &closure.witnesses {
                        if witness.row.left_shape_string_id
                            != parents[0].color.output_color_shape_id()
                            || witness.row.right_shape_string_id
                                != parents[1].color.output_color_shape_id()
                        {
                            continue;
                        }
                        checked_increment(
                            &mut census.closure_witness_attempt_count,
                            "global closure witness-attempt count",
                        )?;
                        let closed = witness
                            .witness
                            .closed_components(&parents[0].color, &parents[1].color)?;
                        let Some(anchor) = physical_lc_selector_closure_anchor(seed, &closed)
                        else {
                            continue;
                        };
                        if !closure_has_canonical_anchor(
                            &left.key,
                            &right.key,
                            anchor,
                            source_count,
                        ) {
                            continue;
                        }
                        for left_orders in &left.orders {
                            for right_orders in &right.orders {
                                let Some(total) = combined_coupling_orders(
                                    left_orders,
                                    right_orders,
                                    &closure.local_orders,
                                    seed.explicit_coupling_limits(),
                                )?
                                else {
                                    continue;
                                };
                                insert_pareto_order(&mut totals, total);
                            }
                        }
                    }
                }
            }
        }
    }
    census.viable_total_order_count = u64::try_from(totals.len())
        .map_err(|_| invalid("global viable-total-order count exceeds u64"))?;
    Ok(totals)
}

fn hierarchy_degree(orders: &[u32], hierarchies: &[u32]) -> RusticolResult<u64> {
    if orders.len() != hierarchies.len() {
        return Err(integrity(
            "global coupling order and hierarchy dimensions disagree",
        ));
    }
    orders
        .iter()
        .zip(hierarchies.iter())
        .try_fold(0_u64, |total, (order, hierarchy)| {
            total
                .checked_add(u64::from(*order) * u64::from(*hierarchy))
                .ok_or_else(|| invalid("hierarchy-weighted coupling degree exceeds u64"))
        })
}

fn minimal_envelope(
    totals: &[Box<[u32]>],
    hierarchies: &[u32],
) -> RusticolResult<Option<Vec<Option<u32>>>> {
    let Some(minimum) = totals
        .iter()
        .map(|orders| hierarchy_degree(orders, hierarchies))
        .collect::<RusticolResult<Vec<_>>>()?
        .into_iter()
        .min()
    else {
        return Ok(None);
    };
    let mut envelope = vec![0_u32; hierarchies.len()];
    for orders in totals {
        if hierarchy_degree(orders, hierarchies)? != minimum {
            continue;
        }
        for (maximum, value) in envelope.iter_mut().zip(orders.iter()) {
            *maximum = (*maximum).max(*value);
        }
    }
    Ok(Some(envelope.into_iter().map(Some).collect()))
}

pub(crate) fn resolve_on_the_fly_coupling_policy_v1(
    templates: &ValidatedRecurrenceTemplateInput,
    seed: &OnTheFlyProcessSeedV1,
) -> RusticolResult<OnTheFlyResolvedCouplingPolicyV1> {
    let catalog = validate_seed_against_templates(templates, seed)?;
    let grammar = prepare_on_the_fly_grammar_v1(templates, &catalog, seed)?;
    resolve_on_the_fly_coupling_policy_from_grammar_v1(templates, seed, &grammar)
}

pub(super) fn resolve_on_the_fly_coupling_policy_from_grammar_v1(
    templates: &ValidatedRecurrenceTemplateInput,
    seed: &OnTheFlyProcessSeedV1,
    grammar: &PreparedOnTheFlyGrammarV1,
) -> RusticolResult<OnTheFlyResolvedCouplingPolicyV1> {
    if seed.coupling_order_policy() == OnTheFlyCouplingOrderPolicyV1::Explicit {
        return Ok(OnTheFlyResolvedCouplingPolicyV1 {
            seed_digest: seed.semantic_digest(),
            effective_limits: seed.explicit_coupling_limits().to_vec().into_boxed_slice(),
            census: OnTheFlyCouplingPolicyCensusV1::default(),
        });
    }
    let mut census = OnTheFlyCouplingPolicyCensusV1::default();
    let nodes = build_global_topologies(templates, seed, grammar, &mut census)?;
    let totals = collect_viable_total_orders(seed, &grammar.closures, &nodes, &mut census)?;
    let effective_limits = minimal_envelope(&totals, seed.coupling_hierarchies())?
        .unwrap_or_else(|| seed.explicit_coupling_limits().to_vec());
    Ok(OnTheFlyResolvedCouplingPolicyV1 {
        seed_digest: seed.semantic_digest(),
        effective_limits: effective_limits.into_boxed_slice(),
        census,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::template::{CouplingOrderTermRow, IndexedRangeRow};
    use crate::recurrence::{CheckedTableRange, validated_template_fixture};

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).unwrap()
    }

    fn scalar_templates() -> ValidatedRecurrenceTemplateInput {
        let mut input = validated_template_fixture().into_input();
        let spin_start = u64::try_from(input.i32_sequence_values.len()).unwrap();
        let spin_sequence_id = u32::try_from(input.i32_sequence_ranges.len()).unwrap();
        input.i32_sequence_ranges.push(IndexedRangeRow {
            id: spin_sequence_id,
            range: CheckedTableRange::new(spin_start, 2),
        });
        input.i32_sequence_values.extend([50_000, 50_000]);
        input.quantum_flows[0].input_spin_sequence_id = spin_sequence_id;
        input.coupling_order_ranges.push(IndexedRangeRow {
            id: 1,
            range: CheckedTableRange::new(0, 1),
        });
        input.coupling_order_terms.push(CouplingOrderTermRow {
            set_id: 1,
            name_string_id: 0,
            power: 1,
        });
        input.validate().unwrap()
    }

    fn with_policy(
        seed: OnTheFlyProcessSeedV1,
        policy: OnTheFlyCouplingOrderPolicyV1,
        hierarchies: Vec<u32>,
        limits: Vec<Option<u32>>,
    ) -> OnTheFlyProcessSeedV1 {
        let OnTheFlyProcessSeedV1 {
            process_digest,
            model_digest,
            template_catalog_digest,
            prepared_pack_digest,
            direct_catalog_digest,
            normalization_semantic_digest,
            normalization_convention,
            source_anchors,
            external_permutation,
            coupling_order_policy: _,
            coupling_hierarchies: _,
            coupling_limits: _,
            pairing_classes,
            semantic_digest: _,
        } = seed;
        OnTheFlyProcessSeedV1::new(
            process_digest,
            model_digest,
            template_catalog_digest,
            prepared_pack_digest,
            direct_catalog_digest,
            normalization_semantic_digest,
            normalization_convention,
            ExactComplexRational::ONE,
            source_anchors.into_vec(),
            external_permutation.into_vec(),
            policy,
            hierarchies,
            limits,
            pairing_classes.into_vec(),
        )
        .unwrap()
    }

    fn with_selector_local_zero(seed: OnTheFlyProcessSeedV1) -> OnTheFlyProcessSeedV1 {
        let OnTheFlyProcessSeedV1 {
            process_digest,
            model_digest,
            template_catalog_digest,
            prepared_pack_digest,
            direct_catalog_digest,
            normalization_semantic_digest,
            normalization_convention,
            source_anchors,
            external_permutation,
            coupling_order_policy,
            coupling_hierarchies,
            coupling_limits,
            pairing_classes,
            semantic_digest: _,
        } = seed;
        let anchors = source_anchors
            .into_vec()
            .into_iter()
            .map(|anchor| {
                let OnTheFlySourceAnchorV1 {
                    source_slot,
                    external_label,
                    is_initial,
                    color_role,
                    is_fermionic,
                    pairing_source_contract_digest,
                    states,
                } = anchor;
                let mut states = states.into_vec();
                let mut zero = states[0].clone();
                zero.state_index = 1;
                zero.public_helicity = 1;
                zero.spin_state = 50_001;
                states.push(zero);
                OnTheFlySourceAnchorV1::new(
                    source_slot,
                    external_label,
                    is_initial,
                    color_role,
                    is_fermionic,
                    pairing_source_contract_digest,
                    states,
                )
            })
            .collect::<RusticolResult<Vec<_>>>()
            .unwrap();
        OnTheFlyProcessSeedV1::new(
            process_digest,
            model_digest,
            template_catalog_digest,
            prepared_pack_digest,
            direct_catalog_digest,
            normalization_semantic_digest,
            normalization_convention,
            ExactComplexRational::ONE,
            anchors,
            external_permutation.into_vec(),
            coupling_order_policy,
            coupling_hierarchies.into_vec(),
            coupling_limits.into_vec(),
            pairing_classes.into_vec(),
        )
        .unwrap()
    }

    #[test]
    fn explicit_policy_is_returned_without_a_structural_sweep() {
        let templates = scalar_templates();
        let summary = templates.summary();
        let seed = scalar_adapter_test_seed(
            summary.compiled_model_digest,
            summary.catalog_digest,
            summary.prepared_kernel_pack_digest,
            digest(80),
        )
        .unwrap();
        let catalog = validate_seed_against_templates(&templates, &seed).unwrap();
        let grammar = prepare_on_the_fly_grammar_v1(&templates, &catalog, &seed).unwrap();
        let policy =
            resolve_on_the_fly_coupling_policy_from_grammar_v1(&templates, &seed, &grammar)
                .unwrap();

        assert_eq!(
            policy.effective_limits(),
            [Some(0)],
            "census={:?}",
            policy.census()
        );
        assert_eq!(policy.census(), OnTheFlyCouplingPolicyCensusV1::default());
    }

    #[test]
    fn minimal_scalar_policy_is_inferred_from_all_sources_and_physical_closure() {
        let templates = scalar_templates();
        let summary = templates.summary();
        let seed = with_policy(
            scalar_adapter_test_seed(
                summary.compiled_model_digest,
                summary.catalog_digest,
                summary.prepared_kernel_pack_digest,
                digest(81),
            )
            .unwrap(),
            OnTheFlyCouplingOrderPolicyV1::Minimal,
            vec![1],
            vec![None],
        );
        let catalog = validate_seed_against_templates(&templates, &seed).unwrap();
        let grammar = prepare_on_the_fly_grammar_v1(&templates, &catalog, &seed).unwrap();
        let policy =
            resolve_on_the_fly_coupling_policy_from_grammar_v1(&templates, &seed, &grammar)
                .unwrap();

        assert_eq!(policy.effective_limits(), [Some(0)]);
        assert_eq!(
            policy.census(),
            OnTheFlyCouplingPolicyCensusV1 {
                source_topology_count: 2,
                retained_topology_count: 2,
                transition_witness_attempt_count: 0,
                closure_witness_attempt_count: 2,
                viable_total_order_count: 1,
            }
        );
    }

    #[test]
    fn global_leading_order_is_not_promoted_by_a_selector_local_zero() {
        // Think of [2, 0] as the process-global QCD LO path.  The chosen
        // selector happens to vanish there and has only the higher weighted
        // [0, 1] electroweak path.  Resolving globally keeps QED=0, so the
        // selected trace is a structural zero instead of silently promoting
        // the process definition to its selector-local higher order.
        let global_totals = vec![
            vec![2_u32, 0].into_boxed_slice(),
            vec![0, 1].into_boxed_slice(),
        ];
        let global = minimal_envelope(&global_totals, &[1, 3]).unwrap().unwrap();
        let selector_only = minimal_envelope(&global_totals[1..], &[1, 3])
            .unwrap()
            .unwrap();

        assert_eq!(global, [Some(2), Some(0)]);
        assert_eq!(selector_only, [Some(0), Some(1)]);
        assert!(
            combined_coupling_orders(&[0, 0], &[0, 0], &[0, 1], &global)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn cached_minimal_policy_is_selector_global_when_one_helicity_is_zero() {
        let templates = scalar_templates();
        let summary = templates.summary();
        let seed = with_selector_local_zero(with_policy(
            scalar_adapter_test_seed(
                summary.compiled_model_digest,
                summary.catalog_digest,
                summary.prepared_kernel_pack_digest,
                digest(82),
            )
            .unwrap(),
            OnTheFlyCouplingOrderPolicyV1::Minimal,
            vec![1],
            vec![None],
        ));
        let catalog = validate_seed_against_templates(&templates, &seed).unwrap();
        let grammar = prepare_on_the_fly_grammar_v1(&templates, &catalog, &seed).unwrap();
        let policy =
            resolve_on_the_fly_coupling_policy_from_grammar_v1(&templates, &seed, &grammar)
                .unwrap();
        assert_eq!(policy.effective_limits(), [Some(0)]);

        let nonzero =
            DecodedLcQueryV1::new(&seed, vec![0, 1], &[0, 0], OnTheFlyLcSelectorV1::Singlet)
                .unwrap();
        let selector_zero =
            DecodedLcQueryV1::new(&seed, vec![0, 1], &[1, 1], OnTheFlyLcSelectorV1::Singlet)
                .unwrap();
        assert!(
            build_selected_lc_trace_impl(
                &templates,
                &seed,
                &grammar,
                policy.effective_limits(),
                &nonzero,
                false,
                false,
            )
            .unwrap()
            .is_some()
        );
        assert!(
            build_selected_lc_trace_impl(
                &templates,
                &seed,
                &grammar,
                policy.effective_limits(),
                &selector_zero,
                false,
                false,
            )
            .unwrap()
            .is_none()
        );
        assert_eq!(policy.effective_limits(), [Some(0)]);
    }

    #[test]
    fn equal_minimum_hierarchy_paths_form_a_componentwise_envelope() {
        let totals = vec![
            vec![2_u32, 0].into_boxed_slice(),
            vec![0, 1].into_boxed_slice(),
            vec![4, 0].into_boxed_slice(),
        ];
        assert_eq!(
            minimal_envelope(&totals, &[1, 2]).unwrap().unwrap(),
            [Some(2), Some(1)]
        );
    }

    #[test]
    fn pareto_front_discards_only_componentwise_higher_orders() {
        let mut orders = Vec::new();
        assert!(insert_pareto_order(&mut orders, vec![2, 0]));
        assert!(insert_pareto_order(&mut orders, vec![0, 2]));
        assert!(!insert_pareto_order(&mut orders, vec![3, 1]));
        assert!(insert_pareto_order(&mut orders, vec![1, 0]));
        assert_eq!(
            orders,
            [vec![0, 2].into_boxed_slice(), vec![1, 0].into_boxed_slice()]
        );
    }
}
