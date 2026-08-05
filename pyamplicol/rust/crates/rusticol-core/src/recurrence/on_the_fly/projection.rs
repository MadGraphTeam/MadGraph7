// SPDX-License-Identifier: 0BSD

//! Query-local counterpart of the established topology-replay color projection.
//!
//! The compact builder may discover multiple dynamic-color witnesses for one
//! color-erased current.  They are aliases only when their selected-query
//! contribution and closure domains form the same complete rectangular proof
//! used by the materialized builder.  This pass proves that condition before
//! retaining one representative; otherwise it leaves the trace unprojected.

use super::*;
use crate::recurrence::DynamicLCColorStateId;
use crate::recurrence::construct::current_key_with_dynamic_color;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ProjectedCurrentIdentity {
    color_erased_key: CurrentCoreKey,
    source_builder_id: Option<u32>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ProjectedContributionIdentity {
    destination_projection_id: u32,
    transition_template_id: u32,
    parent_projection_ids: [u32; 2],
    parent_state_template_ids: Box<[u32]>,
    parent_momenta: Box<[CanonicalMomentumLinearForm]>,
    result_state_template_id: u32,
    quantum_flow_witness_id: u32,
    runtime_coupling_binding_digest: SemanticDigest,
    output_projection_id: u32,
    exact_factor: ExactComplexRational,
}

#[derive(Clone, Debug)]
struct ProjectedContribution {
    identity: ProjectedContributionIdentity,
    representative_key: ContributionKey,
    destination_builder_ids: BTreeSet<u32>,
    builder_parent_tuples: BTreeSet<[u32; 2]>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ProjectedClosureIdentity {
    closure_template_id: u32,
    quantum_flow_template_id: Option<u32>,
    parent_projection_ids: [u32; 2],
    exact_factor: ExactComplexRational,
    component_coefficients: Box<[ExactComplexRational]>,
}

#[derive(Clone, Debug)]
struct ProjectedClosure {
    identity: ProjectedClosureIdentity,
    representative_order: ProjectedClosureRepresentativeOrder,
    representative_key: PendingClosureKey,
    builder_parent_tuples: BTreeSet<[u32; 2]>,
    pairing_lineages: Vec<PendingPairingLineage>,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ProjectedClosureRepresentativeOrder {
    closure_template_id: u32,
    quantum_flow_template_id: Option<u32>,
    parent_keys: Box<[CurrentCoreKey]>,
    color_witness_term_id: LCColorWitnessTermId,
}

fn closure_representative_order(
    pending: &[PendingCurrent],
    closure: &PendingClosure,
) -> RusticolResult<ProjectedClosureRepresentativeOrder> {
    let parent_keys = closure
        .key
        .parent_current_ids
        .iter()
        .map(|id| {
            pending
                .get(*id as usize)
                .map(|current| current.key.clone())
                .ok_or_else(|| integrity("projected closure representative parent is absent"))
        })
        .collect::<RusticolResult<Vec<_>>>()?
        .into_boxed_slice();
    Ok(ProjectedClosureRepresentativeOrder {
        closure_template_id: closure.key.closure_template_id,
        quantum_flow_template_id: closure.key.quantum_flow_template_id,
        parent_keys,
        color_witness_term_id: closure.key.color_witness_term_id,
    })
}

pub(super) struct QueryLocalColorProjection {
    pub(super) currents: Vec<PendingCurrent>,
    pub(super) closures: Vec<PendingClosure>,
}

fn parent_rectangle_is_complete(
    parent_projection_ids: [u32; 2],
    builder_parent_tuples: &BTreeSet<[u32; 2]>,
    old_to_projection: &BTreeMap<u32, u32>,
    projection_members: &[Vec<u32>],
) -> RusticolResult<bool> {
    let expected = parent_projection_ids
        .iter()
        .try_fold(1usize, |count, projection_id| {
            count
                .checked_mul(
                    projection_members
                        .get(*projection_id as usize)
                        .ok_or_else(|| integrity("projected parent class is absent"))?
                        .len(),
                )
                .ok_or_else(|| invalid("projected parent-product count exceeds usize"))
        })?;
    if expected != builder_parent_tuples.len() {
        return Ok(false);
    }
    Ok(builder_parent_tuples.iter().all(|tuple| {
        tuple
            .iter()
            .zip(parent_projection_ids.iter())
            .all(|(builder_id, projection_id)| {
                old_to_projection.get(builder_id) == Some(projection_id)
            })
    }))
}

fn contribution_identity(
    destination_projection_id: u32,
    key: &PendingContributionKey,
    factor: ExactComplexRational,
    old_to_projection: &BTreeMap<u32, u32>,
) -> RusticolResult<ProjectedContributionIdentity> {
    let parent_projection_ids = key
        .parent_current_ids
        .map(|id| {
            old_to_projection
                .get(&id)
                .copied()
                .ok_or_else(|| integrity("projected contribution has a dead parent"))
        })
        .into_iter()
        .collect::<RusticolResult<Vec<_>>>()?
        .try_into()
        .map_err(|_| integrity("projected contribution is not binary"))?;
    Ok(ProjectedContributionIdentity {
        destination_projection_id,
        transition_template_id: key.key.transition_template_id(),
        parent_projection_ids,
        parent_state_template_ids: key.key.parent_state_template_ids().into(),
        parent_momenta: key.key.parent_momenta().into(),
        result_state_template_id: key.key.result_state_template_id(),
        quantum_flow_witness_id: key.key.quantum_flow_witness_id(),
        runtime_coupling_binding_digest: key.key.runtime_coupling_binding_digest(),
        output_projection_id: key.key.output_projection_id(),
        exact_factor: factor,
    })
}

pub(super) fn project_query_local_color_aliases(
    pending: &[PendingCurrent],
    pending_closures: &[PendingClosure],
    live: &BTreeSet<u32>,
) -> RusticolResult<Option<QueryLocalColorProjection>> {
    let canonical_color_id = DynamicLCColorStateId::from_interner(0);
    let mut identity_to_projection = BTreeMap::<ProjectedCurrentIdentity, u32>::new();
    let mut old_to_projection = BTreeMap::<u32, u32>::new();
    let mut projection_members = Vec::<Vec<u32>>::new();
    for old_id in live.iter().copied() {
        let current = pending
            .get(old_id as usize)
            .ok_or_else(|| integrity("live query-local current is absent"))?;
        let identity = ProjectedCurrentIdentity {
            color_erased_key: current_key_with_dynamic_color(&current.key, canonical_color_id)?,
            source_builder_id: (current.key.node_kind() == RecurrenceNodeKind::Source)
                .then_some(old_id),
        };
        let projection_id = if let Some(id) = identity_to_projection.get(&identity).copied() {
            id
        } else {
            let id = checked_u32(projection_members.len(), "query-local projection class")?;
            projection_members.push(Vec::new());
            identity_to_projection.insert(identity, id);
            id
        };
        projection_members[projection_id as usize].push(old_id);
        old_to_projection.insert(old_id, projection_id);
    }
    if projection_members.iter().all(|members| members.len() == 1) {
        return Ok(None);
    }

    let mut contributions = BTreeMap::<ProjectedContributionIdentity, ProjectedContribution>::new();
    for (old_id, projection_id) in &old_to_projection {
        for (key, factor) in &pending[*old_id as usize].contributions {
            if factor.is_zero() {
                continue;
            }
            let identity = contribution_identity(*projection_id, key, *factor, &old_to_projection)?;
            match contributions.entry(identity.clone()) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(ProjectedContribution {
                        identity,
                        representative_key: key.key.clone(),
                        destination_builder_ids: BTreeSet::from([*old_id]),
                        builder_parent_tuples: BTreeSet::from([key.parent_current_ids]),
                    });
                }
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    let row = entry.get_mut();
                    row.destination_builder_ids.insert(*old_id);
                    if !row.builder_parent_tuples.insert(key.parent_current_ids) {
                        return Ok(None);
                    }
                }
            }
        }
    }
    for row in contributions.values() {
        if !parent_rectangle_is_complete(
            row.identity.parent_projection_ids,
            &row.builder_parent_tuples,
            &old_to_projection,
            &projection_members,
        )? {
            return Ok(None);
        }
    }

    let mut closures = BTreeMap::<ProjectedClosureIdentity, ProjectedClosure>::new();
    for closure in pending_closures {
        let parent_projection_ids = closure
            .key
            .parent_current_ids
            .map(|id| {
                old_to_projection
                    .get(&id)
                    .copied()
                    .ok_or_else(|| integrity("projected closure has a dead parent"))
            })
            .into_iter()
            .collect::<RusticolResult<Vec<_>>>()?
            .try_into()
            .map_err(|_| integrity("projected closure is not binary"))?;
        let identity = ProjectedClosureIdentity {
            closure_template_id: closure.key.closure_template_id,
            quantum_flow_template_id: closure.key.quantum_flow_template_id,
            parent_projection_ids,
            exact_factor: closure.factor,
            component_coefficients: closure.component_coefficients.clone(),
        };
        let representative_order = closure_representative_order(pending, closure)?;
        match closures.entry(identity.clone()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(ProjectedClosure {
                    identity,
                    representative_order,
                    representative_key: closure.key.clone(),
                    builder_parent_tuples: BTreeSet::from([closure.key.parent_current_ids]),
                    pairing_lineages: closure.pairing_lineages.clone(),
                });
            }
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                if !entry
                    .get_mut()
                    .builder_parent_tuples
                    .insert(closure.key.parent_current_ids)
                {
                    return Ok(None);
                }
                // Pairing lineage is proof owned by the canonical root-bearing
                // representative, not an additive property of color aliases.
                // Keep the same semantic/current ordering regardless of input
                // enumeration and never union a later alias's Wick lineage.
                if representative_order < entry.get().representative_order {
                    let projected = entry.get_mut();
                    projected.representative_order = representative_order;
                    projected.representative_key = closure.key.clone();
                    projected.pairing_lineages = closure.pairing_lineages.clone();
                }
            }
        }
    }
    for row in closures.values() {
        if !parent_rectangle_is_complete(
            row.identity.parent_projection_ids,
            &row.builder_parent_tuples,
            &old_to_projection,
            &projection_members,
        )? {
            return Ok(None);
        }
    }

    let mut projected_currents = Vec::with_capacity(projection_members.len());
    for (projection_index, members) in projection_members.iter().enumerate() {
        let projection_id = checked_u32(projection_index, "projected query-local current ID")?;
        let representative_id = *members
            .first()
            .ok_or_else(|| integrity("query-local projection class is empty"))?;
        let representative = &pending[representative_id as usize];
        let mut projected = representative.clone();
        projected.contributions.clear();
        projected.pairing_lineages.clear();
        for member in members {
            extend_pairing_lineages(
                &mut projected.pairing_lineages,
                &pending[*member as usize].pairing_lineages,
            )?;
        }
        for row in contributions
            .values()
            .filter(|row| row.identity.destination_projection_id == projection_id)
        {
            if !row
                .destination_builder_ids
                .iter()
                .all(|builder_id| old_to_projection.get(builder_id) == Some(&projection_id))
            {
                return Err(integrity(
                    "projected contribution destinations cross projection classes",
                ));
            }
            let identity = &row.identity;
            let key = ContributionKey::new(
                identity.transition_template_id,
                identity.parent_projection_ids.to_vec(),
                identity.parent_state_template_ids.to_vec(),
                identity.parent_momenta.to_vec(),
                identity.result_state_template_id,
                identity.quantum_flow_witness_id,
                row.representative_key.color_witness_term_id(),
                identity.runtime_coupling_binding_digest,
                identity.output_projection_id,
            )?;
            projected.contributions.insert(
                PendingContributionKey {
                    parent_current_ids: identity.parent_projection_ids,
                    key,
                },
                identity.exact_factor,
            );
        }
        projected_currents.push(projected);
    }

    let projected_closures = closures
        .into_values()
        .map(|row| {
            Ok(PendingClosure {
                key: PendingClosureKey {
                    closure_template_id: row.identity.closure_template_id,
                    quantum_flow_template_id: row.identity.quantum_flow_template_id,
                    parent_current_ids: row.identity.parent_projection_ids,
                    color_witness_term_id: row.representative_key.color_witness_term_id,
                },
                factor: row.identity.exact_factor,
                component_coefficients: row.identity.component_coefficients,
                pairing_lineages: row.pairing_lineages,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    Ok(Some(QueryLocalColorProjection {
        currents: projected_currents,
        closures: projected_closures,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).unwrap()
    }

    fn current(state: u32, color: u32, support: u32) -> PendingCurrent {
        PendingCurrent {
            key: CurrentCoreKey::new(
                digest(1),
                RecurrenceNodeKind::Current,
                state,
                DynamicLCColorStateId::from_interner(color),
                vec![support],
                CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                    source_slot: support,
                    coefficient: 1,
                }])
                .unwrap(),
                CurrentHelicityIdentity::topology_replay(
                    0,
                    vec![SourceStateAssignment::new(support, 0)],
                )
                .unwrap(),
                vec![state as i32],
                0,
                vec![0],
                CurrentSourceBinding::None,
                None,
            )
            .unwrap(),
            source_factor: None,
            contributions: BTreeMap::new(),
            pairing_lineages: vec![PendingPairingLineage {
                completed_pairs: Box::new([]),
                unmatched_endpoint: None,
            }],
            stage: 0,
        }
    }

    fn closure(alias_parent: u32) -> PendingClosure {
        closure_with_lineage(alias_parent, Box::new([]))
    }

    fn closure_with_lineage(alias_parent: u32, completed_pairs: Box<[[u32; 2]]>) -> PendingClosure {
        PendingClosure {
            key: PendingClosureKey {
                closure_template_id: 7,
                quantum_flow_template_id: None,
                parent_current_ids: [0, alias_parent],
                color_witness_term_id: LCColorWitnessTermId::new(9, 0),
            },
            factor: ExactComplexRational::ONE,
            component_coefficients: vec![ExactComplexRational::ONE].into_boxed_slice(),
            pairing_lineages: vec![PendingPairingLineage {
                completed_pairs,
                unmatched_endpoint: None,
            }],
        }
    }

    #[test]
    fn query_local_projection_requires_a_complete_parent_rectangle() {
        let pending = vec![current(1, 0, 0), current(2, 1, 1), current(2, 2, 1)];
        let live = BTreeSet::from([0, 1, 2]);
        assert!(
            project_query_local_color_aliases(&pending, &[closure(1)], &live)
                .unwrap()
                .is_none(),
            "one of two alias parents is not a complete rectangular proof",
        );

        let projected =
            project_query_local_color_aliases(&pending, &[closure(1), closure(2)], &live)
                .unwrap()
                .expect("complete alias rectangle projects");
        assert_eq!(projected.currents.len(), 2);
        assert_eq!(projected.closures.len(), 1);
        assert_eq!(projected.closures[0].key.parent_current_ids, [0, 1]);
    }

    #[test]
    fn query_local_projection_ignores_poison_outside_the_selected_live_set() {
        let base = vec![current(1, 0, 0), current(2, 1, 1), current(2, 2, 1)];
        let live = BTreeSet::from([0, 1, 2]);
        let expected = project_query_local_color_aliases(&base, &[closure(1), closure(2)], &live)
            .unwrap()
            .unwrap();
        let mut poisoned = base;
        poisoned.push(current(99, u32::MAX, 99));
        let observed =
            project_query_local_color_aliases(&poisoned, &[closure(1), closure(2)], &live)
                .unwrap()
                .unwrap();
        assert_eq!(observed.currents.len(), expected.currents.len());
        assert_eq!(observed.closures.len(), expected.closures.len());
        assert!(
            observed
                .currents
                .iter()
                .all(|row| row.key.current_state_template_id() != 99)
        );
    }

    #[test]
    fn projected_pairing_owner_uses_semantic_representative_not_input_order() {
        let pending = vec![current(1, 0, 0), current(2, 2, 1), current(2, 1, 1)];
        let live = BTreeSet::from([0, 1, 2]);
        let high = closure_with_lineage(1, vec![[0, 5]].into_boxed_slice());
        let low = closure_with_lineage(2, vec![[0, 3]].into_boxed_slice());
        for closures in [
            vec![high.clone(), low.clone()],
            vec![low.clone(), high.clone()],
        ] {
            let projected = project_query_local_color_aliases(&pending, &closures, &live)
                .unwrap()
                .expect("complete alias rectangle projects");
            assert_eq!(
                projected.closures[0].pairing_lineages,
                vec![PendingPairingLineage {
                    completed_pairs: vec![[0, 3]].into_boxed_slice(),
                    unmatched_endpoint: None,
                }]
            );
        }
    }
}
