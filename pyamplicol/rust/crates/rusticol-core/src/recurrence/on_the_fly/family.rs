// SPDX-License-Identifier: 0BSD

//! Feature-only structural union census for requested LC query families.
//!
//! This prototype never emits or executes a plan.  It proves which rows from
//! independently constructed query-local traces could share one runtime arena
//! and one grouped invocation schedule.  Release/default builds do not retain
//! the cold contribution identities consumed here.

use super::trace::{OnTheFlyTraceContributionProofRowV1, OnTheFlyTraceOperationV1};
use super::*;
use crate::recurrence::PreparedDirectExecutorCatalog;
use crate::recurrence::construct::current_key_with_dynamic_color;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct FamilySourceIdentity {
    executor: OnTheFlyExecutorKeyV1,
    source_slot: u32,
    momentum: CanonicalMomentumLinearForm,
    source_template_or_dispatch_domain: u32,
    spin_state_class: i32,
    exact_factor: ExactComplexRational,
    selector_domain_id: u32,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct FamilyContributionIdentity {
    executor: OnTheFlyExecutorKeyV1,
    key: ContributionKey,
    exact_factor: ExactComplexRational,
    selector_domain_id: u32,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct FamilyFinalizationIdentity {
    executor: OnTheFlyExecutorKeyV1,
    momentum: CanonicalMomentumLinearForm,
    component_count: u16,
    exact_factor: ExactComplexRational,
    selector_domain_id: u32,
    flags: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FamilyCurrentDefinition {
    component_count: u32,
    source: Option<FamilySourceIdentity>,
    contributions: BTreeSet<FamilyContributionIdentity>,
    finalization: Option<FamilyFinalizationIdentity>,
}

#[derive(Clone, Copy)]
struct QueryFamilyTraceInput<'a> {
    trace: &'a OnTheFlyStructuralTraceV1,
    projection: OnTheFlyProjectionProbeV1,
}

impl FamilyCurrentDefinition {
    const fn new(component_count: u32) -> Self {
        Self {
            component_count,
            source: None,
            contributions: BTreeSet::new(),
            finalization: None,
        }
    }
}

/// Cold projection of both today's serialized-query execution and a
/// proof-identical query-family union.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct OnTheFlyQueryFamilyCensusV1 {
    pub query_count: u32,
    pub source_frame_partition_count: u32,
    pub projection_applied_query_count: u32,
    pub projection_pre_current_count: u32,
    pub projection_pre_contribution_count: u32,
    pub projection_pre_closure_count: u32,
    pub projection_post_current_count: u32,
    pub projection_post_contribution_count: u32,
    pub projection_post_closure_count: u32,
    pub dynamic_current_occurrence_count: u32,
    pub dynamic_current_component_occurrence_count: u32,
    pub dynamic_source_rows: u32,
    pub dynamic_contribution_rows: u32,
    pub dynamic_finalization_rows: u32,
    pub dynamic_closure_rows: u32,
    pub dynamic_source_calls: u32,
    pub dynamic_contribution_calls: u32,
    pub dynamic_finalization_calls: u32,
    pub dynamic_closure_calls: u32,
    pub union_unique_current_count: u32,
    pub union_unique_current_component_count: u32,
    pub union_source_rows: u32,
    pub union_contribution_rows: u32,
    pub union_finalization_rows: u32,
    pub union_closure_rows: u32,
    pub union_amplitude_destination_count: u32,
    pub union_source_executor_call_groups: u32,
    pub union_contribution_executor_call_groups: u32,
    pub union_finalization_executor_call_groups: u32,
    pub union_closure_executor_call_groups: u32,
}

fn add(value: &mut u32, increment: u32, label: &str) -> RusticolResult<()> {
    *value = value
        .checked_add(increment)
        .ok_or_else(|| invalid(format!("{label} exceeds u32")))?;
    Ok(())
}

fn trace_factor(
    trace: &OnTheFlyStructuralTraceV1,
    factor_id: u32,
    label: &str,
) -> RusticolResult<ExactComplexRational> {
    trace
        .exact_factors
        .get(factor_id as usize)
        .copied()
        .ok_or_else(|| integrity(format!("{label} exact factor is absent")))
}

fn trace_momentum(
    trace: &OnTheFlyStructuralTraceV1,
    momentum_id: u32,
    label: &str,
) -> RusticolResult<CanonicalMomentumLinearForm> {
    trace
        .momentum_forms
        .get(momentum_id as usize)
        .cloned()
        .ok_or_else(|| integrity(format!("{label} momentum form is absent")))
}

fn remapped_contribution_key(
    row: &OnTheFlyTraceContributionProofRowV1,
    local_to_union: &[u32],
) -> RusticolResult<ContributionKey> {
    if row.key.parent_value_class_ids() != row.parent_current_ids {
        return Err(integrity(
            "cold contribution identity does not retain its local parent IDs",
        ));
    }
    let parent_ids = row
        .parent_current_ids
        .iter()
        .map(|id| {
            local_to_union
                .get(*id as usize)
                .copied()
                .ok_or_else(|| integrity("contribution parent current is absent"))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    ContributionKey::new(
        row.key.transition_template_id(),
        parent_ids,
        row.key.parent_state_template_ids().to_vec(),
        row.key.parent_momenta().to_vec(),
        row.key.result_state_template_id(),
        row.key.quantum_flow_witness_id(),
        row.key.color_witness_term_id(),
        row.key.runtime_coupling_binding_digest(),
        row.key.output_projection_id(),
    )
}

fn current_id_by_component_base(
    bases: &BTreeMap<u32, u32>,
    base: u32,
    label: &str,
) -> RusticolResult<u32> {
    bases
        .get(&base)
        .copied()
        .ok_or_else(|| integrity(format!("{label} current component base is absent")))
}

fn prepared_executor_id(
    catalog: &PreparedDirectExecutorCatalog,
    key: OnTheFlyExecutorKeyV1,
) -> RusticolResult<u32> {
    if key.direct_catalog_digest() != catalog.direct_template_catalog_digest() {
        return Err(integrity(
            "query-family executor key belongs to a different prepared catalog",
        ));
    }
    match key.evaluator_binding_id() {
        Some(binding_id) => catalog.resolve_evaluator(key.role(), binding_id),
        None => catalog.resolve_identity_finalizer(),
    }
}

/// Analyze the exact traces requested by one existing selector-family call.
///
/// Equal normalized [`CurrentCoreKey`] values are merged only when their
/// complete source/contribution/finalization definitions agree.  Conflicting
/// duplicates fail closed.  Closures remain distinct per requested query so a
/// future executable union cannot lose an amplitude destination.
fn query_family_census_from_traces(
    direct_catalog: &PreparedDirectExecutorCatalog,
    selected: &[QueryFamilyTraceInput<'_>],
) -> RusticolResult<OnTheFlyQueryFamilyCensusV1> {
    let first = selected
        .first()
        .ok_or_else(|| invalid("query-family census requires at least one query"))?;
    let source_count = first.trace.layout.source_count;
    let mut seen_queries = BTreeSet::new();
    let mut seed_partitions = BTreeSet::new();
    let mut colors = DynamicLCColorStateInterner::default();
    let mut union_ids = BTreeMap::<(SemanticDigest, CurrentCoreKey), u32>::new();
    let mut definitions =
        BTreeMap::<(SemanticDigest, CurrentCoreKey), FamilyCurrentDefinition>::new();
    let mut source_groups = BTreeSet::new();
    let mut contribution_groups = BTreeSet::new();
    let mut finalization_groups = BTreeSet::new();
    let mut closure_groups = BTreeSet::new();
    let mut census = OnTheFlyQueryFamilyCensusV1 {
        query_count: checked_u32(selected.len(), "query-family query count")?,
        source_frame_partition_count: 0,
        projection_applied_query_count: 0,
        projection_pre_current_count: 0,
        projection_pre_contribution_count: 0,
        projection_pre_closure_count: 0,
        projection_post_current_count: 0,
        projection_post_contribution_count: 0,
        projection_post_closure_count: 0,
        dynamic_current_occurrence_count: 0,
        dynamic_current_component_occurrence_count: 0,
        dynamic_source_rows: 0,
        dynamic_contribution_rows: 0,
        dynamic_finalization_rows: 0,
        dynamic_closure_rows: 0,
        dynamic_source_calls: 0,
        dynamic_contribution_calls: 0,
        dynamic_finalization_calls: 0,
        dynamic_closure_calls: 0,
        union_unique_current_count: 0,
        union_unique_current_component_count: 0,
        union_source_rows: 0,
        union_contribution_rows: 0,
        union_finalization_rows: 0,
        union_closure_rows: 0,
        union_amplitude_destination_count: checked_u32(
            selected.len(),
            "query-family amplitude destination count",
        )?,
        union_source_executor_call_groups: 0,
        union_contribution_executor_call_groups: 0,
        union_finalization_executor_call_groups: 0,
        union_closure_executor_call_groups: 0,
    };

    for selected_query in selected {
        let trace = selected_query.trace;
        if trace.layout.source_count != source_count {
            return Err(integrity(
                "query-family traces do not share one source-slot arity",
            ));
        }
        seed_partitions.insert(trace.seed_digest);
        if !seen_queries.insert(trace.query_digest) {
            return Err(invalid("query-family repeats a selected query"));
        }
        if trace.layout.amplitude_component_count != 1 {
            return Err(integrity(
                "query-local trace does not expose exactly one amplitude destination",
            ));
        }
        if selected_query.projection.applied {
            add(
                &mut census.projection_applied_query_count,
                1,
                "projection-applied query count",
            )?;
        }
        for (target, value, label) in [
            (
                &mut census.projection_pre_current_count,
                selected_query.projection.pre[0],
                "projection pre-current count",
            ),
            (
                &mut census.projection_pre_contribution_count,
                selected_query.projection.pre[1],
                "projection pre-contribution count",
            ),
            (
                &mut census.projection_pre_closure_count,
                selected_query.projection.pre[2],
                "projection pre-closure count",
            ),
            (
                &mut census.projection_post_current_count,
                selected_query.projection.post[0],
                "projection post-current count",
            ),
            (
                &mut census.projection_post_contribution_count,
                selected_query.projection.post[1],
                "projection post-contribution count",
            ),
            (
                &mut census.projection_post_closure_count,
                selected_query.projection.post[2],
                "projection post-closure count",
            ),
        ] {
            add(target, value, label)?;
        }
        if selected_query.projection.post
            != [
                trace.proof.current_count(),
                trace.proof.contribution_count(),
                trace.proof.closure_count(),
            ]
        {
            return Err(integrity(
                "query-local projection proof does not describe its retained trace",
            ));
        }
        let current_count = trace.current_keys.len();
        if current_count != trace.current_colors.len()
            || current_count != trace.current_component_ranges.len()
            || current_count != trace.proof.current_count() as usize
        {
            return Err(integrity(
                "query-local current identity columns have inconsistent lengths",
            ));
        }
        add(
            &mut census.dynamic_current_occurrence_count,
            checked_u32(current_count, "dynamic current occurrence count")?,
            "dynamic current occurrence count",
        )?;

        let mut local_to_union = Vec::new();
        local_to_union
            .try_reserve_exact(current_count)
            .map_err(|error| {
                invalid(format!(
                    "query-family current map allocation failed: {error}"
                ))
            })?;
        let mut normalized_keys = Vec::new();
        normalized_keys
            .try_reserve_exact(current_count)
            .map_err(|error| {
                invalid(format!(
                    "query-family current key allocation failed: {error}"
                ))
            })?;
        let mut bases = BTreeMap::new();
        for (local_id, ((key, color), [base, count])) in trace
            .current_keys
            .iter()
            .zip(trace.current_colors.iter())
            .zip(trace.current_component_ranges.iter().copied())
            .enumerate()
        {
            if count == 0 || bases.insert(base, local_id as u32).is_some() {
                return Err(integrity(
                    "query-local current component ranges are empty or ambiguous",
                ));
            }
            add(
                &mut census.dynamic_current_component_occurrence_count,
                count,
                "dynamic current component occurrence count",
            )?;
            let color_id = colors.intern(color.clone())?;
            let normalized = current_key_with_dynamic_color(key, color_id)?;
            let family_key = (trace.seed_digest, normalized);
            let next = checked_u32(union_ids.len(), "query-family union current count")?;
            let union_id = *union_ids.entry(family_key.clone()).or_insert(next);
            local_to_union.push(union_id);
            normalized_keys.push(family_key);
        }

        let mut local_definitions = trace
            .current_component_ranges
            .iter()
            .map(|range| FamilyCurrentDefinition::new(range[1]))
            .collect::<Vec<_>>();
        let mut contribution_proofs = trace.contribution_proof_rows.iter();
        for operation in trace.operations.iter() {
            match operation {
                OnTheFlyTraceOperationV1::Source { key, row } => {
                    let current_id = current_id_by_component_base(
                        &bases,
                        row.destination_component_base,
                        "source destination",
                    )?;
                    let definition = &mut local_definitions[current_id as usize];
                    let identity = FamilySourceIdentity {
                        executor: *key,
                        source_slot: row.source_slot,
                        momentum: trace_momentum(trace, row.momentum_form_id, "source")?,
                        source_template_or_dispatch_domain: row.source_template_or_dispatch_domain,
                        spin_state_class: row.spin_state_class,
                        exact_factor: trace_factor(trace, row.exact_factor_id, "source")?,
                        selector_domain_id: row.selector_domain_id,
                    };
                    if definition.source.replace(identity).is_some() {
                        return Err(integrity("query-local current repeats its source row"));
                    }
                    add(&mut census.dynamic_source_rows, 1, "dynamic source rows")?;
                }
                OnTheFlyTraceOperationV1::Contribution { key: executor, row } => {
                    let proof = contribution_proofs.next().ok_or_else(|| {
                        integrity("query-local contribution has no cold semantic identity")
                    })?;
                    let result = current_id_by_component_base(
                        &bases,
                        row.destination_component_base,
                        "contribution destination",
                    )?;
                    if result != proof.result_current_id
                        || trace_factor(trace, row.exact_factor_id, "contribution")?
                            != proof.exact_factor
                    {
                        return Err(integrity(
                            "query-local contribution row disagrees with its cold identity",
                        ));
                    }
                    let mut physical_parents = [
                        current_id_by_component_base(
                            &bases,
                            row.parent0_component_base,
                            "contribution parent 0",
                        )?,
                        current_id_by_component_base(
                            &bases,
                            row.parent1_component_base_or_sentinel,
                            "contribution parent 1",
                        )?,
                    ];
                    let mut semantic_parents = proof.parent_current_ids;
                    physical_parents.sort_unstable();
                    semantic_parents.sort_unstable();
                    if physical_parents != semantic_parents
                        || !matches!(
                            row.flags,
                            0 | DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION
                        )
                    {
                        return Err(integrity(
                            "query-local contribution physical row is not its semantic binary row",
                        ));
                    }
                    let identity = FamilyContributionIdentity {
                        executor: *executor,
                        key: remapped_contribution_key(proof, &local_to_union)?,
                        exact_factor: proof.exact_factor,
                        selector_domain_id: row.selector_domain_id,
                    };
                    if !local_definitions[result as usize]
                        .contributions
                        .insert(identity)
                    {
                        return Err(integrity(
                            "query-local current repeats an exact contribution row",
                        ));
                    }
                    add(
                        &mut census.dynamic_contribution_rows,
                        1,
                        "dynamic contribution rows",
                    )?;
                }
                OnTheFlyTraceOperationV1::Finalization { key, row } => {
                    let current_id = current_id_by_component_base(
                        &bases,
                        row.component_base,
                        "finalization destination",
                    )?;
                    let identity = FamilyFinalizationIdentity {
                        executor: *key,
                        momentum: trace_momentum(trace, row.momentum_form_id, "finalization")?,
                        component_count: row.component_count,
                        exact_factor: trace_factor(trace, row.exact_factor_id, "finalization")?,
                        selector_domain_id: row.selector_domain_id,
                        flags: row.flags,
                    };
                    let definition = &mut local_definitions[current_id as usize];
                    if u32::from(row.component_count) != definition.component_count
                        || definition.finalization.replace(identity).is_some()
                    {
                        return Err(integrity(
                            "query-local current has an invalid finalization row",
                        ));
                    }
                    add(
                        &mut census.dynamic_finalization_rows,
                        1,
                        "dynamic finalization rows",
                    )?;
                }
                OnTheFlyTraceOperationV1::Closure { key, row } => {
                    if row.amplitude_destination_id != 0 {
                        return Err(integrity(
                            "query-local closure does not target its sole amplitude",
                        ));
                    }
                    closure_groups.insert((
                        trace.seed_digest,
                        source_count,
                        prepared_executor_id(direct_catalog, *key)?,
                    ));
                    add(&mut census.dynamic_closure_rows, 1, "dynamic closure rows")?;
                }
            }
        }
        if contribution_proofs.next().is_some() {
            return Err(integrity(
                "query-local cold contribution identities outnumber direct rows",
            ));
        }

        for ((family_key, definition), union_id) in normalized_keys
            .into_iter()
            .zip(local_definitions)
            .zip(local_to_union)
        {
            let key = &family_key.1;
            match key.node_kind() {
                RecurrenceNodeKind::Source
                    if definition.source.is_none()
                        || !definition.contributions.is_empty()
                        || definition.finalization.is_some() =>
                {
                    return Err(integrity(
                        "query-family source current has a non-source definition",
                    ));
                }
                RecurrenceNodeKind::Current
                    if definition.source.is_some() || definition.contributions.is_empty() =>
                {
                    return Err(integrity(
                        "query-family non-source current has no contribution definition",
                    ));
                }
                _ => {}
            }
            match definitions.entry(family_key) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(definition);
                }
                std::collections::btree_map::Entry::Occupied(entry)
                    if entry.get() == &definition => {}
                std::collections::btree_map::Entry::Occupied(_) => {
                    return Err(integrity(format!(
                        "query-family semantic current {union_id} has conflicting definitions"
                    )));
                }
            }
        }
    }

    census.source_frame_partition_count =
        checked_u32(seed_partitions.len(), "source-frame partition count")?;
    for ((seed_digest, key), definition) in &definitions {
        add(
            &mut census.union_unique_current_component_count,
            definition.component_count,
            "union current component count",
        )?;
        let stage = u32::try_from(
            key.support_source_slots()
                .len()
                .checked_sub(1)
                .ok_or_else(|| integrity("query-family current has empty source support"))?,
        )
        .map_err(|_| invalid("query-family current stage exceeds u32"))?;
        if let Some(source) = &definition.source {
            add(&mut census.union_source_rows, 1, "union source rows")?;
            source_groups.insert((
                *seed_digest,
                stage,
                prepared_executor_id(direct_catalog, source.executor)?,
                source.selector_domain_id,
            ));
        }
        for contribution in &definition.contributions {
            add(
                &mut census.union_contribution_rows,
                1,
                "union contribution rows",
            )?;
            contribution_groups.insert((
                *seed_digest,
                stage,
                prepared_executor_id(direct_catalog, contribution.executor)?,
                contribution.selector_domain_id,
            ));
        }
        if let Some(finalization) = &definition.finalization {
            add(
                &mut census.union_finalization_rows,
                1,
                "union finalization rows",
            )?;
            finalization_groups.insert((
                *seed_digest,
                stage,
                prepared_executor_id(direct_catalog, finalization.executor)?,
                finalization.selector_domain_id,
            ));
        }
    }
    census.union_unique_current_count = checked_u32(definitions.len(), "union current count")?;
    census.union_closure_rows = census.dynamic_closure_rows;
    census.union_source_executor_call_groups =
        checked_u32(source_groups.len(), "union source executor groups")?;
    census.union_contribution_executor_call_groups = checked_u32(
        contribution_groups.len(),
        "union contribution executor groups",
    )?;
    census.union_finalization_executor_call_groups = checked_u32(
        finalization_groups.len(),
        "union finalization executor groups",
    )?;
    census.union_closure_executor_call_groups =
        checked_u32(closure_groups.len(), "union closure executor groups")?;
    census.dynamic_source_calls = census.dynamic_source_rows;
    census.dynamic_contribution_calls = census.dynamic_contribution_rows;
    census.dynamic_finalization_calls = census.dynamic_finalization_rows;
    census.dynamic_closure_calls = census.dynamic_closure_rows;

    if census.projection_post_current_count != census.dynamic_current_occurrence_count
        || census.projection_post_contribution_count != census.dynamic_contribution_rows
        || census.projection_post_closure_count != census.dynamic_closure_rows
        || census.union_unique_current_count > census.dynamic_current_occurrence_count
        || census.union_unique_current_component_count
            > census.dynamic_current_component_occurrence_count
        || census.union_source_rows > census.dynamic_source_rows
        || census.union_contribution_rows > census.dynamic_contribution_rows
        || census.union_finalization_rows > census.dynamic_finalization_rows
    {
        return Err(integrity(
            "query-family union census is inconsistent with its local traces",
        ));
    }
    Ok(census)
}

/// Build exactly the requested LC queries with the existing query-local
/// machinery, then compute their cold structural union census.
#[doc(hidden)]
pub fn on_the_fly_query_family_census_v1(
    authenticated: &crate::recurrence::AuthenticatedRecurrenceBuilderInput,
    direct_catalog: &PreparedDirectExecutorCatalog,
    queries: &[(u32, Vec<i32>)],
    enable_projection: bool,
) -> RusticolResult<OnTheFlyQueryFamilyCensusV1> {
    let selected = queries
        .iter()
        .map(|(flow, helicities)| {
            super::test_support::build_on_the_fly_selected_trace_v1(
                authenticated,
                direct_catalog.direct_template_catalog_digest(),
                *flow,
                helicities,
                enable_projection,
            )
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    for selected_query in &selected {
        if selected_query.seed.semantic_digest() != selected_query.trace.seed_digest
            || selected_query.query.semantic_digest() != selected_query.trace.query_digest
        {
            return Err(integrity(
                "selected query-family trace is detached from its compact inputs",
            ));
        }
    }
    let traces = selected
        .iter()
        .map(|selected_query| QueryFamilyTraceInput {
            trace: &selected_query.trace,
            projection: selected_query.projection,
        })
        .collect::<Vec<_>>();
    query_family_census_from_traces(direct_catalog, &traces)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::{
        ExactRational, PreparedDirectExecutorBinding, PreparedDirectExecutorCatalog,
    };

    fn factor(numerator: i128) -> ExactComplexRational {
        ExactComplexRational::new(
            ExactRational::new(numerator, 1).unwrap(),
            ExactRational::ZERO,
        )
    }

    fn direct_catalog() -> PreparedDirectExecutorCatalog {
        let digest = SemanticDigest::new([0x71; 32]).unwrap();
        PreparedDirectExecutorCatalog::new(
            digest,
            vec![
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Source, 0, 0),
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Source, 1, 0),
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Contribution, 2, 1),
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Finalization, 3, 2),
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Closure, 4, 3),
            ],
        )
        .unwrap()
    }

    #[test]
    fn query_family_union_interns_exact_currents_and_retains_destinations() {
        let (first, first_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        let (mut second, second_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x82, factor(1));
        second.test_remap_dynamic_color_id(7);
        let census = query_family_census_from_traces(
            &direct_catalog(),
            &[
                QueryFamilyTraceInput {
                    trace: &first,
                    projection: first_projection,
                },
                QueryFamilyTraceInput {
                    trace: &second,
                    projection: second_projection,
                },
            ],
        )
        .unwrap();

        assert_eq!(census.query_count, 2);
        assert_eq!(census.source_frame_partition_count, 1);
        assert_eq!(census.projection_applied_query_count, 0);
        assert_eq!(census.projection_pre_current_count, 6);
        assert_eq!(census.projection_pre_contribution_count, 2);
        assert_eq!(census.projection_pre_closure_count, 2);
        assert_eq!(census.projection_post_current_count, 6);
        assert_eq!(census.projection_post_contribution_count, 2);
        assert_eq!(census.projection_post_closure_count, 2);
        assert_eq!(census.dynamic_current_occurrence_count, 6);
        assert_eq!(census.dynamic_current_component_occurrence_count, 6);
        assert_eq!(census.dynamic_source_rows, 4);
        assert_eq!(census.dynamic_contribution_rows, 2);
        assert_eq!(census.dynamic_finalization_rows, 2);
        assert_eq!(census.dynamic_closure_rows, 2);
        assert_eq!(census.dynamic_source_calls, 4);
        assert_eq!(census.dynamic_contribution_calls, 2);
        assert_eq!(census.dynamic_finalization_calls, 2);
        assert_eq!(census.dynamic_closure_calls, 2);
        assert_eq!(census.union_unique_current_count, 3);
        assert_eq!(census.union_unique_current_component_count, 3);
        assert_eq!(census.union_source_rows, 2);
        assert_eq!(census.union_contribution_rows, 1);
        assert_eq!(census.union_finalization_rows, 1);
        assert_eq!(census.union_closure_rows, 2);
        assert_eq!(census.union_amplitude_destination_count, 2);
        assert_eq!(census.union_source_executor_call_groups, 1);
        assert_eq!(census.union_contribution_executor_call_groups, 1);
        assert_eq!(census.union_finalization_executor_call_groups, 1);
        assert_eq!(census.union_closure_executor_call_groups, 1);
    }

    #[test]
    fn query_family_groups_distinct_semantic_operations_by_prepared_executor() {
        let (trace, projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        let census = query_family_census_from_traces(
            &direct_catalog(),
            &[QueryFamilyTraceInput {
                trace: &trace,
                projection,
            }],
        )
        .unwrap();

        // The two source operations have different authenticated operation and
        // evaluator-binding identities, but the prepared catalog deliberately
        // maps both onto one Direct-Arena source executor.
        assert_eq!(census.union_source_rows, 2);
        assert_eq!(census.union_source_executor_call_groups, 1);
    }

    #[test]
    fn query_family_union_rejects_conflicting_semantic_current_definitions() {
        let (first, first_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        let (second, second_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x82, factor(2));
        let error = query_family_census_from_traces(
            &direct_catalog(),
            &[
                QueryFamilyTraceInput {
                    trace: &first,
                    projection: first_projection,
                },
                QueryFamilyTraceInput {
                    trace: &second,
                    projection: second_projection,
                },
            ],
        )
        .unwrap_err();
        assert!(error.to_string().contains("conflicting definitions"));
    }

    #[test]
    fn query_family_union_keeps_distinct_source_frame_partitions() {
        let (first, first_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        let (mut second, second_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x82, factor(1));
        second.seed_digest = SemanticDigest::new([0x83; 32]).unwrap();

        let census = query_family_census_from_traces(
            &direct_catalog(),
            &[
                QueryFamilyTraceInput {
                    trace: &first,
                    projection: first_projection,
                },
                QueryFamilyTraceInput {
                    trace: &second,
                    projection: second_projection,
                },
            ],
        )
        .unwrap();
        assert_eq!(census.source_frame_partition_count, 2);
        assert_eq!(census.union_unique_current_count, 6);
        assert_eq!(census.union_source_rows, 4);
        assert_eq!(census.union_contribution_rows, 2);
        assert_eq!(census.union_finalization_rows, 2);
        assert_eq!(census.union_closure_rows, 2);
        assert_eq!(census.union_source_executor_call_groups, 2);
        assert_eq!(census.union_contribution_executor_call_groups, 2);
        assert_eq!(census.union_finalization_executor_call_groups, 2);
        assert_eq!(census.union_closure_executor_call_groups, 2);
    }

    #[test]
    fn query_family_union_rejects_repeated_queries_and_stale_projection_proofs() {
        let (trace, projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        let repeated = query_family_census_from_traces(
            &direct_catalog(),
            &[
                QueryFamilyTraceInput {
                    trace: &trace,
                    projection,
                },
                QueryFamilyTraceInput {
                    trace: &trace,
                    projection,
                },
            ],
        )
        .unwrap_err();
        assert!(repeated.to_string().contains("repeats a selected query"));

        let stale = OnTheFlyProjectionProbeV1 {
            post: [2, 1, 1],
            ..projection
        };
        let error = query_family_census_from_traces(
            &direct_catalog(),
            &[QueryFamilyTraceInput {
                trace: &trace,
                projection: stale,
            }],
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("projection proof does not describe")
        );
    }
}
