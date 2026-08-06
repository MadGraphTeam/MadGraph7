// SPDX-License-Identifier: 0BSD

//! Feature-only structural union and execution for requested LC query families.
//!
//! Independently constructed query-local traces are merged into stable boxed
//! Direct-Arena rows, one shared numeric workspace, and an ordered grouped
//! invocation schedule.  This remains a private lane: it does not serialize a
//! new plan or introduce a public evaluation API.  Release/default builds do
//! not retain the cold contribution identities consumed here.

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

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct FamilyClosureIdentity {
    seed_digest: SemanticDigest,
    query_digest: SemanticDigest,
    stage: u32,
    executor: OnTheFlyExecutorKeyV1,
    parent_current_ids: [u32; 2],
    parent_momenta: [CanonicalMomentumLinearForm; 2],
    exact_factor: ExactComplexRational,
    component_factors: Box<[ExactComplexRational]>,
    component_count: u16,
    selector_domain_id: u32,
    proof_group_id: u32,
    amplitude_destination_id: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FamilyCurrentDefinition {
    component_count: u32,
    source: Option<FamilySourceIdentity>,
    contributions: BTreeSet<FamilyContributionIdentity>,
    finalization: Option<FamilyFinalizationIdentity>,
}

#[derive(Clone, Copy)]
pub(crate) struct QueryFamilyTraceInput<'a> {
    pub(crate) trace: &'a OnTheFlyStructuralTraceV1,
    pub(crate) projection: OnTheFlyProjectionProbeV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QueryFamilyCacheIdentityV1 {
    direct_catalog: PreparedDirectExecutorCatalog,
    queries: Box<
        [(
            SemanticDigest,
            SemanticDigest,
            SemanticDigest,
            OnTheFlyProjectionProbeV1,
        )],
    >,
}

#[derive(Debug)]
enum OnTheFlyFamilyRowsV1 {
    Source(Box<[DirectSourceRow]>),
    Contribution(Box<[DirectContributionRow]>),
    Finalization(Box<[DirectFinalizationRow]>),
    Closure(Box<[DirectClosureRow]>),
}

impl OnTheFlyFamilyRowsV1 {
    fn len(&self) -> usize {
        match self {
            Self::Source(rows) => rows.len(),
            Self::Contribution(rows) => rows.len(),
            Self::Finalization(rows) => rows.len(),
            Self::Closure(rows) => rows.len(),
        }
    }
}

#[derive(Debug)]
struct OnTheFlyFamilyRowGroupV1 {
    seed_digest: SemanticDigest,
    stage: u32,
    role: DirectExecutorRole,
    direct_executor_id: u32,
    representative_key: OnTheFlyExecutorKeyV1,
    member_bindings: Box<[(OnTheFlyExecutorKeyV1, [u8; 2])]>,
    rows: OnTheFlyFamilyRowsV1,
}

#[derive(Debug)]
struct OnTheFlyBuiltQueryFamilyV1 {
    census: OnTheFlyQueryFamilyCensusV1,
    source_count: u32,
    lorentz_component_count: u16,
    parameter_count: u32,
    current_component_count: u32,
    amplitude_destination_count: u32,
    momentum_forms: Box<[CanonicalMomentumLinearForm]>,
    exact_factors: Box<[ExactComplexRational]>,
    row_groups: Box<[OnTheFlyFamilyRowGroupV1]>,
    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    observed_currents: Box<[OnTheFlyFamilyObservedCurrentDescriptorV1]>,
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OnTheFlyFamilyObservedCurrentDescriptorV1 {
    semantic_digest: SemanticDigest,
    stage: u32,
    component_base: u32,
    component_count: u32,
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct OnTheFlyFamilyObservedCurrentV1 {
    pub(crate) semantic_digest: SemanticDigest,
    pub(crate) stage: u32,
    pub(crate) values: Vec<(f64, f64)>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct OnTheFlyQueryFamilyExecutionReportV1 {
    pub(crate) cache_hit: bool,
    pub(crate) source_calls: u32,
    pub(crate) source_rows: u32,
    pub(crate) contribution_calls: u32,
    pub(crate) contribution_rows: u32,
    pub(crate) finalization_calls: u32,
    pub(crate) finalization_rows: u32,
    pub(crate) closure_calls: u32,
    pub(crate) closure_rows: u32,
}

#[derive(Clone, Debug)]
struct FamilySourceDraft {
    seed_digest: SemanticDigest,
    stage: u32,
    direct_executor_id: u32,
    key: OnTheFlyExecutorKeyV1,
    row: DirectSourceRow,
}

#[derive(Clone, Debug)]
struct FamilyContributionDraft {
    seed_digest: SemanticDigest,
    stage: u32,
    destination_current_id: u32,
    direct_executor_id: u32,
    key: OnTheFlyExecutorKeyV1,
    parent_permutation: [u8; 2],
    row: DirectContributionRow,
}

#[derive(Clone, Debug)]
struct FamilyFinalizationDraft {
    seed_digest: SemanticDigest,
    stage: u32,
    direct_executor_id: u32,
    key: OnTheFlyExecutorKeyV1,
    row: DirectFinalizationRow,
}

#[derive(Clone, Debug)]
struct FamilyClosureDraft {
    seed_digest: SemanticDigest,
    stage: u32,
    direct_executor_id: u32,
    key: OnTheFlyExecutorKeyV1,
    row: DirectClosureRow,
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

impl OnTheFlyQueryFamilyCensusV1 {
    pub(crate) fn union_kernel_application_count(self) -> RusticolResult<u32> {
        self.union_source_rows
            .checked_add(self.union_contribution_rows)
            .and_then(|value| value.checked_add(self.union_finalization_rows))
            .and_then(|value| value.checked_add(self.union_closure_rows))
            .ok_or_else(|| integrity("query-family union kernel-application count exceeds u32"))
    }
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

fn current_stage(key: &CurrentCoreKey) -> RusticolResult<u32> {
    u32::try_from(
        key.support_source_slots()
            .len()
            .checked_sub(1)
            .ok_or_else(|| integrity("query-family current has empty source support"))?,
    )
    .map_err(|_| invalid("query-family current stage exceeds u32"))
}

fn intern_factor(
    ids: &mut BTreeMap<ExactComplexRational, u32>,
    factors: &mut Vec<ExactComplexRational>,
    value: ExactComplexRational,
) -> RusticolResult<u32> {
    if let Some(id) = ids.get(&value).copied() {
        return Ok(id);
    }
    let id = checked_u32(factors.len(), "query-family exact-factor ID")?;
    factors.try_reserve(1).map_err(|error| {
        invalid(format!(
            "query-family exact-factor allocation failed: {error}"
        ))
    })?;
    factors.push(value);
    ids.insert(value, id);
    Ok(id)
}

fn intern_factor_block(
    block_ids: &mut BTreeMap<Box<[ExactComplexRational]>, u32>,
    scalar_ids: &mut BTreeMap<ExactComplexRational, u32>,
    factors: &mut Vec<ExactComplexRational>,
    block: &[ExactComplexRational],
) -> RusticolResult<u32> {
    if block.is_empty() {
        return Err(integrity("query-family exact-factor block is empty"));
    }
    if let Some(start) = block_ids.get(block).copied() {
        return Ok(start);
    }
    let start = checked_u32(factors.len(), "query-family exact-factor block start")?;
    let _end = factors
        .len()
        .checked_add(block.len())
        .and_then(|end| u32::try_from(end).ok())
        .ok_or_else(|| invalid("query-family exact-factor block exceeds u32"))?;
    factors.try_reserve(block.len()).map_err(|error| {
        invalid(format!(
            "query-family exact-factor block allocation failed: {error}"
        ))
    })?;
    for (offset, value) in block.iter().copied().enumerate() {
        let id = start
            .checked_add(checked_u32(
                offset,
                "query-family exact-factor block offset",
            )?)
            .ok_or_else(|| invalid("query-family exact-factor block ID exceeds u32"))?;
        factors.push(value);
        scalar_ids.entry(value).or_insert(id);
    }
    block_ids.insert(block.to_vec().into_boxed_slice(), start);
    Ok(start)
}

fn intern_momentum(
    ids: &mut BTreeMap<CanonicalMomentumLinearForm, u32>,
    forms: &mut Vec<CanonicalMomentumLinearForm>,
    form: &CanonicalMomentumLinearForm,
) -> RusticolResult<u32> {
    if let Some(id) = ids.get(form).copied() {
        return Ok(id);
    }
    let id = checked_u32(forms.len(), "query-family momentum-form ID")?;
    forms.try_reserve(1).map_err(|error| {
        invalid(format!(
            "query-family momentum-form allocation failed: {error}"
        ))
    })?;
    forms.push(form.clone());
    ids.insert(form.clone(), id);
    Ok(id)
}

fn prepared_binding(
    catalog: &PreparedDirectExecutorCatalog,
    key: OnTheFlyExecutorKeyV1,
) -> RusticolResult<(u32, [u8; 2])> {
    if key.direct_catalog_digest() != catalog.direct_template_catalog_digest() {
        return Err(integrity(
            "query-family executor key belongs to a different prepared catalog",
        ));
    }
    match key.evaluator_binding_id() {
        Some(binding_id) if key.role() == DirectExecutorRole::Contribution => {
            catalog.resolve_contribution(binding_id)
        }
        Some(binding_id) => Ok((catalog.resolve_evaluator(key.role(), binding_id)?, [0, 1])),
        None => Ok((catalog.resolve_identity_finalizer()?, [0, 1])),
    }
}

/// Analyze the exact traces requested by one existing selector-family call.
///
/// Equal normalized [`CurrentCoreKey`] values are merged only when their
/// complete source/contribution/finalization definitions agree.  Conflicting
/// duplicates fail closed.  Closures remain distinct per requested query so a
/// future executable union cannot lose an amplitude destination.
fn build_query_family_from_traces(
    direct_catalog: &PreparedDirectExecutorCatalog,
    selected: &[QueryFamilyTraceInput<'_>],
) -> RusticolResult<OnTheFlyBuiltQueryFamilyV1> {
    let first = selected
        .first()
        .ok_or_else(|| invalid("query-family census requires at least one query"))?;
    let source_count = first.trace.layout.source_count;
    let lorentz_component_count = first.trace.layout.lorentz_component_count;
    let parameter_count = first.trace.layout.parameter_count;
    let mut seen_queries = BTreeSet::new();
    let mut seed_partitions = BTreeSet::new();
    let mut colors = DynamicLCColorStateInterner::default();
    let mut union_ids = BTreeMap::<(SemanticDigest, CurrentCoreKey), u32>::new();
    let mut definitions =
        BTreeMap::<(SemanticDigest, CurrentCoreKey), FamilyCurrentDefinition>::new();
    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    let mut observed_current_identities = BTreeMap::<u32, (SemanticDigest, u32, u32)>::new();
    let mut closure_definitions = Vec::<FamilyClosureIdentity>::new();
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

    for (query_index, selected_query) in selected.iter().enumerate() {
        let trace = selected_query.trace;
        if trace.layout.source_count != source_count
            || trace.layout.lorentz_component_count != lorentz_component_count
            || trace.layout.parameter_count != parameter_count
        {
            return Err(integrity(
                "query-family traces do not share one source-frame workspace shape",
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
        #[cfg(any(test, feature = "on-the-fly-test-support"))]
        {
            if trace.current_semantic_digests.len() != local_to_union.len() {
                return Err(integrity(
                    "query-local current semantic identities have an inconsistent length",
                ));
            }
            for (local_id, union_id) in local_to_union.iter().copied().enumerate() {
                let semantic_digest = trace.current_semantic_digests[local_id];
                let stage = current_stage(&trace.current_keys[local_id])?;
                let component_count = trace.current_component_ranges[local_id][1];
                match observed_current_identities.entry(union_id) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert((semantic_digest, stage, component_count));
                    }
                    std::collections::btree_map::Entry::Occupied(entry)
                        if *entry.get() == (semantic_digest, stage, component_count) => {}
                    std::collections::btree_map::Entry::Occupied(_) => {
                        return Err(integrity(
                            "query-family union current has inconsistent observed identity",
                        ));
                    }
                }
            }
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
                    let parent0 = current_id_by_component_base(
                        &bases,
                        row.parent0_component_base,
                        "closure parent 0",
                    )?;
                    let parent1 = current_id_by_component_base(
                        &bases,
                        row.parent1_component_base_or_sentinel,
                        "closure parent 1",
                    )?;
                    let component_start = row.component_factor_start as usize;
                    let component_end = component_start
                        .checked_add(usize::from(row.component_count))
                        .ok_or_else(|| integrity("closure component-factor span overflows"))?;
                    let component_factors = trace
                        .exact_factors
                        .get(component_start..component_end)
                        .ok_or_else(|| integrity("closure component-factor span is absent"))?
                        .to_vec()
                        .into_boxed_slice();
                    let amplitude_destination_id =
                        checked_u32(query_index, "query-family amplitude destination ID")?;
                    closure_definitions.push(FamilyClosureIdentity {
                        seed_digest: trace.seed_digest,
                        query_digest: trace.query_digest,
                        stage: current_stage(&trace.current_keys[parent0 as usize])?
                            .max(current_stage(&trace.current_keys[parent1 as usize])?),
                        executor: *key,
                        parent_current_ids: [
                            local_to_union[parent0 as usize],
                            local_to_union[parent1 as usize],
                        ],
                        parent_momenta: [
                            trace_momentum(
                                trace,
                                row.parent0_momentum_form_id,
                                "closure parent 0",
                            )?,
                            trace_momentum(
                                trace,
                                row.parent1_momentum_form_id_or_sentinel,
                                "closure parent 1",
                            )?,
                        ],
                        exact_factor: trace_factor(trace, row.exact_factor_id, "closure")?,
                        component_factors,
                        component_count: row.component_count,
                        selector_domain_id: row.selector_domain_id,
                        proof_group_id: row.flags,
                        amplitude_destination_id,
                    });
                    closure_groups.insert((
                        trace.seed_digest,
                        current_stage(&trace.current_keys[parent0 as usize])?
                            .max(current_stage(&trace.current_keys[parent1 as usize])?),
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
    let closure_queries = closure_definitions
        .iter()
        .map(|closure| closure.query_digest)
        .collect::<BTreeSet<_>>();
    let closure_destinations = closure_definitions
        .iter()
        .map(|closure| closure.amplitude_destination_id)
        .collect::<BTreeSet<_>>();
    if closure_queries != seen_queries
        || closure_destinations.len() != selected.len()
        || closure_destinations
            .iter()
            .copied()
            .enumerate()
            .any(|(expected, actual)| actual as usize != expected)
    {
        return Err(integrity(
            "query-family closures do not retain one distinct destination per requested query",
        ));
    }
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
    materialize_query_family(
        direct_catalog,
        census,
        source_count,
        lorentz_component_count,
        parameter_count,
        union_ids,
        definitions,
        closure_definitions,
        #[cfg(any(test, feature = "on-the-fly-test-support"))]
        observed_current_identities,
    )
}

fn query_family_census_from_traces(
    direct_catalog: &PreparedDirectExecutorCatalog,
    selected: &[QueryFamilyTraceInput<'_>],
) -> RusticolResult<OnTheFlyQueryFamilyCensusV1> {
    Ok(build_query_family_from_traces(direct_catalog, selected)?.census)
}

#[allow(clippy::too_many_arguments)]
fn materialize_query_family(
    direct_catalog: &PreparedDirectExecutorCatalog,
    census: OnTheFlyQueryFamilyCensusV1,
    source_count: u32,
    lorentz_component_count: u16,
    parameter_count: u32,
    union_ids: BTreeMap<(SemanticDigest, CurrentCoreKey), u32>,
    definitions: BTreeMap<(SemanticDigest, CurrentCoreKey), FamilyCurrentDefinition>,
    closure_definitions: Vec<FamilyClosureIdentity>,
    #[cfg(any(test, feature = "on-the-fly-test-support"))] observed_current_identities: BTreeMap<
        u32,
        (SemanticDigest, u32, u32),
    >,
) -> RusticolResult<OnTheFlyBuiltQueryFamilyV1> {
    type GroupKey = (SemanticDigest, u32);
    let mut records = std::iter::repeat_with(|| None)
        .take(definitions.len())
        .collect::<Vec<Option<(SemanticDigest, CurrentCoreKey, FamilyCurrentDefinition)>>>();
    for (family_key, definition) in definitions {
        let id = union_ids
            .get(&family_key)
            .copied()
            .ok_or_else(|| integrity("query-family definition has no union current identity"))?;
        let slot = records
            .get_mut(id as usize)
            .ok_or_else(|| integrity("query-family union current ID is out of bounds"))?;
        if slot
            .replace((family_key.0, family_key.1, definition))
            .is_some()
        {
            return Err(integrity(
                "query-family union current ID repeats a definition",
            ));
        }
    }
    if records.iter().any(Option::is_none) {
        return Err(integrity(
            "query-family union current IDs are not dense zero-based",
        ));
    }
    let records = records
        .into_iter()
        .map(|record| record.expect("validated dense query-family records"))
        .collect::<Vec<_>>();

    let mut component_bases = Vec::new();
    let mut stages = Vec::new();
    let mut current_component_count = 0_u32;
    component_bases
        .try_reserve_exact(records.len())
        .map_err(|error| {
            invalid(format!(
                "query-family component-base allocation failed: {error}"
            ))
        })?;
    stages
        .try_reserve_exact(records.len())
        .map_err(|error| invalid(format!("query-family stage allocation failed: {error}")))?;
    for (_, key, definition) in &records {
        component_bases.push(current_component_count);
        current_component_count = current_component_count
            .checked_add(definition.component_count)
            .ok_or_else(|| invalid("query-family current component count exceeds u32"))?;
        stages.push(current_stage(key)?);
    }
    if current_component_count != census.union_unique_current_component_count {
        return Err(integrity(
            "query-family materialized current shape differs from its census",
        ));
    }
    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    let observed_currents = component_bases
        .iter()
        .copied()
        .enumerate()
        .map(|(current_id, component_base)| {
            let current_id = checked_u32(current_id, "observed query-family current ID")?;
            let (semantic_digest, stage, component_count) = observed_current_identities
                .get(&current_id)
                .copied()
                .ok_or_else(|| integrity("query-family observed current identity is absent"))?;
            Ok(OnTheFlyFamilyObservedCurrentDescriptorV1 {
                semantic_digest,
                stage,
                component_base,
                component_count,
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?
        .into_boxed_slice();

    let mut momentum_ids = BTreeMap::new();
    let mut momentum_forms = Vec::new();
    let mut factor_ids = BTreeMap::new();
    let mut factor_block_ids = BTreeMap::new();
    let mut exact_factors = Vec::new();
    let mut source_groups = BTreeMap::<GroupKey, Vec<FamilySourceDraft>>::new();
    let mut contribution_groups =
        BTreeMap::<u32, BTreeMap<GroupKey, Vec<FamilyContributionDraft>>>::new();
    let mut finalization_groups =
        BTreeMap::<u32, BTreeMap<GroupKey, Vec<FamilyFinalizationDraft>>>::new();
    let mut closure_groups = BTreeMap::<u32, BTreeMap<GroupKey, Vec<FamilyClosureDraft>>>::new();

    for (current_id, (seed_digest, _key, definition)) in records.iter().enumerate() {
        let current_id = checked_u32(current_id, "query-family current ID")?;
        let stage = stages[current_id as usize];
        let destination_component_base = component_bases[current_id as usize];
        if let Some(source) = &definition.source {
            if stage != 0 {
                return Err(integrity(
                    "query-family source current is not in support stage zero",
                ));
            }
            let (direct_executor_id, parent_permutation) =
                prepared_binding(direct_catalog, source.executor)?;
            if parent_permutation != [0, 1] {
                return Err(integrity(
                    "query-family source executor has a parent permutation",
                ));
            }
            let row = DirectSourceRow {
                source_slot: source.source_slot,
                destination_component_base,
                momentum_form_id: intern_momentum(
                    &mut momentum_ids,
                    &mut momentum_forms,
                    &source.momentum,
                )?,
                source_template_or_dispatch_domain: source.source_template_or_dispatch_domain,
                spin_state_class: source.spin_state_class,
                exact_factor_id: intern_factor(
                    &mut factor_ids,
                    &mut exact_factors,
                    source.exact_factor,
                )?,
                selector_domain_id: source.selector_domain_id,
            };
            source_groups
                .entry((*seed_digest, direct_executor_id))
                .or_default()
                .push(FamilySourceDraft {
                    seed_digest: *seed_digest,
                    stage,
                    direct_executor_id,
                    key: source.executor,
                    row,
                });
        }
        for contribution in &definition.contributions {
            let parent_ids = contribution.key.parent_value_class_ids();
            let parent_momenta = contribution.key.parent_momenta();
            if parent_ids.len() != 2 || parent_momenta.len() != 2 {
                return Err(integrity("query-family direct contribution is not binary"));
            }
            let mut parent_ids = [parent_ids[0], parent_ids[1]];
            let mut parent_momenta = [&parent_momenta[0], &parent_momenta[1]];
            for parent_id in parent_ids {
                let parent_stage = stages
                    .get(parent_id as usize)
                    .copied()
                    .ok_or_else(|| integrity("query-family contribution parent is absent"))?;
                if parent_stage >= stage {
                    return Err(integrity(
                        "query-family contribution parent is not in an earlier support stage",
                    ));
                }
            }
            let (direct_executor_id, parent_permutation) =
                prepared_binding(direct_catalog, contribution.executor)?;
            match parent_permutation {
                [0, 1] => {}
                [1, 0] => {
                    parent_ids.swap(0, 1);
                    parent_momenta.swap(0, 1);
                }
                _ => {
                    return Err(integrity(
                        "query-family contribution has an invalid prepared parent permutation",
                    ));
                }
            }
            let row = DirectContributionRow {
                parent0_component_base: component_bases[parent_ids[0] as usize],
                parent1_component_base_or_sentinel: component_bases[parent_ids[1] as usize],
                parent0_momentum_form_id: intern_momentum(
                    &mut momentum_ids,
                    &mut momentum_forms,
                    parent_momenta[0],
                )?,
                parent1_momentum_form_id_or_sentinel: intern_momentum(
                    &mut momentum_ids,
                    &mut momentum_forms,
                    parent_momenta[1],
                )?,
                destination_component_base,
                exact_factor_id: intern_factor(
                    &mut factor_ids,
                    &mut exact_factors,
                    contribution.exact_factor,
                )?,
                selector_domain_id: contribution.selector_domain_id,
                flags: 0,
            };
            contribution_groups
                .entry(stage)
                .or_default()
                .entry((*seed_digest, direct_executor_id))
                .or_default()
                .push(FamilyContributionDraft {
                    seed_digest: *seed_digest,
                    stage,
                    destination_current_id: current_id,
                    direct_executor_id,
                    key: contribution.executor,
                    parent_permutation,
                    row,
                });
        }
        if let Some(finalization) = &definition.finalization {
            let (direct_executor_id, parent_permutation) =
                prepared_binding(direct_catalog, finalization.executor)?;
            if parent_permutation != [0, 1] {
                return Err(integrity(
                    "query-family finalization executor has a parent permutation",
                ));
            }
            let row = DirectFinalizationRow {
                component_base: destination_component_base,
                component_count: finalization.component_count,
                momentum_form_id: intern_momentum(
                    &mut momentum_ids,
                    &mut momentum_forms,
                    &finalization.momentum,
                )?,
                exact_factor_id: intern_factor(
                    &mut factor_ids,
                    &mut exact_factors,
                    finalization.exact_factor,
                )?,
                selector_domain_id: finalization.selector_domain_id,
                flags: finalization.flags,
            };
            finalization_groups
                .entry(stage)
                .or_default()
                .entry((*seed_digest, direct_executor_id))
                .or_default()
                .push(FamilyFinalizationDraft {
                    seed_digest: *seed_digest,
                    stage,
                    direct_executor_id,
                    key: finalization.executor,
                    row,
                });
        }
    }

    for closure in closure_definitions {
        let [parent0, parent1] = closure.parent_current_ids;
        let parent0_count = records
            .get(parent0 as usize)
            .map(|record| record.2.component_count)
            .ok_or_else(|| integrity("query-family closure parent 0 is absent"))?;
        let parent1_count = records
            .get(parent1 as usize)
            .map(|record| record.2.component_count)
            .ok_or_else(|| integrity("query-family closure parent 1 is absent"))?;
        if parent0_count != u32::from(closure.component_count)
            || parent1_count != u32::from(closure.component_count)
            || closure.component_factors.len() != usize::from(closure.component_count)
        {
            return Err(integrity(
                "query-family closure component shape disagrees with its parents",
            ));
        }
        let (direct_executor_id, parent_permutation) =
            prepared_binding(direct_catalog, closure.executor)?;
        if parent_permutation != [0, 1] {
            return Err(integrity(
                "query-family closure executor has a parent permutation",
            ));
        }
        let exact_factor_id =
            intern_factor(&mut factor_ids, &mut exact_factors, closure.exact_factor)?;
        let component_factor_start = intern_factor_block(
            &mut factor_block_ids,
            &mut factor_ids,
            &mut exact_factors,
            &closure.component_factors,
        )?;
        let stage = stages[parent0 as usize].max(stages[parent1 as usize]);
        if stage != closure.stage {
            return Err(integrity(
                "query-family closure stage changed after current remapping",
            ));
        }
        let row = DirectClosureRow {
            parent0_component_base: component_bases[parent0 as usize],
            parent1_component_base_or_sentinel: component_bases[parent1 as usize],
            parent0_momentum_form_id: intern_momentum(
                &mut momentum_ids,
                &mut momentum_forms,
                &closure.parent_momenta[0],
            )?,
            parent1_momentum_form_id_or_sentinel: intern_momentum(
                &mut momentum_ids,
                &mut momentum_forms,
                &closure.parent_momenta[1],
            )?,
            amplitude_destination_id: closure.amplitude_destination_id,
            exact_factor_id,
            component_factor_start,
            component_count: closure.component_count,
            selector_domain_id: closure.selector_domain_id,
            flags: closure.proof_group_id,
        };
        closure_groups
            .entry(stage)
            .or_default()
            .entry((closure.seed_digest, direct_executor_id))
            .or_default()
            .push(FamilyClosureDraft {
                seed_digest: closure.seed_digest,
                stage,
                direct_executor_id,
                key: closure.executor,
                row,
            });
    }

    let row_groups = ordered_family_row_groups(
        source_groups,
        contribution_groups,
        finalization_groups,
        closure_groups,
    )?;
    validate_ordered_family_schedule(&row_groups)?;
    let group_count = |role| row_groups.iter().filter(|group| group.role == role).count();
    if checked_u32(
        group_count(DirectExecutorRole::Source),
        "source group count",
    )? != census.union_source_executor_call_groups
        || checked_u32(
            group_count(DirectExecutorRole::Contribution),
            "contribution group count",
        )? != census.union_contribution_executor_call_groups
        || checked_u32(
            group_count(DirectExecutorRole::Finalization),
            "finalization group count",
        )? != census.union_finalization_executor_call_groups
        || checked_u32(
            group_count(DirectExecutorRole::Closure),
            "closure group count",
        )? != census.union_closure_executor_call_groups
    {
        return Err(integrity(
            "query-family executable groups differ from the structural census",
        ));
    }

    Ok(OnTheFlyBuiltQueryFamilyV1 {
        census,
        source_count,
        lorentz_component_count,
        parameter_count,
        current_component_count,
        amplitude_destination_count: census.union_amplitude_destination_count,
        momentum_forms: momentum_forms.into_boxed_slice(),
        exact_factors: exact_factors.into_boxed_slice(),
        row_groups: row_groups.into_boxed_slice(),
        #[cfg(any(test, feature = "on-the-fly-test-support"))]
        observed_currents,
    })
}

fn validate_ordered_family_schedule(groups: &[OnTheFlyFamilyRowGroupV1]) -> RusticolResult<()> {
    let mut group_keys = BTreeSet::new();
    let mut saw_non_source = false;
    let mut saw_closure = false;
    let mut current_stage = 0_u32;
    let mut finalization_seen_in_stage = false;
    for group in groups {
        if group.rows.len() == 0
            || !group_keys.insert((
                group.seed_digest,
                group.stage,
                group.role,
                group.direct_executor_id,
            ))
        {
            return Err(integrity(
                "query-family schedule has an empty or duplicate row group",
            ));
        }
        match group.role {
            DirectExecutorRole::Source => {
                if saw_non_source || group.stage != 0 {
                    return Err(integrity(
                        "query-family sources are not the first stage-zero groups",
                    ));
                }
            }
            DirectExecutorRole::Contribution => {
                saw_non_source = true;
                if saw_closure || group.stage < current_stage {
                    return Err(integrity(
                        "query-family contribution stages are not topological",
                    ));
                }
                if group.stage > current_stage {
                    current_stage = group.stage;
                    finalization_seen_in_stage = false;
                }
                if finalization_seen_in_stage {
                    return Err(integrity(
                        "query-family contribution follows finalization in one stage",
                    ));
                }
            }
            DirectExecutorRole::Finalization => {
                saw_non_source = true;
                if saw_closure || group.stage < current_stage {
                    return Err(integrity(
                        "query-family finalization stages are not topological",
                    ));
                }
                if group.stage > current_stage {
                    current_stage = group.stage;
                }
                finalization_seen_in_stage = true;
            }
            DirectExecutorRole::Closure => {
                saw_non_source = true;
                saw_closure = true;
            }
        }
    }
    Ok(())
}

fn binding_members(
    bindings: impl IntoIterator<Item = (OnTheFlyExecutorKeyV1, [u8; 2])>,
) -> Box<[(OnTheFlyExecutorKeyV1, [u8; 2])]> {
    bindings
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice()
}

fn ordered_family_row_groups(
    source_groups: BTreeMap<(SemanticDigest, u32), Vec<FamilySourceDraft>>,
    mut contribution_groups: BTreeMap<
        u32,
        BTreeMap<(SemanticDigest, u32), Vec<FamilyContributionDraft>>,
    >,
    mut finalization_groups: BTreeMap<
        u32,
        BTreeMap<(SemanticDigest, u32), Vec<FamilyFinalizationDraft>>,
    >,
    closure_groups: BTreeMap<u32, BTreeMap<(SemanticDigest, u32), Vec<FamilyClosureDraft>>>,
) -> RusticolResult<Vec<OnTheFlyFamilyRowGroupV1>> {
    let mut groups = Vec::new();
    for ((seed_digest, direct_executor_id), mut drafts) in source_groups {
        drafts.sort_by_key(|draft| {
            (
                draft.row.destination_component_base,
                draft.row.source_slot,
                draft.key,
            )
        });
        let representative = drafts
            .first()
            .ok_or_else(|| integrity("query-family source group is empty"))?;
        if drafts.iter().any(|draft| {
            draft.seed_digest != seed_digest
                || draft.stage != 0
                || draft.direct_executor_id != direct_executor_id
        }) {
            return Err(integrity("query-family source group key is inconsistent"));
        }
        groups.push(OnTheFlyFamilyRowGroupV1 {
            seed_digest,
            stage: 0,
            role: DirectExecutorRole::Source,
            direct_executor_id,
            representative_key: representative.key,
            member_bindings: binding_members(drafts.iter().map(|draft| (draft.key, [0, 1]))),
            rows: OnTheFlyFamilyRowsV1::Source(
                drafts
                    .into_iter()
                    .map(|draft| draft.row)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            ),
        });
    }

    let mut schedule_stages = contribution_groups
        .keys()
        .chain(finalization_groups.keys())
        .copied()
        .collect::<BTreeSet<_>>();
    let mut initialized_destinations = BTreeSet::<u32>::new();
    for stage in std::mem::take(&mut schedule_stages) {
        for ((seed_digest, direct_executor_id), mut drafts) in
            contribution_groups.remove(&stage).unwrap_or_default()
        {
            drafts.sort_by_key(|draft| {
                (
                    draft.destination_current_id,
                    draft.row.parent0_component_base,
                    draft.row.parent1_component_base_or_sentinel,
                    draft.row.parent0_momentum_form_id,
                    draft.row.parent1_momentum_form_id_or_sentinel,
                    draft.row.exact_factor_id,
                    draft.key,
                )
            });
            if drafts.iter().any(|draft| {
                draft.seed_digest != seed_digest
                    || draft.stage != stage
                    || draft.direct_executor_id != direct_executor_id
            }) {
                return Err(integrity(
                    "query-family contribution group key is inconsistent",
                ));
            }
            for draft in &mut drafts {
                draft.row.flags = if initialized_destinations.insert(draft.destination_current_id) {
                    DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION
                } else {
                    0
                };
            }
            let representative = drafts
                .first()
                .ok_or_else(|| integrity("query-family contribution group is empty"))?;
            groups.push(OnTheFlyFamilyRowGroupV1 {
                seed_digest,
                stage,
                role: DirectExecutorRole::Contribution,
                direct_executor_id,
                representative_key: representative.key,
                member_bindings: binding_members(
                    drafts
                        .iter()
                        .map(|draft| (draft.key, draft.parent_permutation)),
                ),
                rows: OnTheFlyFamilyRowsV1::Contribution(
                    drafts
                        .into_iter()
                        .map(|draft| draft.row)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                ),
            });
        }
        for ((seed_digest, direct_executor_id), mut drafts) in
            finalization_groups.remove(&stage).unwrap_or_default()
        {
            drafts.sort_by_key(|draft| (draft.row.component_base, draft.key));
            let representative = drafts
                .first()
                .ok_or_else(|| integrity("query-family finalization group is empty"))?;
            if drafts.iter().any(|draft| {
                draft.seed_digest != seed_digest
                    || draft.stage != stage
                    || draft.direct_executor_id != direct_executor_id
            }) {
                return Err(integrity(
                    "query-family finalization group key is inconsistent",
                ));
            }
            groups.push(OnTheFlyFamilyRowGroupV1 {
                seed_digest,
                stage,
                role: DirectExecutorRole::Finalization,
                direct_executor_id,
                representative_key: representative.key,
                member_bindings: binding_members(drafts.iter().map(|draft| (draft.key, [0, 1]))),
                rows: OnTheFlyFamilyRowsV1::Finalization(
                    drafts
                        .into_iter()
                        .map(|draft| draft.row)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                ),
            });
        }
    }
    if !contribution_groups.is_empty() || !finalization_groups.is_empty() {
        return Err(integrity(
            "query-family stage schedule did not consume every row group",
        ));
    }

    for (stage, stage_groups) in closure_groups {
        for ((seed_digest, direct_executor_id), mut drafts) in stage_groups {
            drafts.sort_by_key(|draft| {
                (
                    draft.row.amplitude_destination_id,
                    draft.row.parent0_component_base,
                    draft.row.parent1_component_base_or_sentinel,
                    draft.key,
                )
            });
            let representative = drafts
                .first()
                .ok_or_else(|| integrity("query-family closure group is empty"))?;
            if drafts.iter().any(|draft| {
                draft.seed_digest != seed_digest
                    || draft.stage != stage
                    || draft.direct_executor_id != direct_executor_id
            }) {
                return Err(integrity("query-family closure group key is inconsistent"));
            }
            groups.push(OnTheFlyFamilyRowGroupV1 {
                seed_digest,
                stage,
                role: DirectExecutorRole::Closure,
                direct_executor_id,
                representative_key: representative.key,
                member_bindings: binding_members(drafts.iter().map(|draft| (draft.key, [0, 1]))),
                rows: OnTheFlyFamilyRowsV1::Closure(
                    drafts
                        .into_iter()
                        .map(|draft| draft.row)
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),
                ),
            });
        }
    }
    Ok(groups)
}

fn family_factor_parts(value: ExactComplexRational) -> RusticolResult<(f64, f64)> {
    let real = value.real().numerator() as f64 / value.real().denominator() as f64;
    let imag = value.imag().numerator() as f64 / value.imag().denominator() as f64;
    if !real.is_finite() || !imag.is_finite() {
        return Err(invalid(
            "query-family exact factor cannot be represented as finite binary64",
        ));
    }
    Ok((real, imag))
}

fn family_scalar_len(planes: u32, stride: u32, label: &str) -> RusticolResult<usize> {
    usize::try_from(planes)
        .ok()
        .and_then(|planes| planes.checked_mul(stride as usize))
        .ok_or_else(|| invalid(format!("query-family {label} scalar length exceeds usize")))
}

struct OnTheFlyFamilyWorkspaceV1 {
    current_re: AlignedF64Buffer,
    current_im: AlignedF64Buffer,
    amplitude_re: AlignedF64Buffer,
    amplitude_im: AlignedF64Buffer,
    momenta: AlignedF64Buffer,
    parameters_re: AlignedF64Buffer,
    parameters_im: AlignedF64Buffer,
    factors_re: AlignedF64Buffer,
    factors_im: AlignedF64Buffer,
    logical_point_capacity: u32,
    active_point_count: u32,
    source_count: u32,
    momentum_form_count: u32,
    lorentz_component_count: u16,
    point_stride: u32,
}

impl OnTheFlyFamilyWorkspaceV1 {
    fn new(
        family: &OnTheFlyBuiltQueryFamilyV1,
        logical_point_capacity: u32,
    ) -> RusticolResult<Self> {
        if family.source_count == 0
            || family.lorentz_component_count == 0
            || logical_point_capacity == 0
            || family.momentum_forms.is_empty()
            || family.exact_factors.is_empty()
            || family.current_component_count == 0
            || family.amplitude_destination_count == 0
        {
            return Err(integrity(
                "query-family has an empty authenticated workspace shape",
            ));
        }
        let point_stride = checked_aligned_point_stride(logical_point_capacity)?;
        let current_len = family_scalar_len(
            family.current_component_count,
            point_stride,
            "current arena",
        )?;
        let amplitude_len = family_scalar_len(
            family.amplitude_destination_count,
            point_stride,
            "amplitude arena",
        )?;
        let momentum_form_count = checked_u32(
            family.momentum_forms.len(),
            "query-family momentum-form count",
        )?;
        let momentum_planes = momentum_form_count
            .checked_mul(u32::from(family.lorentz_component_count))
            .ok_or_else(|| invalid("query-family momentum plane count exceeds u32"))?;
        let momentum_len = family_scalar_len(momentum_planes, point_stride, "momentum arena")?;
        let mut factors_re =
            AlignedF64Buffer::zeroed(family.exact_factors.len(), "query-family factor real")?;
        let mut factors_im =
            AlignedF64Buffer::zeroed(family.exact_factors.len(), "query-family factor imaginary")?;
        for (index, factor) in family.exact_factors.iter().copied().enumerate() {
            let (real, imag) = family_factor_parts(factor)?;
            factors_re.as_mut_slice()[index] = real;
            factors_im.as_mut_slice()[index] = imag;
        }
        let parameter_count = usize::try_from(family.parameter_count)
            .map_err(|_| invalid("query-family parameter count exceeds usize"))?;
        Ok(Self {
            current_re: AlignedF64Buffer::zeroed(current_len, "query-family current real")?,
            current_im: AlignedF64Buffer::zeroed(current_len, "query-family current imaginary")?,
            amplitude_re: AlignedF64Buffer::zeroed(amplitude_len, "query-family amplitude real")?,
            amplitude_im: AlignedF64Buffer::zeroed(
                amplitude_len,
                "query-family amplitude imaginary",
            )?,
            momenta: AlignedF64Buffer::zeroed(momentum_len, "query-family momenta")?,
            parameters_re: AlignedF64Buffer::zeroed(
                parameter_count,
                "query-family parameter real",
            )?,
            parameters_im: AlignedF64Buffer::zeroed(
                parameter_count,
                "query-family parameter imaginary",
            )?,
            factors_re,
            factors_im,
            logical_point_capacity,
            active_point_count: 0,
            source_count: family.source_count,
            momentum_form_count,
            lorentz_component_count: family.lorentz_component_count,
            point_stride,
        })
    }

    fn refresh_inputs(
        &mut self,
        family: &OnTheFlyBuiltQueryFamilyV1,
        external_momenta: &[f64],
        point_count: u32,
    ) -> RusticolResult<()> {
        self.active_point_count = 0;
        if point_count == 0 || point_count > self.logical_point_capacity {
            return Err(invalid(
                "query-family active point count is outside the workspace",
            ));
        }
        let expected = usize::try_from(self.source_count)
            .ok()
            .and_then(|sources| sources.checked_mul(usize::from(self.lorentz_component_count)))
            .and_then(|planes| planes.checked_mul(point_count as usize))
            .ok_or_else(|| invalid("query-family external momentum shape exceeds usize"))?;
        if external_momenta.len() != expected {
            return Err(invalid(format!(
                "query-family received {} momentum scalars, expected {expected}",
                external_momenta.len()
            )));
        }
        for (form_id, form) in family.momentum_forms.iter().enumerate() {
            for lorentz in 0..usize::from(self.lorentz_component_count) {
                for point in 0..point_count as usize {
                    let mut value = 0.0;
                    for term in form.terms() {
                        if term.source_slot >= self.source_count {
                            return Err(integrity(
                                "query-family momentum source slot is out of bounds",
                            ));
                        }
                        let input_index = (term.source_slot as usize
                            * usize::from(self.lorentz_component_count)
                            + lorentz)
                            * point_count as usize
                            + point;
                        value += f64::from(term.coefficient) * external_momenta[input_index];
                    }
                    let plane = form_id
                        .checked_mul(usize::from(self.lorentz_component_count))
                        .and_then(|base| base.checked_add(lorentz))
                        .ok_or_else(|| invalid("query-family momentum plane exceeds usize"))?;
                    let index = plane
                        .checked_mul(self.point_stride as usize)
                        .and_then(|base| base.checked_add(point))
                        .ok_or_else(|| invalid("query-family momentum index exceeds usize"))?;
                    self.momenta.as_mut_slice()[index] = value;
                }
            }
        }
        for (real, imag) in [
            (&mut self.current_re, &mut self.current_im),
            (&mut self.amplitude_re, &mut self.amplitude_im),
        ] {
            for plane in 0..real.len() / self.point_stride as usize {
                let start = plane * self.point_stride as usize;
                let end = start + point_count as usize;
                real.as_mut_slice()[start..end].fill(0.0);
                imag.as_mut_slice()[start..end].fill(0.0);
            }
        }
        Ok(())
    }

    fn refresh_parameters(&mut self, parameters: &[(f64, f64)]) -> RusticolResult<()> {
        self.active_point_count = 0;
        if parameters.len() != self.parameters_re.len() {
            return Err(invalid(format!(
                "query-family received {} parameters, expected {}",
                parameters.len(),
                self.parameters_re.len()
            )));
        }
        for (index, &(real, imag)) in parameters.iter().enumerate() {
            if !real.is_finite() || !imag.is_finite() {
                return Err(invalid("query-family parameter value is not finite"));
            }
            self.parameters_re.as_mut_slice()[index] = real;
            self.parameters_im.as_mut_slice()[index] = imag;
        }
        Ok(())
    }

    fn raw_views(
        &mut self,
    ) -> RusticolResult<(
        DirectArenaView,
        DirectMomentumView,
        DirectParameterView,
        DirectFactorView,
    )> {
        let arena = DirectArenaView {
            current_re: self.current_re.as_mut_ptr(),
            current_im: self.current_im.as_mut_ptr(),
            current_scalar_len: self.current_re.len() as u64,
            amplitude_re: self.amplitude_re.as_mut_ptr(),
            amplitude_im: self.amplitude_im.as_mut_ptr(),
            amplitude_scalar_len: self.amplitude_re.len() as u64,
            point_stride: self.point_stride,
        };
        let momenta = DirectMomentumView {
            values: self.momenta.as_ptr(),
            scalar_len: self.momenta.len() as u64,
            form_count: self.momentum_form_count,
            lorentz_component_count: self.lorentz_component_count,
            point_stride: self.point_stride,
        };
        let parameters = DirectParameterView {
            values_re: self.parameters_re.as_ptr(),
            values_im: self.parameters_im.as_ptr(),
            value_count: self.parameters_re.len() as u32,
        };
        let factors = DirectFactorView {
            values_re: self.factors_re.as_ptr(),
            values_im: self.factors_im.as_ptr(),
            value_count: self.factors_re.len() as u32,
        };
        validate_direct_views(arena, momenta, parameters, factors)?;
        Ok((arena, momenta, parameters, factors))
    }

    fn write_outputs(
        &self,
        destination_count: u32,
        point_count: u32,
        outputs: &mut [(f64, f64)],
    ) -> RusticolResult<()> {
        if self.active_point_count != point_count || destination_count == 0 {
            return Err(integrity(
                "query-family outputs do not belong to the last successful execution",
            ));
        }
        Self::validate_output_shape(destination_count, point_count, outputs.len())?;
        for destination in 0..destination_count {
            for point in 0..point_count {
                let arena_index = usize::try_from(destination)
                    .ok()
                    .and_then(|destination| destination.checked_mul(self.point_stride as usize))
                    .and_then(|base| base.checked_add(point as usize))
                    .ok_or_else(|| integrity("query-family amplitude index exceeds usize"))?;
                let output_index = usize::try_from(destination)
                    .ok()
                    .and_then(|destination| destination.checked_mul(point_count as usize))
                    .and_then(|base| base.checked_add(point as usize))
                    .ok_or_else(|| integrity("query-family output index exceeds usize"))?;
                outputs[output_index] = (
                    *self
                        .amplitude_re
                        .as_slice()
                        .get(arena_index)
                        .ok_or_else(|| integrity("query-family amplitude real value is absent"))?,
                    *self
                        .amplitude_im
                        .as_slice()
                        .get(arena_index)
                        .ok_or_else(|| {
                            integrity("query-family amplitude imaginary value is absent")
                        })?,
                );
            }
        }
        Ok(())
    }

    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    fn observed_currents(
        &self,
        descriptors: &[OnTheFlyFamilyObservedCurrentDescriptorV1],
        point_index: u32,
    ) -> RusticolResult<Vec<OnTheFlyFamilyObservedCurrentV1>> {
        if point_index >= self.active_point_count {
            return Err(invalid(
                "query-family observed current point is outside the last successful execution",
            ));
        }
        let stride = self.point_stride as usize;
        let point_index = point_index as usize;
        descriptors
            .iter()
            .map(|descriptor| {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(descriptor.component_count as usize)
                    .map_err(|error| {
                        invalid(format!(
                            "query-family observed current allocation failed: {error}"
                        ))
                    })?;
                for component in 0..descriptor.component_count as usize {
                    let plane = descriptor.component_base as usize + component;
                    let index = plane
                        .checked_mul(stride)
                        .and_then(|base| base.checked_add(point_index))
                        .ok_or_else(|| {
                            integrity("query-family observed current index exceeds usize")
                        })?;
                    values.push((
                        *self.current_re.as_slice().get(index).ok_or_else(|| {
                            integrity("query-family observed current real value is absent")
                        })?,
                        *self.current_im.as_slice().get(index).ok_or_else(|| {
                            integrity("query-family observed current imaginary value is absent")
                        })?,
                    ));
                }
                Ok(OnTheFlyFamilyObservedCurrentV1 {
                    semantic_digest: descriptor.semantic_digest,
                    stage: descriptor.stage,
                    values,
                })
            })
            .collect()
    }

    fn validate_output_shape(
        destination_count: u32,
        point_count: u32,
        output_len: usize,
    ) -> RusticolResult<()> {
        if destination_count == 0 || point_count == 0 {
            return Err(integrity("query-family output shape is empty"));
        }
        let expected = usize::try_from(destination_count)
            .ok()
            .and_then(|destinations| destinations.checked_mul(point_count as usize))
            .ok_or_else(|| invalid("query-family output shape exceeds usize"))?;
        if output_len != expected {
            return Err(invalid(format!(
                "query-family output has {} complex values, expected {expected}",
                output_len
            )));
        }
        Ok(())
    }
}

struct BoundOnTheFlyQueryFamilyV1 {
    identity: QueryFamilyCacheIdentityV1,
    lookup_key: QueryFamilyLookupKeyV1,
    // Single ownership keeps every row address stable for prepared descriptor caches.
    family: OnTheFlyBuiltQueryFamilyV1,
    workspace: OnTheFlyFamilyWorkspaceV1,
    resolved_groups: Box<[ResolvedOnTheFlyExecutor]>,
    applied_parameter_version: u64,
    descriptor_exposed: bool,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct QueryFamilyLookupKeyV1 {
    queries: Box<
        [(
            SemanticDigest,
            SemanticDigest,
            SemanticDigest,
            bool,
            bool,
            [u32; 3],
            [u32; 3],
        )],
    >,
}

/// Runtime-private stable address of one successfully warmed family.  The
/// generation prevents a handle retained above this executor from aliasing a
/// different family after `clear_families` resets the arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlyQueryFamilyHandleV1 {
    generation: u64,
    index: usize,
}

pub(crate) struct OnTheFlyQueryFamilyExecutorV1<R: OnTheFlyPreparedExecutorResolver> {
    // Field order is part of the ownership contract. Retained and pending rows
    // must disappear before the resolver drops the prepared contexts that
    // their copied handles address.
    families: Vec<BoundOnTheFlyQueryFamilyV1>,
    family_lookup: BTreeMap<QueryFamilyLookupKeyV1, Vec<usize>>,
    last_used: Option<usize>,
    pending: Option<BoundOnTheFlyQueryFamilyV1>,
    resolver: R,
    parameter_state: Vec<(f64, f64)>,
    parameter_version: u64,
    family_generation: u64,
}

impl<R: OnTheFlyPreparedExecutorResolver> OnTheFlyQueryFamilyExecutorV1<R> {
    pub(crate) const fn new(resolver: R) -> Self {
        Self {
            families: Vec::new(),
            family_lookup: BTreeMap::new(),
            last_used: None,
            pending: None,
            resolver,
            parameter_state: Vec::new(),
            parameter_version: 0,
            family_generation: 0,
        }
    }

    pub(crate) const fn resolver(&self) -> &R {
        &self.resolver
    }

    pub(crate) const fn resolver_mut(&mut self) -> &mut R {
        &mut self.resolver
    }

    pub(crate) fn set_parameters(&mut self, parameters: &[(f64, f64)]) -> RusticolResult<()> {
        if parameters
            .iter()
            .any(|(real, imag)| !real.is_finite() || !imag.is_finite())
        {
            return Err(invalid("query-family parameter value is not finite"));
        }
        if self.parameter_state == parameters {
            return Ok(());
        }
        let mut replacement = Vec::new();
        replacement
            .try_reserve_exact(parameters.len())
            .map_err(|error| {
                invalid(format!(
                    "query-family parameter-state allocation failed: {error}"
                ))
            })?;
        replacement.extend_from_slice(parameters);
        self.parameter_state = replacement;
        self.parameter_version = self
            .parameter_version
            .checked_add(1)
            .ok_or_else(|| invalid("query-family parameter version exceeds u64"))?;
        Ok(())
    }

    /// Exact census of the row family selected by the last cold prepare.
    pub(crate) fn prepared_census(&self) -> Option<OnTheFlyQueryFamilyCensusV1> {
        self.pending
            .as_ref()
            .or_else(|| self.last_used.and_then(|index| self.families.get(index)))
            .map(|family| family.family.census)
    }

    pub(crate) const fn retained_family_count(&self) -> usize {
        self.families.len()
    }

    pub(crate) fn active_retained_handle(&self) -> Option<OnTheFlyQueryFamilyHandleV1> {
        self.pending.is_none().then_some(())?;
        Some(OnTheFlyQueryFamilyHandleV1 {
            generation: self.family_generation,
            index: self.last_used?,
        })
    }

    fn cache_identity(
        direct_catalog: &PreparedDirectExecutorCatalog,
        selected: &[QueryFamilyTraceInput<'_>],
    ) -> QueryFamilyCacheIdentityV1 {
        QueryFamilyCacheIdentityV1 {
            direct_catalog: direct_catalog.clone(),
            queries: selected
                .iter()
                .map(|selected| {
                    (
                        selected.trace.seed_digest,
                        selected.trace.query_digest,
                        selected.trace.semantic_digest(),
                        selected.projection,
                    )
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn lookup_key(selected: &[QueryFamilyTraceInput<'_>]) -> QueryFamilyLookupKeyV1 {
        QueryFamilyLookupKeyV1 {
            queries: selected
                .iter()
                .map(|selected| {
                    let projection = selected.projection;
                    (
                        selected.trace.seed_digest,
                        selected.trace.query_digest,
                        selected.trace.semantic_digest(),
                        projection.enabled,
                        projection.applied,
                        projection.pre,
                        projection.post,
                    )
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }

    fn matches_inputs(
        family: &BoundOnTheFlyQueryFamilyV1,
        direct_catalog: &PreparedDirectExecutorCatalog,
        selected: &[QueryFamilyTraceInput<'_>],
    ) -> bool {
        family.identity.direct_catalog == *direct_catalog
            && family.identity.queries.len() == selected.len()
            && family
                .identity
                .queries
                .iter()
                .zip(selected)
                .all(|(cached, selected)| {
                    *cached
                        == (
                            selected.trace.seed_digest,
                            selected.trace.query_digest,
                            selected.trace.semantic_digest(),
                            selected.projection,
                        )
                })
    }

    fn bind_groups(
        &self,
        family: &OnTheFlyBuiltQueryFamilyV1,
    ) -> RusticolResult<Box<[ResolvedOnTheFlyExecutor]>> {
        family
            .row_groups
            .iter()
            .map(|group| {
                for &(key, expected_permutation) in group.member_bindings.iter() {
                    let resolved = self.resolver.resolve(key)?;
                    if resolved.direct_executor_id != group.direct_executor_id
                        || resolved.handle.role() != group.role
                        || resolved.parent_permutation != expected_permutation
                    {
                        return Err(integrity(
                            "query-family prepared executor binding changed after row materialization",
                        ));
                    }
                }
                let resolved = self.resolver.resolve(group.representative_key)?;
                if resolved.direct_executor_id != group.direct_executor_id
                    || resolved.handle.role() != group.role
                {
                    return Err(integrity(
                        "query-family representative executor differs from its row group",
                    ));
                }
                Ok(resolved)
            })
            .collect::<RusticolResult<Vec<_>>>()
            .map(Vec::into_boxed_slice)
    }

    fn invalidate_exposed_row_tables(&mut self) -> RusticolResult<()> {
        if !self.families.iter().any(|family| family.descriptor_exposed)
            && !self
                .pending
                .as_ref()
                .is_some_and(|family| family.descriptor_exposed)
        {
            return Ok(());
        }
        self.resolver.invalidate_row_tables()?;
        for family in &mut self.families {
            family.descriptor_exposed = false;
        }
        if let Some(pending) = &mut self.pending {
            pending.descriptor_exposed = false;
        }
        Ok(())
    }

    /// Drop every retained selector family while keeping the loaded executor
    /// pool and parameter state. Descriptor invalidation happens first so no
    /// prepared kernel can retain a pointer into the rows being dropped.
    pub(crate) fn clear_families(&mut self) -> RusticolResult<()> {
        self.invalidate_exposed_row_tables()?;
        self.pending = None;
        self.last_used = None;
        self.family_lookup.clear();
        self.families.clear();
        self.family_generation = self
            .family_generation
            .checked_add(1)
            .ok_or_else(|| invalid("query-family generation exceeds u64"))?;
        Ok(())
    }

    fn activate_retained_index(
        &mut self,
        index: usize,
        logical_point_capacity: u32,
    ) -> RusticolResult<bool> {
        let replacement = self
            .families
            .get(index)
            .ok_or_else(|| integrity("query-family handle index is absent"))?
            .workspace
            .logical_point_capacity
            .lt(&logical_point_capacity)
            .then(|| {
                OnTheFlyFamilyWorkspaceV1::new(&self.families[index].family, logical_point_capacity)
            })
            .transpose()?;
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.descriptor_exposed)
            || replacement.is_some() && self.families[index].descriptor_exposed
            || self.last_used != Some(index)
                && self.families.iter().any(|family| family.descriptor_exposed)
        {
            self.invalidate_exposed_row_tables()?;
        }
        self.pending = None;
        self.last_used = Some(index);
        if let Some(replacement) = replacement {
            let family = self
                .families
                .get_mut(index)
                .expect("validated query-family handle disappeared");
            family.workspace = replacement;
            family.applied_parameter_version = u64::MAX;
        }
        Ok(true)
    }

    pub(crate) fn activate_retained_family(
        &mut self,
        handle: OnTheFlyQueryFamilyHandleV1,
        logical_point_capacity: u32,
    ) -> RusticolResult<bool> {
        if handle.generation != self.family_generation {
            return Err(integrity("query-family handle belongs to a cleared arena"));
        }
        self.activate_retained_index(handle.index, logical_point_capacity)
    }

    pub(crate) fn resize_active_family(
        &mut self,
        logical_point_capacity: u32,
    ) -> RusticolResult<bool> {
        if let Some(pending) = self.pending.as_ref() {
            if logical_point_capacity <= pending.workspace.logical_point_capacity {
                return Ok(false);
            }
            let replacement =
                OnTheFlyFamilyWorkspaceV1::new(&pending.family, logical_point_capacity)?;
            self.invalidate_exposed_row_tables()?;
            let pending = self
                .pending
                .as_mut()
                .expect("query-family pending state disappeared");
            pending.workspace = replacement;
            pending.applied_parameter_version = u64::MAX;
            return Ok(false);
        }
        let index = self
            .last_used
            .ok_or_else(|| invalid("query-family resize requires a cold prepare"))?;
        self.activate_retained_index(index, logical_point_capacity)
    }

    /// Select no executable family without dropping successfully warmed
    /// families. This is used by an all-structural-zero public selection.
    pub(crate) fn deactivate(&mut self) -> RusticolResult<()> {
        self.invalidate_exposed_row_tables()?;
        self.pending = None;
        self.last_used = None;
        Ok(())
    }

    /// Cold family selection/binding. This must remain outside warmed timing.
    pub(crate) fn prepare(
        &mut self,
        direct_catalog: &PreparedDirectExecutorCatalog,
        selected: &[QueryFamilyTraceInput<'_>],
        logical_point_capacity: u32,
    ) -> RusticolResult<bool> {
        // The common warmed path checks the last exact family directly. It
        // does not allocate a lookup key or consult the all-seen index.
        let retained = self
            .last_used
            .filter(|&index| {
                self.families
                    .get(index)
                    .is_some_and(|family| Self::matches_inputs(family, direct_catalog, selected))
            })
            .or_else(|| {
                let lookup_key = Self::lookup_key(selected);
                self.family_lookup.get(&lookup_key).and_then(|indices| {
                    indices.iter().copied().find(|&index| {
                        self.families.get(index).is_some_and(|family| {
                            Self::matches_inputs(family, direct_catalog, selected)
                        })
                    })
                })
            });
        if let Some(index) = retained {
            return self.activate_retained_index(index, logical_point_capacity);
        }

        let identity = Self::cache_identity(direct_catalog, selected);
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.identity == identity)
        {
            let replacement = self
                .pending
                .as_ref()
                .filter(|pending| logical_point_capacity > pending.workspace.logical_point_capacity)
                .map(|pending| {
                    OnTheFlyFamilyWorkspaceV1::new(&pending.family, logical_point_capacity)
                })
                .transpose()?;
            if let Some(replacement) = replacement {
                self.invalidate_exposed_row_tables()?;
                let pending = self
                    .pending
                    .as_mut()
                    .expect("matching pending query family disappeared");
                pending.workspace = replacement;
                pending.applied_parameter_version = u64::MAX;
            }
            return Ok(false);
        }

        let family = build_query_family_from_traces(direct_catalog, selected)?;
        if family.census.source_frame_partition_count != 1 {
            return Err(invalid(
                "executable query families currently require one compact source frame",
            ));
        }
        let workspace = OnTheFlyFamilyWorkspaceV1::new(&family, logical_point_capacity)?;
        let resolved_groups = self.bind_groups(&family)?;
        self.families.try_reserve(1).map_err(|error| {
            invalid(format!(
                "query-family retained-family allocation failed: {error}"
            ))
        })?;
        let candidate = BoundOnTheFlyQueryFamilyV1 {
            identity,
            lookup_key: Self::lookup_key(selected),
            family,
            workspace,
            resolved_groups,
            applied_parameter_version: u64::MAX,
            descriptor_exposed: false,
        };
        // Keep every old pointer owner alive until both the replacement rows
        // and workspace exist, then clear every prepared SymJIT row table
        // before exposing the pending candidate.
        self.invalidate_exposed_row_tables()?;
        self.pending = Some(candidate);
        Ok(false)
    }

    /// Allocation-free warmed execution after one cold [`Self::prepare`] call.
    pub(crate) fn execute_into(
        &mut self,
        external_momenta: &[f64],
        point_count: u32,
        outputs: &mut [(f64, f64)],
    ) -> RusticolResult<OnTheFlyQueryFamilyExecutionReportV1> {
        let pending = self.pending.is_some();
        let family = if pending {
            self.pending
                .as_mut()
                .expect("pending query family disappeared before execution")
        } else {
            self.last_used
                .and_then(|index| self.families.get_mut(index))
                .ok_or_else(|| invalid("query-family execute requires a cold prepare"))?
        };
        OnTheFlyFamilyWorkspaceV1::validate_output_shape(
            family.family.amplitude_destination_count,
            point_count,
            outputs.len(),
        )?;
        if family.applied_parameter_version != self.parameter_version {
            family.workspace.refresh_parameters(&self.parameter_state)?;
            family.applied_parameter_version = self.parameter_version;
        }
        family
            .workspace
            .refresh_inputs(&family.family, external_momenta, point_count)?;
        family.descriptor_exposed = true;
        let mut report = execute_bound_family(family, point_count)?;
        report.cache_hit = !pending;
        family.workspace.write_outputs(
            family.family.amplitude_destination_count,
            point_count,
            outputs,
        )?;
        if pending {
            let candidate = self
                .pending
                .take()
                .expect("successful pending query family disappeared");
            let lookup_key = candidate.lookup_key.clone();
            let index = self.families.len();
            self.families.push(candidate);
            self.family_lookup
                .entry(lookup_key)
                .or_default()
                .push(index);
            self.last_used = Some(index);
        }
        Ok(report)
    }

    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(crate) fn observed_currents(
        &self,
        point_index: u32,
    ) -> RusticolResult<Vec<OnTheFlyFamilyObservedCurrentV1>> {
        let family = self
            .pending
            .as_ref()
            .or_else(|| self.last_used.and_then(|index| self.families.get(index)))
            .ok_or_else(|| invalid("query-family observation requires a prepared execution"))?;
        family
            .workspace
            .observed_currents(&family.family.observed_currents, point_index)
    }
}

impl<R: OnTheFlyPreparedExecutorResolver> Drop for OnTheFlyQueryFamilyExecutorV1<R> {
    fn drop(&mut self) {
        if self.invalidate_exposed_row_tables().is_ok() {
            return;
        }
        // Fail closed if a prepared context is unexpectedly borrowed during
        // teardown: leaked private storage is safer than dangling row tables.
        for family in self.families.drain(..) {
            if family.descriptor_exposed {
                std::mem::forget(family);
            }
        }
        if self
            .pending
            .as_ref()
            .is_some_and(|family| family.descriptor_exposed)
        {
            if let Some(pending) = self.pending.take() {
                std::mem::forget(pending);
            }
        }
    }
}

fn execute_bound_family(
    cached: &mut BoundOnTheFlyQueryFamilyV1,
    point_count: u32,
) -> RusticolResult<OnTheFlyQueryFamilyExecutionReportV1> {
    let (arena, momenta, parameters, factors) = cached.workspace.raw_views()?;
    let mut report = OnTheFlyQueryFamilyExecutionReportV1::default();
    for (group, resolved) in cached
        .family
        .row_groups
        .iter()
        .zip(cached.resolved_groups.iter().copied())
    {
        clear_direct_executor_error_detail();
        let row_count = checked_u32(group.rows.len(), "query-family row-group count")?;
        match group.role {
            DirectExecutorRole::Source => {
                add(&mut report.source_calls, 1, "executed source calls")?;
                add(&mut report.source_rows, row_count, "executed source rows")?;
            }
            DirectExecutorRole::Contribution => {
                add(
                    &mut report.contribution_calls,
                    1,
                    "executed contribution calls",
                )?;
                add(
                    &mut report.contribution_rows,
                    row_count,
                    "executed contribution rows",
                )?;
            }
            DirectExecutorRole::Finalization => {
                add(
                    &mut report.finalization_calls,
                    1,
                    "executed finalization calls",
                )?;
                add(
                    &mut report.finalization_rows,
                    row_count,
                    "executed finalization rows",
                )?;
            }
            DirectExecutorRole::Closure => {
                add(&mut report.closure_calls, 1, "executed closure calls")?;
                add(&mut report.closure_rows, row_count, "executed closure rows")?;
            }
        }
        let status: c_int = unsafe {
            match (&group.rows, resolved.handle) {
                (
                    OnTheFlyFamilyRowsV1::Source(rows),
                    DirectExecutorHandle::Source { call, context },
                ) => call(
                    context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    rows.as_ptr(),
                    row_count,
                    point_count,
                ),
                (
                    OnTheFlyFamilyRowsV1::Contribution(rows),
                    DirectExecutorHandle::Contribution { call, context },
                ) => call(
                    context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    rows.as_ptr(),
                    row_count,
                    point_count,
                ),
                (
                    OnTheFlyFamilyRowsV1::Finalization(rows),
                    DirectExecutorHandle::Finalization { call, context },
                ) => call(
                    context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    rows.as_ptr(),
                    row_count,
                    point_count,
                ),
                (
                    OnTheFlyFamilyRowsV1::Closure(rows),
                    DirectExecutorHandle::Closure { call, context },
                ) => call(
                    context,
                    arena,
                    momenta,
                    parameters,
                    factors,
                    rows.as_ptr(),
                    row_count,
                    point_count,
                ),
                _ => {
                    return Err(integrity(
                        "query-family resolved handle differs from its row storage",
                    ));
                }
            }
        };
        crate::recurrence::direct_backend::check_status(
            group.role,
            group.direct_executor_id,
            status,
        )?;
    }
    cached.workspace.active_point_count = point_count;
    Ok(report)
}

/// Build exactly the requested LC queries with the existing query-local
/// machinery, then compute their cold structural union census.
#[doc(hidden)]
#[cfg(any(test, feature = "on-the-fly-test-support"))]
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
                direct_catalog,
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
    Ok(build_query_family_from_traces(direct_catalog, &traces)?.census)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::{
        ExactRational, PreparedDirectExecutorBinding, PreparedDirectExecutorCatalog,
    };
    use std::cell::{Cell, RefCell};
    use std::ffi::c_void;

    const TEST_STATUS_BOUNDS: c_int = 2;
    const TEST_STATUS_FAILED: c_int = 4;

    #[derive(Default)]
    struct ProbeState {
        calls: RefCell<Vec<(DirectExecutorRole, usize, u32)>>,
        fail_role: Cell<Option<DirectExecutorRole>>,
        invalidations: Cell<u32>,
    }

    impl ProbeState {
        fn record(&self, role: DirectExecutorRole, rows: *const c_void, row_count: u32) -> bool {
            self.calls
                .borrow_mut()
                .push((role, rows as usize, row_count));
            self.fail_role.get() == Some(role)
        }
    }

    unsafe fn probe_state<'a>(context: *const c_void) -> &'a ProbeState {
        unsafe { &*context.cast::<ProbeState>() }
    }

    unsafe extern "C" fn probe_sources(
        context: *const c_void,
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        _parameters: DirectParameterView,
        factors: DirectFactorView,
        rows: *const DirectSourceRow,
        row_count: u32,
        point_count: u32,
    ) -> c_int {
        let state = unsafe { probe_state(context) };
        if state.record(DirectExecutorRole::Source, rows.cast(), row_count) {
            return TEST_STATUS_FAILED;
        }
        let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
        for row in rows {
            if row.exact_factor_id >= factors.value_count {
                return TEST_STATUS_BOUNDS;
            }
            let factor = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
            for point in 0..point_count as usize {
                let source = row.momentum_form_id as usize
                    * momenta.lorentz_component_count as usize
                    * momenta.point_stride as usize
                    + point;
                let destination =
                    row.destination_component_base as usize * arena.point_stride as usize + point;
                if source >= momenta.scalar_len as usize
                    || destination >= arena.current_scalar_len as usize
                {
                    return TEST_STATUS_BOUNDS;
                }
                unsafe {
                    *arena.current_re.add(destination) = *momenta.values.add(source) * factor;
                    *arena.current_im.add(destination) = 0.0;
                }
            }
        }
        crate::recurrence::direct_backend::DIRECT_STATUS_OK
    }

    unsafe extern "C" fn probe_contributions(
        context: *const c_void,
        arena: DirectArenaView,
        _momenta: DirectMomentumView,
        _parameters: DirectParameterView,
        factors: DirectFactorView,
        rows: *const DirectContributionRow,
        row_count: u32,
        point_count: u32,
    ) -> c_int {
        let state = unsafe { probe_state(context) };
        if state.record(DirectExecutorRole::Contribution, rows.cast(), row_count) {
            return TEST_STATUS_FAILED;
        }
        let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
        for row in rows {
            if row.exact_factor_id >= factors.value_count {
                return TEST_STATUS_BOUNDS;
            }
            let factor = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
            for point in 0..point_count as usize {
                let parent0 =
                    row.parent0_component_base as usize * arena.point_stride as usize + point;
                let parent1 = row.parent1_component_base_or_sentinel as usize
                    * arena.point_stride as usize
                    + point;
                let destination =
                    row.destination_component_base as usize * arena.point_stride as usize + point;
                if parent0 >= arena.current_scalar_len as usize
                    || parent1 >= arena.current_scalar_len as usize
                    || destination >= arena.current_scalar_len as usize
                {
                    return TEST_STATUS_BOUNDS;
                }
                let contribution = unsafe {
                    (10.0 * *arena.current_re.add(parent0) + *arena.current_re.add(parent1))
                        * factor
                };
                unsafe {
                    if row.flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION != 0 {
                        *arena.current_re.add(destination) = contribution;
                        *arena.current_im.add(destination) = 0.0;
                    } else {
                        *arena.current_re.add(destination) += contribution;
                    }
                }
            }
        }
        crate::recurrence::direct_backend::DIRECT_STATUS_OK
    }

    unsafe extern "C" fn probe_finalizations(
        context: *const c_void,
        arena: DirectArenaView,
        _momenta: DirectMomentumView,
        parameters: DirectParameterView,
        factors: DirectFactorView,
        rows: *const DirectFinalizationRow,
        row_count: u32,
        point_count: u32,
    ) -> c_int {
        let state = unsafe { probe_state(context) };
        if state.record(DirectExecutorRole::Finalization, rows.cast(), row_count) {
            return TEST_STATUS_FAILED;
        }
        let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
        let parameter = if parameters.value_count == 0 {
            1.0
        } else {
            unsafe { *parameters.values_re }
        };
        for row in rows {
            if row.exact_factor_id >= factors.value_count {
                return TEST_STATUS_BOUNDS;
            }
            let factor = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
            for component in 0..usize::from(row.component_count) {
                for point in 0..point_count as usize {
                    let destination = (row.component_base as usize + component)
                        * arena.point_stride as usize
                        + point;
                    if destination >= arena.current_scalar_len as usize {
                        return TEST_STATUS_BOUNDS;
                    }
                    unsafe {
                        *arena.current_re.add(destination) *= factor * parameter;
                        *arena.current_im.add(destination) *= factor * parameter;
                    }
                }
            }
        }
        crate::recurrence::direct_backend::DIRECT_STATUS_OK
    }

    unsafe extern "C" fn probe_closures(
        context: *const c_void,
        arena: DirectArenaView,
        _momenta: DirectMomentumView,
        _parameters: DirectParameterView,
        factors: DirectFactorView,
        rows: *const DirectClosureRow,
        row_count: u32,
        point_count: u32,
    ) -> c_int {
        let state = unsafe { probe_state(context) };
        if state.record(DirectExecutorRole::Closure, rows.cast(), row_count) {
            return TEST_STATUS_FAILED;
        }
        let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
        for row in rows {
            let factor_end = match row
                .component_factor_start
                .checked_add(u32::from(row.component_count))
            {
                Some(value) => value,
                None => return TEST_STATUS_BOUNDS,
            };
            if row.exact_factor_id >= factors.value_count || factor_end > factors.value_count {
                return TEST_STATUS_BOUNDS;
            }
            let common = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
            for component in 0..usize::from(row.component_count) {
                let component_factor = unsafe {
                    *factors
                        .values_re
                        .add(row.component_factor_start as usize + component)
                };
                for point in 0..point_count as usize {
                    let parent0 = (row.parent0_component_base as usize + component)
                        * arena.point_stride as usize
                        + point;
                    let parent1 = (row.parent1_component_base_or_sentinel as usize + component)
                        * arena.point_stride as usize
                        + point;
                    let destination =
                        row.amplitude_destination_id as usize * arena.point_stride as usize + point;
                    if parent0 >= arena.current_scalar_len as usize
                        || parent1 >= arena.current_scalar_len as usize
                        || destination >= arena.amplitude_scalar_len as usize
                    {
                        return TEST_STATUS_BOUNDS;
                    }
                    unsafe {
                        *arena.amplitude_re.add(destination) += *arena.current_re.add(parent0)
                            * *arena.current_re.add(parent1)
                            * common
                            * component_factor;
                    }
                }
            }
        }
        crate::recurrence::direct_backend::DIRECT_STATUS_OK
    }

    struct ProbeResolver {
        catalog: PreparedDirectExecutorCatalog,
        state: Box<ProbeState>,
    }

    impl ProbeResolver {
        fn new(catalog: PreparedDirectExecutorCatalog) -> Self {
            Self {
                catalog,
                state: Box::new(ProbeState::default()),
            }
        }
    }

    impl OnTheFlyPreparedExecutorResolver for ProbeResolver {
        fn resolve(&self, key: OnTheFlyExecutorKeyV1) -> RusticolResult<ResolvedOnTheFlyExecutor> {
            let (direct_executor_id, parent_permutation) = prepared_binding(&self.catalog, key)?;
            let context = (&*self.state as *const ProbeState).cast::<c_void>();
            let handle = match key.role() {
                DirectExecutorRole::Source => DirectExecutorHandle::Source {
                    call: probe_sources,
                    context,
                },
                DirectExecutorRole::Contribution => DirectExecutorHandle::Contribution {
                    call: probe_contributions,
                    context,
                },
                DirectExecutorRole::Finalization => DirectExecutorHandle::Finalization {
                    call: probe_finalizations,
                    context,
                },
                DirectExecutorRole::Closure => DirectExecutorHandle::Closure {
                    call: probe_closures,
                    context,
                },
            };
            Ok(ResolvedOnTheFlyExecutor {
                direct_executor_id,
                handle,
                parent_permutation,
            })
        }

        fn invalidate_row_tables(&self) -> RusticolResult<()> {
            self.state
                .invalidations
                .set(self.state.invalidations.get() + 1);
            Ok(())
        }
    }

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
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Contribution, 5, 4),
                PreparedDirectExecutorBinding::evaluator_with_parent_permutation(
                    DirectExecutorRole::Contribution,
                    6,
                    1,
                    [1, 0],
                ),
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
        assert_eq!(census.union_kernel_application_count().unwrap(), 6);
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

    fn one_point_momenta(source0: f64, source1: f64) -> Vec<f64> {
        vec![source0, 0.0, 0.0, 0.0, source1, 0.0, 0.0, 0.0]
    }

    #[test]
    fn query_family_executes_union_once_and_reuses_stable_warm_rows() {
        let catalog = direct_catalog();
        let resolver = ProbeResolver::new(catalog.clone());
        let (mut first, first_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        let (mut second, second_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x82, factor(1));
        first.layout.parameter_count = 1;
        second.layout.parameter_count = 1;
        second.test_remap_dynamic_color_id(7);
        let selected = [
            QueryFamilyTraceInput {
                trace: &first,
                projection: first_projection,
            },
            QueryFamilyTraceInput {
                trace: &second,
                projection: second_projection,
            },
        ];
        let mut executor = OnTheFlyQueryFamilyExecutorV1::new(resolver);
        assert!(!executor.prepare(&catalog, &selected, 1).unwrap());
        let census = executor.prepared_census().unwrap();
        assert_eq!(census.union_unique_current_count, 3);
        assert_eq!(census.union_amplitude_destination_count, 2);
        executor.set_parameters(&[(1.0, 0.0)]).unwrap();
        let mut outputs = [(0.0, 0.0); 2];
        let cold = executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut outputs)
            .unwrap();
        assert!(!cold.cache_hit);
        assert_eq!(
            cold,
            OnTheFlyQueryFamilyExecutionReportV1 {
                cache_hit: false,
                source_calls: 1,
                source_rows: 2,
                contribution_calls: 1,
                contribution_rows: 1,
                finalization_calls: 1,
                finalization_rows: 1,
                closure_calls: 1,
                closure_rows: 2,
            }
        );
        assert_eq!(outputs, [(46.0, 0.0), (46.0, 0.0)]);
        let first_calls = executor.resolver().state.calls.borrow().clone();
        assert_eq!(
            first_calls.iter().map(|call| call.0).collect::<Vec<_>>(),
            vec![
                DirectExecutorRole::Source,
                DirectExecutorRole::Contribution,
                DirectExecutorRole::Finalization,
                DirectExecutorRole::Closure,
            ]
        );

        assert!(executor.prepare(&catalog, &selected, 1).unwrap());
        executor.set_parameters(&[(2.0, 0.0)]).unwrap();
        let warm = executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut outputs)
            .unwrap();
        assert!(warm.cache_hit);
        assert_eq!(outputs, [(92.0, 0.0), (92.0, 0.0)]);
        let calls = executor.resolver().state.calls.borrow();
        assert_eq!(calls.len(), 8);
        assert_eq!(
            calls[..4]
                .iter()
                .map(|call| (call.0, call.1, call.2))
                .collect::<Vec<_>>(),
            calls[4..]
                .iter()
                .map(|call| (call.0, call.1, call.2))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn query_family_interns_shared_scalars_and_exact_component_blocks() {
        let mut factors = Vec::new();
        let mut scalar_ids = BTreeMap::new();
        let mut block_ids = BTreeMap::new();
        let one = factor(1);
        let two = factor(2);
        assert_eq!(
            intern_factor(&mut scalar_ids, &mut factors, one).unwrap(),
            0
        );
        assert_eq!(
            intern_factor(&mut scalar_ids, &mut factors, one).unwrap(),
            0
        );
        let forward =
            intern_factor_block(&mut block_ids, &mut scalar_ids, &mut factors, &[one, two])
                .unwrap();
        assert_eq!(forward, 1);
        assert_eq!(
            intern_factor_block(&mut block_ids, &mut scalar_ids, &mut factors, &[one, two],)
                .unwrap(),
            forward
        );
        let reversed =
            intern_factor_block(&mut block_ids, &mut scalar_ids, &mut factors, &[two, one])
                .unwrap();
        assert_eq!(reversed, 3);
        assert_eq!(factors, vec![one, one, two, two, one]);

        let catalog = direct_catalog();
        let (first, first_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, one);
        let (mut second, second_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x82, one);
        second.test_remap_dynamic_color_id(7);
        let family = build_query_family_from_traces(
            &catalog,
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
        assert_eq!(family.exact_factors.as_ref(), &[one, one]);
        let starts = family
            .row_groups
            .iter()
            .flat_map(|group| match &group.rows {
                OnTheFlyFamilyRowsV1::Closure(rows) => rows
                    .iter()
                    .map(|row| row.component_factor_start)
                    .collect::<Vec<_>>(),
                _ => Vec::new(),
            })
            .collect::<Vec<_>>();
        assert_eq!(starts, vec![1, 1]);
    }

    #[test]
    fn query_family_initializes_once_across_executor_groups() {
        let catalog = direct_catalog();
        let resolver = ProbeResolver::new(catalog.clone());
        let (mut trace, mut projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        trace.test_add_query_family_contribution(5, 5, factor(1));
        projection.pre[1] = 2;
        projection.post[1] = 2;
        let selected = [QueryFamilyTraceInput {
            trace: &trace,
            projection,
        }];
        let family = build_query_family_from_traces(&catalog, &selected).unwrap();
        let contribution_groups = family
            .row_groups
            .iter()
            .filter_map(|group| match &group.rows {
                OnTheFlyFamilyRowsV1::Contribution(rows) => Some(rows.as_ref()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(contribution_groups.len(), 2);
        assert_eq!(
            contribution_groups
                .iter()
                .flat_map(|rows| rows.iter())
                .filter(|row| row.flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION != 0)
                .count(),
            1
        );

        let mut executor = OnTheFlyQueryFamilyExecutorV1::new(resolver);
        assert!(!executor.prepare(&catalog, &selected, 1).unwrap());
        let mut output = [(0.0, 0.0)];
        let report = executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut output)
            .unwrap();
        assert_eq!(report.contribution_calls, 2);
        assert_eq!(report.contribution_rows, 2);
        assert_eq!(output, [(92.0, 0.0)]);
    }

    #[test]
    fn query_family_applies_each_prepared_parent_permutation_before_grouping() {
        let catalog = direct_catalog();
        let resolver = ProbeResolver::new(catalog.clone());
        let (mut trace, mut projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        trace.test_add_query_family_contribution(6, 6, factor(1));
        trace.test_set_last_contribution_selector_domain(17);
        projection.pre[1] = 2;
        projection.post[1] = 2;
        let selected = [QueryFamilyTraceInput {
            trace: &trace,
            projection,
        }];
        let family = build_query_family_from_traces(&catalog, &selected).unwrap();
        let rows = family
            .row_groups
            .iter()
            .find_map(|group| match &group.rows {
                OnTheFlyFamilyRowsV1::Contribution(rows) => Some(rows.as_ref()),
                _ => None,
            })
            .unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(
            rows.iter()
                .map(|row| row.selector_domain_id)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([0, 17])
        );
        assert_eq!(
            rows.iter()
                .map(|row| {
                    (
                        row.parent0_component_base,
                        row.parent1_component_base_or_sentinel,
                    )
                })
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([(0, 1), (1, 0)])
        );

        let mut executor = OnTheFlyQueryFamilyExecutorV1::new(resolver);
        executor.prepare(&catalog, &selected, 1).unwrap();
        let mut output = [(0.0, 0.0)];
        let report = executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut output)
            .unwrap();
        assert_eq!(report.contribution_calls, 1);
        assert_eq!(report.contribution_rows, 2);
        assert_eq!(output, [(110.0, 0.0)]);
    }

    #[test]
    fn query_family_rejects_non_topological_group_order() {
        let catalog = direct_catalog();
        let (trace, projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        let mut family = build_query_family_from_traces(
            &catalog,
            &[QueryFamilyTraceInput {
                trace: &trace,
                projection,
            }],
        )
        .unwrap();
        let contribution = family
            .row_groups
            .iter()
            .position(|group| group.role == DirectExecutorRole::Contribution)
            .unwrap();
        let finalization = family
            .row_groups
            .iter()
            .position(|group| group.role == DirectExecutorRole::Finalization)
            .unwrap();
        family.row_groups.swap(contribution, finalization);
        let error = validate_ordered_family_schedule(&family.row_groups).unwrap_err();
        assert!(error.to_string().contains("follows finalization"));
    }

    #[test]
    fn failed_execution_invalidates_outputs_and_same_family_can_retry() {
        let catalog = direct_catalog();
        let resolver = ProbeResolver::new(catalog.clone());
        let (trace, projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        let selected = [QueryFamilyTraceInput {
            trace: &trace,
            projection,
        }];
        let mut executor = OnTheFlyQueryFamilyExecutorV1::new(resolver);
        executor.prepare(&catalog, &selected, 1).unwrap();
        executor
            .resolver_mut()
            .state
            .fail_role
            .set(Some(DirectExecutorRole::Contribution));
        let mut output = [(123.0, 456.0)];
        let error = executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut output)
            .unwrap_err();
        assert!(error.to_string().contains("returned status 4"));
        assert_eq!(output, [(123.0, 456.0)]);
        let pending = executor.pending.as_ref().unwrap();
        assert!(
            pending
                .workspace
                .write_outputs(1, 1, &mut output)
                .unwrap_err()
                .to_string()
                .contains("last successful execution")
        );

        executor.resolver().state.fail_role.set(None);
        let report = executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut output)
            .unwrap();
        assert!(!report.cache_hit);
        assert_eq!(output, [(46.0, 0.0)]);
    }

    #[test]
    fn partially_failed_family_is_invalidated_before_switching_rows() {
        let catalog = direct_catalog();
        let resolver = ProbeResolver::new(catalog.clone());
        let (first, first_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        let first_selected = [QueryFamilyTraceInput {
            trace: &first,
            projection: first_projection,
        }];
        let mut executor = OnTheFlyQueryFamilyExecutorV1::new(resolver);
        executor.prepare(&catalog, &first_selected, 1).unwrap();
        executor
            .resolver()
            .state
            .fail_role
            .set(Some(DirectExecutorRole::Finalization));
        let mut output = [(0.0, 0.0)];
        executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut output)
            .unwrap_err();
        assert_eq!(executor.retained_family_count(), 0);
        let failed_calls = executor.resolver().state.calls.borrow().clone();
        assert_eq!(failed_calls.len(), 3);
        assert_eq!(executor.resolver().state.invalidations.get(), 0);

        let (second, second_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x82, factor(1));
        let second_selected = [QueryFamilyTraceInput {
            trace: &second,
            projection: second_projection,
        }];
        assert!(!executor.prepare(&catalog, &second_selected, 1).unwrap());
        assert_eq!(executor.resolver().state.invalidations.get(), 1);
        executor.resolver().state.fail_role.set(None);
        executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut output)
            .unwrap();
        assert_eq!(executor.retained_family_count(), 1);
        assert_eq!(output, [(46.0, 0.0)]);
        let calls = executor.resolver().state.calls.borrow();
        assert_ne!(failed_calls[0].1, calls[3].1);
        assert_ne!(failed_calls[1].1, calls[4].1);
    }

    #[test]
    fn successful_cached_family_is_invalidated_before_new_candidate_runs() {
        let catalog = direct_catalog();
        let resolver = ProbeResolver::new(catalog.clone());
        let (first, first_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        let first_selected = [QueryFamilyTraceInput {
            trace: &first,
            projection: first_projection,
        }];
        let mut executor = OnTheFlyQueryFamilyExecutorV1::new(resolver);
        executor.prepare(&catalog, &first_selected, 1).unwrap();
        let mut output = [(0.0, 0.0)];
        executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut output)
            .unwrap();
        assert_eq!(executor.resolver().state.invalidations.get(), 0);
        let calls_before_prepare = executor.resolver().state.calls.borrow().len();

        let (second, second_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x82, factor(1));
        let second_selected = [QueryFamilyTraceInput {
            trace: &second,
            projection: second_projection,
        }];
        executor.prepare(&catalog, &second_selected, 1).unwrap();
        assert_eq!(executor.resolver().state.invalidations.get(), 1);
        assert_eq!(
            executor.resolver().state.calls.borrow().len(),
            calls_before_prepare
        );
        assert!(
            executor
                .families
                .iter()
                .all(|family| !family.descriptor_exposed)
        );
        let report = executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut output)
            .unwrap();
        assert!(!report.cache_hit);
        assert_eq!(output, [(46.0, 0.0)]);
    }

    #[test]
    fn observed_currents_never_fall_back_to_a_stale_successful_family() {
        let catalog = direct_catalog();
        let resolver = ProbeResolver::new(catalog.clone());
        let (first, first_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        let first_selected = [QueryFamilyTraceInput {
            trace: &first,
            projection: first_projection,
        }];
        let mut executor = OnTheFlyQueryFamilyExecutorV1::new(resolver);
        let mut output = [(0.0, 0.0)];
        executor.prepare(&catalog, &first_selected, 1).unwrap();
        executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut output)
            .unwrap();
        assert!(!executor.observed_currents(0).unwrap().is_empty());

        let (second, second_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x82, factor(2));
        let second_selected = [QueryFamilyTraceInput {
            trace: &second,
            projection: second_projection,
        }];
        executor.prepare(&catalog, &second_selected, 1).unwrap();
        let prepared_error = executor.observed_currents(0).unwrap_err();
        assert!(
            prepared_error
                .to_string()
                .contains("outside the last successful execution")
        );

        executor
            .resolver()
            .state
            .fail_role
            .set(Some(DirectExecutorRole::Contribution));
        executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut output)
            .unwrap_err();
        let failed_error = executor.observed_currents(0).unwrap_err();
        assert!(
            failed_error
                .to_string()
                .contains("outside the last successful execution")
        );

        executor.clear_families().unwrap();
        assert!(
            executor
                .observed_currents(0)
                .unwrap_err()
                .to_string()
                .contains("requires a prepared execution")
        );
    }

    #[test]
    fn successful_families_are_reused_across_a_b_a_with_stable_rows() {
        let catalog = direct_catalog();
        let resolver = ProbeResolver::new(catalog.clone());
        let (first, first_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        let first_selected = [QueryFamilyTraceInput {
            trace: &first,
            projection: first_projection,
        }];
        let (second, second_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x82, factor(2));
        let second_selected = [QueryFamilyTraceInput {
            trace: &second,
            projection: second_projection,
        }];
        let mut executor = OnTheFlyQueryFamilyExecutorV1::new(resolver);
        let mut output = [(0.0, 0.0)];

        assert!(!executor.prepare(&catalog, &first_selected, 1).unwrap());
        executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut output)
            .unwrap();
        let first_output = output;
        let first_rows = executor.resolver().state.calls.borrow().clone();
        let first_handle = executor.active_retained_handle().unwrap();
        executor.resolver().state.calls.borrow_mut().clear();

        assert!(!executor.prepare(&catalog, &second_selected, 1).unwrap());
        executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut output)
            .unwrap();
        assert_ne!(output, first_output);
        executor.resolver().state.calls.borrow_mut().clear();

        assert!(executor.activate_retained_family(first_handle, 1).unwrap());
        let report = executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut output)
            .unwrap();
        assert!(report.cache_hit);
        assert_eq!(output, first_output);
        assert_eq!(executor.retained_family_count(), 2);
        assert_eq!(*executor.resolver().state.calls.borrow(), first_rows);
        assert_eq!(executor.resolver().state.invalidations.get(), 2);
    }

    #[test]
    fn clear_invalidates_rows_before_dropping_all_retained_families() {
        let catalog = direct_catalog();
        let resolver = ProbeResolver::new(catalog.clone());
        let (trace, projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        let selected = [QueryFamilyTraceInput {
            trace: &trace,
            projection,
        }];
        let mut executor = OnTheFlyQueryFamilyExecutorV1::new(resolver);
        let mut output = [(0.0, 0.0)];
        executor.prepare(&catalog, &selected, 1).unwrap();
        executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut output)
            .unwrap();
        assert_eq!(executor.retained_family_count(), 1);
        let handle = executor.active_retained_handle().unwrap();

        executor.clear_families().unwrap();
        assert_eq!(executor.resolver().state.invalidations.get(), 1);
        assert_eq!(executor.retained_family_count(), 0);
        assert!(executor.prepared_census().is_none());
        assert!(
            executor
                .activate_retained_family(handle, 1)
                .unwrap_err()
                .to_string()
                .contains("cleared arena")
        );
        assert!(!executor.prepare(&catalog, &selected, 1).unwrap());
    }

    #[test]
    fn wrong_output_shape_fails_before_rows_are_exposed() {
        let catalog = direct_catalog();
        let resolver = ProbeResolver::new(catalog.clone());
        let (trace, projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        let selected = [QueryFamilyTraceInput {
            trace: &trace,
            projection,
        }];
        let mut executor = OnTheFlyQueryFamilyExecutorV1::new(resolver);
        executor.prepare(&catalog, &selected, 1).unwrap();
        let error = executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut [])
            .unwrap_err();
        assert!(error.to_string().contains("expected 1"));
        assert!(executor.resolver().state.calls.borrow().is_empty());
        assert!(!executor.pending.as_ref().unwrap().descriptor_exposed);

        let mut output = [(0.0, 0.0)];
        executor
            .execute_into(&one_point_momenta(2.0, 3.0), 1, &mut output)
            .unwrap();
        assert_eq!(output, [(46.0, 0.0)]);
    }

    #[test]
    fn executable_query_family_rejects_multiple_source_frames() {
        let catalog = direct_catalog();
        let resolver = ProbeResolver::new(catalog.clone());
        let (first, first_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x81, factor(1));
        let (mut second, second_projection) =
            OnTheFlyStructuralTraceV1::test_query_family_trace(0x82, factor(1));
        second.seed_digest = SemanticDigest::new([0x83; 32]).unwrap();
        let mut executor = OnTheFlyQueryFamilyExecutorV1::new(resolver);
        let error = executor
            .prepare(
                &catalog,
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
                1,
            )
            .unwrap_err();
        assert!(error.to_string().contains("one compact source frame"));
    }
}
