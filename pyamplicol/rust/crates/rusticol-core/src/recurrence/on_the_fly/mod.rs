// SPDX-License-Identifier: 0BSD

//! Private query-local LC recurrence construction and direct interpretation.
//!
//! This module deliberately consumes one decoded physical LC selector.  It
//! never constructs a process-wide color plan, selector projection, replay
//! catalog, feasibility index, or [`super::DirectRecurrencePlan`].

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ffi::c_int;

use sha2::{Digest, Sha256};

use super::construct::{
    TemplateCatalog, aggregate_factor, combined_coupling_orders, merged_helicity_identity,
    merged_momentum, multiply_factors, output_factor_from_binding, quantum_parent_spin_matches,
    validate_crossed_source_state,
};
use super::contact_orbit_owner::{
    PreparedContactOrbitTransition, prepare_contact_orbit_transition,
    selected_contact_orbit_owner_tokens,
};
use super::direct_backend::{DirectExecutorHandle, clear_direct_executor_error_detail};
use super::direct_plan::{
    DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION, DirectClosureRow, DirectContributionRow,
    DirectExecutorRole, DirectFinalizationRow, DirectSourceRow,
};
use super::process::ProcessSourceStateRow;
use super::template::{
    ClosureRow, EvaluatorContractKind, LCColorTransitionWitnessRow, QuantumFlowRow, SourceRow,
    TransitionRow, ValidatedRecurrenceTemplateInput,
};
use super::{
    CanonicalMomentumLinearForm, ContributionKey, CurrentCoreKey, CurrentHelicityIdentity,
    CurrentSourceBinding, DynamicLCColorState, DynamicLCColorStateInterner, ExactComplexRational,
    LCColorComponent, LCColorComponentKind, LCColorWitnessTermId, MomentumTerm,
    PreparedDirectExecutorBinding, PreparedDirectExecutorCatalog, RecurrenceNodeKind,
    SemanticDigest, SourceStateAssignment,
};
use crate::direct_arena::{
    AlignedF64Buffer, DirectArenaView, DirectFactorView, DirectMomentumView, DirectParameterView,
    checked_aligned_point_stride, validate_direct_views,
};
use crate::{RusticolError, RusticolResult};

const MISSING_U32: u32 = u32::MAX;
const ON_THE_FLY_PROOF_DOMAIN: &[u8] = b"pyamplicol-on-the-fly-structural-proof-v1\0";

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("on-the-fly recurrence: {}", message.into()))
}

fn integrity(message: impl Into<String>) -> RusticolError {
    RusticolError::integrity(format!("on-the-fly recurrence: {}", message.into()))
}

fn evaluation(message: impl Into<String>) -> RusticolError {
    RusticolError::evaluation(format!("on-the-fly recurrence: {}", message.into()))
}

fn checked_u32(value: usize, label: &str) -> RusticolResult<u32> {
    u32::try_from(value).map_err(|_| invalid(format!("{label} exceeds u32")))
}

fn hash_len(hash: &mut Sha256, value: usize, label: &str) -> RusticolResult<()> {
    hash.update(
        u64::try_from(value)
            .map_err(|_| invalid(format!("{label} length exceeds u64")))?
            .to_le_bytes(),
    );
    Ok(())
}

fn hash_digest(hash: &mut Sha256, digest: SemanticDigest) {
    hash.update(digest.as_bytes());
}

fn hash_exact(hash: &mut Sha256, value: ExactComplexRational) {
    for part in [value.real(), value.imag()] {
        hash.update(part.numerator().to_le_bytes());
        hash.update(part.denominator().to_le_bytes());
    }
}

fn final_digest(hash: Sha256) -> RusticolResult<SemanticDigest> {
    SemanticDigest::new(hash.finalize().into())
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
mod family;
mod interpreter;
#[cfg(feature = "on-the-fly-test-support")]
mod probe_guard;
mod projection;
mod public_query;
pub(crate) mod seed_codec;
mod source_seed;
mod sweep;
mod templates;
#[cfg(any(test, feature = "on-the-fly-test-support"))]
mod test_support;
mod trace;

#[cfg(any(test, feature = "on-the-fly-test-support"))]
pub use family::{OnTheFlyQueryFamilyCensusV1, on_the_fly_query_family_census_v1};
#[cfg(feature = "on-the-fly-test-support")]
pub(crate) use family::{
    OnTheFlyQueryFamilyExecutionReportV1, OnTheFlyQueryFamilyExecutorV1, QueryFamilyTraceInput,
};
pub(crate) use interpreter::{
    OnTheFlyPreparedExecutorResolver, OnTheFlyStructuralInterpreter, OnTheFlyWorkspaceV1,
    ResolvedOnTheFlyExecutor,
};
#[cfg(feature = "on-the-fly-test-support")]
pub(crate) use probe_guard::{
    OnTheFlyForbiddenWorkGuardV1, OnTheFlyForbiddenWorkV1, reject_forbidden_work_if_probed,
};
pub(crate) use public_query::OnTheFlyLcSelectorV1;
pub(crate) use public_query::{DecodedLcQueryV1, OnTheFlySelectedSourceV1};
#[cfg(test)]
pub(crate) use source_seed::scalar_adapter_test_seed;
pub(crate) use source_seed::{
    OnTheFlyExternalColorRoleV1, OnTheFlyPairingClassV1, OnTheFlyPairingEndpointV1,
    OnTheFlyProcessSeedV1, OnTheFlySourceAnchorV1, OnTheFlySourceExecutionSpecV1,
    OnTheFlySourceOrientationV1, OnTheFlySourceStateV1, OnTheFlySourceWavefunctionFamilyV1,
};
#[cfg(feature = "on-the-fly-test-support")]
pub(crate) use test_support::build_on_the_fly_selected_trace_v1;
#[cfg(any(test, feature = "on-the-fly-test-support"))]
pub use test_support::{OnTheFlyTestSupportReportV1, on_the_fly_test_support_probe_v1};
#[cfg(any(test, feature = "on-the-fly-test-support"))]
pub(crate) use trace::ON_THE_FLY_WORK_CENSUS_BASIS_V1;
#[cfg(test)]
pub(crate) use trace::scalar_adapter_test_trace;
pub(crate) use trace::{
    OnTheFlyExecutorKeyV1, OnTheFlyStructuralTraceV1, authenticated_prepared_executor_binding,
};

use projection::*;
use sweep::*;
use templates::*;
use trace::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlyProjectionProbeV1 {
    pub(crate) enabled: bool,
    pub(crate) applied: bool,
    pub(crate) pre: [u32; 3],
    pub(crate) post: [u32; 3],
}

/// One compact, selector-local LC trace ready for prepared-executor binding.
///
/// This production seam owns no global recurrence builder or direct plan. Its
/// inputs are precisely the compact process seed, one already-decoded public
/// query, and the authenticated prepared template catalogs.
pub(crate) struct OnTheFlySelectedQueryTraceV1 {
    pub(crate) query: DecodedLcQueryV1,
    pub(crate) trace: OnTheFlyStructuralTraceV1,
    pub(crate) projection: OnTheFlyProjectionProbeV1,
}

pub(crate) fn build_selected_lc_query_trace_v1(
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    seed: &OnTheFlyProcessSeedV1,
    query: DecodedLcQueryV1,
) -> RusticolResult<OnTheFlySelectedQueryTraceV1> {
    build_selected_lc_query_trace_impl(templates, direct, seed, query, true)
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
pub(crate) fn build_selected_lc_query_trace_for_probe_v1(
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    seed: &OnTheFlyProcessSeedV1,
    query: DecodedLcQueryV1,
    enable_projection: bool,
) -> RusticolResult<OnTheFlySelectedQueryTraceV1> {
    build_selected_lc_query_trace_impl(templates, direct, seed, query, enable_projection)
}

fn build_selected_lc_query_trace_impl(
    templates: &ValidatedRecurrenceTemplateInput,
    direct: &PreparedDirectExecutorCatalog,
    seed: &OnTheFlyProcessSeedV1,
    query: DecodedLcQueryV1,
    enable_projection: bool,
) -> RusticolResult<OnTheFlySelectedQueryTraceV1> {
    if seed.direct_catalog_digest() != direct.direct_template_catalog_digest() {
        return Err(integrity(
            "compact query direct-template catalog differs from its prepared catalog",
        ));
    }
    let (trace, projection) =
        build_selected_lc_trace_impl(templates, &seed, &query, enable_projection, true)?;
    Ok(OnTheFlySelectedQueryTraceV1 {
        query,
        trace,
        projection: projection
            .ok_or_else(|| integrity("compact query projection proof was not collected"))?,
    })
}

fn selected_graph_counts(
    currents: &[PendingCurrent],
    closures: &[PendingClosure],
    live: &BTreeSet<u32>,
) -> RusticolResult<[u32; 3]> {
    let contributions = live.iter().try_fold(0usize, |total, id| {
        let count = currents
            .get(*id as usize)
            .ok_or_else(|| integrity("selected graph current is absent"))?
            .contributions
            .values()
            .filter(|factor| !factor.is_zero())
            .count();
        total
            .checked_add(count)
            .ok_or_else(|| invalid("selected graph contribution count exceeds usize"))
    })?;
    Ok([
        checked_u32(live.len(), "selected graph current count")?,
        checked_u32(contributions, "selected graph contribution count")?,
        checked_u32(closures.len(), "selected graph closure count")?,
    ])
}

/// Build one selected LC structural trace without constructing any global
/// recurrence/process projection.
pub(crate) fn build_selected_lc_trace(
    templates: &ValidatedRecurrenceTemplateInput,
    seed: &OnTheFlyProcessSeedV1,
    query: &DecodedLcQueryV1,
) -> RusticolResult<OnTheFlyStructuralTraceV1> {
    build_selected_lc_trace_impl(templates, seed, query, true, false).map(|(trace, _)| trace)
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
pub(crate) fn build_selected_lc_trace_for_probe(
    templates: &ValidatedRecurrenceTemplateInput,
    seed: &OnTheFlyProcessSeedV1,
    query: &DecodedLcQueryV1,
    enable_projection: bool,
) -> RusticolResult<(OnTheFlyStructuralTraceV1, OnTheFlyProjectionProbeV1)> {
    let (trace, probe) =
        build_selected_lc_trace_impl(templates, seed, query, enable_projection, true)?;
    Ok((
        trace,
        probe.ok_or_else(|| integrity("projection probe was not collected"))?,
    ))
}

fn build_selected_lc_trace_impl(
    templates: &ValidatedRecurrenceTemplateInput,
    seed: &OnTheFlyProcessSeedV1,
    query: &DecodedLcQueryV1,
    enable_projection: bool,
    collect_projection_probe: bool,
) -> RusticolResult<(OnTheFlyStructuralTraceV1, Option<OnTheFlyProjectionProbeV1>)> {
    if query.seed_digest != seed.semantic_digest() {
        return Err(integrity(
            "decoded query belongs to a different compact seed",
        ));
    }
    let catalog = validate_seed_against_templates(templates, seed)?;
    let transitions = prepared_transitions(templates, &catalog)?;
    let closures = prepared_closures(templates, &catalog)?;
    let propagators = propagator_by_state(templates)?;
    let mut colors = DynamicLCColorStateInterner::default();
    let mut currents = Vec::new();
    let mut current_ids = BTreeMap::new();
    insert_selected_sources(
        templates,
        &catalog,
        seed,
        query,
        &mut colors,
        &mut currents,
        &mut current_ids,
    )?;
    build_forward_currents(
        templates,
        &transitions,
        seed,
        &propagators,
        &mut colors,
        &mut currents,
        &mut current_ids,
    )?;
    let constructed_contribution_count = currents
        .iter()
        .map(|current| current.contributions.len())
        .try_fold(0usize, |total, count| {
            total
                .checked_add(count)
                .ok_or_else(|| invalid("constructed contribution count exceeds usize"))
        })?;
    let selected_closures =
        build_selected_closures(templates, &closures, seed, query, &colors, &currents)?;
    let live = live_current_ids(&currents, &selected_closures)?;
    let pre_counts = collect_projection_probe
        .then(|| selected_graph_counts(&currents, &selected_closures, &live))
        .transpose()?;
    let projected = enable_projection
        .then(|| project_query_local_color_aliases(&currents, &selected_closures, &live))
        .transpose()?
        .flatten();
    let projection_applied = projected.is_some();
    let (currents, selected_closures, live) = match projected {
        Some(projected) => {
            let live = (0..projected.currents.len())
                .map(|index| checked_u32(index, "projected query-local current ID"))
                .collect::<RusticolResult<BTreeSet<_>>>()?;
            (projected.currents, projected.closures, live)
        }
        None => (currents, selected_closures, live),
    };
    let post_counts = collect_projection_probe
        .then(|| selected_graph_counts(&currents, &selected_closures, &live))
        .transpose()?;
    let pairing_owner = resolve_projected_pairing_owner(seed, &selected_closures)?;
    let trace = lower_trace(
        templates,
        &catalog,
        seed,
        query,
        &colors,
        &currents,
        &selected_closures,
        pairing_owner,
        &transitions,
        &closures,
        &live,
        constructed_contribution_count,
    )?;
    Ok((
        trace,
        pre_counts
            .zip(post_counts)
            .map(|(pre, post)| OnTheFlyProjectionProbeV1 {
                enabled: enable_projection,
                applied: projection_applied,
                pre,
                post,
            }),
    ))
}
