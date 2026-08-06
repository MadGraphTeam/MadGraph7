// SPDX-License-Identifier: 0BSD

//! Private native LC lane for selector-local on-the-fly recurrence.
//!
//! This lane deliberately owns only the compact process seed, authenticated
//! model catalogs, and one warmed query-family executor.  It never accepts or
//! materializes a [`crate::recurrence::DirectRecurrencePlan`], a dense flow
//! catalog, or a process-wide recurrence schedule.  Public selectors are
//! decoded by the existing runtime adapter before they reach this module.
//! Likewise, load/evaluate/profile/benchmark need only this compact state.
//! The existing physics-metadata API can hang a separate lazy cache from
//! [`OnTheFlyNativeRuntime::seed`], deriving dense public axes only when a
//! caller explicitly requests introspection; that cache must not participate
//! in this lane's load or warmed-evaluation path.

use super::recurrence_backend::NativeOnTheFlyPreparedExecutorResolver;
use super::recurrence_lane::{
    DirectProfileDelta, LcResolvedOutputLayout, PreparedParameterProjectionEntry,
    accumulate_lc_diagonal_amplitude, direct_profile_from_delta,
    projected_prepared_parameter_values,
};
use super::recurrence_load::on_the_fly_source_major_momenta_into;
use super::*;
use crate::direct_arena::DirectArenaTrafficCounters;
use crate::recurrence::PreparedDirectExecutorCatalog;
use crate::recurrence::direct_backend::DirectExecutionCounters;
use crate::recurrence::direct_runtime::DirectRuntimeActivityCounters;
use crate::recurrence::on_the_fly::{
    DecodedLcQueryV1, OnTheFlyProcessSeedV1, OnTheFlyQueryFamilyCensusV1,
    OnTheFlyQueryFamilyExecutionReportV1, OnTheFlyQueryFamilyExecutorV1,
    OnTheFlySelectedQueryOutcomeV1, OnTheFlySelectedQueryTraceV1, QueryFamilyTraceInput,
    build_selected_lc_query_trace_v1,
};
use crate::recurrence::template::ValidatedRecurrenceTemplateInput;
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// One public resolved-output destination represented by a selector-local
/// trace.  The coefficient is the established helicity-orbit times LC color
/// weight; process normalization remains a call-local runtime input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct OnTheFlyLcReductionTargetV1 {
    helicity_position: usize,
    color_position: usize,
    coefficient: f64,
}

impl OnTheFlyLcReductionTargetV1 {
    pub(super) fn new(
        helicity_position: usize,
        color_position: usize,
        coefficient: f64,
    ) -> RusticolResult<Self> {
        if !coefficient.is_finite() || coefficient < 0.0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly LC reduction coefficient must be finite and nonnegative",
            ));
        }
        Ok(Self {
            helicity_position,
            color_position,
            coefficient,
        })
    }
}

/// Existing-runtime selector preparation expressed as one compact decoded LC
/// query plus its public resolved placements.  A representative helicity may
/// legitimately address several physical orbit members.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct OnTheFlyLcQueryRequestV1 {
    query: DecodedLcQueryV1,
    reduction_targets: Box<[OnTheFlyLcReductionTargetV1]>,
}

impl OnTheFlyLcQueryRequestV1 {
    pub(super) fn new(
        query: DecodedLcQueryV1,
        reduction_targets: Vec<OnTheFlyLcReductionTargetV1>,
    ) -> RusticolResult<Self> {
        if reduction_targets.is_empty() {
            return Err(RusticolError::invalid_argument(
                "on-the-fly LC query has no selected reduction target",
            ));
        }
        let mut coordinates = reduction_targets
            .iter()
            .map(|target| (target.helicity_position, target.color_position))
            .collect::<Vec<_>>();
        coordinates.sort_unstable();
        if coordinates.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(RusticolError::invalid_argument(
                "on-the-fly LC query repeats a resolved reduction target",
            ));
        }
        let total_coefficient = reduction_targets
            .iter()
            .try_fold(0.0_f64, |total, target| {
                let total = total + target.coefficient;
                total.is_finite().then_some(total).ok_or_else(|| {
                    RusticolError::invalid_argument(
                        "on-the-fly LC reduction coefficient sum is not finite",
                    )
                })
            })?;
        if total_coefficient == 0.0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly LC query has zero total reduction weight",
            ));
        }
        Ok(Self {
            query,
            reduction_targets: reduction_targets.into_boxed_slice(),
        })
    }

    fn total_coefficient(&self) -> f64 {
        self.reduction_targets
            .iter()
            .map(|target| target.coefficient)
            .sum()
    }
}

struct PreparedOnTheFlyLcFamilyV1 {
    lookup_key: OnTheFlyLcFamilyLookupKeyV1,
    requests: Box<[OnTheFlyLcQueryRequestV1]>,
    traces: Box<[OnTheFlySelectedQueryTraceV1]>,
    trace_request_indices: Box<[usize]>,
    logical_point_capacity: u32,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct OnTheFlyLcFamilyLookupKeyV1 {
    requests: Box<
        [(
            crate::recurrence::SemanticDigest,
            Box<[(usize, usize, u64)]>,
        )],
    >,
}

impl OnTheFlyLcFamilyLookupKeyV1 {
    fn from_requests(requests: &[OnTheFlyLcQueryRequestV1]) -> Self {
        Self {
            requests: requests
                .iter()
                .map(|request| {
                    (
                        request.query.semantic_digest(),
                        request
                            .reduction_targets
                            .iter()
                            .map(|target| {
                                (
                                    target.helicity_position,
                                    target.color_position,
                                    target.coefficient.to_bits(),
                                )
                            })
                            .collect::<Vec<_>>()
                            .into_boxed_slice(),
                    )
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        }
    }
}

/// Process-local production owner for the private on-the-fly lane.
///
/// Field order is intentional: the family executor (and therefore every row
/// table borrowed by a prepared kernel) is dropped before the catalogs and
/// compact process seed from which its traces were constructed.
pub(super) struct OnTheFlyNativeRuntime {
    executor: OnTheFlyQueryFamilyExecutorV1<NativeOnTheFlyPreparedExecutorResolver>,
    templates: ValidatedRecurrenceTemplateInput,
    direct_catalog: PreparedDirectExecutorCatalog,
    seed: OnTheFlyProcessSeedV1,
    parameter_defaults: Box<[crate::EagerComplex64]>,
    parameter_projection: Box<[PreparedParameterProjectionEntry]>,
    runtime_parameter_values: Box<[f64]>,
    families: Vec<PreparedOnTheFlyLcFamilyV1>,
    family_lookup: BTreeMap<OnTheFlyLcFamilyLookupKeyV1, Vec<usize>>,
    last_family: Option<usize>,
    pending_family: Option<PreparedOnTheFlyLcFamilyV1>,
    source_momenta_scratch: Vec<f64>,
    amplitude_scratch: Vec<(f64, f64)>,
}

impl OnTheFlyNativeRuntime {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        templates: ValidatedRecurrenceTemplateInput,
        direct_catalog: PreparedDirectExecutorCatalog,
        seed: OnTheFlyProcessSeedV1,
        resolver: NativeOnTheFlyPreparedExecutorResolver,
        parameter_defaults: Vec<crate::EagerComplex64>,
        parameter_projection: Vec<PreparedParameterProjectionEntry>,
        runtime_parameter_values: &[f64],
    ) -> RusticolResult<Self> {
        let summary = templates.summary();
        if seed.template_catalog_digest() != summary.catalog_digest
            || seed.model_digest() != summary.compiled_model_digest
            || seed.prepared_pack_digest() != summary.prepared_kernel_pack_digest
            || seed.direct_catalog_digest() != direct_catalog.direct_template_catalog_digest()
        {
            return Err(RusticolError::integrity(
                "on-the-fly native lane catalogs do not match its compact process seed",
            ));
        }
        if parameter_defaults.len()
            != usize::try_from(summary.parameter_count)
                .map_err(|_| RusticolError::artifact("on-the-fly parameter count exceeds usize"))?
        {
            return Err(RusticolError::integrity(
                "on-the-fly prepared parameter defaults have the wrong size",
            ));
        }
        if runtime_parameter_values
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(RusticolError::invalid_argument(
                "on-the-fly runtime parameter value is not finite",
            ));
        }

        let prepared_parameters = projected_prepared_parameter_values(
            &parameter_defaults,
            &parameter_projection,
            runtime_parameter_values,
        )?;
        let mut executor = OnTheFlyQueryFamilyExecutorV1::new(resolver);
        executor.set_parameters(&prepared_parameters)?;

        Ok(Self {
            executor,
            templates,
            direct_catalog,
            seed,
            parameter_defaults: parameter_defaults.into_boxed_slice(),
            parameter_projection: parameter_projection.into_boxed_slice(),
            runtime_parameter_values: runtime_parameter_values.to_vec().into_boxed_slice(),
            families: Vec::new(),
            family_lookup: BTreeMap::new(),
            last_family: None,
            pending_family: None,
            source_momenta_scratch: Vec::new(),
            amplitude_scratch: Vec::new(),
        })
    }

    pub(super) const fn seed(&self) -> &OnTheFlyProcessSeedV1 {
        &self.seed
    }

    /// Cold selector/trace construction. Repeating an identical public
    /// selection reuses the retained query-local traces and warmed row family;
    /// increasing only the point capacity replaces its numeric workspace.
    pub(super) fn prepare_lc_queries(
        &mut self,
        requests: &[OnTheFlyLcQueryRequestV1],
        logical_point_capacity: u32,
    ) -> RusticolResult<bool> {
        if requests.is_empty() || logical_point_capacity == 0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly LC preparation requires queries and a nonzero point capacity",
            ));
        }
        if requests
            .iter()
            .any(|request| request.query.process_seed_digest() != self.seed.semantic_digest())
        {
            return Err(RusticolError::integrity(
                "on-the-fly LC query belongs to a different compact process seed",
            ));
        }

        // The common same-selector path compares the last family directly and
        // never allocates or consults the all-seen index. When the numeric
        // workspace is already large enough it also avoids a redundant second
        // identity walk inside the executor.
        if self.pending_family.is_none()
            && self.last_family.is_some_and(|index| {
                self.families.get(index).is_some_and(|family| {
                    family.requests.as_ref() == requests
                        && logical_point_capacity <= family.logical_point_capacity
                })
            })
        {
            return Ok(true);
        }
        let retained = self
            .last_family
            .filter(|&index| {
                self.families
                    .get(index)
                    .is_some_and(|family| family.requests.as_ref() == requests)
            })
            .or_else(|| {
                let lookup_key = OnTheFlyLcFamilyLookupKeyV1::from_requests(requests);
                self.family_lookup.get(&lookup_key).and_then(|indices| {
                    indices.iter().copied().find(|&index| {
                        self.families
                            .get(index)
                            .is_some_and(|family| family.requests.as_ref() == requests)
                    })
                })
            });
        if let Some(index) = retained {
            let family = self
                .families
                .get_mut(index)
                .expect("on-the-fly family lookup index is absent");
            let cache_hit = if family.traces.is_empty() {
                self.executor.deactivate()?;
                true
            } else {
                let inputs = family
                    .traces
                    .iter()
                    .map(|selected| QueryFamilyTraceInput {
                        trace: &selected.trace,
                        projection: selected.projection,
                    })
                    .collect::<Vec<_>>();
                self.executor
                    .prepare(&self.direct_catalog, &inputs, logical_point_capacity)?
            };
            family.logical_point_capacity =
                family.logical_point_capacity.max(logical_point_capacity);
            self.pending_family = None;
            self.last_family = Some(index);
            return Ok(cache_hit);
        }

        if self
            .pending_family
            .as_ref()
            .is_some_and(|family| family.requests.as_ref() == requests)
        {
            let family = self
                .pending_family
                .as_mut()
                .expect("matching on-the-fly pending family disappeared");
            let inputs = family
                .traces
                .iter()
                .map(|selected| QueryFamilyTraceInput {
                    trace: &selected.trace,
                    projection: selected.projection,
                })
                .collect::<Vec<_>>();
            let cache_hit =
                self.executor
                    .prepare(&self.direct_catalog, &inputs, logical_point_capacity)?;
            if cache_hit {
                return Err(RusticolError::integrity(
                    "pending on-the-fly family unexpectedly resolved as retained",
                ));
            }
            family.logical_point_capacity =
                family.logical_point_capacity.max(logical_point_capacity);
            return Ok(false);
        }

        let mut traces = Vec::new();
        let mut trace_request_indices = Vec::new();
        traces.try_reserve_exact(requests.len()).map_err(|error| {
            RusticolError::invalid_argument(format!(
                "on-the-fly trace-family allocation failed: {error}"
            ))
        })?;
        trace_request_indices
            .try_reserve_exact(requests.len())
            .map_err(|error| {
                RusticolError::invalid_argument(format!(
                    "on-the-fly trace-index allocation failed: {error}"
                ))
            })?;
        for (request_index, request) in requests.iter().enumerate() {
            match build_selected_lc_query_trace_v1(
                &self.templates,
                &self.direct_catalog,
                &self.seed,
                request.query.clone(),
            )? {
                OnTheFlySelectedQueryOutcomeV1::Trace(selected) => {
                    if selected.query != request.query {
                        return Err(RusticolError::integrity(
                            "on-the-fly trace query changed during construction",
                        ));
                    }
                    traces.push(selected);
                    trace_request_indices.push(request_index);
                }
                OnTheFlySelectedQueryOutcomeV1::StructuralZero { query } => {
                    if query != request.query {
                        return Err(RusticolError::integrity(
                            "on-the-fly structural-zero query changed during construction",
                        ));
                    }
                }
            }
        }
        let cache_hit = if traces.is_empty() {
            self.executor.deactivate()?;
            false
        } else {
            self.executor.resolver_mut().bind_on_the_fly_family(
                &self.templates,
                &self.direct_catalog,
                &self.seed,
                &mut traces,
            )?;
            let inputs = traces
                .iter()
                .map(|selected| QueryFamilyTraceInput {
                    trace: &selected.trace,
                    projection: selected.projection,
                })
                .collect::<Vec<_>>();
            let cache_hit =
                self.executor
                    .prepare(&self.direct_catalog, &inputs, logical_point_capacity)?;
            let census = self.executor.prepared_census().ok_or_else(|| {
                RusticolError::integrity("on-the-fly prepared family has no work census")
            })?;
            if usize::try_from(census.union_amplitude_destination_count).ok() != Some(traces.len())
            {
                return Err(RusticolError::integrity(
                    "on-the-fly family does not retain one destination per nonzero query",
                ));
            }
            cache_hit
        };
        self.families.try_reserve(1).map_err(|error| {
            RusticolError::invalid_argument(format!(
                "on-the-fly retained-family allocation failed: {error}"
            ))
        })?;
        let candidate = PreparedOnTheFlyLcFamilyV1 {
            lookup_key: OnTheFlyLcFamilyLookupKeyV1::from_requests(requests),
            requests: requests.to_vec().into_boxed_slice(),
            traces: traces.into_boxed_slice(),
            trace_request_indices: trace_request_indices.into_boxed_slice(),
            logical_point_capacity,
        };
        if cache_hit || candidate.traces.is_empty() {
            self.retain_family(candidate);
        } else {
            self.pending_family = Some(candidate);
        }
        Ok(cache_hit)
    }

    /// Re-activate the family already selected by the runtime wrapper without
    /// rebuilding its public selector requests.  The common warmed call has
    /// sufficient point capacity and therefore performs no allocation or
    /// identity walk.  Capacity growth remains explicit and fallible.
    pub(super) fn reuse_current_lc_queries(
        &mut self,
        logical_point_capacity: u32,
    ) -> RusticolResult<bool> {
        if logical_point_capacity == 0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly LC preparation requires a nonzero point capacity",
            ));
        }
        if let Some(family) = self.pending_family.as_mut() {
            if logical_point_capacity <= family.logical_point_capacity {
                return Ok(false);
            }
            let inputs = family
                .traces
                .iter()
                .map(|selected| QueryFamilyTraceInput {
                    trace: &selected.trace,
                    projection: selected.projection,
                })
                .collect::<Vec<_>>();
            let cache_hit =
                self.executor
                    .prepare(&self.direct_catalog, &inputs, logical_point_capacity)?;
            if cache_hit {
                return Err(RusticolError::integrity(
                    "pending on-the-fly family unexpectedly resolved as retained",
                ));
            }
            family.logical_point_capacity = logical_point_capacity;
            return Ok(false);
        }
        let index = self.last_family.ok_or_else(|| {
            RusticolError::internal(
                "on-the-fly selector cache has no corresponding prepared family",
            )
        })?;
        let family = self.families.get_mut(index).ok_or_else(|| {
            RusticolError::internal(
                "on-the-fly selector cache references an absent prepared family",
            )
        })?;
        if logical_point_capacity <= family.logical_point_capacity {
            return Ok(true);
        }
        let cache_hit = if family.traces.is_empty() {
            self.executor.deactivate()?;
            true
        } else {
            let inputs = family
                .traces
                .iter()
                .map(|selected| QueryFamilyTraceInput {
                    trace: &selected.trace,
                    projection: selected.projection,
                })
                .collect::<Vec<_>>();
            self.executor
                .prepare(&self.direct_catalog, &inputs, logical_point_capacity)?
        };
        family.logical_point_capacity = logical_point_capacity;
        Ok(cache_hit)
    }

    pub(super) fn prepared_census(&self) -> Option<OnTheFlyQueryFamilyCensusV1> {
        self.executor.prepared_census()
    }

    #[cfg(test)]
    pub(super) fn retained_family_count(&self) -> usize {
        self.families.len() + usize::from(self.pending_family.is_some())
    }

    pub(super) fn clear(&mut self) -> RusticolResult<()> {
        self.executor.clear_families()?;
        self.pending_family = None;
        self.last_family = None;
        self.family_lookup.clear();
        self.families.clear();
        self.source_momenta_scratch = Vec::new();
        self.amplitude_scratch = Vec::new();
        Ok(())
    }

    fn current_family(&self) -> Option<&PreparedOnTheFlyLcFamilyV1> {
        self.pending_family
            .as_ref()
            .or_else(|| self.last_family.and_then(|index| self.families.get(index)))
    }

    fn promote_pending_family(&mut self) {
        let Some(family) = self.pending_family.take() else {
            return;
        };
        self.retain_family(family);
    }

    fn retain_family(&mut self, family: PreparedOnTheFlyLcFamilyV1) {
        let lookup_key = family.lookup_key.clone();
        let index = self.families.len();
        self.families.push(family);
        self.family_lookup
            .entry(lookup_key)
            .or_default()
            .push(index);
        self.last_family = Some(index);
    }

    fn prepare_runtime_parameters(&mut self, runtime_values: &[f64]) -> RusticolResult<()> {
        if runtime_values.iter().any(|value| !value.is_finite()) {
            return Err(RusticolError::invalid_argument(
                "on-the-fly runtime parameter value is not finite",
            ));
        }
        if self.runtime_parameter_values.as_ref() == runtime_values {
            return Ok(());
        }
        let prepared = projected_prepared_parameter_values(
            &self.parameter_defaults,
            &self.parameter_projection,
            runtime_values,
        )?;
        self.executor.set_parameters(&prepared)?;
        self.runtime_parameter_values = runtime_values.to_vec().into_boxed_slice();
        Ok(())
    }

    fn ensure_point_capacity(&mut self, point_count: u32) -> RusticolResult<()> {
        let Some(prepared) = self.current_family() else {
            return Err(RusticolError::invalid_argument(
                "on-the-fly evaluation requires prepared LC queries",
            ));
        };
        if point_count <= prepared.logical_point_capacity {
            return Ok(());
        }
        let requests = prepared.requests.to_vec();
        self.prepare_lc_queries(&requests, point_count)?;
        Ok(())
    }

    fn ensure_amplitude_scratch(&mut self, point_count: u32) -> RusticolResult<usize> {
        let query_count = self
            .current_family()
            .ok_or_else(|| {
                RusticolError::invalid_argument(
                    "on-the-fly evaluation requires prepared LC queries",
                )
            })?
            .traces
            .len();
        let required = query_count
            .checked_mul(point_count as usize)
            .ok_or_else(|| {
                RusticolError::invalid_argument("on-the-fly amplitude shape exceeds usize")
            })?;
        if self.amplitude_scratch.len() < required {
            self.amplitude_scratch
                .try_reserve_exact(required - self.amplitude_scratch.len())
                .map_err(|error| {
                    RusticolError::invalid_argument(format!(
                        "on-the-fly amplitude workspace allocation failed: {error}"
                    ))
                })?;
            self.amplitude_scratch.resize(required, (0.0, 0.0));
        }
        Ok(required)
    }

    fn execute_amplitudes(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
    ) -> RusticolResult<(
        OnTheFlyQueryFamilyExecutionReportV1,
        Duration,
        Duration,
        Duration,
    )> {
        if point_count == 0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly evaluation requires at least one point",
            ));
        }
        self.ensure_point_capacity(point_count)?;

        let parameter_started = Instant::now();
        self.prepare_runtime_parameters(runtime_parameters)?;
        let parameter_setup = parameter_started.elapsed();

        if self
            .current_family()
            .is_some_and(|family| family.traces.is_empty())
        {
            return Ok((
                OnTheFlyQueryFamilyExecutionReportV1 {
                    cache_hit: true,
                    ..OnTheFlyQueryFamilyExecutionReportV1::default()
                },
                parameter_setup,
                Duration::ZERO,
                Duration::ZERO,
            ));
        }

        let input_started = Instant::now();
        let source_major_len = on_the_fly_source_major_momenta_into(
            &self.seed,
            point_major_momenta,
            point_count,
            4,
            &mut self.source_momenta_scratch,
        )?;
        let input_setup = input_started.elapsed();

        let output_len = self.ensure_amplitude_scratch(point_count)?;
        let execution_started = Instant::now();
        let report = self.executor.execute_into(
            &self.source_momenta_scratch[..source_major_len],
            point_count,
            &mut self.amplitude_scratch[..output_len],
        )?;
        let execution = execution_started.elapsed();
        // A cold family becomes reusable only after its first successful
        // prepared-kernel execution.
        self.promote_pending_family();
        Ok((report, parameter_setup, input_setup, execution))
    }

    /// Evaluate total LC matrix elements into the same one-value-per-point
    /// shape consumed by the existing `Runtime.evaluate_into` path.
    pub(super) fn run_f64_into(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
        normalization_factor: f64,
        output: &mut [f64],
    ) -> RusticolResult<(OnTheFlyQueryFamilyExecutionReportV1, RuntimeProfile)> {
        if output.len() != point_count as usize
            || !normalization_factor.is_finite()
            || normalization_factor < 0.0
        {
            return Err(RusticolError::invalid_argument(
                "on-the-fly total output shape or normalization is invalid",
            ));
        }
        let total_started = Instant::now();
        let (report, parameter_setup, input_setup, execution) =
            self.execute_amplitudes(point_major_momenta, point_count, runtime_parameters)?;
        output.fill(0.0);
        let reduction_started = Instant::now();
        let family = self
            .current_family()
            .expect("successful on-the-fly execution lost its prepared family");
        reduce_total_into(
            &family.requests,
            &family.trace_request_indices,
            &self.amplitude_scratch,
            point_count as usize,
            normalization_factor,
            output,
        )?;
        let reduction = reduction_started.elapsed();
        let profile = profile_from_report(
            total_started.elapsed(),
            parameter_setup,
            input_setup,
            execution,
            reduction,
            point_major_momenta.len(),
            point_count,
            report,
        );
        Ok((report, profile))
    }

    /// Evaluate resolved LC values in the established
    /// `[point][helicity][color]` order without creating a dense flow table.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_resolved_f64_into(
        &mut self,
        point_major_momenta: &[f64],
        point_count: u32,
        runtime_parameters: &[f64],
        normalization_factor: f64,
        helicity_count: usize,
        color_count: usize,
        output: &mut [f64],
    ) -> RusticolResult<(OnTheFlyQueryFamilyExecutionReportV1, RuntimeProfile)> {
        if !normalization_factor.is_finite() || normalization_factor < 0.0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly resolved normalization is invalid",
            ));
        }
        let layout =
            LcResolvedOutputLayout::new(point_count as usize, helicity_count, color_count)?;
        if output.len() != layout.output_len()? {
            return Err(RusticolError::invalid_argument(
                "on-the-fly resolved output has the wrong size",
            ));
        }
        let requests = self.current_family().ok_or_else(|| {
            RusticolError::invalid_argument(
                "on-the-fly resolved evaluation requires prepared LC queries",
            )
        })?;
        for target in requests
            .requests
            .iter()
            .flat_map(|request| request.reduction_targets.iter())
        {
            if target.helicity_position >= helicity_count || target.color_position >= color_count {
                return Err(RusticolError::integrity(
                    "on-the-fly resolved target is outside the selected public axes",
                ));
            }
        }

        let total_started = Instant::now();
        let (report, parameter_setup, input_setup, execution) =
            self.execute_amplitudes(point_major_momenta, point_count, runtime_parameters)?;
        output.fill(0.0);
        let reduction_started = Instant::now();
        let family = self
            .current_family()
            .expect("successful on-the-fly execution lost its prepared family");
        reduce_resolved_into(
            &family.requests,
            &family.trace_request_indices,
            &self.amplitude_scratch,
            point_count as usize,
            normalization_factor,
            layout,
            output,
        )?;
        let reduction = reduction_started.elapsed();
        let profile = profile_from_report(
            total_started.elapsed(),
            parameter_setup,
            input_setup,
            execution,
            reduction,
            point_major_momenta.len(),
            point_count,
            report,
        );
        Ok((report, profile))
    }
}

fn reduce_total_into(
    requests: &[OnTheFlyLcQueryRequestV1],
    trace_request_indices: &[usize],
    amplitudes: &[(f64, f64)],
    point_count: usize,
    normalization_factor: f64,
    output: &mut [f64],
) -> RusticolResult<()> {
    for (destination, &request_index) in trace_request_indices.iter().enumerate() {
        let request = requests
            .get(request_index)
            .ok_or_else(|| RusticolError::integrity("on-the-fly trace request index is absent"))?;
        let base = destination.checked_mul(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("on-the-fly amplitude destination offset exceeds usize")
        })?;
        let end = base.checked_add(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("on-the-fly amplitude range exceeds usize")
        })?;
        if end > amplitudes.len() {
            return Err(RusticolError::integrity(
                "on-the-fly amplitude destination is absent",
            ));
        }
        let weight = request.total_coefficient() * normalization_factor;
        accumulate_lc_diagonal_amplitude(
            point_count,
            weight,
            |point| amplitudes[base + point],
            |point, value| output[point] += value,
        );
    }
    Ok(())
}

fn reduce_resolved_into(
    requests: &[OnTheFlyLcQueryRequestV1],
    trace_request_indices: &[usize],
    amplitudes: &[(f64, f64)],
    point_count: usize,
    normalization_factor: f64,
    layout: LcResolvedOutputLayout,
    output: &mut [f64],
) -> RusticolResult<()> {
    for (destination, &request_index) in trace_request_indices.iter().enumerate() {
        let request = requests
            .get(request_index)
            .ok_or_else(|| RusticolError::integrity("on-the-fly trace request index is absent"))?;
        let base = destination.checked_mul(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("on-the-fly amplitude destination offset exceeds usize")
        })?;
        let end = base.checked_add(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("on-the-fly amplitude range exceeds usize")
        })?;
        if end > amplitudes.len() {
            return Err(RusticolError::integrity(
                "on-the-fly amplitude destination is absent",
            ));
        }
        for target in &request.reduction_targets {
            let weight = target.coefficient * normalization_factor;
            accumulate_lc_diagonal_amplitude(
                point_count,
                weight,
                |point| amplitudes[base + point],
                |point, value| {
                    let target_index = layout
                        .index(point, target.helicity_position, target.color_position)
                        .expect("validated on-the-fly resolved coordinate");
                    output[target_index] += value;
                },
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn profile_from_report(
    total: Duration,
    parameter_setup: Duration,
    input_setup: Duration,
    execution: Duration,
    reduction: Duration,
    momentum_scalar_count: usize,
    point_count: u32,
    report: OnTheFlyQueryFamilyExecutionReportV1,
) -> RuntimeProfile {
    let calls = u64::from(report.source_calls)
        + u64::from(report.contribution_calls)
        + u64::from(report.finalization_calls)
        + u64::from(report.closure_calls);
    let rows = u64::from(report.source_rows)
        + u64::from(report.contribution_rows)
        + u64::from(report.finalization_rows)
        + u64::from(report.closure_rows);
    direct_profile_from_delta(
        total,
        parameter_setup,
        input_setup,
        reduction,
        DirectProfileDelta {
            direct_execution: execution,
            execution: DirectExecutionCounters {
                source_calls: u64::from(report.source_calls),
                source_rows: u64::from(report.source_rows),
                contribution_calls: u64::from(report.contribution_calls),
                contribution_rows: u64::from(report.contribution_rows),
                finalization_calls: u64::from(report.finalization_calls),
                finalization_rows: u64::from(report.finalization_rows),
                closure_calls: u64::from(report.closure_calls),
                closure_rows: u64::from(report.closure_rows),
                ..DirectExecutionCounters::default()
            },
            traffic: DirectArenaTrafficCounters {
                calls,
                rows,
                points: calls.saturating_mul(u64::from(point_count)),
                ..DirectArenaTrafficCounters::default()
            },
            activity: DirectRuntimeActivityCounters {
                momentum_fill_calls: 1,
                momentum_scalar_values_filled: momentum_scalar_count as u64,
                schedule_executions: 1,
                ..DirectRuntimeActivityCounters::default()
            },
            ..DirectProfileDelta::default()
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::SemanticDigest;
    use crate::recurrence::on_the_fly::{OnTheFlyLcSelectorV1, scalar_adapter_test_seed};

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).unwrap()
    }

    fn scalar_query() -> DecodedLcQueryV1 {
        let seed = scalar_adapter_test_seed(digest(1), digest(2), digest(3), digest(4)).unwrap();
        DecodedLcQueryV1::new(&seed, vec![0, 1], &[0, 0], OnTheFlyLcSelectorV1::Singlet).unwrap()
    }

    #[test]
    fn lc_reduction_targets_reject_nonfinite_or_negative_coefficients() {
        assert!(OnTheFlyLcReductionTargetV1::new(0, 0, -1.0).is_err());
        assert!(OnTheFlyLcReductionTargetV1::new(0, 0, f64::NAN).is_err());
        assert_eq!(
            OnTheFlyLcReductionTargetV1::new(3, 5, 2.0)
                .unwrap()
                .coefficient,
            2.0
        );
    }

    #[test]
    fn lc_query_rejects_duplicate_resolved_coordinates() {
        let duplicate = vec![
            OnTheFlyLcReductionTargetV1::new(2, 3, 1.0).unwrap(),
            OnTheFlyLcReductionTargetV1::new(2, 3, 2.0).unwrap(),
        ];
        assert!(OnTheFlyLcQueryRequestV1::new(scalar_query(), duplicate).is_err());
    }

    #[test]
    fn lc_query_rejects_zero_total_reduction_weight() {
        let zero = vec![OnTheFlyLcReductionTargetV1::new(0, 0, 0.0).unwrap()];
        assert!(OnTheFlyLcQueryRequestV1::new(scalar_query(), zero).is_err());
    }

    #[test]
    fn profile_maps_grouped_family_role_counters_without_dense_plan_counters() {
        let profile = profile_from_report(
            Duration::from_millis(11),
            Duration::from_millis(1),
            Duration::from_millis(2),
            Duration::from_millis(7),
            Duration::from_millis(1),
            96,
            3,
            OnTheFlyQueryFamilyExecutionReportV1 {
                cache_hit: true,
                source_calls: 2,
                source_rows: 4,
                contribution_calls: 3,
                contribution_rows: 17,
                finalization_calls: 1,
                finalization_rows: 5,
                closure_calls: 2,
                closure_rows: 7,
            },
        );
        assert_eq!(profile.recurrence_schedule_execution_count, 1);
        assert_eq!(profile.recurrence_momentum_scalar_value_count, 96);
        assert_eq!(profile.recurrence_source_call_count, 2);
        assert_eq!(profile.recurrence_source_row_count, 4);
        assert_eq!(profile.recurrence_contribution_call_count, 3);
        assert_eq!(profile.recurrence_contribution_row_count, 17);
        assert_eq!(profile.recurrence_finalization_call_count, 1);
        assert_eq!(profile.recurrence_finalization_row_count, 5);
        assert_eq!(profile.recurrence_closure_call_count, 2);
        assert_eq!(profile.recurrence_closure_row_count, 7);
    }

    #[test]
    fn source_major_workspace_is_reused_after_warmup() {
        let seed = scalar_adapter_test_seed(digest(1), digest(2), digest(3), digest(4)).unwrap();
        let point_major = (0..16).map(f64::from).collect::<Vec<_>>();
        let mut scratch = Vec::new();
        let output_len =
            on_the_fly_source_major_momenta_into(&seed, &point_major, 2, 4, &mut scratch).unwrap();
        assert_eq!(output_len, 16);
        assert_eq!(scratch[..4], [0.0, 8.0, 1.0, 9.0]);
        let warmed_pointer = scratch.as_ptr();
        let warmed_capacity = scratch.capacity();
        on_the_fly_source_major_momenta_into(&seed, &point_major, 2, 4, &mut scratch).unwrap();
        assert_eq!(scratch.as_ptr(), warmed_pointer);
        assert_eq!(scratch.capacity(), warmed_capacity);
    }

    #[test]
    fn mixed_structural_zero_and_trace_preserve_zero_resolved_slot() {
        let query = scalar_query();
        let requests = vec![
            OnTheFlyLcQueryRequestV1::new(
                query.clone(),
                vec![OnTheFlyLcReductionTargetV1::new(0, 0, 1.0).unwrap()],
            )
            .unwrap(),
            OnTheFlyLcQueryRequestV1::new(
                query,
                vec![OnTheFlyLcReductionTargetV1::new(1, 0, 2.0).unwrap()],
            )
            .unwrap(),
        ];
        // Request zero is a structural zero and therefore has no amplitude
        // destination. Request one is the sole nonzero trace destination.
        let trace_request_indices = [1];
        let amplitudes = [(3.0, 4.0)];

        let mut total = [0.0];
        reduce_total_into(
            &requests,
            &trace_request_indices,
            &amplitudes,
            1,
            1.0,
            &mut total,
        )
        .unwrap();
        assert_eq!(total, [50.0]);

        let layout = LcResolvedOutputLayout::new(1, 2, 1).unwrap();
        let mut resolved = [0.0; 2];
        reduce_resolved_into(
            &requests,
            &trace_request_indices,
            &amplitudes,
            1,
            1.0,
            layout,
            &mut resolved,
        )
        .unwrap();
        assert_eq!(resolved, [0.0, 50.0]);
    }

    #[test]
    fn all_structural_zero_family_reduces_to_exact_zero_without_amplitudes() {
        let requests = [OnTheFlyLcQueryRequestV1::new(
            scalar_query(),
            vec![OnTheFlyLcReductionTargetV1::new(0, 0, 1.0).unwrap()],
        )
        .unwrap()];
        let mut total = [0.0];
        reduce_total_into(&requests, &[], &[], 1, 1.0, &mut total).unwrap();
        assert_eq!(total, [0.0]);

        let layout = LcResolvedOutputLayout::new(1, 1, 1).unwrap();
        let mut resolved = [0.0];
        reduce_resolved_into(&requests, &[], &[], 1, 1.0, layout, &mut resolved).unwrap();
        assert_eq!(resolved, [0.0]);
    }
}
