// SPDX-License-Identifier: 0BSD

//! Owned workspace for direct-arena recurrence execution.
//!
//! Construction performs every allocation needed by a point tile. Callers
//! fill the persistent momentum and parameter arenas, then execute the
//! authenticated direct plan without resizing any runtime storage.

use std::mem::size_of;
use std::ops::Range;
use std::time::{Duration, Instant};

use super::direct_backend::{
    DIRECT_STATUS_OK, DirectArenaView, DirectExecutionCounters, DirectExecutorCatalog,
    DirectFactorView, DirectMomentumView, DirectParameterView, DirectUnionSourceDispatchHandle,
    DirectWorkspace, execute_direct_plan,
};
use super::direct_plan::{
    DirectAmplitudeDestinationDescriptor, DirectRecurrencePlan, DirectResolvedHelicityDescriptor,
};
use super::{RecurrenceStrategy, SemanticDigest};
use crate::{RusticolError, RusticolResult};

/// Alignment used for the SIMD-facing numeric arenas.
pub const DIRECT_RUNTIME_ARENA_ALIGNMENT: usize = 64;
const DIRECT_RUNTIME_CACHE_TARGET_BYTES: usize = 4 * 1024 * 1024;

/// Borrowed component-major amplitude output for the most recent tile.
#[derive(Clone, Copy, Debug)]
pub struct DirectRecurrenceTileOutput<'a> {
    amplitude_re: &'a [f64],
    amplitude_im: &'a [f64],
    amplitude_destinations: &'a [DirectAmplitudeDestinationDescriptor],
    point_count: u32,
    point_stride: u32,
    destination_count: u32,
    public_flow_id: Option<u32>,
    representative_flow_id: Option<u32>,
}

impl<'a> DirectRecurrenceTileOutput<'a> {
    pub const fn point_count(&self) -> u32 {
        self.point_count
    }

    pub const fn point_stride(&self) -> u32 {
        self.point_stride
    }

    pub const fn destination_count(&self) -> u32 {
        self.destination_count
    }

    pub const fn public_flow_id(&self) -> Option<u32> {
        self.public_flow_id
    }

    pub const fn representative_flow_id(&self) -> Option<u32> {
        self.representative_flow_id
    }

    /// Full persistent real arena, including inactive tile tails.
    pub const fn storage_re(&self) -> &'a [f64] {
        self.amplitude_re
    }

    /// Full persistent imaginary arena, including inactive tile tails.
    pub const fn storage_im(&self) -> &'a [f64] {
        self.amplitude_im
    }

    pub fn destination_re(&self, destination_id: u32) -> Option<&[f64]> {
        self.destination(self.amplitude_re, destination_id)
    }

    pub fn destination_im(&self, destination_id: u32) -> Option<&[f64]> {
        self.destination(self.amplitude_im, destination_id)
    }

    pub fn selected_destination_ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.amplitude_destinations
            .iter()
            .filter(|destination| {
                self.representative_flow_id
                    .is_none_or(|representative| destination.target_sector_id == representative)
            })
            .map(|destination| destination.id)
    }

    fn destination<'b>(&self, values: &'b [f64], destination_id: u32) -> Option<&'b [f64]> {
        if destination_id >= self.destination_count {
            return None;
        }
        let descriptor = self.amplitude_destinations.get(destination_id as usize)?;
        if self
            .representative_flow_id
            .is_some_and(|representative| descriptor.target_sector_id != representative)
        {
            return None;
        }
        let start = destination_id as usize * self.point_stride as usize;
        values.get(start..start + self.point_count as usize)
    }
}

/// Prepared, plan-authenticated topology-replay selection.
pub struct DirectReplaySelectorPlan {
    runtime_layout_digest: SemanticDigest,
    public_flow_id: u32,
    representative_flow_id: u32,
    source_permutation: Box<[u32]>,
    phase_re: f64,
    phase_im: f64,
    multiplicity: u32,
}

/// Prepared, plan-authenticated all-flow-union helicity selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectUnionHelicitySelectorPlan {
    runtime_layout_digest: SemanticDigest,
    resolved_helicity_id: u32,
    source_selection_start: usize,
    source_selection_count: usize,
}

impl DirectUnionHelicitySelectorPlan {
    pub const fn resolved_helicity_id(&self) -> u32 {
        self.resolved_helicity_id
    }
}

impl DirectReplaySelectorPlan {
    pub const fn public_flow_id(&self) -> u32 {
        self.public_flow_id
    }

    pub const fn representative_flow_id(&self) -> u32 {
        self.representative_flow_id
    }

    pub const fn phase(&self) -> (f64, f64) {
        (self.phase_re, self.phase_im)
    }

    pub const fn multiplicity(&self) -> u32 {
        self.multiplicity
    }

    pub fn mapped_external_source_slot(&self, representative_source_slot: u32) -> Option<u32> {
        self.source_permutation
            .get(representative_source_slot as usize)
            .copied()
    }

    pub(crate) fn source_permutation(&self) -> &[u32] {
        &self.source_permutation
    }

    fn identity(&self) -> DirectReplaySelectorIdentity {
        DirectReplaySelectorIdentity {
            runtime_layout_digest: self.runtime_layout_digest,
            public_flow_id: self.public_flow_id,
            representative_flow_id: self.representative_flow_id,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectRuntimeActivityCounters {
    pub momentum_fill_calls: u64,
    pub momentum_forms_filled: u64,
    pub momentum_terms_filled: u64,
    pub momentum_scalar_values_filled: u64,
    pub schedule_executions: u64,
    pub replay_schedule_executions: u64,
    pub replay_output_values_scaled: u64,
    pub union_source_dispatch_calls: u64,
    pub union_source_rows: u64,
    pub union_schedule_executions: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectRuntimePhaseTimings {
    pub momentum_fill: Duration,
    pub union_source_fill: Duration,
    pub direct_execution: Duration,
    pub replay_output_mapping: Duration,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DirectReplaySelectorIdentity {
    runtime_layout_digest: SemanticDigest,
    public_flow_id: u32,
    representative_flow_id: u32,
}

/// One owned, thread-local direct recurrence execution workspace.
pub struct DirectRecurrenceExecutionRuntime {
    plan: DirectRecurrencePlan,
    executors: DirectExecutorCatalog,
    current_re: AlignedF64Buffer,
    current_im: AlignedF64Buffer,
    amplitude_re: AlignedF64Buffer,
    amplitude_im: AlignedF64Buffer,
    momenta: AlignedF64Buffer,
    parameters_re: AlignedF64Buffer,
    parameters_im: AlignedF64Buffer,
    factors_re: AlignedF64Buffer,
    factors_im: AlignedF64Buffer,
    union_source_dispatch: Option<DirectUnionSourceDispatchHandle>,
    additive_amplitude_ranges: Vec<Range<usize>>,
    momentum_form_count: u32,
    lorentz_component_count: u16,
    point_stride: u32,
    momentum_filled_points: u32,
    momentum_replay_selector: Option<DirectReplaySelectorIdentity>,
    last_point_count: u32,
    last_public_flow_id: Option<u32>,
    last_representative_flow_id: Option<u32>,
    counters: DirectExecutionCounters,
    activity_counters: DirectRuntimeActivityCounters,
    timings: DirectRuntimePhaseTimings,
}

impl DirectRecurrenceExecutionRuntime {
    /// Allocate a complete workspace and convert the plan's exact factors to
    /// their persistent binary64 split-complex catalog.
    pub fn new(
        plan: DirectRecurrencePlan,
        executors: DirectExecutorCatalog,
        lorentz_component_count: u16,
    ) -> RusticolResult<Self> {
        Self::new_inner(plan, executors, lorentz_component_count, None)
    }

    /// Construct an all-flow-union workspace with its authenticated SourceIR
    /// dispatcher. The dispatcher context is owned by the same loaded backend
    /// owner that backs `executors` and must outlive this runtime.
    pub fn new_with_union_source_dispatch(
        plan: DirectRecurrencePlan,
        executors: DirectExecutorCatalog,
        lorentz_component_count: u16,
        union_source_dispatch: DirectUnionSourceDispatchHandle,
    ) -> RusticolResult<Self> {
        if plan.strategy() != RecurrenceStrategy::AllFlowUnion {
            return Err(invalid(
                "union source dispatch requires an all-flow-union recurrence plan",
            ));
        }
        Self::new_inner(
            plan,
            executors,
            lorentz_component_count,
            Some(union_source_dispatch),
        )
    }

    fn new_inner(
        plan: DirectRecurrencePlan,
        executors: DirectExecutorCatalog,
        lorentz_component_count: u16,
        union_source_dispatch: Option<DirectUnionSourceDispatchHandle>,
    ) -> RusticolResult<Self> {
        if lorentz_component_count == 0 {
            return Err(invalid("Lorentz component count must be positive"));
        }

        let momentum_form_count = u32::try_from(plan.momentum_forms().len())
            .map_err(|_| invalid("momentum form count exceeds u32"))?;
        let split_complex_scalar_count = usize::try_from(plan.current_arena_components())
            .ok()
            .and_then(|current| current.checked_mul(2))
            .and_then(|current| {
                usize::try_from(plan.amplitude_destination_count())
                    .ok()
                    .and_then(|amplitude| amplitude.checked_mul(2))
                    .and_then(|amplitude| current.checked_add(amplitude))
            })
            .ok_or_else(|| invalid("split-complex per-point workspace size overflows usize"))?;
        let momentum_scalar_count = usize::try_from(momentum_form_count)
            .ok()
            .and_then(|forms| forms.checked_mul(usize::from(lorentz_component_count)))
            .ok_or_else(|| invalid("per-point momentum workspace size overflows usize"))?;
        let per_point_scalar_count = split_complex_scalar_count
            .checked_add(momentum_scalar_count)
            .ok_or_else(|| invalid("per-point workspace size overflows usize"))?;
        let per_point_bytes = per_point_scalar_count
            .checked_mul(size_of::<f64>())
            .ok_or_else(|| invalid("per-point workspace bytes overflow usize"))?;
        let workspace_bytes = usize::try_from(plan.workspace_mib())
            .ok()
            .and_then(|mib| mib.checked_mul(1024 * 1024))
            .ok_or_else(|| invalid("workspace byte limit overflows usize"))?;
        if per_point_bytes == 0 || per_point_bytes > workspace_bytes {
            return Err(invalid(format!(
                "one point requires {per_point_bytes} workspace bytes, exceeding the configured {workspace_bytes}"
            )));
        }
        let workspace_tile = workspace_bytes / per_point_bytes;
        let split_complex_per_point_bytes = split_complex_scalar_count
            .checked_mul(size_of::<f64>())
            .ok_or_else(|| invalid("split-complex per-point bytes overflow usize"))?;
        let cache_tile = greatest_power_of_two_not_exceeding(
            DIRECT_RUNTIME_CACHE_TARGET_BYTES / split_complex_per_point_bytes,
        );
        let point_stride = plan
            .point_tile_size()
            .min(u32::try_from(workspace_tile).unwrap_or(u32::MAX).max(1))
            .min(u32::try_from(cache_tile).unwrap_or(u32::MAX).max(1));
        let current_len = scalar_len(
            plan.current_arena_components(),
            point_stride,
            "current arena",
        )?;
        let amplitude_len = scalar_len(
            plan.amplitude_destination_count(),
            point_stride,
            "amplitude arena",
        )?;
        let momentum_plane_count = usize::try_from(momentum_form_count)
            .ok()
            .and_then(|forms| forms.checked_mul(usize::from(lorentz_component_count)))
            .ok_or_else(|| invalid("momentum plane count overflows usize"))?;
        let momentum_len = momentum_plane_count
            .checked_mul(point_stride as usize)
            .ok_or_else(|| invalid("momentum arena length overflows usize"))?;
        let parameter_len = usize::try_from(plan.parameter_value_count())
            .map_err(|_| invalid("parameter value count exceeds usize"))?;
        let factor_len = plan.exact_factors().len();

        let additive_amplitude_ranges = amplitude_clear_ranges(&plan)?;
        let current_re = AlignedF64Buffer::zeroed(current_len, "current real")?;
        let current_im = AlignedF64Buffer::zeroed(current_len, "current imaginary")?;
        let amplitude_re = AlignedF64Buffer::zeroed(amplitude_len, "amplitude real")?;
        let amplitude_im = AlignedF64Buffer::zeroed(amplitude_len, "amplitude imaginary")?;
        let momenta = AlignedF64Buffer::zeroed(momentum_len, "momentum")?;
        let parameters_re = AlignedF64Buffer::zeroed(parameter_len, "parameter real")?;
        let parameters_im = AlignedF64Buffer::zeroed(parameter_len, "parameter imaginary")?;
        let mut factors_re = AlignedF64Buffer::zeroed(factor_len, "factor real")?;
        let mut factors_im = AlignedF64Buffer::zeroed(factor_len, "factor imaginary")?;
        for ((factor_re, factor_im), factor) in factors_re
            .as_mut_slice()
            .iter_mut()
            .zip(factors_im.as_mut_slice())
            .zip(plan.exact_factors())
        {
            *factor_re = factor.real().numerator() as f64 / factor.real().denominator() as f64;
            *factor_im = factor.imag().numerator() as f64 / factor.imag().denominator() as f64;
        }

        Ok(Self {
            plan,
            executors,
            current_re,
            current_im,
            amplitude_re,
            amplitude_im,
            momenta,
            parameters_re,
            parameters_im,
            factors_re,
            factors_im,
            union_source_dispatch,
            additive_amplitude_ranges,
            momentum_form_count,
            lorentz_component_count,
            point_stride,
            momentum_filled_points: 0,
            momentum_replay_selector: None,
            last_point_count: 0,
            last_public_flow_id: None,
            last_representative_flow_id: None,
            counters: DirectExecutionCounters::default(),
            activity_counters: DirectRuntimeActivityCounters::default(),
            timings: DirectRuntimePhaseTimings::default(),
        })
    }

    pub const fn plan(&self) -> &DirectRecurrencePlan {
        &self.plan
    }

    pub const fn point_tile_size(&self) -> u32 {
        self.point_stride
    }

    pub const fn momentum_form_count(&self) -> u32 {
        self.momentum_form_count
    }

    pub const fn lorentz_component_count(&self) -> u16 {
        self.lorentz_component_count
    }

    /// Resolve one physical flow once. The returned selector owns its fixed
    /// source permutation and can be reused for every tile without allocation.
    pub fn prepare_replay_selector(
        &self,
        public_flow_id: u32,
    ) -> RusticolResult<DirectReplaySelectorPlan> {
        if self.plan.strategy() != RecurrenceStrategy::TopologyReplay {
            return Err(invalid(
                "replay selectors require a topology-replay recurrence plan",
            ));
        }
        let mut matches = self
            .plan
            .replay_targets()
            .iter()
            .filter(|target| target.public_flow_id == public_flow_id);
        let target = matches
            .next()
            .ok_or_else(|| invalid(format!("public flow {public_flow_id} is not retained")))?;
        if matches.next().is_some() {
            return Err(RusticolError::integrity(format!(
                "direct recurrence public flow {public_flow_id} has multiple replay targets"
            )));
        }
        if !self
            .plan
            .amplitude_destinations()
            .iter()
            .any(|destination| destination.target_sector_id == target.representative_id)
        {
            return Err(RusticolError::integrity(format!(
                "direct recurrence replay representative {} has no amplitude destination",
                target.representative_id
            )));
        }

        let start = usize::try_from(target.source_permutation_start).map_err(|_| {
            RusticolError::integrity(
                "direct recurrence replay source permutation start exceeds usize",
            )
        })?;
        let count = usize::try_from(target.source_permutation_count).map_err(|_| {
            RusticolError::integrity(
                "direct recurrence replay source permutation count exceeds usize",
            )
        })?;
        let end = start.checked_add(count).ok_or_else(|| {
            RusticolError::integrity(
                "direct recurrence replay source permutation range overflows usize",
            )
        })?;
        let permutation = self
            .plan
            .source_permutations()
            .get(start..end)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "direct recurrence replay source permutation is out of bounds",
                )
            })?;
        let mut source_permutation = Vec::new();
        source_permutation
            .try_reserve_exact(permutation.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not allocate direct recurrence replay selector: {error}"
                ))
            })?;
        source_permutation.extend_from_slice(permutation);

        let phase = *self
            .plan
            .exact_factors()
            .get(target.phase_exact_factor_id as usize)
            .ok_or_else(|| {
                RusticolError::integrity("direct recurrence replay phase factor is out of bounds")
            })?;
        Ok(DirectReplaySelectorPlan {
            runtime_layout_digest: self.plan.runtime_layout_digest(),
            public_flow_id,
            representative_flow_id: target.representative_id,
            source_permutation: source_permutation.into_boxed_slice(),
            phase_re: phase.real().numerator() as f64 / phase.real().denominator() as f64,
            phase_im: phase.imag().numerator() as f64 / phase.imag().denominator() as f64,
            multiplicity: target.multiplicity,
        })
    }

    /// Resolve one retained helicity once for all-flow-union execution.
    ///
    /// The returned selector is a pair of authenticated ranges into the
    /// immutable plan. Reusing it for later tiles performs no allocation.
    pub fn prepare_union_helicity_selector(
        &self,
        resolved_helicity_id: u32,
    ) -> RusticolResult<DirectUnionHelicitySelectorPlan> {
        if self.plan.strategy() != RecurrenceStrategy::AllFlowUnion {
            return Err(invalid(
                "union helicity selectors require an all-flow-union recurrence plan",
            ));
        }
        if self.union_source_dispatch.is_none() {
            return Err(invalid(
                "all-flow-union recurrence runtime has no source dispatcher",
            ));
        }
        let descriptor = self
            .plan
            .resolved_helicities()
            .get(resolved_helicity_id as usize)
            .filter(|descriptor| descriptor.id == resolved_helicity_id)
            .ok_or_else(|| {
                invalid(format!(
                    "resolved helicity {resolved_helicity_id} is not retained"
                ))
            })?;
        let (source_selection_start, source_selection_count) =
            self.validate_union_helicity_descriptor(descriptor)?;
        Ok(DirectUnionHelicitySelectorPlan {
            runtime_layout_digest: self.plan.runtime_layout_digest(),
            resolved_helicity_id,
            source_selection_start,
            source_selection_count,
        })
    }

    /// Mutable form-major, Lorentz-component-major momentum arena.
    pub fn momenta_mut(&mut self) -> &mut [f64] {
        self.momentum_filled_points = self.point_stride;
        self.momentum_replay_selector = None;
        self.momenta.as_mut_slice()
    }

    /// One persistent momentum plane, including its inactive tile tail.
    pub fn momentum_plane_mut(
        &mut self,
        momentum_form_id: u32,
        lorentz_component: u16,
    ) -> Option<&mut [f64]> {
        if momentum_form_id >= self.momentum_form_count
            || lorentz_component >= self.lorentz_component_count
        {
            return None;
        }
        self.momentum_filled_points = self.point_stride;
        self.momentum_replay_selector = None;
        let plane = momentum_form_id as usize * usize::from(self.lorentz_component_count)
            + usize::from(lorentz_component);
        let start = plane * self.point_stride as usize;
        self.momenta
            .as_mut_slice()
            .get_mut(start..start + self.point_stride as usize)
    }

    /// Fill every canonical momentum form from point-major external
    /// `[point][source_slot][lorentz_component]` four-momenta.
    pub fn fill_momenta_from_external(
        &mut self,
        selector: &DirectReplaySelectorPlan,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<()> {
        self.validate_replay_selector(selector)?;
        self.validate_point_count(point_count)?;
        if self.lorentz_component_count != 4 {
            return Err(invalid(format!(
                "external four-momentum fill requires 4 Lorentz components, runtime has {}",
                self.lorentz_component_count
            )));
        }
        let source_count = usize::try_from(self.plan.external_source_count())
            .map_err(|_| invalid("external source count exceeds usize"))?;
        let expected_len = (point_count as usize)
            .checked_mul(source_count)
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| invalid("point-major external momentum length overflows usize"))?;
        if external_four_momenta.len() != expected_len {
            return Err(invalid(format!(
                "point-major external momentum length is {}, expected {expected_len}",
                external_four_momenta.len()
            )));
        }

        let started = Instant::now();
        let point_stride = self.point_stride as usize;
        let active_points = point_count as usize;
        let previous_points = self.momentum_filled_points as usize;
        let form_count = self.plan.momentum_forms().len();
        let terms = self.plan.momentum_terms();
        let momenta = self.momenta.as_mut_slice();
        for (form_id, form) in self.plan.momentum_forms().iter().enumerate() {
            let term_start = usize::try_from(form.term_start).map_err(|_| {
                RusticolError::integrity("direct recurrence momentum term start exceeds usize")
            })?;
            let term_end = term_start
                .checked_add(form.term_count as usize)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "direct recurrence momentum term range overflows usize",
                    )
                })?;
            let form_terms = terms.get(term_start..term_end).ok_or_else(|| {
                RusticolError::integrity("direct recurrence momentum term range is out of bounds")
            })?;
            for component in 0..4 {
                let destination_start = (form_id * 4 + component) * point_stride;
                for point in 0..active_points {
                    let mut value = 0.0;
                    for term in form_terms {
                        let external_slot = selector
                            .source_permutation
                            .get(term.source_slot as usize)
                            .copied()
                            .ok_or_else(|| {
                                RusticolError::integrity(
                                    "direct recurrence momentum term source is outside the replay permutation",
                                )
                            })? as usize;
                        let source = (point * source_count + external_slot) * 4 + component;
                        value += f64::from(term.coefficient) * external_four_momenta[source];
                    }
                    momenta[destination_start + point] = value;
                }
            }
        }
        if active_points < previous_points {
            for plane in 0..form_count * 4 {
                let start = plane * point_stride + active_points;
                let end = plane * point_stride + previous_points;
                momenta[start..end].fill(0.0);
            }
        }

        self.momentum_filled_points = point_count;
        self.momentum_replay_selector = Some(selector.identity());
        self.activity_counters.momentum_fill_calls =
            self.activity_counters.momentum_fill_calls.saturating_add(1);
        self.activity_counters.momentum_forms_filled = self
            .activity_counters
            .momentum_forms_filled
            .saturating_add(form_count as u64);
        self.activity_counters.momentum_terms_filled = self
            .activity_counters
            .momentum_terms_filled
            .saturating_add(terms.len() as u64);
        let scalar_values = (form_count as u64)
            .saturating_mul(4)
            .saturating_mul(u64::from(point_count));
        self.activity_counters.momentum_scalar_values_filled = self
            .activity_counters
            .momentum_scalar_values_filled
            .saturating_add(scalar_values);
        self.timings.momentum_fill += started.elapsed();
        Ok(())
    }

    /// Fill canonical momentum forms without a replay permutation.
    fn fill_union_momenta_from_external(
        &mut self,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<()> {
        if self.plan.strategy() != RecurrenceStrategy::AllFlowUnion {
            return Err(invalid(
                "identity momentum fill requires an all-flow-union recurrence plan",
            ));
        }
        self.validate_point_count(point_count)?;
        if self.lorentz_component_count != 4 {
            return Err(invalid(format!(
                "external four-momentum fill requires 4 Lorentz components, runtime has {}",
                self.lorentz_component_count
            )));
        }
        let source_count = usize::try_from(self.plan.external_source_count())
            .map_err(|_| invalid("external source count exceeds usize"))?;
        let expected_len = (point_count as usize)
            .checked_mul(source_count)
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| invalid("point-major external momentum length overflows usize"))?;
        if external_four_momenta.len() != expected_len {
            return Err(invalid(format!(
                "point-major external momentum length is {}, expected {expected_len}",
                external_four_momenta.len()
            )));
        }

        let started = Instant::now();
        let point_stride = self.point_stride as usize;
        let active_points = point_count as usize;
        let previous_points = self.momentum_filled_points as usize;
        let form_count = self.plan.momentum_forms().len();
        let terms = self.plan.momentum_terms();
        let momenta = self.momenta.as_mut_slice();
        for (form_id, form) in self.plan.momentum_forms().iter().enumerate() {
            let term_start = usize::try_from(form.term_start).map_err(|_| {
                RusticolError::integrity("direct recurrence momentum term start exceeds usize")
            })?;
            let term_end = term_start
                .checked_add(form.term_count as usize)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "direct recurrence momentum term range overflows usize",
                    )
                })?;
            let form_terms = terms.get(term_start..term_end).ok_or_else(|| {
                RusticolError::integrity("direct recurrence momentum term range is out of bounds")
            })?;
            for component in 0..4 {
                let destination_start = (form_id * 4 + component) * point_stride;
                for point in 0..active_points {
                    let mut value = 0.0;
                    for term in form_terms {
                        let external_slot = term.source_slot as usize;
                        let source = (point * source_count + external_slot) * 4 + component;
                        value += f64::from(term.coefficient) * external_four_momenta[source];
                    }
                    momenta[destination_start + point] = value;
                }
            }
        }
        if active_points < previous_points {
            for plane in 0..form_count * 4 {
                let start = plane * point_stride + active_points;
                let end = plane * point_stride + previous_points;
                momenta[start..end].fill(0.0);
            }
        }

        self.momentum_filled_points = point_count;
        self.momentum_replay_selector = None;
        self.activity_counters.momentum_fill_calls =
            self.activity_counters.momentum_fill_calls.saturating_add(1);
        self.activity_counters.momentum_forms_filled = self
            .activity_counters
            .momentum_forms_filled
            .saturating_add(form_count as u64);
        self.activity_counters.momentum_terms_filled = self
            .activity_counters
            .momentum_terms_filled
            .saturating_add(terms.len() as u64);
        self.activity_counters.momentum_scalar_values_filled = self
            .activity_counters
            .momentum_scalar_values_filled
            .saturating_add(
                (form_count as u64)
                    .saturating_mul(4)
                    .saturating_mul(u64::from(point_count)),
            );
        self.timings.momentum_fill += started.elapsed();
        Ok(())
    }

    pub fn parameters_mut(&mut self) -> (&mut [f64], &mut [f64]) {
        (
            self.parameters_re.as_mut_slice(),
            self.parameters_im.as_mut_slice(),
        )
    }

    pub fn factors_mut(&mut self) -> (&mut [f64], &mut [f64]) {
        (
            self.factors_re.as_mut_slice(),
            self.factors_im.as_mut_slice(),
        )
    }

    pub fn set_parameters(&mut self, values_re: &[f64], values_im: &[f64]) -> RusticolResult<()> {
        validate_split_values("parameter", self.parameters_re.len(), values_re, values_im)?;
        self.parameters_re.as_mut_slice().copy_from_slice(values_re);
        self.parameters_im.as_mut_slice().copy_from_slice(values_im);
        Ok(())
    }

    pub fn set_factors(&mut self, values_re: &[f64], values_im: &[f64]) -> RusticolResult<()> {
        validate_split_values("factor", self.factors_re.len(), values_re, values_im)?;
        self.factors_re.as_mut_slice().copy_from_slice(values_re);
        self.factors_im.as_mut_slice().copy_from_slice(values_im);
        Ok(())
    }

    pub fn current_arenas(&self) -> (&[f64], &[f64]) {
        (self.current_re.as_slice(), self.current_im.as_slice())
    }

    pub fn amplitude_arenas(&self) -> (&[f64], &[f64]) {
        (self.amplitude_re.as_slice(), self.amplitude_im.as_slice())
    }

    /// Cumulative direct calls and rows since construction or the last reset.
    pub const fn counters(&self) -> DirectExecutionCounters {
        self.counters
    }

    pub const fn activity_counters(&self) -> DirectRuntimeActivityCounters {
        self.activity_counters
    }

    pub const fn phase_timings(&self) -> DirectRuntimePhaseTimings {
        self.timings
    }

    pub fn reset_counters(&mut self) {
        self.counters = DirectExecutionCounters::default();
        self.activity_counters = DirectRuntimeActivityCounters::default();
        self.timings = DirectRuntimePhaseTimings::default();
    }

    pub fn outputs(&self) -> Option<DirectRecurrenceTileOutput<'_>> {
        (self.last_point_count != 0).then(|| {
            self.borrowed_output(
                self.last_point_count,
                self.last_public_flow_id,
                self.last_representative_flow_id,
            )
        })
    }

    /// Execute one tile using the persistent inputs already held by this
    /// workspace.
    pub fn execute_tile(
        &mut self,
        point_count: u32,
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        if self.plan.strategy() == RecurrenceStrategy::AllFlowUnion {
            return Err(invalid(
                "all-flow-union execution requires a prepared runtime helicity selector",
            ));
        }
        self.execute_direct_tile(point_count)?;
        self.last_point_count = point_count;
        self.last_public_flow_id = None;
        self.last_representative_flow_id = None;
        Ok(self.borrowed_output(point_count, None, None))
    }

    /// Execute a tile whose momentum forms were filled for `selector`, then
    /// map representative amplitudes in place to the public flow convention.
    pub fn execute_replay_tile(
        &mut self,
        selector: &DirectReplaySelectorPlan,
        point_count: u32,
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.validate_replay_selector(selector)?;
        self.validate_point_count(point_count)?;
        if self.momentum_filled_points != point_count
            || self.momentum_replay_selector != Some(selector.identity())
        {
            return Err(invalid(
                "replay tile momentum forms were not filled for this selector and point count",
            ));
        }

        self.execute_direct_tile(point_count)?;
        let started = Instant::now();
        let scale_re = selector.phase_re * f64::from(selector.multiplicity);
        let scale_im = selector.phase_im * f64::from(selector.multiplicity);
        let point_stride = self.point_stride as usize;
        let active_points = point_count as usize;
        let mut scaled_values = 0_u64;
        for destination_id in 0..self.plan.amplitude_destinations().len() {
            let destination = self.plan.amplitude_destinations()[destination_id];
            if destination.target_sector_id != selector.representative_flow_id {
                continue;
            }
            let start = destination_id * point_stride;
            let end = start + active_points;
            let values_re = &mut self.amplitude_re.as_mut_slice()[start..end];
            let values_im = &mut self.amplitude_im.as_mut_slice()[start..end];
            for point in 0..active_points {
                let value_re = values_re[point];
                let value_im = values_im[point];
                values_re[point] = value_re * scale_re - value_im * scale_im;
                values_im[point] = value_re * scale_im + value_im * scale_re;
            }
            scaled_values = scaled_values.saturating_add(u64::from(point_count));
        }
        self.activity_counters.replay_schedule_executions = self
            .activity_counters
            .replay_schedule_executions
            .saturating_add(1);
        self.activity_counters.replay_output_values_scaled = self
            .activity_counters
            .replay_output_values_scaled
            .saturating_add(scaled_values);
        self.timings.replay_output_mapping += started.elapsed();
        self.last_point_count = point_count;
        self.last_public_flow_id = Some(selector.public_flow_id);
        self.last_representative_flow_id = Some(selector.representative_flow_id);
        Ok(self.borrowed_output(
            point_count,
            self.last_public_flow_id,
            self.last_representative_flow_id,
        ))
    }

    /// Fill and execute one selected replay tile without any intermediate
    /// momentum-form materialization by the caller.
    pub fn execute_replay_tile_from_external(
        &mut self,
        selector: &DirectReplaySelectorPlan,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.fill_momenta_from_external(selector, point_count, external_four_momenta)?;
        self.execute_replay_tile(selector, point_count)
    }

    /// Fill one runtime helicity's union sources and execute the compact
    /// all-flow schedule exactly once.
    pub fn execute_union_tile_from_external(
        &mut self,
        selector: &DirectUnionHelicitySelectorPlan,
        point_count: u32,
        external_four_momenta: &[f64],
    ) -> RusticolResult<DirectRecurrenceTileOutput<'_>> {
        self.validate_union_selector(selector)?;
        self.fill_union_momenta_from_external(point_count, external_four_momenta)?;
        self.execute_union_sources(selector, point_count)?;
        self.execute_direct_tile(point_count)?;
        self.activity_counters.union_schedule_executions = self
            .activity_counters
            .union_schedule_executions
            .saturating_add(1);
        self.last_point_count = point_count;
        self.last_public_flow_id = None;
        self.last_representative_flow_id = None;
        Ok(self.borrowed_output(point_count, None, None))
    }

    fn execute_union_sources(
        &mut self,
        selector: &DirectUnionHelicitySelectorPlan,
        point_count: u32,
    ) -> RusticolResult<()> {
        let handle = self
            .union_source_dispatch
            .ok_or_else(|| invalid("all-flow-union recurrence runtime has no source dispatcher"))?;
        let selection_end = selector
            .source_selection_start
            .checked_add(selector.source_selection_count)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "direct recurrence union source-selection range overflows usize",
                )
            })?;
        let selections = self
            .plan
            .resolved_source_selections()
            .get(selector.source_selection_start..selection_end)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "direct recurrence union source-selection range is out of bounds",
                )
            })?;

        let current_scalar_len = u64::try_from(self.current_re.len())
            .map_err(|_| invalid("current arena exceeds u64"))?;
        let amplitude_scalar_len = u64::try_from(self.amplitude_re.len())
            .map_err(|_| invalid("amplitude arena exceeds u64"))?;
        let momentum_scalar_len =
            u64::try_from(self.momenta.len()).map_err(|_| invalid("momentum arena exceeds u64"))?;
        let parameter_count = u32::try_from(self.parameters_re.len())
            .map_err(|_| invalid("parameter catalog exceeds u32"))?;
        let factor_count = u32::try_from(self.factors_re.len())
            .map_err(|_| invalid("factor catalog exceeds u32"))?;
        let source_count = u32::try_from(self.plan.sources().len())
            .map_err(|_| invalid("source row count exceeds u32"))?;
        let variant_count = u32::try_from(self.plan.source_dispatch_variants().len())
            .map_err(|_| invalid("source-dispatch variant count exceeds u32"))?;
        let embedding_count = u32::try_from(self.plan.source_embeddings().len())
            .map_err(|_| invalid("source embedding count exceeds u32"))?;
        let selection_count = u32::try_from(selections.len())
            .map_err(|_| invalid("source selection count exceeds u32"))?;

        let arena = DirectArenaView {
            current_re: self.current_re.as_mut_slice().as_mut_ptr(),
            current_im: self.current_im.as_mut_slice().as_mut_ptr(),
            current_scalar_len,
            amplitude_re: self.amplitude_re.as_mut_slice().as_mut_ptr(),
            amplitude_im: self.amplitude_im.as_mut_slice().as_mut_ptr(),
            amplitude_scalar_len,
            point_stride: self.point_stride,
        };
        let momenta = DirectMomentumView {
            values: self.momenta.as_ptr(),
            scalar_len: momentum_scalar_len,
            form_count: self.momentum_form_count,
            lorentz_component_count: self.lorentz_component_count,
            point_stride: self.point_stride,
        };
        let parameters = DirectParameterView {
            values_re: self.parameters_re.as_ptr(),
            values_im: self.parameters_im.as_ptr(),
            value_count: parameter_count,
        };
        let factors = DirectFactorView {
            values_re: self.factors_re.as_ptr(),
            values_im: self.factors_im.as_ptr(),
            value_count: factor_count,
        };

        let started = Instant::now();
        let status = unsafe {
            (handle.call)(
                handle.context,
                arena,
                momenta,
                parameters,
                factors,
                self.plan.sources().as_ptr(),
                source_count,
                self.plan.source_dispatch_variants().as_ptr(),
                variant_count,
                self.plan.source_embeddings().as_ptr(),
                embedding_count,
                selections.as_ptr(),
                selection_count,
                point_count,
            )
        };
        self.timings.union_source_fill += started.elapsed();
        if status != DIRECT_STATUS_OK {
            return Err(RusticolError::evaluation(format!(
                "direct recurrence union source dispatcher returned status {status}"
            )));
        }
        self.counters.source_calls = self.counters.source_calls.saturating_add(1);
        self.counters.source_rows = self
            .counters
            .source_rows
            .saturating_add(u64::from(selection_count));
        self.activity_counters.union_source_dispatch_calls = self
            .activity_counters
            .union_source_dispatch_calls
            .saturating_add(1);
        self.activity_counters.union_source_rows = self
            .activity_counters
            .union_source_rows
            .saturating_add(u64::from(selection_count));
        Ok(())
    }

    fn execute_direct_tile(&mut self, point_count: u32) -> RusticolResult<()> {
        self.validate_point_count(point_count)?;
        self.last_point_count = 0;
        self.last_public_flow_id = None;
        self.last_representative_flow_id = None;
        let active_points = point_count as usize;
        let point_stride = self.point_stride as usize;
        // `execute_direct_plan` initializes every current stage exactly once
        // before its first contribution group. Re-clearing all current planes
        // here only duplicates that work.
        clear_active_planes(
            self.amplitude_re.as_mut_slice(),
            &self.additive_amplitude_ranges,
            point_stride,
            active_points,
        );
        clear_active_planes(
            self.amplitude_im.as_mut_slice(),
            &self.additive_amplitude_ranges,
            point_stride,
            active_points,
        );

        {
            let mut workspace = DirectWorkspace {
                current_re: self.current_re.as_mut_slice(),
                current_im: self.current_im.as_mut_slice(),
                amplitude_re: self.amplitude_re.as_mut_slice(),
                amplitude_im: self.amplitude_im.as_mut_slice(),
                momenta: self.momenta.as_slice(),
                momentum_form_count: self.momentum_form_count,
                lorentz_component_count: self.lorentz_component_count,
                parameters_re: self.parameters_re.as_slice(),
                parameters_im: self.parameters_im.as_slice(),
                factors_re: self.factors_re.as_slice(),
                factors_im: self.factors_im.as_slice(),
                point_stride: self.point_stride,
            };
            let started = Instant::now();
            let result = execute_direct_plan(
                &self.plan,
                &self.executors,
                &mut workspace,
                point_count,
                &mut self.counters,
            );
            self.timings.direct_execution += started.elapsed();
            result?;
        }

        self.activity_counters.schedule_executions =
            self.activity_counters.schedule_executions.saturating_add(1);
        Ok(())
    }

    fn borrowed_output(
        &self,
        point_count: u32,
        public_flow_id: Option<u32>,
        representative_flow_id: Option<u32>,
    ) -> DirectRecurrenceTileOutput<'_> {
        DirectRecurrenceTileOutput {
            amplitude_re: self.amplitude_re.as_slice(),
            amplitude_im: self.amplitude_im.as_slice(),
            amplitude_destinations: self.plan.amplitude_destinations(),
            point_count,
            point_stride: self.point_stride,
            destination_count: self.plan.amplitude_destination_count(),
            public_flow_id,
            representative_flow_id,
        }
    }

    fn validate_point_count(&self, point_count: u32) -> RusticolResult<()> {
        if point_count == 0 {
            return Err(invalid("point count must be positive"));
        }
        if point_count > self.point_stride {
            return Err(invalid(format!(
                "point count {point_count} exceeds point tile size {}",
                self.point_stride
            )));
        }
        Ok(())
    }

    fn validate_replay_selector(&self, selector: &DirectReplaySelectorPlan) -> RusticolResult<()> {
        if selector.runtime_layout_digest != self.plan.runtime_layout_digest() {
            return Err(RusticolError::integrity(
                "direct recurrence replay selector belongs to a different plan",
            ));
        }
        if selector.source_permutation.len() != self.plan.external_source_count() as usize {
            return Err(RusticolError::integrity(
                "direct recurrence replay selector source permutation has an inconsistent length",
            ));
        }
        Ok(())
    }

    fn validate_union_helicity_descriptor(
        &self,
        descriptor: &DirectResolvedHelicityDescriptor,
    ) -> RusticolResult<(usize, usize)> {
        let start = usize::try_from(descriptor.source_selection_start).map_err(|_| {
            RusticolError::integrity("direct recurrence union source-selection start exceeds usize")
        })?;
        let count = usize::try_from(descriptor.source_selection_count).map_err(|_| {
            RusticolError::integrity("direct recurrence union source-selection count exceeds usize")
        })?;
        if count != self.plan.external_source_count() as usize {
            return Err(RusticolError::integrity(
                "direct recurrence union helicity does not select every external source",
            ));
        }
        let end = start.checked_add(count).ok_or_else(|| {
            RusticolError::integrity(
                "direct recurrence union source-selection range overflows usize",
            )
        })?;
        let selections = self
            .plan
            .resolved_source_selections()
            .get(start..end)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "direct recurrence union source-selection range is out of bounds",
                )
            })?;
        for (source_slot, selection) in selections.iter().enumerate() {
            if selection.source_slot != source_slot as u32 {
                return Err(RusticolError::integrity(
                    "direct recurrence union source selections are not in source-slot order",
                ));
            }
            let variant = self
                .plan
                .source_dispatch_variants()
                .get(selection.dispatch_variant_id as usize)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "direct recurrence union source selection references an absent variant",
                    )
                })?;
            let source = self
                .plan
                .sources()
                .get(variant.source_row_id as usize)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "direct recurrence union source variant references an absent row",
                    )
                })?;
            if source.source_slot != selection.source_slot
                || source.source_template_or_dispatch_domain != variant.dispatch_domain_id
            {
                return Err(RusticolError::integrity(
                    "direct recurrence union source selection disagrees with its variant",
                ));
            }
            let embedding_start = usize::try_from(variant.embedding_start).map_err(|_| {
                RusticolError::integrity(
                    "direct recurrence union source embedding start exceeds usize",
                )
            })?;
            let embedding_end = embedding_start
                .checked_add(variant.embedding_count as usize)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "direct recurrence union source embedding range overflows usize",
                    )
                })?;
            if self
                .plan
                .source_embeddings()
                .get(embedding_start..embedding_end)
                .is_none()
            {
                return Err(RusticolError::integrity(
                    "direct recurrence union source embedding range is out of bounds",
                ));
            }
        }
        Ok((start, count))
    }

    fn validate_union_selector(
        &self,
        selector: &DirectUnionHelicitySelectorPlan,
    ) -> RusticolResult<()> {
        if self.plan.strategy() != RecurrenceStrategy::AllFlowUnion {
            return Err(invalid(
                "union helicity selector requires an all-flow-union recurrence plan",
            ));
        }
        if selector.runtime_layout_digest != self.plan.runtime_layout_digest() {
            return Err(RusticolError::integrity(
                "direct recurrence union helicity selector belongs to a different plan",
            ));
        }
        let descriptor = self
            .plan
            .resolved_helicities()
            .get(selector.resolved_helicity_id as usize)
            .filter(|descriptor| descriptor.id == selector.resolved_helicity_id)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "direct recurrence union helicity selector references an absent helicity",
                )
            })?;
        let (start, count) = self.validate_union_helicity_descriptor(descriptor)?;
        if start != selector.source_selection_start || count != selector.source_selection_count {
            return Err(RusticolError::integrity(
                "direct recurrence union helicity selector ranges do not match the plan",
            ));
        }
        Ok(())
    }
}

struct AlignedF64Buffer {
    storage: Vec<f64>,
    start: usize,
    len: usize,
}

impl AlignedF64Buffer {
    fn zeroed(len: usize, label: &str) -> RusticolResult<Self> {
        let alignment_values = DIRECT_RUNTIME_ARENA_ALIGNMENT
            .checked_div(size_of::<f64>())
            .filter(|value| *value != 0)
            .ok_or_else(|| invalid("arena alignment is invalid for binary64 storage"))?;
        let storage_len = len
            .checked_add(alignment_values - 1)
            .ok_or_else(|| invalid(format!("{label} arena length overflows usize")))?;
        let mut storage = Vec::new();
        storage.try_reserve_exact(storage_len).map_err(|error| {
            RusticolError::internal(format!(
                "could not allocate direct recurrence {label} arena: {error}"
            ))
        })?;
        storage.resize(storage_len, 0.0);

        let address = storage.as_ptr() as usize;
        let byte_offset = (DIRECT_RUNTIME_ARENA_ALIGNMENT
            - address % DIRECT_RUNTIME_ARENA_ALIGNMENT)
            % DIRECT_RUNTIME_ARENA_ALIGNMENT;
        if !byte_offset.is_multiple_of(size_of::<f64>()) {
            return Err(RusticolError::internal(
                "direct recurrence arena allocation cannot provide binary64 alignment",
            ));
        }
        let start = byte_offset / size_of::<f64>();
        if start.checked_add(len).is_none_or(|end| end > storage.len()) {
            return Err(RusticolError::internal(
                "direct recurrence aligned arena range exceeds its allocation",
            ));
        }

        Ok(Self {
            storage,
            start,
            len,
        })
    }

    const fn len(&self) -> usize {
        self.len
    }

    fn as_ptr(&self) -> *const f64 {
        self.as_slice().as_ptr()
    }

    fn as_slice(&self) -> &[f64] {
        &self.storage[self.start..self.start + self.len]
    }

    fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.storage[self.start..self.start + self.len]
    }
}

fn scalar_len(plane_count: u32, point_stride: u32, label: &str) -> RusticolResult<usize> {
    usize::try_from(plane_count)
        .ok()
        .and_then(|planes| planes.checked_mul(point_stride as usize))
        .ok_or_else(|| invalid(format!("{label} length overflows usize")))
}

fn greatest_power_of_two_not_exceeding(value: usize) -> usize {
    if value == 0 {
        return 1;
    }
    1_usize << (usize::BITS - 1 - value.leading_zeros())
}

fn amplitude_clear_ranges(plan: &DirectRecurrencePlan) -> RusticolResult<Vec<Range<usize>>> {
    let destination_count = usize::try_from(plan.amplitude_destination_count())
        .map_err(|_| invalid("amplitude destination count exceeds usize"))?;
    let mut marked = zeroed_marks(destination_count, "amplitude clear map")?;
    for row in plan.closures() {
        mark_range(
            &mut marked,
            row.amplitude_destination_id as usize,
            1,
            "amplitude",
        )?;
    }
    compact_ranges(&marked, "amplitude")
}

fn zeroed_marks(len: usize, label: &str) -> RusticolResult<Vec<u8>> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        RusticolError::internal(format!(
            "could not allocate direct recurrence {label}: {error}"
        ))
    })?;
    values.resize(len, 0);
    Ok(values)
}

fn mark_range(marked: &mut [u8], start: usize, count: usize, label: &str) -> RusticolResult<()> {
    let end = start
        .checked_add(count)
        .ok_or_else(|| RusticolError::integrity(format!("{label} clear range overflows usize")))?;
    let range = marked.get_mut(start..end).ok_or_else(|| {
        RusticolError::integrity(format!("{label} clear range exceeds its arena"))
    })?;
    range.fill(1);
    Ok(())
}

fn compact_ranges(marked: &[u8], label: &str) -> RusticolResult<Vec<Range<usize>>> {
    let mut ranges = Vec::new();
    ranges.try_reserve_exact(marked.len()).map_err(|error| {
        RusticolError::internal(format!(
            "could not allocate direct recurrence {label} clear ranges: {error}"
        ))
    })?;
    let mut index = 0;
    while index < marked.len() {
        if marked[index] == 0 {
            index += 1;
            continue;
        }
        let start = index;
        while index < marked.len() && marked[index] != 0 {
            index += 1;
        }
        ranges.push(start..index);
    }
    Ok(ranges)
}

fn clear_active_planes(
    values: &mut [f64],
    ranges: &[Range<usize>],
    point_stride: usize,
    active_points: usize,
) {
    for range in ranges {
        for plane in range.start..range.end {
            let start = plane * point_stride;
            values[start..start + active_points].fill(0.0);
        }
    }
}

fn validate_split_values(
    label: &str,
    expected_len: usize,
    values_re: &[f64],
    values_im: &[f64],
) -> RusticolResult<()> {
    if values_re.len() != expected_len || values_im.len() != expected_len {
        return Err(invalid(format!(
            "{label} split-complex lengths are {}, {}, expected {expected_len}",
            values_re.len(),
            values_im.len()
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!(
        "direct recurrence execution runtime: {}",
        message.into()
    ))
}

#[cfg(test)]
#[path = "direct_runtime_tests.rs"]
mod tests;
