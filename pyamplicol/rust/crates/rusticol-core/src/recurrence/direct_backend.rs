// SPDX-License-Identifier: 0BSD

//! Direct-arena prepared executor ABI for compact recurrence schedules.
//!
//! This module deliberately does not depend on the eager runtime. Direct
//! executors consume fixed-width recurrence rows and persistent split-complex
//! arenas. They never receive a packed evaluator input or output buffer.

use super::RecurrenceStrategy;
use super::SemanticDigest;
use super::direct_plan::{
    DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE, DIRECT_NONE_U32, DirectClosureRow,
    DirectContributionRow, DirectExecutorRole, DirectFinalizationRow, DirectNodeKind,
    DirectRecurrencePlan, DirectRowGroupDescriptor, DirectSourceRow,
};
pub use super::direct_plan::{
    DirectResolvedSourceSelection, DirectSourceDispatchVariantDescriptor, DirectSourceEmbeddingRow,
};
use crate::direct_arena::{DirectArenaTrafficCounters, clear_split_active_range};
pub use crate::direct_arena::{
    DirectArenaView, DirectFactorView, DirectMomentumView, DirectParameterView,
};
use crate::{RusticolError, RusticolResult};
use std::cell::RefCell;
use std::ffi::{c_int, c_void};
use std::time::{Duration, Instant};

pub const RECURRENCE_DIRECT_BACKEND_ABI: &str = "rusticol.recurrence-direct-backend.v1";

pub const DIRECT_STATUS_OK: c_int = 0;

thread_local! {
    static DIRECT_EXECUTOR_ERROR_DETAIL: RefCell<Option<RusticolError>> = const { RefCell::new(None) };
}

/// Clear the synchronous detail channel before executing one direct plan.
pub(crate) fn clear_direct_executor_error_detail() {
    DIRECT_EXECUTOR_ERROR_DETAIL.with(|detail| *detail.borrow_mut() = None);
}

/// Preserve a concrete callback failure across the narrow integer-status ABI.
pub(crate) fn record_direct_executor_error_detail(detail: RusticolError) {
    DIRECT_EXECUTOR_ERROR_DETAIL.with(|slot| *slot.borrow_mut() = Some(detail));
}

fn take_direct_executor_error_detail() -> Option<RusticolError> {
    DIRECT_EXECUTOR_ERROR_DETAIL.with(|detail| detail.borrow_mut().take())
}

pub type DirectSourceExecutor = unsafe extern "C" fn(
    *const c_void,
    DirectArenaView,
    DirectMomentumView,
    DirectParameterView,
    DirectFactorView,
    *const DirectSourceRow,
    u32,
    u32,
) -> c_int;

/// Runtime-helicity source dispatcher used only by all-flow-union plans.
///
/// Unlike a static source executor, this callable receives one authenticated
/// resolved selection plus the process-bound variant and embedding catalogs.
/// It fills every full source current before the remaining direct schedule is
/// executed. The pointed-to tables and context must outlive the call.
pub type DirectUnionSourceDispatchExecutor = unsafe extern "C" fn(
    *const c_void,
    DirectArenaView,
    DirectMomentumView,
    DirectParameterView,
    DirectFactorView,
    *const DirectSourceRow,
    u32,
    *const DirectSourceDispatchVariantDescriptor,
    u32,
    *const DirectSourceEmbeddingRow,
    u32,
    *const DirectResolvedSourceSelection,
    u32,
    u32,
) -> c_int;

/// Typed handle for one immutable all-flow-union source dispatcher.
#[derive(Clone, Copy)]
pub struct DirectUnionSourceDispatchHandle {
    pub call: DirectUnionSourceDispatchExecutor,
    pub context: *const c_void,
}

pub type DirectContributionExecutor = unsafe extern "C" fn(
    *const c_void,
    DirectArenaView,
    DirectMomentumView,
    DirectParameterView,
    DirectFactorView,
    *const DirectContributionRow,
    u32,
    u32,
) -> c_int;

pub type DirectFinalizationExecutor = unsafe extern "C" fn(
    *const c_void,
    DirectArenaView,
    DirectMomentumView,
    DirectParameterView,
    DirectFactorView,
    *const DirectFinalizationRow,
    u32,
    u32,
) -> c_int;

pub type DirectClosureExecutor = unsafe extern "C" fn(
    *const c_void,
    DirectArenaView,
    DirectMomentumView,
    DirectParameterView,
    DirectFactorView,
    *const DirectClosureRow,
    u32,
    u32,
) -> c_int;

#[derive(Clone, Copy)]
pub enum DirectExecutorHandle {
    Source {
        call: DirectSourceExecutor,
        context: *const c_void,
    },
    Contribution {
        call: DirectContributionExecutor,
        context: *const c_void,
    },
    Finalization {
        call: DirectFinalizationExecutor,
        context: *const c_void,
    },
    Closure {
        call: DirectClosureExecutor,
        context: *const c_void,
    },
}

impl DirectExecutorHandle {
    pub(crate) const fn role(self) -> DirectExecutorRole {
        match self {
            Self::Source { .. } => DirectExecutorRole::Source,
            Self::Contribution { .. } => DirectExecutorRole::Contribution,
            Self::Finalization { .. } => DirectExecutorRole::Finalization,
            Self::Closure { .. } => DirectExecutorRole::Closure,
        }
    }
}

pub struct DirectExecutorCatalog {
    handles: Box<[DirectExecutorHandle]>,
    plan_layout_digest: SemanticDigest,
    direct_template_catalog_digest: SemanticDigest,
}

impl DirectExecutorCatalog {
    pub fn new(
        plan: &DirectRecurrencePlan,
        direct_template_catalog_digest: SemanticDigest,
        handles: Vec<DirectExecutorHandle>,
    ) -> RusticolResult<Self> {
        if handles.is_empty() {
            return Err(RusticolError::invalid_argument(
                "direct recurrence executor catalog must not be empty",
            ));
        }
        if direct_template_catalog_digest != plan.direct_template_catalog_digest() {
            return Err(RusticolError::integrity(format!(
                "loaded direct-template catalog digest {direct_template_catalog_digest} does not match plan {}",
                plan.direct_template_catalog_digest()
            )));
        }
        let catalog = Self {
            handles: handles.into_boxed_slice(),
            plan_layout_digest: plan.runtime_layout_digest(),
            direct_template_catalog_digest,
        };
        catalog.validate_for_plan(plan)?;
        Ok(catalog)
    }

    fn require(
        &self,
        executor_id: u32,
        role: DirectExecutorRole,
    ) -> RusticolResult<DirectExecutorHandle> {
        let handle = self
            .handles
            .get(executor_id as usize)
            .copied()
            .ok_or_else(|| {
                RusticolError::evaluation(format!(
                    "direct recurrence executor {executor_id} is not loaded"
                ))
            })?;
        if handle.role() != role {
            return Err(RusticolError::integrity(format!(
                "direct recurrence executor {executor_id} has role {:?}, expected {role:?}",
                handle.role()
            )));
        }
        Ok(handle)
    }

    fn validate_for_plan(&self, plan: &DirectRecurrencePlan) -> RusticolResult<()> {
        if self.handles.len() != plan.direct_executor_count() as usize {
            return Err(RusticolError::integrity(format!(
                "direct recurrence executor catalog has {} entries, expected {}",
                self.handles.len(),
                plan.direct_executor_count()
            )));
        }
        if self.direct_template_catalog_digest != plan.direct_template_catalog_digest() {
            return Err(RusticolError::integrity(
                "direct recurrence executor catalog belongs to a different prepared template catalog",
            ));
        }
        for descriptor in plan.row_groups() {
            if descriptor.role == DirectExecutorRole::Source
                && descriptor.direct_executor_id == DIRECT_NONE_U32
                && plan.strategy() == RecurrenceStrategy::AllFlowUnion
            {
                continue;
            }
            if descriptor.role == DirectExecutorRole::Contribution
                && descriptor.direct_executor_id == DIRECT_NONE_U32
            {
                continue;
            }
            self.require(descriptor.direct_executor_id, descriptor.role)?;
        }
        Ok(())
    }
}

pub struct DirectWorkspace<'a> {
    pub current_re: &'a mut [f64],
    pub current_im: &'a mut [f64],
    pub amplitude_re: &'a mut [f64],
    pub amplitude_im: &'a mut [f64],
    pub momenta: &'a [f64],
    pub momentum_form_count: u32,
    pub lorentz_component_count: u16,
    pub parameters_re: &'a [f64],
    pub parameters_im: &'a [f64],
    pub factors_re: &'a [f64],
    pub factors_im: &'a [f64],
    pub point_stride: u32,
}

impl DirectWorkspace<'_> {
    fn validate(&self, point_count: u32) -> RusticolResult<()> {
        if point_count == 0 || self.point_stride == 0 || point_count > self.point_stride {
            return Err(RusticolError::invalid_argument(
                "direct recurrence point count must be within the workspace stride",
            ));
        }
        if self.current_re.len() != self.current_im.len() {
            return Err(RusticolError::invalid_argument(
                "direct recurrence current real and imaginary arenas differ in length",
            ));
        }
        if self.amplitude_re.len() != self.amplitude_im.len() {
            return Err(RusticolError::invalid_argument(
                "direct recurrence amplitude real and imaginary arenas differ in length",
            ));
        }
        let point_stride = self.point_stride as usize;
        if !self.current_re.len().is_multiple_of(point_stride)
            || !self.amplitude_re.len().is_multiple_of(point_stride)
        {
            return Err(RusticolError::invalid_argument(
                "direct recurrence arena does not contain whole point-stride planes",
            ));
        }
        if self.parameters_re.len() != self.parameters_im.len()
            || self.factors_re.len() != self.factors_im.len()
        {
            return Err(RusticolError::invalid_argument(
                "direct recurrence split-complex catalogs differ in length",
            ));
        }
        let expected_momenta = usize::try_from(self.momentum_form_count)
            .ok()
            .and_then(|forms| forms.checked_mul(usize::from(self.lorentz_component_count)))
            .and_then(|planes| planes.checked_mul(self.point_stride as usize))
            .ok_or_else(|| {
                RusticolError::invalid_argument("direct recurrence momentum arena overflows")
            })?;
        if self.momenta.len() != expected_momenta {
            return Err(RusticolError::invalid_argument(
                "direct recurrence momentum arena has an inconsistent length",
            ));
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
        let current_scalar_len = u64::try_from(self.current_re.len()).map_err(|_| {
            RusticolError::invalid_argument("direct recurrence current arena exceeds u64")
        })?;
        let amplitude_scalar_len = u64::try_from(self.amplitude_re.len()).map_err(|_| {
            RusticolError::invalid_argument("direct recurrence amplitude arena exceeds u64")
        })?;
        let momentum_scalar_len = u64::try_from(self.momenta.len()).map_err(|_| {
            RusticolError::invalid_argument("direct recurrence momentum arena exceeds u64")
        })?;
        let parameter_count = u32::try_from(self.parameters_re.len()).map_err(|_| {
            RusticolError::invalid_argument("direct recurrence parameter catalog exceeds u32")
        })?;
        let factor_count = u32::try_from(self.factors_re.len()).map_err(|_| {
            RusticolError::invalid_argument("direct recurrence factor catalog exceeds u32")
        })?;
        Ok((
            DirectArenaView {
                current_re: self.current_re.as_mut_ptr(),
                current_im: self.current_im.as_mut_ptr(),
                current_scalar_len,
                amplitude_re: self.amplitude_re.as_mut_ptr(),
                amplitude_im: self.amplitude_im.as_mut_ptr(),
                amplitude_scalar_len,
                point_stride: self.point_stride,
            },
            DirectMomentumView {
                values: self.momenta.as_ptr(),
                scalar_len: momentum_scalar_len,
                form_count: self.momentum_form_count,
                lorentz_component_count: self.lorentz_component_count,
                point_stride: self.point_stride,
            },
            DirectParameterView {
                values_re: self.parameters_re.as_ptr(),
                values_im: self.parameters_im.as_ptr(),
                value_count: parameter_count,
            },
            DirectFactorView {
                values_re: self.factors_re.as_ptr(),
                values_im: self.factors_im.as_ptr(),
                value_count: factor_count,
            },
        ))
    }

    fn clear_current_stage(
        &mut self,
        plan: &DirectRecurrencePlan,
        stage: u16,
        point_count: u32,
        selected_sector_id: Option<u32>,
    ) -> RusticolResult<()> {
        let current_plane_count = u32::try_from(self.current_re.len() / self.point_stride as usize)
            .map_err(|_| {
                RusticolError::integrity(
                    "direct recurrence current plane count exceeds u32 during stage clear",
                )
            })?;
        for current in plan.currents().iter().filter(|current| {
            current.node_kind == DirectNodeKind::Current && current.stage == stage
        }) {
            if let Some(sector_id) = selected_sector_id
                && !plan.selector_domain_contains(current.selector_domain_id, sector_id)?
            {
                continue;
            }
            clear_split_active_range(
                self.current_re,
                self.current_im,
                current_plane_count,
                self.point_stride,
                point_count,
                current.component_base,
                u32::from(current.component_count),
                "recurrence current arena",
            )?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectExecutionCounters {
    pub source_calls: u64,
    pub source_rows: u64,
    pub contribution_calls: u64,
    pub contribution_rows: u64,
    pub finalization_calls: u64,
    pub finalization_rows: u64,
    pub closure_calls: u64,
    pub closure_rows: u64,
    pub packed_input_bytes: u64,
    pub packed_output_bytes: u64,
    pub scatter_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DirectExecutionRoleTimings {
    pub source: Duration,
    pub contribution: Duration,
    pub finalization: Duration,
    pub closure: Duration,
}

pub fn execute_direct_plan(
    plan: &DirectRecurrencePlan,
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
    counters: &mut DirectExecutionCounters,
) -> RusticolResult<()> {
    let mut unused_timings = DirectExecutionRoleTimings::default();
    execute_direct_plan_impl::<true>(
        plan,
        plan.row_groups(),
        None,
        executors,
        workspace,
        point_count,
        counters,
        &mut unused_timings,
        None,
    )
}

pub fn execute_direct_plan_profiled(
    plan: &DirectRecurrencePlan,
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
    counters: &mut DirectExecutionCounters,
    timings: &mut DirectExecutionRoleTimings,
) -> RusticolResult<()> {
    execute_direct_plan_impl::<true>(
        plan,
        plan.row_groups(),
        None,
        executors,
        workspace,
        point_count,
        counters,
        timings,
        None,
    )
}

pub(crate) fn execute_direct_plan_profiled_with_traffic(
    plan: &DirectRecurrencePlan,
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
    counters: &mut DirectExecutionCounters,
    timings: &mut DirectExecutionRoleTimings,
    traffic: &mut DirectArenaTrafficCounters,
) -> RusticolResult<()> {
    execute_direct_plan_impl::<true>(
        plan,
        plan.row_groups(),
        None,
        executors,
        workspace,
        point_count,
        counters,
        timings,
        Some(traffic),
    )
}

/// Execute the authenticated schedule without touching profiling counters.
///
/// This is a distinct const-generic specialization so release builds remove
/// every role-counter branch from the ordinary native evaluation path.
pub fn execute_direct_plan_unprofiled(
    plan: &DirectRecurrencePlan,
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<()> {
    let mut unused = DirectExecutionCounters::default();
    let mut unused_timings = DirectExecutionRoleTimings::default();
    execute_direct_plan_impl::<false>(
        plan,
        plan.row_groups(),
        None,
        executors,
        workspace,
        point_count,
        &mut unused,
        &mut unused_timings,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_direct_plan_selected_profiled_with_traffic(
    plan: &DirectRecurrencePlan,
    row_groups: &[DirectRowGroupDescriptor],
    selected_sector_id: u32,
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
    counters: &mut DirectExecutionCounters,
    timings: &mut DirectExecutionRoleTimings,
    traffic: &mut DirectArenaTrafficCounters,
) -> RusticolResult<()> {
    execute_direct_plan_impl::<true>(
        plan,
        row_groups,
        Some(selected_sector_id),
        executors,
        workspace,
        point_count,
        counters,
        timings,
        Some(traffic),
    )
}

pub(crate) fn execute_direct_plan_selected_unprofiled(
    plan: &DirectRecurrencePlan,
    row_groups: &[DirectRowGroupDescriptor],
    selected_sector_id: u32,
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<()> {
    let mut unused = DirectExecutionCounters::default();
    let mut unused_timings = DirectExecutionRoleTimings::default();
    execute_direct_plan_impl::<false>(
        plan,
        row_groups,
        Some(selected_sector_id),
        executors,
        workspace,
        point_count,
        &mut unused,
        &mut unused_timings,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_direct_plan_impl<const PROFILE: bool>(
    plan: &DirectRecurrencePlan,
    row_groups: &[DirectRowGroupDescriptor],
    selected_sector_id: Option<u32>,
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
    counters: &mut DirectExecutionCounters,
    timings: &mut DirectExecutionRoleTimings,
    mut traffic: Option<&mut DirectArenaTrafficCounters>,
) -> RusticolResult<()> {
    clear_direct_executor_error_detail();
    workspace.validate(point_count)?;
    if executors.plan_layout_digest != plan.runtime_layout_digest() {
        return Err(RusticolError::integrity(
            "direct recurrence executor catalog belongs to a different plan",
        ));
    }
    let mut initialized_contribution_stage = None;
    for descriptor in row_groups {
        if descriptor.role == DirectExecutorRole::Source
            && descriptor.direct_executor_id == DIRECT_NONE_U32
            && plan.strategy() == RecurrenceStrategy::AllFlowUnion
        {
            // The selected union source variants were filled once by
            // `DirectRecurrenceExecutionRuntime` before entering this static
            // contribution/finalization/closure schedule.
            continue;
        }
        if descriptor.role == DirectExecutorRole::Contribution
            && initialized_contribution_stage != Some(descriptor.stage)
        {
            workspace.clear_current_stage(
                plan,
                descriptor.stage,
                point_count,
                selected_sector_id,
            )?;
            initialized_contribution_stage = Some(descriptor.stage);
        }
        let start = usize::try_from(descriptor.row_start).map_err(|_| {
            RusticolError::integrity("direct recurrence row-group start exceeds usize")
        })?;
        let count = usize::try_from(descriptor.row_count).map_err(|_| {
            RusticolError::integrity("direct recurrence row-group count exceeds usize")
        })?;
        let end = start.checked_add(count).ok_or_else(|| {
            RusticolError::integrity("direct recurrence row-group range overflows usize")
        })?;
        if descriptor.role == DirectExecutorRole::Contribution
            && descriptor.direct_executor_id == DIRECT_NONE_U32
        {
            let rows = plan.contributions().get(start..end).ok_or_else(|| {
                RusticolError::integrity("certified-reuse contribution row group is out of bounds")
            })?;
            let started = PROFILE.then(Instant::now);
            execute_certified_reuse_rows(rows, workspace, point_count)?;
            if PROFILE {
                counters.contribution_calls += 1;
                counters.contribution_rows += u64::from(descriptor.row_count);
                timings.contribution += started
                    .expect("profiled certified-reuse group has a start time")
                    .elapsed();
                if let Some(traffic) = traffic.as_deref_mut() {
                    traffic.record_call(descriptor.row_count, point_count);
                }
            }
            continue;
        }
        let (arena, momenta, parameters, factors) = workspace.raw_views()?;
        let handle = executors.require(descriptor.direct_executor_id, descriptor.role)?;
        if PROFILE && let Some(traffic) = traffic.as_deref_mut() {
            traffic.record_call(descriptor.row_count, point_count);
        }
        let started = PROFILE.then(Instant::now);
        let status = unsafe {
            match handle {
                DirectExecutorHandle::Source { call, context } => {
                    let rows = plan.sources().get(start..end).ok_or_else(|| {
                        RusticolError::integrity(
                            "direct recurrence source row group is out of bounds",
                        )
                    })?;
                    if PROFILE {
                        counters.source_calls += 1;
                        counters.source_rows += u64::from(descriptor.row_count);
                    }
                    call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        rows.as_ptr(),
                        descriptor.row_count,
                        point_count,
                    )
                }
                DirectExecutorHandle::Contribution { call, context } => {
                    let rows = plan.contributions().get(start..end).ok_or_else(|| {
                        RusticolError::integrity(
                            "direct recurrence contribution row group is out of bounds",
                        )
                    })?;
                    if PROFILE {
                        counters.contribution_calls += 1;
                        counters.contribution_rows += u64::from(descriptor.row_count);
                    }
                    call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        rows.as_ptr(),
                        descriptor.row_count,
                        point_count,
                    )
                }
                DirectExecutorHandle::Finalization { call, context } => {
                    let rows = plan.finalizations().get(start..end).ok_or_else(|| {
                        RusticolError::integrity(
                            "direct recurrence finalization row group is out of bounds",
                        )
                    })?;
                    if PROFILE {
                        counters.finalization_calls += 1;
                        counters.finalization_rows += u64::from(descriptor.row_count);
                    }
                    call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        rows.as_ptr(),
                        descriptor.row_count,
                        point_count,
                    )
                }
                DirectExecutorHandle::Closure { call, context } => {
                    let rows = plan.closures().get(start..end).ok_or_else(|| {
                        RusticolError::integrity(
                            "direct recurrence closure row group is out of bounds",
                        )
                    })?;
                    if PROFILE {
                        counters.closure_calls += 1;
                        counters.closure_rows += u64::from(descriptor.row_count);
                    }
                    call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        rows.as_ptr(),
                        descriptor.row_count,
                        point_count,
                    )
                }
            }
        };
        if PROFILE {
            let elapsed = started
                .expect("profiled direct row group has a start time")
                .elapsed();
            match descriptor.role {
                DirectExecutorRole::Source => timings.source += elapsed,
                DirectExecutorRole::Contribution => timings.contribution += elapsed,
                DirectExecutorRole::Finalization => timings.finalization += elapsed,
                DirectExecutorRole::Closure => timings.closure += elapsed,
            }
        }
        check_status(descriptor.role, descriptor.direct_executor_id, status)?;
    }
    Ok(())
}

fn execute_certified_reuse_rows(
    rows: &[DirectContributionRow],
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<()> {
    let stride = workspace.point_stride as usize;
    let point_count = point_count as usize;
    for (index, row) in rows.iter().enumerate() {
        if row.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE == 0
            || row.parent1_component_base_or_sentinel != DIRECT_NONE_U32
            || row.parent1_momentum_form_id_or_sentinel != DIRECT_NONE_U32
        {
            return Err(RusticolError::integrity(format!(
                "certified-reuse row {index} has an invalid relation encoding"
            )));
        }
        let component_count = row.parent0_momentum_form_id as usize;
        if component_count == 0 {
            return Err(RusticolError::integrity(format!(
                "certified-reuse row {index} has zero components"
            )));
        }
        let factor_index = row.exact_factor_id as usize;
        let factor_re = *workspace
            .factors_re
            .get(factor_index)
            .ok_or_else(|| RusticolError::integrity("certified-reuse factor is out of bounds"))?;
        let factor_im = *workspace
            .factors_im
            .get(factor_index)
            .ok_or_else(|| RusticolError::integrity("certified-reuse factor is out of bounds"))?;
        for component in 0..component_count {
            let source_plane = row.parent0_component_base as usize + component;
            let destination_plane = row.destination_component_base as usize + component;
            let source_start = source_plane.checked_mul(stride).ok_or_else(|| {
                RusticolError::integrity("certified-reuse source range overflows usize")
            })?;
            let destination_start = destination_plane.checked_mul(stride).ok_or_else(|| {
                RusticolError::integrity("certified-reuse destination range overflows usize")
            })?;
            let source_end = source_start.checked_add(point_count).ok_or_else(|| {
                RusticolError::integrity("certified-reuse source range overflows usize")
            })?;
            let destination_end = destination_start.checked_add(point_count).ok_or_else(|| {
                RusticolError::integrity("certified-reuse destination range overflows usize")
            })?;
            if source_end > workspace.current_re.len()
                || source_end > workspace.current_im.len()
                || destination_end > workspace.current_re.len()
                || destination_end > workspace.current_im.len()
            {
                return Err(RusticolError::integrity(
                    "certified-reuse source or destination range is out of bounds",
                ));
            }
            for point in 0..point_count {
                let source_index = source_start + point;
                let destination_index = destination_start + point;
                let source_re = *workspace.current_re.get(source_index).ok_or_else(|| {
                    RusticolError::integrity("certified-reuse source real range is out of bounds")
                })?;
                let source_im = *workspace.current_im.get(source_index).ok_or_else(|| {
                    RusticolError::integrity(
                        "certified-reuse source imaginary range is out of bounds",
                    )
                })?;
                *workspace
                    .current_re
                    .get_mut(destination_index)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "certified-reuse destination real range is out of bounds",
                        )
                    })? = source_re * factor_re - source_im * factor_im;
                *workspace
                    .current_im
                    .get_mut(destination_index)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "certified-reuse destination imaginary range is out of bounds",
                        )
                    })? = source_re * factor_im + source_im * factor_re;
            }
        }
    }
    Ok(())
}

pub(crate) fn check_status(
    role: DirectExecutorRole,
    executor_id: u32,
    status: c_int,
) -> RusticolResult<()> {
    let detail = take_direct_executor_error_detail();
    if status == DIRECT_STATUS_OK {
        Ok(())
    } else if let Some(detail) = detail {
        Err(RusticolError::with_kind(
            detail.kind(),
            format!(
                "direct recurrence {role:?} executor {executor_id} failed: {}",
                detail.message()
            ),
        ))
    } else {
        Err(RusticolError::evaluation(format!(
            "direct recurrence {role:?} executor {executor_id} returned status {status}"
        )))
    }
}
