// SPDX-License-Identifier: 0BSD

//! Direct-arena prepared executor ABI for compact recurrence schedules.
//!
//! This module deliberately does not depend on the eager runtime. Direct
//! executors consume fixed-width recurrence rows and persistent split-complex
//! arenas. They never receive a packed evaluator input or output buffer.

use super::RecurrenceStrategy;
use super::SemanticDigest;
use super::direct_plan::{
    DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE, DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION,
    DIRECT_NONE_U32, DirectClosureRow, DirectContributionRow, DirectExecutorRole,
    DirectFinalizationRow, DirectNodeKind, DirectRecurrencePlan, DirectRowGroupDescriptor,
    DirectSourceRow,
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
use std::collections::BTreeMap;
#[cfg(any(test, feature = "on-the-fly-test-support"))]
use std::collections::BTreeSet;
use std::ffi::{c_int, c_void};
use std::mem::size_of;
use std::time::{Duration, Instant};

pub const RECURRENCE_DIRECT_BACKEND_ABI: &str = "rusticol.recurrence-direct-backend.v1";

pub const DIRECT_STATUS_OK: c_int = 0;

thread_local! {
    static DIRECT_EXECUTOR_ERROR_DETAIL: RefCell<Option<RusticolError>> = const { RefCell::new(None) };
    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    static DIRECT_CURRENT_OBSERVATION: RefCell<Option<DirectCurrentObservation>> = const { RefCell::new(None) };
}

/// One test-only semantic-current observation from the authenticated direct
/// runtime.  This is deliberately absent from release builds and public ABIs.
#[cfg(any(test, feature = "on-the-fly-test-support"))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DirectObservedCurrentValue {
    pub descriptor: super::direct_plan::DirectCurrentDescriptor,
    /// Split-complex values in component-major, then point-major order.
    pub values: Vec<(f64, f64)>,
}

/// Test-only capture of the current values produced for one exact direct-plan
/// execution.  The two plan digests and selected sector bind the values to the
/// authenticated schedule and query which produced them.
#[cfg(any(test, feature = "on-the-fly-test-support"))]
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DirectCurrentObservation {
    pub plan_semantic_digest: SemanticDigest,
    pub runtime_layout_digest: SemanticDigest,
    pub selected_sector_id: Option<u32>,
    pub point_count: u32,
    pub currents: BTreeMap<u32, DirectObservedCurrentValue>,
    /// Representative amplitudes captured after closures and before replay
    /// phase/multiplicity scaling, keyed by DirectPlan destination ID.
    pub amplitudes_before_replay: BTreeMap<u32, Vec<(f64, f64)>>,
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
pub(crate) fn begin_direct_current_observation(
    plan: &DirectRecurrencePlan,
    selected_sector_id: Option<u32>,
    point_count: u32,
) -> RusticolResult<()> {
    if point_count == 0 {
        return Err(RusticolError::invalid_argument(
            "direct current observation requires at least one point",
        ));
    }
    DIRECT_CURRENT_OBSERVATION.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err(RusticolError::invalid_argument(
                "direct current observation is already active on this thread",
            ));
        }
        *slot = Some(DirectCurrentObservation {
            plan_semantic_digest: plan.semantic_digest(),
            runtime_layout_digest: plan.runtime_layout_digest(),
            selected_sector_id,
            point_count,
            currents: BTreeMap::new(),
            amplitudes_before_replay: BTreeMap::new(),
        });
        Ok(())
    })
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
pub(crate) fn take_direct_current_observation() -> RusticolResult<DirectCurrentObservation> {
    DIRECT_CURRENT_OBSERVATION.with(|slot| {
        slot.borrow_mut().take().ok_or_else(|| {
            RusticolError::invalid_argument(
                "direct current observation is not active on this thread",
            )
        })
    })
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
pub(super) fn observe_direct_amplitudes_before_replay(
    plan: &DirectRecurrencePlan,
    selected_sector_id: Option<u32>,
    workspace: &DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<()> {
    let active = DIRECT_CURRENT_OBSERVATION.with(|slot| slot.borrow().is_some());
    if !active {
        return Ok(());
    }
    DIRECT_CURRENT_OBSERVATION.with(|slot| {
        let mut slot = slot.borrow_mut();
        let observation = slot.as_mut().ok_or_else(|| {
            RusticolError::integrity("direct current observation disappeared before amplitudes")
        })?;
        if observation.plan_semantic_digest != plan.semantic_digest()
            || observation.runtime_layout_digest != plan.runtime_layout_digest()
            || observation.selected_sector_id != selected_sector_id
            || observation.point_count != point_count
        {
            return Err(RusticolError::integrity(
                "direct amplitude observation does not match this plan execution",
            ));
        }
        let stride = workspace.point_stride as usize;
        let active_points = point_count as usize;
        for destination in plan.amplitude_destinations().iter().filter(|destination| {
            selected_sector_id.is_none_or(|sector| destination.target_sector_id == sector)
        }) {
            let start = destination.id as usize * stride;
            let end = start.checked_add(active_points).ok_or_else(|| {
                RusticolError::integrity("direct amplitude observation range overflows usize")
            })?;
            let real = workspace.amplitude_re.get(start..end).ok_or_else(|| {
                RusticolError::integrity("direct amplitude observation real range is absent")
            })?;
            let imag = workspace.amplitude_im.get(start..end).ok_or_else(|| {
                RusticolError::integrity("direct amplitude observation imaginary range is absent")
            })?;
            observation.amplitudes_before_replay.insert(
                destination.id,
                real.iter().copied().zip(imag.iter().copied()).collect(),
            );
        }
        Ok(())
    })
}

/// Clear the synchronous detail channel before executing one direct plan.
pub(crate) fn clear_direct_executor_error_detail() {
    DIRECT_EXECUTOR_ERROR_DETAIL.with(|detail| *detail.borrow_mut() = None);
}

/// Preserve the first concrete callback failure across the narrow
/// integer-status ABI.
pub(crate) fn record_direct_executor_error_detail(detail: RusticolError) {
    DIRECT_EXECUTOR_ERROR_DETAIL.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(detail);
        }
    });
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

pub(crate) type DirectContributionFanoutExecutor = for<'a> unsafe fn(
    *const c_void,
    DirectArenaView,
    DirectMomentumView,
    DirectParameterView,
    DirectFactorView,
    DirectContributionFanoutBatch<'a>,
) -> RusticolResult<()>;

#[derive(Clone, Copy)]
pub(crate) struct DirectContributionFanoutExecutorHandle {
    pub(crate) call: DirectContributionFanoutExecutor,
    pub(crate) context: *const c_void,
    pub(crate) destination_component_count: u32,
    pub(crate) parent_component_counts: [u32; 2],
    pub(crate) requires_two_momenta: bool,
}

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
    handles: Box<[Option<DirectExecutorHandle>]>,
    contribution_metadata: Box<[Option<DirectContributionExecutionMetadata>]>,
    contribution_fanout: Box<[Option<DirectContributionFanoutExecutorHandle>]>,
    plan_layout_digest: SemanticDigest,
    direct_template_catalog_digest: SemanticDigest,
}

/// Authenticated contribution shape and row-local scalar dependency.
///
/// `exact_factor_is_kernel_input` is fail-closed. It is false only when the
/// prepared template proves that the exact factor is applied exclusively by
/// the destination adapter (currently Rusticol contribution intrinsics).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DirectContributionExecutionMetadata {
    pub(crate) destination_component_count: u32,
    pub(crate) exact_factor_is_kernel_input: bool,
}

impl DirectContributionExecutionMetadata {
    pub(crate) fn new(
        destination_component_count: u32,
        exact_factor_is_kernel_input: bool,
    ) -> RusticolResult<Self> {
        if destination_component_count == 0 {
            return Err(RusticolError::integrity(
                "direct contribution executor has zero destination components",
            ));
        }
        Ok(Self {
            destination_component_count,
            exact_factor_is_kernel_input,
        })
    }
}

impl DirectExecutorCatalog {
    pub fn new(
        plan: &DirectRecurrencePlan,
        direct_template_catalog_digest: SemanticDigest,
        handles: Vec<DirectExecutorHandle>,
    ) -> RusticolResult<Self> {
        let sparse_handles = handles.into_iter().map(Some).collect::<Vec<_>>();
        let metadata = conservative_contribution_metadata(plan, sparse_handles.len())?;
        Self::new_sparse_with_metadata(
            plan,
            direct_template_catalog_digest,
            sparse_handles,
            metadata,
        )
    }

    pub(crate) fn new_sparse(
        plan: &DirectRecurrencePlan,
        direct_template_catalog_digest: SemanticDigest,
        handles: Vec<Option<DirectExecutorHandle>>,
    ) -> RusticolResult<Self> {
        let metadata = conservative_contribution_metadata(plan, handles.len())?;
        Self::new_sparse_with_metadata(plan, direct_template_catalog_digest, handles, metadata)
    }

    pub(crate) fn new_sparse_with_metadata(
        plan: &DirectRecurrencePlan,
        direct_template_catalog_digest: SemanticDigest,
        handles: Vec<Option<DirectExecutorHandle>>,
        contribution_metadata: Vec<Option<DirectContributionExecutionMetadata>>,
    ) -> RusticolResult<Self> {
        let contribution_fanout = vec![None; handles.len()];
        Self::new_sparse_with_metadata_and_fanout(
            plan,
            direct_template_catalog_digest,
            handles,
            contribution_metadata,
            contribution_fanout,
        )
    }

    pub(crate) fn new_sparse_with_metadata_and_fanout(
        plan: &DirectRecurrencePlan,
        direct_template_catalog_digest: SemanticDigest,
        handles: Vec<Option<DirectExecutorHandle>>,
        contribution_metadata: Vec<Option<DirectContributionExecutionMetadata>>,
        contribution_fanout: Vec<Option<DirectContributionFanoutExecutorHandle>>,
    ) -> RusticolResult<Self> {
        if handles.is_empty() {
            return Err(RusticolError::invalid_argument(
                "direct recurrence executor catalog must not be empty",
            ));
        }
        if contribution_metadata.len() != handles.len() {
            return Err(RusticolError::integrity(
                "direct contribution metadata does not cover the executor catalog",
            ));
        }
        if contribution_fanout.len() != handles.len() {
            return Err(RusticolError::integrity(
                "direct contribution fanout capabilities do not cover the executor catalog",
            ));
        }
        for (executor_id, capability) in contribution_fanout.iter().copied().enumerate() {
            let Some(capability) = capability else {
                continue;
            };
            if !matches!(
                handles[executor_id],
                Some(DirectExecutorHandle::Contribution { .. })
            ) {
                return Err(RusticolError::integrity(format!(
                    "direct fused fanout capability {executor_id} is not a loaded contribution executor"
                )));
            }
            let metadata = contribution_metadata[executor_id].ok_or_else(|| {
                RusticolError::integrity(format!(
                    "direct fused fanout capability {executor_id} has no contribution metadata"
                ))
            })?;
            if metadata.exact_factor_is_kernel_input
                || metadata.destination_component_count != capability.destination_component_count
            {
                return Err(RusticolError::integrity(format!(
                    "direct fused fanout capability {executor_id} is incompatible with its contribution metadata"
                )));
            }
        }
        if direct_template_catalog_digest != plan.direct_template_catalog_digest() {
            return Err(RusticolError::integrity(format!(
                "loaded direct-template catalog digest {direct_template_catalog_digest} does not match plan {}",
                plan.direct_template_catalog_digest()
            )));
        }
        let catalog = Self {
            handles: handles.into_boxed_slice(),
            contribution_metadata: contribution_metadata.into_boxed_slice(),
            contribution_fanout: contribution_fanout.into_boxed_slice(),
            plan_layout_digest: plan.runtime_layout_digest(),
            direct_template_catalog_digest,
        };
        catalog.validate_for_plan(plan)?;
        Ok(catalog)
    }

    fn contribution_metadata(
        &self,
        executor_id: u32,
    ) -> RusticolResult<DirectContributionExecutionMetadata> {
        self.contribution_metadata
            .get(executor_id as usize)
            .and_then(|metadata| *metadata)
            .ok_or_else(|| {
                RusticolError::integrity(format!(
                    "direct contribution executor {executor_id} has no authenticated execution metadata"
                ))
            })
    }

    fn contribution_fanout(
        &self,
        executor_id: u32,
    ) -> Option<DirectContributionFanoutExecutorHandle> {
        self.contribution_fanout
            .get(executor_id as usize)
            .and_then(|capability| *capability)
    }

    fn require(
        &self,
        executor_id: u32,
        role: DirectExecutorRole,
    ) -> RusticolResult<DirectExecutorHandle> {
        let handle = self
            .handles
            .get(executor_id as usize)
            .and_then(|handle| *handle)
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
        if self.contribution_metadata.len() != self.handles.len() {
            return Err(RusticolError::integrity(
                "direct contribution metadata has the wrong executor domain",
            ));
        }
        if self.contribution_fanout.len() != self.handles.len() {
            return Err(RusticolError::integrity(
                "direct contribution fanout capabilities have the wrong executor domain",
            ));
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
            if descriptor.role == DirectExecutorRole::Contribution {
                self.contribution_metadata(descriptor.direct_executor_id)?;
            }
        }
        Ok(())
    }
}

fn conservative_contribution_metadata(
    plan: &DirectRecurrencePlan,
    executor_count: usize,
) -> RusticolResult<Vec<Option<DirectContributionExecutionMetadata>>> {
    let current_components = plan
        .currents()
        .iter()
        .filter(|current| current.node_kind == DirectNodeKind::Current)
        .map(|current| {
            (
                (current.stage, current.component_base),
                u32::from(current.component_count),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut metadata = vec![None; executor_count];
    for descriptor in plan.row_groups().iter().filter(|descriptor| {
        descriptor.role == DirectExecutorRole::Contribution
            && descriptor.direct_executor_id != DIRECT_NONE_U32
    }) {
        let start = usize::try_from(descriptor.row_start).map_err(|_| {
            RusticolError::integrity("direct contribution row-group start exceeds usize")
        })?;
        let end = start
            .checked_add(descriptor.row_count as usize)
            .ok_or_else(|| {
                RusticolError::integrity("direct contribution row-group range overflows usize")
            })?;
        let rows = plan.contributions().get(start..end).ok_or_else(|| {
            RusticolError::integrity("direct contribution row-group range is out of bounds")
        })?;
        for row in rows {
            let component_count = current_components
                .get(&(descriptor.stage, row.destination_component_base))
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity("direct contribution destination current is absent")
                })?;
            let candidate = DirectContributionExecutionMetadata::new(component_count, true)?;
            let slot = metadata
                .get_mut(descriptor.direct_executor_id as usize)
                .ok_or_else(|| {
                    RusticolError::integrity("direct contribution executor is outside its catalog")
                })?;
            if slot.is_some_and(|existing| existing != candidate) {
                return Err(RusticolError::integrity(format!(
                    "direct contribution executor {} has inconsistent destination shapes",
                    descriptor.direct_executor_id
                )));
            }
            *slot = Some(candidate);
        }
    }
    Ok(metadata)
}

const DIRECT_CONTRIBUTION_FANOUT_BATCH_CAPACITY: usize = 64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DirectContributionFanoutRun {
    pub(crate) row_start: u64,
    pub(crate) row_count: u32,
    pub(crate) scratch_component_base: u32,
    pub(crate) bundle_start: u64,
    pub(crate) bundle_count: u32,
}

/// Maximal consecutive fanout outputs which share their effective scale.
///
/// `DIRECT_NONE_U32` denotes unity because factor-consuming executors already
/// applied the authenticated row factor while producing the representative.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DirectContributionFanoutBundle {
    pub(crate) row_start: u64,
    pub(crate) row_count: u32,
    pub(crate) effective_factor_id_or_sentinel: u32,
}

#[derive(Clone, Copy)]
pub(crate) struct DirectContributionFanoutBatch<'a> {
    pub(crate) rows: &'a [DirectContributionRow],
    pub(crate) runs: &'a [DirectContributionFanoutRun],
    pub(crate) bundles: &'a [DirectContributionFanoutBundle],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DirectContributionFanoutStep {
    Direct {
        row_start: u64,
        row_count: u32,
    },
    Fanout {
        row_start: u64,
        row_count: u32,
        representative_start: u64,
        representative_count: u32,
        run_start: u64,
        run_count: u32,
        destination_component_count: u32,
    },
}

impl DirectContributionFanoutStep {
    const fn row_start(self) -> u64 {
        match self {
            Self::Direct { row_start, .. } | Self::Fanout { row_start, .. } => row_start,
        }
    }

    const fn row_count(self) -> u32 {
        match self {
            Self::Direct { row_count, .. } | Self::Fanout { row_count, .. } => row_count,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DirectContributionFanoutGroup {
    stage: u16,
    direct_executor_id: u32,
    row_start: u64,
    row_count: u32,
    step_start: u64,
    step_count: u32,
}

/// Immutable, plan-local execution rewrite for contribution kinematics.
///
/// Rows remain the authoritative schedule. This cold-path program merely
/// evaluates each contiguous authenticated input class once into a bounded
/// private current-arena window, then applies the original rows' destination
/// and exact-factor policies in their deterministic lowered order.
pub(crate) struct DirectContributionFanoutProgram {
    plan_layout_digest: SemanticDigest,
    groups: Box<[DirectContributionFanoutGroup]>,
    steps: Box<[DirectContributionFanoutStep]>,
    representatives: Box<[DirectContributionRow]>,
    runs: Box<[DirectContributionFanoutRun]>,
    bundles: Box<[DirectContributionFanoutBundle]>,
    scratch_component_base: u32,
    scratch_component_count: u32,
    unit_factor_id: u32,
    needs_unit_factor: bool,
}

impl DirectContributionFanoutProgram {
    pub(crate) fn build(
        plan: &DirectRecurrencePlan,
        executors: &DirectExecutorCatalog,
    ) -> RusticolResult<Self> {
        if executors.plan_layout_digest != plan.runtime_layout_digest() {
            return Err(RusticolError::integrity(
                "direct fanout executor catalog belongs to a different plan",
            ));
        }
        let scratch_component_base = plan.current_arena_components();
        let unit_factor_id = u32::try_from(plan.exact_factors().len())
            .map_err(|_| RusticolError::integrity("direct exact-factor catalog exceeds u32"))?;
        let mut groups = Vec::new();
        let mut steps = Vec::new();
        let mut representatives = Vec::new();
        let mut runs = Vec::new();
        let mut bundles = Vec::new();
        let mut scratch_component_count = 0_u32;
        let mut needs_unit_factor = false;
        let destination_component_counts = plan
            .currents()
            .iter()
            .filter(|current| current.node_kind == DirectNodeKind::Current)
            .map(|current| {
                (
                    (current.stage, current.component_base),
                    u32::from(current.component_count),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let arena_component_count = plan.current_arena_components();
        let momentum_form_count = u32::try_from(plan.momentum_forms().len())
            .map_err(|_| RusticolError::integrity("direct momentum-form count exceeds u32"))?;

        for descriptor in plan.row_groups().iter().filter(|descriptor| {
            descriptor.role == DirectExecutorRole::Contribution
                && descriptor.direct_executor_id != DIRECT_NONE_U32
        }) {
            let metadata = executors.contribution_metadata(descriptor.direct_executor_id)?;
            let fused = executors.contribution_fanout(descriptor.direct_executor_id);
            let group_step_start = u64::try_from(steps.len())
                .map_err(|_| RusticolError::integrity("direct fanout step count exceeds u64"))?;
            let start = usize::try_from(descriptor.row_start).map_err(|_| {
                RusticolError::integrity("direct fanout row-group start exceeds usize")
            })?;
            let end = start
                .checked_add(descriptor.row_count as usize)
                .ok_or_else(|| {
                    RusticolError::integrity("direct fanout row-group range overflows usize")
                })?;
            let descriptor_rows = plan.contributions().get(start..end).ok_or_else(|| {
                RusticolError::integrity("direct fanout row-group range is out of bounds")
            })?;
            for row in descriptor_rows {
                let authenticated_component_count = destination_component_counts
                    .get(&(descriptor.stage, row.destination_component_base))
                    .copied()
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "direct fanout contribution destination current is absent",
                        )
                    })?;
                if metadata.destination_component_count != authenticated_component_count {
                    return Err(RusticolError::integrity(format!(
                        "direct contribution executor {} declares {} destination components, expected {authenticated_component_count}",
                        descriptor.direct_executor_id, metadata.destination_component_count
                    )));
                }
                if let Some(fused) = fused {
                    validate_fused_fanout_row(
                        *row,
                        fused.parent_component_counts,
                        fused.requires_two_momenta,
                        arena_component_count,
                        momentum_form_count,
                    )?;
                }
            }
            let mut selector_offset = 0_usize;
            while selector_offset < descriptor_rows.len() {
                let selector_domain_id = descriptor_rows[selector_offset].selector_domain_id;
                let selector_start = selector_offset;
                while selector_offset < descriptor_rows.len()
                    && descriptor_rows[selector_offset].selector_domain_id == selector_domain_id
                {
                    selector_offset += 1;
                }
                let selector_end = selector_offset;
                let segment_step_start = steps.len();
                let mut pending_runs = Vec::<(usize, usize)>::new();
                let mut offset = selector_start;
                while offset < selector_end {
                    let run_start = offset;
                    offset += 1;
                    while offset < selector_end
                        && contribution_fanout_inputs_equal(
                            descriptor_rows[run_start],
                            descriptor_rows[offset],
                            metadata.exact_factor_is_kernel_input,
                        )
                    {
                        offset += 1;
                    }
                    let run_count = offset - run_start;
                    if run_count == 1 {
                        flush_pending_fanout_batch(
                            plan,
                            descriptor,
                            metadata,
                            scratch_component_base,
                            unit_factor_id,
                            &mut pending_runs,
                            &mut steps,
                            &mut representatives,
                            &mut runs,
                            &mut bundles,
                            &mut scratch_component_count,
                            &mut needs_unit_factor,
                        )?;
                        push_direct_fanout_step(
                            &mut steps,
                            segment_step_start,
                            descriptor.row_start + run_start as u64,
                            1,
                        )?;
                    } else {
                        pending_runs.push((run_start, run_count));
                        if pending_runs.len() == DIRECT_CONTRIBUTION_FANOUT_BATCH_CAPACITY {
                            flush_pending_fanout_batch(
                                plan,
                                descriptor,
                                metadata,
                                scratch_component_base,
                                unit_factor_id,
                                &mut pending_runs,
                                &mut steps,
                                &mut representatives,
                                &mut runs,
                                &mut bundles,
                                &mut scratch_component_count,
                                &mut needs_unit_factor,
                            )?;
                        }
                    }
                }
                flush_pending_fanout_batch(
                    plan,
                    descriptor,
                    metadata,
                    scratch_component_base,
                    unit_factor_id,
                    &mut pending_runs,
                    &mut steps,
                    &mut representatives,
                    &mut runs,
                    &mut bundles,
                    &mut scratch_component_count,
                    &mut needs_unit_factor,
                )?;
            }
            groups.push(DirectContributionFanoutGroup {
                stage: descriptor.stage,
                direct_executor_id: descriptor.direct_executor_id,
                row_start: descriptor.row_start,
                row_count: descriptor.row_count,
                step_start: group_step_start,
                step_count: u32::try_from(steps.len() as u64 - group_step_start).map_err(|_| {
                    RusticolError::integrity("direct fanout group step count exceeds u32")
                })?,
            });
        }
        Ok(Self {
            plan_layout_digest: plan.runtime_layout_digest(),
            groups: groups.into_boxed_slice(),
            steps: steps.into_boxed_slice(),
            representatives: representatives.into_boxed_slice(),
            runs: runs.into_boxed_slice(),
            bundles: bundles.into_boxed_slice(),
            scratch_component_base,
            scratch_component_count,
            unit_factor_id,
            needs_unit_factor,
        })
    }

    pub(crate) const fn scratch_component_count(&self) -> u32 {
        self.scratch_component_count
    }

    pub(crate) const fn needs_unit_factor(&self) -> bool {
        self.needs_unit_factor
    }

    #[cfg(test)]
    pub(crate) fn row_counts(&self) -> (u64, u64) {
        let logical = self
            .groups
            .iter()
            .map(|group| u64::from(group.row_count))
            .sum();
        let evaluated = self
            .steps
            .iter()
            .map(|step| match step {
                DirectContributionFanoutStep::Direct { row_count, .. } => u64::from(*row_count),
                DirectContributionFanoutStep::Fanout {
                    representative_count,
                    ..
                } => u64::from(*representative_count),
            })
            .sum();
        (logical, evaluated)
    }

    #[cfg(test)]
    pub(crate) fn scatter_bundle_counts(&self) -> (u64, u64) {
        (
            self.runs.iter().map(|run| u64::from(run.row_count)).sum(),
            self.bundles.len() as u64,
        )
    }

    fn group_steps(
        &self,
        descriptor: &DirectRowGroupDescriptor,
    ) -> RusticolResult<&[DirectContributionFanoutStep]> {
        if descriptor.role != DirectExecutorRole::Contribution
            || descriptor.direct_executor_id == DIRECT_NONE_U32
        {
            return Err(RusticolError::integrity(
                "direct fanout requested for a non-callable contribution group",
            ));
        }
        let requested_end = descriptor
            .row_start
            .checked_add(u64::from(descriptor.row_count))
            .ok_or_else(|| RusticolError::integrity("direct fanout row range overflows u64"))?;
        // Groups retain the authenticated row-group order, so locate the one
        // containing a selected subrange without a warmed linear catalog scan.
        let group_index = self
            .groups
            .partition_point(|group| group.row_start <= descriptor.row_start)
            .checked_sub(1)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "direct contribution row group is absent from its fanout program",
                )
            })?;
        let group = self
            .groups
            .get(group_index)
            .filter(|group| {
                let group_end = group.row_start + u64::from(group.row_count);
                group.stage == descriptor.stage
                    && group.direct_executor_id == descriptor.direct_executor_id
                    && descriptor.row_start >= group.row_start
                    && requested_end <= group_end
            })
            .ok_or_else(|| {
                RusticolError::integrity(
                    "direct contribution row group is absent from its fanout program",
                )
            })?;
        let step_start = usize::try_from(group.step_start)
            .map_err(|_| RusticolError::integrity("direct fanout step start exceeds usize"))?;
        let step_end = step_start
            .checked_add(group.step_count as usize)
            .ok_or_else(|| RusticolError::integrity("direct fanout step range overflows usize"))?;
        let group_steps = self.steps.get(step_start..step_end).ok_or_else(|| {
            RusticolError::integrity("direct fanout group step range is out of bounds")
        })?;
        let first = group_steps
            .iter()
            .position(|step| step.row_start() == descriptor.row_start)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "selected direct fanout range does not begin on a step boundary",
                )
            })?;
        let mut stop = first;
        let mut covered = descriptor.row_start;
        while stop < group_steps.len() && covered < requested_end {
            let step = group_steps[stop];
            if step.row_start() != covered {
                return Err(RusticolError::integrity(
                    "direct fanout steps do not continuously cover their row group",
                ));
            }
            covered = covered
                .checked_add(u64::from(step.row_count()))
                .ok_or_else(|| {
                    RusticolError::integrity("direct fanout step coverage overflows u64")
                })?;
            stop += 1;
        }
        if covered != requested_end {
            return Err(RusticolError::integrity(
                "selected direct fanout range does not end on a step boundary",
            ));
        }
        Ok(&group_steps[first..stop])
    }
}

fn validate_fused_fanout_row(
    row: DirectContributionRow,
    parent_component_counts: [u32; 2],
    requires_two_momenta: bool,
    arena_component_count: u32,
    momentum_form_count: u32,
) -> RusticolResult<()> {
    for (base, required) in [
        (row.parent0_component_base, parent_component_counts[0]),
        (
            row.parent1_component_base_or_sentinel,
            parent_component_counts[1],
        ),
    ] {
        if base == DIRECT_NONE_U32
            || base
                .checked_add(required)
                .is_none_or(|end| end > arena_component_count)
        {
            return Err(RusticolError::integrity(
                "direct fused fanout parent component span is incompatible with its intrinsic",
            ));
        }
    }
    if requires_two_momenta
        && (row.parent0_momentum_form_id >= momentum_form_count
            || row.parent1_momentum_form_id_or_sentinel >= momentum_form_count)
    {
        return Err(RusticolError::integrity(
            "direct fused fanout momentum forms are incompatible with its intrinsic",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod fused_fanout_validation_tests {
    use super::*;

    #[test]
    fn rejects_short_parent_spans_and_missing_momentum_forms() {
        let row = DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: 4,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: 1,
            destination_component_base: 8,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        assert!(validate_fused_fanout_row(row, [9, 4], false, 8, 2).is_err());
        assert!(validate_fused_fanout_row(row, [4, 4], true, 8, 1).is_err());
        validate_fused_fanout_row(row, [4, 4], true, 8, 2).unwrap();
    }
}

fn contribution_fanout_inputs_equal(
    left: DirectContributionRow,
    right: DirectContributionRow,
    exact_factor_is_kernel_input: bool,
) -> bool {
    left.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE == 0
        && right.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE == 0
        && left.selector_domain_id == right.selector_domain_id
        && left.parent0_component_base == right.parent0_component_base
        && left.parent1_component_base_or_sentinel == right.parent1_component_base_or_sentinel
        && left.parent0_momentum_form_id == right.parent0_momentum_form_id
        && left.parent1_momentum_form_id_or_sentinel == right.parent1_momentum_form_id_or_sentinel
        && (!exact_factor_is_kernel_input || left.exact_factor_id == right.exact_factor_id)
}

fn push_direct_fanout_step(
    steps: &mut Vec<DirectContributionFanoutStep>,
    segment_step_start: usize,
    row_start: u64,
    row_count: u32,
) -> RusticolResult<()> {
    if steps.len() > segment_step_start
        && let Some(DirectContributionFanoutStep::Direct {
            row_start: previous_start,
            row_count: previous_count,
        }) = steps.last_mut()
        && previous_start
            .checked_add(u64::from(*previous_count))
            .is_some_and(|end| end == row_start)
    {
        *previous_count = previous_count.checked_add(row_count).ok_or_else(|| {
            RusticolError::integrity("direct fanout direct-step row count exceeds u32")
        })?;
        return Ok(());
    }
    steps.push(DirectContributionFanoutStep::Direct {
        row_start,
        row_count,
    });
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn flush_pending_fanout_batch(
    plan: &DirectRecurrencePlan,
    descriptor: &DirectRowGroupDescriptor,
    metadata: DirectContributionExecutionMetadata,
    scratch_component_base: u32,
    unit_factor_id: u32,
    pending_runs: &mut Vec<(usize, usize)>,
    steps: &mut Vec<DirectContributionFanoutStep>,
    representatives: &mut Vec<DirectContributionRow>,
    runs: &mut Vec<DirectContributionFanoutRun>,
    bundles: &mut Vec<DirectContributionFanoutBundle>,
    scratch_component_count: &mut u32,
    needs_unit_factor: &mut bool,
) -> RusticolResult<()> {
    if pending_runs.is_empty() {
        return Ok(());
    }
    let representative_start = u64::try_from(representatives.len())
        .map_err(|_| RusticolError::integrity("direct fanout representative count exceeds u64"))?;
    let run_table_start = u64::try_from(runs.len())
        .map_err(|_| RusticolError::integrity("direct fanout run count exceeds u64"))?;
    let first_row_start = descriptor
        .row_start
        .checked_add(pending_runs[0].0 as u64)
        .ok_or_else(|| RusticolError::integrity("direct fanout row start overflows u64"))?;
    let mut logical_row_count = 0_u32;
    for (slot, &(relative_start, run_count)) in pending_runs.iter().enumerate() {
        let row_index = usize::try_from(descriptor.row_start)
            .ok()
            .and_then(|start| start.checked_add(relative_start))
            .ok_or_else(|| RusticolError::integrity("direct fanout row index overflows usize"))?;
        let mut representative = *plan.contributions().get(row_index).ok_or_else(|| {
            RusticolError::integrity("direct fanout representative row is out of bounds")
        })?;
        let slot_base = u32::try_from(slot)
            .ok()
            .and_then(|slot| slot.checked_mul(metadata.destination_component_count))
            .and_then(|offset| scratch_component_base.checked_add(offset))
            .ok_or_else(|| {
                RusticolError::integrity("direct fanout scratch component base exceeds u32")
            })?;
        representative.destination_component_base = slot_base;
        representative.flags = DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION;
        if !metadata.exact_factor_is_kernel_input {
            representative.exact_factor_id = unit_factor_id;
            *needs_unit_factor = true;
        }
        representatives.push(representative);
        let run_row_start = descriptor
            .row_start
            .checked_add(relative_start as u64)
            .ok_or_else(|| RusticolError::integrity("direct fanout run start overflows u64"))?;
        let run_row_count = u32::try_from(run_count)
            .map_err(|_| RusticolError::integrity("direct fanout run exceeds u32"))?;
        let bundle_start = u64::try_from(bundles.len())
            .map_err(|_| RusticolError::integrity("direct fanout bundle count exceeds u64"))?;
        let run_end = row_index
            .checked_add(run_count)
            .ok_or_else(|| RusticolError::integrity("direct fanout run range exceeds usize"))?;
        let run_rows = plan
            .contributions()
            .get(row_index..run_end)
            .ok_or_else(|| RusticolError::integrity("direct fanout run range is out of bounds"))?;
        if metadata.exact_factor_is_kernel_input {
            bundles.push(DirectContributionFanoutBundle {
                row_start: run_row_start,
                row_count: run_row_count,
                effective_factor_id_or_sentinel: DIRECT_NONE_U32,
            });
        } else {
            let mut bundle_offset = 0_usize;
            while bundle_offset < run_rows.len() {
                let factor_id = run_rows[bundle_offset].exact_factor_id;
                let first = bundle_offset;
                bundle_offset += 1;
                while bundle_offset < run_rows.len()
                    && run_rows[bundle_offset].exact_factor_id == factor_id
                {
                    bundle_offset += 1;
                }
                bundles.push(DirectContributionFanoutBundle {
                    row_start: run_row_start + first as u64,
                    row_count: u32::try_from(bundle_offset - first).map_err(|_| {
                        RusticolError::integrity("direct fanout factor bundle exceeds u32")
                    })?,
                    effective_factor_id_or_sentinel: factor_id,
                });
            }
        }
        let bundle_count = u32::try_from(bundles.len() as u64 - bundle_start)
            .map_err(|_| RusticolError::integrity("direct fanout run bundle count exceeds u32"))?;
        runs.push(DirectContributionFanoutRun {
            row_start: run_row_start,
            row_count: run_row_count,
            scratch_component_base: slot_base,
            bundle_start,
            bundle_count,
        });
        logical_row_count = logical_row_count
            .checked_add(u32::try_from(run_count).map_err(|_| {
                RusticolError::integrity("direct fanout logical run count exceeds u32")
            })?)
            .ok_or_else(|| RusticolError::integrity("direct fanout batch row count exceeds u32"))?;
    }
    let representative_count = u32::try_from(pending_runs.len())
        .map_err(|_| RusticolError::integrity("direct fanout batch exceeds u32"))?;
    let required_scratch = representative_count
        .checked_mul(metadata.destination_component_count)
        .ok_or_else(|| RusticolError::integrity("direct fanout scratch size exceeds u32"))?;
    *scratch_component_count = (*scratch_component_count).max(required_scratch);
    steps.push(DirectContributionFanoutStep::Fanout {
        row_start: first_row_start,
        row_count: logical_row_count,
        representative_start,
        representative_count,
        run_start: run_table_start,
        run_count: representative_count,
        destination_component_count: metadata.destination_component_count,
    });
    pending_runs.clear();
    Ok(())
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
        None,
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
        None,
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
        None,
        workspace,
        point_count,
        counters,
        timings,
        Some(traffic),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_direct_plan_profiled_with_fanout_and_traffic(
    plan: &DirectRecurrencePlan,
    executors: &DirectExecutorCatalog,
    fanout: &DirectContributionFanoutProgram,
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
        Some(fanout),
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
        None,
        workspace,
        point_count,
        &mut unused,
        &mut unused_timings,
        None,
    )
}

pub(crate) fn execute_direct_plan_unprofiled_with_fanout(
    plan: &DirectRecurrencePlan,
    executors: &DirectExecutorCatalog,
    fanout: &DirectContributionFanoutProgram,
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
        Some(fanout),
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
        None,
        workspace,
        point_count,
        counters,
        timings,
        Some(traffic),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_direct_plan_selected_profiled_with_fanout_and_traffic(
    plan: &DirectRecurrencePlan,
    row_groups: &[DirectRowGroupDescriptor],
    selected_sector_id: u32,
    executors: &DirectExecutorCatalog,
    fanout: &DirectContributionFanoutProgram,
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
        Some(fanout),
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
        None,
        workspace,
        point_count,
        &mut unused,
        &mut unused_timings,
        None,
    )
}

pub(crate) fn execute_direct_plan_selected_unprofiled_with_fanout(
    plan: &DirectRecurrencePlan,
    row_groups: &[DirectRowGroupDescriptor],
    selected_sector_id: u32,
    executors: &DirectExecutorCatalog,
    fanout: &DirectContributionFanoutProgram,
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
        Some(fanout),
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
    fanout: Option<&DirectContributionFanoutProgram>,
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
    if fanout.is_some_and(|fanout| fanout.plan_layout_digest != plan.runtime_layout_digest()) {
        return Err(RusticolError::integrity(
            "direct contribution fanout program belongs to a different plan",
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
        // A full topology/contracted schedule executes every authenticated
        // initializer, so its writes replace the stage clear. All-flow-union
        // sources are prepared outside this static row schedule and may leave
        // additive destinations relying on the clear even without a selected
        // sector. Selected execution likewise visits only a logical subset.
        let requires_contribution_stage_clear =
            selected_sector_id.is_some() || plan.strategy() == RecurrenceStrategy::AllFlowUnion;
        if requires_contribution_stage_clear
            && descriptor.role == DirectExecutorRole::Contribution
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
            #[cfg(any(test, feature = "on-the-fly-test-support"))]
            observe_direct_current_rows(
                plan,
                descriptor,
                start,
                end,
                selected_sector_id,
                workspace,
                point_count,
            )?;
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
        if descriptor.role == DirectExecutorRole::Contribution
            && let Some(fanout) = fanout
        {
            let handle = executors.require(
                descriptor.direct_executor_id,
                DirectExecutorRole::Contribution,
            )?;
            let DirectExecutorHandle::Contribution { call, context } = handle else {
                return Err(RusticolError::integrity(
                    "direct fanout executor does not have the contribution role",
                ));
            };
            if PROFILE && let Some(traffic) = traffic.as_deref_mut() {
                traffic.record_call(descriptor.row_count, point_count);
            }
            let started = PROFILE.then(Instant::now);
            let scatter_bytes = execute_contribution_group_with_fanout(
                plan,
                descriptor,
                fanout,
                call,
                context,
                executors.contribution_fanout(descriptor.direct_executor_id),
                workspace,
                point_count,
            )?;
            if PROFILE {
                counters.contribution_calls += 1;
                counters.contribution_rows += u64::from(descriptor.row_count);
                counters.scatter_bytes = counters.scatter_bytes.saturating_add(scatter_bytes);
                timings.contribution += started
                    .expect("profiled direct fanout group has a start time")
                    .elapsed();
            }
            #[cfg(any(test, feature = "on-the-fly-test-support"))]
            observe_direct_current_rows(
                plan,
                descriptor,
                start,
                end,
                selected_sector_id,
                workspace,
                point_count,
            )?;
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
        #[cfg(any(test, feature = "on-the-fly-test-support"))]
        observe_direct_current_rows(
            plan,
            descriptor,
            start,
            end,
            selected_sector_id,
            workspace,
            point_count,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_contribution_group_with_fanout(
    plan: &DirectRecurrencePlan,
    descriptor: &DirectRowGroupDescriptor,
    fanout: &DirectContributionFanoutProgram,
    call: DirectContributionExecutor,
    context: *const c_void,
    fused: Option<DirectContributionFanoutExecutorHandle>,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<u64> {
    let current_plane_count = u32::try_from(
        workspace.current_re.len() / workspace.point_stride as usize,
    )
    .map_err(|_| RusticolError::integrity("direct fanout current plane count exceeds u32"))?;
    let scratch_end = fanout
        .scratch_component_base
        .checked_add(fanout.scratch_component_count)
        .ok_or_else(|| RusticolError::integrity("direct fanout scratch range overflows u32"))?;
    if fanout.scratch_component_base != plan.current_arena_components()
        || scratch_end > current_plane_count
    {
        return Err(RusticolError::integrity(
            "direct fanout scratch range is outside its private current arena",
        ));
    }
    if fanout.needs_unit_factor {
        let factor = fanout.unit_factor_id as usize;
        if workspace
            .factors_re
            .get(factor)
            .map(|value| value.to_bits())
            != Some(1.0_f64.to_bits())
            || workspace
                .factors_im
                .get(factor)
                .map(|value| value.to_bits())
                != Some(0.0_f64.to_bits())
        {
            return Err(RusticolError::integrity(
                "direct fanout private exact factor is not complex unity",
            ));
        }
    }
    let steps = fanout.group_steps(descriptor)?;
    let (arena, momenta, parameters, factors) = workspace.raw_views()?;
    let mut scatter_bytes = 0_u64;
    for step in steps {
        match *step {
            DirectContributionFanoutStep::Direct {
                row_start,
                row_count,
            } => {
                let start = usize::try_from(row_start).map_err(|_| {
                    RusticolError::integrity("direct fanout direct-row start exceeds usize")
                })?;
                let end = start.checked_add(row_count as usize).ok_or_else(|| {
                    RusticolError::integrity("direct fanout direct-row range overflows usize")
                })?;
                let rows = plan.contributions().get(start..end).ok_or_else(|| {
                    RusticolError::integrity("direct fanout direct-row range is out of bounds")
                })?;
                let status = unsafe {
                    call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        rows.as_ptr(),
                        row_count,
                        point_count,
                    )
                };
                check_status(
                    DirectExecutorRole::Contribution,
                    descriptor.direct_executor_id,
                    status,
                )?;
            }
            DirectContributionFanoutStep::Fanout {
                row_count,
                representative_start,
                representative_count,
                run_start,
                run_count,
                destination_component_count,
                ..
            } => {
                if point_count == 1
                    && let Some(fused) = fused
                {
                    let start = usize::try_from(run_start).map_err(|_| {
                        RusticolError::integrity("direct fanout run start exceeds usize")
                    })?;
                    let end = start.checked_add(run_count as usize).ok_or_else(|| {
                        RusticolError::integrity("direct fanout run range overflows usize")
                    })?;
                    let runs = fanout.runs.get(start..end).ok_or_else(|| {
                        RusticolError::integrity("direct fanout run range is out of bounds")
                    })?;
                    debug_assert_eq!(
                        fused.destination_component_count,
                        destination_component_count
                    );
                    unsafe {
                        (fused.call)(
                            fused.context,
                            arena,
                            momenta,
                            parameters,
                            factors,
                            DirectContributionFanoutBatch {
                                rows: plan.contributions(),
                                runs,
                                bundles: &fanout.bundles,
                            },
                        )
                    }?;
                    continue;
                }
                let representative_start = usize::try_from(representative_start).map_err(|_| {
                    RusticolError::integrity("direct fanout representative start exceeds usize")
                })?;
                let representative_end = representative_start
                    .checked_add(representative_count as usize)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "direct fanout representative range overflows usize",
                        )
                    })?;
                let representatives = fanout
                    .representatives
                    .get(representative_start..representative_end)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "direct fanout representative range is out of bounds",
                        )
                    })?;
                let status = unsafe {
                    call(
                        context,
                        arena,
                        momenta,
                        parameters,
                        factors,
                        representatives.as_ptr(),
                        representative_count,
                        point_count,
                    )
                };
                check_status(
                    DirectExecutorRole::Contribution,
                    descriptor.direct_executor_id,
                    status,
                )?;
                let run_start = usize::try_from(run_start).map_err(|_| {
                    RusticolError::integrity("direct fanout run start exceeds usize")
                })?;
                let run_end = run_start.checked_add(run_count as usize).ok_or_else(|| {
                    RusticolError::integrity("direct fanout run range overflows usize")
                })?;
                let batch_runs = fanout.runs.get(run_start..run_end).ok_or_else(|| {
                    RusticolError::integrity("direct fanout run range is out of bounds")
                })?;
                debug_assert_eq!(run_count, representative_count);
                debug_assert_eq!(
                    batch_runs
                        .iter()
                        .map(|run| u64::from(run.row_count))
                        .sum::<u64>(),
                    u64::from(row_count)
                );
                for run in batch_runs {
                    scatter_contribution_fanout_run(
                        plan,
                        fanout,
                        *run,
                        destination_component_count,
                        workspace,
                        point_count,
                    )?;
                }
                scatter_bytes = scatter_bytes.saturating_add(
                    u64::from(row_count)
                        .saturating_mul(u64::from(destination_component_count))
                        .saturating_mul(u64::from(point_count))
                        .saturating_mul(4 * size_of::<f64>() as u64),
                );
            }
        }
    }
    Ok(scatter_bytes)
}

#[derive(Clone, Copy)]
pub(crate) enum DirectContributionFanoutScale {
    Unity,
    NegativeUnity,
    Real(f64),
    Imaginary(f64),
    Complex(f64, f64),
}

impl DirectContributionFanoutScale {
    #[inline(always)]
    pub(crate) fn new(real: f64, imaginary: f64) -> Self {
        if imaginary == 0.0 {
            if real == 1.0 {
                Self::Unity
            } else if real == -1.0 {
                Self::NegativeUnity
            } else {
                Self::Real(real)
            }
        } else if real == 0.0 {
            Self::Imaginary(imaginary)
        } else {
            Self::Complex(real, imaginary)
        }
    }

    #[inline(always)]
    pub(crate) fn apply(self, real: f64, imaginary: f64) -> (f64, f64) {
        match self {
            Self::Unity => (real, imaginary),
            Self::NegativeUnity => (-real, -imaginary),
            Self::Real(scale) => (real * scale, imaginary * scale),
            Self::Imaginary(scale) => (-imaginary * scale, real * scale),
            Self::Complex(scale_re, scale_im) => (
                real * scale_re - imaginary * scale_im,
                real * scale_im + imaginary * scale_re,
            ),
        }
    }
}

fn scatter_contribution_fanout_run(
    plan: &DirectRecurrencePlan,
    fanout: &DirectContributionFanoutProgram,
    run: DirectContributionFanoutRun,
    destination_component_count: u32,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<()> {
    let stride = workspace.point_stride as usize;
    let active_points = point_count as usize;
    let source_end = run
        .scratch_component_base
        .checked_add(destination_component_count)
        .ok_or_else(|| RusticolError::integrity("direct fanout source range overflows u32"))?;
    let current_plane_count = u32::try_from(workspace.current_re.len() / stride)
        .map_err(|_| RusticolError::integrity("direct fanout current plane count exceeds u32"))?;
    if run.scratch_component_base < fanout.scratch_component_base
        || source_end > current_plane_count
    {
        return Err(RusticolError::integrity(
            "direct fanout source range is outside its private current arena",
        ));
    }
    let bundle_start = usize::try_from(run.bundle_start)
        .map_err(|_| RusticolError::integrity("direct fanout bundle start exceeds usize"))?;
    let bundle_end = bundle_start
        .checked_add(run.bundle_count as usize)
        .ok_or_else(|| RusticolError::integrity("direct fanout bundle range overflows usize"))?;
    let bundles = fanout
        .bundles
        .get(bundle_start..bundle_end)
        .ok_or_else(|| RusticolError::integrity("direct fanout bundle range is out of bounds"))?;
    debug_assert_eq!(
        bundles
            .iter()
            .map(|bundle| u64::from(bundle.row_count))
            .sum::<u64>(),
        u64::from(run.row_count)
    );

    let current_re = workspace.current_re.as_mut_ptr();
    let current_im = workspace.current_im.as_mut_ptr();
    for bundle in bundles {
        let start = usize::try_from(bundle.row_start).map_err(|_| {
            RusticolError::integrity("direct fanout bundle row start exceeds usize")
        })?;
        let end = start
            .checked_add(bundle.row_count as usize)
            .ok_or_else(|| {
                RusticolError::integrity("direct fanout bundle row range overflows usize")
            })?;
        let rows = plan.contributions().get(start..end).ok_or_else(|| {
            RusticolError::integrity("direct fanout bundle row range is out of bounds")
        })?;
        let scale = if bundle.effective_factor_id_or_sentinel == DIRECT_NONE_U32 {
            DirectContributionFanoutScale::Unity
        } else {
            let factor = bundle.effective_factor_id_or_sentinel as usize;
            DirectContributionFanoutScale::new(
                *workspace.factors_re.get(factor).ok_or_else(|| {
                    RusticolError::integrity("direct fanout scatter factor is out of bounds")
                })?,
                *workspace.factors_im.get(factor).ok_or_else(|| {
                    RusticolError::integrity("direct fanout scatter factor is out of bounds")
                })?,
            )
        };
        for component in 0..destination_component_count {
            let source_start = (run.scratch_component_base as usize + component as usize) * stride;
            for point in 0..active_points {
                let source = source_start + point;
                // SAFETY: the source run is checked against the runtime arena
                // above. The destination rows and their full component spans
                // were authenticated by `DirectRecurrencePlan` and matched to
                // this executor's destination shape when the fanout program
                // was built. `workspace.validate` established point < stride.
                let (scaled_re, scaled_im) =
                    unsafe { scale.apply(*current_re.add(source), *current_im.add(source)) };
                for row in rows {
                    let destination =
                        (row.destination_component_base as usize + component as usize) * stride
                            + point;
                    // SAFETY: see the source/destination proof immediately
                    // above. Scratch components form a private suffix, so none
                    // of these writes can alias the representative source.
                    unsafe {
                        if row.flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION != 0 {
                            *current_re.add(destination) = scaled_re;
                            *current_im.add(destination) = scaled_im;
                        } else {
                            *current_re.add(destination) += scaled_re;
                            *current_im.add(destination) += scaled_im;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
#[allow(clippy::too_many_arguments)]
fn observe_direct_current_rows(
    plan: &DirectRecurrencePlan,
    descriptor: &DirectRowGroupDescriptor,
    start: usize,
    end: usize,
    selected_sector_id: Option<u32>,
    workspace: &DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<()> {
    let active = DIRECT_CURRENT_OBSERVATION.with(|slot| slot.borrow().is_some());
    if !active || descriptor.role == DirectExecutorRole::Closure {
        return Ok(());
    }

    let affected = match descriptor.role {
        DirectExecutorRole::Source => plan
            .currents()
            .iter()
            .filter(|current| {
                current.source_row_or_sentinel != DIRECT_NONE_U32
                    && (start..end).contains(&(current.source_row_or_sentinel as usize))
            })
            .map(|current| current.semantic_current_id)
            .collect::<BTreeSet<_>>(),
        DirectExecutorRole::Contribution => {
            let destinations = plan
                .contributions()
                .get(start..end)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "direct current observation contribution range is out of bounds",
                    )
                })?
                .iter()
                .map(|row| row.destination_component_base)
                .collect::<BTreeSet<_>>();
            plan.currents()
                .iter()
                .filter(|current| {
                    current.node_kind == DirectNodeKind::Current
                        && current.stage == descriptor.stage
                        && destinations.iter().any(|destination| {
                            *destination >= current.component_base
                                && *destination
                                    < current.component_base + u32::from(current.component_count)
                        })
                })
                .map(|current| current.semantic_current_id)
                .collect::<BTreeSet<_>>()
        }
        DirectExecutorRole::Finalization => plan
            .currents()
            .iter()
            .filter(|current| {
                current.finalization_row_or_sentinel != DIRECT_NONE_U32
                    && (start..end).contains(&(current.finalization_row_or_sentinel as usize))
            })
            .map(|current| current.semantic_current_id)
            .collect::<BTreeSet<_>>(),
        DirectExecutorRole::Closure => BTreeSet::new(),
    };
    if affected.is_empty() {
        return Ok(());
    }

    DIRECT_CURRENT_OBSERVATION.with(|slot| {
        let mut slot = slot.borrow_mut();
        let observation = slot.as_mut().ok_or_else(|| {
            RusticolError::integrity("direct current observation disappeared during execution")
        })?;
        if observation.plan_semantic_digest != plan.semantic_digest()
            || observation.runtime_layout_digest != plan.runtime_layout_digest()
            || observation.selected_sector_id != selected_sector_id
            || observation.point_count != point_count
        {
            return Err(RusticolError::integrity(
                "direct current observation does not match this plan execution",
            ));
        }
        let stride = workspace.point_stride as usize;
        let active_points = point_count as usize;
        for semantic_current_id in affected {
            let current = plan
                .currents()
                .get(semantic_current_id as usize)
                .filter(|current| current.semantic_current_id == semantic_current_id)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "direct current observation semantic current is absent",
                    )
                })?;
            if let Some(sector_id) = selected_sector_id
                && !plan.selector_domain_contains(current.selector_domain_id, sector_id)?
            {
                continue;
            }
            let mut values = Vec::with_capacity(
                usize::from(current.component_count)
                    .checked_mul(active_points)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "direct current observation value count overflows usize",
                        )
                    })?,
            );
            for component in 0..usize::from(current.component_count) {
                let plane = (current.component_base as usize)
                    .checked_add(component)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "direct current observation component range overflows usize",
                        )
                    })?;
                let begin = plane.checked_mul(stride).ok_or_else(|| {
                    RusticolError::integrity(
                        "direct current observation arena range overflows usize",
                    )
                })?;
                let finish = begin.checked_add(active_points).ok_or_else(|| {
                    RusticolError::integrity(
                        "direct current observation arena range overflows usize",
                    )
                })?;
                let real = workspace.current_re.get(begin..finish).ok_or_else(|| {
                    RusticolError::integrity(
                        "direct current observation real range is out of bounds",
                    )
                })?;
                let imaginary = workspace.current_im.get(begin..finish).ok_or_else(|| {
                    RusticolError::integrity(
                        "direct current observation imaginary range is out of bounds",
                    )
                })?;
                values.extend(real.iter().copied().zip(imaginary.iter().copied()));
            }
            observation.currents.insert(
                semantic_current_id,
                DirectObservedCurrentValue {
                    descriptor: *current,
                    values,
                },
            );
        }
        Ok(())
    })
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
    if status == DIRECT_STATUS_OK {
        return Ok(());
    }
    if let Some(detail) = take_direct_executor_error_detail() {
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
