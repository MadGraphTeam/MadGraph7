// SPDX-License-Identifier: 0BSD

//! Native adapter from public runtime inputs to Direct-Arena recurrence.

use super::recurrence_backend::NativeRecurrenceDirectExecutorOwners;
use super::*;
use crate::direct_arena::DirectArenaTrafficCounters;
use crate::recurrence::direct_backend::{
    DirectExecutionCounters, DirectExecutionRoleTimings, DirectExecutorCatalog,
};
use crate::recurrence::direct_runtime::{
    DirectRecurrenceExecutionRuntime, DirectRecurrenceTileOutput, DirectReplaySelectorPlan,
    DirectRuntimeActivityCounters, DirectRuntimePhaseTimings, DirectUnionHelicitySelectorPlan,
};
use crate::recurrence::{
    DIRECT_NONE_U32, DirectRecurrencePlan, RecurrenceColorContraction, RecurrenceStrategy,
};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug)]
pub(super) struct RecurrenceParameterProjectionEntry {
    pub(super) runtime_slot: usize,
    pub(super) prepared_slot: usize,
    pub(super) component: u8,
}

/// One process-local Direct-Arena runtime.
///
/// `_backend_owner` keeps every immutable source/SymJIT context addressed by
/// the scheduler's lightweight function handles alive. All point-dependent
/// storage belongs to `scheduler` or the fixed-size external-momentum scratch.
pub(super) struct RecurrenceNativeRuntime {
    scheduler: DirectRecurrenceExecutionRuntime,
    _backend_owner: NativeRecurrenceDirectExecutorOwners,
    backend_name: String,
    selectors: RecurrenceNativeSelectors,
    parameter_defaults: Vec<crate::EagerComplex64>,
    parameter_projection: Vec<RecurrenceParameterProjectionEntry>,
    external_source_count: usize,
    external_momenta: Vec<f64>,
    contracted_replay_re: Vec<f64>,
    contracted_replay_im: Vec<f64>,
    color_transform_re: Vec<f64>,
    color_transform_im: Vec<f64>,
}

struct ContractedReplayRoute {
    selector: DirectReplaySelectorPlan,
    destination_copies: Vec<(u32, u32)>,
}

// Boxing the contracted-color payload would add an indirection to every
// contracted recurrence tile; this enum is instantiated only once per runtime.
#[allow(clippy::large_enum_variant)]
enum RecurrenceNativeSelectors {
    TopologyReplay {
        replay_selectors: Vec<DirectReplaySelectorPlan>,
        direct_helicity_to_physics: Vec<usize>,
    },
    AllFlowUnion {
        helicity_selectors_by_physics: Vec<Option<DirectUnionHelicitySelectorPlan>>,
        destination_by_public_flow: Vec<u32>,
    },
    ContractedColorUnion {
        contraction: RecurrenceColorContraction,
        destination_physics_helicity: Vec<usize>,
        replay_routes: Vec<ContractedReplayRoute>,
    },
}

/// Borrowed LC selector and reduction state shared by topology replay and the
/// future compact on-the-fly lane.
///
/// Constructing this view is allocation-free. It deliberately exposes only
/// the public axes and orbit weights needed after an amplitude is computed;
/// an on-the-fly runtime can therefore provide the same backing slices/maps
/// without materializing a dense [`ProcessPhysicsV1`].
#[derive(Clone, Copy)]
pub(super) struct LcSelectorReductionView<'a> {
    helicities: &'a [crate::Helicity],
    color_components: &'a [crate::ColorComponent],
    helicity_index_by_id: &'a BTreeMap<String, usize>,
    color_index_by_id: &'a BTreeMap<String, usize>,
    helicity_members_by_representative: &'a [Vec<usize>],
}

impl<'a> LcSelectorReductionView<'a> {
    pub(super) const fn from_parts(
        helicities: &'a [crate::Helicity],
        color_components: &'a [crate::ColorComponent],
        helicity_index_by_id: &'a BTreeMap<String, usize>,
        color_index_by_id: &'a BTreeMap<String, usize>,
        helicity_members_by_representative: &'a [Vec<usize>],
    ) -> Self {
        Self {
            helicities,
            color_components,
            helicity_index_by_id,
            color_index_by_id,
            helicity_members_by_representative,
        }
    }

    fn validate_selector_ids(
        self,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<()> {
        if let Some(id) = selected_helicities.and_then(|ids| {
            ids.iter()
                .find(|id| !self.helicity_index_by_id.contains_key(*id))
        }) {
            return Err(RusticolError::selector(format!(
                "unknown resolved helicity id {id:?}"
            )));
        }
        if let Some(id) = selected_colors.and_then(|ids| {
            ids.iter()
                .find(|id| !self.color_index_by_id.contains_key(*id))
        }) {
            return Err(RusticolError::selector(format!(
                "unknown resolved color component id {id:?}"
            )));
        }
        Ok(())
    }

    fn selected_indices(
        available: &BTreeMap<String, usize>,
        ids: Option<&BTreeSet<String>>,
        kind: &str,
    ) -> RusticolResult<Vec<usize>> {
        let mut indices = if let Some(ids) = ids {
            ids.iter()
                .map(|id| {
                    available.get(id).copied().ok_or_else(|| {
                        RusticolError::selector(format!("unknown resolved {kind} id {id:?}"))
                    })
                })
                .collect::<RusticolResult<Vec<_>>>()?
        } else {
            (0..available.len()).collect()
        };
        indices.sort_unstable();
        Ok(indices)
    }

    fn selected_helicity_indices(
        self,
        ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<Vec<usize>> {
        Self::selected_indices(self.helicity_index_by_id, ids, "helicity")
    }

    fn selected_color_indices(self, ids: Option<&BTreeSet<String>>) -> RusticolResult<Vec<usize>> {
        Self::selected_indices(self.color_index_by_id, ids, "color component")
    }

    #[inline(always)]
    fn helicity(self, index: usize) -> &'a crate::Helicity {
        &self.helicities[index]
    }

    #[inline(always)]
    fn color(self, index: usize) -> &'a crate::ColorComponent {
        &self.color_components[index]
    }

    #[inline(always)]
    fn color_is_computed(self, index: usize) -> bool {
        match self.color(index) {
            crate::ColorComponent::LcFlow(flow) => flow.computed,
            crate::ColorComponent::ContractedColor(_) => true,
        }
    }

    #[inline(always)]
    fn helicity_orbit_members(self, representative: usize) -> &'a [usize] {
        self.helicity_members_by_representative
            .get(representative)
            .map(Vec::as_slice)
            .unwrap_or_default()
    }

    #[inline(always)]
    fn helicity_is_selected(self, selected: Option<&BTreeSet<String>>, index: usize) -> bool {
        selected.is_none_or(|ids| ids.contains(&self.helicity(index).id))
    }

    #[inline(always)]
    fn helicity_orbit_weight(
        self,
        selected: Option<&BTreeSet<String>>,
        representative_index: usize,
    ) -> f64 {
        self.helicity_orbit_members(representative_index)
            .iter()
            .copied()
            .filter(|index| self.helicity_is_selected(selected, *index))
            .map(|index| self.helicity(index).coefficient)
            .sum()
    }

    #[inline(always)]
    fn color_is_selected(self, selected: Option<&BTreeSet<String>>, index: usize) -> bool {
        selected.is_none_or(|ids| ids.contains(self.color(index).id()))
    }
}

impl PhysicsRuntime {
    pub(super) fn lc_selector_reduction_view(&self) -> LcSelectorReductionView<'_> {
        LcSelectorReductionView::from_parts(
            &self.manifest.helicities,
            &self.manifest.color_components,
            &self.helicity_index_by_id,
            &self.color_index_by_id,
            &self.helicity_members_by_representative,
        )
    }
}

/// Apply the established LC diagonal reduction to one complex amplitude
/// stream. Both topology replay's split real/imaginary planes and the future
/// on-the-fly query-major output can provide their native storage through the
/// inlined accessor without copying or changing the reduction formula.
#[inline(always)]
pub(super) fn accumulate_lc_diagonal_amplitude(
    point_count: usize,
    weight: f64,
    mut value: impl FnMut(usize) -> (f64, f64),
    mut accumulate: impl FnMut(usize, f64),
) {
    for point in 0..point_count {
        let (real, imaginary) = value(point);
        accumulate(point, weight * real.mul_add(real, imaginary * imaginary));
    }
}

impl RecurrenceNativeRuntime {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        plan: DirectRecurrencePlan,
        executors: DirectExecutorCatalog,
        backend_owner: NativeRecurrenceDirectExecutorOwners,
        parameter_defaults: Vec<crate::EagerComplex64>,
        parameter_projection: Vec<RecurrenceParameterProjectionEntry>,
        public_flow_ids: Vec<u32>,
        direct_helicity_to_physics: Vec<usize>,
        color_contraction: Option<RecurrenceColorContraction>,
    ) -> RusticolResult<Self> {
        let parameter_count = usize::try_from(plan.parameter_value_count())
            .map_err(|_| RusticolError::artifact("recurrence parameter count exceeds usize"))?;
        if parameter_defaults.len() != parameter_count {
            return Err(RusticolError::integrity(format!(
                "recurrence prepared defaults have length {}, expected {parameter_count}",
                parameter_defaults.len()
            )));
        }
        for entry in &parameter_projection {
            if entry.prepared_slot >= parameter_count || entry.component > 1 {
                return Err(RusticolError::artifact(
                    "recurrence parameter projection is outside the prepared layout",
                ));
            }
        }
        if direct_helicity_to_physics.len() != plan.resolved_helicities().len() {
            return Err(RusticolError::integrity(
                "recurrence resolved-helicity mapping does not cover the direct plan",
            ));
        }
        let backend_name = backend_owner.summary().backend.clone();
        let strategy = plan.strategy();
        let has_color_contraction = color_contraction.is_some();
        if public_flow_ids.is_empty() && strategy != RecurrenceStrategy::ContractedColorUnion {
            return Err(RusticolError::integrity(
                "recurrence public color-flow mapping is empty",
            ));
        }
        let scheduler = match strategy {
            RecurrenceStrategy::TopologyReplay | RecurrenceStrategy::ContractedColorUnion => {
                DirectRecurrenceExecutionRuntime::new(plan, executors, 4)?
            }
            RecurrenceStrategy::AllFlowUnion => {
                let dispatch = backend_owner.union_source_dispatch()?;
                DirectRecurrenceExecutionRuntime::new_with_union_source_dispatch(
                    plan, executors, 4, dispatch,
                )?
            }
        };
        let color_transform_scratch_len = color_contraction
            .as_ref()
            .and_then(|contraction| {
                contraction
                    .runtime_factorization()
                    .map(|_| contraction.local_group_count())
            })
            .map(|local_group_count| {
                usize::try_from(local_group_count)
                    .ok()
                    .and_then(|groups| {
                        usize::try_from(scheduler.point_tile_size())
                            .ok()
                            .and_then(|points| groups.checked_mul(points))
                    })
                    .ok_or_else(|| {
                        RusticolError::artifact(
                            "recurrence factorized color workspace overflows usize",
                        )
                    })
            })
            .transpose()?
            .unwrap_or(0);
        let selectors = match strategy {
            RecurrenceStrategy::TopologyReplay => {
                let replay_selectors = public_flow_ids
                    .into_iter()
                    .map(|public_flow_id| scheduler.prepare_replay_selector(public_flow_id))
                    .collect::<RusticolResult<Vec<_>>>()?;
                validate_replay_destination_helicity_mappings(
                    scheduler.plan(),
                    &replay_selectors,
                    &direct_helicity_to_physics,
                )?;
                RecurrenceNativeSelectors::TopologyReplay {
                    replay_selectors,
                    direct_helicity_to_physics,
                }
            }
            RecurrenceStrategy::AllFlowUnion => {
                let physics_helicity_count = direct_helicity_to_physics
                    .iter()
                    .copied()
                    .max()
                    .and_then(|value| value.checked_add(1))
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "all-flow-union recurrence has no public helicities",
                        )
                    })?;
                let mut helicity_selectors_by_physics = vec![None; physics_helicity_count];
                for (direct_id, physics_id) in
                    direct_helicity_to_physics.iter().copied().enumerate()
                {
                    let selector = scheduler.prepare_union_helicity_selector(
                        u32::try_from(direct_id).map_err(|_| {
                            RusticolError::artifact("recurrence direct helicity ID exceeds u32")
                        })?,
                    )?;
                    if helicity_selectors_by_physics[physics_id]
                        .replace(selector)
                        .is_some()
                    {
                        return Err(RusticolError::integrity(
                            "all-flow-union recurrence repeats a public helicity",
                        ));
                    }
                }
                if helicity_selectors_by_physics.iter().any(Option::is_none) {
                    return Err(RusticolError::integrity(
                        "all-flow-union recurrence does not cover the public helicity axis",
                    ));
                }
                let destination_by_public_flow =
                    union_destination_ids(scheduler.plan(), &public_flow_ids)?;
                RecurrenceNativeSelectors::AllFlowUnion {
                    helicity_selectors_by_physics,
                    destination_by_public_flow,
                }
            }
            RecurrenceStrategy::ContractedColorUnion => {
                if !public_flow_ids.is_empty() {
                    return Err(RusticolError::integrity(
                        "contracted recurrence unexpectedly exposes public color flows",
                    ));
                }
                let contraction = color_contraction.ok_or_else(|| {
                    RusticolError::integrity(
                        "contracted recurrence has no color-contraction payload",
                    )
                })?;
                let destination_physics_helicity =
                    contracted_destination_helicity_map(&contraction, &direct_helicity_to_physics)?;
                let replay_routes = contracted_replay_routes(&scheduler, &contraction)?;
                RecurrenceNativeSelectors::ContractedColorUnion {
                    contraction,
                    destination_physics_helicity,
                    replay_routes,
                }
            }
        };
        if strategy != RecurrenceStrategy::ContractedColorUnion && has_color_contraction {
            return Err(RusticolError::integrity(
                "LC recurrence unexpectedly carries a color-contraction payload",
            ));
        }
        let external_source_count = usize::try_from(scheduler.plan().external_source_count())
            .map_err(|_| RusticolError::artifact("recurrence source count exceeds usize"))?;
        let scratch_len = scheduler
            .point_tile_size()
            .try_into()
            .ok()
            .and_then(|points: usize| points.checked_mul(external_source_count))
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| {
                RusticolError::artifact("recurrence external-momentum workspace overflows usize")
            })?;
        let external_momenta = vec![0.0; scratch_len];
        let contracted_replay_len = match &selectors {
            RecurrenceNativeSelectors::ContractedColorUnion {
                contraction,
                replay_routes,
                ..
            } if !replay_routes.is_empty() => usize::try_from(contraction.destination_count())
                .ok()
                .and_then(|destinations| {
                    usize::try_from(scheduler.point_tile_size())
                        .ok()
                        .and_then(|points| destinations.checked_mul(points))
                })
                .ok_or_else(|| {
                    RusticolError::artifact("contracted replay amplitude workspace overflows usize")
                })?,
            _ => 0,
        };

        Ok(Self {
            scheduler,
            _backend_owner: backend_owner,
            backend_name,
            selectors,
            parameter_defaults,
            parameter_projection,
            external_source_count,
            external_momenta,
            contracted_replay_re: vec![0.0; contracted_replay_len],
            contracted_replay_im: vec![0.0; contracted_replay_len],
            color_transform_re: vec![0.0; color_transform_scratch_len],
            color_transform_im: vec![0.0; color_transform_scratch_len],
        })
    }

    pub(super) fn backend_name(&self) -> &str {
        &self.backend_name
    }

    pub(super) fn effective_point_tile_size(&self) -> usize {
        self.scheduler.point_tile_size() as usize
    }

    pub(super) fn validate_global_selectors(
        &self,
        physics: &PhysicsRuntime,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<()> {
        self.validate_public_axis_lengths(physics)?;
        if matches!(
            &self.selectors,
            RecurrenceNativeSelectors::TopologyReplay { .. }
        ) {
            physics
                .lc_selector_reduction_view()
                .validate_selector_ids(selected_helicities, selected_colors)?;
        } else {
            validate_recurrence_selector_ids(physics, selected_helicities, selected_colors)?;
        }
        if matches!(
            &self.selectors,
            RecurrenceNativeSelectors::ContractedColorUnion { .. }
        ) {
            reject_contracted_color_selector(selected_colors)?;
        }
        Ok(())
    }

    pub(super) fn run_f64(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        self.run_f64_with_global_selectors(common, batch, None, None)
    }

    /// Clock-free totals path used by the native f64 ABI.
    ///
    /// Momentum input and result storage remain borrowed throughout this
    /// call. Selector membership is tested against the authenticated public
    /// axes directly so a warmed evaluation does not allocate temporary
    /// index vectors.
    pub(super) fn run_f64_view_into_unprofiled(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        if output.len() != batch.point_count() {
            return Err(RusticolError::invalid_argument(format!(
                "recurrence output has length {}, expected {}",
                output.len(),
                batch.point_count()
            )));
        }
        if batch.external_count() != self.external_source_count {
            return Err(RusticolError::invalid_argument(format!(
                "recurrence input has {} external momenta, expected {}",
                batch.external_count(),
                self.external_source_count
            )));
        }
        let physics = common.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact("recurrence execution requires physics metadata")
        })?;
        self.validate_global_selectors(physics, selected_helicities, selected_colors)?;
        output.fill(0.0);
        self.prepare_parameters(common)?;

        match &self.selectors {
            RecurrenceNativeSelectors::TopologyReplay { .. } => self
                .run_replay_view_into_unprofiled(
                    physics.lc_selector_reduction_view(),
                    common.normalization_factor,
                    batch,
                    selected_helicities,
                    selected_colors,
                    output,
                ),
            RecurrenceNativeSelectors::AllFlowUnion { .. } => self.run_union_view_into_unprofiled(
                physics,
                common.normalization_factor,
                batch,
                selected_helicities,
                selected_colors,
                output,
            ),
            RecurrenceNativeSelectors::ContractedColorUnion { .. } => self
                .run_contracted_view_into_unprofiled(
                    physics,
                    common.normalization_factor,
                    batch,
                    selected_helicities,
                    selected_colors,
                    output,
                ),
        }
    }

    fn run_replay_view_into_unprofiled(
        &mut self,
        reduction: LcSelectorReductionView<'_>,
        normalization_factor: f64,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        for color_index in 0..reduction.color_components.len() {
            if !reduction.color_is_computed(color_index)
                || !reduction.color_is_selected(selected_colors, color_index)
            {
                continue;
            }
            let color_weight = reduction.color(color_index).coefficient();

            let mut tile_start = 0usize;
            while tile_start < batch.point_count() {
                let tile_stop =
                    (tile_start + self.effective_point_tile_size()).min(batch.point_count());
                let point_count =
                    self.flatten_external_tile_view(batch.subview(tile_start, tile_stop)?)?;
                let input_len = self.external_tile_input_len(point_count)?;
                let (replay_selector, direct_helicity_to_physics) = match &self.selectors {
                    RecurrenceNativeSelectors::TopologyReplay {
                        replay_selectors,
                        direct_helicity_to_physics,
                    } => (
                        replay_selectors.get(color_index).ok_or_else(|| {
                            RusticolError::integrity(
                                "recurrence replay selector is outside the public color axis",
                            )
                        })?,
                        direct_helicity_to_physics,
                    ),
                    RecurrenceNativeSelectors::AllFlowUnion { .. } => unreachable!(),
                    RecurrenceNativeSelectors::ContractedColorUnion { .. } => unreachable!(),
                };
                let direct_output = self
                    .scheduler
                    .execute_replay_tile_from_external_unprofiled(
                        replay_selector,
                        direct_point_count(point_count)?,
                        &self.external_momenta[..input_len],
                    )?;

                for destination_id in direct_output.selected_destination_ids() {
                    let helicity_index = replay_output_destination_physics_helicity(
                        &direct_output,
                        replay_selector,
                        direct_helicity_to_physics,
                        destination_id,
                    )?;
                    let helicity = reduction.helicity(helicity_index);
                    if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0
                    {
                        continue;
                    }
                    let helicity_weight =
                        reduction.helicity_orbit_weight(selected_helicities, helicity_index);
                    if helicity_weight == 0.0 {
                        continue;
                    }
                    let values_re =
                        direct_output
                            .destination_re(destination_id)
                            .ok_or_else(|| {
                                RusticolError::integrity(
                                    "recurrence selected amplitude destination is absent",
                                )
                            })?;
                    let values_im =
                        direct_output
                            .destination_im(destination_id)
                            .ok_or_else(|| {
                                RusticolError::integrity(
                                    "recurrence selected amplitude destination is absent",
                                )
                            })?;
                    let weight = helicity_weight * color_weight * normalization_factor;
                    accumulate_lc_diagonal_amplitude(
                        point_count,
                        weight,
                        |point| (values_re[point], values_im[point]),
                        |point, value| output[tile_start + point] += value,
                    );
                }
                tile_start = tile_stop;
            }
        }
        Ok(())
    }

    fn run_union_view_into_unprofiled(
        &mut self,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        for helicity_index in 0..physics.manifest.helicities.len() {
            if !recurrence_helicity_is_selected(physics, selected_helicities, helicity_index) {
                continue;
            }
            let helicity = &physics.manifest.helicities[helicity_index];
            if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0 {
                continue;
            }
            let selector = self.union_helicity_selector(helicity_index)?;
            let mut tile_start = 0usize;
            while tile_start < batch.point_count() {
                let tile_stop =
                    (tile_start + self.effective_point_tile_size()).min(batch.point_count());
                let point_count =
                    self.flatten_external_tile_view(batch.subview(tile_start, tile_stop)?)?;
                let input_len = self.external_tile_input_len(point_count)?;
                let destination_by_public_flow = match &self.selectors {
                    RecurrenceNativeSelectors::AllFlowUnion {
                        destination_by_public_flow,
                        ..
                    } => destination_by_public_flow,
                    RecurrenceNativeSelectors::TopologyReplay { .. } => unreachable!(),
                    RecurrenceNativeSelectors::ContractedColorUnion { .. } => unreachable!(),
                };
                let direct_output = self.scheduler.execute_union_tile_from_external_unprofiled(
                    &selector,
                    direct_point_count(point_count)?,
                    &self.external_momenta[..input_len],
                )?;

                for color_index in 0..physics.manifest.color_components.len() {
                    if !physics.color_is_computed(color_index)
                        || !recurrence_color_is_selected(physics, selected_colors, color_index)
                    {
                        continue;
                    }
                    let destination_id =
                        *destination_by_public_flow.get(color_index).ok_or_else(|| {
                            RusticolError::integrity(
                                "all-flow-union color is outside retained public coverage",
                            )
                        })?;
                    let values_re =
                        direct_output
                            .destination_re(destination_id)
                            .ok_or_else(|| {
                                RusticolError::integrity(
                                    "all-flow-union amplitude destination is absent",
                                )
                            })?;
                    let values_im =
                        direct_output
                            .destination_im(destination_id)
                            .ok_or_else(|| {
                                RusticolError::integrity(
                                    "all-flow-union amplitude destination is absent",
                                )
                            })?;
                    let color_weight = physics.manifest.color_components[color_index].coefficient();
                    let weight = helicity.coefficient * color_weight * normalization_factor;
                    for point in 0..point_count {
                        output[tile_start + point] += weight
                            * values_re[point]
                                .mul_add(values_re[point], values_im[point] * values_im[point]);
                    }
                }
                tile_start = tile_stop;
            }
        }
        Ok(())
    }

    fn run_contracted_view_into_unprofiled(
        &mut self,
        physics: &PhysicsRuntime,
        normalization_factor: f64,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        output: &mut [f64],
    ) -> RusticolResult<()> {
        reject_contracted_color_selector(selected_colors)?;
        let mut tile_start = 0usize;
        while tile_start < batch.point_count() {
            let tile_stop =
                (tile_start + self.effective_point_tile_size()).min(batch.point_count());
            let point_count =
                self.flatten_external_tile_view(batch.subview(tile_start, tile_stop)?)?;
            let input_len = self.external_tile_input_len(point_count)?;
            self.execute_and_contract_contracted_tile(
                point_count,
                input_len,
                physics,
                selected_helicities,
                normalization_factor,
                false,
                |point, _helicity, value| output[tile_start + point] += value,
            )?;
            tile_start = tile_stop;
        }
        Ok(())
    }

    pub(super) fn run_f64_with_global_selectors(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        match &self.selectors {
            RecurrenceNativeSelectors::AllFlowUnion { .. } => {
                return self.run_union_f64_with_global_selectors(
                    common,
                    batch,
                    selected_helicities,
                    selected_colors,
                );
            }
            RecurrenceNativeSelectors::ContractedColorUnion { .. } => {
                return self.run_contracted_f64_with_global_selectors(
                    common,
                    batch,
                    selected_helicities,
                    selected_colors,
                );
            }
            RecurrenceNativeSelectors::TopologyReplay { .. } => {}
        }
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "recurrence evaluation requires at least one point",
            ));
        }
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("recurrence execution requires physics metadata")
        })?;
        let reduction_view = physics.lc_selector_reduction_view();
        let helicity_indices = reduction_view.selected_helicity_indices(selected_helicities)?;
        let color_indices = reduction_view.selected_color_indices(selected_colors)?;
        self.validate_public_axes(&physics, &helicity_indices, &color_indices)?;

        let mut values = vec![0.0; batch.len()];
        let profile_before = DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner);
        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let parameter_setup = parameter_started.elapsed();
        let mut external_momentum_flatten = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        for color_index in color_indices {
            if !reduction_view.color_is_computed(color_index) {
                continue;
            }
            let color_weight = reduction_view.color(color_index).coefficient();
            let mut tile_start = 0usize;
            while tile_start < batch.len() {
                let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
                let flatten_started = Instant::now();
                let point_count = self.flatten_external_tile(&batch[tile_start..tile_stop])?;
                external_momentum_flatten += flatten_started.elapsed();
                let input_len = point_count
                    .checked_mul(self.external_source_count)
                    .and_then(|values| values.checked_mul(4))
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "recurrence external-momentum tile length overflows",
                        )
                    })?;
                let (replay_selector, direct_helicity_to_physics) = match &self.selectors {
                    RecurrenceNativeSelectors::TopologyReplay {
                        replay_selectors,
                        direct_helicity_to_physics,
                    } => (
                        replay_selectors.get(color_index).ok_or_else(|| {
                            RusticolError::integrity(
                                "recurrence replay selector is outside the public color axis",
                            )
                        })?,
                        direct_helicity_to_physics,
                    ),
                    RecurrenceNativeSelectors::AllFlowUnion { .. } => unreachable!(),
                    RecurrenceNativeSelectors::ContractedColorUnion { .. } => unreachable!(),
                };
                let output = self.scheduler.execute_replay_tile_from_external(
                    replay_selector,
                    u32::try_from(point_count).map_err(|_| {
                        RusticolError::invalid_argument(
                            "recurrence point tile exceeds the native u32 ABI",
                        )
                    })?,
                    &self.external_momenta[..input_len],
                )?;

                let reduction_started = Instant::now();
                for destination_id in output.selected_destination_ids() {
                    let helicity_index = replay_output_destination_physics_helicity(
                        &output,
                        replay_selector,
                        direct_helicity_to_physics,
                        destination_id,
                    )?;
                    let helicity = reduction_view.helicity(helicity_index);
                    if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0
                    {
                        continue;
                    }
                    let helicity_weight =
                        reduction_view.helicity_orbit_weight(selected_helicities, helicity_index);
                    if helicity_weight == 0.0 {
                        continue;
                    }
                    let values_re = output.destination_re(destination_id).ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence selected amplitude destination is absent",
                        )
                    })?;
                    let values_im = output.destination_im(destination_id).ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence selected amplitude destination is absent",
                        )
                    })?;
                    let weight = helicity_weight * color_weight * common.normalization_factor;
                    accumulate_lc_diagonal_amplitude(
                        point_count,
                        weight,
                        |point| (values_re[point], values_im[point]),
                        |point, value| values[tile_start + point] += value,
                    );
                }
                reduction += reduction_started.elapsed();
                tile_start = tile_stop;
            }
        }

        Ok((
            values,
            direct_profile(
                total_started.elapsed(),
                parameter_setup,
                external_momentum_flatten,
                reduction,
                profile_before,
                DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner),
            ),
        ))
    }

    pub(super) fn run_resolved_f64(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        match &self.selectors {
            RecurrenceNativeSelectors::AllFlowUnion { .. } => {
                return self.run_union_resolved_f64(
                    common,
                    batch,
                    selected_helicities,
                    selected_colors,
                );
            }
            RecurrenceNativeSelectors::ContractedColorUnion { .. } => {
                return self.run_contracted_resolved_f64(
                    common,
                    batch,
                    selected_helicities,
                    selected_colors,
                );
            }
            RecurrenceNativeSelectors::TopologyReplay { .. } => {}
        }
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "resolved recurrence evaluation requires at least one point",
            ));
        }
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("resolved recurrence execution requires physics metadata")
        })?;
        let reduction_view = physics.lc_selector_reduction_view();
        let helicity_indices = reduction_view.selected_helicity_indices(selected_helicities)?;
        let color_indices = reduction_view.selected_color_indices(selected_colors)?;
        self.validate_public_axes(&physics, &helicity_indices, &color_indices)?;
        let component_count = helicity_indices
            .len()
            .checked_mul(color_indices.len())
            .ok_or_else(|| {
                RusticolError::invalid_argument("recurrence resolved shape overflows")
            })?;
        let mut values = vec![
            0.0;
            batch.len().checked_mul(component_count).ok_or_else(|| {
                RusticolError::invalid_argument("recurrence resolved output overflows")
            })?
        ];
        let mut helicity_position = vec![None; reduction_view.helicities.len()];
        for (position, index) in helicity_indices.iter().copied().enumerate() {
            helicity_position[index] = Some(position);
        }

        let profile_before = DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner);
        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let parameter_setup = parameter_started.elapsed();
        let mut external_momentum_flatten = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        for (color_position, color_index) in color_indices.iter().copied().enumerate() {
            if !reduction_view.color_is_computed(color_index) {
                continue;
            }
            let color_weight = reduction_view.color(color_index).coefficient();
            let mut tile_start = 0usize;
            while tile_start < batch.len() {
                let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
                let flatten_started = Instant::now();
                let point_count = self.flatten_external_tile(&batch[tile_start..tile_stop])?;
                external_momentum_flatten += flatten_started.elapsed();
                let input_len = point_count
                    .checked_mul(self.external_source_count)
                    .and_then(|count| count.checked_mul(4))
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "recurrence external-momentum tile length overflows",
                        )
                    })?;
                let (replay_selector, direct_helicity_to_physics) = match &self.selectors {
                    RecurrenceNativeSelectors::TopologyReplay {
                        replay_selectors,
                        direct_helicity_to_physics,
                    } => (
                        replay_selectors.get(color_index).ok_or_else(|| {
                            RusticolError::integrity(
                                "recurrence replay selector is outside the public color axis",
                            )
                        })?,
                        direct_helicity_to_physics,
                    ),
                    RecurrenceNativeSelectors::AllFlowUnion { .. } => unreachable!(),
                    RecurrenceNativeSelectors::ContractedColorUnion { .. } => unreachable!(),
                };
                let output = self.scheduler.execute_replay_tile_from_external(
                    replay_selector,
                    u32::try_from(point_count).map_err(|_| {
                        RusticolError::invalid_argument(
                            "recurrence point tile exceeds the native u32 ABI",
                        )
                    })?,
                    &self.external_momenta[..input_len],
                )?;

                let reduction_started = Instant::now();
                for destination_id in output.selected_destination_ids() {
                    let helicity_index = replay_output_destination_physics_helicity(
                        &output,
                        replay_selector,
                        direct_helicity_to_physics,
                        destination_id,
                    )?;
                    let helicity = reduction_view.helicity(helicity_index);
                    if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0
                    {
                        continue;
                    }
                    let values_re = output.destination_re(destination_id).ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence selected amplitude destination is absent",
                        )
                    })?;
                    let values_im = output.destination_im(destination_id).ok_or_else(|| {
                        RusticolError::integrity(
                            "recurrence selected amplitude destination is absent",
                        )
                    })?;
                    for physical_helicity in reduction_view.helicity_orbit_members(helicity_index) {
                        let Some(helicity_position) = helicity_position[*physical_helicity] else {
                            continue;
                        };
                        let weight = reduction_view.helicity(*physical_helicity).coefficient
                            * color_weight
                            * common.normalization_factor;
                        accumulate_lc_diagonal_amplitude(
                            point_count,
                            weight,
                            |point| (values_re[point], values_im[point]),
                            |point, value| {
                                let target = (tile_start + point) * component_count
                                    + helicity_position * color_indices.len()
                                    + color_position;
                                values[target] += value;
                            },
                        );
                    }
                }
                reduction += reduction_started.elapsed();
                tile_start = tile_stop;
            }
        }

        Ok((
            ResolvedValues {
                values,
                point_count: batch.len(),
                helicity_indices,
                color_indices,
            },
            direct_profile(
                total_started.elapsed(),
                parameter_setup,
                external_momentum_flatten,
                reduction,
                profile_before,
                DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner),
            ),
        ))
    }

    fn run_contracted_f64_with_global_selectors(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "recurrence evaluation requires at least one point",
            ));
        }
        reject_contracted_color_selector(selected_colors)?;
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("recurrence execution requires physics metadata")
        })?;
        let helicity_indices = physics.selected_helicity_indices(selected_helicities)?;
        self.validate_public_axes(&physics, &helicity_indices, &[0])?;
        let mut values = vec![0.0; batch.len()];
        let profile_before = DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner);
        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let parameter_setup = parameter_started.elapsed();
        let mut external_momentum_flatten = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        let mut tile_start = 0usize;
        while tile_start < batch.len() {
            let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
            let flatten_started = Instant::now();
            let point_count = self.flatten_external_tile(&batch[tile_start..tile_stop])?;
            external_momentum_flatten += flatten_started.elapsed();
            let input_len = self.external_tile_input_len(point_count)?;
            reduction += self.execute_and_contract_contracted_tile(
                point_count,
                input_len,
                &physics,
                selected_helicities,
                common.normalization_factor,
                true,
                |point, _helicity, value| values[tile_start + point] += value,
            )?;
            tile_start = tile_stop;
        }

        Ok((
            values,
            direct_profile(
                total_started.elapsed(),
                parameter_setup,
                external_momentum_flatten,
                reduction,
                profile_before,
                DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner),
            ),
        ))
    }

    fn run_contracted_resolved_f64(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "resolved recurrence evaluation requires at least one point",
            ));
        }
        reject_contracted_color_selector(selected_colors)?;
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("resolved recurrence execution requires physics metadata")
        })?;
        let helicity_indices = physics.selected_helicity_indices(selected_helicities)?;
        self.validate_public_axes(&physics, &helicity_indices, &[0])?;
        let mut helicity_position = vec![None; physics.manifest.helicities.len()];
        for (position, index) in helicity_indices.iter().copied().enumerate() {
            helicity_position[index] = Some(position);
        }
        let component_count = helicity_indices.len();
        let mut values = vec![
            0.0;
            batch.len().checked_mul(component_count).ok_or_else(|| {
                RusticolError::invalid_argument("recurrence resolved output overflows")
            })?
        ];
        let profile_before = DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner);
        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let parameter_setup = parameter_started.elapsed();
        let mut external_momentum_flatten = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        let mut tile_start = 0usize;
        while tile_start < batch.len() {
            let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
            let flatten_started = Instant::now();
            let point_count = self.flatten_external_tile(&batch[tile_start..tile_stop])?;
            external_momentum_flatten += flatten_started.elapsed();
            let input_len = self.external_tile_input_len(point_count)?;
            reduction += self.execute_and_contract_contracted_tile(
                point_count,
                input_len,
                &physics,
                selected_helicities,
                common.normalization_factor,
                true,
                |point, helicity, value| {
                    let position = helicity_position[helicity]
                        .expect("selected contracted helicity has a result position");
                    values[(tile_start + point) * component_count + position] += value;
                },
            )?;
            tile_start = tile_stop;
        }

        Ok((
            ResolvedValues {
                values,
                point_count: batch.len(),
                helicity_indices,
                color_indices: vec![0],
            },
            direct_profile(
                total_started.elapsed(),
                parameter_setup,
                external_momentum_flatten,
                reduction,
                profile_before,
                DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner),
            ),
        ))
    }

    fn run_union_f64_with_global_selectors(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "recurrence evaluation requires at least one point",
            ));
        }
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("recurrence execution requires physics metadata")
        })?;
        let helicity_indices = physics.selected_helicity_indices(selected_helicities)?;
        let color_indices = physics.selected_color_indices(selected_colors)?;
        self.validate_public_axes(&physics, &helicity_indices, &color_indices)?;
        let color_destinations = color_indices
            .iter()
            .copied()
            .map(|color_index| Ok((color_index, self.union_destination_id(color_index)?)))
            .collect::<RusticolResult<Vec<_>>>()?;

        let mut values = vec![0.0; batch.len()];
        let profile_before = DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner);
        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let parameter_setup = parameter_started.elapsed();
        let mut external_momentum_flatten = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        for helicity_index in helicity_indices {
            let helicity = &physics.manifest.helicities[helicity_index];
            if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0 {
                continue;
            }
            let selector = self.union_helicity_selector(helicity_index)?;
            let mut tile_start = 0usize;
            while tile_start < batch.len() {
                let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
                let flatten_started = Instant::now();
                let point_count = self.flatten_external_tile(&batch[tile_start..tile_stop])?;
                external_momentum_flatten += flatten_started.elapsed();
                let input_len = point_count
                    .checked_mul(self.external_source_count)
                    .and_then(|count| count.checked_mul(4))
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "recurrence external-momentum tile length overflows",
                        )
                    })?;
                let output = self.scheduler.execute_union_tile_from_external(
                    &selector,
                    u32::try_from(point_count).map_err(|_| {
                        RusticolError::invalid_argument(
                            "recurrence point tile exceeds the native u32 ABI",
                        )
                    })?,
                    &self.external_momenta[..input_len],
                )?;

                let reduction_started = Instant::now();
                for (color_index, destination_id) in color_destinations.iter().copied() {
                    if !physics.color_is_computed(color_index) {
                        continue;
                    }
                    let values_re = output.destination_re(destination_id).ok_or_else(|| {
                        RusticolError::integrity("all-flow-union amplitude destination is absent")
                    })?;
                    let values_im = output.destination_im(destination_id).ok_or_else(|| {
                        RusticolError::integrity("all-flow-union amplitude destination is absent")
                    })?;
                    let color_weight = physics.manifest.color_components[color_index].coefficient();
                    let weight = helicity.coefficient * color_weight * common.normalization_factor;
                    for point in 0..point_count {
                        values[tile_start + point] += weight
                            * values_re[point]
                                .mul_add(values_re[point], values_im[point] * values_im[point]);
                    }
                }
                reduction += reduction_started.elapsed();
                tile_start = tile_stop;
            }
        }

        Ok((
            values,
            direct_profile(
                total_started.elapsed(),
                parameter_setup,
                external_momentum_flatten,
                reduction,
                profile_before,
                DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner),
            ),
        ))
    }

    fn run_union_resolved_f64(
        &mut self,
        common: &mut ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(ResolvedValues<f64>, RuntimeProfile)> {
        if batch.is_empty() {
            return Err(RusticolError::invalid_argument(
                "resolved recurrence evaluation requires at least one point",
            ));
        }
        let total_started = Instant::now();
        let physics = common.physics.clone().ok_or_else(|| {
            RusticolError::artifact("resolved recurrence execution requires physics metadata")
        })?;
        let helicity_indices = physics.selected_helicity_indices(selected_helicities)?;
        let color_indices = physics.selected_color_indices(selected_colors)?;
        self.validate_public_axes(&physics, &helicity_indices, &color_indices)?;
        let color_destinations = color_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(color_position, color_index)| {
                Ok((
                    color_position,
                    color_index,
                    self.union_destination_id(color_index)?,
                ))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let component_count = helicity_indices
            .len()
            .checked_mul(color_indices.len())
            .ok_or_else(|| {
                RusticolError::invalid_argument("recurrence resolved shape overflows")
            })?;
        let mut values = vec![
            0.0;
            batch.len().checked_mul(component_count).ok_or_else(|| {
                RusticolError::invalid_argument("recurrence resolved output overflows")
            })?
        ];

        let profile_before = DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner);
        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let parameter_setup = parameter_started.elapsed();
        let mut external_momentum_flatten = Duration::ZERO;
        let mut reduction = Duration::ZERO;

        for (helicity_position, helicity_index) in helicity_indices.iter().copied().enumerate() {
            let helicity = &physics.manifest.helicities[helicity_index];
            if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0 {
                continue;
            }
            let selector = self.union_helicity_selector(helicity_index)?;
            let mut tile_start = 0usize;
            while tile_start < batch.len() {
                let tile_stop = (tile_start + self.effective_point_tile_size()).min(batch.len());
                let flatten_started = Instant::now();
                let point_count = self.flatten_external_tile(&batch[tile_start..tile_stop])?;
                external_momentum_flatten += flatten_started.elapsed();
                let input_len = point_count
                    .checked_mul(self.external_source_count)
                    .and_then(|count| count.checked_mul(4))
                    .ok_or_else(|| {
                        RusticolError::invalid_argument(
                            "recurrence external-momentum tile length overflows",
                        )
                    })?;
                let output = self.scheduler.execute_union_tile_from_external(
                    &selector,
                    u32::try_from(point_count).map_err(|_| {
                        RusticolError::invalid_argument(
                            "recurrence point tile exceeds the native u32 ABI",
                        )
                    })?,
                    &self.external_momenta[..input_len],
                )?;

                let reduction_started = Instant::now();
                for (color_position, color_index, destination_id) in
                    color_destinations.iter().copied()
                {
                    if !physics.color_is_computed(color_index) {
                        continue;
                    }
                    let values_re = output.destination_re(destination_id).ok_or_else(|| {
                        RusticolError::integrity("all-flow-union amplitude destination is absent")
                    })?;
                    let values_im = output.destination_im(destination_id).ok_or_else(|| {
                        RusticolError::integrity("all-flow-union amplitude destination is absent")
                    })?;
                    let color_weight = physics.manifest.color_components[color_index].coefficient();
                    let weight = helicity.coefficient * color_weight * common.normalization_factor;
                    for point in 0..point_count {
                        let target = (tile_start + point) * component_count
                            + helicity_position * color_indices.len()
                            + color_position;
                        values[target] += weight
                            * values_re[point]
                                .mul_add(values_re[point], values_im[point] * values_im[point]);
                    }
                }
                reduction += reduction_started.elapsed();
                tile_start = tile_stop;
            }
        }

        Ok((
            ResolvedValues {
                values,
                point_count: batch.len(),
                helicity_indices,
                color_indices,
            },
            direct_profile(
                total_started.elapsed(),
                parameter_setup,
                external_momentum_flatten,
                reduction,
                profile_before,
                DirectProfileSnapshot::capture(&self.scheduler, &self._backend_owner),
            ),
        ))
    }

    fn union_helicity_selector(
        &self,
        physics_helicity_index: usize,
    ) -> RusticolResult<DirectUnionHelicitySelectorPlan> {
        match &self.selectors {
            RecurrenceNativeSelectors::AllFlowUnion {
                helicity_selectors_by_physics,
                ..
            } => helicity_selectors_by_physics
                .get(physics_helicity_index)
                .copied()
                .flatten()
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "all-flow-union helicity is outside retained public coverage",
                    )
                }),
            RecurrenceNativeSelectors::TopologyReplay { .. } => Err(RusticolError::integrity(
                "union selector requested from topology replay",
            )),
            RecurrenceNativeSelectors::ContractedColorUnion { .. } => {
                Err(RusticolError::integrity(
                    "union selector requested from contracted-color recurrence",
                ))
            }
        }
    }

    fn union_destination_id(&self, color_index: usize) -> RusticolResult<u32> {
        match &self.selectors {
            RecurrenceNativeSelectors::AllFlowUnion {
                destination_by_public_flow,
                ..
            } => destination_by_public_flow
                .get(color_index)
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "all-flow-union color is outside retained public coverage",
                    )
                }),
            RecurrenceNativeSelectors::TopologyReplay { .. } => Err(RusticolError::integrity(
                "union destination requested from topology replay",
            )),
            RecurrenceNativeSelectors::ContractedColorUnion { .. } => {
                Err(RusticolError::integrity(
                    "union destination requested from contracted-color recurrence",
                ))
            }
        }
    }

    fn prepare_parameters(&mut self, common: &ExecutionRuntime) -> RusticolResult<()> {
        let (parameters_re, parameters_im) = self.scheduler.parameters_mut();
        if parameters_re.len() != self.parameter_defaults.len()
            || parameters_im.len() != self.parameter_defaults.len()
        {
            return Err(RusticolError::integrity(
                "recurrence prepared parameter workspace has the wrong size",
            ));
        }
        for ((real, imaginary), default) in parameters_re
            .iter_mut()
            .zip(parameters_im.iter_mut())
            .zip(&self.parameter_defaults)
        {
            *real = default.re;
            *imaginary = default.im;
        }
        for entry in &self.parameter_projection {
            let value = common
                .model_parameter_values_f64
                .get(entry.runtime_slot)
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "recurrence runtime parameter projection is out of range",
                    )
                })?;
            if entry.component == 0 {
                parameters_re[entry.prepared_slot] = value;
            } else {
                parameters_im[entry.prepared_slot] = value;
            }
        }
        Ok(())
    }

    fn flatten_external_tile(&mut self, batch: &[Vec<[f64; 4]>]) -> RusticolResult<usize> {
        let point_count = batch.len();
        if point_count > self.effective_point_tile_size() {
            return Err(RusticolError::invalid_argument(
                "recurrence point tile exceeds its persistent workspace",
            ));
        }
        for (point_index, point) in batch.iter().enumerate() {
            if point.len() != self.external_source_count {
                return Err(RusticolError::invalid_argument(format!(
                    "recurrence point has {} external momenta, expected {}",
                    point.len(),
                    self.external_source_count
                )));
            }
            for (source_slot, momentum) in point.iter().enumerate() {
                let start =
                    (point_index * self.external_source_count + source_slot) * momentum.len();
                self.external_momenta[start..start + 4].copy_from_slice(momentum);
            }
        }
        Ok(point_count)
    }

    fn flatten_external_tile_view(
        &mut self,
        batch: F64MomentumBatchView<'_>,
    ) -> RusticolResult<usize> {
        let point_count = batch.point_count();
        if point_count > self.effective_point_tile_size() {
            return Err(RusticolError::invalid_argument(
                "recurrence point tile exceeds its persistent workspace",
            ));
        }
        if batch.external_count() != self.external_source_count {
            return Err(RusticolError::invalid_argument(format!(
                "recurrence input has {} external momenta, expected {}",
                batch.external_count(),
                self.external_source_count
            )));
        }
        for point_index in 0..point_count {
            let point = batch.point(point_index);
            for source_slot in 0..self.external_source_count {
                let momentum = point.momentum(source_slot).ok_or_else(|| {
                    RusticolError::integrity(
                        "validated recurrence momentum view is missing an external leg",
                    )
                })?;
                let start = (point_index * self.external_source_count + source_slot) * 4;
                self.external_momenta[start..start + 4].copy_from_slice(&momentum);
            }
        }
        Ok(point_count)
    }

    fn external_tile_input_len(&self, point_count: usize) -> RusticolResult<usize> {
        point_count
            .checked_mul(self.external_source_count)
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| {
                RusticolError::invalid_argument(
                    "recurrence external-momentum tile length overflows",
                )
            })
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_and_contract_contracted_tile(
        &mut self,
        point_count: usize,
        input_len: usize,
        physics: &PhysicsRuntime,
        selected_helicities: Option<&BTreeSet<String>>,
        normalization_factor: f64,
        profiled: bool,
        accumulate: impl FnMut(usize, usize, f64),
    ) -> RusticolResult<Duration> {
        let Self {
            scheduler,
            selectors,
            external_momenta,
            contracted_replay_re,
            contracted_replay_im,
            color_transform_re,
            color_transform_im,
            ..
        } = self;
        let (
            RecurrenceNativeSelectors::ContractedColorUnion {
                contraction,
                destination_physics_helicity,
                replay_routes,
            },
            input,
        ) = (selectors, &external_momenta[..input_len])
        else {
            return Err(RusticolError::integrity(
                "contracted execution requested from a non-contracted recurrence",
            ));
        };
        let point_count_u32 = direct_point_count(point_count)?;
        if replay_routes.is_empty() {
            let direct_output = if profiled {
                scheduler.execute_contracted_tile_from_external(point_count_u32, input)?
            } else {
                scheduler
                    .execute_contracted_tile_from_external_unprofiled(point_count_u32, input)?
            };
            let reduction_started = Instant::now();
            contract_color_tile(
                &direct_output,
                contraction,
                destination_physics_helicity,
                physics,
                selected_helicities,
                normalization_factor,
                color_transform_re,
                color_transform_im,
                accumulate,
            )?;
            return Ok(reduction_started.elapsed());
        }

        let point_stride = scheduler.point_tile_size() as usize;
        let destination_count = contraction.destination_count() as usize;
        let required = destination_count.checked_mul(point_stride).ok_or_else(|| {
            RusticolError::integrity("contracted replay amplitude workspace overflows usize")
        })?;
        if contracted_replay_re.len() < required || contracted_replay_im.len() < required {
            return Err(RusticolError::integrity(
                "contracted replay amplitude workspace is too small",
            ));
        }
        let mut replay_output_copy = Duration::ZERO;
        for route in replay_routes {
            let direct_output = if profiled {
                scheduler.execute_replay_tile_from_external(
                    &route.selector,
                    point_count_u32,
                    input,
                )?
            } else {
                scheduler.execute_replay_tile_from_external_unprofiled(
                    &route.selector,
                    point_count_u32,
                    input,
                )?
            };
            // The scheduler profiles momentum fill, schedule execution, and
            // replay scaling independently.  Start this clock only after it
            // returns: including the scheduler call here would attribute the
            // complete recurrence execution a second time as reduction.
            let replay_output_copy_started = Instant::now();
            for &(source_destination, physical_destination) in &route.destination_copies {
                let source_re = direct_output
                    .destination_re(source_destination)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "contracted replay source amplitude destination is absent",
                        )
                    })?;
                let source_im = direct_output
                    .destination_im(source_destination)
                    .ok_or_else(|| {
                        RusticolError::integrity(
                            "contracted replay source amplitude destination is absent",
                        )
                    })?;
                let start = physical_destination as usize * point_stride;
                contracted_replay_re[start..start + point_count].copy_from_slice(source_re);
                contracted_replay_im[start..start + point_count].copy_from_slice(source_im);
            }
            replay_output_copy += replay_output_copy_started.elapsed();
        }
        let tile = ContractedReplayTile {
            values_re: contracted_replay_re,
            values_im: contracted_replay_im,
            point_count,
            point_stride,
            destination_count,
        };
        let reduction_started = Instant::now();
        contract_color_tile(
            &tile,
            contraction,
            destination_physics_helicity,
            physics,
            selected_helicities,
            normalization_factor,
            color_transform_re,
            color_transform_im,
            accumulate,
        )?;
        Ok(replay_output_copy + reduction_started.elapsed())
    }

    fn validate_public_axis_lengths(&self, physics: &PhysicsRuntime) -> RusticolResult<()> {
        let (color_count, helicity_count) = match &self.selectors {
            RecurrenceNativeSelectors::TopologyReplay {
                replay_selectors, ..
            } => (replay_selectors.len(), physics.manifest.helicities.len()),
            RecurrenceNativeSelectors::AllFlowUnion {
                helicity_selectors_by_physics,
                destination_by_public_flow,
            } => (
                destination_by_public_flow.len(),
                helicity_selectors_by_physics.len(),
            ),
            RecurrenceNativeSelectors::ContractedColorUnion {
                destination_physics_helicity,
                ..
            } => {
                if destination_physics_helicity
                    .iter()
                    .any(|index| *index >= physics.manifest.helicities.len())
                {
                    return Err(RusticolError::integrity(
                        "contracted recurrence maps a destination outside the public helicity axis",
                    ));
                }
                (1, physics.manifest.helicities.len())
            }
        };
        if color_count != physics.manifest.color_components.len()
            || helicity_count != physics.manifest.helicities.len()
        {
            return Err(RusticolError::integrity(
                "recurrence selectors do not cover the public physics axes",
            ));
        }
        Ok(())
    }

    fn validate_public_axes(
        &self,
        physics: &PhysicsRuntime,
        helicity_indices: &[usize],
        color_indices: &[usize],
    ) -> RusticolResult<()> {
        self.validate_public_axis_lengths(physics)?;
        let color_count = physics.manifest.color_components.len();
        let helicity_count = physics.manifest.helicities.len();
        if color_indices.iter().any(|index| *index >= color_count)
            || helicity_indices
                .iter()
                .any(|index| *index >= helicity_count)
        {
            return Err(RusticolError::integrity(
                "recurrence selector mapping is outside the public physics axes",
            ));
        }
        Ok(())
    }
}

fn validate_recurrence_selector_ids(
    physics: &PhysicsRuntime,
    selected_helicities: Option<&BTreeSet<String>>,
    selected_colors: Option<&BTreeSet<String>>,
) -> RusticolResult<()> {
    if let Some(ids) = selected_helicities
        && let Some(id) = ids
            .iter()
            .find(|id| !physics.helicity_index_by_id.contains_key(*id))
    {
        return Err(RusticolError::selector(format!(
            "unknown resolved helicity id {id:?}"
        )));
    }
    if let Some(ids) = selected_colors
        && let Some(id) = ids
            .iter()
            .find(|id| !physics.color_index_by_id.contains_key(*id))
    {
        return Err(RusticolError::selector(format!(
            "unknown resolved color component id {id:?}"
        )));
    }
    Ok(())
}

#[inline(always)]
fn recurrence_helicity_is_selected(
    physics: &PhysicsRuntime,
    selected: Option<&BTreeSet<String>>,
    index: usize,
) -> bool {
    selected.is_none_or(|ids| ids.contains(&physics.manifest.helicities[index].id))
}

#[inline(always)]
fn recurrence_helicity_orbit_weight(
    physics: &PhysicsRuntime,
    selected: Option<&BTreeSet<String>>,
    representative_index: usize,
) -> f64 {
    physics
        .helicity_orbit_members(representative_index)
        .iter()
        .copied()
        .filter(|index| recurrence_helicity_is_selected(physics, selected, *index))
        .map(|index| physics.manifest.helicities[index].coefficient)
        .sum()
}

#[inline(always)]
fn recurrence_color_is_selected(
    physics: &PhysicsRuntime,
    selected: Option<&BTreeSet<String>>,
    index: usize,
) -> bool {
    selected.is_none_or(|ids| ids.contains(physics.manifest.color_components[index].id()))
}

fn direct_point_count(point_count: usize) -> RusticolResult<u32> {
    u32::try_from(point_count).map_err(|_| {
        RusticolError::invalid_argument("recurrence point tile exceeds the native u32 ABI")
    })
}

fn reject_contracted_color_selector(
    selected_colors: Option<&BTreeSet<String>>,
) -> RusticolResult<()> {
    if selected_colors.is_some() {
        return Err(RusticolError::selector(
            "contracted NLC/full recurrence does not expose a color-flow selector",
        ));
    }
    Ok(())
}

fn contracted_destination_helicity_map(
    contraction: &RecurrenceColorContraction,
    direct_helicity_to_physics: &[usize],
) -> RusticolResult<Vec<usize>> {
    let destination_count = contraction.destination_count() as usize;
    if contraction.component_count() as usize != direct_helicity_to_physics.len()
        || contraction.destination_by_group().len() != destination_count
        || contraction.sector_by_group().len() != destination_count
        || contraction.component_by_group().len() != destination_count
    {
        return Err(RusticolError::integrity(
            "contracted color dimensions disagree with the public helicity map",
        ));
    }

    let mut destination_physics_helicity = vec![usize::MAX; destination_count];
    for (group_id, destination_id) in contraction
        .destination_by_group()
        .iter()
        .copied()
        .enumerate()
    {
        let expected_direct_helicity = contraction.component_by_group()[group_id] as usize;
        let physics_helicity = *direct_helicity_to_physics
            .get(expected_direct_helicity)
            .ok_or_else(|| {
                RusticolError::integrity("contracted destination has no public helicity mapping")
            })?;
        let slot = destination_physics_helicity
            .get_mut(destination_id as usize)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "contracted destination mapping exceeds its authenticated range",
                )
            })?;
        if *slot != usize::MAX {
            return Err(RusticolError::integrity(
                "contracted color map repeats an amplitude destination",
            ));
        }
        *slot = physics_helicity;
    }
    if destination_physics_helicity.contains(&usize::MAX) {
        return Err(RusticolError::integrity(
            "contracted color map does not cover every amplitude destination",
        ));
    }
    for entry in contraction.canonical_logical_entries() {
        if contraction.component_by_group()[entry.left_group_id as usize]
            != contraction.component_by_group()[entry.right_group_id as usize]
        {
            return Err(RusticolError::integrity(
                "contracted color entry mixes different helicity components",
            ));
        }
    }
    Ok(destination_physics_helicity)
}

fn contracted_replay_routes(
    scheduler: &DirectRecurrenceExecutionRuntime,
    contraction: &RecurrenceColorContraction,
) -> RusticolResult<Vec<ContractedReplayRoute>> {
    let plan = scheduler.plan();
    if plan.strategy() != RecurrenceStrategy::ContractedColorUnion {
        return Err(RusticolError::integrity(
            "contracted replay routes require a contracted-color plan",
        ));
    }
    if plan.replay_targets().is_empty() {
        if contraction.destination_count() as usize != plan.amplitude_destinations().len() {
            return Err(RusticolError::integrity(
                "non-replayed contracted color destinations disagree with the Direct plan",
            ));
        }
        return Ok(Vec::new());
    }
    if contraction.sector_count() != plan.physical_sector_count()
        || contraction.component_count() as usize != plan.resolved_helicities().len()
    {
        return Err(RusticolError::integrity(
            "contracted replay domain disagrees with the Direct plan",
        ));
    }

    let destination_count = contraction.destination_count() as usize;
    let component_count = contraction.component_count() as usize;
    let mut covered = vec![false; destination_count];
    let mut routes = Vec::with_capacity(plan.replay_targets().len());
    for target in plan.replay_targets() {
        let selector = scheduler.prepare_replay_selector(target.public_flow_id)?;
        let target_sector = selector.public_flow_id() as usize;
        let mut destination_copies = Vec::with_capacity(component_count);
        for destination in plan
            .amplitude_destinations()
            .iter()
            .filter(|destination| destination.target_sector_id == selector.representative_flow_id())
        {
            let direct_helicity = destination.target_helicity_id_or_sentinel;
            if direct_helicity == DIRECT_NONE_U32 {
                return Err(RusticolError::integrity(
                    "contracted replay destination lacks a resolved helicity",
                ));
            }
            let mapped_helicity = selector
                .helicity_map()
                .get(direct_helicity as usize)
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity("contracted replay destination helicity is not mapped")
                })? as usize;
            let physical_destination = target_sector
                .checked_mul(component_count)
                .and_then(|offset| offset.checked_add(mapped_helicity))
                .ok_or_else(|| {
                    RusticolError::artifact(
                        "contracted replay physical destination overflows usize",
                    )
                })?;
            if physical_destination >= destination_count
                || covered[physical_destination]
                || contraction.sector_by_group()[physical_destination] as usize != target_sector
                || contraction.component_by_group()[physical_destination] as usize
                    != mapped_helicity
                || contraction.destination_by_group()[physical_destination] as usize
                    != physical_destination
            {
                return Err(RusticolError::integrity(
                    "contracted replay route is not a dense physical color/helicity bijection",
                ));
            }
            covered[physical_destination] = true;
            destination_copies.push((
                destination.id,
                u32::try_from(physical_destination).map_err(|_| {
                    RusticolError::artifact("contracted replay physical destination exceeds u32")
                })?,
            ));
        }
        if destination_copies.len() != component_count {
            return Err(RusticolError::integrity(
                "contracted replay representative lacks complete helicity coverage",
            ));
        }
        routes.push(ContractedReplayRoute {
            selector,
            destination_copies,
        });
    }
    if covered.contains(&false) {
        return Err(RusticolError::integrity(
            "contracted replay routes do not cover every physical color/helicity destination",
        ));
    }
    Ok(routes)
}

trait ContractedAmplitudeTile {
    fn point_count(&self) -> usize;
    fn destination_re(&self, destination_id: u32) -> Option<&[f64]>;
    fn destination_im(&self, destination_id: u32) -> Option<&[f64]>;
}

impl ContractedAmplitudeTile for DirectRecurrenceTileOutput<'_> {
    fn point_count(&self) -> usize {
        DirectRecurrenceTileOutput::point_count(self) as usize
    }

    fn destination_re(&self, destination_id: u32) -> Option<&[f64]> {
        DirectRecurrenceTileOutput::destination_re(self, destination_id)
    }

    fn destination_im(&self, destination_id: u32) -> Option<&[f64]> {
        DirectRecurrenceTileOutput::destination_im(self, destination_id)
    }
}

struct ContractedReplayTile<'a> {
    values_re: &'a [f64],
    values_im: &'a [f64],
    point_count: usize,
    point_stride: usize,
    destination_count: usize,
}

impl ContractedAmplitudeTile for ContractedReplayTile<'_> {
    fn point_count(&self) -> usize {
        self.point_count
    }

    fn destination_re(&self, destination_id: u32) -> Option<&[f64]> {
        self.destination(self.values_re, destination_id)
    }

    fn destination_im(&self, destination_id: u32) -> Option<&[f64]> {
        self.destination(self.values_im, destination_id)
    }
}

impl ContractedReplayTile<'_> {
    fn destination<'a>(&self, values: &'a [f64], destination_id: u32) -> Option<&'a [f64]> {
        let destination = destination_id as usize;
        if destination >= self.destination_count {
            return None;
        }
        let start = destination.checked_mul(self.point_stride)?;
        values.get(start..start + self.point_count)
    }
}

// These arguments are the authenticated tile views and selector/reduction
// contract; grouping them would obscure ownership without reducing call state.
#[allow(clippy::too_many_arguments)]
fn contract_color_tile<T: ContractedAmplitudeTile>(
    output: &T,
    contraction: &RecurrenceColorContraction,
    destination_physics_helicity: &[usize],
    physics: &PhysicsRuntime,
    selected_helicities: Option<&BTreeSet<String>>,
    normalization_factor: f64,
    color_transform_re: &mut [f64],
    color_transform_im: &mut [f64],
    mut accumulate: impl FnMut(usize, usize, f64),
) -> RusticolResult<()> {
    let point_count = output.point_count();
    if let Some(factorization) = contraction.runtime_factorization() {
        return contract_factorized_color_tile(
            output,
            contraction,
            factorization,
            destination_physics_helicity,
            physics,
            selected_helicities,
            normalization_factor,
            color_transform_re,
            color_transform_im,
            &mut accumulate,
        );
    }
    for entry in contraction.runtime_entries() {
        let left_helicity = *destination_physics_helicity
            .get(entry.left_destination_id as usize)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "contracted left destination has no public helicity mapping",
                )
            })?;
        let right_helicity = *destination_physics_helicity
            .get(entry.right_destination_id as usize)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "contracted right destination has no public helicity mapping",
                )
            })?;
        if left_helicity != right_helicity {
            return Err(RusticolError::integrity(
                "contracted color entry mixes public helicities",
            ));
        }
        let helicity = physics
            .manifest
            .helicities
            .get(left_helicity)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "contracted color entry references an absent public helicity",
                )
            })?;
        if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0 {
            continue;
        }
        if recurrence_helicity_orbit_weight(physics, selected_helicities, left_helicity) == 0.0 {
            continue;
        }
        let left_re = output
            .destination_re(entry.left_destination_id)
            .ok_or_else(|| {
                RusticolError::integrity("contracted left amplitude destination is absent")
            })?;
        let left_im = output
            .destination_im(entry.left_destination_id)
            .ok_or_else(|| {
                RusticolError::integrity("contracted left amplitude destination is absent")
            })?;
        let right_re = output
            .destination_re(entry.right_destination_id)
            .ok_or_else(|| {
                RusticolError::integrity("contracted right amplitude destination is absent")
            })?;
        let right_im = output
            .destination_im(entry.right_destination_id)
            .ok_or_else(|| {
                RusticolError::integrity("contracted right amplitude destination is absent")
            })?;
        for point in 0..point_count {
            let product_re =
                left_re[point].mul_add(right_re[point], left_im[point] * right_im[point]);
            let product_im =
                left_im[point].mul_add(right_re[point], -left_re[point] * right_im[point]);
            let contracted = entry
                .coefficient_re
                .mul_add(product_re, -entry.coefficient_im * product_im);
            for physical_helicity in physics.helicity_orbit_members(left_helicity) {
                if !recurrence_helicity_is_selected(
                    physics,
                    selected_helicities,
                    *physical_helicity,
                ) {
                    continue;
                }
                let scale = physics.manifest.helicities[*physical_helicity].coefficient
                    * normalization_factor;
                accumulate(point, *physical_helicity, scale * contracted);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn contract_factorized_color_tile<T: ContractedAmplitudeTile>(
    output: &T,
    contraction: &RecurrenceColorContraction,
    factorization: &crate::recurrence::RuntimeFactorizedColorContraction,
    destination_physics_helicity: &[usize],
    physics: &PhysicsRuntime,
    selected_helicities: Option<&BTreeSet<String>>,
    normalization_factor: f64,
    color_transform_re: &mut [f64],
    color_transform_im: &mut [f64],
    accumulate: &mut impl FnMut(usize, usize, f64),
) -> RusticolResult<()> {
    let point_count = output.point_count();
    let local_group_count = contraction.local_group_count() as usize;
    let scratch_len = local_group_count.checked_mul(point_count).ok_or_else(|| {
        RusticolError::integrity("factorized recurrence color scratch length overflows")
    })?;
    if color_transform_re.len() < scratch_len || color_transform_im.len() < scratch_len {
        return Err(RusticolError::integrity(
            "factorized recurrence color workspace is smaller than the point tile",
        ));
    }
    let transform_re = &mut color_transform_re[..scratch_len];
    let transform_im = &mut color_transform_im[..scratch_len];
    let component_count = contraction.component_count() as usize;

    for component_index in 0..component_count {
        let representative_destination = contraction
            .ordered_destination_id(0, component_index)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "factorized recurrence component has no amplitude destination",
                )
            })?;
        let helicity_index = *destination_physics_helicity
            .get(representative_destination as usize)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "factorized recurrence destination has no public helicity mapping",
                )
            })?;
        let helicity = physics
            .manifest
            .helicities
            .get(helicity_index)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "factorized recurrence color component references an absent helicity",
                )
            })?;
        if !helicity.computed || helicity.structural_zero || helicity.coefficient == 0.0 {
            continue;
        }
        if recurrence_helicity_orbit_weight(physics, selected_helicities, helicity_index) == 0.0 {
            continue;
        }

        for local_group_index in 0..local_group_count {
            let destination_id = contraction
                .ordered_destination_id(local_group_index, component_index)
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "factorized recurrence local group has no amplitude destination",
                    )
                })?;
            if destination_physics_helicity
                .get(destination_id as usize)
                .copied()
                != Some(helicity_index)
            {
                return Err(RusticolError::integrity(
                    "factorized recurrence local groups mix public helicities",
                ));
            }
            let source_re = output.destination_re(destination_id).ok_or_else(|| {
                RusticolError::integrity(
                    "factorized recurrence real amplitude destination is absent",
                )
            })?;
            let source_im = output.destination_im(destination_id).ok_or_else(|| {
                RusticolError::integrity(
                    "factorized recurrence imaginary amplitude destination is absent",
                )
            })?;
            let start = local_group_index * point_count;
            transform_re[start..start + point_count].copy_from_slice(source_re);
            transform_im[start..start + point_count].copy_from_slice(source_im);
        }

        if factorization.subgroup_order() == 4 {
            let amplitude_scale = factorization.amplitude_scale();
            for coset in factorization.cosets() {
                let [g0, g1, g2, g3] = <[u32; 4]>::try_from(coset.as_slice()).map_err(|_| {
                    RusticolError::integrity(
                        "rank-two recurrence color factorization has a malformed coset",
                    )
                })?;
                let starts = [
                    g0 as usize * point_count,
                    g1 as usize * point_count,
                    g2 as usize * point_count,
                    g3 as usize * point_count,
                ];
                for point in 0..point_count {
                    let re0 = transform_re[starts[0] + point];
                    let re1 = transform_re[starts[1] + point];
                    let re2 = transform_re[starts[2] + point];
                    let re3 = transform_re[starts[3] + point];
                    let im0 = transform_im[starts[0] + point];
                    let im1 = transform_im[starts[1] + point];
                    let im2 = transform_im[starts[2] + point];
                    let im3 = transform_im[starts[3] + point];
                    let re01 = re0 + re1;
                    let re23 = re2 + re3;
                    let re01_difference = re0 - re1;
                    let re23_difference = re2 - re3;
                    let im01 = im0 + im1;
                    let im23 = im2 + im3;
                    let im01_difference = im0 - im1;
                    let im23_difference = im2 - im3;
                    transform_re[starts[0] + point] = (re01 + re23) * amplitude_scale;
                    transform_re[starts[1] + point] =
                        (re01_difference + re23_difference) * amplitude_scale;
                    transform_re[starts[2] + point] = (re01 - re23) * amplitude_scale;
                    transform_re[starts[3] + point] =
                        (re01_difference - re23_difference) * amplitude_scale;
                    transform_im[starts[0] + point] = (im01 + im23) * amplitude_scale;
                    transform_im[starts[1] + point] =
                        (im01_difference + im23_difference) * amplitude_scale;
                    transform_im[starts[2] + point] = (im01 - im23) * amplitude_scale;
                    transform_im[starts[3] + point] =
                        (im01_difference - im23_difference) * amplitude_scale;
                }
            }
        } else {
            for coset in factorization.cosets() {
                let mut stride = 1usize;
                while stride < factorization.subgroup_order() {
                    for start in (0..factorization.subgroup_order()).step_by(stride * 2) {
                        for offset in 0..stride {
                            let left_group = coset[start + offset] as usize;
                            let right_group = coset[start + stride + offset] as usize;
                            let left_start = left_group * point_count;
                            let right_start = right_group * point_count;
                            for point in 0..point_count {
                                let left_re = transform_re[left_start + point];
                                let right_re = transform_re[right_start + point];
                                let left_im = transform_im[left_start + point];
                                let right_im = transform_im[right_start + point];
                                transform_re[left_start + point] = left_re + right_re;
                                transform_re[right_start + point] = left_re - right_re;
                                transform_im[left_start + point] = left_im + right_im;
                                transform_im[right_start + point] = left_im - right_im;
                            }
                        }
                    }
                    stride *= 2;
                }
            }
        }

        for entry in factorization.entries() {
            let left_start = entry.left_group_index as usize * point_count;
            let right_start = entry.right_group_index as usize * point_count;
            for point in 0..point_count {
                let product_re = transform_re[left_start + point].mul_add(
                    transform_re[right_start + point],
                    transform_im[left_start + point] * transform_im[right_start + point],
                );
                let product_im = transform_im[left_start + point].mul_add(
                    transform_re[right_start + point],
                    -transform_re[left_start + point] * transform_im[right_start + point],
                );
                let contracted = entry
                    .coefficient_re
                    .mul_add(product_re, -entry.coefficient_im * product_im);
                for physical_helicity in physics.helicity_orbit_members(helicity_index) {
                    if !recurrence_helicity_is_selected(
                        physics,
                        selected_helicities,
                        *physical_helicity,
                    ) {
                        continue;
                    }
                    let scale = physics.manifest.helicities[*physical_helicity].coefficient
                        * normalization_factor;
                    accumulate(point, *physical_helicity, scale * contracted);
                }
            }
        }
    }
    Ok(())
}

fn union_destination_ids(
    plan: &DirectRecurrencePlan,
    public_flow_ids: &[u32],
) -> RusticolResult<Vec<u32>> {
    if plan.strategy() != RecurrenceStrategy::AllFlowUnion {
        return Err(RusticolError::integrity(
            "union destination mapping requires an all-flow-union plan",
        ));
    }
    let mut destination_by_sector = BTreeMap::new();
    for destination in plan.amplitude_destinations() {
        if destination.target_helicity_id_or_sentinel != DIRECT_NONE_U32 {
            return Err(RusticolError::integrity(
                "all-flow-union amplitude destination fixes a numerical helicity",
            ));
        }
        if destination_by_sector
            .insert(destination.target_sector_id, destination.id)
            .is_some()
        {
            return Err(RusticolError::integrity(
                "all-flow-union repeats a physical-flow destination",
            ));
        }
    }
    let result = public_flow_ids
        .iter()
        .map(|sector_id| {
            destination_by_sector
                .get(sector_id)
                .copied()
                .ok_or_else(|| {
                    RusticolError::integrity(
                        "all-flow-union public flow has no amplitude destination",
                    )
                })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    if result.iter().copied().collect::<BTreeSet<_>>()
        != destination_by_sector
            .values()
            .copied()
            .collect::<BTreeSet<_>>()
    {
        return Err(RusticolError::integrity(
            "all-flow-union amplitude destinations do not match the public flow axis",
        ));
    }
    Ok(result)
}

fn validate_replay_destination_helicity_mappings(
    plan: &DirectRecurrencePlan,
    replay_selectors: &[DirectReplaySelectorPlan],
    direct_helicity_to_physics: &[usize],
) -> RusticolResult<()> {
    for destination in plan.amplitude_destinations() {
        replay_target_helicity_index(
            destination.target_helicity_id_or_sentinel,
            plan.resolved_helicities().len(),
        )?;
    }

    for selector in replay_selectors {
        if selector.helicity_map().len() != plan.resolved_helicities().len() {
            return Err(RusticolError::integrity(format!(
                "recurrence replay flow {} has incomplete helicity coverage",
                selector.public_flow_id()
            )));
        }
        for mapped_direct_id in selector.helicity_map() {
            replay_mapped_direct_physics_helicity(direct_helicity_to_physics, *mapped_direct_id)?;
        }
    }
    Ok(())
}

fn replay_output_destination_physics_helicity(
    output: &DirectRecurrenceTileOutput<'_>,
    selector: &DirectReplaySelectorPlan,
    direct_helicity_to_physics: &[usize],
    destination_id: u32,
) -> RusticolResult<usize> {
    let target_helicity_id_or_sentinel = output
        .destination_target_helicity_id_or_sentinel(destination_id)
        .ok_or_else(|| {
            RusticolError::integrity("recurrence destination-helicity mapping is incomplete")
        })?;
    replay_destination_physics_helicity(
        selector.helicity_map(),
        direct_helicity_to_physics,
        target_helicity_id_or_sentinel,
    )
}

fn replay_destination_physics_helicity(
    replay_helicity_map: &[u32],
    direct_helicity_to_physics: &[usize],
    target_helicity_id_or_sentinel: u32,
) -> RusticolResult<usize> {
    let target_helicity_index =
        replay_target_helicity_index(target_helicity_id_or_sentinel, replay_helicity_map.len())?;
    let mapped_direct_id = replay_helicity_map[target_helicity_index];
    replay_mapped_direct_physics_helicity(direct_helicity_to_physics, mapped_direct_id)
}

fn replay_target_helicity_index(
    target_helicity_id_or_sentinel: u32,
    helicity_count: usize,
) -> RusticolResult<usize> {
    if target_helicity_id_or_sentinel == DIRECT_NONE_U32 {
        return Err(RusticolError::integrity(
            "topology-replay amplitude destination lacks a resolved helicity",
        ));
    }
    let target_helicity_index = target_helicity_id_or_sentinel as usize;
    if target_helicity_index >= helicity_count {
        return Err(RusticolError::integrity(
            "recurrence amplitude destination helicity is not replay-mapped",
        ));
    }
    Ok(target_helicity_index)
}

fn replay_mapped_direct_physics_helicity(
    direct_helicity_to_physics: &[usize],
    mapped_direct_id: u32,
) -> RusticolResult<usize> {
    direct_helicity_to_physics
        .get(mapped_direct_id as usize)
        .copied()
        .ok_or_else(|| {
            RusticolError::integrity("recurrence replay helicity has no public physics mapping")
        })
}

#[cfg(test)]
mod replay_destination_helicity_tests {
    use super::*;

    #[test]
    fn compact_lc_selector_reduction_view_matches_topology_replay_semantics() {
        let helicities = vec![
            crate::Helicity {
                id: "h0".into(),
                index: 0,
                values: vec![1, -1],
                computed: true,
                structural_zero: false,
                representative_id: "h0".into(),
                coefficient: 0.25,
            },
            crate::Helicity {
                id: "h1".into(),
                index: 1,
                values: vec![-1, 1],
                computed: false,
                structural_zero: false,
                representative_id: "h0".into(),
                coefficient: 0.75,
            },
        ];
        let colors = vec![
            crate::ColorComponent::LcFlow(crate::LcColorFlow {
                id: "c0".into(),
                index: 0,
                word: vec![0, 1],
                computed: true,
                representative_id: "c0".into(),
                coefficient: 2.0,
            }),
            crate::ColorComponent::LcFlow(crate::LcColorFlow {
                id: "c1".into(),
                index: 1,
                word: vec![1, 0],
                computed: false,
                representative_id: "c0".into(),
                coefficient: 3.0,
            }),
        ];
        let helicity_ids = BTreeMap::from([("h0".into(), 0), ("h1".into(), 1)]);
        let color_ids = BTreeMap::from([("c0".into(), 0), ("c1".into(), 1)]);
        let orbits = vec![vec![0, 1], Vec::new()];
        let view = LcSelectorReductionView::from_parts(
            &helicities,
            &colors,
            &helicity_ids,
            &color_ids,
            &orbits,
        );
        let selected_helicities = BTreeSet::from(["h1".to_string()]);
        let selected_colors = BTreeSet::from(["c0".to_string()]);

        view.validate_selector_ids(Some(&selected_helicities), Some(&selected_colors))
            .unwrap();
        assert_eq!(
            view.selected_helicity_indices(Some(&selected_helicities))
                .unwrap(),
            [1]
        );
        assert_eq!(
            view.selected_color_indices(Some(&selected_colors)).unwrap(),
            [0]
        );
        assert_eq!(view.helicity_orbit_weight(None, 0), 1.0);
        assert_eq!(
            view.helicity_orbit_weight(Some(&selected_helicities), 0),
            0.75
        );
        assert!(view.color_is_computed(0));
        assert!(!view.color_is_computed(1));
        assert!(view.color_is_selected(Some(&selected_colors), 0));
        assert!(
            view.validate_selector_ids(Some(&BTreeSet::from(["missing".into()])), None)
                .is_err()
        );
    }

    #[test]
    fn composes_distinct_replay_flow_permutations_without_destination_tables() {
        let direct_helicity_to_physics = [2, 0, 1];
        let flow_zero = [0, 1, 2];
        let flow_one = [2, 0, 1];

        for (replay_helicity_map, expected) in
            [(&flow_zero[..], [2, 0, 1]), (&flow_one[..], [1, 2, 0])]
        {
            let actual = (0..3)
                .map(|target_helicity_id| {
                    replay_destination_physics_helicity(
                        replay_helicity_map,
                        &direct_helicity_to_physics,
                        target_helicity_id,
                    )
                    .expect("compose authenticated replay and physics mappings")
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn rejects_absent_or_out_of_range_replay_helicity_mappings() {
        let missing = replay_destination_physics_helicity(&[0], &[0], DIRECT_NONE_U32)
            .expect_err("sentinel destination helicity must fail closed");
        assert_eq!(
            missing.message(),
            "topology-replay amplitude destination lacks a resolved helicity"
        );

        let incomplete = replay_destination_physics_helicity(&[0], &[0], 1)
            .expect_err("incomplete replay permutation must fail closed");
        assert_eq!(
            incomplete.message(),
            "recurrence amplitude destination helicity is not replay-mapped"
        );

        let out_of_range = replay_destination_physics_helicity(&[1], &[0], 0)
            .expect_err("out-of-range direct helicity must fail closed");
        assert_eq!(
            out_of_range.message(),
            "recurrence replay helicity has no public physics mapping"
        );
    }
}

#[derive(Clone, Copy)]
struct DirectProfileSnapshot {
    phases: DirectRuntimePhaseTimings,
    roles: DirectExecutionRoleTimings,
    execution: DirectExecutionCounters,
    traffic: DirectArenaTrafficCounters,
    activity: DirectRuntimeActivityCounters,
    internal_scratch_bytes: u64,
    internal_broadcast_bytes: u64,
}

impl DirectProfileSnapshot {
    fn capture(
        runtime: &DirectRecurrenceExecutionRuntime,
        owners: &NativeRecurrenceDirectExecutorOwners,
    ) -> Self {
        let (internal_scratch_bytes, internal_broadcast_bytes) = owners.internal_traffic_bytes();
        Self {
            phases: runtime.phase_timings(),
            roles: runtime.role_timings(),
            execution: runtime.counters(),
            traffic: runtime.traffic_counters(),
            activity: runtime.activity_counters(),
            internal_scratch_bytes,
            internal_broadcast_bytes,
        }
    }
}

fn direct_profile(
    total: Duration,
    parameter_setup: Duration,
    external_momentum_flatten: Duration,
    reduction: Duration,
    before: DirectProfileSnapshot,
    after: DirectProfileSnapshot,
) -> RuntimeProfile {
    let momentum_fill = after
        .phases
        .momentum_fill
        .saturating_sub(before.phases.momentum_fill);
    let union_source_fill = after
        .phases
        .union_source_fill
        .saturating_sub(before.phases.union_source_fill);
    let direct_execution = after
        .phases
        .direct_execution
        .saturating_sub(before.phases.direct_execution);
    let replay_output_mapping = after
        .phases
        .replay_output_mapping
        .saturating_sub(before.phases.replay_output_mapping);
    let source_kernel = after.roles.source.saturating_sub(before.roles.source);
    let contribution_kernel = after
        .roles
        .contribution
        .saturating_sub(before.roles.contribution);
    let finalization = after
        .roles
        .finalization
        .saturating_sub(before.roles.finalization);
    let closure = after.roles.closure.saturating_sub(before.roles.closure);
    let execution = DirectExecutionCounters {
        source_calls: after
            .execution
            .source_calls
            .saturating_sub(before.execution.source_calls),
        source_rows: after
            .execution
            .source_rows
            .saturating_sub(before.execution.source_rows),
        contribution_calls: after
            .execution
            .contribution_calls
            .saturating_sub(before.execution.contribution_calls),
        contribution_rows: after
            .execution
            .contribution_rows
            .saturating_sub(before.execution.contribution_rows),
        finalization_calls: after
            .execution
            .finalization_calls
            .saturating_sub(before.execution.finalization_calls),
        finalization_rows: after
            .execution
            .finalization_rows
            .saturating_sub(before.execution.finalization_rows),
        closure_calls: after
            .execution
            .closure_calls
            .saturating_sub(before.execution.closure_calls),
        closure_rows: after
            .execution
            .closure_rows
            .saturating_sub(before.execution.closure_rows),
        packed_input_bytes: after
            .execution
            .packed_input_bytes
            .saturating_sub(before.execution.packed_input_bytes),
        packed_output_bytes: after
            .execution
            .packed_output_bytes
            .saturating_sub(before.execution.packed_output_bytes),
        scatter_bytes: after
            .execution
            .scatter_bytes
            .saturating_sub(before.execution.scatter_bytes),
    };
    let traffic = DirectArenaTrafficCounters {
        calls: after.traffic.calls.saturating_sub(before.traffic.calls),
        rows: after.traffic.rows.saturating_sub(before.traffic.rows),
        points: after.traffic.points.saturating_sub(before.traffic.points),
        packet_input_bytes: after
            .traffic
            .packet_input_bytes
            .saturating_sub(before.traffic.packet_input_bytes),
        packet_output_bytes: after
            .traffic
            .packet_output_bytes
            .saturating_sub(before.traffic.packet_output_bytes),
        gather_bytes: after
            .traffic
            .gather_bytes
            .saturating_sub(before.traffic.gather_bytes),
        scatter_bytes: after
            .traffic
            .scatter_bytes
            .saturating_sub(before.traffic.scatter_bytes),
        remap_bytes: after
            .traffic
            .remap_bytes
            .saturating_sub(before.traffic.remap_bytes),
    };
    let internal_scratch_bytes = after
        .internal_scratch_bytes
        .saturating_sub(before.internal_scratch_bytes);
    let internal_broadcast_bytes = after
        .internal_broadcast_bytes
        .saturating_sub(before.internal_broadcast_bytes);
    let activity = DirectRuntimeActivityCounters {
        momentum_fill_calls: after
            .activity
            .momentum_fill_calls
            .saturating_sub(before.activity.momentum_fill_calls),
        momentum_forms_filled: after
            .activity
            .momentum_forms_filled
            .saturating_sub(before.activity.momentum_forms_filled),
        momentum_terms_filled: after
            .activity
            .momentum_terms_filled
            .saturating_sub(before.activity.momentum_terms_filled),
        momentum_scalar_values_filled: after
            .activity
            .momentum_scalar_values_filled
            .saturating_sub(before.activity.momentum_scalar_values_filled),
        schedule_executions: after
            .activity
            .schedule_executions
            .saturating_sub(before.activity.schedule_executions),
        replay_schedule_executions: after
            .activity
            .replay_schedule_executions
            .saturating_sub(before.activity.replay_schedule_executions),
        replay_output_values_scaled: after
            .activity
            .replay_output_values_scaled
            .saturating_sub(before.activity.replay_output_values_scaled),
        union_source_dispatch_calls: after
            .activity
            .union_source_dispatch_calls
            .saturating_sub(before.activity.union_source_dispatch_calls),
        union_source_rows: after
            .activity
            .union_source_rows
            .saturating_sub(before.activity.union_source_rows),
        union_schedule_executions: after
            .activity
            .union_schedule_executions
            .saturating_sub(before.activity.union_schedule_executions),
    };
    RuntimeProfile {
        momentum_setup_s: profile_duration_seconds(external_momentum_flatten + momentum_fill),
        momentum_input_setup_s: profile_duration_seconds(external_momentum_flatten),
        model_parameter_setup_s: profile_duration_seconds(parameter_setup),
        stage_evaluator_call_s: profile_duration_seconds(direct_execution),
        stage_evaluator_s: profile_duration_seconds(direct_execution),
        recurrence_momentum_fill_s: profile_duration_seconds(momentum_fill),
        recurrence_union_source_fill_s: profile_duration_seconds(union_source_fill),
        recurrence_schedule_s: profile_duration_seconds(direct_execution),
        recurrence_source_kernel_s: profile_duration_seconds(source_kernel),
        recurrence_contribution_kernel_s: profile_duration_seconds(contribution_kernel),
        recurrence_finalization_s: profile_duration_seconds(finalization),
        recurrence_closure_s: profile_duration_seconds(closure),
        recurrence_replay_output_mapping_s: profile_duration_seconds(replay_output_mapping),
        recurrence_momentum_scalar_value_count: activity.momentum_scalar_values_filled,
        recurrence_schedule_execution_count: activity.schedule_executions,
        recurrence_replay_schedule_execution_count: activity.replay_schedule_executions,
        recurrence_union_schedule_execution_count: activity.union_schedule_executions,
        recurrence_union_source_row_count: activity.union_source_rows,
        recurrence_replay_output_value_count: activity.replay_output_values_scaled,
        recurrence_source_call_count: execution.source_calls,
        recurrence_source_row_count: execution.source_rows,
        recurrence_contribution_call_count: execution.contribution_calls,
        recurrence_contribution_row_count: execution.contribution_rows,
        recurrence_finalization_call_count: execution.finalization_calls,
        recurrence_finalization_row_count: execution.finalization_rows,
        recurrence_closure_call_count: execution.closure_calls,
        recurrence_closure_row_count: execution.closure_rows,
        recurrence_direct_packed_input_bytes: execution.packed_input_bytes,
        recurrence_direct_packed_output_bytes: execution.packed_output_bytes,
        recurrence_direct_scatter_bytes: execution.scatter_bytes,
        recurrence_direct_packet_input_bytes: traffic.packet_input_bytes,
        recurrence_direct_packet_output_bytes: traffic.packet_output_bytes,
        recurrence_direct_gather_bytes: traffic.gather_bytes,
        recurrence_direct_traffic_scatter_bytes: traffic.scatter_bytes,
        recurrence_direct_remap_bytes: traffic.remap_bytes,
        recurrence_internal_scratch_bytes: internal_scratch_bytes,
        recurrence_internal_broadcast_bytes: internal_broadcast_bytes,
        reduction_s: profile_duration_seconds(reduction),
        total_s: profile_duration_seconds(total),
        ..RuntimeProfile::default()
    }
}
