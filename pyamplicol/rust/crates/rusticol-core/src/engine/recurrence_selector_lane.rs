// SPDX-License-Identifier: 0BSD

//! Selector-only full colour: the helicity-selector companion without the main
//! contracted-colour schedule, evaluated one helicity per point.

use super::*;

/// Error for an API that needs the omitted main contracted-colour schedule.
pub(in crate::engine) fn selector_only_unsupported(operation: &str) -> RusticolError {
    RusticolError::compatibility(format!(
        "{operation} needs the main contracted-colour schedule, which this selector-only \
         full-colour artifact omits (generation.selector_only_full_colour); evaluate one \
         helicity per point, or regenerate without the option"
    ))
}

/// Process lane of a selector-only full-colour artifact. It owns the
/// parameter planes and point-tile scratch the default lane lends its companion.
pub(in crate::engine) struct RecurrenceSelectorOnlyRuntime {
    companion: RecurrencePersistedHelicitySelectorRuntime,
    contraction: RecurrenceColorContraction,
    backend_name: String,
    parameter_defaults: Vec<crate::EagerComplex64>,
    parameter_projection: Vec<PreparedParameterProjectionEntry>,
    parameters_re: Vec<f64>,
    parameters_im: Vec<f64>,
    external_source_count: usize,
    effective_point_tile_size: usize,
    external_momenta: Vec<f64>,
    color_transform_re: Vec<f64>,
    color_transform_im: Vec<f64>,
    symmetric_group_color_workspace: Option<RuntimeSymmetricGroupColorWorkspace>,
    order_catalog: AmplitudeOrderCatalog,
}

impl RecurrenceSelectorOnlyRuntime {
    #[cfg_attr(target_vendor = "apple", unsafe(link_section = crate::rcl_load_section!()))]
    #[cfg_attr(target_vendor = "apple", inline(never))]
    pub(in crate::engine) fn new(
        mut companion: RecurrencePersistedHelicitySelectorRuntime,
        contraction: RecurrenceColorContraction,
        backend_name: String,
        parameter_defaults: Vec<crate::EagerComplex64>,
        parameter_projection: Vec<PreparedParameterProjectionEntry>,
        physics_helicity_count: usize,
    ) -> RusticolResult<Self> {
        if contraction.component_count() != 1 {
            return Err(RusticolError::integrity(
                "selector-only color contraction must have exactly one component",
            ));
        }
        if physics_helicity_count < 2 {
            return Err(RusticolError::integrity(
                "selector-only full colour requires more than one physics helicity",
            ));
        }
        let plan = &companion.plan;
        let parameter_count = usize::try_from(plan.parameter_value_count())
            .map_err(|_| RusticolError::artifact("recurrence parameter count exceeds usize"))?;
        if parameter_defaults.len() != parameter_count {
            return Err(RusticolError::integrity(format!(
                "selector-only prepared defaults have length {}, expected {parameter_count}",
                parameter_defaults.len()
            )));
        }
        if parameter_projection
            .iter()
            .any(|entry| entry.prepared_slot >= parameter_count || entry.component > 1)
        {
            return Err(RusticolError::artifact(
                "selector-only parameter projection is outside the prepared layout",
            ));
        }
        let order_catalog = AmplitudeOrderCatalog::from_plan(plan.amplitude_orders())?;
        let external_source_count = usize::try_from(plan.external_source_count())
            .map_err(|_| RusticolError::artifact("recurrence source count exceeds usize"))?;
        let point_tile_capacity = companion_point_tile_size(plan)?;
        let mut effective_point_tile_size = point_tile_capacity;
        let mut color_transform_len = 0usize;
        let mut symmetric_group_color_workspace = None;
        match contraction.runtime_reducer() {
            Some(RuntimeColorContractionReducer::Walsh(_)) => {
                color_transform_len = usize::try_from(contraction.local_group_count())
                    .ok()
                    .and_then(|groups| groups.checked_mul(point_tile_capacity))
                    .ok_or_else(|| {
                        RusticolError::artifact(
                            "selector-only factorized color workspace overflows usize",
                        )
                    })?;
            }
            Some(RuntimeColorContractionReducer::SymmetricGroupFourier(reducer)) => {
                (effective_point_tile_size, symmetric_group_color_workspace) =
                    lazy_symmetric_group_workspace_for_point_tile(reducer, point_tile_capacity)?;
            }
            None => {}
        }
        companion.bind_primary_color_view(&contraction)?;
        let mut external_momenta = Vec::new();
        ensure_external_momentum_workspace_capacity(
            &mut external_momenta,
            1,
            external_source_count,
            effective_point_tile_size,
        )?;
        Ok(Self {
            companion,
            contraction,
            backend_name,
            parameter_defaults,
            parameter_projection,
            parameters_re: vec![0.0; parameter_count],
            parameters_im: vec![0.0; parameter_count],
            external_source_count,
            effective_point_tile_size,
            external_momenta,
            color_transform_re: vec![0.0; color_transform_len],
            color_transform_im: vec![0.0; color_transform_len],
            symmetric_group_color_workspace,
            order_catalog,
        })
    }

    pub(in crate::engine) fn backend_name(&self) -> &str {
        &self.backend_name
    }

    pub(in crate::engine) fn effective_point_tile_size(&self) -> usize {
        self.effective_point_tile_size
    }

    pub(in crate::engine) fn amplitude_order_catalog(&self) -> &AmplitudeOrderCatalog {
        &self.order_catalog
    }

    pub(in crate::engine) fn clear(&mut self) -> RusticolResult<()> {
        self.companion.clear()
    }

    pub(in crate::engine) fn state_census(
        &self,
        process_id: &str,
    ) -> RusticolResult<Option<serde_json::Value>> {
        Ok(Some(
            RecurrenceNativeRuntime::persisted_helicity_selector_state_census(
                process_id,
                self.companion.cache_census(),
                self.companion.active_family_inspection_census(),
            ),
        ))
    }

    fn prepare_parameters(&mut self, common: &ExecutionRuntime) -> RusticolResult<()> {
        initialize_prepared_parameter_planes(
            &self.parameter_defaults,
            &self.parameter_projection,
            &common.model_parameter_values_f64,
            &mut self.parameters_re,
            &mut self.parameters_im,
        )?;
        self.companion
            .executor
            .set_parameter_planes(&self.parameters_re, &self.parameters_im)
    }

    /// Validate selectors and return the one runnable physics helicity, or
    /// `None` for an empty or structural-zero selection (all values zero).
    fn singleton_helicity(
        &self,
        physics: &PhysicsRuntime,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<Option<(usize, u32)>> {
        if selected_colors.is_some() {
            return Err(selector_only_unsupported("colour-selected evaluation"));
        }
        if physics.manifest.color_components.len() != 1 {
            return Err(RusticolError::integrity(
                "selector-only full colour must expose one contracted colour component",
            ));
        }
        validate_recurrence_selector_ids(physics, selected_helicities, None)?;
        let Some(selected) = selected_helicities else {
            return Err(selector_only_unsupported("helicity-summed evaluation"));
        };
        if selected.is_empty() {
            return Ok(None);
        }
        if selected.len() != 1 {
            return Err(selector_only_unsupported("helicity-summed evaluation"));
        }
        let selected_id = selected
            .first()
            .expect("singleton helicity selector vanished");
        let helicity_index = physics
            .helicity_index_by_id
            .get(selected_id)
            .copied()
            .ok_or_else(|| {
                RusticolError::selector(format!("unknown resolved helicity id {selected_id:?}"))
            })?;
        let helicity = &physics.manifest.helicities[helicity_index];
        if helicity.structural_zero || helicity.coefficient == 0.0 {
            return Ok(None);
        }
        Ok(Some((
            helicity_index,
            self.companion.resolved_helicity_id(helicity_index)?,
        )))
    }

    fn tile_parts(
        &mut self,
    ) -> (
        &mut RecurrencePersistedHelicitySelectorRuntime,
        &RecurrenceColorContraction,
        CompanionTileScratch<'_>,
    ) {
        (
            &mut self.companion,
            &self.contraction,
            CompanionTileScratch {
                external_source_count: self.external_source_count,
                point_tile_size: self.effective_point_tile_size,
                external_momenta: &mut self.external_momenta,
                color_transform_re: &mut self.color_transform_re,
                color_transform_im: &mut self.color_transform_im,
                symmetric_group_color_workspace: &mut self.symmetric_group_color_workspace,
            },
        )
    }

    /// Totals of one selected helicity; every other selection is refused.
    pub(in crate::engine) fn run_f64_view_into_unprofiled(
        &mut self,
        common: &ExecutionRuntime,
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
        let selection = self.singleton_helicity(physics, selected_helicities, selected_colors)?;
        output.fill(0.0);
        let Some((physics_helicity_id, resolved_helicity_id)) = selection else {
            return Ok(());
        };
        self.prepare_parameters(common)?;
        let (companion, contraction, scratch) = self.tile_parts();
        run_companion_tiles_unprofiled(
            companion,
            contraction,
            scratch,
            physics,
            common.normalization_factor,
            batch,
            physics_helicity_id,
            resolved_helicity_id,
            output,
        )
    }

    /// Profiled totals of one selected helicity.
    pub(in crate::engine) fn run_f64_with_global_selectors(
        &mut self,
        common: &ExecutionRuntime,
        batch: &[Vec<[f64; 4]>],
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
    ) -> RusticolResult<(Vec<f64>, RuntimeProfile)> {
        let total_started = Instant::now();
        let physics = common.physics.as_ref().ok_or_else(|| {
            RusticolError::artifact("recurrence execution requires physics metadata")
        })?;
        let selection = self.singleton_helicity(physics, selected_helicities, selected_colors)?;
        let mut values = vec![0.0; batch.len()];
        let Some((physics_helicity_id, resolved_helicity_id)) = selection else {
            return Ok((
                values,
                persisted_companion_profile(
                    total_started.elapsed(),
                    Duration::ZERO,
                    Duration::ZERO,
                    Duration::ZERO,
                    Duration::ZERO,
                    0,
                    persisted_execution_report_aggregate(),
                ),
            ));
        };
        let parameter_started = Instant::now();
        self.prepare_parameters(common)?;
        let parameter_setup = parameter_started.elapsed();
        let (companion, contraction, scratch) = self.tile_parts();
        let profile = run_companion_tiles_profiled(
            companion,
            contraction,
            scratch,
            physics,
            common.normalization_factor,
            batch,
            physics_helicity_id,
            resolved_helicity_id,
            total_started,
            parameter_setup,
            |point, value| values[point] += value,
        )?;
        Ok((values, profile))
    }

    /// Single-order partials copy the totals; mixed orders need the main schedule.
    pub(in crate::engine) fn run_order_partials_f64_view_into(
        &mut self,
        common: &ExecutionRuntime,
        batch: F64MomentumBatchView<'_>,
        selected_helicities: Option<&BTreeSet<String>>,
        selected_colors: Option<&BTreeSet<String>>,
        totals: &mut [f64],
        partials: &mut [f64],
    ) -> RusticolResult<()> {
        if !self.order_catalog.is_single() {
            return Err(selector_only_unsupported("amplitude-order partials"));
        }
        if totals.len() != batch.point_count() || partials.len() != batch.point_count() {
            return Err(RusticolError::invalid_argument(format!(
                "recurrence order output has lengths {} and {}, expected {} each",
                totals.len(),
                partials.len(),
                batch.point_count(),
            )));
        }
        self.run_f64_view_into_unprofiled(
            common,
            batch,
            selected_helicities,
            selected_colors,
            totals,
        )?;
        partials.copy_from_slice(totals);
        Ok(())
    }
}

/// Point tile of the companion from its own footprint: the requested tile,
/// capped by the workspace target and a 4 MiB current+amplitude cache target.
fn companion_point_tile_size(plan: &DirectRecurrencePlan) -> RusticolResult<usize> {
    const CACHE_TARGET_BYTES: usize = 4 * 1024 * 1024;
    const MINIMUM_TILED_POINTS: usize = 8;
    let components = usize::try_from(plan.current_arena_components())
        .ok()
        .and_then(|current| {
            usize::try_from(plan.amplitude_destination_count())
                .ok()
                .and_then(|amplitude| current.checked_add(amplitude))
        })
        .ok_or_else(|| RusticolError::artifact("companion component count overflows usize"))?;
    let split_complex_scalars = components
        .checked_mul(2)
        .ok_or_else(|| RusticolError::artifact("companion workspace size overflows usize"))?;
    let momentum_scalars = plan
        .momentum_forms()
        .len()
        .checked_mul(4)
        .ok_or_else(|| RusticolError::artifact("companion momentum size overflows usize"))?;
    let per_point_scalars = split_complex_scalars
        .checked_add(momentum_scalars)
        .filter(|count| *count != 0)
        .ok_or_else(|| RusticolError::artifact("companion per-point size is invalid"))?;
    let workspace_bytes = usize::try_from(plan.workspace_mib())
        .ok()
        .and_then(|mib| mib.checked_mul(1024 * 1024))
        .ok_or_else(|| RusticolError::artifact("companion workspace target overflows usize"))?;
    let workspace_tile = crate::direct_arena::deterministic_point_tile_size(
        plan.point_tile_size(),
        workspace_bytes,
        usize::MAX,
        per_point_scalars,
    )? as usize;
    let split_complex_bytes = split_complex_scalars.max(1) * std::mem::size_of::<f64>();
    let cache_tile = (CACHE_TARGET_BYTES / split_complex_bytes).max(1);
    let cache_tile = 1usize << (usize::BITS - 1 - cache_tile.leading_zeros());
    Ok(workspace_tile
        .min(cache_tile.max(MINIMUM_TILED_POINTS))
        .max(1))
}
