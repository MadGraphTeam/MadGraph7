// SPDX-License-Identifier: 0BSD

//! Context-owning SymJIT adapter for Direct-Arena recurrence executors.
//!
//! This module is intentionally disjoint from the recurrence plan/runtime. It
//! proves the prepared-call boundary: model-fixed projections are resolved
//! once at load, hot calls build only pointer descriptors on the stack, and
//! generated O2 code mutates persistent arena destinations directly.

use crate::recurrence::direct_backend::{
    DIRECT_STATUS_OK, DirectArenaView, DirectClosureExecutor, DirectContributionExecutor,
    DirectExecutorHandle, DirectFactorView, DirectFinalizationExecutor, DirectMomentumView,
    DirectParameterView,
};
use crate::recurrence::{
    DIRECT_NONE_U32, DirectClosureRow, DirectContributionRow, DirectExecutorRole,
    DirectFinalizationRow,
};
use crate::{RusticolError, RusticolResult};
use std::any::Any;
use std::ffi::{c_int, c_void};
use std::mem::MaybeUninit;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::ptr;
use symjit::{
    Config, DIRECT_APPLICATION_STORAGE_ABI, DIRECT_EXACT_FACTOR_IMAG_SCALAR,
    DIRECT_EXACT_FACTOR_REAL_SCALAR, DIRECT_NO_ALIAS, DIRECT_STATUS_EXECUTION_FAILED,
    DIRECT_STATUS_INVALID_ARGUMENT, DIRECT_STATUS_INVALID_CONTEXT, Defuns, DirectApplication,
    DirectApplicationMetadata, DirectCallable,
    DirectDestinationOperation as SymjitDestinationOperation, DirectInputBinding, DirectPlane,
    DirectScalar, Storage,
};

const MAX_DIRECT_PLANES: usize = 512;
const MAX_DIRECT_SCALARS: usize = 256;
const STATUS_BOUNDS: c_int = 4;
const STATUS_ROLE_MISMATCH: c_int = 5;

/// One model-fixed input-plane projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SymjitDirectPlaneProjection {
    ParentCurrent {
        parent: u8,
        component: u16,
        imaginary: bool,
    },
    Momentum {
        operand: u8,
        lorentz_component: u16,
    },
    DestinationCurrent {
        component: u16,
        imaginary: bool,
    },
    DestinationAmplitude {
        component: u16,
        imaginary: bool,
    },
}

/// One model-fixed scalar projection.
///
/// Couplings use `Parameter`: their runtime slots are fixed by the prepared
/// model rather than copied into a per-call parameter buffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum SymjitDirectScalarProjection {
    Parameter { index: u32, imaginary: bool },
    ExactFactor { imaginary: bool },
    Literal(f64),
}

/// Owns the context addressed by a [`DirectExecutorHandle`].
pub(crate) struct LoadedSymjitDirectExecutor {
    context: Box<SymjitDirectExecutorContext>,
}

struct SymjitDirectExecutorContext {
    role: DirectExecutorRole,
    callable: DirectCallable,
    #[cfg(test)]
    simd_lane_width: usize,
    input_planes: Box<[SymjitDirectPlaneProjection]>,
    scalars: Box<[SymjitDirectScalarProjection]>,
    display_path: PathBuf,
}

impl LoadedSymjitDirectExecutor {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_bytes(
        bytes: &[u8],
        display_path: PathBuf,
        application_abi: &str,
        role: DirectExecutorRole,
        input_planes: Vec<SymjitDirectPlaneProjection>,
        scalars: Vec<SymjitDirectScalarProjection>,
    ) -> RusticolResult<Self> {
        if role == DirectExecutorRole::Source {
            return Err(RusticolError::compatibility(
                "recurrence source executors are Rusticol SourceIR intrinsics, not SymJIT direct applications",
            ));
        }
        if application_abi != DIRECT_APPLICATION_STORAGE_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported SymJIT direct application ABI {application_abi:?}; expected {DIRECT_APPLICATION_STORAGE_ABI:?}"
            )));
        }

        let mut loader_config = Config::default();
        loader_config.set_defuns(Defuns::new());
        let mut input = bytes;
        let mut application = guard_symjit_panic(
            || DirectApplication::load(&mut input, &loader_config),
            &display_path,
            "load",
        )?
        .map_err(|error| {
            RusticolError::compatibility(format!(
                "could not load SymJIT direct application {}: {error}",
                display_path.display()
            ))
        })?;
        if !input.is_empty() {
            return Err(RusticolError::integrity(format!(
                "SymJIT direct application {} has {} trailing bytes",
                display_path.display(),
                input.len()
            )));
        }

        validate_projections(&application, role, &input_planes, &scalars)?;
        application.prepare_simd();
        let applet =
            guard_symjit_panic(|| application.seal(), &display_path, "seal")?.map_err(|error| {
                RusticolError::evaluation(format!(
                    "could not seal SymJIT direct application {}: {error}",
                    display_path.display()
                ))
            })?;
        #[cfg(test)]
        let simd_lane_width = applet.simd_lane_width();
        Ok(Self {
            context: Box::new(SymjitDirectExecutorContext {
                role,
                callable: applet.into_callable(),
                #[cfg(test)]
                simd_lane_width,
                input_planes: input_planes.into_boxed_slice(),
                scalars: scalars.into_boxed_slice(),
                display_path,
            }),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_prepared_application_bytes(
        bytes: &[u8],
        display_path: PathBuf,
        source_application_abi: &str,
        role: DirectExecutorRole,
        direct_metadata: DirectApplicationMetadata,
        input_planes: Vec<SymjitDirectPlaneProjection>,
        scalars: Vec<SymjitDirectScalarProjection>,
    ) -> RusticolResult<Self> {
        if role == DirectExecutorRole::Source {
            return Err(RusticolError::compatibility(
                "recurrence source executors are Rusticol SourceIR intrinsics, not SymJIT direct applications",
            ));
        }
        if source_application_abi != "symjit-application-storage-v3" {
            return Err(RusticolError::compatibility(format!(
                "unsupported prepared SymJIT source application ABI {source_application_abi:?}"
            )));
        }
        let mut loader_config = Config::default();
        loader_config.set_defuns(Defuns::new());
        let mut input = bytes;
        let mut application = guard_symjit_panic(
            || {
                DirectApplication::from_source_storage(
                    &mut input,
                    &loader_config,
                    direct_metadata,
                )
            },
            &display_path,
            "transform",
        )?
        .map_err(|error| {
            RusticolError::compatibility(format!(
                "could not transform prepared SymJIT application {} into a direct application: {error}",
                display_path.display()
            ))
        })?;
        if !input.is_empty() {
            return Err(RusticolError::integrity(format!(
                "prepared SymJIT application {} has {} trailing bytes",
                display_path.display(),
                input.len()
            )));
        }
        validate_projections(&application, role, &input_planes, &scalars)?;
        application.prepare_simd();
        let applet = guard_symjit_panic(|| application.seal(), &display_path, "seal transformed")?
            .map_err(|error| {
                RusticolError::evaluation(format!(
                    "could not seal transformed SymJIT direct application {}: {error}",
                    display_path.display()
                ))
            })?;
        #[cfg(test)]
        let simd_lane_width = applet.simd_lane_width();
        Ok(Self {
            context: Box::new(SymjitDirectExecutorContext {
                role,
                callable: applet.into_callable(),
                #[cfg(test)]
                simd_lane_width,
                input_planes: input_planes.into_boxed_slice(),
                scalars: scalars.into_boxed_slice(),
                display_path,
            }),
        })
    }

    pub(crate) fn handle(&self) -> DirectExecutorHandle {
        let context = ptr::from_ref(self.context.as_ref()).cast();
        match self.context.role {
            DirectExecutorRole::Contribution => DirectExecutorHandle::Contribution {
                call: execute_contribution_rows as DirectContributionExecutor,
                context,
            },
            DirectExecutorRole::Finalization => DirectExecutorHandle::Finalization {
                call: execute_finalization_rows as DirectFinalizationExecutor,
                context,
            },
            DirectExecutorRole::Closure => DirectExecutorHandle::Closure {
                call: execute_closure_rows as DirectClosureExecutor,
                context,
            },
            DirectExecutorRole::Source => {
                unreachable!("source roles are rejected while loading a SymJIT direct executor")
            }
        }
    }

    #[cfg(test)]
    fn context_address(&self) -> *const c_void {
        ptr::from_ref(self.context.as_ref()).cast()
    }

    #[cfg(test)]
    fn simd_lane_width(&self) -> usize {
        self.context.simd_lane_width
    }
}

fn validate_projections(
    application: &DirectApplication,
    role: DirectExecutorRole,
    input_planes: &[SymjitDirectPlaneProjection],
    scalars: &[SymjitDirectScalarProjection],
) -> RusticolResult<()> {
    let metadata = application.metadata();
    if input_planes.len() != metadata.input_plane_count as usize {
        return Err(RusticolError::integrity(format!(
            "SymJIT direct input-plane projection has {} entries, expected {}",
            input_planes.len(),
            metadata.input_plane_count
        )));
    }
    if scalars.len() != metadata.scalar_input_count as usize {
        return Err(RusticolError::integrity(format!(
            "SymJIT direct scalar projection has {} entries, expected {}",
            scalars.len(),
            metadata.scalar_input_count
        )));
    }
    if scalars.get(DIRECT_EXACT_FACTOR_REAL_SCALAR as usize)
        != Some(&SymjitDirectScalarProjection::ExactFactor { imaginary: false })
        || scalars.get(DIRECT_EXACT_FACTOR_IMAG_SCALAR as usize)
            != Some(&SymjitDirectScalarProjection::ExactFactor { imaginary: true })
    {
        return Err(RusticolError::integrity(
            "SymJIT direct scalar slots 0 and 1 must project the exact factor real and imaginary values",
        ));
    }

    let expected_operation = match role {
        DirectExecutorRole::Contribution => SymjitDestinationOperation::Add,
        DirectExecutorRole::Finalization => SymjitDestinationOperation::FinalizeInPlace,
        DirectExecutorRole::Closure => SymjitDestinationOperation::ClosureAdd,
        DirectExecutorRole::Source => {
            return Err(RusticolError::compatibility(
                "recurrence source executors are Rusticol SourceIR intrinsics",
            ));
        }
    };
    if metadata.destination_operation != expected_operation {
        return Err(RusticolError::integrity(format!(
            "SymJIT direct destination operation {:?} does not match executor role {role:?}",
            metadata.destination_operation
        )));
    }
    let total_plane_count = input_planes
        .len()
        .checked_add(metadata.output_alias_inputs.len())
        .ok_or_else(|| RusticolError::integrity("SymJIT direct plane count overflows"))?;
    if total_plane_count > MAX_DIRECT_PLANES || scalars.len() > MAX_DIRECT_SCALARS {
        return Err(RusticolError::compatibility(format!(
            "SymJIT direct template needs {total_plane_count} planes and {} scalars; adapter limits are {MAX_DIRECT_PLANES} and {MAX_DIRECT_SCALARS}",
            scalars.len()
        )));
    }

    for binding in &metadata.parameter_bindings {
        match *binding {
            DirectInputBinding::Plane(index) if index as usize >= input_planes.len() => {
                return Err(RusticolError::integrity(
                    "SymJIT direct parameter plane is absent from its fixed projection",
                ));
            }
            DirectInputBinding::Scalar(index) if index as usize >= scalars.len() => {
                return Err(RusticolError::integrity(
                    "SymJIT direct parameter scalar is absent from its fixed projection",
                ));
            }
            _ => {}
        }
    }

    let expected_destination = |projection: SymjitDirectPlaneProjection| match role {
        DirectExecutorRole::Contribution | DirectExecutorRole::Finalization => {
            matches!(
                projection,
                SymjitDirectPlaneProjection::DestinationCurrent { .. }
            )
        }
        DirectExecutorRole::Closure => matches!(
            projection,
            SymjitDirectPlaneProjection::DestinationAmplitude { .. }
        ),
        DirectExecutorRole::Source => false,
    };
    for &alias in &metadata.output_alias_inputs {
        if alias == DIRECT_NO_ALIAS {
            return Err(RusticolError::compatibility(
                "recurrence direct applications must alias every output into a persistent destination plane",
            ));
        }
        let projection = input_planes
            .get(alias as usize)
            .copied()
            .ok_or_else(|| RusticolError::integrity("direct output alias is out of bounds"))?;
        if !expected_destination(projection) {
            return Err(RusticolError::integrity(
                "SymJIT direct output does not alias a destination projection for its executor role",
            ));
        }
    }
    Ok(())
}

unsafe extern "C" fn execute_contribution_rows(
    context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectContributionRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    invoke_typed_rows(
        context,
        DirectExecutorRole::Contribution,
        rows,
        row_count,
        point_count,
        |context, row| {
            context.execute_row(
                DirectRowRef::Contribution(row),
                arena,
                momenta,
                parameters,
                factors,
                point_count,
            )
        },
    )
}

unsafe extern "C" fn execute_finalization_rows(
    context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectFinalizationRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    invoke_typed_rows(
        context,
        DirectExecutorRole::Finalization,
        rows,
        row_count,
        point_count,
        |context, row| {
            context.execute_row(
                DirectRowRef::Finalization(row),
                arena,
                momenta,
                parameters,
                factors,
                point_count,
            )
        },
    )
}

unsafe extern "C" fn execute_closure_rows(
    context: *const c_void,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
    parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectClosureRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    invoke_typed_rows(
        context,
        DirectExecutorRole::Closure,
        rows,
        row_count,
        point_count,
        |context, row| {
            context.execute_row(
                DirectRowRef::Closure(row),
                arena,
                momenta,
                parameters,
                factors,
                point_count,
            )
        },
    )
}

fn invoke_typed_rows<T>(
    context: *const c_void,
    role: DirectExecutorRole,
    rows: *const T,
    row_count: u32,
    point_count: u32,
    mut execute: impl FnMut(&SymjitDirectExecutorContext, &T) -> c_int,
) -> c_int {
    if context.is_null() {
        return DIRECT_STATUS_INVALID_CONTEXT;
    }
    if row_count == 0 || point_count == 0 || rows.is_null() {
        return DIRECT_STATUS_INVALID_ARGUMENT;
    }
    let result = catch_unwind(AssertUnwindSafe(|| {
        let context = unsafe { &*context.cast::<SymjitDirectExecutorContext>() };
        if context.role != role {
            return STATUS_ROLE_MISMATCH;
        }
        let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
        for row in rows {
            let status = execute(context, row);
            if status != DIRECT_STATUS_OK {
                return status;
            }
        }
        DIRECT_STATUS_OK
    }));
    result.unwrap_or(DIRECT_STATUS_EXECUTION_FAILED)
}

enum DirectRowRef<'a> {
    Contribution(&'a DirectContributionRow),
    Finalization(&'a DirectFinalizationRow),
    Closure(&'a DirectClosureRow),
}

impl DirectRowRef<'_> {
    fn exact_factor_id(&self) -> u32 {
        match self {
            Self::Contribution(row) => row.exact_factor_id,
            Self::Finalization(row) => row.exact_factor_id,
            Self::Closure(row) => row.exact_factor_id,
        }
    }

    fn parent_component_base(&self, parent: u8) -> Option<u32> {
        match (self, parent) {
            (Self::Contribution(row), 0) => Some(row.parent0_component_base),
            (Self::Contribution(row), 1)
                if row.parent1_component_base_or_sentinel != DIRECT_NONE_U32 =>
            {
                Some(row.parent1_component_base_or_sentinel)
            }
            (Self::Finalization(row), 0) => Some(row.component_base),
            (Self::Closure(row), 0) => Some(row.parent0_component_base),
            (Self::Closure(row), 1)
                if row.parent1_component_base_or_sentinel != DIRECT_NONE_U32 =>
            {
                Some(row.parent1_component_base_or_sentinel)
            }
            _ => None,
        }
    }

    fn momentum_form_id(&self, operand: u8) -> Option<u32> {
        match (self, operand) {
            (Self::Contribution(row), 0) => Some(row.parent0_momentum_form_id),
            (Self::Contribution(row), 1)
                if row.parent1_momentum_form_id_or_sentinel != DIRECT_NONE_U32 =>
            {
                Some(row.parent1_momentum_form_id_or_sentinel)
            }
            (Self::Finalization(row), 0) => Some(row.momentum_form_id),
            (Self::Closure(row), 0) => Some(row.parent0_momentum_form_id),
            (Self::Closure(row), 1)
                if row.parent1_momentum_form_id_or_sentinel != DIRECT_NONE_U32 =>
            {
                Some(row.parent1_momentum_form_id_or_sentinel)
            }
            _ => None,
        }
    }

    fn destination_current_base(&self) -> Option<u32> {
        match self {
            Self::Contribution(row) => Some(row.destination_component_base),
            Self::Finalization(row) => Some(row.component_base),
            Self::Closure(_) => None,
        }
    }

    fn destination_amplitude_base(&self) -> Option<u32> {
        match self {
            Self::Closure(row) => Some(row.amplitude_destination_id),
            _ => None,
        }
    }
}

impl SymjitDirectExecutorContext {
    #[allow(clippy::too_many_arguments)]
    fn execute_row(
        &self,
        row: DirectRowRef<'_>,
        arena: DirectArenaView,
        momenta: DirectMomentumView,
        parameters: DirectParameterView,
        factors: DirectFactorView,
        point_count: u32,
    ) -> c_int {
        if point_count == 0 || arena.point_stride == 0 || point_count > arena.point_stride {
            return DIRECT_STATUS_INVALID_ARGUMENT;
        }
        if arena.point_stride != momenta.point_stride {
            return DIRECT_STATUS_INVALID_ARGUMENT;
        }

        let metadata = self.callable.metadata();
        let mut planes = [const { MaybeUninit::<DirectPlane>::uninit() }; MAX_DIRECT_PLANES];
        let mut scalars = [const { MaybeUninit::<DirectScalar>::uninit() }; MAX_DIRECT_SCALARS];

        for (index, projection) in self.input_planes.iter().copied().enumerate() {
            let Some(plane) = resolve_plane(projection, &row, arena, momenta) else {
                return STATUS_BOUNDS;
            };
            planes[index].write(plane);
        }
        let input_plane_count = self.input_planes.len();
        for (output, &alias) in metadata.output_alias_inputs.iter().enumerate() {
            let Some(source) = planes.get(alias as usize) else {
                return STATUS_BOUNDS;
            };
            // Alias indices were authenticated against the initialized input
            // projection prefix when the prepared application was loaded.
            let source = unsafe { source.assume_init() };
            planes[input_plane_count + output].write(source);
        }

        for (index, projection) in self.scalars.iter().enumerate() {
            let Some(scalar) = resolve_scalar(projection, &row, parameters, factors) else {
                return STATUS_BOUNDS;
            };
            scalars[index].write(scalar);
        }

        let plane_count = input_plane_count + metadata.output_alias_inputs.len();
        // Both prefixes are initialized completely above. The remainder of
        // each fixed-capacity stack array is deliberately left untouched.
        let planes = unsafe {
            std::slice::from_raw_parts(planes.as_ptr().cast::<DirectPlane>(), plane_count)
        };
        let scalars = unsafe {
            std::slice::from_raw_parts(scalars.as_ptr().cast::<DirectScalar>(), self.scalars.len())
        };
        let status = unsafe {
            self.callable
                .invoke_unchecked(planes, scalars, 0, point_count as usize)
        };
        if status == DIRECT_STATUS_EXECUTION_FAILED {
            let _ = &self.display_path;
        }
        status
    }
}

fn resolve_plane(
    projection: SymjitDirectPlaneProjection,
    row: &DirectRowRef<'_>,
    arena: DirectArenaView,
    momenta: DirectMomentumView,
) -> Option<DirectPlane> {
    match projection {
        SymjitDirectPlaneProjection::ParentCurrent {
            parent,
            component,
            imaginary,
        } => {
            let base = row.parent_component_base(parent)?;
            current_plane(arena, base, component, imaginary)
        }
        SymjitDirectPlaneProjection::Momentum {
            operand,
            lorentz_component,
        } => {
            let form = row.momentum_form_id(operand)?;
            momentum_plane(momenta, form, lorentz_component)
        }
        SymjitDirectPlaneProjection::DestinationCurrent {
            component,
            imaginary,
        } => {
            let base = row.destination_current_base()?;
            current_plane(arena, base, component, imaginary)
        }
        SymjitDirectPlaneProjection::DestinationAmplitude {
            component,
            imaginary,
        } => {
            let base = row.destination_amplitude_base()?;
            amplitude_plane(arena, base, component, imaginary)
        }
    }
}

fn current_plane(
    arena: DirectArenaView,
    base: u32,
    component: u16,
    imaginary: bool,
) -> Option<DirectPlane> {
    let component = u64::from(base).checked_add(u64::from(component))?;
    let pointer = if imaginary {
        arena.current_im
    } else {
        arena.current_re
    };
    plane_from_arena(
        pointer,
        arena.current_scalar_len,
        component,
        arena.point_stride,
    )
}

fn amplitude_plane(
    arena: DirectArenaView,
    base: u32,
    component: u16,
    imaginary: bool,
) -> Option<DirectPlane> {
    let component = u64::from(base).checked_add(u64::from(component))?;
    let pointer = if imaginary {
        arena.amplitude_im
    } else {
        arena.amplitude_re
    };
    plane_from_arena(
        pointer,
        arena.amplitude_scalar_len,
        component,
        arena.point_stride,
    )
}

fn momentum_plane(
    momenta: DirectMomentumView,
    form: u32,
    lorentz_component: u16,
) -> Option<DirectPlane> {
    if form >= momenta.form_count || lorentz_component >= momenta.lorentz_component_count {
        return None;
    }
    let plane = u64::from(form)
        .checked_mul(u64::from(momenta.lorentz_component_count))?
        .checked_add(u64::from(lorentz_component))?;
    // Generated code never writes a momentum input. The descriptor is mutable
    // only because it shares SymJIT's indirect plane ABI with aliased outputs.
    plane_from_arena(
        momenta.values.cast_mut(),
        momenta.scalar_len,
        plane,
        momenta.point_stride,
    )
}

fn plane_from_arena(
    pointer: *mut f64,
    scalar_len: u64,
    plane: u64,
    point_stride: u32,
) -> Option<DirectPlane> {
    if pointer.is_null() || point_stride == 0 {
        return None;
    }
    let offset = plane.checked_mul(u64::from(point_stride))?;
    let end = offset.checked_add(u64::from(point_stride))?;
    if end > scalar_len {
        return None;
    }
    let offset = usize::try_from(offset).ok()?;
    Some(unsafe { DirectPlane::from_raw_parts(pointer.add(offset), point_stride as usize) })
}

fn resolve_scalar(
    projection: &SymjitDirectScalarProjection,
    row: &DirectRowRef<'_>,
    parameters: DirectParameterView,
    factors: DirectFactorView,
) -> Option<DirectScalar> {
    let pointer = match projection {
        SymjitDirectScalarProjection::Parameter { index, imaginary } => {
            if *index >= parameters.value_count {
                return None;
            }
            let base = if *imaginary {
                parameters.values_im
            } else {
                parameters.values_re
            };
            if base.is_null() {
                return None;
            }
            unsafe { base.add(*index as usize) }
        }
        SymjitDirectScalarProjection::ExactFactor { imaginary } => {
            let index = row.exact_factor_id();
            if index >= factors.value_count {
                return None;
            }
            let base = if *imaginary {
                factors.values_im
            } else {
                factors.values_re
            };
            if base.is_null() {
                return None;
            }
            unsafe { base.add(index as usize) }
        }
        SymjitDirectScalarProjection::Literal(value) => ptr::from_ref(value),
    };
    Some(unsafe { DirectScalar::from_raw(pointer) })
}

/// One generic identity finalizer for every non-propagating state.
///
/// The prepared catalog resolves exactly this one function. Per-state
/// dimensions are deliberately read from `row.component_count`.
pub(crate) unsafe extern "C" fn execute_identity_finalization_rows(
    _context: *const c_void,
    arena: DirectArenaView,
    _momenta: DirectMomentumView,
    _parameters: DirectParameterView,
    factors: DirectFactorView,
    rows: *const DirectFinalizationRow,
    row_count: u32,
    point_count: u32,
) -> c_int {
    if rows.is_null()
        || row_count == 0
        || point_count == 0
        || arena.point_stride == 0
        || point_count > arena.point_stride
    {
        return DIRECT_STATUS_INVALID_ARGUMENT;
    }
    if arena.current_re.is_null()
        || arena.current_im.is_null()
        || factors.values_re.is_null()
        || factors.values_im.is_null()
    {
        return DIRECT_STATUS_INVALID_ARGUMENT;
    }

    let rows = unsafe { std::slice::from_raw_parts(rows, row_count as usize) };
    for row in rows {
        if row.component_count == 0 || row.exact_factor_id >= factors.value_count {
            return STATUS_BOUNDS;
        }
        let factor_re = unsafe { *factors.values_re.add(row.exact_factor_id as usize) };
        let factor_im = unsafe { *factors.values_im.add(row.exact_factor_id as usize) };
        for component in 0..u64::from(row.component_count) {
            let plane = match u64::from(row.component_base).checked_add(component) {
                Some(value) => value,
                None => return STATUS_BOUNDS,
            };
            let offset = match plane.checked_mul(u64::from(arena.point_stride)) {
                Some(value) => value,
                None => return STATUS_BOUNDS,
            };
            let end = match offset.checked_add(u64::from(point_count)) {
                Some(value) => value,
                None => return STATUS_BOUNDS,
            };
            if end > arena.current_scalar_len {
                return STATUS_BOUNDS;
            }
            let Ok(offset) = usize::try_from(offset) else {
                return STATUS_BOUNDS;
            };
            for point in 0..point_count as usize {
                let index = offset + point;
                let value_re = unsafe { *arena.current_re.add(index) };
                let value_im = unsafe { *arena.current_im.add(index) };
                unsafe {
                    *arena.current_re.add(index) = factor_re * value_re - factor_im * value_im;
                    *arena.current_im.add(index) = factor_re * value_im + factor_im * value_re;
                }
            }
        }
    }
    DIRECT_STATUS_OK
}

fn guard_symjit_panic<T>(
    operation: impl FnOnce() -> T,
    path: &std::path::Path,
    action: &str,
) -> RusticolResult<T> {
    catch_unwind(AssertUnwindSafe(operation)).map_err(|payload| {
        RusticolError::compatibility(format!(
            "SymJIT panicked while trying to {action} direct application {}: {}",
            path.display(),
            panic_detail(payload)
        ))
    })
}

fn panic_detail(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::direct_backend::{
        DirectFactorView, DirectMomentumView, DirectParameterView,
    };
    use symjit::{
        Compiler, DirectApplication, DirectApplicationMetadata,
        DirectDestinationOperation as SymjitDestinationOperation, DirectInputBinding, Expr,
    };

    fn direct_contribution_bytes() -> Vec<u8> {
        let mut config = Config::default();
        config.set_opt_level(2);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let parent = Expr::var("parent");
        let coupling = Expr::var("coupling");
        let expression = &coupling * &parent;
        let source = Compiler::with_config(config)
            .compile_params(&[], &[expression], &[parent, coupling])
            .unwrap();
        let metadata = DirectApplicationMetadata::new(
            SymjitDestinationOperation::Add,
            vec![],
            vec![
                DirectInputBinding::Plane(0),
                DirectInputBinding::Plane(1),
                DirectInputBinding::Scalar(2),
                DirectInputBinding::Scalar(3),
            ],
            4,
            4,
            vec![2, 3],
        )
        .unwrap();
        let application = DirectApplication::new(source, metadata).unwrap();
        let mut bytes = Vec::new();
        application.save(&mut bytes).unwrap();
        bytes
    }

    fn direct_closure_bytes() -> Vec<u8> {
        let mut config = Config::default();
        config.set_opt_level(2);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let parent = Expr::var("parent");
        let coupling = Expr::var("coupling");
        let expression = &coupling * &parent;
        let source = Compiler::with_config(config)
            .compile_params(&[], &[expression], &[parent, coupling])
            .unwrap();
        let metadata = DirectApplicationMetadata::new(
            SymjitDestinationOperation::ClosureAdd,
            vec![],
            vec![
                DirectInputBinding::Plane(0),
                DirectInputBinding::Plane(1),
                DirectInputBinding::Scalar(2),
                DirectInputBinding::Scalar(3),
            ],
            4,
            4,
            vec![2, 3],
        )
        .unwrap();
        let application = DirectApplication::new(source, metadata).unwrap();
        let mut bytes = Vec::new();
        application.save(&mut bytes).unwrap();
        bytes
    }

    fn prepared_contribution_bytes() -> (Vec<u8>, DirectApplicationMetadata) {
        let mut config = Config::default();
        config.set_opt_level(2);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let parent = Expr::var("parent");
        let coupling = Expr::var("coupling");
        let expression = &coupling * &parent;
        let source = Compiler::with_config(config)
            .compile_params(&[], &[expression], &[parent, coupling])
            .unwrap();
        let metadata = DirectApplicationMetadata::new(
            SymjitDestinationOperation::Add,
            vec![],
            vec![
                DirectInputBinding::Plane(0),
                DirectInputBinding::Plane(1),
                DirectInputBinding::Scalar(2),
                DirectInputBinding::Scalar(3),
            ],
            4,
            4,
            vec![2, 3],
        )
        .unwrap();
        let mut bytes = Vec::new();
        source.save(&mut bytes).unwrap();
        (bytes, metadata)
    }

    fn prepared_in_place_finalization_bytes() -> (Vec<u8>, DirectApplicationMetadata) {
        let mut config = Config::default();
        config.set_opt_level(2);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let current_names = ["current_0", "current_1", "current_2", "current_3"];
        let current = current_names.map(Expr::var);
        let source = Compiler::with_config(config)
            .compile_params(
                &[],
                &[
                    current[2].clone(),
                    current[3].clone(),
                    current[0].clone(),
                    current[1].clone(),
                ],
                &current,
            )
            .unwrap();
        let parameter_bindings = (0..8).map(DirectInputBinding::Plane).collect::<Vec<_>>();
        let metadata = DirectApplicationMetadata::new(
            SymjitDestinationOperation::FinalizeInPlace,
            vec![],
            parameter_bindings,
            16,
            2,
            (8..16).collect(),
        )
        .unwrap();
        let mut bytes = Vec::new();
        source.save(&mut bytes).unwrap();
        (bytes, metadata)
    }

    fn plane_projections() -> Vec<SymjitDirectPlaneProjection> {
        vec![
            SymjitDirectPlaneProjection::ParentCurrent {
                parent: 0,
                component: 0,
                imaginary: false,
            },
            SymjitDirectPlaneProjection::ParentCurrent {
                parent: 0,
                component: 0,
                imaginary: true,
            },
            SymjitDirectPlaneProjection::DestinationCurrent {
                component: 0,
                imaginary: false,
            },
            SymjitDirectPlaneProjection::DestinationCurrent {
                component: 0,
                imaginary: true,
            },
        ]
    }

    fn scalar_projections() -> Vec<SymjitDirectScalarProjection> {
        vec![
            SymjitDirectScalarProjection::ExactFactor { imaginary: false },
            SymjitDirectScalarProjection::ExactFactor { imaginary: true },
            SymjitDirectScalarProjection::Parameter {
                index: 0,
                imaginary: false,
            },
            SymjitDirectScalarProjection::Parameter {
                index: 0,
                imaginary: true,
            },
        ]
    }

    fn closure_plane_projections() -> Vec<SymjitDirectPlaneProjection> {
        vec![
            SymjitDirectPlaneProjection::ParentCurrent {
                parent: 0,
                component: 0,
                imaginary: false,
            },
            SymjitDirectPlaneProjection::ParentCurrent {
                parent: 0,
                component: 0,
                imaginary: true,
            },
            SymjitDirectPlaneProjection::DestinationAmplitude {
                component: 0,
                imaginary: false,
            },
            SymjitDirectPlaneProjection::DestinationAmplitude {
                component: 0,
                imaginary: true,
            },
        ]
    }

    fn contribution_executor() -> LoadedSymjitDirectExecutor {
        LoadedSymjitDirectExecutor::load_bytes(
            &direct_contribution_bytes(),
            PathBuf::from("synthetic-direct.symjit"),
            DIRECT_APPLICATION_STORAGE_ABI,
            DirectExecutorRole::Contribution,
            plane_projections(),
            scalar_projections(),
        )
        .unwrap()
    }

    fn closure_executor() -> LoadedSymjitDirectExecutor {
        LoadedSymjitDirectExecutor::load_bytes(
            &direct_closure_bytes(),
            PathBuf::from("synthetic-closure-direct.symjit"),
            DIRECT_APPLICATION_STORAGE_ABI,
            DirectExecutorRole::Closure,
            closure_plane_projections(),
            scalar_projections(),
        )
        .unwrap()
    }

    #[test]
    fn existing_portable_o2_payload_is_transformed_at_load_without_duplication() {
        let (bytes, metadata) = prepared_contribution_bytes();
        let executor = LoadedSymjitDirectExecutor::load_prepared_application_bytes(
            &bytes,
            PathBuf::from("kernels/7/application.symjit"),
            "symjit-application-storage-v3",
            DirectExecutorRole::Contribution,
            metadata,
            plane_projections(),
            scalar_projections(),
        )
        .unwrap();
        assert_eq!(executor.handle().role(), DirectExecutorRole::Contribution);
        assert!(
            executor.simd_lane_width() > 1,
            "loaded Direct-Arena O2 applications must retain a SIMD callable"
        );
    }

    #[test]
    fn loaded_o2_executor_uses_owned_context_and_mutates_destination_arena() {
        let executor = contribution_executor();
        let handle = executor.handle();
        assert_eq!(handle.role(), DirectExecutorRole::Contribution);
        let DirectExecutorHandle::Contribution { call, context } = handle else {
            unreachable!()
        };
        assert_eq!(context, executor.context_address());
        assert!(!context.is_null());

        let point_stride = 5_u32;
        let mut current_re = vec![0.0; 2 * point_stride as usize];
        let mut current_im = vec![0.0; 2 * point_stride as usize];
        current_re[..5].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        current_im[..5].copy_from_slice(&[0.5, -0.5, 1.0, -1.0, 0.25]);
        current_re[5..].fill(10.0);
        current_im[5..].fill(-3.0);
        let original_re = current_re[5..].to_vec();
        let original_im = current_im[5..].to_vec();
        let mut amplitude_re = vec![0.0; point_stride as usize];
        let mut amplitude_im = vec![0.0; point_stride as usize];
        let momenta = vec![0.0; 4 * point_stride as usize];
        let parameter_re = [0.25];
        let parameter_im = [-0.5];
        let factor_re = [2.0];
        let factor_im = [0.125];
        let arena = DirectArenaView {
            current_re: current_re.as_mut_ptr(),
            current_im: current_im.as_mut_ptr(),
            current_scalar_len: current_re.len() as u64,
            amplitude_re: amplitude_re.as_mut_ptr(),
            amplitude_im: amplitude_im.as_mut_ptr(),
            amplitude_scalar_len: amplitude_re.len() as u64,
            point_stride,
        };
        let momentum_view = DirectMomentumView {
            values: momenta.as_ptr(),
            scalar_len: momenta.len() as u64,
            form_count: 1,
            lorentz_component_count: 4,
            point_stride,
        };
        let parameter_view = DirectParameterView {
            values_re: parameter_re.as_ptr(),
            values_im: parameter_im.as_ptr(),
            value_count: 1,
        };
        let factor_view = DirectFactorView {
            values_re: factor_re.as_ptr(),
            values_im: factor_im.as_ptr(),
            value_count: 1,
        };
        let row = DirectContributionRow {
            parent0_component_base: 0,
            parent1_component_base_or_sentinel: DIRECT_NONE_U32,
            parent0_momentum_form_id: 0,
            parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
            destination_component_base: 1,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };

        let status = unsafe {
            call(
                context,
                arena,
                momentum_view,
                parameter_view,
                factor_view,
                ptr::from_ref(&row),
                1,
                point_stride,
            )
        };
        assert_eq!(status, DIRECT_STATUS_OK);

        let scale_re = factor_re[0] * parameter_re[0] - factor_im[0] * parameter_im[0];
        let scale_im = factor_re[0] * parameter_im[0] + factor_im[0] * parameter_re[0];
        for point in 0..point_stride as usize {
            let parent_re = current_re[point];
            let parent_im = current_im[point];
            let expected_re = original_re[point] + scale_re * parent_re - scale_im * parent_im;
            let expected_im = original_im[point] + scale_re * parent_im + scale_im * parent_re;
            assert!((current_re[5 + point] - expected_re).abs() < 1.0e-12);
            assert!((current_im[5 + point] - expected_im).abs() < 1.0e-12);
        }
    }

    #[test]
    fn prepared_o2_finalization_reads_all_components_before_in_place_writes() {
        let (bytes, metadata) = prepared_in_place_finalization_bytes();
        let mut projections = Vec::new();
        for destination in [false, true] {
            for component in 0..4 {
                for imaginary in [false, true] {
                    projections.push(if destination {
                        SymjitDirectPlaneProjection::DestinationCurrent {
                            component,
                            imaginary,
                        }
                    } else {
                        SymjitDirectPlaneProjection::ParentCurrent {
                            parent: 0,
                            component,
                            imaginary,
                        }
                    });
                }
            }
        }
        let executor = LoadedSymjitDirectExecutor::load_prepared_application_bytes(
            &bytes,
            PathBuf::from("prepared-in-place-finalization.symjit"),
            "symjit-application-storage-v3",
            DirectExecutorRole::Finalization,
            metadata,
            projections,
            vec![
                SymjitDirectScalarProjection::ExactFactor { imaginary: false },
                SymjitDirectScalarProjection::ExactFactor { imaginary: true },
            ],
        )
        .unwrap();
        let DirectExecutorHandle::Finalization { call, context } = executor.handle() else {
            unreachable!()
        };
        let point_stride = 2_u32;
        let mut current_re = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut current_im = vec![-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0];
        let original_re = current_re.clone();
        let original_im = current_im.clone();
        let mut amplitude_re = [0.0; 2];
        let mut amplitude_im = [0.0; 2];
        let momenta = [0.0; 8];
        let factors_re = [1.0];
        let factors_im = [0.0];
        let row = DirectFinalizationRow {
            component_base: 0,
            component_count: 4,
            momentum_form_id: 0,
            exact_factor_id: 0,
            selector_domain_id: 0,
            flags: 0,
        };
        let status = unsafe {
            call(
                context,
                DirectArenaView {
                    current_re: current_re.as_mut_ptr(),
                    current_im: current_im.as_mut_ptr(),
                    current_scalar_len: current_re.len() as u64,
                    amplitude_re: amplitude_re.as_mut_ptr(),
                    amplitude_im: amplitude_im.as_mut_ptr(),
                    amplitude_scalar_len: amplitude_re.len() as u64,
                    point_stride,
                },
                DirectMomentumView {
                    values: momenta.as_ptr(),
                    scalar_len: momenta.len() as u64,
                    form_count: 1,
                    lorentz_component_count: 4,
                    point_stride,
                },
                DirectParameterView {
                    values_re: ptr::null(),
                    values_im: ptr::null(),
                    value_count: 0,
                },
                DirectFactorView {
                    values_re: factors_re.as_ptr(),
                    values_im: factors_im.as_ptr(),
                    value_count: 1,
                },
                ptr::from_ref(&row),
                1,
                point_stride,
            )
        };
        assert_eq!(status, DIRECT_STATUS_OK);
        assert_eq!(&current_re[0..2], &original_re[4..6]);
        assert_eq!(&current_re[2..4], &original_re[6..8]);
        assert_eq!(&current_re[4..6], &original_re[0..2]);
        assert_eq!(&current_re[6..8], &original_re[2..4]);
        assert_eq!(&current_im[0..2], &original_im[4..6]);
        assert_eq!(&current_im[2..4], &original_im[6..8]);
        assert_eq!(&current_im[4..6], &original_im[0..2]);
        assert_eq!(&current_im[6..8], &original_im[2..4]);
    }

    #[test]
    fn closure_rows_accumulate_nonunit_complex_factors_into_amplitude_arena() {
        let executor = closure_executor();
        let DirectExecutorHandle::Closure { call, context } = executor.handle() else {
            unreachable!()
        };

        let point_stride = 4_u32;
        let current_re = [1.0, -2.0, 0.5, 3.0];
        let current_im = [0.25, 1.0, -0.75, 2.0];
        let mut current_re_storage = current_re;
        let mut current_im_storage = current_im;
        let mut amplitude_re = [4.0, -1.0, 2.5, 0.0];
        let mut amplitude_im = [-2.0, 3.0, 0.5, 1.0];
        let original_re = amplitude_re;
        let original_im = amplitude_im;
        let momenta = [0.0; 16];
        let parameter_re = [0.5];
        let parameter_im = [-0.25];
        let factor_re = [2.0, -0.5];
        let factor_im = [0.75, 1.25];
        let arena = DirectArenaView {
            current_re: current_re_storage.as_mut_ptr(),
            current_im: current_im_storage.as_mut_ptr(),
            current_scalar_len: current_re_storage.len() as u64,
            amplitude_re: amplitude_re.as_mut_ptr(),
            amplitude_im: amplitude_im.as_mut_ptr(),
            amplitude_scalar_len: amplitude_re.len() as u64,
            point_stride,
        };
        let momentum_view = DirectMomentumView {
            values: momenta.as_ptr(),
            scalar_len: momenta.len() as u64,
            form_count: 1,
            lorentz_component_count: 4,
            point_stride,
        };
        let parameter_view = DirectParameterView {
            values_re: parameter_re.as_ptr(),
            values_im: parameter_im.as_ptr(),
            value_count: 1,
        };
        let factor_view = DirectFactorView {
            values_re: factor_re.as_ptr(),
            values_im: factor_im.as_ptr(),
            value_count: 2,
        };
        let rows = [
            DirectClosureRow {
                parent0_component_base: 0,
                parent1_component_base_or_sentinel: DIRECT_NONE_U32,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
                amplitude_destination_id: 0,
                exact_factor_id: 0,
                component_factor_start: 0,
                component_count: 1,
                selector_domain_id: 0,
                flags: 0,
            },
            DirectClosureRow {
                parent0_component_base: 0,
                parent1_component_base_or_sentinel: DIRECT_NONE_U32,
                parent0_momentum_form_id: 0,
                parent1_momentum_form_id_or_sentinel: DIRECT_NONE_U32,
                amplitude_destination_id: 0,
                exact_factor_id: 1,
                component_factor_start: 0,
                component_count: 1,
                selector_domain_id: 0,
                flags: 0,
            },
        ];

        let status = unsafe {
            call(
                context,
                arena,
                momentum_view,
                parameter_view,
                factor_view,
                rows.as_ptr(),
                rows.len() as u32,
                point_stride,
            )
        };
        assert_eq!(status, DIRECT_STATUS_OK);

        let factor_sum_re = factor_re.iter().sum::<f64>();
        let factor_sum_im = factor_im.iter().sum::<f64>();
        let scale_re = factor_sum_re * parameter_re[0] - factor_sum_im * parameter_im[0];
        let scale_im = factor_sum_re * parameter_im[0] + factor_sum_im * parameter_re[0];
        for point in 0..point_stride as usize {
            let expected_re =
                original_re[point] + scale_re * current_re[point] - scale_im * current_im[point];
            let expected_im =
                original_im[point] + scale_re * current_im[point] + scale_im * current_re[point];
            assert!((amplitude_re[point] - expected_re).abs() < 1.0e-12);
            assert!((amplitude_im[point] - expected_im).abs() < 1.0e-12);
        }
    }

    #[test]
    fn one_identity_finalizer_handles_different_row_component_counts() {
        let point_stride = 3_u32;
        let mut current_re = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut current_im = vec![0.5; current_re.len()];
        let mut amplitude_re = vec![0.0; point_stride as usize];
        let mut amplitude_im = vec![0.0; point_stride as usize];
        let momenta = vec![0.0; 4 * point_stride as usize];
        let parameters = [0.0];
        let factors_re = [2.0, -1.0];
        let factors_im = [0.0, 1.0];
        let arena = DirectArenaView {
            current_re: current_re.as_mut_ptr(),
            current_im: current_im.as_mut_ptr(),
            current_scalar_len: current_re.len() as u64,
            amplitude_re: amplitude_re.as_mut_ptr(),
            amplitude_im: amplitude_im.as_mut_ptr(),
            amplitude_scalar_len: amplitude_re.len() as u64,
            point_stride,
        };
        let momentum_view = DirectMomentumView {
            values: momenta.as_ptr(),
            scalar_len: momenta.len() as u64,
            form_count: 1,
            lorentz_component_count: 4,
            point_stride,
        };
        let parameter_view = DirectParameterView {
            values_re: parameters.as_ptr(),
            values_im: parameters.as_ptr(),
            value_count: 1,
        };
        let factor_view = DirectFactorView {
            values_re: factors_re.as_ptr(),
            values_im: factors_im.as_ptr(),
            value_count: 2,
        };
        let rows = [
            DirectFinalizationRow {
                component_base: 0,
                component_count: 1,
                momentum_form_id: 0,
                exact_factor_id: 0,
                selector_domain_id: 0,
                flags: 0,
            },
            DirectFinalizationRow {
                component_base: 1,
                component_count: 2,
                momentum_form_id: 0,
                exact_factor_id: 1,
                selector_domain_id: 0,
                flags: 0,
            },
        ];
        let status = unsafe {
            execute_identity_finalization_rows(
                ptr::null(),
                arena,
                momentum_view,
                parameter_view,
                factor_view,
                rows.as_ptr(),
                rows.len() as u32,
                point_stride,
            )
        };
        assert_eq!(status, DIRECT_STATUS_OK);
        assert_eq!(&current_re[0..3], &[2.0, 4.0, 6.0]);
        assert_eq!(&current_im[0..3], &[1.0, 1.0, 1.0]);
        assert_eq!(&current_re[3..], &[-4.5, -5.5, -6.5, -7.5, -8.5, -9.5]);
        assert_eq!(&current_im[3..], &[3.5, 4.5, 5.5, 6.5, 7.5, 8.5]);
    }

    #[test]
    fn source_roles_remain_typed_source_ir_intrinsics() {
        let error = LoadedSymjitDirectExecutor::load_bytes(
            &direct_contribution_bytes(),
            PathBuf::from("source.symjit"),
            DIRECT_APPLICATION_STORAGE_ABI,
            DirectExecutorRole::Source,
            Vec::new(),
            Vec::new(),
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("SourceIR"));
    }

    #[test]
    fn authenticated_destination_operation_must_match_executor_role() {
        let error = LoadedSymjitDirectExecutor::load_bytes(
            &direct_closure_bytes(),
            PathBuf::from("closure-as-contribution.symjit"),
            DIRECT_APPLICATION_STORAGE_ABI,
            DirectExecutorRole::Contribution,
            closure_plane_projections(),
            scalar_projections(),
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("destination operation"));
    }

    #[test]
    fn prepared_direct_calls_do_not_use_packed_batch_evaluation() {
        let source = include_str!("symjit_direct.rs");
        let forbidden = [
            ["Eager", "KernelInput"].concat(),
            ["evaluate", "_batch("].concat(),
            ["scatter", "_outputs"].concat(),
        ];
        for name in forbidden {
            assert!(
                !source.contains(&name),
                "direct SymJIT adapter must not contain {name}"
            );
        }
    }
}
