// SPDX-License-Identifier: 0BSD

//! Eager-lane adapter for SymJIT table-aware Direct-Arena applications.
//!
//! This module owns no eager scheduler semantics. It is the narrow load and
//! call boundary shared by future eager invocation/fanout events: immutable
//! fixed-width rows select persistent split-complex arena planes, while the
//! generated table callable evaluates rows outermost and stores every output
//! directly into its ordered destinations.

// This bounded milestone is intentionally not wired into the eager scheduler yet.
#![allow(dead_code)]

use crate::artifact::PinnedNativeLibrary;
use crate::direct_arena::{DirectArenaView, DirectArenaWorkspace};
use crate::eager_layout::EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY;
pub(crate) use crate::eager_layout::{
    EAGER_DIRECT_SOURCE_APPLICATION_ABI, EAGER_DIRECT_TABLE_BINDING_ABI,
    EAGER_DIRECT_TABLE_DESCRIPTOR_ABI, EAGER_NATIVE_DIRECT_TABLE_APPLICATION_ABI,
};
use crate::{RusticolError, RusticolResult};
use std::any::Any;
#[cfg(feature = "f64-compiled")]
use std::ffi::CStr;
#[cfg(feature = "f64-compiled")]
use std::os::raw::c_char;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::Arc;
use symjit::{
    Application, Config, DIRECT_STATUS_EXECUTION_FAILED, DIRECT_STATUS_INVALID_ARGUMENT,
    DIRECT_STATUS_INVALID_CONTEXT, DIRECT_STATUS_OK, Defuns, DirectPlane, DirectScalar,
    DirectTableApplication, DirectTableApplicationMetadata, DirectTableAttachmentLayout,
    DirectTableCallViewV1, DirectTableCallable, DirectTableInvocationLayout,
    DirectTableParameterBinding, Storage,
};

pub(crate) const EAGER_DIRECT_ARENA_CAPABILITY: &str = EAGER_DIRECT_ARENA_RUNTIME_CAPABILITY;

/// Build the fixed-width row descriptor used by eager invocation/fanout
/// Direct-Arena calls.
///
/// Prepared eager applications expose every logical complex input as two
/// scalar source parameters. The direct table binds those split components to
/// invocation-selected planes. Row fields are tightly packed little-endian
/// `u32` values in the following order:
///
/// - invocation: all split input-plane IDs, attachment start, attachment count;
/// - attachment: all split destination-plane IDs, factor ID, operation.
pub(crate) fn eager_direct_table_metadata(
    input_complex_count: u32,
    output_complex_count: u32,
) -> RusticolResult<DirectTableApplicationMetadata> {
    if input_complex_count == 0 || output_complex_count == 0 {
        return Err(RusticolError::invalid_argument(
            "eager direct-table input and output widths must be positive",
        ));
    }
    let input_plane_count = input_complex_count.checked_mul(2).ok_or_else(|| {
        RusticolError::invalid_argument("eager direct-table input width overflows")
    })?;
    let output_plane_count = output_complex_count.checked_mul(2).ok_or_else(|| {
        RusticolError::invalid_argument("eager direct-table output width overflows")
    })?;
    let invocation_stride = input_plane_count
        .checked_add(2)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| {
            RusticolError::invalid_argument("eager direct-table invocation row overflows")
        })?;
    let attachment_stride = output_plane_count
        .checked_add(2)
        .and_then(|count| count.checked_mul(4))
        .ok_or_else(|| {
            RusticolError::invalid_argument("eager direct-table attachment row overflows")
        })?;
    let invocation = DirectTableInvocationLayout::new(
        invocation_stride,
        (0..input_plane_count).map(|field| field * 4).collect(),
        input_plane_count * 4,
        (input_plane_count + 1) * 4,
    )
    .map_err(|error| {
        RusticolError::invalid_argument(format!(
            "could not construct eager direct-table invocation layout: {error}"
        ))
    })?;
    let attachment = DirectTableAttachmentLayout::new_with_destination_plane_offsets(
        attachment_stride,
        (0..output_plane_count).map(|field| field * 4).collect(),
        output_plane_count * 4,
        (output_plane_count + 1) * 4,
    )
    .map_err(|error| {
        RusticolError::invalid_argument(format!(
            "could not construct eager direct-table attachment layout: {error}"
        ))
    })?;
    DirectTableApplicationMetadata::new_with_parameter_bindings(
        invocation,
        attachment,
        (0..input_plane_count)
            .map(DirectTableParameterBinding::Plane)
            .collect(),
        0,
    )
    .map_err(|error| {
        RusticolError::invalid_argument(format!(
            "could not construct eager direct-table metadata: {error}"
        ))
    })
}

/// Derive and bind the portable table descriptor for one authenticated eager
/// source application. This is a cold developer-cutover helper; the resulting
/// descriptor and lowered callable are retained for all warmed evaluations.
pub(crate) fn eager_direct_descriptor_for_source_application_bytes(
    source_bytes: &[u8],
    input_complex_count: u32,
    output_complex_count: u32,
    display_path: &Path,
) -> RusticolResult<Vec<u8>> {
    let mut config = Config::default();
    config.set_defuns(Defuns::new());
    let mut input = source_bytes;
    let source = guard_symjit_panic(
        || Application::load(&mut input, &config),
        display_path,
        "load prepared source for descriptor derivation",
    )?
    .map_err(|error| {
        RusticolError::compatibility(format!(
            "could not load eager Direct-Arena source {} for descriptor derivation: {error}",
            display_path.display()
        ))
    })?;
    if !input.is_empty() {
        return Err(RusticolError::integrity(format!(
            "eager Direct-Arena source {} has {} trailing bytes",
            display_path.display(),
            input.len()
        )));
    }
    eager_direct_table_metadata(input_complex_count, output_complex_count)?
        .encode_descriptor(&source)
        .map_err(|error| {
            RusticolError::integrity(format!(
                "could not derive eager Direct-Arena descriptor for {}: {error}",
                display_path.display()
            ))
        })
}

/// One persistent SymJIT plane-catalog entry backed by the eager arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum EagerDirectArenaPlaneBinding {
    CurrentReal(u32),
    CurrentImag(u32),
    AmplitudeReal(u32),
    AmplitudeImag(u32),
}

/// Immutable invocation and attachment tables for one eager callable.
#[derive(Debug)]
pub(crate) struct EagerDirectTableRows {
    invocations: Box<[u8]>,
    invocation_count: u32,
    attachments: Box<[u8]>,
    attachment_count: u32,
}

impl EagerDirectTableRows {
    pub(crate) fn invocation_count(&self) -> u32 {
        self.invocation_count
    }

    pub(crate) fn attachment_count(&self) -> u32 {
        self.attachment_count
    }
}

/// Persistent eager Direct-Arena storage and its stable descriptor catalogs.
///
/// Plane/scalar descriptors are built once. Updating scalar or factor values
/// mutates their stable backing boxes and never rebuilds a descriptor.
pub(crate) struct EagerDirectTableWorkspace {
    arena: DirectArenaWorkspace,
    planes: Box<[DirectPlane]>,
    scalar_values: Box<[f64]>,
    scalars: Box<[DirectScalar]>,
    factor_re: Box<[f64]>,
    factor_im: Box<[f64]>,
}

impl EagerDirectTableWorkspace {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        current_plane_count: u32,
        amplitude_plane_count: u32,
        tile_capacity: u32,
        plane_bindings: &[EagerDirectArenaPlaneBinding],
        scalar_values: Vec<f64>,
        factor_re: Vec<f64>,
        factor_im: Vec<f64>,
    ) -> RusticolResult<Self> {
        if plane_bindings.is_empty() {
            return Err(RusticolError::invalid_argument(
                "eager direct-table plane catalog must not be empty",
            ));
        }
        if factor_re.is_empty() || factor_re.len() != factor_im.len() {
            return Err(RusticolError::invalid_argument(
                "eager direct-table factors must be nonempty split-complex pairs",
            ));
        }
        let plane_count = u32::try_from(plane_bindings.len()).map_err(|_| {
            RusticolError::invalid_argument("eager direct-table plane count exceeds u32")
        })?;
        let scalar_count = u32::try_from(scalar_values.len()).map_err(|_| {
            RusticolError::invalid_argument("eager direct-table scalar count exceeds u32")
        })?;
        let factor_count = u32::try_from(factor_re.len()).map_err(|_| {
            RusticolError::invalid_argument("eager direct-table factor count exceeds u32")
        })?;
        if plane_count == 0 || factor_count == 0 {
            return Err(RusticolError::invalid_argument(
                "eager direct-table catalogs must not be empty",
            ));
        }

        let mut arena =
            DirectArenaWorkspace::new(current_plane_count, amplitude_plane_count, tile_capacity)?;
        // Binding descriptors needs a validated view. The caller chooses the
        // semantic active tail later without changing any pointer or pitch.
        arena.begin_tile(tile_capacity)?;
        let view = arena.view()?;
        let mut planes = Vec::new();
        planes
            .try_reserve_exact(plane_bindings.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not reserve eager direct-table plane catalog: {error}"
                ))
            })?;
        for &binding in plane_bindings {
            planes.push(bind_arena_plane(view, binding)?);
        }

        let scalar_values = scalar_values.into_boxed_slice();
        let mut scalars = Vec::new();
        scalars
            .try_reserve_exact(scalar_values.len())
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not reserve eager direct-table scalar catalog: {error}"
                ))
            })?;
        for value in scalar_values.iter() {
            // The boxed scalar values keep this address stable for the life of
            // the workspace.
            scalars.push(unsafe { DirectScalar::from_raw(ptr::from_ref(value)) });
        }
        debug_assert_eq!(scalar_count as usize, scalars.len());

        Ok(Self {
            arena,
            planes: planes.into_boxed_slice(),
            scalar_values,
            scalars: scalars.into_boxed_slice(),
            factor_re: factor_re.into_boxed_slice(),
            factor_im: factor_im.into_boxed_slice(),
        })
    }

    pub(crate) fn begin_tile(&mut self, point_count: u32) -> RusticolResult<()> {
        self.arena.begin_tile(point_count)
    }

    pub(crate) fn arena(&self) -> &DirectArenaWorkspace {
        &self.arena
    }

    pub(crate) fn split_arena_slices_mut(
        &mut self,
    ) -> (&mut [f64], &mut [f64], &mut [f64], &mut [f64]) {
        self.arena.split_slices_mut()
    }

    pub(crate) fn clear_current_active(
        &mut self,
        component_base: u32,
        component_count: u32,
    ) -> RusticolResult<()> {
        self.arena
            .clear_current_active(component_base, component_count)
    }

    pub(crate) fn clear_amplitude_active(
        &mut self,
        component_base: u32,
        component_count: u32,
    ) -> RusticolResult<()> {
        self.arena
            .clear_amplitude_active(component_base, component_count)
    }

    pub(crate) fn scalar_values_mut(&mut self) -> &mut [f64] {
        &mut self.scalar_values
    }

    pub(crate) fn factors_mut(&mut self) -> (&mut [f64], &mut [f64]) {
        (&mut self.factor_re, &mut self.factor_im)
    }

    fn view(
        &self,
        rows: &EagerDirectTableRows,
        invocation_stride: u32,
        attachment_stride: u32,
        point_start: u32,
        point_count: u32,
    ) -> RusticolResult<DirectTableCallViewV1> {
        if point_count == 0 {
            return Err(RusticolError::invalid_argument(
                "eager direct-table point count must be positive",
            ));
        }
        let point_stop = point_start.checked_add(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("eager direct-table point range overflows u32")
        })?;
        if point_stop > self.arena.active_point_count() {
            return Err(RusticolError::invalid_argument(format!(
                "eager direct-table point range {point_start}..{point_stop} exceeds active tile {}",
                self.arena.active_point_count()
            )));
        }
        Ok(DirectTableCallViewV1 {
            invocations: rows.invocations.as_ptr(),
            invocation_count: rows.invocation_count,
            invocation_stride,
            attachments: rows.attachments.as_ptr(),
            attachment_count: rows.attachment_count,
            attachment_stride,
            planes: self.planes.as_ptr(),
            plane_count: u32::try_from(self.planes.len()).map_err(|_| {
                RusticolError::internal("eager direct-table plane catalog exceeds u32")
            })?,
            scalar_count: u32::try_from(self.scalars.len()).map_err(|_| {
                RusticolError::internal("eager direct-table scalar catalog exceeds u32")
            })?,
            scalars: if self.scalars.is_empty() {
                ptr::null()
            } else {
                self.scalars.as_ptr()
            },
            scale_re: self.factor_re.as_ptr(),
            scale_im: self.factor_im.as_ptr(),
            scale_count: u32::try_from(self.factor_re.len()).map_err(|_| {
                RusticolError::internal("eager direct-table factor catalog exceeds u32")
            })?,
            point_start,
            point_count,
        })
    }
}

/// Loaded owner of one prepared eager table-aware callable.
pub(crate) struct LoadedSymjitEagerDirectTable {
    callable: EagerDirectTableCallable,
    display_path: PathBuf,
}

enum EagerDirectTableCallable {
    Symjit(DirectTableCallable),
    #[cfg(feature = "f64-compiled")]
    Native(LoadedNativeEagerDirectTable),
}

#[cfg(feature = "f64-compiled")]
type NativeEagerDirectTableCall = unsafe extern "C" fn(*const DirectTableCallViewV1) -> i32;

#[cfg(feature = "f64-compiled")]
#[repr(C)]
struct NativeEagerDirectTableMetadataV1 {
    struct_size: u32,
    abi_version: u32,
    flags: u32,
    invocation_stride: u32,
    attachment_stride: u32,
    input_complex_count: u32,
    output_complex_count: u32,
    simd_lane_width: u32,
    application_abi: *const c_char,
    function_name: *const c_char,
    target_triple: *const c_char,
    evaluator_state_sha256: *const c_char,
}

#[cfg(feature = "f64-compiled")]
type NativeEagerDirectTableMetadataCall =
    unsafe extern "C" fn() -> *const NativeEagerDirectTableMetadataV1;

#[cfg(feature = "f64-compiled")]
const NATIVE_EAGER_DIRECT_TABLE_METADATA_VERSION: u32 = 1;
#[cfg(feature = "f64-compiled")]
const NATIVE_EAGER_DIRECT_TABLE_REQUIRED_FLAGS: u32 = 0x1f;

#[cfg(feature = "f64-compiled")]
struct LoadedNativeEagerDirectTable {
    _library: Arc<PinnedNativeLibrary>,
    call: NativeEagerDirectTableCall,
    metadata: DirectTableApplicationMetadata,
}

impl LoadedSymjitEagerDirectTable {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_prepared_application_bytes(
        source_bytes: &[u8],
        descriptor_bytes: &[u8],
        display_path: PathBuf,
        source_application_abi: &str,
        descriptor_abi: &str,
        binding_abi: &str,
    ) -> RusticolResult<Self> {
        if source_application_abi != EAGER_DIRECT_SOURCE_APPLICATION_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported eager Direct-Arena source application ABI \
                 {source_application_abi:?}; expected {EAGER_DIRECT_SOURCE_APPLICATION_ABI:?}"
            )));
        }
        if descriptor_abi != EAGER_DIRECT_TABLE_DESCRIPTOR_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported eager Direct-Arena descriptor ABI {descriptor_abi:?}; \
                 expected {EAGER_DIRECT_TABLE_DESCRIPTOR_ABI:?}"
            )));
        }
        if binding_abi != EAGER_DIRECT_TABLE_BINDING_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported eager Direct-Arena binding ABI {binding_abi:?}; \
                 expected {EAGER_DIRECT_TABLE_BINDING_ABI:?}"
            )));
        }

        let mut config = Config::default();
        config.set_defuns(Defuns::new());
        let mut input = source_bytes;
        let source = guard_symjit_panic(
            || Application::load(&mut input, &config),
            &display_path,
            "load prepared source",
        )?
        .map_err(|error| {
            RusticolError::compatibility(format!(
                "could not load eager Direct-Arena prepared source {}: {error}",
                display_path.display()
            ))
        })?;
        if !input.is_empty() {
            return Err(RusticolError::integrity(format!(
                "eager Direct-Arena prepared source {} has {} trailing bytes",
                display_path.display(),
                input.len()
            )));
        }

        let application = guard_symjit_panic(
            || DirectTableApplication::from_descriptor(source, descriptor_bytes),
            &display_path,
            "validate descriptor and lower table application",
        )?
        .map_err(|error| {
            RusticolError::integrity(format!(
                "could not construct eager Direct-Arena table application {}: {error}",
                display_path.display()
            ))
        })?;
        let applet = guard_symjit_panic(
            || application.seal(),
            &display_path,
            "seal table application",
        )?
        .map_err(|error| {
            RusticolError::compatibility(format!(
                "could not seal eager Direct-Arena table application {}: {error}",
                display_path.display()
            ))
        })?;
        Ok(Self {
            callable: EagerDirectTableCallable::Symjit(applet.into_callable()),
            display_path,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_native_application(
        library: Arc<PinnedNativeLibrary>,
        function_name: &str,
        display_path: PathBuf,
        source_application_abi: &str,
        invocation_stride: u32,
        attachment_stride: u32,
        input_complex_count: u32,
        output_complex_count: u32,
        expected_target_triple: &str,
        expected_evaluator_state_sha256: &str,
        expected_simd_lane_width: u32,
    ) -> RusticolResult<Self> {
        if source_application_abi != EAGER_NATIVE_DIRECT_TABLE_APPLICATION_ABI {
            return Err(RusticolError::compatibility(format!(
                "unsupported native eager DirectTable application ABI \
                 {source_application_abi:?}; expected \
                 {EAGER_NATIVE_DIRECT_TABLE_APPLICATION_ABI:?}"
            )));
        }
        validate_c_symbol(function_name)?;
        let metadata = eager_direct_table_metadata(input_complex_count, output_complex_count)?;
        if metadata.invocation.row_stride != invocation_stride
            || metadata.attachment.row_stride != attachment_stride
        {
            return Err(RusticolError::integrity(format!(
                "native eager DirectTable {} row strides ({invocation_stride}, \
                 {attachment_stride}) do not match ({}, {})",
                display_path.display(),
                metadata.invocation.row_stride,
                metadata.attachment.row_stride,
            )));
        }
        #[cfg(not(feature = "f64-compiled"))]
        {
            let _ = (
                library,
                function_name,
                metadata,
                expected_target_triple,
                expected_evaluator_state_sha256,
                expected_simd_lane_width,
            );
            return Err(RusticolError::compatibility(
                "native eager DirectTable execution requires the f64-compiled feature",
            ));
        }
        #[cfg(feature = "f64-compiled")]
        {
            let library_path = library.display_path();
            let metadata_symbol_name = format!("{function_name}_metadata_v1");
            let metadata_call = unsafe {
                library
                    .get::<NativeEagerDirectTableMetadataCall>(metadata_symbol_name.as_bytes())
                    .map(|symbol| *symbol)
                    .map_err(|error| {
                        RusticolError::compatibility(format!(
                            "could not load native eager DirectTable metadata symbol \
                             {metadata_symbol_name:?} from {}: {error}",
                            library_path.display()
                        ))
                    })?
            };
            let exported = unsafe { metadata_call() };
            if exported.is_null() {
                return Err(RusticolError::integrity(format!(
                    "native eager DirectTable {} returned null ABI metadata",
                    display_path.display()
                )));
            }
            let exported = unsafe { &*exported };
            let expected_struct_size =
                u32::try_from(std::mem::size_of::<NativeEagerDirectTableMetadataV1>())
                    .expect("native eager metadata size fits u32");
            if exported.struct_size != expected_struct_size
                || exported.abi_version != NATIVE_EAGER_DIRECT_TABLE_METADATA_VERSION
                || exported.flags != NATIVE_EAGER_DIRECT_TABLE_REQUIRED_FLAGS
                || exported.invocation_stride != invocation_stride
                || exported.attachment_stride != attachment_stride
                || exported.input_complex_count != input_complex_count
                || exported.output_complex_count != output_complex_count
                || exported.simd_lane_width != expected_simd_lane_width
            {
                return Err(RusticolError::compatibility(format!(
                    "native eager DirectTable {} exports incompatible ABI metadata",
                    display_path.display()
                )));
            }
            for (label, pointer, expected) in [
                (
                    "application ABI",
                    exported.application_abi,
                    source_application_abi,
                ),
                ("function name", exported.function_name, function_name),
                (
                    "target triple",
                    exported.target_triple,
                    expected_target_triple,
                ),
                (
                    "evaluator-state digest",
                    exported.evaluator_state_sha256,
                    expected_evaluator_state_sha256,
                ),
            ] {
                let actual = native_metadata_text(pointer, label, &display_path)?;
                if actual != expected {
                    return Err(RusticolError::compatibility(format!(
                        "native eager DirectTable {} {label} is {actual:?}, expected {expected:?}",
                        display_path.display()
                    )));
                }
            }
            let call = unsafe {
                library
                    .get::<NativeEagerDirectTableCall>(function_name.as_bytes())
                    .map(|symbol| *symbol)
                    .map_err(|error| {
                        RusticolError::evaluation(format!(
                            "could not load native eager DirectTable symbol \
                             {function_name:?} from {}: {error}",
                            library_path.display()
                        ))
                    })?
            };
            Ok(Self {
                callable: EagerDirectTableCallable::Native(LoadedNativeEagerDirectTable {
                    _library: library,
                    call,
                    metadata,
                }),
                display_path,
            })
        }
    }

    fn metadata(&self) -> &DirectTableApplicationMetadata {
        match &self.callable {
            EagerDirectTableCallable::Symjit(callable) => callable.metadata(),
            #[cfg(feature = "f64-compiled")]
            EagerDirectTableCallable::Native(callable) => &callable.metadata,
        }
    }

    pub(crate) fn load_rows(
        &self,
        invocation_bytes: Vec<u8>,
        attachment_bytes: Vec<u8>,
    ) -> RusticolResult<EagerDirectTableRows> {
        let metadata = self.metadata();
        let invocation_count = table_row_count(
            &invocation_bytes,
            metadata.invocation.row_stride,
            "invocation",
            false,
        )?;
        let attachment_count = table_row_count(
            &attachment_bytes,
            metadata.attachment.row_stride,
            "attachment",
            true,
        )?;
        Ok(EagerDirectTableRows {
            invocations: invocation_bytes.into_boxed_slice(),
            invocation_count,
            attachments: attachment_bytes.into_boxed_slice(),
            attachment_count,
        })
    }

    /// Validate rows, catalogs, aliasing, and the complete active point range
    /// without executing the callable.
    pub(crate) fn validate_call(
        &self,
        rows: &EagerDirectTableRows,
        workspace: &EagerDirectTableWorkspace,
        point_start: u32,
        point_count: u32,
    ) -> RusticolResult<()> {
        let view = self.call_view(rows, workspace, point_start, point_count)?;
        guard_symjit_panic(
            || unsafe { self.metadata().validate_call_view(&view) },
            &self.display_path,
            "validate table call",
        )?
        .map_err(|error| {
            RusticolError::integrity(format!(
                "invalid eager Direct-Arena table call {}: {error}",
                self.display_path.display()
            ))
        })
    }

    /// Validate and execute a table call through SymJIT's typed status
    /// trampoline. Constructed callables retain authenticated immutable
    /// metadata, so this checked warmed path does not rebuild metadata.
    pub(crate) fn evaluate_checked(
        &self,
        rows: &EagerDirectTableRows,
        workspace: &mut EagerDirectTableWorkspace,
        point_start: u32,
        point_count: u32,
    ) -> RusticolResult<()> {
        let view = self.call_view(rows, workspace, point_start, point_count)?;
        let status = guard_symjit_panic(
            || unsafe {
                match &self.callable {
                    EagerDirectTableCallable::Symjit(callable) => callable.handle().invoke(&view),
                    #[cfg(feature = "f64-compiled")]
                    EagerDirectTableCallable::Native(callable) => (callable.call)(&view),
                }
            },
            &self.display_path,
            "execute checked table call",
        )?;
        map_direct_status(status, &self.display_path)
    }

    /// Execute a call whose immutable rows/catalogs were authenticated by
    /// [`Self::validate_call`].
    ///
    /// Range checks remain in Rust so a tail tile cannot exceed the persistent
    /// arena. SymJIT row and alias validation is intentionally skipped.
    ///
    /// # Safety
    ///
    /// The caller must not replace the rows, plane catalog, scalar
    /// descriptors, or factor backing storage after the corresponding
    /// successful validation. Scalar and factor values may be updated in
    /// place.
    pub(crate) unsafe fn evaluate_validated_unchecked(
        &self,
        rows: &EagerDirectTableRows,
        workspace: &mut EagerDirectTableWorkspace,
        point_start: u32,
        point_count: u32,
    ) -> RusticolResult<()> {
        let view = self.call_view(rows, workspace, point_start, point_count)?;
        let status = guard_symjit_panic(
            || unsafe {
                match &self.callable {
                    EagerDirectTableCallable::Symjit(callable) => callable.invoke_unchecked(&view),
                    #[cfg(feature = "f64-compiled")]
                    EagerDirectTableCallable::Native(callable) => (callable.call)(&view),
                }
            },
            &self.display_path,
            "execute validated table call",
        )?;
        map_direct_status(status, &self.display_path)
    }

    fn call_view(
        &self,
        rows: &EagerDirectTableRows,
        workspace: &EagerDirectTableWorkspace,
        point_start: u32,
        point_count: u32,
    ) -> RusticolResult<DirectTableCallViewV1> {
        let metadata = self.metadata();
        if workspace.scalars.len() != metadata.scalar_input_count as usize {
            return Err(RusticolError::integrity(format!(
                "eager Direct-Arena scalar catalog has {} entries, expected {}",
                workspace.scalars.len(),
                metadata.scalar_input_count
            )));
        }
        workspace.view(
            rows,
            metadata.invocation.row_stride,
            metadata.attachment.row_stride,
            point_start,
            point_count,
        )
    }
}

#[cfg(feature = "f64-compiled")]
fn native_metadata_text(
    pointer: *const c_char,
    label: &str,
    display_path: &Path,
) -> RusticolResult<String> {
    if pointer.is_null() {
        return Err(RusticolError::integrity(format!(
            "native eager DirectTable {} has a null {label}",
            display_path.display()
        )));
    }
    unsafe { CStr::from_ptr(pointer) }
        .to_str()
        .map(str::to_owned)
        .map_err(|error| {
            RusticolError::integrity(format!(
                "native eager DirectTable {} has non-UTF-8 {label}: {error}",
                display_path.display()
            ))
        })
}

fn validate_c_symbol(symbol: &str) -> RusticolResult<()> {
    let mut bytes = symbol.bytes();
    let Some(first) = bytes.next() else {
        return Err(RusticolError::artifact(
            "native eager DirectTable symbol must not be empty",
        ));
    };
    if !(first == b'_' || first.is_ascii_alphabetic())
        || bytes.any(|byte| !(byte == b'_' || byte.is_ascii_alphanumeric()))
    {
        return Err(RusticolError::artifact(format!(
            "native eager DirectTable symbol {symbol:?} is not a portable C identifier"
        )));
    }
    Ok(())
}

fn bind_arena_plane(
    arena: DirectArenaView,
    binding: EagerDirectArenaPlaneBinding,
) -> RusticolResult<DirectPlane> {
    let (base, shape, component, label) = match binding {
        EagerDirectArenaPlaneBinding::CurrentReal(component) => (
            arena.current_re,
            arena.current_shape()?,
            component,
            "current real",
        ),
        EagerDirectArenaPlaneBinding::CurrentImag(component) => (
            arena.current_im,
            arena.current_shape()?,
            component,
            "current imaginary",
        ),
        EagerDirectArenaPlaneBinding::AmplitudeReal(component) => (
            arena.amplitude_re,
            arena.amplitude_shape()?,
            component,
            "amplitude real",
        ),
        EagerDirectArenaPlaneBinding::AmplitudeImag(component) => (
            arena.amplitude_im,
            arena.amplitude_shape()?,
            component,
            "amplitude imaginary",
        ),
    };
    let range = shape.checked_component_range(component, 1, label)?;
    let len = range.len();
    let values = unsafe { base.add(range.start) };
    Ok(unsafe { DirectPlane::from_raw_parts(values, len) })
}

fn table_row_count(
    bytes: &[u8],
    row_stride: u32,
    label: &str,
    allow_empty: bool,
) -> RusticolResult<u32> {
    let stride = usize::try_from(row_stride)
        .map_err(|_| RusticolError::integrity(format!("{label} row stride exceeds usize")))?;
    if stride == 0 || !bytes.len().is_multiple_of(stride) {
        return Err(RusticolError::integrity(format!(
            "eager Direct-Arena {label} table has {} bytes, not a whole number of {stride}-byte rows",
            bytes.len()
        )));
    }
    let count = u32::try_from(bytes.len() / stride).map_err(|_| {
        RusticolError::integrity(format!("eager Direct-Arena {label} row count exceeds u32"))
    })?;
    if count == 0 && !allow_empty {
        return Err(RusticolError::integrity(format!(
            "eager Direct-Arena {label} table must not be empty"
        )));
    }
    Ok(count)
}

fn map_direct_status(status: i32, path: &Path) -> RusticolResult<()> {
    match status {
        DIRECT_STATUS_OK => Ok(()),
        DIRECT_STATUS_INVALID_ARGUMENT => Err(RusticolError::integrity(format!(
            "eager Direct-Arena callable {} rejected its authenticated call view",
            path.display()
        ))),
        DIRECT_STATUS_INVALID_CONTEXT => Err(RusticolError::internal(format!(
            "eager Direct-Arena callable {} has an invalid context",
            path.display()
        ))),
        DIRECT_STATUS_EXECUTION_FAILED => Err(RusticolError::evaluation(format!(
            "eager Direct-Arena callable {} failed during generated execution",
            path.display()
        ))),
        other => Err(RusticolError::evaluation(format!(
            "eager Direct-Arena callable {} returned unknown status {other}",
            path.display()
        ))),
    }
}

fn guard_symjit_panic<T>(
    operation: impl FnOnce() -> T,
    path: &Path,
    action: &str,
) -> RusticolResult<T> {
    catch_unwind(AssertUnwindSafe(operation)).map_err(|payload| {
        RusticolError::compatibility(format!(
            "SymJIT panicked while trying to {action} eager Direct-Arena application {}: {}",
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
    #[cfg(not(feature = "f64-symjit"))]
    use super::super::native_direct::tests::count_allocations;
    #[cfg(feature = "f64-symjit")]
    use super::super::symjit_direct::tests::count_allocations;
    use super::*;
    use std::hint::black_box;
    use std::mem;
    use std::time::Instant;
    use symjit::{
        Applet, Compiler, Complex, DirectTableApplicationMetadata, DirectTableAttachmentLayout,
        DirectTableInvocationLayout, DirectTableParameterBinding, Expr,
    };

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct InvocationRow {
        x_re: u32,
        x_im: u32,
        y_re: u32,
        y_im: u32,
        attachment_start: u32,
        attachment_count: u32,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct AttachmentRow {
        destination_re: u32,
        destination_im: u32,
        factor: u32,
        operation: u32,
    }

    fn source_application() -> Application {
        let mut config = Config::default();
        config.set_opt_level(2);
        config.set_complex(true);
        config.set_symbolica(true);
        config.set_simd(true);
        config.set_fast_complex(false);
        let x = Expr::var("x");
        let y = Expr::var("y");
        Compiler::with_config(config)
            .compile_params(&[], &[&x + &y], &[x, y])
            .unwrap()
    }

    fn ordinary_applet() -> Applet {
        let mut source = source_application();
        source.prepare_simd();
        source.seal().unwrap()
    }

    fn table_fixture() -> (
        LoadedSymjitEagerDirectTable,
        EagerDirectTableRows,
        EagerDirectTableWorkspace,
    ) {
        let source = source_application();
        let metadata = DirectTableApplicationMetadata::new_with_parameter_bindings(
            DirectTableInvocationLayout::new(
                mem::size_of::<InvocationRow>() as u32,
                vec![
                    mem::offset_of!(InvocationRow, x_re) as u32,
                    mem::offset_of!(InvocationRow, x_im) as u32,
                    mem::offset_of!(InvocationRow, y_re) as u32,
                    mem::offset_of!(InvocationRow, y_im) as u32,
                ],
                mem::offset_of!(InvocationRow, attachment_start) as u32,
                mem::offset_of!(InvocationRow, attachment_count) as u32,
            )
            .unwrap(),
            DirectTableAttachmentLayout::new(
                mem::size_of::<AttachmentRow>() as u32,
                mem::offset_of!(AttachmentRow, destination_re) as u32,
                mem::offset_of!(AttachmentRow, destination_im) as u32,
                mem::offset_of!(AttachmentRow, factor) as u32,
                mem::offset_of!(AttachmentRow, operation) as u32,
            )
            .unwrap(),
            (0..4).map(DirectTableParameterBinding::Plane).collect(),
            0,
        )
        .unwrap();
        let descriptor = metadata.encode_descriptor(&source).unwrap();
        let mut source_bytes = Vec::new();
        source.save(&mut source_bytes).unwrap();
        let application = LoadedSymjitEagerDirectTable::load_prepared_application_bytes(
            &source_bytes,
            &descriptor,
            PathBuf::from("eager-table-test.symjit"),
            EAGER_DIRECT_SOURCE_APPLICATION_ABI,
            EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
            EAGER_DIRECT_TABLE_BINDING_ABI,
        )
        .unwrap();

        // Row zero fans out x+y twice. Row one then consumes both earlier
        // destinations, proving that ordered cross-row aliases remain valid.
        let invocations = [
            InvocationRow {
                x_re: 0,
                x_im: 1,
                y_re: 2,
                y_im: 3,
                attachment_start: 0,
                attachment_count: 2,
            },
            InvocationRow {
                x_re: 4,
                x_im: 5,
                y_re: 6,
                y_im: 7,
                attachment_start: 2,
                attachment_count: 1,
            },
        ];
        let attachments = [
            AttachmentRow {
                destination_re: 4,
                destination_im: 5,
                factor: 0,
                operation: 0,
            },
            AttachmentRow {
                destination_re: 6,
                destination_im: 7,
                factor: 1,
                operation: 0,
            },
            AttachmentRow {
                destination_re: 8,
                destination_im: 9,
                factor: 0,
                operation: 0,
            },
        ];
        let rows = application
            .load_rows(rows_as_bytes(&invocations), rows_as_bytes(&attachments))
            .unwrap();
        let bindings = [
            EagerDirectArenaPlaneBinding::CurrentReal(0),
            EagerDirectArenaPlaneBinding::CurrentImag(0),
            EagerDirectArenaPlaneBinding::CurrentReal(1),
            EagerDirectArenaPlaneBinding::CurrentImag(1),
            EagerDirectArenaPlaneBinding::CurrentReal(2),
            EagerDirectArenaPlaneBinding::CurrentImag(2),
            EagerDirectArenaPlaneBinding::CurrentReal(3),
            EagerDirectArenaPlaneBinding::CurrentImag(3),
            EagerDirectArenaPlaneBinding::AmplitudeReal(0),
            EagerDirectArenaPlaneBinding::AmplitudeImag(0),
        ];
        let workspace = EagerDirectTableWorkspace::new(
            4,
            1,
            7,
            &bindings,
            Vec::new(),
            vec![1.0, 0.5],
            vec![0.0, 0.0],
        )
        .unwrap();
        (application, rows, workspace)
    }

    struct GatherCallScatterScratch {
        inputs: Vec<Complex<f64>>,
        outputs: Vec<Complex<f64>>,
    }

    impl GatherCallScatterScratch {
        fn new() -> Self {
            Self {
                inputs: vec![Complex::new(0.0, 0.0); 14],
                outputs: vec![Complex::new(0.0, 0.0); 7],
            }
        }
    }

    fn gather_call_scatter(
        applet: &Applet,
        workspace: &mut EagerDirectTableWorkspace,
        scratch: &mut GatherCallScatterScratch,
    ) {
        let stride = workspace.arena().point_stride() as usize;
        let (current_re, current_im) = workspace.arena().current_slices();
        for point in 0..7 {
            scratch.inputs[2 * point] = Complex::new(current_re[point], current_im[point]);
            scratch.inputs[2 * point + 1] =
                Complex::new(current_re[stride + point], current_im[stride + point]);
        }
        applet.evaluate_matrix(&scratch.inputs, &mut scratch.outputs, 7);
        {
            let (current_re, current_im, _, _) = workspace.split_arena_slices_mut();
            for point in 0..7 {
                let output = scratch.outputs[point];
                current_re[2 * stride + point] = output.re;
                current_im[2 * stride + point] = output.im;
                current_re[3 * stride + point] = 0.5 * output.re;
                current_im[3 * stride + point] = 0.5 * output.im;
            }
        }

        let (current_re, current_im) = workspace.arena().current_slices();
        for point in 0..7 {
            scratch.inputs[2 * point] = Complex::new(
                current_re[2 * stride + point],
                current_im[2 * stride + point],
            );
            scratch.inputs[2 * point + 1] = Complex::new(
                current_re[3 * stride + point],
                current_im[3 * stride + point],
            );
        }
        applet.evaluate_matrix(&scratch.inputs, &mut scratch.outputs, 7);
        let (_, _, amplitude_re, amplitude_im) = workspace.split_arena_slices_mut();
        for point in 0..7 {
            amplitude_re[point] = scratch.outputs[point].re;
            amplitude_im[point] = scratch.outputs[point].im;
        }
    }

    fn assert_active_arena_bits_equal(
        actual: &EagerDirectTableWorkspace,
        expected: &EagerDirectTableWorkspace,
    ) {
        let stride = actual.arena().point_stride() as usize;
        assert_eq!(stride, expected.arena().point_stride() as usize);
        let (actual_current_re, actual_current_im) = actual.arena().current_slices();
        let (expected_current_re, expected_current_im) = expected.arena().current_slices();
        let (actual_amplitude_re, actual_amplitude_im) = actual.arena().amplitude_slices();
        let (expected_amplitude_re, expected_amplitude_im) = expected.arena().amplitude_slices();
        for component in 0..4 {
            for point in 0..7 {
                let index = component * stride + point;
                assert_eq!(
                    actual_current_re[index].to_bits(),
                    expected_current_re[index].to_bits()
                );
                assert_eq!(
                    actual_current_im[index].to_bits(),
                    expected_current_im[index].to_bits()
                );
            }
        }
        for point in 0..7 {
            assert_eq!(
                actual_amplitude_re[point].to_bits(),
                expected_amplitude_re[point].to_bits()
            );
            assert_eq!(
                actual_amplitude_im[point].to_bits(),
                expected_amplitude_im[point].to_bits()
            );
        }
    }

    fn rows_as_bytes<T: Copy>(rows: &[T]) -> Vec<u8> {
        let byte_len = mem::size_of_val(rows);
        unsafe { std::slice::from_raw_parts(rows.as_ptr().cast::<u8>(), byte_len) }.to_vec()
    }

    fn initialize_inputs(workspace: &mut EagerDirectTableWorkspace) {
        workspace.begin_tile(7).unwrap();
        let stride = workspace.arena().point_stride() as usize;
        let (current_re, current_im, amplitude_re, amplitude_im) =
            workspace.split_arena_slices_mut();
        for point in 0..7 {
            current_re[point] = 1.0 + point as f64;
            current_im[point] = -0.25 * point as f64;
            current_re[stride + point] = 10.0 + 2.0 * point as f64;
            current_im[stride + point] = 0.5 * point as f64;
        }
        current_re[2 * stride..4 * stride].fill(-99.0);
        current_im[2 * stride..4 * stride].fill(77.0);
        amplitude_re.fill(-55.0);
        amplitude_im.fill(44.0);
    }

    fn assert_outputs(workspace: &EagerDirectTableWorkspace) {
        let stride = workspace.arena().point_stride() as usize;
        let (current_re, current_im) = workspace.arena().current_slices();
        let (amplitude_re, amplitude_im) = workspace.arena().amplitude_slices();
        for point in 0..7 {
            let sum_re = 11.0 + 3.0 * point as f64;
            let sum_im = 0.25 * point as f64;
            assert_eq!(current_re[2 * stride + point], sum_re);
            assert_eq!(current_im[2 * stride + point], sum_im);
            assert_eq!(current_re[3 * stride + point], 0.5 * sum_re);
            assert_eq!(current_im[3 * stride + point], 0.5 * sum_im);
            assert_eq!(amplitude_re[point], 1.5 * sum_re);
            assert_eq!(amplitude_im[point], 1.5 * sum_im);
        }
        // Physical pitch is eight, while this test intentionally executes the
        // odd seven-point tail.
        assert_eq!(stride, 8);
        assert_eq!(amplitude_re[7], -55.0);
        assert_eq!(amplitude_im[7], 44.0);
    }

    #[test]
    fn table_adapter_executes_ordered_fanout_directly_into_odd_tail_arena() {
        let (application, rows, mut workspace) = table_fixture();
        assert_eq!(rows.invocation_count(), 2);
        assert_eq!(rows.attachment_count(), 3);
        initialize_inputs(&mut workspace);
        application.validate_call(&rows, &workspace, 0, 7).unwrap();
        unsafe {
            application
                .evaluate_validated_unchecked(&rows, &mut workspace, 0, 7)
                .unwrap();
        }
        assert_outputs(&workspace);
    }

    #[test]
    fn checked_call_rejects_same_row_input_destination_alias() {
        let (application, _rows, mut workspace) = table_fixture();
        let bad_invocation = [InvocationRow {
            x_re: 0,
            x_im: 1,
            y_re: 2,
            y_im: 3,
            attachment_start: 0,
            attachment_count: 1,
        }];
        let bad_attachment = [AttachmentRow {
            destination_re: 0,
            destination_im: 1,
            factor: 0,
            operation: 0,
        }];
        let rows = application
            .load_rows(
                rows_as_bytes(&bad_invocation),
                rows_as_bytes(&bad_attachment),
            )
            .unwrap();
        let error = application
            .validate_call(&rows, &workspace, 0, 7)
            .unwrap_err();
        assert!(error.message().contains("overlaps"));
        assert!(
            application
                .evaluate_checked(&rows, &mut workspace, 0, 7)
                .is_err()
        );
    }

    #[test]
    fn warmed_checked_and_validated_calls_allocate_zero() {
        let (application, rows, mut workspace) = table_fixture();
        initialize_inputs(&mut workspace);
        application
            .evaluate_checked(&rows, &mut workspace, 0, 7)
            .unwrap();
        application.validate_call(&rows, &workspace, 0, 7).unwrap();

        let (checked, checked_count, checked_bytes) =
            count_allocations(|| application.evaluate_checked(&rows, &mut workspace, 0, 7));
        checked.unwrap();
        assert_eq!((checked_count, checked_bytes), (0, 0));

        let (unchecked, unchecked_count, unchecked_bytes) = count_allocations(|| unsafe {
            application.evaluate_validated_unchecked(&rows, &mut workspace, 0, 7)
        });
        unchecked.unwrap();
        assert_eq!((unchecked_count, unchecked_bytes), (0, 0));
    }

    #[test]
    fn gather_call_scatter_oracle_matches_direct_and_allocates_zero() {
        let (application, rows, mut direct_workspace) = table_fixture();
        let (_, _, mut packet_workspace) = table_fixture();
        let applet = ordinary_applet();
        let mut scratch = GatherCallScatterScratch::new();
        initialize_inputs(&mut direct_workspace);
        initialize_inputs(&mut packet_workspace);
        application
            .validate_call(&rows, &direct_workspace, 0, 7)
            .unwrap();
        unsafe {
            application
                .evaluate_validated_unchecked(&rows, &mut direct_workspace, 0, 7)
                .unwrap();
        }
        gather_call_scatter(&applet, &mut packet_workspace, &mut scratch);
        assert_active_arena_bits_equal(&direct_workspace, &packet_workspace);

        let ((), packet_count, packet_bytes) = count_allocations(|| {
            for _ in 0..32 {
                gather_call_scatter(&applet, &mut packet_workspace, &mut scratch);
            }
        });
        assert_eq!((packet_count, packet_bytes), (0, 0));
        let ((), direct_count, direct_bytes) = count_allocations(|| {
            for _ in 0..32 {
                unsafe {
                    application
                        .evaluate_validated_unchecked(&rows, &mut direct_workspace, 0, 7)
                        .unwrap();
                }
            }
        });
        assert_eq!((direct_count, direct_bytes), (0, 0));
    }

    #[test]
    #[ignore = "local raw timing evidence; contains no timing acceptance assertion"]
    fn benchmark_direct_table_against_gather_call_scatter() {
        const SAMPLES: usize = 9;
        const REPETITIONS: usize = 20_000;

        let (application, rows, mut direct_workspace) = table_fixture();
        let (_, _, mut packet_workspace) = table_fixture();
        let applet = ordinary_applet();
        let mut scratch = GatherCallScatterScratch::new();
        initialize_inputs(&mut direct_workspace);
        initialize_inputs(&mut packet_workspace);
        application
            .validate_call(&rows, &direct_workspace, 0, 7)
            .unwrap();
        unsafe {
            application
                .evaluate_validated_unchecked(&rows, &mut direct_workspace, 0, 7)
                .unwrap();
        }
        gather_call_scatter(&applet, &mut packet_workspace, &mut scratch);
        assert_active_arena_bits_equal(&direct_workspace, &packet_workspace);

        let mut direct_ns = Vec::with_capacity(SAMPLES);
        let mut packet_ns = Vec::with_capacity(SAMPLES);
        for sample in 0..SAMPLES {
            if sample.is_multiple_of(2) {
                direct_ns.push(measure_direct(
                    &application,
                    &rows,
                    &mut direct_workspace,
                    REPETITIONS,
                ));
                packet_ns.push(measure_gather_call_scatter(
                    &applet,
                    &mut packet_workspace,
                    &mut scratch,
                    REPETITIONS,
                ));
            } else {
                packet_ns.push(measure_gather_call_scatter(
                    &applet,
                    &mut packet_workspace,
                    &mut scratch,
                    REPETITIONS,
                ));
                direct_ns.push(measure_direct(
                    &application,
                    &rows,
                    &mut direct_workspace,
                    REPETITIONS,
                ));
            }
        }
        direct_ns.sort_by(f64::total_cmp);
        packet_ns.sort_by(f64::total_cmp);
        let direct_median = direct_ns[SAMPLES / 2];
        let packet_median = packet_ns[SAMPLES / 2];
        eprintln!(
            "eager-direct-table raw benchmark: samples={SAMPLES} repetitions={REPETITIONS} \
             points=7 direct_median_ns/call={direct_median:.3} \
             gather_call_scatter_median_ns/call={packet_median:.3} \
             direct_over_packet={:.6}",
            direct_median / packet_median
        );
    }

    fn measure_direct(
        application: &LoadedSymjitEagerDirectTable,
        rows: &EagerDirectTableRows,
        workspace: &mut EagerDirectTableWorkspace,
        repetitions: usize,
    ) -> f64 {
        let start = Instant::now();
        for _ in 0..repetitions {
            unsafe {
                application
                    .evaluate_validated_unchecked(rows, workspace, 0, 7)
                    .unwrap();
            }
        }
        black_box(workspace.arena().amplitude_slices().0[0]);
        start.elapsed().as_nanos() as f64 / repetitions as f64
    }

    fn measure_gather_call_scatter(
        applet: &Applet,
        workspace: &mut EagerDirectTableWorkspace,
        scratch: &mut GatherCallScatterScratch,
        repetitions: usize,
    ) -> f64 {
        let start = Instant::now();
        for _ in 0..repetitions {
            gather_call_scatter(applet, workspace, scratch);
        }
        black_box(workspace.arena().amplitude_slices().0[0]);
        start.elapsed().as_nanos() as f64 / repetitions as f64
    }

    #[test]
    fn abi_boundaries_and_panic_payloads_fail_closed() {
        let (application, _rows, _workspace) = table_fixture();
        let error = application
            .load_rows(vec![0; mem::size_of::<InvocationRow>() - 1], Vec::new())
            .unwrap_err();
        assert!(error.message().contains("whole number"));

        let malformed = catch_unwind(AssertUnwindSafe(|| {
            LoadedSymjitEagerDirectTable::load_prepared_application_bytes(
                &[0xff],
                &[],
                PathBuf::from("malformed-eager-table.symjit"),
                EAGER_DIRECT_SOURCE_APPLICATION_ABI,
                EAGER_DIRECT_TABLE_DESCRIPTOR_ABI,
                EAGER_DIRECT_TABLE_BINDING_ABI,
            )
        }));
        assert!(matches!(malformed, Ok(Err(_))));
        assert_eq!(
            panic_detail(Box::new(String::from("synthetic table panic"))),
            "synthetic table panic"
        );
    }
}
