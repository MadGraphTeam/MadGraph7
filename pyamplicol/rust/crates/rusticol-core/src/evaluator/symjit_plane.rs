// SPDX-License-Identifier: 0BSD

//! Standard SymJIT P-kernel loading, binding, and execution.
//!
//! The generic raw-plane extension applied to SymJIT 2.22 exposes the existing
//! P-kernel machine code through a stable `#[repr(C)]` descriptor. Rusticol
//! keeps those descriptors in a cold-built vector, so duplicate inputs and
//! intentional input/output aliases never require overlapping Rust
//! references. Scalar/SIMD dispatch and descriptor rebinding allocate nothing.

use crate::{RusticolError, RusticolResult};
use std::any::Any;
use std::marker::PhantomData;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
#[cfg(any(test, feature = "python-generation-bridge"))]
use symjit::Compiler;
use symjit::{
    Applet, Application, Compiled, CompiledPlaneFunc, CompilerType, Config, Defuns,
    PlaneDescriptor as SymjitPlaneDescriptor, Storage,
};

const SYMJIT_APPLICATION_MAGIC: usize = 0x4056_8795_410d_08e9;
const SYMJIT_APPLICATION_STORAGE_VERSION: usize = 4;
const MAX_OPTIMIZATION_LEVEL: u8 = 3;

type PlaneKernelFn = CompiledPlaneFunc<f64>;

/// Logical shape of a plane-oriented SymJIT application.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SymjitPlaneLayout {
    input_count: usize,
    output_count: usize,
    complex: bool,
}

impl SymjitPlaneLayout {
    #[cfg(test)]
    pub(crate) const fn real(input_count: usize, output_count: usize) -> Self {
        Self {
            input_count,
            output_count,
            complex: false,
        }
    }

    pub(crate) const fn complex(input_count: usize, output_count: usize) -> Self {
        Self {
            input_count,
            output_count,
            complex: true,
        }
    }

    pub(crate) const fn is_complex(self) -> bool {
        self.complex
    }

    pub(crate) fn input_plane_count(self) -> RusticolResult<usize> {
        self.input_count
            .checked_mul(if self.complex { 2 } else { 1 })
            .ok_or_else(|| {
                RusticolError::invalid_argument("SymJIT input plane count overflows usize")
            })
    }

    pub(crate) fn output_plane_count(self) -> RusticolResult<usize> {
        self.output_count
            .checked_mul(if self.complex { 2 } else { 1 })
            .ok_or_else(|| {
                RusticolError::invalid_argument("SymJIT output plane count overflows usize")
            })
    }

    pub(crate) fn plane_count(self) -> RusticolResult<usize> {
        let components = if self.complex { 2 } else { 1 };
        self.input_count
            .checked_add(self.output_count)
            .and_then(|count| count.checked_mul(components))
            .ok_or_else(|| RusticolError::invalid_argument("SymJIT plane count overflows usize"))
    }
}

/// A lifetime-tagged wrapper around SymJIT's stable raw plane descriptor.
///
/// Safe slice construction ties the tag to a real exclusive borrow. Runtime
/// schedulers may instead use the unsafe cached constructor, whose `'static`
/// tag permits owner-managed storage but does not extend any allocation's
/// lifetime.
#[derive(Debug)]
#[repr(transparent)]
pub(crate) struct SymjitRawPlane<'a> {
    descriptor: SymjitPlaneDescriptor<f64>,
    _lifetime: PhantomData<&'a mut [f64]>,
}

/// Single cold-path descriptor type used by Rusticol schedulers.
pub(crate) type PlaneDescriptor<'a> = SymjitRawPlane<'a>;

impl<'a> SymjitRawPlane<'a> {
    #[cfg(any(test, feature = "python-generation-bridge", feature = "loop-v2"))]
    pub(crate) fn from_slice(values: &'a mut [f64]) -> Self {
        Self {
            // SAFETY: The returned wrapper retains the exclusive slice
            // lifetime, so the allocation outlives every use of the raw
            // descriptor.
            descriptor: unsafe {
                SymjitPlaneDescriptor::from_raw_parts(values.as_mut_ptr(), values.len())
            },
            _lifetime: PhantomData,
        }
    }

    /// Creates a cold-binding descriptor without creating a Rust reference.
    ///
    /// # Safety
    ///
    /// `pointer` must remain allocated, aligned, and writable for `len`
    /// consecutive `f64` values during every invocation that receives this
    /// descriptor. The caller must prevent every later invocation or
    /// dereference after its allocation moves, reallocates, or drops, and
    /// synchronize all accesses.
    /// Duplicate descriptors and exact input/output aliases can be represented
    /// because this wrapper never creates overlapping Rust references. A
    /// complete kernel table rejects overlapping outputs and shifted or
    /// otherwise partial input/output overlaps.
    pub(crate) unsafe fn from_raw_parts(pointer: *mut f64, len: usize) -> RusticolResult<Self> {
        let pointer = if len == 0 && pointer.is_null() {
            NonNull::<f64>::dangling().as_ptr()
        } else {
            NonNull::new(pointer)
                .ok_or_else(|| {
                    RusticolError::invalid_argument(
                        "a non-empty SymJIT plane has a null data pointer",
                    )
                })?
                .as_ptr()
        };
        if pointer.addr() % std::mem::align_of::<f64>() != 0 {
            return Err(RusticolError::invalid_argument(
                "a SymJIT plane data pointer is not aligned for f64",
            ));
        }
        Ok(Self {
            descriptor: unsafe { SymjitPlaneDescriptor::from_raw_parts(pointer, len) },
            _lifetime: PhantomData,
        })
    }

    fn interval(&self) -> RusticolResult<Option<(usize, usize)>> {
        if self.descriptor.len == 0 {
            return Ok(None);
        }
        let start = self.descriptor.data.addr();
        let bytes = self
            .descriptor
            .len
            .checked_mul(std::mem::size_of::<f64>())
            .ok_or_else(|| {
                RusticolError::invalid_argument("SymJIT plane byte length overflows usize")
            })?;
        let end = start.checked_add(bytes).ok_or_else(|| {
            RusticolError::invalid_argument("SymJIT plane address range overflows usize")
        })?;
        Ok(Some((start, end)))
    }

    pub(crate) const fn len(&self) -> usize {
        self.descriptor.len
    }
}

impl SymjitRawPlane<'static> {
    /// Creates a descriptor for an owner-managed, address-stable runtime cache.
    ///
    /// The returned `'static` tag is a storage convenience, not an allocation
    /// lifetime claim.
    ///
    /// # Safety
    ///
    /// The owning runtime must prevent every invocation or descriptor
    /// dereference after the backing allocation moves, reallocates, or drops;
    /// keep the memory aligned and writable for `len` values during each
    /// invocation; and synchronize all descriptor use. Rebinding must finish
    /// before the old allocation is invalidated.
    pub(crate) unsafe fn from_cached_raw_parts(
        pointer: *mut f64,
        len: usize,
    ) -> RusticolResult<Self> {
        unsafe { SymjitRawPlane::from_raw_parts(pointer, len) }
    }
}

/// Loaded P-kernel code. The `Applet` owns the executable mappings referenced
/// by the cached function pointers.
pub(crate) struct SymjitPlaneKernel {
    applet: Applet,
    display_path: PathBuf,
    layout: SymjitPlaneLayout,
    optimization_level: u8,
    compression: bool,
    scalar: PlaneKernelFn,
    simd: Option<PlaneKernelFn>,
    simd_lanes: usize,
}

impl std::fmt::Debug for SymjitPlaneKernel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SymjitPlaneKernel")
            .field("display_path", &self.display_path)
            .field("layout", &self.layout)
            .field("optimization_level", &self.optimization_level)
            .field("compression", &self.compression)
            .field("simd_lanes", &self.simd_lanes)
            .finish_non_exhaustive()
    }
}

impl SymjitPlaneKernel {
    /// Loads, authenticates, recompiles, and seals a storage-v4 P-kernel.
    pub(crate) fn load_bytes(
        bytes: &[u8],
        display_path: impl Into<PathBuf>,
        layout: SymjitPlaneLayout,
    ) -> RusticolResult<Self> {
        let display_path = display_path.into();
        authenticate_storage_v4(bytes, &display_path)?;
        require_symjit_host_cpu(&display_path)?;

        let mut loader_config = explicit_loader_config()?;
        loader_config.set_defuns(scalar_primitive_defuns()?);
        let mut input = bytes;
        let mut application = guard_symjit_panic(
            || Application::load(&mut input, &loader_config),
            || format!("load plane application {}", display_path.display()),
        )?
        .map_err(|error| {
            RusticolError::compatibility(format!(
                "could not load SymJIT storage-v4 plane application {}: {error}",
                display_path.display()
            ))
        })?;
        if !input.is_empty() {
            return Err(RusticolError::integrity(format!(
                "SymJIT plane application {} has {} unauthenticated trailing bytes",
                display_path.display(),
                input.len()
            )));
        }
        let optimization_level = validate_application(&application, &display_path, layout)?;
        let compression = application.config.compress();

        // SIMD compilation is lazy in SymJIT. Calling this before sealing
        // guarantees that a supported SIMD kernel is retained by the Applet.
        guard_symjit_panic(
            || application.prepare_simd(),
            || {
                format!(
                    "prepare SIMD for plane application {}",
                    display_path.display()
                )
            },
        )?;
        validate_compiled_kernels(&application, &display_path)?;

        let applet = guard_symjit_panic(
            || application.seal(),
            || format!("seal plane application {}", display_path.display()),
        )?
        .map_err(|error| {
            RusticolError::evaluation(format!(
                "could not seal SymJIT plane application {}: {error}",
                display_path.display()
            ))
        })?;
        let scalar = applet.scalar_plane_kernel().ok_or_else(|| {
            RusticolError::compatibility(format!(
                "SymJIT plane application {} has no scalar raw-plane kernel",
                display_path.display()
            ))
        })?;
        let (simd, simd_lanes) = match applet.compiled_simd.as_ref() {
            Some(compiled) if compiled.support_indirect() => {
                (applet.simd_plane_kernel(), compiled.count_lanes())
            }
            _ => (None, 1),
        };
        validate_simd_lane_width(simd.is_some(), simd_lanes, &display_path)?;

        Ok(Self {
            applet,
            display_path,
            layout,
            optimization_level,
            compression,
            scalar,
            simd,
            simd_lanes,
        })
    }

    pub(crate) fn input_plane_count(&self) -> usize {
        self.layout
            .input_plane_count()
            .expect("validated SymJIT input plane count")
    }

    pub(crate) fn output_plane_count(&self) -> usize {
        self.layout
            .output_plane_count()
            .expect("validated SymJIT output plane count")
    }

    /// Points per SIMD call (1 without a SIMD kernel).
    #[cfg(any(test, feature = "loop-v2"))]
    pub(crate) const fn simd_lanes(&self) -> usize {
        self.simd_lanes
    }

    pub(crate) fn optimization_level(&self) -> u8 {
        self.optimization_level
    }

    pub(crate) fn compression(&self) -> bool {
        self.compression
    }

    /// Cold-binds safe, mutually disjoint slices into one stable descriptor
    /// table. The vector is never resized on the execution path.
    #[cfg(test)]
    pub(crate) fn bind<'kernel, 'planes>(
        &'kernel self,
        planes: Vec<&'planes mut [f64]>,
    ) -> RusticolResult<SymjitBoundPlaneKernel<'kernel, 'planes>> {
        let table = self.build_table(planes)?;
        Ok(SymjitBoundPlaneKernel {
            kernel: self,
            table,
        })
    }

    /// Cold-builds a descriptor table which is independent of the kernel
    /// borrow and can be cached by recurrence/eager runtimes.
    #[cfg(test)]
    pub(crate) fn build_table<'planes>(
        &self,
        planes: Vec<&'planes mut [f64]>,
    ) -> RusticolResult<SymjitPlaneTable<'planes>> {
        validate_slice_table(self.layout, &planes)?;
        let point_count = planes.first().map_or(0, |plane| plane.len());
        let descriptors = planes
            .into_iter()
            .map(|plane| {
                // SAFETY: The table carries `'planes`, retaining each
                // exclusive slice borrow for the descriptor's lifetime.
                unsafe { SymjitPlaneDescriptor::from_raw_parts(plane.as_mut_ptr(), plane.len()) }
            })
            .collect();
        Ok(SymjitPlaneTable {
            descriptors,
            point_count,
            input_plane_count: self.input_plane_count(),
            output_plane_count: self.output_plane_count(),
            simd_safe: true,
            _lifetime: PhantomData,
        })
    }

    /// Cold-binds raw planes after validating their shape and address ranges.
    /// Duplicate input planes and exact input/output aliases are valid.
    #[cfg(test)]
    pub(crate) fn bind_raw<'kernel, 'planes>(
        &'kernel self,
        planes: Vec<SymjitRawPlane<'planes>>,
    ) -> RusticolResult<SymjitBoundPlaneKernel<'kernel, 'planes>> {
        let table = self.build_raw_table(planes)?;
        Ok(SymjitBoundPlaneKernel {
            kernel: self,
            table,
        })
    }

    /// Cold-builds a reusable table from raw plane descriptors.
    pub(crate) fn build_raw_table<'planes>(
        &self,
        planes: Vec<PlaneDescriptor<'planes>>,
    ) -> RusticolResult<SymjitPlaneTable<'planes>> {
        self.build_raw_table_from_descriptors(&planes)
    }

    /// Cold-builds a reusable table while retaining the caller's descriptor
    /// vector and capacity for subsequent allocation-free row preparation.
    pub(crate) fn build_raw_table_from_descriptors<'planes>(
        &self,
        planes: &[PlaneDescriptor<'planes>],
    ) -> RusticolResult<SymjitPlaneTable<'planes>> {
        validate_raw_table(self.layout, planes)?;
        let point_count = planes.first().map_or(0, PlaneDescriptor::len);
        let input_plane_count = self.input_plane_count();
        let output_plane_count = self.output_plane_count();
        let simd_safe = validate_raw_aliases(input_plane_count, output_plane_count, planes)?;
        Ok(SymjitPlaneTable {
            descriptors: planes.iter().map(|plane| plane.descriptor).collect(),
            point_count,
            input_plane_count,
            output_plane_count,
            simd_safe,
            _lifetime: PhantomData,
        })
    }

    pub(crate) fn execute_table(
        &self,
        table: &mut SymjitPlaneTable<'_>,
        point_start: usize,
        point_count: usize,
    ) -> RusticolResult<()> {
        if table.input_plane_count != self.input_plane_count()
            || table.output_plane_count != self.output_plane_count()
        {
            return Err(RusticolError::invalid_argument(format!(
                "SymJIT plane table shape ({}, {}) does not match kernel shape ({}, {})",
                table.input_plane_count,
                table.output_plane_count,
                self.input_plane_count(),
                self.output_plane_count()
            )));
        }
        self.invoke_descriptors(table, point_start, point_count)
    }

    /// Validates a scheduler-owned raw descriptor slice on the cold path.
    ///
    /// Retained as the checked counterpart required by the generic unsafe raw
    /// callable contract. Current schedulers perform equivalent validation
    /// while constructing their authenticated row storage.
    #[allow(dead_code)]
    pub(crate) fn validate_raw_descriptors(
        &self,
        planes: &[PlaneDescriptor<'_>],
    ) -> RusticolResult<usize> {
        validate_raw_table(self.layout, planes)?;
        let _ = validate_raw_aliases(self.input_plane_count(), self.output_plane_count(), planes)?;
        Ok(planes.first().map_or(0, PlaneDescriptor::len))
    }

    /// Executes a scheduler-owned descriptor slice without allocating or
    /// copying its entries.
    ///
    /// # Safety
    ///
    /// [`Self::validate_raw_descriptors`] or equivalent scheduler-owned cold
    /// validation must previously have succeeded for the same descriptor
    /// shape. Every pointed-to allocation must still be live, synchronized,
    /// and cover `point_start..point_start + point_count`.
    /// When this kernel has a SIMD callable, every input byte range must be
    /// disjoint from every output byte range. A divergent SIMD block writes
    /// its epilogue before returning the replay signal, so an aliased output
    /// could otherwise destroy an input before scalar replay. Callers that
    /// need input/output aliases must use a cold-built [`SymjitPlaneTable`],
    /// which detects the overlap and disables SIMD.
    pub(crate) unsafe fn execute_raw_descriptors_unchecked(
        &self,
        planes: &[PlaneDescriptor<'_>],
        point_start: usize,
        point_count: usize,
    ) -> RusticolResult<()> {
        if planes.len() != self.layout.plane_count()? {
            return Err(RusticolError::invalid_argument(
                "SymJIT raw descriptor shape changed after cold validation",
            ));
        }
        let point_end = point_start.checked_add(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("SymJIT plane execution range overflows usize")
        })?;
        if point_end > planes.first().map_or(0, PlaneDescriptor::len) {
            return Err(RusticolError::invalid_argument(
                "SymJIT raw descriptor range exceeds its validated plane capacity",
            ));
        }
        if point_count == 0 {
            return Ok(());
        }
        // `SymjitRawPlane` is transparent over the upstream `#[repr(C)]`
        // descriptor; the lifetime marker has no storage.
        let table = planes.as_ptr().cast::<SymjitPlaneDescriptor<f64>>();
        let failure = invoke_range(
            self.scalar,
            self.simd,
            self.simd_lanes,
            table,
            self.applet.params.as_ptr(),
            point_start,
            point_end,
        )?;
        if let Some(failure) = failure {
            return Err(RusticolError::evaluation(format!(
                "SymJIT plane application {} returned status {} from its scalar kernel at point index {}",
                self.display_path.display(),
                failure.status,
                failure.index
            )));
        }
        Ok(())
    }

    /// Invokes a prebuilt, validated descriptor table without allocating or
    /// changing its address. This is the only runtime method coupled to
    /// SymJIT's raw P-kernel callable.
    fn invoke_descriptors(
        &self,
        table: &SymjitPlaneTable<'_>,
        point_start: usize,
        point_count: usize,
    ) -> RusticolResult<()> {
        let point_end = point_start.checked_add(point_count).ok_or_else(|| {
            RusticolError::invalid_argument("SymJIT plane execution range overflows usize")
        })?;
        if point_end > table.point_count {
            return Err(RusticolError::invalid_argument(format!(
                "SymJIT plane execution range {point_start}..{point_end} exceeds point capacity {}",
                table.point_count
            )));
        }
        if point_count == 0 {
            return Ok(());
        }

        let failure = invoke_range(
            self.scalar,
            self.simd.filter(|_| table.simd_safe),
            self.simd_lanes,
            table.as_symjit_ptr(),
            self.applet.params.as_ptr(),
            point_start,
            point_end,
        )?;
        if let Some(failure) = failure {
            return Err(RusticolError::evaluation(format!(
                "SymJIT plane application {} returned status {} from its scalar kernel at point index {}",
                self.display_path.display(),
                failure.status,
                failure.index
            )));
        }
        Ok(())
    }
}

/// The single representation-sensitive descriptor table. Descriptors are
/// allocated and validated on the cold path and never resized while bound.
pub(crate) struct SymjitPlaneTable<'planes> {
    descriptors: Vec<SymjitPlaneDescriptor<f64>>,
    point_count: usize,
    input_plane_count: usize,
    output_plane_count: usize,
    /// False when any input byte range overlaps any output byte range.
    ///
    /// SymJIT's vector epilogue runs before it returns a lane-divergence
    /// signal. Such tables therefore use the scalar kernel from the outset so
    /// replay can never observe inputs clobbered by a speculative epilogue.
    simd_safe: bool,
    _lifetime: PhantomData<&'planes mut [f64]>,
}

impl SymjitPlaneTable<'_> {
    fn as_symjit_ptr(&self) -> *const SymjitPlaneDescriptor<f64> {
        self.descriptors.as_ptr()
    }

    #[cfg(test)]
    pub(crate) const fn point_count(&self) -> usize {
        self.point_count
    }

    #[cfg(test)]
    const fn simd_safe(&self) -> bool {
        self.simd_safe
    }
}

#[cfg(test)]
impl<'planes> SymjitPlaneTable<'planes> {
    /// Rebinds the existing descriptor allocation to a new set of raw planes.
    ///
    /// The vector allocation and address remain unchanged. Validation finishes
    /// before any slot is replaced.
    ///
    /// # Safety
    ///
    /// Every descriptor must remain allocated and writable for `'planes`. No
    /// execution may overlap this call, the old descriptors must no longer be
    /// used after rebinding begins, and the caller must synchronize access to
    /// the new storage until this table is rebound again or dropped. Any
    /// input/output alias must cover exactly the same byte range and be valid
    /// for the particular P-kernel operation being invoked.
    pub(crate) unsafe fn rebind_raw(
        &mut self,
        planes: &[PlaneDescriptor<'planes>],
    ) -> RusticolResult<()> {
        validate_raw_plane_count_and_ranges(self.descriptors.len(), planes)?;
        let simd_safe =
            validate_raw_aliases(self.input_plane_count, self.output_plane_count, planes)?;

        for (descriptor, plane) in self.descriptors.iter_mut().zip(planes) {
            *descriptor = plane.descriptor;
        }
        self.point_count = planes.first().map_or(0, PlaneDescriptor::len);
        self.simd_safe = simd_safe;
        Ok(())
    }
}

/// A stable, allocation-free hot binding of a P-kernel to split planes.
#[cfg(test)]
pub(crate) struct SymjitBoundPlaneKernel<'kernel, 'planes> {
    // Keep the Applet reachable for at least as long as every machine-code
    // function pointer invocation.
    kernel: &'kernel SymjitPlaneKernel,
    table: SymjitPlaneTable<'planes>,
}

#[cfg(test)]
impl std::fmt::Debug for SymjitBoundPlaneKernel<'_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SymjitBoundPlaneKernel")
            .field("kernel", &self.kernel)
            .field("plane_count", &self.table.descriptors.len())
            .field("point_count", &self.table.point_count)
            .finish()
    }
}

#[cfg(test)]
impl SymjitBoundPlaneKernel<'_, '_> {
    pub(crate) fn execute_all(&mut self) -> RusticolResult<()> {
        self.execute_range(0, self.table.point_count)
    }

    /// Executes `[point_start, point_start + point_count)`.
    ///
    /// Scalar and SIMD calls both receive a point index. Unaligned heads and
    /// incomplete tails are dispatched through the scalar kernel.
    pub(crate) fn execute_range(
        &mut self,
        point_start: usize,
        point_count: usize,
    ) -> RusticolResult<()> {
        self.kernel
            .execute_table(&mut self.table, point_start, point_count)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StatusFailure {
    index: usize,
    status: i32,
}

fn invoke_range(
    scalar: PlaneKernelFn,
    simd: Option<PlaneKernelFn>,
    simd_lanes: usize,
    table: *const SymjitPlaneDescriptor<f64>,
    params: *const f64,
    point_start: usize,
    point_end: usize,
) -> RusticolResult<Option<StatusFailure>> {
    catch_unwind(AssertUnwindSafe(|| {
        let mut point = point_start;

        if let Some(simd) = simd.filter(|_| simd_lanes > 1) {
            while point < point_end && !point.is_multiple_of(simd_lanes) {
                // SAFETY: The caller has validated every descriptor and the
                // point range; all allocations remain live for this call.
                let status = unsafe { scalar(std::ptr::null(), table, point, params) };
                if status != 0 {
                    return Some(StatusFailure {
                        index: point,
                        status,
                    });
                }
                point += 1;
            }

            while point_end - point >= simd_lanes {
                // SAFETY: As above; the SIMD row index starts an aligned block
                // wholly inside the validated interior.
                let status = unsafe { simd(std::ptr::null(), table, point, params) };
                if status != 0 {
                    // A nonzero vector status is SymJIT's lane-divergence
                    // signal when vector branching is disabled. Match the
                    // standard Applet contract by replaying every lane with
                    // the scalar kernel. Table callers disable SIMD for
                    // input/output aliases, because the speculative vector
                    // epilogue has already written outputs at this point.
                    for replay_point in point..point + simd_lanes {
                        // SAFETY: The aligned block lies entirely inside the
                        // validated range and scalar replay uses point indices.
                        let scalar_status =
                            unsafe { scalar(std::ptr::null(), table, replay_point, params) };
                        if scalar_status != 0 {
                            return Some(StatusFailure {
                                index: replay_point,
                                status: scalar_status,
                            });
                        }
                    }
                }
                point += simd_lanes;
            }
        }

        while point < point_end {
            // SAFETY: The scalar tail lies inside the validated point range.
            let status = unsafe { scalar(std::ptr::null(), table, point, params) };
            if status != 0 {
                return Some(StatusFailure {
                    index: point,
                    status,
                });
            }
            point += 1;
        }
        None
    }))
    .map_err(|payload| {
        RusticolError::evaluation(format!(
            "SymJIT panicked while executing a plane kernel: {}",
            panic_detail(payload)
        ))
    })
}

/// Compiles `repr(Evaluator.get_instructions())` into a complex, SIMD-enabled,
/// direct-arena SymJIT storage-v4 application.
///
/// This is deliberately a cold-path function. It constructs every SymJIT
/// option explicitly so a process-local `symjit.toml` cannot alter generated
/// artifacts.
#[cfg(any(test, feature = "python-generation-bridge"))]
pub(crate) fn compile_symbolica_program_to_plane_application_bytes(
    program_repr: &str,
    input_complex_count: usize,
    output_complex_count: usize,
    opt_level: u8,
    compress: bool,
) -> RusticolResult<Vec<u8>> {
    if program_repr.trim().is_empty() {
        return Err(RusticolError::invalid_argument(
            "Symbolica instruction representation must not be empty",
        ));
    }
    let layout = SymjitPlaneLayout::complex(input_complex_count, output_complex_count);
    let config = explicit_plane_config(true, opt_level, compress)?;
    let mut compiler = Compiler::with_config(config);
    let application = guard_symjit_panic(
        || compiler.translate(program_repr.to_owned(), input_complex_count),
        || "translate Symbolica instructions to a SymJIT plane application".to_string(),
    )?
    .map_err(|error| {
        RusticolError::serialization(format!(
            "could not translate Symbolica instructions to a SymJIT plane application: {error}"
        ))
    })?;
    seal_generated_plane_application(application, layout)
}

/// Validates, prepares and serializes a generated plane application as
/// storage-v4 IR.
#[cfg(any(test, feature = "python-generation-bridge"))]
fn seal_generated_plane_application(
    mut application: Application,
    layout: SymjitPlaneLayout,
) -> RusticolResult<Vec<u8>> {
    let display_path = Path::new("<generated Symbolica plane application>");
    validate_application(&application, display_path, layout)?;
    guard_symjit_panic(
        || application.prepare_simd(),
        || "prepare SIMD for generated SymJIT plane application".to_string(),
    )?;
    validate_compiled_kernels(&application, display_path)?;

    let mut bytes = Vec::new();
    guard_symjit_panic(
        || application.save(&mut bytes),
        || "serialize generated SymJIT plane application".to_string(),
    )?
    .map_err(|error| {
        RusticolError::serialization(format!(
            "could not serialize generated SymJIT plane application: {error}"
        ))
    })?;
    authenticate_storage_v4(&bytes, display_path)?;
    Ok(bytes)
}

/// Compiles a binary64 Symbolica evaluator into a complex, SIMD-enabled,
/// direct-arena SymJIT storage-v4 plane application; returns it with its
/// input and output counts.
///
/// The program is Symbolica's own instruction export of the evaluator
/// (`ExpressionEvaluator::export_instructions`), so registered constants and
/// fixed-argument functions (pi, ...) carry the binary64 values the
/// evaluator itself uses (Symbolica's Python export writes them as 0). Bodies
/// of non-inlined functions (a model's helper functions) are inlined
/// ([`InlinedProgram`]); every other call keeps its name, which SymJIT
/// resolves as a built-in or one of pyAmpliCol's scalar primitives
/// (`scalar_primitive_defuns`) and refuses otherwise. The instructions reach
/// SymJIT through the same `Composer` calls as its text front end.
#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
pub(crate) fn compile_symbolica_evaluator_to_plane_application_bytes(
    evaluator: &symbolica::evaluate::ExpressionEvaluator<symbolica::domains::float::Complex<f64>>,
    opt_level: u8,
    compress: bool,
) -> RusticolResult<(Vec<u8>, usize, usize)> {
    use symjit::{Composer, Translator};
    let exported = evaluator.export_instructions();
    let (inputs, outputs) = (exported.input_count, exported.output_count);
    let program = InlinedProgram::of(&exported)?;
    let layout = SymjitPlaneLayout::complex(inputs, outputs);
    let config = explicit_plane_config(true, opt_level, compress)?;
    let application = guard_symjit_panic(
        || -> Result<Application, String> {
            let mut translator = Translator::new(config);
            for value in &program.constants {
                translator
                    .append_constant(num_complex::Complex::new(value.re, value.im))
                    .map_err(|error| error.to_string())?;
            }
            for row in &program.rows {
                compose(&mut translator, row)?;
            }
            translator.set_num_params(inputs);
            translator.compile().map_err(|error| error.to_string())
        },
        || "translate a Symbolica evaluator to a SymJIT plane application".to_string(),
    )?
    .map_err(|error| {
        RusticolError::serialization(format!(
            "could not translate a Symbolica evaluator to a SymJIT plane application: {error}"
        ))
    })?;
    Ok((
        seal_generated_plane_application(application, layout)?,
        inputs,
        outputs,
    ))
}

/// The outputs of a binary64 Symbolica evaluator whose values at real inputs
/// are not decided by the values off the real axis: those reading, on an
/// argument that depends on the inputs, a function with a branch cut on the
/// real axis (`log`, `sqrt`, `arg`, a power with a slot exponent: its side
/// there is a signed zero's, and SymJIT and Symbolica's evaluator sign a
/// negated real's zero imaginary part differently), a discontinuous one
/// (pyAmpliCol's predicates `truth`, `equal`, `less`, `less_equal`) or any
/// function this rule does not know, a selection on such a condition
/// (`Join`), or written after a jump on such a condition. `exp`, `sin`,
/// `cos`, `conj`, `abs` and the predicates `real_part` and `imaginary_part`
/// are continuous and do not count. Sub-evaluator calls are read through
/// their bodies ([`InlinedProgram`]). The plane check
/// (`loop_v2::trees::compile_plane_application_checked`) certifies these
/// outputs only at real inputs it is given.
#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
pub(crate) fn real_axis_sensitive_outputs(
    evaluator: &symbolica::evaluate::ExpressionEvaluator<symbolica::domains::float::Complex<f64>>,
) -> RusticolResult<Vec<usize>> {
    use symbolica::evaluate::{Instruction as I, Slot};
    let exported = evaluator.export_instructions();
    let program = InlinedProgram::of(&exported)?;
    // Per slot: (depends on the inputs, real-axis sensitive).
    let mut temporaries = vec![(false, false); program.temporaries];
    let mut outputs = vec![(false, false); exported.output_count];
    let read = |slot: &Slot, temporaries: &[(bool, bool)], outputs: &[(bool, bool)]| match *slot
    {
        Slot::Param(_) => (true, false),
        Slot::Const(_) => (false, false),
        Slot::Temp(i) => temporaries.get(i).copied().unwrap_or((true, true)),
        Slot::Out(i) => outputs.get(i).copied().unwrap_or((true, true)),
    };
    let fold = |slots: &[Slot], temporaries: &[(bool, bool)], outputs: &[(bool, bool)]| {
        slots.iter().fold((false, false), |(d, s), slot| {
            let (a, b) = read(slot, temporaries, outputs);
            (d || a, s || b)
        })
    };
    // After a jump on an input-dependent condition every write is sensitive
    // (the evaluator's control flow runs forward).
    let mut controlled = false;
    for row in &program.rows {
        let (destination, (dependent, sensitive)) = match row {
            I::Add(o, args, _) | I::Mul(o, args, _) => (o, fold(args, &temporaries, &outputs)),
            I::Pow(o, base, _, _) => (o, read(base, &temporaries, &outputs)),
            I::Powf(o, base, exponent, _) => {
                let (d, s) = fold(&[*base, *exponent], &temporaries, &outputs);
                (o, (d, s || d))
            }
            I::Fun(o, call, _) => {
                let (symbol, _tags, args) = &**call;
                let (d, s) = fold(args, &temporaries, &outputs);
                let continuous = if symbol.get_namespace() == "symbolica" {
                    matches!(
                        symbol.get_stripped_name(),
                        "exp" | "sin" | "cos" | "conj" | "abs"
                    )
                } else {
                    matches!(
                        symbol.get_name(),
                        "pyamplicol_scalar_v1::real_part" | "pyamplicol_scalar_v1::imaginary_part"
                    )
                };
                (o, (d, s || (d && !continuous)))
            }
            I::Assign(o, value) => (o, read(value, &temporaries, &outputs)),
            I::Join(o, condition, yes, no) => {
                let (c, cs) = read(condition, &temporaries, &outputs);
                let (d, s) = fold(&[*yes, *no], &temporaries, &outputs);
                (o, (c || d, s || cs || c))
            }
            I::IfElse(condition, _) => {
                let (c, cs) = read(condition, &temporaries, &outputs);
                controlled |= c || cs;
                continue;
            }
            I::Goto(_) | I::Label(_) => continue,
        };
        let value = (dependent || controlled, sensitive || controlled);
        match *destination {
            Slot::Temp(i) if i < temporaries.len() => temporaries[i] = value,
            Slot::Out(i) if i < outputs.len() => outputs[i] = value,
            _ => {
                return Err(RusticolError::serialization(
                    "a Symbolica instruction writes outside its temporaries and outputs",
                ));
            }
        }
    }
    Ok(outputs
        .iter()
        .enumerate()
        .filter_map(|(index, &(_, sensitive))| sensitive.then_some(index))
        .collect())
}

/// The inputs of a binary64 Symbolica evaluator whose values can move an
/// argument onto (or off) a branch cut or flip a discontinuous condition,
/// starting from the real input row `point`: every other input changes no
/// branch side as long as it keeps the sign (negative, zero, positive) it
/// has in `point` (`loop_v2::trees`' run-time side check of a parameter
/// program, M-SM MS-MASTERS).
///
/// An abstract interpretation of the program in the signs of the real and
/// imaginary parts of every value (each exactly zero, positive, negative or
/// unknown; an exact zero also records whether its sign bit follows from
/// the operands' signs alone), with the inputs' signs from `point` (real:
/// zero imaginary parts) and the constants' own, along the instructions'
/// forward jumps (a branch whose condition the signs decide is the only one
/// taken). Per path a value depends on the inputs of its last write
/// (Symbolica reuses its temporaries); where paths meet at a label they stay
/// apart until the selections (`Join`) that follow, so a selection on a
/// branch's condition reads, on each path, the operand that path's branch
/// chose (the other operand's slot may still hold an unrelated earlier
/// value). Per value it also keeps the inputs whose values can change its
/// signs: none for an input or a constant (the point fixes their signs) or
/// a value of fixed signs; a product's other factors' when its real factors
/// of fixed sign are all but one, or all its factors are real (so
/// `(b - d) / c^2` keeps b and d, not c); the argument's for `conj`, `abs`,
/// `real_part`, `imaginary_part`, a real base's power or 1 / z; every input
/// it depends on otherwise (a sum of mixed signs).
///
/// A `log` or `sqrt` whose argument's side of the cut on the negative real
/// axis the signs do not decide marks the inputs that can change the
/// argument's signs. The signs decide it off the axis (a known nonzero
/// imaginary part, or an exactly zero one with a positive real part) and on
/// it with a negative real part and an exactly zero imaginary part whose
/// sign bit the signs decide: at every point of the same signs the plane
/// and the exact evaluator then take the side the generation's real-point
/// check certified (they may sign such a zero differently, but each the
/// same way every time). A power with a slot exponent on an undecided base
/// marks every input of base and exponent. A branch or selection on a
/// condition the signs do not decide marks the inputs that can change the
/// condition's signs; a predicate (`truth`, `equal`, `less`, `less_equal`)
/// the signs do not decide marks those of its arguments when it tests one
/// against an exact zero (`truth` included), else every input they depend
/// on; `arg` or a function this rule does not know marks every input of its
/// arguments. `exp`, `sin`, `cos`, `conj`, `abs`, `real_part` and
/// `imaginary_part` are continuous. So `sqrt(4 pi aS)`, `log(mu_R^2 / m^2)`
/// and `log(-m^2 / mu_R^2)` at positive inputs, or the complex-mass
/// scheme's `log(mu_R^2 / (m^2 - i m w))` at a positive width, mark nothing,
/// while `log(m1^2 - m2^2)` marks both masses, and so does `log(-m1^(m2))`
/// (a power with a slot exponent of a positive base has an exactly zero
/// imaginary part whose sign follows the exponent's value). The sign rules
/// are exact for finite values (a product of two positive values is
/// positive, a sum of a positive and a zero one positive, a product of a
/// zero and a value of unknown sign a zero of unknown sign, ...); a product
/// that underflows to zero is outside them.
#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
pub(crate) fn real_axis_side_inputs(
    evaluator: &symbolica::evaluate::ExpressionEvaluator<symbolica::domains::float::Complex<f64>>,
    point: &[f64],
) -> RusticolResult<Vec<usize>> {
    let exported = evaluator.export_instructions();
    if point.len() != exported.input_count {
        return Err(RusticolError::invalid_argument(format!(
            "side inputs: a point of {} values for a program of {} inputs",
            point.len(),
            exported.input_count
        )));
    }
    let program = InlinedProgram::of(&exported)?;
    side_inputs_of(&program, exported.input_count, exported.output_count, point)
}

/// [`real_axis_side_inputs`] of an inlined program of `inputs` inputs and
/// `outputs` outputs.
#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
fn side_inputs_of(
    program: &InlinedProgram,
    inputs: usize,
    outputs: usize,
    point: &[f64],
) -> RusticolResult<Vec<usize>> {
    use symbolica::evaluate::{Instruction as I, Slot};
    let words = inputs.div_ceil(64).max(1);
    let slots = program.temporaries + outputs;
    let index = |slot: &Slot| -> Option<usize> {
        match *slot {
            Slot::Temp(i) if i < program.temporaries => Some(i),
            Slot::Out(i) if i < outputs => Some(program.temporaries + i),
            _ => None,
        }
    };
    let unknown = Signs::UNKNOWN;
    let read = |slot: &Slot, path: &SidePath| -> Signs {
        match *slot {
            Slot::Param(i) => Signs {
                re: Sign::of(point.get(i).copied().unwrap_or(f64::NAN)),
                im: Sign::Zero,
            },
            Slot::Const(i) => program.constants.get(i).map_or(unknown, |c| Signs {
                re: Sign::of(c.re),
                im: Sign::of(c.im),
            }),
            _ => index(slot).map_or(unknown, |i| path.signs[i]),
        }
    };
    // The inputs `slots`' values depend on, on `path`.
    let deps = |slots: &[Slot], path: &SidePath| -> Vec<u64> {
        let mut out = vec![0_u64; words];
        for slot in slots {
            match *slot {
                Slot::Param(i) if i < inputs => out[i / 64] |= 1 << (i % 64),
                Slot::Const(_) | Slot::Param(_) => {}
                _ => {
                    if let Some(k) = index(slot) {
                        for (o, d) in out.iter_mut().zip(path.depends(k, words)) {
                            *o |= d;
                        }
                    } else {
                        out.iter_mut().for_each(|o| *o = u64::MAX);
                    }
                }
            }
        }
        out
    };
    // The inputs whose values can change `slots`' signs (a zero's bit
    // included) at the point's signs, on `path`: none for an input or a
    // constant, whose signs the point fixes.
    let sign_deps = |slots: &[Slot], path: &SidePath| -> Vec<u64> {
        let mut out = vec![0_u64; words];
        for slot in slots {
            match *slot {
                Slot::Const(_) | Slot::Param(_) => {}
                _ => {
                    if let Some(k) = index(slot) {
                        for (o, d) in out.iter_mut().zip(path.sign_depends(k, words)) {
                            *o |= d;
                        }
                    } else {
                        out.iter_mut().for_each(|o| *o = u64::MAX);
                    }
                }
            }
        }
        out
    };
    let zero_constant = |slot: &Slot| match *slot {
        Slot::Const(i) => program
            .constants
            .get(i)
            .is_some_and(|c| c.re == 0.0 && c.im == 0.0),
        _ => false,
    };
    // A condition's truth on `path`: its signs', else the branch taken.
    let truth = |slot: &Slot, path: &SidePath| -> Option<bool> {
        read(slot, path)
            .truth()
            .or_else(|| index(slot).and_then(|k| path.fact(k)))
    };
    let mark = |side: &mut Vec<u64>, d: &[u64]| {
        for (s, d) in side.iter_mut().zip(d) {
            *s |= d;
        }
    };
    // A value, the inputs it depends on and those that can change its signs
    // (fixed signs: none).
    let write = |path: &mut SidePath,
                 slot: &Slot,
                 value: Signs,
                 dependence: &[u64],
                 sign_dependence: &[u64]| {
        let Some(k) = index(slot) else {
            return Err(RusticolError::serialization(
                "a Symbolica instruction writes outside its temporaries and outputs",
            ));
        };
        path.signs[k] = value;
        path.depends[k * words..(k + 1) * words].copy_from_slice(dependence);
        let signs = &mut path.sign_depends[k * words..(k + 1) * words];
        if value.fixed() {
            signs.fill(0);
        } else {
            signs.copy_from_slice(sign_dependence);
        }
        path.facts.retain(|&(fact, _)| fact != k);
        Ok(())
    };
    // The paths reaching the current instruction (`None`: unreachable),
    // kept apart from a label where several meet until the next
    // instruction other than a label or a selection, so that a selection on
    // a branch's condition reads each operand on the path that takes it,
    // and the paths waiting at each label.
    let mut current: Option<Vec<SidePath>> = Some(vec![SidePath::new(slots, words)]);
    let mut pending: std::collections::BTreeMap<usize, Vec<SidePath>> =
        std::collections::BTreeMap::new();
    let mut side = vec![0_u64; words];
    for row in &program.rows {
        match row {
            I::Label(label) => {
                if let Some(mut arriving) = pending.remove(label) {
                    match &mut current {
                        Some(paths) => paths.append(&mut arriving),
                        None => current = Some(arriving),
                    }
                    if let Some(paths) = &mut current
                        && paths.len() > SidePath::KEPT
                    {
                        *paths = vec![SidePath::merge(std::mem::take(paths))];
                    }
                }
                continue;
            }
            I::Join(o, condition, yes, no) => {
                let Some(paths) = &mut current else { continue };
                for path in paths.iter_mut() {
                    let (value, dependence, sign_dependence) = match truth(condition, path) {
                        Some(true) => (
                            read(yes, path),
                            deps(&[*yes], path),
                            sign_deps(&[*yes], path),
                        ),
                        Some(false) => {
                            (read(no, path), deps(&[*no], path), sign_deps(&[*no], path))
                        }
                        None => {
                            mark(&mut side, &sign_deps(&[*condition], path));
                            (
                                read(yes, path).join(read(no, path)),
                                deps(&[*condition, *yes, *no], path),
                                sign_deps(&[*condition, *yes, *no], path),
                            )
                        }
                    };
                    write(path, o, value, &dependence, &sign_dependence)?;
                }
                continue;
            }
            _ => {}
        }
        let Some(paths) = current.take() else {
            continue;
        };
        let mut path = SidePath::merge(paths);
        match row {
            I::Goto(label) => {
                pending.entry(*label).or_default().push(path);
                continue;
            }
            I::IfElse(condition, label) => {
                match truth(condition, &path) {
                    Some(true) => {}
                    Some(false) => {
                        pending.entry(*label).or_default().push(path);
                        continue;
                    }
                    None => {
                        mark(&mut side, &sign_deps(&[*condition], &path));
                        let mut jump = path.clone();
                        if let Some(k) = index(condition) {
                            jump.facts.push((k, false));
                            path.facts.push((k, true));
                        }
                        pending.entry(*label).or_default().push(jump);
                    }
                }
                current = Some(vec![path]);
                continue;
            }
            _ => {}
        }
        // The inputs that can change the value's signs, when fewer than
        // those it depends on (`None`: all of them).
        let (destination, value, dependence, sign_dependence) = match row {
            I::Add(o, args, _) => (
                o,
                args.iter()
                    .map(|a| read(a, &path))
                    .reduce(Signs::add)
                    .unwrap_or(Signs::ZERO),
                deps(args, &path),
                None,
            ),
            I::Mul(o, args, _) => {
                let factors: Vec<Signs> = args.iter().map(|a| read(a, &path)).collect();
                // Real factors of fixed signs scale both parts by a sign the
                // point fixes: with at most one other factor, or only real
                // ones, the product's signs follow the other factors' signs.
                let rest: Vec<Slot> = args
                    .iter()
                    .zip(&factors)
                    .filter(|(_, f)| !(f.fixed() && f.im == Sign::Zero))
                    .map(|(a, _)| *a)
                    .collect();
                let real = factors.iter().all(|f| f.im == Sign::Zero);
                (
                    o,
                    factors.into_iter().reduce(Signs::mul).unwrap_or(Signs::ONE),
                    deps(args, &path),
                    (real || rest.len() <= 1).then(|| sign_deps(&rest, &path)),
                )
            }
            I::Pow(o, base, exponent, _) => {
                let b = read(base, &path);
                // A real base's powers, and 1 / z = conj(z) / |z|^2, take
                // signs that follow the base's.
                (
                    o,
                    b.powi(*exponent),
                    deps(&[*base], &path),
                    (b.im == Sign::Zero || exponent.unsigned_abs() == 1)
                        .then(|| sign_deps(&[*base], &path)),
                )
            }
            I::Powf(o, base, exponent, _) => {
                let (b, e) = (read(base, &path), read(exponent, &path));
                let dependence = deps(&[*base, *exponent], &path);
                if !b.side_decided() {
                    mark(&mut side, &dependence);
                }
                // exp(e log b) at a positive real b: the imaginary part
                // e * (+-0) + Im(e) * log(b) is a zero whose sign follows
                // the exponent's value.
                let value = if b.re == Sign::Pos && b.im.is_zero() && e.im.is_zero() {
                    Signs {
                        re: Sign::Pos,
                        im: Sign::FreeZero,
                    }
                } else {
                    unknown
                };
                (o, value, dependence, None)
            }
            I::Fun(o, call, _) => {
                let (symbol, _tags, args) = &**call;
                let a: Vec<Signs> = args.iter().map(|s| read(s, &path)).collect();
                let dependence = deps(args, &path);
                let name = if symbol.get_namespace() == "symbolica" {
                    symbol.get_stripped_name()
                } else {
                    symbol.get_name()
                };
                let predicate = match (name, a.as_slice()) {
                    ("pyamplicol_scalar_v1::truth", [x]) => Some(x.truth()),
                    ("pyamplicol_scalar_v1::equal", [x, y]) => Some(if args[0] == args[1] {
                        Some(true)
                    } else {
                        x.equal(*y)
                    }),
                    ("pyamplicol_scalar_v1::less", [x, y]) => Some(x.re.less(y.re, false)),
                    ("pyamplicol_scalar_v1::less_equal", [x, y]) => Some(x.re.less(y.re, true)),
                    _ => None,
                };
                if let Some(truth) = predicate {
                    // A predicate the signs decide is constant at the
                    // point's signs. Another one depends on, and marks, the
                    // inputs that can change its arguments' signs when it
                    // tests one against an exact zero, else every input its
                    // arguments depend on.
                    let none = vec![0_u64; words];
                    let (value, inputs) = match truth {
                        Some(true) => (Signs::ONE, none),
                        Some(false) => (Signs::ZERO, none),
                        None => {
                            let tested = if name == "pyamplicol_scalar_v1::truth"
                                || args.iter().any(zero_constant)
                            {
                                sign_deps(args, &path)
                            } else {
                                dependence
                            };
                            mark(&mut side, &tested);
                            (
                                Signs {
                                    re: Sign::Any,
                                    im: Sign::Zero,
                                },
                                tested,
                            )
                        }
                    };
                    (o, value, inputs.clone(), Some(inputs))
                } else {
                    // The parts of `conj`, `abs`, `real_part` and
                    // `imaginary_part` take signs that follow the argument's.
                    let follows = Some(sign_deps(args, &path));
                    let (value, sign_dependence) = match (name, a.as_slice()) {
                        ("sqrt", [x]) if x.off_cut() => (
                            Signs {
                                re: Sign::Pos,
                                im: x.im,
                            },
                            None,
                        ),
                        ("log", [x]) if x.off_cut() => (
                            Signs {
                                re: Sign::Any,
                                im: x.im,
                            },
                            None,
                        ),
                        // On the cut, on the side the zero's sign decides
                        // (+-i pi for log; the sign is not tracked).
                        ("sqrt" | "log", [x]) if x.side_decided() => (unknown, None),
                        // On the cut on a side the signs do not decide: the
                        // inputs that can change the argument's signs.
                        ("sqrt" | "log", [_]) => {
                            mark(&mut side, &sign_deps(args, &path));
                            (unknown, None)
                        }
                        // e^a (cos(+-0) + i sin(+-0)): the zero's sign.
                        ("exp", [x]) if x.im.is_zero() => (
                            Signs {
                                re: Sign::Pos,
                                im: x.im,
                            },
                            None,
                        ),
                        // sin(a) cosh(+-0) + i cos(a) sinh(+-0) (and cos's):
                        // a zero whose sign follows cos(a) (sin(a)).
                        ("sin" | "cos", [x]) if x.im.is_zero() => (
                            Signs {
                                re: Sign::Any,
                                im: Sign::FreeZero,
                            },
                            None,
                        ),
                        ("exp" | "sin" | "cos", [_]) => (unknown, None),
                        ("conj", [x]) => (
                            Signs {
                                re: x.re,
                                im: x.im.neg(),
                            },
                            follows,
                        ),
                        ("abs", [x]) => (
                            Signs {
                                re: x.magnitude(),
                                im: Sign::Zero,
                            },
                            follows,
                        ),
                        ("pyamplicol_scalar_v1::real_part", [x]) => (
                            Signs {
                                re: x.re,
                                im: Sign::Zero,
                            },
                            follows,
                        ),
                        ("pyamplicol_scalar_v1::imaginary_part", [x]) => (
                            Signs {
                                re: x.im,
                                im: Sign::Zero,
                            },
                            follows,
                        ),
                        _ => {
                            // `arg` or unknown.
                            mark(&mut side, &dependence);
                            (unknown, None)
                        }
                    };
                    (o, value, dependence, sign_dependence)
                }
            }
            I::Assign(o, value) => (
                o,
                read(value, &path),
                deps(&[*value], &path),
                Some(sign_deps(&[*value], &path)),
            ),
            I::Join(..) | I::IfElse(..) | I::Goto(_) | I::Label(_) => {
                unreachable!("handled above")
            }
        };
        let sign_dependence = sign_dependence.unwrap_or_else(|| dependence.clone());
        write(&mut path, destination, value, &dependence, &sign_dependence)?;
        current = Some(vec![path]);
    }
    Ok((0..inputs)
        .filter(|&i| side[i / 64] & (1 << (i % 64)) != 0)
        .collect())
}

/// One control-flow path's state in [`real_axis_side_inputs`]: per slot
/// its signs and the inputs its value depends on (written on this path:
/// a write replaces them), and the truth of the conditions of the branches
/// this path took that the signs do not decide (a slot's fact lapses when
/// the slot is written).
#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
#[derive(Clone)]
struct SidePath {
    signs: Vec<Signs>,
    /// `words` bit words per slot.
    depends: Vec<u64>,
    /// The inputs whose values can change the slot's signs (a subset of
    /// `depends`; empty for fixed signs), `words` bit words per slot.
    sign_depends: Vec<u64>,
    facts: Vec<(usize, bool)>,
}

#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
impl SidePath {
    /// The most paths kept apart at a label (more are merged).
    const KEPT: usize = 8;

    fn new(slots: usize, words: usize) -> Self {
        Self {
            signs: vec![Signs::UNKNOWN; slots],
            depends: vec![0; slots * words],
            sign_depends: vec![0; slots * words],
            facts: Vec::new(),
        }
    }

    fn depends(&self, slot: usize, words: usize) -> &[u64] {
        &self.depends[slot * words..(slot + 1) * words]
    }

    fn sign_depends(&self, slot: usize, words: usize) -> &[u64] {
        &self.sign_depends[slot * words..(slot + 1) * words]
    }

    fn fact(&self, slot: usize) -> Option<bool> {
        self.facts
            .iter()
            .find(|&&(fact, _)| fact == slot)
            .map(|&(_, truth)| truth)
    }

    /// One path for several: per slot the signs' join and the dependences'
    /// union, and the facts all of them hold.
    fn merge(mut paths: Vec<Self>) -> Self {
        let mut merged = paths.pop().expect("a path");
        for path in paths {
            for (a, b) in merged.signs.iter_mut().zip(&path.signs) {
                *a = a.join(*b);
            }
            for (a, b) in merged.depends.iter_mut().zip(&path.depends) {
                *a |= b;
            }
            for (a, b) in merged.sign_depends.iter_mut().zip(&path.sign_depends) {
                *a |= b;
            }
            merged.facts.retain(|fact| path.facts.contains(fact));
        }
        merged
    }
}

/// The sign of a binary64 part ([`real_axis_side_inputs`]): exactly zero
/// with its sign bit decided by the operands' signs (an input's, a
/// constant's, a product of such a zero and values of known sign, a sum of
/// such zeros: the same at every point of the same signs, in SymJIT's and
/// Symbolica's arithmetic alike, though they may choose different bits),
/// exactly zero with a sign bit that may change with the values (a product
/// with a factor of unknown sign), positive, negative, or unknown.
#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sign {
    Zero,
    FreeZero,
    Pos,
    Neg,
    Any,
}

#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
impl Sign {
    fn of(x: f64) -> Self {
        if x == 0.0 {
            Self::Zero
        } else if x > 0.0 {
            Self::Pos
        } else if x < 0.0 {
            Self::Neg
        } else {
            Self::Any
        }
    }

    fn neg(self) -> Self {
        match self {
            Self::Pos => Self::Neg,
            Self::Neg => Self::Pos,
            other => other,
        }
    }

    fn add(self, other: Self) -> Self {
        match (self, other) {
            (Self::Zero, Self::Zero) => Self::Zero,
            (a, b) if a.is_zero() && b.is_zero() => Self::FreeZero,
            (z, x) | (x, z) if z.is_zero() => x,
            (a, b) if a == b => a,
            _ => Self::Any,
        }
    }

    fn mul(self, other: Self) -> Self {
        match (self, other) {
            (Self::Zero, x) | (x, Self::Zero) if x != Self::Any && x != Self::FreeZero => {
                Self::Zero
            }
            (z, _) | (_, z) if z.is_zero() => Self::FreeZero,
            (Self::Any, _) | (_, Self::Any) => Self::Any,
            (a, b) if a == b => Self::Pos,
            _ => Self::Neg,
        }
    }

    fn join(self, other: Self) -> Self {
        if self == other {
            self
        } else if self.is_zero() && other.is_zero() {
            Self::FreeZero
        } else {
            Self::Any
        }
    }

    /// Exactly zero (either sign bit).
    fn is_zero(self) -> bool {
        matches!(self, Self::Zero | Self::FreeZero)
    }

    fn known_nonzero(self) -> bool {
        matches!(self, Self::Pos | Self::Neg)
    }

    /// `self < other` (`or_equal`: `<=`) when the signs decide it (a zero's
    /// sign bit does not order it).
    fn less(self, other: Self, or_equal: bool) -> Option<bool> {
        let rank = |s: Self| match s {
            Self::Neg => Some(0),
            Self::Zero | Self::FreeZero => Some(1),
            Self::Pos => Some(2),
            Self::Any => None,
        };
        let (a, b) = (rank(self)?, rank(other)?);
        if a != b {
            Some(a < b)
        } else if a == 1 {
            // Both exactly zero.
            Some(or_equal)
        } else {
            None
        }
    }
}

/// The signs of a complex binary64 value ([`real_axis_side_inputs`]).
#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Signs {
    re: Sign,
    im: Sign,
}

#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
impl Signs {
    const UNKNOWN: Self = Self {
        re: Sign::Any,
        im: Sign::Any,
    };
    const ZERO: Self = Self {
        re: Sign::Zero,
        im: Sign::Zero,
    };
    const ONE: Self = Self {
        re: Sign::Pos,
        im: Sign::Zero,
    };

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re.add(other.re),
            im: self.im.add(other.im),
        }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re.mul(other.re).add(self.im.mul(other.im).neg()),
            im: self.re.mul(other.im).add(self.im.mul(other.re)),
        }
    }

    fn join(self, other: Self) -> Self {
        Self {
            re: self.re.join(other.re),
            im: self.im.join(other.im),
        }
    }

    /// Exactly zero (either sign bits).
    fn is_zero(self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }

    /// Both parts' signs, a zero's bit included, the same at every point of
    /// the same input signs.
    fn fixed(self) -> bool {
        let fixed = |s: Sign| matches!(s, Sign::Zero | Sign::Pos | Sign::Neg);
        fixed(self.re) && fixed(self.im)
    }

    /// Nonzero (`Some(true)`), zero (`Some(false)`) or undecided.
    fn truth(self) -> Option<bool> {
        if self.re.known_nonzero() || self.im.known_nonzero() {
            Some(true)
        } else if self.is_zero() {
            Some(false)
        } else {
            None
        }
    }

    /// `self == other` when the signs decide it (a zero's sign bit does
    /// not: -0 == +0).
    fn equal(self, other: Self) -> Option<bool> {
        let class = |s: Sign| match s {
            Sign::Zero | Sign::FreeZero => Some(0),
            Sign::Pos => Some(1),
            Sign::Neg => Some(-1),
            Sign::Any => None,
        };
        let differ =
            |a: Sign, b: Sign| matches!((class(a), class(b)), (Some(x), Some(y)) if x != y);
        if differ(self.re, other.re) || differ(self.im, other.im) {
            Some(false)
        } else if self.is_zero() && other.is_zero() {
            Some(true)
        } else {
            None
        }
    }

    /// Off the negative real axis and away from zero, the cut of `log`,
    /// `sqrt` and a power with a slot exponent: a known nonzero imaginary
    /// part, or a zero one with a positive real part.
    fn off_cut(self) -> bool {
        self.im.known_nonzero() || (self.im.is_zero() && self.re == Sign::Pos)
    }

    /// The side of that cut decided by the signs: off it, or on it (a
    /// negative real part) with an exactly zero imaginary part whose sign
    /// bit the operands' signs decide.
    fn side_decided(self) -> bool {
        self.off_cut() || (self.im == Sign::Zero && self.re == Sign::Neg)
    }

    /// The sign of |self|.
    fn magnitude(self) -> Sign {
        if self.re.known_nonzero() || self.im.known_nonzero() {
            Sign::Pos
        } else if self.is_zero() {
            Sign::Zero
        } else {
            Sign::Any
        }
    }

    /// `self^n`, an integer power (a negative one of a value that may be
    /// zero is unknown).
    fn powi(self, n: i64) -> Self {
        let base = if n < 0 {
            if self.truth() != Some(true) {
                return Self::UNKNOWN;
            }
            // 1 / (a + ib) = (a - ib) / (a^2 + b^2).
            Self {
                re: self.re,
                im: self.im.neg(),
            }
        } else {
            self
        };
        let (mut result, mut square, mut k) = (Self::ONE, base, n.unsigned_abs());
        let mut first = true;
        while k > 0 {
            if k & 1 == 1 {
                result = if first { square } else { result.mul(square) };
                first = false;
            }
            square = square.mul(square);
            k >>= 1;
        }
        result
    }
}

/// Symbolica's instruction export with every sub-evaluator call inlined.
///
/// A call `dst = f(args)` of a sub-evaluator `f` becomes `f`'s body with its
/// parameters read from the call's argument slots, its temporaries and
/// output in fresh temporaries, its constants appended to the program's and
/// its labels renumbered past every label in use; the call's destination
/// receives the body's output last, so a destination that is also an
/// argument is read before it is written. Nested sub-evaluators inline in
/// turn, on an explicit stack of bodies (no depth limit: the export is a
/// finite tree of owned bodies). The evaluated arithmetic is unchanged (the
/// Python twin is `evaluators/symbolica_helpers._InlinedProgram`).
#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
struct InlinedProgram {
    rows: Vec<symbolica::evaluate::Instruction>,
    constants: Vec<symbolica::domains::float::Complex<f64>>,
    temporaries: usize,
    next_label: usize,
}

#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
type Exported = symbolica::evaluate::ExportedInstructions<symbolica::domains::float::Complex<f64>>;

/// A body being inlined: its next instruction, its slots in the program's
/// slots, its label offset and, for a call, the call's destination and the
/// body's output temporary.
#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
struct InlineFrame<'a> {
    program: &'a Exported,
    next: usize,
    /// `None` for the top level (its slots are the program's).
    call: Option<InlineCall>,
    offset: usize,
}

#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
struct InlineCall {
    destination: symbolica::evaluate::Slot,
    arguments: Vec<symbolica::evaluate::Slot>,
    temporaries: usize,
    output: usize,
    constants: usize,
}

#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
impl InlineFrame<'_> {
    fn slot(&self, slot: symbolica::evaluate::Slot) -> symbolica::evaluate::Slot {
        use symbolica::evaluate::Slot;
        let Some(call) = &self.call else {
            return slot;
        };
        match slot {
            Slot::Param(index) => call.arguments[index],
            Slot::Temp(index) => Slot::Temp(call.temporaries + index),
            Slot::Out(index) => Slot::Temp(call.output + index),
            Slot::Const(index) => Slot::Const(call.constants + index),
        }
    }
}

#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
impl InlinedProgram {
    fn of(exported: &Exported) -> RusticolResult<Self> {
        use symbolica::evaluate::{Instruction as I, Slot};
        let mut inlined = Self {
            rows: Vec::new(),
            constants: exported.constants.clone(),
            temporaries: exported.temporary_count,
            next_label: 1 + max_label(&exported.instructions).unwrap_or(0),
        };
        let mut stack = vec![InlineFrame {
            program: exported,
            next: 0,
            call: None,
            offset: 0,
        }];
        while let Some(frame) = stack.last_mut() {
            let program = frame.program;
            let Some(row) = program.instructions.get(frame.next) else {
                if let Some(call) = &frame.call {
                    inlined
                        .rows
                        .push(I::Assign(call.destination, Slot::Temp(call.output)));
                }
                stack.pop();
                continue;
            };
            frame.next += 1;
            let frame = &*frame;
            let destination = match row {
                I::Add(o, ..)
                | I::Mul(o, ..)
                | I::Pow(o, ..)
                | I::Powf(o, ..)
                | I::Fun(o, ..)
                | I::Assign(o, _)
                | I::Join(o, ..) => Some(*o),
                I::IfElse(..) | I::Goto(_) | I::Label(_) => None,
            };
            if frame.call.is_some() && matches!(destination, Some(Slot::Param(_))) {
                return Err(RusticolError::serialization(
                    "a Symbolica function body writes one of its parameters",
                ));
            }
            let list = |slots: &[Slot]| slots.iter().map(|&slot| frame.slot(slot)).collect();
            let offset = frame.offset;
            let mapped = match row {
                I::Add(o, args, reals) => I::Add(frame.slot(*o), list(args), *reals),
                I::Mul(o, args, reals) => I::Mul(frame.slot(*o), list(args), *reals),
                I::Pow(o, base, exponent, real) => {
                    I::Pow(frame.slot(*o), frame.slot(*base), *exponent, *real)
                }
                I::Powf(o, base, exponent, real) => I::Powf(
                    frame.slot(*o),
                    frame.slot(*base),
                    frame.slot(*exponent),
                    *real,
                ),
                I::Fun(o, call, real) => {
                    let (symbol, tags, args) = &**call;
                    let body = program
                        .sub_evaluators
                        .iter()
                        .find(|sub| sub.symbol == *symbol && sub.tags == *tags);
                    if let Some(body) = body {
                        let entered =
                            inlined.enter(frame.slot(*o), list(args), &body.instructions)?;
                        stack.push(entered);
                        continue;
                    }
                    I::Fun(
                        frame.slot(*o),
                        Box::new((*symbol, tags.clone(), list(args))),
                        *real,
                    )
                }
                I::Assign(o, value) => I::Assign(frame.slot(*o), frame.slot(*value)),
                I::IfElse(condition, label) => I::IfElse(frame.slot(*condition), label + offset),
                I::Goto(label) => I::Goto(label + offset),
                I::Label(label) => I::Label(label + offset),
                I::Join(o, condition, yes, no) => I::Join(
                    frame.slot(*o),
                    frame.slot(*condition),
                    frame.slot(*yes),
                    frame.slot(*no),
                ),
            };
            inlined.rows.push(mapped);
        }
        Ok(inlined)
    }

    /// The frame of a call's body: its temporaries, output and constants
    /// appended to the program's, its labels past every label in use.
    fn enter<'a>(
        &mut self,
        destination: symbolica::evaluate::Slot,
        arguments: Vec<symbolica::evaluate::Slot>,
        body: &'a Exported,
    ) -> RusticolResult<InlineFrame<'a>> {
        if body.output_count != 1 || arguments.len() != body.input_count {
            return Err(RusticolError::serialization(
                "a Symbolica function body must have one output and the call's arity",
            ));
        }
        let temporaries = self.temporaries;
        let output = temporaries + body.temporary_count;
        self.temporaries = output + 1;
        let constants = self.constants.len();
        self.constants.extend(body.constants.iter().cloned());
        let offset = self.next_label;
        self.next_label += 1 + max_label(&body.instructions).unwrap_or(0);
        Ok(InlineFrame {
            program: body,
            next: 0,
            call: Some(InlineCall {
                destination,
                arguments,
                temporaries,
                output,
                constants,
            }),
            offset,
        })
    }
}

/// One instruction through SymJIT's `Composer` (the calls of its text front
/// end, `Translator::parse_model`).
#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
fn compose(
    translator: &mut symjit::Translator,
    row: &symbolica::evaluate::Instruction,
) -> Result<(), String> {
    {
        use symbolica::evaluate::Instruction as I;
        use symjit::Composer;
        let slot = |slot: &symbolica::evaluate::Slot| match *slot {
            symbolica::evaluate::Slot::Param(i) => symjit::Slot::Param(i),
            symbolica::evaluate::Slot::Const(i) => symjit::Slot::Const(i),
            symbolica::evaluate::Slot::Temp(i) => symjit::Slot::Temp(i),
            symbolica::evaluate::Slot::Out(i) => symjit::Slot::Out(i),
        };
        let list = |slots: &[symbolica::evaluate::Slot]| slots.iter().map(slot).collect::<Vec<_>>();
        match row {
            I::Add(o, args, reals) => translator.append_add(&slot(o), &list(args), *reals),
            I::Mul(o, args, reals) => translator.append_mul(&slot(o), &list(args), *reals),
            I::Pow(o, base, exponent, real) => {
                translator.append_pow(&slot(o), &slot(base), *exponent, *real)
            }
            I::Powf(o, base, exponent, real) => {
                translator.append_powf(&slot(o), &slot(base), &slot(exponent), *real)
            }
            I::Fun(o, call, real) => {
                let (symbol, tags, args) = &**call;
                if !tags.is_empty() {
                    return Err(format!(
                        "SymJIT has no tagged function {}",
                        symbol.get_name()
                    ));
                }
                // The name of SymJIT's Symbolica front end: `symbolica_`
                // and the symbol as Symbolica prints it (built-ins without
                // their namespace).
                let name = if symbol.get_namespace() == "symbolica" {
                    symbol.get_stripped_name()
                } else {
                    symbol.get_name()
                };
                translator.append_fun(&slot(o), &format!("symbolica_{name}"), &list(args), *real)
            }
            I::Assign(o, value) => translator.append_assign(&slot(o), &slot(value)),
            I::IfElse(condition, label) => translator.append_if_else(&slot(condition), *label),
            I::Goto(label) => translator.append_goto(*label),
            I::Label(label) => translator.append_label(*label),
            I::Join(o, condition, yes, no) => {
                translator.append_join(&slot(o), &slot(condition), &slot(yes), &slot(no))
            }
        }
        .map_err(|error| error.to_string())
    }
}

#[cfg(all(
    feature = "symbolica-runtime",
    any(test, feature = "python-generation-bridge")
))]
fn max_label(rows: &[symbolica::evaluate::Instruction]) -> Option<usize> {
    use symbolica::evaluate::Instruction as I;
    rows.iter()
        .filter_map(|row| match row {
            I::IfElse(_, label) | I::Goto(label) | I::Label(label) => Some(*label),
            _ => None,
        })
        .max()
}

/// Loads a stored complex plane application through the production loader
/// and evaluates it at `point_count` points.
///
/// `inputs` holds (real, imaginary) per input, row-major by point; the result
/// holds (real, imaginary) per output, row-major by point. A cold-path bridge
/// for validating exports against Symbolica.
#[cfg(any(test, feature = "python-generation-bridge"))]
pub(crate) fn evaluate_plane_application_bytes(
    bytes: &[u8],
    input_complex_count: usize,
    output_complex_count: usize,
    inputs: &[f64],
    point_count: usize,
) -> RusticolResult<Vec<f64>> {
    let expected = input_complex_count
        .checked_mul(2)
        .and_then(|width| width.checked_mul(point_count))
        .ok_or_else(|| RusticolError::invalid_argument("plane input size overflows usize"))?;
    if inputs.len() != expected {
        return Err(RusticolError::invalid_argument(format!(
            "expected {expected} input values ({point_count} points of \
             {input_complex_count} complex inputs), got {}",
            inputs.len()
        )));
    }
    let layout = SymjitPlaneLayout::complex(input_complex_count, output_complex_count);
    let kernel = SymjitPlaneKernel::load_bytes(bytes, "<plane application>", layout)?;
    let input_planes = 2 * input_complex_count;
    let mut planes: Vec<Vec<f64>> = (0..input_planes + 2 * output_complex_count)
        .map(|plane| {
            if plane < input_planes {
                (0..point_count)
                    .map(|point| inputs[point * input_planes + plane])
                    .collect()
            } else {
                vec![f64::NAN; point_count]
            }
        })
        .collect();
    {
        let mut table = kernel.build_raw_table(
            planes
                .iter_mut()
                .map(|plane| PlaneDescriptor::from_slice(plane))
                .collect(),
        )?;
        kernel.execute_table(&mut table, 0, point_count)?;
    }
    let output_planes = &planes[input_planes..];
    Ok((0..point_count)
        .flat_map(|point| output_planes.iter().map(move |plane| plane[point]))
        .collect())
}

fn explicit_plane_config(complex: bool, opt_level: u8, compress: bool) -> RusticolResult<Config> {
    if opt_level > MAX_OPTIMIZATION_LEVEL {
        return Err(RusticolError::invalid_argument(format!(
            "SymJIT optimization level {opt_level} is unsupported; expected 0 through {MAX_OPTIMIZATION_LEVEL}"
        )));
    }
    let mut config = Config::new(CompilerType::Native, 0).map_err(|error| {
        RusticolError::internal(format!(
            "could not construct explicit SymJIT native configuration: {error}"
        ))
    })?;
    config.set_opt_level(opt_level);
    config.set_cse(true);
    config.set_fastmath(true);
    config.set_simd(true);
    // pyAmpliCol's portable direct-plane contract is f64x4 on x86 and
    // f64x2 on AArch64. AVX-512 remains an explicit future ABI.
    config.enable_simd512(false);
    config.set_simd_branch(false);
    config.set_complex(complex);
    config.set_fast_complex(false);
    config.set_threads(false);
    config.set_symbolica(true);
    config.set_compact(true);
    config.set_compress(compress);
    config.set_direct(false);
    config.set_huge(false);
    config.set_parallel_mul(true);
    config.set_direct_arena(true);
    // Standard P-kernels own only an identity overwrite. pyAmpliCol applies
    // factors and accumulation in its alias-safe arena epilogues. Set the
    // reserved operation option and the active 2.22 identity-output option
    // explicitly so a future SymJIT implementation cannot silently change
    // persisted application semantics.
    config.set_direct_arena_operation(0);
    config.set_direct_arena_identity_output(true);
    config.set_defuns(scalar_primitive_defuns()?);
    Ok(config)
}

/// pyAmpliCol's version-one scalar primitives (`source_scalar`) as SymJIT
/// external functions.
///
/// SymJIT's Symbolica front end names a `fun` call `symbolica_<symbol>`; a
/// stored application keeps that name and SymJIT binds it again when it
/// recompiles the IR at load. The compiler and every plane loader register the
/// same Rust functions, so an application calling them (a model's helper
/// functions, inlined by the exporter) is relocatable and evaluates without
/// Symbolica state. A call of any other non-built-in function is refused by the
/// compiler.
fn scalar_primitive_defuns() -> RusticolResult<Defuns> {
    use num_complex::Complex;
    let mut defuns = Defuns::new();
    for (name, mode) in crate::scalar_primitives::SCALAR_PRIMITIVES {
        defuns
            .add_sliced_func::<Complex<f64>>(
                &format!("symbolica_{name}"),
                Box::new(move |args: &[Complex<f64>]| {
                    let mut pairs = [(0.0, 0.0); 3];
                    for (pair, value) in pairs.iter_mut().zip(args) {
                        *pair = (value.re, value.im);
                    }
                    let (re, im) = crate::scalar_primitives::scalar_primitive_f64(
                        &pairs[..args.len().min(pairs.len())],
                        mode,
                    );
                    Complex::new(re, im)
                }),
            )
            .map_err(|error| {
                RusticolError::internal(format!(
                    "could not register the SymJIT scalar primitive {name}: {error}"
                ))
            })?;
    }
    Ok(defuns)
}

/// The x86-64 features that decide SymJIT's code path, as this CPU reports
/// them (CPUID with the OS's XSAVE state, as `is_x86_feature_detected!` reads
/// them; under user-mode emulation the emulated CPU's): `sse2`, `sse4_2`,
/// `avx`, `fma`, `avx2`, `avx512f`. Empty on other architectures.
pub(crate) fn symjit_cpu_features() -> Vec<&'static str> {
    #[cfg(target_arch = "x86_64")]
    {
        [
            ("sse2", std::is_x86_feature_detected!("sse2")),
            ("sse4_2", std::is_x86_feature_detected!("sse4.2")),
            ("avx", std::is_x86_feature_detected!("avx")),
            ("fma", std::is_x86_feature_detected!("fma")),
            ("avx2", std::is_x86_feature_detected!("avx2")),
            ("avx512f", std::is_x86_feature_detected!("avx512f")),
        ]
        .into_iter()
        .filter_map(|(name, present)| present.then_some(name))
        .collect()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec::new()
    }
}

/// Whether SymJIT's load-time code generation runs on this CPU
/// ([`require_symjit_host_cpu`]): x86-64 with AVX and FMA3, or another
/// architecture. Until a SymJIT release with the upstream fixes is pinned
/// (siravan/symjit issues 22 and 23, pull requests 24 and 25) this gate is
/// the fallback; afterwards it stays as a fallback.
pub(crate) fn symjit_host_cpu_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        true
    }
}

/// Refuses a CPU on which SymJIT's load-time code generation would crash.
///
/// Stored applications are IR that SymJIT 2.27 recompiles for the loading CPU
/// (`CompilerType::Native`). Its x86-64 kernels need AVX and FMA3: the shared
/// epilogue emits `vzeroupper` (AVX) for every generator, the SSE one
/// included, and the AVX generator, chosen from the AVX CPUID bit alone, also
/// emits FMA3 instructions (complex products, fused multiply-add). On a CPU
/// without AVX (x86-64-v1/v2) or with AVX and without FMA3 (Intel Sandy and
/// Ivy Bridge, AMD Bulldozer, a virtual-machine CPU model masking them) a
/// kernel dies with SIGILL, which no caller can recover. Such a CPU is refused
/// here with the reason.
pub(crate) fn require_symjit_host_cpu(display_path: &Path) -> RusticolResult<()> {
    if symjit_host_cpu_supported() {
        return Ok(());
    }
    #[cfg(target_arch = "x86_64")]
    {
        let missing: Vec<&str> = [
            ("AVX", std::is_x86_feature_detected!("avx")),
            ("FMA3", std::is_x86_feature_detected!("fma")),
        ]
        .into_iter()
        .filter_map(|(name, present)| (!present).then_some(name))
        .collect();
        if !missing.is_empty() {
            return Err(RusticolError::compatibility(format!(
                "cannot compile SymJIT application {} on this CPU: it lacks {}, which \
                 the SymJIT code generator's x86-64 kernels use; use an x86-64 CPU with \
                 AVX and FMA3 (Intel Haswell, AMD Piledriver or newer) or AArch64",
                display_path.display(),
                missing.join(" and ")
            )));
        }
    }
    let _ = display_path;
    Ok(())
}

pub(crate) fn explicit_loader_config() -> RusticolResult<Config> {
    let mut config = Config::new(CompilerType::Native, 0).map_err(|error| {
        RusticolError::internal(format!(
            "could not construct explicit SymJIT loader configuration: {error}"
        ))
    })?;
    config.set_opt_level(0);
    config.set_cse(false);
    config.set_fastmath(false);
    config.set_simd(false);
    config.enable_simd512(false);
    config.set_simd_branch(false);
    config.set_complex(false);
    config.set_fast_complex(false);
    config.set_threads(false);
    config.set_symbolica(false);
    config.set_compact(false);
    config.set_compress(false);
    config.set_direct(false);
    config.set_huge(false);
    config.set_parallel_mul(false);
    config.set_direct_arena(false);
    Ok(config)
}

fn authenticate_storage_v4(bytes: &[u8], display_path: &Path) -> RusticolResult<()> {
    let word = std::mem::size_of::<usize>();
    let header_len = word
        .checked_mul(2)
        .ok_or_else(|| RusticolError::internal("SymJIT storage header length overflows usize"))?;
    if word != 8 {
        return Err(RusticolError::compatibility(
            "SymJIT storage-v4 plane applications require a 64-bit target",
        ));
    }
    if bytes.len() < header_len {
        return Err(RusticolError::integrity(format!(
            "SymJIT plane application {} is truncated before its storage-v4 header",
            display_path.display()
        )));
    }
    let magic = usize::from_le_bytes(
        bytes[..word]
            .try_into()
            .map_err(|_| RusticolError::internal("could not decode SymJIT application magic"))?,
    );
    if magic != SYMJIT_APPLICATION_MAGIC {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} has invalid application magic",
            display_path.display()
        )));
    }
    let version = usize::from_le_bytes(
        bytes[word..header_len]
            .try_into()
            .map_err(|_| RusticolError::internal("could not decode SymJIT storage version"))?,
    );
    if version != SYMJIT_APPLICATION_STORAGE_VERSION {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} uses storage version {version}, expected {SYMJIT_APPLICATION_STORAGE_VERSION}",
            display_path.display()
        )));
    }
    Ok(())
}

fn validate_application(
    application: &Application,
    display_path: &Path,
    layout: SymjitPlaneLayout,
) -> RusticolResult<u8> {
    let config = &application.config;
    if config.is_enabled_simd512() {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} enables unsupported SIMD512 code generation; \
             regenerate it for the portable scalar/f64x2/f64x4 arena ABI",
            display_path.display()
        )));
    }
    let optimization_level = canonical_plane_optimization_level(config, display_path, layout)?;
    if !config.symbolica()
        || !config.direct_arena()
        || config.direct()
        || config.direct_arena_operation() != 0
        || !config.direct_arena_identity_output()
    {
        return Err(RusticolError::compatibility(format!(
            "SymJIT application {} is not an identity-overwrite indirect Symbolica \
             direct-arena P-kernel",
            display_path.display()
        )));
    }
    if config.is_complex() != layout.is_complex() {
        return Err(RusticolError::integrity(format!(
            "SymJIT plane application {} complex mode does not match its binding layout",
            display_path.display()
        )));
    }
    let input_plane_count = layout.input_plane_count()?;
    let output_plane_count = layout.output_plane_count()?;
    if application.count_states != input_plane_count {
        return Err(RusticolError::integrity(format!(
            "SymJIT plane application {} stores {} input planes, expected {}",
            display_path.display(),
            application.count_states,
            input_plane_count
        )));
    }
    if application.count_obs != output_plane_count {
        return Err(RusticolError::integrity(format!(
            "SymJIT plane application {} stores {} output planes, expected {}",
            display_path.display(),
            application.count_obs,
            output_plane_count
        )));
    }
    if application.count_params != 0 {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} retains {} scalar parameters; direct-arena inputs must all be states",
            display_path.display(),
            application.count_params
        )));
    }
    if application.count_diffs != 0 {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} contains differential outputs",
            display_path.display()
        )));
    }
    Ok(optimization_level)
}

fn canonical_plane_optimization_level(
    config: &Config,
    display_path: &Path,
    layout: SymjitPlaneLayout,
) -> RusticolResult<u8> {
    if config.ty != CompilerType::Native {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} uses a non-native compiler target; \
             regenerate it with the canonical pyAmpliCol plane configuration",
            display_path.display()
        )));
    }
    let mut canonical_optimization_level = None;
    for optimization_level in 0..=MAX_OPTIMIZATION_LEVEL {
        let expected =
            explicit_plane_config(layout.is_complex(), optimization_level, config.compress())?;
        if config.opt == expected.opt {
            canonical_optimization_level = Some(optimization_level);
            break;
        }
    }
    canonical_optimization_level.ok_or_else(|| {
        RusticolError::compatibility(format!(
            "SymJIT plane application {} uses noncanonical compiler options; \
             regenerate it with the canonical pyAmpliCol plane configuration",
            display_path.display()
        ))
    })
}

fn validate_simd_lane_width(
    has_simd_kernel: bool,
    simd_lanes: usize,
    display_path: &Path,
) -> RusticolResult<()> {
    let valid = if has_simd_kernel {
        matches!(simd_lanes, 2 | 4)
    } else {
        simd_lanes == 1
    };
    if !valid {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} reports unsupported SIMD lane width {simd_lanes}; \
             expected scalar fallback, f64x2, or f64x4",
            display_path.display()
        )));
    }
    Ok(())
}

fn validate_compiled_kernels(application: &Application, display_path: &Path) -> RusticolResult<()> {
    let scalar = application.compiled.as_ref().ok_or_else(|| {
        RusticolError::compatibility(format!(
            "SymJIT plane application {} has no scalar native kernel",
            display_path.display()
        ))
    })?;
    if !scalar.support_indirect() {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} scalar kernel does not support indirect plane descriptors",
            display_path.display()
        )));
    }
    if let Some(simd) = application.compiled_simd.as_ref()
        && (!simd.support_indirect() || simd.count_lanes() <= 1)
    {
        return Err(RusticolError::compatibility(format!(
            "SymJIT plane application {} has an invalid indirect SIMD kernel",
            display_path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
fn validate_slice_table(layout: SymjitPlaneLayout, planes: &[&mut [f64]]) -> RusticolResult<()> {
    let expected = layout.plane_count()?;
    if planes.len() != expected {
        return Err(RusticolError::invalid_argument(format!(
            "SymJIT plane table has {} planes, expected {expected}",
            planes.len()
        )));
    }
    let point_count = planes.first().map_or(0, |plane| plane.len());
    if let Some((index, plane)) = planes
        .iter()
        .enumerate()
        .find(|(_, plane)| plane.len() != point_count)
    {
        return Err(RusticolError::invalid_argument(format!(
            "SymJIT plane {index} has length {}, expected {point_count}",
            plane.len()
        )));
    }
    Ok(())
}

fn validate_raw_table(
    layout: SymjitPlaneLayout,
    planes: &[SymjitRawPlane<'_>],
) -> RusticolResult<()> {
    let expected = layout.plane_count()?;
    validate_raw_plane_count_and_ranges(expected, planes)
}

fn validate_raw_plane_count_and_ranges(
    expected: usize,
    planes: &[SymjitRawPlane<'_>],
) -> RusticolResult<()> {
    if planes.len() != expected {
        return Err(RusticolError::invalid_argument(format!(
            "SymJIT raw plane table has {} planes, expected {expected}",
            planes.len()
        )));
    }
    let point_count = planes.first().map_or(0, PlaneDescriptor::len);
    for (index, plane) in planes.iter().enumerate() {
        if plane.len() != point_count {
            return Err(RusticolError::invalid_argument(format!(
                "SymJIT raw plane {index} has length {}, expected {point_count}",
                plane.len()
            )));
        }
        let _ = plane.interval()?;
    }
    Ok(())
}

fn validate_raw_aliases(
    input_plane_count: usize,
    output_plane_count: usize,
    planes: &[SymjitRawPlane<'_>],
) -> RusticolResult<bool> {
    let expected = input_plane_count
        .checked_add(output_plane_count)
        .ok_or_else(|| RusticolError::invalid_argument("SymJIT plane count overflows usize"))?;
    if planes.len() != expected {
        return Err(RusticolError::invalid_argument(format!(
            "SymJIT raw plane table has {} planes, expected {expected}",
            planes.len()
        )));
    }

    let (inputs, outputs) = planes.split_at(input_plane_count);
    for (index, output) in outputs.iter().enumerate() {
        let Some(output_interval) = output.interval()? else {
            continue;
        };
        for other in &outputs[index + 1..] {
            let Some(other_interval) = other.interval()? else {
                continue;
            };
            if output_interval.0 < other_interval.1 && other_interval.0 < output_interval.1 {
                return Err(RusticolError::invalid_argument(
                    "SymJIT output planes overlap; output planes must be mutually disjoint",
                ));
            }
        }
    }

    let mut disjoint = true;
    for input in inputs {
        let Some(input_interval) = input.interval()? else {
            continue;
        };
        for output in outputs {
            let Some(output_interval) = output.interval()? else {
                continue;
            };
            if input_interval.0 < output_interval.1 && output_interval.0 < input_interval.1 {
                if input_interval == output_interval {
                    disjoint = false;
                    continue;
                }
                return Err(RusticolError::invalid_argument(
                    "a SymJIT input/output plane pair partially overlaps; only disjoint planes or exact aliases are supported",
                ));
            }
        }
    }
    Ok(disjoint)
}

fn guard_symjit_panic<T>(
    operation: impl FnOnce() -> T,
    action: impl FnOnce() -> String,
) -> RusticolResult<T> {
    catch_unwind(AssertUnwindSafe(operation)).map_err(|payload| {
        RusticolError::compatibility(format!(
            "SymJIT panicked while trying to {}: {}",
            action(),
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
    use crate::RusticolErrorKind;

    #[cfg(feature = "symbolica-runtime")]
    #[test]
    fn side_inputs_read_a_selection_operand_on_the_path_that_chose_it() {
        // `real_axis_side_inputs` on inputs a, b, s, m (Symbolica's lowering
        // of `if(a - b, m^2, m)` reusing the temporary that held s^2): the
        // selection on the undecided a - b reads m^2 on the branch that wrote
        // it, not the stale s^2 left in the slot on the other path, so the
        // later cut log(sel - b) marks a, b (the branch) and b, m (its
        // argument), not s; when the condition's slot is written between the
        // branch and the selection, the branch's fact lapses and s counts.
        use symbolica::atom::Symbol;
        use symbolica::evaluate::{Instruction as I, Slot};
        let (p, t, c) = (Slot::Param, Slot::Temp, Slot::Const);
        let log = |o: Slot, x: Slot| I::Fun(o, Box::new((Symbol::LOG, Vec::new(), vec![x])), false);
        let program = |lapse: bool| {
            let mut rows = vec![
                I::Mul(t(2), vec![p(2), p(2)], 0),
                I::Assign(Slot::Out(0), t(2)),
                I::Mul(t(3), vec![c(0), p(1)], 0),
                I::Add(t(0), vec![p(0), t(3)], 0),
                I::IfElse(t(0), 1),
                I::Mul(t(2), vec![p(3), p(3)], 0),
                I::Goto(2),
                I::Label(1),
                I::Label(2),
            ];
            if lapse {
                rows.push(I::Assign(t(0), t(0)));
            }
            rows.extend([
                I::Join(t(4), t(0), t(2), p(3)),
                I::Add(t(5), vec![t(4), t(3)], 0),
                log(Slot::Out(1), t(5)),
            ]);
            InlinedProgram {
                rows,
                constants: vec![symbolica::domains::float::Complex::new(-1.0, 0.0)],
                temporaries: 6,
                next_label: 3,
            }
        };
        let point = [2.0, 1.0, 1000.0, 91.188];
        let side = |lapse: bool| side_inputs_of(&program(lapse), 4, 2, &point).unwrap();
        assert_eq!(side(false), vec![0, 1, 3]);
        assert_eq!(side(true), vec![0, 1, 2, 3]);
    }

    #[cfg(feature = "symbolica-runtime")]
    #[test]
    fn a_fixed_sign_real_factor_moves_no_side() {
        // `real_axis_side_inputs` on inputs a, b, c, d at positive values:
        // (b - d) / c^2 has signs that only b and d can change (c^2 > 0
        // scales both parts), so a cut or a zero test on it marks b and d,
        // not c (the scale in the complex-mass scheme's counterterms); a
        // comparison of two values marks every input they depend on.
        use symbolica::atom::{AtomCore, Symbol};
        use symbolica::evaluate::{Instruction as I, Slot};
        let (p, t, c) = (Slot::Param, Slot::Temp, Slot::Const);
        let fun = |name: Symbol, o: Slot, args: Vec<Slot>| {
            I::Fun(o, Box::new((name, Vec::new(), args)), false)
        };
        // The primitives' registered symbols (a bare parse would define
        // them without their evaluation).
        crate::source_scalar::register_scalar_primitives();
        let less = {
            let _unlock = crate::licence::unlock();
            symbolica::parse!("pyamplicol_scalar_v1::less(pyamplicol_symjit_tests::x)")
                .get_symbol()
                .expect("a function")
        };
        let program = |last: I| InlinedProgram {
            rows: vec![
                I::Mul(t(0), vec![c(0), p(3)], 0),
                I::Add(t(1), vec![p(1), t(0)], 0),
                I::Pow(t(2), p(2), -2, false),
                I::Mul(t(3), vec![t(2), t(1)], 0),
                last,
            ],
            constants: vec![
                symbolica::domains::float::Complex::new(-1.0, 0.0),
                symbolica::domains::float::Complex::new(0.0, 0.0),
            ],
            temporaries: 4,
            next_label: 1,
        };
        let point = [0.118, 91.188, 1000.0, 80.4];
        let side = |last: I| side_inputs_of(&program(last), 4, 1, &point).unwrap();
        assert_eq!(side(fun(Symbol::LOG, Slot::Out(0), vec![t(3)])), vec![1, 3]);
        assert_eq!(side(fun(less, Slot::Out(0), vec![t(3), c(1)])), vec![1, 3]);
        assert_eq!(
            side(fun(less, Slot::Out(0), vec![t(3), p(0)])),
            vec![0, 1, 2, 3]
        );
    }

    const MODEL: &str = "
([('fun', ('temp', 0), 'square', [], [('param', 0)], False),
  ('add', ('temp', 1), [('temp', 0), ('param', 1)], 0),
  ('assign', ('out', 0), ('temp', 1))],
 2,
 [])
";
    const DIVERGENT_MODEL: &str = "
([('if_else', ('param', 0), 2),
  ('goto', 3),
  ('label', 2),
  ('label', 3),
  ('join', ('out', 0), ('param', 0), ('param', 1), ('param', 2))],
 2,
 [])
";
    const LENGTHS: &[usize] = &[1, 2, 3, 7, 8, 127, 128, 129, 1023, 1024, 1025];
    const SENTINEL: f64 = -9.876_543_210_123_456e211;

    fn application_bytes(complex: bool, opt_level: u8) -> Vec<u8> {
        let config = explicit_plane_config(complex, opt_level, true).unwrap();
        application_bytes_with_config(config)
    }

    fn application_bytes_with_config(config: Config) -> Vec<u8> {
        let mut compiler = Compiler::with_config(config);
        let mut application = compiler.translate(MODEL.to_string(), 2).unwrap();
        application.prepare_simd();
        let mut bytes = Vec::new();
        application.save(&mut bytes).unwrap();
        bytes
    }

    fn divergent_application_bytes() -> Vec<u8> {
        let config = explicit_plane_config(false, 3, true).unwrap();
        let mut compiler = Compiler::with_config(config);
        let mut application = compiler.translate(DIVERGENT_MODEL.to_string(), 3).unwrap();
        application.prepare_simd();
        let mut bytes = Vec::new();
        application.save(&mut bytes).unwrap();
        bytes
    }

    fn guarded_plane(len: usize, values: impl Fn(usize) -> f64) -> Vec<f64> {
        let mut plane = vec![SENTINEL; len + 2];
        for index in 0..len {
            plane[index + 1] = values(index);
        }
        plane
    }

    fn assert_guards(plane: &[f64]) {
        assert_eq!(plane.first().copied(), Some(SENTINEL));
        assert_eq!(plane.last().copied(), Some(SENTINEL));
    }

    fn assert_native_simd_lanes(kernel: &SymjitPlaneKernel) {
        #[cfg(target_arch = "aarch64")]
        assert_eq!(kernel.simd_lanes(), 2);
        #[cfg(target_arch = "x86_64")]
        assert_eq!(
            kernel.simd_lanes(),
            if std::is_x86_feature_detected!("avx") {
                4
            } else {
                1
            }
        );
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        assert_eq!(kernel.simd_lanes(), 1);
    }

    fn execute_forced_scalar(kernel: &SymjitPlaneKernel, table: &SymjitPlaneTable<'_>, len: usize) {
        for point in 0..len {
            let status = unsafe {
                (kernel.scalar)(
                    std::ptr::null(),
                    table.as_symjit_ptr(),
                    point,
                    kernel.applet.params.as_ptr(),
                )
            };
            assert_eq!(status, 0, "scalar status at point {point}");
        }
    }

    fn execute_forced_simd(kernel: &SymjitPlaneKernel, table: &SymjitPlaneTable<'_>, len: usize) {
        let mut point = 0;
        let mut simd_call_count = 0;
        if let Some(simd) = kernel.simd.filter(|_| kernel.simd_lanes() > 1) {
            while len - point >= kernel.simd_lanes() {
                let status = unsafe {
                    simd(
                        std::ptr::null(),
                        table.as_symjit_ptr(),
                        point,
                        kernel.applet.params.as_ptr(),
                    )
                };
                assert_eq!(status, 0, "SIMD status at row {point}");
                point += kernel.simd_lanes();
                simd_call_count += 1;
            }
        }
        if kernel.simd_lanes() > 1 && len >= kernel.simd_lanes() {
            assert!(simd_call_count > 0, "forced SIMD path executed no blocks");
        }
        while point < len {
            let status = unsafe {
                (kernel.scalar)(
                    std::ptr::null(),
                    table.as_symjit_ptr(),
                    point,
                    kernel.applet.params.as_ptr(),
                )
            };
            assert_eq!(status, 0, "scalar tail status at point {point}");
            point += 1;
        }
    }

    #[test]
    fn real_forced_scalar_and_simd_preserve_sentinels_for_all_lengths() {
        let bytes = application_bytes(false, 3);
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "real-contract.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap();
        assert_eq!(kernel.input_plane_count(), 2);
        assert_eq!(kernel.output_plane_count(), 1);
        assert_native_simd_lanes(&kernel);

        for &len in LENGTHS {
            let mut x = guarded_plane(len, |i| i as f64 * 0.25 - 3.0);
            let mut y = guarded_plane(len, |i| 7.0 - i as f64 * 0.125);
            let mut scalar_out = guarded_plane(len, |_| f64::NAN);
            let mut simd_out = guarded_plane(len, |_| f64::NAN);
            {
                let table = kernel
                    .build_raw_table(vec![
                        PlaneDescriptor::from_slice(&mut x[1..len + 1]),
                        PlaneDescriptor::from_slice(&mut y[1..len + 1]),
                        PlaneDescriptor::from_slice(&mut scalar_out[1..len + 1]),
                    ])
                    .unwrap();
                assert_eq!(table.point_count(), len);
                execute_forced_scalar(&kernel, &table, len);
            }
            {
                let table = kernel
                    .build_raw_table(vec![
                        PlaneDescriptor::from_slice(&mut x[1..len + 1]),
                        PlaneDescriptor::from_slice(&mut y[1..len + 1]),
                        PlaneDescriptor::from_slice(&mut simd_out[1..len + 1]),
                    ])
                    .unwrap();
                execute_forced_simd(&kernel, &table, len);
            }
            for index in 0..len {
                let expected = x[index + 1] * x[index + 1] + y[index + 1];
                assert_eq!(
                    scalar_out[index + 1],
                    expected,
                    "scalar length {len}, point {index}"
                );
                assert_eq!(
                    simd_out[index + 1],
                    expected,
                    "SIMD length {len}, point {index}"
                );
            }
            assert_guards(&x);
            assert_guards(&y);
            assert_guards(&scalar_out);
            assert_guards(&simd_out);
        }
    }

    #[test]
    fn complex_forced_scalar_and_simd_cover_every_length() {
        let bytes =
            compile_symbolica_program_to_plane_application_bytes(MODEL, 2, 1, 3, true).unwrap();
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "complex-contract.symjit",
            SymjitPlaneLayout::complex(2, 1),
        )
        .unwrap();
        assert_eq!(kernel.input_plane_count(), 4);
        assert_eq!(kernel.output_plane_count(), 2);
        assert_native_simd_lanes(&kernel);

        for &len in LENGTHS {
            let mut x_re = guarded_plane(len, |i| i as f64 * 0.25 - 1.0);
            let mut x_im = guarded_plane(len, |i| 0.5 + i as f64 * 0.0625);
            let mut y_re = guarded_plane(len, |i| 2.0 - i as f64 * 0.125);
            let mut y_im = guarded_plane(len, |i| -0.75 + i as f64 * 0.03125);
            let mut scalar_out_re = guarded_plane(len, |_| f64::NAN);
            let mut scalar_out_im = guarded_plane(len, |_| f64::NAN);
            let mut simd_out_re = guarded_plane(len, |_| f64::NAN);
            let mut simd_out_im = guarded_plane(len, |_| f64::NAN);
            {
                let table = kernel
                    .build_raw_table(vec![
                        PlaneDescriptor::from_slice(&mut x_re[1..len + 1]),
                        PlaneDescriptor::from_slice(&mut x_im[1..len + 1]),
                        PlaneDescriptor::from_slice(&mut y_re[1..len + 1]),
                        PlaneDescriptor::from_slice(&mut y_im[1..len + 1]),
                        PlaneDescriptor::from_slice(&mut scalar_out_re[1..len + 1]),
                        PlaneDescriptor::from_slice(&mut scalar_out_im[1..len + 1]),
                    ])
                    .unwrap();
                execute_forced_scalar(&kernel, &table, len);
            }
            {
                let table = kernel
                    .build_raw_table(vec![
                        PlaneDescriptor::from_slice(&mut x_re[1..len + 1]),
                        PlaneDescriptor::from_slice(&mut x_im[1..len + 1]),
                        PlaneDescriptor::from_slice(&mut y_re[1..len + 1]),
                        PlaneDescriptor::from_slice(&mut y_im[1..len + 1]),
                        PlaneDescriptor::from_slice(&mut simd_out_re[1..len + 1]),
                        PlaneDescriptor::from_slice(&mut simd_out_im[1..len + 1]),
                    ])
                    .unwrap();
                execute_forced_simd(&kernel, &table, len);
            }
            for index in 0..len {
                let xr = x_re[index + 1];
                let xi = x_im[index + 1];
                let expected_re = xr * xr - xi * xi + y_re[index + 1];
                let expected_im = 2.0 * xr * xi + y_im[index + 1];
                for (lane, actual_re, actual_im) in [
                    ("scalar", scalar_out_re[index + 1], scalar_out_im[index + 1]),
                    ("SIMD", simd_out_re[index + 1], simd_out_im[index + 1]),
                ] {
                    assert_eq!(
                        actual_re, expected_re,
                        "{lane} length {len}, point {index}, real"
                    );
                    assert_eq!(
                        actual_im, expected_im,
                        "{lane} length {len}, point {index}, imaginary"
                    );
                }
            }
            for plane in [
                &x_re,
                &x_im,
                &y_re,
                &y_im,
                &scalar_out_re,
                &scalar_out_im,
                &simd_out_re,
                &simd_out_im,
            ] {
                assert_guards(plane);
            }
        }
    }

    #[test]
    fn unaligned_subrange_uses_scalar_head_and_tail() {
        let bytes = application_bytes(false, 2);
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "unaligned-contract.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap();
        let len = 19;
        let mut x: Vec<f64> = (0..len).map(|i| i as f64 + 1.0).collect();
        let mut y: Vec<f64> = (0..len).map(|i| i as f64 * 0.5).collect();
        let mut out = vec![SENTINEL; len];
        {
            let mut bound = kernel
                .bind(vec![x.as_mut_slice(), y.as_mut_slice(), out.as_mut_slice()])
                .unwrap();
            bound.execute_range(1, len - 3).unwrap();
        }
        assert_eq!(out[0], SENTINEL);
        assert_eq!(out[len - 2], SENTINEL);
        assert_eq!(out[len - 1], SENTINEL);
        for index in 1..len - 2 {
            assert_eq!(out[index], x[index] * x[index] + y[index]);
        }
    }

    #[test]
    fn raw_binding_executes_duplicate_inputs_without_overlapping_references() {
        let bytes = application_bytes(false, 2);
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "duplicate-contract.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap();
        for &len in LENGTHS {
            let mut input: Vec<f64> = (1..=len).map(|value| value as f64).collect();
            let mut output = vec![0.0; len];
            let duplicate = input.as_mut_ptr();
            {
                let mut bound = kernel
                    .bind_raw(vec![
                        // SAFETY: Both descriptors share live, synchronized
                        // read-only input storage for the duration of execution.
                        unsafe { PlaneDescriptor::from_raw_parts(duplicate, input.len()).unwrap() },
                        unsafe { PlaneDescriptor::from_raw_parts(duplicate, input.len()).unwrap() },
                        PlaneDescriptor::from_slice(output.as_mut_slice()),
                    ])
                    .unwrap();
                bound.execute_all().unwrap();
            }
            for (index, value) in input.iter().copied().enumerate() {
                assert_eq!(
                    output[index],
                    value * value + value,
                    "length {len}, point {index}"
                );
            }
        }
    }

    #[test]
    fn raw_binding_executes_input_output_alias_from_prologue_snapshot() {
        let bytes = application_bytes(false, 2);
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "alias-contract.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap();
        for &len in LENGTHS {
            let mut first: Vec<f64> = (1..=len).map(|value| value as f64).collect();
            let original_first = first.clone();
            let mut second: Vec<f64> = (0..len).map(|index| 10.0 + index as f64).collect();
            let alias = first.as_mut_ptr();
            {
                let mut bound = kernel
                    .bind_raw(vec![
                        // SAFETY: The P-kernel snapshots its inputs in the
                        // prologue before overwriting the aliased output.
                        unsafe { PlaneDescriptor::from_raw_parts(alias, first.len()).unwrap() },
                        PlaneDescriptor::from_slice(second.as_mut_slice()),
                        unsafe { PlaneDescriptor::from_raw_parts(alias, first.len()).unwrap() },
                    ])
                    .unwrap();
                assert!(!bound.table.simd_safe());
                bound.execute_all().unwrap();
            }
            for index in 0..first.len() {
                assert_eq!(
                    first[index],
                    original_first[index] * original_first[index] + second[index],
                    "length {len}, point {index}"
                );
            }
        }
    }

    #[test]
    fn complex_raw_binding_supports_duplicate_inputs_for_all_lengths() {
        let bytes =
            compile_symbolica_program_to_plane_application_bytes(MODEL, 2, 1, 2, true).unwrap();
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "complex-duplicate-contract.symjit",
            SymjitPlaneLayout::complex(2, 1),
        )
        .unwrap();
        for &len in LENGTHS {
            let mut input_re: Vec<f64> = (0..len).map(|index| 0.5 + index as f64 * 0.25).collect();
            let mut input_im: Vec<f64> =
                (0..len).map(|index| -0.75 + index as f64 * 0.125).collect();
            let mut output_re = vec![0.0; len];
            let mut output_im = vec![0.0; len];
            let input_re_pointer = input_re.as_mut_ptr();
            let input_im_pointer = input_im.as_mut_ptr();
            {
                let mut bound = kernel
                    .bind_raw(vec![
                        unsafe {
                            PlaneDescriptor::from_raw_parts(input_re_pointer, input_re.len())
                                .unwrap()
                        },
                        unsafe {
                            PlaneDescriptor::from_raw_parts(input_im_pointer, input_im.len())
                                .unwrap()
                        },
                        unsafe {
                            PlaneDescriptor::from_raw_parts(input_re_pointer, input_re.len())
                                .unwrap()
                        },
                        unsafe {
                            PlaneDescriptor::from_raw_parts(input_im_pointer, input_im.len())
                                .unwrap()
                        },
                        PlaneDescriptor::from_slice(output_re.as_mut_slice()),
                        PlaneDescriptor::from_slice(output_im.as_mut_slice()),
                    ])
                    .unwrap();
                bound.execute_all().unwrap();
            }
            for index in 0..len {
                let real = input_re[index];
                let imaginary = input_im[index];
                let expected_re = real * real - imaginary * imaginary + real;
                let expected_im = 2.0 * real * imaginary + imaginary;
                assert_eq!(
                    output_re[index], expected_re,
                    "length {len}, point {index}, real"
                );
                assert_eq!(
                    output_im[index], expected_im,
                    "length {len}, point {index}, imaginary"
                );
            }
        }
    }

    #[test]
    fn complex_raw_binding_supports_exact_output_aliases_for_all_lengths() {
        let bytes =
            compile_symbolica_program_to_plane_application_bytes(MODEL, 2, 1, 2, true).unwrap();
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "complex-alias-contract.symjit",
            SymjitPlaneLayout::complex(2, 1),
        )
        .unwrap();
        for &len in LENGTHS {
            let mut first_re: Vec<f64> = (0..len).map(|index| 1.0 + index as f64 * 0.25).collect();
            let mut first_im: Vec<f64> =
                (0..len).map(|index| -0.5 + index as f64 * 0.125).collect();
            let original_first: Vec<(f64, f64)> = first_re
                .iter()
                .copied()
                .zip(first_im.iter().copied())
                .collect();
            let mut second_re: Vec<f64> =
                (0..len).map(|index| 3.0 - index as f64 * 0.125).collect();
            let mut second_im: Vec<f64> =
                (0..len).map(|index| 0.25 + index as f64 * 0.0625).collect();
            let first_re_pointer = first_re.as_mut_ptr();
            let first_im_pointer = first_im.as_mut_ptr();
            {
                let mut bound = kernel
                    .bind_raw(vec![
                        unsafe {
                            PlaneDescriptor::from_raw_parts(first_re_pointer, first_re.len())
                                .unwrap()
                        },
                        unsafe {
                            PlaneDescriptor::from_raw_parts(first_im_pointer, first_im.len())
                                .unwrap()
                        },
                        PlaneDescriptor::from_slice(second_re.as_mut_slice()),
                        PlaneDescriptor::from_slice(second_im.as_mut_slice()),
                        unsafe {
                            PlaneDescriptor::from_raw_parts(first_re_pointer, first_re.len())
                                .unwrap()
                        },
                        unsafe {
                            PlaneDescriptor::from_raw_parts(first_im_pointer, first_im.len())
                                .unwrap()
                        },
                    ])
                    .unwrap();
                assert!(!bound.table.simd_safe());
                bound.execute_all().unwrap();
            }
            for index in 0..len {
                let (real, imaginary) = original_first[index];
                let expected_re = real * real - imaginary * imaginary + second_re[index];
                let expected_im = 2.0 * real * imaginary + second_im[index];
                assert_eq!(
                    first_re[index], expected_re,
                    "length {len}, point {index}, real"
                );
                assert_eq!(
                    first_im[index], expected_im,
                    "length {len}, point {index}, imaginary"
                );
            }
        }
    }

    #[test]
    fn shifted_input_output_overlap_is_rejected() {
        let bytes = application_bytes(false, 2);
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "partial-overlap-contract.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap();
        let len = 8;
        let mut overlapping = vec![1.0; len + 1];
        let base = overlapping.as_mut_ptr();
        let checked = [
            unsafe { PlaneDescriptor::from_raw_parts(base, len).unwrap() },
            unsafe { PlaneDescriptor::from_raw_parts(base.add(1), len).unwrap() },
            unsafe { PlaneDescriptor::from_raw_parts(base, len).unwrap() },
        ];
        let checked_error = kernel.validate_raw_descriptors(&checked).unwrap_err();
        assert_eq!(checked_error.kind(), RusticolErrorKind::InvalidArgument);
        assert!(checked_error.message().contains("partially overlaps"));

        let error = kernel
            .bind_raw(vec![
                unsafe { PlaneDescriptor::from_raw_parts(base, len).unwrap() },
                unsafe { PlaneDescriptor::from_raw_parts(base.add(1), len).unwrap() },
                unsafe { PlaneDescriptor::from_raw_parts(base, len).unwrap() },
            ])
            .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::InvalidArgument);
        assert!(error.message().contains("partially overlaps"));
        assert!(error.message().contains("exact aliases"));
    }

    #[test]
    fn overlapping_output_planes_are_rejected() {
        let len = 8;
        let mut input = vec![1.0; len];
        let mut overlapping_outputs = vec![0.0; len + 1];
        let output_base = overlapping_outputs.as_mut_ptr();
        let descriptors = [
            PlaneDescriptor::from_slice(input.as_mut_slice()),
            unsafe { PlaneDescriptor::from_raw_parts(output_base, len).unwrap() },
            unsafe { PlaneDescriptor::from_raw_parts(output_base.add(1), len).unwrap() },
        ];
        let error = validate_raw_aliases(1, 2, &descriptors).unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::InvalidArgument);
        assert!(error.message().contains("output planes overlap"));
        assert!(error.message().contains("mutually disjoint"));
    }

    #[test]
    fn divergent_simd_block_replays_every_lane_with_the_scalar_kernel() {
        let bytes = divergent_application_bytes();
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "divergent-contract.symjit",
            SymjitPlaneLayout::real(3, 1),
        )
        .unwrap();
        let len = kernel.simd_lanes().max(2) * 2;
        let mut condition: Vec<f64> = (0..len)
            .map(|index| if index.is_multiple_of(2) { 1.0 } else { 0.0 })
            .collect();
        let mut if_true: Vec<f64> = (0..len).map(|index| 10.0 + index as f64).collect();
        let mut if_false: Vec<f64> = (0..len).map(|index| -20.0 - index as f64).collect();
        let mut output = vec![SENTINEL; len];

        let mut table = kernel
            .build_raw_table(vec![
                PlaneDescriptor::from_slice(condition.as_mut_slice()),
                PlaneDescriptor::from_slice(if_true.as_mut_slice()),
                PlaneDescriptor::from_slice(if_false.as_mut_slice()),
                PlaneDescriptor::from_slice(output.as_mut_slice()),
            ])
            .unwrap();
        assert!(table.simd_safe());

        if let Some(simd) = kernel.simd.filter(|_| kernel.simd_lanes() > 1) {
            // Establish that this input really exercises SymJIT's nonzero
            // lane-divergence status rather than only the scalar fallback.
            let status = unsafe {
                simd(
                    std::ptr::null(),
                    table.as_symjit_ptr(),
                    0,
                    kernel.applet.params.as_ptr(),
                )
            };
            assert_ne!(status, 0);
        }

        kernel.execute_table(&mut table, 0, len).unwrap();
        for index in 0..len {
            assert_eq!(
                output[index],
                if condition[index] != 0.0 {
                    if_true[index]
                } else {
                    if_false[index]
                }
            );
        }
    }

    #[test]
    fn divergent_input_output_alias_disables_simd_before_execution() {
        let bytes = divergent_application_bytes();
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "divergent-alias-contract.symjit",
            SymjitPlaneLayout::real(3, 1),
        )
        .unwrap();
        let len = kernel.simd_lanes().max(2) * 2;
        let mut condition_and_output: Vec<f64> = (0..len)
            .map(|index| if index.is_multiple_of(2) { 1.0 } else { 0.0 })
            .collect();
        let original_condition = condition_and_output.clone();
        let mut if_true: Vec<f64> = (0..len).map(|index| 30.0 + index as f64).collect();
        let mut if_false: Vec<f64> = (0..len).map(|index| -40.0 - index as f64).collect();
        let alias = condition_and_output.as_mut_ptr();

        let mut table = kernel
            .build_raw_table(vec![
                unsafe {
                    PlaneDescriptor::from_raw_parts(alias, condition_and_output.len()).unwrap()
                },
                PlaneDescriptor::from_slice(if_true.as_mut_slice()),
                PlaneDescriptor::from_slice(if_false.as_mut_slice()),
                unsafe {
                    PlaneDescriptor::from_raw_parts(alias, condition_and_output.len()).unwrap()
                },
            ])
            .unwrap();
        assert!(!table.simd_safe());

        kernel.execute_table(&mut table, 0, len).unwrap();
        for index in 0..len {
            assert_eq!(
                condition_and_output[index],
                if original_condition[index] != 0.0 {
                    if_true[index]
                } else {
                    if_false[index]
                }
            );
        }
    }

    #[test]
    fn disjoint_raw_binding_executes_without_descriptor_repacking() {
        let bytes = application_bytes(false, 2);
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "raw-contract.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap();
        let mut x = vec![2.0, 3.0, 4.0, 5.0];
        let mut y = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 4];
        {
            let mut table = kernel
                .build_raw_table(vec![
                    PlaneDescriptor::from_slice(x.as_mut_slice()),
                    PlaneDescriptor::from_slice(y.as_mut_slice()),
                    PlaneDescriptor::from_slice(out.as_mut_slice()),
                ])
                .unwrap();
            assert_eq!(table.point_count(), 4);
            let table_address = table.as_symjit_ptr().addr();
            kernel.execute_table(&mut table, 0, 4).unwrap();
            assert_eq!(table_address, table.as_symjit_ptr().addr());
        }
        assert_eq!(out, [5.0, 11.0, 19.0, 29.0]);
    }

    #[test]
    fn raw_table_rebind_preserves_descriptor_allocation_and_executes_new_planes() {
        let bytes = application_bytes(false, 2);
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "raw-rebind-contract.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap();
        let mut first_x = vec![2.0, 3.0, 4.0, 5.0];
        let mut first_y = vec![1.0, 2.0, 3.0, 4.0];
        let mut first_out = vec![0.0; 4];
        let mut second_x = vec![6.0, 7.0, 8.0, 9.0];
        let mut second_y = vec![5.0, 6.0, 7.0, 8.0];
        let mut second_out = vec![0.0; 4];

        {
            let mut table = kernel
                .build_raw_table(vec![
                    PlaneDescriptor::from_slice(first_x.as_mut_slice()),
                    PlaneDescriptor::from_slice(first_y.as_mut_slice()),
                    PlaneDescriptor::from_slice(first_out.as_mut_slice()),
                ])
                .unwrap();
            let table_address = table.as_symjit_ptr().addr();
            kernel.execute_table(&mut table, 0, 4).unwrap();

            let replacement = [
                // SAFETY: Each allocation is live, aligned, synchronized, and
                // retains its length until the table is dropped.
                unsafe {
                    PlaneDescriptor::from_raw_parts(second_x.as_mut_ptr(), second_x.len()).unwrap()
                },
                unsafe {
                    PlaneDescriptor::from_raw_parts(second_y.as_mut_ptr(), second_y.len()).unwrap()
                },
                unsafe {
                    PlaneDescriptor::from_raw_parts(second_out.as_mut_ptr(), second_out.len())
                        .unwrap()
                },
            ];
            // SAFETY: No kernel invocation overlaps the rebind and all
            // replacement allocations outlive the table.
            unsafe { table.rebind_raw(&replacement) }.unwrap();
            assert_eq!(table_address, table.as_symjit_ptr().addr());
            kernel.execute_table(&mut table, 0, 4).unwrap();
            assert_eq!(table_address, table.as_symjit_ptr().addr());
        }

        assert_eq!(first_out, [5.0, 11.0, 19.0, 29.0]);
        assert_eq!(second_out, [41.0, 55.0, 71.0, 89.0]);
    }

    #[test]
    fn raw_table_rebind_recomputes_input_output_alias_simd_safety() {
        let bytes = application_bytes(false, 2);
        let kernel = SymjitPlaneKernel::load_bytes(
            &bytes,
            "raw-rebind-alias-contract.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap();
        let mut x = vec![2.0, 3.0, 4.0, 5.0];
        let mut y = vec![1.0, 2.0, 3.0, 4.0];
        let mut first_output = vec![0.0; 4];
        let mut second_output = vec![0.0; 4];
        let x_pointer = x.as_mut_ptr();
        let y_pointer = y.as_mut_ptr();
        let first_output_pointer = first_output.as_mut_ptr();
        let second_output_pointer = second_output.as_mut_ptr();

        let mut table = kernel
            .build_raw_table(vec![
                unsafe { PlaneDescriptor::from_raw_parts(x_pointer, x.len()).unwrap() },
                unsafe { PlaneDescriptor::from_raw_parts(y_pointer, y.len()).unwrap() },
                unsafe {
                    PlaneDescriptor::from_raw_parts(first_output_pointer, first_output.len())
                        .unwrap()
                },
            ])
            .unwrap();
        assert!(table.simd_safe());

        let aliased = [
            unsafe { PlaneDescriptor::from_raw_parts(x_pointer, x.len()).unwrap() },
            unsafe { PlaneDescriptor::from_raw_parts(y_pointer, y.len()).unwrap() },
            unsafe { PlaneDescriptor::from_raw_parts(x_pointer, x.len()).unwrap() },
        ];
        unsafe { table.rebind_raw(&aliased) }.unwrap();
        assert!(!table.simd_safe());

        let disjoint = [
            unsafe { PlaneDescriptor::from_raw_parts(x_pointer, x.len()).unwrap() },
            unsafe { PlaneDescriptor::from_raw_parts(y_pointer, y.len()).unwrap() },
            unsafe {
                PlaneDescriptor::from_raw_parts(second_output_pointer, second_output.len()).unwrap()
            },
        ];
        unsafe { table.rebind_raw(&disjoint) }.unwrap();
        assert!(table.simd_safe());
    }

    unsafe fn status_kernel(
        _mem: *const f64,
        _states: *const SymjitPlaneDescriptor<f64>,
        index: usize,
        _params: *const f64,
    ) -> i32 {
        if index == 2 { 17 } else { 0 }
    }

    unsafe fn replay_scalar_kernel(
        _mem: *const f64,
        states: *const SymjitPlaneDescriptor<f64>,
        index: usize,
        _params: *const f64,
    ) -> i32 {
        unsafe {
            let input = *(*states).data.add(index);
            *(*states.add(1)).data.add(index) = input + 0.5;
        }
        0
    }

    unsafe fn divergent_vector_kernel(
        _mem: *const f64,
        states: *const SymjitPlaneDescriptor<f64>,
        point: usize,
        _params: *const f64,
    ) -> i32 {
        unsafe {
            for lane in 0..4 {
                *(*states.add(1)).data.add(point + lane) = SENTINEL;
            }
        }
        1
    }

    #[test]
    fn synthetic_divergence_status_replays_the_whole_simd_block() {
        let mut input: Vec<f64> = (0..11).map(|index| index as f64).collect();
        let mut output = vec![SENTINEL; input.len()];
        let descriptors = [
            unsafe { SymjitPlaneDescriptor::from_raw_parts(input.as_mut_ptr(), input.len()) },
            unsafe { SymjitPlaneDescriptor::from_raw_parts(output.as_mut_ptr(), output.len()) },
        ];

        let failure = invoke_range(
            replay_scalar_kernel,
            Some(divergent_vector_kernel),
            4,
            descriptors.as_ptr(),
            std::ptr::null(),
            1,
            input.len() - 1,
        )
        .unwrap();
        assert_eq!(failure, None);
        assert_eq!(output[0], SENTINEL);
        assert_eq!(output[input.len() - 1], SENTINEL);
        for index in 1..input.len() - 1 {
            assert_eq!(output[index], input[index] + 0.5);
        }
    }

    unsafe fn replay_status_kernel(
        _mem: *const f64,
        _states: *const SymjitPlaneDescriptor<f64>,
        index: usize,
        _params: *const f64,
    ) -> i32 {
        if index == 5 { 29 } else { 0 }
    }

    unsafe fn divergence_status_only_kernel(
        _mem: *const f64,
        _states: *const SymjitPlaneDescriptor<f64>,
        _point: usize,
        _params: *const f64,
    ) -> i32 {
        1
    }

    #[test]
    fn scalar_status_during_divergent_replay_is_reported_at_the_point() {
        let failure = invoke_range(
            replay_status_kernel,
            Some(divergence_status_only_kernel),
            4,
            std::ptr::null(),
            std::ptr::null(),
            4,
            8,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            failure,
            StatusFailure {
                index: 5,
                status: 29,
            }
        );
    }

    #[test]
    fn nonzero_kernel_status_is_contained() {
        let failure = invoke_range(
            status_kernel,
            None,
            1,
            std::ptr::null(),
            std::ptr::null(),
            0,
            4,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            failure,
            StatusFailure {
                index: 2,
                status: 17,
            }
        );
    }

    #[test]
    fn symjit_panics_are_contained_with_dependency_context() {
        let error = guard_symjit_panic(
            || panic!("probe panic"),
            || "exercise the panic-containment probe".to_string(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Compatibility);
        assert!(
            error
                .message()
                .contains("SymJIT panicked while trying to exercise the panic-containment probe")
        );
        assert!(error.message().contains("probe panic"));
    }

    #[test]
    fn storage_header_dimensions_and_trailing_bytes_fail_closed() {
        let bytes = application_bytes(false, 2);

        // v4 stores Config.num_args; a v3 payload needs regeneration, not
        // metadata relabeling or an implicit historical-format migration.
        for version in [3_usize, 5] {
            let mut unsupported_version = bytes.clone();
            unsupported_version[8..16].copy_from_slice(&version.to_le_bytes());
            let error = SymjitPlaneKernel::load_bytes(
                &unsupported_version,
                "unsupported-storage.symjit",
                SymjitPlaneLayout::real(2, 1),
            )
            .unwrap_err();
            assert_eq!(error.kind(), RusticolErrorKind::Compatibility);
            assert!(error.message().contains("expected 4"));
        }

        let mut bad_magic = bytes.clone();
        bad_magic[0] ^= 1;
        let error = SymjitPlaneKernel::load_bytes(
            &bad_magic,
            "bad-magic.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Compatibility);

        let error = SymjitPlaneKernel::load_bytes(
            &bytes,
            "bad-shape.symjit",
            SymjitPlaneLayout::real(3, 1),
        )
        .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Integrity);

        let mut trailing = bytes;
        trailing.push(0);
        let error = SymjitPlaneKernel::load_bytes(
            &trailing,
            "trailing.symjit",
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Integrity);
    }

    #[test]
    fn migration_contract_rejects_stored_simd512_configuration() {
        let mut config = explicit_plane_config(false, 2, true).unwrap();
        config.enable_simd512(true);
        let bytes = application_bytes_with_config(config);
        let error =
            SymjitPlaneKernel::load_bytes(&bytes, "simd512.symjit", SymjitPlaneLayout::real(2, 1))
                .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Compatibility);
        assert!(error.message().contains("SIMD512"));
    }

    #[test]
    fn migration_contract_rejects_noncanonical_stored_configuration() {
        let mut config = explicit_plane_config(false, 2, true).unwrap();
        config.set_simd(false);
        let bytes = application_bytes_with_config(config);
        let error =
            SymjitPlaneKernel::load_bytes(&bytes, "scalar.symjit", SymjitPlaneLayout::real(2, 1))
                .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Compatibility);
        assert!(error.message().contains("noncanonical compiler options"));
        assert!(error.message().contains("regenerate"));
    }

    #[test]
    fn migration_contract_rejects_non_native_compiler_target() {
        let mut config = explicit_plane_config(false, 2, true).unwrap();
        config.ty = CompilerType::ByteCode;
        let error = canonical_plane_optimization_level(
            &config,
            Path::new("bytecode.symjit"),
            SymjitPlaneLayout::real(2, 1),
        )
        .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Compatibility);
        assert!(error.message().contains("non-native compiler target"));
        assert!(error.message().contains("regenerate"));
    }

    #[test]
    fn migration_contract_accepts_only_scalar_f64x2_or_f64x4_runtime_lanes() {
        let path = Path::new("lane-contract.symjit");
        for (has_simd, lanes) in [(false, 1), (true, 2), (true, 4)] {
            validate_simd_lane_width(has_simd, lanes, path).unwrap();
        }
        for (has_simd, lanes) in [(true, 1), (false, 2), (true, 8)] {
            let error = validate_simd_lane_width(has_simd, lanes, path).unwrap_err();
            assert_eq!(error.kind(), RusticolErrorKind::Compatibility);
            assert!(error.message().contains("unsupported SIMD lane width"));
        }
    }

    #[test]
    fn compile_helper_rejects_invalid_cold_path_inputs() {
        let error =
            compile_symbolica_program_to_plane_application_bytes("", 2, 1, 2, false).unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::InvalidArgument);
        let error = compile_symbolica_program_to_plane_application_bytes(MODEL, 2, 1, 4, false)
            .unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::InvalidArgument);
    }

    /// One output per scalar primitive, and the `equal`-guarded branch shape
    /// of a model helper function (Symbolica's `if`), all inlined calls.
    const PRIMITIVE_MODEL: &str = "
([('fun', ('temp', 0), 'pyamplicol_scalar_v1::equal', [], [('param', 0), ('param', 1)], False),
  ('if_else', ('temp', 0), 1),
  ('assign', ('temp', 1), ('param', 0)),
  ('goto', 2),
  ('label', 1),
  ('mul', ('temp', 2), [('param', 1), ('const', 0)], 0),
  ('label', 2),
  ('join', ('out', 0), ('temp', 0), ('temp', 1), ('temp', 2)),
  ('assign', ('out', 1), ('temp', 0)),
  ('fun', ('out', 2), 'pyamplicol_scalar_v1::truth', [], [('param', 0)], False),
  ('fun', ('out', 3), 'pyamplicol_scalar_v1::less', [], [('param', 0), ('param', 1)], False),
  ('fun', ('out', 4), 'pyamplicol_scalar_v1::less_equal', [], [('param', 0), ('param', 1)], False),
  ('fun', ('out', 5), 'pyamplicol_scalar_v1::real_part', [], [('param', 0)], False),
  ('fun', ('out', 6), 'pyamplicol_scalar_v1::imaginary_part', [], [('param', 0)], False)],
 3,
 [3])
";

    fn same_value(left: f64, right: f64) -> bool {
        left.to_bits() == right.to_bits() || (left.is_nan() && right.is_nan())
    }

    #[test]
    fn scalar_primitives_compile_relocate_and_match_the_native_semantics() {
        let bytes =
            compile_symbolica_program_to_plane_application_bytes(PRIMITIVE_MODEL, 2, 7, 2, false)
                .unwrap();
        // The stored IR names the primitives; loading binds them again.
        let name = b"symbolica_pyamplicol_scalar_v1::equal";
        assert!(bytes.windows(name.len()).any(|window| window == name));
        let pairs: [((f64, f64), (f64, f64)); 11] = [
            ((1.5, 0.0), (1.5, 0.0)),
            ((1.5, 0.0), (2.5, 0.0)),
            ((2.5, 0.0), (1.5, 0.0)),
            ((0.0, 0.0), (-0.0, 0.0)),
            ((1.0, 2.0), (1.0, 2.0)),
            ((1.0, 2.0), (1.0, -2.0)),
            ((0.0, 1.0), (3.0, 0.0)),
            ((f64::NAN, 0.0), (1.0, 0.0)),
            ((f64::INFINITY, 0.0), (1.0, 0.0)),
            ((-4.0, 0.0), (-4.0, 0.0)),
            ((0.0, 0.0), (0.0, 0.0)),
        ];
        // Long enough for SIMD blocks with divergent lanes, scalar replay
        // and a scalar tail.
        let points: Vec<_> = pairs.iter().cycle().take(3 * pairs.len()).collect();
        let inputs: Vec<f64> = points
            .iter()
            .flat_map(|(x, y)| [x.0, x.1, y.0, y.1])
            .collect();
        let outputs =
            evaluate_plane_application_bytes(&bytes, 2, 7, &inputs, points.len()).unwrap();
        for (point, (x, y)) in points.iter().enumerate() {
            let primitive = |args: &[(f64, f64)], mode| {
                crate::scalar_primitives::scalar_primitive_f64(args, mode)
            };
            let equal = primitive(&[*x, *y], 1);
            let branch = if equal.0 != 0.0 || equal.1 != 0.0 {
                *x
            } else {
                (3.0 * y.0, 3.0 * y.1)
            };
            let expected = [
                branch,
                equal,
                primitive(&[*x], 0),
                primitive(&[*x, *y], 2),
                primitive(&[*x, *y], 3),
                primitive(&[*x], 4),
                primitive(&[*x], 5),
            ];
            for (output, value) in expected.iter().enumerate() {
                let at = 14 * point + 2 * output;
                assert!(
                    same_value(outputs[at], value.0) && same_value(outputs[at + 1], value.1),
                    "point {point} {x:?} {y:?} output {output}: {:?} != {value:?}",
                    (outputs[at], outputs[at + 1])
                );
            }
        }
    }

    #[test]
    fn plane_compiler_refuses_unregistered_functions() {
        let program = PRIMITIVE_MODEL.replace("scalar_v1::truth", "scalar_v1::unknown");
        let error = compile_symbolica_program_to_plane_application_bytes(&program, 2, 7, 2, false)
            .unwrap_err();
        assert!(error.message().contains("unknown"), "{}", error.message());
    }
}
