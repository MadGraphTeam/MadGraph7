// SPDX-License-Identifier: 0BSD
// Dependency-free safe Rust 2021 bindings for Rusticol C ABI v1.

use std::error;
use std::ffi::{c_char, c_int, CStr, CString};
use std::fmt;
use std::marker::PhantomData;
use std::path::Path;
use std::ptr::{self, NonNull};
use std::rc::Rc;

/// C ABI version implemented by this source wrapper.
pub const ABI_VERSION: u32 = 1;

/// The category of a Rusticol SDK error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorKind {
    InvalidArgument,
    BufferTooSmall,
    Runtime,
    Panic,
    AbiMismatch,
    InvalidInput,
    InvalidUtf8,
    InvalidResponse,
    SizeOverflow,
    UnknownStatus(c_int),
}

/// A typed error returned by the safe Rusticol wrapper.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    fn from_status(status: c_int) -> Self {
        let kind = match status {
            ffi::STATUS_INVALID_ARGUMENT => ErrorKind::InvalidArgument,
            ffi::STATUS_BUFFER_TOO_SMALL => ErrorKind::BufferTooSmall,
            ffi::STATUS_RUNTIME_ERROR => ErrorKind::Runtime,
            ffi::STATUS_PANIC => ErrorKind::Panic,
            other => ErrorKind::UnknownStatus(other),
        };
        Self::new(kind, last_error_message(status))
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl error::Error for Error {}

/// Result type used by the safe Rusticol wrapper.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExternalParticle {
    pub index: usize,
    pub pdg: i32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HelicityConfiguration {
    pub id: String,
    pub helicities: Vec<i32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ColorComponent {
    pub id: String,
    pub kind: String,
    pub word: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelParameter {
    pub name: String,
}

/// Typed physics metadata used to interpret resolved output axes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PhysicsMetadata {
    pub process: String,
    pub process_key: String,
    pub color_accuracy: String,
    pub external_particles: Vec<ExternalParticle>,
    pub helicities: Vec<HelicityConfiguration>,
    pub colors: Vec<ColorComponent>,
}

/// Physical component selectors for resolved evaluation.
///
/// An empty axis selects every component on that axis. Selected output remains
/// in the artifact's canonical metadata order.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Selectors {
    helicity_ids: Vec<String>,
    color_ids: Vec<String>,
}

impl Selectors {
    pub fn all() -> Self {
        Self::default()
    }

    pub fn with_helicities<I, S>(mut self, ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.helicity_ids = ids.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_colors<I, S>(mut self, ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.color_ids = ids.into_iter().map(Into::into).collect();
        self
    }

    pub fn helicity_ids(&self) -> &[String] {
        &self.helicity_ids
    }

    pub fn color_ids(&self) -> &[String] {
        &self.color_ids
    }
}

/// One atomic model-parameter update in a batch.
#[derive(Clone, Debug, PartialEq)]
pub struct ParameterUpdate {
    pub name: String,
    pub real: f64,
    pub imaginary: f64,
}

impl ParameterUpdate {
    pub fn new(name: impl Into<String>, real: f64, imaginary: f64) -> Self {
        Self {
            name: name.into(),
            real,
            imaginary,
        }
    }
}

/// Owned resolved f64 matrix elements with typed physical axes.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedEvaluation {
    values: Vec<f64>,
    point_count: usize,
    helicities: Vec<HelicityConfiguration>,
    colors: Vec<ColorComponent>,
}

impl ResolvedEvaluation {
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn point_count(&self) -> usize {
        self.point_count
    }

    pub fn helicities(&self) -> &[HelicityConfiguration] {
        &self.helicities
    }

    pub fn colors(&self) -> &[ColorComponent] {
        &self.colors
    }

    pub fn shape(&self) -> (usize, usize, usize) {
        (self.point_count, self.helicities.len(), self.colors.len())
    }

    pub fn get(&self, point: usize, helicity: usize, color: usize) -> Option<f64> {
        if point >= self.point_count
            || helicity >= self.helicities.len()
            || color >= self.colors.len()
        {
            return None;
        }
        let offset = (point * self.helicities.len() + helicity) * self.colors.len() + color;
        self.values.get(offset).copied()
    }

    /// Explicitly sum every resolved component for each phase-space point.
    pub fn totals(&self) -> Vec<f64> {
        let stride = self.helicities.len() * self.colors.len();
        self.values
            .chunks_exact(stride)
            .map(|point| point.iter().sum())
            .collect()
    }
}

/// Owning Rusticol runtime handle.
///
/// Methods requiring mutable runtime state take `&mut self`. The marker keeps
/// a handle on its creating thread because C ABI v1 does not promise that an
/// individual handle may be transferred between threads.
pub struct Runtime {
    handle: NonNull<ffi::RuntimeHandle>,
    _thread_bound: PhantomData<Rc<()>>,
}

impl Runtime {
    /// Load an artifact, optionally selecting a process and applying a JSON
    /// model-parameter card before the runtime becomes visible to the caller.
    pub fn load(
        artifact: impl AsRef<Path>,
        process_key: Option<&str>,
        model_parameters_path: Option<&Path>,
    ) -> Result<Self> {
        ensure_abi_version()?;
        let artifact = path_cstring(artifact.as_ref(), "artifact path")?;
        let process_key = optional_cstring(process_key, "process selector")?;
        let model_parameters = model_parameters_path
            .map(|path| path_cstring(path, "model-parameter path"))
            .transpose()?;
        let mut handle = ptr::null_mut();
        // SAFETY: CStrings remain live for the call and output points to writable storage.
        check(unsafe {
            ffi::runtime_load(
                artifact.as_ptr(),
                optional_pointer(&process_key),
                optional_pointer(&model_parameters),
                &mut handle,
            )
        })?;
        let handle = NonNull::new(handle).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidResponse,
                "Rusticol returned a null runtime handle after a successful load",
            )
        })?;
        Ok(Self {
            handle,
            _thread_bound: PhantomData,
        })
    }

    pub fn process(&self) -> Result<String> {
        self.get_string(ffi::runtime_process)
    }

    pub fn process_key(&self) -> Result<String> {
        self.get_string(ffi::runtime_process_key)
    }

    pub fn color_accuracy(&self) -> Result<String> {
        self.get_string(ffi::runtime_color_accuracy)
    }

    pub fn metadata_json(&self) -> Result<String> {
        self.get_string(ffi::runtime_metadata_json)
    }

    pub fn execution_mode(&self) -> Result<String> {
        self.get_string(ffi::runtime_execution_mode)
    }

    pub fn physics_json(&self) -> Result<String> {
        self.get_string(ffi::runtime_physics_json)
    }

    pub fn external_particles(&self) -> Result<Vec<ExternalParticle>> {
        let mut count = 0;
        // SAFETY: The RAII handle is live and count is writable.
        check(unsafe { ffi::runtime_external_count(self.handle.as_ptr(), &mut count) })?;
        let mut particles = Vec::with_capacity(count);
        for index in 0..count {
            let mut pdg = 0;
            // SAFETY: The RAII handle is live and pdg is writable.
            check(unsafe { ffi::runtime_external_pdg(self.handle.as_ptr(), index, &mut pdg) })?;
            particles.push(ExternalParticle { index, pdg });
        }
        Ok(particles)
    }

    pub fn helicities(&self) -> Result<Vec<HelicityConfiguration>> {
        let mut count = 0;
        // SAFETY: The RAII handle is live and count is writable.
        check(unsafe { ffi::runtime_helicity_count(self.handle.as_ptr(), &mut count) })?;
        let mut result = Vec::with_capacity(count);
        for index in 0..count {
            let id = self.get_indexed_string(ffi::runtime_helicity_id, index)?;
            let helicities = read_i32_vector(|output, capacity, required| {
                // SAFETY: The helper owns a correctly sized output buffer for the live handle.
                unsafe {
                    ffi::runtime_helicity_vector(
                        self.handle.as_ptr(),
                        index,
                        output,
                        capacity,
                        required,
                    )
                }
            })?;
            result.push(HelicityConfiguration { id, helicities });
        }
        Ok(result)
    }

    pub fn colors(&self) -> Result<Vec<ColorComponent>> {
        let mut count = 0;
        // SAFETY: The RAII handle is live and count is writable.
        check(unsafe { ffi::runtime_color_count(self.handle.as_ptr(), &mut count) })?;
        let mut result = Vec::with_capacity(count);
        for index in 0..count {
            let id = self.get_indexed_string(ffi::runtime_color_id, index)?;
            let kind = self.get_indexed_string(ffi::runtime_color_kind, index)?;
            let word = read_usize_vector(|output, capacity, required| {
                // SAFETY: The helper owns a correctly sized output buffer for the live handle.
                unsafe {
                    ffi::runtime_color_word(self.handle.as_ptr(), index, output, capacity, required)
                }
            })?;
            result.push(ColorComponent { id, kind, word });
        }
        Ok(result)
    }

    pub fn model_parameters(&self) -> Result<Vec<ModelParameter>> {
        let mut count = 0;
        // SAFETY: The RAII handle is live and count is writable.
        check(unsafe { ffi::runtime_model_parameter_count(self.handle.as_ptr(), &mut count) })?;
        (0..count)
            .map(|index| {
                self.get_indexed_string(ffi::runtime_model_parameter_name, index)
                    .map(|name| ModelParameter { name })
            })
            .collect()
    }

    pub fn physics(&self) -> Result<PhysicsMetadata> {
        Ok(PhysicsMetadata {
            process: self.process()?,
            process_key: self.process_key()?,
            color_accuracy: self.color_accuracy()?,
            external_particles: self.external_particles()?,
            helicities: self.helicities()?,
            colors: self.colors()?,
        })
    }

    /// Evaluate compatibility totals for one or more flattened momentum points.
    pub fn evaluate_f64(&mut self, momenta: &[f64], point_count: usize) -> Result<Vec<f64>> {
        self.validate_momenta(momenta, point_count)?;
        let mut values = vec![0.0; point_count];
        // SAFETY: Input and output slices are disjoint, sized, and live for the call.
        check(unsafe {
            ffi::runtime_evaluate_f64(
                self.handle.as_ptr(),
                momenta.as_ptr(),
                momenta.len(),
                point_count,
                values.as_mut_ptr(),
                values.len(),
            )
        })?;
        Ok(values)
    }

    /// Evaluate one selected total per point.
    ///
    /// Global string selectors retain subset/sum semantics. Per-point selector
    /// arrays contain zero-based physical-axis indices and must contain exactly
    /// one entry per momentum point. The global and per-point forms are mutually
    /// exclusive on the same axis.
    pub fn evaluate_selected_f64(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        selectors: &Selectors,
        helicity_by_point: Option<&[u32]>,
        color_flow_by_point: Option<&[u32]>,
    ) -> Result<Vec<f64>> {
        self.validate_momenta(momenta, point_count)?;
        if !selectors.helicity_ids.is_empty() && helicity_by_point.is_some() {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
                "global helicity selectors and helicity_by_point are mutually exclusive",
            ));
        }
        if !selectors.color_ids.is_empty() && color_flow_by_point.is_some() {
            return Err(Error::new(
                ErrorKind::InvalidArgument,
                "global color selectors and color_flow_by_point are mutually exclusive",
            ));
        }
        for (name, values) in [
            ("helicity_by_point", helicity_by_point),
            ("color_flow_by_point", color_flow_by_point),
        ] {
            if let Some(values) = values {
                if values.len() != point_count {
                    return Err(Error::new(
                        ErrorKind::InvalidArgument,
                        format!(
                            "{name} contains {} entries, expected {point_count}",
                            values.len()
                        ),
                    ));
                }
            }
        }

        let helicity_strings = cstring_list(&selectors.helicity_ids, "helicity selector")?;
        let color_strings = cstring_list(&selectors.color_ids, "color selector")?;
        let helicity_pointers = cstring_pointers(&helicity_strings);
        let color_pointers = cstring_pointers(&color_strings);
        let (helicity_pointer, helicity_count) = selector_parts(&helicity_pointers);
        let (color_pointer, color_count) = selector_parts(&color_pointers);
        let (helicity_by_point_pointer, helicity_by_point_count) =
            u32_selector_parts(helicity_by_point);
        let (color_flow_by_point_pointer, color_flow_by_point_count) =
            u32_selector_parts(color_flow_by_point);
        let mut values = vec![0.0; point_count];
        // SAFETY: All selector and momentum buffers remain live and the output
        // has exactly one writable entry per point.
        check(unsafe {
            ffi::runtime_evaluate_selected_f64(
                self.handle.as_ptr(),
                momenta.as_ptr(),
                momenta.len(),
                point_count,
                helicity_pointer,
                helicity_count,
                color_pointer,
                color_count,
                helicity_by_point_pointer,
                helicity_by_point_count,
                color_flow_by_point_pointer,
                color_flow_by_point_count,
                values.as_mut_ptr(),
                values.len(),
            )
        })?;
        Ok(values)
    }

    /// Evaluate resolved f64 components for one or more momentum points.
    pub fn evaluate_resolved_f64(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        selectors: &Selectors,
    ) -> Result<ResolvedEvaluation> {
        self.validate_momenta(momenta, point_count)?;
        let helicity_strings = cstring_list(&selectors.helicity_ids, "helicity selector")?;
        let color_strings = cstring_list(&selectors.color_ids, "color selector")?;
        let helicity_pointers = cstring_pointers(&helicity_strings);
        let color_pointers = cstring_pointers(&color_strings);
        let (helicity_pointer, helicity_count) = selector_parts(&helicity_pointers);
        let (color_pointer, color_count) = selector_parts(&color_pointers);

        let mut resolved_helicity_count = 0;
        let mut resolved_color_count = 0;
        // SAFETY: Selector CStrings and output dimensions remain live for the call.
        check(unsafe {
            ffi::runtime_resolved_shape(
                self.handle.as_ptr(),
                helicity_pointer,
                helicity_count,
                color_pointer,
                color_count,
                &mut resolved_helicity_count,
                &mut resolved_color_count,
            )
        })?;
        if resolved_helicity_count == 0 || resolved_color_count == 0 {
            return Err(Error::new(
                ErrorKind::InvalidResponse,
                "Rusticol returned an empty resolved axis",
            ));
        }
        let value_count = point_count
            .checked_mul(resolved_helicity_count)
            .and_then(|count| count.checked_mul(resolved_color_count))
            .ok_or_else(|| Error::new(ErrorKind::SizeOverflow, "resolved output shape overflow"))?;
        let mut values = vec![0.0; value_count];
        let mut returned_helicity_count = resolved_helicity_count;
        let mut returned_color_count = resolved_color_count;
        // SAFETY: Every pointer refers to a live, disjoint, correctly sized allocation.
        check(unsafe {
            ffi::runtime_evaluate_resolved_f64(
                self.handle.as_ptr(),
                momenta.as_ptr(),
                momenta.len(),
                point_count,
                helicity_pointer,
                helicity_count,
                color_pointer,
                color_count,
                values.as_mut_ptr(),
                values.len(),
                &mut returned_helicity_count,
                &mut returned_color_count,
            )
        })?;
        if returned_helicity_count != resolved_helicity_count
            || returned_color_count != resolved_color_count
        {
            return Err(Error::new(
                ErrorKind::InvalidResponse,
                "Rusticol changed the resolved shape during evaluation",
            ));
        }

        let metadata = self.physics()?;
        let helicities = select_helicities(metadata.helicities, &selectors.helicity_ids);
        let colors = select_colors(metadata.colors, &selectors.color_ids);
        if helicities.len() != resolved_helicity_count || colors.len() != resolved_color_count {
            return Err(Error::new(
                ErrorKind::InvalidResponse,
                "Rusticol resolved output does not match its physics metadata",
            ));
        }
        Ok(ResolvedEvaluation {
            values,
            point_count,
            helicities,
            colors,
        })
    }

    pub fn set_model_parameter(&mut self, name: &str, real: f64, imaginary: f64) -> Result<()> {
        let name = checked_cstring(name, "model-parameter name")?;
        // SAFETY: The name CString and owning handle remain live for the call.
        check(unsafe {
            ffi::runtime_set_model_parameter(self.handle.as_ptr(), name.as_ptr(), real, imaginary)
        })
    }

    /// Apply a batch as one atomic update. A failing entry leaves all runtime
    /// model parameters unchanged.
    pub fn set_model_parameters(&mut self, updates: &[ParameterUpdate]) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }
        let names = updates
            .iter()
            .map(|update| checked_cstring(&update.name, "model-parameter name"))
            .collect::<Result<Vec<_>>>()?;
        let name_pointers = cstring_pointers(&names);
        let real = updates.iter().map(|update| update.real).collect::<Vec<_>>();
        let imaginary = updates
            .iter()
            .map(|update| update.imaginary)
            .collect::<Vec<_>>();
        // SAFETY: All arrays have identical nonzero lengths and remain live for the call.
        check(unsafe {
            ffi::runtime_set_model_parameters(
                self.handle.as_ptr(),
                name_pointers.as_ptr(),
                real.as_ptr(),
                imaginary.as_ptr(),
                updates.len(),
            )
        })
    }

    pub fn set_model_parameters_json(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path_cstring(path.as_ref(), "model-parameter path")?;
        // SAFETY: The path CString and owning handle remain live for the call.
        check(unsafe {
            ffi::runtime_set_model_parameters_json(self.handle.as_ptr(), path.as_ptr())
        })
    }

    pub fn set_warnings_muted(&mut self, muted: bool) -> Result<()> {
        // SAFETY: The owning handle remains live and exclusively borrowed.
        check(unsafe { ffi::runtime_mute_warnings(self.handle.as_ptr(), c_int::from(muted)) })
    }

    pub fn mute_warnings(&mut self) -> Result<()> {
        self.set_warnings_muted(true)
    }

    pub fn unmute_warnings(&mut self) -> Result<()> {
        self.set_warnings_muted(false)
    }

    /// Return pending warnings as a JSON array and consume them after copying.
    pub fn take_warnings_json(&mut self) -> Result<String> {
        read_string(|buffer, capacity, required| {
            // SAFETY: The helper owns the output buffer and the handle is exclusively borrowed.
            unsafe {
                ffi::runtime_take_warnings_json(self.handle.as_ptr(), buffer, capacity, required)
            }
        })
    }

    fn validate_momenta(&self, momenta: &[f64], point_count: usize) -> Result<()> {
        if point_count == 0 {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "point_count must be positive",
            ));
        }
        let external_count = self.external_particles()?.len();
        let expected = point_count
            .checked_mul(external_count)
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| Error::new(ErrorKind::SizeOverflow, "momentum shape overflow"))?;
        if momenta.len() != expected {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "momenta contain {} values, expected {expected} for {point_count} point(s)",
                    momenta.len()
                ),
            ));
        }
        if momenta.iter().any(|value| !value.is_finite()) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                "momenta must contain only finite f64 values",
            ));
        }
        Ok(())
    }

    fn get_string(&self, getter: ffi::StringGetter) -> Result<String> {
        read_string(|buffer, capacity, required| {
            // SAFETY: The helper owns the output buffer and the RAII handle is live.
            unsafe { getter(self.handle.as_ptr(), buffer, capacity, required) }
        })
    }

    fn get_indexed_string(&self, getter: ffi::IndexedStringGetter, index: usize) -> Result<String> {
        read_string(|buffer, capacity, required| {
            // SAFETY: The helper owns the output buffer and the RAII handle is live.
            unsafe { getter(self.handle.as_ptr(), index, buffer, capacity, required) }
        })
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // SAFETY: Runtime uniquely owns this live handle and frees it exactly once.
        let _ = unsafe { ffi::runtime_free(self.handle.as_ptr()) };
    }
}

pub fn abi_version() -> u32 {
    // SAFETY: This function has no pointer arguments or mutable state contract.
    unsafe { ffi::abi_version() }
}

pub fn ensure_abi_version() -> Result<()> {
    let observed = abi_version();
    if observed == ABI_VERSION {
        Ok(())
    } else {
        Err(Error::new(
            ErrorKind::AbiMismatch,
            format!(
                "Rusticol ABI version {observed} does not match required version {ABI_VERSION}"
            ),
        ))
    }
}

pub fn supported_runtime_capabilities_json() -> Result<String> {
    read_string(|buffer, capacity, required| {
        // SAFETY: The helper owns the correctly sized output buffer.
        unsafe { ffi::supported_runtime_capabilities_json(buffer, capacity, required) }
    })
}

fn check(status: c_int) -> Result<()> {
    if status == ffi::STATUS_OK {
        Ok(())
    } else {
        Err(Error::from_status(status))
    }
}

fn last_error_message(original_status: c_int) -> String {
    let mut required = 0;
    // SAFETY: A null zero-capacity buffer is the documented size query.
    let query_status = unsafe { ffi::last_error_message(ptr::null_mut(), 0, &mut required) };
    if query_status != ffi::STATUS_OK || required == 0 {
        return format!("Rusticol C ABI call failed with status {original_status}");
    }
    let mut buffer = vec![0_u8; required];
    // SAFETY: The vector provides required writable bytes and remains live for the call.
    let copy_status =
        unsafe { ffi::last_error_message(buffer.as_mut_ptr().cast(), buffer.len(), &mut required) };
    if copy_status != ffi::STATUS_OK {
        return format!("Rusticol C ABI call failed with status {original_status}");
    }
    CStr::from_bytes_until_nul(&buffer)
        .ok()
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("Rusticol C ABI call failed with status {original_status}"))
}

fn read_string(mut getter: impl FnMut(*mut c_char, usize, *mut usize) -> c_int) -> Result<String> {
    let mut required = 0;
    check(getter(ptr::null_mut(), 0, &mut required))?;
    if required == 0 {
        return Err(Error::new(
            ErrorKind::InvalidResponse,
            "Rusticol returned a zero-length string buffer",
        ));
    }
    let mut buffer = vec![0_u8; required];
    let mut copied_required = required;
    check(getter(
        buffer.as_mut_ptr().cast(),
        buffer.len(),
        &mut copied_required,
    ))?;
    if copied_required == 0 || copied_required > buffer.len() {
        return Err(Error::new(
            ErrorKind::InvalidResponse,
            "Rusticol returned an inconsistent string size",
        ));
    }
    let value = CStr::from_bytes_until_nul(&buffer).map_err(|_| {
        Error::new(
            ErrorKind::InvalidResponse,
            "Rusticol returned a string without a trailing NUL",
        )
    })?;
    value.to_str().map(str::to_owned).map_err(|_| {
        Error::new(
            ErrorKind::InvalidUtf8,
            "Rusticol returned a string that is not valid UTF-8",
        )
    })
}

fn read_i32_vector(
    mut getter: impl FnMut(*mut i32, usize, *mut usize) -> c_int,
) -> Result<Vec<i32>> {
    let mut required = 0;
    check(getter(ptr::null_mut(), 0, &mut required))?;
    if required == 0 {
        return Ok(Vec::new());
    }
    let mut values = vec![0_i32; required];
    let mut copied_required = required;
    check(getter(
        values.as_mut_ptr(),
        values.len(),
        &mut copied_required,
    ))?;
    if copied_required != values.len() {
        return Err(Error::new(
            ErrorKind::InvalidResponse,
            "Rusticol returned an inconsistent i32 vector size",
        ));
    }
    Ok(values)
}

fn read_usize_vector(
    mut getter: impl FnMut(*mut usize, usize, *mut usize) -> c_int,
) -> Result<Vec<usize>> {
    let mut required = 0;
    check(getter(ptr::null_mut(), 0, &mut required))?;
    if required == 0 {
        return Ok(Vec::new());
    }
    let mut values = vec![0_usize; required];
    let mut copied_required = required;
    check(getter(
        values.as_mut_ptr(),
        values.len(),
        &mut copied_required,
    ))?;
    if copied_required != values.len() {
        return Err(Error::new(
            ErrorKind::InvalidResponse,
            "Rusticol returned an inconsistent usize vector size",
        ));
    }
    Ok(values)
}

fn checked_cstring(value: &str, description: &str) -> Result<CString> {
    CString::new(value).map_err(|_| {
        Error::new(
            ErrorKind::InvalidInput,
            format!("{description} contains a NUL byte"),
        )
    })
}

fn path_cstring(path: &Path, description: &str) -> Result<CString> {
    let value = path.to_str().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidUtf8,
            format!("{description} is not valid UTF-8: {}", path.display()),
        )
    })?;
    checked_cstring(value, description)
}

fn optional_cstring(value: Option<&str>, description: &str) -> Result<Option<CString>> {
    value
        .map(|item| checked_cstring(item, description))
        .transpose()
}

fn optional_pointer(value: &Option<CString>) -> *const c_char {
    value.as_ref().map_or(ptr::null(), |item| item.as_ptr())
}

fn cstring_list(values: &[String], description: &str) -> Result<Vec<CString>> {
    values
        .iter()
        .map(|value| checked_cstring(value, description))
        .collect()
}

fn cstring_pointers(values: &[CString]) -> Vec<*const c_char> {
    values.iter().map(|value| value.as_ptr()).collect()
}

fn selector_parts(values: &[*const c_char]) -> (*const *const c_char, usize) {
    if values.is_empty() {
        (ptr::null(), 0)
    } else {
        (values.as_ptr(), values.len())
    }
}

fn u32_selector_parts(values: Option<&[u32]>) -> (*const u32, usize) {
    match values {
        Some(values) if !values.is_empty() => (values.as_ptr(), values.len()),
        _ => (ptr::null(), 0),
    }
}

fn select_helicities(
    available: Vec<HelicityConfiguration>,
    selected: &[String],
) -> Vec<HelicityConfiguration> {
    if selected.is_empty() {
        available
    } else {
        available
            .into_iter()
            .filter(|item| selected.iter().any(|id| id == &item.id))
            .collect()
    }
}

fn select_colors(available: Vec<ColorComponent>, selected: &[String]) -> Vec<ColorComponent> {
    if selected.is_empty() {
        available
    } else {
        available
            .into_iter()
            .filter(|item| selected.iter().any(|id| id == &item.id))
            .collect()
    }
}

mod ffi {
    use super::{c_char, c_int};

    pub(super) const STATUS_OK: c_int = 0;
    pub(super) const STATUS_INVALID_ARGUMENT: c_int = 1;
    pub(super) const STATUS_BUFFER_TOO_SMALL: c_int = 2;
    pub(super) const STATUS_RUNTIME_ERROR: c_int = 3;
    pub(super) const STATUS_PANIC: c_int = 4;

    #[repr(C)]
    pub(super) struct RuntimeHandle {
        _private: [u8; 0],
    }

    pub(super) type StringGetter =
        unsafe extern "C" fn(*const RuntimeHandle, *mut c_char, usize, *mut usize) -> c_int;
    pub(super) type IndexedStringGetter =
        unsafe extern "C" fn(*const RuntimeHandle, usize, *mut c_char, usize, *mut usize) -> c_int;

    unsafe extern "C" {
        pub(super) fn rusticol_abi_version() -> u32;
        pub(super) fn rusticol_supported_runtime_capabilities_json(
            buffer: *mut c_char,
            capacity: usize,
            required: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_last_error_message(
            buffer: *mut c_char,
            capacity: usize,
            required: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_load(
            process_dir: *const c_char,
            process_key: *const c_char,
            model_parameters_path: *const c_char,
            output: *mut *mut RuntimeHandle,
        ) -> c_int;
        pub(super) fn rusticol_runtime_free(handle: *mut RuntimeHandle) -> c_int;
        pub(super) fn rusticol_runtime_metadata_json(
            handle: *const RuntimeHandle,
            buffer: *mut c_char,
            capacity: usize,
            required: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_execution_mode(
            handle: *const RuntimeHandle,
            buffer: *mut c_char,
            capacity: usize,
            required: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_physics_json(
            handle: *const RuntimeHandle,
            buffer: *mut c_char,
            capacity: usize,
            required: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_process(
            handle: *const RuntimeHandle,
            buffer: *mut c_char,
            capacity: usize,
            required: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_process_key(
            handle: *const RuntimeHandle,
            buffer: *mut c_char,
            capacity: usize,
            required: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_color_accuracy(
            handle: *const RuntimeHandle,
            buffer: *mut c_char,
            capacity: usize,
            required: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_external_count(
            handle: *const RuntimeHandle,
            output: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_external_pdg(
            handle: *const RuntimeHandle,
            index: usize,
            output: *mut i32,
        ) -> c_int;
        pub(super) fn rusticol_runtime_helicity_count(
            handle: *const RuntimeHandle,
            output: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_helicity_id(
            handle: *const RuntimeHandle,
            index: usize,
            buffer: *mut c_char,
            capacity: usize,
            required: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_helicity_vector(
            handle: *const RuntimeHandle,
            index: usize,
            output: *mut i32,
            capacity: usize,
            required: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_color_count(
            handle: *const RuntimeHandle,
            output: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_color_id(
            handle: *const RuntimeHandle,
            index: usize,
            buffer: *mut c_char,
            capacity: usize,
            required: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_color_kind(
            handle: *const RuntimeHandle,
            index: usize,
            buffer: *mut c_char,
            capacity: usize,
            required: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_color_word(
            handle: *const RuntimeHandle,
            index: usize,
            output: *mut usize,
            capacity: usize,
            required: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_model_parameter_count(
            handle: *const RuntimeHandle,
            output: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_model_parameter_name(
            handle: *const RuntimeHandle,
            index: usize,
            buffer: *mut c_char,
            capacity: usize,
            required: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_resolved_shape(
            handle: *const RuntimeHandle,
            helicity_ids: *const *const c_char,
            helicity_count: usize,
            color_ids: *const *const c_char,
            color_count: usize,
            output_helicity_count: *mut usize,
            output_color_count: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_evaluate_f64(
            handle: *mut RuntimeHandle,
            momenta: *const f64,
            momentum_count: usize,
            point_count: usize,
            output: *mut f64,
            output_capacity: usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_evaluate_selected_f64(
            handle: *mut RuntimeHandle,
            momenta: *const f64,
            momentum_count: usize,
            point_count: usize,
            helicity_ids: *const *const c_char,
            helicity_count: usize,
            color_ids: *const *const c_char,
            color_count: usize,
            helicity_by_point: *const u32,
            helicity_by_point_count: usize,
            color_flow_by_point: *const u32,
            color_flow_by_point_count: usize,
            output: *mut f64,
            output_capacity: usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_evaluate_resolved_f64(
            handle: *mut RuntimeHandle,
            momenta: *const f64,
            momentum_count: usize,
            point_count: usize,
            helicity_ids: *const *const c_char,
            helicity_count: usize,
            color_ids: *const *const c_char,
            color_count: usize,
            output: *mut f64,
            output_capacity: usize,
            output_helicity_count: *mut usize,
            output_color_count: *mut usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_set_model_parameters(
            handle: *mut RuntimeHandle,
            names: *const *const c_char,
            real: *const f64,
            imaginary: *const f64,
            count: usize,
        ) -> c_int;
        pub(super) fn rusticol_runtime_set_model_parameter(
            handle: *mut RuntimeHandle,
            name: *const c_char,
            real: f64,
            imaginary: f64,
        ) -> c_int;
        pub(super) fn rusticol_runtime_set_model_parameters_json(
            handle: *mut RuntimeHandle,
            path: *const c_char,
        ) -> c_int;
        pub(super) fn rusticol_runtime_mute_warnings(
            handle: *mut RuntimeHandle,
            muted: c_int,
        ) -> c_int;
        pub(super) fn rusticol_runtime_take_warnings_json(
            handle: *mut RuntimeHandle,
            buffer: *mut c_char,
            capacity: usize,
            required: *mut usize,
        ) -> c_int;
    }

    pub(super) use rusticol_abi_version as abi_version;
    pub(super) use rusticol_last_error_message as last_error_message;
    pub(super) use rusticol_runtime_color_accuracy as runtime_color_accuracy;
    pub(super) use rusticol_runtime_color_count as runtime_color_count;
    pub(super) use rusticol_runtime_color_id as runtime_color_id;
    pub(super) use rusticol_runtime_color_kind as runtime_color_kind;
    pub(super) use rusticol_runtime_color_word as runtime_color_word;
    pub(super) use rusticol_runtime_evaluate_f64 as runtime_evaluate_f64;
    pub(super) use rusticol_runtime_evaluate_resolved_f64 as runtime_evaluate_resolved_f64;
    pub(super) use rusticol_runtime_evaluate_selected_f64 as runtime_evaluate_selected_f64;
    pub(super) use rusticol_runtime_execution_mode as runtime_execution_mode;
    pub(super) use rusticol_runtime_external_count as runtime_external_count;
    pub(super) use rusticol_runtime_external_pdg as runtime_external_pdg;
    pub(super) use rusticol_runtime_free as runtime_free;
    pub(super) use rusticol_runtime_helicity_count as runtime_helicity_count;
    pub(super) use rusticol_runtime_helicity_id as runtime_helicity_id;
    pub(super) use rusticol_runtime_helicity_vector as runtime_helicity_vector;
    pub(super) use rusticol_runtime_load as runtime_load;
    pub(super) use rusticol_runtime_metadata_json as runtime_metadata_json;
    pub(super) use rusticol_runtime_model_parameter_count as runtime_model_parameter_count;
    pub(super) use rusticol_runtime_model_parameter_name as runtime_model_parameter_name;
    pub(super) use rusticol_runtime_mute_warnings as runtime_mute_warnings;
    pub(super) use rusticol_runtime_physics_json as runtime_physics_json;
    pub(super) use rusticol_runtime_process as runtime_process;
    pub(super) use rusticol_runtime_process_key as runtime_process_key;
    pub(super) use rusticol_runtime_resolved_shape as runtime_resolved_shape;
    pub(super) use rusticol_runtime_set_model_parameter as runtime_set_model_parameter;
    pub(super) use rusticol_runtime_set_model_parameters as runtime_set_model_parameters;
    pub(super) use rusticol_runtime_set_model_parameters_json as runtime_set_model_parameters_json;
    pub(super) use rusticol_runtime_take_warnings_json as runtime_take_warnings_json;
    pub(super) use rusticol_supported_runtime_capabilities_json as supported_runtime_capabilities_json;
}
