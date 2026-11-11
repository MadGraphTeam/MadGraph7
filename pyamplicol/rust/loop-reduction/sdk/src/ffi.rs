// SPDX-License-Identifier: 0BSD
//! Additive ABI following the existing C SDK's panic, buffer and error rules.
use super::*;
use std::{
    cell::RefCell,
    ffi::{CStr, CString, c_char, c_int},
    panic::{AssertUnwindSafe, catch_unwind},
    ptr, slice,
};

pub const STATUS_OK: c_int = 0;
pub const STATUS_INVALID_ARGUMENT: c_int = 1;
pub const STATUS_BUFFER_TOO_SMALL: c_int = 2;
pub const STATUS_RUNTIME_ERROR: c_int = 3;
pub const STATUS_PANIC: c_int = 4;
struct Error(c_int, String);
type Result<T> = std::result::Result<T, Error>;
impl From<RusticolError> for Error {
    fn from(e: RusticolError) -> Self {
        use rusticol_core::RusticolErrorKind::*;
        let code = match e.kind() {
            InvalidArgument | Selector | ModelParameter | UnsupportedPrecision => {
                STATUS_INVALID_ARGUMENT
            }
            _ => STATUS_RUNTIME_ERROR,
        };
        Self(code, e.to_string())
    }
}
fn bad(message: impl ToString) -> Error {
    Error(STATUS_INVALID_ARGUMENT, message.to_string())
}
thread_local! { static LAST_ERROR: RefCell<CString> = RefCell::new(CString::new("").unwrap()); }
fn guard(f: impl FnOnce() -> Result<()>) -> c_int {
    let error = match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => return STATUS_OK,
        Ok(Err(e)) => e,
        Err(e) => Error(
            STATUS_PANIC,
            e.downcast_ref::<&str>()
                .copied()
                .or_else(|| e.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown Rust panic")
                .to_owned(),
        ),
    };
    LAST_ERROR.with(|v| *v.borrow_mut() = CString::new(error.1.replace('\0', "\\0")).unwrap());
    error.0
}
unsafe fn text<'a>(p: *const c_char) -> Result<Option<&'a str>> {
    if p.is_null() {
        return Ok(None);
    }
    // SAFETY: ABI strings must remain readable and NUL-terminated for this call.
    unsafe { CStr::from_ptr(p) }.to_str().map(Some).map_err(bad)
}
unsafe fn required_text<'a>(p: *const c_char) -> Result<&'a str> {
    unsafe { text(p) }?.ok_or_else(|| bad("required string is null"))
}
unsafe fn reference<'a, T>(p: *const T) -> Result<&'a T> {
    if p.is_null() {
        return Err(bad("required handle or structure is null"));
    }
    // SAFETY: caller supplies the live, aligned object documented by the ABI.
    Ok(unsafe { &*p })
}
unsafe fn mutable<'a, T>(p: *mut T) -> Result<&'a mut T> {
    if p.is_null() {
        return Err(bad("required output or handle is null"));
    }
    // SAFETY: caller owns exclusive access for the call.
    Ok(unsafe { &mut *p })
}
unsafe fn values<'a, T>(p: *const T, length: usize) -> Result<&'a [T]> {
    if length == 0 {
        return Ok(&[]);
    }
    if p.is_null()
        || length
            .checked_mul(std::mem::size_of::<T>())
            .is_none_or(|n| n > isize::MAX as usize)
    {
        return Err(bad("invalid input pointer or element count"));
    }
    // SAFETY: non-null pointer has caller-provided initialized storage of length.
    Ok(unsafe { slice::from_raw_parts(p, length) })
}
unsafe fn copy<T: Copy>(data: &[T], output: *mut T, capacity: usize) -> Result<()> {
    if capacity < data.len() {
        return Err(Error(
            STATUS_BUFFER_TOO_SMALL,
            format!("buffer needs {} elements", data.len()),
        ));
    }
    if data.is_empty() {
        return Ok(());
    }
    if output.is_null() {
        return Err(bad("nonempty output buffer is null"));
    }
    // SAFETY: capacity checked, caller's writable storage must not alias result.
    unsafe { ptr::copy_nonoverlapping(data.as_ptr(), output, data.len()) };
    Ok(())
}
unsafe fn json_string(
    value: &str,
    output: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> Result<()> {
    let bytes = CString::new(value).map_err(bad)?;
    unsafe {
        *mutable(needed)? = bytes.as_bytes_with_nul().len();
    }
    if output.is_null() && capacity == 0 {
        return Ok(());
    }
    unsafe { copy(bytes.as_bytes_with_nul(), output.cast::<u8>(), capacity) }
}

/// Opaque handles must not be called concurrently. Results are immutable.
pub struct VirtualHandle(Runtime);
pub struct VirtualResult(Evaluation);
pub struct CorrelatedResult(CorrelatedEvaluation);

#[repr(C)]
pub struct BatchRequest {
    pub momenta: *const f64,
    pub momentum_count: usize,
    pub point_count: usize,
    pub point_ids: *const u64,
    pub mu_r2: *const f64,
    pub scale_count: usize,
    pub resolved: u32,
    pub r2_strategy: *const c_char,
}

/// The selectors of `rusticol_virtual_evaluate_selected` (additive, ABI
/// version 2 kept; m-lit lit-sdk-b): global string IDs and per-point
/// zero-based indices (a zero count omits that selector) and a resolved
/// evaluation's colour resolution (NULL: "contracted").
#[repr(C)]
pub struct SelectionRequest {
    pub helicity_ids: *const *const c_char,
    pub helicity_count: usize,
    pub color_ids: *const *const c_char,
    pub color_count: usize,
    pub helicity_by_point: *const u32,
    pub helicity_by_point_count: usize,
    pub color_flow_by_point: *const u32,
    pub color_flow_by_point_count: usize,
    pub color_resolution: *const c_char,
}

/// Borrowed source rows, separate from BatchRequest to preserve its ABI.
#[repr(C)]
pub struct SpinSourceRequest {
    pub legs: *const usize,
    pub leg_count: usize,
    pub values: *const Complex64,
    pub value_count: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ResultShape {
    pub requested_points: usize,
    pub successful_points: usize,
    pub born_values: usize,
    pub virtual_values: usize,
    pub resolved: u32,
    pub axes: Shape,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CorrelatedResultShape {
    pub requested_points: usize,
    pub successful_points: usize,
    pub born_values: usize,
    pub virtual_values: usize,
    pub axes: CorrelatedShape,
}

#[unsafe(no_mangle)]
pub extern "C" fn rusticol_virtual_abi_version() -> u32 {
    2
}
#[unsafe(no_mangle)]
pub extern "C" fn rusticol_virtual_last_error_message() -> *const c_char {
    LAST_ERROR.with(|v| v.borrow().as_ptr())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_open(
    path: *const c_char,
    process: *const c_char,
    parameters_json: *const c_char,
    strategy: *const c_char,
    output: *mut *mut VirtualHandle,
) -> c_int {
    guard(|| unsafe {
        let output = mutable(output)?;
        *output = ptr::null_mut();
        let parameters = text(parameters_json)?
            .map(serde_json::from_str::<SourceOverrides>)
            .transpose()
            .map_err(bad)?;
        let runtime = Runtime::open(
            required_text(path)?,
            text(process)?,
            parameters.as_ref(),
            text(strategy)?.unwrap_or("auto"),
        )?;
        *output = Box::into_raw(Box::new(VirtualHandle(runtime)));
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_free(handle: *mut VirtualHandle) -> c_int {
    guard(|| {
        if !handle.is_null() {
            unsafe {
                drop(Box::from_raw(handle));
            }
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_clear(handle: *mut VirtualHandle) -> c_int {
    guard(|| {
        unsafe { mutable(handle)? }.0.clear();
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_update_parameters(
    handle: *mut VirtualHandle,
    changes_json: *const c_char,
    revision: *mut u64,
) -> c_int {
    guard(|| unsafe {
        let revision = mutable(revision)?;
        let changes = serde_json::from_str(required_text(changes_json)?).map_err(bad)?;
        *revision = mutable(handle)?.0.update_parameters(&changes)?;
        Ok(())
    })
}
/// The pole selection of the handle's later evaluations, the Python API's
/// four: "all" (NULL, the default), "none" (plan v4: the finite part, pole
/// entries quiet NaNs), "uv" (plan v4: the bare loop's UV poles) or "ir"
/// (plan v4: the renormalised virtual's IR poles).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_set_poles(
    handle: *mut VirtualHandle,
    poles: *const c_char,
) -> c_int {
    guard(|| unsafe {
        let poles = crate::Poles::parse(text(poles)?.unwrap_or("all"))?;
        mutable(handle)?.0.set_poles(poles)?;
        Ok(())
    })
}
/// The arithmetic of the handle's later evaluations, the Python API's
/// `arithmetic=` / `precision=`: "double" (NULL, the default; `digits` 0),
/// "double-double" (`digits` 0) or "arbitrary" with `digits` (> 0) decimal
/// digits. Extended-precision results keep their exact values as decimal
/// literals (`rusticol_virtual_result_decimal_json`); their binary64 copies
/// are the nearest doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_set_arithmetic(
    handle: *mut VirtualHandle,
    arithmetic: *const c_char,
    digits: u32,
) -> c_int {
    guard(|| unsafe {
        let arithmetic = crate::Arithmetic::parse(
            text(arithmetic)?.unwrap_or("double"),
            (digits != 0).then_some(digits),
        )?;
        mutable(handle)?.0.set_arithmetic(arithmetic)?;
        Ok(())
    })
}
/// The reduction backend of the handle's later evaluations, the Python
/// API's `reduction=`: "auto" (NULL, the default: the process type's
/// default where it serves the call, else NinjaRS, the reason recorded),
/// "ninjars" or "tirust". An explicit backend the native library does not
/// build is refused here, one that cannot serve a later call's carrier or
/// route by that call (`RUSTICOL_VIRTUAL_RUNTIME_ERROR`, the Python API's
/// `ReductionUnavailable`); never a silent switch. Each result records the
/// requested and effective backends and the reason in its diagnostics.
///
/// # Safety
///
/// `handle` is NULL or a live handle of `rusticol_virtual_open` used by no
/// other thread; `reduction` is NULL or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_set_reduction(
    handle: *mut VirtualHandle,
    reduction: *const c_char,
) -> c_int {
    guard(|| unsafe {
        mutable(handle)?
            .0
            .set_reduction(text(reduction)?.unwrap_or("auto"))?;
        Ok(())
    })
}
/// The failure policy of the handle's later evaluations, the Python API's
/// `on_failure=` (m-lit ux-besteffort; additive, ABI version 2 kept):
/// `"best-effort"` (the default, also for NULL) returns OK with every point:
/// a point no configured alternative accepts carries the unchecked
/// reduction's value as its row with status `RUSTICOL_VIRTUAL_BEST_EFFORT`
/// (never a success), its refusal (diagnostics "best_effort") and its
/// uncertainty (`rusticol_virtual_result_copy_uncertainties`); a point
/// without any value stays `RUSTICOL_VIRTUAL_FAILED`; nothing is printed.
/// `"return"` keeps such points failed without a best-effort evaluation;
/// `"raise"` (opt-in strict mode) returns `RUSTICOL_VIRTUAL_RUNTIME_ERROR`
/// for a call with such a point. Another value is an invalid argument.
///
/// # Safety
///
/// `handle` is NULL or a live handle of `rusticol_virtual_open` used by no
/// other thread; `on_failure` is NULL or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_set_on_failure(
    handle: *mut VirtualHandle,
    on_failure: *const c_char,
) -> c_int {
    guard(|| unsafe {
        let policy = crate::OnFailure::parse(text(on_failure)?.unwrap_or("best-effort"))
            .map_err(|error| bad(error.to_string()))?;
        mutable(handle)?.0.set_on_failure(policy);
        Ok(())
    })
}
/// The target accuracy of the handle's later evaluations, the Python API's
/// `reduction_target=` (m-lit F3 (a); additive, ABI version 2 kept): the
/// relative accuracy of each point's result (every squared-order sector
/// against its own size), 1e-5 by default (`reduction_policy::
/// DEFAULT_TARGET`). TIRust's
/// method-choice acceptance follows it (calibrated); NinjaRS has no setting
/// it drives. A value that is not a finite positive number is
/// `RUSTICOL_VIRTUAL_INVALID_ARGUMENT` and nothing changes.
///
/// # Safety
///
/// `handle` is NULL or a live handle of `rusticol_virtual_open` used by no
/// other thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_set_reduction_target(
    handle: *mut VirtualHandle,
    target: f64,
) -> c_int {
    guard(|| unsafe {
        mutable(handle)?.0.set_reduction_target(target)?;
        Ok(())
    })
}
/// The unchecked setting of the handle's later evaluations, the Python API's
/// `reduction_unchecked=` (m-lit lit-unchecked; additive, ABI version 2
/// kept): 1 reduces without the reduction backends' own acceptance and
/// refusal checks (TIRust's gates and method choice as method selection
/// without the acceptance pass, its gate-0 PV last resort; NinjaRS's zero
/// stability thresholds), 0 (the default) restores them. Any other value is
/// `RUSTICOL_VIRTUAL_INVALID_ARGUMENT` (`reduction_policy::UNCHECKED_REFUSAL`)
/// and nothing changes.
///
/// # Safety
///
/// `handle` is NULL or a live handle of `rusticol_virtual_open` used by no
/// other thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_set_reduction_unchecked(
    handle: *mut VirtualHandle,
    unchecked: c_int,
) -> c_int {
    guard(|| unsafe {
        let unchecked =
            reduction_policy::unchecked(i64::from(unchecked)).map_err(|refusal| match refusal {
                Refusal::Invalid(message) | Refusal::Unavailable(message) => bad(message),
            })?;
        mutable(handle)?.0.set_reduction_unchecked(unchecked);
        Ok(())
    })
}
/// The kinematics improvement of the handle's later evaluations, the Python
/// API's `kinematics_improvement=` (m-lit lit-psimprove; additive, ABI
/// version 2 kept): `"z-shift"` (MadLoop's ImprovePSPoint 2 with MadLoop's
/// own fallback to 1, finished by the projection where MadLoop leaves the
/// point inexact; the default, also for NULL), `"rescaling"`
/// (ImprovePSPoint 1), `"projection"` (the frame-independent minimal-norm
/// projection) or `"off"`. Before a double-double or arbitrary-precision
/// evaluation (and a binary64 one under
/// `rusticol_virtual_set_kinematics_improvement_binary64`) each point is
/// moved onto exact energy-momentum conservation and the external shells in
/// that carrier, from the caller's inputs in the caller's leg order; the
/// diagnostics record it per point (`"kinematics"`). Another value is
/// `RUSTICOL_VIRTUAL_INVALID_ARGUMENT` (`kinematics_improvement::
/// SETTING_REFUSAL`) and nothing changes.
///
/// # Safety
///
/// `handle` is NULL or a live handle of `rusticol_virtual_open` used by no
/// other thread; `setting` is NULL or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_set_kinematics_improvement(
    handle: *mut VirtualHandle,
    setting: *const c_char,
) -> c_int {
    guard(|| unsafe {
        let default = kinematics_improvement::DEFAULT_VARIANT.name();
        mutable(handle)?
            .0
            .set_kinematics_improvement(text(setting)?.unwrap_or(default))?;
        Ok(())
    })
}
/// Whether the handle's later binary64 evaluations are improved too, the
/// Python API's `kinematics_improvement_binary64=` (m-lit lit-psimprove;
/// additive, ABI version 2 kept): 1 improves them with the selected variant,
/// 0 (the default) evaluates them as given (bitwise as without the
/// improvement). Any other value is `RUSTICOL_VIRTUAL_INVALID_ARGUMENT`
/// (`kinematics_improvement::BINARY64_REFUSAL`) and nothing changes.
///
/// # Safety
///
/// `handle` is NULL or a live handle of `rusticol_virtual_open` used by no
/// other thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_set_kinematics_improvement_binary64(
    handle: *mut VirtualHandle,
    binary64: c_int,
) -> c_int {
    guard(|| unsafe {
        let binary64 = kinematics_improvement::binary64_switch(i64::from(binary64)).map_err(bad)?;
        mutable(handle)?
            .0
            .set_kinematics_improvement_binary64(binary64);
        Ok(())
    })
}
/// The stability checks of the handle's later evaluations, the Python API's
/// `stability_checks=` (`VirtualStabilityChecks`; additive, ABI version 2
/// kept): `rotation_count` default rotations (the Python API's count), or,
/// with `rotation_count` 0, the rotations of `rotations`, a JSON array of
/// rotation specifications (`"<x|y|z>:<cos>:<sin>"`,
/// `"<ax>,<ay>,<az>:<radians>"`, `"euler:<alpha>,<beta>,<gamma>"`; NULL: none);
/// `reduction_direction`, `rescaled` and `cross_backend` are 0 or 1. All zero
/// and NULL (the default) runs no check: no estimate, no extra cost. Checks
/// the Python API refuses are `RUSTICOL_VIRTUAL_INVALID_ARGUMENT` with its
/// message, and nothing changes.
///
/// # Safety
///
/// `handle` is NULL or a live handle of `rusticol_virtual_open` used by no
/// other thread; `rotations` is NULL or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_set_stability_checks(
    handle: *mut VirtualHandle,
    rotation_count: u64,
    rotations: *const c_char,
    reduction_direction: u32,
    rescaled: u32,
    cross_backend: u32,
) -> c_int {
    guard(|| unsafe {
        let flag = |value: u32, name: &str| match value {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(bad(format!("{name} must be a boolean (0 or 1)"))),
        };
        let rotations = match text(rotations)? {
            None => (0..rotation_count)
                .map(crate::default_rotation)
                .collect::<RusticolResult<Vec<_>>>()?,
            Some(_) if rotation_count != 0 => {
                return Err(bad("rotations take a count or rotation records"));
            }
            Some(json) => serde_json::from_str::<Vec<String>>(json)
                .map_err(|e| bad(format!("rotations must be a JSON array of strings: {e}")))?
                .iter()
                .map(|specification| Rotation::parse(specification))
                .collect::<RusticolResult<Vec<_>>>()?,
        };
        let checks = StabilityChecks::new(
            rotations,
            flag(reduction_direction, "reduction_direction")?,
            flag(rescaled, "rescaled")?,
            flag(cross_backend, "cross_backend")?,
        )?;
        mutable(handle)?.0.set_stability_checks(checks);
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_shape(
    handle: *const VirtualHandle,
    output: *mut Shape,
) -> c_int {
    guard(|| unsafe {
        *mutable(output)? = reference(handle)?.0.shape();
        Ok(())
    })
}
macro_rules! runtime_json {
    ($name:ident, $method:ident) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            handle: *const VirtualHandle,
            output: *mut c_char,
            capacity: usize,
            needed: *mut usize,
        ) -> c_int {
            guard(|| unsafe {
                json_string(&reference(handle)?.0.$method()?, output, capacity, needed)
            })
        }
    };
}
runtime_json!(rusticol_virtual_metadata_json, metadata_json);
runtime_json!(rusticol_virtual_capabilities_json, capabilities_json);
runtime_json!(
    rusticol_virtual_correlation_histories_json,
    correlation_histories_json
);

runtime_json!(
    rusticol_virtual_spin_correlation_classes_json,
    spin_correlation_classes_json
);

// Common batch shape and opaque point identities.
unsafe fn request_ids(runtime: &Runtime, request: &BatchRequest) -> Result<Vec<u64>> {
    unsafe {
        if request.momentum_count
            != product(&[request.point_count, runtime.shape().external_count, 4])?
        {
            return Err(bad("momentum count differs from point/leg shape"));
        }
        if request.point_ids.is_null() {
            (0..request.point_count)
                .map(|v| u64::try_from(v).map_err(bad))
                .collect::<Result<Vec<_>>>()
        } else {
            Ok(values(request.point_ids, request.point_count)?.to_vec())
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_evaluate(
    handle: *mut VirtualHandle,
    request: *const BatchRequest,
    output: *mut *mut VirtualResult,
) -> c_int {
    guard(|| unsafe {
        let output = mutable(output)?;
        *output = ptr::null_mut();
        let request = reference(request)?;
        if request.resolved > 1 {
            return Err(bad("resolved must be zero or one"));
        }
        let handle = mutable(handle)?;
        let ids = request_ids(&handle.0, request)?;
        let result = handle.0.evaluate(
            values(request.momenta, request.momentum_count)?,
            &ids,
            values(request.mu_r2, request.scale_count)?,
            request.resolved == 1,
            text(request.r2_strategy)?,
        )?;
        *output = Box::into_raw(Box::new(VirtualResult(result)));
        Ok(())
    })
}
/// String IDs of a selector array (`count` NUL-terminated strings; a zero
/// count omits the selector).
unsafe fn selector_ids(
    ids: *const *const c_char,
    count: usize,
    name: &str,
) -> Result<Option<Vec<String>>> {
    if count == 0 {
        return Ok(None);
    }
    unsafe { values(ids, count) }
        .map_err(|_| bad(format!("{name} array is null")))?
        .iter()
        .enumerate()
        .map(|(index, &id)| {
            unsafe { text(id) }?
                .map(str::to_owned)
                .ok_or_else(|| bad(format!("{name}[{index}] is null")))
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

/// Per-point indices (`count` values; a zero count omits the selector).
unsafe fn per_point(indices: *const u32, count: usize, name: &str) -> Result<Option<Vec<u32>>> {
    if count == 0 {
        return Ok(None);
    }
    Ok(Some(
        unsafe { values(indices, count) }
            .map_err(|_| bad(format!("{name} is null")))?
            .to_vec(),
    ))
}

/// An evaluation with selectors (the Python API's `helicities=` and
/// `color_resolution=`, the tree C ABI's selector conventions; m-lit
/// lit-sdk-b): `request.resolved` 0 is the ordinary evaluation of the
/// selected helicity rows, 1 the resolved evaluation of the selected rows in
/// the colour resolution. A NULL `selection` selects nothing
/// (`rusticol_virtual_evaluate`). Every selector is validated before any
/// evaluation; the result is written only on success.
///
/// # Safety
///
/// `handle` is a live handle used by no other thread; `request`, `selection`
/// (NULL or) and `output` follow the ABI's pointer rules: every nonzero
/// selector count names that many readable entries, each ID a
/// NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_evaluate_selected(
    handle: *mut VirtualHandle,
    request: *const BatchRequest,
    selection: *const SelectionRequest,
    output: *mut *mut VirtualResult,
) -> c_int {
    guard(|| unsafe {
        let output = mutable(output)?;
        *output = ptr::null_mut();
        let request = reference(request)?;
        if request.resolved > 1 {
            return Err(bad("resolved must be zero or one"));
        }
        let selection = if selection.is_null() {
            Selection::default()
        } else {
            let selection = reference(selection)?;
            Selection {
                helicities: selector_ids(
                    selection.helicity_ids,
                    selection.helicity_count,
                    "helicity ids",
                )?,
                color_flows: selector_ids(selection.color_ids, selection.color_count, "color ids")?,
                helicity_by_point: per_point(
                    selection.helicity_by_point,
                    selection.helicity_by_point_count,
                    "helicity_by_point",
                )?,
                color_flow_by_point: per_point(
                    selection.color_flow_by_point,
                    selection.color_flow_by_point_count,
                    "color_flow_by_point",
                )?,
                color_resolution: text(selection.color_resolution)?.map(str::to_owned),
            }
        };
        let handle = mutable(handle)?;
        let ids = request_ids(&handle.0, request)?;
        let result = handle.0.evaluate_selected(
            values(request.momenta, request.momentum_count)?,
            &ids,
            values(request.mu_r2, request.scale_count)?,
            request.resolved == 1,
            text(request.r2_strategy)?,
            &selection,
        )?;
        *output = Box::into_raw(Box::new(VirtualResult(result)));
        Ok(())
    })
}

/// The axes of a result as JSON: `{"helicity_ids": [...], "color_ids":
/// [...]}` (a resolved result's helicity rows and colour components in its
/// layout's order; empty for an ordinary result).
///
/// # Safety
///
/// `result` is a live result; `output`, `capacity` and `needed` follow the
/// ABI's JSON size-query rules.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_result_axes_json(
    result: *const VirtualResult,
    output: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> c_int {
    guard(|| unsafe {
        let result = &reference(result)?.0;
        let axes = serde_json::json!({
            "helicity_ids": result.helicity_ids,
            "color_ids": result.color_ids,
        });
        json_string(&axes.to_string(), output, capacity, needed)
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_result_free(result: *mut VirtualResult) -> c_int {
    guard(|| {
        if !result.is_null() {
            unsafe {
                drop(Box::from_raw(result));
            }
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_result_shape(
    result: *const VirtualResult,
    output: *mut ResultShape,
) -> c_int {
    guard(|| unsafe {
        let result = &reference(result)?.0;
        *mutable(output)? = ResultShape {
            requested_points: result.requested_point_ids.len(),
            successful_points: result.successful_point_ids.len(),
            born_values: result.born.len(),
            virtual_values: result.virtual_values.len(),
            resolved: u32::from(result.resolved),
            axes: result.shape,
        };
        Ok(())
    })
}
macro_rules! copy_result {
    ($name:ident, $result:ty, $field:ident, $ty:ty) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(
            result: *const $result,
            output: *mut $ty,
            capacity: usize,
        ) -> c_int {
            guard(|| unsafe { copy(&reference(result)?.0.$field, output, capacity) })
        }
    };
}
copy_result!(rusticol_virtual_result_copy_born, VirtualResult, born, f64);
copy_result!(
    rusticol_virtual_result_copy_virtual,
    VirtualResult,
    virtual_values,
    f64
);
copy_result!(
    rusticol_virtual_result_copy_requested_ids,
    VirtualResult,
    requested_point_ids,
    u64
);
copy_result!(
    rusticol_virtual_result_copy_successful_ids,
    VirtualResult,
    successful_point_ids,
    u64
);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_result_copy_statuses(
    result: *const VirtualResult,
    output: *mut u32,
    capacity: usize,
) -> c_int {
    guard(|| unsafe {
        copy(
            &reference(result)?
                .0
                .statuses
                .iter()
                .map(|s| *s as u32)
                .collect::<Vec<_>>(),
            output,
            capacity,
        )
    })
}
/// Per requested point the relative uncertainty of a best-effort point's
/// value (`RUSTICOL_VIRTUAL_BEST_EFFORT`), NaN for every other point and for
/// a best-effort point without an estimate (m-lit ux-besteffort).
///
/// # Safety
///
/// `result` is a live result; `output` holds `capacity` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_result_copy_uncertainties(
    result: *const VirtualResult,
    output: *mut f64,
    capacity: usize,
) -> c_int {
    guard(|| unsafe {
        let result = &reference(result)?.0;
        copy(
            &uncertainties(&result.uncertainties, result.statuses.len()),
            output,
            capacity,
        )
    })
}
/// [`rusticol_virtual_result_copy_uncertainties`] of a correlated result.
///
/// # Safety
///
/// `result` is a live result; `output` holds `capacity` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_correlated_result_copy_uncertainties(
    result: *const CorrelatedResult,
    output: *mut f64,
    capacity: usize,
) -> c_int {
    guard(|| unsafe {
        let result = &reference(result)?.0;
        copy(
            &uncertainties(&result.uncertainties, result.statuses.len()),
            output,
            capacity,
        )
    })
}
/// One uncertainty per requested point (NaN where none was recorded).
fn uncertainties(values: &[f64], points: usize) -> Vec<f64> {
    (0..points)
        .map(|i| values.get(i).copied().unwrap_or(f64::NAN))
        .collect()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_result_diagnostics_json(
    result: *const VirtualResult,
    output: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> c_int {
    guard(|| unsafe {
        json_string(
            &reference(result)?.0.diagnostics_json,
            output,
            capacity,
            needed,
        )
    })
}

/// The exact values of an extended-precision evaluation as JSON
/// (`{"arithmetic", "digits", "born", "virtual"}`: decimal literals in the
/// layout of the binary64 copies, absent poles `null`); `null` for a
/// binary64 result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_result_decimal_json(
    result: *const VirtualResult,
    output: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> c_int {
    guard(|| unsafe {
        json_string(
            &reference(result)?.0.decimal_json()?,
            output,
            capacity,
            needed,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn guards_and_short_buffers_never_write_partial_values() {
        let mut canary = [71_u64, 72, 73];
        assert!(unsafe { copy(&[1, 2, 3], canary.as_mut_ptr(), 2) }.is_err());
        assert_eq!(canary, [71, 72, 73]);
        assert_eq!(guard(|| Err(bad("test"))), STATUS_INVALID_ARGUMENT);
        assert_eq!(guard(|| panic!("injected boundary panic")), STATUS_PANIC);
        let mut needed = 0;
        assert!(unsafe { json_string("abc", ptr::null_mut(), 0, &mut needed) }.is_ok());
        assert_eq!(needed, 4);
        assert_eq!(
            unsafe { rusticol_virtual_shape(ptr::null(), ptr::null_mut()) },
            STATUS_INVALID_ARGUMENT
        );
    }
}

/// Grouped complex correlations have their own opaque result type. The input
/// structure is shared, but resolved=1 is rejected before numerical work.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_evaluate_correlated(
    handle: *mut VirtualHandle,
    request: *const BatchRequest,
    output: *mut *mut CorrelatedResult,
) -> c_int {
    guard(|| unsafe {
        let output = mutable(output)?;
        *output = ptr::null_mut();
        let request = reference(request)?;
        if request.resolved != 0 {
            return Err(bad("resolved correlated evaluations are unavailable"));
        }
        let handle = mutable(handle)?;
        let ids = request_ids(&handle.0, request)?;
        let result = handle.0.evaluate_correlated(
            values(request.momenta, request.momentum_count)?,
            &ids,
            values(request.mu_r2, request.scale_count)?,
            text(request.r2_strategy)?,
        )?;
        *output = Box::into_raw(Box::new(CorrelatedResult(result)));
        Ok(())
    })
}
/// Both requests are borrowed only for this call. Exact copies own replay input.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_evaluate_correlated_with_sources(
    handle: *mut VirtualHandle,
    request: *const BatchRequest,
    sources: *const SpinSourceRequest,
    output: *mut *mut CorrelatedResult,
) -> c_int {
    guard(|| unsafe {
        let output = mutable(output)?;
        *output = ptr::null_mut();
        let request = reference(request)?;
        if request.resolved != 0 {
            return Err(bad("resolved correlated evaluations are unavailable"));
        }
        let sources = reference(sources)?;
        let handle = mutable(handle)?;
        let ids = request_ids(&handle.0, request)?;
        let result = handle.0.evaluate_correlated_with_sources(
            values(request.momenta, request.momentum_count)?,
            &ids,
            values(request.mu_r2, request.scale_count)?,
            SpinSources {
                legs: values(sources.legs, sources.leg_count)?,
                values: values(sources.values, sources.value_count)?,
            },
            text(request.r2_strategy)?,
        )?;
        *output = Box::into_raw(Box::new(CorrelatedResult(result)));
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_correlated_shape(
    handle: *const VirtualHandle,
    output: *mut CorrelatedShape,
) -> c_int {
    guard(|| unsafe {
        *mutable(output)? = reference(handle)?.0.correlated_shape();
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_correlated_result_free(
    result: *mut CorrelatedResult,
) -> c_int {
    guard(|| {
        if !result.is_null() {
            unsafe {
                drop(Box::from_raw(result));
            }
        }
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_correlated_result_shape(
    result: *const CorrelatedResult,
    output: *mut CorrelatedResultShape,
) -> c_int {
    guard(|| unsafe {
        let result = &reference(result)?.0;
        *mutable(output)? = CorrelatedResultShape {
            requested_points: result.requested_point_ids.len(),
            successful_points: result.successful_point_ids.len(),
            born_values: result.born.len(),
            virtual_values: result.virtual_values.len(),
            axes: result.shape,
        };
        Ok(())
    })
}
copy_result!(
    rusticol_virtual_correlated_result_copy_born,
    CorrelatedResult,
    born,
    Complex64
);
copy_result!(
    rusticol_virtual_correlated_result_copy_virtual,
    CorrelatedResult,
    virtual_values,
    Complex64
);
copy_result!(
    rusticol_virtual_correlated_result_copy_requested_ids,
    CorrelatedResult,
    requested_point_ids,
    u64
);
copy_result!(
    rusticol_virtual_correlated_result_copy_successful_ids,
    CorrelatedResult,
    successful_point_ids,
    u64
);
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_correlated_result_copy_statuses(
    result: *const CorrelatedResult,
    output: *mut u32,
    capacity: usize,
) -> c_int {
    guard(|| unsafe {
        copy(
            &reference(result)?
                .0
                .statuses
                .iter()
                .map(|s| *s as u32)
                .collect::<Vec<_>>(),
            output,
            capacity,
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_correlated_result_diagnostics_json(
    result: *const CorrelatedResult,
    output: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> c_int {
    guard(|| unsafe {
        json_string(
            &reference(result)?.0.diagnostics_json,
            output,
            capacity,
            needed,
        )
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_correlated_result_histories_json(
    result: *const CorrelatedResult,
    output: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> c_int {
    guard(|| unsafe {
        json_string(
            &reference(result)?.0.histories_json()?,
            output,
            capacity,
            needed,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_correlated_result_spin_legs_json(
    result: *const CorrelatedResult,
    output: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> c_int {
    guard(|| unsafe {
        json_string(
            &reference(result)?.0.spin_legs_json()?,
            output,
            capacity,
            needed,
        )
    })
}

/// The exact values of an extended-precision correlated evaluation as JSON
/// (`[real, imaginary]` decimal literals in the layout of the complex
/// copies, absent poles `null`); `null` for a binary64 result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_correlated_result_decimal_json(
    result: *const CorrelatedResult,
    output: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> c_int {
    guard(|| unsafe {
        json_string(
            &reference(result)?.0.decimal_json()?,
            output,
            capacity,
            needed,
        )
    })
}
