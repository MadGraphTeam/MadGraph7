// SPDX-License-Identifier: 0BSD
//! Additive ABI following the existing C SDK's panic, buffer and error rules.
use super::*;
use serde::Deserialize;
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
    pub verification_json: *const c_char,
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

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Policy {
    higher: Vec<Level>,
    relative_tolerance: String,
    absolute_tolerance: String,
    verify_successful: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Level {
    arithmetic: String,
    precision: Option<u32>,
}
impl From<Policy> for VerificationPolicy {
    fn from(p: Policy) -> Self {
        Self {
            higher: p
                .higher
                .into_iter()
                .map(|l| ArithmeticSpec {
                    arithmetic: l.arithmetic,
                    precision: l.precision,
                })
                .collect(),
            relative_tolerance: p.relative_tolerance,
            absolute_tolerance: p.absolute_tolerance,
            verify_successful: p.verify_successful,
        }
    }
}

unsafe fn read_policy(value: *const c_char) -> Result<Option<VerificationPolicy>> {
    unsafe { text(value)? }
        .map(serde_json::from_str::<Policy>)
        .transpose()
        .map_err(bad)
        .map(|policy| policy.map(VerificationPolicy::from))
}

#[unsafe(no_mangle)]
pub extern "C" fn rusticol_virtual_abi_version() -> u32 {
    1
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
    unsafe {
        rusticol_virtual_open_with_policy(
            path,
            process,
            parameters_json,
            strategy,
            ptr::null(),
            output,
        )
    }
}
/// Additive entry point; the original open call retains eager binding semantics.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rusticol_virtual_open_with_policy(
    path: *const c_char,
    process: *const c_char,
    parameters_json: *const c_char,
    strategy: *const c_char,
    verification_json: *const c_char,
    output: *mut *mut VirtualHandle,
) -> c_int {
    guard(|| unsafe {
        let output = mutable(output)?;
        *output = ptr::null_mut();
        let parameters = text(parameters_json)?
            .map(serde_json::from_str::<SourceOverrides>)
            .transpose()
            .map_err(bad)?;
        let policy = read_policy(verification_json)?;
        let runtime = Runtime::open_with_policy(
            required_text(path)?,
            text(process)?,
            parameters.as_ref(),
            text(strategy)?.unwrap_or("auto"),
            policy.as_ref(),
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

// Common batch shape, opaque point identities and verification policy ingress.
unsafe fn request_arguments(
    runtime: &Runtime,
    request: &BatchRequest,
) -> Result<(Vec<u64>, Option<VerificationPolicy>)> {
    unsafe {
        if request.momentum_count
            != product(&[request.point_count, runtime.shape().external_count, 4])?
        {
            return Err(bad("momentum count differs from point/leg shape"));
        }
        let ids = if request.point_ids.is_null() {
            (0..request.point_count)
                .map(|v| u64::try_from(v).map_err(bad))
                .collect::<Result<Vec<_>>>()?
        } else {
            values(request.point_ids, request.point_count)?.to_vec()
        };
        let policy = read_policy(request.verification_json)?;
        Ok((ids, policy))
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
        let (ids, policy) = request_arguments(&handle.0, request)?;
        let result = handle.0.evaluate(
            values(request.momenta, request.momentum_count)?,
            &ids,
            values(request.mu_r2, request.scale_count)?,
            request.resolved == 1,
            text(request.r2_strategy)?,
            policy.as_ref(),
        )?;
        *output = Box::into_raw(Box::new(VirtualResult(result)));
        Ok(())
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
            return Err(bad("resolved directed correlations are unavailable"));
        }
        let handle = mutable(handle)?;
        let (ids, policy) = request_arguments(&handle.0, request)?;
        let result = handle.0.evaluate_correlated(
            values(request.momenta, request.momentum_count)?,
            &ids,
            values(request.mu_r2, request.scale_count)?,
            text(request.r2_strategy)?,
            policy.as_ref(),
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
            return Err(bad("resolved directed correlations are unavailable"));
        }
        let sources = reference(sources)?;
        let handle = mutable(handle)?;
        let (ids, policy) = request_arguments(&handle.0, request)?;
        let result = handle.0.evaluate_correlated_with_sources(
            values(request.momenta, request.momentum_count)?,
            &ids,
            values(request.mu_r2, request.scale_count)?,
            SpinSources {
                legs: values(sources.legs, sources.leg_count)?,
                values: values(sources.values, sources.value_count)?,
            },
            text(request.r2_strategy)?,
            policy.as_ref(),
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
