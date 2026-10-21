//! MadLoop's dimensioned C ABI for the pinned NinjaRS reduction engines.
//!
//! DP uses the binary64 fast engine directly. MP input records contain actual
//! Fortran binary128 values; reduction uses 256-bit Float on a retained worker
//! with the stack required by Symbolica expression construction.

mod tensor;
mod guarded;

use ninjars::{
    Amplitude, AmplitudeInput, Complex, EvaluationStatus, Float, FloatLike, FourMomentum, Real,
    SingleFloat,
    fast::{C64, FastAmplitude, FastInput, FastOptions, FastResult, FastStatus, TensorNumerator},
};
use ninjars_symbolica::{
    OneLoopNativeMasters, SymbolicaAmplitudeKernels, SymbolicaCuts, SymbolicaHighCuts,
    SymbolicaMasters,
};
use std::{
    cell::RefCell,
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind},
    slice,
    sync::{Mutex, mpsc},
};
use tensor::TensorGenericNumerator;

const SUCCESS: i32 = 0;
const UNSTABLE: i32 = 1;
const ERROR: i32 = 2;
const RECORD: usize = 64;
const MP_BITS: u32 = 256;
const ABI_VERSION: u32 = 3;
type Result<T> = std::result::Result<T, String>;
type Z = Complex<Float>;
type MpAmplitude = Amplitude<Z, SymbolicaAmplitudeKernels<Float>>;

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

/// Opaque caller-owned context. Destroy only after all evaluations have returned.
pub struct Context {
    dp: Mutex<Option<DpState>>,
    mp: Mutex<Option<MpContext>>,
    compensated_dp: Mutex<Option<CompensatedDpState>>,
}

#[unsafe(no_mangle)]
pub extern "C" fn ninjars_abi_version() -> u32 {
    ABI_VERSION
}

/// Create an independent retained context using the requested ABI version.
/// # Safety
/// `output` is a writable pointer slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ninjars_context_create(version: u32, output: *mut *mut Context) -> i32 {
    guard(|| {
        validate_output(output, 1)?;
        if version != ABI_VERSION {
            return Err(format!("unsupported NinjaRS ABI version {version}"));
        }
        let context = Box::new(Context {
            dp: Mutex::new(None),
            mp: Mutex::new(None),
            compensated_dp: Mutex::new(None),
        });
        unsafe {
            output.write(Box::into_raw(context));
        }
        Ok(SUCCESS)
    })
}

/// Free a context's state; the last MP context joins the shared rescue worker.
/// NULL is a no-op.
/// # Safety
/// `context` is NULL or a live handle returned by create, with no active calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ninjars_context_destroy(context: *mut Context) -> i32 {
    guard(|| {
        if !context.is_null() {
            if !context.is_aligned() {
                return Err("misaligned context".into());
            }
            unsafe {
                drop(Box::from_raw(context));
            }
        }
        Ok(SUCCESS)
    })
}

unsafe fn context<'a>(ptr: *mut Context) -> Result<&'a Context> {
    if ptr.is_null() || !ptr.is_aligned() {
        return Err("invalid context".into());
    }
    // The caller guarantees the context remains live until the call returns.
    Ok(unsafe { &*ptr })
}

fn guard(f: impl FnOnce() -> Result<i32>) -> i32 {
    let (status, message) = match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(status)) => (status, String::new()),
        Ok(Err(message)) => (ERROR, message),
        Err(_) => (ERROR, "NinjaRS panic contained at the C boundary".into()),
    };
    LAST_ERROR.with(|slot| *slot.borrow_mut() = message);
    status
}

fn dimensions(n: i32, rank: i32, count: usize) -> Result<(usize, usize)> {
    if n < 1 || rank < 0 || rank > n.saturating_add(2) {
        return Err(format!(
            "unsupported topology: n={n}, rank={rank} (require 0<=rank<=n+2)"
        ));
    }
    let (n, rank) = (n as usize, rank as usize);
    // Check dimensions before either constructing slices or allocating tensors.
    let expected = (1..=4)
        .try_fold(1usize, |v, k| v.checked_mul(rank + k))
        .ok_or("tensor dimension overflow")?
        / 24;
    n.checked_mul(n)
        .and_then(|v| v.checked_mul(RECORD))
        .ok_or("kinematic dimension overflow")?;
    if expected != count {
        return Err(format!("tensor length {count}, expected {expected}"));
    }
    Ok((n, rank))
}

fn length(label: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(format!("{label} length {actual}, expected {expected}"));
    }
    Ok(())
}

unsafe fn read<'a, T>(ptr: *const T, len: usize) -> Result<&'a [T]> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() || !ptr.is_aligned() || len > isize::MAX as usize / size_of::<T>() {
        return Err("invalid input pointer or length".into());
    }
    // The C caller owns readable, aligned storage for the documented length.
    Ok(unsafe { slice::from_raw_parts(ptr, len) })
}

fn validate_output<T>(ptr: *mut T, count: usize) -> Result<std::ops::Range<usize>> {
    let bytes = count
        .checked_mul(size_of::<T>())
        .ok_or("output size overflow")?;
    if ptr.is_null() || !ptr.is_aligned() || bytes > isize::MAX as usize {
        return Err("invalid output pointer".into());
    }
    let start = ptr as usize;
    let end = start.checked_add(bytes).ok_or("output pointer overflow")?;
    Ok(start..end)
}

unsafe fn write<T: Copy>(ptr: *mut T, values: &[T]) -> Result<()> {
    validate_output(ptr, values.len())?;
    // The C caller owns disjoint writable storage for the documented length.
    unsafe { slice::from_raw_parts_mut(ptr, values.len()) }.copy_from_slice(values);
    Ok(())
}

struct DpEntry {
    amplitude: FastAmplitude,
    tensor: TensorNumerator,
    shifts: Vec<[f64; 4]>,
}
struct DpState {
    entries: HashMap<(usize, usize), DpEntry>,
    masters: OneLoopNativeMasters,
}
impl DpState {
    fn new() -> Result<Self> {
        Ok(Self {
            entries: HashMap::new(),
            masters: OneLoopNativeMasters::new().map_err(|e| e.to_string())?,
        })
    }
    fn evaluate(
        &mut self,
        n: usize,
        rank: usize,
        coeff: &[C64],
        s: &[f64],
        p: &[f64],
        m: &[C64],
        mu2: f64,
    ) -> Result<(i32, FastResult)> {
        if !mu2.is_finite()
            || mu2 <= 0.
            || p.iter().chain(s).any(|v| !v.is_finite())
            || coeff.iter().any(|v| !v.is_finite())
            || m.iter().any(|v| !v.is_finite() || v.im > 0.)
        {
            return Err("nonfinite input, invalid mass width, or nonpositive squared scale".into());
        }
        if !self.entries.contains_key(&(n, rank)) {
            self.entries.insert(
                (n, rank),
                DpEntry {
                    amplitude: FastAmplitude::new(n, rank, FastOptions::default())
                        .map_err(|e| e.to_string())?,
                    tensor: TensorNumerator::new(rank, 1),
                    shifts: vec![[0.; 4]; n],
                },
            );
        }
        let state = self.entries.get_mut(&(n, rank)).unwrap();
        state.tensor.coefficients_mut().copy_from_slice(coeff);
        for (v, values) in state.shifts.iter_mut().zip(p.chunks_exact(4)) {
            v.copy_from_slice(values);
        }
        let mut result = [FastResult::default()];
        let mut status = [FastStatus::NotEvaluated];
        state
            .amplitude
            .evaluate(
                &FastInput {
                    lanes: 1,
                    shifts: &state.shifts,
                    masses_squared: m,
                    scales_squared: &[mu2],
                    s_matrix: Some(s),
                },
                &mut state.tensor,
                &mut self.masters,
                &mut result,
                &mut status,
            )
            .map_err(|e| e.to_string())?;
        Ok((
            if status[0] == FastStatus::Success {
                SUCCESS
            } else {
                UNSTABLE
            },
            result[0],
        ))
    }
}

struct CompensatedDpEntry {
    amplitude: FastAmplitude,
    tensor: guarded::GuardedTensor,
    shifts: Vec<[f64; 4]>,
}
struct CompensatedDpState {
    entries: HashMap<(usize, usize), CompensatedDpEntry>,
    masters: OneLoopNativeMasters,
}
impl CompensatedDpState {
    fn new() -> Result<Self> {
        Ok(Self {
            entries: HashMap::new(),
            masters: OneLoopNativeMasters::new().map_err(|e| e.to_string())?,
        })
    }
    fn evaluate(
        &mut self,
        n: usize,
        rank: usize,
        coeff: &[C64],
        s: &[f64],
        p: &[f64],
        m: &[C64],
        mu2: f64,
    ) -> Result<(i32, FastResult)> {
        if !mu2.is_finite()
            || mu2 <= 0.
            || p.iter().chain(s).any(|v| !v.is_finite())
            || coeff.iter().any(|v| !v.is_finite())
            || m.iter().any(|v| !v.is_finite() || v.im > 0.)
        {
            return Err("nonfinite input, invalid mass width, or nonpositive squared scale".into());
        }
        if !self.entries.contains_key(&(n, rank)) {
            self.entries.insert(
                (n, rank),
                CompensatedDpEntry {
                    amplitude: FastAmplitude::new(n, rank, FastOptions::default())
                        .map_err(|e| e.to_string())?,
                    tensor: guarded::GuardedTensor::new(rank, 1),
                    shifts: vec![[0.; 4]; n],
                },
            );
        }
        let state = self.entries.get_mut(&(n, rank)).unwrap();
        state.tensor.coefficients_mut().copy_from_slice(coeff);
        for (v, values) in state.shifts.iter_mut().zip(p.chunks_exact(4)) {
            v.copy_from_slice(values);
        }
        let mut result = [FastResult::default()];
        let mut status = [FastStatus::NotEvaluated];
        state
            .amplitude
            .evaluate(
                &FastInput {
                    lanes: 1,
                    shifts: &state.shifts,
                    masses_squared: m,
                    scales_squared: &[mu2],
                    s_matrix: Some(s),
                },
                &mut state.tensor,
                &mut self.masters,
                &mut result,
                &mut status,
            )
            .map_err(|e| e.to_string())?;
        Ok((
            if status[0] == FastStatus::Success {
                SUCCESS
            } else {
                UNSTABLE
            },
            result[0],
        ))
    }
}

/// Evaluate one DP numerator. Output is committed only on success.
///
/// # Safety
/// Buffers have the dimensions in ninjars.h and are valid/disjoint for this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ninjars_evaluate_dp(
    handle: *mut Context,
    n: i32,
    rank: i32,
    count: usize,
    coefficients: *const C64,
    s_matrix: *const f64,
    s_length: usize,
    momenta: *const f64,
    p_length: usize,
    masses: *const C64,
    m_length: usize,
    scale_squared: f64,
    laurent: *mut C64,
    laurent_length: usize,
    rational: *mut C64,
    rational_length: usize,
) -> i32 {
    guard(|| {
        let (n, rank) = dimensions(n, rank, count)?;
        length("S matrix", s_length, n * n)?;
        length("momenta", p_length, n * 4)?;
        length("masses", m_length, n)?;
        length("Laurent output", laurent_length, 3)?;
        length("rational output", rational_length, 1)?;
        let a = validate_output(laurent, 3)?;
        let b = validate_output(rational, 1)?;
        if a.start < b.end && b.start < a.end {
            return Err("overlapping output buffers".into());
        }
        let context = unsafe { context(handle) }?;
        let coeff = unsafe { read(coefficients, count) }?;
        let s = unsafe { read(s_matrix, n * n) }?;
        let p = unsafe { read(momenta, n * 4) }?;
        let m = unsafe { read(masses, n) }?;
        let mut state = context.dp.lock().map_err(|_| "DP context poisoned")?;
        if state.is_none() {
            *state = Some(DpState::new()?);
        }
        let (status, result) =
            state
                .as_mut()
                .unwrap()
                .evaluate(n, rank, coeff, s, p, m, scale_squared)?;
        if status == SUCCESS {
            unsafe {
                write(laurent, &result.laurent)?;
                write(rational, &[result.rational])?;
            }
        }
        Ok(status)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn ninjars_evaluate_dp_compensated(
    handle: *mut Context,
    n: i32,
    rank: i32,
    count: usize,
    coefficients: *const C64,
    s_matrix: *const f64,
    s_length: usize,
    momenta: *const f64,
    p_length: usize,
    masses: *const C64,
    m_length: usize,
    scale_squared: f64,
    laurent: *mut C64,
    laurent_length: usize,
    rational: *mut C64,
    rational_length: usize,
) -> i32 {
    guard(|| {
        let (n, rank) = dimensions(n, rank, count)?;
        length("S matrix", s_length, n * n)?;
        length("momenta", p_length, n * 4)?;
        length("masses", m_length, n)?;
        length("Laurent output", laurent_length, 3)?;
        length("rational output", rational_length, 1)?;
        let a = validate_output(laurent, 3)?;
        let b = validate_output(rational, 1)?;
        if a.start < b.end && b.start < a.end {
            return Err("overlapping output buffers".into());
        }
        let context = unsafe { context(handle) }?;
        let coeff = unsafe { read(coefficients, count) }?;
        let s = unsafe { read(s_matrix, n * n) }?;
        let p = unsafe { read(momenta, n * 4) }?;
        let m = unsafe { read(masses, n) }?;
        let mut state = context.compensated_dp.lock().map_err(|_| "DP context poisoned")?;
        if state.is_none() {
            *state = Some(CompensatedDpState::new()?);
        }
        let (status, result) =
            state
                .as_mut()
                .unwrap()
                .evaluate(n, rank, coeff, s, p, m, scale_squared)?;
        if status == SUCCESS {
            unsafe {
                write(laurent, &result.laurent)?;
                write(rational, &[result.rational])?;
            }
        }
        Ok(status)
    })
}

#[cfg(test)]
fn parse_record(text: &[u8]) -> Result<Float> {
    parse_record_at_precision(text, MP_BITS)
}
fn parse_record_at_precision(text: &[u8], precision: u32) -> Result<Float> {
    let text = std::str::from_utf8(text)
        .map_err(|_| "MP input is not ASCII")?
        .trim();
    let mut input = Float::parse(text, Some(113)).map_err(|e| format!("invalid MP value: {e}"))?;
    if !input.is_finite() {
        return Err("nonfinite MP input".into());
    }
    // Forty significant decimal digits round-trip binary128. First recover that
    // value at 113 bits, then widen the represented value itself. A rational
    // round-trip would discard the sign of zero, which selects complex branch lips.
    input.set_prec(precision);
    Ok(input)
}
fn parse_records(data: &[u8], precision: u32) -> Result<Vec<Float>> {
    data.chunks_exact(RECORD)
        .map(|record| parse_record_at_precision(record, precision))
        .collect()
}
fn pairs(values: Vec<Float>) -> Vec<Z> {
    let mut values = values.into_iter();
    let mut out = Vec::with_capacity(values.len() / 2);
    while let Some(re) = values.next() {
        out.push(Complex::new(re, values.next().unwrap()));
    }
    out
}
fn real(value: Float) -> Z {
    let imaginary = value.zero();
    Complex::new(value, imaginary)
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StudyKernelPolicy {
    Interpreted,
    NativeDenominators,
    FullNative,
}
#[cfg(test)]
impl StudyKernelPolicy {
    const ALL: [Self; 3] = [Self::Interpreted, Self::NativeDenominators, Self::FullNative];
}

struct MpEntry {
    #[cfg(test)]
    study_backends: [ninjars_symbolica::KernelBackendSelection; 8],
    amplitude: MpAmplitude,
    tensor: TensorGenericNumerator<Float>,
}
struct MpState {
    #[cfg(test)]
    study_policy: StudyKernelPolicy,
    precision: u32,
    entries: HashMap<(usize, usize), MpEntry>,
    low: SymbolicaCuts<Float>,
    high: SymbolicaHighCuts<Float>,
    masters: SymbolicaMasters<Float>,
}
impl MpState {
    fn new() -> Result<Self> {
        Self::with_precision(MP_BITS)
    }
    fn with_precision(precision: u32) -> Result<Self> {
        let prototype = Float::new(precision);
        // Original Ninja QP uses 1e15 * machine epsilon; independent of the
        // dimensionless MadLoop OSThres already applied to the input S-matrix.
        let two = Float::with_val(precision, 2);
        let threshold = Float::with_val(precision, 1_000_000_000_000_000i64) / two.pow(112);
        Ok(Self {
            #[cfg(test)]
            study_policy: StudyKernelPolicy::Interpreted,
            precision,
            entries: HashMap::new(),
            low: SymbolicaCuts::new(prototype.clone()).map_err(|e| e.to_string())?,
            high: SymbolicaHighCuts::new(prototype.clone()).map_err(|e| e.to_string())?,
            masters: SymbolicaMasters::new(prototype).with_onshell_threshold(Some(threshold)),
        })
    }
    #[cfg(test)]
    fn with_study_policy(precision: u32, policy: StudyKernelPolicy) -> Result<Self> {
        let mut state = Self::with_precision(precision)?;
        state.study_policy = policy;
        Ok(state)
    }
    fn evaluate(&mut self, request: &MpRequest) -> Result<(i32, Vec<u8>)> {
        let precision = self.precision;
        let n = request.n;
        let rank = request.rank;
        let coeff = pairs(parse_records(&request.coefficients, precision)?);
        let shifts: Vec<_> = parse_records(&request.momenta, precision)?
            .chunks_exact(4)
            .map(|v| FourMomentum::from(std::array::from_fn(|k| real(v[k].clone()))))
            .collect();
        let masses = pairs(parse_records(&request.masses, precision)?);
        let smat: Vec<_> = parse_records(&request.s_matrix, precision)?
            .into_iter()
            .map(real)
            .collect();
        let scale = parse_record_at_precision(&request.scale, precision)?;
        if scale <= Float::new(precision) || masses.iter().any(|m| m.im > m.im.zero()) {
            return Err("invalid MP squared scale or mass width".into());
        }
        if !self.entries.contains_key(&(n, rank)) {
            let prototype = Float::new(precision);
            let kernels = SymbolicaAmplitudeKernels::new(n, rank, prototype.clone())
                .map_err(|e| e.to_string())?;
            #[cfg(test)]
            let kernels = match self.study_policy {
                StudyKernelPolicy::Interpreted => kernels,
                StudyKernelPolicy::NativeDenominators => kernels.with_native_denominators(),
                StudyKernelPolicy::FullNative => kernels.with_native_kernels(),
            };
            #[cfg(test)]
            let study_backends = kernels.kernel_backends();
            self.entries.insert(
                (n, rank),
                MpEntry {
                    #[cfg(test)]
                    study_backends,
                    amplitude: kernels.into_amplitude(1).map_err(|e| e.to_string())?,
                    tensor: TensorGenericNumerator::new(rank, 0, &prototype)
                        .map_err(|e| e.to_string())?,
                },
            );
        }
        let entry = self.entries.get_mut(&(n, rank)).unwrap();
        entry
            .tensor
            .set_coefficients(coeff, 1)
            .map_err(|e| e.to_string())?;
        entry.amplitude.reset_batch(1);
        let scales = [real(scale)];
        // Match NinjaRS's generic tensor harness/Ninja auxiliary box scale,
        // keeping the comparison and values in the rescue precision.
        let sampling_scale = smat
            .iter()
            .chain(&masses)
            .map(|value| value.re.norm())
            .fold(Float::with_val(precision, 1), |largest, value| {
                if value > largest { value } else { largest }
            });
        let sampling = [real(sampling_scale)];
        entry
            .amplitude
            .evaluate_batch_with_inputs(
                AmplitudeInput {
                    shifts: &shifts,
                    masses_squared: &masses,
                    scales_squared: &scales,
                    sampling_scales: &sampling,
                    s_matrix: Some(&smat),
                },
                &mut entry.tensor,
                &mut self.low,
                &mut self.high,
                &mut self.masters,
            )
            .map_err(|e| e.to_string())?;
        if entry.amplitude.statuses()[0] != EvaluationStatus::Success {
            return Ok((UNSTABLE, Vec::new()));
        }
        let result = &entry.amplitude.results()[0];
        let mut output = vec![b' '; 8 * RECORD];
        for (index, value) in [0, 1, 2, 4]
            .into_iter()
            .flat_map(|k| [&result[k].re, &result[k].im])
            .enumerate()
        {
            if !value.is_finite() {
                return Ok((UNSTABLE, Vec::new()));
            }
            let text = format!("{value:.39e}");
            if text.len() > RECORD {
                return Err("MP output exceeds ABI record width".into());
            }
            output[index * RECORD..index * RECORD + text.len()].copy_from_slice(text.as_bytes());
        }
        Ok((SUCCESS, output))
    }
}

struct MpRequest {
    context_id: u64,
    n: usize,
    rank: usize,
    coefficients: Vec<u8>,
    s_matrix: Vec<u8>,
    momenta: Vec<u8>,
    masses: Vec<u8>,
    scale: Vec<u8>,
    reply: mpsc::SyncSender<Result<(i32, Vec<u8>)>>,
}
// Symbolica's restricted license grants one active thread per user. Route all
// contexts through one process worker, retaining independent numerical state
// for each context. The last context's destroy joins and releases the worker.
struct MpContext {
    id: u64,
    sender: mpsc::Sender<MpMessage>,
}
enum MpMessage {
    Evaluate(MpRequest),
    Remove(u64, mpsc::SyncSender<()>),
    Shutdown,
}
struct MpWorker {
    sender: mpsc::Sender<MpMessage>,
    thread: Option<std::thread::JoinHandle<()>>,
}
struct MpManager {
    worker: Option<MpWorker>,
    users: usize,
    next_id: u64,
}
static MP_MANAGER: Mutex<MpManager> = Mutex::new(MpManager {
    worker: None,
    users: 0,
    next_id: 0,
});
impl MpContext {
    fn new() -> Result<Self> {
        let mut manager = MP_MANAGER.lock().map_err(|_| "MP manager poisoned")?;
        let id = manager.next_id;
        manager.next_id = id.checked_add(1).ok_or("MP context ID overflow")?;
        if manager.worker.is_none() {
            manager.worker = Some(MpWorker::new()?);
        }
        manager.users += 1;
        Ok(Self {
            id,
            sender: manager.worker.as_ref().unwrap().sender.clone(),
        })
    }
}
impl Drop for MpContext {
    fn drop(&mut self) {
        let mut manager = MP_MANAGER.lock().unwrap_or_else(|e| e.into_inner());
        let (reply, receive) = mpsc::sync_channel(1);
        if self.sender.send(MpMessage::Remove(self.id, reply)).is_ok() {
            let _ = receive.recv();
        }
        manager.users -= 1;
        if manager.users == 0 {
            // Hold the manager lock through join so another creation cannot
            // start a second worker before this one's license permit is freed.
            drop(manager.worker.take());
        }
    }
}
impl Drop for MpWorker {
    fn drop(&mut self) {
        let _ = self.sender.send(MpMessage::Shutdown);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
impl MpWorker {
    fn new() -> Result<Self> {
        let (send, recv) = mpsc::channel::<MpMessage>();
        let thread = std::thread::Builder::new()
            .name("madloop-ninjars-mp".into())
            .stack_size(128 * 1024 * 1024)
            .spawn(move || {
                let mut states = HashMap::new();
                while let Ok(message) = recv.recv() {
                    match message {
                        MpMessage::Evaluate(request) => {
                            let result = catch_unwind(AssertUnwindSafe(|| {
                                if !states.contains_key(&request.context_id) {
                                    states.insert(request.context_id, MpState::new()?);
                                }
                                states
                                    .get_mut(&request.context_id)
                                    .unwrap()
                                    .evaluate(&request)
                            }))
                            .unwrap_or_else(|_| {
                                states.remove(&request.context_id);
                                Err("NinjaRS MP panic contained in rescue worker".into())
                            });
                            let _ = request.reply.send(result);
                        }
                        MpMessage::Remove(id, reply) => {
                            states.remove(&id);
                            let _ = reply.send(());
                        }
                        MpMessage::Shutdown => break,
                    }
                }
            })
            .map_err(|e| format!("cannot start NinjaRS rescue worker: {e}"))?;
        Ok(Self {
            sender: send,
            thread: Some(thread),
        })
    }
}

/// Evaluate actual binary128 inputs supplied as fixed-width ASCII records.
///
/// # Safety
/// Buffers contain the number of 64-byte records specified in ninjars.h.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ninjars_evaluate_mp(
    handle: *mut Context,
    n: i32,
    rank: i32,
    count: usize,
    coefficients: *const u8,
    coefficient_bytes: usize,
    s_matrix: *const u8,
    s_bytes: usize,
    momenta: *const u8,
    p_bytes: usize,
    masses: *const u8,
    m_bytes: usize,
    scale_squared: *const u8,
    scale_bytes: usize,
    output: *mut u8,
    output_bytes: usize,
) -> i32 {
    guard(|| {
        let (n, rank) = dimensions(n, rank, count)?;
        length(
            "MP coefficients",
            coefficient_bytes,
            count
                .checked_mul(2 * RECORD)
                .ok_or("coefficient byte overflow")?,
        )?;
        length("MP S matrix", s_bytes, n * n * RECORD)?;
        length("MP momenta", p_bytes, n * 4 * RECORD)?;
        length("MP masses", m_bytes, n * 2 * RECORD)?;
        length("MP scale", scale_bytes, RECORD)?;
        length("MP output", output_bytes, 8 * RECORD)?;
        validate_output(output, 8 * RECORD)?;
        let context = unsafe { context(handle) }?;
        let (reply, receive) = mpsc::sync_channel(1);
        let mut request = MpRequest {
            context_id: 0,
            n,
            rank,
            coefficients: unsafe {
                read(
                    coefficients,
                    count
                        .checked_mul(2 * RECORD)
                        .ok_or("coefficient byte overflow")?,
                )
            }?
            .to_vec(),
            s_matrix: unsafe { read(s_matrix, n * n * RECORD) }?.to_vec(),
            momenta: unsafe { read(momenta, n * 4 * RECORD) }?.to_vec(),
            masses: unsafe { read(masses, n * 2 * RECORD) }?.to_vec(),
            scale: unsafe { read(scale_squared, RECORD) }?.to_vec(),
            reply,
        };
        let mut worker = context.mp.lock().map_err(|_| "MP context poisoned")?;
        if worker.is_none() {
            *worker = Some(MpContext::new()?);
        }
        let worker = worker.as_ref().unwrap();
        request.context_id = worker.id;
        worker
            .sender
            .send(MpMessage::Evaluate(request))
            .map_err(|_| "rescue worker disconnected")?;
        let (status, values) = receive.recv().map_err(|_| "rescue worker disconnected")??;
        if status == SUCCESS {
            unsafe { write(output, &values) }?;
        }
        Ok(status)
    })
}

/// Copy the last calling-thread error, NUL terminated; return required bytes
/// including the terminator. Passing NULL/zero queries the required capacity.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ninjars_last_error(output: *mut u8, capacity: usize) -> usize {
    LAST_ERROR.with(|slot| {
        let message = slot.borrow();
        if !output.is_null() && capacity > 0 {
            let n = message.len().min(capacity - 1);
            unsafe {
                std::ptr::copy_nonoverlapping(message.as_ptr(), output, n);
                output.add(n).write(0);
            }
        }
        message.len() + 1
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ninjars::fast::tensor_layout::{monomial_index, ncoeff};
    use ninjars::{ExpansionKind, ExpansionRequest, Numerator, NumeratorRequest, SampleId};

    struct TestContext(*mut Context);
    impl TestContext {
        fn new() -> Self {
            let mut context = std::ptr::null_mut();
            assert_eq!(
                unsafe { ninjars_context_create(ABI_VERSION, &mut context) },
                SUCCESS
            );
            Self(context)
        }
    }
    impl Drop for TestContext {
        fn drop(&mut self) {
            assert_eq!(unsafe { ninjars_context_destroy(self.0) }, SUCCESS);
        }
    }

    fn dp(context: &TestContext, c: C64, mass: C64, scale: f64) -> (i32, [C64; 3]) {
        let mut out = [C64::real(991.); 3];
        let mut rational = C64::real(992.);
        let status = unsafe {
            ninjars_evaluate_dp(
                context.0,
                1,
                0,
                1,
                &c,
                &[0.][0],
                1,
                &[0.; 4][0],
                4,
                &mass,
                1,
                scale,
                out.as_mut_ptr(),
                out.len(),
                &mut rational,
                1,
            )
        };
        (status, out)
    }

    #[test]
    fn tadpole_rebinds_coefficients_masses_and_scale() {
        let context = TestContext::new();
        let (status, result) = dp(&context, C64::real(1.), C64::real(4.), 4.);
        assert_eq!(status, SUCCESS);
        assert!((result[0].re - 4.).abs() < 1e-12);
        assert_eq!(result[1], C64::real(4.));
        assert_eq!(result[2], C64::ZERO);
        let (status, result) = dp(&context, C64::new(2., -1.), C64::real(9.), 1.);
        assert_eq!(status, SUCCESS);
        assert_eq!(result[1], C64::new(18., -9.));
        assert!((result[0].re - 18. * (1. - 9f64.ln())).abs() < 1e-12);
        let (status, result) = dp(&context, C64::real(1.), C64::new(4., -0.5), 4.);
        assert_eq!(status, SUCCESS);
        assert_eq!(result[1], C64::new(4., -0.5));
    }

    #[test]
    fn invalid_input_preserves_outputs_and_reports_error() {
        let context = TestContext::new();
        let (status, result) = dp(&context, C64::real(1.), C64::new(4., 1.), 4.);
        assert_eq!(status, ERROR);
        assert_eq!(result, [C64::real(991.); 3]);
        let mut error = [0; 128];
        assert!(unsafe { ninjars_last_error(error.as_mut_ptr(), error.len()) } > 1);
        assert!(String::from_utf8_lossy(&error).contains("mass"));
        assert!(dimensions(0, 0, 1).is_err());
        assert!(dimensions(1, 4, 70).is_err());
        assert!(dimensions(1, 0, 2).is_err());
    }

    #[test]
    fn version_and_malformed_outputs_are_transactional() {
        let mut invalid = std::ptr::null_mut();
        assert_eq!(unsafe { ninjars_context_create(1, &mut invalid) }, ERROR);
        assert_eq!(
            unsafe { ninjars_context_create(ABI_VERSION + 1, &mut invalid) },
            ERROR
        );
        assert!(invalid.is_null());
        assert_eq!(ninjars_abi_version(), ABI_VERSION);
        let context = TestContext::new();
        let mut out = [C64::real(991.); 3];
        let mut storage = [0u64; 4];
        // A deliberately misaligned second output must be rejected before the
        // otherwise valid Laurent output receives any data.
        let bad_rational = unsafe { storage.as_mut_ptr().cast::<u8>().add(1).cast::<C64>() };
        let call = |out: *mut C64, rational| unsafe {
            ninjars_evaluate_dp(
                context.0,
                1,
                0,
                1,
                &C64::ONE,
                &[0.][0],
                1,
                &[0.; 4][0],
                4,
                &C64::real(4.),
                1,
                4.,
                out,
                3,
                rational,
                1,
            )
        };
        assert_eq!(call(out.as_mut_ptr(), bad_rational), ERROR);
        assert_eq!(out, [C64::real(991.); 3]);
        assert_eq!(call(out.as_mut_ptr(), out.as_mut_ptr()), ERROR);
        assert_eq!(out, [C64::real(991.); 3]);
    }

    #[test]
    fn every_short_buffer_is_rejected_before_access_or_output() {
        let context = TestContext::new();
        let mut output = [C64::real(991.); 3];
        let mut rational = C64::real(992.);
        let coefficient = C64::ONE;
        let dp_matrix = [0.];
        let dp_momenta = [0.; 4];
        let mass = C64::real(4.);
        for short in 0..6 {
            // A zero-length buffer has no readable/writable elements. Use
            // NULL for that buffer to verify length validation comes first.
            let mut lengths = [1, 1, 4, 1, 3, 1];
            lengths[short] = 0;
            let status = unsafe {
                ninjars_evaluate_dp(
                    context.0,
                    1,
                    0,
                    lengths[0],
                    if short == 0 {
                        std::ptr::null()
                    } else {
                        &coefficient
                    },
                    if short == 1 {
                        std::ptr::null()
                    } else {
                        dp_matrix.as_ptr()
                    },
                    lengths[1],
                    if short == 2 {
                        std::ptr::null()
                    } else {
                        dp_momenta.as_ptr()
                    },
                    lengths[2],
                    if short == 3 { std::ptr::null() } else { &mass },
                    lengths[3],
                    4.,
                    if short == 4 {
                        std::ptr::null_mut()
                    } else {
                        output.as_mut_ptr()
                    },
                    lengths[4],
                    if short == 5 {
                        std::ptr::null_mut()
                    } else {
                        &mut rational
                    },
                    lengths[5],
                )
            };
            assert_eq!(status, ERROR);
            assert_eq!(output, [C64::real(991.); 3]);
            assert_eq!(rational, C64::real(992.));
            let mut message = [0; 256];
            unsafe {
                ninjars_last_error(message.as_mut_ptr(), message.len());
            }
            assert!(String::from_utf8_lossy(&message).contains("length"));
        }
        let coefficients = [record("1"), record("0")].concat();
        let matrix = record("0");
        let momenta = matrix.repeat(4);
        let masses = [record("4"), record("0")].concat();
        let scale = record("4");
        let mut output = [b'!'; 8 * RECORD];
        for short in 0..6 {
            let mut lengths = [
                2 * RECORD,
                RECORD,
                4 * RECORD,
                2 * RECORD,
                RECORD,
                8 * RECORD,
            ];
            // A nonempty truncated ASCII record must also fail immediately.
            lengths[short] -= 1;
            let status = unsafe {
                ninjars_evaluate_mp(
                    context.0,
                    1,
                    0,
                    1,
                    coefficients.as_ptr(),
                    lengths[0],
                    matrix.as_ptr(),
                    lengths[1],
                    momenta.as_ptr(),
                    lengths[2],
                    masses.as_ptr(),
                    lengths[3],
                    scale.as_ptr(),
                    lengths[4],
                    output.as_mut_ptr(),
                    lengths[5],
                )
            };
            assert_eq!(status, ERROR);
            assert_eq!(output, [b'!'; 8 * RECORD]);
            let mut message = [0; 256];
            unsafe {
                ninjars_last_error(message.as_mut_ptr(), message.len());
            }
            assert!(String::from_utf8_lossy(&message).contains("length"));
        }
        assert!(unsafe { &*context.0 }.mp.lock().unwrap().is_none());
    }

    fn record(text: &str) -> [u8; RECORD] {
        assert!(text.len() <= RECORD);
        let mut data = [b' '; RECORD];
        data[..text.len()].copy_from_slice(text.as_bytes());
        data
    }

    #[test]
    fn mp_transport_preserves_binary128_information() {
        let input = Float::with_val(113, 1) + Float::with_val(113, 2).pow(100).inv();
        let parsed = parse_record(&record(&format!("{input:.39e}"))).unwrap();
        assert_eq!(parsed.to_rational(), input.to_rational());
        assert!(parsed > Float::with_val(MP_BITS, 1));
        assert!(parse_record(&record("not-a-number")).is_err());
    }

    #[test]
    fn numerical_tensor_evaluation_and_laurent_expansion() {
        let prototype = Float::new(MP_BITS);
        let z = |x| real(Float::with_val(MP_BITS, x));
        let mut tensor = TensorGenericNumerator::new(2, 0, &prototype).unwrap();
        let mut coefficients = vec![z(0); ncoeff(2)];
        coefficients[0] = z(3);
        coefficients[monomial_index([2, 0, 0, 0])] = z(1);
        coefficients[monomial_index([0, 1, 0, 0])] = z(2);
        tensor.set_coefficients(coefficients, 1).unwrap();
        let q = FourMomentum::from([z(2), z(3), z(4), z(5)]);
        let mut out = [z(0)];
        tensor
            .evaluate_batch(
                &[NumeratorRequest {
                    id: SampleId { lane: 0, sample: 0 },
                    q: &q,
                    mu_squared: &z(0),
                    partition: &[],
                }],
                &mut out,
            )
            .unwrap();
        assert_eq!(out[0], z(13));
        // q = a + t e3 + (beta+mu²)e4/t: q0=1+2t, q1=3.
        // N = 4t²+4t+10, independent of mu².
        let vectors = [
            FourMomentum::from([z(1), z(3), z(0), z(0)]),
            FourMomentum::from([z(2), z(0), z(0), z(0)]),
            FourMomentum::from([z(0), z(0), z(0), z(1)]),
        ];
        let mut out = [z(0), z(0), z(0), z(0)];
        tensor
            .expansion_batch(
                &ExpansionKind::T3 {
                    powers: vec![(2, 0), (1, 0), (0, 0), (0, 1)],
                },
                &[ExpansionRequest {
                    id: SampleId { lane: 0, sample: 0 },
                    partition: &[],
                    vectors: &vectors,
                    parameters: &[z(1)],
                }],
                &mut out,
            )
            .unwrap();
        assert_eq!(out, [z(4), z(4), z(10), z(0)]);
    }

    #[test]
    fn complete_mp_tadpole_retains_input_precision() {
        let context = TestContext::new();
        let one = Float::with_val(113, 1);
        let coefficient = one + Float::with_val(113, 2).pow(100).inv();
        let mut coeff = Vec::new();
        coeff.extend(record(&format!("{coefficient:.39e}")));
        coeff.extend(record("0"));
        let zero = record("0");
        let momenta: Vec<_> = zero.repeat(4);
        let masses: Vec<_> = [record("4"), zero].concat();
        let mut out = [b' '; 8 * RECORD];
        let status = unsafe {
            ninjars_evaluate_mp(
                context.0,
                1,
                0,
                1,
                coeff.as_ptr(),
                coeff.len(),
                zero.as_ptr(),
                zero.len(),
                momenta.as_ptr(),
                momenta.len(),
                masses.as_ptr(),
                masses.len(),
                record("4").as_ptr(),
                RECORD,
                out.as_mut_ptr(),
                out.len(),
            )
        };
        if status != SUCCESS {
            let mut error = [0; 1024];
            unsafe {
                ninjars_last_error(error.as_mut_ptr(), error.len());
            }
            panic!("MP status {status}: {}", String::from_utf8_lossy(&error));
        }
        let pole = parse_record(&out[2 * RECORD..3 * RECORD]).unwrap();
        assert_eq!(
            pole.to_rational(),
            (coefficient * Float::with_val(113, 4)).to_rational()
        );
    }

    #[test]
    fn mp_bubbles_cover_massless_orderings_and_context_rebinding() {
        fn run(context: &TestContext, m0: f64, m1: f64, scale: f64, coefficient: f64) -> [C64; 3] {
            let records = |values: &[f64]| {
                values
                    .iter()
                    .flat_map(|x| record(&format!("{x:.39e}")))
                    .collect::<Vec<_>>()
            };
            let coeff = records(&[coefficient, 0.]);
            let matrix = records(&[0., -4., -4., 0.]);
            let momenta = records(&[0., 0., 0., 0., 0., 2., 0., 0.]);
            let masses = records(&[m0, 0., m1, 0.]);
            let scale = records(&[scale]);
            let mut output = [b' '; 8 * RECORD];
            let status = unsafe {
                ninjars_evaluate_mp(
                    context.0,
                    2,
                    0,
                    1,
                    coeff.as_ptr(),
                    coeff.len(),
                    matrix.as_ptr(),
                    matrix.len(),
                    momenta.as_ptr(),
                    momenta.len(),
                    masses.as_ptr(),
                    masses.len(),
                    scale.as_ptr(),
                    scale.len(),
                    output.as_mut_ptr(),
                    output.len(),
                )
            };
            if status != SUCCESS {
                let mut error = [0; 1024];
                unsafe {
                    ninjars_last_error(error.as_mut_ptr(), error.len());
                }
                panic!(
                    "MP bubble status {status}: {}",
                    String::from_utf8_lossy(&error)
                );
            }
            std::array::from_fn(|i| {
                let component = |j| {
                    std::str::from_utf8(&output[j * RECORD..(j + 1) * RECORD])
                        .unwrap()
                        .trim()
                        .parse()
                        .unwrap()
                };
                C64::new(component(2 * i), component(2 * i + 1))
            })
        }
        let retained = TestContext::new();
        for (m0, m1, scale, coefficient) in [(0., 4., 1., 1.), (4., 0., 9., 2.), (0., 0., 4., -3.)]
        {
            let actual = run(&retained, m0, m1, scale, coefficient);
            let fresh = run(&TestContext::new(), m0, m1, scale, coefficient);
            assert_eq!(actual, fresh);
            let mut dp = [C64::ZERO; 3];
            let mut rational = C64::ZERO;
            assert_eq!(
                unsafe {
                    ninjars_evaluate_dp(
                        retained.0,
                        2,
                        0,
                        1,
                        &C64::real(coefficient),
                        [0., -4., -4., 0.].as_ptr(),
                        4,
                        [0., 0., 0., 0., 0., 2., 0., 0.].as_ptr(),
                        8,
                        [C64::real(m0), C64::real(m1)].as_ptr(),
                        2,
                        scale,
                        dp.as_mut_ptr(),
                        dp.len(),
                        &mut rational,
                        1,
                    )
                },
                SUCCESS
            );
            for i in 0..3 {
                assert!((actual[i] - dp[i]).abs() <= 1e-11 * dp[i].abs().max(1.));
            }
            assert!((actual[1] - C64::real(coefficient)).abs() < 1e-12);
            assert!(actual[2].abs() < 1e-12);
        }
    }

    #[test]
    fn complex_mass_pentagon_converges_at_binary128_output_precision() {
        // The public production ABI remains 256-bit. This independent worker
        // repeats a retained physical rank-five fixture at 384 bits, with
        // complex masses and numerator information below binary64 precision.
        std::thread::Builder::new()
            .stack_size(128 * 1024 * 1024)
            .spawn(|| {
                let case: serde_json::Value = serde_json::from_str(include_str!(
                    "../tests/data/cases/madloop-gg-hhh-top-pentagon-26.json"
                ))
                .unwrap();
                let lane = &case["lanes"][2];
                let numbers = |key: &str| {
                    lane[key]
                        .as_array()
                        .unwrap()
                        .iter()
                        .flat_map(|row| row.as_array().unwrap().iter())
                        .map(|value| value.as_f64().unwrap())
                        .collect::<Vec<_>>()
                };
                let tensors = numbers("tensor");
                let matrix = numbers("s_matrix");
                let momenta = numbers("shifts");
                let mut masses = numbers("masses_squared");
                for mass in masses.chunks_exact_mut(2) {
                    mass[1] = -299.29;
                }
                let scale = lane["mu_squared"].as_f64().unwrap();
                let records = |values: &[f64]| {
                    values
                        .iter()
                        .flat_map(|x| record(&format!("{x:.39e}")))
                        .collect::<Vec<_>>()
                };
                let multiplier = Float::with_val(113, 1) + Float::with_val(113, 2).pow(100).inv();
                let coefficients: Vec<_> = tensors
                    .iter()
                    .flat_map(|x| {
                        let value = Float::with_val(113, *x) * &multiplier;
                        record(&format!("{value:.39e}"))
                    })
                    .collect();
                let (reply, _) = mpsc::sync_channel(1);
                let request = MpRequest {
                    context_id: 0,
                    n: 5,
                    rank: 5,
                    coefficients,
                    s_matrix: records(&matrix),
                    momenta: records(&momenta),
                    masses: records(&masses),
                    scale: records(&[scale]),
                    reply,
                };
                let mut state256 = MpState::with_precision(256).unwrap();
                let (status256, output256) = state256.evaluate(&request).unwrap();
                let mut state384 = MpState::with_precision(384).unwrap();
                let (status384, output384) = state384.evaluate(&request).unwrap();
                assert_eq!(status256, SUCCESS);
                assert_eq!(status384, SUCCESS);
                let decode = |bytes: &[u8]| {
                    bytes
                        .chunks_exact(RECORD)
                        .map(|record| {
                            Float::parse(std::str::from_utf8(record).unwrap().trim(), Some(384))
                                .unwrap()
                        })
                        .collect::<Vec<_>>()
                };
                let values256 = decode(&output256);
                let values384 = decode(&output384);
                let tolerance = Float::parse("1e-31", Some(384)).unwrap();
                let normalization = values384.iter().fold(Float::new(384), |largest, value| {
                    let magnitude = value.norm();
                    if magnitude > largest {
                        magnitude
                    } else {
                        largest
                    }
                });
                assert!(normalization > Float::new(384));
                // The split is basis dependent, but each engine must retain
                // its own complete finite = cut-constructible + rational.
                for state in [&state256, &state384] {
                    let raw = &state.entries[&(5, 5)].amplitude.results()[0];
                    let defect = raw[0].clone() - &raw[3] - &raw[4];
                    let bound = Float::parse("1e-60", Some(384)).unwrap() * &normalization;
                    assert!(defect.re.norm() <= bound && defect.im.norm() <= bound);
                }
                let mut largest_error = Float::new(384);
                for (a, b) in values256.iter().zip(&values384) {
                    let error = (a.clone() - b).norm();
                    assert!(
                        error <= tolerance.clone() * &normalization,
                        "256/384: {a} vs {b}"
                    );
                    if error > largest_error {
                        largest_error = error;
                    }
                }
                let c64 = |values: &[f64]| {
                    values
                        .chunks_exact(2)
                        .map(|z| C64::new(z[0], z[1]))
                        .collect::<Vec<_>>()
                };
                let (status, dp) = DpState::new()
                    .unwrap()
                    .evaluate(
                        5,
                        5,
                        &c64(&tensors),
                        &matrix,
                        &momenta,
                        &c64(&masses),
                        scale,
                    )
                    .unwrap();
                assert_eq!(status, SUCCESS);
                let mut dp_error: f64 = 0.;
                for (index, value) in dp.laurent.into_iter().chain([dp.rational]).enumerate() {
                    let to_f64 = |number: &Float| format!("{number:.17e}").parse::<f64>().unwrap();
                    let expected = C64::new(
                        to_f64(&values384[2 * index]),
                        to_f64(&values384[2 * index + 1]),
                    );
                    assert!((value - expected).abs() <= 1e-10 * expected.abs().max(1.));
                    dp_error = dp_error.max((value - expected).abs() / expected.abs().max(1.));
                }
                let strings = |data: &[u8]| {
                    data.chunks_exact(RECORD).map(|record| {
                        std::str::from_utf8(record).unwrap().trim().to_owned()
                    }).collect::<Vec<_>>()
                };
                println!(
                    "NINJARS_CONVERGENCE {}",
                    serde_json::json!({
                        "fixture": "cases/madloop-gg-hhh-top-pentagon-26.json",
                        "lane": lane["id"],
                        "modification": "Each squared mass gets Im=-299.29 as binary64 promoted to binary128; tensor multiplied by 1+2^-100 in binary128.",
                        "input": {
                            "n": 5, "rank": 5, "coefficients": strings(&request.coefficients),
                            "s_matrix": strings(&request.s_matrix), "momenta": strings(&request.momenta),
                            "masses_squared": strings(&request.masses), "scale_squared": strings(&request.scale),
                        },
                        "production_bits": 256, "reference_bits": 384,
                        "result_256": strings(&output256), "result_384": strings(&output384),
                        "convergence_max_scaled_error": format!("{:.39e}", largest_error / normalization),
                        "convergence_tolerance": "1e-31",
                        "finite_split_identity_tolerance": "1e-60 times largest output component",
                        "dp_max_scaled_error": dp_error, "dp_tolerance": 1e-10,
                        "passed": true
                    })
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }
}

#[cfg(test)]
mod retry_tests;

#[cfg(test)]
mod guarded_tests;

#[cfg(test)]
mod transport_contract_tests;
#[cfg(test)]
mod transport_replay_tests;

#[cfg(test)]
mod native_study_tests;
