// SPDX-License-Identifier: 0BSD

//! Running Direct-Arena recurrence plans on a GPU (cargo feature `pacman`).
//!
//! With `PACMAN_RECURRENCE_RUNNER=<librecurrence.so>` (built by `pacman
//! recurrence-runner`), every multi-point plan run goes to the library's one
//! kernel, which gets the plan's rows as 64-bit words (their layout is in
//! `recurrence_runner.cu`) and runs them all in one launch, one thread per point.
//!
//! - Source rows still run here, on the CPU; their currents (stage 0) are
//!   copied to the device, with the momentum forms, scales and factors.
//! - The current arena stays on the device. Only the amplitude planes that the
//!   closures add to go there and come back; `PACMAN_RECURRENCE_WHOLE=1` also
//!   copies every current plane back (for checks).
//! - The rows run as authenticated rows, as RustiCol does while it observes
//!   currents: no interaction rewrites, no fanout.
//! - `PACMAN_RECURRENCE_DUMP=FILE` writes the first call's words and tables
//!   (for benchmarks without RustiCol): magic `PACRWRD1`, then little-endian
//!   u32 word count, 8 width bytes, the words, u32 scale count, the scales
//!   (re, then im), u32 factor count, the factors (re, then im), u32 current,
//!   amplitude and momentum plane counts.
//! - A row group the kernel lacks (symjit executors, certified reuse) leaves
//!   the whole plan run to the CPU, reported once on standard error.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use super::*;

const RUNNER: &str = "PACMAN_RECURRENCE_RUNNER";
const TILE_POINTS: &str = "RUSTICOL_TILE_POINTS";

/// With `PACMAN_RECURRENCE_RUNNER` set, `RUSTICOL_TILE_POINTS=N` makes the recurrence tile N
/// points instead of the cache-sized one (at most the plan's tile, 64 for Z+6g): the GPU wants
/// many points per launch, and its arena is not the CPU's cache. Read when a plan is loaded.
pub(super) fn gpu_tile_points() -> RusticolResult<Option<u32>> {
    if std::env::var_os(RUNNER).is_none_or(|v| v.is_empty()) {
        return Ok(None);
    }
    let Some(value) = std::env::var_os(TILE_POINTS).filter(|v| !v.is_empty()) else {
        return Ok(None);
    };
    match value.to_str().and_then(|v| v.parse::<u32>().ok()) {
        Some(points) if points > 0 => Ok(Some(points)),
        _ => Err(RusticolError::invalid_argument(format!(
            "{TILE_POINTS}={}: expected a positive number of points",
            value.to_string_lossy()
        ))),
    }
}
const WHOLE: &str = "PACMAN_RECURRENCE_WHOLE";
const DUMP: &str = "PACMAN_RECURRENCE_DUMP";
const ABI_VERSION: u32 = 1;
const RUN_SYMBOL: &[u8] = b"pacman_recurrence_v1_run\0";

const OP_CLEAR_CURRENTS: u64 = 1;
const OP_CLOSURE: u64 = 11;
/// widths: current plane, amplitude plane, momentum form, factor, slot, count
const W_P: usize = 0;
const W_A: usize = 1;
const W_F: usize = 2;
const W_X: usize = 3;
const W_S: usize = 4;
const W_K: usize = 5;
const COUNT_BITS: u8 = 4;

/// `pacman_recurrence_call` of `recurrence_runner.cu`.
#[repr(C)]
struct RunnerCall {
    abi_version: u32,
    struct_size: u32,
    words: *const u64,
    words_key: u64,
    word_count: u32,
    width: [u8; 8],
    current_re: *mut f64,
    current_im: *mut f64,
    amplitude_re: *mut f64,
    amplitude_im: *mut f64,
    momenta: *const f64,
    scale_re: *const f64,
    scale_im: *const f64,
    factor_re: *const f64,
    factor_im: *const f64,
    upload_current: *const u32,
    upload_amplitude: *const u32,
    download_amplitude: *const u32,
    download_current: *const u32,
    current_planes: u32,
    amplitude_planes: u32,
    momentum_planes: u32,
    scale_count: u32,
    factor_count: u32,
    upload_current_runs: u32,
    upload_amplitude_runs: u32,
    download_amplitude_runs: u32,
    download_current_runs: u32,
    stride: u32,
    points: u32,
}

type RunFunction = unsafe extern "C" fn(*const RunnerCall) -> i32;

struct Runner {
    run: RunFunction,
    whole: bool,
    _library: libloading::Library,
}

/// The library of `PACMAN_RECURRENCE_RUNNER` (`None` if unset); a library that
/// cannot be loaded is an error at the first plan run.
fn runner() -> Option<&'static Result<Runner, String>> {
    static RUNNER_LIBRARY: OnceLock<Option<Result<Runner, String>>> = OnceLock::new();
    RUNNER_LIBRARY
        .get_or_init(|| {
            let path = std::env::var_os(RUNNER).filter(|value| !value.is_empty())?;
            let load = || -> Result<Runner, String> {
                // SAFETY: the library is a pacman recurrence runner; its
                // initializers only register an exit handler
                let library = unsafe { libloading::Library::new(&path) }
                    .map_err(|error| format!("{RUNNER}: {}: {error}", path.to_string_lossy()))?;
                let run = unsafe { library.get::<RunFunction>(RUN_SYMBOL) }
                    .map_err(|error| format!("{RUNNER}: {}: {error}", path.to_string_lossy()))?;
                let run = *run;
                eprintln!("pacman: {RUNNER}: {}", path.to_string_lossy());
                Ok(Runner {
                    run,
                    whole: std::env::var_os(WHOLE).is_some_and(|value| value != "0"),
                    _library: library,
                })
            };
            Some(load())
        })
        .as_ref()
}

/// One plan run, encoded once.
struct Encoded {
    /// the device's cache key of `words`
    key: u64,
    words: Box<[u64]>,
    width: [u8; 8],
    /// per slot, the contribution executor whose scale it holds
    slots: Box<[u32]>,
    /// (first plane, count) runs
    source_currents: Box<[u32]>,
    amplitudes: Box<[u32]>,
}

/// What makes a plan run's words: the plan, the row groups and the clears.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Key {
    plan: usize,
    row_groups: usize,
    row_group_count: usize,
    selected_sector_id: Option<u32>,
    clear_stages: Option<usize>,
    requires_stage_clear: bool,
    prepared_clears: bool,
}

type Cache = HashMap<Key, Result<Arc<Encoded>, String>>;

fn cache() -> &'static Mutex<Cache> {
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Runs the plan on the GPU when `PACMAN_RECURRENCE_RUNNER` is set and the
/// kernel has every row group; `None` leaves it to the CPU.
#[allow(clippy::too_many_arguments)]
pub(super) fn try_execute<const PREPARED_CLEARS: bool>(
    plan: &DirectRecurrencePlan,
    row_groups: &[DirectRowGroupDescriptor],
    selected_sector_id: Option<u32>,
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
    requires_stage_clear: bool,
    current_clear_stages: Option<&[DirectStageClearRanges]>,
) -> Option<RusticolResult<()>> {
    let runner = match runner()? {
        Ok(runner) => runner,
        Err(message) => return Some(Err(RusticolError::evaluation(message.clone()))),
    };
    let key = Key {
        plan: std::ptr::from_ref(plan) as usize,
        row_groups: row_groups.as_ptr() as usize,
        row_group_count: row_groups.len(),
        selected_sector_id,
        clear_stages: current_clear_stages.map(|stages| stages.as_ptr() as usize),
        requires_stage_clear,
        prepared_clears: PREPARED_CLEARS,
    };
    let encoded = {
        let mut cache = cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let next_key = cache.len() as u64 + 1;
        cache
            .entry(key)
            .or_insert_with(|| {
                let encoded = encode::<PREPARED_CLEARS>(
                    plan,
                    row_groups,
                    selected_sector_id,
                    executors,
                    workspace,
                    requires_stage_clear,
                    current_clear_stages,
                    next_key,
                );
                if let Err(message) = &encoded {
                    eprintln!("pacman: {RUNNER}: this plan run stays on the CPU: {message}");
                }
                encoded.map(Arc::new)
            })
            .clone()
    };
    let encoded = encoded.ok()?;
    Some(execute(
        runner,
        &encoded,
        plan,
        row_groups,
        executors,
        workspace,
        point_count,
    ))
}

fn bits(value: usize) -> u8 {
    (usize::BITS - value.max(1).leading_zeros()) as u8
}

/// (first, count) runs of the sorted, distinct planes
fn runs(planes: impl IntoIterator<Item = u32>) -> Box<[u32]> {
    let mut planes: Vec<u32> = planes.into_iter().collect();
    planes.sort_unstable();
    planes.dedup();
    let mut runs: Vec<u32> = Vec::new();
    for plane in planes {
        match runs.len() {
            n if n >= 2 && runs[n - 2] + runs[n - 1] == plane => runs[n - 1] += 1,
            _ => runs.extend([plane, 1]),
        }
    }
    runs.into_boxed_slice()
}

struct Word {
    value: u64,
    used: u32,
}

impl Word {
    fn new(op: u64) -> Self {
        Word { value: op, used: 4 }
    }

    fn put(mut self, value: u64, width: u8) -> Result<Self, String> {
        if width < 64 && value >> width != 0 {
            return Err(format!("{value} does not fit in {width} bits"));
        }
        if self.used + u32::from(width) > 64 {
            return Err(format!(
                "a row needs more than 64 bits (op {})",
                self.value & 15
            ));
        }
        self.value |= value << self.used;
        self.used += u32::from(width);
        Ok(self)
    }
}

#[allow(clippy::too_many_arguments)]
fn encode<const PREPARED_CLEARS: bool>(
    plan: &DirectRecurrencePlan,
    row_groups: &[DirectRowGroupDescriptor],
    selected_sector_id: Option<u32>,
    executors: &DirectExecutorCatalog,
    workspace: &DirectWorkspace<'_>,
    requires_stage_clear: bool,
    current_clear_stages: Option<&[DirectStageClearRanges]>,
    key: u64,
) -> Result<Encoded, String> {
    let stride = workspace.point_stride as usize;
    let current_planes = workspace.current_re.len() / stride;
    let amplitude_planes = workspace.amplitude_re.len() / stride;
    // the executors: slots for the contributions' scales, ops for all
    let mut slots: Vec<u32> = Vec::new();
    let mut ops = Vec::with_capacity(row_groups.len());
    let mut seen_rows = false;
    for descriptor in row_groups {
        let id = descriptor.direct_executor_id;
        if descriptor.role == DirectExecutorRole::Source {
            if seen_rows {
                return Err("a source row group follows other rows".into());
            }
            ops.push(0);
            continue;
        }
        seen_rows = true;
        if id == DIRECT_NONE_U32 {
            return Err("certified-reuse contributions are not supported".into());
        }
        let handle = executors
            .require(id, descriptor.role)
            .map_err(|error| error.to_string())?;
        let op = match handle {
            DirectExecutorHandle::Contribution { call, .. } => {
                let op = crate::engine::pacman_contribution_op(call);
                if op.is_some() && !slots.contains(&id) {
                    slots.push(id);
                }
                op
            }
            DirectExecutorHandle::Finalization { call, .. } => {
                crate::engine::pacman_finalization_op(call)
            }
            DirectExecutorHandle::Closure { call, .. } => {
                let closure_reduce: DirectClosureExecutor =
                    crate::engine::execute_closure_reduce_rows;
                (call as usize == closure_reduce as usize).then_some(OP_CLOSURE as u8)
            }
            DirectExecutorHandle::Source { .. } => None,
        };
        ops.push(
            op.ok_or_else(|| format!("executor {id} ({:?}) has no GPU kernel", descriptor.role))?,
        );
    }
    let mut width = [0u8; 8];
    width[W_P] = bits(current_planes);
    width[W_A] = bits(amplitude_planes.saturating_sub(1));
    width[W_F] = bits((workspace.momentum_form_count as usize).saturating_sub(1));
    width[W_X] = bits(workspace.factors_re.len().saturating_sub(1));
    width[W_S] = bits(slots.len().saturating_sub(1));
    width[W_K] = COUNT_BITS;
    let none_as_zero = |value: u32| if value == DIRECT_NONE_U32 { 0 } else { value };

    let mut words = Vec::new();
    let mut amplitudes = Vec::new();
    let clear = |words: &mut Vec<u64>, base: u32, count: u32| -> Result<(), String> {
        if count > 0 {
            let word = Word::new(OP_CLEAR_CURRENTS)
                .put(base.into(), width[W_P])?
                .put(count.into(), width[W_P])?;
            words.push(word.value);
        }
        Ok(())
    };
    let mut initialized_stage = None;
    for (descriptor, &op) in row_groups.iter().zip(&ops) {
        if descriptor.role == DirectExecutorRole::Source {
            continue;
        }
        // as `execute_direct_plan_with_clears_impl`
        if requires_stage_clear
            && descriptor.role == DirectExecutorRole::Contribution
            && initialized_stage != Some(descriptor.stage)
        {
            if PREPARED_CLEARS {
                let stages = current_clear_stages
                    .ok_or("selected recurrence has no prepared stage clears")?;
                if let Ok(index) = stages.binary_search_by_key(&descriptor.stage, |s| s.stage) {
                    for range in stages[index].ranges.iter() {
                        clear(&mut words, range.component_base, range.component_count)?;
                    }
                }
            } else {
                for current in plan.currents().iter().filter(|current| {
                    current.node_kind == DirectNodeKind::Current
                        && current.stage == descriptor.stage
                }) {
                    if let Some(sector_id) = selected_sector_id
                        && !plan
                            .selector_domain_contains(current.selector_domain_id, sector_id)
                            .map_err(|error| error.to_string())?
                    {
                        continue;
                    }
                    clear(
                        &mut words,
                        current.component_base,
                        current.component_count.into(),
                    )?;
                }
            }
            initialized_stage = Some(descriptor.stage);
        }
        let start = descriptor.row_start as usize;
        let range = start..start + descriptor.row_count as usize;
        let op = u64::from(op);
        match descriptor.role {
            DirectExecutorRole::Contribution => {
                let slot = slots
                    .iter()
                    .position(|&id| id == descriptor.direct_executor_id)
                    .expect("every contribution executor has a slot");
                let rows = plan
                    .contributions()
                    .get(range)
                    .ok_or("a contribution row group is out of bounds")?;
                for row in rows {
                    if row.parent1_component_base_or_sentinel == DIRECT_NONE_U32 {
                        return Err("a contribution row has one parent".into());
                    }
                    let initialize =
                        row.flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION != 0;
                    let word = Word::new(op)
                        .put(initialize.into(), 1)?
                        .put(slot as u64, width[W_S])?
                        .put(row.parent0_component_base.into(), width[W_P])?
                        .put(row.parent1_component_base_or_sentinel.into(), width[W_P])?
                        .put(row.destination_component_base.into(), width[W_P])?
                        .put(
                            none_as_zero(row.parent0_momentum_form_id).into(),
                            width[W_F],
                        )?
                        .put(
                            none_as_zero(row.parent1_momentum_form_id_or_sentinel).into(),
                            width[W_F],
                        )?
                        .put(row.exact_factor_id.into(), width[W_X])?;
                    words.push(word.value);
                }
            }
            DirectExecutorRole::Finalization => {
                let rows = plan
                    .finalizations()
                    .get(range)
                    .ok_or("a finalization row group is out of bounds")?;
                for row in rows {
                    let word = Word::new(op).put(row.component_base.into(), width[W_P])?;
                    let word = if op == 10 {
                        word.put(row.component_count.into(), width[W_K])?
                    } else {
                        word.put(none_as_zero(row.momentum_form_id).into(), width[W_F])?
                    };
                    words.push(word.put(row.exact_factor_id.into(), width[W_X])?.value);
                }
            }
            DirectExecutorRole::Closure => {
                let rows = plan
                    .closures()
                    .get(range)
                    .ok_or("a closure row group is out of bounds")?;
                for row in rows {
                    if row.parent1_component_base_or_sentinel == DIRECT_NONE_U32 {
                        return Err("a closure row has one parent".into());
                    }
                    let word = Word::new(op)
                        .put(row.parent0_component_base.into(), width[W_P])?
                        .put(row.parent1_component_base_or_sentinel.into(), width[W_P])?
                        .put(row.amplitude_destination_id.into(), width[W_A])?
                        .put(row.exact_factor_id.into(), width[W_X])?
                        .put(row.component_factor_start.into(), width[W_X])?
                        .put(row.component_count.into(), width[W_K])?;
                    words.push(word.value);
                    amplitudes.push(row.amplitude_destination_id);
                }
            }
            DirectExecutorRole::Source => unreachable!("skipped above"),
        }
    }
    // the planes the sources write: the stage-0 currents
    let source_currents = runs(
        plan.currents()
            .iter()
            .filter(|current| current.stage == 0)
            .flat_map(|current| {
                current.component_base..current.component_base + u32::from(current.component_count)
            }),
    );
    if words.len() > u32::MAX as usize {
        return Err("too many rows".into());
    }
    Ok(Encoded {
        key,
        words: words.into_boxed_slice(),
        width,
        slots: slots.into_boxed_slice(),
        source_currents,
        amplitudes: runs(amplitudes),
    })
}

fn execute(
    runner: &Runner,
    encoded: &Encoded,
    plan: &DirectRecurrencePlan,
    row_groups: &[DirectRowGroupDescriptor],
    executors: &DirectExecutorCatalog,
    workspace: &mut DirectWorkspace<'_>,
    point_count: u32,
) -> RusticolResult<()> {
    workspace.validate(point_count)?;
    // the sources, on the CPU (`encode` checked that they come first)
    for descriptor in row_groups
        .iter()
        .filter(|descriptor| descriptor.role == DirectExecutorRole::Source)
    {
        if descriptor.direct_executor_id == DIRECT_NONE_U32
            && plan.strategy() == RecurrenceStrategy::AllFlowUnion
        {
            continue;
        }
        let start = descriptor.row_start as usize;
        let rows = plan
            .sources()
            .get(start..start + descriptor.row_count as usize)
            .ok_or_else(|| {
                RusticolError::integrity("direct recurrence source row group is out of bounds")
            })?;
        let DirectExecutorHandle::Source { call, context } =
            executors.require(descriptor.direct_executor_id, DirectExecutorRole::Source)?
        else {
            return Err(RusticolError::integrity(
                "a source executor has another role",
            ));
        };
        let (arena, momenta, parameters, factors) = workspace.raw_views()?;
        let status = unsafe {
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
        };
        check_status(
            DirectExecutorRole::Source,
            descriptor.direct_executor_id,
            status,
        )?;
    }

    // the scales of this call's parameters
    let (_, _, parameters, _) = workspace.raw_views()?;
    let mut scale_re = Vec::with_capacity(encoded.slots.len());
    let mut scale_im = Vec::with_capacity(encoded.slots.len());
    for &id in encoded.slots.iter() {
        let DirectExecutorHandle::Contribution { context, .. } =
            executors.require(id, DirectExecutorRole::Contribution)?
        else {
            return Err(RusticolError::integrity(
                "a slot's executor has another role",
            ));
        };
        let (re, im) = unsafe { crate::engine::pacman_contribution_scale(context, parameters) }
            .ok_or_else(|| {
                RusticolError::evaluation(format!(
                    "direct recurrence executor {id}: its model parameter is out of bounds"
                ))
            })?;
        scale_re.push(re);
        scale_im.push(im);
    }

    let stride = workspace.point_stride as usize;
    let planes = |len: usize| -> RusticolResult<u32> {
        u32::try_from(len / stride)
            .map_err(|_| RusticolError::integrity("a recurrence arena exceeds u32 planes"))
    };
    let current_planes = planes(workspace.current_re.len())?;
    let whole_currents = [0, current_planes];
    let download_current: &[u32] = if runner.whole { &whole_currents } else { &[] };
    let call = RunnerCall {
        abi_version: ABI_VERSION,
        struct_size: size_of::<RunnerCall>() as u32,
        words: encoded.words.as_ptr(),
        words_key: encoded.key,
        word_count: encoded.words.len() as u32,
        width: encoded.width,
        current_re: workspace.current_re.as_mut_ptr(),
        current_im: workspace.current_im.as_mut_ptr(),
        amplitude_re: workspace.amplitude_re.as_mut_ptr(),
        amplitude_im: workspace.amplitude_im.as_mut_ptr(),
        momenta: workspace.momenta.as_ptr(),
        scale_re: scale_re.as_ptr(),
        scale_im: scale_im.as_ptr(),
        factor_re: workspace.factors_re.as_ptr(),
        factor_im: workspace.factors_im.as_ptr(),
        upload_current: encoded.source_currents.as_ptr(),
        upload_amplitude: encoded.amplitudes.as_ptr(),
        download_amplitude: encoded.amplitudes.as_ptr(),
        download_current: download_current.as_ptr(),
        current_planes,
        amplitude_planes: planes(workspace.amplitude_re.len())?,
        momentum_planes: planes(workspace.momenta.len())?,
        scale_count: scale_re.len() as u32,
        factor_count: workspace.factors_re.len() as u32,
        upload_current_runs: (encoded.source_currents.len() / 2) as u32,
        upload_amplitude_runs: (encoded.amplitudes.len() / 2) as u32,
        download_amplitude_runs: (encoded.amplitudes.len() / 2) as u32,
        download_current_runs: (download_current.len() / 2) as u32,
        stride: workspace.point_stride,
        points: point_count,
    };
    dump_once(
        &call,
        &scale_re,
        &scale_im,
        workspace.factors_re,
        workspace.factors_im,
    );
    // SAFETY: every pointer addresses a live slice of the stated length; the
    // library copies from and to them only during the call
    let status = unsafe { (runner.run)(&call) };
    if status != 0 {
        return Err(RusticolError::evaluation(format!(
            "{RUNNER}: the GPU run failed with status {status}"
        )));
    }
    Ok(())
}

/// `PACMAN_RECURRENCE_DUMP`: the first call's words and tables.
fn dump_once(
    call: &RunnerCall,
    scale_re: &[f64],
    scale_im: &[f64],
    factor_re: &[f64],
    factor_im: &[f64],
) {
    static DUMPED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let Some(path) = std::env::var_os(DUMP) else {
        return;
    };
    if DUMPED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        return;
    }
    let words = unsafe { std::slice::from_raw_parts(call.words, call.word_count as usize) };
    let mut out = b"PACRWRD1".to_vec();
    out.extend(call.word_count.to_le_bytes());
    out.extend(call.width);
    words.iter().for_each(|word| out.extend(word.to_le_bytes()));
    for table in [(scale_re, scale_im), (factor_re, factor_im)] {
        out.extend((table.0.len() as u32).to_le_bytes());
        table
            .0
            .iter()
            .chain(table.1)
            .for_each(|value| out.extend(value.to_le_bytes()));
    }
    for count in [
        call.current_planes,
        call.amplitude_planes,
        call.momentum_planes,
    ] {
        out.extend(count.to_le_bytes());
    }
    match std::fs::write(&path, out) {
        Ok(()) => eprintln!("pacman: {DUMP}: wrote {}", path.to_string_lossy()),
        Err(error) => eprintln!("pacman: {DUMP}: {}: {error}", path.to_string_lossy()),
    }
}
