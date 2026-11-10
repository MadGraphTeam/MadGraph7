//! Private source-only cause journal. No numerical calls, cache access or timers.
//! All diagnostic strings already exist upstream. Never format a floating value.
use std::{cell::RefCell, fmt::Write as _, fs::{File, OpenOptions}, io::Write as _,
    panic::{catch_unwind, AssertUnwindSafe}, sync::atomic::{AtomicBool, AtomicU64, Ordering}};
use iregi_rust::{LaneRoute, TensorEngineExecution, TensorWorkspace};

const MAX_BYTES: usize = 32 * 1024 * 1024;
const MAX_EVENTS: usize = 200_000;
const MAX_WORDS: usize = 4096;
const MAX_TEXT: usize = 65_536;
const FP_READ: u64 = 1;
const WRONG_THREAD: u64 = 2;
const BORROW: u64 = 4;
const PANIC: u64 = 8;
const FP_CHANGE: u64 = 16;
const CAPACITY: u64 = 32;
const PROTOCOL: u64 = 64;
const IO: u64 = 128;
const UNQUALIFIED_POLICY: u64 = 256;
static OPENED: AtomicBool = AtomicBool::new(false);
static ENABLED: AtomicBool = AtomicBool::new(false);
static HEALTH: AtomicU64 = AtomicU64::new(0);
unsafe extern "C" { fn ml_tirust_private_observer_fp_read(out: *mut u64) -> i32; }

fn fault(bits: u64) { HEALTH.fetch_or(bits, Ordering::Relaxed); }
fn fp() -> [u64; 4] {
    let mut state = [0; 4];
    if unsafe { ml_tirust_private_observer_fp_read(state.as_mut_ptr()) } != 1 { fault(FP_READ); }
    state
}
struct Event { seq: u64, point: i64, phase: i32, native: u64,
    before: [u64; 4], after: [u64; 4], payload: String }
#[derive(Default)]
struct Journal {
    owner: bool, file: Option<File>, point: i64, phase: i32, sequence: u64,
    native_sequence: u64, active_native: u64, flushes: u64,
    dropped: u64, bytes: usize, events: Vec<Event>,
}
thread_local! { static JOURNAL: RefCell<Journal> = RefCell::new(Journal::default()); }
impl Journal {
    fn emit(&mut self, payload: String) {
        self.sequence += 1;
        // Include conservative fixed metadata/storage overhead in the byte cap.
        let size = payload.len().saturating_add(256);
        if self.events.len() >= MAX_EVENTS || size > MAX_BYTES.saturating_sub(self.bytes) {
            self.dropped += 1; fault(CAPACITY); return;
        }
        self.bytes += size;
        self.events.push(Event { seq: self.sequence, point: self.point, phase: self.phase,
            native: self.active_native, before: [0; 4], after: [0; 4], payload });
    }
}

// All allocations/string copying belong inside this observation interval.
// A changed state remains changed, is recorded, and permanently fails health.
fn observe(f: impl FnOnce(&mut Journal)) {
    if !ENABLED.load(Ordering::Relaxed) { return; }
    let before = fp();
    let outcome = catch_unwind(AssertUnwindSafe(|| JOURNAL.with(|slot| {
        let Ok(mut journal) = slot.try_borrow_mut() else { fault(BORROW); return; };
        if !journal.owner { fault(WRONG_THREAD); return; }
        let first = journal.events.len();
        f(&mut journal);
        let after = fp();
        for event in &mut journal.events[first..] { event.before = before; event.after = after; }
        if before != after { fault(FP_CHANGE); }
    })));
    if outcome.is_err() { fault(PANIC); }
    // Include destructor/unwind effects in the health decision too.
    if before != fp() { fault(FP_CHANGE); }
}
fn quoted(text: &str) -> String {
    if text.len() > MAX_TEXT { fault(CAPACITY); return "null".into(); }
    let mut out = String::with_capacity(text.len() + 2); out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"), '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"), c if c <= '\u{1f}' => { write!(out, "\\u{:04x}", c as u32).unwrap(); },
            c => out.push(c),
        }
    }
    out.push('"'); out
}

pub fn native_begin(context: usize, precision: i32, n: i32, rank: i32,
                    poles: i32, finite: i32, check: i32, cache: i32) {
    observe(|j| {
        if j.active_native != 0 { fault(PROTOCOL); }
        j.native_sequence += 1; j.active_native = j.native_sequence;
        j.emit(format!("{{\"kind\":\"native_begin\",\"context\":{},\"input_bits\":{},\"configured_work_bits\":{},\"n\":{},\"rank\":{},\"poles\":{},\"finite\":{},\"check\":{},\"cache\":{}}}",
            context, precision, if precision == 113 { 256 } else { 53 }, n, rank, poles, finite, check, cache));
    });
}
pub fn native_end(status: i32) {
    observe(|j| {
        if j.active_native == 0 { fault(PROTOCOL); }
        // No guarded bridge API is called before copying the actual public error.
        super::LAST_ERROR.with(|slot| {
            let error = slot.borrow();
            j.emit(format!("{{\"kind\":\"native_end\",\"status\":{},\"error\":{}}}", status, quoted(&error)));
        });
        j.active_native = 0;
    });
}
pub fn qp_cache_hit() { observe(|j| j.emit("{\"kind\":\"qp_cache_hit\",\"new_numerical_evaluation\":false}".into())); }

pub fn dp_workspace(workspace: &TensorWorkspace<f64>, completed: bool) {
    observe(|j| {
        let options = workspace.embedder_options();
        if let Some(options) = options {
            let master_policy = format!("{:?}", options.master_policy);
            // Bind actual options, not merely the final route. Never infer pure
            // DP from Direct; partial execution has no complete public counter.
            if options.partial_rescue || options.precision_tiers || !options.method_choice
                || options.gate.is_some() || options.method_choice_gate.is_some()
                || master_policy != "Default" { fault(UNQUALIFIED_POLICY); }
            j.emit(format!("{{\"kind\":\"options\",\"method_choice\":{},\"partial_rescue\":{},\"precision_tiers\":{},\"gate_bits\":{},\"choice_gate_bits\":{},\"master_policy\":{},\"partial_work_observable\":false}}",
                options.method_choice, options.partial_rescue, options.precision_tiers,
                options.gate.map(|v| format!("\"{:016x}\"", v.to_bits())).unwrap_or_else(|| "null".into()),
                options.method_choice_gate.map(|v| format!("\"{:016x}\"", v.to_bits())).unwrap_or_else(|| "null".into()),
                quoted(&master_policy)));
        } else { fault(UNQUALIFIED_POLICY); }
        j.emit(format!("{{\"kind\":\"upstream_completion\",\"result_ok\":{},\"workspace_bits\":{}}}", completed, workspace.binary_precision()));
        // On a Rust-level Err, last_* may belong to an earlier invocation.
        // Do not copy stale routes or counters and call them this call's work.
        if !completed { return; }
        if let Some(routes) = workspace.last_lane_routes() {
            if routes.len() != 1 { fault(PROTOCOL); }
            for (lane, route) in routes.iter().enumerate() {
                let (name, bits) = match route {
                    LaneRoute::Direct => ("direct", 0), LaneRoute::MethodChoice => ("method_choice", 0),
                    LaneRoute::Tier(bits) => ("tier", *bits), LaneRoute::Refused => ("refused", 0),
                };
                j.emit(format!("{{\"kind\":\"lane_route\",\"lane\":{},\"route\":\"{}\",\"accepted_tier_bits\":{},\"direct_is_pure_dp_certificate\":false}}", lane, name, bits));
            }
        } else { fault(UNQUALIFIED_POLICY); }
        if workspace.last_diagnostics().len() != 1 { fault(PROTOCOL); }
        for (lane, notes) in workspace.last_diagnostics().iter().enumerate() {
            for (index, note) in notes.iter().enumerate() {
                j.emit(format!("{{\"kind\":\"diagnostic\",\"lane\":{},\"index\":{},\"text\":{}}}", lane, index, quoted(note)));
            }
        }
        match workspace.engine_execution() {
            TensorEngineExecution::Direct(stats) => {
                if stats.input_rows != 1 { fault(PROTOCOL); }
                j.emit(format!("{{\"kind\":\"execution\",\"input_rows\":{},\"master_rows_all_precisions\":{},\"direct_successes\":{},\"fallback_rows\":{},\"fallback_successes\":{},\"symbolic_rows\":{},\"rotation_rows\":{},\"rotation_failures\":{},\"rotation_unchecked\":{},\"checked_rows\":{},\"checked_successes\":{},\"expansion_rows\":{},\"expansion_successes\":{}}}",
                    stats.input_rows, stats.master_rows, stats.direct_successes, stats.fallback_rows,
                    stats.fallback_successes, stats.symbolic_rows, stats.rotation_rows, stats.rotation_failures,
                    stats.rotation_unchecked, stats.checked_rows, stats.checked_successes,
                    stats.expansion_rows, stats.expansion_successes));
                for (attempt, tier) in stats.precision_rescue.iter().enumerate() {
                    j.emit(format!("{{\"kind\":\"tier_attempt\",\"attempt\":{},\"bits\":{},\"rows\":{},\"successes\":{}}}", attempt, tier.bits, tier.rows, tier.successes));
                }
                if stats.fallback.is_some() || stats.symbolic_rows != 0 { fault(UNQUALIFIED_POLICY); }
                if let Some(error) = stats.fallback_preparation_error {
                    j.emit(format!("{{\"kind\":\"fallback_preparation_error\",\"error\":{}}}", quoted(&error)));
                }
            }
            _ => fault(UNQUALIFIED_POLICY),
        }
    });
}

#[unsafe(no_mangle)]
pub extern "C" fn ml_tirust_observer_begin_point(point: i64, phase: i32) {
    observe(|j| { if j.active_native != 0 { fault(PROTOCOL); }
        j.point = point; j.phase = phase; j.emit("{\"kind\":\"point\"}".into()); });
}
/// # Safety
/// words identifies count readable i64 objects; no numerical buffer is read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_observer_event(tag: i32, words: *const i64, count: usize) {
    observe(|j| {
        if count > MAX_WORDS || (count != 0 && (words.is_null() || !words.is_aligned())) { fault(PROTOCOL); return; }
        let values = if count == 0 { &[] } else { unsafe { std::slice::from_raw_parts(words, count) } };
        let mut payload = format!("{{\"kind\":\"host\",\"tag\":{},\"words\":[", tag);
        for (i, value) in values.iter().enumerate() {
            if i > 0 { payload.push(','); } write!(payload, "\"{:016x}\"", *value as u64).unwrap();
        }
        payload.push_str("]}"); j.emit(payload);
    });
}

// These entrypoints are called only outside complete public amplitude calls.
// The observer never opens a file or flushes a journal from numerical hooks.
fn outside(f: impl FnOnce() -> i32) -> i32 {
    let before = fp();
    let result = catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| { fault(PANIC); 2 });
    if before != fp() { fault(FP_CHANGE); }
    if HEALTH.load(Ordering::Relaxed) != 0 { 2 } else { result }
}
/// # Safety
/// path points to length readable UTF-8 bytes (without embedded NUL).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_observer_open(path: *const u8, length: usize, enabled: i32) -> i32 {
    outside(|| {
        if !matches!(enabled, 0 | 1) || length == 0 || length > 4096 || path.is_null()
            || OPENED.swap(true, Ordering::Relaxed) { fault(PROTOCOL); return 2; }
        let bytes = unsafe { std::slice::from_raw_parts(path, length) };
        let Ok(name) = std::str::from_utf8(bytes) else { fault(PROTOCOL); return 2; };
        if name.as_bytes().contains(&0) { fault(PROTOCOL); return 2; }
        let file = match OpenOptions::new().create_new(true).write(true).open(name) {
            Ok(file) => file, Err(_) => { fault(IO); return 2; }
        };
        JOURNAL.with(|slot| { let mut j = slot.borrow_mut(); j.owner = true; j.file = Some(file); });
        ENABLED.store(enabled != 0, Ordering::Relaxed); 0
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn ml_tirust_observer_flush() -> i32 {
    outside(|| JOURNAL.with(|slot| {
        let mut j = slot.borrow_mut();
        if !j.owner || j.active_native != 0 { fault(PROTOCOL); return 2; }
        j.flushes += 1;
        let mut text = format!("{{\"kind\":\"flush_begin\",\"schema\":1,\"flush\":{},\"enabled\":{},\"events\":{},\"health\":{},\"dropped\":{},\"sequence\":{},\"native_calls\":{},\"partial_work_observable\":false}}\n",
            j.flushes, ENABLED.load(Ordering::Relaxed), j.events.len(), HEALTH.load(Ordering::Relaxed), j.dropped, j.sequence, j.native_sequence);
        for e in &j.events {
            writeln!(text, "{{\"kind\":\"event\",\"seq\":{},\"point\":{},\"phase\":{},\"native\":{},\"fp_before\":{:?},\"fp_after\":{:?},\"data\":{}}}",
                e.seq, e.point, e.phase, e.native, e.before, e.after, e.payload).unwrap();
        }
        writeln!(text, "{{\"kind\":\"flush_end\",\"flush\":{},\"events\":{},\"health\":{},\"dropped\":{}}}",
            j.flushes, j.events.len(), HEALTH.load(Ordering::Relaxed), j.dropped).unwrap();
        let Some(file) = j.file.as_mut() else { fault(PROTOCOL); return 2; };
        if file.write_all(text.as_bytes()).and_then(|_| file.flush()).is_err() { fault(IO); return 2; }
        j.events.clear(); j.bytes = 0; 0
    }))
}
#[unsafe(no_mangle)]
pub extern "C" fn ml_tirust_observer_close() -> i32 {
    let result = ml_tirust_observer_flush();
    outside(|| JOURNAL.with(|slot| {
        let mut j = slot.borrow_mut();
        if !j.owner || j.active_native != 0 { fault(PROTOCOL); return 2; }
        let trailer = format!("{{\"kind\":\"close\",\"schema\":1,\"enabled\":{},\"health\":{},\"dropped\":{},\"sequence\":{},\"native_calls\":{},\"flushes\":{}}}\n",
            ENABLED.load(Ordering::Relaxed), HEALTH.load(Ordering::Relaxed), j.dropped,
            j.sequence, j.native_sequence, j.flushes);
        let Some(mut file) = j.file.take() else { fault(PROTOCOL); return 2; };
        if file.write_all(trailer.as_bytes()).and_then(|_| file.flush()).is_err() { fault(IO); }
        drop(file);
        ENABLED.store(false, Ordering::Relaxed); result
    }))
}
#[unsafe(no_mangle)]
pub extern "C" fn ml_tirust_observer_health() -> u64 { HEALTH.load(Ordering::Relaxed) }
