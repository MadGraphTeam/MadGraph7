//! Diagnostic-only bridge timers. Never installed with production libraries.
use std::{cell::RefCell, collections::BTreeMap, fmt::Write as _, sync::atomic::{AtomicU32, Ordering}, time::Instant};

static LEVEL: AtomicU32 = AtomicU32::new(0);
const NAMES: [&str; 20] = ["calls", "ffi_total_ns", "input_batch_ns", "state_lock_ns",
    "state_dp_checked_ns", "store_attach_ns", "reduce_batch_ns", "output_pack_ns",
    "output_copy_ns", "workspace_setup_ns", "profile_configuration_ns",
    "upstream_arguments_ns", "upstream_scalar_masters_ns", "upstream_reduction_ns",
    "upstream_total_ns", "failed_calls", "direct_lanes", "method_choice_lanes",
    "precision_tier_lanes", "refused_lanes"];
const AUX_NAMES: [&str; 6] = ["clear_calls", "clear_ns", "budget_calls", "budget_ns",
    "register_calls", "register_ns"];
#[derive(Default)]
struct Book {
    current: [i32; 4],
    rows: BTreeMap<[i32; 4], [u64; 20]>,
    auxiliary: [u64; 6],
}
thread_local! { static BOOK: RefCell<Book> = RefCell::new(Book::default()); }
pub fn level() -> u32 { LEVEL.load(Ordering::Relaxed) }
pub fn start() -> Option<Instant> { (level() > 0).then(Instant::now) }
pub fn begin(n: i32, rank: i32, check: i32, cache: i32) {
    if level() == 0 { return; }
    BOOK.with(|b| {
        let mut b = b.borrow_mut(); b.current = [n, rank, check, cache];
        let key = b.current; b.rows.entry(key).or_default()[0] += 1;
    });
}
pub fn end(slot: usize, started: Option<Instant>) {
    if let Some(started) = started {
        let elapsed = started.elapsed().as_nanos() as u64;
        BOOK.with(|b| { let mut b = b.borrow_mut(); let key = b.current;
            b.rows.entry(key).or_default()[slot] += elapsed; });
    }
}
pub fn status(status: i32) {
    if level() == 0 || status == 0 { return; }
    BOOK.with(|b| { let mut b = b.borrow_mut(); let key = b.current;
        b.rows.entry(key).or_default()[15] += 1; });
}
pub fn auxiliary(slot: usize, started: Option<Instant>) {
    if let Some(started) = started {
        let elapsed = started.elapsed().as_nanos() as u64;
        BOOK.with(|b| { let mut b = b.borrow_mut(); b.auxiliary[slot] += 1;
            b.auxiliary[slot + 1] += elapsed; });
    }
}
pub fn upstream(profile: Option<iregi_rust::ExecutionProfile>) {
    if level() < 2 { return; }
    if let Some(profile) = profile {
        BOOK.with(|b| { let mut b = b.borrow_mut(); let key = b.current;
            let row = b.rows.entry(key).or_default();
            for (i, d) in [profile.arguments, profile.scalar_masters,
                profile.reduction, profile.total].iter().enumerate() {
                row[11 + i] += d.as_nanos() as u64;
            }
        });
    }
}
pub fn routes(routes: Option<&[iregi_rust::LaneRoute]>) {
    if level() == 0 { return; }
    if let Some(routes) = routes {
        BOOK.with(|b| { let mut b = b.borrow_mut(); let key = b.current;
            let row = b.rows.entry(key).or_default();
            for route in routes {
                let offset = match route {
                    iregi_rust::LaneRoute::Direct => 0,
                    iregi_rust::LaneRoute::MethodChoice => 1,
                    iregi_rust::LaneRoute::Tier(_) => 2,
                    iregi_rust::LaneRoute::Refused => 3,
                };
                row[16 + offset] += 1;
            }
        });
    }
}
#[unsafe(no_mangle)]
pub extern "C" fn ml_tirust_profile_control(level: i32) -> i32 {
    if !(0..=2).contains(&level) { return 2; }
    LEVEL.store(level as u32, Ordering::Relaxed);
    BOOK.with(|b| *b.borrow_mut() = Book::default());
    0
}
/// Diagnostic single-thread snapshot; returns required capacity including NUL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_profile_json(out: *mut u8, capacity: usize) -> usize {
    let mut text = format!("{{\"schema_version\":1,\"level\":{},\"time_unit\":\"ns\",\"rows\":[", level());
    BOOK.with(|b| {
        let b = b.borrow();
        for (index, (key, values)) in b.rows.iter().enumerate() {
            if index > 0 { text.push(','); }
            write!(text, "{{\"n\":{},\"rank\":{},\"check\":{},\"cache\":{}", key[0], key[1], key[2], key[3]).unwrap();
            for (name, value) in NAMES.iter().zip(values) { write!(text, ",\"{name}\":{value}").unwrap(); }
            text.push('}');
        }
        text.push_str("],\"auxiliary\":{");
        for (index, (name, value)) in AUX_NAMES.iter().zip(b.auxiliary).enumerate() {
            if index > 0 { text.push(','); }
            write!(text, "\"{name}\":{value}").unwrap();
        }
        text.push_str("},\"scope\":\"Diagnostic inclusive timers; inner stages overlap and timing overhead is not subtracted\"}");
    });
    if !out.is_null() && capacity > 0 {
        let count = text.len().min(capacity - 1);
        unsafe { std::ptr::copy_nonoverlapping(text.as_ptr(), out, count); out.add(count).write(0); }
    }
    text.len() + 1
}
