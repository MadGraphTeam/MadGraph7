// SPDX-License-Identifier: 0BSD

//! The derivation bound of the binary plan member's reader
//! (`loop_v2::plan::binary`, `derive` module) against the loader's real
//! allocations: under a counting global allocator, the bytes
//! `PlanModel::program` adds at its peak to what was live before it (the
//! decoded model, which it frees in part) and allocates in all stay within
//! the bound's `memory` and `allocated` (the reader's fallible probe
//! excluded). Crafted plans make each kind of list
//! dominate (fused entries, split vertices, weights, families and edges with
//! their split candidates); `exported_plans_stay_within_their_bound` takes
//! exported plan JSONs (plain or gzip) from `LOOP_PLAN_BOUND_PLANS`
//! (colon-separated), ignored by default.

#![cfg(feature = "loop-v2")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ffi::c_void;
use std::io::Read;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Mutex, Once};

use rusticol_core::loop_v2::plan::binary::PlanModel;
use rusticol_core::loop_v2::plan::{FormPolicy, LoopProgramV4};
use serde_json::{Value, json};

thread_local! {
    /// Whether this thread's allocations are being measured.
    static MEASURING: Cell<bool> = const { Cell::new(false) };
}

/// The measuring thread's bytes allocated less freed in the window (frees
/// of the decoded model included: the address space the derivations add to
/// it), their maximum, and the bytes allocated.
static LIVE: AtomicI64 = AtomicI64::new(0);
static PEAK: AtomicI64 = AtomicI64::new(0);
static TOTAL: AtomicU64 = AtomicU64::new(0);
/// The size of the reader's probe, not counted (an allocation of alignment
/// one and exactly this size).
static IGNORED: AtomicU64 = AtomicU64::new(0);

struct Counting;

#[global_allocator]
static ALLOCATOR: Counting = Counting;

fn measuring(layout: Layout) -> bool {
    MEASURING.try_with(Cell::get).unwrap_or(false)
        && !(layout.align() == 1 && layout.size() as u64 == IGNORED.load(Ordering::Relaxed))
}

fn grow(bytes: usize) {
    let live = LIVE.fetch_add(bytes as i64, Ordering::Relaxed) + bytes as i64;
    PEAK.fetch_max(live, Ordering::Relaxed);
    TOTAL.fetch_add(bytes as u64, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if measuring(layout) {
            grow(layout.size());
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if measuring(layout) {
            grow(layout.size());
        }
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        // A new allocation of `size` replacing the old one.
        if measuring(layout) {
            grow(size);
            LIVE.fetch_sub(layout.size() as i64, Ordering::Relaxed);
        }
        unsafe { System.realloc(pointer, layout, size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if measuring(layout) {
            LIVE.fetch_sub(layout.size() as i64, Ordering::Relaxed);
        }
        unsafe { System.dealloc(pointer, layout) }
    }
}

// GMP's allocations (numerica's multi-precision numbers are rug's, over GMP,
// which allocates through C `malloc`, not the global allocator; MPFR through
// GMP's functions), counted through GMP's memory functions alike.
type GmpAllocate = unsafe extern "C" fn(usize) -> *mut c_void;
type GmpReallocate = unsafe extern "C" fn(*mut c_void, usize, usize) -> *mut c_void;
type GmpFree = unsafe extern "C" fn(*mut c_void, usize);

unsafe extern "C" {
    /// GMP's `mp_set_memory_functions` (linked through numerica's GMP).
    #[link_name = "__gmp_set_memory_functions"]
    fn gmp_set_memory_functions(
        allocate: Option<GmpAllocate>,
        reallocate: Option<GmpReallocate>,
        free: Option<GmpFree>,
    );
}

fn gmp_measuring() -> bool {
    MEASURING.try_with(Cell::get).unwrap_or(false)
}

unsafe extern "C" fn gmp_allocate(size: usize) -> *mut c_void {
    if gmp_measuring() {
        grow(size);
    }
    unsafe { libc::malloc(size) }
}

unsafe extern "C" fn gmp_reallocate(pointer: *mut c_void, old: usize, size: usize) -> *mut c_void {
    if gmp_measuring() {
        grow(size);
        LIVE.fetch_sub(old as i64, Ordering::Relaxed);
    }
    unsafe { libc::realloc(pointer, size) }
}

unsafe extern "C" fn gmp_free(pointer: *mut c_void, size: usize) {
    if gmp_measuring() {
        LIVE.fetch_sub(size as i64, Ordering::Relaxed);
    }
    unsafe { libc::free(pointer) }
}

/// GMP's allocations counted from the first measurement on (C `malloc` both
/// ways, so a number allocated before is freed alike).
static GMP_COUNTED: Once = Once::new();

/// One measurement at a time (the test harness runs tests in threads; the
/// loader's derivations run on the calling thread).
static WINDOW: Mutex<()> = Mutex::new(());

/// What the loader's derivations of `json`'s binary member held at their
/// peak and allocated in all, beyond the decoded model, against the bound;
/// asserts the bound holds and the plan loads.
fn measure(name: &str, json: &[u8]) {
    let member = PlanModel::from_json(json)
        .and_then(|mut model| model.binary())
        .unwrap_or_else(|error| panic!("{name}: {error}"));
    let model = PlanModel::from_binary(&member).unwrap();
    let bound = model.derive_bound().unwrap();
    GMP_COUNTED.call_once(|| unsafe {
        gmp_set_memory_functions(Some(gmp_allocate), Some(gmp_reallocate), Some(gmp_free));
    });
    let window = WINDOW
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    IGNORED.store(bound.reserved(), Ordering::Relaxed);
    LIVE.store(0, Ordering::Relaxed);
    PEAK.store(0, Ordering::Relaxed);
    TOTAL.store(0, Ordering::Relaxed);
    MEASURING.with(|on| on.set(true));
    let program = model.program(FormPolicy::Rule);
    MEASURING.with(|on| on.set(false));
    let peak = PEAK.load(Ordering::Relaxed) as u64;
    let allocated = TOTAL.load(Ordering::Relaxed);
    drop(window);
    let program = program.unwrap_or_else(|error| panic!("{name}: {error}"));
    drop(program);
    eprintln!(
        "{name}: member {} B, peak {peak} <= {} ({:.2}), allocated {allocated} <= {} ({:.2})",
        member.len(),
        bound.memory,
        peak as f64 / bound.memory as f64,
        bound.allocated,
        allocated as f64 / bound.allocated as f64,
    );
    assert!(
        peak <= bound.memory,
        "{name}: peak {peak} > {}",
        bound.memory
    );
    assert!(
        allocated <= bound.allocated,
        "{name}: allocated {allocated} > {}",
        bound.allocated
    );
}

/// A plan of one one-helicity leg and no family.
fn empty() -> Value {
    json!({
        "schema": 4, "legs": 1, "rows": 1, "helicities": [[1]], "columns": 1, "average": 1,
        "born_components": 1, "constants": 0, "masses": 1, "block_rows": [], "kernels": [],
        "contents": [], "factors": [], "seed_classes": [], "seeds": [], "families": [],
        "forward": null, "merged": null,
    })
}

fn text(plan: &Value) -> Vec<u8> {
    serde_json::to_vec(plan).unwrap()
}

/// A content no step reads, of `entries` distinct fused entries over a
/// `width × width` kernel.
fn fused_content(entries: usize, width: usize) -> Value {
    let fused: Vec<Value> = (0..entries)
        .map(|i| json!([i / width, i % width, 0, 0, 0, 0]))
        .collect();
    json!({
        "d_in": width, "d_out": width, "degree": 0, "rows": 0, "mask": 0,
        "row_helicities": [], "fused": fused, "terms": [],
    })
}

/// A content on the leg of `terms` vertex terms (vertex ids from `first`),
/// each with one split kernel (kernel 0) of `entries` entries over as many
/// fused entries of a `width × width` kernel, and one row.
fn vertex_content(terms: usize, entries: usize, width: usize, first: usize) -> Value {
    let mut content = fused_content(entries, width);
    content["rows"] = json!(1);
    content["mask"] = json!(1);
    content["row_helicities"] = json!([[1]]);
    let split: Vec<Value> = (0..entries).map(|e| json!([e, 0])).collect();
    content["terms"] = (0..terms)
        .map(|k| {
            json!({
                "slots": [], "dimensions": [], "bindings": [], "vertex": first + k,
                "split": [{"kernel": 0, "inputs": [], "entries": split}],
                "rows": [[0, []]],
            })
        })
        .collect();
    content
}

/// The split kernel of `entries` outputs, without inputs or terms.
fn split_kernel(entries: usize) -> Value {
    json!({"inputs": [], "outputs": vec![json!([]); entries]})
}

#[test]
fn fused_entries_stay_within_the_bound() {
    for (entries, width) in [(1 << 16, 256), (3000, 64)] {
        let mut plan = empty();
        plan["contents"] = json!([fused_content(entries, width)]);
        measure(&format!("fused {entries}"), &text(&plan));
    }
    // Contents sharing one fused list (stored once in the member).
    let mut plan = empty();
    plan["contents"] = vec![fused_content(4096, 64); 64].into();
    measure("shared fused 64 x 4096", &text(&plan));
}

#[test]
fn split_vertices_stay_within_the_bound() {
    for (terms, entries, width) in [(1, 1 << 14, 128), (64, 512, 32), (2000, 8, 4)] {
        let mut plan = empty();
        plan["kernels"] = json!([split_kernel(entries)]);
        plan["contents"] = json!([vertex_content(terms, entries, width, 0)]);
        measure(&format!("split {terms} x {entries}"), &text(&plan));
    }
}

/// `classes` seed classes of `terms` terms, each of weight `weight`, and one
/// seed key reading them all with it.
fn weights_plan(classes: usize, terms: usize, weight: &str) -> Value {
    let mut plan = empty();
    let column = vec![json!([0, weight, "0"]); terms];
    plan["seed_classes"] = vec![json!({"columns": [column]}); classes].into();
    let keys: Vec<Value> = (0..classes).map(|k| json!([k, "1", weight])).collect();
    plan["seeds"] = json!([{"classes": keys}]);
    plan
}

/// The weights' cases: name, classes, terms, weight.
fn weight_cases(scale: u32) -> Vec<(String, usize, usize, String)> {
    let digits = |n: usize| "1234567890".repeat(n.div_ceil(10))[..n].to_string();
    let power = 10usize.pow(scale);
    let mantissa = digits(power / 10);
    let mut cases = vec![
        ("p/q".into(), 2000, 16, "-3/4".into()),
        ("long fraction".into(), 200, 8, format!("{}/7", digits(300))),
        ("12.5e3000".into(), 50, 4, "12.5e3000".into()),
    ];
    for (name, weight) in [
        (format!("1e{power}"), format!("1e{power}")),
        (format!("1e-{power}"), format!("1e-{power}")),
        (format!("-9e{power}"), format!("-9e{power}")),
        // A long mantissa times a power: the product, and for a negative
        // power the normalising gcd and divisions.
        (
            format!("{} digits e{power}", power / 10),
            format!("{mantissa}e{power}"),
        ),
        (
            format!("{} digits e-{power}", power / 10),
            format!("{mantissa}e-{power}"),
        ),
        (
            format!("0.{} digits e-{power}", power / 10),
            format!("0.{mantissa}e-{power}"),
        ),
        // A long fraction with a common factor (gcd and divisions).
        (
            format!("{} digits / {} digits", power, power),
            format!("{}0/{}5", digits(power), digits(power)),
        ),
    ] {
        cases.push((name, 2, 2, weight));
    }
    cases
}

#[test]
fn weights_stay_within_the_bound() {
    for (name, classes, terms, weight) in weight_cases(5) {
        let plan = weights_plan(classes, terms, &weight);
        measure(&format!("weights {classes} x {terms} {name}"), &text(&plan));
    }
}

/// The decimal powers and long numbers at larger scales: 10^(10^6) to
/// 10^(10^SCALE) (`LOOP_PLAN_BOUND_SCALE`, default 7; GMP's algorithms change
/// with the operands' sizes), ignored by default (seconds to minutes).
#[test]
#[ignore = "large weights (evidence runs)"]
fn large_weights_stay_within_the_bound() {
    let top = std::env::var("LOOP_PLAN_BOUND_SCALE").map_or(7, |s| s.parse().unwrap());
    for scale in 6..=top {
        for (name, classes, terms, weight) in weight_cases(scale).into_iter().skip(3) {
            let plan = weights_plan(classes, terms, &weight);
            measure(&format!("weights {classes} x {terms} {name}"), &text(&plan));
        }
    }
}

#[test]
fn families_and_split_candidates_stay_within_the_bound() {
    for (families, edges, terms) in [(400, 8, 4), (8, 400, 16), (40, 40, 1)] {
        let mut plan = empty();
        plan["kernels"] = json!([split_kernel(1)]);
        let mut fused = fused_content(1, 1);
        fused["rows"] = json!(1);
        fused["mask"] = json!(1);
        fused["row_helicities"] = json!([[1]]);
        plan["contents"] = json!([fused, vertex_content(terms, 1, 1, 0)]);
        plan["factors"] = json!([[0, 0], [1, 0]]);
        plan["seeds"] = json!([{"classes": []}]);
        let step = |factor: usize| {
            let edges: Vec<Value> = (0..edges)
                .map(|e| json!({"leaf": 0, "child": -1, "factor": (e + factor) % 2}))
                .collect();
            json!({
                "seeded": true, "trace": 0, "out": 0, "d_alpha": 1, "d_out": 1, "degree": 0,
                "rank_in": 0, "edges": edges,
            })
        };
        plan["families"] = (0..families)
            .map(|f| {
                json!({
                    "sequence": [[1, 0]], "rank": 1, "orders": [0],
                    "leaves": [{"seed": 0, "weight": "1"}], "steps": [step(f % 2)],
                })
            })
            .collect();
        let name = format!("families {families} x {edges} x {terms}");
        measure(&name, &text(&plan));
    }
}

/// A plan of `families` scalar bubble families over a leg of `helicities`
/// helicity rows (one constant kernel term per row) and a one-row leg, each
/// family a seeded step on the first leg and a trace step on the second,
/// with `columns` observable columns of which each keeps `kept` consecutive
/// Born rows (`column_rows`: helicity-resolved columns, so the plan's
/// live-row pruning and the kept links' read rows, `plan/live.rs`, run).
fn row_restricted_plan(families: usize, helicities: usize, columns: usize, kept: usize) -> Value {
    let mut plan = empty();
    plan["legs"] = json!(2);
    plan["rows"] = json!(helicities);
    plan["helicities"] = json!([(0..helicities).collect::<Vec<_>>(), [0]]);
    plan["columns"] = json!(columns);
    plan["constants"] = json!(2);
    plan["kernels"] = json!([
        {"inputs": [], "outputs": [[[0, []]]]},
        {"inputs": [], "outputs": [[[1, []]]]},
    ]);
    let term = |row: usize, kernel: usize| {
        json!({
            "slots": [], "dimensions": [], "vertex": -1, "split": [], "rows": [[row, []]],
            "bindings": [{"kernel": kernel, "inputs": [], "entries": [0]}],
        })
    };
    plan["contents"] = json!([
        {
            "d_in": 1, "d_out": 1, "degree": 0, "rows": helicities, "mask": 1,
            "row_helicities": (0..helicities).map(|h| json!([h])).collect::<Vec<_>>(),
            "fused": [[0, 0, 0, 0, 0, 0]],
            "terms": (0..helicities).map(|h| term(h, h % 2)).collect::<Vec<_>>(),
        },
        {
            "d_in": 1, "d_out": 1, "degree": 0, "rows": 1, "mask": 2,
            "row_helicities": [[0]], "fused": [[0, 0, 0, 0, 0, 0]], "terms": [term(0, 1)],
        },
    ]);
    plan["factors"] = json!([[0, 0], [1, 1]]);
    plan["seed_classes"] = json!([{"columns": vec![json!([[0, "1", "0"]]); columns]}]);
    plan["seeds"] = json!([{"classes": [[0, "1", "0"]]}]);
    let family = json!({
        "sequence": [[1, 0], [2, 0]], "rank": 0, "orders": [0],
        "leaves": [{"seed": 0, "weight": "3"}],
        "steps": [
            {"seeded": true, "trace": -1, "out": 2, "d_alpha": 1, "d_out": 1, "degree": 0,
             "rank_in": 0, "edges": [{"leaf": 0, "child": -1, "factor": 0}]},
            {"seeded": false, "trace": 0, "out": 0, "d_alpha": 1, "d_out": 1, "degree": 0,
             "rank_in": 0, "edges": [{"leaf": -1, "child": 0, "factor": 1}]},
        ],
    });
    plan["families"] = vec![family; families].into();
    plan["column_rows"] = (0..columns)
        .map(|c| {
            let first = (c * kept) % helicities;
            json!((first..first + kept).map(|r| r % helicities).collect::<Vec<_>>())
        })
        .collect();
    plan
}

#[test]
fn row_restricted_columns_stay_within_the_bound() {
    for (families, helicities, columns, kept) in
        [(400, 2, 2, 1), (8, 256, 4, 3), (64, 64, 16, 1), (40, 16, 2, 8)]
    {
        let plan = row_restricted_plan(families, helicities, columns, kept);
        let name = format!("row-restricted {families} x {helicities} x {columns} x {kept}");
        measure(&name, &text(&plan));
    }
}

#[test]
#[ignore = "exported plans from LOOP_PLAN_BOUND_PLANS (evidence runs)"]
fn exported_plans_stay_within_their_bound() {
    let paths = std::env::var("LOOP_PLAN_BOUND_PLANS").expect("LOOP_PLAN_BOUND_PLANS");
    for path in paths.split(':').filter(|path| !path.is_empty()) {
        let raw = std::fs::read(path).unwrap();
        let json = if path.ends_with(".gz") {
            let mut text = Vec::new();
            flate2::read::GzDecoder::new(raw.as_slice())
                .read_to_end(&mut text)
                .unwrap();
            text
        } else {
            raw
        };
        measure(path, &json);
    }
}

/// The environment variable naming the plan JSON a child of
/// [`weights_fit_their_bound_in_address_space`] loads.
#[cfg(target_os = "linux")]
const ADDRESS_PLAN: &str = "LOOP_PLAN_BOUND_ADDRESS_PLAN";

/// This process's `/proc/self/status` field `name` (kB), in bytes (Linux).
#[cfg(target_os = "linux")]
fn status(name: &str) -> u64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let line = status.lines().find(|line| line.starts_with(name)).unwrap();
    line[name.len()..]
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse::<u64>()
        .unwrap()
        * 1024
}

/// The multi-precision weights at `digits` digits, named: a power of ten of
/// either sign, a long mantissa times or over one (the product; the
/// normalising gcd and divisions), a long fraction with a common factor.
#[cfg(target_os = "linux")]
fn address_cases(digits: usize) -> Vec<(String, String)> {
    let text = |n: usize| "1234567890".repeat(n.div_ceil(10))[..n].to_string();
    let mantissa = text(digits / 10);
    let short = digits / 10;
    vec![
        (format!("1e{digits}"), format!("1e{digits}")),
        (format!("1e-{digits}"), format!("1e-{digits}")),
        (
            format!("{short} digits e{digits}"),
            format!("{mantissa}e{digits}"),
        ),
        (
            format!("{short} digits e-{digits}"),
            format!("{mantissa}e-{digits}"),
        ),
        (
            format!("0.{short} digits e-{digits}"),
            format!("0.{mantissa}e-{digits}"),
        ),
        (
            format!("{digits} digits / {digits} digits"),
            format!("{}0/{}5", text(digits), text(digits)),
        ),
    ]
}

/// The address space the loader's derivations of the plan `json` take
/// beyond the process's size before them, in a fresh child of this test
/// binary (one malloc arena, as the main thread's). The counting allocator
/// sees GMP's allocations (`measure`), not what C `malloc`'s placement makes
/// of them (its brk heap below the dynamic mmap threshold, fragmented, and
/// the mappings above it); an address-space limit sees both. The child reads
/// the plan from a file and derives through the JSON load
/// (`LoopProgramV4::load_unprobed`: the derivations of `LoopProgramV4::load`
/// without its probe of their peak, which would otherwise be the peak):
/// `VmPeak` after it over `VmSize` before it (nothing before it peaks above
/// its size). Linux only (`/proc/self/status`).
#[cfg(target_os = "linux")]
fn address_space(name: &str, json: &[u8]) -> u64 {
    let path = std::env::temp_dir().join(format!(
        "loop_v2_plan_derive_bound-{}.json",
        std::process::id()
    ));
    std::fs::write(&path, json).unwrap();
    let exe = std::env::current_exe().unwrap();
    let output = std::process::Command::new(exe)
        .args(["--exact", "weights_fit_their_bound_in_address_space"])
        .args(["--nocapture", "--test-threads=1"])
        .env(ADDRESS_PLAN, &path)
        .env("MALLOC_ARENA_MAX", "1")
        .output()
        .unwrap();
    std::fs::remove_file(&path).unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "{name}: {output:?}");
    let line = stdout
        .lines()
        .find_map(|line| line.split_once("ADDRESS ").map(|(_, values)| values))
        .unwrap_or_else(|| panic!("{name}: {stdout}"));
    let values: Vec<u64> = line.split(' ').map(|v| v.parse().unwrap()).collect();
    values[1].saturating_sub(values[0])
}

/// A plan whose weight is a GMP number takes, in address space, at most its
/// bound's `memory` (the reader probes that with its margin): 10^6 and
/// 3 × 10^6 digits by default, up to 10^`LOOP_PLAN_BOUND_ADDRESS_SCALE`
/// digits with that variable (evidence runs; scale 8 takes minutes). Linux
/// only: the address space is read from `/proc/self/status`.
#[cfg(target_os = "linux")]
#[test]
fn weights_fit_their_bound_in_address_space() {
    if let Some(path) = std::env::var_os(ADDRESS_PLAN) {
        let json = std::fs::read(path).unwrap();
        let before = status("VmSize:");
        let program = LoopProgramV4::load_unprobed(&json, FormPolicy::Rule).unwrap();
        let peak = status("VmPeak:");
        drop(program);
        println!("ADDRESS {before} {peak}");
        return;
    }
    let scale: u32 =
        std::env::var("LOOP_PLAN_BOUND_ADDRESS_SCALE").map_or(0, |s| s.parse().unwrap());
    let mut sizes = vec![1_000_000, 3_000_000];
    for k in 7..=scale {
        sizes.push(10usize.pow(k));
        if k < scale {
            sizes.push(3 * 10usize.pow(k));
        }
    }
    for digits in sizes {
        for (name, weight) in address_cases(digits) {
            let mut plan = empty();
            plan["seed_classes"] = json!([{"columns": [[[0, weight, "0"]]]}]);
            let json = text(&plan);
            drop(plan);
            let memory = PlanModel::from_json(&json)
                .unwrap()
                .derive_bound()
                .unwrap()
                .memory;
            let used = address_space(&name, &json);
            eprintln!(
                "address {name}: {used} <= {memory} ({:.2})",
                used as f64 / memory as f64
            );
            assert!(used <= memory, "{name}: address space {used} > {memory}");
        }
    }
}
