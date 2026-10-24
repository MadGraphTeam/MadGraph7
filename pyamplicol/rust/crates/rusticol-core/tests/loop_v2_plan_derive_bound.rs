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
use std::io::Read;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

use rusticol_core::loop_v2::plan::FormPolicy;
use rusticol_core::loop_v2::plan::binary::PlanModel;
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

#[test]
fn weights_stay_within_the_bound() {
    let long = format!("{}/7", "1234567890".repeat(30));
    for (classes, terms, weight) in [
        (2000, 16, "-3/4"),
        (200, 8, long.as_str()),
        (50, 4, "12.5e3000"),
        (4, 2, "1e200000"),
    ] {
        let mut plan = empty();
        let column = vec![json!([0, weight, "0"]); terms];
        plan["seed_classes"] = vec![json!({"columns": [column]}); classes].into();
        let keys: Vec<Value> = (0..classes).map(|k| json!([k, "1", weight])).collect();
        plan["seeds"] = json!([{"classes": keys}]);
        measure(
            &format!("weights {classes} x {terms} {}", &weight[..4]),
            &text(&plan),
        );
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
