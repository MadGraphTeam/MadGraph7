// SPDX-License-Identifier: 0BSD

//! The resident memory of a loaded plan v4 process, by part (M-FINAL
//! MF-8G-2 items 2 and 4): under a counting global allocator, the bytes the
//! plan loader (`LoopProgramV4`) and the tree-program loader (`TreeProgram`)
//! leave live, the loaders' peaks, and the plan's parts measured as the
//! bytes a deep copy of each allocates. `load_memory_of_stored_processes`
//! reads the `loop-runtime.pacbin` files listed (colon-separated) in
//! `LOOP_LOAD_MEMORY_PACBIN` and prints one JSON record per file; ignored by
//! default (a measuring tool, not a check).

#![cfg(feature = "loop-v2")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::io::Read;
use std::sync::atomic::{AtomicI64, Ordering};

use rusticol_core::loop_runtime::read_loop_program_v4_json;
use rusticol_core::loop_v2::plan::{FormPolicy, LoopProgramV4};
use rusticol_core::loop_v2::trees::TreeProgram;
use rusticol_core::pacbin::PacbinReader;
use serde_json::{Value, json};

static LIVE: AtomicI64 = AtomicI64::new(0);
static PEAK: AtomicI64 = AtomicI64::new(0);

struct Counting;

#[global_allocator]
static GLOBAL: Counting = Counting;

fn grow(bytes: i64) {
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        grow(layout.size() as i64);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        grow(layout.size() as i64);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        grow(new_size as i64 - layout.size() as i64);
        unsafe { System.realloc(pointer, layout, new_size) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as i64, Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) }
    }
}

const MIB: f64 = 1048576.0;

fn mib(bytes: i64) -> f64 {
    (bytes as f64 / MIB * 10.0).round() / 10.0
}

/// The live bytes `f`'s result holds (the bytes it allocates less those it
/// frees, while the result is alive) and the peak above the start.
fn measured<T>(f: impl FnOnce() -> T) -> (T, i64, i64) {
    let start = LIVE.load(Ordering::Relaxed);
    PEAK.store(start, Ordering::Relaxed);
    let value = f();
    (
        value,
        LIVE.load(Ordering::Relaxed) - start,
        PEAK.load(Ordering::Relaxed) - start,
    )
}

/// The bytes a deep copy of `value` allocates (its parts at exact capacity).
fn deep<T: Clone>(value: &T) -> i64 {
    let (copy, live, _) = measured(|| value.clone());
    drop(copy);
    live
}

fn inflate(reader: &PacbinReader, name: &str) -> Vec<u8> {
    let mut out = Vec::new();
    flate2::read::ZlibDecoder::new(reader.member_bytes(name).expect(name))
        .read_to_end(&mut out)
        .expect("zlib member");
    out
}

fn plan_parts(plan: &LoopProgramV4) -> Value {
    let size = std::mem::size_of::<rusticol_core::loop_v2::plan::Edge>() as i64;
    let mut edges = 0i64;
    let mut links = 0i64;
    let mut split_parts = 0i64;
    let mut split_links = 0i64;
    let mut split_targets = 0i64;
    let mut steps = 0i64;
    let mut leaves = 0i64;
    for family in &plan.families {
        steps += family.steps.len() as i64;
        leaves += family.leaves.len() as i64;
        for step in &family.steps {
            edges += step.edges.len() as i64;
            links += step.edges.iter().map(|e| e.links.len() as i64).sum::<i64>();
            if let Some(split) = &step.split {
                split_parts += split.parts.iter().map(|p| p.len() as i64).sum::<i64>();
                split_links += split
                    .parts
                    .iter()
                    .flatten()
                    .map(|p| p.links.len() as i64)
                    .sum::<i64>();
                split_targets += split
                    .groups
                    .iter()
                    .map(|g| g.targets.len() as i64)
                    .sum::<i64>();
            }
        }
    }
    let content_rows: i64 = plan
        .contents
        .iter()
        .flat_map(|c| c.terms.iter())
        .map(|t| t.rows.len() as i64)
        .sum();
    json!({
        "counts": {
            "families": plan.families.len(), "steps": steps, "edges": edges,
            "edge_links": links, "split_parts": split_parts, "split_links": split_links,
            "split_targets": split_targets, "leaves": leaves, "contents": plan.contents.len(),
            "term_rows": content_rows, "vertices": plan.vertices.len(),
            "seeds": plan.seeds.len(), "seed_classes": plan.seed_classes.len(),
            "products": plan.products.len(), "edge_bytes": size,
        },
        "deep_mib": {
            "families": mib(deep(&plan.families)),
            "contents": mib(deep(&plan.contents)),
            "vertices": mib(deep(&plan.vertices)),
            "kernels": mib(deep(&plan.kernels)),
            "seed_classes": mib(deep(&plan.seed_classes)),
            "seeds": mib(deep(&plan.seeds)),
            "products": mib(deep(&plan.products)),
            "forward": mib(deep(&plan.forward)),
            "merged": mib(deep(&plan.merged)),
            "read_rows": mib(deep(&plan.read_rows)),
            "block_rows": mib(deep(&plan.block_rows)),
            "factors": mib(deep(&plan.factors)),
            "currents": mib(deep(&plan.currents)),
            "vertex_instances": mib(deep(&plan.vertex_instances)),
        },
    })
}

fn report(path: &str) -> Value {
    let reader = PacbinReader::open(path).expect("PACBIN");
    let plan_json = read_loop_program_v4_json(&reader).expect("plan member");
    let (plan, plan_live, plan_peak) =
        measured(|| LoopProgramV4::load_with(&plan_json, FormPolicy::Rule).expect("plan"));
    let plan_bytes = plan_json.len();
    drop(plan_json);
    let parts = plan_parts(&plan);
    let prefix = "point-inputs/";
    let mut kernels = Vec::new();
    let mut planes = Vec::new();
    while reader
        .member(&format!("{prefix}kernel-{}.state", kernels.len()))
        .is_ok()
    {
        let k = kernels.len();
        kernels.push(
            reader
                .member_bytes(&format!("{prefix}kernel-{k}.state"))
                .unwrap()
                .to_vec(),
        );
        planes.push(
            reader
                .member_bytes(&format!("{prefix}kernel-{k}.symjit"))
                .unwrap()
                .to_vec(),
        );
    }
    kernels.push(
        reader
            .member_bytes(&format!("{prefix}parameters.state"))
            .unwrap()
            .to_vec(),
    );
    planes.push(
        reader
            .member_bytes(&format!("{prefix}parameters.symjit"))
            .unwrap()
            .to_vec(),
    );
    let binary = format!("{prefix}tree-program.bin.zlib");
    let (source, form) = if reader.member(&binary).is_ok() {
        (inflate(&reader, &binary), "binary")
    } else {
        (
            inflate(&reader, &format!("{prefix}tree-program.json.zlib")),
            "json",
        )
    };
    let (trees, trees_live, trees_peak) = measured(|| {
        TreeProgram::load_with_components(&source, &kernels, Some(&planes), &plan, None)
            .expect("tree program")
    });
    // The binary64 runtime an evaluation builds once (kernels, interaction
    // and colour weights, interference metrics; SymJIT code is not counted).
    let (runtime, runtime_live, _) = measured(|| trees.runtime::<f64>(&0.0).expect("runtime"));
    drop(runtime);
    let out = json!({
        "path": path,
        "runtime_live_mib": mib(runtime_live),
        "plan_json_mib": mib(plan_bytes as i64),
        "plan_live_mib": mib(plan_live),
        "plan_peak_mib": mib(plan_peak),
        "plan": parts,
        "trees_form": form,
        "trees_source_mib": mib(source.len() as i64),
        "trees_live_mib": mib(trees_live),
        "trees_peak_mib": mib(trees_peak),
    });
    drop(trees);
    out
}

#[test]
#[ignore = "a measuring tool: set LOOP_LOAD_MEMORY_PACBIN"]
fn load_memory_of_stored_processes() {
    let paths = std::env::var("LOOP_LOAD_MEMORY_PACBIN").expect("LOOP_LOAD_MEMORY_PACBIN");
    for path in paths.split(':').filter(|p| !p.is_empty()) {
        println!("{}", report(path));
    }
}
