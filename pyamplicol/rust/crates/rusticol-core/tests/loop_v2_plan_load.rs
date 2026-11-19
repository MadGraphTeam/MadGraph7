// SPDX-License-Identifier: 0BSD
//! The plan loader's parallel family load and shared tables: the program is
//! the sequential load's at any thread count (the families merged in family
//! order), the first error in family order is the one returned, and the
//! shared segment layouts are keyed by their contents' dimensions.

#![cfg(feature = "loop-v2")]

use rusticol_core::loop_v2::plan::binary::{PlanModel, program_digest, read_binary};
use rusticol_core::loop_v2::plan::{LoopProgramV4, force_pipeline_for_tests};
use serde_json::{Value, json};

/// A plan of `families` families of one seeded trace step each, over a fused
/// content and a split vertex content (alternating), each family's leaf of
/// its own weight: the leaf weights, the products tables and the interned
/// link lists are numbered by first use, so a merge out of family order
/// changes the program.
fn families_plan(families: usize) -> Value {
    let mut fused = json!({
        "d_in": 1, "d_out": 1, "degree": 0, "rows": 1, "mask": 1,
        "row_helicities": [[1]], "fused": [[0, 0, 0, 0, 0, 0]], "terms": [],
    });
    let vertex = json!({
        "d_in": 1, "d_out": 1, "degree": 0, "rows": 1, "mask": 1,
        "row_helicities": [[1]], "fused": [[0, 0, 0, 0, 0, 0]],
        "terms": [{
            "slots": [], "dimensions": [], "bindings": [], "vertex": 0,
            "split": [{"kernel": 0, "inputs": [], "entries": [[0, 0]]}],
            "rows": [[0, []]],
        }],
    });
    fused["terms"] = json!([]);
    let step = |f: usize| {
        let edges: Vec<Value> = (0..1 + f % 3)
            .map(|e| json!({"leaf": 0, "child": -1, "factor": (e + f) % 2}))
            .collect();
        json!({
            "seeded": true, "trace": 0, "out": 0, "d_alpha": 1, "d_out": 1, "degree": 0,
            "rank_in": 0, "edges": edges,
        })
    };
    json!({
        "schema": 4, "legs": 1, "rows": 1, "helicities": [[1]], "columns": 1, "average": 1,
        "born_components": 1, "constants": 0, "masses": 1, "block_rows": [],
        "kernels": [{"inputs": [], "outputs": [[]]}],
        "contents": [fused, vertex], "factors": [[0, 0], [1, 0]],
        "seed_classes": [], "seeds": [{"classes": []}],
        "families": (0..families)
            .map(|f| json!({
                "sequence": [[1, 0]], "rank": 1 + f % 2, "orders": [0],
                "leaves": [{"seed": 0, "weight": format!("{}/{}", 1 + f % 7, 2 + f)}],
                "steps": [step(f)],
            }))
            .collect::<Vec<_>>(),
        "forward": null, "merged": null,
    })
}

/// The tests that switch the forced pipeline run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn on_threads<R: Send>(threads: usize, run: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("pool")
        .install(run)
}

fn load(plan: &Value) -> Result<LoopProgramV4, String> {
    LoopProgramV4::load(&serde_json::to_vec(plan).unwrap()).map_err(|e| e.to_string())
}

#[test]
fn families_merge_in_family_order_on_any_thread_count() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let plan = families_plan(400);
    let sequential = program_digest(&load(&plan).unwrap());
    force_pipeline_for_tests(true);
    for threads in [2, 3, 8] {
        for _ in 0..3 {
            let digest = on_threads(threads, || program_digest(&load(&plan).unwrap()));
            assert_eq!(digest, sequential, "{threads} threads");
        }
    }
    force_pipeline_for_tests(false);
    for threads in [2, 8] {
        let digest = on_threads(threads, || program_digest(&load(&plan).unwrap()));
        assert_eq!(
            digest, sequential,
            "{threads} threads, by the families' work"
        );
    }
}

#[test]
fn the_first_error_in_family_order_is_returned_on_any_thread_count() {
    let _serial = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    let mut plan = families_plan(400);
    // A later family with a bad leaf seed, an earlier one with a bad reading.
    plan["families"][350]["leaves"][0]["seed"] = json!(7);
    plan["families"][120]["sequence"] = json!([[0, 0]]);
    let sequential = load(&plan).unwrap_err();
    assert!(sequential.contains("family reading"), "{sequential}");
    force_pipeline_for_tests(true);
    for threads in [2, 3, 8] {
        let error = on_threads(threads, || load(&plan).unwrap_err());
        assert_eq!(error, sequential, "{threads} threads");
    }
    force_pipeline_for_tests(false);
}

#[test]
fn shared_fused_lists_keep_each_contents_layouts() {
    // Two contents of one fused list (stored once in the member, shared when
    // decoded) and different output dimensions: each keeps its own layouts.
    let content = |d_out: usize| {
        json!({
            "d_in": 1, "d_out": d_out, "degree": 0, "rows": 0, "mask": 0,
            "row_helicities": [], "fused": [[0, 0, 0, 0, 0, 0]], "terms": [],
        })
    };
    let plan = json!({
        "schema": 4, "legs": 1, "rows": 1, "helicities": [[1]], "columns": 1, "average": 1,
        "born_components": 1, "constants": 0, "masses": 1, "block_rows": [], "kernels": [],
        "contents": [content(1), content(2), content(1)], "factors": [], "seed_classes": [],
        "seeds": [], "families": [], "forward": null, "merged": null,
    });
    let member = PlanModel::from_json(&serde_json::to_vec(&plan).unwrap())
        .unwrap()
        .binary()
        .unwrap();
    let program = read_binary(&member).unwrap();
    let seeded: Vec<usize> = program
        .contents
        .iter()
        .map(|c| c.seeded.out_dimension())
        .collect();
    let open: Vec<usize> = program
        .contents
        .iter()
        .map(|c| c.open.out_dimension())
        .collect();
    assert_eq!(seeded, [1, 2, 1]);
    assert_eq!(open, [1, 2, 1]);
    assert!(std::sync::Arc::ptr_eq(
        &program.contents[0].seeded,
        &program.contents[2].seeded
    ));
}
