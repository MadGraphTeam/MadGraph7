// SPDX-License-Identifier: 0BSD
//! The safe Rust SDK's loop-induced selections (`Runtime::evaluate_selected`:
//! the amplitude-basis anchors, global and per-point helicity selections; m-lit
//! lit-sdk-b) on a generated artifact, for a bitwise comparison with the C ABI
//! and the Python API. It needs an artifact and a feed, so it is ignored by
//! default and run explicitly by `tests/unit/test_loop_v2_sdk_selection.py`
//! (nightly, given a private cargo target directory):
//! `PYAMPLICOL_V4_SELECTION_FEED=<feed.json> PYAMPLICOL_V4_SELECTION_OUTPUT=<out.txt>
//! cargo test --release --test v4_selection -- --ignored`. The feed names the
//! artifact, the process, the points (doubles as IEEE bit patterns), the
//! helicity IDs and the per-point rows; the output has
//! `examples/selection_v4.c`'s format with each double as `b:` and its bits.
use rusticol_loop_sdk::{Arithmetic, PointStatus, Runtime, Selection};
use std::fmt::Write;

#[derive(serde::Deserialize)]
struct Feed {
    artifact: String,
    process: String,
    /// Per point the momenta (4 per leg) and then mu_R², as bit patterns.
    points: Vec<Vec<u64>>,
    ids: Vec<String>,
    by_point: Vec<u32>,
}

#[test]
#[ignore = "needs PYAMPLICOL_V4_SELECTION_FEED and PYAMPLICOL_V4_SELECTION_OUTPUT; run with --ignored"]
fn v4_selections_follow_the_c_abi() {
    let feed: Feed = serde_json::from_str(
        &std::fs::read_to_string(
            std::env::var("PYAMPLICOL_V4_SELECTION_FEED")
                .expect("set PYAMPLICOL_V4_SELECTION_FEED"),
        )
        .unwrap(),
    )
    .unwrap();
    let output = std::env::var("PYAMPLICOL_V4_SELECTION_OUTPUT")
        .expect("set PYAMPLICOL_V4_SELECTION_OUTPUT");
    let mut runtime = Runtime::open(&feed.artifact, Some(&feed.process), None, "auto").unwrap();
    runtime.set_arithmetic(Arithmetic::Double).unwrap();
    let resolution = |name: &str| Some(name.to_owned());
    let calls: Vec<(&str, bool, Selection)> = vec![
        (
            "anchors",
            true,
            Selection {
                color_resolution: resolution("amplitude-basis"),
                ..Selection::default()
            },
        ),
        (
            "one",
            false,
            Selection {
                helicities: Some(feed.ids[..1].to_vec()),
                ..Selection::default()
            },
        ),
        (
            "subset",
            false,
            Selection {
                helicities: Some(feed.ids.clone()),
                ..Selection::default()
            },
        ),
        (
            "resolved-subset",
            true,
            Selection {
                helicities: Some(feed.ids.clone()),
                color_resolution: resolution("contracted"),
                ..Selection::default()
            },
        ),
        (
            "anchors-subset",
            true,
            Selection {
                helicities: Some(feed.ids.clone()),
                color_resolution: resolution("amplitude-basis"),
                ..Selection::default()
            },
        ),
        (
            "by-point",
            false,
            Selection {
                helicity_by_point: Some(feed.by_point.clone()),
                ..Selection::default()
            },
        ),
        (
            "colour",
            false,
            Selection {
                color_flows: Some(vec!["1".to_owned()]),
                ..Selection::default()
            },
        ),
    ];
    let mut momenta = Vec::new();
    let mut scales = Vec::new();
    for point in &feed.points {
        let (scale, legs) = point.split_last().unwrap();
        momenta.extend(legs.iter().map(|&b| f64::from_bits(b)));
        scales.push(f64::from_bits(*scale));
    }
    let ids: Vec<u64> = (0..feed.points.len() as u64).collect();
    let mut out = String::new();
    for (name, resolved, selection) in calls {
        writeln!(out, "call {name}").unwrap();
        match runtime.evaluate_selected(&momenta, &ids, &scales, resolved, None, &selection) {
            Ok(result) => {
                let statuses: Vec<String> = result
                    .statuses
                    .iter()
                    .map(|s| match s {
                        PointStatus::Failed => "3".to_owned(),
                        other => (*other as u32).to_string(),
                    })
                    .collect();
                writeln!(out, "statuses {}", statuses.join(" ")).unwrap();
                write!(out, "virtual").unwrap();
                for value in &result.virtual_values {
                    write!(out, " b:{:016X}", value.to_bits()).unwrap();
                }
                writeln!(out).unwrap();
                writeln!(out, "decimal {}", result.decimal_json().unwrap()).unwrap();
                writeln!(
                    out,
                    "axes {}",
                    serde_json::json!({
                        "helicity_ids": result.helicity_ids,
                        "color_ids": result.color_ids,
                    })
                )
                .unwrap();
            }
            Err(error) => writeln!(out, "refused {error}").unwrap(),
        }
    }
    std::fs::write(output, out).unwrap();
}
