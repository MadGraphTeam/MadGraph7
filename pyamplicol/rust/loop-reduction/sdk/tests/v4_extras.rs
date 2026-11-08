// SPDX-License-Identifier: 0BSD
//! The safe Rust SDK's plan v4 extras (resolved, colour- and spin-correlated,
//! double-double and arbitrary-precision evaluations) on a generated
//! artifact, for a bitwise comparison with the Python API. It needs an
//! artifact and a feed, so it is ignored by default and run explicitly by
//! `tests/unit/test_loop_v2_sdk_extras.py` (nightly, given a private cargo
//! target directory): `PYAMPLICOL_V4_EXTRAS_FEED=<feed.json>
//! PYAMPLICOL_V4_EXTRAS_OUTPUT=<out.txt> cargo test --release --test
//! v4_extras -- --ignored`. The feed names the artifact, the process, the
//! points (doubles as IEEE bit patterns) and the spin vectors; the output has
//! `examples/extras_v4.c`'s format with each double as `b:` and its bits.
use rusticol_loop_sdk::{Arithmetic, Complex64, Poles, Runtime, SpinSources};
use std::fmt::Write;

#[derive(serde::Deserialize)]
struct Feed {
    artifact: String,
    process: String,
    /// Per point the momenta (4 per leg) and then mu_R², as bit patterns.
    points: Vec<Vec<u64>>,
    spin_legs: Vec<usize>,
    /// Per spin leg its four complex components (re, im), as bit patterns.
    spin: Vec<Vec<u64>>,
}

const CALLS: [(&str, &str, Option<u32>, &str, u8, usize); 13] = [
    ("ordinary", "double", None, "all", 0, 2),
    ("correlated", "double", None, "all", 1, 2),
    ("correlated-spin", "double", None, "all", 2, 2),
    ("correlated-uv", "double", None, "uv", 1, 2),
    ("correlated-none", "double", None, "none", 1, 2),
    ("resolved", "double", None, "all", 3, 2),
    ("resolved-ir", "double", None, "ir", 3, 2),
    ("ordinary-dd", "double-double", None, "all", 0, 1),
    ("correlated-spin-dd", "double-double", None, "all", 2, 1),
    ("resolved-dd", "double-double", None, "all", 3, 1),
    ("ordinary-ap", "arbitrary", Some(32), "all", 0, 1),
    ("correlated-spin-ap", "arbitrary", Some(32), "all", 2, 1),
    ("resolved-ap", "arbitrary", Some(32), "all", 3, 1),
];

fn bits(out: &mut String, values: impl IntoIterator<Item = f64>) {
    for value in values {
        write!(out, " b:{:016X}", value.to_bits()).unwrap();
    }
}

#[test]
#[ignore = "needs PYAMPLICOL_V4_EXTRAS_FEED and PYAMPLICOL_V4_EXTRAS_OUTPUT; run with --ignored"]
fn v4_extras_follow_the_python_api() {
    let feed: Feed = serde_json::from_str(
        &std::fs::read_to_string(
            std::env::var("PYAMPLICOL_V4_EXTRAS_FEED").expect("set PYAMPLICOL_V4_EXTRAS_FEED"),
        )
        .unwrap(),
    )
    .unwrap();
    let output =
        std::env::var("PYAMPLICOL_V4_EXTRAS_OUTPUT").expect("set PYAMPLICOL_V4_EXTRAS_OUTPUT");
    let mut runtime = Runtime::open(&feed.artifact, Some(&feed.process), None, "auto").unwrap();
    let vectors: Vec<Complex64> = feed
        .spin
        .iter()
        .flat_map(|leg| {
            leg.chunks(2).map(|pair| Complex64 {
                real: f64::from_bits(pair[0]),
                imag: f64::from_bits(pair[1]),
            })
        })
        .collect();
    let mut out = String::new();
    for (name, arithmetic, digits, poles, kind, count) in CALLS {
        let n = count.min(feed.points.len());
        let mut momenta = Vec::new();
        let mut scales = Vec::new();
        for point in &feed.points[..n] {
            let (last, momentum) = point.split_last().unwrap();
            momenta.extend(momentum.iter().map(|b| f64::from_bits(*b)));
            scales.push(f64::from_bits(*last));
        }
        let ids: Vec<u64> = (0..n as u64).collect();
        runtime
            .set_arithmetic(Arithmetic::parse(arithmetic, digits).unwrap())
            .unwrap();
        writeln!(out, "call {name}").unwrap();
        if let Err(refused) = runtime.set_poles(Poles::parse(poles).unwrap()) {
            // A pole selection the process does not serve (loop-induced:
            // "all" and "none").
            writeln!(out, "refused {refused}").unwrap();
            continue;
        }
        if kind == 0 || kind == 3 {
            let result = runtime
                .evaluate(&momenta, &ids, &scales, kind == 3, None)
                .unwrap();
            out.push_str("born");
            bits(&mut out, result.born.iter().copied());
            out.push_str("\nvirtual");
            bits(&mut out, result.virtual_values.iter().copied());
            writeln!(out, "\ndecimal {}", result.decimal_json().unwrap()).unwrap();
        } else {
            let result = if kind == 2 {
                let sources = SpinSources {
                    legs: &feed.spin_legs,
                    values: &vectors,
                };
                runtime
                    .evaluate_correlated_with_sources(&momenta, &ids, &scales, sources, None)
                    .unwrap()
            } else {
                runtime
                    .evaluate_correlated(&momenta, &ids, &scales, None)
                    .unwrap()
            };
            out.push_str("born");
            bits(&mut out, result.born.iter().flat_map(|z| [z.real, z.imag]));
            out.push_str("\nvirtual");
            bits(
                &mut out,
                result.virtual_values.iter().flat_map(|z| [z.real, z.imag]),
            );
            writeln!(out, "\ndecimal {}", result.decimal_json().unwrap()).unwrap();
        }
    }
    std::fs::write(output, out).unwrap();
}
