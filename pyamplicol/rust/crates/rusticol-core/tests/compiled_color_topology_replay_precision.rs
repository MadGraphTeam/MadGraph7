// SPDX-License-Identifier: 0BSD
//! Double-double and arbitrary-precision totals of a compiled full-colour
//! pure-gluon artifact, whose amplitudes use the contracted-color topology
//! replay. The momenta are exact massless integer points, so ratios of totals
//! between points cancel the couplings and equal the exact tree-level ratios
//! of 9/2 (3 - tu/s² - su/t² - st/u²).
//!
//! Supply the artifact with
//! `pyamplicol generate 'g g > g g' DIR --color-accuracy full --execution-mode compiled`
//! and `RUSTICOL_COMPILED_COLOR_REPLAY_ARTIFACT=DIR`.
#![cfg(all(feature = "symbolica-runtime", feature = "f64-symjit"))]

use rusticol_core::NativeRuntime;
use std::path::PathBuf;
use symbolica::domains::float::{Float, Real, RealLike};

const BITS: u32 = 400;

/// (E, a, b, c) with a² + b² + c² = E²: outgoing (E, ±a, ±b, ±c), beams (E, 0, 0, ±E).
const POINTS: [[i64; 4]; 3] = [
    [700, 200, 300, 600],
    [450, 50, 200, 400],
    [2200, 1200, 400, -1800],
];

fn momenta(point: [i64; 4]) -> Vec<i64> {
    let [e, a, b, c] = point;
    vec![e, 0, 0, e, e, 0, 0, -e, e, a, b, c, e, -a, -b, -c]
}

fn tree_level_shape(point: [i64; 4]) -> Float {
    let [e, _, _, c] = point;
    let f = |x: i64| Float::with_val(BITS, x);
    let (s, t, u) = (f(4 * e * e), f(-2 * e * (e - c)), f(-2 * e * (e + c)));
    f(3) - t.clone() * &u / (s.clone() * &s)
        - s.clone() * &u / (t.clone() * &t)
        - s.clone() * &t / (u.clone() * &u)
}

fn relative_difference(a: &Float, b: &Float) -> f64 {
    (a.clone() / b - Float::with_val(BITS, 1)).norm().to_f64()
}

#[test]
fn compiled_color_topology_replay_totals_in_double_double_and_arbitrary_precision() {
    let Some(path) = std::env::var_os("RUSTICOL_COMPILED_COLOR_REPLAY_ARTIFACT") else {
        return;
    };
    let path = PathBuf::from(path);
    let execution = std::fs::read_dir(path.join("processes"))
        .unwrap()
        .map(|entry| entry.unwrap().path().join("execution.json"))
        .find(|execution| execution.is_file())
        .map(|execution| std::fs::read_to_string(execution).unwrap())
        .expect("the fixture has one process");
    assert!(
        execution.contains("\"color_topology_replay\":{"),
        "the fixture must use the contracted-color topology replay"
    );
    let mut runtime = NativeRuntime::load(&path, None, None).unwrap();
    assert_eq!(runtime.metadata().execution_mode, "compiled");

    let flat: Vec<i64> = POINTS.iter().flat_map(|point| momenta(*point)).collect();
    let binary64 = runtime
        .evaluate_f64(&flat.iter().map(|x| *x as f64).collect::<Vec<_>>(), POINTS.len())
        .unwrap();
    let decimal: Vec<String> = flat.iter().map(ToString::to_string).collect();
    for (digits, tolerance) in [(32, 1e-27), (64, 1e-58)] {
        let values = runtime
            .evaluate_with_precision(&decimal, POINTS.len(), digits)
            .unwrap()
            .values
            .iter()
            .map(|value| Float::parse(value, Some(BITS)).unwrap())
            .collect::<Vec<_>>();
        for (value, reference) in values.iter().zip(&binary64) {
            assert!((value.to_f64() / reference - 1.0).abs() <= 1e-12);
        }
        for index in 1..POINTS.len() {
            let ratio = values[index].clone() / &values[0];
            let exact = tree_level_shape(POINTS[index]) / &tree_level_shape(POINTS[0]);
            let difference = relative_difference(&ratio, &exact);
            assert!(
                difference <= tolerance,
                "{digits} digits: ratio at point {index} differs by {difference:e}"
            );
        }
    }
}

/// Resolved (per-helicity) evaluation of the same artifact in double-double and arbitrary
/// precision: the components sum to the unresolved total of the same precision, agree with the
/// binary64 components, are parity symmetric (pure QCD), and a helicity selector returns the
/// selected component alone.
#[test]
fn compiled_color_topology_replay_resolved_in_double_double_and_arbitrary_precision() {
    let Some(path) = std::env::var_os("RUSTICOL_COMPILED_COLOR_REPLAY_ARTIFACT") else {
        return;
    };
    let mut runtime = NativeRuntime::load(PathBuf::from(path), None, None).unwrap();
    let flat: Vec<i64> = POINTS.iter().flat_map(|point| momenta(*point)).collect();
    let binary64 = runtime
        .evaluate_resolved_f64(
            &flat.iter().map(|x| *x as f64).collect::<Vec<_>>(),
            POINTS.len(),
            None,
            None,
        )
        .unwrap();
    let helicity_count = binary64.helicity_ids.len();
    assert_eq!(
        binary64.color_ids.len(),
        1,
        "one contracted colour component"
    );
    let decimal: Vec<String> = flat.iter().map(ToString::to_string).collect();
    let parse = |value: &String| Float::parse(value, Some(BITS)).unwrap();
    for (digits, tolerance) in [(32, 1e-27), (64, 1e-58)] {
        let totals = runtime
            .evaluate_with_precision(&decimal, POINTS.len(), digits)
            .unwrap()
            .values
            .iter()
            .map(parse)
            .collect::<Vec<_>>();
        let resolved = runtime
            .evaluate_resolved_with_precision(&decimal, POINTS.len(), digits, None, None)
            .unwrap();
        assert_eq!(resolved.helicity_ids, binary64.helicity_ids);
        for (point, total) in totals.iter().enumerate() {
            let components = &resolved.values[point * helicity_count..(point + 1) * helicity_count];
            let mut sum = Float::with_val(BITS, 0);
            for (index, component) in components.iter().enumerate() {
                let value = parse(component);
                let reference = binary64.values[point * helicity_count + index];
                assert!(
                    (value.to_f64() - reference).abs() <= 1e-12 * total.to_f64(),
                    "{digits} digits, point {point}, {}: {component} vs binary64 {reference:e}",
                    resolved.helicity_ids[index]
                );
                sum += &value;
            }
            let difference = relative_difference(&sum, total);
            assert!(
                difference <= tolerance,
                "{digits} digits: the components of point {point} sum to the total within {difference:e}"
            );
            assert!(relative_difference(&parse(&resolved.totals[point]), total) <= tolerance);
            for (index, id) in resolved.helicity_ids.iter().enumerate() {
                let flipped = flip(id);
                let partner = resolved
                    .helicity_ids
                    .iter()
                    .position(|other| *other == flipped)
                    .unwrap();
                assert_eq!(components[index], components[partner], "parity of {id}");
            }
        }
        let selected = resolved
            .helicity_ids
            .iter()
            .position(|id| id == "h:-1,-1,+1,+1")
            .unwrap();
        let single = runtime
            .evaluate_resolved_with_precision(
                &decimal,
                POINTS.len(),
                digits,
                Some(&[resolved.helicity_ids[selected].clone()]),
                None,
            )
            .unwrap();
        for point in 0..POINTS.len() {
            assert_eq!(
                single.values[point],
                resolved.values[point * helicity_count + selected]
            );
        }
    }
}

/// The helicity ID with every sign flipped (`h:-1,+1,...`).
fn flip(id: &str) -> String {
    let values = id
        .strip_prefix("h:")
        .unwrap()
        .split(',')
        .map(|value| match value {
            "+1" => "-1",
            "-1" => "+1",
            other => other,
        });
    format!("h:{}", values.collect::<Vec<_>>().join(","))
}
