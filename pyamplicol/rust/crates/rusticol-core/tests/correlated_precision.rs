// SPDX-License-Identifier: 0BSD
//! Double-double and arbitrary-precision correlated Born quantities
//! (`NativeRuntime::evaluate_correlated_many_with_precision`).
//!
//! The fixture holds built-in SM artifacts with every dipole and gluon spin
//! class declared, exact integer massless points and the Python exact
//! correlated path's values at those points. Build it with
//! `PYTHONPATH=src python rust/crates/rusticol-core/tests/fixtures/correlated_precision.py DIR`
//! and set `RUSTICOL_CORRELATED_PRECISION_FIXTURE=DIR`.
//!
//! Values are compared after dividing by each path's normalization: the native
//! paths keep the artifact's binary64 normalization constant (exactly, as
//! `evaluate_with_precision` does), the Python executor rebuilds it with Pi at
//! its working precision.
#![cfg(all(feature = "symbolica-runtime", feature = "f64-symjit"))]

use rusticol_core::{NativeDecimalCorrelatedRequest, NativeRuntime};
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use symbolica::domains::float::{Float, Real, RealLike};

/// Working precision of the test's own arithmetic.
const BITS: u32 = 600;
/// Decimal digits selecting double-double, about 256 bits (260) and about 200
/// bits (203) in rusticol's decimal interface.
const DOUBLE_DOUBLE: u32 = 32;
const BITS_256: u32 = 78;
const BITS_200: u32 = 61;

fn fixture() -> Option<PathBuf> {
    std::env::var_os("RUSTICOL_CORRELATED_PRECISION_FIXTURE").map(PathBuf::from)
}

fn reference(root: &std::path::Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(root.join("reference.json")).unwrap()).unwrap()
}

fn float(text: &str) -> Float {
    Float::parse(text, Some(BITS)).unwrap()
}

fn int(value: i64) -> Float {
    Float::with_val(BITS, value)
}

fn abs(value: &Float) -> f64 {
    value.norm().to_f64()
}

/// The runtime's normalization constant (binary64), exactly.
fn normalization(runtime: &NativeRuntime) -> Float {
    let state: Value = serde_json::from_str(&runtime.exact_runtime_state_json().unwrap()).unwrap();
    Float::with_val(BITS, state["normalization_factor"].as_f64().unwrap())
}

fn point_strings(point: &Value) -> Vec<String> {
    point
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|leg| leg.as_array().unwrap().iter().map(|x| x.to_string()))
        .collect()
}

fn request(record: &Value) -> NativeDecimalCorrelatedRequest {
    let vectors = record["spin_vectors"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(leg, vector)| {
            let components = std::array::from_fn(|mu| {
                let pair = &vector[mu];
                [
                    pair[0].as_str().unwrap().to_owned(),
                    pair[1].as_str().unwrap().to_owned(),
                ]
            });
            (leg.parse().unwrap(), vec![components])
        })
        .collect::<BTreeMap<_, _>>();
    NativeDecimalCorrelatedRequest {
        color_correlation: record["color_correlation"].as_str().unwrap().to_owned(),
        spin_vectors: Some(vectors),
    }
}

/// The native correlated values of one entry's requests at `digits`, divided
/// by the normalization: (real, imaginary) per request.
fn native_raw(runtime: &mut NativeRuntime, entry: &Value, digits: u32) -> Vec<[Float; 2]> {
    let requests: Vec<_> = entry["requests"]
        .as_array()
        .unwrap()
        .iter()
        .map(request)
        .collect();
    let evaluation = runtime
        .evaluate_correlated_many_with_precision(
            &point_strings(&entry["point"]),
            1,
            digits,
            &requests,
            None,
        )
        .unwrap();
    assert_eq!(evaluation.request_count, requests.len());
    let norm = normalization(runtime);
    evaluation
        .values
        .iter()
        .map(|[re, im]| [float(re) / &norm, float(im) / &norm])
        .collect()
}

fn ids(entry: &Value) -> Vec<String> {
    entry["requests"]
        .as_array()
        .unwrap()
        .iter()
        .map(|record| record["id"].as_str().unwrap().to_owned())
        .collect()
}

fn builtin(root: &std::path::Path, name: &str) -> NativeRuntime {
    NativeRuntime::load(root.join("builtin").join(name), Some(name), None).unwrap()
}

fn modulus([re, im]: &[Float; 2]) -> f64 {
    (re.clone() * re + im.clone() * im).sqrt().to_f64()
}

/// Largest complex |native - Python| / max(|Python|, 1e-3 |Python Born|) over
/// every request and point.
fn python_deviation(runtime: &mut NativeRuntime, entries: &[Value], digits: u32) -> f64 {
    let mut worst: f64 = 0.0;
    for entry in entries {
        let native = native_raw(runtime, entry, digits);
        let python: Vec<[Float; 2]> = entry["requests"]
            .as_array()
            .unwrap()
            .iter()
            .map(|record| {
                [
                    float(record["raw"][0].as_str().unwrap()),
                    float(record["raw"][1].as_str().unwrap()),
                ]
            })
            .collect();
        assert_eq!(ids(entry)[0], "born");
        let born = modulus(&python[0]);
        for (native, python) in native.iter().zip(&python) {
            let difference = [
                native[0].clone() - &python[0],
                native[1].clone() - &python[1],
            ];
            worst = worst.max(modulus(&difference) / modulus(python).max(1e-3 * born));
        }
    }
    worst
}

#[test]
fn correlated_values_match_the_python_exact_path() {
    let Some(root) = fixture() else {
        return;
    };
    let reference = reference(&root);
    for (name, process) in reference["builtin"].as_object().unwrap() {
        let entries = process["entries"].as_array().unwrap();
        let mut runtime = builtin(&root, name);
        for (digits, tolerance) in [(DOUBLE_DOUBLE, 1e-28), (BITS_256, 1e-60)] {
            let deviation = python_deviation(&mut runtime, entries, digits);
            println!("{name}: {digits} digits: max deviation from Python {deviation:.2e}");
            assert!(
                deviation <= tolerance,
                "{name} at {digits} digits deviates by {deviation:e} from the Python exact path"
            );
        }
    }
}

/// Sum_{l != k} <T_k . T_l> = -C_k <1> for every leg k (all-outgoing charges).
#[test]
fn dipoles_conserve_colour_at_200_bits() {
    let Some(root) = fixture() else {
        return;
    };
    let reference = reference(&root);
    let casimirs = BTreeMap::from([
        ("ud_udg", ["4/3", "4/3", "4/3", "4/3", "3"]),
        ("uux_ggg", ["4/3", "4/3", "3", "3", "3"]),
        ("gg_ggg", ["3", "3", "3", "3", "3"]),
    ]);
    for (name, process) in reference["builtin"].as_object().unwrap() {
        let mut runtime = builtin(&root, name);
        let mut worst: f64 = 0.0;
        for entry in process["entries"].as_array().unwrap() {
            let values = native_raw(&mut runtime, entry, BITS_200);
            let ids = ids(entry);
            let born = values[0][0].clone();
            for (k, casimir) in casimirs[name.as_str()].iter().enumerate() {
                let (numerator, denominator) = casimir.split_once('/').unwrap_or((casimir, "1"));
                let casimir = float(numerator) / &float(denominator);
                let mut sum = casimir.clone() * &born;
                for (id, value) in ids.iter().zip(&values) {
                    let pair = id.strip_prefix('T').and_then(|legs| {
                        let legs: Vec<usize> = legs
                            .chars()
                            .map(|leg| leg.to_digit(10).unwrap() as usize)
                            .collect();
                        (legs.len() == 2).then(|| (legs[0], legs[1]))
                    });
                    if let Some((left, right)) = pair
                        && (left == k + 1 || right == k + 1)
                    {
                        sum += value[0].clone();
                    }
                }
                worst = worst.max(abs(&sum) / abs(&(casimir * &born)));
            }
        }
        println!("{name}: colour conservation at 203 bits: max deviation {worst:.2e}");
        assert!(
            worst <= 1e-55,
            "{name}: colour conservation fails by {worst:e}"
        );
    }
}

/// Sum over two orthogonal transverse vectors e_i of |M(e_i)|^2 / |e_i|^2 is
/// the Born, for every gluon leg.
#[test]
fn transverse_spin_vectors_complete_the_born_at_200_bits() {
    let Some(root) = fixture() else {
        return;
    };
    let reference = reference(&root);
    for (name, process) in reference["builtin"].as_object().unwrap() {
        let mut runtime = builtin(&root, name);
        let mut worst: f64 = 0.0;
        for entry in process["entries"].as_array().unwrap() {
            let values = native_raw(&mut runtime, entry, BITS_200);
            let born = values[0][0].clone();
            let mut sums: BTreeMap<String, Float> = BTreeMap::new();
            for (record, value) in entry["requests"].as_array().unwrap().iter().zip(&values) {
                let id = record["id"].as_str().unwrap();
                let Some(leg) = id
                    .strip_prefix("spin")
                    .and_then(|rest| rest.strip_suffix("_0").or(rest.strip_suffix("_1")))
                else {
                    continue;
                };
                let vector = &record["spin_vectors"][leg];
                let norm = (1..4).fold(int(0), |sum, mu| {
                    let component = float(vector[mu][0].as_str().unwrap());
                    sum + component.clone() * &component
                });
                let term = value[0].clone() / &norm;
                sums.entry(leg.to_owned())
                    .and_modify(|sum| *sum += term.clone())
                    .or_insert(term);
            }
            assert_eq!(
                sums.len(),
                process["spin_legs"].as_array().unwrap().len(),
                "{name}: every gluon leg has two transverse vectors"
            );
            for sum in sums.values() {
                worst = worst.max(abs(&(sum.clone() - &born)) / abs(&born));
            }
        }
        println!("{name}: spin completeness at 203 bits: max deviation {worst:.2e}");
        assert!(
            worst <= 1e-55,
            "{name}: spin completeness fails by {worst:e}"
        );
    }
}
