// SPDX-License-Identifier: 0BSD
//! Double-double and arbitrary-precision correlated Born quantities
//! (`NativeRuntime::evaluate_correlated_many_with_precision`), trees whose
//! couplings are derived model parameters (`evaluate_with_precision`), and
//! model parameters and particle masses at a given precision
//! (`model_parameter_value_with_precision`, `particle_mass_with_precision`).
//!
//! The fixture holds built-in SM artifacts with every dipole and gluon spin
//! class declared, UFO SM artifacts (one set with a W) and built-in SM-HEFT
//! artifacts, exact integer massless points and the Python exact correlated
//! path's values at those points. Build it with
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

fn point_floats(point: &Value) -> Vec<[Float; 4]> {
    point
        .as_array()
        .unwrap()
        .iter()
        .map(|leg| std::array::from_fn(|mu| int(leg[mu].as_i64().unwrap())))
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

fn dot(p: &[Float; 4], q: &[Float; 4]) -> Float {
    p[0].clone() * &q[0] - p[1].clone() * &q[1] - p[2].clone() * &q[2] - p[3].clone() * &q[3]
}

/// s_ij = 2 p_i . p_j of all-outgoing massless momenta (beams reversed).
fn invariants(point: &[[Float; 4]]) -> Vec<Vec<Float>> {
    let outgoing: Vec<[Float; 4]> = point
        .iter()
        .enumerate()
        .map(|(leg, p)| {
            if leg < 2 {
                std::array::from_fn(|mu| -p[mu].clone())
            } else {
                p.clone()
            }
        })
        .collect();
    outgoing
        .iter()
        .map(|p| outgoing.iter().map(|q| int(2) * dot(p, q)).collect())
        .collect()
}

/// Sum over helicities and colours of |M|^2 / g^(2(n-2)) of massless
/// all-gluon (n = 4, 5) and u d -> u d (QCD) scattering.
fn analytic(process: &str, point: &[[Float; 4]]) -> Float {
    let s = invariants(point);
    match process {
        // 16 (s^2 + u^2) / t^2 (averaged: 4/9 (s^2 + u^2) / t^2).
        "ud_ud" => {
            let (s12, t, u) = (s[0][1].clone(), s[0][2].clone(), s[0][3].clone());
            int(16) * (s12.clone() * &s12 + u.clone() * &u) / (t.clone() * &t)
        }
        // 1152 (3 - tu/s^2 - su/t^2 - st/u^2) (averaged: 9/2 (...)).
        "gg_gg" => {
            let (s12, t, u) = (s[0][1].clone(), s[0][2].clone(), s[0][3].clone());
            int(1152)
                * (int(3)
                    - t.clone() * &u / (s12.clone() * &s12)
                    - s12.clone() * &u / (t.clone() * &t)
                    - s12.clone() * &t / (u.clone() * &u))
        }
        // sum |M|^2 = g^6 N^3 (N^2 - 1) sum_{sigma in S_4} sum_h |A(1, sigma)|^2
        // (no subleading colour at five gluons), with the colour-ordered
        // amplitudes of Tr(T^a T^b) = delta^ab: every five-gluon helicity
        // amplitude is MHV or anti-MHV, |A|^2 = s_ij^4 / (s_12 s_23 s_34 s_45
        // s_51) for the two negative (or two positive) helicities i, j, so
        // sum_h |A|^2 = 2 sum_{i<j} s_ij^4 / (...).
        "gg_ggg" => {
            let numerator = s
                .iter()
                .enumerate()
                .flat_map(|(i, row)| &row[i + 1..])
                .fold(int(0), |sum, s_ij| {
                    let square = s_ij.clone() * s_ij;
                    sum + square.clone() * &square
                });
            let mut denominator_sum = int(0);
            let mut order = [1usize, 2, 3, 4];
            permutations(&mut order, 0, &mut |order| {
                let chain = [0, order[0], order[1], order[2], order[3], 0];
                let product = chain
                    .windows(2)
                    .fold(int(1), |product, pair| product * &s[pair[0]][pair[1]]);
                denominator_sum += int(1) / &product;
            });
            int(2 * 27 * 8) * numerator * denominator_sum
        }
        _ => unreachable!(),
    }
}

fn permutations(values: &mut [usize; 4], start: usize, visit: &mut impl FnMut(&[usize; 4])) {
    if start == values.len() {
        visit(values);
        return;
    }
    for index in start..values.len() {
        values.swap(start, index);
        permutations(values, start + 1, visit);
        values.swap(start, index);
    }
}

/// UFO couplings are derived parameters (G = 2 sqrt(Pi aS), GC_*): at the
/// working precision the trees match the analytic results to 1e-60.
#[test]
fn ufo_trees_with_derived_couplings_match_analytic_results_at_256_bits() {
    let Some(root) = fixture() else {
        return;
    };
    let reference = reference(&root);
    let pi = int(1).pi();
    let mut failures = Vec::new();
    for (process, points, power) in [
        ("ud_ud", "points_2to2", 2),
        ("gg_gg", "points_2to2", 2),
        ("gg_ggg", "points_2to3", 3),
    ] {
        let mut runtime = NativeRuntime::load(root.join("ufo"), Some(process), None).unwrap();
        let (alpha_s, _) = runtime.model_parameter_value("aS").unwrap();
        let g_squared = int(4) * &pi * &Float::with_val(BITS, alpha_s);
        let coupling = (0..power).fold(int(1), |product, _| product * &g_squared);
        let points = reference[points].as_array().unwrap();
        let flat: Vec<String> = points.iter().flat_map(point_strings).collect();
        for digits in [DOUBLE_DOUBLE, BITS_256] {
            let values = runtime
                .evaluate_with_precision(&flat, points.len(), digits)
                .unwrap()
                .values;
            let norm = normalization(&runtime);
            let mut worst: f64 = 0.0;
            for (value, point) in values.iter().zip(points) {
                let expected = coupling.clone() * analytic(process, &point_floats(point));
                let raw = float(value) / &norm;
                worst = worst.max(abs(&(raw / &expected - int(1))));
            }
            println!("UFO {process}: {digits} digits: max relative deviation {worst:.2e}");
            let tolerance = if digits == DOUBLE_DOUBLE {
                1e-28
            } else {
                1e-60
            };
            if worst > tolerance {
                failures.push(format!("{process} at {digits} digits: {worst:e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "UFO trees deviate from the analytic results: {failures:?}"
    );
}

/// The built-in SM-HEFT coupling G_H is a derived parameter (G_mu scheme, with
/// the m_H/m_t corrections of the model): sum |M(g g -> H)|^2 = 4 |G_H|^2 m_H^4.
#[test]
fn heft_tree_with_a_derived_coupling_matches_the_analytic_result_at_256_bits() {
    let Some(root) = fixture() else {
        return;
    };
    let mut runtime = NativeRuntime::load(root.join("heft"), Some("gg_h"), None).unwrap();
    let parameter =
        |name: &str| Float::with_val(BITS, runtime.model_parameter_value(name).unwrap().0);
    let (gf, mz, aewm1, alpha_s, mh, mt) = (
        parameter("Gf"),
        parameter("MZ"),
        parameter("aEWM1"),
        parameter("aS"),
        parameter("MH"),
        parameter("MT"),
    );
    let pi = int(1).pi();
    let mz2 = mz.clone() * &mz;
    // m_W^2 = m_Z^2 / 2 + sqrt(m_Z^4 / 4 - pi m_Z^2 / (sqrt(2) G_F aEWM1)).
    let mw2 = (mz2.clone() * &mz2 / &int(4) - pi.clone() * &mz2 / (int(2).sqrt() * &gf * &aewm1))
        .sqrt()
        + mz2.clone() / &int(2);
    let sw2 = int(1) - mw2.clone() / &mz2;
    let r = mh.clone() * &mh / (mt.clone() * &mt);
    let corrections = int(1)
        + int(7) / &int(120) * &r
        + r.clone() * &r / &int(168)
        + int(13) / &int(16800) * &r * &r * &r;
    // |G_H| = aS / (3 sqrt(pi)) sqrt(1 / aEWM1) / (m_W sqrt(s_W^2)) (corrections).
    let coupling = alpha_s / &int(3) / &pi.clone().sqrt() * &(int(1) / &aewm1).sqrt() * corrections
        / &mw2.sqrt()
        / &sw2.sqrt();
    let s = mh.clone() * &mh;
    let expected = int(4) * &coupling * &coupling * &s * &s;
    let energy = format!("{:?}", mh.to_f64() / 2.0);
    let mass = format!("{:?}", mh.to_f64());
    let point: Vec<String> = [&energy, "0", "0", &energy, &energy, "0", "0"]
        .into_iter()
        .map(str::to_owned)
        .chain([
            format!("-{energy}"),
            mass,
            "0".into(),
            "0".into(),
            "0".into(),
        ])
        .collect();
    let mut failures = Vec::new();
    for (digits, tolerance) in [(DOUBLE_DOUBLE, 1e-28), (BITS_256, 1e-60)] {
        let value = runtime
            .evaluate_with_precision(&point, 1, digits)
            .unwrap()
            .values
            .remove(0);
        let deviation = abs(&(float(&value) / &normalization(&runtime) / &expected - int(1)));
        println!("HEFT g g > H: {digits} digits: relative deviation {deviation:.2e}");
        if deviation > tolerance {
            failures.push(format!("{digits} digits: {deviation:e}"));
        }
    }
    assert!(failures.is_empty(), "HEFT g g > H deviates: {failures:?}");
}

/// The G_mu-scheme W mass of the UFO `sm` from the runtime's binary64 inputs,
/// at the test's working precision:
/// m_W^2 = m_Z^2 / 2 + sqrt(m_Z^4 / 4 - pi m_Z^2 / (sqrt(2) G_F aEWM1)).
fn gmu_w_mass(runtime: &NativeRuntime) -> Float {
    let parameter =
        |name: &str| Float::with_val(BITS, runtime.model_parameter_value(name).unwrap().0);
    let (gf, mz, aewm1) = (parameter("Gf"), parameter("MZ"), parameter("aEWM1"));
    let mz2 = mz.clone() * &mz;
    ((mz2.clone() * &mz2 / &int(4) - int(1).pi() * &mz2 / (int(2).sqrt() * &gf * &aewm1)).sqrt()
        + mz2 / &int(2))
        .sqrt()
}

/// `model_parameter_value_with_precision`: binary64 at 16 digits, external
/// parameters promoted exactly, derived ones (here M_W) at the working
/// precision; the real and imaginary parts of an unknown name are refused.
#[test]
fn model_parameters_at_the_working_precision() {
    let Some(root) = fixture() else {
        return;
    };
    let mut runtime = NativeRuntime::load(root.join("ufo_w"), Some("udx_wpg"), None).unwrap();
    let exact = gmu_w_mass(&runtime);
    let (binary64, _) = runtime.model_parameter_value("MW").unwrap();
    let (sixteen, imaginary) = runtime
        .model_parameter_value_with_precision("MW", 16)
        .unwrap();
    assert_eq!(sixteen.parse::<f64>().unwrap(), binary64);
    assert_eq!(float(&imaginary), int(0));
    let mut failures = Vec::new();
    for (digits, tolerance) in [(DOUBLE_DOUBLE, 1e-31), (BITS_200, 1e-60), (BITS_256, 1e-77)] {
        let (value, _) = runtime
            .model_parameter_value_with_precision("MW", digits)
            .unwrap();
        let deviation = abs(&(float(&value) / &exact - int(1)));
        println!("M_W at {digits} digits: relative deviation {deviation:.2e}");
        if deviation > tolerance {
            failures.push(format!("{digits} digits: {deviation:e}"));
        }
    }
    let binary64_deviation = abs(&(Float::with_val(BITS, binary64) / &exact - int(1)));
    println!("M_W in binary64: relative deviation {binary64_deviation:.2e}");
    // An external parameter is its binary64 value, promoted exactly.
    let (mz, _) = runtime.model_parameter_value("MZ").unwrap();
    let (mz_digits, _) = runtime
        .model_parameter_value_with_precision("MZ", BITS_256)
        .unwrap();
    assert_eq!(float(&mz_digits), Float::with_val(BITS, mz));
    assert!(
        runtime
            .model_parameter_value_with_precision("NOT_A_PARAMETER", BITS_256)
            .is_err()
    );
    assert!(failures.is_empty(), "M_W deviates: {failures:?}");
}

/// A massive or massless momentum of mass `mass` and three-momentum `size n`
/// (n a unit vector).
fn on_shell(mass: &Float, size: &Float, n: &[Float; 3]) -> [Float; 4] {
    [
        (size.clone() * size + mass.clone() * mass).sqrt(),
        size.clone() * &n[0],
        size.clone() * &n[1],
        size.clone() * &n[2],
    ]
}

/// The two-body momentum in the rest frame of a system of mass `root_s`.
fn two_body_momentum(root_s: &Float, m1: &Float, m2: &Float) -> Float {
    let s = root_s.clone() * root_s;
    let (a, b) = (m1.clone() * m1, m2.clone() * m2);
    let kallen = s.clone() * &s + a.clone() * &a + b.clone() * &b
        - int(2) * (s.clone() * &a + s.clone() * &b + a.clone() * &b);
    kallen.sqrt() / (int(2) * root_s)
}

/// The boost taking a system at rest to the momentum `total` (mass `mass`).
fn boost(total: &[Float; 4], mass: &Float, p: &[Float; 4]) -> [Float; 4] {
    let gamma = total[0].clone() / mass;
    let beta: Vec<Float> = (1..4).map(|i| total[i].clone() / &total[0]).collect();
    let beta_p = (0..3).fold(int(0), |sum, i| sum + beta[i].clone() * &p[i + 1]);
    let beta2 = (0..3).fold(int(0), |sum, i| sum + beta[i].clone() * &beta[i]);
    let factor = if beta2 == int(0) {
        int(0)
    } else {
        (gamma.clone() - int(1)) * &beta_p / &beta2
    };
    let mut out = [
        gamma.clone() * (p[0].clone() + beta_p.clone()),
        int(0),
        int(0),
        int(0),
    ];
    for i in 0..3 {
        out[i + 1] =
            p[i + 1].clone() + factor.clone() * &beta[i] + gamma.clone() * &beta[i] * &p[0];
    }
    out
}

/// A centre-of-mass point (sqrt(s) = 500) of u d~ > w+ [h] [g] with the W of
/// mass `mw`, the H of mass `mh`, and its boost along z with beta = 3/5.
fn w_points(outgoing: &[&str], mw: &Float, mh: &Float) -> [Vec<[Float; 4]>; 2] {
    let root_s = int(500);
    let half = int(250);
    let zero = int(0);
    let n1 = [int(3) / &int(13), int(4) / &int(13), int(12) / &int(13)];
    let n2 = [int(2) / &int(7), int(-3) / &int(7), int(6) / &int(7)];
    let negate = |n: &[Float; 3]| [-n[0].clone(), -n[1].clone(), -n[2].clone()];
    let masses: Vec<Float> = outgoing
        .iter()
        .map(|name| match *name {
            "w+" => mw.clone(),
            "h" => mh.clone(),
            _ => zero.clone(),
        })
        .collect();
    let rest: Vec<[Float; 4]> = if outgoing.len() == 2 {
        let p = two_body_momentum(&root_s, &masses[0], &masses[1]);
        vec![
            on_shell(&masses[0], &p, &negate(&n1)),
            on_shell(&masses[1], &p, &n1),
        ]
    } else {
        // The gluon recoils against a system X of mass 300 that decays to W H.
        let mx = int(300);
        let p = two_body_momentum(&root_s, &mx, &zero);
        let total = on_shell(&mx, &p, &negate(&n1));
        let q = two_body_momentum(&mx, &masses[0], &masses[1]);
        vec![
            boost(&total, &mx, &on_shell(&masses[0], &q, &n2)),
            boost(&total, &mx, &on_shell(&masses[1], &q, &negate(&n2))),
            on_shell(&zero, &p, &n1),
        ]
    };
    let mut point = vec![
        [half.clone(), zero.clone(), zero.clone(), half.clone()],
        [half.clone(), zero.clone(), zero.clone(), -half.clone()],
    ];
    point.extend(rest);
    let (gamma, beta_gamma) = (int(5) / &int(4), int(3) / &int(4));
    let boosted = point
        .iter()
        .map(|[e, x, y, z]| {
            [
                gamma.clone() * e + beta_gamma.clone() * z,
                x.clone(),
                y.clone(),
                gamma.clone() * z + beta_gamma.clone() * e,
            ]
        })
        .collect();
    [point, boosted]
}

/// u d~ > w+ g, u d~ > w+ h and u d~ > w+ h g at a centre-of-mass point and
/// boosted along z, with the W and the H on the mass shells of
/// `particle_mass_with_precision`: every tree is Lorentz invariant at 200
/// bits. (The 16-digit mass is the binary64 one.)
#[test]
fn particles_on_the_working_precision_mass_shell_are_lorentz_invariant() {
    let Some(root) = fixture() else {
        return;
    };
    let mut failures = Vec::new();
    for (process, outgoing) in [
        ("udx_wpg", &["w+", "g"][..]),
        ("udx_wph", &["w+", "h"][..]),
        ("udx_wphg", &["w+", "h", "g"][..]),
    ] {
        let mut runtime = NativeRuntime::load(root.join("ufo_w"), Some(process), None).unwrap();
        let binary64 = runtime.particle_mass_with_precision(24, 16).unwrap();
        assert_eq!(
            binary64.parse::<f64>().unwrap(),
            runtime.model_parameter_value("MW").unwrap().0
        );
        let mw = float(&runtime.particle_mass_with_precision(24, BITS_200).unwrap());
        let mh = if outgoing.contains(&"h") {
            float(&runtime.particle_mass_with_precision(25, BITS_200).unwrap())
        } else {
            int(0)
        };
        let flat: Vec<String> = w_points(outgoing, &mw, &mh)
            .iter()
            .flatten()
            .flat_map(|leg| leg.iter().map(|component| format!("{component:.120e}")))
            .collect();
        let values = runtime
            .evaluate_with_precision(&flat, 2, BITS_200)
            .unwrap()
            .values;
        let asymmetry = abs(&(float(&values[1]) / &float(&values[0]) - int(1)));
        println!("{process}: boost asymmetry {asymmetry:.2e}");
        if asymmetry > 1e-55 {
            failures.push(format!("{process}: {asymmetry:e}"));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
