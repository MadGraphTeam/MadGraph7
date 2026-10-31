// SPDX-License-Identifier: 0BSD
//! Graded (formal 1/N_c) output of compiled tree artifacts in binary64,
//! double-double and arbitrary precision.
//!
//! The colour sum of u u~ > g g is, for any N_c and with X = (t² + u²)/s²,
//! K [N(N² − 1) X − (N² − 1)/N] = K [N³ X − N (X + 1) + 1/N], so its N_c
//! degrees 3, 1 and −1 are, relative to the total at N_c = 3,
//! 27 X / (24 X − 8/3), −3 (X + 1) / (24 X − 8/3) and (1/3) / (24 X − 8/3),
//! and the degrees 2 and 0 vanish. The momenta are exact massless integer
//! points. For u d > u d the degrees 2 and 0 are 9/8 and −1/8 of the total
//! (the U(1)-subtraction piece of the gluon is graded) and the degree 1
//! vanishes.
//!
//! In double-double and arbitrary precision the graded values agree with
//! binary64; at the working precision itself they need the artifact
//! constants converted at that precision (MadGraphTeam/pyamplicol#26), so
//! those checks are `#[ignore]`d until then (`cargo test -- --ignored`).
//!
//! Supply the artifacts with
//! `pyamplicol generate 'u u~ > g g' DIR --color-accuracy full --color-graded
//! --execution-mode compiled --coupling-order-policy explicit
//! --max-coupling-order QCD=2 --max-coupling-order QED=0` and
//! `RUSTICOL_GRADED_COLOUR_UUX_GG=DIR`, and the same for `u d > u d` with
//! `RUSTICOL_GRADED_COLOUR_UD_UD`.
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

fn f(value: i64) -> Float {
    Float::with_val(BITS, value)
}

/// Degree → expected share of the total, at one point.
fn expected(process: &str, point: [i64; 4]) -> Vec<(i32, Float)> {
    if process == "u d > u d" {
        return vec![(2, f(9) / f(8)), (1, f(0)), (0, f(-1) / f(8))];
    }
    let [e, _, _, c] = point;
    let (s, t, u) = (f(4 * e * e), f(-2 * e * (e - c)), f(-2 * e * (e + c)));
    let x = (t.clone() * &t + u.clone() * &u) / (s.clone() * &s);
    let total = f(24) * &x - f(8) / f(3);
    vec![
        (3, f(27) * &x / &total),
        (2, f(0)),
        (1, f(-3) * (x.clone() + f(1)) / &total),
        (0, f(0)),
        (-1, f(1) / f(3) / &total),
    ]
}

fn runtime(variable: &str) -> Option<NativeRuntime> {
    let path = std::env::var_os(variable)?;
    Some(NativeRuntime::load(PathBuf::from(path), None, None).unwrap())
}

/// `working_precision`: compare the double-double and arbitrary-precision
/// shares with `tolerances`; otherwise with binary64 at 1e-12.
fn check(process: &str, variable: &str, tolerances: [(u32, f64); 2], working_precision: bool) {
    let Some(mut runtime) = runtime(variable) else {
        return;
    };
    let expansion = runtime.nc_expansion().unwrap().expect("a graded artifact");
    assert_eq!(expansion.accuracy, "full");
    let flat: Vec<i64> = POINTS.iter().flat_map(|point| momenta(*point)).collect();
    let binary64 = flat.iter().map(|x| *x as f64).collect::<Vec<_>>();

    // Binary64: degrees, their sum and the resolved colour axis.
    let graded = runtime
        .evaluate_graded_f64(&binary64, POINTS.len(), None)
        .unwrap();
    assert_eq!(graded.degrees, expansion.degrees);
    let totals = runtime.evaluate_f64(&binary64, POINTS.len()).unwrap();
    let resolved = runtime
        .evaluate_resolved_f64(&binary64, POINTS.len(), None, None)
        .unwrap();
    assert_eq!(resolved.color_ids, expansion.color_ids);
    let degree_count = graded.degrees.len();
    for (index, point) in POINTS.iter().enumerate() {
        let row = &graded.values[index * degree_count..(index + 1) * degree_count];
        let sum: f64 = row.iter().sum();
        assert!((sum / totals[index] - 1.0).abs() <= 1e-13);
        for (degree, share) in expected(process, *point) {
            let slot = graded.degrees.iter().position(|d| *d == degree).unwrap();
            assert!(
                (row[slot] / totals[index] - share.to_f64()).abs() <= 1e-13,
                "{process} f64 degree {degree}: {} vs {}",
                row[slot] / totals[index],
                share.to_f64()
            );
        }
        let helicities = resolved.helicity_ids.len();
        for (slot, value) in row.iter().enumerate() {
            let explicit: f64 = (0..helicities)
                .map(|h| resolved.values[(index * helicities + h) * degree_count + slot])
                .sum();
            assert!((explicit - value).abs() <= 1e-13 * totals[index].abs());
        }
    }

    // Double-double and arbitrary precision.
    let decimal: Vec<String> = flat.iter().map(ToString::to_string).collect();
    for (digits, tolerance) in tolerances {
        let tolerance = if working_precision { tolerance } else { 1e-12 };
        let graded = runtime
            .evaluate_graded_with_precision(&decimal, POINTS.len(), digits, None)
            .unwrap();
        let totals = runtime
            .evaluate_with_precision(&decimal, POINTS.len(), digits)
            .unwrap()
            .values;
        assert_eq!(graded.degrees, expansion.degrees);
        for (index, point) in POINTS.iter().enumerate() {
            let total = Float::parse(&totals[index], Some(BITS)).unwrap();
            let graded_total = Float::parse(&graded.totals[index], Some(BITS)).unwrap();
            let difference = (graded_total / &total - f(1)).norm().to_f64();
            assert!(
                difference <= tolerance,
                "{process} {digits} digits: graded total differs by {difference:e}"
            );
            for (degree, share) in expected(process, *point) {
                let slot = graded.degrees.iter().position(|d| *d == degree).unwrap();
                let value =
                    Float::parse(&graded.values[index * degree_count + slot], Some(BITS)).unwrap();
                let difference = (value / &total - share).norm().to_f64();
                assert!(
                    difference <= tolerance,
                    "{process} {digits} digits, degree {degree}: share differs by {difference:e}"
                );
            }
        }
    }
}

const TOLERANCES: [(u32, f64); 2] = [(32, 1e-27), (64, 1e-58)];

#[test]
fn graded_quark_antiquark_to_gluons_in_every_precision() {
    check(
        "u u~ > g g",
        "RUSTICOL_GRADED_COLOUR_UUX_GG",
        TOLERANCES,
        false,
    );
}

#[test]
fn graded_quark_scattering_in_every_precision() {
    check(
        "u d > u d",
        "RUSTICOL_GRADED_COLOUR_UD_UD",
        TOLERANCES,
        false,
    );
}

#[test]
#[ignore = "needs artifact constants at the working precision (MadGraphTeam/pyamplicol#26)"]
fn graded_quark_antiquark_to_gluons_at_working_precision() {
    check(
        "u u~ > g g",
        "RUSTICOL_GRADED_COLOUR_UUX_GG",
        TOLERANCES,
        true,
    );
}

#[test]
#[ignore = "needs artifact constants at the working precision (MadGraphTeam/pyamplicol#26)"]
fn graded_quark_scattering_at_working_precision() {
    check(
        "u d > u d",
        "RUSTICOL_GRADED_COLOUR_UD_UD",
        TOLERANCES,
        true,
    );
}
