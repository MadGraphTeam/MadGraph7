// SPDX-License-Identifier: 0BSD

//! The TIR backend at exactly degenerate kinematics (TIRust 49dc836e).
//!
//! Boxes whose shifts are affinely dependent with a non-zero constant
//! relation, `sum_i c_i D_i = K` with `sum_i c_i = 0` and
//! `sum_i c_i V_i = 0`, are reduced by a plan variant. Before TIRust
//! 49dc836e that variant could need a master family (A0 or B0) its base plan
//! never created; the family was skipped, its masters kept the previous
//! lane's values or zero and the lane was returned as a gate-pass `Success`
//! that pyAmpliCol does not re-check by default. At the old pin de60dc48 the
//! lanes below were served with relative errors of 7 (the gg -> HH top-loop
//! box at threshold, rank 1) and 0.3 (parallelogram boxes, rank 2).
//!
//! The reference is independent of TIRust: the denominator identity
//! `1/prod_j D_j = sum_i (c_i/K) / prod_{j != i} D_j`, whose triangle pinches
//! NinjaRS's generic engine evaluates at 80 digits. The tolerance follows the
//! conditioning of that sum (see [`TOLERANCE`]).
#![cfg(feature = "tir")]

use ninjars::fast::tensor_layout::ncoeff;
use rusticol_core::loop_execution::v2::{FamilyLanes, LaneKey, LaneOutcome, Poles, V2Backend};
use rusticol_loop_reduction::runtime::{PoleSelection, V2Backends, V2Reducer, V2ReductionBackend};
use rusticol_loop_reduction::v2_tir::{TirOptions, TirReductionBackend};
use symbolica::domains::float::{Complex, Float, RealLike};

type C = Complex<f64>;
type Lanes = Vec<Vec<LaneOutcome<f64>>>;

/// Relative error allowed per unit of the conditioning `kappa` of the
/// identity (the cancellation between the weighted pinches, measured per
/// case, 4 to 124 here). Correct binary64 lanes agree with the reference to
/// at most 3.4e-15 x kappa; the wrong values of the old pin are 1.6e-6 x
/// kappa and more (2.0e-4 at kappa 124). 1e-9 sits more than five orders
/// above the first and three below the second.
const TOLERANCE: f64 = 1e-9;

/// A deterministic generator of reals in (-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2. - 1.
    }
}

/// One kinematic point: shifts, squared masses and the scale.
#[derive(Clone)]
struct Point {
    shifts: Vec<[f64; 4]>,
    masses: Vec<f64>,
    mu2: f64,
}

fn minkowski(v: [f64; 4]) -> f64 {
    v[0] * v[0] - v[1] * v[1] - v[2] * v[2] - v[3] * v[3]
}

fn add(a: [f64; 4], b: [f64; 4], s: f64) -> [f64; 4] {
    std::array::from_fn(|i| a[i] + s * b[i])
}

/// One family group: per point one lane per coefficient column.
fn group(points: &[Point], rank: usize, columns: &[Vec<C>]) -> FamilyLanes<f64> {
    let n = points[0].shifts.len();
    let mut out = FamilyLanes {
        family: 0,
        propagators: n,
        rank,
        points: Vec::new(),
        keys: Vec::new(),
        shifts: Vec::new(),
        masses: Vec::new(),
        scales: Vec::new(),
        s_matrix: Vec::new(),
        coefficients: Vec::new(),
    };
    for (p, point) in points.iter().enumerate() {
        for (column, coefficients) in columns.iter().enumerate() {
            out.points.push(p);
            out.keys.push(LaneKey {
                family: 0,
                order: 0,
                column,
            });
            for v in &point.shifts {
                out.shifts.extend(v.iter().map(|&x| C::new(x, 0.)));
            }
            out.masses
                .extend(point.masses.iter().map(|&m| C::new(m, 0.)));
            out.scales.push(C::new(point.mu2, 0.));
            for a in &point.shifts {
                for b in &point.shifts {
                    out.s_matrix.push(C::new(minkowski(add(*a, *b, -1.)), 0.));
                }
            }
            out.coefficients.extend(coefficients.iter().copied());
        }
    }
    out
}

/// The TIR backend with its default options, binary64.
fn tir(groups: &[FamilyLanes<f64>]) -> Lanes {
    let mut backend = V2Backends::TiRust(Box::new(TirReductionBackend::new(TirOptions::default())));
    V2Backend::<f64>::begin(&mut backend);
    V2Backend::<f64>::reduce(&mut backend, groups, Poles::All, None)
        .unwrap()
        .0
}

/// NinjaRS's generic engine at 80 digits, read back in binary64.
fn ninjars_80(group: &FamilyLanes<f64>) -> Vec<Vec<C>> {
    let _unlock = pyamplicol::unlock();
    let mut backend = V2Backends::NinjaRs(V2ReductionBackend {
        reducer: V2Reducer::new("arbitrary", Some(80), "generic", true).unwrap(),
        masters: PoleSelection::BOTH,
    });
    let bits = Float::decimal_digits_to_bits(80.).unwrap();
    let lift = |v: &[C]| -> Vec<Complex<Float>> {
        v.iter()
            .map(|z| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im)))
            .collect()
    };
    let lifted = [FamilyLanes {
        family: group.family,
        propagators: group.propagators,
        rank: group.rank,
        points: group.points.clone(),
        keys: group.keys.clone(),
        shifts: lift(&group.shifts),
        masses: lift(&group.masses),
        scales: lift(&group.scales),
        s_matrix: lift(&group.s_matrix),
        coefficients: lift(&group.coefficients),
    }];
    V2Backend::<Float>::begin(&mut backend);
    let (lanes, _) = V2Backend::<Float>::reduce(&mut backend, &lifted, Poles::All, None).unwrap();
    lanes
        .into_iter()
        .flatten()
        .map(|lane| {
            lane.expect("NinjaRS at 80 digits serves the triangle pinch")
                .into_iter()
                .map(|z| C::new(z.re.to_f64(), z.im.to_f64()))
                .collect()
        })
        .collect()
}

fn size(z: &C) -> f64 {
    (z.re * z.re + z.im * z.im).sqrt()
}

fn scale(lane: &[C]) -> f64 {
    lane.iter().map(size).fold(0., f64::max)
}

/// The largest difference relative to the reference lane's largest Laurent
/// coefficient.
fn relative(a: &[C], b: &[C]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| size(&(*x - y)))
        .fold(0., f64::max)
        / scale(b)
}

/// The identity's value per coefficient column and its conditioning
/// `kappa = sum_i |c_i/K| scale(pinch_i) / scale(sum)` (worst column).
fn identity(point: &Point, c: &[f64], rank: usize, columns: &[Vec<C>]) -> (Vec<Vec<C>>, f64) {
    let k: f64 = point
        .shifts
        .iter()
        .zip(&point.masses)
        .zip(c)
        .map(|((v, m), c)| c * (minkowski(*v) - m))
        .sum();
    let n = point.shifts.len();
    assert_eq!(c.iter().sum::<f64>(), 0., "the relation is exact");
    for r in 0..4 {
        let residual: f64 = point.shifts.iter().zip(c).map(|(v, c)| c * v[r]).sum();
        assert_eq!(residual, 0., "the relation is exact (component {r})");
    }
    assert!(k.abs() > 1., "a non-zero constant relation, K = {k}");
    let mut sum = vec![Vec::new(); columns.len()];
    let mut weight = vec![0.; columns.len()];
    for (i, &ci) in c.iter().enumerate() {
        if ci == 0. {
            continue;
        }
        let keep = (0..n).filter(|&j| j != i);
        let pinch = Point {
            shifts: keep.clone().map(|j| point.shifts[j]).collect(),
            masses: keep.map(|j| point.masses[j]).collect(),
            mu2: point.mu2,
        };
        let w = ci / k;
        for (column, lane) in ninjars_80(&group(&[pinch], rank, columns))
            .into_iter()
            .enumerate()
        {
            weight[column] += w.abs() * scale(&lane);
            if sum[column].is_empty() {
                sum[column] = vec![C::new(0., 0.); lane.len()];
            }
            for (s, v) in sum[column].iter_mut().zip(&lane) {
                *s = C::new(s.re + w * v.re, s.im + w * v.im);
            }
        }
    }
    let kappa = sum
        .iter()
        .zip(&weight)
        .map(|(lane, w)| w / scale(lane))
        .fold(1., f64::max);
    (sum, kappa)
}

/// Generic kinematics of the same size and mass pattern as `like`.
fn generic(like: &Point, rng: &mut Lcg) -> Point {
    let size = like
        .shifts
        .iter()
        .flatten()
        .fold(1f64, |a, &b| a.max(b.abs()));
    let mut shifts = vec![[0.; 4]];
    for _ in 1..like.shifts.len() {
        let p = [
            size * (0.8 + 0.3 * rng.next()),
            0.4 * size * rng.next(),
            0.4 * size * rng.next(),
            0.4 * size * rng.next(),
        ];
        shifts.push(add(*shifts.last().unwrap(), p, 1.));
    }
    let masses = like
        .masses
        .iter()
        .map(|&m| {
            if m == 0. {
                0.
            } else {
                m * (1. + 0.2 * rng.next())
            }
        })
        .collect();
    Point {
        shifts,
        masses,
        mu2: like.mu2,
    }
}

/// gg -> HH through a top loop exactly at threshold (sqrt(s) = 2 mH): the
/// box [0, p1, p1 + p2, (p1 + p2)/2], relation D_0 + D_2 - 2 D_3 = K.
fn threshold_box() -> (Point, Vec<f64>) {
    let (mt2, e) = (173. * 173., 125.);
    let (p1, p2) = ([e, 0., 0., e], [e, 0., 0., -e]);
    let p = add(p1, p2, 1.);
    let point = Point {
        shifts: vec![[0.; 4], p1, p, p.map(|x| x / 2.)],
        masses: vec![mt2; 4],
        mu2: 125. * 125.,
    };
    (point, vec![1., 0., 1., -2.])
}

/// Parallelogram boxes (V_0 + V_3 = V_1 + V_2, integer kinematics).
fn parallelograms() -> Vec<(Point, Vec<f64>)> {
    let c = vec![1., -1., -1., 1.];
    vec![
        (
            Point {
                shifts: vec![
                    [-1., -1., 0., 0.],
                    [0., -3., -1., 0.],
                    [0.; 4],
                    [1., -2., -1., 0.],
                ],
                masses: vec![0., 2., 0., 2.],
                mu2: 9.,
            },
            c.clone(),
        ),
        (
            Point {
                shifts: vec![
                    [1., -2., 0., 2.],
                    [-1., -3., 0., 0.],
                    [1., -3., -2., 0.],
                    [-1., -4., -2., -2.],
                ],
                masses: vec![0., 3., 0., 4.],
                mu2: 9.,
            },
            c,
        ),
    ]
}

/// Reduces the degenerate point after two generic rows of the same family
/// (and alone) and returns the lanes of it that disagree with the identity.
fn check(
    name: &str,
    point: &Point,
    c: &[f64],
    rank: usize,
    alone: bool,
    rng: &mut Lcg,
) -> Vec<String> {
    let columns: Vec<Vec<C>> = (0..2)
        .map(|_| {
            (0..ncoeff(rank))
                .map(|_| C::new(rng.next(), rng.next()))
                .collect()
        })
        .collect();
    let (truth, kappa) = identity(point, c, rank, &columns);
    let mut rows = vec![generic(point, rng), generic(point, rng)];
    rows.push(point.clone());
    let mut runs = vec![("after generic rows", rows)];
    if alone {
        runs.push(("alone", vec![point.clone()]));
    }
    let mut failures = Vec::new();
    for (label, rows) in runs {
        let lanes = tir(&[group(&rows, rank, &columns)]);
        let offset = (rows.len() - 1) * columns.len();
        for (column, reference) in truth.iter().enumerate() {
            let line = format!("{name} r{rank} {label} column {column}");
            match &lanes[0][offset + column] {
                Ok(lane) => {
                    let error = relative(lane, reference);
                    println!("{line}: relative error {error:.2e}, kappa {kappa:.1}");
                    if error.is_nan() || error > TOLERANCE * kappa {
                        failures.push(format!("{line}: relative error {error:.3e}"));
                    }
                }
                Err(reason) => failures.push(format!("{line}: refused ({reason})")),
            }
        }
    }
    failures
}

#[test]
fn degenerate_boxes_match_the_denominator_identity() {
    let mut rng = Lcg(31);
    let (point, c) = threshold_box();
    let mut failures = check("gg -> HH threshold box", &point, &c, 1, true, &mut rng);
    failures.extend(check(
        "gg -> HH threshold box",
        &point,
        &c,
        2,
        false,
        &mut rng,
    ));
    for (index, (point, c)) in parallelograms().iter().enumerate() {
        let name = format!("parallelogram box {index}");
        failures.extend(check(&name, point, c, 2, true, &mut rng));
    }
    assert!(
        failures.is_empty(),
        "lanes off the identity (tolerance {TOLERANCE:e} x kappa):\n{}",
        failures.join("\n")
    );
}
