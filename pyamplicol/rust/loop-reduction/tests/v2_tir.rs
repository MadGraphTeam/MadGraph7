// SPDX-License-Identifier: 0BSD

//! The TIR backend (`rusticol_loop_reduction::v2_tir`, m-lit lit-tir):
//! T1 scalar and tensor integrals against NinjaRS's generic engine in
//! double-double, T3 the layout permutation and the split UV column, T4 rank
//! over-declaration, T5 check frames, T6 chunking and batch composition, T10
//! unit scaling, the method choice and its check, and the typed capability
//! refusals. Binary64 TIR lanes are compared within the lax-precision rule
//! (relative to the lane's largest Laurent coefficient), never bitwise
//! except where an identity makes them so.
#![cfg(feature = "tir")]

use ninjars::fast::tensor_layout::{monomials, ncoeff};
use rusticol_core::RusticolErrorKind;
use rusticol_core::loop_execution::v2::{FamilyLanes, LaneKey, LaneOutcome, Poles, V2Backend};
use rusticol_core::loop_v2::retest::ReductionCheck;
use rusticol_loop_reduction::runtime::{
    PoleSelection, Reduction, V2Backends, V2Reducer, V2ReductionBackend,
};
use rusticol_loop_reduction::v2_tir::{
    TirOptions, TirReductionBackend, layout_positions, tir_frame,
};
use symbolica::domains::float::{Complex, DoubleFloat, RealLike};

type C = Complex<f64>;

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

/// One kinematic point of a family: shifts `V_i` (`V_0 = 0`), squared
/// masses and the scale.
#[derive(Clone)]
struct Point {
    shifts: Vec<[f64; 4]>,
    masses: Vec<C>,
    mu2: f64,
}

fn minkowski(v: [f64; 4]) -> f64 {
    v[0] * v[0] - v[1] * v[1] - v[2] * v[2] - v[3] * v[3]
}

/// Generic kinematics of `n` propagators: external momenta of energy ~500
/// with masses `masses` (GeV²) on the lines.
fn point(n: usize, masses: &[f64], rng: &mut Lcg) -> Point {
    let mut shifts = vec![[0.; 4]];
    for _ in 1..n {
        let last = *shifts.last().unwrap();
        let p = [
            400. + 150. * rng.next(),
            200. * rng.next(),
            200. * rng.next(),
            200. * rng.next(),
        ];
        shifts.push(std::array::from_fn(|mu| last[mu] + p[mu]));
    }
    Point {
        shifts,
        masses: masses.iter().map(|&m| C::new(m, 0.)).collect(),
        mu2: 91.188 * 91.188,
    }
}

/// One family's lanes: per point, `columns` lanes of random coefficients
/// at true rank `rank` (declared `declared`).
fn lanes(
    family: usize,
    points: &[Point],
    rank: usize,
    declared: usize,
    columns: usize,
    rng: &mut Lcg,
) -> FamilyLanes<f64> {
    let n = points[0].shifts.len();
    let mut out = FamilyLanes {
        family,
        propagators: n,
        rank: declared,
        points: Vec::new(),
        keys: Vec::new(),
        shifts: Vec::new(),
        masses: Vec::new(),
        scales: Vec::new(),
        s_matrix: Vec::new(),
        coefficients: Vec::new(),
    };
    for (p, point) in points.iter().enumerate() {
        for column in 0..columns {
            out.points.push(p);
            out.keys.push(LaneKey {
                family,
                order: 0,
                column,
            });
            for v in &point.shifts {
                out.shifts.extend(v.iter().map(|&x| C::new(x, 0.)));
            }
            out.masses.extend(point.masses.iter().copied());
            out.scales.push(C::new(point.mu2, 0.));
            for i in 0..n {
                for j in 0..n {
                    let d = std::array::from_fn(|mu| point.shifts[i][mu] - point.shifts[j][mu]);
                    out.s_matrix.push(C::new(minkowski(d), 0.));
                }
            }
            for k in 0..ncoeff(declared) {
                out.coefficients.push(if k < ncoeff(rank) {
                    C::new(rng.next(), rng.next())
                } else {
                    C::new(0., 0.)
                });
            }
        }
    }
    out
}

fn copy(group: &FamilyLanes<f64>) -> FamilyLanes<f64> {
    FamilyLanes {
        family: group.family,
        propagators: group.propagators,
        rank: group.rank,
        points: group.points.clone(),
        keys: group.keys.clone(),
        shifts: group.shifts.clone(),
        masses: group.masses.clone(),
        scales: group.scales.clone(),
        s_matrix: group.s_matrix.clone(),
        coefficients: group.coefficients.clone(),
    }
}

fn dd(group: &FamilyLanes<f64>) -> FamilyLanes<DoubleFloat> {
    let lift = |v: &[C]| {
        v.iter()
            .map(|z| Complex::new(DoubleFloat::from(z.re), DoubleFloat::from(z.im)))
            .collect()
    };
    FamilyLanes {
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
    }
}

fn tir() -> V2Backends {
    V2Backends::TiRust(Box::new(TirReductionBackend::new(TirOptions::default())))
}

fn ninjars(arithmetic: &str, engine: &str) -> V2Backends {
    V2Backends::NinjaRs(V2ReductionBackend {
        reducer: V2Reducer::new(arithmetic, None, engine, true).unwrap(),
        masters: PoleSelection::BOTH,
    })
}

type Lanes = Vec<Vec<LaneOutcome<f64>>>;

fn reduce(
    backend: &mut V2Backends,
    groups: &[FamilyLanes<f64>],
    poles: Poles,
    check: Option<ReductionCheck>,
) -> (Lanes, Option<Lanes>) {
    V2Backend::<f64>::begin(backend);
    V2Backend::<f64>::reduce(backend, groups, poles, check).unwrap()
}

/// NinjaRS's generic engine in double-double, read back in binary64 (the
/// reference of T1).
fn reference(groups: &[FamilyLanes<f64>], poles: Poles) -> Lanes {
    let _unlock = pyamplicol::unlock();
    let mut backend = ninjars("double-double", "generic");
    let groups: Vec<_> = groups.iter().map(dd).collect();
    V2Backend::<DoubleFloat>::begin(&mut backend);
    let (direct, _) =
        V2Backend::<DoubleFloat>::reduce(&mut backend, &groups, poles, None).unwrap();
    direct
        .into_iter()
        .map(|group| {
            group
                .into_iter()
                .map(|lane| {
                    lane.map(|v| {
                        v.into_iter()
                            .map(|z| C::new(z.re.to_f64(), z.im.to_f64()))
                            .collect()
                    })
                })
                .collect()
        })
        .collect()
}

/// NinjaRS's generic engine at `digits` decimal digits, read back in
/// binary64 (the reference where double-double's own error matters).
fn reference_arbitrary(groups: &[FamilyLanes<f64>], poles: Poles, digits: u32) -> Lanes {
    use symbolica::domains::float::Float;
    let _unlock = pyamplicol::unlock();
    let mut backend = V2Backends::NinjaRs(V2ReductionBackend {
        reducer: V2Reducer::new("arbitrary", Some(digits), "generic", true).unwrap(),
        masters: PoleSelection::BOTH,
    });
    let bits = Float::decimal_digits_to_bits(f64::from(digits)).unwrap();
    let lift = |v: &[C]| -> Vec<Complex<Float>> {
        v.iter()
            .map(|z| Complex::new(Float::with_val(bits, z.re), Float::with_val(bits, z.im)))
            .collect()
    };
    let groups: Vec<FamilyLanes<Float>> = groups
        .iter()
        .map(|g| FamilyLanes {
            family: g.family,
            propagators: g.propagators,
            rank: g.rank,
            points: g.points.clone(),
            keys: g.keys.clone(),
            shifts: lift(&g.shifts),
            masses: lift(&g.masses),
            scales: lift(&g.scales),
            s_matrix: lift(&g.s_matrix),
            coefficients: lift(&g.coefficients),
        })
        .collect();
    V2Backend::<Float>::begin(&mut backend);
    let (direct, _) = V2Backend::<Float>::reduce(&mut backend, &groups, poles, None).unwrap();
    direct
        .into_iter()
        .map(|group| {
            group
                .into_iter()
                .map(|lane| {
                    lane.map(|v| {
                        v.into_iter()
                            .map(|z| C::new(z.re.to_f64(), z.im.to_f64()))
                            .collect()
                    })
                })
                .collect()
        })
        .collect()
}

fn size(z: &C) -> f64 {
    (z.re * z.re + z.im * z.im).sqrt()
}

/// The largest difference of two lanes relative to the reference lane's
/// largest Laurent coefficient.
fn relative(a: &[C], b: &[C]) -> f64 {
    let scale = b.iter().map(size).fold(0., f64::max);
    a.iter()
        .zip(b)
        .map(|(x, y)| size(&(*x - y)))
        .fold(0., f64::max)
        / scale
}

fn worst(found: &Lanes, truth: &Lanes) -> f64 {
    let mut worst: f64 = 0.;
    for (a, b) in found.iter().zip(truth) {
        for (x, y) in a.iter().zip(b) {
            let (x, y) = (x.as_ref().unwrap(), y.as_ref().unwrap());
            worst = worst.max(relative(x, y));
        }
    }
    worst
}

/// Every lane of `found` within `MARGIN` times its own re-test deviation
/// (`check`, floored at four units of binary64 roundoff of the reference
/// lane's largest coefficient) of `truth` (PLAN §4 L6 against a reference
/// without error of its own); returns the worst relative difference.
fn within_estimates(found: &Lanes, check: &Lanes, truth: &Lanes) -> f64 {
    const MARGIN: f64 = 10.;
    let mut worst: f64 = 0.;
    for ((a, b), t) in found.iter().zip(check).zip(truth) {
        for (lane, ((x, y), z)) in a.iter().zip(b).zip(t).enumerate() {
            let (x, y, z) = (x.as_ref().unwrap(), y.as_ref().unwrap(), z.as_ref().unwrap());
            let scale = z.iter().map(size).fold(0., f64::max);
            let estimate = x
                .iter()
                .zip(y)
                .map(|(p, q)| size(&(*p - q)))
                .fold(4. * f64::EPSILON / 2. * scale, f64::max);
            let deviation = x.iter().zip(z).map(|(p, q)| size(&(*p - q))).fold(0., f64::max);
            assert!(
                deviation <= MARGIN * estimate,
                "lane {lane}: deviation {deviation:e}, estimate {estimate:e}, scale {scale:e}"
            );
            worst = worst.max(deviation / scale);
        }
    }
    worst
}

/// T1: A0 to D0 and tensor tadpoles, bubbles, triangles and boxes up to
/// rank 4 (massless, massive and complex masses), all Laurent orders and the
/// UV residue, against NinjaRS's generic engine in double-double.
#[test]
fn scalar_and_tensor_integrals_match_ninjars_double_double() {
    let mut rng = Lcg(7);
    let mut groups = Vec::new();
    let cases: [(usize, Vec<f64>); 7] = [
        (1, vec![173. * 173.]),
        (2, vec![0., 0.]),
        (2, vec![173. * 173., 80. * 80.]),
        (3, vec![0., 0., 0.]),
        (3, vec![173. * 173., 173. * 173., 173. * 173.]),
        (4, vec![0., 0., 0., 0.]),
        (4, vec![173. * 173., 0., 80. * 80., 0.]),
    ];
    let mut family = 0;
    for (n, masses) in cases {
        let points: Vec<_> = (0..2).map(|_| point(n, &masses, &mut rng)).collect();
        for rank in 0..=4.min(n + 2) {
            groups.push(lanes(family, &points, rank, rank, 2, &mut rng));
            family += 1;
        }
    }
    // A complex (CMS-like) mass: Im m² < 0.
    let mut complex = point(3, &[0., 0., 0.], &mut rng);
    complex.masses = vec![C::new(173. * 173., -173. * 1.5); 3];
    groups.push(lanes(family, &[complex], 3, 3, 2, &mut rng));
    let truth = reference(&groups, Poles::Split);
    let (found, check) = reduce(
        &mut tir(),
        &groups,
        Poles::Split,
        Some(ReductionCheck::RESCALED),
    );
    let worst = within_estimates(&found, &check.unwrap(), &truth);
    eprintln!("T1: TIR vs NinjaRS generic DD, worst relative {worst:e}");
    assert!(worst < 1e-6, "TIR vs NinjaRS generic DD: {worst:e}");
}

/// T3: the layout permutation is a bijection onto TIRust's basis with
/// equal exponents at ranks 0..8; the split UV column agrees with NinjaRS's
/// `TensorUvResidues` and the combined columns of "all" and "split" are
/// bitwise equal.
#[test]
fn layout_round_trip_and_split_columns() {
    for rank in 0..=8 {
        let position = layout_positions(rank).unwrap();
        let basis = iregi_rust::basis::tensor_basis(rank as u16);
        let mut seen = vec![false; position.len()];
        for (k, (&b, alpha)) in position.iter().zip(monomials(rank)).enumerate() {
            assert!(!seen[b], "rank {rank}: entry {b} twice");
            seen[b] = true;
            assert_eq!(basis[b].0.map(u32::from), alpha, "rank {rank} monomial {k}");
        }
        assert_eq!(position.len(), basis.len());
        assert_eq!(position.iter().any(|&b| b != position[b]), rank >= 2);
    }
    let mut rng = Lcg(11);
    let points: Vec<_> = (0..2)
        .map(|_| point(4, &[173. * 173., 0., 0., 80. * 80.], &mut rng))
        .collect();
    let groups = vec![lanes(0, &points, 4, 4, 3, &mut rng)];
    let (split, _) = reduce(&mut tir(), &groups, Poles::Split, None);
    let (all, _) = reduce(&mut tir(), &groups, Poles::All, None);
    let (none, _) = reduce(&mut tir(), &groups, Poles::None, None);
    let (ninja, _) = reduce(&mut ninjars("double", "fast"), &groups, Poles::Split, None);
    for lane in 0..split[0].len() {
        let s = split[0][lane].as_ref().unwrap();
        let a = all[0][lane].as_ref().unwrap();
        assert_eq!((s.len(), a.len()), (4, 3));
        assert_eq!(&s[..3], &a[..], "lane {lane}");
        assert_eq!(none[0][lane].as_ref().unwrap(), &a[..1], "lane {lane}");
        let n = ninja[0][lane].as_ref().unwrap();
        let scale = n.iter().map(size).fold(0., f64::max);
        assert!(size(&(s[3] - n[3])) < 1e-11 * scale, "UV column lane {lane}");
    }
}

/// T4: lanes declared above their true rank reduce at their true rank:
/// bitwise the values of the lanes declared at it.
#[test]
fn rank_over_declaration_is_bitwise() {
    let mut rng = Lcg(3);
    let points: Vec<_> = (0..2).map(|_| point(3, &[0., 173. * 173., 0.], &mut rng)).collect();
    let exact = lanes(0, &points, 2, 2, 2, &mut Lcg(5));
    let over = lanes(0, &points, 2, 4, 2, &mut Lcg(5));
    let (a, _) = reduce(&mut tir(), &[exact], Poles::All, None);
    let (b, _) = reduce(&mut tir(), &[over], Poles::All, None);
    assert_eq!(a, b);
}

/// T5: TIR's check frame of every reduction check rotates the propagators
/// with the origin shift (q -> -q leaves its moments bitwise): it is not
/// blind on a generic box and agrees with the direct frame; the rescaled
/// frame too.
#[test]
fn check_frames_rotate_and_rescale() {
    assert_eq!(
        tir_frame(ReductionCheck::REVERSED),
        rusticol_loop_reduction::runtime::ReductionFrame {
            reverse: false,
            rotation: 1,
            shift: true,
            rescale: false,
        }
    );
    let mut rng = Lcg(13);
    let points: Vec<_> = (0..2).map(|_| point(4, &[0., 0., 0., 0.], &mut rng)).collect();
    let groups = vec![lanes(0, &points, 3, 3, 2, &mut rng)];
    for check in [ReductionCheck::REVERSED, ReductionCheck::RESCALED] {
        let (direct, again) = reduce(&mut tir(), &groups, Poles::All, Some(check));
        let again = again.unwrap();
        let deviation = worst(&again, &direct);
        assert!(deviation > 0., "{check:?}: the check frame is blind");
        assert!(deviation < 1e-10, "{check:?}: {deviation:e}");
    }
}

/// T6: a lane's value does not depend on the other lanes, families or
/// points of its call, nor on how a group's points are split into calls.
#[test]
fn batch_composition_and_chunking_are_bitwise() {
    let mut rng = Lcg(17);
    let a_points: Vec<_> = (0..3).map(|_| point(4, &[0., 0., 173. * 173., 0.], &mut rng)).collect();
    let b_points: Vec<_> = (0..2).map(|_| point(4, &[0., 0., 0., 0.], &mut rng)).collect();
    let a = lanes(0, &a_points, 4, 4, 3, &mut rng);
    let b = lanes(1, &b_points, 4, 4, 2, &mut rng);
    let (together, _) = reduce(&mut tir(), &[copy(&a), copy(&b)], Poles::All, None);
    let (swapped, _) = reduce(&mut tir(), &[copy(&b), copy(&a)], Poles::All, None);
    let (alone, _) = reduce(&mut tir(), std::slice::from_ref(&a), Poles::All, None);
    assert_eq!(together[0], alone[0]);
    assert_eq!(together[0], swapped[1]);
    assert_eq!(together[1], swapped[0]);
    // One call per point of `a`, on one backend (its workspaces reused).
    let mut backend = tir();
    let lanes_per_point = 3;
    for p in 0..3 {
        let range = p * lanes_per_point..(p + 1) * lanes_per_point;
        let n = a.propagators;
        let width = ncoeff(a.rank);
        let part = FamilyLanes {
            family: 0,
            propagators: n,
            rank: a.rank,
            points: vec![0; lanes_per_point],
            keys: a.keys[range.clone()].to_vec(),
            shifts: a.shifts[range.start * 4 * n..range.end * 4 * n].to_vec(),
            masses: a.masses[range.start * n..range.end * n].to_vec(),
            scales: a.scales[range.clone()].to_vec(),
            s_matrix: a.s_matrix[range.start * n * n..range.end * n * n].to_vec(),
            coefficients: a.coefficients[range.start * width..range.end * width].to_vec(),
        };
        let (one, _) = reduce(&mut backend, &[part], Poles::All, None);
        assert_eq!(one[0], together[0][range], "point {p}");
    }
}

/// T10: every dimensionful input scaled by 2^±10 (coefficients of degree d
/// by 2^∓10d) scales the integral by 2^±10(4-2N): equal to rounding.
#[test]
fn unit_scaling() {
    let mut rng = Lcg(19);
    let points: Vec<_> = (0..2)
        .map(|_| point(4, &[173. * 173., 0., 80. * 80., 0.], &mut rng))
        .collect();
    let base = lanes(0, &points, 4, 4, 2, &mut rng);
    let (reference, _) = reduce(&mut tir(), std::slice::from_ref(&base), Poles::All, None);
    for exponent in [-10i32, 10] {
        let x = 2f64.powi(exponent);
        let mut scaled = copy(&base);
        let real = |z: &mut C, f: f64| *z = C::new(z.re * f, z.im * f);
        scaled.shifts.iter_mut().for_each(|z| real(z, x));
        scaled.masses.iter_mut().for_each(|z| real(z, x * x));
        scaled.scales.iter_mut().for_each(|z| real(z, x * x));
        scaled.s_matrix.iter_mut().for_each(|z| real(z, x * x));
        let width = ncoeff(base.rank);
        for (k, z) in scaled.coefficients.iter_mut().enumerate() {
            let degree: u32 = monomials(base.rank)[k % width].iter().sum();
            real(z, x.powi(-(degree as i32)));
        }
        let (found, _) = reduce(&mut tir(), &[scaled], Poles::All, None);
        let undo = x.powi(-(4 - 2 * base.propagators as i32));
        let found: Lanes = found
            .into_iter()
            .map(|g| {
                g.into_iter()
                    .map(|l| l.map(|v| v.into_iter().map(|z| C::new(z.re * undo, z.im * undo)).collect()))
                    .collect()
            })
            .collect();
        // Not bitwise (lit-spike-tir T10: the moments' own rounding, e.g.
        // their logarithms, at most 3.3e-11 of the row scale there); lax.
        let deviation = worst(&found, &reference);
        eprintln!("T10: 2^{exponent}: worst relative {deviation:e}");
        assert!(deviation < 1e-8, "2^{exponent}: {deviation:e}");
    }
}

/// The capability refusals are TIRust's own (typed plan construction) or
/// read from the row's exact invariants (an on-shell leg next to a
/// propagator of its mass), never a constant of this crate; a backend
/// without TIRust refuses typed.
#[test]
fn capability_refusals_are_typed() {
    let found = TirReductionBackend::with_histogram([(1, 3), (3, 3)], TirOptions::default())
        .unwrap()
        .plans();
    assert_eq!(found.len(), 4 + 4);
    // TIRust's direct engine at this pin serves at most twelve propagators.
    let refused = match TirReductionBackend::with_histogram([(13, 0)], TirOptions::default()) {
        Err(error) => error,
        Ok(_) => panic!("TIRust admitted 13 propagators at this pin"),
    };
    assert_eq!(refused.kind(), RusticolErrorKind::UnsupportedRuntimeCapability);
    assert!(refused.message().contains("TIRust cannot reduce 13 propagators"), "{}", refused.message());
    assert!(refused.message().contains("direct engine"), "{}", refused.message());
    // A top in its own gauge loop: S[1][2] = m_t² next to a top propagator.
    let mt2 = 173. * 173.;
    let mut rng = Lcg(23);
    let mut onshell = point(3, &[0., mt2, mt2], &mut rng);
    // V_2 - V_1 = (m_t, 0, 0, 0): its invariant is m_t² exactly.
    onshell.shifts[1] = [400., 100., 50., 25.];
    onshell.shifts[2] = [573., 100., 50., 25.];
    let groups = vec![lanes(0, &[onshell], 2, 2, 1, &mut rng)];
    assert_eq!(groups[0].s_matrix[3 + 2].re, mt2);
    let mut backend = tir();
    let (direct, _) = reduce(&mut backend, &groups, Poles::All, None);
    let refusal = direct[0][0].as_ref().unwrap_err();
    assert!(refusal.contains("TIRust capability"), "{refusal}");
    let statistics = V2Backend::<f64>::take_statistics(&mut backend);
    assert!(statistics.counts.contains(&("TIR capability refusals", 1)));
    // Extended carriers are refused typed before any reduction.
    let mut backend = tir();
    let error = V2Backend::<DoubleFloat>::reduce(&mut backend, &[dd(&groups[0])], Poles::All, None)
        .unwrap_err();
    assert_eq!(error.kind(), RusticolErrorKind::UnsupportedPrecision);
    assert_eq!(backend.reduction(), Reduction::TiRust);
}

/// Rows the conditioning gates refuse take the method choice (alternative
/// PV first within the expansions' tolerance) and are verified by their own
/// check row: near-forward boxes of a top loop with massless legs (exact
/// Pythagorean momenta, so every massless invariant is exactly zero: a small
/// Gram determinant, no on-shell decision) agree with NinjaRS's generic
/// engine at 80 digits within their estimates, and the statistics count
/// them.
#[test]
fn method_choice_rows_are_checked() {
    let mut groups = Vec::new();
    let mut family = 0;
    let mt2 = C::new(173. * 173., 0.);
    for m in [100f64, 1000., 9000.] {
        // (m^2 - 1, 2m, m^2 + 1) / 2^k: sin(theta) = 2m / (m^2 + 1).
        let unit = 2f64.powi(-(((m * m + 1.) / 500.).log2().ceil() as i32));
        let (a, b, c) = ((m * m - 1.) * unit, 2. * m * unit, (m * m + 1.) * unit);
        let p1 = [c, 0., 0., c];
        let p2 = [c, 0., 0., -c];
        let p3 = [c, b, 0., a];
        let v2: [f64; 4] = std::array::from_fn(|mu| p1[mu] + p2[mu]);
        let v3: [f64; 4] = std::array::from_fn(|mu| v2[mu] - p3[mu]);
        let point = Point {
            shifts: vec![[0.; 4], p1, v2, v3],
            masses: vec![mt2; 4],
            mu2: 91.188 * 91.188,
        };
        for rank in [4, 6] {
            groups.push(lanes(family, std::slice::from_ref(&point), rank, rank, 2, &mut Lcg(29)));
            family += 1;
        }
    }
    for g in &groups {
        // The massless legs' invariants are exactly zero.
        for (i, j) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            assert_eq!(g.s_matrix[i * 4 + j].re, 0.);
        }
    }
    let mut backend = tir();
    let (found, check) = reduce(&mut backend, &groups, Poles::All, Some(ReductionCheck::REVERSED));
    let statistics = V2Backend::<f64>::take_statistics(&mut backend);
    let count = |name: &str| {
        statistics
            .counts
            .iter()
            .find(|(key, _)| *key == name)
            .map_or(0, |(_, value)| *value)
    };
    eprintln!("method-choice statistics: {:?}", statistics.counts);
    // NinjaRS's generic engine at 80 digits: its double-double reaches only
    // 6e-8 on the rank-6 box at the smallest angle (TIR binary64: 1e-15).
    let truth = reference_arbitrary(&groups, Poles::All, 80);
    let check = check.unwrap();
    // NinjaRS's binary64 fast engine for comparison (recorded, not
    // asserted: at these small Gram determinants it is the less accurate
    // backend, and its re-test underestimates its own error).
    let (ninja, ninja_check) = reduce(
        &mut ninjars("double", "fast"),
        &groups,
        Poles::All,
        Some(ReductionCheck::REVERSED),
    );
    let ninja_check = ninja_check.unwrap();
    let mut served = 0;
    for (g, a) in found.iter().enumerate() {
        for (lane, x) in a.iter().enumerate() {
            match (x, &check[g][lane]) {
                (Ok(x), Ok(y)) => {
                    served += 1;
                    let z = truth[g][lane].as_ref().unwrap();
                    let n = ninja[g][lane].as_ref().unwrap();
                    let n_check = ninja_check[g][lane].as_ref().unwrap();
                    let scale = z.iter().map(size).fold(0., f64::max);
                    let spread = |p: &[C], q: &[C]| {
                        p.iter().zip(q).map(|(a, b)| size(&(*a - b))).fold(0., f64::max)
                    };
                    let truth_error = spread(x, z) / scale;
                    eprintln!(
                        "family {g} lane {lane}: vs 80 digits {truth_error:e}, TIR estimate {:e}, \
                         NinjaRS {:e} (estimate {:e})",
                        spread(x, y) / scale,
                        spread(n, z) / scale,
                        spread(n, n_check) / scale
                    );
                    assert!(truth_error < 1e-12, "family {g} lane {lane}: {truth_error:e}");
                    let one = |v: &LaneOutcome<f64>| vec![vec![v.clone()]];
                    within_estimates(&one(&found[g][lane]), &one(&check[g][lane]), &one(&truth[g][lane]));
                }
                // A refused lane fails closed with TIRust's or the check's reason.
                (Err(reason), _) | (_, Err(reason)) => {
                    assert!(reason.contains("TIRust"), "{reason}")
                }
            }
        }
    }
    assert!(served > 0);
    assert!(count("TIR gate refusals") > 0, "{:?}", statistics.counts);
    assert!(count("TIR method-choice rows") > 0, "{:?}", statistics.counts);
}

/// Diagnostic (recorded in the package report, run with `--ignored`): the
/// near-forward boxes of `method_choice_rows_are_checked` with the
/// alternative PV first (the default) or the expansions first, checked in
/// the rotated and in the rescaled frame, against double-double.
#[test]
#[ignore]
fn method_choice_variants() {
    let mt2 = C::new(173. * 173., 0.);
    for m in [1000f64, 9000.] {
        let unit = 2f64.powi(-(((m * m + 1.) / 500.).log2().ceil() as i32));
        let (a, b, c) = ((m * m - 1.) * unit, 2. * m * unit, (m * m + 1.) * unit);
        let p1 = [c, 0., 0., c];
        let p2 = [c, 0., 0., -c];
        let p3 = [c, b, 0., a];
        let v2: [f64; 4] = std::array::from_fn(|mu| p1[mu] + p2[mu]);
        let v3: [f64; 4] = std::array::from_fn(|mu| v2[mu] - p3[mu]);
        let point = Point {
            shifts: vec![[0.; 4], p1, v2, v3],
            masses: vec![mt2; 4],
            mu2: 91.188 * 91.188,
        };
        for rank in [4, 6] {
            let groups = vec![lanes(0, std::slice::from_ref(&point), rank, rank, 2, &mut Lcg(29))];
            let truth = reference_arbitrary(&groups, Poles::All, 80);
            let dd = reference(&groups, Poles::All);
            for lane in 0..2 {
                eprintln!(
                    "m {m} rank {rank}: lane {lane} double-double vs 80 digits {:e}",
                    relative(dd[0][lane].as_ref().unwrap(), truth[0][lane].as_ref().unwrap())
                );
            }
            for alternative in [Some(iregi_rust::direct::EXPANSION_TOLERANCE), None] {
                for check in [ReductionCheck::REVERSED, ReductionCheck::RESCALED] {
                    let options = TirOptions {
                        alternative_first: alternative,
                        ..TirOptions::default()
                    };
                    let mut backend =
                        V2Backends::TiRust(Box::new(TirReductionBackend::new(options)));
                    let (found, again) = reduce(&mut backend, &groups, Poles::All, Some(check));
                    let again = again.unwrap();
                    for lane in 0..2 {
                        let (x, y, z) = (
                            found[0][lane].as_ref().unwrap(),
                            again[0][lane].as_ref().unwrap(),
                            truth[0][lane].as_ref().unwrap(),
                        );
                        eprintln!(
                            "m {m} rank {rank} alternative-first {} check {}: lane {lane} vs 80 \
                             digits {:e}, estimate {:e}",
                            alternative.is_some(),
                            if check.rescale { "rescaled" } else { "rotated" },
                            relative(x, z),
                            relative(y, x)
                        );
                    }
                }
            }
        }
    }
}
