// SPDX-License-Identifier: 0BSD

//! The TIR backend (`rusticol_loop_reduction::v2_tir`, m-lit lit-tir):
//! T1 scalar and tensor integrals against NinjaRS's generic engine in
//! double-double, T3 the layout permutation and the split UV column, T4 rank
//! over-declaration, T5 check frames, T6 chunking and batch composition, T10
//! unit scaling, the default's fail-closed gate refusals (no method-choice
//! row), the method choice and its check when enabled, and the typed
//! capability refusals. Binary64 TIR lanes are compared within the lax-precision rule
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
use iregi_rust::DirectCapabilities;
use rusticol_loop_reduction::v2_tir::{
    TirOptions, TirReductionBackend, capability_refusal, carrier_gate, layout_positions,
    tir_frame,
};
use rusticol_loop_reduction::NativeFloat;
use symbolica::domains::float::{Complex, DoubleFloat, Float, Real, RealLike, SingleFloat};

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

/// The TIR backend in its default settings (the method choice off:
/// gate-refused rows fail closed, DECISIONS 2026-11-07 ~17:20 (d)).
fn tir() -> V2Backends {
    V2Backends::TiRust(Box::new(TirReductionBackend::new(TirOptions::default())))
}

/// The TIR backend with the method choice on (the one option lit-tir-mp
/// turns back on after its re-pin) and otherwise the defaults.
fn tir_method_choice() -> V2Backends {
    V2Backends::TiRust(Box::new(TirReductionBackend::new(TirOptions {
        method_choice: true,
        ..TirOptions::default()
    })))
}

/// The count `name` of a backend's statistics (0 when absent).
fn count(statistics: &rusticol_core::loop_execution::v2::BackendStatistics, name: &str) -> u64 {
    statistics
        .counts
        .iter()
        .find(|(key, _)| *key == name)
        .map_or(0, |(_, value)| *value)
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
    let (direct, _) = V2Backend::<DoubleFloat>::reduce(&mut backend, &groups, poles, None).unwrap();
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

/// Whether a lane failed closed on TIRust's conditioning gates (the default
/// settings: the method choice off, DECISIONS 2026-11-07 ~17:20 (d)); any
/// other failure panics.
fn gate_refused(lane: &LaneOutcome<f64>) -> bool {
    match lane {
        Ok(_) => false,
        Err(reason) => {
            assert!(
                reason.contains("refused by the conditioning gates"),
                "not a gate refusal: {reason}"
            );
            true
        }
    }
}

/// The worst relative difference of the lanes of `found` and `truth` that
/// neither refuses on the gates, and how many were compared.
fn worst(found: &Lanes, truth: &Lanes) -> (f64, usize) {
    let (mut worst, mut compared): (f64, usize) = (0., 0);
    for (a, b) in found.iter().zip(truth) {
        for (x, y) in a.iter().zip(b) {
            if gate_refused(x) || gate_refused(y) {
                continue;
            }
            let (x, y) = (x.as_ref().unwrap(), y.as_ref().unwrap());
            worst = worst.max(relative(x, y));
            compared += 1;
        }
    }
    (worst, compared)
}

/// Every lane of `found` within `MARGIN` times its own re-test deviation
/// (`check`, floored at four units of binary64 roundoff of the reference
/// lane's largest coefficient) of `truth` (PLAN §4 L6 against a reference
/// without error of its own), skipping the lanes refused on the gates in
/// either frame; returns the worst relative difference and how many lanes
/// were compared.
fn within_estimates(found: &Lanes, check: &Lanes, truth: &Lanes) -> (f64, usize) {
    const MARGIN: f64 = 10.;
    let (mut worst, mut compared): (f64, usize) = (0., 0);
    for ((a, b), t) in found.iter().zip(check).zip(truth) {
        for (lane, ((x, y), z)) in a.iter().zip(b).zip(t).enumerate() {
            if gate_refused(x) || gate_refused(y) {
                continue;
            }
            compared += 1;
            let (x, y, z) = (
                x.as_ref().unwrap(),
                y.as_ref().unwrap(),
                z.as_ref().unwrap(),
            );
            let scale = z.iter().map(size).fold(0., f64::max);
            let estimate = x
                .iter()
                .zip(y)
                .map(|(p, q)| size(&(*p - q)))
                .fold(4. * f64::EPSILON / 2. * scale, f64::max);
            let deviation = x
                .iter()
                .zip(z)
                .map(|(p, q)| size(&(*p - q)))
                .fold(0., f64::max);
            assert!(
                deviation <= MARGIN * estimate,
                "lane {lane}: deviation {deviation:e}, estimate {estimate:e}, scale {scale:e}"
            );
            worst = worst.max(deviation / scale);
        }
    }
    (worst, compared)
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
    let total: usize = groups.iter().map(|g| g.keys.len()).sum();
    // The default (gate-refused rows fail closed) and the method choice on
    // (every lane served: the boxes the gates refuse too).
    for (name, mut backend) in [("default", tir()), ("method choice", tir_method_choice())] {
        let (found, check) = reduce(
            &mut backend,
            &groups,
            Poles::Split,
            Some(ReductionCheck::RESCALED),
        );
        let (worst, compared) = within_estimates(&found, &check.unwrap(), &truth);
        eprintln!(
            "T1 ({name}): TIR vs NinjaRS generic DD, {compared} of {total} lanes, worst relative \
             {worst:e}"
        );
        assert!(worst < 1e-6, "TIR vs NinjaRS generic DD: {worst:e}");
        assert!(2 * compared > total, "{name}: {compared} of {total}");
        if name == "method choice" {
            assert_eq!(compared, total);
        }
    }
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
    let (ninja, _) = reduce(&mut ninjars("double", "fast"), &groups, Poles::Split, None);
    // The default (these boxes may fail closed on the gates, in every pole
    // mode alike) and the method choice on (every lane served).
    let configurations: [(&str, fn() -> V2Backends); 2] =
        [("default", tir), ("method choice", tir_method_choice)];
    for (name, backend) in configurations {
        let (split, _) = reduce(&mut backend(), &groups, Poles::Split, None);
        let (all, _) = reduce(&mut backend(), &groups, Poles::All, None);
        let (none, _) = reduce(&mut backend(), &groups, Poles::None, None);
        let mut compared = 0;
        for lane in 0..split[0].len() {
            if gate_refused(&split[0][lane]) {
                assert!(gate_refused(&all[0][lane]) && gate_refused(&none[0][lane]));
                continue;
            }
            compared += 1;
            let s = split[0][lane].as_ref().unwrap();
            let a = all[0][lane].as_ref().unwrap();
            assert_eq!((s.len(), a.len()), (4, 3));
            assert_eq!(&s[..3], &a[..], "{name} lane {lane}");
            assert_eq!(
                none[0][lane].as_ref().unwrap(),
                &a[..1],
                "{name} lane {lane}"
            );
            let n = ninja[0][lane].as_ref().unwrap();
            let scale = n.iter().map(size).fold(0., f64::max);
            assert!(
                size(&(s[3] - n[3])) < 1e-11 * scale,
                "{name}: UV column lane {lane}"
            );
        }
        eprintln!("T3 ({name}): {compared} of {} lanes", split[0].len());
        if name == "method choice" {
            assert_eq!(compared, split[0].len());
        }
    }
}

/// T4: lanes declared above their true rank reduce at their true rank:
/// bitwise the values of the lanes declared at it.
#[test]
fn rank_over_declaration_is_bitwise() {
    let mut rng = Lcg(3);
    let points: Vec<_> = (0..2)
        .map(|_| point(3, &[0., 173. * 173., 0.], &mut rng))
        .collect();
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
    let points: Vec<_> = (0..2)
        .map(|_| point(4, &[0., 0., 0., 0.], &mut rng))
        .collect();
    let groups = vec![lanes(0, &points, 3, 3, 2, &mut rng)];
    for check in [ReductionCheck::REVERSED, ReductionCheck::RESCALED] {
        for mut backend in [tir(), tir_method_choice()] {
            let (direct, again) = reduce(&mut backend, &groups, Poles::All, Some(check));
            let again = again.unwrap();
            let (deviation, compared) = worst(&again, &direct);
            assert!(compared > 0, "{check:?}: no lane compared");
            assert!(deviation > 0., "{check:?}: the check frame is blind");
            assert!(deviation < 1e-10, "{check:?}: {deviation:e}");
        }
    }
}

/// T6: a lane's value does not depend on the other lanes, families or
/// points of its call, nor on how a group's points are split into calls.
#[test]
fn batch_composition_and_chunking_are_bitwise() {
    let mut rng = Lcg(17);
    let a_points: Vec<_> = (0..3)
        .map(|_| point(4, &[0., 0., 173. * 173., 0.], &mut rng))
        .collect();
    let b_points: Vec<_> = (0..2)
        .map(|_| point(4, &[0., 0., 0., 0.], &mut rng))
        .collect();
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
                    .map(|l| {
                        l.map(|v| {
                            v.into_iter()
                                .map(|z| C::new(z.re * undo, z.im * undo))
                                .collect()
                        })
                    })
                    .collect()
            })
            .collect();
        // The conditioning gates are scale invariant: the same lanes fail
        // closed at every scale.
        for (a, b) in found[0].iter().zip(&reference[0]) {
            assert_eq!(gate_refused(a), gate_refused(b), "2^{exponent}");
        }
        // Not bitwise (lit-spike-tir T10: the moments' own rounding, e.g.
        // their logarithms, at most 3.3e-11 of the row scale there); lax.
        let (deviation, compared) = worst(&found, &reference);
        eprintln!("T10: 2^{exponent}: {compared} lanes, worst relative {deviation:e}");
        assert!(compared > 0, "2^{exponent}: no lane compared");
        assert!(deviation < 1e-8, "2^{exponent}: {deviation:e}");
    }
}

/// The capability refusals are TIRust's own (its capability query and its
/// typed plan construction), never a constant of this crate: a family
/// beyond TIRust's former fixed caps is admitted, one beyond what the
/// query reports is refused typed with TIRust's reason, in every carrier.
#[test]
fn capability_refusals_are_typed() {
    let found = TirReductionBackend::with_histogram([(1, 3), (3, 3)], 53, TirOptions::default())
        .unwrap()
        .plans();
    assert_eq!(found.len(), 4 + 4);
    // Thirteen propagators (beyond the twelve of TIRust's former fixed cap)
    // are admitted: the bound is TIRust's capability query, no constant here.
    let beyond = TirReductionBackend::with_histogram([(13, 2)], 53, TirOptions::default()).unwrap();
    assert_eq!(beyond.plans(), [(13, 0), (13, 1), (13, 2)]);
    for bits in [53, 106, 256] {
        assert_eq!(capability_refusal(13, 2, bits), None);
        // One propagator more than the query's largest count is refused
        // typed, with TIRust's reason.
        let n = DirectCapabilities::of_representation(bits).max_propagators + 1;
        let refused = match TirReductionBackend::with_histogram([(n, 0)], bits, TirOptions::default())
        {
            Err(error) => error,
            Ok(_) => panic!("TIRust admitted {n} propagators"),
        };
        assert_eq!(
            refused.kind(),
            RusticolErrorKind::UnsupportedRuntimeCapability
        );
        let message = refused.message();
        assert!(
            message.contains(&format!("TIRust cannot reduce {n} propagators"))
                && message.contains("direct engine"),
            "{message}"
        );
        let reason = capability_refusal(n, 0, bits).unwrap();
        assert!(message.ends_with(&reason), "{message} / {reason}");
    }
    // The artifacts' rank admission (rank <= N + 2 until lit-eft) comes first.
    assert!(
        capability_refusal(4, 7, 53)
            .unwrap()
            .contains("for both reduction backends")
    );
    assert_eq!(
        rusticol_loop_reduction::runtime::rank_refusal(Reduction::TiRust, 13, 2),
        None
    );
}

/// A top in its own gauge loop: `S[1][2] = m_t²` next to two top
/// propagators (an exact on-shell configuration). TIRust takes the supplied
/// S-matrix (U1), so the row is served on the exact on-shell branch in every
/// carrier and agrees with NinjaRS's generic engine at 80 digits.
#[test]
fn an_onshell_leg_next_to_its_propagator_is_served() {
    let mt2 = 173. * 173.;
    let mut rng = Lcg(23);
    let mut onshell = point(3, &[0., mt2, mt2], &mut rng);
    // V_2 - V_1 = (m_t, 0, 0, 0): its invariant is m_t² exactly.
    onshell.shifts[1] = [400., 100., 50., 25.];
    onshell.shifts[2] = [573., 100., 50., 25.];
    let groups = vec![lanes(0, &[onshell], 2, 2, 2, &mut rng)];
    assert_eq!(groups[0].s_matrix[3 + 2].re, mt2);
    let truth = reference_arbitrary(&groups, Poles::All, 80);
    let (found, _) = reduce(&mut tir(), &groups, Poles::All, None);
    let (worst, compared) = worst(&found, &truth);
    eprintln!("on-shell top, binary64: {compared} lanes, worst relative {worst:e}");
    assert_eq!(compared, 2);
    assert!(worst < 1e-12, "{worst:e}");
    for (bits, bound) in [(106, 1e-24), (256, 1e-60)] {
        let deviation = extended_against_truth(&groups, bits, 80, Poles::All);
        eprintln!("on-shell top, {bits} bits vs 80 digits: {deviation:e}");
        assert!(deviation < bound, "{bits} bits: {deviation:e}");
    }
}

/// The groups lifted exactly into a carrier of `bits` bits, the S-matrix
/// formed in the carrier from the lifted shifts (inputs consistent to the
/// carrier's precision, never exactly formed invariants with rounded
/// shifts: lit-onshell-probe's rule).
fn lifted(groups: &[FamilyLanes<f64>], bits: u32) -> Vec<FamilyLanes<Float>> {
    let real = |x: f64| Float::with_val(bits, x);
    let complex = |z: &C| Complex::new(real(z.re), real(z.im));
    groups
        .iter()
        .map(|g| {
            let n = g.propagators;
            let shifts: Vec<Complex<Float>> = g.shifts.iter().map(complex).collect();
            let mut s_matrix = Vec::new();
            for lane in 0..g.keys.len() {
                let v = &shifts[lane * 4 * n..(lane + 1) * 4 * n];
                for i in 0..n {
                    for j in 0..n {
                        let d = |mu: usize| v[4 * i + mu].re.clone() - &v[4 * j + mu].re;
                        let square = |mu: usize| d(mu) * &d(mu);
                        let x = square(0) - &square(1) - &square(2) - &square(3);
                        s_matrix.push(Complex::new(x, real(0.)));
                    }
                }
            }
            FamilyLanes {
                family: g.family,
                propagators: n,
                rank: g.rank,
                points: g.points.clone(),
                keys: g.keys.clone(),
                shifts,
                masses: g.masses.iter().map(complex).collect(),
                scales: g.scales.iter().map(complex).collect(),
                s_matrix,
                coefficients: g.coefficients.iter().map(complex).collect(),
            }
        })
        .collect()
}

/// `groups` in double-double (`bits` 106, the lifted values rounded once,
/// exactly as they are representable) or at `bits` MPFR bits, reduced by
/// TIR (or, with `ninjars`, by NinjaRS's generic engine in that carrier),
/// against NinjaRS's generic engine at `digits` digits on the same
/// consistent inputs: the worst relative deviation over every lane (all of
/// them must be served) and its (group, lane).
fn extended_against_truth(
    groups: &[FamilyLanes<f64>],
    bits: u32,
    digits: u32,
    poles: Poles,
) -> f64 {
    extended_worst(groups, bits, digits, poles, false).0
}

fn extended_worst(
    groups: &[FamilyLanes<f64>],
    bits: u32,
    digits: u32,
    poles: Poles,
    ninjars: bool,
) -> (f64, (usize, usize)) {
    let _unlock = pyamplicol::unlock();
    let truth_bits = Float::decimal_digits_to_bits(f64::from(digits)).unwrap();
    let exact = lifted(groups, truth_bits.max(bits));
    let mut ninja = V2Backends::NinjaRs(V2ReductionBackend {
        reducer: V2Reducer::new("arbitrary", Some(digits), "generic", true).unwrap(),
        masters: PoleSelection::BOTH,
    });
    V2Backend::<Float>::begin(&mut ninja);
    let (truth, _) = V2Backend::<Float>::reduce(&mut ninja, &exact, poles, None).unwrap();
    let mut backend = if !ninjars {
        tir()
    } else if bits == 106 {
        V2Backends::NinjaRs(V2ReductionBackend {
            reducer: V2Reducer::new("double-double", None, "generic", true).unwrap(),
            masters: PoleSelection::BOTH,
        })
    } else {
        let digits = (f64::from(bits) * std::f64::consts::LOG10_2).floor() as u32;
        V2Backends::NinjaRs(V2ReductionBackend {
            reducer: V2Reducer::new("arbitrary", Some(digits), "generic", true).unwrap(),
            masters: PoleSelection::BOTH,
        })
    };
    let found: Vec<Vec<Vec<Complex<Float>>>> = if bits == 106 {
        let dd: Vec<FamilyLanes<DoubleFloat>> = lifted(groups, 106)
            .iter()
            .map(|g| {
                let c = |v: &[Complex<Float>]| v.iter().map(Complex::to_double_float).collect();
                FamilyLanes {
                    family: g.family,
                    propagators: g.propagators,
                    rank: g.rank,
                    points: g.points.clone(),
                    keys: g.keys.clone(),
                    shifts: c(&g.shifts),
                    masses: c(&g.masses),
                    scales: c(&g.scales),
                    s_matrix: c(&g.s_matrix),
                    coefficients: c(&g.coefficients),
                }
            })
            .collect();
        V2Backend::<DoubleFloat>::begin(&mut backend);
        let (found, _) = V2Backend::<DoubleFloat>::reduce(&mut backend, &dd, poles, None).unwrap();
        // Each double-double value exactly, at the truth's precision.
        let zero = Float::with_val(truth_bits, 0.);
        let exact = |x: &DoubleFloat| zero.from_rational(&NativeFloat::exact_rational(x).unwrap());
        let up = |z: &Complex<DoubleFloat>| Complex::new(exact(&z.re), exact(&z.im));
        found
            .into_iter()
            .map(|g| g.into_iter().map(|l| l.unwrap().iter().map(up).collect()).collect())
            .collect()
    } else {
        let mp = lifted(groups, bits);
        V2Backend::<Float>::begin(&mut backend);
        let (found, _) = V2Backend::<Float>::reduce(&mut backend, &mp, poles, None).unwrap();
        found
            .into_iter()
            .map(|g| g.into_iter().map(|l| l.unwrap()).collect())
            .collect()
    };
    let norm = |z: &Complex<Float>| (z.re.clone() * &z.re + &(z.im.clone() * &z.im)).sqrt();
    let mut worst: (f64, (usize, usize)) = (0., (0, 0));
    for (g, (a, b)) in found.iter().zip(&truth).enumerate() {
        for (lane, (x, y)) in a.iter().zip(b).enumerate() {
            let y = y.as_ref().unwrap();
            let scale = y.iter().map(norm).fold(0f64, |m, v| m.max(v.to_f64()));
            for (p, q) in x.iter().zip(y) {
                let deviation = norm(&(p.clone() - q)).to_f64() / scale;
                if deviation > worst.0 {
                    worst = (deviation, (g, lane));
                }
            }
        }
    }
    worst
}

/// The extended carriers (m-lit lit-tir-mp, DESIGN §11.1): T1's families
/// (tadpoles to boxes, massless, massive and complex masses, ranks up to
/// four), all Laurent orders and the UV residue, in double-double and at
/// 256 bits on inputs consistent to the carrier, against NinjaRS's generic
/// engine at 80 digits on the same exact inputs: every lane served (the
/// carriers' gates, `v2_tir::carrier_gate`); double-double against its
/// target 1e-24 and 256 bits against 1e-60, measured and printed with
/// NinjaRS's generic engine in the same carrier for comparison (R5: a
/// target, asserted within a factor of ten).
#[test]
fn extended_carriers_match_ninjars_at_80_digits() {
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
    let mut complex = point(3, &[0., 0., 0.], &mut rng);
    complex.masses = vec![C::new(173. * 173., -173. * 1.5); 3];
    groups.push(lanes(family, &[complex], 3, 3, 2, &mut rng));
    for (bits, target) in [(106, 1e-24), (256, 1e-60)] {
        let (deviation, (g, lane)) = extended_worst(&groups, bits, 80, Poles::Split, false);
        let (ninjars, at) = extended_worst(&groups, bits, 80, Poles::Split, true);
        eprintln!(
            "T1 at {bits} bits vs NinjaRS generic at 80 digits: TIRust worst relative {deviation:e} \
             (family {g}: {} propagators, rank {}, lane {lane}); NinjaRS generic in the same carrier \
             {ninjars:e} (family {}, lane {})",
            groups[g].propagators, groups[g].rank, at.0, at.1
        );
        // The target (DESIGN §11.1's DD = MPFR, 1e-24; R5: measured and
        // recorded): within a factor of a few of it, as NinjaRS's generic
        // engine in the same carrier.
        assert!(deviation < 10. * target, "{bits} bits: {deviation:e}");
    }
}

/// Every carrier's gate is the binary64 one carried to its precision
/// (`carrier_gate`): the same share of the significand.
#[test]
fn carrier_gates_keep_the_share_of_the_significand() {
    assert_eq!(carrier_gate(1e-12, 53), 1e-12);
    for bits in [106u32, 256, 512] {
        let gate = carrier_gate(1e-12, bits);
        let share = gate.log2() / -f64::from(bits);
        assert!((share - 1e-12f64.log2() / -53.).abs() < 1e-12, "{bits}: {share}");
    }
    assert!((carrier_gate(1e-12, 106) / 1e-24 - 1.).abs() < 1e-12);
}

/// Near-forward boxes of a top loop with massless legs (exact Pythagorean
/// momenta, so every massless invariant is exactly zero: a small Gram
/// determinant, no on-shell decision), ranks 4 and 6: rows the conditioning
/// gates refuse.
fn near_forward_boxes() -> Vec<FamilyLanes<f64>> {
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
            groups.push(lanes(
                family,
                std::slice::from_ref(&point),
                rank,
                rank,
                2,
                &mut Lcg(29),
            ));
            family += 1;
        }
    }
    for g in &groups {
        // The massless legs' invariants are exactly zero.
        for (i, j) in [(0, 1), (1, 2), (2, 3), (3, 0)] {
            assert_eq!(g.s_matrix[i * 4 + j].re, 0.);
        }
    }
    groups
}

/// The default settings serve no method-choice row (DECISIONS 2026-11-07
/// ~17:20 (d)): the near-forward boxes the conditioning gates refuse fail
/// closed with the gates' reason, in the direct and in the check frame,
/// and the statistics count no method-choice row; the lanes of the rows the
/// gates pass agree with NinjaRS at 80 digits.
#[test]
fn the_default_serves_no_method_choice_row() {
    let groups = near_forward_boxes();
    let mut backend = tir();
    let (found, check) = reduce(
        &mut backend,
        &groups,
        Poles::All,
        Some(ReductionCheck::REVERSED),
    );
    let statistics = V2Backend::<f64>::take_statistics(&mut backend);
    eprintln!("default statistics: {:?}", statistics.counts);
    assert!(
        count(&statistics, "TIR gate refusals") > 0,
        "{statistics:?}"
    );
    for name in [
        "TIR method-choice rows",
        "TIR method-choice rows with expansions",
        "TIR method-choice check rows",
        "TIR method-choice lanes refused by their check",
    ] {
        assert_eq!(count(&statistics, name), 0, "{name}: {statistics:?}");
    }
    let truth = reference_arbitrary(&groups, Poles::All, 80);
    let check = check.unwrap();
    let mut refused = 0;
    for (g, a) in found.iter().enumerate() {
        for (lane, x) in a.iter().enumerate() {
            match x {
                Ok(x) => {
                    let error = relative(x, truth[g][lane].as_ref().unwrap());
                    assert!(error < 1e-12, "family {g} lane {lane}: {error:e}");
                }
                Err(reason) => {
                    refused += 1;
                    assert!(
                        reason.contains("refused by the conditioning gates")
                            && reason.contains("method choice is off"),
                        "{reason}"
                    );
                }
            }
            if let Err(reason) = &check[g][lane] {
                assert!(reason.contains("TIRust"), "{reason}");
            }
        }
    }
    assert!(refused > 0);
}

/// With the method choice on (`TirOptions::method_choice`, the option
/// lit-tir-mp turns back on), the rows the conditioning gates refuse take
/// it (alternative PV first within the expansions' tolerance) and are
/// verified by their own check row: the near-forward boxes agree with
/// NinjaRS's generic engine at 80 digits within their estimates, and the
/// statistics count them.
#[test]
fn method_choice_rows_are_checked() {
    let groups = near_forward_boxes();
    let mut backend = tir_method_choice();
    let (found, check) = reduce(
        &mut backend,
        &groups,
        Poles::All,
        Some(ReductionCheck::REVERSED),
    );
    let statistics = V2Backend::<f64>::take_statistics(&mut backend);
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
                        p.iter()
                            .zip(q)
                            .map(|(a, b)| size(&(*a - b)))
                            .fold(0., f64::max)
                    };
                    let truth_error = spread(x, z) / scale;
                    eprintln!(
                        "family {g} lane {lane}: vs 80 digits {truth_error:e}, TIR estimate {:e}, \
                         NinjaRS {:e} (estimate {:e})",
                        spread(x, y) / scale,
                        spread(n, z) / scale,
                        spread(n, n_check) / scale
                    );
                    assert!(
                        truth_error < 1e-12,
                        "family {g} lane {lane}: {truth_error:e}"
                    );
                    let one = |v: &LaneOutcome<f64>| vec![vec![v.clone()]];
                    within_estimates(
                        &one(&found[g][lane]),
                        &one(&check[g][lane]),
                        &one(&truth[g][lane]),
                    );
                }
                // A refused lane fails closed with TIRust's or the check's reason.
                (Err(reason), _) | (_, Err(reason)) => {
                    assert!(reason.contains("TIRust"), "{reason}")
                }
            }
        }
    }
    assert!(served > 0);
    assert!(
        count(&statistics, "TIR gate refusals") > 0,
        "{:?}",
        statistics.counts
    );
    assert!(
        count(&statistics, "TIR method-choice rows") > 0,
        "{:?}",
        statistics.counts
    );
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
            let groups = vec![lanes(
                0,
                std::slice::from_ref(&point),
                rank,
                rank,
                2,
                &mut Lcg(29),
            )];
            let truth = reference_arbitrary(&groups, Poles::All, 80);
            let dd = reference(&groups, Poles::All);
            for lane in 0..2 {
                eprintln!(
                    "m {m} rank {rank}: lane {lane} double-double vs 80 digits {:e}",
                    relative(
                        dd[0][lane].as_ref().unwrap(),
                        truth[0][lane].as_ref().unwrap()
                    )
                );
            }
            for alternative in [Some(iregi_rust::direct::EXPANSION_TOLERANCE), None] {
                for check in [ReductionCheck::REVERSED, ReductionCheck::RESCALED] {
                    let options = TirOptions {
                        method_choice: true,
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
