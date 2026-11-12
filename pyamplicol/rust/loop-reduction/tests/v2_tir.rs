// SPDX-License-Identifier: 0BSD

//! The TIR backend (`rusticol_loop_reduction::v2_tir`, m-lit lit-tir):
//! T1 scalar and tensor integrals against NinjaRS's generic engine in
//! double-double, T3 the layout permutation and the split UV column, T4 rank
//! over-declaration, T5 check frames, T6 chunking and batch composition, T10
//! unit scaling, T11 the rotation copy (bitwise the separate pass), the fail-closed gate refusals without the method choice,
//! the binary64 default's method choice and its rescaled acceptance (the
//! rotation-blind lanes of lit-tir report §H refused), and the typed
//! capability refusals. Binary64 TIR lanes are compared within the lax-precision rule
//! (relative to the lane's largest Laurent coefficient), never bitwise
//! except where an identity makes them so.
#![cfg(feature = "tir")]

use iregi_rust::DirectCapabilities;
use ninjars::fast::tensor_layout::{monomials, ncoeff};
use rusticol_core::RusticolErrorKind;
use rusticol_core::loop_execution::v2::{FamilyLanes, LaneKey, LaneOutcome, Poles, V2Backend};
use rusticol_core::loop_v2::retest::ReductionCheck;
use rusticol_loop_reduction::NativeFloat;
use rusticol_loop_reduction::runtime::{
    PoleSelection, Reduction, V2Backends, V2Reducer, V2ReductionBackend,
};
use rusticol_loop_reduction::v2_tir::{
    TirOptions, TirReductionBackend, capability_refusal, carrier_gate, layout_positions, tir_frame,
};
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

/// The TIR backend in its default settings (binary64: the method choice on,
/// its lanes accepted in the rescaled frame, m-lit lit-tir-mp; the extended
/// carriers: gate-refused rows fail closed).
fn tir() -> V2Backends {
    V2Backends::TiRust(Box::new(TirReductionBackend::new(TirOptions::default())))
}

/// The TIR backend without the method choice (gate-refused rows fail
/// closed, DECISIONS 2026-11-07 ~17:20 (d)) and otherwise the defaults.
fn tir_without_method_choice() -> V2Backends {
    V2Backends::TiRust(Box::new(TirReductionBackend::new(TirOptions {
        method_choice: false,
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

/// Whether a lane failed closed on TIRust's conditioning gates (without
/// the method choice: DECISIONS 2026-11-07 ~17:20 (d)), on the method
/// choice or on its acceptance check; any other failure panics.
fn gate_refused(lane: &LaneOutcome<f64>) -> bool {
    match lane {
        Ok(_) => false,
        Err(reason) => {
            assert!(
                reason.contains("refused by the conditioning gates")
                    || reason.contains("refused by the gates, then by the method choice")
                    || reason.contains("method-choice lane refused by its check"),
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
    // Without the method choice (gate-refused rows fail closed) and the
    // default, the method choice on (every lane served: the boxes the gates
    // refuse too).
    for (name, mut backend) in [
        ("method choice off", tir_without_method_choice()),
        ("default", tir()),
    ] {
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
        if name == "default" {
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
    // Without the method choice (these boxes may fail closed on the gates,
    // in every pole mode alike) and the default, the method choice on
    // (every lane served).
    let configurations: [(&str, fn() -> V2Backends); 2] = [
        ("method choice off", tir_without_method_choice),
        ("default", tir),
    ];
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
        if name == "default" {
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

/// T5: TIR's check frame of a reduction check rotates the propagators
/// (q -> -q leaves its moments bitwise): without `rescale` the relabelling
/// alone (TIRust's rotation copy), with it the rotation with the origin
/// shift at rescaled inputs; each is not blind on a generic box and agrees
/// with the direct frame.
#[test]
fn check_frames_rotate_and_rescale() {
    let frame = |shift, rescale| rusticol_loop_reduction::runtime::ReductionFrame {
        reverse: false,
        rotation: 1,
        shift,
        rescale,
    };
    assert_eq!(tir_frame(ReductionCheck::REVERSED), frame(false, false));
    assert_eq!(tir_frame(ReductionCheck::RESCALED), frame(true, true));
    let mut rng = Lcg(13);
    let points: Vec<_> = (0..2)
        .map(|_| point(4, &[0., 0., 0., 0.], &mut rng))
        .collect();
    let groups = vec![lanes(0, &points, 3, 3, 2, &mut rng)];
    for check in [ReductionCheck::REVERSED, ReductionCheck::RESCALED] {
        for mut backend in [tir_without_method_choice(), tir()] {
            let (direct, again) = reduce(&mut backend, &groups, Poles::All, Some(check));
            let again = again.unwrap();
            let (deviation, compared) = worst(&again, &direct);
            assert!(compared > 0, "{check:?}: no lane compared");
            assert!(deviation > 0., "{check:?}: the check frame is blind");
            assert!(deviation < 1e-10, "{check:?}: {deviation:e}");
        }
    }
}

/// The TIR backend with the rotation copy off (`TirOptions::rotation_copy`:
/// a relabelled check frame is a separate pass) and the method choice as
/// given.
fn tir_separate(method_choice: bool) -> V2Backends {
    V2Backends::TiRust(Box::new(TirReductionBackend::new(TirOptions {
        method_choice,
        rotation_copy: false,
        ..TirOptions::default()
    })))
}

/// `group` with its propagators relabelled cyclically by `rotation`
/// (propagator `k` is the family's `k + rotation`): the same integrals in
/// the same loop-momentum frame, its coefficients unchanged.
fn relabelled(group: &FamilyLanes<f64>, rotation: usize) -> FamilyLanes<f64> {
    let n = group.propagators;
    let from = |k: usize| (k + rotation) % n;
    let mut out = copy(group);
    for lane in 0..group.keys.len() {
        for k in 0..n {
            for mu in 0..4 {
                out.shifts[lane * 4 * n + 4 * k + mu] =
                    group.shifts[lane * 4 * n + 4 * from(k) + mu];
            }
            out.masses[lane * n + k] = group.masses[lane * n + from(k)];
            for j in 0..n {
                out.s_matrix[lane * n * n + k * n + j] =
                    group.s_matrix[lane * n * n + from(k) * n + from(j)];
            }
        }
    }
    out
}

/// Groups whose rows the conditioning gates refuse in the direct frame
/// alone, in the relabelled frame alone and in both (every cyclic
/// relabelling of the near-forward boxes and of lit-tir's rotation-blind
/// row: where one order is refused and the next one passes, some
/// relabelling passes in its direct frame and is refused in its copy), next
/// to generic triangles, boxes and pentagons, massless and massive, every
/// rank to 4.
fn copy_groups() -> Vec<FamilyLanes<f64>> {
    let mut groups = Vec::new();
    for group in near_forward_boxes().iter().chain(&rotation_blind_row()) {
        groups.extend((0..group.propagators).map(|rotation| relabelled(group, rotation)));
    }
    let mut rng = Lcg(31);
    let mt2 = 173. * 173.;
    for masses in [
        vec![0.; 3],
        vec![mt2; 3],
        vec![0.; 4],
        vec![mt2, 0., mt2, 0.],
        vec![0.; 5],
        vec![mt2; 5],
    ] {
        let points: Vec<_> = (0..3)
            .map(|_| point(masses.len(), &masses, &mut rng))
            .collect();
        for rank in [0, 2, 4] {
            groups.push(lanes(0, &points, rank, 4, 2, &mut rng));
        }
    }
    for (family, group) in groups.iter_mut().enumerate() {
        group.family = family;
        for key in &mut group.keys {
            key.family = family;
        }
    }
    groups
}

/// T11 (the rotation copy, m-lit lit-stab-t step C): the reduction-direction
/// check of TIR is TIRust's rotation copy from the direct pass's own calls,
/// bit for bit the separate pass in the same relabelled frame: with and
/// without the method choice (rows refused together with their copy, the
/// direct row alone and the copy alone, all exercised), in binary64 and in
/// double-double; the direct lanes are bitwise those of a call without a
/// check, the rescaled check does not use it, and the statistics count the
/// same rows and refusals.
#[test]
fn the_rotation_copy_is_the_separate_pass() {
    let groups = copy_groups();
    let check = Some(ReductionCheck::REVERSED);
    let gate_refused = |x: &LaneOutcome<f64>| matches!(x, Err(reason) if reason.contains("refused by the conditioning gates"));
    let (direct, again) = reduce(&mut tir_separate(false), &groups, Poles::All, check);
    let again = again.unwrap();
    let mut kinds = [0usize; 3];
    for (a, b) in direct.iter().flatten().zip(again.iter().flatten()) {
        match (gate_refused(a), gate_refused(b)) {
            (true, true) => kinds[0] += 1,
            (true, false) => kinds[1] += 1,
            (false, true) => kinds[2] += 1,
            (false, false) => {}
        }
    }
    eprintln!("gate-refused lanes (both, direct alone, copy alone): {kinds:?}");
    assert!(kinds.iter().all(|&k| k > 0), "{kinds:?}");
    let rows = [
        "tensor moment rows",
        "TIR gate refusals",
        "TIR method-choice rows",
    ];
    for method_choice in [false, true] {
        let mut copied = V2Backends::TiRust(Box::new(TirReductionBackend::new(TirOptions {
            method_choice,
            ..TirOptions::default()
        })));
        let mut separate = tir_separate(method_choice);
        let (a, a_check) = reduce(&mut copied, &groups, Poles::All, check);
        let (b, b_check) = reduce(&mut separate, &groups, Poles::All, check);
        assert_eq!(a, b, "method choice {method_choice}");
        assert_eq!(a_check, b_check, "method choice {method_choice}");
        let a_statistics = V2Backend::<f64>::take_statistics(&mut copied);
        let b_statistics = V2Backend::<f64>::take_statistics(&mut separate);
        for name in rows {
            assert_eq!(
                count(&a_statistics, name),
                count(&b_statistics, name),
                "{name}: {:?} {:?}",
                a_statistics.counts,
                b_statistics.counts
            );
        }
        let (plain, _) = reduce(&mut tir_separate(method_choice), &groups, Poles::All, None);
        assert_eq!(a, plain, "method choice {method_choice}");
        let served = a_check
            .unwrap()
            .iter()
            .flatten()
            .filter(|x| x.is_ok())
            .count();
        assert!(served > 0);
        // The rescaled check is a pass in its own frame either way.
        let rescaled = Some(ReductionCheck::RESCALED);
        let (a, a_check) = reduce(&mut copied, &groups, Poles::All, rescaled);
        let (b, b_check) = reduce(&mut separate, &groups, Poles::All, rescaled);
        assert_eq!((a, a_check), (b, b_check), "method choice {method_choice}");
    }
    let lifted: Vec<_> = groups.iter().map(dd).collect();
    let dd_reduce = |backend: &mut V2Backends| {
        V2Backend::<DoubleFloat>::begin(backend);
        V2Backend::<DoubleFloat>::reduce(backend, &lifted, Poles::All, check).unwrap()
    };
    let (a, a_check) = dd_reduce(&mut tir());
    let (b, b_check) = dd_reduce(&mut tir_separate(true));
    assert_eq!(a, b);
    assert_eq!(a_check, b_check);
    assert!(a_check.unwrap().iter().flatten().any(|x| x.is_ok()));
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
        let refused =
            match TirReductionBackend::with_histogram([(n, 0)], bits, TirOptions::default()) {
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
    // TIRust has no rank <= N + 2 limit (m-lit lit-eft): a family beyond
    // NinjaRS's rank is TIRust's in every carrier, and the artifacts admit
    // it; beyond TIRust's own representation it is refused with its reason.
    for bits in [53, 106, 256] {
        assert_eq!(capability_refusal(4, 7, bits), None);
        assert_eq!(capability_refusal(5, 11, bits), None);
    }
    let max_rank = usize::from(DirectCapabilities::of_representation(53).max_rank);
    let reason = capability_refusal(1, max_rank + 1, 53).unwrap();
    assert!(
        reason.contains("TIRust cannot reduce 1 propagators"),
        "{reason}"
    );
    assert_eq!(
        rusticol_core::loop_v2::bounds::rank_refusal(Reduction::TiRust, 1, max_rank + 1),
        Some(reason)
    );
    for (n, rank) in [(13, 2), (4, 7)] {
        assert_eq!(
            rusticol_loop_reduction::runtime::rank_refusal(Reduction::TiRust, n, rank),
            None
        );
        assert_eq!(
            rusticol_core::loop_v2::bounds::admission_refusal(n, rank),
            None
        );
    }
    assert!(rusticol_loop_reduction::runtime::rank_refusal(Reduction::NinjaRs, 4, 7).is_some());
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
            .map(|g| {
                g.into_iter()
                    .map(|l| l.unwrap().iter().map(up).collect())
                    .collect()
            })
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
        assert!(
            (share - 1e-12f64.log2() / -53.).abs() < 1e-12,
            "{bits}: {share}"
        );
    }
    assert!((carrier_gate(1e-12, 106) / 1e-24 - 1.).abs() < 1e-12);
}

/// Near-forward boxes of a top loop with massless legs (exact Pythagorean
/// momenta, so every massless invariant is exactly zero: a small Gram
/// determinant, no on-shell decision), ranks 4 and 6: rows the conditioning
/// gates refuse.
fn near_forward_boxes() -> Vec<FamilyLanes<f64>> {
    forward_boxes(&[100., 1000., 9000.])
}

/// [`near_forward_boxes`] at the angles `sin(theta) = 2m / (m^2 + 1)` of
/// `ms`.
fn forward_boxes(ms: &[f64]) -> Vec<FamilyLanes<f64>> {
    let mut groups = Vec::new();
    let mut family = 0;
    let mt2 = C::new(173. * 173., 0.);
    for &m in ms {
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

/// Without the method choice no method-choice row is served (DECISIONS
/// 2026-11-07 ~17:20 (d)): the near-forward boxes the conditioning gates
/// refuse fail closed with the gates' reason, in the direct and in the
/// check frame, and the statistics count no method-choice row; the lanes of
/// the rows the gates pass agree with NinjaRS at 80 digits.
#[test]
fn without_the_method_choice_gate_refused_rows_fail_closed() {
    let groups = near_forward_boxes();
    let mut backend = tir_without_method_choice();
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

/// The binary64 default (`TirOptions::method_choice` on, m-lit
/// lit-tir-mp): the rows the conditioning gates refuse take the method
/// choice (alternative PV first within the expansions' tolerance) and are
/// verified in the rescaled acceptance frame: the near-forward boxes agree
/// with NinjaRS's generic engine at 80 digits within their estimates, and
/// the statistics count them.
#[test]
fn method_choice_rows_are_checked() {
    let groups = near_forward_boxes();
    let mut backend = tir();
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

/// One of lit-tir's rotation-blind method-choice rows (report §H.1; the
/// fixture `tir_rotation_blind_row.json`: synthetic LI-32 N4-massive row 24
/// at rank 4, 11 lanes): a massive box the gates refuse on a massive
/// sub-triangle. Its S-matrix is the binary64 rounding of the invariants
/// formed exactly from the binary64 shifts (the inputs of production, and
/// consistent with the lifted truth, lit-onshell-probe's rule).
fn rotation_blind_row() -> Vec<FamilyLanes<f64>> {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/tir_rotation_blind_row.json")).unwrap();
    let number = |v: &serde_json::Value| v.as_str().unwrap().parse::<f64>().unwrap();
    let pair = |v: &serde_json::Value| C::new(number(&v[0]), number(&v[1]));
    let shifts: Vec<[f64; 4]> = data["shifts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| std::array::from_fn(|mu| number(&v[mu])))
        .collect();
    let n = shifts.len();
    let masses: Vec<C> = data["masses"]
        .as_array()
        .unwrap()
        .iter()
        .map(pair)
        .collect();
    let mu2 = number(&data["mu2"]);
    let rank = data["rank"].as_u64().unwrap() as usize;
    let exact = |x: f64| Float::with_val(256, x);
    let mut s_matrix = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let d = |mu: usize| exact(shifts[i][mu]) - &exact(shifts[j][mu]);
            let square = |mu: usize| d(mu) * &d(mu);
            let x = square(0) - &square(1) - &square(2) - &square(3);
            s_matrix.push(C::new(x.to_f64(), 0.));
        }
    }
    let lanes = data["lanes"].as_array().unwrap();
    let mut group = FamilyLanes {
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
    for (column, lane) in lanes.iter().enumerate() {
        group.points.push(0);
        group.keys.push(LaneKey {
            family: 0,
            order: 0,
            column,
        });
        for v in &shifts {
            group.shifts.extend(v.iter().map(|&x| C::new(x, 0.)));
        }
        group.masses.extend(masses.iter().copied());
        group.scales.push(C::new(mu2, 0.));
        group.s_matrix.extend(s_matrix.iter().copied());
        let coefficients = lane["coefficients"].as_array().unwrap();
        assert_eq!(coefficients.len(), ncoeff(rank));
        group.coefficients.extend(coefficients.iter().map(pair));
    }
    vec![group]
}

/// NinjaRS's generic engine at `digits` digits on the groups lifted
/// consistently (`lifted`: the S-matrix formed from the lifted shifts),
/// read back in binary64.
fn consistent_truth(groups: &[FamilyLanes<f64>], digits: u32) -> Lanes {
    let _unlock = pyamplicol::unlock();
    let bits = Float::decimal_digits_to_bits(f64::from(digits)).unwrap();
    let exact = lifted(groups, bits);
    let mut ninja = V2Backends::NinjaRs(V2ReductionBackend {
        reducer: V2Reducer::new("arbitrary", Some(digits), "generic", true).unwrap(),
        masters: PoleSelection::BOTH,
    });
    V2Backend::<Float>::begin(&mut ninja);
    let (truth, _) = V2Backend::<Float>::reduce(&mut ninja, &exact, Poles::All, None).unwrap();
    truth
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

/// The target accuracy's calibrated acceptance (m-lit F3 (a),
/// lit-tirust-default): the default settings take the default target's
/// acceptance, and the calibration factor is the largest true-error to
/// frame-deviation ratio of the calibration record (`fixtures/
/// tir_target_calibration.json`: lit-tir-mp's synthetic lanes and the
/// loop-induced stream rows at the point level, each against a consistent
/// truth), rounded up to two significant digits.
#[test]
fn the_acceptance_follows_the_calibrated_target() {
    use rusticol_loop_reduction::reduction_policy::DEFAULT_TARGET;
    use rusticol_loop_reduction::v2_tir::{UNDERSTATEMENT, acceptance_of};
    let record: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/tir_target_calibration.json")).unwrap();
    let worst = record["sources"]
        .as_array()
        .unwrap()
        .iter()
        .map(|source| source["max_ratio"].as_f64().unwrap())
        .fold(0.0, f64::max);
    assert!(worst > 0.0 && UNDERSTATEMENT >= worst, "{worst}");
    let unit = 10f64.powi(worst.log10().floor() as i32 - 1);
    assert_eq!(UNDERSTATEMENT, (worst / unit).ceil() * unit);
    assert_eq!(record["understatement"].as_f64(), Some(UNDERSTATEMENT));
    assert_eq!(
        TirOptions::default().acceptance,
        acceptance_of(DEFAULT_TARGET)
    );
    assert_eq!(
        TirOptions::for_target(1e-7).acceptance,
        1e-7 / UNDERSTATEMENT
    );
    assert_eq!(
        TirOptions {
            acceptance: TirOptions::default().acceptance,
            ..TirOptions::for_target(1e-7)
        },
        TirOptions::default()
    );
}

/// The default's rescaled acceptance (m-lit lit-tir-mp, ruling 21:50 R4)
/// against a rotated one on a rotation-blind row of lit-tir report §H.1:
/// with a rotated acceptance frame the method choice serves lanes whose
/// error against the consistent 60-digit truth is far beyond their rotated
/// re-test (blind lanes, at this TIRust pin); the default refuses every
/// such lane with its check's reason and serves nothing beyond 1e-5 of the
/// truth.
#[test]
fn the_rescaled_acceptance_refuses_the_rotation_blind_lanes() {
    let groups = rotation_blind_row();
    let truth = consistent_truth(&groups, 60);
    let mut rotated = V2Backends::TiRust(Box::new(TirReductionBackend::new(TirOptions {
        acceptance_frame: tir_frame(ReductionCheck::REVERSED),
        ..TirOptions::default()
    })));
    // The call's check in the rotated acceptance frame: its row is the one
    // the acceptance compared with.
    let (served, again) = reduce(
        &mut rotated,
        &groups,
        Poles::All,
        Some(ReductionCheck::REVERSED),
    );
    let again = again.unwrap();
    let (default, _) = reduce(&mut tir(), &groups, Poles::All, None);
    let statistics = V2Backend::<f64>::take_statistics(&mut rotated);
    assert!(
        count(&statistics, "TIR method-choice rows") > 0,
        "{:?}",
        statistics.counts
    );
    let mut blind = 0;
    for lane in 0..groups[0].keys.len() {
        let z = truth[0][lane].as_ref().unwrap();
        if let (Ok(x), Ok(y)) = (&served[0][lane], &again[0][lane]) {
            let error = relative(x, z);
            let estimate = relative(y, x);
            eprintln!(
                "lane {lane}: rotated acceptance serves it, {error:e} from the truth, rotated \
                 re-test {estimate:e}; default: {}",
                match &default[0][lane] {
                    Ok(v) => format!("served, {:e} from the truth", relative(v, z)),
                    Err(reason) => reason.chars().take(90).collect(),
                }
            );
            if error > 1e-8 && error > 10. * estimate {
                blind += 1;
                let reason = default[0][lane]
                    .as_ref()
                    .expect_err("a blind lane is refused");
                assert!(
                    reason.contains("method-choice lane refused by its check"),
                    "lane {lane}: {reason}"
                );
            }
        }
        match &default[0][lane] {
            Ok(v) => assert!(relative(v, z) < 1e-5, "lane {lane}"),
            Err(reason) => assert!(reason.contains("TIRust"), "{reason}"),
        }
    }
    // The rotated frame was blind on some of these lanes at the pins of
    // lit-tir and lit-tir-mp (lit-tir report §H.1; 21 lanes): the count is
    // recorded, not asserted (a TIRust pin that serves them right makes it
    // zero), and every lane the rotated frame misses is refused by the
    // default above.
    eprintln!("rotation-blind lanes at this TIRust pin: {blind}");
}

/// m-lit ux-polish item 4: the refused lanes' measures come as data next
/// to the unchanged reasons (`V2Backend::take_lane_measures`): a gate-refused
/// lane without the method choice carries TIRust's conditioning estimate
/// (direct and check frame), a lane the method choice's acceptance refuses
/// its deviation relative to its term scale (the numbers its reason states);
/// a served lane carries none, and the next call starts afresh.
#[test]
fn refused_lanes_state_their_measures_as_data() {
    use rusticol_core::loop_execution::v2::MeasureKind;

    let groups = near_forward_boxes();
    let mut backend = tir_without_method_choice();
    let (found, check) = reduce(
        &mut backend,
        &groups,
        Poles::All,
        Some(ReductionCheck::REVERSED),
    );
    let check = check.unwrap();
    let measures = V2Backend::<f64>::take_lane_measures(&mut backend);
    let mut refused = 0;
    for measured in &measures {
        let outcome = if measured.check {
            &check[measured.group][measured.lane]
        } else {
            &found[measured.group][measured.lane]
        };
        assert!(outcome.is_err(), "a measure belongs to a refused lane");
        assert_eq!(measured.measure.kind, MeasureKind::GramConditioning);
        assert!(measured.measure.value >= 0.0 && measured.measure.value < 1.0);
    }
    for (g, lanes) in found.iter().enumerate() {
        for (lane, outcome) in lanes.iter().enumerate() {
            if outcome.is_err() {
                refused += 1;
                assert!(
                    measures
                        .iter()
                        .any(|m| !m.check && m.group == g && m.lane == lane),
                    "family {g} lane {lane}: no measure"
                );
            }
        }
    }
    assert!(refused > 0);
    assert!(V2Backend::<f64>::take_lane_measures(&mut backend).is_empty());

    let groups = rotation_blind_row();
    let mut backend = tir();
    let (found, _) = reduce(&mut backend, &groups, Poles::All, None);
    let measures = V2Backend::<f64>::take_lane_measures(&mut backend);
    let mut deviations = 0;
    for (lane, outcome) in found[0].iter().enumerate() {
        let measure = measures
            .iter()
            .find(|m| !m.check && m.group == 0 && m.lane == lane);
        match outcome {
            Ok(_) => assert!(measure.is_none(), "lane {lane}"),
            Err(reason) if reason.contains("method-choice lane refused by its check") => {
                deviations += 1;
                let measure = measure.expect("a deviation measure").measure;
                assert_eq!(measure.kind, MeasureKind::MethodChoiceDeviation);
                // The numbers the reason states (this test only reads them
                // to compare; nothing parses them in production).
                let number = |after: &str| -> f64 {
                    let start = reason.find(after).unwrap() + after.len();
                    reason[start..].split(' ').next().unwrap().parse().unwrap()
                };
                let stated = number("deviation ") / number("of term scale ");
                assert!(
                    (measure.value - stated).abs() <= 1e-12 * stated,
                    "lane {lane}: {} vs {stated}",
                    measure.value
                );
            }
            Err(_) => {}
        }
    }
    eprintln!("method-choice deviation measures: {deviations}");
}

/// `TirOptions::store` (m-lit lit-cache STEP 3): a box family and its
/// pinched triangle (the box's first three propagators, the same shifts)
/// at the same points share TIRust's per-subset results through the
/// backend's tensor store; every lane and every check lane is bitwise the
/// store-off value, in one call and over several point chunks (`begin`
/// clears the store), and the store's reuse is counted.
#[test]
fn the_tensor_store_leaves_every_lane_bitwise_and_shares_pinched_subsets() {
    let mut rng = Lcg(29);
    let masses = [0., 0., 173. * 173., 0.];
    let box_points: Vec<_> = (0..3).map(|_| point(4, &masses, &mut rng)).collect();
    let pinched: Vec<_> = box_points
        .iter()
        .map(|p| Point {
            shifts: p.shifts[..3].to_vec(),
            masses: p.masses[..3].to_vec(),
            mu2: p.mu2,
        })
        .collect();
    let groups = [
        lanes(0, &box_points, 4, 4, 3, &mut rng),
        lanes(1, &pinched, 3, 3, 2, &mut rng),
    ];
    let stored = || {
        V2Backends::TiRust(Box::new(TirReductionBackend::new(TirOptions {
            store: true,
            ..TirOptions::default()
        })))
    };
    for check in [None, Some(ReductionCheck::RESCALED)] {
        let (off, off_check) = reduce(&mut tir(), &groups, Poles::All, check);
        let mut backend = stored();
        let (on, on_check) = reduce(&mut backend, &groups, Poles::All, check);
        assert_eq!(on, off, "{check:?}");
        assert_eq!(on_check, off_check, "{check:?}");
        // Again on the same backend: the next chunk starts from an empty
        // store and gives the same bits.
        let (again, again_check) = reduce(&mut backend, &groups, Poles::All, check);
        assert_eq!(again, off, "{check:?}");
        assert_eq!(again_check, off_check, "{check:?}");
        let statistics = V2Backend::<f64>::take_statistics(&mut backend);
        let reused = count(&statistics, "TIR store masters reused")
            + count(&statistics, "TIR store sub-integrals reused");
        assert!(reused > 0, "{check:?}: {:?}", statistics.counts);
        assert!(count(&statistics, "TIR store masters computed") > 0);
        // Without the store no store count is reported.
        let mut plain = tir();
        reduce(&mut plain, &groups, Poles::All, check);
        let statistics = V2Backend::<f64>::take_statistics(&mut plain);
        assert!(
            statistics
                .counts
                .iter()
                .all(|(name, _)| !name.starts_with("TIR store")),
            "{:?}",
            statistics.counts
        );
    }
}

/// m-lit lit-perf (lit-cache audit minor (iv)): with the tensor store on,
/// the rotation copy off and a check frame that only relabels the
/// propagators, the check pass (in its own workspaces, without the store)
/// recomputes every integral: every lane and every check lane is bitwise
/// the store-off value.
#[test]
fn a_check_pass_with_the_store_recomputes_its_integrals() {
    let mut rng = Lcg(37);
    let masses = [0., 0., 173. * 173., 0.];
    let box_points: Vec<_> = (0..3).map(|_| point(4, &masses, &mut rng)).collect();
    let pinched: Vec<_> = box_points
        .iter()
        .map(|p| Point {
            shifts: p.shifts[..3].to_vec(),
            masses: p.masses[..3].to_vec(),
            mu2: p.mu2,
        })
        .collect();
    let groups = [
        lanes(0, &box_points, 4, 4, 3, &mut rng),
        lanes(1, &pinched, 3, 3, 2, &mut rng),
    ];
    let backend = |store: bool| {
        V2Backends::TiRust(Box::new(TirReductionBackend::new(TirOptions {
            store,
            rotation_copy: false,
            ..TirOptions::default()
        })))
    };
    let relabelling = ReductionCheck {
        reverse: false,
        rotation: 1,
        shift: false,
        rescale: false,
    };
    let (off, off_check) = reduce(&mut backend(false), &groups, Poles::All, Some(relabelling));
    let (on, on_check) = reduce(&mut backend(true), &groups, Poles::All, Some(relabelling));
    assert_eq!(on, off);
    assert_eq!(on_check, off_check);
    assert!(off_check.is_some());
}

/// m-lit lit-perf (lit-cache audit minor (v)): the masters census counts
/// the store's master misses per family and those repeating an earlier miss
/// up to its relabellings (a box and its cyclically relabelled copy at the
/// same points: the copy's box masters repeat the box's up to a dihedral
/// relabelling), the totals are the families' sums, the values bitwise the
/// store-off ones; without the store it is refused typed.
#[test]
fn the_masters_census_counts_relabelled_repeats_and_needs_the_store() {
    let mut rng = Lcg(41);
    let masses = [0., 173. * 173., 173. * 173., 0.];
    let points: Vec<_> = (0..2).map(|_| point(4, &masses, &mut rng)).collect();
    let direct = lanes(0, &points, 2, 4, 2, &mut rng);
    let mut copy = relabelled(&direct, 2);
    copy.family = 1;
    for key in &mut copy.keys {
        key.family = 1;
    }
    let groups = [direct, copy];
    let census = || {
        V2Backends::TiRust(Box::new(TirReductionBackend::new(TirOptions {
            store: true,
            masters_census: true,
            ..TirOptions::default()
        })))
    };
    let (plain, _) = reduce(&mut tir(), &groups, Poles::All, None);
    let mut backend = census();
    let (counted, _) = reduce(&mut backend, &groups, Poles::All, None);
    assert_eq!(counted, plain);
    let statistics = V2Backend::<f64>::take_statistics(&mut backend);
    let misses = count(&statistics, "TIR masters census: box store misses");
    let repeats = count(&statistics, "TIR masters census: box canonical repeats");
    eprintln!("census: {:?}", statistics.counts);
    assert!(misses > 0, "{:?}", statistics.counts);
    assert!(repeats > 0 && repeats <= misses, "{:?}", statistics.counts);
    let total = |name: &str| {
        ["tadpole", "bubble", "triangle", "box"]
            .iter()
            .map(|family| count(&statistics, &format!("TIR masters census: {family} {name}")))
            .sum::<u64>()
    };
    assert_eq!(
        total("store misses"),
        count(&statistics, "TIR masters census: store misses")
    );
    assert_eq!(
        total("canonical repeats"),
        count(&statistics, "TIR masters census: canonical repeats")
    );
    let refused = TirReductionBackend::with_histogram(
        [(4, 2)],
        53,
        TirOptions {
            masters_census: true,
            ..TirOptions::default()
        },
    );
    assert!(
        refused
            .as_ref()
            .err()
            .is_some_and(|e| e.message().contains("masters_census")),
        "{:?}",
        refused.err().map(|e| e.message().to_owned())
    );
}

/// The TIR backend in the unchecked mode (`TirOptions::unchecked`, m-lit
/// lit-unchecked) and otherwise `options`.
fn tir_unchecked(options: TirOptions) -> V2Backends {
    V2Backends::TiRust(Box::new(TirReductionBackend::new(TirOptions {
        unchecked: true,
        ..options
    })))
}

/// The kinematic rows of `groups` (a family's consecutive lanes of one
/// point are one row).
fn kinematic_rows<T>(groups: &[FamilyLanes<T>]) -> u64 {
    groups
        .iter()
        .map(|g| {
            (0..g.points.len())
                .filter(|&k| k == 0 || g.points[k] != g.points[k - 1])
                .count() as u64
        })
        .sum()
}

/// The near-forward boxes and the rotation-blind row of the method-choice
/// tests, as one set of families (rows the gates refuse, rows the method
/// choice serves, rows its acceptance refuses), with T1's generic families.
fn unchecked_groups() -> Vec<FamilyLanes<f64>> {
    let mut groups = near_forward_boxes();
    for mut group in rotation_blind_row() {
        group.family = groups.len();
        for key in &mut group.keys {
            key.family = group.family;
        }
        groups.push(group);
    }
    let mut rng = Lcg(7);
    for (n, masses) in [
        (3, vec![173. * 173., 173. * 173., 173. * 173.]),
        (4, vec![0., 0., 0., 0.]),
        (5, vec![0., 0., 0., 0., 0.]),
    ] {
        let points: Vec<_> = (0..2).map(|_| point(n, &masses, &mut rng)).collect();
        let family = groups.len();
        groups.push(lanes(family, &points, 3, 3, 2, &mut rng));
    }
    groups
}

/// The unchecked mode (m-lit lit-unchecked, DECISIONS 2026-11-08 22:34 (1),
/// `TirOptions::unchecked`): the acceptance pass is never computed (no
/// check row, no lane refused by its check, and the moment rows are the
/// direct pass's, the method choice's retries and the last resort's only:
/// no acceptance-frame row is requested, unlike `acceptance = 1e300`,
/// which computes the whole pass); every lane the checked default serves is
/// bitwise the same; the rows the method choice refuses (here forced by an
/// expert's method-choice gate of 1, which its PV systems never pass: only
/// its expansions still serve) are served by the gate-0 last resort and
/// counted; every lane is served and finite (its distance from NinjaRS's
/// generic engine at 80 digits printed: the rotation-blind row's
/// method-choice lanes, which the checked acceptance refuses, are served as
/// computed, about 1e-6 to 1e-5 off); the default (unchecked off) is bitwise
/// the checked default and counts no last-resort row.
#[test]
fn the_unchecked_mode_skips_the_acceptance_and_serves_by_the_last_resort() {
    let groups = unchecked_groups();
    let rows = kinematic_rows(&groups);
    let total: usize = groups.iter().map(|g| g.keys.len()).sum();
    let truth = consistent_truth(&groups, 80);
    // The checked default, and the same with an acceptance that admits
    // everything (it still computes the acceptance pass).
    let mut checked = tir();
    let (served, _) = reduce(&mut checked, &groups, Poles::All, None);
    let checked_statistics = V2Backend::<f64>::take_statistics(&mut checked);
    eprintln!("checked: {:?}", checked_statistics.counts);
    assert_eq!(count(&checked_statistics, "TIR last-resort rows"), 0);
    assert!(
        !checked_statistics
            .counts
            .iter()
            .any(|(name, _)| *name == "TIR last-resort rows")
    );
    let mut admitting = V2Backends::TiRust(Box::new(TirReductionBackend::new(TirOptions {
        acceptance: 1e300,
        ..TirOptions::default()
    })));
    reduce(&mut admitting, &groups, Poles::All, None);
    let admitting_statistics = V2Backend::<f64>::take_statistics(&mut admitting);
    assert!(count(&admitting_statistics, "TIR method-choice check rows") > 0);
    for (name, options) in [
        ("unchecked", TirOptions::default()),
        (
            "unchecked, method choice refusing",
            TirOptions {
                method_choice_gate: 1.0,
                ..TirOptions::default()
            },
        ),
    ] {
        let mut backend = tir_unchecked(options);
        let (found, _) = reduce(&mut backend, &groups, Poles::All, None);
        let statistics = V2Backend::<f64>::take_statistics(&mut backend);
        eprintln!("{name}: {:?}", statistics.counts);
        // (i) no acceptance pass.
        for counter in [
            "TIR method-choice check rows",
            "TIR method-choice lanes refused by their check",
        ] {
            assert_eq!(count(&statistics, counter), 0, "{name}: {counter}");
        }
        let refused = count(&statistics, "TIR gate refusals");
        let choice = count(&statistics, "TIR method-choice rows");
        let last = count(&statistics, "TIR last-resort rows");
        assert!(refused > 0, "{name}");
        assert_eq!(
            count(&statistics, "tensor moment rows"),
            rows + refused + (refused - choice),
            "{name}: direct, method-choice and last-resort rows only"
        );
        // (iii) what the method choice refuses, the last resort serves.
        assert_eq!(last, refused - choice, "{name}");
        if name.ends_with("refusing") {
            assert!(last > 0, "{name}: the method choice refuses rows");
        }
        // (ii) and (iv): every lane served, the checked lanes bitwise.
        let mut worst = 0f64;
        for (g, group) in found.iter().enumerate() {
            for (lane, value) in group.iter().enumerate() {
                let value = value
                    .as_ref()
                    .unwrap_or_else(|reason| panic!("{name}: family {g} lane {lane}: {reason}"));
                if let (Ok(checked), false) = (&served[g][lane], name.ends_with("refusing")) {
                    assert_eq!(value, checked, "{name}: family {g} lane {lane}");
                }
                let error = relative(value, truth[g][lane].as_ref().unwrap());
                worst = worst.max(error);
                eprintln!("{name}: family {g} lane {lane}: {error:e} from the truth");
            }
        }
        eprintln!("{name}: {total} lanes served, worst relative {worst:e}");
        assert!(worst.is_finite(), "{name}");
    }
    // Off: the checked default, bitwise.
    let mut off = V2Backends::TiRust(Box::new(TirReductionBackend::new(TirOptions {
        unchecked: false,
        ..TirOptions::default()
    })));
    let (again, _) = reduce(&mut off, &groups, Poles::All, None);
    assert_eq!(again, served);
    let statistics = V2Backend::<f64>::take_statistics(&mut off);
    assert_eq!(statistics.counts, checked_statistics.counts);
}

/// The verified last resort of the checked mode (m-lit tir-uuzggg,
/// `TirOptions::verified_last_resort`, on by default): the rows the method
/// choice refuses (here forced by an expert's method-choice gate of 1, which
/// its PV systems never pass) are served by the gate-0 last resort and each
/// of their lanes is verified like a method-choice lane. Off: those lanes
/// fail closed with the method choice's reason. On: the same lanes the off
/// run serves, bitwise; every other lane either served within the target of
/// the 80-digit truth or refused by the acceptance check with its measure;
/// the last resort counts exactly the rows the method choice refused; at
/// least one refused lane is now served.
#[test]
fn the_verified_last_resort_serves_method_choice_refusals_its_check_accepts() {
    let groups = unchecked_groups();
    let truth = consistent_truth(&groups, 80);
    let refusing = TirOptions {
        method_choice_gate: 1.0,
        ..TirOptions::default()
    };
    assert!(refusing.verified_last_resort, "on by default");
    let run = |options: TirOptions| {
        let mut backend = V2Backends::TiRust(Box::new(TirReductionBackend::new(options)));
        let (found, _) = reduce(&mut backend, &groups, Poles::All, None);
        (found, V2Backend::<f64>::take_statistics(&mut backend))
    };
    let (closed, closed_statistics) = run(TirOptions {
        verified_last_resort: false,
        ..refusing
    });
    eprintln!("off: {:?}", closed_statistics.counts);
    assert_eq!(count(&closed_statistics, "TIR last-resort rows"), 0);
    let (found, statistics) = run(refusing);
    eprintln!("on: {:?}", statistics.counts);
    let refused = count(&statistics, "TIR gate refusals");
    let choice = count(&statistics, "TIR method-choice rows");
    assert!(refused > choice, "the method choice refuses rows");
    assert_eq!(count(&statistics, "TIR last-resort rows"), refused - choice);
    let (mut rescued, mut still) = (0, 0);
    for (g, group) in found.iter().enumerate() {
        for (lane, value) in group.iter().enumerate() {
            match (&closed[g][lane], value) {
                (Ok(before), after) => {
                    assert_eq!(Ok(before), after.as_ref(), "family {g} lane {lane}");
                }
                (Err(before), Ok(after)) => {
                    assert!(
                        before.contains("then by the method choice"),
                        "family {g} lane {lane}: {before}"
                    );
                    let error = relative(after, truth[g][lane].as_ref().unwrap());
                    eprintln!(
                        "family {g} lane {lane}: served by the verified last resort, {error:e} from the truth"
                    );
                    assert!(
                        error <= rusticol_loop_reduction::reduction_policy::DEFAULT_TARGET,
                        "family {g} lane {lane}: {error:e}"
                    );
                    rescued += 1;
                }
                (Err(_), Err(after)) => {
                    assert!(
                        after.contains("refused by its check")
                            || after.contains("check row failed")
                            || after.contains("gate-0 last resort"),
                        "family {g} lane {lane}: {after}"
                    );
                    still += 1;
                }
            }
        }
    }
    eprintln!("verified last resort: {rescued} lanes served, {still} still refused");
    assert!(
        rescued > 0,
        "the verified last resort serves a refused lane"
    );
}

/// The unchecked mode in the extended carriers (m-lit lit-unchecked; their
/// method choice is off, DECISIONS 2026-11-08 06:08 R4): the conditioning
/// gates stay as the selection and the gate-0 last resort serves the rows
/// they refuse. With an expert's gate of 1 the gates refuse (nearly) every
/// row: checked, those lanes fail closed; unchecked, the last resort serves
/// every one, bitwise the values the default gate serves (plain PV either
/// way: a gate only refuses, it never changes a value), in double-double
/// and at 256 bits.
#[test]
fn the_unchecked_extended_carriers_serve_the_gates_refusals_by_the_last_resort() {
    let mut rng = Lcg(7);
    let mut groups = Vec::new();
    for (family, (n, masses)) in [
        (2, vec![173. * 173., 80. * 80.]),
        (3, vec![0., 0., 0.]),
        (4, vec![173. * 173., 0., 80. * 80., 0.]),
    ]
    .into_iter()
    .enumerate()
    {
        let points: Vec<_> = (0..2).map(|_| point(n, &masses, &mut rng)).collect();
        groups.push(lanes(family, &points, 3, 3, 2, &mut rng));
    }
    let rows = kinematic_rows(&groups);
    let refusing = TirOptions {
        gate: 1.0,
        ..TirOptions::default()
    };
    let unchecked = TirOptions {
        unchecked: true,
        ..refusing
    };
    let dd: Vec<FamilyLanes<DoubleFloat>> = lifted(&groups, 106)
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
    let mp = lifted(&groups, 256);
    let checks = [
        extended_last_resort(&dd, rows, refusing, unchecked),
        extended_last_resort(&mp, rows, refusing, unchecked),
    ];
    eprintln!("extended carriers, unchecked statistics: {checks:?}");
}

/// [`the_unchecked_extended_carriers_serve_the_gates_refusals_by_the_last_resort`]
/// in one carrier: `refusing` (every row refused by the gates) fails every
/// lane closed, `unchecked` serves every one, bitwise the default's values.
fn extended_last_resort<T>(
    lanes: &[FamilyLanes<T>],
    rows: u64,
    refusing: TirOptions,
    unchecked: TirOptions,
) -> Vec<(&'static str, u64)>
where
    V2Backends: V2Backend<T>,
    Complex<T>: std::fmt::Debug + PartialEq,
{
    let run = |options: TirOptions| {
        let mut backend = V2Backends::TiRust(Box::new(TirReductionBackend::new(options)));
        V2Backend::<T>::begin(&mut backend);
        let (found, _) = V2Backend::<T>::reduce(&mut backend, lanes, Poles::All, None).unwrap();
        (found, V2Backend::<T>::take_statistics(&mut backend))
    };
    let (default, _) = run(TirOptions::default());
    let (closed, _) = run(refusing);
    let mut failed = 0;
    for group in &closed {
        for lane in group {
            if let Err(reason) = lane {
                failed += 1;
                assert!(reason.contains("method choice is off"), "{reason}");
            }
        }
    }
    assert!(failed > 0);
    let (found, statistics) = run(unchecked);
    let refused = count(&statistics, "TIR gate refusals");
    assert!(refused > 0 && refused <= rows, "{refused} of {rows}");
    assert_eq!(count(&statistics, "TIR last-resort rows"), refused);
    assert_eq!(count(&statistics, "TIR method-choice rows"), 0);
    for (g, group) in found.iter().enumerate() {
        for (lane, value) in group.iter().enumerate() {
            let value = value.as_ref().unwrap();
            assert_eq!(
                Some(value),
                default[g][lane].as_ref().ok(),
                "family {g} lane {lane}"
            );
        }
    }
    statistics.counts
}

/// NinjaRS's unchecked mode (m-lit lit-unchecked; `runtime::fast_options`:
/// zero stability thresholds; `V2Reducer::set_unchecked`): the options are
/// NinjaRS's defaults with `stability_threshold` and `relative_tolerance` 0
/// (the routing `infrared_eps` kept). On T1's generic families and the
/// near-forward boxes down to tiny angles, every lane the checked reducer
/// serves keeps its value bitwise (a threshold only refuses), a lane it
/// refuses is served finite or fails with a residual class (`NonFinite`,
/// `MasterFailed`, or `Unstable` for an exact zero or NaN); the same reducer
/// set back to checked is the checked default again, bitwise (its amplitudes
/// are cached per mode). The physical rows where checked NinjaRS fails
/// (g g > z z z, h h h, h z g) are lit-unchecked's accuracy records.
#[test]
fn ninjars_zero_thresholds_keep_the_served_lanes_and_refuse_only_residuals() {
    use ninjars::fast::FastOptions;
    use rusticol_loop_reduction::runtime::fast_options;
    let defaults = FastOptions::default();
    assert_eq!(fast_options(false), defaults);
    assert_eq!(
        fast_options(true),
        FastOptions {
            stability_threshold: 0.0,
            relative_tolerance: 0.0,
            ..defaults
        }
    );
    let mut rng = Lcg(7);
    let mut groups = forward_boxes(&[1e5, 1e6, 1e7, 9e7]);
    for (n, masses) in [
        (3, vec![0., 0., 0.]),
        (4, vec![173. * 173., 0., 80. * 80., 0.]),
        (5, vec![0., 0., 0., 0., 0.]),
    ] {
        let points: Vec<_> = (0..2).map(|_| point(n, &masses, &mut rng)).collect();
        let family = groups.len();
        groups.push(lanes(family, &points, 3, 3, 2, &mut rng));
    }
    let mut backend = ninjars("double", "fast");
    let (checked, _) = reduce(&mut backend, &groups, Poles::All, None);
    let V2Backends::NinjaRs(inner) = &mut backend else {
        unreachable!("a NinjaRS backend")
    };
    inner.reducer.set_unchecked(true);
    assert!(backend.unchecked());
    let (found, _) = reduce(&mut backend, &groups, Poles::All, None);
    let (mut refused, mut served) = (0, 0);
    for (g, group) in checked.iter().enumerate() {
        for (lane, value) in group.iter().enumerate() {
            match (value, &found[g][lane]) {
                (Ok(a), Ok(b)) => assert_eq!(a, b, "family {g} lane {lane}"),
                (Err(reason), Ok(b)) => {
                    refused += 1;
                    served += 1;
                    eprintln!("family {g} lane {lane}: refused checked ({reason}), served");
                    assert!(b.iter().all(|z| z.re.is_finite() && z.im.is_finite()));
                }
                (Err(reason), Err(again)) => {
                    refused += 1;
                    eprintln!("family {g} lane {lane}: {reason}; unchecked: {again}");
                    assert!(
                        ["NonFinite", "MasterFailed", "Unstable"]
                            .iter()
                            .any(|class| again.contains(class)),
                        "{again}"
                    );
                }
                (Ok(_), Err(again)) => panic!("family {g} lane {lane}: {again}"),
            }
        }
    }
    eprintln!("NinjaRS checked refusals {refused}, served unchecked {served}");
    let V2Backends::NinjaRs(inner) = &mut backend else {
        unreachable!("a NinjaRS backend")
    };
    inner.reducer.set_unchecked(false);
    let (again, _) = reduce(&mut backend, &groups, Poles::All, None);
    assert_eq!(again, checked);
}
