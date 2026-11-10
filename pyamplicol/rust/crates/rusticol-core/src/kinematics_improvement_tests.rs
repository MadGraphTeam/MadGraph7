// SPDX-License-Identifier: 0BSD

//! Tests of the kinematics improvement (m-lit lit-psimprove (P7)(a)).

use super::*;

/// A binary64 2 -> n point in a generic frame (rotated and boosted), from
/// back-to-back massless pairs plus massive legs (`masses`), built in
/// binary64 (so it misses its shells and conservation by binary64
/// rounding).
fn binary64_point(pairs: usize, masses: &[f64], frame: u32) -> (Vec<[f64; 4]>, Vec<f64>) {
    let mut finals: Vec<[f64; 4]> = Vec::new();
    for k in 0..pairs {
        let (c, phi) = (0.3 + 0.17 * k as f64, 0.9 + 1.3 * k as f64);
        let e = 210.0 + 37.0 * k as f64;
        let s = (1.0 - c * c).sqrt();
        let p = [e, e * s * phi.cos(), e * s * phi.sin(), e * c];
        finals.push(p);
        finals.push([e, -p[1], -p[2], -p[3]]);
    }
    let mut total = [0.0; 4];
    for (k, &m) in masses.iter().enumerate() {
        let (c, phi) = (-0.2 + 0.31 * k as f64, 0.4 + 2.1 * k as f64);
        let q = 60.0 + 23.0 * k as f64;
        let s = (1.0 - c * c).sqrt();
        let p = [
            (m * m + q * q).sqrt(),
            q * s * phi.cos(),
            q * s * phi.sin(),
            q * c,
        ];
        for mu in 1..4 {
            total[mu] += p[mu];
        }
        finals.push(p);
    }
    // Balance the massive legs' 3-momentum with one more massless pair
    // direction absorbed by the last massive leg's opposite: a massive
    // recoil of mass 50.
    if !masses.is_empty() {
        let q2 = total[1] * total[1] + total[2] * total[2] + total[3] * total[3];
        finals.push([(2500.0 + q2).sqrt(), -total[1], -total[2], -total[3]]);
    }
    let energy: f64 = finals.iter().map(|p| p[0]).sum();
    let mut legs = vec![
        [energy / 2.0, 0.0, 0.0, energy / 2.0],
        [energy / 2.0, 0.0, 0.0, -energy / 2.0],
    ];
    legs.extend(finals);
    let mut m2 = vec![0.0, 0.0];
    m2.extend(std::iter::repeat_n(0.0, 2 * pairs));
    m2.extend(masses.iter().map(|m| m * m));
    if !masses.is_empty() {
        m2.push(2500.0);
    }
    if frame > 0 {
        legs = legs
            .iter()
            .map(|p| boost(&rotate(p, 0.7, 1.1), [0.2, -0.15, 0.25]))
            .collect();
    }
    (legs, m2)
}

fn rotate(p: &[f64; 4], a: f64, b: f64) -> [f64; 4] {
    let (ca, sa, cb, sb) = (a.cos(), a.sin(), b.cos(), b.sin());
    let x = ca * p[1] - sa * p[2];
    let y = sa * p[1] + ca * p[2];
    [p[0], cb * x + sb * p[3], y, -sb * x + cb * p[3]]
}

fn boost(p: &[f64; 4], beta: [f64; 3]) -> [f64; 4] {
    let b2: f64 = beta.iter().map(|b| b * b).sum();
    let g = 1.0 / (1.0 - b2).sqrt();
    let bp: f64 = (0..3).map(|i| beta[i] * p[i + 1]).sum();
    let mut out = [g * (p[0] + bp), 0.0, 0.0, 0.0];
    for i in 0..3 {
        out[i + 1] = p[i + 1] + ((g - 1.0) * bp / b2 + g * p[0]) * beta[i];
    }
    out
}

#[test]
fn the_default_and_the_carrier_key() {
    assert_eq!(DEFAULT_VARIANT, Variant::ZShift);
    assert!(!DEFAULT_BINARY64);
    assert_eq!(setting("z-shift"), Ok(Some(Variant::ZShift)));
    assert_eq!(setting("rescaling"), Ok(Some(Variant::Rescaling)));
    assert_eq!(setting("projection"), Ok(Some(Variant::Projection)));
    assert_eq!(setting("off"), Ok(None));
    assert_eq!(setting("ORIG"), Err(SETTING_REFUSAL.to_owned()));
    assert_eq!(binary64_switch(1), Ok(true));
    assert_eq!(binary64_switch(2), Err(BINARY64_REFUSAL.to_owned()));
    // DECISIONS 2026-11-10 02:29 (G3) (a): the carrier kind.
    let z = Some(Variant::ZShift);
    assert_eq!(applies(z, false, true), None);
    assert_eq!(applies(z, true, true), z);
    assert_eq!(applies(z, false, false), z);
    assert_eq!(applies(None, true, false), None);
    let policy = policy_json();
    assert_eq!(policy["default"], "z-shift");
    assert_eq!(policy["binary64_default"], false);
}

#[test]
fn binary64_projection_is_exact_in_every_frame() {
    for (pairs, masses) in [
        (1, vec![]),
        (2, vec![]),
        (1, vec![173.0, 125.0]),
        (3, vec![]),
        (2, vec![91.0, 80.0]),
    ] {
        for frame in [0, 1] {
            let (legs, m2) = binary64_point(pairs, &masses, frame);
            let out = improve(
                &legs,
                2,
                &m2,
                Variant::Projection,
                true,
                &Tolerances::default(),
            );
            let record = &out.record;
            assert_eq!(record.status, Status::Improved, "{record:?}");
            assert_eq!(record.produced, Some(Variant::Projection));
            assert!(record.exactness.as_ref().unwrap().exact, "{record:?}");
            assert!(record.warning().is_none());
            // The move is of the input's own defect (binary64 rounding).
            assert!(record.moved < 1e-13, "{record:?}");
        }
    }
}

#[test]
fn binary64_madloop_variants_and_the_default_chain() {
    let tolerances = Tolerances::default();
    // Beams along z: ORIG produces the point.
    let (legs, m2) = binary64_point(2, &[173.0], 0);
    let out = improve(&legs, 2, &m2, Variant::ZShift, true, &tolerances);
    assert_eq!(
        out.record.produced,
        Some(Variant::ZShift),
        "{:?}",
        out.record
    );
    assert_eq!(
        out.record.attempts,
        vec![Attempt {
            step: "z-shift",
            code: 0
        }]
    );
    // A generic frame: ORIG refuses (200), MadLoop's fallback PSMC produces
    // the point.
    let (legs, m2) = binary64_point(2, &[173.0], 1);
    let out = improve(&legs, 2, &m2, Variant::ZShift, true, &tolerances);
    assert_eq!(
        out.record.attempts,
        vec![
            Attempt {
                step: "z-shift",
                code: 200
            },
            Attempt {
                step: "rescaling",
                code: 0
            }
        ]
    );
    assert_eq!(out.record.produced, Some(Variant::Rescaling));
    // Forced ORIG there: refused, evaluated as given.
    let out = improve(&legs, 2, &m2, Variant::ZShift, false, &tolerances);
    assert_eq!(out.record.status, Status::NotImproved);
    assert_eq!(
        out.record.attempts,
        vec![Attempt {
            step: "z-shift",
            code: 200
        }]
    );
    assert_eq!(out.legs, legs);
    assert_eq!(out.record.warning_kind(), Some(WARNING_NOT_IMPROVED));
}

#[test]
fn the_newton_cap_follows_the_carrier_digits() {
    let ctx = Ctx::new(&1.0f64, 4, &Tolerances::default());
    assert_eq!(ctx.newton_cap, 7);
    assert_eq!(ctx.thrs_test, (f64::EPSILON / 2.0).sqrt());
}

#[cfg(feature = "symbolica-runtime")]
mod exact_points {
    use super::super::*;
    use symbolica::domains::float::{DoubleFloat, Float, Real, SingleFloat};
    use symbolica::domains::rational::Rational;

    fn q(n: i64, d: i64) -> Rational {
        Rational::new(n, d)
    }

    fn r(n: i64) -> Rational {
        Rational::from(n)
    }

    type Leg = [Rational; 4];

    fn mass2(leg: &Leg) -> Rational {
        &(&(&(&leg[0] * &leg[0]) - &(&leg[1] * &leg[1])) - &(&leg[2] * &leg[2]))
            - &(&leg[3] * &leg[3])
    }

    /// Massless Pythagorean directions (|p| integer).
    const PYTHAGOREAN: [[i64; 4]; 6] = [
        [3, 1, 2, 2],
        [7, 2, 3, 6],
        [9, 1, 4, 8],
        [9, 4, 4, 7],
        [11, 2, 6, 9],
        [11, 6, 6, 7],
    ];

    /// An exact physical 2 -> n point with beams along z in the CM-like
    /// frame: `pairs` back-to-back massless pairs, `massive` massive legs
    /// (their squared masses from their exact momenta), `along_z` massless
    /// legs along z. Returns the legs and their squared masses.
    fn scattering(pairs: usize, massive: usize, along_z: usize) -> (Vec<Leg>, Vec<Rational>) {
        let mut finals: Vec<Leg> = Vec::new();
        for k in 0..pairs {
            let [e, x, y, z] = PYTHAGOREAN[k % PYTHAGOREAN.len()];
            // (Non-dyadic: the binary64 rounding is inexact.)
            let s = q(20 + 7 * k as i64, 3);
            let leg = [&r(e) * &s, &r(x) * &s, &r(y) * &s, &r(z) * &s];
            finals.push(leg.clone());
            finals.push([
                leg[0].clone(),
                -leg[1].clone(),
                -leg[2].clone(),
                -leg[3].clone(),
            ]);
        }
        let mut px = r(0);
        let mut py = r(0);
        let mut pz = r(0);
        for k in 0..massive {
            let leg = if k + 1 == massive {
                // The last massive leg balances the massive legs' transverse
                // momentum.
                [r(0), -px.clone(), -py.clone(), q(-13 - 5 * k as i64, 3)]
            } else {
                [
                    r(0),
                    q(17 + 3 * k as i64, 2),
                    q(-11 + 4 * k as i64, 3),
                    q(29 - 9 * k as i64, 4),
                ]
            };
            px = &px + &leg[1];
            py = &py + &leg[2];
            pz = &pz + &leg[3];
            let p2 = &(&(&leg[1] * &leg[1]) + &(&leg[2] * &leg[2])) + &(&leg[3] * &leg[3]);
            // E = |p| + 150 + 20 k: massive.
            let e = &p2 * &q(1, 100) + &r(150 + 20 * k as i64);
            finals.push([e, leg[1].clone(), leg[2].clone(), leg[3].clone()]);
        }
        for k in 0..along_z {
            let e = r(70 + 30 * k as i64);
            let z = if k % 2 == 0 { e.clone() } else { -e.clone() };
            finals.push([e, r(0), r(0), z]);
        }
        let mut energy = r(0);
        let mut z = r(0);
        for leg in &finals {
            energy = &energy + &leg[0];
            z = &z + &leg[3];
        }
        let half = q(1, 2);
        let e1 = &(&energy + &z) * &half;
        let e2 = &(&energy - &z) * &half;
        let mut legs = vec![[e1.clone(), r(0), r(0), e1], [e2.clone(), r(0), r(0), -e2]];
        legs.extend(finals);
        let m2 = legs.iter().map(mass2).collect();
        (legs, m2)
    }

    /// An exact 1 -> n decay at rest (n = 3: two massive legs and a
    /// massless one).
    fn decay() -> (Vec<Leg>, Vec<Rational>) {
        let a = [q(301, 2), r(30), r(-40), r(12)];
        let b = [q(210, 1), r(-12), r(25), r(-21)];
        let c = [r(0), r(-18), r(15), r(9)];
        let c0 = {
            // A massive third leg (E chosen above |p|).
            &(&(&(&c[1] * &c[1]) + &(&c[2] * &c[2])) + &(&c[3] * &c[3])) * &q(1, 50) + &r(20)
        };
        let c = [c0, c[1].clone(), c[2].clone(), c[3].clone()];
        let m = &(&a[0] + &b[0]) + &c[0];
        let legs = vec![[m, r(0), r(0), r(0)], a, b, c];
        let m2 = legs.iter().map(mass2).collect();
        (legs, m2)
    }

    /// The rational boost along z with gamma = 5/4.
    fn zboost(legs: &[Leg]) -> Vec<Leg> {
        let (g, bg) = (q(5, 4), q(3, 4));
        legs.iter()
            .map(|p| {
                [
                    &(&g * &p[0]) + &(&bg * &p[3]),
                    p[1].clone(),
                    p[2].clone(),
                    &(&bg * &p[0]) + &(&g * &p[3]),
                ]
            })
            .collect()
    }

    /// A rational rotation (quaternion (1, 2, 3, 4) / 30) and a rational
    /// boost along (2, 1, 2) / 3 with gamma = 5/4: a generic frame.
    fn generic(legs: &[Leg]) -> Vec<Leg> {
        let rot = [[-20, 4, 22], [20, -10, 20], [10, 28, 4]];
        let n = [q(2, 3), q(1, 3), q(2, 3)];
        let (g, bg) = (q(5, 4), q(3, 4));
        legs.iter()
            .map(|p| {
                let s: Vec<Rational> = (0..3)
                    .map(|i| (0..3).fold(r(0), |acc, j| &acc + &(&q(rot[i][j], 30) * &p[j + 1])))
                    .collect();
                let np = (0..3).fold(r(0), |acc, i| &acc + &(&n[i] * &s[i]));
                let e = &(&g * &p[0]) + &(&bg * &np);
                let coefficient = &(&(&g - &r(1)) * &np) + &(&bg * &p[0]);
                [
                    e,
                    &s[0] + &(&coefficient * &n[0]),
                    &s[1] + &(&coefficient * &n[1]),
                    &s[2] + &(&coefficient * &n[2]),
                ]
            })
            .collect()
    }

    fn binary64(legs: &[Leg]) -> Vec<Leg> {
        legs.iter()
            .map(|leg| {
                leg.each_ref()
                    .map(|x| Rational::try_from(x.to_f64()).unwrap())
            })
            .collect()
    }

    /// The exact defects of an exact point: max_k |p_k² - m_k²| / (REF_k S)
    /// and max_μ |conservation_μ| / S, as f64.
    fn exact_defects(legs: &[Leg], n_initial: usize, m2: &[Rational]) -> (f64, f64) {
        let scale = legs.iter().fold(r(0), |acc, leg| &acc + &leg[0].abs());
        let mut shell = 0f64;
        for (leg, m2) in legs.iter().zip(m2) {
            let reference = leg.iter().fold(r(0), |acc, x| &acc + &x.abs());
            let d = &(&mass2(leg) - m2).abs() / &(&reference * &scale);
            shell = shell.max(d.to_f64());
        }
        let mut conservation = 0f64;
        for mu in 0..4 {
            let mut sum = r(0);
            for (k, leg) in legs.iter().enumerate() {
                sum = if k < n_initial {
                    &sum + &leg[mu]
                } else {
                    &sum - &leg[mu]
                };
            }
            conservation = conservation.max((&sum.abs() / &scale).to_f64());
        }
        (shell, conservation)
    }

    const CARRIERS: [Arithmetic; 4] = [
        Arithmetic::Binary64,
        Arithmetic::DoubleDouble,
        Arithmetic::Arbitrary(170),
        Arithmetic::Arbitrary(340),
    ];

    fn unit_roundoff(arithmetic: Arithmetic) -> f64 {
        match arithmetic {
            Arithmetic::Binary64 => 2f64.powi(-53),
            Arithmetic::DoubleDouble => 2f64.powi(-104),
            Arithmetic::Arbitrary(bits) => 2f64.powi(-(bits as i32)),
        }
    }

    /// The exact defects within the record's bounds.
    fn assert_exact(
        out: &[Leg],
        n_initial: usize,
        m2: &[Rational],
        arithmetic: Arithmetic,
        record: &Record,
    ) {
        let e = record.exactness.as_ref().expect("an improved point");
        assert!(e.exact, "{arithmetic:?}: {record:?}");
        let (shell, conservation) = exact_defects(out, n_initial, m2);
        let n = out.len() as f64;
        let u = unit_roundoff(arithmetic);
        assert!(
            shell <= (2.0 * n + 5.0) * u,
            "{arithmetic:?}: shell {shell:e} {record:?}"
        );
        assert!(
            conservation <= 2.0 * n * u,
            "{arithmetic:?}: conservation {conservation:e} {record:?}"
        );
    }

    fn sets() -> Vec<(&'static str, Vec<Leg>, Vec<Rational>, usize, bool)> {
        let mut out = Vec::new();
        for (name, pairs, massive, along_z) in [
            ("2->2 massless", 1, 0, 0),
            ("2->2 massive", 0, 2, 0),
            ("2->3 massless", 1, 0, 1),
            ("2->3 tth-like", 0, 3, 0),
            ("2->4 massless", 2, 0, 0),
            ("2->5 mixed", 1, 2, 1),
            ("2->6 massless", 3, 0, 0),
        ] {
            let (legs, m2) = scattering(pairs, massive, along_z);
            out.push((name, legs.clone(), m2.clone(), 2, true));
            out.push((name, zboost(&legs), m2.clone(), 2, true));
            out.push((name, generic(&legs), m2, 2, false));
        }
        let (legs, m2) = decay();
        out.push(("1->3 at rest", legs.clone(), m2.clone(), 1, false));
        out.push(("1->3 in flight", generic(&legs), m2, 1, false));
        out
    }

    #[test]
    fn the_test_points_are_exactly_physical() {
        for (name, legs, m2, n_initial, _) in sets() {
            let masses: Vec<Option<Rational>> = m2.iter().cloned().map(Some).collect();
            assert!(exactly_physical(&legs, n_initial, &masses), "{name}");
            // Their binary64 roundings are not.
            assert!(
                !exactly_physical(&binary64(&legs), n_initial, &masses),
                "{name}"
            );
        }
    }

    #[test]
    fn an_exactly_physical_point_is_passed_unchanged() {
        let (legs, m2) = scattering(2, 0, 0);
        let masses: Vec<Option<Rational>> = m2.into_iter().map(Some).collect();
        for arithmetic in CARRIERS {
            let (out, record) = improve_exact(
                &legs,
                2,
                &masses,
                arithmetic,
                Variant::ZShift,
                true,
                &Tolerances::default(),
            );
            assert_eq!(out, legs);
            assert_eq!(record.status, Status::Exact);
            assert!(record.warning().is_none());
        }
    }

    #[test]
    fn the_projection_is_exact_on_every_set_and_carrier() {
        for (name, legs, m2, n_initial, _) in sets() {
            let input = binary64(&legs);
            let masses: Vec<Option<Rational>> = m2.iter().cloned().map(Some).collect();
            for arithmetic in CARRIERS {
                let (out, record) = improve_exact(
                    &input,
                    n_initial,
                    &masses,
                    arithmetic,
                    Variant::Projection,
                    true,
                    &Tolerances::default(),
                );
                assert_eq!(
                    record.produced,
                    Some(Variant::Projection),
                    "{name} {arithmetic:?}"
                );
                assert_exact(&out, n_initial, &m2, arithmetic, &record);
                // Quadratic convergence from the binary64 input: 1 to 3 steps
                // above binary64; in binary64 the rounded input may already
                // lie within the carrier's own bound (0 steps).
                let steps = record.iterations.unwrap();
                let fewest = u32::from(arithmetic != Arithmetic::Binary64);
                assert!(
                    (fewest..=4).contains(&steps),
                    "{name} {arithmetic:?}: {steps} steps"
                );
                // The move is of the input's own defect.
                assert!(
                    record.moved < 1e-14,
                    "{name} {arithmetic:?}: moved {}",
                    record.moved
                );
            }
        }
    }

    #[test]
    fn madloop_variants_are_exact_where_madloop_is() {
        for (name, legs, m2, n_initial, beams_along_z) in sets() {
            let input = binary64(&legs);
            let masses: Vec<Option<Rational>> = m2.iter().cloned().map(Some).collect();
            let massless_beams = n_initial == 2 && m2[0].is_zero() && m2[1].is_zero();
            for arithmetic in CARRIERS {
                let (out, record) = improve_exact(
                    &input,
                    n_initial,
                    &masses,
                    arithmetic,
                    Variant::ZShift,
                    false,
                    &Tolerances::default(),
                );
                if beams_along_z && massless_beams {
                    assert_eq!(
                        record.produced,
                        Some(Variant::ZShift),
                        "{name} {arithmetic:?}"
                    );
                    assert_exact(&out, n_initial, &m2, arithmetic, &record);
                } else {
                    assert_eq!(record.status, Status::NotImproved, "{name} {arithmetic:?}");
                    let code = record.attempts[0].code;
                    let expected = if n_initial != 2 { 300 } else { 200 };
                    assert_eq!(code, expected, "{name} {arithmetic:?}");
                    assert_eq!(out, input);
                }
                // PSMC: exact shells everywhere; its conservation exact only
                // where x = 1 (the decay at rest), else its residual
                // (x² - x) p₁ is flagged by the exactness record.
                let (out, record) = improve_exact(
                    &input,
                    n_initial,
                    &masses,
                    arithmetic,
                    Variant::Rescaling,
                    false,
                    &Tolerances::default(),
                );
                assert_eq!(
                    record.produced,
                    Some(Variant::Rescaling),
                    "{name} {arithmetic:?}: {record:?}"
                );
                let (shell, conservation) = exact_defects(&out, n_initial, &m2);
                let n = out.len() as f64;
                let u = unit_roundoff(arithmetic);
                assert!(shell <= (2.0 * n + 5.0) * u, "{name} {arithmetic:?}");
                let e = record.exactness.as_ref().unwrap();
                if name == "1->3 at rest" {
                    assert!(
                        e.exact && conservation <= 2.0 * n * u,
                        "{name} {arithmetic:?}: {record:?}"
                    );
                } else if arithmetic != Arithmetic::Binary64 {
                    // (x² - x) p₁ ~ the input's own defect, far above the
                    // extended carriers' rounding.
                    assert!(
                        !e.exact && conservation > 2.0 * n * u,
                        "{name} {arithmetic:?}: {record:?}"
                    );
                    assert_eq!(record.warning_kind(), Some(WARNING_INEXACT));
                    assert!(record.warning().unwrap().contains("rescaling"));
                }
                assert_eq!(e.exact, record.warning_kind().is_none());
            }
        }
    }

    #[test]
    fn the_default_chain() {
        for (name, legs, m2, n_initial, beams_along_z) in sets() {
            let input = binary64(&legs);
            let masses: Vec<Option<Rational>> = m2.iter().cloned().map(Some).collect();
            let massless_beams = n_initial == 2 && m2[0].is_zero() && m2[1].is_zero();
            let (out, record) = improve_exact(
                &input,
                n_initial,
                &masses,
                Arithmetic::DoubleDouble,
                DEFAULT_VARIANT,
                true,
                &Tolerances::default(),
            );
            if beams_along_z && massless_beams {
                assert_eq!(
                    record.attempts,
                    vec![Attempt {
                        step: "z-shift",
                        code: 0
                    }],
                    "{name}"
                );
                assert_exact(&out, n_initial, &m2, Arithmetic::DoubleDouble, &record);
            } else {
                assert_eq!(record.attempts.len(), 2, "{name}: {record:?}");
                assert_ne!(record.attempts[0].code, 0);
                assert_eq!(
                    record.attempts[1],
                    Attempt {
                        step: "rescaling",
                        code: 0
                    },
                    "{name}"
                );
                assert_eq!(record.produced, Some(Variant::Rescaling));
            }
            // 'rescaling': PSMC first.
            let (_, record) = improve_exact(
                &input,
                n_initial,
                &masses,
                Arithmetic::DoubleDouble,
                Variant::Rescaling,
                true,
                &Tolerances::default(),
            );
            assert_eq!(
                record.attempts,
                vec![Attempt {
                    step: "rescaling",
                    code: 0
                }],
                "{name}"
            );
        }
    }

    fn masses_of(m2: &[Rational]) -> Vec<Option<Rational>> {
        m2.iter().cloned().map(Some).collect()
    }

    #[test]
    fn is_physical_failure_and_the_projection_start() {
        let (legs, m2) = scattering(1, 1, 1);
        let mut input = binary64(&legs);
        // A final shell off by 1e-4 (sqrt(|p² - m²|) / REF above 1e-2).
        input[2][0] = &input[2][0] * &q(1001, 1000);
        let masses = masses_of(&m2);
        for arithmetic in CARRIERS {
            for variant in [Variant::ZShift, Variant::Rescaling] {
                let (out, record) = improve_exact(
                    &input,
                    2,
                    &masses,
                    arithmetic,
                    variant,
                    true,
                    &Tolerances::default(),
                );
                assert_eq!(record.status, Status::NotImproved);
                assert_eq!(
                    record.attempts,
                    vec![Attempt {
                        step: "is_physical",
                        code: CODE_NOT_PHYSICAL
                    }]
                );
                assert_eq!(out, input);
                assert!(record.warning().unwrap().contains("is_physical 1"));
            }
            // The projection: no IS_PHYSICAL test, bounded by IS_CLOSE.
            let (out, record) = improve_exact(
                &input,
                2,
                &masses,
                arithmetic,
                Variant::Projection,
                true,
                &Tolerances::default(),
            );
            assert_eq!(record.produced, Some(Variant::Projection), "{arithmetic:?}");
            assert_exact(&out, 2, &m2, arithmetic, &record);
            // A move above IS_CLOSE: refused (999), evaluated as given.
            let tight = Tolerances {
                close: 1e-6,
                ..Tolerances::default()
            };
            let (out, record) = improve_exact(
                &input,
                2,
                &masses,
                arithmetic,
                Variant::Projection,
                true,
                &tight,
            );
            assert_eq!(record.status, Status::NotImproved);
            assert_eq!(
                record.attempts,
                vec![Attempt {
                    step: "projection",
                    code: CODE_NOT_CLOSE
                }]
            );
            assert_eq!(out, input);
            // Overridden IS_PHYSICAL tolerances admit it to MadLoop's chain.
            let loose = Tolerances {
                onshell: 0.5,
                fourmom: 1e-2,
                ..Tolerances::default()
            };
            let (_, record) = improve_exact(
                &input,
                2,
                &masses,
                arithmetic,
                Variant::ZShift,
                true,
                &loose,
            );
            assert_eq!(record.status, Status::Improved, "{arithmetic:?}");
        }
    }

    #[test]
    fn orig_and_psmc_refusal_codes() {
        let tolerances = Tolerances::default();
        let dd = Arithmetic::DoubleDouble;
        // 300: massive initial states (1 -> n: 300 overwrites 100 / 200).
        let (legs, m2) = decay();
        let (_, record) = improve_exact(
            &binary64(&legs),
            1,
            &masses_of(&m2),
            dd,
            Variant::ZShift,
            false,
            &tolerances,
        );
        assert_eq!(record.attempts[0].code, 300);
        // 100 alone: a 1 -> n point with a massless "initial" state is not
        // physical; use NINITIAL != 2 on a 2 -> 2 point read as 1 -> 3
        // with loose IS_PHYSICAL tolerances: 100, overwritten by 200 (leg
        // 2's transverse momentum) — the order of MadLoop's tests.
        let (legs, m2) = scattering(1, 0, 0);
        let loose = Tolerances {
            fourmom: 10.0,
            onshell: 10.0,
            close: 10.0,
        };
        let (_, record) = improve_exact(
            &binary64(&legs),
            1,
            &masses_of(&m2),
            dd,
            Variant::ZShift,
            false,
            &loose,
        );
        assert_eq!(record.attempts[0].code, 100);
        // 400 + I: a negative energy (leg 3).
        let mut negative = binary64(&legs);
        negative[2] = negative[2].each_ref().map(|x| -x.clone());
        let (_, record) = improve_exact(
            &negative,
            2,
            &masses_of(&m2),
            dd,
            Variant::ZShift,
            false,
            &loose,
        );
        assert_eq!(record.attempts[0].code, 403);
        // 500: neither initial state with pz > 0.
        let beams = vec![
            [r(10), r(0), r(0), r(-10)],
            [r(5), r(0), r(0), r(-5)],
            [q(151, 10), r(0), r(0), r(-15)],
        ];
        let m2b = vec![r(0), r(0), mass2(&beams[2])];
        let (_, record) = improve_exact(
            &binary64(&beams),
            2,
            &masses_of(&m2b),
            dd,
            Variant::ZShift,
            false,
            &loose,
        );
        assert_eq!(record.attempts[0].code, 500);
        // 999: IS_CLOSE (an overridden THRS_CLOSE below the move).
        let (legs, m2) = scattering(1, 2, 0);
        let input = binary64(&legs);
        let tight = Tolerances {
            close: 1e-30,
            ..Tolerances::default()
        };
        let (_, record) = improve_exact(
            &input,
            2,
            &masses_of(&m2),
            dd,
            Variant::ZShift,
            false,
            &tight,
        );
        assert_eq!(record.attempts[0].code, CODE_NOT_CLOSE);
        let (_, record) = improve_exact(
            &input,
            2,
            &masses_of(&m2),
            dd,
            Variant::Rescaling,
            false,
            &tight,
        );
        assert_eq!(record.attempts[0].code, CODE_NOT_CLOSE);
        // PSMC's Newton finds no x (the final masses exceed the energy):
        // 9 + 90 + 900 + 1000.
        let mut heavy = m2.clone();
        heavy[4] = &heavy[4] * &r(100);
        heavy[5] = &heavy[5] * &r(100);
        let (_, record) = improve_exact(
            &input,
            2,
            &masses_of(&heavy),
            dd,
            Variant::Rescaling,
            false,
            &loose,
        );
        assert!(record.attempts[0].code > 1000, "{record:?}");
        // Both refusing: evaluated as given, typed, both codes.
        let generic_point = generic(&legs);
        let (out, record) = improve_exact(
            &binary64(&generic_point),
            2,
            &masses_of(&heavy),
            dd,
            Variant::ZShift,
            true,
            &loose,
        );
        assert_eq!(record.status, Status::NotImproved);
        assert_eq!(record.attempts.len(), 2);
        assert_eq!(
            record.attempts[0],
            Attempt {
                step: "z-shift",
                code: 200
            }
        );
        assert!(record.attempts[1].code > 1000, "{record:?}");
        assert_eq!(out, binary64(&generic_point));
    }

    #[test]
    fn is_physical_and_is_close_do_not_depend_on_the_carrier() {
        let (legs, m2) = scattering(1, 2, 0);
        let masses = masses_of(&m2);
        for factor in [q(10001, 10000), q(1000001, 1000000), q(1001, 1000)] {
            let mut input = binary64(&legs);
            input[3][1] = &input[3][1] * &factor;
            let outcomes: Vec<Status> = CARRIERS
                .iter()
                .map(|&a| {
                    improve_exact(
                        &input,
                        2,
                        &masses,
                        a,
                        Variant::ZShift,
                        false,
                        &Tolerances::default(),
                    )
                    .1
                    .status
                })
                .collect();
            assert!(outcomes.iter().all(|s| *s == outcomes[0]), "{outcomes:?}");
        }
    }

    #[test]
    fn idempotence_and_the_bound_invariants() {
        for (name, legs, m2, n_initial, _) in sets() {
            let masses = masses_of(&m2);
            for arithmetic in [Arithmetic::DoubleDouble, Arithmetic::Arbitrary(170)] {
                let (once, _) = improve_exact(
                    &binary64(&legs),
                    n_initial,
                    &masses,
                    arithmetic,
                    Variant::Projection,
                    true,
                    &Tolerances::default(),
                );
                let (twice, record) = improve_exact(
                    &once,
                    n_initial,
                    &masses,
                    arithmetic,
                    Variant::Projection,
                    true,
                    &Tolerances::default(),
                );
                // Again: within the carrier's rounding (already exact there).
                assert!(
                    record.moved <= 4.0 * unit_roundoff(arithmetic),
                    "{name} {arithmetic:?}: {}",
                    record.moved
                );
                assert_eq!(record.iterations, Some(0));
                // Every invariant a single leg fixes (a single-leg subset or
                // its complement) equals that leg's bound mass to the
                // carrier's rounding (`dress::ExactInvariants`).
                let n = once.len();
                let full = (1u64 << n) - 1;
                let scale = once.iter().fold(r(0), |acc, leg| &acc + &leg[0].abs());
                for k in 0..n {
                    let complement = full ^ (1 << k);
                    let mut p = [r(0), r(0), r(0), r(0)];
                    for (j, leg) in once.iter().enumerate() {
                        if complement & (1 << j) != 0 {
                            for mu in 0..4 {
                                // All-incoming momenta.
                                p[mu] = if j < n_initial {
                                    &p[mu] + &leg[mu]
                                } else {
                                    &p[mu] - &leg[mu]
                                };
                            }
                        }
                    }
                    let defect = (&(&mass2(&p) - &m2[k]).abs() / &(&scale * &scale)).to_f64();
                    assert!(
                        defect <= 8.0 * n as f64 * unit_roundoff(arithmetic),
                        "{name} {arithmetic:?} leg {k}: {defect:e}"
                    );
                    let _ = twice.len();
                }
            }
        }
    }

    #[test]
    fn a_leg_without_a_bound_mass_keeps_its_own_square() {
        let (legs, m2) = scattering(1, 2, 0);
        let input = binary64(&legs);
        let mut masses = masses_of(&m2);
        masses[3] = None;
        let (out, record) = improve_exact(
            &input,
            2,
            &masses,
            Arithmetic::DoubleDouble,
            Variant::Projection,
            true,
            &Tolerances::default(),
        );
        assert_eq!(record.produced, Some(Variant::Projection));
        // Its target: its own input p² (cast once).
        let mut targets = m2.clone();
        targets[3] = mass2(&input[3]);
        assert_exact(&out, 2, &targets, Arithmetic::DoubleDouble, &record);
    }

    #[test]
    fn a_singular_projection_is_refused() {
        // A massless leg of zero momentum: its shell row vanishes.
        let (legs, m2) = scattering(1, 0, 0);
        let mut input = binary64(&legs);
        input.push([r(0), r(0), r(0), r(0)]);
        let mut masses = masses_of(&m2);
        masses.push(Some(r(0)));
        input[0][0] = &input[0][0] + &q(1, 1 << 40);
        let (out, record) = improve_exact(
            &input,
            2,
            &masses,
            Arithmetic::DoubleDouble,
            Variant::Projection,
            true,
            &Tolerances::default(),
        );
        assert_eq!(
            record.attempts,
            vec![Attempt {
                step: "projection",
                code: CODE_PROJECTION_SINGULAR
            }]
        );
        assert_eq!(out, input);
    }

    #[test]
    fn the_carriers_round_trip_their_values() {
        let x = Rational::new(1, 3);
        let dd: DoubleFloat = (&x).into();
        let f = Mpfr(Float::new(200).from_rational(&x));
        assert!((PsReal::to_f64(&dd) - 1.0 / 3.0).abs() < 1e-16);
        assert!((PsReal::to_f64(&f) - 1.0 / 3.0).abs() < 1e-16);
    }

    /// The MPFR carrier keeps its full precision through a cancelling sum
    /// (Symbolica's `Float` lowers it), so MadLoop 2's z-shift of a slow
    /// final state is exact to the carrier's rounding (a slow Higgs whose
    /// `E² - p_T² - m²` cancels about 37 bits).
    #[test]
    fn the_mpfr_carrier_keeps_its_precision_through_cancellation() {
        let a = Mpfr(Float::new(200).from_rational(&q(125_000_001, 1_000_000)));
        let b = Mpfr(Float::new(200).from_rational(&r(125)));
        let d = a.clone() * a - b.clone() * b;
        assert_eq!(PsReal::bits(&d), 200);
        // A slow Higgs (pz = 2.5e-4 GeV, pT = 0.25 GeV) in a z-boosted
        // massless-beam point, rounded to binary64.
        let pt = q(1, 4);
        let pz = q(1, 4000);
        let m2 = r(125 * 125);
        let e3 = (&(&(&m2 + &(&pt * &pt)) + &(&pz * &pz)))
            .to_multi_prec_float(400)
            .sqrt()
            .to_rational();
        let p4 = [r(900), -pt.clone(), r(0), r(850)];
        let e4 = (&(&(&p4[1] * &p4[1]) + &(&p4[3] * &p4[3])) + &m2)
            .to_multi_prec_float(400)
            .sqrt()
            .to_rational();
        let total = &e3 + &e4;
        let pzt = &pz + &p4[3];
        let e1 = &(&total + &pzt) / &r(2);
        let e2 = &(&total - &pzt) / &r(2);
        let legs = binary64(&[
            [e1.clone(), r(0), r(0), e1],
            [e2.clone(), r(0), r(0), -e2],
            [e3, pt, r(0), pz],
            [e4, p4[1].clone(), r(0), p4[3].clone()],
        ]);
        let m2s = vec![r(0), r(0), m2.clone(), m2];
        let masses = masses_of(&m2s);
        for arithmetic in [Arithmetic::Arbitrary(173), Arithmetic::Arbitrary(340)] {
            let (out, record) = improve_exact(
                &legs,
                2,
                &masses,
                arithmetic,
                Variant::ZShift,
                false,
                &Tolerances::default(),
            );
            assert_eq!(record.produced, Some(Variant::ZShift), "{record:?}");
            assert_exact(&out, 2, &m2s, arithmetic, &record);
        }
    }
}
