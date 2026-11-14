// SPDX-License-Identifier: 0BSD

//! Flat phase-space points: RAMBO on diet (S. Plätzer, arXiv:1308.2922),
//! the one generator of every random phase-space point of pyAmpliCol (USER
//! 2026-11-14 ~18:15 and ~18:20 UTC: "really random and flat in
//! phase-space", "arbitrary Symbolica types with the Float trait ... also
//! supporting double-double and Arb Prec"). Python calls it through the
//! loop extension (`flat_phase_space`), never re-implements it.
//!
//! THE CARRIER. The construction is generic over [`FlatReal`]: the
//! kinematics carrier trait [`PsReal`] of [`crate::kinematics_improvement`]
//! (the one the loop executors' kinematics already use: binary64,
//! Symbolica's `DoubleFloat`, and Symbolica's MPFR `Float` at a fixed
//! precision, `Mpfr`) with the exact conversion of a rational into the
//! carrier; a point is drawn natively in each carrier, on its mass shells
//! and momentum conserving to that carrier's rounding. Only `+ - * /` and
//! the carrier's square root (Symbolica's) are evaluated in the carrier: no
//! transcendental function, so a binary64 point is reproducible bit for bit.
//!
//! SYMBOLICA FIRST (USER 2026-11-14 ~18:25 UTC). The random numbers are
//! Symbolica's `numerical_integration::MonteCarloRng` (xoshiro256** seeded
//! through SplitMix64; byte-stable), and the one root of each intermediate
//! mass is Symbolica's exact real-root refinement
//! (`UnivariatePolynomial<Q>::refine_root_interval`) on the bracket (0, 1).
//! Nothing else is solved: the massive mapping is closed-form.
//!
//! CARRIER INDEPENDENCE. The random numbers are 64-bit integers k, each the
//! uniform number r = (2 k + 1) / 2^65 of the open interval (0, 1), an
//! exact rational; the direction's rejection in the unit disk is decided on
//! those integers exactly, and the intermediate-mass roots are refined in
//! exact rational arithmetic to the carrier's relative precision before one
//! conversion. The same seed therefore gives the same point in every
//! carrier up to that carrier's rounding.
//!
//! MASSLESS (n >= 2 particles). The intermediate masses K_1 = sqrt(s) >
//! K_2 > ... > K_n = 0 of the sequential splitting K_(i-1) -> (particle
//! i - 1, system i) take K_i^2 = x_i K_(i-1)^2, x_i the root in [0, 1] of
//! `(n + 1 - i) x^(n - i) - (n - i) x^(n + 1 - i) = r_i` (the paper's
//! eq. 2.9; i = 2 .. n - 1), every splitting isotropic in its parent's rest
//! frame (cos(theta) and phi flat: Marsaglia's direction) and boosted back.
//! The weight is constant: 1 relative to the flat massless phase space.
//!
//! MASSIVE: the paper's mass mapping (its sect. 4). The massless
//! intermediate masses are drawn at K_1 = sqrt(s) - sum m and mapped to
//! M_i = K_i + sum_(j >= i) m_j, so M_(i-1) >= m_(i-1) + M_i always and the
//! whole massive phase space is reached; the directions are the massless
//! ones. The weight, relative to the flat massless phase space at the same
//! sqrt(s), is the Jacobian of the mapping
//! `prod_(i=2)^(n-1) M_i / K_i  *  prod_(i=2)^n |q_i| K_(i-1) / (M_(i-1) |q0_i|)  *  (K_1 / sqrt(s))^(2 n - 4)`,
//! with |q_i| (|q0_i|) the momentum of the massive (massless) splitting i:
//! exact in closed form (no root of the energy balance, which the classic
//! RAMBO rescaling solves iteratively); its mean is the ratio of the massive
//! to the massless phase-space volume.

// The negated comparisons are deliberate (a NaN fails every bound test), as
// in `kinematics_improvement`.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::sync::Arc;

use crate::kinematics_improvement::{Mpfr, PsReal};
use symbolica::domains::float::{DoubleFloat, Float, RealLike, RoundingDirection};
use symbolica::domains::integer::Integer;
use symbolica::domains::rational::{Q, Rational};
use symbolica::numerical_integration::MonteCarloRng;
use symbolica::poly::PolyVariable;
use symbolica::poly::univariate::UnivariatePolynomial;
use symbolica::rand::RngCore;

/// A flat final state: `momenta` `[E, px, py, pz]` in the rest frame of the
/// total momentum `(sqrt(s), 0, 0, 0)`, and its `weight` relative to the
/// flat massless phase space of the same multiplicity at the same sqrt(s)
/// (exactly 1 when every mass is zero).
#[derive(Clone, Debug, PartialEq)]
pub struct FlatFinalState<T> {
    pub momenta: Vec<[T; 4]>,
    pub weight: T,
}

/// A carrier of the generator: a kinematics carrier with the exact entry of
/// the random stream's dyadic numbers and the intermediate-mass root.
///
/// The binary64 carrier performs no Symbolica operation that touches
/// Symbolica's state or needs a licence (the licence-free binary64 runtime's
/// guarantee, docs/user/symbolica-and-licensing.md): its root is
/// [`own_ratio_root`]. Double-double and MPFR take Symbolica's exact
/// real-root refinement ([`symbolica_ratio_root`]), Symbolica being in play
/// in those carriers anyway (USER 2026-11-14 ~18:25 and ~18:30 UTC).
pub trait FlatReal: PsReal {
    /// `a / 2^shift` in this carrier (one rounding at most).
    fn dyadic(&self, a: i128, shift: u32) -> Self;
    /// The root in (0, 1) of (k + 1) x^k - k x^(k + 1) = r, r = `r` (a
    /// dyadic number of the stream, `r_odd / 2^65`).
    fn ratio_root(&self, k: u32, r_odd: i128) -> Self;
}

impl FlatReal for f64 {
    fn dyadic(&self, a: i128, shift: u32) -> Self {
        // One rounding of a, then an exact power-of-two scaling.
        (a as f64) * 2f64.powi(-(shift as i32))
    }
    fn ratio_root(&self, k: u32, r_odd: i128) -> Self {
        let half = 1i128 << 64;
        if r_odd < half {
            own_ratio_root(k, &self.dyadic(r_odd, 65))
        } else {
            // r > 1/2: the root's complement from the exact 1 - r (binary64
            // would round r next to 1, where the root is ill-conditioned).
            1.0 - own_complement_root(k, &self.dyadic((half << 1) - r_odd, 65))
        }
    }
}

impl FlatReal for DoubleFloat {
    fn dyadic(&self, a: i128, shift: u32) -> Self {
        exact_dyadic(a, shift).into()
    }
    fn ratio_root(&self, k: u32, r_odd: i128) -> Self {
        (&symbolica_ratio_root(k, &exact_dyadic(r_odd, 65), PsReal::bits(self))).into()
    }
}

impl FlatReal for Mpfr {
    fn dyadic(&self, a: i128, shift: u32) -> Self {
        self.rational(&exact_dyadic(a, shift))
    }
    fn ratio_root(&self, k: u32, r_odd: i128) -> Self {
        self.rational(&symbolica_ratio_root(
            k,
            &exact_dyadic(r_odd, 65),
            self.0.prec(),
        ))
    }
}

impl Mpfr {
    fn rational(&self, value: &Rational) -> Self {
        Mpfr(Float::from_rational_round(
            value,
            self.0.prec(),
            RoundingDirection::Nearest,
        ))
    }
}

/// `a / 2^shift` exactly.
fn exact_dyadic(a: i128, shift: u32) -> Rational {
    Rational::from((Integer::from(a), Integer::from(2).pow(u64::from(shift))))
}

/// The binary64 value nearest to `value`.
fn to_binary64(value: &Rational) -> f64 {
    const EXACT: i64 = 1 << f64::MANTISSA_DIGITS;
    match (value.numerator_ref(), value.denominator_ref()) {
        (Integer::Single(n), Integer::Single(d)) if n.abs() <= EXACT && *d <= EXACT => {
            *n as f64 / *d as f64
        }
        _ => RealLike::to_f64(&value.to_multi_prec_float(f64::MANTISSA_DIGITS)),
    }
}

/// The random stream of a seed (Symbolica's `MonteCarloRng`, stream 0: a
/// plain xoshiro256** state, no Symbolica state).
pub fn stream(seed: u64) -> MonteCarloRng {
    MonteCarloRng::new(seed, 0)
}

/// The odd integer 2 k + 1 of the uniform number (2 k + 1) / 2^65 of (0, 1).
fn uniform_odd(k: u64) -> i128 {
    2 * i128::from(k) + 1
}

/// A direction uniform on the unit sphere (Marsaglia 1972): (x, y) uniform
/// in the unit disk, n = (2 x w, 2 y w, 1 - 2 rho), rho = x^2 + y^2, w =
/// sqrt(1 - rho); cos(theta) and phi flat. x = 2 r - 1 = (2 k + 1 - 2^64) /
/// 2^64: the disk test is decided on the odd integers 2 k + 1 - 2^64
/// exactly (carrier-independent); x and y enter the carrier once.
fn direction<T: FlatReal>(like: &T, rng: &mut MonteCarloRng) -> [T; 3] {
    let odd = |k: u64| -> i128 { 2 * i128::from(k) + 1 - (1i128 << 64) };
    let (a, b) = loop {
        let (a, b) = (odd(rng.next_u64()), odd(rng.next_u64()));
        let (ua, ub) = (a.unsigned_abs(), b.unsigned_abs());
        // |a|, |b| < 2^64, so each square fits in u128, and rho < 1 <=>
        // a^2 + b^2 < 2^128 <=> the sum does not overflow; rho > 0 (a, b odd).
        if (ua * ua).checked_add(ub * ub).is_some() {
            break (a, b);
        }
    };
    let one = like.int(1);
    let two = like.int(2);
    let x = like.dyadic(a, 64);
    let y = like.dyadic(b, 64);
    let rho = x.clone() * x.clone() + y.clone() * y.clone();
    let w = (one.clone() - rho.clone()).sqrt();
    [
        two.clone() * x * w.clone(),
        two.clone() * y * w,
        one - two * rho,
    ]
}

/// x^k by repeated squaring.
fn power<T: PsReal>(x: &T, mut k: u32) -> T {
    let mut base = x.clone();
    let mut out = x.int(1);
    while k > 0 {
        if k & 1 == 1 {
            out = out * base.clone();
        }
        base = base.clone() * base;
        k >>= 1;
    }
    out
}

/// The root x in (0, 1) of (k + 1) x^k - k x^(k + 1) = r (k >= 1, r in
/// (0, 1)), the paper's eq. 2.9, by Symbolica's exact real-root refinement
/// (`UnivariatePolynomial<Q>::refine_root_interval`): the polynomial
/// increases from 0 to 1 on [0, 1], so (0, 1) brackets exactly one root;
/// the bracket is narrowed to the relative width 2^-(bits + 1) of a carrier
/// of `bits` significand bits (no fixed iteration count). The midpoint, an
/// exact rational.
fn symbolica_ratio_root(k: u32, r: &Rational, bits: u32) -> Rational {
    let degree = k as usize + 1;
    let mut coefficients = vec![Rational::from(0); degree + 1];
    coefficients[0] = -r.clone();
    coefficients[degree - 1] = Rational::from(i64::from(k) + 1);
    coefficients[degree] = Rational::from(-i64::from(k));
    let polynomial = UnivariatePolynomial::from_coefficients(
        &Q,
        coefficients,
        Arc::new(PolyVariable::Temporary(0)),
    );
    let tolerance = exact_dyadic(1, bits + 1);
    let (low, high) =
        polynomial.refine_root_interval((Rational::from(0), Rational::from(1)), &tolerance);
    &(&low + &high) / &Rational::from(2)
}

/// The same root in the carrier of `r` with no Symbolica operation (the
/// binary64 carrier's for r <= 1/2, see [`FlatReal`]):
/// f(x) = (k + 1) x^k - k x^(k + 1) increases from 0 to 1 on [0, 1];
/// [`bracketed_root`] of f(x) = r.
pub fn own_ratio_root<T: PsReal>(k: u32, r: &T) -> T {
    let kk = r.int(i64::from(k));
    let k1 = r.int(i64::from(k) + 1);
    let one = r.int(1);
    bracketed_root(r, |x: &T| {
        let below = power(x, k - 1);
        (
            below.clone() * x.clone() * (k1.clone() - kk.clone() * x.clone()),
            kk.clone() * k1.clone() * below * (one.clone() - x.clone()),
        )
    })
}

/// The complement y = 1 - x of the root, from q = 1 - r (the binary64
/// carrier's for r > 1/2): 1 - f(1 - y) = y^2 sum_(j=0)^(k-1) (j + 1)
/// (1 - y)^j increases from 0 to 1 on [0, 1] (f has a double point at 1);
/// [`bracketed_root`] of it at q, accurate relative to y where x is next
/// to 1.
pub fn own_complement_root<T: PsReal>(k: u32, q: &T) -> T {
    let kk = q.int(i64::from(k));
    let k1 = q.int(i64::from(k) + 1);
    let one = q.int(1);
    bracketed_root(q, |y: &T| {
        let x = one.clone() - y.clone();
        let mut sum = q.int(0);
        let mut power_x = one.clone();
        for j in 0..k {
            sum = sum + q.int(i64::from(j) + 1) * power_x.clone();
            power_x = power_x * x.clone();
        }
        (
            y.clone() * y.clone() * sum,
            kk.clone() * k1.clone() * power(&x, k - 1) * y.clone(),
        )
    })
}

/// The root in [0, 1] of value(t) = target for an increasing function with
/// value(0) = 0 and value(1) = 1, `evaluate(t) = (value, derivative)`:
/// Newton steps kept inside the bracket [low, high] that every evaluation
/// narrows, and a bisection whenever a Newton step leaves the bracket or the
/// previous step did not halve it (so the bracket halves at least every
/// second step and the iteration ends, with no iteration count). It stops at
/// an exact solution, when a step leaves t unchanged, or when the bracket
/// is within 4 unit roundoffs of its upper end (the carrier's precision).
fn bracketed_root<T: PsReal>(target: &T, evaluate: impl Fn(&T) -> (T, T)) -> T {
    let zero = target.int(0);
    let one = target.int(1);
    let two = target.int(2);
    let tolerance = target.unit_roundoff() * target.int(4);
    let (mut low, mut high) = (zero.clone(), one.clone());
    let mut t = (one.clone() + target.clone()) / two.clone();
    let mut width = one;
    let mut bisect = false;
    loop {
        let (value, slope) = evaluate(&t);
        let value = value - target.clone();
        if value == zero {
            return t;
        }
        if value > zero {
            high = t.clone();
        } else {
            low = t.clone();
        }
        let narrowed = high.clone() - low.clone();
        if narrowed <= tolerance.clone() * high.clone() {
            return (low + high) / two;
        }
        let newton = if !bisect && slope > zero {
            Some(t.clone() - value / slope)
        } else {
            None
        };
        let trial = match newton {
            Some(next) if next > low && next < high => next,
            _ => (low.clone() + high.clone()) / two.clone(),
        };
        // A step that did not halve the bracket: bisect next.
        bisect = !(narrowed.clone() * two.clone() <= width);
        width = narrowed;
        if trial == t {
            return t;
        }
        t = trial;
    }
}

/// `momentum` given in the rest frame of `frame` (of mass `mass`), in the
/// frame where `frame` is given.
fn boost<T: PsReal>(momentum: &[T; 4], frame: &[T; 4], mass: &T) -> [T; 4] {
    let energy = (frame[0].clone() * momentum[0].clone()
        + frame[1].clone() * momentum[1].clone()
        + frame[2].clone() * momentum[2].clone()
        + frame[3].clone() * momentum[3].clone())
        / mass.clone();
    let factor = (momentum[0].clone() + energy.clone()) / (frame[0].clone() + mass.clone());
    [
        energy,
        momentum[1].clone() + factor.clone() * frame[1].clone(),
        momentum[2].clone() + factor.clone() * frame[2].clone(),
        momentum[3].clone() + factor * frame[3].clone(),
    ]
}

/// A flat final state of `masses.len()` >= 2 particles at total momentum
/// `(sqrt_s, 0, 0, 0)` (module docs), drawn from `stream` in the carrier of
/// the arguments. Refused: fewer than two particles, a negative mass, or
/// sqrt(s) not above the threshold.
pub fn flat_final_state<T: FlatReal>(
    masses: &[T],
    sqrt_s: &T,
    rng: &mut MonteCarloRng,
) -> Result<FlatFinalState<T>, String> {
    let n = masses.len();
    if n < 2 {
        return Err("a flat final state needs at least two particles".to_owned());
    }
    let zero = sqrt_s.int(0);
    let one = sqrt_s.int(1);
    let two = sqrt_s.int(2);
    if masses.iter().any(|m| !(*m >= zero)) {
        return Err("a final-state mass is negative".to_owned());
    }
    let total = masses.iter().fold(zero.clone(), |sum, m| sum + m.clone());
    if !(*sqrt_s > total) {
        return Err(format!(
            "sqrt(s) = {} is not above the final-state threshold {}",
            sqrt_s.to_f64(),
            total.to_f64()
        ));
    }
    let massive = total > zero;
    // The massless intermediate masses K_1 .. K_n.
    let mut kappa = vec![sqrt_s.clone() - total];
    for i in 2..n {
        let x = sqrt_s.ratio_root((n - i) as u32, uniform_odd(rng.next_u64()));
        let next = kappa[i - 2].clone() * x.sqrt();
        kappa.push(next);
    }
    kappa.push(zero.clone());
    // The massive ones M_i = K_i + sum_(j >= i) m_j.
    let mut tail = vec![zero.clone(); n + 1];
    for j in (0..n).rev() {
        tail[j] = tail[j + 1].clone() + masses[j].clone();
    }
    let big: Vec<T> = (0..n).map(|i| kappa[i].clone() + tail[i].clone()).collect();
    let mut weight = one.clone();
    let mut momenta = Vec::with_capacity(n);
    let mut frame = [sqrt_s.clone(), zero.clone(), zero.clone(), zero.clone()];
    for i in 1..n {
        let (parent, emitted, rest) = (&big[i - 1], &masses[i - 1], &big[i]);
        let lambda = (parent.clone() - emitted.clone() - rest.clone())
            * (parent.clone() + emitted.clone() + rest.clone())
            * (parent.clone() - emitted.clone() + rest.clone())
            * (parent.clone() + emitted.clone() - rest.clone());
        let q = if lambda > zero {
            lambda.sqrt() / (two.clone() * parent.clone())
        } else {
            zero.clone()
        };
        if massive {
            let (k_parent, k_rest) = (&kappa[i - 1], &kappa[i]);
            let massless_q = (k_parent.clone() - k_rest.clone())
                * (k_parent.clone() + k_rest.clone())
                / (two.clone() * k_parent.clone());
            weight = weight * (q.clone() * k_parent.clone()) / (parent.clone() * massless_q);
            if i < n - 1 {
                weight = weight * rest.clone() / k_rest.clone();
            }
        }
        let [nx, ny, nz] = direction(sqrt_s, rng);
        let first = [
            (q.clone() * q.clone() + emitted.clone() * emitted.clone()).sqrt(),
            q.clone() * nx.clone(),
            q.clone() * ny.clone(),
            q.clone() * nz.clone(),
        ];
        let second = [
            (q.clone() * q.clone() + rest.clone() * rest.clone()).sqrt(),
            -(q.clone() * nx),
            -(q.clone() * ny),
            -(q * nz),
        ];
        momenta.push(boost(&first, &frame, parent));
        frame = boost(&second, &frame, parent);
    }
    momenta.push(frame);
    if massive {
        let ratio = kappa[0].clone() / sqrt_s.clone();
        weight = weight * power(&ratio, (2 * n - 4) as u32);
    }
    Ok(FlatFinalState { momenta, weight })
}

/// The carrier of a generation (the kinematics improvement's).
pub use crate::kinematics_improvement::Arithmetic;

fn draw<T: FlatReal>(
    masses: &[Rational],
    sqrt_s: &Rational,
    seed: u64,
    cast: impl Fn(&Rational) -> T,
    back: impl Fn(&T) -> Rational,
) -> Result<(Vec<[Rational; 4]>, Rational), String> {
    let carried: Vec<T> = masses.iter().map(&cast).collect();
    let point = flat_final_state(&carried, &cast(sqrt_s), &mut stream(seed))?;
    Ok((
        point
            .momenta
            .iter()
            .map(|leg| leg.each_ref().map(&back))
            .collect(),
        back(&point.weight),
    ))
}

fn exact(x: f64) -> Rational {
    Rational::try_from(x).expect("a finite carrier value")
}

/// The flat final state of `seed` in `arithmetic` (the masses and sqrt(s)
/// cast once into the carrier): the exact rationals of the carrier's
/// momenta and weight.
pub fn flat_final_state_exact(
    masses: &[Rational],
    sqrt_s: &Rational,
    seed: u64,
    arithmetic: Arithmetic,
) -> Result<(Vec<[Rational; 4]>, Rational), String> {
    match arithmetic {
        Arithmetic::Binary64 => draw(masses, sqrt_s, seed, to_binary64, |x: &f64| exact(*x)),
        Arithmetic::DoubleDouble => draw(
            masses,
            sqrt_s,
            seed,
            |r: &Rational| -> DoubleFloat { r.into() },
            |x: &DoubleFloat| {
                let parts = (*x).into_inner();
                &exact(parts.hi()) + &exact(parts.lo())
            },
        ),
        Arithmetic::Arbitrary(bits) => draw(
            masses,
            sqrt_s,
            seed,
            |r: &Rational| Mpfr(Float::with_val(bits, 0)).rational(r),
            |x: &Mpfr| x.0.to_rational(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot(p: &[f64; 4], q: &[f64; 4]) -> f64 {
        p[0] * q[0] - p[1] * q[1] - p[2] * q[2] - p[3] * q[3]
    }

    #[test]
    fn binary64_points_conserve_and_are_on_shell() {
        let mut rng = stream(7);
        for masses in [
            vec![0.0; 2],
            vec![0.0; 5],
            vec![173.0, 173.0, 0.0],
            vec![91.0, 91.0, 0.0, 0.0],
        ] {
            for _ in 0..200 {
                let point = flat_final_state(&masses, &1000.0, &mut rng).unwrap();
                let mut total = [0.0; 4];
                for (leg, m) in point.momenta.iter().zip(&masses) {
                    for mu in 0..4 {
                        total[mu] += leg[mu];
                    }
                    assert!((dot(leg, leg) - m * m).abs() <= 1e-12 * leg[0] * leg[0]);
                }
                assert!((total[0] - 1000.0).abs() <= 1e-12 * 1000.0);
                assert!(total[1..].iter().all(|x| x.abs() <= 1e-12 * 1000.0));
                if masses.iter().all(|m| *m == 0.0) {
                    assert_eq!(point.weight, 1.0);
                } else {
                    // A Jacobian: positive (its mean is the volume ratio < 1).
                    assert!(point.weight > 0.0 && point.weight.is_finite());
                }
            }
        }
    }

    #[test]
    fn a_seed_is_reproducible_and_refusals_are_typed() {
        let a = flat_final_state(&[0.0; 4], &500.0, &mut stream(3)).unwrap();
        let b = flat_final_state(&[0.0; 4], &500.0, &mut stream(3)).unwrap();
        assert_eq!(a, b);
        assert!(flat_final_state(&[0.0], &500.0, &mut stream(3)).is_err());
        assert!(flat_final_state(&[300.0, 300.0], &500.0, &mut stream(3)).is_err());
        assert!(flat_final_state(&[-1.0, 0.0], &500.0, &mut stream(3)).is_err());
    }

    #[test]
    fn every_carrier_draws_the_same_point_to_its_rounding() {
        let masses: Vec<Rational> = [(0, 1), (0, 1), (0, 1), (173, 1), (91188, 1000)]
            .into_iter()
            .map(Rational::from)
            .collect();
        let sqrt_s = Rational::from(2000);
        // Residuals relative to sqrt(s)^2 (shells) and sqrt(s) (conservation).
        let residuals = |legs: &[[Rational; 4]]| -> (f64, f64) {
            let mut shell = 0.0f64;
            let mut total: [Rational; 4] = std::array::from_fn(|_| Rational::from(0));
            for (leg, m) in legs.iter().zip(&masses) {
                let p2 = &(&(&(&leg[0] * &leg[0]) - &(&leg[1] * &leg[1])) - &(&leg[2] * &leg[2]))
                    - &(&leg[3] * &leg[3]);
                let d = &p2 - &(m * m);
                shell = shell.max((d.to_f64() / 4.0e6).abs());
                for mu in 0..4 {
                    total[mu] = &total[mu] + &leg[mu];
                }
            }
            let mut conservation = ((&total[0] - &sqrt_s).to_f64() / 2000.0).abs();
            for x in &total[1..] {
                conservation = conservation.max((x.to_f64() / 2000.0).abs());
            }
            (shell, conservation)
        };
        for seed in 0..20u64 {
            let (b64, w64) =
                flat_final_state_exact(&masses, &sqrt_s, seed, Arithmetic::Binary64).unwrap();
            let (dd, wdd) =
                flat_final_state_exact(&masses, &sqrt_s, seed, Arithmetic::DoubleDouble).unwrap();
            let (mp, wmp) =
                flat_final_state_exact(&masses, &sqrt_s, seed, Arithmetic::Arbitrary(300)).unwrap();
            let (s64, c64) = residuals(&b64);
            let (sdd, cdd) = residuals(&dd);
            let (smp, cmp) = residuals(&mp);
            assert!(s64 < 1e-14 && c64 < 1e-14, "{s64} {c64}");
            assert!(sdd < 1e-29 && cdd < 1e-29, "{sdd} {cdd}");
            assert!(smp < 1e-85 && cmp < 1e-85, "{smp} {cmp}");
            // The same point in every carrier, to the coarser carrier's rounding.
            for (legs, bound) in [(&b64, 1e-13), (&dd, 1e-28)] {
                for (a, b) in legs.iter().zip(&mp) {
                    for mu in 0..4 {
                        assert!(
                            ((&a[mu] - &b[mu]).to_f64() / 2000.0).abs() < bound,
                            "seed {seed}"
                        );
                    }
                }
            }
            assert!(((&w64 - &wmp).to_f64() / wmp.to_f64()).abs() < 1e-13);
            assert!(((&wdd - &wmp).to_f64() / wmp.to_f64()).abs() < 1e-28);
        }
    }

    #[test]
    fn the_intermediate_mass_root_is_refined_to_the_carrier_precision() {
        for (k, bits) in [(1u32, 53u32), (3, 106), (6, 300)] {
            for numerator in [1i64, 977, 65_535] {
                let r = Rational::from((numerator, 65_536));
                let x = symbolica_ratio_root(k, &r, bits);
                // f(x) = (k + 1) x^k - k x^(k + 1) - r, f' = k (k + 1) x^(k - 1) (1 - x).
                let mut power = Rational::from(1);
                for _ in 0..k {
                    power = &power * &x;
                }
                let value = &(&power
                    * &(&Rational::from(i64::from(k) + 1) - &(&Rational::from(i64::from(k)) * &x)))
                    - &r;
                assert!(x > 0 && x < 1);
                // |f(x)| <= max|f'| * |x| 2^-(bits+1) with max|f'| <= k + 1.
                let bound = Rational::from((
                    Integer::from(i64::from(k) + 1),
                    Integer::from(2).pow(u64::from(bits) + 1),
                ));
                assert!(value.abs() <= bound, "k {k} bits {bits}");
            }
        }
    }

    #[test]
    fn the_own_binary64_root_agrees_with_symbolicas_refinement() {
        // The licence-free binary64 solver against Symbolica's exact
        // refinement, and its residual at binary64's rounding.
        let mut rng = stream(11);
        for k in 1..=8u32 {
            for _ in 0..500 {
                let odd = uniform_odd(rng.next_u64());
                let own = 0f64.ratio_root(k, odd);
                let r = exact_dyadic(odd, 65);
                let reference = to_binary64(&symbolica_ratio_root(k, &r, 53));
                assert!(own > 0.0 && own < 1.0);
                assert!(
                    (own - reference).abs() <= 8.0 * f64::EPSILON * reference,
                    "k {k}"
                );
            }
        }
        // Extreme draws: r next to 0 and 1.
        for odd in [1i128, (1i128 << 65) - 1] {
            for k in 1..=8u32 {
                let own = 0f64.ratio_root(k, odd);
                let reference = to_binary64(&symbolica_ratio_root(k, &exact_dyadic(odd, 65), 53));
                assert!(
                    (own - reference).abs()
                        <= 8.0 * f64::EPSILON * reference.max(f64::MIN_POSITIVE)
                );
            }
        }
        // The same solvers in double-double (they are generic).
        let r = DoubleFloat::from(0.3);
        let x = own_ratio_root(3, &r);
        let residual = power(&x, 3) * (DoubleFloat::from(4.0) - DoubleFloat::from(3.0) * x) - r;
        assert!(RealLike::to_f64(&residual).abs() < 1e-30);
        let q = DoubleFloat::from(1e-12);
        let y = own_complement_root(2, &q);
        let x = DoubleFloat::from(1.0) - y;
        let h = DoubleFloat::from(1.0)
            - power(&x, 2) * (DoubleFloat::from(3.0) - DoubleFloat::from(2.0) * x);
        assert!((RealLike::to_f64(&(h - q)) / 1e-12).abs() < 1e-15);
    }
}
