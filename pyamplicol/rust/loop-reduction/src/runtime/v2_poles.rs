// SPDX-License-Identifier: 0BSD

//! Ultraviolet residues of v2 family integrals (DESIGN.md §12.4).
//!
//! The pole selections `uv` and `ir` split the single pole of every reduction
//! lane, `μ^(2ε)/(iπ^(d/2) r_Γ) ∫ d^d q N(q) / Π_i D_i` with
//! `D_i = (q + V_i)² − m_i²` and a four-dimensional numerator
//! `N(q) = Σ_α c_α q^α`, into its ultraviolet and infrared parts; the double
//! pole is infrared. The ultraviolet residue depends only on the integrand's
//! large-`q` behaviour. With `c_i = V_i² − m_i²`,
//!
//! ```text
//! 1/D_i = Σ_k (−1)^k (2 q·V_i + c_i)^k / (q²)^(k+1),
//! ```
//!
//! and the terms of total degree −4 give `1/ε_UV` times
//!
//! ```text
//! ⟨q^β⟩ = Π_μ (β_μ − 1)!! g_μμ^(β_μ/2) / (2^k (k+1)!)   (|β| = 2k, every β_μ even),
//! ```
//!
//! the residue of `∫ q^β / (q²)^(k+2)`. The degree-`r` block of `N`
//! contributes when `j = r − 2N + 4 ≥ 0`, through the terms with `a_i` factors
//! `2 q·V_i` and `b_i` factors `c_i`, `Σ_i (a_i + 2 b_i) = j`, weighted by
//! `(−1)^(a_i+b_i) C(a_i+b_i, a_i)`. The result is the ultraviolet residue that
//! NinjaRS's reduction obtains from oneloopmaster's ultraviolet master
//! residues (`PoleSelection::UV`: A0 and the tadpole tensors, B0 and the
//! longitudinal bubble moments, scaleless bubbles included; C0, D0 and dB0
//! have none), here without a second reduction and without any master
//! evaluation, in the lane's own carrier. NinjaRS's fast engine drops
//! scaleless bubbles and refuses separated poles; its generic engine with
//! selected masters is the oracle of this residue
//! (`tests/runtime_v2_reduce.rs`).
//!
//! Interim local copy (DESIGN.md §12.4): this is master-integral algebra that
//! belongs upstream. The same functional is proposed to NinjaRS as
//! `ninjars::fast::tensor_uv` (`TensorUvResidues`, `tensor_uv_residue`;
//! branch `pyamplicol/fast-tensor-uv-residue` on main 41a4c00, with the
//! generic engine's `PoleSelection::UV` as its oracle). When it lands on
//! NinjaRS main, the re-pin replaces this module by that function.

use super::Carrier;
use crate::Complex;
use ninjars::fast::tensor_layout::{block_offset, homogeneous_index, monomials, ncoeff};
use std::borrow::Cow;

/// Homogeneous polynomials of every degree up to `max` in the tensor layout:
/// monomials per degree, the index of `m · q_μ`, and the angular average
/// `⟨q^β⟩ = numerator / denominator` of the even degrees.
struct Layout {
    exponents: Vec<Vec<[u32; 4]>>,
    raise: Vec<Vec<[usize; 4]>>,
    average: Vec<(Vec<(usize, i64)>, i64)>,
}

impl Layout {
    fn new(max: usize) -> Self {
        let all = monomials(max);
        let exponents: Vec<Vec<[u32; 4]>> = (0..=max)
            .map(|d| all[block_offset(d)..ncoeff(d)].to_vec())
            .collect();
        let raise = (0..=max)
            .map(|d| {
                exponents[d]
                    .iter()
                    .map(|&m| {
                        std::array::from_fn(|mu| {
                            let mut up = m;
                            up[mu] += 1;
                            homogeneous_index(up)
                        })
                    })
                    .collect()
            })
            .collect();
        let average = (0..=max)
            .map(|d| {
                if d % 2 == 1 {
                    return (Vec::new(), 1);
                }
                let k = (d / 2) as i64;
                let denominator = (1..=k + 1).product::<i64>() << k;
                let terms = exponents[d]
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| m.iter().all(|&e| e % 2 == 0))
                    .map(|(index, m)| {
                        let pairings: i64 = m
                            .iter()
                            .map(|&e| (1..e as i64).step_by(2).product::<i64>())
                            .product();
                        // g_00 = 1, g_ii = −1: (−1)^((β1 + β2 + β3)/2).
                        let sign = if ((m[1] + m[2] + m[3]) / 2) % 2 == 0 {
                            1
                        } else {
                            -1
                        };
                        (index, sign * pairings)
                    })
                    .collect();
                (terms, denominator)
            })
            .collect();
        Self {
            exponents,
            raise,
            average,
        }
    }
}

/// One lane's inputs to the residue walk.
struct Walk<'a, T: Carrier> {
    layout: &'a Layout,
    /// `w_μ = g_μμ V_i^μ` (`q·V_i = Σ_μ w_μ q^μ`) per propagator.
    covariant: Vec<[Complex<T>; 4]>,
    /// `c_i = V_i² − m_i²` per propagator.
    constants: Vec<Complex<T>>,
    zero: Complex<T>,
    total: Complex<T>,
}

impl<T: Carrier> Walk<'_, T> {
    fn integer(&self, value: i64) -> Complex<T> {
        Complex::new(self.zero.re.from_i64(value), self.zero.re.zero())
    }

    /// `polynomial · (q·V_i)` for a homogeneous polynomial of `degree`.
    fn raise(&self, polynomial: &[Complex<T>], degree: usize, i: usize) -> Vec<Complex<T>> {
        let mut out = vec![self.zero.clone(); self.layout.exponents[degree + 1].len()];
        for (value, targets) in polynomial.iter().zip(&self.layout.raise[degree]) {
            if value.re.is_zero() && value.im.is_zero() {
                continue;
            }
            for (target, w) in targets.iter().zip(&self.covariant[i]) {
                out[*target] += value.clone() * w;
            }
        }
        out
    }

    fn average(&self, polynomial: &[Complex<T>], degree: usize) -> Complex<T> {
        let (terms, denominator) = &self.layout.average[degree];
        let mut sum = self.zero.clone();
        for &(index, weight) in terms {
            sum += polynomial[index].clone() * &self.integer(weight);
        }
        sum * &(self.integer(1) / &self.integer(*denominator))
    }

    /// Distribute the remaining `j` over propagators `i..`.
    fn walk(
        &mut self,
        i: usize,
        remaining: usize,
        polynomial: &[Complex<T>],
        degree: usize,
        weight: Complex<T>,
    ) {
        if i == self.constants.len() {
            if remaining == 0 && degree % 2 == 0 {
                let value = self.average(polynomial, degree);
                self.total += weight * &value;
            }
            return;
        }
        let shifted = self.covariant[i]
            .iter()
            .any(|w| !w.re.is_zero() || !w.im.is_zero());
        let constant = self.constants[i].clone();
        let has_constant = !constant.re.is_zero() || !constant.im.is_zero();
        let mut current = polynomial.to_vec();
        let mut current_degree = degree;
        for a in 0..=remaining {
            if a > 0 {
                if !shifted {
                    break;
                }
                current = self.raise(&current, current_degree, i);
                current_degree += 1;
            }
            let mut power = self.integer(1);
            for b in 0..=(remaining - a) / 2 {
                if b > 0 {
                    if !has_constant {
                        break;
                    }
                    power = power * &constant;
                }
                // (−1)^(a+b) C(a+b, a) 2^a c_i^b.
                let binomial = (1..=a as i64).fold(1i64, |c, k| c * (b as i64 + k) / k);
                let sign = if (a + b) % 2 == 0 { 1 } else { -1 };
                let factor = self.integer(sign * (binomial << a)) * &power;
                self.walk(
                    i + 1,
                    remaining - a - 2 * b,
                    &current,
                    current_degree,
                    weight.clone() * &factor,
                );
            }
        }
    }
}

/// The ultraviolet residue (coefficient of `1/ε_UV`) of one lane: `n`
/// propagators with real or complex shifts `shifts` (`4n`, component-major
/// per propagator), squared masses `masses` (`n`) and the dense rank-`rank`
/// coefficients `coefficients` (`binomial(rank + 4, 4)`, NinjaRS's tensor
/// layout), at the largest binary precision of the inputs.
pub fn uv_residue<T: Carrier>(
    n: usize,
    rank: usize,
    shifts: &[Complex<T>],
    masses: &[Complex<T>],
    coefficients: &[Complex<T>],
) -> Complex<T> {
    let precision = shifts
        .iter()
        .chain(masses)
        .chain(coefficients)
        .map(|c| c.re.get_precision().max(c.im.get_precision()))
        .max()
        .unwrap_or(0);
    UvResidues::new(n, rank).residue(shifts, masses, coefficients, precision)
}

/// `value` with both parts at no less than `precision` binary digits. An
/// exact input (a zero or a small integer of an evaluator) can carry a low
/// precision under Symbolica's loose tracking, and an MPFR product rounds to
/// the smaller precision of its factors, so every input of the walk is lifted
/// to the carrier's working precision first (exact; a no-op for binary64 and
/// DoubleFloat).
fn lifted<T: Carrier>(value: &Complex<T>, precision: u32) -> Complex<T> {
    let mut value = value.clone();
    for part in [&mut value.re, &mut value.im] {
        if part.get_precision() < precision {
            part.set_precision(precision);
        }
    }
    value
}

/// `values` lifted to `precision`, borrowed when none needs it (binary64 and
/// DoubleFloat always).
fn lifted_all<T: Carrier>(values: &[Complex<T>], precision: u32) -> Cow<'_, [Complex<T>]> {
    let low = |c: &Complex<T>| c.re.get_precision() < precision || c.im.get_precision() < precision;
    if values.iter().any(low) {
        Cow::Owned(values.iter().map(|c| lifted(c, precision)).collect())
    } else {
        Cow::Borrowed(values)
    }
}

/// [`uv_residue`] for every lane of one `(n, rank)` bucket: the monomial
/// tables are built once.
pub struct UvResidues {
    n: usize,
    rank: usize,
    layout: Option<Layout>,
}

impl UvResidues {
    pub fn new(n: usize, rank: usize) -> Self {
        let divergent = rank + 4 >= 2 * n;
        Self {
            n,
            rank,
            // Degrees reached: r + Σ a_i <= rank + j = 2 rank − 2n + 4.
            layout: divergent.then(|| Layout::new(2 * rank + 4 - 2 * n)),
        }
    }

    /// The residue of one lane (inputs as [`uv_residue`]) at the carrier's
    /// working `precision` (binary digits; every input is lifted to it).
    pub fn residue<T: Carrier>(
        &self,
        shifts: &[Complex<T>],
        masses: &[Complex<T>],
        coefficients: &[Complex<T>],
        precision: u32,
    ) -> Complex<T> {
        let (n, rank) = (self.n, self.rank);
        let prototype = lifted(&coefficients[0], precision);
        let zero = Complex::new(prototype.re.zero(), prototype.re.zero());
        let Some(layout) = &self.layout else {
            return zero;
        };
        let shifts = lifted_all(shifts, precision);
        let masses = lifted_all(masses, precision);
        let metric = [1, -1, -1, -1];
        let mut walk = Walk {
            layout,
            covariant: (0..n)
                .map(|i| {
                    std::array::from_fn(|mu| {
                        let v = shifts[4 * i + mu].clone();
                        if metric[mu] > 0 { v } else { -v }
                    })
                })
                .collect(),
            constants: (0..n)
                .map(|i| {
                    let v = &shifts[4 * i..4 * i + 4];
                    let square = v[0].clone() * &v[0]
                        - v[1].clone() * &v[1]
                        - v[2].clone() * &v[2]
                        - v[3].clone() * &v[3];
                    square - &masses[i]
                })
                .collect(),
            zero: zero.clone(),
            total: zero.clone(),
        };
        for r in (2 * n).saturating_sub(4)..=rank {
            let block = &coefficients[block_offset(r)..ncoeff(r)];
            if block.iter().all(|c| c.re.is_zero() && c.im.is_zero()) {
                continue;
            }
            let block = lifted_all(block, precision);
            let one = walk.integer(1);
            walk.walk(0, r + 4 - 2 * n, &block, r, one);
        }
        walk.total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(x: f64) -> Complex<f64> {
        Complex::new(x, 0.)
    }

    fn tensor(rank: usize, terms: &[([u32; 4], f64)]) -> Vec<Complex<f64>> {
        let mut out = vec![c(0.); ncoeff(rank)];
        for &(m, value) in terms {
            out[ninjars::fast::tensor_layout::monomial_index(m)] = c(value);
        }
        out
    }

    #[test]
    fn exact_low_precision_inputs_are_lifted_to_the_working_precision() {
        // MPFR: an evaluator's exact coefficients and masses at 2 binary digits
        // against 128-bit shifts. Unlifted, every product rounds to 2 digits
        // (the native executor's MPFR residues at rotated points).
        use symbolica::domains::float::{Float, FloatLike, RealLike, SingleFloat};
        let mpfr =
            |x: f64, bits: u32| Complex::new(Float::with_val(bits, x), Float::with_val(bits, 0));
        let shifts = [
            0.0, 0.0, 0.0, 0.0, 0.3, 0.1, -0.7, 0.45, 1.1, -0.35, 0.2, 0.6,
        ];
        let masses = [3.0, 0.0, 2.0];
        let terms = [([2, 0, 0, 0], 3.), ([0, 1, 1, 0], -2.), ([1, 0, 0, 1], 3.)];
        let numerator = tensor(3, &terms);
        let mut cubic = numerator.clone();
        cubic[ninjars::fast::tensor_layout::monomial_index([1, 1, 1, 0])] = c(3.);
        let expected = uv_residue(3, 3, &shifts.map(c), &masses.map(c), &cubic);
        let low = |v: &[Complex<f64>]| -> Vec<Complex<Float>> {
            v.iter().map(|x| mpfr(x.re, 2)).collect()
        };
        let got = UvResidues::new(3, 3).residue(
            &shifts.map(|x| mpfr(x, 128)),
            &low(&masses.map(c)),
            &low(&cubic),
            128,
        );
        assert!(got.re.get_precision() >= 128);
        let difference = (got.re.to_f64() - expected.re).abs();
        assert!(
            difference <= 1e-14 * expected.re.abs().max(1.),
            "{difference}"
        );
        assert!(got.im.is_zero() || got.im.to_f64().abs() <= 1e-14);
    }

    #[test]
    fn scalar_tadpole_and_bubble_residues() {
        // A0(m²): m², at any shift.
        let shift = [c(3.), c(1.), c(-2.), c(0.5)];
        let one = tensor(0, &[([0; 4], 1.)]);
        assert_eq!(uv_residue(1, 0, &[c(0.); 4], &[c(7.)], &one), c(7.));
        assert!((uv_residue(1, 0, &shift, &[c(7.)], &one).re - 7.).abs() < 1e-13);
        // B0: 1, also scaleless.
        let shifts = [c(0.), c(0.), c(0.), c(0.), c(2.), c(0.), c(0.), c(2.)];
        assert_eq!(uv_residue(2, 0, &shifts, &[c(0.), c(0.)], &one), c(1.));
    }

    #[test]
    fn rank_two_bubble_matches_b00_and_b11() {
        // ∫ q^μ q^ν / (q² − m0²)((q + p)² − m1²): UV of
        // g^μν B00 + p^μ p^ν B11 is g^μν ((m0² + m1²)/4 − p²/12) + p^μ p^ν / 3.
        let p = [3., 1., -2., 0.5];
        let p2 = p[0] * p[0] - p[1] * p[1] - p[2] * p[2] - p[3] * p[3];
        let (m0, m1) = (5., 11.);
        let mut shifts = vec![c(0.); 4];
        shifts.extend(p.map(c));
        let metric = [1., -1., -1., -1.];
        for mu in 0..4 {
            for nu in mu..4 {
                let mut e = [0u32; 4];
                e[mu] += 1;
                e[nu] += 1;
                let numerator = tensor(2, &[(e, 1.)]);
                let got = uv_residue(2, 2, &shifts, &[c(m0), c(m1)], &numerator);
                let g = if mu == nu { metric[mu] } else { 0. };
                let expected = g * ((m0 + m1) / 4. - p2 / 12.) + p[mu] * p[nu] / 3.;
                assert!(
                    (got.re - expected).abs() < 1e-13,
                    "{mu}{nu}: {got:?} {expected}"
                );
            }
        }
    }

    #[test]
    fn scaleless_longitudinal_moments() {
        // B_r(0; 0, 0) longitudinal moments: N = (q·n)^r with n·p = 1 for a
        // light-like p gives UV residue (−1)^r/(r+1) (oneloopmaster).
        let p = [2., 0., 0., 2.];
        let mut shifts = vec![c(0.); 4];
        shifts.extend(p.map(c));
        // n = (1/4, 0, 0, −1/4): n·p = 1/2 + 1/2 = 1, n² = 0.
        let n_vec: [f64; 4] = [0.25, 0., 0., -0.25];
        let covariant = [n_vec[0], -n_vec[1], -n_vec[2], -n_vec[3]];
        for rank in 0..=4usize {
            // Expand (q·n)^rank into the tensor layout.
            let mut numerator = vec![c(0.); ncoeff(rank)];
            for m in monomials(rank).into_iter().skip(block_offset(rank)) {
                let mut value = (1..=rank).product::<usize>() as f64;
                for (mu, &e) in m.iter().enumerate() {
                    value *=
                        covariant[mu].powi(e as i32) / (1..=e as usize).product::<usize>() as f64;
                }
                numerator[ninjars::fast::tensor_layout::monomial_index(m)] = c(value);
            }
            let got = uv_residue(2, rank, &shifts, &[c(0.), c(0.)], &numerator);
            let expected = if rank % 2 == 0 { 1. } else { -1. } / (rank + 1) as f64;
            assert!((got.re - expected).abs() < 1e-14, "rank {rank}: {got:?}");
        }
    }

    #[test]
    fn finite_families_have_no_residue() {
        let numerator = tensor(1, &[([0, 1, 0, 0], 1.)]);
        let shifts = vec![c(0.); 12];
        assert_eq!(
            uv_residue(3, 1, &shifts, &[c(1.), c(2.), c(3.)], &numerator),
            c(0.)
        );
    }
}
