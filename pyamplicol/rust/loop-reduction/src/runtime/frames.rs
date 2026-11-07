// SPDX-License-Identifier: 0BSD

//! Reduction frames shared by the reduction backends (m-lit PLAN §2.b.6):
//! the relabelled, origin-shifted and rescaled frames of a family given in
//! loop-cycle order, applied to one lane's (or one kinematic row's) inputs
//! and coefficients, and the exact undo of a rescaled frame's integral.
//!
//! NinjaRS's adapter (`v2_reduce`) re-reduces every lane in one such frame
//! for its reduction-direction check; the TIR backend (`v2_tir`) computes
//! one row of tensor moments per frame and contracts each lane's
//! coefficients, re-expanded into the frame, with it. Both apply the same
//! operations here, in the same order.

use super::Carrier;
use crate::Complex;
use ninjars::fast::tensor_layout::{block_offset, monomial_index, monomials, ncoeff};

/// A relabelled reduction frame of a family given in loop-cycle order.
///
/// Propagator `j` of the frame is propagator `order[j]` of the family, with
/// `order[j] = rotation + j` (or `rotation - j` if `reverse`) modulo the
/// number of propagators. `reverse` also reverses the loop momentum,
/// `σ = -1`. Without `shift` the loop momentum is `q = σ q'`, the shifts are
/// `V'_j = σ V_order[j]` and the coefficients are reused exactly. With
/// `shift`, `q = σ q' - V_order[0]`, so that `V'_0 = 0`, and the numerator is
/// re-expanded exactly in the working precision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReductionFrame {
    pub reverse: bool,
    pub rotation: usize,
    pub shift: bool,
    pub rescale: bool,
}

/// The scale factor `x = 7/10` of a rescaled frame (non-dyadic, so that
/// every scaled input rounds differently).
pub const RESCALING: (i64, i64) = (7, 10);

impl ReductionFrame {
    /// The loop direction reversed: `q -> -q` with the cycle traversed
    /// backwards from the second propagator, so that every family of two or
    /// more propagators also changes its first line (with rotation 0 a bubble
    /// keeps its order, and its reversed reduction is often bitwise equal to
    /// the direct one). The coefficients are reused exactly.
    pub const REVERSED: Self = Self {
        reverse: true,
        rotation: 1,
        shift: false,
        rescale: false,
    };

    /// [`Self::REVERSED`] with every dimensionful input rescaled: the masters
    /// are recomputed (and so tested) at the cost of their evaluation.
    pub const RESCALED: Self = Self {
        rescale: true,
        ..Self::REVERSED
    };

    pub(crate) fn order(self, n: usize) -> Vec<usize> {
        let rotation = self.rotation % n;
        (0..n)
            .map(|j| {
                if self.reverse {
                    (rotation + n - j) % n
                } else {
                    (rotation + j) % n
                }
            })
            .collect()
    }
}

/// Coefficients of `N(q + d)` from those of `N(q)`, rank preserved: one exact
/// binomial (Taylor) shift per loop-momentum component.
pub fn shift_coefficients<T: Carrier>(
    coefficients: &[Complex<T>],
    rank: usize,
    d: &[Complex<T>; 4],
) -> Vec<Complex<T>> {
    let zero = coefficients[0].re.zero();
    let czero = Complex::new(zero.clone(), zero.clone());
    let exponents = monomials(rank);
    let mut current = coefficients[..ncoeff(rank)].to_vec();
    for (component, shift) in d.iter().enumerate() {
        let mut powers = vec![Complex::new(zero.one(), zero.clone())];
        for k in 1..=rank {
            powers.push(powers[k - 1].clone() * shift);
        }
        let mut next = vec![czero.clone(); current.len()];
        for (alpha, c) in exponents.iter().zip(&current) {
            if c.re.is_zero() && c.im.is_zero() {
                continue;
            }
            let e = alpha[component] as usize;
            let mut target = *alpha;
            let mut binomial: i64 = 1;
            // (q + d)^e = Σ_k C(e, k) q^k d^(e - k), from k = e downwards.
            for k in (0..=e).rev() {
                target[component] = k as u32;
                let weight = Complex::new(zero.from_i64(binomial), zero.clone());
                next[monomial_index(target)] += c.clone() * &powers[e - k] * &weight;
                binomial = binomial * k as i64 / (e - k + 1) as i64;
            }
        }
        current = next;
    }
    current
}

/// Negate the odd-degree blocks: the coefficients of `N(-q)`.
pub(crate) fn reverse_coefficients<T: Carrier>(coefficients: &mut [Complex<T>], rank: usize) {
    for degree in (1..=rank).step_by(2) {
        for c in &mut coefficients[block_offset(degree)..ncoeff(degree)] {
            *c = -c.clone();
        }
    }
}

/// The rescaling factor [`RESCALING`] in the carrier of `like`.
pub(crate) fn rescaling<T: Carrier>(like: &Complex<T>) -> Complex<T> {
    let zero = like.re.zero();
    let (numerator, denominator) = RESCALING;
    Complex::new(
        zero.from_i64(numerator) / &zero.from_i64(denominator),
        zero.clone(),
    )
}

/// One frame (or the direct frame) applied to the inputs of the lanes or
/// rows of `n` propagators: the order of the frame's propagators, its loop
/// direction, origin shift and the rescaling factor in the carrier.
pub(crate) struct FrameMap<T> {
    n: usize,
    order: Vec<usize>,
    reverse: bool,
    shift: bool,
    scale: Option<Complex<T>>,
    scale_squared: Option<Complex<T>>,
}

fn scaled<T: Carrier>(v: Complex<T>, factor: &Option<Complex<T>>) -> Complex<T> {
    match factor {
        Some(x) => v * x,
        None => v,
    }
}

impl<T: Carrier> FrameMap<T> {
    /// `frame` (None: the direct frame) over `n` propagators; `like` is any
    /// value of the carrier (the rescaling factor's precision), needed only
    /// for a rescaled frame.
    pub(crate) fn new(frame: Option<ReductionFrame>, n: usize, like: Option<&Complex<T>>) -> Self {
        let identity: Vec<_> = (0..n).collect();
        let order = frame.map_or(identity, |f| f.order(n));
        let scale = frame
            .is_some_and(|f| f.rescale)
            .then(|| like.map(rescaling))
            .flatten();
        let scale_squared = scale.clone().map(|x| x.clone() * &x);
        Self {
            n,
            order,
            reverse: frame.is_some_and(|f| f.reverse),
            shift: frame.is_some_and(|f| f.shift),
            scale,
            scale_squared,
        }
    }

    /// The frame's shifts of one lane's `4n` shifts (component-major per
    /// propagator), appended to `out`; returns the origin `V_order[0]`, from
    /// which [`Self::coefficients`] re-expands a shifted frame's numerator.
    pub(crate) fn shifts(&self, shifts: &[Complex<T>], out: &mut Vec<Complex<T>>) -> [Complex<T>; 4] {
        let vector =
            |i: usize| -> [Complex<T>; 4] { std::array::from_fn(|mu| shifts[4 * i + mu].clone()) };
        let origin = vector(self.order[0]);
        for &i in &self.order {
            for (mu, v) in vector(i).into_iter().enumerate() {
                let v = if self.shift { v - &origin[mu] } else { v };
                out.push(scaled(if self.reverse { -v } else { v }, &self.scale));
            }
        }
        origin
    }

    /// The frame's `n` squared masses, appended to `out`.
    pub(crate) fn masses(&self, masses: &[Complex<T>], out: &mut Vec<Complex<T>>) {
        out.extend(
            self.order
                .iter()
                .map(|&i| scaled(masses[i].clone(), &self.scale_squared)),
        );
    }

    /// The frame's squared scale (or any squared dimensionful input).
    pub(crate) fn squared(&self, value: &Complex<T>) -> Complex<T> {
        scaled(value.clone(), &self.scale_squared)
    }

    /// The frame's `n²` invariants (row-major), appended to `out`.
    pub(crate) fn s_matrix(&self, s: &[Complex<T>], out: &mut Vec<Complex<T>>) {
        let n = self.n;
        for &i in &self.order {
            out.extend(
                self.order
                    .iter()
                    .map(|&j| scaled(s[i * n + j].clone(), &self.scale_squared)),
            );
        }
    }

    /// The frame's coefficients of a rank-`rank` numerator (`source` holds at
    /// least `binomial(rank + 4, 4)` of them), appended to `out`; `origin` is
    /// [`Self::shifts`]'s.
    pub(crate) fn coefficients(
        &self,
        source: &[Complex<T>],
        rank: usize,
        origin: &[Complex<T>; 4],
        out: &mut Vec<Complex<T>>,
    ) {
        let start = out.len();
        if self.shift {
            let d = origin.clone().map(|v| -v);
            out.extend(shift_coefficients(source, rank, &d));
        } else {
            out.extend_from_slice(&source[..ncoeff(rank)]);
        }
        if self.reverse {
            reverse_coefficients(&mut out[start..], rank);
        }
        if let Some(x) = &self.scale {
            // N'(q') = N(q'/x): degree-d coefficients times x^-d.
            let inverse = x.one() / x;
            let mut power = inverse.clone();
            for degree in 1..=rank {
                for c in &mut out[start + block_offset(degree)..start + ncoeff(degree)] {
                    *c = c.clone() * &power;
                }
                power = power * &inverse;
            }
        }
    }

    /// The factor that undoes a rescaled frame on its integrals: they are
    /// `x^(4-2N)` times the family's (every Laurent order, the scale moving
    /// with them), so this is `x^(2N) x^-4`; None for an unscaled frame.
    pub(crate) fn undo(&self) -> Option<Complex<T>> {
        self.scale.as_ref().map(|x| {
            let inverse = x.one() / x;
            (0..2 * self.n).fold(x.one(), |power, _| power * x)
                * &(0..4).fold(x.one(), |power, _| power * &inverse)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_shift_and_reversal_match_direct_evaluation() {
        let rank = 3;
        let coefficients: Vec<_> = (0..ncoeff(rank))
            .map(|k| Complex::new(k as f64 - 7.0, 0.5 * k as f64))
            .collect();
        let evaluate = |c: &[Complex<f64>], q: [f64; 4]| {
            monomials(rank)
                .iter()
                .zip(c)
                .fold(Complex::new(0., 0.), |total, (alpha, c)| {
                    let monomial: f64 = alpha
                        .iter()
                        .zip(q)
                        .map(|(&e, x)| x.powi(e as i32))
                        .product();
                    total + *c * Complex::new(monomial, 0.)
                })
        };
        // Dyadic values keep every operation below exact in binary64.
        let d = [0.5, -1.25, 2.0, 0.75].map(|x| Complex::new(x, 0.));
        let q = [1.5, 0.25, -0.5, 2.0];
        let shifted = shift_coefficients(&coefficients, rank, &d);
        let moved = std::array::from_fn(|mu| q[mu] + d[mu].re);
        assert_eq!(evaluate(&shifted, q), evaluate(&coefficients, moved));
        let mut reversed = coefficients.clone();
        reverse_coefficients(&mut reversed, rank);
        assert_eq!(
            evaluate(&reversed, q),
            evaluate(&coefficients, q.map(|x| -x))
        );
    }

    #[test]
    fn frames_are_dihedral_relabellings() {
        assert_eq!(ReductionFrame::REVERSED.order(4), [1, 0, 3, 2]);
        assert_eq!(ReductionFrame::REVERSED.order(2), [1, 0]);
        let frame = ReductionFrame {
            reverse: false,
            rotation: 6,
            shift: true,
            rescale: false,
        };
        assert_eq!(frame.order(4), [2, 3, 0, 1]);
    }
}
