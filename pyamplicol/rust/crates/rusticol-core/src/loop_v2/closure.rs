// SPDX-License-Identifier: 0BSD

//! Closure of the cut and family numerators (DESIGN.md §5.1, §11.1).
//!
//! A trace step closes the cut (`Tr_α` of a row-free object) into the family
//! accumulator of its loop orders; species with equal masses, cut species and
//! colour classes all accumulate into one polynomial per (family, loop
//! orders, observable column), stored at the family's structural rank in
//! NinjaRS's tensor layout. Schedule F closes each skeleton with its seed and
//! maps it into the family's trie frame by the exact affine relabelling
//! `N'(q') = N(σ q' − V)`.
//!
//! [`FamilyNumerator::lane`] scatters the structure-of-arrays accumulator into
//! one lane-major coefficient vector per point, the ingress of the NinjaRS
//! adapter's buckets.

use super::dress::{Carrier, LaneData, cadd, cmul_into};
use super::kernels::lane_vector_len;
use super::plan::{Family, Step};
use ninjars::fast::tensor_layout::{monomial_index, monomials, ncoeff};
use symbolica::domains::float::Complex;

/// One family's accumulators: per loop orders, `columns × C(R+4,4)` lane
/// vectors (column-major, then monomial).
#[derive(Clone, Debug)]
pub struct FamilyNumerator<T> {
    pub family: usize,
    pub rank: usize,
    pub columns: usize,
    pub accumulators: Vec<Vec<T>>,
}

impl<T: Carrier> FamilyNumerator<T> {
    pub fn zeros<const L: usize>(
        family: usize,
        plan_family: &Family,
        columns: usize,
        zero: &T,
    ) -> Self {
        let len = columns * ncoeff(plan_family.rank) * lane_vector_len(L);
        Self {
            family,
            rank: plan_family.rank,
            columns,
            accumulators: (0..plan_family.orders.len())
                .map(|_| vec![zero.clone(); len])
                .collect(),
        }
    }

    /// The coefficients of accumulator `order`, column `column` at lane `lane`.
    pub fn lane<const L: usize>(
        &self,
        order: usize,
        column: usize,
        lane: usize,
    ) -> Vec<Complex<T>> {
        let v = lane_vector_len(L);
        let width = ncoeff(self.rank);
        let base = &self.accumulators[order][column * width * v..][..width * v];
        base.chunks_exact(v)
            .map(|z| Complex::new(z[lane].clone(), z[L + lane].clone()))
            .collect()
    }

    /// Whether every coefficient of every accumulator is finite on `lane`.
    pub fn finite<const L: usize>(&self, lane: usize) -> bool {
        let v = lane_vector_len(L);
        self.accumulators.iter().all(|values| {
            values
                .chunks_exact(v)
                .all(|z| z[lane].is_finite() && z[L + lane].is_finite())
        })
    }

    pub fn reals(&self) -> usize {
        self.accumulators.iter().map(Vec::len).sum()
    }
}

/// Close a trace step's output (one helicity row per column) into an
/// accumulator: `acc[o][m] += Σ_α out[o][α][α][m]`. The step's rank is at
/// most the family rank (checked when the plan loads), so no coefficient is
/// truncated.
pub fn trace_into<T: Carrier, const L: usize>(
    step: &Step,
    output: &[T],
    accumulator: &mut [T],
    columns: usize,
    rank: usize,
    zero: &T,
) {
    let v = lane_vector_len(L);
    let n_out = ncoeff(step.rank_out);
    let n_family = ncoeff(rank);
    debug_assert!(n_out <= n_family, "a trace step above the family rank");
    let row = step.out_row() * v;
    let mut column = vec![zero.clone(); n_out * v];
    for o in 0..columns {
        let source = &output[o * row..][..row];
        super::kernels::clear(&mut column);
        for a in 0..step.d_alpha {
            let diagonal = &source[(a * step.d_out + a) * n_out * v..][..n_out * v];
            for (c, x) in column.iter_mut().zip(diagonal) {
                *c += x;
            }
        }
        let target = &mut accumulator[o * n_family * v..][..n_family * v];
        for (t, c) in target.chunks_exact_mut(v).zip(column.chunks_exact(v)) {
            cadd::<T, L>(t, c);
        }
    }
}

/// Lane vectors of `-V(mask)`, the shift of a frame map `q = σ q' − V`.
pub fn frame_shift<T: Carrier, const L: usize>(data: &LaneData<T, L>, mask: u64) -> [Vec<T>; 4] {
    let v = lane_vector_len(L);
    let momentum = data.momentum(mask);
    std::array::from_fn(|mu| {
        let mut out = vec![data.zero.clone(); v];
        for (o, x) in out.iter_mut().zip(&momentum[mu * v..][..v]) {
            *o = -x.clone();
        }
        out
    })
}

/// Coefficients of `N(σ q + d)` from those of `N(q)` (`ncoeff(rank)` lane
/// vectors), exactly in the carrier: one univariate Taylor shift per
/// component, then the sign of the odd degrees (the order of
/// `poly_affine.substitute`).
pub fn substitute<T: Carrier, const L: usize>(
    coefficients: &[T],
    rank: usize,
    sign: i8,
    shift: &[Vec<T>; 4],
    zero: &T,
) -> Vec<T> {
    let v = lane_vector_len(L);
    let exponents = monomials(rank);
    let mut current = coefficients.to_vec();
    let mut factor = vec![zero.clone(); v];
    let mut term = vec![zero.clone(); v];
    for (axis, d) in shift.iter().enumerate() {
        let mut powers: Vec<Vec<T>> = Vec::with_capacity(rank + 1);
        let mut one = vec![zero.clone(); v];
        for x in &mut one[..L] {
            *x = zero.one();
        }
        powers.push(one);
        for k in 1..=rank {
            let mut next = vec![zero.clone(); v];
            cmul_into::<T, L>(&mut next, &powers[k - 1], d);
            powers.push(next);
        }
        let mut shifted = vec![zero.clone(); current.len()];
        for (position, source) in exponents.iter().enumerate() {
            let coefficient = &current[position * v..][..v];
            let top = source[axis];
            let mut binomial: i64 = 1;
            for power in 0..=top {
                let mut target = *source;
                target[axis] = power;
                let slot = monomial_index(target);
                // comb(top, power) * d^(top - power)
                let weight = zero.from_i64(binomial);
                for (f, p) in factor.iter_mut().zip(&powers[(top - power) as usize]) {
                    *f = weight.clone() * p;
                }
                cmul_into::<T, L>(&mut term, &factor, coefficient);
                cadd::<T, L>(&mut shifted[slot * v..][..v], &term);
                binomial = binomial * i64::from(top - power) / i64::from(power + 1);
            }
        }
        current = shifted;
    }
    if sign < 0 {
        for (position, alpha) in exponents.iter().enumerate() {
            if alpha.iter().sum::<u32>() % 2 == 1 {
                for x in &mut current[position * v..][..v] {
                    *x = -x.clone();
                }
            }
        }
    }
    current
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::domains::rational::Rational;

    /// Evaluate a lane-0 polynomial (real parts only) at `q`.
    fn evaluate(coefficients: &[Rational], rank: usize, q: [Rational; 4]) -> Rational {
        const V: usize = 2;
        monomials(rank)
            .iter()
            .enumerate()
            .fold(Rational::from(0), |total, (k, alpha)| {
                let mut term = coefficients[k * V].clone();
                for (axis, &power) in alpha.iter().enumerate() {
                    for _ in 0..power {
                        term = term * &q[axis];
                    }
                }
                total + term
            })
    }

    #[test]
    fn substitution_is_the_affine_map() {
        let rank = 3;
        let zero = Rational::from(0);
        let coefficients: Vec<Rational> = (0..ncoeff(rank))
            .flat_map(|k| [Rational::from(((k * 7) % 11) as i64 - 5), Rational::from(0)])
            .collect();
        let d = [(1, 2), (-3, 1), (2, 3), (0, 1)]
            .map(|(a, b)| vec![Rational::from((a, b)), Rational::from(0)]);
        let q = [(3, 1), (-1, 2), (5, 7), (1, 1)].map(|(a, b)| Rational::from((a, b)));
        for sign in [1i8, -1] {
            let mapped = substitute::<Rational, 1>(&coefficients, rank, sign, &d, &zero);
            let moved =
                std::array::from_fn(|mu| Rational::from(i64::from(sign)) * &q[mu] + &d[mu][0]);
            assert_eq!(
                evaluate(&mapped, rank, q.clone()),
                evaluate(&coefficients, rank, moved)
            );
        }
    }
}
