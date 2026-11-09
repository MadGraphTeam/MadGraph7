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

use super::dress::{Carrier, LaneData, Pool, cadd, cmul_into};
use super::kernels::lane_vector_len;
use super::plan::{Family, KeyColumns, Step, Support};
use ninjars::fast::tensor_layout::{monomial_index, monomials, ncoeff};
use std::sync::Arc;
use symbolica::domains::float::Complex;

/// One family's accumulators: per loop orders, one block of `C(R+4,4)` lane
/// vectors per stored observable column (column-major, then monomial). A
/// dressed-row family stores the columns of its support only
/// ([`Support`], lit-dress lever D: every other column is an exact zero);
/// otherwise every column.
#[derive(Clone, Debug)]
pub struct FamilyNumerator<T> {
    pub family: usize,
    pub rank: usize,
    pub columns: usize,
    /// The stored columns (`None`: every column, in order).
    pub support: Option<Arc<Support>>,
    pub accumulators: Vec<Vec<T>>,
}

impl<T: Carrier> FamilyNumerator<T> {
    /// Zero accumulators of every column (schedules F and M, the oracles).
    pub fn zeros<const L: usize>(
        family: usize,
        plan_family: &Family,
        columns: usize,
        zero: &T,
    ) -> Self {
        Self::zeros_on::<L>(family, plan_family, columns, None, zero)
    }

    /// Zero accumulators of the columns of `support` (`None`: every column).
    pub fn zeros_on<const L: usize>(
        family: usize,
        plan_family: &Family,
        columns: usize,
        support: Option<Arc<Support>>,
        zero: &T,
    ) -> Self {
        let stored = support.as_ref().map_or(columns, |s| s.columns.len());
        let len = stored * ncoeff(plan_family.rank) * lane_vector_len(L);
        Self {
            family,
            rank: plan_family.rank,
            columns,
            support,
            accumulators: (0..plan_family.orders.len())
                .map(|_| vec![zero.clone(); len])
                .collect(),
        }
    }

    /// [`Self::zeros_on`] with the accumulator buffers taken from `pool`
    /// (zeroed; lit-dress lever E).
    pub fn zeros_in<const L: usize>(
        family: usize,
        plan_family: &Family,
        columns: usize,
        support: Option<Arc<Support>>,
        pool: &mut Pool<T>,
    ) -> Self {
        let stored = support.as_ref().map_or(columns, |s| s.columns.len());
        let len = stored * ncoeff(plan_family.rank) * lane_vector_len(L);
        Self {
            family,
            rank: plan_family.rank,
            columns,
            support,
            accumulators: (0..plan_family.orders.len())
                .map(|_| pool.take(len))
                .collect(),
        }
    }

    /// Hand the accumulator buffers back to `pool`.
    pub fn recycle(self, pool: &mut Pool<T>) {
        for buffer in self.accumulators {
            pool.give(buffer);
        }
    }

    /// The stored columns in order: `(slot, column)`.
    pub fn stored(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let count = self
            .support
            .as_ref()
            .map_or(self.columns, |s| s.columns.len());
        (0..count).map(move |slot| {
            let column = self
                .support
                .as_ref()
                .map_or(slot, |s| s.columns[slot] as usize);
            (slot, column)
        })
    }

    /// The slot of observable column `column`, or `None` (an exact zero).
    pub fn slot(&self, column: usize) -> Option<usize> {
        match &self.support {
            None => (column < self.columns).then_some(column),
            Some(support) => match support.slots[column] {
                u32::MAX => None,
                slot => Some(slot as usize),
            },
        }
    }

    /// The per-column slots of the closure (empty: every column stored in
    /// order).
    pub fn slots(&self) -> &[u32] {
        self.support.as_ref().map_or(&[], |s| s.slots.as_slice())
    }

    /// The coefficient lane vectors of stored slot `slot` of accumulator
    /// `order`.
    pub fn block<const L: usize>(&self, order: usize, slot: usize) -> &[T] {
        let width = ncoeff(self.rank) * lane_vector_len(L);
        &self.accumulators[order][slot * width..][..width]
    }

    /// The coefficients of accumulator `order`, column `column` at lane `lane`
    /// (zeros for a column outside the support).
    pub fn lane<const L: usize>(
        &self,
        order: usize,
        column: usize,
        lane: usize,
    ) -> Vec<Complex<T>> {
        let width = ncoeff(self.rank);
        match self.slot(column) {
            Some(slot) => self
                .block::<L>(order, slot)
                .chunks_exact(lane_vector_len(L))
                .map(|z| Complex::new(z[lane].clone(), z[L + lane].clone()))
                .collect(),
            None => {
                let zero = self
                    .accumulators
                    .first()
                    .and_then(|a| a.first())
                    .map(|x| x.zero());
                match zero {
                    Some(zero) => vec![Complex::new(zero.clone(), zero); width],
                    None => Vec::new(),
                }
            }
        }
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

    /// Per lane whether every coefficient of every accumulator is finite on
    /// it ([`Self::finite`] of every lane in one pass over the accumulators).
    pub fn finite_lanes<const L: usize>(&self) -> [bool; L] {
        let v = lane_vector_len(L);
        let mut finite = [true; L];
        for values in &self.accumulators {
            for z in values.chunks_exact(v) {
                for (l, ok) in finite.iter_mut().enumerate() {
                    *ok &= z[l].is_finite() && z[L + l].is_finite();
                }
            }
        }
        finite
    }

    pub fn reals(&self) -> usize {
        self.accumulators.iter().map(Vec::len).sum()
    }
}

/// The trace `τ[m] = Σ_α row[α][α][m]` of one output row of a trace step
/// (its `d_alpha × d_out` blocks of `trace.len()` lane values each).
#[inline(always)]
fn row_trace<T: Carrier>(step: &Step, row: &[T], trace: &mut [T]) {
    let block = trace.len();
    super::kernels::clear(trace);
    for a in 0..step.d_alpha {
        let diagonal = &row[(a * step.d_out + a) * block..][..block];
        for (c, x) in trace.iter_mut().zip(diagonal) {
            *c += x;
        }
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
    debug_assert!(n_out <= ncoeff(rank), "a trace step above the family rank");
    let row = step.out_row() * v;
    let mut column = vec![zero.clone(); n_out * v];
    for o in 0..columns {
        row_trace(step, &output[o * row..][..row], &mut column);
        close_column::<T, L>(&column, accumulator, o, rank);
    }
}

crate::loop_v2::kernels::dispatched_avx2! {
    /// The closure of one traced object row of a seeded plan into its
    /// observable column `o`: `acc[o][m] += τ[m]` ([`trace_into`]), in the
    /// kernels' vector width up to AVX2 (lit-dress lever C1, `dispatched_avx2`;
    /// the same additions in every width).
    pub fn close_column<T: Carrier, const L: usize>(
        trace: &[T],
        accumulator: &mut [T],
        o: usize,
        rank: usize,
    ) => close_column_body;
}

#[inline(always)]
fn close_column_body<T: Carrier, const L: usize>(
    trace: &[T],
    accumulator: &mut [T],
    o: usize,
    rank: usize,
) {
    let v = lane_vector_len(L);
    let n_family = ncoeff(rank);
    let target = &mut accumulator[o * n_family * v..][..n_family * v];
    for (t, c) in target.chunks_exact_mut(v).zip(trace.chunks_exact(v)) {
        for (x, y) in t.iter_mut().zip(c) {
            *x += y;
        }
    }
}

/// Close a trace step of a dressed-row plan (route R2, M-LIT LIT-R2) into
/// an accumulator: per object row `r` of `closed` (one object column; the
/// rows some link writes, ascending, `plan::TraceRows`) its trace
/// `τ_r[m] = Σ_α out[r][α][α][m]`, then for each observable column `o` of
/// its Born row `born[r]` whose seed-weight column `w` has a non-zero
/// weight in the amplitude vector `u` of the step's seed key (`columns`,
/// [`KeyColumns`]) `acc[o][m] += u[w] τ_r[m]`, with `u` in the carrier
/// (`(re, im)` per seed-weight column, the same for every lane). The step's
/// rank is at most the family rank (checked when the plan loads). A row no
/// link writes has an exact zero trace, whose products change no
/// accumulator (one started at `+0` never holds `-0`), so skipping it is
/// bitwise; its columns need not be in the family's [`Support`] (a
/// row-pruned copy's support holds the kept rows' columns only).
#[allow(clippy::too_many_arguments)]
pub fn trace_rows_into<T: Carrier, const L: usize>(
    step: &Step,
    output: &[T],
    accumulator: &mut [T],
    born: &[u32],
    closed: &[u32],
    columns: &KeyColumns,
    slots: &[u32],
    vector: &[(T, T)],
    rank: usize,
    zero: &T,
) {
    let v = lane_vector_len(L);
    let n_out = ncoeff(step.rank_out);
    debug_assert!(n_out <= ncoeff(rank), "a trace step above the family rank");
    let row = step.out_row() * v;
    let mut trace = vec![zero.clone(); n_out * v];
    let mut term = vec![zero.clone(); v];
    for &r in closed {
        let (r, h) = (r as usize, born[r as usize]);
        row_trace(step, &output[r * row..][..row], &mut trace);
        close_row::<T, L>(
            &trace,
            accumulator,
            columns.row(h as usize),
            slots,
            vector,
            rank,
            &mut term,
        );
    }
}

crate::loop_v2::kernels::dispatched_avx2! {
    /// The closure of one traced object row of a dressed-row plan into the
    /// observable columns `(o, w)` of its Born row with a non-zero key
    /// weight: `acc[o][m] += u[w] τ[m]` ([`trace_rows_into`]), column `o`
    /// stored at `slots[o]` (empty: at `o`); `term` is one lane vector of
    /// scratch. Runs in the kernels' vector width up to AVX2 (lit-dress
    /// lever C1, `dispatched_avx2`). The columns of an exactly-zero weight
    /// are skipped: on
    /// finite traces their products are exact zeros, which change no
    /// accumulator (one started at `+0` never holds `-0`), so the values
    /// are bitwise those of the closure over every column; a non-finite
    /// trace no longer reaches a column through a zero weight (`0 · ∞`), as
    /// a column it does not belong to (a lane still fails on its non-finite
    /// trace through the columns it is closed into).
    pub fn close_row<T: Carrier, const L: usize>(
        trace: &[T],
        accumulator: &mut [T],
        columns: &[(u32, u32)],
        slots: &[u32],
        vector: &[(T, T)],
        rank: usize,
        term: &mut [T],
    ) => close_row_body;
}

#[inline(always)]
fn close_row_body<T: Carrier, const L: usize>(
    trace: &[T],
    accumulator: &mut [T],
    columns: &[(u32, u32)],
    slots: &[u32],
    vector: &[(T, T)],
    rank: usize,
    term: &mut [T],
) {
    let v = lane_vector_len(L);
    let n_family = ncoeff(rank);
    for &(o, w) in columns {
        let (re, im) = &vector[w as usize];
        let slot = if slots.is_empty() {
            o
        } else {
            slots[o as usize]
        } as usize;
        let target = &mut accumulator[slot * n_family * v..][..n_family * v];
        for (t, c) in target.chunks_exact_mut(v).zip(trace.chunks_exact(v)) {
            let (c_re, c_im) = c.split_at(L);
            let (term_re, term_im) = term[..2 * L].split_at_mut(L);
            for l in 0..L {
                term_re[l] = re.clone() * &c_re[l] - im.clone() * &c_im[l];
                term_im[l] = re.clone() * &c_im[l] + im.clone() * &c_re[l];
            }
            for (x, y) in t.iter_mut().zip(term.iter()) {
                *x += y;
            }
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
    use symbolica::domains::float::DoubleFloat;
    use symbolica::domains::rational::Rational;

    /// The closure of one traced row over every column of its Born row
    /// (before lit-dress lever C1), and over the key's non-zero columns
    /// ([`KeyColumns`], [`close_row`]), into the same accumulator.
    fn closed_both_ways<T: Carrier, const L: usize>(
        trace: &[Rational],
        accumulator: &[Rational],
        row_columns: &[Vec<(usize, usize)>],
        key: &[[Rational; 2]],
        rank: usize,
        convert: impl Fn(&Rational) -> T,
    ) -> [Vec<T>; 2] {
        let v = lane_vector_len(L);
        let n_family = ncoeff(rank);
        let trace: Vec<T> = trace.iter().map(&convert).collect();
        let vector: Vec<(T, T)> = key
            .iter()
            .map(|[re, im]| (convert(re), convert(im)))
            .collect();
        let zero = convert(&Rational::from(0));
        let mut every: Vec<T> = accumulator.iter().map(&convert).collect();
        let mut nonzero = every.clone();
        let mut term = vec![zero.clone(); v];
        for columns in row_columns {
            for &(o, w) in columns {
                let (re, im) = &vector[w];
                let target = &mut every[o * n_family * v..][..n_family * v];
                for (t, c) in target.chunks_exact_mut(v).zip(trace.chunks_exact(v)) {
                    let (c_re, c_im) = c.split_at(L);
                    let (term_re, term_im) = term.split_at_mut(L);
                    for l in 0..L {
                        term_re[l] = re.clone() * &c_re[l] - im.clone() * &c_im[l];
                        term_im[l] = re.clone() * &c_im[l] + im.clone() * &c_re[l];
                    }
                    cadd::<T, L>(t, &term);
                }
            }
        }
        let keys = KeyColumns::new(key, row_columns);
        for h in 0..row_columns.len() {
            close_row::<T, L>(
                &trace,
                &mut nonzero,
                keys.row(h),
                &[],
                &vector,
                rank,
                &mut term,
            );
        }
        [every, nonzero]
    }

    /// A family numerator storing its support only (lit-dress lever D)
    /// reads as the dense one: the stored columns in column order, every
    /// other column zero; finiteness per lane in one pass.
    #[test]
    fn supported_numerators_read_as_dense_ones() {
        const L: usize = 2;
        let v = lane_vector_len(L);
        let family = Family {
            sequence: vec![(1, 0), (2, 0)],
            rank: 1,
            orders: vec![0, 1],
            leaves: Vec::new(),
            steps: Vec::new(),
            factors: Vec::new(),
            vertices: Vec::new(),
            currents: Vec::new(),
            row_traces: Vec::new(),
            support: None,
            walk: Default::default(),
        };
        let columns = 5;
        let support = Arc::new(Support {
            columns: vec![1, 3],
            slots: vec![u32::MAX, 0, u32::MAX, 1, u32::MAX],
        });
        let mut dense = FamilyNumerator::zeros::<L>(0, &family, columns, &0.0);
        let mut compact = FamilyNumerator::zeros_on::<L>(0, &family, columns, Some(support), &0.0);
        let width = ncoeff(1) * v;
        for order in 0..2 {
            for (slot, column) in [(0usize, 1usize), (1, 3)] {
                for k in 0..width {
                    let value = (order * 100 + column * 10 + k) as f64 - 7.5;
                    dense.accumulators[order][column * width + k] = value;
                    compact.accumulators[order][slot * width + k] = value;
                }
            }
        }
        assert_eq!(compact.stored().collect::<Vec<_>>(), [(0, 1), (1, 3)]);
        assert_eq!((compact.slot(3), compact.slot(2)), (Some(1), None));
        for order in 0..2 {
            for column in 0..columns {
                for lane in 0..L {
                    assert_eq!(
                        compact.lane::<L>(order, column, lane),
                        dense.lane::<L>(order, column, lane)
                    );
                }
            }
        }
        compact.accumulators[1][width + L + 1] = f64::NAN;
        assert_eq!(compact.finite_lanes::<L>(), [true, false]);
        assert_eq!(
            (compact.finite::<L>(0), compact.finite::<L>(1)),
            (true, false)
        );
    }

    #[test]
    fn closures_skipping_zero_key_weights_are_bitwise() {
        fn check<const L: usize>(seed: u64) {
            let mut state = seed;
            let mut next = || {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                Rational::from((
                    ((state >> 33) % 2001) as i64 - 1000,
                    ((state >> 13) % 97 + 1) as i64,
                ))
            };
            let (rank, rows, weights) = (2, 3, 4);
            let v = lane_vector_len(L);
            let trace: Vec<Rational> = (0..ncoeff(rank) * v).map(|_| next()).collect();
            let columns = rows * weights;
            // Accumulators that already hold sums (and exact zeros).
            let accumulator: Vec<Rational> = (0..columns * ncoeff(rank) * v)
                .map(|k| {
                    if k % 5 == 0 {
                        Rational::from(0)
                    } else {
                        next()
                    }
                })
                .collect();
            // Born row h closes into its weights' columns h * weights + w.
            let row_columns: Vec<Vec<(usize, usize)>> = (0..rows)
                .map(|h| (0..weights).map(|w| (h * weights + w, w)).collect())
                .collect();
            // A key with two non-zero weights (a summed key), one purely
            // imaginary, and two exact zeros.
            let key = [
                [Rational::from(0), Rational::from(0)],
                [next(), next()],
                [Rational::from(0), Rational::from(0)],
                [Rational::from(0), next()],
            ];
            let keys = KeyColumns::new(&key, &row_columns);
            assert_eq!(keys.row(1), &[(5, 1), (7, 3)]);
            let bits = |values: &[f64]| values.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            let [a, b] =
                closed_both_ways::<f64, L>(&trace, &accumulator, &row_columns, &key, rank, |r| {
                    r.to_f64()
                });
            assert_eq!(bits(&a), bits(&b), "binary64");
            let [a, b] = closed_both_ways::<DoubleFloat, L>(
                &trace,
                &accumulator,
                &row_columns,
                &key,
                rank,
                |r| DoubleFloat::from(r),
            );
            assert!(a == b, "double-double");
            let [a, b] =
                closed_both_ways::<_, L>(&trace, &accumulator, &row_columns, &key, rank, |r| {
                    r.to_multi_prec_float(200)
                });
            assert!(a == b, "MPFR");
            let [a, b] = closed_both_ways::<Rational, L>(
                &trace,
                &accumulator,
                &row_columns,
                &key,
                rank,
                Rational::clone,
            );
            assert!(a == b, "exact");
        }
        check::<2>(11);
        check::<16>(12);
    }

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
