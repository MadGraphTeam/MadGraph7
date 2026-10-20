// SPDX-License-Identifier: 0BSD

//! Open-loop polynomial algebra in the NinjaRS tensor-numerator layout.
//!
//! Open-loop objects and loop-segment kernels are polynomials in the loop
//! momentum `q = (q0, q1, q2, q3)`. A rank-`r` polynomial stores the
//! `ncoeff(r)` raw monomial coefficients of NinjaRS's `TensorNumerator`, in the
//! storage order of `ninjars::fast::tensor_layout`, with no symmetry factors.
//! A closed family therefore scatters into NinjaRS without reordering
//! (DESIGN.md §3.5, §5.2).
//!
//! Dressing an object with a segment `X(q) = sum_k c_k q^(a_k)` uses one
//! monomial-product table built once from `tensor_layout`. There are no
//! per-rank kernels and no rank cap; the usual `Delta = 1` step
//! `O·Y + raise(O·Z_mu, q_mu)` is the degree-one case of the same table.
//! Coefficients are generic over the numerical carrier (`Complex<f64>`,
//! `Complex<DoubleFloat>`, `Complex<Float>` or exact `Complex<Rational>`), and
//! lanes (phase-space points) are the innermost, contiguous index.

use crate::{RusticolError, RusticolResult};
use ninjars::fast::tensor_layout::{monomial_index, monomials, ncoeff};
use symbolica::domains::float::FloatLike;

/// Powers of `(q0, q1, q2, q3)` of one monomial.
pub type Exponents = [u32; 4];

/// Storage index of the product of two monomials.
///
/// `target(a, b)` is the index, in a rank `left_rank + right_rank` tensor, of
/// the product of the monomials stored at index `a` of a rank-`left_rank`
/// tensor and at index `b` of a rank-`right_rank` tensor. Every index comes
/// from `tensor_layout::monomial_index`, so the table follows NinjaRS's
/// ordering by construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MonomialProductTable {
    left_rank: usize,
    right_rank: usize,
    right_len: usize,
    targets: Vec<usize>,
}

impl MonomialProductTable {
    pub fn new(left_rank: usize, right_rank: usize) -> Self {
        let right = monomials(right_rank);
        let targets = monomials(left_rank)
            .iter()
            .flat_map(|left| {
                right.iter().map(move |right| {
                    monomial_index(std::array::from_fn(|axis| left[axis] + right[axis]))
                })
            })
            .collect();
        Self {
            left_rank,
            right_rank,
            right_len: right.len(),
            targets,
        }
    }

    /// The raise table of rank-`rank` objects: multiplication by one `q_mu`.
    pub fn raise(rank: usize) -> Self {
        Self::new(rank, 1)
    }

    pub fn left_rank(&self) -> usize {
        self.left_rank
    }

    pub fn right_rank(&self) -> usize {
        self.right_rank
    }

    pub fn output_rank(&self) -> usize {
        self.left_rank + self.right_rank
    }

    #[inline]
    pub fn target(&self, left: usize, right: usize) -> usize {
        self.targets[left * self.right_len + right]
    }

    /// Index of `q_direction` times the monomial stored at `left`.
    ///
    /// `tensor_layout` stores the rank-one block as `q0, q1, q2, q3` right after
    /// the constant, so `q_direction` is right index `1 + direction`.
    #[inline]
    pub fn raised(&self, left: usize, direction: usize) -> usize {
        debug_assert!(self.right_rank >= 1 && direction < 4);
        self.target(left, 1 + direction)
    }
}

/// One coefficient slot of a segment kernel: `X[loop_in][out] += c q^exponents`.
///
/// This is the native form of the Python catalogue's `LoopMonomialEntry` of a
/// fused segment or propagator numerator.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SegmentEntry {
    pub loop_in: usize,
    pub out: usize,
    pub exponents: Exponents,
}

/// The sparse q-structure of one segment kernel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SegmentLayout {
    loop_in_dimension: usize,
    out_dimension: usize,
    degree: usize,
    entries: Vec<SegmentEntry>,
    monomials: Vec<usize>,
}

impl SegmentLayout {
    pub fn new(
        loop_in_dimension: usize,
        out_dimension: usize,
        entries: Vec<SegmentEntry>,
    ) -> RusticolResult<Self> {
        if entries.is_empty() {
            return Err(RusticolError::invalid_argument(
                "a loop segment needs at least one coefficient",
            ));
        }
        let mut seen = std::collections::HashSet::with_capacity(entries.len());
        for entry in &entries {
            if entry.loop_in >= loop_in_dimension || entry.out >= out_dimension {
                return Err(RusticolError::invalid_argument(format!(
                    "loop segment entry {entry:?} is outside a \
                     {loop_in_dimension}x{out_dimension} kernel"
                )));
            }
            if !seen.insert(*entry) {
                return Err(RusticolError::invalid_argument(format!(
                    "loop segment entry {entry:?} is repeated"
                )));
            }
        }
        let degree = entries
            .iter()
            .map(|entry| entry.exponents.iter().sum::<u32>() as usize)
            .max()
            .unwrap_or(0);
        let monomials = entries
            .iter()
            .map(|entry| monomial_index(entry.exponents))
            .collect();
        Ok(Self {
            loop_in_dimension,
            out_dimension,
            degree,
            entries,
            monomials,
        })
    }

    pub fn loop_in_dimension(&self) -> usize {
        self.loop_in_dimension
    }

    pub fn out_dimension(&self) -> usize {
        self.out_dimension
    }

    /// Structural q-degree `Delta` of the kernel.
    pub fn degree(&self) -> usize {
        self.degree
    }

    pub fn entries(&self) -> &[SegmentEntry] {
        &self.entries
    }
}

/// `rows x columns` polynomials of one rank over `lanes` points.
///
/// Storage is `[row][column][monomial][lane]`: a row is one helicity or
/// observable row of an open-loop object, a column is the current-end index
/// and the lanes of one coefficient are contiguous.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenLoopPolynomial<T> {
    rows: usize,
    columns: usize,
    rank: usize,
    lanes: usize,
    data: Vec<T>,
}

impl<T: FloatLike> OpenLoopPolynomial<T> {
    /// A zero object; `zero` fixes the carrier precision.
    pub fn zeros(rows: usize, columns: usize, rank: usize, lanes: usize, zero: &T) -> Self {
        Self {
            rows,
            columns,
            rank,
            lanes,
            data: vec![zero.zero(); rows * columns * ncoeff(rank) * lanes],
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn columns(&self) -> usize {
        self.columns
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn lanes(&self) -> usize {
        self.lanes
    }

    #[inline]
    fn offset(&self, row: usize, column: usize, monomial: usize) -> usize {
        ((row * self.columns + column) * ncoeff(self.rank) + monomial) * self.lanes
    }

    /// The lane values of one coefficient.
    pub fn coefficient(&self, row: usize, column: usize, monomial: usize) -> &[T] {
        let start = self.offset(row, column, monomial);
        &self.data[start..start + self.lanes]
    }

    pub fn coefficient_mut(&mut self, row: usize, column: usize, monomial: usize) -> &mut [T] {
        let start = self.offset(row, column, monomial);
        &mut self.data[start..start + self.lanes]
    }

    /// Evaluate one polynomial of one lane at a numerical loop momentum.
    pub fn evaluate(&self, row: usize, column: usize, lane: usize, q: &[T; 4]) -> T {
        let one = q[0].one();
        let powers: Vec<Vec<T>> = q
            .iter()
            .map(|component| {
                std::iter::successors(Some(one.clone()), |power| Some(power.clone() * component))
                    .take(self.rank + 1)
                    .collect()
            })
            .collect();
        monomials(self.rank)
            .iter()
            .enumerate()
            .fold(one.zero(), |sum, (index, exponents)| {
                let monomial = exponents
                    .iter()
                    .zip(&powers)
                    .fold(one.clone(), |product, (&power, table)| {
                        product * &table[power as usize]
                    });
                sum + monomial * &self.coefficient(row, column, index)[lane]
            })
    }
}

/// Accumulate `output += input · X` for one segment kernel `X`.
///
/// `coefficients[entry * lanes + lane]` holds the evaluated coefficient of
/// `layout.entries()[entry]` for each lane, and `product` must be the table of
/// `(input.rank(), layout.degree())`. Rows are carried through unchanged.
pub fn apply_segment<T: FloatLike>(
    input: &OpenLoopPolynomial<T>,
    layout: &SegmentLayout,
    coefficients: &[T],
    product: &MonomialProductTable,
    output: &mut OpenLoopPolynomial<T>,
) -> RusticolResult<()> {
    let lanes = input.lanes;
    if input.columns != layout.loop_in_dimension
        || output.columns != layout.out_dimension
        || output.rows != input.rows
        || output.lanes != lanes
    {
        return Err(RusticolError::invalid_argument(
            "loop segment shapes do not match the open-loop objects",
        ));
    }
    if product.left_rank != input.rank
        || product.right_rank != layout.degree
        || product.output_rank() != output.rank
    {
        return Err(RusticolError::invalid_argument(format!(
            "rank {} object times degree {} segment needs a ({}, {}) product table \
             and a rank {} output",
            input.rank,
            layout.degree,
            input.rank,
            layout.degree,
            input.rank + layout.degree
        )));
    }
    if coefficients.len() != layout.entries.len() * lanes {
        return Err(RusticolError::invalid_argument(
            "loop segment coefficients do not cover every entry and lane",
        ));
    }
    let input_terms = ncoeff(input.rank);
    for ((entry, &right), factors) in layout
        .entries
        .iter()
        .zip(&layout.monomials)
        .zip(coefficients.chunks_exact(lanes))
    {
        for row in 0..input.rows {
            for left in 0..input_terms {
                let source = input.offset(row, entry.loop_in, left);
                let target = output.offset(row, entry.out, product.target(left, right));
                for (lane, factor) in factors.iter().enumerate() {
                    output.data[target + lane] += input.data[source + lane].clone() * factor;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ninjars::fast::tensor_layout::{block_offset, homogeneous_index};
    use symbolica::domains::{
        float::{Complex, DoubleFloat, Float},
        rational::Rational,
    };

    /// Deterministic small integers, so exact and floating carriers share data.
    struct Sequence(u64);

    impl Sequence {
        fn next(&mut self) -> i64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 33) % 19) as i64 - 9
        }
    }

    fn exponents(rank: usize) -> impl Iterator<Item = Exponents> {
        let rank = rank as u32;
        (0..=rank).flat_map(move |a| {
            (0..=rank - a)
                .flat_map(move |b| (0..=rank - a - b).map(move |c| [a, b, c, rank - a - b - c]))
        })
    }

    #[test]
    fn raise_order_equals_homogeneous_index() {
        for rank in 0..=8 {
            let table = MonomialProductTable::raise(rank);
            let layout = monomials(rank);
            assert_eq!(layout.len(), ncoeff(rank));
            for (index, monomial) in layout.iter().enumerate() {
                for direction in 0..4 {
                    let mut raised = *monomial;
                    raised[direction] += 1;
                    let degree = raised.iter().sum::<u32>() as usize;
                    assert_eq!(
                        table.raised(index, direction),
                        block_offset(degree) + homogeneous_index(raised),
                        "rank {rank} monomial {monomial:?} direction {direction}"
                    );
                }
            }
        }
    }

    #[test]
    fn product_table_covers_every_homogeneous_block() {
        for left_rank in 0..=6 {
            for right_rank in 0..=3 {
                let table = MonomialProductTable::new(left_rank, right_rank);
                for degree in 0..=right_rank {
                    for right in exponents(degree) {
                        for left_degree in 0..=left_rank {
                            for left in exponents(left_degree) {
                                let product = std::array::from_fn(|axis| left[axis] + right[axis]);
                                assert_eq!(
                                    table.target(monomial_index(left), monomial_index(right)),
                                    block_offset(left_degree + degree) + homogeneous_index(product)
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Random integer segment data shared by every carrier: integers keep
    /// every floating carrier exact, so all carriers must agree bitwise.
    struct Problem {
        rank: usize,
        object: Vec<(i64, i64)>,
        entries: Vec<SegmentEntry>,
        coefficients: Vec<(i64, i64)>,
        momenta: Vec<[i64; 4]>,
    }

    const ROWS: usize = 2;
    const COLUMNS: usize = 4;
    const OUT: usize = 3;
    const LANES: usize = 5;

    impl Problem {
        fn new(rank: usize, degree: usize, seed: u64) -> Self {
            let mut random = Sequence(seed);
            let object = (0..ROWS * COLUMNS * ncoeff(rank) * LANES)
                .map(|_| (random.next(), random.next()))
                .collect();
            let mut entries = Vec::new();
            for loop_in in 0..COLUMNS {
                for out in 0..OUT {
                    for exponents in monomials(degree) {
                        // Sparse, but always with a top-degree entry.
                        if random.next() > -4 || exponents.iter().sum::<u32>() as usize == degree {
                            entries.push(SegmentEntry {
                                loop_in,
                                out,
                                exponents,
                            });
                        }
                    }
                }
            }
            let coefficients = (0..entries.len() * LANES)
                .map(|_| (random.next(), random.next()))
                .collect();
            let momenta = (0..LANES)
                .map(|_| std::array::from_fn(|_| random.next()))
                .collect();
            Self {
                rank,
                object,
                entries,
                coefficients,
                momenta,
            }
        }

        /// Dress the object and return, per (lane, row, column), the dressed
        /// polynomial at the lane's momentum and the product of the separately
        /// evaluated object and kernel.
        fn dress<R: FloatLike>(
            &self,
            convert: impl Fn(i64) -> R,
        ) -> (Vec<Complex<R>>, Vec<Complex<R>>) {
            let complex = |(re, im): (i64, i64)| Complex::new(convert(re), convert(im));
            let zero = complex((0, 0));
            let mut input = OpenLoopPolynomial::zeros(ROWS, COLUMNS, self.rank, LANES, &zero);
            input.data = self.object.iter().map(|&value| complex(value)).collect();
            let layout = SegmentLayout::new(COLUMNS, OUT, self.entries.clone()).unwrap();
            let coefficients: Vec<_> = self
                .coefficients
                .iter()
                .map(|&value| complex(value))
                .collect();
            let product = MonomialProductTable::new(self.rank, layout.degree());
            let mut output =
                OpenLoopPolynomial::zeros(ROWS, OUT, self.rank + layout.degree(), LANES, &zero);
            apply_segment(&input, &layout, &coefficients, &product, &mut output).unwrap();

            let (mut dressed, mut expected) = (Vec::new(), Vec::new());
            for (lane, momentum) in self.momenta.iter().enumerate() {
                let q: [Complex<R>; 4] = std::array::from_fn(|axis| complex((momentum[axis], 0)));
                for row in 0..ROWS {
                    for column in 0..OUT {
                        dressed.push(output.evaluate(row, column, lane, &q));
                        let mut sum = zero.clone();
                        for (index, entry) in layout.entries().iter().enumerate() {
                            if entry.out != column {
                                continue;
                            }
                            let monomial = (0..4).fold(q[0].one(), |product, axis| {
                                product * &q[axis].pow(u64::from(entry.exponents[axis]))
                            });
                            sum += input.evaluate(row, entry.loop_in, lane, &q)
                                * &(coefficients[index * LANES + lane].clone() * &monomial);
                        }
                        expected.push(sum);
                    }
                }
            }
            (dressed, expected)
        }
    }

    const CASES: [(usize, usize); 6] = [(0, 0), (0, 1), (1, 1), (3, 1), (2, 2), (5, 1)];

    #[test]
    fn exact_dressing_equals_product_at_random_momenta() {
        for (seed, (rank, degree)) in CASES.into_iter().enumerate() {
            let problem = Problem::new(rank, degree, seed as u64 + 1);
            let (dressed, expected) = problem.dress(Rational::from);
            assert_eq!(dressed, expected, "rank {rank} degree {degree}");
            assert!(dressed.iter().any(|value| !value.is_fully_zero()));
        }
    }

    #[test]
    fn every_carrier_reproduces_the_exact_dressing() {
        for (seed, (rank, degree)) in CASES.into_iter().enumerate() {
            let problem = Problem::new(rank, degree, seed as u64 + 1);
            let (exact, _) = problem.dress(Rational::from);
            let exact: Vec<(f64, f64)> = exact
                .iter()
                .map(|value| (value.re.to_f64(), value.im.to_f64()))
                .collect();
            let (double, _) = problem.dress(|value| value as f64);
            let (double_float, _) = problem.dress(|value| DoubleFloat::from(value as f64));
            let (arbitrary, _) = problem.dress(|value| Float::with_val(200, value as f64));
            for (index, &(re, im)) in exact.iter().enumerate() {
                assert_eq!((double[index].re, double[index].im), (re, im));
                assert_eq!(double_float[index], Complex::new(re.into(), im.into()));
                assert_eq!(
                    arbitrary[index],
                    Complex::new(Float::with_val(200, re), Float::with_val(200, im))
                );
            }
        }
    }

    #[test]
    fn segment_layout_rejects_inconsistent_entries() {
        let entry = SegmentEntry {
            loop_in: 0,
            out: 0,
            exponents: [1, 0, 0, 0],
        };
        assert!(SegmentLayout::new(1, 1, vec![]).is_err());
        assert!(SegmentLayout::new(1, 1, vec![entry, entry]).is_err());
        assert!(SegmentLayout::new(1, 0, vec![entry]).is_err());
        let layout = SegmentLayout::new(1, 1, vec![entry]).unwrap();
        assert_eq!(layout.degree(), 1);
        let zero = Complex::new(0.0, 0.0);
        let input = OpenLoopPolynomial::zeros(1, 1, 2, 16, &zero);
        let mut output = OpenLoopPolynomial::zeros(1, 1, 3, 16, &zero);
        let coefficients = vec![zero; 16];
        let wrong = MonomialProductTable::new(1, 1);
        assert!(apply_segment(&input, &layout, &coefficients, &wrong, &mut output).is_err());
        let right = MonomialProductTable::new(2, 1);
        assert!(apply_segment(&input, &layout, &coefficients, &right, &mut output).is_ok());
    }
}
