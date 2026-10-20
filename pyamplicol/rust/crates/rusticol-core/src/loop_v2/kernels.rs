// SPDX-License-Identifier: 0BSD

//! Batched trie-node segment kernels (DESIGN.md §3.7, §5.2, §5.4): one
//! dressing step of an open-loop object, generic over the real carrier and
//! the SIMD lane count.
//!
//! The algebra is that of [`super::poly`]: an object holds, per row and
//! open-end column, the raw coefficients of a polynomial in `q` in NinjaRS's
//! tensor layout, a segment is a [`SegmentLayout`] of sparse entries
//! `X[loop_in][out] += c q^a`, and every product of monomials is looked up in
//! one [`MonomialProductTable`], so no kernel is specialised to a rank. What
//! these kernels add for the executor is the trie node:
//!
//! - **row links**: an edge contracts input row `link.input` of its child
//!   object with row `link.block` of its block (current or fused factor) into
//!   output row `link.output`, so helicity rows are gathered per edge;
//! - **cut rows**: every link applies the segment to the `cut` rows of the
//!   cut-line index (4 for vector and Dirac lines, 1 for scalars and ghosts).
//!   An object row of the executor is therefore `cut` consecutive rows of an
//!   [`super::poly::OpenLoopPolynomial`], in the same
//!   `[row][column][monomial][lane]` order;
//! - **lane blocks**: lanes are the innermost index as in `poly.rs`, but one
//!   complex coefficient over `L` lanes is stored as `L` real parts followed
//!   by `L` imaginary parts (a *lane vector*, [`lane_vector_len`]), so the
//!   complex multiply-add vectorises without shuffles. Gate G0 measured this
//!   form; `benches/loop_v2_kernels.rs` also times `poly::apply_segment` on
//!   `Complex` lanes for comparison.
//!
//! Three kernel forms compute the same node:
//!
//! - **fused** ([`fused_edge`]): `O'[h] += O[in] · X(J)[block]`, where the
//!   segment is already contracted with the block current row (vertex times
//!   outgoing propagator numerator);
//! - **register-blocked fused** ([`fused_edge_blocked`]): the same product
//!   for a dense `X` with compile-time end dimensions, accumulating each
//!   output coefficient in registers;
//! - **split** ([`split_edge`] then [`split_vertex`]): first
//!   `T[h][b][nu] += O[in][b] · J[block][nu]` over all edges of a vertex
//!   group, then one application of the block-independent vertex
//!   `V[(b, nu)][b']` per output row. The vertex is a segment layout whose
//!   loop-in column is `b * current + nu` ([`Segment::vertex_column`]).
//!
//! Dense layouts ([`dense_layout`]) give the dense forms and sparse layouts
//! from the kernel catalogue the structured forms. All kernels accumulate into
//! their output; the carrier `T` is any [`FloatLike`] real (`f64`,
//! `DoubleFloat`, `Float`, or `Rational` for exact oracles).

use super::poly::{MonomialProductTable, SegmentEntry, SegmentLayout};
use ninjars::fast::tensor_layout::{monomials, ncoeff};
use symbolica::domains::float::FloatLike;

/// Number of reals in one complex lane vector of `lanes` lanes.
pub const fn lane_vector_len(lanes: usize) -> usize {
    2 * lanes
}

/// Every entry of a `loop_in x out` segment of degree `degree`, in
/// `[monomial][loop_in][out]` order: the layout [`fused_edge_blocked`] reads.
pub fn dense_layout(loop_in: usize, out: usize, degree: usize) -> SegmentLayout {
    let entries = monomials(degree)
        .into_iter()
        .flat_map(|exponents| {
            (0..loop_in).flat_map(move |loop_in| {
                (0..out).map(move |out| SegmentEntry {
                    loop_in,
                    out,
                    exponents,
                })
            })
        })
        .collect();
    SegmentLayout::new(loop_in, out, entries).expect("a dense layout is consistent")
}

/// Dimensions of one dressing step.
///
/// `cut` is the dimension of the cut line, `incoming` and `outgoing` those of
/// the open end before and after the segment, and `current` that of the
/// attached block current. Rows of the input object have
/// `cut * incoming * ncoeff(rank)` lane vectors, rows of the output
/// `cut * outgoing * ncoeff(output_rank)`; the graded layout lets the output
/// rank exceed `rank + degree`.
#[derive(Clone, Copy, Debug)]
pub struct Segment<'a> {
    cut: usize,
    incoming: usize,
    current: usize,
    outgoing: usize,
    products: &'a MonomialProductTable,
    output_rank: usize,
}

impl<'a> Segment<'a> {
    /// `products` is the table of `(input rank, segment degree)`.
    pub fn new(
        cut: usize,
        incoming: usize,
        current: usize,
        outgoing: usize,
        products: &'a MonomialProductTable,
        output_rank: usize,
    ) -> Self {
        assert!(
            output_rank >= products.output_rank(),
            "output rank {output_rank} below input rank {} plus segment degree {}",
            products.left_rank(),
            products.right_rank()
        );
        Self {
            cut,
            incoming,
            current,
            outgoing,
            products,
            output_rank,
        }
    }

    pub fn cut(&self) -> usize {
        self.cut
    }

    pub fn incoming(&self) -> usize {
        self.incoming
    }

    pub fn current(&self) -> usize {
        self.current
    }

    pub fn outgoing(&self) -> usize {
        self.outgoing
    }

    pub fn products(&self) -> &'a MonomialProductTable {
        self.products
    }

    pub fn rank(&self) -> usize {
        self.products.left_rank()
    }

    pub fn degree(&self) -> usize {
        self.products.right_rank()
    }

    pub fn output_rank(&self) -> usize {
        self.output_rank
    }

    /// Lane vectors per input row.
    pub fn input_row_len(&self) -> usize {
        self.cut * self.incoming * ncoeff(self.rank())
    }

    /// Lane vectors per output row.
    pub fn output_row_len(&self) -> usize {
        self.cut * self.outgoing * ncoeff(self.output_rank)
    }

    /// Lane vectors per row of the split-form accumulator `T[a][b][nu][m]`.
    pub fn accumulator_row_len(&self) -> usize {
        self.cut * self.incoming * self.current * ncoeff(self.rank())
    }

    /// Loop-in column of the split-form vertex for open end `incoming` and
    /// current component `current`.
    pub fn vertex_column(&self, incoming: usize, current: usize) -> usize {
        incoming * self.current + current
    }

    /// The dense fused factor `X[n][b][b']`.
    pub fn dense_fused_layout(&self) -> SegmentLayout {
        dense_layout(self.incoming, self.outgoing, self.degree())
    }

    /// The dense split-form vertex `V[n][(b, nu)][b']`.
    pub fn dense_vertex_layout(&self) -> SegmentLayout {
        dense_layout(self.incoming * self.current, self.outgoing, self.degree())
    }

    fn check_layout(&self, layout: &SegmentLayout, loop_in: usize) {
        assert!(
            layout.loop_in_dimension() == loop_in
                && layout.out_dimension() == self.outgoing
                && layout.degree() <= self.degree(),
            "a {}x{} layout of degree {} does not fit a {loop_in}x{} segment of degree {}",
            layout.loop_in_dimension(),
            layout.out_dimension(),
            layout.degree(),
            self.outgoing,
            self.degree()
        );
    }
}

/// One helicity contraction of an edge: output row `output` receives input
/// row `input` of the edge's child object times row `block` of its block
/// current (or of its fused factor).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowLink {
    pub output: u32,
    pub input: u32,
    pub block: u32,
}

/// `acc += x * y` on complex lane vectors.
#[inline(always)]
fn cmac<T: FloatLike, const L: usize>(acc: &mut [T], x: &[T], y: &[T]) {
    let (acc_re, acc_im) = acc[..2 * L].split_at_mut(L);
    let (x_re, x_im) = x[..2 * L].split_at(L);
    let (y_re, y_im) = y[..2 * L].split_at(L);
    for l in 0..L {
        acc_re[l] += x_re[l].clone() * &y_re[l] - x_im[l].clone() * &y_im[l];
        acc_im[l] += x_re[l].clone() * &y_im[l] + x_im[l].clone() * &y_re[l];
    }
}

/// Sets every value to zero, keeping each value's precision.
pub fn clear<T: FloatLike>(values: &mut [T]) {
    for value in values {
        *value = value.zero();
    }
}

/// `output[a][out][m*n] += input[a][loop_in][m] * x[entry]` for every cut
/// index `a` and layout entry, on one input row (`loop_in` columns) and one
/// output row; `x` holds one lane vector per entry.
#[inline(always)]
fn apply_row<T: FloatLike, const L: usize>(
    segment: &Segment<'_>,
    layout: &SegmentLayout,
    loop_in: usize,
    input: &[T],
    x: &[T],
    output: &mut [T],
) {
    let v = lane_vector_len(L);
    let m_in = ncoeff(segment.rank());
    let m_out = ncoeff(segment.output_rank);
    let products = segment.products;
    for a in 0..segment.cut {
        for ((entry, &n), y) in layout
            .entries()
            .iter()
            .zip(layout.monomials())
            .zip(x.chunks_exact(v))
        {
            let src = &input[(a * loop_in + entry.loop_in) * m_in * v..][..m_in * v];
            let dst = &mut output[(a * segment.outgoing + entry.out) * m_out * v..][..m_out * v];
            for (m, o) in src.chunks_exact(v).enumerate() {
                cmac::<T, L>(&mut dst[products.target(m, n) * v..], o, y);
            }
        }
    }
}

/// Fused segment for one edge:
/// `output[h][a][b'][m*n] += input[in][a][b][m] * factor[block][entry(n, b, b')]`
/// for every link `(h, in, block)`. Each factor row holds one lane vector per
/// entry of `layout` (`incoming x outgoing`).
pub fn fused_edge<T: FloatLike, const L: usize>(
    segment: &Segment<'_>,
    layout: &SegmentLayout,
    input: &[T],
    factor: &[T],
    links: &[RowLink],
    output: &mut [T],
) {
    segment.check_layout(layout, segment.incoming);
    let v = lane_vector_len(L);
    let input_row = segment.input_row_len() * v;
    let output_row = segment.output_row_len() * v;
    let factor_row = layout.entries().len() * v;
    debug_assert_eq!(input.len() % input_row, 0, "input is not whole rows");
    debug_assert_eq!(factor.len() % factor_row, 0, "factor is not whole rows");
    debug_assert_eq!(output.len() % output_row, 0, "output is not whole rows");
    for link in links {
        apply_row::<T, L>(
            segment,
            layout,
            segment.incoming,
            &input[link.input as usize * input_row..][..input_row],
            &factor[link.block as usize * factor_row..][..factor_row],
            &mut output[link.output as usize * output_row..][..output_row],
        );
    }
}

/// Register-blocked fused segment for a dense factor with `B` incoming and
/// `BP` outgoing components, stored as `factor[block][n][b][b']` (the order
/// of [`dense_layout`]). Per input coefficient the `B` components are loaded
/// once and the `BP` products of every factor monomial are summed in
/// registers before they are written. `segment.incoming()` and
/// `segment.outgoing()` must equal `B` and `BP`.
pub fn fused_edge_blocked<T: FloatLike, const L: usize, const B: usize, const BP: usize>(
    segment: &Segment<'_>,
    input: &[T],
    factor: &[T],
    links: &[RowLink],
    output: &mut [T],
) {
    assert!(
        segment.incoming == B && segment.outgoing == BP,
        "blocked kernel instantiated for {B}x{BP}, segment is {}x{}",
        segment.incoming,
        segment.outgoing
    );
    let Some(zero) = input.first().map(FloatLike::zero) else {
        return;
    };
    let v = lane_vector_len(L);
    let m_in = ncoeff(segment.rank());
    let m_out = ncoeff(segment.output_rank);
    let n_factor = ncoeff(segment.degree());
    let factor_row = n_factor * B * BP * v;
    debug_assert_eq!(input.len() % (segment.input_row_len() * v), 0);
    debug_assert_eq!(factor.len() % factor_row, 0, "factor is not whole rows");
    debug_assert_eq!(output.len() % (segment.output_row_len() * v), 0);
    let products = segment.products;
    let lanes = || std::array::from_fn::<T, L, _>(|_| zero.clone());
    let mut o_re: [[T; L]; B] = std::array::from_fn(|_| lanes());
    let mut o_im: [[T; L]; B] = std::array::from_fn(|_| lanes());
    let mut acc_re: [[T; L]; BP] = std::array::from_fn(|_| lanes());
    let mut acc_im: [[T; L]; BP] = std::array::from_fn(|_| lanes());
    for link in links {
        let x_row = &factor[link.block as usize * factor_row..][..factor_row];
        for a in 0..segment.cut {
            let src_base = (link.input as usize * segment.cut + a) * B * m_in;
            let dst_base = (link.output as usize * segment.cut + a) * BP * m_out;
            for m in 0..m_in {
                for b in 0..B {
                    let o = &input[(src_base + b * m_in + m) * v..][..v];
                    o_re[b].clone_from_slice(&o[..L]);
                    o_im[b].clone_from_slice(&o[L..]);
                }
                for n in 0..n_factor {
                    for bp in 0..BP {
                        for l in 0..L {
                            acc_re[bp][l] = zero.clone();
                            acc_im[bp][l] = zero.clone();
                        }
                    }
                    for b in 0..B {
                        for bp in 0..BP {
                            let x = &x_row[((n * B + b) * BP + bp) * v..][..v];
                            let (x_re, x_im) = x.split_at(L);
                            for l in 0..L {
                                acc_re[bp][l] +=
                                    o_re[b][l].clone() * &x_re[l] - o_im[b][l].clone() * &x_im[l];
                                acc_im[bp][l] +=
                                    o_re[b][l].clone() * &x_im[l] + o_im[b][l].clone() * &x_re[l];
                            }
                        }
                    }
                    let t = products.target(m, n);
                    for bp in 0..BP {
                        let dst = &mut output[(dst_base + bp * m_out + t) * v..][..v];
                        let (d_re, d_im) = dst.split_at_mut(L);
                        for l in 0..L {
                            d_re[l] += &acc_re[bp][l];
                            d_im[l] += &acc_im[bp][l];
                        }
                    }
                }
            }
        }
    }
}

/// First half of the split form for one edge:
/// `accumulator[h][a][b][nu][m] += input[in][a][b][m] * current[block][nu]`.
/// The accumulator rows are those of the node's output rows; it is shared by
/// all edges of one vertex group and cleared by the caller.
pub fn split_edge<T: FloatLike, const L: usize>(
    segment: &Segment<'_>,
    input: &[T],
    current: &[T],
    links: &[RowLink],
    accumulator: &mut [T],
) {
    let v = lane_vector_len(L);
    let m_in = ncoeff(segment.rank());
    let current_row = segment.current * v;
    debug_assert_eq!(input.len() % (segment.input_row_len() * v), 0);
    debug_assert_eq!(current.len() % current_row, 0, "current is not whole rows");
    debug_assert_eq!(accumulator.len() % (segment.accumulator_row_len() * v), 0);
    for link in links {
        let j_row = &current[link.block as usize * current_row..][..current_row];
        for ab in 0..segment.cut * segment.incoming {
            let src = &input
                [(link.input as usize * segment.cut * segment.incoming + ab) * m_in * v..]
                [..m_in * v];
            let dst_base =
                (link.output as usize * segment.cut * segment.incoming + ab) * segment.current;
            for (nu, j) in j_row.chunks_exact(v).enumerate() {
                let dst = &mut accumulator[(dst_base + nu) * m_in * v..][..m_in * v];
                for (d, o) in dst.chunks_exact_mut(v).zip(src.chunks_exact(v)) {
                    cmac::<T, L>(d, o, j);
                }
            }
        }
    }
}

/// Second half of the split form: for the first `rows` output rows,
/// `output[h][a][b'][m*n] += accumulator[h][a][(b, nu)][m] * vertex[entry(n, (b, nu), b')]`.
/// `layout` is `(incoming * current) x outgoing` with loop-in columns from
/// [`Segment::vertex_column`]; `vertex` holds one lane vector per entry.
pub fn split_vertex<T: FloatLike, const L: usize>(
    segment: &Segment<'_>,
    layout: &SegmentLayout,
    vertex: &[T],
    rows: usize,
    accumulator: &[T],
    output: &mut [T],
) {
    let loop_in = segment.incoming * segment.current;
    segment.check_layout(layout, loop_in);
    let v = lane_vector_len(L);
    let accumulator_row = segment.accumulator_row_len() * v;
    let output_row = segment.output_row_len() * v;
    debug_assert_eq!(vertex.len(), layout.entries().len() * v);
    debug_assert_eq!(accumulator.len() % accumulator_row, 0);
    debug_assert_eq!(output.len() % output_row, 0, "output is not whole rows");
    debug_assert!(accumulator.len() >= rows * accumulator_row && output.len() >= rows * output_row);
    for (source, target) in accumulator
        .chunks_exact(accumulator_row)
        .zip(output.chunks_exact_mut(output_row))
        .take(rows)
    {
        apply_row::<T, L>(segment, layout, loop_in, source, vertex, target);
    }
}

#[cfg(test)]
#[path = "kernels_tests.rs"]
mod tests;
