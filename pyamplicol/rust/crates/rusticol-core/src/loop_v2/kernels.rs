// SPDX-License-Identifier: 0BSD

//! Loop-v2 segment kernels (design §3.7, §5.2): one dressing step of an
//! open-loop object, generic over the real carrier and the SIMD lane count.
//!
//! An open-loop object `O[row][alpha][beta][m]` holds, per helicity row, cut
//! index `alpha` and open-end index `beta`, the raw coefficients `m` of a
//! polynomial in the loop momentum `q` in NinjaRS's tensor layout
//! (`ninjars::fast::tensor_layout`: graded, `ncoeff(r)` coefficients for rank
//! `r`, no symmetry factors). A segment multiplies it by a polynomial of
//! degree `delta` in `q`; [`MonomialProducts`] is the generic raise table for
//! that product, built once from NinjaRS's own monomial ordering, so no
//! kernel is specialised to a rank.
//!
//! Three kernel forms share one storage convention:
//!
//! - **fused** ([`fused_edge`]): `O'[h][a][b'] += O[in][a][b] * X[block][n][b][b']`,
//!   where `X(J)` is the segment already contracted with the block current
//!   row (vertex times outgoing propagator numerator);
//! - **register-blocked fused** ([`fused_edge_blocked`]): the same product
//!   for a dense `X` with compile-time end dimensions, accumulating each
//!   output coefficient in registers;
//! - **split** ([`split_edge`] then [`split_vertex`]): first
//!   `T[h][a][b][nu] += O[in][a][b] * J[block][nu]` over all edges of a
//!   vertex group, then one application of the block-independent vertex
//!   tensor `V[n][b][nu][b']` per output row.
//!
//! [`FusedPattern`] and [`VertexPattern`] list the structurally non-zero
//! entries of `X` and `V`: their `dense` constructors give the dense forms,
//! and sparse patterns from the kernel catalogue give the structured forms.
//!
//! Every complex number is a *lane vector* of `2 * L` reals: the `L` real
//! parts of the batch lanes followed by their `L` imaginary parts, so lanes
//! are the innermost index (structure of arrays). All kernels accumulate into
//! their output; the carrier `T` is any Numerica [`FloatLike`] type (`f64`,
//! `DoubleFloat`, `Float`, or `Rational` for exact oracles).

use ninjars::fast::tensor_layout::{monomial_index, monomials, ncoeff};
use symbolica::domains::float::FloatLike;

/// Number of reals in one complex lane vector of `lanes` lanes.
pub const fn lane_vector_len(lanes: usize) -> usize {
    2 * lanes
}

/// Product table of loop-momentum monomials in NinjaRS's tensor layout.
///
/// `target(m, n)` is the index of `monomial_m * monomial_n` for every
/// coefficient `m` of a rank-`rank` polynomial and every coefficient `n` of a
/// degree-`degree` factor. Because the layout is graded, the index is the
/// same in every output tensor of rank at least `rank + degree`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MonomialProducts {
    rank: usize,
    degree: usize,
    /// Column-major: `targets[n * ncoeff(rank) + m]`.
    targets: Vec<u32>,
}

impl MonomialProducts {
    pub fn new(rank: usize, degree: usize) -> Self {
        let lhs = monomials(rank);
        let rhs = monomials(degree);
        let targets = rhs
            .iter()
            .flat_map(|n| {
                lhs.iter().map(move |m| {
                    let exponents = std::array::from_fn(|k| m[k] + n[k]);
                    u32::try_from(monomial_index(exponents))
                        .expect("monomial index exceeds the u32 product table")
                })
            })
            .collect();
        Self {
            rank,
            degree,
            targets,
        }
    }

    /// Rank of the input polynomial.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Degree of the segment factor in `q`.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Coefficients of the input polynomial, `ncoeff(rank)`.
    pub fn input_len(&self) -> usize {
        ncoeff(self.rank)
    }

    /// Coefficients of the factor, `ncoeff(degree)`.
    pub fn factor_len(&self) -> usize {
        ncoeff(self.degree)
    }

    /// Output index of every input coefficient times factor monomial `n`.
    #[inline]
    pub fn column(&self, n: usize) -> &[u32] {
        let len = self.input_len();
        &self.targets[n * len..(n + 1) * len]
    }
}

/// Dimensions of one dressing step.
///
/// `cut` is the dimension of the cut line (4 for vector and Dirac lines, 1
/// for scalars and ghosts), `incoming` and `outgoing` those of the open end
/// before and after the segment, and `current` that of the attached block
/// current. Rows of the input object have `cut * incoming * ncoeff(rank)`
/// lane vectors, rows of the output `cut * outgoing * ncoeff(output_rank)`.
#[derive(Clone, Copy, Debug)]
pub struct Segment<'a> {
    pub cut: usize,
    pub incoming: usize,
    pub current: usize,
    pub outgoing: usize,
    pub products: &'a MonomialProducts,
    pub output_rank: usize,
}

impl<'a> Segment<'a> {
    pub fn new(
        cut: usize,
        incoming: usize,
        current: usize,
        outgoing: usize,
        products: &'a MonomialProducts,
        output_rank: usize,
    ) -> Self {
        assert!(
            output_rank >= products.rank() + products.degree(),
            "output rank {output_rank} below input rank {} plus segment degree {}",
            products.rank(),
            products.degree()
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

    /// Lane vectors per input row.
    pub fn input_row_len(&self) -> usize {
        self.cut * self.incoming * self.products.input_len()
    }

    /// Lane vectors per output row.
    pub fn output_row_len(&self) -> usize {
        self.cut * self.outgoing * ncoeff(self.output_rank)
    }

    /// Lane vectors per row of the split-form accumulator `T[a][b][nu][m]`.
    pub fn accumulator_row_len(&self) -> usize {
        self.cut * self.incoming * self.current * self.products.input_len()
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

/// Structurally non-zero entry `X[monomial][incoming][outgoing]` of a fused factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusedEntry {
    pub monomial: u32,
    pub incoming: u32,
    pub outgoing: u32,
}

/// Entries of a fused factor; a factor row stores one lane vector per entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FusedPattern {
    entries: Vec<FusedEntry>,
}

impl FusedPattern {
    pub fn new(entries: Vec<FusedEntry>) -> Self {
        Self { entries }
    }

    /// Every entry in `[monomial][incoming][outgoing]` order, the layout
    /// [`fused_edge_blocked`] reads.
    pub fn dense(segment: &Segment<'_>) -> Self {
        let mut entries = Vec::new();
        for monomial in 0..segment.products.factor_len() {
            for incoming in 0..segment.incoming {
                for outgoing in 0..segment.outgoing {
                    entries.push(FusedEntry {
                        monomial: monomial as u32,
                        incoming: incoming as u32,
                        outgoing: outgoing as u32,
                    });
                }
            }
        }
        Self { entries }
    }

    pub fn entries(&self) -> &[FusedEntry] {
        &self.entries
    }
}

/// Structurally non-zero entry `V[monomial][incoming][current][outgoing]` of a vertex tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexEntry {
    pub monomial: u32,
    pub incoming: u32,
    pub current: u32,
    pub outgoing: u32,
}

/// Entries of a split-form vertex tensor; one lane vector per entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VertexPattern {
    entries: Vec<VertexEntry>,
}

impl VertexPattern {
    pub fn new(entries: Vec<VertexEntry>) -> Self {
        Self { entries }
    }

    /// Every entry in `[monomial][incoming][current][outgoing]` order.
    pub fn dense(segment: &Segment<'_>) -> Self {
        let mut entries = Vec::new();
        for monomial in 0..segment.products.factor_len() {
            for incoming in 0..segment.incoming {
                for current in 0..segment.current {
                    for outgoing in 0..segment.outgoing {
                        entries.push(VertexEntry {
                            monomial: monomial as u32,
                            incoming: incoming as u32,
                            current: current as u32,
                            outgoing: outgoing as u32,
                        });
                    }
                }
            }
        }
        Self { entries }
    }

    pub fn entries(&self) -> &[VertexEntry] {
        &self.entries
    }
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

/// Fused segment for one edge:
/// `output[h][a][b'][m*n] += input[in][a][b][m] * factor[block][entry(n, b, b')]`
/// for every link `(h, in, block)`. Each factor row holds one lane vector per
/// pattern entry.
pub fn fused_edge<T: FloatLike, const L: usize>(
    segment: &Segment<'_>,
    pattern: &FusedPattern,
    input: &[T],
    factor: &[T],
    links: &[RowLink],
    output: &mut [T],
) {
    let v = lane_vector_len(L);
    let m_in = segment.products.input_len();
    let m_out = ncoeff(segment.output_rank);
    let entries = pattern.entries();
    let factor_row = entries.len() * v;
    for link in links {
        let x_row = &factor[link.block as usize * factor_row..][..factor_row];
        for a in 0..segment.cut {
            let src_base = (link.input as usize * segment.cut + a) * segment.incoming;
            let dst_base = (link.output as usize * segment.cut + a) * segment.outgoing;
            for (entry, x) in entries.iter().zip(x_row.chunks_exact(v)) {
                let src = &input[(src_base + entry.incoming as usize) * m_in * v..][..m_in * v];
                let dst =
                    &mut output[(dst_base + entry.outgoing as usize) * m_out * v..][..m_out * v];
                let targets = segment.products.column(entry.monomial as usize);
                for (o, &t) in src.chunks_exact(v).zip(targets) {
                    cmac::<T, L>(&mut dst[t as usize * v..], o, x);
                }
            }
        }
    }
}

/// Register-blocked fused segment for a dense factor with `B` incoming and
/// `BP` outgoing components, stored as `factor[block][n][b][b']` (the order
/// of [`FusedPattern::dense`]). Per input coefficient the `B` components are
/// loaded once and the `BP` products of every factor monomial are summed in
/// registers before they are written. `segment.incoming` and
/// `segment.outgoing` must equal `B` and `BP`.
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
    let m_in = segment.products.input_len();
    let m_out = ncoeff(segment.output_rank);
    let n_factor = segment.products.factor_len();
    let factor_row = n_factor * B * BP * v;
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
                    let t = segment.products.column(n)[m] as usize;
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
    let m_in = segment.products.input_len();
    let current_row = segment.current * v;
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
/// `output[h][a][b'][m*n] += accumulator[h][a][b][nu][m] * vertex[entry(n, b, nu, b')]`.
/// `vertex` holds one lane vector per pattern entry.
pub fn split_vertex<T: FloatLike, const L: usize>(
    segment: &Segment<'_>,
    pattern: &VertexPattern,
    vertex: &[T],
    rows: usize,
    accumulator: &[T],
    output: &mut [T],
) {
    let v = lane_vector_len(L);
    let m_in = segment.products.input_len();
    let m_out = ncoeff(segment.output_rank);
    for h in 0..rows {
        for a in 0..segment.cut {
            let src_base = (h * segment.cut + a) * segment.incoming;
            let dst_base = (h * segment.cut + a) * segment.outgoing;
            for (entry, y) in pattern.entries().iter().zip(vertex.chunks_exact(v)) {
                let src_row =
                    (src_base + entry.incoming as usize) * segment.current + entry.current as usize;
                let src = &accumulator[src_row * m_in * v..][..m_in * v];
                let dst =
                    &mut output[(dst_base + entry.outgoing as usize) * m_out * v..][..m_out * v];
                let targets = segment.products.column(entry.monomial as usize);
                for (t, &target) in src.chunks_exact(v).zip(targets) {
                    cmac::<T, L>(&mut dst[target as usize * v..], t, y);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "kernels_tests.rs"]
mod tests;
