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
//!   outgoing propagator numerator); the executor runs it in the
//!   output-stationary order of a [`GatherSchedule`] ([`fused_edge_gathered`],
//!   bitwise the same values) wherever an output coefficient receives more
//!   than one term;
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
//!
//! **Vector width** (WP4.6). The builds target baseline x86-64, so the
//! binary64 kernels ([`fused_edge`], [`split_edge`], [`split_vertex`]) are
//! also compiled for AVX-512 and AVX2 and run in the widest width the running
//! CPU reports ([`Simd::detect`]; never a build-host feature). The wider forms
//! perform the same operations per lane in the same order (no fused
//! multiply-add, no reassociation), so their values are bitwise the
//! baseline's; only the speed depends on the CPU.

use super::poly::{MonomialProductTable, SegmentEntry, SegmentLayout};
use ninjars::fast::tensor_layout::{monomials, ncoeff};
use symbolica::domains::float::FloatLike;

/// Number of reals in one complex lane vector of `lanes` lanes.
pub const fn lane_vector_len(lanes: usize) -> usize {
    2 * lanes
}

/// The vector width the binary64 kernels run in (dressing, seed sums,
/// interference z sums, root closures): binary64 lanes take the widest the
/// running CPU reports, every other carrier the baseline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Simd {
    Baseline,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

/// Caps the kernels' vector width ([`Simd::detect`]) at `baseline` or
/// `avx2`, so that an AVX-512 host also runs the code paths of an AVX2-only
/// or a baseline x86-64 host (the bitwise check of
/// `tests/unit/test_loop_v2_native_trees.py`); any other value, or none,
/// caps nothing. It never enables a feature the CPU lacks.
pub const KERNEL_SIMD_ENV: &str = "RUSTICOL_TEST_KERNEL_SIMD";

impl Simd {
    /// The width for carrier `T` (read once per process).
    #[inline]
    pub fn detect<T: 'static>() -> Self {
        #[cfg(target_arch = "x86_64")]
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            static WIDTH: std::sync::OnceLock<Simd> = std::sync::OnceLock::new();
            return *WIDTH.get_or_init(|| {
                let cap = std::env::var(KERNEL_SIMD_ENV).ok();
                Self::widest(cap.as_deref())
            });
        }
        Self::Baseline
    }

    /// The widest width the CPU reports within the cap `value`.
    #[cfg(target_arch = "x86_64")]
    fn widest(value: Option<&str>) -> Self {
        let cap = Self::cap(value);
        if cap == Self::Avx512 && std::arch::is_x86_feature_detected!("avx512f") {
            return Self::Avx512;
        }
        if cap != Self::Baseline && std::arch::is_x86_feature_detected!("avx2") {
            return Self::Avx2;
        }
        Self::Baseline
    }

    /// The widest width [`KERNEL_SIMD_ENV`]'s value allows.
    #[cfg(target_arch = "x86_64")]
    fn cap(value: Option<&str>) -> Self {
        match value {
            Some("baseline") => Self::Baseline,
            Some("avx2") => Self::Avx2,
            _ => Self::Avx512,
        }
    }
}

/// Defines `$name` as `$body` (an `#[inline(always)]` kernel generic over
/// the carrier `T: $bound` and optionally a const `L`) dispatched on
/// [`Simd::detect`]: the same body compiled for AVX-512, AVX2 and the
/// baseline (the dressing kernels here; the seed sums and the
/// interference z sums).
macro_rules! dispatched {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident<T: $bound:path $(, const $c:ident: usize)?>(
            $($arg:ident: $ty:ty),* $(,)?
        ) => $body:ident;
    ) => {
        $(#[$meta])*
        $vis fn $name<T: $bound + 'static $(, const $c: usize)?>($($arg: $ty),*) {
            #[cfg(target_arch = "x86_64")]
            {
                #[allow(clippy::too_many_arguments)]
                #[target_feature(enable = "avx512f")]
                fn avx512<T: $bound $(, const $c: usize)?>($($arg: $ty),*) {
                    $body::<T $(, $c)?>($($arg),*)
                }
                #[allow(clippy::too_many_arguments)]
                #[target_feature(enable = "avx2")]
                fn avx2<T: $bound $(, const $c: usize)?>($($arg: $ty),*) {
                    $body::<T $(, $c)?>($($arg),*)
                }
                use $crate::loop_v2::kernels::Simd;
                match Simd::detect::<T>() {
                    // SAFETY: `Simd::detect` returns a width only when the
                    // running CPU reports its feature.
                    Simd::Avx512 => return unsafe { avx512::<T $(, $c)?>($($arg),*) },
                    Simd::Avx2 => return unsafe { avx2::<T $(, $c)?>($($arg),*) },
                    Simd::Baseline => {}
                }
            }
            $body::<T $(, $c)?>($($arg),*)
        }
    };
}
pub(crate) use dispatched;

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

dispatched! {
    /// Fused segment for one edge:
    /// `output[h][a][b'][m*n] += input[in][a][b][m] * factor[block][entry(n, b, b')]`
    /// for every link `(h, in, block)`. Each factor row holds one lane vector
    /// per entry of `layout` (`incoming x outgoing`).
    pub fn fused_edge<T: FloatLike, const L: usize>(
        segment: &Segment<'_>,
        layout: &SegmentLayout,
        input: &[T],
        factor: &[T],
        links: &[RowLink],
        output: &mut [T],
    ) => fused_edge_body;
}

#[inline(always)]
fn fused_edge_body<T: FloatLike, const L: usize>(
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

/// The output-stationary order of a fused segment (lit-perf): per output
/// coefficient `(out, t)` of one cut index, the terms
/// `input[loop_in][m] · factor[entry]` with `products.target(m, n) = t`, in
/// layout entry order. [`fused_edge_gathered`] adds them to the output
/// coefficient in a register accumulator and stores it once, where
/// [`fused_edge`] reads and writes the output coefficient once per term.
/// Each output coefficient receives the same products in the same order (an
/// entry reaches a coefficient through at most one input monomial), so both
/// forms give bitwise the same values in every carrier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GatherSchedule {
    /// Open-end columns of the input and the output (the layout's).
    loop_in: usize,
    out: usize,
    /// Input coefficients per column (`ncoeff(rank)`).
    sources: usize,
    /// Output coefficients per column (`ncoeff(rank + degree)`).
    targets: usize,
    /// Factor entries per factor row.
    entries: usize,
    /// Per output coefficient `out * targets + t`, the end (exclusive) of its
    /// terms in `terms`; its start is the previous end.
    ends: Vec<u32>,
    /// `(input lane vector within one cut index, factor entry)` per term.
    terms: Vec<[u32; 2]>,
}

impl GatherSchedule {
    /// The schedule of `layout` (`loop_in x out`) on input rank
    /// `products.left_rank()`, or `None` where no output coefficient receives
    /// two terms: [`fused_edge`] then performs the same loads and stores.
    /// Panics where an index exceeds `u32` (as [`RowLink`]'s rows).
    pub fn new(layout: &SegmentLayout, products: &MonomialProductTable) -> Option<Self> {
        assert!(
            layout.degree() <= products.right_rank(),
            "a layout of degree {} on a products table of degree {}",
            layout.degree(),
            products.right_rank()
        );
        let m_in = ncoeff(products.left_rank());
        let targets = ncoeff(products.output_rank());
        let index = |value: usize| u32::try_from(value).expect("a gather index exceeds u32");
        let mut counts = vec![0usize; layout.out_dimension() * targets];
        for (entry, &n) in layout.entries().iter().zip(layout.monomials()) {
            for m in 0..m_in {
                counts[entry.out * targets + products.target(m, n)] += 1;
            }
        }
        if counts.iter().all(|&count| count <= 1) {
            return None;
        }
        let mut ends = Vec::with_capacity(counts.len());
        let mut next = Vec::with_capacity(counts.len());
        let mut total = 0;
        for &count in &counts {
            next.push(total);
            total += count;
            ends.push(index(total));
        }
        let mut terms = vec![[0u32; 2]; total];
        for (e, (entry, &n)) in layout.entries().iter().zip(layout.monomials()).enumerate() {
            for m in 0..m_in {
                let slot = &mut next[entry.out * targets + products.target(m, n)];
                terms[*slot] = [index(entry.loop_in * m_in + m), index(e)];
                *slot += 1;
            }
        }
        Some(Self {
            loop_in: layout.loop_in_dimension(),
            out: layout.out_dimension(),
            sources: m_in,
            targets,
            entries: layout.entries().len(),
            ends,
            terms,
        })
    }
}

dispatched! {
    /// [`fused_edge`] in the output-stationary order of `schedule` (the
    /// schedule of the segment's layout and products table): bitwise the same
    /// values.
    pub fn fused_edge_gathered<T: FloatLike, const L: usize>(
        segment: &Segment<'_>,
        schedule: &GatherSchedule,
        input: &[T],
        factor: &[T],
        links: &[RowLink],
        output: &mut [T],
    ) => fused_edge_gathered_body;
}

#[inline(always)]
fn fused_edge_gathered_body<T: FloatLike, const L: usize>(
    segment: &Segment<'_>,
    schedule: &GatherSchedule,
    input: &[T],
    factor: &[T],
    links: &[RowLink],
    output: &mut [T],
) {
    let m_in = ncoeff(segment.rank());
    let m_out = ncoeff(segment.output_rank);
    assert!(
        schedule.loop_in == segment.incoming
            && schedule.out == segment.outgoing
            && schedule.sources == m_in
            && schedule.targets <= m_out,
        "a gather schedule does not fit its segment"
    );
    let Some(zero) = input.first().map(FloatLike::zero) else {
        return;
    };
    let v = lane_vector_len(L);
    let input_row = segment.input_row_len() * v;
    let output_row = segment.output_row_len() * v;
    let factor_row = schedule.entries * v;
    let cut_in = segment.incoming * m_in * v;
    let cut_out = segment.outgoing * m_out * v;
    debug_assert_eq!(input.len() % input_row, 0, "input is not whole rows");
    debug_assert_eq!(factor.len() % factor_row, 0, "factor is not whole rows");
    debug_assert_eq!(output.len() % output_row, 0, "output is not whole rows");
    let mut acc_re: [T; L] = std::array::from_fn(|_| zero.clone());
    let mut acc_im: [T; L] = std::array::from_fn(|_| zero.clone());
    for link in links {
        let source = &input[link.input as usize * input_row..][..input_row];
        let x = &factor[link.block as usize * factor_row..][..factor_row];
        let target = &mut output[link.output as usize * output_row..][..output_row];
        for a in 0..segment.cut {
            let src = &source[a * cut_in..][..cut_in];
            let dst = &mut target[a * cut_out..][..cut_out];
            let mut start = 0;
            for (column, ends) in dst
                .chunks_exact_mut(m_out * v)
                .zip(schedule.ends.chunks_exact(schedule.targets))
            {
                for (d, &end) in column.chunks_exact_mut(v).zip(ends) {
                    let end = end as usize;
                    if end == start {
                        continue;
                    }
                    let (d_re, d_im) = d.split_at_mut(L);
                    for l in 0..L {
                        acc_re[l].clone_from(&d_re[l]);
                        acc_im[l].clone_from(&d_im[l]);
                    }
                    for &[s, e] in &schedule.terms[start..end] {
                        let (o_re, o_im) = src[s as usize * v..][..v].split_at(L);
                        let (y_re, y_im) = x[e as usize * v..][..v].split_at(L);
                        for l in 0..L {
                            acc_re[l] += o_re[l].clone() * &y_re[l] - o_im[l].clone() * &y_im[l];
                            acc_im[l] += o_re[l].clone() * &y_im[l] + o_im[l].clone() * &y_re[l];
                        }
                    }
                    for l in 0..L {
                        d_re[l].clone_from(&acc_re[l]);
                        d_im[l].clone_from(&acc_im[l]);
                    }
                    start = end;
                }
            }
        }
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

dispatched! {
    /// First half of the split form for one edge:
    /// `accumulator[h][a][b][nu][m] += input[in][a][b][m] * current[block][nu]`.
    /// The accumulator rows are those of the node's output rows; it is shared
    /// by all edges of one vertex group and cleared by the caller.
    pub fn split_edge<T: FloatLike, const L: usize>(
        segment: &Segment<'_>,
        input: &[T],
        current: &[T],
        links: &[RowLink],
        accumulator: &mut [T],
    ) => split_edge_body;
}

#[inline(always)]
fn split_edge_body<T: FloatLike, const L: usize>(
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

dispatched! {
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
    ) => split_vertex_body;
}

#[inline(always)]
fn split_vertex_body<T: FloatLike, const L: usize>(
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
    assert_eq!(
        vertex.len(),
        layout.entries().len() * v,
        "vertex tensor does not match its layout"
    );
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
