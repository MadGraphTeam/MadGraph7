// SPDX-License-Identifier: 0BSD

//! Batch inputs, fused factors and dressing steps of the v2 executor
//! (DESIGN.md §5.1–5.3).
//!
//! **Carriers.** Every value is a [`Carrier`]: `f64`, `DoubleFloat`, `Float`
//! (any precision) or exact `Rational`. Momenta, scales and masses enter as
//! exact rationals and are cast once per lane, so frame shifts, block momenta
//! and invariants are exact until cast (contracts K4, K6). Block-current
//! rows, Born components and kernel constants enter in the carrier.
//!
//! **Lanes.** A chunk of `L` points is evaluated in lockstep: a complex value
//! is a lane vector of `L` real parts followed by `L` imaginary parts
//! ([`super::kernels::lane_vector_len`]). A chunk with fewer points repeats
//! its last point; padded lanes are never returned.
//!
//! **Fused factors.** `X(J)` of a (content, frame shift) holds, per content
//! row, one lane vector per fused entry: the sum over the content's routing
//! terms and kernel bindings of each kernel output, a polynomial in the loop
//! shift, the slots' block momenta and block-current components with exact
//! catalogue terms and numbered model constants. The order of every sum and
//! product is that of the Python reference executor.

use super::kernels::{RowLink, Segment, clear, fused_edge, lane_vector_len};
use super::plan::{Content, Family, KernelInput, LoopProgramV4};
use super::poly::MonomialProductTable;
use crate::{RusticolError, RusticolResult};
use std::collections::HashMap;
use symbolica::domains::float::{Complex, RealLike, SingleFloat};
use symbolica::domains::integer::Integer;
use symbolica::domains::rational::Rational;

/// A real number type of the executor.
pub trait Carrier: SingleFloat + RealLike + Send + Sync + 'static {}
impl<T: SingleFloat + RealLike + Send + Sync + 'static> Carrier for T {}

/// Parse an exact decimal (`-1.25e-3`) or fraction (`-3/7`) literal.
pub fn parse_rational(value: &str) -> RusticolResult<Rational> {
    let value = value.trim();
    if let Some((numerator, denominator)) = value.split_once('/') {
        let parse = |text: &str| {
            text.trim()
                .parse::<Integer>()
                .map_err(|e| RusticolError::invalid_argument(format!("{text:?}: {e}")))
        };
        let denominator = parse(denominator)?;
        if denominator.is_zero() {
            return Err(RusticolError::invalid_argument("zero denominator"));
        }
        return Ok(Rational::new(parse(numerator)?, denominator));
    }
    crate::source_scalar::decimal_rational(value).map_err(RusticolError::invalid_argument)
}

/// The carrier value nearest to `value`.
///
/// Every carrier rounds once: numerica's binary64 cast divides two rounded
/// integers, which double-rounds numerators or denominators beyond 2^53
/// (17-digit decimal momenta), so binary64 goes through one MPFR rounding
/// like Python's `float(Fraction)`.
pub fn cast<T: Carrier>(zero: &T, value: &Rational) -> T {
    if (zero as &dyn std::any::Any).is::<f64>() {
        const EXACT: i64 = 1 << f64::MANTISSA_DIGITS;
        let rounded = match (value.numerator_ref(), value.denominator_ref()) {
            // Both exact in binary64: one correctly rounded division.
            (Integer::Single(n), Integer::Single(d)) if n.abs() <= EXACT && *d <= EXACT => {
                *n as f64 / *d as f64
            }
            _ => value.to_multi_prec_float(f64::MANTISSA_DIGITS).to_f64(),
        };
        return (&rounded as &dyn std::any::Any)
            .downcast_ref::<T>()
            .expect("binary64 carrier")
            .clone();
    }
    zero.from_rational(value)
}

/// The leg whose mass shell fixes the invariant of `mask` among `legs`
/// legs: the leg of a single-leg subset or of a single-leg complement
/// (momentum conservation), else `None` (`families.onshell_leg`).
pub fn onshell_leg(mask: u64, legs: usize) -> Option<usize> {
    if legs == 0 || legs >= 64 {
        return None;
    }
    let full = (1u64 << legs) - 1;
    [mask & full, full ^ (mask & full)]
        .into_iter()
        .find(|&subset| subset != 0 && subset & (subset - 1) == 0)
        .map(|subset| subset.trailing_zeros() as usize)
}

/// One batch of points: exact kinematics, carrier inputs.
#[derive(Clone)]
pub struct V2Batch<T> {
    /// Per point, `4 × legs` all-incoming momentum components.
    pub momenta: Vec<Vec<Rational>>,
    /// Per point, `mu_R²`.
    pub scales: Vec<Rational>,
    /// Squared mass per plan mass id (its real part).
    pub masses: Vec<Rational>,
    /// The imaginary part of each squared plan mass (complex-mass-scheme
    /// pole masses, `M² - i M Γ`), or empty: every plan mass real.
    pub masses_imaginary: Vec<Rational>,
    /// Squared mass of each leg's external state (in leg order): the value
    /// of every invariant a single leg fixes on its mass shell (a single-leg
    /// arc or its complement), never the `p²` of the given momenta, which
    /// rounding leaves off shell (DESIGN §11.1, on-shell external
    /// invariants). `None` leaves a leg's invariants to the momenta (a mass
    /// that is no model parameter); empty: every invariant from the momenta
    /// (callers without external states, e.g. synthetic benchmark batches).
    pub leg_masses: Vec<Option<Rational>>,
    /// Model constants of the kernel terms.
    pub constants: Vec<Complex<T>>,
    /// Per point, the components of every block row, in plan order.
    pub block_rows: Vec<Vec<Complex<T>>>,
    /// Per point, Born components per (helicity row, component).
    pub born: Vec<Vec<Complex<T>>>,
}

impl<T: Carrier> V2Batch<T> {
    pub fn points(&self) -> usize {
        self.momenta.len()
    }

    /// Check every shape against the plan.
    pub fn validate(&self, plan: &LoopProgramV4) -> RusticolResult<()> {
        let points = self.points();
        let block = plan.block_rows.iter().sum::<usize>();
        let bad = |message: &str| Err(RusticolError::invalid_argument(message.to_string()));
        if points == 0 {
            return bad("a batch needs at least one point");
        }
        if self.scales.len() != points
            || self.block_rows.len() != points
            || self.born.len() != points
        {
            return bad("batch inputs disagree in their number of points");
        }
        if self.momenta.iter().any(|p| p.len() != 4 * plan.legs) {
            return bad("momenta need four components per leg");
        }
        if self.masses.len() != plan.masses || self.constants.len() != plan.constants {
            return bad("masses or kernel constants do not match the plan");
        }
        if !self.masses_imaginary.is_empty() && self.masses_imaginary.len() != plan.masses {
            return bad("imaginary parts of the masses do not match the plan");
        }
        if !self.leg_masses.is_empty() && self.leg_masses.len() != plan.legs {
            return bad("leg masses need one squared mass per leg");
        }
        if self.block_rows.iter().any(|rows| rows.len() != block) {
            return bad("block rows do not match the plan");
        }
        if self
            .born
            .iter()
            .any(|born| born.len() != plan.rows * plan.born_components)
        {
            return bad("Born components do not match the plan");
        }
        Ok(())
    }
}

// --------------------------------------------------------------- lane algebra

/// `out = a * b` on complex lane vectors.
#[inline]
pub(super) fn cmul_into<T: Carrier, const L: usize>(out: &mut [T], a: &[T], b: &[T]) {
    for l in 0..L {
        let re = a[l].clone() * &b[l] - a[L + l].clone() * &b[L + l];
        let im = a[l].clone() * &b[L + l] + a[L + l].clone() * &b[l];
        out[l] = re;
        out[L + l] = im;
    }
}

/// `acc += a` on complex lane vectors.
#[inline]
pub(super) fn cadd<T: Carrier, const L: usize>(acc: &mut [T], a: &[T]) {
    for (x, y) in acc[..2 * L].iter_mut().zip(&a[..2 * L]) {
        *x += y;
    }
}

// ------------------------------------------------------------------ chunk data

/// Exact invariants of one chunk, per distinct point: the summed incoming
/// momentum `p(mask)` and the invariant `s[mask]` of each leg subset in use.
/// A subset's momentum is its lowest leg added to the cached momentum of
/// the rest, so every subset costs one exact addition per component, once
/// per chunk, however many factors, closures and reduction lanes read it.
/// `s[mask]` is `p(mask)²`, except that a subset that is a single leg, or
/// whose complement is, takes that leg's squared mass from `onshell`
/// ([`V2Batch::leg_masses`]; empty: none).
struct ExactInvariants {
    /// Per distinct point, `4 × legs` momentum components.
    legs: Vec<Vec<Rational>>,
    momenta: Vec<HashMap<u64, [Rational; 4]>>,
    squares: Vec<HashMap<u64, Rational>>,
    /// Squared mass of each leg's external state (empty: none).
    onshell: Vec<Option<Rational>>,
}

impl ExactInvariants {
    fn momentum(&mut self, point: usize, mask: u64) -> [Rational; 4] {
        if mask == 0 {
            return std::array::from_fn(|_| Rational::from(0));
        }
        if let Some(found) = self.momenta[point].get(&mask) {
            return found.clone();
        }
        let leg = mask.trailing_zeros() as usize;
        let rest = self.momentum(point, mask & (mask - 1));
        let legs = &self.legs[point];
        let value: [Rational; 4] = std::array::from_fn(|mu| rest[mu].clone() + &legs[4 * leg + mu]);
        self.momenta[point].insert(mask, value.clone());
        value
    }

    fn square(&mut self, point: usize, mask: u64) -> Rational {
        if let Some(found) = self.squares[point].get(&mask) {
            return found.clone();
        }
        if let Some(Some(mass)) =
            onshell_leg(mask, self.onshell.len()).map(|leg| &self.onshell[leg])
        {
            return mass.clone();
        }
        let p = self.momentum(point, mask);
        let value = p[0].clone() * &p[0]
            - p[1].clone() * &p[1]
            - p[2].clone() * &p[2]
            - p[3].clone() * &p[3];
        self.squares[point].insert(mask, value.clone());
        value
    }
}

/// The inputs of one chunk of `L` lanes as lane vectors.
pub struct LaneData<T, const L: usize> {
    /// Batch point of each lane (padded lanes repeat the last point).
    pub points: [usize; L],
    /// Number of real (unpadded) lanes.
    pub active: usize,
    pub zero: T,
    /// Distinct point of each lane (padded lanes share the last one).
    distinct: [usize; L],
    exact: std::sync::Mutex<ExactInvariants>,
    /// Exact `mu_R²` per lane.
    pub scales: Vec<Rational>,
    momentum: std::sync::Mutex<HashMap<u64, std::sync::Arc<Vec<T>>>>,
    square: std::sync::Mutex<HashMap<u64, std::sync::Arc<Vec<T>>>>,
    constants: Vec<Complex<T>>,
    /// Per block row, its components as lane vectors.
    block_rows: Vec<Vec<T>>,
    /// Per (helicity row, component), the Born component as a lane vector.
    pub born: Vec<T>,
}

impl<T: Carrier, const L: usize> LaneData<T, L> {
    /// Gather the lanes `points` (at most `L`) of `batch`; `zero` fixes the
    /// carrier precision.
    pub fn new(
        plan: &LoopProgramV4,
        batch: &V2Batch<T>,
        points: &[usize],
        zero: &T,
    ) -> RusticolResult<Self> {
        if points.is_empty() || points.len() > L {
            return Err(RusticolError::invalid_argument(
                "a chunk holds 1..=L points",
            ));
        }
        let last = *points.last().expect("non-empty");
        let lanes: [usize; L] = std::array::from_fn(|l| points.get(l).copied().unwrap_or(last));
        let v = lane_vector_len(L);
        let zero = zero.zero();
        let mut block_offsets = Vec::with_capacity(plan.block_rows.len());
        let mut offset = 0;
        for &dimension in &plan.block_rows {
            block_offsets.push(offset);
            offset += dimension;
        }
        let block_rows = (0..plan.block_rows.len())
            .map(|row| {
                let dimension = plan.block_rows[row];
                let mut out = vec![zero.clone(); dimension * v];
                for k in 0..dimension {
                    for (l, &point) in lanes.iter().enumerate() {
                        let z = &batch.block_rows[point][block_offsets[row] + k];
                        out[k * v + l] = z.re.clone();
                        out[k * v + L + l] = z.im.clone();
                    }
                }
                out
            })
            .collect();
        let width = plan.rows * plan.born_components;
        let mut born = vec![zero.clone(); width * v];
        for slot in 0..width {
            for (l, &point) in lanes.iter().enumerate() {
                let z = &batch.born[point][slot];
                born[slot * v + l] = z.re.clone();
                born[slot * v + L + l] = z.im.clone();
            }
        }
        let distinct = std::array::from_fn(|l| l.min(points.len() - 1));
        let exact = ExactInvariants {
            legs: points.iter().map(|&p| batch.momenta[p].clone()).collect(),
            momenta: vec![HashMap::new(); points.len()],
            squares: vec![HashMap::new(); points.len()],
            onshell: batch.leg_masses.clone(),
        };
        Ok(Self {
            points: lanes,
            active: points.len(),
            distinct,
            exact: std::sync::Mutex::new(exact),
            scales: lanes.iter().map(|&p| batch.scales[p].clone()).collect(),
            momentum: std::sync::Mutex::new(HashMap::new()),
            square: std::sync::Mutex::new(HashMap::new()),
            constants: batch.constants.clone(),
            block_rows,
            born,
            zero,
        })
    }

    pub fn cast(&self, value: &Rational) -> T {
        cast(&self.zero, value)
    }

    /// Exact summed incoming momentum of the legs in `mask` at `lane`.
    pub fn exact_momentum(&self, lane: usize, mask: u64) -> [Rational; 4] {
        self.exact
            .lock()
            .expect("exact invariants")
            .momentum(self.distinct[lane], mask)
    }

    /// The exact invariant `s[mask]` at `lane`: `p(mask)²`, or the squared
    /// mass of the leg whose mass shell fixes it ([`V2Batch::leg_masses`]).
    pub fn exact_square(&self, lane: usize, mask: u64) -> Rational {
        self.exact
            .lock()
            .expect("exact invariants")
            .square(self.distinct[lane], mask)
    }

    /// The momentum of `mask` as four complex lane vectors `[mu][re | im]`
    /// (exact sums cast once per distinct point, zero imaginary parts).
    pub fn momentum(&self, mask: u64) -> std::sync::Arc<Vec<T>> {
        if let Some(found) = self.momentum.lock().expect("momentum cache").get(&mask) {
            return found.clone();
        }
        let v = lane_vector_len(L);
        let mut out = vec![self.zero.clone(); 4 * v];
        for l in 0..L {
            let source = self.distinct[l];
            if source != l {
                for mu in 0..4 {
                    out[mu * v + l] = out[mu * v + source].clone();
                }
                continue;
            }
            let p = self.exact_momentum(l, mask);
            for mu in 0..4 {
                out[mu * v + l] = self.cast(&p[mu]);
            }
        }
        let out = std::sync::Arc::new(out);
        self.momentum
            .lock()
            .expect("momentum cache")
            .insert(mask, out.clone());
        out
    }

    /// The invariant `s[mask]` ([`Self::exact_square`]) as a real lane
    /// vector (exact values cast once per distinct point).
    pub fn square(&self, mask: u64) -> std::sync::Arc<Vec<T>> {
        if let Some(found) = self.square.lock().expect("square cache").get(&mask) {
            return found.clone();
        }
        let mut out = vec![self.zero.clone(); L];
        for l in 0..L {
            let source = self.distinct[l];
            out[l] = if source == l {
                self.cast(&self.exact_square(l, mask))
            } else {
                out[source].clone()
            };
        }
        let out = std::sync::Arc::new(out);
        self.square
            .lock()
            .expect("square cache")
            .insert(mask, out.clone());
        out
    }

    pub fn constant(&self, index: usize) -> &Complex<T> {
        &self.constants[index]
    }

    /// Component `k` of block row `row` as a lane vector.
    pub fn block_component(&self, row: usize, k: usize) -> &[T] {
        let v = lane_vector_len(L);
        &self.block_rows[row][k * v..][..v]
    }
}

// ----------------------------------------------------------- chunk inputs

/// Lane values of one chunk built on demand per id (fused factors, split
/// vertex tensors or split block currents).
pub struct LaneStore<T> {
    values: Vec<Option<Vec<T>>>,
}

impl<T: Carrier> LaneStore<T> {
    pub fn new(len: usize) -> Self {
        Self {
            values: (0..len).map(|_| None).collect(),
        }
    }

    /// Build every value of `ids` that is not built yet (in parallel).
    pub fn build(&mut self, ids: &[usize], build: impl Fn(usize) -> Vec<T> + Sync) {
        use rayon::prelude::*;
        let missing: Vec<usize> = ids
            .iter()
            .copied()
            .filter(|&id| self.values[id].is_none())
            .collect();
        let built: Vec<(usize, Vec<T>)> = missing.par_iter().map(|&id| (id, build(id))).collect();
        for (id, values) in built {
            self.values[id] = Some(values);
        }
    }

    pub fn get(&self, id: usize) -> &[T] {
        self.values[id]
            .as_deref()
            .expect("lane input built before use")
    }

    /// Drop the values not in `keep` (sorted ascending).
    pub fn retain(&mut self, keep: &[usize]) {
        for (id, value) in self.values.iter_mut().enumerate() {
            if value.is_some() && keep.binary_search(&id).is_err() {
                *value = None;
            }
        }
    }

    /// Reals held by built values.
    pub fn reals(&self) -> usize {
        self.values.iter().flatten().map(Vec::len).sum()
    }
}

/// Ids of the chunk inputs a set of families reads, each sorted ascending.
#[derive(Clone, Debug, Default)]
pub struct Reads {
    pub factors: Vec<usize>,
    pub vertices: Vec<usize>,
    pub currents: Vec<usize>,
}

impl Reads {
    /// Everything the schedule-B steps of `families` read.
    pub fn of(plan: &LoopProgramV4, families: &[usize]) -> Self {
        let union = |ids: fn(&Family) -> &[usize]| {
            let mut out: Vec<usize> = families
                .iter()
                .flat_map(|&f| ids(&plan.families[f]).iter().copied())
                .collect();
            out.sort_unstable();
            out.dedup();
            out
        };
        Self {
            factors: union(|f| &f.factors),
            vertices: union(|f| &f.vertices),
            currents: union(|f| &f.currents),
        }
    }
}

/// The lane inputs of one chunk: fused factors `X(J)` per (content, shift),
/// split vertex tensors per (vertex, shift) and split block currents per
/// (content, term).
pub struct ChunkInputs<T> {
    pub factors: LaneStore<T>,
    pub vertices: LaneStore<T>,
    pub currents: LaneStore<T>,
}

impl<T: Carrier> ChunkInputs<T> {
    pub fn new(plan: &LoopProgramV4) -> Self {
        Self {
            factors: LaneStore::new(plan.factors.len()),
            vertices: LaneStore::new(plan.vertex_instances.len()),
            currents: LaneStore::new(plan.currents.len()),
        }
    }

    /// Keep the inputs in `live`, then build those in `needed`.
    pub fn prepare<const L: usize>(
        &mut self,
        plan: &LoopProgramV4,
        data: &LaneData<T, L>,
        needed: &Reads,
        live: &Reads,
    ) {
        self.factors.retain(&live.factors);
        self.vertices.retain(&live.vertices);
        self.currents.retain(&live.currents);
        self.factors
            .build(&needed.factors, |id| build_factor::<T, L>(plan, id, data));
        self.vertices
            .build(&needed.vertices, |id| build_vertex::<T, L>(plan, id, data));
        self.currents
            .build(&needed.currents, |id| build_current::<T, L>(plan, id, data));
    }

    pub fn reals(&self) -> usize {
        self.factors.reals() + self.vertices.reals() + self.currents.reals()
    }
}

/// Lane vector of one kernel input: loop momentum at the frame shift, a
/// slot's block momentum or block-current component, or `None` for a
/// parameter (folded into the kernel constants).
fn kernel_input<'a, T: Carrier, const L: usize>(
    input: KernelInput,
    loop_momentum: &'a [T],
    slot_momenta: &'a [std::sync::Arc<Vec<T>>],
    blocks: Option<&[usize]>,
    data: &'a LaneData<T, L>,
) -> Option<&'a [T]> {
    let v = lane_vector_len(L);
    match input {
        KernelInput::Loop(mu) => Some(&loop_momentum[mu * v..][..v]),
        KernelInput::Current { slot, component } => {
            Some(data.block_component(blocks.expect("block rows")[slot], component))
        }
        KernelInput::Momentum { slot, mu } => Some(&slot_momenta[slot][mu * v..][..v]),
        KernelInput::Parameter => None,
    }
}

/// `V` of vertex `vertex` at frame shift `shift`: one lane vector per vertex
/// entry, the sum of its kernels' outputs.
fn vertex_values<T: Carrier, const L: usize>(
    plan: &LoopProgramV4,
    vertex: usize,
    shift: u64,
    data: &LaneData<T, L>,
    scratch: &mut Scratch<T>,
) -> Vec<T> {
    let vertex = &plan.vertices[vertex];
    let v = lane_vector_len(L);
    let loop_momentum = data.momentum(shift);
    let slot_momenta: Vec<_> = vertex
        .slots
        .iter()
        .map(|&mask| data.momentum(mask))
        .collect();
    let mut out = vec![data.zero.clone(); vertex.entries.len() * v];
    for binding in &vertex.bindings {
        let kernel = &plan.kernels[binding.kernel];
        let inputs: Vec<Option<&[T]>> = binding
            .inputs
            .iter()
            .map(|&i| kernel_input(i, &loop_momentum, &slot_momenta, None, data))
            .collect();
        for (terms, &entry) in kernel.outputs.iter().zip(&binding.entries) {
            scratch.evaluate::<L>(terms, &inputs, data);
            cadd::<T, L>(&mut out[entry * v..][..v], &scratch.total);
        }
    }
    out
}

/// Vertex instance `id` of the plan (split form).
pub fn build_vertex<T: Carrier, const L: usize>(
    plan: &LoopProgramV4,
    id: usize,
    data: &LaneData<T, L>,
) -> Vec<T> {
    let (vertex, shift) = plan.vertex_instances[id];
    let mut scratch = Scratch::<T>::new(&data.zero, lane_vector_len(L));
    vertex_values(plan, vertex, shift, data, &mut scratch)
}

/// Split current `id` of the plan: per term row, the `width` products `J[ν]`
/// of the slots' block components.
pub fn build_current<T: Carrier, const L: usize>(
    plan: &LoopProgramV4,
    id: usize,
    data: &LaneData<T, L>,
) -> Vec<T> {
    let (content, term) = plan.currents[id];
    let term = &plan.contents[content].terms[term];
    let v = lane_vector_len(L);
    let row_len = term.width() * v;
    let mut out = vec![data.zero.clone(); term.rows.len() * row_len];
    let mut scratch = vec![data.zero.clone(); v];
    // The rows schedule B reads (all without the live-row pruning).
    let read = plan
        .read_rows
        .as_ref()
        .map(|rows| rows.currents[id].as_slice());
    for (k, ((_row, blocks), target)) in term
        .rows
        .iter()
        .zip(out.chunks_exact_mut(row_len))
        .enumerate()
    {
        if read.is_some_and(|read| !read[k]) {
            continue;
        }
        block_components::<T, L>(term, blocks, data, target, &mut scratch);
    }
    out
}

/// `X(J)` of factor `id`: per content row, one lane vector per fused entry.
///
/// A routing term with a split vertex evaluates the vertex tensor `V[e]`
/// once (loop shift and block momenta only) and contracts it with each
/// row's block components, `X[row][fused(e)] += V[e] J[row][ν(e)]`; a term
/// without one evaluates its fused kernels per row. Both are the same
/// polynomial (the reference executor's fused and split forms agree
/// exactly).
pub fn build_factor<T: Carrier, const L: usize>(
    plan: &LoopProgramV4,
    id: usize,
    data: &LaneData<T, L>,
) -> Vec<T> {
    let (content_index, shift) = plan.factors[id];
    let content = &plan.contents[content_index];
    let v = lane_vector_len(L);
    let row_len = content.factor_row() * v;
    let mut out = vec![data.zero.clone(); content.rows * row_len];
    let loop_momentum = data.momentum(shift);
    let mut scratch = Scratch::<T>::new(&data.zero, v);
    // The content rows schedule B reads (all without the live-row pruning):
    // the others stay zero.
    let read = plan
        .read_rows
        .as_ref()
        .map(|rows| rows.factors[id].as_slice());
    let reads = |row: usize| read.is_none_or(|read| read[row]);
    for term in &content.terms {
        if let Some(split) = &term.vertex {
            if !term.rows.iter().any(|(row, _)| reads(*row)) {
                continue;
            }
            let vertex = &plan.vertices[split.id];
            let values = vertex_values(plan, split.id, shift, data, &mut scratch);
            let mut components = vec![data.zero.clone(); term.width() * v];
            let mut product = vec![data.zero.clone(); v];
            for (row, blocks) in term.rows.iter().filter(|(row, _)| reads(*row)) {
                block_components::<T, L>(term, blocks, data, &mut components, &mut product);
                let target = &mut out[row * row_len..][..row_len];
                for (e, (&(_b, nu, _bp, _exponents), &fused)) in
                    vertex.entries.iter().zip(&split.fused).enumerate()
                {
                    cmul_into::<T, L>(
                        &mut product,
                        &values[e * v..][..v],
                        &components[nu * v..][..v],
                    );
                    cadd::<T, L>(&mut target[fused * v..][..v], &product);
                }
            }
            continue;
        }
        let slot_momenta: Vec<_> = term.slots.iter().map(|&mask| data.momentum(mask)).collect();
        for (row, blocks) in term.rows.iter().filter(|(row, _)| reads(*row)) {
            let target = &mut out[row * row_len..][..row_len];
            for binding in &term.bindings {
                let kernel = &plan.kernels[binding.kernel];
                let inputs: Vec<Option<&[T]>> = binding
                    .inputs
                    .iter()
                    .map(|&i| kernel_input(i, &loop_momentum, &slot_momenta, Some(blocks), data))
                    .collect();
                for (terms, &entry) in kernel.outputs.iter().zip(&binding.entries) {
                    scratch.evaluate::<L>(terms, &inputs, data);
                    cadd::<T, L>(&mut target[entry * v..][..v], &scratch.total);
                }
            }
        }
    }
    out
}

/// `J[nu]`: the products of the slots' block components of one row, flat
/// over the slots with the first slot slowest.
fn block_components<T: Carrier, const L: usize>(
    term: &super::plan::ContentTerm,
    blocks: &[usize],
    data: &LaneData<T, L>,
    out: &mut [T],
    scratch: &mut [T],
) {
    let v = lane_vector_len(L);
    let mut filled = 1;
    for x in &mut out[..v] {
        *x = data.zero.clone();
    }
    for x in &mut out[..L] {
        *x = data.zero.one();
    }
    let mut head = vec![data.zero.clone(); v];
    for (slot, &dimension) in term.dimensions.iter().enumerate() {
        // Expand in place from the back: nu = previous * dimension + k.
        for previous in (0..filled).rev() {
            head.clone_from_slice(&out[previous * v..][..v]);
            for k in (0..dimension).rev() {
                cmul_into::<T, L>(scratch, &head, data.block_component(blocks[slot], k));
                out[(previous * dimension + k) * v..][..v].clone_from_slice(scratch);
            }
        }
        filled *= dimension;
    }
}

/// Work vectors of the factor builder.
struct Scratch<T> {
    value: Vec<T>,
    product: Vec<T>,
    total: Vec<T>,
}

impl<T: Carrier> Scratch<T> {
    fn new(zero: &T, v: usize) -> Self {
        Self {
            value: vec![zero.clone(); v],
            product: vec![zero.clone(); v],
            total: vec![zero.clone(); v],
        }
    }

    fn total_clear(&mut self) {
        clear(&mut self.total);
    }

    fn value_set(&mut self, constant: &Complex<T>) {
        let half = self.value.len() / 2;
        for l in 0..half {
            self.value[l] = constant.re.clone();
            self.value[half + l] = constant.im.clone();
        }
    }

    fn value_mul<const L: usize>(&mut self, factor: &[T]) {
        cmul_into::<T, L>(&mut self.product, &self.value, factor);
        std::mem::swap(&mut self.value, &mut self.product);
    }

    fn total_add<const L: usize>(&mut self) {
        cadd::<T, L>(&mut self.total, &self.value);
    }

    /// `total = Σ_terms constant Π input^power` (the reference executor's
    /// `ExactKernel.evaluate` order).
    fn evaluate<const L: usize>(
        &mut self,
        terms: &[super::plan::KernelTerm],
        inputs: &[Option<&[T]>],
        data: &LaneData<T, L>,
    ) {
        self.total_clear();
        for term in terms {
            self.value_set(data.constant(term.constant));
            for &(position, power) in &term.monomial {
                let factor = inputs[position].expect("parameter in a monomial");
                for _ in 0..power {
                    self.value_mul::<L>(factor);
                }
            }
            self.total_add::<L>();
        }
    }
}

// ----------------------------------------------------------------------- pool

/// Reused object buffers of one family walk, with the peak of live reals.
pub struct Pool<T> {
    free: Vec<Vec<T>>,
    zero: T,
    live: usize,
    pub peak: usize,
}

impl<T: Carrier> Pool<T> {
    pub fn new(zero: &T) -> Self {
        Self {
            free: Vec::new(),
            zero: zero.zero(),
            live: 0,
            peak: 0,
        }
    }

    /// A zeroed buffer of `len` reals.
    pub fn take(&mut self, len: usize) -> Vec<T> {
        let mut buffer = self.free.pop().unwrap_or_default();
        if buffer.len() >= len {
            buffer.truncate(len);
            clear(&mut buffer);
        } else {
            clear(&mut buffer);
            buffer.resize(len, self.zero.clone());
        }
        self.live += len;
        self.peak = self.peak.max(self.live);
        buffer
    }

    pub fn give(&mut self, buffer: Vec<T>) {
        self.live -= buffer.len();
        self.free.push(buffer);
    }
}

// --------------------------------------------------------------------- steps

/// One dressing edge: `output += input · X(J)` over the edge's row links.
///
/// `seeded` inputs hold one lane vector per object row (a seed times
/// `δ_ab`) and use the content's seeded layout; open-loop inputs hold
/// `d_alpha × d_in × C(r+4,4)` lane vectors per row.
#[allow(clippy::too_many_arguments)]
pub fn dress_edge<T: Carrier, const L: usize>(
    content: &Content,
    seeded: bool,
    d_alpha: usize,
    products: &MonomialProductTable,
    rank_out: usize,
    input: &[T],
    factor: &[T],
    links: &[RowLink],
    output: &mut [T],
) {
    if seeded {
        let segment = Segment::new(1, 1, 1, content.d_in * content.d_out, products, rank_out);
        fused_edge::<T, L>(&segment, &content.seeded, input, factor, links, output);
    } else {
        let segment = Segment::new(d_alpha, content.d_in, 1, content.d_out, products, rank_out);
        fused_edge::<T, L>(&segment, &content.open, input, factor, links, output);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rationals_parse_from_fractions_and_decimals() {
        assert_eq!(parse_rational("-3/7").unwrap(), Rational::from((-3, 7)));
        assert_eq!(parse_rational("1.25e-1").unwrap(), Rational::from((1, 8)));
        assert_eq!(parse_rational("42").unwrap(), Rational::from(42));
        assert!(parse_rational("1/0").is_err());
        assert!(parse_rational("x").is_err());
    }

    #[test]
    fn binary64_casts_round_once() {
        // 17-digit decimals have numerators beyond 2^53: numerica's direct
        // binary64 cast rounds numerator and quotient separately.
        for text in ["434.25236936308625", "-57.655920542748461", "0.1", "1e-300"] {
            let exact = parse_rational(text).unwrap();
            assert_eq!(cast(&0.0, &exact), text.parse::<f64>().unwrap(), "{text}");
        }
        let third = Rational::from((1, 3));
        assert_eq!(cast(&Rational::from(0), &third), third);
    }

    #[test]
    fn exact_invariants_equal_direct_sums() {
        let legs: Vec<Rational> = (0..12)
            .map(|k| Rational::from(((k * 7 % 11) as i64 - 5, (k % 4 + 1) as i64)))
            .collect();
        let mut table = ExactInvariants {
            legs: vec![legs.clone()],
            momenta: vec![HashMap::new()],
            squares: vec![HashMap::new()],
            onshell: Vec::new(),
        };
        for mask in [0b101u64, 0b111, 0b010, 0] {
            let direct: [Rational; 4] = std::array::from_fn(|mu| {
                (0..3)
                    .filter(|leg| mask >> leg & 1 == 1)
                    .fold(Rational::from(0), |total, leg| total + &legs[4 * leg + mu])
            });
            assert_eq!(table.momentum(0, mask), direct);
            let square = direct[0].clone() * &direct[0]
                - direct[1].clone() * &direct[1]
                - direct[2].clone() * &direct[2]
                - direct[3].clone() * &direct[3];
            assert_eq!(table.square(0, mask), square);
        }
    }

    #[test]
    fn exact_invariants_bind_single_legs_on_shell() {
        let legs: Vec<Rational> = (0..16)
            .map(|k| Rational::from(((k * 5 % 9) as i64 - 4, 1)))
            .collect();
        let table = |onshell| ExactInvariants {
            legs: vec![legs.clone()],
            momenta: vec![HashMap::new()],
            squares: vec![HashMap::new()],
            onshell,
        };
        let mut plain = table(Vec::new());
        let mut bound = table(vec![
            Some(Rational::from(2)),
            Some(Rational::from(7)),
            Some(Rational::from(0)),
            None,
        ]);
        assert_eq!(bound.square(0, 0b0010), Rational::from(7));
        // The complement of leg 1 (momentum conservation).
        assert_eq!(bound.square(0, 0b1101), Rational::from(7));
        assert_eq!(bound.square(0, 0b0100), Rational::from(0));
        // An unbound leg and a two-leg arc keep p(arc)^2; momenta are kept.
        for mask in [0b1000u64, 0b0011, 0b0110] {
            assert_eq!(bound.square(0, mask), plain.square(0, mask));
        }
        assert_eq!(bound.momentum(0, 0b0010), plain.momentum(0, 0b0010));
        assert_eq!(onshell_leg(0b0111, 4), Some(3));
        assert_eq!(onshell_leg(0, 4), None);
    }

    #[test]
    fn lane_products_match_complex_arithmetic() {
        const L: usize = 2;
        let a = [1.0, 2.0, 3.0, -1.0];
        let b = [0.5, -2.0, 4.0, 1.5];
        let mut out = [0.0; 4];
        cmul_into::<f64, L>(&mut out, &a, &b);
        // lane 0: (1 + 3i)(0.5 + 4i) = 0.5 - 12 + (4 + 1.5)i
        assert_eq!((out[0], out[2]), (0.5 - 12.0, 4.0 + 1.5));
        // lane 1: (2 - i)(-2 + 1.5i) = -4 + 1.5 + (3 + 2)i
        assert_eq!((out[1], out[3]), (-4.0 + 1.5, 3.0 + 2.0));
    }
}
