// SPDX-License-Identifier: 0BSD

//! NinjaRS adapter of the v2 loop engine (design §11.1–11.2, WP1.6).
//!
//! **Ingress.** A family numerator is the dense four-dimensional tensor
//! `N(q) = Σ_α c_α q^α` of a declared rank `R`, in NinjaRS's
//! [`tensor_layout`](ninjars::fast::tensor_layout) order (`binomial(R+4, 4)`
//! coefficients, no symmetry factors), lane-major beside the family's real
//! shifts, complex squared masses, scale and exact invariant matrix
//! `S[i,j] = (V_i - V_j)²`. The binary64 engine copies the coefficients
//! straight into upstream [`TensorNumerator`] lanes; no expression is built
//! per call.
//!
//! **Buckets.** Lanes are (family, point, sector, column) channels. Every lane
//! is reduced at its true rank: the highest degree with any coefficient that is
//! exactly non-zero (over-declared ranks cost accuracy; NinjaRS lowers a lane
//! itself only when declared at rank n+2, via `FastNumerator::lane_rank`, so
//! declaring the true rank is equivalent there). Lanes of all families with
//! equal (propagators, true rank) share one engine call.
//!
//! **Engines.** `fast` is NinjaRS's binary64 engine with oneloopmaster's
//! native masters behind a [`MasterCache`]. `generic` is NinjaRS's
//! precision-generic engine (f64, DoubleFloat, arbitrary precision), whose
//! numerator is one Symbolica template per (propagators, rank), built once
//! from the exponent arrays of the tensor layout with one parameter per
//! coefficient; its masters are evaluated uncached by `SymbolicaMasters`,
//! without an on-shell threshold.
//!
//! **Absolute thresholds of the fast engine (genericity audit 2, G2-M3).**
//! On the pinned NinjaRS 41a4c00 the fast path compares inputs with absolute
//! constants in the model's units: `OneLoopNativeMasters` sets squared masses
//! and invariants with `|re| + |im| < 1e-9` to zero and a real invariant
//! within `1e-9` of a real squared mass to that mass
//! ([`NATIVE_ONSHELL_THRESHOLD`]), and the engine itself skips the bubble of
//! two massless lines whose invariant is below `FastOptions::infrared_eps`
//! (`1e-9`) and builds its cut bases from momenta whose square is below it as
//! if they were light-like. A light mass (a keV mass in GeV, `m² = 1e-12`, or
//! the electron in TeV) or a tiny invariant thus changes the integral in
//! binary64 alone, by a unit-dependent amount. Every lane whose inputs reach
//! one of these thresholds ([`reaches_fast_threshold`]: a squared mass or an
//! off-diagonal invariant with `0 < |x| < 1e-9`, or an invariant within
//! `(0, 1e-9)` of a real squared mass of the lane; exact comparisons of the
//! given inputs, never relative to a scale) is reduced instead by the generic
//! binary64 engine, which has none; every other lane takes the fast engine
//! unchanged, so plans and points without such inputs keep its bits. The
//! check runs per frame, on the frame's own (rescaled) inputs.
//! [`V2Reducer::take_exact_lanes`] counts the routed lanes. The generic
//! engine builds and evaluates Symbolica expressions, so a routed lane is the
//! one binary64 reduction that touches Symbolica state: it takes pyAmpliCol's
//! library key on its thread (`pyamplicol::unlock`); every other binary64
//! lane touches none. Exactly zero
//! masses and invariants, and invariants exactly equal to a mass (the
//! executor's on-shell external invariants), never route. The upstream
//! `OneLoopNativeMasters::with_onshell_threshold(None)` (NinjaRS PR prepared
//! by package onshell-threshold) removes the masters' part; the engine's
//! `infrared_eps` part stays a routing rule.
//!
//! **Reduction-direction check.** [`ReductionFrame`] re-reduces the same
//! numerators in a dihedrally relabelled frame of the propagator cycle, by
//! default [`ReductionFrame::REVERSED`] (loop momentum reversed). Families
//! are given in loop-cycle order (oneloopmaster 09b2c74 misclassified some
//! soft boxes otherwise; 8ec256fc relabels D0 inside its numerical entry
//! points, the order is kept); a dihedral relabelling keeps that order, so every
//! master request of either frame keeps consecutive propagators adjacent.
//! Without `shift` the coefficients are reused exactly (odd-degree blocks
//! change sign under `q -> -q`), and box, triangle and tadpole masters are
//! cache hits; bubbles whose two masses swap order are evaluated anew.
//! With `rescale` every dimensionful input is scaled by the non-dyadic
//! [`RESCALING`] factor `x` (shifts by `x`, squared masses, scale and
//! invariants by `x²`, degree-`d` coefficients by `x^-d`), so that every
//! master is evaluated anew at differently rounded arguments; the integral
//! scales exactly as `x^(4-2N)` (all Laurent orders, the scale moving with
//! it) and is scaled back. This also tests the masters, which no relabelling
//! of the same invariants (nor a rotation of the point) changes.
//!
//! **Finite-only mode.** [`Poles::None`] returns only the finite column; the
//! pole columns are absent, not zero, also in the check.
//! Nothing else is pole-only on this NinjaRS pin: the reduction coefficients
//! multiply every Laurent order of the masters, the fast engine contracts
//! all three orders internally (three multiply-adds per master), and
//! oneloopmaster returns `[finite, 1/ε, 1/ε²]` from one evaluation of each
//! master. Numerators are four-dimensional and ε-free here; pole-only ε
//! sectors, counterterm columns and the IR-pole check belong to their owners.
//!
//! **UV and IR poles (DESIGN.md §12.4).** [`Poles::Split`] appends each
//! lane's ultraviolet residue, `[finite, 1/ε, 1/ε², 1/ε_UV]`, computed exactly
//! from the lane's own inputs ([`uv_residue`], also in the check's frame);
//! the combined columns are those of [`Poles::All`] bit for bit, so the
//! finite part never depends on the selection. NinjaRS 41a4c00 also selects
//! UV or IR residues at the masters (`ReduceOptions::masters`, oneloopmaster's
//! `PoleSelection`): its generic engine retains the scaleless bubbles that
//! carry them, its fast engine refuses a selected provider, and so does this
//! adapter. The generic engine's selection is the oracle of the residue.

use super::master_cache::{MasterCache, MasterStatistics};
use super::reduction::{ReductionBatch, Reductions};
use super::v2_poles::UvResidues;
use super::*;
use crate::FourMomentum;
use ninjars::fast::tensor_layout::{block_offset, monomial_index, monomials, ncoeff};
use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastOptions, FastResult, FastStatus, RMom, TensorNumerator,
};
use ninjars_symbolica::OneLoopNativeMasters;
pub use ninjars_symbolica::PoleSelection;
pub use rusticol_core::loop_execution::v2::Poles;
use std::collections::{BTreeMap, HashMap};
use symbolica::{domains::float::Real, evaluate::EvaluationDomain};

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

/// `OneLoopNativeMasters`' absolute on-shell threshold on the pinned NinjaRS
/// 41a4c00 (original Ninja's `INFRARED_EPS`, the crate-private
/// `oneloop_native::ONSHELL_THRESHOLD`; a test pins the provider's behaviour
/// at it).
pub const NATIVE_ONSHELL_THRESHOLD: f64 = 1e-9;

/// The fast engine's routing threshold, in the model's units: the larger of
/// [`NATIVE_ONSHELL_THRESHOLD`] and `FastOptions::infrared_eps` (module
/// docs). The one value [`V2Reducer`]'s binary64 engine routes by, and the
/// one the generation record of light masses compares with
/// (`GeneratedProcess.binary64_exact_engine`).
pub fn fast_engine_threshold() -> f64 {
    NATIVE_ONSHELL_THRESHOLD.max(FastOptions::default().infrared_eps)
}

/// Whether one lane's binary64 inputs reach an absolute threshold of the fast
/// engine below `threshold` (module docs): a squared mass or an off-diagonal
/// invariant `x` with `0 < |re x| + |im x| < threshold`, or an off-diagonal
/// invariant `s` and a real squared mass `m²` of the lane with
/// `0 < |s - m²| < threshold`. `masses` holds the lane's `n` squared masses
/// and `s_matrix` its `n²` invariants. A superset of the rows the masters
/// move (a master row compares an invariant with its own masses only).
pub fn reaches_fast_threshold(
    n: usize,
    masses: &[Complex<f64>],
    s_matrix: &[Complex<f64>],
    threshold: f64,
) -> bool {
    squared_mass_reaches_fast_threshold(masses, threshold)
        || invariant_reaches_fast_threshold(n, masses, s_matrix, threshold)
}

/// The squared masses' clause of [`reaches_fast_threshold`]: a squared mass
/// `x` with `0 < |re x| + |im x| < threshold`.
pub(super) fn squared_mass_reaches_fast_threshold(masses: &[Complex<f64>], threshold: f64) -> bool {
    masses.iter().any(|m| {
        let size = m.re.abs() + m.im.abs();
        size > 0. && size < threshold
    })
}

/// The invariants' clause of [`reaches_fast_threshold`]: an off-diagonal
/// invariant `s` with `0 < |re s| + |im s| < threshold`, or within
/// `(0, threshold)` of a real squared mass of the lane.
fn invariant_reaches_fast_threshold(
    n: usize,
    masses: &[Complex<f64>],
    s_matrix: &[Complex<f64>],
    threshold: f64,
) -> bool {
    let inside = |size: f64| size > 0. && size < threshold;
    (0..n).any(|i| {
        (0..n).filter(|&j| j != i).any(|j| {
            let s = &s_matrix[i * n + j];
            inside(s.re.abs() + s.im.abs())
                || masses
                    .iter()
                    .any(|m| m.im == 0. && inside((s.re - m.re).abs()))
        })
    })
}

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

    fn order(self, n: usize) -> Vec<usize> {
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

/// Options of one [`V2Reducer::reduce`] call.
#[derive(Clone, Copy, Debug)]
pub struct ReduceOptions {
    /// The Laurent columns of every lane: finite only, combined, or combined
    /// plus the ultraviolet residue.
    pub poles: Poles,
    /// Also reduce every lane in this frame, for a stability estimate.
    pub check: Option<ReductionFrame>,
    /// Pole residues the masters keep (oneloopmaster's selection, applied
    /// before the reduction coefficients contract them). Only the generic
    /// engine accepts a selection other than `BOTH`.
    pub masters: PoleSelection,
}

impl Default for ReduceOptions {
    fn default() -> Self {
        Self {
            poles: Poles::All,
            check: None,
            masters: PoleSelection::BOTH,
        }
    }
}

/// One family's lanes as typed batches with equal row identities.
///
/// `shifts` has `4n` columns (`V_i` component-major per propagator),
/// `masses_squared` `n`, `scales_squared` one, `s_matrix` `n²` (row-major)
/// and `coefficients` `binomial(R+4, 4)` for the declared rank `R`.
pub struct FamilyGroup<'a> {
    pub propagators: usize,
    pub shifts: &'a Batch,
    pub masses_squared: &'a Batch,
    pub scales_squared: &'a Batch,
    pub s_matrix: &'a Batch,
    pub coefficients: &'a Batch,
}

/// Raw Laurent integrals `[finite, 1/ε, 1/ε²]` (finite only without poles),
/// one batch per input group, in the input row order.
pub struct V2Output {
    pub direct: Vec<ReductionBatch>,
    pub check: Option<Vec<ReductionBatch>>,
}

/// Declared rank whose dense tensor has `columns` coefficients.
pub fn tensor_rank(columns: usize) -> ExecutionResult<usize> {
    (0..)
        .take_while(|&rank| ncoeff(rank) <= columns)
        .find(|&rank| ncoeff(rank) == columns)
        .ok_or_else(|| error("coefficient columns are not binomial(rank + 4, 4)"))
}

/// Highest degree with an exactly non-zero coefficient (0 for a zero tensor).
pub fn true_rank<T: Carrier>(coefficients: &[Complex<T>], rank: usize) -> usize {
    (1..=rank)
        .rev()
        .find(|&degree| {
            coefficients[block_offset(degree)..ncoeff(degree)]
                .iter()
                .any(|c| !c.re.is_zero() || !c.im.is_zero())
        })
        .unwrap_or(0)
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
fn reverse_coefficients<T: Carrier>(coefficients: &mut [Complex<T>], rank: usize) {
    for degree in (1..=rank).step_by(2) {
        for c in &mut coefficients[block_offset(degree)..ncoeff(degree)] {
            *c = -c.clone();
        }
    }
}

/// Typed lane-major view of one [`FamilyGroup`].
struct Lanes<'a, T> {
    n: usize,
    rank: usize,
    ids: &'a [u64],
    shifts: &'a [Complex<T>],
    masses: &'a [Complex<T>],
    scales: &'a [Complex<T>],
    s_matrix: &'a [Complex<T>],
    coefficients: &'a [Complex<T>],
}

/// Inputs of one engine call, lane-major in the bucket's lane order.
struct BucketInput<T> {
    n: usize,
    rank: usize,
    ids: Vec<u64>,
    shifts: Vec<Complex<T>>,
    masses: Vec<Complex<T>>,
    scales: Vec<Complex<T>>,
    s_matrix: Vec<Complex<T>>,
    coefficients: Vec<Complex<T>>,
}

impl<T: Carrier> BucketInput<T> {
    fn gather(
        groups: &[Lanes<'_, T>],
        n: usize,
        rank: usize,
        members: &[(usize, usize)],
        frame: Option<ReductionFrame>,
    ) -> Self {
        let width = ncoeff(rank);
        let lanes = members.len();
        let mut input = Self {
            n,
            rank,
            ids: Vec::with_capacity(lanes),
            shifts: Vec::with_capacity(lanes * 4 * n),
            masses: Vec::with_capacity(lanes * n),
            scales: Vec::with_capacity(lanes),
            s_matrix: Vec::with_capacity(lanes * n * n),
            coefficients: Vec::with_capacity(lanes * width),
        };
        let identity: Vec<_> = (0..n).collect();
        let order = frame.map_or(identity, |f| f.order(n));
        let reverse = frame.is_some_and(|f| f.reverse);
        let shift = frame.is_some_and(|f| f.shift);
        let scale = frame
            .is_some_and(|f| f.rescale)
            .then(|| {
                members
                    .first()
                    .map(|&(g, _)| rescaling(&groups[g].shifts[0]))
            })
            .flatten();
        let scale_squared = scale.clone().map(|x| x.clone() * &x);
        let scaled = |v: Complex<T>, factor: &Option<Complex<T>>| match factor {
            Some(x) => v * x,
            None => v,
        };
        for &(group, row) in members {
            let lanes = &groups[group];
            let shifts = &lanes.shifts[row * 4 * n..(row + 1) * 4 * n];
            let vector = |i: usize| -> [Complex<T>; 4] {
                std::array::from_fn(|mu| shifts[4 * i + mu].clone())
            };
            let origin = vector(order[0]);
            for &i in &order {
                for (mu, v) in vector(i).into_iter().enumerate() {
                    let v = if shift { v - &origin[mu] } else { v };
                    input
                        .shifts
                        .push(scaled(if reverse { -v } else { v }, &scale));
                }
            }
            let masses = &lanes.masses[row * n..(row + 1) * n];
            input.masses.extend(
                order
                    .iter()
                    .map(|&i| scaled(masses[i].clone(), &scale_squared)),
            );
            input
                .scales
                .push(scaled(lanes.scales[row].clone(), &scale_squared));
            let s = &lanes.s_matrix[row * n * n..(row + 1) * n * n];
            for &i in &order {
                input.s_matrix.extend(
                    order
                        .iter()
                        .map(|&j| scaled(s[i * n + j].clone(), &scale_squared)),
                );
            }
            let declared = ncoeff(lanes.rank);
            let source = &lanes.coefficients[row * declared..row * declared + width];
            let start = input.coefficients.len();
            if shift {
                let d = origin.map(|v| -v);
                input
                    .coefficients
                    .extend(shift_coefficients(source, rank, &d));
            } else {
                input.coefficients.extend_from_slice(source);
            }
            if reverse {
                reverse_coefficients(&mut input.coefficients[start..], rank);
            }
            if let Some(x) = &scale {
                // N'(q') = N(q'/x): degree-d coefficients times x^-d.
                let inverse = x.one() / x;
                let mut power = inverse.clone();
                for degree in 1..=rank {
                    for c in &mut input.coefficients
                        [start + block_offset(degree)..start + ncoeff(degree)]
                    {
                        *c = c.clone() * &power;
                    }
                    power = power * &inverse;
                }
            }
            input.ids.push(lanes.ids[row]);
        }
        input
    }
}

impl<T: Carrier> BucketInput<T> {
    /// The lanes `keep` selects, in order.
    fn select(&self, keep: &[bool]) -> Self {
        let (n, width) = (self.n, ncoeff(self.rank));
        let lanes = keep.iter().filter(|&&k| k).count();
        let mut input = Self {
            n,
            rank: self.rank,
            ids: Vec::with_capacity(lanes),
            shifts: Vec::with_capacity(lanes * 4 * n),
            masses: Vec::with_capacity(lanes * n),
            scales: Vec::with_capacity(lanes),
            s_matrix: Vec::with_capacity(lanes * n * n),
            coefficients: Vec::with_capacity(lanes * width),
        };
        for lane in (0..self.ids.len()).filter(|&lane| keep[lane]) {
            input.ids.push(self.ids[lane]);
            input
                .shifts
                .extend_from_slice(&self.shifts[lane * 4 * n..(lane + 1) * 4 * n]);
            input
                .masses
                .extend_from_slice(&self.masses[lane * n..(lane + 1) * n]);
            input.scales.push(self.scales[lane].clone());
            input
                .s_matrix
                .extend_from_slice(&self.s_matrix[lane * n * n..(lane + 1) * n * n]);
            input
                .coefficients
                .extend_from_slice(&self.coefficients[lane * width..(lane + 1) * width]);
        }
        input
    }
}

/// The rescaling factor [`RESCALING`] in the carrier of `like`.
fn rescaling<T: Carrier>(like: &Complex<T>) -> Complex<T> {
    let zero = like.re.zero();
    let (numerator, denominator) = RESCALING;
    Complex::new(
        zero.from_i64(numerator) / &zero.from_i64(denominator),
        zero.clone(),
    )
}

/// Lanes `(group, row)` per (propagators, true rank), in a fixed order.
type Buckets = BTreeMap<(usize, usize), Vec<(usize, usize)>>;

/// Status and raw Laurent coefficients of one lane (and its UV residue in
/// [`Poles::Split`]).
struct Outcome<T> {
    status: String,
    value: Option<[Complex<T>; 3]>,
    uv: Option<Complex<T>>,
}

trait BucketEngine<T> {
    fn reduce(&mut self, input: &BucketInput<T>) -> ExecutionResult<Vec<Outcome<T>>>;
    /// The carrier's working precision in binary digits (the UV residue
    /// lifts its inputs to it).
    fn precision(&self) -> u32;
}

/// Reduce every lane of `groups` (optionally in `frame`); outcomes per group
/// in row order.
fn reduce_groups<T: Carrier, E: BucketEngine<T>>(
    engine: &mut E,
    groups: &[Lanes<'_, T>],
    buckets: &Buckets,
    frame: Option<ReductionFrame>,
    poles: Poles,
) -> ExecutionResult<Vec<Vec<Option<Outcome<T>>>>> {
    let mut outcomes: Vec<Vec<Option<Outcome<T>>>> = groups
        .iter()
        .map(|g| (0..g.ids.len()).map(|_| None).collect())
        .collect();
    let precision = engine.precision();
    for (&(n, rank), members) in buckets {
        let input = BucketInput::gather(groups, n, rank, members, frame);
        // A rescaled frame's integrals are x^(4-2N) times the family's.
        let undo = frame.filter(|f| f.rescale).map(|_| {
            let x = rescaling(&input.shifts[0]);
            let inverse = x.one() / &x;
            (0..2 * n).fold(x.one(), |power, _| power * &x)
                * &(0..4).fold(x.one(), |power, _| power * &inverse)
        });
        let width = ncoeff(rank);
        let residues = (poles == Poles::Split).then(|| UvResidues::new(n, rank));
        for (lane, (&(group, row), mut outcome)) in
            members.iter().zip(engine.reduce(&input)?).enumerate()
        {
            if let (Some(residues), true) = (&residues, outcome.value.is_some()) {
                outcome.uv = Some(residues.residue(
                    &input.shifts[lane * 4 * n..(lane + 1) * 4 * n],
                    &input.masses[lane * n..(lane + 1) * n],
                    &input.coefficients[lane * width..(lane + 1) * width],
                    precision,
                ));
            }
            if let Some(factor) = &undo {
                if let Some(value) = &mut outcome.value {
                    for v in value.iter_mut() {
                        *v = v.clone() * factor;
                    }
                }
                if let Some(uv) = &mut outcome.uv {
                    *uv = uv.clone() * factor;
                }
            }
            outcomes[group][row] = Some(outcome);
        }
    }
    Ok(outcomes)
}

fn buckets<T: Carrier>(groups: &[Lanes<'_, T>]) -> ExecutionResult<Buckets> {
    let mut buckets = BTreeMap::<_, Vec<_>>::new();
    for (index, lanes) in groups.iter().enumerate() {
        let width = ncoeff(lanes.rank);
        for row in 0..lanes.ids.len() {
            let rank = true_rank(
                &lanes.coefficients[row * width..(row + 1) * width],
                lanes.rank,
            );
            if rank > lanes.n + 2 {
                return Err(error(format!(
                    "numerator rank {rank} exceeds propagators + 2 = {}",
                    lanes.n + 2
                )));
            }
            buckets
                .entry((lanes.n, rank))
                .or_default()
                .push((index, row));
        }
    }
    Ok(buckets)
}

fn output_batches<T: BatchCarrier>(
    groups: &[Lanes<'_, T>],
    outcomes: Vec<Vec<Option<Outcome<T>>>>,
    poles: Poles,
    bits: u32,
) -> ExecutionResult<Vec<ReductionBatch>> {
    let columns = poles.width();
    groups
        .iter()
        .zip(outcomes)
        .map(|(lanes, rows)| {
            let mut values = Vec::with_capacity(shape(rows.len(), columns)?);
            let mut ids = Vec::with_capacity(rows.len());
            let mut statuses = Vec::with_capacity(rows.len());
            for (&id, outcome) in lanes.ids.iter().zip(rows) {
                let outcome = outcome.ok_or_else(|| error("lane was not reduced"))?;
                if let Some(value) = outcome.value {
                    ids.push(id);
                    values.extend(value.into_iter().take(columns.min(3)));
                    if poles == Poles::Split {
                        values.push(
                            outcome
                                .uv
                                .ok_or_else(|| error("lane lacks its UV residue"))?,
                        );
                    }
                }
                statuses.push((id, outcome.status));
            }
            Ok(ReductionBatch {
                values: Batch::typed(values, ids, columns, bits)?,
                statuses,
            })
        })
        .collect()
}

/// Binary64 engine: per-(n, rank) `FastAmplitude`s, upstream tensor
/// numerators and cached native masters; lanes that reach one of their
/// absolute thresholds go to the generic binary64 engine (module docs).
struct FastEngine {
    amplitudes: HashMap<(usize, usize), FastAmplitude>,
    numerators: HashMap<usize, TensorNumerator>,
    masters: MasterCache<OneLoopNativeMasters>,
    results: Vec<FastResult>,
    statuses: Vec<FastStatus>,
    /// The largest absolute threshold of the fast path, in model units.
    threshold: f64,
    /// The exact engine of the lanes at or below `threshold`, built on first
    /// use.
    exact: Option<GenericEngine<f64>>,
    /// Lanes reduced by `exact` since the last [`V2Reducer::take_exact_lanes`].
    exact_lanes: u64,
}

impl FastEngine {
    fn new(cache_masters: bool) -> ExecutionResult<Self> {
        Ok(Self {
            amplitudes: HashMap::new(),
            numerators: HashMap::new(),
            masters: MasterCache::new(OneLoopNativeMasters::new().map_err(error)?, cache_masters),
            results: Vec::new(),
            statuses: Vec::new(),
            threshold: fast_engine_threshold(),
            exact: None,
            exact_lanes: 0,
        })
    }

    /// Whether each lane of `input` reaches the fast path's thresholds.
    fn routed(&self, input: &BucketInput<f64>) -> Vec<bool> {
        let n = input.n;
        (0..input.ids.len())
            .map(|lane| {
                reaches_fast_threshold(
                    n,
                    &input.masses[lane * n..(lane + 1) * n],
                    &input.s_matrix[lane * n * n..(lane + 1) * n * n],
                    self.threshold,
                )
            })
            .collect()
    }
}

fn real(value: &Complex<f64>, name: &str) -> ExecutionResult<f64> {
    if value.im != 0. {
        return Err(error(format!("fast engine requires real {name}")));
    }
    Ok(value.re)
}

impl BucketEngine<f64> for FastEngine {
    fn precision(&self) -> u32 {
        f64::MANTISSA_DIGITS
    }

    fn reduce(&mut self, input: &BucketInput<f64>) -> ExecutionResult<Vec<Outcome<f64>>> {
        let routed = self.routed(input);
        if !routed.contains(&true) {
            return self.reduce_fast(input);
        }
        self.exact_lanes += routed.iter().filter(|&&r| r).count() as u64;
        let kept: Vec<bool> = routed.iter().map(|&r| !r).collect();
        let mut fast = if kept.contains(&true) {
            self.reduce_fast(&input.select(&kept))?
        } else {
            Vec::new()
        }
        .into_iter();
        // The generic engine parses and evaluates Symbolica expressions: it
        // runs under pyAmpliCol's library key on this thread (module docs,
        // 'Absolute thresholds'), whichever thread the executor reduces on.
        let mut exact = {
            let _unlock = pyamplicol::unlock();
            self.exact
                .get_or_insert_with(|| GenericEngine::new(0_f64))
                .reduce(&input.select(&routed))?
        }
        .into_iter();
        routed
            .iter()
            .map(|&r| {
                if r { exact.next() } else { fast.next() }
                    .ok_or_else(|| error("a reduction engine lost a lane"))
            })
            .collect()
    }
}

impl FastEngine {
    fn reduce_fast(&mut self, input: &BucketInput<f64>) -> ExecutionResult<Vec<Outcome<f64>>> {
        let (n, rank, lanes) = (input.n, input.rank, input.ids.len());
        let shifts = input
            .shifts
            .chunks_exact(4)
            .map(|v| -> ExecutionResult<RMom> {
                Ok([
                    real(&v[0], "shifts")?,
                    real(&v[1], "shifts")?,
                    real(&v[2], "shifts")?,
                    real(&v[3], "shifts")?,
                ])
            })
            .collect::<ExecutionResult<Vec<_>>>()?;
        let masses: Vec<_> = input.masses.iter().map(|z| C64::new(z.re, z.im)).collect();
        let scales = input
            .scales
            .iter()
            .map(|z| real(z, "scales"))
            .collect::<ExecutionResult<Vec<_>>>()?;
        let s_matrix = input
            .s_matrix
            .iter()
            .map(|z| real(z, "invariants"))
            .collect::<ExecutionResult<Vec<_>>>()?;
        let coefficients: Vec<_> = input
            .coefficients
            .iter()
            .map(|z| C64::new(z.re, z.im))
            .collect();
        let amplitude = match self.amplitudes.entry((n, rank)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(FastAmplitude::new(n, rank, FastOptions::default()).map_err(error)?)
            }
        };
        let numerator = self
            .numerators
            .entry(rank)
            .or_insert_with(|| TensorNumerator::new(rank, 0));
        numerator
            .set_coefficients(lanes, &coefficients)
            .map_err(error)?;
        self.results.clear();
        self.results.resize(lanes, FastResult::default());
        self.statuses.clear();
        self.statuses.resize(lanes, FastStatus::NotEvaluated);
        amplitude
            .evaluate(
                &FastInput {
                    lanes,
                    shifts: &shifts,
                    masses_squared: &masses,
                    scales_squared: &scales,
                    s_matrix: Some(&s_matrix),
                },
                numerator,
                &mut self.masters,
                &mut self.results,
                &mut self.statuses,
            )
            .map_err(error)?;
        Ok(self
            .statuses
            .iter()
            .zip(&self.results)
            .map(|(&status, result)| {
                if status == FastStatus::Success {
                    Outcome {
                        status: "success".into(),
                        value: Some(result.laurent.map(|z| Complex::new(z.re, z.im))),
                        uv: None,
                    }
                } else {
                    Outcome {
                        status: format!("{status:?}"),
                        value: None,
                        uv: None,
                    }
                }
            })
            .collect())
    }
}

/// Precision-generic engine: one Symbolica template numerator per
/// (n, rank), `Σ_k c_k q^α_k` over the exponent arrays of the tensor layout.
struct GenericEngine<T: Carrier>
where
    Complex<T>: Real + EvaluationDomain,
{
    reductions: HashMap<(usize, usize), Reductions<T>>,
    prototype: T,
    /// The masters' pole selection of every reduction plan.
    selection: PoleSelection,
}

impl<T: Carrier> GenericEngine<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn new(prototype: T) -> Self {
        Self {
            reductions: HashMap::new(),
            prototype,
            selection: PoleSelection::BOTH,
        }
    }

    fn select(&mut self, selection: PoleSelection) {
        if selection != self.selection {
            self.selection = selection;
            for reductions in self.reductions.values_mut() {
                reductions.set_pole_selection(selection);
            }
        }
    }
}

fn template(rank: usize) -> (String, Vec<String>) {
    let parameters: Vec<_> = (0..ncoeff(rank)).map(|k| format!("c{k}")).collect();
    let terms: Vec<_> = monomials(rank)
        .iter()
        .zip(&parameters)
        .map(|(alpha, name)| {
            let mut factors = vec![name.clone()];
            for (variable, &power) in ["q0", "q1", "q2", "q3"].iter().zip(alpha) {
                if power > 0 {
                    factors.push(format!("{variable}^{power}"));
                }
            }
            factors.join("*")
        })
        .collect();
    (terms.join("+"), parameters)
}

/// Lanes per generic-engine call. The engine's workspace grows with the lanes
/// of one call (about 18 MB per lane of a 7-point process at 256 bits: 17 GB
/// for the 968 lanes of four uu~ -> e-e+dd~g points in one bucket stream), so
/// buckets are reduced in chunks of this many lanes; lanes are independent,
/// and chunking leaves every value unchanged.
pub(super) const GENERIC_CHUNK_LANES: usize = 16;

impl<T: Carrier> BucketEngine<T> for GenericEngine<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn precision(&self) -> u32 {
        self.prototype.get_precision()
    }

    fn reduce(&mut self, input: &BucketInput<T>) -> ExecutionResult<Vec<Outcome<T>>> {
        let reductions = match self.reductions.entry((input.n, input.rank)) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let (numerator, parameters) = template(input.rank);
                let mut reductions = Reductions::new(
                    &[numerator, "0".into(), "0".into()],
                    &parameters,
                    input.n,
                    self.prototype.clone(),
                )?;
                reductions.set_pole_selection(self.selection);
                entry.insert(reductions)
            }
        };
        let shifts: Vec<_> = input
            .shifts
            .chunks_exact(4)
            .map(|v| FourMomentum::from([v[0].clone(), v[1].clone(), v[2].clone(), v[3].clone()]))
            .collect();
        let (n, width, lanes) = (input.n, ncoeff(input.rank), input.ids.len());
        let mut outcomes = Vec::with_capacity(lanes);
        for start in (0..lanes).step_by(GENERIC_CHUNK_LANES) {
            let end = (start + GENERIC_CHUNK_LANES).min(lanes);
            let reduced = reductions.evaluate_native(
                &input.ids[start..end],
                &shifts[start * n..end * n],
                &input.masses[start * n..end * n],
                &input.scales[start..end],
                &input.coefficients[start * width..end * width],
                Some(&input.s_matrix[start * n * n..end * n * n]),
            )?;
            outcomes.extend(reduced.into_iter().map(|lane| {
                Outcome {
                    status: lane
                        .status
                        .strip_prefix("epsilon sector 0: ")
                        .map_or(lane.status.clone(), str::to_owned),
                    value: lane.value.map(|v| [v.finite, v.single_pole, v.double_pole]),
                    uv: None,
                }
            }));
        }
        Ok(outcomes)
    }
}

enum Engines {
    Fast(Box<FastEngine>),
    Double(GenericEngine<f64>),
    DoubleDouble(GenericEngine<DoubleFloat>),
    Arbitrary(GenericEngine<Float>),
}

/// The v2 family reducer: true-rank buckets, cached masters and the
/// reduction-direction check. Master values are kept until
/// [`Self::clear_masters`]; call it once per batch of points.
pub struct V2Reducer {
    engines: Engines,
    precision_bits: u32,
}

impl V2Reducer {
    /// `engine` is `fast` (binary64 only) or `generic`. `cache_masters`
    /// applies to the fast engine; canonical master arguments are used
    /// either way.
    pub fn new(
        arithmetic: &str,
        precision: Option<u32>,
        engine: &str,
        cache_masters: bool,
    ) -> ExecutionResult<Self> {
        let (arithmetic, bits) = arithmetic_mode(arithmetic, precision)?;
        let engines = match (engine, arithmetic) {
            ("fast", Arithmetic::Double) => {
                Engines::Fast(Box::new(FastEngine::new(cache_masters)?))
            }
            ("fast", _) => return Err(error("fast engine supports only double arithmetic")),
            ("generic", Arithmetic::Double) => Engines::Double(GenericEngine::new(0_f64)),
            ("generic", Arithmetic::DoubleDouble) => {
                Engines::DoubleDouble(GenericEngine::new(DoubleFloat::from(0.0)))
            }
            ("generic", Arithmetic::Arbitrary) => {
                Engines::Arbitrary(GenericEngine::new(Float::with_val(bits, 0)))
            }
            _ => return Err(error("engine must be fast or generic")),
        };
        Ok(Self {
            engines,
            precision_bits: bits,
        })
    }

    pub fn precision_bits(&self) -> u32 {
        self.precision_bits
    }

    /// Master requests of the fast engine since the last call; `None` for
    /// the generic engine.
    pub fn take_master_statistics(&mut self) -> Option<MasterStatistics> {
        match &mut self.engines {
            Engines::Fast(engine) => Some(engine.masters.take_statistics()),
            _ => None,
        }
    }

    /// Lanes of the fast engine reduced by the generic binary64 engine since
    /// the last call because their inputs reach one of the fast path's
    /// absolute thresholds (module docs); 0 for the generic engines.
    pub fn take_exact_lanes(&mut self) -> u64 {
        match &mut self.engines {
            Engines::Fast(engine) => std::mem::take(&mut engine.exact_lanes),
            _ => 0,
        }
    }

    /// Number of distinct cached master rows.
    pub fn cached_masters(&self) -> usize {
        match &self.engines {
            Engines::Fast(engine) => engine.masters.len(),
            _ => 0,
        }
    }

    pub fn clear_masters(&mut self) {
        if let Engines::Fast(engine) = &mut self.engines {
            engine.masters.clear();
        }
    }

    /// Reduce every lane of every group, then optionally every lane again in
    /// `options.check`'s frame with the same cached masters.
    pub fn reduce(
        &mut self,
        groups: &[FamilyGroup<'_>],
        options: ReduceOptions,
    ) -> ExecutionResult<V2Output> {
        macro_rules! run {
            ($engine:expr, $variant:ident) => {{
                let groups = groups
                    .iter()
                    .map(|g| {
                        lanes(g, self.precision_bits, |b| match b.data() {
                            Data::$variant(values) => Some(values.as_slice()),
                            _ => None,
                        })
                    })
                    .collect::<ExecutionResult<Vec<_>>>()?;
                run($engine, &groups, options, self.precision_bits)
            }};
        }
        match &mut self.engines {
            Engines::Fast(_) if options.masters != PoleSelection::BOTH => Err(error(
                "the fast engine keeps every master pole (it drops scaleless bubbles); \
                 select master poles with the generic engine",
            )),
            Engines::Fast(engine) => run!(engine.as_mut(), Double),
            Engines::Double(engine) => {
                engine.select(options.masters);
                run!(engine, Double)
            }
            Engines::DoubleDouble(engine) => {
                engine.select(options.masters);
                run!(engine, DoubleDouble)
            }
            Engines::Arbitrary(engine) => {
                engine.select(options.masters);
                run!(engine, Arbitrary)
            }
        }
    }
}

fn run<T: BatchCarrier, E: BucketEngine<T>>(
    engine: &mut E,
    groups: &[Lanes<'_, T>],
    options: ReduceOptions,
    bits: u32,
) -> ExecutionResult<V2Output> {
    let buckets = buckets(groups)?;
    let direct = reduce_groups(engine, groups, &buckets, None, options.poles)?;
    let check = options
        .check
        .map(|frame| reduce_groups(engine, groups, &buckets, Some(frame), options.poles))
        .transpose()?;
    Ok(V2Output {
        direct: output_batches(groups, direct, options.poles, bits)?,
        check: check
            .map(|outcomes| output_batches(groups, outcomes, options.poles, bits))
            .transpose()?,
    })
}

fn lanes<'a, T>(
    group: &FamilyGroup<'a>,
    bits: u32,
    values: impl Fn(&'a Batch) -> Option<&'a [Complex<T>]>,
) -> ExecutionResult<Lanes<'a, T>> {
    let n = group.propagators;
    if n == 0 {
        return Err(error("a family needs at least one propagator"));
    }
    let batches = [
        group.shifts,
        group.masses_squared,
        group.scales_squared,
        group.s_matrix,
    ];
    for batch in batches {
        group.coefficients.same_rows(batch)?;
    }
    for (batch, columns) in batches.iter().zip([4 * n, n, 1, n * n]) {
        if batch.columns() != columns {
            return Err(error("family batch dimensions differ"));
        }
    }
    if group.coefficients.precision_bits() != bits {
        return Err(error("batch arithmetic or precision differs"));
    }
    let typed = |batch: &'a Batch| {
        values(batch).ok_or_else(|| error("batch arithmetic or precision differs"))
    };
    Ok(Lanes {
        n,
        rank: tensor_rank(group.coefficients.columns())?,
        ids: group.coefficients.ids(),
        shifts: typed(group.shifts)?,
        masses: typed(group.masses_squared)?,
        scales: typed(group.scales_squared)?,
        s_matrix: typed(group.s_matrix)?,
        coefficients: typed(group.coefficients)?,
    })
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
        assert_eq!(true_rank(&coefficients, rank), 3);
        let mut lower = coefficients.clone();
        lower[block_offset(3)..].fill(Complex::new(0., 0.));
        assert_eq!(true_rank(&lower, rank), 2);
        assert_eq!(tensor_rank(ncoeff(4)).unwrap(), 4);
        assert!(tensor_rank(7).is_err());
    }

    /// A 256-bit generic engine of tadpole lanes `∫ c / (q² - m²)`.
    fn tadpoles(masses: &[Float], scale: &Float) -> (GenericEngine<Float>, BucketInput<Float>) {
        let zero = Float::with_val(256, 0);
        let complex = |x: &Float| Complex::new(x.clone(), zero.clone());
        let lanes = masses.len();
        let engine = GenericEngine::new(zero.clone());
        let input = BucketInput {
            n: 1,
            rank: 0,
            ids: (0..lanes as u64).collect(),
            shifts: vec![complex(&zero); 4 * lanes],
            masses: masses.iter().map(complex).collect(),
            scales: vec![complex(scale); lanes],
            s_matrix: vec![complex(&zero); lanes],
            coefficients: vec![complex(&Float::with_val(256, 1)); lanes],
        };
        (engine, input)
    }

    #[test]
    fn imprecise_float_masters_fail_their_lane() {
        // numerica tracks significant bits: a squared mass known to 53 bits
        // gives an A0 below `required_tracked_bits(256)` = 64 tracked bits,
        // which the master provider reports as `ImpreciseResult`; the lane
        // fails instead of returning a finite but unestablished value.
        let scale = Float::with_val(256, 1_000_000);
        let masses = [Float::with_val(256, 29929), Float::with_val(53, 29929)];
        let (mut engine, input) = tadpoles(&masses, &scale);
        let outcomes = engine.reduce(&input).unwrap();
        assert_eq!(outcomes[0].status, "success");
        assert!(outcomes[0].value.is_some());
        assert!(
            outcomes[1].status.contains("ImpreciseResult"),
            "{}",
            outcomes[1].status
        );
        assert!(outcomes[1].value.is_none());
    }

    #[test]
    fn generic_chunks_leave_values_unchanged() {
        let scale = Float::with_val(256, 1_000_000);
        let masses: Vec<_> = (0..GENERIC_CHUNK_LANES + 3)
            .map(|k| Float::with_val(256, 29929 + 7 * k as i64))
            .collect();
        let (mut engine, input) = tadpoles(&masses, &scale);
        let chunked = engine.reduce(&input).unwrap();
        for (k, mass) in masses.iter().enumerate() {
            let (mut single, one) = tadpoles(std::slice::from_ref(mass), &scale);
            let alone = single.reduce(&one).unwrap();
            assert_eq!(alone[0].status, "success");
            assert_eq!(chunked[k].value, alone[0].value, "lane {k}");
        }
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
