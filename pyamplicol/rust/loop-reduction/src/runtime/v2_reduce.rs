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
//! On the pinned NinjaRS (main b3a29c85) the fast path compares inputs with
//! absolute constants in the model's units: `OneLoopNativeMasters` sets
//! squared masses and invariants with `|re| + |im| < 1e-9` to zero and a
//! real invariant within `1e-9` of a real squared mass to that mass
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
//! executor's on-shell external invariants), never route. NinjaRS #9 (in the
//! pinned main) adds `OneLoopNativeMasters::with_onshell_threshold(None)`,
//! which would remove the masters' part (not adopted: the fast engine keeps
//! the default); the engine's `infrared_eps` part stays a routing rule.
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
//! [`RESCALING`](super::frames::RESCALING) factor `x` (shifts by `x`, squared masses, scale and
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
//! from the lane's own inputs (NinjaRS's [`TensorUvResidues`], also in the check's frame);
//! the combined columns are those of [`Poles::All`] bit for bit, so the
//! finite part never depends on the selection. NinjaRS 41a4c00 also selects
//! UV or IR residues at the masters (`ReduceOptions::masters`, oneloopmaster's
//! `PoleSelection`): its generic engine retains the scaleless bubbles that
//! carry them, its fast engine refuses a selected provider, and so does this
//! adapter. The generic engine's selection is the oracle of the residue.

use super::frames::FrameMap;
pub use super::frames::{ReductionFrame, shift_coefficients};
use super::master_cache::{MasterCache, MasterStatistics};
use super::reduction::{ReductionBatch, Reductions};
use super::*;
use crate::FourMomentum;
use ninjars::fast::TensorUvResidues;
use ninjars::fast::tensor_layout::{block_offset, monomials, ncoeff};
use ninjars::fast::{
    C64, FastAmplitude, FastInput, FastOptions, FastRefinementSource, FastResult, FastStatus, RMom,
    TensorNumerator,
};
use ninjars_symbolica::OneLoopNativeMasters;
pub use ninjars_symbolica::PoleSelection;
pub use rusticol_core::loop_execution::v2::Poles;
use rusticol_core::loop_execution::v2::{RescueCarrier, TrackedPrecision, TrackedSums};
use std::collections::{BTreeMap, HashMap};
use symbolica::{domains::float::Real, evaluate::EvaluationDomain};

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

/// The fast engine's options: NinjaRS's defaults (original Ninja's), or in
/// the UNCHECKED mode (m-lit lit-unchecked, `reduction_policy::
/// DEFAULT_UNCHECKED` off; DECISIONS 2026-11-08 22:34 (1)) zero stability
/// thresholds, `FastOptions { stability_threshold: 0, relative_tolerance: 0
/// }`: Ninja's stability test `relative(x, s)` becomes `x >= 0 && x > 0`, so
/// a lane fails only where a tested quantity is exactly zero or NaN (a
/// Euclidean-plane cut basis, `FastStatus::Unstable`), by the box
/// mu-expansion's conditioning screen (kept), or as `NonFinite` /
/// `MasterFailed`; an `Unstable` lane's status text names the test and its
/// numbers (`super::lane_status`). `infrared_eps` keeps its meaning (a routing rule:
/// [`fast_engine_threshold`] reads the default).
pub fn fast_options(unchecked: bool) -> FastOptions {
    let defaults = FastOptions::default();
    if unchecked {
        // Reduction checks off: no cancellation trigger and no rescue
        // (`FastOptions::cancellation_rescue`), every value the binary64
        // reduction's bit for bit.
        FastOptions {
            stability_threshold: 0.0,
            relative_tolerance: 0.0,
            cancellation_rescue: false,
            ..defaults
        }
    } else {
        // Reduction checks on: the cancellation rescue on explicitly (NinjaRS
        // defaults it off for callers without a test of their own), at
        // NinjaRS's default target unless the caller sets one
        // ([`fast_options_at`]).
        FastOptions {
            cancellation_rescue: true,
            ..defaults
        }
    }
}

/// Whether pyAmpliCol's best-effort pass keeps NinjaRS's cancellation
/// rescue on ([`fast_options_best_effort`]). On: the best-effort pass is the
/// unchecked reduction (zero stability thresholds) with the rescue, so a
/// lane the checked rescue refused is served from its result slot
/// (`FastEngine::reduce_lanes`: NinjaRS keeps the better of the binary64
/// and the re-reduced values there), and the lanes the rescue meets the
/// target on with their re-reduced values; the generic engines' ladder does
/// the same with a refused lane's best value (`GenericEngine::set_best_effort`).
/// Off: the plain unchecked reduction (binary64 values).
pub const RESCUE_IN_BEST_EFFORT: bool = true;

/// The fast engine's options in pyAmpliCol's BEST-EFFORT pass
/// ([`V2Reducer::set_best_effort_pass`]): the unchecked mode's zero
/// stability thresholds (`fast_options(true)`), with the cancellation rescue
/// at `lane_target` when [`RESCUE_IN_BEST_EFFORT`] is on.
pub fn fast_options_best_effort(lane_target: f64) -> FastOptions {
    FastOptions {
        cancellation_rescue: RESCUE_IN_BEST_EFFORT,
        rescue_target: RESCUE_IN_BEST_EFFORT.then_some(lane_target),
        ..fast_options(true)
    }
}

/// [`fast_options`] with the cancellation rescue's per-lane target
/// `lane_target` (`FastOptions::rescue_target`;
/// `reduction_policy::rescue_lane_target` of the call's target). Unchecked: no rescue, the target unused.
pub fn fast_options_at(unchecked: bool, lane_target: f64) -> FastOptions {
    FastOptions {
        rescue_target: (!unchecked).then_some(lane_target),
        ..fast_options(unchecked)
    }
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

/// The tracked-precision test at `working` bits
/// (`rusticol_core::loop_execution::v2::TrackedPrecision`): NinjaRS's rule
/// for its master integrals ([`ninjars_symbolica::required_tracked_bits`]:
/// `max(53, working / 4)`, at most `working / 2`), so that a point keeps at
/// least the precision its masters must keep.
pub fn tracked_rule(working: u32) -> TrackedPrecision {
    TrackedPrecision {
        working,
        required: ninjars_symbolica::required_tracked_bits(working),
    }
}

/// The tracked-precision test of the lanes `lanes` (each its Laurent
/// coefficients) at `working` bits ([`tracked_rule`]): their point's
/// failure, or None.
pub fn tracked_refusal<'a>(
    lanes: impl IntoIterator<Item = &'a [Complex<Float>]>,
    working: u32,
) -> Option<String> {
    let mut sums = TrackedSums::default();
    for lane in lanes {
        sums.add(lane, working);
    }
    sums.refusal(tracked_rule(working))
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

/// The rescue-precision kinematics of a [`V2Reducer::reduce_refined`]
/// call's lanes, on demand (pyAmpliCol's
/// `rusticol_core::loop_execution::v2::RefinementSource`; NinjaRS's
/// `FastRefinementSource`): the low parts of lane `row` of group `group`
/// written into `shifts` (`n`), `s_matrix` (`n²`) and `masses_squared`
/// (`n`), zeroed on entry; `false`: none for this lane (NinjaRS's own
/// repair serves it). Asked by the fast engine's cancellation rescue for the
/// lanes it flags only (direct frame).
pub trait GroupRefinementSource {
    fn lane(
        &self,
        group: usize,
        row: usize,
        shifts: &mut [RMom],
        s_matrix: &mut [f64],
        masses_squared: &mut [C64],
    ) -> bool;
}

/// Raw Laurent integrals `[finite, 1/ε, 1/ε²]` (finite only without poles),
/// one batch per input group, in the input row order.
pub struct V2Output {
    pub direct: Vec<ReductionBatch>,
    pub check: Option<Vec<ReductionBatch>>,
    /// The direct frame's lanes `(group, row)` the engine re-reduced in a
    /// higher carrier within its own reduction (the fast engine's
    /// cancellation rescue), with that carrier and the re-reduction's
    /// relative error estimate.
    pub rescued: Vec<(usize, usize, RescueCarrier, f64)>,
    /// The lanes `(group, row)` served without accuracy evidence, in either
    /// frame, with the reason ([`UNCERTIFIED_BELOW_RESOLUTION`],
    /// [`UNCERTIFIED_OUTSIDE_DOMAIN`]).
    pub uncertified: Vec<(usize, usize, &'static str)>,
    /// Of the lanes of [`Self::uncertified`], those whose served value comes
    /// with an absolute error bound per Laurent coefficient `[finite, 1/eps,
    /// 1/eps^2]` (the largest over the frames that served the lane
    /// uncertified): the fast engine's double-double re-reduction's own
    /// absolute estimate of a lane consistent with zero, where credible
    /// (finite, normal and positive). An uncertified lane absent here has
    /// no bound a check may use: one outside NinjaRS's rescue domain keeps
    /// its binary64 value, and the generic engine's ladder records give none
    /// (their rung bound is formed in f64, can overflow to a reported zero
    /// and can belong to another rung than the value served). A check that
    /// reads such a lane's value tests its residue together with these
    /// bounds, and is inconclusive where one is missing.
    pub uncertified_bounds: Vec<(usize, usize, [f64; 3])>,
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
    /// Each lane's `(group, row)` in the reduce call (empty in tests that
    /// build an input by hand): what a [`GroupRefinementSource`] is asked
    /// by.
    origins: Vec<(usize, usize)>,
}

impl<T: Carrier> BucketInput<T> {
    fn gather(
        groups: &[Lanes<'_, T>],
        n: usize,
        rank: usize,
        members: &[(usize, usize)],
        frame: Option<ReductionFrame>,
        prototype: &T,
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
            origins: Vec::with_capacity(lanes),
        };
        let map = FrameMap::new(frame, n, prototype);
        for &(group, row) in members {
            let lanes = &groups[group];
            let origin = map.shifts(
                &lanes.shifts[row * 4 * n..(row + 1) * 4 * n],
                &mut input.shifts,
            );
            map.masses(&lanes.masses[row * n..(row + 1) * n], &mut input.masses);
            input.scales.push(map.squared(&lanes.scales[row]));
            map.s_matrix(
                &lanes.s_matrix[row * n * n..(row + 1) * n * n],
                &mut input.s_matrix,
            );
            let declared = ncoeff(lanes.rank);
            let source = &lanes.coefficients[row * declared..row * declared + width];
            map.coefficients(source, rank, &origin, &mut input.coefficients);
            input.ids.push(lanes.ids[row]);
            input.origins.push((group, row));
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
            origins: Vec::with_capacity(if self.origins.is_empty() { 0 } else { lanes }),
        };
        for lane in (0..self.ids.len()).filter(|&lane| keep[lane]) {
            if !self.origins.is_empty() {
                input.origins.push(self.origins[lane]);
            }
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

/// Lanes `(group, row)` per (propagators, true rank), in a fixed order.
type Buckets = BTreeMap<(usize, usize), Vec<(usize, usize)>>;

/// Status and raw Laurent coefficients of one lane (and its UV residue in
/// [`Poles::Split`]).
struct Outcome<T> {
    status: String,
    value: Option<[Complex<T>; 3]>,
    uv: Option<Complex<T>>,
    /// The carrier the engine re-reduced the lane in within its own
    /// reduction (the fast engine's cancellation rescue), with the
    /// re-reduction's own relative error estimate, if it did.
    rescued: Option<(RescueCarrier, f64)>,
    /// Why the lane's value, though served, is no accuracy evidence
    /// ([`UNCERTIFIED_BELOW_RESOLUTION`], [`UNCERTIFIED_OUTSIDE_DOMAIN`]):
    /// its point's accuracy is not established.
    uncertified: Option<&'static str>,
    /// With `uncertified`, the absolute error bound of the served value per
    /// Laurent coefficient, where the engine states one
    /// ([`V2Output::uncertified_bounds`]).
    bound: Option<[f64; 3]>,
}

/// The reason of a lane the generic engine serves below the resolution of
/// its own terms (NinjaRS's `BelowResolution`): its value is returned, its
/// relative accuracy is not certified (a symmetry zero and a lane whose
/// terms cancel beyond every rung look alike).
pub const UNCERTIFIED_BELOW_RESOLUTION: &str =
    "a reduction lane vanishes within the resolution of its own terms (no symmetry proof)";
/// The reason of a lane NinjaRS's rescue measured above its target but
/// could not take without refined kinematics (an on-shell massive leg), its
/// binary64 value kept for a caller that supplies none.
pub const UNCERTIFIED_OUTSIDE_DOMAIN: &str =
    "a reduction lane is outside NinjaRS's rescue domain without refined kinematics";
/// The reasons of a lane refused by NinjaRS's rescue and by every carrier of
/// pyAmpliCol's lane rescue whose best value at the highest precision tried
/// is consistent with zero within its own absolute bound
/// ([`V2Reducer::set_vanishing`]): returned, its accuracy not established.
pub const UNCERTIFIED_VANISHING_DD: &str = "a reduction lane vanishes within the resolution of \
     its own terms at double-double; no symmetry proof";
pub const UNCERTIFIED_VANISHING_ARBITRARY: &str = "a reduction lane vanishes within the \
     resolution of its own terms at arbitrary precision; no symmetry proof";

trait BucketEngine<T> {
    /// `source`: the lanes' rescue-precision kinematics on demand (the
    /// fast engine's cancellation rescue; others ignore it).
    fn reduce(
        &mut self,
        input: &BucketInput<T>,
        source: Option<&dyn GroupRefinementSource>,
    ) -> ExecutionResult<Vec<Outcome<T>>>;
    /// The carrier's working precision in binary digits (the UV residue
    /// lifts its inputs to it).
    fn precision(&self) -> u32;
    /// A value of the carrier at its working precision (a rescaled frame's
    /// factor and undo, `FrameMap::new`).
    fn prototype(&self) -> T;
}

/// Reduce every lane of `groups` (optionally in `frame`); outcomes per group
/// in row order.
fn reduce_groups<T: Carrier, E: BucketEngine<T>>(
    engine: &mut E,
    groups: &[Lanes<'_, T>],
    buckets: &Buckets,
    frame: Option<ReductionFrame>,
    poles: Poles,
    source: Option<&dyn GroupRefinementSource>,
) -> ExecutionResult<Vec<Vec<Option<Outcome<T>>>>> {
    let mut outcomes: Vec<Vec<Option<Outcome<T>>>> = groups
        .iter()
        .map(|g| (0..g.ids.len()).map(|_| None).collect())
        .collect();
    let precision = engine.precision();
    let prototype = engine.prototype();
    for (&(n, rank), members) in buckets {
        let input = BucketInput::gather(groups, n, rank, members, frame, &prototype);
        // A rescaled frame's integrals are x^(4-2N) times the family's.
        let undo = FrameMap::new(frame, n, &prototype).undo();
        let width = ncoeff(rank);
        let residues = (poles == Poles::Split).then(|| TensorUvResidues::new(n, rank));
        for (lane, (&(group, row), mut outcome)) in members
            .iter()
            .zip(engine.reduce(&input, source)?)
            .enumerate()
        {
            if let (Some(residues), true) = (&residues, outcome.value.is_some()) {
                outcome.uv = Some(residues.residue_at(
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
            if let Some(refusal) = rusticol_core::loop_v2::bounds::rank_refusal(
                rusticol_core::loop_v2::bounds::Reducer::NinjaRs,
                lanes.n,
                rank,
            ) {
                return Err(error(refusal));
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
    /// Per (propagators, rank, unchecked, best-effort pass, lane target
    /// bits) the engine's amplitude: the options are part of the key, so a
    /// checked, an unchecked and a best-effort reduction, or two targets,
    /// never share one ([`fast_options_at`], [`fast_options_best_effort`]).
    amplitudes: HashMap<(usize, usize, bool, bool, u64), FastAmplitude>,
    /// The best-effort pass ([`V2Reducer::set_best_effort_pass`]).
    best_effort: bool,
    /// The unchecked mode of the following reductions ([`fast_options`]).
    unchecked: bool,
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
    /// Lanes NinjaRS's cancellation rescue re-reduced since the last
    /// [`V2Reducer::take_rescued_lanes`].
    rescued_lanes: u64,
    /// Lanes per call ([`FAST_CHUNK_LANES`]).
    chunk_lanes: usize,
    /// The cancellation rescue's per-lane target ([`V2Reducer::set_target`]).
    lane_target: f64,
}

impl FastEngine {
    fn new(cache_masters: bool) -> ExecutionResult<Self> {
        Ok(Self {
            amplitudes: HashMap::new(),
            unchecked: crate::reduction_policy::DEFAULT_UNCHECKED,
            numerators: HashMap::new(),
            masters: MasterCache::new(OneLoopNativeMasters::new().map_err(error)?, cache_masters),
            results: Vec::new(),
            statuses: Vec::new(),
            threshold: fast_engine_threshold(),
            exact: None,
            exact_lanes: 0,
            rescued_lanes: 0,
            best_effort: false,
            chunk_lanes: FAST_CHUNK_LANES,
            lane_target: crate::reduction_policy::rescue_lane_target(
                crate::reduction_policy::DEFAULT_TARGET,
            ),
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

    fn prototype(&self) -> f64 {
        0.0
    }

    fn reduce(
        &mut self,
        input: &BucketInput<f64>,
        source: Option<&dyn GroupRefinementSource>,
    ) -> ExecutionResult<Vec<Outcome<f64>>> {
        let routed = self.routed(input);
        if !routed.contains(&true) {
            return self.reduce_fast(input, source);
        }
        self.exact_lanes += routed.iter().filter(|&&r| r).count() as u64;
        let kept: Vec<bool> = routed.iter().map(|&r| !r).collect();
        let mut fast = if kept.contains(&true) {
            self.reduce_fast(&input.select(&kept), source)?
        } else {
            Vec::new()
        }
        .into_iter();
        // The generic engine parses and evaluates Symbolica expressions: it
        // runs under pyAmpliCol's library key on this thread (module docs,
        // 'Absolute thresholds'), whichever thread the executor reduces on.
        let (unchecked, lane_target, best_effort) =
            (self.unchecked, self.lane_target, self.best_effort);
        let mut exact = {
            let _unlock = pyamplicol::unlock();
            self.exact
                .get_or_insert_with(|| {
                    let mut engine = GenericEngine::new(0_f64);
                    engine.set_checked(!unchecked);
                    engine.set_cancellation_target(Some(lane_target));
                    engine.set_best_effort(best_effort);
                    engine
                })
                .reduce(&input.select(&routed), None)?
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

/// Lanes per fast-engine call (M-FINAL MF-8G-2: the evaluation base). The
/// engine's per-lane workspace (cut data, rows, staged results; about 0.1 MB
/// per lane at seven propagators) grows to the largest call and stays with
/// the executor between evaluations: a bucket chunk of 512 families is
/// 8,192 lanes per point chunk, about 1 GiB kept. Buckets are reduced in
/// calls of this many lanes; lanes are independent (the shared master cache
/// returns a key's value whichever call evaluated it), so chunking leaves
/// every value unchanged.
pub(super) const FAST_CHUNK_LANES: usize = 1024;

impl FastEngine {
    fn reduce_fast(
        &mut self,
        input: &BucketInput<f64>,
        source: Option<&dyn GroupRefinementSource>,
    ) -> ExecutionResult<Vec<Outcome<f64>>> {
        let (lanes, chunk) = (input.ids.len(), self.chunk_lanes.max(1));
        if lanes <= chunk {
            return self.reduce_lanes(input, source);
        }
        let mut outcomes = Vec::with_capacity(lanes);
        for start in (0..lanes).step_by(chunk) {
            let end = (start + chunk).min(lanes);
            let keep: Vec<bool> = (0..lanes)
                .map(|lane| (start..end).contains(&lane))
                .collect();
            outcomes.extend(self.reduce_lanes(&input.select(&keep), source)?);
        }
        Ok(outcomes)
    }

    fn reduce_lanes(
        &mut self,
        input: &BucketInput<f64>,
        source: Option<&dyn GroupRefinementSource>,
    ) -> ExecutionResult<Vec<Outcome<f64>>> {
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
        let key = (
            n,
            rank,
            self.unchecked,
            self.best_effort,
            self.lane_target.to_bits(),
        );
        let options = if self.best_effort {
            fast_options_best_effort(self.lane_target)
        } else {
            fast_options_at(self.unchecked, self.lane_target)
        };
        let amplitude = match self.amplitudes.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(FastAmplitude::new(n, rank, options).map_err(error)?)
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
        // The rescue-precision kinematics on demand (`FastRefinementSource`):
        // NinjaRS asks for the lanes it flags.
        let adapter = source
            .filter(|_| input.origins.len() == lanes)
            .map(|source| RefinementAdapter {
                origins: &input.origins,
                source,
            });
        amplitude
            .evaluate_refined_with(
                &FastInput {
                    lanes,
                    shifts: &shifts,
                    masses_squared: &masses,
                    scales_squared: &scales,
                    s_matrix: Some(&s_matrix),
                },
                adapter.as_ref().map(|a| a as &dyn FastRefinementSource),
                numerator,
                &mut self.masters,
                &mut self.results,
                &mut self.statuses,
            )
            .map_err(error)?;
        // NinjaRS's record of its rescue: per re-reduced lane its precision,
        // and whether it met the target (a miss is the lane's `Unstable`
        // status, typed).
        let mut rescued = vec![None; lanes];
        // Per refused lane whose slot holds its re-reduced value, that
        // re-reduction's absolute error estimate (the vanishing rule below).
        let mut refused_bound = vec![None; lanes];
        for (lane, rescue) in ninjars::fast::take_rescues() {
            if rescue.served
                && let Some(slot) = rescued.get_mut(lane)
            {
                *slot = Some((rescue_carrier(rescue.precision_bits), rescue.estimate));
            } else if !rescue.served
                && rescue.outcome == ninjars::fast::RescueOutcome::Missed
                && rescue.held_bits == rescue.precision_bits
                && let Some(slot) = refused_bound.get_mut(lane)
            {
                *slot = Some(rescue.absolute);
            }
        }
        // NinjaRS's own count (lanes re-reduced, with or without a
        // refinement): the statistics.
        self.rescued_lanes += ninjars::fast::take_rescued_lanes();
        // Taken after a successful evaluation only, which suffices: NinjaRS's
        // `FastAmplitude::evaluate` forgets the thread's earlier records on
        // entry (`clear_instabilities`), so those of a failed evaluation
        // never reach the next one.
        let records = super::lane_status::fast_instabilities(lanes);
        let best_effort = self.best_effort;
        let refined = source.is_some();
        Ok(self
            .statuses
            .iter()
            .zip(&self.results)
            .zip(&records)
            .zip(&rescued)
            .zip(&refused_bound)
            .map(
                |((((&status, result), record), &rescued), &refused_bound)| {
                    // The best-effort pass serves a lane the rescue refused from
                    // its result slot: the value NinjaRS keeps there for a refused
                    // lane (`CancellationRescue`: the better of the binary64 and
                    // the re-reduced values; `RescueOutsideDomain`, a flagged lane
                    // the rescue cannot take: the binary64 value).
                    let missed = status == FastStatus::Unstable
                        && record.as_ref().is_some_and(|r| {
                            matches!(
                                r.test,
                                ninjars::fast::InstabilityTest::CancellationRescue
                                    | ninjars::fast::InstabilityTest::RescueOutsideDomain
                            )
                        })
                        && result.laurent.iter().all(|z| z.is_finite());
                    // A caller that supplies no refined kinematics (the Python
                    // reference executor, the slice evaluation, a call without
                    // initial states) cannot give NinjaRS's rescue the point it
                    // needs for a lane outside its domain (an on-shell massive
                    // leg): such a lane keeps its binary64 value and status, as
                    // before the rescue existed, instead of failing there.
                    let unrefined_outside = !refined
                        && status == FastStatus::Unstable
                        && record.as_ref().is_some_and(|r| {
                            matches!(r.test, ninjars::fast::InstabilityTest::RescueOutsideDomain)
                        })
                        && result.laurent.iter().all(|z| z.is_finite());
                    // A lane the rescue refused whose re-reduced (double-double)
                    // value, held in its slot, is consistent with zero within
                    // that re-reduction's own absolute error estimate in every
                    // Laurent coefficient (`vanishes_within`): served at that
                    // value, uncertified, without pyAmpliCol's lane rescue (whose
                    // higher carriers could only confirm a zero or, for a lane
                    // that loses more than about 100 bits and is in truth not
                    // zero, recover it; a rescue policy escalates such a point,
                    // its levels never met by an uncertified result).
                    let vanishing = !best_effort
                        && missed
                        && record.as_ref().is_some_and(|r| {
                            matches!(r.test, ninjars::fast::InstabilityTest::CancellationRescue)
                        })
                        && refused_bound
                            .is_some_and(|bound| vanishes_within(&result.laurent, bound));
                    if status == FastStatus::Success
                        || (best_effort && missed)
                        || unrefined_outside
                        || vanishing
                    {
                        Outcome {
                            status: "success".into(),
                            value: Some(result.laurent.map(|z| Complex::new(z.re, z.im))),
                            uv: None,
                            rescued,
                            uncertified: if vanishing {
                                Some(UNCERTIFIED_VANISHING_DD)
                            } else {
                                unrefined_outside.then_some(UNCERTIFIED_OUTSIDE_DOMAIN)
                            },
                            // NinjaRS states one absolute estimate, of the
                            // finite part's terms: taken for every coefficient,
                            // where credible (`credible_bound`).
                            bound: refused_bound
                                .filter(|&b| vanishing && credible_bound(b))
                                .map(|b| [b; 3]),
                        }
                    } else {
                        Outcome {
                            status: super::lane_status::fast_status(status, record.as_ref()),
                            value: None,
                            uv: None,
                            rescued: None,
                            uncertified: None,
                            bound: None,
                        }
                    }
                },
            )
            .collect())
    }
}

/// Whether an absolute bound is credible as a check budget: finite, and a
/// normal positive number (a bound that underflowed to zero or into the
/// subnormal range cannot be told from one whose terms are nonzero; an exact
/// zero is not proven to come from exactly zero terms here).
fn credible_bound(bound: f64) -> bool {
    bound.is_finite() && bound >= f64::MIN_POSITIVE
}

/// Whether a refused lane's re-reduced Laurent coefficients are all
/// consistent with zero within `bound`, that re-reduction's own absolute
/// error estimate (NinjaRS's `Rescue::absolute`: the moduli of its terms at
/// their carriers' units, at least the box-Gram and mu-expansion losses).
/// All three coefficients are tested, whatever the pole selection (the fast
/// engine computes all three): a finite bound and finite values only.
fn vanishes_within(laurent: &[C64; 3], bound: f64) -> bool {
    bound.is_finite()
        && laurent.iter().all(|z| {
            let modulus = z.re.hypot(z.im);
            modulus.is_finite() && modulus <= bound
        })
}

/// A [`GroupRefinementSource`] as NinjaRS's `FastRefinementSource` for one
/// fast-engine call: lane `i` of the call is `origins[i]`; the lanes asked
/// NinjaRS asks for the lanes it re-reduces only.
struct RefinementAdapter<'a> {
    origins: &'a [(usize, usize)],
    source: &'a dyn GroupRefinementSource,
}

/// The lane-rescue carrier of NinjaRS's rescue precision in bits
/// (`Rescue::precision_bits`: 106 is double-double, more is MPFR at those
/// bits' decimal digits).
fn rescue_carrier(bits: u32) -> RescueCarrier {
    if bits <= 106 {
        RescueCarrier::DoubleDouble
    } else {
        RescueCarrier::Arbitrary((f64::from(bits) * std::f64::consts::LOG10_2).floor() as u32)
    }
}

impl FastRefinementSource for RefinementAdapter<'_> {
    fn lane(
        &self,
        lane: usize,
        shifts: &mut [RMom],
        s_matrix: &mut [f64],
        masses_squared: &mut [C64],
    ) -> bool {
        let Some(&(group, row)) = self.origins.get(lane) else {
            return false;
        };
        self.source
            .lane(group, row, shifts, s_matrix, masses_squared)
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
    /// NinjaRS's cancellation ladder on every plan (the checked mode; off in
    /// the UNCHECKED mode, [`V2Reducer::set_unchecked`]).
    checked: bool,
    /// The ladder's lane target ([`Self::set_cancellation_target`]).
    target: Option<f64>,
    /// pyAmpliCol's best-effort pass ([`Self::set_best_effort`]).
    best_effort: bool,
    /// The reason under which a lane the ladder refuses is returned when its
    /// best value is consistent with zero ([`V2Reducer::set_vanishing`]: the
    /// lane rescue's last carrier only); None: such lanes stay refused.
    vanishing: Option<&'static str>,
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
            checked: !crate::reduction_policy::DEFAULT_UNCHECKED,
            // The same per-lane target as the fast engine's cancellation
            // rescue (`reduction_policy::rescue_lane_target` of the call's
            // target; [`V2Reducer::set_target`] moves both).
            target: Some(crate::reduction_policy::rescue_lane_target(
                crate::reduction_policy::DEFAULT_TARGET,
            )),
            best_effort: false,
            vanishing: None,
        }
    }

    /// Whether the plans carry NinjaRS's cancellation ladder: the checked
    /// mode, and the best-effort pass (which serves a refused lane's best
    /// value, as the fast engine's best-effort pass does).
    fn ladder_on(&self) -> bool {
        self.checked || (self.best_effort && RESCUE_IN_BEST_EFFORT)
    }

    /// pyAmpliCol's best-effort pass ([`V2Reducer::set_best_effort_pass`]),
    /// following the fast engine's [`RESCUE_IN_BEST_EFFORT`]: when it is on,
    /// the ladder runs (also in the unchecked mode) and a lane it refuses
    /// (`Missed`, `Failed`) is served from its best available value (the
    /// ladder's `LaneRescue::best`: the smallest estimate among the base pass
    /// and its rungs); off (today), the pass is the unchecked reduction, as
    /// on the fast engine.
    fn set_best_effort(&mut self, on: bool) {
        if on != self.best_effort {
            let before = self.ladder_on();
            self.best_effort = on;
            if self.ladder_on() != before {
                self.reductions.clear();
            }
        }
    }

    /// The ladder's target relative accuracy per lane of every plan (`None`:
    /// NinjaRS's carrier default). Plans built later take it too.
    fn set_cancellation_target(&mut self, target: Option<f64>) {
        self.target = target;
        for reductions in self.reductions.values_mut() {
            reductions.set_cancellation_target(target);
        }
    }

    /// The checked mode of the following reductions: plans are rebuilt with
    /// or without NinjaRS's cancellation ladder (measurement included).
    fn set_checked(&mut self, checked: bool) {
        if checked != self.checked {
            let before = self.ladder_on();
            self.checked = checked;
            if self.ladder_on() != before {
                self.reductions.clear();
            }
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

/// A ladder record's own absolute bound per Laurent coefficient (the
/// record's finite bound, else its estimate times its term moduli), or None
/// without either.
fn zero_bound<T: NativeFloat>(record: &crate::LadderLaneRecord<T>) -> Option<[f64; 3]>
where
    Complex<T>: Real + EvaluationDomain,
{
    let bound = match (record.absolute_bound, record.term_moduli) {
        (Some(bound), _) if bound.iter().all(|b| b.is_finite()) => bound,
        (_, Some(moduli)) if record.estimate.is_finite() => moduli.map(|m| m * record.estimate),
        _ => return None,
    };
    bound.iter().all(|b| b.is_finite()).then_some(bound)
}

/// The bound ([`zero_bound`]) of a refused ladder record whose best value is
/// consistent with zero within it in every Laurent coefficient; None when
/// the lane is not eligible (no best value, no finite bound, or a value
/// outside it).
fn consistent_with_zero<T: NativeFloat>(record: &crate::LadderLaneRecord<T>) -> Option<[f64; 3]>
where
    Complex<T>: Real + EvaluationDomain,
{
    let best = record.best.as_ref()?;
    let bound = zero_bound(record)?;
    best.iter()
        .zip(bound)
        .all(|(z, b)| {
            let modulus = z.re.to_f64().hypot(z.im.to_f64());
            modulus.is_finite() && modulus <= b
        })
        .then_some(bound)
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

    fn prototype(&self) -> T {
        self.prototype.clone()
    }

    fn reduce(
        &mut self,
        input: &BucketInput<T>,
        _source: Option<&dyn GroupRefinementSource>,
    ) -> ExecutionResult<Vec<Outcome<T>>> {
        let ladder = self.ladder_on();
        let serve_best = self.best_effort && RESCUE_IN_BEST_EFFORT;
        let vanishing = self.vanishing;
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
                if ladder {
                    reductions.enable_cancellation_ladder()?;
                    reductions.set_cancellation_target(self.target);
                }
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
            // Chunk-local lane ids (the position in the chunk): the bucket's
            // ids number each family's rows from zero, so two families of one
            // chunk share ids; the ladder's records are matched by these.
            let local: Vec<u64> = (0..(end - start) as u64).collect();
            let reduced = reductions.evaluate_native(
                &local,
                &shifts[start * n..end * n],
                &input.masses[start * n..end * n],
                &input.scales[start..end],
                &input.coefficients[start * width..end * width],
                Some(&input.s_matrix[start * n * n..end * n * n]),
            )?;
            // NinjaRS's cancellation ladder (checked mode, best-effort pass):
            // a lane it re-reduced and served is recorded with the carrier of
            // its last rung (the point records, as the fast engine's
            // rescue); in the best-effort pass a lane it refused is served
            // from its best value.
            let mut rescued: Vec<Option<(RescueCarrier, f64)>> = vec![None; end - start];
            let mut best: Vec<Option<[Complex<T>; 3]>> = vec![None; end - start];
            let mut rescued_vanishing: Vec<Option<(RescueCarrier, f64)>> = vec![None; end - start];
            let mut zero_reason: Vec<Option<&'static str>> = vec![None; end - start];
            for record in reductions.take_ladder_records() {
                let Ok(row) = usize::try_from(record.id) else {
                    continue;
                };
                if row >= end - start {
                    continue;
                }
                if record.served && !record.rungs.is_empty() {
                    // One lane per epsilon sector: the highest carrier and the
                    // largest estimate of its sectors.
                    let carrier = rescue_carrier(record.precision_bits);
                    rescued[row] = Some(match rescued[row] {
                        Some((before, estimate)) => (
                            match (before, carrier) {
                                (RescueCarrier::Arbitrary(d), RescueCarrier::Arbitrary(e)) => {
                                    RescueCarrier::Arbitrary(d.max(e))
                                }
                                (RescueCarrier::Arbitrary(d), _) => RescueCarrier::Arbitrary(d),
                                _ => carrier,
                            },
                            estimate.max(record.estimate),
                        ),
                        None => (carrier, record.estimate),
                    });
                } else if !record.served
                    && record.sector == 0
                    && record.outcome == crate::LadderOutcome::BelowResolution
                {
                    // A vanishing lane (its value below the rung's resolution of
                    // its own terms: no relative accuracy can be stated, its
                    // absolute error is below the target times the base pass's
                    // rounding of its terms) is served at its rung value, not
                    // refused: a lane that cancels to zero by symmetry would
                    // otherwise fail at every carrier. Recorded as a rescue.
                    best[row] = record.best;
                    let carrier = rescue_carrier(record.precision_bits);
                    rescued_vanishing[row] = Some((carrier, record.estimate));
                } else if let Some(reason) = vanishing
                    && !record.served
                    && record.sector == 0
                    && consistent_with_zero(&record).is_some()
                {
                    // Refused by the fast rescue and every earlier carrier of
                    // the lane rescue, and here at the highest precision
                    // tried, with a best value within its own absolute bound
                    // of zero: returned, its accuracy not established.
                    best[row] = record.best;
                    zero_reason[row] = Some(reason);
                } else if !record.served && serve_best && record.sector == 0 {
                    // The engine's numerators live in epsilon sector 0 (the
                    // others are zero, `template`): its best value is the lane's.
                    best[row] = record.best;
                }
            }
            outcomes.extend(
                reduced
                    .into_iter()
                    .zip(rescued)
                    .zip(best)
                    .zip(rescued_vanishing)
                    .zip(zero_reason)
                    .map(|((((lane, rescued), best), vanishing), zero)| match best {
                        Some(value) if lane.value.is_none() => Outcome {
                            status: "success".into(),
                            value: Some(value),
                            uv: None,
                            uncertified: vanishing.map(|_| UNCERTIFIED_BELOW_RESOLUTION).or(zero),
                            // No check budget from the generic ladder's records:
                            // its rung bound is formed in f64 (a term modulus
                            // times 256 can overflow and read as zero) and can
                            // belong to another rung than the value served.
                            // The lane is served, its bound unavailable
                            // (`V2Output::uncertified_bounds`).
                            bound: None,
                            rescued: vanishing,
                        },
                        _ => Outcome {
                            status: lane
                                .status
                                .strip_prefix("epsilon sector 0: ")
                                .map_or(lane.status.clone(), str::to_owned),
                            value: lane.value.map(|v| [v.finite, v.single_pole, v.double_pole]),
                            uv: None,
                            rescued,
                            uncertified: None,
                            bound: None,
                        },
                    }),
            );
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
    /// [`Self::set_unchecked`].
    unchecked: bool,
    /// [`Self::set_target`].
    target: f64,
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
            unchecked: crate::reduction_policy::DEFAULT_UNCHECKED,
            target: crate::reduction_policy::DEFAULT_TARGET,
        })
    }

    /// The call's target accuracy (the user's `reduction_target`,
    /// `reduction_policy::DEFAULT_TARGET` by default): the fast engine's
    /// cancellation rescue re-reduces a lane whose measured error estimate
    /// exceeds `reduction_policy::rescue_lane_target(target)` (checked mode
    /// only; [`fast_options`]).
    pub fn set_target(&mut self, target: f64) {
        self.target = target;
        // One lane target for both engines: the fast engine's cancellation
        // rescue and the generic engines' cancellation ladder.
        let lane_target = crate::reduction_policy::rescue_lane_target(target);
        match &mut self.engines {
            Engines::Fast(engine) => {
                engine.lane_target = lane_target;
                if let Some(exact) = &mut engine.exact {
                    exact.set_cancellation_target(Some(lane_target));
                }
            }
            Engines::Double(engine) => engine.set_cancellation_target(Some(lane_target)),
            Engines::DoubleDouble(engine) => engine.set_cancellation_target(Some(lane_target)),
            Engines::Arbitrary(engine) => engine.set_cancellation_target(Some(lane_target)),
        }
    }

    /// The call's target accuracy ([`Self::set_target`]).
    pub fn target(&self) -> f64 {
        self.target
    }

    /// The reason under which the generic engines return a lane their
    /// ladder refuses but whose best value is consistent with zero within its
    /// own absolute bound (None: such lanes stay refused). pyAmpliCol's lane
    /// rescue sets it on its LAST carrier only, so that such a lane has been
    /// refused by the fast rescue and every earlier carrier first.
    pub fn set_vanishing(&mut self, reason: Option<&'static str>) {
        match &mut self.engines {
            Engines::Fast(engine) => {
                if let Some(exact) = &mut engine.exact {
                    exact.vanishing = reason;
                }
            }
            Engines::Double(engine) => engine.vanishing = reason,
            Engines::DoubleDouble(engine) => engine.vanishing = reason,
            Engines::Arbitrary(engine) => engine.vanishing = reason,
        }
    }

    /// The generic engine's cancellation ladder: its lane target and whether
    /// its plans carry it (`None` for the fast engine). Tests.
    #[cfg(test)]
    pub(crate) fn ladder_state(&self) -> Option<(Option<f64>, bool)> {
        match &self.engines {
            Engines::Fast(_) => None,
            Engines::Double(e) => Some((e.target, e.ladder_on())),
            Engines::DoubleDouble(e) => Some((e.target, e.ladder_on())),
            Engines::Arbitrary(e) => Some((e.target, e.ladder_on())),
        }
    }

    /// pyAmpliCol's best-effort pass (a point no configured alternative
    /// accepted, evaluated again for its best value, `on_failure=
    /// "best-effort"`): the fast engine reduces unchecked, with its
    /// cancellation rescue only when [`RESCUE_IN_BEST_EFFORT`] is on (then a
    /// lane whose rescue misses the target is served from its result slot,
    /// [`fast_options_best_effort`]); the generic engines follow the same
    /// constant (`GenericEngine::set_best_effort`). Off (the default)
    /// everywhere else; the user's unchecked mode is unaffected.
    pub fn set_best_effort_pass(&mut self, on: bool) {
        match &mut self.engines {
            Engines::Fast(engine) => {
                engine.best_effort = on;
                if let Some(exact) = &mut engine.exact {
                    exact.set_best_effort(on);
                }
            }
            Engines::Double(engine) => engine.set_best_effort(on),
            Engines::DoubleDouble(engine) => engine.set_best_effort(on),
            Engines::Arbitrary(engine) => engine.set_best_effort(on),
        }
    }

    pub fn precision_bits(&self) -> u32 {
        self.precision_bits
    }

    /// The unchecked mode of the following reductions (m-lit lit-unchecked,
    /// off by default): the fast engine's zero stability thresholds
    /// ([`fast_options`]; its amplitudes are kept per mode) and no
    /// cancellation rescue. The generic engines (double-double, MPFR, and the
    /// binary64 lanes routed to them) run NinjaRS's cancellation ladder in
    /// the checked mode only (its stability test, measured cancellation and
    /// re-reduction in a higher precision, `GenericEngine::set_checked`); in
    /// the unchecked mode their plans carry neither, and their values are the
    /// generic reduction's bit for bit. Either way pyAmpliCol's stability
    /// checks, its rescue stack and the per-point fallback's guard check the
    /// values.
    pub fn set_unchecked(&mut self, unchecked: bool) {
        self.unchecked = unchecked;
        match &mut self.engines {
            Engines::Fast(engine) => {
                engine.unchecked = unchecked;
                if let Some(exact) = &mut engine.exact {
                    exact.set_checked(!unchecked);
                }
            }
            Engines::Double(engine) => engine.set_checked(!unchecked),
            Engines::DoubleDouble(engine) => engine.set_checked(!unchecked),
            Engines::Arbitrary(engine) => engine.set_checked(!unchecked),
        }
    }

    /// The unchecked mode of the following reductions.
    pub fn unchecked(&self) -> bool {
        self.unchecked
    }

    /// The point-level tracked-precision test of this reducer's lanes
    /// (`rusticol_core::loop_execution::v2::TrackedPrecision`): the
    /// arbitrary-precision generic engine's `Float` lanes carry the precision
    /// numerica tracked through the reduction, against NinjaRS's rule for
    /// its master integrals at the working precision
    /// ([`ninjars_symbolica::required_tracked_bits`]). `None` for binary64
    /// and double-double (fixed precision, nothing tracked) and in the
    /// unchecked mode (its values are served untested).
    pub fn tracked_precision(&self) -> Option<TrackedPrecision> {
        match &self.engines {
            Engines::Arbitrary(engine) if !self.unchecked => Some(tracked_rule(engine.precision())),
            _ => None,
        }
    }

    /// The tracked-precision test ([`Self::tracked_precision`]) of the
    /// direct outcomes of one point's [`Self::reduce`] (its successful
    /// lanes, every group): the point's failure, or None (also when the
    /// reducer takes no test). The Python reference path's per-point test;
    /// the executor sums its lanes itself.
    pub fn tracked_refusal<'a>(
        &self,
        batches: impl IntoIterator<Item = &'a ReductionBatch>,
    ) -> Option<String> {
        let rule = self.tracked_precision()?;
        let mut sums = TrackedSums::default();
        for batch in batches {
            let columns = batch.values.columns().max(1);
            let Data::Arbitrary(values) = batch.values.data() else {
                continue;
            };
            // The successful lanes only, as the executor sums them: a row
            // with a value whose status failed is not served.
            let succeeded: std::collections::HashSet<u64> = batch
                .statuses
                .iter()
                .filter(|(_, status)| status == "success")
                .map(|&(id, _)| id)
                .collect();
            for (id, lane) in batch.values.ids().iter().zip(values.chunks_exact(columns)) {
                if succeeded.contains(id) {
                    sums.add(lane, rule.working);
                }
            }
        }
        sums.refusal(rule)
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

    /// Lanes the fast engine's cancellation rescue re-reduced since the
    /// last call (NinjaRS's `take_rescued_lanes`); 0 for the generic engines.
    pub fn take_rescued_lanes(&mut self) -> u64 {
        match &mut self.engines {
            Engines::Fast(engine) => std::mem::take(&mut engine.rescued_lanes),
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

    /// The carrier of the rescue-precision kinematics this reducer reads
    /// ([`Self::reduce_refined`]): the fast engine's cancellation rescue
    /// (NinjaRS `FastOptions::cancellation_rescue`), on in the checked mode
    /// only (`fast_options`), reads double-double, the lowest lane-rescue
    /// carrier above binary64
    /// (`rusticol_core::loop_execution::v2::rescue_carriers`); none in the
    /// unchecked mode and on the generic engines (their own rescue is not wired here).
    pub fn refinement_carrier(&self) -> Option<RescueCarrier> {
        match &self.engines {
            Engines::Fast(engine)
                if !self.unchecked || (engine.best_effort && RESCUE_IN_BEST_EFFORT) =>
            {
                super::v2_execution::rescue_carriers(self.precision_bits)
                    .first()
                    .copied()
            }
            _ => None,
        }
    }

    /// Reduce every lane of every group, then optionally every lane again in
    /// `options.check`'s frame with the same cached masters.
    pub fn reduce(
        &mut self,
        groups: &[FamilyGroup<'_>],
        options: ReduceOptions,
    ) -> ExecutionResult<V2Output> {
        self.reduce_refined(groups, None, options)
    }

    /// [`Self::reduce`] with the lanes' rescue-precision kinematics on
    /// demand ([`GroupRefinementSource`]): the fast engine's cancellation
    /// rescue asks for the lanes it flags (direct frame); every other value
    /// is [`Self::reduce`]'s bit for bit.
    pub fn reduce_refined(
        &mut self,
        groups: &[FamilyGroup<'_>],
        source: Option<&dyn GroupRefinementSource>,
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
                run($engine, &groups, options, self.precision_bits, source)
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
    source: Option<&dyn GroupRefinementSource>,
) -> ExecutionResult<V2Output> {
    let buckets = buckets(groups)?;
    let direct = reduce_groups(engine, groups, &buckets, None, options.poles, source)?;
    let rescued = direct
        .iter()
        .enumerate()
        .flat_map(|(group, rows)| {
            rows.iter().enumerate().filter_map(move |(row, outcome)| {
                outcome
                    .as_ref()
                    .and_then(|o| o.rescued)
                    .map(|(carrier, estimate)| (group, row, carrier, estimate))
            })
        })
        .collect();
    let check = options
        .check
        .map(|frame| reduce_groups(engine, groups, &buckets, Some(frame), options.poles, None))
        .transpose()?;
    let mut uncertified: Vec<(usize, usize, &'static str)> = Vec::new();
    // Per uncertified lane its bound, the largest over the frames; None once
    // a frame served it uncertified without one.
    let mut bounds: BTreeMap<(usize, usize), Option<[f64; 3]>> = BTreeMap::new();
    for frame in std::iter::once(&direct).chain(check.as_ref()) {
        for (group, rows) in frame.iter().enumerate() {
            for (row, outcome) in rows.iter().enumerate() {
                let Some(outcome) = outcome.as_ref() else {
                    continue;
                };
                let Some(reason) = outcome.uncertified else {
                    continue;
                };
                if !uncertified.iter().any(|&(g, r, _)| (g, r) == (group, row)) {
                    uncertified.push((group, row, reason));
                }
                let entry = bounds.entry((group, row)).or_insert(Some([0.0; 3]));
                *entry = match (*entry, outcome.bound) {
                    (Some(a), Some(b)) => Some([a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])]),
                    _ => None,
                };
            }
        }
    }
    let uncertified_bounds = bounds
        .into_iter()
        .filter_map(|((group, row), bound)| bound.map(|b| (group, row, b)))
        .collect();
    Ok(V2Output {
        direct: output_batches(groups, direct, options.poles, bits)?,
        check: check
            .map(|outcomes| output_batches(groups, outcomes, options.poles, bits))
            .transpose()?,
        rescued,
        uncertified,
        uncertified_bounds,
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

    /// The call's target and the checks switch reach NinjaRS's options: the
    /// checked mode rescues at `target / RESCUE_SAFETY`, the unchecked mode
    /// has no rescue, the best-effort pass rescues at the target with zero
    /// stability thresholds; the reducer's `set_target` and
    /// `set_best_effort_pass` reach its fast engine, and its refinement
    /// carrier follows the switch.
    #[test]
    fn the_target_and_the_switch_reach_the_fast_engine_options() {
        let lane = crate::reduction_policy::rescue_lane_target(1e-6);
        assert_eq!(lane, 1e-6 / crate::reduction_policy::RESCUE_SAFETY);
        let checked = fast_options_at(false, lane);
        assert!(checked.cancellation_rescue);
        assert_eq!(checked.rescue_target, Some(lane));
        let unchecked = fast_options_at(true, lane);
        assert!(!unchecked.cancellation_rescue);
        assert_eq!(unchecked.rescue_target, None);
        assert_eq!(unchecked.stability_threshold, 0.0);
        let best = fast_options_best_effort(lane);
        assert_eq!(best.cancellation_rescue, RESCUE_IN_BEST_EFFORT);
        assert_eq!(best.rescue_target, RESCUE_IN_BEST_EFFORT.then_some(lane));
        assert_eq!(
            (best.stability_threshold, best.relative_tolerance),
            (0.0, 0.0)
        );
        let mut reducer = V2Reducer::new("double", None, "fast", true).unwrap();
        reducer.set_target(1e-6);
        let Engines::Fast(engine) = &reducer.engines else {
            panic!("the fast engine");
        };
        assert_eq!(engine.lane_target, lane);
        assert_eq!(
            reducer.refinement_carrier(),
            Some(RescueCarrier::DoubleDouble)
        );
        reducer.set_unchecked(true);
        assert_eq!(reducer.refinement_carrier(), None);
        reducer.set_best_effort_pass(true);
        let Engines::Fast(engine) = &reducer.engines else {
            panic!("the fast engine");
        };
        assert!(engine.best_effort);
        assert_eq!(
            reducer.refinement_carrier(),
            RESCUE_IN_BEST_EFFORT.then_some(RescueCarrier::DoubleDouble)
        );
        let generic = V2Reducer::new("double-double", None, "generic", true).unwrap();
        assert_eq!(generic.refinement_carrier(), None);
    }

    #[test]
    fn true_and_declared_ranks() {
        let rank = 3;
        let coefficients: Vec<_> = (0..ncoeff(rank))
            .map(|k| Complex::new(k as f64 - 7.0, 0.5 * k as f64))
            .collect();
        assert_eq!(true_rank(&coefficients, rank), 3);
        let mut lower = coefficients.clone();
        lower[block_offset(3)..].fill(Complex::new(0., 0.));
        assert_eq!(true_rank(&lower, rank), 2);
        assert_eq!(tensor_rank(ncoeff(4)).unwrap(), 4);
        assert!(tensor_rank(7).is_err());
    }

    /// A 256-bit generic engine of tadpole lanes `∫ c / (q² - m²)`.
    /// The generic engine builds and evaluates Symbolica expressions: like
    /// every production caller (the executor's worker threads, a routed
    /// binary64 lane), a test driving it directly holds pyAmpliCol's library
    /// key on its thread. Without it and without `SYMBOLICA_LICENSE`,
    /// Symbolica's restricted mode admits one thread per user, so two such
    /// tests on parallel test threads aborted the test binary (SIGABRT).
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
            origins: Vec::new(),
        };
        (engine, input)
    }

    /// A bucket reduced in calls of `FAST_CHUNK_LANES` lanes gives the values
    /// of one call bitwise: tadpoles and massive bubbles of distinct masses
    /// and shifts, lanes of different calls sharing their masters.
    #[test]
    fn fast_chunks_leave_values_unchanged() {
        for n in [1usize, 2] {
            let lanes = FAST_CHUNK_LANES + 3;
            let input = |masses: &[f64]| {
                let lanes = masses.len() / n;
                let c = |x: f64| Complex::new(x, 0.0);
                BucketInput {
                    n,
                    rank: 1,
                    ids: (0..lanes as u64).collect(),
                    shifts: (0..lanes)
                        .flat_map(|lane| {
                            (0..n).flat_map(move |i| {
                                [0.0, 0.0, 0.0, 250.0 * i as f64 + lane as f64]
                                    .into_iter()
                                    .map(c)
                            })
                        })
                        .collect(),
                    masses: masses.iter().map(|&m| c(m)).collect(),
                    scales: vec![c(1.0e6); lanes],
                    s_matrix: vec![c(0.0); lanes * n * n],
                    coefficients: vec![c(1.0); lanes * ncoeff(1)],
                    origins: Vec::new(),
                }
            };
            let masses: Vec<f64> = (0..lanes * n)
                .map(|k| 29929.0 + 7.0 * (k % 700) as f64)
                .collect();
            let mut engine = FastEngine::new(true).unwrap();
            let chunked = engine.reduce(&input(&masses), None).unwrap();
            let mut whole = FastEngine::new(true).unwrap();
            whole.chunk_lanes = usize::MAX;
            let one = whole.reduce(&input(&masses), None).unwrap();
            assert_eq!(chunked.len(), lanes);
            for lane in 0..lanes {
                assert_eq!(one[lane].status, chunked[lane].status, "n {n} lane {lane}");
                assert_eq!(one[lane].value, chunked[lane].value, "n {n} lane {lane}");
            }
        }
    }

    /// One lane target for both NinjaRS engines: the call's target reaches
    /// the fast engine's cancellation rescue and the generic engines'
    /// cancellation ladder as the same number (from construction on); the
    /// ladder runs in the checked mode and in the best-effort pass only.
    #[test]
    fn one_lane_target_and_one_switch_for_both_engines() {
        use crate::reduction_policy::{DEFAULT_TARGET, rescue_lane_target};
        // The switch is explicit: NinjaRS's own default of the rescue does
        // not decide pyAmpliCol's checked mode.
        assert!(
            fast_options(false).cancellation_rescue,
            "checked: rescue on"
        );
        assert!(
            !fast_options(true).cancellation_rescue,
            "unchecked: rescue off"
        );
        assert!(fast_options_at(false, 1e-8).cancellation_rescue);
        assert_eq!(fast_options_at(false, 1e-8).rescue_target, Some(1e-8));
        assert!(!fast_options_at(true, 1e-8).cancellation_rescue);
        assert!(crate::facade_options().cancellation_rescue, "legacy facade");
        let mut fast = V2Reducer::new("double", None, "fast", true).unwrap();
        let Engines::Fast(engine) = &fast.engines else {
            panic!("fast engine")
        };
        assert_eq!(engine.lane_target, rescue_lane_target(DEFAULT_TARGET));
        fast.set_target(1e-3);
        let Engines::Fast(engine) = &fast.engines else {
            panic!("fast engine")
        };
        assert_eq!(engine.lane_target, rescue_lane_target(1e-3));
        for (arithmetic, digits) in [
            ("double", None),
            ("double-double", None),
            ("arbitrary", Some(40)),
        ] {
            let mut reducer = V2Reducer::new(arithmetic, digits, "generic", true).unwrap();
            let state = |reducer: &V2Reducer| match &reducer.engines {
                Engines::Double(e) => (e.target, e.ladder_on()),
                Engines::DoubleDouble(e) => (e.target, e.ladder_on()),
                Engines::Arbitrary(e) => (e.target, e.ladder_on()),
                Engines::Fast(_) => panic!("generic engine"),
            };
            let default = Some(rescue_lane_target(DEFAULT_TARGET));
            assert_eq!(state(&reducer), (default, true), "{arithmetic}");
            reducer.set_target(1e-3);
            assert_eq!(
                state(&reducer),
                (Some(rescue_lane_target(1e-3)), true),
                "{arithmetic}"
            );
            reducer.set_unchecked(true);
            assert!(!state(&reducer).1, "{arithmetic}: no ladder unchecked");
            reducer.set_best_effort_pass(true);
            assert_eq!(
                state(&reducer).1,
                RESCUE_IN_BEST_EFFORT,
                "{arithmetic}: the best-effort pass follows the fast engine's"
            );
            reducer.set_best_effort_pass(false);
            assert!(!state(&reducer).1, "{arithmetic}");
        }
    }

    /// The best-effort pass serves each refused lane its OWN best value:
    /// two lanes of different families that share the row id 0 in one
    /// generic chunk (tadpoles of distinct masses), refused by the ladder at
    /// a target no rung meets, come back with their own values (those of
    /// the checked reduction at the default target, which serves them).
    #[test]
    fn best_effort_serves_each_lane_its_own_best_value() {
        let _unlock = pyamplicol::unlock(); // `tadpoles`
        let scale = Float::with_val(256, 1_000_000);
        let masses = [Float::with_val(256, 29929), Float::with_val(256, 8315)];
        let (mut checked, input) = tadpoles(&masses, &scale);
        let expected = checked.reduce(&input, None).unwrap();
        assert!(expected.iter().all(|o| o.status == "success"));
        let (mut engine, mut shared) = tadpoles(&masses, &scale);
        shared.ids = vec![0, 0];
        engine.set_checked(false);
        engine.set_best_effort(true);
        engine.set_cancellation_target(Some(1e-320));
        let served = engine.reduce(&shared, None).unwrap();
        use symbolica::domains::float::RealLike;
        let finite = |o: &Outcome<Float>| o.value.as_ref().expect("a value")[0].re.clone();
        let relative =
            |x: &Float, y: &Float| (x.clone() - y.clone()).to_f64().abs() / y.to_f64().abs();
        let (a, b) = (finite(&expected[0]), finite(&expected[1]));
        assert!(relative(&a, &b) > 1e-3, "distinct values");
        for (lane, want) in [(0, a), (1, b)] {
            if RESCUE_IN_BEST_EFFORT {
                assert_eq!(served[lane].status, "success", "lane {lane}");
                let got = finite(&served[lane]);
                assert!(
                    relative(&got, &want) <= 1e-60,
                    "lane {lane}: {got} vs its own {want}"
                );
            }
        }
    }

    /// One bubble lane through the reducer (shifts 0 and (1,0,0,0), squared
    /// masses 4, mu^2 = 1, poles none) with the rank-1 numerator
    /// `c0 + c1 q0`, in the reducer's carrier `T`: its status, whether it was
    /// served without accuracy evidence (`V2Output::uncertified`), the
    /// estimate of a rescue within the reduction, and its finite part.
    fn bubble<T: BatchCarrier>(
        reducer: &mut V2Reducer,
        zero: T,
        c0: Complex<T>,
        c1: T,
        finite: impl Fn(&super::super::batch::Data) -> Complex<f64>,
    ) -> (String, bool, Option<f64>, Option<Complex<f64>>)
    where
        T: From<f64> + Clone,
    {
        let _unlock = pyamplicol::unlock(); // generic engine
        let bits = reducer.precision_bits();
        let c = |x: f64| Complex::new(T::from(x), zero.clone());
        let batch = |values: Vec<Complex<T>>, columns| {
            Batch::typed(values, vec![0], columns, bits).unwrap()
        };
        let mut shifts = vec![c(0.0); 8];
        shifts[4] = c(1.0);
        let shifts = batch(shifts, 8);
        let masses = batch(vec![c(4.0), c(4.0)], 2);
        let scales = batch(vec![c(1.0)], 1);
        let s_matrix = batch(vec![c(0.0), c(1.0), c(1.0), c(0.0)], 4);
        let mut numerator = vec![c(0.0); ncoeff(1)];
        numerator[0] = c0;
        numerator[1] = Complex::new(c1, zero.clone());
        let coefficients = batch(numerator, ncoeff(1));
        let group = FamilyGroup {
            propagators: 2,
            shifts: &shifts,
            masses_squared: &masses,
            scales_squared: &scales,
            s_matrix: &s_matrix,
            coefficients: &coefficients,
        };
        let options = ReduceOptions {
            poles: Poles::None,
            ..ReduceOptions::default()
        };
        let output = reducer.reduce(&[group], options).unwrap();
        let status = output.direct[0].statuses()[0].1.clone();
        let uncertified = output.uncertified.iter().any(|&(g, r, _)| (g, r) == (0, 0));
        let estimate = output
            .rescued
            .iter()
            .find(|r| (r.0, r.1) == (0, 0))
            .map(|r| r.3);
        let value =
            (status == "success").then(|| finite(output.direct[0].values().unwrap().data()));
        (status, uncertified, estimate, value)
    }

    fn dd_finite(data: &super::super::batch::Data) -> Complex<f64> {
        use symbolica::domains::float::RealLike;
        match data {
            super::super::batch::Data::DoubleDouble(v) => {
                Complex::new(v[0].re.to_f64(), v[0].im.to_f64())
            }
            _ => panic!("a double-double batch"),
        }
    }

    /// The external review's lane: `N = 2^k (1 + 2 q0) + i` on the bubble
    /// above, whose 2^k part integrates to zero by symmetry (the integral is
    /// that of `N = i` for every k), its terms cancelling by 2^k. Never a
    /// certified success with a wrong value: either a success equal to the
    /// `N = i` lane's value (k = 360: the ladder climbs and meets the target),
    /// or served without accuracy evidence or refused (k = 800: below the
    /// resolution of every rung).
    #[test]
    fn a_lane_below_every_rung_is_never_a_certified_success() {
        let dd = |x: f64| DoubleFloat::from(x);
        let mut reducer = V2Reducer::new("double-double", None, "generic", true).unwrap();
        let (status, _, _, reference) = bubble(
            &mut reducer,
            dd(0.0),
            Complex::new(dd(0.0), dd(1.0)),
            dd(0.0),
            dd_finite,
        );
        assert_eq!(status, "success");
        let reference = reference.unwrap();
        for k in [360, 800] {
            let mut reducer = V2Reducer::new("double-double", None, "generic", true).unwrap();
            let big = dd(2f64.powi(k));
            let (status, uncertified, _, value) = bubble(
                &mut reducer,
                dd(0.0),
                Complex::new(big.clone(), dd(1.0)),
                big.clone() + big,
                dd_finite,
            );
            if status == "success" && !uncertified {
                let value = value.unwrap();
                let error =
                    ((value.re - reference.re).powi(2) + (value.im - reference.im).powi(2)).sqrt();
                let size = (reference.re.powi(2) + reference.im.powi(2)).sqrt();
                assert!(
                    error <= 1e-12 * size,
                    "k {k}: certified success {value:?} against {reference:?}"
                );
            }
        }
    }

    /// The vanishing rule of the lane rescue's last carrier
    /// (`V2Reducer::set_vanishing`): a lane zero by symmetry (`N = 1 + 2 q0`
    /// on the bubble above, whose integral is exactly 0), refused by the
    /// ladder, is returned with the uncertified reason when the rule is on,
    /// and never a certified success when it is off; a refused lane whose
    /// best value is NOT consistent with zero (`N = 1` at a lane target no
    /// rung meets) stays refused with the rule on.
    #[test]
    fn a_vanishing_lane_is_returned_uncertified_and_a_refused_one_stays_refused() {
        let dd = |x: f64| DoubleFloat::from(x);
        let mut off = V2Reducer::new("double-double", None, "generic", true).unwrap();
        let (status, uncertified, _, _) = bubble(
            &mut off,
            dd(0.0),
            Complex::new(dd(1.0), dd(0.0)),
            dd(2.0),
            dd_finite,
        );
        assert!(
            status != "success" || uncertified,
            "rule off: {status} certified"
        );
        let mut on = V2Reducer::new("double-double", None, "generic", true).unwrap();
        on.set_vanishing(Some(UNCERTIFIED_VANISHING_DD));
        let (status, uncertified, _, value) = bubble(
            &mut on,
            dd(0.0),
            Complex::new(dd(1.0), dd(0.0)),
            dd(2.0),
            dd_finite,
        );
        assert_eq!(status, "success");
        assert!(uncertified, "returned without its uncertified reason");
        let value = value.unwrap();
        assert!(value.re.hypot(value.im) < 1e-20, "{value:?}");
        let mut refused = V2Reducer::new("double-double", None, "generic", true).unwrap();
        refused.set_vanishing(Some(UNCERTIFIED_VANISHING_DD));
        refused.set_target(1e-40 * crate::reduction_policy::RESCUE_SAFETY);
        let (status, uncertified, _, _) = bubble(
            &mut refused,
            dd(0.0),
            Complex::new(dd(1.0), dd(0.0)),
            dd(0.0),
            dd_finite,
        );
        assert!(
            status != "success" && !uncertified,
            "a non-vanishing lane served: {status}"
        );
    }

    /// The ordinary scalar bubble (`N = 1`) in binary64 at a lane target of
    /// 1e-25, below what the fast engine's rescue can certify (its estimate
    /// keeps the binary64 coefficients' 2^-53): a typed refusal for the lane
    /// rescue, never a success claiming an estimate above the target.
    #[test]
    fn a_target_below_the_carrier_floor_is_refused_not_accepted() {
        let mut reducer = V2Reducer::new("double", None, "fast", true).unwrap();
        reducer.set_target(1e-25 * crate::reduction_policy::RESCUE_SAFETY);
        let f64_finite = |data: &super::super::batch::Data| match data {
            super::super::batch::Data::Double(v) => v[0],
            _ => panic!("a binary64 batch"),
        };
        let (status, uncertified, estimate, _) = bubble(
            &mut reducer,
            0.0_f64,
            Complex::new(1.0, 0.0),
            0.0,
            f64_finite,
        );
        assert!(
            status != "success" || (!uncertified && estimate.is_none_or(|e| e <= 1e-25)),
            "{status}, estimate {estimate:?}"
        );
    }

    /// The fast engine's vanishing rule (`vanishes_within`): in binary64 a
    /// lane zero by symmetry (`N = 1 + 2 q0` on the bubble above) that
    /// NinjaRS's rescue refuses is served at its re-reduced value,
    /// uncertified, by the reducer itself (no `set_vanishing`: the lane rescue
    /// is not involved). The reviewer's lane `N = 2^k (1 + 2 q0) + i`, in
    /// truth the `N = i` lane's value for every k, is never a certified
    /// success with a wrong value: certified and right (small k), served
    /// uncertified, or refused (large k: beyond double-double's resolution of
    /// its own terms, the case the rule gives up to an explicit rescue
    /// policy); and the double-double generic engine, the next level of
    /// such a policy, recovers it.
    #[test]
    fn the_fast_engine_serves_a_vanishing_lane_uncertified_and_never_certifies_a_wrong_one() {
        let f64_finite = |data: &super::super::batch::Data| match data {
            super::super::batch::Data::Double(v) => v[0],
            _ => panic!("a binary64 batch"),
        };
        let mut fast = V2Reducer::new("double", None, "fast", true).unwrap();
        let (status, uncertified, _, value) =
            bubble(&mut fast, 0.0_f64, Complex::new(1.0, 0.0), 2.0, f64_finite);
        assert_eq!(status, "success");
        assert!(
            uncertified,
            "a vanishing lane served as a certified success"
        );
        let value = value.unwrap();
        assert!(value.re.hypot(value.im) < 1e-14, "{value:?}");
        let mut fast = V2Reducer::new("double", None, "fast", true).unwrap();
        let (status, uncertified, _, reference) =
            bubble(&mut fast, 0.0_f64, Complex::new(0.0, 1.0), 0.0, f64_finite);
        assert_eq!(status, "success");
        assert!(!uncertified);
        let reference = reference.unwrap();
        let size = reference.re.hypot(reference.im);
        let mut outcomes = Vec::new();
        for k in [20, 60, 120, 400] {
            let big = 2f64.powi(k);
            let mut fast = V2Reducer::new("double", None, "fast", true).unwrap();
            let (status, uncertified, _, value) = bubble(
                &mut fast,
                0.0_f64,
                Complex::new(big, 1.0),
                2.0 * big,
                f64_finite,
            );
            if status == "success" && !uncertified {
                let value = value.unwrap();
                let error = (value.re - reference.re).hypot(value.im - reference.im);
                assert!(
                    error <= 1e-6 * size,
                    "k {k}: certified success {value:?} against {reference:?}"
                );
            }
            outcomes.push((k, status, uncertified));
        }
        // The next carrier recovers a lane the binary64 rule gives up: the
        // double-double generic engine, whose cancellation ladder climbs
        // (a policy's double-double level).
        let dd = |x: f64| DoubleFloat::from(x);
        let big = dd(2f64.powi(120));
        let mut generic = V2Reducer::new("double-double", None, "generic", true).unwrap();
        let (status, uncertified, _, value) = bubble(
            &mut generic,
            dd(0.0),
            Complex::new(big.clone(), dd(1.0)),
            big.clone() + big,
            dd_finite,
        );
        assert_eq!(status, "success", "{outcomes:?}");
        assert!(!uncertified, "{outcomes:?}");
        let value = value.unwrap();
        let error = (value.re - reference.re).hypot(value.im - reference.im);
        assert!(error <= 1e-12 * size, "{value:?} against {reference:?}");
        eprintln!("fast-engine outcomes of N = 2^k (1 + 2 q0) + i: {outcomes:?}");
    }

    /// The bound of a vanishing lane (`V2Output::uncertified_bounds`): the
    /// symmetry-zero bubble (`N = 1 + 2 q0`) carries one that covers its
    /// served value and is small next to its terms; the reviewer's lane
    /// `N = 2^400 (1 + 2 q0) + i`, served ~0 uncertified while its true
    /// value is the `N = i` lane's, carries one that covers that true
    /// value: a check that adds the bounds of the uncertified lanes to its
    /// residue cannot pass on two matching wrong zeros.
    #[test]
    fn a_vanishing_lane_carries_a_bound_that_covers_its_true_value() {
        let _unlock = pyamplicol::unlock();
        let run = |c0: Complex<f64>, c1: f64| {
            let mut fast = V2Reducer::new("double", None, "fast", true).unwrap();
            let batch = |values: Vec<Complex<f64>>, columns| {
                Batch::typed(values, vec![0], columns, 53).unwrap()
            };
            let c = |x: f64| Complex::new(x, 0.0);
            let mut shifts = vec![c(0.0); 8];
            shifts[4] = c(1.0);
            let shifts = batch(shifts, 8);
            let masses = batch(vec![c(4.0), c(4.0)], 2);
            let scales = batch(vec![c(1.0)], 1);
            let s_matrix = batch(vec![c(0.0), c(1.0), c(1.0), c(0.0)], 4);
            let mut numerator = vec![c(0.0); ncoeff(1)];
            numerator[0] = c0;
            numerator[1] = c(c1);
            let coefficients = batch(numerator, ncoeff(1));
            let group = FamilyGroup {
                propagators: 2,
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                s_matrix: &s_matrix,
                coefficients: &coefficients,
            };
            let options = ReduceOptions {
                poles: Poles::None,
                check: Some(ReductionFrame::REVERSED),
                ..ReduceOptions::default()
            };
            fast.reduce(&[group], options).unwrap()
        };
        let finite = |output: &V2Output| match output.direct[0].values().unwrap().data() {
            super::super::batch::Data::Double(v) => v[0],
            _ => panic!("a binary64 batch"),
        };
        let zero = run(Complex::new(1.0, 0.0), 2.0);
        assert_eq!(zero.uncertified.len(), 1);
        let &(_, _, bound) = zero.uncertified_bounds.first().expect("a bound");
        let value = finite(&zero);
        assert!(value.re.hypot(value.im) <= bound[0], "{value:?} {bound:?}");
        assert!(bound[0] < 1e-12, "{bound:?}");
        let truth = finite(&run(Complex::new(0.0, 1.0), 0.0));
        let big = 2f64.powi(400);
        let wrong = run(Complex::new(big, 1.0), 2.0 * big);
        if !wrong.uncertified.is_empty() {
            let value = finite(&wrong);
            let bound = wrong
                .uncertified_bounds
                .first()
                .map(|&(_, _, b)| b[0])
                .unwrap_or(f64::INFINITY);
            let error = (value.re - truth.re).hypot(value.im - truth.im);
            assert!(error <= bound, "error {error:e} above the bound {bound:e}");
        }
    }

    /// The external reviewer's generic double-double case: the bubble
    /// `N = 2^1018 (1 + 2 q0) + i` (true finite part -1.3435472012813139 i)
    /// comes back from the generic double-double engine at or near zero as
    /// an uncertified lane, its ladder having reached a rung where a single
    /// cut's term modulus (4.3e306) times 256 overflows in f64 and the
    /// record's bound reads zero. No bound is exported for it: a check that
    /// depends on the lane is inconclusive.
    #[test]
    fn a_generic_ladder_lane_exports_no_bound() {
        let _unlock = pyamplicol::unlock();
        let dd = |x: f64| DoubleFloat::from(x);
        let mut generic = V2Reducer::new("double-double", None, "generic", true).unwrap();
        let bits = generic.precision_bits();
        let c = |x: f64| Complex::new(dd(x), dd(0.0));
        let batch = |values: Vec<Complex<DoubleFloat>>, columns| {
            Batch::typed(values, vec![0], columns, bits).unwrap()
        };
        let mut shifts = vec![c(0.0); 8];
        shifts[4] = c(1.0);
        let shifts = batch(shifts, 8);
        let masses = batch(vec![c(4.0), c(4.0)], 2);
        let scales = batch(vec![c(1.0)], 1);
        let s_matrix = batch(vec![c(0.0), c(1.0), c(1.0), c(0.0)], 4);
        let big = 2f64.powi(1018);
        let mut numerator = vec![c(0.0); ncoeff(1)];
        numerator[0] = Complex::new(dd(big), dd(1.0));
        numerator[1] = c(2.0 * big);
        let coefficients = batch(numerator, ncoeff(1));
        let group = FamilyGroup {
            propagators: 2,
            shifts: &shifts,
            masses_squared: &masses,
            scales_squared: &scales,
            s_matrix: &s_matrix,
            coefficients: &coefficients,
        };
        let options = ReduceOptions {
            poles: Poles::None,
            ..ReduceOptions::default()
        };
        let output = generic.reduce(&[group], options).unwrap();
        let status = output.direct[0].statuses()[0].1.clone();
        let uncertified = output.uncertified.iter().any(|&(g, r, _)| (g, r) == (0, 0));
        // Whatever the ladder decides, an uncertified lane of the generic
        // engine carries no check budget, and a served value is never a
        // certified success unless it is right.
        assert!(
            output.uncertified_bounds.is_empty(),
            "{:?}",
            output.uncertified_bounds
        );
        if status == "success" && !uncertified {
            let value = dd_finite(output.direct[0].values().unwrap().data());
            assert!(
                (value.im + 1.3435472012813139).abs() <= 1e-12 && value.re.abs() <= 1e-12,
                "a certified success {value:?}"
            );
        }
        assert!(credible_bound(1e-300) && !credible_bound(0.0));
        assert!(!credible_bound(f64::INFINITY) && !credible_bound(1e-310));
    }

    #[test]
    fn imprecise_float_masters_fail_their_lane() {
        let _unlock = pyamplicol::unlock(); // `tadpoles`
        // numerica tracks significant bits: a squared mass known to 53 bits
        // gives an A0 below `required_tracked_bits(256)` = 64 tracked bits,
        // which the master provider reports as `ImpreciseResult`; the lane
        // fails instead of returning a finite but unestablished value.
        let scale = Float::with_val(256, 1_000_000);
        let masses = [Float::with_val(256, 29929), Float::with_val(53, 29929)];
        let (mut engine, input) = tadpoles(&masses, &scale);
        let outcomes = engine.reduce(&input, None).unwrap();
        assert_eq!(outcomes[0].status, "success");
        assert!(outcomes[0].value.is_some());
        assert!(
            outcomes[1].status.contains("ImpreciseResult"),
            "{}",
            outcomes[1].status
        );
        assert!(outcomes[1].value.is_none());
    }

    /// The tracked-precision test of the reducer: NinjaRS's rule at the
    /// working bits of the arbitrary-precision engine only, off in the
    /// unchecked mode; on the generic engine's own lanes (a rank-1
    /// tadpole) a point whose shift is known to 20 bits (its lane keeps
    /// about them, its masters all 256) fails typed, the full-precision
    /// point passes.
    #[test]
    fn the_tracked_precision_test_reads_the_lanes_precision() {
        let _unlock = pyamplicol::unlock(); // generic engine
        for (arithmetic, digits, engine) in [
            ("double", None, "fast"),
            ("double", None, "generic"),
            ("double-double", None, "generic"),
        ] {
            let reducer = V2Reducer::new(arithmetic, digits, engine, true).unwrap();
            assert_eq!(reducer.tracked_precision(), None, "{arithmetic} {engine}");
        }
        // The working digits of pyAmpliCol's 40, 57, 60 and 80 digits
        // (20 guard digits).
        for (digits, working, required) in
            [(60, 200, 53), (77, 256, 64), (80, 266, 66), (100, 333, 83)]
        {
            let reducer = V2Reducer::new("arbitrary", Some(digits), "generic", true).unwrap();
            let rule = reducer.tracked_precision().unwrap();
            assert_eq!(
                (rule.working, rule.required),
                (working, required),
                "{digits}"
            );
        }
        let mut reducer = V2Reducer::new("arbitrary", Some(77), "generic", true).unwrap();
        let bits = reducer.precision_bits();
        let zero = Float::with_val(bits, 0);
        let c = |x: Float| Complex::new(x, zero.clone());
        // A rank-1 tadpole `q0 / ((q + p)^2 - m^2)` (integral `-p0 A0`):
        // the shift enters the reduction's arithmetic, the masters take the
        // exact squared mass and scale.
        let reduce = |reducer: &mut V2Reducer, shift: Float| {
            let batch = |values: Vec<Complex<Float>>, columns| {
                Batch::typed(values, vec![0], columns, bits).unwrap()
            };
            let shifts = batch(
                vec![c(shift), c(zero.clone()), c(zero.clone()), c(zero.clone())],
                4,
            );
            let masses = batch(vec![c(Float::with_val(bits, 29929))], 1);
            let scales = batch(vec![c(Float::with_val(bits, 1_000_000))], 1);
            let s_matrix = batch(vec![c(zero.clone())], 1);
            let mut numerator = vec![c(zero.clone()); ncoeff(1)];
            numerator[1] = c(Float::with_val(bits, 1));
            let coefficients = batch(numerator, ncoeff(1));
            let group = FamilyGroup {
                propagators: 1,
                shifts: &shifts,
                masses_squared: &masses,
                scales_squared: &scales,
                s_matrix: &s_matrix,
                coefficients: &coefficients,
            };
            let output = reducer.reduce(&[group], ReduceOptions::default()).unwrap();
            assert_eq!(output.direct[0].statuses()[0].1, "success");
            output.direct
        };
        let exact = reduce(&mut reducer, Float::with_val(bits, 3));
        assert_eq!(reducer.tracked_refusal(&exact), None);
        let coarse = reduce(&mut reducer, Float::with_val(20, 3));
        let refusal = reducer.tracked_refusal(&coarse).unwrap();
        assert!(
            refusal.starts_with("NinjaRS tracked precision: the point's lanes keep 2")
                && refusal.contains("64 required at 256 working bits; "),
            "{refusal}"
        );
        // Both points together: the coarse lane is as large as the exact one.
        assert!(
            reducer
                .tracked_refusal(exact.iter().chain(&coarse))
                .is_some()
        );
        // Only successful rows count: the coarse value under a failed status
        // is not summed (the executor sums successes only).
        let failed = ReductionBatch {
            values: coarse[0].values.clone(),
            statuses: vec![(0, "Unstable".into())],
        };
        assert_eq!(reducer.tracked_refusal(exact.iter().chain([&failed])), None);
        reducer.set_unchecked(true);
        assert_eq!(reducer.tracked_precision(), None);
        assert_eq!(reducer.tracked_refusal(&coarse), None);
    }

    #[test]
    fn generic_chunks_leave_values_unchanged() {
        let _unlock = pyamplicol::unlock(); // `tadpoles`
        let scale = Float::with_val(256, 1_000_000);
        let masses: Vec<_> = (0..GENERIC_CHUNK_LANES + 3)
            .map(|k| Float::with_val(256, 29929 + 7 * k as i64))
            .collect();
        let (mut engine, input) = tadpoles(&masses, &scale);
        let chunked = engine.reduce(&input, None).unwrap();
        for (k, mass) in masses.iter().enumerate() {
            let (mut single, one) = tadpoles(std::slice::from_ref(mass), &scale);
            let alone = single.reduce(&one, None).unwrap();
            assert_eq!(alone[0].status, "success");
            assert_eq!(chunked[k].value, alone[0].value, "lane {k}");
        }
    }
}
