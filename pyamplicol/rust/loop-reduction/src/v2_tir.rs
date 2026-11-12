// SPDX-License-Identifier: 0BSD

//! The TIR backend of the v2 executor (m-lit PLAN §2.b.2–§2.b.6): tensor
//! integrals from TIRust's direct engine (`iregi_rust::DirectWorkspace`), in
//! every carrier of the executor (binary64, double-double, MPFR; m-lit
//! lit-tir-mp), contracted natively with every lane's numerator
//! coefficients.
//!
//! **Object.** NinjaRS reduces `μ^{2ε}/(iπ^{2−ε} r_Γ) ∫ N4(q̄)/Π D_d`
//! (`crate` docs); TIRust's `Convention::OneLoop` with
//! `TensorWeights::Components` returns that normalisation's Cartesian tensor
//! moments `M_α = ∫ q^α / Π D` with the slots `[UV1, IR1, IR2, finite]`, so
//! a lane's integral is `Σ_α c_α M_α` with NinjaRS's own coefficients (no
//! symmetry factors; lit-spike-tir T1). The Laurent columns are those of the
//! NinjaRS adapter: finite `Σ c_α M^fin_α`, `1/ε` `Σ c_α (M^UV1_α + M^IR1_α)`
//! (formed identically for [`Poles::All`] and [`Poles::Split`], so the
//! combined columns are bitwise equal), `1/ε²` `Σ c_α M^IR2_α`, and with
//! [`Poles::Split`] the UV residue `Σ c_α M^UV1_α`. [`Poles::None`]
//! contracts the finite slot only (the moments carry every slot anyway).
//!
//! **Inputs (TIRust U1 and G6, lit-tirust-io).** Each row hands TIRust the
//! family's Cartesian shifts, the S-matrix the executor forms for the row
//! (`InvariantInput::SuppliedInvariants`: the invariants NinjaRS receives,
//! cast once from the exact sums, with the on-shell bound entries exactly the
//! legs' squared masses, DESIGN §11.1), the squared masses and `μ²` itself
//! (`ScaleInput::MuSquared`, no square root of a rounded `μ²`). Every scalar
//! TIRust forms from a pair of shifts reads the supplied invariant, so an
//! on-shell invariant equal to an adjacent squared mass stays equal to it bit
//! for bit and its masters stay on the exact on-shell branch in every
//! carrier (a massive leg next to a propagator of its own mass, e.g. a top in
//! its own gauge loop, the O(1) DD/MPFR failure of DESIGN §11.1 otherwise).
//! A high-precision reference for these lanes must take inputs consistent to
//! its own precision: lift the exact sums, or re-derive the non-bound entries
//! from the lifted shifts, never pair exactly formed multi-leg invariants
//! with rounded shifts (lit-onshell-probe; NinjaRS moves by up to 5e-4 on
//! such inconsistent inputs, lit-tir report §G).
//!
//! **Rows.** The executor's lanes of one point are contiguous in a family
//! group with bitwise equal kinematics (`lanes_of`): each run is one
//! kinematic row, whose tensor moments are computed once, at the row's
//! moment rank (the largest true rank of its lanes; over-declaring costs
//! time and gate refusals, not accuracy, lit-spike-tir T4/F4), and serve
//! every lane of the row: all amplitude lanes (helicity x colour x loop
//! order) of a loop-induced (family, point), or the sector columns of a
//! Born-interfered one. Rows of equal (propagators, moment rank) share one
//! TIRust call. The moments are permuted once per row from TIRust's
//! `tensor_basis` into NinjaRS's `tensor_layout` order (the permutation is
//! read from the plan's basis, a bijection at every rank).
//!
//! **Carriers and gates.** One plan per (propagators, rank) serves every
//! carrier; each carrier has its own workspaces (MPFR one per precision).
//! Every row goes first through the conditioning gates of the carrier's
//! workspace ([`TirOptions::gate`], by default TIRust's own binary64 gate).
//! The extended carriers
//! (double-double, MPFR) would otherwise inherit that binary64-tuned value
//! at every precision (lit-tirust-io's interface note): their gate is the
//! binary64 one carried to the carrier's precision, `g^(p / 53)` for a
//! carrier of `p` bits ([`carrier_gate`]), so that every carrier admits the
//! same share of its significand as rounding amplification (TIRust's 1e-12
//! at 53 bits is about 75 % of the binary64 significand: 1e-24 in
//! double-double). TIRust's partial double-double rescue and precision tiers
//! stay off (PLAN §2.b.6): a gate-refused row fails closed with the gates'
//! reason, and the caller's rescue ladder decides.
//!
//! **Method choice and its acceptance (lit-spike-tir F2; DECISIONS
//! 2026-11-07 ~10:05, ~17:20 (d), 21:50 R4).** With
//! [`TirOptions::method_choice`] (binary64; on by default) or
//! [`TirOptions::extended_method_choice`] (double-double, MPFR; off by
//! default) the rows the gates refuse are retried in a second workspace
//! with TIRust's a priori (COLLIER-style) method choice: small-Gram
//! expansions or the alternative PV reduction for the boxes and triangles
//! whose PV estimate is too large, with the alternative PV first wherever
//! its estimate is within the expansions' own tolerance
//! (`DirectWorkspace::set_alternative_first`,
//! [`TirOptions::alternative_first`]). TIRust leaves those values unchecked
//! ("callers verify the lanes"; its convergence test bounds the expansions'
//! truncation, not the rounding of what they read), so every lane of a
//! method-choice row is verified here, never by TIRust's hidden rotation
//! test: its value in the acceptance frame of the same row
//! ([`TirOptions::acceptance_frame`]; the call's check row when the call
//! checks in that very frame) must agree with the direct one within
//! [`TirOptions::acceptance`] of the lane's term scale `Σ_α |c_α| |M_α|`
//! (its size before the contraction's cancellations); a lane that does not,
//! or whose acceptance row is refused, fails closed with its reason. The
//! acceptance frame is the rescaled one (rotation by one with the origin
//! shift at inputs rescaled by 7/10, masters recomputed): a rotated frame
//! alone is blind on some wrong method-choice lanes (massive boxes and
//! pentagons the gates refuse on a massive sub-triangle; lit-tir report
//! §H), the rescaled one sees them (m-lit lit-tir-mp: every one of those 21
//! lanes refused; no method-choice lane of the synthetic set served beyond
//! 1.3e-7 of a consistent truth; about 3 % of the lanes the rotated frame
//! accepts refused, nearly all of them right). A row still refused after the
//! method choice goes to the verified last resort (below; with it off it
//! fails every lane of the row); a row with a TIRust `InvalidInput` /
//! `MasterFailure` status fails every lane of the row (a best-effort array
//! is never a value). The
//! extended carriers keep it off until it is measured there: their
//! gate-refused rows fail closed and the caller's rescue ladder decides.
//!
//! **Unchecked mode (m-lit lit-unchecked, USER 2026-11-08 ~21:14 / ~22:34,
//! DECISIONS 22:34 (1); [`TirOptions::unchecked`], off by default; the
//! public setting `reduction_unchecked`).** The a priori conditioning gates
//! and the method choice stay, as METHOD SELECTION (as COLLIER inside
//! Recola or OpenLoops in their optimistic settings); the a posteriori
//! acceptance pass of the method-choice rows (their second frame) and its
//! comparison are never computed, so their lanes are returned as computed
//! and the call's target accuracy drives nothing here. The rows the method
//! choice still refuses (for any of its causes: an expansion that does not
//! converge, an alternative PV estimate above its limit, a cancelling det X,
//! vanishing adjugates, a triangle below the requested rank, its own gate),
//! and in a carrier whose method choice is off the rows the gates refuse,
//! are served by their LAST RESORT: the same rows in a third workspace at
//! gate 0, i.e. plain PV ([`Bin::last_resort`], counted as "TIR
//! last-resort rows"). Every row is served by construction except those
//! without a finite value in any scheme: an exactly singular pivot after
//! TIRust's affine-relation variants, a non-finite output, and TIRust's
//! `InvalidInput` / `MasterFailure` statuses (which keep their reasons); the
//! capability refusal stays a call-level typed refusal. Gate-passed and
//! method-choice values are bitwise those of the checked mode. A call's own
//! check frame (pyAmpliCol's stability checks, when the caller enables
//! them) runs through the same three passes.
//!
//! **Verified last resort (m-lit tir-uuzggg; [`TirOptions::
//! verified_last_resort`], on by default).** A box or triangle whose Gram
//! system the gates refuse and whose method choice finds no method (a
//! small-Gram expansion that does not converge because the Cayley
//! determinant is small as well, and an alternative PV whose modified
//! Cayley determinant cancels) is not a lane without a value: plain PV at
//! gate 0 is often accurate there, as the gates bound the worst case
//! `condition^rank` of one subset, not the error of the lane. In the
//! checked mode such a row is served by the unchecked mode's last resort
//! (plain PV at gate 0, the same third workspace) and every lane of it is
//! then verified as a method-choice lane: by its acceptance frame within
//! [`TirOptions::acceptance`] of its term scale, failing closed with the
//! measured deviation otherwise. Where the method choice is off (the
//! extended carriers by default), gate-refused rows still fail closed.
//! Lanes the gates or the method choice serve are bitwise unchanged; only
//! lanes that were refused can be served. Measured on u u~ > z g g g
//! [QCD] (every flat point refused by one or two rank-4 boxes of Gram
//! conditioning 1e-4 to 5e-10 with a near-degenerate sub-triangle;
//! reports/engine-v2/m-lit/tir-uuzggg-report.md).
//!
//! **Check frames (lit-spike-tir F3; the rotation copy, m-lit lit-stab-t,
//! Q21 (b)).** `q -> -q` leaves the moments bitwise (odd moments flip
//! exactly), so TIR's frame of a [`ReductionCheck`] never reverses
//! ([`tir_frame`]). A check without `rescale` (the reduction-direction
//! check) is the cyclic relabelling of the propagators by `check.rotation`
//! in the same loop-momentum frame: the same integrals reduced through other
//! PV systems (another reference propagator), with the lanes' coefficients
//! reused exactly. It is TIRust's rotation copy (U3, `DirectWorkspace::
//! reduce_batch_with_rotation`): the copy's moments come from the direct
//! pass's own calls, bit for bit a separate call on the relabelled rows
//! (no masters provider is attached), the subsets that keep their order
//! computed once ([`TirOptions::rotation_copy`]; off: that separate call).
//! A check with `rescale` is the rotation with the origin moved to the
//! frame's first propagator (the numerator re-expanded exactly) at inputs
//! rescaled by `x = 7/10`, with the exact `x^(4-2N)` undo (the frames
//! NinjaRS's adapter uses, `runtime::frames`; the supplied S-matrix
//! relabelled and rescaled with them), a second pass in that frame: its
//! masters are evaluated anew. Every check row (the copy's too) runs
//! through the same two passes as a direct row: the gates, then the method
//! choice on what they refuse (a row and its copy both refused: one
//! rotation-copy call of the method choice; one of them: a call for it).
//!
//! **Capability.** What TIRust serves is read from TIRust: its capability
//! query (`DirectCapabilities::admits` of the plan's settings at the
//! carrier's precision, [`capability_refusal`]) and its typed plan
//! construction, each refusal carried verbatim (`ReductionUnavailable` in
//! Python); it is the artifacts' rank admission of TIRust as well
//! ([`rank_refusal`], which asks the same query at binary64: TIRust has no
//! rank <= N + 2 limit, so its artifacts may hold families NinjaRS cannot
//! reduce, m-lit lit-eft's TIR-only artifacts). No TIRust limit is stated here; the
//! validated domain (N <= 6, R <= 6) is reported, never enforced. A carrier
//! exponent range left by a lane is TIRust's typed per-lane `Unstable`
//! status. Plans are built from an artifact's (propagators, rank) histogram
//! when the backend is created ([`TirReductionBackend::with_histogram`]:
//! every rank up to each family's declared one, since rows reduce at their
//! true rank) and on first use otherwise.
//!
//! **Symbolica.** The binary64 direct engine with its native master kernels
//! builds and evaluates no Symbolica expression; nothing here takes
//! pyAmpliCol's library key (test T8 in `tests/unit`).

use crate::Complex;
use crate::runtime::frames::{FrameMap, ReductionFrame};
use crate::runtime::{Carrier, TypedBatch};
use iregi_rust::{
    BatchResult, Convention, DirectCapabilities, DirectReductionPlan, DirectWorkspace,
    InvariantInput, KinematicsBatch, LaneStatus, MasterFamily, MasterProvider, ReductionSettings,
    ScaleInput, TensorStore, TensorStoreOptions, TensorWeights,
};
use ninjars::fast::tensor_layout::{monomials, ncoeff};
use rusticol_core::loop_execution::v2::{
    BackendStatistics, FamilyLanes, LaneMeasure, LaneOutcome, MeasureKind, MeasuredLane, Poles,
    Reduced,
};
use rusticol_core::loop_v2::bounds::{Reducer, rank_refusal};
use rusticol_core::loop_v2::retest::ReductionCheck;
use rusticol_core::{RusticolError, RusticolResult};
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;
use symbolica::domains::float::{DoubleFloat, Float};
use symbolica::evaluate::EvaluationDomain;

/// The settings of the TIR backend. Every value is overridable; the
/// defaults are TIRust's own (module docs), apart from the method choice
/// (on in binary64 with its rescaled acceptance frame, off in the extended
/// carriers; DECISIONS 2026-11-07 ~17:20 (d), 21:50 R4; module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TirOptions {
    /// Whether binary64 rows the conditioning gates refuse get the method
    /// choice and its acceptance check (default on, module docs; off: they
    /// fail closed with the gates' reason).
    pub method_choice: bool,
    /// The same for the extended carriers (double-double, MPFR); default
    /// off (not measured there: their gate-refused rows fail closed).
    pub extended_method_choice: bool,
    /// The binary64 conditioning gate of the first pass, set on every
    /// carrier's gates workspace: by default TIRust's direct engine default
    /// (1e-12, `DirectExecutor`'s `stability_threshold`; not a public
    /// constant at this pin, so the value is repeated here). Binary64 takes
    /// it as it is; the extended carriers take it to their precision
    /// ([`carrier_gate`]).
    pub gate: f64,
    /// The binary64 conditioning gate of the method-choice pass: TIRust's
    /// facade gate of its checked lanes (`dispatch/direct_hybrid.rs`
    /// `ROTATION_GATE`, 1e-15, validated there on boxes to heptagons; a
    /// private constant of TIRust at this pin, the value TIRust's embedder
    /// mode uses too). The extended carriers take it to their precision.
    pub method_choice_gate: f64,
    /// Extra triangle ranks of the small-Gram expansions (TIRust's facade
    /// `GRAM_BOOST`, 12; private at this pin, as `method_choice_gate`).
    pub expansion_boost: u16,
    /// `DirectWorkspace::set_alternative_first` of the method choice: the
    /// alternative PV first wherever its a priori estimate is at most this
    /// (DECISIONS 2026-11-07 ~10:05: [`iregi_rust::direct::EXPANSION_TOLERANCE`]).
    pub alternative_first: Option<f64>,
    /// Largest accepted deviation of a binary64 method-choice lane between
    /// its two frames, relative to the lane's term scale: by default the
    /// calibrated acceptance of the default target accuracy
    /// ([`acceptance_of`] of `reduction_policy::DEFAULT_TARGET`; a call's
    /// target sets it, [`TirOptions::for_target`]; overridable for experts);
    /// the extended carriers take it to their precision (where the method
    /// choice is off by default, so it acts on nothing).
    pub acceptance: f64,
    /// The frame in which every method-choice row is verified (the call's
    /// check row serves when the call checks in this frame): by default the
    /// rescaled rotation-by-one frame (`tir_frame(ReductionCheck::
    /// RESCALED)`; a rotated frame alone is blind on some wrong lanes,
    /// module docs).
    pub acceptance_frame: ReductionFrame,
    /// Whether a check frame that only relabels the propagators (every
    /// check without `rescale`, [`tir_frame`]) is TIRust's rotation copy from
    /// the direct pass's own calls (default on, module docs 'Check frames';
    /// bit for bit the separate call, 0.92x (gates) and 0.68x (method choice)
    /// of its cost in lit-tirust-store's timing); off: a separate call in that
    /// frame.
    pub rotation_copy: bool,
    /// TIRust's own profile of every moment call (`DirectWorkspace::
    /// set_profiling`: geometry, scalar masters, recursion, gather, output),
    /// added to the statistics as the "reduction: TIR ..." seconds; a
    /// measurement option (default off: its timers cost time), never a value
    /// change.
    pub profile: bool,
    /// Share TIRust's per-subset results (scalar masters, sub-integrals,
    /// gathered moments) between the families of a chunk of points through
    /// one `iregi_rust::TensorStore` per backend: every plan registered
    /// before the first reduce, every workspace attached, the store cleared
    /// once per point chunk ([`TirReductionBackend::begin`], and per
    /// family-group call). Every entry is a pure function of its exact key,
    /// so values are bit-identical with the store on and off (TIRust's
    /// rotation copy reads no store entry: its `check_view`); opt-in (m-lit
    /// lit-cache STEP 3: adopted only at >= 1.3x on the reduction stage,
    /// Q19).
    pub store: bool,
    /// With [`Self::store`]: the bound of a masters provider over a
    /// canonical cache (m-lit lit-cache STEP 3, PLAN §3.4: pyAmpliCol's
    /// `MasterCache` canonical order). The store gets a provider that
    /// declines every master (TIRust's kernels compute it: values
    /// unchanged) and counts the masters that reach it, i.e. the store's
    /// misses, and those equal to an earlier miss of the chunk up to the
    /// relabellings `MasterCache` canonicalises (the masters such a provider
    /// could serve). A measurement option (default off); check passes count
    /// too, so measure without checks.
    pub masters_census: bool,
    /// The UNCHECKED mode (`reduction_policy::DEFAULT_UNCHECKED`, off; m-lit
    /// lit-unchecked, module docs 'Unchecked mode'): the conditioning gates
    /// and the method choice stay as method selection; the acceptance pass
    /// of the method-choice rows and its comparison are never computed (the
    /// lanes are returned as computed: [`Self::acceptance`] and
    /// [`Self::acceptance_frame`] drive nothing); the rows the method choice
    /// refuses (where it is off for the carrier, the rows the gates refuse)
    /// are served by plain PV in a third workspace at gate 0, the last
    /// resort ("TIR last-resort rows").
    pub unchecked: bool,
    /// The VERIFIED LAST RESORT of the checked mode (module docs 'Verified
    /// last resort'; default on): the rows the method choice still refuses
    /// are served by the gate-0 last resort (plain PV, as in the unchecked
    /// mode) and every lane of such a row is then verified exactly as a
    /// method-choice lane, by its acceptance frame within
    /// [`Self::acceptance`] of its term scale; a lane that does not agree
    /// fails closed. Only where the method choice runs (binary64 by
    /// default): a carrier without it keeps failing its gate-refused rows
    /// closed. Off: those rows fail closed with the method choice's reason
    /// (the behaviour before tir-uuzggg).
    pub verified_last_resort: bool,
}

/// The calibration of the method-choice acceptance against a target
/// accuracy (m-lit F3 (a), lit-tirust-default): the largest measured ratio
/// of a true relative error to the frame deviation the acceptance compares
/// (the deviation between a method-choice lane's two frames relative to its
/// term scale, what [`TirOptions::acceptance`] bounds). The deviation
/// under-states the error by up to this factor, so admitting deviations up
/// to `target / UNDERSTATEMENT` keeps every calibration item served within
/// `target`. Measured against consistent truths (DECISIONS 2026-11-07 19:49
/// / 20:20): lit-tir-mp's synthetic method-choice lanes (the lanes its
/// rescaled acceptance refuses at the expansions' tolerance and a rotated one
/// serves, error against NinjaRS at 60 digits: largest ratio 579.2, error
/// 2.69e-6 at deviation 4.64e-9) and the loop-induced rows of the frozen
/// streams at the point level (|A_loop|^2 finite part against TIRust at 256
/// bits with supplied invariants; rust/loop-reduction/tests/fixtures/
/// tir_target_calibration.json, which a test reads against this value).
pub const UNDERSTATEMENT: f64 = 580.0;

/// The method-choice acceptance of a target accuracy (the relative accuracy
/// of each point's result, `reduction_policy::DEFAULT_TARGET`): `target /`
/// [`UNDERSTATEMENT`]. The one mapping: the runtime's and the SDKs' target,
/// and each rescue level's target in the Python API, go through it.
pub fn acceptance_of(target: f64) -> f64 {
    target / UNDERSTATEMENT
}

impl TirOptions {
    /// The default settings with the acceptance of `target`
    /// ([`acceptance_of`]).
    pub fn for_target(target: f64) -> Self {
        Self {
            acceptance: acceptance_of(target),
            ..Self::default()
        }
    }
}

impl Default for TirOptions {
    fn default() -> Self {
        Self {
            method_choice: true,
            extended_method_choice: false,
            gate: 1e-12,
            method_choice_gate: 1e-15,
            expansion_boost: 12,
            alternative_first: Some(iregi_rust::direct::EXPANSION_TOLERANCE),
            acceptance: acceptance_of(crate::reduction_policy::DEFAULT_TARGET),
            acceptance_frame: tir_frame(ReductionCheck::RESCALED),
            rotation_copy: true,
            profile: false,
            store: false,
            masters_census: false,
            unchecked: crate::reduction_policy::DEFAULT_UNCHECKED,
            verified_last_resort: true,
        }
    }
}

/// TIR's frame of a reduction check (module docs, 'Check frames'): the
/// cyclic relabelling by `check.rotation` (TIRust's rotation copy), or with
/// `check.rescale` that rotation with the origin shift at rescaled inputs.
pub fn tir_frame(check: ReductionCheck) -> ReductionFrame {
    ReductionFrame {
        reverse: false,
        rotation: check.rotation,
        shift: check.rescale,
        rescale: check.rescale,
    }
}

/// The rotation of a frame that only relabels the propagators cyclically
/// (TIRust's rotation copy, `DirectReductionPlan::rotated_batch`), or None.
fn relabelling(frame: ReductionFrame) -> Option<usize> {
    (!frame.reverse && !frame.shift && !frame.rescale).then_some(frame.rotation)
}

/// A binary64 threshold `g` taken to a carrier of `bits` bits: `g^(bits /
/// 53)`, the same share of the carrier's significand (module docs,
/// 'Carriers and gates'); `g` itself at 53 bits.
pub fn carrier_gate(g: f64, bits: u32) -> f64 {
    if bits == f64::MANTISSA_DIGITS {
        g
    } else {
        g.powf(f64::from(bits) / f64::from(f64::MANTISSA_DIGITS))
    }
}

/// [`carrier_gate`] in the carrier of `prototype` (its precision; a value
/// below binary64's range is formed in the carrier).
fn gate_in<T: Carrier>(prototype: &T, g: f64) -> T {
    let bits = prototype.get_precision();
    if bits == f64::MANTISSA_DIGITS {
        return T::from(g);
    }
    let exponent = f64::from(bits) / f64::from(f64::MANTISSA_DIGITS);
    // exp(ln(g) * bits / 53) at the carrier's precision.
    (prototype.zero() + &T::from(g.ln() * exponent)).exp()
}

/// The settings of the plan of `propagators` and `rank`: TIRust's one-loop
/// normalisation and Cartesian components, the supplied S-matrix and μ²
/// (module docs, 'Inputs').
fn settings(propagators: usize, rank: usize) -> RusticolResult<ReductionSettings> {
    let rank = u16::try_from(rank).map_err(|_| {
        RusticolError::unsupported_runtime_capability(
            "tirust",
            format!("numerator rank {rank} does not fit TIRust's ReductionSettings::rank"),
        )
    })?;
    let mut settings = ReductionSettings::new(propagators, rank);
    settings.convention = Convention::OneLoop;
    settings.tensor_weights = TensorWeights::Components;
    settings.invariants = InvariantInput::SuppliedInvariants;
    settings.scale_input = ScaleInput::MuSquared;
    Ok(settings)
}

/// TIRust's refusal of a family of `propagators` at numerator rank `rank`
/// in a carrier of `bits` bits, or `None` (module docs, 'Capability'):
/// TIRust's capability query of this backend's settings, its reason
/// carried (the artifacts' admission of TIRust, [`rank_refusal`], asks it
/// at binary64).
pub fn capability_refusal(propagators: usize, rank: usize, bits: u32) -> Option<String> {
    let settings = match settings(propagators, rank) {
        Ok(settings) => settings,
        Err(error) => return Some(error.message().to_owned()),
    };
    DirectCapabilities::of_representation(bits)
        .admits(&settings)
        .err()
        .map(|e| format!("TIRust cannot reduce {propagators} propagators at rank {rank}: {e}"))
}

/// TIRust's own structural bounds at a carrier of `bits` bits
/// (`DirectCapabilities::of_representation`): the largest number of
/// propagators and the largest tensor rank it represents (the numbers
/// [`capability_refusal`] decides on; never a pyAmpliCol constant).
pub fn capabilities(bits: u32) -> (usize, usize) {
    let found = DirectCapabilities::of_representation(bits);
    (found.max_propagators, usize::from(found.max_rank))
}

/// A carrier of the TIR backend: binary64, double-double or MPFR, each with
/// its own workspaces of a plan.
pub trait TirCarrier: TypedBatch {
    /// This carrier's workspaces among a plan's, `bits` its precision.
    fn slots(workspaces: &mut Workspaces, bits: u32) -> &mut Slots<Self>;

    /// Whether the method choice serves this carrier's gate-refused rows.
    fn method_choice(options: &TirOptions) -> bool {
        options.extended_method_choice
    }
}

impl TirCarrier for f64 {
    fn slots(workspaces: &mut Workspaces, _bits: u32) -> &mut Slots<Self> {
        &mut workspaces.double
    }

    fn method_choice(options: &TirOptions) -> bool {
        options.method_choice
    }
}

impl TirCarrier for DoubleFloat {
    fn slots(workspaces: &mut Workspaces, _bits: u32) -> &mut Slots<Self> {
        &mut workspaces.double_double
    }
}

impl TirCarrier for Float {
    fn slots(workspaces: &mut Workspaces, bits: u32) -> &mut Slots<Self> {
        workspaces.arbitrary.entry(bits).or_default()
    }
}

/// A plan's workspaces in one carrier: the conditioning gates' and the
/// method choice's, each built on first use; with a tensor store
/// ([`TirOptions::store`]) also both for the check passes, which take no
/// store (a check pass in another frame or order recomputes every integral,
/// so it never reads the primary pass's entries and keeps seeing their
/// rounding; TIRust's rotation copy inside a primary call reads none
/// either, its `check_view`); in the unchecked mode
/// ([`TirOptions::unchecked`]) also the last resort's (gate 0, no store).
pub struct Slots<T: Carrier> {
    gates: Option<DirectWorkspace<T>>,
    choice: Option<DirectWorkspace<T>>,
    check_gates: Option<DirectWorkspace<T>>,
    check_choice: Option<DirectWorkspace<T>>,
    last_resort: Option<DirectWorkspace<T>>,
}

impl<T: Carrier> Default for Slots<T> {
    fn default() -> Self {
        Self {
            gates: None,
            choice: None,
            check_gates: None,
            check_choice: None,
            last_resort: None,
        }
    }
}

/// A plan's workspaces per carrier (MPFR per precision).
#[derive(Default)]
pub struct Workspaces {
    double: Slots<f64>,
    double_double: Slots<DoubleFloat>,
    arbitrary: BTreeMap<u32, Slots<Float>>,
}

/// The plan of one (propagators, rank), its workspaces per carrier (built
/// on first use) and the permutation `position[k]` of NinjaRS monomial `k`
/// in the plan's basis.
struct Bin {
    plan: DirectReductionPlan,
    workspaces: Workspaces,
    position: Vec<usize>,
}

impl Bin {
    fn new(propagators: usize, rank: usize) -> RusticolResult<Self> {
        let plan = DirectReductionPlan::new(settings(propagators, rank)?).map_err(|e| {
            RusticolError::unsupported_runtime_capability(
                "tirust",
                format!("TIRust cannot reduce {propagators} propagators at rank {rank}: {e}"),
            )
        })?;
        let index: HashMap<[u32; 4], usize> = plan
            .basis()
            .iter()
            .enumerate()
            .map(|(b, m)| (m.0.map(u32::from), b))
            .collect();
        let position = monomials(rank)
            .iter()
            .map(|alpha| {
                index.get(alpha).copied().ok_or_else(|| {
                    RusticolError::internal(format!(
                        "TIRust's rank-{rank} basis lacks the monomial {alpha:?}"
                    ))
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        if plan.basis().len() != position.len() {
            return Err(RusticolError::internal(format!(
                "TIRust's rank-{rank} basis is not NinjaRS's tensor layout"
            )));
        }
        Ok(Self {
            plan,
            workspaces: Workspaces::default(),
            position,
        })
    }

    /// The carrier's gates (or, with `choice`, method-choice) workspace,
    /// built on first use at the precision of `prototype`; with a `store`,
    /// a `check` pass's own workspace without it ([`Slots`]; without a store
    /// a check pass shares the primary pass's workspaces).
    fn workspace<T: TirCarrier>(
        &mut self,
        choice: bool,
        check: bool,
        options: &TirOptions,
        prototype: &T,
        store: Option<&TensorStore>,
    ) -> RusticolResult<&mut DirectWorkspace<T>>
    where
        Complex<T>: EvaluationDomain,
    {
        let bits = prototype.get_precision();
        let slots = T::slots(&mut self.workspaces, bits);
        let check = check && store.is_some();
        let store = if check { None } else { store };
        let slot = match (choice, check) {
            (false, false) => &mut slots.gates,
            (true, false) => &mut slots.choice,
            (false, true) => &mut slots.check_gates,
            (true, true) => &mut slots.check_choice,
        };
        if slot.is_none() {
            let mut workspace = new_workspace(&self.plan, options, prototype, store)?;
            if choice {
                let gate = gate_in(prototype, options.method_choice_gate);
                workspace.set_stability_threshold(gate.clone());
                workspace.enable_estimated_method_choice(options.expansion_boost, gate);
                workspace.set_alternative_first(options.alternative_first);
            } else {
                // Every carrier from the option (binary64: the option's value
                // itself, TIRust's own default 1e-12 by default).
                workspace.set_stability_threshold(gate_in(prototype, options.gate));
            }
            *slot = Some(workspace);
        }
        Ok(slot.as_mut().expect("workspace"))
    }

    /// The carrier's last-resort workspace of the unchecked mode
    /// ([`TirOptions::unchecked`]): plain PV at gate 0 (no method choice;
    /// every `condition < 0` gate test is false, so only an exactly singular
    /// pivot or a non-finite value still refuses), built on first use at the
    /// precision of `prototype`, without the tensor store (its rows are
    /// rare, and a store entry never changes a value), so that primary and
    /// check passes share it.
    fn last_resort<T: TirCarrier>(
        &mut self,
        options: &TirOptions,
        prototype: &T,
    ) -> RusticolResult<&mut DirectWorkspace<T>>
    where
        Complex<T>: EvaluationDomain,
    {
        let slot = &mut T::slots(&mut self.workspaces, prototype.get_precision()).last_resort;
        if slot.is_none() {
            let mut workspace = new_workspace(&self.plan, options, prototype, None)?;
            workspace.set_stability_threshold(prototype.zero());
            *slot = Some(workspace);
        }
        Ok(slot.as_mut().expect("workspace"))
    }
}

/// A new workspace of `plan` at the precision of `prototype` (TIRust's
/// capability query of the plan at that precision first), with the options'
/// profiling and the backend's `store`.
fn new_workspace<T: TirCarrier>(
    plan: &DirectReductionPlan,
    options: &TirOptions,
    prototype: &T,
    store: Option<&TensorStore>,
) -> RusticolResult<DirectWorkspace<T>>
where
    Complex<T>: EvaluationDomain,
{
    let bits = prototype.get_precision();
    let settings = plan.settings();
    DirectReductionPlan::capabilities(prototype)
        .admits(settings)
        .map_err(|e| {
            RusticolError::unsupported_runtime_capability(
                "tirust",
                format!(
                    "TIRust cannot reduce {} propagators at rank {} at {bits} bits: {e}",
                    settings.propagators, settings.rank
                ),
            )
        })?;
    let mut workspace = plan
        .workspace(prototype.clone())
        .map_err(|e| RusticolError::unsupported_runtime_capability("tirust", e.to_string()))?;
    if options.profile {
        workspace.set_profiling(true);
    }
    if store.is_some() {
        workspace.set_tensor_store(store);
    }
    Ok(workspace)
}

/// One kinematic row: lanes `start..end` of group `group`, reduced at
/// `rank`.
#[derive(Clone, Copy)]
struct Row {
    group: usize,
    start: usize,
    end: usize,
    rank: usize,
}

/// A row's moments in NinjaRS order, per monomial `[finite, UV1 + IR1,
/// IR2, UV1]`, and whether the method-choice pass served it.
struct Moments<T> {
    values: Vec<[Complex<T>; 4]>,
    choice: bool,
}

/// A row's moments in one frame, with its frame map's origin and undo,
/// and, for a refused row, TIRust's conditioning estimate where it states
/// one (`last_conditions`, m-lit ux-polish item 4).
struct FrameRow<T> {
    moments: Result<Moments<T>, String>,
    origin: [Complex<T>; 4],
    undo: Option<Complex<T>>,
    measure: Option<LaneMeasure>,
}

/// The rows of one (propagators, rank) and their TIRust inputs.
type RowInputs<T> = (Vec<usize>, Vec<Complex<T>>);

/// A pass's rows in its frame and, with a rotation copy, the copy's rows.
type Pass<T> = (Vec<FrameRow<T>>, Option<Vec<FrameRow<T>>>);

/// Counters of the `reduce` calls since the last `take_statistics`.
#[derive(Default)]
struct Counters {
    moments: f64,
    contraction: f64,
    rows: u64,
    lanes: u64,
    shared: u64,
    gate_refusals: u64,
    choice_rows: u64,
    expansion_rows: u64,
    acceptance_rows: u64,
    unverified: u64,
    /// [`TirOptions::unchecked`] and [`TirOptions::verified_last_resort`]:
    /// rows the gate-0 last resort served.
    last_resort_rows: u64,
    /// Scalar-master evaluations of TIRust's moment calls.
    masters: u64,
    /// [`TirOptions::profile`]: TIRust's profile (geometry, masters,
    /// recursion, gather, output), in seconds.
    profile: Option<[f64; 5]>,
    /// [`TirOptions::store`]: the store's counts, summed over its clears
    /// (masters reused / computed, sub-integrals reused / computed, moments
    /// reused / computed).
    store: Option<[u64; 6]>,
    /// [`TirOptions::masters_census`]: per master family (tadpole, bubble,
    /// triangle, box) the store's master misses and those repeating an
    /// earlier miss of the chunk up to the census's relabellings.
    census: Option<[[u64; 2]; 4]>,
}

impl Counters {
    /// Add the store's counts since its last clear (before clearing it),
    /// and the census's ([`TirOptions::masters_census`]).
    fn store(&mut self, store: &TensorStore, census: Option<&MastersCensus>) {
        if let Some(census) = census {
            let counts = census.take();
            let total = self.census.get_or_insert([[0; 2]; 4]);
            for (t, c) in total.iter_mut().zip(counts) {
                t[0] += c[0];
                t[1] += c[1];
            }
        }
        let s = store.statistics();
        let counts = [
            s.masters_reused,
            s.masters_computed,
            s.subsets_reused,
            s.subsets_computed,
            s.moments_reused,
            s.moments_computed,
        ];
        let total = self.store.get_or_insert([0; 6]);
        for (t, c) in total.iter_mut().zip(counts) {
            *t += c;
        }
    }

    /// Add one TIRust call's scalar-master count, and its profile when
    /// profiling.
    fn call<T: TirCarrier>(&mut self, workspace: &DirectWorkspace<T>, profiling: bool)
    where
        Complex<T>: EvaluationDomain,
    {
        self.masters += workspace.last_master_rows() as u64;
        if profiling {
            let p = workspace.last_profile();
            let seconds = [p.geometry, p.masters, p.recurrence, p.gather, p.output];
            let total = self.profile.get_or_insert([0.; 5]);
            for (t, s) in total.iter_mut().zip(seconds) {
                *t += s.as_secs_f64();
            }
        }
    }
}

/// [`TirOptions::masters_census`]: the store's master misses of a chunk, by
/// canonical argument bits, and per master family the counts (misses,
/// canonical repeats). The relabellings are `MasterCache`'s for the
/// triangle and the box (`canonical_master_row`) and the exchange of the two
/// masses for the bubble, which the scalar B0 TIRust requests is symmetric
/// under (`MasterCache` keys its bubbles by argument order: their tensor
/// coefficients are not symmetric); a tadpole has none.
#[derive(Default)]
struct MastersCensus {
    state: std::sync::Mutex<CensusState>,
}

/// The census of one chunk: the canonical argument bits seen, the misses
/// and the repeats.
#[derive(Default)]
struct CensusState {
    seen: std::collections::HashSet<(u8, Vec<u64>)>,
    /// Per family code (0 tadpole, 1 bubble, 2 triangle, 3 box): misses,
    /// repeats.
    counts: [[u64; 2]; 4],
}

impl MastersCensus {
    /// The counts per family since the last call, and a new chunk.
    fn take(&self) -> [[u64; 2]; 4] {
        let mut state = self.state.lock().expect("census");
        let counts = state.counts;
        *state = CensusState::default();
        counts
    }
}

/// The census counts' names: the totals, then per family.
const CENSUS_NAMES: [[&str; 2]; 5] = [
    [
        "TIR masters census: store misses",
        "TIR masters census: canonical repeats",
    ],
    [
        "TIR masters census: tadpole store misses",
        "TIR masters census: tadpole canonical repeats",
    ],
    [
        "TIR masters census: bubble store misses",
        "TIR masters census: bubble canonical repeats",
    ],
    [
        "TIR masters census: triangle store misses",
        "TIR masters census: triangle canonical repeats",
    ],
    [
        "TIR masters census: box store misses",
        "TIR masters census: box canonical repeats",
    ],
];

impl MasterProvider for MastersCensus {
    fn master(
        &self,
        family: MasterFamily,
        arguments: &[Complex<f64>],
    ) -> Option<[Complex<f64>; 3]> {
        use ninjars::fast::{C64, FastMaster};
        let row: Vec<C64> = arguments.iter().map(|z| C64::new(z.re, z.im)).collect();
        let mut canonical = row.clone();
        let code = match family {
            MasterFamily::Triangle => {
                crate::runtime::canonical_master_row(FastMaster::C0, &row, &mut canonical);
                2
            }
            MasterFamily::Box => {
                crate::runtime::canonical_master_row(FastMaster::D0, &row, &mut canonical);
                3
            }
            // The scalar B0 is symmetric in its two masses.
            MasterFamily::Bubble => {
                if (row[2].re.to_bits(), row[2].im.to_bits())
                    < (row[1].re.to_bits(), row[1].im.to_bits())
                {
                    canonical.swap(1, 2);
                }
                1
            }
            MasterFamily::Tadpole => 0,
        };
        let bits = canonical
            .iter()
            .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
            .collect();
        let mut state = self.state.lock().expect("census");
        state.counts[usize::from(code)][0] += 1;
        if !state.seen.insert((code, bits)) {
            state.counts[usize::from(code)][1] += 1;
        }
        None
    }
}

/// The TIR backend (module docs).
pub struct TirReductionBackend {
    options: TirOptions,
    bins: BTreeMap<(usize, usize), Bin>,
    counters: Counters,
    /// [`TirOptions::store`]: the backend's one tensor store.
    store: Option<TensorStore>,
    /// [`TirOptions::masters_census`]: the store's census provider.
    census: Option<std::sync::Arc<MastersCensus>>,
    /// The measures of the last `reduce`'s refused lanes
    /// ([`Self::take_lane_measures`]).
    measures: Vec<MeasuredLane>,
}

/// `|z|` in binary64.
fn norm<T: Carrier>(z: &Complex<T>) -> f64 {
    (z.re.clone() * &z.re + &(z.im.clone() * &z.im))
        .sqrt()
        .to_f64()
}

/// TIRust's conditioning estimate of the lanes of `result` it refused as
/// `Unstable` (`last_conditions` of `workspace`, the call's own; empty when
/// no lane is: nothing is copied on the common path).
fn conditions_of<T: TirCarrier>(
    result: &BatchResult<T>,
    workspace: &DirectWorkspace<T>,
) -> Vec<Option<f64>>
where
    Complex<T>: EvaluationDomain,
{
    if result
        .statuses
        .iter()
        .any(|status| matches!(status, LaneStatus::Unstable(_)))
    {
        workspace.last_conditions().to_vec()
    } else {
        Vec::new()
    }
}

/// The conditioning measure of lane `k` (`conditions_of`), if stated.
fn conditioning(conditions: &[Option<f64>], k: usize) -> Option<LaneMeasure> {
    conditions
        .get(k)
        .copied()
        .flatten()
        .map(|value| LaneMeasure {
            kind: MeasureKind::GramConditioning,
            value,
        })
}

fn status_text(status: &LaneStatus) -> String {
    match status {
        LaneStatus::Success => "success".into(),
        LaneStatus::InvalidInput(e) => format!("TIRust InvalidInput: {e}"),
        LaneStatus::MasterFailure(e) => format!("TIRust MasterFailure: {e}"),
        LaneStatus::Unstable(e) => format!("TIRust Unstable: {e}"),
    }
}

/// A gates pass's lanes into the rows `members[k]` of `out`: a served
/// lane's moments, any other status the row's reason (and an `Unstable`
/// lane's conditioning measure, `conditions_of`; empty where TIRust's
/// diagnostics describe another result). Returns the lanes the gates
/// refused (`Unstable`: for the method choice), ascending.
fn gated<T: Carrier>(
    result: &BatchResult<T>,
    members: &[usize],
    position: &[usize],
    conditions: &[Option<f64>],
    out: &mut [FrameRow<T>],
) -> Vec<usize> {
    let mut refused = Vec::new();
    for (k, &index) in members.iter().enumerate() {
        match result.lane(k) {
            Ok(values) => {
                out[index].moments = Ok(Moments {
                    values: permuted(values, position),
                    choice: false,
                });
            }
            Err(status @ LaneStatus::Unstable(_)) => {
                out[index].moments = Err(status_text(status));
                out[index].measure = conditioning(conditions, k);
                refused.push(k);
            }
            Err(status) => out[index].moments = Err(status_text(status)),
        }
    }
    refused
}

/// A method-choice pass's lanes into `out`: lane `r` of `result` is the
/// gate-refused row `rows[r]`; `flagged` the lanes TIRust served with
/// small-Gram expansions (`last_gram_lanes`; empty where TIRust's
/// diagnostics describe another result: the rotation copy's) and
/// `conditions` their conditioning measures (`conditions_of`; empty
/// alike); a lane refused again keeps the gates' measure unless the method
/// choice states its own. Returns the lanes `r` the method choice refused
/// (`Unstable`: for the unchecked mode's last resort), ascending.
fn retried<T: Carrier>(
    result: &BatchResult<T>,
    rows: &[usize],
    position: &[usize],
    (flagged, conditions): (&[bool], &[Option<f64>]),
    out: &mut [FrameRow<T>],
    counters: &mut Counters,
) -> Vec<usize> {
    let mut refused = Vec::new();
    for (r, &index) in rows.iter().enumerate() {
        match result.lane(r) {
            Ok(values) => {
                counters.choice_rows += 1;
                if flagged.get(r).copied().unwrap_or(false) {
                    counters.expansion_rows += 1;
                }
                out[index].moments = Ok(Moments {
                    values: permuted(values, position),
                    choice: true,
                });
                out[index].measure = None;
            }
            Err(status) => {
                if matches!(status, LaneStatus::Unstable(_)) {
                    refused.push(r);
                    if let Some(measure) = conditioning(conditions, r) {
                        out[index].measure = Some(measure);
                    }
                }
                out[index].moments = Err(format!(
                    "{} (refused by the gates, then by the method choice: {})",
                    match &out[index].moments {
                        Err(first) => first.clone(),
                        Ok(_) => String::new(),
                    },
                    status_text(status)
                ));
            }
        }
    }
    refused
}

/// The last-resort pass into `out` (module docs 'Unchecked mode' and
/// 'Verified last resort'): lane `r` of `result` is the refused row
/// `rows[r]`, served by plain PV at gate 0 (with `verify`, the checked
/// mode's verified last resort, marked for the acceptance comparison of a
/// method-choice row), or still refused (its reason extended).
fn resorted<T: Carrier>(
    result: &BatchResult<T>,
    rows: &[usize],
    position: &[usize],
    verify: bool,
    out: &mut [FrameRow<T>],
    counters: &mut Counters,
) {
    for (r, &index) in rows.iter().enumerate() {
        match result.lane(r) {
            Ok(values) => {
                counters.last_resort_rows += 1;
                out[index].moments = Ok(Moments {
                    values: permuted(values, position),
                    choice: verify,
                });
                out[index].measure = None;
            }
            Err(status) => {
                if let Err(reason) = &mut out[index].moments {
                    reason.push_str(&format!(
                        " (then by the gate-0 last resort: {})",
                        status_text(status)
                    ));
                }
            }
        }
    }
}

impl TirReductionBackend {
    pub fn new(options: TirOptions) -> Self {
        let census = (options.store && options.masters_census)
            .then(|| std::sync::Arc::new(MastersCensus::default()));
        Self {
            options,
            bins: BTreeMap::new(),
            counters: Counters::default(),
            store: options.store.then(|| {
                TensorStore::with_options(TensorStoreOptions {
                    masters: census
                        .clone()
                        .map(|census| census as std::sync::Arc<dyn MasterProvider>),
                    ..Default::default()
                })
            }),
            census,
            measures: Vec::new(),
        }
    }

    /// The measures of the refused lanes of the last [`Self::reduce`]
    /// (m-lit ux-polish item 4): a method-choice lane refused by its
    /// acceptance, its deviation between its two frames relative to its
    /// term scale; any other refused lane, TIRust's conditioning estimate
    /// where it states one (`last_conditions`). The reason texts are
    /// unchanged.
    pub fn take_lane_measures(&mut self) -> Vec<MeasuredLane> {
        std::mem::take(&mut self.measures)
    }

    /// A backend whose plans are built for every (propagators, rank up to
    /// the declared one) of `families` (an artifact's histogram of
    /// (propagators, declared rank)) in a carrier of `bits` bits: TIRust's
    /// typed refusal of one of them is the capability refusal of the
    /// artifact.
    pub fn with_histogram(
        families: impl IntoIterator<Item = (usize, usize)>,
        bits: u32,
        options: TirOptions,
    ) -> RusticolResult<Self> {
        if options.masters_census && !options.store {
            // The census counts the store's misses: without the store there
            // is nothing to count (never a silently ignored option).
            return Err(RusticolError::invalid_argument(
                "TIRust's masters census (masters_census) counts the tensor store's \
                 master misses: it needs the store (store=True)",
            ));
        }
        let mut backend = Self::new(options);
        backend.prepare(families, bits)?;
        Ok(backend)
    }

    /// Build the plans of `families` ([`Self::with_histogram`]).
    pub fn prepare(
        &mut self,
        families: impl IntoIterator<Item = (usize, usize)>,
        bits: u32,
    ) -> RusticolResult<()> {
        for (n, declared) in families {
            for rank in 0..=declared {
                if let Some(refusal) = capability_refusal(n, rank, bits) {
                    return Err(RusticolError::unsupported_runtime_capability(
                        "tirust", refusal,
                    ));
                }
                self.bin(n, rank)?;
            }
        }
        Ok(())
    }

    /// The (propagators, rank) plans built.
    pub fn plans(&self) -> Vec<(usize, usize)> {
        self.bins.keys().copied().collect()
    }

    pub fn options(&self) -> &TirOptions {
        &self.options
    }

    /// Set the method-choice acceptance ([`TirOptions::acceptance`]: a
    /// call's target accuracy, [`acceptance_of`]); the plans stay (the
    /// acceptance only admits lanes, never changes a value).
    pub fn set_acceptance(&mut self, acceptance: f64) {
        self.options.acceptance = acceptance;
    }

    /// Set the unchecked mode ([`TirOptions::unchecked`]) of the following
    /// reductions; the plans and workspaces stay (the last resort's are
    /// built on first use).
    pub fn set_unchecked(&mut self, unchecked: bool) {
        self.options.unchecked = unchecked;
    }

    fn bin(&mut self, n: usize, rank: usize) -> RusticolResult<&mut Bin> {
        bin_of(&mut self.bins, n, rank, self.store.as_ref())
    }

    /// A new chunk of points: the store ([`TirOptions::store`]) forgets the
    /// last chunk's entries (its counts are kept in the statistics).
    pub fn begin(&mut self) {
        if let Some(store) = &self.store {
            self.counters.store(store, self.census.as_deref());
            store.clear();
        }
    }

    pub fn take_statistics(&mut self) -> BackendStatistics {
        if let Some(store) = &self.store {
            self.counters.store(store, self.census.as_deref());
            store.clear();
        }
        let c = std::mem::take(&mut self.counters);
        let mut seconds = vec![
            ("reduction: tensor moments", c.moments),
            ("reduction: contraction", c.contraction),
        ];
        if let Some([geometry, masters, recursion, gather, output]) = c.profile {
            seconds.extend([
                ("reduction: TIR geometry", geometry),
                ("reduction: TIR masters", masters),
                ("reduction: TIR recursion", recursion),
                ("reduction: TIR gather", gather),
                ("reduction: TIR output", output),
            ]);
        }
        BackendStatistics {
            seconds,
            counts: vec![
                ("tensor moment rows", c.rows),
                ("contracted lanes", c.lanes),
                ("moment rows shared", c.shared),
                ("TIR gate refusals", c.gate_refusals),
                ("TIR method-choice rows", c.choice_rows),
                ("TIR method-choice rows with expansions", c.expansion_rows),
                ("TIR method-choice check rows", c.acceptance_rows),
                (
                    "TIR method-choice lanes refused by their check",
                    c.unverified,
                ),
                ("TIR master requests", c.masters),
            ]
            .into_iter()
            // The last resort's count: the unchecked mode's, and the checked
            // mode's verified last resort when it served a row (absent
            // otherwise).
            .chain(
                (self.options.unchecked || c.last_resort_rows > 0)
                    .then_some(("TIR last-resort rows", c.last_resort_rows)),
            )
            .chain(c.store.into_iter().flat_map(|[mr, mc, sr, sc, or, oc]| {
                [
                    ("TIR store masters reused", mr),
                    ("TIR store masters computed", mc),
                    ("TIR store sub-integrals reused", sr),
                    ("TIR store sub-integrals computed", sc),
                    ("TIR store moments reused", or),
                    ("TIR store moments computed", oc),
                ]
            }))
            .chain(c.census.into_iter().flat_map(|families| {
                let total = families
                    .iter()
                    .fold([0; 2], |[m, r], [fm, fr]| [m + fm, r + fr]);
                std::iter::once(total)
                    .chain(families)
                    .zip(CENSUS_NAMES)
                    .flat_map(|([misses, repeats], [m, r])| [(m, misses), (r, repeats)])
                    .collect::<Vec<_>>()
            }))
            .collect(),
        }
    }

    /// The tensor moments of `rows` in `frame` (None: direct): the gates
    /// first, the method choice on the rows they refuse (when it serves the
    /// carrier of `prototype`). With `copy`, also those of every row's
    /// rotation copy by `copy` (the frame relabelled cyclically, module docs
    /// 'Check frames'), from the same TIRust calls and through the same two
    /// passes.
    fn moments<T: TirCarrier>(
        &mut self,
        groups: &[FamilyLanes<T>],
        rows: &[Row],
        frame: Option<ReductionFrame>,
        copy: Option<usize>,
        prototype: &T,
    ) -> RusticolResult<Pass<T>>
    where
        Complex<T>: EvaluationDomain,
    {
        let started = Instant::now();
        let mut out: Vec<FrameRow<T>> = Vec::with_capacity(rows.len());
        // Per (propagators, rank) the rows' indices and TIRust inputs.
        let mut bins: BTreeMap<(usize, usize), RowInputs<T>> = BTreeMap::new();
        let (mut shifts, mut masses, mut invariants) = (Vec::new(), Vec::new(), Vec::new());
        for (index, row) in rows.iter().enumerate() {
            let g = &groups[row.group];
            let n = g.propagators;
            let lane = row.start;
            let map = FrameMap::new(frame, n, g.shifts.first());
            shifts.clear();
            masses.clear();
            invariants.clear();
            let origin = map.shifts(&g.shifts[lane * 4 * n..(lane + 1) * 4 * n], &mut shifts);
            map.masses(&g.masses[lane * n..(lane + 1) * n], &mut masses);
            map.s_matrix(
                &g.s_matrix[lane * n * n..(lane + 1) * n * n],
                &mut invariants,
            );
            let scale = map.squared(&g.scales[lane]);
            out.push(FrameRow {
                moments: Err(String::new()),
                origin,
                undo: map.undo(),
                measure: None,
            });
            let (members, values) = bins.entry((n, row.rank)).or_default();
            members.push(index);
            values.extend(shifts.iter().cloned());
            // The supplied invariants s_ij, i < j, row-major (TIRust's
            // `InvariantInput::invariant_index`).
            for i in 0..n {
                values.extend(invariants[i * n + i + 1..(i + 1) * n].iter().cloned());
            }
            values.extend(masses.iter().cloned());
            // μ² itself (`ScaleInput::MuSquared`): real and positive.
            values.push(Complex::new(scale.re, prototype.zero()));
        }
        // A relabelled row stays in its row's loop-momentum frame: the same
        // origin and undo.
        let mut copies: Option<Vec<FrameRow<T>>> = copy.map(|_| {
            out.iter()
                .map(|row| FrameRow {
                    moments: Err(String::new()),
                    origin: row.origin.clone(),
                    undo: row.undo.clone(),
                    measure: None,
                })
                .collect()
        });
        let tirust =
            |e: iregi_rust::ReductionError| RusticolError::evaluation(format!("TIRust: {e}"));
        let choice = T::method_choice(&self.options);
        for ((n, rank), (members, values)) in bins {
            let options = self.options;
            let bin = bin_of(&mut self.bins, n, rank, self.store.as_ref())?;
            let width = values.len() / members.len();
            let batch = KinematicsBatch {
                rows: members.len(),
                components: 1,
                values,
            };
            // A pass in another frame than the direct one is a check pass
            // (the call's check, the method-choice acceptance).
            let check = frame.is_some();
            let workspace =
                bin.workspace(false, check, &options, prototype, self.store.as_ref())?;
            let (result, copied) = match copy {
                Some(rotation) => {
                    let (result, copied) = workspace
                        .reduce_batch_with_rotation(&batch, rotation)
                        .map_err(tirust)?;
                    (result, Some(copied))
                }
                None => (workspace.reduce_batch(&batch).map_err(tirust)?, None),
            };
            self.counters.call(workspace, options.profile);
            // TIRust's diagnostics describe the primary.
            let conditions = conditions_of(&result, workspace);
            let refused = gated(&result, &members, &bin.position, &conditions, &mut out);
            let refused_copy = match (&copied, &mut copies) {
                (Some(copied), Some(copies)) => gated(copied, &members, &bin.position, &[], copies),
                _ => Vec::new(),
            };
            let passes = if copy.is_some() { 2 } else { 1 };
            self.counters.rows += (passes * members.len()) as u64;
            self.counters.gate_refusals += (refused.len() + refused_copy.len()) as u64;
            if refused.is_empty() && refused_copy.is_empty() {
                continue;
            }
            if !choice {
                if options.unchecked {
                    // The unchecked mode without the method choice: the
                    // last resort serves the gates' refusals.
                    self.last_resort(
                        (n, rank),
                        &batch,
                        &members,
                        [&refused, &refused_copy],
                        copy,
                        (&mut out, copies.as_mut()),
                        prototype,
                    )?;
                    continue;
                }
                let closed = |row: &mut FrameRow<T>| {
                    if let Err(reason) = &mut row.moments {
                        reason.push_str(
                            " (refused by the conditioning gates; TIRust's method choice is off \
                             in this carrier: it serves gate-passed rows only)",
                        );
                    }
                };
                for &k in &refused {
                    closed(&mut out[members[k]]);
                }
                if let Some(copies) = &mut copies {
                    for &k in &refused_copy {
                        closed(&mut copies[members[k]]);
                    }
                }
                continue;
            }
            let retry = |ks: &[usize]| KinematicsBatch {
                rows: ks.len(),
                components: 1,
                values: ks
                    .iter()
                    .flat_map(|&k| batch.values[k * width..(k + 1) * width].iter().cloned())
                    .collect(),
            };
            let indices = |ks: &[usize]| ks.iter().map(|&k| members[k]).collect::<Vec<_>>();
            // The method choice: one rotation-copy call for the rows refused
            // together with their copy, one call for those refused alone.
            let (both, alone): (Vec<usize>, Vec<usize>) = refused
                .iter()
                .partition(|&&k| refused_copy.binary_search(&k).is_ok());
            let copy_alone: Vec<usize> = refused_copy
                .iter()
                .copied()
                .filter(|k| refused.binary_search(k).is_err())
                .collect();
            // The rows the method choice refuses (unchecked mode: its last
            // resort's), direct and copies, as row indices `k` of the bin.
            let (mut last, mut last_copy) = (Vec::new(), Vec::new());
            if let (false, Some(rotation), Some(copies)) = (both.is_empty(), copy, &mut copies) {
                let workspace =
                    bin.workspace(true, check, &options, prototype, self.store.as_ref())?;
                let (result, copied) = workspace
                    .reduce_batch_with_rotation(&retry(&both), rotation)
                    .map_err(tirust)?;
                self.counters.call(workspace, options.profile);
                // TIRust's diagnostics describe the primary.
                let flagged = workspace.last_gram_lanes().to_vec();
                let conditions = conditions_of(&result, workspace);
                self.counters.rows += 2 * both.len() as u64;
                let (rows, position, c) = (indices(&both), &bin.position, &mut self.counters);
                let diagnostics = (flagged.as_slice(), conditions.as_slice());
                let refused = retried(&result, &rows, position, diagnostics, &mut out, c);
                last.extend(refused.into_iter().map(|r| both[r]));
                let refused = retried(&copied, &rows, position, (&[], &[]), copies, c);
                last_copy.extend(refused.into_iter().map(|r| both[r]));
            }
            if !alone.is_empty() {
                let workspace =
                    bin.workspace(true, check, &options, prototype, self.store.as_ref())?;
                let result = workspace.reduce_batch(&retry(&alone)).map_err(tirust)?;
                self.counters.call(workspace, options.profile);
                let flagged = workspace.last_gram_lanes().to_vec();
                let conditions = conditions_of(&result, workspace);
                self.counters.rows += alone.len() as u64;
                let (rows, position, c) = (indices(&alone), &bin.position, &mut self.counters);
                let diagnostics = (flagged.as_slice(), conditions.as_slice());
                let refused = retried(&result, &rows, position, diagnostics, &mut out, c);
                last.extend(refused.into_iter().map(|r| alone[r]));
            }
            if let (false, Some(rotation), Some(copies)) =
                (copy_alone.is_empty(), copy, &mut copies)
            {
                let rotated = bin
                    .plan
                    .rotated_batch(&retry(&copy_alone), rotation)
                    .map_err(tirust)?;
                // The copy rows refused alone: a check computation of their
                // own (the check pass's workspace).
                let workspace =
                    bin.workspace(true, true, &options, prototype, self.store.as_ref())?;
                let result = workspace.reduce_batch(&rotated).map_err(tirust)?;
                self.counters.call(workspace, options.profile);
                let flagged = workspace.last_gram_lanes().to_vec();
                let conditions = conditions_of(&result, workspace);
                self.counters.rows += copy_alone.len() as u64;
                let (rows, position, c) = (indices(&copy_alone), &bin.position, &mut self.counters);
                let diagnostics = (flagged.as_slice(), conditions.as_slice());
                let refused = retried(&result, &rows, position, diagnostics, copies, c);
                last_copy.extend(refused.into_iter().map(|r| copy_alone[r]));
            }
            if (options.unchecked || options.verified_last_resort)
                && !(last.is_empty() && last_copy.is_empty())
            {
                last.sort_unstable();
                last_copy.sort_unstable();
                self.last_resort(
                    (n, rank),
                    &batch,
                    &members,
                    [&last, &last_copy],
                    copy,
                    (&mut out, copies.as_mut()),
                    prototype,
                )?;
            }
        }
        self.counters.moments += started.elapsed().as_secs_f64();
        Ok((out, copies))
    }

    /// The unchecked mode's last resort (module docs 'Unchecked mode') in
    /// the bin `key`: the rows `rows[0]` of its `batch` (row `k` is
    /// `out[members[k]]`) and the rotation copies (by `copy`) of the rows
    /// `rows[1]` (into `copies`), by plain PV at gate 0
    /// ([`Bin::last_resort`]).
    #[allow(clippy::too_many_arguments)]
    fn last_resort<T: TirCarrier>(
        &mut self,
        key: (usize, usize),
        batch: &KinematicsBatch<T>,
        members: &[usize],
        rows: [&[usize]; 2],
        copy: Option<usize>,
        (out, copies): (&mut [FrameRow<T>], Option<&mut Vec<FrameRow<T>>>),
        prototype: &T,
    ) -> RusticolResult<()>
    where
        Complex<T>: EvaluationDomain,
    {
        let tirust =
            |e: iregi_rust::ReductionError| RusticolError::evaluation(format!("TIRust: {e}"));
        let options = self.options;
        let bin = bin_of(&mut self.bins, key.0, key.1, self.store.as_ref())?;
        let width = batch.values.len() / batch.rows;
        let subset = |ks: &[usize]| KinematicsBatch {
            rows: ks.len(),
            components: 1,
            values: ks
                .iter()
                .flat_map(|&k| batch.values[k * width..(k + 1) * width].iter().cloned())
                .collect(),
        };
        let [direct, copied] = rows;
        // The copies' rows relabelled first (TIRust's rotation copy is bit
        // for bit a separate call on the relabelled rows).
        let rotated = match (copied.is_empty(), copy) {
            (false, Some(rotation)) => Some(
                bin.plan
                    .rotated_batch(&subset(copied), rotation)
                    .map_err(tirust)?,
            ),
            _ => None,
        };
        let workspace = bin.last_resort(&options, prototype)?;
        let mut results = [None, None];
        if !direct.is_empty() {
            results[0] = Some(workspace.reduce_batch(&subset(direct)).map_err(tirust)?);
            self.counters.call(workspace, options.profile);
            self.counters.rows += direct.len() as u64;
        }
        if let Some(rotated) = &rotated {
            results[1] = Some(workspace.reduce_batch(rotated).map_err(tirust)?);
            self.counters.call(workspace, options.profile);
            self.counters.rows += copied.len() as u64;
        }
        let indices = |ks: &[usize]| ks.iter().map(|&k| members[k]).collect::<Vec<_>>();
        let [direct_result, copy_result] = results;
        // The checked mode verifies a last-resort row's lanes (module docs
        // 'Verified last resort'); the check copies are the call's check
        // itself and are compared by the caller, as every copy row.
        let verify = !options.unchecked;
        if let Some(result) = direct_result {
            let rows = indices(direct);
            resorted(
                &result,
                &rows,
                &bin.position,
                verify,
                out,
                &mut self.counters,
            );
        }
        if let (Some(result), Some(copies)) = (copy_result, copies) {
            let rows = indices(copied);
            resorted(
                &result,
                &rows,
                &bin.position,
                false,
                copies,
                &mut self.counters,
            );
        }
        Ok(())
    }

    /// Reduce every lane of `groups` (module docs), in their carrier.
    pub fn reduce<T: TirCarrier>(
        &mut self,
        groups: &[FamilyLanes<T>],
        poles: Poles,
        check: Option<ReductionCheck>,
    ) -> RusticolResult<Reduced<T>>
    where
        Complex<T>: EvaluationDomain,
    {
        self.measures.clear();
        // The carrier (its precision) of the groups' values.
        let Some(prototype) = groups
            .iter()
            .find_map(|g| g.scales.first())
            .map(|scale| scale.re.zero())
        else {
            let empty = || groups.iter().map(|_| Vec::new()).collect();
            return Ok((empty(), check.map(|_| empty())));
        };
        // The kinematic rows and their moment ranks.
        let mut rows = Vec::new();
        for (index, g) in groups.iter().enumerate() {
            let declared = ncoeff(g.rank);
            if g.coefficients.len() != g.keys.len() * declared {
                return Err(RusticolError::internal(
                    "TIR: family coefficients are not binomial(rank + 4, 4) per lane",
                ));
            }
            let mut start = 0;
            while start < g.keys.len() {
                let point = g.points[start];
                let end = (start..g.keys.len())
                    .find(|&k| g.points[k] != point)
                    .unwrap_or(g.keys.len());
                let rank = (start..end)
                    .map(|lane| {
                        crate::runtime::true_rank(
                            &g.coefficients[lane * declared..(lane + 1) * declared],
                            g.rank,
                        )
                    })
                    .max()
                    .unwrap_or(0);
                if let Some(refusal) = rank_refusal(Reducer::TiRust, g.propagators, rank) {
                    return Err(RusticolError::unsupported_runtime_capability(
                        "tirust", refusal,
                    ));
                }
                rows.push(Row {
                    group: index,
                    start,
                    end,
                    rank,
                });
                start = end;
            }
        }
        // The call's check of every row: TIRust's rotation copy from the
        // direct pass's own calls when its frame only relabels the
        // propagators, else a pass in its frame (module docs 'Check
        // frames'); and the acceptance frame of the method-choice rows (the
        // check row itself when the frames agree).
        let check_frame = check.map(tir_frame);
        let copy = check_frame
            .filter(|_| self.options.rotation_copy)
            .and_then(relabelling);
        let (direct, copied) = self.moments(groups, &rows, None, copy, &prototype)?;
        let checked = match (copied, check_frame) {
            (Some(copied), _) => Some(copied),
            (None, Some(frame)) => Some(
                self.moments(groups, &rows, Some(frame), None, &prototype)?
                    .0,
            ),
            (None, None) => None,
        };
        let acceptance = self.options.acceptance_frame;
        // The unchecked mode computes no acceptance pass and no comparison:
        // its method-choice lanes are returned as computed (module docs
        // 'Unchecked mode').
        let verify = !self.options.unchecked;
        let choice_rows: Vec<usize> = (0..rows.len())
            .filter(|&r| verify && matches!(&direct[r].moments, Ok(m) if m.choice))
            .collect();
        let own = if choice_rows.is_empty() || check_frame == Some(acceptance) {
            Vec::new()
        } else {
            self.counters.acceptance_rows += choice_rows.len() as u64;
            let subset: Vec<Row> = choice_rows.iter().map(|&r| rows[r]).collect();
            self.moments(groups, &subset, Some(acceptance), None, &prototype)?
                .0
        };
        let mut accepted: Vec<Option<&FrameRow<T>>> = vec![None; rows.len()];
        if check_frame == Some(acceptance) {
            if let Some(checked) = &checked {
                for &r in &choice_rows {
                    accepted[r] = Some(&checked[r]);
                }
            }
        } else {
            for (&r, row) in choice_rows.iter().zip(&own) {
                accepted[r] = Some(row);
            }
        }
        let tolerance = carrier_gate(self.options.acceptance, prototype.get_precision());
        let started = Instant::now();
        let width = poles.width();
        let mut direct_out: Vec<Vec<LaneOutcome<T>>> = groups
            .iter()
            .map(|g| Vec::with_capacity(g.keys.len()))
            .collect();
        let mut check_out: Option<Vec<Vec<LaneOutcome<T>>>> = check.map(|_| {
            groups
                .iter()
                .map(|g| Vec::with_capacity(g.keys.len()))
                .collect()
        });
        let mut shifted = Vec::new();
        for (r, row) in rows.iter().enumerate() {
            let g = &groups[row.group];
            let declared = ncoeff(g.rank);
            let monomials = ncoeff(row.rank);
            // A direct row serving more than one lane (PLAN §2.b.2 item 6).
            if row.end - row.start > 1 {
                self.counters.shared += 1;
            }
            let first = &direct[r];
            let frame_map = |frame: ReductionFrame| {
                FrameMap::<T>::new(Some(frame), g.propagators, g.shifts.first())
            };
            let check_map = checked.as_ref().zip(check_frame).map(|(_, f)| frame_map(f));
            // A method-choice row's acceptance moments taken back to the
            // direct frame once per row (`FrameMap::moments`, the transpose
            // of the lanes' re-expansion, with the frame's undo): each lane
            // then contracts its own coefficients with them.
            let acceptance_moments = accepted[r].map(|other| {
                other
                    .moments
                    .as_ref()
                    .map(|m| {
                        let mut moments =
                            frame_map(acceptance).moments(&m.values, row.rank, &other.origin);
                        if let Some(factor) = &other.undo {
                            for slot in &mut moments {
                                for v in slot.iter_mut() {
                                    *v = v.clone() * factor;
                                }
                            }
                        }
                        moments
                    })
                    .map_err(Clone::clone)
            });
            // The magnitudes of a method-choice row's moments per slot, once
            // per row (the lanes' term scales, `scale_of`); not formed in the
            // unchecked mode, which has no comparison.
            let magnitudes: Option<Vec<[f64; SLOTS]>> = match &first.moments {
                Ok(m) if verify && m.choice => Some(
                    m.values
                        .iter()
                        .map(|slots| std::array::from_fn(|s| norm(&slots[s])))
                        .collect(),
                ),
                _ => None,
            };
            for lane in row.start..row.end {
                let c = &g.coefficients[lane * declared..lane * declared + monomials];
                // A checked method-choice lane's every slot (its acceptance
                // compares them all; its value is their first `width`, the
                // same sums: `contract` forms each slot on its own); any other
                // lane's `width` slots, also every lane of the unchecked mode.
                let full = match &first.moments {
                    Ok(m) if verify && m.choice => Some(contract(&m.values, c, SLOTS)),
                    _ => None,
                };
                let value = first.moments.as_ref().map(|m| match &full {
                    Some(full) => full[..width].to_vec(),
                    None => contract(&m.values, c, width),
                });
                let mut in_frame = |other: &FrameRow<T>,
                                    map: &FrameMap<T>,
                                    width: usize|
                 -> Result<Vec<Complex<T>>, String> {
                    let moments = other.moments.as_ref().map_err(Clone::clone)?;
                    shifted.clear();
                    map.coefficients(c, row.rank, &other.origin, &mut shifted);
                    let mut value = contract(&moments.values, &shifted, width);
                    if let Some(factor) = &other.undo {
                        for v in &mut value {
                            *v = v.clone() * factor;
                        }
                    }
                    Ok(value)
                };
                let again = checked
                    .as_ref()
                    .zip(check_map.as_ref())
                    .map(|(checked, map)| in_frame(&checked[r], map, width));
                self.counters.lanes += 1;
                // The measure of a refused lane: the acceptance's deviation
                // (below), else the row's conditioning.
                let mut measure = first.measure;
                let verified = match (&first.moments, &value) {
                    (Ok(m), Ok(value)) if verify && m.choice => {
                        // The acceptance compares every Laurent slot the
                        // moments carry, whatever the call's pole mode, so
                        // that a lane is served or refused alike in every
                        // mode (its finite part identical, ADDENDUM I).
                        let second = acceptance_moments
                            .as_ref()
                            .expect("a method-choice row's acceptance row")
                            .as_ref()
                            .map(|moments| contract(moments, c, SLOTS));
                        let full = full.as_ref().expect("a method-choice lane's slots");
                        match second {
                            Ok(second) => {
                                let scale = scale_of(
                                    magnitudes
                                        .as_ref()
                                        .expect("a method-choice row's magnitudes"),
                                    c,
                                );
                                let deviation = full
                                    .iter()
                                    .zip(&second)
                                    .map(|(a, b)| norm(&(a.clone() - b)))
                                    .fold(0f64, f64::max);
                                if deviation <= tolerance * scale {
                                    Ok(value.clone())
                                } else {
                                    self.counters.unverified += 1;
                                    measure = Some(LaneMeasure {
                                        kind: MeasureKind::MethodChoiceDeviation,
                                        value: if scale > 0.0 {
                                            deviation / scale
                                        } else {
                                            f64::INFINITY
                                        },
                                    });
                                    Err(format!(
                                        "TIRust method-choice lane refused by its check: deviation \
                                         {deviation:e} of term scale {scale:e} between its two frames \
                                         exceeds {tolerance:e}"
                                    ))
                                }
                            }
                            Err(status) => {
                                self.counters.unverified += 1;
                                Err(format!(
                                    "TIRust method-choice lane unverified: its check row failed ({status})"
                                ))
                            }
                        }
                    }
                    (_, value) => value.clone().map_err(Clone::clone),
                };
                if let (Err(_), Some(measure)) = (&verified, measure) {
                    self.measures.push(MeasuredLane {
                        group: row.group,
                        lane,
                        check: false,
                        measure,
                    });
                }
                direct_out[row.group].push(verified);
                if let Some(check_out) = &mut check_out {
                    let again = again.expect("every row is checked");
                    if let (Err(_), Some(checked)) = (&again, &checked)
                        && let Some(measure) = checked[r].measure
                    {
                        self.measures.push(MeasuredLane {
                            group: row.group,
                            lane,
                            check: true,
                            measure,
                        });
                    }
                    check_out[row.group].push(again);
                }
            }
        }
        self.counters.contraction += started.elapsed().as_secs_f64();
        Ok((direct_out, check_out))
    }
}

impl TirReductionBackend {
    /// [`Self::reduce`] on the NinjaRS adapter's batch groups (`runtime::
    /// V2Reducer::reduce`'s input and output; the Python reference path's
    /// `V2TirReducer`), in their carrier: one family per group; consecutive
    /// lanes with bitwise equal shifts, masses and scale form one kinematic
    /// row.
    pub fn reduce_family_groups(
        &mut self,
        groups: &[crate::runtime::FamilyGroup<'_>],
        poles: Poles,
        check: Option<ReductionCheck>,
    ) -> RusticolResult<crate::runtime::V2Output> {
        use crate::runtime::Data;
        // One call is one set of points: the store starts empty.
        self.begin();
        match groups.first().map(|g| g.shifts.data()) {
            None | Some(Data::Double(_)) => self.family_groups::<f64>(groups, poles, check),
            Some(Data::DoubleDouble(_)) => self.family_groups::<DoubleFloat>(groups, poles, check),
            Some(Data::Arbitrary(_)) => self.family_groups::<Float>(groups, poles, check),
        }
    }

    fn family_groups<T: TirCarrier>(
        &mut self,
        groups: &[crate::runtime::FamilyGroup<'_>],
        poles: Poles,
        check: Option<ReductionCheck>,
    ) -> RusticolResult<crate::runtime::V2Output>
    where
        Complex<T>: EvaluationDomain,
    {
        use crate::runtime::{Batch, ReductionBatch, V2Output, tensor_rank};
        let values = |batch: &Batch| -> RusticolResult<Vec<Complex<T>>> {
            T::values(batch.data()).map(<[_]>::to_vec).ok_or_else(|| {
                RusticolError::invalid_argument("TIR: the family groups mix carriers")
            })
        };
        let bits = groups
            .first()
            .map_or(f64::MANTISSA_DIGITS, |g| g.shifts.precision_bits());
        let mut lanes = Vec::with_capacity(groups.len());
        let mut ids = Vec::with_capacity(groups.len());
        for (family, g) in groups.iter().enumerate() {
            let n = g.propagators;
            let rows = g.coefficients.rows();
            let rank = tensor_rank(g.coefficients.columns())
                .map_err(|e| RusticolError::invalid_argument(e.to_string()))?;
            let (shifts, masses, scales) = (
                values(g.shifts)?,
                values(g.masses_squared)?,
                values(g.scales_squared)?,
            );
            let mut points = Vec::with_capacity(rows);
            for lane in 0..rows {
                let same = lane > 0
                    && shifts[(lane - 1) * 4 * n..lane * 4 * n]
                        == shifts[lane * 4 * n..(lane + 1) * 4 * n]
                    && masses[(lane - 1) * n..lane * n] == masses[lane * n..(lane + 1) * n]
                    && scales[lane - 1] == scales[lane];
                points.push(if same { points[lane - 1] } else { lane });
            }
            ids.push(g.coefficients.ids().to_vec());
            lanes.push(FamilyLanes {
                family,
                propagators: n,
                rank,
                points,
                keys: (0..rows)
                    .map(|column| rusticol_core::loop_execution::v2::LaneKey {
                        family,
                        order: 0,
                        column,
                    })
                    .collect(),
                shifts,
                masses,
                scales,
                s_matrix: values(g.s_matrix)?,
                coefficients: values(g.coefficients)?,
            });
        }
        let (direct, again) = self.reduce(&lanes, poles, check)?;
        let width = poles.width();
        let batches = |outcomes: Vec<Vec<LaneOutcome<T>>>| {
            outcomes
                .into_iter()
                .zip(&ids)
                .map(|(group, ids)| {
                    let mut values = Vec::new();
                    let mut kept = Vec::new();
                    let mut statuses = Vec::with_capacity(group.len());
                    for (outcome, &id) in group.into_iter().zip(ids) {
                        match outcome {
                            Ok(value) => {
                                values.extend(value);
                                kept.push(id);
                                statuses.push((id, "success".to_owned()));
                            }
                            Err(status) => statuses.push((id, status)),
                        }
                    }
                    Ok(ReductionBatch {
                        values: Batch::typed(values, kept, width, bits)
                            .map_err(|e| RusticolError::internal(e.to_string()))?,
                        statuses,
                    })
                })
                .collect::<RusticolResult<Vec<_>>>()
        };
        Ok(V2Output {
            direct: batches(direct)?,
            check: again.map(batches).transpose()?,
        })
    }
}

/// The bin of (`n`, `rank`), built on first use.
fn bin_of<'a>(
    bins: &'a mut BTreeMap<(usize, usize), Bin>,
    n: usize,
    rank: usize,
    store: Option<&TensorStore>,
) -> RusticolResult<&'a mut Bin> {
    Ok(match bins.entry((n, rank)) {
        std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
        std::collections::btree_map::Entry::Vacant(entry) => {
            let bin = Bin::new(n, rank)?;
            // Registered when built: `prepare` builds every plan of the call
            // before the first reduce, so which moments meet does not depend
            // on the order of the families (TIRust's store docs).
            if let Some(store) = store {
                store.register(&bin.plan);
            }
            entry.insert(bin)
        }
    })
}

/// The Laurent slots of a row's moments (`[finite, UV1 + IR1, IR2, UV1]`).
const SLOTS: usize = 4;

/// TIRust's lane array (`[UV1, IR1, IR2, finite]` per basis entry) in
/// NinjaRS order: `position[k]` is monomial `k`'s basis entry.
fn permuted<T: Carrier>(values: &[Complex<T>], position: &[usize]) -> Vec<[Complex<T>; 4]> {
    position
        .iter()
        .map(|&b| {
            let s = &values[4 * b..4 * b + 4];
            [
                s[3].clone(),
                s[0].clone() + &s[1],
                s[2].clone(),
                s[0].clone(),
            ]
        })
        .collect()
}

/// `Σ_α c_α M_α` in the first `width` slots (finite, 1/ε, 1/ε², UV).
fn contract<T: Carrier>(
    moments: &[[Complex<T>; 4]],
    c: &[Complex<T>],
    width: usize,
) -> Vec<Complex<T>> {
    let zero = c.first().map_or_else(|| T::from(0.), |c| c.re.zero());
    let mut out = vec![Complex::new(zero.clone(), zero); width];
    for (m, c) in moments.iter().zip(c) {
        if c.re.is_zero() && c.im.is_zero() {
            continue;
        }
        for (slot, value) in out.iter_mut().enumerate() {
            *value += c.clone() * &m[slot];
        }
    }
    out
}

/// A lane's term scale `max_s Σ_α |c_α| |M_α[s]|` over every slot, from the
/// row's moment magnitudes `|M_α[s]|` (formed once per row).
fn scale_of<T: Carrier>(magnitudes: &[[f64; SLOTS]], c: &[Complex<T>]) -> f64 {
    let mut sums = [0f64; SLOTS];
    for (m, c) in magnitudes.iter().zip(c) {
        let size = norm(c);
        for (sum, m) in sums.iter_mut().zip(m) {
            *sum += size * m;
        }
    }
    sums.into_iter().fold(0f64, f64::max)
}

/// Position `position[k]` of NinjaRS monomial `k` in TIRust's rank-`rank`
/// basis (the fixed layout permutation, lit-spike-tir T3).
pub fn layout_positions(rank: usize) -> RusticolResult<Vec<usize>> {
    Ok(Bin::new(1, rank)?.position)
}
