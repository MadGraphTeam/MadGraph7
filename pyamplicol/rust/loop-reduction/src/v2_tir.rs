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
//! workspace. Binary64 keeps TIRust's own default gate. The extended carriers
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
//! **Method choice (off by default on every carrier) and its check
//! (lit-spike-tir F2, DECISIONS 2026-11-07 ~10:05, ~17:20 (d), 21:50 R4).**
//! With [`TirOptions::method_choice`] (binary64) or
//! [`TirOptions::extended_method_choice`] on, the rows the gates refuse are
//! retried in a second workspace with TIRust's a priori (COLLIER-style)
//! method choice: small-Gram expansions or the alternative PV reduction for
//! the boxes and triangles whose PV estimate is too large, with the
//! alternative PV first wherever its estimate is within the expansions' own
//! tolerance (`DirectWorkspace::set_alternative_first`,
//! [`TirOptions::alternative_first`]). TIRust leaves those values unchecked
//! ("callers verify the lanes"; its convergence test bounds the expansions'
//! truncation, not the rounding of what they read), so every lane of a
//! method-choice row is verified here, never by TIRust's hidden rotation
//! test: its value in the acceptance frame of the same row
//! ([`TirOptions::acceptance_frame`]; the call's check row when the call
//! checks in that very frame) must agree with the direct one within
//! [`TirOptions::acceptance`] of the lane's term scale `Σ_α |c_α| |M_α|`
//! (its size before the contraction's cancellations); a lane that does not,
//! or whose acceptance row is refused, fails closed with its reason. A
//! rotated acceptance frame alone is blind on some wrong method-choice lanes
//! (massive boxes and pentagons refused on a massive sub-triangle; lit-tir
//! report §H): only a rescaled frame (masters recomputed) sees them. A row
//! still refused after the method choice, or with a TIRust `InvalidInput` /
//! `MasterFailure` status, fails every lane of the row (a best-effort array
//! is never a value).
//!
//! **Check frames (lit-spike-tir F3).** `q -> -q` leaves the moments bitwise
//! (odd moments flip exactly), so TIR's frame of a [`ReductionCheck`] never
//! reverses: it is the cyclic rotation by `check.rotation` with the origin
//! moved to the frame's first propagator (another PV reference propagator;
//! the numerator re-expanded exactly), rescaled by `x = 7/10` with the exact
//! `x^(4-2N)` undo when `check.rescale` (the frames NinjaRS's adapter uses,
//! `runtime::frames`; the supplied S-matrix relabelled and rescaled with
//! them). The check rows run through the same two passes.
//!
//! **Capability.** What TIRust serves is read from TIRust: its capability
//! query (`DirectCapabilities::admits` of the plan's settings at the
//! carrier's precision, [`capability_refusal`]) and its typed plan
//! construction, each refusal carried verbatim (`ReductionUnavailable` in
//! Python); the artifacts' rank admission ([`rank_refusal`], rank <= N + 2
//! until m-lit lit-eft) comes first. No TIRust limit is stated here; the
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
    Convention, DirectCapabilities, DirectReductionPlan, DirectWorkspace, InvariantInput,
    KinematicsBatch, LaneStatus, ReductionSettings, ScaleInput, TensorWeights,
};
use ninjars::fast::tensor_layout::{monomials, ncoeff};
use rusticol_core::loop_execution::v2::{
    BackendStatistics, FamilyLanes, LaneOutcome, Poles, Reduced,
};
use rusticol_core::loop_v2::bounds::{Reducer, rank_refusal};
use rusticol_core::loop_v2::retest::ReductionCheck;
use rusticol_core::{RusticolError, RusticolResult};
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;
use symbolica::domains::float::{DoubleFloat, Float};
use symbolica::evaluate::EvaluationDomain;

/// The settings of the TIR backend. Every value is overridable; the
/// defaults are TIRust's own (module docs), apart from the method choice,
/// off on every carrier (DECISIONS 2026-11-07 ~17:20 (d), 21:50 R4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TirOptions {
    /// Whether binary64 rows the conditioning gates refuse get the method
    /// choice (default off: they fail closed with the gates' reason; on: the
    /// method choice and its acceptance check, module docs).
    pub method_choice: bool,
    /// The same for the extended carriers (double-double, MPFR); default
    /// off.
    pub extended_method_choice: bool,
    /// The binary64 conditioning gate of the first pass: TIRust's direct
    /// engine default (1e-12; `EmbedderOptions::gate` `None`, not a public
    /// constant at this pin). Binary64 runs on TIRust's own default; the
    /// extended carriers take it to their precision ([`carrier_gate`]).
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
    /// its two frames, relative to the lane's term scale: the expansions'
    /// own acceptance (`EXPANSION_TOLERANCE`); the extended carriers take it
    /// to their precision.
    pub acceptance: f64,
    /// The frame in which every method-choice row is verified (the call's
    /// check row serves when the call checks in this frame).
    pub acceptance_frame: ReductionFrame,
}

impl Default for TirOptions {
    fn default() -> Self {
        Self {
            method_choice: false,
            extended_method_choice: false,
            gate: 1e-12,
            method_choice_gate: 1e-15,
            expansion_boost: 12,
            alternative_first: Some(iregi_rust::direct::EXPANSION_TOLERANCE),
            acceptance: iregi_rust::direct::EXPANSION_TOLERANCE,
            acceptance_frame: ReductionFrame {
                reverse: false,
                rotation: 1,
                shift: true,
                rescale: false,
            },
        }
    }
}

/// TIR's frame of a reduction check (module docs, 'Check frames').
pub fn tir_frame(check: ReductionCheck) -> ReductionFrame {
    ReductionFrame {
        reverse: false,
        rotation: check.rotation,
        shift: true,
        rescale: check.rescale,
    }
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
/// in a carrier of `bits` bits, or `None` (module docs, 'Capability'): the
/// artifacts' rank admission ([`rank_refusal`]), then TIRust's capability
/// query, its reason carried.
pub fn capability_refusal(propagators: usize, rank: usize, bits: u32) -> Option<String> {
    if let Some(refusal) = rank_refusal(Reducer::TiRust, propagators, rank) {
        return Some(refusal);
    }
    let settings = match settings(propagators, rank) {
        Ok(settings) => settings,
        Err(error) => return Some(error.message().to_owned()),
    };
    DirectCapabilities::of_representation(bits)
        .admits(&settings)
        .err()
        .map(|e| format!("TIRust cannot reduce {propagators} propagators at rank {rank}: {e}"))
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
/// method choice's, each built on first use.
pub struct Slots<T: Carrier> {
    gates: Option<DirectWorkspace<T>>,
    choice: Option<DirectWorkspace<T>>,
}

impl<T: Carrier> Default for Slots<T> {
    fn default() -> Self {
        Self {
            gates: None,
            choice: None,
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
    /// built on first use at the precision of `prototype`.
    fn workspace<T: TirCarrier>(
        &mut self,
        choice: bool,
        options: &TirOptions,
        prototype: &T,
    ) -> RusticolResult<&mut DirectWorkspace<T>>
    where
        Complex<T>: EvaluationDomain,
    {
        let bits = prototype.get_precision();
        let slots = T::slots(&mut self.workspaces, bits);
        let slot = if choice {
            &mut slots.choice
        } else {
            &mut slots.gates
        };
        if slot.is_none() {
            let settings = self.plan.settings();
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
            let mut workspace = self.plan.workspace(prototype.clone()).map_err(|e| {
                RusticolError::unsupported_runtime_capability("tirust", e.to_string())
            })?;
            let binary64 = bits == f64::MANTISSA_DIGITS;
            if choice {
                let gate = gate_in(prototype, options.method_choice_gate);
                workspace.set_stability_threshold(gate.clone());
                workspace.enable_estimated_method_choice(options.expansion_boost, gate);
                workspace.set_alternative_first(options.alternative_first);
            } else if !binary64 {
                workspace.set_stability_threshold(gate_in(prototype, options.gate));
            }
            *slot = Some(workspace);
        }
        Ok(slot.as_mut().expect("workspace"))
    }
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

/// A row's moments in one frame, with its frame map's origin and undo.
struct FrameRow<T> {
    moments: Result<Moments<T>, String>,
    origin: [Complex<T>; 4],
    undo: Option<Complex<T>>,
}

/// The rows of one (propagators, rank) and their TIRust inputs.
type RowInputs<T> = (Vec<usize>, Vec<Complex<T>>);

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
}

/// The TIR backend (module docs).
pub struct TirReductionBackend {
    options: TirOptions,
    bins: BTreeMap<(usize, usize), Bin>,
    counters: Counters,
}

/// `|z|` in binary64.
fn norm<T: Carrier>(z: &Complex<T>) -> f64 {
    (z.re.clone() * &z.re + &(z.im.clone() * &z.im))
        .sqrt()
        .to_f64()
}

fn status_text(status: &LaneStatus) -> String {
    match status {
        LaneStatus::Success => "success".into(),
        LaneStatus::InvalidInput(e) => format!("TIRust InvalidInput: {e}"),
        LaneStatus::MasterFailure(e) => format!("TIRust MasterFailure: {e}"),
        LaneStatus::Unstable(e) => format!("TIRust Unstable: {e}"),
    }
}

impl TirReductionBackend {
    pub fn new(options: TirOptions) -> Self {
        Self {
            options,
            bins: BTreeMap::new(),
            counters: Counters::default(),
        }
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

    fn bin(&mut self, n: usize, rank: usize) -> RusticolResult<&mut Bin> {
        bin_of(&mut self.bins, n, rank)
    }

    pub fn begin(&mut self) {}

    pub fn take_statistics(&mut self) -> BackendStatistics {
        let c = std::mem::take(&mut self.counters);
        BackendStatistics {
            seconds: vec![
                ("reduction: tensor moments", c.moments),
                ("reduction: contraction", c.contraction),
            ],
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
            ],
        }
    }

    /// The tensor moments of `rows` in `frame` (None: direct): the gates
    /// first, the method choice on the rows they refuse (when it serves the
    /// carrier of `prototype`).
    fn moments<T: TirCarrier>(
        &mut self,
        groups: &[FamilyLanes<T>],
        rows: &[Row],
        frame: Option<ReductionFrame>,
        prototype: &T,
    ) -> RusticolResult<Vec<FrameRow<T>>>
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
            map.s_matrix(&g.s_matrix[lane * n * n..(lane + 1) * n * n], &mut invariants);
            let scale = map.squared(&g.scales[lane]);
            out.push(FrameRow {
                moments: Err(String::new()),
                origin,
                undo: map.undo(),
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
        let choice = T::method_choice(&self.options);
        for ((n, rank), (members, values)) in bins {
            let options = self.options;
            let bin = bin_of(&mut self.bins, n, rank)?;
            let width = values.len() / members.len();
            let batch = KinematicsBatch {
                rows: members.len(),
                components: 1,
                values,
            };
            let result = bin
                .workspace(false, &options, prototype)?
                .reduce_batch(&batch)
                .map_err(|e| RusticolError::evaluation(format!("TIRust: {e}")))?;
            let mut refused = Vec::new();
            for (k, &index) in members.iter().enumerate() {
                match result.lane(k) {
                    Ok(values) => {
                        out[index].moments = Ok(Moments {
                            values: permuted(values, &bin.position),
                            choice: false,
                        });
                    }
                    Err(status @ LaneStatus::Unstable(_)) => {
                        out[index].moments = Err(status_text(status));
                        refused.push(k);
                    }
                    Err(status) => out[index].moments = Err(status_text(status)),
                }
            }
            self.counters.rows += members.len() as u64;
            self.counters.gate_refusals += refused.len() as u64;
            if refused.is_empty() {
                continue;
            }
            if !choice {
                for &k in &refused {
                    if let Err(reason) = &mut out[members[k]].moments {
                        reason.push_str(
                            " (refused by the conditioning gates; TIRust's method choice is off \
                             in this carrier: it serves gate-passed rows only)",
                        );
                    }
                }
                continue;
            }
            let retry = KinematicsBatch {
                rows: refused.len(),
                components: 1,
                values: refused
                    .iter()
                    .flat_map(|&k| batch.values[k * width..(k + 1) * width].iter().cloned())
                    .collect(),
            };
            let workspace = bin.workspace(true, &options, prototype)?;
            let result = workspace
                .reduce_batch(&retry)
                .map_err(|e| RusticolError::evaluation(format!("TIRust: {e}")))?;
            let flagged = workspace.last_gram_lanes().to_vec();
            self.counters.rows += refused.len() as u64;
            for (r, &k) in refused.iter().enumerate() {
                let index = members[k];
                match result.lane(r) {
                    Ok(values) => {
                        self.counters.choice_rows += 1;
                        if flagged.get(r).copied().unwrap_or(false) {
                            self.counters.expansion_rows += 1;
                        }
                        out[index].moments = Ok(Moments {
                            values: permuted(values, &bin.position),
                            choice: true,
                        });
                    }
                    Err(status) => {
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
        }
        self.counters.moments += started.elapsed().as_secs_f64();
        Ok(out)
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
        let direct = self.moments(groups, &rows, None, &prototype)?;
        // The call's check of every row, and the acceptance frame of the
        // method-choice rows (the check row itself when the frames agree).
        let check_frame = check.map(tir_frame);
        let checked = match check_frame {
            Some(frame) => Some(self.moments(groups, &rows, Some(frame), &prototype)?),
            None => None,
        };
        let acceptance = self.options.acceptance_frame;
        let choice_rows: Vec<usize> = (0..rows.len())
            .filter(|&r| matches!(&direct[r].moments, Ok(m) if m.choice))
            .collect();
        let own = if choice_rows.is_empty() || check_frame == Some(acceptance) {
            Vec::new()
        } else {
            self.counters.acceptance_rows += choice_rows.len() as u64;
            let subset: Vec<Row> = choice_rows.iter().map(|&r| rows[r]).collect();
            self.moments(groups, &subset, Some(acceptance), &prototype)?
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
            let acceptance_map = accepted[r].map(|_| frame_map(acceptance));
            for lane in row.start..row.end {
                let c = &g.coefficients[lane * declared..lane * declared + monomials];
                let value = first
                    .moments
                    .as_ref()
                    .map(|m| contract(&m.values, c, width));
                let mut in_frame =
                    |other: &FrameRow<T>, map: &FrameMap<T>| -> Result<Vec<Complex<T>>, String> {
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
                    .map(|(checked, map)| in_frame(&checked[r], map));
                self.counters.lanes += 1;
                let verified = match (&first.moments, &value) {
                    (Ok(m), Ok(value)) if m.choice => {
                        let other = accepted[r].expect("a method-choice row's acceptance row");
                        let map = acceptance_map.as_ref().expect("its frame");
                        let shared = check_frame == Some(acceptance);
                        let second = if shared {
                            again.clone().expect("the check row")
                        } else {
                            in_frame(other, map)
                        };
                        match second {
                            Ok(second) => {
                                let scale = term_scale(&m.values, c, width);
                                let deviation = value
                                    .iter()
                                    .zip(&second)
                                    .map(|(a, b)| norm(&(a.clone() - b)))
                                    .fold(0f64, f64::max);
                                if deviation <= tolerance * scale {
                                    Ok(value.clone())
                                } else {
                                    self.counters.unverified += 1;
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
                direct_out[row.group].push(verified);
                if let Some(check_out) = &mut check_out {
                    check_out[row.group].push(again.expect("every row is checked"));
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
        match groups.first().map(|g| g.shifts.data()) {
            None | Some(Data::Double(_)) => self.family_groups::<f64>(groups, poles, check),
            Some(Data::DoubleDouble(_)) => {
                self.family_groups::<DoubleFloat>(groups, poles, check)
            }
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
        let bits = groups.first().map_or(f64::MANTISSA_DIGITS, |g| g.shifts.precision_bits());
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
fn bin_of(
    bins: &mut BTreeMap<(usize, usize), Bin>,
    n: usize,
    rank: usize,
) -> RusticolResult<&mut Bin> {
    Ok(match bins.entry((n, rank)) {
        std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
        std::collections::btree_map::Entry::Vacant(entry) => entry.insert(Bin::new(n, rank)?),
    })
}

/// TIRust's lane array (`[UV1, IR1, IR2, finite]` per basis entry) in
/// NinjaRS order: `position[k]` is monomial `k`'s basis entry.
fn permuted<T: Carrier>(values: &[Complex<T>], position: &[usize]) -> Vec<[Complex<T>; 4]> {
    position
        .iter()
        .map(|&b| {
            let s = &values[4 * b..4 * b + 4];
            [s[3].clone(), s[0].clone() + &s[1], s[2].clone(), s[0].clone()]
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

/// The lane's term scale: `max_s Σ_α |c_α| |M_α[s]|` over its `width` slots.
fn term_scale<T: Carrier>(moments: &[[Complex<T>; 4]], c: &[Complex<T>], width: usize) -> f64 {
    (0..width)
        .map(|slot| {
            moments
                .iter()
                .zip(c)
                .map(|(m, c)| norm(c) * norm(&m[slot]))
                .sum::<f64>()
        })
        .fold(0f64, f64::max)
}

/// Position `position[k]` of NinjaRS monomial `k` in TIRust's rank-`rank`
/// basis (the fixed layout permutation, lit-spike-tir T3).
pub fn layout_positions(rank: usize) -> RusticolResult<Vec<usize>> {
    Ok(Bin::new(1, rank)?.position)
}
