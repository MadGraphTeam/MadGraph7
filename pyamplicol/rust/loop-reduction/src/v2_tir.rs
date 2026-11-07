// SPDX-License-Identifier: 0BSD

//! The TIR backend of the v2 executor (m-lit PLAN §2.b.2–§2.b.6): tensor
//! integrals from TIRust's direct engine (`iregi_rust::DirectWorkspace`),
//! contracted natively with every lane's numerator coefficients.
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
//! **Method choice and its check (lit-spike-tir F2, DECISIONS 2026-11-07
//! ~10:05).** Every row goes first through the conditioning gates of a
//! default workspace. Rows the gates refuse (`Unstable`) are retried in a
//! second workspace with TIRust's binary64 method choice: the a priori
//! (COLLIER-style) choice of small-Gram expansions or the alternative PV
//! reduction for the boxes and triangles whose PV estimate is too large,
//! with the alternative PV first wherever its estimate is within the
//! expansions' own tolerance (`DirectWorkspace::set_alternative_first`,
//! [`iregi_rust::direct::EXPANSION_TOLERANCE`]). TIRust leaves those values
//! unchecked ("callers verify the lanes"), so every lane of a method-choice
//! row is verified here, never by TIRust's hidden rotation test: its value
//! in a second frame of the same row (the caller's check frame when the call
//! asks for one, else [`TirOptions::acceptance_frame`]) must agree with the
//! direct one within [`TirOptions::acceptance`] of the lane's term scale
//! `Σ_α |c_α| |M_α|` (its size before the contraction's cancellations); a
//! lane that does not, or whose check row is refused, fails closed with its
//! reason. A row still refused after the method choice, or with a TIRust
//! `InvalidInput` / `MasterFailure` status, fails every lane of the row (a
//! best-effort array is never a value).
//!
//! **Check frames (lit-spike-tir F3).** `q -> -q` leaves the moments bitwise
//! (odd moments flip exactly), so TIR's frame of a [`ReductionCheck`] never
//! reverses: it is the cyclic rotation by `check.rotation` with the origin
//! moved to the frame's first propagator (another PV reference propagator;
//! the numerator re-expanded exactly), rescaled by `x = 7/10` with the exact
//! `x^(4-2N)` undo when `check.rescale` (the frames NinjaRS's adapter uses,
//! `runtime::frames`). The check rows run through the same two passes.
//!
//! **Supplied invariants (lit-spike-tir F1).** At this TIRust pin the direct
//! engine derives the invariants from the Cartesian shifts (`FromShifts`),
//! so an on-shell invariant equal to an adjacent internal squared mass (a
//! massive external leg next to a propagator of its own mass, e.g. a top in
//! its own gauge loop) comes out off by a few ulps and its masters leave the
//! exact on-shell branch. Such a row needs the supplied S-matrix (TIRust's
//! U1, m-lit `lit-tirust-io`, consumed by `lit-tir-mp`): until then its lanes
//! fail typed, a capability refusal read from the row's own exact invariants
//! and masses, never a process key.
//!
//! **Plans.** One `DirectReductionPlan` per (propagators, rank), built from
//! an artifact's (propagators, rank) histogram when the backend is created
//! ([`TirReductionBackend::with_histogram`]: every rank up to each family's
//! declared one, since rows reduce at their true rank) and on first use
//! otherwise. TIRust's typed plan-construction refusal (its `Unsupported`
//! reason) is the capability refusal, carried verbatim; no TIRust limit is
//! stated here.
//!
//! **Symbolica.** The binary64 direct engine with its native master kernels
//! builds and evaluates no Symbolica expression; nothing here takes
//! pyAmpliCol's library key (test T8 in `tests/unit`).

use crate::Complex;
use crate::runtime::frames::{FrameMap, ReductionFrame};
use iregi_rust::{
    Convention, DirectReductionPlan, DirectWorkspace, KinematicsBatch, LaneStatus,
    ReductionSettings, TensorWeights,
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

/// The settings of the TIR backend. Every value is overridable; the
/// defaults are TIRust's own (module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TirOptions {
    /// Whether rows the conditioning gates refuse get the method choice
    /// (on; off: they fail closed with the gates' reason, a diagnostic).
    pub method_choice: bool,
    /// Conditioning gate of the method-choice pass: TIRust's facade gate of
    /// its checked lanes (`dispatch/direct_hybrid.rs` `ROTATION_GATE`,
    /// 1e-15, validated there on boxes to heptagons); the first pass keeps
    /// the direct engine's default gate.
    pub method_choice_gate: f64,
    /// Extra triangle ranks of the small-Gram expansions (TIRust's facade
    /// `GRAM_BOOST`, 12).
    pub expansion_boost: u16,
    /// `DirectWorkspace::set_alternative_first`: the alternative PV first
    /// wherever its a priori estimate is at most this (DECISIONS 2026-11-07
    /// ~10:05: [`iregi_rust::direct::EXPANSION_TOLERANCE`]).
    pub alternative_first: Option<f64>,
    /// Largest accepted deviation of a method-choice lane between its two
    /// frames, relative to the lane's term scale: the expansions' own
    /// acceptance (`EXPANSION_TOLERANCE`).
    pub acceptance: f64,
    /// The second frame of a method-choice row when the call asks for no
    /// check: rotation by one with the origin shift.
    pub acceptance_frame: ReductionFrame,
}

impl Default for TirOptions {
    fn default() -> Self {
        Self {
            method_choice: true,
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

/// The settings of the plan of `propagators` and `rank`.
fn settings(propagators: usize, rank: usize) -> RusticolResult<ReductionSettings> {
    let rank = u16::try_from(rank).map_err(|_| {
        RusticolError::unsupported_runtime_capability(
            "tirust",
            format!("numerator rank {rank} does not fit TIRust's u16 rank"),
        )
    })?;
    let mut settings = ReductionSettings::new(propagators, rank);
    settings.convention = Convention::OneLoop;
    settings.tensor_weights = TensorWeights::Components;
    Ok(settings)
}

/// The plan of one (propagators, rank), its two workspaces (built on first
/// use) and the permutation `position[k]` of NinjaRS monomial `k` in the
/// plan's basis.
struct Bin {
    plan: DirectReductionPlan,
    gates: Option<DirectWorkspace<f64>>,
    choice: Option<DirectWorkspace<f64>>,
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
            gates: None,
            choice: None,
            position,
        })
    }

    fn workspace(
        &mut self,
        choice: bool,
        options: &TirOptions,
    ) -> RusticolResult<&mut DirectWorkspace<f64>> {
        let slot = if choice {
            &mut self.choice
        } else {
            &mut self.gates
        };
        if slot.is_none() {
            let mut workspace = self.plan.workspace(0.0f64).map_err(|e| {
                RusticolError::unsupported_runtime_capability("tirust", e.to_string())
            })?;
            if choice {
                workspace.set_stability_threshold(options.method_choice_gate);
                workspace.enable_estimated_method_choice(
                    options.expansion_boost,
                    options.method_choice_gate,
                );
                workspace.set_alternative_first(options.alternative_first);
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
struct Moments {
    values: Vec<[Complex<f64>; 4]>,
    choice: bool,
}

/// A row's moments in one frame, with its frame map's origin and undo.
struct FrameRow {
    moments: Result<Moments, String>,
    origin: [Complex<f64>; 4],
    undo: Option<Complex<f64>>,
}

/// The rows of one (propagators, rank) and their TIRust inputs.
type RowInputs = (Vec<usize>, Vec<Complex<f64>>);

/// Counters of the `reduce` calls since the last `take_statistics`.
#[derive(Default)]
struct Counters {
    moments: f64,
    contraction: f64,
    rows: u64,
    lanes: u64,
    gate_refusals: u64,
    choice_rows: u64,
    expansion_rows: u64,
    acceptance_rows: u64,
    unverified: u64,
    capability: u64,
}

/// The TIR backend (module docs): binary64.
pub struct TirReductionBackend {
    options: TirOptions,
    bins: BTreeMap<(usize, usize), Bin>,
    counters: Counters,
}

fn complex(z: &Complex<f64>) -> f64 {
    (z.re * z.re + z.im * z.im).sqrt()
}

fn status_text(status: &LaneStatus) -> String {
    match status {
        LaneStatus::Success => "success".into(),
        LaneStatus::InvalidInput(e) => format!("TIRust InvalidInput: {e}"),
        LaneStatus::MasterFailure(e) => format!("TIRust MasterFailure: {e}"),
        LaneStatus::Unstable(e) => format!("TIRust Unstable: {e}"),
    }
}

/// The capability refusal of a row whose exact invariants put an on-shell
/// leg next to a propagator of its mass (module docs, 'Supplied
/// invariants'), or None.
fn onshell_refusal(n: usize, masses: &[Complex<f64>], s: &[Complex<f64>]) -> Option<String> {
    for i in 0..n {
        for j in (0..n).filter(|&j| j != i) {
            let x = &s[i * n + j];
            if x.im != 0. || x.re == 0. {
                continue;
            }
            let equal = |m: &Complex<f64>| m.im == 0. && m.re == x.re;
            if equal(&masses[i]) || equal(&masses[j]) {
                return Some(format!(
                    "TIRust capability: the invariant (V_{i} - V_{j})^2 = {} equals the squared mass \
                     of propagator {} (an exact on-shell configuration); TIRust's direct engine \
                     derives invariants from the shifts at this pin and needs the supplied S-matrix \
                     (TIRust U1) to keep it on shell",
                    x.re,
                    if equal(&masses[i]) { i } else { j }
                ));
            }
        }
    }
    None
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
    /// (propagators, declared rank)): TIRust's typed refusal of one of them
    /// is the capability refusal of the artifact.
    pub fn with_histogram(
        families: impl IntoIterator<Item = (usize, usize)>,
        options: TirOptions,
    ) -> RusticolResult<Self> {
        let mut backend = Self::new(options);
        backend.prepare(families)?;
        Ok(backend)
    }

    /// Build the plans of `families` ([`Self::with_histogram`]).
    pub fn prepare(
        &mut self,
        families: impl IntoIterator<Item = (usize, usize)>,
    ) -> RusticolResult<()> {
        for (n, declared) in families {
            if let Some(refusal) = rank_refusal(Reducer::TiRust, n, declared) {
                return Err(RusticolError::unsupported_runtime_capability(
                    "tirust", refusal,
                ));
            }
            for rank in 0..=declared {
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
                ("TIR gate refusals", c.gate_refusals),
                ("TIR method-choice rows", c.choice_rows),
                ("TIR method-choice rows with expansions", c.expansion_rows),
                ("TIR method-choice check rows", c.acceptance_rows),
                (
                    "TIR method-choice lanes refused by their check",
                    c.unverified,
                ),
                ("TIR capability refusals", c.capability),
            ],
        }
    }

    /// The tensor moments of `rows` in `frame` (None: direct): the gates
    /// first, the method choice on the rows they refuse.
    fn moments(
        &mut self,
        groups: &[FamilyLanes<f64>],
        rows: &[Row],
        frame: Option<ReductionFrame>,
    ) -> RusticolResult<Vec<FrameRow>> {
        let started = Instant::now();
        let mut out: Vec<FrameRow> = Vec::with_capacity(rows.len());
        // Per (propagators, rank) the rows' indices and TIRust inputs.
        let mut bins: BTreeMap<(usize, usize), RowInputs> = BTreeMap::new();
        let (mut shifts, mut masses) = (Vec::new(), Vec::new());
        for (index, row) in rows.iter().enumerate() {
            let g = &groups[row.group];
            let n = g.propagators;
            let lane = row.start;
            let map = FrameMap::new(frame, n, g.shifts.first());
            shifts.clear();
            masses.clear();
            let origin = map.shifts(&g.shifts[lane * 4 * n..(lane + 1) * 4 * n], &mut shifts);
            map.masses(&g.masses[lane * n..(lane + 1) * n], &mut masses);
            let scale = map.squared(&g.scales[lane]);
            out.push(FrameRow {
                moments: Err(String::new()),
                origin,
                undo: map.undo(),
            });
            let s = &g.s_matrix[lane * n * n..(lane + 1) * n * n];
            if let Some(refusal) = onshell_refusal(n, &g.masses[lane * n..(lane + 1) * n], s) {
                self.counters.capability += 1;
                out[index].moments = Err(refusal);
                continue;
            }
            let (members, values) = bins.entry((n, row.rank)).or_default();
            members.push(index);
            values.extend(shifts.iter().copied());
            values.extend(masses.iter().copied());
            // TIRust takes the scale mu (not mu^2): one rounding.
            values.push(Complex::new(scale.re.sqrt(), 0.));
        }
        for ((n, rank), (members, values)) in bins {
            let options = self.options;
            let bin = bin_of(&mut self.bins, n, rank)?;
            let batch = KinematicsBatch {
                rows: members.len(),
                components: 1,
                values,
            };
            let width = 5 * n + 1;
            let result = bin
                .workspace(false, &options)?
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
            if refused.is_empty() || !options.method_choice {
                continue;
            }
            let retry = KinematicsBatch {
                rows: refused.len(),
                components: 1,
                values: refused
                    .iter()
                    .flat_map(|&k| batch.values[k * width..(k + 1) * width].iter().copied())
                    .collect(),
            };
            let workspace = bin.workspace(true, &options)?;
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

    /// Reduce every lane of `groups` (module docs).
    pub fn reduce(
        &mut self,
        groups: &[FamilyLanes<f64>],
        poles: Poles,
        check: Option<ReductionCheck>,
    ) -> RusticolResult<Reduced<f64>> {
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
        let direct = self.moments(groups, &rows, None)?;
        // The second frame: the caller's check of every row, else the
        // acceptance frame of the method-choice rows only.
        let (frame, checked): (ReductionFrame, Vec<usize>) = match check {
            Some(check) => (tir_frame(check), (0..rows.len()).collect()),
            None => (
                self.options.acceptance_frame,
                (0..rows.len())
                    .filter(|&r| matches!(&direct[r].moments, Ok(m) if m.choice))
                    .collect(),
            ),
        };
        if check.is_none() {
            self.counters.acceptance_rows += checked.len() as u64;
        }
        let second_rows: Vec<Row> = checked.iter().map(|&r| rows[r]).collect();
        let second = self.moments(groups, &second_rows, Some(frame))?;
        let mut second_of: Vec<Option<&FrameRow>> = vec![None; rows.len()];
        for (&r, row) in checked.iter().zip(&second) {
            second_of[r] = Some(row);
        }
        let started = Instant::now();
        let width = poles.width();
        let mut direct_out: Vec<Vec<LaneOutcome<f64>>> = groups
            .iter()
            .map(|g| Vec::with_capacity(g.keys.len()))
            .collect();
        let mut check_out: Option<Vec<Vec<LaneOutcome<f64>>>> = check.map(|_| {
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
            let first = &direct[r];
            let other = second_of[r];
            let map =
                other.map(|_| FrameMap::<f64>::new(Some(frame), g.propagators, g.shifts.first()));
            for lane in row.start..row.end {
                let c = &g.coefficients[lane * declared..lane * declared + monomials];
                let value = first
                    .moments
                    .as_ref()
                    .map(|m| contract(&m.values, c, width));
                let again = other.map(|other| -> Result<Vec<Complex<f64>>, String> {
                    let moments = other.moments.as_ref().map_err(Clone::clone)?;
                    shifted.clear();
                    map.as_ref().expect("a checked row's frame").coefficients(
                        c,
                        row.rank,
                        &other.origin,
                        &mut shifted,
                    );
                    let mut value = contract(&moments.values, &shifted, width);
                    if let Some(factor) = &other.undo {
                        for v in &mut value {
                            *v *= *factor;
                        }
                    }
                    Ok(value)
                });
                self.counters.lanes += 1;
                let verified = match (&first.moments, &value, &again) {
                    (Ok(m), Ok(value), Some(again)) if m.choice => match again {
                        Ok(again) => {
                            let scale = term_scale(&m.values, c, width);
                            let deviation = value
                                .iter()
                                .zip(again)
                                .map(|(a, b)| complex(&(*a - b)))
                                .fold(0f64, f64::max);
                            if deviation <= self.options.acceptance * scale {
                                Ok(value.clone())
                            } else {
                                self.counters.unverified += 1;
                                Err(format!(
                                    "TIRust method-choice lane refused by its check: deviation \
                                     {deviation:e} of term scale {scale:e} between its two frames \
                                     exceeds {:e}",
                                    self.options.acceptance
                                ))
                            }
                        }
                        Err(status) => {
                            self.counters.unverified += 1;
                            Err(format!(
                                "TIRust method-choice lane unverified: its check row failed ({status})"
                            ))
                        }
                    },
                    (_, value, _) => value.clone().map_err(Clone::clone),
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
    /// `V2TirReducer`): one family per group; consecutive lanes with bitwise
    /// equal shifts, masses and scale form one kinematic row.
    pub fn reduce_family_groups(
        &mut self,
        groups: &[crate::runtime::FamilyGroup<'_>],
        poles: Poles,
        check: Option<ReductionCheck>,
    ) -> RusticolResult<crate::runtime::V2Output> {
        use crate::runtime::{Batch, Data, ReductionBatch, V2Output, tensor_rank};
        let values = |batch: &Batch| -> RusticolResult<Vec<Complex<f64>>> {
            match batch.data() {
                Data::Double(values) => Ok(values.to_vec()),
                _ => Err(RusticolError::unsupported_precision(
                    "the TIRust reduction backend serves binary64 in this build",
                )),
            }
        };
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
        let batches = |outcomes: Vec<Vec<LaneOutcome<f64>>>| {
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
                        values: Batch::typed(values, kept, width, 53)
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
fn permuted(values: &[Complex<f64>], position: &[usize]) -> Vec<[Complex<f64>; 4]> {
    position
        .iter()
        .map(|&b| {
            let s = &values[4 * b..4 * b + 4];
            [s[3], s[0] + s[1], s[2], s[0]]
        })
        .collect()
}

/// `Σ_α c_α M_α` in the first `width` slots (finite, 1/ε, 1/ε², UV).
fn contract(moments: &[[Complex<f64>; 4]], c: &[Complex<f64>], width: usize) -> Vec<Complex<f64>> {
    let mut out = vec![Complex::new(0., 0.); width];
    for (m, c) in moments.iter().zip(c) {
        if c.re == 0. && c.im == 0. {
            continue;
        }
        for (slot, value) in out.iter_mut().enumerate() {
            *value += *c * m[slot];
        }
    }
    out
}

/// The lane's term scale: `max_s Σ_α |c_α| |M_α[s]|` over its `width` slots.
fn term_scale(moments: &[[Complex<f64>; 4]], c: &[Complex<f64>], width: usize) -> f64 {
    (0..width)
        .map(|slot| {
            moments
                .iter()
                .zip(c)
                .map(|(m, c)| complex(c) * complex(&m[slot]))
                .sum::<f64>()
        })
        .fold(0f64, f64::max)
}

/// Position `position[k]` of NinjaRS monomial `k` in TIRust's rank-`rank`
/// basis (the fixed layout permutation, lit-spike-tir T3).
pub fn layout_positions(rank: usize) -> RusticolResult<Vec<usize>> {
    Ok(Bin::new(1, rank)?.position)
}
