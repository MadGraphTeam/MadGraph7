// SPDX-License-Identifier: 0BSD

//! The reduction backends of the native v2 executor
//! (`rusticol_core::loop_execution::v2`): the executor streams each bucket
//! chunk's family lanes into the [`V2Reducer`] (NinjaRS: true-rank buckets,
//! cached masters, reduction-direction check) as typed batches, without any
//! decimal round trip, and reads back each lane's status and Laurent
//! integrals; or into the TIR backend (`crate::v2_tir`, feature `tir`).
//! [`V2Backends`] is the one choice per evaluation ([`Reduction`]).

use super::batch::{Batch, BatchCarrier, Data};
use super::v2_reduce::{
    FamilyGroup, PoleSelection, Poles, ReduceOptions, ReductionFrame, V2Reducer,
};
use crate::Complex;
use rusticol_core::loop_execution::v2::{
    BackendStatistics, FamilyLanes, LaneOutcome, MeasuredLane, Reduced, V2Backend,
};
use rusticol_core::loop_v2::retest::ReductionCheck;
use rusticol_core::{RusticolError, RusticolResult};
use symbolica::domains::float::{DoubleFloat, Float};

/// Typed access to the values of a [`Batch`], and the carrier's TIR
/// reduction (every carrier, m-lit lit-tir-mp).
pub trait TypedBatch: BatchCarrier {
    fn values(data: &Data) -> Option<&[Complex<Self>]>;

    /// The TIR backend's reduction in this carrier.
    #[cfg(feature = "tir")]
    fn tir_reduce(
        backend: &mut crate::v2_tir::TirReductionBackend,
        groups: &[FamilyLanes<Self>],
        poles: Poles,
        check: Option<ReductionCheck>,
    ) -> RusticolResult<Reduced<Self>>;
}

impl TypedBatch for f64 {
    fn values(data: &Data) -> Option<&[Complex<Self>]> {
        match data {
            Data::Double(values) => Some(values),
            _ => None,
        }
    }

    #[cfg(feature = "tir")]
    fn tir_reduce(
        backend: &mut crate::v2_tir::TirReductionBackend,
        groups: &[FamilyLanes<Self>],
        poles: Poles,
        check: Option<ReductionCheck>,
    ) -> RusticolResult<Reduced<Self>> {
        backend.reduce(groups, poles, check)
    }
}

impl TypedBatch for DoubleFloat {
    fn values(data: &Data) -> Option<&[Complex<Self>]> {
        match data {
            Data::DoubleDouble(values) => Some(values),
            _ => None,
        }
    }

    #[cfg(feature = "tir")]
    fn tir_reduce(
        backend: &mut crate::v2_tir::TirReductionBackend,
        groups: &[FamilyLanes<Self>],
        poles: Poles,
        check: Option<ReductionCheck>,
    ) -> RusticolResult<Reduced<Self>> {
        backend.reduce(groups, poles, check)
    }
}

impl TypedBatch for Float {
    fn values(data: &Data) -> Option<&[Complex<Self>]> {
        match data {
            Data::Arbitrary(values) => Some(values),
            _ => None,
        }
    }

    #[cfg(feature = "tir")]
    fn tir_reduce(
        backend: &mut crate::v2_tir::TirReductionBackend,
        groups: &[FamilyLanes<Self>],
        poles: Poles,
        check: Option<ReductionCheck>,
    ) -> RusticolResult<Reduced<Self>> {
        backend.reduce(groups, poles, check)
    }
}

/// A reduction backend of the executor: one choice per evaluation, matched
/// once per `begin` (point chunk) and per `reduce` (bucket chunk), never per
/// point or lane (m-lit PLAN §2.a.6): `evaluate` is monomorphised once per
/// carrier, not once more per backend.
pub enum V2Backends {
    /// NinjaRS (integrand reduction), [`V2ReductionBackend`].
    NinjaRs(V2ReductionBackend),
    /// TIRust (tensor-integral reduction), [`crate::v2_tir`].
    #[cfg(feature = "tir")]
    TiRust(Box<crate::v2_tir::TirReductionBackend>),
}

/// A reduction backend by name: `"ninjars"` or `"tirust"` (the rank
/// admission's backend, one definition).
pub use rusticol_core::loop_v2::bounds::Reducer as Reduction;

/// The refusal of a TIRust request in a build without it.
pub const TIRUST_NOT_BUILT: &str =
    "this native library is built without TIRust (the cargo feature tir)";

/// The reduction backends this build serves, by name (`"ninjars"`,
/// `"tirust"`), with the carriers each serves: what the PyO3 capability
/// query (`reduction_backends`) reports. TIRust's direct engine serves
/// every carrier of the executor (`v2_tir`; its own capability query
/// decides per family, [`crate::v2_tir::capability_refusal`]).
pub fn backend_carriers() -> Vec<(&'static str, Vec<&'static str>)> {
    let carriers = vec!["double", "double-double", "arbitrary"];
    let mut out = vec![("ninjars", carriers.clone())];
    if crate::tir::available() {
        out.push(("tirust", carriers));
    }
    out
}

/// `None` if `reduction` reduces a numerator of rank `rank` over
/// `propagators` loop propagators, else its refusal: NinjaRS's rank
/// admission (`rusticol_core::loop_v2::bounds::rank_refusal`), TIRust's
/// capability query (structural bounds, the same for every carrier: asked
/// for binary64 with the TIR backend's settings;
/// [`crate::v2_tir::capability_refusal`]).
pub fn rank_refusal(reduction: Reduction, propagators: usize, rank: usize) -> Option<String> {
    #[cfg(feature = "tir")]
    if reduction == Reduction::TiRust {
        return crate::v2_tir::capability_refusal(propagators, rank, f64::MANTISSA_DIGITS);
    }
    rusticol_core::loop_v2::bounds::rank_refusal(reduction, propagators, rank)
}

/// NinjaRS's refusal of a plan whose families it cannot reduce: a TIR-only
/// plan (a family beyond its rank <= N + 2, m-lit lit-eft;
/// `rusticol_core::loop_v2::bounds::tir_only`), refused typed when its
/// backend is built for the plan, before any point is evaluated (never a
/// switch to TIRust: the caller names the backend).
pub fn ninjars_refusal(plan: &rusticol_core::loop_v2::plan::LoopProgramV4) -> RusticolResult<()> {
    match rusticol_core::loop_v2::bounds::refused_families(Reduction::NinjaRs, plan_histogram(plan))
    {
        None => Ok(()),
        Some(refusal) => Err(RusticolError::unsupported_runtime_capability(
            "ninjars",
            format!("this artifact is TIR-only, it admits the reduction tirust only: {refusal}"),
        )),
    }
}

/// A plan's (propagators, declared rank) histogram: the TIR backend's plans
/// ([`V2Backends::tirust`]).
pub fn plan_histogram(
    plan: &rusticol_core::loop_v2::plan::LoopProgramV4,
) -> std::collections::BTreeSet<(usize, usize)> {
    plan.families
        .iter()
        .map(|f| (f.propagators(), f.rank))
        .collect()
}

impl V2Backends {
    /// The TIR backend, its plans built for `families` (an artifact's
    /// (propagators, declared rank) histogram,
    /// `v2_tir::TirReductionBackend::with_histogram`) in a carrier of `bits`
    /// bits, at the target accuracy `target` (the relative accuracy of each
    /// point's result, `reduction_policy::DEFAULT_TARGET` by default; its
    /// calibrated method-choice acceptance, `v2_tir::TirOptions::
    /// for_target`): TIRust's typed refusal (its capability query, its plan
    /// construction) is the capability refusal; refused typed in a build
    /// without TIRust (the cargo feature `tir`).
    pub fn tirust(
        families: impl IntoIterator<Item = (usize, usize)>,
        bits: u32,
        target: f64,
    ) -> RusticolResult<Self> {
        #[cfg(feature = "tir")]
        {
            Self::tirust_with(
                families,
                bits,
                crate::v2_tir::TirOptions::for_target(target),
            )
        }
        #[cfg(not(feature = "tir"))]
        {
            let _ = (families.into_iter(), bits, target);
            Err(RusticolError::unsupported_runtime_capability(
                "tirust",
                TIRUST_NOT_BUILT,
            ))
        }
    }

    /// The TIR backend on the NinjaRS adapter's batch groups
    /// (`v2_tir::TirReductionBackend::reduce_family_groups`); NinjaRS takes
    /// [`V2Reducer::reduce`] directly.
    pub fn reduce_family_groups(
        &mut self,
        groups: &[FamilyGroup<'_>],
        poles: Poles,
        check: Option<ReductionCheck>,
    ) -> RusticolResult<super::v2_reduce::V2Output> {
        match self {
            #[cfg(feature = "tir")]
            Self::TiRust(backend) => backend.reduce_family_groups(groups, poles, check),
            _ => {
                let _ = (groups, poles, check);
                Err(RusticolError::invalid_argument(
                    "batch groups reduce through V2Reducer::reduce on NinjaRS",
                ))
            }
        }
    }

    /// The target accuracy of the following reductions (TIRust: its
    /// calibrated method-choice acceptance, `v2_tir::acceptance_of`, set on
    /// the built backend; NinjaRS: nothing it drives).
    pub fn set_target(&mut self, target: f64) {
        match self {
            Self::NinjaRs(_) => {}
            #[cfg(feature = "tir")]
            Self::TiRust(backend) => backend.set_acceptance(crate::v2_tir::acceptance_of(target)),
        }
        #[cfg(not(feature = "tir"))]
        let _ = target;
    }

    /// The unchecked mode of the following reductions (m-lit lit-unchecked,
    /// `reduction_policy::DEFAULT_UNCHECKED` off): TIRust's method selection
    /// without the acceptance pass, with the gate-0 last resort
    /// (`v2_tir::TirOptions::unchecked`); NinjaRS's zero stability
    /// thresholds ([`V2Reducer::set_unchecked`]). Set on the built backend
    /// (TIRust's plans stay); fixed for a call, never per point.
    pub fn set_unchecked(&mut self, unchecked: bool) {
        match self {
            Self::NinjaRs(backend) => backend.reducer.set_unchecked(unchecked),
            #[cfg(feature = "tir")]
            Self::TiRust(backend) => backend.set_unchecked(unchecked),
        }
    }

    /// The unchecked mode of the following reductions.
    pub fn unchecked(&self) -> bool {
        match self {
            Self::NinjaRs(backend) => backend.reducer.unchecked(),
            #[cfg(feature = "tir")]
            Self::TiRust(backend) => backend.options().unchecked,
        }
    }

    /// [`Self::set_target`] by the acceptance itself (an expert's
    /// `TirOptions::acceptance`).
    pub fn set_target_acceptance(&mut self, acceptance: f64) {
        match self {
            Self::NinjaRs(_) => {}
            #[cfg(feature = "tir")]
            Self::TiRust(backend) => backend.set_acceptance(acceptance),
        }
        #[cfg(not(feature = "tir"))]
        let _ = acceptance;
    }

    /// The TIR backend's (propagators, rank) plans built so far (empty for
    /// NinjaRS).
    pub fn plans(&self) -> Vec<(usize, usize)> {
        match self {
            Self::NinjaRs(_) => Vec::new(),
            #[cfg(feature = "tir")]
            Self::TiRust(backend) => backend.plans(),
        }
    }

    /// [`Self::tirust`] with explicit settings.
    #[cfg(feature = "tir")]
    pub fn tirust_with(
        families: impl IntoIterator<Item = (usize, usize)>,
        bits: u32,
        options: crate::v2_tir::TirOptions,
    ) -> RusticolResult<Self> {
        Ok(Self::TiRust(Box::new(
            crate::v2_tir::TirReductionBackend::with_histogram(families, bits, options)?,
        )))
    }

    pub fn reduction(&self) -> Reduction {
        match self {
            Self::NinjaRs(_) => Reduction::NinjaRs,
            #[cfg(feature = "tir")]
            Self::TiRust(_) => Reduction::TiRust,
        }
    }

    /// The masters' pole selection (NinjaRS's generic engine, the oracle of
    /// selected residues); TIRust's moments keep every residue, so it
    /// accepts `BOTH` only.
    pub fn set_masters(&mut self, masters: PoleSelection) -> RusticolResult<()> {
        match self {
            Self::NinjaRs(backend) => {
                backend.masters = masters;
                Ok(())
            }
            #[cfg(feature = "tir")]
            Self::TiRust(_) if masters == PoleSelection::BOTH => Ok(()),
            #[cfg(feature = "tir")]
            Self::TiRust(_) => Err(RusticolError::invalid_argument(
                "the TIRust backend keeps every master pole; select master poles with NinjaRS's \
                 generic engine",
            )),
        }
    }
}

impl<T: TypedBatch> V2Backend<T> for V2Backends {
    fn begin(&mut self) {
        match self {
            Self::NinjaRs(backend) => V2Backend::<T>::begin(backend),
            #[cfg(feature = "tir")]
            Self::TiRust(backend) => backend.begin(),
        }
    }

    fn take_statistics(&mut self) -> BackendStatistics {
        match self {
            Self::NinjaRs(backend) => V2Backend::<T>::take_statistics(backend),
            #[cfg(feature = "tir")]
            Self::TiRust(backend) => backend.take_statistics(),
        }
    }

    fn take_lane_measures(&mut self) -> Vec<MeasuredLane> {
        match self {
            Self::NinjaRs(backend) => V2Backend::<T>::take_lane_measures(backend),
            #[cfg(feature = "tir")]
            Self::TiRust(backend) => backend.take_lane_measures(),
        }
    }

    fn reduce(
        &mut self,
        groups: &[FamilyLanes<T>],
        poles: Poles,
        check: Option<ReductionCheck>,
    ) -> RusticolResult<Reduced<T>> {
        match self {
            Self::NinjaRs(backend) => backend.reduce(groups, poles, check),
            #[cfg(feature = "tir")]
            Self::TiRust(backend) => T::tir_reduce(backend, groups, poles, check),
        }
    }
}

/// A [`V2Reducer`] as the executor's reduction backend. `masters` is the
/// masters' pole selection (`BOTH` in production; the generic engine's
/// selected residues are the oracle and cost reference of the `uv` and `ir`
/// modes, DESIGN.md §12.4).
pub struct V2ReductionBackend {
    pub reducer: V2Reducer,
    pub masters: PoleSelection,
}

impl V2ReductionBackend {
    pub fn new(reducer: V2Reducer, masters: PoleSelection) -> Self {
        Self { reducer, masters }
    }
}

fn failure(error: impl std::fmt::Display) -> RusticolError {
    RusticolError::evaluation(format!("v2 reduction: {error}"))
}

fn frame(check: ReductionCheck) -> ReductionFrame {
    ReductionFrame {
        reverse: check.reverse,
        rotation: check.rotation,
        shift: check.shift,
        rescale: check.rescale,
    }
}

impl<T: TypedBatch> V2Backend<T> for V2ReductionBackend {
    fn begin(&mut self) {
        self.reducer.clear_masters();
    }

    /// The master integrals inside the `reduction` stage (binary64 fast
    /// engine: canonicalisation, cache and oneloopmaster; oneloopmaster
    /// alone), the master rows requested and evaluated, and the lanes the
    /// fast engine left to the exact binary64 engine (inputs at its absolute
    /// thresholds, `v2_reduce` module docs).
    fn take_statistics(&mut self) -> BackendStatistics {
        let Some(masters) = self.reducer.take_master_statistics() else {
            return BackendStatistics::default();
        };
        let exact = self.reducer.take_exact_lanes();
        BackendStatistics {
            seconds: vec![
                ("reduction: masters", masters.nanoseconds as f64 * 1e-9),
                (
                    "reduction: masters (oneloopmaster)",
                    masters.provider_nanoseconds as f64 * 1e-9,
                ),
            ],
            counts: vec![
                ("master requests", masters.total_requests()),
                ("master evaluations", masters.total_evaluated()),
                ("binary64 lanes on the exact engine", exact),
            ],
        }
    }

    fn reduce(
        &mut self,
        groups: &[FamilyLanes<T>],
        poles: Poles,
        check: Option<ReductionCheck>,
    ) -> RusticolResult<Reduced<T>> {
        let bits = self.reducer.precision_bits();
        let batches = groups
            .iter()
            .map(|g| {
                let rows = g.keys.len();
                let ids: Vec<u64> = (0..rows as u64).collect();
                let n = g.propagators;
                let typed = |values: &[Complex<T>], columns: usize| {
                    Batch::typed(values.to_vec(), ids.clone(), columns, bits).map_err(failure)
                };
                Ok([
                    typed(&g.shifts, 4 * n)?,
                    typed(&g.masses, n)?,
                    typed(&g.scales, 1)?,
                    typed(&g.s_matrix, n * n)?,
                    typed(&g.coefficients, g.coefficients.len() / rows.max(1))?,
                ])
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let family_groups: Vec<FamilyGroup<'_>> = groups
            .iter()
            .zip(&batches)
            .map(
                |(g, [shifts, masses, scales, s_matrix, coefficients])| FamilyGroup {
                    propagators: g.propagators,
                    shifts,
                    masses_squared: masses,
                    scales_squared: scales,
                    s_matrix,
                    coefficients,
                },
            )
            .collect();
        let output = self
            .reducer
            .reduce(
                &family_groups,
                ReduceOptions {
                    poles,
                    check: check.map(frame),
                    masters: self.masters,
                },
            )
            .map_err(failure)?;
        let read = |batches: Vec<super::reduction::ReductionBatch>| {
            batches
                .into_iter()
                .map(|batch| {
                    let values = T::values(batch.values.data())
                        .ok_or_else(|| failure("reduction carrier differs"))?;
                    let columns = batch.values.columns();
                    let mut successes = batch.values.ids().iter().zip(values.chunks_exact(columns));
                    let mut next = successes.next();
                    batch
                        .statuses
                        .iter()
                        .map(|(id, status)| -> RusticolResult<LaneOutcome<T>> {
                            let value = match next {
                                Some((&row, value)) if row == *id => {
                                    next = successes.next();
                                    Some(value.to_vec())
                                }
                                _ => None,
                            };
                            Ok(match value {
                                Some(value) if status == "success" => Ok(value),
                                _ => Err(status.clone()),
                            })
                        })
                        .collect::<RusticolResult<Vec<_>>>()
                })
                .collect::<RusticolResult<Vec<_>>>()
        };
        Ok((read(output.direct)?, output.check.map(read).transpose()?))
    }
}

/// The (propagators, rank) classes whose engines `reduction`'s first
/// reductions of `plan` build ([`warm`]): TIRust's are the plan's declared
/// classes (its plans); NinjaRS buckets a lane by its true rank (the highest
/// monomial degree with a nonzero coefficient, `true_rank`), and a family's
/// accumulator holds a polynomial of at most the largest output rank of the
/// trace steps closing into it, below the family's declared rank when no
/// such step reaches it: per family and accumulator that rank. A dressed-row
/// plan's lanes (amplitude mode) fall below it too (measured: g g > Z Z and
/// g g > h h build lower classes on their first reduction): every rank up to
/// it. A lane whose top coefficients vanish otherwise falls into a lower
/// class, built on its first use. Classes the backend refuses are left out
/// (an evaluation refuses them).
pub fn warm_classes(
    plan: &rusticol_core::loop_v2::plan::LoopProgramV4,
    reduction: Reduction,
) -> std::collections::BTreeSet<(usize, usize)> {
    let classes: std::collections::BTreeSet<(usize, usize)> = match reduction {
        Reduction::TiRust => plan_histogram(plan),
        Reduction::NinjaRs => plan
            .families
            .iter()
            .flat_map(|family| {
                let mut top: Vec<Option<usize>> = vec![None; family.orders.len()];
                for step in &family.steps {
                    if let Some(k) = step.trace {
                        top[k] = top[k].max(Some(step.rank_out));
                    }
                }
                let n = family.propagators();
                let lowest = |rank: usize| if plan.dressed { 0 } else { rank };
                top.into_iter()
                    .flatten()
                    .flat_map(move |rank| (lowest(rank)..=rank).map(move |r| (n, r)))
            })
            .collect(),
    };
    classes
        .into_iter()
        .filter(|&(n, rank)| rank_refusal(reduction, n, rank).is_none())
        .collect()
}

/// Build now what a backend's first reductions of the (propagators, rank)
/// classes `classes` ([`warm_classes`]) would build lazily, in the carrier
/// of `zero`, without reducing a point of the process (the warm entry,
/// `V2Executor.warm`): NinjaRS's engine of each class (binary64: its fast
/// amplitude and tensor numerator; the extended carriers: its generic
/// engine, whose cut expansions are built by the first evaluation), TIRust's
/// plans and workspaces. One synthetic lane per class (a fixed generic
/// massive configuration with exact rational inputs cast to the carrier,
/// unit-size coefficients) runs through the backend's own `reduce` in its
/// current mode, and its outcome is dropped; the master cache and the
/// counters it filled are reset after (`begin`, `take_statistics`,
/// `take_lane_measures`), so the next call reports and caches only its own
/// lanes. A later reduction's values do not depend on it: every lane's
/// outcome is a function of its own inputs (the engines' caches are keyed by
/// them). Not built: the binary64 fast engine's exact fallback (its lanes
/// stay below the absolute thresholds that route a lane to it).
pub fn warm<T>(backend: &mut V2Backends, classes: &[(usize, usize)], zero: &T) -> RusticolResult<()>
where
    T: TypedBatch + rusticol_core::loop_v2::dress::Carrier,
    V2Backends: V2Backend<T>,
{
    let groups = classes
        .iter()
        .map(|&(n, rank)| synthetic_lane(n, rank, zero, 0))
        .collect::<RusticolResult<Vec<_>>>()?;
    V2Backend::<T>::begin(backend);
    // One class per call: a class the backend refuses (its typed refusal)
    // is the caller's error, as in an evaluation.
    for group in &groups {
        V2Backend::<T>::reduce(backend, std::slice::from_ref(group), Poles::All, None)?;
    }
    V2Backend::<T>::begin(backend);
    let _ = V2Backend::<T>::take_statistics(backend);
    let _ = V2Backend::<T>::take_lane_measures(backend);
    Ok(())
}

/// One synthetic reduction lane of `n` propagators and rank `rank` in the
/// carrier of `zero` ([`warm`]): shifts `q_0 = 0` and `q_k` in general
/// position (`variant` moves them), distinct positive squared masses, the
/// invariants `(q_i - q_j)^2`, unit scale and `O(1)` coefficients, every
/// value an exact rational cast to the carrier (at its precision).
fn synthetic_lane<T>(
    n: usize,
    rank: usize,
    zero: &T,
    variant: i64,
) -> RusticolResult<FamilyLanes<T>>
where
    T: TypedBatch + rusticol_core::loop_v2::dress::Carrier,
{
    use ninjars::fast::tensor_layout::ncoeff;
    use rusticol_core::loop_execution::v2::LaneKey;
    use rusticol_core::loop_v2::dress::{cast, parse_rational};
    use symbolica::domains::rational::Rational;

    let exact = |numerator: i64, denominator: i64| -> RusticolResult<T> {
        Ok(cast(
            zero,
            &parse_rational(&format!("{numerator}/{denominator}"))?,
        ))
    };
    let complex = |re: T| Complex::new(re, zero.clone());
    let shift = |k: usize, mu: usize| -> (i64, i64) {
        if k == 0 {
            return (0, 1);
        }
        let k = k as i64 + variant;
        match mu {
            0 => (3 * k * k + 5 * k + 2, 7),
            1 => (2 * k + 1, 3 + k),
            2 => (5 - 3 * k, 11),
            _ => (k * k - 4, 13 + 2 * k),
        }
    };
    let mut shifts = Vec::with_capacity(4 * n);
    for k in 0..n {
        for mu in 0..4 {
            let (a, b) = shift(k, mu);
            shifts.push(complex(exact(a, b)?));
        }
    }
    let mut s_matrix = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            // (q_i - q_j)^2 exactly: a rational of the shifts.
            let mut square = Rational::from(0);
            for mu in 0..4 {
                let (a, b) = shift(i, mu);
                let (c, d) = shift(j, mu);
                let difference = parse_rational(&format!("{}/{}", a * d - c * b, b * d))?;
                let term = difference.clone() * &difference;
                square += &if mu == 0 {
                    term
                } else {
                    Rational::from(0) - term
                };
            }
            s_matrix.push(complex(cast(zero, &square)));
        }
    }
    let masses = (0..n)
        .map(|k| Ok(complex(exact(17 + 6 * (k as i64 + variant), 13)?)))
        .collect::<RusticolResult<Vec<_>>>()?;
    let coefficients = (0..ncoeff(rank))
        .map(|c| {
            let c = c as i64 + variant;
            Ok(complex(exact(1 + c % 5, 2 + c % 3)?))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    Ok(FamilyLanes {
        family: 0,
        propagators: n,
        rank,
        points: vec![0],
        keys: vec![LaneKey {
            family: 0,
            order: 0,
            column: 0,
        }],
        shifts,
        masses,
        scales: vec![complex(exact(1, 1)?)],
        s_matrix,
        coefficients,
    })
}

#[cfg(test)]
mod warm_tests {
    use super::*;
    use symbolica::domains::float::DoubleFloat;

    fn ninjars(arithmetic: &str, engine: &str) -> V2Backends {
        V2Backends::NinjaRs(V2ReductionBackend::new(
            V2Reducer::new(arithmetic, None, engine, true).expect("reducer"),
            PoleSelection::BOTH,
        ))
    }

    /// The outcomes of `lanes` reduced by `backend`, as text (exact: the
    /// carriers' `Debug` forms).
    fn reduced<T>(backend: &mut V2Backends, lanes: &[FamilyLanes<T>]) -> String
    where
        T: TypedBatch + rusticol_core::loop_v2::dress::Carrier,
        V2Backends: V2Backend<T>,
    {
        V2Backend::<T>::begin(backend);
        let out = V2Backend::<T>::reduce(backend, lanes, Poles::All, None).expect("reduce");
        format!("{:?}", out.0)
    }

    fn check<T>(
        make: impl Fn() -> V2Backends,
        zero: &T,
        classes: &[(usize, usize)],
        lower: bool,
        sensitive: bool,
    ) where
        T: TypedBatch + rusticol_core::loop_v2::dress::Carrier,
        V2Backends: V2Backend<T>,
    {
        // Other configurations than the warm lanes', declared and lower ranks.
        let lanes: Vec<FamilyLanes<T>> = classes
            .iter()
            .flat_map(|&(n, rank)| {
                let below = (lower && rank > 0).then(|| (n, rank - 1));
                std::iter::once((n, rank)).chain(below)
            })
            .map(|(n, rank)| synthetic_lane(n, rank, zero, 3).expect("lane"))
            .collect();
        let mut plain = make();
        let cold = reduced(&mut plain, &lanes);
        // A reduction fills the master cache and the counters (binary64 fast
        // engine): what the warm must leave empty.
        let (masters, statistics, _) = residue::<T>(&mut plain);
        if sensitive {
            assert!(masters > 0 && statistics > 0, "{masters} {statistics}");
        }
        let mut warmed = make();
        warm(&mut warmed, classes, zero).expect("warm");
        assert_eq!(residue::<T>(&mut warmed), (0, 0, 0), "the warm resets");
        assert_eq!(reduced(&mut warmed, &lanes), cold);
        assert!(cold.contains("Ok("), "the test lanes reduce: {cold}");
    }

    /// The cached master rows (NinjaRS), the counters and seconds taken and
    /// the lane measures taken, of a backend.
    fn residue<T>(backend: &mut V2Backends) -> (usize, usize, usize)
    where
        T: TypedBatch,
        V2Backends: V2Backend<T>,
    {
        let masters = match &*backend {
            V2Backends::NinjaRs(backend) => backend.reducer.cached_masters(),
            #[cfg(feature = "tir")]
            V2Backends::TiRust(_) => 0,
        };
        // Counters and seconds that recorded something (a backend may list
        // its counters at zero).
        let statistics = V2Backend::<T>::take_statistics(backend);
        let recorded = statistics.counts.iter().filter(|(_, n)| *n > 0).count()
            + statistics.seconds.iter().filter(|(_, s)| *s > 0.0).count();
        let measures = V2Backend::<T>::take_lane_measures(backend).len();
        (masters, recorded, measures)
    }

    #[test]
    fn a_warmed_backend_reduces_bit_identically() {
        let _unlock = pyamplicol::unlock();
        let classes = [(2, 2), (3, 3), (4, 4)];
        // NinjaRS buckets a lane by its true rank: lanes below the
        // declared ranks too.
        check(|| ninjars("double", "fast"), &0.0, &classes, true, true);
        check(
            || ninjars("double-double", "generic"),
            &DoubleFloat::from(0.0),
            &classes,
            true,
            false,
        );
        #[cfg(feature = "tir")]
        {
            let histogram: std::collections::BTreeSet<_> = classes.into_iter().collect();
            let tir = || {
                V2Backends::tirust(
                    histogram.clone(),
                    53,
                    crate::reduction_policy::DEFAULT_TARGET,
                )
                .expect("tirust")
            };
            check(tir, &0.0, &classes, false, false);
        }
    }
}
