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
    BackendStatistics, ExactLanes, FamilyLanes, LaneOutcome, MeasuredLane, Reduced, RescueCarrier,
    RescueOutput, TrackedPrecision, V2Backend, tracked_uncertainty,
};
use rusticol_core::loop_v2::dress::{Carrier, cast, exact_value};
use rusticol_core::loop_v2::retest::ReductionCheck;
use rusticol_core::{RusticolError, RusticolResult};
use symbolica::domains::float::{DoubleFloat, Float};

/// Typed access to the values of a [`Batch`], and the carrier's TIR
/// reduction (every carrier, m-lit lit-tir-mp).
pub trait TypedBatch: BatchCarrier + Carrier {
    fn values(data: &Data) -> Option<&[Complex<Self>]>;

    /// The carrier's bits (its precision).
    fn bits(zero: &Self) -> u32;

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

    fn bits(_: &Self) -> u32 {
        f64::MANTISSA_DIGITS
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

    fn bits(_: &Self) -> u32 {
        106
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

    fn bits(zero: &Self) -> u32 {
        zero.prec()
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

    /// NinjaRS's ([`V2Reducer::tracked_precision`]); none on TIRust, whose
    /// extended-precision values its own acceptance tests (method-choice
    /// deviations, Gram and Cayley conditioning) and whose direct kernels
    /// round at the working precision (TIRust #43), so that their precision
    /// as numerica tracks it says nothing about their loss.
    fn tracked_precision(&self) -> Option<TrackedPrecision> {
        match self {
            Self::NinjaRs(backend) => V2Backend::<T>::tracked_precision(backend),
            #[cfg(feature = "tir")]
            Self::TiRust(_) => None,
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

    /// The carriers above the evaluation's ([`rescue_carriers`]), in the
    /// checked and the unchecked mode alike (USER 2026-11-18, "Both, with a
    /// switch"): in the unchecked mode the lanes it still refuses have no
    /// value at all (NinjaRS's conditioning screen, a Euclidean-plane cut,
    /// a non-finite result). The rescue backends are always checked
    /// ([`RescueBackends`]).
    fn rescue_carriers(&self, zero: &T) -> Vec<RescueCarrier> {
        rescue_carriers(T::bits(zero))
    }

    /// The lanes reduced by this backend's own reduction in `carrier`
    /// ([`RescueBackends`]), rounded back to the evaluation's carrier.
    fn rescue(
        &mut self,
        carrier: RescueCarrier,
        groups: &[ExactLanes],
        poles: Poles,
        check: Option<ReductionCheck>,
        zero: &T,
    ) -> RusticolResult<RescueOutput<T>> {
        let index = self.rescue_index(carrier)?;
        let backend = &mut self.rescue_store().0[index].1;
        match carrier {
            RescueCarrier::DoubleDouble => {
                reduce_exact(backend, groups, poles, check, &DoubleFloat::from(0.0), zero)
            }
            RescueCarrier::Arbitrary(_) => reduce_exact(
                backend,
                groups,
                poles,
                check,
                &Float::with_val(carrier.bits(), 0),
                zero,
            ),
        }
    }
}

/// The lane rescue's backends of one evaluation backend
/// ([`V2Backend::rescue`]): per rescue carrier one backend of the same
/// reduction (NinjaRS's generic engine with the same master selection;
/// TIRust with the same plans and settings, checked), built on first use
/// and kept with the evaluation backend, so that its engines, plans and
/// caches serve every later call (never one build per call or point).
#[derive(Default)]
pub struct RescueBackends(Vec<(RescueCarrier, V2Backends)>);

/// The lane rescue's carriers above a carrier of `bits` bits, lowest
/// first: double-double, then arbitrary precision at the default
/// arbitrary-precision digits
/// ([`crate::reduction_policy::DEFAULT_ARBITRARY_DIGITS`], at their working
/// digits).
pub fn rescue_carriers(bits: u32) -> Vec<RescueCarrier> {
    let digits = crate::v2_runtime::V2Precision::Arbitrary(
        crate::reduction_policy::DEFAULT_ARBITRARY_DIGITS,
    )
    .working_digits();
    [
        RescueCarrier::DoubleDouble,
        RescueCarrier::Arbitrary(digits),
    ]
    .into_iter()
    .filter(|carrier| carrier.bits() > bits)
    .collect()
}

/// A complex value of one carrier rounded to the carrier of `zero` (once,
/// through its exact rational), `None` when it is not finite.
fn rounded<U: Carrier, T: Carrier>(value: &Complex<U>, zero: &T) -> Option<Complex<T>> {
    Some(Complex::new(
        cast(zero, &exact_value(&value.re)?),
        cast(zero, &exact_value(&value.im)?),
    ))
}

/// Reduce exact lanes on `backend` in the carrier of `high`, the outcomes
/// rounded to the carrier of `zero` (a non-finite value refuses its lane).
fn reduce_exact<U: TypedBatch, T: TypedBatch>(
    backend: &mut V2Backends,
    groups: &[ExactLanes],
    poles: Poles,
    check: Option<ReductionCheck>,
    high: &U,
    zero: &T,
) -> RusticolResult<RescueOutput<T>>
where
    U: symbolica::domains::float::Real,
{
    let complex = |[re, im]: &[symbolica::domains::rational::Rational; 2]| {
        Complex::new(cast(high, re), cast(high, im))
    };
    let real =
        |x: &symbolica::domains::rational::Rational| Complex::new(cast(high, x), high.zero());
    let lanes: Vec<FamilyLanes<U>> = groups
        .iter()
        .map(|g| FamilyLanes {
            family: g.family,
            propagators: g.propagators,
            rank: g.rank,
            points: g.points.clone(),
            keys: g.keys.clone(),
            shifts: g.shifts.iter().map(real).collect(),
            masses: g.masses.iter().map(complex).collect(),
            scales: g.scales.iter().map(real).collect(),
            s_matrix: g.s_matrix.iter().map(real).collect(),
            coefficients: g.coefficients.iter().map(complex).collect(),
        })
        .collect();
    V2Backend::<U>::begin(backend);
    let (direct, retest) = V2Backend::<U>::reduce(backend, &lanes, poles, check)?;
    let _ = V2Backend::<U>::take_statistics(backend);
    let _ = V2Backend::<U>::take_lane_measures(backend);
    // The tracked uncertainty of every served direct value, where the
    // carrier's reduction tracks its precision (NinjaRS's arbitrary
    // precision), before it is rounded (the executor's tracked-precision
    // test of rescued lanes), in units of the evaluation carrier's unit
    // roundoff 2^-p (scaled in the rescue carrier: no binary64 underflow at
    // any evaluation precision, review m2).
    let unit = high.from_i64(2).pow(u64::from(T::bits(zero)));
    let tracked = V2Backend::<U>::tracked_precision(backend).map(|_| {
        direct
            .iter()
            .map(|group| {
                group
                    .iter()
                    .map(|outcome| match outcome {
                        Ok(values) => values
                            .iter()
                            .map(|z| (tracked_uncertainty(z) * &unit).to_f64())
                            .collect(),
                        Err(_) => Vec::new(),
                    })
                    .collect()
            })
            .collect()
    });
    let read = |outcomes: Vec<Vec<LaneOutcome<U>>>| -> Vec<Vec<LaneOutcome<T>>> {
        outcomes
            .into_iter()
            .map(|group| {
                group
                    .into_iter()
                    .map(|outcome| {
                        let values = outcome?;
                        values
                            .iter()
                            .map(|z| rounded(z, zero))
                            .collect::<Option<Vec<_>>>()
                            .ok_or_else(|| "NonFinite".to_string())
                    })
                    .collect()
            })
            .collect()
    };
    Ok(RescueOutput {
        outcomes: (read(direct), retest.map(read)),
        tracked,
    })
}

impl V2Backends {
    /// The index of this backend's rescue backend in `carrier`
    /// ([`RescueBackends`]), built on first use and kept, with this
    /// backend's CURRENT settings (review F1, 2026-11-19): the acceptance
    /// (TIRust: the call's target, `set_target` / `set_target_acceptance`)
    /// and the masters (NinjaRS, `set_masters`) are set on the kept backend
    /// at every rescue; a kept TIRust backend whose other options no longer
    /// follow this backend's ([`Self::rescue_options`]) is built again.
    pub fn rescue_index(&mut self, carrier: RescueCarrier) -> RusticolResult<usize> {
        let found = self
            .rescue_store()
            .0
            .iter()
            .position(|(c, _)| *c == carrier);
        let index = match found {
            Some(index) if self.rescue_current(index) => index,
            Some(index) => {
                let built = self.rescue_backend(carrier)?;
                self.rescue_store().0[index].1 = built;
                index
            }
            None => {
                let built = self.rescue_backend(carrier)?;
                let store = self.rescue_store();
                store.0.push((carrier, built));
                store.0.len() - 1
            }
        };
        self.sync_rescue(index);
        Ok(index)
    }

    /// The options of TIRust's rescue backends for an evaluation backend
    /// with `options`: checked, and the method choice in the extended
    /// carriers exactly when the evaluation's has it (an expert's
    /// `method_choice=False` keeps it off; review m7).
    #[cfg(feature = "tir")]
    fn rescue_options(options: &crate::v2_tir::TirOptions) -> crate::v2_tir::TirOptions {
        crate::v2_tir::TirOptions {
            unchecked: false,
            extended_method_choice: options.method_choice,
            ..*options
        }
    }

    /// Whether kept rescue backend `index` follows this backend's options,
    /// the acceptance aside (set in place by [`Self::sync_rescue`]).
    fn rescue_current(&mut self, index: usize) -> bool {
        match self {
            Self::NinjaRs(_) => true,
            #[cfg(feature = "tir")]
            Self::TiRust(main) => {
                let want = Self::rescue_options(main.options());
                match &main.rescue.0[index].1 {
                    Self::TiRust(kept) => {
                        crate::v2_tir::TirOptions {
                            acceptance: want.acceptance,
                            ..*kept.options()
                        } == want
                    }
                    Self::NinjaRs(_) => false,
                }
            }
        }
    }

    /// This backend's acceptance (TIRust) or masters (NinjaRS) on its kept
    /// rescue backend `index`.
    fn sync_rescue(&mut self, index: usize) {
        match self {
            Self::NinjaRs(main) => {
                let masters = main.masters;
                if let Self::NinjaRs(kept) = &mut main.rescue.0[index].1 {
                    kept.masters = masters;
                }
            }
            #[cfg(feature = "tir")]
            Self::TiRust(main) => {
                let acceptance = main.options().acceptance;
                if let Self::TiRust(kept) = &mut main.rescue.0[index].1 {
                    kept.set_acceptance(acceptance);
                }
            }
        }
    }

    /// The settings of kept rescue backend `index` (tests: the acceptance
    /// of a TIRust one, the masters of a NinjaRS one).
    #[cfg(test)]
    fn rescue_settings(&mut self, index: usize) -> (Option<f64>, Option<PoleSelection>, bool) {
        match &self.rescue_store().0[index].1 {
            Self::NinjaRs(kept) => (None, Some(kept.masters), false),
            #[cfg(feature = "tir")]
            Self::TiRust(kept) => (
                Some(kept.options().acceptance),
                None,
                kept.options().extended_method_choice,
            ),
        }
    }

    /// The lane rescue's backends ([`RescueBackends`]).
    fn rescue_store(&mut self) -> &mut RescueBackends {
        match self {
            Self::NinjaRs(backend) => &mut backend.rescue,
            #[cfg(feature = "tir")]
            Self::TiRust(backend) => &mut backend.rescue,
        }
    }

    /// A backend of this reduction in `carrier` (checked).
    fn rescue_backend(&self, carrier: RescueCarrier) -> RusticolResult<V2Backends> {
        match self {
            Self::NinjaRs(backend) => {
                let reducer = match carrier {
                    RescueCarrier::DoubleDouble => {
                        V2Reducer::new("double-double", None, "generic", true)
                    }
                    RescueCarrier::Arbitrary(digits) => {
                        V2Reducer::new("arbitrary", Some(digits), "generic", true)
                    }
                }
                .map_err(failure)?;
                Ok(Self::NinjaRs(V2ReductionBackend::new(
                    reducer,
                    backend.masters,
                )))
            }
            #[cfg(feature = "tir")]
            Self::TiRust(backend) => {
                // Checked, with the method choice in the rescue carriers too
                // (when the evaluation has it): a row the gates refuse there
                // gets the method choice and its acceptance check instead of
                // failing closed (and going on to the arbitrary-precision
                // step).
                let options = Self::rescue_options(backend.options());
                Self::tirust_with(backend.plans(), carrier.bits(), options)
            }
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
    /// The lane rescue's backends ([`RescueBackends`]).
    pub rescue: RescueBackends,
}

impl V2ReductionBackend {
    pub fn new(reducer: V2Reducer, masters: PoleSelection) -> Self {
        Self {
            reducer,
            masters,
            rescue: RescueBackends::default(),
        }
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

    fn tracked_precision(&self) -> Option<TrackedPrecision> {
        self.reducer.tracked_precision()
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

#[cfg(test)]
mod rescue_settings_tests {
    use super::*;

    /// Review F1: a kept rescue backend follows the evaluation backend's
    /// later target (TIRust's acceptance) and masters (NinjaRS), and a
    /// TIRust one is built again when another option changes (the method
    /// choice, review m7).
    #[cfg(feature = "tir")]
    #[test]
    fn kept_rescue_backends_follow_the_evaluation_backend() {
        use crate::v2_tir::{TirOptions, acceptance_of};
        let mut tir = V2Backends::tirust_with([(4, 4)], 53, TirOptions::for_target(1e-5))
            .expect("TIRust backend");
        let carrier = RescueCarrier::DoubleDouble;
        let index = tir.rescue_index(carrier).unwrap();
        assert_eq!(tir.rescue_settings(index).0, Some(acceptance_of(1e-5)));
        assert!(
            tir.rescue_settings(index).2,
            "the method choice on by default"
        );
        tir.set_target(1e-12);
        let again = tir.rescue_index(carrier).unwrap();
        assert_eq!(again, index);
        assert_eq!(tir.rescue_settings(index).0, Some(acceptance_of(1e-12)));
        tir.set_target_acceptance(0.25);
        tir.rescue_index(carrier).unwrap();
        assert_eq!(tir.rescue_settings(index).0, Some(0.25));
        // An expert's method_choice=false on the SAME evaluation backend (its
        // kept rescue backends moved over, review m-c): the kept one, built
        // with the method choice, is rebuilt without it, at the same index.
        if let V2Backends::TiRust(main) = &mut tir {
            let mut options = *main.options();
            options.method_choice = false;
            let mut changed =
                crate::v2_tir::TirReductionBackend::with_histogram([(4, 4)], 53, options)
                    .expect("TIRust backend");
            changed.rescue = std::mem::take(&mut main.rescue);
            **main = changed;
        }
        assert!(tir.rescue_settings(index).2, "kept as built");
        let rebuilt = tir.rescue_index(carrier).unwrap();
        assert_eq!(rebuilt, index);
        assert!(
            !tir.rescue_settings(index).2,
            "rebuilt without the method choice"
        );
    }

    /// Review m-c (m2): `reduce_exact` returns a rescued lane's tracked
    /// uncertainty in units of the EVALUATION carrier's unit roundoff: a
    /// tadpole reduced by NinjaRS at 30 digits (about 2^-100 relative) for a
    /// binary64 evaluation reads about 2^(53-100) |A0|, not 2^-100 |A0|.
    #[test]
    fn rescued_tracked_uncertainties_are_in_units_of_the_evaluation_roundoff() {
        use rusticol_core::loop_execution::v2::LaneKey;
        use symbolica::domains::rational::Rational;
        let carrier = RescueCarrier::Arbitrary(30);
        let reducer = V2Reducer::new("arbitrary", Some(30), "generic", true).expect("NinjaRS");
        let mut ninja = V2Backends::NinjaRs(V2ReductionBackend::new(reducer, PoleSelection::BOTH));
        let q = |n: i64| Rational::from(n);
        let tadpole = ExactLanes {
            family: 0,
            propagators: 1,
            rank: 0,
            points: vec![0],
            keys: vec![LaneKey {
                family: 0,
                order: 0,
                column: 0,
            }],
            shifts: vec![q(0), q(0), q(0), q(0)],
            masses: vec![[q(1), q(0)]],
            scales: vec![q(1)],
            s_matrix: vec![q(0)],
            coefficients: vec![[q(1), q(0)]],
        };
        let high = Float::with_val(carrier.bits(), 0);
        let out = reduce_exact(&mut ninja, &[tadpole], Poles::All, None, &high, &0.0_f64)
            .expect("the tadpole reduces");
        let value = match &out.outcomes.0[0][0] {
            Ok(value) => value[0],
            Err(status) => panic!("refused: {status}"),
        };
        let size = value.re.abs() + value.im.abs();
        let tracked = out.tracked.expect("NinjaRS's arbitrary precision tracks");
        let u = tracked[0][0][0];
        assert!(size > 0.0 && u > 0.0, "{size} {u}");
        // In units of 2^-53: far above the unscaled 2^-100 |A0|, below |A0|.
        let bits = carrier.bits() as i32;
        assert!(u > size * 2f64.powi(53 - bits - 16), "{u} vs {size}");
        assert!(u < size * 2f64.powi(53 - bits + 16), "{u} vs {size}");
    }

    #[test]
    fn a_kept_ninjars_rescue_backend_follows_the_masters() {
        let reducer = V2Reducer::new("double", None, "fast", true).expect("NinjaRS");
        let mut ninja = V2Backends::NinjaRs(V2ReductionBackend::new(reducer, PoleSelection::BOTH));
        let carrier = RescueCarrier::DoubleDouble;
        let index = ninja.rescue_index(carrier).unwrap();
        assert_eq!(ninja.rescue_settings(index).1, Some(PoleSelection::BOTH));
        ninja.set_masters(PoleSelection::UV).unwrap();
        assert_eq!(ninja.rescue_index(carrier).unwrap(), index);
        assert_eq!(ninja.rescue_settings(index).1, Some(PoleSelection::UV));
    }
}
