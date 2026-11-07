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
    BackendStatistics, FamilyLanes, LaneOutcome, Reduced, V2Backend,
};
use rusticol_core::loop_v2::retest::ReductionCheck;
use rusticol_core::{RusticolError, RusticolResult};
use symbolica::domains::float::{DoubleFloat, Float};

/// Typed access to the values of a [`Batch`], and the carrier's TIR
/// reduction (binary64; m-lit `lit-tir-mp` adds the extended carriers).
pub trait TypedBatch: BatchCarrier {
    fn values(data: &Data) -> Option<&[Complex<Self>]>;

    /// The TIR backend's reduction in this carrier.
    #[cfg(feature = "tir")]
    fn tir_reduce(
        _backend: &mut crate::v2_tir::TirReductionBackend,
        _groups: &[FamilyLanes<Self>],
        _poles: Poles,
        _check: Option<ReductionCheck>,
    ) -> RusticolResult<Reduced<Self>> {
        Err(RusticolError::unsupported_precision(
            "the TIRust reduction backend serves binary64 in this build",
        ))
    }
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
}

impl TypedBatch for Float {
    fn values(data: &Data) -> Option<&[Complex<Self>]> {
        match data {
            Data::Arbitrary(values) => Some(values),
            _ => None,
        }
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

/// A reduction backend by name: `"ninjars"` or `"tirust"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Reduction {
    NinjaRs,
    TiRust,
}

impl Reduction {
    pub fn parse(name: &str) -> RusticolResult<Self> {
        match name {
            "ninjars" => Ok(Self::NinjaRs),
            "tirust" => Ok(Self::TiRust),
            other => Err(RusticolError::invalid_argument(format!(
                "reduction must be ninjars or tirust, not {other}"
            ))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::NinjaRs => "ninjars",
            Self::TiRust => "tirust",
        }
    }
}

/// The reduction backends this build serves, by name (`"ninjars"`,
/// `"tirust"`), with the carriers each serves: what the PyO3 capability
/// query (`reduction_backends`) reports.
pub fn backend_carriers() -> Vec<(&'static str, Vec<&'static str>)> {
    let mut out = vec![("ninjars", vec!["double", "double-double", "arbitrary"])];
    if crate::tir::available() {
        out.push(("tirust", vec!["double"]));
    }
    out
}

impl V2Backends {
    /// The TIR backend, its plans built for `families` (an artifact's
    /// (propagators, declared rank) histogram,
    /// `v2_tir::TirReductionBackend::with_histogram`): TIRust's typed
    /// refusal is the capability refusal; refused typed in a build without
    /// TIRust (the cargo feature `tir`).
    pub fn tirust(families: impl IntoIterator<Item = (usize, usize)>) -> RusticolResult<Self> {
        #[cfg(feature = "tir")]
        {
            Self::tirust_with(families, crate::v2_tir::TirOptions::default())
        }
        #[cfg(not(feature = "tir"))]
        {
            let _ = families.into_iter();
            Err(RusticolError::unsupported_runtime_capability(
                "tirust",
                "this native library is built without TIRust (the cargo feature tir)",
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
        options: crate::v2_tir::TirOptions,
    ) -> RusticolResult<Self> {
        Ok(Self::TiRust(Box::new(
            crate::v2_tir::TirReductionBackend::with_histogram(families, options)?,
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
