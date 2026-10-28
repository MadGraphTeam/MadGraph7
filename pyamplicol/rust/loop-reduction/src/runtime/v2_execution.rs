// SPDX-License-Identifier: 0BSD

//! The reduction backend of the native v2 executor
//! (`rusticol_core::loop_execution::v2`): the executor streams each bucket
//! chunk's family lanes into the [`V2Reducer`] (true-rank buckets, cached
//! masters, reduction-direction check) as typed batches, without any decimal
//! round trip, and reads back each lane's status and Laurent integrals.

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

/// Typed access to the values of a [`Batch`].
pub trait TypedBatch: BatchCarrier {
    fn values(data: &Data) -> Option<&[Complex<Self>]>;
}

impl TypedBatch for f64 {
    fn values(data: &Data) -> Option<&[Complex<Self>]> {
        match data {
            Data::Double(values) => Some(values),
            _ => None,
        }
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
