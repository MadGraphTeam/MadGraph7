// SPDX-License-Identifier: 0BSD

//! Python ownership of the v2 family reducer for the vertical slice.
use super::*;
use batch::{NativeBatch, ReductionBatch};
use rusticol_loop_reduction::runtime::{
    FamilyGroup, MasterStatistics, ReduceOptions, ReductionFrame,
};
use std::collections::BTreeMap;

type Group<'py> = (
    usize,
    PyRef<'py, NativeBatch>,
    PyRef<'py, NativeBatch>,
    PyRef<'py, NativeBatch>,
    PyRef<'py, NativeBatch>,
    PyRef<'py, NativeBatch>,
);

/// Family reductions with true-rank buckets, cached masters and the
/// reduction-direction check. Groups are tuples
/// `(propagators, shifts, masses_squared, scales_squared, s_matrix, coefficients)`
/// of batches with equal row identities; see `runtime::FamilyGroup`.
#[pyclass]
struct V2Reducer(runtime::V2Reducer);

#[pymethods]
impl V2Reducer {
    #[new]
    #[pyo3(signature = (arithmetic="double", precision=None, engine="fast", cache_masters=true))]
    fn new(
        arithmetic: &str,
        precision: Option<u32>,
        engine: &str,
        cache_masters: bool,
    ) -> PyResult<Self> {
        runtime::V2Reducer::new(arithmetic, precision, engine, cache_masters)
            .map(Self)
            .map_err(error)
    }
    #[getter]
    fn precision_bits(&self) -> u32 {
        self.0.precision_bits()
    }
    #[getter]
    fn cached_masters(&self) -> usize {
        self.0.cached_masters()
    }
    fn clear_masters(&mut self) {
        self.0.clear_masters();
    }
    /// Per master family `{requests, hits, evaluated}` since the last call,
    /// or None for the generic engine.
    fn take_master_statistics(
        &mut self,
    ) -> Option<BTreeMap<&'static str, BTreeMap<&'static str, u64>>> {
        self.0.take_master_statistics().map(|s: MasterStatistics| {
            MasterStatistics::FAMILY_NAMES
                .iter()
                .enumerate()
                .map(|(i, &name)| {
                    (
                        name,
                        BTreeMap::from([
                            ("requests", s.requests[i]),
                            ("hits", s.hits[i]),
                            ("evaluated", s.evaluated[i]),
                        ]),
                    )
                })
                .collect()
        })
    }
    /// Returns `(direct, check)`: one `ReductionBatch` per group, and the
    /// same for the relabelled frame when `check` is set (else None).
    /// Without `poles` the batches hold only the finite column. Cached
    /// masters are dropped first unless `keep_masters`.
    #[pyo3(signature = (groups, *, poles=true, check=false, reverse=true, rotation=0, shift=false, keep_masters=false))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn reduce(
        &mut self,
        py: Python<'_>,
        groups: Vec<Group<'_>>,
        poles: bool,
        check: bool,
        reverse: bool,
        rotation: usize,
        shift: bool,
        keep_masters: bool,
    ) -> PyResult<(Vec<ReductionBatch>, Option<Vec<ReductionBatch>>)> {
        let batches: Vec<_> = groups
            .iter()
            .map(|(n, shifts, masses, scales, matrix, coefficients)| {
                (
                    *n,
                    &shifts.0,
                    &masses.0,
                    &scales.0,
                    &matrix.0,
                    &coefficients.0,
                )
            })
            .collect();
        let options = ReduceOptions {
            poles,
            check: check.then_some(ReductionFrame {
                reverse,
                rotation,
                shift,
            }),
        };
        let reducer = &mut self.0;
        let output = py
            .detach(|| {
                if !keep_masters {
                    reducer.clear_masters();
                }
                let groups: Vec<_> = batches
                    .iter()
                    .map(
                        |&(propagators, shifts, masses, scales, matrix, coefficients)| {
                            FamilyGroup {
                                propagators,
                                shifts,
                                masses_squared: masses,
                                scales_squared: scales,
                                s_matrix: matrix,
                                coefficients,
                            }
                        },
                    )
                    .collect();
                reducer.reduce(&groups, options)
            })
            .map_err(error)?;
        let wrap = |batches: Vec<_>| batches.into_iter().map(ReductionBatch).collect();
        Ok((wrap(output.direct), output.check.map(wrap)))
    }
}

/// Exponents `(q0, q1, q2, q3)` of every coefficient of a dense rank-`rank`
/// tensor numerator, in NinjaRS's storage order.
#[pyfunction]
fn v2_tensor_monomials(rank: usize) -> Vec<[u32; 4]> {
    rusticol_loop_reduction::runtime::tensor_monomials(rank)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<V2Reducer>()?;
    module.add_function(wrap_pyfunction!(v2_tensor_monomials, module)?)
}
