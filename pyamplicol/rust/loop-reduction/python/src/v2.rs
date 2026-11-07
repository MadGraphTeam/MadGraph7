// SPDX-License-Identifier: 0BSD

//! Python ownership of the v2 family reducer for the vertical slice.
use super::*;
use batch::{NativeBatch, ReductionBatch};
use rusticol_loop_reduction::runtime::{
    FamilyGroup, MasterStatistics, PoleSelection, Poles, ReduceOptions, ReductionFrame,
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

/// The Laurent columns of a `poles` option: `"all"` (combined poles),
/// `"none"` (the finite part) or `"split"` (combined poles and each lane's UV
/// residue, DESIGN.md §12.4). One spelling: no bool.
pub(super) fn poles_option(value: &Bound<'_, PyAny>) -> PyResult<Poles> {
    let Ok(value) = value.extract::<String>() else {
        return Err(error("poles must be all, none or split".to_string()));
    };
    match value.as_str() {
        "all" => Ok(Poles::All),
        "none" => Ok(Poles::None),
        "split" => Ok(Poles::Split),
        other => Err(error(format!(
            "poles must be all, none or split, not {other}"
        ))),
    }
}

/// oneloopmaster's master pole selection: `both`, `uv`, `ir` or `none`.
pub(super) fn master_selection(value: &str) -> PyResult<PoleSelection> {
    match value {
        "both" => Ok(PoleSelection::BOTH),
        "uv" => Ok(PoleSelection::UV),
        "ir" => Ok(PoleSelection::IR),
        "none" => Ok(PoleSelection::NONE),
        other => Err(error(format!(
            "master poles must be both, uv, ir or none, not {other}"
        ))),
    }
}

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
    /// Lanes the fast engine left to the generic binary64 engine since the
    /// last call (inputs at its absolute thresholds; 0 for the generic
    /// engines).
    fn take_exact_lanes(&mut self) -> u64 {
        self.0.take_exact_lanes()
    }
    /// Returns `(direct, check)`: one `ReductionBatch` per group, and the
    /// same for the relabelled frame when `check` is set (else None). The
    /// default frame is `ReductionFrame::REVERSED`.
    /// `poles` (`poles_option`): without poles the batches hold only the
    /// finite column, with `"split"` a fourth column, the lane's UV residue.
    /// `master_poles` selects the masters' residues (`master_selection`;
    /// generic engine only). Cached masters are dropped first unless
    /// `keep_masters`.
    #[pyo3(signature = (groups, *, poles=None, check=false, reverse=true, rotation=1, shift=false, rescale=false, keep_masters=false, master_poles="both"))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn reduce(
        &mut self,
        py: Python<'_>,
        groups: Vec<Group<'_>>,
        poles: Option<Bound<'_, PyAny>>,
        check: bool,
        reverse: bool,
        rotation: usize,
        shift: bool,
        rescale: bool,
        keep_masters: bool,
        master_poles: &str,
    ) -> PyResult<(Vec<ReductionBatch>, Option<Vec<ReductionBatch>>)> {
        let poles = poles.map_or(Ok(Poles::All), |value| poles_option(&value))?;
        let masters = master_selection(master_poles)?;
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
                rescale,
            }),
            masters,
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

/// The `alternative_first` keyword of TIRust's validation settings
/// ([`V2TirReducer`], `V2Executor.prepare_reduction`): a limit, `None` (the
/// lever off: the expansions first) or `"default"` (the backend's own,
/// `v2_tir::TirOptions::default`, TIRust's `EXPANSION_TOLERANCE`).
#[derive(Clone, Debug, PartialEq, FromPyObject)]
pub(super) enum Limit {
    Value(f64),
    Named(String),
}

impl Limit {
    /// The keyword's default: the backend's own limit.
    pub(super) fn default_limit() -> Option<Self> {
        Some(Self::Named("default".to_owned()))
    }
}

/// Whether TIRust's validation keywords differ from their defaults
/// (`method_choice` and `acceptance` omitted, `alternative_first`
/// `"default"`).
pub(super) fn tir_settings_given(
    method_choice: Option<bool>,
    alternative_first: &Option<Limit>,
    acceptance: Option<f64>,
) -> bool {
    method_choice.is_some() || *alternative_first != Limit::default_limit() || acceptance.is_some()
}

/// TIRust's settings from the validation keywords of [`V2TirReducer`] and
/// `V2Executor.prepare_reduction`: `method_choice` and `acceptance` (None:
/// the default), `alternative_first` ([`Limit`]); the others the defaults
/// (`v2_tir::TirOptions`).
#[cfg(feature = "tir")]
pub(super) fn tir_options(
    method_choice: Option<bool>,
    alternative_first: &Option<Limit>,
    acceptance: Option<f64>,
) -> PyResult<rusticol_loop_reduction::v2_tir::TirOptions> {
    let defaults = rusticol_loop_reduction::v2_tir::TirOptions::default();
    let alternative_first = match alternative_first {
        None => None,
        Some(Limit::Value(limit)) => Some(*limit),
        Some(Limit::Named(name)) if name == "default" => defaults.alternative_first,
        Some(Limit::Named(other)) => {
            return Err(error(format!(
                "alternative_first must be a limit, None or \"default\", not {other:?}"
            )));
        }
    };
    Ok(rusticol_loop_reduction::v2_tir::TirOptions {
        method_choice: method_choice.unwrap_or(defaults.method_choice),
        alternative_first,
        acceptance: acceptance.unwrap_or(defaults.acceptance),
        ..defaults
    })
}

/// The TIR backend (`rusticol_loop_reduction::v2_tir`, binary64) on the
/// groups of [`V2Reducer::reduce`] (the Python reference path's reducer
/// when the call's reduction backend is TIRust): one family per group,
/// consecutive lanes of bitwise equal kinematics one moment row. Its check
/// frame is the rotation by `rotation` with the origin shift (`rescale`
/// adds the rescaling); `reverse` and `shift` are NinjaRS's and ignored.
/// The keywords override the backend's settings (`v2_tir::TirOptions`,
/// whose defaults they keep when omitted; for validation): `method_choice`
/// (default off: gate-refused rows fail closed; True: TIRust's method
/// choice and its check), `alternative_first` ([`Limit`]) and `acceptance`
/// (the method-choice check's tolerance). Refused typed in a build without
/// TIRust.
#[pyclass]
struct V2TirReducer(rusticol_loop_reduction::runtime::V2Backends);

#[pymethods]
impl V2TirReducer {
    #[new]
    #[pyo3(signature = (*, method_choice=None, alternative_first=Limit::default_limit(), acceptance=None))]
    fn new(
        method_choice: Option<bool>,
        alternative_first: Option<Limit>,
        acceptance: Option<f64>,
    ) -> PyResult<Self> {
        #[cfg(feature = "tir")]
        {
            let options = tir_options(method_choice, &alternative_first, acceptance)?;
            rusticol_loop_reduction::runtime::V2Backends::tirust_with([], options)
                .map(Self)
                .map_err(error)
        }
        #[cfg(not(feature = "tir"))]
        {
            let _ = (method_choice, alternative_first, acceptance);
            rusticol_loop_reduction::runtime::V2Backends::tirust([])
                .map(Self)
                .map_err(error)
        }
    }
    /// The (propagators, rank) plans built so far.
    fn plans(&self) -> Vec<(usize, usize)> {
        self.0.plans()
    }
    #[pyo3(signature = (groups, *, poles=None, check=false, reverse=true, rotation=1, shift=false, rescale=false))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn reduce(
        &mut self,
        py: Python<'_>,
        groups: Vec<Group<'_>>,
        poles: Option<Bound<'_, PyAny>>,
        check: bool,
        reverse: bool,
        rotation: usize,
        shift: bool,
        rescale: bool,
    ) -> PyResult<(Vec<ReductionBatch>, Option<Vec<ReductionBatch>>)> {
        let poles = poles.map_or(Ok(Poles::All), |value| poles_option(&value))?;
        let check = check.then_some(rusticol_core::loop_v2::retest::ReductionCheck {
            reverse,
            rotation,
            shift,
            rescale,
        });
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
        let backend = &mut self.0;
        let output = py
            .detach(|| {
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
                backend.reduce_family_groups(&groups, poles, check)
            })
            .map_err(error)?;
        let wrap = |batches: Vec<_>| batches.into_iter().map(ReductionBatch).collect();
        Ok((wrap(output.direct), output.check.map(wrap)))
    }
}

/// Binary64 seed weights on certified regular symmetric-group orbits
/// (`runtime::V2SeedFourier`). `kernels[k]` lists `(re, im)` of `k(x)` at the
/// lexicographic ranks of `S_degree`; `pairs` are `(kernel, Born orbit, loop
/// orbit)`. `weights` maps a batch with columns `a * |G| + g` (`B_a(g)`) to
/// columns `l * |G| + h` holding `sum_{a,g} conj(B_a(g)) k_al(g^-1 h)`.
#[pyclass]
struct V2SeedFourier(runtime::V2SeedFourier);

#[pymethods]
impl V2SeedFourier {
    #[new]
    fn new(
        degree: usize,
        kernels: Vec<Vec<(f64, f64)>>,
        pairs: Vec<[usize; 3]>,
        born_orbits: usize,
        loop_orbits: usize,
    ) -> PyResult<Self> {
        let kernels: Vec<Vec<_>> = kernels
            .into_iter()
            .map(|k| {
                k.into_iter()
                    .map(|(re, im)| rusticol_loop_reduction::C64::new(re, im))
                    .collect()
            })
            .collect();
        runtime::V2SeedFourier::new(degree, &kernels, &pairs, born_orbits, loop_orbits)
            .map(Self)
            .map_err(error)
    }
    #[getter]
    fn group_order(&self) -> usize {
        self.0.group_order()
    }
    fn weights(&mut self, py: Python<'_>, input: &NativeBatch) -> PyResult<NativeBatch> {
        py.detach(|| self.0.weights(&input.0))
            .map(NativeBatch)
            .map_err(error)
    }
}

/// Exponents `(q0, q1, q2, q3)` of every coefficient of a dense rank-`rank`
/// tensor numerator, in NinjaRS's storage order.
#[pyfunction]
fn v2_tensor_monomials(rank: usize) -> Vec<[u32; 4]> {
    rusticol_loop_reduction::runtime::tensor_monomials(rank)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    // The binary64 fast engine's absolute routing threshold in model units
    // (`runtime::fast_engine_threshold`): the generation record of light
    // masses compares with it (`GeneratedProcess.binary64_exact_engine`).
    module.add(
        "V2_FAST_ENGINE_THRESHOLD",
        rusticol_loop_reduction::runtime::fast_engine_threshold(),
    )?;
    module.add_class::<V2Reducer>()?;
    module.add_class::<V2TirReducer>()?;
    module.add_class::<V2SeedFourier>()?;
    module.add_function(wrap_pyfunction!(v2_tensor_monomials, module)?)
}
