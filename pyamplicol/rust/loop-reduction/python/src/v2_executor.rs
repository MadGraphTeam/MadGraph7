// SPDX-License-Identifier: 0BSD

//! Python ownership of the native v2 executor (`loops/v2/plan_export.py`).
//!
//! A `V2Executor` loads one plan v4 for one carrier: `double` (binary64,
//! NinjaRS fast engine), `double-double` and `arbitrary` (decimal digits;
//! the generic engine), or `exact` (rational numerators only). Batch inputs
//! are a dict of exact momenta, scales and squared masses (decimal or
//! fraction strings) and carrier values of kernel constants, block rows and
//! Born components (float pairs in binary64, string pairs otherwise).
use super::*;
use pyo3::IntoPyObject;
use pyo3::types::PyDict;
use rusticol_core::loop_execution::v2::{
    PointNumerators, PointResult, Report, Schedule, V2Executor as Executor, V2Options,
};
use rusticol_core::loop_v2::dress::{Carrier as V2Carrier, V2Batch, parse_rational};
use rusticol_core::loop_v2::plan::LoopProgramV4;
use rusticol_core::loop_v2::retest::ReductionCheck;
use rusticol_loop_reduction::runtime::{TypedBatch, V2ReductionBackend};
use std::collections::BTreeMap;
use std::sync::Arc;
use symbolica::domains::float::Real;

/// Complex values handed back to Python.
#[derive(IntoPyObject)]
enum Values {
    Float(Vec<[f64; 2]>),
    Text(Vec<[String; 2]>),
}

/// A native carrier with its Python value conversions.
trait Exchanged: V2Carrier {
    fn import(values: &Bound<'_, PyAny>, zero: &Self) -> PyResult<Vec<Complex<Self>>>;
    fn export(values: &[Complex<Self>], bits: u32) -> Values;
}

fn exact_pairs<T: V2Carrier>(values: &Bound<'_, PyAny>, zero: &T) -> PyResult<Vec<Complex<T>>> {
    values
        .extract::<Vec<[String; 2]>>()?
        .iter()
        .map(|[re, im]| {
            Ok(Complex::new(
                zero.from_rational(&parse_rational(re).map_err(error)?),
                zero.from_rational(&parse_rational(im).map_err(error)?),
            ))
        })
        .collect()
}

impl Exchanged for f64 {
    fn import(values: &Bound<'_, PyAny>, _zero: &Self) -> PyResult<Vec<Complex<Self>>> {
        Ok(values
            .extract::<Vec<[f64; 2]>>()?
            .into_iter()
            .map(|[re, im]| Complex::new(re, im))
            .collect())
    }
    fn export(values: &[Complex<Self>], _bits: u32) -> Values {
        Values::Float(values.iter().map(|z| [z.re, z.im]).collect())
    }
}

impl Exchanged for DoubleFloat {
    fn import(values: &Bound<'_, PyAny>, zero: &Self) -> PyResult<Vec<Complex<Self>>> {
        exact_pairs(values, zero)
    }
    fn export(values: &[Complex<Self>], bits: u32) -> Values {
        Values::Text(
            values
                .iter()
                .cloned()
                .map(|z| decimal_complex(z, bits))
                .collect(),
        )
    }
}

impl Exchanged for Float {
    fn import(values: &Bound<'_, PyAny>, zero: &Self) -> PyResult<Vec<Complex<Self>>> {
        exact_pairs(values, zero)
    }
    fn export(values: &[Complex<Self>], bits: u32) -> Values {
        Values::Text(
            values
                .iter()
                .cloned()
                .map(|z| decimal_complex(z, bits))
                .collect(),
        )
    }
}

impl Exchanged for Rational {
    fn import(values: &Bound<'_, PyAny>, zero: &Self) -> PyResult<Vec<Complex<Self>>> {
        exact_pairs(values, zero)
    }
    fn export(values: &[Complex<Self>], _bits: u32) -> Values {
        Values::Text(
            values
                .iter()
                .map(|z| [z.re.to_string(), z.im.to_string()])
                .collect(),
        )
    }
}

fn item<'py>(inputs: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    inputs
        .get_item(key)?
        .ok_or_else(|| error(format!("batch input {key:?} is missing")))
}

fn rationals(values: Vec<String>) -> PyResult<Vec<Rational>> {
    values
        .iter()
        .map(|value| parse_rational(value).map_err(error))
        .collect()
}

fn batch<T: Exchanged>(inputs: &Bound<'_, PyDict>, zero: &T) -> PyResult<V2Batch<T>> {
    let per_point = |key: &str| -> PyResult<Vec<Vec<Complex<T>>>> {
        item(inputs, key)?
            .try_iter()?
            .map(|values| T::import(&values?, zero))
            .collect()
    };
    Ok(V2Batch {
        momenta: item(inputs, "momenta")?
            .extract::<Vec<Vec<String>>>()?
            .into_iter()
            .map(rationals)
            .collect::<PyResult<_>>()?,
        scales: rationals(item(inputs, "scales")?.extract()?)?,
        masses: rationals(item(inputs, "masses")?.extract()?)?,
        constants: T::import(&item(inputs, "constants")?, zero)?,
        block_rows: per_point("block_rows")?,
        born: per_point("born")?,
    })
}

type ExportedNumerators = Vec<Vec<(usize, usize, usize, Vec<Values>)>>;

fn export_numerators<T: Exchanged>(
    points: Vec<PointNumerators<T>>,
    bits: u32,
) -> ExportedNumerators {
    points
        .into_iter()
        .map(|families| {
            families
                .into_iter()
                .map(|(family, order, rank, columns)| {
                    (
                        family,
                        order,
                        rank,
                        columns.iter().map(|c| T::export(c, bits)).collect(),
                    )
                })
                .collect()
        })
        .collect()
}

type Lanes = Vec<((usize, usize, usize), Values)>;

/// Python form of an evaluation.
#[derive(IntoPyObject)]
struct Evaluation {
    families: Vec<Lanes>,
    check: Option<Vec<Lanes>>,
    bare: Vec<Vec<Values>>,
    budget: Vec<Option<Vec<Vec<f64>>>>,
    failures: Vec<Vec<String>>,
    report: ReportObject,
}

#[derive(IntoPyObject)]
struct ReportObject {
    seconds: BTreeMap<&'static str, f64>,
    counts: BTreeMap<&'static str, u64>,
    reals_per_lane: BTreeMap<&'static str, u64>,
    real_bytes: usize,
    lanes: usize,
    threads: usize,
}

fn export_evaluation<T: Exchanged + Real>(
    results: Vec<PointResult<T>>,
    report: Report,
    bits: u32,
) -> Evaluation {
    let lanes = |values: &[(rusticol_core::loop_execution::v2::LaneKey, Vec<Complex<T>>)]| {
        values
            .iter()
            .map(|(key, value)| ((key.family, key.order, key.column), T::export(value, bits)))
            .collect::<Lanes>()
    };
    let checked = results.iter().any(|r| r.check.is_some());
    Evaluation {
        families: results.iter().map(|r| lanes(&r.families)).collect(),
        check: checked.then(|| {
            results
                .iter()
                .map(|r| r.check.as_deref().map(&lanes).unwrap_or_default())
                .collect()
        }),
        bare: results
            .iter()
            .map(|r| {
                r.bare
                    .iter()
                    .map(|column| {
                        let zero = column.first().map(|x| x.zero());
                        T::export(
                            &column
                                .iter()
                                .map(|x| Complex::new(x.clone(), zero.clone().expect("value")))
                                .collect::<Vec<_>>(),
                            bits,
                        )
                    })
                    .collect()
            })
            .collect(),
        budget: results
            .iter()
            .map(|r| {
                r.budget.as_ref().map(|columns| {
                    columns
                        .iter()
                        .map(|column| column.iter().map(|x| x.to_f64()).collect())
                        .collect()
                })
            })
            .collect(),
        failures: results.into_iter().map(|r| r.failures).collect(),
        report: ReportObject {
            seconds: report.seconds,
            counts: report.counts,
            reals_per_lane: report.reals_per_lane,
            real_bytes: report.real_bytes,
            lanes: report.lanes,
            threads: report.threads,
        },
    }
}

enum Carriers {
    Double,
    DoubleDouble,
    Arbitrary(u32),
    Exact,
}

/// The native v2 executor of one plan and carrier. `lanes` is the SIMD
/// lane count per chunk (default 16, gate G0); `threads` bounds the family
/// threads (default: all cores).
#[pyclass]
struct V2Executor {
    executor: Executor,
    carrier: Carriers,
    backend: Option<V2ReductionBackend>,
}

impl V2Executor {
    fn backend(&mut self) -> PyResult<&mut V2ReductionBackend> {
        if self.backend.is_none() {
            let reducer = match self.carrier {
                Carriers::Double => runtime::V2Reducer::new("double", None, "fast", true),
                Carriers::DoubleDouble => {
                    runtime::V2Reducer::new("double-double", None, "generic", true)
                }
                Carriers::Arbitrary(digits) => {
                    runtime::V2Reducer::new("arbitrary", Some(digits), "generic", true)
                }
                Carriers::Exact => {
                    return Err(error("the exact carrier dresses numerators only"));
                }
            }
            .map_err(error)?;
            self.backend = Some(V2ReductionBackend { reducer });
        }
        Ok(self.backend.as_mut().expect("backend"))
    }

    fn bits(&self) -> PyResult<u32> {
        Ok(match self.carrier {
            Carriers::Double => 53,
            Carriers::DoubleDouble => 106,
            Carriers::Arbitrary(digits) => {
                Float::decimal_digits_to_bits(f64::from(digits)).map_err(error)?
            }
            Carriers::Exact => 0,
        })
    }
}

fn evaluate_with<T: Exchanged + Real + TypedBatch>(
    py: Python<'_>,
    executor: &Executor,
    backend: &mut V2ReductionBackend,
    inputs: &Bound<'_, PyDict>,
    zero: T,
    bits: u32,
    options: V2Options,
) -> PyResult<Evaluation> {
    let batch = batch(inputs, &zero)?;
    let (results, report) = py
        .detach(|| executor.evaluate(&batch, &zero, backend, options))
        .map_err(error)?;
    Ok(export_evaluation(results, report, bits))
}

fn numerators_with<T: Exchanged>(
    py: Python<'_>,
    executor: &Executor,
    inputs: &Bound<'_, PyDict>,
    zero: T,
    bits: u32,
    schedule: Schedule,
) -> PyResult<ExportedNumerators> {
    let batch = batch(inputs, &zero)?;
    let numerators = py
        .detach(|| executor.numerators(&batch, &zero, schedule))
        .map_err(error)?;
    Ok(export_numerators(numerators, bits))
}

#[pymethods]
impl V2Executor {
    #[new]
    #[pyo3(signature = (plan, arithmetic="double", precision=None, *, lanes=16, threads=None))]
    fn new(
        plan: &str,
        arithmetic: &str,
        precision: Option<u32>,
        lanes: usize,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        let carrier = match (arithmetic, precision) {
            ("double", None) => Carriers::Double,
            ("double-double", None) => Carriers::DoubleDouble,
            ("arbitrary", Some(digits)) if digits > 0 => Carriers::Arbitrary(digits),
            ("exact", None) => Carriers::Exact,
            _ => {
                return Err(error(
                    "arithmetic must be double, double-double, exact, or arbitrary with digits",
                ));
            }
        };
        let plan = Arc::new(LoopProgramV4::load(plan.as_bytes()).map_err(error)?);
        Ok(Self {
            executor: Executor::new(plan, lanes, threads).map_err(error)?,
            carrier,
            backend: None,
        })
    }

    #[getter]
    fn lanes(&self) -> usize {
        self.executor.lanes()
    }

    #[getter]
    fn families(&self) -> usize {
        self.executor.plan().families.len()
    }

    /// Family numerators per point: `(family, order, rank, columns)` with
    /// one coefficient vector per observable column, schedule `B` or `F`.
    #[pyo3(signature = (inputs, schedule="B"))]
    fn numerators(
        &self,
        py: Python<'_>,
        inputs: &Bound<'_, PyDict>,
        schedule: &str,
    ) -> PyResult<ExportedNumerators> {
        let schedule = match schedule {
            "B" => Schedule::B,
            "F" => Schedule::F,
            _ => return Err(error("schedule must be B or F")),
        };
        let bits = self.bits()?;
        let executor = &self.executor;
        match self.carrier {
            Carriers::Double => numerators_with(py, executor, inputs, 0.0, bits, schedule),
            Carriers::DoubleDouble => {
                numerators_with(py, executor, inputs, DoubleFloat::from(0.0), bits, schedule)
            }
            Carriers::Arbitrary(_) => numerators_with(
                py,
                executor,
                inputs,
                Float::with_val(bits, 0),
                bits,
                schedule,
            ),
            Carriers::Exact => {
                numerators_with(py, executor, inputs, Rational::from(0), bits, schedule)
            }
        }
    }

    /// Dress, reduce and assemble the bare virtual of every point. Returns
    /// `families` (per point `((family, order, column), Laurent)`), `check`
    /// (the re-test integrals or None), `bare` (per point and column),
    /// `budget` (re-test error budgets), `failures` and `report`.
    #[pyo3(signature = (inputs, *, poles=true, check=false, reverse=true, rotation=1, shift=false, chunk_families=None))]
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &mut self,
        py: Python<'_>,
        inputs: &Bound<'_, PyDict>,
        poles: bool,
        check: bool,
        reverse: bool,
        rotation: usize,
        shift: bool,
        chunk_families: Option<usize>,
    ) -> PyResult<Evaluation> {
        let mut options = V2Options {
            poles,
            check: check.then_some(ReductionCheck {
                reverse,
                rotation,
                shift,
            }),
            ..V2Options::default()
        };
        if let Some(chunk) = chunk_families {
            options.chunk_families = chunk;
        }
        let bits = self.bits()?;
        let carrier = match self.carrier {
            Carriers::Double => 0,
            Carriers::DoubleDouble => 1,
            Carriers::Arbitrary(_) => 2,
            Carriers::Exact => 3,
        };
        self.backend()?;
        let backend = self.backend.as_mut().expect("backend");
        let executor = &self.executor;
        match carrier {
            0 => evaluate_with(py, executor, backend, inputs, 0.0, bits, options),
            1 => evaluate_with(
                py,
                executor,
                backend,
                inputs,
                DoubleFloat::from(0.0),
                bits,
                options,
            ),
            _ => evaluate_with(
                py,
                executor,
                backend,
                inputs,
                Float::with_val(bits, 0),
                bits,
                options,
            ),
        }
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<V2Executor>()
}
