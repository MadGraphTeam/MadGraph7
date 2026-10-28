// SPDX-License-Identifier: 0BSD

//! Python ownership of the native v2 executor (`loops/v2/plan_export.py`).
//!
//! A `V2Executor` loads one plan v4 for one carrier: `double` (binary64,
//! NinjaRS fast engine), `double-double` and `arbitrary` (decimal digits;
//! the generic engine), or `exact` (rational numerators only). Batch inputs
//! are a dict of exact momenta, scales, squared masses and the legs'
//! squared masses (`leg_masses`; decimal or fraction strings) and carrier
//! values of kernel constants, block rows and Born components (float pairs in
//! binary64, string pairs otherwise).
//! With a tree program (`trees=`, WP3.1) the block rows and Born components
//! are formed natively instead: the batch then carries `parameters`, the
//! exact external parameter values of the program
//! (`TreeProgram::parameter_names`), from which the kernel constants, masses,
//! couplings and model parameters are bound natively (the scale parameter
//! from each point's mu_R²); it has no `constants`, `masses`, `leg_masses`,
//! `block_rows` or `born`.
use super::*;
use pyo3::IntoPyObject;
use pyo3::types::PyDict;
use rusticol_core::loop_execution::v2::{
    PointNumerators, PointResult, Report, Schedule, V2Executor as Executor, V2Options,
};
use rusticol_core::loop_v2::dress::{Carrier as V2Carrier, V2Batch, parse_rational};
use rusticol_core::loop_v2::plan::{FormPolicy, LoopProgramV4};
use rusticol_core::loop_v2::retest::ReductionCheck;
use rusticol_core::loop_v2::trees::{
    BoundParameters, TreeCarrier, TreeInputs, TreeProgram, TreeRuntime,
};
use rusticol_loop_reduction::runtime::{TypedBatch, V2ReductionBackend};
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
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
    let native = inputs.contains("parameters")?;
    let per_point = |key: &str| -> PyResult<Vec<Vec<Complex<T>>>> {
        if native {
            return Ok(Vec::new());
        }
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
        // Bound natively from the external parameters (`Trees::fill`).
        masses: if native {
            Vec::new()
        } else {
            rationals(item(inputs, "masses")?.extract()?)?
        },
        // The legs' squared masses, the on-shell invariants (DESIGN §11.1);
        // bound natively from the external parameters like `masses`. A
        // synthetic batch without external states (timing tools) may omit
        // them: its invariants then all come from the momenta.
        leg_masses: match inputs.get_item("leg_masses")? {
            Some(values) if !native => values
                .extract::<Vec<String>>()?
                .into_iter()
                .map(|text| parse_rational(&text).map(Some).map_err(error))
                .collect::<PyResult<_>>()?,
            _ => Vec::new(),
        },
        constants: if native {
            Vec::new()
        } else {
            T::import(&item(inputs, "constants")?, zero)?
        },
        block_rows: per_point("block_rows")?,
        born: per_point("born")?,
    })
}

/// The native tree program of an executor and its carrier form.
struct Trees {
    program: TreeProgram,
    /// The program in the executor's carrier (a `TreeRuntime<T>`), built on
    /// first use; the executor is shared with Python threads, hence the lock.
    runtime: Mutex<Option<Box<dyn Any + Send>>>,
}

impl Trees {
    /// The batch's bound parameters (`TreeProgram::bind`) and the carrier's
    /// tree runtime, built on first use.
    fn bind<'a, T: Exchanged + Real + TreeCarrier>(
        &self,
        guard: &'a mut Option<Box<dyn Any + Send>>,
        inputs: &Bound<'_, PyDict>,
        scales: &[Rational],
        zero: &T,
    ) -> PyResult<(&'a mut TreeRuntime<T>, BoundParameters<T>)>
    where
        Complex<T>: Real + symbolica::evaluate::EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        let values = rationals(item(inputs, "parameters")?.extract::<Vec<String>>()?)?;
        let fresh = !guard
            .as_ref()
            .is_some_and(|runtime| runtime.is::<TreeRuntime<T>>());
        if fresh {
            *guard = Some(Box::new(self.program.runtime(zero).map_err(error)?));
        }
        let runtime = guard
            .as_mut()
            .and_then(|runtime| runtime.downcast_mut::<TreeRuntime<T>>())
            .expect("a runtime of this carrier");
        let bound = self
            .program
            .bind(runtime, &values, scales, zero)
            .map_err(error)?;
        Ok((runtime, bound))
    }

    /// Native block rows and Born components of a batch (filled into
    /// `batch`); returns every input for inspection.
    fn fill<T: Exchanged + Real + TreeCarrier>(
        &mut self,
        plan: &LoopProgramV4,
        inputs: &Bound<'_, PyDict>,
        batch: &mut V2Batch<T>,
        poles: bool,
        zero: &T,
    ) -> PyResult<TreeInputs<T>>
    where
        Complex<T>: Real + symbolica::evaluate::EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        let mut guard = self
            .runtime
            .lock()
            .map_err(|_| error("a poisoned tree runtime"))?;
        let (runtime, bound) = self.bind(&mut guard, inputs, &batch.scales, zero)?;
        batch.constants = bound.constants;
        batch.masses = bound.masses;
        batch.leg_masses = bound.leg_masses;
        let sources = match inputs.get_item("sources")? {
            Some(value) if !value.is_none() => Some(point_sources(&value, zero)?),
            _ => None,
        };
        let out = self
            .program
            .inputs_with(
                runtime,
                plan,
                &bound.trees,
                &batch.momenta,
                &batch.constants,
                poles,
                sources.as_deref(),
                zero,
            )
            .map_err(error)?;
        batch.block_rows = out.block_rows.clone();
        batch.born = out.born.clone();
        Ok(out)
    }
}

/// Per point spin sources (`inputs["sources"]`): per point `(vectors,
/// rows)`, `vectors` (1-based leg label, four `[re, im]` exact texts) and
/// `rows` a kept flag per Born row, or None (every row).
#[allow(clippy::type_complexity)]
fn point_sources<T: V2Carrier>(
    value: &Bound<'_, PyAny>,
    zero: &T,
) -> PyResult<Vec<rusticol_core::loop_v2::trees::PointSources<T>>> {
    let parsed: Vec<(Vec<(usize, Vec<[String; 2]>)>, Option<Vec<bool>>)> = value.extract()?;
    parsed
        .into_iter()
        .map(|(vectors, rows)| {
            let vectors = vectors
                .into_iter()
                .map(|(leg, components)| {
                    if components.len() != 4 {
                        return Err(error("a spin source is a complex four-vector"));
                    }
                    let parsed = components
                        .iter()
                        .map(|[re, im]| {
                            Ok(Complex::new(
                                zero.from_rational(&parse_rational(re).map_err(error)?),
                                zero.from_rational(&parse_rational(im).map_err(error)?),
                            ))
                        })
                        .collect::<PyResult<Vec<_>>>()?;
                    let array: [Complex<T>; 4] = parsed
                        .try_into()
                        .map_err(|_| error("a spin source is a complex four-vector"))?;
                    Ok((leg, array))
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(rusticol_core::loop_v2::trees::PointSources { vectors, rows })
        })
        .collect()
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
    /// |A0|² per point and squared Born order (native trees with a colour
    /// metric), else None.
    born: Option<Vec<Values>>,
    /// The counterterm and external-field outputs per point (native trees
    /// with counterterms), else None.
    counterterms: Option<Vec<Values>>,
    /// Per point and observable column its complex outputs (native trees
    /// with observable columns), else None.
    columns: Option<Vec<Vec<Values>>>,
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

fn real_values<T: Exchanged + Real>(values: &[T], zero: &T, bits: u32) -> Values {
    T::export(
        &values
            .iter()
            .map(|x| Complex::new(x.clone(), zero.clone()))
            .collect::<Vec<_>>(),
        bits,
    )
}

fn export_evaluation<T: Exchanged + Real>(
    results: Vec<PointResult<T>>,
    report: Report,
    bits: u32,
    born: Option<Vec<Values>>,
    counterterms: Option<Vec<Values>>,
    columns: Option<Vec<Vec<Values>>>,
) -> Evaluation {
    let lanes = |values: &[(rusticol_core::loop_execution::v2::LaneKey, Vec<Complex<T>>)]| {
        values
            .iter()
            .map(|(key, value)| ((key.family, key.order, key.column), T::export(value, bits)))
            .collect::<Lanes>()
    };
    let checked = results.iter().any(|r| r.check.is_some());
    Evaluation {
        born,
        counterterms,
        columns,
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

#[derive(Clone, Copy)]
enum Carriers {
    Double,
    DoubleDouble,
    Arbitrary(u32),
    Exact,
}

/// The native v2 executor of one plan and carrier. `lanes` is the SIMD
/// lane count per chunk (default 16, gate G0); `threads` bounds the family
/// threads (default: all cores); `forms` picks the kernel form per step:
/// `rule` (gate G0's selection rule), `fused` or `split` (wherever a step
/// can split).
#[pyclass]
struct V2Executor {
    executor: Executor,
    carrier: Carriers,
    /// The binary64 reduction engine: `fast` (production) or `generic`
    /// (oracle and cost reference of selected master poles).
    engine: String,
    backend: Option<V2ReductionBackend>,
    trees: Option<Trees>,
}

/// The reduction backend of a carrier, built on first use. A free function
/// of the backend slot, so an evaluation can borrow the executor alongside.
/// `engine` is the binary64 reduction engine; the extended carriers always
/// reduce with the generic engine.
fn backend_of<'a>(
    slot: &'a mut Option<V2ReductionBackend>,
    carrier: Carriers,
    engine: &str,
) -> PyResult<&'a mut V2ReductionBackend> {
    if slot.is_none() {
        let reducer = match carrier {
            Carriers::Double => runtime::V2Reducer::new("double", None, engine, true),
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
        *slot = Some(V2ReductionBackend {
            reducer,
            masters: v2::master_selection("both")?,
        });
    }
    Ok(slot.as_mut().expect("backend"))
}

impl V2Executor {
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

#[allow(clippy::too_many_arguments)]
fn evaluate_with<T: Exchanged + Real + TypedBatch + TreeCarrier>(
    py: Python<'_>,
    executor: &Executor,
    trees: Option<&mut Trees>,
    backend: &mut V2ReductionBackend,
    inputs: &Bound<'_, PyDict>,
    zero: T,
    bits: u32,
    options: V2Options,
) -> PyResult<Evaluation>
where
    Complex<T>: Real + symbolica::evaluate::EvaluationDomain,
    TreeRuntime<T>: Send,
{
    let mut batch = batch(inputs, &zero)?;
    let mut born = None;
    let mut counterterms = None;
    let mut columns = None;
    let mut tree_seconds = None;
    if inputs.contains("parameters")? {
        let trees = trees.ok_or_else(|| error("the executor has no tree program"))?;
        // The counterterm pole outputs are formed unless finite-only (the
        // split modes read the same per-power outputs).
        let poles = !matches!(
            options.poles,
            rusticol_core::loop_execution::v2::Poles::None
        );
        let started = std::time::Instant::now();
        let native = trees.fill(executor.plan(), inputs, &mut batch, poles, &zero)?;
        // The report's "trees" stage: the whole native input fill, then its
        // parts (`TreeInputs::seconds`).
        tree_seconds = Some((started.elapsed().as_secs_f64(), native.seconds.clone()));
        let export = |points: Vec<Vec<T>>| -> Vec<Values> {
            points
                .iter()
                .map(|values| real_values(values, &zero, bits))
                .collect()
        };
        born = native.born_squared.map(export);
        counterterms = native.counterterms.map(export);
        columns = native.columns.map(|points| {
            points
                .iter()
                .map(|columns| {
                    columns
                        .iter()
                        .map(|values| T::export(values, bits))
                        .collect()
                })
                .collect()
        });
    }
    let (results, mut report) = py
        .detach(|| executor.evaluate(&batch, &zero, backend, options))
        .map_err(error)?;
    if let Some((total, parts)) = tree_seconds {
        report.seconds.insert("trees", total);
        report.seconds.extend(parts);
    }
    Ok(export_evaluation(
        results,
        report,
        bits,
        born,
        counterterms,
        columns,
    ))
}

/// Native tree inputs of a batch: (block rows, plan Born components, every
/// Born component, |A0|² or None, counterterm outputs or None) per point.
type ExportedTreeInputs = (
    Vec<Values>,
    Vec<Values>,
    Vec<Values>,
    Option<Vec<Values>>,
    Option<Vec<Values>>,
);

fn tree_inputs_with<T: Exchanged + Real + TreeCarrier>(
    executor: &Executor,
    trees: &mut Trees,
    inputs: &Bound<'_, PyDict>,
    zero: T,
    bits: u32,
) -> PyResult<ExportedTreeInputs>
where
    Complex<T>: Real + symbolica::evaluate::EvaluationDomain,
    TreeRuntime<T>: Send,
{
    let mut batch = batch(inputs, &zero)?;
    let out = trees.fill(executor.plan(), inputs, &mut batch, true, &zero)?;
    let export = |points: &[Vec<Complex<T>>]| -> Vec<Values> {
        points
            .iter()
            .map(|values| T::export(values, bits))
            .collect()
    };
    let reals = |points: Vec<Vec<T>>| -> Vec<Values> {
        points
            .iter()
            .map(|values| real_values(values, &zero, bits))
            .collect()
    };
    Ok((
        export(&out.block_rows),
        export(&out.born),
        export(&out.amplitudes),
        out.born_squared.map(reals),
        out.counterterms.map(reals),
    ))
}

/// Per plan mass and per leg, the bound squared masses (exact fractions).
type BoundMasses = (Vec<String>, Vec<Option<String>>);

fn bound_masses_with<T: Exchanged + Real + TreeCarrier>(
    trees: &Trees,
    inputs: &Bound<'_, PyDict>,
    scales: &[Rational],
    zero: &T,
) -> PyResult<BoundMasses>
where
    Complex<T>: Real + symbolica::evaluate::EvaluationDomain,
    TreeRuntime<T>: Send,
{
    let mut guard = trees
        .runtime
        .lock()
        .map_err(|_| error("a poisoned tree runtime"))?;
    let (_runtime, bound) = trees.bind(&mut guard, inputs, scales, zero)?;
    Ok((
        bound.masses.iter().map(ToString::to_string).collect(),
        bound
            .leg_masses
            .iter()
            .map(|mass| mass.as_ref().map(ToString::to_string))
            .collect(),
    ))
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
    /// `trees` is a tree program (`loops/v2/native_trees.py`): its JSON,
    /// the saved exact evaluator of each kernel and of the parameter program
    /// (last), and their binary64 SymJIT plane applications in the same order
    /// (or None: binary64 then runs the exact states).
    /// `columns` (JSON, `loops/v2/native_observables`) adds observable
    /// columns to the tree program and `plan_components` replaces its plan
    /// Born components (an observable column set's plan).
    #[new]
    #[pyo3(signature = (plan, arithmetic="double", precision=None, *, lanes=16, threads=None, forms="rule", engine="fast", trees=None, columns=None, plan_components=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        plan: &str,
        arithmetic: &str,
        precision: Option<u32>,
        lanes: usize,
        threads: Option<usize>,
        forms: &str,
        engine: &str,
        trees: Option<(String, Vec<Vec<u8>>, Option<Vec<Vec<u8>>>)>,
        columns: Option<String>,
        plan_components: Option<Vec<Option<usize>>>,
    ) -> PyResult<Self> {
        if !matches!(engine, "fast" | "generic") {
            return Err(error("engine must be fast or generic"));
        }
        let policy = match forms {
            "rule" => FormPolicy::Rule,
            "fused" => FormPolicy::Fused,
            "split" => FormPolicy::Split,
            _ => return Err(error("forms must be rule, fused or split")),
        };
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
        let plan = Arc::new(LoopProgramV4::load_with(plan.as_bytes(), policy).map_err(error)?);
        let trees = trees
            .map(|(json, kernels, planes)| {
                let mut program = TreeProgram::load_with_components(
                    json.as_bytes(),
                    &kernels,
                    planes.as_deref(),
                    &plan,
                    plan_components.clone(),
                )?;
                if let Some(columns) = &columns {
                    program.set_columns(columns.as_bytes())?;
                }
                Ok::<_, rusticol_core::RusticolError>(Trees {
                    program,
                    runtime: Mutex::new(None),
                })
            })
            .transpose()
            .map_err(error)?;
        if trees.is_none() && (columns.is_some() || plan_components.is_some()) {
            return Err(error("observable columns need a tree program"));
        }
        Ok(Self {
            executor: Executor::new(plan, lanes, threads).map_err(error)?,
            carrier,
            engine: engine.to_owned(),
            backend: None,
            trees,
        })
    }

    /// The native tree inputs of a batch (needs a tree program): per point
    /// `(block_rows, born, amplitudes, born_squared, counterterms)`, see
    /// `TreeInputs` (counterterms with poles).
    fn tree_inputs(&mut self, inputs: &Bound<'_, PyDict>) -> PyResult<ExportedTreeInputs> {
        let bits = self.bits()?;
        let executor = &self.executor;
        let trees = self
            .trees
            .as_mut()
            .ok_or_else(|| error("the executor has no tree program"))?;
        match self.carrier {
            Carriers::Double => tree_inputs_with(executor, trees, inputs, 0.0, bits),
            Carriers::DoubleDouble => {
                tree_inputs_with(executor, trees, inputs, DoubleFloat::from(0.0), bits)
            }
            Carriers::Arbitrary(_) => {
                tree_inputs_with(executor, trees, inputs, Float::with_val(bits, 0), bits)
            }
            Carriers::Exact => Err(error("the exact carrier has no native trees")),
        }
    }

    /// The squared masses a batch's external parameters bind natively
    /// (needs a tree program; `inputs` as for `tree_inputs`): per plan mass
    /// and per leg (the on-shell invariants, DESIGN §11.1), exact fractions.
    fn bound_masses(&self, inputs: &Bound<'_, PyDict>) -> PyResult<BoundMasses> {
        let trees = self
            .trees
            .as_ref()
            .ok_or_else(|| error("the executor has no tree program"))?;
        let scales = rationals(item(inputs, "scales")?.extract()?)?;
        match self.carrier {
            Carriers::Double => bound_masses_with(trees, inputs, &scales, &0.0),
            Carriers::DoubleDouble => {
                bound_masses_with(trees, inputs, &scales, &DoubleFloat::from(0.0))
            }
            Carriers::Arbitrary(_) => {
                bound_masses_with(trees, inputs, &scales, &Float::with_val(self.bits()?, 0))
            }
            Carriers::Exact => Err(error("the exact carrier has no native trees")),
        }
    }

    #[getter]
    fn lanes(&self) -> usize {
        self.executor.lanes()
    }

    #[getter]
    fn families(&self) -> usize {
        self.executor.plan().families.len()
    }

    /// Schedule-B steps per kernel form: `{"fused": n, "split": m}`.
    #[getter]
    fn forms(&self) -> BTreeMap<&'static str, usize> {
        let steps = self
            .executor
            .plan()
            .families
            .iter()
            .flat_map(|family| &family.steps);
        let mut out = BTreeMap::from([("fused", 0), ("split", 0)]);
        for step in steps {
            *out.entry(if step.split.is_some() {
                "split"
            } else {
                "fused"
            })
            .or_default() += 1;
        }
        out
    }

    /// Whether the plan has schedule M, and the families it dresses.
    #[getter]
    fn merged_families(&self) -> Option<Vec<usize>> {
        self.executor
            .plan()
            .merged
            .as_ref()
            .map(|merged| merged.families.clone())
    }

    /// Family numerators per point: `(family, order, rank, columns)` with
    /// one coefficient vector per observable column, schedule `B`, `F` or
    /// `M` (schedule M for its families, B for the others).
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
            "M" => Schedule::M,
            _ => return Err(error("schedule must be B, F or M")),
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
    /// `budget` (re-test error budgets), `failures` and `report`. `schedule`
    /// `M` dresses the plan's schedule-M families by M (streamed in parts of
    /// `part_chunks` chunks) and the others by B. `poles` is `"all"`, `"none"`
    /// or `"split"` (each lane's UV residue as a fourth column, DESIGN.md
    /// §12.4); `master_poles` selects the masters' residues (generic engine).
    #[pyo3(signature = (inputs, *, poles=None, check=false, reverse=true, rotation=1, shift=false, rescale=false, chunk_families=None, schedule="B", part_chunks=None, master_poles="both"))]
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &mut self,
        py: Python<'_>,
        inputs: &Bound<'_, PyDict>,
        poles: Option<Bound<'_, PyAny>>,
        check: bool,
        reverse: bool,
        rotation: usize,
        shift: bool,
        rescale: bool,
        chunk_families: Option<usize>,
        schedule: &str,
        part_chunks: Option<usize>,
        master_poles: &str,
    ) -> PyResult<Evaluation> {
        let poles = match poles {
            Some(value) => v2::poles_option(&value)?,
            None => V2Options::default().poles,
        };
        let masters = v2::master_selection(master_poles)?;
        let mut options = V2Options {
            poles,
            check: check.then_some(ReductionCheck {
                reverse,
                rotation,
                shift,
                rescale,
            }),
            schedule: match schedule {
                "B" => Schedule::B,
                "M" => Schedule::M,
                _ => return Err(error("an evaluation runs schedule B or M")),
            },
            ..V2Options::default()
        };
        if let Some(chunk) = chunk_families {
            options.chunk_families = chunk;
        }
        if let Some(chunks) = part_chunks {
            options.part_chunks = chunks;
        }
        let bits = self.bits()?;
        let carrier = self.carrier;
        // The generic reduction builds Symbolica expressions and the exact
        // tree kernels decode Symbolica states: pyAmpliCol's library key
        // (binary64 with the fast engine and plane kernels touches neither,
        // except lanes the fast engine routes to the generic one at its
        // absolute thresholds, which take the key themselves: `v2_reduce`
        // module docs).
        let _unlock = (self.engine == "generic" || !matches!(carrier, Carriers::Double))
            .then(pyamplicol::unlock)
            .flatten();
        let backend = backend_of(&mut self.backend, carrier, &self.engine)?;
        backend.masters = masters;
        let executor = &self.executor;
        let trees = self.trees.as_mut();
        match carrier {
            Carriers::Double => {
                evaluate_with(py, executor, trees, backend, inputs, 0.0, bits, options)
            }
            Carriers::DoubleDouble => evaluate_with(
                py,
                executor,
                trees,
                backend,
                inputs,
                DoubleFloat::from(0.0),
                bits,
                options,
            ),
            Carriers::Arbitrary(_) => evaluate_with(
                py,
                executor,
                trees,
                backend,
                inputs,
                Float::with_val(bits, 0),
                bits,
                options,
            ),
            Carriers::Exact => unreachable!("backend_of refuses the exact carrier"),
        }
    }
}

/// Admit one plan v4 process of an artifact through the native loader
/// (`rusticol_core::loop_artifact::open_loop_v2_artifact`) and describe it as
/// JSON: identity, plan digest, families, reduction buckets, program shape,
/// the declared metadata, the generation request and the companion plan of
/// certified-scaleless families (digest and families, or null).
#[pyfunction]
#[pyo3(signature = (path, process=None))]
fn admit_v2_artifact(py: Python<'_>, path: &str, process: Option<&str>) -> PyResult<String> {
    use rusticol_core::loop_artifact::{LOOP_V2_RUNTIME_CAPABILITY, open_loop_v2_artifact};

    let loaded = py
        .detach(|| open_loop_v2_artifact(path, process))
        .map_err(error)?;
    let program = &loaded.program;
    let buckets: Vec<[usize; 3]> = program
        .buckets()
        .into_iter()
        .map(|((n, r), families)| [n, r, families.len()])
        .collect();
    serde_json::to_string(&serde_json::json!({
        "artifact_id": loaded.artifact_id,
        "process_id": loaded.selection.process.id,
        "runtime_capability": LOOP_V2_RUNTIME_CAPABILITY,
        "plan_sha256": loaded.plan_sha256,
        "plan_bytes": loaded.plan_json.len(),
        "families": program.families.len(),
        "buckets": buckets,
        "legs": program.legs,
        "rows": program.rows,
        "columns": program.columns,
        "metadata": loaded.metadata,
        "request": loaded.request,
        "scaleless_plan_sha256": loaded.scaleless_plan_sha256,
        "scaleless_families": loaded.scaleless_program.as_ref().map(|p| p.families.len()),
    }))
    .map_err(error)
}

pyo3::create_exception!(
    _loop_reduction,
    PlaneMismatchError,
    pyo3::exceptions::PyValueError,
    "A plane application that disagrees with its exact evaluator at the plane \
     check's probes (`compile_v2_plane`). Its arguments are the message, the \
     output index, the probe row and whether the imaginary part disagrees."
);

/// Generation: the binary64 SymJIT plane application of a saved exact
/// evaluator of a tree program (a kernel or the parameter program), checked
/// against the evaluator (`rusticol_core::loop_v2::trees::compile_plane_
/// application_checked`): `(application, inputs, outputs)`. A disagreement
/// raises `PlaneMismatchError(message, output, row, imaginary)`, a
/// `ValueError` (`native_trees._plane_applications` refuses the program
/// naming the output); any other failure a plain `ValueError`.
#[pyfunction]
#[pyo3(signature = (state, optimization_level=2, compress=false))]
fn compile_v2_plane<'py>(
    py: Python<'py>,
    state: &[u8],
    optimization_level: u8,
    compress: bool,
) -> PyResult<(Bound<'py, pyo3::types::PyBytes>, usize, usize)> {
    let checked = py
        .detach(|| {
            rusticol_core::loop_v2::trees::compile_plane_application_checked(
                state,
                optimization_level,
                compress,
            )
        })
        .map_err(error)?;
    match checked {
        Ok((plane, inputs, outputs)) => {
            Ok((pyo3::types::PyBytes::new(py, &plane), inputs, outputs))
        }
        Err(mismatch) => Err(PlaneMismatchError::new_err((
            format!("loop v2 trees: {mismatch}"),
            mismatch.output,
            mismatch.row,
            mismatch.imaginary,
        ))),
    }
}

/// Whether pyAmpliCol's Symbolica library key unlocks the native code of
/// this process (crate `pyamplicol`).
#[pyfunction]
fn native_library_key_active() -> bool {
    pyamplicol::is_active()
}

/// One point of [`evaluate_v2_artifact`]: (Born per declared Born order,
/// complete virtual per declared sector), or None with the failures.
type V2ArtifactPoint = (Option<(Vec<f64>, Vec<Vec<f64>>)>, Vec<String>);

/// Evaluate a plan v4 process with native point inputs through the Rust
/// runtime the SDKs use (`rusticol_loop_reduction::v2_runtime::V2Runtime`):
/// no Python generation or assembly, binary64. `momenta` are per point the
/// physical momenta of every leg (`4 × legs` exact decimal or fraction
/// strings), `scales` mu_R² per point, `parameters` the caller's model
/// parameter values (exact strings), `poles` the pole selection ("all",
/// "none", "uv" or "ir"; True and False are "all" and "none").
#[pyfunction]
#[pyo3(signature = (path, momenta, scales, *, process=None, parameters=None, poles=None, threads=None))]
#[allow(clippy::too_many_arguments)]
fn evaluate_v2_artifact(
    py: Python<'_>,
    path: &str,
    momenta: Vec<Vec<String>>,
    scales: Vec<String>,
    process: Option<&str>,
    parameters: Option<BTreeMap<String, String>>,
    poles: Option<Bound<'_, PyAny>>,
    threads: Option<usize>,
) -> PyResult<Vec<V2ArtifactPoint>> {
    use rusticol_loop_reduction::v2_runtime::{V2Poles, V2Runtime};

    let poles = match poles {
        None => V2Poles::All,
        Some(value) => match value.extract::<bool>() {
            Ok(true) => V2Poles::All,
            Ok(false) => V2Poles::None,
            Err(_) => match value.extract::<String>()?.as_str() {
                "all" => V2Poles::All,
                "none" => V2Poles::None,
                "uv" => V2Poles::Uv,
                "ir" => V2Poles::Ir,
                other => {
                    return Err(error(format!(
                        "poles must be \"all\", \"none\", \"uv\" or \"ir\", not {other:?}"
                    )));
                }
            },
        },
    };
    let momenta = momenta
        .into_iter()
        .map(rationals)
        .collect::<PyResult<Vec<_>>>()?;
    let scales = rationals(scales)?;
    let points = py
        .detach(|| {
            let mut runtime = V2Runtime::open(path, process, threads)?;
            if let Some(values) = &parameters {
                runtime.set_parameters(values.iter().map(|(k, v)| (k.as_str(), v.as_str())))?;
            }
            runtime.evaluate(&momenta, &scales, poles)
        })
        .map_err(error)?;
    Ok(points
        .into_iter()
        .map(|point| match point {
            Ok(values) => (Some(values), Vec::new()),
            Err(failures) => (None, failures),
        })
        .collect())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<V2Executor>()?;
    module.add_function(wrap_pyfunction!(compile_v2_plane, module)?)?;
    module.add(
        "PlaneMismatchError",
        module.py().get_type::<PlaneMismatchError>(),
    )?;
    module.add_function(wrap_pyfunction!(native_library_key_active, module)?)?;
    module.add_function(wrap_pyfunction!(admit_v2_artifact, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_v2_artifact, module)?)
}
