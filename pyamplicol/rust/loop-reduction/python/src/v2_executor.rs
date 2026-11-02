// SPDX-License-Identifier: 0BSD

//! Python ownership of the native v2 executor (`loops/v2/plan_export.py`).
//!
//! A `V2Executor` loads one plan v4 for one carrier: `double` (binary64,
//! NinjaRS fast engine), `double-double` and `arbitrary` (decimal digits;
//! the generic engine), or `exact` (rational numerators only). Batch inputs
//! are a dict of exact momenta, scales, squared masses (an `[re, im]` pair
//! for a complex-mass scheme's pole mass) and the legs' squared masses
//! (`leg_masses`; decimal or fraction strings) and carrier
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
    Contraction, PointNumerators, PointResult, Report, Schedule, V2Executor as Executor, V2Options,
};
use crate::v2_colour::NativeDipoleCorrelator;
use rusticol_core::loop_v2::dipole_orbits::Correlator;
use rusticol_core::loop_v2::dress::{Carrier as V2Carrier, V2Batch, parse_rational};
use rusticol_core::loop_v2::plan::{FormPolicy, LoopProgramV4};
use rusticol_core::loop_v2::retest::ReductionCheck;
use rusticol_core::loop_v2::trees::{
    BoundParameters, TREE_FILL_BUDGET, TreeCarrier, TreeInputs, TreeProgram, TreeRuntime,
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

/// A batch's squared plan masses (`inputs["masses"]`): per mass id an exact
/// text, or an `[re, im]` pair of them (a complex-mass scheme's pole mass);
/// returns the real parts and the imaginary parts (empty when every entry is
/// a text, `V2Batch::masses_imaginary`).
fn squared_masses(value: &Bound<'_, PyAny>) -> PyResult<(Vec<Rational>, Vec<Rational>)> {
    let mut real = Vec::new();
    let mut imaginary = Vec::new();
    let mut complex = false;
    for item in value.try_iter()? {
        let item = item?;
        if let Ok(text) = item.extract::<String>() {
            real.push(parse_rational(&text).map_err(error)?);
            imaginary.push(Rational::from(0));
        } else {
            let [re, im]: [String; 2] = item.extract()?;
            real.push(parse_rational(&re).map_err(error)?);
            imaginary.push(parse_rational(&im).map_err(error)?);
            complex = true;
        }
    }
    if !complex {
        imaginary.clear();
    }
    Ok((real, imaginary))
}

fn batch<T: Exchanged>(inputs: &Bound<'_, PyDict>, zero: &T) -> PyResult<V2Batch<T>> {
    let native = inputs.contains("parameters")?;
    // Bound natively from the external parameters (`Trees::fill`).
    let (masses, masses_imaginary) = if native {
        (Vec::new(), Vec::new())
    } else {
        squared_masses(&item(inputs, "masses")?)?
    };
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
        masses,
        masses_imaginary,
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
    /// Shared by the executors `V2Executor::derive` makes.
    program: Arc<TreeProgram>,
    /// The program in the executor's carrier (a `TreeRuntime<T>`), built on
    /// first use; the executor is shared with Python threads, hence the lock
    /// (and executors of one carrier derived from each other share it).
    runtime: Arc<Mutex<Option<Box<dyn Any + Send>>>>,
}

/// What a batch's tree fills share (`Trees::prepare`): the bound
/// parameters of every point, its spin and loop-side sources, and whether
/// they hold one entry per point (else the batch fills at once, which
/// refuses them).
struct Filling<T> {
    bound: BoundParameters<T>,
    sources: Option<Vec<rusticol_core::loop_v2::trees::PointSources<T>>>,
    loop_sources: Option<Vec<rusticol_core::loop_v2::trees::LoopSources<T>>>,
    points: usize,
}

impl<T: Clone> Filling<T> {
    fn cuts(&self) -> bool {
        self.bound.trees.holds(self.points)
            && self.sources.as_ref().is_none_or(|s| s.len() == self.points)
            && self
                .loop_sources
                .as_ref()
                .is_none_or(|s| s.len() == self.points)
    }
}

/// Bytes of block rows and Born components of one streamed part of a batch
/// (`evaluate_with`): the batch fills, dresses and reduces that many points
/// (whole lane chunks) at a time.
const ROW_BUDGET: usize = 3 << 30;

/// Per-call memory bounds of an evaluation (`V2Executor.evaluate`): points
/// per streamed part (rounded up to whole lane chunks) and per tree fill;
/// None: from [`ROW_BUDGET`] and [`TREE_FILL_BUDGET`].
#[derive(Clone, Copy, Default)]
struct Bounds {
    part_points: Option<usize>,
    fill_points: Option<usize>,
}

impl Trees {
    /// The batch's bound parameters (`TreeProgram::bind`) and the carrier's
    /// tree runtime, built on first use.
    fn bind<'a, T: Exchanged + Real + TreeCarrier>(
        program: &TreeProgram,
        guard: &'a mut Option<Box<dyn Any + Send>>,
        values: &[Rational],
        scales: &[Rational],
        zero: &T,
    ) -> PyResult<(&'a mut TreeRuntime<T>, BoundParameters<T>)>
    where
        Complex<T>: Real + symbolica::evaluate::EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        let fresh = !guard
            .as_ref()
            .is_some_and(|runtime| runtime.is::<TreeRuntime<T>>());
        if fresh {
            *guard = Some(Box::new(program.runtime(zero).map_err(error)?));
        }
        let runtime = guard
            .as_mut()
            .and_then(|runtime| runtime.downcast_mut::<TreeRuntime<T>>())
            .expect("a runtime of this carrier");
        let bound = program
            .bind(runtime, values, scales, zero)
            .map_err(|failure| {
                if rusticol_core::loop_v2::trees::is_nonfinite_counterterm(&failure) {
                    NonFiniteCountertermError::new_err(failure.to_string())
                } else {
                    error(failure)
                }
            })?;
        Ok((runtime, bound))
    }

    /// The batch's exact external parameter values (`parameters`).
    fn values(inputs: &Bound<'_, PyDict>) -> PyResult<Vec<Rational>> {
        rationals(item(inputs, "parameters")?.extract::<Vec<String>>()?)
    }

    /// Native block rows and Born components of a batch (filled into
    /// `batch`); returns every other input for inspection. The binding and
    /// the tree and closure stages run on `executor`'s pool (its `threads`
    /// bound), not on the global one, as the SDK runtime's
    /// (`v2_runtime::V2Runtime`).
    fn fill<T: Exchanged + Real + TreeCarrier + Send + Sync>(
        &mut self,
        executor: &Executor,
        inputs: &Bound<'_, PyDict>,
        batch: &mut V2Batch<T>,
        poles: bool,
        zero: &T,
    ) -> PyResult<TreeInputs<T>>
    where
        Complex<T>: Real + symbolica::evaluate::EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        let filling = self.prepare(executor, inputs, &batch.scales, zero)?;
        let points = batch.momenta.len();
        self.fill_part(executor, &filling, 0..points, batch, poles, None, zero)
    }

    /// Bind a batch (`scales`: every point's) and read its sources.
    fn prepare<T: Exchanged + Real + TreeCarrier + Send + Sync>(
        &self,
        executor: &Executor,
        inputs: &Bound<'_, PyDict>,
        scales: &[Rational],
        zero: &T,
    ) -> PyResult<Filling<T>>
    where
        Complex<T>: Real + symbolica::evaluate::EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        let values = Self::values(inputs)?;
        let sources = match inputs.get_item("sources")? {
            Some(value) if !value.is_none() => Some(point_sources(&value, zero)?),
            _ => None,
        };
        let loop_sources = match inputs.get_item("loop_sources")? {
            Some(value) if !value.is_none() => Some(loop_sources(&value, zero)?),
            _ => None,
        };
        let mut guard = self
            .runtime
            .lock()
            .map_err(|_| error("a poisoned tree runtime"))?;
        let slot: &mut Option<Box<dyn Any + Send>> = &mut guard;
        let program = &*self.program;
        let bound = executor.install(|| -> PyResult<_> {
            Ok(Self::bind(program, slot, &values, scales, zero)?.1)
        })?;
        Ok(Filling {
            bound,
            sources,
            loop_sources,
            points: scales.len(),
        })
    }

    /// The tree fill of the points `range` of a prepared batch into `batch`
    /// (a batch of those points: momenta and scales), `fill` points at a
    /// time (None: `TreeProgram::fill_points` within its budget). The block
    /// rows and Born components move into `batch`; the other inputs are
    /// returned.
    #[allow(clippy::too_many_arguments)]
    fn fill_part<T: Exchanged + Real + TreeCarrier + Send + Sync>(
        &self,
        executor: &Executor,
        filling: &Filling<T>,
        range: std::ops::Range<usize>,
        batch: &mut V2Batch<T>,
        poles: bool,
        fill: Option<usize>,
        zero: &T,
    ) -> PyResult<TreeInputs<T>>
    where
        Complex<T>: Real + symbolica::evaluate::EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        let whole = range.len() == filling.points;
        if !whole && !filling.cuts() {
            return Err(error(
                "a batch with per-point inputs of another length fills at once",
            ));
        }
        let mut guard = self
            .runtime
            .lock()
            .map_err(|_| error("a poisoned tree runtime"))?;
        let runtime = guard
            .as_mut()
            .and_then(|runtime| runtime.downcast_mut::<TreeRuntime<T>>())
            .ok_or_else(|| error("the tree runtime of another carrier"))?;
        let program = &*self.program;
        let selected;
        let trees = if whole {
            &filling.bound.trees
        } else {
            selected = filling.bound.trees.select(range.clone());
            &selected
        };
        let sources = filling
            .sources
            .as_deref()
            .map(|all| if whole { all } else { &all[range.clone()] });
        let loop_sources = filling
            .loop_sources
            .as_deref()
            .map(|all| if whole { all } else { &all[range.clone()] });
        let momenta = &batch.momenta;
        let constants = &filling.bound.constants;
        let mut out = executor.install(|| -> PyResult<_> {
            let chunk = fill.unwrap_or_else(|| {
                program.fill_points::<T>(momenta.len(), loop_sources.is_some(), TREE_FILL_BUDGET)
            });
            program
                .inputs_chunked(
                    runtime,
                    executor.plan(),
                    trees,
                    momenta,
                    constants,
                    poles,
                    sources,
                    loop_sources,
                    chunk,
                    zero,
                )
                .map_err(error)
        })?;
        batch.constants = filling.bound.constants.clone();
        batch.masses = filling.bound.masses.clone();
        batch.masses_imaginary = filling.bound.masses_imaginary.clone();
        batch.leg_masses = filling.bound.leg_masses.clone();
        batch.block_rows = std::mem::take(&mut out.block_rows);
        batch.born = std::mem::take(&mut out.born);
        Ok(out)
    }

    /// Points per streamed part of a batch of `points` points on `lanes`
    /// lanes: whole lane chunks whose block rows and Born components fit in
    /// [`ROW_BUDGET`] (at least one chunk; the whole batch when it fits).
    fn part_points<T>(&self, points: usize, lanes: usize) -> usize {
        let point = self.program.row_values().max(1) * std::mem::size_of::<Complex<T>>();
        let fit = ROW_BUDGET / point / lanes.max(1) * lanes.max(1);
        if fit >= points {
            points.max(1)
        } else {
            fit.max(lanes.max(1))
        }
    }
}

/// The masks of one evaluation call (`V2Executor.evaluate`'s `rows`,
/// `basis`, `components` and `families`): an evaluation-time selection run
/// on the executor of a template column set, one program for every
/// selection (`loops/v2/native_observables.py`; DESIGN §8.5). The kept Born
/// helicity `rows` seed and interfere (every other row's Born components
/// vanish, as a spin source's fixed row does); the observable columns'
/// interferences keep the bra Born `basis` elements only; the seeds' bra
/// keeps the plan Born `components` only (the others read zero); and only
/// the plan `families` kept are dressed (the caller's statement that the
/// others' seeds vanish under these masks; their lanes are absent). None
/// keeps every one.
#[derive(Default)]
struct Selection {
    rows: Option<Vec<bool>>,
    basis: Option<Vec<bool>>,
    components: Option<Vec<bool>>,
    families: Option<Vec<bool>>,
}

impl Selection {
    fn is_empty(&self) -> bool {
        self.rows.is_none()
            && self.basis.is_none()
            && self.components.is_none()
            && self.families.is_none()
    }

    /// Each point's sources with the kept rows (and the point's own) and the
    /// kept basis elements; points without sources get an empty one.
    fn sources<T>(
        &self,
        sources: Option<Vec<rusticol_core::loop_v2::trees::PointSources<T>>>,
        points: usize,
    ) -> Option<Vec<rusticol_core::loop_v2::trees::PointSources<T>>> {
        if self.rows.is_none() && self.basis.is_none() {
            return sources;
        }
        let mut sources = sources.unwrap_or_else(|| {
            (0..points)
                .map(|_| rusticol_core::loop_v2::trees::PointSources {
                    vectors: Vec::new(),
                    rows: None,
                    basis: None,
                })
                .collect()
        });
        for point in &mut sources {
            if let Some(kept) = &self.rows {
                point.rows = Some(match point.rows.take() {
                    Some(own) => own.iter().zip(kept).map(|(a, b)| *a && *b).collect(),
                    None => kept.clone(),
                });
            }
            if self.basis.is_some() {
                point.basis.clone_from(&self.basis);
            }
        }
        Some(sources)
    }

    /// Zero the plan Born components the seeds' bra does not keep (`born`:
    /// per point `rows x components`).
    fn mask_born<T: Real>(&self, born: &mut [Vec<Complex<T>>], zero: &T) -> PyResult<()> {
        let Some(kept) = &self.components else {
            return Ok(());
        };
        for point in born {
            if kept.is_empty() || point.len() % kept.len() != 0 {
                return Err(error("one component flag per plan Born component"));
            }
            for row in point.chunks_exact_mut(kept.len()) {
                for (value, keep) in row.iter_mut().zip(kept) {
                    if !keep {
                        *value = Complex::new(zero.clone(), zero.clone());
                    }
                }
            }
        }
        Ok(())
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
            let vectors = source_vectors(vectors, zero)?;
            Ok(rusticol_core::loop_v2::trees::PointSources {
                vectors,
                rows,
                basis: None,
            })
        })
        .collect()
}

/// Per point loop-side sources (`inputs["loop_sources"]`, the asymmetric
/// Ward mode, `loops/v2/ward.py`): per point its (1-based leg label, four
/// `[re, im]` exact texts) pairs (empty: the point's physical trees).
fn loop_sources<T: V2Carrier>(
    value: &Bound<'_, PyAny>,
    zero: &T,
) -> PyResult<Vec<rusticol_core::loop_v2::trees::LoopSources<T>>> {
    let parsed: Vec<Vec<(usize, Vec<[String; 2]>)>> = value.extract()?;
    parsed
        .into_iter()
        .map(|vectors| source_vectors(vectors, zero))
        .collect()
}

/// Literal complex four-vectors per leg from exact `[re, im]` texts.
#[allow(clippy::type_complexity)]
fn source_vectors<T: V2Carrier>(
    vectors: Vec<(usize, Vec<[String; 2]>)>,
    zero: &T,
) -> PyResult<Vec<(usize, [Complex<T>; 4])>> {
    vectors
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
    /// With loop-side sources (`inputs["loop_sources"]`): per point and
    /// squared Born order the Born interference of the physical bra with the
    /// loop-side ket (`TreeInputs::loop_born`), else None.
    loop_born: Option<Vec<Values>>,
    /// The counterterm and external-field outputs per point (native trees
    /// with counterterms), else None.
    counterterms: Option<Vec<Values>>,
    /// Per point and observable column its complex outputs (native trees
    /// with observable columns), else None.
    columns: Option<Vec<Vec<Values>>>,
    /// With `dipoles` (native trees): per point the colour correlations of
    /// the Born vectors (`Correlator::evaluate`: per sandwich the `c × c`
    /// values, flattened), else None.
    dipoles: Option<Vec<Values>>,
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
    loop_born: Option<Vec<Values>>,
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
        loop_born,
        counterterms,
        columns,
        dipoles: None,
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

/// Add a streamed part's report to the batch's: seconds and counts add up,
/// the peak reals per lane are the largest.
fn merge_report(total: &mut Option<Report>, part: Report) {
    let Some(total) = total else {
        *total = Some(part);
        return;
    };
    for (stage, seconds) in part.seconds {
        *total.seconds.entry(stage).or_default() += seconds;
    }
    for (name, count) in part.counts {
        *total.counts.entry(name).or_default() += count;
    }
    for (name, reals) in part.reals_per_lane {
        let entry = total.reals_per_lane.entry(name).or_default();
        *entry = (*entry).max(reals);
    }
}

/// Evaluate a batch. With native trees it streams in parts of whole lane
/// chunks (`Bounds`; `Trees::part_points`): each part's tree fill, then its
/// dressing and reduction, so the batch's tree states, block rows and Born
/// components never live at once. The executor dresses lane chunks of the
/// same points in the same order, every lane alone, so each point's values
/// are bitwise those of the batch at once.
#[allow(clippy::too_many_arguments)]
fn evaluate_with<T: Exchanged + Real + TypedBatch + TreeCarrier + Send + Sync>(
    py: Python<'_>,
    executor: &Executor,
    trees: Option<&mut Trees>,
    backend: &mut V2ReductionBackend,
    inputs: &Bound<'_, PyDict>,
    zero: T,
    bits: u32,
    options: V2Options,
    bounds: Bounds,
    selection: &Selection,
    dipoles: Option<&Correlator>,
) -> PyResult<Evaluation>
where
    Complex<T>: Real + symbolica::evaluate::EvaluationDomain,
    TreeRuntime<T>: Send,
{
    let batch = batch(inputs, &zero)?;
    if !inputs.contains("parameters")? {
        if dipoles.is_some() {
            return Err(error(
                "dipole correlations need the executor's tree program (native inputs)",
            ));
        }
        if !selection.is_empty() {
            return Err(error(
                "selection masks need the executor's tree program (native inputs)",
            ));
        }
        let (results, report) = py
            .detach(|| executor.evaluate(&batch, &zero, backend, options))
            .map_err(error)?;
        return Ok(export_evaluation(
            results, report, bits, None, None, None, None,
        ));
    }
    let trees = trees.ok_or_else(|| error("the executor has no tree program"))?;
    // The counterterm pole outputs are formed unless finite-only (the
    // split modes read the same per-power outputs).
    let poles = !matches!(
        options.poles,
        rusticol_core::loop_execution::v2::Poles::None
    );
    let started = std::time::Instant::now();
    let mut filling = trees.prepare(executor, inputs, &batch.scales, &zero)?;
    filling.sources = selection.sources(filling.sources.take(), batch.momenta.len());
    // The report's "trees" stage: the whole native input fill, then its
    // parts (`TreeInputs::seconds`).
    let mut tree_seconds = started.elapsed().as_secs_f64();
    let mut tree_parts: Vec<(&'static str, f64)> = Vec::new();
    let points = batch.momenta.len();
    let lanes = executor.lanes().max(1);
    let part = match bounds.part_points {
        _ if !filling.cuts() => points,
        Some(part) => part.max(1).div_ceil(lanes) * lanes,
        None => trees.part_points::<T>(points, lanes),
    };
    let export = |points: Vec<Vec<T>>| -> Vec<Values> {
        points
            .iter()
            .map(|values| real_values(values, &zero, bits))
            .collect()
    };
    let extend = |total: &mut Option<Vec<Values>>, part: Option<Vec<Values>>| {
        if let Some(part) = part {
            total.get_or_insert_with(Vec::new).extend(part);
        }
    };
    let (mut born, mut loop_born, mut counterterms) = (None, None, None);
    let mut columns: Option<Vec<Vec<Values>>> = None;
    let mut correlations: Option<Vec<Values>> = None;
    let mut results = Vec::with_capacity(points);
    let mut report = None;
    for first in (0..points.max(1)).step_by(part.max(1)) {
        let range = first..(first + part).min(points);
        let mut sub = V2Batch {
            momenta: batch.momenta[range.clone()].to_vec(),
            scales: batch.scales[range.clone()].to_vec(),
            masses: Vec::new(),
            masses_imaginary: Vec::new(),
            leg_masses: Vec::new(),
            constants: Vec::new(),
            block_rows: Vec::new(),
            born: Vec::new(),
        };
        let started = std::time::Instant::now();
        let native = trees.fill_part(
            executor,
            &filling,
            range,
            &mut sub,
            poles,
            bounds.fill_points,
            &zero,
        )?;
        tree_seconds += started.elapsed().as_secs_f64();
        for (stage, seconds) in &native.seconds {
            match tree_parts.iter_mut().find(|(name, _)| name == stage) {
                Some((_, total)) => *total += seconds,
                None => tree_parts.push((stage, *seconds)),
            }
        }
        extend(&mut born, native.born_squared.map(export));
        extend(&mut loop_born, native.loop_born.map(export));
        extend(&mut counterterms, native.counterterms.map(export));
        if let Some(correlator) = dipoles {
            // The Born vectors' colour correlations (M-FINAL MF-8G-3), in
            // the report's "trees: dipoles" stage.
            let started = std::time::Instant::now();
            let values = executor
                .install(|| correlator.evaluate(&native.amplitudes, &zero))
                .map_err(error)?;
            extend(
                &mut correlations,
                Some(
                    values
                        .into_iter()
                        .map(|sandwiches| sandwiches.concat())
                        .collect(),
                )
                .map(export),
            );
            let seconds = started.elapsed().as_secs_f64();
            tree_seconds += seconds;
            match tree_parts.iter_mut().find(|(name, _)| *name == "trees: dipoles") {
                Some((_, total)) => *total += seconds,
                None => tree_parts.push(("trees: dipoles", seconds)),
            }
        }
        if let Some(part) = native.columns {
            columns
                .get_or_insert_with(Vec::new)
                .extend(part.iter().map(|columns| {
                    columns
                        .iter()
                        .map(|values| T::export(values, bits))
                        .collect::<Vec<_>>()
                }));
        }
        selection.mask_born(&mut sub.born, &zero)?;
        let families = selection.families.as_deref();
        let (part_results, part_report) = py
            .detach(|| executor.evaluate_families(&sub, &zero, backend, options, families))
            .map_err(error)?;
        results.extend(part_results);
        merge_report(&mut report, part_report);
    }
    let mut report = report.expect("a batch has a part");
    report.seconds.insert("trees", tree_seconds);
    report.seconds.extend(tree_parts);
    let mut evaluation = export_evaluation(
        results,
        report,
        bits,
        born,
        loop_born,
        counterterms,
        columns,
    );
    evaluation.dipoles = correlations;
    Ok(evaluation)
}

#[allow(clippy::too_many_arguments)]
fn seed_values_with<T: Exchanged + Real + TreeCarrier + Send + Sync>(
    py: Python<'_>,
    executor: &Executor,
    trees: Option<&mut Trees>,
    inputs: &Bound<'_, PyDict>,
    zero: T,
    bits: u32,
    contraction: Contraction,
) -> PyResult<Vec<Vec<Values>>>
where
    Complex<T>: Real + symbolica::evaluate::EvaluationDomain,
    TreeRuntime<T>: Send,
{
    let mut batch = batch(inputs, &zero)?;
    if inputs.contains("parameters")? {
        let trees = trees.ok_or_else(|| error("the executor has no tree program"))?;
        trees.fill(executor, inputs, &mut batch, false, &zero)?;
    }
    let values = py
        .detach(|| executor.seed_values(&batch, &zero, contraction))
        .map_err(error)?;
    Ok(values
        .iter()
        .map(|keys| keys.iter().map(|key| T::export(key, bits)).collect())
        .collect())
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

fn tree_inputs_with<T: Exchanged + Real + TreeCarrier + Send + Sync>(
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
    let out = trees.fill(executor, inputs, &mut batch, true, &zero)?;
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
        export(&batch.block_rows),
        export(&batch.born),
        export(&out.amplitudes),
        out.born_squared.map(reals),
        out.counterterms.map(reals),
    ))
}

/// The observable columns' native interferences of a batch, per point and
/// column (`evaluate`'s `columns`, with poles), or None without columns.
type ExportedColumns = Option<Vec<Vec<Values>>>;

fn tree_columns_with<T: Exchanged + Real + TreeCarrier + Send + Sync>(
    executor: &Executor,
    trees: &mut Trees,
    inputs: &Bound<'_, PyDict>,
    zero: T,
    bits: u32,
    selection: &Selection,
) -> PyResult<ExportedColumns>
where
    Complex<T>: Real + symbolica::evaluate::EvaluationDomain,
    TreeRuntime<T>: Send,
{
    let mut batch = batch(inputs, &zero)?;
    let points = batch.momenta.len();
    let mut filling = trees.prepare(executor, inputs, &batch.scales, &zero)?;
    filling.sources = selection.sources(filling.sources.take(), points);
    let out = trees.fill_part(executor, &filling, 0..points, &mut batch, true, None, &zero)?;
    Ok(out.columns.map(|points| {
        points
            .iter()
            .map(|columns| {
                columns
                    .iter()
                    .map(|values| T::export(values, bits))
                    .collect()
            })
            .collect()
    }))
}

/// Per plan mass and per leg, the bound squared masses (exact fractions),
/// and the imaginary parts of the plan's (empty: every plan mass real).
type BoundMasses = (Vec<String>, Vec<Option<String>>, Vec<String>);

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
    let values = Trees::values(inputs)?;
    let (_runtime, bound) = Trees::bind(&trees.program, &mut guard, &values, scales, zero)?;
    Ok((
        bound.masses.iter().map(ToString::to_string).collect(),
        bound
            .leg_masses
            .iter()
            .map(|mass| mass.as_ref().map(ToString::to_string))
            .collect(),
        bound
            .masses_imaginary
            .iter()
            .map(ToString::to_string)
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
    /// `plan` is the plan's binary member (bytes) or its JSON (text).
    /// `trees` is a tree program (`loops/v2/native_trees.py`): its JSON,
    /// the saved exact evaluator of each kernel and of the parameter program
    /// (last), and their binary64 SymJIT plane applications in the same order
    /// (or None: binary64 then runs the exact states).
    /// `columns` (JSON, `loops/v2/native_observables`) adds observable
    /// columns to the tree program and `plan_components` replaces its plan
    /// Born components (an observable column set's plan).
    #[new]
    #[pyo3(signature = (plan, arithmetic="double", precision=None, *, lanes=16, threads=None, forms="rule", engine="fast", trees=None, columns=None, plan_components=None, fourier=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        plan: Bound<'_, PyAny>,
        arithmetic: &str,
        precision: Option<u32>,
        lanes: usize,
        threads: Option<usize>,
        forms: &str,
        engine: &str,
        trees: Option<(Bound<'_, PyAny>, Vec<Vec<u8>>, Option<Vec<Vec<u8>>>)>,
        columns: Option<String>,
        plan_components: Option<Vec<Option<usize>>>,
        fourier: Option<String>,
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
        // The stored plan: its binary member as bytes (M-FINAL MF-8G-1 item
        // 5) or a plan JSON as text; `load_with` reads either.
        let plan = {
            let data: &[u8] = match plan.cast::<pyo3::types::PyBytes>() {
                Ok(bytes) => bytes.as_bytes(),
                Err(_) => plan.extract::<&str>().map_err(error)?.as_bytes(),
            };
            Arc::new(LoopProgramV4::load_with(data, policy).map_err(error)?)
        };
        // The program's JSON text, or a binary tree program as bytes
        // (`loop_v2::trees::binary`, M-FINAL MF-8G-1).
        let trees = trees
            .map(|(source, kernels, planes)| {
                let program = match source.cast::<pyo3::types::PyBytes>() {
                    Ok(bytes) => bytes.as_bytes().to_vec(),
                    Err(_) => source.extract::<String>().map_err(error)?.into_bytes(),
                };
                Ok::<_, PyErr>((program, kernels, planes))
            })
            .transpose()?;
        let trees = trees
            .map(|(json, kernels, planes)| {
                let mut program = TreeProgram::load_with_components(
                    &json,
                    &kernels,
                    planes.as_deref(),
                    &plan,
                    plan_components.clone(),
                )?;
                if let Some(columns) = &columns {
                    program.set_columns(columns.as_bytes())?;
                }
                Ok::<_, rusticol_core::RusticolError>(Trees {
                    program: Arc::new(program),
                    runtime: Arc::new(Mutex::new(None)),
                })
            })
            .transpose()
            .map_err(error)?;
        if trees.is_none() && (columns.is_some() || plan_components.is_some()) {
            return Err(error("observable columns need a tree program"));
        }
        let mut executor = Executor::new(plan, lanes, threads).map_err(error)?;
        if let Some(record) = fourier {
            // The seed FFT needs square roots: the exact carrier contracts
            // directly (`loop_v2::fourier`).
            if matches!(carrier, Carriers::Exact) {
                return Err(error("the exact carrier contracts seeds directly"));
            }
            executor = executor.with_fourier(record.as_bytes()).map_err(error)?;
        }
        Ok(Self {
            executor,
            carrier,
            engine: engine.to_owned(),
            backend: None,
            trees,
        })
    }

    /// An executor of `arithmetic` (and `precision`, as for the
    /// constructor) on this one's loaded plan, tree program (with `trees`)
    /// and certified seed FFT (with `fourier`), shared instead of decoded
    /// again: the auxiliary carriers and lane counts of one process (double-
    /// double and arbitrary precision, the rescue levels) cost their own
    /// runtimes only. An executor of the same carrier also shares the tree
    /// runtime; the kernel forms are this one's.
    #[pyo3(signature = (arithmetic="double", precision=None, *, lanes=16, threads=None, engine="fast", trees=true, fourier=true))]
    #[allow(clippy::too_many_arguments)]
    fn derive(
        &self,
        arithmetic: &str,
        precision: Option<u32>,
        lanes: usize,
        threads: Option<usize>,
        engine: &str,
        trees: bool,
        fourier: bool,
    ) -> PyResult<Self> {
        if !matches!(engine, "fast" | "generic") {
            return Err(error("engine must be fast or generic"));
        }
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
        if trees && self.trees.is_none() {
            return Err(error("the executor has no tree program to share"));
        }
        if fourier && self.executor.fourier().is_none() {
            return Err(error("the executor has no seed Fourier record to share"));
        }
        if fourier && matches!(carrier, Carriers::Exact) {
            return Err(error("the exact carrier contracts seeds directly"));
        }
        let same = std::mem::discriminant(&carrier) == std::mem::discriminant(&self.carrier)
            && match (carrier, self.carrier) {
                (Carriers::Arbitrary(a), Carriers::Arbitrary(b)) => a == b,
                _ => true,
            };
        let trees = match &self.trees {
            Some(shared) if trees => Some(Trees {
                program: shared.program.clone(),
                runtime: if same {
                    shared.runtime.clone()
                } else {
                    Arc::new(Mutex::new(None))
                },
            }),
            _ => None,
        };
        Ok(Self {
            executor: self
                .executor
                .derive(lanes, threads, fourier)
                .map_err(error)?,
            carrier,
            engine: engine.to_owned(),
            backend: None,
            trees,
        })
    }

    /// Whether the executor has a tree program (native inputs).
    #[getter]
    fn has_trees(&self) -> bool {
        self.trees.is_some()
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

    /// The observable columns' native interferences of a batch (needs a
    /// tree program; `inputs` as for `tree_inputs`): per point and column
    /// what `evaluate` returns in `columns` (with poles), without dressing
    /// or reducing the loops; None when the program has no columns. The
    /// split selections of a model with pole regulators read them at the IR
    /// regulator 0 (`native_observables.StoredColumns`). `rows` and `basis`
    /// are `evaluate`'s selection masks.
    #[pyo3(signature = (inputs, *, rows=None, basis=None))]
    fn tree_columns(
        &mut self,
        inputs: &Bound<'_, PyDict>,
        rows: Option<Vec<bool>>,
        basis: Option<Vec<bool>>,
    ) -> PyResult<ExportedColumns> {
        let bits = self.bits()?;
        let executor = &self.executor;
        let trees = self
            .trees
            .as_mut()
            .ok_or_else(|| error("the executor has no tree program"))?;
        let selection = Selection {
            rows,
            basis,
            ..Selection::default()
        };
        let selection = &selection;
        match self.carrier {
            Carriers::Double => tree_columns_with(executor, trees, inputs, 0.0, bits, selection),
            Carriers::DoubleDouble => tree_columns_with(
                executor,
                trees,
                inputs,
                DoubleFloat::from(0.0),
                bits,
                selection,
            ),
            Carriers::Arbitrary(_) => tree_columns_with(
                executor,
                trees,
                inputs,
                Float::with_val(bits, 0),
                bits,
                selection,
            ),
            Carriers::Exact => Err(error("the exact carrier has no native trees")),
        }
    }

    /// The squared masses a batch's external parameters bind natively
    /// (needs a tree program; `inputs` as for `tree_inputs`): per plan mass
    /// and per leg (the on-shell invariants, DESIGN §11.1), exact fractions,
    /// and the plan masses' imaginary parts (empty: every plan mass real).
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

    /// The certified seed FFT (`fourier=`): `{"jobs", "pairs", "covered",
    /// "group_order"}`, or None when every seed is contracted directly.
    #[getter]
    fn fourier(&self) -> Option<BTreeMap<&'static str, usize>> {
        self.executor.fourier().map(|fourier| {
            BTreeMap::from([
                ("jobs", fourier.jobs()),
                ("pairs", fourier.pairs()),
                ("covered", fourier.covered()),
                ("group_order", fourier.group_order()),
            ])
        })
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

    /// The seeds `U0` per point and seed key (`rows × columns` complex
    /// values in object order), contracted as `evaluate` with `contraction`
    /// contracts them: the oracle of the certified seed FFT (DESIGN.md §7.3:
    /// FFT = direct to the contraction's own rounding). `inputs` as for
    /// `evaluate` (with `parameters`, the native trees fill the Born rows).
    #[pyo3(signature = (inputs, *, contraction="auto"))]
    fn seed_values(
        &mut self,
        py: Python<'_>,
        inputs: &Bound<'_, PyDict>,
        contraction: &str,
    ) -> PyResult<Vec<Vec<Values>>> {
        let contraction = match contraction {
            "auto" => Contraction::Auto,
            "direct" => Contraction::Direct,
            "fft" => Contraction::Fft,
            _ => return Err(error("contraction must be auto, direct or fft")),
        };
        let bits = self.bits()?;
        let executor = &self.executor;
        let trees = self.trees.as_mut();
        match self.carrier {
            Carriers::Double => {
                seed_values_with(py, executor, trees, inputs, 0.0, bits, contraction)
            }
            Carriers::DoubleDouble => seed_values_with(
                py,
                executor,
                trees,
                inputs,
                DoubleFloat::from(0.0),
                bits,
                contraction,
            ),
            Carriers::Arbitrary(_) => seed_values_with(
                py,
                executor,
                trees,
                inputs,
                Float::with_val(bits, 0),
                bits,
                contraction,
            ),
            Carriers::Exact => Err(error("the exact carrier contracts seeds directly")),
        }
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
    /// Memory bounds (values bitwise unchanged): `part_points` points per
    /// streamed part of a batch with native trees (rounded up to whole lane
    /// chunks; None: block rows and Born components within 3 GiB),
    /// `fill_points` points per tree fill (None: tree states within 4 GiB)
    /// and `input_budget` bytes of chunk inputs per lane chunk (None: 8 GiB).
    /// Selection masks (native trees only; None keeps every one; `Selection`):
    /// `rows` the kept Born helicity rows, `basis` the observable columns'
    /// kept bra Born basis elements, `components` the seeds' kept plan Born
    /// components, `families` the plan families dressed (the others' lanes
    /// are absent): one executor serves every evaluation-time selection.
    /// `dipoles` (a `NativeDipoleCorrelator`, native trees only) adds per
    /// point the Born vectors' colour correlations contracted over the
    /// stored dipole orbits (`"dipoles"`, the native Catani operator's
    /// input; report stage "trees: dipoles").
    #[pyo3(signature = (inputs, *, poles=None, check=false, reverse=true, rotation=1, shift=false, rescale=false, chunk_families=None, schedule="B", part_chunks=None, master_poles="both", contraction="auto", part_points=None, fill_points=None, input_budget=None, rows=None, basis=None, components=None, families=None, dipoles=None))]
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
        contraction: &str,
        part_points: Option<usize>,
        fill_points: Option<usize>,
        input_budget: Option<usize>,
        rows: Option<Vec<bool>>,
        basis: Option<Vec<bool>>,
        components: Option<Vec<bool>>,
        families: Option<Vec<bool>>,
        dipoles: Option<PyRef<'_, NativeDipoleCorrelator>>,
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
            contraction: match contraction {
                "auto" => Contraction::Auto,
                "direct" => Contraction::Direct,
                "fft" => Contraction::Fft,
                _ => return Err(error("contraction must be auto, direct or fft")),
            },
            ..V2Options::default()
        };
        if let Some(chunk) = chunk_families {
            options.chunk_families = chunk;
        }
        if let Some(chunks) = part_chunks {
            options.part_chunks = chunks;
        }
        if let Some(budget) = input_budget {
            options.input_budget = budget;
        }
        let bounds = Bounds {
            part_points,
            fill_points,
        };
        let selection = Selection {
            rows,
            basis,
            components,
            families,
        };
        let plan = self.executor.plan();
        if selection
            .rows
            .as_ref()
            .is_some_and(|kept| kept.len() != plan.rows)
            || selection
                .components
                .as_ref()
                .is_some_and(|kept| kept.len() != plan.born_components)
        {
            return Err(error(
                "selection masks: one flag per plan Born row and per plan Born component",
            ));
        }
        let selection = &selection;
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
        let dipoles = dipoles.as_ref().map(|d| &*d.correlator);
        match carrier {
            Carriers::Double => evaluate_with(
                py, executor, trees, backend, inputs, 0.0, bits, options, bounds, selection,
                dipoles,
            ),
            Carriers::DoubleDouble => evaluate_with(
                py,
                executor,
                trees,
                backend,
                inputs,
                DoubleFloat::from(0.0),
                bits,
                options,
                bounds,
                selection,
                dipoles,
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
                bounds,
                selection,
                dipoles,
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
     check's probes, or whose real-axis value no real point certifies \
     (`compile_v2_plane`). Its arguments are the message, the output index, \
     the probe row (None without a real point), whether the imaginary part \
     disagrees (None without a real point) and whether the row is one of the \
     caller's real points."
);

pyo3::create_exception!(
    _loop_reduction,
    NonFiniteCountertermError,
    pyo3::exceptions::PyValueError,
    "A batch the tree program's parameter binding refuses because a \
     counterterm coupling's diagnostic is not finite at some point's mu_R^2 \
     (`rusticol_core::loop_v2::trees::nonfinite_counterterm`; e.g. mu_r2 = \
     1e-400 in binary64): it depends on the scale, so the Python runtimes \
     re-run the batch per scale (`runtime.virtual.isolate_scale_failures`). \
     A `ValueError`; its argument is the message."
);

/// Generation: the binary64 SymJIT plane application of a saved exact
/// evaluator of a tree program (a kernel or the parameter program), checked
/// against the evaluator (`rusticol_core::loop_v2::trees::compile_plane_
/// application_checked`) off the real axis and at `real_points` (rows of real
/// input values at which the program runs): `(application, inputs,
/// outputs)`. A disagreement, or an output whose real-axis value needs a real
/// point when none is given, raises `PlaneMismatchError(message, output, row,
/// imaginary, real_axis)`, a `ValueError` (`native_trees._plane_applications`
/// refuses the program naming the output); any other failure a plain
/// `ValueError`.
#[pyfunction]
#[pyo3(signature = (state, optimization_level=2, compress=false, real_points=None))]
fn compile_v2_plane<'py>(
    py: Python<'py>,
    state: &[u8],
    optimization_level: u8,
    compress: bool,
    real_points: Option<Vec<Vec<f64>>>,
) -> PyResult<(Bound<'py, pyo3::types::PyBytes>, usize, usize)> {
    let real_points = real_points.unwrap_or_default();
    let checked = py
        .detach(|| {
            rusticol_core::loop_v2::trees::compile_plane_application_checked(
                state,
                optimization_level,
                compress,
                &real_points,
            )
        })
        .map_err(error)?;
    match checked {
        Ok((plane, inputs, outputs)) => {
            Ok((pyo3::types::PyBytes::new(py, &plane), inputs, outputs))
        }
        Err(mismatch) => {
            use rusticol_core::loop_v2::trees::PlaneProbe;
            let (row, imaginary, real_axis) = match &mismatch.probe {
                PlaneProbe::Row {
                    row,
                    imaginary,
                    real_axis,
                    ..
                } => (Some(*row), Some(*imaginary), *real_axis),
                PlaneProbe::NoRealPoint => (None, None, true),
            };
            Err(PlaneMismatchError::new_err((
                format!("loop v2 trees: {mismatch}"),
                mismatch.output,
                row,
                imaginary,
                real_axis,
            )))
        }
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
    module.add(
        "NonFiniteCountertermError",
        module.py().get_type::<NonFiniteCountertermError>(),
    )?;
    module.add_function(wrap_pyfunction!(native_library_key_active, module)?)?;
    module.add_function(wrap_pyfunction!(admit_v2_artifact, module)?)?;
    module.add_function(wrap_pyfunction!(evaluate_v2_artifact, module)?)
}
