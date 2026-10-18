// SPDX-License-Identifier: 0BSD

//! Experimental native consumer, separate until NinjaRS is self-contained.
//! Results are raw integrals, never Born/loop interference observables.

mod batch;
mod exact;
mod fast;
mod portable;
mod stability;
mod states;
use batch::{Data, NativeBatch};
use fast::FastReductions;

use pyo3::{exceptions::PyValueError, prelude::*};
use rusticol_loop_reduction::{
    AmplitudeInput, Complex, EvaluationStatus, FourMomentum, Lane, Laurent, PolynomialReducer,
    SamplingScaleProvider, SamplingScaleSource, SamplingScaleStatus, SymbolicaSamplingScales,
};
use symbolica::{
    atom::Atom,
    domains::float::{DoubleFloat, Float, Real},
    domains::rational::Rational,
    evaluate::EvaluationDomain,
};

use rusticol_loop_reduction::runtime::{
    self, Arithmetic, Carrier, DecimalComplex, decimal_complex,
};
type DecimalMomentum = [DecimalComplex; 4];
type RawLane = (u64, String, Option<[DecimalComplex; 3]>);

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}

fn arithmetic_mode(arithmetic: &str, precision: Option<u32>) -> PyResult<(Arithmetic, u32)> {
    runtime::arithmetic_mode(arithmetic, precision).map_err(error)
}
fn parse_complex<T: Carrier>(value: &DecimalComplex, bits: u32) -> PyResult<Complex<T>> {
    runtime::parse_complex(value, bits).map_err(error)
}

/// Python ownership of the shared native saved evaluator.
#[pyclass]
struct CoefficientEvaluator(runtime::CoefficientEvaluator);
#[pymethods]
impl CoefficientEvaluator {
    #[new]
    #[pyo3(signature = (state, arithmetic="double", precision=None))]
    fn new(state: &[u8], arithmetic: &str, precision: Option<u32>) -> PyResult<Self> {
        runtime::CoefficientEvaluator::new(state, arithmetic, precision)
            .map(Self)
            .map_err(error)
    }
    #[getter]
    fn precision_bits(&self) -> u32 {
        self.0.precision_bits()
    }
    fn evaluate_decimal(
        &mut self,
        py: Python<'_>,
        values: Vec<DecimalComplex>,
        rows: usize,
    ) -> PyResult<Vec<DecimalComplex>> {
        py.detach(|| self.0.evaluate_decimal(values, rows))
            .map_err(error)
    }
    fn evaluate_batch(&mut self, py: Python<'_>, input: &NativeBatch) -> PyResult<NativeBatch> {
        py.detach(|| self.0.evaluate_batch(&input.0))
            .map(NativeBatch)
            .map_err(error)
    }
    fn evaluate_f64(
        &mut self,
        py: Python<'_>,
        values: Vec<[f64; 2]>,
        rows: usize,
    ) -> PyResult<Vec<[f64; 2]>> {
        py.detach(|| self.0.evaluate_f64(values, rows))
            .map_err(error)
    }
}

struct Reductions<T: Carrier>
where
    Complex<T>: Real + EvaluationDomain,
{
    sectors: [PolynomialReducer<T>; 3],
    active: [bool; 3],
    zero: Complex<T>,
    propagators: usize,
    parameter_count: usize,
    sampling: Option<SymbolicaSamplingScales<T>>,
    sampling_values: Vec<Complex<T>>,
    sampling_statuses: Vec<SamplingScaleStatus>,
}

impl<T: Carrier> Reductions<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn new(
        expressions: &[String; 3],
        parameters: &[String],
        n: usize,
        prototype: T,
    ) -> PyResult<Self> {
        let parse =
            |text: &str| Atom::parse(text, "pyamplicol_loop", Default::default()).map_err(error);
        let variables = [
            parse("q0")?,
            parse("q1")?,
            parse("q2")?,
            parse("q3")?,
            parse("mu2")?,
        ];
        let parameters = parameters
            .iter()
            .map(|p| parse(p))
            .collect::<PyResult<Vec<_>>>()?;
        let mut sectors = Vec::with_capacity(3);
        let mut active = [false; 3];
        for (index, expression) in expressions.iter().enumerate() {
            let expression = parse(expression)?;
            active[index] = !expression.is_zero();
            sectors.push(
                PolynomialReducer::new(
                    n,
                    expression,
                    variables.clone(),
                    parameters.clone(),
                    prototype.clone(),
                    None,
                )
                .map_err(error)?,
            );
        }
        // Still execute one zero sector for authoritative input validation.
        if !active.iter().any(|value| *value) {
            active[0] = true;
        }
        Ok(Self {
            active,
            zero: Complex::new(prototype.zero(), prototype.zero()),
            sectors: sectors.try_into().ok().unwrap(),
            propagators: n,
            parameter_count: parameters.len(),
            sampling: if n >= 4 {
                Some(SymbolicaSamplingScales::new(n, prototype).map_err(error)?)
            } else {
                None
            },
            sampling_values: Vec::new(),
            sampling_statuses: Vec::new(),
        })
    }

    fn evaluate(
        &mut self,
        ids: &[u64],
        shifts: &[DecimalMomentum],
        masses: &[DecimalComplex],
        scales: &[DecimalComplex],
        parameters: &[DecimalComplex],
        bits: u32,
    ) -> PyResult<Vec<RawLane>> {
        let parse_all = |values: &[DecimalComplex]| {
            values
                .iter()
                .map(|v| parse_complex::<T>(v, bits))
                .collect::<PyResult<Vec<_>>>()
        };
        let shifts = shifts
            .iter()
            .map(|p| {
                Ok(FourMomentum::from([
                    parse_complex::<T>(&p[0], bits)?,
                    parse_complex::<T>(&p[1], bits)?,
                    parse_complex::<T>(&p[2], bits)?,
                    parse_complex::<T>(&p[3], bits)?,
                ]))
            })
            .collect::<PyResult<Vec<_>>>()?;
        let masses = parse_all(masses)?;
        let scales = parse_all(scales)?;
        let parameters = parse_all(parameters)?;
        Ok(self
            .evaluate_native(ids, &shifts, &masses, &scales, &parameters, None)?
            .into_iter()
            .map(|lane| {
                (
                    lane.id,
                    lane.status,
                    lane.value.map(|value| {
                        [
                            decimal_complex(value.finite, bits),
                            decimal_complex(value.single_pole, bits),
                            decimal_complex(value.double_pole, bits),
                        ]
                    }),
                )
            })
            .collect())
    }

    fn evaluate_native(
        &mut self,
        ids: &[u64],
        shifts: &[FourMomentum<Complex<T>>],
        masses: &[Complex<T>],
        scales: &[Complex<T>],
        parameters: &[Complex<T>],
        s_matrix: Option<&[Complex<T>]>,
    ) -> PyResult<Vec<Lane<Complex<T>, String>>> {
        if let Some(sampling) = &mut self.sampling {
            self.sampling_values.resize(ids.len(), self.zero.clone());
            self.sampling_values.fill(self.zero.clone());
            self.sampling_statuses.resize(
                ids.len(),
                SamplingScaleStatus::Selected(SamplingScaleSource::UnitFallback),
            );
            sampling
                .evaluate_batch(
                    shifts,
                    masses,
                    ids.len(),
                    &mut self.sampling_values,
                    &mut self.sampling_statuses,
                )
                .map_err(error)?;
        }
        let mut results = Vec::with_capacity(3);
        for (index, sector) in self.sectors.iter_mut().enumerate() {
            if !self.active[index] {
                results.push(
                    ids.iter()
                        .map(|&id| Lane {
                            id,
                            status: EvaluationStatus::Success,
                            value: Some(Laurent::from([
                                self.zero.clone(),
                                self.zero.clone(),
                                self.zero.clone(),
                            ])),
                        })
                        .collect(),
                );
                continue;
            }
            results.push(
                sector
                    .evaluate(
                        ids,
                        AmplitudeInput {
                            shifts,
                            masses_squared: masses,
                            scales_squared: scales,
                            sampling_scales: &self.sampling_values,
                            s_matrix,
                        },
                        parameters,
                    )
                    .map_err(error)?,
            );
        }
        let mut sectors = results.into_iter();
        let zero = sectors.next().unwrap();
        let one = sectors.next().unwrap();
        let two = sectors.next().unwrap();
        Ok(zero
            .into_iter()
            .zip(one)
            .zip(two)
            .enumerate()
            .map(|(row, ((zero, one), two))| {
                if let Some(SamplingScaleStatus::InvalidInput(failure)) =
                    self.sampling_statuses.get(row)
                {
                    return Lane {
                        id: zero.id,
                        status: format!("sampling scale: {failure:?}"),
                        value: None,
                    };
                }
                let status = [zero.status, one.status, two.status]
                    .iter()
                    .enumerate()
                    .find(|(_, status)| **status != EvaluationStatus::Success)
                    .map(|(sector, status)| format!("epsilon sector {sector}: {status:?}"));
                let value = match (zero.value, one.value, two.value) {
                    (Some(a), Some(b), Some(c)) => Some(Laurent::epsilon_sectors(a, b, c)),
                    _ => None,
                };
                Lane {
                    id: zero.id,
                    status: status.unwrap_or_else(|| "success".into()),
                    value,
                }
            })
            .collect())
    }
}

enum Plans {
    Double(Reductions<f64>),
    Fast(FastReductions),
    DoubleDouble(Reductions<DoubleFloat>),
    Arbitrary(Reductions<Float>),
}

/// Native polynomial numerator reduction with explicit epsilon sectors.
/// The output is raw [finite, single pole, double pole], without model factors.
#[pyclass]
struct PolynomialPlan {
    plan: Plans,
    #[pyo3(get)]
    precision_bits: u32,
}

#[pymethods]
impl PolynomialPlan {
    /// Enable optional fast-engine/master-provider wall clocks, resetting counts.
    /// Ordinary calls leave profiling disabled and execute no timing callbacks.
    fn set_profiling(&mut self, enabled: bool) -> PyResult<()> {
        match &mut self.plan {
            Plans::Fast(plan) => {
                plan.set_profiling(enabled);
                Ok(())
            }
            _ => Err(error(
                "profiling currently requires fast or fast-tensor engine",
            )),
        }
    }

    /// Read and reset instrumented totals across all active epsilon sectors.
    /// Engine-exclusive time excludes delegated masters and numerator callbacks.
    /// Numerator preparation occurs outside the engine and is reported separately.
    fn take_profile(&mut self) -> PyResult<std::collections::BTreeMap<String, f64>> {
        let profile = match &mut self.plan {
            Plans::Fast(plan) => plan.take_profile(),
            _ => {
                return Err(error(
                    "profiling currently requires fast or fast-tensor engine",
                ));
            }
        };
        Ok([
            ("engine_seconds", profile.engine_time.as_secs_f64()),
            ("masters_seconds", profile.masters_time.as_secs_f64()),
            (
                "numerator_callback_seconds",
                profile.numerator_callback_time.as_secs_f64(),
            ),
            (
                "numerator_preparation_seconds",
                profile.numerator_preparation_time.as_secs_f64(),
            ),
            (
                "engine_excluding_masters_seconds",
                profile
                    .engine_time
                    .saturating_sub(profile.masters_time)
                    .as_secs_f64(),
            ),
            (
                "reduction_exclusive_seconds",
                profile
                    .engine_time
                    .saturating_sub(profile.masters_time + profile.numerator_callback_time)
                    .as_secs_f64(),
            ),
            ("engine_calls", profile.engine_calls as f64),
            ("master_calls", profile.master_calls as f64),
            ("master_rows", profile.master_rows as f64),
            ("numerator_calls", profile.numerator_calls as f64),
            ("numerator_rows", profile.numerator_rows as f64),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value))
        .collect())
    }

    #[new]
    #[pyo3(signature = (propagators, epsilon_sectors, parameters, arithmetic="double", precision=None, engine="generic"))]
    fn new(
        propagators: usize,
        epsilon_sectors: [String; 3],
        parameters: Vec<String>,
        arithmetic: &str,
        precision: Option<u32>,
        engine: &str,
    ) -> PyResult<Self> {
        let (arithmetic, bits) = arithmetic_mode(arithmetic, precision)?;
        if !matches!(engine, "generic" | "fast" | "fast-tensor") {
            return Err(error("engine must be generic, fast or fast-tensor"));
        }
        if engine != "generic" {
            if !matches!(arithmetic, Arithmetic::Double) {
                return Err(error("fast engine supports only double arithmetic"));
            }
            return Ok(Self {
                plan: Plans::Fast(FastReductions::new(
                    &epsilon_sectors,
                    &parameters,
                    propagators,
                    engine == "fast-tensor",
                )?),
                precision_bits: bits,
            });
        }
        let plan = match arithmetic {
            Arithmetic::Double => Plans::Double(Reductions::new(
                &epsilon_sectors,
                &parameters,
                propagators,
                0_f64,
            )?),
            Arithmetic::DoubleDouble => Plans::DoubleDouble(Reductions::new(
                &epsilon_sectors,
                &parameters,
                propagators,
                DoubleFloat::from(0.0),
            )?),
            Arithmetic::Arbitrary => Plans::Arbitrary(Reductions::new(
                &epsilon_sectors,
                &parameters,
                propagators,
                Float::with_val(bits, 0),
            )?),
        };
        Ok(Self {
            plan,
            precision_bits: bits,
        })
    }

    /// Every buffer is lane-major. The caller's original lane IDs survive
    /// compaction and resizing; failed lanes return None, never stale values.
    fn evaluate_decimal(
        &mut self,
        py: Python<'_>,
        ids: Vec<u64>,
        shifts: Vec<DecimalMomentum>,
        masses_squared: Vec<DecimalComplex>,
        scales_squared: Vec<DecimalComplex>,
        parameters: Vec<DecimalComplex>,
    ) -> PyResult<Vec<RawLane>> {
        py.detach(|| match &mut self.plan {
            Plans::Fast(p) => {
                p.evaluate_decimal(&ids, &shifts, &masses_squared, &scales_squared, &parameters)
            }
            Plans::Double(p) => p.evaluate(
                &ids,
                &shifts,
                &masses_squared,
                &scales_squared,
                &parameters,
                self.precision_bits,
            ),
            Plans::DoubleDouble(p) => p.evaluate(
                &ids,
                &shifts,
                &masses_squared,
                &scales_squared,
                &parameters,
                self.precision_bits,
            ),
            Plans::Arbitrary(p) => p.evaluate(
                &ids,
                &shifts,
                &masses_squared,
                &scales_squared,
                &parameters,
                self.precision_bits,
            ),
        })
    }
}

fn spinors<T: Carrier>(
    momenta: &[[String; 4]],
    helicities: &[i32],
    kinds: &[String],
    bits: u32,
) -> PyResult<Vec<[DecimalComplex; 4]>> {
    if momenta.len() != kinds.len() || momenta.len() != helicities.len() {
        return Err(error("massless state batch dimensions differ"));
    }
    let antiquarks = kinds
        .iter()
        .map(|kind| match kind.as_str() {
            "u" | "u_bar" => Ok(false),
            "v" | "v_bar" => Ok(true),
            _ => Err(error("massless state kind must be u, v, u_bar, or v_bar")),
        })
        .collect::<PyResult<Vec<_>>>()?;
    let momenta = momenta
        .iter()
        .map(|p| {
            let p = p
                .iter()
                .map(|x| T::parse_decimal(x, bits).map_err(error))
                .collect::<PyResult<Vec<_>>>()?;
            if p.iter().any(|x| !x.is_finite()) || p[0] <= p[0].zero() {
                return Err(error(
                    "massless states require finite physical momenta with positive energy",
                ));
            }
            Ok(p.try_into().unwrap_or_else(|_| unreachable!()))
        })
        .collect::<PyResult<Vec<[T; 4]>>>()?;
    Ok(physical_spinors(&momenta, helicities, kinds, &antiquarks)?
        .into_iter()
        .map(|row| row.map(|component| decimal_complex(component, bits)))
        .collect())
}

fn physical_spinors<T: Carrier>(
    momenta: &[[T; 4]],
    helicities: &[i32],
    kinds: &[String],
    antiquarks: &[bool],
) -> PyResult<Vec<[Complex<T>; 4]>> {
    let states = rusticol_core::__private::massless_dirac_batch(momenta, helicities, antiquarks)
        .map_err(error)?;
    Ok(states
        .into_iter()
        .zip(kinds)
        .map(|(x, kind)| {
            if kind == "u" || kind == "v_bar" {
                // Chiral Dirac adjoint: conjugation followed by gamma^0.
                [x[2].conj(), x[3].conj(), x[0].conj(), x[1].conj()]
            } else {
                x
            }
        })
        .collect())
}

/// Existing massless tree states for positive on-shell physical momenta.
/// Rows may mix helicities and kinds; all rows execute in one native call.
#[pyfunction]
#[pyo3(signature = (momenta, helicities, kinds, arithmetic="double", precision=None))]
fn massless_spinors_decimal(
    py: Python<'_>,
    momenta: Vec<[String; 4]>,
    helicities: Vec<i32>,
    kinds: Vec<String>,
    arithmetic: &str,
    precision: Option<u32>,
) -> PyResult<Vec<[DecimalComplex; 4]>> {
    let (arithmetic, bits) = arithmetic_mode(arithmetic, precision)?;
    py.detach(|| match arithmetic {
        Arithmetic::Double => spinors::<f64>(&momenta, &helicities, &kinds, bits),
        Arithmetic::DoubleDouble => spinors::<DoubleFloat>(&momenta, &helicities, &kinds, bits),
        Arithmetic::Arbitrary => spinors::<Float>(&momenta, &helicities, &kinds, bits),
    })
}

#[pymodule]
fn _loop_reduction(module: &Bound<'_, PyModule>) -> PyResult<()> {
    runtime::register_scalar_primitives();
    portable::register(module)?;
    batch::register(module)?;
    exact::register(module)?;
    states::register(module)?;
    module.add_class::<CoefficientEvaluator>()?;
    module.add_class::<PolynomialPlan>()?;
    module.add_function(wrap_pyfunction!(massless_spinors_decimal, module)?)?;
    Ok(())
}
