// SPDX-License-Identifier: 0BSD

//! Experimental native consumer, separate until NinjaRS is self-contained.
//! Results are raw integrals, never Born/loop interference observables.

use std::fmt::LowerExp;

mod batch;
mod exact;
mod fast;
mod predicates;
mod states;
mod stability;
use batch::{BatchCarrier, Data, NativeBatch};
use fast::FastReductions;

use pyo3::{exceptions::PyValueError, prelude::*};
use rusticol_loop_reduction::{
    AmplitudeInput, Complex, EvaluationStatus, FourMomentum, Lane, Laurent, NativeFloat,
    PolynomialReducer, SamplingScaleProvider, SamplingScaleSource, SamplingScaleStatus,
    SymbolicaSamplingScales,
};
use symbolica::{
    atom::Atom,
    domains::float::{DoubleFloat, Float, Real},
    domains::rational::Rational,
    evaluate::{BatchExpressionEvaluator, EvaluationDomain, ExpressionEvaluator},
};

type DecimalComplex = [String; 2];
type DecimalMomentum = [DecimalComplex; 4];
type RawLane = (u64, String, Option<[DecimalComplex; 3]>);

fn error(value: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(value.to_string())
}

trait Carrier: NativeFloat + LowerExp + From<f64> {
    fn parse_decimal(value: &str, bits: u32) -> Result<Self, String>;
}

impl Carrier for f64 {
    fn parse_decimal(value: &str, _: u32) -> Result<Self, String> {
        value.parse::<f64>().map_err(|e| e.to_string())
    }
}

impl Carrier for DoubleFloat {
    fn parse_decimal(value: &str, _: u32) -> Result<Self, String> {
        Float::parse(value, Some(106)).map(|f| f.to_double_float())
    }
}

impl Carrier for Float {
    fn parse_decimal(value: &str, bits: u32) -> Result<Self, String> {
        Float::parse(value, Some(bits))
    }
}

fn parse_complex<T: Carrier>(value: &DecimalComplex, bits: u32) -> PyResult<Complex<T>> {
    Ok(Complex::new(
        T::parse_decimal(&value[0], bits).map_err(error)?,
        T::parse_decimal(&value[1], bits).map_err(error)?,
    ))
}

fn decimal_complex<T: Carrier>(value: Complex<T>, bits: u32) -> DecimalComplex {
    // Default Symbolica Display intentionally drops some bits. Explicit output
    // digits allow each carrier to round-trip through the private Python seam.
    let digits = (f64::from(bits) * std::f64::consts::LOG10_2).ceil() as usize + 2;
    [
        format!("{:.digits$e}", value.re),
        format!("{:.digits$e}", value.im),
    ]
}

#[derive(Clone, Copy)]
enum Arithmetic {
    Double,
    DoubleDouble,
    Arbitrary,
}

fn arithmetic_mode(arithmetic: &str, precision: Option<u32>) -> PyResult<(Arithmetic, u32)> {
    match (arithmetic, precision) {
        ("double", None) => Ok((Arithmetic::Double, 53)),
        ("double-double", None) => Ok((Arithmetic::DoubleDouble, 106)),
        ("arbitrary", Some(digits)) if digits > 0 => Ok((
            Arithmetic::Arbitrary,
            Float::decimal_digits_to_bits(f64::from(digits)).map_err(error)?,
        )),
        ("arbitrary", _) => Err(error(
            "arbitrary arithmetic requires a positive decimal precision",
        )),
        ("double" | "double-double", Some(_)) => {
            Err(error("precision applies only to arbitrary arithmetic"))
        }
        _ => Err(error(
            "arithmetic must be double, double-double, or arbitrary",
        )),
    }
}

fn exact_state(bytes: &[u8]) -> PyResult<ExpressionEvaluator<Complex<Rational>>> {
    rusticol_core::__private::decode_exact_evaluator_state(bytes).map_err(error)
}

fn batch_evaluator<T: Carrier>(
    exact: ExpressionEvaluator<Complex<Rational>>,
    prototype: &T,
) -> BatchExpressionEvaluator<Complex<T>>
where
    Complex<T>: Real + EvaluationDomain,
{
    exact
        .map_coeff_with_prec(
            &|r| {
                Complex::new(
                    prototype.from_rational(&r.re),
                    prototype.from_rational(&r.im),
                )
            },
            prototype.get_precision(),
        )
        .into_batch_evaluator()
}

enum Coefficients {
    Double(BatchExpressionEvaluator<Complex<f64>>),
    DoubleDouble(BatchExpressionEvaluator<Complex<DoubleFloat>>),
    Arbitrary(BatchExpressionEvaluator<Complex<Float>>),
}

/// Execute saved scalar coefficient instructions across all lanes natively.
#[pyclass]
struct CoefficientEvaluator {
    evaluator: Coefficients,
    #[pyo3(get)]
    precision_bits: u32,
}

fn evaluate_coefficients<T: Carrier>(
    evaluator: &mut BatchExpressionEvaluator<Complex<T>>,
    values: &[DecimalComplex],
    rows: usize,
    bits: u32,
) -> PyResult<Vec<DecimalComplex>>
where
    Complex<T>: Real + EvaluationDomain,
{
    let parameters: Vec<_> = values
        .iter()
        .map(|value| parse_complex::<T>(value, bits))
        .collect::<PyResult<_>>()?;
    let zero = parse_complex::<T>(&["0".into(), "0".into()], bits)?;
    let size = rows
        .checked_mul(evaluator.get_output_len())
        .ok_or_else(|| error("coefficient output dimensions overflow"))?;
    let mut output = vec![zero; size];
    evaluator
        .evaluate_batch(&parameters, &mut output, rows)
        .map_err(error)?;
    Ok(output
        .into_iter()
        .map(|value| decimal_complex(value, bits))
        .collect())
}

#[pymethods]
impl CoefficientEvaluator {
    #[new]
    #[pyo3(signature = (state, arithmetic="double", precision=None))]
    fn new(state: &[u8], arithmetic: &str, precision: Option<u32>) -> PyResult<Self> {
        let (arithmetic, bits) = arithmetic_mode(arithmetic, precision)?;
        let exact = exact_state(state)?;
        let evaluator = match arithmetic {
            Arithmetic::Double => Coefficients::Double(batch_evaluator(exact, &0_f64)),
            Arithmetic::DoubleDouble => {
                Coefficients::DoubleDouble(batch_evaluator(exact, &DoubleFloat::from(0.0)))
            }
            Arithmetic::Arbitrary => {
                Coefficients::Arbitrary(batch_evaluator(exact, &Float::with_val(bits, 0)))
            }
        };
        Ok(Self {
            evaluator,
            precision_bits: bits,
        })
    }

    /// Inputs and outputs are flattened row-major complex decimal pairs.
    fn evaluate_decimal(
        &mut self,
        py: Python<'_>,
        values: Vec<DecimalComplex>,
        rows: usize,
    ) -> PyResult<Vec<DecimalComplex>> {
        py.detach(|| match &mut self.evaluator {
            Coefficients::Double(e) => evaluate_coefficients(e, &values, rows, self.precision_bits),
            Coefficients::DoubleDouble(e) => {
                evaluate_coefficients(e, &values, rows, self.precision_bits)
            }
            Coefficients::Arbitrary(e) => {
                evaluate_coefficients(e, &values, rows, self.precision_bits)
            }
        })
    }

    fn evaluate_batch(&mut self, py: Python<'_>, input: &NativeBatch) -> PyResult<NativeBatch> {
        if self.precision_bits != input.precision_bits {
            return Err(error("batch arithmetic or precision differs"));
        }
        fn run<T: BatchCarrier>(
            evaluator: &mut BatchExpressionEvaluator<Complex<T>>,
            values: &[Complex<T>],
            input: &NativeBatch,
        ) -> PyResult<NativeBatch>
        where
            Complex<T>: Real + EvaluationDomain,
        {
            if input.columns != evaluator.get_input_len() {
                return Err(error("coefficient input dimensions differ"));
            }
            let zero = T::parse_decimal("0", input.precision_bits).map_err(error)?;
            let count = evaluator.get_output_len();
            let size = input
                .rows
                .checked_mul(count)
                .ok_or_else(|| error("batch dimensions overflow"))?;
            let mut output = vec![Complex::new(zero.clone(), zero); size];
            evaluator
                .evaluate_batch(values, &mut output, input.rows)
                .map_err(error)?;
            NativeBatch::typed(output, input.ids.clone(), count, input.precision_bits)
        }
        py.detach(|| match (&mut self.evaluator, &input.data) {
            (Coefficients::Double(e), Data::Double(v)) => run(e, v, input),
            (Coefficients::DoubleDouble(e), Data::DoubleDouble(v)) => run(e, v, input),
            (Coefficients::Arbitrary(e), Data::Arbitrary(v)) => run(e, v, input),
            _ => Err(error("batch arithmetic or precision differs")),
        })
    }

    /// Binary64 inputs stay binary64; higher precision requires decimal inputs.
    fn evaluate_f64(
        &mut self,
        py: Python<'_>,
        values: Vec<[f64; 2]>,
        rows: usize,
    ) -> PyResult<Vec<[f64; 2]>> {
        let Coefficients::Double(evaluator) = &mut self.evaluator else {
            return Err(error(
                "use evaluate_decimal to preserve higher-precision inputs",
            ));
        };
        py.detach(|| {
            let parameters: Vec<_> = values
                .into_iter()
                .map(|[re, im]| Complex::new(re, im))
                .collect();
            let size = rows
                .checked_mul(evaluator.get_output_len())
                .ok_or_else(|| error("coefficient output dimensions overflow"))?;
            let mut output = vec![Complex::new(0., 0.); size];
            evaluator
                .evaluate_batch(&parameters, &mut output, rows)
                .map_err(error)?;
            Ok(output.into_iter().map(|z| [z.re, z.im]).collect())
        })
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
        if engine != "generic" && engine != "fast" {
            return Err(error("engine must be generic or fast"));
        }
        if engine == "fast" {
            if !matches!(arithmetic, Arithmetic::Double) {
                return Err(error("fast engine supports only double arithmetic"));
            }
            return Ok(Self {
                plan: Plans::Fast(FastReductions::new(
                    &epsilon_sectors,
                    &parameters,
                    propagators,
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
    predicates::register();
    batch::register(module)?;
    exact::register(module)?;
    states::register(module)?;
    module.add_class::<CoefficientEvaluator>()?;
    module.add_class::<PolynomialPlan>()?;
    module.add_function(wrap_pyfunction!(massless_spinors_decimal, module)?)?;
    Ok(())
}
