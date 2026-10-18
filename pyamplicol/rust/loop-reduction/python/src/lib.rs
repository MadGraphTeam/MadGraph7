// SPDX-License-Identifier: 0BSD

//! Experimental native consumer, separate until NinjaRS is self-contained.
//! Results are raw integrals, never Born/loop interference observables.

use std::fmt::LowerExp;

use pyo3::{exceptions::PyValueError, prelude::*};
use rusticol_loop_reduction::{
    AmplitudeInput, Complex, EvaluationStatus, FourMomentum, Laurent, NativeFloat,
    PolynomialReducer,
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

trait Carrier: NativeFloat + LowerExp {
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

fn precision_bits(digits: u32) -> PyResult<u32> {
    match digits {
        16 => Ok(53),
        32 => Ok(106),
        33.. => Float::decimal_digits_to_bits(f64::from(digits)).map_err(error),
        _ => Err(error(
            "precision must be 16, 32, or more than 32 decimal digits",
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
        .map_coeff(&|r| {
            Complex::new(
                prototype.from_rational(&r.re),
                prototype.from_rational(&r.im),
            )
        })
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
    #[pyo3(signature = (state, precision=16))]
    fn new(state: &[u8], precision: u32) -> PyResult<Self> {
        let bits = precision_bits(precision)?;
        let exact = exact_state(state)?;
        let evaluator = match precision {
            16 => Coefficients::Double(batch_evaluator(exact, &0_f64)),
            32 => Coefficients::DoubleDouble(batch_evaluator(exact, &DoubleFloat::from(0.0))),
            _ => Coefficients::Arbitrary(batch_evaluator(exact, &Float::with_val(bits, 0))),
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
        for expression in expressions {
            sectors.push(
                PolynomialReducer::new(
                    n,
                    parse(expression)?,
                    variables.clone(),
                    parameters.clone(),
                    prototype.clone(),
                    None,
                )
                .map_err(error)?,
            );
        }
        Ok(Self {
            sectors: sectors.try_into().ok().unwrap(),
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
        let mut results = Vec::with_capacity(3);
        for sector in &mut self.sectors {
            results.push(
                sector
                    .evaluate(
                        ids,
                        AmplitudeInput {
                            shifts: &shifts,
                            masses_squared: &masses,
                            scales_squared: &scales,
                            sampling_scales: &[],
                            s_matrix: None,
                        },
                        &parameters,
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
            .map(|((zero, one), two)| {
                let status = [zero.status, one.status, two.status]
                    .iter()
                    .enumerate()
                    .find(|(_, status)| **status != EvaluationStatus::Success)
                    .map(|(sector, status)| format!("epsilon sector {sector}: {status:?}"));
                let value = match (zero.value, one.value, two.value) {
                    (Some(a), Some(b), Some(c)) => {
                        let value = Laurent::epsilon_sectors(a, b, c);
                        Some([
                            decimal_complex(value.finite, bits),
                            decimal_complex(value.single_pole, bits),
                            decimal_complex(value.double_pole, bits),
                        ])
                    }
                    _ => None,
                };
                (zero.id, status.unwrap_or_else(|| "success".into()), value)
            })
            .collect())
    }
}

enum Plans {
    Double(Reductions<f64>),
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
    #[pyo3(signature = (propagators, epsilon_sectors, parameters, precision=16))]
    fn new(
        propagators: usize,
        epsilon_sectors: [String; 3],
        parameters: Vec<String>,
        precision: u32,
    ) -> PyResult<Self> {
        let bits = precision_bits(precision)?;
        let plan = match precision {
            16 => Plans::Double(Reductions::new(
                &epsilon_sectors,
                &parameters,
                propagators,
                0_f64,
            )?),
            32 => Plans::DoubleDouble(Reductions::new(
                &epsilon_sectors,
                &parameters,
                propagators,
                DoubleFloat::from(0.0),
            )?),
            _ => Plans::Arbitrary(Reductions::new(
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

#[pymodule]
fn _loop_reduction(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CoefficientEvaluator>()?;
    module.add_class::<PolynomialPlan>()?;
    Ok(())
}
