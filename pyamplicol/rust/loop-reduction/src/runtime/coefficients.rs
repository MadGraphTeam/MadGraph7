// SPDX-License-Identifier: 0BSD

//! Saved dependency evaluators shared by native callers and the Python binding.
use super::*;
use symbolica::{
    domains::{
        float::{DoubleFloat, Float, Real},
        rational::Rational,
    },
    evaluate::{BatchExpressionEvaluator, EvaluationDomain, ExpressionEvaluator},
};

pub fn exact_state(bytes: &[u8]) -> ExecutionResult<ExpressionEvaluator<Complex<Rational>>> {
    super::register_scalar_primitives();
    rusticol_core::__private::decode_exact_evaluator_state(bytes).map_err(error)
}

pub(super) fn batch_evaluator<T: Carrier>(
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
pub struct CoefficientEvaluator {
    evaluator: Coefficients,
    precision_bits: u32,
}

fn evaluate_coefficients<T: Carrier>(
    evaluator: &mut BatchExpressionEvaluator<Complex<T>>,
    values: &[DecimalComplex],
    rows: usize,
    bits: u32,
) -> ExecutionResult<Vec<DecimalComplex>>
where
    Complex<T>: Real + EvaluationDomain,
{
    let parameters: Vec<_> = values
        .iter()
        .map(|value| parse_complex::<T>(value, bits))
        .collect::<ExecutionResult<_>>()?;
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

impl CoefficientEvaluator {
    pub fn precision_bits(&self) -> u32 {
        self.precision_bits
    }
    pub fn new(state: &[u8], arithmetic: &str, precision: Option<u32>) -> ExecutionResult<Self> {
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
    pub fn evaluate_decimal(
        &mut self,
        values: Vec<DecimalComplex>,
        rows: usize,
    ) -> ExecutionResult<Vec<DecimalComplex>> {
        match &mut self.evaluator {
            Coefficients::Double(e) => evaluate_coefficients(e, &values, rows, self.precision_bits),
            Coefficients::DoubleDouble(e) => {
                evaluate_coefficients(e, &values, rows, self.precision_bits)
            }
            Coefficients::Arbitrary(e) => {
                evaluate_coefficients(e, &values, rows, self.precision_bits)
            }
        }
    }

    pub fn evaluate_batch(&mut self, input: &Batch) -> ExecutionResult<Batch> {
        if self.precision_bits != input.precision_bits() {
            return Err(error("batch arithmetic or precision differs"));
        }
        fn run<T: BatchCarrier>(
            evaluator: &mut BatchExpressionEvaluator<Complex<T>>,
            values: &[Complex<T>],
            input: &Batch,
        ) -> ExecutionResult<Batch>
        where
            Complex<T>: Real + EvaluationDomain,
        {
            if input.columns() != evaluator.get_input_len() {
                return Err(error("coefficient input dimensions differ"));
            }
            let zero = T::parse_decimal("0", input.precision_bits()).map_err(error)?;
            let count = evaluator.get_output_len();
            let size = input
                .rows()
                .checked_mul(count)
                .ok_or_else(|| error("batch dimensions overflow"))?;
            let mut output = vec![Complex::new(zero.clone(), zero); size];
            evaluator
                .evaluate_batch(values, &mut output, input.rows())
                .map_err(error)?;
            Batch::typed(output, input.ids().to_vec(), count, input.precision_bits())
        }
        match (&mut self.evaluator, input.data()) {
            (Coefficients::Double(e), Data::Double(v)) => run(e, v, input),
            (Coefficients::DoubleDouble(e), Data::DoubleDouble(v)) => run(e, v, input),
            (Coefficients::Arbitrary(e), Data::Arbitrary(v)) => run(e, v, input),
            _ => Err(error("batch arithmetic or precision differs")),
        }
    }

    /// Binary64 inputs stay binary64; higher precision requires decimal inputs.
    pub fn evaluate_f64(
        &mut self,
        values: Vec<[f64; 2]>,
        rows: usize,
    ) -> ExecutionResult<Vec<[f64; 2]>> {
        let Coefficients::Double(evaluator) = &mut self.evaluator else {
            return Err(error(
                "use evaluate_decimal to preserve higher-precision inputs",
            ));
        };
        {
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
        }
    }
}
