// SPDX-License-Identifier: 0BSD

//! Exact decimal ingress and dependency-owned rational polynomial evaluation.

use super::*;
use symbolica::{
    domains::{float::FloatField, integer::Integer},
    evaluate::Instruction,
};

pub(super) fn decimal_rational(value: &str) -> PyResult<Rational> {
    // Lexical decimal conversion only. Integer powers, products and rational
    // normalization all use Numerica, without a floating intermediate.
    let (negative, value) = if let Some(value) = value.strip_prefix('-') {
        (true, value)
    } else {
        (false, value.strip_prefix('+').unwrap_or(value))
    };
    let mut exponent_parts = value.split(['e', 'E']);
    let mantissa = exponent_parts.next().unwrap_or_default();
    let exponent = exponent_parts
        .next()
        .map(|e| e.parse::<i64>().map_err(error))
        .transpose()?
        .unwrap_or(0);
    if exponent_parts.next().is_some() {
        return Err(error("expected a finite decimal literal"));
    }
    let mut decimal_parts = mantissa.split('.');
    let whole = decimal_parts.next().unwrap_or_default();
    let fractional = decimal_parts.next().unwrap_or_default();
    if decimal_parts.next().is_some()
        || (whole.is_empty() && fractional.is_empty())
        || !whole
            .bytes()
            .chain(fractional.bytes())
            .all(|c| c.is_ascii_digit())
    {
        return Err(error("expected a finite decimal literal"));
    }
    let digits = format!("{whole}{fractional}");
    let mut numerator = digits.parse::<Integer>().map_err(error)?;
    if negative {
        numerator = -numerator;
    }
    let decimal_places = i64::try_from(fractional.len()).map_err(error)?;
    let exponent = exponent
        .checked_sub(decimal_places)
        .ok_or_else(|| error("decimal exponent overflows"))?;
    let scale = Integer::from(10).pow(exponent.unsigned_abs());
    Ok(if exponent < 0 {
        Rational::from((numerator, scale))
    } else {
        Rational::from(numerator * scale)
    })
}

#[pyclass]
pub(super) struct ExactPolynomialEvaluator {
    evaluator: ExpressionEvaluator<Complex<Rational>>,
    output: Vec<Complex<Rational>>,
    #[pyo3(get)]
    input_count: usize,
    #[pyo3(get)]
    output_count: usize,
}

fn convert<T: BatchCarrier>(
    values: &[Complex<Rational>],
    prototype: T,
    ids: Vec<u64>,
    columns: usize,
    bits: u32,
) -> PyResult<NativeBatch> {
    NativeBatch::typed(
        values
            .iter()
            .map(|v| {
                Complex::new(
                    prototype.from_rational(&v.re),
                    prototype.from_rational(&v.im),
                )
            })
            .collect(),
        ids,
        columns,
        bits,
    )
}

#[pymethods]
impl ExactPolynomialEvaluator {
    #[new]
    fn new(state: &[u8]) -> PyResult<Self> {
        let exact = exact_state(state)?;
        let exported = exact.export_instructions();
        if !exported.constant_functions.is_empty()
            || !exported.sub_evaluators.is_empty()
            || exported
                .instructions
                .iter()
                .any(|instruction| match instruction {
                    Instruction::Add(..) | Instruction::Mul(..) | Instruction::Assign(..) => false,
                    Instruction::Pow(_, _, exponent, _) => *exponent < 0,
                    _ => true,
                })
        {
            return Err(error(
                "exact kinematic evaluator requires straight-line rational polynomial operations",
            ));
        }
        Ok(Self {
            input_count: exact.get_input_len(),
            output_count: exact.get_output_len(),
            evaluator: exact,
            output: Vec::new(),
        })
    }
    #[pyo3(signature = (values, rows, arithmetic="double", precision=None, row_ids=None))]
    fn evaluate_decimal(
        &mut self,
        py: Python<'_>,
        values: Vec<DecimalComplex>,
        rows: usize,
        arithmetic: &str,
        precision: Option<u32>,
        row_ids: Option<Vec<u64>>,
    ) -> PyResult<NativeBatch> {
        let (mode, bits) = arithmetic_mode(arithmetic, precision)?;
        let expected = rows
            .checked_mul(self.input_count)
            .ok_or_else(|| error("exact batch dimensions overflow"))?;
        if values.len() != expected {
            return Err(error("exact polynomial input dimensions differ"));
        }
        let ids = row_ids.unwrap_or_else(|| (0..rows as u64).collect());
        if ids.len() != rows {
            return Err(error("exact polynomial row identities differ"));
        }
        py.detach(|| {
            let parameters = values
                .iter()
                .map(|v| {
                    Ok(Complex::new(
                        decimal_rational(&v[0])?,
                        decimal_rational(&v[1])?,
                    ))
                })
                .collect::<PyResult<Vec<_>>>()?;
            let count = rows
                .checked_mul(self.output_count)
                .ok_or_else(|| error("exact batch dimensions overflow"))?;
            self.output
                .resize(count, Complex::new(Rational::from(0), Rational::from(0)));
            // Symbolica's rational-ring API is scalar; execute the whole
            // preprocessing block natively, reusing its evaluator stack.
            let ring = FloatField::from_rep(Complex::new(Rational::from(0), Rational::from(0)));
            for row in 0..rows {
                self.evaluator
                    .try_evaluate_in_ring(
                        &parameters[row * self.input_count..(row + 1) * self.input_count],
                        &mut self.output[row * self.output_count..(row + 1) * self.output_count],
                        &ring,
                    )
                    .map_err(error)?;
            }
            match mode {
                Arithmetic::Double => convert(&self.output, 0_f64, ids, self.output_count, bits),
                Arithmetic::DoubleDouble => convert(
                    &self.output,
                    DoubleFloat::from(0.0),
                    ids,
                    self.output_count,
                    bits,
                ),
                Arithmetic::Arbitrary => convert(
                    &self.output,
                    Float::with_val(bits, 0),
                    ids,
                    self.output_count,
                    bits,
                ),
            }
        })
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ExactPolynomialEvaluator>()?;
    Ok(())
}
