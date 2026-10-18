// SPDX-License-Identifier: 0BSD

//! Exact decimal ingress and dependency-owned rational polynomial evaluation.

use super::*;
use symbolica::{domains::float::FloatField, evaluate::Instruction};
use symbolica::{domains::rational::Rational, evaluate::ExpressionEvaluator};

pub fn decimal_rational(value: &str) -> ExecutionResult<Rational> {
    rusticol_core::source_scalar::decimal_rational(value).map_err(error)
}

pub struct ExactPolynomialEvaluator {
    evaluator: ExpressionEvaluator<Complex<Rational>>,
    output: Vec<Complex<Rational>>,
    input_count: usize,
    output_count: usize,
}

fn convert<T: BatchCarrier>(
    values: &[Complex<Rational>],
    prototype: T,
    ids: Vec<u64>,
    columns: usize,
    bits: u32,
) -> ExecutionResult<Batch> {
    Batch::typed(
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

impl Batch {
    /// Parse finite decimal literals exactly, then convert once to the carrier.
    pub fn from_exact_decimal(
        values: Vec<DecimalComplex>,
        rows: usize,
        columns: usize,
        arithmetic: &str,
        precision: Option<u32>,
        row_ids: Option<Vec<u64>>,
    ) -> ExecutionResult<Self> {
        let (mode, bits) = arithmetic_mode(arithmetic, precision)?;
        let expected = rows
            .checked_mul(columns)
            .ok_or_else(|| error("exact batch dimensions overflow"))?;
        if values.len() != expected {
            return Err(error("exact batch values do not match dimensions"));
        }
        let ids = row_ids.unwrap_or_else(|| (0..rows as u64).collect());
        if ids.len() != rows {
            return Err(error("exact batch row identities do not match dimensions"));
        }
        {
            let exact = values
                .iter()
                .map(|value| {
                    Ok(Complex::new(
                        decimal_rational(&value[0])?,
                        decimal_rational(&value[1])?,
                    ))
                })
                .collect::<ExecutionResult<Vec<_>>>()?;
            match mode {
                Arithmetic::Double => convert(&exact, 0_f64, ids, columns, bits),
                Arithmetic::DoubleDouble => {
                    convert(&exact, DoubleFloat::from(0.0), ids, columns, bits)
                }
                Arithmetic::Arbitrary => {
                    convert(&exact, Float::with_val(bits, 0), ids, columns, bits)
                }
            }
        }
    }
}

impl ExactPolynomialEvaluator {
    pub fn input_count(&self) -> usize {
        self.input_count
    }
    pub fn output_count(&self) -> usize {
        self.output_count
    }
    pub fn new(state: &[u8]) -> ExecutionResult<Self> {
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
    pub fn evaluate_decimal(
        &mut self,
        values: Vec<DecimalComplex>,
        rows: usize,
        arithmetic: &str,
        precision: Option<u32>,
        row_ids: Option<Vec<u64>>,
    ) -> ExecutionResult<Batch> {
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
        {
            let parameters = values
                .iter()
                .map(|v| {
                    Ok(Complex::new(
                        decimal_rational(&v[0])?,
                        decimal_rational(&v[1])?,
                    ))
                })
                .collect::<ExecutionResult<Vec<_>>>()?;
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
        }
    }
}
