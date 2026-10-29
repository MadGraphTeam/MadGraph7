// SPDX-License-Identifier: 0BSD

//! Exact decimal ingress and dependency-owned rational polynomial evaluation.

use super::*;

use symbolica::domains::rational::Rational;

pub fn decimal_rational(value: &str) -> ExecutionResult<Rational> {
    rusticol_core::source_scalar::decimal_rational(value).map_err(error)
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
