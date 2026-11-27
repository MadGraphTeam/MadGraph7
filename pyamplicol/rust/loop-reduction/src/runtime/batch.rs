// SPDX-License-Identifier: 0BSD

//! Immutable typed blocks and native gathers and concatenation.
use super::*;
use std::sync::Arc;

#[derive(Clone)]
pub enum Data {
    Double(Arc<Vec<Complex<f64>>>),
    DoubleDouble(Arc<Vec<Complex<DoubleFloat>>>),
    Arbitrary(Arc<Vec<Complex<Float>>>),
}

pub trait BatchCarrier: Carrier {
    fn data(values: Vec<Complex<Self>>) -> Data;
}
impl BatchCarrier for f64 {
    fn data(values: Vec<Complex<Self>>) -> Data {
        Data::Double(Arc::new(values))
    }
}
impl BatchCarrier for DoubleFloat {
    fn data(values: Vec<Complex<Self>>) -> Data {
        Data::DoubleDouble(Arc::new(values))
    }
}
impl BatchCarrier for Float {
    fn data(values: Vec<Complex<Self>>) -> Data {
        Data::Arbitrary(Arc::new(values))
    }
}

pub fn shape(rows: usize, columns: usize) -> ExecutionResult<usize> {
    rows.checked_mul(columns)
        .ok_or_else(|| error("batch dimensions overflow"))
}

#[derive(Clone)]
pub struct Batch {
    data: Data,
    ids: Vec<u64>,
    rows: usize,
    columns: usize,
    precision_bits: u32,
}

impl Batch {
    pub fn data(&self) -> &Data {
        &self.data
    }
    pub fn ids(&self) -> &[u64] {
        &self.ids
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn columns(&self) -> usize {
        self.columns
    }
    pub fn precision_bits(&self) -> u32 {
        self.precision_bits
    }
    pub fn typed<T: BatchCarrier>(
        values: Vec<Complex<T>>,
        ids: Vec<u64>,
        columns: usize,
        bits: u32,
    ) -> ExecutionResult<Self> {
        let data = T::data(values);
        let length = match &data {
            Data::Double(values) => values.len(),
            Data::DoubleDouble(values) => values.len(),
            Data::Arbitrary(values) => values.len(),
        };
        if length != shape(ids.len(), columns)? {
            return Err(error("batch values do not match dimensions"));
        }
        Ok(Self {
            data,
            rows: ids.len(),
            ids,
            columns,
            precision_bits: bits,
        })
    }
    pub fn compatible(&self, other: &Self) -> ExecutionResult<()> {
        if self.precision_bits != other.precision_bits
            || std::mem::discriminant(&self.data) != std::mem::discriminant(&other.data)
        {
            return Err(error("batch arithmetic or precision differs"));
        }
        Ok(())
    }
    pub fn same_rows(&self, other: &Self) -> ExecutionResult<()> {
        self.compatible(other)?;
        if self.ids != other.ids {
            return Err(error("batch row identities differ"));
        }
        Ok(())
    }
}

fn parse_values<T: Carrier>(
    values: &[DecimalComplex],
    bits: u32,
) -> ExecutionResult<Vec<Complex<T>>> {
    values.iter().map(|v| parse_complex(v, bits)).collect()
}
fn decimals<T: Carrier>(values: &[Complex<T>], bits: u32) -> Vec<DecimalComplex> {
    values
        .iter()
        .cloned()
        .map(|v| decimal_complex(v, bits))
        .collect()
}
fn nonfinite<T: Carrier>(values: &[Complex<T>], columns: usize, ids: &[u64]) -> Vec<u64> {
    ids.iter()
        .enumerate()
        .filter_map(|(row, &id)| {
            values[row * columns..(row + 1) * columns]
                .iter()
                .any(|value| !value.re.is_finite() || !value.im.is_finite())
                .then_some(id)
        })
        .collect()
}
fn take<T: BatchCarrier>(
    values: &[Complex<T>],
    source_columns: usize,
    rows: &[usize],
    columns: &[usize],
    ids: Vec<u64>,
    bits: u32,
) -> ExecutionResult<Batch> {
    let mut out = Vec::with_capacity(shape(rows.len(), columns.len())?);
    for row in rows {
        for column in columns {
            out.push(values[row * source_columns + column].clone());
        }
    }
    Batch::typed(out, ids, columns.len(), bits)
}

impl Batch {
    pub fn from_decimal(
        values: Vec<DecimalComplex>,
        rows: usize,
        columns: usize,
        arithmetic: &str,
        precision: Option<u32>,
        row_ids: Option<Vec<u64>>,
    ) -> ExecutionResult<Self> {
        let (mode, bits) = arithmetic_mode(arithmetic, precision)?;
        if values.len() != shape(rows, columns)? {
            return Err(error("batch values do not match dimensions"));
        }
        let ids = row_ids.unwrap_or_else(|| (0..rows as u64).collect());
        if ids.len() != rows {
            return Err(error("batch row identities do not match dimensions"));
        }
        match mode {
            Arithmetic::Double => {
                Self::typed(parse_values::<f64>(&values, bits)?, ids, columns, bits)
            }
            Arithmetic::DoubleDouble => Self::typed(
                parse_values::<DoubleFloat>(&values, bits)?,
                ids,
                columns,
                bits,
            ),
            Arithmetic::Arbitrary => {
                Self::typed(parse_values::<Float>(&values, bits)?, ids, columns, bits)
            }
        }
    }
    pub fn row_ids(&self) -> Vec<u64> {
        self.ids.clone()
    }
    pub fn to_decimal(&self) -> Vec<DecimalComplex> {
        match &self.data {
            Data::Double(v) => decimals(v, self.precision_bits),
            Data::DoubleDouble(v) => decimals(v, self.precision_bits),
            Data::Arbitrary(v) => decimals(v, self.precision_bits),
        }
    }
    pub fn nonfinite_rows(&self) -> Vec<u64> {
        match &self.data {
            Data::Double(v) => nonfinite(v, self.columns, &self.ids),
            Data::DoubleDouble(v) => nonfinite(v, self.columns, &self.ids),
            Data::Arbitrary(v) => nonfinite(v, self.columns, &self.ids),
        }
    }
    pub fn take_rows(
        &self,
        indices: Vec<usize>,
        row_ids: Option<Vec<u64>>,
    ) -> ExecutionResult<Self> {
        if indices.iter().any(|&i| i >= self.rows) {
            return Err(error("batch row index out of range"));
        }
        let ids = row_ids.unwrap_or_else(|| indices.iter().map(|&i| self.ids[i]).collect());
        if ids.len() != indices.len() {
            return Err(error("batch row identities do not match dimensions"));
        }
        let columns: Vec<_> = (0..self.columns).collect();
        match &self.data {
            Data::Double(v) => take(
                v,
                self.columns,
                &indices,
                &columns,
                ids,
                self.precision_bits,
            ),
            Data::DoubleDouble(v) => take(
                v,
                self.columns,
                &indices,
                &columns,
                ids,
                self.precision_bits,
            ),
            Data::Arbitrary(v) => take(
                v,
                self.columns,
                &indices,
                &columns,
                ids,
                self.precision_bits,
            ),
        }
    }
    pub fn take_columns(&self, indices: Vec<usize>) -> ExecutionResult<Self> {
        if indices.iter().any(|&i| i >= self.columns) {
            return Err(error("batch column index out of range"));
        }
        let rows: Vec<_> = (0..self.rows).collect();
        match &self.data {
            Data::Double(v) => take(
                v,
                self.columns,
                &rows,
                &indices,
                self.ids.clone(),
                self.precision_bits,
            ),
            Data::DoubleDouble(v) => take(
                v,
                self.columns,
                &rows,
                &indices,
                self.ids.clone(),
                self.precision_bits,
            ),
            Data::Arbitrary(v) => take(
                v,
                self.columns,
                &rows,
                &indices,
                self.ids.clone(),
                self.precision_bits,
            ),
        }
    }
    pub fn reshape_rows(
        &self,
        rows: usize,
        columns: usize,
        row_ids: Vec<u64>,
    ) -> ExecutionResult<Self> {
        if shape(rows, columns)? != shape(self.rows, self.columns)? || row_ids.len() != rows {
            return Err(error("reshape dimensions or row identities differ"));
        }
        Ok(Self {
            data: self.data.clone(),
            ids: row_ids,
            rows,
            columns,
            precision_bits: self.precision_bits,
        })
    }
    pub fn concat_columns(batches: &[&Batch]) -> ExecutionResult<Self> {
        let first = batches
            .first()
            .ok_or_else(|| error("concatenation needs at least one batch"))?;
        for batch in batches {
            first.same_rows(batch)?;
        }
        let columns = batches.iter().try_fold(0_usize, |n, b| {
            n.checked_add(b.columns)
                .ok_or_else(|| error("batch dimensions overflow"))
        })?;
        macro_rules! concat {
            ($variant:ident) => {{
                let mut values = Vec::with_capacity(shape(first.rows, columns)?);
                for row in 0..first.rows {
                    for batch in batches {
                        let Data::$variant(source) = &batch.data else {
                            unreachable!()
                        };
                        values.extend_from_slice(
                            &source[row * batch.columns..(row + 1) * batch.columns],
                        );
                    }
                }
                Self::typed(values, first.ids.clone(), columns, first.precision_bits)
            }};
        }
        match &first.data {
            Data::Double(_) => concat!(Double),
            Data::DoubleDouble(_) => concat!(DoubleDouble),
            Data::Arbitrary(_) => concat!(Arbitrary),
        }
    }
}
