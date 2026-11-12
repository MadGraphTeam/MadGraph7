// SPDX-License-Identifier: 0BSD

//! Native carrier conversion and lane transport, not a reduction algorithm.

use super::*;
use std::collections::{HashMap, HashSet};
use symbolica::domains::float::{FloatLike, RealLike, RoundingDirection};

fn floats(input: &Batch, bits: u32) -> Vec<Complex<Float>> {
    fn from_double_double(value: DoubleFloat, bits: u32) -> Float {
        let value = value.into_inner();
        Float::with_val(53, value.hi()).add_round(
            &Float::with_val(53, value.lo()),
            bits,
            RoundingDirection::Nearest,
        )
    }
    fn at_precision(mut value: Float, bits: u32) -> Float {
        value.set_precision(bits);
        value
    }
    match input.data() {
        Data::Double(values) => values
            .iter()
            .map(|x| Complex::new(Float::with_val(bits, x.re), Float::with_val(bits, x.im)))
            .collect(),
        Data::DoubleDouble(values) => values
            .iter()
            .map(|x| {
                Complex::new(
                    from_double_double(x.re, bits),
                    from_double_double(x.im, bits),
                )
            })
            .collect(),
        Data::Arbitrary(values) => values
            .iter()
            .map(|x| {
                Complex::new(
                    at_precision(x.re.clone(), bits),
                    at_precision(x.im.clone(), bits),
                )
            })
            .collect(),
    }
}

impl Batch {
    /// Convert existing numbers directly through dependency carriers. Promotion
    /// cannot recover input bits or establish more accuracy; rescue must rerun
    /// the original exact input before converting its final output back.
    pub fn convert_arithmetic(
        &self,
        arithmetic: &str,
        precision: Option<u32>,
    ) -> ExecutionResult<Self> {
        let (mode, bits) = arithmetic_mode(arithmetic, precision)?;
        if bits == self.precision_bits()
            && matches!(
                (mode, self.data()),
                (Arithmetic::Double, Data::Double(_))
                    | (Arithmetic::DoubleDouble, Data::DoubleDouble(_))
                    | (Arithmetic::Arbitrary, Data::Arbitrary(_))
            )
        {
            return Ok(self.clone());
        }
        match mode {
            Arithmetic::Double => {
                let values = match self.data() {
                    Data::Double(v) => v.as_ref().clone(),
                    Data::DoubleDouble(v) => v
                        .iter()
                        .map(|x| Complex::new(x.re.to_f64(), x.im.to_f64()))
                        .collect(),
                    Data::Arbitrary(v) => v
                        .iter()
                        .map(|x| Complex::new(x.re.to_f64(), x.im.to_f64()))
                        .collect(),
                };
                Self::typed(values, self.ids().to_vec(), self.columns(), bits)
            }
            Arithmetic::DoubleDouble => {
                let values = match self.data() {
                    Data::Double(v) => v
                        .iter()
                        .map(|x| Complex::new(DoubleFloat::from(x.re), DoubleFloat::from(x.im)))
                        .collect(),
                    Data::DoubleDouble(v) => v.as_ref().clone(),
                    Data::Arbitrary(v) => v
                        .iter()
                        .map(|x| Complex::new(x.re.to_double_float(), x.im.to_double_float()))
                        .collect(),
                };
                Self::typed(values, self.ids().to_vec(), self.columns(), bits)
            }
            Arithmetic::Arbitrary => Self::typed(
                floats(self, bits),
                self.ids().to_vec(),
                self.columns(),
                bits,
            ),
        }
    }

    /// Scatter disjoint successful groups into original-ID order. Missing rows
    /// remain absent, never zero-filled. All groups use the requested output
    /// carrier; convert verified rescue groups explicitly before this call.
    pub fn merge_rows(row_ids: Vec<u64>, batches: &[&Self]) -> ExecutionResult<Self> {
        let first = batches
            .first()
            .ok_or_else(|| error("row merge needs a typed batch, possibly empty"))?;
        let expected: HashSet<_> = row_ids.iter().copied().collect();
        if expected.len() != row_ids.len() {
            return Err(error("original row identities must be unique"));
        }
        let mut locations = HashMap::new();
        for (group, batch) in batches.iter().enumerate() {
            first.compatible(batch)?;
            if first.columns() != batch.columns() {
                return Err(error("row merge column counts differ"));
            }
            for (row, &id) in batch.ids().iter().enumerate() {
                if !expected.contains(&id) || locations.insert(id, (group, row)).is_some() {
                    return Err(error("merged row identities are duplicated or foreign"));
                }
            }
        }
        let ids: Vec<_> = row_ids
            .into_iter()
            .filter(|id| locations.contains_key(id))
            .collect();
        // Keep immutable Arc-backed source storage while arranging output rows.
        let sources: Vec<_> = batches.iter().map(|batch| batch.data().clone()).collect();
        let columns = first.columns();
        let bits = first.precision_bits();
        macro_rules! merge {
            ($variant:ident) => {{
                let mut values = Vec::new();
                for id in &ids {
                    let (group, row) = locations[id];
                    let Data::$variant(source) = &sources[group] else {
                        unreachable!("validated common arithmetic")
                    };
                    values.extend_from_slice(&source[row * columns..(row + 1) * columns]);
                }
                Self::typed(values, ids, columns, bits)
            }};
        }
        match &sources[0] {
            Data::Double(_) => merge!(Double),
            Data::DoubleDouble(_) => merge!(DoubleDouble),
            Data::Arbitrary(_) => merge!(Arbitrary),
        }
    }
}
