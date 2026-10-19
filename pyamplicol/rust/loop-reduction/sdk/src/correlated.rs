// SPDX-License-Identifier: 0BSD
//! Grouped complex output: histories are a separate axis, never summed implicitly.
use super::*;

/// Explicit C-compatible pair; no C/C++/Fortran complex ABI assumptions.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex64 {
    pub real: f64,
    pub imag: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CorrelatedShape {
    pub histories: usize,
    pub born_orders: usize,
    pub virtual_orders: usize,
    pub epsilon_count: usize,
}

/// Owned buffers in point/history/order/(epsilon) order. Only successful points
/// have numerical rows. All saved histories are evaluated in catalogue order.
pub struct CorrelatedEvaluation {
    pub requested_point_ids: Vec<u64>,
    pub successful_point_ids: Vec<u64>,
    pub statuses: Vec<PointStatus>,
    pub born: Vec<Complex64>,
    pub virtual_values: Vec<Complex64>,
    pub shape: CorrelatedShape,
    pub history_ids: Vec<String>,
    pub diagnostics_json: String,
}

impl Runtime {
    /// Saved history IDs in the same order as the complex output axis. This
    /// query does not relabel internal colour-operation leg indices.
    pub fn correlation_history_ids(&self) -> &[String] {
        &self.history_ids
    }
    pub fn correlation_histories_json(&self) -> RusticolResult<String> {
        serde_json::to_string(&self.history_ids).map_err(serialization)
    }
    pub fn correlated_shape(&self) -> CorrelatedShape {
        CorrelatedShape {
            histories: self.history_ids.len(),
            born_orders: self.shape.born_orders,
            virtual_orders: self.shape.virtual_orders,
            epsilon_count: 3,
        }
    }
    /// All saved directed histories in one native batched call. The returned
    /// carrier is binary64; optional verification recomputes original exact
    /// input snapshots in the configured higher arithmetic.
    pub fn evaluate_correlated(
        &mut self,
        momenta: &[f64],
        point_ids: &[u64],
        scales: &[f64],
        strategy: Option<&str>,
        policy: Option<&VerificationPolicy>,
    ) -> RusticolResult<CorrelatedEvaluation> {
        self.evaluate_correlated_decimal(
            momenta
                .iter()
                .map(|&v| exact_double(v))
                .collect::<RusticolResult<_>>()?,
            point_ids.to_vec(),
            scales
                .iter()
                .map(|&v| exact_double(v))
                .collect::<RusticolResult<_>>()?,
            strategy,
            policy,
        )
    }
    pub fn evaluate_correlated_decimal(
        &mut self,
        momenta: Vec<String>,
        point_ids: Vec<u64>,
        scales: Vec<String>,
        strategy: Option<&str>,
        policy: Option<&VerificationPolicy>,
    ) -> RusticolResult<CorrelatedEvaluation> {
        let inputs = self.prepare_inputs(momenta, point_ids, scales)?;
        let (result, verification) = match policy {
            Some(policy) => {
                let output = self
                    .native
                    .evaluate_directed_verified(inputs, strategy, policy)?;
                (output.result, output.diagnostics)
            }
            None => (
                self.native.evaluate_directed(inputs, strategy, true)?,
                vec![],
            ),
        };
        CorrelatedEvaluation::from_native(
            result,
            verification,
            self.correlated_shape(),
            &self.history_ids,
        )
    }
}

fn pack_complex(
    batch: &Batch,
    points: usize,
    histories: usize,
    orders: usize,
    epsilon: usize,
) -> RusticolResult<Vec<Complex64>> {
    let Data::Double(values) = batch.data() else {
        return Err(invalid("standalone output is not binary64"));
    };
    if batch.rows() != points || batch.columns() != product(&[histories, orders, epsilon])? {
        return Err(invalid(
            "native correlated axes differ from history/order catalogue",
        ));
    }
    values
        .iter()
        .map(|value| {
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(invalid("correlated result is not finite complex"));
            }
            Ok(Complex64 {
                real: value.re,
                imag: value.im,
            })
        })
        .collect()
}
impl CorrelatedEvaluation {
    fn from_native(
        result: CompleteLoopResult<Arc<Batch>>,
        verification: Vec<VerificationDiagnostic>,
        shape: CorrelatedShape,
        history_ids: &[String],
    ) -> RusticolResult<Self> {
        let directed = result
            .directed
            .as_ref()
            .ok_or_else(|| invalid("native result is not directed"))?;
        if directed.history_ids != history_ids || shape.histories != history_ids.len() {
            return Err(invalid("native directed history catalogue differs"));
        }
        let points = result.point_ids.len();
        let born = pack_complex(&result.born, points, shape.histories, shape.born_orders, 1)?;
        let virtual_values = pack_complex(
            &result.components[0],
            points,
            shape.histories,
            shape.virtual_orders,
            3,
        )?;
        let (statuses, diagnostics_json) = diagnostics(&result, &verification)?;
        Ok(Self {
            requested_point_ids: result.requested_point_ids,
            successful_point_ids: result.point_ids,
            statuses,
            born,
            virtual_values,
            shape,
            history_ids: history_ids.to_vec(),
            diagnostics_json,
        })
    }
    pub fn histories_json(&self) -> RusticolResult<String> {
        serde_json::to_string(&self.history_ids).map_err(serialization)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusticol_loop_reduction::Complex;
    #[test]
    fn complex_axes_keep_unequal_orders_and_labels_separate() {
        for (orders, epsilon) in [(2, 1), (3, 3)] {
            let values: Vec<_> = (0..2 * 4 * orders * epsilon)
                .map(|i| Complex::new(-(i as f64) - 1., i as f64 + 0.25))
                .collect();
            let batch = Batch::typed(values.clone(), vec![0, 1], 4 * orders * epsilon, 53).unwrap();
            let packed = pack_complex(&batch, 2, 4, orders, epsilon).unwrap();
            for p in 0..2 {
                for h in 0..4 {
                    for o in 0..orders {
                        for e in 0..epsilon {
                            let i = ((p * 4 + h) * orders + o) * epsilon + e;
                            assert_eq!(
                                packed[i],
                                Complex64 {
                                    real: values[i].re,
                                    imag: values[i].im
                                }
                            );
                        }
                    }
                }
            }
            assert!(pack_complex(&batch, 2, 3, orders, epsilon).is_err());
            let empty =
                Batch::typed(Vec::<Complex<f64>>::new(), vec![], 4 * orders * epsilon, 53).unwrap();
            assert!(
                pack_complex(&empty, 0, 4, orders, epsilon)
                    .unwrap()
                    .is_empty()
            );
        }
    }
}
