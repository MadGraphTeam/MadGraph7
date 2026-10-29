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

/// Literal complex sources in row/selected-leg/component order. `legs` use
/// one-based physical labels. One row broadcasts; otherwise one row per point.
/// No component is conjugated, normalized or projected by this transport.
#[derive(Clone, Copy)]
pub struct SpinSources<'a> {
    pub legs: &'a [usize],
    pub values: &'a [Complex64],
}
impl SpinSources<'_> {
    fn exact(&self, points: usize) -> RusticolResult<BTreeMap<usize, Vec<[String; 2]>>> {
        let width = product(&[self.legs.len(), 4])?;
        if self.values.len() != width && self.values.len() != product(&[points, width])? {
            return Err(invalid(
                "spin vectors require one row or one row per physical point",
            ));
        }
        let values = self
            .values
            .iter()
            .map(|value| Ok([exact_double(value.real)?, exact_double(value.imag)?]))
            .collect::<RusticolResult<Vec<_>>>()?;
        let mut sources = BTreeMap::new();
        for (column, &leg) in self.legs.iter().enumerate() {
            if leg == 0 || sources.contains_key(&leg) {
                return Err(invalid("spin source legs must be unique one-based labels"));
            }
            let source = (0..points)
                .flat_map(|row| {
                    let base = if values.len() == width {
                        4 * column
                    } else {
                        row * width + 4 * column
                    };
                    values[base..base + 4].iter().cloned()
                })
                .collect();
            sources.insert(leg, source);
        }
        Ok(sources)
    }
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
    pub spin_legs: Vec<usize>,
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
    /// Declared joint sets, with the same one-based source labels as input.
    pub fn spin_correlation_classes(&self) -> &[Vec<usize>] {
        &self.spin_classes
    }
    pub fn spin_correlation_classes_json(&self) -> RusticolResult<String> {
        serde_json::to_string(&self.spin_classes).map_err(serialization)
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
        self.evaluate_correlated_inputs(inputs, strategy, policy)
    }
    /// Additive binary64 source ingress; the old method always uses ordinary states.
    pub fn evaluate_correlated_with_sources(
        &mut self,
        momenta: &[f64],
        point_ids: &[u64],
        scales: &[f64],
        sources: SpinSources<'_>,
        strategy: Option<&str>,
        policy: Option<&VerificationPolicy>,
    ) -> RusticolResult<CorrelatedEvaluation> {
        self.evaluate_correlated_decimal_with_sources(
            momenta
                .iter()
                .map(|&v| exact_double(v))
                .collect::<RusticolResult<_>>()?,
            point_ids.to_vec(),
            scales
                .iter()
                .map(|&v| exact_double(v))
                .collect::<RusticolResult<_>>()?,
            sources.exact(point_ids.len())?,
            strategy,
            policy,
        )
    }
    /// Exact per-point complex pairs are retained for optional precision replay.
    pub fn evaluate_correlated_decimal_with_sources(
        &mut self,
        momenta: Vec<String>,
        point_ids: Vec<u64>,
        scales: Vec<String>,
        sources: BTreeMap<usize, Vec<[String; 2]>>,
        strategy: Option<&str>,
        policy: Option<&VerificationPolicy>,
    ) -> RusticolResult<CorrelatedEvaluation> {
        let mut inputs = self.prepare_inputs(momenta, point_ids, scales)?;
        inputs.spin_sources = sources;
        self.evaluate_correlated_inputs(inputs, strategy, policy)
    }
    /// Correlated (directed colour and spin) virtuals were served by the
    /// removed legacy engine's runtime; plan v4 SDK runtimes serve ordinary
    /// virtuals, and correlated ones go through the Python API.
    fn evaluate_correlated_inputs(
        &mut self,
        _inputs: LoopInputs,
        _strategy: Option<&str>,
        _policy: Option<&VerificationPolicy>,
    ) -> RusticolResult<CorrelatedEvaluation> {
        Err(invalid(V4_ONLY_ORDINARY))
    }
}

impl CorrelatedEvaluation {
    pub fn spin_legs_json(&self) -> RusticolResult<String> {
        serde_json::to_string(&self.spin_legs).map_err(serialization)
    }
    pub fn histories_json(&self) -> RusticolResult<String> {
        serde_json::to_string(&self.history_ids).map_err(serialization)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusticol_loop_reduction::Complex;
    use rusticol_loop_reduction::runtime::{Batch, Data};

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
