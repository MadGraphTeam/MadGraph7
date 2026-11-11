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
    /// Per requested point the relative uncertainty of a `BestEffort`
    /// point's value (NaN otherwise; m-lit ux-besteffort).
    pub uncertainties: Vec<f64>,
    /// Double-double and arbitrary-precision evaluations: the exact values
    /// (the Python API's `Decimal` results) as `[real, imaginary]` decimal
    /// literals in the layout of `born` and `virtual_values` (absent poles
    /// `None`); `born` and `virtual_values` then hold their nearest binary64
    /// values.
    pub decimal: Option<DecimalComplexValues>,
}

/// The exact complex values of an extended-precision correlated
/// evaluation, at `digits` working digits of `arithmetic`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecimalComplexValues {
    pub arithmetic: &'static str,
    pub digits: u32,
    pub born: Vec<[String; 2]>,
    #[serde(rename = "virtual")]
    pub virtual_values: Vec<Option<[String; 2]>>,
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
            born_orders: self
                .observables
                .as_ref()
                .map_or(self.shape.born_orders, |axes| axes.born_orders),
            virtual_orders: self.shape.virtual_orders,
            epsilon_count: 3,
        }
    }
    /// All saved directed histories in one native batched call. The returned
    /// carrier is binary64.
    pub fn evaluate_correlated(
        &mut self,
        momenta: &[f64],
        point_ids: &[u64],
        scales: &[f64],
        strategy: Option<&str>,
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
        )
    }
    pub fn evaluate_correlated_decimal(
        &mut self,
        momenta: Vec<String>,
        point_ids: Vec<u64>,
        scales: Vec<String>,
        strategy: Option<&str>,
    ) -> RusticolResult<CorrelatedEvaluation> {
        let inputs = self.prepare_inputs(momenta.clone(), point_ids.clone(), scales.clone())?;
        let base = self.evaluate_correlated_inputs(inputs, strategy)?;
        let call = crate::best_effort::CallInputs {
            momenta: &momenta,
            point_ids: &point_ids,
            scales: &scales,
            selection: None,
            sources: None,
        };
        self.with_policy(base, call, |runtime, subset| {
            let inputs = runtime.prepare_inputs(subset.momenta, subset.point_ids, subset.scales)?;
            runtime.evaluate_correlated_inputs(inputs, strategy)
        })
    }
    /// Additive binary64 source ingress; the old method always uses ordinary states.
    pub fn evaluate_correlated_with_sources(
        &mut self,
        momenta: &[f64],
        point_ids: &[u64],
        scales: &[f64],
        sources: SpinSources<'_>,
        strategy: Option<&str>,
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
        )
    }
    /// Exact per-point complex pairs.
    pub fn evaluate_correlated_decimal_with_sources(
        &mut self,
        momenta: Vec<String>,
        point_ids: Vec<u64>,
        scales: Vec<String>,
        sources: BTreeMap<usize, Vec<[String; 2]>>,
        strategy: Option<&str>,
    ) -> RusticolResult<CorrelatedEvaluation> {
        let mut inputs = self.prepare_inputs(momenta.clone(), point_ids.clone(), scales.clone())?;
        inputs.spin_sources = sources.clone();
        let base = self.evaluate_correlated_inputs(inputs, strategy)?;
        let call = crate::best_effort::CallInputs {
            momenta: &momenta,
            point_ids: &point_ids,
            scales: &scales,
            selection: None,
            sources: Some(&sources),
        };
        self.with_policy(base, call, |runtime, subset| {
            let mut inputs =
                runtime.prepare_inputs(subset.momenta, subset.point_ids, subset.scales)?;
            inputs.spin_sources = subset.sources.unwrap_or_default();
            runtime.evaluate_correlated_inputs(inputs, strategy)
        })
    }
    /// Every stored colour history (`born` first, the declared histories in
    /// their order) in one batch, with the spin sources of `inputs` (one
    /// declared spin class; none: the physical states): the Python API's
    /// `evaluate_correlated_many` with one request per history, the same
    /// spin vectors in each, in the runtime's arithmetic and pole selection.
    fn evaluate_correlated_inputs(
        &mut self,
        inputs: LoopInputs,
        strategy: Option<&str>,
    ) -> RusticolResult<CorrelatedEvaluation> {
        if let Some(strategy) = strategy {
            v4_r2_strategy(strategy)?;
        }
        let axes = self.observables.clone().map_err(|reason| {
            RusticolError::compatibility(format!("{NO_OBSERVABLES}: {reason}"))
        })?;
        let shape = CorrelatedShape {
            born_orders: axes.born_orders,
            ..self.correlated_shape()
        };
        let (poles, arithmetic) = (self.poles, self.arithmetic);
        let reduction = self.reduce_on(reduction_policy::RESOLVED)?;
        let stability = Monitoring {
            checks: None,
            requested: self.stability_checks().enabled(),
        }
        .not_run_resolved();
        let (inputs, kinematics) = self.improved_inputs(inputs)?;
        let (momenta, scales) = exact_inputs(&inputs, self.shape.external_count)?;
        let points = inputs.point_ids.len();
        let spin_legs: Vec<usize> = inputs.spin_sources.keys().copied().collect();
        let spins = spin_rows(&inputs.spin_sources, points)?;
        let Engine::V4(native) = &mut self.engine;
        let evaluation = match arithmetic {
            Arithmetic::Double => {
                let (found, fallbacks) = with_fallback(
                    native,
                    &reduction,
                    momenta.len(),
                    arithmetic,
                    |native, indices| {
                        native.evaluate_columns(
                            "correlated",
                            &pick(&momenta, indices),
                            &pick(&scales, indices),
                            &pick_rows(&spins, indices),
                            poles.v2(),
                        )
                    },
                )?;
                CorrelatedEvaluation::from_columns(
                    inputs, shape, poles, arithmetic, found, &reduction, &fallbacks, stability,
                )?
            }
            _ => {
                let (found, fallbacks) = with_fallback(
                    native,
                    &reduction,
                    momenta.len(),
                    arithmetic,
                    // The fallback's points run with the guard's check of
                    // the call's outputs (m-lit lit-fallback-guard).
                    |native, indices| match indices {
                        None => unguarded(native.evaluate_columns_extended(
                            "correlated",
                            &momenta,
                            &scales,
                            &spins,
                            poles.v2(),
                            arithmetic.v2(),
                        )),
                        Some(_) => native.evaluate_columns_guarded(
                            "correlated",
                            &pick(&momenta, indices),
                            &pick(&scales, indices),
                            &pick_rows(&spins, indices),
                            poles.v2(),
                            arithmetic.v2(),
                        ),
                    },
                )?;
                let found = found.into_iter().map(|(point, _)| point).collect();
                CorrelatedEvaluation::from_columns(
                    inputs, shape, poles, arithmetic, found, &reduction, &fallbacks, stability,
                )?
            }
        };
        let mut evaluation = CorrelatedEvaluation {
            history_ids: self.history_ids.clone(),
            spin_legs,
            ..evaluation
        };
        with_kinematics(&mut evaluation.diagnostics_json, kinematics)?;
        Ok(evaluation)
    }
}

/// The spin rows of the points at `indices` ([`pick`]; none without spin
/// sources).
fn pick_rows<'a, T: Clone>(rows: &'a [T], indices: Option<&[usize]>) -> std::borrow::Cow<'a, [T]> {
    if rows.is_empty() {
        std::borrow::Cow::Borrowed(rows)
    } else {
        pick(rows, indices)
    }
}

/// Per point the exact spin sources of every leg: `sources` maps a one-based
/// leg to its four complex components per point (`4 * points` pairs, point
/// major). A leg 0 or a vector of another length (e.g. one broadcast row
/// passed to the per-point decimal API) is an invalid argument, never an
/// out-of-bounds read.
fn spin_rows(
    sources: &BTreeMap<usize, Vec<[String; 2]>>,
    points: usize,
) -> RusticolResult<Vec<BTreeMap<usize, [(Rational, Rational); 4]>>> {
    use rusticol_core::loop_v2::dress::parse_rational;

    if sources.is_empty() {
        return Ok(Vec::new());
    }
    let width = product(&[points, 4])?;
    for (leg, values) in sources {
        if *leg == 0 {
            return Err(invalid("spin source legs must be unique one-based labels"));
        }
        if values.len() != width {
            return Err(invalid(format!(
                "the spin source of leg {leg} holds {} complex components; \
                 {points} points need 4 per point ({width})",
                values.len()
            )));
        }
    }
    (0..points)
        .map(|point| {
            sources
                .iter()
                .map(|(leg, values)| {
                    let component = |k: usize| -> RusticolResult<_> {
                        let [re, im] = &values[4 * point + k];
                        Ok((parse_rational(re)?, parse_rational(im)?))
                    };
                    Ok((
                        *leg,
                        [component(0)?, component(1)?, component(2)?, component(3)?],
                    ))
                })
                .collect::<RusticolResult<BTreeMap<_, _>>>()
        })
        .collect()
}

/// A complex output (`complex(value)` / the Python API's
/// `CorrelatedValue`) and, for the extended carriers, its literals.
#[derive(Default)]
struct Complexes {
    values: Vec<Complex64>,
    literals: Vec<Option<[String; 2]>>,
}

impl Complexes {
    fn push<N: Output>(&mut self, value: Option<&C<N>>) {
        match value {
            Some(z) => {
                self.values.push(Complex64 {
                    real: z.re.double(),
                    imag: z.im.double(),
                });
                self.literals
                    .push(z.re.literal().zip(z.im.literal()).map(|(re, im)| [re, im]));
            }
            None => {
                self.values.push(Complex64 {
                    real: f64::NAN,
                    imag: f64::NAN,
                });
                self.literals.push(None);
            }
        }
    }
}

impl CorrelatedEvaluation {
    fn from_columns<N: Output>(
        inputs: LoopInputs,
        shape: CorrelatedShape,
        poles: Poles,
        arithmetic: Arithmetic,
        points: Vec<ColumnPoint<N>>,
        reduction: &Choice,
        fallbacks: &Fallbacks,
        stability: serde_json::Value,
    ) -> RusticolResult<Self> {
        let (successful_point_ids, statuses, failures) = statuses_of(&inputs.point_ids, &points);
        let mut born = Complexes::default();
        let mut virtual_values = Complexes::default();
        for columns in points.into_iter().flatten() {
            if columns.len() != shape.histories {
                return Err(invalid(
                    "the stored correlated columns are not the histories",
                ));
            }
            for column in &columns {
                for value in &column.born {
                    born.push(Some(value));
                }
                for laurent in &column.virtual_values {
                    for k in 0..shape.epsilon_count {
                        virtual_values.push(laurent.get(k));
                    }
                }
            }
        }
        let diagnostics_json = diagnostics(
            &inputs.point_ids,
            &successful_point_ids,
            &failures,
            poles,
            arithmetic,
            reduction,
            fallbacks,
            stability,
        )?;
        let decimal = (arithmetic != Arithmetic::Double).then(|| DecimalComplexValues {
            arithmetic: arithmetic.name(),
            digits: arithmetic.working_digits(),
            born: born
                .literals
                .iter()
                .cloned()
                .map(|value| value.expect("a Born value"))
                .collect(),
            virtual_values: virtual_values.literals.clone(),
        });
        Ok(Self {
            requested_point_ids: inputs.point_ids,
            successful_point_ids,
            statuses,
            born: born.values,
            virtual_values: virtual_values.values,
            shape,
            history_ids: vec![],
            spin_legs: vec![],
            diagnostics_json,
            uncertainties: Vec::new(),
            decimal,
        })
    }
}

impl CorrelatedEvaluation {
    /// The `decimal` record as JSON (`null` for a binary64 evaluation).
    pub fn decimal_json(&self) -> RusticolResult<String> {
        serde_json::to_string(&self.decimal).map_err(serialization)
    }
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

    fn literal(values: &[&str]) -> Vec<[String; 2]> {
        values
            .iter()
            .map(|value| [value.to_string(), "0".to_string()])
            .collect()
    }

    #[test]
    fn spin_sources_need_four_components_per_point() {
        // Two points: one row per point is read point by point.
        let rows = BTreeMap::from([(3, literal(&["1", "2", "3", "4", "5", "6", "7", "8"]))]);
        let found = spin_rows(&rows, 2).unwrap();
        assert_eq!(found.len(), 2);
        assert_eq!(found[1][&3][0].0, Rational::from(5));
        assert_eq!(found[1][&3][3].0, Rational::from(8));
        // A broadcast row (4 pairs) for 2 points, a short or a long vector,
        // and leg 0 are typed invalid arguments, not panics.
        for (leg, values) in [
            (3, literal(&["1", "2", "3", "4"])),
            (3, literal(&["1", "2", "3", "4", "5", "6", "7"])),
            (3, literal(&["0"; 12])),
            (0, literal(&["0"; 8])),
        ] {
            let error = spin_rows(&BTreeMap::from([(leg, values)]), 2).unwrap_err();
            assert_eq!(
                error.kind(),
                rusticol_core::RusticolErrorKind::InvalidArgument
            );
        }
        let message = spin_rows(&BTreeMap::from([(4, literal(&["1"; 4]))]), 2)
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("leg 4 holds 4") && message.contains("(8)"),
            "{message}"
        );
        // No sources: the physical states, whatever the point count.
        assert!(spin_rows(&BTreeMap::new(), 3).unwrap().is_empty());
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
