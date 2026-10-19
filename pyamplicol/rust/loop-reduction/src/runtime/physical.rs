// SPDX-License-Identifier: 0BSD
//! Numerical backend for the shared core process coordinator. All arithmetic,
//! states and reductions delegate to the same runtime primitives as Python.
use super::*;
use rusticol_core::loop_execution::LoopBackend;
use rusticol_core::{RusticolError, RusticolResult};
use std::sync::Arc;

fn runtime_error(value: ExecutionError) -> RusticolError {
    RusticolError::evaluation(value.to_string())
}
fn pairs(values: &[String]) -> Vec<DecimalComplex> {
    values
        .iter()
        .map(|value| [value.clone(), "0".into()])
        .collect()
}

#[derive(Clone)]
pub struct PhysicalBackend {
    arithmetic: String,
    precision: Option<u32>,
    coefficient_engine: String,
    reduction_engine: String,
}
impl PhysicalBackend {
    pub fn new(
        arithmetic: &str,
        precision: Option<u32>,
        coefficient_engine: &str,
        reduction_engine: &str,
    ) -> ExecutionResult<Self> {
        arithmetic_mode(arithmetic, precision)?;
        Ok(Self {
            arithmetic: arithmetic.into(),
            precision,
            coefficient_engine: coefficient_engine.into(),
            reduction_engine: reduction_engine.into(),
        })
    }
    pub fn precision_bits(&self) -> u32 {
        arithmetic_mode(&self.arithmetic, self.precision)
            .expect("validated arithmetic")
            .1
    }
}
impl LoopBackend for PhysicalBackend {
    fn arithmetic(&self) -> &str {
        &self.arithmetic
    }
    type Batch = Arc<Batch>;
    type Evaluator = CoefficientEvaluator;
    type DualBornWeights = DualBornWeights;
    type ExactEvaluator = ExactPolynomialEvaluator;
    type Schedule = CoefficientPlan;
    // Reducer variants contain sizable inline dependency workspaces. Keep the
    // coordinator family record small, including during generic construction.
    type Reducer = Box<PolynomialPlan>;
    fn evaluator(&self, state: &[u8]) -> RusticolResult<Self::Evaluator> {
        CoefficientEvaluator::new(state, &self.arithmetic, self.precision).map_err(runtime_error)
    }
    fn dual_born_weights(
        &self,
        state: &[u8],
        rectangle: Option<&rusticol_core::loop_runtime::RectangularDualWeights>,
        constants: Option<&[u8]>,
        columns: &[[usize; 3]],
    ) -> RusticolResult<Self::DualBornWeights> {
        DualBornWeights::new(
            state,
            rectangle,
            constants,
            columns,
            &self.arithmetic,
            self.precision,
        )
        .map_err(runtime_error)
    }
    fn evaluate_dual_born_weights(
        weights: &mut Self::DualBornWeights,
        input: &Arc<Batch>,
    ) -> RusticolResult<Arc<Batch>> {
        weights
            .evaluate_batch(input)
            .map(Arc::new)
            .map_err(runtime_error)
    }
    fn exact_evaluator(&self, state: &[u8]) -> RusticolResult<Self::ExactEvaluator> {
        ExactPolynomialEvaluator::new(state).map_err(runtime_error)
    }
    fn schedule(
        &self,
        states: Vec<Vec<u8>>,
        inputs: Vec<Vec<usize>>,
        input_count: usize,
        outputs: Vec<usize>,
    ) -> RusticolResult<Self::Schedule> {
        CoefficientPlan::new(
            states,
            inputs,
            input_count,
            outputs,
            &self.arithmetic,
            self.precision,
            &self.coefficient_engine,
            None,
            None,
        )
        .map_err(runtime_error)
    }
    fn reducer(
        &self,
        propagators: usize,
        numerators: [String; 3],
        coefficients: usize,
    ) -> RusticolResult<Self::Reducer> {
        PolynomialPlan::new(
            propagators,
            numerators,
            (0..coefficients).map(|i| format!("c{i}")).collect(),
            &self.arithmetic,
            self.precision,
            &self.reduction_engine,
        )
        .map(Box::new)
        .map_err(runtime_error)
    }
    fn ingress(
        &self,
        values: &[String],
        rows: usize,
        columns: usize,
        ids: Vec<u64>,
    ) -> RusticolResult<Arc<Batch>> {
        self.ingress_complex(&pairs(values), rows, columns, ids)
    }
    fn ingress_complex(
        &self,
        values: &[[String; 2]],
        rows: usize,
        columns: usize,
        ids: Vec<u64>,
    ) -> RusticolResult<Arc<Batch>> {
        Batch::from_exact_decimal(
            values.to_vec(),
            rows,
            columns,
            &self.arithmetic,
            self.precision,
            Some(ids),
        )
        .map(Arc::new)
        .map_err(runtime_error)
    }
    fn exact(
        &self,
        evaluator: &mut Self::ExactEvaluator,
        values: &[String],
        rows: usize,
        ids: Vec<u64>,
    ) -> RusticolResult<Arc<Batch>> {
        evaluator
            .evaluate_decimal(
                pairs(values),
                rows,
                &self.arithmetic,
                self.precision,
                Some(ids),
            )
            .map(Arc::new)
            .map_err(runtime_error)
    }
    fn evaluate(evaluator: &mut Self::Evaluator, input: &Arc<Batch>) -> RusticolResult<Arc<Batch>> {
        evaluator
            .evaluate_batch(input)
            .map(Arc::new)
            .map_err(runtime_error)
    }
    fn execute(schedule: &mut Self::Schedule, input: &Arc<Batch>) -> RusticolResult<Arc<Batch>> {
        schedule
            .evaluate_batch(input)
            .map(Arc::new)
            .map_err(runtime_error)
    }
    fn states_partial(
        momenta: &Arc<Batch>,
        helicities: Vec<i32>,
        kinds: Vec<String>,
        components: usize,
    ) -> RusticolResult<(Arc<Batch>, Vec<(u64, String)>)> {
        wavefunctions_partial_batch(momenta, helicities, kinds, Some(components))
            .map(|(values, statuses)| (Arc::new(values), statuses))
            .map_err(runtime_error)
    }
    fn reduce(
        reducer: &mut Self::Reducer,
        shifts: &Arc<Batch>,
        masses: &Arc<Batch>,
        scales: &Arc<Batch>,
        coefficients: &Arc<Batch>,
        matrix: &Arc<Batch>,
    ) -> RusticolResult<(Arc<Batch>, Vec<(u64, String)>)> {
        let result = reducer
            .evaluate_batch(shifts, masses, scales, coefficients, Some(matrix))
            .map_err(runtime_error)?;
        Ok((
            Arc::new(result.successful_values()),
            result.statuses().to_vec(),
        ))
    }
    fn nonfinite_rows(batch: &Arc<Batch>) -> Vec<u64> {
        batch.nonfinite_rows()
    }
    fn rows(batch: &Arc<Batch>) -> usize {
        batch.rows()
    }
    fn row_ids(batch: &Arc<Batch>) -> &[u64] {
        batch.ids()
    }
    fn columns(batch: &Arc<Batch>) -> usize {
        batch.columns()
    }
    fn select(batch: &Arc<Batch>, columns: Vec<usize>) -> RusticolResult<Arc<Batch>> {
        batch
            .take_columns(columns)
            .map(Arc::new)
            .map_err(runtime_error)
    }
    fn gather(batch: &Arc<Batch>, rows: Vec<usize>, ids: Vec<u64>) -> RusticolResult<Arc<Batch>> {
        batch
            .take_rows(rows, Some(ids))
            .map(Arc::new)
            .map_err(runtime_error)
    }
    fn reshape(
        batch: &Arc<Batch>,
        rows: usize,
        columns: usize,
        ids: Vec<u64>,
    ) -> RusticolResult<Arc<Batch>> {
        batch
            .reshape_rows(rows, columns, ids)
            .map(Arc::new)
            .map_err(runtime_error)
    }
    fn concat(batches: &[&Arc<Batch>]) -> RusticolResult<Arc<Batch>> {
        Batch::concat_columns(&batches.iter().map(|b| b.as_ref()).collect::<Vec<_>>())
            .map(Arc::new)
            .map_err(runtime_error)
    }
    fn require_finite(
        batch: &Arc<Batch>,
        stage: &str,
        rows_per_point: usize,
    ) -> RusticolResult<()> {
        let failed = batch.nonfinite_rows();
        if failed.is_empty() {
            return Ok(());
        }
        let points = failed
            .iter()
            .map(|id| id / rows_per_point as u64)
            .collect::<std::collections::BTreeSet<_>>();
        let helicities = if rows_per_point > 1 {
            let lanes = failed
                .iter()
                .map(|id| (id / rows_per_point as u64, id % rows_per_point as u64))
                .collect::<Vec<_>>();
            format!("; (point, helicity) {lanes:?}")
        } else {
            String::new()
        };
        Err(RusticolError::evaluation(format!(
            "{stage} has nonfinite values at original points {points:?}{helicities}"
        )))
    }
}
