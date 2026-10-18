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
    type Batch = Arc<Batch>;
    type Evaluator = CoefficientEvaluator;
    type ExactEvaluator = ExactPolynomialEvaluator;
    type Schedule = CoefficientPlan;
    // Reducer variants contain sizable inline dependency workspaces. Keep the
    // coordinator family record small, including during generic construction.
    type Reducer = Box<PolynomialPlan>;
    fn evaluator(&self, state: &[u8]) -> RusticolResult<Self::Evaluator> {
        CoefficientEvaluator::new(state, &self.arithmetic, self.precision).map_err(runtime_error)
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
        Batch::from_exact_decimal(
            pairs(values),
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
    fn states(
        momenta: &Arc<Batch>,
        helicities: Vec<i32>,
        kinds: Vec<String>,
        components: usize,
    ) -> RusticolResult<Arc<Batch>> {
        wavefunctions_batch(momenta, helicities, kinds, Some(components))
            .map(Arc::new)
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
        let failed = result
            .statuses()
            .iter()
            .filter(|(_, status)| status != "success")
            .collect::<Vec<_>>();
        if !failed.is_empty() {
            return Err(RusticolError::evaluation(format!(
                "loop reduction failed (original channel ID, status): {failed:?}"
            )));
        }
        Ok((
            Arc::new(result.values().map_err(runtime_error)?),
            result.statuses().to_vec(),
        ))
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
            .into_iter()
            .map(|id| id / rows_per_point as u64)
            .collect::<std::collections::BTreeSet<_>>();
        Err(RusticolError::evaluation(format!(
            "{stage} has nonfinite values at original points {points:?}"
        )))
    }
}
