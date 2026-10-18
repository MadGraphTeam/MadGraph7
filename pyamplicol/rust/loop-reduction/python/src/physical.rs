// SPDX-License-Identifier: 0BSD
//! Experimental complete native execution of validated legacy loop PACBIN plans.
use crate::{batch::NativeBatch, error};
use pyo3::prelude::*;
use rusticol_core::loop_execution::{CompleteLoopResult, LoopExecutor, LoopInputs};
use rusticol_core::loop_runtime::{LoadedLoopRuntime, load_loop_runtime_pacbin};
use rusticol_loop_reduction::runtime::{Batch, physical::PhysicalBackend};
use std::sync::Arc;

#[pyclass]
struct NativeLoopProgram {
    executor: LoopExecutor<PhysicalBackend>,
    bits: u32,
    arithmetic: String,
}
impl NativeLoopProgram {
    fn bind(
        loaded: Arc<LoadedLoopRuntime>,
        arithmetic: &str,
        precision: Option<u32>,
        coefficient_engine: &str,
        reduction_engine: &str,
    ) -> PyResult<Self> {
        let backend =
            PhysicalBackend::new(arithmetic, precision, coefficient_engine, reduction_engine)
                .map_err(error)?;
        let bits = backend.precision_bits();
        let executor = LoopExecutor::new(loaded, backend).map_err(error)?;
        Ok(Self {
            executor,
            bits,
            arithmetic: arithmetic.into(),
        })
    }
}
#[pymethods]
impl NativeLoopProgram {
    #[new]
    #[pyo3(signature=(path, arithmetic="double", precision=None, coefficient_engine="interpreted", reduction_engine="generic"))]
    fn new(
        py: Python<'_>,
        path: &str,
        arithmetic: &str,
        precision: Option<u32>,
        coefficient_engine: &str,
        reduction_engine: &str,
    ) -> PyResult<Self> {
        py.detach(|| {
            Self::bind(
                Arc::new(load_loop_runtime_pacbin(path).map_err(error)?),
                arithmetic,
                precision,
                coefficient_engine,
                reduction_engine,
            )
        })
    }
    #[pyo3(signature=(arithmetic="double", precision=None, coefficient_engine="interpreted", reduction_engine="generic"))]
    fn rebind(
        &self,
        py: Python<'_>,
        arithmetic: &str,
        precision: Option<u32>,
        coefficient_engine: &str,
        reduction_engine: &str,
    ) -> PyResult<Self> {
        py.detach(|| {
            Self::bind(
                self.executor.loaded().clone(),
                arithmetic,
                precision,
                coefficient_engine,
                reduction_engine,
            )
        })
    }
    #[getter]
    fn precision_bits(&self) -> u32 {
        self.bits
    }
    #[getter]
    fn arithmetic(&self) -> &str {
        &self.arithmetic
    }
    #[getter]
    fn r2_strategy(&self) -> &str {
        &self.executor.loaded().plan().r2_strategy
    }
    fn metadata_json(&self) -> PyResult<String> {
        self.executor.loaded().metadata_json().map_err(error)
    }
    #[pyo3(signature=(momenta, scales_squared, model_values, point_ids=None))]
    fn evaluate(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        model_values: Vec<String>,
        point_ids: Option<Vec<u64>>,
    ) -> PyResult<NativeCompleteLoopResult> {
        let legs = self.executor.loaded().plan().state_legs.len();
        if momenta.iter().any(|point| point.len() != legs) {
            return Err(error("physical momentum leg shape differs"));
        }
        let ids = point_ids.unwrap_or_else(|| (0..momenta.len() as u64).collect());
        if ids.len() != momenta.len() {
            return Err(error("physical point identity count differs"));
        }
        let input = LoopInputs {
            momenta: momenta.into_iter().flatten().flatten().collect(),
            scales_squared,
            model_values,
            point_ids: ids,
        };
        py.detach(|| self.executor.evaluate(input))
            .map(NativeCompleteLoopResult)
            .map_err(error)
    }
}

#[pyclass]
struct NativeCompleteLoopResult(CompleteLoopResult<Arc<Batch>>);
#[pymethods]
impl NativeCompleteLoopResult {
    #[getter]
    fn point_ids(&self) -> Vec<u64> {
        self.0.point_ids.clone()
    }
    #[getter]
    fn r2_strategy(&self) -> &str {
        &self.0.r2_strategy
    }
    #[getter]
    fn born(&self) -> NativeBatch {
        NativeBatch(self.0.born.as_ref().clone())
    }
    #[getter]
    fn component_batches(&self) -> Vec<NativeBatch> {
        self.0
            .components
            .iter()
            .map(|b| NativeBatch(b.as_ref().clone()))
            .collect()
    }
    #[getter]
    fn totals(&self) -> NativeBatch {
        NativeBatch(self.0.totals.as_ref().clone())
    }
    #[getter]
    fn statuses(&self) -> Vec<Vec<(u64, String)>> {
        self.0.statuses.clone()
    }
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeLoopProgram>()?;
    module.add_class::<NativeCompleteLoopResult>()
}
