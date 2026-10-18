// SPDX-License-Identifier: 0BSD
//! Experimental complete native execution of validated legacy loop PACBIN plans.
use crate::{batch::NativeBatch, error};
use pyo3::prelude::*;
use rusticol_core::loop_execution::{CompleteLoopResult, LoopExecutor, LoopInputs};
use rusticol_core::loop_runtime::{LoadedLoopRuntime, load_loop_runtime_pacbin};
use rusticol_core::source_bindings::{SourceOverrides, SourceOwner, SourceSnapshot};
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
        parameters: Option<&SourceOverrides>,
    ) -> PyResult<Self> {
        let backend =
            PhysicalBackend::new(arithmetic, precision, coefficient_engine, reduction_engine)
                .map_err(error)?;
        let bits = backend.precision_bits();
        let executor = LoopExecutor::with_parameters(loaded, backend, parameters).map_err(error)?;
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
    #[pyo3(signature=(path, arithmetic="double", precision=None, coefficient_engine="interpreted", reduction_engine="generic", parameters=None))]
    fn new(
        py: Python<'_>,
        path: &str,
        arithmetic: &str,
        precision: Option<u32>,
        coefficient_engine: &str,
        reduction_engine: &str,
        parameters: Option<SourceOverrides>,
    ) -> PyResult<Self> {
        py.detach(|| {
            Self::bind(
                Arc::new(load_loop_runtime_pacbin(path).map_err(error)?),
                arithmetic,
                precision,
                coefficient_engine,
                reduction_engine,
                parameters.as_ref(),
            )
        })
    }
    #[pyo3(signature=(arithmetic="double", precision=None, coefficient_engine="interpreted", reduction_engine="generic", parameters=None))]
    fn rebind(
        &self,
        py: Python<'_>,
        arithmetic: &str,
        precision: Option<u32>,
        coefficient_engine: &str,
        reduction_engine: &str,
        parameters: Option<SourceOverrides>,
    ) -> PyResult<Self> {
        let captured = self
            .executor
            .source_owner()
            .map(|owner| owner.snapshot())
            .transpose()
            .map_err(error)?;
        let mut overrides = captured
            .as_ref()
            .map(|snapshot| snapshot.overrides().clone());
        if let Some(changes) = parameters {
            overrides
                .get_or_insert_with(SourceOverrides::new)
                .extend(changes);
        }
        py.detach(|| {
            Self::bind(
                self.executor.loaded().clone(),
                arithmetic,
                precision,
                coefficient_engine,
                reduction_engine,
                overrides.as_ref(),
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
    #[getter]
    fn source_state(&self) -> Option<NativeSourceState> {
        self.executor.source_owner().map(|owner| NativeSourceState {
            owner: owner.clone(),
            backend: self.executor.backend().clone(),
        })
    }
    #[pyo3(signature=(momenta, scales_squared, model_values=None, point_ids=None, snapshot=None))]
    fn evaluate(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        model_values: Option<Vec<String>>,
        point_ids: Option<Vec<u64>>,
        snapshot: Option<&NativeSourceSnapshot>,
    ) -> PyResult<NativeCompleteLoopResult> {
        if self.executor.source_owner().is_some() && model_values.is_some() {
            return Err(error(
                "source model parameters require named owner updates, not positional model_values",
            ));
        }
        let snapshot = snapshot.map(|value| value.0.clone());
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
            model_values: model_values.unwrap_or_default(),
            point_ids: ids,
        };
        py.detach(|| self.executor.evaluate_with_snapshot(input, snapshot))
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
    fn source_revision(&self) -> Option<u64> {
        self.0.source_revision
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
/// Separate owner handle: updates do not borrow mutable process workspaces.
#[pyclass]
struct NativeSourceState {
    owner: Arc<SourceOwner<PhysicalBackend>>,
    backend: PhysicalBackend,
}
#[pymethods]
impl NativeSourceState {
    fn snapshot(&self) -> PyResult<NativeSourceSnapshot> {
        self.owner
            .snapshot()
            .map(NativeSourceSnapshot)
            .map_err(error)
    }
    fn update(
        &self,
        py: Python<'_>,
        parameters: SourceOverrides,
    ) -> PyResult<NativeSourceSnapshot> {
        py.detach(|| self.owner.update(&self.backend, &parameters))
            .map(NativeSourceSnapshot)
            .map_err(error)
    }
}
#[pyclass(frozen)]
struct NativeSourceSnapshot(Arc<SourceSnapshot<Arc<Batch>>>);
#[pymethods]
impl NativeSourceSnapshot {
    #[getter]
    fn revision(&self) -> u64 {
        self.0.revision()
    }
    #[getter]
    fn overrides(&self) -> SourceOverrides {
        self.0.overrides().clone()
    }
    #[getter]
    fn arguments(&self) -> NativeBatch {
        NativeBatch(self.0.arguments().as_ref().clone())
    }
    #[getter]
    fn values(&self) -> NativeBatch {
        NativeBatch(self.0.values().as_ref().clone())
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeLoopProgram>()?;
    module.add_class::<NativeCompleteLoopResult>()?;
    module.add_class::<NativeSourceState>()?;
    module.add_class::<NativeSourceSnapshot>()
}
