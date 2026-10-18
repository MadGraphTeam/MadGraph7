// SPDX-License-Identifier: 0BSD
//! Thin ownership bridge; PACBIN and all runtime descriptor validation live in core.
use crate::error;
use pyo3::{prelude::*, types::PyBytes};
use rusticol_core::loop_runtime::{LoadedLoopRuntime, load_loop_runtime_pacbin};

#[pyclass]
struct PortableLoopPlan {
    loaded: LoadedLoopRuntime,
}
#[pymethods]
impl PortableLoopPlan {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Ok(Self {
            loaded: load_loop_runtime_pacbin(path).map_err(error)?,
        })
    }
    fn metadata_json(&self) -> PyResult<String> {
        self.loaded.metadata_json().map_err(error)
    }
    fn evaluator_state<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyBytes>> {
        Ok(PyBytes::new(
            py,
            self.loaded.evaluator_state(name).map_err(error)?,
        ))
    }
}
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PortableLoopPlan>()
}
