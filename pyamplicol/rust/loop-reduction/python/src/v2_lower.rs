// SPDX-License-Identifier: 0BSD

//! Python entry points of the native plan model (WP3.1 package D2,
//! `rusticol_core::loop_v2::plan::binary`): the plan v4 model with its
//! byte-identical JSON writer and its binary columnar member, and the loader
//! from either encoding. In phase 0 the model is fed with the records of a
//! Python `ExportedPlan` (`loops/v2/native_lowering.py`).
use super::*;
use pyo3::types::PyBytes;
use rusticol_core::loop_v2::plan::binary::{
    PlanModel, column_names, is_binary, program_digest, read_binary,
};
use rusticol_core::loop_v2::plan::{FormPolicy, LoopProgramV4};
use std::time::Instant;

/// A plan v4 model (`PlanModel`).
#[pyclass]
struct V2PlanModel(PlanModel);

#[pymethods]
impl V2PlanModel {
    /// The model of a plan v4 JSON (`ExportedPlan.json()`).
    #[staticmethod]
    fn from_json(py: Python<'_>, json: &[u8]) -> PyResult<Self> {
        py.detach(|| PlanModel::from_json(json))
            .map(Self)
            .map_err(error)
    }

    /// The model of a binary plan member.
    #[staticmethod]
    fn from_binary(py: Python<'_>, member: &[u8]) -> PyResult<Self> {
        py.detach(|| PlanModel::from_binary(member))
            .map(Self)
            .map_err(error)
    }

    /// The plan JSON, byte-identical to `ExportedPlan.json()`.
    fn json<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let model = &self.0;
        let json = py.detach(|| model.json());
        PyBytes::new(py, &json)
    }

    /// The binary columnar member.
    fn binary<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let model = &mut self.0;
        let member = py.detach(|| model.binary()).map_err(error)?;
        Ok(PyBytes::new(py, &member))
    }

    /// The bound of the loader's derivations for this plan,
    /// `(peak bytes, allocated bytes, iterations)` (`PlanModel::derive_bound`),
    /// after its refusal of untrusted scalars.
    fn derive_bound(&self, py: Python<'_>) -> PyResult<(u64, u64, u64)> {
        let model = &self.0;
        py.detach(|| model.derive_bound())
            .map(|bound| (bound.memory, bound.allocated, bound.work))
            .map_err(error)
    }

    /// The binary member's column names, in member order (diagnostics and
    /// tests of crafted members).
    #[staticmethod]
    fn columns() -> Vec<&'static str> {
        column_names().to_vec()
    }
}

/// Load a plan (binary member or JSON) as `V2Executor` does, timing the
/// decode into the model and the loader's validation and derivations
/// separately: `(form, decode_s, derive_s, digest)`, with the SHA-256 of
/// the loaded program's tables (`program_digest`) when `digest`.
#[pyfunction]
#[pyo3(signature = (data, *, digest=false))]
fn v2_plan_load(
    py: Python<'_>,
    data: &[u8],
    digest: bool,
) -> PyResult<(&'static str, f64, f64, Option<String>)> {
    py.detach(|| {
        let binary = is_binary(data);
        let start = Instant::now();
        let model = if binary {
            PlanModel::from_binary(data)
        } else {
            PlanModel::from_json(data)
        }?;
        let decoded = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let program: LoopProgramV4 = model.program(FormPolicy::Rule)?;
        let derived = start.elapsed().as_secs_f64();
        let digest = digest.then(|| program_digest(&program));
        Ok((
            if binary { "binary" } else { "json" },
            decoded,
            derived,
            digest,
        ))
    })
    .map_err(|e: rusticol_core::RusticolError| error(e))
}

/// The SHA-256 of the program the production readers load from a plan
/// (`program_digest`): a binary member through `read_binary`, the one reader
/// of the member, and a JSON through `LoopProgramV4::load`. A diagnostic: the
/// digest formats the program beyond the reader's bound (`program_digest`),
/// so fail-closed probes call `v2_plan_load(data, digest=False)` instead.
#[pyfunction]
fn v2_plan_read(py: Python<'_>, data: &[u8]) -> PyResult<String> {
    py.detach(|| {
        let program = if is_binary(data) {
            read_binary(data)
        } else {
            LoopProgramV4::load(data)
        }?;
        Ok(program_digest(&program))
    })
    .map_err(|e: rusticol_core::RusticolError| error(e))
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<V2PlanModel>()?;
    module.add_function(wrap_pyfunction!(v2_plan_load, module)?)?;
    module.add_function(wrap_pyfunction!(v2_plan_read, module)?)
}
