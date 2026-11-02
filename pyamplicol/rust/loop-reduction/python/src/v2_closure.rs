// SPDX-License-Identifier: 0BSD

//! Python entry point of the closed root colours (M-FINAL mf-8g-1b;
//! `rusticol_core::loop_v2::generate::root_colours`): `loops/v2/ct_sites.py`
//! encodes the colour networks a closure needs as factored networks
//! (`native_seeds._Networks`) and decodes the reduced tensors
//! (`native_seeds._decoded`). `SiteClosure._closed` reducing one network at
//! a time in Python stays the oracle. No Symbolica licence call is made.
use super::*;
use crate::v2_seeds::{RawTerm, raw_tensor};
use rusticol_core::loop_v2::generate::root_colours::reduce_factored;
use rusticol_core::loop_v2::generate::seeds::FactoredSpec;

/// The reduced tensors `(normalisation, terms)` of the contents of `spec`
/// (a `FactoredSpec` JSON) in their order, on the process legs `legs`.
#[pyfunction]
fn loop_root_colours(
    py: Python<'_>,
    legs: Vec<i64>,
    spec: &str,
) -> PyResult<Vec<(i64, Vec<RawTerm>)>> {
    let spec: FactoredSpec = serde_json::from_str(spec).map_err(error)?;
    let tensors = py
        .detach(|| reduce_factored(&spec, &legs))
        .map_err(error)?;
    Ok(tensors.iter().map(raw_tensor).collect())
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(loop_root_colours, module)?)?;
    Ok(())
}
