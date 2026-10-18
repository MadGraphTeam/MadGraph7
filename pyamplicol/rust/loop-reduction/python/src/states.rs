// SPDX-License-Identifier: 0BSD

//! Python argument conversion for shared dependency external-state blocks.
use super::*;
type PartialStates = (NativeBatch, Vec<(u64, String)>);

#[pyfunction]
#[pyo3(signature = (momenta, helicities, kinds, components=None))]
fn wavefunctions_batch(
    py: Python<'_>,
    momenta: &NativeBatch,
    helicities: Vec<i32>,
    kinds: Vec<String>,
    components: Option<usize>,
) -> PyResult<NativeBatch> {
    py.detach(|| runtime::wavefunctions_batch(&momenta.0, helicities, kinds, components))
        .map(NativeBatch)
        .map_err(error)
}

#[pyfunction]
#[pyo3(signature = (momenta, helicities, kinds, components=None))]
fn wavefunctions_partial_batch(
    py: Python<'_>,
    momenta: &NativeBatch,
    helicities: Vec<i32>,
    kinds: Vec<String>,
    components: Option<usize>,
) -> PyResult<PartialStates> {
    py.detach(|| runtime::wavefunctions_partial_batch(&momenta.0, helicities, kinds, components))
        .map(|(batch, statuses)| (NativeBatch(batch), statuses))
        .map_err(error)
}

#[pyfunction]
#[pyo3(signature = (spatial_momenta, masses, helicities, kinds, components=None))]
fn on_shell_wavefunctions_batch(
    py: Python<'_>,
    spatial_momenta: &NativeBatch,
    masses: &NativeBatch,
    helicities: Vec<i32>,
    kinds: Vec<String>,
    components: Option<usize>,
) -> PyResult<(NativeBatch, NativeBatch)> {
    py.detach(|| {
        runtime::on_shell_wavefunctions_batch(
            &spatial_momenta.0,
            &masses.0,
            helicities,
            kinds,
            components,
        )
    })
    .map(|(momenta, states)| (NativeBatch(momenta), NativeBatch(states)))
    .map_err(error)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(wavefunctions_batch, module)?)?;
    module.add_function(wrap_pyfunction!(wavefunctions_partial_batch, module)?)?;
    module.add_function(wrap_pyfunction!(on_shell_wavefunctions_batch, module)?)?;
    Ok(())
}
