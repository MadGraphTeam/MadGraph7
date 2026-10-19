// SPDX-License-Identifier: 0BSD

//! Python ownership for shared native numerical agreement and lane transport.

use super::*;
use rusticol_loop_reduction::runtime::Batch;

#[pymethods]
impl NativeBatch {
    /// Convert existing values; higher accuracy requires reevaluating exact inputs.
    #[pyo3(signature = (arithmetic, precision=None))]
    fn convert_arithmetic(
        &self,
        py: Python<'_>,
        arithmetic: &str,
        precision: Option<u32>,
    ) -> PyResult<Self> {
        py.detach(|| self.0.convert_arithmetic(arithmetic, precision))
            .map(Self)
            .map_err(error)
    }

    /// Empirical component agreement, not an independent accuracy certificate.
    fn agreement_rows(
        &self,
        py: Python<'_>,
        other: &NativeBatch,
        relative_tolerance: &str,
        absolute_tolerance: &str,
    ) -> PyResult<Vec<(u64, bool)>> {
        py.detach(|| {
            self.0
                .agreement_rows(&other.0, relative_tolerance, absolute_tolerance)
        })
        .map_err(error)
    }

    /// Scatter disjoint successful groups into original-ID order; omit failures.
    #[staticmethod]
    fn merge_rows(
        py: Python<'_>,
        row_ids: Vec<u64>,
        batches: Vec<PyRef<'_, NativeBatch>>,
    ) -> PyResult<Self> {
        // Release Python references before native work; Batch clones share storage.
        let batches: Vec<_> = batches.iter().map(|batch| batch.0.clone()).collect();
        py.detach(|| Batch::merge_rows(row_ids, &batches.iter().collect::<Vec<_>>()))
            .map(Self)
            .map_err(error)
    }
}
