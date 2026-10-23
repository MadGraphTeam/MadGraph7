// SPDX-License-Identifier: 0BSD

//! Python ownership of the native loop-request DAG builder
//! (`loops/v2/dag_input.py`, `rusticol_core::loop_v2::dag`).
//!
//! A `V2Dag` holds one columnar DAG: built natively from the input tables
//! (`V2Dag.build`) or read from Python's encoding of a `GenericDAG`
//! (`V2Dag.from_columns`). Errors are `ValueError`s; a feature the builder
//! does not cover yet starts with `"unsupported: "` (Python raises
//! `NativeDagUnsupported` for it). Long work releases the GIL.
use super::*;
use pyo3::IntoPyObject;
use pyo3::types::PyBytes;
use rusticol_core::loop_v2::dag::{
    self, ColumnarDag, DagError, blocks::BlockInput, input::DagInput,
};

fn dag_error(error: DagError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

type ColourTuple = (String, i64, Vec<i64>, Vec<String>, Vec<Vec<u8>>, Vec<Vec<u8>>);

/// One block for Python (`dag_input._python_blocks`).
#[derive(IntoPyObject)]
struct PyBlock {
    key_current: u32,
    mask: u64,
    pdg: i64,
    word: Vec<u8>,
    orders: Vec<(String, i64)>,
    colour: ColourTuple,
    chirality: i8,
    spin_state: i32,
    flavour: Vec<i64>,
    quantum_numbers: Vec<(String, String)>,
    auxiliary: Option<String>,
    strings: Vec<Vec<u8>>,
    incoming_open: bool,
    outgoing_open: bool,
    singlet_labels: Vec<u8>,
    colour_sign: i32,
    traces: Vec<Vec<u8>>,
    nc_grade: Option<i64>,
    rows: Vec<(Vec<(u8, i32)>, u32)>,
    connection_sign: i32,
    pairing_sign: i32,
}

/// A columnar loop-request DAG.
#[pyclass(frozen)]
struct V2Dag {
    dag: ColumnarDag,
}

#[pymethods]
impl V2Dag {
    /// Build the loop-request DAG of `dag_input.dag_input`'s JSON tables.
    #[staticmethod]
    fn build(py: Python<'_>, input: &str) -> PyResult<Self> {
        let input = input.to_owned();
        py.detach(|| {
            let input = DagInput::from_json(&input)?;
            dag::build(&input)
        })
        .map(|dag| Self { dag })
        .map_err(dag_error)
    }

    /// Read a columnar container (`dag_input.encode_dag` or `columns()`).
    #[staticmethod]
    fn from_columns(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        py.detach(|| ColumnarDag::from_bytes(data))
            .map(|dag| Self { dag })
            .map_err(dag_error)
    }

    /// `(currents, interactions, sources)`.
    fn counts(&self) -> (usize, usize, usize) {
        (
            self.dag.current_count(),
            self.dag.interaction_count(),
            self.dag.sources.len(),
        )
    }

    /// The columnar container (decoded by `dag_input._decode`).
    fn columns<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes = py.detach(|| self.dag.to_bytes());
        PyBytes::new(py, &bytes)
    }

    /// Streaming SHA-256 of the DAG's flat records (S1′/S1″ DAG part).
    fn digest(&self, py: Python<'_>) -> String {
        py.detach(|| self.dag.digest())
    }

    /// The blocks (`loops/v2/blocks._blocks`) for `dag_input.block_input`.
    fn blocks(&self, py: Python<'_>, tables: &str) -> PyResult<Vec<PyBlock>> {
        let tables = tables.to_owned();
        let raw = py
            .detach(|| {
                let tables = BlockInput::from_json(&tables)?;
                self.dag.blocks(&tables)
            })
            .map_err(dag_error)?;
        let t = &self.dag.tables;
        Ok(raw
            .into_iter()
            .map(|block| {
                let id = block.key_current as usize;
                let species = &t.species[self.dag.c_species[id] as usize];
                let colour = &t.colours[self.dag.c_colour[id] as usize];
                PyBlock {
                    key_current: block.key_current,
                    mask: self.dag.c_mask[id],
                    pdg: species.pdg,
                    word: self.dag.word(id).to_vec(),
                    orders: t.orders[self.dag.c_orders[id] as usize].clone(),
                    colour: (
                        colour.accuracy.clone(),
                        colour.sector_id,
                        colour.line_groups.clone(),
                        colour.basis_key.clone(),
                        colour.traces.clone(),
                        colour.lines.clone(),
                    ),
                    chirality: self.dag.c_chirality[id],
                    spin_state: self.dag.c_spin[id],
                    flavour: t.flavours[self.dag.c_flavour[id] as usize].clone(),
                    quantum_numbers: species.quantum_numbers.clone(),
                    auxiliary: species.auxiliary.clone(),
                    strings: block.colour_flow.strings,
                    incoming_open: block.colour_flow.incoming_open,
                    outgoing_open: block.colour_flow.outgoing_open,
                    singlet_labels: block.colour_flow.singlet_labels,
                    colour_sign: block.colour_flow.sign,
                    traces: block.colour_flow.traces,
                    nc_grade: block.nc_grade,
                    rows: block.rows,
                    connection_sign: block.connection_sign,
                    pairing_sign: block.pairing_sign,
                }
            })
            .collect())
    }

    /// Coloured particle ids propagated by non-source, non-auxiliary
    /// currents (the current scan of `blocks.ward_breaking_fields`).
    fn coloured_propagated(&self, py: Python<'_>, tables: &str) -> PyResult<Vec<i64>> {
        let tables = tables.to_owned();
        py.detach(|| {
            let tables = BlockInput::from_json(&tables)?;
            Ok(self.dag.coloured_propagated(&tables))
        })
        .map_err(dag_error)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<V2Dag>()
}
