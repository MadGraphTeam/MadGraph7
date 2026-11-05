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
use pyo3::types::{PyBytes, PyDict, PyList, PyString};
use rusticol_core::loop_v2::dag::{
    self, ColumnarDag, DagError,
    blocks::BlockInput,
    input::DagInput,
    program::{KernelUse, Program, ProgramInput, Variant},
    reuse::ReuseTables,
};
use std::collections::HashMap;

fn dag_error(error: DagError) -> PyErr {
    PyValueError::new_err(error.to_string())
}

/// `ColorState` fields: accuracy, sector id, line groups, basis key,
/// traces, lines, N_c grade.
type ColourTuple = (
    String,
    i64,
    Vec<i64>,
    Vec<String>,
    Vec<Vec<u8>>,
    Vec<Vec<u8>>,
    Option<i64>,
);

/// `(representative, factor re bits, im bits)` per current.
type EquivalenceRecords = Vec<(u32, u64, u64)>;

/// One block for Python (`dag_input.NativeDag.blocks`).
#[derive(IntoPyObject)]
struct PyBlock {
    key_current: u32,
    mask: u64,
    /// The key's particle: the field a clone block is read as.
    pdg: i64,
    /// The clone particle of a clone line's block (its currents' particle).
    clone: Option<i64>,
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
    /// `(helicity assignment, current id)`, the assignment an index into
    /// the assignments `V2Dag.blocks` returns with the blocks.
    rows: Vec<(u32, u32)>,
    connection_sign: i32,
    pairing_sign: i32,
}

/// One row's helicity assignment: `(leg label, helicity)` per leg.
type Assignment = Vec<(u8, i32)>;

/// The distinct helicity assignments of the blocks' rows, in first-seen
/// order, and each row's index into them.
#[derive(Default)]
struct Assignments {
    index: HashMap<Assignment, u32>,
    table: Vec<Assignment>,
}

impl Assignments {
    fn intern(&mut self, helicities: Assignment) -> PyResult<u32> {
        if let Some(found) = self.index.get(&helicities) {
            return Ok(*found);
        }
        let id = rusticol_core::narrow::narrow(self.table.len(), "helicity assignment id")
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        self.index.insert(helicities.clone(), id);
        self.table.push(helicities);
        Ok(id)
    }
}

/// A columnar loop-request DAG.
#[pyclass(frozen)]
struct V2Dag {
    dag: ColumnarDag,
}

/// One source current for the runtime schema's source records
/// (`dag_input.native_tree_rows`): `(current id, pdg, chirality, spin
/// state, colour, dimension, leg or -1, helicity or NO_HELICITY, helicity
/// ancestry)`.
type SourceRow = (u32, i64, i8, i32, ColourTuple, i32, i32, i32, u64);

/// `(vertex kind, vertex particles, coupling)` of an interaction.
type InteractionRow = (i64, [i64; 3], (f64, f64));

/// The interaction program of a [`V2Dag`] (`ColumnarDag::program`; not
/// frozen: `take_step_section` takes its steps).
#[pyclass]
struct V2Program {
    program: Program,
}

#[pymethods]
impl V2Program {
    /// Value components (the runtime schema's `value_component_count`).
    fn value_count(&self) -> u64 {
        self.program.value_count
    }

    /// The momentum slots' masks, in slot order.
    fn momentum_masks(&self) -> Vec<u64> {
        self.program.momentum_masks.clone()
    }

    /// Per current its value slot's start (little-endian `u64`) and variant
    /// (`u8`: 0 source, 1 unpropagated, 2 propagated).
    fn slots<'py>(&self, py: Python<'py>) -> (Bound<'py, PyBytes>, Bound<'py, PyBytes>) {
        let starts: Vec<u8> = self
            .program
            .starts
            .iter()
            .flat_map(|start| start.to_le_bytes())
            .collect();
        let variants: Vec<u8> = self
            .program
            .variants
            .iter()
            .map(|variant| match variant {
                Variant::Source => 0,
                Variant::Unpropagated => 1,
                Variant::Propagated => 2,
            })
            .collect();
        (PyBytes::new(py, &starts), PyBytes::new(py, &variants))
    }

    /// The kernels in first use: `("kernels", vertex key, interaction)` or
    /// `("propagators", (pdg, chirality, dimension), current)`.
    fn kernels(&self) -> Vec<(&'static str, Vec<i64>, u32)> {
        self.program
            .kernels
            .iter()
            .map(|kernel| match kernel {
                KernelUse::Vertex { key, interaction } => ("kernels", key.to_vec(), *interaction),
                KernelUse::Propagator { key, current } => ("propagators", key.to_vec(), *current),
            })
            .collect()
    }

    /// The distinct colour weights `(re, im)`, in first use.
    fn weights(&self) -> Vec<(f64, f64)> {
        self.program
            .weights
            .iter()
            .map(|[re, im]| (*re, *im))
            .collect()
    }

    /// One interaction per distinct (vertex, coupling), in first occurrence.
    fn signatures(&self) -> Vec<u32> {
        self.program.signatures.clone()
    }

    /// The distinct particle ids of the currents, sorted.
    fn particles(&self) -> Vec<i64> {
        self.program.particles.clone()
    }

    /// The steps of `native_trees._interaction_program`, with `texts[w]` the
    /// exact `(re, im)` texts of weight `w`. Equal values are one Python
    /// object (the oracle's rows share the schema's ints and the texts).
    fn steps<'py>(
        &self,
        py: Python<'py>,
        texts: Vec<(Bound<'py, PyString>, Bound<'py, PyString>)>,
    ) -> PyResult<Bound<'py, PyList>> {
        if texts.len() != self.program.weights.len() {
            return Err(PyValueError::new_err("one text pair per colour weight"));
        }
        // Every non-negative value of a row or output is below the momentum
        // components' end (starts, momentum starts, accumulators, kernels).
        let end = self.program.value_count as usize + 4 * self.program.momentum_masks.len() + 1;
        let mut shared: Vec<Option<Bound<'py, PyAny>>> = vec![None; end];
        let mut int = |value: i64| -> PyResult<Bound<'py, PyAny>> {
            match usize::try_from(value).ok().filter(|index| *index < end) {
                Some(index) => match &shared[index] {
                    Some(found) => Ok(found.clone()),
                    None => {
                        let made = value.into_pyobject(py)?.into_any();
                        shared[index] = Some(made.clone());
                        Ok(made)
                    }
                },
                None => Ok(value.into_pyobject(py)?.into_any()),
            }
        };
        let steps = PyList::empty(py);
        for step in &self.program.steps {
            let rows = PyList::empty(py);
            for row in step.rows.iter() {
                let (real, imag) = &texts[row[8] as usize];
                let mut items: Vec<Bound<'py, PyAny>> = Vec::with_capacity(10);
                for value in &row[..8] {
                    items.push(int(*value)?);
                }
                items.push(real.clone().into_any());
                items.push(imag.clone().into_any());
                rows.append(PyList::new(py, items)?)?;
            }
            let outputs = PyList::empty(py);
            for output in &step.outputs {
                let items = output
                    .iter()
                    .map(|value| int(*value))
                    .collect::<PyResult<Vec<_>>>()?;
                outputs.append(PyList::new(py, items)?)?;
            }
            let accumulators = step
                .accumulators
                .iter()
                .map(|value| int(*value))
                .collect::<PyResult<Vec<_>>>()?;
            let record = PyDict::new(py);
            record.set_item("interactions", rows)?;
            record.set_item("accumulators", PyList::new(py, accumulators)?)?;
            record.set_item("outputs", outputs)?;
            if !step.accumulator_masks.is_empty() {
                record.set_item(
                    "accumulator_masks",
                    PyList::new(py, step.accumulator_masks.iter())?,
                )?;
            }
            steps.append(record)?;
        }
        Ok(steps)
    }
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
    fn digest(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.dag.digest()).map_err(dag_error)
    }

    /// The blocks (`loops/v2/blocks._blocks`) for `dag_input.block_input`,
    /// and the distinct helicity assignments their rows index (Python
    /// builds each assignment once).
    fn blocks(&self, py: Python<'_>, tables: &str) -> PyResult<(Vec<PyBlock>, Vec<Assignment>)> {
        let tables = tables.to_owned();
        let raw = py
            .detach(|| {
                let tables = BlockInput::from_json(&tables)?;
                self.dag.blocks(&tables)
            })
            .map_err(dag_error)?;
        let t = &self.dag.tables;
        let mut assignments = Assignments::default();
        let blocks = raw
            .into_iter()
            .map(|block| -> PyResult<PyBlock> {
                let id = block.key_current as usize;
                let species = &t.species[self.dag.c_species[id] as usize];
                let colour = &t.colours[self.dag.c_colour[id] as usize];
                Ok(PyBlock {
                    key_current: block.key_current,
                    mask: self.dag.c_mask[id],
                    pdg: block.clone_field.unwrap_or(species.pdg),
                    clone: block.clone_field.map(|_| species.pdg),
                    word: self.dag.word(id).to_vec(),
                    orders: t.orders[self.dag.c_orders[id] as usize].clone(),
                    colour: (
                        colour.accuracy.clone(),
                        colour.sector_id,
                        colour.line_groups.clone(),
                        colour.basis_key.clone(),
                        colour.traces.clone(),
                        colour.lines.clone(),
                        colour.nc_grade,
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
                    rows: block
                        .rows
                        .into_iter()
                        .map(|(helicities, current)| {
                            Ok((assignments.intern(helicities)?, current))
                        })
                        .collect::<PyResult<_>>()?,
                    connection_sign: block.connection_sign,
                    pairing_sign: block.pairing_sign,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok((blocks, assignments.table))
    }

    /// The value-equivalence derivation of `dag_input.reuse_tables`' JSON
    /// (`restricted`: only the currents sharing a contract, as production;
    /// otherwise all) and the DAG with its interactions regrouped:
    /// `([(representative, factor re bits, im bits)], V2Dag)`.
    fn reuse(
        &self,
        py: Python<'_>,
        tables: &str,
        restricted: bool,
    ) -> PyResult<(EquivalenceRecords, Self)> {
        let tables = tables.to_owned();
        py.detach(|| {
            let tables = ReuseTables::from_json(&tables)?;
            self.dag.reuse(&tables, restricted)
        })
        .map(|(equivalences, dag)| {
            (
                equivalences
                    .into_iter()
                    .map(|item| {
                        (
                            item.representative,
                            item.factor[0].to_bits(),
                            item.factor[1].to_bits(),
                        )
                    })
                    .collect(),
                Self { dag },
            )
        })
        .map_err(dag_error)
    }

    /// The vertex kinds `ct_closure.u1_leaking_structures` reports
    /// (`ColumnarDag::u1_leaking`).
    fn u1_leaking(
        &self,
        py: Python<'_>,
        kinds: Vec<i64>,
        generator_kinds: Vec<i64>,
        adjoint: Vec<i64>,
        fundamental: u64,
        rest: u64,
    ) -> Vec<i64> {
        py.detach(|| {
            self.dag
                .u1_leaking(&kinds, &generator_kinds, &adjoint, fundamental, rest)
        })
    }

    /// Distinct `(particle id, amputated)` of the non-source, non-auxiliary
    /// currents (`ColumnarDag::propagated_species`; the current scans of
    /// `blocks.ward_breaking_fields` and `artifact.coloured_widths`).
    fn propagated_species(&self, py: Python<'_>, full_mask: u64) -> Vec<(i64, bool)> {
        py.detach(|| self.dag.propagated_species(full_mask))
    }

    /// Distinct `(particle id, chirality)` of the currents, sorted.
    fn current_species(&self) -> Vec<(i64, i8)> {
        self.dag.current_species()
    }

    /// The interaction program for `dag_input.program_input`'s JSON
    /// (`ColumnarDag::program`).
    fn program(&self, py: Python<'_>, input: &str) -> PyResult<V2Program> {
        let input = input.to_owned();
        py.detach(|| {
            let input = ProgramInput::from_json(&input)?;
            let program = self.dag.program(&input);
            rusticol_core::loop_v2::dag::stage_memory("program", self.dag.interaction_count());
            program
        })
        .map(|program| V2Program { program })
        .map_err(dag_error)
    }

    /// The source currents, in `sources` order (see [`SourceRow`]).
    fn source_rows(&self) -> Vec<SourceRow> {
        let t = &self.dag.tables;
        self.dag
            .sources
            .iter()
            .map(|source| {
                let id = *source as usize;
                let colour = &t.colours[self.dag.c_colour[id] as usize];
                (
                    *source,
                    t.species[self.dag.c_species[id] as usize].pdg,
                    self.dag.c_chirality[id],
                    self.dag.c_spin[id],
                    (
                        colour.accuracy.clone(),
                        colour.sector_id,
                        colour.line_groups.clone(),
                        colour.basis_key.clone(),
                        colour.traces.clone(),
                        colour.lines.clone(),
                        colour.nc_grade,
                    ),
                    self.dag.c_dimension[id],
                    self.dag.c_leg[id],
                    self.dag.c_helicity[id],
                    self.dag.c_ancestry[id],
                )
            })
            .collect()
    }

    /// `(vertex kind, vertex particles, coupling)` of the interactions `ids`.
    fn interaction_rows(&self, ids: Vec<u32>) -> PyResult<Vec<InteractionRow>> {
        ids.into_iter()
            .map(|id| {
                let k = id as usize;
                if k >= self.dag.interaction_count() {
                    return Err(PyValueError::new_err(format!("no interaction {id}")));
                }
                let vertex = &self.dag.tables.vertices[self.dag.i_vertex[k] as usize];
                let [re, im] = self.dag.i_coupling[k];
                Ok((vertex.kind, vertex.particles, (re, im)))
            })
            .collect()
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<V2Dag>()?;
    module.add_class::<V2Program>()?;
    register_tree_binary(module)?;
    register_tree_terms(module)
}

// ------------------------------------------------- binary tree program (MF-8G-1)

/// `loop_v2::trees::binary`: the binary tree-program member and its parts.
mod tree_binary {
    use super::*;
    use pyo3::types::PyByteArray;
    use rusticol_core::loop_v2::trees::binary::{self, CellTable, StepColumns};

    fn failed(error: impl std::fmt::Display) -> PyErr {
        PyValueError::new_err(error.to_string())
    }

    /// A binary tree program of `(name, encoding, payload)` sections
    /// (encoding 0 JSON, 1 binary).
    #[pyfunction]
    pub(super) fn tree_program_container<'py>(
        py: Python<'py>,
        sections: Vec<(String, u8, Bound<'py, PyBytes>)>,
    ) -> Bound<'py, PyBytes> {
        let parts: Vec<(&str, u8, &[u8])> = sections
            .iter()
            .map(|(name, encoding, payload)| (name.as_str(), *encoding, payload.as_bytes()))
            .collect();
        let out = py.detach(|| binary::write_container(&parts));
        PyBytes::new(py, &out)
    }

    /// The `(name, encoding)` of every section of a binary tree program.
    #[pyfunction]
    pub(super) fn tree_program_index(container: &[u8]) -> PyResult<Vec<(String, u8)>> {
        Ok(binary::read_container(container)
            .map_err(failed)?
            .into_iter()
            .map(|section| (section.name, section.encoding))
            .collect())
    }

    /// One section's payload (None: absent).
    #[pyfunction]
    pub(super) fn tree_program_section<'py>(
        py: Python<'py>,
        container: &[u8],
        name: &str,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        Ok(binary::read_container(container)
            .map_err(failed)?
            .into_iter()
            .find(|section| section.name == name)
            .map(|section| PyBytes::new(py, section.payload)))
    }

    /// The step section of the oracle's JSON steps.
    #[pyfunction]
    pub(super) fn tree_steps_from_json<'py>(
        py: Python<'py>,
        json: &[u8],
    ) -> PyResult<Bound<'py, PyBytes>> {
        let out = py
            .detach(|| StepColumns::from_json(json).map(|c| c.encode()))
            .map_err(failed)?;
        Ok(PyBytes::new(py, &out))
    }

    /// A step section as the oracle's JSON steps.
    #[pyfunction]
    pub(super) fn tree_steps_json(py: Python<'_>, section: &[u8]) -> PyResult<String> {
        py.detach(|| StepColumns::decode(section).map(|c| c.json()))
            .map_err(failed)
    }

    /// `(rows, accumulators, outputs)` of a step section.
    #[pyfunction]
    pub(super) fn tree_steps_counts(section: &[u8]) -> PyResult<(u64, u64, u64)> {
        let c = StepColumns::decode(section).map_err(failed)?;
        Ok((
            c.rows.iter().sum(),
            c.accumulators.iter().sum(),
            c.outputs.iter().sum(),
        ))
    }

    /// `native_trees._prune_interactions` on a step section: the pruned
    /// section and the used vertex kernels and propagators (old indices),
    /// or None (unchanged) where a component is written twice. `live` is
    /// the readers' marks over the program's state (not modified).
    #[pyfunction]
    pub(super) fn tree_steps_prune<'py>(
        py: Python<'py>,
        section: &[u8],
        live: &Bound<'py, PyByteArray>,
        kernels: usize,
        propagators: usize,
    ) -> PyResult<Option<(Bound<'py, PyBytes>, Vec<usize>, Vec<usize>)>> {
        let mut marks = live.to_vec();
        // One copy of the section at a time beside the pruned columns: the
        // decoded steps go before the encoding, the pruned columns before
        // the Python bytes (MF-8G-1: the counterterm program's section is
        // the 2->6 export's largest).
        let pruned = py
            .detach(|| {
                let decoded = StepColumns::decode(section)?;
                let found = decoded.prune(&mut marks, kernels, propagators)?;
                drop(decoded);
                Ok::<_, rusticol_core::RusticolError>(found.map(|(columns, k, p)| {
                    let encoded = columns.encode();
                    drop(columns);
                    (encoded, k, p)
                }))
            })
            .map_err(failed)?;
        Ok(pruned.map(|(bytes, k, p)| (PyBytes::new(py, &bytes), k, p)))
    }

    /// A cell section as the oracle's cell list (tests and tools).
    #[pyfunction]
    pub(super) fn tree_cells_list(
        section: &[u8],
    ) -> PyResult<Vec<(usize, usize, i32, String, String)>> {
        Ok(CellTable::decode(section).map_err(failed)?.list())
    }

    /// `(basis, tensors, distinct cells)` of a cell section.
    #[pyfunction]
    pub(super) fn tree_cells_counts(section: &[u8]) -> PyResult<(usize, usize, usize)> {
        let c = CellTable::decode(section).map_err(failed)?;
        Ok((c.basis, c.tensors, c.values.len()))
    }

    /// `native_trees._live_ket` on a cell section.
    #[allow(clippy::type_complexity)]
    #[pyfunction]
    pub(super) fn tree_cells_live_ket(
        py: Python<'_>,
        section: &[u8],
        born: Vec<(usize, usize, i32)>,
        entries: Vec<(String, usize, usize, Option<usize>, i32, usize, Option<i32>)>,
        ket: Vec<(usize, usize, i32)>,
        depth: Option<i32>,
    ) -> PyResult<Vec<bool>> {
        py.detach(|| {
            CellTable::decode(section)?.live_ket(&born, &entries, &ket, depth)
        })
        .map_err(failed)
    }
}

#[pymethods]
impl V2Program {
    /// The steps as a binary step section (`loop_v2::trees::binary::
    /// StepColumns`), with `texts[w]` the exact `(re, im)` texts of weight
    /// `w`: the columns `steps` would give as Python lists. The program gives
    /// its steps up (`steps` is empty afterwards): their rows are released
    /// as the section is formed (MF-8G-1 item 4).
    fn take_step_section<'py>(
        &mut self,
        py: Python<'py>,
        texts: Vec<(String, String)>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        if texts.len() != self.program.weights.len() {
            return Err(PyValueError::new_err("one text pair per colour weight"));
        }
        let steps = std::mem::take(&mut self.program.steps);
        let out = py
            .detach(|| {
                let columns =
                    rusticol_core::loop_v2::trees::binary::StepColumns::from_program_owned(
                        steps, texts,
                    )?;
                rusticol_core::loop_v2::dag::stage_memory("program step columns", 0);
                Ok::<_, rusticol_core::RusticolError>(columns.encode())
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyBytes::new(py, &out))
    }
}

pub(super) fn register_tree_binary(module: &Bound<'_, PyModule>) -> PyResult<()> {
    use tree_binary::*;
    module.add_function(wrap_pyfunction!(tree_program_container, module)?)?;
    module.add_function(wrap_pyfunction!(tree_program_index, module)?)?;
    module.add_function(wrap_pyfunction!(tree_program_section, module)?)?;
    module.add_function(wrap_pyfunction!(tree_steps_from_json, module)?)?;
    module.add_function(wrap_pyfunction!(tree_steps_json, module)?)?;
    module.add_function(wrap_pyfunction!(tree_steps_counts, module)?)?;
    module.add_function(wrap_pyfunction!(tree_steps_prune, module)?)?;
    module.add_function(wrap_pyfunction!(tree_cells_list, module)?)?;
    module.add_function(wrap_pyfunction!(tree_cells_counts, module)?)?;
    module.add_function(wrap_pyfunction!(tree_cells_live_ket, module)?)
}

/// Closure-term sections of a binary tree program (MF-8G-1).
mod tree_terms {
    use super::*;
    use rusticol_core::loop_v2::trees::binary::{TermBuilder, TermColumns};

    fn failed(error: impl std::fmt::Display) -> PyErr {
        PyValueError::new_err(error.to_string())
    }

    /// Closure terms gathered one at a time (`TermBuilder`; `native_trees.
    /// export_trees`'s counterterm closure: no Python record list kept and
    /// no JSON in between), written as a term section by `section`.
    #[pyclass]
    #[derive(Default)]
    pub(super) struct TreeTermBuilder(TermBuilder);

    #[pymethods]
    impl TreeTermBuilder {
        #[new]
        fn new() -> Self {
            Self::default()
        }

        /// Append one closure-term record (``record``'s dict: kernel,
        /// layout, blocks, masks, sign, contributions).
        fn append(&mut self, record: &Bound<'_, PyAny>) -> PyResult<()> {
            let field = |name: &str| record.get_item(name);
            let contributions: Vec<(usize, usize, usize)> = field("contributions")?
                .try_iter()?
                .map(|item| {
                    let item = item?;
                    Ok((
                        item.get_item(0)?.extract()?,
                        item.get_item(1)?.extract()?,
                        item.get_item(2)?.extract()?,
                    ))
                })
                .collect::<PyResult<_>>()?;
            self.0.push(
                field("kernel")?.extract()?,
                field("layout")?.extract()?,
                &field("blocks")?.extract::<Vec<usize>>()?,
                &field("masks")?.extract::<Vec<u64>>()?,
                field("sign")?.extract()?,
                &contributions,
            );
            Ok(())
        }

        fn __len__(&self) -> usize {
            self.0.len()
        }

        /// The term section of the appended terms (the builder emptied).
        fn section<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
            let terms = std::mem::take(&mut self.0);
            let out = py.detach(|| terms.finish().encode());
            PyBytes::new(py, &out)
        }
    }

    /// A term section without the ket components `live` does not mark
    /// (`TermColumns::prune_ket`, `native_trees._prune_counterterm_terms`).
    #[pyfunction]
    pub(super) fn tree_terms_prune_ket<'py>(
        py: Python<'py>,
        section: &[u8],
        live: Vec<bool>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let out = py
            .detach(|| TermColumns::decode(section)?.prune_ket(&live).map(|c| c.encode()))
            .map_err(failed)?;
        Ok(PyBytes::new(py, &out))
    }

    /// Per closure block the widest row prefix a term of the section reads
    /// (`TermColumns::block_widths`).
    #[pyfunction]
    pub(super) fn tree_terms_block_widths(
        py: Python<'_>,
        section: &[u8],
        widths: Vec<Vec<usize>>,
        blocks: usize,
    ) -> PyResult<Vec<usize>> {
        py.detach(|| TermColumns::decode(section)?.block_widths(&widths, blocks))
            .map_err(failed)
    }

    /// The term section of the oracle's JSON closure terms.
    #[pyfunction]
    pub(super) fn tree_terms_from_json<'py>(
        py: Python<'py>,
        json: &[u8],
    ) -> PyResult<Bound<'py, PyBytes>> {
        let out = py
            .detach(|| TermColumns::from_json(json).map(|c| c.encode()))
            .map_err(failed)?;
        Ok(PyBytes::new(py, &out))
    }

    /// A term section as the oracle's JSON closure terms.
    #[pyfunction]
    pub(super) fn tree_terms_json(py: Python<'_>, section: &[u8]) -> PyResult<String> {
        py.detach(|| TermColumns::decode(section).map(|c| c.json()))
            .map_err(failed)
    }

    /// The number of terms of a term section.
    #[pyfunction]
    pub(super) fn tree_terms_count(section: &[u8]) -> PyResult<usize> {
        Ok(TermColumns::decode(section).map_err(failed)?.len())
    }
}

pub(super) fn register_tree_terms(module: &Bound<'_, PyModule>) -> PyResult<()> {
    use tree_terms::*;
    module.add_function(wrap_pyfunction!(tree_terms_from_json, module)?)?;
    module.add_function(wrap_pyfunction!(tree_terms_json, module)?)?;
    module.add_function(wrap_pyfunction!(tree_terms_prune_ket, module)?)?;
    module.add_function(wrap_pyfunction!(tree_terms_block_widths, module)?)?;
    module.add_class::<TreeTermBuilder>()?;
    module.add_function(wrap_pyfunction!(tree_terms_count, module)?)
}
