// SPDX-License-Identifier: 0BSD

//! Python entry points of the exact colour-cell engine (WP3.1 package C:
//! `loop_v2::generate::cells`, `loop_v2::dipole_orbits`). Tensors arrive as
//! the plain tuples of `loops/v2/native_cells.py` with exact numbers as
//! fraction texts; cells stay native in [`NativeCellTable`] handles, which
//! count their decodes. Nothing here touches Symbolica state.
use super::*;
use rusticol_core::loop_v2::dipole_orbits::DipoleOrbits;
use rusticol_core::loop_v2::generate::FastMap;
use rusticol_core::loop_v2::generate::cells::{
    self, Accuracy, AdjointOrbits, CellEngine, ClassInput, Exact, History, Laurent, OpenString,
    OrderedTable, RecordColumn, Structure, Tensor, Term,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// (strings (q, word, qbar), traces).
type RawStructure = (Vec<(i64, Vec<i64>, i64)>, Vec<Vec<i64>>);
/// (re, im, N_c power, structure).
type RawTerm = (String, String, i32, RawStructure);
/// (legs (label, representation), terms, normalisation power).
type RawTensor = (Vec<(i64, i32)>, Vec<RawTerm>, u32);
/// (kind: 0 adjoint ring, 1 fundamental trace, 2 quark ring, 3 generic;
/// word; ring id; forward relabelling; generic tensor).
type RawClass = (u8, Vec<i64>, usize, Vec<(i64, i64)>, Option<RawTensor>);
/// One Laurent cell: (N_c power, re, im) per nonzero coefficient.
type RawLaurent = Vec<(i32, String, String)>;
/// Born grades per (basis element, Born-order rank).
type RawGrades = Vec<((usize, u32), Vec<i32>)>;
/// Dipole operators per (i, j) leg pair: rows of fraction texts.
type RawDipoles = Vec<((i64, i64), Vec<Vec<String>>)>;

fn structure(raw: RawStructure) -> Structure {
    let (strings, traces) = raw;
    Structure::new(
        strings
            .into_iter()
            .map(|(q, word, qbar)| OpenString { q, word, qbar })
            .collect(),
        traces,
    )
}

fn tensor(raw: RawTensor) -> PyResult<Tensor> {
    let (legs, terms, normalization) = raw;
    let terms = terms
        .into_iter()
        .map(|(re, im, nc_power, s)| {
            Ok(Term {
                coefficient: Exact::parse(&re, &im).map_err(error)?,
                structure: structure(s),
                nc_power,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(Tensor::connected(legs, terms, normalization))
}

fn raw_laurent(cell: &Laurent) -> RawLaurent {
    cell.iter()
        .map(|(&power, value)| {
            let (re, im) = value.texts();
            (power, re, im)
        })
        .collect()
}

fn raw_tensor(tensor: &Tensor) -> RawTensor {
    (
        tensor.legs.clone(),
        tensor
            .terms
            .iter()
            .map(|term| {
                let (re, im) = term.coefficient.texts();
                let structure = &term.structure;
                (
                    re,
                    im,
                    term.nc_power,
                    (
                        structure
                            .strings
                            .iter()
                            .map(|s| (s.q, s.word.clone(), s.qbar))
                            .collect(),
                        structure.traces.clone(),
                    ),
                )
            })
            .collect(),
        tensor.normalization,
    )
}

fn class(raw: RawClass) -> PyResult<ClassInput> {
    let (kind, word, ring, forward, generic) = raw;
    Ok(match kind {
        0 => ClassInput::AdjointRing(word),
        1 => ClassInput::FundamentalTrace(word),
        2 => ClassInput::QuarkRing { ring, forward },
        3 => ClassInput::Generic(tensor(
            generic.ok_or_else(|| error("a generic class needs its tensor"))?,
        )?),
        other => return Err(error(format!("unknown seed class kind {other}"))),
    })
}

fn accuracy(name: &str) -> PyResult<Accuracy> {
    Accuracy::parse(name).map_err(error)
}

/// Run `work` on `threads` worker threads (None: the global pool).
fn on_threads<T: Send>(
    py: Python<'_>,
    threads: Option<usize>,
    work: impl FnOnce() -> rusticol_core::RusticolResult<T> + Send,
) -> PyResult<T> {
    py.detach(|| cells::with_threads(threads, work))
        .map_err(error)?
        .map_err(error)
}

/// Exact seed cells of one table (`SeedTable.cells`), kept native.
#[pyclass(frozen)]
struct NativeCellTable {
    rows: Vec<Vec<Arc<Laurent>>>,
    decodes: AtomicUsize,
}

impl NativeCellTable {
    fn new(rows: Vec<Vec<Arc<Laurent>>>) -> Self {
        Self {
            rows,
            decodes: AtomicUsize::new(0),
        }
    }
}

#[pymethods]
impl NativeCellTable {
    /// A table from Python cells (`[[[(power, re, im), ...] per b] per c]`).
    #[staticmethod]
    fn from_cells(rows: Vec<Vec<RawLaurent>>) -> PyResult<Self> {
        let rows = rows
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|cell| {
                        let mut out = Laurent::new();
                        for (power, re, im) in cell {
                            let value = Exact::parse(&re, &im).map_err(error)?;
                            if !value.is_zero() {
                                out.insert(power, value);
                            }
                        }
                        Ok(Arc::new(out))
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self::new(rows))
    }

    fn __len__(&self) -> usize {
        self.rows.len()
    }

    /// Classes decoded to Python so far (rule 9 of the migration plan).
    #[getter]
    fn decodes(&self) -> usize {
        self.decodes.load(Ordering::Relaxed)
    }

    /// `(values, ids)`: the distinct cells and, per class and basis element,
    /// the index of its cell (rows `start..stop`).
    #[pyo3(signature = (start=0, stop=None))]
    fn decode(
        &self,
        start: usize,
        stop: Option<usize>,
    ) -> PyResult<(Vec<RawLaurent>, Vec<Vec<usize>>)> {
        let stop = stop.unwrap_or(self.rows.len()).min(self.rows.len());
        let start = start.min(stop);
        // Shared cells are looked up by address first, then by value: the
        // ids are the first-seen order of the distinct values either way.
        let mut shared: FastMap<*const Laurent, usize> = FastMap::default();
        let mut index: FastMap<&Laurent, usize> = FastMap::default();
        let mut values = Vec::new();
        let ids = self.rows[start..stop]
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| {
                        *shared.entry(Arc::as_ptr(cell)).or_insert_with(|| {
                            *index.entry(cell.as_ref()).or_insert_with(|| {
                                values.push(raw_laurent(cell));
                                values.len() - 1
                            })
                        })
                    })
                    .collect()
            })
            .collect();
        self.decodes.fetch_add(stop - start, Ordering::Relaxed);
        Ok((values, ids))
    }

    /// The rows of both tables, in order.
    fn concatenated(&self, other: &NativeCellTable) -> NativeCellTable {
        let mut rows = self.rows.clone();
        rows.extend(other.rows.iter().cloned());
        NativeCellTable::new(rows)
    }

    /// `SeedTable.degree` (born_grades None: every Born amplitude grade 0).
    #[pyo3(signature = (class_orders, basis_orders, pairs, born_grades=None))]
    fn degree(
        &self,
        class_orders: Vec<Vec<u32>>,
        basis_orders: Vec<Vec<u32>>,
        pairs: Vec<(u32, u32)>,
        born_grades: Option<RawGrades>,
    ) -> Option<i32> {
        let table = OrderedTable {
            cells: &self.rows,
            class_orders: &class_orders,
            basis_orders: &basis_orders,
        };
        let grades: Option<FastMap<(usize, u32), Vec<i32>>> =
            born_grades.map(|g| g.into_iter().collect());
        table.degree(&pairs.into_iter().collect(), grades.as_ref())
    }
}

/// Seed cells against one basis, with the `_PrimitiveCells` caches.
#[pyclass]
struct NativeCellEngine {
    engine: CellEngine,
}

#[pymethods]
impl NativeCellEngine {
    #[new]
    fn new(legs: Vec<(i64, i32)>, basis: Vec<RawTensor>) -> PyResult<Self> {
        let basis = basis
            .into_iter()
            .map(tensor)
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            engine: CellEngine::new(legs, basis).map_err(error)?,
        })
    }

    /// The engine against another basis on the same legs, sharing caches.
    fn rebased(&self, basis: Vec<RawTensor>) -> PyResult<Self> {
        let basis = basis
            .into_iter()
            .map(tensor)
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            engine: self.engine.rebased(basis).map_err(error)?,
        })
    }

    /// The engine against the operator images of this basis under a colour
    /// history of gluon emissions (`born`, `loop_side`: (emitter, emitted)
    /// in order), sharing the caches (`DirectedColour._table`).
    #[pyo3(signature = (born, loop_side, threads=None))]
    fn directed(
        &self,
        py: Python<'_>,
        born: Vec<(i64, i64)>,
        loop_side: Vec<(i64, i64)>,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        let engine = &self.engine;
        let directed = on_threads(py, threads, || engine.directed(&born, &loop_side))?;
        Ok(Self { engine: directed })
    }

    /// Register a canonical quark ring by its reduced tensor.
    fn register_ring(&self, ring: RawTensor) -> PyResult<usize> {
        self.engine.register_ring(tensor(ring)?).map_err(error)
    }

    /// Sizes of the caches this engine shares with its rebased engines:
    /// (relabelled bras, registered rings, primitive overlaps, ring
    /// overlaps).
    #[getter]
    fn cache_sizes(&self) -> PyResult<(usize, usize, usize, usize)> {
        self.engine.cache_sizes().map_err(error)
    }

    /// The cells of `classes` against the basis (`colour.seed_cells`).
    #[pyo3(signature = (classes, threads=None))]
    fn table(
        &mut self,
        py: Python<'_>,
        classes: Vec<RawClass>,
        threads: Option<usize>,
    ) -> PyResult<NativeCellTable> {
        let classes = classes
            .into_iter()
            .map(class)
            .collect::<PyResult<Vec<_>>>()?;
        let engine = &mut self.engine;
        let rows = on_threads(py, threads, || engine.cells(&classes))?;
        Ok(NativeCellTable::new(rows))
    }
}

/// `BornBasis.gram`: `<b|b'>` exact in N_c, as a table (rows b).
#[pyfunction]
#[pyo3(signature = (basis, threads=None))]
fn colour_gram(
    py: Python<'_>,
    basis: Vec<RawTensor>,
    threads: Option<usize>,
) -> PyResult<NativeCellTable> {
    let basis = basis
        .into_iter()
        .map(tensor)
        .collect::<PyResult<Vec<_>>>()?;
    let rows = on_threads(py, threads, || cells::gram(&basis))?;
    Ok(NativeCellTable::new(rows))
}

/// `colour.contract_colour` per pair (bra, ket).
#[pyfunction]
#[pyo3(signature = (pairs, threads=None))]
fn contract_colours(
    py: Python<'_>,
    pairs: Vec<(RawTensor, RawTensor)>,
    threads: Option<usize>,
) -> PyResult<Vec<RawLaurent>> {
    let pairs = pairs
        .into_iter()
        .map(|(a, b)| Ok((tensor(a)?, tensor(b)?)))
        .collect::<PyResult<Vec<_>>>()?;
    let cells = on_threads(py, threads, || cells::contract_many(&pairs))?;
    Ok(cells.iter().map(raw_laurent).collect())
}

/// `colour.face_count` per pair of structures (bra, ket).
#[pyfunction]
fn face_counts(pairs: Vec<(RawStructure, RawStructure)>) -> PyResult<Vec<i32>> {
    pairs
        .into_iter()
        .map(|(a, b)| cells::face_count(&structure(a), &structure(b)).map_err(error))
        .collect()
}

/// The two-level seed records of a plan (`_Exporter.seed_records`):
/// `(class records JSON, seed records JSON, Born components)`.
///
/// `tables[t]` = (cells, loop-order ranks per class, Born-order ranks per
/// basis element); `columns` = (table, (Born rank, loop rank) pairs,
/// degree, basis subset); `seeds` = (combination of (class, re, im), loop
/// rank) in plan order; `components` the Born components seen so far;
/// `threads` a private pool (None: the global one).
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyo3(signature = (tables, columns, born_grades, accuracy, components, seeds, threads=None))]
fn project_records(
    py: Python<'_>,
    tables: Vec<(PyRef<'_, NativeCellTable>, Vec<Vec<u32>>, Vec<Vec<u32>>)>,
    columns: Vec<(usize, Vec<(u32, u32)>, Option<i32>, Option<Vec<usize>>)>,
    born_grades: Vec<((usize, u32), Vec<i32>)>,
    accuracy: &str,
    components: Vec<(usize, u32, i32)>,
    seeds: Vec<(Vec<(usize, String, String)>, u32)>,
    threads: Option<usize>,
) -> PyResult<(Vec<u8>, Vec<u8>, Vec<(usize, u32, i32)>)> {
    let accuracy = self::accuracy(accuracy)?;
    let ordered: Vec<OrderedTable<'_>> = tables
        .iter()
        .map(|(table, class_orders, basis_orders)| OrderedTable {
            cells: &table.rows,
            class_orders,
            basis_orders,
        })
        .collect();
    let columns: Vec<RecordColumn> = columns
        .into_iter()
        .map(|(table, pairs, degree, basis)| RecordColumn {
            table,
            pairs: pairs.into_iter().collect(),
            degree,
            basis: basis.map(|b| b.into_iter().collect()),
        })
        .collect();
    let grades: FastMap<(usize, u32), Vec<i32>> = born_grades.into_iter().collect();
    let seeds = seeds
        .into_iter()
        .map(|(combination, order)| {
            Ok((
                combination
                    .into_iter()
                    .map(|(c, re, im)| Ok((c, Exact::parse(&re, &im).map_err(error)?)))
                    .collect::<PyResult<Vec<_>>>()?,
                order,
            ))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let mut components = components;
    let records = on_threads(py, threads, || {
        cells::project_records(
            &ordered,
            &columns,
            &grades,
            accuracy,
            &mut components,
            &seeds,
        )
    })?;
    Ok((
        records.class_records.into_bytes(),
        records.records.into_bytes(),
        components,
    ))
}

/// `assemble._AdjointOrbits` (inclusive Born key sets).
#[pyclass(frozen)]
struct NativeAdjointOrbits {
    orbits: AdjointOrbits,
}

#[pymethods]
impl NativeAdjointOrbits {
    /// `adjoint` = (label, outgoing PDG) of the adjoint legs; `born_keys` =
    /// the Born key sets {(structure, N_c power)}.
    #[new]
    fn new(adjoint: Vec<(i64, i64)>, born_keys: Vec<Vec<(RawStructure, i32)>>) -> Self {
        let keys: Vec<Vec<(Structure, i32)>> = born_keys
            .into_iter()
            .map(|set| set.into_iter().map(|(s, p)| (structure(s), p)).collect())
            .collect();
        Self {
            orbits: AdjointOrbits::new(&adjoint, &keys),
        }
    }

    #[getter]
    fn classes(&self) -> Vec<Vec<i64>> {
        self.orbits.classes.clone()
    }

    /// `_AdjointOrbits.rename` of one structure, as (label, image) pairs.
    fn rename(&self, raw: RawStructure) -> Vec<(i64, i64)> {
        let mut out: Vec<(i64, i64)> = self.orbits.rename(&structure(raw)).into_iter().collect();
        out.sort();
        out
    }
}

/// `LoopSlice.degrees`'s overlap degrees: per (bra set, ket set, offset)
/// query, the largest N_c degree of a nonzero overlap plus the key powers
/// and the offset (None if every overlap vanishes).
#[pyfunction]
#[pyo3(signature = (legs, sets, queries, orbits=None, threads=None))]
fn overlap_degrees(
    py: Python<'_>,
    legs: Vec<(i64, i32)>,
    sets: Vec<Vec<(RawStructure, i32)>>,
    queries: Vec<(usize, usize, i32)>,
    orbits: Option<PyRef<'_, NativeAdjointOrbits>>,
    threads: Option<usize>,
) -> PyResult<Vec<Option<i32>>> {
    let mut legs = legs;
    legs.sort();
    let sets: Vec<Vec<(Structure, i32)>> = sets
        .into_iter()
        .map(|set| set.into_iter().map(|(s, p)| (structure(s), p)).collect())
        .collect();
    let orbits = orbits.as_ref().map(|o| o.orbits.clone());
    on_threads(py, threads, || {
        cells::overlap_degrees(&legs, &sets, &queries, orbits.as_ref())
    })
}

/// The overlap degrees under a colour history (`DirectedColour.overlap`):
/// per (bra set, ket set, offset) query over sets of (tensor, N_c power),
/// the largest N_c degree of a nonzero overlap plus both powers and the
/// offset (`cells::tensor_overlap_degrees`).
#[pyfunction]
#[pyo3(signature = (bras, kets, queries, threads=None))]
fn tensor_overlap_degrees(
    py: Python<'_>,
    bras: Vec<Vec<(RawTensor, i32)>>,
    kets: Vec<Vec<(RawTensor, i32)>>,
    queries: Vec<(usize, usize, i32)>,
    threads: Option<usize>,
) -> PyResult<Vec<Option<i32>>> {
    let sets = |raw: Vec<Vec<(RawTensor, i32)>>| -> PyResult<Vec<Vec<(Tensor, i32)>>> {
        raw.into_iter()
            .map(|set| {
                set.into_iter()
                    .map(|(t, p)| Ok((tensor(t)?, p)))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect()
    };
    let (bras, kets) = (sets(bras)?, sets(kets)?);
    on_threads(py, threads, || {
        cells::tensor_overlap_degrees(&bras, &kets, &queries)
    })
}

/// `DirectedColour.operator` of literal tensors under a colour history of
/// gluon emissions (`cells::operator_image`), as raw tensors (exact texts).
#[pyfunction]
#[pyo3(signature = (tensors, born, loop_side))]
fn operator_images(
    py: Python<'_>,
    tensors: Vec<RawTensor>,
    born: Vec<(i64, i64)>,
    loop_side: Vec<(i64, i64)>,
) -> PyResult<Vec<RawTensor>> {
    let tensors = tensors
        .into_iter()
        .map(tensor)
        .collect::<PyResult<Vec<_>>>()?;
    let images = py
        .detach(|| {
            tensors
                .iter()
                .map(|t| cells::operator_image(t, &born, &loop_side))
                .collect::<rusticol_core::RusticolResult<Vec<_>>>()
        })
        .map_err(error)?;
    Ok(images.iter().map(raw_tensor).collect())
}

/// The overlap degrees under colour histories of gluon emissions, with the
/// operator images of the bra structures formed natively
/// (`cells::directed_overlap_degrees`): per (history, bra set, ket set,
/// offset) query.
#[pyfunction]
#[pyo3(signature = (legs, bras, kets, histories, queries, threads=None))]
fn directed_overlap_degrees(
    py: Python<'_>,
    legs: Vec<(i64, i32)>,
    bras: Vec<Vec<(RawStructure, i32)>>,
    kets: Vec<Vec<(RawStructure, i32)>>,
    histories: Vec<History>,
    queries: Vec<(usize, usize, usize, i32)>,
    threads: Option<usize>,
) -> PyResult<Vec<Option<i32>>> {
    let sets = |raw: Vec<Vec<(RawStructure, i32)>>| -> Vec<Vec<(Structure, i32)>> {
        raw.into_iter()
            .map(|set| set.into_iter().map(|(s, p)| (structure(s), p)).collect())
            .collect()
    };
    let (bras, kets) = (sets(bras), sets(kets));
    on_threads(py, threads, || {
        cells::directed_overlap_degrees(&legs, &bras, &kets, &histories, &queries)
    })
}

/// `_prune_colour`'s class liveness per (class, loop rank) query.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (criteria, sectors, basis_orders, grades, accuracy, queries, threads=None))]
fn live_classes(
    py: Python<'_>,
    criteria: Vec<(PyRef<'_, NativeCellTable>, Vec<Option<i32>>)>,
    sectors: Vec<Vec<(u32, u32)>>,
    basis_orders: Vec<Vec<u32>>,
    grades: Vec<((usize, u32), Vec<i32>)>,
    accuracy: &str,
    queries: Vec<(usize, u32)>,
    threads: Option<usize>,
) -> PyResult<Vec<bool>> {
    let accuracy = self::accuracy(accuracy)?;
    let tables: Vec<cells::Criterion<'_>> = criteria
        .iter()
        .map(|(table, degrees)| (table.rows.as_slice(), degrees.clone()))
        .collect();
    let grades: FastMap<(usize, u32), Vec<i32>> = grades.into_iter().collect();
    on_threads(py, threads, || {
        cells::live_classes(
            &tables,
            &sectors,
            &basis_orders,
            &grades,
            accuracy,
            &queries,
        )
    })
}

/// IR dipole operators per relabelling orbit (`loop_v2::dipole_orbits`).
/// Expansions to Python objects count their entries (rule 9).
#[pyclass(frozen)]
struct NativeDipoleOrbits {
    orbits: DipoleOrbits,
    decodes: AtomicUsize,
}

impl NativeDipoleOrbits {
    fn new(orbits: DipoleOrbits) -> Self {
        Self {
            orbits,
            decodes: AtomicUsize::new(0),
        }
    }
}

#[pymethods]
impl NativeDipoleOrbits {
    /// Build from the Born basis on `legs`; `classes` are the candidate sets
    /// of identical adjoint labels.
    #[staticmethod]
    #[pyo3(signature = (legs, basis, classes, threads=None))]
    fn build(
        py: Python<'_>,
        legs: Vec<(i64, i32)>,
        basis: Vec<RawTensor>,
        classes: Vec<Vec<i64>>,
        threads: Option<usize>,
    ) -> PyResult<Self> {
        let basis = basis
            .into_iter()
            .map(tensor)
            .collect::<PyResult<Vec<_>>>()?;
        let orbits = on_threads(py, threads, || DipoleOrbits::build(&legs, &basis, &classes))?;
        Ok(Self::new(orbits))
    }

    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(Self::new(DipoleOrbits::from_json(text).map_err(error)?))
    }

    fn to_json(&self) -> PyResult<String> {
        self.orbits.to_json().map_err(error)
    }

    #[getter]
    fn representatives(&self) -> usize {
        self.orbits.representatives.len()
    }

    #[getter]
    fn generators(&self) -> usize {
        self.orbits.generators.len()
    }

    #[getter]
    fn coloured(&self) -> Vec<(i64, i32)> {
        self.orbits.coloured.clone()
    }

    /// Entries decoded to Python so far: `expand`'s fraction texts (i != j)
    /// and `expand_f64`'s dense floats (rule 9 of the migration plan;
    /// `digest` and `to_json` decode no entry).
    #[getter]
    fn decodes(&self) -> usize {
        self.decodes.load(Ordering::Relaxed)
    }

    fn check_conservation(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.orbits.check_conservation())
            .map_err(error)
    }

    /// Every entry as fraction texts: `{(i, j): rows}` over coloured leg
    /// labels i != j (`LoopSlice.dipoles`).
    fn expand(&self, py: Python<'_>) -> PyResult<RawDipoles> {
        let c = self.orbits.coloured.len();
        let n = self.orbits.basis;
        // Dense [i][j][b][b'] texts, allocated fallibly (the stored form's
        // index space was validated at load).
        let dense = py
            .detach(|| {
                self.orbits
                    .expand_dense(String::new(), cells::fraction_text)
            })
            .map_err(error)?;
        let entries = self.orbits.entry_count().map_err(error)?;
        self.decodes.fetch_add(entries, Ordering::Relaxed);
        let mut values = dense.into_iter();
        let mut out = Vec::with_capacity(c * c.saturating_sub(1));
        for (i, a) in self.orbits.coloured.iter().enumerate() {
            for (j, b) in self.orbits.coloured.iter().enumerate() {
                if i == j {
                    // The empty diagonal block (n * n <= c² n², validated).
                    values.by_ref().take(n * n).for_each(drop);
                    continue;
                }
                let rows: Vec<Vec<String>> =
                    (0..n).map(|_| values.by_ref().take(n).collect()).collect();
                out.push(((a.0, b.0), rows));
            }
        }
        Ok(out)
    }

    /// SHA-256 of the expanded entries in `LoopSlice.dipoles` order.
    fn digest(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.orbits.digest()).map_err(error)
    }

    /// Dense binary64 operators `[i][j][b][b']`.
    fn expand_f64(&self, py: Python<'_>) -> PyResult<Vec<f64>> {
        let out = py.detach(|| self.orbits.expand_f64()).map_err(error)?;
        self.decodes.fetch_add(out.len(), Ordering::Relaxed);
        Ok(out)
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeCellTable>()?;
    module.add_class::<NativeCellEngine>()?;
    module.add_class::<NativeAdjointOrbits>()?;
    module.add_class::<NativeDipoleOrbits>()?;
    module.add_function(wrap_pyfunction!(colour_gram, module)?)?;
    module.add_function(wrap_pyfunction!(contract_colours, module)?)?;
    module.add_function(wrap_pyfunction!(face_counts, module)?)?;
    module.add_function(wrap_pyfunction!(project_records, module)?)?;
    module.add_function(wrap_pyfunction!(overlap_degrees, module)?)?;
    module.add_function(wrap_pyfunction!(tensor_overlap_degrees, module)?)?;
    module.add_function(wrap_pyfunction!(directed_overlap_degrees, module)?)?;
    module.add_function(wrap_pyfunction!(operator_images, module)?)?;
    module.add_function(wrap_pyfunction!(live_classes, module)?)?;
    Ok(())
}
