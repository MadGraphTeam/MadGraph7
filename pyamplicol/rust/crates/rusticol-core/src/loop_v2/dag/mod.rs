// SPDX-License-Identifier: 0BSD

//! Native loop-request DAG builder (WP3.1 package A, `RUST-MIGRATION.md` §3.A).
//!
//! The Python oracle is the tree engine's loop-request mode,
//! `GenericDAGCompiler(loop_blocks=True).compile` (`generation/dag_compiler.py`),
//! followed by `loops/v2/blocks._blocks`. This module reproduces it from the
//! model and process tables of `loops/v2/dag_input.py` ([`input`]):
//!
//! * [`recursion`] sweeps the leg subsets in the oracle's order (masks by
//!   size, then value; ordered splits; candidate currents in creation order)
//!   and deduplicates currents by their full index, so ids follow Python's
//!   first-seen order (rule 5);
//! * [`colour`] is the loop-block colour engine (`ColorEngine._loop_block_*`,
//!   `ordered_combination_labels`, `combine`);
//! * [`grades`] grades the blocks whose trees mix N_c grades
//!   (U(1)-subtraction pieces) for leading and next-to-leading colour
//!   (`generation/dag_grades.split_u1_grades`);
//! * [`reuse`] is `assign_recursive_current_evaluation_reuse`: the
//!   value-equivalence derivation of the currents that share an evaluation
//!   contract (the closed colour components of the CT program's currents)
//!   in the oracle's bit-exact binary64 projective arithmetic ([`exact`]),
//!   and the regrouped evaluations;
//! * [`signs`] folds the colour-flow convention's external-fermion ordering
//!   signs (`apply_fermion_ordering_signs(coloured_lines=False)`);
//! * [`blocks`] groups the currents into blocks (`loops/v2/blocks._blocks`)
//!   and answers generation's current scans (`propagated_species`,
//!   `u1_leaking`);
//! * [`program`] is the DAG's interaction program for the native lane
//!   (`native_trees._interaction_program` on the runtime schema's layout).
//!
//! The result is a [`ColumnarDag`]: struct-of-arrays currents and
//! interactions with interned tables. It round-trips through a byte container
//! ([`ColumnarDag::to_bytes`], [`ColumnarDag::from_bytes`]) that Python decodes
//! into a `GenericDAG` (counted, rule 9) and encodes from one (the oracle side
//! of the digests). [`ColumnarDag::digest`] is the streaming flat-record
//! SHA-256 of the DAG part of the S1′/S1″ stage digests (rule 6).
//!
//! Nothing here calls Symbolica: the builder needs no licence (rule 4).

pub mod blocks;
pub mod colour;
pub mod exact;
pub mod grades;
pub mod input;
pub mod program;
pub mod recursion;
pub mod reuse;
pub mod signs;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Errors of the native DAG builder.
///
/// `Unsupported` names a feature the native builder does not cover yet
/// (rule 2: fail closed, never silently); `Invalid` mirrors an error the
/// Python oracle raises for the same input. An `Unsupported` message that
/// starts with a key of [`ADMISSION_REFUSALS`] and `": "` is a refusal of the
/// request itself (no lane covers it), which Python raises as its typed
/// generation refusal (`loops.v2.refusals.LOOP_ADMISSION_REFUSALS`).
#[derive(Debug, thiserror::Error)]
pub enum DagError {
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("{0}")]
    Invalid(String),
    #[error("malformed native DAG data: {0}")]
    Malformed(String),
}

pub type DagResult<T> = Result<T, DagError>;

/// [`DagError::Unsupported`] key: a merge of a coloured and a colour-singlet
/// fermion current, or a block whose coloured fermion ends do not pair (one
/// fermion line changing colour representation, e.g. at a leptoquark vertex
/// u e S): the colour flow signs coloured fermion lines and the Wick order
/// colour-singlet ones, so such a line has no sign (genericity audit 2,
/// G2-m1). Python admission refuses it before the DAG.
pub const FERMION_LINE_REPRESENTATION: &str = "fermion-line-representation";

/// The keys of the [`DagError::Unsupported`] refusals that refuse the request
/// (listed in Python's `LOOP_ADMISSION_REFUSALS`), not fallbacks.
pub const ADMISSION_REFUSALS: &[&str] = &[FERMION_LINE_REPRESENTATION];

pub(crate) fn refused<T>(key: &str, message: impl std::fmt::Display) -> DagResult<T> {
    debug_assert!(ADMISSION_REFUSALS.contains(&key));
    Err(DagError::Unsupported(format!("{key}: {message}")))
}

pub(crate) fn invalid<T>(message: impl Into<String>) -> DagResult<T> {
    Err(DagError::Invalid(message.into()))
}

pub(crate) fn unsupported<T>(message: impl Into<String>) -> DagResult<T> {
    Err(DagError::Unsupported(message.into()))
}

/// A current species: particle id, auxiliary kind and quantum-number flow
/// (all three are functions of the particle id in a model; the table keeps
/// what the DAG says).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Species {
    pub pdg: i64,
    pub auxiliary: Option<String>,
    pub quantum_numbers: Vec<(String, String)>,
}

/// `ColorState` of `generation/dag_types.py`. `nc_grade` is the N_c grade
/// of every current of a block whose trees have several grades ([`grades`]:
/// a current whose own derivations mix grades is split into one current per
/// grade, a current of one grade in such a block takes its grade as a
/// label); `None` for the currents of a block of one grade and without the
/// split (and absent from the JSON tables then).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ColourState {
    pub accuracy: String,
    pub sector_id: i64,
    pub line_groups: Vec<i64>,
    pub basis_key: Vec<String>,
    pub traces: Vec<Vec<u8>>,
    pub lines: Vec<Vec<u8>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nc_grade: Option<i64>,
}

impl ColourState {
    /// Labels of the closed traces and lines (`ColorState.closed_labels`).
    pub fn closed_labels(&self) -> Vec<u8> {
        let mut labels: Vec<u8> = self
            .traces
            .iter()
            .chain(self.lines.iter())
            .flatten()
            .copied()
            .collect();
        labels.sort_unstable();
        labels.dedup();
        labels
    }
}

/// An interaction's vertex: kind, oriented particles and lowering metadata.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VertexKey {
    pub kind: i64,
    pub particles: [i64; 3],
    pub backend: String,
    pub full_tensor_network_ready: bool,
}

pub type Orders = Vec<(String, i64)>;

/// Interned tables of a [`ColumnarDag`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Tables {
    pub species: Vec<Species>,
    pub flavours: Vec<Vec<i64>>,
    pub colours: Vec<ColourState>,
    pub orders: Vec<Orders>,
    pub vertices: Vec<VertexKey>,
}

/// Sentinel of a missing source leg / helicity / evaluation group.
pub const NO_LEG: i32 = -1;
pub const NO_HELICITY: i32 = i32::MIN;
pub const NO_GROUP: i64 = -1;

/// A loop-request DAG in columns (`GenericDAG` without process and colour
/// plan, which Python keeps).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ColumnarDag {
    pub tables: Tables,
    // currents
    pub c_species: Vec<u32>,
    pub c_mask: Vec<u64>,
    pub c_momentum: Vec<u64>,
    pub c_ancestry: Vec<u64>,
    pub c_chirality: Vec<i8>,
    pub c_spin: Vec<i32>,
    pub c_flavour: Vec<u32>,
    pub c_colour: Vec<u32>,
    pub c_orders: Vec<u32>,
    pub c_dimension: Vec<i32>,
    pub c_source: Vec<u8>,
    pub c_leg: Vec<i32>,
    pub c_helicity: Vec<i32>,
    /// `ordered_external_labels` of current `i`:
    /// `c_words[c_word_offsets[i]..c_word_offsets[i + 1]]`.
    pub c_word_offsets: Vec<u32>,
    pub c_words: Vec<u8>,
    pub sources: Vec<u32>,
    // interactions
    pub i_vertex: Vec<u32>,
    pub i_left: Vec<u32>,
    pub i_right: Vec<u32>,
    pub i_result: Vec<u32>,
    pub i_coupling: Vec<[f64; 2]>,
    pub i_colour_weight: Vec<[f64; 2]>,
    pub i_group: Vec<i64>,
    pub i_factor: Vec<[f64; 2]>,
}

const MAGIC: &[u8; 8] = b"PYAMDAG1";
pub const COLUMNS_SCHEMA: &str = "pyamplicol.loop-v2.dag-columns/1";
const DIGEST_SCHEMA: &[u8] = b"pyamplicol.loop-v2.dag-digest/1\0";

#[derive(Serialize, Deserialize)]
struct Header {
    schema: String,
    tables: Tables,
    columns: Vec<(String, String, u64)>,
}

macro_rules! column_io {
    ($( $name:ident : $ty:ty => $dtype:literal ),* $(,)?) => {
        fn write_columns(&self, columns: &mut Vec<(String, String, u64)>, payload: &mut Vec<u8>) {
            $(
                columns.push((stringify!($name).to_string(), $dtype.to_string(), self.$name.len() as u64));
                for value in &self.$name {
                    payload.extend_from_slice(&value.to_le_bytes());
                }
            )*
        }
        fn read_columns(&mut self, columns: &[(String, String, u64)], mut payload: &[u8]) -> DagResult<()> {
            let mut seen = std::collections::BTreeSet::new();
            for (name, dtype, count) in columns {
                let count = *count as usize;
                match name.as_str() {
                    $(
                        stringify!($name) => {
                            if dtype != $dtype {
                                return Err(DagError::Malformed(format!("column {name} has dtype {dtype}")));
                            }
                            const WIDTH: usize = std::mem::size_of::<$ty>();
                            let bytes = count.checked_mul(WIDTH).filter(|n| *n <= payload.len()).ok_or_else(
                                || DagError::Malformed(format!("column {name} is truncated")),
                            )?;
                            self.$name = payload[..bytes]
                                .chunks_exact(WIDTH)
                                .map(|chunk| <$ty>::from_le_bytes(chunk.try_into().expect("chunk width")))
                                .collect();
                            payload = &payload[bytes..];
                        }
                    )*
                    "i_coupling" | "i_colour_weight" | "i_factor" => {
                        if dtype != "f64" || count % 2 != 0 {
                            return Err(DagError::Malformed(format!("column {name} is not f64 pairs")));
                        }
                        let bytes = count.checked_mul(8).filter(|n| *n <= payload.len()).ok_or_else(
                            || DagError::Malformed(format!("column {name} is truncated")),
                        )?;
                        let values: Vec<[f64; 2]> = payload[..bytes]
                            .chunks_exact(16)
                            .map(|chunk| {
                                [
                                    f64::from_le_bytes(chunk[..8].try_into().expect("width")),
                                    f64::from_le_bytes(chunk[8..].try_into().expect("width")),
                                ]
                            })
                            .collect();
                        match name.as_str() {
                            "i_coupling" => self.i_coupling = values,
                            "i_colour_weight" => self.i_colour_weight = values,
                            _ => self.i_factor = values,
                        }
                        payload = &payload[bytes..];
                    }
                    _ => return Err(DagError::Malformed(format!("unknown column {name}"))),
                }
                if !seen.insert(name.clone()) {
                    return Err(DagError::Malformed(format!("column {name} repeated")));
                }
            }
            if !payload.is_empty() {
                return Err(DagError::Malformed("trailing column bytes".into()));
            }
            Ok(())
        }
    };
}

impl ColumnarDag {
    column_io! {
        c_species: u32 => "u32",
        c_mask: u64 => "u64",
        c_momentum: u64 => "u64",
        c_ancestry: u64 => "u64",
        c_chirality: i8 => "i8",
        c_spin: i32 => "i32",
        c_flavour: u32 => "u32",
        c_colour: u32 => "u32",
        c_orders: u32 => "u32",
        c_dimension: i32 => "i32",
        c_source: u8 => "u8",
        c_leg: i32 => "i32",
        c_helicity: i32 => "i32",
        c_word_offsets: u32 => "u32",
        c_words: u8 => "u8",
        sources: u32 => "u32",
        i_vertex: u32 => "u32",
        i_left: u32 => "u32",
        i_right: u32 => "u32",
        i_result: u32 => "u32",
        i_group: i64 => "i64",
    }

    pub fn current_count(&self) -> usize {
        self.c_species.len()
    }

    pub fn interaction_count(&self) -> usize {
        self.i_vertex.len()
    }

    /// `ordered_external_labels` of current `id`.
    pub fn word(&self, id: usize) -> &[u8] {
        &self.c_words[self.c_word_offsets[id] as usize..self.c_word_offsets[id + 1] as usize]
    }

    /// The byte container Python decodes (`dag_input._read_container`).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut columns = Vec::new();
        let mut payload = Vec::new();
        self.write_columns(&mut columns, &mut payload);
        for (name, values) in [
            ("i_coupling", &self.i_coupling),
            ("i_colour_weight", &self.i_colour_weight),
            ("i_factor", &self.i_factor),
        ] {
            columns.push((name.to_string(), "f64".to_string(), 2 * values.len() as u64));
            for [re, im] in values {
                payload.extend_from_slice(&re.to_le_bytes());
                payload.extend_from_slice(&im.to_le_bytes());
            }
        }
        let header = serde_json::to_vec(&Header {
            schema: COLUMNS_SCHEMA.to_string(),
            tables: self.tables.clone(),
            columns,
        })
        .expect("serialisable header");
        let mut out = Vec::with_capacity(16 + header.len() + payload.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(&header);
        out.extend_from_slice(&payload);
        out
    }

    /// Reads [`ColumnarDag::to_bytes`] (or Python's `encode_dag`) and checks
    /// that every column and reference is consistent.
    pub fn from_bytes(bytes: &[u8]) -> DagResult<Self> {
        if bytes.len() < 16 || &bytes[..8] != MAGIC {
            return Err(DagError::Malformed("not a native DAG container".into()));
        }
        let length = u64::from_le_bytes(bytes[8..16].try_into().expect("width")) as usize;
        let header_end = 16usize
            .checked_add(length)
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| DagError::Malformed("truncated header".into()))?;
        let header: Header = serde_json::from_slice(&bytes[16..header_end])
            .map_err(|error| DagError::Malformed(format!("header: {error}")))?;
        if header.schema != COLUMNS_SCHEMA {
            return Err(DagError::Malformed(format!("schema {:?}", header.schema)));
        }
        let mut dag = ColumnarDag {
            tables: header.tables,
            ..Default::default()
        };
        dag.read_columns(&header.columns, &bytes[header_end..])?;
        dag.validate()?;
        Ok(dag)
    }

    fn validate(&self) -> DagResult<()> {
        let n = self.current_count();
        let m = self.interaction_count();
        let malformed = |what: &str| Err(DagError::Malformed(what.to_string()));
        for column in [
            self.c_mask.len(),
            self.c_momentum.len(),
            self.c_ancestry.len(),
            self.c_chirality.len(),
            self.c_spin.len(),
            self.c_flavour.len(),
            self.c_colour.len(),
            self.c_orders.len(),
            self.c_dimension.len(),
            self.c_source.len(),
            self.c_leg.len(),
            self.c_helicity.len(),
        ] {
            if column != n {
                return malformed("current columns differ in length");
            }
        }
        if self.c_word_offsets.len() != n + 1
            || self.c_word_offsets.first().copied().unwrap_or(1) != 0
            || self.c_word_offsets.windows(2).any(|pair| pair[0] > pair[1])
            || *self.c_word_offsets.last().expect("n + 1 offsets") as usize != self.c_words.len()
        {
            return malformed("word offsets");
        }
        for column in [
            self.i_left.len(),
            self.i_right.len(),
            self.i_result.len(),
            self.i_coupling.len(),
            self.i_colour_weight.len(),
            self.i_group.len(),
            self.i_factor.len(),
        ] {
            if column != m {
                return malformed("interaction columns differ in length");
            }
        }
        let t = &self.tables;
        let within = |ids: &[u32], len: usize| ids.iter().all(|id| (*id as usize) < len);
        if !within(&self.c_species, t.species.len())
            || !within(&self.c_flavour, t.flavours.len())
            || !within(&self.c_colour, t.colours.len())
            || !within(&self.c_orders, t.orders.len())
            || !within(&self.i_vertex, t.vertices.len())
            || !within(&self.i_left, n)
            || !within(&self.i_right, n)
            || !within(&self.i_result, n)
            || !within(&self.sources, n)
        {
            return malformed("reference out of range");
        }
        Ok(())
    }

    /// Streaming SHA-256 over the DAG's flat records (rule 6), resolved
    /// through the tables so that interning order does not matter.
    /// The record layout is documented in `loops/v2/dag_input.py`.
    pub fn digest(&self) -> String {
        let mut hash = Sha256::new();
        let mut record = Vec::with_capacity(256);
        hash.update(DIGEST_SCHEMA);
        hash.update((self.current_count() as u64).to_le_bytes());
        hash.update((self.interaction_count() as u64).to_le_bytes());
        hash.update((self.sources.len() as u64).to_le_bytes());
        let t = &self.tables;
        for id in 0..self.current_count() {
            record.clear();
            let species = &t.species[self.c_species[id] as usize];
            put_i64(&mut record, species.pdg);
            put_opt_str(&mut record, species.auxiliary.as_deref());
            put_u32(&mut record, species.quantum_numbers.len() as u32);
            for (key, value) in &species.quantum_numbers {
                put_str(&mut record, key);
                put_str(&mut record, value);
            }
            put_u64(&mut record, self.c_mask[id]);
            put_u64(&mut record, self.c_momentum[id]);
            put_bytes(&mut record, self.word(id));
            put_u64(&mut record, self.c_ancestry[id]);
            record.push(self.c_chirality[id] as u8);
            put_i32(&mut record, self.c_spin[id]);
            let flavour = &t.flavours[self.c_flavour[id] as usize];
            put_u32(&mut record, flavour.len() as u32);
            for value in flavour {
                put_i64(&mut record, *value);
            }
            let colour = &t.colours[self.c_colour[id] as usize];
            put_str(&mut record, &colour.accuracy);
            put_i64(&mut record, colour.sector_id);
            put_u32(&mut record, colour.line_groups.len() as u32);
            for group in &colour.line_groups {
                put_i64(&mut record, *group);
            }
            put_u32(&mut record, colour.basis_key.len() as u32);
            for key in &colour.basis_key {
                put_str(&mut record, key);
            }
            if let Some(grade) = colour.nc_grade {
                // A trace count is never u32::MAX: the marker is unambiguous
                // and states without a grade keep their records.
                put_u32(&mut record, u32::MAX);
                put_i64(&mut record, grade);
            }
            for parts in [&colour.traces, &colour.lines] {
                put_u32(&mut record, parts.len() as u32);
                for part in parts {
                    put_bytes(&mut record, part);
                }
            }
            let orders = &t.orders[self.c_orders[id] as usize];
            put_u32(&mut record, orders.len() as u32);
            for (name, value) in orders {
                put_str(&mut record, name);
                put_i64(&mut record, *value);
            }
            put_i32(&mut record, self.c_dimension[id]);
            record.push(self.c_source[id]);
            put_i32(&mut record, self.c_leg[id]);
            put_i32(&mut record, self.c_helicity[id]);
            hash.update(&record);
        }
        for source in &self.sources {
            hash.update(source.to_le_bytes());
        }
        for id in 0..self.interaction_count() {
            record.clear();
            let vertex = &t.vertices[self.i_vertex[id] as usize];
            put_i64(&mut record, vertex.kind);
            for particle in vertex.particles {
                put_i64(&mut record, particle);
            }
            put_u32(&mut record, self.i_left[id]);
            put_u32(&mut record, self.i_right[id]);
            put_u32(&mut record, self.i_result[id]);
            for pair in [self.i_coupling[id], self.i_colour_weight[id]] {
                put_u64(&mut record, pair[0].to_bits());
                put_u64(&mut record, pair[1].to_bits());
            }
            put_str(&mut record, &vertex.backend);
            record.push(vertex.full_tensor_network_ready as u8);
            put_i64(&mut record, self.i_group[id]);
            put_u64(&mut record, self.i_factor[id][0].to_bits());
            put_u64(&mut record, self.i_factor[id][1].to_bits());
            hash.update(&record);
        }
        hash.finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, value: &[u8]) {
    put_u32(out, value.len() as u32);
    out.extend_from_slice(value);
}

fn put_str(out: &mut Vec<u8>, value: &str) {
    put_bytes(out, value.as_bytes());
}

fn put_opt_str(out: &mut Vec<u8>, value: Option<&str>) {
    match value {
        None => out.push(0),
        Some(value) => {
            out.push(1);
            put_str(out, value);
        }
    }
}

/// Builds the loop-request DAG of `input` (the JSON of
/// `dag_input.dag_input`): recursion, prune, N_c grade split (leading and
/// next-to-leading colour), colour-connection signs, evaluation reuse,
/// colour-flow signs. The connection signs depend on the colour state, which
/// the evaluation contracts of the reuse leave out, so they are folded
/// before the reuse derives its value equivalences.
pub fn build(input: &input::DagInput) -> DagResult<ColumnarDag> {
    let mut dag = recursion::compile(input)?;
    if dag.finish {
        recursion::prune_to_admitted(input, &mut dag);
        if input.u1_grades {
            grades::split_u1_grades(input, &mut dag)?;
        }
        signs::apply_connection_signs(input, &mut dag)?;
        reuse::assign_reuse(input, &mut dag)?;
        signs::apply_colour_flow_signs(input, &mut dag)?;
    }
    Ok(dag.into_columns(input))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_round_trips() {
        let dag = ColumnarDag {
            tables: Tables {
                species: vec![Species {
                    pdg: 21,
                    auxiliary: None,
                    quantum_numbers: vec![("charge".into(), "0".into())],
                }],
                flavours: vec![vec![21]],
                colours: vec![ColourState {
                    accuracy: "full".into(),
                    sector_id: 0,
                    line_groups: vec![0],
                    basis_key: vec![],
                    traces: vec![],
                    lines: vec![],
                    nc_grade: None,
                }],
                orders: vec![vec![], vec![("QCD".into(), 1)]],
                vertices: vec![VertexKey {
                    kind: 7,
                    particles: [21, 21, 21],
                    backend: "spenso-ufo".into(),
                    full_tensor_network_ready: true,
                }],
            },
            c_species: vec![0, 0, 0],
            c_mask: vec![1, 2, 3],
            c_momentum: vec![1, 2, 3],
            c_ancestry: vec![1, 2, 3],
            c_chirality: vec![0, 0, 0],
            c_spin: vec![-1, 1, 0],
            c_flavour: vec![0, 0, 0],
            c_colour: vec![0, 0, 0],
            c_orders: vec![0, 0, 1],
            c_dimension: vec![4, 4, 4],
            c_source: vec![1, 1, 0],
            c_leg: vec![1, 2, NO_LEG],
            c_helicity: vec![-1, 1, NO_HELICITY],
            c_word_offsets: vec![0, 1, 2, 4],
            c_words: vec![1, 2, 1, 2],
            sources: vec![0, 1],
            i_vertex: vec![0],
            i_left: vec![0],
            i_right: vec![1],
            i_result: vec![2],
            i_coupling: vec![[1.0, 0.0]],
            i_colour_weight: vec![[0.0, -1.0]],
            i_group: vec![0],
            i_factor: vec![[-0.0, 1.0]],
        };
        let bytes = dag.to_bytes();
        let back = ColumnarDag::from_bytes(&bytes).expect("round trip");
        assert_eq!(back.digest(), dag.digest());
        assert_eq!(back.i_factor[0][0].to_bits(), (-0.0f64).to_bits());
        assert_eq!(back.word(2), &[1, 2]);
    }
}
