// SPDX-License-Identifier: 0BSD

//! The binary tree-program member (M-FINAL MF-8G-1, `loops/v2/native_trees.py`
//! `NativeTrees.binary`): a container of the tree program's parts, so that
//! neither generation nor load holds a large program as one JSON text or as
//! Python objects.
//!
//! **Container** (little-endian): the magic `PYAMTRP1`, a `u32` section
//! count, then per section a `u32` name length, the UTF-8 name, a `u8`
//! encoding (0 JSON, 1 binary), a `u64` payload length and the payload.
//! Sections:
//! * `head` (JSON, required): the program (`TreeFile`) without the keys
//!   stored in sections of their own;
//! * one JSON section per moved top-level key (`interactions`,
//!   `block_rows`, `block_row_terms`, `born`, `counterterms`,
//!   `counterterm_trees`, `layouts`, `weights`), each its key's value;
//! * binary sections that replace one large array of a JSON part, which
//!   then omits it: `interactions.steps` and
//!   `counterterm_trees.interactions.steps` ([`StepColumns`]) and
//!   `counterterms.cells` ([`CellTable`]).
//!
//! **Columns.** Integer columns are narrow: a `u8` width (4 or 8; signed
//! columns the same), a `u64` count and the values; the writer takes 4
//! bytes when every value fits (no cap: 8 otherwise). Texts are a `u32`
//! byte length and UTF-8.
//!
//! The JSON program stays the oracle: [`StepColumns::json`] and the cell
//! list of [`CellTable::list`] decode a binary part to the oracle's form,
//! and `TreeProgram::load` reads both forms into the same program
//! (`tests/unit/test_loop_v2_tree_binary.py`).

use super::{BornFile, CountertermFile, CountertermTreesFile, InteractionFile, StepFile, TreeFile};
use super::{exact, invalid, require};
use crate::RusticolResult;
use crate::loop_v2::interference::MetricCells;
use std::collections::{BTreeMap, BTreeSet};
use symbolica::domains::rational::Rational;

/// The container's magic.
pub const MAGIC: &[u8; 8] = b"PYAMTRP1";
/// Section encodings.
pub const JSON: u8 = 0;
pub const BINARY: u8 = 1;
/// The binary section names.
pub const TREE_STEPS: &str = "interactions.steps";
pub const COUNTERTERM_STEPS: &str = "counterterm_trees.interactions.steps";
pub const CELLS: &str = "counterterms.cells";
/// The JSON section names besides `head`: moved top-level keys.
pub const KEYS: [&str; 8] = [
    "interactions",
    "block_rows",
    "block_row_terms",
    "born",
    "counterterms",
    "counterterm_trees",
    "layouts",
    "weights",
];
const STEPS_VERSION: u32 = 1;
const CELLS_VERSION: u32 = 1;

/// Whether `bytes` is a container (else a JSON tree program).
pub fn is_container(bytes: &[u8]) -> bool {
    bytes.starts_with(MAGIC)
}

// ------------------------------------------------------------------ columns

/// An unsigned column, 4 bytes per value when every value fits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Column {
    U32(Vec<u32>),
    U64(Vec<u64>),
}

impl Default for Column {
    fn default() -> Self {
        Column::U32(Vec::new())
    }
}

impl Column {
    /// The narrowest column of `values`.
    pub fn of(values: Vec<u64>) -> Self {
        if values.iter().all(|&v| v <= u64::from(u32::MAX)) {
            Column::U32(values.into_iter().map(|v| v as u32).collect())
        } else {
            Column::U64(values)
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Column::U32(v) => v.len(),
            Column::U64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(&self, index: usize) -> u64 {
        match self {
            Column::U32(v) => u64::from(v[index]),
            Column::U64(v) => v[index],
        }
    }

    #[inline]
    pub fn at(&self, index: usize) -> usize {
        self.get(index) as usize
    }
}

/// A signed column, 4 bytes per value when every value fits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SignedColumn {
    I32(Vec<i32>),
    I64(Vec<i64>),
}

impl Default for SignedColumn {
    fn default() -> Self {
        SignedColumn::I32(Vec::new())
    }
}

impl SignedColumn {
    pub fn of(values: Vec<i64>) -> Self {
        if values
            .iter()
            .all(|&v| i64::from(i32::MIN) <= v && v <= i64::from(i32::MAX))
        {
            SignedColumn::I32(values.into_iter().map(|v| v as i32).collect())
        } else {
            SignedColumn::I64(values)
        }
    }

    pub fn len(&self) -> usize {
        match self {
            SignedColumn::I32(v) => v.len(),
            SignedColumn::I64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(&self, index: usize) -> i64 {
        match self {
            SignedColumn::I32(v) => i64::from(v[index]),
            SignedColumn::I64(v) => v[index],
        }
    }
}

/// A little-endian writer.
#[derive(Default)]
pub struct Writer {
    pub bytes: Vec<u8>,
}

impl Writer {
    pub fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }
    pub fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    pub fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    pub fn i32(&mut self, value: i32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }
    pub fn text(&mut self, value: &str) {
        self.u32(u32::try_from(value.len()).expect("a text below 4 GiB"));
        self.bytes.extend_from_slice(value.as_bytes());
    }
    pub fn column(&mut self, column: &Column) {
        match column {
            Column::U32(values) => {
                self.u8(4);
                self.u64(values.len() as u64);
                self.bytes.reserve(4 * values.len());
                for v in values {
                    self.bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
            Column::U64(values) => {
                self.u8(8);
                self.u64(values.len() as u64);
                self.bytes.reserve(8 * values.len());
                for v in values {
                    self.bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
    pub fn signed(&mut self, column: &SignedColumn) {
        match column {
            SignedColumn::I32(values) => {
                self.u8(4);
                self.u64(values.len() as u64);
                for v in values {
                    self.bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
            SignedColumn::I64(values) => {
                self.u8(8);
                self.u64(values.len() as u64);
                for v in values {
                    self.bytes.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
    }
}

/// A little-endian reader over a section.
pub struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, at: 0 }
    }

    fn take(&mut self, length: usize) -> RusticolResult<&'a [u8]> {
        let end = self
            .at
            .checked_add(length)
            .filter(|&end| end <= self.data.len())
            .ok_or_else(|| invalid("a truncated binary tree-program section"))?;
        let out = &self.data[self.at..end];
        self.at = end;
        Ok(out)
    }

    pub fn u8(&mut self) -> RusticolResult<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn u32(&mut self) -> RusticolResult<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn u64(&mut self) -> RusticolResult<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    pub fn i32(&mut self) -> RusticolResult<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    pub fn count(&mut self) -> RusticolResult<usize> {
        usize::try_from(self.u64()?).map_err(|_| invalid("a section count"))
    }
    pub fn text(&mut self) -> RusticolResult<String> {
        let length = self.u32()? as usize;
        String::from_utf8(self.take(length)?.to_vec())
            .map_err(|_| invalid("a section text is not UTF-8"))
    }
    pub fn column(&mut self) -> RusticolResult<Column> {
        let width = self.u8()?;
        let count = self.count()?;
        match width {
            4 => {
                let raw = self.take(count.checked_mul(4).ok_or_else(|| invalid("column"))?)?;
                Ok(Column::U32(
                    raw.chunks_exact(4)
                        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                ))
            }
            8 => {
                let raw = self.take(count.checked_mul(8).ok_or_else(|| invalid("column"))?)?;
                Ok(Column::U64(
                    raw.chunks_exact(8)
                        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                ))
            }
            _ => Err(invalid("a column width")),
        }
    }
    pub fn signed(&mut self) -> RusticolResult<SignedColumn> {
        let width = self.u8()?;
        let count = self.count()?;
        match width {
            4 => {
                let raw = self.take(count.checked_mul(4).ok_or_else(|| invalid("column"))?)?;
                Ok(SignedColumn::I32(
                    raw.chunks_exact(4)
                        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                ))
            }
            8 => {
                let raw = self.take(count.checked_mul(8).ok_or_else(|| invalid("column"))?)?;
                Ok(SignedColumn::I64(
                    raw.chunks_exact(8)
                        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                        .collect(),
                ))
            }
            _ => Err(invalid("a column width")),
        }
    }
    pub fn done(&self) -> RusticolResult<()> {
        require(self.at == self.data.len(), "trailing bytes in a tree-program section")
    }
}

// ---------------------------------------------------------------- container

/// One section of a container.
pub struct Section<'a> {
    pub name: String,
    pub encoding: u8,
    pub payload: &'a [u8],
}

/// A container of `sections` (name, encoding, payload).
pub fn write_container(sections: &[(&str, u8, &[u8])]) -> Vec<u8> {
    let total: usize = sections
        .iter()
        .map(|(name, _, payload)| 13 + name.len() + payload.len())
        .sum();
    let mut out = Writer {
        bytes: Vec::with_capacity(12 + total),
    };
    out.bytes.extend_from_slice(MAGIC);
    out.u32(u32::try_from(sections.len()).expect("sections"));
    for (name, encoding, payload) in sections {
        out.text(name);
        out.u8(*encoding);
        out.u64(payload.len() as u64);
        out.bytes.extend_from_slice(payload);
    }
    out.bytes
}

/// The sections of a container, in order; names unique.
pub fn read_container(bytes: &[u8]) -> RusticolResult<Vec<Section<'_>>> {
    require(is_container(bytes), "not a binary tree program")?;
    let mut reader = Reader::new(&bytes[MAGIC.len()..]);
    let count = reader.u32()? as usize;
    let mut seen = BTreeSet::new();
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let name = reader.text()?;
        let encoding = reader.u8()?;
        require(encoding == JSON || encoding == BINARY, "a section encoding")?;
        let length = reader.count()?;
        let payload = reader.take(length)?;
        require(seen.insert(name.clone()), "a repeated tree-program section")?;
        out.push(Section {
            name,
            encoding,
            payload,
        });
    }
    reader.done()?;
    Ok(out)
}

/// The binary parts of a container that replace arrays of its JSON parts.
#[derive(Default)]
pub struct Overrides {
    pub tree_steps: Option<StepColumns>,
    pub counterterm_steps: Option<StepColumns>,
    pub cells: Option<CellTable>,
}

fn json<T: serde::de::DeserializeOwned>(name: &str, payload: &[u8]) -> RusticolResult<T> {
    serde_json::from_slice(payload).map_err(|e| invalid(format!("section {name}: {e}")))
}

/// The program of a container and its binary parts.
pub(super) fn tree_file(bytes: &[u8]) -> RusticolResult<(TreeFile, Overrides)> {
    let sections = read_container(bytes)?;
    let head = sections
        .iter()
        .find(|section| section.name == "head" && section.encoding == JSON)
        .ok_or_else(|| invalid("a binary tree program without its head"))?;
    let mut file: TreeFile = json("head", head.payload)?;
    let mut overrides = Overrides::default();
    let mut moved = BTreeSet::new();
    for section in &sections {
        let name = section.name.as_str();
        let payload = section.payload;
        if name == "head" {
            continue;
        }
        if section.encoding == JSON {
            require(KEYS.contains(&name), "an unknown JSON tree-program section")?;
            moved.insert(name);
            match name {
                "interactions" => file.interactions = json::<InteractionFile>(name, payload)?,
                "block_rows" => file.block_rows = json(name, payload)?,
                "block_row_terms" => file.block_row_terms = json(name, payload)?,
                "born" => file.born = json::<BornFile>(name, payload)?,
                "counterterms" => {
                    file.counterterms = json::<Option<CountertermFile>>(name, payload)?
                }
                "counterterm_trees" => {
                    file.counterterm_trees = json::<Option<CountertermTreesFile>>(name, payload)?
                }
                "layouts" => file.layouts = json(name, payload)?,
                "weights" => file.weights = json(name, payload)?,
                _ => unreachable!(),
            }
            continue;
        }
        match name {
            TREE_STEPS => overrides.tree_steps = Some(StepColumns::decode(payload)?),
            COUNTERTERM_STEPS => overrides.counterterm_steps = Some(StepColumns::decode(payload)?),
            CELLS => overrides.cells = Some(CellTable::decode(payload)?),
            _ => return Err(invalid("an unknown binary tree-program section")),
        }
    }
    // Every required part is the head's or a section's (the writer moves
    // all of them); the optional ones default to absent.
    for key in ["interactions", "block_rows", "born", "layouts", "weights"] {
        require(moved.contains(key), "a binary tree program misses a part")?;
    }
    require(
        overrides.tree_steps.is_none() || file.interactions.steps.is_empty(),
        "tree steps in both forms",
    )?;
    require(
        overrides.counterterm_steps.is_none()
            || file
                .counterterm_trees
                .as_ref()
                .is_some_and(|ct| ct.interactions.steps.is_empty()),
        "counterterm steps without their program or in both forms",
    )?;
    require(
        overrides.cells.is_none()
            || file
                .counterterms
                .as_ref()
                .is_some_and(|ct| ct.cells.is_empty()),
        "overlap cells without their record or in both forms",
    )?;
    Ok((file, overrides))
}

// -------------------------------------------------------------------- steps

/// The steps of an interaction program as columns (`native_trees.
/// _interaction_program`'s `steps`): per step its row, accumulator and
/// output counts; the rows' nine columns (kernel, left start, left
/// dimension, right start, right dimension, left momentum, right momentum,
/// accumulator, colour weight index into `weights`, the exact (re, im)
/// texts); the accumulators' dimensions; the outputs' five columns
/// (accumulator, propagator or -1, momentum start, state start, dimension);
/// and, per step with them, its accumulators' slot sets.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepColumns {
    pub weights: Vec<(String, String)>,
    pub rows: Vec<u64>,
    pub row_columns: [Column; 9],
    pub accumulators: Vec<u64>,
    pub dimensions: Column,
    pub outputs: Vec<u64>,
    pub output_columns: [Column; 4],
    pub propagators: SignedColumn,
    /// Per step whether it records slot sets (a counterterm program).
    pub masked: Vec<bool>,
    pub masks: Column,
}

/// One step's view: offsets into the flat columns.
#[derive(Clone, Copy, Debug)]
pub struct StepRange {
    pub rows: (usize, usize),
    pub accumulators: (usize, usize),
    pub outputs: (usize, usize),
    /// Into `masks`: the step's slot sets (empty without them).
    pub masks: (usize, usize),
}

fn columns<const N: usize>(values: [Vec<u64>; N]) -> [Column; N] {
    values.map(Column::of)
}

impl StepColumns {
    /// Columns of the native program's steps (`loop_v2::dag::program`),
    /// with the exact texts of each colour weight. The weights are renumbered
    /// in their first use by the rows (unused ones dropped), as
    /// [`StepColumns::from_json`] numbers them: equal steps give equal
    /// columns whichever form they come from.
    pub fn from_program(
        steps: &[crate::loop_v2::dag::program::Step],
        weights: Vec<(String, String)>,
    ) -> RusticolResult<Self> {
        let mut renumbered = vec![u64::MAX; weights.len()];
        let mut used = Vec::new();
        for step in steps {
            for row in &step.rows {
                let w = usize::try_from(row[8])
                    .ok()
                    .filter(|&w| w < weights.len())
                    .ok_or_else(|| invalid("a step row's colour weight"))?;
                if renumbered[w] == u64::MAX {
                    renumbered[w] = used.len() as u64;
                    used.push(weights[w].clone());
                }
            }
        }
        let weights = used;
        let total: usize = steps.iter().map(|s| s.rows.len()).sum();
        let mut rows: [Vec<u64>; 9] = Default::default();
        for column in &mut rows {
            column.reserve(total);
        }
        let mut dimensions = Vec::new();
        let mut outs: [Vec<u64>; 4] = Default::default();
        let mut propagators = Vec::new();
        let mut masks = Vec::new();
        let mut out = StepColumns {
            weights,
            ..Default::default()
        };
        let unsigned = |value: i64| {
            u64::try_from(value).map_err(|_| invalid("a negative step entry"))
        };
        for step in steps {
            out.rows.push(step.rows.len() as u64);
            for row in &step.rows {
                for (k, value) in row[..8].iter().enumerate() {
                    rows[k].push(unsigned(*value)?);
                }
                rows[8].push(renumbered[row[8] as usize]);
            }
            out.accumulators.push(step.accumulators.len() as u64);
            for &dimension in &step.accumulators {
                dimensions.push(unsigned(dimension)?);
            }
            out.outputs.push(step.outputs.len() as u64);
            for output in &step.outputs {
                outs[0].push(unsigned(output[0])?);
                propagators.push(output[1]);
                outs[1].push(unsigned(output[2])?);
                outs[2].push(unsigned(output[3])?);
                outs[3].push(unsigned(output[4])?);
            }
            // As `V2Program.steps`: slot sets only where the program has them.
            let masked = !step.accumulator_masks.is_empty();
            out.masked.push(masked);
            masks.extend(step.accumulator_masks.iter().map(|&m| u64::from(m)));
        }
        require(
            rows[8].iter().all(|&w| (w as usize) < out.weights.len()),
            "a step row's colour weight",
        )?;
        out.row_columns = columns(rows);
        out.dimensions = Column::of(dimensions);
        out.output_columns = columns(outs);
        out.propagators = SignedColumn::of(propagators);
        out.masks = Column::of(masks);
        Ok(out)
    }

    /// Columns of the oracle's JSON steps (`_interaction_program`'s
    /// `steps`; the weight texts interned in first use).
    pub fn from_json(json: &[u8]) -> RusticolResult<Self> {
        Self::from_files(
            serde_json::from_slice(json).map_err(|e| invalid(format!("steps: {e}")))?,
        )
    }

    /// Columns of JSON steps (`StepFile`s; the weight texts interned in
    /// first use).
    pub(super) fn from_files(steps: Vec<StepFile>) -> RusticolResult<Self> {
        let mut texts: BTreeMap<(String, String), u64> = BTreeMap::new();
        let mut out = StepColumns::default();
        let mut rows: [Vec<u64>; 9] = Default::default();
        let mut dimensions = Vec::new();
        let mut outs: [Vec<u64>; 4] = Default::default();
        let mut propagators = Vec::new();
        let mut masks = Vec::new();
        for step in steps {
            out.rows.push(step.interactions.len() as u64);
            for (k, l, ld, r, rd, pl, pr, a, re, im) in step.interactions {
                for (column, value) in rows.iter_mut().zip([k, l, ld, r, rd, pl, pr, a]) {
                    column.push(value as u64);
                }
                let next = texts.len() as u64;
                let key = (re, im);
                let index = match texts.get(&key) {
                    Some(&index) => index,
                    None => {
                        out.weights.push(key.clone());
                        texts.insert(key, next);
                        next
                    }
                };
                rows[8].push(index);
            }
            out.accumulators.push(step.accumulators.len() as u64);
            dimensions.extend(step.accumulators.iter().map(|&d| d as u64));
            out.outputs.push(step.outputs.len() as u64);
            for (a, p, m, s, d) in step.outputs {
                outs[0].push(a as u64);
                propagators.push(p);
                outs[1].push(m as u64);
                outs[2].push(s as u64);
                outs[3].push(d as u64);
            }
            out.masked.push(!step.accumulator_masks.is_empty());
            masks.extend(step.accumulator_masks.iter().map(|&m| m as u64));
        }
        out.row_columns = columns(rows);
        out.dimensions = Column::of(dimensions);
        out.output_columns = columns(outs);
        out.propagators = SignedColumn::of(propagators);
        out.masks = Column::of(masks);
        Ok(out)
    }

    /// Per step its ranges into the flat columns.
    pub fn ranges(&self) -> Vec<StepRange> {
        let (mut r, mut a, mut o, mut m) = (0, 0, 0, 0);
        (0..self.rows.len())
            .map(|s| {
                let (rows, accumulators, outputs) = (
                    self.rows[s] as usize,
                    self.accumulators[s] as usize,
                    self.outputs[s] as usize,
                );
                let masks = if self.masked[s] { accumulators } else { 0 };
                let range = StepRange {
                    rows: (r, r + rows),
                    accumulators: (a, a + accumulators),
                    outputs: (o, o + outputs),
                    masks: (m, m + masks),
                };
                r += rows;
                a += accumulators;
                o += outputs;
                m += masks;
                range
            })
            .collect()
    }

    fn validate(&self) -> RusticolResult<()> {
        let steps = self.rows.len();
        require(
            self.accumulators.len() == steps
                && self.outputs.len() == steps
                && self.masked.len() == steps,
            "step column counts",
        )?;
        let rows: u64 = self.rows.iter().sum();
        let accumulators: u64 = self.accumulators.iter().sum();
        let outputs: u64 = self.outputs.iter().sum();
        let masks: u64 = self
            .accumulators
            .iter()
            .zip(&self.masked)
            .filter(|(_, masked)| **masked)
            .map(|(a, _)| a)
            .sum();
        require(
            self.row_columns.iter().all(|c| c.len() as u64 == rows)
                && self.dimensions.len() as u64 == accumulators
                && self.output_columns.iter().all(|c| c.len() as u64 == outputs)
                && self.propagators.len() as u64 == outputs
                && self.masks.len() as u64 == masks,
            "step column lengths",
        )?;
        let weights = self.weights.len() as u64;
        require(
            (0..rows as usize).all(|k| self.row_columns[8].get(k) < weights),
            "a step row's colour weight",
        )
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Writer::default();
        out.u32(STEPS_VERSION);
        out.u64(self.weights.len() as u64);
        for (re, im) in &self.weights {
            out.text(re);
            out.text(im);
        }
        out.column(&Column::of(self.rows.clone()));
        for column in &self.row_columns {
            out.column(column);
        }
        out.column(&Column::of(self.accumulators.clone()));
        out.column(&self.dimensions);
        out.column(&Column::of(self.outputs.clone()));
        for column in &self.output_columns {
            out.column(column);
        }
        out.signed(&self.propagators);
        out.column(&Column::of(self.masked.iter().map(|&m| u64::from(m)).collect()));
        out.column(&self.masks);
        out.bytes
    }

    pub fn decode(bytes: &[u8]) -> RusticolResult<Self> {
        let mut input = Reader::new(bytes);
        require(input.u32()? == STEPS_VERSION, "a step-column version")?;
        let count = input.count()?;
        let mut weights = Vec::with_capacity(count.min(1 << 20));
        for _ in 0..count {
            weights.push((input.text()?, input.text()?));
        }
        let flat = |column: Column| (0..column.len()).map(|k| column.get(k)).collect::<Vec<_>>();
        let rows = flat(input.column()?);
        let row_columns = [
            input.column()?,
            input.column()?,
            input.column()?,
            input.column()?,
            input.column()?,
            input.column()?,
            input.column()?,
            input.column()?,
            input.column()?,
        ];
        let accumulators = flat(input.column()?);
        let dimensions = input.column()?;
        let outputs = flat(input.column()?);
        let output_columns = [
            input.column()?,
            input.column()?,
            input.column()?,
            input.column()?,
        ];
        let propagators = input.signed()?;
        let masked = flat(input.column()?).into_iter().map(|m| m != 0).collect();
        let masks = input.column()?;
        input.done()?;
        let out = StepColumns {
            weights,
            rows,
            row_columns,
            accumulators,
            dimensions,
            outputs,
            output_columns,
            propagators,
            masked,
            masks,
        };
        out.validate()?;
        Ok(out)
    }

    /// The exact weights, parsed once.
    pub fn exact_weights(&self) -> RusticolResult<Vec<[Rational; 2]>> {
        self.weights
            .iter()
            .map(|(re, im)| Ok([exact(re)?, exact(im)?]))
            .collect()
    }

    /// The steps as the oracle's JSON (`json.dumps(steps, separators=(",",
    /// ":"))` of `_interaction_program`'s lists).
    pub fn json(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::from("[");
        for (s, range) in self.ranges().iter().enumerate() {
            if s > 0 {
                out.push(',');
            }
            out.push_str("{\"interactions\":[");
            for (n, k) in (range.rows.0..range.rows.1).enumerate() {
                if n > 0 {
                    out.push(',');
                }
                out.push('[');
                for c in 0..8 {
                    let _ = write!(out, "{},", self.row_columns[c].get(k));
                }
                let (re, im) = &self.weights[self.row_columns[8].at(k)];
                let _ = write!(out, "\"{re}\",\"{im}\"]");
            }
            out.push_str("],\"accumulators\":[");
            for (n, k) in (range.accumulators.0..range.accumulators.1).enumerate() {
                if n > 0 {
                    out.push(',');
                }
                let _ = write!(out, "{}", self.dimensions.get(k));
            }
            out.push_str("],\"outputs\":[");
            for (n, k) in (range.outputs.0..range.outputs.1).enumerate() {
                if n > 0 {
                    out.push(',');
                }
                let _ = write!(
                    out,
                    "[{},{},{},{},{}]",
                    self.output_columns[0].get(k),
                    self.propagators.get(k),
                    self.output_columns[1].get(k),
                    self.output_columns[2].get(k),
                    self.output_columns[3].get(k)
                );
            }
            out.push(']');
            if self.masked[s] {
                out.push_str(",\"accumulator_masks\":[");
                for (n, k) in (range.masks.0..range.masks.1).enumerate() {
                    if n > 0 {
                        out.push(',');
                    }
                    let _ = write!(out, "{}", self.masks.get(k));
                }
                out.push(']');
            }
            out.push('}');
        }
        out.push(']');
        out
    }

    /// `native_trees._prune_interactions` on the columns: the steps without
    /// the outputs no component marked in `live` (the program's readers'
    /// reads; marked further here) needs, directly or through the
    /// interactions of the outputs it needs, with the kernels renumbered to
    /// those still used; and the used vertex kernels and propagators (old
    /// indices, increasing). `None` (unchanged) where a state component is
    /// the output of two slots. `kernels` and `propagators` are the
    /// program's kernel counts.
    pub fn prune(
        &self,
        live: &mut [u8],
        kernels: usize,
        propagators: usize,
    ) -> RusticolResult<Option<(StepColumns, Vec<usize>, Vec<usize>)>> {
        let size = live.len();
        let ranges = self.ranges();
        let [o_acc, o_mom, o_start, o_dim] = &self.output_columns;
        let mut written = vec![false; size];
        for range in &ranges {
            for k in range.outputs.0..range.outputs.1 {
                let (start, dimension) = (o_start.at(k), o_dim.at(k));
                require(start + dimension <= size, "a tree program read lies outside its state")?;
                if written[start..start + dimension].iter().any(|&w| w) {
                    return Ok(None);
                }
                written[start..start + dimension].fill(true);
            }
        }
        drop(written);
        let r = &self.row_columns;
        // Backward: the outputs a later reader reads, then the rows summed
        // into their accumulators, whose inputs are read.
        let mut kept: Vec<(Vec<usize>, Vec<usize>)> = vec![(Vec::new(), Vec::new()); ranges.len()];
        for (s, range) in ranges.iter().enumerate().rev() {
            let outputs: Vec<usize> = (range.outputs.0..range.outputs.1)
                .filter(|&k| {
                    let (start, dimension) = (o_start.at(k), o_dim.at(k));
                    live[start..start + dimension].iter().any(|&x| x != 0)
                })
                .collect();
            let mut wanted = vec![false; range.accumulators.1 - range.accumulators.0];
            for &k in &outputs {
                wanted[o_acc.at(k)] = true;
            }
            let mut rows = Vec::new();
            for k in range.rows.0..range.rows.1 {
                let accumulator = r[7].at(k);
                require(accumulator < wanted.len(), "an interaction accumulator")?;
                if wanted[accumulator] {
                    rows.push(k);
                }
            }
            for &k in &rows {
                for (start, length) in [(r[1].at(k), r[2].at(k)), (r[3].at(k), r[4].at(k))] {
                    require(start + length <= size, "an interaction reads a current outside its state")?;
                    live[start..start + length].fill(1);
                }
            }
            kept[s] = (rows, outputs);
        }
        let used: BTreeSet<usize> = kept
            .iter()
            .flat_map(|(rows, _)| rows.iter().map(|&k| r[0].at(k)))
            .collect();
        let used_propagators: BTreeSet<usize> = kept
            .iter()
            .flat_map(|(_, outputs)| outputs.iter().map(|&k| self.propagators.get(k)))
            .filter(|&p| p >= 0)
            .map(|p| p as usize)
            .collect();
        require(
            used.iter().all(|&k| k < kernels)
                && used_propagators.iter().all(|&p| p < propagators),
            "an interaction kernel index",
        )?;
        let mut kernel_of = vec![u64::MAX; kernels];
        for (new, &old) in used.iter().enumerate() {
            kernel_of[old] = new as u64;
        }
        let mut propagator_of = vec![-1_i64; propagators];
        for (new, &old) in used_propagators.iter().enumerate() {
            propagator_of[old] = new as i64;
        }
        let mut out = StepColumns {
            weights: self.weights.clone(),
            ..Default::default()
        };
        let mut rows: [Vec<u64>; 9] = Default::default();
        let mut dimensions = Vec::new();
        let mut outs: [Vec<u64>; 4] = Default::default();
        let mut props = Vec::new();
        let mut masks = Vec::new();
        for (s, ((kept_rows, kept_outputs), range)) in kept.iter().zip(&ranges).enumerate() {
            let accumulators: BTreeSet<usize> =
                kept_outputs.iter().map(|&k| o_acc.at(k)).collect();
            let mut accumulator_of = vec![u64::MAX; range.accumulators.1 - range.accumulators.0];
            for (new, &old) in accumulators.iter().enumerate() {
                accumulator_of[old] = new as u64;
            }
            out.rows.push(kept_rows.len() as u64);
            for &k in kept_rows {
                rows[0].push(kernel_of[r[0].at(k)]);
                for c in 1..7 {
                    rows[c].push(r[c].get(k));
                }
                rows[7].push(accumulator_of[r[7].at(k)]);
                rows[8].push(r[8].get(k));
            }
            out.accumulators.push(accumulators.len() as u64);
            dimensions.extend(
                accumulators
                    .iter()
                    .map(|&a| self.dimensions.get(range.accumulators.0 + a)),
            );
            out.outputs.push(kept_outputs.len() as u64);
            for &k in kept_outputs {
                outs[0].push(accumulator_of[o_acc.at(k)]);
                let p = self.propagators.get(k);
                props.push(if p >= 0 { propagator_of[p as usize] } else { p });
                outs[1].push(o_mom.get(k));
                outs[2].push(o_start.get(k));
                outs[3].push(o_dim.get(k));
            }
            let masked = self.masked[s];
            out.masked.push(masked);
            if masked {
                masks.extend(accumulators.iter().map(|&a| self.masks.get(range.masks.0 + a)));
            }
        }
        out.row_columns = columns(rows);
        out.dimensions = Column::of(dimensions);
        out.output_columns = columns(outs);
        out.propagators = SignedColumn::of(props);
        out.masks = Column::of(masks);
        Ok(Some((
            out,
            used.into_iter().collect(),
            used_propagators.into_iter().collect(),
        )))
    }
}

// -------------------------------------------------------------------- cells

/// The exact overlaps `<b|t>` of the Born basis with every ket tensor of a
/// counterterm record (`native_trees._counterterm_program`'s `cells`) as a
/// table: the distinct Laurent cells, each its nonzero (N_c power, re, im)
/// in increasing power, and per ket tensor and basis element the index of
/// its cell (t-major; an empty cell is a zero overlap).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CellTable {
    pub basis: usize,
    pub tensors: usize,
    pub values: Vec<Vec<(i32, String, String)>>,
    pub ids: Column,
}

impl CellTable {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Writer::default();
        out.u32(CELLS_VERSION);
        out.u64(self.basis as u64);
        out.u64(self.tensors as u64);
        out.u64(self.values.len() as u64);
        for cell in &self.values {
            out.u32(u32::try_from(cell.len()).expect("cell powers"));
            for (power, re, im) in cell {
                out.i32(*power);
                out.text(re);
                out.text(im);
            }
        }
        out.column(&self.ids);
        out.bytes
    }

    pub fn decode(bytes: &[u8]) -> RusticolResult<Self> {
        let mut input = Reader::new(bytes);
        require(input.u32()? == CELLS_VERSION, "a cell-table version")?;
        let basis = input.count()?;
        let tensors = input.count()?;
        let count = input.count()?;
        let mut values = Vec::with_capacity(count.min(1 << 20));
        for _ in 0..count {
            let n = input.u32()? as usize;
            let mut cell = Vec::with_capacity(n);
            for _ in 0..n {
                cell.push((input.i32()?, input.text()?, input.text()?));
            }
            values.push(cell);
        }
        let ids = input.column()?;
        input.done()?;
        require(
            tensors >= basis
                && ids.len() == basis * tensors
                && (0..ids.len()).all(|k| ids.at(k) < values.len()),
            "a cell table's ids",
        )?;
        Ok(Self {
            basis,
            tensors,
            values,
            ids,
        })
    }

    /// The table as interference metric cells.
    pub fn metric(&self) -> RusticolResult<MetricCells> {
        let values = self
            .values
            .iter()
            .map(|cell| {
                cell.iter()
                    .map(|(power, re, im)| Ok((*power, [exact(re)?, exact(im)?])))
                    .collect::<RusticolResult<Vec<_>>>()
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        require(
            u32::try_from(values.len()).is_ok(),
            "a cell table with more distinct cells than an index holds",
        )?;
        let ids = (0..self.tensors)
            .map(|t| {
                (0..self.basis)
                    .map(|b| self.ids.get(t * self.basis + b) as u32)
                    .collect()
            })
            .collect();
        Ok(MetricCells::Table { values, ids })
    }

    /// The oracle's cell list: (basis element, ket tensor, N_c power, re,
    /// im) per nonzero cell, b-major over the basis's own tensors, then
    /// b-major over the other ket tensors.
    pub fn list(&self) -> Vec<(usize, usize, i32, String, String)> {
        let mut out = Vec::new();
        for (from, to) in [(0, self.basis), (self.basis, self.tensors)] {
            for b in 0..self.basis {
                for t in from..to {
                    for (power, re, im) in &self.values[self.ids.at(t * self.basis + b)] {
                        if re != "0" || im != "0" {
                            out.push((b, t, *power, re.clone(), im.clone()));
                        }
                    }
                }
            }
        }
        out
    }

    /// `native_trees._live_ket` on the table: per ket component (tensor,
    /// vector, N_c power) of a record whether an interference can read it.
    /// `born` is (basis element, Born order, N_c power) per Born
    /// component, `entries` (kind, bra order, vector, coupling, grade,
    /// output, degree) per entry (the vector ones count), `depth` the
    /// projection's.
    #[allow(clippy::type_complexity)]
    pub fn live_ket(
        &self,
        born: &[(usize, usize, i32)],
        entries: &[(String, usize, usize, Option<usize>, i32, usize, Option<i32>)],
        ket: &[(usize, usize, i32)],
        depth: Option<i32>,
    ) -> RusticolResult<Vec<bool>> {
        let mut bra: BTreeMap<usize, BTreeSet<(usize, i32)>> = BTreeMap::new();
        for &(b, order, power) in born {
            require(b < self.basis, "a Born component outside the basis")?;
            bra.entry(order).or_default().insert((b, power));
        }
        let mut by_vector: BTreeMap<usize, BTreeSet<(usize, i32, Option<i32>)>> = BTreeMap::new();
        for (kind, order, vector, _coupling, grade, _output, degree) in entries {
            if kind == "vector" {
                by_vector
                    .entry(*vector)
                    .or_default()
                    .insert((*order, *grade, *degree));
            }
        }
        let kept = |exponent: i32, degree: Option<i32>| match depth {
            None => true,
            Some(depth) => degree.is_some_and(|d| d - depth <= exponent && exponent <= d),
        };
        let mut exponents: BTreeMap<(usize, usize), BTreeSet<(i32, Option<i32>)>> = BTreeMap::new();
        let empty = BTreeSet::new();
        let mut out = Vec::with_capacity(ket.len());
        for &(t, vector, q) in ket {
            require(t < self.tensors, "a ket component outside the ket tensors")?;
            let found = exponents.entry((t, vector)).or_insert_with(|| {
                let mut found = BTreeSet::new();
                for &(order, grade, degree) in by_vector.get(&vector).unwrap_or(&empty) {
                    for &(b, p) in bra.get(&order).into_iter().flatten() {
                        for (n, re, im) in &self.values[self.ids.at(t * self.basis + b)] {
                            if re != "0" || im != "0" {
                                found.insert((n + p + grade, degree));
                            }
                        }
                    }
                }
                found
            });
            out.push(found.iter().any(|&(e, degree)| kept(e + q, degree)));
        }
        Ok(out)
    }
}
