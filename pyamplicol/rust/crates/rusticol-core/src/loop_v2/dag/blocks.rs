// SPDX-License-Identifier: 0BSD

//! Block grouping of a loop-request DAG (`loops/v2/blocks._blocks`) and the
//! Ward-kernel query (`ward_breaking_fields`' current scan).
//!
//! A block is the set of currents with one `BlockKey` (the full current
//! index without its helicity ancestry); its rows are their helicity
//! assignments. The colour-flow strings and sign, the colour-connection
//! parity, the N_c grade and the identity-pairing sign follow the oracle
//! rule by rule. The kernel variant only labels the blocks (every block of
//! the plain request model is a "plain" block), so Python attaches it.
//!
//! The tables come from `dag_input.block_input` (the model's colour
//! representations and fermion roles of the legs and particles), so blocks
//! can be grouped for a DAG the native builder made or one Python encoded.

use super::{ColumnarDag, DagError, DagResult, invalid};
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap};

#[derive(Deserialize)]
struct RawBlockInput {
    schema: String,
    /// `(label, model.color_rep(outgoing pdg))` of every leg with a pdg.
    label_reps: Vec<(u8, Option<i32>)>,
    /// `(pdg, model.color_rep, model.is_fermion)`; `None` where the model
    /// raises.
    particles: Vec<(i64, Option<i32>, Option<bool>)>,
    /// Labels of coloured fermion legs (`|rep| == 3` and a fermion).
    endpoints: Vec<u8>,
    u1_subtraction_kind: String,
}

pub const BLOCK_INPUT_SCHEMA: &str = "pyamplicol.loop-v2.dag-block-input/1";

pub struct BlockInput {
    label_reps: HashMap<u8, Option<i32>>,
    particles: HashMap<i64, (Option<i32>, Option<bool>)>,
    endpoints: HashMap<u8, i32>,
    u1_subtraction_kind: String,
}

impl BlockInput {
    pub fn from_json(text: &str) -> DagResult<Self> {
        let raw: RawBlockInput = serde_json::from_str(text)
            .map_err(|error| DagError::Malformed(format!("block tables: {error}")))?;
        if raw.schema != BLOCK_INPUT_SCHEMA {
            return Err(DagError::Malformed(format!(
                "block schema {:?}",
                raw.schema
            )));
        }
        let label_reps: HashMap<u8, Option<i32>> = raw.label_reps.into_iter().collect();
        let endpoints = raw
            .endpoints
            .into_iter()
            .map(|label| {
                label_reps
                    .get(&label)
                    .copied()
                    .flatten()
                    .map(|rep| (label, rep))
                    .ok_or_else(|| DagError::Malformed(format!("endpoint {label}")))
            })
            .collect::<DagResult<_>>()?;
        Ok(Self {
            label_reps,
            particles: raw
                .particles
                .into_iter()
                .map(|(pdg, rep, fermion)| (pdg, (rep, fermion)))
                .collect(),
            endpoints,
            u1_subtraction_kind: raw.u1_subtraction_kind,
        })
    }

    fn rep(&self, pdg: i64) -> DagResult<i32> {
        self.particles
            .get(&pdg)
            .and_then(|(rep, _)| *rep)
            .ok_or_else(|| DagError::Invalid(format!("particle not in model: {pdg}")))
    }

    fn fermion(&self, pdg: i64) -> DagResult<bool> {
        self.particles
            .get(&pdg)
            .and_then(|(_, fermion)| *fermion)
            .ok_or_else(|| DagError::Invalid(format!("particle not in model: {pdg}")))
    }

    fn label_rep(&self, label: u8) -> DagResult<i32> {
        self.label_reps
            .get(&label)
            .copied()
            .flatten()
            .ok_or_else(|| DagError::Invalid(format!("leg {label} has no colour representation")))
    }
}

/// `BlockColourFlow` without its N_c grade.
#[derive(Clone, Debug, PartialEq)]
pub struct ColourFlow {
    pub strings: Vec<Vec<u8>>,
    pub incoming_open: bool,
    pub outgoing_open: bool,
    pub singlet_labels: Vec<u8>,
    pub sign: i32,
    pub traces: Vec<Vec<u8>>,
}

/// One block, keyed by the first current of its key (`key_current`).
#[derive(Clone, Debug)]
pub struct RawBlock {
    pub key_current: u32,
    pub colour_flow: ColourFlow,
    pub nc_grade: Option<i64>,
    /// `(helicities, current id)`, sorted by helicities.
    pub rows: Vec<(Vec<(u8, i32)>, u32)>,
    pub connection_sign: i32,
    pub pairing_sign: i32,
}

fn python_tuple(labels: &[u8]) -> String {
    match labels {
        [single] => format!("({single},)"),
        _ => format!(
            "({})",
            labels
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// `_parity` with `None` (the open end, ranked last).
fn parity(sequence: &[Option<u8>]) -> i32 {
    let rank = |item: &Option<u8>| (item.is_none(), item.unwrap_or(0));
    let mut inversions = 0usize;
    for (i, left) in sequence.iter().enumerate() {
        for right in &sequence[i + 1..] {
            if rank(left) > rank(right) {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 1 { -1 } else { 1 }
}

/// `_pairing_sign`.
fn pairing_sign(strings: &[Vec<u8>], reps: &BlockInput) -> DagResult<i32> {
    // Insertion-ordered map start -> end.
    let mut end_of: Vec<(Option<u8>, Option<u8>)> = Vec::new();
    for string in strings {
        let first = *string.first().expect("strings are not empty");
        let last = *string.last().expect("strings are not empty");
        let start = (reps.label_rep(first)? == 3).then_some(first);
        let end = (reps.label_rep(last)? == -3).then_some(last);
        match end_of.iter_mut().find(|(key, _)| *key == start) {
            Some(entry) => entry.1 = end,
            None => end_of.push((start, end)),
        }
    }
    let has_open_start = end_of.iter().any(|(start, _)| start.is_none());
    let has_open_end = end_of.iter().any(|(_, end)| end.is_none());
    if has_open_start && has_open_end {
        let position = end_of
            .iter()
            .position(|(start, _)| start.is_none())
            .expect("open start");
        let (_, end) = end_of.remove(position);
        if end.is_some() {
            let open: Vec<usize> = end_of
                .iter()
                .enumerate()
                .filter(|(_, (_, other))| other.is_none())
                .map(|(index, _)| index)
                .collect();
            if open.len() != 1 {
                return invalid("not enough values to unpack in the colour-flow pairing");
            }
            end_of[open[0]].1 = end;
        }
    }
    let mut starts: Vec<Option<u8>> = end_of.iter().map(|(start, _)| *start).collect();
    starts.sort_by_key(|start| (start.is_none(), start.unwrap_or(0)));
    let ends: Vec<Option<u8>> = starts
        .iter()
        .map(|start| {
            end_of
                .iter()
                .find(|(key, _)| key == start)
                .expect("start key")
                .1
        })
        .collect();
    Ok(parity(&ends))
}

/// `block_colour_flow`.
fn block_colour_flow(
    word: &[u8],
    reps: &BlockInput,
    field_representation: i32,
    lines: &[Vec<u8>],
    traces: &[Vec<u8>],
) -> DagResult<ColourFlow> {
    let mut coloured = Vec::new();
    let mut singlets = Vec::new();
    for &label in word {
        match reps.label_rep(label)? {
            1 => singlets.push(label),
            3 | -3 | 8 => coloured.push(label),
            _ => return invalid(format!("leg {label} is not colour-flow lowerable")),
        }
    }
    let mut strings: Vec<Vec<u8>> = Vec::new();
    for &label in &coloured {
        if !strings.is_empty() && reps.label_rep(label)? == 3 {
            strings.push(vec![label]);
        } else if let Some(last) = strings.last_mut() {
            last.push(label);
        } else {
            strings.push(vec![label]);
        }
    }
    for pair in coloured.windows(2) {
        if (reps.label_rep(pair[0])? == -3) != (reps.label_rep(pair[1])? == 3) {
            return invalid(format!(
                "word {} is not a colour-flow block word",
                python_tuple(word)
            ));
        }
    }
    let incoming_open = match coloured.first() {
        Some(first) => reps.label_rep(*first)? != 3,
        None => false,
    };
    let outgoing_open = match coloured.last() {
        Some(last) => reps.label_rep(*last)? != -3,
        None => false,
    };
    let expected = match field_representation {
        1 => (false, false),
        3 => (false, true),
        -3 => (true, false),
        _ => (true, true),
    };
    if !coloured.is_empty() && (incoming_open, outgoing_open) != expected {
        return invalid(format!(
            "word {} does not match a field in representation {field_representation}",
            python_tuple(word)
        ));
    }
    strings.extend(lines.iter().cloned());
    let sign = pairing_sign(&strings, reps)?;
    Ok(ColourFlow {
        strings,
        incoming_open,
        outgoing_open,
        singlet_labels: singlets,
        sign,
        traces: traces.to_vec(),
    })
}

/// `identity_pairing_sign`.
fn identity_pairing_sign(
    fundamentals: &[Option<u8>],
    antifundamentals: &[Option<u8>],
) -> DagResult<i32> {
    let key = |item: &Option<u8>| (item.is_none(), item.unwrap_or(0));
    let mut starts = fundamentals.to_vec();
    let mut ends = antifundamentals.to_vec();
    starts.sort_by_key(key);
    ends.sort_by_key(key);
    if starts.len() != ends.len() {
        return invalid("unpaired fermion endpoints");
    }
    let sequence: Vec<Option<u8>> = starts
        .iter()
        .zip(&ends)
        .flat_map(|(start, end)| [*start, *end])
        .collect();
    Ok(parity(&sequence))
}

/// Canonical (by-value) ids of the tables, so that equal keys compare
/// equal whatever produced the columns.
fn canonical_ids<T: Eq + std::hash::Hash>(values: &[T]) -> Vec<u32> {
    let mut seen: HashMap<&T, u32> = HashMap::new();
    values
        .iter()
        .enumerate()
        .map(|(index, value)| *seen.entry(value).or_insert(index as u32))
        .collect()
}

impl ColumnarDag {
    fn sources_set(&self) -> Vec<bool> {
        let mut set = vec![false; self.current_count()];
        for source in &self.sources {
            set[*source as usize] = true;
        }
        set
    }

    /// Particle ids that a non-source, non-auxiliary current propagates and
    /// whose colour representation is not a singlet (`ward_breaking_fields`
    /// filters them by width and custom propagators).
    pub fn coloured_propagated(&self, tables: &BlockInput) -> Vec<i64> {
        let sources = self.sources_set();
        let mut out = BTreeSet::new();
        for (id, source) in sources.iter().enumerate() {
            let species = &self.tables.species[self.c_species[id] as usize];
            if *source || species.auxiliary.is_some() {
                continue;
            }
            if let Some((Some(rep), _)) = tables.particles.get(&species.pdg)
                && rep.abs() != 1
            {
                out.insert(species.pdg);
            }
        }
        out.into_iter().collect()
    }

    /// `_connection_parities`.
    fn connection_parities(&self, tables: &BlockInput) -> DagResult<Vec<Option<bool>>> {
        let n = self.current_count();
        let u1 = tables.u1_subtraction_kind.as_str();
        let mut info = Vec::with_capacity(n);
        for id in 0..n {
            let species = &self.tables.species[self.c_species[id] as usize];
            let rep = tables.rep(species.pdg)?;
            let u1_auxiliary = species.auxiliary.as_deref() == Some(u1);
            let has_endpoints = self
                .word(id)
                .iter()
                .any(|label| tables.endpoints.contains_key(label));
            info.push((
                rep == 8 || u1_auxiliary,
                rep == 1 || u1_auxiliary,
                has_endpoints,
                species.pdg,
            ));
        }
        let mut parity: Vec<Option<bool>> = (0..n)
            .map(|id| (self.c_source[id] == 1).then_some(false))
            .collect();
        let mut order: Vec<usize> = (0..self.interaction_count()).collect();
        order.sort_by_key(|id| self.c_mask[self.i_result[*id] as usize].count_ones());
        for id in order {
            let (left, right, result) = (
                self.i_left[id] as usize,
                self.i_right[id] as usize,
                self.i_result[id] as usize,
            );
            let (left_octet, left_singlet, left_ends, _) = info[left];
            let (right_octet, right_singlet, right_ends, _) = info[right];
            let result_pdg = info[result].3;
            let connects = if left_ends && right_ends && (left_octet || right_octet) {
                (tables.fermion(result_pdg)? && tables.rep(result_pdg)?.abs() == 3)
                    || !(left_singlet || right_singlet)
            } else {
                false
            };
            let (Some(left_parity), Some(right_parity)) = (parity[left], parity[right]) else {
                return invalid("unsupported operand type(s) for ^: 'NoneType'");
            };
            let value = left_parity ^ right_parity ^ connects;
            match parity[result] {
                None => parity[result] = Some(value),
                Some(known) if known != value => {
                    return invalid(format!(
                        "current {result} has derivations with different colour-connection signs"
                    ));
                }
                Some(_) => {}
            }
        }
        Ok(parity)
    }

    /// `u1_grades`.
    fn u1_grades(&self, tables: &BlockInput) -> Vec<Option<i64>> {
        let n = self.current_count();
        let u1 = tables.u1_subtraction_kind.as_str();
        let mut grade: Vec<Option<i64>> = (0..n)
            .map(|id| (self.c_source[id] == 1).then_some(0))
            .collect();
        let mut mixed = vec![false; n];
        let mut order: Vec<usize> = (0..self.interaction_count()).collect();
        order.sort_by_key(|id| self.c_mask[self.i_result[*id] as usize].count_ones());
        for id in order {
            let result = self.i_result[id] as usize;
            let (Some(left), Some(right)) = (
                grade[self.i_left[id] as usize],
                grade[self.i_right[id] as usize],
            ) else {
                mixed[result] = true;
                continue;
            };
            let auxiliary = self.tables.species[self.c_species[result] as usize]
                .auxiliary
                .as_deref();
            let value = left + right - (auxiliary == Some(u1)) as i64;
            if grade[result].is_none() && !mixed[result] {
                grade[result] = Some(value);
            } else if grade[result] != Some(value) {
                grade[result] = None;
                mixed[result] = true;
            }
        }
        grade
    }

    /// `_blocks`: every block, in the first-seen order of its key.
    pub fn blocks(&self, tables: &BlockInput) -> DagResult<Vec<RawBlock>> {
        // `_helicity_decoder`.
        let mut bits = Vec::with_capacity(self.sources.len());
        for source in &self.sources {
            let id = *source as usize;
            if self.c_leg[id] < 0 || self.c_helicity[id] == super::NO_HELICITY {
                return invalid("block source has no leg or helicity");
            }
            bits.push((
                self.c_ancestry[id],
                self.c_leg[id] as u8,
                self.c_helicity[id],
            ));
        }
        let decode = |ancestry: u64| -> Vec<(u8, i32)> {
            let mut helicities: Vec<(u8, i32)> = bits
                .iter()
                .filter(|(bit, _, _)| ancestry & bit != 0)
                .map(|(_, label, helicity)| (*label, *helicity))
                .collect();
            helicities.sort_unstable();
            helicities
        };
        let connections = self.connection_parities(tables)?;
        let grades = self.u1_grades(tables);
        let species = canonical_ids(&self.tables.species);
        let flavours = canonical_ids(&self.tables.flavours);
        let colours = canonical_ids(&self.tables.colours);
        let orders = canonical_ids(&self.tables.orders);
        type Key<'a> = (u32, u64, &'a [u8], u32, u32, i8, i32, u32);
        let mut index: HashMap<Key, usize> = HashMap::new();
        let mut members: Vec<(u32, Vec<u32>)> = Vec::new();
        for id in 0..self.current_count() {
            let key: Key = (
                species[self.c_species[id] as usize],
                self.c_mask[id],
                self.word(id),
                orders[self.c_orders[id] as usize],
                colours[self.c_colour[id] as usize],
                self.c_chirality[id],
                self.c_spin[id],
                flavours[self.c_flavour[id] as usize],
            );
            match index.get(&key) {
                Some(block) => members[*block].1.push(id as u32),
                None => {
                    index.insert(key, members.len());
                    members.push((id as u32, vec![id as u32]));
                }
            }
        }
        let mut out = Vec::with_capacity(members.len());
        for (first, ids) in members {
            let first_id = first as usize;
            let pdg = self.tables.species[self.c_species[first_id] as usize].pdg;
            let field_representation = tables.rep(pdg)?;
            let mut ends: Vec<(Option<u8>, i32)> = Vec::new();
            let mask = self.c_mask[first_id];
            for bit in 0..64u8 {
                if mask >> bit & 1 == 1
                    && let Some(rep) = tables.endpoints.get(&(bit + 1))
                {
                    ends.push((Some(bit + 1), *rep));
                }
            }
            if tables.fermion(pdg)? && field_representation.abs() == 3 {
                ends.push((None, -field_representation));
            }
            let parities: BTreeSet<Option<bool>> =
                ids.iter().map(|id| connections[*id as usize]).collect();
            if parities.len() != 1 {
                return invalid(format!(
                    "block {} mixes colour-connection signs",
                    python_tuple(self.word(first_id))
                ));
            }
            let nc_grades: BTreeSet<Option<i64>> =
                ids.iter().map(|id| grades[*id as usize]).collect();
            let colour = &self.tables.colours[self.c_colour[first_id] as usize];
            let colour_flow = block_colour_flow(
                self.word(first_id),
                tables,
                field_representation,
                &colour.lines,
                &colour.traces,
            )?;
            let mut rows: Vec<(Vec<(u8, i32)>, u32)> = ids
                .iter()
                .map(|id| (decode(self.c_ancestry[*id as usize]), *id))
                .collect();
            rows.sort_by(|left, right| left.0.cmp(&right.0));
            let fundamentals: Vec<Option<u8>> = ends
                .iter()
                .filter(|(_, rep)| *rep == 3)
                .map(|(label, _)| *label)
                .collect();
            let antifundamentals: Vec<Option<u8>> = ends
                .iter()
                .filter(|(_, rep)| *rep == -3)
                .map(|(label, _)| *label)
                .collect();
            out.push(RawBlock {
                key_current: first,
                colour_flow,
                nc_grade: if nc_grades.len() == 1 {
                    *nc_grades.iter().next().expect("one grade")
                } else {
                    None
                },
                rows,
                connection_sign: if parities.into_iter().next().flatten() == Some(true) {
                    -1
                } else {
                    1
                },
                pairing_sign: identity_pairing_sign(&fundamentals, &antifundamentals)?,
            });
        }
        Ok(out)
    }
}
