// SPDX-License-Identifier: 0BSD

//! Block grouping of a loop-request DAG (`loops/v2/blocks._blocks`) and the
//! current scans of generation's queries (`propagated_species`: the
//! Ward-kernel and coloured-width scans; `u1_leaking`).
//!
//! A block is the set of currents with one `BlockKey` (the full current
//! index without its helicity ancestry); its rows are their helicity
//! assignments. The colour-flow strings and sign, the colour-connection
//! parity, the N_c grade and the identity-pairing sign follow the oracle
//! rule by rule. The kernel variant only labels the blocks (every block of
//! a loop-request DAG is a "plain" block), so Python attaches it.
//!
//! For the CT program (`ct_closure.CountertermBlocks.build`) the tables
//! carry the clone lines of the lowered model: a block of a clone particle
//! is grouped as that particle and keyed by the field the line is read as
//! (`CountertermLoweredModel.line_fields`). [`ColumnarDag::u1_leaking`] is
//! the DAG scan of `ct_closure.u1_leaking_structures`.
//!
//! The tables come from `dag_input.block_input` (the model's colour
//! representations and fermion roles of the legs and particles), so blocks
//! can be grouped for a DAG the native builder made or one Python encoded.

use super::recursion::FxMap;
use super::{ColumnarDag, DagError, DagResult, FERMION_LINE_REPRESENTATION, invalid, refused};
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
    /// `(clone pdg, pdg of the field it is read as)` of the CT program's
    /// clone lines (`ct_closure._clone_lines`); empty otherwise.
    #[serde(default)]
    clones: Vec<(i64, i64)>,
}

pub const BLOCK_INPUT_SCHEMA: &str = "pyamplicol.loop-v2.dag-block-input/1";

pub struct BlockInput {
    label_reps: HashMap<u8, Option<i32>>,
    particles: HashMap<i64, (Option<i32>, Option<bool>)>,
    endpoints: HashMap<u8, i32>,
    u1_subtraction_kind: String,
    clones: HashMap<i64, i64>,
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
            clones: raw.clones.into_iter().collect(),
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

/// The grouping of currents into blocks (`BlockKey`): canonical species,
/// mask, word, orders, colour state, chirality, spin and flavour flow.
type BlockKey<'a> = (u32, u64, &'a [u8], u32, u32, i8, i32, u32);

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
    /// The field a clone block is read as (its key's particle), else `None`.
    pub clone_field: Option<i64>,
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
        return refused(
            FERMION_LINE_REPRESENTATION,
            format!(
                "unpaired fermion endpoints ({} fundamental, {} antifundamental): a \
                 coloured fermion line that does not end on an opposite coloured end",
                starts.len(),
                ends.len()
            ),
        );
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
pub(super) fn canonical_ids<T: Eq + std::hash::Hash>(values: &[T]) -> Vec<u32> {
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

    /// Distinct `(particle id, amputated)` of the non-source, non-auxiliary
    /// currents, sorted: the current scans of `blocks.ward_breaking_fields`
    /// (every pair) and `artifact.coloured_widths` (pairs not `amputated`),
    /// which test each particle's colour, width and propagator themselves.
    /// `amputated`: the current's momentum leaves exactly one leg of
    /// `full_mask` out (a Born root or an external self-energy reads it).
    pub fn propagated_species(&self, full_mask: u64) -> Vec<(i64, bool)> {
        let sources = self.sources_set();
        let mut out = BTreeSet::new();
        for (id, source) in sources.iter().enumerate() {
            let species = &self.tables.species[self.c_species[id] as usize];
            if *source || species.auxiliary.is_some() {
                continue;
            }
            let amputated = (full_mask & !self.c_momentum[id]).count_ones() == 1;
            out.insert((species.pdg, amputated));
        }
        out.into_iter().collect()
    }

    /// The vertex kinds `ct_closure.u1_leaking_structures` reports: kernels
    /// of `kinds` (CT colours coupling to the U(1) part of a gluon) that
    /// absorb an adjoint current holding a fundamental leg, or produce an
    /// adjoint current a `generator_kinds` kernel (a fundamental generator)
    /// reads. `adjoint` are the particles of IR colour 8, `fundamental` the
    /// mask of the legs of IR colour ±3; interactions into the leg set
    /// `rest` (every leg but leg 1) are never read by a root and skipped.
    pub fn u1_leaking(
        &self,
        kinds: &[i64],
        generator_kinds: &[i64],
        adjoint: &[i64],
        fundamental: u64,
        rest: u64,
    ) -> Vec<i64> {
        let kinds: BTreeSet<i64> = kinds.iter().copied().collect();
        let generators: BTreeSet<i64> = generator_kinds.iter().copied().collect();
        let adjoint: BTreeSet<i64> = adjoint.iter().copied().collect();
        let is_adjoint = |current: u32| {
            adjoint.contains(&self.tables.species[self.c_species[current as usize] as usize].pdg)
        };
        let read: Vec<usize> = (0..self.interaction_count())
            .filter(|id| self.c_mask[self.i_result[*id] as usize] != rest)
            .collect();
        let kind = |id: usize| self.tables.vertices[self.i_vertex[id] as usize].kind;
        let mut leaking = BTreeSet::new();
        let mut producers: HashMap<u32, BTreeSet<i64>> = HashMap::new();
        for &id in &read {
            let vertex_kind = kind(id);
            if !kinds.contains(&vertex_kind) {
                continue;
            }
            for input in [self.i_left[id], self.i_right[id]] {
                if is_adjoint(input) && self.c_mask[input as usize] & fundamental != 0 {
                    leaking.insert(vertex_kind);
                }
            }
            if is_adjoint(self.i_result[id]) {
                producers
                    .entry(self.i_result[id])
                    .or_default()
                    .insert(vertex_kind);
            }
        }
        for &id in &read {
            if generators.contains(&kind(id)) {
                for input in [self.i_left[id], self.i_right[id]] {
                    if let Some(found) = producers.get(&input) {
                        leaking.extend(found.iter().copied());
                    }
                }
            }
        }
        leaking.into_iter().collect()
    }

    /// `_connection_parities`: the colour-connection parity of every
    /// current ([`super::signs::connection_parities`] over the dense block
    /// ids `block_of`; a block derived with both parities had the other
    /// parity's `-1` folded into its colour weights when it was built,
    /// `signs::apply_connection_signs`). `order`: the merges by result size
    /// (`signs::size_order`).
    fn connection_parities(
        &self,
        tables: &BlockInput,
        block_of: &[u32],
        block_count: usize,
        order: Option<&[usize]>,
    ) -> DagResult<Vec<Option<bool>>> {
        let n = self.current_count();
        let u1 = tables.u1_subtraction_kind.as_str();
        let mut endpoints = 0u64;
        for label in tables.endpoints.keys() {
            endpoints |= 1u64 << (label - 1);
        }
        // `(octet, singlet)` of every species, raised per current below so a
        // species of no current raises nothing.
        let roles: Vec<DagResult<(bool, bool)>> = self
            .tables
            .species
            .iter()
            .map(|species| {
                let u1_auxiliary = species.auxiliary.as_deref() == Some(u1);
                tables
                    .rep(species.pdg)
                    .map(|rep| (rep == 8 || u1_auxiliary, rep == 1 || u1_auxiliary))
            })
            .collect();
        let mut info = Vec::with_capacity(n);
        for id in 0..n {
            let (octet, singlet) = match &roles[self.c_species[id] as usize] {
                Ok(roles) => *roles,
                Err(error) => return invalid(error.to_string()),
            };
            let has_endpoints = self
                .word(id)
                .iter()
                .any(|label| endpoints >> (label - 1) & 1 == 1);
            info.push((octet, singlet, has_endpoints));
        }
        let mut connects = Vec::with_capacity(self.interaction_count());
        for id in 0..self.interaction_count() {
            let (left_octet, left_singlet, left_ends) = info[self.i_left[id] as usize];
            let (right_octet, right_singlet, right_ends) = info[self.i_right[id] as usize];
            let result_pdg =
                self.tables.species[self.c_species[self.i_result[id] as usize] as usize].pdg;
            connects.push(if left_ends && right_ends && (left_octet || right_octet) {
                (tables.fermion(result_pdg)? && tables.rep(result_pdg)?.abs() == 3)
                    || !(left_singlet || right_singlet)
            } else {
                false
            });
        }
        let (parity, _) = super::signs::connection_parities(
            |id| self.c_source[id] == 1,
            block_of,
            block_count,
            self.interaction_count(),
            |id| {
                (
                    self.i_left[id] as usize,
                    self.i_right[id] as usize,
                    self.i_result[id] as usize,
                    connects[id],
                )
            },
            order,
        )?;
        Ok(parity)
    }

    /// `u1_grades`; `order`: the merges by result size
    /// (`signs::size_order`).
    fn u1_grades(&self, tables: &BlockInput, order: Option<&[usize]>) -> Vec<Option<i64>> {
        let u1 = tables.u1_subtraction_kind.as_str();
        let u1_auxiliary: Vec<bool> = self
            .tables
            .species
            .iter()
            .map(|species| species.auxiliary.as_deref() == Some(u1))
            .collect();
        let sources = (0..self.current_count())
            .map(|id| (self.c_source[id] == 1).then_some(0))
            .collect();
        let merge = |id: usize| {
            let result = self.i_result[id] as usize;
            (
                self.i_left[id] as usize,
                self.i_right[id] as usize,
                result,
                u1_auxiliary[self.c_species[result] as usize] as i64,
            )
        };
        match order {
            None => fold_u1_grades(sources, (0..self.interaction_count()).map(merge)),
            Some(order) => fold_u1_grades(sources, order.iter().map(|id| merge(*id))),
        }
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
        let species = canonical_ids(&self.tables.species);
        let flavours = canonical_ids(&self.tables.flavours);
        let colours = canonical_ids(&self.tables.colours);
        let orders = canonical_ids(&self.tables.orders);
        let key = |id: usize| -> BlockKey<'_> {
            (
                species[self.c_species[id] as usize],
                self.c_mask[id],
                self.word(id),
                orders[self.c_orders[id] as usize],
                colours[self.c_colour[id] as usize],
                self.c_chirality[id],
                self.c_spin[id],
                flavours[self.c_flavour[id] as usize],
            )
        };
        let mut index: FxMap<BlockKey, usize> = FxMap::default();
        let mut members: Vec<(u32, Vec<u32>)> = Vec::new();
        let mut block_of: Vec<u32> = Vec::with_capacity(self.current_count());
        for id in 0..self.current_count() {
            let key = key(id);
            match index.get(&key) {
                Some(block) => {
                    members[*block].1.push(id as u32);
                    block_of.push(*block as u32);
                }
                None => {
                    block_of.push(members.len() as u32);
                    index.insert(key, members.len());
                    members.push((id as u32, vec![id as u32]));
                }
            }
        }
        let order = super::signs::size_order(self.interaction_count(), |id| {
            self.c_mask[self.i_result[id] as usize].count_ones()
        });
        let connections =
            self.connection_parities(tables, &block_of, members.len(), order.as_deref())?;
        let grades = self.u1_grades(tables, order.as_deref());
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
                clone_field: tables.clones.get(&pdg).copied(),
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

/// The N_c grade of every current (`blocks.u1_grades`, the single-grade
/// view of `generation.dag_grades.u1_grade_sets`): `grade` holds the
/// sources' (0), `merges` are the derivations (left, right, result, 1 if the
/// result is the U(1)-subtraction auxiliary) with every input's derivations
/// before it is read. A current has a grade only if every derivation gives
/// it the same one from inputs of one grade each, whatever the order of its
/// derivations; otherwise, or with none, it has none.
fn fold_u1_grades(
    mut grade: Vec<Option<i64>>,
    merges: impl Iterator<Item = (usize, usize, usize, i64)>,
) -> Vec<Option<i64>> {
    let mut mixed = vec![false; grade.len()];
    for (left, right, result, shift) in merges {
        let value = match (grade[left], grade[right]) {
            (Some(left), Some(right)) => Some(left + right - shift),
            _ => None,
        };
        if value.is_some() && grade[result].is_none() && !mixed[result] {
            grade[result] = value;
        } else if value.is_none() || grade[result] != value {
            grade[result] = None;
            mixed[result] = true;
        }
    }
    grade
}

#[cfg(test)]
mod tests {
    use super::{DagError, FERMION_LINE_REPRESENTATION, fold_u1_grades, identity_pairing_sign};

    #[test]
    fn a_derivation_from_a_mixed_input_leaves_no_grade_in_either_order() {
        // 0, 1: sources; 2: the U(1)-subtraction auxiliary of (0, 1); 3:
        // derived with grades 0 and -1 (mixed); 4 and 5: grade -1 from
        // (2, 1) and a derivation reading 3, in both orders; 6: grade -1
        // twice; 7: underived.
        let sources = vec![Some(0), Some(0), None, None, None, None, None, None];
        let merges = [
            (0, 1, 2, 1),
            (0, 1, 3, 0),
            (2, 0, 3, 0),
            (2, 1, 4, 0),
            (3, 0, 4, 0),
            (3, 0, 5, 0),
            (2, 1, 5, 0),
            (2, 1, 6, 0),
            (1, 2, 6, 0),
        ];
        assert_eq!(
            fold_u1_grades(sources, merges.into_iter()),
            vec![Some(0), Some(0), Some(-1), None, None, None, Some(-1), None]
        );
    }

    #[test]
    fn unpaired_fermion_ends_are_the_listed_refusal() {
        // A quark line ending on a colour-singlet fermion: one fundamental
        // end, no antifundamental one.
        match identity_pairing_sign(&[Some(1)], &[]) {
            Err(DagError::Unsupported(message)) => assert!(
                message.starts_with(&format!("{FERMION_LINE_REPRESENTATION}: unpaired")),
                "{message}"
            ),
            other => panic!("expected the listed refusal, got {other:?}"),
        }
        assert_eq!(identity_pairing_sign(&[Some(1)], &[Some(2)]).unwrap(), 1);
    }
}
