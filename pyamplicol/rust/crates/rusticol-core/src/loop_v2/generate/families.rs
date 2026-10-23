// SPDX-License-Identifier: 0BSD

//! Denominator families from the loop-prefix recursion: the native port of
//! `loops/v2/families.py` (`_Recursion`, design §3.3, §4.2 (1)).
//!
//! The port reproduces the Python generator exactly, including its order of
//! discovery: cut species are processed in sorted order from a LIFO stack,
//! sub-masks are enumerated as `sub = (sub - 1) & remaining`, admitted family
//! keys are emitted sorted and prefix states are interned in Python's order.
//! Every name is an id of the input's sorted tables, so comparing ids
//! compares the Python objects (strings by code point, tuples
//! lexicographically with shorter prefixes first). No diagram is listed:
//! positions consume block currents (sums of all trees on their leg sets) and
//! only prefixes that can still close are visited (memoised closability).

use super::orders::{OrderSpace, OrderVec};
use serde::Deserialize;
use std::collections::HashMap;

pub type LegSet = u64;

/// One external leg (incoming field; outgoing legs are crossed).
#[derive(Debug, Clone, Deserialize)]
pub struct LegInput {
    pub field: u32,
    pub mass: u32,
}

/// A loop species (a propagating field).
#[derive(Debug, Clone, Deserialize)]
pub struct SpeciesInput {
    pub antiname: u32,
    pub colour: i32,
    pub mass: u32,
    /// Ids of the coupling-order names the species is perturbative in.
    pub perturbative: Vec<u16>,
}

/// A loop-vertex template; vertex ids are ranks of Python's `LoopVertex`
/// order, so comparing ids compares the templates.
#[derive(Debug, Clone, Deserialize)]
pub struct VertexInput {
    pub loop_in: u32,
    pub attached: Vec<u32>,
    pub loop_out: u32,
    pub orders: OrderVec,
    pub q_degree: i64,
    pub reflected: u32,
}

/// A field-level block current; ids are ranks of Python's `BlockCurrent`
/// order (mask, field, orders).
#[derive(Debug, Clone, Deserialize)]
pub struct CurrentInput {
    pub mask: LegSet,
    pub field: u32,
    pub orders: OrderVec,
}

/// Everything the family recursion reads: Θ(model) tables plus the
/// field-level block table (`BlockTable.currents` of every leg set).
#[derive(Debug, Clone, Deserialize)]
pub struct FamilyInput {
    pub legs: Vec<LegInput>,
    /// Indexed by field id; `None` for fields that are not loop species.
    pub species: Vec<Option<SpeciesInput>>,
    /// Per mass id (ranks of Python's `MassId` order): structurally zero.
    pub massless: Vec<bool>,
    pub vertices: Vec<VertexInput>,
    /// Per field id: `catalogue.vertices(field)` as vertex ids, in order.
    pub vertices_by_loop_in: Vec<Vec<u32>>,
    /// Every block current, in Python order.
    pub currents: Vec<CurrentInput>,
    /// Eligible cut species (representatives of merged classes), sorted.
    pub eligible: Vec<u32>,
    /// Per field id: closed-loop flavour multiplicity of a representative.
    pub multiplicity: Vec<u32>,
    pub corrections: Vec<u16>,
    pub loop_orders: Vec<OrderVec>,
    pub cap: OrderVec,
}

/// Reasons of the selection rule (`selection_rejection`), in Python spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Rejection {
    ExternalWavefunctionCorrection,
    VevTadpole,
    SingleColouredAttachment,
}

impl Rejection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExternalWavefunctionCorrection => "external-wavefunction-correction",
            Self::VevTadpole => "vev-tadpole",
            Self::SingleColouredAttachment => "single-coloured-attachment",
        }
    }
}

/// A family key: the canonical cyclic sequence of (leg mask, mass id).
pub type FamilyKey = Vec<(LegSet, u32)>;

/// An interned position: a vertex with the block currents it attaches.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Position {
    pub vertex: u32,
    /// Block-current ids sorted by mask (Python's `sorted` of disjoint masks).
    pub blocks: Vec<u32>,
}

#[derive(Debug, Clone, Copy)]
pub struct Closure {
    pub state: u32,
    pub symmetric: bool,
    pub multiplicity: u32,
}

#[derive(Debug, Clone)]
pub struct Family {
    pub key: FamilyKey,
    pub rank: i64,
    pub closures: Vec<Closure>,
}

/// The output of the recursion (`FamilySet` without the Python objects).
#[derive(Debug, Default)]
pub struct FamilyOutput {
    /// Interned positions referenced by `state_position`.
    pub positions: Vec<Position>,
    /// Prefix graph: parent state (-1 for roots), the position of the last
    /// step (`u32::MAX` for roots) and the species leaving it (the cut species
    /// for roots).
    pub state_parent: Vec<i64>,
    pub state_position: Vec<u32>,
    pub state_species: Vec<u32>,
    pub families: Vec<Family>,
    pub scaleless: Vec<FamilyKey>,
    pub rejected: Vec<(FamilyKey, Rejection)>,
}

#[derive(Debug)]
struct PositionData {
    mask: LegSet,
    orders: u32,
    loop_out: u32,
    q_degree: i64,
}

fn lowest_leg(mask: LegSet) -> u32 {
    mask.trailing_zeros()
}

struct Recursion<'a> {
    input: &'a FamilyInput,
    full: LegSet,
    orders: OrderSpace,
    cap: OrderVec,
    loop_orders: Vec<u32>,
    corrections: u64,
    eligible: Vec<bool>,
    current_orders: Vec<u32>,
    currents_by_mask: HashMap<LegSet, (usize, usize)>,
    vertex_orders: Vec<u32>,
    positions: Vec<Position>,
    position_data: Vec<PositionData>,
    position_index: HashMap<Position, u32>,
    attachments: HashMap<(LegSet, Vec<u32>), Vec<Vec<u32>>>,
    by_mask_species: HashMap<(LegSet, u32), Vec<u32>>,
    closable: HashMap<(LegSet, u32, u32, u32), bool>,
}

impl<'a> Recursion<'a> {
    fn new(input: &'a FamilyInput) -> Result<Self, String> {
        let legs = input.legs.len();
        if legs == 0 || legs > 64 {
            return Err(format!(
                "the native family recursion takes 1..=64 legs, not {legs}"
            ));
        }
        let names = input
            .corrections
            .iter()
            .chain(input.cap.iter().map(|(name, _)| name))
            .copied()
            .max()
            .unwrap_or(0);
        if names >= 64 {
            return Err("more than 64 coupling-order names".into());
        }
        let full = if legs == 64 {
            u64::MAX
        } else {
            (1u64 << legs) - 1
        };
        let mut orders = OrderSpace::new();
        let current_orders = input
            .currents
            .iter()
            .map(|current| orders.intern(current.orders.clone()))
            .collect();
        let vertex_orders = input
            .vertices
            .iter()
            .map(|vertex| orders.intern(vertex.orders.clone()))
            .collect();
        let loop_orders = input
            .loop_orders
            .iter()
            .map(|order| orders.intern(order.clone()))
            .collect();
        let mut currents_by_mask: HashMap<LegSet, (usize, usize)> = HashMap::new();
        for (index, current) in input.currents.iter().enumerate() {
            let entry = currents_by_mask
                .entry(current.mask)
                .or_insert((index, index));
            if entry.1 != index {
                return Err("block currents must be grouped by leg set".into());
            }
            entry.1 = index + 1;
        }
        let mut eligible = vec![false; input.species.len()];
        for &name in &input.eligible {
            eligible[name as usize] = true;
        }
        let mut cap = input.cap.clone();
        cap.retain(|&(_, power)| power != 0);
        cap.sort_unstable();
        Ok(Self {
            input,
            full,
            orders,
            cap,
            loop_orders,
            corrections: input
                .corrections
                .iter()
                .fold(0, |bits, &name| bits | (1u64 << name)),
            eligible,
            current_orders,
            currents_by_mask,
            vertex_orders,
            positions: Vec::new(),
            position_data: Vec::new(),
            position_index: HashMap::new(),
            attachments: HashMap::new(),
            by_mask_species: HashMap::new(),
            closable: HashMap::new(),
        })
    }

    fn currents(&self, mask: LegSet) -> std::ops::Range<usize> {
        self.currents_by_mask
            .get(&mask)
            .map_or(0..0, |&(start, end)| start..end)
    }

    fn species(&self, field: u32) -> Result<&SpeciesInput, String> {
        self.input
            .species
            .get(field as usize)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("field id {field} is not a loop species"))
    }

    /// `_attachments(mask, fields)`: every assignment of disjoint sub-masks
    /// covering `mask` to the attached fields, in Python's order.
    fn attachments(&mut self, mask: LegSet, fields: &[u32]) -> Vec<Vec<u32>> {
        let key = (mask, fields.to_vec());
        if let Some(found) = self.attachments.get(&key) {
            return found.clone();
        }
        let mut out: Vec<Vec<u32>> = Vec::new();
        if fields.len() == 1 {
            for index in self.currents(mask) {
                if self.input.currents[index].field == fields[0] {
                    out.push(vec![index as u32]);
                }
            }
        } else {
            let low = mask & mask.wrapping_neg();
            let rest = mask ^ low;
            let mut unique: Vec<u32> = Vec::new();
            for &name in fields {
                if !unique.contains(&name) {
                    unique.push(name);
                }
            }
            let mut sub = rest;
            loop {
                let part = sub | low;
                if part != mask {
                    for &name in &unique {
                        let mut others = fields.to_vec();
                        let at = others.iter().position(|&f| f == name).expect("present");
                        others.remove(at);
                        let tails = self.attachments(mask ^ part, &others);
                        if tails.is_empty() {
                            continue;
                        }
                        for index in self.currents(part) {
                            if self.input.currents[index].field == name {
                                for tail in &tails {
                                    let mut item = Vec::with_capacity(tail.len() + 1);
                                    item.push(index as u32);
                                    item.extend_from_slice(tail);
                                    out.push(item);
                                }
                            }
                        }
                    }
                }
                if sub == 0 {
                    break;
                }
                sub = (sub - 1) & rest;
            }
            let currents = &self.input.currents;
            for item in &mut out {
                item.sort_unstable_by_key(|&index| currents[index as usize].mask);
            }
        }
        self.attachments.insert(key, out.clone());
        out
    }

    fn intern_position(&mut self, vertex: u32, blocks: Vec<u32>) -> u32 {
        let position = Position { vertex, blocks };
        if let Some(&id) = self.position_index.get(&position) {
            return id;
        }
        let mut orders = self.vertex_orders[vertex as usize];
        let mut mask = 0;
        for &block in &position.blocks {
            orders = self.orders.add(orders, self.current_orders[block as usize]);
            mask |= self.input.currents[block as usize].mask;
        }
        let template = &self.input.vertices[vertex as usize];
        let id = self.positions.len() as u32;
        self.position_data.push(PositionData {
            mask,
            orders,
            loop_out: template.loop_out,
            q_degree: template.q_degree,
        });
        self.position_index.insert(position.clone(), id);
        self.positions.push(position);
        id
    }

    /// `_positions(mask, species)`.
    fn positions(&mut self, mask: LegSet, species: u32) -> Vec<u32> {
        if let Some(found) = self.by_mask_species.get(&(mask, species)) {
            return found.clone();
        }
        let size = mask.count_ones() as usize;
        let mut out = Vec::new();
        let vertices = self
            .input
            .vertices_by_loop_in
            .get(species as usize)
            .cloned()
            .unwrap_or_default();
        for vertex in vertices {
            let template = &self.input.vertices[vertex as usize];
            if !self.eligible[template.loop_out as usize] || template.attached.len() > size {
                continue;
            }
            let attached = template.attached.clone();
            for blocks in self.attachments(mask, &attached) {
                let id = self.intern_position(vertex, blocks);
                if self
                    .orders
                    .within(self.position_data[id as usize].orders, &self.cap)
                {
                    out.push(id);
                }
            }
        }
        self.by_mask_species.insert((mask, species), out.clone());
        out
    }

    /// `_closable(remaining, species, cut, orders)`.
    fn closable(&mut self, remaining: LegSet, species: u32, cut: u32, orders: u32) -> bool {
        let key = (remaining, species, cut, orders);
        if let Some(&found) = self.closable.get(&key) {
            return found;
        }
        let mut result = false;
        let mut sub = remaining;
        'outer: while sub != 0 {
            for position in self.positions(sub, species) {
                let data = &self.position_data[position as usize];
                let (loop_out, position_orders) = (data.loop_out, data.orders);
                let total = self.orders.add(orders, position_orders);
                if !self.orders.within(total, &self.cap) {
                    continue;
                }
                if sub == remaining {
                    if loop_out == cut && self.loop_orders.contains(&total) {
                        result = true;
                        break 'outer;
                    }
                } else if self.closable(remaining ^ sub, loop_out, cut, total) {
                    result = true;
                    break 'outer;
                }
            }
            sub = (sub - 1) & remaining;
        }
        self.closable.insert(key, result);
        result
    }

    fn mass_of_species(&self, field: u32) -> Result<u32, String> {
        Ok(self.species(field)?.mass)
    }

    /// The N <= 2 routing read in the opposite direction from position 1, as
    /// (cut species, positions) with positions that are not interned.
    fn reflected(&self, positions: &[u32]) -> (u32, Vec<Position>) {
        let mut order = vec![positions[0]];
        order.extend(positions[1..].iter().rev());
        let mirrored: Vec<Position> = order
            .iter()
            .map(|&id| {
                let position = &self.positions[id as usize];
                Position {
                    vertex: self.input.vertices[position.vertex as usize].reflected,
                    blocks: position.blocks.clone(),
                }
            })
            .collect();
        let cut = self.input.vertices[mirrored.last().expect("non-empty").vertex as usize].loop_out;
        (cut, mirrored)
    }

    fn selection_rejection(&self, positions: &[u32]) -> Result<Option<Rejection>, String> {
        let blocks: Vec<u32> = positions
            .iter()
            .flat_map(|&id| self.positions[id as usize].blocks.iter().copied())
            .collect();
        let single = |block: &u32| {
            let mask = self.input.currents[*block as usize].mask;
            mask & (mask - 1) == 0
        };
        if positions.len() <= 2 && blocks.len() == 2 && blocks.iter().any(single) {
            return Ok(Some(Rejection::ExternalWavefunctionCorrection));
        }
        if positions.len() == 1 && blocks.len() <= 1 {
            return Ok(Some(Rejection::VevTadpole));
        }
        let mut coloured = 0;
        for &block in &blocks {
            if self
                .species(self.input.currents[block as usize].field)?
                .colour
                .abs()
                != 1
            {
                coloured += 1;
            }
        }
        Ok((coloured == 1).then_some(Rejection::SingleColouredAttachment))
    }
}

/// `canonical_family_key`: the lexicographic minimum over the 2N readings.
pub fn canonical_family_key(masks: &[LegSet], masses: &[u32]) -> FamilyKey {
    let count = masks.len();
    let mut best: Option<FamilyKey> = None;
    for reflected in [false, true] {
        for start in 0..count {
            let reading: FamilyKey = (0..count)
                .map(|step| {
                    if reflected {
                        let i = (start + count - step % count) % count;
                        (masks[i], masses[(i + count - 1) % count])
                    } else {
                        let i = (start + step) % count;
                        (masks[i], masses[i])
                    }
                })
                .collect();
            if best.as_ref().is_none_or(|current| reading < *current) {
                best = Some(reading);
            }
        }
    }
    best.expect("a family has at least one position")
}

/// `Family.invariant_masks`: arcs of the ring, represented without leg 0.
pub fn invariant_masks(key: &FamilyKey) -> Vec<LegSet> {
    let full = key.iter().fold(0, |acc, &(mask, _)| acc | mask);
    let count = key.len();
    let mut out = Vec::new();
    for first in 0..count {
        let mut arc = 0;
        for step in 0..count.saturating_sub(1) {
            arc |= key[(first + step) % count].0;
            let representative = if arc & 1 == 0 { arc } else { full ^ arc };
            if !out.contains(&representative) {
                out.push(representative);
            }
        }
    }
    out.sort_unstable();
    out
}

/// `certified_scaleless`: massless lines and every invariant bound on shell
/// to a structurally zero mass.
pub fn certified_scaleless(key: &FamilyKey, legs: &[LegInput], massless: &[bool]) -> bool {
    if key.iter().any(|&(_, mass)| !massless[mass as usize]) {
        return false;
    }
    let full = key.iter().fold(0, |acc, &(mask, _)| acc | mask);
    let invariants = invariant_masks(key);
    let mut bound = 0;
    for &mask in &invariants {
        let mut binding = None;
        for arc in [mask, full ^ mask] {
            if arc & arc.wrapping_sub(1) == 0 {
                binding = Some(legs[lowest_leg(arc) as usize].mass);
            }
        }
        if let Some(mass) = binding {
            if !massless[mass as usize] {
                return false;
            }
            bound += 1;
        }
    }
    bound == invariants.len()
}

/// Run the loop-prefix recursion (`generate_families`).
pub fn generate_families(input: &FamilyInput) -> Result<FamilyOutput, String> {
    let mut recursion = Recursion::new(input)?;
    let full = recursion.full;
    let first = full & full.wrapping_neg();
    // Admitted routings per key in discovery order; keys are sorted later.
    let mut admitted: HashMap<FamilyKey, Vec<(u32, Vec<u32>, bool)>> = HashMap::new();
    let mut rejected: HashMap<FamilyKey, Rejection> = HashMap::new();
    // Prefix arena: (parent node, position); the stack holds node indices.
    let mut nodes: Vec<(u32, u32)> = Vec::new();
    const ROOT: u32 = u32::MAX;
    for &cut in &input.eligible {
        let mut stack: Vec<(u32, LegSet, u32, u32)> = vec![(ROOT, 0, cut, OrderSpace::EMPTY)];
        while let Some((node, used, species, orders)) = stack.pop() {
            let remaining = full ^ used;
            let mut sub = remaining;
            while sub != 0 {
                if node != ROOT || sub & first != 0 {
                    for position in recursion.positions(sub, species) {
                        let data = &recursion.position_data[position as usize];
                        let (loop_out, position_orders) = (data.loop_out, data.orders);
                        let total = recursion.orders.add(orders, position_orders);
                        if !recursion.orders.within(total, &recursion.cap) {
                            continue;
                        }
                        if sub == remaining {
                            if loop_out == cut && recursion.loop_orders.contains(&total) {
                                let mut positions = Vec::new();
                                let mut at = node;
                                while at != ROOT {
                                    positions.push(nodes[at as usize].1);
                                    at = nodes[at as usize].0;
                                }
                                positions.reverse();
                                positions.push(position);
                                close(&recursion, cut, positions, &mut admitted, &mut rejected)?;
                            }
                        } else if recursion.closable(remaining ^ sub, loop_out, cut, total) {
                            nodes.push((node, position));
                            stack.push(((nodes.len() - 1) as u32, used | sub, loop_out, total));
                        }
                    }
                }
                sub = (sub - 1) & remaining;
            }
        }
    }
    let mut keys: Vec<FamilyKey> = admitted.keys().cloned().collect();
    keys.sort_unstable();
    let mut output = FamilyOutput::default();
    let mut states: HashMap<(i64, u32), u32> = HashMap::new();
    let mut intern = |output: &mut FamilyOutput, parent: i64, label: u32, species: u32| -> u32 {
        *states.entry((parent, label)).or_insert_with(|| {
            output.state_parent.push(parent);
            output
                .state_position
                .push(if parent < 0 { u32::MAX } else { label });
            output.state_species.push(species);
            (output.state_parent.len() - 1) as u32
        })
    };
    for key in keys {
        let routings = admitted.remove(&key).expect("admitted key");
        rejected.remove(&key);
        if certified_scaleless(&key, &input.legs, &input.massless) {
            output.scaleless.push(key);
            continue;
        }
        let mut closures = Vec::with_capacity(routings.len());
        let mut rank = 0;
        for (cut, positions, symmetric) in routings {
            let mut state = intern(&mut output, -1, cut, cut);
            let mut degree = 0;
            for position in positions {
                let data = &recursion.position_data[position as usize];
                degree += data.q_degree;
                state = intern(&mut output, i64::from(state), position, data.loop_out);
            }
            rank = rank.max(degree);
            closures.push(Closure {
                state,
                symmetric,
                multiplicity: input.multiplicity.get(cut as usize).copied().unwrap_or(1),
            });
        }
        output.families.push(Family {
            key,
            rank,
            closures,
        });
    }
    let mut logged: Vec<(FamilyKey, Rejection)> = rejected.into_iter().collect();
    logged.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    output.rejected = logged;
    output.positions = recursion.positions;
    Ok(output)
}

fn close(
    recursion: &Recursion<'_>,
    cut: u32,
    positions: Vec<u32>,
    admitted: &mut HashMap<FamilyKey, Vec<(u32, Vec<u32>, bool)>>,
    rejected: &mut HashMap<FamilyKey, Rejection>,
) -> Result<(), String> {
    let count = positions.len();
    let data = |id: u32| &recursion.position_data[id as usize];
    let mut symmetric = false;
    if count >= 3 {
        if lowest_leg(data(positions[1]).mask) > lowest_leg(data(positions[count - 1]).mask) {
            return Ok(());
        }
    } else {
        let (mirror_cut, mirrored) = recursion.reflected(&positions);
        let routing: Vec<&Position> = positions
            .iter()
            .map(|&id| &recursion.positions[id as usize])
            .collect();
        let ordering = mirror_cut
            .cmp(&cut)
            .then_with(|| mirrored.iter().cmp(routing.iter().copied()));
        if ordering == std::cmp::Ordering::Less {
            return Ok(());
        }
        symmetric = ordering == std::cmp::Ordering::Equal;
    }
    let masks: Vec<LegSet> = positions.iter().map(|&id| data(id).mask).collect();
    let mut masses = Vec::with_capacity(count);
    let mut species_ok = true;
    let mut vertex_names = 0u64;
    for &id in &positions {
        let species = recursion.species(data(id).loop_out)?;
        masses.push(recursion.mass_of_species(data(id).loop_out)?);
        let bits = species
            .perturbative
            .iter()
            .fold(0u64, |bits, &name| bits | (1u64 << name));
        species_ok &= bits & recursion.corrections != 0;
        vertex_names |= recursion
            .orders
            .names(recursion.vertex_orders[recursion.positions[id as usize].vertex as usize]);
    }
    let key = canonical_family_key(&masks, &masses);
    if !(species_ok && vertex_names & recursion.corrections != 0) {
        return Ok(());
    }
    if let Some(reason) = recursion.selection_rejection(&positions)? {
        rejected.entry(key).or_insert(reason);
        return Ok(());
    }
    admitted
        .entry(key)
        .or_default()
        .push((cut, positions, symmetric));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_key_is_the_minimum_reading() {
        // Python: canonical_family_key([4, 1, 2], [m0, m1, m2]).
        let key = canonical_family_key(&[4, 1, 2], &[0, 1, 2]);
        assert_eq!(key, vec![(1, 0), (4, 2), (2, 1)]);
        // The reflected reading from leg 0 takes the mass before it.
        let reflected = canonical_family_key(&[1, 2, 4], &[2, 0, 1]);
        assert_eq!(reflected, vec![(1, 1), (4, 0), (2, 2)]);
    }

    #[test]
    fn invariants_drop_leg_zero() {
        let key = vec![(1, 0), (2, 0), (4, 0), (8, 0)];
        assert_eq!(invariant_masks(&key), vec![2, 4, 6, 8, 12, 14]);
    }
}
