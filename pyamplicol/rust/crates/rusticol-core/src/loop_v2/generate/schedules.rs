// SPDX-License-Identifier: 0BSD

//! Skeletons, the skeleton view, frames and schedule-B suffix tries: the
//! native port of `loops/v2/skeletons.py` (`family_skeletons`,
//! `reflect_ring`) and `loops/v2/schedules.py` (`SkeletonView`, `orient`,
//! `build_trie`, `cheapest`, `build_schedules`), design §3.4, §4.2 (2).
//!
//! Colour states come from the Python colour model (`ColourModel.states`,
//! evaluated once per distinct position, and `ColourModel.reflected` once per
//! state); state ids are ranks of the states' Python order. Contents,
//! skeletons, readings and trie nodes are then built here exactly as Python
//! builds them, in the same order, so the decoded objects are equal.
//! Segment labels are interned first-seen across families in one sequential
//! pass; the tries of the families are independent and built in parallel.

use super::families::{
    Family, FamilyInput, FamilyKey, FamilyOutput, LegSet, Position, canonical_family_key,
    certified_scaleless,
};
use super::orders::{OrderSpace, OrderVec};
use super::{FastMap as HashMap, FastSet};
use rayon::prelude::*;
use serde::Deserialize;
use std::cmp::Ordering;
use std::sync::Arc;

/// The colour model's states and the frame policy.
#[derive(Debug, Clone, Deserialize)]
pub struct ScheduleInput {
    /// Per position id of the family output: its colour-state ids, in the
    /// order of `ColourModel.states`; `None` for unused positions.
    pub states: Vec<Option<Vec<u32>>>,
    /// Per state id: `ColourModel.reflected(state)`.
    pub reflected: Vec<u32>,
    /// Per family (in output order), the skeletons kept by the view, as
    /// flags over the family's skeletons; `None` keeps every skeleton.
    #[serde(default)]
    pub keep: Option<Vec<Vec<bool>>>,
    pub policy: FramePolicy,
    /// The families to schedule: `None` schedules the admitted families
    /// (`FamilyOutput::families`); `Some(indices)` the certified-scaleless
    /// families at those indices of `FamilyOutput::scaleless_families`, in
    /// that order (the poles companion, `LoopSlice._scaleless_companion`).
    /// `keep`, a fixed frame policy and `FamilySchedule::family` index this
    /// list.
    #[serde(default)]
    pub scaleless: Option<Vec<usize>>,
}

/// Frame policies of the production generator.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum FramePolicy {
    /// `schedules.cheapest(rows, model)` with `rows(mask)` the product of
    /// the helicity rows of the mask's legs: `leg_rows[i]` for leg `i` (bit
    /// `i`; the request's helicity count of the leg,
    /// `generator_input.leg_rows`). Required: no count per leg is assumed.
    Cheapest {
        model: CostModel,
        leg_rows: Vec<u64>,
    },
    /// Per family (in output order) the frame `(start, reflected)` to use;
    /// only that frame's trie is built.
    Fixed { frames: Vec<Option<(usize, bool)>> },
}

/// `schedules.CostModel` (work in complex MACs; evaluated in `u128`).
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct CostModel {
    pub fused: u64,
    pub split: u64,
    pub vertex: u64,
    pub seed_fused: u64,
    pub seed_split: u64,
    pub closure: u64,
    pub columns: u64,
}

fn binomial4(n: u128) -> u128 {
    // C(n + 4, 4) for n >= 0.
    (n + 1) * (n + 2) * (n + 3) * (n + 4) / 24
}

impl CostModel {
    fn node(&self, rows_out: u128, rank_in: i64, x_rows: u128) -> Option<u128> {
        let wide = |value: u64| u128::from(value);
        let work = if rank_in == 0 {
            (wide(self.seed_fused).checked_mul(x_rows)?).min(
                wide(self.seed_split)
                    .checked_mul(x_rows)?
                    .checked_add(wide(self.vertex))?,
            )
        } else {
            binomial4(rank_in as u128).checked_mul(
                (wide(self.fused).checked_mul(x_rows)?).min(
                    wide(self.split)
                        .checked_mul(x_rows)?
                        .checked_add(wide(self.vertex))?,
                ),
            )?
        };
        rows_out.checked_mul(wide(self.columns))?.checked_mul(work)
    }

    fn close(&self, rank: i64) -> Option<u128> {
        u128::from(self.closure).checked_mul(binomial4(rank as u128))
    }
}

/// A content: one colour state of one position between two loop species.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Content {
    pub state: u32,
    pub loop_in: u32,
    pub loop_out: u32,
    pub mask: LegSet,
    pub orders: u32,
    pub terms: Vec<Position>,
    pub q_degree: i64,
}

/// A skeleton in its F reading.
#[derive(Debug, Clone)]
pub struct Skeleton {
    pub contents: Vec<u32>,
    pub half_weight: bool,
    pub multiplicity: u32,
    pub orders: u32,
}

#[derive(Debug, Clone)]
pub struct TrieNode {
    pub depth: usize,
    pub orders: u32,
    /// Labels of the undressed positions depth+1..N.
    pub suffix: Vec<u32>,
    pub rank_in: i64,
    pub edges: Vec<(u32, u32)>,
}

/// The schedule-B data of one family.
#[derive(Debug, Clone)]
pub struct FamilySchedule {
    /// Index of the family in the scheduled list (`ScheduleInput::scaleless`;
    /// by default the family output's admitted families).
    pub family: usize,
    /// The view's selection: indices into the family's skeletons
    /// (`Retained::skeletons`).
    pub viewed: Vec<usize>,
    pub frame: (usize, bool),
    /// Leaves: (index into `viewed`, reading as labels).
    pub leaves: Vec<(usize, Vec<u32>)>,
    pub nodes: Vec<TrieNode>,
    pub rank: i64,
    /// `Family.master_keys`: (key, certified scaleless), sorted.
    pub masters: Vec<(FamilyKey, bool)>,
}

#[derive(Debug, Default)]
pub struct ScheduleOutput {
    /// Interned contents; `labels[l]` is the content of segment label `l`.
    pub contents: Vec<Content>,
    pub labels: Vec<u32>,
    pub orders: Vec<OrderVec>,
    pub schedules: Vec<FamilySchedule>,
}

/// The interning state of a schedule build: order vectors, contents and
/// their reflections. A retained build ([`Retained`]) keeps it, so a rebuild
/// from its skeletons continues the same ids.
#[derive(Debug, Clone, Default)]
pub struct ScheduleState {
    pub(crate) orders: OrderSpace,
    position_orders: Vec<u32>,
    position_mask: Vec<LegSet>,
    pub(crate) contents: Vec<Content>,
    content_index: HashMap<Content, u32>,
    /// Per content id: its reflection (`u32::MAX` until computed).
    reflection: Vec<u32>,
}

struct Builder<'a> {
    families: &'a FamilyInput,
    output: &'a FamilyOutput,
    input: &'a ScheduleInput,
    /// The scheduled families (`ScheduleInput::scaleless`).
    list: Vec<&'a Family>,
    state: ScheduleState,
}

fn content_cmp(space: &OrderSpace, a: &Content, b: &Content) -> Ordering {
    a.state
        .cmp(&b.state)
        .then(a.loop_in.cmp(&b.loop_in))
        .then(a.loop_out.cmp(&b.loop_out))
        .then(a.mask.cmp(&b.mask))
        .then_with(|| space.get(a.orders).cmp(space.get(b.orders)))
        .then_with(|| a.terms.cmp(&b.terms))
}

impl<'a> Builder<'a> {
    fn new(
        families: &'a FamilyInput,
        output: &'a FamilyOutput,
        input: &'a ScheduleInput,
    ) -> Result<Self, String> {
        let mut orders = OrderSpace::new();
        let mut position_orders = Vec::with_capacity(output.positions.len());
        let mut position_mask = Vec::with_capacity(output.positions.len());
        for position in &output.positions {
            let mut total =
                orders.intern(families.vertices[position.vertex as usize].orders.clone());
            let mut mask = 0;
            for &block in &position.blocks {
                let current = &families.currents[block as usize];
                let id = orders.intern(current.orders.clone());
                total = orders.add(total, id);
                mask |= current.mask;
            }
            position_orders.push(total);
            position_mask.push(mask);
        }
        let state = ScheduleState {
            orders,
            position_orders,
            position_mask,
            ..ScheduleState::default()
        };
        Self::resume(families, output, input, state)
    }

    /// A builder continuing the interning `state` of an earlier build of the
    /// same family output.
    fn resume(
        families: &'a FamilyInput,
        output: &'a FamilyOutput,
        input: &'a ScheduleInput,
        state: ScheduleState,
    ) -> Result<Self, String> {
        let list = match &input.scaleless {
            None => output.families.iter().collect(),
            Some(indices) => indices
                .iter()
                .map(|&index| {
                    output
                        .scaleless_families
                        .get(index)
                        .ok_or_else(|| format!("no certified-scaleless family {index} to schedule"))
                })
                .collect::<Result<Vec<_>, _>>()?,
        };
        if state.position_orders.len() != output.positions.len() {
            return Err("the schedule state belongs to another family output".into());
        }
        Ok(Self {
            families,
            output,
            input,
            list,
            state,
        })
    }

    fn intern(&mut self, content: Content) -> u32 {
        if let Some(&id) = self.state.content_index.get(&content) {
            return id;
        }
        let id = self.state.contents.len() as u32;
        self.state.content_index.insert(content.clone(), id);
        self.state.contents.push(content);
        id
    }

    fn antiname(&self, field: u32) -> Result<u32, String> {
        self.families
            .species
            .get(field as usize)
            .and_then(Option::as_ref)
            .map(|species| species.antiname)
            .ok_or_else(|| format!("field id {field} is not a loop species"))
    }

    fn reflect_content(&mut self, id: u32) -> Result<u32, String> {
        if let Some(&found) = self.state.reflection.get(id as usize)
            && found != u32::MAX
        {
            return Ok(found);
        }
        let content = self.state.contents[id as usize].clone();
        let mut terms: Vec<Position> = content
            .terms
            .iter()
            .map(|term| Position {
                vertex: self.families.vertices[term.vertex as usize].reflected,
                blocks: term.blocks.clone(),
            })
            .collect();
        terms.sort_unstable();
        let reflected = Content {
            state: self.input.reflected[content.state as usize],
            loop_in: self.antiname(content.loop_out)?,
            loop_out: self.antiname(content.loop_in)?,
            mask: content.mask,
            orders: content.orders,
            q_degree: terms
                .iter()
                .map(|t| self.families.vertices[t.vertex as usize].q_degree)
                .max()
                .unwrap_or(0),
            terms,
        };
        let mirror = self.intern(reflected);
        if self.state.reflection.len() < self.state.contents.len() {
            self.state
                .reflection
                .resize(self.state.contents.len(), u32::MAX);
        }
        self.state.reflection[id as usize] = mirror;
        Ok(mirror)
    }

    /// `reflect_ring`: read a ring backwards from its first position.
    fn reflect_ring(&mut self, contents: &[u32]) -> Result<Vec<u32>, String> {
        let mut order = vec![contents[0]];
        order.extend(contents[1..].iter().rev());
        order.into_iter().map(|c| self.reflect_content(c)).collect()
    }

    fn ring_cmp(&self, a: &[u32], b: &[u32]) -> Ordering {
        for (x, y) in a.iter().zip(b) {
            let ordering = content_cmp(
                &self.state.orders,
                &self.state.contents[*x as usize],
                &self.state.contents[*y as usize],
            );
            if ordering != Ordering::Equal {
                return ordering;
            }
        }
        a.len().cmp(&b.len())
    }

    fn path(&self, state: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let mut at = state as usize;
        while self.output.state_parent[at] >= 0 {
            out.push(self.output.state_position[at]);
            at = self.output.state_parent[at] as usize;
        }
        out.reverse();
        out
    }

    /// `family_skeletons`: the skeletons of one family, sorted.
    fn family_skeletons(&mut self, family: usize) -> Result<Vec<Skeleton>, String> {
        type Label = (u32, u32, u32, LegSet, u32);
        let record = self.list[family];
        let mut ring_index: HashMap<Vec<Label>, usize> = HashMap::default();
        // Per ring: per-slot positions (insertion order, distinct), the
        // distinct position combinations, symmetric flag and multiplicity.
        // (labels, slots, combinations, symmetric, multiplicity) per ring.
        type Ring = (Vec<Label>, Vec<Vec<u32>>, FastSet<Vec<u32>>, bool, u32);
        let mut rings: Vec<Ring> = Vec::new();
        for closure in &record.closures {
            let positions = self.path(closure.state);
            let mut options = Vec::with_capacity(positions.len());
            for &position in &positions {
                options.push(
                    self.input
                        .states
                        .get(position as usize)
                        .and_then(Option::as_ref)
                        .ok_or_else(|| format!("no colour states for position {position}"))?
                        .clone(),
                );
            }
            if options.iter().any(Vec::is_empty) {
                continue;
            }
            let mut choice = vec![0usize; positions.len()];
            'product: loop {
                let labels: Vec<Label> = positions
                    .iter()
                    .zip(&choice)
                    .enumerate()
                    .map(|(slot, (&position, &pick))| {
                        let vertex = &self.families.vertices
                            [self.output.positions[position as usize].vertex as usize];
                        (
                            options[slot][pick],
                            vertex.loop_in,
                            vertex.loop_out,
                            self.state.position_mask[position as usize],
                            self.state.position_orders[position as usize],
                        )
                    })
                    .collect();
                let index = match ring_index.get(&labels) {
                    Some(&index) => index,
                    None => {
                        let index = rings.len();
                        ring_index.insert(labels.clone(), index);
                        rings.push((
                            labels,
                            vec![Vec::new(); positions.len()],
                            Default::default(),
                            false,
                            closure.multiplicity,
                        ));
                        index
                    }
                };
                let entry = &mut rings[index];
                for (slot, &position) in entry.1.iter_mut().zip(&positions) {
                    if !slot.contains(&position) {
                        slot.push(position);
                    }
                }
                entry.2.insert(positions.clone());
                entry.3 |= closure.symmetric;
                if entry.4 != closure.multiplicity {
                    return Err(format!(
                        "inconsistent flavour multiplicity in family {family}"
                    ));
                }
                // itertools.product order: the last slot varies fastest.
                let mut slot = positions.len();
                loop {
                    if slot == 0 {
                        break 'product;
                    }
                    slot -= 1;
                    choice[slot] += 1;
                    if choice[slot] < options[slot].len() {
                        break;
                    }
                    choice[slot] = 0;
                }
            }
        }
        let mut out: Vec<Skeleton> = Vec::new();
        let mut seen: HashMap<Vec<u32>, ()> = HashMap::default();
        for (labels, slots, combos, symmetric, multiplicity) in rings {
            let count: usize = slots.iter().map(Vec::len).product();
            if count != combos.len() {
                return Err(format!(
                    "routings of family {family} do not factorise per position"
                ));
            }
            let mut contents = Vec::with_capacity(labels.len());
            for ((state, loop_in, loop_out, mask, orders), slot) in labels.into_iter().zip(slots) {
                let mut terms: Vec<Position> = slot
                    .iter()
                    .map(|&p| self.output.positions[p as usize].clone())
                    .collect();
                terms.sort_unstable();
                let q_degree = terms
                    .iter()
                    .map(|t| self.families.vertices[t.vertex as usize].q_degree)
                    .max()
                    .unwrap_or(0);
                contents.push(self.intern(Content {
                    state,
                    loop_in,
                    loop_out,
                    mask,
                    orders,
                    terms,
                    q_degree,
                }));
            }
            let mut half_weight = false;
            if symmetric {
                let mirror = self.reflect_ring(&contents)?;
                if mirror == contents {
                    half_weight = true;
                }
                if self.ring_cmp(&mirror, &contents) == Ordering::Less {
                    contents = mirror;
                }
            }
            if seen.contains_key(&contents) {
                continue;
            }
            seen.insert(contents.clone(), ());
            let mut orders = OrderSpace::EMPTY;
            for &content in &contents {
                orders = self
                    .state
                    .orders
                    .add(orders, self.state.contents[content as usize].orders);
            }
            out.push(Skeleton {
                contents,
                half_weight,
                multiplicity,
                orders,
            });
        }
        let space = &self.state.orders;
        let table = &self.state.contents;
        out.sort_by(|a, b| {
            for (x, y) in a.contents.iter().zip(&b.contents) {
                let ordering = content_cmp(space, &table[*x as usize], &table[*y as usize]);
                if ordering != Ordering::Equal {
                    return ordering;
                }
            }
            a.contents.len().cmp(&b.contents.len())
        });
        Ok(out)
    }
}

/// `ring_readings(count)`: forward from each start, then reflected.
fn ring_readings(count: usize) -> impl Iterator<Item = (usize, bool)> {
    [false, true]
        .into_iter()
        .flat_map(move |reflected| (0..count).map(move |start| (start, reflected)))
}

/// `FamilyFrame.apply`: a reading of the family key.
fn apply_frame(key: &FamilyKey, start: usize, reflected: bool) -> FamilyKey {
    let count = key.len();
    (0..count)
        .map(|step| {
            if reflected {
                let i = (start + count - step % count) % count;
                (key[i].0, key[(i + count - 1) % count].1)
            } else {
                key[(start + step) % count]
            }
        })
        .collect()
}

/// Advance `kept` to the next combination of `0..n` (itertools order).
fn next_combination(kept: &mut [usize], n: usize) -> bool {
    let k = kept.len();
    let mut i = k;
    while i > 0 {
        i -= 1;
        if kept[i] < n - k + i {
            kept[i] += 1;
            for j in i + 1..k {
                kept[j] = kept[j - 1] + 1;
            }
            return true;
        }
    }
    false
}

/// `Family.master_keys`: canonical pinched sub-rings with 1..4 propagators.
fn master_keys(key: &FamilyKey, families: &FamilyInput) -> Vec<(FamilyKey, bool)> {
    let masks: Vec<LegSet> = key.iter().map(|&(mask, _)| mask).collect();
    let masses: Vec<u32> = key.iter().map(|&(_, mass)| mass).collect();
    let count = masks.len();
    let mut found: HashMap<FamilyKey, bool> = HashMap::default();
    for lines in 1..=count.min(4) {
        let mut kept: Vec<usize> = (0..lines).collect();
        loop {
            let mut merged = Vec::with_capacity(lines);
            let mut merged_masses = Vec::with_capacity(lines);
            for (index, &gap) in kept.iter().enumerate() {
                let following = kept[(index + 1) % lines];
                let mut mask = 0;
                let mut step = (gap + 1) % count;
                loop {
                    mask |= masks[step];
                    if step == following {
                        break;
                    }
                    step = (step + 1) % count;
                }
                merged.push(mask);
                merged_masses.push(masses[following]);
            }
            let sub = canonical_family_key(&merged, &merged_masses);
            let scaleless = certified_scaleless(&sub, &families.legs, &families.massless);
            found.insert(sub, scaleless);
            if !next_combination(&mut kept, count) {
                break;
            }
        }
    }
    let mut out: Vec<(FamilyKey, bool)> = found.into_iter().collect();
    out.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Per-family data the tries need, after the sequential label pass.
struct Viewed {
    /// Per viewed skeleton: its 2N readings as label rings.
    readings: Vec<Vec<Vec<u32>>>,
    /// Loop orders of each viewed skeleton (order ids).
    orders: Vec<u32>,
}

/// Segment data per label: (mask, mass id, q-degree).
type SegmentTable = Vec<(LegSet, u32, i64)>;

fn build_trie(
    key: &FamilyKey,
    oriented: &[(usize, Vec<u32>)],
    orders: &[u32],
    space: &OrderSpace,
    segments: &SegmentTable,
) -> (Vec<TrieNode>, i64) {
    let count = key.len();
    let mut nodes: Vec<TrieNode> = Vec::new();
    let mut previous: HashMap<(u32, Vec<u32>), u32> = HashMap::default();
    for (leaf, (skeleton, ring)) in oriented.iter().enumerate() {
        previous
            .entry((orders[*skeleton], ring.clone()))
            .or_insert(leaf as u32);
    }
    for depth in 1..=count {
        let mut edges: HashMap<(u32, Vec<u32>), HashMap<u32, u32>> = HashMap::default();
        let mut rank_in: HashMap<(u32, Vec<u32>), i64> = HashMap::default();
        for (skeleton, ring) in oriented {
            let order = orders[*skeleton];
            let key = (order, ring[depth..].to_vec());
            let child = previous[&(order, ring[depth - 1..].to_vec())];
            edges
                .entry(key.clone())
                .or_default()
                .insert(ring[depth - 1], child);
            let degree: i64 = ring[..depth - 1]
                .iter()
                .map(|&label| segments[label as usize].2)
                .sum();
            let entry = rank_in.entry(key).or_insert(0);
            *entry = (*entry).max(degree);
        }
        let mut keys: Vec<(u32, Vec<u32>)> = edges.keys().cloned().collect();
        keys.sort_unstable_by(|a, b| {
            space
                .get(a.0)
                .cmp(space.get(b.0))
                .then_with(|| a.1.cmp(&b.1))
        });
        let mut current: HashMap<(u32, Vec<u32>), u32> =
            HashMap::with_capacity_and_hasher(keys.len(), Default::default());
        for key in keys {
            let mut node_edges: Vec<(u32, u32)> =
                edges[&key].iter().map(|(&s, &c)| (s, c)).collect();
            node_edges.sort_unstable();
            current.insert(key.clone(), nodes.len() as u32);
            nodes.push(TrieNode {
                depth,
                orders: key.0,
                rank_in: rank_in[&key],
                suffix: key.1,
                edges: node_edges,
            });
        }
        previous = current;
    }
    let rank = oriented
        .iter()
        .map(|(_, ring)| ring.iter().map(|&l| segments[l as usize].2).sum::<i64>())
        .max()
        .unwrap_or(0);
    (nodes, rank)
}

/// The helicity rows of a segment's legs: the product of `leg_rows` over
/// the mask's legs; None past 128 bits or for a leg without its count.
fn segment_rows(mask: LegSet, leg_rows: &[u64]) -> Option<u128> {
    let mut rows: u128 = 1;
    let mut rest = mask;
    while rest != 0 {
        let leg = rest.trailing_zeros() as usize;
        rows = rows.checked_mul(u128::from(*leg_rows.get(leg)?))?;
        rest &= rest - 1;
    }
    Some(rows)
}

fn trie_cost(
    nodes: &[TrieNode],
    rank: i64,
    count: usize,
    model: &CostModel,
    leg_rows: &[u64],
    segments: &SegmentTable,
) -> Result<u128, String> {
    let overflow =
        || "the schedule-B cost exceeds 128 bits (or a leg has no helicity count)".to_string();
    let rows = |label: u32| -> Option<u128> { segment_rows(segments[label as usize].0, leg_rows) };
    let accumulators = nodes.iter().filter(|node| node.depth == count).count() as u128;
    let mut total = model
        .close(rank)
        .and_then(|c| c.checked_mul(accumulators))
        .ok_or_else(overflow)?;
    for node in nodes {
        let mut rows_out: u128 = 1;
        for &label in &node.suffix {
            rows_out = rows_out
                .checked_mul(rows(label).ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        let mut x_rows: u128 = 0;
        for &(label, _) in &node.edges {
            x_rows = x_rows
                .checked_add(rows(label).ok_or_else(overflow)?)
                .ok_or_else(overflow)?;
        }
        total = total
            .checked_add(
                model
                    .node(rows_out, node.rank_in, x_rows)
                    .ok_or_else(overflow)?,
            )
            .ok_or_else(overflow)?;
    }
    Ok(total)
}

/// Contents, order vectors and the skeletons of every family.
pub type SkeletonOutput = (Vec<Content>, Vec<OrderVec>, Vec<Vec<Skeleton>>);

/// The skeletons of every family (`family_skeletons`) without the view:
/// `(contents, orders, skeletons per family)`.
pub fn build_skeletons(
    families: &FamilyInput,
    output: &FamilyOutput,
    input: &ScheduleInput,
) -> Result<SkeletonOutput, String> {
    let mut builder = Builder::new(families, output, input)?;
    let mut all = Vec::with_capacity(builder.list.len());
    for family in 0..builder.list.len() {
        all.push(builder.family_skeletons(family)?);
    }
    let orders = builder.state.order_vectors();
    Ok((builder.state.contents, orders, all))
}

/// Skeletons, the view and the schedule-B tries of every family
/// (`schedules.build_schedules` with the native frame policies).
pub fn build_schedules(
    families: &FamilyInput,
    output: &FamilyOutput,
    input: &ScheduleInput,
) -> Result<ScheduleOutput, String> {
    let retained = build_retained(families, output, input)?;
    Ok(ScheduleOutput {
        orders: retained.state.order_vectors(),
        contents: retained.state.contents,
        labels: retained.labels,
        schedules: retained.schedules,
    })
}

/// A schedule build kept by the arena (`arena.rs`): the colour input it was
/// built with, every skeleton of the scheduled families (shared with the
/// rebuilds), the builder's interning state and the view and tries.
#[derive(Debug)]
pub struct Retained {
    pub(crate) states: Vec<Option<Vec<u32>>>,
    pub(crate) reflected: Vec<u32>,
    pub(crate) scaleless: Option<Vec<usize>>,
    /// Built without keep flags: every skeleton was offered to the view, so
    /// the build is the base of a rebuild ([`rebuild`]).
    pub(crate) complete: bool,
    /// Every skeleton of every scheduled family (`family_skeletons`), in the
    /// scheduled list's order.
    pub(crate) skeletons: Arc<Vec<Vec<Skeleton>>>,
    pub(crate) state: ScheduleState,
    /// `labels[l]`: the content of segment label `l`.
    pub(crate) labels: Vec<u32>,
    pub(crate) schedules: Vec<FamilySchedule>,
}

impl ScheduleState {
    /// The order vectors by id.
    pub fn order_vectors(&self) -> Vec<OrderVec> {
        (0..self.orders.len() as u32)
            .map(|id| self.orders.get(id).clone())
            .collect()
    }
}

/// `build_schedules`, kept for rebuilds and in-crate readers.
pub fn build_retained(
    families: &FamilyInput,
    output: &FamilyOutput,
    input: &ScheduleInput,
) -> Result<Retained, String> {
    let mut builder = Builder::new(families, output, input)?;
    // Skeletons per family (sequential: contents are interned first-seen).
    let mut all: Vec<Vec<Skeleton>> = Vec::with_capacity(builder.list.len());
    for family in 0..builder.list.len() {
        all.push(builder.family_skeletons(family)?);
    }
    let all = Arc::new(all);
    let (labels, schedules) = view_and_tries(&mut builder, &all)?;
    Ok(Retained {
        states: input.states.clone(),
        reflected: input.reflected.clone(),
        scaleless: input.scaleless.clone(),
        complete: input.keep.is_none(),
        skeletons: all,
        state: builder.state,
        labels,
        schedules,
    })
}

/// The schedules of `input` (its keep flags and frame policy) built from the
/// skeletons of a complete retained build of the same colour states, without
/// recomputing them: equal to `build_schedules(families, output, input)`
/// (the LC/NLC prune, `LoopSlice._prune_colour`).
pub fn rebuild(
    families: &FamilyInput,
    output: &FamilyOutput,
    base: &Retained,
    input: &ScheduleInput,
) -> Result<Retained, String> {
    if !base.complete {
        return Err("a schedule rebuild needs a build that kept every skeleton".into());
    }
    if input.states != base.states
        || input.reflected != base.reflected
        || input.scaleless != base.scaleless
    {
        return Err("a schedule rebuild needs the colour states and families of its base".into());
    }
    let mut builder = Builder::resume(families, output, input, base.state.clone())?;
    let (labels, schedules) = view_and_tries(&mut builder, &base.skeletons)?;
    Ok(Retained {
        states: base.states.clone(),
        reflected: base.reflected.clone(),
        scaleless: base.scaleless.clone(),
        complete: input.keep.is_none(),
        skeletons: Arc::clone(&base.skeletons),
        state: builder.state,
        labels,
        schedules,
    })
}

/// The view (labels first-seen across families, in family order) and the
/// schedule-B trie of every family with a viewed skeleton.
fn view_and_tries(
    builder: &mut Builder<'_>,
    all: &[Vec<Skeleton>],
) -> Result<(Vec<u32>, Vec<FamilySchedule>), String> {
    let input = builder.input;
    let families = builder.families;
    let list = builder.list.clone();
    // The view: labels first-seen across families, in family order.
    let mut label_of: HashMap<u32, u32> = HashMap::default();
    let mut labels: Vec<u32> = Vec::new();
    let mut viewed_all: Vec<(Vec<usize>, Viewed)> = Vec::with_capacity(all.len());
    for (family, skeletons) in all.iter().enumerate() {
        let mut seen: HashMap<Vec<u32>, ()> = HashMap::default();
        let mut viewed = Vec::new();
        let mut data = Viewed {
            readings: Vec::new(),
            orders: Vec::new(),
        };
        for (index, skeleton) in skeletons.iter().enumerate() {
            if let Some(keep) = &input.keep {
                let flags = keep
                    .get(family)
                    .ok_or_else(|| format!("no keep flags for family {family}"))?;
                if !*flags.get(index).ok_or_else(|| {
                    format!("no keep flag for skeleton {index} of family {family}")
                })? {
                    continue;
                }
            }
            let mut label = |content: u32, label_of: &mut HashMap<u32, u32>| -> u32 {
                *label_of.entry(content).or_insert_with(|| {
                    labels.push(content);
                    (labels.len() - 1) as u32
                })
            };
            let forward: Vec<u32> = skeleton
                .contents
                .iter()
                .map(|&c| label(c, &mut label_of))
                .collect();
            let mirror_contents = builder.reflect_ring(&skeleton.contents)?;
            let mirror: Vec<u32> = mirror_contents
                .iter()
                .map(|&c| label(c, &mut label_of))
                .collect();
            let count = forward.len();
            let readings: Vec<Vec<u32>> = ring_readings(count)
                .map(|(start, reflected)| {
                    (0..count)
                        .map(|step| {
                            if reflected {
                                mirror[(step + count - start % count) % count]
                            } else {
                                forward[(start + step) % count]
                            }
                        })
                        .collect()
                })
                .collect();
            let canonical = readings.iter().min().expect("non-empty").clone();
            if seen.contains_key(&canonical) {
                continue;
            }
            seen.insert(canonical, ());
            viewed.push(index);
            data.readings.push(readings);
            data.orders.push(skeleton.orders);
        }
        viewed_all.push((viewed, data));
    }
    let segments: SegmentTable = labels
        .iter()
        .map(|&content| {
            let content = &builder.state.contents[content as usize];
            let mass = families.species[content.loop_out as usize]
                .as_ref()
                .map_or(u32::MAX, |species| species.mass);
            (content.mask, mass, content.q_degree)
        })
        .collect();
    if segments.iter().any(|segment| segment.1 == u32::MAX) {
        return Err("a content leaves the loop on a field that is not a species".into());
    }
    let space = &builder.state.orders;
    let schedules: Vec<Result<Option<FamilySchedule>, String>> = viewed_all
        .into_par_iter()
        .enumerate()
        .map(|(family, (viewed, data))| {
            if viewed.is_empty() {
                return Ok(None);
            }
            let key = &list[family].key;
            let count = key.len();
            // frames(family): distinct readings of the key, first seen.
            let mut targets: Vec<(FamilyKey, (usize, bool))> = Vec::new();
            for (start, reflected) in ring_readings(count) {
                let target = apply_frame(key, start, reflected);
                if !targets.iter().any(|(t, _)| *t == target) {
                    targets.push((target, (start, reflected)));
                }
            }
            let wanted: Option<(usize, bool)> = match &input.policy {
                FramePolicy::Cheapest { .. } => None,
                FramePolicy::Fixed { frames } => Some(
                    frames
                        .get(family)
                        .copied()
                        .flatten()
                        .ok_or_else(|| format!("no fixed frame for family {family}"))?,
                ),
            };
            // (frame, leaves, nodes, rank, cost) of the cheapest frame so far.
            type Candidate = (
                (usize, bool),
                Vec<(usize, Vec<u32>)>,
                Vec<TrieNode>,
                i64,
                u128,
            );
            let mut best: Option<Candidate> = None;
            // orient: per skeleton and frame, the fitting reading that
            // `shared_suffix` picks (first minimum of (ring[1:], ring)).
            let target_index: HashMap<&FamilyKey, usize> = targets
                .iter()
                .enumerate()
                .map(|(i, (target, _))| (target, i))
                .collect();
            let mut chosen: Vec<Vec<Option<usize>>> =
                vec![vec![None; targets.len()]; data.readings.len()];
            for (index, readings) in data.readings.iter().enumerate() {
                for (r, ring) in readings.iter().enumerate() {
                    let sequence: FamilyKey = ring
                        .iter()
                        .map(|&label| {
                            let segment = segments[label as usize];
                            (segment.0, segment.1)
                        })
                        .collect();
                    let Some(&t) = target_index.get(&sequence) else {
                        continue;
                    };
                    let better = match chosen[index][t] {
                        None => true,
                        Some(previous) => {
                            let old = &readings[previous];
                            (&ring[1..], ring) < (&old[1..], old)
                        }
                    };
                    if better {
                        chosen[index][t] = Some(r);
                    }
                }
            }
            if chosen.iter().any(|row| row.iter().any(Option::is_none)) {
                return Err(format!("skeleton does not belong to family {family}"));
            }
            for (t, (_target, frame)) in targets.iter().enumerate() {
                let oriented: Vec<(usize, Vec<u32>)> = chosen
                    .iter()
                    .enumerate()
                    .map(|(index, row)| {
                        (
                            index,
                            data.readings[index][row[t].expect("checked")].clone(),
                        )
                    })
                    .collect();
                if let Some(frame_wanted) = wanted
                    && *frame != frame_wanted
                {
                    continue;
                }
                let (nodes, rank) = build_trie(key, &oriented, &data.orders, space, &segments);
                let cost = match &input.policy {
                    FramePolicy::Cheapest { model, leg_rows } => {
                        trie_cost(&nodes, rank, count, model, leg_rows, &segments)?
                    }
                    FramePolicy::Fixed { .. } => 0,
                };
                if best.as_ref().is_none_or(|b| cost < b.4) {
                    best = Some((*frame, oriented, nodes, rank, cost));
                }
            }
            let (frame, leaves, nodes, rank, _) =
                best.ok_or_else(|| format!("the fixed frame is not a frame of family {family}"))?;
            Ok(Some(FamilySchedule {
                family,
                viewed,
                frame,
                leaves,
                nodes,
                rank,
                masters: master_keys(key, families),
            }))
        })
        .collect();
    let mut out = Vec::with_capacity(schedules.len());
    for schedule in schedules {
        if let Some(schedule) = schedule? {
            out.push(schedule);
        }
    }
    Ok((labels, out))
}

#[cfg(test)]
mod tests {
    use super::segment_rows;

    #[test]
    fn segment_rows_are_the_product_of_the_legs_helicity_counts() {
        // Two rows per leg (massless vectors and fermions).
        assert_eq!(segment_rows(0b1011, &[2; 4]), Some(8));
        // A scalar (1), a massive vector (3), a fermion (2), a massive
        // spin-2 state (5): legs 0..3.
        let counts = [1, 3, 2, 5];
        assert_eq!(segment_rows(0b0001, &counts), Some(1));
        assert_eq!(segment_rows(0b0110, &counts), Some(6));
        assert_eq!(segment_rows(0b1111, &counts), Some(30));
        // A leg without its count, and a product past 128 bits.
        assert_eq!(segment_rows(0b10000, &counts), None);
        let wide = [u64::MAX; 3];
        assert_eq!(segment_rows(0b11, &wide), Some(u128::from(u64::MAX).pow(2)));
        assert_eq!(segment_rows(0b111, &wide), None);
    }

    #[test]
    fn the_cheapest_policy_requires_the_legs_helicity_counts() {
        // No silent two-rows-per-leg default (genericity audit 2, m8).
        let model = r#"{"fused": 1, "split": 1, "vertex": 1, "seed_fused": 1,
            "seed_split": 1, "closure": 1, "columns": 1}"#;
        let without = format!(r#"{{"kind": "cheapest", "model": {model}}}"#);
        let error = serde_json::from_str::<super::FramePolicy>(&without).unwrap_err();
        assert!(error.to_string().contains("leg_rows"), "{error}");
        let with = format!(r#"{{"kind": "cheapest", "model": {model}, "leg_rows": [2, 3]}}"#);
        let policy: super::FramePolicy = serde_json::from_str(&with).unwrap();
        assert!(
            matches!(policy, super::FramePolicy::Cheapest { leg_rows, .. } if leg_rows == [2, 3])
        );
    }
}
