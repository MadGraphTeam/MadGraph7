// SPDX-License-Identifier: 0BSD

//! The loop-request recursion of `GenericDAGCompiler.compile`
//! (`loop_blocks=True`): sources, the sweep over admitted leg subsets and the
//! final prune to admitted blocks (`_finish_loop_blocks`).
//!
//! In loop-request mode the oracle has no Born-closure pruning, no
//! reflection reuse and no online evaluation reuse; colour states are
//! sector-free and candidate right currents are looked up by particle. The
//! order of every loop below is the oracle's, so current and interaction ids
//! are its first-seen ids.

use super::colour::ColourEngine;
use super::exact::complex_mul;
use super::input::{DagInput, FlavourOp};
use super::reuse::{Kernel, canonical_kernel_evaluation};
use super::{
    ColourState, ColumnarDag, DagError, DagResult, NO_HELICITY, NO_LEG, Orders, Species, Tables,
    VertexKey, invalid, unsupported,
};
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};

/// FxHash (rustc's hasher): the builder's maps are keyed by small integers.
#[derive(Default, Clone, Copy)]
pub struct FxHasher(u64);

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(word));
        }
    }
    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.write_u64(value as u64);
    }
    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.write_u64(value as u64);
    }
    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

pub type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;
pub type FxSet<K> = std::collections::HashSet<K, BuildHasherDefault<FxHasher>>;

/// The oracle's interaction deduplication key: vertex, inputs, result and
/// colour weight (variant 0).
type InteractionKey = (u32, u32, u32, u32, [u64; 2]);
/// The compile-time evaluation group key: kernel class, canonical inputs,
/// result particle and chirality, coupling.
type GroupKey = (u32, u32, u32, u32, i8, [u64; 2]);
/// (left colour, right colour, vertex, merged word) of a colour merge.
type CombineKey = (u32, u32, u32, u32);

/// An interning table: ids in first-seen order.
pub struct Interner<T: Eq + Hash + Clone> {
    pub values: Vec<T>,
    ids: FxMap<T, u32>,
}

impl<T: Eq + Hash + Clone> Default for Interner<T> {
    fn default() -> Self {
        Self {
            values: Vec::new(),
            ids: FxMap::default(),
        }
    }
}

impl<T: Eq + Hash + Clone> Interner<T> {
    pub fn intern(&mut self, value: T) -> u32 {
        if let Some(id) = self.ids.get(&value) {
            return *id;
        }
        let id = self.values.len() as u32;
        self.ids.insert(value.clone(), id);
        self.values.push(value);
        id
    }

    pub fn get(&self, id: u32) -> &T {
        &self.values[id as usize]
    }
}

/// The full `CurrentIndex` identity; every other field is a function of
/// these (labels and momentum of the mask; quantum numbers and auxiliary
/// kind of the particle).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CurrentKey {
    pub particle: u32,
    pub mask: u64,
    pub ancestry: u64,
    pub chirality: i8,
    pub spin: i32,
    pub flavour: u32,
    pub colour: u32,
    pub orders: u32,
    pub word: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Current {
    pub key: CurrentKey,
    pub is_source: bool,
    pub leg: i32,
    pub helicity: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct Interaction {
    pub vertex: u32,
    pub left: u32,
    pub right: u32,
    pub result: u32,
    pub coupling: [f64; 2],
    pub coupling_identity: u32,
    pub colour_weight: [f64; 2],
    pub group: i64,
    pub factor: [f64; 2],
}

/// The builder's DAG before it is written in columns.
#[derive(Default)]
pub struct Dag {
    /// Whether the oracle finishes the loop blocks (prune, reuse, signs);
    /// false when a leg's particle is ignored (it returns the sources).
    pub finish: bool,
    pub currents: Vec<Current>,
    pub interactions: Vec<Interaction>,
    pub sources: Vec<u32>,
    pub words: Interner<Vec<u8>>,
    pub colours: Interner<ColourState>,
    pub flavours: Interner<Vec<i64>>,
    pub orders: Interner<Orders>,
}

/// `_masks_by_size`: every submask of `full`, by (size, value).
pub fn masks_by_size(full: u64) -> Vec<u64> {
    let mut masks = Vec::new();
    let mut submask = full;
    while submask != 0 {
        masks.push(submask);
        submask = (submask - 1) & full;
    }
    masks.sort_by_key(|mask| (mask.count_ones(), *mask));
    masks
}

/// `_ordered_splits`.
fn ordered_splits(mask: u64) -> Vec<(u64, u64)> {
    let mut splits = Vec::new();
    let mut left = (mask - 1) & mask;
    while left != 0 {
        let right = mask ^ left;
        if right != 0 {
            splits.push((left, right));
        }
        left = (left - 1) & mask;
    }
    splits
}

/// `_loop_block_masks`: admitted leg sets and all of their subsets.
fn loop_block_masks(input: &DagInput) -> std::collections::HashSet<u64> {
    let full = input.full_mask;
    let mut allowed = std::collections::HashSet::new();
    let legs: Vec<u64> = (0..64 - full.leading_zeros())
        .map(|bit| 1u64 << bit)
        .collect();
    for mask in masks_by_size(full).into_iter().rev() {
        if mask & (mask - 1) == 0 || mask == full {
            continue;
        }
        if input.admitted_masks.contains(&mask)
            || legs
                .iter()
                .any(|leg| mask & leg == 0 && allowed.contains(&(mask | leg)))
        {
            allowed.insert(mask);
        }
    }
    allowed
}

struct Table {
    ids: FxMap<CurrentKey, u32>,
    by_mask: FxMap<u64, Vec<u32>>,
    by_mask_particle: FxMap<(u64, u32), Vec<u32>>,
}

impl Dag {
    fn add_or_get(
        &mut self,
        table: &mut Table,
        key: CurrentKey,
        is_source: bool,
        leg: i32,
        helicity: i32,
    ) -> u32 {
        if let Some(id) = table.ids.get(&key) {
            return *id;
        }
        let id = self.currents.len() as u32;
        table.ids.insert(key, id);
        self.currents.push(Current {
            key,
            is_source,
            leg,
            helicity,
        });
        table.by_mask.entry(key.mask).or_default().push(id);
        table
            .by_mask_particle
            .entry((key.mask, key.particle))
            .or_default()
            .push(id);
        id
    }

    /// The columns of the finished DAG (`GenericDAG` currents, sources and
    /// interactions).
    pub fn into_columns(self, input: &DagInput) -> ColumnarDag {
        let mut species_ids: FxMap<u32, u32> = FxMap::default();
        let mut tables = Tables {
            flavours: self.flavours.values,
            colours: self.colours.values,
            orders: self.orders.values,
            ..Default::default()
        };
        let n = self.currents.len();
        let mut dag = ColumnarDag {
            c_word_offsets: Vec::with_capacity(n + 1),
            ..Default::default()
        };
        dag.c_word_offsets.push(0);
        for current in &self.currents {
            let key = &current.key;
            let particle = &input.particles[key.particle as usize];
            let species = *species_ids.entry(key.particle).or_insert_with(|| {
                tables.species.push(Species {
                    pdg: particle.pdg,
                    auxiliary: particle.auxiliary.clone(),
                    quantum_numbers: particle.quantum_numbers.clone(),
                });
                (tables.species.len() - 1) as u32
            });
            dag.c_species.push(species);
            dag.c_mask.push(key.mask);
            dag.c_momentum.push(key.mask);
            dag.c_ancestry.push(key.ancestry);
            dag.c_chirality.push(key.chirality);
            dag.c_spin.push(key.spin);
            dag.c_flavour.push(key.flavour);
            dag.c_colour.push(key.colour);
            dag.c_orders.push(key.orders);
            dag.c_dimension.push(particle.dimension(key.chirality));
            dag.c_source.push(current.is_source as u8);
            dag.c_leg.push(current.leg);
            dag.c_helicity.push(current.helicity);
            dag.c_words.extend_from_slice(self.words.get(key.word));
            dag.c_word_offsets.push(dag.c_words.len() as u32);
        }
        dag.sources = self.sources;
        let mut vertex_ids: FxMap<u32, u32> = FxMap::default();
        for interaction in &self.interactions {
            let vertex = &input.vertices[interaction.vertex as usize];
            let id = *vertex_ids.entry(interaction.vertex).or_insert_with(|| {
                tables.vertices.push(VertexKey {
                    kind: vertex.kind,
                    particles: vertex.pdgs,
                    backend: vertex.backend.clone(),
                    full_tensor_network_ready: vertex.full_tensor_network_ready,
                });
                (tables.vertices.len() - 1) as u32
            });
            dag.i_vertex.push(id);
            dag.i_left.push(interaction.left);
            dag.i_right.push(interaction.right);
            dag.i_result.push(interaction.result);
            dag.i_coupling.push(interaction.coupling);
            dag.i_colour_weight.push(interaction.colour_weight);
            dag.i_group.push(interaction.group);
            dag.i_factor.push(interaction.factor);
        }
        dag.tables = tables;
        dag
    }
}

fn label_bit(label: u8) -> u64 {
    1u64 << (label - 1)
}

fn mask_labels(mask: u64) -> Vec<u8> {
    (0..64u8)
        .filter(|bit| mask >> bit & 1 == 1)
        .map(|bit| bit + 1)
        .collect()
}

/// A float pair as the oracle's tuple equality sees it (`0.0 == -0.0`).
fn weight_key(weight: [f64; 2]) -> DagResult<[u64; 2]> {
    let key = |value: f64| {
        if value.is_nan() {
            unsupported("a NaN colour weight")
        } else if value == 0.0 {
            Ok(0u64)
        } else {
            Ok(value.to_bits())
        }
    };
    Ok([key(weight[0])?, key(weight[1])?])
}

/// `GenericDAGCompiler.compile` in loop-request mode up to
/// `_finish_loop_blocks`.
pub fn compile(input: &DagInput) -> DagResult<Dag> {
    let engine = ColourEngine::new(input);
    let mut dag = Dag::default();
    let mut table = Table {
        ids: FxMap::default(),
        by_mask: FxMap::default(),
        by_mask_particle: FxMap::default(),
    };
    let empty_orders = dag.orders.intern(Vec::new());
    // `_build_sources`: one helicity-ancestry bit per (leg, colour, state).
    let mut next_bit = 0u32;
    for sources in &input.sources {
        let leg = input
            .legs
            .iter()
            .find(|leg| leg.label == sources.label)
            .ok_or_else(|| DagError::Malformed(format!("source leg {}", sources.label)))?;
        let flavour = dag.flavours.intern(vec![leg.pdg]);
        let word = dag.words.intern(vec![leg.label]);
        for (sector_id, line_groups) in &sources.colours {
            let colour = dag.colours.intern(ColourState {
                accuracy: input.accuracy.clone(),
                sector_id: *sector_id,
                line_groups: line_groups.clone(),
                basis_key: Vec::new(),
                traces: Vec::new(),
                lines: Vec::new(),
                nc_grade: None,
            });
            for state in &sources.states {
                if next_bit >= 64 {
                    return unsupported("more than 64 source states (helicity ancestry bits)");
                }
                let key = CurrentKey {
                    particle: leg.particle,
                    mask: label_bit(leg.label),
                    ancestry: 1u64 << next_bit,
                    chirality: state.chirality,
                    spin: state.spin_state,
                    flavour,
                    colour,
                    orders: empty_orders,
                    word,
                };
                next_bit += 1;
                let id = dag.add_or_get(&mut table, key, true, leg.label as i32, state.helicity);
                dag.sources.push(id);
            }
        }
    }
    if input.early_sources_only {
        return Ok(dag);
    }
    let loop_masks = loop_block_masks(input);
    let full = input.full_mask;
    let recursive: Vec<u64> = masks_by_size(full)
        .into_iter()
        .filter(|mask| mask & (mask - 1) != 0 && *mask != full && loop_masks.contains(mask))
        .collect();
    let mut order_cache: FxMap<(u32, u32, u32), u32> = FxMap::default();
    let mut limit_cache: FxMap<u32, bool> = FxMap::default();
    // The oracle's `ordered_combination_labels` cache, keyed as it is
    // (sector, left word, right word, vertex), first value kept.
    let mut label_cache: FxMap<(i64, u32, u32, u32), Option<u32>> = FxMap::default();
    let mut interaction_keys: FxSet<InteractionKey> = FxSet::default();
    let mut compile_groups: FxMap<GroupKey, i64> = FxMap::default();
    let mut combine_cache: FxMap<CombineKey, Option<(u32, u32)>> = FxMap::default();
    let mut flavour_cache: FxMap<(u32, u32, u32), u32> = FxMap::default();
    let mut candidates: Vec<u32> = Vec::new();
    for mask in recursive {
        let labels = mask_labels(mask);
        for (left_mask, right_mask) in ordered_splits(mask) {
            let Some(left_ids) = table.by_mask.get(&left_mask) else {
                continue;
            };
            if left_ids.is_empty() || !table.by_mask.contains_key(&right_mask) {
                continue;
            }
            let left_ids = left_ids.clone();
            for left_id in left_ids {
                let left = dag.currents[left_id as usize];
                let rights = input.right_particles(left.key.particle);
                if rights.is_empty() {
                    continue;
                }
                candidates.clear();
                for particle in rights {
                    if let Some(ids) = table.by_mask_particle.get(&(right_mask, *particle)) {
                        candidates.extend_from_slice(ids);
                    }
                }
                for &right_id in &candidates {
                    let right = dag.currents[right_id as usize];
                    for &row in input.vertices_accepting(left.key.particle, right.key.particle) {
                        let vertex = &input.vertices[row as usize];
                        if !vertex.usable {
                            continue;
                        }
                        let orders =
                            match order_cache.get(&(left.key.orders, right.key.orders, row)) {
                                Some(orders) => *orders,
                                None => {
                                    let mut totals =
                                        std::collections::BTreeMap::<String, i64>::new();
                                    for (name, value) in dag
                                        .orders
                                        .get(left.key.orders)
                                        .iter()
                                        .chain(dag.orders.get(right.key.orders).iter())
                                        .chain(vertex.orders.iter())
                                    {
                                        *totals.entry(name.to_uppercase()).or_insert(0) += value;
                                    }
                                    let combined: Orders = totals
                                        .into_iter()
                                        .filter(|(_, value)| *value != 0)
                                        .collect();
                                    let id = dag.orders.intern(combined);
                                    order_cache
                                        .insert((left.key.orders, right.key.orders, row), id);
                                    id
                                }
                            };
                        let within = *limit_cache
                            .entry(orders)
                            .or_insert_with(|| input.orders_within_limits(dag.orders.get(orders)));
                        if !within {
                            continue;
                        }
                        let left_state = dag.colours.get(left.key.colour);
                        let label_key = (left_state.sector_id, left.key.word, right.key.word, row);
                        let ordered = match label_cache.get(&label_key) {
                            Some(cached) => *cached,
                            None => {
                                let labels = engine
                                    .ordered_combination_labels(
                                        left.key.mask,
                                        dag.words.get(left.key.word),
                                        left_state,
                                        right.key.mask,
                                        dag.words.get(right.key.word),
                                        vertex,
                                    )
                                    .map(|labels| dag.words.intern(labels));
                                label_cache.insert(label_key, labels);
                                labels
                            }
                        };
                        let Some(ordered) = ordered else {
                            continue;
                        };
                        let flows = vertex.flows(left.key.chirality, right.key.chirality)?;
                        if flows.is_empty() {
                            continue;
                        }
                        // The merged colour state and open word, deterministic
                        // in their inputs (cached; the oracle recomputes them).
                        let combined_key = (left.key.colour, right.key.colour, row, ordered);
                        let combined = match combine_cache.get(&combined_key) {
                            Some(found) => *found,
                            None => {
                                let found = match engine.combine(
                                    dag.colours.get(left.key.colour),
                                    dag.colours.get(right.key.colour),
                                    vertex,
                                    dag.words.get(ordered),
                                ) {
                                    None => None,
                                    Some(state) => {
                                        // `CurrentIndex.__post_init__`: the open word
                                        // holds every label outside the closed
                                        // components.
                                        let closed = state.closed_labels();
                                        let word = if closed.is_empty() {
                                            ordered
                                        } else {
                                            let open: Vec<u8> = dag
                                                .words
                                                .get(ordered)
                                                .iter()
                                                .copied()
                                                .filter(|label| {
                                                    closed.binary_search(label).is_err()
                                                })
                                                .collect();
                                            let mut all: Vec<u8> =
                                                open.iter().chain(closed.iter()).copied().collect();
                                            all.sort_unstable();
                                            if all != labels {
                                                return invalid(
                                                    "ordered_external_labels and the closed colour \
                                                     components must contain the same labels as \
                                                     external_labels",
                                                );
                                            }
                                            dag.words.intern(open)
                                        };
                                        Some((dag.colours.intern(state), word))
                                    }
                                };
                                combine_cache.insert(combined_key, found);
                                found
                            }
                        };
                        let Some((colour, word)) = combined else {
                            continue;
                        };
                        let result_particle = vertex.particles[2];
                        let result_pdg = vertex.pdgs[2];
                        let flavour_key = (row, left.key.flavour, right.key.flavour);
                        let flavour = match flavour_cache.get(&flavour_key) {
                            Some(flavour) => *flavour,
                            None => {
                                let flavour = match vertex.flavour {
                                    FlavourOp::Constant => dag.flavours.intern(vec![result_pdg]),
                                    FlavourOp::Left | FlavourOp::Right => {
                                        let source = if vertex.flavour == FlavourOp::Left {
                                            left.key.flavour
                                        } else {
                                            right.key.flavour
                                        };
                                        let flow = dag.flavours.get(source);
                                        if flow.last() == Some(&result_pdg) {
                                            source
                                        } else {
                                            let mut flow = flow.clone();
                                            flow.push(result_pdg);
                                            dag.flavours.intern(flow)
                                        }
                                    }
                                };
                                flavour_cache.insert(flavour_key, flavour);
                                flavour
                            }
                        };
                        for flow in flows {
                            if !input.particles[result_particle as usize].allowed(flow.chirality) {
                                continue;
                            }
                            let key = CurrentKey {
                                particle: result_particle,
                                mask,
                                ancestry: left.key.ancestry | right.key.ancestry,
                                chirality: flow.chirality,
                                spin: flow.spin_state,
                                flavour,
                                colour,
                                orders,
                                word,
                            };
                            let result =
                                dag.add_or_get(&mut table, key, false, NO_LEG, NO_HELICITY);
                            let vertex_weight =
                                vertex.colour_weight.clone().map_err(DagError::Invalid)?;
                            let weight =
                                complex_mul(complex_mul([1.0, 0.0], vertex_weight), [1.0, 0.0]);
                            if !interaction_keys.insert((
                                row,
                                left_id,
                                right_id,
                                result,
                                weight_key(weight)?,
                            )) {
                                continue;
                            }
                            // The compile-time evaluation group: kernel
                            // class, canonical inputs, result and coupling.
                            let (inputs, factor) = canonical_kernel_evaluation(
                                &Kernel::of(vertex, vertex.class),
                                left_id,
                                right_id,
                            );
                            let next = compile_groups.len() as i64;
                            let group = *compile_groups
                                .entry((
                                    vertex.class,
                                    inputs.0,
                                    inputs.1,
                                    result_particle,
                                    flow.chirality,
                                    weight_key(flow.coupling)?,
                                ))
                                .or_insert(next);
                            dag.interactions.push(Interaction {
                                vertex: row,
                                left: left_id,
                                right: right_id,
                                result,
                                coupling: flow.coupling,
                                coupling_identity: flow.coupling_identity,
                                colour_weight: weight,
                                group,
                                factor,
                            });
                        }
                    }
                }
            }
        }
    }
    dag.finish = true;
    Ok(dag)
}

/// `_finish_loop_blocks`' prune: keep the sources, the admitted blocks and
/// the sub-DAG feeding them (`prune_dag_to_required_currents`).
pub fn prune_to_admitted(input: &DagInput, dag: &mut Dag) {
    let mut required: Vec<bool> = dag
        .currents
        .iter()
        .map(|current| {
            current.is_source
                || input.admits_block(
                    current.key.mask,
                    current.key.particle,
                    dag.orders.get(current.key.orders),
                )
        })
        .collect();
    let mut kept = vec![false; dag.interactions.len()];
    for (id, interaction) in dag.interactions.iter().enumerate().rev() {
        if required[interaction.result as usize] {
            kept[id] = true;
            required[interaction.left as usize] = true;
            required[interaction.right as usize] = true;
        }
    }
    if required.iter().all(|value| *value) && kept.iter().all(|value| *value) {
        return;
    }
    let mut map = vec![u32::MAX; dag.currents.len()];
    let mut next = 0u32;
    for (id, keep) in required.iter().enumerate() {
        if *keep {
            map[id] = next;
            next += 1;
        }
    }
    dag.currents = dag
        .currents
        .iter()
        .zip(&required)
        .filter(|(_, keep)| **keep)
        .map(|(current, _)| *current)
        .collect();
    dag.interactions = dag
        .interactions
        .iter()
        .zip(&kept)
        .filter(|(_, keep)| **keep)
        .map(|(interaction, _)| Interaction {
            left: map[interaction.left as usize],
            right: map[interaction.right as usize],
            result: map[interaction.result as usize],
            ..*interaction
        })
        .collect();
    dag.sources = dag
        .sources
        .iter()
        .filter(|id| required[**id as usize])
        .map(|id| map[*id as usize])
        .collect();
}
