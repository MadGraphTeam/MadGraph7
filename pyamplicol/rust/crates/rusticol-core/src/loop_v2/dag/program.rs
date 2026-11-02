// SPDX-License-Identifier: 0BSD

//! The interaction program of a loop-request DAG (A5, R2: straight from the
//! columns, no decode).
//!
//! The Python oracle is the native lane's export of a DAG's tree rows:
//! `assemble.TreeRows` builds the runtime schema of the DAG
//! (`generation/runtime_schema.build_runtime_schema_layout`) and
//! `loops/v2/native_trees._interaction_program` walks its stages. For a
//! loop-request DAG (no amplitude roots) whose every non-source current is
//! propagated (a CT program's tree rows, `CountertermProgram.rows`:
//! `propagated_currents = ct_closure.program_propagated_currents(dag)`) the
//! layout is as below. The slice's `TreeRows(dag)` is not covered: there a
//! current that feeds no interaction gets an `unpropagated` slot.
//!
//! * one value slot per current, in current order, each `dimension`
//!   components wide: `source` for a source, otherwise `propagated` where
//!   the model's propagator of `(particle id, chirality)` applies and
//!   `unpropagated` where it does not; a current's value slot id is its id
//!   and its value components are its current components;
//! * one momentum slot per distinct momentum mask, sorted by (bit count,
//!   mask), four components each after the values;
//! * one `current-combine` stage per result-current size (the bit count of
//!   its external mask), in increasing size, holding the interactions into
//!   currents of that size in DAG order.
//!
//! Per stage the program sums every interaction with a non-zero evaluation
//! factor into its result current's accumulator (the stage's result
//! currents in increasing id) and writes each accumulator to its value slot,
//! through the propagator kernel of a propagated slot. Kernels are numbered
//! per kind in first use, as the oracle compiles them; [`Program::kernels`]
//! lists their keys in that interleaved order so that Python compiles each
//! once, from the representative the oracle builds it from (the model's
//! component expressions stay Symbolica's, rule 4). Colour weights are
//! interned in first use; Python writes their exact texts.

use super::{ColumnarDag, DagError, DagResult, invalid};
use serde::Deserialize;
use std::collections::HashMap;

pub const PROGRAM_INPUT_SCHEMA: &str = "pyamplicol.loop-v2.dag-program-input/2";

#[derive(Deserialize)]
struct RawProgramInput {
    schema: String,
    /// `(particle id, chirality, applies_propagator)` of the model's
    /// propagator of every `(particle id, chirality)` the DAG's currents
    /// carry (`Model._propagator_ir`).
    propagators: Vec<(i64, i8, bool)>,
    /// A CT program's slot table: the pseudo-order of each slot, in slot
    /// order (`native_trees.counterterm_slots`); empty for other programs.
    #[serde(default)]
    slot_orders: Vec<String>,
}

/// The model data of a program (`dag_input.program_input`).
pub struct ProgramInput {
    propagators: HashMap<(i64, i8), bool>,
    slot_orders: Vec<String>,
}

impl ProgramInput {
    pub fn from_json(text: &str) -> DagResult<Self> {
        let raw: RawProgramInput = serde_json::from_str(text)
            .map_err(|error| DagError::Malformed(format!("program tables: {error}")))?;
        if raw.schema != PROGRAM_INPUT_SCHEMA {
            return Err(DagError::Malformed(format!(
                "program schema {:?}",
                raw.schema
            )));
        }
        Ok(Self {
            propagators: raw
                .propagators
                .into_iter()
                .map(|(pdg, chirality, applies)| ((pdg, chirality), applies))
                .collect(),
            slot_orders: raw.slot_orders,
        })
    }
}

/// A value slot's variant (`runtime_schema._value_slots`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Source,
    Unpropagated,
    Propagated,
}

impl Variant {
    pub fn name(self) -> &'static str {
        match self {
            Variant::Source => "source",
            Variant::Unpropagated => "unpropagated",
            Variant::Propagated => "propagated",
        }
    }
}

/// A kernel in its first use: a vertex kernel
/// `(vertex kind, result pdg, result chirality, left chirality, right
/// chirality, left dimension, right dimension)` and the interaction the
/// oracle builds it from, or a propagator kernel
/// `(pdg, chirality, dimension)` and the current it is built for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernelUse {
    Vertex { key: [i64; 7], interaction: u32 },
    Propagator { key: [i64; 3], current: u32 },
}

/// One row of a step: `[vertex kernel, left start, left dimension, right
/// start, right dimension, left momentum start, right momentum start,
/// accumulator, colour weight]`, the weight an index into
/// [`Program::weights`].
pub type Row = [i64; 9];

/// One output of a step: `[accumulator, propagator kernel or -1, momentum
/// start, value start, dimension]`.
pub type Output = [i64; 5];

/// One stage of the program (`_interaction_program`'s step).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Step {
    pub rows: Vec<Row>,
    /// Dimension of each accumulator.
    pub accumulators: Vec<i64>,
    pub outputs: Vec<Output>,
    /// A CT program's per-accumulator slot set (`native_trees.slot_masks`:
    /// 0 every slot, `1 + s` slot `s` alone); empty for other programs.
    pub accumulator_masks: Vec<u32>,
}

/// The interaction program and the schema layout the export reads.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Program {
    /// Value components (`parameter_layout.value_component_count`).
    pub value_count: u64,
    /// Start of each current's value slot (its value slot id is its id).
    pub starts: Vec<u64>,
    pub variants: Vec<Variant>,
    /// The momentum slots' masks, in slot order.
    pub momentum_masks: Vec<u64>,
    pub kernels: Vec<KernelUse>,
    /// Distinct colour weights in first use (`-0.0` read as `0.0`: equal
    /// weights have equal exact texts).
    pub weights: Vec<[f64; 2]>,
    pub steps: Vec<Step>,
    /// One interaction per distinct (vertex, coupling bits), in first
    /// occurrence: the inputs of the runtime model-parameter records
    /// (`physics_metadata._used_coupling_parameter_names`).
    pub signatures: Vec<u32>,
    /// Distinct particle ids of the currents, sorted.
    pub particles: Vec<i64>,
}

fn popcount(mask: u64) -> u32 {
    mask.count_ones()
}

impl ColumnarDag {
    /// Distinct `(particle id, chirality)` of the currents, sorted (the
    /// propagators `program_input` asks the model for).
    pub fn current_species(&self) -> Vec<(i64, i8)> {
        let mut out: Vec<(i64, i8)> = (0..self.current_count())
            .map(|id| {
                (
                    self.tables.species[self.c_species[id] as usize].pdg,
                    self.c_chirality[id],
                )
            })
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The interaction program of the DAG (module docs).
    pub fn program(&self, input: &ProgramInput) -> DagResult<Program> {
        let n = self.current_count();
        let t = &self.tables;
        let pdg = |id: usize| t.species[self.c_species[id] as usize].pdg;
        // Value slots: one per current.
        let mut starts = Vec::with_capacity(n);
        let mut variants = Vec::with_capacity(n);
        let mut offset = 0u64;
        for id in 0..n {
            let key = (pdg(id), self.c_chirality[id]);
            let applies = *input.propagators.get(&key).ok_or_else(|| {
                DagError::Malformed(format!(
                    "no propagator for particle {} chirality {}",
                    key.0, key.1
                ))
            })?;
            let variant = if self.c_source[id] != 0 {
                Variant::Source
            } else if applies {
                Variant::Propagated
            } else {
                Variant::Unpropagated
            };
            let dimension = u64::try_from(self.c_dimension[id]).map_err(|_| {
                DagError::Malformed(format!("current {id} has a negative dimension"))
            })?;
            starts.push(offset);
            variants.push(variant);
            offset += dimension;
        }
        let value_count = offset;
        // Momentum slots.
        let mut masks: Vec<u64> = self.c_momentum.clone();
        masks.sort_unstable_by_key(|mask| (popcount(*mask), *mask));
        masks.dedup();
        let slot_of: HashMap<u64, i64> = masks
            .iter()
            .enumerate()
            .map(|(slot, mask)| (*mask, slot as i64))
            .collect();
        let momentum = |id: usize| value_count as i64 + 4 * slot_of[&self.c_momentum[id]];
        // Per-slot reachability (`native_trees.slot_masks`): the slot set
        // of a current's order tuple, 0 (every slot) without an insertion;
        // a current with insertions of two slots is refused. Per order
        // tuple once, for the tuples of the program's result currents only
        // (the order table may hold others).
        let slots = !input.slot_orders.is_empty();
        let mut order_masks: Vec<Option<u32>> = vec![None; t.orders.len()];
        let mut mask = |id: usize| -> DagResult<u32> {
            let index = self.c_orders[id] as usize;
            if let Some(found) = order_masks[index] {
                return Ok(found);
            }
            let orders = &t.orders[index];
            let mut found: Vec<usize> = orders
                .iter()
                .filter(|(_, power)| *power != 0)
                .filter_map(|(name, _)| input.slot_orders.iter().position(|slot| slot == name))
                .collect();
            found.sort_unstable();
            found.dedup();
            let value = match found.as_slice() {
                [] => 0,
                [slot] => 1 + *slot as u32,
                _ => {
                    return invalid(format!(
                        "a counterterm program current carries two slot orders: {orders:?}"
                    ));
                }
            };
            order_masks[index] = Some(value);
            Ok(value)
        };
        // Stages by result size, interactions in DAG order.
        let mut by_size: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
        for (k, result) in self.i_result.iter().enumerate() {
            by_size
                .entry(popcount(self.c_mask[*result as usize]))
                .or_default()
                .push(k as u32);
        }
        let mut kernels = Vec::new();
        let mut vertex_ids: HashMap<[i64; 7], i64> = HashMap::new();
        let mut propagator_ids: HashMap<[i64; 3], i64> = HashMap::new();
        let mut weight_ids: HashMap<[u64; 2], i64> = HashMap::new();
        let mut weights = Vec::new();
        let mut steps = Vec::with_capacity(by_size.len());
        let mut accumulator_of = vec![u32::MAX; n];
        for interactions in by_size.values() {
            let mut results: Vec<u32> = interactions
                .iter()
                .map(|k| self.i_result[*k as usize])
                .collect();
            results.sort_unstable();
            results.dedup();
            for (index, current) in results.iter().enumerate() {
                accumulator_of[*current as usize] = index as u32;
            }
            let mut step = Step {
                accumulators: results
                    .iter()
                    .map(|current| self.c_dimension[*current as usize] as i64)
                    .collect(),
                ..Default::default()
            };
            for k in interactions {
                let k = *k as usize;
                let factor = self.i_factor[k];
                if factor[0] == 0.0 && factor[1] == 0.0 {
                    continue; // a certified zero attachment (as the stage program)
                }
                let (left, right, result) = (
                    self.i_left[k] as usize,
                    self.i_right[k] as usize,
                    self.i_result[k] as usize,
                );
                let key = [
                    t.vertices[self.i_vertex[k] as usize].kind,
                    pdg(result),
                    self.c_chirality[result] as i64,
                    self.c_chirality[left] as i64,
                    self.c_chirality[right] as i64,
                    self.c_dimension[left] as i64,
                    self.c_dimension[right] as i64,
                ];
                let next = vertex_ids.len() as i64;
                let kernel = *vertex_ids.entry(key).or_insert_with(|| {
                    kernels.push(KernelUse::Vertex {
                        key,
                        interaction: k as u32,
                    });
                    next
                });
                let weight = self.i_colour_weight[k];
                let bits = [(weight[0] + 0.0).to_bits(), (weight[1] + 0.0).to_bits()];
                let next = weights.len() as i64;
                let weight = *weight_ids.entry(bits).or_insert_with(|| {
                    weights.push([weight[0] + 0.0, weight[1] + 0.0]);
                    next
                });
                step.rows.push([
                    kernel,
                    starts[left] as i64,
                    self.c_dimension[left] as i64,
                    starts[right] as i64,
                    self.c_dimension[right] as i64,
                    momentum(left),
                    momentum(right),
                    accumulator_of[result] as i64,
                    weight,
                ]);
            }
            for (index, current) in results.iter().enumerate() {
                let current = *current as usize;
                if slots {
                    step.accumulator_masks.push(mask(current)?);
                }
                let propagator = match variants[current] {
                    Variant::Source => {
                        return invalid("a stage output of variant 'source'");
                    }
                    Variant::Unpropagated => -1,
                    Variant::Propagated => {
                        let key = [
                            pdg(current),
                            self.c_chirality[current] as i64,
                            self.c_dimension[current] as i64,
                        ];
                        let next = propagator_ids.len() as i64;
                        *propagator_ids.entry(key).or_insert_with(|| {
                            kernels.push(KernelUse::Propagator {
                                key,
                                current: current as u32,
                            });
                            next
                        })
                    }
                };
                step.outputs.push([
                    index as i64,
                    propagator,
                    momentum(current),
                    starts[current] as i64,
                    self.c_dimension[current] as i64,
                ]);
            }
            steps.push(step);
        }
        let mut seen: HashMap<(u32, [u64; 2]), ()> = HashMap::new();
        let mut signatures = Vec::new();
        for k in 0..self.interaction_count() {
            let coupling = self.i_coupling[k];
            let key = (
                self.i_vertex[k],
                [coupling[0].to_bits(), coupling[1].to_bits()],
            );
            if seen.insert(key, ()).is_none() {
                signatures.push(k as u32);
            }
        }
        let mut particles: Vec<i64> = (0..n).map(pdg).collect();
        particles.sort_unstable();
        particles.dedup();
        Ok(Program {
            value_count,
            starts,
            variants,
            momentum_masks: masks,
            kernels,
            weights,
            steps,
            signatures,
            particles,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ColourState, NO_HELICITY, NO_LEG, Species, Tables, VertexKey};
    use super::*;

    fn dag() -> ColumnarDag {
        ColumnarDag {
            tables: Tables {
                species: vec![Species {
                    pdg: 21,
                    auxiliary: None,
                    quantum_numbers: vec![],
                }],
                flavours: vec![vec![21]],
                colours: vec![ColourState {
                    accuracy: "full".into(),
                    sector_id: 0,
                    line_groups: vec![],
                    basis_key: vec![],
                    traces: vec![],
                    lines: vec![],
                    nc_grade: None,
                }],
                orders: vec![vec![], vec![("QCD".into(), 1)], vec![("CT1".into(), 1)]],
                vertices: vec![VertexKey {
                    kind: 7,
                    particles: [21, 21, 21],
                    backend: "spenso-ufo".into(),
                    full_tensor_network_ready: true,
                }],
            },
            c_species: vec![0; 5],
            c_mask: vec![1, 2, 4, 3, 7],
            c_momentum: vec![1, 2, 4, 3, 7],
            c_ancestry: vec![0; 5],
            c_chirality: vec![0; 5],
            c_spin: vec![0; 5],
            c_flavour: vec![0; 5],
            c_colour: vec![0; 5],
            c_orders: vec![0, 0, 0, 1, 2],
            c_dimension: vec![4; 5],
            c_source: vec![1, 1, 1, 0, 0],
            c_leg: vec![1, 2, 3, NO_LEG, NO_LEG],
            c_helicity: vec![1, 1, 1, NO_HELICITY, NO_HELICITY],
            c_word_offsets: vec![0, 1, 2, 3, 5, 8],
            c_words: vec![1, 2, 3, 1, 2, 1, 2, 3],
            sources: vec![0, 1, 2],
            i_vertex: vec![0, 0, 0],
            i_left: vec![0, 3, 0],
            i_right: vec![1, 2, 2],
            i_result: vec![3, 4, 4],
            i_coupling: vec![[1.0, 0.0]; 3],
            i_colour_weight: vec![[-0.0, 1.0], [0.0, 1.0], [0.5, 0.0]],
            i_group: vec![-1; 3],
            i_factor: vec![[1.0, 0.0], [1.0, 0.0], [-0.0, 0.0]],
        }
    }

    #[test]
    fn layout_follows_the_runtime_schema() {
        let input = ProgramInput::from_json(&format!(
            r#"{{"schema":"{PROGRAM_INPUT_SCHEMA}","propagators":[[21,0,true]],"slot_orders":["CT0","CT1"]}}"#
        ))
        .expect("input");
        let program = dag().program(&input).expect("program");
        assert_eq!(program.value_count, 20);
        assert_eq!(program.starts, vec![0, 4, 8, 12, 16]);
        assert_eq!(program.momentum_masks, vec![1, 2, 4, 3, 7]);
        assert_eq!(program.steps.len(), 2);
        // The zero-factor interaction is skipped; -0.0 and 0.0 are one weight.
        assert_eq!(program.steps[1].rows.len(), 1);
        assert_eq!(program.weights, vec![[0.0, 1.0]]);
        assert_eq!(program.steps[0].rows[0], [0, 0, 4, 4, 4, 20, 24, 0, 0]);
        assert_eq!(program.steps[1].outputs, vec![[0, 0, 36, 16, 4]]);
        // Current 3 (no insertion) reaches every slot, current 4 slot 1.
        assert_eq!(program.steps[0].accumulator_masks, vec![0]);
        assert_eq!(program.steps[1].accumulator_masks, vec![2]);
        assert_eq!(
            program.kernels,
            vec![
                KernelUse::Vertex {
                    key: [7, 21, 0, 0, 0, 4, 4],
                    interaction: 0
                },
                KernelUse::Propagator {
                    key: [21, 0, 4],
                    current: 3
                }
            ]
        );
        assert_eq!(program.signatures, vec![0]);
        assert_eq!(program.particles, vec![21]);
    }

    #[test]
    fn a_missing_propagator_is_refused() {
        let input = ProgramInput::from_json(&format!(
            r#"{{"schema":"{PROGRAM_INPUT_SCHEMA}","propagators":[]}}"#
        ))
        .expect("input");
        assert!(dag().program(&input).is_err());
    }
}
