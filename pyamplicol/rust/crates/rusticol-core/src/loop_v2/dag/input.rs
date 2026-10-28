// SPDX-License-Identifier: 0BSD

//! Model and process tables of the native DAG builder.
//!
//! `loops/v2/dag_input.py` writes them from the Python model and the tree
//! engine's own predicates (the oracle's `ColorEngine`, `allowed_quantum_flows`,
//! `vertices_accepting`, ...), so every table entry is what the Python
//! recursion would compute for the same key. Θ(model) particle and vertex
//! rows, the coupling-order limits, the ignored ids and the loop-aware
//! reachability (admitted leg sets and the block predicate).
//!
//! Floating-point entries travel as their IEEE-754 bit patterns.

use super::{DagError, DagResult, Orders};
use serde::Deserialize;
use std::collections::HashMap;

pub const INPUT_SCHEMA: &str = "pyamplicol.loop-v2.dag-input/1";

/// `(is fermion, orientation, requires the exterior sign)`.
pub type SignContract = (bool, String, bool);
/// `(chirality, spin state, coupling re bits, im bits, coupling identity)`.
type RawFlow = (i8, i32, u64, u64, u32);
/// `(leg label, [(chirality, helicity, spin state)], colour state ids)`.
type RawSources = (u8, Vec<(i8, i32, i32)>, Vec<u32>);

#[derive(Deserialize)]
struct RawLeg {
    label: u8,
    pdg: i64,
    initial: bool,
    statistics: String,
    colour_role: String,
    rep: Option<i32>,
}

#[derive(Deserialize)]
struct RawParticle {
    pdg: i64,
    rep: Option<i32>,
    has_colour: bool,
    fla: bool,
    loop_rep: Option<i32>,
    physical_adjoint: bool,
    u1_singlet: bool,
    auxiliary: Option<String>,
    quantum_numbers: Vec<(String, String)>,
    dimension: [i32; 3],
    allowed: [bool; 3],
    sign_contract: Option<SignContract>,
    sign_contract_error: Option<String>,
    block_rep: Option<i32>,
    block_fermion: Option<bool>,
}

#[derive(Deserialize)]
struct RawVertex {
    kind: i64,
    particles: [i64; 3],
    usable: bool,
    orders: Vec<(String, i64)>,
    colour_weight: Option<[u64; 2]>,
    colour_weight_error: Option<String>,
    absorbing: Option<[bool; 2]>,
    backend: String,
    full_tensor_network_ready: bool,
    class: String,
    factor: [u64; 2],
    swap: bool,
    exchange: Option<[u64; 2]>,
    flows: Vec<(i8, i8, Vec<RawFlow>)>,
    flavour: String,
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
enum RawBlockRule {
    Mask,
    Particles { particles: Vec<i64> },
    Requests { requests: Vec<(u64, i64, Orders)> },
}

#[derive(Deserialize)]
struct RawInput {
    schema: String,
    accuracy: String,
    legs: Vec<RawLeg>,
    particles: Vec<RawParticle>,
    vertices: Vec<RawVertex>,
    accepting: Vec<(i64, i64, Vec<u32>)>,
    right_particles: Vec<(i64, Vec<i64>)>,
    order_limits: Vec<(String, i64)>,
    ignored_particles: Vec<i64>,
    sources: Vec<RawSources>,
    colours: Vec<(i64, Vec<i64>)>,
    admitted_masks: Vec<u64>,
    block_rule: RawBlockRule,
    early_sources_only: bool,
    /// Rank of every interned runtime coupling identity in Python's sort
    /// order (the evaluation keys of the reuse derivation sort by it).
    coupling_identity_ranks: Vec<u32>,
    /// The colour sectors of a process without coloured legs.
    #[serde(default)]
    sectors: Vec<RawSector>,
    /// `models.contracts.U1_SUBTRACTION_AUXILIARY_KIND` (the colour-connection
    /// signs, `signs::apply_connection_signs`).
    u1_subtraction_kind: String,
    /// The N_c grade split of leading- and next-to-leading-colour requests
    /// ([`super::grades`]); absent (false) at full colour.
    #[serde(default)]
    u1_grades: bool,
}

#[derive(Deserialize)]
struct RawSector {
    id: i64,
    words: Vec<Vec<u8>>,
    singlet_labels: Vec<u8>,
}

/// A colour sector as `ColorEngine.ordered_combination_labels` reads it:
/// its intermediate words (`_sector_intermediate_order_words`) and singlet
/// labels.
pub struct Sector {
    pub words: Vec<Vec<u8>>,
    pub singlet_labels: u64,
}

/// Flavour-flow operation of a vertex (`Model.combine_flavour_flow`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlavourOp {
    /// `(result,)`.
    Constant,
    /// Append the result to the left (right) input's flow unless it ends
    /// with it already.
    Left,
    Right,
}

/// One quantum flow of a vertex for given input chiralities.
#[derive(Clone, Copy, Debug)]
pub struct Flow {
    pub chirality: i8,
    pub spin_state: i32,
    pub coupling: [f64; 2],
    /// Interned `_runtime_coupling_identity` (equal ids iff equal identities).
    pub coupling_identity: u32,
}

pub struct Particle {
    pub pdg: i64,
    pub rep: Option<i32>,
    pub has_colour: bool,
    pub fla: bool,
    pub loop_rep: Option<i32>,
    pub physical_adjoint: bool,
    pub u1_singlet: bool,
    pub auxiliary: Option<String>,
    pub quantum_numbers: Vec<(String, String)>,
    pub dimension: [i32; 3],
    pub allowed: [bool; 3],
    /// `FermionOrderingSign._particle_contract`: (is fermion, orientation,
    /// requires the exterior sign), or the error the oracle raises.
    pub sign_contract: Result<SignContract, String>,
    /// `model.color_rep` / `model.is_fermion` as `_blocks` reads them.
    pub block_rep: Option<i32>,
    pub block_fermion: Option<bool>,
}

impl Particle {
    pub fn dimension(&self, chirality: i8) -> i32 {
        self.dimension[(chirality + 1) as usize]
    }

    pub fn allowed(&self, chirality: i8) -> bool {
        self.allowed[(chirality + 1) as usize]
    }
}

pub struct Vertex {
    pub kind: i64,
    /// Particle rows of the two inputs and the result.
    pub particles: [u32; 3],
    pub pdgs: [i64; 3],
    pub usable: bool,
    pub orders: Vec<(String, i64)>,
    pub colour_weight: Result<[f64; 2], String>,
    pub absorbing: Option<[bool; 2]>,
    pub backend: String,
    pub full_tensor_network_ready: bool,
    /// Interned evaluation class id.
    pub class: u32,
    pub factor: [f64; 2],
    pub swap: bool,
    pub exchange: Option<[f64; 2]>,
    flows: HashMap<(i8, i8), Vec<Flow>>,
    pub flavour: FlavourOp,
}

impl Vertex {
    pub fn flows(&self, left: i8, right: i8) -> DagResult<&[Flow]> {
        self.flows
            .get(&(left, right))
            .map(Vec::as_slice)
            .ok_or_else(|| {
                DagError::Malformed(format!(
                    "vertex {} has no quantum-flow entry for input chiralities ({left}, {right})",
                    self.kind
                ))
            })
    }
}

pub struct Leg {
    pub label: u8,
    pub pdg: i64,
    pub particle: u32,
    pub initial: bool,
    /// The source slot whose exterior sign the colour-flow convention owns
    /// (colour-singlet fermions), else `None`; the error the oracle raises
    /// for an unsupported fermion colour role.
    pub sign_rank: Result<Option<usize>, String>,
    pub rep: Option<i32>,
}

pub struct SourceState {
    pub chirality: i8,
    pub helicity: i32,
    pub spin_state: i32,
}

pub struct LegSources {
    pub label: u8,
    pub states: Vec<SourceState>,
    /// `(sector_id, line_groups)` of each source colour state.
    pub colours: Vec<(i64, Vec<i64>)>,
}

pub enum BlockRule {
    Mask,
    Particles(Vec<bool>),
    Requests(HashMap<(u64, u32), Vec<Orders>>),
}

/// Decoded and indexed builder input.
pub struct DagInput {
    pub accuracy: String,
    pub legs: Vec<Leg>,
    pub full_mask: u64,
    pub particles: Vec<Particle>,
    pub vertices: Vec<Vertex>,
    pub classes: Vec<String>,
    by_pdg: HashMap<i64, u32>,
    by_input: HashMap<(u32, u32), Vec<u32>>,
    right_particles: HashMap<u32, Vec<u32>>,
    pub order_limits: Vec<(String, i64)>,
    pub ignored_particles: Vec<i64>,
    pub sources: Vec<LegSources>,
    pub admitted_masks: std::collections::HashSet<u64>,
    pub block_rule: BlockRule,
    pub early_sources_only: bool,
    /// Python-order rank of each runtime coupling identity id.
    pub identity_ranks: Vec<u32>,
    /// Colour sectors by id (a process without coloured legs only).
    pub sectors: HashMap<i64, Sector>,
    /// Representation of every leg label (`None`: unknown to the model).
    pub label_rep: [Option<i32>; 256],
    pub coloured_labels: u64,
    pub singlet_labels: u64,
    pub fundamental_labels: u64,
    pub antifundamental_labels: u64,
    /// The auxiliary kind of the U(1)-subtraction currents.
    pub u1_subtraction_kind: String,
    /// Whether blocks of several N_c grades are graded ([`super::grades`]).
    pub u1_grades: bool,
}

fn bits(pair: [u64; 2]) -> [f64; 2] {
    [f64::from_bits(pair[0]), f64::from_bits(pair[1])]
}

fn label_bit(label: u8) -> u64 {
    1u64 << (label - 1)
}

impl DagInput {
    pub fn from_json(text: &str) -> DagResult<Self> {
        let raw: RawInput = serde_json::from_str(text)
            .map_err(|error| DagError::Malformed(format!("input tables: {error}")))?;
        if raw.schema != INPUT_SCHEMA {
            return Err(DagError::Malformed(format!(
                "input schema {:?}",
                raw.schema
            )));
        }
        let mut by_pdg = HashMap::new();
        let mut particles = Vec::with_capacity(raw.particles.len());
        for p in raw.particles {
            if by_pdg.insert(p.pdg, particles.len() as u32).is_some() {
                return Err(DagError::Malformed(format!("particle {} repeated", p.pdg)));
            }
            particles.push(Particle {
                pdg: p.pdg,
                rep: p.rep,
                has_colour: p.has_colour,
                fla: p.fla,
                loop_rep: p.loop_rep,
                physical_adjoint: p.physical_adjoint,
                u1_singlet: p.u1_singlet,
                auxiliary: p.auxiliary,
                quantum_numbers: p.quantum_numbers,
                dimension: p.dimension,
                allowed: p.allowed,
                sign_contract: match (p.sign_contract, p.sign_contract_error) {
                    (Some(value), None) => Ok(value),
                    (_, Some(error)) => Err(error),
                    (None, None) => Err(format!("particle {} has no sign contract", p.pdg)),
                },
                block_rep: p.block_rep,
                block_fermion: p.block_fermion,
            });
        }
        let particle = |pdg: i64| -> DagResult<u32> {
            by_pdg
                .get(&pdg)
                .copied()
                .ok_or_else(|| DagError::Malformed(format!("particle {pdg} not in the tables")))
        };
        let mut classes: Vec<String> = Vec::new();
        let mut class_ids: HashMap<String, u32> = HashMap::new();
        let mut vertices = Vec::with_capacity(raw.vertices.len());
        let mut kinds = std::collections::HashSet::new();
        for v in raw.vertices {
            if !kinds.insert(v.kind) {
                return Err(DagError::Unsupported(format!(
                    "two vertex rows share kind {} (the oracle keys interactions by kind)",
                    v.kind
                )));
            }
            let class = *class_ids.entry(v.class.clone()).or_insert_with(|| {
                classes.push(v.class.clone());
                (classes.len() - 1) as u32
            });
            let mut flows = HashMap::new();
            for (left, right, rows) in v.flows {
                flows.insert(
                    (left, right),
                    rows.into_iter()
                        .map(|(chirality, spin_state, re, im, coupling_identity)| Flow {
                            chirality,
                            spin_state,
                            coupling: bits([re, im]),
                            coupling_identity,
                        })
                        .collect(),
                );
            }
            vertices.push(Vertex {
                kind: v.kind,
                particles: [
                    particle(v.particles[0])?,
                    particle(v.particles[1])?,
                    particle(v.particles[2])?,
                ],
                pdgs: v.particles,
                usable: v.usable,
                orders: v.orders,
                colour_weight: match (v.colour_weight, v.colour_weight_error) {
                    (Some(value), None) => Ok(bits(value)),
                    (_, Some(error)) => Err(error),
                    (None, None) => Err(format!("vertex {} has no colour weight", v.kind)),
                },
                absorbing: v.absorbing,
                backend: v.backend,
                full_tensor_network_ready: v.full_tensor_network_ready,
                class,
                factor: bits(v.factor),
                swap: v.swap,
                exchange: v.exchange.map(bits),
                flows,
                flavour: match v.flavour.as_str() {
                    "constant" => FlavourOp::Constant,
                    "left" => FlavourOp::Left,
                    "right" => FlavourOp::Right,
                    other => {
                        return Err(DagError::Malformed(format!("flavour operation {other:?}")));
                    }
                },
            });
        }
        let mut by_input = HashMap::new();
        for (left, right, rows) in raw.accepting {
            if rows.iter().any(|row| *row as usize >= vertices.len()) {
                return Err(DagError::Malformed("vertex row out of range".into()));
            }
            by_input.insert((particle(left)?, particle(right)?), rows);
        }
        let mut right_particles = HashMap::new();
        for (left, rights) in raw.right_particles {
            right_particles.insert(
                particle(left)?,
                rights
                    .into_iter()
                    .map(particle)
                    .collect::<DagResult<Vec<_>>>()?,
            );
        }
        let mut label_rep = [None; 256];
        let mut legs = Vec::with_capacity(raw.legs.len());
        let mut full_mask = 0u64;
        for (slot, leg) in raw.legs.into_iter().enumerate() {
            if leg.label == 0 || leg.label > 64 {
                return Err(DagError::Unsupported(format!(
                    "leg label {} (masks hold at most 64 legs)",
                    leg.label
                )));
            }
            full_mask |= label_bit(leg.label);
            label_rep[leg.label as usize] = leg.rep;
            let sign_rank = if leg.statistics != "fermion" {
                Ok(None)
            } else {
                match leg.colour_role.as_str() {
                    "singlet" => Ok(Some(slot)),
                    "fundamental" | "antifundamental" => Ok(None),
                    other => Err(format!(
                        "external-fermion ordering does not support fermion color role '{other}'"
                    )),
                }
            };
            legs.push(Leg {
                label: leg.label,
                pdg: leg.pdg,
                particle: particle(leg.pdg)?,
                initial: leg.initial,
                sign_rank,
                rep: leg.rep,
            });
        }
        let mut coloured_labels = 0;
        let mut singlet_labels = 0;
        let mut fundamental_labels = 0;
        let mut antifundamental_labels = 0;
        for leg in &legs {
            let bit = label_bit(leg.label);
            match leg.rep {
                Some(1) => singlet_labels |= bit,
                Some(rep) => {
                    coloured_labels |= bit;
                    if rep == 3 {
                        fundamental_labels |= bit;
                    } else if rep == -3 {
                        antifundamental_labels |= bit;
                    }
                }
                None => {}
            }
        }
        let sources = raw
            .sources
            .into_iter()
            .map(|(label, states, colours)| {
                Ok(LegSources {
                    label,
                    states: states
                        .into_iter()
                        .map(|(chirality, helicity, spin_state)| SourceState {
                            chirality,
                            helicity,
                            spin_state,
                        })
                        .collect(),
                    colours: colours
                        .into_iter()
                        .map(|id| {
                            raw.colours.get(id as usize).cloned().ok_or_else(|| {
                                DagError::Malformed("source colour out of range".into())
                            })
                        })
                        .collect::<DagResult<_>>()?,
                })
            })
            .collect::<DagResult<Vec<_>>>()?;
        let block_rule = match raw.block_rule {
            RawBlockRule::Mask => BlockRule::Mask,
            RawBlockRule::Particles {
                particles: admitted,
            } => {
                let mut table = vec![false; particles.len()];
                for pdg in admitted {
                    if let Some(id) = by_pdg.get(&pdg) {
                        table[*id as usize] = true;
                    }
                }
                BlockRule::Particles(table)
            }
            RawBlockRule::Requests { requests } => {
                let mut table: HashMap<(u64, u32), Vec<Orders>> = HashMap::new();
                for (mask, pdg, orders) in requests {
                    if let Some(id) = by_pdg.get(&pdg) {
                        table.entry((mask, *id)).or_default().push(orders);
                    }
                }
                BlockRule::Requests(table)
            }
        };
        let identities = raw.coupling_identity_ranks.len();
        if vertices.iter().any(|vertex| {
            vertex
                .flows
                .values()
                .flatten()
                .any(|flow| flow.coupling_identity as usize >= identities)
        }) {
            return Err(DagError::Malformed(
                "coupling identity without a rank".into(),
            ));
        }
        Ok(DagInput {
            accuracy: raw.accuracy,
            legs,
            full_mask,
            particles,
            vertices,
            classes,
            by_pdg,
            by_input,
            right_particles,
            order_limits: raw.order_limits,
            ignored_particles: raw.ignored_particles,
            sources,
            admitted_masks: raw.admitted_masks.into_iter().collect(),
            block_rule,
            early_sources_only: raw.early_sources_only,
            identity_ranks: raw.coupling_identity_ranks,
            sectors: raw
                .sectors
                .into_iter()
                .map(|sector| {
                    let mut singlet_labels = 0u64;
                    for label in sector.singlet_labels {
                        if label == 0 || label > 64 {
                            return Err(DagError::Malformed(format!("sector label {label}")));
                        }
                        singlet_labels |= label_bit(label);
                    }
                    Ok((
                        sector.id,
                        Sector {
                            words: sector.words,
                            singlet_labels,
                        },
                    ))
                })
                .collect::<DagResult<_>>()?,
            label_rep,
            coloured_labels,
            singlet_labels,
            fundamental_labels,
            antifundamental_labels,
            u1_subtraction_kind: raw.u1_subtraction_kind,
            u1_grades: raw.u1_grades,
        })
    }

    pub fn particle_id(&self, pdg: i64) -> Option<u32> {
        self.by_pdg.get(&pdg).copied()
    }

    /// `Model.vertices_accepting(left, right)`, in model order.
    pub fn vertices_accepting(&self, left: u32, right: u32) -> &[u32] {
        self.by_input
            .get(&(left, right))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// `_right_particles_by_left(model)[left]`, sorted by particle id.
    pub fn right_particles(&self, left: u32) -> &[u32] {
        self.right_particles
            .get(&left)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// `LoopBlockReachability.admits_block` on (mask, particle, orders).
    pub fn admits_block(&self, mask: u64, particle: u32, orders: &Orders) -> bool {
        match &self.block_rule {
            BlockRule::Mask => self.admitted_masks.contains(&mask),
            BlockRule::Particles(table) => {
                self.admitted_masks.contains(&mask) && table[particle as usize]
            }
            BlockRule::Requests(table) => table
                .get(&(mask, particle))
                .is_some_and(|rows| rows.iter().any(|row| row == orders)),
        }
    }

    /// `_coupling_orders_within_limits(orders, limits)`.
    pub fn orders_within_limits(&self, orders: &Orders) -> bool {
        self.order_limits.iter().all(|(name, limit)| {
            orders
                .iter()
                .find(|(other, _)| other == name)
                .map_or(0, |(_, value)| *value)
                <= *limit
        })
    }
}
