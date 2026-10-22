// SPDX-License-Identifier: 0BSD

//! Plan v4: the generated schedule program of the v2 executor (DESIGN.md
//! §3.8), as exported by `pyamplicol.loops.v2.plan_export`.
//!
//! A plan holds no numerical value. It lists
//!
//! - **kernels**: catalogue segment kernels as exact polynomial terms over
//!   their inputs, with numbered model constants valued per batch;
//! - **contents**: per position content its loop-line dimensions, q-degree
//!   and fused entries `(β, β', q^a)`, and per routing term the kernel
//!   bindings that build the fused factor `X(J)` of each content row from
//!   block-current rows, block momenta and the frame shift;
//! - **factors**: the (content, frame shift) pairs whose `X(J)` steps read;
//! - **seeds**: exact colour weights `K[c, b]` per seed key and observable
//!   column, contracted with the Born components at run time;
//! - **families**: frame reading, rank and loop orders, the leaves and the
//!   schedule-B steps (edges, row links, trace targets);
//! - **forward**: the schedule-F prefix tree, an oracle of B.
//!
//! [`LoopProgramV4::load`] parses and validates the JSON and derives the
//! executor tables: segment layouts per content (open-loop and seeded
//! forms), product tables per (input rank, degree), per-family factor sets
//! and row links expanded over the observable columns.

use super::poly::{MonomialProductTable, SegmentEntry, SegmentLayout};
use crate::{RusticolError, RusticolResult};
use ninjars::fast::tensor_layout::ncoeff;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use symbolica::domains::rational::Rational;

/// The plan schema version this executor runs.
pub const PLAN_SCHEMA: u32 = 4;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::compatibility(format!("loop v2 plan: {}", message.into()))
}

fn require(condition: bool, message: &str) -> RusticolResult<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}

// ------------------------------------------------------------------ file form

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PlanFile {
    schema: u32,
    legs: usize,
    rows: usize,
    columns: usize,
    average: u64,
    born_components: usize,
    constants: usize,
    masses: usize,
    block_rows: Vec<usize>,
    kernels: Vec<KernelFile>,
    contents: Vec<ContentFile>,
    factors: Vec<(usize, u64)>,
    seeds: Vec<SeedFile>,
    families: Vec<FamilyFile>,
    forward: Option<ForwardFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KernelFile {
    inputs: Vec<(String, usize, usize)>,
    outputs: Vec<Vec<(usize, Vec<(usize, u32)>)>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContentFile {
    d_in: usize,
    d_out: usize,
    degree: usize,
    rows: usize,
    fused: Vec<[u32; 6]>,
    terms: Vec<TermFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TermFile {
    slots: Vec<u64>,
    dimensions: Vec<usize>,
    bindings: Vec<BindingFile>,
    vertex: i64,
    split: Vec<SplitFile>,
    rows: Vec<(usize, Vec<usize>)>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SplitFile {
    kernel: usize,
    inputs: Vec<(String, usize, usize)>,
    entries: Vec<(usize, usize)>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BindingFile {
    kernel: usize,
    inputs: Vec<(String, usize, usize)>,
    entries: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SeedFile {
    columns: Vec<Vec<(usize, String, String)>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FamilyFile {
    sequence: Vec<(u64, usize)>,
    rank: usize,
    orders: Vec<usize>,
    leaves: Vec<LeafFile>,
    steps: Vec<StepFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LeafFile {
    seed: usize,
    weight: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StepFile {
    seeded: bool,
    trace: i64,
    rows: usize,
    d_alpha: usize,
    d_out: usize,
    degree: usize,
    rank_in: usize,
    edges: Vec<EdgeFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EdgeFile {
    leaf: i64,
    child: i64,
    factor: usize,
    links: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ForwardFile {
    states: Vec<StateFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StateFile {
    parent: i64,
    factor: usize,
    seeded: bool,
    rows: usize,
    d_alpha: usize,
    d_out: usize,
    degree: usize,
    rank: usize,
    links: Vec<u32>,
    closures: Vec<ClosureFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ClosureFile {
    family: usize,
    accumulator: usize,
    seed: usize,
    weight: String,
    rows: Vec<i64>,
    relabel: Option<(i8, u64)>,
}

// --------------------------------------------------------------- program form

/// One input of a kernel binding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelInput {
    /// Component `mu` of the loop-line momentum at the frame position.
    Loop(usize),
    /// Component of the block current of a slot.
    Current { slot: usize, component: usize },
    /// Component `mu` of the summed incoming momentum of a slot's block.
    Momentum { slot: usize, mu: usize },
    /// A model parameter, folded into the kernel constants.
    Parameter,
}

/// One term `constant × Π input^power` of a kernel output.
#[derive(Clone, Debug)]
pub struct KernelTerm {
    pub constant: usize,
    pub monomial: Vec<(usize, u32)>,
}

/// A catalogue kernel: per output (fused entry of its binding) its terms.
#[derive(Clone, Debug)]
pub struct Kernel {
    pub inputs: usize,
    pub outputs: Vec<Vec<KernelTerm>>,
}

/// A kernel applied to a routing term's slots: its resolved inputs and the
/// entry (of the fused factor, or of a split vertex) each output adds into.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binding {
    pub kernel: usize,
    pub inputs: Vec<KernelInput>,
    pub entries: Vec<usize>,
}

/// One entry `(β, ν, β', q exponents)` of a split vertex tensor.
pub type VertexEntry = (usize, usize, usize, [u32; 4]);

/// A split-form vertex tensor `V[(β, ν), β']` (DESIGN.md §5.2): the
/// block-independent kernels of every routing term with one vertex key
/// (rule, couplings, legs, chiralities, block masks and loop species), so
/// one evaluation per frame shift serves every content and edge of a step
/// that attaches the same blocks through the same vertex. `ν` runs over the
/// products of the slots' current components, first slot slowest.
#[derive(Clone, Debug)]
pub struct Vertex {
    pub width: usize,
    pub d_in: usize,
    pub d_out: usize,
    pub degree: usize,
    /// Leg mask of each slot's block (the block momenta of the kernels).
    pub slots: Vec<u64>,
    /// Kernels whose outputs add into vertex entries.
    pub bindings: Vec<Binding>,
    pub entries: Vec<VertexEntry>,
    /// `(d_in · width) × d_out` layout for open-loop inputs, loop-in column
    /// `β · width + ν` ([`super::kernels::Segment::vertex_column`]).
    pub open: SegmentLayout,
    /// `width × (d_in · d_out)` layout for seeded inputs (`δ_αβ` seeds).
    pub seeded: SegmentLayout,
}

impl Vertex {
    fn same_kernels(&self, other: &Self) -> bool {
        self.width == other.width
            && self.d_in == other.d_in
            && self.d_out == other.d_out
            && self.slots == other.slots
            && self.entries == other.entries
            && self.bindings == other.bindings
    }
}

/// The split vertex of a routing term and, per vertex entry, the fused
/// entry it feeds (`X[row][fused] += V[entry] J[row][ν]`).
#[derive(Clone, Debug)]
pub struct TermVertex {
    pub id: usize,
    pub fused: Vec<usize>,
}

/// One routing term of a content with fixed block chiralities.
#[derive(Clone, Debug)]
pub struct ContentTerm {
    /// Leg mask of each slot's block.
    pub slots: Vec<u64>,
    /// Current components of each slot.
    pub dimensions: Vec<usize>,
    /// Kernels of `X(J)` itself (block currents among their inputs).
    pub bindings: Vec<Binding>,
    /// The term as a vertex tensor times block components, when its kernels
    /// split; the factor builder then evaluates the vertex once per (content,
    /// shift) and contracts it with each row's block components.
    pub vertex: Option<TermVertex>,
    /// (content row, block row id per slot).
    pub rows: Vec<(usize, Vec<usize>)>,
}

impl ContentTerm {
    /// Block components `nu` of the split form: the product of the slots'.
    pub fn width(&self) -> usize {
        self.dimensions.iter().product()
    }
}

/// One position content: dimensions, q-degree, fused entries and the
/// segment layouts of both step kinds (same entry order).
#[derive(Clone, Debug)]
pub struct Content {
    pub d_in: usize,
    pub d_out: usize,
    pub degree: usize,
    pub rows: usize,
    pub terms: Vec<ContentTerm>,
    /// `(d_in, d_out)` layout for open-loop inputs.
    pub open: SegmentLayout,
    /// `(1, d_in * d_out)` layout for seeded inputs: a seed is `δ_ab` times
    /// a scalar, so the cut index `a` becomes the incoming index `β`.
    pub seeded: SegmentLayout,
}

impl Content {
    /// Lane vectors per content row of the fused factor.
    pub fn factor_row(&self) -> usize {
        self.open.entries().len()
    }
}

/// A seed term `K × conj(A0[component])`, exact.
#[derive(Clone, Debug)]
pub struct SeedTerm {
    pub component: usize,
    pub weight: [Rational; 2],
}

/// Seed key (class, loop orders): per observable column its terms.
#[derive(Clone, Debug)]
pub struct Seed {
    pub columns: Vec<Vec<SeedTerm>>,
}

#[derive(Clone, Debug)]
pub struct Leaf {
    pub seed: usize,
    /// Exact leaf weight (automorphism, flavour multiplicity, fermion sign).
    pub weight: Rational,
}

/// The source of an edge: a leaf seed or the output of an earlier step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Leaf(usize),
    Step(usize),
}

/// One edge of a step: source, fused factor, products table of (source
/// rank, content degree) and row links over object rows `(row, column)`.
#[derive(Clone, Debug)]
pub struct Edge {
    pub source: Source,
    pub factor: usize,
    pub content: usize,
    pub source_rank: usize,
    pub products: usize,
    pub links: Vec<super::kernels::RowLink>,
}

/// The kernel form of a dressing step (DESIGN.md §5.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// `O' += O · X(J)` per edge with the fused factors.
    Fused,
    /// `T += O ⊗ J` per edge, then each vertex tensor once per output row.
    Split,
}

/// How [`LoopProgramV4::load_with`] picks each step's form.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormPolicy {
    /// Gate G0's rule (production default).
    Rule,
    /// Every step fused.
    Fused,
    /// Split wherever a step can split (oracle of the split kernels).
    Split,
}

/// Gate G0's per-form step-time model (DESIGN.md §5.4 item 4): `t =
/// per_iteration · iterations + work · lanes / rate`, fitted on the
/// gg→gggg FC program at 16 lanes in the production code generation
/// (`workspace/tmp/g0-real-tries/run2/costmodel.json`, `selection_rule`).
/// Only the ratio of the two forms' times matters.
struct FormModel {
    per_iteration: f64,
    rate: f64,
}

const FUSED_MODEL: FormModel = FormModel {
    per_iteration: 4.51e-9,
    rate: 1.389e9,
};
const SPLIT_MODEL: FormModel = FormModel {
    per_iteration: 0.0,
    rate: 1.276e9,
};
const RULE_LANES: f64 = 16.0;

/// Kernel work of one step in one form: complex multiply-adds per lane and
/// the kernels' inner-loop iterations (the work without the factor of input
/// coefficients), counted as the G0 bench counts them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepCost {
    pub work: u64,
    pub iterations: u64,
}

impl FormModel {
    fn seconds(&self, cost: StepCost) -> f64 {
        self.per_iteration * cost.iterations as f64 + cost.work as f64 * RULE_LANES / self.rate
    }
}

/// The split form of a step: its vertex groups and, per edge, one part per
/// routing term of the edge's content.
#[derive(Clone, Debug)]
pub struct SplitStep {
    pub groups: Vec<SplitGroup>,
    pub parts: Vec<Vec<SplitPart>>,
}

/// One vertex tensor of a split step: the edges whose terms share it
/// accumulate `T = Σ O ⊗ J` into one buffer, and the tensor is applied once
/// per target row.
#[derive(Clone, Debug)]
pub struct SplitGroup {
    pub vertex: usize,
    /// Frame shift of the step's position.
    pub shift: u64,
    /// Index into [`LoopProgramV4::vertex_instances`] (vertex, shift).
    pub instance: usize,
    /// Products table of (step input rank, vertex degree).
    pub products: usize,
    /// Object rows (`row × columns + column`) the group writes.
    pub targets: Vec<u32>,
}

/// One routing term of one edge in a split step.
#[derive(Clone, Debug)]
pub struct SplitPart {
    pub content: usize,
    pub term: usize,
    /// Index into [`LoopProgramV4::currents`] (content, term).
    pub current: usize,
    /// The step group of the term's vertex.
    pub group: usize,
    /// Row links whose `block` is the term row (its current row).
    pub links: Vec<super::kernels::RowLink>,
}

/// One schedule-B step: a dressed trie node, or one cut species of a
/// family accumulator (`trace`).
#[derive(Clone, Debug)]
pub struct Step {
    pub seeded: bool,
    pub trace: Option<usize>,
    /// Helicity rows of the output (objects carry `rows × columns` rows).
    pub rows: usize,
    pub d_alpha: usize,
    pub d_out: usize,
    pub degree: usize,
    pub rank_in: usize,
    pub rank_out: usize,
    pub edges: Vec<Edge>,
    /// Present when the step runs in the split form.
    pub split: Option<SplitStep>,
    /// Modelled kernel work in the fused and (when it can split) split form.
    pub costs: (StepCost, Option<StepCost>),
}

impl Step {
    /// Lane vectors per object row of the output.
    pub fn out_row(&self) -> usize {
        self.d_alpha * self.d_out * ncoeff(self.rank_out)
    }
}

/// One family: its trie-frame reading `(leg mask, mass id)`, rank, loop
/// orders per accumulator, leaves and steps (children before parents).
#[derive(Clone, Debug)]
pub struct Family {
    pub sequence: Vec<(u64, usize)>,
    pub rank: usize,
    pub orders: Vec<usize>,
    pub leaves: Vec<Leaf>,
    pub steps: Vec<Step>,
    /// Distinct fused factors the fused steps read, ascending.
    pub factors: Vec<usize>,
    /// Distinct vertex instances the split steps read, ascending.
    pub vertices: Vec<usize>,
    /// Distinct split currents the split steps read, ascending.
    pub currents: Vec<usize>,
}

impl Family {
    pub fn propagators(&self) -> usize {
        self.sequence.len()
    }

    /// Cumulative leg masks `V_1..V_N` of the frame shifts.
    pub fn shift_masks(&self) -> Vec<u64> {
        let mut total = 0;
        self.sequence
            .iter()
            .map(|&(mask, _)| {
                total |= mask;
                total
            })
            .collect()
    }
}

/// A closure of a schedule-F state into a family accumulator.
#[derive(Clone, Debug)]
pub struct Closure {
    pub family: usize,
    pub accumulator: usize,
    pub seed: usize,
    pub weight: Rational,
    /// Born row of each state row (`None`: no seed row).
    pub rows: Vec<Option<usize>>,
    /// `q = sign q' - V(mask)` into the family's trie frame.
    pub relabel: Option<(i8, u64)>,
}

/// One schedule-F state: dressing of its parent (or the unit seed) by one
/// content, with helicity-explicit rows of the attached legs.
#[derive(Clone, Debug)]
pub struct State {
    pub parent: Option<usize>,
    pub factor: usize,
    pub content: usize,
    pub seeded: bool,
    pub rows: usize,
    pub d_alpha: usize,
    pub d_out: usize,
    pub source_rank: usize,
    pub rank: usize,
    pub products: usize,
    pub links: Vec<super::kernels::RowLink>,
    pub closures: Vec<Closure>,
    pub children: usize,
}

impl State {
    pub fn out_row(&self) -> usize {
        self.d_alpha * self.d_out * ncoeff(self.rank)
    }
}

/// A validated plan v4 with its derived executor tables.
#[derive(Debug)]
pub struct LoopProgramV4 {
    pub legs: usize,
    /// Born helicity rows (seed rows).
    pub rows: usize,
    /// Observable columns (squared-order sectors).
    pub columns: usize,
    pub average: u64,
    pub born_components: usize,
    pub constants: usize,
    pub masses: usize,
    pub block_rows: Vec<usize>,
    pub kernels: Vec<Kernel>,
    pub contents: Vec<Content>,
    pub factors: Vec<(usize, u64)>,
    pub vertices: Vec<Vertex>,
    /// Split vertex tensors at a frame shift: (vertex, shift mask).
    pub vertex_instances: Vec<(usize, u64)>,
    /// Block currents of the split form: (content, term), one row of `width`
    /// components per term row.
    pub currents: Vec<(usize, usize)>,
    pub seeds: Vec<Seed>,
    pub families: Vec<Family>,
    pub forward: Option<Vec<State>>,
    pub products: Vec<MonomialProductTable>,
    /// Loop orders labels used by the families (indices as exported).
    pub order_count: usize,
}

struct Tables {
    index: HashMap<(usize, usize), usize>,
    tables: Vec<MonomialProductTable>,
}

/// Ids of the split inputs the chosen split steps read.
#[derive(Default)]
struct SplitInputs {
    instances: HashMap<(usize, u64), usize>,
    currents: HashMap<(usize, usize), usize>,
}

impl SplitInputs {
    fn instance(&mut self, vertex: usize, shift: u64) -> usize {
        let next = self.instances.len();
        *self.instances.entry((vertex, shift)).or_insert(next)
    }

    fn current(&mut self, content: usize, term: usize) -> usize {
        let next = self.currents.len();
        *self.currents.entry((content, term)).or_insert(next)
    }

    fn ordered<K: Copy>(map: &HashMap<K, usize>) -> Vec<K> {
        let mut out: Vec<(usize, K)> = map.iter().map(|(&key, &id)| (id, key)).collect();
        out.sort_unstable_by_key(|&(id, _)| id);
        out.into_iter().map(|(_, key)| key).collect()
    }
}

impl Tables {
    fn get(&mut self, rank: usize, degree: usize) -> usize {
        *self.index.entry((rank, degree)).or_insert_with(|| {
            self.tables.push(MonomialProductTable::new(rank, degree));
            self.tables.len() - 1
        })
    }
}

/// Row links `(target, source, block)` of one helicity row each, expanded
/// over the observable columns: object row `row * columns + column`.
fn expand_links(flat: &[u32], columns: usize) -> RusticolResult<Vec<super::kernels::RowLink>> {
    require(flat.len() % 3 == 0, "row links are not triples")?;
    let columns = u32::try_from(columns).map_err(|_| invalid("too many columns"))?;
    let mut out = Vec::with_capacity(flat.len() / 3 * columns as usize);
    for link in flat.chunks_exact(3) {
        for o in 0..columns {
            let row = |r: u32| {
                r.checked_mul(columns)
                    .and_then(|x| x.checked_add(o))
                    .ok_or_else(|| invalid("object row overflows"))
            };
            out.push(super::kernels::RowLink {
                output: row(link[0])?,
                input: row(link[1])?,
                block: link[2],
            });
        }
    }
    Ok(out)
}

fn exact(value: &str) -> RusticolResult<Rational> {
    super::dress::parse_rational(value).map_err(|error| invalid(error.message()))
}

fn kernel_input(kind: &str, a: usize, b: usize) -> RusticolResult<KernelInput> {
    Ok(match kind {
        "loop" => KernelInput::Loop(a),
        "current" => KernelInput::Current {
            slot: a,
            component: b,
        },
        "momentum" => KernelInput::Momentum { slot: a, mu: b },
        "parameter" => KernelInput::Parameter,
        _ => return Err(invalid(format!("unknown kernel input kind {kind:?}"))),
    })
}

fn layouts(file: &ContentFile) -> RusticolResult<(SegmentLayout, SegmentLayout)> {
    let entry = |loop_in: usize, out: usize, e: &[u32; 6]| SegmentEntry {
        loop_in,
        out,
        exponents: [e[2], e[3], e[4], e[5]],
    };
    let open = file
        .fused
        .iter()
        .map(|e| entry(e[0] as usize, e[1] as usize, e))
        .collect();
    let seeded = file
        .fused
        .iter()
        .map(|e| entry(0, e[0] as usize * file.d_out + e[1] as usize, e))
        .collect();
    Ok((
        SegmentLayout::new(file.d_in, file.d_out, open)?,
        SegmentLayout::new(1, file.d_in * file.d_out, seeded)?,
    ))
}

/// The split vertex of one routing term: its vertex (shared by every term of
/// equal key) and, per vertex entry, the term's fused entry.
fn term_vertex(
    term: &TermFile,
    content: &ContentFile,
    split: &[(usize, Vec<KernelInput>, Vec<(usize, usize)>)],
    width: usize,
) -> RusticolResult<(Vertex, Vec<usize>)> {
    let key = |e: usize, nu: usize| -> VertexEntry {
        let f = content.fused[e];
        (f[0] as usize, nu, f[1] as usize, [f[2], f[3], f[4], f[5]])
    };
    let mut entries: Vec<VertexEntry> = split
        .iter()
        .flat_map(|(_, _, outputs)| outputs.iter().map(|&(e, nu)| key(e, nu)))
        .collect();
    entries.sort_unstable();
    entries.dedup();
    let index: HashMap<VertexEntry, usize> =
        entries.iter().enumerate().map(|(i, &e)| (e, i)).collect();
    let fused_index: HashMap<(usize, usize, [u32; 4]), usize> = content
        .fused
        .iter()
        .enumerate()
        .map(|(i, f)| ((f[0] as usize, f[1] as usize, [f[2], f[3], f[4], f[5]]), i))
        .collect();
    let fused = entries
        .iter()
        .map(|&(b, _nu, bp, exponents)| {
            fused_index
                .get(&(b, bp, exponents))
                .copied()
                .ok_or_else(|| invalid("a vertex entry outside its content's fused entries"))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let bindings = split
        .iter()
        .map(|(kernel, inputs, outputs)| Binding {
            kernel: *kernel,
            inputs: inputs.clone(),
            entries: outputs.iter().map(|&(e, nu)| index[&key(e, nu)]).collect(),
        })
        .collect();
    let (d_in, d_out) = (content.d_in, content.d_out);
    let layout = |seeded: bool| {
        let entries = entries
            .iter()
            .map(|&(b, nu, bp, exponents)| {
                let (loop_in, out) = if seeded {
                    (nu, b * d_out + bp)
                } else {
                    (b * width + nu, bp)
                };
                SegmentEntry {
                    loop_in,
                    out,
                    exponents,
                }
            })
            .collect();
        if seeded {
            SegmentLayout::new(width, d_in * d_out, entries)
        } else {
            SegmentLayout::new(d_in * width, d_out, entries)
        }
    };
    let degree = entries
        .iter()
        .map(|e| e.3.iter().sum::<u32>() as usize)
        .max()
        .unwrap_or(0);
    let vertex = Vertex {
        width,
        d_in,
        d_out,
        degree,
        slots: term.slots.clone(),
        bindings,
        open: layout(false)?,
        seeded: layout(true)?,
        entries,
    };
    Ok((vertex, fused))
}

/// What every family of one plan reads while loading.
struct Shared<'a> {
    file: &'a PlanFile,
    contents: &'a [Content],
    vertices: &'a [Vertex],
    seeds: usize,
    policy: FormPolicy,
}

/// Cut index and input coefficients of a step's kernels: a seeded step's
/// input is `δ_αβ` times one value per row.
fn step_shape(seeded: bool, d_alpha: usize, rank: usize) -> (usize, usize) {
    if seeded {
        (1, 1)
    } else {
        (d_alpha, ncoeff(rank))
    }
}

/// Kernel work of a step in the fused form.
fn fused_cost(seeded: bool, d_alpha: usize, edges: &[Edge], contents: &[Content]) -> StepCost {
    let mut cost = StepCost::default();
    for edge in edges {
        let (cut, m_in) = step_shape(seeded, d_alpha, edge.source_rank);
        let iterations = (edge.links.len() * cut * contents[edge.content].factor_row()) as u64;
        cost.iterations += iterations;
        cost.work += iterations * m_in as u64;
    }
    cost
}

/// The split form of a step and its kernel work, if the step can split:
/// every term of every edge's content has a vertex, and every source is
/// stored at the step's input rank (one accumulator layout per group).
fn split_candidate(
    s: &StepFile,
    edges: &[Edge],
    shared: &Shared<'_>,
    tables: &mut Tables,
) -> RusticolResult<Option<(SplitStep, StepCost)>> {
    let rank_in = if s.seeded { 0 } else { s.rank_in };
    let splits = edges.iter().all(|edge| {
        edge.source_rank == rank_in
            && shared.contents[edge.content]
                .terms
                .iter()
                .all(|term| term.vertex.is_some())
    });
    if !splits {
        return Ok(None);
    }
    let (cut, m_in) = step_shape(s.seeded, s.d_alpha, rank_in);
    let mut groups: Vec<SplitGroup> = Vec::new();
    let mut group_of: HashMap<(usize, u64), usize> = HashMap::new();
    let mut targets: Vec<BTreeSet<u32>> = Vec::new();
    let mut iterations = 0;
    let mut parts = Vec::with_capacity(edges.len());
    for edge in edges {
        let content = &shared.contents[edge.content];
        let shift = shared.file.factors[edge.factor].1;
        let mut edge_parts = Vec::with_capacity(content.terms.len());
        for (t, term) in content.terms.iter().enumerate() {
            let vertex = term.vertex.as_ref().expect("checked above").id;
            let shape = &shared.vertices[vertex];
            require(
                shape.d_in == content.d_in && shape.d_out == content.d_out,
                "a vertex does not fit its content",
            )?;
            let group = *group_of.entry((vertex, shift)).or_insert_with(|| {
                groups.push(SplitGroup {
                    vertex,
                    shift,
                    instance: 0,
                    products: tables.get(rank_in, shape.degree),
                    targets: Vec::new(),
                });
                targets.push(BTreeSet::new());
                groups.len() - 1
            });
            let mut row_of = vec![None; content.rows];
            for (k, (row, _blocks)) in term.rows.iter().enumerate() {
                row_of[*row] = Some(k as u32);
            }
            let links: Vec<_> = edge
                .links
                .iter()
                .filter_map(|link| {
                    row_of[link.block as usize]
                        .map(|block| super::kernels::RowLink { block, ..*link })
                })
                .collect();
            targets[group].extend(links.iter().map(|link| link.output));
            let incoming = if s.seeded { 1 } else { content.d_in };
            iterations += links.len() * cut * incoming * shape.width;
            edge_parts.push(SplitPart {
                content: edge.content,
                term: t,
                current: 0,
                group,
                links,
            });
        }
        parts.push(edge_parts);
    }
    for (group, rows) in groups.iter_mut().zip(targets) {
        group.targets = rows.into_iter().collect();
        iterations += group.targets.len() * cut * shared.vertices[group.vertex].entries.len();
    }
    let cost = StepCost {
        work: (iterations * m_in) as u64,
        iterations: iterations as u64,
    };
    Ok(Some((SplitStep { groups, parts }, cost)))
}

impl LoopProgramV4 {
    /// Parse and validate a plan exported by `plan_export.export_plan`,
    /// choosing kernel forms by gate G0's rule.
    pub fn load(json: &[u8]) -> RusticolResult<Self> {
        Self::load_with(json, FormPolicy::Rule)
    }

    /// [`Self::load`] with an explicit kernel-form policy.
    pub fn load_with(json: &[u8], policy: FormPolicy) -> RusticolResult<Self> {
        let file: PlanFile =
            serde_json::from_slice(json).map_err(|error| invalid(error.to_string()))?;
        Self::from_file(file, policy)
    }

    fn from_file(file: PlanFile, policy: FormPolicy) -> RusticolResult<Self> {
        require(file.schema == PLAN_SCHEMA, "unsupported plan schema")?;
        require(
            file.legs > 0 && file.legs < 64 && file.rows > 0 && file.columns > 0,
            "empty process",
        )?;
        require(file.average > 0, "normalisation divisor must be positive")?;
        let columns = file.columns;
        let kernels = file
            .kernels
            .iter()
            .map(|k| {
                let outputs = k
                    .outputs
                    .iter()
                    .map(|terms| {
                        terms
                            .iter()
                            .map(|(constant, monomial)| {
                                require(*constant < file.constants, "kernel constant index")?;
                                require(
                                    monomial.iter().all(|&(p, _)| p < k.inputs.len()),
                                    "kernel monomial input index",
                                )?;
                                Ok(KernelTerm {
                                    constant: *constant,
                                    monomial: monomial.clone(),
                                })
                            })
                            .collect::<RusticolResult<Vec<_>>>()
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                Ok(Kernel {
                    inputs: k.inputs.len(),
                    outputs,
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let mut contents = Vec::with_capacity(file.contents.len());
        let mut vertices: Vec<Option<Vertex>> = Vec::new();
        for c in &file.contents {
            let (open, seeded) = layouts(c)?;
            require(
                open.degree() <= c.degree,
                "content kernel degree above q-degree",
            )?;
            let mut terms = Vec::with_capacity(c.terms.len());
            for t in &c.terms {
                let bindings = t
                    .bindings
                    .iter()
                    .map(|b| {
                        let kernel = kernels
                            .get(b.kernel)
                            .ok_or_else(|| invalid("binding kernel index"))?;
                        require(
                            b.inputs.len() == kernel.inputs
                                && b.entries.len() == kernel.outputs.len()
                                && b.entries.iter().all(|&e| e < c.fused.len()),
                            "binding shape",
                        )?;
                        let inputs = b
                            .inputs
                            .iter()
                            .map(|(kind, a, b)| kernel_input(kind, *a, *b))
                            .collect::<RusticolResult<Vec<_>>>()?;
                        for input in &inputs {
                            match *input {
                                KernelInput::Loop(mu) | KernelInput::Momentum { mu, .. }
                                    if mu >= 4 =>
                                {
                                    return Err(invalid("momentum component"));
                                }
                                KernelInput::Current { slot, .. }
                                | KernelInput::Momentum { slot, .. }
                                    if slot >= t.slots.len() =>
                                {
                                    return Err(invalid("binding slot index"));
                                }
                                _ => {}
                            }
                        }
                        Ok(Binding {
                            kernel: b.kernel,
                            inputs,
                            entries: b.entries.clone(),
                        })
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                for (row, blocks) in &t.rows {
                    require(
                        *row < c.rows
                            && blocks.len() == t.slots.len()
                            && blocks.iter().all(|&b| b < file.block_rows.len()),
                        "content term row",
                    )?;
                    for binding in &bindings {
                        for input in &binding.inputs {
                            if let KernelInput::Current { slot, component } = *input {
                                require(
                                    component < file.block_rows[blocks[slot]],
                                    "block current component",
                                )?;
                            }
                        }
                    }
                }
                require(
                    t.dimensions.len() == t.slots.len(),
                    "content term dimensions",
                )?;
                let width: usize = t.dimensions.iter().product();
                let split = t
                    .split
                    .iter()
                    .map(|b| {
                        let kernel = kernels
                            .get(b.kernel)
                            .ok_or_else(|| invalid("split kernel index"))?;
                        require(
                            b.inputs.len() == kernel.inputs
                                && b.entries.len() == kernel.outputs.len()
                                && b.entries
                                    .iter()
                                    .all(|&(e, nu)| e < c.fused.len() && nu < width),
                            "split binding shape",
                        )?;
                        let inputs = b
                            .inputs
                            .iter()
                            .map(|(kind, a, b)| kernel_input(kind, *a, *b))
                            .collect::<RusticolResult<Vec<_>>>()?;
                        for input in &inputs {
                            match *input {
                                KernelInput::Current { .. } => {
                                    return Err(invalid("a split vertex reads a block current"));
                                }
                                KernelInput::Loop(mu) | KernelInput::Momentum { mu, .. }
                                    if mu >= 4 =>
                                {
                                    return Err(invalid("momentum component"));
                                }
                                KernelInput::Momentum { slot, .. } if slot >= t.slots.len() => {
                                    return Err(invalid("split slot index"));
                                }
                                _ => {}
                            }
                        }
                        Ok((b.kernel, inputs, b.entries.clone()))
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                for (_row, blocks) in &t.rows {
                    for (slot, &block) in blocks.iter().enumerate() {
                        require(
                            file.block_rows[block] == t.dimensions[slot],
                            "block row dimension",
                        )?;
                    }
                }
                let vertex = match usize::try_from(t.vertex) {
                    Err(_) => {
                        require(t.split.is_empty(), "split kernels without a vertex")?;
                        None
                    }
                    Ok(id) => {
                        require(!t.split.is_empty(), "a vertex without split kernels")?;
                        let (vertex, fused) = term_vertex(t, c, &split, width)?;
                        if vertices.len() <= id {
                            vertices.resize(id + 1, None);
                        }
                        match &vertices[id] {
                            Some(known) => require(
                                known.same_kernels(&vertex),
                                "one vertex key with different split kernels",
                            )?,
                            None => vertices[id] = Some(vertex),
                        }
                        Some(TermVertex { id, fused })
                    }
                };
                terms.push(ContentTerm {
                    slots: t.slots.clone(),
                    dimensions: t.dimensions.clone(),
                    bindings,
                    vertex,
                    rows: t.rows.clone(),
                });
            }
            contents.push(Content {
                d_in: c.d_in,
                d_out: c.d_out,
                degree: c.degree,
                rows: c.rows,
                terms,
                open,
                seeded,
            });
        }
        for &(content, _shift) in &file.factors {
            require(content < contents.len(), "factor content index")?;
        }
        let vertices = vertices
            .into_iter()
            .map(|vertex| vertex.ok_or_else(|| invalid("a vertex id no term defines")))
            .collect::<RusticolResult<Vec<_>>>()?;
        let seeds = file
            .seeds
            .iter()
            .map(|s| {
                require(s.columns.len() == columns, "seed columns")?;
                Ok(Seed {
                    columns: s
                        .columns
                        .iter()
                        .map(|terms| {
                            terms
                                .iter()
                                .map(|(component, re, im)| {
                                    require(*component < file.born_components, "seed component")?;
                                    Ok(SeedTerm {
                                        component: *component,
                                        weight: [exact(re)?, exact(im)?],
                                    })
                                })
                                .collect::<RusticolResult<Vec<_>>>()
                        })
                        .collect::<RusticolResult<Vec<_>>>()?,
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let mut tables = Tables {
            index: HashMap::new(),
            tables: Vec::new(),
        };
        let mut order_count = 0;
        let mut families = Vec::with_capacity(file.families.len());
        let mut split_inputs = SplitInputs::default();
        let shared = Shared {
            file: &file,
            contents: &contents,
            vertices: &vertices,
            seeds: seeds.len(),
            policy,
        };
        for f in &file.families {
            order_count = order_count.max(f.orders.iter().map(|&o| o + 1).max().unwrap_or(0));
            families.push(Self::family(f, &shared, &mut tables, &mut split_inputs)?);
        }
        let forward = file
            .forward
            .as_ref()
            .map(|forward| Self::forward(forward, &file, &contents, &families, &mut tables))
            .transpose()?;
        Ok(Self {
            legs: file.legs,
            rows: file.rows,
            columns,
            average: file.average,
            born_components: file.born_components,
            constants: file.constants,
            masses: file.masses,
            block_rows: file.block_rows,
            kernels,
            contents,
            factors: file.factors,
            vertices,
            vertex_instances: SplitInputs::ordered(&split_inputs.instances),
            currents: SplitInputs::ordered(&split_inputs.currents),
            seeds,
            families,
            forward,
            products: tables.tables,
            order_count,
        })
    }

    fn family(
        f: &FamilyFile,
        shared: &Shared<'_>,
        tables: &mut Tables,
        split_inputs: &mut SplitInputs,
    ) -> RusticolResult<Family> {
        let (file, contents, seeds) = (shared.file, shared.contents, shared.seeds);
        let full = (1u64 << file.legs) - 1;
        let mut covered = 0u64;
        for &(mask, mass) in &f.sequence {
            require(
                mask != 0 && mask & covered == 0 && mass < file.masses,
                "family reading",
            )?;
            covered |= mask;
        }
        require(covered == full, "family reading does not cover every leg")?;
        require(
            f.rank <= f.sequence.len() + 2,
            "family rank exceeds propagators + 2",
        )?;
        let leaves = f
            .leaves
            .iter()
            .map(|leaf| {
                require(leaf.seed < seeds, "leaf seed index")?;
                Ok(Leaf {
                    seed: leaf.seed,
                    weight: exact(&leaf.weight)?,
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let mut steps: Vec<Step> = Vec::with_capacity(f.steps.len());
        let mut consumed = vec![false; f.steps.len()];
        let mut factors = BTreeSet::new();
        let mut instances = BTreeSet::new();
        let mut currents = BTreeSet::new();
        for (index, s) in f.steps.iter().enumerate() {
            let rank_out = s.rank_in + s.degree;
            let trace = usize::try_from(s.trace).ok();
            if let Some(trace) = trace {
                require(trace < f.orders.len(), "trace accumulator index")?;
                require(
                    s.rows == 1 && s.d_alpha == s.d_out,
                    "trace of an open object",
                )?;
            }
            let mut edges = Vec::with_capacity(s.edges.len());
            for e in &s.edges {
                require(e.factor < file.factors.len(), "edge factor index")?;
                let content = file.factors[e.factor].0;
                let c = &contents[content];
                require(
                    c.d_out == s.d_out && c.degree <= s.degree,
                    "edge content does not fit its step",
                )?;
                let (source, source_rank, source_rows) = match (e.leaf, e.child) {
                    (leaf, -1) if leaf >= 0 => {
                        require(s.seeded, "leaf edge of an open step")?;
                        require((leaf as usize) < leaves.len(), "edge leaf index")?;
                        require(c.d_in == s.d_alpha, "seed does not fit its content")?;
                        (Source::Leaf(leaf as usize), 0, file.rows)
                    }
                    (-1, child) if child >= 0 => {
                        let child = child as usize;
                        require(!s.seeded, "child edge of a seeded step")?;
                        require(child < index, "child step after its parent")?;
                        require(!consumed[child], "a step output feeds two edges")?;
                        consumed[child] = true;
                        let previous = &steps[child];
                        require(
                            previous.trace.is_none()
                                && previous.d_alpha == s.d_alpha
                                && previous.d_out == c.d_in,
                            "child object does not fit its edge",
                        )?;
                        (Source::Step(child), previous.rank_out, previous.rows)
                    }
                    _ => return Err(invalid("edge needs one leaf or one child")),
                };
                require(source_rank <= s.rank_in, "source rank above step rank")?;
                for link in e.links.chunks(3) {
                    require(
                        link.len() == 3
                            && (link[0] as usize) < s.rows
                            && (link[1] as usize) < source_rows
                            && (link[2] as usize) < c.rows,
                        "row link out of range",
                    )?;
                }
                edges.push(Edge {
                    source,
                    factor: e.factor,
                    content,
                    source_rank,
                    products: tables.get(source_rank, c.degree),
                    links: expand_links(&e.links, file.columns)?,
                });
            }
            require(!edges.is_empty(), "a step without edges")?;
            let fused_cost = fused_cost(s.seeded, s.d_alpha, &edges, contents);
            let candidate = split_candidate(s, &edges, shared, tables)?;
            let split_cost = candidate.as_ref().map(|(_, cost)| *cost);
            let split = match (shared.policy, candidate) {
                (_, None) | (FormPolicy::Fused, _) => None,
                (FormPolicy::Split, Some((split, _))) => Some(split),
                (FormPolicy::Rule, Some((split, cost))) => {
                    (SPLIT_MODEL.seconds(cost) < FUSED_MODEL.seconds(fused_cost)).then_some(split)
                }
            };
            let split = split.map(|mut split| {
                for group in &mut split.groups {
                    group.instance = split_inputs.instance(group.vertex, group.shift);
                    instances.insert(group.instance);
                }
                for part in split.parts.iter_mut().flatten() {
                    part.current = split_inputs.current(part.content, part.term);
                    currents.insert(part.current);
                }
                split
            });
            if split.is_none() {
                factors.extend(edges.iter().map(|edge| edge.factor));
            }
            steps.push(Step {
                seeded: s.seeded,
                trace,
                rows: s.rows,
                d_alpha: s.d_alpha,
                d_out: s.d_out,
                degree: s.degree,
                rank_in: s.rank_in,
                rank_out,
                edges,
                split,
                costs: (fused_cost, split_cost),
            });
        }
        for (index, step) in steps.iter().enumerate() {
            require(
                consumed[index] == step.trace.is_none(),
                "every open object feeds exactly one edge",
            )?;
        }
        Ok(Family {
            sequence: f.sequence.clone(),
            rank: f.rank,
            orders: f.orders.clone(),
            leaves,
            steps,
            factors: factors.into_iter().collect(),
            vertices: instances.into_iter().collect(),
            currents: currents.into_iter().collect(),
        })
    }

    fn forward(
        forward: &ForwardFile,
        file: &PlanFile,
        contents: &[Content],
        families: &[Family],
        tables: &mut Tables,
    ) -> RusticolResult<Vec<State>> {
        let mut states: Vec<State> = Vec::with_capacity(forward.states.len());
        for (index, s) in forward.states.iter().enumerate() {
            require(s.factor < file.factors.len(), "state factor index")?;
            let content = file.factors[s.factor].0;
            let c = &contents[content];
            let parent = usize::try_from(s.parent).ok();
            let (source_rows, source_rank) = match parent {
                None => {
                    require(s.seeded && c.d_in == s.d_alpha, "root state")?;
                    (1, 0)
                }
                Some(parent) => {
                    require(parent < index && !s.seeded, "state parent")?;
                    let p = &states[parent];
                    require(
                        p.d_out == c.d_in && p.d_alpha == s.d_alpha,
                        "state does not fit its parent",
                    )?;
                    (p.rows, p.rank)
                }
            };
            require(
                c.d_out == s.d_out && c.degree == s.degree && s.rank == source_rank + s.degree,
                "state shape",
            )?;
            for link in s.links.chunks(3) {
                require(
                    link.len() == 3
                        && (link[0] as usize) < s.rows
                        && (link[1] as usize) < source_rows
                        && (link[2] as usize) < c.rows,
                    "state row link",
                )?;
            }
            let closures = s
                .closures
                .iter()
                .map(|closure| {
                    let family = families
                        .get(closure.family)
                        .ok_or_else(|| invalid("closure family index"))?;
                    require(
                        closure.accumulator < family.orders.len()
                            && closure.seed < file.seeds.len()
                            && closure.rows.len() == s.rows
                            && s.d_alpha == s.d_out,
                        "state closure",
                    )?;
                    require(
                        closure
                            .rows
                            .iter()
                            .all(|&r| r < 0 || (r as usize) < file.rows),
                        "closure seed row",
                    )?;
                    require(
                        closure
                            .relabel
                            .is_none_or(|(sign, _)| sign == 1 || sign == -1),
                        "closure frame map sign",
                    )?;
                    Ok(Closure {
                        family: closure.family,
                        accumulator: closure.accumulator,
                        seed: closure.seed,
                        weight: exact(&closure.weight)?,
                        rows: closure
                            .rows
                            .iter()
                            .map(|&r| usize::try_from(r).ok())
                            .collect(),
                        relabel: closure.relabel,
                    })
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            if let Some(parent) = parent {
                states[parent].children += 1;
            }
            states.push(State {
                parent,
                factor: s.factor,
                content,
                seeded: s.seeded,
                rows: s.rows,
                d_alpha: s.d_alpha,
                d_out: s.d_out,
                source_rank,
                rank: s.rank,
                products: tables.get(source_rank, c.degree),
                links: expand_links(&s.links, 1)?,
                closures,
                children: 0,
            });
        }
        Ok(states)
    }

    /// Families grouped by NinjaRS bucket `(propagators, rank)`, buckets in
    /// ascending order and families in plan order within a bucket.
    pub fn buckets(&self) -> BTreeMap<(usize, usize), Vec<usize>> {
        let mut out: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for (index, family) in self.families.iter().enumerate() {
            out.entry((family.propagators(), family.rank))
                .or_default()
                .push(index);
        }
        out
    }

    /// Lane vectors of one fused factor.
    pub fn factor_len(&self, factor: usize) -> usize {
        let content = &self.contents[self.factors[factor].0];
        content.rows * content.factor_row()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn links_expand_over_columns() {
        let links = expand_links(&[1, 0, 2, 0, 1, 1], 2).unwrap();
        let triples: Vec<_> = links.iter().map(|l| (l.output, l.input, l.block)).collect();
        assert_eq!(triples, [(2, 0, 2), (3, 1, 2), (0, 2, 1), (1, 3, 1)]);
        assert!(expand_links(&[1, 0], 1).is_err());
    }

    #[test]
    fn rejects_other_schemas() {
        let text = br#"{"schema":3,"legs":4,"rows":1,"columns":1,"average":1,
            "born_components":0,"constants":0,"masses":0,"block_rows":[],"kernels":[],
            "contents":[],"factors":[],"seeds":[],"families":[],"forward":null}"#;
        let error = LoopProgramV4::load(text).unwrap_err();
        assert!(error.message().contains("schema"), "{error}");
    }
}
