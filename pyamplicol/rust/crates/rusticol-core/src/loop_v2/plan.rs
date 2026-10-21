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
/// fused entry each output adds into.
#[derive(Clone, Debug)]
pub struct Binding {
    pub kernel: usize,
    pub inputs: Vec<KernelInput>,
    pub entries: Vec<usize>,
}

/// A block-independent vertex kernel of a routing term (the split form):
/// each output multiplies block component `nu` (flat over the slots'
/// current components, first slot slowest) into a fused entry.
#[derive(Clone, Debug)]
pub struct SplitBinding {
    pub kernel: usize,
    pub inputs: Vec<KernelInput>,
    /// (fused entry, nu) per kernel output.
    pub entries: Vec<(usize, usize)>,
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
    /// The same term as vertex kernels times block components; when present
    /// the factor builder evaluates the vertex once per (content, shift)
    /// and contracts it with each row's block components.
    pub split: Vec<SplitBinding>,
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
    /// Distinct fused factors the steps read, ascending.
    pub factors: Vec<usize>,
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

impl LoopProgramV4 {
    /// Parse and validate a plan exported by `plan_export.export_plan`.
    pub fn load(json: &[u8]) -> RusticolResult<Self> {
        let file: PlanFile =
            serde_json::from_slice(json).map_err(|error| invalid(error.to_string()))?;
        Self::from_file(file)
    }

    fn from_file(file: PlanFile) -> RusticolResult<Self> {
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
                        Ok(SplitBinding {
                            kernel: b.kernel,
                            inputs,
                            entries: b.entries.clone(),
                        })
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
                terms.push(ContentTerm {
                    slots: t.slots.clone(),
                    dimensions: t.dimensions.clone(),
                    bindings,
                    split,
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
        for f in &file.families {
            order_count = order_count.max(f.orders.iter().map(|&o| o + 1).max().unwrap_or(0));
            families.push(Self::family(f, &file, &contents, seeds.len(), &mut tables)?);
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
            seeds,
            families,
            forward,
            products: tables.tables,
            order_count,
        })
    }

    fn family(
        f: &FamilyFile,
        file: &PlanFile,
        contents: &[Content],
        seeds: usize,
        tables: &mut Tables,
    ) -> RusticolResult<Family> {
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
                factors.insert(e.factor);
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
