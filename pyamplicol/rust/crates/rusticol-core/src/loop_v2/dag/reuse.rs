// SPDX-License-Identifier: 0BSD

//! Evaluation reuse of loop-request DAGs
//! (`dag_equivalence.assign_recursive_current_evaluation_reuse`).
//!
//! The value-equivalence derivation (`_derive_current_value_equivalences`)
//! classifies currents in increasing leg-set order. Only the currents whose
//! evaluation contract (`_current_evaluation_contract`, every field but the
//! colour state) another current shares get a term vector: the sum, per
//! evaluation key (kernel class, canonical inputs, result particle and
//! chirality, runtime coupling identity), of colour weight times evaluation
//! factor (`_current_term_vector`). A current joins the class of an earlier
//! one when its vector is equal, opposite, or projective through a finite
//! binary64 factor that reconstructs every coefficient bit for bit
//! (`_classify_current_term_vector`); every other current is its own
//! representative (A0). The interactions are then regrouped by those
//! representatives (`_rewrite_interaction_evaluation_reuse`).
//!
//! The arithmetic is the oracle's to the bit ([`super::exact`]); keys
//! compare floats as Python does (`0.0 == -0.0`), and term vectors sort as
//! Python sorts its key tuples: by the kernel class string, the inputs and
//! the runtime coupling identity (both passed as ranks of Python's order).

use super::exact::{
    Complex, bits_equal, canonical, canonical_zero, complex_mul, exact_product, exact_ratio,
    finite, fsum, key_bits, pair_eq, roundtrip_ratio,
};
use super::blocks::canonical_ids;
use super::input::{DagInput, Vertex};
use super::recursion::{Dag, FxMap};
use super::{ColumnarDag, DagError, DagResult, NO_HELICITY, NO_LEG, invalid};
use std::collections::HashMap;

/// A vertex kernel's model-certified evaluation symmetry
/// (`VertexEvaluationEquivalence`), with its class as the rank of the class
/// string in Python's order (equal ranks iff equal classes).
#[derive(Clone, Copy, Debug)]
pub struct Kernel {
    pub class_rank: u32,
    pub factor: Complex,
    pub swap: bool,
    pub exchange: Option<Complex>,
}

/// `current = factor * representative` (`_CurrentValueEquivalence`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Equivalence {
    pub representative: u32,
    pub factor: Complex,
}

impl Kernel {
    /// The kernel of a vertex row, with `class_rank` for its class.
    pub fn of(vertex: &Vertex, class_rank: u32) -> Self {
        Self {
            class_rank,
            factor: vertex.factor,
            swap: vertex.swap,
            exchange: vertex.exchange,
        }
    }
}

const IDENTITY_FACTOR: Complex = [1.0, 0.0];

/// What the derivation reads of a DAG.
pub trait ReuseDag {
    fn current_count(&self) -> usize;
    fn interaction_count(&self) -> usize;
    /// External mask (its size orders the derivation).
    fn mask(&self, current: usize) -> u64;
    fn is_source(&self, current: usize) -> bool;
    /// `(source leg, source helicity)`, sentinels where missing.
    fn source_state(&self, current: usize) -> (i32, i32);
    fn pdg(&self, current: usize) -> i64;
    fn chirality(&self, current: usize) -> i8;
    /// Index into the kernel table.
    fn kernel(&self, interaction: usize) -> usize;
    fn left(&self, interaction: usize) -> u32;
    fn right(&self, interaction: usize) -> u32;
    fn result(&self, interaction: usize) -> u32;
    fn colour_weight(&self, interaction: usize) -> Complex;
    /// Rank of the interaction's runtime coupling identity in Python's order.
    fn identity_rank(&self, interaction: usize) -> u32;
}

/// `_canonical_kernel_evaluation`: canonical representative inputs and the
/// concrete kernel's factor.
pub fn canonical_kernel_evaluation(kernel: &Kernel, left: u32, right: u32) -> ((u32, u32), Complex) {
    let mut inputs = (left, right);
    if kernel.swap {
        inputs = (inputs.1, inputs.0);
    }
    let mut factor = kernel.factor;
    if let Some(exchange) = kernel.exchange
        && inputs.1 < inputs.0
    {
        inputs = (inputs.1, inputs.0);
        factor = complex_mul(factor, exchange);
    }
    (inputs, factor)
}

/// `_canonical_interaction_evaluation`: the representatives' inputs with
/// the exact product of the kernel and parent factors, or the concrete
/// inputs when a product is not exactly representable.
fn canonical_interaction_evaluation(
    kernel: &Kernel,
    left_id: u32,
    right_id: u32,
    left: Equivalence,
    right: Equivalence,
) -> ((u32, u32), Complex) {
    let (inputs, kernel_factor) =
        canonical_kernel_evaluation(kernel, left.representative, right.representative);
    if let Some(parent) = exact_product(left.factor, right.factor)
        && let Some(factor) = exact_product(kernel_factor, parent)
    {
        return (inputs, factor);
    }
    canonical_kernel_evaluation(kernel, left_id, right_id)
}

/// One evaluation key of a current's term vector, in Python's tuple order:
/// (class, canonical inputs, coupling identity). The result particle and
/// chirality of the oracle's key are the current's own.
type TermKey = (u32, u32, u32, u32);

/// A term with its coefficient's canonical bits.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Term {
    key: TermKey,
    value: [u64; 2],
}

impl Term {
    fn coefficient(&self) -> Complex {
        [f64::from_bits(self.value[0]), f64::from_bits(self.value[1])]
    }
}

type TermVector = Vec<Term>;

struct ProjectiveRepresentative {
    representative: u32,
    terms: TermVector,
    normalisation: Complex,
}

fn vector_is_finite_nonzero(terms: &[Term]) -> bool {
    !terms.is_empty()
        && terms.iter().all(|term| {
            let value = term.coefficient();
            !pair_eq(value, [0.0, 0.0]) && finite(value)
        })
}

fn negated(terms: &[Term]) -> TermVector {
    terms
        .iter()
        .map(|term| {
            let value = term.coefficient();
            Term {
                key: term.key,
                value: key_bits([-value[0], -value[1]]),
            }
        })
        .collect()
}

/// `_term_vector_scaled_exactly`.
fn scaled_exactly(representative: &[Term], factor: Complex, candidate: &[Term]) -> bool {
    if !finite(factor) || pair_eq(factor, [0.0, 0.0]) || representative.len() != candidate.len()
    {
        return false;
    }
    representative.iter().zip(candidate).all(|(rep, cand)| {
        rep.key == cand.key
            && bits_equal(
                canonical(complex_mul(factor, rep.coefficient())),
                canonical(cand.coefficient()),
            )
    })
}

/// `_term_vector_divided_by_exact_factor` (the sign fast path).
fn divided_by_sign(candidate: &[Term], factor: Complex) -> Option<TermVector> {
    if pair_eq(factor, [1.0, 0.0]) {
        return Some(candidate.to_vec());
    }
    if !pair_eq(factor, [-1.0, 0.0]) {
        return None;
    }
    Some(negated(candidate))
}

/// `_canonicalize_projective_term_vector`: the lookup key normalised by the
/// first coefficient, and that factor.
fn canonical_projective(terms: &[Term]) -> Option<(TermVector, Complex)> {
    if !vector_is_finite_nonzero(terms) {
        return None;
    }
    if terms.windows(2).any(|pair| pair[0].key >= pair[1].key) {
        return None;
    }
    let factor = canonical(terms[0].coefficient());
    let mut normalised = Vec::with_capacity(terms.len());
    for term in terms {
        let coefficient = canonical(term.coefficient());
        let value = roundtrip_ratio(coefficient, factor)?;
        normalised.push(Term {
            key: term.key,
            value: key_bits(value),
        });
    }
    Some((normalised, factor))
}

#[derive(Default)]
struct Classes {
    exact: FxMap<(u32, TermVector), Equivalence>,
    projective: FxMap<(u32, TermVector), Vec<ProjectiveRepresentative>>,
}

impl Classes {
    /// `_classify_current_term_vector`.
    fn classify(&mut self, current: u32, contract: u32, terms: TermVector) -> Equivalence {
        let identity = Equivalence {
            representative: current,
            factor: IDENTITY_FACTOR,
        };
        if terms.is_empty() {
            return *self.exact.entry((contract, terms)).or_insert(identity);
        }
        if !vector_is_finite_nonzero(&terms) {
            return identity;
        }
        let key = (contract, terms);
        if let Some(found) = self.exact.get(&key) {
            return *found;
        }
        let terms = &key.1;
        if let Some(opposite) = self.exact.get(&(contract, negated(terms))) {
            let factor = [
                canonical_zero(-opposite.factor[0]),
                canonical_zero(-opposite.factor[1]),
            ];
            if let Some(representative) = divided_by_sign(terms, factor)
                && scaled_exactly(&representative, factor, terms)
            {
                let equivalence = Equivalence {
                    representative: opposite.representative,
                    factor,
                };
                self.exact.insert(key, equivalence);
                return equivalence;
            }
        }
        let Some((normalised, normalisation)) = canonical_projective(terms) else {
            self.exact.insert(key, identity);
            return identity;
        };
        let projective_key = (contract, normalised);
        let mut matches = Vec::new();
        if let Some(representatives) = self.projective.get(&projective_key) {
            for representative in representatives {
                let Some(factor) = exact_ratio(normalisation, representative.normalisation) else {
                    continue;
                };
                if scaled_exactly(&representative.terms, factor, terms) {
                    matches.push(Equivalence {
                        representative: representative.representative,
                        factor,
                    });
                }
            }
        }
        if matches.len() == 1 {
            self.exact.insert(key, matches[0]);
            return matches[0];
        }
        // No match: the normalised key only collided after rounding; several:
        // the representative is ambiguous. Both keep this current.
        self.projective
            .entry(projective_key)
            .or_default()
            .push(ProjectiveRepresentative {
                representative: current,
                terms: key.1.clone(),
                normalisation,
            });
        self.exact.insert(key, identity);
        identity
    }
}

/// Interactions grouped by result, in DAG order (CSR).
fn by_result<D: ReuseDag>(dag: &D) -> (Vec<u32>, Vec<u32>) {
    let n = dag.current_count();
    let mut offsets = vec![0u32; n + 1];
    for id in 0..dag.interaction_count() {
        offsets[dag.result(id) as usize + 1] += 1;
    }
    for id in 0..n {
        offsets[id + 1] += offsets[id];
    }
    let mut fill = offsets.clone();
    let mut members = vec![0u32; dag.interaction_count()];
    for id in 0..dag.interaction_count() {
        let slot = &mut fill[dag.result(id) as usize];
        members[*slot as usize] = id as u32;
        *slot += 1;
    }
    (offsets, members)
}

/// `_derive_current_value_equivalences`; with `restricted` only the
/// currents sharing their contract with another are derived
/// (`derived_current_ids=_shared_contract_current_ids(dag)`, the production
/// pass), otherwise every current (the full derivation).
pub fn derive<D: ReuseDag>(
    dag: &D,
    contracts: &[u32],
    kernels: &[Kernel],
    restricted: bool,
) -> DagResult<Vec<Equivalence>> {
    let n = dag.current_count();
    let mut members_per_contract: FxMap<u32, u32> = FxMap::default();
    for contract in contracts {
        *members_per_contract.entry(*contract).or_default() += 1;
    }
    let derived = |current: usize| !restricted || members_per_contract[&contracts[current]] > 1;
    let (offsets, members) = by_result(dag);
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_by_key(|id| (dag.mask(*id as usize).count_ones(), *id));
    let mut equivalences: Vec<Option<Equivalence>> = vec![None; n];
    let mut sources: FxMap<(u32, i32, i32), u32> = FxMap::default();
    let mut classes = Classes::default();
    let mut grouped: Vec<(TermKey, Complex)> = Vec::new();
    for current in order {
        let id = current as usize;
        let interactions = &members[offsets[id] as usize..offsets[id + 1] as usize];
        if dag.is_source(id) {
            let (leg, helicity) = dag.source_state(id);
            if leg == NO_LEG || helicity == NO_HELICITY {
                return invalid(format!("source current {id} lacks physical source metadata"));
            }
            let representative = if derived(id) {
                *sources
                    .entry((contracts[id], leg, helicity))
                    .or_insert(current)
            } else {
                current
            };
            equivalences[id] = Some(Equivalence {
                representative,
                factor: IDENTITY_FACTOR,
            });
            continue;
        }
        let parents = |interaction: u32| {
            let i = interaction as usize;
            match (
                equivalences[dag.left(i) as usize],
                equivalences[dag.right(i) as usize],
            ) {
                (Some(left), Some(right)) => Ok((left, right)),
                _ => invalid(
                    "current-value equivalence requires parents from an earlier subset",
                ),
            }
        };
        if !derived(id) {
            for interaction in interactions {
                parents(*interaction)?;
            }
            equivalences[id] = Some(Equivalence {
                representative: current,
                factor: IDENTITY_FACTOR,
            });
            continue;
        }
        // `_current_term_vector`.
        grouped.clear();
        for &interaction in interactions {
            let (left, right) = parents(interaction)?;
            let i = interaction as usize;
            let kernel = &kernels[dag.kernel(i)];
            let (inputs, factor) =
                canonical_interaction_evaluation(kernel, dag.left(i), dag.right(i), left, right);
            grouped.push((
                (kernel.class_rank, inputs.0, inputs.1, dag.identity_rank(i)),
                complex_mul(dag.colour_weight(i), factor),
            ));
        }
        // Stable: each key's coefficients keep the oracle's summation order.
        grouped.sort_by_key(|(key, _)| *key);
        let mut terms: TermVector = Vec::new();
        let mut start = 0;
        while start < grouped.len() {
            let key = grouped[start].0;
            let mut stop = start;
            while stop < grouped.len() && grouped[stop].0 == key {
                stop += 1;
            }
            let values = &grouped[start..stop];
            let coefficient = [
                canonical_zero(fsum(values.iter().map(|(_, value)| value[0]))?),
                canonical_zero(fsum(values.iter().map(|(_, value)| value[1]))?),
            ];
            if !pair_eq(coefficient, [0.0, 0.0]) {
                terms.push(Term {
                    key,
                    value: [coefficient[0].to_bits(), coefficient[1].to_bits()],
                });
            }
            start = stop;
        }
        equivalences[id] = Some(classes.classify(current, contracts[id], terms));
    }
    equivalences
        .into_iter()
        .map(|item| {
            item.ok_or_else(|| {
                DagError::Invalid(
                    "current-value equivalence derivation left an unclassified current".into(),
                )
            })
        })
        .collect()
}

/// `_rewrite_interaction_evaluation_reuse`: every interaction's evaluation
/// group (first seen) and factor through `equivalences`.
pub fn rewrite<D: ReuseDag>(
    dag: &D,
    kernels: &[Kernel],
    equivalences: &[Equivalence],
) -> Vec<(i64, Complex)> {
    let mut groups: FxMap<(u32, u32, u32, i64, i8, u32), i64> = FxMap::default();
    (0..dag.interaction_count())
        .map(|i| {
            let kernel = &kernels[dag.kernel(i)];
            let (left, right) = (dag.left(i), dag.right(i));
            let (inputs, factor) = canonical_interaction_evaluation(
                kernel,
                left,
                right,
                equivalences[left as usize],
                equivalences[right as usize],
            );
            let result = dag.result(i) as usize;
            let next = groups.len() as i64;
            let group = *groups
                .entry((
                    kernel.class_rank,
                    inputs.0,
                    inputs.1,
                    dag.pdg(result),
                    dag.chirality(result),
                    dag.identity_rank(i),
                ))
                .or_insert(next);
            (group, factor)
        })
        .collect()
}

/// Ranks of `values` in their sorted order (Python's `str` order is the
/// code-point order, which UTF-8 byte order preserves).
fn ranks(values: &[String]) -> Vec<u32> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|a, b| values[*a].cmp(&values[*b]));
    let mut rank = vec![0u32; values.len()];
    let mut next = 0u32;
    for (position, index) in order.iter().enumerate() {
        if position > 0 && values[order[position - 1]] != values[*index] {
            next += 1;
        }
        rank[*index] = next;
    }
    rank
}

struct BuilderView<'a> {
    input: &'a DagInput,
    dag: &'a Dag,
}

impl ReuseDag for BuilderView<'_> {
    fn current_count(&self) -> usize {
        self.dag.currents.len()
    }
    fn interaction_count(&self) -> usize {
        self.dag.interactions.len()
    }
    fn mask(&self, current: usize) -> u64 {
        self.dag.currents[current].key.mask
    }
    fn is_source(&self, current: usize) -> bool {
        self.dag.currents[current].is_source
    }
    fn source_state(&self, current: usize) -> (i32, i32) {
        let current = &self.dag.currents[current];
        (current.leg, current.helicity)
    }
    fn pdg(&self, current: usize) -> i64 {
        self.input.particles[self.dag.currents[current].key.particle as usize].pdg
    }
    fn chirality(&self, current: usize) -> i8 {
        self.dag.currents[current].key.chirality
    }
    fn kernel(&self, interaction: usize) -> usize {
        self.dag.interactions[interaction].vertex as usize
    }
    fn left(&self, interaction: usize) -> u32 {
        self.dag.interactions[interaction].left
    }
    fn right(&self, interaction: usize) -> u32 {
        self.dag.interactions[interaction].right
    }
    fn result(&self, interaction: usize) -> u32 {
        self.dag.interactions[interaction].result
    }
    fn colour_weight(&self, interaction: usize) -> Complex {
        self.dag.interactions[interaction].colour_weight
    }
    fn identity_rank(&self, interaction: usize) -> u32 {
        self.input.identity_ranks[self.dag.interactions[interaction].coupling_identity as usize]
    }
}

/// `assign_recursive_current_evaluation_reuse` on the builder's DAG. The
/// compile-time groups and factors stay when the rewrite equals them (the
/// oracle then returns its input).
pub fn assign_reuse(input: &DagInput, dag: &mut Dag) -> DagResult<()> {
    if dag.interactions.is_empty() {
        return Ok(());
    }
    // `_current_evaluation_contract`: every `CurrentKey` field but the
    // colour state (labels, momentum, dimension, quantum numbers and
    // auxiliary kind are functions of these).
    let mut ids = FxMap::default();
    let contracts: Vec<u32> = dag
        .currents
        .iter()
        .map(|current| {
            let key = &current.key;
            let next = ids.len() as u32;
            *ids.entry((
                key.particle,
                key.mask,
                key.word,
                key.ancestry,
                key.chirality,
                key.spin,
                key.flavour,
                key.orders,
                current.is_source,
            ))
            .or_insert(next)
        })
        .collect();
    let class_ranks = ranks(&input.classes);
    let kernels: Vec<Kernel> = input
        .vertices
        .iter()
        .map(|vertex| Kernel::of(vertex, class_ranks[vertex.class as usize]))
        .collect();
    let view = BuilderView { input, dag };
    let equivalences = derive(&view, &contracts, &kernels, true)?;
    let rewritten = rewrite(&view, &kernels, &equivalences);
    if rewritten
        .iter()
        .zip(&dag.interactions)
        .all(|((group, factor), old)| *group == old.group && pair_eq(*factor, old.factor))
    {
        return Ok(());
    }
    for ((group, factor), interaction) in rewritten.into_iter().zip(&mut dag.interactions) {
        interaction.group = group;
        interaction.factor = factor;
    }
    Ok(())
}

/// Tables of the derivation on a [`ColumnarDag`] (`dag_input.reuse_tables`):
/// per vertex kind its kernel, per interaction its coupling identity rank.
pub struct ReuseTables {
    pub kernels: Vec<Kernel>,
    pub kernel_by_kind: HashMap<i64, usize>,
    pub identity_ranks: Vec<u32>,
}

pub const REUSE_SCHEMA: &str = "pyamplicol.loop-v2.dag-reuse/1";

/// `(kind, class, factor re bits, im bits, swap, exchange bits)`.
type RawKernel = (i64, String, u64, u64, bool, Option<[u64; 2]>);

#[derive(serde::Deserialize)]
struct RawReuseTables {
    schema: String,
    kernels: Vec<RawKernel>,
    identity_ranks: Vec<u32>,
}

impl ReuseTables {
    pub fn from_json(text: &str) -> DagResult<Self> {
        let raw: RawReuseTables = serde_json::from_str(text)
            .map_err(|error| DagError::Malformed(format!("reuse tables: {error}")))?;
        if raw.schema != REUSE_SCHEMA {
            return Err(DagError::Malformed(format!(
                "reuse schema {:?}",
                raw.schema
            )));
        }
        let classes: Vec<String> = raw.kernels.iter().map(|row| row.1.clone()).collect();
        let class_ranks = ranks(&classes);
        let mut kernel_by_kind = HashMap::new();
        let mut kernels = Vec::with_capacity(raw.kernels.len());
        for (row, (kind, _class, re, im, swap, exchange)) in raw.kernels.into_iter().enumerate() {
            if kernel_by_kind.insert(kind, row).is_some() {
                return Err(DagError::Malformed(format!("kernel of kind {kind} repeated")));
            }
            let bits = |pair: [u64; 2]| [f64::from_bits(pair[0]), f64::from_bits(pair[1])];
            kernels.push(Kernel {
                class_rank: class_ranks[row],
                factor: bits([re, im]),
                swap,
                exchange: exchange.map(bits),
            });
        }
        Ok(Self {
            kernels,
            kernel_by_kind,
            identity_ranks: raw.identity_ranks,
        })
    }
}

struct ColumnarView<'a> {
    dag: &'a ColumnarDag,
    kernel: Vec<usize>,
    identity_ranks: &'a [u32],
}

impl ReuseDag for ColumnarView<'_> {
    fn current_count(&self) -> usize {
        self.dag.current_count()
    }
    fn interaction_count(&self) -> usize {
        self.dag.interaction_count()
    }
    fn mask(&self, current: usize) -> u64 {
        self.dag.c_mask[current]
    }
    fn is_source(&self, current: usize) -> bool {
        self.dag.c_source[current] == 1
    }
    fn source_state(&self, current: usize) -> (i32, i32) {
        (self.dag.c_leg[current], self.dag.c_helicity[current])
    }
    fn pdg(&self, current: usize) -> i64 {
        self.dag.tables.species[self.dag.c_species[current] as usize].pdg
    }
    fn chirality(&self, current: usize) -> i8 {
        self.dag.c_chirality[current]
    }
    fn kernel(&self, interaction: usize) -> usize {
        self.kernel[interaction]
    }
    fn left(&self, interaction: usize) -> u32 {
        self.dag.i_left[interaction]
    }
    fn right(&self, interaction: usize) -> u32 {
        self.dag.i_right[interaction]
    }
    fn result(&self, interaction: usize) -> u32 {
        self.dag.i_result[interaction]
    }
    fn colour_weight(&self, interaction: usize) -> Complex {
        self.dag.i_colour_weight[interaction]
    }
    fn identity_rank(&self, interaction: usize) -> u32 {
        self.identity_ranks[interaction]
    }
}

impl ColumnarDag {
    /// The value-equivalence derivation and the rewritten DAG
    /// (`_derive_current_value_equivalences` with `restricted` as in
    /// [`derive`], then `_rewrite_interaction_evaluation_reuse`) of a DAG
    /// read from columns, e.g. a Python `GenericDAG` (the oracle fixtures).
    pub fn reuse(
        &self,
        tables: &ReuseTables,
        restricted: bool,
    ) -> DagResult<(Vec<Equivalence>, ColumnarDag)> {
        if tables.identity_ranks.len() != self.interaction_count() {
            return Err(DagError::Malformed(
                "one coupling identity per interaction".into(),
            ));
        }
        let kernel = self
            .i_vertex
            .iter()
            .map(|vertex| {
                let kind = self.tables.vertices[*vertex as usize].kind;
                tables.kernel_by_kind.get(&kind).copied().ok_or_else(|| {
                    DagError::Malformed(format!("no kernel equivalence of vertex kind {kind}"))
                })
            })
            .collect::<DagResult<Vec<_>>>()?;
        let species = canonical_ids(&self.tables.species);
        let flavours = canonical_ids(&self.tables.flavours);
        let orders = canonical_ids(&self.tables.orders);
        let mut ids = FxMap::default();
        let contracts: Vec<u32> = (0..self.current_count())
            .map(|id| {
                let next = ids.len() as u32;
                *ids.entry((
                    species[self.c_species[id] as usize],
                    self.c_mask[id],
                    self.c_momentum[id],
                    self.word(id),
                    self.c_ancestry[id],
                    self.c_chirality[id],
                    self.c_spin[id],
                    flavours[self.c_flavour[id] as usize],
                    orders[self.c_orders[id] as usize],
                    self.c_dimension[id],
                    self.c_source[id],
                ))
                .or_insert(next)
            })
            .collect();
        let view = ColumnarView {
            dag: self,
            kernel,
            identity_ranks: &tables.identity_ranks,
        };
        let equivalences = derive(&view, &contracts, &tables.kernels, restricted)?;
        let rewritten = rewrite(&view, &tables.kernels, &equivalences);
        let mut out = self.clone();
        if !rewritten
            .iter()
            .zip(self.i_group.iter().zip(&self.i_factor))
            .all(|((group, factor), (old_group, old_factor))| {
                group == old_group && pair_eq(*factor, *old_factor)
            })
        {
            for (id, (group, factor)) in rewritten.into_iter().enumerate() {
                out.i_group[id] = group;
                out.i_factor[id] = factor;
            }
        }
        Ok((equivalences, out))
    }
}
