// SPDX-License-Identifier: 0BSD

//! Exact colour cells of the loop engine's seed tables (WP3.1 package C,
//! `workspace/reports/engine-v2/wp31/RUST-MIGRATION.md` §3.C).
//!
//! The Python stage (`loops/v2/colour.py`: `seed_cells`, `_PrimitiveCells`,
//! `contract_colour`, `face_count`, `BornBasis.gram`, `project`) stays the
//! oracle; every cell here equals its Python twin exactly.
//!
//! - **Literal tensors** ([`Structure`], [`Term`], [`Tensor`]) are the tree
//!   engine's `color.connections` records: open strings `q -> word -> qbar`
//!   and cyclic traces, exact complex rational coefficients, an `N_c` power
//!   per term and a `1/sqrt(2)` normalisation power per tensor.
//! - **Overlaps** `<left|right>` ([`contract`]) glue both tensors into a
//!   product of adjoint traces ([`overlap_traces`]) and count it exactly:
//!   the subset sum of [`super::cycles`] up to its label limit, an exact
//!   Fierz recursion beyond it (no cap).
//! - **Seed cells** `K[c, b] = <b|kappa_c>` ([`CellEngine`]): adjoint rings
//!   through their U(N_c) expansion and face counts, fundamental traces and
//!   quark rings (given as their reduced canonical tensors) by exact
//!   contraction, generic classes term-factored. Overlaps depend on labels
//!   only through relabellings of the summed adjoint labels, so they are
//!   cached on relabelled bras, as in Python. Missing overlaps are evaluated
//!   in parallel and assembled in input order: the output does not depend on
//!   the thread count.
//! - **Projection** ([`project`], [`project_records`]): the plan's two-level
//!   seed records, byte-identical to `plan_export._Exporter.seed_records`.
//! - **Degrees** ([`OrderedTable::degree`], [`overlap_degrees`],
//!   [`AdjointOrbits`], [`live_classes`]): the structural LC/NLC degree
//!   machinery of `assemble.LoopSlice.degrees`, `_prune_colour` and
//!   `_AdjointOrbits`.
//!
//! No Symbolica state is touched (exact numbers are numerica's `Rational`),
//! so nothing here needs a licence.

use super::cycles;
use super::{FastMap, FastSet};
use crate::loop_v2::dress::parse_rational;
use crate::{RusticolError, RusticolResult};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, MutexGuard};
use symbolica::domains::integer::Integer;
use symbolica::domains::rational::Rational;

/// A colour label: a process leg label or an internal (summed) label.
pub type Label = i64;
/// A typed colour leg: (label, representation in {1, 3, -3, 8}).
pub type Leg = (Label, i32);
/// Exact Laurent coefficients in N_c, nonzero only.
pub type Laurent = BTreeMap<i32, Exact>;

/// The number of colours the projection substitutes (`contraction_types.NC`).
pub const NC: i64 = 3;
const ADJOINT: i32 = 8;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("colour cells: {}", message.into()))
}

// ------------------------------------------------------------------- numbers

/// An exact complex rational (`ExactColorCoefficient`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Exact {
    pub re: Rational,
    pub im: Rational,
}

impl Exact {
    pub fn zero() -> Self {
        Self {
            re: Rational::zero(),
            im: Rational::zero(),
        }
    }

    pub fn one() -> Self {
        Self::real(Rational::one())
    }

    pub fn real(re: Rational) -> Self {
        Self {
            re,
            im: Rational::zero(),
        }
    }

    pub fn integer(value: i64) -> Self {
        Self::real(Rational::from(value))
    }

    /// Parse the exact real and imaginary parts (`"3"`, `"-1/2"`).
    pub fn parse(re: &str, im: &str) -> RusticolResult<Self> {
        Ok(Self {
            re: parse_rational(re)?,
            im: parse_rational(im)?,
        })
    }

    pub fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }

    pub fn is_one(&self) -> bool {
        self.re.is_one() && self.im.is_zero()
    }

    pub fn conj(&self) -> Self {
        Self {
            re: self.re.clone(),
            im: self.im.clone().neg(),
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            re: &self.re + &other.re,
            im: &self.im + &other.im,
        }
    }

    pub fn add_assign(&mut self, other: &Self) {
        *self = self.add(other);
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self {
            re: &(&self.re * &other.re) - &(&self.im * &other.im),
            im: &(&self.re * &other.im) + &(&self.im * &other.re),
        }
    }

    pub fn scale(&self, factor: &Rational) -> Self {
        Self {
            re: &self.re * factor,
            im: &self.im * factor,
        }
    }

    /// `1 / self`, refused for zero.
    pub fn inv(&self) -> RusticolResult<Self> {
        let norm = &(&self.re * &self.re) + &(&self.im * &self.im);
        if norm.is_zero() {
            return Err(invalid("inverse of an exact zero"));
        }
        Ok(self.conj().scale(&norm.inv()))
    }

    /// The real and imaginary parts as `plan_export._fraction_text`.
    pub fn texts(&self) -> (String, String) {
        (fraction_text(&self.re), fraction_text(&self.im))
    }
}

/// `plan_export._fraction_text`: `"n"` for integers, else `"n/d"`.
pub fn fraction_text(value: &Rational) -> String {
    if value.is_integer() {
        value.numerator_ref().to_string()
    } else {
        format!("{}/{}", value.numerator_ref(), value.denominator_ref())
    }
}

fn nc_power(exponent: i32) -> Rational {
    let base = Rational::from(NC);
    if exponent >= 0 {
        base.pow(exponent as u64)
    } else {
        base.pow(u64::from(exponent.unsigned_abs())).inv()
    }
}

fn add_laurent(out: &mut Laurent, power: i32, value: Exact) {
    if value.is_zero() {
        return;
    }
    match out.entry(power) {
        std::collections::btree_map::Entry::Vacant(slot) => {
            slot.insert(value);
        }
        std::collections::btree_map::Entry::Occupied(mut slot) => {
            let sum = slot.get().add(&value);
            if sum.is_zero() {
                slot.remove();
            } else {
                *slot.get_mut() = sum;
            }
        }
    }
}

/// The value of a Laurent cell at N_c = 3 (`evaluate_color_nc_terms`).
pub fn at_three(cell: &Laurent) -> Exact {
    let mut out = Exact::zero();
    for (&power, value) in cell {
        out.add_assign(&value.scale(&nc_power(power)));
    }
    out
}

// ------------------------------------------------------------------ tensors

/// `OpenColorString(q, word, qbar)`: the block-word string q -> word -> qbar.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpenString {
    pub q: Label,
    pub word: Vec<Label>,
    pub qbar: Label,
}

/// `ColorTensor`: an unordered product of open strings and cyclic traces,
/// canonical (strings sorted; traces in their least rotation, sorted).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Structure {
    pub strings: Vec<OpenString>,
    pub traces: Vec<Vec<Label>>,
}

/// The least rotation of a cyclic trace (`color.connections._canonical_trace`).
pub fn canonical_trace(trace: &[Label]) -> Vec<Label> {
    let n = trace.len();
    let mut best: Option<Vec<Label>> = None;
    for start in 0..n {
        let rotated: Vec<Label> = trace[start..]
            .iter()
            .chain(&trace[..start])
            .copied()
            .collect();
        if best.as_ref().is_none_or(|current| rotated < *current) {
            best = Some(rotated);
        }
    }
    best.unwrap_or_default()
}

impl Structure {
    pub fn new(mut strings: Vec<OpenString>, traces: Vec<Vec<Label>>) -> Self {
        strings.sort();
        let mut traces: Vec<Vec<Label>> = traces.iter().map(|t| canonical_trace(t)).collect();
        traces.sort();
        Self { strings, traces }
    }

    /// The structure with its adjoint labels renamed (fundamental ones kept).
    pub fn relabelled(&self, rename: &impl Fn(Label) -> Label) -> Self {
        Self::new(
            self.strings
                .iter()
                .map(|s| OpenString {
                    q: s.q,
                    word: s.word.iter().map(|&a| rename(a)).collect(),
                    qbar: s.qbar,
                })
                .collect(),
            self.traces
                .iter()
                .map(|t| t.iter().map(|&a| rename(a)).collect())
                .collect(),
        )
    }

    /// Adjoint labels in reading order (strings, then traces, as stored).
    pub fn adjoint_reading(&self) -> impl Iterator<Item = Label> + '_ {
        self.strings
            .iter()
            .flat_map(|s| s.word.iter().copied())
            .chain(self.traces.iter().flatten().copied())
    }
}

/// `TensorTerm`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Term {
    pub coefficient: Exact,
    pub structure: Structure,
    pub nc_power: i32,
}

/// `ConnectedColorTensor`: typed legs (sorted), combined terms and the
/// inverse sqrt(2) normalisation power.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Tensor {
    pub legs: Vec<Leg>,
    pub terms: Vec<Term>,
    pub normalization: u32,
}

/// `color.connections._combine_terms`: single-letter traces vanish, empty
/// traces are N_c, equal (structure, power) terms merge, zeros drop; terms
/// in (structure, power) order.
pub fn combine_terms(terms: impl IntoIterator<Item = Term>) -> Vec<Term> {
    let mut combined: BTreeMap<(Structure, i32), Exact> = BTreeMap::new();
    for term in terms {
        if term.structure.traces.iter().any(|t| t.len() == 1) {
            continue;
        }
        let empty = term
            .structure
            .traces
            .iter()
            .filter(|t| t.is_empty())
            .count() as i32;
        let structure = if empty > 0 {
            Structure {
                strings: term.structure.strings,
                traces: term
                    .structure
                    .traces
                    .into_iter()
                    .filter(|t| !t.is_empty())
                    .collect(),
            }
        } else {
            term.structure
        };
        combined
            .entry((structure, term.nc_power + empty))
            .and_modify(|value| value.add_assign(&term.coefficient))
            .or_insert(term.coefficient);
    }
    combined
        .into_iter()
        .filter(|(_, value)| !value.is_zero())
        .map(|((structure, nc_power), coefficient)| Term {
            coefficient,
            structure,
            nc_power,
        })
        .collect()
}

impl Tensor {
    /// A connected tensor: legs sorted, terms combined.
    pub fn connected(mut legs: Vec<Leg>, terms: Vec<Term>, normalization: u32) -> Self {
        legs.sort();
        Self {
            legs,
            terms: combine_terms(terms),
            normalization,
        }
    }

    /// A single structure with coefficient one (`assemble._single`).
    pub fn single(legs: &[Leg], structure: Structure) -> Self {
        Self::connected(
            legs.to_vec(),
            vec![Term {
                coefficient: Exact::one(),
                structure,
                nc_power: 0,
            }],
            0,
        )
    }

    /// The tensor with its adjoint labels renamed (`colour._relabelled`):
    /// legs kept, terms re-canonicalised (a bijection merges none).
    pub fn relabelled(&self, rename: &impl Fn(Label) -> Label) -> Self {
        let mut terms: Vec<Term> = self
            .terms
            .iter()
            .map(|term| Term {
                coefficient: term.coefficient.clone(),
                structure: term.structure.relabelled(rename),
                nc_power: term.nc_power,
            })
            .collect();
        terms.sort_by(|a, b| (&a.structure, a.nc_power).cmp(&(&b.structure, b.nc_power)));
        Self {
            legs: self.legs.clone(),
            terms,
            normalization: self.normalization,
        }
    }

    pub fn adjoint_labels(&self) -> Vec<Label> {
        self.legs
            .iter()
            .filter(|leg| leg.1 == ADJOINT)
            .map(|leg| leg.0)
            .collect()
    }
}

// ----------------------------------------------------------------- overlaps

/// `color.connections.overlap_traces`: the adjoint traces whose product is
/// `<left|right>` (the traces of `right`, the reversed traces of `left` and
/// one trace per closed index loop of the open strings glued end to end).
/// A string of either side left unglued (structures on different quark
/// legs) is refused, where Python drops a bra string silently (rule 2).
pub fn overlap_traces(left: &Structure, right: &Structure) -> RusticolResult<Vec<Vec<Label>>> {
    let mut traces: Vec<Vec<Label>> = right.traces.clone();
    traces.extend(
        left.traces
            .iter()
            .map(|t| t.iter().rev().copied().collect::<Vec<_>>()),
    );
    if right.strings.is_empty() && left.strings.is_empty() {
        return Ok(traces);
    }
    let right_by_q: BTreeMap<Label, &OpenString> = right.strings.iter().map(|s| (s.q, s)).collect();
    let left_by_qbar: FastMap<Label, &OpenString> =
        left.strings.iter().map(|s| (s.qbar, s)).collect();
    if left.strings.len() != right.strings.len() {
        return Err(invalid(
            "a bra string is not glued to a ket string (different quark legs)",
        ));
    }
    let mut visited: FastSet<Label> = FastSet::default();
    for &start in right_by_q.keys() {
        if visited.contains(&start) {
            continue;
        }
        let mut current = start;
        let mut trace = Vec::new();
        while visited.insert(current) {
            let r = right_by_q.get(&current).ok_or_else(|| {
                invalid(format!("fundamental label {current} is not a ket string"))
            })?;
            let l = left_by_qbar.get(&r.qbar).ok_or_else(|| {
                invalid(format!(
                    "antifundamental label {} is not a bra string",
                    r.qbar
                ))
            })?;
            trace.extend(&r.word);
            trace.extend(l.word.iter().rev());
            current = l.q;
        }
        traces.push(trace);
    }
    Ok(traces)
}

/// The exact Laurent coefficients of a product of adjoint traces (every
/// label twice): the subset sum of [`cycles::trace_overlap`] up to its label
/// limit, the Fierz recursion beyond it.
pub fn trace_laurent(traces: &[Vec<Label>]) -> RusticolResult<Vec<(i32, Integer)>> {
    let labels: FastSet<Label> = traces.iter().flatten().copied().collect();
    if labels.len() <= cycles::MAX_LABELS {
        return Ok(cycles::trace_overlap(traces)?
            .into_iter()
            .map(|(power, coefficient)| (power, Integer::from(coefficient)))
            .collect());
    }
    fierz_laurent(traces)
}

/// A trace product reduced by the SU(N_c) Fierz identity, label by label:
/// `Tr(a X a Y) = Tr(X) Tr(Y) - Tr(X Y)/N_c` and
/// `Tr(a X) Tr(a Y) = Tr(X Y) - Tr(X) Tr(Y)/N_c`; `Tr(tau^a) = 0` and an
/// empty trace is N_c. Products are memoised on their canonical form.
pub fn fierz_laurent(traces: &[Vec<Label>]) -> RusticolResult<Vec<(i32, Integer)>> {
    let mut counts: FastMap<Label, usize> = FastMap::default();
    for &label in traces.iter().flatten() {
        *counts.entry(label).or_insert(0) += 1;
    }
    if counts.values().any(|&count| count != 2) {
        return Err(invalid("a trace label does not occur exactly twice"));
    }
    let mut memo: FastMap<Vec<Vec<Label>>, Arc<BTreeMap<i32, Integer>>> = FastMap::default();
    let out = match fierz_normal(traces.to_vec()) {
        None => BTreeMap::new(),
        Some((power, key)) => fierz_shift(&fierz_reduce(key, &mut memo), power),
    };
    Ok(out.into_iter().filter(|(_, c)| !c.is_zero()).collect())
}

/// (N_c power of the empty traces, canonical non-empty traces), or None for
/// a vanishing product (a single-letter trace).
fn fierz_normal(traces: Vec<Vec<Label>>) -> Option<(i32, Vec<Vec<Label>>)> {
    let mut power = 0;
    let mut out = Vec::with_capacity(traces.len());
    for trace in traces {
        match trace.len() {
            0 => power += 1,
            1 => return None,
            _ => out.push(canonical_trace(&trace)),
        }
    }
    out.sort();
    Some((power, out))
}

fn fierz_shift(map: &BTreeMap<i32, Integer>, shift: i32) -> BTreeMap<i32, Integer> {
    map.iter().map(|(&p, c)| (p + shift, c.clone())).collect()
}

fn fierz_add(
    out: &mut BTreeMap<i32, Integer>,
    part: Option<(i32, Vec<Vec<Label>>)>,
    shift: i32,
    sign: i64,
    memo: &mut FastMap<Vec<Vec<Label>>, Arc<BTreeMap<i32, Integer>>>,
) {
    let Some((power, key)) = part else { return };
    let reduced = fierz_reduce(key, memo);
    for (&p, c) in reduced.iter() {
        let entry = out.entry(p + power + shift).or_insert_with(Integer::zero);
        let term = if sign < 0 { -c.clone() } else { c.clone() };
        *entry = &*entry + &term;
    }
}

fn fierz_reduce(
    key: Vec<Vec<Label>>,
    memo: &mut FastMap<Vec<Vec<Label>>, Arc<BTreeMap<i32, Integer>>>,
) -> Arc<BTreeMap<i32, Integer>> {
    if key.is_empty() {
        return Arc::new(BTreeMap::from([(0, Integer::one())]));
    }
    if let Some(found) = memo.get(&key) {
        return found.clone();
    }
    let first = &key[0];
    let label = first[0];
    let mut out: BTreeMap<i32, Integer> = BTreeMap::new();
    if let Some(second) = first[1..].iter().position(|&l| l == label) {
        // Tr(a Y a X) = Tr(Y) Tr(X) - Tr(Y X) / N_c.
        let second = second + 1;
        let y: Vec<Label> = first[1..second].to_vec();
        let x: Vec<Label> = first[second + 1..].to_vec();
        let rest = &key[1..];
        let mut split: Vec<Vec<Label>> = rest.to_vec();
        split.push(y.clone());
        split.push(x.clone());
        fierz_add(&mut out, fierz_normal(split), 0, 1, memo);
        let mut joined: Vec<Vec<Label>> = rest.to_vec();
        joined.push(y.into_iter().chain(x).collect());
        fierz_add(&mut out, fierz_normal(joined), -1, -1, memo);
    } else {
        // Tr(a X) Tr(a Y) = Tr(X Y) - Tr(X) Tr(Y) / N_c.
        let (t, position) = key
            .iter()
            .enumerate()
            .skip(1)
            .find_map(|(t, trace)| trace.iter().position(|&l| l == label).map(|p| (t, p)))
            .expect("every label occurs twice");
        let x: Vec<Label> = first[1..].to_vec();
        let other = &key[t];
        let y: Vec<Label> = other[position + 1..]
            .iter()
            .chain(&other[..position])
            .copied()
            .collect();
        let rest: Vec<Vec<Label>> = key
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != 0 && i != t)
            .map(|(_, trace)| trace.clone())
            .collect();
        let mut joined = rest.clone();
        joined.push(x.iter().chain(&y).copied().collect());
        fierz_add(&mut out, fierz_normal(joined), 0, 1, memo);
        let mut split = rest;
        split.push(x);
        split.push(y);
        fierz_add(&mut out, fierz_normal(split), -1, -1, memo);
    }
    out.retain(|_, c| !c.is_zero());
    let out = Arc::new(out);
    memo.insert(key, out.clone());
    out
}

/// `colour.face_count`: the U(N_c) index loops of `<left|right>` (its N_c
/// power without U(1) parts).
pub fn face_count(left: &Structure, right: &Structure) -> RusticolResult<i32> {
    let traces = overlap_traces(left, right)?;
    let mut faces = 0i32;
    let mut following = Vec::new();
    let mut label_at = Vec::new();
    for trace in &traces {
        if trace.is_empty() {
            faces += 1;
            continue;
        }
        let start = following.len();
        for (i, &label) in trace.iter().enumerate() {
            following.push(start + (i + 1) % trace.len());
            label_at.push(label);
        }
    }
    let mut first: FastMap<Label, usize> = FastMap::default();
    let mut partner = vec![usize::MAX; label_at.len()];
    for (position, &label) in label_at.iter().enumerate() {
        match first.get(&label) {
            None => {
                first.insert(label, position);
            }
            Some(&other) => {
                if partner[other] != usize::MAX {
                    return Err(invalid("an adjoint label is not contracted exactly twice"));
                }
                partner[other] = position;
                partner[position] = other;
            }
        }
    }
    if partner.contains(&usize::MAX) {
        return Err(invalid("an adjoint label is not contracted exactly twice"));
    }
    let mut visited = vec![false; label_at.len()];
    for start in 0..label_at.len() {
        if visited[start] {
            continue;
        }
        faces += 1;
        let mut current = start;
        while !visited[current] {
            visited[current] = true;
            current = following[partner[current]];
        }
    }
    Ok(faces)
}

/// [`contract`] of many (bra, ket) pairs, in parallel, in input order.
pub fn contract_many(pairs: &[(Tensor, Tensor)]) -> RusticolResult<Vec<Laurent>> {
    pairs.par_iter().map(|(a, b)| contract(a, b)).collect()
}

/// Run `work` on a pool of `threads` workers (None: rayon's global pool).
/// Every output of this module is independent of the thread count.
pub fn with_threads<T: Send>(
    threads: Option<usize>,
    work: impl FnOnce() -> RusticolResult<T> + Send,
) -> RusticolResult<RusticolResult<T>> {
    match threads {
        None => Ok(work()),
        Some(threads) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| invalid(e.to_string()))?;
            Ok(pool.install(work))
        }
    }
}

/// `colour.contract_colour`: the exact Laurent coefficients of
/// `left.conjugate() * right`.
pub fn contract(left: &Tensor, right: &Tensor) -> RusticolResult<Laurent> {
    if left.legs != right.legs {
        return Err(invalid(
            "bra and ket connections must have identical typed output legs",
        ));
    }
    let power = left.normalization + right.normalization;
    if power % 2 == 1 {
        return Err(invalid(
            "bra/ket normalization has an odd inverse sqrt(2) power",
        ));
    }
    let normalization = Rational::new(Integer::one(), Integer::from(2).pow((power / 2) as u64));
    let mut out = Laurent::new();
    for a in &left.terms {
        for b in &right.terms {
            let traces = overlap_traces(&a.structure, &b.structure)?;
            let coefficient = a
                .coefficient
                .conj()
                .mul(&b.coefficient)
                .scale(&normalization);
            for (nc, factor) in trace_laurent(&traces)? {
                add_laurent(
                    &mut out,
                    nc + a.nc_power + b.nc_power,
                    coefficient.scale(&Rational::from(factor)),
                );
            }
        }
    }
    Ok(out)
}

/// `color.connections._emit`: emit gluon `emitted` from the leg `emitter`
/// of a structure, as (sign, image) terms.
pub fn emit(
    structure: &Structure,
    emitter: Label,
    emitted: Label,
) -> RusticolResult<Vec<(i64, Structure)>> {
    let both = |before: &[Label], after: &[Label]| -> [(i64, Vec<Label>); 2] {
        let mut forward = before.to_vec();
        forward.extend([emitter, emitted]);
        forward.extend(after);
        let mut backward = before.to_vec();
        backward.extend([emitted, emitter]);
        backward.extend(after);
        [(1, forward), (-1, backward)]
    };
    for (index, string) in structure.strings.iter().enumerate() {
        let replace = |word: Vec<Label>| {
            let mut strings = structure.strings.clone();
            strings[index] = OpenString {
                q: string.q,
                word,
                qbar: string.qbar,
            };
            Structure::new(strings, structure.traces.clone())
        };
        if emitter == string.q {
            let mut word = vec![emitted];
            word.extend(&string.word);
            return Ok(vec![(1, replace(word))]);
        }
        if emitter == string.qbar {
            let mut word = string.word.clone();
            word.push(emitted);
            return Ok(vec![(-1, replace(word))]);
        }
        if let Some(position) = string.word.iter().position(|&a| a == emitter) {
            return Ok(both(&string.word[..position], &string.word[position + 1..])
                .into_iter()
                .map(|(sign, word)| (sign, replace(word)))
                .collect());
        }
    }
    for (index, trace) in structure.traces.iter().enumerate() {
        if let Some(position) = trace.iter().position(|&a| a == emitter) {
            return Ok(both(&trace[..position], &trace[position + 1..])
                .into_iter()
                .map(|(sign, word)| {
                    let mut traces = structure.traces.clone();
                    traces[index] = word;
                    (sign, Structure::new(structure.strings.clone(), traces))
                })
                .collect());
        }
    }
    Err(invalid(format!("emitter {emitter} has no tensor index")))
}

/// A gluon emission `EmitGluon(emitter, emitted)` of a colour history:
/// T^emitted on the emitter's colour index.
pub type Emission = (Label, Label);

/// A colour history of gluon emissions: (Born side, loop side).
pub type History = (Vec<Emission>, Vec<Emission>);

/// `observables.connect` for gluon emissions (`apply_color_connection` per
/// term): each emission applied in order to every term, coefficients and
/// N_c powers kept, on the output legs (the emitted adjoint labels added)
/// with one more 1/sqrt(2) per emission.
pub fn connect(tensor: &Tensor, emissions: &[Emission]) -> RusticolResult<Tensor> {
    let mut legs = tensor.legs.clone();
    let mut terms = tensor.terms.clone();
    for &(emitter, emitted) in emissions {
        if !legs.iter().any(|leg| leg.0 == emitter && leg.1 != 1) {
            return Err(invalid(format!("emitter {emitter} is not a coloured leg")));
        }
        if legs.iter().any(|leg| leg.0 == emitted) {
            return Err(invalid(format!("emitted label {emitted} is already a leg")));
        }
        let mut next = Vec::with_capacity(2 * terms.len());
        for term in &terms {
            for (sign, structure) in emit(&term.structure, emitter, emitted)? {
                next.push(Term {
                    coefficient: term.coefficient.scale(&Rational::from(sign)),
                    structure,
                    nc_power: term.nc_power,
                });
            }
        }
        terms = combine_terms(next);
        legs.push((emitted, ADJOINT));
    }
    Ok(Tensor::connected(
        legs,
        terms,
        tensor.normalization + emissions.len() as u32,
    ))
}

/// The SU(N_c) completeness relation on the two occurrences of `label` in a
/// structure, with generators normalised to `Tr(tau^a tau^b) = delta^ab`:
/// `tau^a_ij tau^a_kl = delta_il delta_kj - delta_ij delta_kl / N_c`, as
/// (sign, N_c power, structure) terms. Words read in matrix order (open
/// strings q -> word -> qbar, traces cyclic); an empty trace (N_c) and a
/// single-letter one (zero) are left to `combine_terms`.
pub fn fierz_pair(
    structure: &Structure,
    label: Label,
) -> RusticolResult<Vec<(i64, i32, Structure)>> {
    // Occurrences: (in a string?, index, position).
    let mut found: Vec<(bool, usize, usize)> = Vec::with_capacity(2);
    for (s, string) in structure.strings.iter().enumerate() {
        for (p, &a) in string.word.iter().enumerate() {
            if a == label {
                found.push((true, s, p));
            }
        }
    }
    for (t, trace) in structure.traces.iter().enumerate() {
        for (p, &a) in trace.iter().enumerate() {
            if a == label {
                found.push((false, t, p));
            }
        }
    }
    if found.len() != 2 {
        return Err(invalid(format!(
            "label {label} occurs {} times, not twice",
            found.len()
        )));
    }
    // A trace rotated to start at position p, with that letter dropped.
    let rotated = |trace: &[Label], p: usize| -> Vec<Label> {
        trace[p + 1..].iter().chain(&trace[..p]).copied().collect()
    };
    let concat = |a: &[Label], b: &[Label]| -> Vec<Label> { a.iter().chain(b).copied().collect() };
    let traces_without = |drop: &[usize]| -> Vec<Vec<Label>> {
        structure
            .traces
            .iter()
            .enumerate()
            .filter(|(t, _)| !drop.contains(t))
            .map(|(_, trace)| trace.clone())
            .collect()
    };
    let strings_with = |replaced: &[(usize, OpenString)]| -> Vec<OpenString> {
        let mut strings = structure.strings.clone();
        for (s, string) in replaced {
            strings[*s] = string.clone();
        }
        strings
    };
    let (first, second) = (found[0], found[1]);
    Ok(match (first, second) {
        // Tr(a Y a X) = Tr(Y) Tr(X) - Tr(Y X)/N.
        ((false, t, p), (false, t2, _)) if t == t2 => {
            let rest = rotated(&structure.traces[t], p);
            let k = rest
                .iter()
                .position(|&a| a == label)
                .ok_or_else(|| invalid("trace occurrence lost"))?;
            let (y, x) = (&rest[..k], &rest[k + 1..]);
            let others = traces_without(&[t]);
            let mut split = others.clone();
            split.extend([y.to_vec(), x.to_vec()]);
            let mut joined = others;
            joined.push(concat(y, x));
            vec![
                (1, 0, Structure::new(structure.strings.clone(), split)),
                (-1, -1, Structure::new(structure.strings.clone(), joined)),
            ]
        }
        // Tr(a X) Tr(a Y) = Tr(X Y) - Tr(X) Tr(Y)/N.
        ((false, t, p), (false, t2, p2)) => {
            let x = rotated(&structure.traces[t], p);
            let y = rotated(&structure.traces[t2], p2);
            let others = traces_without(&[t, t2]);
            let mut joined = others.clone();
            joined.push(concat(&x, &y));
            let mut split = others;
            split.extend([x, y]);
            vec![
                (1, 0, Structure::new(structure.strings.clone(), joined)),
                (-1, -1, Structure::new(structure.strings.clone(), split)),
            ]
        }
        // (U a V a Z) = (U Z) Tr(V) - (U V Z)/N.
        ((true, s, p), (true, s2, p2)) if s == s2 => {
            let string = &structure.strings[s];
            let (u, v, z) = (
                &string.word[..p],
                &string.word[p + 1..p2],
                &string.word[p2 + 1..],
            );
            let outer = OpenString {
                q: string.q,
                word: concat(u, z),
                qbar: string.qbar,
            };
            let all = OpenString {
                q: string.q,
                word: concat(&concat(u, v), z),
                qbar: string.qbar,
            };
            let mut traces = structure.traces.clone();
            traces.push(v.to_vec());
            vec![
                (1, 0, Structure::new(strings_with(&[(s, outer)]), traces)),
                (
                    -1,
                    -1,
                    Structure::new(strings_with(&[(s, all)]), structure.traces.clone()),
                ),
            ]
        }
        // (U a Z) Tr(a Y) = (U Y Z) - (U Z) Tr(Y)/N.
        ((true, s, p), (false, t, p2)) => {
            let string = &structure.strings[s];
            let (u, z) = (&string.word[..p], &string.word[p + 1..]);
            let y = rotated(&structure.traces[t], p2);
            let inserted = OpenString {
                q: string.q,
                word: concat(&concat(u, &y), z),
                qbar: string.qbar,
            };
            let outer = OpenString {
                q: string.q,
                word: concat(u, z),
                qbar: string.qbar,
            };
            let others = traces_without(&[t]);
            let mut split = others.clone();
            split.push(y);
            vec![
                (1, 0, Structure::new(strings_with(&[(s, inserted)]), others)),
                (-1, -1, Structure::new(strings_with(&[(s, outer)]), split)),
            ]
        }
        // (U a Z)_{q1 qb1} (P a R)_{q2 qb2}
        //   = (U R)_{q1 qb2} (P Z)_{q2 qb1} - (U Z)_{q1 qb1} (P R)_{q2 qb2}/N.
        ((true, s, p), (true, s2, p2)) => {
            let (one, two) = (&structure.strings[s], &structure.strings[s2]);
            let (u, z) = (&one.word[..p], &one.word[p + 1..]);
            let (pp, r) = (&two.word[..p2], &two.word[p2 + 1..]);
            let crossed = [
                (
                    s,
                    OpenString {
                        q: one.q,
                        word: concat(u, r),
                        qbar: two.qbar,
                    },
                ),
                (
                    s2,
                    OpenString {
                        q: two.q,
                        word: concat(pp, z),
                        qbar: one.qbar,
                    },
                ),
            ];
            let kept = [
                (
                    s,
                    OpenString {
                        q: one.q,
                        word: concat(u, z),
                        qbar: one.qbar,
                    },
                ),
                (
                    s2,
                    OpenString {
                        q: two.q,
                        word: concat(pp, r),
                        qbar: two.qbar,
                    },
                ),
            ];
            vec![
                (
                    1,
                    0,
                    Structure::new(strings_with(&crossed), structure.traces.clone()),
                ),
                (
                    -1,
                    -1,
                    Structure::new(strings_with(&kept), structure.traces.clone()),
                ),
            ]
        }
        ((false, _, _), (true, _, _)) => {
            return Err(invalid("trace occurrence listed before a string one"));
        }
    })
}

/// `observables.absorb`: the adjoint of one gluon emission, sum_a T^a_e X_a
/// (the emitted label a of `tensor` contracted with a generator on its
/// emitter e: a fresh label emitted from e and identified with a, then
/// Fierz-reduced), so that <X|E Y> = <absorb(X, E)|Y> for every Y on the
/// emitter's legs; the emission's 1/sqrt(2) stays in the normalisation.
pub fn absorb(tensor: &Tensor, (emitter, emitted): Emission) -> RusticolResult<Tensor> {
    let legs: Vec<Leg> = tensor
        .legs
        .iter()
        .filter(|leg| leg.0 != emitted)
        .copied()
        .collect();
    if legs.len() + 1 != tensor.legs.len() {
        return Err(invalid(format!("absorbed label {emitted} is not a leg")));
    }
    if !legs.iter().any(|leg| leg.0 == emitter && leg.1 != 1) {
        return Err(invalid(format!("emitter {emitter} is not a coloured leg")));
    }
    let fresh = tensor.legs.iter().map(|leg| leg.0).min().unwrap_or(0) - 1;
    let join = |a: Label| if a == fresh { emitted } else { a };
    let mut terms = Vec::new();
    for term in &tensor.terms {
        for (sign, structure) in emit(&term.structure, emitter, fresh)? {
            for (fierz, power, reduced) in fierz_pair(&structure.relabelled(&join), emitted)? {
                terms.push(Term {
                    coefficient: term.coefficient.scale(&Rational::from(sign * fierz)),
                    structure: reduced,
                    nc_power: term.nc_power + power,
                });
            }
        }
    }
    Ok(Tensor::connected(legs, terms, tensor.normalization + 1))
}

/// `DirectedColour.operator`: ket^dagger bra |tensor> on the Born legs, so
/// that <bra(t)|ket(k)> = <operator(t)|k> for every k: the Born side's
/// emissions applied (`connect`), then the loop side's absorbed in reverse.
pub fn operator_image(
    tensor: &Tensor,
    born: &[Emission],
    loop_side: &[Emission],
) -> RusticolResult<Tensor> {
    let mut image = connect(tensor, born)?;
    for &emission in loop_side.iter().rev() {
        image = absorb(&image, emission)?;
    }
    Ok(image)
}

/// `tensor_overlap_degrees` with the bra sets given as literal structures
/// and the colour histories (Born side, loop side) whose operator images
/// form them (`DirectedColour.overlap` of a history whose steps are all
/// gluon emissions): per (history, bra set, ket set, offset) query. Each
/// distinct (history, structure) image is formed once, in parallel.
pub fn directed_overlap_degrees(
    legs: &[Leg],
    bras: &[Vec<(Structure, i32)>],
    kets: &[Vec<(Structure, i32)>],
    histories: &[History],
    queries: &[(usize, usize, usize, i32)],
) -> RusticolResult<Vec<Option<i32>>> {
    let mut legs = legs.to_vec();
    legs.sort();
    // The (history, bra set) pairs the queries need, as image sets.
    let mut set_ids: FastMap<(usize, usize), usize> = FastMap::default();
    let mut needed: Vec<(usize, usize)> = Vec::new();
    let mut plain: Vec<(usize, usize, i32)> = Vec::with_capacity(queries.len());
    for &(history, bra, ket, offset) in queries {
        if history >= histories.len() || bra >= bras.len() {
            return Err(invalid("history or bra key set out of range"));
        }
        let id = *set_ids.entry((history, bra)).or_insert_with(|| {
            needed.push((history, bra));
            needed.len() - 1
        });
        plain.push((id, ket, offset));
    }
    let image_sets: Vec<Vec<(Tensor, i32)>> = needed
        .par_iter()
        .map(|&(history, bra)| {
            let (born, loop_side) = &histories[history];
            bras[bra]
                .iter()
                .map(|(structure, p)| {
                    let literal = Tensor::single(&legs, structure.clone());
                    Ok((operator_image(&literal, born, loop_side)?, *p))
                })
                .collect::<RusticolResult<Vec<_>>>()
        })
        .collect::<RusticolResult<_>>()?;
    let ket_sets: Vec<Vec<(Tensor, i32)>> = kets
        .iter()
        .map(|set| {
            set.iter()
                .map(|(structure, q)| (Tensor::single(&legs, structure.clone()), *q))
                .collect()
        })
        .collect();
    tensor_overlap_degrees(&image_sets, &ket_sets, &plain)
}

// --------------------------------------------------------------- primitives

/// The kind of a certified primitive seed class (`colour.Primitive.kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PrimitiveKind {
    AdjointRing,
    FundamentalTrace,
}

/// The literal tensor of `tr(tau^w_1 ... tau^w_n)` (`Primitive.tensor` of a
/// fundamental trace): one reversed literal trace, N_c for the empty word,
/// zero for a single letter.
pub fn fundamental_trace_tensor(word: &[Label], legs: &[Leg]) -> RusticolResult<Tensor> {
    let distinct: FastSet<Label> = word.iter().copied().collect();
    if distinct.len() != word.len() {
        return Err(invalid(
            "a fundamental-trace primitive repeats a label (not a certified class)",
        ));
    }
    let trace: Vec<Label> = word.iter().rev().copied().collect();
    Ok(Tensor::connected(
        legs.to_vec(),
        vec![Term {
            coefficient: Exact::one(),
            structure: Structure::new(Vec::new(), vec![trace]),
            nc_power: 0,
        }],
        0,
    ))
}

/// The U(N_c) terms of an adjoint ring.
type UnTerms = Vec<(i64, Vec<Vec<Label>>, i32)>;

/// `Primitive.un_terms` of an adjoint ring: (sign, traces, N_c power) of
/// Tr_adj(prod ad tau^w_i) = sum_S (-1)^|not S| tr(tau^w_S) tr(tau^rev(w_not S)).
/// A ring whose 2^n subsets overflow the mask is refused (rule 2).
pub fn adjoint_un_terms(word: &[Label]) -> RusticolResult<UnTerms> {
    let n = word.len();
    let subsets = u32::try_from(n)
        .ok()
        .and_then(|n| 1u64.checked_shl(n))
        .ok_or_else(|| {
            invalid(format!(
                "an adjoint ring of {n} letters has too many subsets"
            ))
        })?;
    let mut out = Vec::new();
    out.try_reserve_exact(usize::try_from(subsets).map_err(|e| invalid(e.to_string()))?)
        .map_err(|error| invalid(format!("an adjoint ring of {n} letters: {error}")))?;
    for mask in 0u64..subsets {
        let inside: Vec<Label> = (0..n)
            .filter(|i| mask >> i & 1 == 1)
            .map(|i| word[i])
            .collect();
        let outside: Vec<Label> = (0..n)
            .rev()
            .filter(|i| mask >> i & 1 == 0)
            .map(|i| word[i])
            .collect();
        let sign = if (n - inside.len()) % 2 == 1 { -1 } else { 1 };
        let traces: Vec<Vec<Label>> = [inside, outside]
            .into_iter()
            .filter(|t| !t.is_empty())
            .collect();
        let power = 2 - traces.len() as i32;
        out.push((sign, traces, power));
    }
    Ok(out)
}

/// `_PrimitiveCells._overlap` of an adjoint ring: its U(N_c) expansion
/// counted by faces (adjoint rings are U(1)-free in every external label).
pub fn adjoint_ring_overlap(word: &[Label], bra: &Tensor) -> RusticolResult<Laurent> {
    // An adjoint ring carries no fundamental index, so its overlap with a
    // Born tensor holding open quark strings is not defined: refused (rule 2;
    // Python's `_PrimitiveCells._overlap` drops the strings' index loops,
    // recorded oracle defect 3).
    if bra.terms.iter().any(|t| !t.structure.strings.is_empty()) {
        return Err(invalid(
            "an adjoint-ring class against a Born tensor with open quark strings",
        ));
    }
    if bra.normalization % 2 == 1 {
        return Err(invalid("a Born tensor has an odd inverse sqrt(2) power"));
    }
    let scale = Rational::new(
        Integer::one(),
        Integer::from(2).pow((bra.normalization / 2) as u64),
    );
    let mut out = Laurent::new();
    for (sign, traces, power) in adjoint_un_terms(word)? {
        // Literal traces list tr(tau^w_k .. tau^w_1).
        let ket = Structure {
            strings: Vec::new(),
            traces: traces
                .iter()
                .map(|t| t.iter().rev().copied().collect())
                .collect(),
        };
        let factor = &scale * &Rational::from(sign);
        for term in &bra.terms {
            let exponent = power + term.nc_power + face_count(&term.structure, &ket)?;
            add_laurent(&mut out, exponent, term.coefficient.conj().scale(&factor));
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------- engine

/// One seed class of a table (`SeedTable.classes`).
#[derive(Clone, Debug)]
pub enum ClassInput {
    /// `Primitive("adjoint-ring", word)`.
    AdjointRing(Vec<Label>),
    /// `Primitive("fundamental-trace", word)`.
    FundamentalTrace(Vec<Label>),
    /// A quark ring: the registered canonical ring `ring`
    /// ([`CellEngine::register_ring`]) and the relabelling `forward` from the
    /// class's adjoint labels to the canonical ring's (`_ring_relabelling`).
    QuarkRing {
        ring: usize,
        forward: Vec<(Label, Label)>,
    },
    /// A generic colour vector (`ConnectedColorTensor`).
    Generic(Tensor),
}

/// One distinct overlap still to be evaluated.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Task {
    /// The canonical primitive word 1..n of `kind` against a relabelled bra.
    Primitive(PrimitiveKind, usize, usize),
    /// A primitive word against a bra that cannot be relabelled.
    Direct(PrimitiveKind, Vec<Label>, usize),
    /// A registered quark ring against a relabelled bra.
    Ring(usize, usize),
    /// A generic structure against basis element b.
    Structure(usize, usize),
}

/// How the cells of one class are formed from evaluated tasks.
enum Recipe {
    /// One task per basis element.
    Tasks(Vec<usize>),
    /// A generic class: per basis element b, sum_t coefficient_t
    /// N_c^power_t <b|structure_t>; the parts are (coefficient pattern id,
    /// [(power, structure row)]) and a structure row holds the task of
    /// every basis element.
    Generic(usize, Vec<(i32, usize)>),
}

/// The basis-independent caches of `_PrimitiveCells`: relabelled bras by
/// id, the registered canonical quark rings, and the primitive and ring
/// overlaps keyed by them. An engine and every engine rebased from it share
/// one instance both ways, as Python's `rebased` shares these dicts through
/// `copy`.
#[derive(Default)]
struct SharedCaches {
    /// Relabelled bras by id (keys of the caches).
    bras: Vec<Tensor>,
    bra_ids: FastMap<Tensor, usize>,
    /// Registered canonical quark rings (reduced tensors).
    rings: Vec<Tensor>,
    /// Primitives (kind, length, relabelled bra) and rings (ring,
    /// relabelled bra).
    primitive_cache: FastMap<(PrimitiveKind, usize, usize), Arc<Laurent>>,
    ring_cache: FastMap<(usize, usize), Arc<Laurent>>,
}

impl SharedCaches {
    fn bra_id(&mut self, bra: Tensor) -> usize {
        if let Some(&id) = self.bra_ids.get(&bra) {
            return id;
        }
        self.bras.push(bra.clone());
        self.bra_ids.insert(bra, self.bras.len() - 1);
        self.bras.len() - 1
    }
}

/// Sizes of the shared caches: (relabelled bras, registered rings, cached
/// primitive overlaps, cached ring overlaps).
pub type CacheSizes = (usize, usize, usize, usize);

/// Seed cells against one basis of literal tensors, with the relabelling
/// caches of `_PrimitiveCells`. [`CellEngine::rebased`] engines (another
/// basis on the same legs) share the basis-independent caches both ways;
/// the generic-structure overlaps and the relabelling index are per basis.
pub struct CellEngine {
    legs: Vec<Leg>,
    basis: Vec<Tensor>,
    shared: Arc<Mutex<SharedCaches>>,
    /// Generic structures and their overlaps per element of this basis.
    structures: Vec<Structure>,
    structure_ids: FastMap<Structure, usize>,
    structure_cache: FastMap<(usize, usize), Arc<Laurent>>,
    /// Bra ids of this basis per relabelling (`_relabellings`).
    relabelled: FastMap<Vec<(Label, Label)>, Vec<usize>>,
}

fn locked(shared: &Mutex<SharedCaches>) -> RusticolResult<MutexGuard<'_, SharedCaches>> {
    shared
        .lock()
        .map_err(|_| invalid("the shared cell caches are poisoned"))
}

impl CellEngine {
    pub fn new(legs: Vec<Leg>, basis: Vec<Tensor>) -> RusticolResult<Self> {
        let mut legs = legs;
        legs.sort();
        if basis.iter().any(|b| b.legs != legs) {
            return Err(invalid("a basis tensor is not on the basis legs"));
        }
        Ok(Self::on(legs, basis, Arc::default()))
    }

    fn on(legs: Vec<Leg>, basis: Vec<Tensor>, shared: Arc<Mutex<SharedCaches>>) -> Self {
        Self {
            legs,
            basis,
            shared,
            structures: Vec::new(),
            structure_ids: FastMap::default(),
            structure_cache: FastMap::default(),
            relabelled: FastMap::default(),
        }
    }

    pub fn legs(&self) -> &[Leg] {
        &self.legs
    }

    pub fn basis(&self) -> &[Tensor] {
        &self.basis
    }

    /// The engine against another basis on the same legs, sharing the
    /// basis-independent caches both ways (`_PrimitiveCells.rebased`).
    pub fn rebased(&self, basis: Vec<Tensor>) -> RusticolResult<Self> {
        if basis.iter().any(|b| b.legs != self.legs) {
            return Err(invalid("rebased primitive cells need the same colour legs"));
        }
        Ok(Self::on(self.legs.clone(), basis, Arc::clone(&self.shared)))
    }

    /// The engine against the operator images of this basis under a colour
    /// history whose steps are all gluon emissions ([`operator_image`],
    /// formed in parallel), sharing the caches (`DirectedColour._table`).
    pub fn directed(&self, born: &[Emission], loop_side: &[Emission]) -> RusticolResult<Self> {
        let images = self
            .basis
            .par_iter()
            .map(|b| operator_image(b, born, loop_side))
            .collect::<RusticolResult<Vec<_>>>()?;
        self.rebased(images)
    }

    /// The sizes of the shared caches ([`CacheSizes`]).
    pub fn cache_sizes(&self) -> RusticolResult<CacheSizes> {
        let caches = locked(&self.shared)?;
        Ok((
            caches.bras.len(),
            caches.rings.len(),
            caches.primitive_cache.len(),
            caches.ring_cache.len(),
        ))
    }

    /// Register a canonical quark ring by its reduced tensor; returns its id
    /// (shared with the rebased engines).
    pub fn register_ring(&self, tensor: Tensor) -> RusticolResult<usize> {
        if tensor.legs != self.legs {
            return Err(invalid("a quark ring is not on the basis legs"));
        }
        let mut caches = locked(&self.shared)?;
        caches.rings.push(tensor);
        Ok(caches.rings.len() - 1)
    }

    fn structure_id(&mut self, structure: &Structure) -> usize {
        if let Some(&id) = self.structure_ids.get(structure) {
            return id;
        }
        self.structures.push(structure.clone());
        self.structure_ids
            .insert(structure.clone(), self.structures.len() - 1);
        self.structures.len() - 1
    }

    /// `_relabel_tensor`: a pure-adjoint basis tensor with its trace labels
    /// renamed to the primitive's 1..n (legs renamed, singlets dropped), or
    /// None where a label is not mapped (or a leg is not adjoint/singlet).
    fn primitive_bra(&self, bra: &Tensor, rename: &FastMap<Label, Label>) -> Option<Tensor> {
        if bra.legs.iter().any(|leg| leg.1 != 1 && leg.1 != ADJOINT) {
            return None;
        }
        if bra.terms.iter().any(|t| !t.structure.strings.is_empty()) {
            return None;
        }
        let mut legs = Vec::new();
        for leg in &bra.legs {
            if leg.1 == ADJOINT {
                legs.push((*rename.get(&leg.0)?, ADJOINT));
            }
        }
        for term in &bra.terms {
            for &label in term.structure.traces.iter().flatten() {
                rename.get(&label)?;
            }
        }
        legs.sort();
        let mut terms: Vec<Term> = bra
            .terms
            .iter()
            .map(|term| Term {
                coefficient: term.coefficient.clone(),
                structure: Structure::new(
                    Vec::new(),
                    term.structure
                        .traces
                        .iter()
                        .map(|t| t.iter().map(|l| rename[l]).collect())
                        .collect(),
                ),
                nc_power: term.nc_power,
            })
            .collect();
        terms.sort_by(|a, b| (&a.structure, a.nc_power).cmp(&(&b.structure, b.nc_power)));
        Some(Tensor {
            legs,
            terms,
            normalization: bra.normalization,
        })
    }

    /// Bra ids of the basis relabelled by `forward` (`_relabelled_bras`).
    fn relabelled_bras(
        &mut self,
        caches: &mut SharedCaches,
        forward: &[(Label, Label)],
    ) -> Vec<usize> {
        let mut key = forward.to_vec();
        key.sort();
        if let Some(found) = self.relabelled.get(&key) {
            return found.clone();
        }
        let map: FastMap<Label, Label> = key.iter().copied().collect();
        let rename = |a: Label| *map.get(&a).unwrap_or(&a);
        let ids: Vec<usize> = (0..self.basis.len())
            .map(|b| {
                let bra = if map.is_empty() {
                    self.basis[b].clone()
                } else {
                    self.basis[b].relabelled(&rename)
                };
                caches.bra_id(bra)
            })
            .collect();
        self.relabelled.insert(key, ids.clone());
        ids
    }

    /// The cells `<b|kappa_c>` of `classes` against every basis tensor
    /// (`colour.seed_cells`), in input order. Missing overlaps are
    /// evaluated in parallel; caches persist across calls (and are shared
    /// with the rebased engines, which wait for each other's calls).
    ///
    /// A generic class is term-factored: each distinct term structure is
    /// contracted once per basis element (its structure row of tasks) and
    /// the class's cells are combined from those rows in parallel; equal
    /// cells of a class's combination share one value (the cells are
    /// values: sharing never shows in the output).
    pub fn cells(&mut self, classes: &[ClassInput]) -> RusticolResult<Vec<Vec<Arc<Laurent>>>> {
        let shared = Arc::clone(&self.shared);
        let mut guard = locked(&shared)?;
        let caches: &mut SharedCaches = &mut guard;
        let mut tasks: Vec<Task> = Vec::new();
        let mut task_ids: FastMap<Task, usize> = FastMap::default();
        let mut add_task = |task: Task, tasks: &mut Vec<Task>| -> usize {
            *task_ids.entry(task.clone()).or_insert_with(|| {
                tasks.push(task);
                tasks.len() - 1
            })
        };
        let basis_len = self.basis.len();
        // Generic classes: structure rows (structure id -> row index) and
        // the distinct coefficient patterns [(coefficient, power)].
        let mut structure_rows: Vec<Vec<usize>> = Vec::new();
        let mut structure_row_of: FastMap<usize, usize> = FastMap::default();
        let mut patterns: Vec<Vec<Exact>> = Vec::new();
        let mut pattern_ids: FastMap<Vec<Exact>, usize> = FastMap::default();
        let mut recipes: Vec<Recipe> = Vec::with_capacity(classes.len());
        for class in classes {
            match class {
                ClassInput::AdjointRing(word) | ClassInput::FundamentalTrace(word) => {
                    let kind = match class {
                        ClassInput::AdjointRing(_) => PrimitiveKind::AdjointRing,
                        _ => PrimitiveKind::FundamentalTrace,
                    };
                    let rename: FastMap<Label, Label> = word
                        .iter()
                        .enumerate()
                        .map(|(i, &label)| (label, i as Label + 1))
                        .collect();
                    let mut row = Vec::with_capacity(basis_len);
                    for b in 0..basis_len {
                        let task = match self.primitive_bra(&self.basis[b], &rename) {
                            Some(bra) => Task::Primitive(kind, word.len(), caches.bra_id(bra)),
                            None => Task::Direct(kind, word.clone(), b),
                        };
                        row.push(add_task(task, &mut tasks));
                    }
                    recipes.push(Recipe::Tasks(row));
                }
                ClassInput::QuarkRing { ring, forward } => {
                    if *ring >= caches.rings.len() {
                        return Err(invalid(format!("unregistered quark ring {ring}")));
                    }
                    let row = self
                        .relabelled_bras(caches, forward)
                        .into_iter()
                        .map(|bra| add_task(Task::Ring(*ring, bra), &mut tasks))
                        .collect();
                    recipes.push(Recipe::Tasks(row));
                }
                ClassInput::Generic(tensor) => {
                    if tensor.normalization != 0 {
                        return Err(invalid("a seed class carries an irrational sqrt(2) factor"));
                    }
                    if tensor.legs != self.legs {
                        return Err(invalid("a seed class is not on the basis legs"));
                    }
                    let mut pattern = Vec::with_capacity(tensor.terms.len());
                    let mut parts = Vec::with_capacity(tensor.terms.len());
                    for term in &tensor.terms {
                        let s = self.structure_id(&term.structure);
                        let row = *structure_row_of.entry(s).or_insert_with(|| {
                            structure_rows.push(
                                (0..basis_len)
                                    .map(|b| add_task(Task::Structure(s, b), &mut tasks))
                                    .collect(),
                            );
                            structure_rows.len() - 1
                        });
                        pattern.push(term.coefficient.clone());
                        parts.push((term.nc_power, row));
                    }
                    let pattern = match pattern_ids.get(&pattern) {
                        Some(&id) => id,
                        None => {
                            patterns.push(pattern.clone());
                            pattern_ids.insert(pattern, patterns.len() - 1);
                            patterns.len() - 1
                        }
                    };
                    recipes.push(Recipe::Generic(pattern, parts));
                }
            }
        }
        // Evaluate the tasks missing from the caches, in parallel.
        let mut results: Vec<Option<Arc<Laurent>>> =
            tasks.iter().map(|t| self.cached(caches, t)).collect();
        let missing: Vec<usize> = (0..tasks.len()).filter(|&t| results[t].is_none()).collect();
        let values: Vec<RusticolResult<Laurent>> = {
            let caches: &SharedCaches = caches;
            missing
                .par_iter()
                .map(|&t| self.evaluate(caches, &tasks[t]))
                .collect()
        };
        for (&t, value) in missing.iter().zip(values) {
            let value = Arc::new(value?);
            self.store(caches, &tasks[t], value.clone());
            results[t] = Some(value);
        }
        drop(guard);
        let results: Vec<Arc<Laurent>> = results
            .into_iter()
            .map(|value| value.expect("every task is evaluated"))
            .collect();
        // Equal task values share one value (`distinct[value_of[t]]`).
        let mut distinct: Vec<Arc<Laurent>> = Vec::new();
        let value_of: Vec<usize> = {
            let mut index: FastMap<&Laurent, usize> = FastMap::default();
            results
                .iter()
                .map(|value| {
                    *index.entry(value.as_ref()).or_insert_with(|| {
                        distinct.push(value.clone());
                        distinct.len() - 1
                    })
                })
                .collect()
        };
        drop(results);
        // Assemble the rows in parallel, in input order. A generic cell is
        // determined by its class's coefficient pattern and its parts'
        // (power, value): equal keys share one cell per worker.
        type CellKey = (usize, Vec<(i32, usize)>);
        Ok(recipes
            .into_par_iter()
            .map_init(
                FastMap::<CellKey, Arc<Laurent>>::default,
                |memo, recipe| match recipe {
                    Recipe::Tasks(row) => row
                        .into_iter()
                        .map(|t| distinct[value_of[t]].clone())
                        .collect(),
                    Recipe::Generic(pattern, parts) => {
                        let coefficients = &patterns[pattern];
                        (0..basis_len)
                            .map(|b| {
                                let key: CellKey = (
                                    pattern,
                                    parts
                                        .iter()
                                        .map(|&(power, row)| {
                                            (power, value_of[structure_rows[row][b]])
                                        })
                                        .collect(),
                                );
                                if let Some(found) = memo.get(&key) {
                                    return found.clone();
                                }
                                let mut cell = Laurent::new();
                                for (coefficient, &(power, v)) in coefficients.iter().zip(&key.1) {
                                    for (&p, value) in distinct[v].iter() {
                                        add_laurent(&mut cell, p + power, coefficient.mul(value));
                                    }
                                }
                                let cell = Arc::new(cell);
                                memo.insert(key, cell.clone());
                                cell
                            })
                            .collect()
                    }
                },
            )
            .collect())
    }

    fn cached(&self, caches: &SharedCaches, task: &Task) -> Option<Arc<Laurent>> {
        match task {
            Task::Primitive(kind, n, bra) => {
                caches.primitive_cache.get(&(*kind, *n, *bra)).cloned()
            }
            Task::Ring(ring, bra) => caches.ring_cache.get(&(*ring, *bra)).cloned(),
            Task::Structure(s, b) => self.structure_cache.get(&(*s, *b)).cloned(),
            Task::Direct(..) => None,
        }
    }

    fn store(&mut self, caches: &mut SharedCaches, task: &Task, value: Arc<Laurent>) {
        match task {
            Task::Primitive(kind, n, bra) => {
                caches.primitive_cache.insert((*kind, *n, *bra), value);
            }
            Task::Ring(ring, bra) => {
                caches.ring_cache.insert((*ring, *bra), value);
            }
            Task::Structure(s, b) => {
                self.structure_cache.insert((*s, *b), value);
            }
            // Not cached (as Python): only this call's results keep it.
            Task::Direct(..) => {}
        }
    }

    fn evaluate(&self, caches: &SharedCaches, task: &Task) -> RusticolResult<Laurent> {
        match task {
            Task::Primitive(kind, n, bra) => {
                let word: Vec<Label> = (1..=*n as Label).collect();
                primitive_overlap(*kind, &word, &caches.bras[*bra])
            }
            Task::Direct(kind, word, b) => primitive_overlap(*kind, word, &self.basis[*b]),
            Task::Ring(ring, bra) => contract(&caches.bras[*bra], &caches.rings[*ring]),
            Task::Structure(s, b) => {
                let ket = Tensor::single(&self.legs, self.structures[*s].clone());
                contract(&self.basis[*b], &ket)
            }
        }
    }
}

/// `_PrimitiveCells._overlap`.
pub fn primitive_overlap(
    kind: PrimitiveKind,
    word: &[Label],
    bra: &Tensor,
) -> RusticolResult<Laurent> {
    match kind {
        PrimitiveKind::AdjointRing => adjoint_ring_overlap(word, bra),
        PrimitiveKind::FundamentalTrace => {
            contract(bra, &fundamental_trace_tensor(word, &bra.legs)?)
        }
    }
}

// --------------------------------------------------------------------- gram

/// `colour._adjoint_rename`: the permutation of the sorted adjoint labels
/// `adjoint` numbering them in the order `tensor` reads them (terms, then
/// open strings, then traces, as stored); unread labels take the rest.
pub fn adjoint_rename(tensor: &Tensor, adjoint: &[Label]) -> FastMap<Label, Label> {
    let members: FastSet<Label> = adjoint.iter().copied().collect();
    let mut seen: FastSet<Label> = FastSet::default();
    let mut order: Vec<Label> = Vec::with_capacity(adjoint.len());
    for term in &tensor.terms {
        for label in term.structure.adjoint_reading() {
            if members.contains(&label) && seen.insert(label) {
                order.push(label);
            }
        }
    }
    order.extend(
        adjoint
            .iter()
            .copied()
            .filter(|label| !seen.contains(label)),
    );
    order.into_iter().zip(adjoint.iter().copied()).collect()
}

/// `BornBasis.gram`: `<b|b'>` exact in N_c. Each row is relabelled by the
/// permutation numbering its adjoint labels in reading order and each
/// distinct relabelled pair is contracted once (in parallel).
pub fn gram(basis: &[Tensor]) -> RusticolResult<Vec<Vec<Arc<Laurent>>>> {
    let Some(first) = basis.first() else {
        return Ok(Vec::new());
    };
    let mut adjoint = first.adjoint_labels();
    adjoint.sort();
    let mut tensors: Vec<Tensor> = Vec::new();
    let mut ids: FastMap<Tensor, usize> = FastMap::default();
    let mut intern = |tensor: Tensor, tensors: &mut Vec<Tensor>| -> usize {
        *ids.entry(tensor.clone()).or_insert_with(|| {
            tensors.push(tensor);
            tensors.len() - 1
        })
    };
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    let mut pair_ids: FastMap<(usize, usize), usize> = FastMap::default();
    let mut rows: Vec<Vec<usize>> = Vec::with_capacity(basis.len());
    for left in basis {
        let map = adjoint_rename(left, &adjoint);
        let rename = |a: Label| *map.get(&a).unwrap_or(&a);
        let l = intern(left.relabelled(&rename), &mut tensors);
        let mut row = Vec::with_capacity(basis.len());
        for right in basis {
            let r = intern(right.relabelled(&rename), &mut tensors);
            let id = *pair_ids.entry((l, r)).or_insert_with(|| {
                pairs.push((l, r));
                pairs.len() - 1
            });
            row.push(id);
        }
        rows.push(row);
    }
    let values: Vec<Arc<Laurent>> = pairs
        .par_iter()
        .map(|&(l, r)| contract(&tensors[l], &tensors[r]).map(Arc::new))
        .collect::<RusticolResult<_>>()?;
    Ok(rows
        .into_iter()
        .map(|row| row.into_iter().map(|id| values[id].clone()).collect())
        .collect())
}

// --------------------------------------------------------------- projection

/// A colour accuracy (`ColourAccuracy`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accuracy {
    Lc,
    Nlc,
    Full,
}

impl Accuracy {
    pub fn parse(name: &str) -> RusticolResult<Self> {
        match name {
            "lc" => Ok(Self::Lc),
            "nlc" => Ok(Self::Nlc),
            "full" => Ok(Self::Full),
            other => Err(invalid(format!("unknown colour accuracy {other:?}"))),
        }
    }
}

/// `colour.kept`: LC keeps P, NLC keeps P, P-1 and P-2, FC every power.
pub fn kept(power: i32, degree: Option<i32>, accuracy: Accuracy) -> bool {
    let depth = match accuracy {
        Accuracy::Full => return true,
        Accuracy::Lc => 0,
        Accuracy::Nlc => 2,
    };
    degree.is_some_and(|degree| degree - depth <= power && power <= degree)
}

/// `colour.project`: N_c^shift times one cell at N_c = 3 after LC/NLC/FC
/// truncation at leading degree P.
pub fn project(cell: &Laurent, degree: Option<i32>, accuracy: Accuracy, shift: i32) -> Exact {
    let mut out = Exact::zero();
    for (&power, value) in cell {
        if kept(power + shift, degree, accuracy) {
            out.add_assign(&value.scale(&nc_power(power + shift)));
        }
    }
    out
}

/// One seed table as the projection reads it: cells, the loop orders of
/// each class and the Born orders of each basis element, as ranks in the
/// sorted order of all orders (so sorting ranks sorts the orders).
pub struct OrderedTable<'a> {
    pub cells: &'a [Vec<Arc<Laurent>>],
    pub class_orders: &'a [Vec<u32>],
    pub basis_orders: &'a [Vec<u32>],
}

impl OrderedTable<'_> {
    /// `SeedTable.admitted`: (class, loop order, basis element, Born order)
    /// entering a sector, in table order.
    pub fn admitted<'p>(
        &'p self,
        pairs: &'p FastSet<(u32, u32)>,
    ) -> impl Iterator<Item = (usize, u32, usize, u32)> + 'p {
        self.class_orders
            .iter()
            .enumerate()
            .flat_map(move |(c, orders)| {
                let mut orders = orders.clone();
                orders.sort_unstable();
                orders.dedup();
                orders.into_iter().flat_map(move |order| {
                    self.basis_orders
                        .iter()
                        .enumerate()
                        .flat_map(move |(b, born)| {
                            let mut born = born.clone();
                            born.sort_unstable();
                            born.dedup();
                            born.into_iter()
                                .filter(move |&a| pairs.contains(&(a, order)))
                                .map(move |a| (c, order, b, a))
                        })
                })
            })
    }

    /// `SeedTable.degree`: the leading N_c degree of a sector's nonzero
    /// cells, with the Born grades of each (basis element, Born order)
    /// (grade 0 when `born_grades` is None).
    pub fn degree(
        &self,
        pairs: &FastSet<(u32, u32)>,
        born_grades: Option<&FastMap<(usize, u32), Vec<i32>>>,
    ) -> Option<i32> {
        let zero = [0i32];
        let mut out: Option<i32> = None;
        for (c, _order, b, a) in self.admitted(pairs) {
            let Some(&power) = self.cells[c][b].keys().next_back() else {
                continue;
            };
            let grades: &[i32] = match born_grades {
                None => &zero,
                Some(map) => map.get(&(b, a)).map_or(&[], |g| g.as_slice()),
            };
            for &grade in grades {
                out = Some(out.map_or(power + grade, |o| o.max(power + grade)));
            }
        }
        out
    }
}

/// One observable column of a plan (`plan_export.SeedColumn`).
pub struct RecordColumn {
    pub table: usize,
    pub pairs: FastSet<(u32, u32)>,
    pub degree: Option<i32>,
    pub basis: Option<FastSet<usize>>,
}

/// Records of `_Exporter.seed_records`, as plan JSON.
pub struct SeedRecords {
    /// `[{"columns":[[[j,"re","im"],...],...]},...]`.
    pub class_records: String,
    /// `[{"classes":[[k,"re","im"],...]},...]`.
    pub records: String,
}

/// The two-level seed records of a plan (`plan_export._Exporter.seed_records`):
/// per (class, loop order) the exact projected weights K[c, b] of each
/// column's admitted Born components, and per seed key its combination of
/// class records. `components` holds the Born components (b, Born order,
/// grade) in first-seen order and is extended in place. `seeds` are the
/// seed keys (combination of (class, x), loop order) in plan order.
pub fn project_records(
    tables: &[OrderedTable<'_>],
    columns: &[RecordColumn],
    born_grades: &FastMap<(usize, u32), Vec<i32>>,
    accuracy: Accuracy,
    components: &mut Vec<(usize, u32, i32)>,
    seeds: &[(Vec<(usize, Exact)>, u32)],
) -> RusticolResult<SeedRecords> {
    let mut component_ids: FastMap<(usize, u32, i32), usize> = components
        .iter()
        .enumerate()
        .map(|(j, &key)| (key, j))
        .collect();
    let mut admitted: FastSet<(usize, u32)> = FastSet::default();
    type Entries = FastMap<(usize, u32), Vec<(usize, u32)>>;
    let mut indexed: Vec<(usize, Option<i32>, Entries)> = Vec::with_capacity(columns.len());
    for column in columns {
        let table = tables
            .get(column.table)
            .ok_or_else(|| invalid(format!("column table {} out of range", column.table)))?;
        let mut degree = column.degree;
        if accuracy != Accuracy::Full && degree.is_none() {
            degree = table.degree(&column.pairs, Some(born_grades));
        }
        let mut entries: Entries = FastMap::default();
        for (c, order, b, a) in table.admitted(&column.pairs) {
            admitted.insert((c, order));
            if column
                .basis
                .as_ref()
                .is_some_and(|basis| !basis.contains(&b))
            {
                continue;
            }
            entries.entry((c, order)).or_default().push((b, a));
        }
        indexed.push((column.table, degree, entries));
    }
    let mut projected: FastMap<(usize, usize, usize, Option<i32>, i32), Exact> = FastMap::default();
    let mut classes: FastMap<(usize, u32), usize> = FastMap::default();
    let mut class_records = String::from("[");
    let mut class_count = 0usize;
    let mut records = String::from("[");
    let no_grades: Vec<i32> = Vec::new();
    for (s, (combination, order)) in seeds.iter().enumerate() {
        for (c, _x) in combination {
            if !admitted.contains(&(*c, *order)) {
                return Err(invalid(format!(
                    "no sector admits class {c} at order rank {order}"
                )));
            }
        }
        if s > 0 {
            records.push(',');
        }
        records.push_str("{\"classes\":[");
        let mut first = true;
        for (c, x) in combination {
            if x.is_zero() {
                continue;
            }
            let record = match classes.get(&(*c, *order)) {
                Some(&found) => found,
                None => {
                    if class_count > 0 {
                        class_records.push(',');
                    }
                    class_records.push_str("{\"columns\":[");
                    for (k, (t, degree, entries)) in indexed.iter().enumerate() {
                        if k > 0 {
                            class_records.push(',');
                        }
                        let mut weights: Vec<(usize, Exact)> = Vec::new();
                        let mut slots: FastMap<usize, usize> = FastMap::default();
                        for &(b, a) in entries.get(&(*c, *order)).map_or(&[][..], |e| e.as_slice())
                        {
                            let mut grades = born_grades.get(&(b, a)).unwrap_or(&no_grades).clone();
                            grades.sort_unstable();
                            grades.dedup();
                            for grade in grades {
                                let weight = projected
                                    .entry((*t, *c, b, *degree, grade))
                                    .or_insert_with(|| {
                                        project(&tables[*t].cells[*c][b], *degree, accuracy, grade)
                                    })
                                    .clone();
                                if weight.is_zero() {
                                    continue;
                                }
                                let j = *component_ids.entry((b, a, grade)).or_insert_with(|| {
                                    components.push((b, a, grade));
                                    components.len() - 1
                                });
                                match slots.get(&j) {
                                    Some(&slot) => weights[slot].1.add_assign(&weight),
                                    None => {
                                        slots.insert(j, weights.len());
                                        weights.push((j, weight));
                                    }
                                }
                            }
                        }
                        class_records.push('[');
                        let mut first_weight = true;
                        for (j, w) in weights.iter().filter(|(_, w)| !w.is_zero()) {
                            if !first_weight {
                                class_records.push(',');
                            }
                            first_weight = false;
                            let (re, im) = w.texts();
                            let _ = write!(class_records, "[{j},\"{re}\",\"{im}\"]");
                        }
                        class_records.push(']');
                    }
                    class_records.push_str("]}");
                    classes.insert((*c, *order), class_count);
                    class_count += 1;
                    class_count - 1
                }
            };
            if !first {
                records.push(',');
            }
            first = false;
            let (re, im) = x.texts();
            let _ = write!(records, "[{record},\"{re}\",\"{im}\"]");
        }
        records.push_str("]}");
    }
    class_records.push(']');
    records.push(']');
    Ok(SeedRecords {
        class_records,
        records,
    })
}

// ------------------------------------------------------------------ degrees

/// `colour._canonical_rotation`: a trace rotated to its least label.
fn least_label_rotation(trace: &[Label]) -> Vec<Label> {
    match trace.iter().enumerate().min_by_key(|&(_, l)| *l) {
        None => Vec::new(),
        Some((i, _)) => trace[i..].iter().chain(&trace[..i]).copied().collect(),
    }
}

/// `assemble._colour_key`: open strings and traces of a structure with its
/// adjoint labels renamed, strings sorted, traces rotated to their least
/// label and sorted.
pub fn colour_key(structure: &Structure, rename: &FastMap<Label, Label>) -> Structure {
    let label = |a: &Label| *rename.get(a).unwrap_or(a);
    let mut strings: Vec<OpenString> = structure
        .strings
        .iter()
        .map(|s| OpenString {
            q: s.q,
            word: s.word.iter().map(label).collect(),
            qbar: s.qbar,
        })
        .collect();
    strings.sort();
    let mut traces: Vec<Vec<Label>> = structure
        .traces
        .iter()
        .map(|t| least_label_rotation(&t.iter().map(label).collect::<Vec<_>>()))
        .collect();
    traces.sort();
    Structure { strings, traces }
}

/// `assemble._AdjointOrbits`: relabellings of identical external adjoint
/// legs (one particle) whose transpositions leave every given Born key set
/// {(literal structure, N_c power)} invariant, checked exactly.
#[derive(Clone, Debug)]
pub struct AdjointOrbits {
    pub classes: Vec<Vec<Label>>,
}

impl AdjointOrbits {
    pub fn new(adjoint: &[(Label, i64)], born_keys: &[Vec<(Structure, i32)>]) -> Self {
        let mut labels: Vec<(Label, i64)> = adjoint.to_vec();
        labels.sort();
        let empty: FastMap<Label, Label> = FastMap::default();
        let reference: Vec<FastSet<(Structure, i32)>> = born_keys
            .iter()
            .map(|keys| {
                keys.iter()
                    .map(|(t, p)| (colour_key(t, &empty), *p))
                    .collect()
            })
            .collect();
        let mut parent: FastMap<Label, Label> = labels.iter().map(|&(l, _)| (l, l)).collect();
        fn find(parent: &FastMap<Label, Label>, mut label: Label) -> Label {
            while parent[&label] != label {
                label = parent[&label];
            }
            label
        }
        for (i, &(a, pdg_a)) in labels.iter().enumerate() {
            for &(b, pdg_b) in &labels[i + 1..] {
                if pdg_a != pdg_b || find(&parent, a) == find(&parent, b) {
                    continue;
                }
                let swap: FastMap<Label, Label> = [(a, b), (b, a)].into_iter().collect();
                let invariant = born_keys.iter().zip(&reference).all(|(keys, reference)| {
                    let swapped: FastSet<(Structure, i32)> = keys
                        .iter()
                        .map(|(t, p)| (colour_key(t, &swap), *p))
                        .collect();
                    swapped == *reference
                });
                if invariant {
                    let root_b = find(&parent, b);
                    let root_a = find(&parent, a);
                    parent.insert(root_b, root_a);
                }
            }
        }
        let mut roots: Vec<Label> = Vec::new();
        let mut members: FastMap<Label, Vec<Label>> = FastMap::default();
        for &(label, _) in &labels {
            let root = find(&parent, label);
            members
                .entry(root)
                .or_insert_with(|| {
                    roots.push(root);
                    Vec::new()
                })
                .push(label);
        }
        let classes = roots
            .into_iter()
            .map(|root| members.remove(&root).expect("root"))
            .filter(|class| class.len() > 1)
            .collect();
        Self { classes }
    }

    /// `_AdjointOrbits.rename`: the relabelling taking a structure to its
    /// orbit's representative (each class's members in reading order of its
    /// canonical key take the class's labels in increasing order).
    pub fn rename(&self, structure: &Structure) -> FastMap<Label, Label> {
        let mut out: FastMap<Label, Label> = FastMap::default();
        if self.classes.is_empty() {
            return out;
        }
        let key = colour_key(structure, &FastMap::default());
        let mut read: Vec<Label> = Vec::new();
        let mut seen: FastSet<Label> = FastSet::default();
        for label in key.adjoint_reading() {
            if seen.insert(label) {
                read.push(label);
            }
        }
        for members in &self.classes {
            let set: FastSet<Label> = members.iter().copied().collect();
            let mut order: Vec<Label> = read.iter().copied().filter(|a| set.contains(a)).collect();
            let placed: FastSet<Label> = order.iter().copied().collect();
            order.extend(members.iter().copied().filter(|a| !placed.contains(a)));
            out.extend(order.into_iter().zip(members.iter().copied()));
        }
        out
    }
}

/// The overlap degrees of `LoopSlice.degrees`: for (bra key set, ket key
/// set, offset), the largest N_c degree of a nonzero overlap
/// `<left|right>` plus the bra's and the ket's N_c powers and the offset.
/// With `orbits` (the inclusive metric), each ket structure is replaced by
/// its orbit representative (the bra sets are invariant).
pub fn overlap_degrees(
    legs: &[Leg],
    sets: &[Vec<(Structure, i32)>],
    queries: &[(usize, usize, i32)],
    orbits: Option<&AdjointOrbits>,
) -> RusticolResult<Vec<Option<i32>>> {
    // Distinct (bra set, ket structure) leading degrees.
    let mut tasks: Vec<(usize, Structure)> = Vec::new();
    let mut task_ids: FastMap<(usize, Structure), usize> = FastMap::default();
    let mut plans: Vec<Vec<(usize, i32)>> = Vec::with_capacity(queries.len());
    for &(bra, ket, offset) in queries {
        if bra >= sets.len() {
            return Err(invalid("bra key set out of range"));
        }
        let ket_set = sets
            .get(ket)
            .ok_or_else(|| invalid("ket key set out of range"))?;
        let mut plan = Vec::with_capacity(ket_set.len());
        for (right, q) in ket_set {
            let image = match orbits {
                Some(orbits) => {
                    let rename = orbits.rename(right);
                    colour_key(right, &rename)
                }
                None => right.clone(),
            };
            let id = *task_ids.entry((bra, image.clone())).or_insert_with(|| {
                tasks.push((bra, image));
                tasks.len() - 1
            });
            plan.push((id, q + offset));
        }
        plans.push(plan);
    }
    let leading: Vec<Option<i32>> = tasks
        .par_iter()
        .map(|(bra, right)| {
            let ket = Tensor::single(
                legs,
                Structure::new(right.strings.clone(), right.traces.clone()),
            );
            let mut best: Option<i32> = None;
            for (left, p) in &sets[*bra] {
                let cell = contract(&Tensor::single(legs, left.clone()), &ket)?;
                if let Some(&power) = cell.keys().next_back() {
                    best = Some(best.map_or(power + p, |b| b.max(power + p)));
                }
            }
            Ok(best)
        })
        .collect::<RusticolResult<_>>()?;
    Ok(plans
        .into_iter()
        .map(|plan| {
            plan.into_iter()
                .filter_map(|(id, shift)| leading[id].map(|d| d + shift))
                .max()
        })
        .collect())
}

/// The overlap degrees of `LoopSlice.degrees` under a colour history
/// (`observables.DirectedColour.overlap`, design §7.7; the complete-family
/// degrees of `ObservableSlice._family_degrees`): per (bra set, ket set,
/// offset) query over sets of general tensors with their N_c powers, the
/// largest N_c degree of a nonzero `<left|right>` plus both powers and the
/// offset (None if every overlap vanishes). For a history whose ket steps
/// are all gluon emissions the bra sets hold the operator images
/// ket^dagger bra(left) of the Born keys and the ket sets the literal
/// structures (single tensors); otherwise both hold the connected images.
///
/// Each distinct (bra term structure, ket term structure) overlap is counted
/// once per ket tensor, in parallel over the distinct ket tensors; the
/// result does not depend on the thread count. The tensors' 1/sqrt(2)
/// normalisations scale whole overlaps by nonzero factors, so they do not
/// move a degree; an odd total power is refused as `contract` refuses it.
pub fn tensor_overlap_degrees(
    bras: &[Vec<(Tensor, i32)>],
    kets: &[Vec<(Tensor, i32)>],
    queries: &[(usize, usize, i32)],
) -> RusticolResult<Vec<Option<i32>>> {
    // Distinct ket tensors, and the bra sets each one meets.
    let mut ket_ids: FastMap<&Tensor, usize> = FastMap::default();
    let mut ket_tensors: Vec<&Tensor> = Vec::new();
    let mut meets: Vec<Vec<usize>> = Vec::new();
    let mut task_ids: FastMap<(usize, usize), usize> = FastMap::default();
    let mut tasks: Vec<(usize, usize)> = Vec::new();
    let mut plans: Vec<Vec<(usize, i32)>> = Vec::with_capacity(queries.len());
    for &(bra, ket, offset) in queries {
        if bra >= bras.len() {
            return Err(invalid("bra key set out of range"));
        }
        let ket_set = kets
            .get(ket)
            .ok_or_else(|| invalid("ket key set out of range"))?;
        let mut plan = Vec::with_capacity(ket_set.len());
        for (right, q) in ket_set {
            let k = *ket_ids.entry(right).or_insert_with(|| {
                ket_tensors.push(right);
                meets.push(Vec::new());
                ket_tensors.len() - 1
            });
            let id = *task_ids.entry((bra, k)).or_insert_with(|| {
                tasks.push((bra, k));
                meets[k].push(bra);
                tasks.len() - 1
            });
            plan.push((id, q + offset));
        }
        plans.push(plan);
    }
    // Distinct bra term structures (the cache index of the overlaps).
    let mut structure_ids: FastMap<&Structure, usize> = FastMap::default();
    let mut entries: Vec<Vec<(&Tensor, i32, Vec<usize>)>> = Vec::with_capacity(bras.len());
    for set in bras {
        let mut out = Vec::with_capacity(set.len());
        for (image, p) in set {
            let ids = image
                .terms
                .iter()
                .map(|term| {
                    let next = structure_ids.len();
                    *structure_ids.entry(&term.structure).or_insert(next)
                })
                .collect();
            out.push((image, *p, ids));
        }
        entries.push(out);
    }
    // Per ket tensor: (bra set, leading degree) of the bra sets it meets.
    type Leading = RusticolResult<Vec<(usize, Option<i32>)>>;
    let per_ket: Vec<Leading> = (0..ket_tensors.len())
        .into_par_iter()
        .map(|k| {
            let ket = ket_tensors[k];
            // Structure overlaps with each ket term: (bra structure, ket term).
            let mut cache: FastMap<(usize, usize), Vec<(i32, Integer)>> = FastMap::default();
            let mut out = Vec::with_capacity(meets[k].len());
            for &bra in &meets[k] {
                let mut best: Option<i32> = None;
                for (image, p, ids) in &entries[bra] {
                    if image.legs != ket.legs {
                        return Err(invalid(
                            "bra and ket connections must have identical typed output legs",
                        ));
                    }
                    if (image.normalization + ket.normalization) % 2 == 1 {
                        return Err(invalid(
                            "bra/ket normalization has an odd inverse sqrt(2) power",
                        ));
                    }
                    let mut cell = Laurent::new();
                    for (x, &s) in image.terms.iter().zip(ids) {
                        for (t, y) in ket.terms.iter().enumerate() {
                            let laurent = match cache.entry((s, t)) {
                                std::collections::hash_map::Entry::Occupied(found) => {
                                    found.into_mut()
                                }
                                std::collections::hash_map::Entry::Vacant(slot) => slot.insert(
                                    trace_laurent(&overlap_traces(&x.structure, &y.structure)?)?,
                                ),
                            };
                            if laurent.is_empty() {
                                continue;
                            }
                            let coefficient = x.coefficient.conj().mul(&y.coefficient);
                            for (nc, factor) in laurent.iter() {
                                add_laurent(
                                    &mut cell,
                                    nc + x.nc_power + y.nc_power,
                                    coefficient.scale(&Rational::from(factor.clone())),
                                );
                            }
                        }
                    }
                    if let Some(&power) = cell.keys().next_back() {
                        best = Some(best.map_or(power + p, |b| b.max(power + p)));
                    }
                }
                out.push((bra, best));
            }
            Ok(out)
        })
        .collect();
    let mut leading: FastMap<(usize, usize), Option<i32>> = FastMap::default();
    for (k, found) in per_ket.into_iter().enumerate() {
        for (bra, best) in found? {
            leading.insert((bra, k), best);
        }
    }
    Ok(plans
        .into_iter()
        .map(|plan| {
            plan.into_iter()
                .filter_map(|(id, shift)| leading[&tasks[id]].map(|d| d + shift))
                .max()
        })
        .collect())
}

/// One liveness criterion: a table's cells and its degree per sector.
pub type Criterion<'a> = (&'a [Vec<Arc<Laurent>>], Vec<Option<i32>>);

/// `LoopSlice._prune_colour`'s liveness: whether a class at a loop order has
/// a nonzero cell kept by the projection in any criterion table and sector.
/// `criteria` are (table cells, degree per sector); `sectors` are the
/// (Born rank, loop rank) pairs of each sector; `basis_orders` the Born
/// ranks of each basis element; `grades` the Born grades per (b, Born rank).
pub fn live_classes(
    criteria: &[Criterion<'_>],
    sectors: &[Vec<(u32, u32)>],
    basis_orders: &[Vec<u32>],
    grades: &FastMap<(usize, u32), Vec<i32>>,
    accuracy: Accuracy,
    queries: &[(usize, u32)],
) -> RusticolResult<Vec<bool>> {
    queries
        .par_iter()
        .map(|&(c, order)| {
            for (cells, degrees) in criteria {
                let row = cells
                    .get(c)
                    .ok_or_else(|| invalid(format!("class {c} out of range")))?;
                for (s, pairs) in sectors.iter().enumerate() {
                    let degree = *degrees
                        .get(s)
                        .ok_or_else(|| invalid("sector degree missing"))?;
                    for &(a, loop_order) in pairs {
                        if loop_order != order {
                            continue;
                        }
                        for (b, born) in basis_orders.iter().enumerate() {
                            if !born.contains(&a) {
                                continue;
                            }
                            let Some(grades) = grades.get(&(b, a)) else {
                                continue;
                            };
                            for &power in row[b].keys() {
                                if grades.iter().any(|&g| kept(power + g, degree, accuracy)) {
                                    return Ok(true);
                                }
                            }
                        }
                    }
                }
            }
            Ok(false)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator (no external crate).
    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 33) % n as u64) as usize
        }
    }

    fn products(count: usize, seed: u64) -> Vec<Vec<Vec<Label>>> {
        let mut rng = Lcg(seed);
        (0..count)
            .map(|_| {
                let n = rng.below(8);
                let mut labels: Vec<Label> = (1..=n as Label).chain(1..=n as Label).collect();
                for i in (1..labels.len()).rev() {
                    labels.swap(i, rng.below(i + 1));
                }
                let mut cuts: Vec<usize> = (0..rng.below(4))
                    .map(|_| rng.below(labels.len() + 1))
                    .collect();
                cuts.sort_unstable();
                cuts.push(labels.len());
                let mut previous = 0;
                cuts.into_iter()
                    .map(|cut| {
                        let trace = labels[previous..cut].to_vec();
                        previous = cut;
                        trace
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn fierz_recursion_equals_the_subset_sum() {
        for traces in products(2000, 20261023) {
            let subset: Vec<(i32, Integer)> = cycles::trace_overlap(&traces)
                .unwrap()
                .into_iter()
                .map(|(p, c)| (p, Integer::from(c)))
                .collect();
            assert_eq!(fierz_laurent(&traces).unwrap(), subset, "{traces:?}");
        }
    }

    #[test]
    fn many_labels_take_the_fierz_recursion() {
        // Tr(tau^1 .. tau^n tau^n .. tau^1) with n beyond the subset-sum
        // label limit: tau^a tau^a = C_F with C_F = (N^2 - 1)/N, so the
        // nested trace is N C_F^n = N^(1-n) (N^2 - 1)^n
        // = sum_k C(n, k) (-1)^k N^(n+1-2k), compared term by term.
        let n = cycles::MAX_LABELS as i64 + 2;
        let mut trace: Vec<Label> = (1..=n).collect();
        trace.extend((1..=n).rev());
        let mut expected: Vec<(i32, Integer)> = Vec::new();
        let mut binomial = Integer::one();
        for k in 0..=n {
            let sign = if k % 2 == 1 { -1 } else { 1 };
            expected.push(((n + 1 - 2 * k) as i32, &binomial * &Integer::from(sign)));
            binomial = &(&binomial * &Integer::from(n - k)) / &Integer::from(k + 1);
        }
        expected.reverse();
        assert_eq!(trace_laurent(&[trace]).unwrap(), expected);
    }

    fn adjoint_tensor(legs: &[Leg], trace: Vec<Label>) -> Tensor {
        Tensor::connected(
            legs.to_vec(),
            vec![Term {
                coefficient: Exact::one(),
                structure: Structure::new(Vec::new(), vec![trace]),
                nc_power: 0,
            }],
            0,
        )
    }

    #[test]
    fn adjoint_ring_faces_equal_the_exact_contraction() {
        // <tr(1 2 3 4)| Tr_adj(F^1 F^2 F^3 F^4)>: the U(N) face count of the
        // ring's expansion equals contracting its SU(N) literal expansion.
        let legs: Vec<Leg> = (1..=4).map(|l| (l, ADJOINT)).collect();
        let word: Vec<Label> = vec![1, 3, 2, 4];
        for bra in [vec![1, 2, 3, 4], vec![1, 3, 2, 4], vec![1, 4, 2, 3]] {
            let bra = adjoint_tensor(&legs, bra);
            let faces = adjoint_ring_overlap(&word, &bra).unwrap();
            let mut ket_terms = Vec::new();
            for (sign, traces, power) in adjoint_un_terms(&word).unwrap() {
                ket_terms.push(Term {
                    coefficient: Exact::integer(sign),
                    structure: Structure::new(
                        Vec::new(),
                        traces
                            .iter()
                            .map(|t| t.iter().rev().copied().collect())
                            .collect(),
                    ),
                    nc_power: power,
                });
            }
            let ket = Tensor::connected(legs.clone(), ket_terms, 0);
            assert_eq!(faces, contract(&bra, &ket).unwrap());
        }
    }

    #[test]
    fn cells_do_not_depend_on_the_thread_count() {
        let legs: Vec<Leg> = (1..=5).map(|l| (l, ADJOINT)).collect();
        let basis: Vec<Tensor> = [[1, 2, 3, 4, 5], [1, 3, 2, 5, 4], [1, 5, 4, 3, 2]]
            .iter()
            .map(|t| adjoint_tensor(&legs, t.to_vec()))
            .collect();
        let classes = vec![
            ClassInput::AdjointRing(vec![2, 1, 3, 5, 4]),
            ClassInput::FundamentalTrace(vec![1, 2, 3, 4, 5]),
            ClassInput::Generic(adjoint_tensor(&legs, vec![1, 2, 4, 3, 5])),
        ];
        let run = |threads| {
            let mut engine = CellEngine::new(legs.clone(), basis.clone()).unwrap();
            with_threads(Some(threads), || engine.cells(&classes))
                .unwrap()
                .unwrap()
        };
        assert_eq!(run(1), run(4));
        // The fundamental trace equals the generic class of its literal tensor.
        let one = run(1);
        let generic = {
            let mut engine = CellEngine::new(legs.clone(), basis.clone()).unwrap();
            engine
                .cells(&[ClassInput::Generic(
                    fundamental_trace_tensor(&[1, 2, 3, 4, 5], &legs).unwrap(),
                )])
                .unwrap()
        };
        assert_eq!(one[1], generic[0]);
    }

    /// `c * cell` with each N_c power shifted by `shift`.
    fn shifted(cell: &Laurent, shift: i32, factor: &Rational) -> Laurent {
        let mut out = Laurent::new();
        for (&power, value) in cell {
            add_laurent(&mut out, power + shift, value.scale(factor));
        }
        out
    }

    #[test]
    fn operator_images_obey_casimirs_and_colour_conservation() {
        // q(1) qbar(2) g(3) g(4): T_i.T_i = C_i and sum_{j != i} T_i.T_j = -C_i
        // on every basis vector, through the native images (connect, then
        // absorb by the completeness relation), contracted with the basis.
        let legs: Vec<Leg> = vec![(1, 3), (2, -3), (3, ADJOINT), (4, ADJOINT)];
        let string = |word: Vec<Label>, traces: Vec<Vec<Label>>| {
            Tensor::connected(
                legs.clone(),
                vec![Term {
                    coefficient: Exact::one(),
                    structure: Structure::new(
                        vec![OpenString {
                            q: 1,
                            word,
                            qbar: 2,
                        }],
                        traces,
                    ),
                    nc_power: 0,
                }],
                0,
            )
        };
        let basis = [
            string(vec![3, 4], vec![]),
            string(vec![4, 3], vec![]),
            string(vec![], vec![vec![3, 4]]),
        ];
        let half = Rational::new(1, 2);
        let one = Rational::one();
        // C_F = N/2 - 1/(2N), C_A = N.
        let casimir = |label: Label, cell: &Laurent| -> Laurent {
            if label <= 2 {
                let mut out = shifted(cell, 1, &half);
                for (&power, value) in &shifted(cell, -1, &half) {
                    add_laurent(&mut out, power, value.scale(&Rational::from(-1)));
                }
                out
            } else {
                shifted(cell, 1, &one)
            }
        };
        for b in &basis {
            for label in [1, 2, 3, 4] {
                let square = operator_image(b, &[(label, -1)], &[(label, -1)]).unwrap();
                for b2 in &basis {
                    let cell = contract(b, b2).unwrap();
                    assert_eq!(contract(&square, b2).unwrap(), casimir(label, &cell));
                    let mut total = Laurent::new();
                    for other in [1, 2, 3, 4] {
                        if other == label {
                            continue;
                        }
                        let image = operator_image(b, &[(label, -1)], &[(other, -1)]).unwrap();
                        for (&power, value) in &contract(&image, b2).unwrap() {
                            add_laurent(&mut total, power, value.clone());
                        }
                    }
                    let expected = casimir(label, &cell);
                    let mut negated = Laurent::new();
                    for (&power, value) in &expected {
                        add_laurent(&mut negated, power, value.scale(&Rational::from(-1)));
                    }
                    assert_eq!(total, negated, "leg {label}");
                }
            }
        }
    }
}
