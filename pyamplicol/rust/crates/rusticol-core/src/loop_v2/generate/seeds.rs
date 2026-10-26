// SPDX-License-Identifier: 0BSD

//! Seed combinations and seed classes of the exact colour seed tables
//! (WP3.1 I10; `workspace/reports/engine-v2/wp31/RUST-MIGRATION.md`,
//! package native-seeds, increments B2 and B3).
//!
//! Every function reproduces its Python twin in `loops/v2/colour.py`, which
//! stays the oracle (`tests/unit/test_loop_v2_native_seeds.py`):
//!
//! - **B2, combinations:** `CompressedColour.kind`, `combination`,
//!   `_ring_combination` (with its per-label-renaming memo), `_combination`,
//!   `_ring_word` and `canonical_primitive`; the certificate bookkeeping of
//!   `CompressedColour.certificate` (which skeletons qualify as the smallest
//!   single-leg instance of each kind, up to the `repr` tie-break of their
//!   contents); `Skeleton.orders`.
//! - **B3, classes:** `_ClassKeys`, `_ring_relabelling`,
//!   `_proportional_key`, `_ratio` and the `row` / `class_of` accumulation of
//!   `_extended`, with `class_of` as distinct rows of (class, exact x); the
//!   class orders and the LC/NLC prune liveness of
//!   `assemble.LoopSlice._prune_colour`.
//!
//! The work runs in steps, because one input only Python computes (the
//! certificate check of each compression on its smallest instance):
//! [`SeedBuilder::analyse`] (combinations and certificate candidates), then
//! [`SeedBuilder::needs`] once the caller has certified the compressions
//! (the distinct canonical quark rings whose exact SU(N_c) reduction keys
//! their classes, and the skeletons that keep a generic seed), then
//! [`SeedBuilder::reduce`], which reduces those needs in-crate
//! (`networks`, the twin of `reduce_network`), and
//! [`SeedBuilder::complete_reduced`]. [`SeedBuilder::complete`] takes the
//! reduced tensors from the caller instead (the Python reductions, with
//! `native_seeds.NATIVE_REDUCTIONS` off).
//!
//! **What Python hands over.** Per distinct content its compression
//! (`CompressedColour._content`: position kind, single-leg flag, terms as
//! (exact coefficient, descriptor)), its coupling orders, and per skeleton its
//! content ids. Descriptors are `("adjoint", letters)`,
//! `("fundamental", letters, forward)` or `("exact", network)`; an exact
//! network arrives as the `repr` of the network with every external adjoint
//! label replaced by a slot (a *template*), so that its relabellings and
//! their `repr` (the order of `canonical_primitive` for quark rings) are
//! reproduced without the network itself.
//!
//! **Python's order.** Classes are numbered first-seen; combinations and the
//! rotations of quark rings are ordered by the `repr` strings Python sorts by,
//! compared as UTF-8 bytes (= code points) without building the joined
//! strings. Labels and coefficients are exact (`Rational`); no float is used.
//! No Symbolica licence call is made. The output does not depend on the
//! thread count: every step is sequential except [`SeedBuilder::reduce`],
//! whose needs reduce in parallel into an order-preserving collect, each
//! result depending on its need only (a thread's Fierz memo caches a pure
//! function).

use super::FastMap;
use super::networks::{self, NetString, Network, Product};
use super::orders::OrderSpace;
use crate::loop_v2::dress::parse_rational;
use crate::{RusticolError, RusticolResult};
use rayon::prelude::*;
use serde::Deserialize;
use std::cmp::Ordering;
use symbolica::domains::rational::Rational;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("seed table: {}", message.into()))
}

// ------------------------------------------------------------ exact numbers

/// An exact complex rational (`color.connections.ExactColorCoefficient`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Exact {
    pub re: Rational,
    pub im: Rational,
}

impl Exact {
    pub fn one() -> Self {
        Self {
            re: Rational::one(),
            im: Rational::zero(),
        }
    }

    pub fn zero() -> Self {
        Self {
            re: Rational::zero(),
            im: Rational::zero(),
        }
    }

    pub fn parse(re: &str, im: &str) -> RusticolResult<Self> {
        Ok(Self {
            re: parse_rational(re)?,
            im: parse_rational(im)?,
        })
    }

    pub fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self {
            re: self.re.clone() * &other.re - self.im.clone() * &other.im,
            im: self.re.clone() * &other.im + self.im.clone() * &other.re,
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self {
            re: self.re.clone() + &other.re,
            im: self.im.clone() + &other.im,
        }
    }

    pub fn neg(&self) -> Self {
        Self {
            re: self.re.neg(),
            im: self.im.neg(),
        }
    }

    /// `1 / self` (`colour._ratio(_ONE, self)`).
    pub fn inverse(&self) -> RusticolResult<Self> {
        let norm = self.re.clone() * &self.re + self.im.clone() * &self.im;
        if norm.is_zero() {
            return Err(invalid("division by an exact zero"));
        }
        Ok(Self {
            re: self.re.clone() / &norm,
            im: self.im.neg() / &norm,
        })
    }

    /// `(numerator, denominator)` texts of both parts, `Fraction(text)` form.
    pub fn text(&self) -> (String, String) {
        (fraction_text(&self.re), fraction_text(&self.im))
    }

    /// Python's `repr` of the coefficient.
    pub fn repr(&self) -> String {
        format!(
            "ExactColorCoefficient(real={}, imag={})",
            fraction_repr(&self.re),
            fraction_repr(&self.im)
        )
    }
}

fn fraction_text(value: &Rational) -> String {
    format!("{}/{}", value.numerator_ref(), value.denominator_ref())
}

fn fraction_repr(value: &Rational) -> String {
    format!(
        "Fraction({}, {})",
        value.numerator_ref(),
        value.denominator_ref()
    )
}

/// Interned exact coefficients; id 0 is one.
#[derive(Debug, Default)]
struct Coefficients {
    values: Vec<Exact>,
    index: FastMap<Exact, u32>,
}

impl Coefficients {
    const ONE: u32 = 0;

    fn new() -> Self {
        let mut out = Self::default();
        out.intern(Exact::one());
        out
    }

    fn intern(&mut self, value: Exact) -> u32 {
        if let Some(&id) = self.index.get(&value) {
            return id;
        }
        let id = self.values.len() as u32;
        self.index.insert(value.clone(), id);
        self.values.push(value);
        id
    }

    fn get(&self, id: u32) -> &Exact {
        &self.values[id as usize]
    }
}

// ------------------------------------------------------------ repr strings

/// Python's `repr` of a tuple of items given by their `repr`, as chunks:
/// `(a, b)`, `(a,)` and `()`.
fn tuple_chunks<'a>(items: impl ExactSizeIterator<Item = &'a str>) -> Vec<&'a [u8]> {
    let count = items.len();
    let mut out = Vec::with_capacity(2 * count + 3);
    out.push(b"(".as_slice());
    for (i, item) in items.enumerate() {
        if i > 0 {
            out.push(b", ".as_slice());
        }
        out.push(item.as_bytes());
    }
    if count == 1 {
        out.push(b",".as_slice());
    }
    out.push(b")".as_slice());
    out
}

/// Compare two strings given as chunks, as Python compares `str` (code
/// points; UTF-8 bytes order the same way).
fn compare_chunks(a: &[&[u8]], b: &[&[u8]]) -> Ordering {
    let (mut i, mut j, mut x, mut y) = (0, 0, 0, 0);
    loop {
        while i < a.len() && x == a[i].len() {
            i += 1;
            x = 0;
        }
        while j < b.len() && y == b[j].len() {
            j += 1;
            y = 0;
        }
        match (i < a.len(), j < b.len()) {
            (false, false) => return Ordering::Equal,
            (false, true) => return Ordering::Less,
            (true, false) => return Ordering::Greater,
            (true, true) => {}
        }
        let n = (a[i].len() - x).min(b[j].len() - y);
        match a[i][x..x + n].cmp(&b[j][y..y + n]) {
            Ordering::Equal => {
                x += n;
                y += n;
            }
            other => return other,
        }
    }
}

fn int_tuple_repr(values: &[i64]) -> String {
    let items: Vec<String> = values.iter().map(i64::to_string).collect();
    let chunks = tuple_chunks(items.iter().map(String::as_str));
    String::from_utf8(chunks.concat()).expect("ascii")
}

// ------------------------------------------------------------ descriptors

/// A position descriptor of `CompressedColour._content` (module docs).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Descriptor {
    Adjoint(Vec<i64>),
    Fundamental(Vec<i64>, bool),
    /// A template id and the external adjoint label of each of its slots.
    Exact(u32, Vec<i64>),
}

impl Descriptor {
    /// The labels `_ring_relabelling` visits, in its order.
    fn labels(&self) -> &[i64] {
        match self {
            Self::Adjoint(letters) | Self::Fundamental(letters, _) => letters,
            Self::Exact(_, labels) => labels,
        }
    }

    fn renamed(&self, rename: impl Fn(i64) -> i64) -> Self {
        let map = |labels: &[i64]| labels.iter().map(|&a| rename(a)).collect();
        match self {
            Self::Adjoint(letters) => Self::Adjoint(map(letters)),
            Self::Fundamental(letters, forward) => Self::Fundamental(map(letters), *forward),
            Self::Exact(template, labels) => Self::Exact(*template, map(labels)),
        }
    }
}

#[derive(Debug, Default)]
struct Descriptors {
    items: Vec<Descriptor>,
    reprs: Vec<String>,
    index: FastMap<Descriptor, u32>,
    /// Template parts (slots between them), interned by text.
    templates: Vec<Vec<String>>,
    template_index: FastMap<Vec<String>, u32>,
}

impl Descriptors {
    fn template(&mut self, parts: Vec<String>) -> RusticolResult<u32> {
        if parts.is_empty() {
            return Err(invalid("an empty network template"));
        }
        if let Some(&id) = self.template_index.get(&parts) {
            return Ok(id);
        }
        let id = self.templates.len() as u32;
        self.template_index.insert(parts.clone(), id);
        self.templates.push(parts);
        Ok(id)
    }

    fn intern(&mut self, descriptor: Descriptor) -> RusticolResult<u32> {
        if let Some(&id) = self.index.get(&descriptor) {
            return Ok(id);
        }
        let repr = match &descriptor {
            Descriptor::Adjoint(letters) => format!("('adjoint', {})", int_tuple_repr(letters)),
            Descriptor::Fundamental(letters, forward) => format!(
                "('fundamental', {}, {})",
                int_tuple_repr(letters),
                if *forward { "True" } else { "False" }
            ),
            Descriptor::Exact(template, labels) => {
                let parts = self
                    .templates
                    .get(*template as usize)
                    .ok_or_else(|| invalid("unknown network template"))?;
                if parts.len() != labels.len() + 1 {
                    return Err(invalid("template slots and labels differ"));
                }
                let mut text = String::from("('exact', ");
                for (k, part) in parts.iter().enumerate() {
                    if k > 0 {
                        text.push_str(&labels[k - 1].to_string());
                    }
                    text.push_str(part);
                }
                text.push(')');
                text
            }
        };
        let id = self.items.len() as u32;
        self.index.insert(descriptor.clone(), id);
        self.items.push(descriptor);
        self.reprs.push(repr);
        Ok(id)
    }

    fn get(&self, id: u32) -> &Descriptor {
        &self.items[id as usize]
    }

    fn repr(&self, id: u32) -> &str {
        &self.reprs[id as usize]
    }

    /// Python's order of two rings of descriptors (`repr` of the tuples).
    fn compare_rings(&self, a: &[u32], b: &[u32]) -> Ordering {
        if a == b {
            return Ordering::Equal;
        }
        let left = tuple_chunks(a.iter().map(|&d| self.repr(d)));
        let right = tuple_chunks(b.iter().map(|&d| self.repr(d)));
        compare_chunks(&left, &right)
    }

    /// The rotation of a ring with the smallest `repr`
    /// (`canonical_primitive("quark-ring", word)`); its offset.
    fn canonical_rotation(&self, ring: &[u32]) -> usize {
        let n = ring.len();
        let rotated =
            |k: usize| -> Vec<u32> { ring[k..].iter().chain(&ring[..k]).copied().collect() };
        let mut best = 0;
        let mut best_ring = rotated(0);
        for k in 1..n {
            let candidate = rotated(k);
            if self.compare_rings(&candidate, &best_ring) == Ordering::Less {
                best = k;
                best_ring = candidate;
            }
        }
        best
    }
}

// ------------------------------------------------------------- primitives

/// The kind of a certified compression (`colour.PrimitiveKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    AdjointRing,
    FundamentalTrace,
    QuarkRing,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Self::AdjointRing => "adjoint-ring",
            Self::FundamentalTrace => "fundamental-trace",
            Self::QuarkRing => "quark-ring",
        }
    }

    pub fn from_name(name: &str) -> RusticolResult<Self> {
        match name {
            "adjoint-ring" => Ok(Self::AdjointRing),
            "fundamental-trace" => Ok(Self::FundamentalTrace),
            "quark-ring" => Ok(Self::QuarkRing),
            other => Err(invalid(format!("unknown primitive kind {other:?}"))),
        }
    }
}

/// A primitive word: external labels, or the descriptors of a quark ring.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Word {
    Letters(Vec<i64>),
    Ring(Vec<u32>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Primitive {
    kind: Kind,
    word: Word,
}

#[derive(Debug, Default)]
struct Primitives {
    items: Vec<Primitive>,
    index: FastMap<Primitive, u32>,
    /// `repr` of the letter primitives (quark rings compare by descriptors).
    reprs: Vec<Option<String>>,
}

impl Primitives {
    fn intern(&mut self, primitive: Primitive) -> u32 {
        if let Some(&id) = self.index.get(&primitive) {
            return id;
        }
        let repr = match &primitive.word {
            Word::Letters(letters) => Some(format!(
                "Primitive(kind='{}', word={})",
                primitive.kind.name(),
                int_tuple_repr(letters)
            )),
            Word::Ring(_) => None,
        };
        let id = self.items.len() as u32;
        self.index.insert(primitive.clone(), id);
        self.items.push(primitive);
        self.reprs.push(repr);
        id
    }

    fn get(&self, id: u32) -> &Primitive {
        &self.items[id as usize]
    }

    /// Python's order of two distinct primitives of one combination (by
    /// `repr`; their coefficients never decide, see `sort_combination`).
    fn compare(&self, descriptors: &Descriptors, a: u32, b: u32) -> Ordering {
        let (x, y) = (self.get(a), self.get(b));
        match (&x.word, &y.word) {
            (Word::Ring(u), Word::Ring(v)) => descriptors.compare_rings(u, v),
            _ => {
                let (Some(left), Some(right)) = (&self.reprs[a as usize], &self.reprs[b as usize])
                else {
                    unreachable!("a combination mixes quark rings with letter words")
                };
                left.as_bytes().cmp(right.as_bytes())
            }
        }
    }
}

/// `canonical_primitive(kind, word)` of an adjoint ring or a fundamental
/// trace: the smallest (rotation, sign); adjoint rings also take their
/// reflections, with sign (-1)^n.
fn canonical_letters(kind: Kind, word: &[i64]) -> (Vec<i64>, i64) {
    let n = word.len();
    if n == 0 {
        return (Vec::new(), 1);
    }
    let rotation =
        |w: &[i64], k: usize| -> Vec<i64> { w[k..].iter().chain(&w[..k]).copied().collect() };
    let mut best = (rotation(word, 0), 1i64);
    for k in 1..n {
        let candidate = (rotation(word, k), 1);
        if candidate < best {
            best = candidate;
        }
    }
    if kind == Kind::AdjointRing {
        let mirror: Vec<i64> = word.iter().rev().copied().collect();
        let sign = if n % 2 == 1 { -1 } else { 1 };
        for k in 0..n {
            let candidate = (rotation(&mirror, k), sign);
            if candidate < best {
                best = candidate;
            }
        }
    }
    best
}

/// `CompressedColour._ring_word`: the word of one product of position terms
/// around an adjoint ring or a closed fundamental loop (None: the strings do
/// not close into one loop).
fn ring_word(kind: Kind, descriptors: &[&Descriptor]) -> Option<Vec<i64>> {
    if kind == Kind::AdjointRing {
        return Some(
            descriptors
                .iter()
                .flat_map(|d| d.labels().iter().copied())
                .collect(),
        );
    }
    let n = descriptors.len();
    // String p: (row, col) = (p, p - 1) forward, else (p - 1, p).
    let mut by_row = vec![usize::MAX; n];
    let mut col_used = vec![false; n];
    let mut ends = Vec::with_capacity(n);
    for (p, descriptor) in descriptors.iter().enumerate() {
        let Descriptor::Fundamental(_, forward) = descriptor else {
            return None;
        };
        let previous = (p + n - 1) % n;
        let (row, col) = if *forward {
            (p, previous)
        } else {
            (previous, p)
        };
        if by_row[row] != usize::MAX {
            return None; // a fundamental index occurs twice as row
        }
        by_row[row] = p;
        ends.push(col);
    }
    for &col in &ends {
        if col_used[col] {
            return None; // a fundamental index occurs twice as column
        }
        col_used[col] = true;
    }
    let first = by_row
        .iter()
        .position(|&p| p == 0)
        .expect("position 0 has a row");
    let mut word = Vec::new();
    let mut current = first;
    let mut visited = 0;
    loop {
        let p = by_row[current];
        word.extend_from_slice(descriptors[p].labels());
        visited += 1;
        current = ends[p];
        if current == first {
            break;
        }
    }
    (visited == n).then_some(word)
}

// ---------------------------------------------------------------- tensors

/// `color.connections.OpenColorString`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpenString {
    pub fundamental: i64,
    pub adjoint: Vec<i64>,
    pub antifundamental: i64,
}

/// `color.connections.ColorTensor`: sorted open strings, sorted canonical
/// traces (field order = Python's dataclass order).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ColourTensor {
    pub strings: Vec<OpenString>,
    pub traces: Vec<Vec<i64>>,
}

fn canonical_trace(trace: &[i64]) -> Vec<i64> {
    let n = trace.len();
    (0..n)
        .map(|k| {
            trace[k..]
                .iter()
                .chain(&trace[..k])
                .copied()
                .collect::<Vec<_>>()
        })
        .min()
        .unwrap_or_default()
}

impl ColourTensor {
    /// The constructor's normal form.
    pub fn new(mut strings: Vec<OpenString>, traces: Vec<Vec<i64>>) -> Self {
        strings.sort();
        let mut traces: Vec<Vec<i64>> = traces.iter().map(|t| canonical_trace(t)).collect();
        traces.sort();
        Self { strings, traces }
    }

    /// `colour._relabelled_colour`: adjoint labels renamed (fundamental ones
    /// kept, unmapped labels kept).
    fn relabelled(&self, rename: &FastMap<i64, i64>) -> Self {
        let label = |a: &i64| *rename.get(a).unwrap_or(a);
        Self::new(
            self.strings
                .iter()
                .map(|s| OpenString {
                    fundamental: s.fundamental,
                    adjoint: s.adjoint.iter().map(label).collect(),
                    antifundamental: s.antifundamental,
                })
                .collect(),
            self.traces
                .iter()
                .map(|t| t.iter().map(label).collect())
                .collect(),
        )
    }
}

/// A `ConnectedColorTensor` on the table's legs: its inverse sqrt(2) power
/// and its terms (coefficient, structure, N_c power) in canonical order.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub normalization: i64,
    pub terms: Vec<(Exact, ColourTensor, i64)>,
}

// ---------------------------------------------------------- input records

/// A descriptor: `(kind, letters or slot labels, forward, template)`; kind 0
/// adjoint, 1 fundamental, 2 exact.
type DescriptorSpec = (u8, Vec<i64>, bool, u32);
/// A content: `(position kind, single-leg, terms (coefficient, descriptor),
/// coupling orders)`; position kind 0 adjoint, 1 fundamental, 2 exact.
type ContentSpec = (u8, bool, Vec<(u32, u32)>, Vec<(u16, i64)>);
/// A term of a tensor: `(re, im, N_c power, open strings, traces)`.
pub type TermSpec = (
    String,
    String,
    i64,
    Vec<(i64, Vec<i64>, i64)>,
    Vec<Vec<i64>>,
);
/// A tensor: `(inverse sqrt(2) power, terms)`.
pub type TensorSpec = (i64, Vec<TermSpec>);
/// An existing class: `(tag, items, orders)`; tag 0 adjoint ring, 1
/// fundamental trace (items: letters), 2 quark ring (items: descriptor
/// ids), 3 generic (items: `[tensor index]`).
type ExistingSpec = (u8, Vec<i64>, Vec<Vec<(u16, i64)>>);

/// The Θ(contents) input of one seed-table stage (`loops/v2/native_seeds.py`).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedSpec {
    /// External adjoint labels (`_ClassKeys._adjoint`).
    pub adjoint: Vec<i64>,
    /// Certified compressions requested (`compress=`); also the class-key
    /// merge of proportional classes.
    pub compress: bool,
    pub coefficients: Vec<(String, String)>,
    pub templates: Vec<Vec<String>>,
    descriptors: Vec<DescriptorSpec>,
    contents: Vec<ContentSpec>,
    existing: Vec<ExistingSpec>,
    pub tensors: Vec<TensorSpec>,
}

pub fn parse_tensor(spec: &TensorSpec) -> RusticolResult<Tensor> {
    let (normalization, terms) = spec;
    Ok(Tensor {
        normalization: *normalization,
        terms: terms
            .iter()
            .map(|(re, im, power, strings, traces)| {
                Ok((
                    Exact::parse(re, im)?,
                    ColourTensor::new(
                        strings
                            .iter()
                            .map(|(f, adjoint, af)| OpenString {
                                fundamental: *f,
                                adjoint: adjoint.clone(),
                                antifundamental: *af,
                            })
                            .collect(),
                        traces.clone(),
                    ),
                    *power,
                ))
            })
            .collect::<RusticolResult<_>>()?,
    })
}

// ------------------------------------------------------------- the stages

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Position {
    Adjoint,
    Fundamental,
    Exact,
}

#[derive(Clone, Copy, Debug)]
struct Content {
    position: Position,
    single: bool,
    terms: u32,
    orders: u32,
}

/// The seed of a class or of a combination term.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Seed {
    Primitive(u32),
    /// A generic seed: index into the generic seeds (`needs().1`).
    Generic(u32),
    /// An existing generic class (`SeedSpec::existing`): its tensor index.
    Tensor(u32),
}

/// `_ring_relabelling` of a ring: (canonical relabelled ring, the original
/// descriptors in its rotation, forward map or empty for the identity).
type Relabelling = (Vec<u32>, Vec<u32>, Vec<(i64, i64)>);

/// Where a skeleton's seed comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Source {
    Combination(u32),
    Generic(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ClassKey {
    Primitive(u32),
    Zero,
    Proportional(u32),
    Exact(u32),
}

type ProportionalKey = (i64, Vec<(u32, i64, u32)>);
type ExactKey = (i64, Vec<(u32, u32, i64)>);

/// A distinct canonical quark ring whose reduction keys its classes: the
/// original descriptors in the canonical rotation and the relabelling
/// (`_ring_relabelling`'s forward map, empty when it is the identity).
#[derive(Clone, Debug)]
pub struct RingNeed {
    pub descriptors: Vec<u32>,
    pub forward: Vec<(i64, i64)>,
}

/// Output of [`SeedBuilder::complete`].
#[derive(Debug, Default)]
pub struct SeedOutput {
    /// Seeds of the classes after the existing ones, in class order.
    pub classes: Vec<Seed>,
    /// Per class (existing included) its loop orders, as order ids.
    pub orders: Vec<Vec<u32>>,
    /// Distinct `class_of` rows: (class, coefficient id).
    pub rows: Vec<Vec<(u32, u32)>>,
    /// Per skeleton its row.
    pub skeleton_rows: Vec<u32>,
}

/// One seed-table stage: `build_seed_table` or `extend_seed_table` on the
/// given skeletons (module docs).
#[derive(Debug)]
pub struct SeedBuilder {
    compress: bool,
    adjoint: Vec<i64>,
    coefficients: Coefficients,
    descriptors: Descriptors,
    term_lists: Vec<Vec<(u32, u32)>>,
    term_index: FastMap<Vec<(u32, u32)>, u32>,
    contents: Vec<Content>,
    orders: OrderSpace,
    offsets: Vec<usize>,
    skeleton_contents: Vec<u32>,
    skeleton_orders: Vec<u32>,
    /// Per descriptor id the input descriptor it was first given as.
    spec_of: FastMap<u32, u32>,
    existing: Vec<Seed>,
    existing_orders: Vec<Vec<u32>>,
    tensors: Vec<Tensor>,
    primitives: Primitives,
    combinations: Vec<Vec<(u32, u32)>>,
    combination_index: FastMap<Vec<(u32, u32)>, u32>,
    combination_memo: FastMap<(Kind, Vec<u32>), Option<u32>>,
    renamed_memo: FastMap<(Kind, Vec<u32>), Option<u32>>,
    kinds: Vec<Option<Kind>>,
    skeleton_combinations: Vec<Option<u32>>,
    analysed: bool,
    // needs
    certified: Vec<Kind>,
    sources: Vec<Source>,
    generic_skeletons: Vec<usize>,
    rings: Vec<RingNeed>,
    ring_index: FastMap<Vec<u32>, u32>,
    /// Per quark-ring primitive: (ring need, back map).
    ring_of: FastMap<u32, (u32, Vec<(i64, i64)>)>,
    has_needs: bool,
    /// The needs reduced natively ([`SeedBuilder::reduce`]): ring tensors
    /// (canonical seeds of `_reduced_primitive`) and generic seeds.
    reduced: Option<(Vec<Tensor>, Vec<Tensor>)>,
}

impl SeedBuilder {
    /// The stage's input: `spec` and the skeletons as content-id spans
    /// (`offsets` has one more entry than there are skeletons).
    pub fn new(
        spec: SeedSpec,
        offsets: Vec<usize>,
        skeleton_contents: Vec<u32>,
    ) -> RusticolResult<Self> {
        if offsets.first() != Some(&0) || offsets.last() != Some(&skeleton_contents.len()) {
            return Err(invalid("skeleton offsets do not span the contents"));
        }
        if offsets.windows(2).any(|w| w[0] >= w[1]) {
            return Err(invalid("a skeleton without contents"));
        }
        let mut coefficients = Coefficients::new();
        let coefficient_ids = spec
            .coefficients
            .iter()
            .map(|(re, im)| Ok(coefficients.intern(Exact::parse(re, im)?)))
            .collect::<RusticolResult<Vec<_>>>()?;
        let mut descriptors = Descriptors::default();
        let templates = spec
            .templates
            .into_iter()
            .map(|parts| descriptors.template(parts))
            .collect::<RusticolResult<Vec<_>>>()?;
        let descriptor_ids = spec
            .descriptors
            .iter()
            .map(|(kind, labels, forward, template)| {
                let descriptor = match kind {
                    0 => Descriptor::Adjoint(labels.clone()),
                    1 => Descriptor::Fundamental(labels.clone(), *forward),
                    2 => Descriptor::Exact(
                        *templates
                            .get(*template as usize)
                            .ok_or_else(|| invalid("unknown template"))?,
                        labels.clone(),
                    ),
                    other => return Err(invalid(format!("descriptor kind {other}"))),
                };
                descriptors.intern(descriptor)
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let mut orders = OrderSpace::new();
        let mut term_lists = Vec::new();
        let mut term_index = FastMap::default();
        let mut contents = Vec::with_capacity(spec.contents.len());
        for (position, single, terms, content_orders) in &spec.contents {
            let position = match position {
                0 => Position::Adjoint,
                1 => Position::Fundamental,
                2 => Position::Exact,
                other => return Err(invalid(format!("position kind {other}"))),
            };
            let terms = terms
                .iter()
                .map(|&(c, d)| {
                    Ok((
                        *coefficient_ids
                            .get(c as usize)
                            .ok_or_else(|| invalid("unknown coefficient"))?,
                        *descriptor_ids
                            .get(d as usize)
                            .ok_or_else(|| invalid("unknown descriptor"))?,
                    ))
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            let id = *term_index.entry(terms.clone()).or_insert_with(|| {
                term_lists.push(terms);
                (term_lists.len() - 1) as u32
            });
            contents.push(Content {
                position,
                single: *single,
                terms: id,
                orders: orders.intern(content_orders.clone()),
            });
        }
        if skeleton_contents
            .iter()
            .any(|&c| c as usize >= contents.len())
        {
            return Err(invalid("a skeleton refers to an unknown content"));
        }
        let skeleton_orders = offsets
            .windows(2)
            .map(|span| {
                skeleton_contents[span[0]..span[1]]
                    .iter()
                    .fold(OrderSpace::EMPTY, |total, &c| {
                        orders.add(total, contents[c as usize].orders)
                    })
            })
            .collect();
        let tensors = spec
            .tensors
            .iter()
            .map(parse_tensor)
            .collect::<RusticolResult<Vec<_>>>()?;
        let mut builder = Self {
            compress: spec.compress,
            adjoint: {
                let mut labels = spec.adjoint.clone();
                labels.sort_unstable();
                labels.dedup();
                labels
            },
            coefficients,
            descriptors,
            term_lists,
            term_index,
            contents,
            orders,
            offsets,
            skeleton_contents,
            skeleton_orders,
            spec_of: {
                let mut spec_of = FastMap::default();
                for (k, &d) in descriptor_ids.iter().enumerate() {
                    spec_of.entry(d).or_insert(k as u32);
                }
                spec_of
            },
            existing: Vec::new(),
            existing_orders: Vec::new(),
            tensors,
            primitives: Primitives::default(),
            combinations: Vec::new(),
            combination_index: FastMap::default(),
            combination_memo: FastMap::default(),
            renamed_memo: FastMap::default(),
            kinds: Vec::new(),
            skeleton_combinations: Vec::new(),
            analysed: false,
            certified: Vec::new(),
            sources: Vec::new(),
            generic_skeletons: Vec::new(),
            rings: Vec::new(),
            ring_index: FastMap::default(),
            ring_of: FastMap::default(),
            has_needs: false,
            reduced: None,
        };
        for (tag, items, class_orders) in &spec.existing {
            let seed = match tag {
                0 | 1 => Seed::Primitive(builder.primitives.intern(Primitive {
                    kind: if *tag == 0 {
                        Kind::AdjointRing
                    } else {
                        Kind::FundamentalTrace
                    },
                    word: Word::Letters(items.clone()),
                })),
                2 => {
                    let ring = items
                        .iter()
                        .map(|&d| {
                            usize::try_from(d)
                                .ok()
                                .and_then(|d| descriptor_ids.get(d).copied())
                                .ok_or_else(|| invalid("unknown descriptor of a class"))
                        })
                        .collect::<RusticolResult<Vec<_>>>()?;
                    Seed::Primitive(builder.primitives.intern(Primitive {
                        kind: Kind::QuarkRing,
                        word: Word::Ring(ring),
                    }))
                }
                3 => {
                    let index = items
                        .first()
                        .and_then(|&t| u32::try_from(t).ok())
                        .filter(|&t| (t as usize) < builder.tensors.len())
                        .ok_or_else(|| invalid("unknown tensor of a class"))?;
                    Seed::Tensor(index)
                }
                other => return Err(invalid(format!("class tag {other}"))),
            };
            builder.existing.push(seed);
            builder.existing_orders.push(
                class_orders
                    .iter()
                    .map(|o| builder.orders.intern(o.clone()))
                    .collect(),
            );
        }
        Ok(builder)
    }

    pub fn skeletons(&self) -> usize {
        self.offsets.len() - 1
    }

    fn skeleton(&self, s: usize) -> &[u32] {
        &self.skeleton_contents[self.offsets[s]..self.offsets[s + 1]]
    }

    /// The order vectors of the order ids (names are the caller's ids).
    pub fn order_vectors(&self) -> Vec<Vec<(u16, i64)>> {
        (0..self.orders.len() as u32)
            .map(|id| self.orders.get(id).clone())
            .collect()
    }

    /// `Skeleton.orders` of every skeleton, as order ids.
    pub fn skeleton_orders(&self) -> &[u32] {
        &self.skeleton_orders
    }

    pub fn coefficient(&self, id: u32) -> &Exact {
        self.coefficients.get(id)
    }

    pub fn coefficient_count(&self) -> usize {
        self.coefficients.values.len()
    }

    // -- B2: combinations ------------------------------------------------

    /// `CompressedColour.kind`.
    fn kind_of(&self, s: usize) -> Option<Kind> {
        let mut seen = [false; 3];
        for &c in self.skeleton(s) {
            seen[self.contents[c as usize].position as usize] = true;
        }
        match seen {
            [true, false, false] => Some(Kind::AdjointRing),
            [false, true, false] => Some(Kind::FundamentalTrace),
            [_, _, true] if seen[0] || seen[1] => Some(Kind::QuarkRing),
            _ => None,
        }
    }

    fn intern_combination(&mut self, combination: Vec<(u32, u32)>) -> u32 {
        if let Some(&id) = self.combination_index.get(&combination) {
            return id;
        }
        let id = self.combinations.len() as u32;
        self.combination_index.insert(combination.clone(), id);
        self.combinations.push(combination);
        id
    }

    /// `sorted(out, key=repr)` for the (primitive, coefficient) pairs of one
    /// combination: its primitives are distinct, and the `repr` of a
    /// primitive is never a proper prefix of another's, so the primitives
    /// decide.
    fn sort_combination(&self, combination: &mut [(u32, u32)]) {
        combination.sort_by(|a, b| self.primitives.compare(&self.descriptors, a.0, b.0));
    }

    /// `CompressedColour._combination`: the primitive expansion of the
    /// product of the position terms (None: a product does not close).
    fn product_combination(
        &mut self,
        kind: Kind,
        lists: &[u32],
    ) -> RusticolResult<Option<Vec<(u32, u32)>>> {
        let sizes: Vec<usize> = lists
            .iter()
            .map(|&l| self.term_lists[l as usize].len())
            .collect();
        if sizes.contains(&0) {
            return Ok(Some(Vec::new()));
        }
        let mut out: Vec<(u32, Exact)> = Vec::new();
        let mut slot: FastMap<u32, usize> = FastMap::default();
        let mut choice = vec![0usize; lists.len()];
        loop {
            let terms: Vec<(u32, u32)> = choice
                .iter()
                .zip(lists)
                .map(|(&k, &l)| self.term_lists[l as usize][k])
                .collect();
            let (primitive, sign) = if kind == Kind::QuarkRing {
                let ring: Vec<u32> = terms.iter().map(|&(_, d)| d).collect();
                let k = self.descriptors.canonical_rotation(&ring);
                let rotated = ring[k..].iter().chain(&ring[..k]).copied().collect();
                (
                    self.primitives.intern(Primitive {
                        kind,
                        word: Word::Ring(rotated),
                    }),
                    1,
                )
            } else {
                let descriptors: Vec<&Descriptor> = terms
                    .iter()
                    .map(|&(_, d)| self.descriptors.get(d))
                    .collect();
                let Some(word) = ring_word(kind, &descriptors) else {
                    return Ok(None);
                };
                let (word, sign) = canonical_letters(kind, &word);
                (
                    self.primitives.intern(Primitive {
                        kind,
                        word: Word::Letters(word),
                    }),
                    sign,
                )
            };
            let mut coefficient = Exact::one();
            for &(c, _) in &terms {
                if c != Coefficients::ONE {
                    coefficient = coefficient.mul(self.coefficients.get(c));
                }
            }
            if sign == -1 {
                coefficient = coefficient.neg();
            }
            match slot.get(&primitive) {
                Some(&k) => out[k].1 = out[k].1.add(&coefficient),
                None => {
                    slot.insert(primitive, out.len());
                    out.push((primitive, coefficient));
                }
            }
            // Next choice: the last position varies fastest (itertools.product).
            let mut position = lists.len();
            loop {
                if position == 0 {
                    let mut combination: Vec<(u32, u32)> = out
                        .into_iter()
                        .filter(|(_, c)| !c.is_zero())
                        .map(|(p, c)| (p, self.coefficients.intern(c)))
                        .collect();
                    self.sort_combination(&mut combination);
                    return Ok(Some(combination));
                }
                position -= 1;
                choice[position] += 1;
                if choice[position] < sizes[position] {
                    break;
                }
                choice[position] = 0;
            }
        }
    }

    /// `CompressedColour._ring_combination`: through the combination of the
    /// key with its labels renamed 1, 2, ... in reading order.
    fn ring_combination(
        &mut self,
        kind: Kind,
        lists: &[u32],
    ) -> RusticolResult<Option<Vec<(u32, u32)>>> {
        let mut rename: FastMap<i64, i64> = FastMap::default();
        let mut order: Vec<i64> = Vec::new();
        for &l in lists {
            for &(_, d) in &self.term_lists[l as usize] {
                for &label in self.descriptors.get(d).labels() {
                    rename.entry(label).or_insert_with(|| {
                        order.push(label);
                        order.len() as i64
                    });
                }
            }
        }
        let mut renamed = Vec::with_capacity(lists.len());
        for &l in lists {
            let terms = self.term_lists[l as usize].clone();
            let mut list = Vec::with_capacity(terms.len());
            for (c, d) in terms {
                let descriptor = self.descriptors.get(d).renamed(|a| rename[&a]);
                list.push((c, self.descriptors.intern(descriptor)?));
            }
            let id = match self.term_index.get(&list) {
                Some(&id) => id,
                None => {
                    let id = self.term_lists.len() as u32;
                    self.term_index.insert(list.clone(), id);
                    self.term_lists.push(list);
                    id
                }
            };
            renamed.push(id);
        }
        let key = (kind, renamed);
        let found = match self.renamed_memo.get(&key) {
            Some(&found) => found,
            None => {
                let found = self
                    .product_combination(kind, &key.1)?
                    .map(|combination| self.intern_combination(combination));
                self.renamed_memo.insert(key, found);
                found
            }
        };
        let Some(found) = found else {
            return Ok(None);
        };
        let mut out = Vec::new();
        for (p, c) in self.combinations[found as usize].clone() {
            let Word::Letters(word) = &self.primitives.get(p).word else {
                unreachable!("ring combinations have letter words")
            };
            let back: Vec<i64> = word.iter().map(|&a| order[(a - 1) as usize]).collect();
            let (word, sign) = canonical_letters(kind, &back);
            let primitive = self.primitives.intern(Primitive {
                kind,
                word: Word::Letters(word),
            });
            let coefficient = if sign == 1 {
                c
            } else {
                let negated = self.coefficients.get(c).neg();
                self.coefficients.intern(negated)
            };
            out.push((primitive, coefficient));
        }
        self.sort_combination(&mut out);
        Ok(Some(out))
    }

    /// `CompressedColour.combination` of skeleton `s` (a combination id).
    fn combination(&mut self, s: usize) -> RusticolResult<Option<u32>> {
        let Some(kind) = self.kinds[s] else {
            return Ok(None);
        };
        let lists: Vec<u32> = self
            .skeleton(s)
            .iter()
            .map(|&c| self.contents[c as usize].terms)
            .collect();
        let key = (kind, lists);
        if let Some(&found) = self.combination_memo.get(&key) {
            return Ok(found);
        }
        let found = if kind == Kind::QuarkRing {
            self.product_combination(kind, &key.1)?
        } else {
            self.ring_combination(kind, &key.1)?
        };
        let found = found.map(|combination| self.intern_combination(combination));
        self.combination_memo.insert(key, found);
        Ok(found)
    }

    /// Kinds and combinations of every skeleton (with `compress`), and the
    /// candidates of `CompressedColour.certificate`'s smallest single-leg
    /// instances: per kind, in the order the kinds first occur, the
    /// skeletons of the smallest size; the caller breaks the tie by the
    /// `repr` of their contents (the first smallest wins).
    pub fn analyse(&mut self) -> RusticolResult<Vec<(Kind, Vec<usize>)>> {
        if self.analysed {
            return Err(invalid("analyse runs once"));
        }
        self.analysed = true;
        let count = self.skeletons();
        self.kinds = if self.compress {
            (0..count).map(|s| self.kind_of(s)).collect()
        } else {
            vec![None; count]
        };
        self.skeleton_combinations = Vec::with_capacity(count);
        let mut candidates: Vec<(Kind, usize, Vec<usize>)> = Vec::new();
        for s in 0..count {
            let combination = self.combination(s)?;
            self.skeleton_combinations.push(combination);
            if combination.is_none() {
                continue;
            }
            let single = self
                .skeleton(s)
                .iter()
                .all(|&c| self.contents[c as usize].single);
            if !single {
                continue;
            }
            let kind = self.kinds[s].expect("a combination has a kind");
            let size = self.skeleton(s).len();
            match candidates.iter_mut().find(|(k, _, _)| *k == kind) {
                None => candidates.push((kind, size, vec![s])),
                Some((_, best, list)) => match size.cmp(best) {
                    Ordering::Less => {
                        *best = size;
                        list.clear();
                        list.push(s);
                    }
                    Ordering::Equal => list.push(s),
                    Ordering::Greater => {}
                },
            }
        }
        Ok(candidates
            .into_iter()
            .map(|(kind, _, list)| (kind, list))
            .collect())
    }

    /// The combination of skeleton `s` (after [`Self::analyse`]).
    pub fn combination_of(&self, s: usize) -> Option<&[(u32, u32)]> {
        self.skeleton_combinations
            .get(s)
            .copied()
            .flatten()
            .map(|c| self.combinations[c as usize].as_slice())
    }

    /// A primitive's kind and word.
    pub fn primitive(&self, id: u32) -> (Kind, &Word) {
        let primitive = self.primitives.get(id);
        (primitive.kind, &primitive.word)
    }

    // -- B3: classes -------------------------------------------------------

    /// `_ring_relabelling`: (canonical relabelled ring, its rotation of the
    /// original descriptors, forward map, back map).
    fn ring_relabelling(&mut self, ring: &[u32]) -> RusticolResult<Relabelling> {
        let mut seen: Vec<i64> = Vec::new();
        for &d in ring {
            for &label in self.descriptors.get(d).labels() {
                if self.adjoint.binary_search(&label).is_ok() && !seen.contains(&label) {
                    seen.push(label);
                }
            }
        }
        let mut sorted = seen.clone();
        sorted.sort_unstable();
        let forward: Vec<(i64, i64)> = seen.iter().copied().zip(sorted).collect();
        let renamed: Vec<u32> = if forward.iter().all(|(a, b)| a == b) {
            ring.to_vec()
        } else {
            let map: FastMap<i64, i64> = forward.iter().copied().collect();
            ring.iter()
                .map(|&d| {
                    let descriptor = self
                        .descriptors
                        .get(d)
                        .renamed(|a| *map.get(&a).unwrap_or(&a));
                    self.descriptors.intern(descriptor)
                })
                .collect::<RusticolResult<_>>()?
        };
        let k = self.descriptors.canonical_rotation(&renamed);
        let rotate = |r: &[u32]| -> Vec<u32> { r[k..].iter().chain(&r[..k]).copied().collect() };
        let identity = forward.iter().all(|(a, b)| a == b);
        Ok((
            rotate(&renamed),
            rotate(ring),
            if identity { Vec::new() } else { forward },
        ))
    }

    /// The reductions the class keys need (after [`Self::analyse`]; the
    /// caller certified `certified`): the distinct canonical quark rings,
    /// and one skeleton per distinct content sequence that keeps a generic
    /// seed (`SeedColour.seed`).
    pub fn needs(&mut self, certified: &[Kind]) -> RusticolResult<(&[RingNeed], &[usize])> {
        if !self.analysed || self.has_needs {
            return Err(invalid("needs runs once, after analyse"));
        }
        self.has_needs = true;
        self.certified = certified.to_vec();
        let count = self.skeletons();
        let mut generic_index: FastMap<Vec<u32>, u32> = FastMap::default();
        self.sources = Vec::with_capacity(count);
        for s in 0..count {
            let combination = match (self.kinds[s], self.skeleton_combinations[s]) {
                (Some(kind), Some(c)) if certified.contains(&kind) => Some(c),
                _ => None,
            };
            let source = match combination {
                Some(c) => Source::Combination(c),
                None => {
                    let contents = self.skeleton(s).to_vec();
                    let next = self.generic_skeletons.len() as u32;
                    let g = *generic_index.entry(contents).or_insert(next);
                    if g == next {
                        self.generic_skeletons.push(s);
                    }
                    Source::Generic(g)
                }
            };
            self.sources.push(source);
        }
        if self.compress {
            let mut rings: Vec<u32> = self
                .existing
                .iter()
                .filter_map(|seed| match seed {
                    Seed::Primitive(p) => Some(*p),
                    _ => None,
                })
                .collect();
            for source in &self.sources {
                if let Source::Combination(c) = source {
                    rings.extend(self.combinations[*c as usize].iter().map(|&(p, _)| p));
                }
            }
            for p in rings {
                if self.ring_of.contains_key(&p) {
                    continue;
                }
                let Word::Ring(ring) = self.primitives.get(p).word.clone() else {
                    continue;
                };
                let (canonical, original, forward) = self.ring_relabelling(&ring)?;
                let next = self.rings.len() as u32;
                let need = *self.ring_index.entry(canonical).or_insert(next);
                if need == next {
                    self.rings.push(RingNeed {
                        descriptors: original,
                        forward: forward.clone(),
                    });
                }
                let back = forward.iter().map(|&(a, b)| (b, a)).collect();
                self.ring_of.insert(p, (need, back));
            }
        }
        Ok((&self.rings, &self.generic_skeletons))
    }

    /// The input index of descriptor `id` (descriptors of class words and
    /// ring needs are always input descriptors).
    pub fn input_descriptor(&self, id: u32) -> RusticolResult<u32> {
        self.spec_of
            .get(&id)
            .copied()
            .ok_or_else(|| invalid("a relabelled descriptor has no input index"))
    }

    /// The rows of `_extended` (after [`Self::needs`]): `rings[k]` is
    /// `canonical_seed(_reduced_primitive(...))` of ring need k, `generics[g]`
    /// the seed of generic need g.
    pub fn complete(
        &mut self,
        rings: Vec<Tensor>,
        generics: Vec<Tensor>,
    ) -> RusticolResult<SeedOutput> {
        if !self.has_needs {
            return Err(invalid("complete runs after needs"));
        }
        if rings.len() != self.rings.len() || generics.len() != self.generic_skeletons.len() {
            return Err(invalid("one tensor per need"));
        }
        let mut keys = Keys {
            merge: self.compress,
            colour_tensors: Vec::new(),
            colour_index: FastMap::default(),
            proportional: FastMap::default(),
            exact: FastMap::default(),
            cache: FastMap::default(),
        };
        // Existing classes keep their index (`index.setdefault`).
        let mut index: FastMap<ClassKey, u32> = FastMap::default();
        let existing = self.existing.clone();
        let mut classes: Vec<Seed> = existing.clone();
        let mut scales: Vec<u32> = Vec::with_capacity(classes.len());
        let mut orders: Vec<Vec<u32>> = self.existing_orders.clone();
        for (c, &seed) in existing.iter().enumerate() {
            let (key, scale) = keys.key(self, &rings, &generics, seed)?;
            index.entry(key).or_insert(c as u32);
            scales.push(scale);
        }
        let mut ratios: FastMap<(u32, u32), u32> = FastMap::default();
        let mut memo: FastMap<(Source, u32), u32> = FastMap::default();
        let mut rows: Vec<Vec<(u32, u32)>> = Vec::new();
        let mut row_index: FastMap<Vec<(u32, u32)>, u32> = FastMap::default();
        let mut skeleton_rows = Vec::with_capacity(self.skeletons());
        for s in 0..self.skeletons() {
            let source = self.sources[s];
            let order = self.skeleton_orders[s];
            if let Some(&row) = memo.get(&(source, order)) {
                skeleton_rows.push(row);
                continue;
            }
            let terms: Vec<(Seed, u32)> = match source {
                Source::Combination(c) => self.combinations[c as usize]
                    .iter()
                    .map(|&(p, x)| (Seed::Primitive(p), x))
                    .collect(),
                Source::Generic(g) => vec![(Seed::Generic(g), Coefficients::ONE)],
            };
            let mut merged: Vec<(u32, Exact)> = Vec::new();
            for (seed, x) in terms {
                // row(seed, order)
                let (key, scale) = keys.key(self, &rings, &generics, seed)?;
                let c = match index.get(&key) {
                    Some(&c) => c,
                    None => {
                        let c = classes.len() as u32;
                        index.insert(key, c);
                        classes.push(seed);
                        scales.push(scale);
                        orders.push(Vec::new());
                        c
                    }
                };
                let class_orders = &mut orders[c as usize];
                if !class_orders.contains(&order) {
                    class_orders.push(order);
                }
                let pair = (scale, scales[c as usize]);
                let factor = match ratios.get(&pair) {
                    Some(&f) => f,
                    None => {
                        let f = if pair.0 == pair.1 {
                            Coefficients::ONE
                        } else {
                            let value = self
                                .coefficients
                                .get(pair.0)
                                .mul(&self.coefficients.get(pair.1).inverse()?);
                            self.coefficients.intern(value)
                        };
                        ratios.insert(pair, f);
                        f
                    }
                };
                let term = if factor == Coefficients::ONE {
                    self.coefficients.get(x).clone()
                } else {
                    self.coefficients.get(x).mul(self.coefficients.get(factor))
                };
                match merged.iter_mut().find(|(class, _)| *class == c) {
                    Some((_, value)) => *value = value.add(&term),
                    None => merged.push((c, term)),
                }
            }
            let row: Vec<(u32, u32)> = merged
                .into_iter()
                .map(|(c, value)| (c, self.coefficients.intern(value)))
                .collect();
            let next = rows.len() as u32;
            let id = *row_index.entry(row.clone()).or_insert(next);
            if id == next {
                rows.push(row);
            }
            memo.insert((source, order), id);
            skeleton_rows.push(id);
        }
        for class_orders in &mut orders {
            class_orders.sort_unstable();
        }
        Ok(SeedOutput {
            classes: classes[existing.len()..].to_vec(),
            orders,
            rows,
            skeleton_rows,
        })
    }
}

// ---------------------------------------------- native reductions (phase 1)

/// A colour network as `loops/v2/native_seeds.py` encodes it: per product
/// `(re, im, N_c power, 1/sqrt(2) power, strings (row, word, col), traces)`
/// with template labels (`networks.rs`: external legs non-negative, `-1` the
/// loop index in, `-2` out, `-(3 + k)` local label `k`).
pub type NetworkSpec = Vec<(
    String,
    String,
    i64,
    i64,
    Vec<(i64, Vec<i64>, i64)>,
    Vec<Vec<i64>>,
)>;

pub fn parse_network(spec: &NetworkSpec) -> RusticolResult<Network> {
    spec.iter()
        .map(|(re, im, nc_power, root2_power, strings, traces)| {
            Ok(Product {
                coefficient: Exact::parse(re, im)?,
                strings: strings
                    .iter()
                    .map(|(row, word, col)| NetString {
                        row: *row,
                        word: word.clone(),
                        col: *col,
                    })
                    .collect(),
                traces: traces.clone(),
                nc_power: *nc_power,
                root2_power: *root2_power,
            })
        })
        .collect()
}

/// Factored networks (`native_seeds._Networks.factored`): the distinct
/// factor networks and, per key, the factor ids whose expanded product
/// ([`networks::multiply`], in list order) is its network
/// (`CompressedColour._exact_network`: the relabelled vertex structure, then
/// each attached block's word).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FactoredSpec {
    pub factors: Vec<NetworkSpec>,
    pub contents: Vec<(u32, Vec<u32>)>,
}

/// The networks of a [`FactoredSpec`] by key.
pub fn factored_networks(spec: &FactoredSpec) -> RusticolResult<FastMap<u32, Network>> {
    let factors = spec
        .factors
        .iter()
        .map(parse_network)
        .collect::<RusticolResult<Vec<Network>>>()?;
    spec.contents
        .iter()
        .map(|(key, ids)| {
            let parts = ids
                .iter()
                .map(|&f| {
                    factors
                        .get(f as usize)
                        .ok_or_else(|| invalid("a content refers to an unknown network factor"))
                })
                .collect::<RusticolResult<Vec<&Network>>>()?;
            Ok((*key, networks::multiply(parts)))
        })
        .collect()
}

/// `_adjoint_chain(letters, in, out, p)`: `(F^l_1 ... F^l_k)_{in out}` with
/// local chain labels.
fn adjoint_chain(letters: &[i64]) -> Network {
    let trace = |coefficient: Exact, labels: Vec<i64>| Product {
        coefficient,
        traces: vec![labels],
        ..Product::one()
    };
    if letters.is_empty() {
        return vec![trace(Exact::one(), vec![networks::IN, networks::OUT])];
    }
    let k = letters.len();
    let label = |i: usize| -> i64 {
        if i == 0 {
            networks::IN
        } else if i == k {
            networks::OUT
        } else {
            networks::local(i - 1)
        }
    };
    let factors: Vec<Network> = letters
        .iter()
        .enumerate()
        .map(|(i, &letter)| {
            let (x, y) = (label(i), label(i + 1));
            vec![
                trace(Exact::one(), vec![x, letter, y]),
                trace(Exact::one().neg(), vec![x, y, letter]),
            ]
        })
        .collect();
    networks::multiply(factors.iter())
}

impl SeedBuilder {
    /// The stage's content ids read by the generic needs (sorted): the
    /// contents whose exact networks [`Self::reduce`] needs.
    pub fn generic_contents(&self) -> Vec<u32> {
        let mut out: Vec<u32> = self
            .generic_skeletons
            .iter()
            .flat_map(|&s| self.skeleton(s).iter().copied())
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The input indices of the exact descriptors of the ring needs
    /// (sorted): the descriptors whose networks [`Self::reduce`] needs.
    pub fn ring_exact_descriptors(&self) -> RusticolResult<Vec<u32>> {
        let mut out = Vec::new();
        for need in &self.rings {
            for &d in &need.descriptors {
                if let Descriptor::Exact(..) = self.descriptors.get(d) {
                    out.push(self.input_descriptor(d)?);
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        Ok(out)
    }

    /// The position template of ring-need descriptor `d` with the ring's
    /// relabelling `forward` (`_ring_relabelling`) applied to its labels.
    fn ring_template(
        &self,
        d: u32,
        forward: &FastMap<i64, i64>,
        exact: &FastMap<u32, Network>,
    ) -> RusticolResult<Network> {
        let rename = |a: i64| -> i64 {
            if a >= 0 {
                *forward.get(&a).unwrap_or(&a)
            } else {
                a
            }
        };
        let template = match self.descriptors.get(d) {
            Descriptor::Adjoint(letters) => adjoint_chain(letters),
            Descriptor::Fundamental(letters, forward_flow) => {
                let (row, col) = if *forward_flow {
                    (networks::OUT, networks::IN)
                } else {
                    (networks::IN, networks::OUT)
                };
                vec![Product {
                    strings: vec![NetString {
                        row,
                        word: letters.clone(),
                        col,
                    }],
                    ..Product::one()
                }]
            }
            Descriptor::Exact(..) => exact
                .get(&self.input_descriptor(d)?)
                .cloned()
                .ok_or_else(|| invalid("an exact descriptor of a ring without its network"))?,
        };
        Ok(template
            .into_iter()
            .map(|product| Product {
                strings: product
                    .strings
                    .iter()
                    .map(|s| NetString {
                        row: rename(s.row),
                        word: s.word.iter().map(|&a| rename(a)).collect(),
                        col: rename(s.col),
                    })
                    .collect(),
                traces: product
                    .traces
                    .iter()
                    .map(|t| t.iter().map(|&a| rename(a)).collect())
                    .collect(),
                ..product
            })
            .collect())
    }

    /// Reduce every need natively (after [`Self::needs`]): each ring need is
    /// the ring of its descriptors (`_reduced_primitive` of the canonical
    /// relabelled ring, on legs `ring_legs`), each generic need the ring of
    /// its contents' exact networks (`SeedColour.seed`, on legs
    /// `seed_legs`); `exact` holds the networks of
    /// [`Self::ring_exact_descriptors`] by input index, `contents` those of
    /// [`Self::generic_contents`]. The needs reduce in parallel; each result
    /// depends on its need only.
    pub fn reduce(
        &mut self,
        ring_legs: &[i64],
        seed_legs: &[i64],
        exact: &FastMap<u32, Network>,
        contents: &FastMap<u32, Network>,
    ) -> RusticolResult<()> {
        if !self.has_needs || self.reduced.is_some() {
            return Err(invalid("reduce runs once, after needs"));
        }
        let rings: Vec<Network> = self
            .rings
            .iter()
            .map(|need| {
                let forward: FastMap<i64, i64> = need.forward.iter().copied().collect();
                let templates = need
                    .descriptors
                    .iter()
                    .map(|&d| self.ring_template(d, &forward, exact))
                    .collect::<RusticolResult<Vec<_>>>()?;
                Ok(networks::ring(&templates.iter().collect::<Vec<_>>()))
            })
            .collect::<RusticolResult<_>>()?;
        let generics: Vec<Network> = self
            .generic_skeletons
            .iter()
            .map(|&s| {
                let templates = self
                    .skeleton(s)
                    .iter()
                    .map(|c| {
                        contents
                            .get(c)
                            .ok_or_else(|| invalid("a generic content without its network"))
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                Ok(networks::ring(&templates))
            })
            .collect::<RusticolResult<_>>()?;
        let reduce_all = |networks_: &[Network], legs: &[i64]| -> RusticolResult<Vec<Tensor>> {
            networks_
                .par_iter()
                .map_init(
                    || networks::Reducer::new(legs),
                    |reducer, network| reducer.reduce(network).map_err(invalid),
                )
                .collect()
        };
        let rings = reduce_all(&rings, ring_legs)?;
        let generics = reduce_all(&generics, seed_legs)?;
        self.reduced = Some((rings, generics));
        Ok(())
    }

    /// The natively reduced needs (after [`Self::reduce`]).
    pub fn reduced(&self) -> Option<(&[Tensor], &[Tensor])> {
        self.reduced
            .as_ref()
            .map(|(rings, generics)| (rings.as_slice(), generics.as_slice()))
    }

    /// [`Self::complete`] with the natively reduced needs.
    pub fn complete_reduced(&mut self) -> RusticolResult<SeedOutput> {
        let (rings, generics) = self
            .reduced
            .clone()
            .ok_or_else(|| invalid("complete_reduced runs after reduce"))?;
        self.complete(rings, generics)
    }
}

/// `_ClassKeys` of one stage.
struct Keys {
    merge: bool,
    colour_tensors: Vec<ColourTensor>,
    colour_index: FastMap<ColourTensor, u32>,
    proportional: FastMap<ProportionalKey, u32>,
    exact: FastMap<ExactKey, u32>,
    cache: FastMap<Seed, (ClassKey, u32)>,
}

impl Keys {
    fn colour(&mut self, tensor: ColourTensor) -> u32 {
        if let Some(&id) = self.colour_index.get(&tensor) {
            return id;
        }
        let id = self.colour_tensors.len() as u32;
        self.colour_index.insert(tensor.clone(), id);
        self.colour_tensors.push(tensor);
        id
    }

    /// `_proportional_key(tensor, rename)`: (key, s) with tensor = s T_key.
    fn proportional(
        &mut self,
        builder: &mut SeedBuilder,
        tensor: &Tensor,
        rename: &[(i64, i64)],
    ) -> RusticolResult<(ClassKey, u32)> {
        if tensor.terms.is_empty() {
            return Ok((ClassKey::Zero, Coefficients::ONE));
        }
        let mut terms: Vec<(ColourTensor, i64, &Exact)> = if rename.is_empty() {
            tensor
                .terms
                .iter()
                .map(|(c, t, p)| (t.clone(), *p, c))
                .collect()
        } else {
            let map: FastMap<i64, i64> = rename.iter().copied().collect();
            let mut terms: Vec<(ColourTensor, i64, &Exact)> = tensor
                .terms
                .iter()
                .map(|(c, t, p)| (t.relabelled(&map), *p, c))
                .collect();
            terms.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
            terms
        };
        let first = terms[0].2.clone();
        let inverse = first.inverse()?;
        let key: Vec<(u32, i64, u32)> = terms
            .drain(..)
            .map(|(t, p, c)| {
                let id = self.colour(t);
                (id, p, builder.coefficients.intern(c.mul(&inverse)))
            })
            .collect();
        let key = (tensor.normalization, key);
        let next = self.proportional.len() as u32;
        let id = *self.proportional.entry(key).or_insert(next);
        Ok((
            ClassKey::Proportional(id),
            builder.coefficients.intern(first),
        ))
    }

    /// Exact equality of generic seeds (no merge).
    fn exact(&mut self, builder: &mut SeedBuilder, tensor: &Tensor) -> ClassKey {
        let terms: Vec<(u32, u32, i64)> = tensor
            .terms
            .iter()
            .map(|(c, t, p)| {
                (
                    builder.coefficients.intern(c.clone()),
                    self.colour(t.clone()),
                    *p,
                )
            })
            .collect();
        let key = (tensor.normalization, terms);
        let next = self.exact.len() as u32;
        ClassKey::Exact(*self.exact.entry(key).or_insert(next))
    }

    fn key(
        &mut self,
        builder: &mut SeedBuilder,
        rings: &[Tensor],
        generics: &[Tensor],
        seed: Seed,
    ) -> RusticolResult<(ClassKey, u32)> {
        if let Some(&found) = self.cache.get(&seed) {
            return Ok(found);
        }
        let found = match seed {
            Seed::Primitive(p) => {
                if self.merge && builder.primitives.get(p).kind == Kind::QuarkRing {
                    let (need, back) = builder
                        .ring_of
                        .get(&p)
                        .cloned()
                        .ok_or_else(|| invalid("a quark ring without its reduction"))?;
                    self.proportional(builder, &rings[need as usize], &back)?
                } else {
                    (ClassKey::Primitive(p), Coefficients::ONE)
                }
            }
            Seed::Generic(g) => {
                let tensor = &generics[g as usize];
                if self.merge {
                    self.proportional(builder, tensor, &[])?
                } else {
                    (self.exact(builder, tensor), Coefficients::ONE)
                }
            }
            Seed::Tensor(t) => {
                let tensor = builder.tensors[t as usize].clone();
                if self.merge {
                    self.proportional(builder, &tensor, &[])?
                } else {
                    (self.exact(builder, &tensor), Coefficients::ONE)
                }
            }
        };
        self.cache.insert(seed, found);
        Ok(found)
    }
}

// --------------------------------------------------------------- liveness

/// An order vector: (name id, power), the caller's name ids.
pub type OrderVector = Vec<(u16, i64)>;
/// (basis element, Born order, N_c grades of its amplitude).
pub type BornGrades = (usize, OrderVector, Vec<i64>);

/// One criterion of `_prune_colour`: a seed table's nonzero cell powers per
/// (class, basis element) and the leading degree P of each sector.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LivenessTable {
    pub powers: Vec<Vec<Vec<i64>>>,
    pub degrees: Vec<Option<i64>>,
}

/// The input of `_prune_colour`'s `class_live`; orders are the stage's
/// order vectors (caller's name ids).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LivenessSpec {
    /// "lc" or "nlc".
    pub accuracy: String,
    pub tables: Vec<LivenessTable>,
    /// Per sector its (Born order, loop order) pairs.
    pub sectors: Vec<Vec<(OrderVector, OrderVector)>>,
    /// Per basis element its Born orders.
    pub basis_orders: Vec<Vec<Vec<(u16, i64)>>>,
    /// (basis element, Born order, N_c grades of its amplitude).
    pub grades: Vec<BornGrades>,
}

/// `colour.kept`.
fn kept(power: i64, degree: Option<i64>, depth: i64) -> bool {
    degree.is_some_and(|d| d - depth <= power && power <= d)
}

impl SeedBuilder {
    /// `_prune_colour`'s `class_live(c, order)` for every class and loop
    /// order of `output`, and `keep` of every skeleton of this stage.
    pub fn liveness(
        &mut self,
        output: &SeedOutput,
        spec: &LivenessSpec,
    ) -> RusticolResult<(Vec<Vec<bool>>, Vec<bool>)> {
        let depth = match spec.accuracy.as_str() {
            "lc" => 0,
            "nlc" => 2,
            other => return Err(invalid(format!("no prune at accuracy {other:?}"))),
        };
        let mut intern = |o: &Vec<(u16, i64)>| self.orders.intern(o.clone());
        let sectors: Vec<Vec<(u32, u32)>> = spec
            .sectors
            .iter()
            .map(|pairs| pairs.iter().map(|(a, l)| (intern(a), intern(l))).collect())
            .collect();
        let basis: Vec<Vec<u32>> = spec
            .basis_orders
            .iter()
            .map(|orders| orders.iter().map(&mut intern).collect())
            .collect();
        let mut grades: FastMap<(usize, u32), Vec<i64>> = FastMap::default();
        for (b, a, values) in &spec.grades {
            grades.insert((*b, intern(a)), values.clone());
        }
        for table in &spec.tables {
            if table.degrees.len() != sectors.len() {
                return Err(invalid("one degree per sector"));
            }
            if table.powers.len() < output.orders.len()
                || table.powers.iter().any(|row| row.len() != basis.len())
            {
                return Err(invalid("cell powers per class and basis element"));
            }
        }
        let live = |c: usize, order: u32| -> bool {
            spec.tables.iter().any(|table| {
                sectors.iter().enumerate().any(|(s, pairs)| {
                    pairs.iter().any(|&(a, loop_order)| {
                        loop_order == order
                            && basis.iter().enumerate().any(|(b, born_orders)| {
                                born_orders.contains(&a)
                                    && table.powers[c][b].iter().any(|&power| {
                                        grades.get(&(b, a)).is_some_and(|values| {
                                            values.iter().any(|&grade| {
                                                kept(power + grade, table.degrees[s], depth)
                                            })
                                        })
                                    })
                            })
                    })
                })
            })
        };
        let flags: Vec<Vec<bool>> = output
            .orders
            .iter()
            .enumerate()
            .map(|(c, orders)| orders.iter().map(|&o| live(c, o)).collect())
            .collect();
        let is_live = |c: u32, order: u32| -> bool {
            output.orders[c as usize]
                .iter()
                .position(|&o| o == order)
                .is_some_and(|k| flags[c as usize][k])
        };
        let keep = output
            .skeleton_rows
            .iter()
            .zip(&self.skeleton_orders)
            .map(|(&row, &order)| {
                output.rows[row as usize]
                    .iter()
                    .any(|&(c, _)| is_live(c, order))
            })
            .collect();
        Ok((flags, keep))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuple_reprs_compare_as_python_strings() {
        let compare = |a: &[&str], b: &[&str]| {
            compare_chunks(
                &tuple_chunks(a.iter().copied()),
                &tuple_chunks(b.iter().copied()),
            )
        };
        // "(a, b)" < "(a,)" since ' ' < ')'.
        assert_eq!(compare(&["a", "b"], &["a"]), Ordering::Less);
        assert_eq!(compare(&["a"], &["a"]), Ordering::Equal);
        // "()" < "(a,)" since ')' < 'a'.
        assert_eq!(compare(&[], &["a"]), Ordering::Less);
        assert_eq!(int_tuple_repr(&[1]), "(1,)");
        assert_eq!(int_tuple_repr(&[1, 10]), "(1, 10)");
        assert_eq!(int_tuple_repr(&[]), "()");
    }

    #[test]
    fn canonical_letters_follow_python_min() {
        // Rotations only for traces; reflections with (-1)^n for rings.
        assert_eq!(
            canonical_letters(Kind::FundamentalTrace, &[3, 1, 2]),
            (vec![1, 2, 3], 1)
        );
        assert_eq!(
            canonical_letters(Kind::AdjointRing, &[1, 3, 2]),
            (vec![1, 2, 3], -1)
        );
        assert_eq!(
            canonical_letters(Kind::AdjointRing, &[2, 1]),
            (vec![1, 2], 1)
        );
        assert_eq!(canonical_letters(Kind::AdjointRing, &[]), (vec![], 1));
    }

    #[test]
    fn fundamental_ring_words_close_one_loop() {
        let f = |letters: Vec<i64>, forward| Descriptor::Fundamental(letters, forward);
        let (a, b, c) = (f(vec![1], true), f(vec![2, 3], true), f(vec![4], true));
        assert_eq!(
            ring_word(Kind::FundamentalTrace, &[&a, &b, &c]),
            Some(vec![1, 4, 2, 3])
        );
        let (x, y) = (f(vec![1], true), f(vec![2], false));
        assert_eq!(ring_word(Kind::FundamentalTrace, &[&x, &y]), None);
        let (u, v) = (f(vec![1], false), f(vec![2], false));
        assert_eq!(
            ring_word(Kind::FundamentalTrace, &[&u, &v]),
            Some(vec![1, 2])
        );
    }

    #[test]
    fn exact_inverse_and_text() {
        let x = Exact::parse("1/2", "-3").unwrap();
        let y = x.mul(&x.inverse().unwrap());
        assert_eq!(y, Exact::one());
        assert_eq!(x.text(), ("1/2".to_owned(), "-3/1".to_owned()));
        assert_eq!(
            x.repr(),
            "ExactColorCoefficient(real=Fraction(1, 2), imag=Fraction(-3, 1))"
        );
    }

    #[test]
    fn traces_take_their_smallest_rotation() {
        let t = ColourTensor::new(vec![], vec![vec![3, 1, 2], vec![2, 1]]);
        assert_eq!(t.traces, vec![vec![1, 2], vec![1, 2, 3]]);
    }
}
