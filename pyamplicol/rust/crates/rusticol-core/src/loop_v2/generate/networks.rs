// SPDX-License-Identifier: 0BSD

//! Exact SU(N_c) reduction of colour networks (WP3.1 I10, package
//! native-seeds): the native twin of `loops/v2/colour.py` `reduce_network`
//! (with `_join_strings`, `_Numbering`, `_fierz` and `_decode`) and of the
//! ring products of `_ring_network` (quark rings, `_reduced_primitive`) and
//! `SeedColour.seed` (generic seeds).
//!
//! A network is a sum of products of generator strings and traces with an
//! exact coefficient, an explicit N_c power and a power of 1/sqrt(2)
//! (`ColourProduct`). Every internal index is contracted: fundamental ones
//! by joining strings, adjoint ones by the SU(N_c) Fierz identities on traces
//! (open strings ride along as traces closed by a sentinel), exactly as the
//! Python stage does, so the reduced tensor (`ConnectedColorTensor`: combined
//! terms in canonical order, the 1/sqrt(2) parity folded) is the oracle's.
//!
//! Labels are integers: external legs keep their (non-negative) labels; every
//! other label is internal. A ring position is a *template* whose label
//! [`IN`] is the loop index entering it, [`OUT`] the one leaving it and
//! `-(3 + k)` its local label `k` ([`local`]); [`ring`] instantiates the
//! templates of a ring and multiplies them out.
//!
//! No float and no Symbolica licence call; the coefficients of the Fierz
//! steps are exact integers of any size (`symbolica` `Integer`, a machine
//! word until it needs more), as the oracle's Python integers: no domain cap.

use super::FastMap;
use super::seeds::{ColourTensor, Exact, OpenString, Tensor};
use symbolica::domains::integer::Integer;
use symbolica::domains::rational::Rational;

pub type Label = i64;

/// The loop index entering a template position.
pub const IN: Label = -1;
/// The loop index leaving a template position.
pub const OUT: Label = -2;

/// The template label of local label `k` of a position.
pub fn local(k: usize) -> Label {
    -(3 + k as Label)
}

/// `ColourString`: `(tau^word[0] ... tau^word[-1])` with upper index `row`
/// and lower index `col`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NetString {
    pub row: Label,
    pub word: Vec<Label>,
    pub col: Label,
}

/// `ColourProduct`: one product of strings and traces.
#[derive(Clone, Debug)]
pub struct Product {
    pub coefficient: Exact,
    pub strings: Vec<NetString>,
    pub traces: Vec<Vec<Label>>,
    pub nc_power: i64,
    pub root2_power: i64,
}

/// A sum of products (`ColourNetwork`).
pub type Network = Vec<Product>;

impl Product {
    pub fn one() -> Self {
        Self {
            coefficient: Exact::one(),
            strings: Vec::new(),
            traces: Vec::new(),
            nc_power: 0,
            root2_power: 0,
        }
    }

    /// `ColourProduct.__mul__`.
    fn times(&self, other: &Self) -> Self {
        Self {
            coefficient: self.coefficient.mul(&other.coefficient),
            strings: self.strings.iter().chain(&other.strings).cloned().collect(),
            traces: self.traces.iter().chain(&other.traces).cloned().collect(),
            nc_power: self.nc_power + other.nc_power,
            root2_power: self.root2_power + other.root2_power,
        }
    }

    fn relabel(&self, rename: &impl Fn(Label) -> Label) -> Self {
        Self {
            coefficient: self.coefficient.clone(),
            strings: self
                .strings
                .iter()
                .map(|s| NetString {
                    row: rename(s.row),
                    word: s.word.iter().map(|&l| rename(l)).collect(),
                    col: rename(s.col),
                })
                .collect(),
            traces: self
                .traces
                .iter()
                .map(|t| t.iter().map(|&l| rename(l)).collect())
                .collect(),
            nc_power: self.nc_power,
            root2_power: self.root2_power,
        }
    }
}

/// The expanded product of networks (`multiply`).
pub fn multiply<'a>(networks: impl IntoIterator<Item = &'a Network>) -> Network {
    let mut out = vec![Product::one()];
    for network in networks {
        out = out
            .iter()
            .flat_map(|a| network.iter().map(move |b| a.times(b)))
            .collect();
    }
    out
}

/// The ring of position templates (`_ring_network`, `SeedColour.seed`):
/// position `p` reads the loop index `p - 1` (mod the ring) as [`IN`] and
/// `p` as [`OUT`]; its local labels are its own.
pub fn ring(templates: &[&Network]) -> Network {
    let count = templates.len() as Label;
    let stride = 1 + templates
        .iter()
        .flat_map(|t| t.iter())
        .flat_map(|p| {
            p.traces.iter().flatten().chain(
                p.strings
                    .iter()
                    .flat_map(|s| s.word.iter().chain([&s.row, &s.col])),
            )
        })
        .filter(|&&l| l < OUT)
        .map(|&l| -l - 3)
        .max()
        .unwrap_or(0);
    let instantiated: Vec<Network> = templates
        .iter()
        .enumerate()
        .map(|(p, template)| {
            let p = p as Label;
            let rename = |label: Label| -> Label {
                if label >= 0 {
                    label
                } else if label == IN {
                    -(1 + (p + count - 1) % count)
                } else if label == OUT {
                    -(1 + p)
                } else {
                    -(1 + count + p * stride + (-label - 3))
                }
            };
            template.iter().map(|t| t.relabel(&rename)).collect()
        })
        .collect();
    multiply(instantiated.iter())
}

fn error(message: impl Into<String>) -> String {
    format!("colour network: {}", message.into())
}

// ------------------------------------------------------------- reduction

/// A Fierz term: integer coefficient, N_c power and its traces (rows).
///
/// Each Fierz step removes one contracted label pair and splits a term into
/// coefficients +c and -c, so after k steps |c| <= 2^k. The coefficient is
/// an unbounded `Integer` (one machine word while |c| < 2^63, i.e. for
/// every network of fewer than 63 contracted pairs), so no network is
/// refused for its size.
type FierzTerm = (Integer, i64, Vec<Vec<Label>>);
/// A Fierz input: traces and the sorted sentinel labels.
type FierzKey = (Vec<Vec<Label>>, Vec<Label>);
/// Joined strings: open strings, traces of closed loops, empty loops.
type Joined = (Vec<NetString>, Vec<Vec<Label>>, i64);

/// `reduce_network(network, legs, external)` with the legs' labels external,
/// memoising the Fierz reductions of trace shapes (`_fierz`'s cache).
#[derive(Debug, Default)]
pub struct Reducer {
    external: Vec<Label>,
    first: Label,
    fierz: FastMap<FierzKey, Vec<FierzTerm>>,
}

impl Reducer {
    /// A reducer for tensors on legs labelled `legs`.
    pub fn new(legs: &[Label]) -> Self {
        let mut external = legs.to_vec();
        external.sort_unstable();
        external.dedup();
        let first = external.len() as Label + 1 + external.iter().copied().max().unwrap_or(0);
        Self {
            external,
            first,
            fierz: FastMap::default(),
        }
    }

    fn is_external(&self, label: Label) -> bool {
        self.external.binary_search(&label).is_ok()
    }

    /// The reduced tensor of `network`: combined terms in canonical order,
    /// normalisation = the parity of the 1/sqrt(2) powers (`reduce_network`;
    /// `canonical_seed` leaves it unchanged).
    pub fn reduce(&mut self, network: &Network) -> Result<Tensor, String> {
        // (coefficient, tensor, N_c power, 1/sqrt(2) power)
        let mut reduced: Vec<(Exact, ColourTensor, i64, i64)> = Vec::new();
        for term in network {
            if term.coefficient.is_zero() {
                continue;
            }
            let (opened, joined, empty) = self.join_strings(&term.strings)?;
            let mut numbering = Numbering::new(self.first);
            let mut traces: Vec<Vec<Label>> = term
                .traces
                .iter()
                .chain(&joined)
                .map(|trace| trace.iter().map(|&l| numbering.number(l, self)).collect())
                .collect();
            let mut sentinels: Vec<(Label, (Label, Label))> = Vec::new();
            for string in &opened {
                let sentinel = numbering.fresh();
                sentinels.push((sentinel, (string.row, string.col)));
                let mut trace: Vec<Label> = string
                    .word
                    .iter()
                    .map(|&l| numbering.number(l, self))
                    .collect();
                trace.push(sentinel);
                traces.push(trace);
            }
            let mut sentinel_labels: Vec<Label> = sentinels.iter().map(|s| s.0).collect();
            sentinel_labels.sort_unstable();
            let key = (traces, sentinel_labels);
            if !self.fierz.contains_key(&key) {
                let value = fierz(&key.0, &key.1);
                self.fierz.insert(key.clone(), value);
            }
            let ends: FastMap<Label, (Label, Label)> = sentinels.into_iter().collect();
            for (coefficient, power, rows) in &self.fierz[&key] {
                let tensor = self.decode(rows, &ends)?;
                reduced.push((
                    term.coefficient.mul(&integer(coefficient)),
                    tensor,
                    term.nc_power + empty + power,
                    term.root2_power,
                ));
            }
        }
        let parities: Vec<i64> = reduced.iter().map(|t| t.3.rem_euclid(2)).collect();
        if parities.windows(2).any(|w| w[0] != w[1]) {
            return Err(error(
                "colour network mixes rational and sqrt(2) coefficients",
            ));
        }
        let target = parities.first().copied().unwrap_or(0);
        // ConnectedColorTensor: drop single-letter traces, fold empty traces
        // into N_c, combine equal (tensor, power), sort, drop zeros.
        let mut index: FastMap<(ColourTensor, i64), usize> = FastMap::default();
        let mut combined: Vec<((ColourTensor, i64), Exact)> = Vec::new();
        for (coefficient, tensor, power, root2) in reduced {
            if tensor.traces.iter().any(|t| t.len() == 1) {
                continue;
            }
            let value = coefficient.mul(&power_of_two((target - root2) / 2));
            let empty = tensor.traces.iter().filter(|t| t.is_empty()).count() as i64;
            let tensor = ColourTensor {
                strings: tensor.strings,
                traces: tensor
                    .traces
                    .into_iter()
                    .filter(|t| !t.is_empty())
                    .collect(),
            };
            let key = (tensor, power + empty);
            match index.get(&key) {
                Some(&i) => combined[i].1 = combined[i].1.add(&value),
                None => {
                    index.insert(key.clone(), combined.len());
                    combined.push((key, value));
                }
            }
        }
        combined.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(Tensor {
            normalization: target,
            terms: combined
                .into_iter()
                .filter(|(_, value)| !value.is_zero())
                .map(|((tensor, power), value)| (value, tensor, power))
                .collect(),
        })
    }

    /// `_join_strings`: open strings (external row to external column),
    /// traces of closed loops and the number of empty loops.
    fn join_strings(&self, strings: &[NetString]) -> Result<Joined, String> {
        let mut by_row: FastMap<Label, usize> = FastMap::default();
        for (i, string) in strings.iter().enumerate() {
            if by_row.insert(string.row, i).is_some() {
                return Err(error(format!(
                    "fundamental index {} occurs twice as row",
                    string.row
                )));
            }
        }
        let mut columns: Vec<Label> = strings.iter().map(|s| s.col).collect();
        columns.sort_unstable();
        if columns.windows(2).any(|w| w[0] == w[1]) {
            return Err(error("a fundamental index occurs twice as column"));
        }
        let mut used = vec![false; strings.len()];
        let mut opened = Vec::new();
        for (start, string) in strings.iter().enumerate() {
            if !self.is_external(string.row) {
                continue;
            }
            let mut word = Vec::new();
            let mut current = start;
            used[current] = true;
            while !self.is_external(strings[current].col) {
                word.extend_from_slice(&strings[current].word);
                let following = *by_row.get(&strings[current].col).ok_or_else(|| {
                    error(format!(
                        "fundamental index {} is not contracted",
                        strings[current].col
                    ))
                })?;
                used[following] = true;
                current = following;
            }
            word.extend_from_slice(&strings[current].word);
            opened.push(NetString {
                row: string.row,
                word,
                col: strings[current].col,
            });
        }
        let mut traces = Vec::new();
        let mut empty = 0;
        for row in 0..strings.len() {
            if used[row] {
                continue;
            }
            let mut word = Vec::new();
            let mut current = row;
            loop {
                if used[current] {
                    break;
                }
                used[current] = true;
                let string = &strings[current];
                if self.is_external(string.col) {
                    return Err(error(format!(
                        "fundamental index {} is not contracted",
                        string.row
                    )));
                }
                word.extend_from_slice(&string.word);
                current = *by_row.get(&string.col).ok_or_else(|| {
                    error(format!(
                        "fundamental index {} is not contracted",
                        string.col
                    ))
                })?;
            }
            if current != row {
                return Err(error("fundamental indices do not close into a loop"));
            }
            if word.is_empty() {
                empty += 1;
            } else {
                traces.push(word);
            }
        }
        Ok((opened, traces, empty))
    }

    /// `_decode`: Fierz rows to literal structures.
    fn decode(
        &self,
        rows: &[Vec<Label>],
        ends: &FastMap<Label, (Label, Label)>,
    ) -> Result<ColourTensor, String> {
        let mut strings = Vec::new();
        let mut traces = Vec::new();
        for row in rows {
            if row
                .iter()
                .any(|l| !self.is_external(*l) && !ends.contains_key(l))
            {
                return Err(error(
                    "an internal adjoint index survived the Fierz reduction",
                ));
            }
            let Some(first) = row.iter().position(|l| ends.contains_key(l)) else {
                traces.push(row.iter().rev().copied().collect::<Vec<_>>());
                continue;
            };
            let rotated: Vec<Label> = row[first..].iter().chain(&row[..first]).copied().collect();
            let marks: Vec<usize> = rotated
                .iter()
                .enumerate()
                .filter(|(_, l)| ends.contains_key(l))
                .map(|(i, _)| i)
                .collect();
            for (index, &start) in marks.iter().enumerate() {
                let stop = marks.get(index + 1).copied().unwrap_or(rotated.len());
                let (upper, _) = ends[&rotated[start]];
                let (_, lower) = ends[&rotated[stop % rotated.len()]];
                strings.push(OpenString {
                    fundamental: lower,
                    adjoint: rotated[start + 1..stop].iter().rev().copied().collect(),
                    antifundamental: upper,
                });
            }
        }
        Ok(ColourTensor::new(strings, traces))
    }
}

/// `_Numbering`: external labels keep theirs; internal ones are numbered
/// from `first` on first use.
struct Numbering {
    next: Label,
    map: FastMap<Label, Label>,
}

impl Numbering {
    fn new(first: Label) -> Self {
        Self {
            next: first,
            map: FastMap::default(),
        }
    }

    fn fresh(&mut self) -> Label {
        let value = self.next;
        self.next += 1;
        value
    }

    fn number(&mut self, label: Label, reducer: &Reducer) -> Label {
        if reducer.is_external(label) {
            return label;
        }
        if let Some(&found) = self.map.get(&label) {
            return found;
        }
        let value = self.fresh();
        self.map.insert(label, value);
        value
    }
}

fn integer(value: &Integer) -> Exact {
    Exact {
        re: Rational::from(value),
        im: Rational::from(0),
    }
}

/// `Fraction(2) ** exponent`, exact for every exponent (a bignum power).
fn power_of_two(exponent: i64) -> Exact {
    let two = Rational::from(2);
    let mut magnitude = Rational::from(1);
    for _ in 0..exponent.unsigned_abs() {
        magnitude *= &two;
    }
    Exact {
        re: if exponent >= 0 {
            magnitude
        } else {
            Rational::from(1) / &magnitude
        },
        im: Rational::from(0),
    }
}

/// `_fierz`: remove every repeated adjoint label (sentinels close open
/// strings; a trace of one sentinel is a Kronecker delta).
fn fierz(traces: &[Vec<Label>], sentinels: &[Label]) -> Vec<FierzTerm> {
    let is_sentinel = |l: Label| sentinels.binary_search(&l).is_ok();
    let trivial = |terms: Vec<FierzTerm>| -> Vec<FierzTerm> {
        let mut out = Vec::with_capacity(terms.len());
        for (coefficient, power, rows) in terms {
            if rows.iter().any(|r| r.len() == 1 && !is_sentinel(r[0])) {
                continue; // tr(tau^a) = 0
            }
            let empty = rows.iter().filter(|r| r.is_empty()).count() as i64;
            out.push((
                coefficient,
                power + empty,
                rows.into_iter().filter(|r| !r.is_empty()).collect(),
            ));
        }
        combine(out)
    };
    let mut terms = trivial(vec![(Integer::one(), 0, traces.to_vec())]);
    loop {
        let (next, paired) = pair_simplify(terms);
        terms = trivial(next);
        let (next, duplicated) = duplicate_simplify(terms);
        terms = trivial(next);
        if !paired && !duplicated {
            return terms;
        }
    }
}

/// `combine_nc_power_terms`: equal (power, sorted traces) summed, first-seen
/// order, zeros dropped.
fn combine(terms: Vec<FierzTerm>) -> Vec<FierzTerm> {
    let mut index: FastMap<(i64, Vec<Vec<Label>>), usize> = FastMap::default();
    let mut out: Vec<FierzTerm> = Vec::new();
    for (coefficient, power, mut rows) in terms {
        if coefficient.is_zero() {
            continue;
        }
        rows.sort();
        let key = (power, rows);
        match index.get(&key) {
            Some(&i) => out[i].0 += &coefficient,
            None => {
                index.insert(key.clone(), out.len());
                out.push((coefficient, key.0, key.1));
            }
        }
    }
    out.retain(|t| !t.0.is_zero());
    out
}

/// `trace_pair_simplify_nc_power`: per term, the first pair of traces with a
/// common label is fused.
fn pair_simplify(terms: Vec<FierzTerm>) -> (Vec<FierzTerm>, bool) {
    let mut out = Vec::with_capacity(terms.len() * 2);
    let mut changed = false;
    for (coefficient, power, rows) in terms {
        let mut found = None;
        'search: for i in 0..rows.len() {
            for j in i + 1..rows.len() {
                for (a, label) in rows[i].iter().enumerate() {
                    if let Some(b) = rows[j].iter().position(|other| other == label) {
                        found = Some((i, j, a, b));
                        break 'search;
                    }
                }
            }
        }
        let Some((i, j, a, b)) = found else {
            out.push((coefficient, power, rows));
            continue;
        };
        let (first, second) = (&rows[i], &rows[j]);
        let combined: Vec<Label> = first[..a]
            .iter()
            .chain(&second[b + 1..])
            .chain(&second[..b])
            .chain(&first[a + 1..])
            .copied()
            .collect();
        let mut fused = vec![combined];
        fused.extend(
            rows.iter()
                .enumerate()
                .filter(|(k, _)| *k != i && *k != j)
                .map(|(_, r)| r.clone()),
        );
        let negated = -&coefficient;
        out.push((coefficient, power, fused));
        let reduced: Vec<Vec<Label>> = rows
            .iter()
            .enumerate()
            .map(|(k, r)| {
                if k == i {
                    r[..a].iter().chain(&r[a + 1..]).copied().collect()
                } else if k == j {
                    r[..b].iter().chain(&r[b + 1..]).copied().collect()
                } else {
                    r.clone()
                }
            })
            .collect();
        out.push((negated, power - 1, reduced));
        changed = true;
    }
    (out, changed)
}

/// `trace_duplicate_simplify_nc_power`: per term, the first trace with a
/// repeated label is split.
fn duplicate_simplify(terms: Vec<FierzTerm>) -> (Vec<FierzTerm>, bool) {
    let mut out = Vec::with_capacity(terms.len() * 2);
    let mut changed = false;
    for (coefficient, power, rows) in terms {
        let mut found = None;
        for (t, row) in rows.iter().enumerate() {
            let mut seen: FastMap<Label, usize> = FastMap::default();
            for (index, &label) in row.iter().enumerate() {
                if let Some(&previous) = seen.get(&label) {
                    found = Some((t, previous, index));
                    break;
                }
                seen.insert(label, index);
            }
            if found.is_some() {
                break;
            }
        }
        let Some((t, first, second)) = found else {
            out.push((coefficient, power, rows));
            continue;
        };
        let row = &rows[t];
        let (a, b, c) = (&row[..first], &row[first + 1..second], &row[second + 1..]);
        let rest = rows
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != t)
            .map(|(_, r)| r.clone());
        let mut split = vec![a.iter().chain(c).copied().collect::<Vec<_>>(), b.to_vec()];
        split.extend(rest.clone());
        let negated = -&coefficient;
        out.push((coefficient, power, split));
        let mut joined = vec![a.iter().chain(b).chain(c).copied().collect::<Vec<_>>()];
        joined.extend(rest);
        out.push((negated, power - 1, joined));
        changed = true;
    }
    (out, changed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(labels: &[Label]) -> Product {
        Product {
            traces: vec![labels.to_vec()],
            ..Product::one()
        }
    }

    #[test]
    fn fierz_of_two_traces_sharing_a_label() {
        // tr(1 a 2) tr(a 3 4) = tr(1 3 4 2) - 1/N tr(1 2) tr(3 4).
        let mut reducer = Reducer::new(&[1, 2, 3, 4]);
        let network = vec![trace(&[1, -5, 2]).times(&trace(&[-5, 3, 4]))];
        let tensor = reducer.reduce(&network).unwrap();
        assert_eq!(tensor.normalization, 0);
        let terms: Vec<(String, Vec<Vec<i64>>, i64)> = tensor
            .terms
            .iter()
            .map(|(c, t, p)| (c.text().0, t.traces.clone(), *p))
            .collect();
        // Literal traces are the reversed network traces.
        assert_eq!(
            terms,
            vec![
                ("-1/1".to_string(), vec![vec![1, 2], vec![3, 4]], -1),
                ("1/1".to_string(), vec![vec![1, 2, 4, 3]], 0),
            ]
        );
    }

    #[test]
    fn an_adjoint_loop_closes_to_nc_squared_minus_one() {
        // tr(a a) = N^2 - 1 (tr(a) tr() - 1/N tr()) with tr() = N.
        let mut reducer = Reducer::new(&[1]);
        let tensor = reducer.reduce(&vec![trace(&[-5, -5])]).unwrap();
        let terms: Vec<(String, i64)> = tensor
            .terms
            .iter()
            .map(|(c, _, p)| (c.text().0, *p))
            .collect();
        assert_eq!(terms, vec![("-1/1".to_string(), 0), ("1/1".to_string(), 2)]);
    }

    #[test]
    fn fierz_coefficients_are_exact_beyond_machine_words() {
        // Sums and negations past i64 (|c| = 2^63 needs 63 contracted pairs
        // in a network, so the steps are exercised directly).
        let rows = vec![vec![1, 2]];
        let max = Integer::from(i64::MAX);
        let summed = combine(vec![
            (max.clone(), 0, rows.clone()),
            (max.clone(), 0, rows.clone()),
        ]);
        assert_eq!(summed.len(), 1);
        assert_eq!(summed[0].0, &max + &max);
        assert_eq!(summed[0].0.to_string(), "18446744073709551614");
        // -(i64::MIN) = 2^63, one past i64.
        let (split, changed) = pair_simplify(vec![(
            Integer::from(i64::MIN),
            0,
            vec![vec![1, -5, 2], vec![-5, 3]],
        )]);
        assert!(changed);
        assert_eq!(split[1].0.to_string(), "9223372036854775808");
        let (split, changed) =
            duplicate_simplify(vec![(Integer::from(i64::MIN), 0, vec![vec![1, -5, 2, -5]])]);
        assert!(changed);
        assert_eq!(split[1].0.to_string(), "9223372036854775808");
        assert!(combine(vec![(max.clone(), 0, rows.clone()), (-&max, 0, rows)]).is_empty());
    }

    #[test]
    fn powers_of_two_are_exact_beyond_machine_words() {
        let big = "1180591620717411303424/1"; // 2^70
        assert_eq!(power_of_two(70).text().0, big);
        assert_eq!(power_of_two(-70).text().0, "1/1180591620717411303424");
        assert_eq!(power_of_two(0).text().0, "1/1");
    }
}
