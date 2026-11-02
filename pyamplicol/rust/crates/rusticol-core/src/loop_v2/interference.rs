// SPDX-License-Identifier: 0BSD

//! Colour interferences of the native counterterm and external-field
//! components (WP3.1 I5, `loops/v2/native_trees._counterterm_program`).
//!
//! An output (component, sector, epsilon power) sums, over the Born helicity
//! rows, entries `2 Re(c <A0(order)|M|ket>)`: the bra is the Born amplitude
//! of one Born order in its components `A0[b, order, p]` (basis element `b`,
//! N_c power `p`), the ket a counterterm vector in its components
//! `K[t, vector, q]` (ket tensor `t`) or the Born amplitude of another Born
//! order (an external-field insertion, coupling `c`, N_c grade `g`). `M` is
//! the exact overlap `<b|t>` per N_c power, each cell of power `n` kept at
//! exponent `e = n + p + q + g` as the request's projection keeps it below
//! the sector's degree `P` (full colour: always; LC: `e = P`; NLC:
//! `P - 2 <= e <= P`), and weighted `3^e` (`assemble._interfere`).
//!
//! Per row, `z[t] = sum_b conj(A0[b, order, p]) M_(P, p+q+g)[b, t]` is formed
//! once per (bra order, `p`, metric class) and dotted with each ket.

use super::dress::{Carrier, cast};
use super::kernels::dispatched;
use crate::{RusticolError, RusticolResult};
use rayon::prelude::*;
use std::collections::BTreeMap;
use symbolica::domains::float::{Complex, Real};
use symbolica::domains::rational::Rational;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::compatibility(format!("loop v2 interference: {}", message.into()))
}

/// Components of one amplitude grouped by N_c power: (power, (basis or ket
/// tensor, component)).
type Groups = Vec<(i32, Vec<(usize, usize)>)>;

/// Where an entry's ket comes from.
#[derive(Clone, Copy, Debug)]
pub enum Ket {
    /// A counterterm vector.
    Vector(usize),
    /// The Born amplitude of a Born order (its components are the bra's).
    Born(usize),
}

/// One interference entry: `2 Re(c <A0(bra)|M|ket>)` into `output`.
#[derive(Clone, Debug)]
pub struct Entry {
    pub bra: usize,
    pub ket: Ket,
    pub coupling: Option<usize>,
    pub grade: i32,
    pub output: usize,
    pub degree: Option<i32>,
}

/// The exact overlap cells `<b|t>` of one colour metric.
#[derive(Clone, Debug)]
pub enum MetricCells {
    /// (basis element, ket tensor, N_c power, exact cell) per nonzero cell
    /// (the counterterm interference's own Born metric).
    List(Vec<(usize, usize, i32, [Rational; 2])>),
    /// The distinct cells, each a list of (N_c power, exact cell), and per
    /// ket tensor and basis element the index of its cell (an observable
    /// column's metric, `loops/v2/native_observables`: a few dozen distinct
    /// cells over every (b, t) pair).
    Table {
        values: Vec<Vec<(i32, [Rational; 2])>>,
        ids: Vec<Vec<u32>>,
    },
}

/// The interference program of a process.
pub struct Interference {
    basis: usize,
    /// Ket tensors (validated at construction; the per-row oracle of the
    /// tests sizes its `z` by it).
    #[cfg_attr(not(test), allow(dead_code))]
    tensors: usize,
    cells: MetricCells,
    /// Per Born order, its Born components by N_c power.
    bra: Vec<Groups>,
    /// Per counterterm vector, its ket components by N_c power.
    vectors: Vec<Groups>,
    entries: Vec<Entry>,
    /// Per entry, per (bra group, ket group) the metric class.
    classes: Vec<Vec<Vec<usize>>>,
    /// (degree, exponent shift) per metric class.
    class_keys: Vec<(Option<i32>, i32)>,
    depth: Option<i32>,
    /// Epsilon power per output.
    outputs: Vec<i32>,
}

/// Per metric class and basis element: its nonzero (ket tensor, value)
/// cells.
type MetricClasses<T> = Vec<Vec<Vec<(usize, Complex<T>)>>>;

/// The metric classes of an interference program in a carrier: per class
/// and basis element, its nonzero (ket tensor, value) cells.
pub struct InterferenceRuntime<T> {
    metrics: MetricClasses<T>,
}

fn kept(exponent: i32, degree: Option<i32>, depth: Option<i32>) -> bool {
    match depth {
        None => true,
        Some(depth) => {
            degree.is_some_and(|degree| degree - depth <= exponent && exponent <= degree)
        }
    }
}

fn nc_power(exponent: i32) -> Rational {
    let three = Rational::from(3);
    let mut value = Rational::from(1);
    for _ in 0..exponent.unsigned_abs() {
        value = if exponent > 0 {
            &value * &three
        } else {
            &value / &three
        };
    }
    value
}

/// Rows per block of [`Interference::rows`]: the rows of a block share
/// each streamed metric row (one pass over a metric class per block and
/// point instead of per row).
const ROW_BLOCK: usize = 16;

/// One point's amplitudes as [`Interference::evaluate`] reads them.
struct Point<'a, T> {
    born: &'a [Complex<T>],
    born_ket: Option<&'a [Complex<T>]>,
    born_width: usize,
    ket: &'a [Complex<T>],
    ket_width: usize,
    couplings: &'a [Complex<T>],
}

/// `z[t][lane] += x[j][lane] M[b, t]` over the bra components `(b, j)` in
/// order and each metric row's cells `t < limit` in order (metric rows
/// ascend in `t`, [`Interference::runtime`]), per lane in the operations of
/// `Complex<T>`'s `+=` and `*`; `x` holds `conj(A0)`, [`ROW_BLOCK`] lanes per
/// Born component (real parts and imaginary parts apart).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn z_sums_body<T: Carrier>(
    bras: &[(usize, usize)],
    metric: &[Vec<(usize, Complex<T>)>],
    keep_bra: Option<&[bool]>,
    x_re: &[T],
    x_im: &[T],
    limit: usize,
    z_re: &mut [T],
    z_im: &mut [T],
) {
    const R: usize = ROW_BLOCK;
    for &(b, j) in bras {
        if keep_bra.is_some_and(|keep| !keep[b]) {
            continue;
        }
        let xr: &[T; R] = x_re[j * R..][..R].try_into().expect("a lane block");
        let xi: &[T; R] = x_im[j * R..][..R].try_into().expect("a lane block");
        for (t, m) in &metric[b] {
            if *t >= limit {
                break;
            }
            let zr: &mut [T; R] = (&mut z_re[t * R..][..R]).try_into().expect("a lane block");
            let zi: &mut [T; R] = (&mut z_im[t * R..][..R]).try_into().expect("a lane block");
            for lane in 0..R {
                zr[lane] += xr[lane].clone() * &m.re - xi[lane].clone() * &m.im;
                zi[lane] += xr[lane].clone() * &m.im + xi[lane].clone() * &m.re;
            }
        }
    }
}

dispatched! {
    /// [`z_sums_body`] in the widest vectors the running CPU reports for
    /// binary64 ([`super::kernels::Simd`]; the same operations per lane, so
    /// the same bits).
    #[allow(clippy::too_many_arguments)]
    fn z_sums<T: Carrier>(
        bras: &[(usize, usize)],
        metric: &[Vec<(usize, Complex<T>)>],
        keep_bra: Option<&[bool]>,
        x_re: &[T],
        x_im: &[T],
        limit: usize,
        z_re: &mut [T],
        z_im: &mut [T],
    ) => z_sums_body;
}

fn groups(
    components: impl Iterator<Item = (usize, usize, i32, usize)>,
    count: usize,
) -> Vec<Groups> {
    // (owner, tensor, power, component)
    let mut out: Vec<BTreeMap<i32, Vec<(usize, usize)>>> = vec![BTreeMap::new(); count];
    for (owner, tensor, power, component) in components {
        out[owner]
            .entry(power)
            .or_default()
            .push((tensor, component));
    }
    out.into_iter()
        .map(|groups| groups.into_iter().collect())
        .collect()
}

impl Interference {
    /// Validate and index an interference program. `born` is (basis
    /// element, Born order, N_c power) per Born component, `ket` (ket
    /// tensor, vector, N_c power) per ket component; `entries` are
    /// (kind, bra order, vector or Born order, coupling, grade, output,
    /// degree).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        basis: usize,
        orders: usize,
        tensors: usize,
        cells: MetricCells,
        born: &[(usize, usize, i32)],
        ket: &[(usize, usize, i32)],
        vectors: usize,
        entries: Vec<Entry>,
        outputs: Vec<i32>,
        couplings: usize,
        depth: Option<i32>,
    ) -> RusticolResult<Self> {
        if tensors < basis {
            return Err(invalid("the ket tensors start with the Born basis"));
        }
        match &cells {
            MetricCells::List(cells) => {
                if cells.iter().any(|&(b, t, _, _)| b >= basis || t >= tensors) {
                    return Err(invalid("overlap cell outside the basis or the ket tensors"));
                }
            }
            MetricCells::Table { values, ids } => {
                if ids.len() != tensors
                    || ids.iter().any(|row| {
                        row.len() != basis || row.iter().any(|&id| id as usize >= values.len())
                    })
                {
                    return Err(invalid(
                        "an overlap table needs one cell index per ket tensor and basis element",
                    ));
                }
            }
        }
        if born.iter().any(|&(b, o, _)| b >= basis || o >= orders) {
            return Err(invalid("Born component"));
        }
        if ket.iter().any(|&(t, v, _)| t >= tensors || v >= vectors) {
            return Err(invalid("ket component"));
        }
        let bra = groups(
            born.iter().enumerate().map(|(j, &(b, o, p))| (o, b, p, j)),
            orders,
        );
        let vector_groups = groups(
            ket.iter().enumerate().map(|(k, &(t, v, q))| (v, t, q, k)),
            vectors,
        );
        let mut class_index: BTreeMap<(Option<i32>, i32), usize> = BTreeMap::new();
        let mut classes = Vec::with_capacity(entries.len());
        for entry in &entries {
            let valid = entry.bra < orders
                && entry.output < outputs.len()
                && entry.coupling.is_none_or(|c| c < couplings)
                && match entry.ket {
                    Ket::Vector(v) => v < vectors && entry.coupling.is_none(),
                    Ket::Born(o) => o < orders,
                };
            if !valid {
                return Err(invalid("interference entry"));
            }
            let kets = match entry.ket {
                Ket::Vector(v) => &vector_groups[v],
                Ket::Born(o) => &bra[o],
            };
            let per_bra = bra[entry.bra]
                .iter()
                .map(|(p, _)| {
                    kets.iter()
                        .map(|(q, _)| {
                            let key = (entry.degree, p + q + entry.grade);
                            let next = class_index.len();
                            *class_index.entry(key).or_insert(next)
                        })
                        .collect()
                })
                .collect();
            classes.push(per_bra);
        }
        let mut class_keys = vec![(None, 0); class_index.len()];
        for (key, index) in class_index {
            class_keys[index] = key;
        }
        Ok(Self {
            basis,
            tensors,
            cells,
            bra,
            vectors: vector_groups,
            entries,
            classes,
            class_keys,
            depth,
            outputs,
        })
    }

    /// Number of outputs.
    pub fn outputs(&self) -> &[i32] {
        &self.outputs
    }

    /// The metric classes in carrier `T`: per class, the exact sum of each
    /// (b, t) cell's kept powers times `3^e`, cast once (a table's distinct
    /// cells are summed once per class, then indexed).
    pub fn runtime<T: Carrier>(&self, zero: &T) -> InterferenceRuntime<T> {
        let metrics = self
            .class_keys
            .iter()
            .map(|&(degree, shift)| {
                let sum = |cell: &[(i32, [Rational; 2])]| -> [Rational; 2] {
                    let mut out = [Rational::from(0), Rational::from(0)];
                    for (power, [re, im]) in cell {
                        let exponent = power + shift;
                        if !kept(exponent, degree, self.depth) {
                            continue;
                        }
                        let scale = nc_power(exponent);
                        out[0] = &out[0] + &(re * &scale);
                        out[1] = &out[1] + &(im * &scale);
                    }
                    out
                };
                let complex = |[re, im]: &[Rational; 2]| -> Option<Complex<T>> {
                    (!re.is_zero() || !im.is_zero())
                        .then(|| Complex::new(cast(zero, re), cast(zero, im)))
                };
                match &self.cells {
                    MetricCells::List(cells) => {
                        let mut rows: Vec<BTreeMap<usize, [Rational; 2]>> =
                            vec![BTreeMap::new(); self.basis];
                        for (b, t, power, [re, im]) in cells {
                            let exponent = power + shift;
                            if !kept(exponent, degree, self.depth) {
                                continue;
                            }
                            let scale = nc_power(exponent);
                            let cell = rows[*b]
                                .entry(*t)
                                .or_insert_with(|| [Rational::from(0), Rational::from(0)]);
                            cell[0] = &cell[0] + &(re * &scale);
                            cell[1] = &cell[1] + &(im * &scale);
                        }
                        rows.into_iter()
                            .map(|row| {
                                row.into_iter()
                                    .filter_map(|(t, cell)| complex(&cell).map(|value| (t, value)))
                                    .collect()
                            })
                            .collect()
                    }
                    MetricCells::Table { values, ids } => {
                        let scaled: Vec<Option<Complex<T>>> =
                            values.iter().map(|cell| complex(&sum(cell))).collect();
                        (0..self.basis)
                            .map(|b| {
                                ids.iter()
                                    .enumerate()
                                    .filter_map(|(t, row)| {
                                        scaled[row[b] as usize].clone().map(|value| (t, value))
                                    })
                                    .collect()
                            })
                            .collect()
                    }
                }
            })
            .collect();
        InterferenceRuntime { metrics }
    }

    /// The complex sums `sum_rows c <A0(bra)|M|ket>` of one point per output
    /// (an observable column's interferences, `loops/v2/native_observables`):
    /// `born` holds `rows x born components` (the bra; `born_ket`, when
    /// given, the external-field insertions' Born ket, else `born`), `ket`
    /// `rows x ket components`, `couplings` the point's coupling slots.
    /// `keep_rows` restricts the rows (helicity-resolved columns, a spin
    /// source's fixed row) and `keep_bra` the bra's basis elements
    /// (leading-colour resolved columns; the ket is never masked). Without `poles`, outputs of a
    /// nonzero epsilon power stay zero. Not divided by the average.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_complex<T: Carrier>(
        &self,
        runtime: &InterferenceRuntime<T>,
        born: &[Complex<T>],
        born_ket: Option<&[Complex<T>]>,
        born_width: usize,
        ket: &[Complex<T>],
        ket_width: usize,
        keep_rows: &[bool],
        keep_bra: Option<&[bool]>,
        couplings: &[Complex<T>],
        poles: bool,
        zero: &T,
    ) -> Vec<Complex<T>>
    where
        Complex<T>: Real,
    {
        let czero = Complex::new(zero.clone(), zero.clone());
        let rows: Vec<usize> = (0..keep_rows.len()).filter(|&row| keep_rows[row]).collect();
        let point = Point {
            born,
            born_ket,
            born_width,
            ket,
            ket_width,
            couplings,
        };
        let per_row = self.rows(
            runtime,
            &point,
            &rows,
            keep_bra,
            poles,
            zero,
            |out, entry, value| {
                out[entry.output] += value;
            },
            &czero,
        );
        let mut out = vec![czero; self.outputs.len()];
        for row in per_row {
            for (total, value) in out.iter_mut().zip(row) {
                *total += value;
            }
        }
        out
    }

    /// Number of basis elements of the bra.
    pub fn basis(&self) -> usize {
        self.basis
    }

    /// The overlap cells of the program's metric.
    pub fn cells(&self) -> &MetricCells {
        &self.cells
    }

    /// The outputs of one point: `born` holds `rows x born components` (the
    /// bra; `born_ket`, when given, the external-field insertions' Born ket,
    /// else `born`: the loop-side Born of `trees::TreeProgram::
    /// inputs_loop_side`), `ket` `rows x ket components`, `couplings` the
    /// point's coupling slots. Without `poles`, outputs of a nonzero epsilon power stay zero.
    /// Not divided by the average.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<T: Carrier>(
        &self,
        runtime: &InterferenceRuntime<T>,
        born: &[Complex<T>],
        born_ket: Option<&[Complex<T>]>,
        born_width: usize,
        ket: &[Complex<T>],
        ket_width: usize,
        rows: usize,
        couplings: &[Complex<T>],
        poles: bool,
        zero: &T,
    ) -> Vec<T>
    where
        Complex<T>: Real,
    {
        let rows: Vec<usize> = (0..rows).collect();
        let point = Point {
            born,
            born_ket,
            born_width,
            ket,
            ket_width,
            couplings,
        };
        let per_row = self.rows(
            runtime,
            &point,
            &rows,
            None,
            poles,
            zero,
            |out, entry, value| {
                out[entry.output] += value.re.clone() + value.re;
            },
            zero,
        );
        let mut out = vec![zero.clone(); self.outputs.len()];
        for row in per_row {
            for (total, value) in out.iter_mut().zip(row) {
                *total += value;
            }
        }
        out
    }

    /// The z sums each kept entry reads: per key `(bra order, bra group,
    /// metric class)` the number of leading ket tensors its kets read (all
    /// a counterterm vector touches; the Born basis for a Born ket), and per
    /// entry, bra group and ket group the key's index.
    fn keys(&self, poles: bool) -> (Vec<((usize, usize, usize), usize)>, Vec<Vec<Vec<usize>>>) {
        let mut index: BTreeMap<(usize, usize, usize), usize> = BTreeMap::new();
        let mut limits: Vec<usize> = Vec::new();
        let mut per_entry = Vec::with_capacity(self.entries.len());
        for (entry, classes) in self.entries.iter().zip(&self.classes) {
            if !poles && self.outputs[entry.output] != 0 {
                per_entry.push(Vec::new());
                continue;
            }
            let kets = match entry.ket {
                Ket::Vector(v) => &self.vectors[v],
                Ket::Born(o) => &self.bra[o],
            };
            let groups = classes
                .iter()
                .enumerate()
                .map(|(g, classes)| {
                    kets.iter()
                        .zip(classes)
                        .map(|((_q, components), &class)| {
                            let next = index.len();
                            let key = *index.entry((entry.bra, g, class)).or_insert(next);
                            if key == limits.len() {
                                limits.push(0);
                            }
                            let read = components.iter().map(|&(t, _)| t + 1).max().unwrap_or(0);
                            limits[key] = limits[key].max(read);
                            key
                        })
                        .collect()
                })
                .collect();
            per_entry.push(groups);
        }
        let mut keys = vec![((0, 0, 0), 0); index.len()];
        for (key, at) in index {
            keys[at] = (key, limits[at]);
        }
        (keys, per_entry)
    }

    /// Per kept row (in `rows` order) the outputs `combine` accumulates
    /// from each kept entry's `c <A0(bra)|M|ket>`, row blocks of
    /// [`ROW_BLOCK`] rows on the rayon pool. Each row's operations are those
    /// of the row-by-row evaluation (`z[t] = sum_b conj(A0[b]) M[b, t]` in
    /// bra-component and metric-cell order, then the kets in entry order),
    /// so the values are bitwise the same; a block streams each metric row
    /// once for all its rows.
    #[allow(clippy::too_many_arguments)]
    fn rows<T: Carrier, O: Clone + Send>(
        &self,
        runtime: &InterferenceRuntime<T>,
        point: &Point<'_, T>,
        rows: &[usize],
        keep_bra: Option<&[bool]>,
        poles: bool,
        zero: &T,
        combine: impl Fn(&mut [O], &Entry, Complex<T>) + Sync,
        out_zero: &O,
    ) -> Vec<Vec<O>>
    where
        Complex<T>: Real,
        O: Sync,
    {
        let (keys, per_entry) = self.keys(poles);
        let czero = Complex::new(zero.clone(), zero.clone());
        let blocks: Vec<&[usize]> = rows.chunks(ROW_BLOCK).collect();
        blocks
            .par_iter()
            .map(|block| {
                let lanes = ROW_BLOCK;
                // conj(A0) per Born component, one lane per row of the block.
                let mut x_re = vec![zero.clone(); point.born_width * lanes];
                let mut x_im = vec![zero.clone(); point.born_width * lanes];
                for (lane, &row) in block.iter().enumerate() {
                    let a = &point.born[row * point.born_width..][..point.born_width];
                    for (j, value) in a.iter().enumerate() {
                        let x = value.conj();
                        x_re[j * lanes + lane] = x.re;
                        x_im[j * lanes + lane] = x.im;
                    }
                }
                // The z sums and then the block's rows also run on the pool
                // (a few row blocks alone leave threads idle).
                let zs: Vec<(Vec<T>, Vec<T>)> = keys
                    .par_iter()
                    .map(|&((bra, g, class), limit)| {
                        let mut z_re = vec![zero.clone(); limit * lanes];
                        let mut z_im = vec![zero.clone(); limit * lanes];
                        z_sums::<T>(
                            &self.bra[bra][g].1,
                            &runtime.metrics[class],
                            keep_bra,
                            &x_re,
                            &x_im,
                            limit,
                            &mut z_re,
                            &mut z_im,
                        );
                        (z_re, z_im)
                    })
                    .collect();
                block
                    .par_iter()
                    .enumerate()
                    .map(|(lane, &row)| {
                        let a = &point.born[row * point.born_width..][..point.born_width];
                        let b = point.born_ket.map_or(a, |born| {
                            &born[row * point.born_width..][..point.born_width]
                        });
                        let k = &point.ket[row * point.ket_width..][..point.ket_width];
                        let mut out = vec![out_zero.clone(); self.outputs.len()];
                        for (entry, groups) in self.entries.iter().zip(&per_entry) {
                            if !poles && self.outputs[entry.output] != 0 {
                                continue;
                            }
                            let (kets, values) = match entry.ket {
                                Ket::Vector(v) => (&self.vectors[v], k),
                                Ket::Born(o) => (&self.bra[o], b),
                            };
                            let mut value = czero.clone();
                            for keys in groups {
                                for ((_q, components), &key) in kets.iter().zip(keys) {
                                    let (z_re, z_im) = &zs[key];
                                    for &(t, c) in components {
                                        let z = Complex::new(
                                            z_re[t * lanes + lane].clone(),
                                            z_im[t * lanes + lane].clone(),
                                        );
                                        value += z * values[c].clone();
                                    }
                                }
                            }
                            if let Some(c) = entry.coupling {
                                value = point.couplings[c].clone() * value;
                            }
                            combine(&mut out, entry, value);
                        }
                        out
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(n: i64) -> Rational {
        Rational::from(n)
    }

    fn program(cells: MetricCells, depth: Option<i32>) -> Interference {
        // Two basis elements, three ket tensors; the Born components carry
        // N_c powers 0 and 1, the ket's 0, 0 and -1.
        Interference::new(
            2,
            1,
            3,
            cells,
            &[(0, 0, 0), (1, 0, 1)],
            &[(0, 0, 0), (1, 0, 0), (2, 0, -1)],
            1,
            vec![Entry {
                bra: 0,
                ket: Ket::Vector(0),
                coupling: None,
                grade: 0,
                output: 0,
                degree: depth.map(|_| 1),
            }],
            vec![0],
            0,
            depth,
        )
        .unwrap()
    }

    #[test]
    fn a_table_metric_equals_its_list() {
        let list = vec![
            (0, 0, 0, [r(1), r(0)]),
            (0, 0, -2, [r(-1), r(0)]),
            (0, 2, 1, [r(2), r(1)]),
            (1, 1, 0, [r(3), r(0)]),
            (1, 2, 1, [r(2), r(1)]),
        ];
        // ids[t][b]: (0,0) -> cell 0, (1,0) -> the empty cell 1, (1,1) -> 3,
        // (2,0) and (2,1) -> 2.
        let table = MetricCells::Table {
            values: vec![
                vec![(0, [r(1), r(0)]), (-2, [r(-1), r(0)])],
                vec![],
                vec![(1, [r(2), r(1)])],
                vec![(0, [r(3), r(0)])],
            ],
            ids: vec![vec![0, 1], vec![1, 3], vec![2, 2]],
        };
        for depth in [None, Some(0), Some(2)] {
            let a = program(MetricCells::List(list.clone()), depth).runtime(&0.0_f64);
            let b = program(table.clone(), depth).runtime(&0.0_f64);
            assert_eq!(a.metrics.len(), b.metrics.len());
            for (x, y) in a.metrics.iter().zip(&b.metrics) {
                assert_eq!(x.len(), y.len());
                for (row_x, row_y) in x.iter().zip(y) {
                    let bits = |row: &Vec<(usize, Complex<f64>)>| -> Vec<(usize, u64, u64)> {
                        row.iter()
                            .map(|(t, z)| (*t, z.re.to_bits(), z.im.to_bits()))
                            .collect()
                    };
                    assert_eq!(bits(row_x), bits(row_y), "depth {depth:?}");
                }
            }
        }
    }

    #[test]
    fn a_field_entry_reads_the_loop_side_born_ket() {
        // A Born-ket (external-field) entry over the Hermitian Born metric
        // M = [[1, 1 + i], [1 - i, 2]]: the bra is the physical Born, the
        // ket the loop side's when given (`trees::TreeProgram::
        // inputs_loop_side`), else the bra itself.
        let cells = MetricCells::List(vec![
            (0, 0, 0, [r(1), r(0)]),
            (0, 1, 0, [r(1), r(1)]),
            (1, 0, 0, [r(1), r(-1)]),
            (1, 1, 0, [r(2), r(0)]),
        ]);
        let program = Interference::new(
            2,
            1,
            2,
            cells,
            &[(0, 0, 0), (1, 0, 0)],
            &[],
            0,
            vec![Entry {
                bra: 0,
                ket: Ket::Born(0),
                coupling: None,
                grade: 0,
                output: 0,
                degree: None,
            }],
            vec![0],
            0,
            None,
        )
        .unwrap();
        let runtime = program.runtime(&0.0_f64);
        let c = Complex::new;
        let bra = [c(1.0, 2.0), c(-1.0, 1.0)];
        let ket = [c(3.0, -1.0), c(2.0, 2.0)];
        let real = |born_ket: Option<&[Complex<f64>]>| {
            program.evaluate(&runtime, &bra, born_ket, 2, &[], 0, 1, &[], true, &0.0)
        };
        let complex = |born_ket: Option<&[Complex<f64>]>| {
            let keep = [true];
            program.evaluate_complex(
                &runtime,
                &bra,
                born_ket,
                2,
                &[],
                0,
                &keep,
                None,
                &[],
                true,
                &0.0,
            )
        };
        // <bra|M|ket> = 3 - 9i; <bra|M|bra> = 5.
        assert_eq!(real(Some(&ket)), vec![6.0]);
        assert_eq!(complex(Some(&ket)), vec![c(3.0, -9.0)]);
        assert_eq!(real(None), vec![10.0]);
        assert_eq!(real(Some(&bra)), real(None));
        assert_eq!(complex(None), vec![c(5.0, 0.0)]);
    }

    /// The row-by-row evaluation the row blocks replaced (oracle): per row
    /// the z sums of the keys it reaches, cached per row, then the entries.
    #[allow(clippy::too_many_arguments)]
    fn row_by_row(
        program: &Interference,
        runtime: &InterferenceRuntime<f64>,
        born: &[Complex<f64>],
        born_ket: Option<&[Complex<f64>]>,
        born_width: usize,
        ket: &[Complex<f64>],
        ket_width: usize,
        keep_rows: &[bool],
        keep_bra: Option<&[bool]>,
        couplings: &[Complex<f64>],
        poles: bool,
    ) -> Vec<Vec<Complex<f64>>> {
        let czero = Complex::new(0.0, 0.0);
        (0..keep_rows.len())
            .filter(|&row| keep_rows[row])
            .map(|row| {
                let a = &born[row * born_width..][..born_width];
                let b = born_ket.map_or(a, |born| &born[row * born_width..][..born_width]);
                let k = &ket[row * ket_width..][..ket_width];
                let mut out = vec![czero; program.outputs.len()];
                let mut cache: BTreeMap<(usize, usize, usize), Vec<Complex<f64>>> = BTreeMap::new();
                for (entry, classes) in program.entries.iter().zip(&program.classes) {
                    if !poles && program.outputs[entry.output] != 0 {
                        continue;
                    }
                    let (kets, values) = match entry.ket {
                        Ket::Vector(v) => (&program.vectors[v], k),
                        Ket::Born(o) => (&program.bra[o], b),
                    };
                    let mut value = czero;
                    for (g, ((_p, bras), classes)) in
                        program.bra[entry.bra].iter().zip(classes).enumerate()
                    {
                        for ((_q, components), &class) in kets.iter().zip(classes) {
                            let z = cache.entry((entry.bra, g, class)).or_insert_with(|| {
                                let mut z = vec![czero; program.tensors];
                                let metric = &runtime.metrics[class];
                                for &(b, j) in bras {
                                    if keep_bra.is_some_and(|keep| !keep[b]) {
                                        continue;
                                    }
                                    let x = a[j].conj();
                                    for (t, m) in &metric[b] {
                                        z[*t] += x * *m;
                                    }
                                }
                                z
                            });
                            for &(t, c) in components {
                                value += z[t] * values[c];
                            }
                        }
                    }
                    if let Some(c) = entry.coupling {
                        value = couplings[c] * value;
                    }
                    out[entry.output] += value;
                }
                out
            })
            .collect()
    }

    #[test]
    fn row_blocks_equal_the_row_by_row_evaluation_bitwise() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let (basis, tensors, orders, vectors) = (5, 9, 2, 2);
        let mut cells = Vec::new();
        for b in 0..basis {
            for t in 0..tensors {
                for power in -1..=1 {
                    if next() % 3 != 0 {
                        let re = (next() % 11) as i64 - 5;
                        let im = (next() % 7) as i64 - 3;
                        cells.push((b, t, power, [Rational::from((re, 3)), Rational::from(im)]));
                    }
                }
            }
        }
        let born: Vec<(usize, usize, i32)> = (0..2 * basis)
            .map(|j| (j % basis, j / basis, (j % 3) as i32 - 1))
            .collect();
        let ket: Vec<(usize, usize, i32)> = (0..14)
            .map(|k| ((k * 5) % tensors, k % vectors, (k % 2) as i32 - 1))
            .collect();
        let mut entries = Vec::new();
        for output in 0..3 {
            for bra in 0..orders {
                entries.push(Entry {
                    bra,
                    ket: Ket::Vector((bra + output) % vectors),
                    coupling: None,
                    grade: output as i32 - 1,
                    output,
                    degree: Some(1),
                });
                entries.push(Entry {
                    bra,
                    ket: Ket::Born((bra + 1) % orders),
                    coupling: Some(output % 2),
                    grade: 0,
                    output,
                    degree: Some(0),
                });
            }
        }
        let c = |re: u64, im: u64| Complex::new(re as f64 / 7.0 - 3.1, im as f64 / 13.0 - 1.7);
        let rows = 37;
        let width = born.len();
        let amplitudes: Vec<Complex<f64>> = (0..rows * width)
            .map(|_| c(next() % 50, next() % 50))
            .collect();
        let loop_side: Vec<Complex<f64>> = (0..rows * width)
            .map(|_| c(next() % 50, next() % 50))
            .collect();
        let kets: Vec<Complex<f64>> = (0..rows * ket.len())
            .map(|_| c(next() % 50, next() % 50))
            .collect();
        let couplings = [c(3, 5), c(40, 2)];
        let keep_rows: Vec<bool> = (0..rows).map(|row| row % 5 != 3).collect();
        let keep_bra: Vec<bool> = (0..basis).map(|b| b != 2).collect();
        let bits = |values: &[Complex<f64>]| -> Vec<(u64, u64)> {
            values
                .iter()
                .map(|z| (z.re.to_bits(), z.im.to_bits()))
                .collect()
        };
        for depth in [None, Some(1)] {
            let program = Interference::new(
                basis,
                orders,
                tensors,
                MetricCells::List(cells.clone()),
                &born,
                &ket,
                vectors,
                entries.clone(),
                vec![0, -1, 0],
                2,
                depth,
            )
            .unwrap();
            let runtime = program.runtime(&0.0_f64);
            for poles in [true, false] {
                for born_ket in [None, Some(loop_side.as_slice())] {
                    for keep in [None, Some(keep_bra.as_slice())] {
                        let expected = row_by_row(
                            &program,
                            &runtime,
                            &amplitudes,
                            born_ket,
                            width,
                            &kets,
                            ket.len(),
                            &keep_rows,
                            keep,
                            &couplings,
                            poles,
                        );
                        let mut total = vec![Complex::new(0.0, 0.0); 3];
                        for row in &expected {
                            for (t, v) in total.iter_mut().zip(row) {
                                *t += v;
                            }
                        }
                        let got = program.evaluate_complex(
                            &runtime,
                            &amplitudes,
                            born_ket,
                            width,
                            &kets,
                            ket.len(),
                            &keep_rows,
                            keep,
                            &couplings,
                            poles,
                            &0.0,
                        );
                        assert_eq!(bits(&got), bits(&total), "complex {depth:?} {poles}");
                    }
                    // The real form: every row, 2 Re per entry.
                    let got = program.evaluate(
                        &runtime,
                        &amplitudes,
                        born_ket,
                        width,
                        &kets,
                        ket.len(),
                        rows,
                        &couplings,
                        poles,
                        &0.0,
                    );
                    let mut reference = vec![0.0; 3];
                    for row in 0..rows {
                        let values = row_by_row_real(
                            &program,
                            &runtime,
                            &amplitudes,
                            born_ket,
                            width,
                            &kets,
                            ket.len(),
                            row,
                            &couplings,
                            poles,
                        );
                        for (t, v) in reference.iter_mut().zip(values) {
                            *t += v;
                        }
                    }
                    assert_eq!(
                        got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "real {depth:?} {poles}"
                    );
                }
            }
        }
    }

    /// One row of the row-by-row real evaluation (2 Re per entry).
    #[allow(clippy::too_many_arguments)]
    fn row_by_row_real(
        program: &Interference,
        runtime: &InterferenceRuntime<f64>,
        born: &[Complex<f64>],
        born_ket: Option<&[Complex<f64>]>,
        born_width: usize,
        ket: &[Complex<f64>],
        ket_width: usize,
        row: usize,
        couplings: &[Complex<f64>],
        poles: bool,
    ) -> Vec<f64> {
        let czero = Complex::new(0.0, 0.0);
        let a = &born[row * born_width..][..born_width];
        let b = born_ket.map_or(a, |born| &born[row * born_width..][..born_width]);
        let k = &ket[row * ket_width..][..ket_width];
        let mut out = vec![0.0; program.outputs.len()];
        let mut cache: BTreeMap<(usize, usize, usize), Vec<Complex<f64>>> = BTreeMap::new();
        for (entry, classes) in program.entries.iter().zip(&program.classes) {
            if !poles && program.outputs[entry.output] != 0 {
                continue;
            }
            let (kets, values) = match entry.ket {
                Ket::Vector(v) => (&program.vectors[v], k),
                Ket::Born(o) => (&program.bra[o], b),
            };
            let mut value = czero;
            for (g, ((_p, bras), classes)) in program.bra[entry.bra].iter().zip(classes).enumerate()
            {
                for ((_q, components), &class) in kets.iter().zip(classes) {
                    let z = cache.entry((entry.bra, g, class)).or_insert_with(|| {
                        let mut z = vec![czero; program.tensors];
                        for &(b, j) in bras {
                            let x = a[j].conj();
                            for (t, m) in &runtime.metrics[class][b] {
                                z[*t] += x * *m;
                            }
                        }
                        z
                    });
                    for &(t, c) in components {
                        value += z[t] * values[c];
                    }
                }
            }
            if let Some(c) = entry.coupling {
                value = couplings[c] * value;
            }
            out[entry.output] += value.re + value.re;
        }
        out
    }

    #[test]
    fn a_table_metric_is_validated() {
        let table = |ids: Vec<Vec<u32>>| MetricCells::Table {
            values: vec![vec![(0, [r(1), r(0)])]],
            ids,
        };
        let new = |cells| Interference::new(2, 1, 3, cells, &[], &[], 0, vec![], vec![], 0, None);
        assert!(new(table(vec![vec![0, 0]; 3])).is_ok());
        // One index per ket tensor and basis element, each a stored cell.
        assert!(new(table(vec![vec![0, 0]; 2])).is_err());
        assert!(new(table(vec![vec![0]; 3])).is_err());
        assert!(new(table(vec![vec![0, 1]; 3])).is_err());
    }
}
