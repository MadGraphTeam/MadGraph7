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
        let per_row: Vec<Vec<Complex<T>>> = keep_rows
            .par_iter()
            .enumerate()
            .filter(|(_, keep)| **keep)
            .map(|(row, _)| {
                let a = &born[row * born_width..][..born_width];
                let b = born_ket.map_or(a, |born| &born[row * born_width..][..born_width]);
                let k = &ket[row * ket_width..][..ket_width];
                let mut out = vec![czero.clone(); self.outputs.len()];
                let mut cache: BTreeMap<(usize, usize, usize), Vec<Complex<T>>> = BTreeMap::new();
                for (entry, classes) in self.entries.iter().zip(&self.classes) {
                    if !poles && self.outputs[entry.output] != 0 {
                        continue;
                    }
                    let (kets, values) = match entry.ket {
                        Ket::Vector(v) => (&self.vectors[v], k),
                        Ket::Born(o) => (&self.bra[o], b),
                    };
                    let mut value = czero.clone();
                    for (g, ((_p, bras), classes)) in
                        self.bra[entry.bra].iter().zip(classes).enumerate()
                    {
                        for ((_q, components), &class) in kets.iter().zip(classes) {
                            let z = cache.entry((entry.bra, g, class)).or_insert_with(|| {
                                let mut z = vec![czero.clone(); self.tensors];
                                let metric = &runtime.metrics[class];
                                for &(b, j) in bras {
                                    if keep_bra.is_some_and(|keep| !keep[b]) {
                                        continue;
                                    }
                                    let x = a[j].conj();
                                    for (t, m) in &metric[b] {
                                        z[*t] += x.clone() * m.clone();
                                    }
                                }
                                z
                            });
                            for &(t, c) in components {
                                value += z[t].clone() * values[c].clone();
                            }
                        }
                    }
                    if let Some(c) = entry.coupling {
                        value = couplings[c].clone() * value;
                    }
                    out[entry.output] += value;
                }
                out
            })
            .collect();
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
        let czero = Complex::new(zero.clone(), zero.clone());
        let per_row: Vec<Vec<T>> = (0..rows)
            .into_par_iter()
            .map(|row| {
                let a = &born[row * born_width..][..born_width];
                let b = born_ket.map_or(a, |born| &born[row * born_width..][..born_width]);
                let k = &ket[row * ket_width..][..ket_width];
                let mut out = vec![zero.clone(); self.outputs.len()];
                // z per (bra order, bra group, class).
                let mut cache: BTreeMap<(usize, usize, usize), Vec<Complex<T>>> = BTreeMap::new();
                for (entry, classes) in self.entries.iter().zip(&self.classes) {
                    if !poles && self.outputs[entry.output] != 0 {
                        continue;
                    }
                    let (kets, values) = match entry.ket {
                        Ket::Vector(v) => (&self.vectors[v], k),
                        Ket::Born(o) => (&self.bra[o], b),
                    };
                    let mut value = czero.clone();
                    for (g, ((_p, bras), classes)) in
                        self.bra[entry.bra].iter().zip(classes).enumerate()
                    {
                        for ((_q, components), &class) in kets.iter().zip(classes) {
                            let z = cache.entry((entry.bra, g, class)).or_insert_with(|| {
                                let mut z = vec![czero.clone(); self.tensors];
                                let metric = &runtime.metrics[class];
                                for &(b, j) in bras {
                                    let x = a[j].conj();
                                    for (t, m) in &metric[b] {
                                        z[*t] += x.clone() * m.clone();
                                    }
                                }
                                z
                            });
                            for &(t, c) in components {
                                value += z[t].clone() * values[c].clone();
                            }
                        }
                    }
                    if let Some(c) = entry.coupling {
                        value = couplings[c].clone() * value;
                    }
                    out[entry.output] += value.re.clone() + value.re;
                }
                out
            })
            .collect();
        let mut out = vec![zero.clone(); self.outputs.len()];
        for row in per_row {
            for (total, value) in out.iter_mut().zip(row) {
                *total += value;
            }
        }
        out
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
