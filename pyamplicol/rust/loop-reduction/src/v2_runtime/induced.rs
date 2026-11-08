// SPDX-License-Identifier: 0BSD

//! The loop-induced assembly of a stored process (m-lit LIT-SDK; the Python
//! API's native route, `loops/v2/induced.read_native`, `native_kets`,
//! `add_amplitudes` and `assemble`): per point the executor's bare loop
//! amplitude per (loop order, helicity row, amplitude basis element) and the
//! tree program's counterterm kets on the amplitude basis at N_c = 3 are
//! added into the amplitude vectors `a = a_bare + A^CT`, and |A|² per
//! sector is their square with the full-colour metric G in the canonical
//! order (DESIGN §7.10): in binary64 by the one native square of RULED Q41
//! (`rusticol_core::loop_v2::interference::square_amplitudes`, which the
//! Python API's `induced.native_square` calls through PyO3), in
//! double-double and arbitrary precision in Python's `Decimal` arithmetic
//! (`v2_decimal::square_amplitudes_python`, `induced.square`). Every
//! operation is the Python API's, in its order, so binary64 values are
//! bitwise and `Decimal` values value-identical.

use super::{Lift, Native, Orders, invalid};
use crate::Complex;
use crate::v2_decimal::{
    AmplitudeVector, Arith, Binary64, C, DecimalArith, cadd, square_amplitudes_python,
};
use rusticol_core::RusticolResult;
use rusticol_core::loop_v2::dress::parse_rational;
use serde::Deserialize;
use std::collections::BTreeMap;
use symbolica::domains::rational::Rational;

/// A stored loop-induced process's amplitude record
/// (`induced.amplitude_record`, read by `induced.StoredAmplitude`).
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct AmplitudeRecord {
    /// The amplitude basis size T.
    pub(crate) size: usize,
    /// G_{tt'} at N_c = 3: exact rationals `[re, im]` per cell.
    pub(crate) metric: Vec<Vec<(String, String)>>,
    /// The basis elements the counterterm kets carry.
    pub(crate) counterterms: Vec<usize>,
    /// Per counterterm ket component of the tree program: its basis element
    /// t, Laurent index k, loop orders and N_c power (`induced.ket_map`).
    pub(crate) ket_map: Vec<(usize, usize, Orders, i64)>,
    /// The helicity rows of the amplitude-mode plan (per leg its helicity).
    pub(crate) rows: Vec<Vec<i32>>,
    /// The exact colour-zero loop orders (RULED Q13).
    pub(crate) colour_zero_orders: Vec<Orders>,
    /// Whether the whole request is an exact colour zero.
    pub(crate) colour_zero: bool,
    /// The declared observables (`induced_observables.correlators_record`):
    /// absent without a declaration.
    #[serde(default)]
    pub(crate) correlators: Option<CorrelatorsRecord>,
}

/// A loop-induced process's declared observables
/// (`induced_observables.correlators_record`): the resolved
/// `CorrelatorConfig` (its colour histories in order, its spin classes) and
/// each history's matrix M_X on the amplitude basis at N_c = 3.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct CorrelatorsRecord {
    pub(crate) declaration: Declaration,
    pub(crate) matrices: Vec<MatrixRecord>,
}

/// The parts of a stored `CorrelatorConfig` the outputs read.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Declaration {
    #[serde(default)]
    pub(crate) color_correlations: Vec<HistoryRecord>,
    #[serde(default)]
    pub(crate) spin_correlations: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct HistoryRecord {
    pub(crate) id: String,
}

/// One history's matrix: exact rationals `[re, im]` per cell.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct MatrixRecord {
    pub(crate) id: String,
    pub(crate) matrix: Vec<Vec<(String, String)>>,
}

/// The carrier arithmetic of the loop-induced assembly: the Python API's
/// `numbers.real(Fraction)` and `numbers.real(3) ** nc`, and its squaring
/// (`native_square` in binary64, `square` in `Decimal`).
pub(crate) trait Squaring: Arith {
    /// `numbers.real(fraction)`: `float(Fraction)` (correctly rounded) or
    /// `Decimal(numerator) / Decimal(denominator)`.
    fn rational(&self, value: &Rational) -> RusticolResult<Self::N>;
    /// `numbers.real(3) ** power`: `3.0 ** power` (C `pow`) or
    /// `Decimal(3) ** power` (exact for `power >= 0`, correctly rounded
    /// otherwise).
    fn three_power(&self, power: i64) -> Self::N;
    /// |A|² per sector of the amplitude vectors (`vectors[(slot, row)]`,
    /// `None`: absent in the Python dict) in the canonical order.
    #[allow(clippy::too_many_arguments)]
    fn square(
        &self,
        vectors: &[Option<AmplitudeVector<Self::N>>],
        slots: usize,
        rows: usize,
        size: usize,
        width: usize,
        metric: &[C<Self::N>],
        pairs: &[Vec<(usize, usize)>],
        average: i64,
    ) -> RusticolResult<Vec<Vec<Self::N>>>;
}

/// `float(Fraction(n, d))`: correctly rounded (exact operands below 2^53,
/// else a 256-bit quotient rounded once more, which only a quotient within
/// 2^-200 of a binary64 midpoint could round differently).
fn rational_f64(value: &Rational) -> f64 {
    let limit = Rational::from(1_i64 << 53);
    let numerator = Rational::from(value.numerator());
    let denominator = Rational::from(value.denominator());
    let small = |x: &Rational| {
        (if x.is_negative() {
            -x.clone()
        } else {
            x.clone()
        }) <= limit
    };
    if small(&numerator) && small(&denominator) {
        return numerator.to_f64() / denominator.to_f64();
    }
    symbolica::domains::float::RealLike::to_f64(&value.to_multi_prec_float(256))
}

impl Squaring for Binary64 {
    fn rational(&self, value: &Rational) -> RusticolResult<f64> {
        Ok(rational_f64(value))
    }
    fn three_power(&self, power: i64) -> f64 {
        3.0_f64.powf(power as f64)
    }
    fn square(
        &self,
        vectors: &[Option<AmplitudeVector<f64>>],
        slots: usize,
        rows: usize,
        size: usize,
        width: usize,
        metric: &[C<f64>],
        pairs: &[Vec<(usize, usize)>],
        average: i64,
    ) -> RusticolResult<Vec<Vec<f64>>> {
        // `induced.native_square`: dense slots (loop order, row, t), zero
        // where absent, `[re, im]` per Laurent coefficient up to `width`.
        let zero = vec![Complex::new(0.0, 0.0); width];
        let mut values = Vec::with_capacity(slots * rows * size);
        for vector in vectors {
            for t in 0..size {
                match vector {
                    None => values.push(zero.clone()),
                    Some(vector) => values.push(
                        vector[t][..width]
                            .iter()
                            .map(|z| Complex::new(z.re, z.im))
                            .collect(),
                    ),
                }
            }
        }
        let metric: Vec<Complex<f64>> = metric.iter().map(|z| Complex::new(z.re, z.im)).collect();
        rusticol_core::loop_v2::interference::square_amplitudes(
            &values,
            slots,
            rows,
            size,
            width,
            &metric,
            pairs,
            &(average as f64),
        )
    }
}

impl Squaring for DecimalArith {
    fn rational(&self, value: &Rational) -> RusticolResult<crate::v2_decimal::Decimal> {
        let read = |x: String| crate::v2_decimal::Decimal::parse(&x).map_err(invalid);
        Ok(self.div(
            &read(value.numerator().to_string())?,
            &read(value.denominator().to_string())?,
        ))
    }
    fn three_power(&self, power: i64) -> crate::v2_decimal::Decimal {
        let magnitude = symbolica::domains::integer::Integer::from(3).pow(power.unsigned_abs());
        let exact =
            crate::v2_decimal::Decimal::parse(&magnitude.to_string()).expect("an integer literal");
        if power >= 0 {
            self.context.plus(&exact)
        } else {
            self.div(&self.int(1), &exact)
        }
    }
    fn square(
        &self,
        vectors: &[Option<AmplitudeVector<crate::v2_decimal::Decimal>>],
        _slots: usize,
        rows: usize,
        size: usize,
        width: usize,
        metric: &[C<crate::v2_decimal::Decimal>],
        pairs: &[Vec<(usize, usize)>],
        average: i64,
    ) -> RusticolResult<Vec<Vec<crate::v2_decimal::Decimal>>> {
        Ok(square_amplitudes_python(
            self, vectors, rows, size, width, metric, pairs, average,
        ))
    }
}

/// The loop-induced assembly of one stored process.
pub(crate) struct Induced {
    pub(crate) record: AmplitudeRecord,
    /// G_{tt'} as exact rationals (real, imaginary), row-major.
    metric: Vec<Exact>,
    /// The plan's loop orders in the executor's slot order (the native
    /// record's `orders`).
    plan_orders: Vec<Orders>,
    /// The sectors' loop orders, sorted: the squaring's slots
    /// (`induced.native_square`'s `orders`).
    slots: Vec<Orders>,
    /// Per record sector its loop-order slot pairs, sorted.
    pairs: Vec<Vec<(usize, usize)>>,
}

impl Induced {
    /// The assembly of a record with the plan's loop orders `plan_orders` and
    /// the record's sectors (per sector its (l, l') loop-order pairs).
    pub(crate) fn new(
        record: AmplitudeRecord,
        plan_orders: Vec<Orders>,
        sectors: &[(Orders, Vec<(Orders, Orders)>)],
        rows: usize,
    ) -> RusticolResult<Self> {
        if record.metric.len() != record.size
            || record.metric.iter().any(|row| row.len() != record.size)
        {
            return Err(invalid("the amplitude metric is basis x basis"));
        }
        if record.rows.len() != rows {
            return Err(invalid(
                "the amplitude record's rows differ from the plan's",
            ));
        }
        if record.ket_map.iter().any(|(t, ..)| *t >= record.size) {
            return Err(invalid("a counterterm ket outside the amplitude basis"));
        }
        let metric = record
            .metric
            .iter()
            .flatten()
            .map(|(re, im)| {
                Ok((
                    parse_rational(re).map_err(|e| invalid(e.message()))?,
                    parse_rational(im).map_err(|e| invalid(e.message()))?,
                ))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let mut slots: Vec<Orders> = sectors
            .iter()
            .flat_map(|(_, pairs)| pairs.iter().flat_map(|(l, r)| [l.clone(), r.clone()]))
            .collect();
        slots.sort();
        slots.dedup();
        let slot = |order: &Orders| slots.iter().position(|o| o == order).expect("a slot");
        let pairs = sectors
            .iter()
            .map(|(_, pairs)| {
                let mut found: Vec<(usize, usize)> =
                    pairs.iter().map(|(l, r)| (slot(l), slot(r))).collect();
                found.sort();
                found
            })
            .collect();
        Ok(Self {
            record,
            metric,
            plan_orders,
            slots,
            pairs,
        })
    }

    /// The plan's helicity rows as the Python API's `rx.Row` (per leg, in
    /// order, its 0-based index and helicity): `induced.rows_of`.
    pub(crate) fn plan_rows(&self) -> Vec<Row> {
        self.record
            .rows
            .iter()
            .map(|row| row.iter().copied().enumerate().collect())
            .collect()
    }

    /// The amplitude basis size T.
    pub(crate) fn size(&self) -> usize {
        self.record.size
    }

    /// The colour metric G at N_c = 3, exact (row-major).
    pub(crate) fn metric(&self) -> &[Exact] {
        &self.metric
    }

    /// The amplitude vectors a = a_bare + A^CT of one point per (loop
    /// order, helicity row), in the Python API's insertion order
    /// (`read_native`'s bare vectors per (plan order, row), then the
    /// counterterm kets' new keys; `add_amplitudes`), from the executor's
    /// bare amplitude slots (`amplitudes[order * rows * T + row * T + t]`,
    /// Laurent coefficients) and the point's counterterm kets (`rows × ket
    /// components`; none without counterterms), in the carrier of `arith`
    /// (`bits`: the executor carrier's).
    pub(crate) fn vectors_point<T: Native, A: Lift<T> + Squaring>(
        &self,
        arith: &A,
        amplitudes: &[Vec<Complex<T>>],
        kets: Option<&[Complex<T>]>,
        width: usize,
        bits: u32,
    ) -> RusticolResult<Vectors<A::N>>
    where
        Complex<T>: symbolica::domains::float::Real + symbolica::evaluate::EvaluationDomain,
        rusticol_core::loop_v2::trees::TreeRuntime<T>: Send,
    {
        Ok(self
            .vector_parts(arith, amplitudes, kets, width, bits)?
            .total)
    }

    /// The sectors' loop orders, sorted (the squaring's slots).
    pub(crate) fn slots(&self) -> &[Orders] {
        &self.slots
    }

    /// Per record sector its loop-order slot pairs, sorted.
    pub(crate) fn pairs(&self) -> &[Vec<(usize, usize)>] {
        &self.pairs
    }

    /// The plan's loop orders in the executor's slot order.
    pub(crate) fn plan_orders(&self) -> &[Orders] {
        &self.plan_orders
    }

    /// [`Induced::vectors_point`] with its parts: the bare vectors
    /// (`bare_vectors`), the counterterm kets (`counterterm_vectors`) and
    /// their sum (`amplitude_vectors`).
    pub(crate) fn vector_parts<T: Native, A: Lift<T> + Squaring>(
        &self,
        arith: &A,
        amplitudes: &[Vec<Complex<T>>],
        kets: Option<&[Complex<T>]>,
        width: usize,
        bits: u32,
    ) -> RusticolResult<VectorParts<A::N>>
    where
        Complex<T>: symbolica::domains::float::Real + symbolica::evaluate::EvaluationDomain,
        rusticol_core::loop_v2::trees::TreeRuntime<T>: Send,
    {
        let size = self.record.size;
        let rows = self.plan_rows();
        let zero = C::new(arith.zero(), arith.zero());
        let columns = rows.len() * size;
        // `read_native`'s bare vectors per (plan order, row): every slot's
        // Laurent coefficients (an empty basis has none).
        let mut bare: Vectors<A::N> = Vectors::default();
        if size > 0 && amplitudes.len() != self.plan_orders.len() * columns {
            return Err(invalid(
                "the executor's amplitude slots differ from the plan's",
            ));
        }
        for (o, order) in self.plan_orders.iter().enumerate() {
            for (r, row) in rows.iter().enumerate() {
                let mut vector = Vec::with_capacity(size);
                for t in 0..size {
                    let slot = &amplitudes[o * columns + r * size + t];
                    vector.push(
                        slot.iter()
                            .map(|z| arith.complex(z, bits))
                            .collect::<RusticolResult<Vec<_>>>()?,
                    );
                }
                bare.insert((order.clone(), row.clone()), vector);
            }
        }
        // `native_kets`: A^CT per (loop order, row) on the basis.
        let mut ket_vectors: Vectors<A::N> = Vectors::default();
        if let Some(values) = kets {
            let components = self.record.ket_map.len();
            if values.len() != rows.len() * components {
                return Err(invalid("the counterterm kets differ from the ket map"));
            }
            for (r, row) in rows.iter().enumerate() {
                for (j, (t, k, order, nc)) in self.record.ket_map.iter().enumerate() {
                    if *k >= width {
                        continue;
                    }
                    let value = arith.complex(&values[r * components + j], bits)?;
                    let ket = ket_vectors.entry((order.clone(), row.clone()), || {
                        vec![vec![zero.clone(); width]; size]
                    });
                    let scaled = arith.times_real(&value, &arith.three_power(*nc));
                    ket[*t][*k] = cadd(arith, &ket[*t][*k], &scaled);
                }
            }
        } else if !self.record.counterterms.is_empty() {
            // Never a silent loss of the R2/UV part (`induced.assemble`).
            return Err(invalid(
                "a loop-induced native point needs the executor's counterterm \
                 kets (native trees with counterterms)",
            ));
        }
        // `add_amplitudes(bare, kets)`: zero, plus every part's cells in
        // order.
        let mut total: Vectors<A::N> = Vectors::default();
        for part in [&bare, &ket_vectors] {
            for (key, vector) in part.iter() {
                let cells = total.entry(key.clone(), || vec![vec![zero.clone(); width]; size]);
                for t in 0..size {
                    for k in 0..width {
                        cells[t][k] = cadd(arith, &cells[t][k], &vector[t][k]);
                    }
                }
            }
        }
        Ok(VectorParts {
            bare,
            kets: ket_vectors,
            total,
        })
    }

    /// An exact matrix on the amplitude basis in the carrier ([`carrier`]).
    pub(crate) fn carrier<A: Squaring>(
        &self,
        arith: &A,
        cells: &[Exact],
    ) -> RusticolResult<Vec<C<A::N>>> {
        carrier(arith, cells)
    }

    /// The canonical square (`induced.native_square` / `induced.square`) of
    /// `vectors` with the carrier matrix `metric` over the helicity rows
    /// `rows` (in their order): per record sector its Laurent coefficients.
    pub(crate) fn square_vectors<A: Squaring>(
        &self,
        arith: &A,
        vectors: &Vectors<A::N>,
        rows: &[Row],
        metric: &[C<A::N>],
        width: usize,
        average: i64,
    ) -> RusticolResult<Vec<Vec<A::N>>> {
        let dense: Vec<Option<AmplitudeVector<A::N>>> = self
            .slots
            .iter()
            .flat_map(|order| rows.iter().map(move |row| (order, row)))
            .map(|(order, row)| vectors.get(&(order.clone(), row.clone())).cloned())
            .collect();
        arith.square(
            &dense,
            self.slots.len(),
            rows.len(),
            self.record.size,
            width,
            metric,
            &self.pairs,
            average,
        )
    }
}

/// An exact matrix on the amplitude basis in the carrier
/// (`numbers.colour(cell, 0)`: the cell times `2^0`, Python's `complex *
/// 1.0`, `Complex * Decimal(1)`).
pub(crate) fn carrier<A: Squaring>(arith: &A, cells: &[Exact]) -> RusticolResult<Vec<C<A::N>>> {
    let one = arith.int(1);
    cells
        .iter()
        .map(|(re, im)| {
            let cell = C::new(arith.rational(re)?, arith.rational(im)?);
            Ok(arith.times_real(&cell, &one))
        })
        .collect()
}

/// One point's amplitude vectors with their parts (`induced.assemble`).
pub(crate) struct VectorParts<N> {
    pub(crate) bare: Vectors<N>,
    pub(crate) kets: Vectors<N>,
    pub(crate) total: Vectors<N>,
}

/// A helicity row as the Python API's `rx.Row`: per leg its 0-based index
/// and helicity (a spin-correlated output's rows drop the spin legs).
pub type Row = Vec<(usize, i32)>;
/// An exact complex cell (real, imaginary).
pub type Exact = (Rational, Rational);

/// Amplitude vectors per (loop order, row) in insertion order, as the
/// Python API's dicts (`amplitude_vectors`): iteration in insertion order,
/// lookup by key.
#[derive(Clone, Debug)]
pub struct Vectors<N> {
    keys: Vec<(Orders, Row)>,
    values: BTreeMap<(Orders, Row), AmplitudeVector<N>>,
}

impl<N> Default for Vectors<N> {
    fn default() -> Self {
        Self {
            keys: Vec::new(),
            values: BTreeMap::new(),
        }
    }
}

impl<N> Vectors<N> {
    /// `out[key] = vector` (a new key appends, an old one keeps its place).
    pub fn insert(&mut self, key: (Orders, Row), vector: AmplitudeVector<N>) {
        if !self.values.contains_key(&key) {
            self.keys.push(key.clone());
        }
        self.values.insert(key, vector);
    }
    /// The vector of `key`, made by `make` when absent (`out.get(key)` or
    /// `out[key] = make()`).
    pub fn entry(
        &mut self,
        key: (Orders, Row),
        make: impl FnOnce() -> AmplitudeVector<N>,
    ) -> &mut AmplitudeVector<N> {
        if !self.values.contains_key(&key) {
            self.keys.push(key.clone());
            self.values.insert(key.clone(), make());
        }
        self.values.get_mut(&key).expect("an entry")
    }
    pub fn get(&self, key: &(Orders, Row)) -> Option<&AmplitudeVector<N>> {
        self.values.get(key)
    }
    /// (key, vector) in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&(Orders, Row), &AmplitudeVector<N>)> {
        self.keys.iter().map(|key| (key, &self.values[key]))
    }
    pub fn len(&self) -> usize {
        self.keys.len()
    }
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}
