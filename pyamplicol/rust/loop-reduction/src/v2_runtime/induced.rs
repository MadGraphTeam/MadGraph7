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
use crate::v2_decimal::{Arith, Binary64, C, DecimalArith, cadd, square_amplitudes_python};
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
    /// The declared observables (`induced_observables.correlators_record`).
    #[serde(default)]
    pub(crate) correlators: Option<serde_json::Value>,
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
        vectors: &[Option<Vec<Vec<C<Self::N>>>>],
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
    let small = |x: &Rational| (if x.is_negative() { -x.clone() } else { x.clone() }) <= limit;
    if small(&numerator) && small(&denominator) {
        return numerator.to_f64() / denominator.to_f64();
    }
    value.to_multi_prec_float(256).to_f64()
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
        vectors: &[Option<Vec<Vec<C<f64>>>>],
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
        let exact = crate::v2_decimal::Decimal::parse(&magnitude.to_string())
            .expect("an integer literal");
        if power >= 0 {
            self.context.plus(&exact)
        } else {
            self.div(&self.int(1), &exact)
        }
    }
    fn square(
        &self,
        vectors: &[Option<Vec<Vec<C<crate::v2_decimal::Decimal>>>>],
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
    metric: Vec<(Rational, Rational)>,
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
            return Err(invalid("the amplitude record's rows differ from the plan's"));
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

    /// |A|² per record sector (`[finite, 1/ε, 1/ε²]` or the finite part,
    /// `width`) of one point from the executor's bare amplitude slots
    /// (`amplitudes[order * rows * T + row * T + t]`, Laurent coefficients)
    /// and the point's counterterm kets (`rows × ket components`; none
    /// without counterterms), in the carrier of `arith` (`bits`: the
    /// executor carrier's).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn square_point<T: Native, A: Lift<T> + Squaring>(
        &self,
        arith: &A,
        amplitudes: &[Vec<Complex<T>>],
        kets: Option<&[Complex<T>]>,
        rows: usize,
        width: usize,
        bits: u32,
        average: i64,
    ) -> RusticolResult<Vec<Vec<A::N>>>
    where
        Complex<T>: symbolica::domains::float::Real + symbolica::evaluate::EvaluationDomain,
        rusticol_core::loop_v2::trees::TreeRuntime<T>: Send,
    {
        let size = self.record.size;
        let zero = C::new(arith.zero(), arith.zero());
        let columns = rows * size;
        // `read_native`'s bare vectors per (plan order, row): every slot's
        // Laurent coefficients (an empty basis has none).
        let mut bare: BTreeMap<(Orders, usize), Vec<Vec<C<A::N>>>> = BTreeMap::new();
        let mut order_keys: Vec<(Orders, usize)> = Vec::new();
        if size > 0 && amplitudes.len() != self.plan_orders.len() * columns {
            return Err(invalid("the executor's amplitude slots differ from the plan's"));
        }
        for (o, order) in self.plan_orders.iter().enumerate() {
            for r in 0..rows {
                let mut vector = Vec::with_capacity(size);
                for t in 0..size {
                    let slot = &amplitudes[o * columns + r * size + t];
                    vector.push(
                        slot.iter()
                            .map(|z| arith.complex(z, bits))
                            .collect::<RusticolResult<Vec<_>>>()?,
                    );
                }
                order_keys.push((order.clone(), r));
                bare.insert((order.clone(), r), vector);
            }
        }
        // `native_kets`: A^CT per (loop order, row) on the basis.
        let mut ket_vectors: BTreeMap<(Orders, usize), Vec<Vec<C<A::N>>>> = BTreeMap::new();
        let mut ket_keys: Vec<(Orders, usize)> = Vec::new();
        if let Some(values) = kets {
            let components = self.record.ket_map.len();
            if values.len() != rows * components {
                return Err(invalid("the counterterm kets differ from the ket map"));
            }
            for r in 0..rows {
                for (j, (t, k, order, nc)) in self.record.ket_map.iter().enumerate() {
                    if *k >= width {
                        continue;
                    }
                    let value = arith.complex(&values[r * components + j], bits)?;
                    let key = (order.clone(), r);
                    let ket = ket_vectors.entry(key.clone()).or_insert_with(|| {
                        ket_keys.push(key);
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
        let mut total: BTreeMap<(Orders, usize), Vec<Vec<C<A::N>>>> = BTreeMap::new();
        for (keys, part) in [(&order_keys, &bare), (&ket_keys, &ket_vectors)] {
            for key in keys {
                let vector = &part[key];
                let cells = total
                    .entry(key.clone())
                    .or_insert_with(|| vec![vec![zero.clone(); width]; size]);
                for t in 0..size {
                    for k in 0..width {
                        cells[t][k] = cadd(arith, &cells[t][k], &vector[t][k]);
                    }
                }
            }
        }
        // `metric(slice_, numbers)`: `numbers.colour(cell, 0)`, the cell
        // times `2^0` (Python's `complex * 1.0`, `Complex * Decimal(1)`).
        let one = arith.int(1);
        let gram = self
            .metric
            .iter()
            .map(|(re, im)| {
                let cell = C::new(arith.rational(re)?, arith.rational(im)?);
                Ok(arith.times_real(&cell, &one))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let vectors: Vec<Option<Vec<Vec<C<A::N>>>>> = self
            .slots
            .iter()
            .flat_map(|order| (0..rows).map(move |r| (order.clone(), r)))
            .map(|key| total.remove(&key))
            .collect();
        arith.square(
            &vectors,
            self.slots.len(),
            rows,
            size,
            width,
            &gram,
            &self.pairs,
            average,
        )
    }
}
