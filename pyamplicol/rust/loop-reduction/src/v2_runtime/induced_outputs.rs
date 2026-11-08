// SPDX-License-Identifier: 0BSD

//! The observables of a loop-induced process (m-lit LIT-SDK; the Python
//! API's `loops/v2/induced_observables`, LIT-OBS, RULED Q9; DECISIONS
//! 2026-11-07 21:50 R7: one shared implementation).
//!
//! Every output is a quadratic form on the point's amplitude vectors
//! ([`Induced::vectors_point`]), squared by the one canonical square:
//!
//! * colour-correlated: a declared history's stored matrix M_X as its
//!   Hermitian parts H = (M + M^dag)/2 and K = (M - M^dag)/2i
//!   ([`hermitian_parts`]), each squared (`form`: real for a Hermitian M,
//!   else H-square + i K-square); the history `born` is G itself;
//! * spin-correlated: a spin vector v of a declared leg j enters through its
//!   coefficients c_h = -v . conj(w_h) on the leg's polarisation vectors
//!   ([`polarisations`], [`spin_coefficients`]: `Decimal` at the working
//!   digits plus [`SPIN_GUARD_DIGITS`], as `PointOutputs`), A(v) = sum_h c_h
//!   A(h) ([`spin_amplitudes`]);
//! * helicity-resolved: per helicity row the inclusive form over that row
//!   alone (the colour-contracted resolution, `CONTRACTED`); the
//!   amplitude-basis anchors ([`anchor_metric`]) are served to the Python
//!   API.
//!
//! The Python API's binary64 outputs call these functions through PyO3
//! (`induced_spin_coefficients`, `induced_spin_amplitudes`,
//! `induced_forms`), so the SDK's binary64 outputs are bitwise the Python
//! API's by construction; its `Decimal` code stays the oracle and the
//! double-double / arbitrary-precision path, which [`DecimalArith`] follows
//! operation for operation (equal `Decimal` values).

use super::columns::{ColumnPoint, ColumnValues, INCLUSIVE};
pub use super::induced::{Exact, Row, Vectors};
use super::induced::{Induced, Squaring, carrier};
use super::{Lift, Native, V2Poles, V2Precision, V2Runtime, contraction, invalid, v2_batch};
use crate::Complex;
use crate::v2_decimal::{Arith, C, Decimal, DecimalArith, DecimalContext, cadd, cmul};
use rusticol_core::RusticolResult;
use rusticol_core::loop_execution::v2::V2Options;
use rusticol_core::loop_v2::dress::parse_rational;
use rusticol_core::loop_v2::trees::TreeRuntime;
use std::cmp::Ordering;
use std::collections::BTreeMap;
use symbolica::domains::float::Real;
use symbolica::domains::rational::Rational;
use symbolica::evaluate::EvaluationDomain;

/// A polarisation vector: four complex `Decimal` components.
pub type Wave = [(Decimal, Decimal); 4];
/// Per 0-based spin leg its carrier coefficient per helicity.
pub type Coefficients<N> = BTreeMap<usize, Vec<(i32, C<N>)>>;
/// A metric's carrier forms: H and, for a directed history, K.
type Form<N> = (Vec<C<N>>, Option<Vec<C<N>>>);

/// (H, K) with M = H + i K (`induced_observables.hermitian_parts`): H =
/// (M + M^dag)/2, K = (M - M^dag)/2i, exact, row-major; K is None when M is
/// Hermitian (a dipole).
pub fn hermitian_parts(matrix: &[Exact], size: usize) -> (Vec<Exact>, Option<Vec<Exact>>) {
    let half = Rational::new(1, 2);
    let mut h = Vec::with_capacity(size * size);
    let mut k = Vec::with_capacity(size * size);
    for t in 0..size {
        for u in 0..size {
            let (ar, ai) = &matrix[t * size + u];
            let (br, bi) = &matrix[u * size + t];
            // M_tu + conj(M_ut), and (M_tu - conj(M_ut)) (-i/2).
            h.push(((ar + br) * &half, (ai - bi) * &half));
            let (dr, di) = (ar - br, ai + bi);
            k.push((di * &half, -(dr * &half)));
        }
    }
    let hermitian = k.iter().all(|(re, im)| re.is_zero() && im.is_zero());
    (h, (!hermitian).then_some(k))
}

/// (E_t G + G E_t)/2 (`induced_observables.anchor_metric`): a^dag of it a =
/// Re(conj(a_t) (G a)_t).
pub fn anchor_metric(gram: &[Exact], size: usize, t: usize) -> Vec<Exact> {
    let half = Rational::new(1, 2);
    let zero = Rational::new(0, 1);
    let mut out = Vec::with_capacity(size * size);
    for r in 0..size {
        for u in 0..size {
            let cell = &gram[r * size + u];
            let left = if r == t {
                cell.clone()
            } else {
                (zero.clone(), zero.clone())
            };
            let right = if u == t {
                cell.clone()
            } else {
                (zero.clone(), zero.clone())
            };
            out.push(((&left.0 + &right.0) * &half, (&left.1 + &right.1) * &half));
        }
    }
    out
}

/// Python's unary minus of a `Decimal` (`-0` is `0`), rounded.
fn neg(ctx: &DecimalContext, x: &Decimal) -> Decimal {
    ctx.plus(&x.negated())
}

/// `_fortran_sign(value, source)`: |value| with the sign of `source`
/// (positive for `source >= 0`).
fn fortran_sign(ctx: &DecimalContext, value: &Decimal, source: &Decimal) -> Decimal {
    let magnitude = ctx.abs(value);
    if source.is_negative() {
        neg(ctx, &magnitude)
    } else {
        magnitude
    }
}

/// Python's `min(a, b)` of decimals: `a` unless `b < a`.
fn least(a: Decimal, b: Decimal) -> Decimal {
    if b.cmp_value(&a) == Ordering::Less {
        b
    } else {
        a
    }
}

fn sqrt(ctx: &DecimalContext, x: &Decimal, context: &str) -> RusticolResult<Decimal> {
    if x.is_negative() {
        return Err(invalid(format!(
            "{context} encountered a negative square root"
        )));
    }
    Ok(ctx.sqrt(x))
}

/// `symbolica_exact._massless_vector` in the context.
fn massless_vector(ctx: &DecimalContext, p: &[Decimal; 4], helicity: i32) -> RusticolResult<Wave> {
    let [energy, px, py, pz] = p;
    if energy.is_zero() {
        return Err(invalid("cannot build a massless vector with zero energy"));
    }
    let zero = Decimal::zero();
    let sqh = sqrt(
        ctx,
        &ctx.div(&Decimal::integer(1), &Decimal::integer(2)),
        "vector source",
    )?;
    let positive = !energy.is_negative();
    let (hel, pp) = if positive {
        (Decimal::integer(i64::from(helicity)), energy.clone())
    } else {
        (Decimal::integer(-i64::from(helicity)), neg(ctx, energy))
    };
    let pt = sqrt(
        ctx,
        &ctx.add(&ctx.mul(px, px), &ctx.mul(py, py)),
        "vector source",
    )?;
    let wf3 = (
        ctx.mul(&ctx.div(&ctx.mul(&hel, &pt), &pp), &sqh),
        zero.clone(),
    );
    let (wf1, wf2);
    if !pt.is_zero() {
        let pz = if positive { pz.clone() } else { neg(ctx, pz) };
        let pzpt = ctx.mul(&ctx.mul(&ctx.div(&pz, &ctx.mul(&pp, &pt)), &sqh), &hel);
        if positive {
            wf1 = (
                ctx.mul(&neg(ctx, px), &pzpt),
                ctx.mul(&ctx.div(&neg(ctx, py), &pt), &sqh),
            );
            wf2 = (
                ctx.mul(&neg(ctx, py), &pzpt),
                ctx.mul(&ctx.div(px, &pt), &sqh),
            );
        } else {
            wf1 = (ctx.mul(px, &pzpt), ctx.mul(&ctx.div(py, &pt), &sqh));
            wf2 = (
                ctx.mul(py, &pzpt),
                ctx.mul(&ctx.div(&neg(ctx, px), &pt), &sqh),
            );
        }
    } else {
        wf1 = (ctx.mul(&neg(ctx, &hel), &sqh), zero.clone());
        let sign = fortran_sign(ctx, &sqh, pz);
        wf2 = (zero.clone(), if positive { sign } else { neg(ctx, &sign) });
    }
    Ok([(zero.clone(), zero), wf1, wf2, wf3])
}

/// `symbolica_exact._massive_vector` in the context.
fn massive_vector(
    ctx: &DecimalContext,
    p: &[Decimal; 4],
    helicity: i32,
    mass: &Decimal,
) -> RusticolResult<Wave> {
    if mass.is_zero() {
        return Err(invalid("massive-vector source has zero mass"));
    }
    if p[0].is_negative() {
        let flipped = [
            neg(ctx, &p[0]),
            neg(ctx, &p[1]),
            neg(ctx, &p[2]),
            neg(ctx, &p[3]),
        ];
        return massive_vector(ctx, &flipped, -helicity, mass);
    }
    let [energy, px, py, pz] = p;
    let zero = Decimal::zero();
    let context = "massive vector source";
    let sqh = sqrt(
        ctx,
        &ctx.div(&Decimal::integer(1), &Decimal::integer(2)),
        context,
    )?;
    let hel = Decimal::integer(i64::from(helicity));
    let nsvahl = Decimal::integer(i64::from(helicity.unsigned_abs()));
    let pt2 = ctx.add(&ctx.mul(px, px), &ctx.mul(py, py));
    let pp = least(
        energy.clone(),
        sqrt(ctx, &ctx.add(&pt2, &ctx.mul(pz, pz)), context)?,
    );
    let pt = least(pp.clone(), sqrt(ctx, &pt2, context)?);
    let hel0 = ctx.sub(&Decimal::integer(1), &ctx.abs(&hel));
    if pp.is_zero() {
        return Ok([
            (zero.clone(), zero.clone()),
            (ctx.mul(&neg(ctx, &hel), &sqh), zero.clone()),
            (zero.clone(), ctx.mul(&nsvahl, &sqh)),
            (hel0, zero),
        ]);
    }
    let emp = ctx.div(energy, &ctx.mul(mass, &pp));
    let wf0 = (ctx.div(&ctx.mul(&hel0, &pp), mass), zero.clone());
    let wf3 = (
        ctx.add(
            &ctx.mul(&ctx.mul(&hel0, pz), &emp),
            &ctx.mul(&ctx.div(&ctx.mul(&hel, &pt), &pp), &sqh),
        ),
        zero.clone(),
    );
    let (wf1, wf2);
    if !pt.is_zero() {
        let pzpt = ctx.mul(&ctx.mul(&ctx.div(pz, &ctx.mul(&pp, &pt)), &sqh), &hel);
        let minus = neg(ctx, &nsvahl);
        wf1 = (
            ctx.sub(&ctx.mul(&ctx.mul(&hel0, px), &emp), &ctx.mul(px, &pzpt)),
            ctx.mul(&ctx.div(&ctx.mul(&minus, py), &pt), &sqh),
        );
        wf2 = (
            ctx.sub(&ctx.mul(&ctx.mul(&hel0, py), &emp), &ctx.mul(py, &pzpt)),
            ctx.mul(&ctx.div(&ctx.mul(&nsvahl, px), &pt), &sqh),
        );
    } else {
        wf1 = (ctx.mul(&neg(ctx, &hel), &sqh), zero.clone());
        wf2 = (zero.clone(), ctx.mul(&nsvahl, &fortran_sign(ctx, &sqh, pz)));
    }
    Ok([wf0, wf1, wf2, wf3])
}

/// The polarisation vectors w_h of vector leg `leg` (its 1-based label) at
/// its physical momentum per source helicity h, in the tree program's
/// source order (`induced_observables.polarisations`: the crossing's
/// momentum transform, the source helicity; a massive leg's mass its
/// on-shell invariant), in the context. `momentum` holds the exact
/// components (the Python API reads its decimal texts exactly,
/// `Decimal(str(x))`).
pub fn polarisations(
    ctx: &DecimalContext,
    sources: &[serde_json::Value],
    leg: usize,
    momentum: &[Decimal; 4],
) -> RusticolResult<Vec<(i32, Wave)>> {
    let field = |source: &serde_json::Value, name: &str| -> RusticolResult<i64> {
        source
            .get(name)
            .and_then(serde_json::Value::as_i64)
            .ok_or_else(|| invalid(format!("a tree source record has no integer {name}")))
    };
    let mut out: Vec<(i32, Wave)> = Vec::new();
    for source in sources {
        if field(source, "leg_label")? != leg as i64 {
            continue;
        }
        let vector = source.get("wavefunction_kind").and_then(|v| v.as_str()) == Some("vector");
        if !vector || field(source, "dimension")? != 4 {
            return Err(invalid(format!(
                "spin-correlated leg {leg} requires a full four-component vector source"
            )));
        }
        let mut p = momentum.clone();
        let negate = source
            .pointer("/applied_crossing/momentum_transform")
            .and_then(|v| v.as_str())
            == Some("negate-four-momentum");
        if negate {
            p = [
                neg(ctx, &p[0]),
                neg(ctx, &p[1]),
                neg(ctx, &p[2]),
                neg(ctx, &p[3]),
            ];
        }
        let helicity = i32::try_from(field(source, "source_helicity")?)
            .map_err(|_| invalid("a source helicity out of range"))?;
        let massless = source
            .pointer("/source_ir/mass_parameter")
            .is_none_or(serde_json::Value::is_null);
        let wave = if massless {
            massless_vector(ctx, &p, helicity)?
        } else {
            let square = ctx.sub(
                &ctx.sub(
                    &ctx.sub(&ctx.mul(&p[0], &p[0]), &ctx.mul(&p[1], &p[1])),
                    &ctx.mul(&p[2], &p[2]),
                ),
                &ctx.mul(&p[3], &p[3]),
            );
            if square.is_negative() {
                return Err(invalid(format!(
                    "the momentum of massive spin-correlated leg {leg} is spacelike"
                )));
            }
            massive_vector(ctx, &p, helicity, &ctx.sqrt(&square))?
        };
        match out.iter_mut().find(|(h, _)| *h == helicity) {
            Some(found) => found.1 = wave,
            None => out.push((helicity, wave)),
        }
    }
    if out.is_empty() {
        return Err(invalid(format!(
            "leg {leg} has no source of the tree program"
        )));
    }
    Ok(out)
}

/// c_h = -v . conj(w_h) (Minkowski, mostly minus) per helicity h
/// (`induced_observables.spin_coefficients`), in the context; `vector` the
/// spin vector's exact components.
pub fn spin_coefficients(
    ctx: &DecimalContext,
    vector: &[(Decimal, Decimal); 4],
    waves: &[(i32, Wave)],
) -> Vec<(i32, (Decimal, Decimal))> {
    const SIGNS: [i64; 4] = [1, -1, -1, -1];
    waves
        .iter()
        .map(|(helicity, wave)| {
            let (mut re, mut im) = (Decimal::zero(), Decimal::zero());
            for ((sign, (vr, vi)), (wr, wi)) in SIGNS.iter().zip(vector).zip(wave) {
                let sign = Decimal::integer(*sign);
                re = ctx.add(
                    &re,
                    &ctx.mul(&sign, &ctx.add(&ctx.mul(vr, wr), &ctx.mul(vi, wi))),
                );
                im = ctx.add(
                    &im,
                    &ctx.mul(&sign, &ctx.sub(&ctx.mul(vi, wr), &ctx.mul(vr, wi))),
                );
            }
            (*helicity, (neg(ctx, &re), neg(ctx, &im)))
        })
        .collect()
}

/// The guard digits of the spin coefficients over the carrier's working
/// digits (`induced_observables.SPIN_GUARD_DIGITS`): the polarisation
/// vectors' square roots and the coefficients' products and sums round in
/// this context, so that a coefficient enters the carrier correctly rounded.
pub const SPIN_GUARD_DIGITS: u32 = 10;

/// The `Decimal` context of the spin coefficients of a carrier with
/// `working_digits` (`PointOutputs`: the working digits, which
/// `Precision.working_digits` already floors at its tree-replay digits, plus
/// [`SPIN_GUARD_DIGITS`]).
pub fn spin_context(working_digits: u32) -> DecimalContext {
    DecimalContext::new(working_digits + SPIN_GUARD_DIGITS)
}

/// An exact input as the `Decimal` value the Python API reads from its text
/// (`Decimal(str(x))`): a rational with a finite decimal expansion (every
/// decimal or binary64 input).
fn exact(x: &Rational) -> RusticolResult<Decimal> {
    Decimal::from_rational_exact(x)
        .ok_or_else(|| invalid("an input component is not a finite decimal"))
}

/// A spin vector's exact components as `Decimal` values (the Python API
/// reads them exactly, `correlated_exact._prepare_spin_vectors`).
pub fn exact_vector(vector: &[(Rational, Rational); 4]) -> RusticolResult<[(Decimal, Decimal); 4]> {
    let read = exact;
    Ok([
        (read(&vector[0].0)?, read(&vector[0].1)?),
        (read(&vector[1].0)?, read(&vector[1].1)?),
        (read(&vector[2].0)?, read(&vector[2].1)?),
        (read(&vector[3].0)?, read(&vector[3].1)?),
    ])
}

/// The spin coefficients of every spin leg of one point (`point`: its
/// physical momenta, four exact components per leg) in the carrier: per
/// 0-based leg its coefficient per helicity (`PointOutputs`:
/// `numbers.pair(re, im)` in the spin context, i.e. the nearest binary64
/// values or the `Decimal` values at the spin context's digits).
pub fn point_coefficients<A: CarrierPair>(
    arith: &A,
    sources: &[serde_json::Value],
    point: &[Rational],
    spin: &BTreeMap<usize, [(Rational, Rational); 4]>,
    working_digits: u32,
) -> RusticolResult<Coefficients<A::N>> {
    let ctx = spin_context(working_digits);
    let mut out = BTreeMap::new();
    for (&leg, vector) in spin {
        if leg == 0 || 4 * leg > point.len() {
            return Err(invalid(format!("spin-vector leg {leg} is not a leg")));
        }
        let momentum = &point[4 * (leg - 1)..4 * leg];
        let momentum = [
            exact(&momentum[0])?,
            exact(&momentum[1])?,
            exact(&momentum[2])?,
            exact(&momentum[3])?,
        ];
        let waves = polarisations(&ctx, sources, leg, &momentum)?;
        let exact = spin_coefficients(&ctx, &exact_vector(vector)?, &waves);
        out.insert(
            leg - 1,
            exact
                .into_iter()
                .map(|(h, (re, im))| (h, arith.pair(&ctx, &re, &im)))
                .collect(),
        );
    }
    Ok(out)
}

/// `numbers.pair(re, im)` of spin-context decimals in the carrier.
pub trait CarrierPair: Arith {
    fn pair(&self, ctx: &DecimalContext, re: &Decimal, im: &Decimal) -> C<Self::N>;
}

impl CarrierPair for crate::v2_decimal::Binary64 {
    fn pair(&self, _ctx: &DecimalContext, re: &Decimal, im: &Decimal) -> C<f64> {
        C::new(re.to_f64(), im.to_f64())
    }
}

impl CarrierPair for DecimalArith {
    /// `rx.Complex(+re, +im)` in the spin context (the coefficients keep
    /// its digits; `PointOutputs` forms them inside it).
    fn pair(&self, ctx: &DecimalContext, re: &Decimal, im: &Decimal) -> C<Decimal> {
        C::new(ctx.plus(re), ctx.plus(im))
    }
}

/// A(v) per (loop order, remaining row) (`induced_observables.
/// spin_amplitudes`): over the amplitude vectors in order, the product of
/// the spin legs' coefficients of the row's helicities times the row's
/// vector, added into the row without the spin legs; a row with a helicity
/// a spin leg has no coefficient for is skipped.
pub fn spin_amplitudes<A: Arith>(
    arith: &A,
    amplitudes: &Vectors<A::N>,
    coefficients: &Coefficients<A::N>,
    width: usize,
    size: usize,
) -> Vectors<A::N> {
    let zero = C::new(arith.zero(), arith.zero());
    let mut out: Vectors<A::N> = Vectors::default();
    'rows: for ((order, row), vector) in amplitudes.iter() {
        let mut weight: Option<C<A::N>> = None;
        let mut rest: Row = Vec::new();
        for &(leg, h) in row {
            let Some(found) = coefficients.get(&leg) else {
                rest.push((leg, h));
                continue;
            };
            let Some((_, c)) = found.iter().find(|(helicity, _)| *helicity == h) else {
                continue 'rows;
            };
            weight = Some(match weight {
                None => c.clone(),
                Some(w) => cmul(arith, &w, c),
            });
        }
        let Some(weight) = weight else {
            continue;
        };
        let total = out.entry((order.clone(), rest), || {
            vec![vec![zero.clone(); width]; size]
        });
        for t in 0..size {
            for k in 0..width {
                total[t][k] = cadd(arith, &total[t][k], &cmul(arith, &weight, &vector[t][k]));
            }
        }
    }
    out
}

/// The binary64 carrier forms of an exact matrix on the amplitude basis
/// (`Forms`, `numbers.colour(cell, 0)`): a history's (H, K)
/// ([`hermitian_parts`]) when `hermitian`, else the matrix itself (G, an
/// anchor's metric) and no K; row-major `size x size`.
#[allow(clippy::type_complexity)]
pub fn binary64_form(
    cells: &[Exact],
    size: usize,
    hermitian: bool,
) -> RusticolResult<(Vec<C<f64>>, Option<Vec<C<f64>>>)> {
    if cells.len() != size * size {
        return Err(invalid("a form is basis x basis"));
    }
    let arith = crate::v2_decimal::Binary64;
    if !hermitian {
        return Ok((carrier(&arith, cells)?, None));
    }
    let (h, k) = hermitian_parts(cells, size);
    Ok((
        carrier(&arith, &h)?,
        k.map(|k| carrier(&arith, &k)).transpose()?,
    ))
}

/// The binary64 spin coefficients of one spin leg (`PointOutputs`): the
/// polarisation vectors at the leg's exact momentum, c_h in the spin
/// context of `working_digits`, each the nearest binary64 pair.
pub fn binary64_spin_coefficients(
    sources: &[serde_json::Value],
    leg: usize,
    momentum: &[Decimal; 4],
    vector: &[(Decimal, Decimal); 4],
    working_digits: u32,
) -> RusticolResult<Vec<(i32, C<f64>)>> {
    let ctx = spin_context(working_digits);
    let waves = polarisations(&ctx, sources, leg, momentum)?;
    let arith = crate::v2_decimal::Binary64;
    Ok(spin_coefficients(&ctx, vector, &waves)
        .into_iter()
        .map(|(h, (re, im))| (h, arith.pair(&ctx, &re, &im)))
        .collect())
}

/// The rows of a point's outputs: those of the amplitude vectors in their
/// first-appearance order (`PointOutputs.rows` with spin legs), else the
/// plan's.
pub fn rows_of<N>(vectors: &Vectors<N>) -> Vec<Row> {
    let mut rows: Vec<Row> = Vec::new();
    for ((_, row), _) in vectors.iter() {
        if !rows.contains(row) {
            rows.push(row.clone());
        }
    }
    rows
}

/// The observables of a loop-induced process (`InducedObservables`): its
/// declared histories (in order) with their matrices, its spin classes and
/// the tree program's source records.
pub(crate) struct Observables {
    histories: Vec<String>,
    matrices: BTreeMap<String, Vec<Exact>>,
    spin_classes: Vec<Vec<usize>>,
    sources: Vec<serde_json::Value>,
}

impl Observables {
    pub(crate) fn new(induced: &Induced, trees_json: &[u8]) -> RusticolResult<Self> {
        let size = induced.size();
        let (histories, matrices, spin_classes) = match &induced.record.correlators {
            None => (Vec::new(), BTreeMap::new(), Vec::new()),
            Some(record) => {
                let mut matrices = BTreeMap::new();
                for item in &record.matrices {
                    if item.matrix.len() != size || item.matrix.iter().any(|row| row.len() != size)
                    {
                        return Err(invalid(format!(
                            "the matrix of colour history {:?} is not basis x basis",
                            item.id
                        )));
                    }
                    let cells = item
                        .matrix
                        .iter()
                        .flatten()
                        .map(|(re, im)| {
                            Ok((
                                parse_rational(re).map_err(|e| invalid(e.message()))?,
                                parse_rational(im).map_err(|e| invalid(e.message()))?,
                            ))
                        })
                        .collect::<RusticolResult<Vec<_>>>()?;
                    matrices.insert(item.id.clone(), cells);
                }
                let histories: Vec<String> = record
                    .declaration
                    .color_correlations
                    .iter()
                    .map(|history| history.id.clone())
                    .filter(|id| id != INCLUSIVE)
                    .collect();
                if let Some(missing) = histories.iter().find(|id| !matrices.contains_key(*id)) {
                    return Err(invalid(format!(
                        "colour histories [{missing:?}] have no stored matrix"
                    )));
                }
                (
                    histories,
                    matrices,
                    record.declaration.spin_correlations.clone(),
                )
            }
        };
        let sources = source_records(trees_json)?;
        Ok(Self {
            histories,
            matrices,
            spin_classes,
            sources,
        })
    }

    /// The history ids of the correlated outputs (`born` first).
    pub(crate) fn history_ids(&self) -> Vec<String> {
        std::iter::once(INCLUSIVE.to_owned())
            .chain(self.histories.iter().cloned())
            .collect()
    }

    pub(crate) fn spin_classes(&self) -> &[Vec<usize>] {
        &self.spin_classes
    }
}

/// The tree program's source records (`source_fill.sources`, which
/// `induced_observables.polarisations` reads): from the head of a binary
/// tree program (`rusticol_core::loop_v2::trees::binary`) or the JSON one.
fn source_records(trees_json: &[u8]) -> RusticolResult<Vec<serde_json::Value>> {
    use rusticol_core::loop_v2::trees::binary;
    let head = if binary::is_container(trees_json) {
        binary::read_container(trees_json)?
            .into_iter()
            .find(|section| section.name == "head" && section.encoding == binary::JSON)
            .ok_or_else(|| invalid("a binary tree program without its head"))?
            .payload
    } else {
        trees_json
    };
    let program: serde_json::Value =
        serde_json::from_slice(head).map_err(|e| invalid(format!("the tree program: {e}")))?;
    program
        .pointer("/source_fill/sources")
        .and_then(serde_json::Value::as_array)
        .cloned()
        .ok_or_else(|| invalid("the tree program has no source records"))
}

/// The metric of one output (`Output.metric`).
#[derive(Clone, Debug)]
enum Metric {
    /// G (`INCLUSIVE`).
    Inclusive,
    /// A declared history's matrix.
    History(String),
}

/// One output: its metric and the helicity rows it reads (None: all).
struct Output {
    metric: Metric,
    rows: Option<Row>,
}

/// The carrier forms of a call's outputs (`Forms`): per metric (H, K).
struct Forms<N> {
    found: BTreeMap<String, Form<N>>,
}

impl<N: Clone> Forms<N> {
    fn new<A: Squaring<N = N>>(
        arith: &A,
        induced: &Induced,
        observables: &Observables,
        outputs: &[Output],
    ) -> RusticolResult<Self> {
        let size = induced.size();
        let mut found = BTreeMap::new();
        for output in outputs {
            let (key, h, k) = match &output.metric {
                Metric::Inclusive => (INCLUSIVE.to_owned(), induced.metric().to_vec(), None),
                Metric::History(id) => {
                    let (h, k) = hermitian_parts(&observables.matrices[id], size);
                    (id.clone(), h, k)
                }
            };
            if found.contains_key(&key) {
                continue;
            }
            let h = carrier(arith, &h)?;
            let k = k.map(|k| carrier(arith, &k)).transpose()?;
            found.insert(key, (h, k));
        }
        Ok(Self { found })
    }

    fn of(&self, metric: &Metric) -> &Form<N> {
        match metric {
            Metric::Inclusive => &self.found[INCLUSIVE],
            Metric::History(id) => &self.found[id],
        }
    }
}

impl V2Runtime {
    fn induced_observables(&mut self) -> RusticolResult<&Observables> {
        if self.induced_observables.is_none() {
            let induced = self.induced.clone().expect("a loop-induced process");
            self.induced_observables =
                Some(Observables::new(&induced, &self.process_inputs.trees_json)?);
        }
        Ok(self.induced_observables.as_ref().expect("the observables"))
    }

    /// The correlated outputs' history ids of a loop-induced process.
    pub(super) fn induced_histories(&mut self) -> RusticolResult<Vec<String>> {
        Ok(self.induced_observables()?.history_ids())
    }

    /// The declared spin classes of a loop-induced process (1-based legs).
    pub(super) fn induced_spin_classes(&mut self) -> RusticolResult<Vec<Vec<usize>>> {
        Ok(self.induced_observables()?.spin_classes().to_vec())
    }

    /// The resolved outputs' helicity rows (the plan's, per row its
    /// (0-based leg, helicity) pairs) of a loop-induced process.
    pub(super) fn induced_rows(&self) -> Vec<Vec<(usize, i32)>> {
        self.induced
            .as_ref()
            .expect("a loop-induced process")
            .plan_rows()
    }

    /// The outputs of the column set `kind` of a loop-induced process
    /// (`InducedObservables.native_columns`): `correlated`, `born` and the
    /// declared histories in order (every point's spin sources, if any, in
    /// each); `resolved`, per helicity row its colour-contracted value.
    fn induced_outputs(&mut self, kind: &str) -> RusticolResult<Vec<Output>> {
        match kind {
            "correlated" => {
                let histories = self.induced_observables()?.histories.clone();
                Ok(std::iter::once(Metric::Inclusive)
                    .chain(histories.into_iter().map(Metric::History))
                    .map(|metric| Output { metric, rows: None })
                    .collect())
            }
            "resolved" => Ok(self
                .induced_rows()
                .into_iter()
                .map(|row| Output {
                    metric: Metric::Inclusive,
                    rows: Some(row),
                })
                .collect()),
            _ => Err(invalid(format!("the process stores no {kind} columns"))),
        }
    }

    /// One batch of a loop-induced process's outputs of `kind`
    /// (`InducedColumns.evaluate`): one executor run (every row dressed),
    /// per point its amplitude vectors and every output's form, in the
    /// carrier of `arith`; `spins` per point the spin vectors by 1-based
    /// leg (empty: none). Points fail alone.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn induced_columns<T: Native, A: Lift<T> + Squaring + CarrierPair>(
        &mut self,
        kind: &str,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        spins: &[BTreeMap<usize, [(Rational, Rational); 4]>],
        mode: V2Poles,
        precision: V2Precision,
        zero: &T,
        arith: &A,
    ) -> RusticolResult<Vec<ColumnPoint<A::N>>>
    where
        Complex<T>: Real + EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        if !spins.is_empty() && spins.len() != momenta.len() {
            return Err(invalid("one spin source assignment per point"));
        }
        if kind != "correlated" && !spins.is_empty() {
            return Err(invalid("spin sources apply to correlated outputs"));
        }
        let outputs = self.induced_outputs(kind)?;
        let induced = self.induced.clone().expect("a loop-induced process");
        let classes = self.induced_spin_classes()?;
        for spin in spins.iter().filter(|spin| !spin.is_empty()) {
            let legs: Vec<usize> = spin.keys().copied().collect();
            if !classes.contains(&legs) {
                return Err(invalid(format!(
                    "spin legs {legs:?} are not a declared spin class"
                )));
            }
        }
        let observables = self.induced_observables.take().expect("the observables");
        let found = (|| -> RusticolResult<_> {
            let forms = Forms::new(arith, &induced, &observables, &outputs)?;
            let average = self.record.average;
            let digits = precision.working_digits();
            let plan_rows = induced.plan_rows();
            let sector_index = self.sector_index.clone();
            self.induced_points(
                &induced,
                momenta,
                scales,
                mode,
                precision,
                zero,
                arith,
                |index, vectors, width| {
                    let spin = spins.get(index).filter(|spin| !spin.is_empty());
                    let (amplitudes, rows) = match spin {
                        None => (None, plan_rows.clone()),
                        Some(spin) => {
                            let coefficients = point_coefficients(
                                arith,
                                &observables.sources,
                                &momenta[index],
                                spin,
                                digits,
                            )?;
                            let moved = spin_amplitudes(
                                arith,
                                vectors,
                                &coefficients,
                                width,
                                induced.size(),
                            );
                            let rows = rows_of(&moved);
                            (Some(moved), rows)
                        }
                    };
                    let amplitudes = amplitudes.as_ref().unwrap_or(vectors);
                    outputs
                        .iter()
                        .map(|output| {
                            let rows: Vec<Row> = match &output.rows {
                                None => rows.clone(),
                                Some(kept) => {
                                    rows.iter().filter(|row| *row == kept).cloned().collect()
                                }
                            };
                            let (h, k) = forms.of(&output.metric);
                            let real = induced
                                .square_vectors(arith, amplitudes, &rows, h, width, average)?;
                            let imaginary = k
                                .as_ref()
                                .map(|k| {
                                    induced
                                        .square_vectors(arith, amplitudes, &rows, k, width, average)
                                })
                                .transpose()?;
                            let virtual_values = sector_index
                                .iter()
                                .map(|&s| {
                                    (0..width)
                                        .map(|j| {
                                            C::new(
                                                real[s][j].clone(),
                                                imaginary.as_ref().map_or_else(
                                                    || arith.zero(),
                                                    |y| y[s][j].clone(),
                                                ),
                                            )
                                        })
                                        .collect()
                                })
                                .collect();
                            Ok(ColumnValues {
                                born: Vec::new(),
                                virtual_values,
                                real: imaginary.is_none(),
                            })
                        })
                        .collect::<RusticolResult<Vec<_>>>()
                },
            )
        })();
        self.induced_observables = Some(observables);
        found
    }

    /// The executor run of a loop-induced batch (`plan_export.
    /// NativeExecutor.reduce` of an amplitude-mode plan): the tree program's
    /// inputs and counterterm kets, the executor's bare amplitudes, and per
    /// point its amplitude vectors ([`Induced::vectors_point`]) given to
    /// `per_point` (with the point index and the Laurent width). Points
    /// fail alone.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn induced_points<T: Native, A: Lift<T> + Squaring, R>(
        &mut self,
        induced: &Induced,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
        precision: V2Precision,
        zero: &T,
        arith: &A,
        mut per_point: impl FnMut(usize, &Vectors<A::N>, usize) -> RusticolResult<R>,
    ) -> RusticolResult<Vec<Result<R, Vec<String>>>>
    where
        Complex<T>: Real + EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        let poles = mode.poles();
        let width = if poles { 3 } else { 1 };
        let bits = precision.bits()?;
        let lanes = precision.lanes(momenta.len());
        let incoming = self.incoming(momenta);
        let values = self.parameter_values()?;
        let (bound, inputs) = self.process.fill(
            precision, lanes, &values, scales, &incoming, poles, None, zero,
        )?;
        let kets = inputs.kets;
        let batch = v2_batch(incoming, scales, bound, inputs.block_rows, inputs.born);
        let options = V2Options {
            poles: mode.lanes(),
            contraction: contraction(&self.artifact.metadata.color_contraction),
            ..V2Options::default()
        };
        self.backend(precision)?;
        let backend = self.backends.get_mut(&precision).expect("the backend");
        let results = self
            .process
            .evaluate(lanes, &batch, zero, backend, options)?;
        let mut out = Vec::with_capacity(results.len());
        for (index, result) in results.into_iter().enumerate() {
            if !result.failures.is_empty() {
                out.push(Err(result.failures));
                continue;
            }
            let vectors = induced.vectors_point(
                arith,
                &result.amplitudes,
                kets.as_ref().map(|kets| kets[index].as_slice()),
                width,
                bits,
            )?;
            out.push(Ok(per_point(index, &vectors, width)?));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(text: &str) -> Decimal {
        Decimal::parse(text).unwrap()
    }

    #[test]
    fn hermitian_parts_split_a_directed_matrix() {
        let r = |n: i64, m: i64| Rational::new(n, m);
        // M = [[1, 2+i], [3, 4i]]: H = [[1, (5+i)/2], [(5-i)/2, 0]],
        // K = [[0, (1-i)/2... ]] from (M - M^dag)/2i.
        let m = vec![
            (r(1, 1), r(0, 1)),
            (r(2, 1), r(1, 1)),
            (r(3, 1), r(0, 1)),
            (r(0, 1), r(4, 1)),
        ];
        let (h, k) = hermitian_parts(&m, 2);
        assert_eq!(h[1], (r(5, 2), r(1, 2)));
        assert_eq!(h[2], (r(5, 2), r(-1, 2)));
        let k = k.expect("a directed matrix");
        // H + i K = M cell by cell.
        for (cell, (hk, kk)) in m.iter().zip(h.iter().zip(&k)) {
            assert_eq!(cell.0, &hk.0 - &kk.1);
            assert_eq!(cell.1, &hk.1 + &kk.0);
        }
        let symmetric = vec![
            (r(1, 1), r(0, 1)),
            (r(2, 1), r(1, 1)),
            (r(2, 1), r(-1, 1)),
            (r(3, 1), r(0, 1)),
        ];
        assert!(hermitian_parts(&symmetric, 2).1.is_none());
    }

    #[test]
    fn massless_polarisations_are_transverse_and_unit() {
        let ctx = spin_context(super::V2Precision::Double.working_digits());
        let sources = vec![
            serde_json::json!({"leg_label": 1, "wavefunction_kind": "vector", "dimension": 4,
                "applied_crossing": {"momentum_transform": "identity"},
                "source_helicity": 1, "source_ir": {"mass_parameter": null}}),
            serde_json::json!({"leg_label": 1, "wavefunction_kind": "vector", "dimension": 4,
                "applied_crossing": {"momentum_transform": "identity"},
                "source_helicity": -1, "source_ir": {}}),
        ];
        let p = [13, 3, 4, 12].map(Decimal::integer);
        let waves = polarisations(&ctx, &sources, 1, &p).unwrap();
        assert_eq!(waves.iter().map(|(h, _)| *h).collect::<Vec<_>>(), [1, -1]);
        // The coefficients of a polarisation vector on the basis are
        // -w.conj(w') = delta (w.conj(w) = -1).
        for (h, wave) in &waves {
            let exact = wave.clone();
            let found = spin_coefficients(&ctx, &exact, &waves);
            for (g, (re, im)) in found {
                let expected = if g == *h { 1.0 } else { 0.0 };
                assert!((re.to_f64() - expected).abs() < 1e-30, "{h} {g} {re:?}");
                assert!(im.to_f64().abs() < 1e-30);
            }
        }
        assert!(polarisations(&ctx, &sources, 2, &p).is_err());
        let _ = d("0");
    }

    #[test]
    fn decimal_square_roots_round_half_even_as_python() {
        let ctx = DecimalContext::new(5);
        // Python: localcontext(prec=5): Decimal(2).sqrt() = 1.4142,
        // Decimal("0.5").sqrt() = 0.70711, Decimal(144).sqrt() = 12,
        // Decimal("1E-7").sqrt() = 0.00031623.
        assert_eq!(ctx.sqrt(&d("2")).text(), d("1.4142").text());
        assert_eq!(ctx.sqrt(&d("0.5")).text(), d("0.70711").text());
        assert!(ctx.sqrt(&d("144")) == d("12"));
        assert!(ctx.sqrt(&d("1E-7")) == d("0.00031623"));
        let wide = DecimalContext::new(44);
        assert_eq!(
            wide.sqrt(&d("0.5")).to_f64(),
            std::f64::consts::FRAC_1_SQRT_2
        );
    }
}
