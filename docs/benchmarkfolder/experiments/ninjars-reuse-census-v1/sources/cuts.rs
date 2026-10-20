//! Batched cut geometry generated from exact Symbolica expressions.
//!
//! The longitudinal solutions follow Ninja's `basis.cc` and `cuts_utils.cc`.
//! Unlike the real-only reference basis, the discriminant is evaluated over
//! the working complex carrier. The stable gamma branch uses `e1.e2=gamma/2`,
//! avoiding the reference's removable `s/(1+r1*r2)` singularity at orthogonal
//! spatial vectors. Null invariant limits use the same expressions, retaining
//! Dual derivatives transverse to a null shell.
//!
//! FeynKit owns all momentum and Weyl operations. Symbolica prepares every
//! arithmetic expression, its evaluator processes active lanes instruction by
//! instruction, and Dualizer transforms the same expressions for derivatives.
//! Only chart selection, conditioning checks, and lane packing are handwritten.

use ninjars::{
    BatchShape, BubbleCut, BubbleCutRequest, BubbleSampleRequest, CutBasis, CutFailure, CutOutcome,
    CutProvider, CutSampler, Error, FourMomentum, Result, TadpoleCut, TadpoleCutRequest,
    TriangleCut, TriangleCutRequest, TriangleSampleRequest,
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        dual::{DualNumber, HyperDual},
        float::{Complex, FloatLike, Real, SingleFloat},
        rational::Rational,
    },
    evaluate::{BatchExpressionEvaluator, Dualizer, EvaluationDomain, ExpressionEvaluator},
    symbol,
};

use crate::{
    PolynomialNumerator, TadpoleGeometryProbe, positional::positional_evaluator,
    tadpole_plans::tadpole_reference_bases,
};

/// Complex cut geometry at the supplied coefficient precision.
pub struct SymbolicaCuts<T> {
    engine: CutEngine<T>,
}

/// The same cut geometry with static or runtime Symbolica Dual components.
/// Numerical chart selection uses primal values, while each selected expression
/// carries every higher/mixed component through Symbolica's Dualizer.
pub struct SymbolicaDualCuts<T, D> {
    engine: CutEngine<T>,
    prototype: D,
    shape: Vec<Vec<usize>>,
}

struct Kernel<T> {
    evaluator: BatchExpressionEvaluator<Complex<T>>,
    inputs: Vec<Complex<T>>,
    outputs: Vec<Complex<T>>,
    columns: usize,
    width: usize,
    zero: Complex<T>,
}

impl<T: Real + SingleFloat> Kernel<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn new(
        expressions: &[Atom],
        parameters: &[Atom],
        prototype: &T,
        shape: Option<&[Vec<usize>]>,
    ) -> Result<Self> {
        let exact: ExpressionEvaluator<Complex<Rational>> =
            positional_evaluator(expressions, parameters, &[], true)?;
        let width = shape.map_or(1, |s| s.len());
        let exact = if let Some(shape) = shape {
            exact
                .vectorize(&Dualizer::new(
                    HyperDual::<Complex<Rational>>::new(shape.to_vec()),
                    Vec::new(),
                ))
                .map_err(|e| Error::Symbolic(e.to_string()))?
        } else {
            exact
        };
        Ok(Self {
            evaluator: PolynomialNumerator::map_evaluator(exact, prototype),
            inputs: Vec::new(),
            outputs: Vec::new(),
            columns: expressions.len(),
            width,
            zero: Complex::new(prototype.zero(), prototype.zero()),
        })
    }
    fn evaluate(&mut self, rows: usize) -> Result<()> {
        self.outputs
            .resize_with(rows * self.columns * self.width, || self.zero.clone());
        self.evaluator
            .evaluate_batch(&self.inputs, &mut self.outputs, rows)
            .map_err(|e| Error::Symbolic(e.to_string()))
    }
    fn component(&self, row: usize, column: usize) -> &[Complex<T>] {
        let start = (row * self.columns + column) * self.width;
        &self.outputs[start..start + self.width]
    }
}

struct CutEngine<T> {
    prototype: T,
    width: usize,
    tolerance: T,
    probes: [Kernel<T>; 2],
    seeds: [Kernel<T>; 2],
    finishes: [Vec<Kernel<T>>; 2],
    tadpole_probe: Kernel<T>,
    tadpole_finishes: Vec<Kernel<T>>,
    tadpole_probe_rows: Vec<usize>,
    tadpole_scores: Vec<[T; 6]>,
    bubble_complement_probe: Kernel<T>,
    bubble_complement_starts: Vec<usize>,
    samplers: [Kernel<T>; 2],
    active: Vec<usize>,
    seeded: Vec<usize>,
    groups: [Vec<usize>; 16],
    failures: Vec<Option<CutFailure>>,
    values: Vec<Vec<Complex<T>>>,
}

fn variables(n: usize) -> Vec<Atom> {
    (0..n)
        .map(|i| Atom::var(symbol!(format!("ninjars_cut_kernel::p{i}"))))
        .collect()
}
fn vector(p: &[Atom], offset: usize) -> FourMomentum<Atom> {
    FourMomentum::from_args(
        p[offset].clone(),
        p[offset + 1].clone(),
        p[offset + 2].clone(),
        p[offset + 3].clone(),
    )
}
fn append_vector(out: &mut Vec<Atom>, v: &FourMomentum<Atom>) {
    out.extend(v.components().into_iter().cloned());
}

/// The bubble's auxiliary null references, in probe column order.
///
/// As in Ninja's one-momentum Basis constructor, keep every spatial component
/// nonzero. Axis references put the two transverse vectors outside a planar
/// scattering geometry. Exact Weyl pairs give eight references
/// (1, ±2/3, ±2/3, ±1/3), without a square-root branch or binary64 constants.
/// Being rational, any one of them can lie exactly in the plane of the cut
/// momentum and an uncut shift difference at rational kinematics (Ninja's
/// irrational reference avoids this only generically), so the provider scores
/// all of them against the uncut lines of each request.
fn bubble_references() -> Vec<FourMomentum<Atom>> {
    let mut references = Vec::new();
    for swap in [false, true] {
        for x in [-1, 1] {
            for y in [-1, 1] {
                let mut angle = [Atom::num(1), Atom::num(x) + Atom::i() * y];
                let mut square = [Atom::num(1), Atom::num(x) - Atom::i() * y];
                if swap {
                    angle.swap(0, 1);
                    square.swap(0, 1);
                }
                references
                    .push(FourMomentum::from_weyl_spinors(&angle, &square) * (Atom::num(2) / 3));
            }
        }
    }
    references
}

fn tadpole_kernels<T: Real + SingleFloat>(
    prototype: &T,
    shape: Option<&[Vec<usize>]>,
) -> Result<(Kernel<T>, Vec<Kernel<T>>)>
where
    Complex<T>: Real + EvaluationDomain,
{
    let p = variables(8);
    let probe = TadpoleGeometryProbe::new();
    let mut finishes = Vec::new();
    // Exact rational Weyl components give six distinct null bases without
    // introducing constant square-root branches or f64 parameter round trips.
    // Every component modulus is below 32, used in the conditioning check.
    for [e1, e2, e3, e4] in tadpole_reference_bases() {
        let g = e1.dot(&e2);
        let mut output = Vec::new();
        for e in [&e1, &e2, &e3, &e4] {
            append_vector(&mut output, e);
        }
        output.push(g.clone());
        append_vector(&mut output, &(e2 * (Atom::num(1) / (&g * 2))));
        append_vector(&mut output, &(e4 * (-Atom::num(1) / (&g * 2))));
        append_vector(&mut output, &(-vector(&p, 0)));
        output.push(p[4].clone());
        finishes.push(Kernel::new(&output, &p[..5], prototype, shape)?);
    }
    Ok((
        Kernel::new(
            &probe.exact.expressions,
            &probe.exact.parameters,
            prototype,
            shape,
        )?,
        finishes,
    ))
}

impl<T: Real + SingleFloat> CutEngine<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn new(prototype: T, shape: Option<&[Vec<usize>]>) -> Result<Self> {
        let width = shape.map_or(1, |s| s.len());
        let p = variables(12);
        let k1 = vector(&p, 0) - vector(&p, 8);
        let k2 = vector(&p, 4) - vector(&p, 0);
        let a = k1.dot(&k1);
        let b = k2.dot(&k2);
        let s = k1.dot(&k2);
        let delta = s.pow(2) - &a * &b;
        let root = delta.sqrt();
        let mut expressions = Vec::new();
        append_vector(&mut expressions, &k1);
        append_vector(&mut expressions, &k2);
        expressions.extend([&s + &root, &s - &root, delta]);
        let triangle_probe = Kernel::new(&expressions, &p, &prototype, shape)?;
        let p = variables(8);
        let k = vector(&p, 0) - vector(&p, 4);
        let mut expressions = Vec::new();
        append_vector(&mut expressions, &k);
        let references = bubble_references();
        for reference in &references {
            expressions.push(k.dot(reference));
            append_vector(&mut expressions, reference);
        }
        let bubble_probe = Kernel::new(&expressions, &p, &prototype, shape)?;
        // Uncut-line probe: [V_first, V_second, V_uncut] gives v = V_uncut -
        // V_first and, per reference r, the Gram determinant of (k, r, v).
        // With e1.e2 = k.r it equals 2 (k.r) (e3.v)(e4.v): it vanishes exactly
        // when the uncut denominator loses its leading Laurent coefficient in
        // either orientation, independently of the spinor chart normalization.
        let p = variables(12);
        let k = vector(&p, 0) - vector(&p, 4);
        let v = vector(&p, 8) - vector(&p, 0);
        let mut expressions = Vec::new();
        append_vector(&mut expressions, &v);
        for reference in &references {
            let rows = [&k, reference, &v];
            let gram: Vec<Vec<Atom>> = rows
                .iter()
                .map(|a| rows.iter().map(|b| a.dot(b)).collect())
                .collect();
            let minor = |i: usize, j: usize, k: usize, l: usize| {
                &gram[i][k] * &gram[j][l] - &gram[i][l] * &gram[j][k]
            };
            expressions.push(
                &gram[0][0] * minor(1, 2, 1, 2) - &gram[0][1] * minor(1, 2, 0, 2)
                    + &gram[0][2] * minor(1, 2, 0, 1),
            );
        }
        let bubble_complement_probe = Kernel::new(&expressions, &p, &prototype, shape)?;

        let mut seeds = Vec::new();
        for mode in 0..2 {
            let p = variables(if mode == 0 { 9 } else { 8 });
            let k1 = vector(&p, 0);
            let k2 = vector(&p, 4);
            let a = k1.dot(&k1);
            let b = k2.dot(&k2);
            let (e1, e2, g, r1, r2, d) = if mode == 0 {
                let g = &p[8] / 2;
                let r1 = &a / &p[8];
                let r2 = &b / &p[8];
                let d = Atom::num(1) - &r1 * &r2;
                (
                    (k1 - k2.clone() * r1.clone()) * (Atom::num(1) / &d),
                    (k2 - vector(&p, 0) * r2.clone()) * (Atom::num(1) / &d),
                    g,
                    r1,
                    r2,
                    d,
                )
            } else {
                let g = k1.dot(&k2);
                let r1 = &a / (&g * 2);
                (
                    k1 - k2.clone() * r1.clone(),
                    k2,
                    g,
                    r1,
                    Atom::Zero,
                    Atom::num(1),
                )
            };
            let mut expressions = Vec::new();
            append_vector(&mut expressions, &e1);
            append_vector(&mut expressions, &e2);
            expressions.extend([g, r1, r2, a, b, d]);
            expressions.extend(e1.weyl_matrix().into_iter().flatten());
            expressions.extend(e2.weyl_matrix().into_iter().flatten());
            seeds.push(Kernel::new(&expressions, &p, &prototype, shape)?);
        }
        let mut finishes = [Vec::new(), Vec::new()];
        for (mode, kernels) in finishes.iter_mut().enumerate() {
            let p = variables(21);
            let e1 = vector(&p, 0);
            let e2 = vector(&p, 4);
            let g = &p[8];
            let r1 = &p[9];
            let r2 = &p[10];
            let k1sq = &p[11];
            let k2sq = &p[12];
            let d = &p[13];
            let anchor = vector(&p, 14);
            let masses = &p[18..21];
            for chart in 0..16 {
                let left = chart / 4;
                let right = chart % 4;
                let [angle1, square1] = e1
                    .null_weyl_spinors([left / 2, left % 2])
                    .map_err(|e| Error::Symbolic(e.to_string()))?;
                let [angle2, square2] = e2
                    .null_weyl_spinors([right / 2, right % 2])
                    .map_err(|e| Error::Symbolic(e.to_string()))?;
                let e3 = FourMomentum::from_weyl_spinors(&angle1, &square2);
                let e4 = FourMomentum::from_weyl_spinors(&angle2, &square1);
                let effective = e4.clone() * (-Atom::num(1) / (g * 2));
                let mut expressions = Vec::new();
                append_vector(&mut expressions, &e3);
                append_vector(&mut expressions, &e4);
                append_vector(&mut expressions, &effective);
                if mode == 0 {
                    let a1 = (&masses[0] - &masses[2] + k1sq) / (g * 2);
                    let a2 = (&masses[1] - &masses[0] - k2sq) / (g * 2);
                    let x1 = (&a2 - r2 * &a1) / d;
                    let x2 = (&a1 - r1 * &a2) / d;
                    let a = e1.clone() * x1.clone() + e2.clone() * x2.clone() - anchor.clone();
                    append_vector(&mut expressions, &a);
                    expressions.push(&masses[0] - g * 2 * x1 * x2);
                } else {
                    let b0 = (&masses[0] - &masses[1] + k1sq) / (g * 2);
                    let b1 = -r1;
                    append_vector(
                        &mut expressions,
                        &(e2.clone() * b0.clone() - anchor.clone()),
                    );
                    append_vector(&mut expressions, &(e1.clone() + e2.clone() * b1.clone()));
                    expressions.extend([masses[0].clone(), -g * 2 * b0, -g * 2 * b1]);
                }
                kernels.push(Kernel::new(&expressions, &p, &prototype, shape)?);
            }
        }
        // A conditioning threshold, separate from the fixed comparison tolerance.
        // Construct at working precision instead of passing through f64.
        let tolerance = prototype
            .from_i64(2)
            .pow(u64::from(prototype.get_precision() / 2))
            .inv();
        let (tadpole_probe, tadpole_finishes) = tadpole_kernels(&prototype, shape)?;
        let p = variables(15);
        let sample = vector(&p, 0)
            + vector(&p, 4) * p[13].clone()
            + vector(&p, 8) * ((&p[12] + &p[14]) / &p[13]);
        let triangle_sampler = Kernel::new(
            &sample.components().into_iter().cloned().collect::<Vec<_>>(),
            &p,
            &prototype,
            shape,
        )?;
        let p = variables(22);
        let sample = vector(&p, 0)
            + vector(&p, 4) * p[19].clone()
            + vector(&p, 8) * p[20].clone()
            + vector(&p, 12)
                * ((&p[16] + &p[17] * &p[19] + &p[18] * p[19].pow(2) + &p[21]) / &p[20]);
        let bubble_sampler = Kernel::new(
            &sample.components().into_iter().cloned().collect::<Vec<_>>(),
            &p,
            &prototype,
            shape,
        )?;
        Ok(Self {
            prototype,
            width,
            tolerance,
            probes: [triangle_probe, bubble_probe],
            seeds: seeds.try_into().ok().unwrap(),
            finishes,
            tadpole_probe,
            tadpole_finishes,
            tadpole_probe_rows: Vec::new(),
            tadpole_scores: Vec::new(),
            bubble_complement_probe,
            bubble_complement_starts: Vec::new(),
            samplers: [triangle_sampler, bubble_sampler],
            active: Vec::new(),
            seeded: Vec::new(),
            groups: std::array::from_fn(|_| Vec::new()),
            failures: Vec::new(),
            values: Vec::new(),
        })
    }

    fn magnitude(value: &Complex<T>) -> T {
        value.norm().re
    }
    fn greater(a: &T, b: &T) -> bool {
        a.real_cmp(b) == Some(std::cmp::Ordering::Greater)
    }
    fn too_small(&self, value: &Complex<T>, scale: &T) -> bool {
        !Self::greater(&Self::magnitude(value), &(self.tolerance.clone() * scale))
    }

    /// Choose the bubble reference for one probed row: the probe column of
    /// its k.r, or None when no reference is well conditioned.
    ///
    /// Each uncut difference v has, for reference r, the scale-free ratio
    /// |G(k,r,v)|/((k.r)^2 |v|^2) = 2|(e3.v)(e4.v)/(e3.e4)|/|v|^2 (|.| is
    /// the largest component modulus). It vanishes when v lies in the plane
    /// of k and r, where that uncut denominator's leading Laurent coefficient
    /// is exactly zero and only round-off would otherwise be divided by. A
    /// reference scores the minimum of (|k.r|/|k|)^2 and these ratios, and
    /// the highest score wins (the largest |k.r| without uncut lines), the
    /// lowest index among scores tied within the tolerance. As
    /// before, |k.r| must exceed the tolerance relative to |k|. The plane
    /// ratios, quadratic in the transverse components like the triangle Gram
    /// determinant, must exceed the tolerance itself; only an uncut line
    /// (nearly) along k is in the plane of every reference.
    fn bubble_reference(&self, probe_row: usize, scale: &T) -> Option<usize> {
        let probe = &self.probes[1];
        let uncut = &self.bubble_complement_probe;
        let zero = self.prototype.zero();
        let products: [T; 8] = std::array::from_fn(|candidate| {
            Self::magnitude(&probe.component(probe_row, 4 + 5 * candidate)[0])
        });
        let mut planes: [Option<T>; 8] = std::array::from_fn(|_| None);
        let lines =
            self.bubble_complement_starts[probe_row]..self.bubble_complement_starts[probe_row + 1];
        for line in lines {
            let mut norm = zero.clone();
            for i in 0..4 {
                let value = Self::magnitude(&uncut.component(line, i)[0]);
                if Self::greater(&value, &norm) {
                    norm = value;
                }
            }
            if !norm.is_finite() || norm.is_fully_zero() {
                return None;
            }
            for (candidate, plane) in planes.iter_mut().enumerate() {
                let gram = Self::magnitude(&uncut.component(line, 4 + candidate)[0]);
                let product = &products[candidate];
                let mut ratio = gram / (product.clone() * product * &norm * &norm);
                if !ratio.is_finite() {
                    ratio = zero.clone();
                }
                if plane
                    .as_ref()
                    .is_none_or(|plane| Self::greater(plane, &ratio))
                {
                    *plane = Some(ratio);
                }
            }
        }
        let score = |candidate: usize| {
            let ratio = products[candidate].clone() / scale;
            let leading = ratio.clone() * ratio;
            match &planes[candidate] {
                Some(plane) if Self::greater(&leading, plane) => plane.clone(),
                _ => leading,
            }
        };
        // Symmetric rational kinematics often tie several references exactly
        // (here the eight differ only by signs and a swap). The reference fixes
        // the bubble's e2 direction and hence how the residue splits between
        // B1/B11, the explicit mu^2 term, and the tadpoles' A0 subtraction:
        // only their sum is reference independent, so the reported
        // cut-constructible/rational split must not follow round-off. Like
        // Ninja's fixed reference, break ties deterministically: take the
        // lowest-index candidate within the conditioning tolerance of the best.
        let scores: [T; 8] = std::array::from_fn(score);
        let mut maximum = scores[0].clone();
        for candidate_score in &scores[1..] {
            if Self::greater(candidate_score, &maximum) {
                maximum = candidate_score.clone();
            }
        }
        let threshold = maximum * (self.prototype.one() - &self.tolerance);
        let best = (0..8)
            .find(|&candidate| !Self::greater(&threshold, &scores[candidate]))
            .unwrap_or(0);
        let column = 4 + 5 * best;
        (!self.too_small(&probe.component(probe_row, column)[0], scale)
            && planes[best]
                .as_ref()
                .is_none_or(|plane| Self::greater(plane, &self.tolerance)))
        .then_some(column)
    }

    fn run_tadpole<D>(
        &mut self,
        requests: &[TadpoleCutRequest<'_, D>],
        components: impl for<'v> Fn(&'v D) -> &'v [Complex<T>],
        valid: impl Fn(&D) -> bool,
    ) -> Result<()> {
        self.failures.clear();
        self.failures.resize(requests.len(), None);
        if self.values.len() < requests.len() {
            self.values.resize_with(requests.len(), Vec::new);
        }
        for row in self.values.iter_mut().take(requests.len()) {
            row.clear();
        }
        for group in &mut self.groups {
            group.clear();
        }
        self.tadpole_scores.clear();
        self.tadpole_scores.resize_with(requests.len(), || {
            std::array::from_fn(|_| self.prototype.one())
        });
        self.tadpole_probe.inputs.clear();
        self.tadpole_probe_rows.clear();
        for (row, request) in requests.iter().enumerate() {
            for value in request
                .shift
                .components()
                .into_iter()
                .chain([request.mass_squared])
                .chain(
                    request
                        .complementary_shifts
                        .iter()
                        .flat_map(FourMomentum::components),
                )
            {
                if !valid(value) || components(value).len() != self.width {
                    self.failures[row] = Some(CutFailure::DerivativeShape);
                    break;
                }
                if components(value).iter().any(|value| !value.is_finite()) {
                    self.failures[row] = Some(CutFailure::NonFiniteInput);
                    break;
                }
            }
            if self.failures[row].is_some() {
                continue;
            }
            for shift in request.complementary_shifts {
                self.tadpole_probe_rows.push(row);
                for value in request
                    .shift
                    .components()
                    .into_iter()
                    .chain(shift.components())
                {
                    self.tadpole_probe
                        .inputs
                        .extend_from_slice(components(value));
                }
            }
        }
        self.tadpole_probe.evaluate(self.tadpole_probe_rows.len())?;
        for (probe_row, &row) in self.tadpole_probe_rows.iter().enumerate() {
            let mut scale = self.prototype.zero();
            for column in 0..4 {
                let value = Self::magnitude(&self.tadpole_probe.component(probe_row, column)[0]);
                if Self::greater(&value, &scale) {
                    scale = value;
                }
            }
            if scale.is_fully_zero() || !scale.is_finite() {
                self.failures[row] = Some(CutFailure::DegenerateReference);
                continue;
            }
            for candidate in 0..6 {
                for direction in 0..4 {
                    let projection = &self
                        .tadpole_probe
                        .component(probe_row, 4 + 4 * candidate + direction)[0];
                    let score = Self::magnitude(projection) / &scale;
                    if !score.is_finite() {
                        self.failures[row] = Some(CutFailure::NonFiniteResult);
                    } else if Self::greater(&self.tadpole_scores[row][candidate], &score) {
                        self.tadpole_scores[row][candidate] = score;
                    }
                }
            }
        }
        for row in 0..requests.len() {
            if self.failures[row].is_some() {
                continue;
            }
            let mut selected = 0;
            for candidate in 1..6 {
                if Self::greater(
                    &self.tadpole_scores[row][candidate],
                    &self.tadpole_scores[row][selected],
                ) {
                    selected = candidate;
                }
            }
            if !Self::greater(
                &self.tadpole_scores[row][selected],
                &(self.tolerance.clone() * self.prototype.from_i64(32)),
            ) {
                self.failures[row] = Some(CutFailure::DegenerateReference);
            } else {
                self.groups[selected].push(row);
            }
        }
        for candidate in 0..6 {
            let group = &self.groups[candidate];
            if group.is_empty() {
                continue;
            }
            let kernel = &mut self.tadpole_finishes[candidate];
            kernel.inputs.clear();
            for &row in group {
                let request = &requests[row];
                for value in request
                    .shift
                    .components()
                    .into_iter()
                    .chain([request.mass_squared])
                {
                    kernel.inputs.extend_from_slice(components(value));
                }
            }
            kernel.evaluate(group.len())?;
            for (index, &row) in group.iter().enumerate() {
                if (0..kernel.columns).any(|column| {
                    kernel
                        .component(index, column)
                        .iter()
                        .any(|value| !value.is_finite())
                }) {
                    self.failures[row] = Some(CutFailure::NonFiniteResult);
                    continue;
                }
                for column in 0..kernel.columns {
                    self.values[row].extend_from_slice(kernel.component(index, column));
                }
            }
        }
        Ok(())
    }

    fn sample<'a, D: 'a>(
        &mut self,
        rows: usize,
        mode: usize,
        argument: impl Fn(usize, usize) -> &'a D,
        components: impl for<'v> Fn(&'v D) -> &'v [Complex<T>],
        valid: impl Fn(&D) -> bool,
    ) -> Result<()> {
        self.failures.clear();
        self.failures.resize(rows, None);
        if self.values.len() < rows {
            self.values.resize_with(rows, Vec::new);
        }
        for row in self.values.iter_mut().take(rows) {
            row.clear();
        }
        self.active.clear();
        let kernel = &mut self.samplers[mode];
        kernel.inputs.clear();
        let columns = if mode == 0 { 15 } else { 22 };
        for row in 0..rows {
            for column in 0..columns {
                let value = argument(row, column);
                if !valid(value) || components(value).len() != self.width {
                    self.failures[row] = Some(CutFailure::DerivativeShape);
                    break;
                }
                if components(value).iter().any(|v| !v.is_finite()) {
                    self.failures[row] = Some(CutFailure::NonFiniteInput);
                    break;
                }
            }
            if self.failures[row].is_some() {
                continue;
            }
            if components(argument(row, if mode == 0 { 13 } else { 20 }))[0].is_fully_zero() {
                self.failures[row] = Some(CutFailure::SingularParameter);
                continue;
            }
            self.active.push(row);
            for column in 0..columns {
                kernel
                    .inputs
                    .extend_from_slice(components(argument(row, column)));
            }
        }
        kernel.evaluate(self.active.len())?;
        for (index, &row) in self.active.iter().enumerate() {
            if (0..4).any(|column| {
                kernel
                    .component(index, column)
                    .iter()
                    .any(|value| !value.is_finite())
            }) {
                self.failures[row] = Some(CutFailure::NonFiniteResult);
                continue;
            }
            for column in 0..4 {
                self.values[row].extend_from_slice(kernel.component(index, column));
            }
        }
        Ok(())
    }

    /// `complement` lists a bubble row's uncut shifts (ignored for triangles).
    #[allow(clippy::too_many_arguments)]
    fn run<'a, D: 'a>(
        &mut self,
        rows: usize,
        mode: usize,
        shift: impl Fn(usize, usize) -> &'a FourMomentum<D>,
        mass: impl Fn(usize, usize) -> &'a D,
        complement: impl Fn(usize) -> &'a [FourMomentum<D>],
        components: impl for<'v> Fn(&'v D) -> &'v [Complex<T>],
        valid: impl Fn(&D) -> bool,
    ) -> Result<()> {
        let shifts = if mode == 0 { 3 } else { 2 };
        self.failures.clear();
        self.failures.resize(rows, None);
        if self.values.len() < rows {
            self.values.resize_with(rows, Vec::new);
        }
        for values in self.values.iter_mut().take(rows) {
            values.clear();
        }
        self.active.clear();
        self.seeded.clear();
        for group in &mut self.groups {
            group.clear();
        }
        self.probes[mode].inputs.clear();
        self.seeds[mode].inputs.clear();
        self.bubble_complement_probe.inputs.clear();
        self.bubble_complement_starts.clear();
        let mut complement_rows = 0;
        for row in 0..rows {
            let mut failure = None;
            let uncut: &[FourMomentum<D>] = if mode == 1 { complement(row) } else { &[] };
            let values = (0..shifts)
                .flat_map(|i| shift(row, i).components().into_iter().chain([mass(row, i)]))
                .chain(uncut.iter().flat_map(FourMomentum::components));
            for value in values {
                if !valid(value) || components(value).len() != self.width {
                    failure = Some(CutFailure::DerivativeShape);
                    break;
                }
                if components(value).iter().any(|v| !v.is_finite()) {
                    failure = Some(CutFailure::NonFiniteInput);
                    break;
                }
            }
            if let Some(failure) = failure {
                self.failures[row] = Some(failure);
                continue;
            }
            self.active.push(row);
            for i in 0..shifts {
                for value in shift(row, i).components() {
                    self.probes[mode]
                        .inputs
                        .extend_from_slice(components(value));
                }
            }
            self.bubble_complement_starts.push(complement_rows);
            for line in uncut {
                complement_rows += 1;
                for value in shift(row, 0)
                    .components()
                    .into_iter()
                    .chain(shift(row, 1).components())
                    .chain(line.components())
                {
                    self.bubble_complement_probe
                        .inputs
                        .extend_from_slice(components(value));
                }
            }
        }
        self.bubble_complement_starts.push(complement_rows);
        self.probes[mode].evaluate(self.active.len())?;
        if mode == 1 {
            self.bubble_complement_probe.evaluate(complement_rows)?;
        }
        for (probe_row, &row) in self.active.iter().enumerate() {
            let probe = &self.probes[mode];
            let columns = if mode == 0 { 8 } else { 4 };
            let mut scale = self.prototype.zero();
            for i in 0..columns {
                let value = Self::magnitude(&probe.component(probe_row, i)[0]);
                if Self::greater(&value, &scale) {
                    scale = value;
                }
            }
            if !scale.is_finite() || scale.is_fully_zero() {
                self.failures[row] = Some(CutFailure::DegenerateReference);
                continue;
            }
            let selected = if mode == 0 {
                if probe
                    .component(probe_row, 10)
                    .iter()
                    .any(|v| !v.is_finite())
                    || self.too_small(&probe.component(probe_row, 10)[0], &scale.pow(4))
                {
                    self.failures[row] = Some(CutFailure::DegenerateGram);
                    continue;
                }
                if Self::greater(
                    &Self::magnitude(&probe.component(probe_row, 8)[0]),
                    &Self::magnitude(&probe.component(probe_row, 9)[0]),
                ) {
                    8
                } else {
                    9
                }
            } else {
                match self.bubble_reference(probe_row, &scale) {
                    Some(column) => column,
                    None => {
                        self.failures[row] = Some(CutFailure::DegenerateReference);
                        continue;
                    }
                }
            };
            self.seeded.push(row);
            for i in 0..columns {
                self.seeds[mode]
                    .inputs
                    .extend_from_slice(probe.component(probe_row, i));
            }
            if mode == 0 {
                self.seeds[mode]
                    .inputs
                    .extend_from_slice(probe.component(probe_row, selected));
            } else {
                for i in selected + 1..selected + 5 {
                    self.seeds[mode]
                        .inputs
                        .extend_from_slice(probe.component(probe_row, i));
                }
            }
        }
        self.seeds[mode].evaluate(self.seeded.len())?;
        for (seed_row, &row) in self.seeded.iter().enumerate() {
            let seed = &self.seeds[mode];
            if (0..seed.columns).any(|i| seed.component(seed_row, i).iter().any(|v| !v.is_finite()))
            {
                self.failures[row] = Some(CutFailure::NonFiniteResult);
                continue;
            }
            if self.too_small(&seed.component(seed_row, 13)[0], &self.prototype.one()) {
                self.failures[row] = Some(CutFailure::DegenerateGram);
                continue;
            }
            let mut chart = 0;
            for start in [14, 18] {
                let mut best = start;
                for i in start + 1..start + 4 {
                    if Self::greater(
                        &Self::magnitude(&seed.component(seed_row, i)[0]),
                        &Self::magnitude(&seed.component(seed_row, best)[0]),
                    ) {
                        best = i;
                    }
                }
                chart = chart * 4 + best - start;
            }
            self.groups[chart].push(seed_row);
        }
        for chart in 0..16 {
            let group = &self.groups[chart];
            if group.is_empty() {
                continue;
            }
            let kernel = &mut self.finishes[mode][chart];
            kernel.inputs.clear();
            for &seed_row in group {
                let row = self.seeded[seed_row];
                for i in 0..14 {
                    kernel
                        .inputs
                        .extend_from_slice(self.seeds[mode].component(seed_row, i));
                }
                for value in shift(row, 0).components() {
                    kernel.inputs.extend_from_slice(components(value));
                }
                for i in 0..3 {
                    if i < shifts {
                        kernel.inputs.extend_from_slice(components(mass(row, i)));
                    } else {
                        kernel
                            .inputs
                            .extend((0..self.width).map(|_| kernel.zero.clone()));
                    }
                }
            }
            kernel.evaluate(group.len())?;
            for (index, &seed_row) in group.iter().enumerate() {
                let row = self.seeded[seed_row];
                if (0..kernel.columns)
                    .any(|i| kernel.component(index, i).iter().any(|v| !v.is_finite()))
                {
                    self.failures[row] = Some(CutFailure::NonFiniteResult);
                    continue;
                }
                for i in 0..14 {
                    self.values[row].extend_from_slice(self.seeds[mode].component(seed_row, i));
                }
                for i in 0..kernel.columns {
                    self.values[row].extend_from_slice(kernel.component(index, i));
                }
            }
        }
        Ok(())
    }

    fn basis<D>(&self, row: usize, make: &impl Fn(&[Complex<T>]) -> D) -> CutBasis<D> {
        CutBasis {
            e1: self.momentum(row, 0, make),
            e2: self.momentum(row, 4, make),
            e3: self.momentum(row, 14, make),
            e4: self.momentum(row, 18, make),
            e12: make(self.value(row, 8)),
        }
    }
    fn tadpole<D>(
        &self,
        row: usize,
        make: &impl Fn(&[Complex<T>]) -> D,
    ) -> CutOutcome<TadpoleCut<D>> {
        if let Some(failure) = self.failures[row] {
            return CutOutcome::Unstable(failure);
        }
        CutOutcome::Stable(TadpoleCut {
            basis: CutBasis {
                e1: self.momentum(row, 0, make),
                e2: self.momentum(row, 4, make),
                e3: self.momentum(row, 8, make),
                e4: self.momentum(row, 12, make),
                e12: make(self.value(row, 16)),
            },
            a: self.momentum(row, 25, make),
            beta: make(self.value(row, 29)),
            e2_effective: self.momentum(row, 17, make),
            e4_effective: self.momentum(row, 21, make),
        })
    }
    fn value(&self, row: usize, column: usize) -> &[Complex<T>] {
        &self.values[row][column * self.width..(column + 1) * self.width]
    }
    fn momentum<D>(
        &self,
        row: usize,
        column: usize,
        make: &impl Fn(&[Complex<T>]) -> D,
    ) -> FourMomentum<D> {
        FourMomentum::from_args(
            make(self.value(row, column)),
            make(self.value(row, column + 1)),
            make(self.value(row, column + 2)),
            make(self.value(row, column + 3)),
        )
    }
    fn sample_result<D>(
        &self,
        row: usize,
        make: &impl Fn(&[Complex<T>]) -> D,
    ) -> CutOutcome<FourMomentum<D>> {
        if let Some(failure) = self.failures[row] {
            CutOutcome::Unstable(failure)
        } else {
            CutOutcome::Stable(self.momentum(row, 0, make))
        }
    }
    fn triangle<D>(
        &self,
        row: usize,
        make: &impl Fn(&[Complex<T>]) -> D,
    ) -> CutOutcome<TriangleCut<D>> {
        if let Some(failure) = self.failures[row] {
            return CutOutcome::Unstable(failure);
        }
        CutOutcome::Stable(TriangleCut {
            basis: self.basis(row, make),
            e4_effective: self.momentum(row, 22, make),
            a: self.momentum(row, 26, make),
            beta: make(self.value(row, 30)),
        })
    }
    fn bubble<D>(
        &self,
        row: usize,
        make: &impl Fn(&[Complex<T>]) -> D,
    ) -> CutOutcome<BubbleCut<D>> {
        if let Some(failure) = self.failures[row] {
            return CutOutcome::Unstable(failure);
        }
        CutOutcome::Stable(BubbleCut {
            basis: self.basis(row, make),
            e4_effective: self.momentum(row, 22, make),
            a0: self.momentum(row, 26, make),
            a1: self.momentum(row, 30, make),
            parameters: std::array::from_fn(|i| make(self.value(row, 34 + i))),
        })
    }
}

impl<T: Real + SingleFloat> SymbolicaCuts<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(prototype: T) -> Result<Self> {
        Ok(Self {
            engine: CutEngine::new(prototype, None)?,
        })
    }
}

impl<T: Real + SingleFloat> CutProvider<Complex<T>> for SymbolicaCuts<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn tadpole_batch(
        &mut self,
        requests: &[TadpoleCutRequest<'_, Complex<T>>],
        output: &mut [CutOutcome<TadpoleCut<Complex<T>>>],
    ) -> Result<()> {
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("tadpole cut output", output.len())?;
        self.engine
            .run_tadpole(requests, std::slice::from_ref, |_| true)?;
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.engine.tadpole(row, &|values| values[0].clone());
        }
        Ok(())
    }
    fn triangle_batch(
        &mut self,
        requests: &[TriangleCutRequest<'_, Complex<T>>],
        output: &mut [CutOutcome<TriangleCut<Complex<T>>>],
    ) -> Result<()> {
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("triangle cut output", output.len())?;
        self.engine.run(
            requests.len(),
            0,
            |row, i| requests[row].shifts[i],
            |row, i| requests[row].masses_squared[i],
            |_| &[][..],
            std::slice::from_ref,
            |_| true,
        )?;
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.engine.triangle(row, &|values| values[0].clone());
        }
        Ok(())
    }
    fn bubble_batch(
        &mut self,
        requests: &[BubbleCutRequest<'_, Complex<T>>],
        output: &mut [CutOutcome<BubbleCut<Complex<T>>>],
    ) -> Result<()> {
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("bubble cut output", output.len())?;
        self.engine.run(
            requests.len(),
            1,
            |row, i| requests[row].shifts[i],
            |row, i| requests[row].masses_squared[i],
            |row| requests[row].complementary_shifts,
            std::slice::from_ref,
            |_| true,
        )?;
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.engine.bubble(row, &|values| values[0].clone());
        }
        Ok(())
    }
}

impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> SymbolicaDualCuts<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(prototype: T, dual: D) -> Result<Self> {
        let shape: Vec<_> = dual
            .get_shape()
            .into_iter()
            .map(<[usize]>::to_vec)
            .collect();
        Ok(Self {
            engine: CutEngine::new(prototype, Some(&shape))?,
            prototype: dual,
            shape,
        })
    }
    fn make(&self, values: &[Complex<T>]) -> D {
        let mut value = self.prototype.zero();
        value.values_mut().clone_from_slice(values);
        value
    }
}

impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> CutProvider<D>
    for SymbolicaDualCuts<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn tadpole_batch(
        &mut self,
        requests: &[TadpoleCutRequest<'_, D>],
        output: &mut [CutOutcome<TadpoleCut<D>>],
    ) -> Result<()> {
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("tadpole cut output", output.len())?;
        let shape = &self.shape;
        self.engine.run_tadpole(
            requests,
            |value| value.values(),
            |value| {
                value
                    .get_shape()
                    .iter()
                    .copied()
                    .eq(shape.iter().map(Vec::as_slice))
            },
        )?;
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.engine.tadpole(row, &|values| self.make(values));
        }
        Ok(())
    }
    fn triangle_batch(
        &mut self,
        requests: &[TriangleCutRequest<'_, D>],
        output: &mut [CutOutcome<TriangleCut<D>>],
    ) -> Result<()> {
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("triangle cut output", output.len())?;
        let shape = &self.shape;
        self.engine.run(
            requests.len(),
            0,
            |row, i| requests[row].shifts[i],
            |row, i| requests[row].masses_squared[i],
            |_| &[][..],
            |value| value.values(),
            |value| {
                value
                    .get_shape()
                    .iter()
                    .copied()
                    .eq(shape.iter().map(Vec::as_slice))
            },
        )?;
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.engine.triangle(row, &|values| self.make(values));
        }
        Ok(())
    }
    fn bubble_batch(
        &mut self,
        requests: &[BubbleCutRequest<'_, D>],
        output: &mut [CutOutcome<BubbleCut<D>>],
    ) -> Result<()> {
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("bubble cut output", output.len())?;
        let shape = &self.shape;
        self.engine.run(
            requests.len(),
            1,
            |row, i| requests[row].shifts[i],
            |row, i| requests[row].masses_squared[i],
            |row| requests[row].complementary_shifts,
            |value| value.values(),
            |value| {
                value
                    .get_shape()
                    .iter()
                    .copied()
                    .eq(shape.iter().map(Vec::as_slice))
            },
        )?;
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.engine.bubble(row, &|values| self.make(values));
        }
        Ok(())
    }
}

fn triangle_sample_argument<'a, T>(request: &TriangleSampleRequest<'a, T>, column: usize) -> &'a T {
    match column {
        0..4 => request.cut.a.components()[column],
        4..8 => request.cut.basis.e3.components()[column - 4],
        8..12 => request.cut.e4_effective.components()[column - 8],
        12 => &request.cut.beta,
        13 => request.t,
        14 => request.mu_squared,
        _ => unreachable!("triangle sampler argument index"),
    }
}

fn bubble_sample_argument<'a, T>(request: &BubbleSampleRequest<'a, T>, column: usize) -> &'a T {
    match column {
        0..4 => request.cut.a0.components()[column],
        4..8 => request.cut.a1.components()[column - 4],
        8..12 => request.cut.basis.e3.components()[column - 8],
        12..16 => request.cut.e4_effective.components()[column - 12],
        16..19 => &request.cut.parameters[column - 16],
        19 => request.x,
        20 => request.t,
        21 => request.mu_squared,
        _ => unreachable!("bubble sampler argument index"),
    }
}

impl<T: Real + SingleFloat> CutSampler<Complex<T>> for SymbolicaCuts<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn sample_triangle_batch(
        &mut self,
        requests: &[TriangleSampleRequest<'_, Complex<T>>],
        output: &mut [CutOutcome<FourMomentum<Complex<T>>>],
    ) -> Result<()> {
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("triangle sample output", output.len())?;
        self.engine.sample(
            requests.len(),
            0,
            |row, column| triangle_sample_argument(&requests[row], column),
            std::slice::from_ref,
            |_| true,
        )?;
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.engine.sample_result(row, &|values| values[0].clone());
        }
        Ok(())
    }
    fn sample_bubble_batch(
        &mut self,
        requests: &[BubbleSampleRequest<'_, Complex<T>>],
        output: &mut [CutOutcome<FourMomentum<Complex<T>>>],
    ) -> Result<()> {
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("bubble sample output", output.len())?;
        self.engine.sample(
            requests.len(),
            1,
            |row, column| bubble_sample_argument(&requests[row], column),
            std::slice::from_ref,
            |_| true,
        )?;
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.engine.sample_result(row, &|values| values[0].clone());
        }
        Ok(())
    }
}

impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> CutSampler<D>
    for SymbolicaDualCuts<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    fn sample_triangle_batch(
        &mut self,
        requests: &[TriangleSampleRequest<'_, D>],
        output: &mut [CutOutcome<FourMomentum<D>>],
    ) -> Result<()> {
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("triangle sample output", output.len())?;
        let shape = &self.shape;
        self.engine.sample(
            requests.len(),
            0,
            |row, column| triangle_sample_argument(&requests[row], column),
            |value| value.values(),
            |value| {
                value
                    .get_shape()
                    .iter()
                    .copied()
                    .eq(shape.iter().map(Vec::as_slice))
            },
        )?;
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.engine.sample_result(row, &|values| self.make(values));
        }
        Ok(())
    }
    fn sample_bubble_batch(
        &mut self,
        requests: &[BubbleSampleRequest<'_, D>],
        output: &mut [CutOutcome<FourMomentum<D>>],
    ) -> Result<()> {
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("bubble sample output", output.len())?;
        let shape = &self.shape;
        self.engine.sample(
            requests.len(),
            1,
            |row, column| bubble_sample_argument(&requests[row], column),
            |value| value.values(),
            |value| {
                value
                    .get_shape()
                    .iter()
                    .copied()
                    .eq(shape.iter().map(Vec::as_slice))
            },
        )?;
        for (row, value) in output.iter_mut().enumerate() {
            *value = self.engine.sample_result(row, &|values| self.make(values));
        }
        Ok(())
    }
}
