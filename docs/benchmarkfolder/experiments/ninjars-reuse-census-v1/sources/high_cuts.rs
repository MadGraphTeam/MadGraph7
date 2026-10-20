//! Symbolica-generated linear cut solutions and complex quadratic sampling.
//!
//! Algebra reuse audit before implementation:
//! 1. Public Atom::system_to_matrix and Matrix::solve/det provide exact solves.
//! 2. Symbolica examples/solve_linear_system.rs and Numerica tensors/matrix.rs
//!    (solve at1949, determinant at1256, their tests) implement these operations.
//! 3. FeynKit tensor/weingarten.rs and graph/integrals.rs use the same matrix
//!    solve and to_expression APIs; shared FourMomentum supplies every dot.
//!
//! No numerical elimination or polynomial solver is implemented here. Four
//! symbolic coordinate charts cover each rank-three box constraint matrix.
//! The largest normalized minor chooses the chart using primal values only.
//! Solved vectors, their orthogonal-center projection and complex roots remain
//! inside the same batched Symbolica graphs, including all Dual components.

use crate::kernel::{CompiledDualKernel, CompiledKernel, ExactKernel, KernelExecutor};
use ninjars::{
    BatchShape, BoxCut, BoxCutRequest, BoxSampleRequest, Complex, CutFailure, CutOutcome, Error,
    FloatLike, FourMomentum, HighCutProvider, PentagonCut, PentagonCutRequest, Real, Result,
    SingleFloat,
};
use symbolica::{
    atom::{Atom, AtomCore},
    domains::dual::DualNumber,
    evaluate::EvaluationDomain,
    symbol,
};

fn vars(n: usize, scope: &str) -> Vec<Atom> {
    (0..n)
        .map(|i| Atom::var(symbol!(format!("ninjars_high_cut_{scope}::p{i}"))))
        .collect()
}
fn momentum(p: &[Atom], start: usize) -> FourMomentum<Atom> {
    FourMomentum::from_args(
        p[start].clone(),
        p[start + 1].clone(),
        p[start + 2].clone(),
        p[start + 3].clone(),
    )
}
fn append(p: &mut Vec<Atom>, v: &FourMomentum<Atom>) {
    p.extend(v.components().into_iter().cloned());
}
fn euclidean_squared(p: &FourMomentum<Atom>) -> Atom {
    p.components()
        .into_iter()
        .fold(Atom::Zero, |a, v| a + v * v.conj())
}

/// Result order: all shift differences, all h_i=(mi²-m0²-Ki²)/2,
/// Vfirst's four components, mfirst².
fn preparation(n: usize) -> ExactKernel {
    let p = vars(
        5 * n,
        if n == 4 {
            "box_input"
        } else {
            "pentagon_input"
        },
    );
    let v0 = momentum(&p, 0);
    let differences: Vec<_> = (1..n).map(|i| momentum(&p, 4 * i) - v0.clone()).collect();
    let mut expressions = Vec::new();
    for k in &differences {
        append(&mut expressions, k);
    }
    for (i, k) in differences.iter().enumerate() {
        expressions.push((&p[4 * n + i + 1] - &p[4 * n] - k.dot(k)) / 2);
    }
    append(&mut expressions, &v0);
    expressions.push(p[4 * n].clone());
    ExactKernel {
        expressions,
        parameters: p,
    }
}

/// Use Symbolica's exact rational-polynomial solver for a selected free
/// coordinate. None selects the full four-component pentagon solve.
fn solve_chart(n: usize, free: Option<usize>) -> Result<(Vec<Atom>, ExactKernel)> {
    let p = vars(
        5 * n,
        if n == 4 {
            "box_prepared"
        } else {
            "pentagon_prepared"
        },
    );
    let count = n - 1;
    let unknowns = vars(
        count,
        if n == 4 {
            "box_unknown"
        } else {
            "pentagon_unknown"
        },
    );
    let coordinates: Vec<_> = (0..4).filter(|i| Some(*i) != free).collect();
    let mut vector = vec![Atom::Zero; 4];
    for (&index, u) in coordinates.iter().zip(&unknowns) {
        vector[index] = u.clone();
    }
    let trial = momentum(&vector, 0);
    let equations: Vec<_> = (0..count)
        .map(|i| momentum(&p, 4 * i).dot(&trial) - &p[4 * count + i])
        .collect();
    let (matrix, rhs) = Atom::system_to_matrix::<u16, _, _>(&equations, &unknowns)
        .map_err(|e| Error::Symbolic(e.to_string()))?;
    let determinant = matrix
        .det()
        .map_err(|e| Error::Symbolic(e.to_string()))?
        .to_expression();
    let solved = matrix
        .solve(&rhs)
        .map_err(|e| Error::Symbolic(e.to_string()))?
        .into_vec();
    for (&index, value) in coordinates.iter().zip(solved) {
        vector[index] = value.to_expression();
    }
    let particular = momentum(&vector, 0);
    let mut expressions = Vec::new();
    if let Some(free) = free {
        let mut normal = vec![Atom::Zero; 4];
        normal[free] = Atom::num(1);
        for (&index, u) in coordinates.iter().zip(&unknowns) {
            normal[index] = u.clone();
        }
        let trial = momentum(&normal, 0);
        let equations: Vec<_> = (0..count)
            .map(|i| momentum(&p, 4 * i).dot(&trial))
            .collect();
        let (matrix, rhs) = Atom::system_to_matrix::<u16, _, _>(&equations, &unknowns)
            .map_err(|e| Error::Symbolic(e.to_string()))?;
        let solved = matrix
            .solve(&rhs)
            .map_err(|e| Error::Symbolic(e.to_string()))?
            .into_vec();
        for (&index, value) in coordinates.iter().zip(solved) {
            normal[index] = value.to_expression();
        }
        let normal = momentum(&normal, 0);
        let norm = normal.dot(&normal);
        let scale = euclidean_squared(&normal);
        append(&mut expressions, &particular);
        append(&mut expressions, &normal);
        expressions.push(norm.clone());
        expressions.push(&norm * norm.conj() / scale.pow(2));
        append(&mut expressions, &momentum(&p, 5 * count));
        expressions.push(p[5 * count + 4].clone());
    } else {
        append(
            &mut expressions,
            &(particular.clone() - momentum(&p, 5 * count)),
        );
        expressions.push(particular.dot(&particular) - &p[5 * count + 4]);
    }
    let scale = (0..count).fold(Atom::num(1), |a, i| {
        a * euclidean_squared(&momentum(&p, 4 * i))
    });
    let score = &determinant * determinant.conj() / scale;
    Ok((
        vec![score],
        ExactKernel {
            expressions,
            parameters: p,
        },
    ))
}

fn finish_box() -> ExactKernel {
    let p = vars(15, "box_finish");
    let particular = momentum(&p, 0);
    let normal = momentum(&p, 4);
    let center = particular.clone() - normal.clone() * (particular.dot(&normal) / &p[8]);
    let mut expressions = Vec::new();
    append(&mut expressions, &(center.clone() - momentum(&p, 10)));
    append(&mut expressions, &normal);
    expressions.push(p[8].clone());
    expressions.push(&p[14] - center.dot(&center));
    ExactKernel {
        expressions,
        parameters: p,
    }
}
fn sample_probe() -> ExactKernel {
    let p = vars(3, "sample_probe");
    ExactKernel {
        expressions: vec![&p[1] + &p[2], (&p[1] + &p[2]) / &p[0]],
        parameters: p,
    }
}
fn sample_finish() -> ExactKernel {
    let p = vars(9, "sample_finish");
    let a = momentum(&p, 0);
    let v = momentum(&p, 4) * p[8].sqrt();
    let mut expressions = Vec::new();
    append(&mut expressions, &(a.clone() + v.clone()));
    append(&mut expressions, &(a - v));
    ExactKernel {
        expressions,
        parameters: p,
    }
}

struct Engine<T, V, K> {
    prepare: [K; 2],
    probes: [K; 2],
    charts: Vec<K>,
    box_finish: K,
    pentagon_finish: K,
    sample_probe: K,
    sample_finish: K,
    zero: V,
    threshold: T,
    components: for<'a> fn(&'a V) -> &'a [Complex<T>],
    shape_matches: fn(&V, &V) -> bool,
    input: Vec<V>,
    prepared: Vec<V>,
    probed: Vec<V>,
    seeded: Vec<V>,
    finished: Vec<V>,
    active: Vec<usize>,
    accepted: Vec<usize>,
    groups: [Vec<usize>; 4],
    failures: Vec<Option<CutFailure>>,
    values: Vec<Vec<V>>,
}
impl<T: Real + SingleFloat, V: FloatLike, K: KernelExecutor<V>> Engine<T, V, K> {
    fn new(
        prototype: &T,
        zero: &V,
        mut compile: impl FnMut(&ExactKernel) -> Result<K>,
        components: for<'a> fn(&'a V) -> &'a [Complex<T>],
        shape_matches: fn(&V, &V) -> bool,
    ) -> Result<Self> {
        if components(zero).is_empty() {
            return Err(Error::Unsupported(
                "high-cut carrier has no primal component".into(),
            ));
        }
        let prepare = [compile(&preparation(4))?, compile(&preparation(5))?];
        let mut charts = Vec::new();
        let mut scores = Vec::new();
        for chart in 0..4 {
            let (score, plan) = solve_chart(4, Some(chart))?;
            scores.extend(score);
            charts.push(compile(&plan)?);
        }
        let box_probe = compile(&ExactKernel {
            expressions: scores,
            parameters: vars(20, "box_prepared"),
        })?;
        let (score, pentagon) = solve_chart(5, None)?;
        let pentagon_probe = compile(&ExactKernel {
            expressions: score,
            parameters: pentagon.parameters.clone(),
        })?;
        Ok(Self {
            prepare,
            probes: [box_probe, pentagon_probe],
            charts,
            box_finish: compile(&finish_box())?,
            pentagon_finish: compile(&pentagon)?,
            sample_probe: compile(&sample_probe())?,
            sample_finish: compile(&sample_finish())?,
            zero: zero.zero(),
            threshold: prototype
                .from_i64(2)
                .pow(u64::from(prototype.get_precision()))
                .inv(),
            components,
            shape_matches,
            input: Vec::new(),
            prepared: Vec::new(),
            probed: Vec::new(),
            seeded: Vec::new(),
            finished: Vec::new(),
            active: Vec::new(),
            accepted: Vec::new(),
            groups: std::array::from_fn(|_| Vec::new()),
            failures: Vec::new(),
            values: Vec::new(),
        })
    }
    fn finite(&self, v: &V) -> bool {
        (self.components)(v)
            .iter()
            .all(|c| c.re.is_finite() && c.im.is_finite())
    }
    fn zero_primal(&self, v: &V) -> bool {
        (self.components)(v)[0].is_fully_zero()
    }
    fn positive_score(&self, v: &V) -> bool {
        self.finite(v)
            && (self.components)(v)[0].re.real_cmp(&self.threshold)
                == Some(std::cmp::Ordering::Greater)
    }
    fn reset(&mut self, rows: usize) {
        self.failures.clear();
        self.failures.resize(rows, None);
        self.active.clear();
        self.input.clear();
        if self.values.len() < rows {
            self.values.resize_with(rows, Vec::new);
        }
        for v in self.values.iter_mut().take(rows) {
            v.clear();
        }
    }
    fn input_status<'a>(&self, values: impl IntoIterator<Item = &'a V>) -> Option<CutFailure>
    where
        V: 'a,
    {
        for v in values {
            if !(self.shape_matches)(v, &self.zero) || (self.components)(v).is_empty() {
                return Some(CutFailure::DerivativeShape);
            }
            if !self.finite(v) {
                return Some(CutFailure::NonFiniteInput);
            }
        }
        None
    }
    fn geometry<'a>(
        &mut self,
        n: usize,
        rows: usize,
        argument: impl Fn(usize, usize) -> &'a V,
    ) -> Result<()>
    where
        V: 'a,
    {
        self.reset(rows);
        let mode = n - 4;
        let columns = 5 * n;
        for row in 0..rows {
            self.failures[row] =
                self.input_status((0..columns).map(|column| argument(row, column)));
            if self.failures[row].is_none() {
                self.active.push(row);
                self.input
                    .extend((0..columns).map(|column| argument(row, column).clone()));
            }
        }
        self.prepared
            .resize(self.active.len() * columns, self.zero.clone());
        self.prepare[mode].evaluate(&self.input, &mut self.prepared, self.active.len())?;
        let probe_columns = if n == 4 { 4 } else { 1 };
        self.probed
            .resize(self.active.len() * probe_columns, self.zero.clone());
        self.probes[mode].evaluate(&self.prepared, &mut self.probed, self.active.len())?;
        for group in &mut self.groups {
            group.clear();
        }
        for (index, &row) in self.active.iter().enumerate() {
            if self.prepared[index * columns..(index + 1) * columns]
                .iter()
                .any(|v| !self.finite(v))
            {
                self.failures[row] = Some(CutFailure::NonFiniteResult);
                continue;
            }
            let scores = &self.probed[index * probe_columns..(index + 1) * probe_columns];
            let mut best = None;
            for (i, score) in scores.iter().enumerate() {
                if self.positive_score(score)
                    && best.is_none_or(|j| {
                        (self.components)(score)[0]
                            .re
                            .real_cmp(&(self.components)(&scores[j])[0].re)
                            == Some(std::cmp::Ordering::Greater)
                    })
                {
                    best = Some(i);
                }
            }
            if let Some(best) = best {
                self.groups[best].push(index);
            } else {
                self.failures[row] = Some(CutFailure::DegenerateGram);
            }
        }
        for chart in 0..probe_columns {
            self.input.clear();
            for &index in &self.groups[chart] {
                self.input
                    .extend_from_slice(&self.prepared[index * columns..(index + 1) * columns]);
            }
            if n == 4 {
                self.seeded
                    .resize(self.groups[chart].len() * 15, self.zero.clone());
                self.charts[chart].evaluate(
                    &self.input,
                    &mut self.seeded,
                    self.groups[chart].len(),
                )?;
                self.input.clear();
                self.accepted.clear();
                for (index, &active_index) in self.groups[chart].iter().enumerate() {
                    let row = self.active[active_index];
                    let values = &self.seeded[index * 15..(index + 1) * 15];
                    if values.iter().any(|v| !self.finite(v)) {
                        self.failures[row] = Some(CutFailure::NonFiniteResult);
                        continue;
                    }
                    if !self.positive_score(&values[9]) {
                        self.failures[row] = Some(CutFailure::DegenerateGram);
                        continue;
                    }
                    self.accepted.push(row);
                    self.input.extend_from_slice(values);
                }
                self.finished
                    .resize(self.accepted.len() * 10, self.zero.clone());
                self.box_finish
                    .evaluate(&self.input, &mut self.finished, self.accepted.len())?;
                for (index, &row) in self.accepted.iter().enumerate() {
                    let values = &self.finished[index * 10..(index + 1) * 10];
                    if values.iter().all(|v| self.finite(v)) {
                        self.values[row].extend_from_slice(values);
                    } else {
                        self.failures[row] = Some(CutFailure::NonFiniteResult);
                    }
                }
            } else {
                self.finished
                    .resize(self.groups[0].len() * 5, self.zero.clone());
                self.pentagon_finish.evaluate(
                    &self.input,
                    &mut self.finished,
                    self.groups[0].len(),
                )?;
                for (index, &active_index) in self.groups[0].iter().enumerate() {
                    let row = self.active[active_index];
                    let values = &self.finished[index * 5..(index + 1) * 5];
                    if values.iter().any(|v| !self.finite(v)) {
                        self.failures[row] = Some(CutFailure::NonFiniteResult);
                    } else if self.zero_primal(&values[4]) {
                        self.failures[row] = Some(CutFailure::SingularParameter);
                    } else {
                        self.values[row].extend_from_slice(values);
                    }
                }
            }
        }
        Ok(())
    }
    fn sample(&mut self, requests: &[BoxSampleRequest<'_, V>]) -> Result<()> {
        self.reset(requests.len());
        for (row, r) in requests.iter().enumerate() {
            let c = r.cut;
            self.failures[row] = self.input_status(
                c.a.components()
                    .into_iter()
                    .chain(c.v_orth.components())
                    .chain([&c.v_orth_squared, &c.beta, r.mu_squared]),
            );
            if self.failures[row].is_some() {
                continue;
            }
            if self.zero_primal(&c.v_orth_squared) {
                self.failures[row] = Some(CutFailure::DegenerateGram);
                continue;
            }
            self.active.push(row);
            self.input.extend([
                c.v_orth_squared.clone(),
                c.beta.clone(),
                r.mu_squared.clone(),
            ]);
        }
        self.probed.resize(self.active.len() * 2, self.zero.clone());
        self.sample_probe
            .evaluate(&self.input, &mut self.probed, self.active.len())?;
        self.input.clear();
        self.accepted.clear();
        for (index, &row) in self.active.iter().enumerate() {
            let values = &self.probed[index * 2..(index + 1) * 2];
            if values.iter().any(|v| !self.finite(v)) {
                self.failures[row] = Some(CutFailure::NonFiniteResult);
                continue;
            }
            if self.zero_primal(&values[0]) {
                self.failures[row] = Some(CutFailure::SingularParameter);
                continue;
            }
            let c = requests[row].cut;
            self.accepted.push(row);
            self.input.extend(
                c.a.components()
                    .into_iter()
                    .chain(c.v_orth.components())
                    .cloned(),
            );
            self.input.push(values[1].clone());
        }
        self.finished
            .resize(self.accepted.len() * 8, self.zero.clone());
        self.sample_finish
            .evaluate(&self.input, &mut self.finished, self.accepted.len())?;
        for (index, &row) in self.accepted.iter().enumerate() {
            let values = &self.finished[index * 8..(index + 1) * 8];
            if values.iter().all(|v| self.finite(v)) {
                self.values[row].extend_from_slice(values);
            } else {
                self.failures[row] = Some(CutFailure::NonFiniteResult);
            }
        }
        Ok(())
    }
    fn vector(&self, row: usize, start: usize) -> FourMomentum<V> {
        let p = &self.values[row];
        FourMomentum::from_args(
            p[start].clone(),
            p[start + 1].clone(),
            p[start + 2].clone(),
            p[start + 3].clone(),
        )
    }
    fn box_result(&self, row: usize) -> CutOutcome<BoxCut<V>> {
        if let Some(f) = self.failures[row] {
            CutOutcome::Unstable(f)
        } else {
            CutOutcome::Stable(BoxCut {
                a: self.vector(row, 0),
                v_orth: self.vector(row, 4),
                v_orth_squared: self.values[row][8].clone(),
                beta: self.values[row][9].clone(),
            })
        }
    }
    fn pentagon_result(&self, row: usize) -> CutOutcome<PentagonCut<V>> {
        if let Some(f) = self.failures[row] {
            CutOutcome::Unstable(f)
        } else {
            CutOutcome::Stable(PentagonCut {
                q: self.vector(row, 0),
                mu_squared: self.values[row][4].clone(),
            })
        }
    }
    fn sample_result(&self, row: usize) -> CutOutcome<[FourMomentum<V>; 2]> {
        if let Some(f) = self.failures[row] {
            CutOutcome::Unstable(f)
        } else {
            CutOutcome::Stable([self.vector(row, 0), self.vector(row, 4)])
        }
    }
}

pub struct SymbolicaHighCuts<T> {
    engine: Engine<T, Complex<T>, CompiledKernel<T>>,
}
impl<T: Real + SingleFloat> SymbolicaHighCuts<T>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(prototype: T) -> Result<Self> {
        let zero = Complex::new(prototype.zero(), prototype.zero());
        Ok(Self {
            engine: Engine::new(
                &prototype,
                &zero,
                |e| e.compile(&prototype),
                std::slice::from_ref,
                |_, _| true,
            )?,
        })
    }
}
pub struct SymbolicaDualHighCuts<T, D> {
    engine: Engine<T, D, CompiledDualKernel<T, D>>,
}
impl<T: Real + SingleFloat, D: Real + DualNumber<Complex<T>>> SymbolicaDualHighCuts<T, D>
where
    Complex<T>: Real + EvaluationDomain,
{
    pub fn new(prototype: T, dual: D) -> Result<Self> {
        Ok(Self {
            engine: Engine::new(
                &prototype,
                &dual,
                |e| e.compile_dual(&prototype, &dual),
                DualNumber::values,
                |a, b| a.values().len() == b.values().len() && a.get_shape() == b.get_shape(),
            )?,
        })
    }
}
macro_rules! implement {
    ($name:ident<$($g:ident),+>,$value:ty,$($bounds:tt)*)=> {
        impl<$($g),+> HighCutProvider<$value> for $name<$($g),+> where $($bounds)* {
            fn box_batch(&mut self,requests:&[BoxCutRequest<'_,$value>],output:&mut[CutOutcome<BoxCut<$value>>])->Result<()> {
                BatchShape{rows:requests.len(),columns:1}.validate("box cut output",output.len())?;
                self.engine.geometry(4,requests.len(),|row,column|if column<16{requests[row].shifts[column/4].components()[column%4]}else{requests[row].masses_squared[column-16]})?;
                for (row,out) in output.iter_mut().enumerate(){*out=self.engine.box_result(row);}Ok(())
            }
            fn pentagon_batch(&mut self,requests:&[PentagonCutRequest<'_,$value>],output:&mut[CutOutcome<PentagonCut<$value>>])->Result<()> {
                BatchShape{rows:requests.len(),columns:1}.validate("pentagon cut output",output.len())?;
                self.engine.geometry(5,requests.len(),|row,column|if column<20{requests[row].shifts[column/4].components()[column%4]}else{requests[row].masses_squared[column-20]})?;
                for (row,out) in output.iter_mut().enumerate(){*out=self.engine.pentagon_result(row);}Ok(())
            }
            fn sample_box_batch(&mut self,requests:&[BoxSampleRequest<'_,$value>],output:&mut[CutOutcome<[FourMomentum<$value>;2]>])->Result<()> {
                BatchShape{rows:requests.len(),columns:1}.validate("box sample output",output.len())?;
                self.engine.sample(requests)?;for (row,out) in output.iter_mut().enumerate(){*out=self.engine.sample_result(row);}Ok(())
            }
        }
    };
}
implement!(SymbolicaHighCuts<T>,Complex<T>,T:Real+SingleFloat,Complex<T>:Real+EvaluationDomain);
implement!(SymbolicaDualHighCuts<T,D>,D,T:Real+SingleFloat,D:Real+DualNumber<Complex<T>>,Complex<T>:Real+EvaluationDomain);
