// Adapted from NinjaRS 0d0d7e7a8033431b60a8580731084742e0655cda,
// crates/ninjars-cli/src/case_check/harness.rs, precision-generic tensor section.
// Copyright the NinjaRS contributors; GPL-3.0-or-later.
// Keep the numerical recurrence in sync with the pinned upstream implementation.
use ninjars::fast::tensor_layout::{monomial_index, monomials, mu_block_offset, ncoeff, ncoeff_mu};
use ninjars::{
    BatchShape, Complex, ExpansionKind, ExpansionRequest, FloatLike, FourMomentum, Numerator,
    NumeratorRequest, Real, SingleFloat,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
/// Shape of one Laurent expansion: `q = t F(u, x, mu^2)` with `u = 1/t`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// `Mu` (`beta = 0`) and `MuShifted`: `F = v0 + v1 u`, `mu^2 = t^2 (v0^2 - beta u^2)`.
    Mu,
    /// `F = e3 + a u + (beta + mu^2) e4 u^2`.
    T3,
    /// `F = e3 + (a0 + x a1) u + (b0 + b1 x + b2 x^2 + mu^2) e4 u^2`.
    T2,
}

/// A dense tensor numerator `N(q, mu^2) = sum_k (mu^2)^k T_k(q)` in the
/// `tensor_layout` (and `MuTensorNumerator` block) order for the generic
/// engine in any precision.
///
/// Expansions are computed numerically: each monomial `q^m = t^|m| prod_c
/// F_c^(m_c)` is a truncated series in `u = 1/t` (and `x`, `mu^2` for the
/// triangle and bubble parametrisations), built from its parent monomial with
/// one multiplication by `F_c`; the requested Laurent coefficients are then
/// contracted with the lane's coefficients. Exact arithmetic would reproduce
/// [`ninjars_symbolica::PolynomialNumerator`] exactly; in floating point both
/// round differently.
pub struct TensorGenericNumerator<T> {
    rank: usize,
    mu_powers: usize,
    width: usize,
    degrees: Vec<usize>,
    /// `(parent monomial, component)` of every monomial but the constant.
    parents: Vec<(usize, usize)>,
    coefficients: Vec<Complex<T>>,
    lanes: usize,
    zero: Complex<T>,
    one: Complex<T>,
    values_work: Vec<Complex<T>>,
    series_work: Vec<Complex<T>>,
}

impl<T: Real + SingleFloat> TensorGenericNumerator<T>
where
    Complex<T>: Real,
{
    pub fn new(rank: usize, mu_powers: usize, prototype: &T) -> Result<Self> {
        if 2 * mu_powers > rank {
            return Err("mu_powers exceeds rank / 2".into());
        }
        let list = monomials(rank);
        let degrees = list
            .iter()
            .map(|m| m.iter().sum::<u32>() as usize)
            .collect();
        let parents = list
            .iter()
            .map(|m| {
                let Some(c) = (0..4).rev().find(|&c| m[c] > 0) else {
                    return (0, 0);
                };
                let mut parent = *m;
                parent[c] -= 1;
                (monomial_index(parent), c)
            })
            .collect();
        let zero = Complex::new(prototype.zero(), prototype.zero());
        let one = Complex::new(prototype.one(), prototype.zero());
        Ok(Self {
            rank,
            mu_powers,
            width: ncoeff_mu(rank, mu_powers),
            degrees,
            parents,
            coefficients: Vec::new(),
            lanes: 0,
            zero,
            one,
            values_work: Vec::new(),
            series_work: Vec::new(),
        })
    }

    /// Bind lane-major coefficients (`width()` per lane).
    pub fn set_coefficients(&mut self, coefficients: Vec<Complex<T>>, lanes: usize) -> Result<()> {
        if coefficients.len() != lanes * self.width {
            return Err(format!(
                "{} coefficients for {lanes} lanes of width {}",
                coefficients.len(),
                self.width
            )
            .into());
        }
        self.coefficients = coefficients;
        self.lanes = lanes;
        Ok(())
    }

    /// `(offset, size)` of the `mu^(2k)` block.
    fn block(&self, k: usize) -> (usize, usize) {
        (mu_block_offset(self.rank, k), ncoeff(self.rank - 2 * k))
    }

    fn lane_coefficients(&self, lane: usize) -> ninjars::Result<&[Complex<T>]> {
        if lane >= self.lanes {
            return Err(ninjars::Error::Lane {
                lane,
                lanes: self.lanes,
            });
        }
        Ok(&self.coefficients[lane * self.width..(lane + 1) * self.width])
    }
}

// Allocation-only expansion of the pinned Complex<T>::mul expression.
// The four scalar products, operand order and two additions are unchanged.
#[inline]
fn borrowed_product<T: FloatLike>(a: &Complex<T>, b: &Complex<T>) -> Complex<T> {
    Complex::new(
        a.re.clone() * &b.re - a.im.clone() * &b.im,
        a.re.clone() * &b.im + a.im.clone() * &b.re,
    )
}

fn minkowski<T: Real>(a: &[Complex<T>; 4], b: &[Complex<T>; 4]) -> Complex<T>
where
    Complex<T>: Real,
{
    borrowed_product(&a[0], &b[0]) - borrowed_product(&a[1], &b[1]) - borrowed_product(&a[2], &b[2]) - borrowed_product(&a[3], &b[3])
}

fn components<T: Clone>(v: &FourMomentum<T>) -> [T; 4] {
    v.components().map(|x| x.clone())
}

impl<T: Real + SingleFloat> Numerator<Complex<T>> for TensorGenericNumerator<T>
where
    Complex<T>: Real,
{
    fn rank(&self) -> usize {
        self.rank
    }

    fn evaluate_batch(
        &mut self,
        requests: &[NumeratorRequest<'_, Complex<T>>],
        output: &mut [Complex<T>],
    ) -> ninjars::Result<()> {
        BatchShape {
            rows: requests.len(),
            columns: 1,
        }
        .validate("numerator output", output.len())?;
        let count = self.degrees.len();
        let mut values = std::mem::take(&mut self.values_work);
        values.resize_with(count, || self.zero.clone());
        for (request, out) in requests.iter().zip(output.iter_mut()) {
            let coefficients = self.lane_coefficients(request.id.lane)?;
            let q = components(request.q);
            values[0].set_from(&self.one);
            for i in 1..count {
                let (parent, c) = self.parents[i];
                values[i] = borrowed_product(&values[parent], &q[c]);
            }
            let mut total = self.zero.clone();
            let mut mu_power = self.one.clone();
            for k in 0..=self.mu_powers {
                let (offset, size) = self.block(k);
                let mut block = self.zero.clone();
                for (c, v) in coefficients[offset..offset + size].iter().zip(&values) {
                    block += borrowed_product(c, v);
                }
                total += block * &mu_power;
                mu_power *= request.mu_squared;
            }
            *out = total;
        }
        self.values_work = values;
        Ok(())
    }

    fn expansion_batch(
        &mut self,
        kind: &ExpansionKind,
        requests: &[ExpansionRequest<'_, Complex<T>>],
        output: &mut [Complex<T>],
    ) -> ninjars::Result<()> {
        let columns = kind.len();
        BatchShape {
            rows: requests.len(),
            columns,
        }
        .validate("expansion output", output.len())?;
        let (shape, vectors, scalars, shifted) = match kind {
            ExpansionKind::Mu { .. } => (Shape::Mu, 2, 0, false),
            ExpansionKind::MuShifted { .. } => (Shape::Mu, 2, 1, true),
            ExpansionKind::T3 { .. } => (Shape::T3, 3, 1, false),
            ExpansionKind::T2 { .. } => (Shape::T2, 4, 3, false),
        };
        for request in requests {
            self.lane_coefficients(request.id.lane)?;
            if request.vectors.len() != vectors
                && !(matches!(kind, ExpansionKind::Mu { .. }) && request.vectors.len() == 1)
            {
                return Err(ninjars::Error::Shape {
                    name: "cut vectors",
                    expected: vectors,
                    actual: request.vectors.len(),
                });
            }
            BatchShape {
                rows: 1,
                columns: scalars,
            }
            .validate("cut parameters", request.parameters.len())?;
        }
        if requests.is_empty() || columns == 0 {
            return Ok(());
        }
        // Requested (t power, x power, mu^2 power) per column.
        let powers: Vec<(i64, usize, usize)> = match kind {
            ExpansionKind::Mu { powers } | ExpansionKind::MuShifted { powers } => {
                powers.iter().map(|&p| (i64::from(p), 0, 0)).collect()
            }
            ExpansionKind::T3 { powers } => powers
                .iter()
                .map(|&(p, l)| (i64::from(p), 0, usize::from(l)))
                .collect(),
            ExpansionKind::T2 { powers } => powers
                .iter()
                .map(|&(p, a, l)| (i64::from(p), usize::from(a), usize::from(l)))
                .collect(),
        };
        let lowest = powers.iter().map(|p| p.0).min().unwrap_or(0);
        // Deepest u order any term reaches at the requested powers.
        let depth = (self.rank as i64 - lowest).max(0) as usize;
        let (xs, mus) = match shape {
            Shape::Mu => (1, 1),
            Shape::T3 => (1, depth / 2 + 1),
            Shape::T2 => (depth + 1, depth / 2 + 1),
        };
        let size = (depth + 1) * xs * mus;
        let at = |j: usize, a: usize, l: usize| (j * xs + a) * mus + l;
        let count = self.degrees.len();
        let mut series = std::mem::take(&mut self.series_work);
        series.resize_with(count * size, || self.zero.clone());
        for (row, request) in requests.iter().enumerate() {
            let coefficients = self.lane_coefficients(request.id.lane)?;
            let v: Vec<[Complex<T>; 4]> = request.vectors.iter().map(components).collect();
            let zero4 = || std::array::from_fn::<_, 4, _>(|_| self.zero.clone());
            // Factor coefficients per component: [1, u, u x, u^2, u^2 x, u^2 x^2, u^2 mu^2].
            let factors: Vec<[Complex<T>; 7]> = (0..4)
                .map(|c| {
                    let z = || self.zero.clone();
                    match shape {
                        Shape::Mu => {
                            let v1 = v.get(1).cloned().unwrap_or_else(zero4);
                            [v[0][c].clone(), v1[c].clone(), z(), z(), z(), z(), z()]
                        }
                        Shape::T3 => {
                            let e4 = &v[2][c];
                            [
                                v[1][c].clone(),
                                v[0][c].clone(),
                                z(),
                                borrowed_product(&request.parameters[0], e4),
                                z(),
                                z(),
                                e4.clone(),
                            ]
                        }
                        Shape::T2 => {
                            let e4 = &v[3][c];
                            [
                                v[2][c].clone(),
                                v[0][c].clone(),
                                v[1][c].clone(),
                                borrowed_product(&request.parameters[0], e4),
                                borrowed_product(&request.parameters[1], e4),
                                borrowed_product(&request.parameters[2], e4),
                                e4.clone(),
                            ]
                        }
                    }
                })
                .collect();
            for value in series.iter_mut() {
                value.set_from(&self.zero);
            }
            series[at(0, 0, 0)].set_from(&self.one);
            for i in 1..count {
                let (parent, c) = self.parents[i];
                let f = &factors[c];
                let (head, tail) = series.split_at_mut(i * size);
                let source = &head[parent * size..(parent + 1) * size];
                let target = &mut tail[..size];
                for j in 0..=depth {
                    for a in 0..xs.min(j + 1) {
                        for l in 0..mus {
                            if a + 2 * l > j {
                                break;
                            }
                            let mut acc = borrowed_product(&f[0], &source[at(j, a, l)]);
                            if j >= 1 {
                                acc += borrowed_product(&f[1], &source[at(j - 1, a, l)]);
                                if shape == Shape::T2 && a >= 1 {
                                    acc += borrowed_product(&f[2], &source[at(j - 1, a - 1, l)]);
                                }
                            }
                            if j >= 2 && shape != Shape::Mu {
                                acc += borrowed_product(&f[3], &source[at(j - 2, a, l)]);
                                if shape == Shape::T2 {
                                    if a >= 1 {
                                        acc += borrowed_product(&f[4], &source[at(j - 2, a - 1, l)]);
                                    }
                                    if a >= 2 {
                                        acc += borrowed_product(&f[5], &source[at(j - 2, a - 2, l)]);
                                    }
                                }
                                if l >= 1 {
                                    acc += borrowed_product(&f[6], &source[at(j - 2, a, l - 1)]);
                                }
                            }
                            target[at(j, a, l)] = acc;
                        }
                    }
                }
            }
            let out = &mut output[row * columns..(row + 1) * columns];
            match shape {
                Shape::Mu => {
                    // mu^2 = t^2 G(u), G = v0^2 - beta u^2; powers G^k truncated at depth.
                    let g0 = minkowski(&v[0], &v[0]);
                    let beta = if shifted {
                        request.parameters[0].clone()
                    } else {
                        self.zero.clone()
                    };
                    let mut g_powers = vec![{
                        let mut s = vec![self.zero.clone(); depth + 1];
                        s[0] = self.one.clone();
                        s
                    }];
                    for k in 1..=self.mu_powers {
                        let previous = &g_powers[k - 1];
                        let mut s = vec![self.zero.clone(); depth + 1];
                        for j in 0..=depth {
                            let mut acc = borrowed_product(&g0, &previous[j]);
                            if j >= 2 {
                                acc -= borrowed_product(&beta, &previous[j - 2]);
                            }
                            s[j] = acc;
                        }
                        g_powers.push(s);
                    }
                    for (column, &(p, _, _)) in powers.iter().enumerate() {
                        let mut acc = self.zero.clone();
                        for (k, g) in g_powers.iter().enumerate() {
                            let (offset, block) = self.block(k);
                            for i in 0..block {
                                let c = &coefficients[offset + i];
                                if c.is_fully_zero() {
                                    continue;
                                }
                                let j = (self.degrees[i] + 2 * k) as i64 - p;
                                if j < 0 || j as usize > depth {
                                    continue;
                                }
                                let j = j as usize;
                                let s = &series[i * size..(i + 1) * size];
                                let mut term = self.zero.clone();
                                for j1 in 0..=j {
                                    term += borrowed_product(&s[at(j1, 0, 0)], &g[j - j1]);
                                }
                                acc += term * c;
                            }
                        }
                        out[column] = acc;
                    }
                }
                Shape::T3 | Shape::T2 => {
                    for (column, &(p, a, total_mu)) in powers.iter().enumerate() {
                        let mut acc = self.zero.clone();
                        for k in 0..=self.mu_powers.min(total_mu) {
                            let l = total_mu - k;
                            if l >= mus || a >= xs {
                                continue;
                            }
                            let (offset, block) = self.block(k);
                            for i in 0..block {
                                let c = &coefficients[offset + i];
                                if c.is_fully_zero() {
                                    continue;
                                }
                                let j = self.degrees[i] as i64 - p;
                                if j < 0 || j as usize > depth || a + 2 * l > j as usize {
                                    continue;
                                }
                                acc += borrowed_product(&series[i * size + at(j as usize, a, l)], c);
                            }
                        }
                        out[column] = acc;
                    }
                }
            }
        }
        self.series_work = series;
        Ok(())
    }
}

#[cfg(test)]
#[path = "tensor_dense_reference.rs"]
mod dense_reference;

#[cfg(test)]
#[path = "tensor_allocation_tests.rs"]
mod allocation_tests;

#[cfg(test)]
#[path = "tensor_capture_tests.rs"]
mod capture_tests;
