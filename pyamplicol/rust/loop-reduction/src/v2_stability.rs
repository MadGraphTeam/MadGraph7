// SPDX-License-Identifier: 0BSD
//! Binary64 stability monitoring of v2 one-loop evaluations (m-lit PLAN §3.5,
//! the revision-2c ruling; REQUIREMENTS_ADDENDUM C): ONE implementation of the
//! estimate's assembly, called by the Python API (through PyO3, in binary64)
//! and by the native SDKs (`v2_runtime::V2Runtime`). The Python assembly in
//! `loops/v2/stability.py` (and the budgets of `assemble.py` / `induced.py`)
//! stays as the oracle: every function here is bitwise the Python function it
//! names, operation by operation.
//!
//! * [`Rotation`]: an exact rational orthogonal 3x3 matrix
//!   (`stability.Rotation`), with [`Rotation::about`], [`Rotation::from_axis_angle`],
//!   [`Rotation::from_euler`], [`Rotation::parse`] (`virtual_results.parse_rotation`)
//!   and [`default_rotation`] (the `k`-th default rotation check), whose
//!   floating-point steps are Python's (the quaternion rounded with
//!   `Fraction.limit_denominator`, ported exactly: [`limit_denominator`]).
//! * [`StabilityChecks`]: which checks run (`stability.StabilityChecks`).
//! * [`lane_budget`], [`rotation_lanes`], [`retest_deviations`]: the
//!   Born-interfered budgets (`assemble._PointEvaluation._lane_budget`,
//!   `rotation_budget`, `family_deviations`), per sector and Laurent
//!   coefficient.
//! * [`Estimate::from_checks`]: the combination
//!   `E = SAFETY (max_rotations + sum_additive) + FLOOR_UNITS u S`
//!   (`StabilityEstimate.from_checks`), with [`relative`] and
//!   [`per_coefficient`].

use rusticol_core::{RusticolError, RusticolResult};
use symbolica::domains::integer::Integer;
use symbolica::domains::rational::Rational;

/// `stability.SAFETY`, `stability.FLOOR_UNITS`.
pub const SAFETY: f64 = 2.0;
pub const FLOOR_UNITS: f64 = 16.0;
/// The check names (`stability.ROTATION`, `REDUCTION`, `RESCALED`, `CROSS`,
/// `TRANSVERSE`).
pub const REDUCTION: &str = "reduction-direction";
pub const RESCALED: &str = "reduction-rescaled";
pub const CROSS: &str = "cross-backend";
pub const TRANSVERSE: &str = "spin-transverse";
/// The budgets that add up (`stability.ADDITIVE`); the rotations' is their
/// largest.
pub const ADDITIVE: [&str; 4] = [REDUCTION, RESCALED, TRANSVERSE, CROSS];
/// `stability.ROTATION.format(k)`.
pub fn rotation_name(k: usize) -> String {
    format!("rotation-{k}")
}
/// The loop factor denominator of the lane budgets
/// (`conventions.LOOP_FACTOR_DENOMINATOR`).
const LOOP_FACTOR_DENOMINATOR: f64 = 16.0;
/// `Rotation.from_axis_angle`'s and `Rotation.from_euler`'s default
/// resolutions.
pub const AXIS_ANGLE_RESOLUTION: u64 = 1_000_000;
pub const EULER_RESOLUTION: u64 = 1_000_000_000_000;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message.into())
}

/// Python's `max` of two floats: `a` unless `b > a` (NaN never replaces,
/// and a NaN `a` stays).
fn py_max(a: f64, b: f64) -> f64 {
    if b > a { b } else { a }
}

/// Python's `max(iterable)` of floats (the first maximal element; `None` for
/// an empty iterable).
fn py_max_of(values: impl IntoIterator<Item = f64>) -> Option<f64> {
    let mut values = values.into_iter();
    let first = values.next()?;
    Some(values.fold(first, py_max))
}

// ------------------------------------------------------------------ rotations

/// A spatial rotation as an exact rational orthogonal 3x3 matrix
/// (`stability.Rotation`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rotation {
    matrix: [[Rational; 3]; 3],
}

fn r(n: i64) -> Rational {
    Rational::from(n)
}

impl Rotation {
    /// The rotation of `matrix`, refused unless exactly orthogonal with
    /// determinant +1 (`Rotation.__post_init__`).
    pub fn new(matrix: [[Rational; 3]; 3]) -> RusticolResult<Self> {
        let mut identity = true;
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = r(0);
                for row in &matrix {
                    sum = &sum + &(&row[i] * &row[j]);
                }
                identity &= sum == r(i64::from(i == j));
            }
        }
        if !identity || determinant(&matrix) != r(1) {
            return Err(invalid(
                "a rotation matrix must be exactly orthogonal, det +1",
            ));
        }
        Ok(Self { matrix })
    }

    /// The matrix, row by row.
    pub fn matrix(&self) -> &[[Rational; 3]; 3] {
        &self.matrix
    }

    /// Rotation about `axis` ('x', 'y' or 'z') with exact `cos` and `sin`
    /// (`Rotation.about`); cos^2 + sin^2 must be exactly 1.
    pub fn about(axis: &str, cos: Rational, sin: Rational) -> RusticolResult<Self> {
        if &(&cos * &cos) + &(&sin * &sin) != r(1) {
            return Err(invalid("cos^2 + sin^2 must be exactly 1"));
        }
        let index = match axis {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            _ => return Err(invalid(format!("unknown rotation axis {axis:?}"))),
        };
        let others: Vec<usize> = (0..3).filter(|k| *k != index).collect();
        let (a, b) = (others[0], others[1]);
        let mut rows: [[Rational; 3]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| r(i64::from(i == j))));
        rows[a][a] = cos.clone();
        rows[a][b] = -sin.clone();
        rows[b][a] = sin;
        rows[b][b] = cos;
        Self::new(rows)
    }

    /// An exact rotation within ~1/`resolution` of `angle` (radians,
    /// right-handed) about the direction `axis` (`Rotation.from_axis_angle`):
    /// the unit quaternion (cos(angle/2), sin(angle/2) n) rounded to rationals
    /// with denominators up to `resolution`.
    pub fn from_axis_angle(axis: [f64; 3], angle: f64, resolution: u64) -> RusticolResult<Self> {
        // sum(float(x) ** 2 for x in axis): 0 + x0^2 + x1^2 + x2^2.
        let norm = axis.iter().fold(0.0_f64, |sum, x| sum + x.powf(2.0)).sqrt();
        if norm == 0.0 || !angle.is_finite() {
            return Err(invalid(
                "a rotation needs a non-zero 3-axis and a finite angle",
            ));
        }
        let half = angle / 2.0;
        let sin = half.sin();
        let values = [
            half.cos(),
            sin * axis[0] / norm,
            sin * axis[1] / norm,
            sin * axis[2] / norm,
        ];
        let quaternion = values
            .iter()
            .map(|value| {
                let exact = Rational::try_from(*value).map_err(invalid)?;
                Ok(limit_denominator(&exact, resolution))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        Self::from_quaternion(&quaternion)
    }

    /// The exact rotation closest to Rz(gamma) Ry(beta) Rx(alpha) (radians,
    /// active right-handed rotations: Rx acts first; `Rotation.from_euler`):
    /// the unit quaternion qz(gamma) qy(beta) qx(alpha) rounded once to
    /// integers over the common denominator `resolution`.
    pub fn from_euler(alpha: f64, beta: f64, gamma: f64, resolution: u64) -> RusticolResult<Self> {
        if ![alpha, beta, gamma].iter().all(|x| x.is_finite()) {
            return Err(invalid("Euler angles must be finite"));
        }
        type Q = [f64; 4];
        let half = |angle: f64, axis: usize| -> Q {
            let mut q = [(angle / 2.0).cos(), 0.0, 0.0, 0.0];
            q[axis] = (angle / 2.0).sin();
            q
        };
        // Hamilton's product, Python's expression order.
        let product = |p: Q, q: Q| -> Q {
            let [a1, b1, c1, d1] = p;
            let [a2, b2, c2, d2] = q;
            [
                a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2,
                a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2,
                a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2,
                a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2,
            ]
        };
        let quaternion = product(product(half(gamma, 3), half(beta, 2)), half(alpha, 1));
        // round(x * resolution): the float product, rounded half to even.
        let scale = resolution as f64;
        let rounded = quaternion
            .iter()
            .map(|x| {
                let value = (x * scale).round_ties_even();
                Rational::try_from(value).map_err(invalid)
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        Self::from_quaternion(&rounded)
    }

    /// The Euler-Rodrigues matrix of a rational quaternion (a, b, c, d),
    /// exactly orthogonal for any nonzero quaternion
    /// (`Rotation._from_quaternion`).
    fn from_quaternion(q: &[Rational]) -> RusticolResult<Self> {
        let (a, b, c, d) = (&q[0], &q[1], &q[2], &q[3]);
        let n = &(&(&(a * a) + &(b * b)) + &(c * c)) + &(d * d);
        if n.is_zero() {
            return Err(invalid(
                "the rounded quaternion vanishes; raise the resolution",
            ));
        }
        let two = r(2);
        let sq = |x: &Rational| x * x;
        let rows = [
            [
                &(&(&sq(a) + &sq(b)) - &sq(c)) - &sq(d),
                &two * &(&(b * c) - &(a * d)),
                &two * &(&(b * d) + &(a * c)),
            ],
            [
                &two * &(&(b * c) + &(a * d)),
                &(&(&sq(a) - &sq(b)) + &sq(c)) - &sq(d),
                &two * &(&(c * d) - &(a * b)),
            ],
            [
                &two * &(&(b * d) - &(a * c)),
                &two * &(&(c * d) + &(a * b)),
                &(&(&sq(a) - &sq(b)) - &sq(c)) + &sq(d),
            ],
        ];
        Self::new(rows.map(|row| row.map(|x| &x / &n)))
    }

    /// One rotation check from text (`virtual_results.parse_rotation`):
    /// `"<x|y|z>:<cos>:<sin>"` (exact rationals), `"<ax>,<ay>,<az>:<radians>"`
    /// ([`Rotation::from_axis_angle`]) or `"euler:<alpha>,<beta>,<gamma>"`
    /// ([`Rotation::from_euler`]).
    pub fn parse(specification: &str) -> RusticolResult<Self> {
        let parts: Vec<&str> = specification.split(':').collect();
        if parts.len() == 3 && matches!(parts[0], "x" | "y" | "z") {
            return Self::about(
                parts[0],
                parse_fraction(parts[1])?,
                parse_fraction(parts[2])?,
            );
        }
        if parts.len() == 2 && parts[0] == "euler" {
            let angles = parts[1]
                .split(',')
                .map(parse_float)
                .collect::<RusticolResult<Vec<_>>>()?;
            if angles.len() != 3 {
                return Err(invalid(format!(
                    "rotation {} needs three Euler angles",
                    py_repr(specification)
                )));
            }
            return Self::from_euler(angles[0], angles[1], angles[2], EULER_RESOLUTION);
        }
        if parts.len() == 2 {
            let axis = parts[0]
                .split(',')
                .map(parse_float)
                .collect::<RusticolResult<Vec<_>>>()?;
            let angle = parse_float(parts[1])?;
            let axis: [f64; 3] = axis
                .try_into()
                .map_err(|_| invalid("a rotation needs a non-zero 3-axis and a finite angle"))?;
            return Self::from_axis_angle(axis, angle, AXIS_ANGLE_RESOLUTION);
        }
        Err(invalid(format!(
            "rotation {} is neither '<x|y|z>:<cos>:<sin>', \
             '<ax>,<ay>,<az>:<radians>' nor 'euler:<alpha>,<beta>,<gamma>'",
            py_repr(specification)
        )))
    }

    pub fn is_identity(&self) -> bool {
        (0..3).all(|i| (0..3).all(|j| self.matrix[i][j] == r(i64::from(i == j))))
    }

    /// Rotate the spatial parts of every momentum (E, px, py, pz) exactly
    /// (`Rotation.apply`).
    pub fn apply(&self, point: &[Vec<Rational>]) -> RusticolResult<Vec<Vec<Rational>>> {
        point
            .iter()
            .map(|momentum| {
                if momentum.len() != 4 {
                    return Err(invalid("a momentum has four components"));
                }
                let mut out = Vec::with_capacity(4);
                out.push(momentum[0].clone());
                for row in &self.matrix {
                    let mut sum = r(0);
                    for (m, x) in row.iter().zip(&momentum[1..]) {
                        sum = &sum + &(m * x);
                    }
                    out.push(sum);
                }
                Ok(out)
            })
            .collect()
    }
}

/// Python's `repr` of a plain text (single quotes unless it holds one).
fn py_repr(text: &str) -> String {
    if text.contains('\'') && !text.contains('"') {
        format!("\"{text}\"")
    } else {
        format!("'{}'", text.replace('\\', "\\\\").replace('\'', "\\'"))
    }
}

fn determinant(m: &[[Rational; 3]; 3]) -> Rational {
    let minor =
        |a: usize, b: usize, c: usize, d: usize| &(&m[1][a] * &m[2][b]) - &(&m[1][c] * &m[2][d]);
    &(&(&m[0][0] * &minor(1, 2, 2, 1)) - &(&m[0][1] * &minor(0, 2, 2, 0)))
        + &(&m[0][2] * &minor(0, 1, 1, 0))
}

/// The `index`-th default rotation check (`stability.default_rotation`):
/// axes on a spherical Fibonacci sequence and angles golden-ratio steps of a
/// turn, rounded to an exact rotation ([`Rotation::from_axis_angle`]); every
/// floating-point step is Python's.
pub fn default_rotation(index: u64) -> RusticolResult<Rotation> {
    let golden = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let i = index as f64;
    // (2 * index + 1) / (2 * (index + 2)): Python's int true division.
    let z = 1.0 - (2.0 * i + 1.0) / (2.0 * (i + 2.0));
    let tau = 2.0 * std::f64::consts::PI;
    let phi = tau * i / golden;
    let radius = (1.0 - z * z).sqrt();
    let angle = tau * (((i + 1.0) / golden) % 1.0);
    Rotation::from_axis_angle(
        [radius * phi.cos(), radius * phi.sin(), z],
        py_max(angle, 0.5),
        AXIS_ANGLE_RESOLUTION,
    )
}

/// `Fraction.limit_denominator` (CPython 3.11): the closest rational to
/// `value` with a denominator at most `max_denominator`.
pub fn limit_denominator(value: &Rational, max_denominator: u64) -> Rational {
    let limit = Integer::from(max_denominator as i64);
    let denominator = value.denominator();
    if denominator <= limit {
        return value.clone();
    }
    let (mut p0, mut q0, mut p1, mut q1) = (
        Integer::from(0),
        Integer::from(1),
        Integer::from(1),
        Integer::from(0),
    );
    let (mut n, mut d) = (value.numerator(), denominator);
    loop {
        // Python's floor division (d > 0: the Euclidean quotient).
        let (a, rest) = n.quot_rem(&d);
        let q2 = &q0 + &(&a * &q1);
        if q2 > limit {
            break;
        }
        let p2 = &p0 + &(&a * &p1);
        (p0, q0, p1, q1) = (p1, q1, p2, q2);
        (n, d) = (d, rest);
    }
    let (k, _) = (&limit - &q0).quot_rem(&q1);
    let bound1 = Rational::from((&p0 + &(&k * &p1), &q0 + &(&k * &q1)));
    let bound2 = Rational::from((p1, q1));
    if (&bound2 - value).abs() <= (&bound1 - value).abs() {
        bound2
    } else {
        bound1
    }
}

/// A rational from Python `Fraction` text: an optional sign, then digits
/// (`_` between digits), with an optional `/denominator`, or a decimal with an
/// optional fractional part and exponent; surrounding whitespace allowed.
pub fn parse_fraction(text: &str) -> RusticolResult<Rational> {
    let refused = || invalid(format!("Invalid literal for Fraction: {text:?}"));
    let body = text.trim();
    let (negative, body) = match body.as_bytes().first() {
        Some(b'-') => (true, &body[1..]),
        Some(b'+') => (false, &body[1..]),
        _ => (false, body),
    };
    let digits = |part: &str, empty: bool| -> Option<String> {
        if part.is_empty() {
            return empty.then(String::new);
        }
        let bytes = part.as_bytes();
        let ok = bytes.iter().enumerate().all(|(i, c)| {
            c.is_ascii_digit()
                || (*c == b'_'
                    && i > 0
                    && i + 1 < bytes.len()
                    && bytes[i - 1].is_ascii_digit()
                    && bytes[i + 1].is_ascii_digit())
        });
        ok.then(|| part.replace('_', ""))
    };
    let integer =
        |s: &str| -> RusticolResult<Integer> { s.parse::<Integer>().map_err(|_| refused()) };
    let value = if let Some((num, den)) = body.split_once('/') {
        let num = digits(num, false).ok_or_else(refused)?;
        let den = digits(den, false).ok_or_else(refused)?;
        let den = integer(&den)?;
        if den.is_zero() {
            return Err(invalid(format!("Fraction({num}, 0)")));
        }
        Rational::from((integer(&num)?, den))
    } else {
        let (mantissa, exponent) = match body.find(['e', 'E']) {
            Some(at) => (&body[..at], Some(&body[at + 1..])),
            None => (body, None),
        };
        let (whole, fraction) = match mantissa.split_once('.') {
            Some((whole, fraction)) => (whole, fraction),
            None => (mantissa, ""),
        };
        let whole = digits(whole, true).ok_or_else(refused)?;
        let fraction = digits(fraction, true).ok_or_else(refused)?;
        // (?=\d|\.\d): a digit before the point or right after it.
        if whole.is_empty() && fraction.is_empty() {
            return Err(refused());
        }
        let exponent: i64 = match exponent {
            Some(e) => {
                let (sign, e) = match e.as_bytes().first() {
                    Some(b'-') => (-1, &e[1..]),
                    Some(b'+') => (1, &e[1..]),
                    _ => (1, e),
                };
                let e = digits(e, false).ok_or_else(refused)?;
                sign * e.parse::<i64>().map_err(|_| refused())?
            }
            None => 0,
        };
        let mantissa = Rational::from(integer(&format!("{whole}{fraction}"))?);
        let power = exponent - fraction.len() as i64;
        let ten = Rational::from(Integer::from(10).pow(power.unsigned_abs()));
        if power >= 0 {
            &mantissa * &ten
        } else {
            &mantissa / &ten
        }
    };
    Ok(if negative { -value } else { value })
}

/// A float from Python `float` text (surrounding whitespace, `_` between
/// digits).
fn parse_float(text: &str) -> RusticolResult<f64> {
    let trimmed = text.trim();
    let bytes = trimmed.as_bytes();
    let underscores_ok = bytes.iter().enumerate().all(|(i, c)| {
        *c != b'_'
            || (i > 0
                && i + 1 < bytes.len()
                && bytes[i - 1].is_ascii_digit()
                && bytes[i + 1].is_ascii_digit())
    });
    let cleaned = trimmed.replace('_', "");
    match (underscores_ok, cleaned.parse::<f64>()) {
        (true, Ok(value)) => Ok(value),
        _ => Err(invalid(format!(
            "could not convert string to float: {text:?}"
        ))),
    }
}

// ------------------------------------------------------------------- checks

/// The re-reduction frame of the reduction check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Frame {
    /// The loop direction reversed, cached masters (`"reversed"`).
    Reversed,
    /// The reversed frame at rescaled inputs, masters recomputed
    /// (`"rescaled"`).
    Rescaled,
}

/// Which checks an evaluation runs (`stability.StabilityChecks`): rotations
/// (any number), the reduction check (reversed, or rescaled) and the
/// cross-backend check (with at least one other). The default runs none.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StabilityChecks {
    rotations: Vec<Rotation>,
    reduction_direction: bool,
    rescaled: bool,
    cross_backend: bool,
}

impl StabilityChecks {
    /// The checks, validated as `StabilityChecks.__post_init__`.
    pub fn new(
        rotations: Vec<Rotation>,
        reduction_direction: bool,
        rescaled: bool,
        cross_backend: bool,
    ) -> RusticolResult<Self> {
        if rotations.iter().any(Rotation::is_identity) {
            return Err(invalid("the identity is no rotation check"));
        }
        if rescaled && !reduction_direction {
            return Err(invalid(
                "rescaled is a variant of the reduction-direction check",
            ));
        }
        if cross_backend && rotations.is_empty() && !reduction_direction {
            return Err(invalid(
                "the cross-backend check adds to the estimate of the other checks \
                 (a point the other backend does not serve keeps theirs): combine it \
                 with a rotation or the reduction-direction check",
            ));
        }
        Ok(Self {
            rotations,
            reduction_direction,
            rescaled,
            cross_backend,
        })
    }

    /// `count` default rotations ([`default_rotation`]) and the other checks
    /// (`VirtualStabilityChecks(count, ...)`).
    pub fn with_default_rotations(
        count: u64,
        reduction_direction: bool,
        rescaled: bool,
        cross_backend: bool,
    ) -> RusticolResult<Self> {
        let rotations = (0..count)
            .map(default_rotation)
            .collect::<RusticolResult<_>>()?;
        Self::new(rotations, reduction_direction, rescaled, cross_backend)
    }

    pub fn rotations(&self) -> &[Rotation] {
        &self.rotations
    }
    pub fn reduction_direction(&self) -> bool {
        self.reduction_direction
    }
    pub fn rescaled(&self) -> bool {
        self.rescaled
    }
    pub fn cross_backend(&self) -> bool {
        self.cross_backend
    }
    /// Whether any check runs (an estimate is returned).
    pub fn enabled(&self) -> bool {
        !self.rotations.is_empty() || self.reduction_direction
    }
    /// The name of the reduction check that runs, if any.
    pub fn reduction(&self) -> Option<&'static str> {
        self.reduction_direction
            .then_some(if self.rescaled { RESCALED } else { REDUCTION })
    }
    /// The re-reduction frame of the reduction check, if any.
    pub fn frame(&self) -> Option<Frame> {
        self.reduction_direction.then_some(if self.rescaled {
            Frame::Rescaled
        } else {
            Frame::Reversed
        })
    }
}

// ------------------------------------------------------------------ budgets

/// Absolute errors per sector and Laurent coefficient (a `Budget` per
/// sector).
pub type Budget = Vec<Vec<f64>>;

/// The factor of a lane's |delta Omega| in the bare virtual's budget,
/// 2 / (16 pi^2 average) (`_PointEvaluation._lane_budget`).
pub fn lane_factor(average: f64) -> f64 {
    2.0 / (LOOP_FACTOR_DENOMINATOR * std::f64::consts::PI.powf(2.0) * average)
}

/// `_PointEvaluation._lane_budget`: per sector (of `sectors`, `width`
/// Laurent coefficients) sum over the lanes (sector, absolute errors per
/// coefficient), in their order, of `factor x error`, plus the absolute
/// local components (per component, per sector, the Laurent values), in
/// their order.
pub fn lane_budget<'a>(
    sectors: usize,
    width: usize,
    factor: f64,
    lanes: impl IntoIterator<Item = (usize, &'a [f64])>,
    local: &[Budget],
) -> RusticolResult<Budget> {
    let mut budget = vec![vec![0.0_f64; width]; sectors];
    for (sector, errors) in lanes {
        let cell = budget
            .get_mut(sector)
            .ok_or_else(|| invalid("a lane outside the sectors"))?;
        if errors.len() > width {
            return Err(invalid("a lane wider than the Laurent coefficients"));
        }
        for (k, error) in errors.iter().enumerate() {
            cell[k] += factor * error.abs();
        }
    }
    for component in local {
        for (sector, laurent) in component.iter().enumerate() {
            let cell = budget
                .get_mut(sector)
                .ok_or_else(|| invalid("a local component outside the sectors"))?;
            for (k, value) in laurent.iter().enumerate() {
                cell[k] += value.abs();
            }
        }
    }
    Ok(budget)
}

/// |z| of a binary64 complex value (Python's `abs(complex)`, libm `hypot`).
pub fn modulus(re: f64, im: f64) -> f64 {
    re.hypot(im)
}

/// One lane of a point: its identity (matched between two evaluations of the
/// same point), its sector and its Laurent coefficients (re, im).
pub type Lane<K> = (K, usize, Vec<(f64, f64)>);

/// The lanes of a rotation check (`_PointEvaluation.rotation_budget`'s
/// `lanes`): over the union of both evaluations' lanes in insertion order
/// (`dict.fromkeys([*direct, *rotated])`), per lane |rotated - direct| per
/// coefficient (an absent lane is zero), as (sector, errors).
pub fn rotation_lanes<K: PartialEq + Clone>(
    direct: &[Lane<K>],
    rotated: &[Lane<K>],
    width: usize,
) -> RusticolResult<Vec<(usize, Vec<f64>)>> {
    let mut order: Vec<&Lane<K>> = Vec::with_capacity(direct.len());
    for lane in direct.iter().chain(rotated) {
        if !order.iter().any(|seen| seen.0 == lane.0) {
            order.push(lane);
        }
    }
    fn values<'v, K: PartialEq>(
        lanes: &'v [Lane<K>],
        key: &K,
        width: usize,
    ) -> RusticolResult<Option<&'v [(f64, f64)]>> {
        match lanes.iter().find(|lane| lane.0 == *key) {
            None => Ok(None),
            Some(lane) if lane.2.len() == width => Ok(Some(lane.2.as_slice())),
            Some(_) => Err(invalid("a lane of another Laurent width")),
        }
    }
    order
        .into_iter()
        .map(|lane| {
            let a = values(rotated, &lane.0, width)?;
            let b = values(direct, &lane.0, width)?;
            let errors = (0..width)
                .map(|k| {
                    // `rotated.get(lane, zero) - direct.get(lane, zero)`: a
                    // real zero becomes the complex (0.0, 0.0).
                    let (ar, ai) = a.map_or((0.0, 0.0), |v| v[k]);
                    let (br, bi) = b.map_or((0.0, 0.0), |v| v[k]);
                    modulus(ar - br, ai - bi)
                })
                .collect();
            Ok((lane.1, errors))
        })
        .collect()
}

/// The re-test's per-lane deviations (`family_deviations`): per lane of
/// `direct`, in its order, |direct - retest| per coefficient; a lane absent
/// from the re-test is refused (Python's `KeyError`).
pub fn retest_deviations<K: PartialEq>(
    direct: &[Lane<K>],
    retest: &[Lane<K>],
) -> RusticolResult<Vec<(usize, Vec<f64>)>> {
    direct
        .iter()
        .map(|(key, sector, values)| {
            let other = retest
                .iter()
                .find(|lane| lane.0 == *key)
                .ok_or_else(|| invalid("a lane is missing from the re-test"))?;
            if other.2.len() != values.len() {
                return Err(invalid("a re-test lane of another width"));
            }
            let errors = values
                .iter()
                .zip(&other.2)
                .map(|((ar, ai), (br, bi))| modulus(ar - br, ai - bi))
                .collect();
            Ok((*sector, errors))
        })
        .collect()
}

/// |b - a| per sector and coefficient of two real per-sector Laurent values
/// (`stability.differences(a, b)`).
pub fn differences(a: &Budget, b: &Budget) -> Budget {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.iter().zip(y).map(|(x, y)| (y - x).abs()).collect())
        .collect()
}

// ------------------------------------------------------------------ estimate

/// Largest absolute deviation relative to each sector's largest Laurent
/// coefficient (`stability.relative`).
pub fn relative(reference: &Budget, absolute: &Budget) -> f64 {
    let mut worst = 0.0_f64;
    for (laurent, errors) in reference.iter().zip(absolute) {
        let scale = py_max_of(laurent.iter().map(|x| x.abs())).unwrap_or(0.0);
        if scale != 0.0 {
            let error = py_max_of(errors.iter().map(|x| x.abs())).unwrap_or(0.0);
            worst = py_max(worst, error / scale);
        }
    }
    worst
}

/// Largest absolute deviation of each Laurent coefficient relative to the
/// coefficient itself over sectors (`stability.per_coefficient`; `inf` for a
/// deviation of an exact 0).
pub fn per_coefficient(reference: &Budget, absolute: &Budget) -> Vec<f64> {
    let width = reference.first().map_or(0, Vec::len);
    let mut worst = vec![0.0_f64; width];
    for (laurent, errors) in reference.iter().zip(absolute) {
        for (k, (value, error)) in laurent.iter().zip(errors).enumerate() {
            let (value, error) = (value.abs(), error.abs());
            if error != 0.0 {
                worst[k] = py_max(
                    worst[k],
                    if value != 0.0 {
                        error / value
                    } else {
                        f64::INFINITY
                    },
                );
            }
        }
    }
    worst
}

/// The relative accuracy estimate of one evaluation
/// (`stability.StabilityEstimate`).
#[derive(Clone, Debug, PartialEq)]
pub struct Estimate {
    /// E against each sector's largest Laurent coefficient.
    pub relative: f64,
    /// E per Laurent coefficient against that coefficient.
    pub coefficients: Vec<f64>,
    /// Each check's own budget against the Laurent vector, in check order.
    pub checks: Vec<(String, f64)>,
    /// E per sector and Laurent coefficient.
    pub absolute: Budget,
    /// The absolute budget of every check, in check order.
    pub budgets: Vec<(String, Budget)>,
    /// The absolute sums S per sector and Laurent coefficient.
    pub scale: Budget,
    /// The carrier's unit roundoff.
    pub roundoff: f64,
    /// Each requested check that did not run, with its reason.
    pub not_run: Vec<(String, String)>,
}

impl Estimate {
    /// `StabilityEstimate.from_checks`: combine the absolute budgets of named
    /// checks for the `reference` Laurent values per sector; `scale` the
    /// absolute sums S, `roundoff` the unit roundoff.
    pub fn from_checks(
        reference: &Budget,
        budgets: Vec<(String, Budget)>,
        scale: Budget,
        roundoff: f64,
        not_run: Vec<(String, String)>,
    ) -> RusticolResult<Self> {
        if budgets.is_empty() {
            return Err(invalid("a stability estimate needs at least one check"));
        }
        let unknown: Vec<&str> = budgets
            .iter()
            .map(|(name, _)| name.as_str())
            .filter(|name| !ADDITIVE.contains(name) && !name.starts_with("rotation-"))
            .collect();
        if !unknown.is_empty() {
            return Err(invalid(format!("unknown stability checks {unknown:?}")));
        }
        let additive = |name: &str| ADDITIVE.contains(&name);
        let at = |budget: &Budget, sector: usize, k: usize| -> RusticolResult<f64> {
            budget
                .get(sector)
                .and_then(|row| row.get(k))
                .map(|x| x.abs())
                .ok_or_else(|| invalid("a budget without the reference's coefficients"))
        };
        let mut combined = Vec::with_capacity(reference.len());
        for (sector, laurent) in reference.iter().enumerate() {
            let mut row = Vec::with_capacity(laurent.len());
            for k in 0..laurent.len() {
                let mut rotated: Option<f64> = None;
                let mut reductions = 0.0_f64;
                for (name, budget) in &budgets {
                    let value = at(budget, sector, k)?;
                    if additive(name) {
                        reductions += value;
                    } else {
                        rotated = Some(rotated.map_or(value, |r| py_max(r, value)));
                    }
                }
                let sampled = rotated.unwrap_or(0.0) + reductions;
                row.push(SAFETY * sampled + FLOOR_UNITS * roundoff * at(&scale, sector, k)?);
            }
            combined.push(row);
        }
        let checks = budgets
            .iter()
            .map(|(name, budget)| (name.clone(), relative(reference, budget)))
            .collect();
        Ok(Self {
            relative: relative(reference, &combined),
            coefficients: per_coefficient(reference, &combined),
            checks,
            absolute: combined,
            budgets,
            scale,
            roundoff,
            not_run,
        })
    }
}

/// The inputs of one Born-interfered evaluation of a point that its
/// budgets read: its reduction lanes and its local components (per
/// component `r2`, `uv_local`, fields (`assemble.LOCAL_COMPONENTS`), per
/// sector the Laurent values; empty without native counterterm components).
#[derive(Clone, Debug, Default)]
pub struct PointLanes<K> {
    pub lanes: Vec<Lane<K>>,
    pub local: Vec<Budget>,
}

/// The cross-backend check of a point: its evaluation on the other backend,
/// or why the check did not run there (`stability.CrossRuns.of`).
pub enum CrossRun<P> {
    Run(P),
    NotRun(String),
}

/// The Born-interfered budgets' common input: the slice's sectors, the
/// Laurent width and the lanes' factor ([`lane_factor`]).
#[derive(Clone, Copy, Debug)]
pub struct LaneShape {
    pub sectors: usize,
    pub width: usize,
    pub factor: f64,
}

impl LaneShape {
    fn budget<'a>(
        self,
        lanes: impl IntoIterator<Item = (usize, &'a [f64])>,
        local: &[Budget],
    ) -> RusticolResult<Budget> {
        lane_budget(self.sectors, self.width, self.factor, lanes, local)
    }

    /// The absolute sums S of a point (`_PointEvaluation.absolute`): every
    /// lane's |Omega| and the absolute local components.
    pub fn absolute<K>(self, point: &PointLanes<K>) -> RusticolResult<Budget> {
        let lanes: Vec<(usize, Vec<f64>)> = point
            .lanes
            .iter()
            .map(|(_, sector, values)| {
                (
                    *sector,
                    values.iter().map(|(re, im)| modulus(*re, *im)).collect(),
                )
            })
            .collect();
        self.budget(lanes.iter().map(|(s, e)| (*s, e.as_slice())), &point.local)
    }

    /// The budget of one rotation check (or of the cross-backend check, the
    /// same comparison at the unrotated point): every lane and local
    /// component is invariant (`_PointEvaluation.rotation_budget`).
    pub fn rotation<K: PartialEq + Clone>(
        self,
        direct: &PointLanes<K>,
        rotated: &PointLanes<K>,
    ) -> RusticolResult<Budget> {
        let lanes = rotation_lanes(&direct.lanes, &rotated.lanes, self.width)?;
        let local: Vec<Budget> = direct
            .local
            .iter()
            .zip(&rotated.local)
            .map(|(a, b)| differences(a, b))
            .collect();
        if local.len() != direct.local.len() || direct.local.len() != rotated.local.len() {
            return Err(invalid("two evaluations with other local components"));
        }
        self.budget(lanes.iter().map(|(s, e)| (*s, e.as_slice())), &local)
    }

    /// The reduction re-test's budget (`retest_budget`) from the point's
    /// lanes and their re-test.
    pub fn retest<K: PartialEq>(
        self,
        direct: &PointLanes<K>,
        retest: &[Lane<K>],
    ) -> RusticolResult<Budget> {
        let deviations = retest_deviations(&direct.lanes, retest)?;
        self.budget(deviations.iter().map(|(s, e)| (*s, e.as_slice())), &[])
    }
}

/// The stability estimate of one Born-interfered point
/// (`plan_export.point_estimate`): the rotation checks' budgets of the
/// evaluations at the rotated points, the reduction check's re-test budget
/// (named `reduction`), the cross-backend check's (or its `not_run`
/// record), against the `reference` Laurent values per sector, with the
/// absolute sums S as the floor's scale.
#[allow(clippy::too_many_arguments)]
pub fn born_interfered_estimate<K: PartialEq + Clone>(
    shape: LaneShape,
    roundoff: f64,
    reference: &Budget,
    direct: &PointLanes<K>,
    rotated: &[PointLanes<K>],
    retest: Option<(&str, &[Lane<K>])>,
    cross: Option<&CrossRun<PointLanes<K>>>,
) -> RusticolResult<Estimate> {
    let mut budgets = Vec::with_capacity(rotated.len() + 2);
    for (k, run) in rotated.iter().enumerate() {
        budgets.push((rotation_name(k), shape.rotation(direct, run)?));
    }
    if let Some((name, lanes)) = retest {
        budgets.push((name.to_string(), shape.retest(direct, lanes)?));
    }
    let mut not_run = Vec::new();
    match cross {
        Some(CrossRun::Run(run)) => budgets.push((CROSS.to_string(), shape.rotation(direct, run)?)),
        Some(CrossRun::NotRun(why)) => not_run.push((CROSS.to_string(), why.clone())),
        None => {}
    }
    Estimate::from_checks(
        reference,
        budgets,
        shape.absolute(direct)?,
        roundoff,
        not_run,
    )
}

// ------------------------------------------------------------ loop-induced

/// A binary64 complex value (re, im), operated on as CPython's `complex`
/// (`_Py_c_sum`, `_Py_c_diff`, `_Py_c_prod`; a real operand is promoted to
/// (x, 0.0), Python 3.11's rule).
pub type Cell = (f64, f64);

fn c_add(a: Cell, b: Cell) -> Cell {
    (a.0 + b.0, a.1 + b.1)
}
fn c_sub(a: Cell, b: Cell) -> Cell {
    (a.0 - b.0, a.1 - b.1)
}
fn c_mul(a: Cell, b: Cell) -> Cell {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}
fn c_conj(a: Cell) -> Cell {
    (a.0, -a.1)
}

/// A key of a loop-induced point's vectors: (loop order, helicity row), as
/// ids of the caller's order and row tables.
pub type Key = (usize, usize);
/// Values per key in insertion order (the Python API's dicts).
pub type Keyed<V> = Vec<(Key, V)>;
/// Per basis element t the Laurent coefficients (complex).
pub type AmplitudeCells = Vec<Vec<Cell>>;
/// Per basis element t the Laurent coefficients (absolute, real).
pub type ErrorCells = Vec<Vec<f64>>;

fn lookup<'v, V>(values: &'v Keyed<V>, key: &Key) -> Option<&'v V> {
    values.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

/// The loop-induced budgets' common input: the amplitude basis size, the
/// Laurent width, the colour metric G (carrier cells, row-major), the plan's
/// helicity rows (ids, in order), per record sector its (loop order, loop
/// order) pairs as Python's `sorted(pairs)` (order ids) and the point's
/// average.
#[derive(Clone, Debug)]
pub struct InducedShape {
    pub size: usize,
    pub width: usize,
    pub gram: Vec<Cell>,
    pub rows: Vec<usize>,
    pub sectors: Vec<Vec<(usize, usize)>>,
    pub average: f64,
}

/// One binary64 evaluation of a loop-induced point (`induced.read_native` and
/// `assemble`, the native amplitude sum without raw lanes): the amplitude
/// vectors a = a_bare + A^CT, the bare vectors and the counterterm kets, with
/// a check the re-test budget (`native_budget`), with the term scale its
/// values (`native_terms`).
#[derive(Clone, Debug, Default)]
pub struct InducedPoint {
    pub amplitudes: Keyed<AmplitudeCells>,
    pub bare: Keyed<AmplitudeCells>,
    pub kets: Keyed<AmplitudeCells>,
    pub budget: Option<Keyed<ErrorCells>>,
    pub terms: Option<Keyed<ErrorCells>>,
}

impl InducedShape {
    fn metric(&self, t: usize, u: usize) -> Cell {
        self.gram[t * self.size + u]
    }

    /// ||x||_G = sqrt(x^dag G x) (`induced.g_norm` in binary64: 0 for a
    /// non-positive form; `value ** 0.5`).
    pub fn g_norm(&self, vector: &[Cell]) -> f64 {
        let mut total: Cell = (0.0, 0.0);
        for (t, x) in vector.iter().enumerate() {
            let mut inner: Cell = (0.0, 0.0);
            for (u, y) in vector.iter().enumerate() {
                inner = c_add(inner, c_mul(self.metric(t, u), *y));
            }
            total = c_add(total, c_mul(c_conj(*x), inner));
        }
        let value = total.0;
        if value <= 0.0 { 0.0 } else { value.powf(0.5) }
    }

    /// sqrt(e^T |G| e) (`induced._bound_norm`).
    pub fn bound_norm(&self, errors: &[f64]) -> f64 {
        let mut total = 0.0_f64;
        for (t, x) in errors.iter().enumerate() {
            for (u, y) in errors.iter().enumerate() {
                let (re, im) = self.metric(t, u);
                total += x * modulus(re, im) * y;
            }
        }
        total.powf(0.5)
    }

    /// `induced.sector_form`: per record sector and Laurent coefficient k,
    /// sum_pairs kappa sum_rows sum_{i+j=k} kernel(x_i, y_j) / avg, with
    /// x = left((l, row)), y = right((l', row)) (None: no term).
    fn sector_form<X>(
        &self,
        left: impl Fn(Key) -> Option<Vec<X>>,
        right: impl Fn(Key) -> Option<Vec<X>>,
        kernel: impl Fn(&X, &X) -> f64,
    ) -> Budget {
        self.sectors
            .iter()
            .map(|pairs| {
                let mut totals = vec![0.0_f64; self.width];
                for &(a, b) in pairs {
                    let weight = if a == b { 1.0 } else { 2.0 };
                    for &row in &self.rows {
                        let (Some(x), Some(y)) = (left((a, row)), right((b, row))) else {
                            continue;
                        };
                        for (k, total) in totals.iter_mut().enumerate() {
                            for i in 0..=k {
                                *total += weight * kernel(&x[i], &y[k - i]);
                            }
                        }
                    }
                }
                totals
                    .into_iter()
                    .map(|value| value / self.average)
                    .collect()
            })
            .collect()
    }

    /// The |A|^2 budget of absolute amplitude errors per key
    /// (`induced.quadratic_budget`): sum_pairs kappa sum_h sum_{i+j=k}
    /// (||a_i||_G e_j + e_i ||a_j||_G + e_i e_j) / avg with e the bound
    /// norms of the errors.
    pub fn quadratic_budget(&self, point: &InducedPoint, errors: &Keyed<ErrorCells>) -> Budget {
        let terms = |key: Key| -> Option<Vec<(f64, f64)>> {
            let vector = lookup(&point.amplitudes, &key);
            let error = lookup(errors, &key);
            Some(
                (0..self.width)
                    .map(|k| {
                        let norm = vector.map_or(0.0, |v| {
                            self.g_norm(&(0..self.size).map(|t| v[t][k]).collect::<Vec<_>>())
                        });
                        let bound = error.map_or(0.0, |e| {
                            self.bound_norm(&(0..self.size).map(|t| e[t][k]).collect::<Vec<_>>())
                        });
                        (norm, bound)
                    })
                    .collect(),
            )
        };
        self.sector_form(terms, terms, |&(a, da), &(b, db)| a * db + da * b + da * db)
    }

    /// Per helicity row the little-group phase of `rotated` against
    /// `direct` (`induced._row_phases` with the finite parts): the phase of
    /// sum conj(a') a over the row's loop orders and t (1 where it vanishes).
    pub fn row_phases(&self, direct: &InducedPoint, rotated: &InducedPoint) -> Vec<(usize, Cell)> {
        self.rows
            .iter()
            .map(|&row| {
                let mut total: Cell = (0.0, 0.0);
                for ((order, key_row), vector) in &direct.amplitudes {
                    if *key_row != row {
                        continue;
                    }
                    let Some(other) = lookup(&rotated.amplitudes, &(*order, row)) else {
                        continue;
                    };
                    for t in 0..self.size {
                        total = c_add(total, c_mul(c_conj(other[t][0]), vector[t][0]));
                    }
                }
                let magnitude = modulus(total.0, total.1);
                let phase = if magnitude == 0.0 {
                    (1.0, 0.0)
                } else {
                    c_mul(total, (1.0 / magnitude, 0.0))
                };
                (row, phase)
            })
            .collect()
    }

    /// The phase-corrected deviations of `rotated` from `direct`
    /// (`induced._rotation_deviations` of a native point without raw lanes):
    /// per key of the counterterm kets, then of the bare vectors, in their
    /// union order, |y phase - x| per t and k, summed.
    pub fn deviations(
        &self,
        direct: &InducedPoint,
        rotated: &InducedPoint,
        phases: &[(usize, Cell)],
    ) -> RusticolResult<Keyed<ErrorCells>> {
        let phase = |row: usize| -> RusticolResult<Cell> {
            phases
                .iter()
                .find(|(r, _)| *r == row)
                .map(|(_, p)| *p)
                .ok_or_else(|| invalid("a vector of a row outside the plan's rows"))
        };
        let mut errors: Keyed<ErrorCells> = Vec::new();
        for (mine, theirs) in [(&direct.kets, &rotated.kets), (&direct.bare, &rotated.bare)] {
            let mut keys: Vec<Key> = Vec::new();
            for (key, _) in mine.iter().chain(theirs.iter()) {
                if !keys.contains(key) {
                    keys.push(*key);
                }
            }
            for key in keys {
                let (a, b) = (lookup(mine, &key), lookup(theirs, &key));
                let p = phase(key.1)?;
                let at = match errors.iter().position(|(k, _)| *k == key) {
                    Some(at) => at,
                    None => {
                        errors.push((key, vec![vec![0.0; self.width]; self.size]));
                        errors.len() - 1
                    }
                };
                let cell = &mut errors[at].1;
                for t in 0..self.size {
                    for k in 0..self.width {
                        let x = a.map_or((0.0, 0.0), |v| v[t][k]);
                        let y = b.map_or((0.0, 0.0), |v| v[t][k]);
                        let d = c_sub(c_mul(y, p), x);
                        cell[t][k] += modulus(d.0, d.1);
                    }
                }
            }
        }
        Ok(errors)
    }

    /// The budget of one rotation check (`induced.rotation_budget`).
    pub fn rotation(
        &self,
        direct: &InducedPoint,
        rotated: &InducedPoint,
    ) -> RusticolResult<Budget> {
        let phases = self.row_phases(direct, rotated);
        Ok(self.quadratic_budget(direct, &self.deviations(direct, rotated, &phases)?))
    }

    /// The budget of the cross-backend check (`induced.cross_budget`): the
    /// deviations without a little-group phase.
    pub fn cross(&self, direct: &InducedPoint, other: &InducedPoint) -> RusticolResult<Budget> {
        let unit: Vec<(usize, Cell)> = self.rows.iter().map(|&row| (row, (1.0, 0.0))).collect();
        Ok(self.quadratic_budget(direct, &self.deviations(direct, other, &unit)?))
    }

    /// The re-test's budget (`induced.budgets`: `quadratic_budget` of the
    /// executor's per-component re-test deviations).
    pub fn retest(&self, point: &InducedPoint) -> RusticolResult<Budget> {
        let budget = point
            .budget
            .as_ref()
            .ok_or_else(|| invalid("the reduction check returned no amplitude budget"))?;
        Ok(self.quadratic_budget(point, budget))
    }

    /// The per-row term floors (`induced.row_scales`): per key of the term
    /// scale, then of the kets, per Laurent coefficient k, S_k = sum_t
    /// T_{t,k} sqrt(G_tt) + ||A^CT_k||_G, floored at S_0.
    pub fn row_scales(&self, point: &InducedPoint) -> RusticolResult<Keyed<Vec<f64>>> {
        let empty = Vec::new();
        let terms = match &point.terms {
            Some(terms) => terms,
            // No (loop order, row) slot: no family term to scale.
            None if point.bare.is_empty() => &empty,
            None => {
                return Err(invalid(
                    "a native loop-induced point has no term scale: the executor forms it \
                     with the stability checks (term_scale)",
                ));
            }
        };
        let roots: Vec<f64> = (0..self.size)
            .map(|t| py_max(self.metric(t, t).0, 0.0).powf(0.5))
            .collect();
        let mut keys: Vec<Key> = Vec::new();
        for key in terms
            .iter()
            .map(|(k, _)| k)
            .chain(point.kets.iter().map(|(k, _)| k))
        {
            if !keys.contains(key) {
                keys.push(*key);
            }
        }
        Ok(keys
            .into_iter()
            .map(|key| {
                let (vector, ket) = (lookup(terms, &key), lookup(&point.kets, &key));
                let values: Vec<f64> = (0..self.width)
                    .map(|k| {
                        let mut value = 0.0_f64;
                        if let Some(vector) = vector {
                            value =
                                (0..self.size).fold(0.0, |sum, t| sum + vector[t][k] * roots[t]);
                        }
                        if let Some(ket) = ket {
                            value +=
                                self.g_norm(&(0..self.size).map(|t| ket[t][k]).collect::<Vec<_>>());
                        }
                        value
                    })
                    .collect();
                let floor = values[0];
                (
                    key,
                    values
                        .into_iter()
                        .map(|value| py_max(value, floor))
                        .collect(),
                )
            })
            .collect())
    }

    /// The scale S of the estimate's floor (`induced.scale`): the |A|^2
    /// product of the per-row term floors.
    pub fn scale(&self, point: &InducedPoint) -> RusticolResult<Budget> {
        let scales = self.row_scales(point)?;
        let get = |key: Key| lookup(&scales, &key).cloned();
        Ok(self.sector_form(get, get, |a, b| a * b))
    }

    /// The point's finiteness ratio (`induced.point_finiteness` with the
    /// term scale): the largest ||a_{-k}||_G / S_k(h) over its keys and pole
    /// coefficients; None without poles.
    pub fn finiteness(&self, point: &InducedPoint) -> RusticolResult<Option<f64>> {
        if self.width == 1 {
            return Ok(None);
        }
        let scales = self.row_scales(point)?;
        let mut worst: Option<f64> = None;
        for (key, vector) in &point.amplitudes {
            let mut ratio = 0.0_f64;
            for k in 1..self.width {
                let residue =
                    self.g_norm(&(0..self.size).map(|t| vector[t][k]).collect::<Vec<_>>());
                if residue != 0.0 {
                    let bound = lookup(&scales, key).map_or(0.0, |s| s[k]);
                    ratio = py_max(
                        ratio,
                        if bound != 0.0 {
                            residue / bound
                        } else {
                            f64::INFINITY
                        },
                    );
                }
            }
            worst = Some(worst.map_or(ratio, |w| py_max(w, ratio)));
        }
        Ok(Some(worst.unwrap_or(0.0)))
    }
}

/// The stability estimate of one loop-induced point (`plan_export.
/// point_estimate` of an amplitude-mode point): its rotation checks' budgets,
/// the re-test's (named `reduction`), the cross-backend check's (or its
/// `not_run` record), against |A|^2 per sector (`reference`), with the term
/// floor's scale.
pub fn induced_estimate(
    shape: &InducedShape,
    roundoff: f64,
    reference: &Budget,
    direct: &InducedPoint,
    rotated: &[InducedPoint],
    reduction: Option<&str>,
    cross: Option<&CrossRun<InducedPoint>>,
) -> RusticolResult<Estimate> {
    let mut budgets = Vec::with_capacity(rotated.len() + 2);
    for (k, run) in rotated.iter().enumerate() {
        budgets.push((rotation_name(k), shape.rotation(direct, run)?));
    }
    if let Some(name) = reduction {
        budgets.push((name.to_string(), shape.retest(direct)?));
    }
    let mut not_run = Vec::new();
    match cross {
        Some(CrossRun::Run(run)) => budgets.push((CROSS.to_string(), shape.cross(direct, run)?)),
        Some(CrossRun::NotRun(why)) => not_run.push((CROSS.to_string(), why.clone())),
        None => {}
    }
    Estimate::from_checks(reference, budgets, shape.scale(direct)?, roundoff, not_run)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fq(n: i64, d: i64) -> Rational {
        Rational::from((n, d))
    }

    #[test]
    fn limit_denominator_matches_python() {
        // Fraction('3.141592653589793').limit_denominator(10) == 22/7, (100) == 311/99.
        let pi = parse_fraction("3.141592653589793").unwrap();
        assert_eq!(limit_denominator(&pi, 10), fq(22, 7));
        assert_eq!(limit_denominator(&pi, 100), fq(311, 99));
        assert_eq!(limit_denominator(&fq(4321, 8765), 10000), fq(4321, 8765));
        assert_eq!(limit_denominator(&fq(-4321, 8765), 100), fq(-35, 71));
    }

    #[test]
    fn rotations_are_exact_and_validated() {
        let z = Rotation::about("z", fq(3, 5), fq(4, 5)).unwrap();
        let p = vec![vec![r(5), r(1), r(0), r(2)]];
        assert_eq!(
            z.apply(&p).unwrap(),
            vec![vec![r(5), fq(3, 5), fq(4, 5), r(2)]]
        );
        assert!(Rotation::about("z", fq(1, 2), fq(1, 2)).is_err());
        assert!(StabilityChecks::new(vec![], false, true, false).is_err());
        assert!(StabilityChecks::new(vec![], false, false, true).is_err());
        let identity = Rotation::about("x", r(1), r(0)).unwrap();
        assert!(StabilityChecks::new(vec![identity], false, false, false).is_err());
        for k in 0..8 {
            assert!(!default_rotation(k).unwrap().is_identity());
        }
        assert!(Rotation::parse("euler:0.1,0.2,0.3").is_ok());
        assert!(Rotation::parse("1,1,0:0.7").is_ok());
        assert_eq!(parse_fraction(" -1_0.5e-1 ").unwrap(), fq(-21, 20));
        assert_eq!(parse_fraction("5.").unwrap(), r(5));
        assert!(parse_fraction("3 / 5").is_err());
        assert!(parse_fraction(".").is_err());
    }

    #[test]
    fn the_combination_is_pythons() {
        let reference = vec![vec![1.0, -2.0, 0.0]];
        let budgets = vec![
            ("rotation-0".to_string(), vec![vec![1e-15, 2e-15, 0.0]]),
            ("rotation-1".to_string(), vec![vec![3e-15, 1e-15, 0.0]]),
            (REDUCTION.to_string(), vec![vec![1e-16, 1e-16, 1e-17]]),
        ];
        let scale = vec![vec![2.0, 3.0, 1.0]];
        let u = 2f64.powi(-53);
        let estimate = Estimate::from_checks(&reference, budgets, scale, u, vec![]).unwrap();
        let e0 = 2.0 * (3e-15 + 1e-16) + 16.0 * u * 2.0;
        assert_eq!(estimate.absolute[0][0], e0);
        assert_eq!(estimate.coefficients[2], f64::INFINITY);
        assert!(Estimate::from_checks(&reference, vec![], vec![], u, vec![]).is_err());
    }
}
