//! Guarded compensated T3 expansion for an explicit DP retry. No stability threshold or status changes.
//! The supported domain is N(q), rank 2..=8, mindeg=2. Every other callback is
//! delegated verbatim to the pinned upstream tensor, including n+2 callbacks.
//! Scratch storage belongs to the retained tensor and is reused between calls.
//! Traversal/layout adapt the pinned NinjaRS fast tensor, tensor_layout and
//! tensor_jets implementations (see Cargo.lock and UPSTREAM.md). Compensation
//! carries first-order product/sum residuals; it is not a universal guarantee
//! of correctly rounded polynomial evaluation. MadLoop retains its independent
//! direction/rotation checks and genuine higher-precision fallback.
use ninjars::fast::{C64, CMom, FastNumerator, TensorNumerator};
use ninjars::fast::tensor_layout::{block_offset, ncoeff_homogeneous};
use ninjars::{Error, Result};

unsafe extern "C" {
    fn ninjars_t3_standard_environment() -> i32;
}

#[derive(Clone, Copy, Default)]
struct Sum { sum: f64, err: f64 }
impl Sum {
    #[inline]
    fn add(&mut self, x: f64) {
        let s = self.sum + x;
        self.err += if self.sum.abs() >= x.abs() {
            (self.sum - s) + x
        } else {
            (x - s) + self.sum
        };
        self.sum = s;
    }
    #[inline]
    fn product(&mut self, a: f64, b: f64, sign: f64) {
        let hi = a * b;
        let lo = a.mul_add(b, -hi);
        self.add(sign * hi);
        self.add(sign * lo);
    }
    #[inline]
    fn value(self) -> f64 { self.sum + self.err }
}
#[derive(Clone, Copy, Default)]
struct Acc { r: Sum, i: Sum }
impl Acc {
    #[inline]
    fn add(&mut self, z: C64) { self.r.add(z.re); self.i.add(z.im); }
    #[inline]
    fn product(&mut self, g: Self, x: C64) {
        self.r.product(g.r.sum, x.re, 1.);
        self.r.product(g.i.sum, x.im, -1.);
        self.i.product(g.r.sum, x.im, 1.);
        self.i.product(g.i.sum, x.re, 1.);
        // First-order carried residual; this multiplication is rounded normally.
        // This is compensation, not a correctly-rounded polynomial guarantee.
        self.add(C64::new(g.r.err, g.i.err) * x);
    }
    #[inline]
    fn value(self) -> C64 { C64::new(self.r.value(), self.i.value()) }
}

/// Integer-only bounds on nonzero input components. `low` is the exact binary
/// lattice exponent, not a logarithmic approximation; `high` is a strict
/// magnitude upper bound. Signed zero is allowed. Subnormals and nonfinite
/// values are rejected before any floating arithmetic or output writes.
#[derive(Clone, Copy)]
struct Bounds { low: i32, high: i32 }
impl Bounds {
    fn values(values: impl Iterator<Item = C64>) -> Option<Self> {
        let mut result = Self { low: 0, high: 0 };
        for z in values {
            for x in [z.re, z.im] {
                let b = x.to_bits() & 0x7fff_ffff_ffff_ffff;
                if b == 0 { continue; }
                let exponent = ((b >> 52) & 0x7ff) as i32;
                if exponent == 0 || exponent == 0x7ff { return None; }
                let significand = (b & 0x000f_ffff_ffff_ffff) | (1u64 << 52);
                let low = exponent - 1023 - 52 + significand.trailing_zeros() as i32;
                result.low = result.low.min(low);
                result.high = result.high.max(exponent - 1023 + 1);
            }
        }
        Some(result)
    }
}

/// Conservative preflight for *every intermediate*, including FMA residuals.
/// Each recurrence product increases degree by one; final param multiplication
/// adds at most one further factor. Addition/subtraction cannot refine the
/// common binary lattice. A 16-bit-per-level magnitude allowance dominates
/// the <=4 complex child contributions, all compensation operations and the
/// inserted lower coefficient. Bounds +/-900 leave >120 exponent bits spare.
/// No speculative calculation is attempted outside this domain, so fallback
/// cannot leak new exception flags produced by an abandoned correction.
fn safe(rank: usize, c: &[C64], a: &CMom, e3: &CMom, e4: &CMom, p: C64) -> bool {
    let Some(c) = Bounds::values(c.iter().copied()) else { return false };
    let Some(v) = Bounds::values(a.iter().chain(e3).chain(e4).copied()) else { return false };
    let Some(p) = Bounds::values(std::iter::once(p)) else { return false };
    c.low + rank as i32 * v.low + p.low >= -900
        && c.high + rank as i32 * (v.high + 16) + p.high + 16 <= 900
}

pub struct GuardedTensor {
    native: TensorNumerator,
    work: Vec<[Acc; 4]>,
    pub corrected: usize,
    pub fallback: usize,
}
impl GuardedTensor {
    pub fn new(rank: usize, lanes: usize) -> Self {
        Self { native: TensorNumerator::new(rank, lanes),
            work: vec![[Acc::default(); 4]; 2 * ncoeff_homogeneous(rank)],
            corrected: 0, fallback: 0 }
    }
    pub fn coefficients_mut(&mut self) -> &mut [C64] { self.native.coefficients_mut() }

    fn carried(&mut self, lane: usize, a: &CMom, e3: &CMom, e4: &CMom, p: C64) -> [C64; 4] {
        let rank = self.native.rank();
        let c = self.native.lane_coefficients(lane);
        let level = ncoeff_homogeneous(rank);
        let (mut cur, mut next) = self.work.split_at_mut(level);
        cur.fill([Acc::default(); 4]);
        for (g, &z) in cur.iter_mut().zip(&c[block_offset(rank)..]) { g[0].add(z); }
        for d in (1..=rank).rev() {
            let size = ncoeff_homogeneous(d - 1);
            next[..size].fill([Acc::default(); 4]);
            for (i, sums) in next[..size].iter_mut().enumerate() {
                for component in [3, 2, 1, 0] {
                    let j = match component {
                        0 if i == 0 => Some(0),
                        1 if i < d => Some(1 + i),
                        2 if i < d * (d + 1) / 2 => Some(1 + d + i),
                        3 => Some(1 + d + d * (d + 1) / 2 + i),
                        _ => None,
                    };
                    if let Some(j) = j {
                        let v = cur[j];
                        sums[0].product(v[0], e3[component]);
                        sums[1].product(v[1], e3[component]);
                        sums[1].product(v[0], a[component]);
                        sums[2].product(v[2], e3[component]);
                        sums[2].product(v[1], a[component]);
                        sums[3].product(v[3], e3[component]);
                        sums[3].product(v[0], e4[component]);
                    }
                }
                let power = rank - d + 1;
                if power <= 2 { sums[power].add(c[block_offset(d - 1) + i]); }
            }
            std::mem::swap(&mut cur, &mut next);
        }
        let jet = cur[0];
        let mut constant = jet[2];
        constant.product(jet[3], p);
        [jet[0].value(), jet[1].value(), constant.value(), jet[3].value()]
    }
}

impl FastNumerator for GuardedTensor {
    fn rank(&self) -> usize { self.native.rank() }
    fn evaluate(&mut self, cut: &[usize], lanes: &[usize], q: &[CMom], mu: &[C64], out: &mut [C64]) -> Result<()> {
        self.native.evaluate(cut, lanes, q, mu, out)
    }
    fn mu_expansion(&mut self, cut: &[usize], lanes: &[usize], v0: &[CMom], v1: Option<&[CMom]>, out: &mut [C64]) -> Result<()> {
        self.native.mu_expansion(cut, lanes, v0, v1, out)
    }
    fn mu_expansion_x2(&mut self, cut: &[usize], lanes: &[usize], v0: &[CMom], v1: &[CMom], beta: &[C64], out: &mut [C64]) -> Result<()> {
        self.native.mu_expansion_x2(cut, lanes, v0, v1, beta, out)
    }
    fn t2_expansion(&mut self, cut: &[usize], lanes: &[usize], a0: &[CMom], a1: &[CMom], e3: &[CMom], e4: &[CMom], p: &[[C64;3]], k: usize, out: &mut [C64]) -> Result<()> {
        self.native.t2_expansion(cut, lanes, a0, a1, e3, e4, p, k, out)
    }
    fn t2_expansion_x2(&mut self, cut: &[usize], lanes: &[usize], a0: &[CMom], a1: &[CMom], e3: &[CMom], e4: &[CMom], p: &[[C64;3]], out: &mut [C64]) -> Result<()> {
        self.native.t2_expansion_x2(cut, lanes, a0, a1, e3, e4, p, out)
    }
    fn t3_expansion_x2(&mut self, cut: &[usize], lanes: &[usize], a: &[CMom], e3: &[CMom], e4: &[CMom], p: &[C64], out: &mut [C64]) -> Result<()> {
        self.fallback += lanes.len();
        self.native.t3_expansion_x2(cut, lanes, a, e3, e4, p, out)
    }
    fn t3_expansion(&mut self, cut: &[usize], lanes: &[usize], a: &[CMom], e3: &[CMom], e4: &[CMom], p: &[C64], k: usize, out: &mut [C64]) -> Result<()> {
        let rank = self.native.rank();
        if k != 2 || !(2..=8).contains(&rank) || unsafe { ninjars_t3_standard_environment() } != 1 {
            self.fallback += lanes.len();
            return self.native.t3_expansion(cut, lanes, a, e3, e4, p, k, out);
        }
        if out.len() != 4 * lanes.len() {
            return Err(Error::Shape { name: "tensor numerator output", expected: 4 * lanes.len(), actual: out.len() });
        }
        for (row, &lane) in lanes.iter().enumerate() {
            if lane >= self.native.lanes() { return Err(Error::Lane { lane, lanes: self.native.lanes() }); }
            let target = &mut out[4 * row..4 * row + 4];
            if safe(rank, self.native.lane_coefficients(lane), &a[row], &e3[row], &e4[row], p[row]) {
                target.copy_from_slice(&self.carried(lane, &a[row], &e3[row], &e4[row], p[row]));
                self.corrected += 1;
            } else {
                self.native.t3_expansion(cut, &[lane], &a[row..row+1], &e3[row..row+1], &e4[row..row+1], &p[row..row+1], k, target)?;
                self.fallback += 1;
            }
        }
        Ok(())
    }
}
