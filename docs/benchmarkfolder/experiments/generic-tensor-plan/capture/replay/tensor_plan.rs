//! Index-only backward slicing of the original dense recurrence.
//!
//! A computed zero remains a computed zero: precision and sign are observable.
//! Only slots never read by a requested contraction or its ancestors are omitted.
use super::Shape;

pub const MAX_DENSE_SLOTS: usize = 1 << 20;
pub const MAX_STEPS: usize = 1 << 16;
pub const MAX_ROOT_VISITS: usize = 1 << 22;
pub const CACHE_BYTES: usize = 32 << 20;
pub const CACHE_ENTRIES: usize = 8;
const ABSENT: u32 = u32::MAX;

#[derive(Clone, PartialEq, Eq)]
pub struct Key {
    pub shape: Shape,
    pub rank: usize,
    pub mu_powers: usize,
    pub powers: Vec<(i64, usize, usize)>,
}

/// Terms retain the exact original factor order, including products with zero.
pub struct Step {
    pub component: usize,
    pub factors: [usize; 7],
    pub sources: [usize; 7],
    pub terms: usize,
}

pub struct Plan {
    pub key: Key,
    pub steps: Vec<Step>,
    /// Dense addresses map to compact addresses. 0 and 1 are the original
    /// prototype zero/one. ABSENT means no permitted caller can read that slot.
    mapping: Vec<u32>,
    pub dense_recurrence_steps: usize,
}

#[derive(Clone, Copy)]
struct Grid {
    shape: Shape,
    depth: usize,
    xs: usize,
    mus: usize,
    size: usize,
}

impl Grid {
    fn computed(&self, j: usize, a: usize, l: usize) -> bool {
        j <= self.depth && a < self.xs && l < self.mus && a + 2 * l <= j
    }
    fn at(&self, i: usize, j: usize, a: usize, l: usize) -> usize {
        i * self.size + (j * self.xs + a) * self.mus + l
    }
    fn unpack(&self, address: usize) -> (usize, usize, usize, usize) {
        let local = address % self.size;
        (
            address / self.size,
            local / (self.xs * self.mus),
            (local / self.mus) % self.xs,
            local % self.mus,
        )
    }
    fn predecessors(
        &self,
        j: usize,
        a: usize,
        l: usize,
        mut append: impl FnMut(usize, usize, usize, usize),
    ) {
        // This is a literal index transcription of tensor.rs, not a
        // polynomial sparsity rule. Out-of-lattice reads are original zeros.
        append(0, j, a, l);
        if j >= 1 {
            append(1, j - 1, a, l);
            if self.shape == Shape::T2 && a >= 1 {
                append(2, j - 1, a - 1, l);
            }
        }
        if j >= 2 && self.shape != Shape::Mu {
            append(3, j - 2, a, l);
            if self.shape == Shape::T2 {
                if a >= 1 {
                    append(4, j - 2, a - 1, l);
                }
                if a >= 2 {
                    append(5, j - 2, a - 2, l);
                }
            }
            if l >= 1 {
                append(6, j - 2, a, l - 1);
            }
        }
    }
}

impl Plan {
    pub fn new(
        key: Key,
        degrees: &[usize],
        parents: &[(usize, usize)],
        blocks: &[(usize, usize)],
    ) -> Option<Self> {
        if degrees.is_empty()
            || degrees.len() != parents.len()
            || blocks.len() != key.mu_powers.checked_add(1)?
            || key.powers.is_empty()
            || key.powers.len() > 2048
            || key.rank > 64
        {
            return None;
        }
        let lowest = key.powers.iter().map(|p| p.0).min()?;
        let depth = usize::try_from((key.rank as i64).checked_sub(lowest)?.max(0)).ok()?;
        let (xs, mus) = match key.shape {
            Shape::Mu => (1, 1),
            Shape::T3 => (1, depth.checked_div(2)?.checked_add(1)?),
            Shape::T2 => (depth.checked_add(1)?, depth.checked_div(2)?.checked_add(1)?),
        };
        let size = depth.checked_add(1)?.checked_mul(xs)?.checked_mul(mus)?;
        let slots = degrees.len().checked_mul(size)?;
        if slots > MAX_DENSE_SLOTS {
            return None;
        }
        // Bound the nested root scan even when many repeated columns select
        // the same few states. Mu adds an explicit j1 convolution loop.
        let block_visits = blocks
            .iter()
            .try_fold(0usize, |sum, &(_, count)| sum.checked_add(count))?;
        let root_visits =
            key.powers
                .len()
                .checked_mul(block_visits)?
                .checked_mul(if key.shape == Shape::Mu {
                    depth.checked_add(1)?
                } else {
                    1
                })?;
        if root_visits > MAX_ROOT_VISITS {
            return None;
        }
        let grid = Grid {
            shape: key.shape,
            depth,
            xs,
            mus,
            size,
        };
        let mut needed = vec![false; slots];
        let mut pending = Vec::new();
        let mut seed = |i: usize, j: usize, a: usize, l: usize| {
            if i > 0 && grid.computed(j, a, l) {
                let address = grid.at(i, j, a, l);
                if !needed[address] {
                    needed[address] = true;
                    pending.push(address);
                }
            }
        };
        // Match the original contraction's column/k/monomial selection.
        // Do not inspect coefficients; rebinding never invalidates a plan.
        for &(power, a, total_mu) in &key.powers {
            let max_k = if key.shape == Shape::Mu {
                key.mu_powers
            } else {
                key.mu_powers.min(total_mu)
            };
            for (k, &(_, count)) in blocks.iter().enumerate().take(max_k + 1) {
                if count > degrees.len() {
                    return None;
                }
                let l = if key.shape == Shape::Mu {
                    0
                } else {
                    total_mu - k
                };
                if l >= mus || a >= xs {
                    continue;
                }
                for (i, &degree) in degrees.iter().enumerate().take(count) {
                    let degree = degree.checked_add(if key.shape == Shape::Mu {
                        k.checked_mul(2)?
                    } else {
                        0
                    })?;
                    let j = (degree as i64).checked_sub(power)?;
                    if j < 0 || j as u64 > depth as u64 {
                        continue;
                    }
                    let j = j as usize;
                    if key.shape == Shape::Mu {
                        // Keep every convolution term, including structural zeros.
                        for j1 in 0..=j {
                            seed(i, j1, 0, 0);
                        }
                    } else if a + 2 * l <= j {
                        seed(i, j, a, l);
                    }
                }
            }
        }
        drop(seed);
        let mut required_steps = pending.len();
        if required_steps > MAX_STEPS {
            return None;
        }
        while let Some(address) = pending.pop() {
            let (i, j, a, l) = grid.unpack(address);
            let (parent, component) = parents[i];
            if parent >= i || component >= 4 {
                return None;
            }
            grid.predecessors(j, a, l, |_, j, a, l| {
                if parent > 0 && grid.computed(j, a, l) {
                    let address = grid.at(parent, j, a, l);
                    if !needed[address] {
                        needed[address] = true;
                        pending.push(address);
                        required_steps += 1;
                    }
                }
            });
            if required_steps > MAX_STEPS {
                return None;
            }
        }
        let mut mapping = vec![ABSENT; slots];
        // These are the slots the original clearing loop initializes and
        // never overwrites, plus the constant monomial's one.
        for (address, mapped) in mapping.iter_mut().enumerate() {
            let (i, j, a, l) = grid.unpack(address);
            if i == 0 || !grid.computed(j, a, l) {
                *mapped = 0;
            }
        }
        mapping[0] = 1;
        let mut steps = Vec::with_capacity(required_steps);
        for (address, &required) in needed.iter().enumerate() {
            if !required {
                continue;
            }
            let (i, j, a, l) = grid.unpack(address);
            let (parent, component) = parents[i];
            let mut step = Step {
                component,
                factors: [0; 7],
                sources: [0; 7],
                terms: 0,
            };
            let mut complete = true;
            grid.predecessors(j, a, l, |factor, j, a, l| {
                let source = mapping[grid.at(parent, j, a, l)];
                complete &= source != ABSENT;
                step.factors[step.terms] = factor;
                step.sources[step.terms] = source as usize;
                step.terms += 1;
            });
            if !complete {
                return None;
            }
            mapping[address] = u32::try_from(steps.len().checked_add(2)?).ok()?;
            steps.push(step);
        }
        let lattice = (0..size)
            .filter(|&local| {
                let (_, j, a, l) = grid.unpack(local);
                grid.computed(j, a, l)
            })
            .count();
        Some(Self {
            key,
            steps,
            mapping,
            dense_recurrence_steps: (degrees.len() - 1).checked_mul(lattice)?,
        })
    }

    pub fn slot(&self, dense_address: usize) -> usize {
        let slot = self.mapping[dense_address];
        assert_ne!(slot, ABSENT, "unplanned dense tensor read");
        slot as usize
    }
    pub fn slots(&self) -> usize {
        self.steps.len() + 2
    }
    pub fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.steps.capacity() * std::mem::size_of::<Step>()
            + self.mapping.capacity() * std::mem::size_of::<u32>()
            + self.key.powers.capacity() * std::mem::size_of::<(i64, usize, usize)>()
    }
}
