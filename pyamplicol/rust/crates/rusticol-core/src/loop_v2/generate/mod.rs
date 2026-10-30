// SPDX-License-Identifier: 0BSD

//! Native v2 generation (WP3.1, `workspace/reports/engine-v2/wp31/PLAN.md`).
//!
//! Each stage replaces one stage of the Python generator (`loops/v2`) and
//! reproduces it exactly: the Python generator stays the oracle. Inputs are
//! Θ(model) tables plus the field-level block table, exported by
//! `loops/v2/generator_input.py` with every name replaced by its rank in a
//! sorted table, so id comparisons follow Python's object order.
//!
//! - [`families`]: the loop-prefix family recursion (`families.py`);
//! - [`schedules`]: skeletons, the view, frames and schedule-B tries
//!   (`skeletons.py`, `schedules.py`);
//! - [`cycles`]: SU(N_c) overlaps of trace products by cycle counting (the
//!   exact colour contractions of `colour.py`);
//! - [`seeds`]: seed combinations and seed classes (`colour.py`
//!   `build_seed_table` / `extend_seed_table`);
//! - [`networks`]: exact SU(N_c) reduction of colour networks (quark rings
//!   and generic seeds of the seed stage);
//! - [`arena`]: the retained schedule builds of a generation, read in place
//!   by the seed stage, the LC/NLC prune rebuild and plan lowering;
//! - [`catalogue`]: the loop-kernel catalogue's polynomial stage (monomial
//!   coefficients, canonical strings and inputs of every segment;
//!   `catalogue.py`);
//! - [`rename`]: symbol renaming on Symbolica's portable text (the canonical
//!   stages of `assemble._canonical_stage`).

pub mod arena;
pub mod catalogue;
pub mod cells;
pub mod cycles;
pub mod families;
pub mod networks;
pub mod orders;
pub mod rename;
pub mod schedules;
pub mod seeds;

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// The multiplicative word hash of rustc (`FxHasher`): the generator's maps
/// are keyed by small integer tuples, for which SipHash dominates the cost.
/// Iteration order never reaches an output (every output is sorted or in
/// insertion order).
#[derive(Default, Clone, Copy)]
pub struct WordHasher(u64);

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl Hasher for WordHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(word));
        }
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.write_u64(u64::from(value));
    }

    #[inline]
    fn write_u16(&mut self, value: u16) {
        self.write_u64(u64::from(value));
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.write_u64(u64::from(value));
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(SEED);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

pub type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<WordHasher>>;
pub type FastSet<K> = HashSet<K, BuildHasherDefault<WordHasher>>;
