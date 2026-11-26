// SPDX-License-Identifier: 0BSD

//! The loader's hash tables (row maps, link interner, table indices).
//!
//! The loader's maps are keyed by small integers (helicity rows, table and
//! vertex indices) and by packed link lists, and are probed once per row
//! link: std's SipHash-1-3 was about a third of a large plan's load (perf,
//! g g > 7g). These tables use a multiply-rotate word hasher instead (the
//! scheme of rustc's `FxHasher`), deterministic and without per-process
//! seed. The tables are only probed, never iterated in an order a result
//! depends on (every derived list is ordered by first insertion or sorted),
//! so the hasher changes no loaded value and no order. A plan is the caller's
//! own artifact, digest-checked before it is read: a crafted plan could
//! slow its own load down, nothing else. The tables keep std's layout
//! (`std::collections::HashMap` with another `BuildHasher`), which the
//! derivation bound models (`binary/derive.rs`).

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// Multiply-rotate word hasher (module docs).
#[derive(Clone, Copy, Debug, Default)]
pub struct WordHasher {
    hash: u64,
}

/// The multiplier of rustc's `FxHasher` (64-bit).
const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl WordHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for WordHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut words = bytes.chunks_exact(8);
        for word in &mut words {
            self.add(u64::from_le_bytes(word.try_into().expect("eight bytes")));
        }
        let rest = words.remainder();
        if !rest.is_empty() {
            let mut word = [0u8; 8];
            word[..rest.len()].copy_from_slice(rest);
            // The tail's length keeps "ab" and "ab\0" apart.
            self.add(u64::from_le_bytes(word) ^ ((rest.len() as u64) << 59));
        }
    }

    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.add(u64::from(value));
    }

    #[inline]
    fn write_u16(&mut self, value: u16) {
        self.add(u64::from(value));
    }

    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.add(u64::from(value));
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.add(value);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.add(value as u64);
    }

    /// The product's high bits are its best mixed: rotated into the low bits
    /// the table indexes by (its control byte reads the top bits).
    #[inline]
    fn finish(&self) -> u64 {
        self.hash.rotate_left(26)
    }
}

pub type WordBuildHasher = BuildHasherDefault<WordHasher>;

/// A std map with the [`WordHasher`].
pub type WordMap<K, V> = HashMap<K, V, WordBuildHasher>;

/// A std set with the [`WordHasher`].
pub type WordSet<T> = HashSet<T, WordBuildHasher>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, Hash};

    fn hash_of<T: Hash>(value: &T) -> u64 {
        WordBuildHasher::default().hash_one(value)
    }

    #[test]
    fn the_word_hasher_is_deterministic_and_tells_tails_apart() {
        assert_eq!(hash_of(&(3u64, 5u64)), hash_of(&(3u64, 5u64)));
        assert_ne!(hash_of(&(3u64, 5u64)), hash_of(&(5u64, 3u64)));
        let mut a = WordHasher::default();
        a.write(b"ab");
        let mut b = WordHasher::default();
        b.write(b"ab\0");
        assert_ne!(a.finish(), b.finish());
        let mut long = WordHasher::default();
        long.write(&[7u8; 17]);
        let mut other = WordHasher::default();
        other.write(&[7u8; 16]);
        assert_ne!(long.finish(), other.finish());
    }

    #[test]
    fn word_maps_hold_dense_integer_keys() {
        // Row keys of a large plan: (leg mask, mixed-radix key).
        let mut map: WordMap<(u64, u64), u32> = WordMap::default();
        for legs in [0b1111_1111u64, 0b0000_0110, 0b1010_0000] {
            for key in 0..4096u64 {
                map.insert((legs, key), key as u32);
            }
        }
        assert_eq!(map.len(), 3 * 4096);
        assert_eq!(map.get(&(0b0000_0110, 4095)), Some(&4095));
        assert_eq!(map.get(&(0b0000_0111, 1)), None);
    }
}
