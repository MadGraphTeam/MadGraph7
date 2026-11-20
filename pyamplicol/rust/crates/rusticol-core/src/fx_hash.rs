// SPDX-License-Identifier: 0BSD
//! Fast non-cryptographic hashing for transient maps whose iteration order is never observed.

use std::hash::{BuildHasherDefault, Hasher};

/// FxHash (rustc's hasher): the builder's maps are keyed by small integers.
#[derive(Default, Clone, Copy)]
pub struct FxHasher(u64);

impl Hasher for FxHasher {
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
        self.write_u64(value as u64);
    }
    #[inline]
    fn write_u32(&mut self, value: u32) {
        self.write_u64(value as u64);
    }
    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
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

pub(crate) type FxBuildHasher = BuildHasherDefault<FxHasher>;
