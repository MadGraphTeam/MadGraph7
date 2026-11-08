// SPDX-License-Identifier: 0BSD

//! TIRust (<https://github.com/huashengshao/TIRust>), the tensor-integral
//! reduction backend, in this build: the default cargo feature `tir` links
//! its `iregi-rust` and `iregi-masters` crates (m-lit PLAN §2.c.2 item 3;
//! USER 2026-11-06 ~13:35); `--no-default-features` builds without them.
//! The reduction backend on top of it is m-lit `lit-tir`'s.

/// Whether this build links TIRust and its direct engine answers a plan
/// query (the one-propagator, rank-0 integral is one it serves; no Symbolica
/// state is touched). `false` in a build without the `tir` feature.
pub fn available() -> bool {
    #[cfg(feature = "tir")]
    {
        iregi_rust::DirectReductionPlan::supports(&iregi_rust::ReductionSettings::new(1, 0))
    }
    #[cfg(not(feature = "tir"))]
    {
        false
    }
}

/// TIRust's own structural bounds at a carrier of `bits` bits, (the largest
/// number of propagators, the largest tensor rank), the numbers its
/// capability query decides on (`v2_tir::capabilities`); None in a build
/// without the `tir` feature.
pub fn capabilities(bits: u32) -> Option<(usize, usize)> {
    #[cfg(feature = "tir")]
    {
        Some(crate::v2_tir::capabilities(bits))
    }
    #[cfg(not(feature = "tir"))]
    {
        let _ = bits;
        None
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn available_matches_the_tir_feature() {
        assert_eq!(super::available(), cfg!(feature = "tir"));
    }
}
