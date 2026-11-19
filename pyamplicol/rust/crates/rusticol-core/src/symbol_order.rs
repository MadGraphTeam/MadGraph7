// SPDX-License-Identifier: 0BSD

//! The symbols this library registers first: NinjaRS's positional slots.
//!
//! NinjaRS builds its exact evaluators (every double-double and
//! arbitrary-precision NinjaRS reduction: numerators, cuts, kernels) on
//! positional slots (`ninjars_positional`): 4096 symbols registered together
//! the first time one is needed, in aligned blocks of two-byte IDs, so that
//! an evaluator's instructions depend only on its expressions. A process
//! that registered more than about 61000 symbols before that first use gets
//! function slots instead (and, past 65535, a three-byte head), whose
//! evaluators have other instructions and round differently. The NinjaRS
//! results in the extended carriers then depend on what the process did
//! before: a generation in the same process (the whole producer state that
//! `canonical_saved_evaluator` imports, about 4e5 symbols) moved later
//! double-double NinjaRS values at near-forward g g > h h points by up to
//! 3e-16 relative and its fallback-guard estimates from 2.4e-19 to 4.1e-16.
//!
//! [`reserve`] builds the pool once per process before this library
//! registers anything else: every operation here that registers symbols
//! (importing a saved state, parsing an expression, defining the scalar
//! primitives) calls it first. Binary64 paths that touch no Symbolica state
//! never reach it, so they still need no licence (the SDKs' licence-free
//! tests). Without the `loop-v2` feature (no NinjaRS) it does nothing.

/// Build NinjaRS's positional slot pool, once per process (see the module).
pub fn reserve() {
    #[cfg(feature = "loop-v2")]
    {
        static RESERVED: std::sync::Once = std::sync::Once::new();
        RESERVED.call_once(|| {
            // Registering symbols is Symbolica state (`crate::licence`).
            let _unlock = crate::licence::unlock();
            // Never fails for no slots; the pool is built on this first use.
            let _ = ninjars_positional::slots(0);
        });
    }
}
