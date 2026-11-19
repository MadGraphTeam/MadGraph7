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
//! NinjaRS's master-integral evaluators are built from the caller's symbols
//! (its `ninjars::master_argument_*` parameters, oneloopmaster's expressions
//! and helper functions, registered lazily) instead: their instructions
//! depend on those symbols' IDs, and so on when each master is first needed
//! (after the same import, a double-double reduction before a 57-digit one
//! moved the 57-digit values by 1e-75 relative).
//!
//! [`reserve`] builds the pool and registers the masters' symbols once per
//! process, at fixed IDs, before this library registers anything else: every
//! operation here that registers symbols
//! (importing a saved state, parsing an expression, defining the scalar
//! primitives) calls it first. Binary64 paths that touch no Symbolica state
//! never reach it, so they still need no licence (the SDKs' licence-free
//! tests). Without the `loop-v2` feature (no NinjaRS) it does nothing.

/// Build NinjaRS's positional slot pool and register its masters' symbols,
/// once per process (see the module).
pub fn reserve() {
    #[cfg(feature = "loop-v2")]
    {
        static RESERVED: std::sync::Once = std::sync::Once::new();
        RESERVED.call_once(|| {
            // Registering symbols is Symbolica state (`crate::licence`).
            let _unlock = crate::licence::unlock();
            // Never fails for no slots; the pool is built on this first use.
            let _ = ninjars_positional::slots(0);
            masters();
        });
    }
}

/// Register the symbols of NinjaRS's master-integral evaluators (its
/// `ninjars::master_argument_*` parameters and oneloopmaster's expressions
/// and helper functions), in one fixed order: those evaluators are built from
/// the caller's symbols, not on positional slots, so their instructions
/// depend on the IDs of these symbols, and the IDs on when the masters are
/// first needed (a double-double reduction before an arbitrary-precision
/// one, after a generation's import, moved 57-digit values by 1e-75
/// relative). Preparing every master once registers them all; the prepared
/// evaluators are dropped (each reduction keeps its own).
#[cfg(feature = "loop-v2")]
fn masters() {
    use ninjars::MasterIntegral;
    use symbolica::domains::float::Float;

    // Arbitrary precision prepares every family from its exact expression
    // (double-double only the tadpole tensors); the expressions, and hence
    // the symbols, do not depend on the precision.
    let mut masters = ninjars_symbolica::SymbolicaMasters::new(Float::with_val(64, 0));
    let integrals = [
        MasterIntegral::A0,
        MasterIntegral::B0,
        MasterIntegral::DB0,
        MasterIntegral::C0,
        MasterIntegral::D0,
        MasterIntegral::B1,
        MasterIntegral::B11,
        MasterIntegral::B111,
        MasterIntegral::B1111,
    ]
    .into_iter()
    .chain((0..=4).map(|rank| MasterIntegral::An { rank }));
    for integral in integrals {
        // A family NinjaRS cannot prepare registers nothing it would use.
        let _ = masters.prepare(integral, 1);
    }
}
