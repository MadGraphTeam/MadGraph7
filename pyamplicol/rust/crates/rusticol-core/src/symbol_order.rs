// SPDX-License-Identifier: 0BSD

//! The symbols this library registers first: NinjaRS's reserved symbols.
//!
//! Symbolica orders terms, and breaks ties while building an evaluator, by
//! the encoded bytes of their symbols, which hold their registration
//! indices. NinjaRS keeps its double-double and arbitrary-precision results
//! independent of those indices only if its positional slot pool and its
//! master integrals' symbols are registered before the process registers
//! symbols of its own (`ninjars_symbolica::reserve`, whose documentation is
//! the contract). Without that, the NinjaRS results in the extended carriers
//! depended on what the process did before: after a generation in the same
//! process (the whole producer state that `canonical_saved_evaluator`
//! imports, about 4e5 symbols, which also left the pool without variable
//! slots) double-double values at near-forward g g > h h points moved by up
//! to 3e-16 relative and a fallback-guard estimate from 2.4e-19 to 4.1e-16;
//! after a 20,000-symbol import, double-double values moved by 1e-28.
//!
//! [`reserve`] makes that call once per process, before this library
//! registers anything else: every operation here that registers symbols
//! (importing a saved state, parsing an expression through [`parse`],
//! defining the scalar primitives) goes through it. It fails loudly when the
//! pool could not get two-byte variable slots. It compiles nothing; binary64
//! paths that touch no Symbolica state never reach it, so they still need no
//! licence (the SDKs' licence-free tests). Without the `loop-v2` feature (no
//! NinjaRS) it does nothing.

use symbolica::{atom::Atom, parser::ParseSettings};

/// Reserve NinjaRS's symbols, once per process (see the module); the error
/// names the slot regime NinjaRS could not get.
pub fn reserve() -> Result<(), String> {
    #[cfg(feature = "loop-v2")]
    {
        // Registering symbols is Symbolica state (`crate::licence`).
        let _unlock = crate::licence::unlock();
        ninjars_symbolica::reserve()
            .map(|_| ())
            .map_err(|error| format!("NinjaRS symbol reservation: {error}"))
    }
    #[cfg(not(feature = "loop-v2"))]
    Ok(())
}

/// [`reserve`] for the operations that cannot report an error (the scalar
/// primitives' registration): a failed reservation is a defect of this
/// library (a symbol registration that bypassed it), never of the input.
pub(crate) fn reserved() {
    if let Err(error) = reserve() {
        panic!("{error}");
    }
}

/// `text` parsed in `namespace` after [`reserve`]: every expression this
/// library parses goes through here.
pub fn parse(text: &str, namespace: &str) -> Result<Atom, String> {
    reserve()?;
    Atom::parse(text, namespace, ParseSettings::default()).map_err(|error| error.to_string())
}

/// The slot regime NinjaRS obtained ([`reserve`] first): `"variables"`, or the
/// reservation's error. For the history tests.
pub fn regime() -> Result<&'static str, String> {
    reserve()?;
    #[cfg(feature = "loop-v2")]
    {
        Ok(match ninjars_symbolica::reserve() {
            Ok(ninjars_symbolica::Regime::Variables) => "variables",
            Ok(ninjars_symbolica::Regime::Functions) => "functions",
            Err(_) => unreachable!("reserve() succeeded"),
        })
    }
    #[cfg(not(feature = "loop-v2"))]
    Ok("no NinjaRS")
}
