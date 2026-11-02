// SPDX-License-Identifier: 0BSD

//! pyAmpliCol's Symbolica library key in the native code.
//!
//! The Python package registers its key with the Python `symbolica` module
//! (`pyamplicol/__init__.py`); that unlocks Symbolica calls made from frames of
//! the `pyamplicol` package, not the Rust copy of Symbolica linked into the
//! native extension (`pyamplicol._rusticol`) and the loop SDK. Without a
//! licence, that copy runs in Symbolica's restricted mode, which admits one
//! thread per user: the first state access (a symbol, an expression, a saved
//! evaluator decoded) takes the user's single permit and every other
//! unlicensed process aborts. The same key, issued for the package
//! `pyamplicol`, is registered here, in the crate of that name, through
//! Symbolica's documented `set_library_key!` (which binds the key to this
//! crate), and kept private to this crate as Symbolica asks. pyAmpliCol's
//! other crates take thread-bound guards from [`unlock`]: every native
//! operation that reaches Symbolica state (decoding saved evaluators,
//! registering the scalar primitives, evaluating exact states in DoubleFloat
//! or arbitrary precision, the DoubleFloat and arbitrary-precision reduction)
//! holds one on its thread, and the loop executor's worker threads hold one
//! for their lifetime ([`unlock_thread`]). Binary64 evaluation of stored
//! SymJIT applications takes none, except the binary64 reduction of a loop
//! at the fast engine's absolute thresholds, which runs the generic engine
//! (`rusticol-loop-reduction`, `runtime::reaches_fast_threshold`) under a
//! guard.
//!
//! A user's own licence (`SYMBOLICA_LICENSE`) is used as before; an expired or
//! rejected key yields inert guards (Symbolica's rule), so the user's licence
//! or the restricted mode applies. Setting `PYAMPLICOL_NATIVE_LIBRARY_KEY=off`
//! disables the key in the native code (a diagnostic: the tests use it to show
//! that binary64 evaluation touches no Symbolica state at all).

use std::sync::LazyLock;
use symbolica::license::{LibraryUnlock, LibraryUnlockGuard};

/// The environment switch that disables the key in the native code.
pub const DISABLE_VARIABLE: &str = "PYAMPLICOL_NATIVE_LIBRARY_KEY";

// The key of `pyamplicol/__init__.py`, issued for the package `pyamplicol`.
const KEY: [&str; 3] = [
    "SU-419-pyamplicol-2028.01.01-",
    "KJBO3RLZJ3H5FKU5CL756QLKZCURABBCKRGHZRK5OVIUQXZBCLJTOUQPHXYVTANIX",
    "SCKI6YTZDWGH2NWHQIBCPDUO5GVW2STVMP3MCQ",
];

// Private to this crate (Symbolica: "do not use `pub` scope to prevent key
// leakage"); other crates receive guards only.
pub(crate) static UNLOCK: LazyLock<Option<LibraryUnlock>> = LazyLock::new(|| {
    if !enabled_by_environment() {
        return None;
    }
    // A rejected key leaves the user's licence or the restricted mode.
    symbolica::set_library_key!(&KEY.concat()).ok()
});

fn enabled_by_environment() -> bool {
    !matches!(
        std::env::var(DISABLE_VARIABLE).as_deref(),
        Ok("off" | "0" | "false" | "no")
    )
}

/// Authorise Symbolica calls on the current thread until the guard drops
/// (none when the key is disabled or was rejected). For pyAmpliCol's own
/// crates.
#[doc(hidden)]
#[inline]
pub fn unlock() -> Option<LibraryUnlockGuard> {
    UNLOCK.as_ref().map(LibraryUnlock::unlock)
}

/// Authorise Symbolica calls on the current thread for its lifetime: for
/// worker threads owned by pyAmpliCol (a thread pool's start handler).
#[doc(hidden)]
pub fn unlock_thread() {
    if let Some(guard) = unlock() {
        // The guard is thread-bound; leaking it keeps this thread unlocked
        // until it exits.
        std::mem::forget(guard);
    }
}

/// Whether the key unlocks the native code in this process.
pub fn is_active() -> bool {
    UNLOCK.is_some()
}
