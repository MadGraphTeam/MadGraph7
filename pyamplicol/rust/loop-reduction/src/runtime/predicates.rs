// SPDX-License-Identifier: 0BSD
//! Delegate to the single core registry used by saved evaluators and PACBIN.
pub(super) fn register() {
    rusticol_core::source_scalar::register_scalar_primitives();
}
