// SPDX-License-Identifier: 0BSD

//! Lane-neutral Direct-Arena building blocks.
//!
//! This module contains no recurrence-current, eager-row, or compiled-stage
//! semantics. Each execution lane owns its authenticated plan, event schedule,
//! selectors, reductions, and hot loop.

mod layout;

pub use layout::{
    DirectArenaAssignment, DirectArenaInterval, DirectArenaLayout, assign_direct_arena,
};
