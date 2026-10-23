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
//!   (`skeletons.py`, `schedules.py`).

pub mod families;
pub mod orders;
pub mod schedules;
