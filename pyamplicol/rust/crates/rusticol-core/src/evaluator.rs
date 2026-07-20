// SPDX-License-Identifier: 0BSD

#[path = "evaluator/crossing.rs"]
mod crossing;
pub(crate) use crossing::*;

#[path = "evaluator/backend.rs"]
mod backend;
pub(crate) use backend::{ensure_evaluator_capabilities_supported, evaluator_runtime_capabilities};

#[cfg(feature = "f64-compiled")]
#[path = "evaluator/compiled.rs"]
mod compiled;
#[cfg(feature = "f64-compiled")]
pub(crate) use compiled::*;

#[cfg(feature = "f64-symjit")]
#[path = "evaluator/symjit.rs"]
mod symjit;
#[cfg(feature = "f64-symjit")]
pub(crate) use symjit::*;

#[path = "evaluator/stage.rs"]
mod stage;
pub(crate) use stage::*;

#[path = "evaluator/amplitude.rs"]
mod amplitude;
pub(crate) use amplitude::{
    build_color_contraction_runtime, build_raw_sum_groups, generic_root_group_id,
};
