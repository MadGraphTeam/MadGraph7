// SPDX-License-Identifier: 0BSD

//! Native core of the shared-recursion one-loop engine v2
//! (`docs/development/loop_v2/DESIGN.md`).
//!
//! [`plan`] loads plan v4 and its [`observables`] columns; [`dress`],
//! [`seeds`], [`trie`] (schedule B), [`forward`] (schedule F), [`merged`]
//! (schedule M) and [`closure`] execute it on SIMD lanes in any carrier;
//! [`retest`] holds the reduction-direction check. The evaluation
//! driver with bucket streaming into the NinjaRS adapter is
//! `crate::loop_execution::v2`.
//!
//! Storage: the executor keeps every object, factor and accumulator in the
//! lane-block form of [`kernels`] (`L` real parts, then `L` imaginary parts
//! per coefficient), the form gate G0 measured; [`poly`]'s `Complex`-per-lane
//! `OpenLoopPolynomial` and `apply_segment` remain the exact reference algebra
//! of its tests.

pub mod closure;
pub mod dag;
pub mod dipole_orbits;
pub mod dress;
pub mod forward;
pub mod fourier;
pub mod generate;
pub mod interference;
pub mod kernels;
pub mod merged;
pub mod observables;
pub mod plan;
pub mod poly;
pub mod retest;
pub mod seeds;
pub mod trees;
pub mod trie;
