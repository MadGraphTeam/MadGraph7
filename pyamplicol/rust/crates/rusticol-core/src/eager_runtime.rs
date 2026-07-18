// SPDX-License-Identifier: 0BSD

use crate::{RusticolError, RusticolResult};
use num_complex::Complex;

pub type EagerComplex64 = Complex<f64>;

pub const DEFAULT_EAGER_POINT_TILE_SIZE: usize = 1024;
pub const DEFAULT_EAGER_WORKSPACE_MIB: usize = 256;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum EagerKernelRole {
    Vertex,
    Finalization,
    Closure,
}

/// One prepared evaluator input in its exact, deterministic parameter order.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum EagerKernelInput {
    FirstCurrentComponent(u32),
    SecondCurrentComponent(u32),
    FirstMomentumComponent(u32),
    SecondMomentumComponent(u32),
    CouplingReal,
    CouplingImag,
    ModelParameter(u32),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EagerKernelSpec {
    pub kernel_id: u32,
    pub role: EagerKernelRole,
    pub inputs: Vec<EagerKernelInput>,
    pub output_component_count: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EagerPlanDimensions {
    pub value_slot_component_counts: Vec<u32>,
    pub momentum_slot_component_counts: Vec<u32>,
    pub current_component_counts: Vec<u32>,
    pub parameter_count: u32,
    pub amplitude_count: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EagerDirectClosureSpec {
    pub closure_index: u32,
    pub coefficients: Vec<EagerComplex64>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EagerReductionTerm {
    pub left_amplitude_index: u32,
    pub right_amplitude_index: u32,
    pub coefficient: EagerComplex64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EagerPlanDefinition {
    pub dimensions: EagerPlanDimensions,
    pub kernels: Vec<EagerKernelSpec>,
    pub direct_closures: Vec<EagerDirectClosureSpec>,
    pub reduction_terms: Vec<EagerReductionTerm>,
}

#[derive(Clone, Copy, Debug)]
pub struct EagerStagePayload<'a> {
    pub stage_index: u32,
    pub invocations: &'a [u8],
    pub attachments: &'a [u8],
    pub finalizations: &'a [u8],
}

#[derive(Clone, Copy, Debug)]
pub struct EagerPlanPayloads<'a> {
    pub couplings: &'a [u8],
    pub stages: &'a [EagerStagePayload<'a>],
    pub closures: &'a [u8],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EagerRuntimeOptions {
    pub point_tile_size: usize,
    pub workspace_bytes: usize,
}

impl EagerRuntimeOptions {
    pub fn from_mib(point_tile_size: usize, workspace_mib: usize) -> RusticolResult<Self> {
        let workspace_bytes = workspace_mib.checked_mul(1024 * 1024).ok_or_else(|| {
            RusticolError::invalid_argument("eager workspace size overflows bytes")
        })?;
        Ok(Self {
            point_tile_size,
            workspace_bytes,
        })
    }
}

impl Default for EagerRuntimeOptions {
    fn default() -> Self {
        Self {
            point_tile_size: DEFAULT_EAGER_POINT_TILE_SIZE,
            workspace_bytes: DEFAULT_EAGER_WORKSPACE_MIB * 1024 * 1024,
        }
    }
}

pub struct EagerKernelCall<'a> {
    pub kernel_id: u32,
    pub lane_count: usize,
    pub input_component_count: usize,
    pub output_component_count: usize,
    pub inputs: &'a [EagerComplex64],
    pub outputs: &'a mut [EagerComplex64],
}

/// Evaluates component-major packets whose lanes are ordered by invocation, then point.
///
/// The scheduler sorts invocations stably by kernel id while loading a plan. Each packet
/// therefore contains one kernel only; within that packet, all points for the first
/// invocation precede all points for the next invocation. Input components follow the
/// exact order in [`EagerKernelSpec::inputs`].
pub trait EagerKernelBackend {
    fn evaluate_batch(&mut self, call: EagerKernelCall<'_>) -> RusticolResult<()>;
}

mod execute;
mod plan;
mod runtime;

pub use plan::EagerExecutionPlan;
pub use runtime::EagerExecutionRuntime;
