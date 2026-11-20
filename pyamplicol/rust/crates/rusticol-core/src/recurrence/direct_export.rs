// SPDX-License-Identifier: 0BSD

//! Portable, resolved description of the batched direct recurrence program.
//!
//! Addresses are complex component offsets in the current arena, rather than
//! semantic current IDs: the direct scheduler is allowed to reuse storage.
//! Instruction and operand order are significant.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::direct_backend::{
    DirectContributionFanoutProgram, DirectContributionFanoutStep, DirectStageClearRanges,
};
use super::direct_plan::DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION;
use super::{
    DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE, DIRECT_NONE_U32, DirectExecutorRole,
    DirectRecurrencePlan, DirectRowGroupDescriptor,
};
use crate::{RusticolError, RusticolResult};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportComplex {
    pub re: f64,
    pub im: f64,
}

impl ExportComplex {
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };

    pub(crate) fn multiply(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectExportMode {
    Set,
    Add,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectExportSourceEmbedding {
    pub component: u32,
    pub source_component: Option<u32>,
    pub factor: ExportComplex,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectExportSource {
    pub current: u32,
    pub component_count: u32,
    pub family: String,
    pub orientation: String,
    pub leg: u32,
    pub momentum_form: u32,
    pub helicity: i32,
    pub chirality: i32,
    pub mass: ExportComplex,
    pub factor: ExportComplex,
    /// Empty for ordinary sources. Otherwise scatter the source formula's
    /// components in this order, multiplying each by its embedding factor.
    pub embedding: Vec<DirectExportSourceEmbedding>,
}

/// One real input plane of a saved SymJIT application, in application order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DirectExportKernelInput {
    Current { component: u32, imaginary: bool },
    Amplitude { amplitude: u32, imaginary: bool },
    Momentum { form: u32, component: u16 },
    Literal { value: f64 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectExportKernelBinding {
    pub id: u32,
    pub sha256: String,
    pub inputs: Vec<RelativeKernelInput>,
    /// Real/imaginary relative projections in application output order.
    pub outputs: Vec<RelativeKernelInput>,
    /// Bypass output scaling; the row's set/add mode still applies.
    pub direct: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DirectExportInstruction {
    Clear {
        current: u32,
        component_count: u32,
    },
    ClearAmplitude {
        amplitude: u32,
    },
    Source {
        source: u32,
    },
    Vertex {
        vertex: String,
        current: u32,
        mode: DirectExportMode,
        left: u32,
        right: Option<u32>,
        left_momentum: u32,
        right_momentum: Option<u32>,
        coupling: ExportComplex,
        factor: ExportComplex,
    },
    Propagator {
        propagator: String,
        current: u32,
        component_count: u32,
        momentum_form: u32,
        factor: ExportComplex,
    },
    Scale {
        current: u32,
        component_count: u32,
        factor: ExportComplex,
    },
    Copy {
        current: u32,
        source: u32,
        component_count: u32,
        factor: ExportComplex,
    },
    /// Batched fanout scatter. Unlike `Copy`, unit/real/imaginary factors
    /// use the runtime's specialized arithmetic, including signed zeros.
    Scatter {
        current: u32,
        source: u32,
        component_count: u32,
        mode: DirectExportMode,
        factor: ExportComplex,
    },
    Closure {
        amplitude: u32,
        left: u32,
        right: u32,
        component_factors: Vec<ExportComplex>,
        factor: ExportComplex,
    },
    Kernel {
        binding: u32,
        destination: u32,
        amplitude: bool,
        left: Option<u32>,
        right: Option<u32>,
        left_momentum: Option<u32>,
        right_momentum: Option<u32>,
        mode: DirectExportMode,
        factor: ExportComplex,
    },
}

/// Cold export-only representation of a loaded executor. This adds no hot
/// dispatch state and is reconstructed from the existing process binding.
pub(crate) enum DirectExportExecutor {
    Source,
    Intrinsic {
        template: String,
        coupling: ExportComplex,
    },
    Kernel {
        sha256: String,
        inputs: Vec<RelativeKernelInput>,
        output_projections: Vec<RelativeKernelInput>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RelativeKernelInput {
    Parent {
        parent: u8,
        component: u16,
        imaginary: bool,
    },
    Destination {
        component: u16,
        imaginary: bool,
        amplitude: bool,
    },
    Momentum {
        operand: u8,
        component: u16,
    },
    Literal {
        value: f64,
    },
    Factor {
        imaginary: bool,
    },
}

impl PartialEq for RelativeKernelInput {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Parent {
                    parent: a,
                    component: b,
                    imaginary: c,
                },
                Self::Parent {
                    parent: x,
                    component: y,
                    imaginary: z,
                },
            ) => (a, b, c) == (x, y, z),
            (
                Self::Destination {
                    component: a,
                    imaginary: b,
                    amplitude: c,
                },
                Self::Destination {
                    component: x,
                    imaginary: y,
                    amplitude: z,
                },
            ) => (a, b, c) == (x, y, z),
            (
                Self::Momentum {
                    operand: a,
                    component: b,
                },
                Self::Momentum {
                    operand: x,
                    component: y,
                },
            ) => (a, b) == (x, y),
            (Self::Literal { value: a }, Self::Literal { value: b }) => a.to_bits() == b.to_bits(),
            (Self::Factor { imaginary: a }, Self::Factor { imaginary: b }) => a == b,
            _ => false,
        }
    }
}

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::integrity(format!("recurrence export: {}", message.into()))
}

fn optional(value: u32) -> Option<u32> {
    (value != DIRECT_NONE_U32).then_some(value)
}

pub(crate) struct Operands {
    pub(crate) left: u32,
    pub(crate) right: u32,
    pub(crate) left_momentum: u32,
    pub(crate) right_momentum: u32,
    pub(crate) destination: u32,
}

impl RelativeKernelInput {
    pub(crate) fn resolve(
        &self,
        operands: &Operands,
        factor: ExportComplex,
    ) -> RusticolResult<DirectExportKernelInput> {
        Ok(match *self {
            Self::Parent {
                parent,
                component,
                imaginary,
            } => {
                let base = match parent {
                    0 => operands.left,
                    1 => operands.right,
                    _ => return Err(invalid("kernel parent exceeds two operands")),
                };
                if base == DIRECT_NONE_U32 {
                    return Err(invalid("kernel reads an absent parent"));
                }
                DirectExportKernelInput::Current {
                    component: base + u32::from(component),
                    imaginary,
                }
            }
            Self::Destination {
                component,
                imaginary,
                amplitude,
            } => {
                if amplitude {
                    DirectExportKernelInput::Amplitude {
                        amplitude: operands.destination + u32::from(component),
                        imaginary,
                    }
                } else {
                    DirectExportKernelInput::Current {
                        component: operands.destination + u32::from(component),
                        imaginary,
                    }
                }
            }
            Self::Momentum { operand, component } => {
                let form = match operand {
                    0 => operands.left_momentum,
                    1 => operands.right_momentum,
                    _ => return Err(invalid("kernel momentum exceeds two operands")),
                };
                if form == DIRECT_NONE_U32 {
                    return Err(invalid("kernel reads an absent momentum"));
                }
                DirectExportKernelInput::Momentum { form, component }
            }
            Self::Literal { value } => DirectExportKernelInput::Literal { value },
            Self::Factor { imaginary } => DirectExportKernelInput::Literal {
                value: if imaginary { factor.im } else { factor.re },
            },
        })
    }
}

fn kernel_instruction(
    executor: &DirectExportExecutor,
    operands: Operands,
    factor: ExportComplex,
    mode: DirectExportMode,
    role: DirectExecutorRole,
    bindings: &mut Vec<DirectExportKernelBinding>,
) -> RusticolResult<DirectExportInstruction> {
    let DirectExportExecutor::Kernel {
        sha256,
        inputs,
        output_projections,
    } = executor
    else {
        return Err(invalid("expected saved kernel executor"));
    };
    let resolved_inputs = inputs
        .iter()
        .map(|input| input.resolve(&operands, factor))
        .collect::<RusticolResult<Vec<_>>>()?;
    let resolved_outputs = output_projections
        .iter()
        .map(|output| match output.resolve(&operands, factor)? {
            address @ (DirectExportKernelInput::Current { .. }
            | DirectExportKernelInput::Amplitude { .. }) => Ok(address),
            _ => Err(invalid("kernel output is not an arena plane")),
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let direct = role == DirectExecutorRole::Finalization
        && factor.re.to_bits() == 1.0_f64.to_bits()
        && factor.im.to_bits() == 0.0_f64.to_bits()
        && resolved_outputs.iter().enumerate().all(|(index, output)| {
            !resolved_outputs[..index].contains(output) && !resolved_inputs.contains(output)
        });
    let id = if let Some(binding) = bindings.iter().find(|binding| {
        binding.sha256 == *sha256
            && binding.inputs == *inputs
            && binding.outputs == *output_projections
            && binding.direct == direct
    }) {
        binding.id
    } else {
        let id = u32::try_from(bindings.len())
            .map_err(|_| invalid("kernel binding count exceeds u32"))?;
        bindings.push(DirectExportKernelBinding {
            id,
            sha256: sha256.clone(),
            inputs: inputs.clone(),
            outputs: output_projections.clone(),
            direct,
        });
        id
    };
    Ok(DirectExportInstruction::Kernel {
        binding: id,
        destination: operands.destination,
        amplitude: role == DirectExecutorRole::Closure,
        left: optional(operands.left),
        right: optional(operands.right),
        left_momentum: optional(operands.left_momentum),
        right_momentum: optional(operands.right_momentum),
        mode,
        factor,
    })
}

fn contribution_mode(flags: u32) -> DirectExportMode {
    if flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION != 0 {
        DirectExportMode::Set
    } else {
        DirectExportMode::Add
    }
}

fn contribution_instruction(
    row: &super::DirectContributionRow,
    executor: &DirectExportExecutor,
    factor: ExportComplex,
    bindings: &mut Vec<DirectExportKernelBinding>,
) -> RusticolResult<DirectExportInstruction> {
    let mode = contribution_mode(row.flags);
    match executor {
        DirectExportExecutor::Intrinsic { template, coupling } => {
            Ok(DirectExportInstruction::Vertex {
                vertex: template.clone(),
                current: row.destination_component_base,
                mode,
                left: row.parent0_component_base,
                right: optional(row.parent1_component_base_or_sentinel),
                left_momentum: row.parent0_momentum_form_id,
                right_momentum: optional(row.parent1_momentum_form_id_or_sentinel),
                coupling: *coupling,
                factor,
            })
        }
        DirectExportExecutor::Kernel { .. } => kernel_instruction(
            executor,
            Operands {
                left: row.parent0_component_base,
                right: row.parent1_component_base_or_sentinel,
                left_momentum: row.parent0_momentum_form_id,
                right_momentum: row.parent1_momentum_form_id_or_sentinel,
                destination: row.destination_component_base,
            },
            factor,
            mode,
            DirectExecutorRole::Contribution,
            bindings,
        ),
        _ => Err(invalid("source executor used as contribution")),
    }
}

/// Lower the already bound batched schedule without executing a phase-space
/// point. The caller supplies the *selected* row groups and prepared clears.
pub(crate) fn export_direct_body(
    plan: &DirectRecurrencePlan,
    groups: &[DirectRowGroupDescriptor],
    clears: &[DirectStageClearRanges],
    executors: &BTreeMap<u32, DirectExportExecutor>,
    fanout: Option<&DirectContributionFanoutProgram>,
) -> RusticolResult<(
    Vec<DirectExportInstruction>,
    Vec<DirectExportKernelBinding>,
    Vec<u32>,
)> {
    let mut bindings = Vec::new();
    let mut source_rows = Vec::new();
    let factors = plan
        .exact_factors()
        .iter()
        .map(|factor| ExportComplex {
            re: factor.real().numerator() as f64 / factor.real().denominator() as f64,
            im: factor.imag().numerator() as f64 / factor.imag().denominator() as f64,
        })
        .collect::<Vec<_>>();
    let factor = |id: u32| {
        factors
            .get(id as usize)
            .copied()
            .ok_or_else(|| invalid("factor ID is absent"))
    };
    let mut body = (0..plan.amplitude_destinations().len())
        .map(|amplitude| DirectExportInstruction::ClearAmplitude {
            amplitude: amplitude as u32,
        })
        .collect::<Vec<_>>();
    let mut cleared_stage = None;
    for group in groups {
        if group.role == DirectExecutorRole::Contribution && cleared_stage != Some(group.stage) {
            if let Some(stage) = clears.iter().find(|entry| entry.stage == group.stage) {
                body.extend(
                    stage
                        .ranges
                        .iter()
                        .map(|range| DirectExportInstruction::Clear {
                            current: range.component_base,
                            component_count: range.component_count,
                        }),
                );
            }
            cleared_stage = Some(group.stage);
        }
        let start = group.row_start as usize;
        let end = start
            .checked_add(group.row_count as usize)
            .ok_or_else(|| invalid("row group overflows"))?;
        let executor = executors.get(&group.direct_executor_id);
        match group.role {
            DirectExecutorRole::Source => {
                for id in start..end {
                    body.push(DirectExportInstruction::Source {
                        source: source_rows.len() as u32,
                    });
                    source_rows.push(id as u32);
                }
            }
            DirectExecutorRole::Contribution => {
                if let Some(fanout) = fanout.filter(|_| group.direct_executor_id != DIRECT_NONE_U32)
                {
                    let executor =
                        executor.ok_or_else(|| invalid("contribution executor is absent"))?;
                    for step in fanout.group_steps(group)? {
                        match *step {
                            DirectContributionFanoutStep::Direct {
                                row_start,
                                row_count,
                            } => {
                                for row in &plan.contributions()
                                    [row_start as usize..row_start as usize + row_count as usize]
                                {
                                    body.push(contribution_instruction(
                                        row,
                                        executor,
                                        factor(row.exact_factor_id)?,
                                        &mut bindings,
                                    )?);
                                }
                            }
                            DirectContributionFanoutStep::Fanout {
                                representative_start,
                                representative_count,
                                run_start,
                                run_count,
                                destination_component_count,
                                ..
                            } => {
                                for row in &fanout.representatives[representative_start as usize
                                    ..representative_start as usize + representative_count as usize]
                                {
                                    let resolved_factor =
                                        if row.exact_factor_id as usize == factors.len() {
                                            ExportComplex::ONE
                                        } else {
                                            factor(row.exact_factor_id)?
                                        };
                                    body.push(contribution_instruction(
                                        row,
                                        executor,
                                        resolved_factor,
                                        &mut bindings,
                                    )?);
                                }
                                for run in &fanout.runs
                                    [run_start as usize..run_start as usize + run_count as usize]
                                {
                                    for bundle in &fanout.bundles[run.bundle_start as usize
                                        ..run.bundle_start as usize + run.bundle_count as usize]
                                    {
                                        let resolved_factor = if bundle
                                            .effective_factor_id_or_sentinel
                                            == DIRECT_NONE_U32
                                        {
                                            ExportComplex::ONE
                                        } else {
                                            factor(bundle.effective_factor_id_or_sentinel)?
                                        };
                                        for row in &plan.contributions()[bundle.row_start as usize
                                            ..bundle.row_start as usize + bundle.row_count as usize]
                                        {
                                            body.push(DirectExportInstruction::Scatter {
                                                current: row.destination_component_base,
                                                source: run.scratch_component_base,
                                                component_count: destination_component_count,
                                                mode: contribution_mode(row.flags),
                                                factor: resolved_factor,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    for row in plan
                        .contributions()
                        .get(start..end)
                        .ok_or_else(|| invalid("contribution range is absent"))?
                    {
                        let factor = factor(row.exact_factor_id)?;
                        if row.flags & DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE != 0 {
                            body.push(DirectExportInstruction::Copy {
                                current: row.destination_component_base,
                                source: row.parent0_component_base,
                                component_count: row.parent0_momentum_form_id,
                                factor,
                            });
                        } else {
                            body.push(contribution_instruction(
                                row,
                                executor
                                    .ok_or_else(|| invalid("contribution executor is absent"))?,
                                factor,
                                &mut bindings,
                            )?);
                        }
                    }
                }
            }
            DirectExecutorRole::Finalization => {
                for row in plan
                    .finalizations()
                    .get(start..end)
                    .ok_or_else(|| invalid("finalization range is absent"))?
                {
                    let factor = factor(row.exact_factor_id)?;
                    let executor =
                        executor.ok_or_else(|| invalid("finalization executor is absent"))?;
                    match executor {
                        DirectExportExecutor::Intrinsic { template, .. }
                            if template == "rusticol.identity-finalize-in-place.v1" =>
                        {
                            body.push(DirectExportInstruction::Scale {
                                current: row.component_base,
                                component_count: u32::from(row.component_count),
                                factor,
                            })
                        }
                        DirectExportExecutor::Intrinsic { template, .. } => {
                            body.push(DirectExportInstruction::Propagator {
                                propagator: template.clone(),
                                current: row.component_base,
                                component_count: u32::from(row.component_count),
                                momentum_form: row.momentum_form_id,
                                factor,
                            })
                        }
                        DirectExportExecutor::Kernel { .. } => body.push(kernel_instruction(
                            executor,
                            Operands {
                                left: row.component_base,
                                right: DIRECT_NONE_U32,
                                left_momentum: row.momentum_form_id,
                                right_momentum: DIRECT_NONE_U32,
                                destination: row.component_base,
                            },
                            factor,
                            DirectExportMode::Set,
                            group.role,
                            &mut bindings,
                        )?),
                        _ => return Err(invalid("source executor used as finalization")),
                    }
                }
            }
            DirectExecutorRole::Closure => {
                for row in plan
                    .closures()
                    .get(start..end)
                    .ok_or_else(|| invalid("closure range is absent"))?
                {
                    let row_factor = factor(row.exact_factor_id)?;
                    let executor = executor.ok_or_else(|| invalid("closure executor is absent"))?;
                    match executor {
                        DirectExportExecutor::Intrinsic { template, .. }
                            if template.starts_with("rusticol.closure-reduce.v1:") =>
                        {
                            let component_factors = (0..u32::from(row.component_count))
                                .map(|component| factor(row.component_factor_start + component))
                                .collect::<RusticolResult<Vec<_>>>()?;
                            body.push(DirectExportInstruction::Closure {
                                amplitude: row.amplitude_destination_id,
                                left: row.parent0_component_base,
                                right: row.parent1_component_base_or_sentinel,
                                component_factors,
                                factor: row_factor,
                            });
                        }
                        DirectExportExecutor::Kernel { .. } => body.push(kernel_instruction(
                            executor,
                            Operands {
                                left: row.parent0_component_base,
                                right: row.parent1_component_base_or_sentinel,
                                left_momentum: row.parent0_momentum_form_id,
                                right_momentum: row.parent1_momentum_form_id_or_sentinel,
                                destination: row.amplitude_destination_id,
                            },
                            row_factor,
                            DirectExportMode::Add,
                            group.role,
                            &mut bindings,
                        )?),
                        _ => return Err(invalid("unsupported closure executor")),
                    }
                }
            }
        }
    }
    Ok((body, bindings, source_rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recurrence::direct_backend::DirectCurrentClearRange;

    #[test]
    fn recurrence_export_preserves_bound_stage_clear_and_row_order() {
        let plan = crate::recurrence::valid_direct_plan_fixture();
        let executors = BTreeMap::from([
            (0, DirectExportExecutor::Source),
            (
                1,
                DirectExportExecutor::Intrinsic {
                    template: "vertex".into(),
                    coupling: ExportComplex { re: 0.5, im: -0.25 },
                },
            ),
            (
                2,
                DirectExportExecutor::Intrinsic {
                    template: "rusticol.identity-finalize-in-place.v1".into(),
                    coupling: ExportComplex::ONE,
                },
            ),
        ]);
        let clears = [DirectStageClearRanges {
            stage: 1,
            ranges: vec![DirectCurrentClearRange {
                component_base: 2,
                component_count: 1,
            }]
            .into_boxed_slice(),
        }];
        let (body, _, source_rows) =
            export_direct_body(&plan, &plan.row_groups()[..3], &clears, &executors, None).unwrap();
        assert_eq!(source_rows, [0]);
        assert!(matches!(
            body[0],
            DirectExportInstruction::ClearAmplitude { amplitude: 0 }
        ));
        assert!(matches!(body[1], DirectExportInstruction::Source { .. }));
        assert!(matches!(
            body[2],
            DirectExportInstruction::Clear {
                current: 2,
                component_count: 1
            }
        ));
        assert!(matches!(
            body[3],
            DirectExportInstruction::Vertex {
                current: 2,
                left: 0,
                right: None,
                mode: DirectExportMode::Set,
                ..
            }
        ));
        assert!(matches!(
            body[4],
            DirectExportInstruction::Scale {
                current: 2,
                component_count: 1,
                factor: ExportComplex::ONE
            }
        ));
        assert_eq!(body.len(), 5);
        let restored: Vec<DirectExportInstruction> =
            serde_json::from_str(&serde_json::to_string(&body).unwrap()).unwrap();
        assert_eq!(restored, body);
    }

    #[test]
    fn recurrence_export_kernel_resolves_inputs_and_direct_alias_policy() {
        let executor = DirectExportExecutor::Kernel {
            sha256: "abcd".into(),
            inputs: vec![
                RelativeKernelInput::Parent {
                    parent: 0,
                    component: 1,
                    imaginary: true,
                },
                RelativeKernelInput::Momentum {
                    operand: 0,
                    component: 3,
                },
                RelativeKernelInput::Factor { imaginary: true },
            ],
            output_projections: vec![
                RelativeKernelInput::Destination {
                    component: 0,
                    imaginary: false,
                    amplitude: false,
                },
                RelativeKernelInput::Destination {
                    component: 0,
                    imaginary: true,
                    amplitude: false,
                },
            ],
        };
        let operands = || Operands {
            left: 4,
            right: DIRECT_NONE_U32,
            left_momentum: 2,
            right_momentum: DIRECT_NONE_U32,
            destination: 8,
        };
        let mut bindings = Vec::new();
        let instruction = kernel_instruction(
            &executor,
            operands(),
            ExportComplex::ONE,
            DirectExportMode::Set,
            DirectExecutorRole::Finalization,
            &mut bindings,
        )
        .unwrap();
        let DirectExportInstruction::Kernel {
            binding,
            destination,
            left,
            right,
            left_momentum,
            right_momentum,
            amplitude,
            ..
        } = instruction
        else {
            panic!("expected kernel")
        };
        assert_eq!(
            (
                binding,
                destination,
                left,
                right,
                left_momentum,
                right_momentum,
                amplitude
            ),
            (0, 8, Some(4), None, Some(2), None, false)
        );
        assert!(bindings[0].direct);
        assert_eq!(
            bindings[0]
                .inputs
                .iter()
                .map(|input| input.resolve(&operands(), ExportComplex::ONE).unwrap())
                .collect::<Vec<_>>(),
            vec![
                DirectExportKernelInput::Current {
                    component: 5,
                    imaginary: true
                },
                DirectExportKernelInput::Momentum {
                    form: 2,
                    component: 3
                },
                DirectExportKernelInput::Literal { value: 0.0 },
            ]
        );
        // A second row with different physical parents/destination reuses the
        // relative layout, including the ordered parent and momentum operands.
        let mut second = operands();
        second.left = 11;
        second.destination = 15;
        let _ = kernel_instruction(
            &executor,
            second,
            ExportComplex::ONE,
            DirectExportMode::Set,
            DirectExecutorRole::Finalization,
            &mut bindings,
        )
        .unwrap();
        assert_eq!(bindings.len(), 1);
        let mut aliases = operands();
        aliases.destination = 5;
        for (ops, factor, mode, role) in [
            (
                aliases,
                ExportComplex::ONE,
                DirectExportMode::Set,
                DirectExecutorRole::Finalization,
            ),
            (
                operands(),
                ExportComplex { re: 1.0, im: -0.0 },
                DirectExportMode::Set,
                DirectExecutorRole::Finalization,
            ),
            (
                operands(),
                ExportComplex::ONE,
                DirectExportMode::Add,
                DirectExecutorRole::Contribution,
            ),
        ] {
            let DirectExportInstruction::Kernel { binding, .. } =
                kernel_instruction(&executor, ops, factor, mode, role, &mut bindings).unwrap()
            else {
                panic!("expected kernel")
            };
            assert!(!bindings[binding as usize].direct);
        }
        assert_eq!(bindings.len(), 2);
        let roundtrip: Vec<DirectExportKernelBinding> =
            serde_json::from_str(&serde_json::to_string(&bindings).unwrap()).unwrap();
        assert_eq!(roundtrip, bindings);
        assert_ne!(
            RelativeKernelInput::Literal { value: 0.0 },
            RelativeKernelInput::Literal { value: -0.0 }
        );
    }

    #[test]
    fn recurrence_export_lowers_certified_reuse_with_zero_and_unit_factors() {
        for factor in [
            super::super::ExactComplexRational::ZERO,
            super::super::ExactComplexRational::ONE,
        ] {
            let mut parts = crate::recurrence::valid_direct_plan_parts_fixture();
            parts.currents[0].component_count = 1;
            parts.contributions[0].flags = DIRECT_CONTRIBUTION_FLAG_CERTIFIED_REUSE
                | DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION;
            parts.contributions[0].parent0_momentum_form_id = 1;
            parts.contributions[0].exact_factor_id = parts.exact_factors.len() as u32;
            parts.exact_factors.push(factor);
            parts.row_groups[1].direct_executor_id = DIRECT_NONE_U32;
            let plan = DirectRecurrencePlan::new(parts).unwrap();
            let (body, bindings, sources) =
                export_direct_body(&plan, &plan.row_groups()[1..2], &[], &BTreeMap::new(), None)
                    .unwrap();
            assert!(bindings.is_empty());
            assert!(sources.is_empty());
            assert_eq!(body.len(), 2);
            let DirectExportInstruction::Copy {
                current,
                source,
                component_count,
                factor: resolved,
            } = body[1]
            else {
                panic!("expected certified copy")
            };
            assert_eq!((current, source, component_count), (2, 0, 1));
            assert_eq!(
                resolved.re,
                if factor == super::super::ExactComplexRational::ZERO {
                    0.0
                } else {
                    1.0
                }
            );
            assert_eq!(resolved.im.to_bits(), 0.0_f64.to_bits());
        }
    }

    #[test]
    fn recurrence_export_expands_batched_fanout_into_scratch_then_scatter() {
        use crate::recurrence::direct_backend::{
            DirectArenaView, DirectContributionExecutionMetadata, DirectExecutorCatalog,
            DirectExecutorHandle, DirectFactorView, DirectMomentumView, DirectParameterView,
        };
        unsafe extern "C" fn never_execute<T>(
            _: *const std::ffi::c_void,
            _: DirectArenaView,
            _: DirectMomentumView,
            _: DirectParameterView,
            _: DirectFactorView,
            _: *const T,
            _: u32,
            _: u32,
        ) -> i32 {
            panic!("export must not execute")
        }
        let mut parts = crate::recurrence::valid_direct_plan_parts_fixture();
        let mut second = parts.contributions[0];
        second.flags = 0;
        parts.contributions.push(second);
        parts.row_groups[1].row_count = 2;
        let plan = DirectRecurrencePlan::new(parts).unwrap();
        let handles = vec![
            Some(DirectExecutorHandle::Source {
                call: never_execute,
                context: std::ptr::null(),
            }),
            Some(DirectExecutorHandle::Contribution {
                call: never_execute,
                context: std::ptr::null(),
            }),
            Some(DirectExecutorHandle::Finalization {
                call: never_execute,
                context: std::ptr::null(),
            }),
            Some(DirectExecutorHandle::Closure {
                call: never_execute,
                context: std::ptr::null(),
            }),
        ];
        let catalog = DirectExecutorCatalog::new_sparse_with_metadata(
            &plan,
            plan.direct_template_catalog_digest(),
            handles,
            vec![
                None,
                Some(DirectContributionExecutionMetadata::new(1, false).unwrap()),
                None,
                None,
            ],
        )
        .unwrap();
        let fanout = DirectContributionFanoutProgram::build(&plan, &catalog).unwrap();
        assert_eq!(fanout.scratch_component_count(), 1);
        let executors = BTreeMap::from([(
            1,
            DirectExportExecutor::Intrinsic {
                template: "vertex".into(),
                coupling: ExportComplex { re: 0.3, im: 0.2 },
            },
        )]);
        let (body, _, _) = export_direct_body(
            &plan,
            &plan.row_groups()[1..2],
            &[],
            &executors,
            Some(&fanout),
        )
        .unwrap();
        assert_eq!(body.len(), 4);
        assert!(matches!(
            body[1],
            DirectExportInstruction::Vertex {
                current: 3,
                mode: DirectExportMode::Set,
                factor: ExportComplex::ONE,
                ..
            }
        ));
        assert!(matches!(
            body[2],
            DirectExportInstruction::Scatter {
                current: 2,
                source: 3,
                mode: DirectExportMode::Set,
                factor: ExportComplex::ONE,
                ..
            }
        ));
        assert!(matches!(
            body[3],
            DirectExportInstruction::Scatter {
                current: 2,
                source: 3,
                mode: DirectExportMode::Add,
                factor: ExportComplex::ONE,
                ..
            }
        ));
    }
}
