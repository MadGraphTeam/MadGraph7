// SPDX-License-Identifier: 0BSD

use super::{
    EagerComplex64, EagerDirectClosureSpec, EagerKernelInput, EagerKernelRole, EagerKernelSpec,
    EagerPlanDefinition, EagerPlanPayloads, EagerReductionEntry, EagerReductionGroup,
    EagerStagePayload,
};
use crate::{
    EagerAttachmentRow, EagerClosureRow, EagerCouplingRow, EagerFinalizationRow,
    EagerInvocationRow, MISSING_U32, RusticolError, RusticolResult,
};
use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ComponentRange {
    pub(super) start: usize,
    pub(super) len: usize,
}

#[derive(Clone, Debug)]
pub(super) struct ComponentLayout {
    slots: Vec<ComponentRange>,
    pub(super) component_count: usize,
}

impl ComponentLayout {
    fn new(name: &str, component_counts: &[u32]) -> RusticolResult<Self> {
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(component_counts.len())
            .map_err(|error| {
                RusticolError::artifact(format!(
                    "could not reserve eager {name} slot layout: {error}"
                ))
            })?;
        let mut start = 0usize;
        for (slot_id, count) in component_counts.iter().copied().enumerate() {
            let len = usize::try_from(count).map_err(|_| {
                RusticolError::artifact(format!(
                    "eager {name} slot {slot_id} component count does not fit usize"
                ))
            })?;
            if len == 0 {
                return Err(RusticolError::artifact(format!(
                    "eager {name} slot {slot_id} has zero components"
                )));
            }
            slots.push(ComponentRange { start, len });
            start = start.checked_add(len).ok_or_else(|| {
                RusticolError::artifact(format!("eager {name} component layout overflows usize"))
            })?;
        }
        Ok(Self {
            slots,
            component_count: start,
        })
    }

    fn get(&self, id: u32, context: &str) -> RusticolResult<ComponentRange> {
        let index = usize::try_from(id).map_err(|_| {
            RusticolError::artifact(format!("{context} id {id} does not fit usize"))
        })?;
        self.slots.get(index).copied().ok_or_else(|| {
            RusticolError::artifact(format!("{context} references unknown slot {id}"))
        })
    }
}

#[derive(Clone, Debug)]
pub(super) struct ScheduledInvocation {
    pub(super) row: EagerInvocationRow,
    pub(super) left_values: ComponentRange,
    pub(super) right_values: ComponentRange,
    pub(super) left_momenta: ComponentRange,
    pub(super) right_momenta: ComponentRange,
    pub(super) attachment_range: Range<usize>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ScheduledAttachment {
    pub(super) row: EagerAttachmentRow,
    pub(super) current: ComponentRange,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct FinalizationCopy {
    pub(super) current: ComponentRange,
    pub(super) unpropagated: ComponentRange,
}

#[derive(Clone, Debug)]
pub(super) struct ScheduledFinalization {
    pub(super) row: EagerFinalizationRow,
    pub(super) current: ComponentRange,
    pub(super) propagated: Option<ComponentRange>,
    pub(super) momentum: ComponentRange,
}

#[derive(Clone, Debug)]
pub(super) struct ScheduledClosure {
    pub(super) row: EagerClosureRow,
    pub(super) left_values: ComponentRange,
    pub(super) right_values: ComponentRange,
}

#[derive(Clone, Debug)]
pub(super) struct ScheduledDirectClosure {
    pub(super) row: EagerClosureRow,
    pub(super) left_values: ComponentRange,
    pub(super) right_values: ComponentRange,
    pub(super) coefficients: Vec<EagerComplex64>,
}

#[derive(Clone, Debug)]
pub(super) struct EagerStagePlan {
    pub(super) stage_index: u32,
    pub(super) invocations: Vec<ScheduledInvocation>,
    pub(super) attachments: Vec<ScheduledAttachment>,
    pub(super) finalization_copies: Vec<FinalizationCopy>,
    pub(super) finalizations: Vec<ScheduledFinalization>,
}

#[derive(Clone, Debug)]
pub struct EagerExecutionPlan {
    pub(super) values: ComponentLayout,
    pub(super) momenta: ComponentLayout,
    pub(super) currents: ComponentLayout,
    pub(super) parameter_count: usize,
    pub(super) amplitude_count: usize,
    pub(super) kernels: BTreeMap<u32, EagerKernelSpec>,
    pub(super) couplings: Vec<EagerCouplingRow>,
    pub(super) stages: Vec<EagerStagePlan>,
    pub(super) closures: Vec<ScheduledClosure>,
    pub(super) direct_closures: Vec<ScheduledDirectClosure>,
    pub(super) reduction_groups: Vec<EagerReductionGroup>,
    pub(super) reduction_entries: Vec<EagerReductionEntry>,
}

impl EagerExecutionPlan {
    pub fn from_payloads(
        definition: EagerPlanDefinition,
        payloads: EagerPlanPayloads<'_>,
    ) -> RusticolResult<Self> {
        let values =
            ComponentLayout::new("value", &definition.dimensions.value_slot_component_counts)?;
        let momenta = ComponentLayout::new(
            "momentum",
            &definition.dimensions.momentum_slot_component_counts,
        )?;
        let currents =
            ComponentLayout::new("current", &definition.dimensions.current_component_counts)?;
        let parameter_count = usize::try_from(definition.dimensions.parameter_count)
            .map_err(|_| RusticolError::artifact("eager parameter count does not fit usize"))?;
        let amplitude_count = usize::try_from(definition.dimensions.amplitude_count)
            .map_err(|_| RusticolError::artifact("eager amplitude count does not fit usize"))?;
        if amplitude_count == 0 {
            return Err(RusticolError::artifact(
                "eager execution plan has no amplitude outputs",
            ));
        }

        let kernels = validate_kernel_specs(&definition.kernels, parameter_count)?;
        let couplings = EagerCouplingRow::decode_table(payloads.couplings)?;
        validate_couplings(&couplings, parameter_count)?;

        let mut stages = Vec::new();
        stages
            .try_reserve_exact(payloads.stages.len())
            .map_err(|error| {
                RusticolError::artifact(format!("could not reserve eager stages: {error}"))
            })?;
        let mut previous_stage = None;
        let mut finalized_currents = BTreeSet::new();
        let mut stored_value_slots = BTreeSet::new();
        for payload in payloads.stages {
            if previous_stage.is_some_and(|previous| payload.stage_index <= previous) {
                return Err(RusticolError::artifact(
                    "eager stage indices must be strictly increasing",
                ));
            }
            previous_stage = Some(payload.stage_index);
            stages.push(load_stage(
                *payload,
                &values,
                &momenta,
                &currents,
                &kernels,
                couplings.len(),
                parameter_count,
                &mut finalized_currents,
                &mut stored_value_slots,
            )?);
        }

        let closure_rows = EagerClosureRow::decode_table(payloads.closures)?;
        let (closures, direct_closures) = load_closures(
            &closure_rows,
            &definition.direct_closures,
            &values,
            &kernels,
            couplings.len(),
            amplitude_count,
            parameter_count,
        )?;
        validate_reduction_plan(
            &definition.reduction_groups,
            &definition.reduction_entries,
            amplitude_count,
        )?;

        Ok(Self {
            values,
            momenta,
            currents,
            parameter_count,
            amplitude_count,
            kernels,
            couplings,
            stages,
            closures,
            direct_closures,
            reduction_groups: definition.reduction_groups,
            reduction_entries: definition.reduction_entries,
        })
    }

    pub fn value_component_count(&self) -> usize {
        self.values.component_count
    }

    pub fn momentum_component_count(&self) -> usize {
        self.momenta.component_count
    }

    pub fn current_component_count(&self) -> usize {
        self.currents.component_count
    }

    pub fn parameter_count(&self) -> usize {
        self.parameter_count
    }

    pub fn amplitude_count(&self) -> usize {
        self.amplitude_count
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    pub fn stage_indices(&self) -> impl ExactSizeIterator<Item = u32> + '_ {
        self.stages.iter().map(|stage| stage.stage_index)
    }

    pub fn invocation_count(&self) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.invocations.len())
            .sum()
    }

    pub fn attachment_count(&self) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.attachments.len())
            .sum()
    }

    pub fn closure_count(&self) -> usize {
        self.closures.len() + self.direct_closures.len()
    }

    pub fn reduction_group_count(&self) -> usize {
        self.reduction_groups.len()
    }

    pub fn reduction_entry_count(&self) -> usize {
        self.reduction_entries.len()
    }
}

fn validate_kernel_specs(
    specs: &[EagerKernelSpec],
    parameter_count: usize,
) -> RusticolResult<BTreeMap<u32, EagerKernelSpec>> {
    let mut kernels = BTreeMap::new();
    for (index, spec) in specs.iter().cloned().enumerate() {
        if spec.kernel_id == MISSING_U32 {
            return Err(RusticolError::artifact(format!(
                "eager kernel spec {index} uses the reserved missing id"
            )));
        }
        if spec.inputs.is_empty() || spec.output_component_count == 0 {
            return Err(RusticolError::artifact(format!(
                "eager kernel {} has a zero input or output width",
                spec.kernel_id
            )));
        }
        let mut unique_inputs = BTreeSet::new();
        for input in &spec.inputs {
            if !unique_inputs.insert(*input) {
                return Err(RusticolError::artifact(format!(
                    "eager kernel {} repeats input descriptor {input:?}",
                    spec.kernel_id
                )));
            }
            if let EagerKernelInput::ModelParameter(parameter_id) = input {
                required_index(
                    *parameter_id,
                    parameter_count,
                    &format!("eager kernel {} model parameter", spec.kernel_id),
                )?;
            }
        }
        let kernel_id = spec.kernel_id;
        if kernels.insert(kernel_id, spec).is_some() {
            return Err(RusticolError::artifact(format!(
                "duplicate eager kernel id {kernel_id}"
            )));
        }
    }
    Ok(kernels)
}

fn validate_couplings(rows: &[EagerCouplingRow], parameter_count: usize) -> RusticolResult<()> {
    for (index, row) in rows.iter().enumerate() {
        validate_optional_index(
            row.real_parameter_id,
            parameter_count,
            &format!("eager coupling {index} real parameter"),
        )?;
        validate_optional_index(
            row.imag_parameter_id,
            parameter_count,
            &format!("eager coupling {index} imaginary parameter"),
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn load_stage(
    payload: EagerStagePayload<'_>,
    values: &ComponentLayout,
    momenta: &ComponentLayout,
    currents: &ComponentLayout,
    kernels: &BTreeMap<u32, EagerKernelSpec>,
    coupling_count: usize,
    parameter_count: usize,
    globally_finalized_currents: &mut BTreeSet<u32>,
    globally_stored_value_slots: &mut BTreeSet<u32>,
) -> RusticolResult<EagerStagePlan> {
    let rows = EagerInvocationRow::decode_table(payload.invocations)?;
    let attachments = EagerAttachmentRow::decode_table(payload.attachments)?;
    let finalization_rows = EagerFinalizationRow::decode_table(payload.finalizations)?;
    let mut invocations = Vec::new();
    invocations.try_reserve_exact(rows.len()).map_err(|error| {
        RusticolError::artifact(format!("could not reserve eager invocations: {error}"))
    })?;
    let mut attachment_cursor = 0usize;
    let mut attached_currents = BTreeSet::new();
    let mut scheduled_attachments = Vec::new();
    scheduled_attachments
        .try_reserve_exact(attachments.len())
        .map_err(|error| {
            RusticolError::artifact(format!("could not reserve eager attachments: {error}"))
        })?;

    for (index, row) in rows.into_iter().enumerate() {
        let kernel = require_kernel(
            kernels,
            row.kernel_id,
            EagerKernelRole::Vertex,
            "invocation",
        )?;
        let left_values = values.get(row.left_value_slot_id, "eager invocation left value")?;
        let right_values = values.get(row.right_value_slot_id, "eager invocation right value")?;
        let left_momenta =
            momenta.get(row.left_momentum_slot_id, "eager invocation left momentum")?;
        let right_momenta = momenta.get(
            row.right_momentum_slot_id,
            "eager invocation right momentum",
        )?;
        let coupling_index = required_index(
            row.coupling_slot_id,
            coupling_count,
            "eager invocation coupling",
        )?;
        let _ = coupling_index;
        let attachment_start = usize::try_from(row.attachment_start).map_err(|_| {
            RusticolError::artifact(format!(
                "eager invocation {index} attachment start does not fit usize"
            ))
        })?;
        let attachment_count = usize::try_from(row.attachment_count).map_err(|_| {
            RusticolError::artifact(format!(
                "eager invocation {index} attachment count does not fit usize"
            ))
        })?;
        if attachment_count == 0 {
            return Err(RusticolError::artifact(format!(
                "eager invocation {index} has no attachments"
            )));
        }
        if attachment_start != attachment_cursor {
            return Err(RusticolError::artifact(format!(
                "eager invocation {index} attachment range is not contiguous"
            )));
        }
        let attachment_end = attachment_start
            .checked_add(attachment_count)
            .ok_or_else(|| {
                RusticolError::artifact(format!(
                    "eager invocation {index} attachment range overflows usize"
                ))
            })?;
        if attachment_end > attachments.len() {
            return Err(RusticolError::artifact(format!(
                "eager invocation {index} attachment range exceeds the table"
            )));
        }
        validate_kernel_inputs(
            kernel,
            KernelInputBounds {
                first_current: left_values.len,
                second_current: right_values.len,
                first_momentum: left_momenta.len,
                second_momentum: right_momenta.len,
                has_coupling: true,
                parameter_count,
            },
        )?;
        for attachment in &attachments[attachment_start..attachment_end] {
            let current = currents.get(
                attachment.result_current_id,
                "eager attachment result current",
            )?;
            if current.len
                != usize::try_from(kernel.output_component_count).map_err(|_| {
                    RusticolError::artifact("eager kernel output width does not fit usize")
                })?
            {
                return Err(RusticolError::artifact(format!(
                    "eager invocation kernel {} output width does not match current {}",
                    kernel.kernel_id, attachment.result_current_id
                )));
            }
            attached_currents.insert(attachment.result_current_id);
            scheduled_attachments.push(ScheduledAttachment {
                row: *attachment,
                current,
            });
        }
        invocations.push(ScheduledInvocation {
            row,
            left_values,
            right_values,
            left_momenta,
            right_momenta,
            attachment_range: attachment_start..attachment_end,
        });
        attachment_cursor = attachment_end;
    }
    if attachment_cursor != attachments.len() {
        return Err(RusticolError::artifact(
            "eager invocation ranges do not cover the attachment table",
        ));
    }

    let mut finalizations = Vec::new();
    let mut finalization_copies = Vec::new();
    finalizations
        .try_reserve_exact(finalization_rows.len())
        .map_err(|error| {
            RusticolError::artifact(format!("could not reserve eager finalizations: {error}"))
        })?;
    let mut finalized_currents = BTreeSet::new();
    for (index, row) in finalization_rows.into_iter().enumerate() {
        if !finalized_currents.insert(row.current_id) {
            return Err(RusticolError::artifact(format!(
                "eager current {} is finalized more than once in stage {}",
                row.current_id, payload.stage_index
            )));
        }
        if !globally_finalized_currents.insert(row.current_id) {
            return Err(RusticolError::artifact(format!(
                "eager current {} is finalized in more than one stage",
                row.current_id
            )));
        }
        if row.unpropagated_value_slot_id != MISSING_U32
            && row.unpropagated_value_slot_id == row.propagated_value_slot_id
        {
            return Err(RusticolError::artifact(format!(
                "eager finalization {index} aliases propagated and unpropagated outputs"
            )));
        }
        for value_slot_id in [row.unpropagated_value_slot_id, row.propagated_value_slot_id] {
            if value_slot_id != MISSING_U32 && !globally_stored_value_slots.insert(value_slot_id) {
                return Err(RusticolError::artifact(format!(
                    "eager value slot {value_slot_id} is written by more than one finalization"
                )));
            }
        }
        let current = currents.get(row.current_id, "eager finalization current")?;
        let unpropagated = optional_component_range(
            values,
            row.unpropagated_value_slot_id,
            "eager unpropagated value",
        )?;
        let propagated = optional_component_range(
            values,
            row.propagated_value_slot_id,
            "eager propagated value",
        )?;
        if unpropagated.is_none() && propagated.is_none() {
            return Err(RusticolError::artifact(format!(
                "eager finalization {index} stores no current value"
            )));
        }
        for output in [unpropagated, propagated].into_iter().flatten() {
            if output.len != current.len {
                return Err(RusticolError::artifact(format!(
                    "eager finalization {index} output width does not match current width"
                )));
            }
        }
        let momentum = momenta.get(row.momentum_slot_id, "eager finalization momentum")?;
        if let Some(unpropagated) = unpropagated {
            finalization_copies.push(FinalizationCopy {
                current,
                unpropagated,
            });
        }
        if row.applies_kernel() {
            let kernel = require_kernel(
                kernels,
                row.kernel_id,
                EagerKernelRole::Finalization,
                "finalization",
            )?;
            if propagated.is_none() {
                return Err(RusticolError::artifact(format!(
                    "eager finalization {index} applies a kernel without a propagated output"
                )));
            }
            validate_kernel_inputs(
                kernel,
                KernelInputBounds {
                    first_current: current.len,
                    second_current: 0,
                    first_momentum: momentum.len,
                    second_momentum: 0,
                    has_coupling: false,
                    parameter_count,
                },
            )?;
            if usize::try_from(kernel.output_component_count).ok() != Some(current.len) {
                return Err(RusticolError::artifact(format!(
                    "eager finalization kernel {} output width does not match current width",
                    kernel.kernel_id
                )));
            }
            finalizations.push(ScheduledFinalization {
                row,
                current,
                propagated,
                momentum,
            });
        } else if propagated.is_some() {
            return Err(RusticolError::artifact(format!(
                "eager finalization {index} has a propagated output but no kernel"
            )));
        }
    }
    if !attached_currents.is_subset(&finalized_currents) {
        let missing = attached_currents
            .difference(&finalized_currents)
            .next()
            .copied()
            .unwrap_or(MISSING_U32);
        return Err(RusticolError::artifact(format!(
            "eager stage {} does not finalize attached current {missing}",
            payload.stage_index
        )));
    }

    // Stable sorting makes equal-kernel rows contiguous for packetization while
    // preserving artifact order, which defines deterministic lane ordering.
    invocations.sort_by_key(|item| item.row.kernel_id);
    finalizations.sort_by_key(|item| item.row.kernel_id);
    Ok(EagerStagePlan {
        stage_index: payload.stage_index,
        invocations,
        attachments: scheduled_attachments,
        finalization_copies,
        finalizations,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_closures(
    rows: &[EagerClosureRow],
    direct_specs: &[EagerDirectClosureSpec],
    values: &ComponentLayout,
    kernels: &BTreeMap<u32, EagerKernelSpec>,
    coupling_count: usize,
    amplitude_count: usize,
    parameter_count: usize,
) -> RusticolResult<(Vec<ScheduledClosure>, Vec<ScheduledDirectClosure>)> {
    let mut direct_by_index = BTreeMap::new();
    for spec in direct_specs {
        let index = usize::try_from(spec.closure_index).map_err(|_| {
            RusticolError::artifact("eager direct closure index does not fit usize")
        })?;
        if spec.coefficients.is_empty() {
            return Err(RusticolError::artifact(format!(
                "eager direct closure {index} has no coefficients"
            )));
        }
        if spec
            .coefficients
            .iter()
            .any(|coefficient| !complex_is_finite(*coefficient))
        {
            return Err(RusticolError::artifact(format!(
                "eager direct closure {index} has a non-finite coefficient"
            )));
        }
        if direct_by_index.insert(index, spec).is_some() {
            return Err(RusticolError::artifact(format!(
                "duplicate eager direct closure specification {index}"
            )));
        }
    }

    let mut closures = Vec::new();
    let mut direct_closures = Vec::new();
    for (index, row) in rows.iter().copied().enumerate() {
        let left_values = values.get(row.left_value_slot_id, "eager closure left value")?;
        let right_values = values.get(row.right_value_slot_id, "eager closure right value")?;
        required_index(
            row.amplitude_index,
            amplitude_count,
            "eager closure amplitude",
        )?;
        if row.kernel_id == MISSING_U32 {
            if row.coupling_slot_id != MISSING_U32 {
                return Err(RusticolError::artifact(format!(
                    "eager direct closure {index} unexpectedly references a coupling"
                )));
            }
            let spec = direct_by_index.remove(&index).ok_or_else(|| {
                RusticolError::artifact(format!(
                    "eager direct closure {index} lacks contraction coefficients"
                ))
            })?;
            if left_values.len != right_values.len || left_values.len != spec.coefficients.len() {
                return Err(RusticolError::artifact(format!(
                    "eager direct closure {index} component widths do not match"
                )));
            }
            direct_closures.push(ScheduledDirectClosure {
                row,
                left_values,
                right_values,
                coefficients: spec.coefficients.clone(),
            });
            continue;
        }
        if direct_by_index.contains_key(&index) {
            return Err(RusticolError::artifact(format!(
                "eager kernel closure {index} has direct-contraction metadata"
            )));
        }
        let kernel = require_kernel(kernels, row.kernel_id, EagerKernelRole::Closure, "closure")?;
        required_index(
            row.coupling_slot_id,
            coupling_count,
            "eager closure coupling",
        )?;
        validate_kernel_inputs(
            kernel,
            KernelInputBounds {
                first_current: left_values.len,
                second_current: right_values.len,
                first_momentum: 0,
                second_momentum: 0,
                has_coupling: true,
                parameter_count,
            },
        )?;
        if kernel.output_component_count != 1 {
            return Err(RusticolError::artifact(format!(
                "eager closure kernel {} must produce one component",
                kernel.kernel_id
            )));
        }
        closures.push(ScheduledClosure {
            row,
            left_values,
            right_values,
        });
    }
    if let Some(index) = direct_by_index.keys().next() {
        return Err(RusticolError::artifact(format!(
            "eager direct closure specification {index} has no matching row"
        )));
    }
    // Keep the same stable equal-kernel ordering contract as stage invocations.
    closures.sort_by_key(|item| item.row.kernel_id);
    Ok((closures, direct_closures))
}

fn validate_reduction_plan(
    groups: &[EagerReductionGroup],
    entries: &[EagerReductionEntry],
    amplitude_count: usize,
) -> RusticolResult<()> {
    if groups.is_empty() || entries.is_empty() {
        return Err(RusticolError::artifact(
            "eager reduction requires nonempty groups and entries",
        ));
    }
    let mut covered = BTreeSet::new();
    for (group_index, group) in groups.iter().enumerate() {
        if group.amplitude_indices.is_empty() {
            return Err(RusticolError::artifact(format!(
                "eager reduction group {group_index} is empty"
            )));
        }
        for amplitude_index in &group.amplitude_indices {
            required_index(
                *amplitude_index,
                amplitude_count,
                "eager reduction amplitude",
            )?;
            if !covered.insert(*amplitude_index) {
                return Err(RusticolError::artifact(format!(
                    "eager amplitude {amplitude_index} belongs to more than one reduction group"
                )));
            }
        }
    }
    if covered.len() != amplitude_count {
        return Err(RusticolError::artifact(format!(
            "eager reduction groups cover {} of {amplitude_count} amplitudes",
            covered.len()
        )));
    }
    for (index, entry) in entries.iter().enumerate() {
        required_index(
            entry.left_group_index,
            groups.len(),
            "eager reduction left group",
        )?;
        required_index(
            entry.right_group_index,
            groups.len(),
            "eager reduction right group",
        )?;
        if !complex_is_finite(entry.coefficient) {
            return Err(RusticolError::artifact(format!(
                "eager reduction entry {index} has a non-finite coefficient"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct KernelInputBounds {
    first_current: usize,
    second_current: usize,
    first_momentum: usize,
    second_momentum: usize,
    has_coupling: bool,
    parameter_count: usize,
}

fn validate_kernel_inputs(
    kernel: &EagerKernelSpec,
    bounds: KernelInputBounds,
) -> RusticolResult<()> {
    for input in &kernel.inputs {
        let (allowed, index, count) = match *input {
            EagerKernelInput::FirstCurrentComponent(index) => {
                (true, Some(index), bounds.first_current)
            }
            EagerKernelInput::SecondCurrentComponent(index) => (
                kernel.role != EagerKernelRole::Finalization,
                Some(index),
                bounds.second_current,
            ),
            EagerKernelInput::FirstMomentumComponent(index) => (
                kernel.role != EagerKernelRole::Closure,
                Some(index),
                bounds.first_momentum,
            ),
            EagerKernelInput::SecondMomentumComponent(index) => (
                kernel.role == EagerKernelRole::Vertex,
                Some(index),
                bounds.second_momentum,
            ),
            EagerKernelInput::CouplingReal | EagerKernelInput::CouplingImag => {
                (bounds.has_coupling, None, 0)
            }
            EagerKernelInput::ModelParameter(index) => (true, Some(index), bounds.parameter_count),
        };
        if !allowed {
            return Err(RusticolError::artifact(format!(
                "eager {:?} kernel {} cannot use input descriptor {input:?}",
                kernel.role, kernel.kernel_id
            )));
        }
        if let Some(index) = index {
            let index = usize::try_from(index).map_err(|_| {
                RusticolError::artifact(format!(
                    "eager kernel {} input descriptor {input:?} does not fit usize",
                    kernel.kernel_id
                ))
            })?;
            if index >= count {
                return Err(RusticolError::artifact(format!(
                    "eager kernel {} input descriptor {input:?} is outside 0..{count}",
                    kernel.kernel_id
                )));
            }
        }
    }
    Ok(())
}

fn require_kernel<'a>(
    kernels: &'a BTreeMap<u32, EagerKernelSpec>,
    kernel_id: u32,
    role: EagerKernelRole,
    context: &str,
) -> RusticolResult<&'a EagerKernelSpec> {
    let kernel = kernels.get(&kernel_id).ok_or_else(|| {
        RusticolError::artifact(format!(
            "eager {context} references unknown kernel {kernel_id}"
        ))
    })?;
    if kernel.role != role {
        return Err(RusticolError::artifact(format!(
            "eager {context} kernel {kernel_id} has role {:?}, expected {role:?}",
            kernel.role
        )));
    }
    Ok(kernel)
}

fn optional_component_range(
    layout: &ComponentLayout,
    id: u32,
    context: &str,
) -> RusticolResult<Option<ComponentRange>> {
    if id == MISSING_U32 {
        return Ok(None);
    }
    layout.get(id, context).map(Some)
}

fn validate_optional_index(id: u32, count: usize, context: &str) -> RusticolResult<()> {
    if id == MISSING_U32 {
        return Ok(());
    }
    required_index(id, count, context).map(|_| ())
}

fn required_index(id: u32, count: usize, context: &str) -> RusticolResult<usize> {
    if id == MISSING_U32 {
        return Err(RusticolError::artifact(format!(
            "{context} uses the reserved missing id"
        )));
    }
    let index = usize::try_from(id)
        .map_err(|_| RusticolError::artifact(format!("{context} id does not fit usize")))?;
    if index >= count {
        return Err(RusticolError::artifact(format!(
            "{context} index {index} is outside 0..{count}"
        )));
    }
    Ok(index)
}

pub(super) fn complex_is_finite(value: EagerComplex64) -> bool {
    value.re.is_finite() && value.im.is_finite()
}
