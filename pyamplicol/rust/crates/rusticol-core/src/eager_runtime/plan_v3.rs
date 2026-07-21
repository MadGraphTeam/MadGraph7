// SPDX-License-Identifier: 0BSD

//! Direct plan-v3 fixed-row adapter for the eager f64 runtime.
//!
//! This module intentionally depends only on the public plan-v3 row model. It
//! does not depend on PACBIN, manifests, engine loader types, JSON, or encoded
//! plan-v2 tables.

use super::plan::{
    ComponentLayout, ComponentRange, EagerStagePlan, FinalizationCopy, ScheduledAttachment,
    ScheduledClosure, ScheduledDirectClosure, ScheduledFinalization, ScheduledInvocation,
    SelectorDomainPlan, mark_initial_amplitude_writes, mark_initial_current_writes,
};
use super::{
    EAGER_INDEPENDENT_BLOCK_SIZE, EagerComplex64, EagerExecutionPlan, EagerKernelInput,
    EagerKernelRole, EagerKernelSpec, EagerReductionEntry, EagerReductionGroup,
};
use crate::{
    EAGER_OUTPUT_FACTOR_NONE, EagerAttachmentRow, EagerClosureRow, EagerCouplingRow,
    EagerFinalizationRow, EagerInvocationRow, EagerPlanAttachmentRow, EagerPlanClosureRow,
    EagerPlanCouplingRow, EagerPlanCurrentRow, EagerPlanDirectCoefficientRow,
    EagerPlanExactFactorRow, EagerPlanFinalizationRow, EagerPlanInvocationRow,
    EagerPlanMomentumRow, EagerPlanParameterRow, EagerPlanReductionEntryKind,
    EagerPlanReductionEntryRow, EagerPlanReductionGroupRow, EagerPlanSelectorDomainRow,
    EagerPlanStageRow, EagerPlanValueRow, EagerValueSlotKind, MISSING_U32, RusticolError,
    RusticolResult,
};
use std::collections::{BTreeMap, BTreeSet};

/// Borrowed execution-relevant sections of a plan-v3 artifact.
///
/// Catalogs used only for source filling, public selector aliases, inspection,
/// or exact provenance are deliberately absent. `exact_factors` retains the
/// bit-exact binary64 values needed to materialize final f64 runtime factors.
#[derive(Clone, Copy, Debug)]
pub struct EagerPlanV3Sections<'a> {
    pub kernels: &'a [EagerKernelSpec],
    /// Final prepared-pack evaluator parameter width after runtime projection
    /// and synthetic coupling-slot insertion.
    pub prepared_parameter_count: u32,
    pub currents: &'a [EagerPlanCurrentRow],
    pub values: &'a [EagerPlanValueRow],
    pub momenta: &'a [EagerPlanMomentumRow],
    pub parameters: &'a [EagerPlanParameterRow],
    pub stages: &'a [EagerPlanStageRow],
    pub couplings: &'a [EagerPlanCouplingRow],
    pub invocations: &'a [EagerPlanInvocationRow],
    pub attachments: &'a [EagerPlanAttachmentRow],
    pub finalizations: &'a [EagerPlanFinalizationRow],
    pub closures: &'a [EagerPlanClosureRow],
    pub direct_coefficients: &'a [EagerPlanDirectCoefficientRow],
    pub selector_domains: &'a [EagerPlanSelectorDomainRow],
    pub selector_memberships: &'a [u32],
    pub reduction_groups: &'a [EagerPlanReductionGroupRow],
    pub reduction_entries: &'a [EagerPlanReductionEntryRow],
    pub exact_factors: &'a [EagerPlanExactFactorRow],
    pub color_contraction_entry_start: u64,
    pub color_contraction_entry_count: u64,
}

impl EagerPlanV3Sections<'_> {
    pub fn to_owned(self) -> EagerOwnedPlanV3Sections {
        EagerOwnedPlanV3Sections {
            kernels: self.kernels.to_vec(),
            prepared_parameter_count: self.prepared_parameter_count,
            currents: self.currents.to_vec(),
            values: self.values.to_vec(),
            momenta: self.momenta.to_vec(),
            parameters: self.parameters.to_vec(),
            stages: self.stages.to_vec(),
            couplings: self.couplings.to_vec(),
            invocations: self.invocations.to_vec(),
            attachments: self.attachments.to_vec(),
            finalizations: self.finalizations.to_vec(),
            closures: self.closures.to_vec(),
            direct_coefficients: self.direct_coefficients.to_vec(),
            selector_domains: self.selector_domains.to_vec(),
            selector_memberships: self.selector_memberships.to_vec(),
            reduction_groups: self.reduction_groups.to_vec(),
            reduction_entries: self.reduction_entries.to_vec(),
            exact_factors: self.exact_factors.to_vec(),
            color_contraction_entry_start: self.color_contraction_entry_start,
            color_contraction_entry_count: self.color_contraction_entry_count,
        }
    }
}

/// Owned counterpart of [`EagerPlanV3Sections`].
#[derive(Clone, Debug, PartialEq)]
pub struct EagerOwnedPlanV3Sections {
    pub kernels: Vec<EagerKernelSpec>,
    pub prepared_parameter_count: u32,
    pub currents: Vec<EagerPlanCurrentRow>,
    pub values: Vec<EagerPlanValueRow>,
    pub momenta: Vec<EagerPlanMomentumRow>,
    pub parameters: Vec<EagerPlanParameterRow>,
    pub stages: Vec<EagerPlanStageRow>,
    pub couplings: Vec<EagerPlanCouplingRow>,
    pub invocations: Vec<EagerPlanInvocationRow>,
    pub attachments: Vec<EagerPlanAttachmentRow>,
    pub finalizations: Vec<EagerPlanFinalizationRow>,
    pub closures: Vec<EagerPlanClosureRow>,
    pub direct_coefficients: Vec<EagerPlanDirectCoefficientRow>,
    pub selector_domains: Vec<EagerPlanSelectorDomainRow>,
    pub selector_memberships: Vec<u32>,
    pub reduction_groups: Vec<EagerPlanReductionGroupRow>,
    pub reduction_entries: Vec<EagerPlanReductionEntryRow>,
    pub exact_factors: Vec<EagerPlanExactFactorRow>,
    pub color_contraction_entry_start: u64,
    pub color_contraction_entry_count: u64,
}

impl EagerOwnedPlanV3Sections {
    pub fn as_borrowed(&self) -> EagerPlanV3Sections<'_> {
        EagerPlanV3Sections {
            kernels: &self.kernels,
            prepared_parameter_count: self.prepared_parameter_count,
            currents: &self.currents,
            values: &self.values,
            momenta: &self.momenta,
            parameters: &self.parameters,
            stages: &self.stages,
            couplings: &self.couplings,
            invocations: &self.invocations,
            attachments: &self.attachments,
            finalizations: &self.finalizations,
            closures: &self.closures,
            direct_coefficients: &self.direct_coefficients,
            selector_domains: &self.selector_domains,
            selector_memberships: &self.selector_memberships,
            reduction_groups: &self.reduction_groups,
            reduction_entries: &self.reduction_entries,
            exact_factors: &self.exact_factors,
            color_contraction_entry_start: self.color_contraction_entry_start,
            color_contraction_entry_count: self.color_contraction_entry_count,
        }
    }

    pub fn into_execution_plan(self) -> RusticolResult<EagerExecutionPlan> {
        EagerExecutionPlan::from_plan_v3_sections(self.as_borrowed())
    }
}

impl EagerExecutionPlan {
    /// Validate plan-v3 fixed rows and materialize final eager f64 semantics.
    pub fn from_plan_v3_sections(input: EagerPlanV3Sections<'_>) -> RusticolResult<Self> {
        PlanBuilder::new(input)?.build()
    }
}

struct PlanBuilder<'a> {
    input: EagerPlanV3Sections<'a>,
    values: ComponentLayout,
    momenta: ComponentLayout,
    currents: ComponentLayout,
    factors: Vec<EagerComplex64>,
    kernels: BTreeMap<u32, EagerKernelSpec>,
    parameter_count: usize,
    amplitude_count: usize,
    selector_plan: SelectorDomainPlan,
}

impl<'a> PlanBuilder<'a> {
    fn new(input: EagerPlanV3Sections<'a>) -> RusticolResult<Self> {
        let current_counts = validate_current_layout(input.currents)?;
        let value_counts = validate_value_layout(input.values, input.currents)?;
        let momentum_counts = validate_momentum_layout(input.momenta)?;
        validate_dense_ids(
            input.parameters.iter().map(|row| row.parameter_id),
            "eager plan-v3 parameter",
        )?;
        let parameter_count = usize::try_from(input.prepared_parameter_count)
            .map_err(|_| artifact("eager prepared parameter count does not fit usize"))?;
        let factors = load_factors(input.exact_factors)?;
        let kernels = validate_kernel_specs(input.kernels, parameter_count)?;
        let amplitude_count = validate_amplitude_layout(input.closures)?;
        let selector_plan = load_selector_plan(
            input.selector_domains,
            input.selector_memberships,
            input.reduction_groups,
        )?;
        Ok(Self {
            input,
            values: ComponentLayout::new("value", &value_counts)?,
            momenta: ComponentLayout::new("momentum", &momentum_counts)?,
            currents: ComponentLayout::new("current", &current_counts)?,
            factors,
            kernels,
            parameter_count,
            amplitude_count,
            selector_plan,
        })
    }

    fn build(self) -> RusticolResult<EagerExecutionPlan> {
        let couplings = self.load_couplings()?;
        let (reduction_groups, reduction_entries) = self.load_reductions()?;
        self.validate_selector_dependency_proof(&reduction_groups)?;
        let (stages, stored_value_slots) = self.load_stages(couplings.len())?;
        let (mut closures, mut direct_closures) = self.load_closures(couplings.len())?;

        let mut initial_value_ranges = Vec::new();
        for slot_id in 0..self.input.values.len() {
            let slot_id = usize_u32(slot_id, "eager value slot")?;
            if !stored_value_slots.contains(&slot_id) {
                initial_value_ranges.push(self.values.get(slot_id, "eager initial value")?);
            }
        }
        let zero_amplitude_indices = mark_initial_amplitude_writes(
            &mut closures,
            &mut direct_closures,
            self.amplitude_count,
        );

        Ok(EagerExecutionPlan {
            values: self.values,
            momenta: self.momenta,
            currents: self.currents,
            parameter_count: self.parameter_count,
            amplitude_count: self.amplitude_count,
            kernels: self.kernels,
            couplings,
            stages,
            closures,
            direct_closures,
            reduction_groups,
            reduction_entries,
            selector_domains: Some(self.selector_plan),
            initial_value_ranges,
            zero_amplitude_indices,
        })
    }

    fn load_couplings(&self) -> RusticolResult<Vec<EagerCouplingRow>> {
        validate_dense_ids(
            self.input.couplings.iter().map(|row| row.coupling_id),
            "eager plan-v3 coupling",
        )?;
        self.input
            .couplings
            .iter()
            .enumerate()
            .map(|(index, row)| {
                validate_optional_index(
                    row.real_parameter_id,
                    self.parameter_count,
                    &format!("eager coupling {index} real parameter"),
                )?;
                validate_optional_index(
                    row.imaginary_parameter_id,
                    self.parameter_count,
                    &format!("eager coupling {index} imaginary parameter"),
                )?;
                let constant = self.factor(row.constant_factor_id, "eager coupling constant")?;
                Ok(EagerCouplingRow {
                    real_parameter_id: row.real_parameter_id,
                    imag_parameter_id: row.imaginary_parameter_id,
                    constant_real: constant.re,
                    constant_imag: constant.im,
                })
            })
            .collect()
    }

    fn load_stages(
        &self,
        coupling_count: usize,
    ) -> RusticolResult<(Vec<EagerStagePlan>, BTreeSet<u32>)> {
        validate_stage_ranges(
            self.input.stages,
            self.input.invocations.len(),
            self.input.attachments.len(),
            self.input.finalizations.len(),
        )?;
        let mut stages = Vec::with_capacity(self.input.stages.len());
        let mut finalized_currents = BTreeSet::new();
        let mut stored_value_slots = BTreeSet::new();
        let mut previous_stage = None;
        for stage in self.input.stages {
            if previous_stage.is_some_and(|previous| stage.stage_index <= previous) {
                return Err(artifact("eager stage indices must be strictly increasing"));
            }
            previous_stage = Some(stage.stage_index);
            stages.push(self.load_stage(
                stage,
                coupling_count,
                &mut finalized_currents,
                &mut stored_value_slots,
            )?);
        }
        Ok((stages, stored_value_slots))
    }

    fn load_stage(
        &self,
        stage: &EagerPlanStageRow,
        coupling_count: usize,
        globally_finalized_currents: &mut BTreeSet<u32>,
        globally_stored_value_slots: &mut BTreeSet<u32>,
    ) -> RusticolResult<EagerStagePlan> {
        let invocation_range = checked_range(
            self.input.invocations,
            stage.invocation_start,
            stage.invocation_count,
            "eager stage invocations",
        )?;
        let attachment_range = checked_range(
            self.input.attachments,
            stage.attachment_start,
            stage.attachment_count,
            "eager stage attachments",
        )?;
        let finalization_range = checked_range(
            self.input.finalizations,
            stage.finalization_start,
            stage.finalization_count,
            "eager stage finalizations",
        )?;
        let stage_attachment_start = usize_count(stage.attachment_start, "attachment start")?;

        let mut invocations = Vec::with_capacity(invocation_range.len());
        let mut attachments = Vec::with_capacity(attachment_range.len());
        let mut attached_currents = BTreeSet::new();
        let mut attachment_cursor = stage_attachment_start;
        for (index, row) in invocation_range.iter().enumerate() {
            let kernel = require_kernel(
                &self.kernels,
                row.kernel_id,
                EagerKernelRole::Vertex,
                "invocation",
            )?;
            let left_values = self
                .values
                .get(row.left_value_slot_id, "eager invocation left value")?;
            let right_values = self
                .values
                .get(row.right_value_slot_id, "eager invocation right value")?;
            let left_momenta = self
                .momenta
                .get(row.left_momentum_slot_id, "eager invocation left momentum")?;
            let right_momenta = self.momenta.get(
                row.right_momentum_slot_id,
                "eager invocation right momentum",
            )?;
            required_index(
                row.coupling_slot_id,
                coupling_count,
                "eager invocation coupling",
            )?;
            let output_factor_source = validate_output_factor(
                row.output_factor_source,
                row.coupling_slot_id,
                "eager invocation",
            )?;
            validate_kernel_inputs(
                kernel,
                KernelInputBounds {
                    first_current: left_values.len,
                    second_current: right_values.len,
                    first_momentum: left_momenta.len,
                    second_momentum: right_momenta.len,
                    has_coupling: true,
                    parameter_count: self.parameter_count,
                },
            )?;
            let global_start = usize_count(row.attachment_start, "invocation attachment start")?;
            let count = usize_count(row.attachment_count, "invocation attachment count")?;
            if count == 0 || global_start != attachment_cursor {
                return Err(artifact(format!(
                    "eager invocation {index} has an empty or noncontiguous attachment range"
                )));
            }
            let global_stop = global_start
                .checked_add(count)
                .ok_or_else(|| artifact("eager invocation attachment range overflows usize"))?;
            let stage_stop = stage_attachment_start
                .checked_add(attachment_range.len())
                .ok_or_else(|| artifact("eager stage attachment range overflows usize"))?;
            if global_stop > stage_stop {
                return Err(artifact(format!(
                    "eager invocation {index} attachment range exceeds its stage"
                )));
            }
            let local_start = global_start - stage_attachment_start;
            let local_stop = global_stop - stage_attachment_start;
            self.validate_domain_id(row.selector_domain_id, "eager invocation selector")?;
            for (offset, attachment) in attachment_range[local_start..local_stop].iter().enumerate()
            {
                let current = self.currents.get(
                    attachment.result_current_id,
                    "eager attachment result current",
                )?;
                if usize::try_from(kernel.output_component_count).ok() != Some(current.len) {
                    return Err(artifact(format!(
                        "eager invocation kernel {} output width does not match current {}",
                        kernel.kernel_id, attachment.result_current_id
                    )));
                }
                let factor = self.attachment_factor(attachment)?;
                self.validate_domain_id(
                    attachment.selector_domain_id,
                    "eager attachment selector",
                )?;
                attached_currents.insert(attachment.result_current_id);
                attachments.push(ScheduledAttachment {
                    row: EagerAttachmentRow {
                        result_current_id: attachment.result_current_id,
                        factor_real: factor.re,
                        factor_imag: factor.im,
                    },
                    current,
                    selector_domain_id: Some(attachment.selector_domain_id),
                    initializes_current: false,
                });
                let expected_position = local_start + offset;
                if attachments.len() != expected_position + 1 {
                    return Err(artifact("eager attachment ranges overlap"));
                }
            }
            invocations.push(ScheduledInvocation {
                row: EagerInvocationRow {
                    kernel_id: row.kernel_id,
                    left_value_slot_id: row.left_value_slot_id,
                    right_value_slot_id: row.right_value_slot_id,
                    left_momentum_slot_id: row.left_momentum_slot_id,
                    right_momentum_slot_id: row.right_momentum_slot_id,
                    coupling_slot_id: row.coupling_slot_id,
                    output_factor_source,
                    attachment_start: local_start as u64,
                    attachment_count: count as u64,
                },
                left_values,
                right_values,
                left_momenta,
                right_momenta,
                attachment_range: local_start..local_stop,
                selector_domain_id: Some(row.selector_domain_id),
            });
            attachment_cursor = global_stop;
        }
        if attachments.len() != attachment_range.len() {
            return Err(artifact(
                "eager invocation ranges do not cover the stage attachment table",
            ));
        }

        let mut finalizations = Vec::new();
        let mut finalization_copies = Vec::new();
        let mut stage_current_ranges = BTreeMap::new();
        let mut current_component_count = 0usize;
        let mut stage_finalized_currents = BTreeSet::new();
        for (index, row) in finalization_range.iter().enumerate() {
            if !stage_finalized_currents.insert(row.current_id)
                || !globally_finalized_currents.insert(row.current_id)
            {
                return Err(artifact(format!(
                    "eager current {} is finalized more than once",
                    row.current_id
                )));
            }
            if row.unpropagated_value_slot_id != MISSING_U32
                && row.unpropagated_value_slot_id == row.propagated_value_slot_id
            {
                return Err(artifact(format!(
                    "eager finalization {index} aliases its outputs"
                )));
            }
            for value_slot_id in [row.unpropagated_value_slot_id, row.propagated_value_slot_id] {
                if value_slot_id != MISSING_U32
                    && !globally_stored_value_slots.insert(value_slot_id)
                {
                    return Err(artifact(format!(
                        "eager value slot {value_slot_id} is stored more than once"
                    )));
                }
            }
            let global_current = self
                .currents
                .get(row.current_id, "eager finalization current")?;
            let local_current = ComponentRange {
                start: current_component_count,
                len: global_current.len,
            };
            current_component_count = current_component_count
                .checked_add(global_current.len)
                .ok_or_else(|| artifact("eager stage current workspace overflows usize"))?;
            stage_current_ranges.insert(row.current_id, local_current);
            let unpropagated = self.optional_value(
                row.unpropagated_value_slot_id,
                EagerValueSlotKind::Unpropagated,
                "eager unpropagated finalization value",
            )?;
            let propagated = self.optional_value(
                row.propagated_value_slot_id,
                EagerValueSlotKind::Propagated,
                "eager propagated finalization value",
            )?;
            if unpropagated.is_none() && propagated.is_none() {
                return Err(artifact(format!(
                    "eager finalization {index} stores no current value"
                )));
            }
            for output in [unpropagated, propagated].into_iter().flatten() {
                if output.len != local_current.len {
                    return Err(artifact(format!(
                        "eager finalization {index} output width does not match current width"
                    )));
                }
            }
            let momentum = self
                .momenta
                .get(row.momentum_slot_id, "eager finalization momentum")?;
            self.validate_domain_id(
                row.unpropagated_selector_domain_id,
                "eager unpropagated finalization selector",
            )?;
            self.validate_domain_id(
                row.propagated_selector_domain_id,
                "eager propagated finalization selector",
            )?;
            if let Some(unpropagated) = unpropagated {
                finalization_copies.push(FinalizationCopy {
                    current: local_current,
                    unpropagated,
                    selector_domain_id: Some(row.unpropagated_selector_domain_id),
                });
            } else {
                self.require_empty_domain(
                    row.unpropagated_selector_domain_id,
                    "missing eager unpropagated output",
                )?;
            }
            if row.kernel_id == MISSING_U32 {
                if propagated.is_some() {
                    return Err(artifact(format!(
                        "eager finalization {index} has a propagated output but no kernel"
                    )));
                }
                self.require_empty_domain(
                    row.propagated_selector_domain_id,
                    "missing eager propagated output",
                )?;
            } else {
                let kernel = require_kernel(
                    &self.kernels,
                    row.kernel_id,
                    EagerKernelRole::Finalization,
                    "finalization",
                )?;
                if propagated.is_none() {
                    return Err(artifact(format!(
                        "eager finalization {index} applies a kernel without an output"
                    )));
                }
                validate_kernel_inputs(
                    kernel,
                    KernelInputBounds {
                        first_current: local_current.len,
                        second_current: 0,
                        first_momentum: momentum.len,
                        second_momentum: 0,
                        has_coupling: false,
                        parameter_count: self.parameter_count,
                    },
                )?;
                if usize::try_from(kernel.output_component_count).ok() != Some(local_current.len) {
                    return Err(artifact(format!(
                        "eager finalization kernel {} output width does not match current width",
                        kernel.kernel_id
                    )));
                }
                finalizations.push(ScheduledFinalization {
                    row: EagerFinalizationRow {
                        kernel_id: row.kernel_id,
                        current_id: row.current_id,
                        unpropagated_value_slot_id: row.unpropagated_value_slot_id,
                        propagated_value_slot_id: row.propagated_value_slot_id,
                        momentum_slot_id: row.momentum_slot_id,
                    },
                    current: local_current,
                    propagated,
                    momentum,
                    selector_domain_id: Some(row.propagated_selector_domain_id),
                });
            }
        }
        if attached_currents != stage_finalized_currents {
            return Err(artifact(format!(
                "eager stage {} attached and finalized current sets differ",
                stage.stage_index
            )));
        }
        for attachment in &mut attachments {
            attachment.current = *stage_current_ranges
                .get(&attachment.row.result_current_id)
                .ok_or_else(|| artifact("eager attachment has no stage-local current"))?;
        }
        invocations.sort_by_key(|item| item.row.kernel_id);
        finalizations.sort_by_key(|item| item.row.kernel_id);
        let zero_current_ranges =
            mark_initial_current_writes(&invocations, &mut attachments, &finalizations);
        Ok(EagerStagePlan {
            stage_index: stage.stage_index,
            current_component_count,
            invocations,
            attachments,
            finalization_copies,
            finalizations,
            zero_current_ranges,
        })
    }

    fn load_closures(
        &self,
        coupling_count: usize,
    ) -> RusticolResult<(Vec<ScheduledClosure>, Vec<ScheduledDirectClosure>)> {
        let mut closures = Vec::new();
        let mut direct_closures = Vec::new();
        for (index, row) in self.input.closures.iter().enumerate() {
            let left_values = self
                .values
                .get(row.left_value_slot_id, "eager closure left value")?;
            let right_values = self
                .values
                .get(row.right_value_slot_id, "eager closure right value")?;
            required_index(
                row.amplitude_index,
                self.amplitude_count,
                "eager closure amplitude",
            )?;
            self.validate_domain_id(row.selector_domain_id, "eager closure selector")?;
            let color = self.factor(row.color_factor_id, "eager closure color factor")?;
            self.factor(row.coupling_factor_id, "eager closure coupling factor")?;
            self.factor(
                row.normalization_factor_id,
                "eager closure normalization factor",
            )?;
            let direct = checked_range(
                self.input.direct_coefficients,
                row.direct_coefficient_start,
                row.direct_coefficient_count,
                "eager direct closure coefficients",
            )?;
            if row.kernel_id == MISSING_U32 {
                if row.coupling_slot_id != MISSING_U32
                    || row.output_factor_source != EAGER_OUTPUT_FACTOR_NONE as u8
                    || direct.is_empty()
                {
                    return Err(artifact(format!(
                        "eager direct closure {index} has inconsistent kernel metadata"
                    )));
                }
                if left_values.len != right_values.len || left_values.len != direct.len() {
                    return Err(artifact(format!(
                        "eager direct closure {index} component widths do not match"
                    )));
                }
                let mut coefficients = Vec::with_capacity(direct.len());
                for (component, coefficient) in direct.iter().enumerate() {
                    if usize::try_from(coefficient.component_index).ok() != Some(component) {
                        return Err(artifact(format!(
                            "eager direct closure {index} coefficients are not component ordered"
                        )));
                    }
                    coefficients.push(
                        self.factor(coefficient.factor_id, "eager direct closure coefficient")?,
                    );
                }
                direct_closures.push(ScheduledDirectClosure {
                    row: EagerClosureRow {
                        kernel_id: MISSING_U32,
                        left_value_slot_id: row.left_value_slot_id,
                        right_value_slot_id: row.right_value_slot_id,
                        amplitude_index: row.amplitude_index,
                        coupling_slot_id: MISSING_U32,
                        output_factor_source: EAGER_OUTPUT_FACTOR_NONE,
                        factor_real: color.re,
                        factor_imag: color.im,
                    },
                    left_values,
                    right_values,
                    coefficients,
                    selector_domain_id: Some(row.selector_domain_id),
                    initializes_amplitude: false,
                });
                continue;
            }
            if !direct.is_empty() {
                return Err(artifact(format!(
                    "eager kernel closure {index} also has direct coefficients"
                )));
            }
            let kernel = require_kernel(
                &self.kernels,
                row.kernel_id,
                EagerKernelRole::Closure,
                "closure",
            )?;
            required_index(
                row.coupling_slot_id,
                coupling_count,
                "eager closure coupling",
            )?;
            self.validate_coupling_factor(row)?;
            let output_factor_source = validate_output_factor(
                row.output_factor_source,
                row.coupling_slot_id,
                "eager closure",
            )?;
            validate_kernel_inputs(
                kernel,
                KernelInputBounds {
                    first_current: left_values.len,
                    second_current: right_values.len,
                    first_momentum: 0,
                    second_momentum: 0,
                    has_coupling: true,
                    parameter_count: self.parameter_count,
                },
            )?;
            if kernel.output_component_count != 1 {
                return Err(artifact(format!(
                    "eager closure kernel {} must produce one component",
                    kernel.kernel_id
                )));
            }
            // plan-v2 lowering multiplies color by prepared-kernel normalization.
            let factor = color
                * self.factor(
                    row.normalization_factor_id,
                    "eager closure normalization factor",
                )?;
            require_finite(factor, "eager closure combined factor")?;
            closures.push(ScheduledClosure {
                row: EagerClosureRow {
                    kernel_id: row.kernel_id,
                    left_value_slot_id: row.left_value_slot_id,
                    right_value_slot_id: row.right_value_slot_id,
                    amplitude_index: row.amplitude_index,
                    coupling_slot_id: row.coupling_slot_id,
                    output_factor_source,
                    factor_real: factor.re,
                    factor_imag: factor.im,
                },
                left_values,
                right_values,
                selector_domain_id: Some(row.selector_domain_id),
                initializes_amplitude: false,
            });
        }
        closures.sort_by_key(|item| item.row.kernel_id);
        Ok((closures, direct_closures))
    }

    fn load_reductions(
        &self,
    ) -> RusticolResult<(Vec<EagerReductionGroup>, Vec<EagerReductionEntry>)> {
        if self.input.reduction_groups.is_empty() {
            return Err(artifact("eager reduction requires nonempty groups"));
        }
        let mut group_index_by_id = BTreeMap::new();
        let mut covered_entries = vec![false; self.input.reduction_entries.len()];
        let mut covered_amplitudes = BTreeSet::new();
        let mut groups = Vec::with_capacity(self.input.reduction_groups.len());
        for (group_index, group) in self.input.reduction_groups.iter().enumerate() {
            if group_index_by_id
                .insert(group.coherent_group_id, group_index)
                .is_some()
            {
                return Err(artifact(format!(
                    "duplicate eager coherent group {}",
                    group.coherent_group_id
                )));
            }
            self.factor(
                group.helicity_weight_factor_id,
                "eager reduction helicity weight",
            )?;
            self.factor(
                group.all_sector_weight_factor_id,
                "eager reduction all-sector weight",
            )?;
            let amplitudes = self.reduction_owned_range(
                group.amplitude_entry_start,
                group.amplitude_entry_count,
                EagerPlanReductionEntryKind::AmplitudeMember,
                group.coherent_group_id,
                &mut covered_entries,
                "eager amplitude reduction entries",
            )?;
            if amplitudes.is_empty() {
                return Err(artifact(format!(
                    "eager reduction group {group_index} has no amplitudes"
                )));
            }
            let mut amplitude_indices = Vec::with_capacity(amplitudes.len());
            for entry in amplitudes {
                required_index(
                    entry.left_id,
                    self.amplitude_count,
                    "eager reduction amplitude",
                )?;
                if !covered_amplitudes.insert(entry.left_id) {
                    return Err(artifact(format!(
                        "eager amplitude {} belongs to multiple groups",
                        entry.left_id
                    )));
                }
                amplitude_indices.push(entry.left_id);
            }
            self.reduction_owned_range(
                group.selector_entry_start,
                group.selector_entry_count,
                EagerPlanReductionEntryKind::SelectorMember,
                group.coherent_group_id,
                &mut covered_entries,
                "eager selector reduction entries",
            )?;
            groups.push(EagerReductionGroup {
                coherent_group_id: group.coherent_group_id,
                amplitude_indices,
            });
        }
        if covered_amplitudes.len() != self.amplitude_count {
            return Err(artifact(format!(
                "eager reduction groups cover {} of {} amplitudes",
                covered_amplitudes.len(),
                self.amplitude_count
            )));
        }
        let contraction = checked_range(
            self.input.reduction_entries,
            self.input.color_contraction_entry_start,
            self.input.color_contraction_entry_count,
            "eager color contraction entries",
        )?;
        if contraction.is_empty() {
            return Err(artifact("eager reduction requires contraction entries"));
        }
        let contraction_start = usize_count(
            self.input.color_contraction_entry_start,
            "eager color contraction start",
        )?;
        let mut entries = Vec::with_capacity(contraction.len());
        for (offset, entry) in contraction.iter().enumerate() {
            if entry.kind != EagerPlanReductionEntryKind::ColorContraction {
                return Err(artifact(
                    "eager color contraction range contains another entry kind",
                ));
            }
            mark_covered(&mut covered_entries, contraction_start + offset)?;
            let left_group_index = *group_index_by_id.get(&entry.left_id).ok_or_else(|| {
                artifact(format!(
                    "eager contraction references unknown left group {}",
                    entry.left_id
                ))
            })?;
            let right_group_index = *group_index_by_id.get(&entry.right_id).ok_or_else(|| {
                artifact(format!(
                    "eager contraction references unknown right group {}",
                    entry.right_id
                ))
            })?;
            let mut coefficient = self.factor(entry.factor_id, "eager color contraction factor")?;
            if entry.auxiliary_factor_id != MISSING_U32 {
                coefficient = coefficient
                    * self.factor(
                        entry.auxiliary_factor_id,
                        "eager color contraction symmetry factor",
                    )?;
            }
            require_finite(coefficient, "eager color contraction coefficient")?;
            entries.push(EagerReductionEntry {
                left_group_index: usize_u32(left_group_index, "eager reduction group")?,
                right_group_index: usize_u32(right_group_index, "eager reduction group")?,
                coefficient,
            });
        }
        if covered_entries.iter().any(|covered| !covered) {
            return Err(artifact(
                "eager reduction entry catalog contains unowned rows",
            ));
        }
        Ok((groups, entries))
    }

    fn reduction_owned_range<'b>(
        &self,
        start: u64,
        count: u64,
        kind: EagerPlanReductionEntryKind,
        owner_id: u32,
        covered: &mut [bool],
        context: &str,
    ) -> RusticolResult<&'b [EagerPlanReductionEntryRow]>
    where
        'a: 'b,
    {
        let entries = checked_range(self.input.reduction_entries, start, count, context)?;
        let start = usize_count(start, context)?;
        for (offset, entry) in entries.iter().enumerate() {
            if entry.kind != kind || entry.owner_id != owner_id {
                return Err(artifact(format!("{context} have inconsistent tags/owners")));
            }
            mark_covered(covered, start + offset)?;
        }
        Ok(entries)
    }

    fn validate_selector_dependency_proof(
        &self,
        groups: &[EagerReductionGroup],
    ) -> RusticolResult<()> {
        let mut group_by_amplitude = vec![None; self.amplitude_count];
        for group in groups {
            for amplitude in &group.amplitude_indices {
                group_by_amplitude[*amplitude as usize] = Some(group.coherent_group_id);
            }
        }
        let mut value_domains = BTreeMap::<u32, BTreeSet<u32>>::new();
        for (index, closure) in self.input.closures.iter().enumerate() {
            let amplitude = required_index(
                closure.amplitude_index,
                self.amplitude_count,
                "eager closure amplitude",
            )?;
            let group = group_by_amplitude[amplitude]
                .ok_or_else(|| artifact("eager closure amplitude has no coherent group"))?;
            if closure.coherent_group_id != group {
                return Err(artifact(format!(
                    "eager closure {index} coherent group disagrees with reduction ownership"
                )));
            }
            self.validate_domain_members(
                closure.selector_domain_id,
                &BTreeSet::from([group]),
                &format!("eager closure {index}"),
            )?;
            value_domains
                .entry(closure.left_value_slot_id)
                .or_default()
                .insert(group);
            value_domains
                .entry(closure.right_value_slot_id)
                .or_default()
                .insert(group);
        }

        for stage in self.input.stages.iter().rev() {
            let invocations = checked_range(
                self.input.invocations,
                stage.invocation_start,
                stage.invocation_count,
                "eager selector stage invocations",
            )?;
            let attachments = checked_range(
                self.input.attachments,
                stage.attachment_start,
                stage.attachment_count,
                "eager selector stage attachments",
            )?;
            let finalizations = checked_range(
                self.input.finalizations,
                stage.finalization_start,
                stage.finalization_count,
                "eager selector stage finalizations",
            )?;
            let mut current_domains = BTreeMap::<u32, BTreeSet<u32>>::new();
            for (index, finalization) in finalizations.iter().enumerate() {
                let unpropagated = value_domains_for_optional(
                    &value_domains,
                    finalization.unpropagated_value_slot_id,
                );
                self.validate_domain_members(
                    finalization.unpropagated_selector_domain_id,
                    &unpropagated,
                    &format!(
                        "eager stage {} unpropagated finalization {index}",
                        stage.stage_index
                    ),
                )?;
                let propagated = value_domains_for_optional(
                    &value_domains,
                    finalization.propagated_value_slot_id,
                );
                self.validate_domain_members(
                    finalization.propagated_selector_domain_id,
                    &propagated,
                    &format!(
                        "eager stage {} propagated finalization {index}",
                        stage.stage_index
                    ),
                )?;
                let mut current = unpropagated;
                current.extend(propagated);
                current_domains.insert(finalization.current_id, current);
            }
            let mut attachment_domains = Vec::with_capacity(attachments.len());
            for (index, attachment) in attachments.iter().enumerate() {
                let expected = current_domains
                    .get(&attachment.result_current_id)
                    .ok_or_else(|| {
                        artifact(format!(
                            "eager stage {} attachment {index} has no finalized current",
                            stage.stage_index
                        ))
                    })?;
                self.validate_domain_members(
                    attachment.selector_domain_id,
                    expected,
                    &format!("eager stage {} attachment {index}", stage.stage_index),
                )?;
                attachment_domains.push(expected.clone());
            }
            let stage_attachment_start =
                usize_count(stage.attachment_start, "eager stage attachment start")?;
            for (index, invocation) in invocations.iter().enumerate() {
                let start = usize_count(
                    invocation.attachment_start,
                    "eager invocation attachment start",
                )?;
                let count = usize_count(
                    invocation.attachment_count,
                    "eager invocation attachment count",
                )?;
                if start < stage_attachment_start {
                    return Err(artifact("eager invocation attachment precedes its stage"));
                }
                let local_start = start - stage_attachment_start;
                let local_stop = local_start
                    .checked_add(count)
                    .ok_or_else(|| artifact("eager invocation attachment range overflows"))?;
                let mut expected = BTreeSet::new();
                for domain in attachment_domains
                    .get(local_start..local_stop)
                    .ok_or_else(|| {
                        artifact("eager invocation attachment range exceeds its stage")
                    })?
                {
                    expected.extend(domain);
                }
                self.validate_domain_members(
                    invocation.selector_domain_id,
                    &expected,
                    &format!("eager stage {} invocation {index}", stage.stage_index),
                )?;
                value_domains
                    .entry(invocation.left_value_slot_id)
                    .or_default()
                    .extend(&expected);
                value_domains
                    .entry(invocation.right_value_slot_id)
                    .or_default()
                    .extend(expected);
            }
        }
        Ok(())
    }

    fn attachment_factor(&self, row: &EagerPlanAttachmentRow) -> RusticolResult<EagerComplex64> {
        let color = self.factor(row.color_factor_id, "eager attachment color factor")?;
        let evaluation = self.factor(
            row.evaluation_factor_id,
            "eager attachment evaluation factor",
        )?;
        let normalization = self.factor(
            row.normalization_factor_id,
            "eager attachment normalization factor",
        )?;
        let representative = self.factor(
            row.representative_evaluation_factor_id,
            "eager attachment representative factor",
        )?;
        if representative == EagerComplex64::new(0.0, 0.0) {
            return Err(artifact(
                "eager attachment representative evaluation factor is zero",
            ));
        }
        // Match plan-v2 lowering: color * evaluation * (normalization / representative).
        let factor = color * evaluation * (normalization / representative);
        require_finite(factor, "eager attachment combined factor")?;
        Ok(factor)
    }

    fn validate_coupling_factor(&self, closure: &EagerPlanClosureRow) -> RusticolResult<()> {
        let coupling = self
            .input
            .couplings
            .get(closure.coupling_slot_id as usize)
            .ok_or_else(|| artifact("eager closure references unknown coupling"))?;
        if coupling.constant_factor_id != closure.coupling_factor_id {
            return Err(artifact(
                "eager closure coupling factor disagrees with its coupling row",
            ));
        }
        Ok(())
    }

    fn factor(&self, id: u32, context: &str) -> RusticolResult<EagerComplex64> {
        let index = required_index(id, self.factors.len(), context)?;
        Ok(self.factors[index])
    }

    fn optional_value(
        &self,
        id: u32,
        expected_kind: EagerValueSlotKind,
        context: &str,
    ) -> RusticolResult<Option<ComponentRange>> {
        if id == MISSING_U32 {
            return Ok(None);
        }
        let index = required_index(id, self.input.values.len(), context)?;
        if self.input.values[index].kind != expected_kind {
            return Err(artifact(format!("{context} has the wrong value-slot kind")));
        }
        self.values.get(id, context).map(Some)
    }

    fn validate_domain_id(&self, id: u32, context: &str) -> RusticolResult<usize> {
        required_index(id, self.selector_plan.memberships.len(), context)
    }

    fn require_empty_domain(&self, id: u32, context: &str) -> RusticolResult<()> {
        let index = self.validate_domain_id(id, context)?;
        if !self.selector_plan.memberships[index].is_empty() {
            return Err(artifact(format!(
                "{context} has a nonempty selector domain"
            )));
        }
        Ok(())
    }

    fn validate_domain_members(
        &self,
        id: u32,
        expected: &BTreeSet<u32>,
        context: &str,
    ) -> RusticolResult<()> {
        let index = self.validate_domain_id(id, context)?;
        if !self.selector_plan.memberships[index]
            .iter()
            .copied()
            .eq(expected.iter().copied())
        {
            return Err(artifact(format!(
                "{context} selector domain does not match its dependency closure"
            )));
        }
        Ok(())
    }
}

fn validate_current_layout(rows: &[EagerPlanCurrentRow]) -> RusticolResult<Vec<u32>> {
    validate_component_rows(
        rows.iter().enumerate().map(|(index, row)| {
            if row.flags & !1 != 0 {
                return Err(artifact(format!(
                    "eager current {index} has unsupported flags"
                )));
            }
            Ok((row.current_id, row.component_start, row.component_count))
        }),
        "current",
    )
}

fn validate_value_layout(
    rows: &[EagerPlanValueRow],
    currents: &[EagerPlanCurrentRow],
) -> RusticolResult<Vec<u32>> {
    let counts = validate_component_rows(
        rows.iter()
            .map(|row| Ok((row.value_slot_id, row.component_start, row.component_count))),
        "value",
    )?;
    for (index, row) in rows.iter().enumerate() {
        let current = required_index(row.current_id, currents.len(), "eager value current")?;
        if currents[current].component_count != row.component_count {
            return Err(artifact(format!(
                "eager value {index} width does not match current {}",
                row.current_id
            )));
        }
    }
    Ok(counts)
}

fn validate_momentum_layout(rows: &[EagerPlanMomentumRow]) -> RusticolResult<Vec<u32>> {
    validate_component_rows(
        rows.iter().map(|row| {
            Ok((
                row.momentum_slot_id,
                row.component_start,
                row.component_count,
            ))
        }),
        "momentum",
    )
}

fn validate_component_rows(
    rows: impl Iterator<Item = RusticolResult<(u32, u64, u32)>>,
    name: &str,
) -> RusticolResult<Vec<u32>> {
    let mut counts = Vec::new();
    let mut component_start = 0u64;
    for (index, row) in rows.enumerate() {
        let (id, start, count) = row?;
        if usize::try_from(id).ok() != Some(index) {
            return Err(artifact(format!("eager {name} IDs are not dense")));
        }
        if start != component_start || count == 0 {
            return Err(artifact(format!(
                "eager {name} {index} has a noncontiguous or empty component range"
            )));
        }
        component_start = component_start
            .checked_add(u64::from(count))
            .ok_or_else(|| artifact(format!("eager {name} component range overflows")))?;
        counts.push(count);
    }
    Ok(counts)
}

fn validate_amplitude_layout(rows: &[EagerPlanClosureRow]) -> RusticolResult<usize> {
    validate_dense_ids(
        rows.iter().map(|row| row.root_id),
        "eager plan-v3 closure root",
    )?;
    if rows.is_empty() {
        return Err(artifact("eager execution plan has no amplitude outputs"));
    }
    let amplitudes = rows
        .iter()
        .map(|row| row.amplitude_index)
        .collect::<BTreeSet<_>>();
    if amplitudes
        .iter()
        .copied()
        .ne(0..usize_u32(amplitudes.len(), "eager amplitude count")?)
    {
        return Err(artifact("eager closure amplitude indices are not dense"));
    }
    Ok(amplitudes.len())
}

fn load_factors(rows: &[EagerPlanExactFactorRow]) -> RusticolResult<Vec<EagerComplex64>> {
    validate_dense_ids(
        rows.iter().map(|row| row.factor_id),
        "eager plan-v3 exact factor",
    )?;
    rows.iter()
        .enumerate()
        .map(|(index, row)| {
            if row.exact_source > 1 {
                return Err(artifact(format!(
                    "eager exact factor {index} has an unsupported source"
                )));
            }
            let value = EagerComplex64::new(
                f64::from_bits(row.real_bits),
                f64::from_bits(row.imaginary_bits),
            );
            require_finite(value, &format!("eager exact factor {index}"))?;
            Ok(value)
        })
        .collect()
}

fn load_selector_plan(
    rows: &[EagerPlanSelectorDomainRow],
    memberships: &[u32],
    groups: &[EagerPlanReductionGroupRow],
) -> RusticolResult<SelectorDomainPlan> {
    if rows.is_empty() {
        return Err(artifact("eager selector-domain table is empty"));
    }
    let known_groups = groups
        .iter()
        .map(|group| group.coherent_group_id)
        .collect::<BTreeSet<_>>();
    if known_groups.len() != groups.len() {
        return Err(artifact("eager reduction groups repeat coherent group IDs"));
    }
    let mut result = Vec::with_capacity(rows.len());
    let mut cursor = 0usize;
    let mut unique = BTreeSet::new();
    for (domain_id, row) in rows.iter().enumerate() {
        let start = usize_count(row.member_start, "eager selector member start")?;
        let count = usize_count(row.member_count, "eager selector member count")?;
        if start != cursor {
            return Err(artifact(format!(
                "eager selector domain {domain_id} is not contiguous"
            )));
        }
        let stop = start
            .checked_add(count)
            .ok_or_else(|| artifact("eager selector membership range overflows"))?;
        let members = memberships
            .get(start..stop)
            .ok_or_else(|| artifact("eager selector domain exceeds its membership table"))?
            .to_vec();
        if members.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(artifact(format!(
                "eager selector domain {domain_id} is not sorted and unique"
            )));
        }
        if let Some(unknown) = members.iter().find(|member| !known_groups.contains(member)) {
            return Err(artifact(format!(
                "eager selector domain references unknown coherent group {unknown}"
            )));
        }
        if !unique.insert(members.clone()) {
            return Err(artifact(format!(
                "eager selector domain {domain_id} duplicates an earlier domain"
            )));
        }
        result.push(members);
        cursor = stop;
    }
    if cursor != memberships.len() || !result.iter().any(Vec::is_empty) {
        return Err(artifact(
            "eager selector domains do not exactly cover memberships and an empty domain",
        ));
    }
    Ok(SelectorDomainPlan {
        memberships: result,
        group_ids: known_groups.into_iter().collect(),
    })
}

fn validate_stage_ranges(
    stages: &[EagerPlanStageRow],
    invocation_count: usize,
    attachment_count: usize,
    finalization_count: usize,
) -> RusticolResult<()> {
    let mut invocations = 0u64;
    let mut attachments = 0u64;
    let mut finalizations = 0u64;
    for (index, stage) in stages.iter().enumerate() {
        if stage.invocation_start != invocations
            || stage.attachment_start != attachments
            || stage.finalization_start != finalizations
        {
            return Err(artifact(format!(
                "eager stage {index} ranges are not contiguous"
            )));
        }
        invocations = checked_add(
            stage.invocation_start,
            stage.invocation_count,
            "invocations",
        )?;
        attachments = checked_add(
            stage.attachment_start,
            stage.attachment_count,
            "attachments",
        )?;
        finalizations = checked_add(
            stage.finalization_start,
            stage.finalization_count,
            "finalizations",
        )?;
    }
    if usize::try_from(invocations).ok() != Some(invocation_count)
        || usize::try_from(attachments).ok() != Some(attachment_count)
        || usize::try_from(finalizations).ok() != Some(finalization_count)
    {
        return Err(artifact("eager stage ranges do not cover execution tables"));
    }
    Ok(())
}

fn validate_kernel_specs(
    specs: &[EagerKernelSpec],
    parameter_count: usize,
) -> RusticolResult<BTreeMap<u32, EagerKernelSpec>> {
    let mut kernels = BTreeMap::new();
    for (index, spec) in specs.iter().cloned().enumerate() {
        if spec.kernel_id == MISSING_U32
            || spec.inputs.is_empty()
            || spec.output_component_count == 0
        {
            return Err(artifact(format!("invalid eager kernel spec {index}")));
        }
        if spec.independent_block_size != 1 {
            if spec.independent_block_size != EAGER_INDEPENDENT_BLOCK_SIZE
                || spec.role != EagerKernelRole::Vertex
                || spec.inputs.iter().any(|input| {
                    !matches!(
                        input,
                        EagerKernelInput::FirstCurrentComponent(_)
                            | EagerKernelInput::SecondCurrentComponent(_)
                    )
                })
            {
                return Err(artifact(format!(
                    "eager kernel {} has an invalid independent block contract",
                    spec.kernel_id
                )));
            }
        }
        let mut inputs = BTreeSet::new();
        for input in &spec.inputs {
            if !inputs.insert(*input) {
                return Err(artifact(format!(
                    "eager kernel {} repeats an input descriptor",
                    spec.kernel_id
                )));
            }
            if let EagerKernelInput::ModelParameter(parameter) = input {
                required_index(
                    *parameter,
                    parameter_count,
                    &format!("eager kernel {} model parameter", spec.kernel_id),
                )?;
            }
        }
        let kernel_id = spec.kernel_id;
        if kernels.insert(kernel_id, spec).is_some() {
            return Err(artifact(format!("duplicate eager kernel ID {kernel_id}")));
        }
    }
    Ok(kernels)
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
        if !allowed
            || index.is_some_and(|index| {
                usize::try_from(index)
                    .ok()
                    .is_none_or(|index| index >= count)
            })
        {
            return Err(artifact(format!(
                "eager {:?} kernel {} cannot use input descriptor {input:?}",
                kernel.role, kernel.kernel_id
            )));
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
        artifact(format!(
            "eager {context} references unknown kernel {kernel_id}"
        ))
    })?;
    if kernel.role != role {
        return Err(artifact(format!(
            "eager {context} kernel {kernel_id} has role {:?}, expected {role:?}",
            kernel.role
        )));
    }
    Ok(kernel)
}

fn validate_output_factor(source: u8, coupling: u32, context: &str) -> RusticolResult<u32> {
    let source = u32::from(source);
    if source > 2 || (source != EAGER_OUTPUT_FACTOR_NONE && coupling == MISSING_U32) {
        return Err(artifact(format!(
            "{context} has an invalid output-factor source"
        )));
    }
    Ok(source)
}

fn value_domains_for_optional(
    domains: &BTreeMap<u32, BTreeSet<u32>>,
    value_slot_id: u32,
) -> BTreeSet<u32> {
    if value_slot_id == MISSING_U32 {
        BTreeSet::new()
    } else {
        domains.get(&value_slot_id).cloned().unwrap_or_default()
    }
}

fn validate_dense_ids(ids: impl Iterator<Item = u32>, context: &str) -> RusticolResult<()> {
    for (index, id) in ids.enumerate() {
        if usize::try_from(id).ok() != Some(index) {
            return Err(artifact(format!("{context} IDs are not dense")));
        }
    }
    Ok(())
}

fn validate_optional_index(id: u32, count: usize, context: &str) -> RusticolResult<()> {
    if id == MISSING_U32 {
        Ok(())
    } else {
        required_index(id, count, context).map(|_| ())
    }
}

fn required_index(id: u32, count: usize, context: &str) -> RusticolResult<usize> {
    if id == MISSING_U32 {
        return Err(artifact(format!("{context} uses the reserved missing ID")));
    }
    let index =
        usize::try_from(id).map_err(|_| artifact(format!("{context} index does not fit usize")))?;
    if index >= count {
        return Err(artifact(format!(
            "{context} index {index} is outside 0..{count}"
        )));
    }
    Ok(index)
}

fn checked_range<'a, T>(
    values: &'a [T],
    start: u64,
    count: u64,
    context: &str,
) -> RusticolResult<&'a [T]> {
    let start = usize_count(start, context)?;
    let count = usize_count(count, context)?;
    let stop = start
        .checked_add(count)
        .ok_or_else(|| artifact(format!("{context} range overflows usize")))?;
    values
        .get(start..stop)
        .ok_or_else(|| artifact(format!("{context} range exceeds its table")))
}

fn mark_covered(covered: &mut [bool], index: usize) -> RusticolResult<()> {
    let slot = covered
        .get_mut(index)
        .ok_or_else(|| artifact("eager reduction range exceeds its table"))?;
    if std::mem::replace(slot, true) {
        return Err(artifact("eager reduction entry belongs to multiple ranges"));
    }
    Ok(())
}

fn checked_add(start: u64, count: u64, context: &str) -> RusticolResult<u64> {
    start
        .checked_add(count)
        .ok_or_else(|| artifact(format!("eager {context} range overflows u64")))
}

fn usize_count(value: u64, context: &str) -> RusticolResult<usize> {
    usize::try_from(value).map_err(|_| artifact(format!("{context} does not fit usize")))
}

fn usize_u32(value: usize, context: &str) -> RusticolResult<u32> {
    u32::try_from(value).map_err(|_| artifact(format!("{context} exceeds u32")))
}

fn require_finite(value: EagerComplex64, context: &str) -> RusticolResult<()> {
    if !value.re.is_finite() || !value.im.is_finite() {
        return Err(artifact(format!("{context} is not finite")));
    }
    Ok(())
}

fn artifact(message: impl Into<String>) -> RusticolError {
    RusticolError::artifact(message)
}
