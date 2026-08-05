// SPDX-License-Identifier: 0BSD

use super::*;

#[derive(Clone, Debug)]
pub(super) struct PreparedFlavourFlow {
    operation: PreparedFlavourFlowOperation,
    result_particle: i32,
    static_result: Box<[i32]>,
}

#[derive(Clone, Copy, Debug)]
enum PreparedFlavourFlowOperation {
    Constant,
    AppendLeft,
    AppendRight,
    ConcatLeftRight,
}

impl PreparedFlavourFlow {
    fn new(quantum: QuantumFlowRow, catalog: &TemplateCatalog<'_>) -> RusticolResult<Self> {
        let operation = match catalog.string(
            quantum.flavour_flow_operation_string_id,
            "quantum-flow flavour operation",
        )? {
            "constant-result" => PreparedFlavourFlowOperation::Constant,
            "append-left-result" => PreparedFlavourFlowOperation::AppendLeft,
            "append-right-result" => PreparedFlavourFlowOperation::AppendRight,
            "concat-left-right-result" => PreparedFlavourFlowOperation::ConcatLeftRight,
            value => {
                return Err(invalid(format!(
                    "unsupported quantum-flow flavour operation {value:?}"
                )));
            }
        };
        let static_result = catalog
            .flavour_flow(quantum.result_flavour_flow_id, "quantum result flavour")?
            .to_vec();
        let result_particle = *static_result
            .last()
            .ok_or_else(|| invalid("quantum result flavour ancestry is empty"))?;
        Ok(Self {
            operation,
            result_particle,
            static_result: static_result.into_boxed_slice(),
        })
    }

    pub(super) fn apply(&self, parents: [&CurrentCoreKey; 2]) -> Vec<i32> {
        let append = |parent: &CurrentCoreKey, result_particle| {
            let mut result = parent.flavour_flow().to_vec();
            if result.last().copied() != Some(result_particle) {
                result.push(result_particle);
            }
            result
        };
        match self.operation {
            PreparedFlavourFlowOperation::Constant => self.static_result.to_vec(),
            PreparedFlavourFlowOperation::AppendLeft => append(parents[0], self.result_particle),
            PreparedFlavourFlowOperation::AppendRight => append(parents[1], self.result_particle),
            PreparedFlavourFlowOperation::ConcatLeftRight => {
                let mut result = parents[0].flavour_flow().to_vec();
                result.extend_from_slice(parents[1].flavour_flow());
                result.push(self.result_particle);
                result
            }
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct PreparedWitness {
    pub(super) row: LCColorTransitionWitnessRow,
    pub(super) witness: crate::recurrence::LCColorTransitionWitness,
}

#[derive(Clone, Debug)]
pub(super) struct PreparedTransition {
    pub(super) row: TransitionRow,
    pub(super) input_states: [u32; 2],
    pub(super) quantum: QuantumFlowRow,
    pub(super) input_spins: [i32; 2],
    pub(super) local_orders: Box<[u32]>,
    canonical_input_order: [u32; 2],
    input_exchange_factor: Option<ExactComplexRational>,
    pub(super) base_factor: ExactComplexRational,
    pub(super) flavour: PreparedFlavourFlow,
    pub(super) quantum_semantic_digest: SemanticDigest,
    pub(super) transition_semantic_digest: SemanticDigest,
    pub(super) evaluator_binding_digest: SemanticDigest,
    pub(super) witnesses: Box<[PreparedWitness]>,
}

impl PreparedTransition {
    fn new(
        row: TransitionRow,
        templates: &ValidatedRecurrenceTemplateInput,
        catalog: &TemplateCatalog<'_>,
    ) -> RusticolResult<Self> {
        let input = templates.input();
        let input_states: [u32; 2] = catalog
            .u32_sequence(row.input_state_sequence_id, "transition input states")?
            .try_into()
            .map_err(|_| invalid("on-the-fly transition must be binary"))?;
        let quantum = *input
            .quantum_flows
            .get(row.quantum_flow_template_id as usize)
            .ok_or_else(|| invalid("transition quantum flow is absent"))?;
        let input_spins: [i32; 2] = catalog
            .i32_sequence(quantum.input_spin_sequence_id, "quantum input spins")?
            .try_into()
            .map_err(|_| invalid("on-the-fly quantum flow must be binary"))?;
        let quantum_states =
            catalog.u32_sequence(quantum.input_state_sequence_id, "quantum input states")?;
        if quantum_states != input_states {
            return Err(invalid("transition and quantum-flow input states disagree"));
        }
        let canonical_input_order = match catalog.u32_sequence(
            row.canonical_input_order_sequence_id,
            "transition canonical input order",
        )? {
            [0, 1] => [0, 1],
            [1, 0] => [1, 0],
            _ => return Err(invalid("transition canonical input order is not binary")),
        };
        let input_exchange_factor = (row.input_exchange_factor_id != MISSING_U32)
            .then(|| catalog.factor(row.input_exchange_factor_id, "transition input exchange"))
            .transpose()?;
        let contraction = input
            .color_contractions
            .get(row.color_contraction_template_id as usize)
            .ok_or_else(|| invalid("transition color contraction is absent"))?;
        let quantum_coupling =
            catalog.factor(quantum.exact_coupling_factor_id, "quantum coupling")?;
        let binding_coupling = catalog.factor(
            row.binding_coupling_factor_id,
            "transition binding coupling",
        )?;
        if quantum_coupling != binding_coupling {
            return Err(invalid("transition binding and quantum coupling disagree"));
        }
        let base_factor = multiply_factors(&[
            catalog.factor(row.exact_factor_id, "transition exact")?,
            catalog.factor(
                contraction.exact_coefficient_factor_id,
                "transition color contraction",
            )?,
            output_factor_from_binding(
                binding_coupling,
                row.output_factor_source,
                "on-the-fly transition",
            )?,
        ])?;
        let binding = input
            .evaluator_bindings
            .get(row.evaluator_binding_id as usize)
            .ok_or_else(|| invalid("transition evaluator binding is absent"))?;
        if binding.id != row.evaluator_binding_id
            || EvaluatorContractKind::try_from(binding.contract_kind)?
                != EvaluatorContractKind::Vertex
        {
            return Err(invalid("transition evaluator binding has the wrong role"));
        }
        let witnesses = catalog
            .witness_rows(row.color_contraction_template_id)?
            .iter()
            .copied()
            .map(|witness_row| {
                Ok(PreparedWitness {
                    row: witness_row,
                    witness: catalog.witness(witness_row)?,
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        Ok(Self {
            row,
            input_states,
            quantum,
            input_spins,
            local_orders: catalog
                .coupling_orders(row.coupling_order_set_id)?
                .into_boxed_slice(),
            canonical_input_order,
            input_exchange_factor,
            base_factor,
            flavour: PreparedFlavourFlow::new(quantum, catalog)?,
            quantum_semantic_digest: catalog
                .digest(quantum.semantic_digest_id, "quantum semantic")?,
            transition_semantic_digest: catalog
                .digest(row.semantic_digest_id, "transition semantic")?,
            evaluator_binding_digest: catalog
                .digest(binding.semantic_digest_id, "transition evaluator binding")?,
            witnesses: witnesses.into_boxed_slice(),
        })
    }

    pub(super) fn parent_ids(
        &self,
        left_state: u32,
        right_state: u32,
        left_id: u32,
        right_id: u32,
    ) -> Option<[u32; 2]> {
        if self.input_states == [left_state, right_state] {
            Some([left_id, right_id])
        } else if left_state != right_state && self.input_states == [right_state, left_state] {
            Some([right_id, left_id])
        } else {
            None
        }
    }

    pub(super) fn evaluator_parents(&self, concrete: [u32; 2]) -> ([u32; 2], ExactComplexRational) {
        let mut ordered = match self.canonical_input_order {
            [0, 1] => concrete,
            [1, 0] => [concrete[1], concrete[0]],
            _ => unreachable!("validated transition input order"),
        };
        let mut factor = ExactComplexRational::ONE;
        if let Some(exchange) = self.input_exchange_factor
            && ordered[1] < ordered[0]
        {
            ordered.swap(0, 1);
            factor = exchange;
        }
        (ordered, factor)
    }
}

#[derive(Clone, Debug)]
pub(super) struct PreparedClosureQuantum {
    pub(super) row: Option<QuantumFlowRow>,
    pub(super) input_states: Option<[u32; 2]>,
    pub(super) input_spins: Option<[i32; 2]>,
    pub(super) output_factor: ExactComplexRational,
}

#[derive(Clone, Debug)]
pub(super) struct PreparedClosure {
    pub(super) row: ClosureRow,
    pub(super) input_states: [u32; 2],
    canonical_input_order: [u32; 2],
    input_exchange_factor: Option<ExactComplexRational>,
    pub(super) base_factor: ExactComplexRational,
    pub(super) closure_semantic_digest: SemanticDigest,
    pub(super) evaluator_binding_digest: SemanticDigest,
    pub(super) quantum_flows: Box<[PreparedClosureQuantum]>,
    pub(super) witnesses: Box<[PreparedWitness]>,
}

impl PreparedClosure {
    fn new(
        row: ClosureRow,
        templates: &ValidatedRecurrenceTemplateInput,
        catalog: &TemplateCatalog<'_>,
    ) -> RusticolResult<Self> {
        let input = templates.input();
        let input_states: [u32; 2] = catalog
            .u32_sequence(row.input_state_sequence_id, "closure input states")?
            .try_into()
            .map_err(|_| invalid("on-the-fly closure must be binary"))?;
        let canonical_input_order = match catalog.u32_sequence(
            row.canonical_input_order_sequence_id,
            "closure canonical input order",
        )? {
            [0, 1] => [0, 1],
            [1, 0] => [1, 0],
            _ => return Err(invalid("closure canonical input order is not binary")),
        };
        let input_exchange_factor = (row.input_exchange_factor_id != MISSING_U32)
            .then(|| catalog.factor(row.input_exchange_factor_id, "closure input exchange"))
            .transpose()?;
        let contraction = input
            .color_contractions
            .get(row.color_contraction_template_id as usize)
            .ok_or_else(|| invalid("closure color contraction is absent"))?;
        let binding_coupling =
            catalog.factor(row.binding_coupling_factor_id, "closure binding coupling")?;
        let base_factor = multiply_factors(&[
            catalog.factor(row.exact_factor_id, "closure exact")?,
            catalog.factor(
                contraction.exact_coefficient_factor_id,
                "closure color contraction",
            )?,
        ])?;
        let binding = input
            .evaluator_bindings
            .get(row.evaluator_binding_id as usize)
            .ok_or_else(|| invalid("closure evaluator binding is absent"))?;
        if binding.id != row.evaluator_binding_id
            || EvaluatorContractKind::try_from(binding.contract_kind)?
                != EvaluatorContractKind::Closure
        {
            return Err(invalid("closure evaluator binding has the wrong role"));
        }
        let eligible = catalog.u32_sequence(
            row.eligible_quantum_flow_sequence_id,
            "closure eligible quantum flows",
        )?;
        let quantum_flows = if eligible.is_empty() {
            vec![PreparedClosureQuantum {
                row: None,
                input_states: None,
                input_spins: None,
                output_factor: output_factor_from_binding(
                    binding_coupling,
                    row.output_factor_source,
                    "on-the-fly closure",
                )?,
            }]
        } else {
            eligible
                .iter()
                .copied()
                .map(|id| {
                    let quantum = *input
                        .quantum_flows
                        .get(id as usize)
                        .ok_or_else(|| invalid("closure quantum flow is absent"))?;
                    let states: [u32; 2] = catalog
                        .u32_sequence(quantum.input_state_sequence_id, "closure quantum states")?
                        .try_into()
                        .map_err(|_| invalid("closure quantum flow must be binary"))?;
                    let spins: [i32; 2] = catalog
                        .i32_sequence(quantum.input_spin_sequence_id, "closure quantum spins")?
                        .try_into()
                        .map_err(|_| invalid("closure quantum flow must be binary"))?;
                    let quantum_coupling = catalog
                        .factor(quantum.exact_coupling_factor_id, "closure quantum coupling")?;
                    if quantum_coupling != binding_coupling {
                        return Err(invalid(
                            "closure binding and quantum-flow coupling disagree",
                        ));
                    }
                    Ok(PreparedClosureQuantum {
                        row: Some(quantum),
                        input_states: Some(states),
                        input_spins: Some(spins),
                        output_factor: output_factor_from_binding(
                            binding_coupling,
                            row.output_factor_source,
                            "on-the-fly closure quantum flow",
                        )?,
                    })
                })
                .collect::<RusticolResult<Vec<_>>>()?
        };
        let witnesses = catalog
            .witness_rows(row.color_contraction_template_id)?
            .iter()
            .copied()
            .map(|witness_row| {
                Ok(PreparedWitness {
                    row: witness_row,
                    witness: catalog.witness(witness_row)?,
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        Ok(Self {
            row,
            input_states,
            canonical_input_order,
            input_exchange_factor,
            base_factor,
            closure_semantic_digest: catalog.digest(row.semantic_digest_id, "closure semantic")?,
            evaluator_binding_digest: catalog
                .digest(binding.semantic_digest_id, "closure evaluator binding")?,
            quantum_flows: quantum_flows.into_boxed_slice(),
            witnesses: witnesses.into_boxed_slice(),
        })
    }

    pub(super) fn parent_ids(
        &self,
        anchor_state: u32,
        complement_state: u32,
        anchor_id: u32,
        complement_id: u32,
    ) -> Option<[u32; 2]> {
        if self.input_states == [complement_state, anchor_state] {
            Some([complement_id, anchor_id])
        } else if anchor_state != complement_state
            && self.input_states == [anchor_state, complement_state]
        {
            Some([anchor_id, complement_id])
        } else {
            None
        }
    }

    pub(super) fn evaluator_parents(&self, concrete: [u32; 2]) -> ([u32; 2], ExactComplexRational) {
        let mut ordered = match self.canonical_input_order {
            [0, 1] => concrete,
            [1, 0] => [concrete[1], concrete[0]],
            _ => unreachable!("validated closure input order"),
        };
        let mut factor = ExactComplexRational::ONE;
        if let Some(exchange) = self.input_exchange_factor
            && ordered[1] < ordered[0]
        {
            ordered.swap(0, 1);
            factor = exchange;
        }
        (ordered, factor)
    }
}

pub(super) fn canonical_state_pair(left: u32, right: u32) -> (u32, u32) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

pub(super) fn propagator_by_state(
    templates: &ValidatedRecurrenceTemplateInput,
) -> RusticolResult<BTreeMap<u32, Option<u32>>> {
    let mut result = BTreeMap::new();
    for row in &templates.input().propagators {
        let value = (row.applies_propagator != 0).then_some(row.id);
        if result.insert(row.state_template_id, value).is_some() {
            return Err(invalid(format!(
                "current state {} has multiple propagator contracts",
                row.state_template_id
            )));
        }
    }
    Ok(result)
}

pub(super) fn prepared_transitions(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
) -> RusticolResult<BTreeMap<(u32, u32), Vec<PreparedTransition>>> {
    let mut result = BTreeMap::<(u32, u32), Vec<PreparedTransition>>::new();
    for row in templates.input().transitions.iter().copied() {
        let prepared = PreparedTransition::new(row, templates, catalog)?;
        result
            .entry(canonical_state_pair(
                prepared.input_states[0],
                prepared.input_states[1],
            ))
            .or_default()
            .push(prepared);
    }
    Ok(result)
}

pub(super) fn prepared_closures(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
) -> RusticolResult<BTreeMap<(u32, u32), Vec<PreparedClosure>>> {
    let mut result = BTreeMap::<(u32, u32), Vec<PreparedClosure>>::new();
    for row in templates.input().closures.iter().copied() {
        let prepared = PreparedClosure::new(row, templates, catalog)?;
        result
            .entry(canonical_state_pair(
                prepared.input_states[0],
                prepared.input_states[1],
            ))
            .or_default()
            .push(prepared);
    }
    Ok(result)
}
