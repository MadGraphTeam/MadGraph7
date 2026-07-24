// SPDX-License-Identifier: 0BSD

use std::collections::BTreeMap;

use super::*;
use crate::recurrence::direct_lowering::{
    RuntimeSourceChoice, effective_runtime_helicity_variant, lower_union_resolved_helicities,
};
use crate::recurrence::direct_plan::DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION;
use crate::recurrence::layout::RuntimeSourceVariantBinding;
use crate::recurrence::template::{
    CatalogHeaderRow, ClosureRow, ColorContractionRow, CurrentOrientation, CurrentStateRow,
    DigestCatalogRow, EvaluatorBindingRow, EvaluatorCallableKind, EvaluatorContractKind,
    ExactFactorRow, IndexedRangeRow, LCColorTransitionWitnessRow, MISSING_U32,
    OwnedRecurrenceTemplateInput, ParticleStatistics, PropagatorRow, QuantumFlowRow,
    RECURRENCE_TEMPLATE_CANONICALIZATION_ABI, RECURRENCE_TEMPLATE_EXACT_SCALAR_ABI,
    RECURRENCE_TEMPLATE_INPUT_ABI, RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION,
    RuntimeHelicityContractRow, RuntimeHelicityEmbeddingRow, RuntimeHelicityProjectionRow,
    RuntimeHelicityVariantRow, SourceRow, TransitionRow, ValidatedRecurrenceTemplateInput,
};
use crate::recurrence::{
    CanonicalMomentumLinearForm, CheckedTableRange, ContributionKey, CurrentCoreKey,
    CurrentHelicityIdentity, CurrentSourceBinding, DynamicLCColorState, DynamicLCColorStateId,
    ExactRational, LCColorComponentOperation, LCColorComponentRole, LCColorSourceSeedOperation,
    LCColorWitnessTermId, MomentumTerm, RECURRENCE_TEMPLATE_ABI, RecurrenceAmplitudeDestination,
    RecurrenceClosureTerm, RecurrenceContribution, RecurrenceCurrent, RecurrenceFinalization,
    RecurrenceReplayTarget, RecurrenceResolvedHelicity, RecurrenceStrategy, SourceStateAssignment,
};

const CATALOG_DIGEST_SEED: u8 = 3;
const QUANTUM_FLOW_DIGEST_SEED: u8 = 6;

fn digest(seed: u8) -> SemanticDigest {
    SemanticDigest::new([seed; 32]).unwrap()
}

fn rational(numerator: i128) -> ExactComplexRational {
    ExactComplexRational::new(
        ExactRational::new(numerator, 1).unwrap(),
        ExactRational::ZERO,
    )
}

fn encoded_strings(values: &[&str]) -> (Vec<CheckedTableRange>, Vec<u8>) {
    let mut values = values.to_vec();
    values.sort_unstable();
    values.dedup();
    let mut ranges = Vec::new();
    let mut bytes = Vec::new();
    for value in values {
        ranges.push(CheckedTableRange::new(
            bytes.len() as u64,
            value.len() as u64,
        ));
        bytes.extend_from_slice(value.as_bytes());
    }
    (ranges, bytes)
}

fn string_id(strings: &[&str], value: &str) -> u32 {
    let mut strings = strings.to_vec();
    strings.sort_unstable();
    strings.dedup();
    strings
        .iter()
        .position(|candidate| *candidate == value)
        .unwrap() as u32
}

fn indexed_u32_sequences(
    sequences: &[Vec<u32>],
) -> (Vec<IndexedRangeRow>, Vec<u32>, BTreeMap<Vec<u32>, u32>) {
    let mut sequences = sequences.to_vec();
    sequences.sort();
    sequences.dedup();
    let mut ranges = Vec::new();
    let mut values = Vec::new();
    let mut ids = BTreeMap::new();
    for (id, sequence) in sequences.into_iter().enumerate() {
        ranges.push(IndexedRangeRow {
            id: id as u32,
            range: CheckedTableRange::new(values.len() as u64, sequence.len() as u64),
        });
        values.extend_from_slice(&sequence);
        ids.insert(sequence, id as u32);
    }
    (ranges, values, ids)
}

fn indexed_i32_sequences(sequences: &[Vec<i32>]) -> (Vec<IndexedRangeRow>, Vec<i32>) {
    let mut sequences = sequences.to_vec();
    sequences.sort();
    sequences.dedup();
    let mut ranges = Vec::new();
    let mut values = Vec::new();
    for (id, sequence) in sequences.into_iter().enumerate() {
        ranges.push(IndexedRangeRow {
            id: id as u32,
            range: CheckedTableRange::new(values.len() as u64, sequence.len() as u64),
        });
        values.extend_from_slice(&sequence);
    }
    (ranges, values)
}

fn validated_template() -> ValidatedRecurrenceTemplateInput {
    let strings = [
        "0",
        "1",
        "a-source-resolver",
        "any",
        "b-vertex-resolver",
        "basis",
        "c-propagator-resolver",
        "closure-a",
        "closure-b",
        "a-color-transition",
        "b-color-close",
        "component",
        "constant-result",
        "crossing",
        "d-closure-a-resolver",
        "e-closure-b-resolver",
        "equivalence",
        "input-layout",
        "momentum",
        "particle-static-result",
        "projection",
        "propagator",
        "quantum-flow",
        "rule",
        "scalar",
        "singlet-forest",
        "source",
        "species",
        "state",
        "transition",
        RECURRENCE_TEMPLATE_ABI,
        RECURRENCE_TEMPLATE_CANONICALIZATION_ABI,
        RECURRENCE_TEMPLATE_EXACT_SCALAR_ABI,
        RECURRENCE_TEMPLATE_INPUT_ABI,
    ];
    let sid = |value| string_id(&strings, value);
    let empty = vec![];
    let state = vec![0];
    let state_pair = vec![0, 0];
    let order_pair = vec![0, 1];
    let momentum_pair = vec![sid("momentum"), sid("momentum")];
    let component = vec![sid("component")];
    let input_layout = vec![sid("input-layout")];
    let eligible_flow = vec![0];
    let source_semantic = vec![sid("source")];
    let transition_semantic = vec![sid("transition")];
    let propagator_semantic = vec![sid("propagator")];
    let closure_a_semantic = vec![sid("closure-a")];
    let closure_b_semantic = vec![sid("closure-b")];
    let expression_source = vec![16];
    let expression_vertex = vec![17];
    let expression_propagator = vec![19];
    let expression_closure_a = vec![10];
    let expression_closure_b = vec![11];
    let sequences = vec![
        empty.clone(),
        state.clone(),
        state_pair.clone(),
        order_pair.clone(),
        momentum_pair.clone(),
        component.clone(),
        input_layout.clone(),
        eligible_flow.clone(),
        source_semantic.clone(),
        transition_semantic.clone(),
        propagator_semantic.clone(),
        closure_a_semantic.clone(),
        closure_b_semantic.clone(),
        expression_source.clone(),
        expression_vertex.clone(),
        expression_propagator.clone(),
        expression_closure_a.clone(),
        expression_closure_b.clone(),
    ];
    let (u32_sequence_ranges, u32_sequence_values, sequence_ids) =
        indexed_u32_sequences(&sequences);
    let seq = |value: &Vec<u32>| *sequence_ids.get(value).unwrap();
    let (i32_sequence_ranges, i32_sequence_values) =
        indexed_i32_sequences(&[vec![0, 0], vec![1, 1]]);
    let i32_seq = |value: &[i32]| {
        i32_sequence_ranges
            .iter()
            .position(|range| {
                let rows = range
                    .range
                    .as_usize_range(i32_sequence_values.len(), "test i32 sequence")
                    .unwrap();
                &i32_sequence_values[rows] == value
            })
            .unwrap() as u32
    };
    let spin_pair_sequence_id = i32_seq(&[0, 0]);
    let singlet_pair_sequence_id = i32_seq(&[1, 1]);
    let (string_ranges, string_bytes) = encoded_strings(&strings);
    let digest_catalog = (1_u8..=25)
        .map(|seed| DigestCatalogRow {
            id: u32::from(seed - 1),
            value: [seed; 32],
        })
        .collect::<Vec<_>>();
    let digest_id = |seed: u8| u32::from(seed - 1);
    let evaluator = |id: u32,
                     resolver: &'static str,
                     prepared_kernel_id: u32,
                     contract_kind: EvaluatorContractKind,
                     input_states: &Vec<u32>,
                     output_state: u32,
                     exact_expression: &Vec<u32>,
                     semantic_template: &Vec<u32>,
                     signature_seed: u8,
                     semantic_seed: u8| EvaluatorBindingRow {
        id,
        resolver_key_string_id: sid(resolver),
        prepared_kernel_id,
        contract_kind: contract_kind as u8,
        callable_signature_digest_id: digest_id(signature_seed),
        input_state_sequence_id: seq(input_states),
        output_state_template_id: output_state,
        input_layout_sequence_id: seq(&input_layout),
        output_layout_sequence_id: seq(&component),
        exact_expression_digest_sequence_id: seq(exact_expression),
        semantic_template_sequence_id: seq(semantic_template),
        callable_kind: EvaluatorCallableKind::PreparedKernel as u8,
        runtime_template_string_id: MISSING_U32,
        semantic_digest_id: digest_id(semantic_seed),
    };

    OwnedRecurrenceTemplateInput {
        input_abi: RECURRENCE_TEMPLATE_INPUT_ABI.to_owned(),
        catalog_digest: digest(CATALOG_DIGEST_SEED),
        compiled_model_digest: digest(1),
        prepared_kernel_pack_digest: digest(2),
        catalog_header: vec![CatalogHeaderRow {
            schema_version: RECURRENCE_TEMPLATE_INPUT_SCHEMA_VERSION,
            abi_string_id: sid(RECURRENCE_TEMPLATE_ABI),
            canonicalization_abi_string_id: sid(RECURRENCE_TEMPLATE_CANONICALIZATION_ABI),
            exact_scalar_abi_string_id: sid(RECURRENCE_TEMPLATE_EXACT_SCALAR_ABI),
            compiled_model_digest_id: digest_id(1),
            prepared_kernel_pack_digest_id: digest_id(2),
            catalog_digest_id: digest_id(3),
            parameter_count: 0,
            current_state_count: 1,
            source_count: 1,
            quantum_flow_count: 1,
            transition_count: 1,
            propagator_count: 1,
            closure_count: 2,
            color_contraction_count: 2,
            symmetry_proof_count: 0,
            runtime_helicity_contract_count: 0,
            evaluator_binding_count: 5,
        }],
        coupling_order_ranges: vec![IndexedRangeRow {
            id: 0,
            range: CheckedTableRange::new(0, 0),
        }],
        coupling_order_terms: vec![],
        current_states: vec![CurrentStateRow {
            id: 0,
            template_string_id: sid("state"),
            particle_id: 1,
            anti_particle_id: 1,
            species_string_id: sid("species"),
            orientation: CurrentOrientation::SelfConjugate as u8,
            statistics: ParticleStatistics::Boson as u8,
            color_representation: 1,
            basis_string_id: sid("basis"),
            tensor_ordering_sequence_id: seq(&component),
            dimension: 1,
            chirality: 0,
            lc_color_shape_string_id: sid("singlet-forest"),
            auxiliary_kind_string_id: MISSING_U32,
            mass_parameter_id: MISSING_U32,
            width_parameter_id: MISSING_U32,
            semantic_digest_id: digest_id(4),
        }],
        digest_catalog,
        evaluator_bindings: vec![
            evaluator(
                0,
                "a-source-resolver",
                0,
                EvaluatorContractKind::Source,
                &empty,
                0,
                &expression_source,
                &source_semantic,
                20,
                13,
            ),
            evaluator(
                1,
                "b-vertex-resolver",
                1,
                EvaluatorContractKind::Vertex,
                &state_pair,
                0,
                &expression_vertex,
                &transition_semantic,
                21,
                14,
            ),
            evaluator(
                2,
                "c-propagator-resolver",
                2,
                EvaluatorContractKind::Propagator,
                &state,
                0,
                &expression_propagator,
                &propagator_semantic,
                22,
                15,
            ),
            evaluator(
                3,
                "d-closure-a-resolver",
                3,
                EvaluatorContractKind::Closure,
                &state_pair,
                MISSING_U32,
                &expression_closure_a,
                &closure_a_semantic,
                23,
                16,
            ),
            evaluator(
                4,
                "e-closure-b-resolver",
                4,
                EvaluatorContractKind::Closure,
                &state_pair,
                MISSING_U32,
                &expression_closure_b,
                &closure_b_semantic,
                24,
                17,
            ),
        ],
        exact_factors: vec![ExactFactorRow {
            id: 0,
            real_numerator_string_id: sid("1"),
            real_denominator_string_id: sid("1"),
            imag_numerator_string_id: sid("0"),
            imag_denominator_string_id: sid("1"),
        }],
        flavour_flow_ranges: vec![IndexedRangeRow {
            id: 0,
            range: CheckedTableRange::new(0, 1),
        }],
        flavour_flow_values: vec![1],
        i32_sequence_ranges,
        i32_sequence_values,
        parameters: vec![],
        propagators: vec![PropagatorRow {
            id: 0,
            template_string_id: sid("propagator"),
            state_template_id: 0,
            applies_propagator: 1,
            evaluator_binding_id: 2,
            numerator_expression_digest_id: digest_id(18),
            denominator_expression_digest_id: digest_id(19),
            mass_parameter_id: MISSING_U32,
            width_parameter_id: MISSING_U32,
            gauge_string_id: MISSING_U32,
            linearity_proof_template_id: MISSING_U32,
            semantic_digest_id: digest_id(10),
        }],
        quantum_flows: vec![QuantumFlowRow {
            id: 0,
            template_string_id: sid("quantum-flow"),
            input_state_sequence_id: seq(&state_pair),
            input_spin_sequence_id: spin_pair_sequence_id,
            input_flavour_sequence_id: seq(&state_pair),
            input_quantum_sequence_id: seq(&state_pair),
            flavour_flow_operation_string_id: sid("constant-result"),
            quantum_number_flow_operation_string_id: sid("particle-static-result"),
            coupling_order_set_id: 0,
            result_state_template_id: 0,
            result_spin_state: 0,
            result_flavour_flow_id: 0,
            result_quantum_number_flow_id: 0,
            exact_coupling_factor_id: 0,
            predicate_digest_id: digest_id(17),
            semantic_digest_id: digest_id(QUANTUM_FLOW_DIGEST_SEED),
        }],
        quantum_number_flow_ranges: vec![IndexedRangeRow {
            id: 0,
            range: CheckedTableRange::new(0, 0),
        }],
        quantum_number_flow_terms: vec![],
        runtime_helicity_contracts: vec![],
        runtime_helicity_variants: vec![],
        runtime_helicity_embeddings: vec![],
        runtime_helicity_projections: vec![],
        sources: vec![SourceRow {
            id: 0,
            template_string_id: sid("source"),
            state_template_id: 0,
            crossing_string_id: sid("crossing"),
            wavefunction_family_string_id: sid("scalar"),
            helicity: 0,
            spin_state: 50_000,
            flavour_flow_id: 0,
            quantum_number_flow_id: 0,
            lc_color_seed_operation: LCColorSourceSeedOperation::Empty as u8,
            lc_color_seed_shape_string_id: sid("singlet-forest"),
            lc_color_seed_component_kind: u8::MAX,
            lc_color_seed_component_role: LCColorComponentRole::None as u8,
            lc_color_seed_proof_digest_id: digest_id(17),
            lc_color_seed_provenance_sequence_id: seq(&empty),
            wavefunction_expression_digest_id: digest_id(17),
            evaluator_binding_id: 0,
            mass_parameter_id: MISSING_U32,
            width_parameter_id: MISSING_U32,
            semantic_digest_id: digest_id(5),
        }],
        string_ranges,
        string_bytes,
        symmetry_proofs: vec![],
        transitions: vec![TransitionRow {
            id: 0,
            template_string_id: sid("transition"),
            input_state_sequence_id: seq(&state_pair),
            result_state_template_id: 0,
            quantum_flow_template_id: 0,
            evaluator_binding_id: 1,
            canonical_input_order_sequence_id: seq(&order_pair),
            momentum_convention_sequence_id: seq(&momentum_pair),
            coupling_parameter_sequence_id: seq(&empty),
            coupling_order_set_id: 0,
            color_contraction_template_id: 0,
            binding_coupling_factor_id: 0,
            exact_factor_id: 0,
            output_factor_source: 0,
            equivalence_class_string_id: sid("equivalence"),
            input_exchange_factor_id: MISSING_U32,
            output_projection_string_id: sid("projection"),
            semantic_digest_id: digest_id(9),
        }],
        closures: vec![
            ClosureRow {
                id: 0,
                template_string_id: sid("closure-a"),
                input_state_sequence_id: seq(&state_pair),
                result_state_template_id: 0,
                evaluator_binding_id: 3,
                canonical_input_order_sequence_id: seq(&order_pair),
                coupling_parameter_sequence_id: seq(&empty),
                coupling_order_set_id: 0,
                eligible_quantum_flow_sequence_id: seq(&eligible_flow),
                color_contraction_template_id: 1,
                binding_coupling_factor_id: 0,
                exact_factor_id: 0,
                output_factor_source: 0,
                equivalence_class_string_id: sid("equivalence"),
                input_exchange_factor_id: MISSING_U32,
                projection_string_id: sid("projection"),
                component_coefficient_sequence_id: seq(&eligible_flow),
                chirality_relation_string_id: sid("any"),
                metric_signature_string_id: MISSING_U32,
                semantic_digest_id: digest_id(7),
            },
            ClosureRow {
                id: 1,
                template_string_id: sid("closure-b"),
                input_state_sequence_id: seq(&state_pair),
                result_state_template_id: 0,
                evaluator_binding_id: 4,
                canonical_input_order_sequence_id: seq(&order_pair),
                coupling_parameter_sequence_id: seq(&empty),
                coupling_order_set_id: 0,
                eligible_quantum_flow_sequence_id: seq(&eligible_flow),
                color_contraction_template_id: 1,
                binding_coupling_factor_id: 0,
                exact_factor_id: 0,
                output_factor_source: 0,
                equivalence_class_string_id: sid("equivalence"),
                input_exchange_factor_id: MISSING_U32,
                projection_string_id: sid("projection"),
                component_coefficient_sequence_id: seq(&eligible_flow),
                chirality_relation_string_id: sid("any"),
                metric_signature_string_id: MISSING_U32,
                semantic_digest_id: digest_id(8),
            },
        ],
        color_contractions: vec![
            ColorContractionRow {
                id: 0,
                template_string_id: sid("a-color-transition"),
                rule_kind_string_id: sid("rule"),
                input_representation_sequence_id: singlet_pair_sequence_id,
                has_output_representation: 1,
                output_representation: 1,
                ordered_open_string_arity: 0,
                exact_coefficient_factor_id: 0,
                witness_start: 0,
                witness_count: 1,
                nc_term_start: 0,
                nc_term_count: 0,
                expression_digest_id: digest_id(17),
                semantic_digest_id: digest_id(11),
            },
            ColorContractionRow {
                id: 1,
                template_string_id: sid("b-color-close"),
                rule_kind_string_id: sid("rule"),
                input_representation_sequence_id: singlet_pair_sequence_id,
                has_output_representation: 0,
                output_representation: 0,
                ordered_open_string_arity: 0,
                exact_coefficient_factor_id: 0,
                witness_start: 1,
                witness_count: 1,
                nc_term_start: 0,
                nc_term_count: 0,
                expression_digest_id: digest_id(17),
                semantic_digest_id: digest_id(12),
            },
        ],
        lc_color_transition_witnesses: vec![
            LCColorTransitionWitnessRow {
                color_contraction_id: 0,
                ordinal: 0,
                left_shape_string_id: sid("singlet-forest"),
                right_shape_string_id: sid("singlet-forest"),
                input_permutation: 0,
                reverse_parent_mask: 0,
                component_operation: LCColorComponentOperation::Empty as u8,
                result_component_kind: u8::MAX,
                result_component_role: LCColorComponentRole::None as u8,
                result_shape_string_id: sid("singlet-forest"),
                exact_factor_id: 0,
                proof_digest_id: digest_id(17),
                input_port_pairing_sequence_id: seq(&empty),
                result_port_binding_sequence_id: seq(&empty),
                provenance_sequence_id: seq(&empty),
            },
            LCColorTransitionWitnessRow {
                color_contraction_id: 1,
                ordinal: 0,
                left_shape_string_id: sid("singlet-forest"),
                right_shape_string_id: sid("singlet-forest"),
                input_permutation: 0,
                reverse_parent_mask: 0,
                component_operation: LCColorComponentOperation::Close as u8,
                result_component_kind: u8::MAX,
                result_component_role: LCColorComponentRole::None as u8,
                result_shape_string_id: MISSING_U32,
                exact_factor_id: 0,
                proof_digest_id: digest_id(17),
                input_port_pairing_sequence_id: seq(&empty),
                result_port_binding_sequence_id: seq(&empty),
                provenance_sequence_id: seq(&empty),
            },
        ],
        color_nc_terms: vec![],
        u32_sequence_ranges,
        u32_sequence_values,
    }
    .validate()
    .unwrap()
}

fn validated_union_template() -> ValidatedRecurrenceTemplateInput {
    let mut input = validated_template().into_input();
    input.catalog_header[0].runtime_helicity_contract_count = 1;
    let source_resolver_string_id = input.evaluator_bindings[0].resolver_key_string_id;
    let proof_string_id = input.transitions[0].equivalence_class_string_id;
    let proof_digest_id = input.evaluator_bindings[0].callable_signature_digest_id;
    input.runtime_helicity_contracts = vec![RuntimeHelicityContractRow {
        id: 0,
        template_string_id: source_resolver_string_id,
        full_state_template_id: 0,
        variant_range: CheckedTableRange::new(0, 1),
        proof_algorithm_string_id: proof_string_id,
        proof_digest_id,
        semantic_digest_id: proof_digest_id,
    }];
    input.runtime_helicity_variants = vec![RuntimeHelicityVariantRow {
        id: 0,
        contract_id: 0,
        source_template_id: 0,
        source_state_template_id: 0,
        embedding_range: CheckedTableRange::new(0, 1),
        projection_range: CheckedTableRange::new(0, 1),
        proof_digest_id,
    }];
    input.runtime_helicity_embeddings = vec![RuntimeHelicityEmbeddingRow {
        variant_id: 0,
        full_component: 0,
        source_component: 0,
        factor_id: 0,
    }];
    input.runtime_helicity_projections = vec![RuntimeHelicityProjectionRow {
        variant_id: 0,
        source_component: 0,
        full_component: 0,
    }];
    input.validate().unwrap()
}

#[test]
fn union_crossing_uses_the_effective_state_embedding() {
    let mut input = validated_union_template().into_input();
    input
        .runtime_helicity_variants
        .push(RuntimeHelicityVariantRow {
            id: 1,
            contract_id: 0,
            source_template_id: 0,
            source_state_template_id: 17,
            embedding_range: CheckedTableRange::new(1, 1),
            projection_range: CheckedTableRange::new(1, 1),
            proof_digest_id: 0,
        });
    input
        .runtime_helicity_embeddings
        .push(RuntimeHelicityEmbeddingRow {
            variant_id: 1,
            full_component: 0,
            source_component: MISSING_U32,
            factor_id: 0,
        });
    input
        .runtime_helicity_projections
        .push(RuntimeHelicityProjectionRow {
            variant_id: 1,
            source_component: 0,
            full_component: 1,
        });

    let effective = effective_runtime_helicity_variant(&input, 0..2, 17).unwrap();
    assert_eq!(effective.id, 1);
    assert_eq!(effective.source_state_template_id, 17);

    input
        .runtime_helicity_variants
        .push(RuntimeHelicityVariantRow {
            id: 2,
            contract_id: 0,
            source_template_id: 0,
            source_state_template_id: 17,
            embedding_range: CheckedTableRange::new(2, 1),
            projection_range: CheckedTableRange::new(2, 1),
            proof_digest_id: 1,
        });
    input
        .runtime_helicity_embeddings
        .push(RuntimeHelicityEmbeddingRow {
            variant_id: 2,
            full_component: 0,
            source_component: MISSING_U32,
            factor_id: 0,
        });
    input
        .runtime_helicity_projections
        .push(RuntimeHelicityProjectionRow {
            variant_id: 2,
            source_component: 0,
            full_component: 1,
        });
    assert_eq!(
        effective_runtime_helicity_variant(&input, 0..3, 17)
            .unwrap()
            .id,
        1
    );

    input.runtime_helicity_embeddings[2].source_component = 0;
    assert!(effective_runtime_helicity_variant(&input, 0..3, 17).is_err());
}

fn momentum(slots: &[u32]) -> CanonicalMomentumLinearForm {
    CanonicalMomentumLinearForm::new(
        slots
            .iter()
            .copied()
            .map(|source_slot| MomentumTerm {
                source_slot,
                coefficient: 1,
            })
            .collect(),
    )
    .unwrap()
}

fn source_key(slot: u32) -> CurrentCoreKey {
    CurrentCoreKey::new(
        digest(CATALOG_DIGEST_SEED),
        RecurrenceNodeKind::Source,
        0,
        DynamicLCColorStateId::from_interner(slot),
        vec![slot],
        momentum(&[slot]),
        CurrentHelicityIdentity::topology_replay(50_000, vec![SourceStateAssignment::new(slot, 0)])
            .unwrap(),
        vec![1],
        0,
        vec![],
        CurrentSourceBinding::FixedTemplate(0),
        None,
    )
    .unwrap()
}

fn propagated_key(id: u32, slots: &[u32]) -> CurrentCoreKey {
    CurrentCoreKey::new(
        digest(CATALOG_DIGEST_SEED),
        RecurrenceNodeKind::Current,
        0,
        DynamicLCColorStateId::from_interner(id),
        slots.to_vec(),
        momentum(slots),
        CurrentHelicityIdentity::topology_replay(
            0,
            slots
                .iter()
                .copied()
                .map(|slot| SourceStateAssignment::new(slot, 0))
                .collect(),
        )
        .unwrap(),
        vec![1],
        0,
        vec![],
        CurrentSourceBinding::None,
        Some(0),
    )
    .unwrap()
}

fn contribution_key(
    parent_ids: &[u32],
    parents: &[&CurrentCoreKey],
    output_projection_id: u32,
) -> ContributionKey {
    ContributionKey::new(
        0,
        parent_ids.to_vec(),
        vec![0; parents.len()],
        parents
            .iter()
            .map(|parent| parent.momentum().clone())
            .collect(),
        0,
        0,
        LCColorWitnessTermId::new(0, 0),
        digest(QUANTUM_FLOW_DIGEST_SEED),
        output_projection_id,
    )
    .unwrap()
}

fn count_fixture_program(
    templates: &ValidatedRecurrenceTemplateInput,
    external_count: usize,
    current_count: usize,
    contribution_count: usize,
    closure_count: usize,
    physical_sector_count: u32,
) -> RecurrenceProgram {
    let source_keys = (0..external_count)
        .map(|slot| source_key(slot as u32))
        .collect::<Vec<_>>();
    let mut propagated_specs = Vec::<(Vec<u32>, Vec<u32>)>::new();
    if external_count == 4 {
        propagated_specs.push((vec![0, 1], vec![0, 1]));
        propagated_specs.push((vec![2, 3], vec![2, 3]));
    } else {
        propagated_specs.push((vec![0, 1], vec![0, 1]));
        propagated_specs.push((vec![2, 3], vec![2, 3]));
        propagated_specs.push((vec![2, 3, 4], vec![(external_count + 1) as u32, 4]));
    }
    while propagated_specs.len() < current_count - external_count {
        let pair = if propagated_specs.len() % 2 == 0 {
            vec![0, 1]
        } else {
            vec![2, 3]
        };
        propagated_specs.push((pair.clone(), pair));
    }
    let propagated_keys = propagated_specs
        .iter()
        .enumerate()
        .map(|(index, (support, _))| propagated_key((external_count + index) as u32, support))
        .collect::<Vec<_>>();

    let output_projection_id = templates.input().transitions[0].output_projection_string_id;
    let non_source_count = current_count - external_count;
    let extra_contributions = contribution_count - non_source_count;
    let mut contributions = Vec::with_capacity(contribution_count);
    let mut currents = source_keys
        .iter()
        .enumerate()
        .map(|(id, key)| {
            RecurrenceCurrent::new(
                id as u32,
                key.clone(),
                Some(ExactComplexRational::ONE),
                CheckedTableRange::new(0, 0),
                None,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let all_keys = source_keys
        .iter()
        .chain(propagated_keys.iter())
        .collect::<Vec<_>>();
    let mut finalizations = Vec::with_capacity(non_source_count);
    for (index, (support, parent_ids)) in propagated_specs.iter().enumerate() {
        let current_id = (external_count + index) as u32;
        let start = contributions.len() as u64;
        let fan_in = 1 + usize::from(index < extra_contributions);
        for _ in 0..fan_in {
            let parent_keys = parent_ids
                .iter()
                .map(|id| all_keys[*id as usize])
                .collect::<Vec<_>>();
            contributions.push(
                RecurrenceContribution::new(
                    contributions.len() as u32,
                    current_id,
                    parent_ids.clone(),
                    contribution_key(parent_ids, &parent_keys, output_projection_id),
                    ExactComplexRational::ONE,
                )
                .unwrap(),
            );
        }
        finalizations.push(
            RecurrenceFinalization::new(
                index as u32,
                current_id,
                Some(0),
                ExactComplexRational::ONE,
            )
            .unwrap(),
        );
        currents.push(
            RecurrenceCurrent::new(
                current_id,
                propagated_keys[index].clone(),
                None,
                CheckedTableRange::new(start, fan_in as u64),
                Some(index as u32),
            )
            .unwrap(),
        );
        assert_eq!(support, propagated_keys[index].support_source_slots());
    }

    let closure_parents = if external_count == 4 {
        vec![external_count as u32, external_count as u32 + 1]
    } else {
        vec![external_count as u32, external_count as u32 + 2]
    };
    let closure_terms = (0..closure_count)
        .map(|id| {
            RecurrenceClosureTerm::new(
                id as u32,
                0,
                (id % 2) as u32,
                Some(0),
                closure_parents.clone(),
                if id % 3 == 0 {
                    rational(-1)
                } else {
                    ExactComplexRational::ONE
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let replay_targets = (0..physical_sector_count)
        .map(|target| {
            let mut permutation = (0..external_count as u32).collect::<Vec<_>>();
            if target != 0 {
                permutation.swap(external_count - 2, external_count - 1);
            }
            RecurrenceReplayTarget::new(
                target,
                0,
                target,
                permutation,
                if target == 0 {
                    ExactComplexRational::ONE
                } else {
                    rational(-1)
                },
            )
            .unwrap()
        })
        .collect();
    let resolved_helicities = vec![
        RecurrenceResolvedHelicity::new(
            0,
            (0..external_count as u32)
                .map(|slot| SourceStateAssignment::new(slot, 0))
                .collect(),
            vec![0; external_count],
        )
        .unwrap(),
    ];
    RecurrenceProgram::new(
        RecurrenceStrategy::TopologyReplay,
        physical_sector_count,
        1,
        (0..current_count)
            .map(|id| DynamicLCColorState::new(id as u32, None, vec![]).unwrap())
            .collect(),
        currents,
        contributions,
        finalizations,
        replay_targets,
        resolved_helicities,
        vec![
            RecurrenceAmplitudeDestination::new(
                0,
                0,
                Some(0),
                CheckedTableRange::new(0, closure_count as u64),
            )
            .unwrap(),
        ],
        closure_terms,
    )
    .unwrap()
}

fn union_key(key: &CurrentCoreKey, runtime_variant_id: u32) -> CurrentCoreKey {
    let source_binding = if key.node_kind() == RecurrenceNodeKind::Source {
        CurrentSourceBinding::runtime_dispatch_with_variants(
            0,
            vec![
                RuntimeSourceVariantBinding::new(
                    0,
                    0,
                    runtime_variant_id,
                    0,
                    0,
                    0,
                    key.spin_state_class(),
                    ExactComplexRational::ONE,
                )
                .unwrap(),
            ],
        )
        .unwrap()
    } else {
        CurrentSourceBinding::None
    };
    CurrentCoreKey::new(
        key.catalog_digest(),
        key.node_kind(),
        key.current_state_template_id(),
        key.dynamic_lc_color_state_id(),
        key.support_source_slots().to_vec(),
        key.momentum().clone(),
        CurrentHelicityIdentity::all_flow_union(key.spin_state_class()),
        key.flavour_flow().to_vec(),
        key.quantum_number_flow_id(),
        key.coupling_orders().to_vec(),
        source_binding,
        key.propagator_template_id(),
    )
    .unwrap()
}

fn zgg_union_program(
    templates: &ValidatedRecurrenceTemplateInput,
    runtime_variant_id: u32,
) -> RecurrenceProgram {
    let base = count_fixture_program(templates, 5, 69, 126, 24, 2);
    let currents = base
        .currents()
        .iter()
        .map(|current| {
            RecurrenceCurrent::new(
                current.id(),
                union_key(current.key(), runtime_variant_id),
                None,
                current.contribution_range(),
                current.finalization_id(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let closures = base
        .closure_terms()
        .iter()
        .enumerate()
        .map(|(index, closure)| {
            RecurrenceClosureTerm::new(
                closure.id(),
                u32::from(index >= 12),
                closure.closure_template_id(),
                closure.quantum_flow_template_id(),
                closure.parent_current_ids().to_vec(),
                closure.exact_factor(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    RecurrenceProgram::new(
        RecurrenceStrategy::AllFlowUnion,
        2,
        1,
        base.dynamic_color_states().to_vec(),
        currents,
        base.contributions().to_vec(),
        base.finalizations().to_vec(),
        vec![],
        vec![],
        vec![
            RecurrenceAmplitudeDestination::new(0, 0, None, CheckedTableRange::new(0, 12)).unwrap(),
            RecurrenceAmplitudeDestination::new(1, 1, None, CheckedTableRange::new(12, 12))
                .unwrap(),
        ],
        closures,
    )
    .unwrap()
}

fn runtime_options() -> DirectRecurrenceRuntimeOptions {
    DirectRecurrenceRuntimeOptions::new(128, 16).unwrap()
}

fn direct_catalog(catalog_digest: SemanticDigest) -> PreparedDirectExecutorCatalog {
    PreparedDirectExecutorCatalog::new(
        catalog_digest,
        vec![
            PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Source, 0, 0),
            PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Contribution, 1, 1),
            PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Finalization, 2, 2),
            PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Closure, 3, 3),
            PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Closure, 4, 4),
            PreparedDirectExecutorBinding::identity_finalizer(5),
        ],
    )
    .unwrap()
}

fn lower(
    program: &RecurrenceProgram,
    templates: &ValidatedRecurrenceTemplateInput,
    semantic_digest: SemanticDigest,
) -> crate::RusticolResult<DirectRecurrencePlanParts> {
    let catalog_digest = digest(40);
    lower_recurrence_direct_v2(
        program,
        templates,
        &direct_catalog(catalog_digest),
        semantic_digest,
        digest(2),
        catalog_digest,
        runtime_options(),
    )
}

#[test]
fn deterministic_lowering_uses_stable_prepared_executor_ids_and_i32_spin() {
    let templates = validated_template();
    let program = count_fixture_program(&templates, 4, 31, 34, 12, 1);
    let first = lower(&program, &templates, digest(30)).unwrap();
    let second = lower(&program, &templates, digest(30)).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.direct_executor_count, 6);
    assert_eq!(first.direct_template_catalog_digest, digest(40));
    assert_eq!(first.sources[0].spin_state_class, 50_000);
    assert_eq!(
        first
            .row_groups
            .iter()
            .find(|row_group| row_group.role == DirectExecutorRole::Source)
            .unwrap()
            .direct_executor_id,
        0
    );
    assert!(first.row_groups.iter().any(|row_group| {
        row_group.role == DirectExecutorRole::Contribution && row_group.direct_executor_id == 1
    }));
    assert!(first.row_groups.iter().any(|row_group| {
        row_group.role == DirectExecutorRole::Finalization && row_group.direct_executor_id == 2
    }));
}

#[test]
fn prepared_executor_ids_do_not_depend_on_process_encounter_order() {
    let templates = validated_template();
    let program = count_fixture_program(&templates, 4, 31, 34, 12, 1);
    let catalog_digest = digest(43);
    let catalog = PreparedDirectExecutorCatalog::new(
        catalog_digest,
        vec![
            PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Closure, 4, 0),
            PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Closure, 3, 1),
            PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Contribution, 1, 2),
            PreparedDirectExecutorBinding::identity_finalizer(3),
            PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Source, 0, 4),
            PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Finalization, 2, 5),
        ],
    )
    .unwrap();
    let parts = lower_recurrence_direct_v2(
        &program,
        &templates,
        &catalog,
        digest(44),
        digest(2),
        catalog_digest,
        runtime_options(),
    )
    .unwrap();

    assert!(parts.row_groups.iter().any(|group| {
        group.role == DirectExecutorRole::Source && group.direct_executor_id == 4
    }));
    assert!(parts.row_groups.iter().any(|group| {
        group.role == DirectExecutorRole::Contribution && group.direct_executor_id == 2
    }));
    assert!(parts.row_groups.iter().any(|group| {
        group.role == DirectExecutorRole::Finalization && group.direct_executor_id == 5
    }));
    assert!(parts.row_groups.iter().any(|group| {
        group.role == DirectExecutorRole::Closure && matches!(group.direct_executor_id, 0 | 1)
    }));
}

#[test]
fn zg_and_zgg_gate_counts_are_preserved_exactly() {
    let templates = validated_template();
    for (external, currents, contributions, closures, sectors) in
        [(4, 31, 34, 12, 1), (5, 69, 126, 24, 2)]
    {
        let program = count_fixture_program(
            &templates,
            external,
            currents,
            contributions,
            closures,
            sectors,
        );
        let parts = lower(&program, &templates, digest(31)).unwrap();
        assert_eq!(parts.currents.len(), currents);
        assert_eq!(parts.contributions.len(), contributions);
        assert_eq!(parts.finalizations.len(), currents - external);
        assert_eq!(parts.closures.len(), closures);
    }
}

#[test]
fn zgg_all_flow_union_lowers_runtime_sources_without_replay_expansion() {
    let templates = validated_union_template();
    let program = zgg_union_program(&templates, 0);
    let first = lower(&program, &templates, digest(46)).unwrap();
    let second = lower(&program, &templates, digest(46)).unwrap();

    assert_eq!(first, second);
    assert_eq!(first.strategy, RecurrenceStrategy::AllFlowUnion);
    assert_eq!(first.currents.len(), 69);
    assert_eq!(first.contributions.len(), 126);
    assert_eq!(first.closures.len(), 24);
    assert_eq!(first.amplitude_destinations.len(), 2);
    assert_eq!(
        first
            .amplitude_destinations
            .iter()
            .map(|destination| destination.target_sector_id)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    assert!(first.replay_targets.is_empty());
    assert!(first.source_permutations.is_empty());
    assert_eq!(first.sources.len(), 5);
    assert_eq!(first.source_dispatch_variants.len(), 5);
    assert_eq!(first.resolved_helicities.len(), 1);
    assert_eq!(first.resolved_source_selections.len(), 5);
    assert_eq!(first.source_state_assignments.len(), 5);
    assert_eq!(first.public_helicities, [0; 5]);
    assert!(first.row_groups.iter().any(|group| {
        group.role == DirectExecutorRole::Source
            && group.direct_executor_id == DIRECT_NONE_U32
            && group.row_count == 5
    }));
    for (source_slot, selection) in first.resolved_source_selections.iter().enumerate() {
        assert_eq!(selection.source_slot, source_slot as u32);
        let variant = first.source_dispatch_variants[selection.dispatch_variant_id as usize];
        assert_eq!(
            first.sources[variant.source_row_id as usize].source_slot,
            source_slot as u32
        );
        assert_eq!(
            first.exact_factors[variant.crossing_exact_factor_id as usize],
            ExactComplexRational::ONE
        );
        assert_eq!(variant.embedding_count, 1);
        assert_eq!(variant.projection_count, 1);
    }
}

#[test]
fn union_resolved_helicities_are_a_deterministic_source_cartesian_product() {
    let choices = vec![
        vec![
            RuntimeSourceChoice {
                source_state_index: 0,
                public_helicity: -1,
                dispatch_variant_id: 4,
            },
            RuntimeSourceChoice {
                source_state_index: 1,
                public_helicity: 1,
                dispatch_variant_id: 5,
            },
        ],
        vec![
            RuntimeSourceChoice {
                source_state_index: 0,
                public_helicity: 0,
                dispatch_variant_id: 8,
            },
            RuntimeSourceChoice {
                source_state_index: 1,
                public_helicity: 1,
                dispatch_variant_id: 9,
            },
        ],
    ];
    let (descriptors, states, selections, public) =
        lower_union_resolved_helicities(&choices, 4).unwrap();
    assert_eq!(descriptors.len(), 4);
    assert_eq!(
        public
            .chunks_exact(2)
            .map(|values| values.to_vec())
            .collect::<Vec<_>>(),
        [vec![-1, 0], vec![-1, 1], vec![1, 0], vec![1, 1]]
    );
    assert_eq!(
        states
            .chunks_exact(2)
            .map(|values| values.iter().map(|row| row.state_index).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        [vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]
    );
    assert_eq!(
        selections
            .chunks_exact(2)
            .map(|values| {
                values
                    .iter()
                    .map(|row| row.dispatch_variant_id)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        [vec![4, 8], vec![4, 9], vec![5, 8], vec![5, 9]]
    );
}

#[test]
fn union_runtime_variant_outside_its_prepared_contract_fails_closed() {
    let templates = validated_union_template();
    let program = zgg_union_program(&templates, 1);
    let error = lower(&program, &templates, digest(47))
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("runtime variant 1 is outside contract 0"),
        "{error}"
    );
}

#[test]
fn union_source_choice_count_must_match_retained_helicity_count() {
    let choices = vec![vec![RuntimeSourceChoice {
        source_state_index: 0,
        public_helicity: 0,
        dispatch_variant_id: 0,
    }]];
    let error = lower_union_resolved_helicities(&choices, 2)
        .unwrap_err()
        .to_string();
    assert!(error.contains("span 1 helicities, expected 2"), "{error}");
}

#[test]
fn closure_ranges_remain_destination_contiguous_with_extra_executor_groups() {
    let templates = validated_template();
    let base = count_fixture_program(&templates, 4, 31, 34, 12, 1);
    let mut closures = base.closure_terms().to_vec();
    for (id, closure) in closures.iter_mut().enumerate() {
        *closure = RecurrenceClosureTerm::new(
            id as u32,
            u32::from(id >= 6),
            (id % 2) as u32,
            Some(0),
            closure.parent_current_ids().to_vec(),
            closure.exact_factor(),
        )
        .unwrap();
    }
    let program = RecurrenceProgram::new(
        base.strategy(),
        base.physical_sector_count(),
        base.retained_helicity_count(),
        base.dynamic_color_states().to_vec(),
        base.currents().to_vec(),
        base.contributions().to_vec(),
        base.finalizations().to_vec(),
        base.replay_targets().to_vec(),
        base.resolved_helicities().to_vec(),
        vec![
            RecurrenceAmplitudeDestination::new(0, 0, Some(0), CheckedTableRange::new(0, 6))
                .unwrap(),
            RecurrenceAmplitudeDestination::new(1, 0, Some(0), CheckedTableRange::new(6, 6))
                .unwrap(),
        ],
        closures,
    )
    .unwrap();
    let parts = lower(&program, &templates, digest(32)).unwrap();
    assert_eq!(
        parts
            .amplitude_destinations
            .iter()
            .map(|row| (row.closure_row_start, row.closure_row_count))
            .collect::<Vec<_>>(),
        [(0, 6), (6, 6)]
    );
    assert!(
        parts.closures[..6]
            .iter()
            .all(|row| row.amplitude_destination_id == 0)
    );
    assert!(
        parts.closures[6..]
            .iter()
            .all(|row| row.amplitude_destination_id == 1)
    );
    assert_eq!(
        parts
            .row_groups
            .iter()
            .filter(|row_group| row_group.role == DirectExecutorRole::Closure)
            .count(),
        4
    );
}

#[test]
fn unsupported_parent_arity_fails_closed() {
    let templates = validated_template();
    let base = count_fixture_program(&templates, 4, 31, 34, 12, 1);
    let result_id = 4_u32;
    let result = &base.currents()[result_id as usize];
    let parent_ids = vec![0, 1, 2];
    let parent_keys = parent_ids
        .iter()
        .map(|id| base.currents()[*id as usize].key())
        .collect::<Vec<_>>();
    let mut contributions = base.contributions().to_vec();
    contributions[0] = RecurrenceContribution::new(
        0,
        result_id,
        parent_ids.clone(),
        contribution_key(
            &parent_ids,
            &parent_keys,
            templates.input().transitions[0].output_projection_string_id,
        ),
        ExactComplexRational::ONE,
    )
    .unwrap();
    let program = RecurrenceProgram::new(
        base.strategy(),
        base.physical_sector_count(),
        base.retained_helicity_count(),
        base.dynamic_color_states().to_vec(),
        base.currents().to_vec(),
        contributions,
        base.finalizations().to_vec(),
        base.replay_targets().to_vec(),
        base.resolved_helicities().to_vec(),
        base.amplitude_destinations().to_vec(),
        base.closure_terms().to_vec(),
    )
    .unwrap();
    let error = lower(&program, &templates, digest(33))
        .unwrap_err()
        .to_string();
    assert!(error.contains("unsupported parent arity 3"));
    assert_eq!(result.key().support_source_slots(), &[0, 1]);
}

#[test]
fn non_unit_source_factor_is_interned_and_referenced_by_the_source_row() {
    let templates = validated_template();
    let base = count_fixture_program(&templates, 4, 31, 34, 12, 1);
    let mut currents = base.currents().to_vec();
    currents[0] = RecurrenceCurrent::new(
        0,
        currents[0].key().clone(),
        Some(rational(-1)),
        CheckedTableRange::new(0, 0),
        None,
    )
    .unwrap();
    let program = RecurrenceProgram::new(
        base.strategy(),
        base.physical_sector_count(),
        base.retained_helicity_count(),
        base.dynamic_color_states().to_vec(),
        currents,
        base.contributions().to_vec(),
        base.finalizations().to_vec(),
        base.replay_targets().to_vec(),
        base.resolved_helicities().to_vec(),
        base.amplitude_destinations().to_vec(),
        base.closure_terms().to_vec(),
    )
    .unwrap();
    let parts = lower(&program, &templates, digest(34)).unwrap();
    let factor_id = parts.sources[0].exact_factor_id as usize;
    assert_eq!(parts.exact_factors[factor_id], rational(-1));
}

#[test]
fn prepared_pack_digest_must_match_the_validated_catalog() {
    let templates = validated_template();
    let program = count_fixture_program(&templates, 4, 31, 34, 12, 1);
    let catalog_digest = digest(40);
    let error = lower_recurrence_direct_v2(
        &program,
        &templates,
        &direct_catalog(catalog_digest),
        digest(35),
        digest(36),
        catalog_digest,
        runtime_options(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("does not match validated template pack"));
}

#[test]
fn direct_template_catalog_digest_must_match_the_prepared_resolver() {
    let templates = validated_template();
    let program = count_fixture_program(&templates, 4, 31, 34, 12, 1);
    let error = lower_recurrence_direct_v2(
        &program,
        &templates,
        &direct_catalog(digest(40)),
        digest(35),
        digest(2),
        digest(41),
        runtime_options(),
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("does not match resolver catalog"));
}

fn with_first_identity_finalizer(
    base: &RecurrenceProgram,
    exact_factor: ExactComplexRational,
) -> RecurrenceProgram {
    let mut finalizations = base.finalizations().to_vec();
    let current_id = finalizations[0].current_id();
    finalizations[0] =
        RecurrenceFinalization::new(finalizations[0].id(), current_id, None, exact_factor).unwrap();
    let mut currents = base.currents().to_vec();
    let current = &currents[current_id as usize];
    let key = current.key();
    let identity_key = CurrentCoreKey::new(
        key.catalog_digest(),
        key.node_kind(),
        key.current_state_template_id(),
        key.dynamic_lc_color_state_id(),
        key.support_source_slots().to_vec(),
        key.momentum().clone(),
        key.helicity_identity().clone(),
        key.flavour_flow().to_vec(),
        key.quantum_number_flow_id(),
        key.coupling_orders().to_vec(),
        key.source_binding().clone(),
        None,
    )
    .unwrap();
    currents[current_id as usize] = RecurrenceCurrent::new(
        current.id(),
        identity_key,
        current.source_exact_factor(),
        current.contribution_range(),
        current.finalization_id(),
    )
    .unwrap();
    RecurrenceProgram::new(
        base.strategy(),
        base.physical_sector_count(),
        base.retained_helicity_count(),
        base.dynamic_color_states().to_vec(),
        currents,
        base.contributions().to_vec(),
        finalizations,
        base.replay_targets().to_vec(),
        base.resolved_helicities().to_vec(),
        base.amplitude_destinations().to_vec(),
        base.closure_terms().to_vec(),
    )
    .unwrap()
}

#[test]
fn unit_identity_finalizer_is_elided() {
    let templates = validated_template();
    let base = count_fixture_program(&templates, 4, 31, 34, 12, 1);
    let current_id = base.finalizations()[0].current_id();
    let program = with_first_identity_finalizer(&base, ExactComplexRational::ONE);

    let parts = lower(&program, &templates, digest(42)).unwrap();
    assert_eq!(
        parts.currents[current_id as usize].finalization_row_or_sentinel,
        DIRECT_NONE_U32
    );
    assert_eq!(parts.finalizations.len(), base.finalizations().len() - 1);
    assert!(!parts.row_groups.iter().any(|group| {
        group.role == DirectExecutorRole::Finalization && group.direct_executor_id == 5
    }));
}

#[test]
fn nonunit_identity_and_nonidentity_finalizers_are_preserved() {
    let templates = validated_template();
    let base = count_fixture_program(&templates, 4, 31, 34, 12, 1);
    let identity = with_first_identity_finalizer(&base, rational(2));
    let identity_parts = lower(&identity, &templates, digest(43)).unwrap();
    assert_eq!(
        identity_parts.finalizations.len(),
        base.finalizations().len()
    );
    assert!(identity_parts.row_groups.iter().any(|group| {
        group.role == DirectExecutorRole::Finalization && group.direct_executor_id == 5
    }));

    let propagated_parts = lower(&base, &templates, digest(44)).unwrap();
    assert_eq!(
        propagated_parts.finalizations.len(),
        base.finalizations().len()
    );
    assert!(propagated_parts.row_groups.iter().any(|group| {
        group.role == DirectExecutorRole::Finalization && group.direct_executor_id == 2
    }));
}

#[test]
fn lowering_marks_exactly_one_initializing_contribution_per_destination() {
    let templates = validated_template();
    let program = count_fixture_program(&templates, 4, 31, 34, 12, 1);
    let parts = lower(&program, &templates, digest(45)).unwrap();
    let mut initialization_counts = BTreeMap::<(u16, u32), usize>::new();
    let mut contribution_counts = BTreeMap::<(u16, u32), usize>::new();
    for group in parts
        .row_groups
        .iter()
        .filter(|group| group.role == DirectExecutorRole::Contribution)
    {
        let start = group.row_start as usize;
        let end = start + group.row_count as usize;
        for row in &parts.contributions[start..end] {
            let key = (group.stage, row.destination_component_base);
            *contribution_counts.entry(key).or_default() += 1;
            if row.flags & DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION != 0 {
                *initialization_counts.entry(key).or_default() += 1;
            }
        }
    }
    assert!(
        contribution_counts.values().any(|count| *count > 1),
        "fixture must exercise a multi-contribution destination"
    );
    assert!(
        contribution_counts
            .keys()
            .all(|key| initialization_counts.get(key) == Some(&1))
    );
}
