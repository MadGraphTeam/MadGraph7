// SPDX-License-Identifier: 0BSD

use super::*;

fn valid_physics() -> ProcessPhysics {
    ProcessPhysics {
        schema_version: RUNTIME_PHYSICS_SCHEMA_VERSION,
        kind: "pyamplicol-resolved-physics".to_string(),
        process_id: "p0".to_string(),
        process: "a b > c".to_string(),
        color_accuracy: ColorAccuracy::Full,
        coverage: Coverage {
            helicities: "complete".to_string(),
            color: "contracted".to_string(),
            color_kind: "contracted-color".to_string(),
            structural_zero_helicity_count: 0,
        },
        external_particles: (0..3)
            .map(|index| ExternalParticle {
                index,
                label: index + 1,
                particle: format!("particle-{index}"),
                pdg: index as i32 + 1,
                role: if index < 2 {
                    ParticleRole::Initial
                } else {
                    ParticleRole::Final
                },
                momentum_slot: index,
                momentum_components: [
                    "E".to_string(),
                    "px".to_string(),
                    "py".to_string(),
                    "pz".to_string(),
                ],
            })
            .collect(),
        helicities: vec![Helicity {
            id: "helicity:0".to_string(),
            index: 0,
            values: vec![1, -1, 1],
            computed: true,
            structural_zero: false,
            representative_id: "helicity:0".to_string(),
            coefficient: 1.0,
        }],
        color_components: vec![ColorComponent::ContractedColor(ContractedColor {
            id: "contracted".to_string(),
            index: 0,
            description: "coherent contracted color".to_string(),
        })],
        reduction: Reduction {
            kind: ReductionKind::ContractedColor,
            groups: vec![ReductionGroup {
                id: "group:0".to_string(),
                representative_helicity_id: "helicity:0".to_string(),
                representative_color_id: "contracted".to_string(),
                physical_helicity_ids: vec!["helicity:0".to_string()],
                physical_color_ids: vec!["contracted".to_string()],
            }],
        },
        model_parameters: Vec::new(),
        selectors: SelectorCapabilities {
            helicity: true,
            color_flow: false,
            contracted_color: false,
        },
        extensions: BTreeMap::new(),
    }
}

#[test]
fn normative_physics_metadata_validates() {
    valid_physics().validate().unwrap();
}

#[test]
fn nonzero_members_require_computed_representatives() {
    let mut physics = valid_physics();
    physics.helicities[0].computed = false;

    let error = physics.validate().unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("non-computed representative"));
}

#[test]
fn momentum_slots_must_be_a_complete_permutation() {
    let mut physics = valid_physics();
    physics.external_particles[2].momentum_slot = 1;

    let error = physics.validate().unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("momentum slot"));
}
