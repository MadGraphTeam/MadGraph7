// SPDX-License-Identifier: 0BSD

use super::*;
use serde_json::json;

fn test_physics_runtime(color_accuracy: &str) -> PhysicsRuntime {
    let contracted = color_accuracy != "lc";
    let color_components = if contracted {
        vec![crate::ColorComponent::ContractedColor(
            crate::ContractedColor {
                id: "contracted".to_string(),
                index: 0,
                description: "contracted color sum".to_string(),
            },
        )]
    } else {
        vec![
            crate::ColorComponent::LcFlow(crate::LcColorFlow {
                id: "flow:0".to_string(),
                index: 0,
                word: vec![1, 2],
                representative_id: "flow:0".to_string(),
                computed: true,
                coefficient: 1.0,
            }),
            crate::ColorComponent::LcFlow(crate::LcColorFlow {
                id: "flow:1".to_string(),
                index: 1,
                word: vec![2, 1],
                representative_id: "flow:0".to_string(),
                computed: false,
                coefficient: 1.0,
            }),
        ]
    };
    let physical_color_ids = color_components
        .iter()
        .map(|item| item.id().to_string())
        .collect();
    let mut helicities = vec![
        crate::Helicity {
            id: "hel:+-".to_string(),
            index: 0,
            values: vec![1, -1, 1],
            representative_id: "hel:+-".to_string(),
            computed: true,
            structural_zero: false,
            coefficient: 1.0,
        },
        crate::Helicity {
            id: "hel:-+".to_string(),
            index: 1,
            values: vec![-1, 1, 1],
            representative_id: "hel:+-".to_string(),
            computed: false,
            structural_zero: false,
            coefficient: 1.0,
        },
    ];
    if !contracted {
        helicities.push(crate::Helicity {
            id: "hel:zero".to_string(),
            index: 2,
            values: vec![1, 1, 1],
            representative_id: "hel:zero".to_string(),
            computed: false,
            structural_zero: true,
            coefficient: 0.0,
        });
    }
    PhysicsRuntime::new(ProcessPhysicsV1 {
        schema_version: crate::RUNTIME_PHYSICS_SCHEMA_VERSION,
        kind: "pyamplicol-resolved-physics".to_string(),
        process_id: "x_x_to_y".to_string(),
        process: "x x > y".to_string(),
        color_accuracy: if contracted {
            if color_accuracy == "nlc" {
                crate::ColorAccuracy::Nlc
            } else {
                crate::ColorAccuracy::Full
            }
        } else {
            crate::ColorAccuracy::Lc
        },
        coverage: crate::Coverage {
            helicities: "complete".to_string(),
            color: if contracted { "contracted" } else { "complete" }.to_string(),
            color_kind: if contracted {
                "contracted-color"
            } else {
                "physical-lc-flows"
            }
            .to_string(),
            structural_zero_helicity_count: usize::from(!contracted),
        },
        external_particles: vec![
            test_external_particle(0, "x", 1, crate::ParticleRole::Initial),
            test_external_particle(1, "x~", -1, crate::ParticleRole::Initial),
            test_external_particle(2, "y", 23, crate::ParticleRole::Final),
        ],
        helicities,
        color_components,
        reduction: crate::Reduction {
            kind: if contracted {
                crate::ReductionKind::ContractedColor
            } else {
                crate::ReductionKind::LcDiagonal
            },
            groups: vec![crate::ReductionGroup {
                id: "group:7".to_string(),
                representative_helicity_id: "hel:+-".to_string(),
                physical_helicity_ids: vec!["hel:+-".to_string(), "hel:-+".to_string()],
                representative_color_id: if contracted {
                    "contracted".to_string()
                } else {
                    "flow:0".to_string()
                },
                physical_color_ids,
            }],
        },
        model_parameters: Vec::new(),
        selectors: crate::SelectorCapabilities {
            helicity: true,
            color_flow: !contracted,
            contracted_color: false,
        },
        extensions: BTreeMap::new(),
    })
    .unwrap()
}

fn test_external_particle(
    index: usize,
    particle: &str,
    pdg: i32,
    role: crate::ParticleRole,
) -> crate::ExternalParticle {
    crate::ExternalParticle {
        index,
        label: index + 1,
        particle: particle.to_string(),
        pdg,
        role,
        momentum_slot: index,
        momentum_components: [
            "E".to_string(),
            "px".to_string(),
            "py".to_string(),
            "pz".to_string(),
        ],
    }
}

fn replay_test_physics() -> PhysicsRuntime {
    PhysicsRuntime::new(ProcessPhysicsV1 {
        schema_version: crate::RUNTIME_PHYSICS_SCHEMA_VERSION,
        kind: "pyamplicol-resolved-physics".to_string(),
        process_id: "d_dbar_to_g_g".to_string(),
        process: "d d~ > g g".to_string(),
        color_accuracy: crate::ColorAccuracy::Lc,
        coverage: crate::Coverage {
            helicities: "complete".to_string(),
            color: "complete".to_string(),
            color_kind: "physical-lc-flows".to_string(),
            structural_zero_helicity_count: 0,
        },
        external_particles: vec![
            test_external_particle(0, "d", 1, crate::ParticleRole::Initial),
            test_external_particle(1, "d~", -1, crate::ParticleRole::Initial),
            test_external_particle(2, "g", 21, crate::ParticleRole::Final),
            test_external_particle(3, "g", 21, crate::ParticleRole::Final),
        ],
        helicities: vec![
            crate::Helicity {
                id: "h:+1,-1,+1,-1".to_string(),
                index: 0,
                values: vec![1, -1, 1, -1],
                computed: true,
                structural_zero: false,
                representative_id: "h:+1,-1,+1,-1".to_string(),
                coefficient: 1.0,
            },
            crate::Helicity {
                id: "h:+1,-1,-1,+1".to_string(),
                index: 1,
                values: vec![1, -1, -1, 1],
                computed: false,
                structural_zero: false,
                representative_id: "h:+1,-1,+1,-1".to_string(),
                coefficient: 1.0,
            },
        ],
        color_components: vec![
            crate::ColorComponent::LcFlow(crate::LcColorFlow {
                id: "flow:1,2,3,4".to_string(),
                index: 0,
                word: vec![1, 2, 3, 4],
                computed: true,
                representative_id: "flow:1,2,3,4".to_string(),
                coefficient: 1.0,
            }),
            crate::ColorComponent::LcFlow(crate::LcColorFlow {
                id: "flow:1,4,3,2".to_string(),
                index: 1,
                word: vec![1, 4, 3, 2],
                computed: false,
                representative_id: "flow:1,2,3,4".to_string(),
                coefficient: 1.0,
            }),
            crate::ColorComponent::LcFlow(crate::LcColorFlow {
                id: "flow:1,2,4,3".to_string(),
                index: 2,
                word: vec![1, 2, 4, 3],
                computed: false,
                representative_id: "flow:1,2,3,4".to_string(),
                coefficient: 1.0,
            }),
        ],
        reduction: crate::Reduction {
            kind: crate::ReductionKind::LcDiagonal,
            groups: vec![crate::ReductionGroup {
                id: "reduction:7".to_string(),
                representative_helicity_id: "h:+1,-1,+1,-1".to_string(),
                representative_color_id: "flow:1,2,3,4".to_string(),
                physical_helicity_ids: vec!["h:+1,-1,+1,-1".to_string()],
                physical_color_ids: vec!["flow:1,2,3,4".to_string()],
            }],
        },
        model_parameters: Vec::new(),
        selectors: crate::SelectorCapabilities {
            helicity: true,
            color_flow: true,
            contracted_color: false,
        },
        extensions: BTreeMap::new(),
    })
    .unwrap()
}

#[test]
fn lc_replay_routes_materialized_cells_to_public_axes_and_selectors() {
    let physics = replay_test_physics();
    let mappings = vec![Vec::new(), vec![(2, 3), (3, 2)]];
    let routes = vec![
        vec![LcTopologyReplaySectorRoute {
            physical_sector_id: 0,
            materialized_sector_id: 0,
            weight: 2.0,
            sign: 1,
            amplitude_factor: [2.0, 0.0],
            residual: false,
        }],
        vec![LcTopologyReplaySectorRoute {
            physical_sector_id: 1,
            materialized_sector_id: 0,
            weight: 1.0,
            sign: -1,
            amplitude_factor: [-1.0, 0.0],
            residual: false,
        }],
    ];
    let plan = physics
        .lc_resolved_replay_plan(
            &mappings,
            &routes,
            &BTreeMap::from([(
                0,
                LcMaterializedSector {
                    color_index: 0,
                    reduction_weight: 2.0,
                },
            )]),
        )
        .unwrap();
    assert_eq!(plan.color_count, 3);
    let replay_selection = physics
        .select_lc_resolved_replay_plan(
            &plan,
            Some(&BTreeSet::from(["h:+1,-1,-1,+1".to_string()])),
            Some(&BTreeSet::from(["flow:1,2,4,3".to_string()])),
        )
        .unwrap();
    assert_eq!(replay_selection.mapping_indices, vec![1]);
    assert_eq!(replay_selection.entries.len(), 1);
    assert_eq!(replay_selection.entries[0].routes.len(), 1);
    assert_eq!(replay_selection.entries[0].routes[0].source_index, 0);
    assert_eq!(replay_selection.entries[0].routes[0].target_index, 0);
    assert_eq!(replay_selection.source_helicity_indices, vec![vec![0]]);
    assert_eq!(replay_selection.source_color_indices, vec![vec![0]]);

    let materialized = ResolvedValues {
        values: vec![
            3.0, 0.0, 0.0, 0.0, 0.0, 0.0, // identity, point 0
            5.0, 0.0, 0.0, 0.0, 0.0, 0.0, // identity, point 1
            7.0, 0.0, 0.0, 0.0, 0.0, 0.0, // swap, point 0
            11.0, 0.0, 0.0, 0.0, 0.0, 0.0, // swap, point 1
        ],
        point_count: 4,
        helicity_indices: vec![0, 1],
        color_indices: vec![0, 1, 2],
    };
    let mut full = vec![0.0; 12];
    super::evaluation::accumulate_lc_replay_resolved_f64(
        &mut full,
        2,
        &materialized,
        &plan.entries,
        6,
    )
    .unwrap();

    assert_eq!(
        full,
        vec![3.0, 3.0, 0.0, 0.0, 0.0, 7.0, 5.0, 5.0, 0.0, 0.0, 0.0, 11.0]
    );
    assert_eq!(full[..6].iter().sum::<f64>(), 2.0 * 3.0 + 7.0);
    assert_eq!(full[6..].iter().sum::<f64>(), 2.0 * 5.0 + 11.0);

    let selected = super::evaluation::select_resolved_values(
        full,
        2,
        &physics,
        Some(&BTreeSet::from(["h:+1,-1,-1,+1".to_string()])),
        Some(&BTreeSet::from(["flow:1,2,4,3".to_string()])),
    )
    .unwrap();
    assert_eq!(selected.helicity_indices, vec![1]);
    assert_eq!(selected.color_indices, vec![2]);
    assert_eq!(selected.values, vec![7.0, 11.0]);
}

#[test]
fn lc_replay_requires_every_public_flow_reduction_member() {
    let mut manifest = replay_test_physics().manifest;
    manifest.color_components.remove(1);
    manifest.color_components[1] = match manifest.color_components[1].clone() {
        crate::ColorComponent::LcFlow(mut flow) => {
            flow.index = 1;
            crate::ColorComponent::LcFlow(flow)
        }
        value => value,
    };
    manifest.coverage.color = "selected".to_string();
    let physics = PhysicsRuntime::new(manifest).unwrap();

    let error = physics
        .lc_resolved_replay_plan(
            &vec![Vec::new()],
            &[vec![LcTopologyReplaySectorRoute {
                physical_sector_id: 0,
                materialized_sector_id: 0,
                weight: 2.0,
                sign: 1,
                amplitude_factor: [2.0, 0.0],
                residual: false,
            }]],
            &BTreeMap::from([(
                0,
                LcMaterializedSector {
                    color_index: 0,
                    reduction_weight: 2.0,
                },
            )]),
        )
        .unwrap_err();

    assert!(error.to_string().contains("missing replayed LC flow word"));
}

#[test]
fn lc_replay_expands_trace_reflection_for_residual_sectors() {
    fn flow(
        index: usize,
        word: &[usize],
        computed: bool,
        representative: &[usize],
    ) -> crate::ColorComponent {
        let id = format!(
            "flow:{}",
            word.iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        let representative_id = format!(
            "flow:{}",
            representative
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        );
        crate::ColorComponent::LcFlow(crate::LcColorFlow {
            id,
            index,
            word: word.to_vec(),
            computed,
            representative_id,
            coefficient: 1.0,
        })
    }

    let mut manifest = replay_test_physics().manifest;
    manifest.color_components = vec![
        flow(0, &[1, 2, 3, 4], true, &[1, 2, 3, 4]),
        flow(1, &[1, 2, 4, 3], false, &[1, 2, 3, 4]),
        flow(2, &[1, 3, 2, 4], true, &[1, 3, 2, 4]),
        flow(3, &[1, 3, 4, 2], false, &[1, 2, 3, 4]),
        flow(4, &[1, 4, 2, 3], false, &[1, 3, 2, 4]),
        flow(5, &[1, 4, 3, 2], false, &[1, 2, 3, 4]),
    ];
    manifest.reduction.groups = vec![
        crate::ReductionGroup {
            id: "reduction:0".to_string(),
            representative_helicity_id: "h:+1,-1,+1,-1".to_string(),
            representative_color_id: "flow:1,2,3,4".to_string(),
            physical_helicity_ids: vec!["h:+1,-1,+1,-1".to_string()],
            physical_color_ids: vec!["flow:1,2,3,4".to_string(), "flow:1,4,3,2".to_string()],
        },
        crate::ReductionGroup {
            id: "reduction:1".to_string(),
            representative_helicity_id: "h:+1,-1,+1,-1".to_string(),
            representative_color_id: "flow:1,3,2,4".to_string(),
            physical_helicity_ids: vec!["h:+1,-1,+1,-1".to_string()],
            physical_color_ids: vec!["flow:1,3,2,4".to_string(), "flow:1,4,2,3".to_string()],
        },
    ];
    let physics = PhysicsRuntime::new(manifest).unwrap();
    let mappings = vec![Vec::new(), vec![(2, 3), (3, 2)]];
    let routes = vec![
        vec![
            LcTopologyReplaySectorRoute {
                physical_sector_id: 0,
                materialized_sector_id: 0,
                weight: 2.0,
                sign: 1,
                amplitude_factor: [2.0, 0.0],
                residual: false,
            },
            LcTopologyReplaySectorRoute {
                physical_sector_id: 2,
                materialized_sector_id: 2,
                weight: 1.0,
                sign: 1,
                amplitude_factor: [1.0, 0.0],
                residual: true,
            },
        ],
        vec![LcTopologyReplaySectorRoute {
            physical_sector_id: 1,
            materialized_sector_id: 0,
            weight: 2.0,
            sign: 1,
            amplitude_factor: [2.0, 0.0],
            residual: false,
        }],
    ];
    let materialized = BTreeMap::from([
        (
            0,
            LcMaterializedSector {
                color_index: 0,
                reduction_weight: 2.0,
            },
        ),
        (
            2,
            LcMaterializedSector {
                color_index: 2,
                reduction_weight: 2.0,
            },
        ),
    ]);

    let plan = physics
        .lc_resolved_replay_plan(&mappings, &routes, &materialized)
        .unwrap();
    let target_colors = plan
        .entries
        .iter()
        .flat_map(|entry| entry.routes.iter())
        .map(|route| route.target_index % plan.color_count)
        .collect::<BTreeSet<_>>();

    assert_eq!(target_colors, (0..6).collect());
    assert!(
        plan.entries
            .iter()
            .flat_map(|entry| entry.routes.iter())
            .all(|route| route.weight.to_bits() == 1.0f64.to_bits())
    );
}

fn alias_test_physics(color_accuracy: &str) -> ProcessPhysicsV1 {
    let contracted = color_accuracy != "lc";
    let representative_helicity_id = "h:+1,-1,+0,+1,-1".to_string();
    let helicities = vec![
        crate::Helicity {
            id: representative_helicity_id.clone(),
            index: 0,
            values: vec![1, -1, 0, 1, -1],
            representative_id: representative_helicity_id.clone(),
            computed: true,
            structural_zero: false,
            coefficient: 1.0,
        },
        crate::Helicity {
            id: "h:-1,+1,+0,-1,+1".to_string(),
            index: 1,
            values: vec![-1, 1, 0, -1, 1],
            representative_id: representative_helicity_id.clone(),
            computed: false,
            structural_zero: false,
            coefficient: 1.0,
        },
    ];
    let (color_components, reduction_kind, representative_color_id, physical_color_ids) =
        if contracted {
            (
                vec![crate::ColorComponent::ContractedColor(
                    crate::ContractedColor {
                        id: "color:contracted".to_string(),
                        index: 0,
                        description: "contracted color sum".to_string(),
                    },
                )],
                crate::ReductionKind::ContractedColor,
                "color:contracted".to_string(),
                vec!["color:contracted".to_string()],
            )
        } else {
            (
                vec![
                    crate::ColorComponent::LcFlow(crate::LcColorFlow {
                        id: "flow:1,3,4,5,2".to_string(),
                        index: 0,
                        word: vec![1, 3, 4, 5, 2],
                        representative_id: "flow:1,3,4,5,2".to_string(),
                        computed: true,
                        coefficient: 1.0,
                    }),
                    crate::ColorComponent::LcFlow(crate::LcColorFlow {
                        id: "flow:1,5,4,3,2".to_string(),
                        index: 1,
                        word: vec![1, 5, 4, 3, 2],
                        representative_id: "flow:1,3,4,5,2".to_string(),
                        computed: false,
                        coefficient: 1.0,
                    }),
                ],
                crate::ReductionKind::LcDiagonal,
                "flow:1,3,4,5,2".to_string(),
                vec!["flow:1,3,4,5,2".to_string(), "flow:1,5,4,3,2".to_string()],
            )
        };
    ProcessPhysicsV1 {
        schema_version: crate::RUNTIME_PHYSICS_SCHEMA_VERSION,
        kind: "pyamplicol-resolved-physics".to_string(),
        process_id: "representative".to_string(),
        process: "d d~ > z g a".to_string(),
        color_accuracy: match color_accuracy {
            "lc" => crate::ColorAccuracy::Lc,
            "nlc" => crate::ColorAccuracy::Nlc,
            _ => crate::ColorAccuracy::Full,
        },
        coverage: crate::Coverage {
            helicities: "complete".to_string(),
            color: if contracted { "contracted" } else { "complete" }.to_string(),
            color_kind: if contracted {
                "contracted-color"
            } else {
                "physical-lc-flows"
            }
            .to_string(),
            structural_zero_helicity_count: 0,
        },
        external_particles: vec![
            test_external_particle(0, "d", 1, crate::ParticleRole::Initial),
            test_external_particle(1, "d~", -1, crate::ParticleRole::Initial),
            test_external_particle(2, "z", 23, crate::ParticleRole::Final),
            test_external_particle(3, "g", 21, crate::ParticleRole::Final),
            test_external_particle(4, "a", 22, crate::ParticleRole::Final),
        ],
        helicities,
        color_components,
        reduction: crate::Reduction {
            kind: reduction_kind,
            groups: vec![crate::ReductionGroup {
                id: "group:7".to_string(),
                representative_helicity_id,
                representative_color_id,
                physical_helicity_ids: vec![
                    "h:+1,-1,+0,+1,-1".to_string(),
                    "h:-1,+1,+0,-1,+1".to_string(),
                ],
                physical_color_ids,
            }],
        },
        model_parameters: Vec::new(),
        selectors: crate::SelectorCapabilities {
            helicity: true,
            color_flow: !contracted,
            contracted_color: false,
        },
        extensions: BTreeMap::new(),
    }
}

fn three_cycle_alias() -> crate::ProcessAlias {
    crate::ProcessAlias {
        id: "cycled".to_string(),
        expression: "d d~ > a z g".to_string(),
        external_pdgs: vec![1, -1, 22, 23, 21],
        external_permutation: vec![0, 1, 3, 4, 2],
    }
}

#[test]
fn final_state_alias_three_cycle_remaps_lc_metadata_and_selectors() {
    let representative_manifest = alias_test_physics("lc");
    representative_manifest.validate().unwrap();
    let representative_physics = PhysicsRuntime::new(representative_manifest.clone()).unwrap();
    let alias = three_cycle_alias();
    let alias_manifest =
        apply_final_state_alias_metadata(representative_manifest.clone(), &alias).unwrap();

    let helicity_id_map = representative_manifest
        .helicities
        .iter()
        .zip(&alias_manifest.helicities)
        .map(|(representative, public)| (representative.id.clone(), public.id.clone()))
        .collect::<BTreeMap<_, _>>();
    let color_id_map = representative_manifest
        .color_components
        .iter()
        .zip(&alias_manifest.color_components)
        .map(|(representative, public)| (representative.id().to_string(), public.id().to_string()))
        .collect::<BTreeMap<_, _>>();
    let mut override_runtime = empty_generic_runtime();
    override_runtime.physics_reduction_override = Some(representative_manifest.reduction.clone());
    override_runtime
        .remap_physics_reduction_overrides(&helicity_id_map, &color_id_map)
        .unwrap();
    assert_eq!(
        override_runtime.physics_reduction_override,
        Some(alias_manifest.reduction.clone())
    );

    assert_eq!(alias_manifest.process_id, "cycled");
    assert_eq!(alias_manifest.process, "d d~ > a z g");
    assert_eq!(
        alias_manifest
            .external_particles
            .iter()
            .map(|particle| (
                particle.index,
                particle.label,
                particle.momentum_slot,
                particle.pdg
            ))
            .collect::<Vec<_>>(),
        vec![
            (0, 1, 0, 1),
            (1, 2, 1, -1),
            (2, 3, 2, 22),
            (3, 4, 3, 23),
            (4, 5, 4, 21),
        ]
    );
    assert_eq!(
        alias_manifest
            .external_particles
            .iter()
            .map(|particle| particle.particle.as_str())
            .collect::<Vec<_>>(),
        vec!["d", "d~", "a", "z", "g"]
    );
    assert_eq!(alias_manifest.helicities[0].values, vec![1, -1, -1, 0, 1]);
    assert_eq!(alias_manifest.helicities[0].id, "h:+1,-1,-1,+0,+1");
    assert_eq!(
        alias_manifest.helicities[1].representative_id,
        "h:+1,-1,-1,+0,+1"
    );
    let PhysicsColorComponentV1::LcFlow(first_flow) = &alias_manifest.color_components[0] else {
        panic!("expected LC flow");
    };
    let PhysicsColorComponentV1::LcFlow(second_flow) = &alias_manifest.color_components[1] else {
        panic!("expected LC flow");
    };
    assert_eq!(first_flow.word, vec![1, 4, 5, 3, 2]);
    assert_eq!(first_flow.id, "flow:1,4,5,3,2");
    assert_eq!(second_flow.id, "flow:1,3,5,4,2");
    assert_eq!(second_flow.representative_id, "flow:1,4,5,3,2");
    assert_eq!(
        alias_manifest.reduction.groups[0].physical_helicity_ids,
        vec![
            "h:+1,-1,-1,+0,+1".to_string(),
            "h:-1,+1,+1,+0,-1".to_string(),
        ]
    );
    assert_eq!(
        alias_manifest.reduction.groups[0].physical_color_ids,
        vec!["flow:1,4,5,3,2".to_string(), "flow:1,3,5,4,2".to_string(),]
    );

    let alias_physics = PhysicsRuntime::new(alias_manifest.clone()).unwrap();
    assert_eq!(
        alias_physics
            .selected_helicity_indices(Some(&BTreeSet::from(["h:-1,+1,+1,+0,-1".to_string(),])))
            .unwrap(),
        vec![1]
    );
    assert_eq!(
        alias_physics
            .selected_color_indices(Some(&BTreeSet::from(["flow:1,3,5,4,2".to_string(),])))
            .unwrap(),
        vec![1]
    );

    let representative_total = test_amplitude_runtime(vec![c64(2.0, 0.0)], None)
        .reduce_scratch_f64_resolved(1, &representative_physics, 4.0, None, None)
        .unwrap()
        .values
        .iter()
        .sum::<f64>();
    let alias_total = test_amplitude_runtime(vec![c64(2.0, 0.0)], None)
        .reduce_scratch_f64_resolved(1, &alias_physics, 4.0, None, None)
        .unwrap()
        .values
        .iter()
        .sum::<f64>();
    assert_eq!(alias_total, representative_total);

    let representative_point = (0..5)
        .map(|index| [index as f64, index as f64 + 0.1, 0.0, 0.0])
        .collect::<Vec<_>>();
    let mut alias_point = vec![[0.0; 4]; 5];
    for (representative_index, alias_index) in
        alias.external_permutation.iter().copied().enumerate()
    {
        alias_point[alias_index] = representative_point[representative_index];
    }
    let crossing_map = alias
        .external_permutation
        .iter()
        .copied()
        .enumerate()
        .map(|(target_index, source_index)| InputCrossingMapEntry {
            target_index,
            source_index,
            sign: 1.0,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        apply_input_crossing_map(vec![alias_point], 5, Some(&crossing_map)).unwrap(),
        vec![representative_point]
    );

    let mut execution = empty_generic_runtime();
    execution.external_pdg_order = alias.external_pdgs.clone();
    execution.external_count = 5;
    execution.physics = Some(Arc::new(alias_physics));
    let runtime = NativeRuntime {
        root: PathBuf::new(),
        runtime: execution,
        execution_lane: NativeExecutionLane::Compiled,
        process: alias.expression,
        process_key: alias.id,
        input_crossing_map: Some(crossing_map),
        final_state_permutation_alias_of: Some("representative".to_string()),
        physics_v1: alias_manifest,
        warnings_muted: false,
        warned_kinds: BTreeSet::new(),
        pending_warnings: Vec::new(),
        point_selector_scratch: PointSelectorExecutionScratch::default(),
        selector_simd_lane_width: 1,
    };
    assert_eq!(
        runtime.metadata().external_pdg_order,
        vec![1, -1, 22, 23, 21]
    );
}

#[test]
fn final_state_alias_three_cycle_preserves_contracted_color_reduction() {
    let representative_manifest = alias_test_physics("full");
    representative_manifest.validate().unwrap();
    let representative_physics = PhysicsRuntime::new(representative_manifest.clone()).unwrap();
    let alias_manifest =
        apply_final_state_alias_metadata(representative_manifest, &three_cycle_alias()).unwrap();
    let alias_physics = PhysicsRuntime::new(alias_manifest.clone()).unwrap();

    assert_eq!(alias_manifest.color_components[0].id(), "color:contracted");
    assert_eq!(
        alias_manifest.reduction.groups[0].representative_color_id,
        "color:contracted"
    );
    assert_eq!(
        alias_physics
            .selected_helicity_indices(Some(&BTreeSet::from(["h:-1,+1,+1,+0,-1".to_string(),])))
            .unwrap(),
        vec![1]
    );
    assert_eq!(
        alias_physics
            .selected_color_indices(Some(&BTreeSet::from(["color:contracted".to_string()])))
            .unwrap(),
        vec![0]
    );

    let contraction = || {
        let groups = [RawSumGroup {
            id: 7,
            indices: vec![0],
            weight: 1.0,
            all_sector_weight: 1.0,
            sector_ids: vec![0],
        }];
        ColorContractionRuntime::new(
            &groups,
            vec![ColorContractionEntry {
                left_group_index: 0,
                right_group_index: 0,
                weight_re: 2.0,
                weight_im: 0.0,
                symmetry_factor: 1.0,
            }],
        )
    };
    let representative_total = test_amplitude_runtime(vec![c64(3.0, 0.0)], Some(contraction()))
        .reduce_scratch_f64_resolved(1, &representative_physics, 2.0, None, None)
        .unwrap()
        .values
        .iter()
        .sum::<f64>();
    let alias_total = test_amplitude_runtime(vec![c64(3.0, 0.0)], Some(contraction()))
        .reduce_scratch_f64_resolved(1, &alias_physics, 2.0, None, None)
        .unwrap()
        .values
        .iter()
        .sum::<f64>();
    assert_eq!(alias_total, representative_total);
}

fn empty_evaluator_group() -> EvaluatorGroup {
    EvaluatorGroup {
        evaluators: Vec::new(),
        input_len: 0,
        input_mappings: Vec::new(),
        input_mapping_spans: Vec::new(),
        output_len: 0,
        chunk_parameter_scratch_f64: Vec::new(),
        chunk_scratch_f64: Vec::new(),
        chunk_parameter_scratch_aosoa_f64: Vec::new(),
        chunk_scratch_aosoa_f64: Vec::new(),
        chunk_input_mapping_scratch: Vec::new(),
    }
}

fn test_amplitude_runtime(
    outputs: Vec<Complex<f64>>,
    color_contraction: Option<ColorContractionRuntime>,
) -> AmplitudeRuntime {
    let output_length = outputs.len();
    AmplitudeRuntime {
        output_length,
        raw_sum_weights: vec![1.0; output_length],
        raw_sum_all_sector_weights: vec![1.0; output_length],
        raw_sum_color_sector_ids: vec![None; output_length],
        raw_sum_groups: vec![RawSumGroup {
            id: 7,
            indices: (0..output_length).collect(),
            weight: 1.0,
            all_sector_weight: 1.0,
            sector_ids: vec![0],
        }],
        has_coherent_groups: true,
        color_contraction,
        input_components: None,
        input_spans: Vec::new(),
        parameter_scratch_f64: Vec::new(),
        evaluator_output_scratch_f64: Vec::new(),
        output_scratch_f64: outputs,
        resolved_source_row_scratch_f64: Vec::new(),
        resolved_target_row_scratch_f64: Vec::new(),
        evaluator_output_order: None,
        evaluator: empty_evaluator_group(),
    }
}

#[test]
fn resolved_lc_reduction_expands_symmetries_and_structural_zeros() {
    let physics = test_physics_runtime("lc");
    let mut amplitude = test_amplitude_runtime(vec![c64(2.0, 0.0)], None);

    let resolved = amplitude
        .reduce_scratch_f64_resolved(1, &physics, 4.0, None, None)
        .unwrap();

    assert_eq!(resolved.helicity_indices, vec![0, 1, 2]);
    assert_eq!(resolved.color_indices, vec![0, 1]);
    assert_eq!(resolved.values, vec![4.0, 4.0, 4.0, 4.0, 0.0, 0.0]);
    assert_eq!(resolved.values.iter().sum::<f64>(), 16.0);

    let helicities = BTreeSet::from(["hel:-+".to_string()]);
    let colors = BTreeSet::from(["flow:1".to_string()]);
    let selected = amplitude
        .reduce_scratch_f64_resolved(1, &physics, 4.0, Some(&helicities), Some(&colors))
        .unwrap();
    assert_eq!(selected.values, vec![4.0]);
}

#[test]
fn contracted_reductions_have_one_color_component_and_sum_to_total() {
    for color_accuracy in ["nlc", "full"] {
        let physics = test_physics_runtime(color_accuracy);
        let groups = [RawSumGroup {
            id: 7,
            indices: vec![0],
            weight: 1.0,
            all_sector_weight: 1.0,
            sector_ids: vec![0],
        }];
        let contraction = ColorContractionRuntime::new(
            &groups,
            vec![ColorContractionEntry {
                left_group_index: 0,
                right_group_index: 0,
                weight_re: 2.0,
                weight_im: 0.0,
                symmetry_factor: 1.0,
            }],
        );
        let mut amplitude = test_amplitude_runtime(vec![c64(3.0, 0.0)], Some(contraction));

        let resolved = amplitude
            .reduce_scratch_f64_resolved(1, &physics, 2.0, None, None)
            .unwrap();

        assert_eq!(resolved.helicity_indices, vec![0, 1]);
        assert_eq!(resolved.color_indices, vec![0]);
        assert_eq!(resolved.values, vec![18.0, 18.0]);
        assert_eq!(resolved.values.iter().sum::<f64>(), 36.0);
    }
}

fn repeated_test_group(id: i64, output_index: usize, sector_id: i64) -> RawSumGroup {
    RawSumGroup {
        id,
        indices: vec![output_index],
        weight: 1.0,
        all_sector_weight: 1.0,
        sector_ids: vec![sector_id],
    }
}

fn legacy_color_contraction_totals(
    amplitudes: &[Complex<f64>],
    output_length: usize,
    groups: &[RawSumGroup],
    entries: &[ColorContractionEntry],
) -> Vec<f64> {
    amplitudes
        .chunks_exact(output_length)
        .map(|row| {
            let group_values = groups
                .iter()
                .map(|group| {
                    group
                        .indices
                        .iter()
                        .map(|index| row[*index])
                        .sum::<Complex<f64>>()
                })
                .collect::<Vec<_>>();
            entries.iter().fold(0.0, |total, entry| {
                let product = group_values[entry.left_group_index]
                    * group_values[entry.right_group_index].conj();
                total
                    + entry.symmetry_factor
                        * (entry.weight_re * product.re - entry.weight_im * product.im)
            })
        })
        .collect()
}

fn reduction_test_amplitude(
    output_length: usize,
    outputs: Vec<Complex<f64>>,
    groups: Vec<RawSumGroup>,
    entries: Vec<ColorContractionEntry>,
) -> AmplitudeRuntime {
    let contraction = ColorContractionRuntime::new(&groups, entries);
    AmplitudeRuntime {
        output_length,
        raw_sum_weights: vec![1.0; output_length],
        raw_sum_all_sector_weights: vec![1.0; output_length],
        raw_sum_color_sector_ids: vec![None; output_length],
        raw_sum_groups: groups,
        has_coherent_groups: true,
        color_contraction: Some(contraction),
        input_components: None,
        input_spans: Vec::new(),
        parameter_scratch_f64: Vec::new(),
        evaluator_output_scratch_f64: Vec::new(),
        output_scratch_f64: outputs,
        resolved_source_row_scratch_f64: Vec::new(),
        resolved_target_row_scratch_f64: Vec::new(),
        evaluator_output_order: None,
        evaluator: empty_evaluator_group(),
    }
}

#[test]
fn repeated_real_color_blocks_match_legacy_reduction_for_permuted_outputs() {
    let groups = vec![
        repeated_test_group(10, 2, 100),
        repeated_test_group(11, 5, 100),
        repeated_test_group(12, 0, 200),
        repeated_test_group(13, 3, 200),
        repeated_test_group(14, 1, 300),
        repeated_test_group(15, 4, 300),
    ];
    // Deliberately interleave the two disconnected components and do not
    // present their left indices monotonically. Runtime canonicalization may
    // change floating-point association, but it must preserve the contraction.
    let entries = vec![
        ColorContractionEntry {
            left_group_index: 3,
            right_group_index: 5,
            weight_re: 0.5,
            weight_im: 0.0,
            symmetry_factor: 2.0,
        },
        ColorContractionEntry {
            left_group_index: 0,
            right_group_index: 0,
            weight_re: 1.25,
            weight_im: 0.0,
            symmetry_factor: 1.0,
        },
        ColorContractionEntry {
            left_group_index: 1,
            right_group_index: 3,
            weight_re: -0.75,
            weight_im: 0.0,
            symmetry_factor: 2.0,
        },
        ColorContractionEntry {
            left_group_index: 4,
            right_group_index: 4,
            weight_re: 2.0,
            weight_im: 0.0,
            symmetry_factor: 1.0,
        },
        ColorContractionEntry {
            left_group_index: 2,
            right_group_index: 4,
            weight_re: 0.5,
            weight_im: 0.0,
            symmetry_factor: 2.0,
        },
        ColorContractionEntry {
            left_group_index: 1,
            right_group_index: 1,
            weight_re: 1.25,
            weight_im: 0.0,
            symmetry_factor: 1.0,
        },
        ColorContractionEntry {
            left_group_index: 0,
            right_group_index: 2,
            weight_re: -0.75,
            weight_im: 0.0,
            symmetry_factor: 2.0,
        },
        ColorContractionEntry {
            left_group_index: 5,
            right_group_index: 5,
            weight_re: 2.0,
            weight_im: 0.0,
            symmetry_factor: 1.0,
        },
    ];
    let outputs = vec![
        c64(0.5, -1.0),
        c64(1.5, 0.25),
        c64(-0.75, 2.0),
        c64(0.125, -0.5),
        c64(2.25, 1.0),
        c64(-1.25, -0.75),
        c64(1.0, 0.5),
        c64(-0.25, 1.25),
        c64(0.75, -1.5),
        c64(2.0, 0.125),
        c64(-1.0, 0.75),
        c64(0.25, -2.0),
    ];
    let expected = legacy_color_contraction_totals(&outputs, 6, &groups, &entries);
    let mut amplitude = reduction_test_amplitude(6, outputs, groups, entries);
    let repeated = amplitude
        .color_contraction
        .as_ref()
        .and_then(|contraction| contraction.repeated_block.as_ref())
        .expect("two identical color components should be canonicalized");
    assert_eq!(repeated.component_count, 2);
    assert_eq!(repeated.entries.len(), 4);
    assert_eq!(
        repeated.singleton_output_indices.as_deref(),
        Some([2, 5, 0, 3, 1, 4].as_slice())
    );
    assert!(repeated.all_weights_real);

    let mut actual = vec![0.0; 2];
    amplitude
        .reduce_scratch_f64_into_selected_slice(2, &mut actual, None)
        .unwrap();
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
            "repeated-block reduction {actual} differs from legacy {expected}"
        );
    }
}

#[test]
fn repeated_color_block_requires_identical_component_coefficients() {
    let groups = vec![
        repeated_test_group(10, 0, 100),
        repeated_test_group(11, 1, 100),
        repeated_test_group(12, 2, 200),
        repeated_test_group(13, 3, 200),
    ];
    let entries = vec![
        ColorContractionEntry {
            left_group_index: 0,
            right_group_index: 2,
            weight_re: 1.0,
            weight_im: 0.0,
            symmetry_factor: 2.0,
        },
        ColorContractionEntry {
            left_group_index: 1,
            right_group_index: 3,
            weight_re: 1.0 + f64::EPSILON,
            weight_im: 0.0,
            symmetry_factor: 2.0,
        },
    ];
    let contraction = ColorContractionRuntime::new(&groups, entries);
    assert!(contraction.repeated_block.is_none());
}

#[test]
fn compact_repeated_color_manifest_builds_without_expanded_entries() {
    let groups = vec![
        repeated_test_group(10, 2, 100),
        repeated_test_group(11, 3, 100),
        repeated_test_group(12, 0, 200),
        repeated_test_group(13, 1, 200),
    ];
    let manifest = GenericColorContractionManifest {
        supported: true,
        reason: None,
        group_count: groups.len(),
        includes_color_factor: true,
        entries: Vec::new(),
        repeated_block: Some(GenericRepeatedColorContractionBlockManifest {
            component_count: 2,
            component_group_ids: vec![10, 11, 12, 13],
            entries: vec![
                GenericRepeatedColorContractionEntryManifest {
                    left_group_index: 0,
                    right_group_index: 0,
                    weight: vec![1.25, 0.0],
                    symmetry_factor: 1.0,
                },
                GenericRepeatedColorContractionEntryManifest {
                    left_group_index: 0,
                    right_group_index: 1,
                    weight: vec![-0.75, 0.0],
                    symmetry_factor: 2.0,
                },
                GenericRepeatedColorContractionEntryManifest {
                    left_group_index: 1,
                    right_group_index: 1,
                    weight: vec![2.0, 0.0],
                    symmetry_factor: 1.0,
                },
            ],
        }),
    };
    let contraction = build_color_contraction_runtime(Some(&manifest), &groups)
        .unwrap()
        .expect("compact repeated contraction");
    assert!(contraction.entries.is_empty());
    assert_eq!(contraction.logical_entry_count().unwrap(), 6);
    let logical_entries = contraction.logical_entries().collect::<Vec<_>>();
    assert_eq!(
        logical_entries
            .iter()
            .map(|entry| (entry.left_group_index, entry.right_group_index))
            .collect::<Vec<_>>(),
        vec![(0, 0), (0, 2), (2, 2), (1, 1), (1, 3), (3, 3)]
    );

    let outputs = vec![
        c64(0.5, -1.0),
        c64(1.5, 0.25),
        c64(-0.75, 2.0),
        c64(0.125, -0.5),
    ];
    let expected = legacy_color_contraction_totals(&outputs, 4, &groups, &logical_entries)[0];
    let mut amplitude = test_amplitude_runtime(outputs, Some(contraction));
    amplitude.raw_sum_groups = groups;
    let mut actual = vec![0.0];
    amplitude
        .reduce_scratch_f64_into_selected_slice(1, &mut actual, None)
        .unwrap();
    assert!((actual[0] - expected).abs() <= 1.0e-12 * expected.abs().max(1.0));
}

#[test]
fn compact_repeated_color_manifest_rejects_duplicate_group_mapping() {
    let groups = vec![
        repeated_test_group(10, 0, 100),
        repeated_test_group(11, 1, 100),
    ];
    let manifest = GenericColorContractionManifest {
        supported: true,
        reason: None,
        group_count: groups.len(),
        includes_color_factor: true,
        entries: Vec::new(),
        repeated_block: Some(GenericRepeatedColorContractionBlockManifest {
            component_count: 2,
            component_group_ids: vec![10, 10],
            entries: Vec::new(),
        }),
    };
    let error = match build_color_contraction_runtime(Some(&manifest), &groups) {
        Ok(_) => panic!("duplicate repeated color group mapping must fail"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("maps a coherent group more than once")
    );
}

#[test]
fn compact_repeated_color_manifest_rejects_malformed_storage() {
    let missing_entries = serde_json::from_value::<GenericColorContractionManifest>(json!({
        "supported": true,
        "group_count": 2,
        "includes_color_factor": true,
    }));
    assert!(missing_entries.is_err());

    let groups = vec![
        repeated_test_group(10, 0, 100),
        repeated_test_group(11, 1, 100),
    ];
    let repeated = |weight: Vec<f64>, left_group_index: usize| {
        Some(GenericRepeatedColorContractionBlockManifest {
            component_count: 2,
            component_group_ids: vec![10, 11],
            entries: vec![GenericRepeatedColorContractionEntryManifest {
                left_group_index,
                right_group_index: 0,
                weight,
                symmetry_factor: 1.0,
            }],
        })
    };
    let malformed = [
        (
            "two components",
            GenericColorContractionManifest {
                supported: true,
                reason: None,
                group_count: 2,
                includes_color_factor: true,
                entries: Vec::new(),
                repeated_block: repeated(vec![1.0], 0),
            },
        ),
        (
            "out of bounds",
            GenericColorContractionManifest {
                supported: true,
                reason: None,
                group_count: 2,
                includes_color_factor: true,
                entries: Vec::new(),
                repeated_block: repeated(vec![1.0, 0.0], 1),
            },
        ),
        (
            "cannot mix",
            GenericColorContractionManifest {
                supported: true,
                reason: None,
                group_count: 2,
                includes_color_factor: true,
                entries: vec![GenericColorContractionEntryManifest {
                    left_group_id: 10,
                    right_group_id: 10,
                    weight: vec![1.0, 0.0],
                    symmetry_factor: 1.0,
                }],
                repeated_block: repeated(vec![1.0, 0.0], 0),
            },
        ),
    ];
    for (expected, manifest) in malformed {
        let error = match build_color_contraction_runtime(Some(&manifest), &groups) {
            Ok(_) => panic!("malformed repeated color storage must fail"),
            Err(error) => error,
        };
        assert!(
            error.to_string().contains(expected),
            "unexpected error for {expected}: {error}"
        );
    }
}

#[test]
fn repeated_complex_color_blocks_match_legacy_reduction() {
    let groups = vec![
        repeated_test_group(10, 0, 100),
        repeated_test_group(11, 1, 100),
        repeated_test_group(12, 2, 200),
        repeated_test_group(13, 3, 200),
    ];
    let entries = vec![
        ColorContractionEntry {
            left_group_index: 0,
            right_group_index: 2,
            weight_re: 0.75,
            weight_im: -0.25,
            symmetry_factor: 2.0,
        },
        ColorContractionEntry {
            left_group_index: 1,
            right_group_index: 3,
            weight_re: 0.75,
            weight_im: -0.25,
            symmetry_factor: 2.0,
        },
    ];
    let outputs = vec![
        c64(1.0, 2.0),
        c64(-0.5, 0.25),
        c64(0.75, -1.0),
        c64(2.0, 0.5),
    ];
    let expected = legacy_color_contraction_totals(&outputs, 4, &groups, &entries);
    let mut amplitude = reduction_test_amplitude(4, outputs, groups, entries);
    assert!(
        !amplitude
            .color_contraction
            .as_ref()
            .unwrap()
            .repeated_block
            .as_ref()
            .unwrap()
            .all_weights_real
    );
    let mut actual = vec![0.0];
    amplitude
        .reduce_scratch_f64_into_selected_slice(1, &mut actual, None)
        .unwrap();
    assert!((actual[0] - expected[0]).abs() <= 1.0e-12 * expected[0].abs().max(1.0));
}

#[test]
fn contracted_color_coverage_does_not_warn_as_incomplete() {
    for color_accuracy in ["nlc", "full"] {
        let physics = test_physics_runtime(color_accuracy);
        let physics_v1 = physics.manifest.clone();
        let mut execution = empty_generic_runtime();
        execution.color_accuracy = color_accuracy.to_string();
        execution.physics = Some(Arc::new(physics));
        let mut runtime = NativeRuntime {
            root: PathBuf::new(),
            runtime: execution,
            execution_lane: NativeExecutionLane::Compiled,
            process: "x x > y".to_string(),
            process_key: "x_x_to_y".to_string(),
            input_crossing_map: None,
            final_state_permutation_alias_of: None,
            physics_v1,
            warnings_muted: false,
            warned_kinds: BTreeSet::new(),
            pending_warnings: Vec::new(),
            point_selector_scratch: PointSelectorExecutionScratch::default(),
            selector_simd_lane_width: 1,
        };

        runtime.record_resolved_warnings(None, None).unwrap();

        assert!(runtime.pending_warnings.is_empty());
    }
}

#[test]
fn inconsistent_helicity_weights_are_rejected() {
    let mut physics = test_physics_runtime("nlc").manifest;
    physics.helicities[1].coefficient = 2.0;

    let error = PhysicsRuntime::new(physics).err().unwrap();

    assert!(error.to_string().contains("inconsistent helicity weights"));
}

fn empty_generic_runtime() -> ExecutionRuntime {
    ExecutionRuntime {
        process: "a b > c".to_string(),
        key: "p0".to_string(),
        color_accuracy: "lc".to_string(),
        external_pdg_order: Vec::new(),
        external_count: 0,
        parameter_count: 1,
        value_parameter_count: 0,
        momentum_parameter_count: 0,
        current_count: 0,
        source_count: 0,
        interaction_count: 0,
        stage_count: 0,
        amplitude_output_count: 0,
        lc_topology_replay_enabled: false,
        lc_topology_replay_mappings: Arc::new(Vec::new()),
        lc_topology_replay_public_mappings: Vec::new(),
        lc_topology_replay_routes: Vec::new(),
        lc_topology_replay_materialized_sector_ids: BTreeSet::new(),
        lc_resolved_replay_plan: None,
        lc_resolved_replay_selection_cache: None,
        helicity_recurrence: None,
        compiled_helicity_execution_plan: None,
        compiled_color_execution_plan: None,
        helicity_sum_runtime: None,
        helicity_selector_runtimes: Vec::new(),
        helicity_selector_runtime_schedule_modes: Vec::new(),
        helicity_selector_lane_by_domain: BTreeMap::new(),
        color_selector_runtimes: BTreeMap::new(),
        runtime_unavailable_message: None,
        sources: Vec::new(),
        momentum_slots: Vec::new(),
        external_is_initial: Vec::new(),
        particle_masses: BTreeMap::new(),
        particle_mass_parameter_names: BTreeMap::new(),
        normalization_factor: 1.0,
        normalization_color_factor: 1.0,
        normalization_average_factor: 1.0,
        normalization_identical_factor: 1.0,
        normalization_qcd_coupling_power: 0,
        normalization_electroweak_coupling_power: 0,
        model_parameters: Vec::new(),
        model_parameter_name_to_index: BTreeMap::new(),
        model_parameter_runtime_slots: BTreeMap::new(),
        model_parameter_values_f64: vec![0.118],
        model_parameter_evaluator: None,
        physics_reduction_override: None,
        physics: None,
        stages: None,
        amplitude_stage: None,
        state_scratch_f64: Vec::new(),
        state_scratch_f64_requires_clear: false,
        values_scratch_f64: Vec::new(),
    }
}

fn zero_native_runtime() -> NativeRuntime {
    let physics = test_physics_runtime("lc");
    let physics_v1 = physics.manifest.clone();
    let mut execution = empty_generic_runtime();
    execution.external_count = 3;
    execution.external_pdg_order = vec![1, -1, 23];
    execution.external_is_initial = vec![true, true, false];
    execution.physics = Some(Arc::new(physics));
    execution.stages = Some(Vec::new());
    let mut amplitude = test_amplitude_runtime(Vec::new(), None);
    amplitude.input_components = Some(Vec::new());
    execution.amplitude_stage = Some(amplitude);
    NativeRuntime {
        root: PathBuf::new(),
        runtime: execution,
        execution_lane: NativeExecutionLane::Compiled,
        process: "x x > y".to_string(),
        process_key: "x_x_to_y".to_string(),
        input_crossing_map: None,
        final_state_permutation_alias_of: None,
        physics_v1,
        warnings_muted: false,
        warned_kinds: BTreeSet::new(),
        pending_warnings: Vec::new(),
        point_selector_scratch: PointSelectorExecutionScratch::default(),
        selector_simd_lane_width: 1,
    }
}

#[test]
fn native_f64_into_matches_allocating_wrapper_and_validates_output() {
    let point = [
        10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, -10.0, 20.0, 0.0, 0.0, 0.0,
    ];
    let momenta = point.repeat(4);
    let mut runtime = zero_native_runtime();
    let allocated = runtime.evaluate_f64(&momenta, 4).unwrap();
    let mut output = vec![f64::NAN; 4];
    runtime.evaluate_f64_into(&momenta, 4, &mut output).unwrap();
    assert_eq!(output, allocated);

    let helicity_by_point = [0_u32, 1, 2, 0];
    runtime
        .evaluate_f64_into_with_selectors(
            &momenta,
            4,
            None,
            None,
            Some(&helicity_by_point),
            None,
            &mut output,
        )
        .unwrap();
    assert_eq!(output, vec![0.0; 4]);

    let error = runtime
        .evaluate_f64_into(&momenta, 4, &mut [0.0; 3])
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("output has length 3, expected 4")
    );
    let error = runtime
        .evaluate_f64_into(&momenta, 4, &mut [0.0; 5])
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("output has length 5, expected 4")
    );
}

#[test]
fn model_parameter_override_batch_is_atomic() {
    let mut runtime = empty_generic_runtime();
    runtime.model_parameter_runtime_slots.insert(
        "alpha_s".to_string(),
        RuntimeParameterSlots {
            real: 0,
            imaginary: None,
        },
    );
    let invalid_batch = BTreeMap::from([
        ("alpha_s".to_string(), (0.101, 0.0)),
        ("unknown.parameter".to_string(), (1.0, 0.0)),
    ]);

    let error = runtime
        .apply_model_parameter_overrides(&invalid_batch)
        .unwrap_err();

    assert!(error.to_string().contains("unknown.parameter"));
    assert_eq!(runtime.model_parameter_values_f64, vec![0.118]);
    runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("alpha_s".to_string(), (0.101, 0.0))]))
        .unwrap();
    assert_eq!(runtime.model_parameter_values_f64, vec![0.101]);
}

#[test]
fn model_parameter_overrides_are_atomic_across_helicity_lanes() {
    let mut runtime = empty_generic_runtime();
    runtime.model_parameter_runtime_slots.insert(
        "alpha_s".to_string(),
        RuntimeParameterSlots {
            real: 0,
            imaginary: None,
        },
    );
    let mut sum_runtime = empty_generic_runtime();
    sum_runtime.model_parameter_runtime_slots.insert(
        "alpha_s".to_string(),
        RuntimeParameterSlots {
            real: 0,
            imaginary: None,
        },
    );
    runtime.helicity_sum_runtime = Some(Box::new(sum_runtime));

    runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("alpha_s".to_string(), (0.101, 0.0))]))
        .unwrap();

    assert_eq!(runtime.model_parameter_values_f64, vec![0.101]);
    assert_eq!(
        runtime
            .helicity_sum_runtime
            .as_ref()
            .unwrap()
            .model_parameter_values_f64,
        vec![0.101]
    );

    runtime
        .helicity_sum_runtime
        .as_mut()
        .unwrap()
        .model_parameter_runtime_slots
        .clear();
    let error = runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("alpha_s".to_string(), (0.089, 0.0))]))
        .unwrap_err();

    assert!(error.to_string().contains("alpha_s"));
    assert_eq!(runtime.model_parameter_values_f64, vec![0.101]);
    assert_eq!(
        runtime
            .helicity_sum_runtime
            .as_ref()
            .unwrap()
            .model_parameter_values_f64,
        vec![0.101]
    );
}

#[test]
fn model_parameter_overrides_are_atomic_across_color_selector_lanes() {
    fn runtime_with_alpha_s() -> ExecutionRuntime {
        let mut runtime = empty_generic_runtime();
        runtime.model_parameter_runtime_slots.insert(
            "alpha_s".to_string(),
            RuntimeParameterSlots {
                real: 0,
                imaginary: None,
            },
        );
        runtime
    }

    let mut runtime = runtime_with_alpha_s();
    runtime
        .color_selector_runtimes
        .insert(0, Box::new(runtime_with_alpha_s()));
    runtime
        .color_selector_runtimes
        .insert(1, Box::new(runtime_with_alpha_s()));

    runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("alpha_s".to_string(), (0.101, 0.0))]))
        .unwrap();
    assert_eq!(runtime.model_parameter_values_f64, vec![0.101]);
    assert!(
        runtime
            .color_selector_runtimes
            .values()
            .all(|lane| lane.model_parameter_values_f64 == vec![0.101])
    );

    runtime
        .color_selector_runtimes
        .get_mut(&1)
        .unwrap()
        .model_parameter_runtime_slots
        .clear();
    let error = runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("alpha_s".to_string(), (0.089, 0.0))]))
        .unwrap_err();

    assert!(error.to_string().contains("alpha_s"));
    assert_eq!(runtime.model_parameter_values_f64, vec![0.101]);
    assert!(
        runtime
            .color_selector_runtimes
            .values()
            .all(|lane| lane.model_parameter_values_f64 == vec![0.101])
    );
}

#[test]
fn model_parameter_overrides_are_atomic_across_shared_helicity_selector_lanes() {
    fn runtime_with_alpha_s() -> ExecutionRuntime {
        let mut runtime = empty_generic_runtime();
        runtime.model_parameter_runtime_slots.insert(
            "alpha_s".to_string(),
            RuntimeParameterSlots {
                real: 0,
                imaginary: None,
            },
        );
        runtime
    }

    let mut runtime = runtime_with_alpha_s();
    runtime
        .helicity_selector_runtimes
        .push(Box::new(runtime_with_alpha_s()));
    runtime
        .helicity_selector_runtime_schedule_modes
        .push(HelicitySelectorScheduleMode::ParentClosure);
    runtime
        .helicity_selector_runtimes
        .push(Box::new(runtime_with_alpha_s()));
    runtime
        .helicity_selector_runtime_schedule_modes
        .push(HelicitySelectorScheduleMode::ParentClosure);
    runtime.helicity_selector_lane_by_domain = BTreeMap::from([(0, 0), (1, 0), (2, 1)]);

    runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("alpha_s".to_string(), (0.101, 0.0))]))
        .unwrap();
    assert_eq!(runtime.model_parameter_values_f64, vec![0.101]);
    assert_eq!(runtime.helicity_selector_runtimes.len(), 2);
    assert_eq!(runtime.helicity_selector_lane_by_domain.len(), 3);
    assert!(
        runtime
            .helicity_selector_runtimes
            .iter()
            .all(|lane| lane.model_parameter_values_f64 == vec![0.101])
    );

    runtime.helicity_selector_runtimes[1]
        .model_parameter_runtime_slots
        .clear();
    let error = runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("alpha_s".to_string(), (0.089, 0.0))]))
        .unwrap_err();

    assert!(error.to_string().contains("alpha_s"));
    assert_eq!(runtime.model_parameter_values_f64, vec![0.101]);
    assert!(
        runtime
            .helicity_selector_runtimes
            .iter()
            .all(|lane| lane.model_parameter_values_f64 == vec![0.101])
    );
}

#[test]
fn alias_external_order_initialization_reaches_shared_helicity_selector_lanes() {
    let mut runtime = empty_generic_runtime();
    runtime
        .helicity_selector_runtimes
        .push(Box::new(empty_generic_runtime()));
    runtime
        .helicity_selector_runtime_schedule_modes
        .push(HelicitySelectorScheduleMode::ParentClosure);
    runtime.helicity_selector_lane_by_domain = BTreeMap::from([(0, 0), (1, 0)]);

    runtime.set_external_pdg_order_recursive(&[1, -1, 23]);

    assert_eq!(runtime.external_pdg_order, vec![1, -1, 23]);
    assert_eq!(runtime.helicity_selector_runtimes.len(), 1);
    assert_eq!(
        runtime.helicity_selector_runtimes[0].external_pdg_order,
        vec![1, -1, 23]
    );
}

#[test]
fn model_parameter_override_rejects_mass_class_changes_atomically() {
    let mut runtime = empty_generic_runtime();
    runtime.model_parameter_values_f64 = vec![91.188];
    runtime.model_parameter_runtime_slots.insert(
        "MZ".to_string(),
        RuntimeParameterSlots {
            real: 0,
            imaginary: None,
        },
    );
    runtime
        .particle_mass_parameter_names
        .insert(23, "MZ".to_string());
    runtime.particle_masses.insert(23, 91.188);

    let error = runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("MZ".to_string(), (0.0, 0.0))]))
        .unwrap_err();

    assert!(error.to_string().contains("mass class"));
    assert!(error.to_string().contains("regenerate"));
    assert_eq!(runtime.model_parameter_values_f64, vec![91.188]);
    assert_eq!(runtime.particle_masses.get(&23), Some(&91.188));

    runtime
        .apply_model_parameter_overrides(&BTreeMap::from([("MZ".to_string(), (100.0, 0.0))]))
        .expect("massive-to-massive update remains valid");
    assert_eq!(runtime.model_parameter_values_f64, vec![100.0]);
    assert_eq!(runtime.particle_masses.get(&23), Some(&100.0));
}

#[test]
fn compiled_color_topology_lane_requires_physics_reduction() {
    let mut manifest: ExecutionManifest = serde_json::from_slice(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../src/pyamplicol/assets/selftest/portable-64le/artifact/processes/",
        "d_dbar_to_z/execution.json"
    )))
    .expect("parse packaged execution fixture");
    let mut lane = manifest.clone();
    lane.physics_reduction = None;
    lane.helicity_sum_execution = None;
    lane.helicity_selector_executions.clear();
    lane.color_selector_executions.clear();
    manifest
        .color_selector_executions
        .push(ColorSelectorExecutionManifest {
            materialized_sector_id: 0,
            execution: Box::new(lane),
        });

    let error = ExecutionRuntime::from_manifest(manifest)
        .err()
        .expect("color topology lane without reduction must fail");

    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
    assert!(
        error.to_string().contains("has no reduction override"),
        "unexpected error: {error}"
    );
}

#[test]
fn eager_native_profile_accepts_non_overlapping_top_level_phases() {
    let mut profile: NativeRuntimeProfile = RuntimeProfile::default().into();
    profile.total_s = 10.0e-3;
    profile.source_fill_s = 1.0e-3;
    profile.momentum_input_setup_s = 0.5e-3;
    profile.momentum_setup_s = 0.5e-3;
    profile.stage_evaluator_call_s = 7.0e-3;
    profile.eager_initialize_s = 0.5e-3;
    profile.eager_kernel_call_s = 6.0e-3;
    profile.eager_copy_out_s = 0.5e-3;

    profile.validate_eager_top_level_accounting().unwrap();
}

#[test]
fn native_profile_preserves_legacy_momentum_setup_aggregate() {
    let profile: NativeRuntimeProfile = RuntimeProfile {
        momentum_input_setup_s: 0.25,
        momentum_setup_s: 0.75,
        model_parameter_setup_s: 0.5,
        ..RuntimeProfile::default()
    }
    .into();

    assert_eq!(profile.momentum_input_setup_s, 0.25);
    assert_eq!(profile.momentum_setup_s, 0.75);
    assert_eq!(profile.model_parameter_setup_s, 0.5);
}

#[test]
fn eager_native_profile_rejects_top_level_overlap() {
    let mut profile: NativeRuntimeProfile = RuntimeProfile::default().into();
    profile.total_s = 10.0e-3;
    profile.source_fill_s = 4.0e-3;
    profile.stage_evaluator_call_s = 7.0e-3;

    let error = profile.validate_eager_top_level_accounting().unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Internal);
    assert!(error.to_string().contains("exclusive top-level phases"));
    assert!(error.to_string().contains("exceeding wall time"));
}

#[test]
fn compiled_native_profile_rejects_top_level_overlap() {
    let mut profile: NativeRuntimeProfile = RuntimeProfile::default().into();
    profile.total_s = 10.0e-3;
    profile.stage_input_pack_s = 4.0e-3;
    profile.stage_evaluator_call_s = 7.0e-3;

    let error = profile
        .validate_compiled_top_level_accounting()
        .unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Internal);
    assert!(error.to_string().contains("exclusive top-level phases"));
    assert!(error.to_string().contains("exceeding wall time"));
}

#[test]
fn native_profile_accumulates_compiled_accounting() {
    let mut profile: NativeRuntimeProfile = RuntimeProfile {
        orchestration_s: 1.0,
        stage_leaf_input_pack_s: 2.0,
        stage_leaf_input_pack_by_stage_s: vec![2.0],
        stage_leaf_input_copy_component_count: 3,
        evaluator_backend_call_count: 4,
        scratch_reallocation_count: 5,
        ..RuntimeProfile::default()
    }
    .into();
    let repeated: NativeRuntimeProfile = RuntimeProfile {
        orchestration_s: 10.0,
        stage_leaf_input_pack_s: 20.0,
        stage_leaf_input_pack_by_stage_s: vec![20.0, 30.0],
        stage_leaf_input_copy_component_count: 30,
        evaluator_backend_call_count: 40,
        scratch_reallocation_count: 50,
        ..RuntimeProfile::default()
    }
    .into();

    profile.accumulate(&repeated);

    assert_eq!(profile.orchestration_s, 11.0);
    assert_eq!(profile.stage_leaf_input_pack_s, 22.0);
    assert_eq!(profile.stage_leaf_input_pack_by_stage_s, [22.0, 30.0]);
    assert_eq!(profile.stage_leaf_input_copy_component_count, 33);
    assert_eq!(profile.evaluator_backend_call_count, 44);
    assert_eq!(profile.observed_scratch_reallocation_count, 55);
}
