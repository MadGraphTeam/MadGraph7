// SPDX-License-Identifier: 0BSD

use super::*;

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
    let plan = physics
        .lc_resolved_replay_plan(&mappings, &[2.0, 1.0])
        .unwrap();
    assert_eq!(plan.helicity_count, 2);
    assert_eq!(plan.color_count, 3);

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
        .lc_resolved_replay_plan(&vec![Vec::new()], &[2.0])
        .unwrap_err();

    assert!(error.to_string().contains("missing replayed LC flow word"));
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
    let alias_manifest = apply_final_state_alias_metadata(representative_manifest, &alias).unwrap();

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
    execution.physics = Some(alias_physics);
    let runtime = NativeRuntime {
        root: PathBuf::new(),
        runtime: execution,
        process: alias.expression,
        process_key: alias.id,
        input_crossing_map: Some(crossing_map),
        final_state_permutation_alias_of: Some("representative".to_string()),
        physics_v1: alias_manifest,
        warnings_muted: false,
        warned_kinds: BTreeSet::new(),
        pending_warnings: Vec::new(),
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

    let contraction = || ColorContractionRuntime {
        group_count: 1,
        entries: vec![ColorContractionEntry {
            left_group_index: 0,
            right_group_index: 0,
            weight_re: 2.0,
            weight_im: 0.0,
            symmetry_factor: 1.0,
        }],
        group_scratch_f64: Vec::new(),
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
        output_len: 0,
        chunk_scratch_f64: Vec::new(),
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
        output_scratch_f64: outputs,
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
        let contraction = ColorContractionRuntime {
            group_count: 1,
            entries: vec![ColorContractionEntry {
                left_group_index: 0,
                right_group_index: 0,
                weight_re: 2.0,
                weight_im: 0.0,
                symmetry_factor: 1.0,
            }],
            group_scratch_f64: Vec::new(),
        };
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

#[test]
fn contracted_color_coverage_does_not_warn_as_incomplete() {
    for color_accuracy in ["nlc", "full"] {
        let physics = test_physics_runtime(color_accuracy);
        let physics_v1 = physics.manifest.clone();
        let mut execution = empty_generic_runtime();
        execution.color_accuracy = color_accuracy.to_string();
        execution.physics = Some(physics);
        let mut runtime = NativeRuntime {
            root: PathBuf::new(),
            runtime: execution,
            process: "x x > y".to_string(),
            process_key: "x_x_to_y".to_string(),
            input_crossing_map: None,
            final_state_permutation_alias_of: None,
            physics_v1,
            warnings_muted: false,
            warned_kinds: BTreeSet::new(),
            pending_warnings: Vec::new(),
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
        lc_topology_replay_mappings: Vec::new(),
        lc_topology_replay_public_mappings: Vec::new(),
        lc_topology_replay_weights: Vec::new(),
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
        physics: None,
        stages: None,
        amplitude_stage: None,
        state_scratch_f64: Vec::new(),
        values_scratch_f64: Vec::new(),
    }
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
