// SPDX-License-Identifier: 0BSD

use super::*;
use crate::{
    EAGER_KERNEL_ABI, EagerClosureRow, EagerCouplingRow, EagerExecutionPlan, EagerExecutionRuntime,
    EagerKernelInput, EagerKernelRole, EagerKernelSpec, EagerPlanDefinition, EagerPlanDimensions,
    EagerPlanPayloads, EagerReductionEntry, EagerReductionGroup, EagerRuntimeOptions, MISSING_U32,
};
use serde_json::json;

const TEST_SYMJIT_APPLICATION_ABI: &str = "symjit-application-storage-v3";

fn symjit_manifest(application_path: &str, exact_state_path: &str, input_len: usize) -> Value {
    json!({
        "kind": "symjit-application-evaluator",
        "runtime_capability": SYMJIT_APPLICATION_RUNTIME_CAPABILITY,
        "backend": "jit",
        "label": "prepared_test_kernel",
        "application_path": application_path,
        "application_abi": TEST_SYMJIT_APPLICATION_ABI,
        "input_len": input_len,
        "output_len": 1,
        "element_layout": "complex-f64",
        "batch_layout": "row-major",
        "compiler_type": "native",
        "translation_mode": "indirect",
        "optimization_level": 3,
        "word_bits": 64,
        "endianness": "little",
        "required_defuns": [],
        "evaluator_state_path": exact_state_path,
        "evaluator_state_runtime_capability": SYMBOLICA_LEGACY_JIT_RUNTIME_CAPABILITY,
        "settings": {"jit_optimization_level": 3},
        "build_timing": {"jit_materialize_s": 0.0},
    })
}

fn compiled_manifest(runtime_capability: &str, exact_state_path: &str, input_len: usize) -> Value {
    json!({
        "kind": "compiled-complex-evaluator",
        "runtime_capability": runtime_capability,
        "backend": "compiled-complex",
        "function_name": "prepared_test_kernel",
        "library_path": "kernels/7/library-0",
        "source_path": "kernels/7/source-0.cpp",
        "input_len": input_len,
        "output_len": 1,
        "number_type": "complex-f64",
        "evaluator_state_path": exact_state_path,
        "settings": {"optimization_level": "o3"},
        "build_timing": {"compile_s": 0.0},
    })
}

fn input(role: &str, component: u32) -> PreparedKernelInputManifest {
    PreparedKernelInputManifest {
        role: role.to_string(),
        component,
        symbol: format!("pyamplicol::test::{role}::{component}"),
        model_parameter_name: None,
        model_parameter_index: None,
    }
}

fn kernel(
    kernel_id: u32,
    contract_kind: &str,
    inputs: Vec<PreparedKernelInputManifest>,
    application_path: &str,
) -> PreparedKernelManifest {
    let input_arity = inputs.len();
    let exact_evaluator_state_path = format!("kernels/{kernel_id}/exact.bin");
    PreparedKernelManifest {
        kernel_id,
        contract_kind: contract_kind.to_string(),
        canonical_signature: format!("test-kernel-{kernel_id}"),
        input_arity,
        output_arity: 1,
        input_layout: (0..input_arity)
            .map(|index| format!("input-{index}"))
            .collect(),
        input_contracts: inputs,
        output_layout: vec!["output-0".to_string()],
        exact_expressions: vec!["test-expression".to_string()],
        f64_evaluator_manifest: symjit_manifest(
            application_path,
            &exact_evaluator_state_path,
            input_arity,
        ),
        exact_evaluator_state_path,
    }
}

fn filtered_pack(kernels: Vec<PreparedKernelManifest>) -> PreparedKernelPackManifest {
    PreparedKernelPackManifest {
        eager_kernel_abi: EAGER_KERNEL_ABI.to_string(),
        backend: "jit".to_string(),
        optimization_settings: json!({"jit_optimization_level": 3}),
        producer: json!({"distribution": "pyamplicol", "version": "test"}),
        dependency_abis: json!({"symjit_application": TEST_SYMJIT_APPLICATION_ABI}),
        provenance: json!({"compiled_model_digest": "test"}),
        target: PreparedKernelTargetManifest {
            portable: true,
            word_bits: 64,
            endianness: "little".to_string(),
            target_triple: "portable-symjit-mir".to_string(),
            cpu_features: Vec::new(),
        },
        resolver_manifest: json!({
            "abi": "pyamplicol-prepared-kernel-catalog-v1",
            "model_name": "test-model",
            "model_parameter_kernel_id": null,
            "propagator_bindings": [
                {"key": {"particle_id": 22, "chirality": 0}, "applies_propagator": false, "kernel_id": null}
            ],
            "vertex_bindings": [],
            "closure_bindings": []
        }),
        kernels,
    }
}

fn compiled_pack(backend: &str, runtime_capability: &str) -> PreparedKernelPackManifest {
    let exact_evaluator_state_path = "kernels/7/exact.bin".to_string();
    let kernel = PreparedKernelManifest {
        kernel_id: 7,
        contract_kind: "vertex".to_string(),
        canonical_signature: format!("test-{backend}-kernel"),
        input_arity: 1,
        output_arity: 1,
        input_layout: vec!["input-0".to_string()],
        input_contracts: vec![input("left-current", 0)],
        output_layout: vec!["output-0".to_string()],
        exact_expressions: vec!["test-expression".to_string()],
        f64_evaluator_manifest: compiled_manifest(
            runtime_capability,
            &exact_evaluator_state_path,
            1,
        ),
        exact_evaluator_state_path,
    };
    let target = crate::runtime_target_info();
    PreparedKernelPackManifest {
        eager_kernel_abi: EAGER_KERNEL_ABI.to_string(),
        backend: backend.to_string(),
        optimization_settings: json!({"optimization_level": "o3"}),
        producer: json!({"distribution": "pyamplicol", "version": "test"}),
        dependency_abis: json!({"compiled_complex": "test"}),
        provenance: json!({"compiled_model_digest": "test"}),
        target: PreparedKernelTargetManifest {
            portable: false,
            word_bits: 64,
            endianness: "little".to_string(),
            target_triple: target.triple,
            cpu_features: target.cpu_features,
        },
        resolver_manifest: json!({
            "abi": "pyamplicol-prepared-kernel-catalog-v1",
            "model_name": "test-model",
            "model_parameter_kernel_id": null,
            "propagator_bindings": [],
            "vertex_bindings": [],
            "closure_bindings": []
        }),
        kernels: vec![kernel],
    }
}

#[test]
fn runtime_options_are_positive_and_bounded() {
    let options = EagerRuntimeOptionsManifest {
        point_tile_size: 1024,
        workspace_mib: 256,
    }
    .validate()
    .expect("valid eager runtime options");
    assert_eq!(options.point_tile_size, 1024);
    assert_eq!(options.workspace_bytes, 256 * 1024 * 1024);

    for invalid in [
        EagerRuntimeOptionsManifest {
            point_tile_size: 0,
            workspace_mib: 256,
        },
        EagerRuntimeOptionsManifest {
            point_tile_size: 1024,
            workspace_mib: 0,
        },
        EagerRuntimeOptionsManifest {
            point_tile_size: MAX_EAGER_POINT_TILE_SIZE + 1,
            workspace_mib: 256,
        },
        EagerRuntimeOptionsManifest {
            point_tile_size: 1024,
            workspace_mib: MAX_EAGER_WORKSPACE_MIB + 1,
        },
    ] {
        assert!(invalid.validate().is_err());
    }
}

#[test]
fn filtered_pack_accepts_only_referenced_kernel_ids_and_preserves_input_order() {
    let pack = filtered_pack(vec![
        kernel(
            7,
            "propagator",
            vec![input("current", 0), input("momentum", 3)],
            "kernels/7/application.symjit",
        ),
        kernel(
            78,
            "vertex",
            vec![
                input("right-current", 1),
                input("left-current", 0),
                input("right-momentum", 2),
                input("coupling-imag", 0),
            ],
            "kernels/78/application.symjit",
        ),
    ]);
    pack.validate().expect("filtered prepared pack");

    let specs = pack.kernel_specs().expect("prepared kernel specs");
    assert_eq!(
        specs.iter().map(|spec| spec.kernel_id).collect::<Vec<_>>(),
        [7, 78]
    );
    assert_eq!(specs[0].role, EagerKernelRole::Finalization);
    assert_eq!(
        specs[1].inputs,
        vec![
            EagerKernelInput::SecondCurrentComponent(1),
            EagerKernelInput::FirstCurrentComponent(0),
            EagerKernelInput::SecondMomentumComponent(2),
            EagerKernelInput::CouplingImag,
        ]
    );
}

#[test]
fn prepared_pack_schema_requires_an_explicit_kernel_abi() {
    let missing = json!({
        "backend": "jit",
        "optimization_settings": {"jit_optimization_level": 3},
        "producer": {"distribution": "pyamplicol"},
        "dependency_abis": {"symjit_application": "test"},
        "provenance": {"compiled_model_digest": "test"},
        "target": {
            "portable": true,
            "word_bits": 64,
            "endianness": "little",
            "target_triple": "portable-symjit-mir",
            "cpu_features": []
        },
        "resolver_manifest": {"abi": "pyamplicol-prepared-kernel-catalog-v1"},
        "kernels": []
    });
    assert!(serde_json::from_value::<PreparedKernelPackManifest>(missing.clone()).is_err());

    let mut unknown = missing;
    unknown["eager_kernel_abi"] = json!(EAGER_KERNEL_ABI);
    unknown["unexpected"] = json!(true);
    assert!(serde_json::from_value::<PreparedKernelPackManifest>(unknown).is_err());
}

#[test]
fn compiled_prepared_backends_validate_runtime_identity_tuples() {
    compiled_pack("asm", SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY)
        .validate()
        .expect("ASM prepared evaluator tuple");
    compiled_pack("cpp", SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY)
        .validate()
        .expect("C++ prepared evaluator tuple");

    let mut wrong_backend = compiled_pack("asm", SYMBOLICA_COMPILED_ASM_RUNTIME_CAPABILITY);
    wrong_backend.kernels[0]
        .f64_evaluator_manifest
        .as_object_mut()
        .expect("compiled evaluator object")
        .insert("backend".to_string(), json!("asm"));
    assert!(wrong_backend.validate().is_err());

    let wrong_capability = compiled_pack("asm", SYMBOLICA_COMPILED_CPP_RUNTIME_CAPABILITY);
    assert!(wrong_capability.validate().is_err());
}

#[cfg(feature = "f64-symjit")]
#[test]
fn prepared_symjit_backend_executes_a_filtered_eager_plan() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use symjit::{Compiler, CompilerType, Config, Storage};

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!(
        "rusticol-eager-backend-{}-{}",
        std::process::id(),
        NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(root.join("kernels/50")).expect("create prepared-kernel test root");

    let mut config = Config::new(CompilerType::Native, 0).expect("native SymJIT config");
    config.set_complex(true);
    config.set_symbolica(true);
    config.set_opt_level(3);
    config.set_simd(true);
    let mut compiler = Compiler::with_config(config);
    let instructions = r#"[[{"Add":[{"Out":0},[{"Param":0},{"Param":1}],0]}],1,[]]"#;
    let application = compiler
        .translate(instructions.to_string(), 2)
        .expect("translate prepared test application");
    let mut application_bytes = Vec::new();
    application
        .save(&mut application_bytes)
        .expect("serialize prepared test application");
    fs::write(
        root.join("kernels/50/application.symjit"),
        application_bytes,
    )
    .expect("write prepared test application");

    let pack = filtered_pack(vec![kernel(
        50,
        "closure",
        vec![input("left-current", 0), input("right-current", 0)],
        "kernels/50/application.symjit",
    )]);
    pack.validate().expect("valid filtered test pack");
    let mut backend = PreparedEvaluatorBackend::load(&pack, &root)
        .expect("load prepared SymJIT evaluator backend");
    let definition = EagerPlanDefinition {
        dimensions: EagerPlanDimensions {
            value_slot_component_counts: vec![1, 1],
            momentum_slot_component_counts: Vec::new(),
            current_component_counts: Vec::new(),
            parameter_count: 0,
            amplitude_count: 1,
        },
        kernels: vec![EagerKernelSpec {
            kernel_id: 50,
            role: EagerKernelRole::Closure,
            inputs: vec![
                EagerKernelInput::FirstCurrentComponent(0),
                EagerKernelInput::SecondCurrentComponent(0),
            ],
            output_component_count: 1,
        }],
        direct_closures: Vec::new(),
        reduction_groups: vec![EagerReductionGroup {
            amplitude_indices: vec![0],
        }],
        reduction_entries: vec![EagerReductionEntry {
            left_group_index: 0,
            right_group_index: 0,
            coefficient: crate::EagerComplex64::new(1.0, 0.0),
        }],
    };
    let coupling_bytes = EagerCouplingRow::encode_table(&[EagerCouplingRow {
        real_parameter_id: MISSING_U32,
        imag_parameter_id: MISSING_U32,
        constant_real: 1.0,
        constant_imag: 0.0,
    }])
    .expect("encode eager coupling table");
    let closure_bytes = EagerClosureRow::encode_table(&[EagerClosureRow {
        kernel_id: 50,
        left_value_slot_id: 0,
        right_value_slot_id: 1,
        amplitude_index: 0,
        coupling_slot_id: 0,
        factor_real: 1.0,
        factor_imag: 0.0,
    }])
    .expect("encode eager closure table");
    let plan = EagerExecutionPlan::from_payloads(
        definition,
        EagerPlanPayloads {
            couplings: &coupling_bytes,
            stages: &[],
            closures: &closure_bytes,
        },
    )
    .expect("build eager test plan");
    let mut runtime = EagerExecutionRuntime::new(
        plan,
        EagerRuntimeOptions {
            point_tile_size: 2,
            workspace_bytes: 4096,
        },
    )
    .expect("build eager test runtime");
    let values = [
        crate::EagerComplex64::new(1.0, 0.0),
        crate::EagerComplex64::new(2.0, 0.0),
        crate::EagerComplex64::new(3.0, 0.0),
        crate::EagerComplex64::new(10.0, 0.0),
        crate::EagerComplex64::new(20.0, 0.0),
        crate::EagerComplex64::new(30.0, 0.0),
    ];
    let mut amplitudes = [crate::EagerComplex64::new(0.0, 0.0); 3];
    let mut reduced = [0.0; 3];
    runtime
        .evaluate_into(
            &mut backend,
            3,
            &values,
            &[],
            &[],
            &mut amplitudes,
            &mut reduced,
        )
        .expect("evaluate prepared eager plan");
    assert_eq!(
        amplitudes,
        [
            crate::EagerComplex64::new(11.0, 0.0),
            crate::EagerComplex64::new(22.0, 0.0),
            crate::EagerComplex64::new(33.0, 0.0),
        ]
    );
    assert_eq!(reduced, [121.0, 484.0, 1089.0]);

    amplitudes.fill(crate::EagerComplex64::new(0.0, 0.0));
    reduced.fill(0.0);
    let profile = runtime
        .evaluate_profile_into(
            &mut backend,
            3,
            &values,
            &[],
            &[],
            &mut amplitudes,
            &mut reduced,
        )
        .expect("profile prepared eager plan");
    assert_eq!(
        amplitudes,
        [
            crate::EagerComplex64::new(11.0, 0.0),
            crate::EagerComplex64::new(22.0, 0.0),
            crate::EagerComplex64::new(33.0, 0.0),
        ]
    );
    assert_eq!(reduced, [121.0, 484.0, 1089.0]);
    assert!(profile.total_s > 0.0);
    assert!(profile.initialize_s > 0.0);
    assert!(profile.kernel_call_s > 0.0);
    assert!(profile.closure_s > 0.0);
    assert!(profile.reduction_s > 0.0);
    assert!(profile.copy_out_s > 0.0);
    assert!(profile.accounted_s() <= profile.total_s);

    let _ = fs::remove_dir_all(root);
}

#[cfg(feature = "f64-symjit")]
#[test]
fn generated_eager_artifact_loads_when_fixture_is_supplied() {
    let Some(root) = std::env::var_os("RUSTICOL_EAGER_ARTIFACT") else {
        return;
    };
    let mut runtime = NativeRuntime::load(PathBuf::from(root), None, None)
        .expect("load generated eager artifact through NativeRuntime");
    assert_eq!(runtime.metadata().execution_mode, "eager");
    assert!(matches!(
        runtime.metadata().prepared_backend.as_deref(),
        Some("jit" | "asm" | "cpp")
    ));
    let validation_path = runtime
        .root()
        .join("processes")
        .join(&runtime.metadata().representative_process_key)
        .join("validation-momenta.json");
    let validation: Value =
        serde_json::from_slice(&fs::read(&validation_path).expect("read eager validation momenta"))
            .expect("parse eager validation momenta");
    let momenta = validation["points"][0]
        .as_array()
        .expect("one eager validation point")
        .iter()
        .flat_map(|leg| {
            leg["momentum"]
                .as_array()
                .expect("four momentum components")
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .expect("decimal momentum string")
                        .parse::<f64>()
                        .expect("f64 validation momentum")
                })
        })
        .collect::<Vec<_>>();
    let values = runtime
        .evaluate_f64(&momenta, 1)
        .expect("evaluate generated eager artifact");
    assert_eq!(values.len(), 1);
    assert!(values[0].is_finite());

    let resolved = runtime
        .evaluate_resolved_f64(&momenta, 1, None, None)
        .expect("resolve generated eager artifact");
    let resolved_total = resolved.totals()[0];
    assert!((resolved_total - values[0]).abs() <= 1.0e-12 * values[0].abs().max(1.0));

    let selected_helicity = runtime
        .helicities()
        .expect("eager helicity metadata")
        .into_iter()
        .find(|helicity| helicity.computed)
        .expect("one computed helicity")
        .id;
    let selected_color = runtime
        .color_ids()
        .expect("eager color metadata")
        .into_iter()
        .next()
        .expect("one color component");
    let selected_colors =
        (runtime.metadata().color_accuracy == "lc").then(|| std::slice::from_ref(&selected_color));
    let selected = runtime
        .evaluate_resolved_f64(
            &momenta,
            1,
            Some(std::slice::from_ref(&selected_helicity)),
            selected_colors,
        )
        .expect("select eager resolved component");
    assert_eq!(selected.shape(), (1, 1, 1));
    assert!(selected.values[0].is_finite());

    if let Some(compiled_root) = std::env::var_os("RUSTICOL_COMPILED_ARTIFACT") {
        let mut compiled = NativeRuntime::load(PathBuf::from(compiled_root), None, None)
            .expect("load matching compiled artifact");
        let compiled_values = compiled
            .evaluate_f64(&momenta, 1)
            .expect("evaluate matching compiled artifact");
        assert_close_f64(values[0], compiled_values[0], "eager/compiled total");
        let compiled_resolved = compiled
            .evaluate_resolved_f64(&momenta, 1, None, None)
            .expect("resolve matching compiled artifact");
        assert_eq!(resolved.helicity_ids, compiled_resolved.helicity_ids);
        assert_eq!(resolved.color_ids, compiled_resolved.color_ids);
        assert_eq!(resolved.values.len(), compiled_resolved.values.len());
        for (eager, compiled) in resolved.values.iter().zip(&compiled_resolved.values) {
            assert_close_f64(*eager, *compiled, "eager/compiled resolved component");
        }
    }

    let parameters = runtime.model_parameters().expect("eager model parameters");
    let candidates = if let Some(parameter) = parameters
        .iter()
        .find(|parameter| parameter.name == "aEWM1")
    {
        vec![(
            BTreeMap::from([(parameter.name.clone(), (parameter.default * 1.05, 0.0))]),
            BTreeMap::from([(
                parameter.name.clone(),
                (parameter.default, parameter.default_imaginary),
            )]),
        )]
    } else {
        let mut groups = BTreeMap::<String, Vec<(String, f64, f64)>>::new();
        for parameter in parameters
            .iter()
            .filter(|parameter| parameter.mutable && parameter.name.starts_with("coupling."))
        {
            let prefix = parameter
                .name
                .split_once(".component_")
                .map(|(prefix, _)| prefix.to_string())
                .expect("coupling component suffix");
            groups.entry(prefix).or_default().push((
                parameter.name.clone(),
                parameter.default,
                parameter.default_imaginary,
            ));
        }
        groups
            .into_values()
            .map(|selected| {
                let changed = selected
                    .iter()
                    .map(|(name, _, _)| (name.clone(), (0.0, 0.0)))
                    .collect();
                let restored = selected
                    .iter()
                    .map(|(name, real, imaginary)| (name.clone(), (*real, *imaginary)))
                    .collect();
                (changed, restored)
            })
            .collect::<Vec<_>>()
    };
    assert!(!candidates.is_empty(), "one mutable eager parameter group");
    let mut observed_parameter_effect = false;
    for (changed, restored) in candidates {
        runtime
            .set_model_parameters(&changed)
            .expect("update eager model parameters atomically");
        let changed_value = runtime
            .evaluate_f64(&momenta, 1)
            .expect("evaluate updated eager parameters")[0];
        runtime
            .set_model_parameters(&restored)
            .expect("restore eager model parameters atomically");
        let restored_value = runtime
            .evaluate_f64(&momenta, 1)
            .expect("evaluate restored eager parameters")[0];
        assert_eq!(restored_value.to_bits(), values[0].to_bits());
        if changed_value.to_bits() != values[0].to_bits() {
            observed_parameter_effect = true;
            break;
        }
    }
    assert!(
        observed_parameter_effect,
        "one eager parameter affects output"
    );

    if parameters.iter().any(|parameter| parameter.name == "aEWM1")
        && parameters.iter().any(|parameter| parameter.name == "MZ")
    {
        let before_failed_derivation = runtime
            .exact_runtime_state_json()
            .expect("eager state before failed derivation");
        assert!(runtime.set_model_parameter("MZ", 0.0, 0.0).is_err());
        assert_eq!(
            runtime
                .exact_runtime_state_json()
                .expect("eager state after failed derivation"),
            before_failed_derivation
        );
    }

    let before_failed_update = runtime
        .exact_runtime_state_json()
        .expect("eager parameter state before failed update");
    assert!(
        runtime
            .set_model_parameter("not-a-model-parameter", 1.0, 0.0)
            .is_err()
    );
    assert_eq!(
        runtime
            .exact_runtime_state_json()
            .expect("eager parameter state after failed update"),
        before_failed_update
    );
}

fn assert_close_f64(left: f64, right: f64, context: &str) {
    let tolerance = 1.0e-15 + 1.0e-12 * left.abs().max(right.abs());
    assert!(
        (left - right).abs() <= tolerance,
        "{context}: {left:.17e} != {right:.17e} (tolerance {tolerance:.3e})"
    );
}

#[cfg(feature = "f64-symjit")]
#[test]
fn generated_filtered_pack_and_binary_plan_execute_when_fixture_is_supplied() {
    use crate::{
        EagerAttachmentRow, EagerCouplingRow, EagerFinalizationRow, EagerInvocationRow,
        EagerStagePayload,
    };

    let Some(root) = std::env::var_os("RUSTICOL_EAGER_ARTIFACT") else {
        return;
    };
    let root = PathBuf::from(root);
    let process_root = root.join("processes/d_dbar_to_z");
    let execution: EagerExecutionManifest = serde_json::from_slice(
        &fs::read(process_root.join("execution.json")).expect("read eager execution fixture"),
    )
    .expect("parse eager execution fixture");
    execution
        .validate_header()
        .expect("validate eager execution header");
    let pack: PreparedKernelPackManifest = serde_json::from_slice(
        &fs::read(root.join("model/eager-kernel-pack.json"))
            .expect("read filtered prepared-kernel fixture"),
    )
    .expect("parse filtered prepared-kernel fixture");
    pack.validate()
        .expect("validate filtered prepared-kernel fixture");
    let coupling_bytes = fs::read(process_root.join(&execution.plan.couplings.path))
        .expect("read eager coupling table");
    let closures = fs::read(process_root.join(&execution.plan.closures.path))
        .expect("read eager closure table");
    assert_eq!(
        closures.len(),
        execution.plan.closures.count * EagerClosureRow::ENCODED_LEN
    );
    let stage_bytes = execution
        .plan
        .stages
        .iter()
        .map(|stage| {
            let invocations = fs::read(process_root.join(&stage.invocations.path))
                .expect("read eager invocation table");
            let attachments = fs::read(process_root.join(&stage.attachments.path))
                .expect("read eager attachment table");
            let finalizations = fs::read(process_root.join(&stage.finalizations.path))
                .expect("read eager finalization table");
            assert_eq!(
                invocations.len(),
                stage.invocations.count * EagerInvocationRow::ENCODED_LEN
            );
            assert_eq!(
                attachments.len(),
                stage.attachments.count * EagerAttachmentRow::ENCODED_LEN
            );
            assert_eq!(
                finalizations.len(),
                stage.finalizations.count * EagerFinalizationRow::ENCODED_LEN
            );
            (invocations, attachments, finalizations)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        coupling_bytes.len(),
        execution.plan.couplings.count * EagerCouplingRow::ENCODED_LEN
    );
    let mut common = ExecutionRuntime::from_manifest(execution.compiled_metadata_manifest())
        .expect("load shared source and physics execution metadata");
    let (parameter_projection, couplings, model_parameter_evaluator) =
        prepare_eager_parameter_state(
            &pack,
            &execution.runtime_schema.model_parameters,
            &coupling_bytes,
            &root.join("model/eager-kernels"),
        )
        .expect("prepare eager model-parameter projection");
    common.model_parameter_evaluator = model_parameter_evaluator;
    common
        .refresh_derived_model_parameters()
        .expect("refresh prepared derived parameters");
    let definition = execution
        .plan_definition(
            &pack,
            u32::try_from(parameter_projection.parameter_count)
                .expect("prepared parameter count fits u32"),
        )
        .expect("derive eager plan definition from runtime schema and filtered pack");
    let stages = execution
        .plan
        .stages
        .iter()
        .zip(&stage_bytes)
        .map(|(stage, bytes)| EagerStagePayload {
            stage_index: stage.stage_index,
            invocations: &bytes.0,
            attachments: &bytes.1,
            finalizations: &bytes.2,
        })
        .collect::<Vec<_>>();
    let plan = EagerExecutionPlan::from_payloads(
        definition,
        EagerPlanPayloads {
            couplings: &couplings,
            stages: &stages,
            closures: &closures,
        },
    )
    .expect("load generated eager binary plan");
    let scheduler = EagerExecutionRuntime::new(
        plan,
        execution
            .runtime_options
            .validate()
            .expect("runtime options"),
    )
    .expect("construct generated eager scheduler");
    let backend = PreparedEvaluatorBackend::load(&pack, &root.join("model/eager-kernels"))
        .expect("load filtered prepared evaluator pack");
    let point = vec![
        [500.0, 0.0, 0.0, 500.0],
        [500.0, 0.0, 0.0, -500.0],
        [1000.0, 0.0, 0.0, 0.0],
    ];
    let (raw_sum_groups, color_contraction) = execution
        .raw_reduction_runtime()
        .expect("load eager resolved reduction metadata");
    let mut eager = EagerNativeRuntime::new(
        scheduler,
        backend,
        "jit".to_string(),
        parameter_projection,
        raw_sum_groups,
        color_contraction,
    );
    let (values, _) = eager
        .run_f64(&mut common, &[point])
        .expect("execute generated filtered eager artifact");
    assert_eq!(values.len(), 1);
    assert!(values[0].is_finite());
}
