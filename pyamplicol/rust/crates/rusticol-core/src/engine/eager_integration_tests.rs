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
            coefficient: Complex::new(1.0, 0.0),
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
        Complex::new(1.0, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(3.0, 0.0),
        Complex::new(10.0, 0.0),
        Complex::new(20.0, 0.0),
        Complex::new(30.0, 0.0),
    ];
    let mut amplitudes = [Complex::new(0.0, 0.0); 3];
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
            Complex::new(11.0, 0.0),
            Complex::new(22.0, 0.0),
            Complex::new(33.0, 0.0),
        ]
    );
    assert_eq!(reduced, [121.0, 484.0, 1089.0]);

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
    assert_eq!(runtime.metadata().prepared_backend.as_deref(), Some("jit"));
    let momenta = [
        500.0, 0.0, 0.0, 500.0, 500.0, 0.0, 0.0, -500.0, 1000.0, 0.0, 0.0, 0.0,
    ];
    let values = runtime
        .evaluate_f64(&momenta, 1)
        .expect("evaluate generated eager artifact");
    assert_eq!(values.len(), 1);
    assert!(values[0].is_finite());
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
    let definition = execution
        .plan_definition(&pack)
        .expect("derive eager plan definition from runtime schema and filtered pack");
    let couplings = fs::read(process_root.join(&execution.plan.couplings.path))
        .expect("read eager coupling table");
    assert_eq!(
        couplings.len(),
        execution.plan.couplings.count * EagerCouplingRow::ENCODED_LEN
    );
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
    let mut common = ExecutionRuntime::from_manifest(execution.compiled_metadata_manifest())
        .expect("load shared source and physics execution metadata");
    let point = vec![
        [500.0, 0.0, 0.0, 500.0],
        [500.0, 0.0, 0.0, -500.0],
        [1000.0, 0.0, 0.0, 0.0],
    ];
    let mut eager = EagerNativeRuntime::new(scheduler, backend, "jit".to_string(), false);
    let (values, _) = eager
        .run_f64(&mut common, &[point])
        .expect("execute generated filtered eager artifact");
    assert_eq!(values.len(), 1);
    assert!(values[0].is_finite());
}
