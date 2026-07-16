// SPDX-License-Identifier: 0BSD

use super::*;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static TEST_ARTIFACT_COUNTER: AtomicU64 = AtomicU64::new(0);

#[test]
fn distribution_version_normalization_is_narrow() {
    assert_eq!(
        canonical_distribution_version("0.1.0-dev.0+candidate.0123"),
        "0.1.0.dev0+candidate.0123"
    );
    assert_eq!(canonical_distribution_version("0.1.0"), "0.1.0");
    assert_ne!(
        canonical_distribution_version("0.1.0-dev.0+candidate.0123"),
        canonical_distribution_version("0.1.0-dev.0+candidate.4567")
    );
}

struct TestArtifact {
    root: PathBuf,
    manifest: Value,
}

impl TestArtifact {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos();
        let temp_root = std::env::temp_dir()
            .canonicalize()
            .expect("canonical temporary directory");
        let root = temp_root.join(format!(
            "pyamplicol-rusticol-manifest-{}-{nonce}-{}",
            std::process::id(),
            TEST_ARTIFACT_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).expect("create test artifact");
        let files = [
            ("requested.toml", b"action = \"generate\"\n".as_slice()),
            ("effective.toml", b"action = \"generate\"\n".as_slice()),
            ("evaluator.json", b"{}".as_slice()),
            ("physics.json", b"{}".as_slice()),
        ];
        let mut payloads = Vec::new();
        for (path, bytes) in files {
            fs::write(root.join(path), bytes).expect("write payload");
            let role = match path {
                "requested.toml" => "configuration-requested",
                "effective.toml" => "configuration-effective",
                "evaluator.json" => "evaluator-manifest",
                "physics.json" => "runtime-physics",
                _ => unreachable!(),
            };
            let mut payload = json!({
                "path": path,
                "role": role,
                "media_type": "application/octet-stream",
                "size_bytes": bytes.len(),
                "sha256": sha256(bytes),
                "executable": false,
            });
            if path == "physics.json" {
                payload["process_id"] = json!("p0");
            }
            payloads.push(payload);
        }
        let manifest = json!({
            "schema_version": 3,
            "kind": "pyamplicol-process",
            "artifact_id": "0".repeat(64),
            "created_utc": "2026-07-15T00:00:00Z",
            "producer": {
                "distribution": "pyamplicol",
                "version": env!("CARGO_PKG_VERSION"),
                "versions": {
                    "python_api": PYTHON_API_VERSION,
                    "toml": TOML_SCHEMA_VERSION,
                    "compiled_model": COMPILED_MODEL_SCHEMA_VERSION,
                    "process_artifact": PROCESS_ARTIFACT_SCHEMA_VERSION,
                    "runtime_physics": RUNTIME_PHYSICS_SCHEMA_VERSION,
                    "symbolica_serialization": crate::SYMBOLICA_SERIALIZATION_ABI,
                    "c_abi": C_ABI_VERSION,
                },
                "target": {"triple": current_target_triple(), "cpu_features": []},
            },
            "model": {
                "name": "test-model",
                "source_kind": "compiled-model",
                "content_sha256": "1".repeat(64),
                "compiled_schema_version": COMPILED_MODEL_SCHEMA_VERSION,
            },
            "configuration": {
                "toml_schema_version": TOML_SCHEMA_VERSION,
                "requested_path": "requested.toml",
                "effective_path": "effective.toml",
                "adjustments": [],
            },
            "processes": [{
                "id": "p0",
                "expression": "a b > c",
                "color_accuracy": "lc",
                "external_pdgs": [1, -1, 22],
                "physics_path": "physics.json",
                "required_runtime_capabilities": [
                    "symjit.application.complex-f64.v1"
                ],
                "aliases": [],
            }],
            "default_process_id": "p0",
            "runtime": {
                "engine": "rusticol",
                "engine_version": env!("CARGO_PKG_VERSION"),
                "evaluator_manifest_path": "evaluator.json",
                "required_runtime_capabilities": [
                    "symjit.application.complex-f64.v1"
                ],
                "api_bundle_path": null,
            },
            "payloads": payloads,
            "dependencies": [],
        });
        let mut artifact = Self { root, manifest };
        artifact.write_manifest();
        artifact
    }

    fn write_manifest(&mut self) {
        if self.manifest["schema_version"] == json!(PROCESS_ARTIFACT_SCHEMA_VERSION) {
            self.manifest["artifact_id"] =
                json!(compute_artifact_id(&self.manifest).expect("compute artifact identity"));
        }
        let mut bytes = String::new();
        write_python_canonical_json(&self.manifest, &mut bytes)
            .expect("serialize canonical manifest");
        bytes.push('\n');
        fs::write(self.root.join(ARTIFACT_MANIFEST_FILE), bytes).expect("write manifest");
    }
}

impl Drop for TestArtifact {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(feature = "f64-symjit")]
fn add_test_payload(
    artifact: &mut TestArtifact,
    path: &str,
    role: &str,
    bytes: &[u8],
    process_id: Option<&str>,
    targeted: bool,
) {
    let output = artifact.root.join(path);
    fs::create_dir_all(output.parent().expect("payload parent")).expect("create payload parent");
    fs::write(&output, bytes).expect("write mixed-runtime payload");
    let mut payload = json!({
        "path": path,
        "role": role,
        "media_type": "application/octet-stream",
        "size_bytes": bytes.len(),
        "sha256": sha256(bytes),
        "executable": false,
    });
    if let Some(process_id) = process_id {
        payload["process_id"] = json!(process_id);
    }
    if targeted {
        payload["target"] = json!({
            "triple": current_target_triple(),
            "cpu_features": [],
        });
    }
    artifact.manifest["payloads"]
        .as_array_mut()
        .expect("payload array")
        .push(payload);
}

#[cfg(feature = "f64-symjit")]
fn direct_symjit_application_bytes(input_len: usize) -> Vec<u8> {
    use symjit::{Compiler, CompilerType, Config, Storage};

    let mut config = Config::new(CompilerType::Native, 0).expect("native SymJIT config");
    config.set_complex(true);
    config.set_symbolica(true);
    config.set_opt_level(3);
    config.set_simd(true);
    let mut compiler = Compiler::with_config(config);
    let instructions = r#"[[{"Add":[{"Out":0},[{"Param":0},{"Param":1}],0]}],1,[]]"#;
    let application = compiler
        .translate(instructions.to_string(), input_len)
        .expect("translate direct SymJIT test application");
    let mut bytes = Vec::new();
    application
        .save(&mut bytes)
        .expect("serialize direct SymJIT test application");
    bytes
}

#[cfg(feature = "f64-symjit")]
fn minimal_runtime_physics(process_id: &str, process: &str, final_particle: &str) -> Value {
    json!({
        "schema_version": RUNTIME_PHYSICS_SCHEMA_VERSION,
        "kind": "pyamplicol-resolved-physics",
        "process_id": process_id,
        "process": process,
        "color_accuracy": "lc",
        "coverage": {
            "helicities": "complete",
            "color": "complete",
            "color_kind": "physical-lc-flows",
            "structural_zero_helicity_count": 0,
        },
        "external_particles": [
            {
                "index": 0,
                "label": 1,
                "particle": "a",
                "pdg": 1,
                "role": "initial",
                "momentum_slot": 0,
                "momentum_components": ["E", "px", "py", "pz"],
            },
            {
                "index": 1,
                "label": 2,
                "particle": "b",
                "pdg": -1,
                "role": "initial",
                "momentum_slot": 1,
                "momentum_components": ["E", "px", "py", "pz"],
            },
            {
                "index": 2,
                "label": 3,
                "particle": final_particle,
                "pdg": 22,
                "role": "final",
                "momentum_slot": 2,
                "momentum_components": ["E", "px", "py", "pz"],
            },
        ],
        "helicities": [{
            "id": "h:+0,+0,+0",
            "index": 0,
            "values": [0, 0, 0],
            "computed": true,
            "structural_zero": false,
            "representative_id": "h:+0,+0,+0",
            "coefficient": 1.0,
        }],
        "color_components": [{
            "kind": "lc-flow",
            "id": "flow:singlet",
            "index": 0,
            "word": [],
            "computed": true,
            "representative_id": "flow:singlet",
            "coefficient": 1.0,
        }],
        "reduction": {
            "kind": "lc-diagonal",
            "groups": [{
                "id": "group:7",
                "representative_helicity_id": "h:+0,+0,+0",
                "representative_color_id": "flow:singlet",
                "physical_helicity_ids": ["h:+0,+0,+0"],
                "physical_color_ids": ["flow:singlet"],
            }],
        },
        "model_parameters": [],
        "selectors": {
            "helicity": true,
            "color_flow": true,
            "contracted_color": false,
        },
        "extensions": {},
    })
}

#[cfg(feature = "f64-symjit")]
fn direct_evaluator_manifest(application_path: &str) -> Value {
    json!({
        "kind": "symjit-application-evaluator",
        "runtime_capability": "symjit.application.complex-f64.v1",
        "application_path": application_path,
        "application_abi": crate::engine::SYMJIT_APPLICATION_STORAGE_ABI,
        "input_len": 14,
        "output_len": 1,
        "element_layout": "complex-f64",
        "batch_layout": "row-major",
        "compiler_type": "native",
        "translation_mode": "indirect",
        "optimization_level": 3,
        "word_bits": 64,
        "endianness": "little",
        "required_defuns": [],
        "evaluator_state_path": null,
        "evaluator_state_runtime_capability": null,
    })
}

#[cfg(feature = "f64-symjit")]
fn compiled_evaluator_manifest(library_path: &str, runtime_capability: &str) -> Value {
    json!({
        "kind": "compiled-complex-evaluator",
        "runtime_capability": runtime_capability,
        "function_name": "rusticol_test_evaluator",
        "input_len": 14,
        "output_len": 1,
        "library_path": library_path,
        "evaluator_state_path": null,
        "number_type": "complex-f64",
    })
}

#[cfg(feature = "f64-symjit")]
fn minimal_execution_manifest(
    process_id: &str,
    process: &str,
    capability: &str,
    evaluator: Value,
) -> Value {
    let value_slot = |value_slot_id: usize, current_id: usize, component: usize| {
        json!({
            "value_slot_id": value_slot_id,
            "current_id": current_id,
            "variant": "source",
            "component_start": component,
            "component_stop": component + 1,
            "dimension": 1,
        })
    };
    let slot = |current_id: usize, component: usize| {
        json!({
            "current_id": current_id,
            "component_start": component,
            "component_stop": component + 1,
            "dimension": 1,
        })
    };
    let crossing_ir = || {
        json!({
            "momentum_transform": "negate-four-momentum",
            "helicity_factor": 1,
            "chirality_factor": 1,
            "spin_state_factor": 1,
            "phase": [1.0, 0.0],
        })
    };
    let source_ir = |particle_id: i32, anti_particle_id: i32, orientation: &str| {
        json!({
            "identity": {
                "canonical_id": format!("model:minimal:state:{particle_id}"),
                "species_id": "model:minimal:species:1",
                "anti_canonical_id": format!("model:minimal:state:{anti_particle_id}"),
                "display_name": format!("state_{particle_id}"),
                "anti_display_name": format!("state_{anti_particle_id}"),
                "pdg_label": particle_id,
                "anti_pdg_label": anti_particle_id,
                "orientation": orientation,
                "self_conjugate": false,
            },
            "statistics": "boson",
            "wavefunction_family": "scalar",
            "component_dimension": 1,
            "states": [{"helicity": 0, "chirality": 0, "spin_state": 1}],
            "crossing": crossing_ir(),
            "basis": "scalar",
            "mass_parameter": null,
            "width_parameter": null,
        })
    };
    let propagator_ir = |particle_id: i32, anti_particle_id: i32, orientation: &str| {
        json!({
            "identity": {
                "canonical_id": format!("model:minimal:state:{particle_id}"),
                "species_id": "model:minimal:species:1",
                "anti_canonical_id": format!(
                    "model:minimal:state:{anti_particle_id}"
                ),
                "display_name": format!("state_{particle_id}"),
                "anti_display_name": format!("state_{anti_particle_id}"),
                "pdg_label": particle_id,
                "anti_pdg_label": anti_particle_id,
                "orientation": orientation,
                "self_conjugate": false,
            },
            "particle_id": particle_id,
            "chirality": 0,
            "kind": "scalar",
            "backend": "test",
            "basis": "scalar",
            "applies_propagator": true,
            "kernel": "test_scalar",
            "full_tensor_network_ready": true,
            "mass_class": "massless",
            "gauge": null,
            "numerator": "i",
            "denominator": "momentum_squared-mass_squared+i*mass*width",
            "mass_parameter": null,
            "width_parameter": null,
            "custom_source": null,
            "auxiliary_policy": null,
            "goldstone_policy": "not-applicable",
            "description": "test scalar propagator",
        })
    };
    let real_inputs = (2..14).collect::<Vec<_>>();
    let current_storage = json!({
        "component_count": 2,
        "number_type": "complex",
        "metadata_compacted": true,
        "current_slots": [
            {
                "current_id": 0, "component_start": 0, "component_stop": 1,
                "dimension": 1, "is_source": true, "particle_id": 1,
                "external_mask": 1, "external_labels": [], "helicity_ancestry": null,
                "chirality": 0, "spin_state": null, "flavour_flow": [],
                "color_state": null, "momentum_mask": 1,
                "auxiliary_kind": null
            },
            {
                "current_id": 1, "component_start": 1, "component_stop": 2,
                "dimension": 1, "is_source": true, "particle_id": -1,
                "external_mask": 2, "external_labels": [], "helicity_ancestry": null,
                "chirality": 0, "spin_state": null, "flavour_flow": [],
                "color_state": null, "momentum_mask": 2,
                "auxiliary_kind": null
            }
        ],
    });
    let value_storage = json!({
        "component_count": 2,
        "number_type": "complex",
        "metadata_compacted": true,
        "value_slots": [
            {
                "value_slot_id": 0, "current_id": 0, "variant": "source",
                "component_start": 0, "component_stop": 1, "dimension": 1,
                "current_component_start": 0, "current_component_stop": 1,
                "is_source": true, "applies_propagator": false, "particle_id": 1,
                "external_mask": 1, "external_labels": [], "momentum_mask": 1,
                "chirality": 0,
                "propagator": propagator_ir(1, -1, "particle")
            },
            {
                "value_slot_id": 1, "current_id": 1, "variant": "source",
                "component_start": 1, "component_stop": 2, "dimension": 1,
                "current_component_start": 1, "current_component_stop": 2,
                "is_source": true, "applies_propagator": false, "particle_id": -1,
                "external_mask": 2, "external_labels": [], "momentum_mask": 2,
                "chirality": 0,
                "propagator": propagator_ir(-1, 1, "antiparticle")
            }
        ],
    });
    let source_fill = json!({
        "source_count": 2,
        "sources": [
            {
                "source_id": 0, "current_id": 0,
                "current_component_start": 0, "current_component_stop": 1,
                "value_slot": value_slot(0, 0, 0),
                "source_parameter_start": 0, "source_parameter_stop": 1,
                "leg_label": 1, "input_momentum_slot": 0, "side": "initial",
                "crossing": "negate-incoming-momentum", "physical_pdg": 1, "outgoing_pdg": 1,
                "particle_id": 1, "anti_particle_id": -1,
                "source_kind": "external-wavefunction",
                "wavefunction_kind": "scalar", "source_orientation": "particle",
                "source_basis": "scalar", "source_ir": source_ir(1, -1, "particle"),
                "applied_crossing": crossing_ir(),
                "source_helicity": 0,
                "chirality": 0, "spin_state": 1, "dimension": 1,
                "helicity_ancestry": 1, "color_state": {}
            },
            {
                "source_id": 1, "current_id": 1,
                "current_component_start": 1, "current_component_stop": 2,
                "value_slot": value_slot(1, 1, 1),
                "source_parameter_start": 1, "source_parameter_stop": 2,
                "leg_label": 2, "input_momentum_slot": 1, "side": "initial",
                "crossing": "negate-incoming-momentum", "physical_pdg": -1, "outgoing_pdg": -1,
                "particle_id": -1, "anti_particle_id": 1,
                "source_kind": "external-wavefunction",
                "wavefunction_kind": "scalar", "source_orientation": "antiparticle",
                "source_basis": "scalar", "source_ir": source_ir(-1, 1, "antiparticle"),
                "applied_crossing": crossing_ir(),
                "source_helicity": 0,
                "chirality": 0, "spin_state": 1, "dimension": 1,
                "helicity_ancestry": 1, "color_state": {}
            }
        ],
    });
    let amplitude_stage = json!({
        "stage_kind": "amplitude-roots",
        "output_count": 1,
        "color_contraction": null,
        "roots": [{
            "output_index": 0, "root_id": 0, "kind": "direct-contraction",
            "left_current_id": 0, "right_current_id": 1,
            "left_slot": slot(0, 0), "right_slot": slot(1, 1),
            "left_value_slot": value_slot(0, 0, 0),
            "right_value_slot": value_slot(1, 1, 1),
            "vertex_kind": null, "vertex_particles": null,
            "coupling": [1.0, 0.0], "color_weight": [1.0, 0.0],
            "color_sector_id": 0, "contraction": "minimal-test",
            "contraction_ir": {
                "name": "minimal-test",
                "left_basis": "scalar",
                "right_basis": "scalar",
                "coefficients": [[1.0, 0.0]],
                "chirality_relation": "any",
                "metric_signature": null
            },
            "coherent_group_id": 7, "helicity_weight": 1.0,
            "all_sector_weight": 1.0
        }],
    });
    let runtime_schema = json!({
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": "pyamplicol-runtime-execution-plan",
        "process_key": process_id,
        "process": process,
        "external_particles": [
            {"label": 1, "index": 0, "pdg": 1, "outgoing_pdg": 1, "role": "initial", "momentum_slot": 0},
            {"label": 2, "index": 1, "pdg": -1, "outgoing_pdg": -1, "role": "initial", "momentum_slot": 1},
            {"label": 3, "index": 2, "pdg": 22, "outgoing_pdg": 22, "role": "final", "momentum_slot": 2},
        ],
        "model": null,
        "model_parameters": [],
        "normalization": null,
        "parameter_layout": {
            "source_component_parameter_count": 2,
            "momentum_parameter_count": 12,
            "model_parameter_count": 0,
            "parameter_count_if_flattened": 14,
            "value_component_count": 2,
            "source_components_complex": true,
            "momentum_components_real": true,
            "real_valued_inputs": real_inputs,
        },
        "current_storage": current_storage,
        "value_storage": value_storage,
        "source_fill": source_fill,
        "momentum_slots": [
            {"momentum_slot_id": 0, "momentum_mask": 1, "external_labels": [1], "component_start": 0, "component_stop": 4, "real_valued": true},
            {"momentum_slot_id": 1, "momentum_mask": 2, "external_labels": [2], "component_start": 4, "component_stop": 8, "real_valued": true},
            {"momentum_slot_id": 2, "momentum_mask": 4, "external_labels": [3], "component_start": 8, "component_stop": 12, "real_valued": true},
        ],
        "stages": [],
        "amplitude_stage": amplitude_stage,
    });
    let serialized_amplitude = json!({
        "stage_index": 0,
        "stage_kind": "amplitude-roots",
        "subset_size": null,
        "evaluator_label": "minimal_amplitude",
        "parameter_layout": "global-value-momentum",
        "output_length": 1,
        "output_slots": [{
            "value_slot_id": -1,
            "current_id": -1,
            "variant": "amplitude-root",
            "component_start": 0,
            "component_stop": 1,
            "output_start": 0,
            "output_stop": 1,
        }],
        "input_value_slot_ids": [0, 1],
        "output_value_slot_ids": [],
        "interaction_ids": [],
        "input_components": [],
        "parameter_count": 14,
        "value_parameter_count": 2,
        "momentum_parameter_count": 12,
        "model_parameter_count": 0,
        "real_valued_inputs": (2..14).collect::<Vec<_>>(),
        "expression_ready": true,
        "blockers": [],
        "evaluator": evaluator,
    });
    let compiled = json!({
        "kind": "generic-dag-stage-blueprint",
        "runtime_available": true,
        "runtime_unavailable_message": null,
        "model_parameter_evaluator": null,
        "stage_evaluators": {
            "kind": "generic-dag-stage-evaluator-artifacts",
            "required_runtime_capabilities": [capability],
            "runtime_available": true,
            "runtime_unavailable_message": null,
            "parameter_count": 14,
            "value_parameter_count": 2,
            "momentum_parameter_count": 12,
            "model_parameter_count": 0,
            "real_valued_inputs": (2..14).collect::<Vec<_>>(),
            "parameter_layout": "global-value-momentum",
            "stage_count": 1,
            "stages": [],
            "amplitude_stage": serialized_amplitude,
        },
    });
    json!({
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": "pyamplicol-runtime-execution",
        "required_runtime_capabilities": [capability],
        "process": process,
        "key": process_id,
        "color_accuracy": "lc",
        "external_pdg_order": [1, -1, 22],
        "compiled": compiled,
        "dag_summary": {
            "current_count": 2,
            "source_count": 2,
            "interaction_count": 0,
            "amplitude_root_count": 1,
            "truncated": false,
        },
        "runtime_schema": runtime_schema,
    })
}

#[cfg(feature = "f64-symjit")]
fn mixed_backend_runtime_artifact() -> TestArtifact {
    const DIRECT: &str = "symjit.application.complex-f64.v1";
    const ASM: &str = "symbolica.compiled-asm.complex-f64.v1";
    const CPP: &str = "symbolica.compiled-cpp.complex-f64.v1";
    let mut artifact = TestArtifact::new();
    artifact.manifest["kind"] = json!("pyamplicol-process-set");
    artifact.manifest["default_process_id"] = json!("direct");
    artifact.manifest["processes"] = json!([
        {
            "id": "direct",
            "expression": "a b > c",
            "color_accuracy": "lc",
            "external_pdgs": [1, -1, 22],
            "physics_path": "processes/direct/physics.json",
            "required_runtime_capabilities": [DIRECT],
            "aliases": [],
        },
        {
            "id": "asm",
            "expression": "a b > e",
            "color_accuracy": "lc",
            "external_pdgs": [1, -1, 22],
            "physics_path": "processes/asm/physics.json",
            "required_runtime_capabilities": [ASM],
            "aliases": [],
        },
        {
            "id": "cpp",
            "expression": "a b > d",
            "color_accuracy": "lc",
            "external_pdgs": [1, -1, 22],
            "physics_path": "processes/cpp/physics.json",
            "required_runtime_capabilities": [CPP],
            "aliases": [],
        }
    ]);
    artifact.manifest["runtime"]["evaluator_manifest_path"] = json!("processes/evaluators.json");
    artifact.manifest["runtime"]["required_runtime_capabilities"] = json!([ASM, CPP, DIRECT]);

    let direct_application = direct_symjit_application_bytes(14);
    let direct_execution = minimal_execution_manifest(
        "direct",
        "a b > c",
        DIRECT,
        direct_evaluator_manifest("evaluators/direct.symjit"),
    );
    let cpp_execution = minimal_execution_manifest(
        "cpp",
        "a b > d",
        CPP,
        compiled_evaluator_manifest("evaluators/fake-cpp.dylib", CPP),
    );
    let asm_execution = minimal_execution_manifest(
        "asm",
        "a b > e",
        ASM,
        compiled_evaluator_manifest("evaluators/fake-asm.dylib", ASM),
    );
    let execution_set = json!({
        "schema_version": PROCESS_ARTIFACT_SCHEMA_VERSION,
        "kind": "pyamplicol-runtime-execution-set",
        "required_runtime_capabilities": [ASM, CPP, DIRECT],
        "processes": [
            {
                "process_id": "direct",
                "manifest_path": "direct/execution.json",
                "required_runtime_capabilities": [DIRECT]
            },
            {
                "process_id": "asm",
                "manifest_path": "asm/execution.json",
                "required_runtime_capabilities": [ASM]
            },
            {
                "process_id": "cpp",
                "manifest_path": "cpp/execution.json",
                "required_runtime_capabilities": [CPP]
            }
        ]
    });
    for (path, role, value, process_id) in [
        (
            "processes/evaluators.json",
            "evaluator-manifest",
            execution_set,
            None,
        ),
        (
            "processes/direct/execution.json",
            "evaluator-manifest",
            direct_execution,
            Some("direct"),
        ),
        (
            "processes/asm/execution.json",
            "evaluator-manifest",
            asm_execution,
            Some("asm"),
        ),
        (
            "processes/cpp/execution.json",
            "evaluator-manifest",
            cpp_execution,
            Some("cpp"),
        ),
        (
            "processes/direct/physics.json",
            "runtime-physics",
            minimal_runtime_physics("direct", "a b > c", "c"),
            Some("direct"),
        ),
        (
            "processes/asm/physics.json",
            "runtime-physics",
            minimal_runtime_physics("asm", "a b > e", "e"),
            Some("asm"),
        ),
        (
            "processes/cpp/physics.json",
            "runtime-physics",
            minimal_runtime_physics("cpp", "a b > d", "d"),
            Some("cpp"),
        ),
    ] {
        let bytes = serde_json::to_vec(&value).expect("serialize mixed-runtime JSON");
        add_test_payload(&mut artifact, path, role, &bytes, process_id, false);
    }
    add_test_payload(
        &mut artifact,
        "processes/direct/evaluators/direct.symjit",
        "evaluator-state",
        &direct_application,
        Some("direct"),
        true,
    );
    add_test_payload(
        &mut artifact,
        "processes/asm/evaluators/fake-asm.dylib",
        "evaluator-state",
        b"unselected fake ASM evaluator",
        Some("asm"),
        true,
    );
    add_test_payload(
        &mut artifact,
        "processes/cpp/evaluators/fake-cpp.dylib",
        "evaluator-state",
        b"unselected fake C++ evaluator",
        Some("cpp"),
        true,
    );
    artifact.write_manifest();
    artifact
}

#[test]
fn artifact_identity_matches_python_canonical_json() {
    let value = json!({
        "artifact_id": "0".repeat(64),
        "ascii": "line\\nquote\"",
        "float": 1e-6,
        "integer": 7,
        "unicode": "é𝄞",
    });

    assert_eq!(
        compute_artifact_id(&value).unwrap(),
        "68a18ed104e4bd88b1f2728869398f7a751d5c63642eb9fe8019559c62585619"
    );
}

#[test]
fn artifact_identity_preserves_python_exponent_spelling() {
    let bytes = concat!(
        "{\"artifact_id\":\"",
        "0000000000000000000000000000000000000000000000000000000000000000",
        "\",\"timing\":2.5417190045118332e-05}\n",
    );

    assert_eq!(
        compute_artifact_id_from_bytes(bytes.as_bytes()).unwrap(),
        "baf08640782801524444ad250d48322d3ad303abf83e788371e1485f209182d1"
    );
}

#[test]
fn valid_manifest_verifies_all_payloads() {
    let artifact = TestArtifact::new();
    let verified = VerifiedArtifact::open(&artifact.root).expect("valid artifact");

    assert_eq!(verified.manifest().schema_version, 3);
    assert_eq!(verified.select_process(None).unwrap().requested_id, "p0");
    assert_eq!(
        verified
            .select_process(Some("  a   b  >  c "))
            .expect("select process by normalized expression")
            .requested_id,
        "p0"
    );
}

#[test]
fn ambiguous_process_expression_lists_stable_ids() {
    let artifact = TestArtifact::new();
    let verified = VerifiedArtifact::open(&artifact.root).expect("valid artifact");
    let mut manifest = verified.manifest().clone();
    let mut duplicate = manifest.processes[0].clone();
    duplicate.id = "p1".to_string();
    manifest.processes.push(duplicate);

    let error = manifest
        .select_process(Some("a b > c"))
        .expect_err("duplicate expressions must be ambiguous");

    assert_eq!(error.kind(), crate::RusticolErrorKind::Selector);
    assert!(error.to_string().contains("ambiguous"));
    assert!(error.to_string().contains("p0, p1"));
}

#[test]
fn direct_runtime_defers_symbolica_serialization_abi_validation() {
    let mut artifact = TestArtifact::new();
    artifact.manifest["producer"]["versions"]["symbolica_serialization"] =
        json!("a-future-symbolica-serialization-abi");
    artifact.write_manifest();

    let verified = VerifiedArtifact::open(&artifact.root)
        .expect("outer artifact validation must not require an unused Symbolica serialization ABI");

    assert_eq!(
        verified
            .manifest()
            .producer
            .versions
            .symbolica_serialization,
        Some("a-future-symbolica-serialization-abi".to_string())
    );

    let mut no_symbolica_abi = TestArtifact::new();
    no_symbolica_abi.manifest["producer"]["versions"]
        .as_object_mut()
        .unwrap()
        .remove("symbolica_serialization");
    no_symbolica_abi.write_manifest();

    let verified = VerifiedArtifact::open(&no_symbolica_abi.root)
        .expect("direct f64 artifact must not require Symbolica ABI metadata");
    assert_eq!(
        verified
            .manifest()
            .producer
            .versions
            .symbolica_serialization,
        None
    );
}

#[test]
fn artifact_identity_is_recomputed_before_loading_payloads() {
    let artifact = TestArtifact::new();
    let path = artifact.root.join(ARTIFACT_MANIFEST_FILE);
    let mut manifest: Value =
        serde_json::from_slice(&fs::read(&path).unwrap()).expect("parse fixture manifest");
    manifest["model"]["name"] = json!("tampered-model");
    fs::write(&path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();

    let error = VerifiedArtifact::open(&artifact.root).unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
    assert!(error.to_string().contains("identity digest mismatch"));
}

#[test]
fn unknown_nested_manifest_keys_are_rejected() {
    let mut artifact = TestArtifact::new();
    artifact.manifest["producer"]["target"]["unexpected"] = json!(true);
    artifact.write_manifest();

    let error = VerifiedArtifact::open(&artifact.root).unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Serialization);
    assert!(error.to_string().contains("unknown field"));
}

#[test]
fn only_canonical_artifact_filename_is_accepted() {
    let artifact = TestArtifact::new();
    let legacy = artifact.root.join("manifest.json");
    fs::rename(artifact.root.join(ARTIFACT_MANIFEST_FILE), &legacy)
        .expect("rename canonical manifest");

    let directory_error = VerifiedArtifact::open(&artifact.root).unwrap_err();
    let direct_error = VerifiedArtifact::open(&legacy).unwrap_err();

    assert_eq!(directory_error.kind(), crate::RusticolErrorKind::Artifact);
    assert_eq!(direct_error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(directory_error.to_string().contains("artifact.json"));
    assert!(direct_error.to_string().contains("artifact.json"));
}

#[test]
fn legacy_schema_reports_actionable_regeneration_error() {
    let mut artifact = TestArtifact::new();
    artifact.manifest = json!({"schema_version": 2, "kind": "legacy"});
    artifact.write_manifest();

    let error = VerifiedArtifact::open(&artifact.root).unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
    assert!(error.to_string().contains("regenerate"));
    assert!(error.to_string().contains("schema v3"));
}

#[test]
fn traversal_and_duplicate_payload_paths_are_rejected() {
    let mut traversal = TestArtifact::new();
    traversal.manifest["payloads"][3]["path"] = json!("../physics.json");
    traversal.write_manifest();
    let error = VerifiedArtifact::open(&traversal.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Security);

    let mut duplicate = TestArtifact::new();
    duplicate.manifest["payloads"][1]["path"] = json!("requested.toml");
    duplicate.write_manifest();
    let error = VerifiedArtifact::open(&duplicate.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Security);
    assert!(error.to_string().contains("duplicate"));
}

#[test]
fn payload_size_and_digest_are_checked_before_use() {
    let artifact = TestArtifact::new();
    fs::write(artifact.root.join("physics.json"), b"tampered").unwrap();

    let error = VerifiedArtifact::open(&artifact.root).unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
    assert!(error.to_string().contains("size") || error.to_string().contains("SHA-256"));
}

#[test]
fn manifest_preflight_runs_before_payload_verification() {
    let mut artifact = TestArtifact::new();
    artifact.manifest["processes"][0]["required_runtime_capabilities"] =
        json!(["symbolica.compiled-cpp.complex-f64.v1"]);
    artifact.manifest["runtime"]["required_runtime_capabilities"] =
        json!(["symbolica.compiled-cpp.complex-f64.v1"]);
    artifact.write_manifest();
    fs::write(artifact.root.join("physics.json"), b"tampered").unwrap();

    let error = VerifiedArtifact::open_with_manifest_preflight(&artifact.root, |manifest| {
        assert_eq!(
            manifest.runtime.required_runtime_capabilities,
            ["symbolica.compiled-cpp.complex-f64.v1"]
        );
        Err(RusticolError::unsupported_runtime_capability(
            &manifest.runtime.required_runtime_capabilities[0],
            "test runtime supports direct SymJIT only",
        ))
    })
    .unwrap_err();

    assert_eq!(
        error.kind(),
        crate::RusticolErrorKind::UnsupportedRuntimeCapability
    );
}

#[test]
fn process_capabilities_are_strict_and_form_the_runtime_union() {
    for (capabilities, message) in [
        (json!([]), "at least one"),
        (
            json!([
                "symjit.application.complex-f64.v1",
                "symjit.application.complex-f64.v1"
            ]),
            "duplicates",
        ),
        (
            json!([
                "symjit.application.complex-f64.v1",
                "symbolica.compiled-cpp.complex-f64.v1"
            ]),
            "sorted",
        ),
        (json!(["unknown.runtime.v1"]), "unsupported capabilities"),
    ] {
        let mut artifact = TestArtifact::new();
        artifact.manifest["processes"][0]["required_runtime_capabilities"] = capabilities.clone();
        artifact.manifest["runtime"]["required_runtime_capabilities"] = capabilities;
        artifact.write_manifest();

        let error = VerifiedArtifact::open(&artifact.root).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
        assert!(error.to_string().contains(message), "{error}");
    }

    let mut mismatch = TestArtifact::new();
    mismatch.manifest["runtime"]["required_runtime_capabilities"] =
        json!(["symbolica.compiled-cpp.complex-f64.v1"]);
    mismatch.write_manifest();
    let error = VerifiedArtifact::open(&mismatch.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("union"));

    let mut mixed = TestArtifact::new();
    mixed.manifest["kind"] = json!("pyamplicol-process-set");
    mixed.manifest["processes"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "id": "p1",
            "expression": "a b > d",
            "color_accuracy": "lc",
            "external_pdgs": [1, -1, 23],
            "physics_path": "physics.json",
            "required_runtime_capabilities": [
                "symbolica.compiled-cpp.complex-f64.v1"
            ],
            "aliases": [],
        }));
    mixed.manifest["runtime"]["required_runtime_capabilities"] = json!([
        "symbolica.compiled-cpp.complex-f64.v1",
        "symjit.application.complex-f64.v1"
    ]);
    let parsed: ArtifactManifest =
        serde_json::from_value(mixed.manifest.clone()).expect("parse mixed manifest");
    validate_manifest(&parsed).expect("mixed process capability union validates");
}

#[cfg(all(feature = "f64-symjit", not(feature = "symbolica-runtime")))]
#[test]
fn selected_unsupported_process_capability_preflights_before_payloads() {
    let mut artifact = TestArtifact::new();
    artifact.manifest["processes"][0]["required_runtime_capabilities"] =
        json!(["symbolica.compiled-cpp.complex-f64.v1"]);
    artifact.manifest["runtime"]["required_runtime_capabilities"] =
        json!(["symbolica.compiled-cpp.complex-f64.v1"]);
    artifact.write_manifest();
    fs::write(artifact.root.join("physics.json"), b"tampered").unwrap();

    let error = match crate::NativeRuntime::load(&artifact.root, Some("p0"), None) {
        Ok(_) => panic!("unsupported selected process unexpectedly loaded"),
        Err(error) => error,
    };

    assert_eq!(
        error.kind(),
        crate::RusticolErrorKind::UnsupportedRuntimeCapability
    );
    assert!(error.to_string().contains("compiled-cpp"));
}

#[cfg(feature = "f64-symjit")]
#[test]
fn mixed_backend_process_set_loads_selected_direct_symjit_process() {
    let artifact = mixed_backend_runtime_artifact();

    let runtime = crate::NativeRuntime::load(&artifact.root, Some("direct"), None)
        .expect("selected direct-SymJIT process loads despite unselected C++/ASM processes");

    assert_eq!(runtime.metadata().process_key, "direct");
    assert_eq!(runtime.metadata().external_pdg_order, vec![1, -1, 22]);
}

#[cfg(all(feature = "f64-symjit", not(feature = "symbolica-runtime")))]
#[test]
fn mixed_backend_process_set_rejects_unsupported_selection_before_tampered_payload() {
    let artifact = mixed_backend_runtime_artifact();
    fs::write(
        artifact.root.join("processes/direct/physics.json"),
        b"tampered after manifest creation",
    )
    .unwrap();

    for (process_id, capability) in [("asm", "compiled-asm"), ("cpp", "compiled-cpp")] {
        let error = match crate::NativeRuntime::load(&artifact.root, Some(process_id), None) {
            Ok(_) => panic!("unsupported selected {process_id} process unexpectedly loaded"),
            Err(error) => error,
        };

        assert_eq!(
            error.kind(),
            crate::RusticolErrorKind::UnsupportedRuntimeCapability
        );
        assert!(error.to_string().contains(capability));
    }
}

#[test]
fn required_nullable_and_target_fields_remain_strict() {
    let mut missing_nullable = TestArtifact::new();
    missing_nullable.manifest["runtime"]
        .as_object_mut()
        .unwrap()
        .remove("api_bundle_path");
    missing_nullable.write_manifest();
    let error = VerifiedArtifact::open(&missing_nullable.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("api_bundle_path"));

    let mut missing_target = TestArtifact::new();
    let state = b"serialized state";
    fs::write(missing_target.root.join("state.bin"), state).unwrap();
    missing_target.manifest["payloads"]
        .as_array_mut()
        .unwrap()
        .push(json!({
            "path": "state.bin",
            "role": "evaluator-state",
            "media_type": "application/octet-stream",
            "size_bytes": state.len(),
            "sha256": sha256(state),
            "executable": false,
        }));
    missing_target.write_manifest();
    let error = VerifiedArtifact::open(&missing_target.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("target"));
}

#[test]
fn producer_and_payload_cpu_features_are_checked() {
    let mut unavailable = TestArtifact::new();
    unavailable.manifest["producer"]["target"]["cpu_features"] =
        json!(["definitely-not-a-real-cpu-feature"]);
    unavailable.write_manifest();
    fs::write(unavailable.root.join("physics.json"), b"tampered payload").unwrap();
    let error = VerifiedArtifact::open(&unavailable.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
    assert!(error.to_string().contains("CPU feature"));

    let mut mismatch = TestArtifact::new();
    mismatch.manifest["payloads"][2]["target"] = json!({
        "triple": current_target_triple(),
        "cpu_features": ["baseline"],
    });
    mismatch.write_manifest();
    let error = VerifiedArtifact::open(&mismatch.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Compatibility);
    assert!(
        error
            .to_string()
            .contains("do not match producer CPU features")
    );
}

#[test]
fn target_metadata_requires_canonical_sorted_cpu_features() {
    let target = runtime_target_info();
    assert_eq!(target.triple, current_target_triple());
    assert!(target.cpu_features.windows(2).all(|pair| pair[0] < pair[1]));
    validate_target(&target, "runtime target").expect("runtime target validates");

    let mut unsorted = TestArtifact::new();
    unsorted.manifest["producer"]["target"]["cpu_features"] = json!(["z-feature", "a-feature"]);
    unsorted.write_manifest();
    let error = VerifiedArtifact::open(&unsorted.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("sorted and unique"));

    let mut noncanonical = TestArtifact::new();
    noncanonical.manifest["producer"]["target"]["cpu_features"] = json!(["+avx2"]);
    noncanonical.write_manifest();
    let error = VerifiedArtifact::open(&noncanonical.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("canonical feature ID"));
}

#[test]
fn malformed_timestamp_and_initial_state_alias_are_rejected() {
    let mut timestamp = TestArtifact::new();
    timestamp.manifest["created_utc"] = json!("2026-02-30T25:00:00Z");
    timestamp.write_manifest();
    let error = VerifiedArtifact::open(&timestamp.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);

    let mut alias = TestArtifact::new();
    alias.manifest["processes"][0]["aliases"] = json!([{
        "id": "alias0",
        "expression": "b a > c",
        "external_pdgs": [-1, 1, 22],
        "external_permutation": [1, 0, 2],
    }]);
    alias.write_manifest();
    let error = VerifiedArtifact::open(&alias.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(error.to_string().contains("final-state"));
}

#[test]
fn non_self_inverse_alias_pdg_order_is_verified() {
    let mut valid = TestArtifact::new();
    valid.manifest["processes"][0]["expression"] = json!("d d~ > z g a");
    valid.manifest["processes"][0]["external_pdgs"] = json!([1, -1, 23, 21, 22]);
    valid.manifest["processes"][0]["aliases"] = json!([{
        "id": "cycled",
        "expression": "d d~ > a z g",
        "external_pdgs": [1, -1, 22, 23, 21],
        "external_permutation": [0, 1, 3, 4, 2],
    }]);
    valid.write_manifest();

    let verified = VerifiedArtifact::open(&valid.root).expect("valid three-cycle alias");
    let selected = verified
        .select_process(Some("cycled"))
        .expect("select three-cycle alias");
    assert_eq!(
        selected.process.required_runtime_capabilities,
        ["symjit.application.complex-f64.v1"]
    );
    assert_eq!(
        selected.alias.expect("alias metadata").external_pdgs,
        vec![1, -1, 22, 23, 21]
    );
    let selected_by_expression = verified
        .select_process(Some("d   d~ > a z g"))
        .expect("select three-cycle alias by expression");
    assert_eq!(selected_by_expression.requested_id, "cycled");
    assert_eq!(
        selected_by_expression
            .alias
            .expect("alias metadata selected by expression")
            .external_pdgs,
        vec![1, -1, 22, 23, 21]
    );

    let mut inconsistent = TestArtifact::new();
    inconsistent.manifest["processes"][0]["expression"] = json!("d d~ > z g a");
    inconsistent.manifest["processes"][0]["external_pdgs"] = json!([1, -1, 23, 21, 22]);
    inconsistent.manifest["processes"][0]["aliases"] = json!([{
        "id": "cycled",
        "expression": "d d~ > a z g",
        "external_pdgs": [1, -1, 21, 23, 22],
        "external_permutation": [0, 1, 3, 4, 2],
    }]);
    inconsistent.write_manifest();

    let error = VerifiedArtifact::open(&inconsistent.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Artifact);
    assert!(
        error
            .to_string()
            .contains("does not match external_permutation")
    );
}

#[cfg(unix)]
#[test]
fn symlink_payload_is_rejected_even_when_it_stays_inside_root() {
    use std::os::unix::fs::symlink;

    let artifact = TestArtifact::new();
    fs::rename(
        artifact.root.join("physics.json"),
        artifact.root.join("physics-target.json"),
    )
    .unwrap();
    symlink("physics-target.json", artifact.root.join("physics.json")).unwrap();

    let error = VerifiedArtifact::open(&artifact.root).unwrap_err();

    assert_eq!(error.kind(), crate::RusticolErrorKind::Security);
    assert!(error.to_string().contains("symlink"));
}

#[cfg(unix)]
#[test]
fn undeclared_symlinks_and_executables_are_rejected_anywhere_in_tree() {
    use std::os::unix::fs::{PermissionsExt, symlink};

    let symlink_artifact = TestArtifact::new();
    fs::create_dir(symlink_artifact.root.join("extra")).unwrap();
    symlink(
        "../physics.json",
        symlink_artifact.root.join("extra/undeclared-link"),
    )
    .unwrap();
    let error = VerifiedArtifact::open(&symlink_artifact.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Security);
    assert!(error.to_string().contains("symlink"));

    let executable_artifact = TestArtifact::new();
    let executable = executable_artifact.root.join("undeclared-tool");
    fs::write(&executable, b"#!/bin/sh\n").unwrap();
    let mut permissions = fs::metadata(&executable).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&executable, permissions).unwrap();
    let error = VerifiedArtifact::open(&executable_artifact.root).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::Security);
    assert!(error.to_string().contains("undeclared executable"));
}
