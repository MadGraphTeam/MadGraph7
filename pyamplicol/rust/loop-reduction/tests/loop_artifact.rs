// SPDX-License-Identifier: 0BSD
//! Graph-free schema-v3 transport, using the retained complete source gg plan.
//! No diagram generation, numerical reduction or producer import is repeated.

use rusticol_core::loop_artifact::{load_loop_artifact, open_loop_artifact};
use rusticol_core::{NativeRuntime, VerifiedArtifact};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

struct Fixture(PathBuf);
impl Fixture {
    fn copy() -> Option<Self> {
        let original = std::env::var_os("PYAMPLICOL_PUBLIC_SOURCE_ARTIFACT")?;
        let path = std::env::temp_dir().join(format!(
            "loop-artifact-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fn copy_dir(source: &Path, destination: &Path) {
            std::fs::create_dir_all(destination).unwrap();
            for item in std::fs::read_dir(source).unwrap() {
                let item = item.unwrap();
                let target = destination.join(item.file_name());
                if item.file_type().unwrap().is_dir() {
                    copy_dir(&item.path(), &target);
                } else {
                    std::fs::copy(item.path(), target).unwrap();
                }
            }
        }
        copy_dir(Path::new(&original), &path);
        Some(Self(path))
    }
    fn alter(&self, relative: &str, change: impl FnOnce(&mut Value)) {
        let path = self.0.join(relative);
        let mut data: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        change(&mut data);
        let bytes = serde_json::to_vec(&data).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        // Normal loading intentionally does not hash JSON metadata. Keep the
        // declared length coherent to reach the typed semantic boundary.
        let manifest_path = self.0.join("artifact.json");
        let mut manifest: Value =
            serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
        for payload in manifest["payloads"].as_array_mut().unwrap() {
            if payload["path"] == relative {
                payload["size_bytes"] = json!(bytes.len());
            }
        }
        std::fs::write(manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn complete_source_plan_loads_without_a_producer_or_scalar_tree_lane() {
    let Some(fixture) = Fixture::copy() else {
        return;
    };
    let loaded = open_loop_artifact(&fixture.0, Some("gg")).unwrap();
    assert_eq!(loaded.selection.process.id, "gg");
    assert_eq!(loaded.runtime.plan().external_pdgs, [21; 4]);
    assert_eq!(loaded.metadata.epsilon_powers, [0, -1, -2]);
    assert_eq!(loaded.metadata.available_r2_strategies, ["model"]);
    assert_eq!(
        loaded.metadata.orders.r#virtual[0].order,
        [("QCD".to_owned(), 6)]
    );
    assert_eq!(loaded.metadata.scheme.dimensional_regularization, "HV");
    assert!(!loaded.artifact_id.is_empty());
    assert_eq!(
        open_loop_artifact(&fixture.0, Some("gg_second"))
            .unwrap()
            .selection
            .process
            .id,
        "gg_second"
    );
    let error = NativeRuntime::load(&fixture.0, Some("gg"), None)
        .err()
        .expect("tree scalar runtime must reject virtual handles");
    assert!(
        error.to_string().contains("rusticol.loop-runtime.v3"),
        "{error}"
    );
}

#[test]
fn malformed_public_metadata_cannot_change_execution_semantics() {
    let changes: &[(&str, Value)] = &[
        ("epsilon_powers", json!([0, -2, -1])),
        ("r2_strategy", json!("direct")),
        ("available_r2_strategies", json!(["model", "direct"])),
        ("contribution_kind", json!("tree")),
        ("arithmetic", json!(["double"])),
        ("source_owned", json!(false)),
        ("resolved", json!(true)),
        ("correlations", json!(true)),
        ("partial_results", json!(true)),
        ("scheme_status", json!("certified")),
    ];
    for (key, value) in changes {
        let Some(fixture) = Fixture::copy() else {
            return;
        };
        fixture.alter("processes/gg/physics.json", |p| {
            p["metadata"][key] = value.clone()
        });
        assert!(
            open_loop_artifact(&fixture.0, Some("gg")).is_err(),
            "accepted malformed {key}"
        );
    }
    for case in 0..5 {
        let Some(fixture) = Fixture::copy() else {
            return;
        };
        fixture.alter("processes/gg/physics.json", |p| match case {
            0 => p["metadata"]["orders"]["virtual"][0]["order"][0][1] = json!(8),
            1 => p["initial_average"] = json!(256),
            2 => p["metadata"]["scheme"]["gamma5"] = json!(""),
            3 => p["state_legs"][0]["incoming"] = json!(false),
            _ => p["metadata"]["undeclared_capability"] = json!(true),
        });
        assert!(
            open_loop_artifact(&fixture.0, Some("gg")).is_err(),
            "accepted malformed metadata case {case}"
        );
    }
}

#[test]
fn checked_artifact_rejects_replaced_runtime_and_incoherent_set() {
    let Some(fixture) = Fixture::copy() else {
        return;
    };
    let artifact = VerifiedArtifact::open(&fixture.0).unwrap();
    let path = fixture.0.join("processes/gg/loop-runtime.pacbin");
    let mut bytes = std::fs::read(&path).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(path, bytes).unwrap();
    let error = load_loop_artifact(&artifact, Some("gg"))
        .err()
        .expect("changed PACBIN must be authenticated through checked handle");
    assert!(error.to_string().contains("digest mismatch"), "{error}");
    let Some(fixture) = Fixture::copy() else {
        return;
    };
    fixture.alter("processes/evaluators.json", |p| {
        p["processes"][0]["required_runtime_capabilities"] = json!([])
    });
    assert!(open_loop_artifact(&fixture.0, Some("gg")).is_err());
}
