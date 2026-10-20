// SPDX-License-Identifier: 0BSD

//! Cold validation of normally exported, owned projected flavour fixtures.
//! Run explicitly with `--ignored` and PYAMPLICOL_PROJECTED_FLAVOR_FIXTURES
//! pointing to the directory containing identity-lc.pacbin and
//! admitted-{lc,nlc}.pacbin. No process construction or event evaluation occurs.

#![cfg(feature = "symbolica-runtime")]

use rusticol_core::loop_runtime::{
    LOOP_RUNTIME_MEMBER, PROJECTED_FAMILY_CONVENTION, PROJECTED_IDENTITY_CONVENTION,
    load_loop_runtime_pacbin,
};
use rusticol_core::pacbin::{
    PacbinReader, PacbinWriteMember, PacbinWriteOptions, write_pacbin_atomic,
};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_TEMPORARY: AtomicU64 = AtomicU64::new(0);

struct TemporaryArtifact(PathBuf);

impl TemporaryArtifact {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "rusticol-projected-flavor-{}-{}",
            std::process::id(),
            NEXT_TEMPORARY.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::create_dir(&path).expect("create isolated mutation directory");
        Self(path)
    }
}

impl Drop for TemporaryArtifact {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn check_positive(path: &Path, convention: &str, accuracy: &str) {
    let loaded = load_loop_runtime_pacbin(path).expect("load actual admitted projected fixture");
    let plan = loaded.plan();
    assert!(
        !plan
            .admitted_flavor_members
            .as_ref()
            .expect("actual flavour admission")
            .is_empty()
    );
    assert_eq!(plan.color_projection.as_ref().unwrap().accuracy, accuracy);
    let ordinary = plan.directed_projection.as_ref().unwrap();
    assert_eq!(ordinary.convention, convention);
    let complete = convention == PROJECTED_FAMILY_CONVENTION;
    assert_eq!(ordinary.color_family_projection.is_some(), complete);
    assert_eq!(
        ordinary
            .histories
            .iter()
            .any(|history| !history.bra.is_empty() || !history.ket.is_empty()),
        complete,
    );
    let sources = plan.spin_projections.as_deref().unwrap_or_default();
    if complete {
        assert!(
            !sources.is_empty(),
            "complete fixture includes source classes"
        );
    }
    for source in sources {
        assert_eq!(source.directed.convention, convention);
        assert_eq!(
            source.directed.color_family_projection,
            ordinary.color_family_projection
        );
    }
}

fn alter_all_directed(plan: &mut Value, mut mutation: impl FnMut(&mut Value)) {
    mutation(&mut plan["directed_projection"]);
    for source in plan["spin_projections"].as_array_mut().unwrap() {
        mutation(&mut source["directed"]);
    }
}

fn rejected_mutation(
    original: &PacbinReader,
    label: &str,
    expected_error: &str,
    mutation: impl FnOnce(&mut Value),
) {
    let mut plan: Value =
        serde_json::from_slice(original.member_bytes(LOOP_RUNTIME_MEMBER).unwrap()).unwrap();
    mutation(&mut plan);
    let changed = serde_json::to_vec(&plan).unwrap();
    // All evaluator payloads remain byte-for-byte the actual source fixture.
    // The normal writer supplies the envelope and its integrity metadata.
    let members = original
        .members()
        .iter()
        .map(|member| {
            let bytes = if member.logical_path() == LOOP_RUNTIME_MEMBER {
                changed.as_slice()
            } else {
                original.member_bytes(member.logical_path()).unwrap()
            };
            PacbinWriteMember::from_bytes(member.logical_path(), member.kind(), bytes).unwrap()
        })
        .collect::<Vec<_>>();
    let temporary = TemporaryArtifact::new();
    let destination = temporary.0.join("mutated.pacbin");
    write_pacbin_atomic(&destination, members, PacbinWriteOptions::default()).unwrap();
    let error = match load_loop_runtime_pacbin(&destination) {
        Ok(_) => panic!("{label}: malformed projected flavour descriptor was admitted"),
        Err(error) => error.to_string(),
    };
    assert!(
        error.contains(expected_error),
        "{label}: expected {expected_error:?}, got {error:?}"
    );
}

#[test]
#[ignore = "requires actual exported projected flavour fixtures and licensed evaluator decoding"]
fn actual_projected_flavor_descriptors_use_existing_native_validators() {
    let root = PathBuf::from(
        std::env::var_os("PYAMPLICOL_PROJECTED_FLAVOR_FIXTURES")
            .expect("set PYAMPLICOL_PROJECTED_FLAVOR_FIXTURES"),
    );
    check_positive(
        &root.join("identity-lc.pacbin"),
        PROJECTED_IDENTITY_CONVENTION,
        "lc",
    );
    for accuracy in ["lc", "nlc"] {
        check_positive(
            &root.join(format!("admitted-{accuracy}.pacbin")),
            PROJECTED_FAMILY_CONVENTION,
            accuracy,
        );
    }
    let original = PacbinReader::open(root.join("admitted-lc.pacbin")).unwrap();
    rejected_mutation(
        &original,
        "missing complete family",
        "projected flavour transport requires identity histories or complete colour families",
        |plan| {
            alter_all_directed(plan, |directed| {
                directed["color_family_projection"] = Value::Null;
            });
        },
    );
    rejected_mutation(
        &original,
        "nonidentity family advertised as identity-only",
        "projected flavour transport requires identity histories or complete colour families",
        |plan| {
            alter_all_directed(plan, |directed| {
                directed["convention"] = json!(PROJECTED_IDENTITY_CONVENTION);
            });
        },
    );
    rejected_mutation(
        &original,
        "foreign family accuracy",
        "invalid complete directed colour family descriptor",
        |plan| {
            alter_all_directed(plan, |directed| {
                directed["color_family_projection"]["accuracy"] = json!("nlc");
            });
        },
    );
    rejected_mutation(
        &original,
        "foreign degree axis",
        "directed colour family degree axes differ",
        |plan| {
            alter_all_directed(plan, |directed| {
                directed["color_family_projection"]["families"][0]["born_degrees"]
                    .as_array_mut()
                    .unwrap()
                    .push(Value::Null);
            });
        },
    );
    rejected_mutation(
        &original,
        "identity family differs from ordinary authority",
        "identity family degrees differ from ordinary projection",
        |plan| {
            alter_all_directed(plan, |directed| {
                let family = directed["color_family_projection"]["families"]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .find(|family| family["order"] == 0)
                    .expect("mixed complete fixture includes identity family");
                let degree = &mut family["born_degrees"][0];
                *degree = json!(degree.as_i64().expect("nonzero identity Born sector") + 1);
            });
        },
    );
    rejected_mutation(
        &original,
        "foreign history-family index",
        "invalid complete directed colour family descriptor",
        |plan| {
            alter_all_directed(plan, |directed| {
                let family = &mut directed["color_family_projection"];
                family["history_families"][0] = json!(family["families"].as_array().unwrap().len());
            });
        },
    );
    rejected_mutation(
        &original,
        "altered source copy of valid ordinary descriptor",
        "spin and ordinary directed history columns differ",
        |plan| {
            let family =
                plan["spin_projections"][0]["directed"]["color_family_projection"]["families"]
                    .as_array_mut()
                    .unwrap()
                    .iter_mut()
                    .find(|family| family["order"].as_u64().unwrap() > 0)
                    .expect("complete fixture includes a nonidentity family");
            let degree = &mut family["born_degrees"][0];
            *degree = json!(degree.as_i64().unwrap_or(0) + 1);
        },
    );
}
