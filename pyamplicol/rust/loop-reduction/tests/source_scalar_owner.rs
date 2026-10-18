// SPDX-License-Identifier: 0BSD
//! Standalone source-owner checks use an actual compiled source-model artifact.
//! No process reducer or Python producer is constructed by this test.

use rusticol_core::loop_execution::LoopBackend;
use rusticol_core::loop_runtime::{ModelBinding, load_loop_runtime_pacbin};
use rusticol_core::source_bindings::{SourceOverrides, SourceOwner, SourceScale};
use rusticol_loop_reduction::runtime::physical::PhysicalBackend;
use std::sync::Arc;

fn changes(values: &[(&str, &str, &str)]) -> SourceOverrides {
    values
        .iter()
        .map(|(name, real, imaginary)| {
            (
                (*name).to_owned(),
                [(*real).to_owned(), (*imaginary).to_owned()],
            )
        })
        .collect()
}

#[test]
#[ignore = "requires PYAMPLICOL_NATIVE_SOURCE_PACBIN from source gg portable fixture"]
fn native_source_owner_updates_snapshots_scales_three_carriers() {
    let loaded = Arc::new(
        load_loop_runtime_pacbin(
            std::env::var("PYAMPLICOL_NATIVE_SOURCE_PACBIN").expect("compiled source gg PACBIN"),
        )
        .unwrap(),
    );
    let ModelBinding::Source {
        owner: record,
        initial_overrides,
        ..
    } = &loaded.plan().model_binding
    else {
        panic!("source fixture required");
    };
    let mass_output = record
        .outputs
        .iter()
        .position(|x| x == &("MT".into(), 0))
        .unwrap();
    for (arithmetic, precision) in [
        ("double", None),
        ("double-double", None),
        ("arbitrary", Some(80)),
    ] {
        let backend =
            PhysicalBackend::new(arithmetic, precision, "interpreted", "generic").unwrap();
        let owner = SourceOwner::new(
            loaded.clone(),
            &backend,
            Some(&changes(&[("MU_R", "1500", "0")])),
        )
        .unwrap();
        let before = owner.snapshot().unwrap();
        owner.require_snapshot(&before).unwrap();
        assert_eq!(before.revision(), 0);
        assert_eq!(before.overrides()["aS"], initial_overrides["aS"]);
        assert_eq!(before.overrides()["MT"], initial_overrides["MT"]);
        assert_eq!(
            before.overrides()["MU_R"],
            ["1500".to_owned(), "0".to_owned()]
        );
        let mut ordinary = SourceScale::ordinary(&owner, &backend).unwrap().unwrap();
        let mut counterterms = SourceScale::counterterms(&owner, &backend).unwrap();
        let scales = backend
            .ingress_complex(
                &[
                    ["1000000".into(), "0".into()],
                    ["900000".into(), "0".into()],
                    ["2100000".into(), "0".into()],
                ],
                3,
                1,
                vec![19, 3, 88],
            )
            .unwrap();
        let ordinary_before = ordinary.evaluate(&before, &scales).unwrap();
        let ct_before = counterterms.evaluate(&before, &scales).unwrap();
        assert_eq!(ordinary_before.ids(), [19, 3, 88]);
        assert_eq!(ct_before.ids(), [19, 3, 88]);
        assert!(ct_before.nonfinite_rows().is_empty());
        let rows = ct_before.to_decimal();
        assert_ne!(
            &rows[..ct_before.columns()],
            &rows[ct_before.columns()..2 * ct_before.columns()]
        );
        // Per-point scale substitution supersedes the owner's global MU_R.
        let original_scale_owner = SourceOwner::new(loaded.clone(), &backend, None).unwrap();
        let mut original_scale_view =
            SourceScale::counterterms(&original_scale_owner, &backend).unwrap();
        assert_eq!(
            original_scale_view
                .evaluate(&original_scale_owner.snapshot().unwrap(), &scales)
                .unwrap()
                .to_decimal(),
            ct_before.to_decimal()
        );
        let exact_mass = if arithmetic == "arbitrary" {
            "173.000000000000000000000000000000000000000000000000000000000001"
        } else {
            "173.000000000000000000001"
        };
        let after = owner
            .update(
                &backend,
                &changes(&[
                    ("MT", exact_mass, "0"),
                    ("aS", "0.120000000000000000000001", "0"),
                ]),
            )
            .unwrap();
        assert_eq!(after.revision(), 1);
        assert_eq!(after.overrides()["MT"][0], exact_mass);
        assert_eq!(before.overrides()["MT"], initial_overrides["MT"]);
        assert_eq!(
            counterterms
                .evaluate(&before, &scales)
                .unwrap()
                .to_decimal(),
            ct_before.to_decimal()
        );
        assert_eq!(
            ordinary.evaluate(&before, &scales).unwrap().to_decimal(),
            ordinary_before.to_decimal()
        );
        let ct_after = counterterms.evaluate(&after, &scales).unwrap();
        assert_ne!(ct_after.to_decimal(), ct_before.to_decimal());
        let mass = PhysicalBackend::select(after.values(), vec![mass_output]).unwrap();
        let expected = backend
            .ingress_complex(&[[exact_mass.into(), "0".into()]], 1, 1, vec![0])
            .unwrap();
        assert_eq!(mass.to_decimal(), expected.to_decimal());
        if arithmetic != "double" {
            assert_ne!(
                mass.to_decimal(),
                PhysicalBackend::select(before.values(), vec![mass_output])
                    .unwrap()
                    .to_decimal()
            );
        }
        // Reconstruct from exact overrides, never the old rounded arguments.
        let rebound = SourceOwner::new(loaded.clone(), &backend, Some(after.overrides())).unwrap();
        let rebound_snapshot = rebound.snapshot().unwrap();
        assert!(owner.require_snapshot(&rebound_snapshot).is_err());
        owner.require_snapshot(&before).unwrap();
        assert_eq!(
            rebound_snapshot.values().to_decimal(),
            after.values().to_decimal()
        );
        assert_eq!(rebound_snapshot.overrides(), after.overrides());
        assert!(counterterms.evaluate(&rebound_snapshot, &scales).is_err());
        for bad in [
            changes(&[("MT", "1", "1e-1000")]),
            changes(&[("aS", "NaN", "0")]),
            changes(&[("MB", "1", "0")]),
            changes(&[("unknown", "1", "0")]),
        ] {
            assert!(owner.update(&backend, &bad).is_err());
            assert!(Arc::ptr_eq(&owner.snapshot().unwrap(), &after));
        }
        for invalid in ["0", "-1"] {
            let scales = backend
                .ingress_complex(&[[invalid.into(), "0".into()]], 1, 1, vec![71])
                .unwrap();
            assert!(counterterms.evaluate(&after, &scales).is_err());
        }
        let complex = backend
            .ingress_complex(&[["1".into(), "1".into()]], 1, 1, vec![71])
            .unwrap();
        assert!(ordinary.evaluate(&after, &complex).is_err());
        let empty = backend.ingress_complex(&[], 0, 1, vec![]).unwrap();
        assert_eq!(counterterms.evaluate(&after, &empty).unwrap().rows(), 0);
        // The source zero-mass conditional remains live and skips inactive logs.
        let zero_mass = owner
            .update(&backend, &changes(&[("MT", "0", "0")]))
            .unwrap();
        assert!(
            counterterms
                .evaluate(&zero_mass, &scales)
                .unwrap()
                .nonfinite_rows()
                .is_empty()
        );
        assert_eq!(ct_before.ids(), [19, 3, 88]);
    }
}

#[test]
#[ignore = "requires PYAMPLICOL_NATIVE_SOURCE_PACBIN from source gg portable fixture"]
fn native_source_updates_publish_one_complete_revision() {
    let loaded = Arc::new(
        load_loop_runtime_pacbin(
            std::env::var("PYAMPLICOL_NATIVE_SOURCE_PACBIN").expect("compiled source gg PACBIN"),
        )
        .unwrap(),
    );
    let backend = PhysicalBackend::new("double-double", None, "interpreted", "generic").unwrap();
    let owner = SourceOwner::new(loaded.clone(), &backend, None).unwrap();
    let before = owner.snapshot().unwrap();
    let gate = std::sync::Barrier::new(3);
    std::thread::scope(|scope| {
        let left = scope.spawn(|| {
            gate.wait();
            owner
                .update(
                    &backend,
                    &changes(&[("aS", "0.12100000000000000000001", "0")]),
                )
                .unwrap()
        });
        let right = scope.spawn(|| {
            gate.wait();
            owner
                .update(
                    &backend,
                    &changes(&[("MT", "191.000000000000000000001", "0")]),
                )
                .unwrap()
        });
        gate.wait();
        let mut revisions = [
            left.join().unwrap().revision(),
            right.join().unwrap().revision(),
        ];
        revisions.sort();
        assert_eq!(revisions, [1, 2]);
    });
    let after = owner.snapshot().unwrap();
    assert_eq!(before.revision(), 0);
    assert_eq!(after.revision(), 2);
    assert_eq!(after.overrides()["aS"][0], "0.12100000000000000000001");
    assert_eq!(after.overrides()["MT"][0], "191.000000000000000000001");
    let replay = SourceOwner::new(loaded, &backend, Some(after.overrides())).unwrap();
    assert_eq!(
        replay.snapshot().unwrap().values().to_decimal(),
        after.values().to_decimal()
    );
}
