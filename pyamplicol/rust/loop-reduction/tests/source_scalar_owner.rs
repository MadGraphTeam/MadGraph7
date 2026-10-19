// SPDX-License-Identifier: 0BSD
//! Standalone source-owner checks use an actual compiled source-model artifact.
//! No process reducer or Python producer is constructed by this test.

use rusticol_core::loop_execution::LoopBackend;
use rusticol_core::loop_runtime::{ModelBinding, load_loop_runtime_pacbin};
use rusticol_core::source_bindings::{SourceOverrides, SourceOwner, SourceScale};
use rusticol_loop_reduction::runtime::physical::PhysicalBackend;
use std::sync::Arc;

/// Replace only an existing source-value evaluator with an explicit scalar
/// stress control. Its signatures and the normal PACBIN loader are unchanged;
/// no graph, tensor or reduction is constructed or exercised by this fixture.
fn scalar_overflow_control() -> Arc<rusticol_core::loop_runtime::LoadedLoopRuntime> {
    scalar_overflow_fixture(false)
}

fn scalar_overflow_fixture(guard: bool) -> Arc<rusticol_core::loop_runtime::LoadedLoopRuntime> {
    use rusticol_core::loop_runtime::LOOP_RUNTIME_MEMBER;
    use rusticol_core::pacbin::{
        PacbinMemberKind, PacbinWriteMember, PacbinWriteOptions, write_pacbin_atomic,
    };
    let template = load_loop_runtime_pacbin(
        std::env::var("PYAMPLICOL_NATIVE_SOURCE_PACBIN").expect("retained source PACBIN"),
    )
    .unwrap();
    let ModelBinding::Source { owner, .. } = &template.plan().model_binding else {
        panic!("source owner required");
    };
    let python = std::env::var_os("PYAMPLICOL_TEST_PYTHON").unwrap_or_else(|| "python3".into());
    let producer = std::process::Command::new(python)
        .env("SYMBOLICA_HIDE_BANNER", "1")
        .args([
            "-c",
            r#"import json,sys
from symbolica import Expression,S
names=json.loads(sys.argv[1]); count=int(sys.argv[2]); types=json.loads(sys.argv[3])
inputs=[S('source_binding_test::'+name) for name in names]
a,b,c=[inputs[names.index(name)] for name in ('aS','MT','MU_R')]
if sys.argv[4]=='guards':
    from types import SimpleNamespace
    from pyamplicol.loops.scalar_frontend import ScalarExpression,ScalarFunctionLibrary
    library=ScalarFunctionLibrary(SimpleNamespace(name='source_binding_stress',collections={'functions':[]},metadata={}))
    guards=[library.lower('1 / (1 if z.imag == 0 else 0)',locals={'z':symbol})
            for kind,symbol in zip(types,inputs,strict=True) if kind=='real']
    ratio=((a+800).exp()+(b+801).exp())/(c+802).exp()
    first=guards[0]
    guards[0]=ScalarExpression(first.value,first.diagnostic+ratio)
    evaluator=library.evaluator(tuple(guards),tuple(inputs))
else:
    ratio=(a.exp()+b.exp())/c.exp()
    diagnostic=1/(b-1)
    evaluator=Expression.evaluator_multiple([ratio,diagnostic]*count,inputs,n_cores=1,jit_compile=False)
sys.stdout.buffer.write(evaluator.save())
"#,
            &serde_json::to_string(&owner.external_names).unwrap(),
            &owner.outputs.len().to_string(),
            &serde_json::to_string(&owner.external_types).unwrap(),
            if guard { "guards" } else { "values" },
        ])
        .output()
        .expect("launch existing Symbolica scalar producer");
    assert!(
        producer.status.success(),
        "{}",
        String::from_utf8_lossy(&producer.stderr)
    );
    let metadata = template.metadata_json().unwrap();
    let mut members = vec![
        PacbinWriteMember::from_bytes(
            LOOP_RUNTIME_MEMBER,
            PacbinMemberKind::LoopRuntimePlan,
            metadata.as_bytes(),
        )
        .unwrap(),
    ];
    let changed = if guard {
        owner.guards_state.as_ref().expect("real input guard")
    } else {
        &owner.values_state
    };
    for (name, bytes) in template.evaluator_states() {
        members.push(
            PacbinWriteMember::from_bytes(
                name,
                PacbinMemberKind::SymbolicaExactState,
                if name == *changed {
                    &producer.stdout
                } else {
                    bytes
                },
            )
            .unwrap(),
        );
    }
    let path = std::env::temp_dir().join(format!(
        "source-binding-overflow-{}-{guard}.pacbin",
        std::process::id()
    ));
    write_pacbin_atomic(
        &path,
        members,
        PacbinWriteOptions::new(1 << 20, 0o600).unwrap(),
    )
    .unwrap();
    let result = load_loop_runtime_pacbin(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    Arc::new(result)
}

#[test]
#[ignore = "requires PYAMPLICOL_NATIVE_SOURCE_PACBIN and licensed Symbolica scalar producer"]
fn exact_source_alternate_admission_preserves_primary_failure_and_atomic_revision() {
    use rusticol_core::{RusticolError, RusticolErrorKind};
    use std::cell::Cell;

    let loaded = scalar_overflow_control();
    let primary = PhysicalBackend::new("double", None, "interpreted", "generic").unwrap();
    let dd = PhysicalBackend::new("double-double", None, "interpreted", "generic").unwrap();
    let high = PhysicalBackend::new("arbitrary", Some(80), "interpreted", "generic").unwrap();
    let exact_a = "800.00000000000000000000000000000000000000000000000000000001";
    let inputs = changes(&[
        ("aS", exact_a, "0"),
        ("MT", "801", "0"),
        ("MU_R", "802", "0"),
    ]);
    // Actual saved dependency instructions expose nonfinite rows in the lower
    // carriers, rather than an opaque evaluator error or a rounded finite zero.
    for backend in [&primary, &dd] {
        let failed = SourceOwner::validate_exact(&loaded, backend, &inputs)
            .unwrap()
            .unwrap();
        assert_eq!(failed.stage, "model scalar values and active diagnostics");
    }
    assert!(
        SourceOwner::validate_exact(&loaded, &high, &inputs)
            .unwrap()
            .is_none()
    );
    match SourceOwner::new(loaded.clone(), &primary, Some(&inputs)) {
        Err(error) => assert_eq!(error.kind(), RusticolErrorKind::Evaluation),
        Ok(_) => panic!("default primary binding must fail"),
    }
    let calls = Cell::new(0);
    let admit_high =
        |candidate: &SourceOverrides, _: &rusticol_core::source_bindings::SourceBindingFailure| {
            calls.set(calls.get() + 1);
            assert!(SourceOwner::validate_exact(&loaded, &high, candidate)?.is_none());
            Ok(())
        };
    let owner =
        SourceOwner::new_with_alternate(loaded.clone(), &primary, Some(&inputs), admit_high)
            .unwrap();
    let failed = owner.snapshot().unwrap();
    assert_eq!(calls.get(), 1);
    assert!(failed.binding_failure().is_some());
    assert!(failed.arguments().is_err() && failed.values().is_err());
    assert_eq!(failed.overrides()["aS"][0], exact_a);
    let independent = SourceOwner::new(loaded.clone(), &high, Some(failed.overrides())).unwrap();
    let replay = independent
        .reconstruct_snapshot(&high, &owner, &failed)
        .unwrap();
    assert_eq!(
        replay.values().unwrap().to_decimal(),
        independent
            .snapshot()
            .unwrap()
            .values()
            .unwrap()
            .to_decimal()
    );
    assert_eq!(replay.revision(), failed.revision());
    assert_eq!(replay.overrides()["aS"][0], exact_a);
    let mut rounded_inputs = inputs.clone();
    rounded_inputs.insert("aS".into(), ["800".into(), "0".into()]);
    let rounded = SourceOwner::new(loaded.clone(), &high, Some(&rounded_inputs)).unwrap();
    assert_ne!(
        replay.values().unwrap().to_decimal(),
        rounded.snapshot().unwrap().values().unwrap().to_decimal(),
        "the exact sub-binary64 input perturbation must change the AP output"
    );

    let before = owner.snapshot().unwrap();
    let invalid_changes = [
        changes(&[("aS", "not-a-number", "0")]),
        changes(&[("aS", "1", "1e-1000")]),
        changes(&[("unknown", "1", "0")]),
        changes(&[("MB", "1", "0")]),
    ];
    for invalid in invalid_changes {
        assert!(
            owner
                .update_with_alternate(&primary, &invalid, |_, _| {
                    panic!("schema/fixed-tie failures must not retry")
                })
                .is_err()
        );
        assert!(Arc::ptr_eq(&before, &owner.snapshot().unwrap()));
    }
    // An actual active diagnostic pole stays invalid in the higher carrier.
    assert!(
        owner
            .update_with_alternate(&primary, &changes(&[("MT", "1", "0")]), |candidate, _| {
                let failure = SourceOwner::validate_exact(&loaded, &high, candidate)?
                    .expect("active diagnostic failure");
                Err(failure.error())
            })
            .is_err()
    );
    assert!(Arc::ptr_eq(&before, &owner.snapshot().unwrap()));
    // A rejected alternate cannot publish even otherwise valid exact values.
    assert!(
        owner
            .update_with_alternate(&primary, &inputs, |_, _| {
                Err(RusticolError::evaluation("configured alternatives failed"))
            })
            .is_err()
    );
    assert!(Arc::ptr_eq(&before, &owner.snapshot().unwrap()));
    let updated = owner
        .update_with_alternate(&primary, &changes(&[("MT", "803", "0")]), admit_high)
        .unwrap();
    assert_eq!(updated.revision(), 1);
    assert_eq!(before.revision(), 0);
    assert_eq!(updated.overrides()["aS"][0], exact_a);
    assert_eq!(before.overrides()["MT"][0], "801");
    assert!(updated.binding_failure().is_some());
    assert_eq!(calls.get(), 2);
    let rebound_dd = SourceOwner::from_snapshot(&dd, &owner, &before).unwrap();
    let rebound_dd_snapshot = rebound_dd.snapshot().unwrap();
    assert!(rebound_dd_snapshot.binding_failure().is_some());
    assert_eq!(rebound_dd_snapshot.revision(), 0);
    assert_eq!(rebound_dd_snapshot.overrides(), before.overrides());
    assert!(owner.require_snapshot(&rebound_dd_snapshot).is_err());
    assert!(SourceOwner::from_snapshot(&high, &owner, &replay).is_err());
    let foreign = SourceOwner::new(scalar_overflow_control(), &high, Some(&inputs)).unwrap();
    assert!(
        foreign
            .reconstruct_snapshot(&high, &owner, &before)
            .is_err()
    );
}

#[test]
#[ignore = "requires retained source gg PACBIN and licensed scalar producer; synthetic rescue plumbing"]
fn global_source_failure_uses_complete_precision_attempts_and_original_lanes() {
    use rusticol_core::loop_execution::{LoopExecutor, LoopInputs};
    use rusticol_loop_reduction::verification::{
        ArithmeticSpec, SourceBindingValidation, VerificationPolicy, VerificationWorkspace,
        evaluate_verified,
    };
    let pristine = Arc::new(
        load_loop_runtime_pacbin(std::env::var("PYAMPLICOL_NATIVE_SOURCE_PACBIN").unwrap())
            .unwrap(),
    );
    let loaded = scalar_overflow_fixture(true);
    let original_record = match &pristine.plan().model_binding {
        ModelBinding::Source { owner, .. } => owner,
        _ => unreachable!(),
    };
    // Only the real-input guard catalogue changed. Physical outputs, active
    // scalar diagnostics, defaults, kernels and reduction instructions are exact
    // copies. This is failure-routing plumbing, not another physical model.
    for (name, bytes) in pristine.evaluator_states() {
        if Some(name) != original_record.guards_state.as_deref() {
            assert_eq!(bytes, loaded.evaluator_state(&name).unwrap());
        }
    }
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tools/loop_reference/gluon_scattering_qcd.json"
    ))
    .unwrap();
    let momenta = fixture["points"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|point| {
            point.as_array().unwrap().iter().flat_map(|leg| {
                leg.as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.to_string())
            })
        })
        .collect();
    let inputs = LoopInputs {
        momenta,
        scales_squared: ["1000000", "900000", "2100000"].map(String::from).to_vec(),
        point_ids: vec![u64::MAX, 7, 91],
        model_values: vec![],
        spin_sources: Default::default(),
    };
    let primary_spec = ArithmeticSpec {
        arithmetic: "double".into(),
        precision: None,
    };
    let high_specs = [
        ArithmeticSpec {
            arithmetic: "double-double".into(),
            precision: None,
        },
        ArithmeticSpec {
            arithmetic: "arbitrary".into(),
            precision: Some(60),
        },
        ArithmeticSpec {
            arithmetic: "arbitrary".into(),
            precision: Some(80),
        },
    ];
    let policy = VerificationPolicy {
        higher: high_specs.to_vec(),
        relative_tolerance: "1e-10".into(),
        absolute_tolerance: "1e-10".into(),
        verify_successful: false,
    };
    let primary = PhysicalBackend::new("double", None, "interpreted", "fast").unwrap();
    assert!(
        SourceBindingValidation::default()
            .owner(loaded.clone(), &primary, None)
            .is_err()
    );
    let validation = SourceBindingValidation::new(&primary_spec, Some(&policy)).unwrap();
    let owner = validation.owner(loaded.clone(), &primary, None).unwrap();
    assert_eq!(
        owner.snapshot().unwrap().binding_failure().unwrap().stage,
        "real model input type"
    );
    let mut executor = LoopExecutor::with_source_owner(owner.clone(), primary).unwrap();
    let mut workspace = VerificationWorkspace::default();
    let raw = executor
        .evaluate_partial_with_snapshot(inputs.clone(), None, false)
        .unwrap();
    assert!(raw.point_ids.is_empty());
    assert_eq!(
        raw.failures.iter().map(|f| f.point_id).collect::<Vec<_>>(),
        inputs.point_ids
    );
    assert!(
        raw.failures
            .iter()
            .all(|f| f.stage == "real model input type")
    );
    // Empty input does not manufacture failed lanes or require successful model
    // values, but retains the ordinary public result axes.
    let empty = LoopInputs {
        momenta: vec![],
        scales_squared: vec![],
        point_ids: vec![],
        model_values: vec![],
        spin_sources: Default::default(),
    };
    let empty_result = evaluate_verified(
        &mut executor,
        &mut workspace,
        empty,
        &primary_spec,
        &policy,
        false,
    )
    .unwrap();
    assert!(empty_result.result.failures.is_empty() && empty_result.diagnostics.is_empty());
    assert_eq!(empty_result.result.totals.columns(), 19);
    let mut malformed = inputs.clone();
    malformed.momenta[0] = "not-a-number".into();
    assert!(
        executor
            .evaluate_partial_with_snapshot(malformed, None, false)
            .is_err()
    );

    let mut baselines = Vec::new();
    for spec in &high_specs[1..] {
        let backend =
            PhysicalBackend::new(&spec.arithmetic, spec.precision, "interpreted", "generic")
                .unwrap();
        let mut reference = LoopExecutor::with_parameters(pristine.clone(), backend, None).unwrap();
        baselines.push(reference.evaluate(inputs.clone()).unwrap());
    }
    let result = evaluate_verified(
        &mut executor,
        &mut workspace,
        inputs.clone(),
        &primary_spec,
        &policy,
        false,
    )
    .unwrap();
    assert_eq!(result.result.point_ids, inputs.point_ids);
    assert_eq!(result.result.requested_point_ids, inputs.point_ids);
    assert!(result.result.failures.is_empty());
    assert_eq!(result.result.source_revision, Some(0));
    for diagnostic in &result.diagnostics {
        assert_eq!(diagnostic.status, "rescued");
        assert!(diagnostic.accuracy_established);
        assert_eq!(diagnostic.attempts.len(), 4);
        assert_eq!(
            diagnostic
                .attempts
                .iter()
                .map(|a| a.numerical_success)
                .collect::<Vec<_>>(),
            [false, false, true, true]
        );
        assert_eq!(
            diagnostic.agreeing_bits,
            Some([high_specs[1].bits().unwrap(), high_specs[2].bits().unwrap()])
        );
        for attempt in &diagnostic.attempts[..2] {
            assert_eq!(attempt.failures.len(), 1);
            assert_eq!(attempt.failures[0].point_id, diagnostic.point_id);
            assert_eq!(attempt.failures[0].stage, "real model input type");
        }
    }
    let actual = std::iter::once(&result.result.born)
        .chain(result.result.components.iter())
        .chain(std::iter::once(&result.result.totals));
    for (column, value) in actual.enumerate() {
        assert_eq!(value.precision_bits(), primary_spec.bits().unwrap());
        for baseline in &baselines {
            let expected = std::iter::once(&baseline.born)
                .chain(baseline.components.iter())
                .chain(std::iter::once(&baseline.totals))
                .nth(column)
                .unwrap();
            assert!(
                value
                    .agreement_rows(expected, "1e-10", "1e-10")
                    .unwrap()
                    .iter()
                    .all(|(_, agrees)| *agrees)
            );
        }
    }
    // One successful high-precision *source admission* is not a successful
    // complete result pair. A one-level policy must not claim rescue accuracy.
    let one_level = VerificationPolicy {
        higher: vec![high_specs[2].clone()],
        ..policy.clone()
    };
    let singleton = inputs
        .select_points(&[7], pristine.plan().state_legs.len() * 4)
        .unwrap();
    let unverified = evaluate_verified(
        &mut executor,
        &mut workspace,
        singleton,
        &primary_spec,
        &one_level,
        false,
    )
    .unwrap();
    assert!(unverified.result.point_ids.is_empty());
    assert_eq!(unverified.diagnostics[0].status, "accuracy-not-established");
    assert!(!unverified.diagnostics[0].accuracy_established);
    assert_eq!(unverified.diagnostics[0].attempts.len(), 2);
    assert!(unverified.diagnostics[0].attempts[1].numerical_success);
}

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
        let mass = PhysicalBackend::select(after.values().unwrap(), vec![mass_output]).unwrap();
        let expected = backend
            .ingress_complex(&[[exact_mass.into(), "0".into()]], 1, 1, vec![0])
            .unwrap();
        assert_eq!(mass.to_decimal(), expected.to_decimal());
        if arithmetic != "double" {
            assert_ne!(
                mass.to_decimal(),
                PhysicalBackend::select(before.values().unwrap(), vec![mass_output])
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
            rebound_snapshot.values().unwrap().to_decimal(),
            after.values().unwrap().to_decimal()
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
        replay.snapshot().unwrap().values().unwrap().to_decimal(),
        after.values().unwrap().to_decimal()
    );
}
