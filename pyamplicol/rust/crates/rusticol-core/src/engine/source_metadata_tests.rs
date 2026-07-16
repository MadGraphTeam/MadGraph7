// SPDX-License-Identifier: 0BSD

use super::*;
use serde_json::{Value, json};

fn source_value(
    particle_id: i32,
    anti_particle_id: i32,
    source_orientation: &str,
    wavefunction_kind: &str,
    dimension: usize,
) -> Value {
    json!({
        "source_id": 0,
        "current_id": 0,
        "current_component_start": 0,
        "current_component_stop": dimension,
        "value_slot": {
            "value_slot_id": 0,
            "current_id": 0,
            "variant": "source",
            "component_start": 0,
            "component_stop": dimension,
            "dimension": dimension,
        },
        "source_parameter_start": 0,
        "source_parameter_stop": dimension,
        "leg_label": 1,
        "input_momentum_slot": 0,
        "side": "final",
        "crossing": "identity",
        "physical_pdg": particle_id,
        "outgoing_pdg": particle_id,
        "particle_id": particle_id,
        "anti_particle_id": anti_particle_id,
        "source_kind": "external-wavefunction",
        "wavefunction_kind": wavefunction_kind,
        "source_orientation": source_orientation,
        "source_helicity": 1,
        "chirality": 1,
        "spin_state": 1,
        "dimension": dimension,
        "helicity_ancestry": 1,
        "color_state": {},
    })
}

fn source_record(
    particle_id: i32,
    anti_particle_id: i32,
    source_orientation: &str,
    wavefunction_kind: &str,
    dimension: usize,
) -> GenericSourceRecordManifest {
    serde_json::from_value(source_value(
        particle_id,
        anti_particle_id,
        source_orientation,
        wavefunction_kind,
        dimension,
    ))
    .expect("source manifest record")
}

#[test]
fn source_manifest_requires_structural_orientation() {
    let mut missing = source_value(810_001, -810_001, "particle", "fermion", 2);
    missing
        .as_object_mut()
        .expect("source object")
        .remove("source_orientation");

    let error = serde_json::from_value::<GenericSourceRecordManifest>(missing).unwrap_err();

    assert!(error.to_string().contains("source_orientation"));
}

#[test]
fn source_manifest_validation_rejects_self_conjugate_fermions() {
    let source = source_record(810_001, 810_001, "self-conjugate", "fermion", 2);

    let error = validate_source_wavefunction_metadata(0, &source).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("unsupported self-conjugate fermion source")
    );
}

#[test]
fn source_manifest_validation_rejects_family_dimension_mismatches() {
    let source = source_record(910_101, 910_101, "self-conjugate", "vector", 2);

    let error = validate_source_wavefunction_metadata(0, &source).unwrap_err();

    assert!(error.to_string().contains("wavefunction kind \"vector\""));
}

#[test]
fn f64_sources_use_orientation_instead_of_pdg_sign() {
    let source = source_record(810_001, -810_001, "antiparticle", "fermion", 2);
    let point = [[5.0, 3.0, 4.0, 0.0]];
    let mut output = [c64(0.0, 0.0); 2];

    ExecutionRuntime::write_source_wavefunction(&source, 1, &BTreeMap::new(), &point, &mut output)
        .expect("evaluate relabeled antiparticle source");

    assert_eq!(output, ext_antiquark_weyl_array(point[0], 1, 1));
    assert_ne!(output, ext_quark_weyl_array(point[0], 1, 1));
}

#[test]
fn f64_source_mass_uses_explicit_antiparticle_relation() {
    let source = source_record(810_001, 910_002, "particle", "vector", 4);
    let point = [[13.0, 0.0, 0.0, 12.0]];
    let masses = BTreeMap::from([(-810_001, 99.0), (910_002, 5.0)]);
    let mut output = [c64(0.0, 0.0); 4];

    ExecutionRuntime::write_source_wavefunction(&source, 1, &masses, &point, &mut output)
        .expect("evaluate source with explicit antiparticle mass record");

    assert_eq!(output, ext_massive_vector(point[0], 1, 5.0));
}

#[cfg(feature = "symbolica-runtime")]
#[test]
fn high_precision_sources_use_orientation_instead_of_pdg_sign() {
    let source = source_record(810_001, -810_001, "antiparticle", "fermion", 2);
    let point = [[
        DoubleFloat::from(5.0),
        DoubleFloat::from(3.0),
        DoubleFloat::from(4.0),
        DoubleFloat::from(0.0),
    ]];
    let zero = || Complex::new(DoubleFloat::from(0.0), DoubleFloat::from(0.0));
    let mut output = vec![zero(), zero()];

    ExecutionRuntime::write_source_wavefunction_generic(
        &source,
        1,
        &BTreeMap::new(),
        &point,
        &mut output,
    )
    .expect("evaluate high-precision relabeled antiparticle source");

    assert_eq!(output, ext_antiquark_weyl_generic(&point[0], 1, 1));
    assert_ne!(output, ext_quark_weyl_generic(&point[0], 1, 1));
}
