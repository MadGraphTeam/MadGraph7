// SPDX-License-Identifier: 0BSD

use super::*;
use crate::recurrence::direct_plan::tests::valid_plan;

#[test]
fn direct_codec_round_trips_deterministically() {
    let plan = valid_plan();
    let first = encode_recurrence_direct_plan_v2(&plan).unwrap();
    let second = encode_recurrence_direct_plan_v2(&plan).unwrap();
    assert_eq!(first, second);
    assert_eq!(decode_recurrence_direct_plan_v2(&first).unwrap(), plan);
}

#[test]
fn direct_codec_rejects_every_tested_truncation_boundary() {
    let bytes = encode_recurrence_direct_plan_v2(&valid_plan()).unwrap();
    for length in [0, 1, 7, 8, 15, 64, bytes.len() / 2, bytes.len() - 1] {
        let error = decode_recurrence_direct_plan_v2(&bytes[..length]).unwrap_err();
        assert!(
            error.to_string().contains("truncated")
                || error.to_string().contains("unsupported recurrence")
                || error.to_string().contains("cannot fit"),
            "unexpected error at truncation {length}: {error}"
        );
    }
}

#[test]
fn direct_codec_rejects_old_v1_magic_explicitly() {
    let mut bytes = encode_recurrence_direct_plan_v2(&valid_plan()).unwrap();
    bytes[..8].copy_from_slice(b"PACRPLAN");
    let error = decode_recurrence_direct_plan_v2(&bytes).unwrap_err();
    assert!(error.to_string().contains("regenerate with direct-plan v2"));
}

#[test]
fn direct_codec_detects_a_valid_field_mutation_through_layout_digest() {
    let mut bytes = encode_recurrence_direct_plan_v2(&valid_plan()).unwrap();
    // Header offset 20 is point_tile_size. Keep it nonzero and structurally
    // valid so the authenticated layout digest is the rejecting invariant.
    bytes[20..24].copy_from_slice(&128_u32.to_le_bytes());
    let error = decode_recurrence_direct_plan_v2(&bytes).unwrap_err();
    assert!(error.to_string().contains("runtime-layout digest mismatch"));
}

#[test]
fn direct_codec_rejects_nonzero_reserved_fields() {
    let mut bytes = encode_recurrence_direct_plan_v2(&valid_plan()).unwrap();
    bytes[12..16].copy_from_slice(&1_u32.to_le_bytes());
    let error = decode_recurrence_direct_plan_v2(&bytes).unwrap_err();
    assert!(error.to_string().contains("header flags"));
}
