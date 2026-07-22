// SPDX-License-Identifier: 0BSD

use std::cmp::Ordering;

use super::*;

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::new([byte; 32]).expect("test digest must be nonzero")
}

fn momentum(slot: u32) -> CanonicalMomentumLinearForm {
    CanonicalMomentumLinearForm::new(vec![MomentumTerm {
        source_slot: slot,
        coefficient: 1,
    }])
    .expect("test momentum is canonical")
}

#[test]
fn abi_constants_are_frozen() {
    assert_eq!(RECURRENCE_TEMPLATE_ABI, "pyamplicol-recurrence-template-v1");
    assert_eq!(
        RECURRENCE_BUILDER_INPUT_ABI,
        "pyamplicol-recurrence-builder-input-v1"
    );
    assert_eq!(
        RECURRENCE_BUILDER_RESULT_ABI,
        "pyamplicol-recurrence-builder-result-v1"
    );
    assert_eq!(RECURRENCE_PLAN_ABI, "pyamplicol-recurrence-plan-v1");
    assert_eq!(
        RECURRENCE_RUNTIME_LAYOUT_ABI,
        "pyamplicol-recurrence-runtime-layout-v1"
    );
    assert_eq!(
        RECURRENCE_RUNTIME_KIND,
        "pyamplicol-runtime-recurrence-execution"
    );
    assert_eq!(
        RECURRENCE_RUNTIME_CAPABILITY,
        "rusticol.recurrence-runtime.complex-f64.v1"
    );
    assert_eq!(
        RECURRENCE_LC_COLOR_CAPABILITY,
        "rusticol.recurrence-color.lc.v1"
    );
}

#[test]
fn exact_rationals_parse_reduce_and_canonicalize_zero() {
    let value = ExactRational::parse_parts("-42", "56").unwrap();
    assert_eq!(value.numerator(), -3);
    assert_eq!(value.denominator(), 4);
    assert_eq!(value.to_string(), "-3/4");
    assert_eq!(
        "000/009".parse::<ExactRational>().unwrap(),
        ExactRational::ZERO
    );
    assert!(ExactRational::parse_parts("1", "0").is_err());
    assert!(ExactRational::parse_parts("1", "-2").is_err());
    assert!("1/2/3".parse::<ExactRational>().is_err());
}

#[test]
fn exact_aggregation_distinguishes_binary64_collision() {
    let one = ExactRational::ONE;
    let epsilon = ExactRational::new(1, 1_i128 << 54).unwrap();
    let sum = one.checked_add(epsilon).unwrap();
    assert_ne!(sum, one);
    assert_eq!(sum.checked_sub(one).unwrap(), epsilon);
    assert_eq!(1.0_f64 + 2.0_f64.powi(-54), 1.0_f64);
}

#[test]
fn exact_ordering_avoids_cross_multiplication_overflow() {
    let maximum = i128::MAX;
    let left = ExactRational::new(maximum - 1, maximum).unwrap();
    let right = ExactRational::new(maximum - 2, maximum - 1).unwrap();
    assert_eq!(left.cmp(&right), Ordering::Greater);
}

#[test]
fn exact_arithmetic_fails_closed_on_overflow() {
    let large = ExactRational::new(i128::MAX, 1).unwrap();
    assert!(large.checked_add(ExactRational::ONE).is_err());
    assert!(
        large
            .checked_mul(ExactRational::new(2, 1).unwrap())
            .is_err()
    );
    assert!(ExactRational::ONE.checked_div(ExactRational::ZERO).is_err());
}

#[test]
fn binary64_conversion_is_exact_or_rejected() {
    assert_eq!(
        ExactRational::from_f64_exact(0.1).unwrap(),
        ExactRational::new(3_602_879_701_896_397, 36_028_797_018_963_968).unwrap()
    );
    assert_eq!(
        ExactRational::from_f64_exact(-0.0).unwrap(),
        ExactRational::ZERO
    );
    assert!(ExactRational::from_f64_exact(f64::INFINITY).is_err());
    assert!(ExactRational::from_f64_exact(f64::from_bits(1)).is_err());
}

#[test]
fn exact_complex_arithmetic_is_canonical() {
    let left = ExactComplexRational::parse_parts("1", "2", "1", "3").unwrap();
    let right = ExactComplexRational::parse_parts("2", "5", "-1", "7").unwrap();
    let product = left.checked_mul(right).unwrap();
    assert_eq!(product.real(), ExactRational::new(26, 105).unwrap());
    assert_eq!(product.imag(), ExactRational::new(13, 210).unwrap());
    assert_eq!(product.checked_div(right).unwrap(), left);
}

#[test]
fn checked_ranges_reject_overflow_gaps_and_bounds() {
    assert!(CheckedTableRange::new(u64::MAX, 1).end("test").is_err());
    assert!(validate_ranges_within("test", &[CheckedTableRange::new(2, 2)], 3).is_err());
    assert!(
        validate_packed_ranges(
            "test",
            &[CheckedTableRange::new(0, 1), CheckedTableRange::new(2, 1)],
            3
        )
        .is_err()
    );
    validate_packed_ranges(
        "test",
        &[CheckedTableRange::new(0, 1), CheckedTableRange::new(1, 2)],
        3,
    )
    .unwrap();
}

#[test]
fn table_validators_reject_mismatched_columns_and_references() {
    assert!(validate_equal_column_lengths("table", &[("a", 2), ("b", 1)]).is_err());
    assert!(validate_u32_references(&[0, 2], 2, "reference").is_err());
    validate_equal_column_lengths("table", &[("a", 2), ("b", 2)]).unwrap();
    validate_u32_references(&[0, 1], 2, "reference").unwrap();
}

#[test]
fn multiword_masks_retain_high_bits_and_validate_canonical_form() {
    let ranges = [CheckedTableRange::new(0, 0), CheckedTableRange::new(0, 2)];
    let populations = [0, 2];
    let words = [1, 1_u64 << 63];
    let catalog = MultiwordMaskCatalogView {
        ranges: &ranges,
        populations: &populations,
        words: &words,
    };
    catalog.validate(true).unwrap();
    assert!(catalog.contains(1, 0).unwrap());
    assert!(catalog.contains(1, 127).unwrap());
    assert!(!catalog.contains(1, 126).unwrap());

    let invalid_words = [1, 0];
    let invalid = MultiwordMaskCatalogView {
        ranges: &[CheckedTableRange::new(0, 2)],
        populations: &[1],
        words: &invalid_words,
    };
    assert!(invalid.validate(false).is_err());
}

fn header_for(
    strategy: RecurrenceStrategy,
    sections: &[CanonicalInputSection<'_>],
) -> RecurrenceBuilderInputHeader {
    let placeholder = digest(9);
    let mut header = RecurrenceBuilderInputHeader::canonical(
        strategy,
        sections.len() as u32,
        digest(1),
        digest(2),
        digest(3),
        placeholder,
    );
    header.input_digest = canonical_input_digest(&header, sections).unwrap();
    header
}

#[test]
fn input_header_digest_is_deterministic_and_authenticated() {
    let sections = [
        CanonicalInputSection {
            name: "catalogs/states",
            row_width: 2,
            row_count: 2,
            bytes: &[1, 2, 3, 4],
        },
        CanonicalInputSection {
            name: "process/legs",
            row_width: 1,
            row_count: 2,
            bytes: &[5, 6],
        },
    ];
    let header = header_for(RecurrenceStrategy::TopologyReplay, &sections);
    validate_header_and_sections(&header, &sections).unwrap();
    assert_eq!(
        canonical_input_digest(&header, &sections).unwrap(),
        header.input_digest
    );

    let modified = [
        sections[0],
        CanonicalInputSection {
            bytes: &[5, 7],
            ..sections[1]
        },
    ];
    assert!(validate_header_and_sections(&header, &modified).is_err());
}

#[test]
fn input_header_rejects_noncanonical_sections_and_abi_drift() {
    let sections = [
        CanonicalInputSection {
            name: "z",
            row_width: 1,
            row_count: 1,
            bytes: &[0],
        },
        CanonicalInputSection {
            name: "a",
            row_width: 1,
            row_count: 1,
            bytes: &[0],
        },
    ];
    let mut header = RecurrenceBuilderInputHeader::canonical(
        RecurrenceStrategy::AllFlowUnion,
        2,
        digest(1),
        digest(2),
        digest(3),
        digest(4),
    );
    assert!(validate_header_and_sections(&header, &sections).is_err());
    let canonical_section = [CanonicalInputSection {
        name: "a",
        row_width: 1,
        row_count: 1,
        bytes: &[0],
    }];
    let wrong_count_header = RecurrenceBuilderInputHeader::canonical(
        RecurrenceStrategy::AllFlowUnion,
        2,
        digest(1),
        digest(2),
        digest(3),
        digest(4),
    );
    assert!(canonical_input_digest(&wrong_count_header, &canonical_section).is_err());
    header.builder_input_abi = "future-abi".to_owned();
    assert!(header.validate_identity().is_err());
}

#[test]
fn strategy_and_digest_validation_fail_closed() {
    assert_eq!(
        RecurrenceStrategy::try_from(0).unwrap(),
        RecurrenceStrategy::TopologyReplay
    );
    assert_eq!(
        RecurrenceStrategy::try_from(1).unwrap(),
        RecurrenceStrategy::AllFlowUnion
    );
    assert!(RecurrenceStrategy::try_from(2).is_err());
    assert!(SemanticDigest::new([0; 32]).is_err());
}

#[test]
fn current_core_key_preserves_every_model_visible_axis() {
    let key = CurrentCoreKey::new(
        digest(1),
        RecurrenceNodeKind::Current,
        7,
        vec![0, 2],
        CanonicalMomentumLinearForm::new(vec![
            MomentumTerm {
                source_slot: 0,
                coefficient: 1,
            },
            MomentumTerm {
                source_slot: 2,
                coefficient: -1,
            },
        ])
        .unwrap(),
        -1,
        vec![2, -2],
        3,
        vec![1, 2],
        None,
        Some(9),
    )
    .unwrap();
    let changed = CurrentCoreKey::new(
        digest(1),
        RecurrenceNodeKind::Current,
        7,
        vec![0, 2],
        key.momentum().clone(),
        1,
        key.flavour_flow().to_vec(),
        key.quantum_number_flow_id(),
        key.coupling_orders().to_vec(),
        None,
        Some(9),
    )
    .unwrap();
    assert_ne!(key, changed);
    assert!(
        CurrentCoreKey::new(
            digest(1),
            RecurrenceNodeKind::Current,
            7,
            vec![2, 0],
            momentum(0),
            0,
            vec![],
            0,
            vec![],
            None,
            None,
        )
        .is_err()
    );
}

#[test]
fn contribution_key_validates_aligned_parent_contracts() {
    let key = ContributionKey::new(
        1,
        vec![2, 3],
        vec![4, 5],
        vec![momentum(0), momentum(1)],
        6,
        7,
        8,
        digest(9),
        10,
    )
    .unwrap();
    assert_eq!(key.parent_value_class_ids(), &[2, 3]);
    assert!(
        ContributionKey::new(
            1,
            vec![2, 3],
            vec![4],
            vec![momentum(0), momentum(1)],
            6,
            7,
            8,
            digest(9),
            10,
        )
        .is_err()
    );
}
