// Proposed tests inside vendor/ninjars/src/lib.rs existing cfg(test) module.
// Expected values are exact integer dyadics; no engine/context is constructed.
#[test]
fn mp_transport_preserves_binary128_finite_dyadics_and_sign() {
    use symbolica::domains::rational::Rational;
    let cases: &[(&str, &str, i128, i32, bool)] = &[
        ("+zero", "0.000000000000000000000000000000000000000E+39", 0_i128, 0, false),
        ("-zero", "-0.000000000000000000000000000000000000000E+39", 0_i128, 0, true),
        ("+min_subnormal", "6.475175119438025110924438958227646552500E-4966", 1_i128, -16494, false),
        ("-min_subnormal", "-6.475175119438025110924438958227646552500E-4966", 1_i128, -16494, true),
        ("+max_subnormal", "3.362103143112093506262677817321751955081E-4932", 5192296858534827628530496329220095_i128, -16494, false),
        ("-max_subnormal", "-3.362103143112093506262677817321751955081E-4932", 5192296858534827628530496329220095_i128, -16494, true),
        ("+min_normal", "3.362103143112093506262677817321752602598E-4932", 1_i128, -16382, false),
        ("-min_normal", "-3.362103143112093506262677817321752602598E-4932", 1_i128, -16382, true),
        ("+max_finite", "1.189731495357231765085759326628007016196E+4932", 10384593717069655257060992658440191_i128, 16271, false),
        ("-max_finite", "-1.189731495357231765085759326628007016196E+4932", 10384593717069655257060992658440191_i128, 16271, true),
        ("+beyond_dp", "1.000000000000000000000000000000788860905E+0", 1267650600228229401496703205377_i128, -100, false),
        ("-beyond_dp", "-1.000000000000000000000000000000788860905E+0", 1267650600228229401496703205377_i128, -100, true),
        ("+last_binary128_bit", "1.000000000000000000000000000000000192593E+0", 5192296858534827628530496329220097_i128, -112, false),
        ("-last_binary128_bit", "-1.000000000000000000000000000000000192593E+0", 5192296858534827628530496329220097_i128, -112, true),
    ];
    for &(name, text, significand, exponent, negative) in cases {
        let mut expected = Rational::from(significand);
        let scale = Rational::from(2).pow(u64::from(exponent.unsigned_abs()));
        expected = if exponent < 0 { expected / scale } else { expected * scale };
        if negative { expected = -expected; }
        for precision in [256, 384] {
            let parsed = parse_record_at_precision(&record(text), precision).unwrap();
            assert_eq!(parsed.prec(), precision, "{name}");
            assert_eq!(parsed.to_rational(), expected, "{name}");
            assert_eq!(parsed.is_sign_negative(), negative, "{name}");
            let formatted = format!("{parsed:.39e}");
            assert!(formatted.len() <= RECORD);
            let back = Float::parse(&formatted, Some(113)).unwrap();
            assert_eq!(back.to_rational(), expected, "output {name}");
            assert_eq!(back.is_sign_negative(), negative, "output {name}");
        }
    }
    for precision in [256, 384] {
        for text in ["NaN", "nan", "+inf", "-inf", "not-a-number"] {
            assert!(parse_record_at_precision(&record(text), precision).is_err());
        }
    }
}

#[test]
fn mp_transport_preserves_complex_branch_lips() {
    for precision in [256, 384] {
        let real = parse_record_at_precision(&record("-1"), precision).unwrap();
        let plus = parse_record_at_precision(&record("0"), precision).unwrap();
        let minus = parse_record_at_precision(&record("-0"), precision).unwrap();
        let upper = Complex::new(real.clone(), plus);
        let lower = Complex::new(real, minus);
        let upper_sqrt = upper.sqrt();
        let lower_sqrt = lower.sqrt();
        assert_eq!(upper_sqrt.im, Float::with_val(precision, 1));
        assert_eq!(lower_sqrt.im, Float::with_val(precision, -1));
        assert!(!upper.log().im.is_sign_negative());
        assert!(lower.log().im.is_sign_negative());
    }
}
