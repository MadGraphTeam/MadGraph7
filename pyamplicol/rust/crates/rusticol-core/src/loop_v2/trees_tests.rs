// SPDX-License-Identifier: 0BSD

use super::*;
use crate::loop_v2::plan::KernelTerm;

/// A kernel of `terms` monomials of degree up to 5 over `inputs` lanes,
/// with constants and inputs from a fixed xorshift stream (signed,
/// spread over a few binades).
fn case(terms: usize, inputs: usize) -> (Vec<KernelTerm>, Vec<Complex<f64>>, Vec<f64>) {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let value = |next: &mut dyn FnMut() -> u64| {
        let r = next();
        let mantissa = (r >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        mantissa * f64::powi(2.0, (r % 7) as i32 - 3)
    };
    let constants = (0..terms)
        .map(|_| Complex::new(value(&mut next), value(&mut next)))
        .collect();
    let kernel = (0..terms)
        .map(|t| KernelTerm {
            constant: t,
            monomial: (0..1 + next() % 4)
                .map(|_| ((next() % inputs as u64) as usize, 1 + (next() % 2) as u32))
                .collect(),
        })
        .collect();
    let x = (0..inputs * 16).map(|_| value(&mut next)).collect();
    (kernel, constants, x)
}

#[test]
fn wide_closure_kernels_are_bitwise_the_baseline() {
    let (kernel, constants, x) = case(72, 24);
    let baseline = kernel_lanes::<f64, 8>(&kernel, &constants, &x, &0.0);
    let bits = |(re, im): ([f64; 8], [f64; 8])| (re.map(f64::to_bits), im.map(f64::to_bits));
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: the CPU reports AVX2.
            let wide = unsafe { kernel_lanes_avx2::<f64, 8>(&kernel, &constants, &x, &0.0) };
            assert_eq!(bits(wide), bits(baseline));
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: the CPU reports AVX-512F.
            let wide = unsafe { kernel_lanes_avx512::<f64, 8>(&kernel, &constants, &x, &0.0) };
            assert_eq!(bits(wide), bits(baseline));
        }
    }
    // Each lane is the scalar evaluation of `Complex<f64>`.
    for lane in 0..8 {
        let mut value = Complex::new(0.0, 0.0);
        for term in &kernel {
            let mut product = constants[term.constant];
            for &(input, power) in &term.monomial {
                for _ in 0..power {
                    product *= Complex::new(x[input * 16 + lane], x[input * 16 + 8 + lane]);
                }
            }
            value += product;
        }
        assert_eq!(baseline.0[lane].to_bits(), value.re.to_bits());
        assert_eq!(baseline.1[lane].to_bits(), value.im.to_bits());
    }
}

#[test]
fn lane_runs_cover_the_batch_in_the_fewest_chunks() {
    for points in 0..40 {
        let runs = lane_runs(points);
        let mut next = 0;
        let mut chunks = 0;
        for &(first, count, lanes) in &runs {
            assert_eq!(first, next);
            assert!(
                count > 0 && lanes.is_power_of_two() && lanes <= 8,
                "{runs:?}"
            );
            next += count;
            chunks += count.div_ceil(lanes);
        }
        assert_eq!(next, points);
        assert_eq!(chunks, points.div_ceil(8), "{points}: {runs:?}");
    }
    assert_eq!(lane_runs(9), vec![(0, 8, 8), (8, 1, 1)]);
    assert_eq!(lane_runs(11), vec![(0, 8, 8), (8, 3, 4)]);
    assert_eq!(lane_runs(7), vec![(0, 7, 8)]);
    assert_eq!(lane_runs(16), vec![(0, 16, 8)]);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn the_closure_simd_cap_only_lowers_the_width() {
    assert_eq!(Wide::cap(Some("baseline")), Wide::Baseline);
    assert_eq!(Wide::cap(Some("avx2")), Wide::Avx2);
    assert_eq!(Wide::cap(Some("avx512")), Wide::Avx512);
    assert_eq!(Wide::cap(Some("AVX2")), Wide::Avx512);
    assert_eq!(Wide::cap(None), Wide::Avx512);
    // Other carriers take the baseline whatever the cap.
    assert_eq!(Wide::detect::<f32>(), Wide::Baseline);
}

#[test]
fn carrier_values_read_exactly() {
    // A derived mass is bound to its carrier value's exact rational (its
    // square is the legs' invariant): every carrier reads back exactly.
    let third = parse_rational("1/3").unwrap();
    let double = cast(&0.0_f64, &third);
    assert_eq!(cast(&0.0_f64, &double.exact().unwrap()), double);
    assert_ne!(double.exact().unwrap(), third);
    let pair = cast(&DoubleFloat::from(0.0), &third);
    let parts = pair.into_inner();
    assert_ne!(parts.lo(), 0.0);
    assert_ne!(pair.exact().unwrap(), parts.hi().exact().unwrap());
    assert_eq!(cast(&DoubleFloat::from(0.0), &pair.exact().unwrap()), pair);
    let zero = Float::with_val(200, 0);
    let float = cast(&zero, &third);
    assert_eq!(cast(&zero, &float.exact().unwrap()), float);
    assert!(f64::NAN.exact().is_err());
}

#[test]
fn plan_masses_read_inputs_or_outputs() {
    // A plan mass is massless, an external parameter's index or
    // `["output", o]`, a derived mass (`load_parameters` checks the tag, the
    // output and that it is in `positive`).
    let masses: Vec<Option<MassFile>> =
        serde_json::from_str(r#"[null, 2, ["output", 5]]"#).unwrap();
    assert!(masses[0].is_none());
    assert!(matches!(masses[1], Some(MassFile::Input(2))));
    assert!(matches!(&masses[2], Some(MassFile::Output(tag, 5)) if tag == "output"));
}

#[test]
fn complex_mass_master_guard_refuses_widths_below_1e150_of_the_mass() {
    // M^2 - i M Gamma at M = 91: a physical width and a zero width pass, 0 <
    // |Im| / Re < 1e-150 refuses.
    let re = Rational::from(8281);
    let physical = Rational::from((-227, 1));
    let tiny = Rational::from((-1, 1)) / Rational::from(10).pow(160);
    assert!(require_complex_mass_master(0, &re, &physical).is_ok());
    assert!(require_complex_mass_master(1, &re, &Rational::from(0)).is_ok());
    let error = require_complex_mass_master(3, &re, &tiny).unwrap_err();
    assert_eq!(error.kind(), crate::RusticolErrorKind::ModelParameter);
    let text = error.to_string();
    assert!(
        text.starts_with(
            "complex-mass-master: the complex pole mass m = 91.000000 (m^2 = 8.281e3 \
             -1e-160 i; loop mass 3)"
        ),
        "{text}"
    );
}

#[test]
fn the_complex_mass_master_guard_is_double_double_only() {
    assert!(<DoubleFloat as TreeCarrier>::COMPLEX_MASS_MASTER_GUARD);
    assert!(!<f64 as TreeCarrier>::COMPLEX_MASS_MASTER_GUARD);
    assert!(!<Float as TreeCarrier>::COMPLEX_MASS_MASTER_GUARD);
}

#[test]
fn a_plane_takes_the_exact_side_of_a_cut() {
    // sqrt(-x) through the plane: SymJIT negates a real into a -0
    // imaginary part where Symbolica's evaluator (and exact arithmetic) has
    // +0; the translation canonicalises a cut argument's zero
    // (`symjit_plane::canonical_cut_row`), so on the cut (x > 0) and off it
    // the plane equals the exact evaluator, at -O2 and level 0, with no
    // run-time check (M-SM MS-MASTERS).
    use symbolica::atom::AtomCore;
    let _unlock = crate::licence::unlock();
    let x = symbolica::parse!("pyamplicol_trees_tests::x");
    let evaluator = symbolica::parse!("sqrt(-pyamplicol_trees_tests::x)")
        .evaluator(&[x])
        .build()
        .expect("an evaluator");
    let binary64 = evaluator.map_coeff(&|r| Complex::new(r.re.to_f64(), r.im.to_f64()));
    let mut exact = Kernel::Exact(binary64.clone().into_batch_evaluator());
    for level in [2, 0] {
        let (bytes, inputs, outputs) =
            crate::engine::compile_symbolica_evaluator_to_plane_application_bytes(
                &binary64, level, false,
            )
            .expect("a plane application");
        let mut plane = Kernel::Plane(Box::new(
            BinaryPlane::load(&bytes, (inputs, outputs)).expect("loads"),
        ));
        for x in [-4.0, 4.0] {
            let input = [Complex::new(x, 0.0); 2];
            let rows = |kernel: &mut Kernel<f64>| {
                let mut rows = [Complex::new(0.0, 0.0); 2];
                kernel.evaluate_batch(&input, &mut rows, 2).expect("runs");
                rows
            };
            let (got, expected) = (rows(&mut plane), rows(&mut exact));
            assert_eq!(got, expected, "x = {x}, level {level}");
            assert!(x < 0.0 || got[0].im > 0.0, "x = {x}: +i sqrt(x)");
        }
    }
}

#[test]
fn scale_dependent_shared_outputs_are_the_recorded_born_constant_and_mass_ones() {
    // Generation records every output that reads the scale
    // (`native_trees._scale_outputs`); the batch-shared ones among them are
    // the Born couplings, kernel constants and derived (external and loop)
    // masses. Counterterm couplings (the usual NLO models' MU_R) are not.
    let parameters = |scale_outputs: Option<Vec<(usize, String)>>| Parameters {
        names: vec!["MU_R".into(), "aS".into()],
        defaults: vec![Rational::from(91), Rational::from(0)],
        saved: Vec::new(),
        plane: None,
        outputs: 6,
        model: Vec::new(),
        counterterm_model: Vec::new(),
        couplings: vec![0],
        counterterm_couplings: vec![1],
        finite: Vec::new(),
        constants: vec![2],
        positive: vec![3, 4],
        masses: Vec::new(),
        derived_masses: vec![(6, 3)],
        plan_masses: vec![None, Some(MassSource::Output(4))],
        scale: Some(0),
        scale_outputs,
        photons: None,
    };
    let named = |indices: &[usize]| -> Option<Vec<(usize, String)>> {
        Some(
            indices
                .iter()
                .map(|&o| (o, format!("output {o}")))
                .collect(),
        )
    };
    // An artifact written before the record: unknown.
    assert_eq!(parameters(None).scale_dependent_shared(), None);
    // NLOCT: only a counterterm coupling (and an unrelated output) reads it.
    assert_eq!(
        parameters(named(&[1, 5])).scale_dependent_shared(),
        Some(Vec::<&str>::new())
    );
    // A running coupling and an MSbar mass: the Born coupling, a constant,
    // the derived external mass and the loop mass are listed, in output order.
    let running = parameters(named(&[0, 1, 2, 3, 4]));
    assert_eq!(
        running.scale_dependent_shared(),
        Some(vec!["output 0", "output 2", "output 3", "output 4"])
    );
}

#[test]
fn slot_sets_without_a_finite_slot_are_pole_work() {
    // Slot table (epsilon powers): two finite slots, a 1/eps and a 1/eps^2
    // slot. masks[0] is every slot, then one set per slot, then a pole pair.
    let powers = [0, -1, 0, -2];
    let masks = vec![
        vec![0, 1, 2, 3],
        vec![0],
        vec![1],
        vec![2],
        vec![3],
        vec![1, 3],
    ];
    assert_eq!(
        pole_slot_sets(&masks, &powers).unwrap(),
        vec![false, false, true, false, true, true]
    );
    // A pole-only table skips every current; a finite-only one none.
    assert_eq!(
        pole_slot_sets(&[vec![0, 1]], &[-1, -2]).unwrap(),
        vec![true]
    );
    assert_eq!(pole_slot_sets(&[vec![0, 1]], &[0, 0]).unwrap(), vec![false]);
    // Malformed sets are refused: none, an empty set, slots out of order or
    // repeated, a slot outside the table.
    for bad in [
        vec![],
        vec![vec![]],
        vec![vec![1, 0]],
        vec![vec![0, 0]],
        vec![vec![4]],
    ] {
        assert!(pole_slot_sets(&bad, &powers).is_err(), "{bad:?}");
    }
}

#[test]
fn plane_mismatch_names_the_output_row_and_part() {
    // Two rows of three outputs; rounding-level differences agree.
    let expected: Vec<Complex<f64>> = (0..6)
        .map(|k| Complex::new(1.0 + k as f64, -0.5 * k as f64))
        .collect();
    let mut got = expected.clone();
    got[4].re += 1e-14;
    assert_eq!(first_mismatch(&expected, &got, 3), None);
    // A sign on a branch cut: output 1 of row 1 (value 4), imaginary part.
    got[4].im = -got[4].im;
    let mismatch = first_mismatch(&expected, &got, 3).expect("a mismatch");
    assert_eq!(
        mismatch,
        PlaneMismatch {
            output: 1,
            probe: PlaneProbe::Row {
                row: 1,
                real_axis: false,
                imaginary: true,
                plane: 2.0,
                exact: -2.0
            }
        }
    );
    let text = mismatch.to_string();
    assert!(
        text.contains("output 1 (imaginary part, complex probe row 1"),
        "{text}"
    );
    assert!(!text.contains("exact evaluator;"), "{text}");
    // The rounding floor is the output's own scale: a small output's error
    // is not hidden by a large output (row 0's output 2 at 1e12), while a
    // part that cancels to rounding against its own size agrees.
    let mut expected = vec![
        Complex::new(1e-3, 0.0),
        Complex::new(2.0, 0.0),
        Complex::new(1e12, 0.0),
    ];
    let mut got = expected.clone();
    got[0].re = 1.5e-3;
    let mismatch = first_mismatch(&expected, &got, 3).expect("a mismatch");
    assert_eq!(mismatch.output, 0);
    got[0].re = 1e-3;
    got[1].im = 1e-13;
    assert_eq!(first_mismatch(&expected, &got, 3), None);
    expected[1].im = 0.0;
    got[1].im = 3e-12;
    assert_eq!(first_mismatch(&expected, &got, 3).map(|m| m.output), Some(1));
    let text = PlaneMismatch {
        output: 3,
        probe: PlaneProbe::NoRealPoint,
    }
    .to_string();
    assert!(text.starts_with("output 3 reads a branch-cut"), "{text}");
}

#[test]
fn off_axis_probes_are_distinct_and_off_the_real_axis() {
    // Two rows of five inputs: every imaginary part at least 0.25 in
    // magnitude, positive in the first row and negative in the second, every
    // value distinct, real parts in (0.5, 1.5).
    let probes = off_axis_probes(5, 2);
    assert_eq!(probes.len(), 10);
    for (k, z) in probes.iter().enumerate() {
        assert!(z.re > 0.5 && z.re < 1.5, "{z:?}");
        assert!(z.im.abs() >= 0.25 && z.im.abs() <= 0.75, "{z:?}");
        assert_eq!(z.im > 0.0, k < 5, "{z:?}");
    }
    for (i, a) in probes.iter().enumerate() {
        for b in &probes[i + 1..] {
            assert!(a.re != b.re && a.im != b.im, "{a:?} {b:?}");
        }
    }
}

#[test]
fn a_probe_row_mismatch_falls_back_to_level_zero() {
    // `level_zero_fallback` (`compile_plane_application_checked`): SymJIT's
    // optimised SIMD code gave NaN where level 0 is right (CM05, M-SM
    // MS-MASTERS, DESIGN risk 13): a plane that fails the check at a probe
    // row (here the optimised one, "O2") is compiled again at level 0 and
    // kept if that one passes; refused with the first mismatch when both
    // fail; never recompiled for a program without a real point, or when
    // the first compilation was already at level 0.
    let row = PlaneMismatch {
        output: 7,
        probe: PlaneProbe::Row {
            row: 0,
            real_axis: true,
            imaginary: false,
            // A wrong value (NaN in CM05; not NaN here: PartialEq compares).
            plane: 0.5,
            exact: 1.0,
        },
    };
    let fails = |bad: &'static [&'static str], mismatch: PlaneMismatch| {
        move |plane: &&str| -> RusticolResult<Option<PlaneMismatch>> {
            Ok(bad.contains(plane).then(|| mismatch.clone()))
        }
    };
    // The optimised plane fails, level 0 passes: level 0 is kept.
    assert_eq!(
        level_zero_fallback("O2", 2, || Ok("O0"), fails(&["O2"], row.clone())).unwrap(),
        Ok("O0")
    );
    // Both fail: refused, with the optimised plane's mismatch.
    let mut seen = Vec::new();
    let verdict = level_zero_fallback(
        "O2",
        2,
        || Ok("O0"),
        |plane: &&str| {
            seen.push(*plane);
            Ok(Some(PlaneMismatch {
                output: seen.len(),
                ..row.clone()
            }))
        },
    )
    .unwrap();
    assert_eq!(
        verdict,
        Err(PlaneMismatch {
            output: 1,
            ..row.clone()
        })
    );
    assert_eq!(seen, ["O2", "O0"]);
    // A passing plane is kept without a recompilation.
    assert_eq!(
        level_zero_fallback(
            "O2",
            2,
            || panic!("no recompilation"),
            fails(&[], row.clone())
        )
        .unwrap(),
        Ok("O2")
    );
    // No real point to certify a side: no level certifies it.
    let no_point = PlaneMismatch {
        output: 3,
        probe: PlaneProbe::NoRealPoint,
    };
    assert_eq!(
        level_zero_fallback(
            "O2",
            2,
            || panic!("no recompilation"),
            fails(&["O2"], no_point.clone())
        )
        .unwrap(),
        Err(no_point)
    );
    // Already at level 0: refused as it is.
    assert_eq!(
        level_zero_fallback(
            "O0",
            0,
            || panic!("no recompilation"),
            fails(&["O0"], row.clone())
        )
        .unwrap(),
        Err(row)
    );
}

#[test]
fn single_row_checks_measure_against_every_rows_scale() {
    // The plane check's single-row SIMD calls (`verify_plane`) compare a row
    // with the scales of all its rows: output 0 is O(1) in row 0 and cancels
    // to rounding in row 1, where the plane's -6.6e-16 against the exact
    // -1.67e-15 is rounding of an O(1) output, not a mistranslation.
    let expected = [Complex::new(1.0, 0.0), Complex::new(-1.67e-15, 0.0)];
    let got = [Complex::new(-6.6e-16, 0.0)];
    let scales = output_scales(&expected, 1);
    assert_eq!(first_mismatch_at(&expected[1..], &got, 1, &scales), None);
    assert!(first_mismatch(&expected[1..], &got, 1).is_some());
    // A wrong value is still one.
    let wrong = [Complex::new(0.5, 0.0)];
    assert!(first_mismatch_at(&expected[1..], &wrong, 1, &scales).is_some());
}
