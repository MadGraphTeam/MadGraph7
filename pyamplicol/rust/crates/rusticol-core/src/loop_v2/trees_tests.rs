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
    assert!(
        error
            .to_string()
            .starts_with("complex-mass-master: plan mass 3")
    );
}

#[test]
fn the_complex_mass_master_guard_is_double_double_only() {
    assert!(<DoubleFloat as TreeCarrier>::COMPLEX_MASS_MASTER_GUARD);
    assert!(!<f64 as TreeCarrier>::COMPLEX_MASS_MASTER_GUARD);
    assert!(!<Float as TreeCarrier>::COMPLEX_MASS_MASTER_GUARD);
}

#[test]
fn the_run_time_side_check_takes_the_exact_side_of_a_cut_the_plane_misses() {
    // sqrt(-x) through SymJIT: a negated real takes a -0 imaginary part
    // where Symbolica's evaluator takes +0, so at x > 0 (on the cut) the
    // plane and the exact evaluator give opposite imaginary parts; off the
    // cut (x < 0) they agree. `TreeProgram::bind` settles a binary64 batch
    // through `ParameterSide::settle`: checked once per parameter values,
    // the exact evaluator's rows where the plane disagreed.
    use symbolica::atom::AtomCore;
    let _unlock = crate::licence::unlock();
    let x = symbolica::parse!("pyamplicol_trees_tests::x");
    let evaluator = symbolica::parse!("sqrt(-pyamplicol_trees_tests::x)")
        .evaluator(&[x])
        .build()
        .expect("an evaluator");
    let binary64 = evaluator.map_coeff(&|r| Complex::new(r.re.to_f64(), r.im.to_f64()));
    let (bytes, inputs, outputs) =
        crate::engine::compile_symbolica_evaluator_to_plane_application_bytes(&binary64, 2, false)
            .expect("a plane application");
    let mut plane = Kernel::Plane(Box::new(
        BinaryPlane::load(&bytes, (inputs, outputs)).expect("loads"),
    ));
    let mut exact = Kernel::Exact(binary64.clone().into_batch_evaluator());
    let build = || Ok(Kernel::Exact(binary64.clone().into_batch_evaluator()));
    let mut side = ParameterSide {
        exact: None,
        checked: None,
    };
    let rows = |kernel: &mut Kernel<f64>, x: f64| {
        let input = [Complex::new(x, 0.0); 2];
        let mut rows = [Complex::new(0.0, 0.0); 2];
        kernel.evaluate_batch(&input, &mut rows, 2).expect("runs");
        (input, rows)
    };
    for x in [-4.0, 4.0] {
        let (input, mut got) = rows(&mut plane, x);
        let (_, expected) = rows(&mut exact, x);
        let on_cut = x > 0.0;
        assert_eq!(
            got != expected,
            on_cut,
            "x = {x}: plane {got:?}, exact {expected:?}"
        );
        let key = vec![Rational::from(x as i64)];
        let kept = side
            .settle(key.clone(), &input, &mut got, (2, 1, 1), build)
            .expect("settles");
        assert_eq!(kept, !on_cut, "x = {x}");
        assert_eq!(got, expected, "x = {x}: the exact side");
        assert_eq!(side.checked, Some((key.clone(), !on_cut)));
        // The same values again: the recorded verdict, not a new check (rows
        // that agree with the exact evaluator stay replaced on the cut).
        let mut again = expected;
        let kept = side
            .settle(key, &input, &mut again, (2, 1, 1), || panic!("built once"))
            .expect("settles");
        assert_eq!((kept, again), (!on_cut, expected), "x = {x}");
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
