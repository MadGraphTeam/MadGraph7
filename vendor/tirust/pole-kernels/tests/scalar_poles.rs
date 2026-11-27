use madloop_pole_kernels::{Complex, NativeFloat, ScalarIntegral, ScalarPoleEvaluator};
use oneloop::NativeEvaluator;
use symbolica::domains::float::{DoubleFloat, Float};

fn real(value: f64) -> Complex<f64> {
    Complex::new(value, 0.0)
}

fn check<T: NativeFloat>(
    family: ScalarIntegral,
    input: &[Complex<f64>],
    bits: u32,
    convert: impl Fn(f64) -> T,
    tolerance: f64,
) {
    let input = input
        .iter()
        .map(|z| Complex::new(convert(z.re), convert(z.im)))
        .collect::<Vec<_>>();
    let zero = || Complex::new(T::zero_at(bits), T::zero_at(bits));
    let mut truncated = ScalarPoleEvaluator::<T>::with_binary_precision(family, bits).unwrap();
    let mut complete = NativeEvaluator::<T>::with_binary_precision(family, bits).unwrap();
    let mut actual = [zero(), zero()];
    let mut expected = [zero(), zero(), zero()];
    truncated.evaluate(&input, &mut actual).unwrap();
    complete.evaluate(&input, &mut expected).unwrap();
    for (index, (actual, expected)) in actual.iter().zip(&expected[1..]).enumerate() {
        assert!(
            expected.re.is_finite() && expected.im.is_finite(),
            "nonfinite reference {family:?} slot {index}"
        );
        let error = (actual.re.clone() - &expected.re)
            .to_f64()
            .hypot((actual.im.clone() - &expected.im).to_f64());
        let scale = expected.re.to_f64().hypot(expected.im.to_f64());
        assert!(
            error <= tolerance * scale.max(1e-100),
            "{family:?} {bits}-bit slot {index}: error={error:e}, scale={scale:e}, actual={actual:?}, expected={expected:?}, input={input:?}"
        );
    }
}

fn cases() -> Vec<(ScalarIntegral, Vec<Complex<f64>>)> {
    let mut cases = Vec::new();
    for mass in [real(0.0), real(3.0), Complex::new(3.0, -0.25)] {
        cases.push((ScalarIntegral::A0, vec![mass, real(2.0)]));
    }
    for point in [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 2.0, 2.0, 1.0],
        [-3.0, 0.0, 0.0, 2.0],
        [4.0, 0.0, 4.0, 2.0],
        [9.0, 1.0, 4.0, 3.0],
        [1e-12, 1.0, 4.0, 3.0],
    ] {
        cases.push((ScalarIntegral::B0, point.map(real).to_vec()));
    }
    cases.push((
        ScalarIntegral::B0,
        vec![
            real(5.0),
            Complex::new(2.0, -0.125),
            Complex::new(3.0, -0.25),
            real(2.0),
        ],
    ));
    // All invariant orders exercise exact lazy branches next to massless legs.
    for momenta in [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -2.0],
        [0.0, 0.0, 2.0],
        [0.0, -2.0, 3.0],
        [0.0, -2.0, -2.0],
    ] {
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            let mut point = order.map(|i| real(momenta[i])).to_vec();
            point.extend([real(0.0), real(0.0), real(0.0), real(3.0)]);
            cases.push((ScalarIntegral::C0, point));
        }
    }
    for point in [
        [1.0, 1.0, -2.0, 0.0, 1.0, 1.0, 3.0],
        [-1.0, -2.0, -3.0, 1.0, 2.0, 3.0, 2.0],
        [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 2.0],
    ] {
        cases.push((ScalarIntegral::C0, point.map(real).to_vec()));
    }
    cases.push((
        ScalarIntegral::C0,
        vec![
            real(1.0),
            real(2.0),
            real(-3.0),
            Complex::new(1.0, -0.1),
            Complex::new(2.0, -0.2),
            Complex::new(3.0, -0.3),
            real(2.0),
        ],
    ));
    for channels in [[-2.0, -3.0], [2.0, -3.0], [2.0, 3.0]] {
        cases.push((
            ScalarIntegral::D0,
            [
                0.0,
                0.0,
                0.0,
                0.0,
                channels[0],
                channels[1],
                0.0,
                0.0,
                0.0,
                0.0,
                2.0,
            ]
            .map(real)
            .to_vec(),
        ));
    }
    // Soft and collinear box fixtures from oneloopmaster's public regression
    // tests, including orders that put a soft line's on-shell pair on a channel.
    for point in [
        [
            29929.0, 29929.0, 6464.0, 6464.0, -40000.0, 250000.0, 0.0, 29929.0, 0.0, 6464.0, 1.0,
        ],
        [
            0.0, 29929.0, 15625.0, 8315.0, -50000.0, 200000.0, 0.0, 0.0, 29929.0, 8315.0, 1.0,
        ],
    ] {
        let point = point.map(real);
        const SLOT: [[usize; 4]; 4] = [[0, 0, 4, 3], [0, 0, 1, 5], [4, 1, 0, 2], [3, 5, 2, 0]];
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    if a == b || a == c || b == c {
                        continue;
                    }
                    let d = 6 - a - b - c;
                    let order = [a, b, c, d];
                    let mut image = [(a, b), (b, c), (c, d), (d, a), (a, c), (b, d)]
                        .map(|(i, j)| point[SLOT[i][j]])
                        .to_vec();
                    image.extend(order.map(|i| point[6 + i]));
                    image.push(point[10]);
                    cases.push((ScalarIntegral::D0, image));
                }
            }
        }
    }
    cases.push((
        ScalarIntegral::D0,
        vec![
            real(-1.0),
            real(-2.0),
            real(-3.0),
            real(-4.0),
            real(-5.0),
            real(-6.0),
            Complex::new(1.0, -0.1),
            Complex::new(2.0, -0.2),
            Complex::new(3.0, -0.3),
            Complex::new(4.0, -0.4),
            real(2.0),
        ],
    ));
    cases
}

#[test]
fn native_precisions_match_full_master_poles() {
    for (family, point) in cases() {
        check::<f64>(family, &point, 53, |x| x, 2e-12);
        check::<DoubleFloat>(family, &point, 106, DoubleFloat::from, 2e-27);
        check::<Float>(family, &point, 192, |x| Float::with_val(192, x), 2e-48);
    }
}

#[test]
fn domain_errors_preserve_output_and_precision_is_explicit() {
    let mut evaluator = ScalarPoleEvaluator::<f64>::new(ScalarIntegral::B0).unwrap();
    let unchanged = [real(123.0), real(456.0)];
    let mut output = unchanged;
    for input in [
        vec![real(1.0)],
        vec![real(f64::NAN), real(1.0), real(1.0), real(1.0)],
        vec![Complex::new(1.0, 1.0), real(1.0), real(1.0), real(1.0)],
        vec![real(1.0), Complex::new(1.0, 0.1), real(1.0), real(1.0)],
        vec![real(1.0), real(1.0), real(1.0), real(0.0)],
        vec![real(1.0), real(1.0), real(1.0), Complex::new(1.0, 0.1)],
    ] {
        assert!(evaluator.evaluate(&input, &mut output).is_err());
        assert_eq!(output, unchanged);
    }
    assert!(
        evaluator
            .evaluate(&[real(1.0); 4], &mut [real(0.0); 3])
            .is_err()
    );
    assert!(ScalarPoleEvaluator::<f64>::with_binary_precision(ScalarIntegral::A0, 106).is_err());
    assert!(
        ScalarPoleEvaluator::<DoubleFloat>::with_binary_precision(ScalarIntegral::A0, 53).is_err()
    );
    assert!(ScalarPoleEvaluator::<Float>::with_binary_precision(ScalarIntegral::A0, 0).is_err());
    assert!(ScalarPoleEvaluator::<Float>::new(ScalarIntegral::DB0).is_err());
}

#[test]
fn scaleless_total_residues_cancel_and_no_finite_roots_are_generated() {
    for family in [ScalarIntegral::A0, ScalarIntegral::B0, ScalarIntegral::C0] {
        let mut point = vec![real(0.0); family.arity()];
        *point.last_mut().unwrap() = real(1.0);
        let mut output = [real(123.0), real(456.0)];
        ScalarPoleEvaluator::<f64>::new(family)
            .unwrap()
            .evaluate(&point, &mut output)
            .unwrap();
        assert_eq!(output, [real(0.0); 2], "{family:?}");
    }
    assert!(
        madloop_pole_kernels::GENERATION_MANIFEST.contains("laurent_tags=-1,-2\nfinite_roots=0")
    );
    for source in [
        include_str!("../src/generated_a0.rs"),
        include_str!("../src/generated_b0.rs"),
        include_str!("../src/generated_c0.rs"),
        include_str!("../src/generated_d0.rs"),
    ] {
        assert!(
            !source.contains("polylog2("),
            "finite-only dilogarithm in residue graph"
        );
    }
}

#[test]
fn undefined_collapsed_box_is_an_explicit_error() {
    // The upstream exact D0 expressions have no definition at s=t=0 with
    // every mass/invariant zero. Do not invent a formula or silently return
    // plausible zero residues; the tensor executor handles its scaleless
    // reductions before requesting scalar masters.
    let mut input = [real(0.0); 11];
    input[10] = real(1.0);
    let unchanged = [real(123.0), real(456.0)];
    let mut output = unchanged;
    let result = ScalarPoleEvaluator::<f64>::new(ScalarIntegral::D0)
        .unwrap()
        .evaluate(&input, &mut output);
    assert!(result.is_err());
    assert_eq!(output, unchanged);
    let mut complete = [real(0.0); 3];
    NativeEvaluator::<f64>::new(ScalarIntegral::D0)
        .unwrap()
        .evaluate(&input, &mut complete)
        .unwrap();
    assert!(
        complete[1..]
            .iter()
            .any(|z| !z.re.is_finite() || !z.im.is_finite())
    );
}
