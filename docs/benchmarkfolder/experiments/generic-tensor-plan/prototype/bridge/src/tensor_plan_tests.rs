//! Numerical oracle is the frozen production dense source, not an algebraic
//! reimplementation. These tests are drafted and have not yet been executed.
use super::*;
use ninjars::{Float, SampleId};

fn equal_component(a: &Float, b: &Float) {
    assert_eq!(a.prec(), b.prec(), "tracked precision differs");
    assert_eq!(a.is_sign_negative(), b.is_sign_negative(), "sign differs");
    assert_eq!(a.is_finite(), b.is_finite());
    if a.is_finite() {
        assert_eq!(
            a.to_rational(),
            b.to_rational(),
            "exact stored value differs"
        );
    } else {
        assert_eq!(format!("{a:?}"), format!("{b:?}"));
    }
}

fn equal_rows(a: &[Complex<Float>], b: &[Complex<Float>]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        equal_component(&a.re, &b.re);
        equal_component(&a.im, &b.im);
    }
}

fn number(bits: u32, seed: usize) -> Complex<Float> {
    // Different tracked precisions, signs, rounded fractions and signed zeros.
    let p = [bits, bits.saturating_sub(17).max(2), bits / 2][seed % 3];
    let re = if seed % 7 == 0 {
        Float::with_val(p, if seed % 2 == 0 { -0.0 } else { 0.0 })
    } else {
        Float::with_val(p, seed as i64 - 11) / Float::with_val(p, 7)
    };
    let im = if seed % 5 == 0 {
        Float::with_val(p, -0.0)
    } else {
        Float::with_val(p, 3 - seed as i64) / Float::with_val(p, 13)
    };
    Complex::new(re, im)
}

fn kinds(rank: usize) -> Vec<ExpansionKind> {
    let r = rank as i32;
    let mut t3 = Vec::new();
    let mut t2 = Vec::new();
    for p in (-1..=r).rev() {
        for l in 0..=(r - p) / 2 {
            t3.push((p, l as u16));
        }
        for a in 0..=r - p {
            for l in 0..=(r - p - a) / 2 {
                t2.push((p, a as u16, l as u16));
            }
        }
    }
    // Duplicate/reordered output columns and a deliberately empty contraction.
    t3.extend([(r, 0), (r + 2, 0), (-1, 0)]);
    t2.extend([(r, 0, 0), (r + 2, 0, 0), (-1, 0, 0)]);
    vec![
        ExpansionKind::Mu {
            powers: vec![r, 0, r - 1, -1, r],
        },
        ExpansionKind::MuShifted {
            powers: vec![r, 0, r - 1, -1, r],
        },
        ExpansionKind::T3 { powers: t3 },
        ExpansionKind::T2 { powers: t2 },
    ]
}

#[test]
fn planned_tensor_matches_dense_values_precision_and_signed_zero() {
    for bits in [113, 256, 384] {
        for rank in 0..=5 {
            for mu_powers in 0..=2.min(rank / 2) {
                let prototype = Float::new(bits);
                let mut planned = TensorGenericNumerator::new(rank, mu_powers, &prototype).unwrap();
                let mut dense =
                    dense_reference::TensorGenericNumerator::new(rank, mu_powers, &prototype)
                        .unwrap();
                let coefficients: Vec<_> = (0..2 * planned.width)
                    .map(|i| number(bits, i + 1))
                    .collect();
                planned.set_coefficients(coefficients.clone(), 2).unwrap();
                dense.set_coefficients(coefficients.clone(), 2).unwrap();
                for kind in kinds(rank) {
                    let (vectors, scalars) = match kind {
                        ExpansionKind::Mu { .. } => (1, 0),
                        ExpansionKind::MuShifted { .. } => (2, 1),
                        ExpansionKind::T3 { .. } => (3, 1),
                        ExpansionKind::T2 { .. } => (4, 3),
                    };
                    for lane in 0..2 {
                        let v: Vec<_> = (0..vectors)
                            .map(|i| {
                                FourMomentum::from(std::array::from_fn(|c| {
                                    number(bits, i * 4 + c + lane * 9 + 1)
                                }))
                            })
                            .collect();
                        let p: Vec<_> = (0..scalars).map(|i| number(bits, i + lane + 2)).collect();
                        let request = ExpansionRequest {
                            id: SampleId { lane, sample: 0 },
                            partition: &[],
                            vectors: &v,
                            parameters: &p,
                        };
                        let mut a = vec![number(bits, 99); kind.len()];
                        let mut b = a.clone();
                        planned.expansion_batch(&kind, &[request], &mut a).unwrap();
                        // Construct the second request without requiring Clone on its type.
                        let request = ExpansionRequest {
                            id: SampleId { lane, sample: 0 },
                            partition: &[],
                            vectors: &v,
                            parameters: &p,
                        };
                        dense.expansion_batch(&kind, &[request], &mut b).unwrap();
                        equal_rows(&a, &b);
                    }
                }
                assert_eq!(planned.dense_rows, 0);
                assert_eq!(planned.planned_rows, 8);
                assert!(planned.plan_bytes <= tensor_plan::CACHE_BYTES);
                // New coefficients use the existing shape plans.
                let count = planned.plans.len();
                let rebound: Vec<_> = coefficients.into_iter().rev().collect();
                planned.set_coefficients(rebound.clone(), 2).unwrap();
                dense.set_coefficients(rebound, 2).unwrap();
                let kind = kinds(rank).pop().unwrap();
                let v0: Vec<_> = (0..4)
                    .map(|i| {
                        FourMomentum::from(std::array::from_fn(|c| number(bits, i * 4 + c + 31)))
                    })
                    .collect();
                let v1: Vec<_> = (0..4)
                    .map(|i| {
                        FourMomentum::from(std::array::from_fn(|c| number(bits, i * 4 + c + 57)))
                    })
                    .collect();
                let p0: Vec<_> = (0..3).map(|i| number(bits, i + 7)).collect();
                let p1: Vec<_> = (0..3).map(|i| number(bits, i + 21)).collect();
                let requests = || {
                    [
                        ExpansionRequest {
                            id: SampleId { lane: 1, sample: 0 },
                            partition: &[],
                            vectors: &v0,
                            parameters: &p0,
                        },
                        ExpansionRequest {
                            id: SampleId { lane: 0, sample: 1 },
                            partition: &[],
                            vectors: &v1,
                            parameters: &p1,
                        },
                    ]
                };
                let mut a = vec![number(bits, 99); 2 * kind.len()];
                let mut b = a.clone();
                planned.expansion_batch(&kind, &requests(), &mut a).unwrap();
                dense.expansion_batch(&kind, &requests(), &mut b).unwrap();
                equal_rows(&a, &b);
                assert_eq!(planned.plans.len(), count);
                assert_eq!(planned.planned_rows, 10);
            }
        }
    }
}

#[test]
fn planned_tensor_keeps_computed_zero_nodes_and_rejects_oversized_plans() {
    let prototype = Float::new(256);
    let mut tensor = TensorGenericNumerator::new(2, 0, &prototype).unwrap();
    // A degree-one monomial requested at u^2 is algebraically zero but the
    // dense algorithm computes it. It must survive the backward slice.
    let plan = tensor.expansion_plan(Shape::Mu, &[(-1, 0, 0)]).unwrap();
    let index = monomial_index([1, 0, 0, 0]);
    let depth = 3;
    assert!(plan.slot(index * (depth + 1) + 2) >= 2);
    assert!(plan.steps.len() <= plan.dense_recurrence_steps);
    assert!(
        tensor
            .expansion_plan(Shape::T2, &[(-100_000, 0, 0)])
            .is_none()
    );
    assert!(tensor.plan_bytes <= tensor_plan::CACHE_BYTES);
    let mut constant = TensorGenericNumerator::new(0, 0, &prototype).unwrap();
    // Few retained states do not make a huge repeated convolution scan cheap.
    assert!(
        constant
            .expansion_plan(Shape::Mu, &vec![(-5000, 0, 0); 2048])
            .is_none()
    );
}

#[test]
fn planned_tensor_nonfinite_inputs_use_dense_fallback() {
    let bits = 113;
    let prototype = Float::new(bits);
    let mut planned = TensorGenericNumerator::new(2, 0, &prototype).unwrap();
    let mut dense = dense_reference::TensorGenericNumerator::new(2, 0, &prototype).unwrap();
    let coefficients: Vec<_> = (0..planned.width).map(|i| number(bits, i)).collect();
    planned.set_coefficients(coefficients.clone(), 1).unwrap();
    dense.set_coefficients(coefficients, 1).unwrap();
    let mut v = vec![FourMomentum::from(std::array::from_fn(|i| {
        number(bits, i + 1)
    }))];
    let kind = ExpansionKind::Mu {
        powers: vec![2, 1, 0],
    };
    v[0] = FourMomentum::from([
        Complex::new(Float::with_val(bits, f64::INFINITY), Float::new(bits)),
        number(bits, 2),
        number(bits, 3),
        number(bits, 4),
    ]);
    let mut a = vec![number(bits, 8); kind.len()];
    let mut b = a.clone();
    planned
        .expansion_batch(
            &kind,
            &[ExpansionRequest {
                id: SampleId { lane: 0, sample: 0 },
                partition: &[],
                vectors: &v,
                parameters: &[],
            }],
            &mut a,
        )
        .unwrap();
    dense
        .expansion_batch(
            &kind,
            &[ExpansionRequest {
                id: SampleId { lane: 0, sample: 0 },
                partition: &[],
                vectors: &v,
                parameters: &[],
            }],
            &mut b,
        )
        .unwrap();
    equal_rows(&a, &b);
    assert_eq!(planned.planned_rows, 0);
    assert_eq!(planned.dense_rows, 1);
}
