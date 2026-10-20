//! Integer-only inspection of the actual prototype planner; no numeric library.
use std::collections::BTreeMap;
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    Mu,
    T3,
    T2,
}
#[path = "bridge/src/tensor_plan.rs"]
mod tensor_plan;

fn monomials(rank: usize) -> Vec<[u32; 4]> {
    let mut all = vec![[0; 4]];
    let mut previous = all.clone();
    for degree in 1..=rank {
        let mut next = Vec::new();
        for (component, count) in [
            1,
            degree,
            degree * (degree + 1) / 2,
            degree * (degree + 1) * (degree + 2) / 6,
        ]
        .into_iter()
        .enumerate()
        {
            for powers in previous.iter().take(count) {
                let mut powers = *powers;
                powers[component] += 1;
                next.push(powers);
            }
        }
        all.extend_from_slice(&next);
        previous = next;
    }
    all
}

fn main() {
    println!(
        "rank,propagators,shape,mu_powers,columns,dense_steps,planned_steps,plan_bytes,products,zero_products,unique_zero_products"
    );
    for rank in 0..=8 {
        let monomials = monomials(rank);
        let index: BTreeMap<_, _> = monomials.iter().copied().zip(0..).collect();
        let degrees: Vec<_> = monomials
            .iter()
            .map(|m| m.iter().sum::<u32>() as usize)
            .collect();
        let parents: Vec<_> = monomials
            .iter()
            .map(|m| {
                let Some(component) = (0..4).rev().find(|&c| m[c] > 0) else {
                    return (0, 0);
                };
                let mut parent = *m;
                parent[component] -= 1;
                (index[&parent], component)
            })
            .collect();
        for mu_powers in 0..=2.min(rank / 2) {
            let mut offset = 0;
            let blocks: Vec<_> = (0..=mu_powers)
                .map(|k| {
                    let r = rank - 2 * k;
                    let count = (r + 1) * (r + 2) * (r + 3) * (r + 4) / 24;
                    let result = (offset, count);
                    offset += count;
                    result
                })
                .collect();
            for propagators in rank.saturating_sub(2).max(1)..=rank.max(1) + 2 {
                for (shape, name, cut) in [(Shape::T3, "T3", 3), (Shape::T2, "T2", 2)] {
                    if cut > propagators {
                        continue;
                    }
                    let shift = propagators - cut;
                    if shift > rank {
                        continue;
                    }
                    let degree = rank - shift;
                    let mut powers = Vec::new();
                    for t in (0..=degree).rev() {
                        let max_x = if shape == Shape::T2 { degree - t } else { 0 };
                        for x in 0..=max_x {
                            for mu in 0..=(degree - t - x) / 2 {
                                powers.push(((t + shift) as i64, x, mu));
                            }
                        }
                    }
                    let columns = powers.len();
                    let key = tensor_plan::Key {
                        shape,
                        rank,
                        mu_powers,
                        powers,
                    };
                    let plan = tensor_plan::Plan::new(key, &degrees, &parents, &blocks).unwrap();
                    let mut products = 0;
                    let mut zero_products = 0;
                    let mut unique = std::collections::BTreeSet::new();
                    for (i, step) in plan.steps.iter().enumerate() {
                        assert!(step.terms >= 1 && step.terms <= 7);
                        assert_eq!(step.factors[0], 0);
                        assert!(
                            step.sources[..step.terms]
                                .iter()
                                .all(|&source| source < i + 2)
                        );
                        assert!(step.factors[..step.terms].windows(2).all(|w| w[0] < w[1]));
                        products += step.terms;
                        for term in 0..step.terms {
                            if step.sources[term] == 0 {
                                zero_products += 1;
                                unique.insert((step.component, step.factors[term]));
                            }
                        }
                    }
                    println!(
                        "{rank},{propagators},{name},{mu_powers},{columns},{},{},{},{products},{zero_products},{}",
                        plan.dense_recurrence_steps,
                        plan.steps.len(),
                        plan.bytes(),
                        unique.len()
                    );
                }
            }
        }
    }
}
