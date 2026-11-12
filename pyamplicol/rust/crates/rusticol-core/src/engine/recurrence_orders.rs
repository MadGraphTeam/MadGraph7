// SPDX-License-Identifier: 0BSD

//! Squared coupling-order partials of Direct-Arena recurrence results.
//!
//! The plan records the exact amplitude coupling-order component of every
//! closure row. A squared order `s` collects the interference of every
//! ordered component pair `(a, b)` with `a + b = s`:
//! `P_s = sum_{a+b=s} Re(A_a^dagger C A_b)`. Any real-bilinear expansion of
//! the colour contraction yields these partials when evaluated on ordered
//! pairs, because the pair set of every squared order is swap symmetric.
//! The partials therefore reuse the ordinary contraction entries with
//! distinct left and right amplitudes; the total path is never modified.

use std::collections::BTreeMap;

use crate::recurrence::direct_runtime::{DirectAmplitudeOrderSplit, DirectDestinationOrders};
use crate::recurrence::{DirectAmplitudeOrders, RuntimeSymmetricGroupPairWorkspace};
use crate::{RusticolError, RusticolResult};

/// One squared coupling order and its contributing ordered component pairs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SquaredAmplitudeOrder {
    powers: Box<[u32]>,
    pairs: Box<[(u32, u32)]>,
}

impl SquaredAmplitudeOrder {
    /// Coupling powers aligned with [`AmplitudeOrderCatalog::names`].
    pub fn powers(&self) -> &[u32] {
        &self.powers
    }

    /// Ordered `(left, right)` amplitude-component pairs.
    pub fn pairs(&self) -> &[(u32, u32)] {
        &self.pairs
    }
}

/// Amplitude and squared coupling-order axes of one recurrence process.
///
/// Squared orders are every pairwise sum of amplitude components, including
/// structurally vanishing interference, in canonical coupling-order order:
/// their nonzero `(name, power)` pairs over the sorted names, compared
/// lexicographically. This is the order of Python's canonical
/// `CouplingOrders` tuples, so native, SDK and Python columns agree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AmplitudeOrderCatalog {
    names: Box<[String]>,
    components: Box<[Box<[u32]>]>,
    squared: Box<[SquaredAmplitudeOrder]>,
}

impl AmplitudeOrderCatalog {
    pub fn from_plan(orders: &DirectAmplitudeOrders) -> RusticolResult<Self> {
        let components = (0..orders.component_count())
            .map(|component| {
                orders
                    .component(component)
                    .map(Box::<[u32]>::from)
                    .ok_or_else(|| RusticolError::integrity("amplitude order component is absent"))
            })
            .collect::<RusticolResult<Box<[Box<[u32]>]>>>()?;
        let mut squared = BTreeMap::<Vec<(usize, u32)>, (Vec<u32>, Vec<(u32, u32)>)>::new();
        for (left, left_powers) in components.iter().enumerate() {
            for (right, right_powers) in components.iter().enumerate() {
                let powers = left_powers
                    .iter()
                    .zip(right_powers.iter())
                    .map(|(left, right)| {
                        left.checked_add(*right).ok_or_else(|| {
                            RusticolError::integrity("squared coupling order exceeds u32")
                        })
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                let canonical = powers
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, power)| *power != 0)
                    .collect::<Vec<_>>();
                squared
                    .entry(canonical)
                    .or_insert_with(|| (powers, Vec::new()))
                    .1
                    .push((left as u32, right as u32));
            }
        }
        Ok(Self {
            names: orders.names().into(),
            components,
            squared: squared
                .into_values()
                .map(|(powers, pairs)| SquaredAmplitudeOrder {
                    powers: powers.into_boxed_slice(),
                    pairs: pairs.into_boxed_slice(),
                })
                .collect(),
        })
    }

    pub fn names(&self) -> &[String] {
        &self.names
    }

    pub fn components(&self) -> &[Box<[u32]>] {
        &self.components
    }

    pub fn squared_orders(&self) -> &[SquaredAmplitudeOrder] {
        &self.squared
    }

    /// One amplitude component: its only squared order is the total.
    pub fn is_single(&self) -> bool {
        self.components.len() == 1
    }
}

/// Lazily allocated order-resolved workspace of one recurrence runtime.
pub(super) struct RecurrenceOrderState {
    pub(super) split: DirectAmplitudeOrderSplit,
    pub(super) slots_re: Vec<f64>,
    pub(super) slots_im: Vec<f64>,
    pub(super) rows: Vec<crate::recurrence::DirectClosureRow>,
    /// Contracted colour: dense `[component][destination][point]` planes.
    pub(super) components_re: Vec<f64>,
    pub(super) components_im: Vec<f64>,
    /// Contracted Walsh colour: `[component][local group][point]` transforms.
    pub(super) transform_re: Vec<f64>,
    pub(super) transform_im: Vec<f64>,
    pub(super) symmetric_pairs: Option<RuntimeSymmetricGroupPairWorkspace>,
    /// Destination-major LC weights collected while accumulating a total.
    pub(super) lc_weights: Vec<(u32, f64)>,
}

impl RecurrenceOrderState {
    pub(super) fn new(split: DirectAmplitudeOrderSplit) -> Self {
        Self {
            split,
            slots_re: Vec::new(),
            slots_im: Vec::new(),
            rows: Vec::new(),
            components_re: Vec::new(),
            components_im: Vec::new(),
            transform_re: Vec::new(),
            transform_im: Vec::new(),
            symmetric_pairs: None,
            lc_weights: Vec::new(),
        }
    }

    /// Grow the order slots to the scheduler's current point pitch, which
    /// changes between packed-singleton and tiled layouts.
    pub(super) fn ensure_slot_capacity(&mut self, point_stride: usize) -> RusticolResult<()> {
        let needed = (self.split.slot_count() as usize)
            .checked_mul(point_stride)
            .ok_or_else(|| RusticolError::artifact("order slot workspace overflows usize"))?;
        if self.slots_re.len() < needed {
            self.slots_re.resize(needed, 0.0);
            self.slots_im.resize(needed, 0.0);
        }
        Ok(())
    }

    /// Accumulate `weight * sum_{(a,b)} Re(A_a conj(A_b))` of one LC
    /// destination into the point-major squared-order partials.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn accumulate_lc_destination(
        &self,
        catalog: &AmplitudeOrderCatalog,
        amplitude_re: &[f64],
        amplitude_im: &[f64],
        destination_id: u32,
        weight: f64,
        point_stride: usize,
        point_count: usize,
        partials: &mut [f64],
    ) -> RusticolResult<()> {
        let squared_count = catalog.squared_orders().len();
        for (squared_index, squared) in catalog.squared_orders().iter().enumerate() {
            for &(left, right) in squared.pairs() {
                let amplitude = |component| {
                    order_component(
                        &self.split,
                        (&self.slots_re, &self.slots_im),
                        (amplitude_re, amplitude_im),
                        destination_id,
                        component,
                        point_stride,
                        point_count,
                    )
                };
                let (Some((left_re, left_im)), Some((right_re, right_im))) =
                    (amplitude(left)?, amplitude(right)?)
                else {
                    continue;
                };
                for point in 0..point_count {
                    partials[point * squared_count + squared_index] += weight
                        * left_re[point].mul_add(right_re[point], left_im[point] * right_im[point]);
                }
            }
        }
        Ok(())
    }
}

/// Amplitude part of `component` at `destination_id`; `None` is zero.
///
/// Single-component destinations read the ordinary amplitude planes; split
/// destinations read their replayed order slot.
pub(super) fn order_component<'a>(
    split: &DirectAmplitudeOrderSplit,
    slots: (&'a [f64], &'a [f64]),
    amplitudes: (&'a [f64], &'a [f64]),
    destination_id: u32,
    component: u32,
    point_stride: usize,
    point_count: usize,
) -> RusticolResult<Option<(&'a [f64], &'a [f64])>> {
    let orders = split.destination_orders(destination_id).ok_or_else(|| {
        RusticolError::integrity("order-resolved amplitude destination is absent")
    })?;
    let ((values_re, values_im), plane) = match orders {
        DirectDestinationOrders::Single(only) if only == component => {
            (amplitudes, destination_id as usize)
        }
        DirectDestinationOrders::Single(_) => return Ok(None),
        DirectDestinationOrders::Split { first_slot } => (slots, (first_slot + component) as usize),
    };
    let start = plane * point_stride;
    match (
        values_re.get(start..start + point_count),
        values_im.get(start..start + point_count),
    ) {
        (Some(real), Some(imaginary)) => Ok(Some((real, imaginary))),
        _ => Err(RusticolError::integrity(
            "order-resolved amplitude plane is out of bounds",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_catalog_pairs_every_ordered_component_sum() {
        let orders = DirectAmplitudeOrders {
            names: vec!["QCD".into(), "QED".into()],
            component_count: 2,
            powers: vec![0, 2, 2, 0],
            closure_components: vec![0, 1],
        };
        let catalog = AmplitudeOrderCatalog::from_plan(&orders).unwrap();
        let squared = catalog
            .squared_orders()
            .iter()
            .map(|order| (order.powers().to_vec(), order.pairs().to_vec()))
            .collect::<Vec<_>>();
        assert_eq!(
            squared,
            vec![
                (vec![2, 2], vec![(0, 1), (1, 0)]),
                (vec![4, 0], vec![(1, 1)]),
                (vec![0, 4], vec![(0, 0)]),
            ]
        );
        assert!(!catalog.is_single());
    }
}
