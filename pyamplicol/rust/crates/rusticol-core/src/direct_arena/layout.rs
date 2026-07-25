// SPDX-License-Identifier: 0BSD

//! Deterministic variable-width interval allocation for arena planes.

use std::collections::BTreeMap;

use crate::{RusticolError, RusticolResult};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

/// Inclusive liveness interval and required component-plane width.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectArenaInterval {
    pub semantic_value_id: u32,
    pub first_use: u64,
    pub last_use: u64,
    pub component_count: u32,
}

impl DirectArenaInterval {
    pub fn new(
        semantic_value_id: u32,
        first_use: u64,
        last_use: u64,
        component_count: u32,
    ) -> RusticolResult<Self> {
        if first_use > last_use {
            return Err(invalid(format!(
                "direct-arena value {semantic_value_id} starts at {first_use} after its last use {last_use}"
            )));
        }
        if component_count == 0 {
            return Err(invalid(format!(
                "direct-arena value {semantic_value_id} has no components"
            )));
        }
        Ok(Self {
            semantic_value_id,
            first_use,
            last_use,
            component_count,
        })
    }
}

/// One semantic value's physical component range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DirectArenaAssignment {
    pub semantic_value_id: u32,
    pub component_base: u32,
    pub component_count: u32,
    pub first_use: u64,
    pub last_use: u64,
}

impl DirectArenaAssignment {
    pub fn component_stop(self) -> RusticolResult<u32> {
        self.component_base
            .checked_add(self.component_count)
            .ok_or_else(|| invalid("direct-arena component range overflows u32"))
    }
}

/// Deterministic interval-coloring result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirectArenaLayout {
    assignments: Box<[DirectArenaAssignment]>,
    component_count: u32,
    total_semantic_components: u64,
    reused_semantic_components: u64,
}

impl DirectArenaLayout {
    pub fn assignments(&self) -> &[DirectArenaAssignment] {
        &self.assignments
    }

    pub const fn component_count(&self) -> u32 {
        self.component_count
    }

    pub const fn total_semantic_components(&self) -> u64 {
        self.total_semantic_components
    }

    pub const fn reused_semantic_components(&self) -> u64 {
        self.reused_semantic_components
    }

    pub fn assignment(&self, semantic_value_id: u32) -> Option<DirectArenaAssignment> {
        self.assignments
            .get(semantic_value_id as usize)
            .copied()
            .filter(|assignment| assignment.semantic_value_id == semantic_value_id)
    }

    pub fn validate(&self) -> RusticolResult<()> {
        if self.assignments.is_empty() {
            if self.component_count != 0
                || self.total_semantic_components != 0
                || self.reused_semantic_components != 0
            {
                return Err(invalid("empty direct-arena layout has nonempty accounting"));
            }
            return Ok(());
        }

        let mut semantic_components = 0_u64;
        for (index, assignment) in self.assignments.iter().copied().enumerate() {
            if assignment.semantic_value_id != index as u32 {
                return Err(invalid(format!(
                    "direct-arena assignment row {index} has semantic value ID {}",
                    assignment.semantic_value_id
                )));
            }
            if assignment.component_count == 0 || assignment.first_use > assignment.last_use {
                return Err(invalid(format!(
                    "direct-arena assignment {index} has invalid width or liveness"
                )));
            }
            if assignment.component_stop()? > self.component_count {
                return Err(invalid(format!(
                    "direct-arena assignment {index} exceeds the physical arena"
                )));
            }
            semantic_components = semantic_components
                .checked_add(u64::from(assignment.component_count))
                .ok_or_else(|| invalid("direct-arena semantic component count overflows u64"))?;
        }
        if semantic_components != self.total_semantic_components {
            return Err(invalid(format!(
                "direct-arena semantic component total {semantic_components} does not match recorded {}",
                self.total_semantic_components
            )));
        }
        if self
            .total_semantic_components
            .checked_sub(u64::from(self.component_count))
            != Some(self.reused_semantic_components)
        {
            return Err(invalid(
                "direct-arena reused component accounting is inconsistent",
            ));
        }

        for (left_index, left) in self.assignments.iter().copied().enumerate() {
            for right in self.assignments[left_index + 1..].iter().copied() {
                let lifetimes_overlap =
                    left.first_use <= right.last_use && right.first_use <= left.last_use;
                let ranges_overlap = left.component_base < right.component_stop()?
                    && right.component_base < left.component_stop()?;
                if lifetimes_overlap && ranges_overlap {
                    return Err(invalid(format!(
                        "live direct-arena values {} and {} overlap physical components",
                        left.semantic_value_id, right.semantic_value_id
                    )));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct ActiveRange {
    last_use: u64,
    component_base: u32,
    component_count: u32,
}

/// Assign variable-width semantic intervals to reusable physical planes.
///
/// Inputs use dense semantic value IDs. Intervals are processed by
/// `(first_use, semantic_value_id)`. The allocator releases ranges whose
/// inclusive lifetime ended before the next value starts, then chooses the
/// smallest fitting range with the lowest base as a deterministic tie-breaker.
pub fn assign_direct_arena(intervals: &[DirectArenaInterval]) -> RusticolResult<DirectArenaLayout> {
    if intervals.is_empty() {
        return Ok(DirectArenaLayout {
            assignments: Box::new([]),
            component_count: 0,
            total_semantic_components: 0,
            reused_semantic_components: 0,
        });
    }

    let mut ordered = intervals.to_vec();
    ordered.sort_by_key(|interval| (interval.first_use, interval.semantic_value_id));
    let mut seen = vec![false; intervals.len()];
    for interval in &ordered {
        let id = interval.semantic_value_id as usize;
        if id >= seen.len() || seen[id] {
            return Err(invalid(
                "direct-arena semantic value IDs must be dense and unique",
            ));
        }
        seen[id] = true;
        if interval.component_count == 0 || interval.first_use > interval.last_use {
            return Err(invalid(format!(
                "direct-arena value {} has invalid width or liveness",
                interval.semantic_value_id
            )));
        }
    }
    if seen.contains(&false) {
        return Err(invalid(
            "direct-arena semantic value IDs must cover zero through count minus one",
        ));
    }

    let mut assignments = vec![None; intervals.len()];
    let mut active = Vec::<ActiveRange>::new();
    let mut free = BTreeMap::<u32, u32>::new();
    let mut arena_stop = 0_u32;
    let mut total_semantic_components = 0_u64;

    for interval in ordered {
        let mut retained = Vec::with_capacity(active.len() + 1);
        for range in active.drain(..) {
            if range.last_use < interval.first_use {
                insert_free_range(&mut free, range.component_base, range.component_count)?;
            } else {
                retained.push(range);
            }
        }
        active = retained;

        let (component_base, available_count) = free
            .iter()
            .filter(|(_, count)| **count >= interval.component_count)
            .min_by_key(|(base, count)| (**count, **base))
            .map(|(base, count)| (*base, *count))
            .unwrap_or((arena_stop, 0));

        if available_count == 0 {
            arena_stop = arena_stop
                .checked_add(interval.component_count)
                .ok_or_else(|| invalid("direct-arena physical component count exceeds u32"))?;
        } else {
            free.remove(&component_base);
            let remaining = available_count - interval.component_count;
            if remaining != 0 {
                free.insert(
                    component_base
                        .checked_add(interval.component_count)
                        .ok_or_else(|| invalid("direct-arena free range overflows u32"))?,
                    remaining,
                );
            }
        }

        let assignment = DirectArenaAssignment {
            semantic_value_id: interval.semantic_value_id,
            component_base,
            component_count: interval.component_count,
            first_use: interval.first_use,
            last_use: interval.last_use,
        };
        assignments[interval.semantic_value_id as usize] = Some(assignment);
        active.push(ActiveRange {
            last_use: interval.last_use,
            component_base,
            component_count: interval.component_count,
        });
        total_semantic_components = total_semantic_components
            .checked_add(u64::from(interval.component_count))
            .ok_or_else(|| invalid("direct-arena semantic component count exceeds u64"))?;
    }

    let assignments = assignments
        .into_iter()
        .enumerate()
        .map(|(id, assignment)| {
            assignment.ok_or_else(|| {
                invalid(format!(
                    "direct-arena value {id} was not assigned a physical range"
                ))
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let layout = DirectArenaLayout {
        assignments: assignments.into_boxed_slice(),
        component_count: arena_stop,
        total_semantic_components,
        reused_semantic_components: total_semantic_components - u64::from(arena_stop),
    };
    layout.validate()?;
    Ok(layout)
}

fn insert_free_range(
    free: &mut BTreeMap<u32, u32>,
    mut component_base: u32,
    mut component_count: u32,
) -> RusticolResult<()> {
    if component_count == 0 {
        return Err(invalid("cannot release an empty direct-arena range"));
    }

    if let Some((&previous_base, &previous_count)) = free.range(..component_base).next_back() {
        let previous_stop = previous_base
            .checked_add(previous_count)
            .ok_or_else(|| invalid("direct-arena free range overflows u32"))?;
        if previous_stop > component_base {
            return Err(invalid("released direct-arena ranges overlap"));
        }
        if previous_stop == component_base {
            component_base = previous_base;
            component_count = component_count
                .checked_add(previous_count)
                .ok_or_else(|| invalid("direct-arena merged range overflows u32"))?;
            free.remove(&previous_base);
        }
    }

    let component_stop = component_base
        .checked_add(component_count)
        .ok_or_else(|| invalid("direct-arena free range overflows u32"))?;
    if let Some((&next_base, &next_count)) = free.range(component_base..).next() {
        if component_stop > next_base {
            return Err(invalid("released direct-arena ranges overlap"));
        }
        if component_stop == next_base {
            component_count = component_count
                .checked_add(next_count)
                .ok_or_else(|| invalid("direct-arena merged range overflows u32"))?;
            free.remove(&next_base);
        }
    }
    if free.insert(component_base, component_count).is_some() {
        return Err(invalid("released direct-arena range repeats its base"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn interval(id: u32, first: u64, last: u64, count: u32) -> DirectArenaInterval {
        DirectArenaInterval::new(id, first, last, count).unwrap()
    }

    #[test]
    fn non_overlapping_values_reuse_the_same_range() {
        let layout = assign_direct_arena(&[
            interval(0, 0, 2, 4),
            interval(1, 3, 5, 4),
            interval(2, 6, 8, 4),
        ])
        .unwrap();
        assert_eq!(layout.component_count(), 4);
        assert_eq!(layout.total_semantic_components(), 12);
        assert_eq!(layout.reused_semantic_components(), 8);
        assert!(
            layout
                .assignments()
                .iter()
                .all(|row| row.component_base == 0)
        );
    }

    #[test]
    fn inclusive_lifetimes_do_not_reuse_at_the_same_event() {
        let layout = assign_direct_arena(&[interval(0, 0, 3, 2), interval(1, 3, 4, 3)]).unwrap();
        assert_eq!(layout.component_count(), 5);
        assert_eq!(layout.assignment(0).unwrap().component_base, 0);
        assert_eq!(layout.assignment(1).unwrap().component_base, 2);
    }

    #[test]
    fn allocator_chooses_the_smallest_fitting_range_deterministically() {
        let layout = assign_direct_arena(&[
            interval(0, 0, 1, 8),
            interval(1, 0, 5, 2),
            interval(2, 0, 1, 3),
            interval(3, 2, 4, 2),
            interval(4, 2, 4, 7),
        ])
        .unwrap();
        assert_eq!(layout.assignment(3).unwrap().component_base, 10);
        assert_eq!(layout.assignment(4).unwrap().component_base, 0);
        assert_eq!(layout.component_count(), 13);
    }

    #[test]
    fn assignment_is_stable_for_unsorted_input() {
        let intervals = [
            interval(3, 8, 9, 2),
            interval(0, 0, 3, 4),
            interval(2, 4, 7, 3),
            interval(1, 1, 2, 1),
        ];
        let expected = assign_direct_arena(&intervals).unwrap();
        for _ in 0..32 {
            assert_eq!(assign_direct_arena(&intervals).unwrap(), expected);
        }
    }

    #[test]
    fn malformed_ids_and_component_overflow_fail_closed() {
        assert!(assign_direct_arena(&[interval(1, 0, 1, 1)]).is_err());
        assert!(assign_direct_arena(&[interval(0, 0, 1, 1), interval(0, 2, 3, 1)]).is_err());
        assert!(assign_direct_arena(&[interval(0, 0, 1, u32::MAX), interval(1, 0, 1, 1)]).is_err());
    }
}
