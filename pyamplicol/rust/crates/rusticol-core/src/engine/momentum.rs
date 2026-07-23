// SPDX-License-Identifier: 0BSD

use super::*;

#[derive(Clone, Copy)]
enum F64MomentumBatchStorage<'a> {
    Contiguous(&'a [f64]),
    Nested(&'a [Vec<[f64; 4]>]),
}

/// Borrowed row-major momentum input used by the ordinary f64 lane.
///
/// Contiguous inputs have layout `[point][external][E, px, py, pz]`.  An
/// optional crossing lookup is validated and normalized when the runtime is
/// loaded, so individual momentum reads need no allocation or map search.
#[derive(Clone, Copy)]
pub(super) struct F64MomentumBatchView<'a> {
    storage: F64MomentumBatchStorage<'a>,
    point_count: usize,
    external_count: usize,
    crossing_lookup: Option<&'a [InputCrossingMapEntry]>,
}

#[derive(Clone, Copy)]
pub(super) enum F64MomentumPointView<'a> {
    ContiguousIdentity {
        values: &'a [f64],
        external_count: usize,
    },
    ContiguousCrossed {
        values: &'a [f64],
        external_count: usize,
        crossing_lookup: &'a [InputCrossingMapEntry],
    },
    Nested(&'a [[f64; 4]]),
}

pub(super) trait F64MomentumPoint {
    fn momentum(&self, external_index: usize) -> Option<[f64; 4]>;
}

impl F64MomentumPoint for F64MomentumPointView<'_> {
    #[inline(always)]
    fn momentum(&self, external_index: usize) -> Option<[f64; 4]> {
        match self {
            Self::ContiguousIdentity {
                values,
                external_count,
            } => {
                if external_index >= *external_count {
                    return None;
                }
                let start = external_index * 4;
                Some([
                    values[start],
                    values[start + 1],
                    values[start + 2],
                    values[start + 3],
                ])
            }
            Self::ContiguousCrossed {
                values,
                external_count,
                crossing_lookup,
            } => {
                if external_index >= *external_count {
                    return None;
                }
                let entry = &crossing_lookup[external_index];
                debug_assert_eq!(entry.target_index, external_index);
                let start = entry.source_index * 4;
                Some([
                    entry.sign * values[start],
                    entry.sign * values[start + 1],
                    entry.sign * values[start + 2],
                    entry.sign * values[start + 3],
                ])
            }
            Self::Nested(values) => values.get(external_index).copied(),
        }
    }
}

impl F64MomentumPoint for [[f64; 4]] {
    #[inline(always)]
    fn momentum(&self, external_index: usize) -> Option<[f64; 4]> {
        self.get(external_index).copied()
    }
}

impl F64MomentumPoint for Vec<[f64; 4]> {
    #[inline(always)]
    fn momentum(&self, external_index: usize) -> Option<[f64; 4]> {
        self.get(external_index).copied()
    }
}

impl<const N: usize> F64MomentumPoint for [[f64; 4]; N] {
    #[inline(always)]
    fn momentum(&self, external_index: usize) -> Option<[f64; 4]> {
        self.get(external_index).copied()
    }
}

impl<'a> F64MomentumBatchView<'a> {
    pub(super) fn from_contiguous_prevalidated(
        values: &'a [f64],
        point_count: usize,
        external_count: usize,
        crossing_lookup: Option<&'a [InputCrossingMapEntry]>,
    ) -> RusticolResult<Self> {
        if point_count == 0 {
            return Err(RusticolError::invalid_argument(
                "point_count must be positive",
            ));
        }
        let values_per_point = external_count
            .checked_mul(4)
            .ok_or_else(|| RusticolError::invalid_argument("momentum shape overflow"))?;
        let expected = point_count
            .checked_mul(values_per_point)
            .ok_or_else(|| RusticolError::invalid_argument("momentum shape overflow"))?;
        if values.len() != expected {
            return Err(RusticolError::invalid_argument(format!(
                "momenta contain {} values, expected {expected} for shape ({point_count}, {external_count}, 4)",
                values.len()
            )));
        }
        if crossing_lookup.is_some_and(|lookup| lookup.len() != external_count) {
            return Err(RusticolError::integrity(
                "prevalidated input crossing lookup has an inconsistent length",
            ));
        }
        Ok(Self {
            storage: F64MomentumBatchStorage::Contiguous(values),
            point_count,
            external_count,
            crossing_lookup,
        })
    }

    pub(super) fn from_nested(
        values: &'a [Vec<[f64; 4]>],
        external_count: usize,
    ) -> RusticolResult<Self> {
        if values.is_empty() {
            return Err(RusticolError::invalid_argument(
                "point_count must be positive",
            ));
        }
        if let Some(point) = values.iter().find(|point| point.len() != external_count) {
            return Err(RusticolError::invalid_argument(format!(
                "momentum point contains {} external legs, expected {external_count}",
                point.len()
            )));
        }
        Ok(Self {
            storage: F64MomentumBatchStorage::Nested(values),
            point_count: values.len(),
            external_count,
            crossing_lookup: None,
        })
    }

    pub(super) fn point_count(self) -> usize {
        self.point_count
    }

    pub(super) fn external_count(self) -> usize {
        self.external_count
    }

    #[inline(always)]
    pub(super) fn point(self, point_index: usize) -> F64MomentumPointView<'a> {
        assert!(point_index < self.point_count);
        match self.storage {
            F64MomentumBatchStorage::Contiguous(values) => {
                let values_per_point = self.external_count * 4;
                let start = point_index * values_per_point;
                let values = &values[start..start + values_per_point];
                match self.crossing_lookup {
                    Some(crossing_lookup) => F64MomentumPointView::ContiguousCrossed {
                        values,
                        external_count: self.external_count,
                        crossing_lookup,
                    },
                    None => F64MomentumPointView::ContiguousIdentity {
                        values,
                        external_count: self.external_count,
                    },
                }
            }
            F64MomentumBatchStorage::Nested(values) => {
                F64MomentumPointView::Nested(values[point_index].as_slice())
            }
        }
    }

    pub(super) fn subview(self, start: usize, end: usize) -> RusticolResult<Self> {
        if start >= end || end > self.point_count {
            return Err(RusticolError::invalid_argument(
                "momentum batch subview has an invalid point range",
            ));
        }
        match self.storage {
            F64MomentumBatchStorage::Contiguous(values) => {
                let values_per_point = self.external_count * 4;
                Self::from_contiguous_prevalidated(
                    &values[start * values_per_point..end * values_per_point],
                    end - start,
                    self.external_count,
                    self.crossing_lookup,
                )
            }
            F64MomentumBatchStorage::Nested(values) => {
                Self::from_nested(&values[start..end], self.external_count)
            }
        }
    }

    pub(super) fn materialize_nested(self) -> Vec<Vec<[f64; 4]>> {
        (0..self.point_count)
            .map(|point_index| {
                let point = self.point(point_index);
                (0..self.external_count)
                    .map(|external_index| {
                        point
                            .momentum(external_index)
                            .expect("validated momentum view covers every external leg")
                    })
                    .collect()
            })
            .collect()
    }
}

pub(super) fn prevalidate_input_crossing_lookup(
    expected_legs: usize,
    input_crossing_map: Option<Vec<InputCrossingMapEntry>>,
) -> RusticolResult<Option<Vec<InputCrossingMapEntry>>> {
    let Some(mut map) = input_crossing_map else {
        return Ok(None);
    };
    if map.len() != expected_legs {
        return Err(RusticolError::invalid_argument(format!(
            "input crossing map has {} entries, expected {expected_legs}",
            map.len()
        )));
    }
    map.sort_unstable_by_key(|entry| entry.target_index);
    for (target_index, entry) in map.iter().enumerate() {
        if entry.target_index != target_index || entry.source_index >= expected_legs {
            return Err(RusticolError::invalid_argument(
                "input crossing map is not a complete in-range target lookup",
            ));
        }
    }
    Ok(Some(map))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_view_matches_materialized_crossing_and_subviews() {
        let values = [
            1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0, 5.0, 6.0, 7.0,
            8.0, 50.0, 60.0, 70.0, 80.0, 500.0, 600.0, 700.0, 800.0,
        ];
        let lookup = prevalidate_input_crossing_lookup(
            3,
            Some(vec![
                InputCrossingMapEntry {
                    target_index: 2,
                    source_index: 0,
                    sign: -1.0,
                },
                InputCrossingMapEntry {
                    target_index: 0,
                    source_index: 2,
                    sign: 1.0,
                },
                InputCrossingMapEntry {
                    target_index: 1,
                    source_index: 1,
                    sign: 1.0,
                },
            ]),
        )
        .unwrap()
        .unwrap();
        let view = F64MomentumBatchView::from_contiguous_prevalidated(&values, 2, 3, Some(&lookup))
            .unwrap();
        let nested = vec![
            vec![
                [1.0, 2.0, 3.0, 4.0],
                [10.0, 20.0, 30.0, 40.0],
                [100.0, 200.0, 300.0, 400.0],
            ],
            vec![
                [5.0, 6.0, 7.0, 8.0],
                [50.0, 60.0, 70.0, 80.0],
                [500.0, 600.0, 700.0, 800.0],
            ],
        ];
        let expected = apply_input_crossing_map(nested, 3, Some(&lookup)).unwrap();

        assert_eq!(view.materialize_nested(), expected);
        assert_eq!(
            view.point(0).momentum(0),
            Some([100.0, 200.0, 300.0, 400.0])
        );
        assert_eq!(view.point(0).momentum(2), Some([-1.0, -2.0, -3.0, -4.0]));
        assert_eq!(
            view.subview(1, 2).unwrap().materialize_nested(),
            expected[1..]
        );
    }

    #[test]
    fn contiguous_view_rejects_invalid_shapes() {
        assert!(F64MomentumBatchView::from_contiguous_prevalidated(&[], 0, 2, None).is_err());
        assert!(F64MomentumBatchView::from_contiguous_prevalidated(&[0.0; 7], 1, 2, None).is_err());
    }
}
