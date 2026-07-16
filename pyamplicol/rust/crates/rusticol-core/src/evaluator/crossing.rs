// SPDX-License-Identifier: 0BSD

use super::super::*;

pub(crate) fn artifact_path(root: &Path, value: &str) -> RusticolResult<PathBuf> {
    Ok(root.join(confined_internal_path(value, "evaluator artifact path")?))
}

pub(crate) fn apply_input_crossing_map(
    batch: Vec<Vec<[f64; 4]>>,
    expected_legs: usize,
    input_crossing_map: Option<&[InputCrossingMapEntry]>,
) -> RusticolResult<Vec<Vec<[f64; 4]>>> {
    let Some(map) = input_crossing_map else {
        return Ok(batch);
    };
    if map.len() != expected_legs {
        return Err(RusticolError::invalid_argument(format!(
            "input crossing map has {} entries, expected {expected_legs}",
            map.len()
        )));
    }
    let mut seen = vec![false; expected_legs];
    for entry in map {
        if entry.target_index >= expected_legs || entry.source_index >= expected_legs {
            return Err(RusticolError::invalid_argument(
                "input crossing map references an out-of-range external leg",
            ));
        }
        if seen[entry.target_index] {
            return Err(RusticolError::invalid_argument(
                "input crossing map contains a duplicate target index",
            ));
        }
        seen[entry.target_index] = true;
    }
    if seen.iter().any(|value| !*value) {
        return Err(RusticolError::invalid_argument(
            "input crossing map does not cover every target index",
        ));
    }
    let mut mapped_batch = Vec::with_capacity(batch.len());
    for point in batch {
        let mut mapped = vec![[0.0; 4]; expected_legs];
        for entry in map {
            let source = point[entry.source_index];
            mapped[entry.target_index] = [
                entry.sign * source[0],
                entry.sign * source[1],
                entry.sign * source[2],
                entry.sign * source[3],
            ];
        }
        mapped_batch.push(mapped);
    }
    Ok(mapped_batch)
}

#[cfg(feature = "symbolica-runtime")]
pub(crate) fn validate_input_crossing_map(
    expected_legs: usize,
    input_crossing_map: Option<&[InputCrossingMapEntry]>,
) -> RusticolResult<Option<&[InputCrossingMapEntry]>> {
    let Some(map) = input_crossing_map else {
        return Ok(None);
    };
    if map.len() != expected_legs {
        return Err(RusticolError::invalid_argument(format!(
            "input crossing map has {} entries, expected {expected_legs}",
            map.len()
        )));
    }
    let mut seen = vec![false; expected_legs];
    for entry in map {
        if entry.target_index >= expected_legs || entry.source_index >= expected_legs {
            return Err(RusticolError::invalid_argument(
                "input crossing map references an out-of-range external leg",
            ));
        }
        if seen[entry.target_index] {
            return Err(RusticolError::invalid_argument(
                "input crossing map contains a duplicate target index",
            ));
        }
        seen[entry.target_index] = true;
    }
    if seen.iter().any(|value| !*value) {
        return Err(RusticolError::invalid_argument(
            "input crossing map does not cover every target index",
        ));
    }
    Ok(Some(map))
}

#[cfg(feature = "symbolica-runtime")]
pub(crate) fn apply_input_crossing_map_generic<T>(
    batch: &[Vec<[T; 4]>],
    expected_legs: usize,
    input_crossing_map: Option<&[InputCrossingMapEntry]>,
) -> RusticolResult<Vec<Vec<[T; 4]>>>
where
    T: RusticolHighPrecisionNumber,
    Complex<T>: Real + EvaluationDomain,
{
    let Some(map) = validate_input_crossing_map(expected_legs, input_crossing_map)? else {
        return Ok(batch.to_vec());
    };
    let mut mapped_batch = Vec::with_capacity(batch.len());
    for point in batch {
        let mut mapped = vec![std::array::from_fn(|_| T::new_zero()); expected_legs];
        for entry in map {
            let source = &point[entry.source_index];
            for component in 0..4 {
                mapped[entry.target_index][component] =
                    T::from(entry.sign) * source[component].clone();
            }
        }
        mapped_batch.push(mapped);
    }
    Ok(mapped_batch)
}

pub(crate) fn apply_lc_topology_label_permutation(
    batch: &[Vec<[f64; 4]>],
    expected_legs: usize,
    mapping: &[(usize, usize)],
) -> RusticolResult<Vec<Vec<[f64; 4]>>> {
    let mut seen = vec![false; expected_legs];
    for (representative_index, sector_index) in mapping {
        if *representative_index >= expected_legs || *sector_index >= expected_legs {
            return Err(RusticolError::invalid_argument(
                "LC topology replay label permutation references an out-of-range external leg",
            ));
        }
        if seen[*representative_index] {
            return Err(RusticolError::invalid_argument(
                "LC topology replay label permutation contains a duplicate representative label",
            ));
        }
        seen[*representative_index] = true;
    }
    let mut mapped_batch = Vec::with_capacity(batch.len());
    for point in batch {
        if point.len() != expected_legs {
            return Err(RusticolError::invalid_argument(format!(
                "LC topology replay point has {} external legs, expected {expected_legs}",
                point.len(),
            )));
        }
        let mut mapped = point.clone();
        for (representative_index, sector_index) in mapping {
            mapped[*representative_index] = point[*sector_index];
        }
        mapped_batch.push(mapped);
    }
    Ok(mapped_batch)
}

pub(crate) fn apply_lc_topology_label_permutations(
    batch: &[Vec<[f64; 4]>],
    expected_legs: usize,
    mappings: &[Vec<(usize, usize)>],
) -> RusticolResult<Vec<Vec<[f64; 4]>>> {
    let mut expanded_batch = Vec::with_capacity(batch.len() * mappings.len());
    for mapping in mappings {
        expanded_batch.extend(apply_lc_topology_label_permutation(
            batch,
            expected_legs,
            mapping,
        )?);
    }
    Ok(expanded_batch)
}

#[cfg(feature = "symbolica-runtime")]
pub(crate) fn apply_lc_topology_label_permutation_generic<T>(
    batch: &[Vec<[T; 4]>],
    expected_legs: usize,
    mapping: &[(usize, usize)],
) -> RusticolResult<Vec<Vec<[T; 4]>>>
where
    T: RusticolHighPrecisionNumber,
    Complex<T>: Real + EvaluationDomain,
{
    let mut seen = vec![false; expected_legs];
    for (representative_index, sector_index) in mapping {
        if *representative_index >= expected_legs || *sector_index >= expected_legs {
            return Err(RusticolError::invalid_argument(
                "LC topology replay label permutation references an out-of-range external leg",
            ));
        }
        if seen[*representative_index] {
            return Err(RusticolError::invalid_argument(
                "LC topology replay label permutation contains a duplicate representative label",
            ));
        }
        seen[*representative_index] = true;
    }
    let mut mapped_batch = Vec::with_capacity(batch.len());
    for point in batch {
        if point.len() != expected_legs {
            return Err(RusticolError::invalid_argument(format!(
                "LC topology replay point has {} external legs, expected {expected_legs}",
                point.len(),
            )));
        }
        let mut mapped = point.clone();
        for (representative_index, sector_index) in mapping {
            mapped[*representative_index] = point[*sector_index].clone();
        }
        mapped_batch.push(mapped);
    }
    Ok(mapped_batch)
}

#[cfg(feature = "symbolica-runtime")]
pub(crate) fn apply_lc_topology_label_permutations_generic<T>(
    batch: &[Vec<[T; 4]>],
    expected_legs: usize,
    mappings: &[Vec<(usize, usize)>],
) -> RusticolResult<Vec<Vec<[T; 4]>>>
where
    T: RusticolHighPrecisionNumber,
    Complex<T>: Real + EvaluationDomain,
{
    let mut expanded_batch = Vec::with_capacity(batch.len() * mappings.len());
    for mapping in mappings {
        expanded_batch.extend(apply_lc_topology_label_permutation_generic(
            batch,
            expected_legs,
            mapping,
        )?);
    }
    Ok(expanded_batch)
}

pub(crate) fn replay_mappings_per_expanded_batch(n_points: usize) -> usize {
    if n_points == 0 {
        return 1;
    }
    (MAX_LC_TOPOLOGY_REPLAY_EXPANDED_POINTS / n_points).max(1)
}

#[cfg(feature = "symbolica-runtime")]
pub(crate) fn decimal_digits_to_bits(decimal_digits: u32) -> u32 {
    (decimal_digits as f64 * std::f64::consts::LOG2_10).ceil() as u32
}
