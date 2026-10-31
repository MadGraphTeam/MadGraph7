// SPDX-License-Identifier: 0BSD

//! Larger tiles (cargo feature `tile-override`).
//!
//! RustiCol sizes a compiled engine's tiles to fit a cache target, at most
//! 32 points. An engine whose reduction footprint per point exceeds the target
//! gets tiles of a single point, and as many engine calls as points.
//! `RUSTICOL_TILE_POINTS=N` (read when an artifact is loaded) gives such an
//! engine tiles of N points instead, for its leaves and its reduction alike;
//! other engines keep their cache-sized tiles. It also lets the LC replay
//! evaluate several colour mappings per engine call (N / points).

use crate::{RusticolError, RusticolResult};

const TILE_POINTS: &str = "RUSTICOL_TILE_POINTS";

/// The points per tile of `RUSTICOL_TILE_POINTS` (`None` if unset or empty).
pub(crate) fn tile_override() -> RusticolResult<Option<u32>> {
    let Some(value) = std::env::var_os(TILE_POINTS).filter(|value| !value.is_empty()) else {
        return Ok(None);
    };
    match value.to_str().and_then(|value| value.parse::<u32>().ok()) {
        Some(points) if points > 0 => Ok(Some(points)),
        _ => Err(RusticolError::invalid_argument(format!(
            "{TILE_POINTS}={}: expected a positive number of points",
            value.to_string_lossy()
        ))),
    }
}
