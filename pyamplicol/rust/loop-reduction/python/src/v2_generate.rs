// SPDX-License-Identifier: 0BSD

//! Python entry points of the native v2 generator stages (WP3.1). Inputs are
//! the JSON tables of `loops/v2/generator_input.py`; outputs are plain tuples
//! of ids that the module decodes into the Python generator's objects.
use super::*;
use rusticol_core::loop_v2::generate::arena::{self, Arena, Readings};
use rusticol_core::loop_v2::generate::cycles;
use rusticol_core::loop_v2::generate::families::{FamilyInput, FamilyOutput, generate_families};
use rusticol_core::loop_v2::generate::schedules::{
    Content, FamilySchedule, Retained, ScheduleInput, build_skeletons,
};
use std::sync::Arc;

type Key = Vec<(u64, u32)>;
type RawFamily = (Key, i64, Vec<(u32, bool, u32)>);
type RawFamilies = (
    Vec<(u32, Vec<u32>)>,
    Vec<i64>,
    Vec<u32>,
    Vec<u32>,
    Vec<RawFamily>,
    Vec<Key>,
    Vec<(Key, &'static str)>,
    Vec<RawFamily>,
);
type RawContent = (u32, u32, u32, u64, u32, Vec<(u32, Vec<u32>)>, i64);
type RawNode = (usize, u32, Vec<u32>, i64, Vec<(u32, u32)>);
/// `(family, skeletons, viewed, frame, leaves, nodes, rank, masters)`; a
/// rebuild sends no skeletons (its base's are reused).
type RawSchedule = (
    usize,
    Option<Vec<RawSkeleton>>,
    Vec<usize>,
    (usize, bool),
    Vec<(usize, Vec<u32>)>,
    Vec<RawNode>,
    i64,
    Vec<(Key, bool)>,
);
type RawSkeleton = (Vec<u32>, bool, u32);
type RawSkeletons = (Vec<RawContent>, Vec<Vec<(u16, i64)>>, Vec<Vec<RawSkeleton>>);
/// `(handle, first new content id, new contents, labels, orders,
/// schedules)`.
type RawSchedules = (
    usize,
    usize,
    Vec<RawContent>,
    Vec<u32>,
    Vec<Vec<(u16, i64)>>,
    Vec<RawSchedule>,
);

/// One process's native generation: the family recursion runs at
/// construction (`families.generate_families`); later stages read its
/// output in place. Every schedule build is retained in the generation's
/// arena (`loop_v2::generate::arena`) under the handle it returns.
#[pyclass(frozen)]
pub(crate) struct V2Generation {
    input: FamilyInput,
    families: FamilyOutput,
    arena: Arena,
}

impl V2Generation {
    /// A retained build (for the in-crate readers, `v2_seeds.rs`).
    pub(crate) fn retained(&self, handle: usize) -> PyResult<Arc<Retained>> {
        self.arena.get(handle).map_err(error)
    }

    fn raw_schedules(
        build: &Retained,
        handle: usize,
        first: usize,
        skeletons: bool,
    ) -> RawSchedules {
        let raw = |s: &FamilySchedule| -> RawSchedule {
            (
                s.family,
                skeletons.then(|| {
                    build
                        .family_skeletons(s.family)
                        .iter()
                        .map(|k| (k.contents.clone(), k.half_weight, k.multiplicity))
                        .collect()
                }),
                s.viewed.clone(),
                s.frame,
                s.leaves.clone(),
                build
                    .nodes(s)
                    .iter()
                    .map(|n| {
                        (
                            n.depth,
                            n.orders,
                            n.suffix.clone(),
                            n.rank_in,
                            n.edges.clone(),
                        )
                    })
                    .collect(),
                s.rank,
                s.masters.clone(),
            )
        };
        (
            handle,
            first,
            build.contents()[first..]
                .iter()
                .cloned()
                .map(raw_content)
                .collect(),
            build.labels().to_vec(),
            build.order_vectors(),
            build.schedules().iter().map(raw).collect(),
        )
    }
}

/// Little-endian bytes of a reading table: `(u64 offsets, u32 contents)`.
fn reading_bytes(readings: &Readings) -> (Vec<u8>, Vec<u8>) {
    (
        readings
            .offsets
            .iter()
            .flat_map(|&o| (o as u64).to_le_bytes())
            .collect(),
        readings
            .contents
            .iter()
            .flat_map(|c| c.to_le_bytes())
            .collect(),
    )
}

/// Flags as bytes (one per flag, 0 or 1).
fn flags(bytes: &[u8]) -> PyResult<Vec<bool>> {
    bytes
        .iter()
        .map(|&b| match b {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(error("keep flags must be 0 or 1 bytes")),
        })
        .collect()
}

#[pymethods]
impl V2Generation {
    #[new]
    fn new(py: Python<'_>, input: &str) -> PyResult<Self> {
        let input: FamilyInput = serde_json::from_str(input).map_err(error)?;
        let families = py.detach(|| generate_families(&input)).map_err(error)?;
        Ok(Self {
            input,
            families,
            arena: Arena::default(),
        })
    }

    /// `(positions, state_parents, state_labels, state_species, families,
    /// scaleless, rejected, scaleless_families)`; a root's label is its cut
    /// species; the scaleless families' states follow every admitted one's.
    fn families(&self) -> RawFamilies {
        let output = &self.families;
        let raw = |family: &rusticol_core::loop_v2::generate::families::Family| -> RawFamily {
            (
                family.key.clone(),
                family.rank,
                family
                    .closures
                    .iter()
                    .map(|c| (c.state, c.symmetric, c.multiplicity))
                    .collect(),
            )
        };
        let labels = output
            .state_position
            .iter()
            .zip(&output.state_species)
            .map(|(&position, &species)| {
                if position == u32::MAX {
                    species
                } else {
                    position
                }
            })
            .collect();
        (
            output
                .positions
                .iter()
                .map(|position| (position.vertex, position.blocks.clone()))
                .collect(),
            output.state_parent.clone(),
            labels,
            output.state_species.clone(),
            output.families.iter().map(raw).collect(),
            output.scaleless.clone(),
            output
                .rejected
                .iter()
                .map(|(key, reason)| (key.clone(), reason.as_str()))
                .collect(),
            output.scaleless_families.iter().map(raw).collect(),
        )
    }

    /// The skeletons of every family (`skeletons.family_skeletons`):
    /// `(contents, orders, skeletons per family)`.
    fn skeletons(&self, py: Python<'_>, input: &str) -> PyResult<RawSkeletons> {
        let input: ScheduleInput = serde_json::from_str(input).map_err(error)?;
        let (contents, orders, skeletons) = py
            .detach(|| build_skeletons(&self.input, &self.families, &input))
            .map_err(error)?;
        Ok((
            contents.into_iter().map(raw_content).collect(),
            orders,
            skeletons
                .into_iter()
                .map(|family| {
                    family
                        .into_iter()
                        .map(|k| (k.contents, k.half_weight, k.multiplicity))
                        .collect()
                })
                .collect(),
        ))
    }

    /// Skeletons, the view and the schedule-B tries
    /// (`schedules.build_schedules`), retained in the arena. Returns
    /// `(handle, 0, contents, labels, orders, schedules)` with every
    /// family's skeletons; see `loops/v2/generator_input.py`.
    fn schedules(&self, py: Python<'_>, input: &str) -> PyResult<RawSchedules> {
        let input: ScheduleInput = serde_json::from_str(input).map_err(error)?;
        let build = py
            .detach(|| arena::build_retained(&self.input, &self.families, &input))
            .map_err(error)?;
        let raw_build = |build: &Retained, handle| Self::raw_schedules(build, handle, 0, true);
        let raw = raw_build(&build, self.arena.len());
        let handle = self.arena.insert(build);
        debug_assert_eq!(handle, raw.0);
        Ok(raw)
    }

    /// The schedules of `input` (its fixed frames and keep flags) rebuilt
    /// from the skeletons of the complete build `base` (the LC/NLC prune):
    /// retained under a new handle. `keep` holds one byte per skeleton of
    /// every scheduled family in order, or with `readings` one per reading
    /// of the base (`Retained::readings`; a skeleton is kept when its F
    /// reading is). Returns `(handle, first new content id, the new
    /// contents, labels, orders, schedules)`, without skeletons.
    #[pyo3(signature = (base, input, keep, readings = false))]
    fn rebuild(
        &self,
        py: Python<'_>,
        base: usize,
        input: &str,
        keep: &[u8],
        readings: bool,
    ) -> PyResult<RawSchedules> {
        let mut input: ScheduleInput = serde_json::from_str(input).map_err(error)?;
        if input.keep.is_some() {
            return Err(error("rebuild keep flags come as bytes"));
        }
        let retained = self.retained(base)?;
        let keep = flags(keep)?;
        input.keep = Some(if readings {
            arena::keep_from_readings(&retained, &keep).map_err(error)?
        } else {
            let mut rest = keep.as_slice();
            let mut out = Vec::new();
            for family in 0..retained.family_count() {
                let count = retained.family_skeletons(family).len();
                if rest.len() < count {
                    return Err(error("too few keep flags for the base's skeletons"));
                }
                out.push(rest[..count].to_vec());
                rest = &rest[count..];
            }
            if !rest.is_empty() {
                return Err(error("too many keep flags for the base's skeletons"));
            }
            out
        });
        let build = py
            .detach(|| arena::rebuild(&self.input, &self.families, &retained, &input))
            .map_err(error)?;
        let first = retained.content_count();
        let raw = Self::raw_schedules(&build, self.arena.len(), first, false);
        self.arena.insert(build);
        Ok(raw)
    }

    /// The readings of a build (`reference_executor.reading_skeletons`):
    /// little-endian `(u64 offsets, u32 content ids)` bytes and the distinct
    /// content ids, first seen (the seed stage's content order).
    fn readings(&self, py: Python<'_>, handle: usize) -> PyResult<(Vec<u8>, Vec<u8>, Vec<u32>)> {
        let retained = self.retained(handle)?;
        Ok(py.detach(|| {
            let readings = retained.readings();
            let (offsets, contents) = reading_bytes(&readings);
            (offsets, contents, readings.dense().0)
        }))
    }

    /// `MonomialLayout`'s rank bound of a build (`Retained::max_rank`).
    fn max_rank(&self, handle: usize) -> PyResult<i64> {
        self.retained(handle)?.max_rank().map_err(error)
    }
}

fn raw_content(c: Content) -> RawContent {
    (
        c.state,
        c.loop_in,
        c.loop_out,
        c.mask,
        c.orders,
        c.terms
            .into_iter()
            .map(|term| (term.vertex, term.blocks))
            .collect(),
        c.q_degree,
    )
}

/// SU(N_c) overlaps of trace products (`loop_v2::generate::cycles`): per
/// product its nonzero `(N_c power, coefficient)` pairs, or None where it
/// has more labels than the subset sum takes (the caller's trace algebra
/// then evaluates it). Products are lists of traces of adjoint labels, each
/// label twice.
/// Per product its `(N_c power, coefficient)` pairs, or None.
type Overlaps = Vec<Option<Vec<(i32, i64)>>>;

#[pyfunction]
fn trace_overlaps(py: Python<'_>, products: Vec<Vec<Vec<i64>>>) -> PyResult<Overlaps> {
    py.detach(|| {
        products
            .iter()
            .map(|traces| {
                let labels: std::collections::BTreeSet<i64> =
                    traces.iter().flatten().copied().collect();
                if labels.len() > cycles::MAX_LABELS {
                    return Ok(None);
                }
                cycles::trace_overlap(traces).map(Some)
            })
            .collect::<Result<Vec<_>, _>>()
    })
    .map_err(error)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<V2Generation>()?;
    module.add_function(wrap_pyfunction!(trace_overlaps, module)?)?;
    Ok(())
}
