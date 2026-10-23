// SPDX-License-Identifier: 0BSD

//! Python entry points of the native v2 generator stages (WP3.1). Inputs are
//! the JSON tables of `loops/v2/generator_input.py`; outputs are plain tuples
//! of ids that the module decodes into the Python generator's objects.
use super::*;
use rusticol_core::loop_v2::generate::families::{FamilyInput, FamilyOutput, generate_families};
use rusticol_core::loop_v2::generate::schedules::{
    Content, ScheduleInput, build_schedules, build_skeletons,
};

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
);
type RawContent = (u32, u32, u32, u64, u32, Vec<(u32, Vec<u32>)>, i64);
type RawNode = (usize, u32, Vec<u32>, i64, Vec<(u32, u32)>);
type RawSchedule = (
    usize,
    Vec<RawSkeleton>,
    Vec<usize>,
    (usize, bool),
    Vec<(usize, Vec<u32>)>,
    Vec<RawNode>,
    i64,
    Vec<(Key, bool)>,
);
type RawSkeleton = (Vec<u32>, bool, u32);
type RawSkeletons = (Vec<RawContent>, Vec<Vec<(u16, i64)>>, Vec<Vec<RawSkeleton>>);
type RawSchedules = (
    Vec<RawContent>,
    Vec<u32>,
    Vec<Vec<(u16, i64)>>,
    Vec<RawSchedule>,
);

/// One process's native generation: the family recursion runs at
/// construction (`families.generate_families`); later stages read its
/// output in place.
#[pyclass(frozen)]
struct V2Generation {
    input: FamilyInput,
    families: FamilyOutput,
}

#[pymethods]
impl V2Generation {
    #[new]
    fn new(py: Python<'_>, input: &str) -> PyResult<Self> {
        let input: FamilyInput = serde_json::from_str(input).map_err(error)?;
        let families = py.detach(|| generate_families(&input)).map_err(error)?;
        Ok(Self { input, families })
    }

    /// `(positions, state_parents, state_labels, state_species, families,
    /// scaleless, rejected)`; a root's label is its cut species.
    fn families(&self) -> RawFamilies {
        let output = &self.families;
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
            output
                .families
                .iter()
                .map(|family| {
                    (
                        family.key.clone(),
                        family.rank,
                        family
                            .closures
                            .iter()
                            .map(|c| (c.state, c.symmetric, c.multiplicity))
                            .collect(),
                    )
                })
                .collect(),
            output.scaleless.clone(),
            output
                .rejected
                .iter()
                .map(|(key, reason)| (key.clone(), reason.as_str()))
                .collect(),
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
    /// (`schedules.build_schedules`). Returns `(contents, labels, orders,
    /// schedules)`; see `loops/v2/generator_input.py`.
    fn schedules(&self, py: Python<'_>, input: &str) -> PyResult<RawSchedules> {
        let input: ScheduleInput = serde_json::from_str(input).map_err(error)?;
        let output = py
            .detach(|| build_schedules(&self.input, &self.families, &input))
            .map_err(error)?;
        Ok((
            output.contents.into_iter().map(raw_content).collect(),
            output.labels,
            output.orders,
            output
                .schedules
                .into_iter()
                .map(|s| {
                    (
                        s.family,
                        s.skeletons
                            .into_iter()
                            .map(|k| (k.contents, k.half_weight, k.multiplicity))
                            .collect(),
                        s.viewed,
                        s.frame,
                        s.leaves,
                        s.nodes
                            .into_iter()
                            .map(|n| (n.depth, n.orders, n.suffix, n.rank_in, n.edges))
                            .collect(),
                        s.rank,
                        s.masters,
                    )
                })
                .collect(),
        ))
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

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<V2Generation>()?;
    Ok(())
}
