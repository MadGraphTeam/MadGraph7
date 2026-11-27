// SPDX-License-Identifier: 0BSD

//! The retained generation arena (WP3.1 I10 remainder, package native-seeds
//! B1; RUST-MIGRATION.md §3.B).
//!
//! A native generation (`V2Generation`, binding `v2_generate.rs`) keeps every
//! schedule build it made, with its skeletons, interning state, view and
//! tries ([`Retained`]), and hands out handles: indices into the arena. The
//! later stages read a build in place instead of its decoded Python objects:
//!
//! - the seed stage (`seeds.rs`) takes a build's readings
//!   ([`Retained::readings`]), the skeletons of `reference_executor.
//!   reading_skeletons`, instead of bytes tables encoded in Python;
//! - the LC/NLC prune rebuilds the schedules from a complete build's
//!   skeletons ([`schedules::rebuild`]) with keep flags evaluated on them;
//! - `MonomialLayout`'s largest rank ([`Retained::max_rank`]);
//! - plan lowering (package native-lowering, D3) reads builds through the
//!   accessors below (`contents`, `orders`, `labels`, `schedules`,
//!   `family_skeletons`, `leaf_skeleton`, `nodes`, `readings`).
//!
//! Handles are stable: a build is never changed once added. A build no
//! stage reads any more (the complete build once the LC/NLC prune has
//! rebuilt from it) can be released ([`Arena::release`]); its handle then
//! reads as an error and is never reused.

use super::FastMap;
use super::orders::OrderSpace;
use super::schedules::{self, Content, FamilySchedule, Retained, Skeleton, TrieNode};
use std::sync::{Arc, Mutex, PoisonError};

/// The retained builds of one generation, by handle (None: released).
#[derive(Debug, Default)]
pub struct Arena {
    builds: Mutex<Vec<Option<Arc<Retained>>>>,
}

impl Arena {
    /// Keep a build; its handle.
    pub fn insert(&self, build: Retained) -> usize {
        let mut builds = self.builds.lock().unwrap_or_else(PoisonError::into_inner);
        builds.push(Some(Arc::new(build)));
        builds.len() - 1
    }

    /// The build of a handle.
    pub fn get(&self, handle: usize) -> Result<Arc<Retained>, String> {
        match self
            .builds
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(handle)
        {
            Some(Some(build)) => Ok(Arc::clone(build)),
            Some(None) => Err(format!("schedule build {handle} was released")),
            None => Err(format!("no retained schedule build {handle}")),
        }
    }

    /// Drop the arena's hold on a build: its memory is freed once no reader
    /// holds it (a rebuild shares only its skeletons, [`super::schedules::
    /// rebuild`]). Releasing a released handle is a no-op.
    pub fn release(&self, handle: usize) -> Result<(), String> {
        match self
            .builds
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_mut(handle)
        {
            Some(slot) => {
                slot.take();
                Ok(())
            }
            None => Err(format!("no retained schedule build {handle}")),
        }
    }

    /// The number of handles given out (released builds included).
    pub fn len(&self) -> usize {
        self.builds
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// The readings of a build as content-id spans (`reading_skeletons`): per
/// schedule, per trie leaf, the skeleton in its F reading, then as its trie
/// reads it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Readings {
    /// `offsets[i]..offsets[i + 1]` spans reading `i` in `contents`.
    pub offsets: Vec<usize>,
    pub contents: Vec<u32>,
}

impl Readings {
    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reading(&self, index: usize) -> &[u32] {
        &self.contents[self.offsets[index]..self.offsets[index + 1]]
    }

    /// The distinct contents, first seen, and the readings over their
    /// positions in that list (the seed stage's dense content ids).
    pub fn dense(&self) -> (Vec<u32>, Vec<u32>) {
        let mut index: FastMap<u32, u32> = FastMap::default();
        let mut used = Vec::new();
        let dense = self
            .contents
            .iter()
            .map(|&content| {
                // Distinct u32 contents: a dense id fits u32.
                *index.entry(content).or_insert_with(|| {
                    used.push(content);
                    (used.len() - 1) as u32
                })
            })
            .collect();
        (used, dense)
    }
}

impl Retained {
    /// The interned contents (by id; every id a skeleton, reading or label
    /// refers to).
    pub fn contents(&self) -> &[Content] {
        &self.state.contents
    }

    /// The interned order vectors (`Content::orders`, `Skeleton::orders`,
    /// `TrieNode::orders` are ids into it).
    pub fn orders(&self) -> &OrderSpace {
        &self.state.orders
    }

    /// The order vectors by id.
    pub fn order_vectors(&self) -> Vec<super::orders::OrderVec> {
        self.state.order_vectors()
    }

    /// The number of scheduled families (the scheduled list,
    /// `ScheduleInput::scaleless`).
    pub fn family_count(&self) -> usize {
        self.skeletons.len()
    }

    /// `labels[l]`: the content of segment label `l` (`SkeletonView`).
    pub fn labels(&self) -> &[u32] {
        &self.labels
    }

    /// The tries, in family order (families without a viewed skeleton have
    /// none).
    pub fn schedules(&self) -> &[FamilySchedule] {
        &self.schedules
    }

    /// Every skeleton of a scheduled family (`FamilySchedule::family`).
    pub fn family_skeletons(&self, family: usize) -> &[Skeleton] {
        &self.skeletons[family]
    }

    /// The leaf skeleton of a trie leaf.
    pub fn leaf_skeleton(&self, schedule: &FamilySchedule, leaf: usize) -> &Skeleton {
        let (viewed, _ring) = &schedule.leaves[leaf];
        &self.skeletons[schedule.family][schedule.viewed[*viewed]]
    }

    /// The trie nodes of a schedule.
    pub fn nodes<'a>(&self, schedule: &'a FamilySchedule) -> &'a [TrieNode] {
        &schedule.nodes
    }

    /// The readings of `reference_executor.reading_skeletons(plan)`.
    pub fn readings(&self) -> Readings {
        let leaves: usize = self.schedules.iter().map(|s| s.leaves.len()).sum();
        let mut offsets = Vec::with_capacity(2 * leaves + 1);
        let mut contents = Vec::new();
        offsets.push(0);
        for schedule in &self.schedules {
            for (leaf, (_viewed, ring)) in schedule.leaves.iter().enumerate() {
                contents.extend_from_slice(&self.leaf_skeleton(schedule, leaf).contents);
                offsets.push(contents.len());
                contents.extend(ring.iter().map(|&label| self.labels[label as usize]));
                offsets.push(contents.len());
            }
        }
        Readings { offsets, contents }
    }

    /// Per scheduled family and skeleton, the index of its F reading among
    /// [`Self::readings`] (`None` for a skeleton the view left out).
    pub fn reading_of(&self) -> Vec<Vec<Option<usize>>> {
        let mut out: Vec<Vec<Option<usize>>> = self
            .skeletons
            .iter()
            .map(|family| vec![None; family.len()])
            .collect();
        let mut reading = 0;
        for schedule in &self.schedules {
            for (viewed, _ring) in &schedule.leaves {
                out[schedule.family][schedule.viewed[*viewed]] = Some(reading);
                reading += 2;
            }
        }
        out
    }

    /// `MonomialLayout`'s rank bound (`LoopSlice.generate`): the largest
    /// `rank_in + max(edge q-degree)` over every trie node. Refused, as
    /// Python's `max` refuses them, without nodes or for a node without
    /// edges.
    pub fn max_rank(&self) -> Result<i64, String> {
        let mut best: Option<i64> = None;
        for node in self.schedules.iter().flat_map(|s| s.nodes.iter()) {
            let edge = node
                .edges
                .iter()
                .map(|&(label, _)| {
                    self.state.contents[self.labels[label as usize] as usize].q_degree
                })
                .max()
                .ok_or("a trie node without edges")?;
            let rank = node.rank_in + edge;
            best = Some(best.map_or(rank, |b| b.max(rank)));
        }
        best.ok_or_else(|| "no trie nodes to bound the rank".to_string())
    }

    /// Whether this build kept every skeleton (a rebuild base).
    pub fn complete(&self) -> bool {
        self.complete
    }

    /// The number of interned contents (a rebuild adds its new reflections
    /// after its base's).
    pub fn content_count(&self) -> usize {
        self.state.contents.len()
    }
}

/// Keep flags of a rebuild from per-reading flags of a build's readings: a
/// skeleton is kept when its F reading is. A skeleton the view left out has
/// no reading and is refused, as `class_of[skeleton]` refuses it in
/// `LoopSlice._prune_colour`.
pub fn keep_from_readings(
    base: &Retained,
    reading_keep: &[bool],
) -> Result<Vec<Vec<bool>>, String> {
    if reading_keep.len() != 2 * base.schedules.iter().map(|s| s.leaves.len()).sum::<usize>() {
        return Err("reading flags of another build".into());
    }
    base.reading_of()
        .into_iter()
        .enumerate()
        .map(|(family, row)| {
            row.into_iter()
                .enumerate()
                .map(|(skeleton, reading)| {
                    reading.map(|r| reading_keep[r]).ok_or_else(|| {
                        format!("skeleton {skeleton} of family {family} has no seed row")
                    })
                })
                .collect()
        })
        .collect()
}

pub use schedules::{build_retained, rebuild};
