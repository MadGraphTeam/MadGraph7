// SPDX-License-Identifier: 0BSD

//! Running a compiled Direct-Arena engine's leaves through a pacman runner
//! (cargo feature `pacman`).
//!
//! A runner is a shared library that `pacman runner` builds for one compiled
//! execution: the execution's kernels with their plane tables compiled in,
//! behind the interface `pacman-runner-v1` (`pacman_runner.h`). With
//! `PACMAN_RUNNER=<lib.so>[:<lib.so>...]` set when an artifact is loaded, every
//! engine opens the libraries and takes the one whose metadata matches it (the
//! same leaves in execution order, the same current-plane map, the same
//! regions); its evaluate methods then call the runner instead of their leaves.
//! An engine that no library matches runs its own leaves.
//!
//! The runner sees the engine's planes through one pointer per plane, in the
//! arena order of `pacman_capture` (the recordings): current planes (real,
//! then imaginary), amplitude planes (real, then imaginary), momenta, the zero
//! plane, and two planes per model parameter (real, imaginary), which this
//! module owns and fills with the parameter values.
//!
//! A runner that exports `pacman_runner_v1_run_reduce` also applies the
//! amplitude reduction of the selected sum (coherent groups, or a group per
//! amplitude), so that only one value per point comes back from a GPU; the
//! engine's caller hands it the groups (`PacmanReduction`).
//! `PACMAN_RUNNER_REDUCE=0` keeps the reduction in Rusticol.
//!
//! A GPU runner needs many points per call: set `RUSTICOL_TILE_POINTS=N`
//! (`tile_override`, which feature `pacman` enables). With `PACMAN_RUNNER`
//! set, every engine gets tiles of N points, not only those whose cache-sized
//! tile holds a single point.

use std::ffi::{CStr, c_char};
use std::path::PathBuf;

use super::pacman_capture::CaptureLayout;
use crate::direct_arena::{AlignedF64Buffer, DirectArenaView};
use crate::{RusticolError, RusticolResult};

const RUNNER: &str = "PACMAN_RUNNER";
const ABI_VERSION: u32 = 1;
const METADATA_SYMBOL: &[u8] = b"pacman_runner_v1_metadata\0";
const RUN_SYMBOL: &[u8] = b"pacman_runner_v1_run\0";
const RUN_REDUCE_SYMBOL: &[u8] = b"pacman_runner_v1_run_reduce\0";
const REDUCE: &str = "PACMAN_RUNNER_REDUCE";
const STATUS_OK: i32 = 0;

/// `pacman_runner_metadata` of `pacman_runner.h`.
#[repr(C)]
struct RunnerMetadataV1 {
    abi_version: u32,
    struct_size: u32,
    num_planes: u32,
    num_leaves: u32,
    regions: [[u32; 2]; 7],
    process: *const c_char,
    execution: *const c_char,
    leaves: *const *const c_char,
    leaf_planes: *const *const u32,
    leaf_inputs: *const u32,
    leaf_outputs: *const u32,
    canonical_to_physical: *const u32,
    num_components: u32,
    reserved: u32,
}

/// `pacman_runner_reduction` of `pacman_runner.h`.
#[repr(C)]
struct RunnerReduction {
    struct_size: u32,
    num_groups: u32,
    offsets: *const u32,
    indices: *const u32,
    weights: *const f64,
}

type MetadataFunction = unsafe extern "C" fn() -> *const RunnerMetadataV1;
type RunFunction = unsafe extern "C" fn(*const *mut f64, u32, *const u32, u32, u32, u32) -> i32;
type RunReduceFunction = unsafe extern "C" fn(
    *const *mut f64,
    u32,
    *const u32,
    u32,
    u32,
    u32,
    *const RunnerReduction,
    *mut f64,
) -> i32;

/// The amplitude reduction a runner applies after the leaves: per group, in
/// order, its weight and amplitude indices; the value of a point is the sum of
/// `weight * |sum of its amplitudes|²` (`pacman_runner.h`), as the selected
/// sum's reduction computes it, without the normalization.
#[derive(Debug, Default)]
pub(crate) struct PacmanReduction {
    offsets: Vec<u32>,
    indices: Vec<u32>,
    weights: Vec<f64>,
}

impl PacmanReduction {
    pub(crate) fn clear(&mut self) {
        self.offsets.clear();
        self.offsets.push(0);
        self.indices.clear();
        self.weights.clear();
    }

    pub(crate) fn push_group(&mut self, weight: f64, indices: impl IntoIterator<Item = usize>) {
        self.indices
            .extend(indices.into_iter().map(|index| index as u32));
        self.offsets.push(self.indices.len() as u32);
        self.weights.push(weight);
    }
}

/// The leaves a schedule runs: per current stage the active leaves, then the
/// amplitude stage's.
pub(crate) type ActiveLeaves<'a> = (&'a [Box<[usize]>], &'a [usize]);

/// A runner bound to one engine's planes.
pub(crate) struct PacmanRunner {
    // the plane pointers address the engine's boxed buffers and `broadcast`
    planes: Box<[*mut f64]>,
    broadcast: AlignedF64Buffer,
    /// per parameter plane, the bits of the value it holds (`None`: not filled)
    broadcast_bits: Box<[Option<u64>]>,
    stride: usize,
    /// the runner index of each stage's first leaf (the amplitude stage last)
    stage_offsets: Box<[u32]>,
    leaves: Vec<u32>,
    run: RunFunction,
    /// `pacman_runner_v1_run_reduce` (`None`: absent, or `PACMAN_RUNNER_REDUCE=0`)
    run_reduce: Option<RunReduceFunction>,
    reduction: PacmanReduction,
    /// the next run reduces (`arm_reduction`)
    armed: bool,
    /// the values of the last run's reduction, if it reduced
    values: Vec<f64>,
    values_ready: bool,
    path: PathBuf,
    execution: String,
    // last: the code `run` points into
    _library: libloading::Library,
}

impl std::fmt::Debug for PacmanRunner {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PacmanRunner")
            .field("path", &self.path)
            .field("execution", &self.execution)
            .finish_non_exhaustive()
    }
}

/// Why a library does not fit an engine, or `None` if it does.
fn mismatch(
    metadata: &RunnerMetadataV1,
    layout: &CaptureLayout,
    leaves: &[&str],
) -> Option<String> {
    if metadata.abi_version != ABI_VERSION
        || metadata.struct_size as usize != std::mem::size_of::<RunnerMetadataV1>()
        || metadata.reserved != 0
    {
        return Some(format!(
            "interface version {} (size {}), expected {ABI_VERSION} (size {})",
            metadata.abi_version,
            metadata.struct_size,
            std::mem::size_of::<RunnerMetadataV1>()
        ));
    }
    let regions = layout.regions();
    let mut first = 0usize;
    for (index, (name, count)) in regions.iter().enumerate() {
        if metadata.regions[index] != [first as u32, *count as u32] {
            return Some(format!(
                "region {name} is planes {}..+{}, the engine's {first}..+{count}",
                metadata.regions[index][0], metadata.regions[index][1]
            ));
        }
        first += count;
    }
    if metadata.num_planes as usize != first {
        return Some(format!(
            "{} planes, the engine has {first}",
            metadata.num_planes
        ));
    }
    if metadata.num_leaves as usize != leaves.len() {
        return Some(format!(
            "{} leaves, the engine has {}",
            metadata.num_leaves,
            leaves.len()
        ));
    }
    for (index, leaf) in leaves.iter().enumerate() {
        // SAFETY: the metadata holds `num_leaves` NUL-terminated strings
        let name = unsafe { CStr::from_ptr(*metadata.leaves.add(index)) };
        if name.to_bytes() != leaf.as_bytes() {
            return Some(format!(
                "leaf {index} is {}, the engine's {leaf}",
                name.to_string_lossy()
            ));
        }
    }
    if metadata.num_components as usize != layout.canonical_to_physical.len() {
        return Some("another current-plane map".to_string());
    }
    // SAFETY: the metadata holds `num_components` map entries
    let map = unsafe {
        std::slice::from_raw_parts(
            metadata.canonical_to_physical,
            metadata.num_components as usize,
        )
    };
    if map != layout.canonical_to_physical.as_slice() {
        return Some("another current-plane map".to_string());
    }
    None
}

/// Whether `PACMAN_RUNNER` names any library.
pub(crate) fn requested() -> bool {
    libraries().is_some()
}

/// The libraries named by `PACMAN_RUNNER` (`None` if unset or empty).
fn libraries() -> Option<Vec<PathBuf>> {
    let value = std::env::var_os(RUNNER).filter(|value| !value.is_empty())?;
    Some(
        std::env::split_paths(&value)
            .filter(|path| !path.as_os_str().is_empty())
            .collect(),
    )
}

impl PacmanRunner {
    /// The runner of `PACMAN_RUNNER` that matches the engine of `layout`,
    /// bound to its planes. `None` if the variable is unset, or if no library
    /// matches (reported on standard error); an error if a library cannot be
    /// opened or lacks the interface.
    ///
    /// # Safety
    ///
    /// `arena`, `momenta` (`layout.momentum_planes` planes) and `zero` (one
    /// plane) must address allocations of `stride`-point planes that outlive
    /// the runner and never move.
    pub(crate) unsafe fn from_env(
        layout: &CaptureLayout,
        arena: DirectArenaView,
        momenta: *mut f64,
        zero: *mut f64,
    ) -> RusticolResult<Option<Box<Self>>> {
        let Some(paths) = libraries() else {
            return Ok(None);
        };
        let leaves: Vec<&str> = layout
            .stages
            .iter()
            .flat_map(|stage| stage.leaves.iter().map(String::as_str))
            .collect();
        let mut reasons = Vec::new();
        for path in paths {
            // SAFETY: a pacman runner's initializers do nothing; loading it is
            // the documented purpose of PACMAN_RUNNER
            let library = unsafe { libloading::Library::new(&path) }.map_err(|error| {
                RusticolError::evaluation(format!(
                    "{RUNNER}: cannot open {}: {error}",
                    path.display()
                ))
            })?;
            // SAFETY: the symbols have the signatures of pacman_runner.h
            let (metadata, run) = unsafe {
                let metadata = library
                    .get::<MetadataFunction>(METADATA_SYMBOL)
                    .map(|symbol| *symbol);
                let run = library.get::<RunFunction>(RUN_SYMBOL).map(|symbol| *symbol);
                match (metadata, run) {
                    (Ok(metadata), Ok(run)) => (metadata(), run),
                    _ => {
                        return Err(RusticolError::compatibility(format!(
                            "{RUNNER}: {} is not a pacman runner (no pacman_runner_v1_*)",
                            path.display()
                        )));
                    }
                }
            };
            if metadata.is_null() {
                return Err(RusticolError::compatibility(format!(
                    "{RUNNER}: {} returns no metadata",
                    path.display()
                )));
            }
            // SAFETY: the metadata is static data of the loaded library
            let metadata = unsafe { &*metadata };
            if let Some(reason) = mismatch(metadata, layout, &leaves) {
                reasons.push(format!("{}: {reason}", path.display()));
                continue;
            }
            // SAFETY: as above
            let execution = unsafe { CStr::from_ptr(metadata.execution) }
                .to_string_lossy()
                .into_owned();
            let reduce = std::env::var(REDUCE).map_or(true, |value| value != "0");
            // SAFETY: the symbol has the signature of pacman_runner.h
            let run_reduce = reduce
                .then(|| unsafe {
                    library
                        .get::<RunReduceFunction>(RUN_REDUCE_SYMBOL)
                        .map(|symbol| *symbol)
                        .ok()
                })
                .flatten();
            // SAFETY: the caller's contract
            return unsafe {
                Self::bind(
                    layout, arena, momenta, zero, library, run, run_reduce, path, execution,
                )
            }
            .map(|runner| Some(Box::new(runner)));
        }
        eprintln!(
            "pacman: {RUNNER}: no runner matches the engine of {} leaves (first {}); it runs its own \
             leaves ({})",
            leaves.len(),
            leaves.first().copied().unwrap_or("none"),
            reasons.join("; ")
        );
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn bind(
        layout: &CaptureLayout,
        arena: DirectArenaView,
        momenta: *mut f64,
        zero: *mut f64,
        library: libloading::Library,
        run: RunFunction,
        run_reduce: Option<RunReduceFunction>,
        path: PathBuf,
        execution: String,
    ) -> RusticolResult<Self> {
        let stride = arena.point_stride as usize;
        let mut broadcast = AlignedF64Buffer::zeroed(
            2 * layout.model_parameters * stride,
            "pacman runner parameter planes",
        )?;
        let broadcast_base = broadcast.as_mut_ptr();
        let mut planes = Vec::new();
        // SAFETY (all `add`s): the offsets lie within the buffers, as the
        // caller guarantees for the arena, momenta and zero plane
        unsafe {
            for base in [arena.current_re, arena.current_im] {
                planes.extend((0..layout.current_planes).map(|plane| base.add(plane * stride)));
            }
            for base in [arena.amplitude_re, arena.amplitude_im] {
                planes.extend((0..layout.amplitude_planes).map(|plane| base.add(plane * stride)));
            }
            planes.extend((0..layout.momentum_planes).map(|plane| momenta.add(plane * stride)));
            planes.push(zero);
            planes.extend(
                (0..2 * layout.model_parameters).map(|plane| broadcast_base.add(plane * stride)),
            );
        }
        let mut stage_offsets = Vec::with_capacity(layout.stages.len());
        let mut next = 0u32;
        for stage in &layout.stages {
            stage_offsets.push(next);
            next += stage.leaves.len() as u32;
        }
        eprintln!(
            "pacman: {RUNNER}: the engine of {next} leaves runs {} ({execution}{})",
            path.display(),
            if run_reduce.is_some() {
                ", reductions included"
            } else {
                ""
            }
        );
        Ok(Self {
            planes: planes.into_boxed_slice(),
            broadcast,
            broadcast_bits: vec![None; 2 * layout.model_parameters].into_boxed_slice(),
            stride,
            stage_offsets: stage_offsets.into_boxed_slice(),
            leaves: Vec::with_capacity(next as usize),
            run,
            run_reduce,
            reduction: PacmanReduction::default(),
            armed: false,
            values: Vec::new(),
            values_ready: false,
            path,
            execution,
            _library: library,
        })
    }

    /// Runs the active leaves (`stage_leaves[k]` of stage `k`, then
    /// `amplitude_leaves`; `None`: every leaf in order) on points
    /// `0..point_count` with the model parameters `parameter_re`/`parameter_im`.
    pub(crate) fn run(
        &mut self,
        schedule: Option<ActiveLeaves<'_>>,
        parameter_re: &[f64],
        parameter_im: &[f64],
        point_count: u32,
    ) -> RusticolResult<()> {
        self.fill_parameters(parameter_re, parameter_im);
        let (leaves, leaf_count) = match schedule {
            None => (std::ptr::null(), 0),
            Some((stage_leaves, amplitude_leaves)) => {
                self.leaves.clear();
                let stages = stage_leaves
                    .iter()
                    .map(|leaves| &leaves[..])
                    .chain(std::iter::once(amplitude_leaves));
                for (offset, leaves) in self.stage_offsets.iter().zip(stages) {
                    self.leaves
                        .extend(leaves.iter().map(|&leaf| offset + leaf as u32));
                }
                (self.leaves.as_ptr(), self.leaves.len() as u32)
            }
        };
        let reduce = std::mem::take(&mut self.armed);
        self.values_ready = false;
        // SAFETY: the plane pointers address live planes of `stride` points
        // (`from_env`), `point_count` does not exceed the stride (checked by
        // the engine), and the engine's evaluate methods hold `&mut self`; the
        // reduction's tables and `values` (point_count values) live through
        // the call
        let status = unsafe {
            match (reduce, self.run_reduce) {
                (true, Some(run_reduce)) => {
                    self.values.resize(point_count as usize, 0.0);
                    let reduction = RunnerReduction {
                        struct_size: std::mem::size_of::<RunnerReduction>() as u32,
                        num_groups: self.reduction.weights.len() as u32,
                        offsets: self.reduction.offsets.as_ptr(),
                        indices: self.reduction.indices.as_ptr(),
                        weights: self.reduction.weights.as_ptr(),
                    };
                    let status = run_reduce(
                        self.planes.as_ptr(),
                        self.planes.len() as u32,
                        leaves,
                        leaf_count,
                        0,
                        point_count,
                        &reduction,
                        self.values.as_mut_ptr(),
                    );
                    self.values_ready = status == STATUS_OK;
                    status
                }
                _ => (self.run)(
                    self.planes.as_ptr(),
                    self.planes.len() as u32,
                    leaves,
                    leaf_count,
                    0,
                    point_count,
                ),
            }
        };
        if status != STATUS_OK {
            return Err(RusticolError::evaluation(format!(
                "pacman runner {} returned status {status}",
                self.path.display()
            )));
        }
        Ok(())
    }

    /// The reduction to fill for `arm_reduction`, if the runner can reduce
    /// (`None`: Rusticol reduces the amplitude planes).
    pub(crate) fn reduction(&mut self) -> Option<&mut PacmanReduction> {
        self.run_reduce.is_some().then_some(&mut self.reduction)
    }

    /// The next run applies the reduction (as filled through `reduction`)
    /// and leaves the amplitude planes unspecified.
    pub(crate) fn arm_reduction(&mut self) {
        self.armed = self.run_reduce.is_some();
    }

    /// Copies the values of the last run's reduction into `out`; `false` if
    /// that run did not reduce (its amplitude planes hold the amplitudes).
    pub(crate) fn take_values(&mut self, out: &mut [f64]) -> bool {
        if !self.values_ready || out.len() != self.values.len() {
            return false;
        }
        out.copy_from_slice(&self.values);
        self.values_ready = false;
        true
    }

    /// Broadcast the parameters into their planes when their bits changed.
    fn fill_parameters(&mut self, parameter_re: &[f64], parameter_im: &[f64]) {
        let values = parameter_re
            .iter()
            .zip(parameter_im)
            .flat_map(|(re, im)| [*re, *im]);
        let planes = self.broadcast.as_mut_slice();
        for (slot, value) in values.enumerate() {
            if self.broadcast_bits[slot] != Some(value.to_bits()) {
                planes[slot * self.stride..(slot + 1) * self.stride].fill(value);
                self.broadcast_bits[slot] = Some(value.to_bits());
            }
        }
    }
}
