// SPDX-License-Identifier: 0BSD

//! Arena capture for out-of-tree kernel development (cargo feature `pacman`).
//!
//! With `PACMAN_SAVE_ARENA=<directory>` set when an artifact is loaded, every
//! compiled Direct-Arena engine records its arena just before the first leaf
//! of a tile runs (`NAME.pre`) and after the last one (`NAME.post`), tile after
//! tile, until a recording holds `PACMAN_SAVE_ARENA_POINTS` points (default
//! 256); `PACMAN_SAVE_ARENA_RECORDINGS` (default 1) recordings are written per
//! engine. `NAME.json` describes the planes and the leaves that ran.
//!
//! Both arenas are little-endian f64, plane-major: plane `p` of point `i` is
//! value `p * num_points + i`. The planes are the physical current planes
//! (real, then imaginary), the amplitude planes (real, then imaginary), the
//! momentum planes, one zero plane, and two planes per model parameter (real,
//! imaginary), the order of `pacman`'s `bindings::Arena`. Running the recorded
//! leaves on `.pre`, tile by tile as in `tiles`, reproduces `.post`.

use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};

const SAVE_ARENA: &str = "PACMAN_SAVE_ARENA";
const SAVE_ARENA_POINTS: &str = "PACMAN_SAVE_ARENA_POINTS";
const SAVE_ARENA_RECORDINGS: &str = "PACMAN_SAVE_ARENA_RECORDINGS";
const FORMAT: &str = "pacman-arena-v1";

/// Engines loaded by this process, in load order (the `e<N>` of file names).
static ENGINES: AtomicUsize = AtomicUsize::new(0);

/// How the engine ran the leaves of a tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CaptureMethod {
    All,
    Validated,
    MaterializedHelicity,
    /// Every leaf, but a leaf whose inputs equal an earlier call of the same
    /// memo restores its saved outputs instead of running (same bits).
    Correlated,
}

impl CaptureMethod {
    const fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Validated => "validated",
            Self::MaterializedHelicity => "materialized_helicity",
            Self::Correlated => "correlated",
        }
    }
}

/// One stage of the engine: its label and its leaves' application paths.
pub(crate) struct CaptureStage {
    pub(crate) label: String,
    pub(crate) leaves: Vec<String>,
}

/// The fixed shape of an engine's arena.
pub(crate) struct CaptureLayout {
    pub(crate) current_planes: usize,
    pub(crate) amplitude_planes: usize,
    pub(crate) momentum_planes: usize,
    pub(crate) model_parameters: usize,
    pub(crate) canonical_to_physical: Vec<u32>,
    pub(crate) source_planes: Vec<u32>,
    /// The current stages in order, then the amplitude stage.
    pub(crate) stages: Vec<CaptureStage>,
}

impl CaptureLayout {
    /// The regions of the arena in order: (name, planes).
    pub(crate) fn regions(&self) -> [(&'static str, usize); 7] {
        [
            ("current_re", self.current_planes),
            ("current_im", self.current_planes),
            ("amplitude_re", self.amplitude_planes),
            ("amplitude_im", self.amplitude_planes),
            ("momenta", self.momentum_planes),
            ("zero", 1),
            ("parameters", 2 * self.model_parameters),
        ]
    }

    fn plane_count(&self) -> usize {
        self.regions().iter().map(|(_, n)| n).sum()
    }
}

/// The engine's buffers during one tile, plane `p` at `[p * stride..][..points]`.
pub(crate) struct ArenaSource<'a> {
    pub(crate) current_re: &'a [f64],
    pub(crate) current_im: &'a [f64],
    pub(crate) amplitude_re: &'a [f64],
    pub(crate) amplitude_im: &'a [f64],
    pub(crate) momenta: &'a [f64],
    pub(crate) zero: &'a [f64],
    pub(crate) parameter_re: &'a [f64],
    pub(crate) parameter_im: &'a [f64],
    pub(crate) stride: usize,
    pub(crate) points: usize,
}

/// The leaves a tile ran: per stage (the amplitude stage last) their indices.
pub(crate) type CaptureSchedule = Vec<Vec<usize>>;

struct Recording {
    method: CaptureMethod,
    schedule: CaptureSchedule,
    tiles: Vec<usize>,
    pre: Vec<Vec<f64>>,
    post: Vec<Vec<f64>>,
    points: usize,
}

struct Pending {
    method: CaptureMethod,
    schedule: CaptureSchedule,
    points: usize,
    arena: Vec<f64>,
}

pub(crate) struct ArenaRecorder {
    directory: PathBuf,
    point_limit: usize,
    recording_limit: usize,
    engine: usize,
    process: String,
    layout: CaptureLayout,
    written: usize,
    current: Option<Recording>,
    pending: Option<Pending>,
}

fn env_count(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(value) if !value.trim().is_empty() => match value.trim().parse::<usize>() {
            Ok(count) if count > 0 => count,
            _ => {
                eprintln!("pacman: ignoring {name}={value:?} (not a positive integer)");
                default
            }
        },
        _ => default,
    }
}

impl ArenaRecorder {
    /// A recorder if `PACMAN_SAVE_ARENA` names a directory, else `None`.
    pub(crate) fn from_env(layout: impl FnOnce() -> CaptureLayout) -> Option<Box<Self>> {
        let directory = std::env::var_os(SAVE_ARENA).filter(|value| !value.is_empty())?;
        Some(Box::new(Self::new(
            PathBuf::from(directory),
            env_count(SAVE_ARENA_POINTS, 256),
            env_count(SAVE_ARENA_RECORDINGS, 1),
            layout(),
        )))
    }

    fn new(
        directory: PathBuf,
        point_limit: usize,
        recording_limit: usize,
        layout: CaptureLayout,
    ) -> Self {
        Self {
            directory,
            point_limit,
            recording_limit,
            engine: ENGINES.fetch_add(1, Ordering::Relaxed),
            process: String::from("process"),
            layout,
            written: 0,
            current: None,
            pending: None,
        }
    }

    pub(crate) fn set_process(&mut self, process: &str) {
        self.process = process
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || "-_".contains(c) {
                    c
                } else {
                    '_'
                }
            })
            .collect();
    }

    /// Whether the next tile is recorded at all (checked before any copy).
    pub(crate) fn wants(&self) -> bool {
        self.written < self.recording_limit
    }

    /// The schedule that runs every leaf.
    pub(crate) fn full_schedule(&self) -> CaptureSchedule {
        self.layout
            .stages
            .iter()
            .map(|stage| (0..stage.leaves.len()).collect())
            .collect()
    }

    pub(crate) fn pre(
        &mut self,
        method: CaptureMethod,
        schedule: CaptureSchedule,
        source: &ArenaSource<'_>,
    ) {
        if !self.wants() {
            return;
        }
        if self
            .current
            .as_ref()
            .is_some_and(|rec| rec.method != method || rec.schedule != schedule)
        {
            self.flush();
            if !self.wants() {
                return;
            }
        }
        self.pending = Some(Pending {
            method,
            schedule,
            points: source.points,
            arena: self.snapshot(source),
        });
    }

    pub(crate) fn post(&mut self, source: &ArenaSource<'_>) {
        let Some(pending) = self.pending.take() else {
            return;
        };
        if pending.points != source.points {
            eprintln!("pacman: tile point count changed during evaluation; tile dropped");
            return;
        }
        let post = self.snapshot(source);
        let recording = self.current.get_or_insert_with(|| Recording {
            method: pending.method,
            schedule: pending.schedule,
            tiles: Vec::new(),
            pre: Vec::new(),
            post: Vec::new(),
            points: 0,
        });
        recording.tiles.push(pending.points);
        recording.pre.push(pending.arena);
        recording.post.push(post);
        recording.points += pending.points;
        if recording.points >= self.point_limit {
            self.flush();
        }
    }

    // One tile, plane-major with leading dimension `points`.
    fn snapshot(&self, source: &ArenaSource<'_>) -> Vec<f64> {
        let n = source.points;
        let mut out = Vec::with_capacity(self.layout.plane_count() * n);
        let mut planes = |buffer: &[f64], count: usize| {
            for plane in 0..count {
                out.extend_from_slice(&buffer[plane * source.stride..][..n]);
            }
        };
        planes(source.current_re, self.layout.current_planes);
        planes(source.current_im, self.layout.current_planes);
        planes(source.amplitude_re, self.layout.amplitude_planes);
        planes(source.amplitude_im, self.layout.amplitude_planes);
        planes(source.momenta, self.layout.momentum_planes);
        out.extend_from_slice(&source.zero[..n]);
        for k in 0..self.layout.model_parameters {
            out.extend(std::iter::repeat_n(source.parameter_re[k], n));
            out.extend(std::iter::repeat_n(source.parameter_im[k], n));
        }
        out
    }

    /// Writes the current recording, if any.
    pub(crate) fn flush(&mut self) {
        let Some(recording) = self.current.take() else {
            return;
        };
        let name = format!("{}.e{}.r{}", self.process, self.engine, self.written);
        self.written += 1;
        if let Err(error) = self.write(&name, &recording) {
            eprintln!(
                "pacman: cannot save arena {name} in {}: {error}",
                self.directory.display()
            );
            self.written = self.recording_limit;
        }
    }

    fn write(&self, name: &str, recording: &Recording) -> std::io::Result<()> {
        fs::create_dir_all(&self.directory)?;
        let planes = self.layout.plane_count();
        for (suffix, tiles) in [("pre", &recording.pre), ("post", &recording.post)] {
            write_atomic(&self.directory.join(format!("{name}.{suffix}")), |out| {
                for plane in 0..planes {
                    for (tile, &n) in tiles.iter().zip(&recording.tiles) {
                        for value in &tile[plane * n..][..n] {
                            out.write_all(&value.to_le_bytes())?;
                        }
                    }
                }
                Ok(())
            })?;
        }
        let meta = self.metadata(name, recording);
        write_atomic(&self.directory.join(format!("{name}.json")), |out| {
            serde_json::to_writer_pretty(&mut *out, &meta)?;
            out.write_all(b"\n")
        })
    }

    fn metadata(&self, name: &str, recording: &Recording) -> Value {
        let layout = &self.layout;
        let mut first = 0;
        let regions: Vec<Value> = layout
            .regions()
            .iter()
            .map(|&(region, count)| {
                let entry = json!({"name": region, "first": first, "count": count});
                first += count;
                entry
            })
            .collect();
        let stage_count = layout.stages.len();
        let schedule: Vec<Value> = recording
            .schedule
            .iter()
            .zip(&layout.stages)
            .enumerate()
            .map(|(index, (leaves, stage))| {
                json!({
                    "stage": index,
                    "label": stage.label,
                    "amplitude": index + 1 == stage_count,
                    "leaves": leaves.iter().map(|&leaf| json!({
                        "leaf": leaf,
                        "application_path": stage.leaves[leaf],
                    })).collect::<Vec<_>>(),
                })
            })
            .collect();
        json!({
            "format": FORMAT,
            "name": name,
            "dtype": "f64-le",
            "layout": "plane-major",
            "num_planes": layout.plane_count(),
            "num_points": recording.points,
            "regions": regions,
            "process": self.process,
            "engine": self.engine,
            "recording": self.written - 1,
            "rusticol_version": env!("CARGO_PKG_VERSION"),
            "method": recording.method.name(),
            "tiles": recording.tiles,
            "current_planes": layout.current_planes,
            "amplitude_planes": layout.amplitude_planes,
            "momentum_planes": layout.momentum_planes,
            "model_parameters": layout.model_parameters,
            "canonical_to_physical": layout.canonical_to_physical,
            "unmapped_physical": u32::MAX,
            "source_planes": layout.source_planes,
            "schedule": schedule,
        })
    }
}

impl Drop for ArenaRecorder {
    fn drop(&mut self) {
        self.flush();
    }
}

fn write_atomic(
    path: &Path,
    body: impl FnOnce(&mut BufWriter<fs::File>) -> std::io::Result<()>,
) -> std::io::Result<()> {
    let temporary = PathBuf::from(format!("{}.tmp{}", path.display(), std::process::id()));
    let mut out = BufWriter::new(fs::File::create(&temporary)?);
    body(&mut out)?;
    out.flush()?;
    drop(out);
    fs::rename(&temporary, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    // value of plane `p`, point `i` (global), phase 0 = pre, 1 = post
    fn value(plane: usize, point: usize, phase: usize) -> f64 {
        (1000 * plane + 10 * point + phase) as f64
    }

    fn layout() -> CaptureLayout {
        CaptureLayout {
            current_planes: 2,
            amplitude_planes: 1,
            momentum_planes: 4,
            model_parameters: 1,
            canonical_to_physical: vec![0, u32::MAX, 1],
            source_planes: vec![0],
            stages: vec![
                CaptureStage {
                    label: "stage1".into(),
                    leaves: vec!["evaluators/a.plane.symjit".into()],
                },
                CaptureStage {
                    label: "amplitude".into(),
                    leaves: vec!["evaluators/b.plane.symjit".into()],
                },
            ],
        }
    }

    // the engine's buffers for one tile: planes in arena order, padded to `stride`
    fn tile(first_point: usize, points: usize, phase: usize) -> Vec<Vec<f64>> {
        let stride = 8;
        let buffer = |first_plane: usize, count: usize| {
            let mut out = vec![-1.0; count * stride];
            for p in 0..count {
                for i in 0..points {
                    out[p * stride + i] = value(first_plane + p, first_point + i, phase);
                }
            }
            out
        };
        vec![
            buffer(0, 2),
            buffer(2, 2),
            buffer(4, 1),
            buffer(5, 1),
            buffer(6, 4),
            vec![0.0; stride],
            vec![2.5],
            vec![0.0],
        ]
    }

    fn source(buffers: &[Vec<f64>], points: usize) -> ArenaSource<'_> {
        ArenaSource {
            current_re: &buffers[0],
            current_im: &buffers[1],
            amplitude_re: &buffers[2],
            amplitude_im: &buffers[3],
            momenta: &buffers[4],
            zero: &buffers[5],
            parameter_re: &buffers[6],
            parameter_im: &buffers[7],
            stride: 8,
            points,
        }
    }

    fn read(path: &Path) -> Vec<f64> {
        fs::read(path)
            .unwrap()
            .chunks_exact(8)
            .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn tiles_are_joined_plane_major() {
        let directory =
            std::env::temp_dir().join(format!("pacman-capture-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&directory);
        let mut recorder = ArenaRecorder::new(directory.clone(), 5, 1, layout());
        recorder.set_process("u u~ > z");
        let full = recorder.full_schedule();
        for (first, points) in [(0, 3), (3, 2)] {
            recorder.pre(
                CaptureMethod::All,
                full.clone(),
                &source(&tile(first, points, 0), points),
            );
            recorder.post(&source(&tile(first, points, 1), points));
        }
        // the recording is complete: later tiles are not recorded
        assert!(!recorder.wants());
        recorder.pre(CaptureMethod::All, full, &source(&tile(5, 1, 0), 1));
        drop(recorder);

        let mut names: Vec<_> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        let engine = names[0].split('.').nth(1).unwrap().to_string();
        let name = format!("u_u____z.{engine}.r0");
        assert_eq!(
            names,
            [
                format!("{name}.json"),
                format!("{name}.post"),
                format!("{name}.pre")
            ]
        );
        let planes = 13;
        for (suffix, phase) in [("pre", 0), ("post", 1)] {
            let data = read(&directory.join(format!("{name}.{suffix}")));
            assert_eq!(data.len(), planes * 5);
            for point in 0..5 {
                for plane in 0..10 {
                    assert_eq!(data[plane * 5 + point], value(plane, point, phase));
                }
                assert_eq!(data[10 * 5 + point], 0.0);
                assert_eq!(data[11 * 5 + point], 2.5);
                assert_eq!(data[12 * 5 + point], 0.0);
            }
        }
        let meta: Value =
            serde_json::from_slice(&fs::read(directory.join(format!("{name}.json"))).unwrap())
                .unwrap();
        assert_eq!(meta["format"], FORMAT);
        assert_eq!(meta["num_planes"], 13);
        assert_eq!(meta["num_points"], 5);
        assert_eq!(meta["tiles"], json!([3, 2]));
        assert_eq!(meta["method"], "all");
        assert_eq!(
            meta["regions"][6],
            json!({"name": "parameters", "first": 11, "count": 2})
        );
        assert_eq!(
            meta["schedule"][1]["leaves"][0]["application_path"],
            "evaluators/b.plane.symjit"
        );
        assert_eq!(meta["schedule"][1]["amplitude"], true);
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn a_new_schedule_starts_a_new_recording() {
        let directory = std::env::temp_dir().join(format!(
            "pacman-capture-schedule-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&directory);
        let mut recorder = ArenaRecorder::new(directory.clone(), 100, 3, layout());
        let full = recorder.full_schedule();
        let partial = vec![vec![], vec![0]];
        for schedule in [&full, &full, &partial] {
            recorder.pre(
                CaptureMethod::Validated,
                schedule.clone(),
                &source(&tile(0, 2, 0), 2),
            );
            recorder.post(&source(&tile(0, 2, 1), 2));
        }
        drop(recorder);
        let mut metas: Vec<Value> = fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|e| e == "json"))
            .map(|path| serde_json::from_slice(&fs::read(path).unwrap()).unwrap())
            .collect();
        metas.sort_by_key(|meta| meta["recording"].as_u64());
        assert_eq!(metas.len(), 2);
        assert_eq!(metas[0]["tiles"], json!([2, 2]));
        assert_eq!(metas[1]["tiles"], json!([2]));
        assert_eq!(metas[1]["schedule"][0]["leaves"], json!([]));
        fs::remove_dir_all(&directory).unwrap();
    }
}
