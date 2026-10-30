// SPDX-License-Identifier: 0BSD

//! Loop execution of plan v4 (`v2`, the shared-recursion engine) and the
//! decimal point inputs of the SDK. The plan v3 executor of the removed legacy
//! one-loop engine was deleted with it (MQ-LEGACY-REMOVE, 2026-10-29).

use std::collections::BTreeMap;

/// Original decimal inputs remain available for exact family invariants. The
/// legacy model has one global real input row; source-owned states are separate.
#[derive(Clone)]
pub struct LoopInputs {
    pub momenta: Vec<String>,
    pub scales_squared: Vec<String>,
    pub model_values: Vec<String>,
    pub point_ids: Vec<u64>,
    /// Literal complex vector sources, keyed by one-based physical leg. Each
    /// value contains four components per physical point; no rounding precedes
    /// numerical ingress and subset replay keeps these original strings.
    pub spin_sources: BTreeMap<usize, Vec<[String; 2]>>,
}

pub mod v2 {
    use crate::loop_v2::closure::FamilyNumerator;
    use crate::loop_v2::dress::{Carrier, ChunkInputs, LaneData, Reads, V2Batch};
    use crate::loop_v2::forward::{forward_factors, run_forward};
    pub use crate::loop_v2::fourier::SeedFourier;
    use crate::loop_v2::fourier::seeds_fourier;
    use crate::loop_v2::merged::{DEFAULT_PART_CHUNKS, MergedStats, parts, run_merged};
    use crate::loop_v2::plan::{LoopProgramV4, Merged};
    use crate::loop_v2::retest::{ReductionCheck, add_to_budget, deviation};
    use crate::loop_v2::seeds::seeds;
    use crate::loop_v2::trie::run_family;
    use crate::{RusticolError, RusticolResult};
    use rayon::prelude::*;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::Instant;
    use symbolica::domains::float::{Complex, Real};

    /// Default SIMD lanes per chunk (gate G0, DESIGN.md §5.4).
    pub const DEFAULT_LANES: usize = 16;
    /// Default families per streamed bucket chunk (bounds the live fused
    /// factors and family numerators).
    pub const DEFAULT_CHUNK_FAMILIES: usize = 512;

    /// The dressing schedule.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Schedule {
        /// Per-family suffix tries (production).
        B,
        /// The forward prefix tree (oracle).
        F,
        /// The merged planar state graph for the plan's certified families,
        /// schedule B for the others (all of them when the plan's graph is
        /// empty: no certified family). Plans exported without a graph
        /// refuse it (WP2.8).
        M,
    }

    /// The Laurent columns of every reduction lane (DESIGN.md §12.4).
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum Poles {
        /// The finite part only (finite-only mode).
        None,
        /// `[finite, 1/ε, 1/ε²]` with `ε_UV = ε_IR`.
        #[default]
        All,
        /// `[finite, 1/ε, 1/ε², 1/ε_UV]`: the combined Laurent integrals and
        /// the ultraviolet part of the single pole (the double pole is
        /// infrared), for the `uv` and `ir` pole selections.
        Split,
    }

    impl Poles {
        /// Values per lane.
        pub fn width(self) -> usize {
            match self {
                Self::None => 1,
                Self::All => 3,
                Self::Split => 4,
            }
        }
    }

    /// Options of one evaluation.
    #[derive(Clone, Copy, Debug)]
    pub struct V2Options {
        /// The Laurent columns of every lane (finite-only, combined, split).
        pub poles: Poles,
        /// Reduce every lane a second time in this frame (stability estimate).
        pub check: Option<ReductionCheck>,
        pub chunk_families: usize,
        /// Schedule B, or M where the plan has it (F is an oracle of
        /// numerators only).
        pub schedule: Schedule,
        /// Schedule-M chunks per streamed part.
        pub part_chunks: usize,
        /// The seed contraction (DESIGN.md §7.3).
        pub contraction: Contraction,
    }

    impl Default for V2Options {
        fn default() -> Self {
            Self {
                poles: Poles::All,
                check: None,
                chunk_families: DEFAULT_CHUNK_FAMILIES,
                schedule: Schedule::B,
                part_chunks: DEFAULT_PART_CHUNKS,
                contraction: Contraction::Auto,
            }
        }
    }

    /// The seed contraction of an evaluation (DESIGN.md §7.3): `Auto`
    /// contracts the certified orbit pairs of the executor's seed Fourier
    /// record ([`V2Executor::with_fourier`]) with the symmetric-group FFT
    /// and everything else directly (all of it without a record); `Direct`
    /// contracts every weight directly; `Fft` requires a record.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub enum Contraction {
        #[default]
        Auto,
        Direct,
        Fft,
    }

    /// Report count of the seed-FFT jobs an evaluation ran (one per job and
    /// lane chunk; absent when every seed was contracted directly).
    pub const FFT_SEED_JOBS: &str = "symmetric-group-fft seed jobs";
    /// Report stage of the seed contraction alone (direct or FFT), within
    /// the "seeds" stage, which also gathers each chunk's lane data.
    pub const SEED_CONTRACTION: &str = "seed contraction";

    /// A reduction lane of one point: (family, loop orders, observable column).
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct LaneKey {
        pub family: usize,
        pub order: usize,
        pub column: usize,
    }

    /// The reduction lanes of one family, lane-major: `shifts` has `4N`
    /// values per lane, `masses` `N`, `scales` one, `s_matrix` `N²` and
    /// `coefficients` `C(R+4, 4)` at the family rank (the NinjaRS adapter's
    /// `FamilyGroup`).
    pub struct FamilyLanes<T> {
        pub family: usize,
        pub propagators: usize,
        pub rank: usize,
        pub points: Vec<usize>,
        pub keys: Vec<LaneKey>,
        pub shifts: Vec<Complex<T>>,
        pub masses: Vec<Complex<T>>,
        pub scales: Vec<Complex<T>>,
        pub s_matrix: Vec<Complex<T>>,
        pub coefficients: Vec<Complex<T>>,
    }

    /// One reduction lane: its raw Laurent integrals in the columns of
    /// [`Poles`], or the reducer status of a failed lane.
    pub type LaneOutcome<T> = Result<Vec<Complex<T>>, String>;

    /// Direct outcomes, and the re-test outcomes when a check was asked; one
    /// vector per group in its lane order.
    pub type Reduced<T> = (Vec<Vec<LaneOutcome<T>>>, Option<Vec<Vec<LaneOutcome<T>>>>);

    /// Development counters a [`V2Backend`] collects inside the executor's
    /// `reduction` stage: seconds of its sub-stages (e.g. the master
    /// integrals) and counts. They are added to the run's [`Report`].
    #[derive(Clone, Debug, Default)]
    pub struct BackendStatistics {
        pub seconds: Vec<(&'static str, f64)>,
        pub counts: Vec<(&'static str, u64)>,
    }

    /// The reduction side of the v2 path: NinjaRS engines and caches.
    pub trait V2Backend<T> {
        /// Start a chunk of points (master caches hold one chunk's masters).
        fn begin(&mut self);
        fn reduce(
            &mut self,
            groups: &[FamilyLanes<T>],
            poles: Poles,
            check: Option<ReductionCheck>,
        ) -> RusticolResult<Reduced<T>>;
        /// The counters collected since the last call (none by default).
        fn take_statistics(&mut self) -> BackendStatistics {
            BackendStatistics::default()
        }
    }

    /// Modelled kernel work of the chosen forms of `steps` (complex
    /// multiply-adds per lane, as the `loop_v2_schedules` bench counts it).
    fn modelled_work<'a>(steps: impl Iterator<Item = &'a crate::loop_v2::plan::Step>) -> u64 {
        steps
            .map(|step| match (step.split.is_some(), step.costs) {
                (true, (_, Some(split))) => split.work,
                (_, (fused, _)) => fused.work,
            })
            .sum()
    }

    /// Report count of the modelled dressing work (complex MAC per lane).
    const DRESSING_WORK: &str = "dressing work (modelled complex MAC per lane)";

    /// One point's family numerators: (family, loop orders, rank, per
    /// column the coefficients in the family's trie frame).
    pub type PointNumerators<T> = Vec<(usize, usize, usize, Vec<Vec<Complex<T>>>)>;

    /// One point of an evaluation.
    #[derive(Clone)]
    pub struct PointResult<T> {
        /// Raw Laurent integrals of every reduced lane.
        pub families: Vec<(LaneKey, Vec<Complex<T>>)>,
        /// The re-test integrals, when a check was asked.
        pub check: Option<Vec<(LaneKey, Vec<Complex<T>>)>>,
        /// Per observable column, the bare virtual `2 Re(i/(16π² avg) ΣΩ)`
        /// per Laurent coefficient.
        pub bare: Vec<Vec<T>>,
        /// Per observable column, the re-test error budget.
        pub budget: Option<Vec<Vec<T>>>,
        /// Numerical failures; a failed point has no values.
        pub failures: Vec<String>,
    }

    /// Development measurements of one run: seconds per stage, work counts
    /// and the peak reals held per lane (chunk inputs: fused factors, split
    /// vertex tensors and currents; live open-loop objects of one family
    /// walk; numerators of one bucket chunk). Bytes per lane are the reals
    /// times `real_bytes` (heap limbs of arbitrary-precision reals excluded).
    #[derive(Clone, Debug, Default)]
    pub struct Report {
        pub seconds: BTreeMap<&'static str, f64>,
        pub counts: BTreeMap<&'static str, u64>,
        pub reals_per_lane: BTreeMap<&'static str, u64>,
        pub real_bytes: usize,
        pub lanes: usize,
        pub threads: usize,
    }

    impl Report {
        fn time(&mut self, stage: &'static str, start: Instant) {
            *self.seconds.entry(stage).or_default() += start.elapsed().as_secs_f64();
        }
        fn count(&mut self, name: &'static str, value: u64) {
            *self.counts.entry(name).or_default() += value;
        }
        fn peak(&mut self, name: &'static str, reals: usize, lanes: usize) {
            let value = (reals / lanes) as u64;
            let entry = self.reals_per_lane.entry(name).or_default();
            *entry = (*entry).max(value);
        }
    }

    /// A loaded plan with its lane count, thread pool and, optionally, the
    /// certified seed FFT of its seeds ([`SeedFourier`]).
    pub struct V2Executor {
        plan: Arc<LoopProgramV4>,
        lanes: usize,
        pool: rayon::ThreadPool,
        fourier: Option<Arc<SeedFourier>>,
    }

    macro_rules! with_lanes {
        ($lanes:expr, $f:ident :: <$t:ty>($($arg:expr),*)) => {
            match $lanes {
                1 => $f::<$t, 1>($($arg),*),
                2 => $f::<$t, 2>($($arg),*),
                4 => $f::<$t, 4>($($arg),*),
                8 => $f::<$t, 8>($($arg),*),
                16 => $f::<$t, 16>($($arg),*),
                32 => $f::<$t, 32>($($arg),*),
                other => Err(RusticolError::invalid_argument(format!(
                    "unsupported SIMD lane count {other} (1, 2, 4, 8, 16 or 32)"
                ))),
            }
        };
    }

    impl V2Executor {
        /// `threads = None` uses the machine's parallelism.
        pub fn new(
            plan: Arc<LoopProgramV4>,
            lanes: usize,
            threads: Option<usize>,
        ) -> RusticolResult<Self> {
            if ![1, 2, 4, 8, 16, 32].contains(&lanes) {
                return Err(RusticolError::invalid_argument(format!(
                    "unsupported SIMD lane count {lanes} (1, 2, 4, 8, 16 or 32)"
                )));
            }
            // The DoubleFloat and arbitrary-precision reduction builds
            // Symbolica expressions on these workers: pyAmpliCol's library
            // key unlocks them for their lifetime (`crate::licence`).
            let mut builder =
                rayon::ThreadPoolBuilder::new().start_handler(|_| crate::licence::unlock_thread());
            if let Some(threads) = threads {
                builder = builder.num_threads(threads.max(1));
            }
            let pool = builder
                .build()
                .map_err(|e| RusticolError::internal(e.to_string()))?;
            Ok(Self {
                plan,
                lanes,
                pool,
                fourier: None,
            })
        }

        /// Attach the plan's seed Fourier record (`plan_export.fourier_record`),
        /// certified on the plan's weights at load ([`SeedFourier::load`]).
        pub fn with_fourier(mut self, record: &[u8]) -> RusticolResult<Self> {
            self.fourier = Some(Arc::new(SeedFourier::load(record, &self.plan)?));
            Ok(self)
        }

        /// The certified seed FFT, if the executor has one.
        pub fn fourier(&self) -> Option<&SeedFourier> {
            self.fourier.as_deref()
        }

        pub fn plan(&self) -> &LoopProgramV4 {
            &self.plan
        }

        pub fn lanes(&self) -> usize {
            self.lanes
        }

        /// Run `op` on this executor's thread pool (its `threads` bound):
        /// parallel work inside it (the tree program's stages, its
        /// interferences) stays on these workers instead of the global pool.
        pub fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
            self.pool.install(op)
        }

        /// Every family numerator of every point (oracles O2, O4 and the
        /// comparison with the Python reference executor).
        pub fn numerators<T: Carrier>(
            &self,
            batch: &V2Batch<T>,
            zero: &T,
            schedule: Schedule,
        ) -> RusticolResult<Vec<PointNumerators<T>>> {
            batch.validate(&self.plan)?;
            self.pool.install(|| {
                with_lanes!(
                    self.lanes,
                    numerators_lanes::<T>(&self.plan, batch, zero, schedule)
                )
            })
        }

        /// The certified seed FFT an evaluation with `contraction` runs.
        fn contraction(&self, contraction: Contraction) -> RusticolResult<Option<&SeedFourier>> {
            Ok(match contraction {
                Contraction::Direct => None,
                Contraction::Auto => self.fourier.as_deref(),
                Contraction::Fft => Some(self.fourier.as_deref().ok_or_else(|| {
                    RusticolError::invalid_argument(
                        "the seed contraction 'fft' needs the plan's certified seed \
                         Fourier record (the plan has no certified orbit pair)",
                    )
                })?),
            })
        }

        /// The seeds `U0` of every point (the oracle of the seed
        /// contraction, DESIGN.md §7.3): per point, per seed key, its
        /// `rows × columns` values in object order, contracted as an
        /// evaluation with `contraction` contracts them.
        pub fn seed_values<T: Carrier + Real>(
            &self,
            batch: &V2Batch<T>,
            zero: &T,
            contraction: Contraction,
        ) -> RusticolResult<Vec<Vec<Vec<Complex<T>>>>> {
            batch.validate(&self.plan)?;
            let fourier = self.contraction(contraction)?;
            self.pool.install(|| {
                with_lanes!(
                    self.lanes,
                    seed_values_lanes::<T>(&self.plan, fourier, batch, zero)
                )
            })
        }

        /// Dress, reduce and assemble the bare virtual of every point.
        pub fn evaluate<T: Carrier + Real, B: V2Backend<T> + Send>(
            &self,
            batch: &V2Batch<T>,
            zero: &T,
            backend: &mut B,
            options: V2Options,
        ) -> RusticolResult<(Vec<PointResult<T>>, Report)> {
            batch.validate(&self.plan)?;
            let threads = self.pool.current_num_threads();
            let fourier = self.contraction(options.contraction)?;
            self.pool.install(|| {
                let (results, mut report) = with_lanes!(
                    self.lanes,
                    evaluate_lanes::<T>(&self.plan, fourier, batch, zero, backend, options)
                )?;
                report.threads = threads;
                Ok((results, report))
            })
        }
    }

    fn seed_values_lanes<T: Carrier + Real, const L: usize>(
        plan: &LoopProgramV4,
        fourier: Option<&SeedFourier>,
        batch: &V2Batch<T>,
        zero: &T,
    ) -> RusticolResult<Vec<Vec<Vec<Complex<T>>>>> {
        let v = crate::loop_v2::kernels::lane_vector_len(L);
        let mut out = vec![Vec::new(); batch.points()];
        for chunk in chunks(batch.points(), L) {
            let data = LaneData::<T, L>::new(plan, batch, &chunk, zero)?;
            let seeds = match fourier {
                Some(fourier) => seeds_fourier(plan, fourier, &data),
                None => seeds(plan, &data),
            };
            for (lane, &point) in chunk.iter().enumerate() {
                out[point] = seeds
                    .values
                    .iter()
                    .map(|key| {
                        key.chunks(v)
                            .map(|vector| {
                                Complex::new(vector[lane].clone(), vector[L + lane].clone())
                            })
                            .collect()
                    })
                    .collect();
            }
        }
        Ok(out)
    }

    fn chunks(points: usize, lanes: usize) -> Vec<Vec<usize>> {
        (0..points)
            .collect::<Vec<_>>()
            .chunks(lanes)
            .map(<[usize]>::to_vec)
            .collect()
    }

    /// A bucket chunk: its families, the inputs they read, and the inputs
    /// read by it or a later chunk (an input lives from its first to its
    /// last chunk).
    struct BucketChunk {
        families: Vec<usize>,
        reads: Reads,
        live: Reads,
    }

    /// The schedule-M graph an evaluation runs: the plan's under schedule M
    /// (possibly empty: no certified family), none otherwise.
    fn merged_of(plan: &LoopProgramV4, schedule: Schedule) -> RusticolResult<Option<&Merged>> {
        match schedule {
            Schedule::M => plan.merged.as_ref().map(Some).ok_or_else(|| {
                RusticolError::invalid_argument(
                    "the plan has no schedule M (it was exported without a merged graph)",
                )
            }),
            _ => Ok(None),
        }
    }

    /// The families schedule B dresses: all, or those schedule M leaves.
    fn b_families(plan: &LoopProgramV4, merged: Option<&Merged>) -> Vec<usize> {
        (0..plan.families.len())
            .filter(|f| merged.is_none_or(|m| m.families.binary_search(f).is_err()))
            .collect()
    }

    /// Bucket chunks of at most `size` of `families` (ascending) in bucket
    /// order.
    fn bucket_chunks(plan: &LoopProgramV4, size: usize, families: &[usize]) -> Vec<BucketChunk> {
        let parts: Vec<Vec<usize>> = plan
            .buckets()
            .values()
            .map(|members| {
                members
                    .iter()
                    .copied()
                    .filter(|f| families.binary_search(f).is_ok())
                    .collect::<Vec<_>>()
            })
            .filter(|members| !members.is_empty())
            .flat_map(|members| {
                members
                    .chunks(size.max(1))
                    .map(<[usize]>::to_vec)
                    .collect::<Vec<_>>()
            })
            .collect();
        let reads: Vec<Reads> = parts.iter().map(|part| Reads::of(plan, part)).collect();
        let last = |count: usize, ids: &dyn Fn(&Reads) -> &[usize]| {
            let mut last = vec![0; count];
            for (k, chunk) in reads.iter().enumerate() {
                for &id in ids(chunk) {
                    last[id] = k;
                }
            }
            last
        };
        let factors = last(plan.factors.len(), &|r| &r.factors);
        let vertices = last(plan.vertex_instances.len(), &|r| &r.vertices);
        let currents = last(plan.currents.len(), &|r| &r.currents);
        let alive = |last: &[usize], k: usize| -> Vec<usize> {
            (0..last.len()).filter(|&id| last[id] >= k).collect()
        };
        parts
            .into_iter()
            .zip(reads)
            .enumerate()
            .map(|(k, (families, reads))| BucketChunk {
                families,
                reads,
                live: Reads {
                    factors: alive(&factors, k),
                    vertices: alive(&vertices, k),
                    currents: alive(&currents, k),
                },
            })
            .collect()
    }

    /// Dress every family of `chunk` on one chunk of lanes (schedule B).
    fn dress_families<T: Carrier, const L: usize>(
        plan: &LoopProgramV4,
        chunk: &BucketChunk,
        inputs: &mut ChunkInputs<T>,
        seeds: &crate::loop_v2::seeds::Seeds<T>,
        data: &LaneData<T, L>,
        report: &mut Report,
    ) -> RusticolResult<Vec<FamilyNumerator<T>>> {
        let families = &chunk.families;
        let start = Instant::now();
        inputs.prepare(plan, data, &chunk.reads, &chunk.live);
        report.time("factors", start);
        report.peak(
            "chunk inputs (factors, vertex tensors, currents)",
            inputs.reals(),
            L,
        );
        report.count("factors", chunk.reads.factors.len() as u64);
        report.count("vertex tensors", chunk.reads.vertices.len() as u64);
        let start = Instant::now();
        let inputs = &*inputs;
        let runs = families
            .par_iter()
            .map(|&f| run_family::<T, L>(plan, f, inputs, seeds, data))
            .collect::<RusticolResult<Vec<_>>>()?;
        report.time("dressing", start);
        let peak = runs.iter().map(|run| run.peak_reals).max().unwrap_or(0);
        report.peak("open objects (one family walk)", peak, L);
        report.count("families", families.len() as u64);
        let steps = || families.iter().flat_map(|&f| &plan.families[f].steps);
        report.count("steps", steps().count() as u64);
        report.count(
            "split steps",
            steps().filter(|s| s.split.is_some()).count() as u64,
        );
        report.count(DRESSING_WORK, modelled_work(steps()));
        let numerators: Vec<_> = runs.into_iter().map(|run| run.numerator).collect();
        report.peak(
            "family numerators (one bucket chunk)",
            numerators.iter().map(FamilyNumerator::reals).sum(),
            L,
        );
        Ok(numerators)
    }

    fn numerators_lanes<T: Carrier, const L: usize>(
        plan: &LoopProgramV4,
        batch: &V2Batch<T>,
        zero: &T,
        schedule: Schedule,
    ) -> RusticolResult<Vec<PointNumerators<T>>> {
        let mut out: Vec<PointNumerators<T>> = vec![Vec::new(); batch.points()];
        let mut report = Report::default();
        let merged = merged_of(plan, schedule)?;
        for points in chunks(batch.points(), L) {
            let data = LaneData::<T, L>::new(plan, batch, &points, zero)?;
            let seeds = seeds(plan, &data);
            let mut inputs = ChunkInputs::new(plan);
            let numerators = match schedule {
                Schedule::B | Schedule::M => {
                    let families = b_families(plan, merged);
                    let reads = Reads::of(plan, &families);
                    let chunk = BucketChunk {
                        families,
                        live: reads.clone(),
                        reads,
                    };
                    let mut numerators =
                        dress_families(plan, &chunk, &mut inputs, &seeds, &data, &mut report)?;
                    if let Some(merged) = merged {
                        let parts = parts(merged, DEFAULT_PART_CHUNKS, plan);
                        run_merged(plan, &parts, &mut inputs, &seeds, &data, |done| {
                            numerators.extend(done);
                            Ok(())
                        })?;
                    }
                    numerators
                }
                Schedule::F => {
                    let needed = Reads {
                        factors: forward_factors(plan),
                        ..Reads::default()
                    };
                    inputs.prepare(plan, &data, &needed, &needed);
                    run_forward(plan, &inputs.factors, &seeds, &data)?.0
                }
            };
            for (lane, &point) in points.iter().enumerate() {
                for numerator in &numerators {
                    let family = &plan.families[numerator.family];
                    for (k, &order) in family.orders.iter().enumerate() {
                        let columns = (0..plan.columns)
                            .map(|o| numerator.lane::<L>(k, o, lane))
                            .collect();
                        out[point].push((numerator.family, order, family.rank, columns));
                    }
                }
            }
        }
        Ok(out)
    }

    /// The reduction lanes of one family numerator on a chunk of lanes,
    /// skipping points that failed already (a non-finite numerator fails its
    /// point here). Shifts `p(mask)` and invariants `s[mask]` (`p(mask)²`, or
    /// the squared mass of the leg whose mass shell fixes it) are the chunk's
    /// exact invariants cast once per mask ([`LaneData`]); an on-shell
    /// invariant thus equals the cast squared mass of its propagators bit for
    /// bit, as the masters' exact on-shell tests require.
    fn lanes_of<T: Carrier, const L: usize>(
        plan: &LoopProgramV4,
        masses: &[Complex<T>],
        scales: &[Complex<T>],
        numerator: &FamilyNumerator<T>,
        data: &LaneData<T, L>,
        failures: &mut [Vec<String>],
    ) -> FamilyLanes<T> {
        let family = &plan.families[numerator.family];
        let n = family.propagators();
        let masks = family.shift_masks();
        let v = 2 * L;
        let real = |x: &T| Complex::new(x.clone(), data.zero.clone());
        let shifts: Vec<_> = masks.iter().map(|&mask| data.momentum(mask)).collect();
        let squares: Vec<_> = masks
            .iter()
            .flat_map(|&a| masks.iter().map(move |&b| a ^ b))
            .map(|mask| data.square(mask))
            .collect();
        let mut lanes = FamilyLanes {
            family: numerator.family,
            propagators: n,
            rank: family.rank,
            points: Vec::new(),
            keys: Vec::new(),
            shifts: Vec::new(),
            masses: Vec::new(),
            scales: Vec::new(),
            s_matrix: Vec::new(),
            coefficients: Vec::new(),
        };
        for lane in 0..data.active {
            let point = data.points[lane];
            if !failures[point].is_empty() {
                continue;
            }
            if !numerator.finite::<L>(lane) {
                failures[point].push(format!("family {}: non-finite numerator", numerator.family));
                continue;
            }
            for (k, &order) in family.orders.iter().enumerate() {
                for column in 0..plan.columns {
                    let coefficients = numerator.lane::<L>(k, column, lane);
                    if coefficients
                        .iter()
                        .all(|z| z.re.is_fully_zero() && z.im.is_fully_zero())
                    {
                        continue;
                    }
                    lanes.points.push(point);
                    lanes.keys.push(LaneKey {
                        family: numerator.family,
                        order,
                        column,
                    });
                    for shift in &shifts {
                        lanes
                            .shifts
                            .extend((0..4).map(|mu| real(&shift[mu * v + lane])));
                    }
                    lanes.masses.extend(
                        family
                            .sequence
                            .iter()
                            .map(|&(_, mass)| masses[mass].clone()),
                    );
                    lanes.scales.push(scales[lane].clone());
                    lanes
                        .s_matrix
                        .extend(squares.iter().map(|square| real(&square[lane])));
                    lanes.coefficients.extend(coefficients);
                }
            }
        }
        lanes
    }

    /// What the reductions of one evaluation collect per point.
    struct Lanes<T> {
        failures: Vec<Vec<String>>,
        families: Vec<Vec<(LaneKey, Vec<Complex<T>>)>>,
        checks: Vec<Vec<(LaneKey, Vec<Complex<T>>)>>,
    }

    /// The reduction side of one chunk of lanes: scatters family numerators
    /// into NinjaRS lanes (skipping failed points), reduces them and records
    /// each lane's outcome.
    struct Reduction<'a, T, B, const L: usize> {
        plan: &'a LoopProgramV4,
        masses: &'a [Complex<T>],
        /// `mu_R²` per lane.
        scales: &'a [Complex<T>],
        data: &'a LaneData<T, L>,
        backend: &'a mut B,
        options: V2Options,
        sink: &'a mut Lanes<T>,
    }

    impl<T: Carrier, B: V2Backend<T>, const L: usize> Reduction<'_, T, B, L> {
        fn reduce(
            &mut self,
            numerators: &[FamilyNumerator<T>],
            report: &mut Report,
        ) -> RusticolResult<()> {
            let start = Instant::now();
            let groups: Vec<FamilyLanes<T>> = numerators
                .iter()
                .map(|numerator| {
                    lanes_of(
                        self.plan,
                        self.masses,
                        self.scales,
                        numerator,
                        self.data,
                        &mut self.sink.failures,
                    )
                })
                .filter(|group| !group.keys.is_empty())
                .collect();
            report.time("scatter", start);
            report.count(
                "reduction lanes",
                groups.iter().map(|g| g.keys.len() as u64).sum(),
            );
            if groups.is_empty() {
                return Ok(());
            }
            let start = Instant::now();
            let (direct, check) =
                self.backend
                    .reduce(&groups, self.options.poles, self.options.check)?;
            report.time("reduction", start);
            let statistics = self.backend.take_statistics();
            for (stage, seconds) in statistics.seconds {
                *report.seconds.entry(stage).or_default() += seconds;
            }
            for (name, value) in statistics.counts {
                report.count(name, value);
            }
            let sink = &mut *self.sink;
            for (index, group) in groups.iter().enumerate() {
                for (row, (&point, &key)) in group.points.iter().zip(&group.keys).enumerate() {
                    match &direct[index][row] {
                        Ok(value) => sink.families[point].push((key, value.clone())),
                        Err(status) => sink.failures[point]
                            .push(format!("family {} lane {key:?}: {status}", group.family)),
                    }
                    if let Some(check) = &check {
                        match &check[index][row] {
                            Ok(value) => sink.checks[point].push((key, value.clone())),
                            Err(status) => sink.failures[point].push(format!(
                                "re-test of family {} lane {key:?}: {status}",
                                group.family
                            )),
                        }
                    }
                }
            }
            Ok(())
        }
    }

    /// Record a schedule-M run in the report (reals per lane).
    fn merged_report(report: &mut Report, stats: &MergedStats, lanes: usize) {
        *report.seconds.entry("factors").or_default() += stats.seconds_inputs;
        *report.seconds.entry("dressing").or_default() += stats.seconds_dressing;
        report.count("merged states", stats.states);
        report.count("merged split states", stats.split_states);
        report.count("merged chunks", stats.chunks);
        report.peak(
            "chunk inputs (factors, vertex tensors, currents)",
            stats.peak_inputs,
            lanes,
        );
        report.peak("open objects (one merged chunk)", stats.peak_objects, lanes);
        report.peak("merged family accumulators", stats.peak_accumulators, lanes);
    }

    fn evaluate_lanes<T: Carrier + Real, const L: usize>(
        plan: &LoopProgramV4,
        fourier: Option<&SeedFourier>,
        batch: &V2Batch<T>,
        zero: &T,
        backend: &mut (impl V2Backend<T> + Send),
        options: V2Options,
    ) -> RusticolResult<(Vec<PointResult<T>>, Report)> {
        let points = batch.points();
        let width = options.poles.width();
        let mut report = Report {
            lanes: L,
            real_bytes: std::mem::size_of::<T>(),
            ..Report::default()
        };
        if options.schedule == Schedule::F {
            return Err(RusticolError::invalid_argument(
                "schedule F is an oracle of family numerators only",
            ));
        }
        let merged = merged_of(plan, options.schedule)?;
        let b_families = b_families(plan, merged);
        let chunks_of_buckets = bucket_chunks(plan, options.chunk_families, &b_families);
        let merged_parts = merged
            .map(|merged| parts(merged, options.part_chunks, plan))
            .unwrap_or_default();
        let mut sink = Lanes {
            failures: vec![Vec::new(); points],
            families: vec![Vec::new(); points],
            checks: vec![Vec::new(); points],
        };
        for chunk in chunks(points, L) {
            backend.begin();
            let start = Instant::now();
            let data = LaneData::<T, L>::new(plan, batch, &chunk, zero)?;
            let contraction = Instant::now();
            let seeds = match fourier {
                Some(fourier) => {
                    report.count(FFT_SEED_JOBS, fourier.jobs() as u64);
                    seeds_fourier(plan, fourier, &data)
                }
                None => seeds(plan, &data),
            };
            // The contraction alone ("seeds" also gathers the lane data).
            report.time(SEED_CONTRACTION, contraction);
            report.time("seeds", start);
            let mut inputs = ChunkInputs::new(plan);
            let masses: Vec<Complex<T>> = batch
                .masses
                .iter()
                .enumerate()
                .map(|(i, m)| {
                    let im = batch
                        .masses_imaginary
                        .get(i)
                        .map_or_else(|| data.zero.clone(), |im| data.cast(im));
                    Complex::new(data.cast(m), im)
                })
                .collect();
            let scales: Vec<Complex<T>> = data
                .scales
                .iter()
                .map(|scale| Complex::new(data.cast(scale), data.zero.clone()))
                .collect();
            let mut reduction = Reduction {
                plan,
                masses: &masses,
                scales: &scales,
                data: &data,
                backend: &mut *backend,
                options,
                sink: &mut sink,
            };
            for part in &chunks_of_buckets {
                let numerators =
                    dress_families(plan, part, &mut inputs, &seeds, &data, &mut report)?;
                reduction.reduce(&numerators, &mut report)?;
            }
            if !merged_parts.is_empty() {
                let stats = run_merged(plan, &merged_parts, &mut inputs, &seeds, &data, |done| {
                    reduction.reduce(&done, &mut report)
                })?;
                merged_report(&mut report, &stats, L);
                if let Some(merged) = merged {
                    report.count(
                        DRESSING_WORK,
                        modelled_work(merged.states.iter().map(|state| &state.step)),
                    );
                }
            }
        }
        let Lanes {
            mut failures,
            mut families,
            mut checks,
        } = sink;
        let zero = zero.zero();
        let pi = zero.pi();
        let factor = Complex::new(
            zero.clone(),
            zero.one() / (zero.from_usize(16) * &pi * &pi * zero.from_usize(plan.average as usize)),
        );
        let mut results = Vec::with_capacity(points);
        for point in 0..points {
            if !failures[point].is_empty() {
                results.push(PointResult {
                    families: Vec::new(),
                    check: None,
                    bare: Vec::new(),
                    budget: None,
                    failures: std::mem::take(&mut failures[point]),
                });
                continue;
            }
            let mut sums =
                vec![vec![Complex::new(zero.clone(), zero.clone()); width]; plan.columns];
            let mut lanes = std::mem::take(&mut families[point]);
            lanes.sort_by_key(|(key, _)| *key);
            for (key, value) in &lanes {
                for (s, z) in sums[key.column].iter_mut().zip(value) {
                    *s += z;
                }
            }
            let two = zero.from_usize(2);
            let bare = sums
                .iter()
                .map(|column| {
                    column
                        .iter()
                        .map(|z| two.clone() * (factor.clone() * z).re)
                        .collect()
                })
                .collect();
            let (check, budget) = if options.check.is_some() {
                let mut tests = std::mem::take(&mut checks[point]);
                tests.sort_by_key(|(key, _)| *key);
                let mut budget = vec![vec![zero.clone(); width]; plan.columns];
                for ((key, direct), (other, test)) in lanes.iter().zip(&tests) {
                    debug_assert_eq!(key, other);
                    add_to_budget(
                        &mut budget[key.column],
                        &deviation(direct, test),
                        plan.average,
                    );
                }
                (Some(tests), Some(budget))
            } else {
                (None, None)
            };
            results.push(PointResult {
                families: lanes,
                check,
                bare,
                budget,
                failures: Vec::new(),
            });
        }
        report.count("points", points as u64);
        Ok((results, report))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::loop_v2::dress::V2Batch;
        use symbolica::domains::rational::Rational;

        /// Two scalar bubble families of a two-leg "process": one seeded
        /// trace step each, the factor a constant, so each family numerator
        /// is `2 conj(A0)` at rank 0.
        const PLAN: &[u8] =
            br#"{"schema":4,"legs":2,"rows":1,"helicities":[[0],[0]],"columns":1,"average":1,
            "born_components":1,"constants":1,"masses":1,"block_rows":[],
            "kernels":[{"inputs":[],"outputs":[[[0,[]]]]}],
            "contents":[{"d_in":1,"d_out":1,"degree":0,"rows":1,"mask":0,"row_helicities":[[]],
                "fused":[[0,0,0,0,0,0]],
                "terms":[{"slots":[],"dimensions":[],"vertex":-1,"split":[],"rows":[[0,[]]],
                    "bindings":[{"kernel":0,"inputs":[],"entries":[0]}]}]}],
            "factors":[[0,0]],
            "seed_classes":[{"columns":[[[0,"1","0"]]]}],"seeds":[{"classes":[[0,"1","0"]]}],
            "families":[FAMILY,FAMILY],"forward":null}"#;
        const FAMILY: &str = r#"{"sequence":[[1,0],[2,0]],"rank":0,"orders":[0],
            "leaves":[{"seed":0,"weight":"1"}],
            "steps":[{"seeded":true,"trace":0,"out":0,"rows":1,"d_alpha":1,"d_out":1,"degree":0,
                "rank_in":0,"edges":[{"leaf":0,"child":-1,"factor":0,"links":[0,0,0]}]}]}"#;

        /// Fails every lane of point 0 in its first reduction and records the
        /// points each call receives.
        #[derive(Default)]
        struct Failing {
            calls: Vec<Vec<usize>>,
        }

        impl V2Backend<f64> for Failing {
            fn begin(&mut self) {}

            fn reduce(
                &mut self,
                groups: &[FamilyLanes<f64>],
                poles: Poles,
                check: Option<ReductionCheck>,
            ) -> RusticolResult<Reduced<f64>> {
                assert!(poles == Poles::All && check.is_none());
                let first = self.calls.is_empty();
                self.calls
                    .push(groups.iter().flat_map(|g| g.points.clone()).collect());
                let direct = groups
                    .iter()
                    .map(|group| {
                        group
                            .points
                            .iter()
                            .zip(group.coefficients.iter())
                            .map(|(&point, c)| -> LaneOutcome<f64> {
                                if first && point == 0 {
                                    Err("ImpreciseResult".into())
                                } else {
                                    let zero = Complex::new(0.0, 0.0);
                                    Ok(vec![*c, zero, zero])
                                }
                            })
                            .collect()
                    })
                    .collect();
                Ok((direct, None))
            }
        }

        #[test]
        fn failed_points_are_compacted_out_of_later_buckets() {
            let text = std::str::from_utf8(PLAN).unwrap().replace("FAMILY", FAMILY);
            let plan = LoopProgramV4::load(text.as_bytes()).unwrap();
            let executor = V2Executor::new(Arc::new(plan), DEFAULT_LANES, Some(1)).unwrap();
            let momentum = |e: i64| -> Vec<Rational> {
                [e, 0, 0, e, e, 0, 0, -e].map(Rational::from).to_vec()
            };
            let batch = V2Batch {
                momenta: vec![momentum(1), momentum(2)],
                scales: vec![Rational::from(1); 2],
                masses: vec![Rational::from(0)],
                masses_imaginary: Vec::new(),
                leg_masses: Vec::new(),
                constants: vec![Complex::new(2.0, 0.0)],
                block_rows: vec![Vec::new(); 2],
                born: vec![vec![Complex::new(1.0, 1.0)], vec![Complex::new(3.0, 0.0)]],
            };
            let options = V2Options {
                chunk_families: 1,
                ..V2Options::default()
            };
            let mut backend = Failing::default();
            let (results, report) = executor
                .evaluate(&batch, &0.0, &mut backend, options)
                .unwrap();
            // One bucket chunk per family; point 0 fails in the first and is
            // not reduced again.
            assert_eq!(backend.calls, [vec![0, 1], vec![1]]);
            assert_eq!(results[0].failures.len(), 1);
            assert!(results[0].failures[0].contains("ImpreciseResult"));
            assert!(results[0].bare.is_empty());
            let lanes = &results[1].families;
            assert!(results[1].failures.is_empty() && lanes.len() == 2);
            for (_key, value) in lanes {
                assert_eq!(value[0], Complex::new(6.0, 0.0));
            }
            assert_eq!(report.counts["reduction lanes"], 3);
        }

        /// Schedule M needs a plan exported with a merged graph; an empty
        /// graph (no certified family) runs every family on schedule B.
        #[test]
        fn schedule_m_falls_back_to_b_without_certified_families() {
            let text = std::str::from_utf8(PLAN).unwrap().replace("FAMILY", FAMILY);
            let batch = V2Batch {
                momenta: vec![[1, 0, 0, 1, 1, 0, 0, -1].map(Rational::from).to_vec()],
                scales: vec![Rational::from(1)],
                masses: vec![Rational::from(0)],
                masses_imaginary: Vec::new(),
                leg_masses: Vec::new(),
                constants: vec![Complex::new(2.0, 0.0)],
                block_rows: vec![Vec::new()],
                born: vec![vec![Complex::new(1.0, 1.0)]],
            };
            let executor = |text: &str| {
                let plan = LoopProgramV4::load(text.as_bytes()).unwrap();
                V2Executor::new(Arc::new(plan), 2, Some(1)).unwrap()
            };
            let without = executor(&text);
            let error = without
                .numerators(&batch, &0.0, Schedule::M)
                .expect_err("schedule M without a merged graph");
            assert!(error.message().contains("no schedule M"), "{error}");
            let options = V2Options {
                schedule: Schedule::M,
                ..V2Options::default()
            };
            assert!(
                without
                    .evaluate(&batch, &0.0, &mut Failing::default(), options)
                    .is_err()
            );
            let empty = text.replace(
                r#""forward":null}"#,
                r#""forward":null,"merged":{"families":[],"entries":[],"states":[],"chunks":[]}}"#,
            );
            let with = executor(&empty);
            assert_eq!(with.plan().merged.as_ref().unwrap().families.len(), 0);
            let b = with.numerators(&batch, &0.0, Schedule::B).unwrap();
            let m = with.numerators(&batch, &0.0, Schedule::M).unwrap();
            assert_eq!(m.len(), 1);
            assert_eq!(m[0].len(), 2);
            for (mine, theirs) in m[0].iter().zip(&b[0]) {
                assert_eq!((mine.0, mine.1, mine.2), (theirs.0, theirs.1, theirs.2));
                assert_eq!(mine.3, theirs.3);
            }
        }
    }
}
