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
    use crate::loop_v2::dress::{
        Carrier, ChunkInputs, LaneData, Reads, V2Batch, Workspaces, input_vectors,
    };
    use crate::loop_v2::forward::{forward_factors, run_forward};
    pub use crate::loop_v2::fourier::SeedFourier;
    use crate::loop_v2::fourier::seeds_fourier;
    use crate::loop_v2::merged::{DEFAULT_PART_CHUNKS, MergedStats, parts, run_merged};
    pub use crate::loop_v2::plan::LiveRows;
    use crate::loop_v2::plan::{Family, LoopProgramV4, Merged};
    use crate::loop_v2::retest::{ReductionCheck, add_to_budget, deviation};
    use crate::loop_v2::seeds::seeds;
    use crate::loop_v2::trie::run_family_on;
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
    /// Default bytes of chunk inputs (fused factors, split vertex tensors
    /// and currents) one lane chunk holds where it can
    /// ([`V2Options::input_budget`]).
    pub const DEFAULT_INPUT_BUDGET: usize = 8 << 30;

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
        /// Bytes of chunk inputs a lane chunk holds where it can (schedule
        /// B): a live input the current bucket chunk does not read is
        /// dropped, the one read again latest first, and rebuilt bitwise
        /// when read again (`ChunkInputs::prepare_within`; heap limbs of
        /// arbitrary-precision reals excluded).
        pub input_budget: usize,
        /// Amplitude-mode plans (`LoopProgramV4::amplitude`): also return
        /// every raw reduction lane ([`PointResult::families`], and the
        /// re-test's), a diagnostic and oracle; by default a point returns
        /// only its amplitudes, their re-test and budget (O(rows x basis)
        /// values per loop order). Seeded plans always return their lanes.
        pub raw_lanes: bool,
        /// Amplitude-mode plans: also return each (loop order, column)
        /// slot's term scale `Σ_f Σ_α |N_{f,α}| |I_{f,α}| / (16π²)` per
        /// Laurent coefficient ([`PointResult::amplitude_scale`]), the size
        /// of the amplitude before the reduction's cancellations (the
        /// finiteness and stability scale of a loop-induced point, m-lit
        /// DECISIONS 2026-11-06 ~09:00 (1)). `I_{f,α}` is the integral of
        /// monomial `α` of family `f`: one unit-numerator reduction per
        /// monomial, family and point, so the option costs `C(R+4, 4)`
        /// lanes per family and point; refused for seeded plans.
        pub term_scale: bool,
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
                input_budget: DEFAULT_INPUT_BUDGET,
                raw_lanes: false,
                term_scale: false,
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

    /// The kind of a refused lane's measure ([`LaneMeasure`]).
    #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub enum MeasureKind {
        /// A method-choice lane's deviation between its two frames relative
        /// to its term scale (what TIRust's acceptance bounds).
        MethodChoiceDeviation,
        /// The reducer's conditioning estimate of a refused lane (TIRust's
        /// `last_conditions`: the Gram, Cayley or coordinate-system measure
        /// compared with its stability threshold; zero for an exactly
        /// singular pivot).
        GramConditioning,
    }

    impl MeasureKind {
        /// The kind's name in the Python and SDK records.
        pub fn name(self) -> &'static str {
            match self {
                Self::MethodChoiceDeviation => "method-choice-deviation",
                Self::GramConditioning => "gram-conditioning",
            }
        }
    }

    /// The backend's own accuracy measure of a refused lane, as data next
    /// to its reason text (m-lit ux-polish item 4).
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct LaneMeasure {
        pub kind: MeasureKind,
        pub value: f64,
    }

    /// A refused lane's measure of the last [`V2Backend::reduce`]: lane
    /// `lane` of group `group`, of the re-test (`check`) or the direct
    /// outcome.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct MeasuredLane {
        pub group: usize,
        pub lane: usize,
        pub check: bool,
        pub measure: LaneMeasure,
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
        /// The measures of the refused lanes of the last `reduce` that the
        /// backend states one for (none by default; TIRust's).
        fn take_lane_measures(&mut self) -> Vec<MeasuredLane> {
            Vec::new()
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

    /// Report stage and count of the term scale's unit-numerator reductions
    /// ([`V2Options::term_scale`]).
    pub const TERM_SCALE: &str = "term scale";
    pub const TERM_SCALE_LANES: &str = "term-scale lanes";
    /// Report count of the modelled dressing work (complex MAC per lane).
    const DRESSING_WORK: &str = "dressing work (modelled complex MAC per lane)";
    /// Report count of the chunk inputs dropped while still live (rebuilt
    /// when read again, `V2Options::input_budget`).
    pub const INPUTS_DROPPED: &str = "chunk inputs dropped early";

    /// One point's family numerators: (family, loop orders, rank, per
    /// column the coefficients in the family's trie frame).
    pub type PointNumerators<T> = Vec<(usize, usize, usize, Vec<Vec<Complex<T>>>)>;

    /// One point of an evaluation.
    #[derive(Clone)]
    pub struct PointResult<T> {
        /// Raw Laurent integrals of every reduced lane (amplitude-mode
        /// plans: only with [`V2Options::raw_lanes`]).
        pub families: Vec<(LaneKey, Vec<Complex<T>>)>,
        /// The re-test integrals, when a check was asked (amplitude-mode
        /// plans: only with [`V2Options::raw_lanes`]).
        pub check: Option<Vec<(LaneKey, Vec<Complex<T>>)>>,
        /// Per observable column, the bare virtual `2 Re(i/(16π² avg) ΣΩ)`
        /// per Laurent coefficient (empty for amplitude-mode plans).
        pub bare: Vec<Vec<T>>,
        /// Per observable column, the re-test error budget (seeded plans).
        pub budget: Option<Vec<Vec<T>>>,
        /// Amplitude-mode plans (loop-induced, M-LIT LIT-AMP): per loop
        /// order and column, at `order * columns + column`, the bare loop
        /// amplitude `a = i/(16π²) Σ_f Ω_f` per Laurent coefficient, the
        /// families summed in lane order (family, then column); empty for
        /// seeded plans.
        pub amplitudes: Vec<Vec<Complex<T>>>,
        /// The re-test's amplitudes, when a check was asked.
        pub amplitude_check: Option<Vec<Vec<Complex<T>>>>,
        /// Per loop order and column `Σ_f |Ω_f − Ω'_f| / (16π²)` per
        /// coefficient: the amplitude's re-test budget in which no two
        /// families cancel (the caller forms the quadratic budget of |A|²
        /// with the counterterm kets and the colour metric).
        pub amplitude_budget: Option<Vec<Vec<T>>>,
        /// With [`V2Options::term_scale`], per loop order and column the
        /// term scale `Σ_f Σ_α |N_{f,α}| |I_{f,α}| / (16π²)` per
        /// coefficient: the amplitude's size before the reduction's own
        /// cancellations (the caller forms the per-row term floor with the
        /// counterterm kets and the colour metric); else empty.
        pub amplitude_scale: Vec<Vec<T>>,
        /// Numerical failures; a failed point has no values.
        pub failures: Vec<String>,
        /// The backend's measures of the point's refused lanes, where it
        /// states one ([`V2Backend::take_lane_measures`]); empty otherwise.
        pub measures: Vec<LaneMeasure>,
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
        /// The latest row liveness formed ([`Self::live_rows`]): calls
        /// repeating its rows and families reuse it (one is kept).
        live: std::sync::Mutex<Option<Arc<LiveRows>>>,
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
                live: std::sync::Mutex::new(None),
            })
        }

        /// An executor of the same loaded plan (and, with `fourier`, the same
        /// certified seed FFT), shared rather than decoded again, with its own
        /// lane count and thread pool.
        pub fn derive(
            &self,
            lanes: usize,
            threads: Option<usize>,
            fourier: bool,
        ) -> RusticolResult<Self> {
            let mut out = Self::new(self.plan.clone(), lanes, threads)?;
            if fourier {
                out.fourier = self.fourier.clone();
            }
            Ok(out)
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
            self.evaluate_families(batch, zero, backend, options, None, None)
        }

        /// The per-call row liveness of an evaluation by `schedule` keeping
        /// the Born rows `rows` (one flag per plan row) and dressing the
        /// families `families` keeps (None: every family): the families
        /// pruned to the rows the kept rows reach and the input rows they
        /// read (`LiveRows`, DESIGN.md §8.5), or None when `rows` drops no
        /// seeded row or the schedule is not B (schedules M and F dress
        /// their own programs whole). [`Self::evaluate_families`] and the
        /// tree fill (`TreeProgram::live_rows`) take it. The latest one is
        /// kept and returned again for the same rows and families (and
        /// dropped before another one is formed).
        pub fn live_rows(
            &self,
            rows: &[bool],
            families: Option<&[bool]>,
            schedule: Schedule,
        ) -> RusticolResult<Option<Arc<LiveRows>>> {
            if families.is_some_and(|kept| kept.len() != self.plan.families.len()) {
                return Err(RusticolError::invalid_argument(
                    "one family flag per plan family",
                ));
            }
            if schedule != Schedule::B {
                return Ok(None);
            }
            let mut listed = b_families(&self.plan, None);
            if let Some(kept) = families {
                listed.retain(|&f| kept[f]);
            }
            // A hit returns the latest; a miss drops it first, so one call
            // never holds two (the pruned copies are the bulk of one).
            let latest = self.live.lock().ok().and_then(|mut latest| {
                let hit = latest
                    .as_ref()
                    .filter(|live| live.rows == rows && live.listed() == listed)
                    .cloned();
                if hit.is_none() {
                    *latest = None;
                }
                hit
            });
            if latest.is_some() {
                return Ok(latest);
            }
            let live = self.plan.live_rows(rows, &listed)?.map(Arc::new);
            if let (Some(live), Ok(mut latest)) = (&live, self.live.lock()) {
                *latest = Some(live.clone());
            }
            Ok(live)
        }

        /// [`Self::evaluate`] dressing only the families `families` keeps
        /// (one flag per plan family; None: every family): an evaluation
        /// whose seeds vanish on the other families (an evaluation-time
        /// selection's masks, `V2Executor.evaluate`) skips them, and their
        /// lanes are absent, as a plan without them has none. With `live`
        /// (`Self::live_rows` of the batch's kept Born rows and the same
        /// family mask; every other row's Born components vanish) each
        /// family runs its steps pruned to the live rows and the factors and
        /// currents are built on the rows those read; families left without
        /// a step are skipped too.
        pub fn evaluate_families<T: Carrier + Real, B: V2Backend<T> + Send>(
            &self,
            batch: &V2Batch<T>,
            zero: &T,
            backend: &mut B,
            options: V2Options,
            families: Option<&[bool]>,
            live: Option<&LiveRows>,
        ) -> RusticolResult<(Vec<PointResult<T>>, Report)> {
            batch.validate(&self.plan)?;
            if families.is_some_and(|kept| kept.len() != self.plan.families.len()) {
                return Err(RusticolError::invalid_argument(
                    "one family flag per plan family",
                ));
            }
            let threads = self.pool.current_num_threads();
            let fourier = self.contraction(options.contraction)?;
            self.pool.install(|| {
                let (results, mut report) = with_lanes!(
                    self.lanes,
                    evaluate_lanes::<T>(
                        &self.plan, fourier, batch, zero, backend, options, families, live
                    )
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
            let mut seeds = match fourier {
                Some(fourier) => seeds_fourier(plan, fourier, &data),
                None => seeds(plan, &data),
            };
            seeds.materialise::<L>(zero);
            for (lane, &point) in chunk.iter().enumerate() {
                out[point] = seeds
                    .values()
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

    /// A bucket chunk: its families and the inputs they read.
    struct BucketChunk {
        families: Vec<usize>,
        reads: Reads,
    }

    /// Per chunk-input store (factors, vertex tensors, currents) and id, the
    /// bucket chunks reading it, ascending (compressed rows): whether a held
    /// input is still live and which chunk reads it next
    /// (`ChunkInputs::prepare_within`).
    struct ReadChunks([(Vec<u32>, Vec<u32>); 3]);

    impl ReadChunks {
        fn of(plan: &LoopProgramV4, chunks: &[BucketChunk]) -> Self {
            let lens = [
                plan.factors.len(),
                plan.vertex_instances.len(),
                plan.currents.len(),
            ];
            fn ids(chunk: &BucketChunk, store: usize) -> &[usize] {
                match store {
                    0 => &chunk.reads.factors,
                    1 => &chunk.reads.vertices,
                    _ => &chunk.reads.currents,
                }
            }
            Self(std::array::from_fn(|store| {
                let mut offsets = vec![0u32; lens[store] + 1];
                for chunk in chunks {
                    for &id in ids(chunk, store) {
                        offsets[id + 1] += 1;
                    }
                }
                for id in 0..lens[store] {
                    offsets[id + 1] += offsets[id];
                }
                let mut fill = offsets.clone();
                let mut reads = vec![0u32; offsets[lens[store]] as usize];
                for (k, chunk) in chunks.iter().enumerate() {
                    for &id in ids(chunk, store) {
                        reads[fill[id] as usize] = k as u32;
                        fill[id] += 1;
                    }
                }
                (offsets, reads)
            }))
        }

        fn reads(&self, store: usize, id: usize) -> &[u32] {
            let (offsets, reads) = &self.0[store];
            &reads[offsets[id] as usize..offsets[id + 1] as usize]
        }

        /// Whether input `id` of `store` is read by chunk `k` or a later one.
        fn live(&self, store: usize, id: usize, k: usize) -> bool {
            self.reads(store, id)
                .last()
                .is_some_and(|&j| j as usize >= k)
        }

        /// The first chunk after `k` reading input `id` of `store`
        /// (`usize::MAX`: none).
        fn next(&self, store: usize, id: usize, k: usize) -> usize {
            let reads = self.reads(store, id);
            let at = reads.partition_point(|&j| j as usize <= k);
            reads.get(at).map_or(usize::MAX, |&j| j as usize)
        }
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

    /// The lane vectors an amplitude-mode family's numerator and its
    /// reduction lanes take in one bucket chunk: per (loop order, column)
    /// its coefficients twice (the numerator, and the lanes' copy) and the
    /// lanes' kinematics (`4N` shifts, `N` masses, the scale and the `N²`
    /// invariants; `lanes_of`), one lane vector each (m-lit PLAN §2.b.3:
    /// byte-bounded numerator chunks for loop-induced, within
    /// [`V2Options::input_budget`]).
    fn numerator_vectors(plan: &LoopProgramV4, family: &Family) -> usize {
        let n = family.propagators();
        let per_column = 2 * ninjars::fast::tensor_layout::ncoeff(family.rank) + 5 * n + 1 + n * n;
        family.orders.len() * plan.columns * per_column
    }

    /// Bucket chunks of `families` (ascending) in bucket order: runs of at
    /// most `size` families of one bucket whose chunk inputs take at most
    /// `vectors` lane vectors (a family alone may take more; `usize::MAX`:
    /// runs of `size`). In an amplitude-mode plan (loop-induced; decided
    /// once per plan) a family's numerator and reduction lanes count too
    /// ([`numerator_vectors`]): its many columns (rows × colour components)
    /// make them the largest objects of a chunk, so the one budget bounds
    /// them; seeded plans keep their chunks.
    fn bucket_chunks(
        plan: &LoopProgramV4,
        size: usize,
        families: &[usize],
        vectors: usize,
        live: Option<&LiveRows>,
    ) -> Vec<BucketChunk> {
        let bounded = vectors < usize::MAX;
        let lens = [
            plan.factors.len(),
            plan.vertex_instances.len(),
            plan.currents.len(),
        ];
        let mut seen: [Vec<u32>; 3] = if bounded {
            lens.map(|len| vec![0; len])
        } else {
            Default::default()
        };
        let mut stamp = 0u32;
        let mut parts: Vec<Vec<usize>> = Vec::new();
        for members in plan.buckets().values() {
            let mut current: Vec<usize> = Vec::new();
            let mut held = 0usize;
            stamp += 1;
            for &f in members {
                if families.binary_search(&f).is_err() {
                    continue;
                }
                let family = family_of(plan, live, f);
                let ids = [&family.factors, &family.vertices, &family.currents];
                // The lane vectors `f` adds to a chunk holding the `stamp`ed ids.
                let new = |seen: &[Vec<u32>; 3], stamp: u32| -> usize {
                    if !bounded {
                        return 0;
                    }
                    let mut total = 0usize;
                    for (store, ids) in ids.iter().enumerate() {
                        for &id in ids.iter() {
                            if seen[store][id] != stamp {
                                total += input_vectors(plan, store, id);
                            }
                        }
                    }
                    total
                };
                let numerator = if bounded && plan.amplitude {
                    numerator_vectors(plan, family)
                } else {
                    0
                };
                let mut add = new(&seen, stamp) + numerator;
                if !current.is_empty()
                    && (current.len() >= size.max(1) || held.saturating_add(add) > vectors)
                {
                    parts.push(std::mem::take(&mut current));
                    stamp += 1;
                    held = 0;
                    add = new(&seen, stamp) + numerator;
                }
                if bounded {
                    for (store, ids) in ids.iter().enumerate() {
                        for &id in ids.iter() {
                            seen[store][id] = stamp;
                        }
                    }
                }
                held += add;
                current.push(f);
            }
            if !current.is_empty() {
                parts.push(current);
            }
        }
        parts
            .into_iter()
            .map(|families| BucketChunk {
                reads: Reads::of_families(families.iter().map(|&f| family_of(plan, live, f))),
                families,
            })
            .collect()
    }

    /// Plan family `f`, or its copy pruned to an evaluation's live rows.
    fn family_of<'a>(plan: &'a LoopProgramV4, live: Option<&'a LiveRows>, f: usize) -> &'a Family {
        live.and_then(|live| live.family(f))
            .unwrap_or(&plan.families[f])
    }

    /// Dress every family of `chunk` on one chunk of lanes (schedule B);
    /// `bound`: the chunk's index, the bucket chunks' reads and the reals
    /// the chunk inputs hold where they can (none: unbounded).
    #[allow(clippy::too_many_arguments)]
    fn dress_families<T: Carrier, const L: usize>(
        plan: &LoopProgramV4,
        chunk: &BucketChunk,
        inputs: &mut ChunkInputs<T>,
        seeds: &crate::loop_v2::seeds::Seeds<T>,
        data: &LaneData<T, L>,
        report: &mut Report,
        bound: Option<(usize, &ReadChunks, usize)>,
        live: Option<&LiveRows>,
        workspaces: &Workspaces<T>,
    ) -> RusticolResult<Vec<FamilyNumerator<T>>> {
        let families = &chunk.families;
        let start = Instant::now();
        match bound {
            Some((k, reads, budget)) => {
                let dropped = inputs.prepare_within(
                    plan,
                    data,
                    &chunk.reads,
                    budget,
                    |store, id| reads.live(store, id, k),
                    |store, id| reads.next(store, id, k),
                );
                if dropped > 0 {
                    report.count(INPUTS_DROPPED, dropped as u64);
                }
            }
            // One chunk: every input it reads is live.
            None => inputs.prepare(plan, data, &chunk.reads, &chunk.reads),
        }
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
            .map(|&f| {
                run_family_on::<T, L>(
                    plan,
                    f,
                    family_of(plan, live, f),
                    inputs,
                    seeds,
                    data,
                    workspaces,
                )
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        report.time("dressing", start);
        let peak = runs.iter().map(|run| run.peak_reals).max().unwrap_or(0);
        report.peak("open objects (one family walk)", peak, L);
        report.count(
            "closure columns",
            runs.iter().map(|run| run.closures as u64).sum(),
        );
        report.count("families", families.len() as u64);
        let steps = || {
            families
                .iter()
                .flat_map(|&f| &family_of(plan, live, f).steps)
        };
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
        let workspaces = Workspaces::new(rayon::current_num_threads(), zero);
        for points in chunks(batch.points(), L) {
            let data = LaneData::<T, L>::new(plan, batch, &points, zero)?;
            let mut seeds = seeds(plan, &data);
            if schedule != Schedule::B || merged.is_some() {
                seeds.materialise::<L>(zero);
            }
            let mut inputs = ChunkInputs::new(plan);
            let numerators = match schedule {
                Schedule::B | Schedule::M => {
                    let families = b_families(plan, merged);
                    let reads = Reads::of(plan, &families);
                    let chunk = BucketChunk { families, reads };
                    let mut numerators = dress_families(
                        plan,
                        &chunk,
                        &mut inputs,
                        &seeds,
                        &data,
                        &mut report,
                        None,
                        None,
                        &workspaces,
                    )?;
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
        // One pass over the accumulators gives every lane's finiteness; the
        // stored columns (a dressed-row family's support, lit-dress lever D)
        // are those that can be non-zero, read in column order straight into
        // the lanes (an all-zero column has no lane, as before). A dependent
        // column (lit-fold F1d, `plan::Dependents`) has no lane: it is formed
        // from its representative's after the reduction.
        let finite = numerator.finite_lanes::<L>();
        let dependents = family_dependents(plan, family);
        for lane in 0..data.active {
            let point = data.points[lane];
            if !failures[point].is_empty() {
                continue;
            }
            if !finite[lane] {
                failures[point].push(format!("family {}: non-finite numerator", numerator.family));
                continue;
            }
            for (k, &order) in family.orders.iter().enumerate() {
                let skip = dependents.map(|d| d.dependent[k].as_slice());
                for (slot, column) in numerator.stored() {
                    if skip.is_some_and(|skip| skip[column]) {
                        continue;
                    }
                    let block = numerator.block::<L>(k, slot);
                    if block
                        .chunks_exact(v)
                        .all(|z| z[lane].is_fully_zero() && z[L + lane].is_fully_zero())
                    {
                        continue;
                    }
                    let coefficients = block
                        .chunks_exact(v)
                        .map(|z| Complex::new(z[lane].clone(), z[L + lane].clone()));
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

    /// The dependent columns of a dressed-row family (lit-fold F1d,
    /// [`crate::loop_v2::plan::Dependents`], derived by its first
    /// reduction), or `None` when every stored column is reduced (seeded
    /// plans, families without one).
    fn family_dependents<'a>(
        plan: &LoopProgramV4,
        family: &'a Family,
    ) -> Option<&'a crate::loop_v2::plan::Dependents> {
        if !plan.dressed {
            return None;
        }
        let dependents = family.dependents.get_or_init(|| plan.dependents_of(family));
        (!dependents.is_empty()).then_some(dependents)
    }

    /// The dependent columns of one family as a reduction forms them
    /// (lit-fold F1d): the family's [`crate::loop_v2::plan::Dependents`]
    /// with each exact `c` cast to the carrier once.
    struct Formed<'a, T> {
        family: &'a Family,
        dependents: &'a crate::loop_v2::plan::Dependents,
        /// Per accumulator, per entry of `dependents.columns[k]`: `c`
        /// (re, and im unless it is exactly zero) and `|c|`.
        factors: Vec<Vec<(T, Option<T>, T)>>,
    }

    impl<'a, T: Carrier + Real> Formed<'a, T> {
        fn of<const L: usize>(
            plan: &'a LoopProgramV4,
            family: usize,
            data: &LaneData<T, L>,
        ) -> Option<Self> {
            let family = &plan.families[family];
            let dependents = family_dependents(plan, family)?;
            let factors = dependents
                .columns
                .iter()
                .map(|list| {
                    list.iter()
                        .map(|(_, _, [re, im])| {
                            let (x, y) = (data.cast(re), data.cast(im));
                            let size = Complex::new(x.clone(), y.clone()).norm().re;
                            (x, (!im.is_zero()).then_some(y), size)
                        })
                        .collect()
                })
                .collect();
            Some(Self {
                family,
                dependents,
                factors,
            })
        }

        /// The accumulator of `key`'s loop orders.
        fn accumulator(&self, key: &LaneKey) -> usize {
            self.family
                .orders
                .iter()
                .position(|&order| order == key.order)
                .expect("a lane of the family's loop orders")
        }

        /// Push the dependents of `key`'s column: `c value` (a real `c`
        /// scales both parts) under their own keys.
        fn push(
            &self,
            key: &LaneKey,
            value: &[Complex<T>],
            out: &mut Vec<(LaneKey, Vec<Complex<T>>)>,
        ) {
            let k = self.accumulator(key);
            for e in self.dependents.of(k, key.column) {
                let (_, column, _) = &self.dependents.columns[k][e];
                let (re, im, _) = &self.factors[k][e];
                let formed = match im {
                    None => value
                        .iter()
                        .map(|z| Complex::new(re.clone() * &z.re, re.clone() * &z.im))
                        .collect(),
                    Some(im) => {
                        let factor = Complex::new(re.clone(), im.clone());
                        value.iter().map(|z| factor.clone() * z).collect()
                    }
                };
                out.push((
                    LaneKey {
                        column: *column as usize,
                        ..*key
                    },
                    formed,
                ));
            }
        }

        /// The dependents of `key`'s column with `|c|`.
        fn sizes(&self, key: &LaneKey) -> Vec<(usize, T)> {
            let k = self.accumulator(key);
            self.dependents
                .of(k, key.column)
                .map(|e| {
                    (
                        self.dependents.columns[k][e].1 as usize,
                        self.factors[k][e].2.clone(),
                    )
                })
                .collect()
        }
    }

    /// What the reductions of one evaluation collect per point.
    struct Lanes<T> {
        failures: Vec<Vec<String>>,
        /// Per point the backend's measures of its refused lanes.
        measures: Vec<Vec<LaneMeasure>>,
        families: Vec<Vec<(LaneKey, Vec<Complex<T>>)>>,
        checks: Vec<Vec<(LaneKey, Vec<Complex<T>>)>>,
        /// [`V2Options::term_scale`]: per point and (loop order, column)
        /// slot `Σ_f Σ_α |N_{f,α}| |I_{f,α}|` per coefficient (without the
        /// loop factor); empty until a point's first lane is reduced.
        terms: Vec<Vec<Vec<T>>>,
    }

    /// The unit-numerator lanes of a family group: per point of the group
    /// (its lanes are point-major, `lanes_of`) one lane per monomial `α` of
    /// the family's coefficient layout with numerator `q^α`, at that point's
    /// kinematics (key column = `α`). Their integrals are the family's
    /// tensor moments `I_{f,α}` ([`V2Options::term_scale`]).
    fn unit_lanes<T: Carrier>(group: &FamilyLanes<T>, zero: &T) -> FamilyLanes<T> {
        let lanes = group.keys.len();
        let monomials = group.coefficients.len() / lanes.max(1);
        let n = group.propagators;
        let one = Complex::new(zero.one(), zero.clone());
        let nil = Complex::new(zero.clone(), zero.clone());
        let mut units = FamilyLanes {
            family: group.family,
            propagators: n,
            rank: group.rank,
            points: Vec::new(),
            keys: Vec::new(),
            shifts: Vec::new(),
            masses: Vec::new(),
            scales: Vec::new(),
            s_matrix: Vec::new(),
            coefficients: Vec::new(),
        };
        for lane in 0..lanes {
            let point = group.points[lane];
            if lane > 0 && group.points[lane - 1] == point {
                continue;
            }
            for alpha in 0..monomials {
                units.points.push(point);
                units.keys.push(LaneKey {
                    family: group.family,
                    order: 0,
                    column: alpha,
                });
                units
                    .shifts
                    .extend_from_slice(&group.shifts[lane * 4 * n..(lane + 1) * 4 * n]);
                units
                    .masses
                    .extend_from_slice(&group.masses[lane * n..(lane + 1) * n]);
                units.scales.push(group.scales[lane].clone());
                units
                    .s_matrix
                    .extend_from_slice(&group.s_matrix[lane * n * n..(lane + 1) * n * n]);
                units.coefficients.extend(
                    (0..monomials).map(|m| if m == alpha { one.clone() } else { nil.clone() }),
                );
            }
        }
        units
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

    impl<T: Carrier + Real, B: V2Backend<T>, const L: usize> Reduction<'_, T, B, L> {
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
            for measured in self.backend.take_lane_measures() {
                if let Some(&point) = groups
                    .get(measured.group)
                    .and_then(|group| group.points.get(measured.lane))
                {
                    sink.measures[point].push(measured.measure);
                }
            }
            for (index, group) in groups.iter().enumerate() {
                // A family's dependent columns (lit-fold F1d): `c` times
                // their representative's integrals, direct and re-test.
                let dependents = Formed::of(self.plan, group.family, self.data);
                for (row, (&point, &key)) in group.points.iter().zip(&group.keys).enumerate() {
                    match &direct[index][row] {
                        Ok(value) => {
                            if let Some(formed) = &dependents {
                                formed.push(&key, value, &mut sink.families[point]);
                            }
                            sink.families[point].push((key, value.clone()));
                        }
                        Err(status) => sink.failures[point]
                            .push(format!("family {} lane {key:?}: {status}", group.family)),
                    }
                    if let Some(check) = &check {
                        match &check[index][row] {
                            Ok(value) => {
                                if let Some(formed) = &dependents {
                                    formed.push(&key, value, &mut sink.checks[point]);
                                }
                                sink.checks[point].push((key, value.clone()));
                            }
                            Err(status) => sink.failures[point].push(format!(
                                "re-test of family {} lane {key:?}: {status}",
                                group.family
                            )),
                        }
                    }
                }
            }
            if self.options.term_scale {
                self.term_scale(&groups, report)?;
            }
            Ok(())
        }

        /// [`V2Options::term_scale`]: reduce every group's unit-numerator
        /// lanes ([`unit_lanes`], no re-test) and add, per lane,
        /// `Σ_α |N_α| |I_α|` per Laurent coefficient to its point's (loop
        /// order, column) slot.
        fn term_scale(
            &mut self,
            groups: &[FamilyLanes<T>],
            report: &mut Report,
        ) -> RusticolResult<()> {
            let start = Instant::now();
            let zero = &self.data.zero;
            let units: Vec<FamilyLanes<T>> =
                groups.iter().map(|group| unit_lanes(group, zero)).collect();
            report.count(
                TERM_SCALE_LANES,
                units.iter().map(|g| g.keys.len() as u64).sum(),
            );
            let (moments, _) = self.backend.reduce(&units, self.options.poles, None)?;
            let width = self.options.poles.width();
            let slots = self.plan.order_count * self.plan.columns;
            let sink = &mut *self.sink;
            for measured in self.backend.take_lane_measures() {
                if let Some(&point) = units
                    .get(measured.group)
                    .and_then(|unit| unit.points.get(measured.lane))
                {
                    sink.measures[point].push(measured.measure);
                }
            }
            for ((group, unit), moments) in groups.iter().zip(&units).zip(&moments) {
                let monomials = group.coefficients.len() / group.keys.len().max(1);
                let formed = Formed::of(self.plan, group.family, self.data);
                let mut first = 0;
                for (lane, (&point, &key)) in group.points.iter().zip(&group.keys).enumerate() {
                    if lane > 0 && group.points[lane - 1] != point {
                        first += monomials;
                    }
                    debug_assert_eq!(unit.points[first], point);
                    let terms = &mut sink.terms[point];
                    if terms.is_empty() {
                        *terms = vec![vec![zero.clone(); width]; slots];
                    }
                    let slot = &mut terms[key.order * self.plan.columns + key.column];
                    let numerator = &group.coefficients[lane * monomials..(lane + 1) * monomials];
                    for (alpha, n) in numerator.iter().enumerate() {
                        if n.re.is_fully_zero() && n.im.is_fully_zero() {
                            continue;
                        }
                        match &moments[first + alpha] {
                            Ok(moment) => {
                                let size = n.clone().norm().re;
                                for (s, z) in slot.iter_mut().zip(moment) {
                                    *s += size.clone() * &z.clone().norm().re;
                                }
                            }
                            Err(status) => sink.failures[point].push(format!(
                                "term scale of family {} monomial {alpha}: {status}",
                                group.family
                            )),
                        }
                    }
                    // Dependent columns (lit-fold F1d): |c| times the term
                    // scale of their representative, in the same terms.
                    if let Some(formed) = &formed {
                        for (column, size) in formed.sizes(&key) {
                            let slot = &mut terms[key.order * self.plan.columns + column];
                            for (alpha, n) in numerator.iter().enumerate() {
                                if n.re.is_fully_zero() && n.im.is_fully_zero() {
                                    continue;
                                }
                                if let Ok(moment) = &moments[first + alpha] {
                                    let n = size.clone() * &n.clone().norm().re;
                                    for (s, z) in slot.iter_mut().zip(moment) {
                                        *s += n.clone() * &z.clone().norm().re;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            report.time(TERM_SCALE, start);
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

    #[allow(clippy::too_many_arguments)]
    fn evaluate_lanes<T: Carrier + Real, const L: usize>(
        plan: &LoopProgramV4,
        fourier: Option<&SeedFourier>,
        batch: &V2Batch<T>,
        zero: &T,
        backend: &mut (impl V2Backend<T> + Send),
        options: V2Options,
        families: Option<&[bool]>,
        live: Option<&LiveRows>,
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
        let mut b_families = b_families(plan, merged);
        if let Some(kept) = families {
            if merged.is_some() {
                return Err(RusticolError::invalid_argument(
                    "a family mask runs schedule B (schedule M dresses its merged families together)",
                ));
            }
            b_families.retain(|&f| kept[f]);
        }
        // The row liveness: the listed families' pruned steps, or their own
        // where a copy would keep most links (those left without a step
        // have a zero numerator and are not dressed).
        if let Some(live) = live {
            if merged.is_some() || live.listed() != b_families.as_slice() {
                return Err(RusticolError::invalid_argument(
                    "a row liveness of the family mask's schedule-B families",
                ));
            }
            b_families = live.families().to_vec();
            report.count("live-row links", live.links() as u64);
            report.count("row-pruned families", live.pruned() as u64);
            report.count("row-pruned family bytes", live.bytes() as u64);
            report.count("row-liveness dressed bytes", live.dressed_bytes() as u64);
        }
        // The chunk-input budget in reals, and in lane vectors for the
        // bucket chunks' own inputs.
        let input_budget = options.input_budget / std::mem::size_of::<T>().max(1);
        let vectors = if options.input_budget == usize::MAX {
            usize::MAX
        } else {
            input_budget / crate::loop_v2::kernels::lane_vector_len(L)
        };
        let chunks_of_buckets =
            bucket_chunks(plan, options.chunk_families, &b_families, vectors, live);
        let read_chunks = ReadChunks::of(plan, &chunks_of_buckets);
        let merged_parts = merged
            .map(|merged| parts(merged, options.part_chunks, plan))
            .unwrap_or_default();
        if options.term_scale && !plan.amplitude {
            return Err(RusticolError::invalid_argument(
                "the term scale is a diagnostic of amplitude-mode plans (loop-induced)",
            ));
        }
        let mut sink = Lanes {
            failures: vec![Vec::new(); points],
            measures: vec![Vec::new(); points],
            families: vec![Vec::new(); points],
            checks: vec![Vec::new(); points],
            terms: vec![Vec::new(); points],
        };
        // The call's buffer pools, kept across its family walks and chunks
        // (lit-dress lever E).
        let workspaces = Workspaces::new(rayon::current_num_threads(), zero);
        for chunk in chunks(points, L) {
            backend.begin();
            let start = Instant::now();
            let data = LaneData::<T, L>::new(plan, batch, &chunk, zero)?;
            let contraction = Instant::now();
            let mut seeds = match fourier {
                Some(fourier) => {
                    report.count(FFT_SEED_JOBS, fourier.jobs() as u64);
                    seeds_fourier(plan, fourier, &data)
                }
                None => seeds(plan, &data),
            };
            // Schedule B forms each key where a leaf reads it; schedule M
            // reads them all.
            if !merged_parts.is_empty() {
                seeds.materialise::<L>(zero);
            }
            // The contraction alone ("seeds" also gathers the lane data).
            report.time(SEED_CONTRACTION, contraction);
            report.time("seeds", start);
            let mut inputs =
                ChunkInputs::new(plan).with_rows(live.map(|live| Arc::clone(&live.read)));
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
            for (k, part) in chunks_of_buckets.iter().enumerate() {
                let numerators = dress_families(
                    plan,
                    part,
                    &mut inputs,
                    &seeds,
                    &data,
                    &mut report,
                    Some((k, &read_chunks, input_budget)),
                    live,
                    &workspaces,
                )?;
                reduction.reduce(&numerators, &mut report)?;
                let mut pool = workspaces
                    .numerators
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                for numerator in numerators {
                    numerator.recycle(&mut pool);
                }
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
            mut measures,
            mut families,
            mut checks,
            mut terms,
        } = sink;
        let zero = zero.zero();
        let pi = zero.pi();
        let factor = Complex::new(
            zero.clone(),
            zero.one() / (zero.from_usize(16) * &pi * &pi * zero.from_usize(plan.average as usize)),
        );
        let mut results = Vec::with_capacity(points);
        let failed = |failures: Vec<String>, measures: Vec<LaneMeasure>| PointResult {
            families: Vec::new(),
            check: None,
            bare: Vec::new(),
            budget: None,
            amplitudes: Vec::new(),
            amplitude_check: None,
            amplitude_budget: None,
            amplitude_scale: Vec::new(),
            failures,
            measures,
        };
        if plan.amplitude {
            // Amplitude-mode plans (decided once per call): the native
            // amplitude sum per (loop order, column).
            for point in 0..points {
                if !failures[point].is_empty() {
                    results.push(failed(
                        std::mem::take(&mut failures[point]),
                        std::mem::take(&mut measures[point]),
                    ));
                    continue;
                }
                let mut lanes = std::mem::take(&mut families[point]);
                lanes.sort_by_key(|(key, _)| *key);
                let tests = options.check.map(|_| {
                    let mut tests = std::mem::take(&mut checks[point]);
                    tests.sort_by_key(|(key, _)| *key);
                    tests
                });
                let terms = options
                    .term_scale
                    .then(|| std::mem::take(&mut terms[point]));
                results.push(amplitude_result(
                    plan,
                    &zero,
                    width,
                    lanes,
                    tests,
                    terms,
                    options.raw_lanes,
                ));
            }
            report.count("points", points as u64);
            return Ok((results, report));
        }
        for point in 0..points {
            if !failures[point].is_empty() {
                results.push(failed(
                    std::mem::take(&mut failures[point]),
                    std::mem::take(&mut measures[point]),
                ));
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
                amplitudes: Vec::new(),
                amplitude_check: None,
                amplitude_budget: None,
                amplitude_scale: Vec::new(),
                failures: Vec::new(),
                measures: Vec::new(),
            });
        }
        report.count("points", points as u64);
        Ok((results, report))
    }

    /// The loop amplitude of an amplitude-mode point (M-LIT LIT-AMP, the
    /// native amplitude sum): per (loop order, column) `i/(16π²) Σ_f Ω_f`
    /// over the point's lanes sorted by key (families in plan order), the
    /// re-test's amplitude and the budget `Σ_f |ΔΩ_f| / (16π²)` when a
    /// check ran, the term scale `Σ_f Σ_α |N_α| |I_α| / (16π²)` when asked
    /// (`terms`, [`V2Options::term_scale`]); the raw lanes kept only when
    /// asked (`raw_lanes`).
    fn amplitude_result<T: Real>(
        plan: &LoopProgramV4,
        zero: &T,
        width: usize,
        lanes: Vec<(LaneKey, Vec<Complex<T>>)>,
        tests: Option<Vec<(LaneKey, Vec<Complex<T>>)>>,
        terms: Option<Vec<Vec<T>>>,
        raw_lanes: bool,
    ) -> PointResult<T> {
        let pi = zero.pi();
        let scale = zero.one() / (zero.from_usize(16) * &pi * &pi);
        let factor = Complex::new(zero.clone(), scale.clone());
        let slots = plan.order_count * plan.columns;
        let slot = |key: &LaneKey| key.order * plan.columns + key.column;
        let sum = |lanes: &[(LaneKey, Vec<Complex<T>>)]| {
            let mut out = vec![vec![Complex::new(zero.clone(), zero.clone()); width]; slots];
            for (key, value) in lanes {
                for (s, z) in out[slot(key)].iter_mut().zip(value) {
                    *s += z;
                }
            }
            for amplitude in &mut out {
                for z in amplitude.iter_mut() {
                    *z = factor.clone() * &*z;
                }
            }
            out
        };
        let amplitudes = sum(&lanes);
        // A point none of whose lanes was reduced has no term (all zero).
        let amplitude_scale = match terms {
            Some(terms) if terms.is_empty() => vec![vec![zero.clone(); width]; slots],
            Some(mut terms) => {
                for z in terms.iter_mut().flatten() {
                    *z = scale.clone() * &*z;
                }
                terms
            }
            None => Vec::new(),
        };
        let (amplitude_check, amplitude_budget) = match &tests {
            Some(tests) => {
                let mut budget = vec![vec![zero.clone(); width]; slots];
                for ((key, direct), (other, test)) in lanes.iter().zip(tests) {
                    debug_assert_eq!(key, other);
                    for (b, d) in budget[slot(key)].iter_mut().zip(deviation(direct, test)) {
                        *b += scale.clone() * &d;
                    }
                }
                (Some(sum(tests)), Some(budget))
            }
            None => (None, None),
        };
        PointResult {
            families: if raw_lanes { lanes } else { Vec::new() },
            check: tests.filter(|_| raw_lanes),
            bare: Vec::new(),
            budget: None,
            amplitudes,
            amplitude_check,
            amplitude_budget,
            amplitude_scale,
            failures: Vec::new(),
            measures: Vec::new(),
        }
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

        /// A family mask (an evaluation-time selection, `V2Executor.
        /// evaluate`'s `families`) dresses and reduces the kept families
        /// only: the dropped family's lanes are absent, the kept one's are
        /// bitwise those of the whole evaluation; a mask of the wrong length
        /// is refused.
        #[test]
        fn a_family_mask_dresses_the_kept_families_only() {
            let text = std::str::from_utf8(PLAN).unwrap().replace("FAMILY", FAMILY);
            let plan = LoopProgramV4::load(text.as_bytes()).unwrap();
            let executor = V2Executor::new(Arc::new(plan), DEFAULT_LANES, Some(1)).unwrap();
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
            let run = |families: Option<&[bool]>| {
                let mut backend = Failing {
                    calls: vec![Vec::new()],
                };
                executor
                    .evaluate_families(
                        &batch,
                        &0.0,
                        &mut backend,
                        V2Options::default(),
                        families,
                        None,
                    )
                    .map(|(results, report)| {
                        let lanes = report.counts.get("reduction lanes").copied();
                        (results, lanes.unwrap_or(0))
                    })
            };
            let (every, lanes) = run(None).unwrap();
            assert_eq!(lanes, 2);
            let (kept, lanes) = run(Some(&[false, true])).unwrap();
            assert_eq!(lanes, 1);
            assert_eq!(kept[0].families.len(), 1);
            assert_eq!(kept[0].families[0], every[0].families[1]);
            let (none, lanes) = run(Some(&[false, false])).unwrap();
            assert_eq!(lanes, 0);
            assert!(none[0].families.is_empty() && none[0].failures.is_empty());
            assert!(run(Some(&[true])).is_err());
        }

        /// In amplitude mode a family's numerator and reduction lanes
        /// ([`numerator_vectors`]) count against the bucket chunks' lane-
        /// vector budget: the two families of [`PLAN`] share their one
        /// factor, so their chunk inputs alone fit one chunk at the first
        /// family's inputs, and a seeded plan keeps that one chunk; an
        /// amplitude-mode plan (the flag the chunker reads) splits them where
        /// the second family's numerator would exceed the budget, and keeps
        /// them together where both fit or without a budget.
        #[test]
        fn amplitude_numerators_count_against_the_chunk_budget() {
            let text = std::str::from_utf8(PLAN).unwrap().replace("FAMILY", FAMILY);
            let mut plan = LoopProgramV4::load(text.as_bytes()).unwrap();
            let families = [0usize, 1];
            let chunks = |plan: &LoopProgramV4, vectors: usize| -> Vec<Vec<usize>> {
                bucket_chunks(plan, usize::MAX, &families, vectors, None)
                    .into_iter()
                    .map(|chunk| chunk.families)
                    .collect()
            };
            let first = &plan.families[0];
            let inputs: usize = [&first.factors, &first.vertices, &first.currents]
                .iter()
                .enumerate()
                .map(|(store, ids)| {
                    ids.iter()
                        .map(|&id| input_vectors(&plan, store, id))
                        .sum::<usize>()
                })
                .sum();
            assert!(inputs > 0);
            assert!(!plan.amplitude);
            assert_eq!(chunks(&plan, inputs), [vec![0, 1]]);
            assert_eq!(chunks(&plan, usize::MAX), [vec![0, 1]]);
            plan.amplitude = true;
            let numerator = numerator_vectors(&plan, &plan.families[0]);
            // Rank 0, two propagators, one order and one column: the
            // coefficient twice and 4N + N + 1 + N² kinematics.
            let coefficients = ninjars::fast::tensor_layout::ncoeff(0);
            assert_eq!(numerator, 2 * coefficients + 5 * 2 + 1 + 2 * 2);
            assert_eq!(numerator, numerator_vectors(&plan, &plan.families[1]));
            assert_eq!(chunks(&plan, inputs), [vec![0], vec![1]]);
            assert_eq!(chunks(&plan, inputs + numerator), [vec![0], vec![1]]);
            assert_eq!(chunks(&plan, inputs + 2 * numerator), [vec![0, 1]]);
            assert_eq!(chunks(&plan, usize::MAX), [vec![0, 1]]);
        }

        /// The next bucket chunk reading an input follows the chunks'
        /// ascending reads; none after the last.
        #[test]
        fn next_reads_follow_the_bucket_chunks() {
            let reads = ReadChunks([
                (vec![0, 3, 3], vec![0, 2, 5]),
                (vec![0], Vec::new()),
                (vec![0, 1], vec![1]),
            ]);
            assert!(reads.live(0, 0, 5) && !reads.live(0, 0, 6) && !reads.live(0, 1, 0));
            assert_eq!(reads.next(0, 0, 0), 2);
            assert_eq!(reads.next(0, 0, 1), 2);
            assert_eq!(reads.next(0, 0, 2), 5);
            assert_eq!(reads.next(0, 0, 5), usize::MAX);
            assert_eq!(reads.next(0, 1, 0), usize::MAX);
            assert_eq!(reads.next(2, 0, 0), 1);
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
