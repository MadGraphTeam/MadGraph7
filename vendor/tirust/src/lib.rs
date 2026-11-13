//! A dimensioned, ownership-preserving C boundary for MadLoop's Fortran adapter.
//! Only this crate handles foreign pointers. TIRust's safe numerical API remains
//! separate. Full and truncated results have independent stores/workspaces.
use iregi_rust::{
    Convention, DirectReductionPlan, DirectWorkspace, EmbedderOptions, InvariantInput,
    KinematicsBatch, LaneRoute, LaneStatus, ReductionSettings, ScaleInput, TensorEngineExecution, TensorReductionPlan, TensorStore,
    TensorStoreOptions, TensorWeights, TensorWorkspace,
};
use std::{
    cell::RefCell,
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind},
    slice,
    sync::Mutex,
};
use symbolica::domains::float::{Complex, Float, FloatLike};

#[cfg(feature = "poles-only")]
use iregi_rust::{DirectPoleWorkspace, PoleTensorStore};

pub const ABI_VERSION: u32 = 1;
const SUCCESS: i32 = 0;
const UNSTABLE: i32 = 1;
const ERROR: i32 = 2;
const UNSUPPORTED: i32 = 3;
const RECORD: usize = 64;
const MP_BITS: u32 = 256;
const DEFAULT_BUDGET: u64 = 268_435_456;
const DP_ROUTE_FIELDS: usize = 10;
const DP_ROUTE_STATS_COUNT: usize = 2 * DP_ROUTE_FIELDS;
type Key = (usize, u16);
type Result<T> = std::result::Result<T, String>;

/// GNU/Linux Fortran COMPLEX(C_DOUBLE_COMPLEX) memory representation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

thread_local! { static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) }; }

struct State {
    budget: u64,
    plans: HashMap<Key, DirectReductionPlan>,
    dp_plans: HashMap<Key, TensorReductionPlan>,
    dp: HashMap<(Key, bool), TensorWorkspace<f64>>,
    mp: HashMap<Key, DirectWorkspace<Float>>,
    store: TensorStore,
    /// Subintegral store of MadLoop's independent direction checks. It never
    /// shares an entry with the primary store, so a check recomputes every
    /// integral it compares; entries are exact pure functions of their keys,
    /// so check values are bit-identical to an unshared evaluation.
    check_store: TensorStore,
    dp_cache: DpCache,
    qp_cache: QpCache,
    dp_routes: [u64; DP_ROUTE_STATS_COUNT],
    last_dp_route: Option<usize>,
    #[cfg(feature = "poles-only")]
    pole_dp: HashMap<Key, DirectPoleWorkspace<f64>>,
    #[cfg(feature = "poles-only")]
    pole_mp: HashMap<Key, DirectPoleWorkspace<Float>>,
    #[cfg(feature = "poles-only")]
    pole_store: PoleTensorStore,
}

impl State {
    fn new() -> Self {
        Self {
            budget: DEFAULT_BUDGET,
            plans: HashMap::new(),
            dp_plans: HashMap::new(),
            dp: HashMap::new(),
            mp: HashMap::new(),
            qp_cache: QpCache::new(DEFAULT_BUDGET / 2),
            dp_routes: [0; DP_ROUTE_STATS_COUNT],
            last_dp_route: None,
            store: TensorStore::with_options(TensorStoreOptions {
                byte_budget: Some(budgets(DEFAULT_BUDGET).primary),
                ..Default::default()
            }),
            check_store: TensorStore::with_options(TensorStoreOptions {
                byte_budget: Some(budgets(DEFAULT_BUDGET).check),
                ..Default::default()
            }),
            dp_cache: DpCache::new(budgets(DEFAULT_BUDGET).results),
            #[cfg(feature = "poles-only")]
            pole_dp: HashMap::new(),
            #[cfg(feature = "poles-only")]
            pole_mp: HashMap::new(),
            #[cfg(feature = "poles-only")]
            pole_store: PoleTensorStore::new(Some(budgets(DEFAULT_BUDGET).poles)),
        }
    }

    fn register(&mut self, key: Key) -> Result<()> {
        if !self.plans.contains_key(&key) {
            let plan = DirectReductionPlan::new(settings(key)).map_err(|e| e.to_string())?;
            self.store.register(&plan);
            self.check_store.register(&plan);
            #[cfg(feature = "poles-only")]
            self.pole_store.register(&plan);
            self.plans.insert(key, plan);
        }
        Ok(())
    }

    fn dp(&mut self, key: Key, batch: KinematicsBatch<f64>, mode: Mode) -> Result<(i32, Vec<C64>)> {
        self.last_dp_route = None;
        self.register(key)?;
        if !mode.finite {
            #[cfg(feature = "poles-only")]
            return self.dp_poles(key, batch, mode);
            #[cfg(not(feature = "poles-only"))]
            return Ok((UNSUPPORTED, Vec::new()));
        }
        // Keep each direction's structural variants and scratch buffers warm.
        // Reusing one workspace makes alternating lightlike configurations
        // repeatedly discard and rediscover their active geometry variants.
        let workspace_key = (key, mode.check);
        if !self.dp.contains_key(&workspace_key) {
            let plan = match self.dp_plans.entry(key) {
                std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::hash_map::Entry::Vacant(entry) => entry
                    .insert(TensorReductionPlan::new(settings(key)).map_err(|e| e.to_string())?),
            };
            let workspace = TensorWorkspace::new_embedded(
                plan, 0.0f64,
                // A lane starts at the lowest tier whose own gate admits its
                // binary64 estimate (no skipping margin): each tier's gate
                // bounds the error amplification, so near-limit lanes get
                // double-double instead of 256-bit evaluations. The double-double
                // tier accepts conditioning down to 2^-80 (26 bits of headroom):
                // on MadLoop's Zggg its results round to the 256-bit binary64
                // values, at a fifth of their cost.
                EmbedderOptions { precision_tiers: true, tier_margin_bits: 0, double_double_headroom_bits: 26, ..EmbedderOptions::default() },
            ).map_err(|e| e.to_string())?;
            self.dp.insert(workspace_key, workspace);
        }
        // check_view() may alias the primary when no master provider exists.
        // MadLoop's independent direction checks must recompute every entry
        // of the primary pass: they share subintegrals only with each other.
        let store = mode.cache.then(|| if mode.check { self.check_store.clone() } else { self.store.clone() });
        let workspace = self.dp.get_mut(&workspace_key).unwrap();
        workspace.set_tensor_store(store.as_ref());
        let result = workspace.reduce_batch(&batch).map_err(|e| e.to_string())?;
        self.last_dp_route = workspace.last_lane_routes().and_then(|routes| routes.first()).map(|route| {
            match route {
                LaneRoute::Direct => 0,
                LaneRoute::MethodChoice => 1,
                LaneRoute::Tier(106) => 2,
                LaneRoute::Tier(256) => 3,
                LaneRoute::Tier(512) => 4,
                LaneRoute::Refused => 5,
                LaneRoute::Tier(_) => 6,
            }
        });
        // Attempts include work whose rounded DP output is later rejected.
        if let TensorEngineExecution::Direct(execution) = workspace.engine_execution() {
            let offset = usize::from(mode.check) * DP_ROUTE_FIELDS;
            for tier in execution.precision_rescue {
                let index = match tier.bits { 106 => 7, 256 => 8, 512 => 9, _ => continue };
                self.dp_routes[offset + index] += tier.rows as u64;
            }
        }
        match result.lane(0) {
            Ok(values) => Ok((SUCCESS, pack_dp(values, mode.poles, 4))),
            Err(status) => {
                set_error(&format!("TIRust DP: {status:?}"));
                Ok((lane_status(status), Vec::new()))
            }
        }
    }

    fn dp_checked(
        &mut self,
        key: Key,
        batch: KinematicsBatch<f64>,
        mode: Mode,
    ) -> Result<(i32, Vec<C64>)> {
        // MadLoop requests the same tensor integral once per helicity while it
        // learns its filters. A primary result is a pure function of its exact
        // inputs (bit-identical with or without store sharing), so an exact
        // repeat within the phase-space point returns the stored tensor.
        let cache_key = (mode.finite && mode.cache && !mode.check).then(|| dp_cache_key(key, &batch, mode.poles));
        if let Some(cache_key) = &cache_key {
            if let Some(values) = self.dp_cache.entries.get(cache_key) {
                self.dp_cache.hits += 1;
                self.last_dp_route = None;
                return Ok((SUCCESS, values.to_vec()));
            }
            self.dp_cache.misses += 1;
        }
        let mut result = self.dp(key, batch, mode);
        if let Ok((SUCCESS, values)) = &result {
            if values
                .iter()
                .any(|z| !z.re.is_finite() || !z.im.is_finite())
            {
                set_error("nonfinite DP tensor result");
                result = Ok((UNSTABLE, Vec::new()));
            }
        }
        let failed = !matches!(&result, Ok((SUCCESS, _)));
        if mode.finite {
            let offset = usize::from(mode.check) * DP_ROUTE_FIELDS;
            let final_route = match &result {
                Ok((SUCCESS, _)) => self.last_dp_route.unwrap_or(6),
                Ok(_) => 5,
                Err(_) => 6,
            };
            self.dp_routes[offset + final_route] += 1;
            // Store entries are exact pure functions of their keys (method,
            // settings and input bits), so entries computed on any route stay
            // valid for every later integral. Only a failure discards them.
            if failed {
                self.store.clear();
                self.check_store.clear();
                self.dp_cache.clear();
            } else if let (Some(cache_key), Ok((SUCCESS, values))) = (cache_key, &result) {
                self.dp_cache.insert(cache_key, values);
            }
            // Retain structural preparation after ordinary numerical refusals.
            if failed && !matches!(&result, Ok((UNSTABLE, _))) {
                self.dp.remove(&(key, false));
                self.dp.remove(&(key, true));
            }
        } else if failed {
            #[cfg(feature = "poles-only")]
            {
                self.pole_store.clear();
                self.pole_dp.remove(&key);
            }
        }
        result
    }

    fn mp(
        &mut self,
        key: Key,
        batch: KinematicsBatch<Float>,
        mode: Mode,
    ) -> Result<(i32, Vec<u8>)> {
        self.register(key)?;
        if !mode.finite {
            #[cfg(feature = "poles-only")]
            return self.mp_poles(key, batch, mode);
            #[cfg(not(feature = "poles-only"))]
            return Ok((UNSUPPORTED, Vec::new()));
        }
        if !self.mp.contains_key(&key) {
            let mut workspace = self.plans[&key]
                .workspace(Float::new(MP_BITS))
                .map_err(|e| e.to_string())?;
            workspace.set_input_precision(113);
            // 256-bit arithmetic with substantial headroom above binary128.
            // The external MadLoop check still decides numerical acceptance.
            workspace.set_stability_threshold(Float::with_val(MP_BITS, 2).pow(136).inv());
            self.mp.insert(key, workspace);
        }
        // MPFR store reuse is not supported by TIRust; do not claim DP hits as QP reuse.
        let workspace = self.mp.get_mut(&key).unwrap();
        let result = workspace.reduce_batch(&batch).map_err(|e| e.to_string())?;
        match result.lane(0) {
            Ok(values) => Ok((SUCCESS, pack_mp(values, mode.poles, 4)?)),
            Err(status) => {
                set_error(&format!("TIRust QP: {status:?}"));
                Ok((lane_status(status), Vec::new()))
            }
        }
    }

    fn mp_records(&mut self, key: Key, records: [&[u8]; 4], mode: Mode) -> Result<(i32, Vec<u8>)> {
        let cache_key = if mode.cache && !mode.check {
            let key_bytes = qp_cache_key(key, records, mode);
            if let Some(values) = self.qp_cache.entries.get(&key_bytes) {
                self.qp_cache.hits += 1;
                return Ok((SUCCESS, values.to_vec()));
            }
            self.qp_cache.misses += 1;
            Some(key_bytes)
        } else {
            self.qp_cache.bypassed += 1;
            None
        };
        let batch = mp_batch(key.0, records[0], records[1], records[2], records[3])?;
        let (status, values) = self.mp(key, batch, mode)?;
        if status == SUCCESS {
            if let Some(cache_key) = cache_key {
                self.qp_cache.insert(cache_key, &values);
            }
        }
        Ok((status, values))
    }

    #[cfg(feature = "poles-only")]
    fn dp_poles(
        &mut self,
        key: Key,
        batch: KinematicsBatch<f64>,
        mode: Mode,
    ) -> Result<(i32, Vec<C64>)> {
        if !self.pole_dp.contains_key(&key) {
            self.pole_dp.insert(
                key,
                self.plans[&key]
                    .pole_workspace(0.0f64)
                    .map_err(|e| e.to_string())?,
            );
        }
        let workspace = self.pole_dp.get_mut(&key).unwrap();
        // Independent check passes must not read primary cached values.
        workspace.set_tensor_store((mode.cache && !mode.check).then_some(&self.pole_store));
        let result = workspace.reduce_batch(&batch).map_err(|e| e.to_string())?;
        match result.lane(0) {
            Ok(values) => Ok((SUCCESS, pack_dp(values, mode.poles, 3))),
            Err(status) => {
                set_error(&format!("TIRust poles DP: {status:?}"));
                Ok((lane_status(status), Vec::new()))
            }
        }
    }

    #[cfg(feature = "poles-only")]
    fn mp_poles(
        &mut self,
        key: Key,
        batch: KinematicsBatch<Float>,
        mode: Mode,
    ) -> Result<(i32, Vec<u8>)> {
        if !self.pole_mp.contains_key(&key) {
            let mut workspace = self.plans[&key]
                .pole_workspace(Float::new(MP_BITS))
                .map_err(|e| e.to_string())?;
            workspace.set_input_precision(113);
            workspace.set_stability_threshold(Float::with_val(MP_BITS, 2).pow(136).inv());
            self.pole_mp.insert(key, workspace);
        }
        let result = self
            .pole_mp
            .get_mut(&key)
            .unwrap()
            .reduce_batch(&batch)
            .map_err(|e| e.to_string())?;
        match result.lane(0) {
            Ok(values) => Ok((SUCCESS, pack_mp(values, mode.poles, 3)?)),
            Err(status) => {
                set_error(&format!("TIRust poles QP: {status:?}"));
                Ok((lane_status(status), Vec::new()))
            }
        }
    }
}

/// Byte budgets of the numerical stores for a total budget. Half belongs to
/// QP tensor results ([`QpCache`]); of the binary64 half, the primary
/// subintegral store keeps a quarter of the total and the rest goes to the
/// pole-only store (when built), the check-pass store and the exact-result
/// cache.
struct Budgets {
    primary: u64,
    check: u64,
    results: u64,
    #[cfg_attr(not(feature = "poles-only"), allow(dead_code))]
    poles: u64,
}
fn budgets(total: u64) -> Budgets {
    if cfg!(feature = "poles-only") {
        Budgets { primary: total / 4, poles: total / 8, check: total / 16, results: total / 16 }
    } else {
        Budgets { primary: total / 4, poles: 0, check: total / 8, results: total / 8 }
    }
}

/// Exact full-tensor QP results, including the original decimal records.
/// This is separate from upstream's binary64 subintegral cache. Full and
/// truncated tensors have different keys, as do every pole selection and
/// precision. Failed and independent-check results are never retained.
struct QpCache {
    entries: HashMap<Vec<u8>, Box<[u8]>>,
    budget: u64,
    bytes: u64,
    hits: u64,
    misses: u64,
    not_stored: u64,
    bypassed: u64,
}
impl QpCache {
    fn new(budget: u64) -> Self {
        Self {
            entries: HashMap::new(),
            budget,
            bytes: 0,
            hits: 0,
            misses: 0,
            not_stored: 0,
            bypassed: 0,
        }
    }
    fn clear(&mut self) {
        *self = Self::new(self.budget);
    }
    fn insert(&mut self, key: Vec<u8>, values: &[u8]) {
        // Account for owned payloads plus a conservative allowance for the
        // hash table and entry metadata. No eviction makes timing predictable.
        let cost = (key.capacity() + values.len() + 256) as u64;
        if cost > self.budget.saturating_sub(self.bytes) {
            self.not_stored += 1;
            return;
        }
        self.entries.insert(key, values.into());
        self.bytes += cost;
    }
    fn statistics(&self) -> [u64; 6] {
        [
            self.hits,
            self.misses,
            self.entries.len() as u64,
            self.bytes,
            self.not_stored,
            self.bypassed,
        ]
    }
}
/// Exact-input full-tensor binary64 results of the current phase-space point
/// (primary passes only). Keys hold the bits of every input value, the
/// dimensions and the pole selection of the packed output.
struct DpCache {
    entries: HashMap<Box<[u64]>, Box<[C64]>>,
    budget: u64,
    bytes: u64,
    hits: u64,
    misses: u64,
    not_stored: u64,
}
impl DpCache {
    fn new(budget: u64) -> Self {
        Self { entries: HashMap::new(), budget, bytes: 0, hits: 0, misses: 0, not_stored: 0 }
    }
    fn clear(&mut self) {
        *self = Self::new(self.budget);
    }
    fn insert(&mut self, key: Box<[u64]>, values: &[C64]) {
        let cost = (8 * key.len() + std::mem::size_of_val(values) + 64) as u64;
        if cost > self.budget.saturating_sub(self.bytes) {
            self.not_stored += 1;
            return;
        }
        self.entries.insert(key, values.into());
        self.bytes += cost;
    }
}
fn dp_cache_key(key: Key, batch: &KinematicsBatch<f64>, poles: i32) -> Box<[u64]> {
    let mut words = Vec::with_capacity(1 + 2 * batch.values.len());
    words.push((key.0 as u64) | (u64::from(key.1) << 8) | ((poles as u64) << 24));
    for z in &batch.values {
        words.push(z.re.to_bits());
        words.push(z.im.to_bits());
    }
    words.into()
}
fn qp_cache_key(key: Key, records: [&[u8]; 4], mode: Mode) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16 + records.iter().map(|x| x.len()).sum::<usize>());
    bytes.extend_from_slice(&(key.0 as u32).to_le_bytes());
    bytes.extend_from_slice(&key.1.to_le_bytes());
    bytes.push(mode.poles as u8);
    bytes.push(u8::from(mode.finite));
    bytes.extend_from_slice(&113u32.to_le_bytes());
    bytes.extend_from_slice(&MP_BITS.to_le_bytes());
    for record in records {
        bytes.extend_from_slice(record);
    }
    bytes
}

/// Opaque owner; callers destroy it after all calls have returned.
pub struct Context {
    state: Mutex<State>,
}

fn settings((n, rank): Key) -> ReductionSettings {
    let mut s = ReductionSettings::new(n, rank);
    s.convention = Convention::MadLoop;
    s.tensor_weights = TensorWeights::Components;
    s.invariants = InvariantInput::SuppliedInvariants;
    s.scale_input = ScaleInput::MuSquared;
    s
}
fn dimensions(n: i32, rank: i32) -> Result<(Key, usize)> {
    if !(1..=6).contains(&n) || !(0..=6).contains(&rank) {
        return Err("qualified TIRust bridge domain is 1 <= N <= 6 and 0 <= rank <= 6".into());
    }
    let r = rank as usize;
    Ok((
        (n as usize, rank as u16),
        (r + 1) * (r + 2) * (r + 3) * (r + 4) / 24,
    ))
}
#[derive(Clone, Copy)]
struct Mode {
    poles: i32,
    finite: bool,
    check: bool,
    cache: bool,
}
impl Mode {
    fn new(poles: i32, finite: i32, check: i32, cache: i32) -> Result<Self> {
        if !(0..=3).contains(&poles) || [finite, check, cache].iter().any(|&v| v != 0 && v != 1) {
            return Err("invalid TIRust pole mode or logical flag".into());
        }
        if finite == 0 && poles == 3 {
            return Err("pole-only evaluation requires at least one pole origin".into());
        }
        Ok(Self {
            poles,
            finite: finite != 0,
            check: check != 0,
            cache: cache != 0,
        })
    }
}
fn selected(slot: usize, poles: i32) -> bool {
    match slot {
        0 => poles == 0 || poles == 1,
        1 | 2 => poles == 0 || poles == 2,
        _ => true,
    }
}
fn lane_status(status: &LaneStatus) -> i32 {
    match status {
        LaneStatus::Success => SUCCESS,
        LaneStatus::Unstable(_) | LaneStatus::MasterFailure(_) => UNSTABLE,
        LaneStatus::InvalidInput(_) => ERROR,
    }
}
fn set_error(message: &str) {
    LAST_ERROR.with(|e| *e.borrow_mut() = message.into());
}
fn guard(f: impl FnOnce() -> Result<i32>) -> i32 {
    set_error("");
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(status)) => status,
        Ok(Err(error)) => {
            set_error(&error);
            ERROR
        }
        Err(_) => {
            set_error("TIRust panic contained at the C boundary");
            ERROR
        }
    }
}
// Successful diagnostic queries must not erase the reduction error that a
// caller may still need. Keep ordinary guard semantics on a failed query.
fn query_guard(f: impl FnOnce() -> Result<i32>) -> i32 {
    match catch_unwind(AssertUnwindSafe(|| {
        let previous = LAST_ERROR.with(|error| error.borrow().clone());
        let status = guard(f);
        if status == SUCCESS {
            LAST_ERROR.with(|error| *error.borrow_mut() = previous);
        }
        status
    })) {
        Ok(status) => status,
        Err(_) => {
            set_error("TIRust panic contained at the C boundary");
            ERROR
        }
    }
}
unsafe fn context<'a>(ptr: *mut Context) -> Result<&'a Context> {
    if ptr.is_null() || !ptr.is_aligned() {
        return Err("invalid TIRust context".into());
    }
    Ok(unsafe { &*ptr })
}
fn length(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual != expected {
        return Err(format!("{name} length {actual}, expected {expected}"));
    }
    Ok(())
}
unsafe fn read<'a, T>(ptr: *const T, count: usize) -> Result<&'a [T]> {
    if ptr.is_null() || !ptr.is_aligned() {
        return Err("null or misaligned input buffer".into());
    }
    Ok(unsafe { slice::from_raw_parts(ptr, count) })
}
fn validate_output<T>(ptr: *mut T) -> Result<()> {
    if ptr.is_null() || !ptr.is_aligned() {
        Err("null or misaligned output buffer".into())
    } else {
        Ok(())
    }
}
unsafe fn write<T: Copy>(ptr: *mut T, values: &[T]) {
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), ptr, values.len());
    }
}

fn dp_batch(n: usize, p: &[f64], s: &[f64], m: &[C64], mu2: f64) -> Result<KinematicsBatch<f64>> {
    if !mu2.is_finite()
        || mu2 <= 0.0
        || p.iter().chain(s).any(|v| !v.is_finite())
        || m.iter()
            .any(|v| !v.re.is_finite() || !v.im.is_finite() || v.im > 0.0)
    {
        return Err("nonfinite kinematics, nonpositive mu squared, or positive mass width".into());
    }
    let mut values: Vec<_> = p.iter().map(|&x| Complex::new(x, 0.0)).collect();
    for i in 0..n {
        for j in i + 1..n {
            if s[i + n * j] != s[j + n * i] {
                return Err("supplied invariant matrix must be symmetric".into());
            }
            values.push(Complex::new(s[i + n * j], 0.0));
        }
    }
    values.extend(m.iter().map(|z| Complex::new(z.re, z.im)));
    values.push(Complex::new(mu2, 0.0));
    Ok(KinematicsBatch {
        rows: 1,
        components: 1,
        values,
    })
}
fn pack_dp(values: &[Complex<f64>], poles: i32, slots: usize) -> Vec<C64> {
    let nc = values.len() / slots;
    let mut out = vec![C64::default(); nc * 4];
    for slot in 0..slots {
        if selected(slot, poles) {
            for m in 0..nc {
                let z = &values[m * slots + slot];
                out[slot * nc + m] = C64 { re: z.re, im: z.im };
            }
        }
    }
    out
}
fn parse_records(data: &[u8]) -> Result<Vec<Float>> {
    data.chunks_exact(RECORD)
        .map(|record| {
            let text = std::str::from_utf8(record)
                .map_err(|_| "QP input is not ASCII")?
                .trim();
            let mut value =
                Float::parse(text, Some(113)).map_err(|e| format!("invalid QP value: {e}"))?;
            if !value.is_finite() {
                return Err("nonfinite QP input".into());
            }
            value.set_prec(MP_BITS); // preserves signed zero and exact binary128 value
            Ok(value)
        })
        .collect()
}
fn mp_batch(
    n: usize,
    p: &[u8],
    s: &[u8],
    m: &[u8],
    scale: &[u8],
) -> Result<KinematicsBatch<Float>> {
    let p = parse_records(p)?;
    let s = parse_records(s)?;
    let m = parse_records(m)?;
    let scale = parse_records(scale)?;
    let z = || Float::new(MP_BITS);
    if scale[0] <= z() || m.chunks_exact(2).any(|v| v[1] > z()) {
        return Err("invalid QP scale or mass width".into());
    }
    let mut values: Vec<_> = p.into_iter().map(|v| Complex::new(v, z())).collect();
    for i in 0..n {
        for j in i + 1..n {
            if s[i + n * j] != s[j + n * i] {
                return Err("supplied QP invariant matrix must be symmetric".into());
            }
            values.push(Complex::new(s[i + n * j].clone(), z()));
        }
    }
    values.extend(
        m.chunks_exact(2)
            .map(|v| Complex::new(v[0].clone(), v[1].clone())),
    );
    values.push(Complex::new(scale[0].clone(), z()));
    Ok(KinematicsBatch {
        rows: 1,
        components: 1,
        values,
    })
}
fn pack_mp(values: &[Complex<Float>], poles: i32, slots: usize) -> Result<Vec<u8>> {
    let nc = values.len() / slots;
    let mut out = vec![b' '; 8 * nc * RECORD];
    let zero = Float::new(MP_BITS);
    for slot in 0..4 {
        for m in 0..nc {
            for component in 0..2 {
                let value = if slot < slots && selected(slot, poles) {
                    if component == 0 {
                        &values[m * slots + slot].re
                    } else {
                        &values[m * slots + slot].im
                    }
                } else {
                    &zero
                };
                if !value.is_finite() {
                    return Err("nonfinite QP result".into());
                }
                let text = format!("{value:.39e}");
                if text.len() > RECORD {
                    return Err("QP result exceeds record width".into());
                }
                let offset = (2 * (slot * nc + m) + component) * RECORD;
                out[offset..offset + text.len()].copy_from_slice(text.as_bytes());
            }
        }
    }
    Ok(out)
}

#[unsafe(no_mangle)]
pub extern "C" fn ml_tirust_abi_version() -> u32 {
    ABI_VERSION
}

/// # Safety
/// `out` is a valid writable pointer slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_context_create(version: u32, out: *mut *mut Context) -> i32 {
    guard(|| {
        validate_output(out)?;
        if version != ABI_VERSION {
            return Err(format!("unsupported TIRust ABI {version}"));
        }
        unsafe {
            out.write(Box::into_raw(Box::new(Context {
                state: Mutex::new(State::new()),
            })));
        }
        Ok(SUCCESS)
    })
}
/// # Safety
/// NULL or a live context owned by the caller, with no concurrent calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_context_destroy(handle: *mut Context) -> i32 {
    guard(|| {
        if !handle.is_null() {
            unsafe {
                context(handle)?;
                drop(Box::from_raw(handle));
            }
        }
        Ok(SUCCESS)
    })
}
#[unsafe(no_mangle)]
pub extern "C" fn ml_tirust_supports(n: i32, rank: i32, compute_finite: i32) -> i32 {
    if dimensions(n, rank).is_err()
        || !(0..=1).contains(&compute_finite)
        || (compute_finite == 0 && !cfg!(feature = "poles-only"))
    {
        UNSUPPORTED
    } else {
        SUCCESS
    }
}
/// # Safety
/// `handle` is a live context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_register_plan(handle: *mut Context, n: i32, rank: i32) -> i32 {
    guard(|| {
        let (key, _) = dimensions(n, rank)?;
        unsafe { context(handle)? }
            .state
            .lock()
            .map_err(|_| "TIRust context poisoned")?
            .register(key)?;
        Ok(SUCCESS)
    })
}
/// # Safety
/// `handle` is a live context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_clear_cache(handle: *mut Context) -> i32 {
    guard(|| {
        let mut state = unsafe { context(handle)? }
            .state
            .lock()
            .map_err(|_| "TIRust context poisoned")?;
        state.store.clear();
        state.check_store.clear();
        state.dp_cache.clear();
        state.qp_cache.clear();
        #[cfg(feature = "poles-only")]
        state.pole_store.clear();
        Ok(SUCCESS)
    })
}
/// # Safety
/// `handle` is a live context.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_set_cache_budget(handle: *mut Context, bytes: i64) -> i32 {
    guard(|| {
        if bytes < 0 {
            return Err("cache budget must be nonnegative".into());
        }
        let mut state = unsafe { context(handle)? }
            .state
            .lock()
            .map_err(|_| "TIRust context poisoned")?;
        let bytes = bytes as u64;
        if state.budget != bytes {
            let budget = budgets(bytes);
            state.store.clear();
            state.store.set_byte_budget(Some(budget.primary));
            state.check_store.clear();
            state.check_store.set_byte_budget(Some(budget.check));
            state.dp_cache = DpCache::new(budget.results);
            state.qp_cache = QpCache::new(bytes / 2);
            #[cfg(feature = "poles-only")]
            {
                state.pole_store.clear();
                state.pole_store.set_byte_budget(Some(budget.poles));
            }
            state.budget = bytes;
        }
        Ok(SUCCESS)
    })
}
/// # Safety
/// `handle` is live and output has exactly nine writable u64 entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_cache_statistics(
    handle: *mut Context,
    out: *mut u64,
    count: usize,
) -> i32 {
    guard(|| {
        length("cache statistics", count, 9)?;
        validate_output(out)?;
        let state = unsafe { context(handle)? }
            .state
            .lock()
            .map_err(|_| "TIRust context poisoned")?;
        let s = state.store.statistics();
        #[allow(unused_mut)] // The optional pole store adds to these counters.
        let mut counts = [
            s.subsets_reused,
            s.subsets_computed,
            s.masters_reused,
            s.masters_computed,
            s.moments_reused,
            s.moments_computed,
            s.entries,
            s.bytes,
            s.not_stored,
        ];
        let c = state.check_store.statistics();
        for (a, b) in counts.iter_mut().zip([
            c.subsets_reused, c.subsets_computed, c.masters_reused, c.masters_computed,
            c.moments_reused, c.moments_computed, c.entries, c.bytes, c.not_stored,
        ]) {
            *a += b;
        }
        counts[4] += state.qp_cache.hits + state.dp_cache.hits;
        counts[5] += state.qp_cache.misses + state.dp_cache.misses;
        counts[6] += (state.qp_cache.entries.len() + state.dp_cache.entries.len()) as u64;
        counts[7] += state.qp_cache.bytes + state.dp_cache.bytes;
        counts[8] += state.qp_cache.not_stored + state.dp_cache.not_stored;
        #[cfg(feature = "poles-only")]
        {
            let p = state.pole_store.statistics();
            let pc = [
                p.subsets_reused,
                p.subsets_computed,
                p.masters_reused,
                p.masters_computed,
                p.moments_reused,
                p.moments_computed,
                p.entries,
                p.bytes,
                p.not_stored,
            ];
            for (a, b) in counts.iter_mut().zip(pc) {
                *a += b;
            }
        }
        unsafe {
            write(out, &counts);
        }
        Ok(SUCCESS)
    })
}

/// # Safety
/// `handle` is live and output has exactly six writable u64 entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_qp_cache_statistics(
    handle: *mut Context,
    out: *mut u64,
    count: usize,
) -> i32 {
    guard(|| {
        length("QP cache statistics", count, 6)?;
        validate_output(out)?;
        let state = unsafe { context(handle)? }
            .state
            .lock()
            .map_err(|_| "TIRust context poisoned")?;
        unsafe {
            write(out, &state.qp_cache.statistics());
        }
        Ok(SUCCESS)
    })
}

/// Cumulative full-DP final routes and actual tier attempts, primary then check.
/// Each ten-counter block: Direct, MethodChoice, Tier106, Tier256, Tier512,
/// Refused, unreported/reduction-error, attempt106, attempt256, attempt512.
/// A finite SUCCESS is required for an accepted route. Tensor rescue with
/// DP inputs is distinct from the genuine binary128-input evaluation path.
/// Cache clears do not reset these context-lifetime diagnostic counters.
/// # Safety
/// Live handle and exactly twenty writable u64 entries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_dp_route_statistics(
    handle: *mut Context, out: *mut u64, count: usize,
) -> i32 {
    query_guard(|| {
        length("DP route statistics", count, DP_ROUTE_STATS_COUNT)?;
        validate_output(out)?;
        let state = unsafe { context(handle)? }
            .state.lock().map_err(|_| "TIRust context poisoned")?;
        unsafe { write(out, &state.dp_routes); }
        Ok(SUCCESS)
    })
}

/// Evaluate binary64 tensor moments. Inputs use Fortran column-major matrices.
/// # Safety
/// Live context and aligned buffers of the stated lengths; output does not alias inputs.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_evaluate_dp(
    handle: *mut Context,
    n: i32,
    rank: i32,
    p: *const f64,
    p_len: usize,
    s: *const f64,
    s_len: usize,
    m: *const C64,
    m_len: usize,
    mu2: f64,
    out: *mut C64,
    out_len: usize,
    poles: i32,
    finite: i32,
    check: i32,
    cache: i32,
) -> i32 {
    guard(|| {
        let (key, nc) = dimensions(n, rank)?;
        let n = key.0;
        let mode = Mode::new(poles, finite, check, cache)?;
        if ml_tirust_supports(n as i32, rank, finite) != SUCCESS {
            set_error("pole-only build support is not installed");
            return Ok(UNSUPPORTED);
        }
        length("shifts", p_len, 4 * n)?;
        length("invariants", s_len, n * n)?;
        length("masses", m_len, n)?;
        length("tensor output", out_len, 4 * nc)?;
        validate_output(out)?;
        let batch = dp_batch(
            n,
            unsafe { read(p, p_len)? },
            unsafe { read(s, s_len)? },
            unsafe { read(m, m_len)? },
            mu2,
        )?;
        let (status, values) = unsafe { context(handle)? }
            .state
            .lock()
            .map_err(|_| "TIRust context poisoned")?
            .dp_checked(key, batch, mode)?;
        if status == SUCCESS {
            unsafe {
                write(out, &values);
            }
        }
        Ok(status)
    })
}

/// Evaluate genuine binary128 inputs transported in 64-byte ASCII records.
/// # Safety
/// Live context and buffers of stated lengths; output does not alias input records.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_evaluate_mp(
    handle: *mut Context,
    n: i32,
    rank: i32,
    p: *const u8,
    p_len: usize,
    s: *const u8,
    s_len: usize,
    m: *const u8,
    m_len: usize,
    scale: *const u8,
    scale_len: usize,
    out: *mut u8,
    out_len: usize,
    poles: i32,
    finite: i32,
    check: i32,
    cache: i32,
) -> i32 {
    guard(|| {
        let (key, nc) = dimensions(n, rank)?;
        let n = key.0;
        let mode = Mode::new(poles, finite, check, cache)?;
        if ml_tirust_supports(n as i32, rank, finite) != SUCCESS {
            set_error("pole-only build support is not installed");
            return Ok(UNSUPPORTED);
        }
        length("QP shifts", p_len, 4 * n * RECORD)?;
        length("QP invariants", s_len, n * n * RECORD)?;
        length("QP masses", m_len, 2 * n * RECORD)?;
        length("QP scale", scale_len, RECORD)?;
        length("QP output", out_len, 8 * nc * RECORD)?;
        validate_output(out)?;
        let records = [
            unsafe { read(p, p_len)? },
            unsafe { read(s, s_len)? },
            unsafe { read(m, m_len)? },
            unsafe { read(scale, scale_len)? },
        ];
        let (status, values) = unsafe { context(handle)? }
            .state
            .lock()
            .map_err(|_| "TIRust context poisoned")?
            .mp_records(key, records, mode)?;
        if status == SUCCESS {
            unsafe {
                write(out, &values);
            }
        }
        Ok(status)
    })
}
/// # Safety
/// NULL/zero queries required size; otherwise `out` has `capacity` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_last_error(out: *mut u8, capacity: usize) -> usize {
    LAST_ERROR.with(|e| {
        let e = e.borrow();
        if !out.is_null() && capacity > 0 {
            let n = e.len().min(capacity - 1);
            unsafe {
                write(out, &e.as_bytes()[..n]);
                out.add(n).write(0);
            }
        }
        e.len() + 1
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn handle() -> *mut Context {
        let mut c = std::ptr::null_mut();
        assert_eq!(
            unsafe { ml_tirust_context_create(ABI_VERSION, &mut c) },
            SUCCESS
        );
        c
    }
    fn tadpole(c: *mut Context, poles: i32, mass: f64) -> (i32, [C64; 4]) {
        let mut out = [C64 {
            re: 999.0,
            im: 999.0,
        }; 4];
        let m = [C64 { re: mass, im: 0.0 }];
        let status = unsafe {
            ml_tirust_evaluate_dp(
                c,
                1,
                0,
                [0.0; 4].as_ptr(),
                4,
                [0.0].as_ptr(),
                1,
                m.as_ptr(),
                1,
                4.0,
                out.as_mut_ptr(),
                4,
                poles,
                1,
                0,
                1,
            )
        };
        (status, out)
    }
    fn qp_tadpole(
        c: *mut Context,
        mass: &str,
        poles: i32,
        finite: i32,
        check: i32,
        cache: i32,
    ) -> Vec<u8> {
        let records = |values: &[&str]| {
            let mut bytes = vec![b' '; values.len() * RECORD];
            for (i, value) in values.iter().enumerate() {
                bytes[i * RECORD..i * RECORD + value.len()].copy_from_slice(value.as_bytes());
            }
            bytes
        };
        let p = records(&["0"; 4]);
        let s = records(&["0"]);
        let m = records(&[mass, "-0"]);
        let scale = records(&[mass]);
        let mut output = vec![0; 8 * RECORD];
        let status = unsafe {
            ml_tirust_evaluate_mp(
                c,
                1,
                0,
                p.as_ptr(),
                p.len(),
                s.as_ptr(),
                s.len(),
                m.as_ptr(),
                m.len(),
                scale.as_ptr(),
                scale.len(),
                output.as_mut_ptr(),
                output.len(),
                poles,
                finite,
                check,
                cache,
            )
        };
        assert_eq!(status, SUCCESS);
        output
    }
    fn qp_counts(c: *mut Context) -> [u64; 6] {
        let mut counts = [0; 6];
        assert_eq!(
            unsafe { ml_tirust_qp_cache_statistics(c, counts.as_mut_ptr(), 6) },
            SUCCESS
        );
        counts
    }
    #[test]
    fn qp_cache_exact_inputs_modes_bypass_and_budget() {
        let c = handle();
        let original = qp_tadpole(c, "4", 0, 1, 0, 1);
        assert_eq!(qp_counts(c)[..3], [0, 1, 1]);
        assert_eq!(qp_tadpole(c, "4", 0, 1, 0, 1), original);
        assert_eq!(qp_counts(c)[..3], [1, 1, 1]);
        assert_eq!(qp_tadpole(c, "4", 0, 1, 1, 1), original);
        assert_eq!(qp_tadpole(c, "4", 0, 1, 0, 0), original);
        assert_eq!(qp_counts(c)[..3], [1, 1, 1]);
        assert_eq!(qp_counts(c)[5], 2);
        let changed = qp_tadpole(c, "9", 0, 1, 0, 1);
        assert_ne!(changed, original);
        let ir = qp_tadpole(c, "4", 2, 1, 0, 1);
        assert_ne!(ir, original);
        assert_eq!(parse_records(&ir).unwrap()[0], Float::new(MP_BITS));
        assert_eq!(qp_counts(c)[..3], [1, 3, 3]);
        assert_eq!(
            unsafe { ml_tirust_set_cache_budget(c, DEFAULT_BUDGET as i64) },
            SUCCESS
        );
        assert_eq!(qp_counts(c)[..3], [1, 3, 3]);
        // Neither the independent check nor cache-disabled path can consume
        // retained values, even when a deliberately corrupted test entry exists.
        {
            let mut state = unsafe { context(c).unwrap() }.state.lock().unwrap();
            for value in state.qp_cache.entries.values_mut() {
                value.fill(b'x');
            }
        }
        assert_eq!(qp_tadpole(c, "4", 0, 1, 1, 1), original);
        assert_eq!(qp_tadpole(c, "4", 0, 1, 0, 0), original);
        assert_eq!(unsafe { ml_tirust_clear_cache(c) }, SUCCESS);
        assert_eq!(qp_counts(c), [0; 6]);
        assert_eq!(qp_tadpole(c, "4", 0, 1, 0, 1), original);
        let used = qp_counts(c)[3];
        assert!(used <= DEFAULT_BUDGET / 2);
        assert_eq!(unsafe { ml_tirust_set_cache_budget(c, 0) }, SUCCESS);
        assert_eq!(qp_tadpole(c, "4", 0, 1, 0, 1), original);
        assert_eq!(qp_counts(c), [0, 1, 0, 0, 1, 0]);
        assert_eq!(unsafe { ml_tirust_context_destroy(c) }, SUCCESS);
    }
    #[cfg(feature = "poles-only")]
    #[test]
    fn qp_full_and_pole_only_results_never_alias() {
        let c = handle();
        let full = qp_tadpole(c, "4", 0, 1, 0, 1);
        let partial = qp_tadpole(c, "4", 0, 0, 0, 1);
        assert_ne!(full, partial);
        assert_eq!(qp_counts(c)[..3], [0, 2, 2]);
        assert_eq!(qp_tadpole(c, "4", 0, 1, 0, 1), full);
        assert_eq!(qp_tadpole(c, "4", 0, 0, 0, 1), partial);
        assert_eq!(qp_counts(c)[..3], [2, 2, 2]);
        let values = parse_records(&partial).unwrap();
        assert_eq!(values[0], Float::with_val(MP_BITS, 4));
        assert_eq!(values[6], Float::new(MP_BITS));
        assert_eq!(unsafe { ml_tirust_context_destroy(c) }, SUCCESS);
    }
    #[test]
    fn full_tadpole_selection_and_cache_lifecycle() {
        let c = handle();
        let (status, full) = tadpole(c, 0, 4.0);
        assert_eq!(status, SUCCESS);
        assert_eq!(full[0].re, 4.0);
        assert!((full[3].re - 4.0).abs() < 1e-13);
        let (status, ir) = tadpole(c, 2, 4.0);
        assert_eq!(status, SUCCESS);
        assert_eq!(ir[0], C64::default());
        assert_eq!(ir[3], full[3]);
        let (status, again) = tadpole(c, 0, 4.0);
        assert_eq!(status, SUCCESS);
        assert_eq!(again, full);
        assert_eq!(unsafe { ml_tirust_clear_cache(c) }, SUCCESS);
        let (status, changed) = tadpole(c, 0, 9.0);
        assert_eq!(status, SUCCESS);
        assert_eq!(changed[0].re, 9.0);
        assert_eq!(unsafe { ml_tirust_context_destroy(c) }, SUCCESS);
    }
    #[test]
    fn route_query_preserves_existing_error_after_success() {
        let c = handle();
        assert_eq!(tadpole(c, 0, f64::NAN).0, ERROR);
        let previous = LAST_ERROR.with(|error| error.borrow().clone());
        assert!(!previous.is_empty());
        let mut counts = [0; DP_ROUTE_STATS_COUNT];
        assert_eq!(unsafe {
            ml_tirust_dp_route_statistics(c, counts.as_mut_ptr(), DP_ROUTE_STATS_COUNT)
        }, SUCCESS);
        assert_eq!(LAST_ERROR.with(|error| error.borrow().clone()), previous);
        assert_eq!(unsafe {
            ml_tirust_dp_route_statistics(c, counts.as_mut_ptr(), DP_ROUTE_STATS_COUNT - 1)
        }, ERROR);
        let query_error = LAST_ERROR.with(|error| error.borrow().clone());
        assert_ne!(query_error, previous);
        assert_eq!(unsafe {
            ml_tirust_dp_route_statistics(c, counts.as_mut_ptr(), DP_ROUTE_STATS_COUNT)
        }, SUCCESS);
        assert_eq!(LAST_ERROR.with(|error| error.borrow().clone()), query_error);
        assert_eq!(unsafe { ml_tirust_context_destroy(c) }, SUCCESS);
    }
    #[test]
    fn invalid_requests_leave_output_untouched() {
        let c = handle();
        let (status, out) = tadpole(c, 0, f64::NAN);
        assert_eq!(status, ERROR);
        assert_eq!(
            out,
            [C64 {
                re: 999.0,
                im: 999.0
            }; 4]
        );
        unsafe {
            ml_tirust_context_destroy(c);
        }
    }
    #[test]
    fn exact_primary_repeats_reuse_the_tensor_and_checks_never_do() {
        let key = (3, 2);
        let mass = 91.188;
        let shifts = [0., 0., 0., 0., 300., 120., -40., 250., 520., -60., 110., 400.];
        let batch = || {
            let p: Vec<f64> = shifts.to_vec();
            let mut s = vec![0.; 9];
            for i in 0..3 {
                for j in 0..3 {
                    let d: Vec<f64> = (0..4).map(|a| p[4 * i + a] - p[4 * j + a]).collect();
                    s[i + 3 * j] = d[0] * d[0] - d[1] * d[1] - d[2] * d[2] - d[3] * d[3];
                }
            }
            let m = [C64 { re: mass * mass, im: -mass * 2.4 }; 3];
            dp_batch(3, &p, &s, &m, 1e6).unwrap()
        };
        let mode = Mode { poles: 0, finite: true, check: false, cache: true };
        let mut state = State::new();
        let first = state.dp_checked(key, batch(), mode).unwrap();
        assert_eq!(first.0, SUCCESS);
        assert_eq!((state.dp_cache.hits, state.dp_cache.misses, state.dp_cache.entries.len()), (0, 1, 1));
        let routes = state.dp_routes;
        // An exact repeat returns the stored tensor without a reduction.
        assert_eq!(state.dp_checked(key, batch(), mode).unwrap(), first);
        assert_eq!((state.dp_cache.hits, state.dp_cache.misses), (1, 1));
        assert_eq!(state.dp_routes, routes);
        // It equals a fresh, unshared evaluation bit for bit.
        assert_eq!(State::new().dp_checked(key, batch(), Mode { cache: false, ..mode }).unwrap(), first);
        // Checks and cache-disabled calls neither read nor fill it.
        let check = state.dp_checked(key, batch(), Mode { check: true, ..mode }).unwrap();
        assert_eq!(check.0, SUCCESS);
        assert_eq!(state.dp_checked(key, batch(), Mode { cache: false, ..mode }).unwrap(), first);
        assert_eq!((state.dp_cache.hits, state.dp_cache.misses, state.dp_cache.entries.len()), (1, 1, 1));
        // Checks share subintegrals in their own store, never the primary one.
        assert!(state.check_store.statistics().entries > 0);
        // Another pole selection is another entry; the point boundary empties it.
        let ir = state.dp_checked(key, batch(), Mode { poles: 2, ..mode }).unwrap();
        assert_eq!(ir.0, SUCCESS);
        assert_eq!(state.dp_cache.entries.len(), 2);
        let c = Box::into_raw(Box::new(Context { state: Mutex::new(state) }));
        assert_eq!(unsafe { ml_tirust_clear_cache(c) }, SUCCESS);
        let state = unsafe { Box::from_raw(c) }.state.into_inner().unwrap();
        assert_eq!((state.dp_cache.hits, state.dp_cache.misses, state.dp_cache.entries.len()), (0, 0, 0));
        assert_eq!(state.check_store.statistics().entries, 0);
    }
    #[test]
    fn failed_dp_reduction_discards_intermediate_cache_values() {
        let c = handle();
        let (_, reference) = tadpole(c, 0, 4.0);
        let (status, failed) = tadpole(c, 0, 1e308);
        assert_ne!(status, SUCCESS);
        assert_eq!(failed, [C64 { re: 999., im: 999. }; 4]);
        let mut counts = [0; 9];
        assert_eq!(
            unsafe { ml_tirust_cache_statistics(c, counts.as_mut_ptr(), 9) },
            SUCCESS
        );
        assert_eq!(counts[6], 0);
        assert_eq!(tadpole(c, 0, 4.0), (SUCCESS, reference));
        assert_eq!(unsafe { ml_tirust_context_destroy(c) }, SUCCESS);
    }
    #[test]
    fn direction_workspaces_reuse_one_immutable_plan() {
        let mut state = State::new();
        let key = (3, 2);
        let mass = 91.188;
        let forward = [
            0.,
            0.,
            0.,
            0.,
            mass / 2.,
            0.,
            0.,
            mass / 2.,
            mass,
            0.,
            0.,
            0.,
        ];
        let reverse = [
            0.,
            0.,
            0.,
            0.,
            mass,
            0.,
            0.,
            0.,
            mass / 2.,
            0.,
            0.,
            mass / 2.,
        ];
        let s_forward = [0., 0., mass * mass, 0., 0., 0., mass * mass, 0., 0.];
        let s_reverse = [0., mass * mass, 0., mass * mass, 0., 0., 0., 0., 0.];
        let evaluate = |state: &mut State, check: bool, cache: bool| {
            state
                .dp_checked(
                    key,
                    dp_batch(
                        3,
                        if check { &reverse } else { &forward },
                        if check { &s_reverse } else { &s_forward },
                        &[C64::default(); 3],
                        mass * mass,
                    )
                    .unwrap(),
                    Mode {
                        poles: 0,
                        finite: true,
                        check,
                        cache,
                    },
                )
                .unwrap()
        };
        let forward_result = evaluate(&mut state, false, true);
        let reverse_result = evaluate(&mut state, true, true);
        assert_eq!(forward_result.0, SUCCESS);
        assert_eq!(reverse_result.0, SUCCESS);
        assert_eq!(state.dp.len(), 2);
        assert_eq!(state.dp_plans.len(), 1);
        let workspace_addresses = |state: &State| {
            [false, true].map(|check| &state.dp[&(key, check)] as *const TensorWorkspace<f64>)
        };
        let original_workspaces = workspace_addresses(&state);
        assert_ne!(original_workspaces[0], original_workspaces[1]);
        let original_plan = &state.dp_plans[&key] as *const TensorReductionPlan;
        for cache in [true, false, true] {
            state.store.clear();
            assert_eq!(evaluate(&mut state, false, cache), forward_result);
            let before = state.store.statistics();
            assert_eq!(evaluate(&mut state, true, cache), reverse_result);
            let after = state.store.statistics();
            assert_eq!(before.entries, after.entries);
            assert_eq!(before.masters_reused, after.masters_reused);
            assert_eq!(before.moments_reused, after.moments_reused);
            assert_eq!(workspace_addresses(&state), original_workspaces);
            assert_eq!(
                &state.dp_plans[&key] as *const TensorReductionPlan,
                original_plan
            );
        }
    }
    #[test]
    fn unstable_dp_call_clears_numerics_but_retains_direction_workspaces() {
        let key = (1, 0);
        let evaluate = |state: &mut State, check: bool, mass: f64| {
            state
                .dp_checked(
                    key,
                    dp_batch(1, &[0.; 4], &[0.], &[C64 { re: mass, im: 0. }], 4.).unwrap(),
                    Mode {
                        poles: 0,
                        finite: true,
                        check,
                        cache: true,
                    },
                )
                .unwrap()
        };
        let workspace_addresses = |state: &State| {
            [false, true].map(|check| &state.dp[&(key, check)] as *const TensorWorkspace<f64>)
        };
        for failed_check in [false, true] {
            let mut state = State::new();
            let primary = evaluate(&mut state, false, 4.);
            let checked = evaluate(&mut state, true, 4.);
            assert_eq!(primary.0, SUCCESS);
            assert_eq!(checked.0, SUCCESS);
            assert_eq!(state.dp.len(), 2);
            assert!(state.store.statistics().entries > 0);
            let plan = &state.dp_plans[&key] as *const TensorReductionPlan;
            let workspaces = workspace_addresses(&state);
            assert_ne!(workspaces[0], workspaces[1]);
            // This existing overflow fixture exercises a returned unstable
            // lane, not a Rust error. No failed tensor or intermediate remains
            // in the event store, while both structural owners survive.
            for _ in 0..2 {
                assert_eq!(evaluate(&mut state, failed_check, 1e308), (UNSTABLE, Vec::new()));
                assert_eq!(state.store.statistics().entries, 0);
                assert_eq!(workspace_addresses(&state), workspaces);
                assert_eq!(state.dp_plans.len(), 1);
                assert_eq!(&state.dp_plans[&key] as *const TensorReductionPlan, plan);
                // Change the input after the overflow and check recovery
                // against a fresh state, then return to the original point.
                let changed = evaluate(&mut State::new(), false, 9.);
                assert_eq!(evaluate(&mut state, false, 9.), changed);
                let before_check = state.store.statistics();
                assert_eq!(evaluate(&mut state, true, 9.), changed);
                let after_check = state.store.statistics();
                assert_eq!(before_check, after_check);
                assert_eq!(evaluate(&mut state, false, 4.), primary);
                let before_check = state.store.statistics();
                assert_eq!(evaluate(&mut state, true, 4.), checked);
                assert_eq!(before_check, state.store.statistics());
                assert_eq!(workspace_addresses(&state), workspaces);
            }
        }
    }
    #[test]
    fn rust_dp_error_discards_both_workspaces_but_retains_plan() {
        let key = (1, 0);
        let mode = Mode { poles: 0, finite: true, check: false, cache: true };
        let batch = || dp_batch(1, &[0.; 4], &[0.], &[C64 { re: 4., im: 0. }], 4.).unwrap();
        let mut state = State::new();
        let reference = state.dp_checked(key, batch(), mode).unwrap();
        assert_eq!(reference.0, SUCCESS);
        let check_mode = Mode { check: true, ..mode };
        assert_eq!(state.dp_checked(key, batch(), check_mode).unwrap(), reference);
        assert_eq!(state.dp.len(), 2);
        let plan = &state.dp_plans[&key] as *const TensorReductionPlan;
        let malformed = KinematicsBatch { rows: 1, components: 2, values: Vec::new() };
        assert!(state.dp_checked(key, malformed, mode).is_err());
        assert!(state.dp.is_empty());
        assert_eq!(state.store.statistics().entries, 0);
        assert_eq!(&state.dp_plans[&key] as *const TensorReductionPlan, plan);
        assert_eq!(state.dp_checked(key, batch(), mode).unwrap(), reference);
        assert_eq!(state.dp_checked(key, batch(), check_mode).unwrap(), reference);
    }
    #[test]
    fn binary128_last_bit_and_signed_zero_transport() {
        let value = Float::with_val(113, 1) + Float::with_val(113, 2).pow(112).inv();
        let record = |s: &str| {
            let mut b = [b' '; RECORD];
            b[..s.len()].copy_from_slice(s.as_bytes());
            b
        };
        let parsed = parse_records(&record(&format!("{value:.39e}"))).unwrap();
        assert_eq!(parsed[0].to_rational(), value.to_rational());
        assert!(parse_records(&record("-0")).unwrap()[0].is_sign_negative());
        assert!(parse_records(&record("NaN")).is_err());
    }
    #[test]
    fn complex_mass_box_converges_between_256_and_384_bits() {
        let key = (4, 4);
        let plan = DirectReductionPlan::new(settings(key)).unwrap();
        let input = dp_batch(
            4,
            &[
                0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
            ],
            &[
                0., -1., -1., -1., -1., 0., -2., -2., -1., -2., 0., -2., -1., -2., -2., 0.,
            ],
            &[C64 { re: 4., im: -0.125 }; 4],
            4.,
        )
        .unwrap();
        let evaluate = |bits| {
            let mut workspace = plan.workspace(Float::new(bits)).unwrap();
            workspace.set_input_precision(113);
            workspace.set_stability_threshold(Float::with_val(bits, 2).pow(136).inv());
            let batch = KinematicsBatch {
                rows: 1,
                components: 1,
                values: input
                    .values
                    .iter()
                    .map(|v| Complex::new(Float::with_val(bits, v.re), Float::with_val(bits, v.im)))
                    .collect(),
            };
            workspace
                .reduce_batch(&batch)
                .unwrap()
                .lane(0)
                .unwrap()
                .to_vec()
        };
        let low = evaluate(256);
        let high = evaluate(384);
        let tolerance = Float::with_val(384, 2).pow(180).inv();
        for (a, b) in low.iter().zip(&high) {
            for (value, reference) in [(&a.re, &b.re), (&a.im, &b.im)] {
                let mut widened = value.clone();
                widened.set_prec(384);
                let magnitude = |v: Float| if v < Float::new(384) { -v } else { v };
                let difference = magnitude(widened - reference);
                assert!(
                    difference
                        <= (magnitude(reference.clone()) + Float::with_val(384, 1)) * &tolerance
                );
            }
        }
    }
    #[test]
    fn tensor_order_and_pole_selection() {
        let v: Vec<_> = (0..12).map(|v| Complex::new(v as f64, 0.0)).collect();
        let p = pack_dp(&v, 0, 4);
        assert_eq!(
            p.iter().map(|v| v.re).collect::<Vec<_>>(),
            vec![0., 4., 8., 1., 5., 9., 2., 6., 10., 3., 7., 11.]
        );
    }
    #[test]
    fn rank_six_scaleless_poles_sign_and_packing() {
        let c = handle();
        let key = (2, 6);
        let nc = 210;
        let p = [0., 0., 0., 0., 2., 0., 0., 2.];
        let s = [0.; 4];
        let m = [C64::default(); 2];
        let mut output = vec![C64::default(); 4 * nc];
        let evaluate = |out: &mut [C64], poles, check, cache| unsafe {
            ml_tirust_evaluate_dp(
                c,
                2,
                6,
                p.as_ptr(),
                8,
                s.as_ptr(),
                4,
                m.as_ptr(),
                2,
                4.,
                out.as_mut_ptr(),
                out.len(),
                poles,
                1,
                check,
                cache,
            )
        };
        assert_eq!(evaluate(&mut output, 0, 0, 1), SUCCESS);
        for (i, monomial) in DirectReductionPlan::new(settings(key))
            .unwrap()
            .basis()
            .iter()
            .enumerate()
        {
            let rank = monomial.rank();
            let expected = if monomial.0[1] + monomial.0[2] == 0 {
                2f64.powi(rank as i32) / (rank + 1) as f64
            } else {
                0.
            };
            assert!(
                (output[i].re - expected).abs() < 1e-12,
                "UV {:?}: {:?}",
                monomial,
                output[i]
            );
            assert!(
                (output[nc + i].re + expected).abs() < 1e-12,
                "IR {:?}",
                monomial
            );
            assert_eq!(output[2 * nc + i], C64::default());
            assert!(output[3 * nc + i].re.abs() < 1e-12);
        }
        for selection in 1..=3 {
            let mut selected = vec![C64::default(); 4 * nc];
            assert_eq!(evaluate(&mut selected, selection, 0, 1), SUCCESS);
            for i in 0..nc {
                assert_eq!(selected[3 * nc + i], output[3 * nc + i]);
            }
        }
        let mut checked = vec![C64::default(); 4 * nc];
        assert_eq!(evaluate(&mut checked, 0, 1, 1), SUCCESS);
        assert_eq!(checked, output);
        let mut uncached = vec![C64::default(); 4 * nc];
        assert_eq!(evaluate(&mut uncached, 0, 0, 0), SUCCESS);
        assert_eq!(uncached, output);
        assert_eq!(unsafe { ml_tirust_context_destroy(c) }, SUCCESS);
    }
}

#[cfg(test)]
mod captured_failures;
