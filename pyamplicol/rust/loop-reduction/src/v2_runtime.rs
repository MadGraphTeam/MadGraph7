// SPDX-License-Identifier: 0BSD

//! Plan v4 processes evaluated without Python (WP3.1 I6, the SDK's v4 path).
//!
//! A plan v4 artifact with native point inputs (`point_inputs = "native-v2"`,
//! `rusticol_core::loop_artifact::LoopV2NativeInputs`) stores, next to its
//! plan, the process's tree program (`loops/v2/native_trees.py`: the trees,
//! the Born root closure, the counterterm program and closure, the external
//! fields and the parameter program, with their saved exact evaluators and
//! binary64 SymJIT plane applications) and an assembly record
//! (`loops/v2/native_process.py`). [`V2Runtime`] evaluates a batch from those
//! alone, in binary64, through the plane applications: no Python and no
//! Symbolica state (so no licence, REQUIREMENTS_ADDENDUM G (e)):
//!
//! 1. the tree program binds the external model parameters (the caller's
//!    over the defaults, the request's fixed parameters at their fixed
//!    values; the scale parameter is `sqrt(mu_R²)` of each point) and forms
//!    every point's block rows, Born components, `|A0|²` per squared Born
//!    order and counterterm and field components
//!    (`loop_v2::trees::TreeProgram`), on the runtime's thread pool (the
//!    `threads` bound of [`V2Runtime::open`]);
//! 2. the plan's executor dresses and reduces every family
//!    (`loop_execution::v2::V2Executor`, NinjaRS through
//!    [`crate::runtime::V2ReductionBackend`] by default, or TIRust,
//!    [`V2Runtime::set_reduction`]; a TIR-only process, whose families go
//!    beyond NinjaRS's rank <= N + 2 (m-lit lit-eft), reduces with TIRust by
//!    default and refuses NinjaRS typed);
//! 3. the components are assembled per sector exactly as
//!    `loops/v2/assemble._PointEvaluation._assemble` does for the Python
//!    API's native lane (`plan_export.NativeExecutor(inputs="native")`), in
//!    the same binary64 operations and order: the bare virtual
//!    `2 Re(i/(16π² avg) Σ lanes)` summed over the lanes in the executor's
//!    order, `uv = uv_local + uv_external_fields`,
//!    `virtual = (bare + uv) + r2`. Results are therefore bitwise those of
//!    `pyamplicol.Runtime.load(...).evaluate` on the same exact inputs.
//!
//! Every pole selection of REQUIREMENTS_ADDENDUM I is served ([`V2Poles`]):
//! "all", "none" (finite-only: the pole-only tree and counterterm work and
//! the reduction's pole columns skipped natively) and the split selections
//! "uv" and "ir". A split selection reduces every lane with its UV residue
//! (`[finite, 1/ε, 1/ε², 1/ε_UV]`) and runs the stored companion of the
//! certified-scaleless families (its plan and tree program, read with the
//! process; only their UV residues are kept, their combined integrals
//! vanishing), then splits every component by origin as
//! `loops/v2/poles.split_sector` / `select_sector` do, in the same binary64
//! operations and order (`assemble._PointEvaluation.set_lanes` and
//! `_assemble`): bitwise the Python API's values. Catani's operator is not
//! needed for the selection (it is the Python API's separate IR-pole
//! prediction, `PointResult.ir_poles`, not a part of the values), so it is not
//! formed here.
//!
//! **Carriers** ([`V2Precision`]): binary64 as above, and double-double or
//! arbitrary precision ([`V2Runtime::evaluate_extended`]), where the tree
//! program and the executor run in the carrier (`DoubleFloat`, MPFR at the
//! working digits; the generic reduction engine) and the outputs enter the
//! assembly as their decimal texts in Python's `Decimal` arithmetic at the
//! working digits (`v2_decimal`), operation for operation, as the Python
//! API's `Runtime.evaluate(..., arithmetic=...)`: equal values.
//!
//! **Observable columns** (`columns`): the stored correlated, spin-source
//! and resolved column sets of the process (`loops/v2/native_observables`),
//! evaluated and assembled as the Python API's `evaluate_correlated_many`
//! and `evaluate_resolved` in every carrier ([`V2Runtime::evaluate_columns`],
//! [`V2Runtime::evaluate_columns_extended`]).
//!
//! **Model parameters** ([`V2Runtime::set_parameters`]) follow the Python
//! API (`loops/v2/artifact.V2VirtualRuntime._checked_parameters`): the whole
//! mapping is validated before any value is applied (all or nothing), and
//! unknown names (not an external parameter of the artifact's model,
//! `model/parameters.json`), the renormalisation scale's parameter (resolved
//! at generation as the Python API resolves it, the native record's
//! `scale_parameter`; each point's `mu_R²` sets it) and a value other than the
//! one a parameter was fixed to at generation (the native record's
//! `fixed_parameters`: the request's and those the model preparation fixes)
//! are refused with a typed
//! [`RusticolErrorKind::ModelParameter`](rusticol_core::RusticolErrorKind)
//! error, in the Python API's words and order (values read as Python's
//! `Decimal` of ASCII text, then names, both by name; an empty name is a
//! typed `InvalidArgument` error, the Python API's `TypeError`); a refused
//! fixed value names who fixed it (the request, or the model preparation:
//! the record's `preparation_fixed_parameters`).
//!
//! **Scale-dependent batches.** A batch shares its Born couplings, kernel
//! constants and masses across its points. Generation records which outputs
//! of the parameter program depend on the scale
//! ([`TreeProgram::scale_dependent_shared`]); for a model whose Born
//! couplings, constants or masses run with the scale (running-coupling or
//! MSbar-mass UFOs), [`V2Runtime::evaluate`] partitions a batch by exact
//! `mu_R²` and evaluates each part on its own, so every point equals its
//! single-point evaluation.

mod columns;
mod induced;
pub mod induced_outputs;
mod monitor;
pub use monitor::{Monitored, cross_reduction};

pub use columns::{COLUMN_SETS, ColumnPoint, ColumnValues, INCLUSIVE};

use crate::Complex;
use crate::runtime::{
    Carrier as DecimalCarrier, PoleSelection, Reduction, TypedBatch, V2Backends, V2Reducer,
    V2ReductionBackend, decimal_complex,
};
pub use crate::v2_decimal::{Arith, Binary64, C, Decimal, DecimalArith, cadd, cmul, conj};
use rusticol_core::VerifiedArtifact;
use rusticol_core::loop_artifact::{
    LOOP_V2_AMPLITUDE_RUNTIME_CAPABILITY, LOOP_V2_RUNTIME_CAPABILITY, LoadedLoopV2Artifact,
    LoopV2ExecutionMetadata, LoopV2NativeTrees, is_loop_v2_manifest, load_loop_v2_artifact,
    read_loop_v2_native_inputs, read_loop_v2_observables,
};
use rusticol_core::loop_execution::v2::{
    Contraction, LaneKey, PointResult, Poles, V2Executor, V2Options,
};
use rusticol_core::loop_v2::dress::{V2Batch, parse_rational};
use rusticol_core::loop_v2::plan::{LiveRows, LoopProgramV4};
use rusticol_core::loop_v2::retest::ReductionCheck;
use rusticol_core::loop_v2::trees::{
    BoundParameters, PointSources, TREE_FILL_BUDGET, TreeCarrier, TreeInputs, TreeProgram,
    TreeRows, TreeRuntime,
};
use rusticol_core::{RusticolError, RusticolResult};
use serde::Deserialize;
use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use symbolica::domains::float::{DoubleFloat, Float, Real};
pub use symbolica::domains::rational::Rational;
use symbolica::evaluate::EvaluationDomain;

/// The native assembly record's schema (`native_process.SCHEMA`).
pub const NATIVE_PROCESS_SCHEMA: u32 = 2;
/// SIMD lanes per chunk, as the Python API's executor (`plan_export.DEFAULT_LANES`).
pub const V2_LANES: usize = 16;

/// The `uv_external_fields` component (`loops/v2/fields.FIELD_COMPONENT`).
const FIELD_COMPONENT: &str = "uv_external_fields";
/// Laurent powers in coefficient order (finite, 1/ε, 1/ε²).
const LAURENT_POWERS: [i32; 3] = [0, -1, -2];
/// The bare virtual's loop factor denominator (`conventions.LOOP_FACTOR_DENOMINATOR`).
pub(crate) const LOOP_FACTOR_DENOMINATOR: i64 = 16;
/// The artifact's model parameter defaults (`generation/artifact_writer.py`):
/// every external parameter of the model, by name.
const MODEL_PARAMETERS_PATH: &str = "model/parameters.json";

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("plan v4 runtime: {}", message.into()))
}

pub type Orders = Vec<(String, i64)>;

/// Coupling orders compared as sets of (name, power) with zero powers dropped.
fn canonical(orders: &[(String, i64)]) -> Orders {
    let mut out: Orders = orders.iter().filter(|(_, p)| *p != 0).cloned().collect();
    out.sort();
    out
}

#[derive(Deserialize)]
struct NativeTreesRecord {
    squared_orders: Vec<Orders>,
    counterterm_outputs: Vec<(String, Orders, i32)>,
    #[serde(flatten)]
    _rest: BTreeMap<String, serde_json::Value>,
}

/// The assembly record of a native process (`native_process.NativeProcess`).
#[derive(Deserialize)]
struct NativeRecord {
    schema: u32,
    /// (label, initial, PDG code) per leg, by label.
    legs: Vec<(usize, bool, i32)>,
    /// (sector, its (Born, loop) order pairs), in the plan's column order.
    sectors: Vec<(Orders, Vec<(Orders, Orders)>)>,
    average: i64,
    fixed_parameters: BTreeMap<String, String>,
    /// Those of `fixed_parameters` the model preparation fixed (scheme
    /// constants; the others are the request's): what a refusal names.
    #[serde(default)]
    preparation_fixed_parameters: BTreeSet<String>,
    /// The renormalisation scale's external parameter, resolved at
    /// generation (`LoopSlice.scale_parameter`): each point's `mu_R²` sets it.
    #[serde(default)]
    scale_parameter: Option<String>,
    trees: NativeTreesRecord,
    /// The plan's certified seed FFT (`plan_export.fourier_record`), absent
    /// when every seed is contracted directly.
    #[serde(default)]
    fourier: Option<serde_json::Value>,
    /// The certified-scaleless companion's record (its plan's seed FFT).
    #[serde(default)]
    scaleless: Option<CompanionRecord>,
    /// The model's pole regulators (`poles.pole_regulators`: the external
    /// parameters multiplying the 1/ε_UV and 1/ε_IR of its counterterm
    /// parameters, epsUV and epsIR of MadLoop's electroweak UFOs), absent
    /// for a model without them. The split selections then take the
    /// counterterms' UV parts from the tree program at the IR regulator 0.
    #[serde(default)]
    pole_regulators: Option<PoleRegulators>,
    /// The plan's loop orders (`ExportedPlan.orders`): an amplitude-mode
    /// plan's executor slots are per loop order and column.
    #[serde(default)]
    orders: Vec<Orders>,
    /// A loop-induced process's amplitude basis
    /// (`induced.amplitude_record`); none for a Born-interfered one.
    #[serde(default)]
    amplitude: Option<induced::AmplitudeRecord>,
    #[serde(flatten)]
    _rest: BTreeMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct PoleRegulators {
    #[allow(dead_code)]
    uv: String,
    ir: String,
}

#[derive(Deserialize)]
struct CompanionRecord {
    #[serde(default)]
    fourier: Option<serde_json::Value>,
    #[serde(flatten)]
    _rest: BTreeMap<String, serde_json::Value>,
}

/// An executor with the seed Fourier record of its plan attached, if any.
fn with_fourier(
    executor: V2Executor,
    fourier: Option<&serde_json::Value>,
) -> RusticolResult<V2Executor> {
    match fourier {
        None => Ok(executor),
        Some(record) => executor.with_fourier(
            &serde_json::to_vec(record)
                .map_err(|e| invalid(format!("seed Fourier record: {e}")))?,
        ),
    }
}

/// The seed contraction of the process's declared policy (DESIGN.md §7.3):
/// "fft" requires the certified FFT, "direct" never runs it.
fn contraction(policy: &str) -> Contraction {
    match policy {
        "direct" => Contraction::Direct,
        "fft" => Contraction::Fft,
        _ => Contraction::Auto,
    }
}

#[derive(Deserialize)]
struct ModelParametersFile {
    kind: String,
    parameters: BTreeMap<String, serde_json::Value>,
}

/// A typed refusal of a model parameter update (the Python API's
/// `ValueError`, in its words).
fn refused(message: impl Into<String>) -> RusticolError {
    RusticolError::model_parameter(message.into())
}

/// A model parameter value as the Python API reads it
/// (`runtime.virtual._number`, `Decimal(str(value))` of ASCII text): text
/// with a non-ASCII character is refused; surrounding whitespace (Python's
/// ASCII `str.isspace` set: space, `\t`, `\n`, `\v`, `\f`, `\r` and
/// `\x1c`-`\x1f`) and then underscores are ignored; a decimal literal (a
/// sign, digits with an optional fraction part, an optional exponent) is its
/// exact value; any other text, a fraction such as `59/500` included, is
/// refused with "`description` is not a decimal number" and an infinity or
/// NaN with "`description` must be finite" (typed `ModelParameter` errors,
/// the Python API's `ValueError`).
pub fn decimal_parameter(value: &str, description: &str) -> RusticolResult<Rational> {
    if !value.is_ascii() {
        return Err(refused(format!("{description} is not a decimal number")));
    }
    let text: String = value
        .trim_matches(|c: char| matches!(c, ' ' | '\t'..='\r' | '\x1c'..='\x1f'))
        .chars()
        .filter(|c| *c != '_')
        .collect();
    let unsigned = text
        .strip_prefix(['+', '-'])
        .unwrap_or(&text)
        .to_ascii_lowercase();
    let payload = |prefix: &str| {
        unsigned
            .strip_prefix(prefix)
            .is_some_and(|digits| digits.bytes().all(|c| c.is_ascii_digit()))
    };
    if unsigned == "inf" || unsigned == "infinity" || payload("nan") || payload("snan") {
        return Err(refused(format!("{description} must be finite")));
    }
    rusticol_core::source_scalar::decimal_rational(&text)
        .map_err(|_| refused(format!("{description} is not a decimal number")))
}

/// Refuse an empty model parameter name, as the Python API does
/// (`runtime.virtual._parameters`' `TypeError`, in its words; a typed
/// `InvalidArgument` error). The empty name sorts first, so this precedes
/// every value check, as in Python's by-name reading.
pub fn named_parameters<'a>(names: impl IntoIterator<Item = &'a str>) -> RusticolResult<()> {
    if names.into_iter().any(str::is_empty) {
        return Err(RusticolError::invalid_argument(
            "model parameter names must be nonempty strings",
        ));
    }
    Ok(())
}

/// The caller's model parameter values, validated as a whole before any is
/// applied (`artifact.V2VirtualRuntime._checked_parameters` and
/// `assemble.check_model_parameters`), in the Python API's order: by name,
/// every name nonempty ([`named_parameters`]), every value
/// ([`decimal_parameter`]), then every name an external
/// parameter of the model (`externals`), none the renormalisation scale's
/// parameter (`scale`, set per point by `mu_R²`), and a parameter fixed at
/// generation (`fixed`) only at its fixed value, refused in words that say
/// who fixed it: the request (`[loop.preparation] fixed_parameters`) or the
/// model preparation (`prepared`, a scheme constant such as a colour
/// declaration's; the native record's `preparation_fixed_parameters`).
/// Returns the exact values by name, or the first refusal (nothing applied).
pub fn checked_parameters<'a>(
    values: impl IntoIterator<Item = (&'a str, &'a str)>,
    externals: &BTreeSet<String>,
    scale: Option<&str>,
    fixed: &BTreeMap<String, String>,
    prepared: &BTreeSet<String>,
) -> RusticolResult<Vec<(String, Rational)>> {
    let mut values: Vec<(&str, &str)> = values.into_iter().collect();
    values.sort_by_key(|(name, _)| *name);
    named_parameters(values.iter().map(|(name, _)| *name))?;
    let exact = values
        .iter()
        .map(|(name, value)| decimal_parameter(value, &format!("parameter '{name}' real part")))
        .collect::<RusticolResult<Vec<_>>>()?;
    for ((name, _), exact) in values.iter().zip(&exact) {
        let name = *name;
        if !externals.contains(name) {
            return Err(refused(format!("unknown external parameter '{name}'")));
        }
        if scale == Some(name) {
            return Err(refused(format!(
                "'{name}' is the renormalisation scale: pass it per point as mu_r2"
            )));
        }
        if let Some(fixed) = fixed.get(name) {
            let pinned = parse_rational(fixed).map_err(|e| invalid(e.message()))?;
            if *exact != pinned {
                return Err(refused(fixed_refusal(name, fixed, prepared.contains(name))));
            }
        }
    }
    Ok(values
        .into_iter()
        .zip(exact)
        .map(|((name, _), exact)| (name.to_owned(), exact))
        .collect())
}

/// The refusal of a value other than the one `name` was fixed to at
/// generation (`assemble.check_model_parameters`' words): by the request's
/// `[loop.preparation] fixed_parameters`, or by the model preparation
/// (`prepared`: a scheme constant, which no card entry sets).
pub fn fixed_refusal(name: &str, fixed: &str, prepared: bool) -> String {
    if prepared {
        format!(
            "parameter '{name}' was fixed to {fixed} at generation by the model \
             preparation (a scheme constant, e.g. of the colour declaration); \
             it is not a run-time parameter"
        )
    } else {
        format!(
            "parameter '{name}' was fixed to {fixed} at generation \
             ([loop.preparation] fixed_parameters); regenerate the artifact to \
             change it"
        )
    }
}

/// The external parameters of the artifact's model (`model/parameters.json`,
/// the Python API's `compiled.ir.parameters` of nature "external").
fn model_externals(artifact: &VerifiedArtifact) -> RusticolResult<BTreeSet<String>> {
    let bytes = artifact.read_payload(MODEL_PARAMETERS_PATH)?;
    let file: ModelParametersFile = serde_json::from_slice(&bytes)
        .map_err(|e| invalid(format!("{MODEL_PARAMETERS_PATH}: {e}")))?;
    if file.kind != "pyamplicol-model-parameter-defaults" {
        return Err(invalid(format!(
            "{MODEL_PARAMETERS_PATH} is not a model parameter record"
        )));
    }
    Ok(file.parameters.into_keys().collect())
}

/// One evaluated point: the Born per declared Born order and the complete
/// virtual per declared sector, `[finite, 1/ε, 1/ε²]` (finite only without
/// poles), or the executor's failures; binary64 values by default, Python
/// `Decimal` values ([`Decimal`]) in double-double and arbitrary precision.
pub type V2Point<N = f64> = Result<(Vec<N>, Vec<Vec<N>>), Vec<String>>;

/// The pole selection of an evaluation (REQUIREMENTS_ADDENDUM I,
/// `loops/v2/poles.py`): the renormalised virtual's poles (`All`), the
/// finite part alone (`None`), the bare loop's UV poles (`Uv`: the virtual's
/// poles are the unrenormalised amplitude's UV poles, its finite part the
/// renormalised one's) or the renormalised virtual's IR poles (`Ir`). The
/// finite part is identical in every selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum V2Poles {
    #[default]
    All,
    None,
    Uv,
    Ir,
}

impl V2Poles {
    /// The selection's name (`PoleSelection.value`).
    pub fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::None => "none",
            Self::Uv => "uv",
            Self::Ir => "ir",
        }
    }

    pub(crate) fn poles(self) -> bool {
        self != Self::None
    }

    pub(crate) fn split(self) -> bool {
        matches!(self, Self::Uv | Self::Ir)
    }

    /// The executor's lane columns.
    pub(crate) fn lanes(self) -> Poles {
        match self {
            Self::All => Poles::All,
            Self::None => Poles::None,
            Self::Uv | Self::Ir => Poles::Split,
        }
    }
}

/// Decimal digits of the Python API's binary64 tree replay
/// (`assemble._SYMBOLICA_PRECISION_FLOOR`), the floor of every carrier's
/// working digits.
const PRECISION_FLOOR: u32 = 40;
/// The double-double carrier's nominal digits and the guard digits of the
/// extended carriers' working digits (`assemble.DOUBLE_DOUBLE_DIGITS`,
/// `GUARD_DIGITS`).
const DOUBLE_DOUBLE_DIGITS: u32 = 32;
const GUARD_DIGITS: u32 = 20;

/// The carrier of an evaluation, the Python API's `assemble.Precision`:
/// binary64 (the NinjaRS fast engine), double-double, or `digits` decimal
/// digits of arbitrary precision (MPFR at the working digits); both
/// extended carriers reduce with the generic engine, and their outputs are
/// assembled in Python's `Decimal` arithmetic at the working digits
/// ([`V2Precision::working_digits`]), as the Python API assembles them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum V2Precision {
    Double,
    DoubleDouble,
    Arbitrary(u32),
}

impl V2Precision {
    /// `Precision.working_digits`: the decimal digits of the assembly (and
    /// of the arbitrary-precision carrier).
    pub fn working_digits(self) -> u32 {
        match self {
            Self::Double => PRECISION_FLOOR,
            Self::DoubleDouble => (DOUBLE_DOUBLE_DIGITS + GUARD_DIGITS).max(PRECISION_FLOOR),
            Self::Arbitrary(digits) => (digits + GUARD_DIGITS).max(PRECISION_FLOOR),
        }
    }

    /// The native carrier's bits (the executor's `bits`, which its decimal
    /// texts round-trip, `runtime::decimal_complex`).
    pub fn bits(self) -> RusticolResult<u32> {
        Ok(match self {
            Self::Double => 53,
            Self::DoubleDouble => 106,
            Self::Arbitrary(_) => Float::decimal_digits_to_bits(f64::from(self.working_digits()))
                .map_err(|e| invalid(e.to_string()))?,
        })
    }

    /// Lanes per chunk (`plan_export.NativeExecutor.lanes_for`): the SIMD
    /// lanes in binary64; in the extended carriers, whose lanes are scalar
    /// loops, the smallest power of two holding the batch (at most the SIMD
    /// lanes).
    pub fn lanes(self, points: usize) -> usize {
        if self == Self::Double {
            return V2_LANES;
        }
        let mut lanes = 1;
        while lanes < points.min(V2_LANES) {
            lanes *= 2;
        }
        lanes
    }

    /// The `reduction` backend of the carrier (`v2_executor.backend_of`):
    /// NinjaRS, or TIRust (its plans built for `plan`'s (propagators, rank)
    /// histogram at the carrier's precision, the other plans' on first use;
    /// its acceptance from the target accuracy `target`).
    fn backend(
        self,
        reduction: Reduction,
        plan: &LoopProgramV4,
        target: f64,
    ) -> RusticolResult<V2Backends> {
        if reduction == Reduction::TiRust {
            return V2Backends::tirust(crate::runtime::plan_histogram(plan), self.bits()?, target);
        }
        let reducer = match self {
            Self::Double => V2Reducer::new("double", None, "fast", true),
            Self::DoubleDouble => V2Reducer::new("double-double", None, "generic", true),
            Self::Arbitrary(_) => {
                V2Reducer::new("arbitrary", Some(self.working_digits()), "generic", true)
            }
        }
        .map_err(|e| invalid(e.to_string()))?;
        Ok(V2Backends::NinjaRs(V2ReductionBackend {
            reducer,
            masters: PoleSelection::BOTH,
        }))
    }
}

/// A native carrier of the tree program and the executor (binary64,
/// `DoubleFloat`, MPFR `Float`), with the assembly arithmetic its outputs
/// enter (the Python API's): binary64 values as they are, the extended
/// carriers' through their decimal texts (`runtime::decimal_complex`, the
/// executor's export) read as `Decimal` at the working digits.
pub(crate) trait Lift<T>: Arith {
    fn real(&self, value: &T, bits: u32) -> RusticolResult<Self::N>;
    fn complex(&self, value: &Complex<T>, bits: u32) -> RusticolResult<C<Self::N>>;
}

impl Lift<f64> for Binary64 {
    fn real(&self, value: &f64, _bits: u32) -> RusticolResult<f64> {
        Ok(*value)
    }
    fn complex(&self, value: &Complex<f64>, _bits: u32) -> RusticolResult<C<f64>> {
        Ok(C::new(value.re, value.im))
    }
}

impl<T: DecimalCarrier + Real> Lift<T> for DecimalArith {
    fn real(&self, value: &T, bits: u32) -> RusticolResult<Decimal> {
        let [re, _] = decimal_complex(Complex::new(value.clone(), value.zero()), bits);
        self.context.read(&re).map_err(invalid)
    }
    fn complex(&self, value: &Complex<T>, bits: u32) -> RusticolResult<C<Decimal>> {
        let [re, im] = decimal_complex(value.clone(), bits);
        Ok(C::new(
            self.context.read(&re).map_err(invalid)?,
            self.context.read(&im).map_err(invalid)?,
        ))
    }
}

/// The bounds of a native carrier.
pub(crate) trait Native:
    TreeCarrier + Real + TypedBatch + DecimalCarrier + Send + Sync + 'static
where
    Complex<Self>: Real + EvaluationDomain,
    TreeRuntime<Self>: Send,
{
}

impl Native for f64 {}
impl Native for DoubleFloat {}
impl Native for Float {}

/// One plan with its tree program, evaluated in any carrier: an executor
/// per lane count and a tree runtime per carrier, each built on first use
/// (`plan_export.NativeExecutor.native`, one PyO3 executor per carrier and
/// lane count).
pub(crate) struct Runner {
    plan: Arc<LoopProgramV4>,
    pub(crate) trees: TreeProgram,
    /// The plan's certified seed FFT record (none: every seed is contracted
    /// directly).
    fourier: Option<serde_json::Value>,
    threads: Option<usize>,
    executors: BTreeMap<usize, V2Executor>,
    runtimes: HashMap<V2Precision, Box<dyn Any + Send>>,
}

impl Runner {
    pub(crate) fn new(
        plan: Arc<LoopProgramV4>,
        trees: TreeProgram,
        fourier: Option<serde_json::Value>,
        threads: Option<usize>,
    ) -> RusticolResult<Self> {
        Ok(Self {
            plan,
            trees,
            fourier,
            threads,
            executors: BTreeMap::new(),
            runtimes: HashMap::new(),
        })
    }

    fn executor(&mut self, lanes: usize) -> RusticolResult<&V2Executor> {
        if !self.executors.contains_key(&lanes) {
            let executor = with_fourier(
                V2Executor::new(self.plan.clone(), lanes, self.threads)?,
                self.fourier.as_ref(),
            )?;
            self.executors.insert(lanes, executor);
        }
        Ok(&self.executors[&lanes])
    }

    /// Bind a batch's parameters and form its native inputs in the carrier
    /// of `zero` on the executor's pool (`TreeProgram::bind`, then
    /// `inputs_with`): the bound parameters and the tree inputs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fill<T: Native>(
        &mut self,
        precision: V2Precision,
        lanes: usize,
        values: &[Rational],
        scales: &[Rational],
        incoming: &[Vec<Rational>],
        poles: bool,
        sources: Option<&[PointSources<T>]>,
        zero: &T,
    ) -> RusticolResult<(BoundParameters<T>, TreeInputs<T>)>
    where
        Complex<T>: Real + EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        self.bind_and_fill(
            precision,
            lanes,
            values,
            scales,
            zero,
            |trees, runtime, plan, bound| {
                trees.inputs_with(
                    runtime,
                    plan,
                    &bound.trees,
                    incoming,
                    &bound.constants,
                    poles,
                    sources,
                    zero,
                )
            },
        )
    }

    /// The tree runtime of `precision` (formed once) and the executor of
    /// `lanes`: the batch's parameters bound, then its tree inputs formed by
    /// `inputs` (the program, the runtime, the plan, the bound parameters),
    /// both on the executor's pool ([`Runner::fill`], [`Runner::fill_live`]).
    fn bind_and_fill<T: Native, F>(
        &mut self,
        precision: V2Precision,
        lanes: usize,
        values: &[Rational],
        scales: &[Rational],
        zero: &T,
        inputs: F,
    ) -> RusticolResult<(BoundParameters<T>, TreeInputs<T>)>
    where
        Complex<T>: Real + EvaluationDomain,
        TreeRuntime<T>: Send,
        F: FnOnce(
                &TreeProgram,
                &mut TreeRuntime<T>,
                &LoopProgramV4,
                &BoundParameters<T>,
            ) -> RusticolResult<TreeInputs<T>>
            + Send,
    {
        self.executor(lanes)?;
        if !self.runtimes.contains_key(&precision) {
            let runtime = self.trees.runtime(zero)?;
            self.runtimes.insert(precision, Box::new(runtime));
        }
        let runtime = self
            .runtimes
            .get_mut(&precision)
            .and_then(|runtime| runtime.downcast_mut::<TreeRuntime<T>>())
            .ok_or_else(|| invalid("a tree runtime of another carrier"))?;
        let executor = &self.executors[&lanes];
        let trees = &self.trees;
        executor.install(|| -> RusticolResult<_> {
            let bound = trees.bind(runtime, values, scales, zero)?;
            let inputs = inputs(trees, runtime, executor.plan(), &bound)?;
            Ok((bound, inputs))
        })
    }

    /// Dress and reduce a batch (`V2Executor::evaluate`).
    pub(crate) fn evaluate<T: Native>(
        &mut self,
        lanes: usize,
        batch: &V2Batch<T>,
        zero: &T,
        backend: &mut V2Backends,
        options: V2Options,
    ) -> RusticolResult<Vec<PointResult<T>>>
    where
        Complex<T>: Real + EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        let (results, _report) = self
            .executor(lanes)?
            .evaluate(batch, zero, backend, options)?;
        Ok(results)
    }

    /// The row liveness of a call keeping the plan's helicity rows `rows`
    /// (one flag per plan row), as the Python binding's
    /// `V2Executor.evaluate(rows=...)` forms it (`evaluate_with`): the
    /// executor's families pruned to the rows the kept rows reach
    /// (`V2Executor::live_rows`, schedule B) and the tree fill's liveness
    /// (`TreeProgram::live_rows`: an amplitude-mode program forms the kept
    /// rows' trees and counterterm kets only). Both None when `rows` drops
    /// no seeded row.
    pub(crate) fn row_liveness(
        &mut self,
        lanes: usize,
        rows: &[bool],
        options: &V2Options,
    ) -> RusticolResult<RowLiveness> {
        self.executor(lanes)?;
        let executor = &self.executors[&lanes];
        let live = executor.live_rows(rows, None, options.schedule)?;
        let trees = live
            .as_deref()
            .and_then(|live| self.trees.live_rows(executor.plan(), live));
        Ok(RowLiveness { live, trees })
    }

    /// [`Runner::fill`] of a call with a row liveness (the Python binding's
    /// `Trees::fill_part` with `filling.rows`): the tree fill of
    /// `TreeProgram::inputs_with`, restricted to the live tree rows.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fill_live<T: Native>(
        &mut self,
        precision: V2Precision,
        lanes: usize,
        values: &[Rational],
        scales: &[Rational],
        incoming: &[Vec<Rational>],
        poles: bool,
        rows: &RowLiveness,
        zero: &T,
    ) -> RusticolResult<(BoundParameters<T>, TreeInputs<T>)>
    where
        Complex<T>: Real + EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        let Some(tree_rows) = &rows.trees else {
            return self.fill(
                precision, lanes, values, scales, incoming, poles, None, zero,
            );
        };
        self.bind_and_fill(
            precision,
            lanes,
            values,
            scales,
            zero,
            |trees, runtime, plan, bound| {
                let chunk = trees.fill_points::<T>(incoming.len(), false, TREE_FILL_BUDGET);
                trees.inputs_chunked(
                    runtime,
                    plan,
                    &bound.trees,
                    incoming,
                    &bound.constants,
                    poles,
                    None,
                    None,
                    chunk,
                    Some(tree_rows),
                    zero,
                )
            },
        )
    }

    /// [`Runner::evaluate`] of a call with a row liveness
    /// (`V2Executor::evaluate_families` with the live rows: every family
    /// pruned to them, as the Python binding's).
    pub(crate) fn evaluate_live<T: Native>(
        &mut self,
        lanes: usize,
        batch: &V2Batch<T>,
        zero: &T,
        backend: &mut V2Backends,
        options: V2Options,
        rows: &RowLiveness,
    ) -> RusticolResult<Vec<PointResult<T>>>
    where
        Complex<T>: Real + EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        let (results, _report) = self.executor(lanes)?.evaluate_families(
            batch,
            zero,
            backend,
            options,
            None,
            rows.live.as_deref(),
        )?;
        Ok(results)
    }
}

/// The row liveness of one call keeping some plan rows
/// ([`Runner::row_liveness`]).
pub(crate) struct RowLiveness {
    live: Option<Arc<LiveRows>>,
    trees: Option<TreeRows>,
}

/// The executor batch of bound parameters and tree inputs.
pub(crate) fn v2_batch<T: Clone>(
    incoming: Vec<Vec<Rational>>,
    scales: &[Rational],
    bound: BoundParameters<T>,
    block_rows: Vec<Vec<Complex<T>>>,
    born: Vec<Vec<Complex<T>>>,
) -> V2Batch<T> {
    V2Batch {
        momenta: incoming,
        scales: scales.to_vec(),
        masses: bound.masses,
        masses_imaginary: bound.masses_imaginary,
        // The on-shell external invariants (as the Python binding's).
        leg_masses: bound.leg_masses,
        constants: bound.constants,
        block_rows,
        born,
    }
}

/// Laurent coefficients `[finite, 1/ε, 1/ε²]` added coefficient-wise
/// (`assemble.Laurent.__add__`).
fn add<A: Arith>(arith: &A, a: &[A::N; 3], b: &[A::N; 3]) -> [A::N; 3] {
    [
        arith.add(&a[0], &b[0]),
        arith.add(&a[1], &b[1]),
        arith.add(&a[2], &b[2]),
    ]
}

/// The UV Laurent values `(r2, uv_local, fields)` of the counterterm
/// components (finite parts unused): the tree program at the IR regulator 0
/// of a model with pole regulators.
pub(crate) type CountertermsUv<N> = ([N; 3], [N; 3], [N; 3]);

/// The selected virtual of one sector (or output column) in a split
/// selection, from its combined bare Laurent `bare`, the bare loop's UV
/// Laurent `bare_uv` (finite part unused), the counterterm components and,
/// for a model with pole regulators, their UV parts `counterterms_uv`:
/// `poles.split_sector` then `poles.select_sector`, operation for operation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn split_virtual<A: Arith>(
    arith: &A,
    mode: V2Poles,
    bare: [A::N; 3],
    bare_uv: [A::N; 3],
    r2: [A::N; 3],
    uv_local: [A::N; 3],
    fields: [A::N; 3],
    counterterms_uv: Option<CountertermsUv<A::N>>,
) -> [A::N; 3] {
    // The UV (single, double) poles per component: the counterterms' own
    // (pole regulators), else the local counterterms' poles, with the field
    // factors carrying what makes the renormalised virtual UV finite.
    let poles = |value: &[A::N; 3]| [value[1].clone(), value[2].clone()];
    let (r2_uv, local_uv, field_uv) = match &counterterms_uv {
        Some((r2_uv, local_uv, field_uv)) => (poles(r2_uv), poles(local_uv), poles(field_uv)),
        None => (
            poles(&r2),
            poles(&uv_local),
            [
                arith.neg(&arith.add(&arith.add(&bare_uv[1], &uv_local[1]), &r2[1])),
                arith.neg(&arith.add(&arith.add(&bare_uv[2], &uv_local[2]), &r2[2])),
            ],
        ),
    };
    let part = |value: &[A::N; 3], uv: &[A::N; 2]| -> [A::N; 3] {
        match mode {
            V2Poles::Uv => [value[0].clone(), uv[0].clone(), uv[1].clone()],
            _ => [
                value[0].clone(),
                arith.sub(&value[1], &uv[0]),
                arith.sub(&value[2], &uv[1]),
            ],
        }
    };
    let bare_part = part(&bare, &[bare_uv[1].clone(), bare_uv[2].clone()]);
    let r2_part = part(&r2, &r2_uv);
    let local_part = part(&uv_local, &local_uv);
    let field_part = part(&fields, &field_uv);
    let uv = add(arith, &local_part, &field_part);
    let virtual_part = add(arith, &add(arith, &bare_part, &uv), &r2_part);
    match mode {
        // The bare loop's UV poles with the renormalised finite part.
        V2Poles::Uv => [
            virtual_part[0].clone(),
            bare_part[1].clone(),
            bare_part[2].clone(),
        ],
        _ => virtual_part,
    }
}

/// The complex loop factor `i / (16 π² average)` of the bare virtual
/// (`numbers.pair(0, 1) * (1 / (16 * pi * pi * average))`).
pub(crate) fn loop_factor<A: Arith>(arith: &A, average: i64) -> C<A::N> {
    let denominator = arith.mul(
        &arith.mul(
            &arith.mul(&arith.int(LOOP_FACTOR_DENOMINATOR), &arith.pi()),
            &arith.pi(),
        ),
        &arith.int(average),
    );
    let f = arith.div(&arith.int(1), &denominator);
    arith.times_real(&C::new(arith.real_int(0), arith.real_int(1)), &f)
}

/// A plan v4 process with native point inputs, evaluated in binary64,
/// double-double or arbitrary precision.
pub struct V2Runtime {
    path: PathBuf,
    artifact: LoadedLoopV2Artifact,
    record: NativeRecord,
    process: Runner,
    /// The reduction backend of every evaluation ([`V2Runtime::set_reduction`]).
    reduction: Reduction,
    /// The target accuracy of every evaluation ([`V2Runtime::set_target`]).
    target: f64,
    backends: BTreeMap<V2Precision, V2Backends>,
    /// The other backend's, for the stability monitoring's cross-backend
    /// check (`monitor`), built on first use.
    cross_backends: BTreeMap<V2Precision, V2Backends>,
    /// The stored companion's tree program (none: the process has no
    /// certified-scaleless families) and, once a split selection ran, the
    /// companion itself.
    scaleless_inputs: Option<Arc<LoopV2NativeTrees>>,
    scaleless: Option<Runner>,
    threads: Option<usize>,
    /// The external parameters of the artifact's model.
    externals: BTreeSet<String>,
    /// The caller's model parameter values (exact; validated by
    /// [`checked_parameters`]).
    parameters: BTreeMap<String, Rational>,
    /// Per declared Born order its squared Born order in the tree program
    /// (none: zero), per declared sector its record sector.
    born_index: Vec<Option<usize>>,
    sector_index: Vec<usize>,
    /// The process's stored tree program (the observable column sets load
    /// it with their plans).
    process_inputs: Arc<LoopV2NativeTrees>,
    /// The stored observable columns, read on first use.
    observables: Option<columns::Observables>,
    /// A loop-induced process's amplitude assembly (|A_loop|², m-lit
    /// LIT-SDK); none for a Born-interfered process. Chosen at open: a
    /// runtime assembles one process type (no per-point branch).
    induced: Option<Arc<induced::Induced>>,
    /// A loop-induced process's declared observables, read on first use.
    induced_observables: Option<induced_outputs::Observables>,
}

/// Whether the artifact at `path` serves its processes through the v2 loop
/// runtimes (every process requires one of the v2 loop runtime capabilities,
/// Born-interfered and loop-induced processes in any mix,
/// `loop_artifact::is_loop_v2_manifest`); [`V2Runtime::open`] serves both
/// process types.
pub fn is_v2_artifact(path: impl AsRef<Path>) -> RusticolResult<bool> {
    let artifact = rusticol_core::VerifiedArtifact::open(path)?;
    Ok(is_loop_v2_manifest(artifact.manifest()))
}

impl V2Runtime {
    /// Open one process (`process`: its id or expression; none: the only
    /// one) of a plan v4 artifact with native point inputs. `threads` bounds
    /// the executor's threads (none: every core).
    pub fn open(
        path: impl AsRef<Path>,
        process: Option<&str>,
        threads: Option<usize>,
    ) -> RusticolResult<Self> {
        let path = path.as_ref().to_path_buf();
        let verified = VerifiedArtifact::open(&path)?;
        let artifact = load_loop_v2_artifact(&verified, process)?;
        let externals = model_externals(&verified)?;
        let native = read_loop_v2_native_inputs(&verified, &artifact)?.ok_or_else(|| {
            invalid("the process has no native point inputs; regenerate the artifact")
        })?;
        let record: NativeRecord = serde_json::from_slice(&native.record_json)
            .map_err(|e| invalid(format!("native process record: {e}")))?;
        if record.schema != NATIVE_PROCESS_SCHEMA {
            return Err(invalid(
                "unsupported native process schema; regenerate the artifact",
            ));
        }
        let program = artifact.program.clone();
        let trees = TreeProgram::load(
            &native.trees_json,
            &native.evaluators,
            Some(&native.planes),
            &program,
        )?;
        // A loop-induced process (an amplitude-mode plan,
        // `rusticol.loop-runtime.v5`) reads its amplitude record; its
        // counterterms are kets on the amplitude basis (`induced`).
        let loop_induced = artifact.metadata.contribution_kind == "loop-induced";
        if loop_induced != program.amplitude || loop_induced != record.amplitude.is_some() {
            return Err(invalid(format!(
                "process {:?} is {} ({}) but its plan or native record is not",
                artifact.selection.process.id,
                artifact.metadata.contribution_kind,
                if loop_induced {
                    LOOP_V2_AMPLITUDE_RUNTIME_CAPABILITY
                } else {
                    LOOP_V2_RUNTIME_CAPABILITY
                }
            )));
        }
        if !loop_induced && !trees.has_counterterms() {
            return Err(invalid("the tree program has no counterterm components"));
        }
        // The split selections set the IR regulator by name: it must be a
        // parameter the tree program binds (fail closed, as
        // `plan_export._regulated`).
        if let Some(regulators) = &record.pole_regulators
            && !trees
                .parameter_names()
                .iter()
                .any(|name| *name == regulators.ir)
        {
            return Err(invalid(format!(
                "the pole regulator {:?} is not a parameter of the tree program; \
                 regenerate the artifact",
                regulators.ir
            )));
        }
        if native.scaleless.is_some() != artifact.scaleless_program.is_some() {
            return Err(invalid(
                "the companion's tree program is stored exactly with its plan",
            ));
        }
        let metadata = &artifact.metadata;
        let squared: Vec<Orders> = record
            .trees
            .squared_orders
            .iter()
            .map(|o| canonical(o))
            .collect();
        let born_index = metadata
            .born_orders
            .iter()
            .map(|order| {
                squared
                    .iter()
                    .position(|o| *o == canonical(&order_pairs(order)))
            })
            .collect();
        let sectors: Vec<Orders> = record.sectors.iter().map(|(s, _)| canonical(s)).collect();
        let sector_index = metadata
            .sectors
            .iter()
            .map(|sector| {
                let wanted = canonical(&order_pairs(&sector.order));
                sectors
                    .iter()
                    .position(|s| *s == wanted)
                    .ok_or_else(|| invalid("a declared sector is not in the native record"))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        if record.legs.len() != program_legs(&artifact) {
            return Err(invalid("the native record's legs differ from the plan's"));
        }
        if trees
            .parameter_names()
            .iter()
            .any(|name| !externals.contains(name))
        {
            return Err(invalid(
                "the tree program reads a parameter the model does not declare",
            ));
        }
        let induced = match &record.amplitude {
            Some(amplitude) => Some(Arc::new(induced::Induced::new(
                amplitude.clone(),
                record.orders.clone(),
                &record.sectors,
                program.rows,
            )?)),
            None => None,
        };
        let mut process = Runner::new(program, trees, record.fourier.clone(), threads)?;
        // The binary64 executor, tree runtime and backend (the Python API's,
        // `v2_executor.backend_of`) are built at open.
        process.executor(V2_LANES)?;
        let runtime = process.trees.runtime(&0.0)?;
        process
            .runtimes
            .insert(V2Precision::Double, Box::new(runtime));
        // The default backend: what "auto" takes for the process type at
        // binary64 (the one policy, `reduction_policy::resolve`: TIRust for a
        // loop-induced or a TIR-only process where this library serves it,
        // else NinjaRS); a TIR-only process in a build without TIRust is
        // refused typed.
        let tir_only = crate::runtime::ninjars_refusal(&process.plan).is_err();
        let admitted = tir_only.then(|| BTreeSet::from(["tirust".to_owned()]));
        let route = crate::reduction_policy::route(crate::reduction_policy::SUMMED)
            .expect("the summed route");
        let (reduction, fallback) = match crate::reduction_policy::resolve(
            "auto",
            &artifact.metadata.contribution_kind,
            "double",
            &crate::reduction_policy::served(),
            admitted.as_ref(),
            &route,
        ) {
            Ok(choice) => (
                Reduction::parse(&choice.effective)?,
                choice.fallback.is_some(),
            ),
            Err(_) if tir_only => {
                return Err(RusticolError::unsupported_runtime_capability(
                    "tirust",
                    format!(
                        "this process is TIR-only (a family beyond NinjaRS's rank <= N+2) and {}",
                        crate::runtime::TIRUST_NOT_BUILT
                    ),
                ));
            }
            Err(crate::reduction_policy::Refusal::Unavailable(message))
            | Err(crate::reduction_policy::Refusal::Invalid(message)) => {
                return Err(RusticolError::unsupported_runtime_capability(
                    "auto", message,
                ));
            }
        };
        let target = crate::reduction_policy::DEFAULT_TARGET;
        let mut backends = BTreeMap::new();
        match V2Precision::Double.backend(reduction, &process.plan, target) {
            Ok(backend) => {
                backends.insert(V2Precision::Double, backend);
            }
            // TIRust refuses the plans (its capability query) where "auto"
            // has the NinjaRS fallback: built on use, each call's refusal
            // leaves its points to the fallback.
            Err(error)
                if fallback
                    && error.kind()
                        == rusticol_core::RusticolErrorKind::UnsupportedRuntimeCapability => {}
            Err(error) => return Err(error),
        }
        Ok(Self {
            path,
            artifact,
            record,
            process,
            reduction,
            target,
            backends,
            cross_backends: BTreeMap::new(),
            scaleless_inputs: native.scaleless.map(Arc::new),
            scaleless: None,
            threads,
            externals,
            parameters: BTreeMap::new(),
            born_index,
            sector_index,
            process_inputs: Arc::new(LoopV2NativeTrees {
                trees_json: native.trees_json,
                evaluators: native.evaluators,
                planes: native.planes,
            }),
            observables: None,
            induced,
            induced_observables: None,
        })
    }

    /// The companion of the certified-scaleless families, built on its
    /// first use (`plan_export.NativeExecutor.scaleless`): its plan's
    /// columns are the process's sectors.
    fn companion(&mut self) -> RusticolResult<Option<&mut Runner>> {
        if self.scaleless.is_none() {
            let (Some(stored), Some(program)) =
                (&self.scaleless_inputs, &self.artifact.scaleless_program)
            else {
                return Ok(None);
            };
            if program.columns != self.record.sectors.len() {
                return Err(invalid("the companion's columns are not the sectors"));
            }
            let trees = TreeProgram::load(
                &stored.trees_json,
                &stored.evaluators,
                Some(&stored.planes),
                program,
            )?;
            if trees.parameter_names() != self.process.trees.parameter_names() {
                return Err(invalid(
                    "the companion's model parameters differ from the process's",
                ));
            }
            let fourier = self
                .record
                .scaleless
                .as_ref()
                .and_then(|twin| twin.fourier.clone());
            self.scaleless = Some(Runner::new(program.clone(), trees, fourier, self.threads)?);
        }
        Ok(self.scaleless.as_mut())
    }

    /// Whether the process stores the companion of certified-scaleless
    /// families (the split selections run it).
    pub fn has_companion(&self) -> bool {
        self.scaleless_inputs.is_some()
    }

    pub fn metadata(&self) -> &LoopV2ExecutionMetadata {
        &self.artifact.metadata
    }

    /// The declared metadata as JSON (orders, sectors, scheme).
    pub fn metadata_json(&self) -> RusticolResult<String> {
        serde_json::to_string(&self.artifact.metadata)
            .map_err(|e| RusticolError::serialization(e.to_string()))
    }

    pub fn external_count(&self) -> usize {
        self.record.legs.len()
    }

    pub fn helicities(&self) -> usize {
        self.artifact.helicities.len()
    }

    pub fn born_orders(&self) -> usize {
        self.born_index.len()
    }

    /// Per declared (public) sector its record sector, the sector order of
    /// a stability estimate's per-sector values.
    pub fn sector_index(&self) -> &[usize] {
        &self.sector_index
    }

    pub fn sectors(&self) -> usize {
        self.sector_index.len()
    }

    pub fn plan_sha256(&self) -> &str {
        &self.artifact.plan_sha256
    }

    /// The renormalisation scale's external parameter: the one generation
    /// resolved (the native record's, as the Python API reads it), else the
    /// one the tree program reads.
    pub fn scale_parameter(&self) -> Option<&str> {
        self.record
            .scale_parameter
            .as_deref()
            .or_else(|| self.process.trees.scale_parameter())
    }

    /// The parameters fixed at generation (the request's and those its model
    /// preparation fixes, e.g. a colour declaration's external scheme
    /// constants: `LoopSlice.fixed_parameters`), with their exact decimal
    /// values.
    pub fn fixed_parameters(&self) -> &BTreeMap<String, String> {
        &self.record.fixed_parameters
    }

    /// Set the caller's model parameter values (exact decimal strings, read
    /// as Python's `Decimal`: [`decimal_parameter`]) by name, as the Python
    /// API's `set_model_parameters`: the whole mapping is validated first
    /// ([`checked_parameters`]: a value that is not a finite decimal, unknown
    /// names, the scale parameter and a fixed parameter at another value are
    /// refused with a typed `ModelParameter` error), then every value is
    /// applied; a refused update changes nothing. A known parameter the
    /// process does not read is accepted (it has no effect), as in Python.
    pub fn set_parameters<'a>(
        &mut self,
        values: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> RusticolResult<()> {
        let checked = checked_parameters(
            values,
            &self.externals,
            self.scale_parameter(),
            &self.record.fixed_parameters,
            &self.record.preparation_fixed_parameters,
        )?;
        self.parameters.extend(checked);
        Ok(())
    }

    /// The external parameter values of a batch (`native_trees.
    /// batch_parameters`): the caller's over the defaults, a fixed parameter
    /// at its fixed value (`set_parameters` admits no other).
    fn parameter_values(&self) -> RusticolResult<Vec<Rational>> {
        let names = self.process.trees.parameter_names();
        let defaults = self.process.trees.parameter_defaults();
        names
            .iter()
            .zip(defaults)
            .map(|(name, default)| {
                if let Some(value) = self.record.fixed_parameters.get(name) {
                    return parse_rational(value).map_err(|e| invalid(e.message()));
                }
                Ok(self
                    .parameters
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| default.clone()))
            })
            .collect()
    }

    /// The parameter values with the model's IR pole regulator at 0 (the
    /// counterterms' UV parts in the split selections), or None for a model
    /// without pole regulators.
    fn regulated_values(&self, values: &[Rational]) -> Option<Vec<Rational>> {
        let regulators = self.record.pole_regulators.as_ref()?;
        Some(
            self.process
                .trees
                .parameter_names()
                .iter()
                .zip(values)
                .map(|(name, value)| {
                    if *name == regulators.ir {
                        Rational::from(0)
                    } else {
                        value.clone()
                    }
                })
                .collect(),
        )
    }

    /// Whether a batch of mixed `mu_R²` is evaluated per scale: the tree
    /// program (or, in a split selection, the companion's) has Born
    /// couplings, kernel constants or masses that depend on the scale
    /// (recorded at generation; an artifact written before that record is
    /// checked per batch by `TreeProgram::bind`, which refuses a batch whose
    /// shared values differ, naming them).
    fn partitions(&mut self, mode: V2Poles) -> RusticolResult<bool> {
        let runs = |trees: &TreeProgram| {
            trees
                .scale_dependent_shared()
                .is_some_and(|names| !names.is_empty())
        };
        if runs(&self.process.trees) {
            return Ok(true);
        }
        Ok(mode.split() && self.companion()?.is_some_and(|twin| runs(&twin.trees)))
    }

    /// Evaluate a batch in binary64: `momenta` holds per point the physical
    /// momenta of every leg (`4 × legs` exact decimals, label order),
    /// `scales` mu_R² per point, `mode` the pole selection. Points fail
    /// alone. A batch of a model whose Born couplings, kernel constants or
    /// masses depend on the scale ([`TreeProgram::scale_dependent_shared`])
    /// is evaluated per exact `mu_R²` (each point as in a batch of its scale
    /// alone).
    pub fn evaluate(
        &mut self,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
    ) -> RusticolResult<Vec<V2Point>> {
        self.partitioned(momenta, scales, mode, |runtime, momenta, scales| {
            runtime.evaluate_batch::<_, _, false>(
                momenta,
                scales,
                mode,
                V2Precision::Double,
                &0.0,
                &Binary64,
                None,
                &mut Vec::new(),
            )
        })
    }

    /// Evaluate a batch in double-double or arbitrary precision
    /// (`precision`; binary64 is [`V2Runtime::evaluate`]), as `evaluate`:
    /// the executor runs in the carrier and the outputs are assembled in
    /// Python's `Decimal` arithmetic at the working digits, as the Python
    /// API's `Runtime.evaluate(..., arithmetic=...)` assembles them: equal
    /// values.
    pub fn evaluate_extended(
        &mut self,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
        precision: V2Precision,
    ) -> RusticolResult<Vec<V2Point<Decimal>>> {
        let arith = DecimalArith::new(precision.working_digits());
        match precision {
            V2Precision::Double => Err(invalid("binary64 evaluations are V2Runtime::evaluate's")),
            V2Precision::DoubleDouble => {
                self.partitioned(momenta, scales, mode, |runtime, momenta, scales| {
                    runtime.evaluate_batch::<_, _, false>(
                        momenta,
                        scales,
                        mode,
                        precision,
                        &DoubleFloat::from(0.0),
                        &arith,
                        None,
                        &mut Vec::new(),
                    )
                })
            }
            V2Precision::Arbitrary(_) => {
                let zero = Float::with_val(precision.bits()?, 0);
                self.partitioned(momenta, scales, mode, |runtime, momenta, scales| {
                    runtime.evaluate_batch::<_, _, false>(
                        momenta,
                        scales,
                        mode,
                        precision,
                        &zero,
                        &arith,
                        None,
                        &mut Vec::new(),
                    )
                })
            }
        }
    }

    /// The input checks of a batch and its evaluation, per exact `mu_R²`
    /// where the model needs it (`partitions`).
    fn partitioned<P>(
        &mut self,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
        mut batch: impl FnMut(&mut Self, &[Vec<Rational>], &[Rational]) -> RusticolResult<Vec<P>>,
    ) -> RusticolResult<Vec<P>> {
        self.check_batch(momenta, scales)?;
        self.check_poles(mode)?;
        if momenta.is_empty() {
            return Ok(Vec::new());
        }
        if scales.iter().all(|scale| *scale == scales[0]) || !self.partitions(mode)? {
            return batch(self, momenta, scales);
        }
        // Exact partition by mu_R² (first-seen order); every part is a batch
        // of one scale.
        let mut parts: Vec<(&Rational, Vec<usize>)> = Vec::new();
        let mut part_of: BTreeMap<&Rational, usize> = BTreeMap::new();
        for (index, scale) in scales.iter().enumerate() {
            let part = *part_of.entry(scale).or_insert_with(|| {
                parts.push((scale, Vec::new()));
                parts.len() - 1
            });
            parts[part].1.push(index);
        }
        let mut out: Vec<Option<P>> = (0..momenta.len()).map(|_| None).collect();
        for (scale, indices) in parts {
            let part_momenta: Vec<Vec<Rational>> =
                indices.iter().map(|&i| momenta[i].clone()).collect();
            let part_scales = vec![scale.clone(); indices.len()];
            let points = batch(self, &part_momenta, &part_scales)?;
            for (index, point) in indices.into_iter().zip(points) {
                out[index] = Some(point);
            }
        }
        Ok(out
            .into_iter()
            .map(|point| point.expect("every point is in one part"))
            .collect())
    }

    /// The pole selection `mode` is one the process serves (its metadata's
    /// `poles`: every selection for a Born-interfered process, "all" and
    /// "none" for a loop-induced one, whose UV and IR parts are not additive
    /// in |A|², m-lit PLAN §2.a.7), refused typed in the Python API's words
    /// otherwise (`V2VirtualRuntime.evaluate`'s `CompatibilityError`).
    pub fn check_poles(&self, mode: V2Poles) -> RusticolResult<()> {
        let name = mode.name();
        let served = &self.artifact.metadata.poles;
        if served.iter().any(|selection| selection == name) {
            return Ok(());
        }
        let listed: Vec<String> = served.iter().map(|s| format!("'{s}'")).collect();
        Err(RusticolError::compatibility(format!(
            "the pole selection '{name}' is not served by this {} process (it \
             serves [{}])",
            self.artifact.metadata.contribution_kind,
            listed.join(", ")
        )))
    }

    /// Whether the process is loop-induced (|A_loop|², no Born).
    pub fn loop_induced(&self) -> bool {
        self.induced.is_some()
    }

    /// A loop-induced process's colour-zero record (m-lit Q13: the one exact
    /// zero, by colour algebra alone): whether the whole request is an exact
    /// colour zero and its exact colour-zero loop orders; none for a
    /// Born-interfered process.
    pub fn colour_zero(&self) -> Option<(bool, Vec<Orders>)> {
        self.induced.as_ref().map(|induced| {
            (
                induced.record.colour_zero,
                induced.record.colour_zero_orders.clone(),
            )
        })
    }

    /// The generation request's reduction backend (`[loop] reduction`, the
    /// artifact's default of a call that names none; "auto" when absent).
    pub fn artifact_reduction(&self) -> &str {
        self.artifact
            .request
            .get("reduction")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("auto")
    }

    /// One mu_R² per point, `4 × legs` components per point, positive
    /// scales.
    pub(crate) fn check_batch(
        &self,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
    ) -> RusticolResult<()> {
        let legs = self.record.legs.len();
        if momenta.len() != scales.len() {
            return Err(invalid("one mu_R² per point"));
        }
        if momenta.iter().any(|point| point.len() != 4 * legs) {
            return Err(invalid("momenta need 4 components per leg"));
        }
        if scales.iter().any(|scale| *scale <= Rational::from(0)) {
            return Err(invalid("mu_R² must be positive"));
        }
        Ok(())
    }

    /// All-incoming momenta: final legs negated.
    pub(crate) fn incoming(&self, momenta: &[Vec<Rational>]) -> Vec<Vec<Rational>> {
        momenta
            .iter()
            .map(|point| {
                point
                    .chunks_exact(4)
                    .zip(&self.record.legs)
                    .flat_map(|(p, &(_, initial, _))| {
                        p.iter()
                            .map(move |x| if initial { x.clone() } else { -x.clone() })
                    })
                    .collect()
            })
            .collect()
    }

    fn backend(&mut self, precision: V2Precision) -> RusticolResult<&mut V2Backends> {
        if !self.backends.contains_key(&precision) {
            let backend = precision.backend(self.reduction, &self.process.plan, self.target)?;
            self.backends.insert(precision, backend);
        }
        Ok(self.backends.get_mut(&precision).expect("the backend"))
    }

    /// The reduction backend of the following evaluations: NinjaRS (the
    /// default; refused typed on a TIR-only process, whose default is
    /// TIRust) or TIRust (every carrier; a build without it refuses typed,
    /// a family outside TIRust's capability when a batch would reduce on
    /// it). The choice
    /// is the caller's (m-lit PLAN §2.b.4: the SDK has the per-call
    /// backend, `lit-sdk`), never switched by the runtime.
    pub fn set_reduction(&mut self, reduction: Reduction) -> RusticolResult<()> {
        if reduction == Reduction::TiRust && !crate::tir::available() {
            return Err(RusticolError::unsupported_runtime_capability(
                "tirust",
                crate::runtime::TIRUST_NOT_BUILT,
            ));
        }
        if reduction == Reduction::NinjaRs {
            crate::runtime::ninjars_refusal(&self.process.plan)?;
        }
        if reduction != self.reduction {
            self.reduction = reduction;
            self.backends.clear();
            self.cross_backends.clear();
        }
        Ok(())
    }

    /// The reduction backend of the following evaluations.
    pub fn reduction(&self) -> Reduction {
        self.reduction
    }

    /// The target accuracy of the following evaluations (the relative
    /// accuracy of each point's result, `reduction_policy::DEFAULT_TARGET`
    /// by default; validated by `reduction_policy::target`): TIRust's
    /// method-choice acceptance follows it (`v2_tir::acceptance_of`), set on
    /// the built backends (their plans stay); NinjaRS has no setting it
    /// drives.
    pub fn set_target(&mut self, target: f64) -> RusticolResult<()> {
        let target = crate::reduction_policy::target(target).map_err(|refusal| match refusal {
            crate::reduction_policy::Refusal::Invalid(message)
            | crate::reduction_policy::Refusal::Unavailable(message) => invalid(message),
        })?;
        if target != self.target {
            self.target = target;
            for backend in self
                .backends
                .values_mut()
                .chain(self.cross_backends.values_mut())
            {
                backend.set_target(target);
            }
        }
        Ok(())
    }

    /// The target accuracy of the following evaluations.
    pub fn target(&self) -> f64 {
        self.target
    }

    /// One batch through the tree program, the executor and the assembly in
    /// the carrier of `zero` and the arithmetic `arith`
    /// (`plan_export.NativeExecutor.reduce`, `assemble._PointEvaluation.
    /// set_lanes` and `_assemble`, in the same operations and order).
    ///
    /// `check` re-reduces every lane in that frame (the stability checks'
    /// reduction check); with `COLLECT` (a compile-time choice: an
    /// evaluation without checks collects nothing) every successful point's
    /// lanes, re-test lanes, local components and reference values per
    /// record sector go to `collected` (`monitor`), a failed point's `None`.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_batch<T: Native, A: Lift<T> + induced::Squaring, const COLLECT: bool>(
        &mut self,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
        precision: V2Precision,
        zero: &T,
        arith: &A,
        check: Option<ReductionCheck>,
        collected: &mut Vec<Option<monitor::Collected<A::N>>>,
    ) -> RusticolResult<Vec<V2Point<A::N>>>
    where
        Complex<T>: Real + EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        if let Some(induced) = self.induced.clone() {
            if COLLECT || check.is_some() {
                return Err(invalid(
                    "the loop-induced stability checks are the induced monitor's",
                ));
            }
            return self.evaluate_induced(&induced, momenta, scales, mode, precision, zero, arith);
        }
        let poles = mode.poles();
        let bits = precision.bits()?;
        let lanes = precision.lanes(momenta.len());
        let incoming = self.incoming(momenta);
        let values = self.parameter_values()?;
        // The tree and closure stages run on the executor's pool (the
        // runtime's `threads` bound), not on the global one.
        let (bound, inputs) = self.process.fill(
            precision, lanes, &values, scales, &incoming, poles, None, zero,
        )?;
        let born = inputs
            .born_squared
            .ok_or_else(|| invalid("the tree program forms no |A0|²"))?;
        let counterterms = inputs
            .counterterms
            .ok_or_else(|| invalid("the tree program forms no counterterm components"))?;
        let batch = v2_batch(
            incoming.clone(),
            scales,
            bound,
            inputs.block_rows,
            inputs.born,
        );
        let options = V2Options {
            poles: mode.lanes(),
            contraction: contraction(&self.artifact.metadata.color_contraction),
            check,
            ..V2Options::default()
        };
        self.backend(precision)?;
        let backend = self.backends.get_mut(&precision).expect("the backend");
        let results = self
            .process
            .evaluate(lanes, &batch, zero, backend, options)?;
        // The companion's seeds take the FFT where certified, never required
        // (`plan_export.NativeExecutor.scaleless`).
        let options = V2Options {
            contraction: match options.contraction {
                Contraction::Direct => Contraction::Direct,
                _ => Contraction::Auto,
            },
            ..options
        };
        // A model with pole regulators: the split selections also form the
        // counterterm and field components with the IR regulator at 0, the
        // counterterms' UV parts (`plan_export.NativeExecutor.reduce`).
        let counterterms_uv =
            match (self.regulated_values(&values), mode.split()) {
                (Some(regulated), true) => {
                    let (_, inputs) = self.process.fill(
                        precision, lanes, &regulated, scales, &incoming, true, None, zero,
                    )?;
                    Some(inputs.counterterms.ok_or_else(|| {
                        invalid("the tree program forms no counterterm components")
                    })?)
                }
                _ => None,
            };
        // The split selections also run the companion of the certified-
        // scaleless families on its own tree inputs (schedule B), for the UV
        // residues of its lanes (`plan_export.NativeExecutor.reduce`).
        let twin_results = if mode.split() && self.companion()?.is_some() {
            let twin = self.scaleless.as_mut().expect("the companion is built");
            let (bound, inputs) = twin.fill(
                precision, lanes, &values, scales, &incoming, true, None, zero,
            )?;
            let batch = v2_batch(incoming, scales, bound, inputs.block_rows, inputs.born);
            let backend = self.backends.get_mut(&precision).expect("the backend");
            let twin_results = twin.evaluate(lanes, &batch, zero, backend, options)?;
            if twin_results.len() != results.len() {
                return Err(invalid("the companion evaluated another batch"));
            }
            Some(twin_results)
        } else {
            None
        };
        let width = if poles { 3 } else { 1 };
        let lane_width = mode.lanes().width();
        let factor = loop_factor(arith, self.record.average);
        let two = arith.int(2);
        // 2 * (factor * sum).real
        let bare_of = |sum: &C<A::N>| arith.mul(&two, &cmul(arith, &factor, sum).re);
        let sectors = self.record.sectors.len();
        let outputs: Vec<(String, usize, i32)> = self
            .record
            .trees
            .counterterm_outputs
            .iter()
            .map(|(component, sector, power)| {
                let wanted = canonical(sector);
                let s = self
                    .record
                    .sectors
                    .iter()
                    .position(|(o, _)| canonical(o) == wanted)
                    .ok_or_else(|| invalid("a counterterm output's sector is not in the record"))?;
                Ok((component.clone(), s, *power))
            })
            .collect::<RusticolResult<_>>()?;
        let zero_c = C::new(arith.zero(), arith.zero());
        // `numbers.real(0)`: the default of an absent component.
        let real_zero = arith.real_int(0);
        let mut out = Vec::with_capacity(results.len());
        for (index, result) in results.into_iter().enumerate() {
            let twin = twin_results.as_ref().map(|twin| &twin[index]);
            let mut failures = result.failures;
            if let Some(twin) = twin {
                failures.extend(twin.failures.iter().cloned());
            }
            if !failures.is_empty() {
                if COLLECT {
                    collected.push(None);
                }
                out.push(Err(failures));
                continue;
            }
            // Per sector the lanes' combined Laurent (and in a split
            // selection their UV Laurent `(finite, 1/ε_UV, 0)`), summed in
            // the executor's order, the companion's lanes after the
            // process's (`set_lanes`: their combined integrals vanish, only
            // their UV residues are kept).
            let mut sums: Vec<Option<Vec<C<A::N>>>> = vec![None; sectors];
            let mut uv_sums: Vec<Option<[C<A::N>; 3]>> = vec![None; sectors];
            let mut lanes_of: Vec<(&LaneKey, &[Complex<T>], bool)> = result
                .families
                .iter()
                .map(|(key, value)| (key, value.as_slice(), false))
                .collect();
            if let Some(twin) = twin {
                lanes_of.extend(
                    twin.families
                        .iter()
                        .map(|(key, value)| (key, value.as_slice(), true)),
                );
            }
            for (key, value, vanishing) in lanes_of {
                if key.column >= sectors || value.len() != lane_width {
                    return Err(invalid("an executor lane outside the record's sectors"));
                }
                let lifted = value
                    .iter()
                    .map(|z| arith.complex(z, bits))
                    .collect::<RusticolResult<Vec<_>>>()?;
                let terms = sums[key.column].get_or_insert_with(|| vec![zero_c.clone(); width]);
                for (k, term) in terms.iter_mut().enumerate() {
                    let value = if vanishing { &zero_c } else { &lifted[k] };
                    *term = cadd(arith, term, value);
                }
                if mode.split() {
                    let finite = if vanishing { &zero_c } else { &lifted[0] };
                    let terms = uv_sums[key.column]
                        .get_or_insert_with(|| [zero_c.clone(), zero_c.clone(), zero_c.clone()]);
                    for (term, value) in terms.iter_mut().zip([finite, &lifted[3], &zero_c]) {
                        *term = cadd(arith, term, value);
                    }
                }
            }
            // (counterterm component, sector, power) -> value.
            let table = |values: &[T]| -> RusticolResult<BTreeMap<(&str, usize, i32), A::N>> {
                outputs
                    .iter()
                    .zip(values)
                    .map(|((component, s, power), value)| {
                        Ok(((component.as_str(), *s, *power), arith.real(value, bits)?))
                    })
                    .collect()
            };
            let local = table(&counterterms[index])?;
            let local_uv = counterterms_uv
                .as_ref()
                .map(|values| table(&values[index]))
                .transpose()?;
            let value_of =
                |table: &BTreeMap<(&str, usize, i32), A::N>, name: &str, s: usize, k: usize| {
                    table
                        .get(&(name, s, LAURENT_POWERS[k]))
                        .cloned()
                        .unwrap_or_else(|| real_zero.clone())
                };
            let laurent = |table: &BTreeMap<(&str, usize, i32), A::N>, name: &str, s: usize| {
                [
                    value_of(table, name, s, 0),
                    value_of(table, name, s, 1),
                    value_of(table, name, s, 2),
                ]
            };
            let mut virtual_values = Vec::with_capacity(self.sector_index.len());
            for &s in &self.sector_index {
                let bare = |k: usize| {
                    sums[s]
                        .as_ref()
                        .map_or_else(|| real_zero.clone(), |terms| bare_of(&terms[k]))
                };
                if mode.split() {
                    let bare_uv = |k: usize| {
                        uv_sums[s]
                            .as_ref()
                            .map_or_else(|| real_zero.clone(), |terms| bare_of(&terms[k]))
                    };
                    let counterterms_uv = local_uv.as_ref().map(|table| {
                        (
                            laurent(table, "r2", s),
                            laurent(table, "uv_local", s),
                            laurent(table, FIELD_COMPONENT, s),
                        )
                    });
                    let selected = split_virtual(
                        arith,
                        mode,
                        [bare(0), bare(1), bare(2)],
                        [bare_uv(0), bare_uv(1), bare_uv(2)],
                        laurent(&local, "r2", s),
                        laurent(&local, "uv_local", s),
                        laurent(&local, FIELD_COMPONENT, s),
                        counterterms_uv,
                    );
                    virtual_values.push(selected.to_vec());
                    continue;
                }
                let mut values = Vec::with_capacity(width);
                for k in 0..width {
                    let uv = arith.add(
                        &value_of(&local, "uv_local", s, k),
                        &value_of(&local, FIELD_COMPONENT, s, k),
                    );
                    values
                        .push(arith.add(&arith.add(&bare(k), &uv), &value_of(&local, "r2", s, k)));
                }
                virtual_values.push(values);
            }
            let born_values = self
                .born_index
                .iter()
                .map(|found| match found {
                    Some(k) => arith.real(&born[index][*k], bits),
                    // `_zero(carrier)`: 0.0 or Decimal(0).
                    None => Ok(arith.zero()),
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            if COLLECT {
                // What the point's stability budgets read
                // (`_PointEvaluation.families`, `retest_families` and
                // `result.components`, per record sector).
                if mode.split() {
                    return Err(invalid(
                        "the native stability checks serve the pole selections all and none",
                    ));
                }
                let lift = |lanes: &[(LaneKey, Vec<Complex<T>>)]| {
                    lanes
                        .iter()
                        .map(|(key, value)| {
                            let lifted = value
                                .iter()
                                .map(|z| arith.complex(z, bits))
                                .collect::<RusticolResult<Vec<_>>>()?;
                            Ok((*key, key.column, lifted))
                        })
                        .collect::<RusticolResult<Vec<_>>>()
                };
                let truncated =
                    |values: [A::N; 3]| -> Vec<A::N> { values.into_iter().take(width).collect() };
                let components = ["r2", "uv_local", FIELD_COMPONENT]
                    .map(|name| {
                        (0..sectors)
                            .map(|s| truncated(laurent(&local, name, s)))
                            .collect::<Vec<_>>()
                    })
                    .to_vec();
                let reference = (0..sectors)
                    .map(|s| {
                        (0..width)
                            .map(|k| {
                                let bare = sums[s]
                                    .as_ref()
                                    .map_or_else(|| real_zero.clone(), |terms| bare_of(&terms[k]));
                                let uv = arith.add(
                                    &value_of(&local, "uv_local", s, k),
                                    &value_of(&local, FIELD_COMPONENT, s, k),
                                );
                                arith.add(&arith.add(&bare, &uv), &value_of(&local, "r2", s, k))
                            })
                            .collect()
                    })
                    .collect();
                collected.push(Some(monitor::Collected {
                    lanes: lift(&result.families)?,
                    retest: result.check.as_deref().map(lift).transpose()?,
                    local: components,
                    reference,
                }));
            }
            out.push(Ok((born_values, virtual_values)));
        }
        Ok(out)
    }
}

impl V2Runtime {
    /// One batch of a loop-induced process (`plan_export.NativeExecutor.
    /// reduce` of an amplitude-mode plan and `induced.assemble`): the tree
    /// program's inputs and counterterm kets, the executor's bare loop
    /// amplitudes per (loop order, row, basis element), and |A|² per sector
    /// ([`induced::Induced::square_vectors`] with G over every row); no Born
    /// values (Q36: the Born fields are empty).
    #[allow(clippy::too_many_arguments)]
    fn evaluate_induced<T: Native, A: Lift<T> + induced::Squaring>(
        &mut self,
        induced: &induced::Induced,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
        precision: V2Precision,
        zero: &T,
        arith: &A,
    ) -> RusticolResult<Vec<V2Point<A::N>>>
    where
        Complex<T>: Real + EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        // `induced.assemble`: G in the carrier, every plan row.
        let gram = induced.carrier(arith, induced.metric())?;
        let rows = induced.plan_rows();
        let average = self.record.average;
        let sector_index = self.sector_index.clone();
        self.induced_points(
            induced,
            momenta,
            scales,
            mode,
            precision,
            zero,
            arith,
            |_, vectors, width| {
                let squared =
                    induced.square_vectors(arith, vectors, &rows, &gram, width, average)?;
                Ok((
                    Vec::new(),
                    sector_index.iter().map(|&s| squared[s].clone()).collect(),
                ))
            },
        )
    }
}

fn order_pairs(orders: &[(String, u32)]) -> Orders {
    orders
        .iter()
        .map(|(name, power)| (name.clone(), i64::from(*power)))
        .collect()
}

fn program_legs(artifact: &LoadedLoopV2Artifact) -> usize {
    artifact.program.legs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orders(pairs: &[(&str, i64)]) -> Orders {
        pairs.iter().map(|(n, p)| ((*n).to_owned(), *p)).collect()
    }

    #[test]
    fn orders_compare_as_sets_without_zero_powers() {
        assert_eq!(
            canonical(&orders(&[("QED", 0), ("QCD", 2), ("NP", 1)])),
            orders(&[("NP", 1), ("QCD", 2)])
        );
        assert!(canonical(&orders(&[("QCD", 0)])).is_empty());
        // The plan's declared (unsigned) orders meet the record's the same way.
        let declared = vec![("QED".to_owned(), 0_u32), ("QCD".to_owned(), 4)];
        assert_eq!(
            canonical(&order_pairs(&declared)),
            canonical(&orders(&[("QCD", 4)]))
        );
    }

    #[test]
    fn native_record_reads_the_assembly_fields_and_keeps_the_rest() {
        // `native_process.NativeProcess`'s record: the fields the assembly
        // reads; any other key (the tree program's own, observables) passes.
        let json = r#"{
            "schema": 2,
            "legs": [[1, true, 21], [2, true, 21], [3, false, 21], [4, false, 21]],
            "sectors": [[[["QCD", 6]], [[[["QCD", 2]], [["QCD", 4]]]]]],
            "average": 256,
            "fixed_parameters": {"MB": "0"},
            "trees": {
                "squared_orders": [[["QCD", 4], ["QED", 0]]],
                "counterterm_outputs": [["r2", [["QCD", 6]], 0], ["uv_local", [["QCD", 6]], -1]],
                "program": {"anything": [1, 2]}
            },
            "observables": null
        }"#;
        let record: NativeRecord = serde_json::from_str(json).unwrap();
        // A model without pole regulators records none.
        assert!(record.pole_regulators.is_none());
        let regulated = json.replace(
            r#""observables": null"#,
            r#""observables": null, "pole_regulators": {"uv": "epsUV", "ir": "epsIR"}"#,
        );
        let regulated: NativeRecord = serde_json::from_str(&regulated).unwrap();
        assert_eq!(regulated.pole_regulators.unwrap().ir, "epsIR");
        assert_eq!(record.schema, NATIVE_PROCESS_SCHEMA);
        assert_eq!(record.legs.len(), 4);
        assert!(record.legs[0].1 && !record.legs[3].1);
        assert_eq!(record.sectors.len(), 1);
        assert_eq!(canonical(&record.sectors[0].0), orders(&[("QCD", 6)]));
        assert_eq!(record.average, 256);
        assert_eq!(record.fixed_parameters["MB"], "0");
        assert_eq!(
            canonical(&record.trees.squared_orders[0]),
            orders(&[("QCD", 4)])
        );
        assert_eq!(record.trees.counterterm_outputs[1].0, "uv_local");
        assert_eq!(record.trees.counterterm_outputs[1].2, -1);
        // A field the assembly needs is required.
        let missing = json.replace(r#""average": 256,"#, "");
        assert!(serde_json::from_str::<NativeRecord>(&missing).is_err());
    }

    #[test]
    fn split_selections_split_every_component_as_the_python_api() {
        // Exactly representable values, so the identities below are exact.
        let bare = [10.0, -6.0, 4.0];
        let bare_uv = [10.0, 3.0, 0.0];
        let r2 = [0.5, 0.0, 0.0];
        let uv_local = [1.0, -2.0, 0.0];
        let fields = [0.25, 0.75, 0.0];
        let finite = bare[0] + (uv_local[0] + fields[0]) + r2[0];
        let uv = split_virtual(
            &Binary64,
            V2Poles::Uv,
            bare,
            bare_uv,
            r2,
            uv_local,
            fields,
            None,
        );
        let ir = split_virtual(
            &Binary64,
            V2Poles::Ir,
            bare,
            bare_uv,
            r2,
            uv_local,
            fields,
            None,
        );
        // The finite part is the renormalised virtual's in every selection.
        assert_eq!(uv[0], finite);
        assert_eq!(ir[0], finite);
        // "uv": the bare loop's UV poles (the double pole is infrared).
        assert_eq!(&uv[1..], &[3.0, 0.0]);
        // "ir": the renormalised virtual's poles less their UV parts; the
        // field factors make the renormalised virtual UV finite, so they are
        // its combined poles: bare + uv_local + fields + r2.
        let combined = [
            bare[1] + (uv_local[1] + fields[1]) + r2[1],
            bare[2] + (uv_local[2] + fields[2]) + r2[2],
        ];
        assert_eq!(&ir[1..], &combined);
        // A model with pole regulators: the counterterms' UV parts are given
        // (the tree program at the IR regulator 0), so the renormalised
        // virtual's IR poles are bare_IR + CT_IR, with no remainder.
        let given = Some(([0.5, 0.0, 0.0], [1.0, -1.5, 0.0], [0.25, -1.0, 0.0]));
        let uv = split_virtual(
            &Binary64,
            V2Poles::Uv,
            bare,
            bare_uv,
            r2,
            uv_local,
            fields,
            given,
        );
        let ir = split_virtual(
            &Binary64,
            V2Poles::Ir,
            bare,
            bare_uv,
            r2,
            uv_local,
            fields,
            given,
        );
        assert_eq!(uv, [finite, 3.0, 0.0]);
        let parts = [
            (bare[1] - 3.0) + ((uv_local[1] + 1.5) + (fields[1] + 1.0)) + r2[1],
            (bare[2] - 0.0) + ((uv_local[2] - 0.0) + (fields[2] - 0.0)) + r2[2],
        ];
        assert_eq!(ir, [finite, parts[0], parts[1]]);
        // A finite-only or combined selection is not split here.
        assert!(!V2Poles::All.split() && !V2Poles::None.split());
        assert!(V2Poles::Uv.split() && V2Poles::Ir.split());
        assert_eq!(V2Poles::Ir.lanes(), Poles::Split);
        assert_eq!(V2Poles::None.lanes(), Poles::None);
        assert!(!V2Poles::None.poles() && V2Poles::Uv.poles());
    }

    #[test]
    fn parameter_updates_are_validated_whole_as_the_python_api() {
        use rusticol_core::RusticolErrorKind;
        let externals: BTreeSet<String> = ["MT", "WT", "MU_R", "aS"]
            .into_iter()
            .map(str::to_owned)
            .collect();
        let fixed: BTreeMap<String, String> = [("WT".to_owned(), "0".to_owned())].into();
        let check = |values: &[(&str, &str)]| {
            checked_parameters(
                values.iter().copied(),
                &externals,
                Some("MU_R"),
                &fixed,
                &BTreeSet::new(),
            )
        };
        // Accepted: known names, a fixed parameter at its fixed value (any
        // exact decimal spelling, as Python's `Decimal`), by name.
        let accepted = check(&[("aS", " 1_18e-3 "), ("WT", "-0.0"), ("MT", "173.5")]).unwrap();
        assert_eq!(
            accepted,
            vec![
                ("MT".to_owned(), parse_rational("347/2").unwrap()),
                ("WT".to_owned(), Rational::from(0)),
                ("aS".to_owned(), parse_rational("0.118").unwrap()),
            ]
        );
        // Refused, typed, in the Python API's words; one refusal refuses the
        // whole mapping.
        for (values, message) in [
            (
                vec![("MT", "174"), ("WX", "1")],
                "unknown external parameter 'WX'",
            ),
            (
                vec![("MU_R", "100")],
                "'MU_R' is the renormalisation scale: pass it per point as mu_r2",
            ),
            (
                vec![("WT", "1.4915")],
                "parameter 'WT' was fixed to 0 at generation ([loop.preparation] \
                 fixed_parameters); regenerate the artifact to change it",
            ),
        ] {
            let error = check(&values).unwrap_err();
            assert_eq!(error.kind(), RusticolErrorKind::ModelParameter);
            assert_eq!(error.message(), message);
        }
        // Values are read first and names in sorted order, as in Python: a
        // fraction is not a decimal number, an infinity not finite, and of
        // two refused names the first by name is reported.
        for (values, message) in [
            (
                vec![("aS", "59/500")],
                "parameter 'aS' real part is not a decimal number",
            ),
            (
                vec![("MT", "one")],
                "parameter 'MT' real part is not a decimal number",
            ),
            (
                vec![("MT", "-Infinity")],
                "parameter 'MT' real part must be finite",
            ),
            (
                vec![("aS", "NaN"), ("WX", "1")],
                "parameter 'aS' real part must be finite",
            ),
            (
                vec![("WT", "1.4915"), ("MU_R", "100")],
                "'MU_R' is the renormalisation scale: pass it per point as mu_r2",
            ),
        ] {
            let error = check(&values).unwrap_err();
            assert_eq!(error.kind(), RusticolErrorKind::ModelParameter);
            assert_eq!(error.message(), message);
        }
        // Without a scale parameter (a program that reads none) the name is
        // just a model parameter.
        let none = BTreeSet::new();
        assert!(checked_parameters([("MU_R", "100")], &externals, None, &fixed, &none).is_ok());
        // A scheme constant the model preparation fixed is named as such.
        let prepared: BTreeSet<String> = ["WT".to_owned()].into();
        let error =
            checked_parameters([("WT", "1")], &externals, None, &fixed, &prepared).unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::ModelParameter);
        assert_eq!(
            error.message(),
            "parameter 'WT' was fixed to 0 at generation by the model preparation \
             (a scheme constant, e.g. of the colour declaration); it is not a \
             run-time parameter"
        );
        // Python's reader: ASCII text only (no other digits or whitespace),
        // its ASCII whitespace (`\x1c`-`\x1f` included) and underscores
        // ignored; an empty name is an invalid argument (Python's TypeError).
        for text in ["\u{661}", "\u{2003}1", "1\u{a0}"] {
            assert_eq!(
                check(&[("MT", text)]).unwrap_err().message(),
                "parameter 'MT' real part is not a decimal number"
            );
        }
        let spaced = check(&[("MT", "\x1c_17_3 \x0b\x1f")]).unwrap();
        assert_eq!(spaced[0].1, Rational::from(173));
        let error = check(&[("", "1"), ("MT", "one")]).unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::InvalidArgument);
        assert_eq!(
            error.message(),
            "model parameter names must be nonempty strings"
        );
    }

    #[test]
    fn a_path_without_an_artifact_is_refused() {
        let missing = std::env::temp_dir().join(format!(
            "rusticol-v2-runtime-no-artifact-{}",
            std::process::id()
        ));
        assert!(!missing.exists());
        assert!(is_v2_artifact(&missing).is_err());
        assert!(V2Runtime::open(&missing, None, Some(1)).is_err());
    }
}
