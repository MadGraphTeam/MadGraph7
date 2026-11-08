// SPDX-License-Identifier: 0BSD
//! Source-only standalone virtual SDK. The safe Rust and C interfaces share the
//! existing native executor; neither links Python or implements loop reduction.
mod correlated;
mod ffi;
pub use correlated::{
    Complex64, CorrelatedEvaluation, CorrelatedShape, DecimalComplexValues, SpinSources,
};
pub use ffi::*;
use rusticol_core::loop_artifact::{LOOP_RUNTIME_CAPABILITY, LOOP_V2_RUNTIME_CAPABILITY};
use rusticol_core::loop_execution::LoopInputs;
pub use rusticol_core::source_bindings::SourceOverrides;
use rusticol_core::{RusticolError, RusticolResult};
use rusticol_loop_reduction::reduction_policy::{self, Choice, Refusal};
use rusticol_loop_reduction::runtime::Reduction;
#[cfg(test)]
use rusticol_loop_reduction::runtime::{Batch, Data};
use rusticol_loop_reduction::v2_runtime::cross_reduction;
use rusticol_loop_reduction::v2_runtime::{
    C, ColumnPoint, Decimal, Rational, V2Point, V2Poles, V2Precision, V2Runtime, decimal_parameter,
    is_v2_artifact, named_parameters,
};
pub use rusticol_loop_reduction::v2_stability::{
    Estimate, Rotation, StabilityChecks, default_rotation,
};
use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

fn invalid(message: impl ToString) -> RusticolError {
    RusticolError::invalid_argument(message.to_string())
}
fn serialization(error: impl ToString) -> RusticolError {
    RusticolError::serialization(error.to_string())
}
fn product(values: &[usize]) -> RusticolResult<usize> {
    values.iter().try_fold(1_usize, |a, b| {
        a.checked_mul(*b)
            .ok_or_else(|| invalid("SDK dimensions overflow"))
    })
}

/// Point statuses are independent of function/ABI errors. A failed point has no
/// numerical row. Ordinary success does not establish numerical accuracy.
/// Plan v4 runtimes return `AccuracyNotEstablished` or `Failed`; `Verified`
/// and `Rescued` keep their values but are never returned (the stability
/// rescue is the Python API's).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum PointStatus {
    Success = 0,
    Verified = 1,
    Rescued = 2,
    Failed = 3,
    AccuracyNotEstablished = 4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Shape {
    pub external_count: usize,
    pub born_orders: usize,
    pub virtual_orders: usize,
    pub helicities: usize,
    pub born_color_components: usize,
    pub virtual_color_components: usize,
    pub epsilon_count: usize,
    pub resolved_available: u32,
}

/// Owned result buffers remain valid after clearing or freeing their runtime.
/// Resolved layout is point/order/helicity/colour-component/(epsilon).
pub struct Evaluation {
    pub requested_point_ids: Vec<u64>,
    pub successful_point_ids: Vec<u64>,
    pub statuses: Vec<PointStatus>,
    pub born: Vec<f64>,
    pub virtual_values: Vec<f64>,
    pub shape: Shape,
    pub resolved: bool,
    pub diagnostics_json: String,
    /// Double-double and arbitrary-precision evaluations: the exact values
    /// (the Python API's `Decimal` results) as decimal literals, in the
    /// layout of `born` and `virtual_values` (absent poles `None`); `born`
    /// and `virtual_values` then hold their nearest binary64 values.
    pub decimal: Option<DecimalValues>,
}

/// The exact values of an extended-precision evaluation: decimal literals
/// equal to the Python API's `Decimal` results, at `digits` working digits
/// of `arithmetic`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecimalValues {
    pub arithmetic: &'static str,
    pub digits: u32,
    pub born: Vec<String>,
    #[serde(rename = "virtual")]
    pub virtual_values: Vec<Option<String>>,
}

/// The arithmetic of later evaluations, the Python API's `arithmetic` and
/// `precision` (`Runtime.load(..., arithmetic=, precision=)`): binary64
/// (the default), double-double, or `digits` decimal digits of arbitrary
/// precision. The extended carriers run the native executor in their
/// arithmetic (the generic reduction engine) and assemble its outputs in
/// Python's `Decimal` arithmetic at the working digits, exactly as the
/// Python API: their values equal the Python API's `Decimal` results.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Arithmetic {
    #[default]
    Double,
    DoubleDouble,
    Arbitrary(u32),
}

impl Arithmetic {
    /// `"double"`, `"double-double"` or `"arbitrary"` with `digits` decimal
    /// digits (positive; the other carriers take none).
    pub fn parse(name: &str, digits: Option<u32>) -> RusticolResult<Self> {
        match (name, digits) {
            ("double", None) => Ok(Self::Double),
            ("double-double", None) => Ok(Self::DoubleDouble),
            ("arbitrary", Some(digits)) if digits > 0 => Ok(Self::Arbitrary(digits)),
            ("arbitrary", _) => Err(invalid(
                "arbitrary precision needs a positive decimal digit count",
            )),
            ("double" | "double-double", Some(_)) => {
                Err(invalid(format!("{name} arithmetic takes no digits")))
            }
            _ => Err(invalid(format!(
                "arithmetic must be \"double\", \"double-double\" or \"arbitrary\", not {name:?}"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Double => "double",
            Self::DoubleDouble => "double-double",
            Self::Arbitrary(_) => "arbitrary",
        }
    }
    /// The requested digits of arbitrary precision.
    pub fn digits(self) -> Option<u32> {
        match self {
            Self::Arbitrary(digits) => Some(digits),
            _ => None,
        }
    }
    fn v2(self) -> V2Precision {
        match self {
            Self::Double => V2Precision::Double,
            Self::DoubleDouble => V2Precision::DoubleDouble,
            Self::Arbitrary(digits) => V2Precision::Arbitrary(digits),
        }
    }
    /// The decimal digits of the assembly (`Precision.working_digits`).
    pub fn working_digits(self) -> u32 {
        self.v2().working_digits()
    }
}

/// The runtime behind an SDK handle: a plan v4 process with native point
/// inputs (`v2_runtime`, WP3.1 I6). The plan v1-v3 runtime of the removed
/// legacy one-loop engine is refused at open (`LEGACY_ENGINE_REMOVED`).
enum Engine {
    V4(Box<V2Runtime>),
}

/// The refusal of a resolved or correlated evaluation of a process stored
/// without its observable columns (or with a forced seed FFT).
const NO_OBSERVABLES: &str = "this plan v4 process serves no resolved or correlated \
    evaluations";

/// The refusal of a loop artifact of the removed legacy engine (the Python
/// API's `LEGACY_LOOP_ARTIFACT_REMOVED`).
pub(crate) const LEGACY_ENGINE_REMOVED: &str = "this artifact holds legacy-engine \
    virtual processes (rusticol.loop-runtime.v3): the legacy one-loop engine was \
    removed; v2 (the shared-recursion engine) is the only one-loop engine; \
    regenerate the artifact";

/// The R2 strategy of a plan v4 process, as the Python API's
/// `runtime.virtual.v2_r2_strategy`: v4 artifacts carry the model-supplied
/// R2 rules only, so "auto" (the default) and "model" select them and
/// "direct" is refused.
fn v4_r2_strategy(strategy: &str) -> RusticolResult<()> {
    match strategy {
        "auto" | "model" => Ok(()),
        "direct" => Err(RusticolError::compatibility(
            "v2 virtual artifacts carry model-supplied R2 only",
        )),
        other => Err(invalid(format!(
            "r2_strategy must be \"auto\", \"model\" or \"direct\", not {other:?}"
        ))),
    }
}

/// The poles an ordinary virtual returns (REQUIREMENTS_ADDENDUM I, the
/// API's `poles` option): every Laurent coefficient (`All`), the finite part
/// alone (`None`, whose pole-only work a plan v4 runtime skips natively; the
/// pole entries are then quiet NaNs, absent), the bare loop's UV poles
/// (`Uv`) or the renormalised virtual's IR poles (`Ir`), the split
/// selections, which run the stored companion of the certified-scaleless
/// families (`v2_runtime::V2Poles`). These are the four selections of the
/// Python API (`PoleSelection`: "all", "none", "uv", "ir"). The finite part
/// is the same in every selection. Plan v1-v3 runtimes serve `All` only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Poles {
    #[default]
    All,
    None,
    Uv,
    Ir,
}

impl Poles {
    pub fn parse(text: &str) -> RusticolResult<Self> {
        match text {
            "all" => Ok(Self::All),
            "none" => Ok(Self::None),
            "uv" => Ok(Self::Uv),
            "ir" => Ok(Self::Ir),
            _ => Err(invalid(format!(
                "poles must be \"all\", \"none\", \"uv\" or \"ir\", not {text:?}"
            ))),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::None => "none",
            Self::Uv => "uv",
            Self::Ir => "ir",
        }
    }
    fn v2(self) -> V2Poles {
        match self {
            Self::All => V2Poles::All,
            Self::None => V2Poles::None,
            Self::Uv => V2Poles::Uv,
            Self::Ir => V2Poles::Ir,
        }
    }
}

pub struct Runtime {
    engine: Engine,
    shape: Shape,
    poles: Poles,
    arithmetic: Arithmetic,
    /// The requested reduction backend of later evaluations ("auto",
    /// "ninjars" or "tirust"; the artifact's generation default until
    /// [`Runtime::set_reduction`]).
    reduction: String,
    /// The stability checks of later binary64 evaluations (none by default:
    /// no estimate, no extra cost; [`Runtime::set_stability_checks`]).
    checks: StabilityChecks,
    color_components: Vec<String>,
    history_ids: Vec<String>,
    spin_classes: Vec<Vec<usize>>,
    /// The stored observable columns: the resolved helicity rows (1-based
    /// legs), their colour component ids, the squared Born orders of the
    /// observable outputs, or the reason they are unavailable.
    observables: Result<Observables, String>,
}

/// The axes of a process's stored observable columns.
#[derive(Clone, Debug)]
struct Observables {
    rows: Vec<Vec<(usize, i32)>>,
    colour_ids: Vec<String>,
    born_orders: usize,
}

impl Runtime {
    pub fn open(
        path: impl AsRef<Path>,
        process: Option<&str>,
        parameters: Option<&SourceOverrides>,
        r2_strategy: &str,
    ) -> RusticolResult<Self> {
        if is_v2_artifact(path.as_ref())? {
            return Self::open_v4(path, process, parameters, r2_strategy);
        }
        let verified = rusticol_core::VerifiedArtifact::open(path.as_ref())?;
        let capabilities = &verified.manifest().runtime.required_runtime_capabilities;
        if capabilities.iter().any(|c| c == LOOP_RUNTIME_CAPABILITY) {
            return Err(RusticolError::compatibility(LEGACY_ENGINE_REMOVED));
        }
        Err(RusticolError::compatibility(format!(
            "the loop SDK opens plan v4 virtual artifacts ({LOOP_V2_RUNTIME_CAPABILITY}) \
             only; this artifact requires {capabilities:?}"
        )))
    }
    /// A plan v4 process: its stored native program evaluates every point
    /// (no Python, no generation; `rusticol_loop_reduction::v2_runtime`).
    /// `r2_strategy` "auto" (the C ABI's default) or "model" selects the
    /// artifact's model R2; `parameters` are validated as a whole as the
    /// Python API's `Runtime.load(..., model_parameters=...)` does (a
    /// refusal opens nothing).
    fn open_v4(
        path: impl AsRef<Path>,
        process: Option<&str>,
        parameters: Option<&SourceOverrides>,
        r2_strategy: &str,
    ) -> RusticolResult<Self> {
        v4_r2_strategy(r2_strategy)?;
        let mut native = V2Runtime::open(path, process, None)?;
        if let Some(values) = parameters {
            native.set_parameters(real_overrides(values)?)?;
        }
        let accuracy = native.metadata().color_accuracy.clone();
        let reduction = native.artifact_reduction().to_owned();
        // The stored observable columns (correlated, spin-correlated and
        // resolved outputs), when the process has them.
        let observables = (|| -> RusticolResult<_> {
            let (rows, basis, resolution) = native.resolved_axes()?;
            let colour_ids = if resolution == "helicity-colour" {
                (0..basis).map(|b| format!("basis-{b}")).collect()
            } else {
                vec![accuracy.clone()]
            };
            Ok((
                Observables {
                    rows: rows
                        .into_iter()
                        .map(|row| row.into_iter().map(|(leg, h)| (leg + 1, h)).collect())
                        .collect(),
                    colour_ids,
                    born_orders: native.column_born_axis()?.len(),
                },
                native.correlation_histories()?,
                native.spin_correlation_classes()?,
            ))
        })();
        let (observables, history_ids, spin_classes) = match observables {
            Ok((observables, histories, classes)) => (Ok(observables), histories, classes),
            Err(error) => (Err(error.to_string()), vec![], vec![]),
        };
        let (helicities, colours) = match &observables {
            Ok(axes) => (axes.rows.len(), axes.colour_ids.len()),
            Err(_) => (native.helicities(), 1),
        };
        // A loop-induced process has no Born (Q36): zero Born orders and
        // colour components.
        let born_colours = if native.loop_induced() { 0 } else { colours };
        let shape = Shape {
            external_count: native.external_count(),
            born_orders: native.born_orders(),
            virtual_orders: native.sectors(),
            helicities,
            born_color_components: born_colours,
            virtual_color_components: colours,
            epsilon_count: 3,
            resolved_available: u32::from(observables.is_ok()),
        };
        Ok(Self {
            color_components: vec![accuracy],
            engine: Engine::V4(Box::new(native)),
            shape,
            poles: Poles::All,
            arithmetic: Arithmetic::Double,
            reduction,
            checks: StabilityChecks::default(),
            history_ids,
            spin_classes,
            observables,
        })
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }
    /// The pole selection of later evaluations (default all): "all",
    /// "none", "uv" or "ir", the Python API's four (`Poles`); a selection
    /// the process does not serve (a loop-induced process serves "all" and
    /// "none", m-lit PLAN §2.a.7) is refused with a typed `Compatibility`
    /// error in the Python API's words, and nothing changes.
    pub fn set_poles(&mut self, poles: Poles) -> RusticolResult<()> {
        let Engine::V4(native) = &self.engine;
        native.check_poles(poles.v2())?;
        self.poles = poles;
        Ok(())
    }
    /// The reduction backend of later evaluations, the Python API's
    /// `reduction=`: "auto", "ninjars" or "tirust" (the one policy,
    /// `rusticol_loop_reduction::reduction_policy`). A backend this native
    /// library does not build is refused here (typed `Compatibility`, the
    /// Python API's `ReductionUnavailable`), one that cannot serve a later
    /// call's carrier by that call; nothing changes on a refusal.
    pub fn set_reduction(&mut self, reduction: &str) -> RusticolResult<()> {
        let served = reduction_policy::served();
        match reduction {
            "auto" => {}
            "ninjars" | "tirust" if served.contains_key(reduction) => {}
            "ninjars" | "tirust" => {
                let route =
                    reduction_policy::route(reduction_policy::SUMMED).expect("the summed route");
                let carrier = self.arithmetic.name();
                return match reduction_policy::resolve(
                    reduction,
                    self.contribution(),
                    carrier,
                    &served,
                    self.admitted().as_ref(),
                    &route,
                ) {
                    Err(Refusal::Unavailable(message)) => {
                        Err(RusticolError::compatibility(message))
                    }
                    Err(Refusal::Invalid(message)) => Err(invalid(message)),
                    Ok(_) => unreachable!("an unbuilt backend serves no carrier"),
                };
            }
            other => {
                return Err(invalid(format!(
                    "unknown reduction backend '{other}': reduction must be \
                     \"auto\", \"ninjars\" or \"tirust\""
                )));
            }
        }
        self.reduction = reduction.to_owned();
        Ok(())
    }
    /// The requested reduction backend of later evaluations.
    pub fn reduction(&self) -> &str {
        &self.reduction
    }
    /// The stability checks of later evaluations, the Python API's
    /// `stability_checks=` (`VirtualStabilityChecks`): rotations, the
    /// reduction-direction check (rescaled: masters recomputed) and the
    /// cross-backend check (`StabilityChecks`). The default runs none: no
    /// estimate and nothing else runs. Binary64 ordinary evaluations (of a
    /// Born-interfered process in the pole selections "all" and "none", of a
    /// loop-induced one in both of its selections) return each successful
    /// point's estimate in their diagnostics
    /// (`"stability"`), bitwise the Python API's; the other evaluations
    /// return none and say why there. A cross-backend check in a library
    /// without the other backend is refused when a call is resolved (the
    /// Python API's `ReductionUnavailable`).
    pub fn set_stability_checks(&mut self, checks: StabilityChecks) {
        self.checks = checks;
    }
    /// The stability checks of later evaluations.
    pub fn stability_checks(&self) -> &StabilityChecks {
        &self.checks
    }
    /// The process type: "virtual" (Born-interfered) or "loop-induced".
    pub fn contribution(&self) -> &str {
        let Engine::V4(native) = &self.engine;
        &native.metadata().contribution_kind
    }
    /// The reduction backends the artifact admits: those its declaration
    /// names (`reductions`, a TIR-only process: TIRust, m-lit lit-eft), else
    /// every one (`None`), as the Python API's runtime reads them.
    fn admitted(&self) -> Option<BTreeSet<String>> {
        let Engine::V4(native) = &self.engine;
        native
            .metadata()
            .reductions
            .as_ref()
            .map(|names| names.iter().cloned().collect())
    }
    /// The backend of a call on `route` at the current arithmetic
    /// (`reduction_policy::resolve`), applied to the native runtime: once
    /// per call, never per point.
    fn reduce_on(&mut self, route: &str) -> RusticolResult<Choice> {
        let route = reduction_policy::route(route).expect("an artifact route");
        let admitted = self.admitted();
        let choice = reduction_policy::resolve(
            &self.reduction,
            self.contribution(),
            self.arithmetic.name(),
            &reduction_policy::served(),
            admitted.as_ref(),
            &route,
        )
        .map_err(|refusal| match refusal {
            Refusal::Unavailable(message) => RusticolError::compatibility(message),
            Refusal::Invalid(message) => invalid(message),
        })?;
        let Engine::V4(native) = &mut self.engine;
        native.set_reduction(Reduction::parse(&choice.effective)?)?;
        Ok(choice)
    }
    pub fn poles(&self) -> Poles {
        self.poles
    }
    /// The arithmetic of later evaluations (default binary64), the Python
    /// API's `arithmetic=` / `precision=` ([`Arithmetic`]).
    pub fn set_arithmetic(&mut self, arithmetic: Arithmetic) -> RusticolResult<()> {
        self.arithmetic = arithmetic;
        Ok(())
    }
    pub fn arithmetic(&self) -> Arithmetic {
        self.arithmetic
    }
    /// Existing artifact metadata includes exact order pairings and schemes.
    pub fn metadata_json(&self) -> RusticolResult<String> {
        let Engine::V4(native) = &self.engine;
        native.metadata_json()
    }
    pub fn capabilities_json(&self) -> RusticolResult<String> {
        let Engine::V4(native) = &self.engine;
        let metadata = native.metadata();
        let (resolved, rows, colour_ids, unavailable) = match &self.observables {
            Ok(axes) => (true, axes.rows.clone(), axes.colour_ids.clone(), None),
            Err(reason) => (
                false,
                vec![],
                self.color_components.clone(),
                Some(reason.clone()),
            ),
        };
        let served: serde_json::Map<String, serde_json::Value> = reduction_policy::served()
            .into_iter()
            .map(|(name, carriers)| (name, serde_json::json!(carriers)))
            .collect();
        // The backends the artifact admits (sorted names), every one unless
        // it is TIR-only (m-lit lit-eft), as the Python API's inspect record.
        let admitted: Vec<String> = match self.admitted() {
            Some(names) => names.into_iter().collect(),
            None => reduction_policy::BACKENDS
                .iter()
                .map(|name| (*name).to_owned())
                .collect(),
        };
        let tir_only = admitted == ["tirust"];
        let colour_zero = native.colour_zero().map(|(exact, orders)| {
            serde_json::json!({
                "exact_colour_zero": exact,
                "colour_zero_loop_orders": orders
                    .iter()
                    .map(|order| order.iter().cloned().collect::<BTreeMap<String, i64>>())
                    .collect::<Vec<_>>(),
            })
        });
        serde_json::to_string(&serde_json::json!({
            "contribution_kind":metadata.contribution_kind,
            "interference":metadata.interference,
            "reduction": {
                "requested": self.reduction,
                "artifact_default": native.artifact_reduction(),
                "auto_default": if tir_only {
                    Some("tirust")
                } else {
                    reduction_policy::default_of(&metadata.contribution_kind)
                },
                "admitted": admitted,
                "native": served,
            },
            "colour_zero": colour_zero,
            "execution_mode":"loop_recurrence",
            "engine":"v2", "plan_sha256":native.plan_sha256(),
            "point_inputs":"native-v2",
            "arithmetic":["double", "double-double", "arbitrary"],
            "current_arithmetic": self.arithmetic.name(),
            "current_digits": self.arithmetic.digits(),
            "epsilon_powers":[0,-1,-2],
            "pole_selections":metadata.poles,
            "poles":self.poles.name(),
            "resolved":resolved, "correlations":resolved,
            "correlation_histories":self.history_ids,
            "spin_correlation_classes":self.spin_classes,
            "correlated_resolved":false,
            "helicity_rows": rows,
            "helicity_ids": (0..rows.len()).map(|i| format!("h{i}")).collect::<Vec<_>>(),
            "resolved_color_ids": colour_ids,
            "observables_unavailable": unavailable,
            "color_accuracy":metadata.color_accuracy,
            "color_contraction":metadata.color_contraction,
            "color_components":self.color_components,
            "color_component_semantics":"contracted colour",
            "available_r2_strategies":["model"],
            "r2_strategy":"model",
            "source_only":true,
            "normalization":"physical couplings and inclusive initial spin/color average"
        }))
        .map_err(serialization)
    }
    /// Update model parameters. A plan v4 process validates the whole
    /// mapping first, as the Python API's `set_model_parameters` (in its
    /// order and words): a value that is not a finite decimal, unknown
    /// names, the renormalisation scale's parameter (set per point by
    /// `mu_r2`) and a value other than the one generation fixed are refused
    /// with a typed `ModelParameter` error, a complex value with a typed
    /// `Compatibility` error, and a refused update changes nothing (its
    /// revision is always 0).
    pub fn update_parameters(&mut self, values: &SourceOverrides) -> RusticolResult<u64> {
        let Engine::V4(native) = &mut self.engine;
        native.set_parameters(real_overrides(values)?)?;
        Ok(0)
    }
    /// Plan v4 runtimes keep no evaluation caches to clear.
    pub fn clear(&mut self) {}
    /// Binary64 inputs are encoded as their exact terminating decimal values,
    /// so the native program reads the original C/Rust input numbers.
    pub fn evaluate(
        &mut self,
        momenta: &[f64],
        point_ids: &[u64],
        scales: &[f64],
        resolved: bool,
        strategy: Option<&str>,
    ) -> RusticolResult<Evaluation> {
        let exact_momenta = momenta
            .iter()
            .map(|&v| exact_double(v))
            .collect::<RusticolResult<_>>()?;
        let exact_scales = scales
            .iter()
            .map(|&v| exact_double(v))
            .collect::<RusticolResult<Vec<_>>>()?;
        self.evaluate_decimal(
            exact_momenta,
            point_ids.to_vec(),
            exact_scales,
            resolved,
            strategy,
        )
    }
    /// Decimal literals remain exact input snapshots; the returned carrier is
    /// binary64.
    pub fn evaluate_decimal(
        &mut self,
        momenta: Vec<String>,
        point_ids: Vec<u64>,
        scales: Vec<String>,
        resolved: bool,
        strategy: Option<&str>,
    ) -> RusticolResult<Evaluation> {
        let inputs = self.prepare_inputs(momenta, point_ids, scales)?;
        if let Some(strategy) = strategy {
            v4_r2_strategy(strategy)?;
        }
        let (shape, poles, arithmetic) = (self.shape, self.poles, self.arithmetic);
        if resolved {
            let axes = self.observables.clone().map_err(|reason| {
                RusticolError::compatibility(format!("{NO_OBSERVABLES}: {reason}"))
            })?;
            let reduction = self.reduce_on(reduction_policy::RESOLVED)?;
            let requested = self.checks.enabled();
            let Engine::V4(native) = &mut self.engine;
            return Evaluation::resolved_v4(
                native,
                inputs,
                shape,
                poles,
                arithmetic,
                &axes,
                &reduction,
                Monitoring {
                    checks: None,
                    requested,
                },
            );
        }
        let reduction = self.reduce_on(reduction_policy::SUMMED)?;
        let monitored = arithmetic == Arithmetic::Double && self.checks.enabled();
        if monitored && self.checks.cross_backend() {
            self.cross_available(&reduction)?;
        }
        let checks = monitored.then(|| self.checks.clone());
        let Engine::V4(native) = &mut self.engine;
        Evaluation::from_v4(
            native,
            inputs,
            shape,
            poles,
            arithmetic,
            &reduction,
            Monitoring {
                checks: checks.as_ref(),
                requested: self.checks.enabled(),
            },
        )
    }
    /// The cross-backend check's backend serves the call (the Python API's
    /// `V2VirtualRuntime._reduction_choices`): the other backend, resolved as
    /// an explicit request at the call's carrier and route, else refused
    /// typed in the Python API's words.
    fn cross_available(&self, reduction: &Choice) -> RusticolResult<()> {
        let other = cross_reduction(Reduction::parse(&reduction.effective)?);
        let route = reduction_policy::route(reduction_policy::SUMMED).expect("the summed route");
        reduction_policy::resolve(
            other.name(),
            self.contribution(),
            self.arithmetic.name(),
            &reduction_policy::served(),
            self.admitted().as_ref(),
            &route,
        )
        .map(|_| ())
        .map_err(|refusal| {
            let message = match refusal {
                Refusal::Unavailable(message) | Refusal::Invalid(message) => message,
            };
            RusticolError::compatibility(format!(
                "the cross-backend check (stability_checks cross_backend) compares with the \
                 other reduction backend: {message}"
            ))
        })
    }
    fn prepare_inputs(
        &self,
        momenta: Vec<String>,
        point_ids: Vec<u64>,
        mut scales: Vec<String>,
    ) -> RusticolResult<LoopInputs> {
        if momenta.len() != product(&[point_ids.len(), self.shape.external_count, 4])? {
            return Err(invalid(
                "momentum count must equal point_count * external_count * 4",
            ));
        }
        if scales.len() == 1 {
            scales = vec![scales[0].clone(); point_ids.len()];
        }
        if scales.len() != point_ids.len() {
            return Err(invalid("mu_r2 requires one scale or one per point"));
        }
        Ok(LoopInputs {
            spin_sources: Default::default(),
            momenta,
            point_ids,
            scales_squared: scales,
            model_values: vec![],
        })
    }
}

/// Real model parameter overrides, read as the Python API reads them
/// (`runtime.virtual._parameters`, then `artifact._real_parameters`): by
/// name, every name nonempty ([`named_parameters`]: a typed
/// `InvalidArgument` error otherwise, the Python `TypeError`), each real and
/// imaginary part a finite decimal
/// ([`decimal_parameter`]: a typed `ModelParameter` error otherwise, the
/// Python `ValueError`), then a plan v4 process takes real external
/// parameters only: a nonzero imaginary part is refused with a typed
/// `Compatibility` error (the Python `CompatibilityError`, in its words).
fn real_overrides(values: &SourceOverrides) -> RusticolResult<Vec<(&str, &str)>> {
    named_parameters(values.keys().map(String::as_str))?;
    let mut imaginary = Vec::with_capacity(values.len());
    for (name, [re, im]) in values {
        decimal_parameter(re, &format!("parameter '{name}' real part"))?;
        imaginary.push(decimal_parameter(
            im,
            &format!("parameter '{name}' imaginary part"),
        )?);
    }
    values
        .iter()
        .zip(imaginary)
        .map(|((name, [re, _]), im)| {
            if im.is_zero() {
                Ok((name.as_str(), re.as_str()))
            } else {
                Err(RusticolError::compatibility(format!(
                    "parameter '{name}': v2 loop runtimes take real external parameters"
                )))
            }
        })
        .collect()
}

fn exact_double(value: f64) -> RusticolResult<String> {
    if !value.is_finite() {
        return Err(invalid("SDK binary64 inputs must be finite"));
    }
    // IEEE binary64 denominators divide 2^1074. Standard fixed formatting at
    // 1074 fractional places therefore emits the exact value, without a custom
    // binary-to-decimal algorithm. Remove only redundant fractional zeros.
    let fractional_digits = (f64::MANTISSA_DIGITS as i32 - f64::MIN_EXP) as usize;
    let decimal = format!("{value:.fractional_digits$}");
    Ok(decimal
        .trim_end_matches('0')
        .trim_end_matches('.')
        .to_owned())
}

#[cfg(test)]
fn pack(
    batch: &Batch,
    points: usize,
    orders: usize,
    helicities: usize,
    colors: usize,
    epsilon: usize,
) -> RusticolResult<Vec<f64>> {
    let Data::Double(values) = batch.data() else {
        return Err(invalid("standalone output is not binary64"));
    };
    if batch.rows() != product(&[points, helicities, colors])?
        || batch.columns() != product(&[orders, epsilon])?
    {
        return Err(invalid(
            "native output axes differ from the virtual catalogue",
        ));
    }
    let mut output = Vec::with_capacity(product(&[points, orders, helicities, colors, epsilon])?);
    for point in 0..points {
        for order in 0..orders {
            for helicity in 0..helicities {
                for color in 0..colors {
                    for pole in 0..epsilon {
                        let value = &values[(((point * helicities + helicity) * colors + color)
                            * orders
                            + order)
                            * epsilon
                            + pole];
                        if !value.re.is_finite() || value.im != 0.0 {
                            return Err(invalid("ordinary virtual result is not finite real"));
                        }
                        output.push(value.re);
                    }
                }
            }
        }
    }
    Ok(output)
}

/// Exact decimal momenta and scales of a batch (`LoopInputs`), as rationals
/// per point.
#[allow(clippy::type_complexity)]
fn exact_inputs(
    inputs: &LoopInputs,
    external_count: usize,
) -> RusticolResult<(Vec<Vec<Rational>>, Vec<Rational>)> {
    use rusticol_core::loop_v2::dress::parse_rational;
    let width = 4 * external_count;
    let momenta = inputs
        .momenta
        .chunks(width.max(1))
        .map(|point| point.iter().map(|x| parse_rational(x)).collect())
        .collect::<RusticolResult<Vec<Vec<_>>>>()?;
    let scales = inputs
        .scales_squared
        .iter()
        .map(|x| parse_rational(x))
        .collect::<RusticolResult<Vec<_>>>()?;
    Ok((momenta, scales))
}

/// A value of an evaluation: binary64, or an exact decimal (its nearest
/// binary64 value and its literal).
pub(crate) trait Output: Clone {
    fn double(&self) -> f64;
    fn literal(&self) -> Option<String>;
}

impl Output for f64 {
    fn double(&self) -> f64 {
        *self
    }
    fn literal(&self) -> Option<String> {
        None
    }
}

impl Output for Decimal {
    fn double(&self) -> f64 {
        self.to_f64()
    }
    fn literal(&self) -> Option<String> {
        Some(self.text())
    }
}

/// Real outputs and, for the extended carriers, their literals; an absent
/// pole is a quiet NaN (its literal `None`).
#[derive(Default)]
struct Reals {
    values: Vec<f64>,
    literals: Vec<Option<String>>,
}

impl Reals {
    fn push<N: Output>(&mut self, value: Option<&N>) {
        self.values.push(value.map_or(f64::NAN, Output::double));
        self.literals.push(value.and_then(Output::literal));
    }
}

/// The `decimal` record of an evaluation in `arithmetic`.
fn decimal_values(
    arithmetic: Arithmetic,
    born: Reals,
    virtual_values: Reals,
) -> Option<DecimalValues> {
    (arithmetic != Arithmetic::Double).then(|| DecimalValues {
        arithmetic: arithmetic.name(),
        digits: arithmetic.working_digits(),
        born: born
            .literals
            .into_iter()
            .map(|value| value.expect("a Born value"))
            .collect(),
        virtual_values: virtual_values.literals,
    })
}

/// Per point the evaluated values of an evaluation in the runtime's
/// arithmetic, or its failures (`ordinary`: the inclusive virtual).
fn statuses_of<T>(
    ids: &[u64],
    points: &[Result<T, Vec<String>>],
) -> (Vec<u64>, Vec<PointStatus>, BTreeMap<u64, Vec<String>>) {
    let mut successful = Vec::new();
    let mut statuses = Vec::with_capacity(points.len());
    let mut failures = BTreeMap::new();
    for (id, point) in ids.iter().zip(points) {
        match point {
            Ok(_) => {
                successful.push(*id);
                statuses.push(PointStatus::AccuracyNotEstablished);
            }
            Err(reasons) => {
                statuses.push(PointStatus::Failed);
                failures.insert(*id, reasons.clone());
            }
        }
    }
    (successful, statuses, failures)
}

impl Evaluation {
    /// One ordinary batch of a plan v4 process in the runtime's arithmetic:
    /// Born per order and the complete virtual per sector and epsilon power
    /// (with `Poles::None` the finite part, the poles quiet NaNs), failed
    /// points without rows.
    fn from_v4(
        native: &mut V2Runtime,
        inputs: LoopInputs,
        shape: Shape,
        poles: Poles,
        arithmetic: Arithmetic,
        reduction: &Choice,
        monitoring: Monitoring<'_>,
    ) -> RusticolResult<Self> {
        let (momenta, scales) = exact_inputs(&inputs, shape.external_count)?;
        match (arithmetic, monitoring.checks) {
            (Arithmetic::Double, Some(checks)) => {
                let monitored = native.evaluate_checked(&momenta, &scales, poles.v2(), checks)?;
                let mut points = Vec::with_capacity(monitored.len());
                let mut records = Vec::new();
                let mut finiteness = Vec::new();
                for (point, estimate, ratio) in monitored {
                    if let Some(estimate) = &estimate {
                        records.push(public_estimate(estimate, poles != Poles::None, native));
                        finiteness.push(ratio.map(number));
                    }
                    points.push(point);
                }
                // A loop-induced point's finiteness ratio, with poles
                // (`VirtualLaneDiagnostic.finiteness`): per successful point.
                let finiteness =
                    (native.loop_induced() && poles != Poles::None).then_some(finiteness);
                let stability = serde_json::json!({
                    "estimates": records,
                    "finiteness": finiteness,
                });
                Self::ordinary(
                    inputs, shape, poles, arithmetic, points, reduction, stability,
                )
            }
            (Arithmetic::Double, None) => {
                let points = native.evaluate(&momenta, &scales, poles.v2())?;
                let stability = monitoring.not_run();
                Self::ordinary(
                    inputs, shape, poles, arithmetic, points, reduction, stability,
                )
            }
            _ => {
                let points =
                    native.evaluate_extended(&momenta, &scales, poles.v2(), arithmetic.v2())?;
                let stability = monitoring.not_run();
                Self::ordinary(
                    inputs, shape, poles, arithmetic, points, reduction, stability,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn ordinary<N: Output>(
        inputs: LoopInputs,
        shape: Shape,
        poles: Poles,
        arithmetic: Arithmetic,
        points: Vec<V2Point<N>>,
        reduction: &Choice,
        stability: serde_json::Value,
    ) -> RusticolResult<Self> {
        let (successful_point_ids, statuses, failures) = statuses_of(&inputs.point_ids, &points);
        let mut born = Reals::default();
        let mut virtual_values = Reals::default();
        for (born_values, sectors) in points.into_iter().flatten() {
            for value in &born_values {
                born.push(Some(value));
            }
            for laurent in sectors {
                for k in 0..shape.epsilon_count {
                    // Absent poles (finite-only).
                    virtual_values.push(laurent.get(k));
                }
            }
        }
        let diagnostics_json = diagnostics(
            &inputs.point_ids,
            &successful_point_ids,
            &failures,
            poles,
            arithmetic,
            reduction,
            stability,
        )?;
        Ok(Self {
            requested_point_ids: inputs.point_ids,
            successful_point_ids,
            statuses,
            born: born.values.clone(),
            virtual_values: virtual_values.values.clone(),
            shape,
            resolved: false,
            diagnostics_json,
            decimal: decimal_values(arithmetic, born, virtual_values),
        })
    }

    /// One helicity-resolved batch of a plan v4 process (the Python API's
    /// `evaluate_resolved`): per Born helicity row (and at leading colour
    /// per Born basis element) the Born and the virtual, from the stored
    /// resolved columns, in the layout point/order/helicity/colour/(epsilon).
    #[allow(clippy::too_many_arguments)]
    fn resolved_v4(
        native: &mut V2Runtime,
        inputs: LoopInputs,
        shape: Shape,
        poles: Poles,
        arithmetic: Arithmetic,
        axes: &Observables,
        reduction: &Choice,
        monitoring: Monitoring<'_>,
    ) -> RusticolResult<Self> {
        let (momenta, scales) = exact_inputs(&inputs, shape.external_count)?;
        let keys = native.resolved_keys()?;
        let stability = monitoring.not_run_resolved();
        match arithmetic {
            Arithmetic::Double => {
                let points =
                    native.evaluate_columns("resolved", &momenta, &scales, &[], poles.v2())?;
                Self::resolved(
                    inputs, shape, poles, arithmetic, axes, &keys, points, reduction, stability,
                )
            }
            _ => {
                let points = native.evaluate_columns_extended(
                    "resolved",
                    &momenta,
                    &scales,
                    &[],
                    poles.v2(),
                    arithmetic.v2(),
                )?;
                Self::resolved(
                    inputs, shape, poles, arithmetic, axes, &keys, points, reduction, stability,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn resolved<N: Output>(
        inputs: LoopInputs,
        shape: Shape,
        poles: Poles,
        arithmetic: Arithmetic,
        axes: &Observables,
        keys: &[(usize, usize)],
        points: Vec<ColumnPoint<N>>,
        reduction: &Choice,
        stability: serde_json::Value,
    ) -> RusticolResult<Self> {
        let helicities = axes.rows.len();
        let colours = axes.colour_ids.len();
        // Column index of each (helicity row, colour component).
        let mut index = vec![usize::MAX; helicities * colours];
        for (column, (row, colour)) in keys.iter().enumerate() {
            index[row * colours + colour] = column;
        }
        if index.contains(&usize::MAX) {
            return Err(invalid("the stored resolved columns do not cover the rows"));
        }
        let (successful_point_ids, statuses, failures) = statuses_of(&inputs.point_ids, &points);
        let mut born = Reals::default();
        let mut virtual_values = Reals::default();
        for columns in points.into_iter().flatten() {
            for order in 0..axes.born_orders {
                for &column in &index {
                    born.push(Some(&columns[column].born[order].re));
                }
            }
            for sector in 0..shape.virtual_orders {
                for &column in &index {
                    let laurent = &columns[column].virtual_values[sector];
                    for k in 0..shape.epsilon_count {
                        virtual_values.push(laurent.get(k).map(|z| &z.re));
                    }
                }
            }
        }
        let diagnostics_json = diagnostics(
            &inputs.point_ids,
            &successful_point_ids,
            &failures,
            poles,
            arithmetic,
            reduction,
            stability,
        )?;
        Ok(Self {
            requested_point_ids: inputs.point_ids,
            successful_point_ids,
            statuses,
            born: born.values.clone(),
            virtual_values: virtual_values.values.clone(),
            shape: Shape {
                born_orders: axes.born_orders,
                ..shape
            },
            resolved: true,
            diagnostics_json,
            decimal: decimal_values(arithmetic, born, virtual_values),
        })
    }
}

impl Evaluation {
    /// The `decimal` record as JSON (`null` for a binary64 evaluation).
    pub fn decimal_json(&self) -> RusticolResult<String> {
        serde_json::to_string(&self.decimal).map_err(serialization)
    }
}

/// The stability monitoring of one evaluation: the checks it runs (binary64
/// ordinary evaluations), and whether checks were set at all.
struct Monitoring<'a> {
    checks: Option<&'a StabilityChecks>,
    requested: bool,
}

/// The reason an evaluation with checks set forms no estimate.
const BINARY64_MONITORING: &str = "the native SDKs monitor binary64 evaluations (the Python \
    API estimates every carrier)";
const ORDINARY_MONITORING: &str = "the native SDKs monitor the ordinary evaluations (the \
    Python API estimates the resolved and correlated outputs)";

impl Monitoring<'_> {
    fn record(&self, reason: &str) -> serde_json::Value {
        if self.requested {
            serde_json::json!({ "estimates": null, "reason": reason })
        } else {
            serde_json::Value::Null
        }
    }
    fn not_run(&self) -> serde_json::Value {
        self.record(BINARY64_MONITORING)
    }
    fn not_run_resolved(&self) -> serde_json::Value {
        self.record(ORDINARY_MONITORING)
    }
}

/// A binary64 value of an estimate record: a JSON number, or `"inf"` /
/// `"nan"` (an estimate on an exact zero is infinite).
fn number(value: f64) -> serde_json::Value {
    if value.is_finite() {
        serde_json::json!(value)
    } else if value.is_nan() {
        serde_json::json!("nan")
    } else if value > 0.0 {
        serde_json::json!("inf")
    } else {
        serde_json::json!("-inf")
    }
}

/// E / S (`runtime.virtual._ratio`): 0 for an exact zero estimate, inf for
/// E > 0 at S = 0.
fn ratio(error: f64, scale: f64) -> f64 {
    if error == 0.0 {
        0.0
    } else if scale != 0.0 {
        error / scale
    } else {
        f64::INFINITY
    }
}

/// The public record of a point's estimate, the Python API's
/// `VirtualStabilityEstimate` (`runtime.virtual.v2_public_estimate`):
/// `relative`, `finite`, `single_pole` and `double_pole` (null without
/// poles; a loop-induced result's are E / S), `checks` (name, relative budget),
/// `scale` (the largest E / S), `absolute` E and `magnitude` S per declared
/// sector and Laurent coefficient (absent poles null), `not_run` (name,
/// reason).
fn public_estimate(estimate: &Estimate, with_poles: bool, native: &V2Runtime) -> serde_json::Value {
    let width = if with_poles { 3 } else { 1 };
    let floats = |values: &[Vec<f64>], sector: usize| -> Vec<f64> {
        (0..width)
            .map(|k| {
                values
                    .get(sector)
                    .and_then(|row| row.get(k))
                    .map_or(0.0, |x| x.abs())
            })
            .collect()
    };
    let errors: Vec<Vec<f64>> = native
        .sector_index()
        .iter()
        .map(|&s| floats(&estimate.absolute, s))
        .collect();
    let sizes: Vec<Vec<f64>> = native
        .sector_index()
        .iter()
        .map(|&s| floats(&estimate.scale, s))
        .collect();
    let mut scale: Option<f64> = None;
    for k in 0..width {
        for (error, size) in errors.iter().zip(&sizes) {
            let value = ratio(error[k], size[k]);
            // Python's `max`: the current value unless `value` is larger.
            scale = Some(match scale {
                Some(current) if value > current => value,
                Some(current) => current,
                None => value,
            });
        }
    }
    let laurent = |values: &[f64]| -> serde_json::Value {
        let mut out: Vec<serde_json::Value> = values.iter().map(|x| number(*x)).collect();
        out.resize(3, serde_json::Value::Null);
        serde_json::Value::Array(out)
    };
    // A loop-induced result's pole estimates are E / S too (its poles
    // vanish, so their own size is no scale; m-lit PLAN §2.a.7).
    let induced = native.loop_induced();
    let pole = |k: usize| {
        if !with_poles {
            serde_json::Value::Null
        } else if induced {
            let mut worst: Option<f64> = None;
            for (error, size) in errors.iter().zip(&sizes) {
                let value = ratio(error[k], size[k]);
                worst = Some(match worst {
                    Some(current) if value > current => value,
                    Some(current) => current,
                    None => value,
                });
            }
            number(worst.unwrap_or(0.0))
        } else {
            number(estimate.coefficients[k])
        }
    };
    serde_json::json!({
        "relative": number(estimate.relative),
        "finite": number(estimate.coefficients[0]),
        "single_pole": pole(1),
        "double_pole": pole(2),
        "checks": estimate
            .checks
            .iter()
            .map(|(name, value)| serde_json::json!([name, number(*value)]))
            .collect::<Vec<_>>(),
        "scale": number(scale.unwrap_or(0.0)),
        "absolute": errors.iter().map(|row| laurent(row)).collect::<Vec<_>>(),
        "magnitude": sizes.iter().map(|row| laurent(row)).collect::<Vec<_>>(),
        "not_run": estimate
            .not_run
            .iter()
            .map(|(name, reason)| serde_json::json!([name, reason]))
            .collect::<Vec<_>>(),
    })
}

/// The diagnostics record of an evaluation.
#[allow(clippy::too_many_arguments)]
fn diagnostics(
    requested: &[u64],
    successful: &[u64],
    failures: &BTreeMap<u64, Vec<String>>,
    poles: Poles,
    arithmetic: Arithmetic,
    reduction: &Choice,
    stability: serde_json::Value,
) -> RusticolResult<String> {
    serde_json::to_string(&serde_json::json!({
        // The stability monitoring (`Runtime::set_stability_checks`): null
        // without checks, else per successful point its estimate (the
        // Python API's `VirtualStabilityEstimate`), or why none is formed.
        "stability": stability,
        // The reduction backend of the call (the Python API's
        // `VirtualReduction`): requested, effective and why.
        "reduction": {
            "requested": reduction.requested,
            "effective": reduction.effective,
            "reason": reduction.reason,
        },
        "requested_point_ids":requested,
        "successful_point_ids":successful,
        "failures":failures,
        "r2_strategy":"model", "engine":"v2", "poles":poles.name(),
        "arithmetic": arithmetic.name(),
        "working_digits": (arithmetic != Arithmetic::Double).then(|| arithmetic.working_digits()),
        "primary_color_contraction":"direct",
    }))
    .map_err(serialization)
}

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::domains::rational::Rational;

    #[test]
    fn pole_selection_parses_every_selection_of_the_api() {
        assert_eq!(Poles::default(), Poles::All);
        for (text, poles) in [
            ("all", Poles::All),
            ("none", Poles::None),
            ("uv", Poles::Uv),
            ("ir", Poles::Ir),
        ] {
            assert_eq!(Poles::parse(text).unwrap(), poles);
            assert_eq!(poles.name(), text);
        }
        for other in ["", "All", "finite", "split"] {
            assert!(Poles::parse(other).is_err(), "{other:?}");
        }
    }
    #[test]
    fn binary64_ingress_preserves_exact_values() {
        for value in [
            0.0,
            -0.0,
            0.1,
            -0.1,
            f64::from_bits(1),
            f64::MIN_POSITIVE,
            f64::MAX,
            -f64::MAX,
            1.0000000000000002,
        ] {
            let encoded = exact_double(value).unwrap();
            assert_eq!(
                rusticol_loop_reduction::runtime::decimal_rational(&encoded).unwrap(),
                Rational::try_from(value).unwrap()
            );
        }
        assert!(exact_double(f64::INFINITY).is_err());
    }

    #[test]
    fn packing_keeps_order_helicity_colour_and_epsilon_axes_distinct() {
        use rusticol_loop_reduction::Complex;
        // Two sparse physical IDs, three helicities, two colours and orders.
        let ids = [91, 4]
            .into_iter()
            .flat_map(|p| (0..6).map(move |r| p * 6 + r))
            .collect();
        let values = (0..2)
            .flat_map(|p| {
                (0..3).flat_map(move |h| {
                    (0..2).flat_map(move |c| {
                        (0..2).flat_map(move |o| {
                            (0..3).map(move |e| {
                                Complex::new(
                                    -((p * 10000 + o * 1000 + h * 100 + c * 10 + e + 1) as f64),
                                    0.,
                                )
                            })
                        })
                    })
                })
            })
            .collect();
        let batch = Batch::typed(values, ids, 6, 53).unwrap();
        let actual = pack(&batch, 2, 2, 3, 2, 3).unwrap();
        let expected: Vec<_> = (0..2)
            .flat_map(|p| {
                (0..2).flat_map(move |o| {
                    (0..3).flat_map(move |h| {
                        (0..2).flat_map(move |c| {
                            (0..3).map(move |e| {
                                -((p * 10000 + o * 1000 + h * 100 + c * 10 + e + 1) as f64)
                            })
                        })
                    })
                })
            })
            .collect();
        assert_eq!(actual, expected);
        assert!(pack(&batch, 2, 2, 3, 1, 3).is_err());
        let empty = Batch::typed(Vec::<Complex<f64>>::new(), vec![], 6, 53).unwrap();
        assert!(pack(&empty, 0, 2, 3, 2, 3).unwrap().is_empty());
    }
}
