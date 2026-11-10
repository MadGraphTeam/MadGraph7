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
use rusticol_loop_reduction::v2_runtime::induced_outputs::{
    AMPLITUDE_BASIS, CONTRACTED, InducedCall, Resolution, anchor_id,
};
use rusticol_loop_reduction::v2_runtime::kinematics_improvement;
use rusticol_loop_reduction::v2_runtime::{
    C, ColumnPoint, Decimal, Rational, V2Point, V2Poles, V2Precision, V2Runtime, decimal_parameter,
    is_v2_artifact, named_parameters,
};
use rusticol_loop_reduction::v2_runtime::{Monitored, ResolvedPoints, cross_reduction};
use rusticol_loop_reduction::v2_stability::CROSS;
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
    /// A resolved result's helicity row IDs (`h<i>`, the capabilities'
    /// `helicity_ids`) and colour component IDs, in the layout's order
    /// (the result's own axes: a selection keeps its rows, the
    /// amplitude-basis anchors are `amplitude-<t>`); empty for an ordinary
    /// result.
    pub helicity_ids: Vec<String>,
    pub color_ids: Vec<String>,
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
    /// The carrier's name (`V2Precision::name`, `reduction_policy::CARRIERS`).
    pub fn name(self) -> &'static str {
        self.v2().name()
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

/// The selection of an evaluation ([`Runtime::evaluate_selected`]; m-lit
/// lit-sdk-b, the USER's F1 ruling: the tree-level selectors at parity).
/// Global selectors are string IDs, per-point selectors zero-based indices
/// (one per point); global and per-point selectors are mutually exclusive
/// on the same axis, as the tree C ABI's `rusticol_runtime_evaluate_selected_f64`.
/// A loop-induced process serves helicity selection (the Python API's
/// `helicities=`: the runtime's `h<i>` IDs or the tree engine's
/// `h:<values in leg order>`) and, for a resolved evaluation, the colour
/// resolution (`color_resolution=`: "contracted", the default, or
/// "amplitude-basis"); it has no colour flows (full colour only, RULED Q8).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Selection {
    /// Global helicity IDs (None or empty: every row).
    pub helicities: Option<Vec<String>>,
    /// Global colour-flow IDs or ordinals (None or empty: every flow).
    pub color_flows: Option<Vec<String>>,
    /// Per point its zero-based helicity row (ordinary evaluations).
    pub helicity_by_point: Option<Vec<u32>>,
    /// Per point its zero-based colour flow (ordinary evaluations).
    pub color_flow_by_point: Option<Vec<u32>>,
    /// The colour axis of a resolved evaluation (None: "contracted").
    pub color_resolution: Option<String>,
}

impl Selection {
    fn helicities(&self) -> Option<&[String]> {
        self.helicities.as_deref().filter(|ids| !ids.is_empty())
    }
    fn colour_selected(&self) -> bool {
        self.color_flows.as_ref().is_some_and(|ids| !ids.is_empty())
            || self.color_flow_by_point.is_some()
    }
    fn selects(&self) -> bool {
        self.helicities().is_some() || self.helicity_by_point.is_some() || self.colour_selected()
    }
}

/// The Python API's refusal of colour flows on a loop-induced process
/// (`V2InducedObservablesRuntime._require_selection`).
const NO_COLOUR_FLOWS: &str = "a loop-induced process is evaluated at full colour only and \
    has no colour flows to select; select helicity rows (helicities=...) or resolve the \
    amplitude-basis anchors (evaluate_resolved(color_resolution='amplitude-basis'))";

/// The Python API's refusal of a colour resolution on a Born-interfered
/// process (`V2ObservablesRuntime._resolution`).
const BORN_RESOLUTION: &str = "color_resolution selects the colour axis of a loop-induced \
    process's resolved output; a Born-interfered virtual resolves its colour by its accuracy";

/// A Born-interfered process's leading-colour colour-flow selection is the
/// Python API's: the native SDKs select its helicity rows at every colour
/// accuracy (lit-fc-hel), not its colour flows.
const BORN_FLOWS: &str = "the native SDKs select helicity rows of Born-interfered \
    processes at every colour accuracy; a leading-colour colour-flow selection is \
    served by the Python API (Runtime.evaluate(color_flows=...))";

/// The Python API's refusal of colour flows below leading colour
/// (`selection.require_colour_flows`).
fn no_flows(accuracy: &str) -> String {
    let name = match accuracy {
        "nlc" => "next-to-leading",
        other => other,
    };
    format!(
        "v2 virtual runtimes contract the colour at {name} colour, which has no colour \
         flows to select; colour-flow selection is available at leading colour \
         (color.accuracy='lc'); helicities=... selects Born helicity rows at every \
         colour accuracy"
    )
}

/// Python's `repr` of a string (the quotes of its messages).
fn python_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// The tree engine's ID of a helicity row (`selection.helicity_id`:
/// `h:` and the helicities in leg order, signed).
fn tree_helicity_id(row: &[(usize, i32)]) -> String {
    let mut row = row.to_vec();
    row.sort();
    let values: Vec<String> = row.iter().map(|(_, h)| format!("{h:+}")).collect();
    format!("h:{}", values.join(","))
}

/// The helicity rows (indices, sorted, each once) of global helicity IDs
/// (`selection.Catalogue.resolve`): the runtime's `h<i>` or the tree
/// engine's `h:<values in leg order>`, surrounding whitespace ignored; an
/// unknown ID is refused in the Python API's words (typed `Selector`).
fn helicity_rows(rows: &[Vec<(usize, i32)>], ids: &[String]) -> RusticolResult<Vec<usize>> {
    let mut index: BTreeMap<String, usize> = BTreeMap::new();
    for (k, row) in rows.iter().enumerate() {
        index.insert(tree_helicity_id(row), k);
        index.insert(format!("h{k}"), k);
    }
    let mut kept = BTreeSet::new();
    for name in ids {
        let row = index.get(name.trim()).ok_or_else(|| {
            RusticolError::selector(format!(
                "unknown helicity selector {}; choose a helicity ID of the process \
                 (h:<values in leg order>)",
                python_repr(name)
            ))
        })?;
        kept.insert(*row);
    }
    Ok(kept.into_iter().collect())
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
    /// The target accuracy of later evaluations
    /// (`reduction_policy::DEFAULT_TARGET` until
    /// [`Runtime::set_reduction_target`]).
    target: f64,
    /// The unchecked setting of later evaluations
    /// (`reduction_policy::DEFAULT_UNCHECKED`, off, until
    /// [`Runtime::set_reduction_unchecked`]).
    unchecked: bool,
    /// The stability checks of later binary64 evaluations (none by default:
    /// no estimate, no extra cost; [`Runtime::set_stability_checks`]).
    checks: StabilityChecks,
    /// The kinematics improvement of later evaluations (m-lit
    /// lit-psimprove; [`Runtime::set_kinematics_improvement`]): the variant
    /// (None: "off"), MadLoop's default until set.
    kinematics: Option<kinematics_improvement::Variant>,
    /// Whether binary64 evaluations are improved too
    /// ([`Runtime::set_kinematics_improvement_binary64`]; off by default).
    kinematics_binary64: bool,
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
    /// A loop-induced process's amplitude-basis size (its anchors); 0 for
    /// a Born-interfered one.
    anchors: usize,
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
        let target = reduction_policy::DEFAULT_TARGET;
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
                    anchors: if native.loop_induced() { basis } else { 0 },
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
            target,
            unchecked: reduction_policy::DEFAULT_UNCHECKED,
            checks: StabilityChecks::default(),
            kinematics: Some(kinematics_improvement::DEFAULT_VARIANT),
            kinematics_binary64: kinematics_improvement::DEFAULT_BINARY64,
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
        let backend = reduction_policy::BACKENDS.contains(&reduction);
        match reduction {
            "auto" => {}
            _ if backend && served.contains_key(reduction) => {}
            _ if backend => {
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
                     \"auto\", {}",
                    reduction_policy::BACKENDS
                        .iter()
                        .map(|name| format!("\"{name}\""))
                        .collect::<Vec<_>>()
                        .join(" or ")
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
    /// The target accuracy of later evaluations, the Python API's
    /// `reduction_target=` (m-lit F3 (a)): the relative accuracy of each
    /// point's result (every squared-order sector against its own size),
    /// `reduction_policy::DEFAULT_TARGET` (1e-5) by default. TIRust's
    /// method-choice acceptance follows it (calibrated,
    /// `v2_tir::acceptance_of`); NinjaRS has no setting it drives. A value
    /// that is not a finite positive number is refused (typed
    /// `InvalidArgument`, `reduction_policy::TARGET_REFUSAL`) and nothing
    /// changes.
    pub fn set_reduction_target(&mut self, target: f64) -> RusticolResult<()> {
        self.target = reduction_policy::target(target).map_err(|refusal| match refusal {
            Refusal::Invalid(message) | Refusal::Unavailable(message) => invalid(message),
        })?;
        Ok(())
    }
    /// The target accuracy of later evaluations.
    pub fn reduction_target(&self) -> f64 {
        self.target
    }
    /// The unchecked setting of later evaluations, the Python API's
    /// `reduction_unchecked=` (m-lit lit-unchecked; off by default,
    /// `reduction_policy::DEFAULT_UNCHECKED`): the reduction backends reduce
    /// without their own acceptance and refusal checks
    /// (`reduction_policy::UNCHECKED_RECORD`: TIRust's gates and method
    /// choice as method selection without the acceptance pass, its gate-0
    /// last resort; NinjaRS's zero stability thresholds); "auto" keeps its
    /// backend choice and its per-point fallback, which then acts only on
    /// the points the unchecked backend still fails. The stability checks
    /// ([`Runtime::set_stability_checks`]) stay separate settings. Each
    /// result records it (its "reduction" diagnostics).
    pub fn set_reduction_unchecked(&mut self, unchecked: bool) {
        self.unchecked = unchecked;
    }
    /// The unchecked setting of later evaluations.
    pub fn reduction_unchecked(&self) -> bool {
        self.unchecked
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
    /// The kinematics improvement of later evaluations (m-lit
    /// lit-psimprove; the Python API's `kinematics_improvement=`):
    /// `"z-shift"` (MadLoop's ImprovePSPoint 2 with MadLoop's own fallback to
    /// 1; the default), `"rescaling"` (ImprovePSPoint 1, with the fallback to
    /// 2), `"projection"` (the frame-independent minimal-norm projection) or
    /// `"off"`. Before anything is evaluated in a carrier above binary64
    /// (double-double, arbitrary precision), each point's momenta are moved
    /// onto exact energy-momentum conservation and the external shells in
    /// that carrier's arithmetic, from the caller's exact inputs, in the
    /// caller's leg order; the per-point fallback and the stability rotations
    /// act on that point. Each result records it per point (its
    /// "kinematics" diagnostics). Another value is refused typed with the
    /// one refusal text (`kinematics_improvement::SETTING_REFUSAL`).
    pub fn set_kinematics_improvement(&mut self, setting: &str) -> RusticolResult<()> {
        self.kinematics = kinematics_improvement::setting(setting).map_err(invalid)?;
        Ok(())
    }
    /// The kinematics improvement setting of later evaluations.
    pub fn kinematics_improvement(&self) -> &'static str {
        self.kinematics.map_or(
            kinematics_improvement::OFF,
            kinematics_improvement::Variant::name,
        )
    }
    /// Whether binary64 evaluations are improved too (the Python API's
    /// `kinematics_improvement_binary64=`; off by default: a binary64
    /// evaluation is then bitwise as without the improvement).
    pub fn set_kinematics_improvement_binary64(&mut self, binary64: bool) {
        self.kinematics_binary64 = binary64;
    }
    /// The binary64 switch of later evaluations.
    pub fn kinematics_improvement_binary64(&self) -> bool {
        self.kinematics_binary64
    }
    /// The call's points improved in its carrier ([`V2Runtime::
    /// improve_kinematics`]) and their records, or the inputs unchanged
    /// with no record (a binary64 call without the binary64 switch); the
    /// setting "off" records "off" where the settings reach. One branch per
    /// call when nothing applies.
    pub(crate) fn improved_inputs(
        &mut self,
        mut inputs: LoopInputs,
    ) -> RusticolResult<(LoopInputs, Option<Vec<serde_json::Value>>)> {
        let double = self.arithmetic == Arithmetic::Double;
        if double && !self.kinematics_binary64 {
            return Ok((inputs, None));
        }
        let Some(variant) =
            kinematics_improvement::applies(self.kinematics, self.kinematics_binary64, double)
        else {
            let off =
                vec![kinematics_off_record(self.kinematics_improvement()); inputs.point_ids.len()];
            return Ok((inputs, Some(off)));
        };
        let (momenta, scales) = exact_inputs(&inputs, self.shape.external_count)?;
        let arithmetic = self.arithmetic.v2();
        let Engine::V4(native) = &mut self.engine;
        let (points, records) =
            native.improve_kinematics(&momenta, &scales, arithmetic, variant, true)?;
        inputs.momenta = points
            .iter()
            .flat_map(|point| point.iter().map(ToString::to_string))
            .collect();
        Ok((
            inputs,
            Some(records.iter().map(|record| record.to_json()).collect()),
        ))
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
        native.set_target(self.target)?;
        native.set_unchecked(self.unchecked);
        Ok(Choice {
            unchecked: self.unchecked,
            ..choice
        })
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
        // A loop-induced process's resolved colour axes by colour resolution
        // and its helicity selection (`Runtime::evaluate_selected`).
        let induced = resolved && native.loop_induced();
        let colour_resolutions = match &self.observables {
            Ok(axes) if induced => {
                let mut found = serde_json::Map::new();
                found.insert(CONTRACTED.into(), serde_json::json!(axes.colour_ids));
                found.insert(
                    AMPLITUDE_BASIS.into(),
                    serde_json::json!((0..axes.anchors).map(anchor_id).collect::<Vec<_>>()),
                );
                Some(serde_json::Value::Object(found))
            }
            _ => None,
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
        let admitted_set: BTreeSet<String> = admitted.iter().cloned().collect();
        let colour_zero = native.colour_zero().map(|(exact, orders)| {
            serde_json::json!({
                "exact_colour_zero": exact,
                "colour_zero_loop_orders": orders
                    .iter()
                    .map(|order| order.iter().cloned().collect::<BTreeMap<String, i64>>())
                    .collect::<Vec<_>>(),
            })
        });
        // M-LIT lit-fold: the fold zero's record beside the colour zero's
        // (null without one).
        let fold_zero = native.fold_zero().map(|(exact, orders, certificate)| {
            serde_json::json!({
                "exact_fold_zero": exact,
                "fold_zero_loop_orders": orders
                    .iter()
                    .map(|order| order.iter().cloned().collect::<BTreeMap<String, i64>>())
                    .collect::<Vec<_>>(),
                "certificate": certificate,
            })
        });
        serde_json::to_string(&serde_json::json!({
            "contribution_kind":metadata.contribution_kind,
            "interference":metadata.interference,
            "reduction": {
                "requested": self.reduction,
                "artifact_default": native.artifact_reduction(),
                "auto_default": reduction_policy::auto_default(
                    &metadata.contribution_kind,
                    Some(&admitted_set),
                ),
                "admitted": admitted,
                "native": served,
                "target": self.target,
                "default_target": reduction_policy::DEFAULT_TARGET,
                "unchecked": self.unchecked,
                "default_unchecked": reduction_policy::DEFAULT_UNCHECKED,
            },
            "colour_zero": colour_zero,
            "fold_zero": fold_zero,
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
            "color_resolutions": colour_resolutions,
            // Helicity rows are selected wherever the stored observables
            // serve them: loop-induced (lit-sdk-b) and Born-interfered at
            // every colour accuracy (lit-fc-hel).
            "helicity_selection": resolved,
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
            let (inputs, kinematics) = self.improved_inputs(inputs)?;
            let Engine::V4(native) = &mut self.engine;
            let mut evaluation = Evaluation::resolved_v4(
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
            )?;
            with_kinematics(&mut evaluation.diagnostics_json, kinematics)?;
            return Ok(evaluation);
        }
        let reduction = self.reduce_on(reduction_policy::SUMMED)?;
        // The native monitoring runs in binary64 for the pole selections
        // "all" and "none" (a loop-induced process serves no other); a
        // Born-interfered "uv" / "ir" call returns its values with the
        // documented null record (the Python API estimates those selections).
        let monitored = arithmetic == Arithmetic::Double
            && self.checks.enabled()
            && matches!(poles, Poles::All | Poles::None);
        if monitored && self.checks.cross_backend() {
            self.cross_available(&reduction)?;
        }
        let checks = monitored.then(|| self.checks.clone());
        let requested = self.checks.enabled();
        let (inputs, kinematics) = self.improved_inputs(inputs)?;
        let Engine::V4(native) = &mut self.engine;
        let mut evaluation = Evaluation::from_v4(
            native,
            inputs,
            shape,
            poles,
            arithmetic,
            &reduction,
            Monitoring {
                checks: checks.as_ref(),
                requested,
            },
        )?;
        with_kinematics(&mut evaluation.diagnostics_json, kinematics)?;
        Ok(evaluation)
    }
    /// Whether the process is loop-induced (|A_loop|^2, no Born).
    pub fn loop_induced(&self) -> bool {
        let Engine::V4(native) = &self.engine;
        native.loop_induced()
    }

    /// One evaluation with a selection ([`Selection`]; m-lit lit-sdk-b):
    /// the Python API's `evaluate(helicities=...)` (`resolved` false: per
    /// point the sum over the selected helicity rows of |A(h)|^2/avg with
    /// the total's average, RULED Q39; with `helicity_by_point` each point's
    /// own row, its value that of `evaluate(helicities=[that row's ID])` on
    /// that point) and `evaluate_resolved(helicities=...,
    /// color_resolution=...)` (`resolved` true: the selected rows, each with
    /// its colour-contracted value or its amplitude-basis anchors) of a
    /// loop-induced process, through the one Rust path the Python API's
    /// binary64 path runs: only the selected rows are dressed (the
    /// executor's row liveness). Without any selector (and with the
    /// contracted colour resolution) this is [`Runtime::evaluate`]. Refused (nothing evaluated): colour
    /// flows of a loop-induced process (full colour only, RULED Q8) and an
    /// unknown colour resolution, in the Python API's words (typed
    /// `Compatibility`); an unknown helicity ID (the Python API's words) or
    /// row index, per-point selectors of a resolved evaluation and the
    /// mutually exclusive axes (typed `Selector` / `InvalidArgument`); a
    /// colour resolution of a Born-interfered process (the Python API's
    /// words). A Born-interfered process's helicity selection is
    /// [`Runtime::born_selected`]'s (lit-fc-hel).
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_selected(
        &mut self,
        momenta: &[f64],
        point_ids: &[u64],
        scales: &[f64],
        resolved: bool,
        strategy: Option<&str>,
        selection: &Selection,
    ) -> RusticolResult<Evaluation> {
        let exact_momenta = momenta
            .iter()
            .map(|&v| exact_double(v))
            .collect::<RusticolResult<_>>()?;
        let exact_scales = scales
            .iter()
            .map(|&v| exact_double(v))
            .collect::<RusticolResult<Vec<_>>>()?;
        self.evaluate_selected_decimal(
            exact_momenta,
            point_ids.to_vec(),
            exact_scales,
            resolved,
            strategy,
            selection,
        )
    }

    /// [`Runtime::evaluate_selected`] of exact decimal inputs.
    pub fn evaluate_selected_decimal(
        &mut self,
        momenta: Vec<String>,
        point_ids: Vec<u64>,
        scales: Vec<String>,
        resolved: bool,
        strategy: Option<&str>,
        selection: &Selection,
    ) -> RusticolResult<Evaluation> {
        let points = point_ids.len();
        // The Python API's selector validation (`api.services._selector_ids`):
        // nonempty IDs, each once.
        for (name, kind, ids) in [
            ("helicity", "HelicityConfiguration", &selection.helicities),
            ("color-flow", "ColorFlow", &selection.color_flows),
        ] {
            let ids = ids.as_deref().unwrap_or_default();
            if ids.iter().any(String::is_empty) {
                return Err(invalid(format!("{name} selectors must be IDs or {kind}")));
            }
            if ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
                return Err(invalid(format!("{name} selectors must be unique")));
            }
        }
        for (name, by_point) in [
            ("helicity_by_point", &selection.helicity_by_point),
            ("color_flow_by_point", &selection.color_flow_by_point),
        ] {
            if let Some(by_point) = by_point
                && by_point.len() != points
            {
                return Err(invalid(format!(
                    "{name} has length {}, expected {points} (one selector per point)",
                    by_point.len()
                )));
            }
        }
        if selection.helicities().is_some() && selection.helicity_by_point.is_some() {
            return Err(invalid(
                "helicity ids and helicity_by_point are mutually exclusive",
            ));
        }
        if selection
            .color_flows
            .as_ref()
            .is_some_and(|ids| !ids.is_empty())
            && selection.color_flow_by_point.is_some()
        {
            return Err(invalid(
                "color ids and color_flow_by_point are mutually exclusive",
            ));
        }
        if !resolved && selection.color_resolution.is_some() {
            return Err(invalid(
                "color_resolution selects the colour axis of a resolved evaluation",
            ));
        }
        if !self.loop_induced() {
            if selection.color_resolution.is_some() {
                return Err(RusticolError::compatibility(BORN_RESOLUTION));
            }
            if selection.selects() {
                return self
                    .born_selected(momenta, point_ids, scales, resolved, strategy, selection);
            }
            return self.evaluate_decimal(momenta, point_ids, scales, resolved, strategy);
        }
        // A loop-induced process: the Python API's order (the colour
        // resolution, then the colour flows, then the helicity IDs).
        let resolution = match selection.color_resolution.as_deref() {
            None => Resolution::Contracted,
            Some(name) => Resolution::parse(name).ok_or_else(|| {
                RusticolError::compatibility(format!(
                    "unknown loop-induced colour resolution {}; choose one of [{}, {}]",
                    python_repr(name),
                    python_repr(CONTRACTED),
                    python_repr(AMPLITUDE_BASIS)
                ))
            })?,
        };
        if selection.colour_selected() {
            return Err(RusticolError::compatibility(NO_COLOUR_FLOWS));
        }
        if resolved && selection.helicity_by_point.is_some() {
            return Err(invalid(
                "helicity_by_point selects ordinary evaluations; a resolved evaluation \
                 takes global helicity ids",
            ));
        }
        let axes = self.observables.clone().map_err(|reason| {
            RusticolError::compatibility(format!("{NO_OBSERVABLES}: {reason}"))
        })?;
        let rows = axes.rows.len();
        let kept = match selection.helicities() {
            Some(ids) => Some(helicity_rows(&axes.rows, ids)?),
            None => None,
        };
        if let Some(by_point) = &selection.helicity_by_point
            && let Some((point, &row)) = by_point
                .iter()
                .enumerate()
                .find(|(_, row)| **row as usize >= rows)
        {
            return Err(RusticolError::selector(format!(
                "helicity_by_point[{point}] = {row} is out of range; choose a zero-based \
                 helicity row below {rows}"
            )));
        }
        if kept.is_none()
            && selection.helicity_by_point.is_none()
            && (!resolved || resolution == Resolution::Contracted)
        {
            // Nothing selected: the ordinary or resolved call
            // (`rusticol_virtual_evaluate`), its records included.
            return self.evaluate_decimal(momenta, point_ids, scales, resolved, strategy);
        }
        let inputs = self.prepare_inputs(momenta, point_ids, scales)?;
        if let Some(strategy) = strategy {
            v4_r2_strategy(strategy)?;
        }
        let reduction = self.reduce_on(reduction_policy::RESOLVED)?;
        // The cross-backend check's backend must serve the call's carrier
        // (the Python API's `_reduction_choices`, every carrier), even though
        // the selections form no estimate here.
        if self.checks.enabled() && self.checks.cross_backend() {
            self.cross_available(&reduction)?;
        }
        let monitoring = Monitoring {
            checks: None,
            requested: self.checks.enabled(),
        };
        let (inputs, kinematics) = self.improved_inputs(inputs)?;
        let (shape, poles, arithmetic) = (self.shape, self.poles, self.arithmetic);
        let Engine::V4(native) = &mut self.engine;
        let request = InducedRequest {
            shape,
            poles,
            arithmetic,
            reduction: &reduction,
            stability: monitoring.not_selected(),
        };
        if resolved {
            let kept_rows: Vec<usize> = match &kept {
                Some(kept) => (0..rows).filter(|row| kept.contains(row)).collect(),
                None => (0..rows).collect(),
            };
            let components: Vec<String> = match resolution {
                Resolution::Contracted => axes.colour_ids.clone(),
                Resolution::AmplitudeBasis => (0..axes.anchors).map(anchor_id).collect(),
            };
            let call = InducedCall::Resolved {
                resolution,
                rows: kept.is_some().then(|| kept_rows.clone()),
            };
            let mut evaluation = Evaluation::induced_resolved(
                native, inputs, &call, &kept_rows, components, request,
            )?;
            with_kinematics(&mut evaluation.diagnostics_json, kinematics)?;
            return Ok(evaluation);
        }
        let by_point: Vec<Vec<usize>> = match (&kept, &selection.helicity_by_point) {
            (Some(kept), _) => vec![kept.clone(); inputs.point_ids.len()],
            (None, Some(by_point)) => by_point.iter().map(|&row| vec![row as usize]).collect(),
            (None, None) => unreachable!("an unselected ordinary evaluation returned above"),
        };
        let mut evaluation = Evaluation::induced_selected(native, inputs, &by_point, request)?;
        with_kinematics(&mut evaluation.diagnostics_json, kinematics)?;
        Ok(evaluation)
    }

    /// A Born-interfered process's helicity selection (lit-fc-hel; USER
    /// 2026-11-09 ~19:53): the Python API's `evaluate(helicities=...)`
    /// (`resolved` false: per point the selected rows' value, with
    /// `helicity_by_point` each point's own row, points sharing a row in one
    /// batch) and `evaluate_resolved(helicities=...)` (`resolved` true: the
    /// selected rows in the accuracy's resolved layout) at every colour
    /// accuracy, through the one Rust formation the Python API's stored
    /// path runs (`v2_runtime::template`): only the selected rows' trees,
    /// links, lanes (and resolved columns) are formed. Refused: colour flows
    /// below leading colour (the Python API's words) and at leading colour
    /// (served by the Python API: [`BORN_FLOWS`]), per-point selectors of a
    /// resolved evaluation, an unknown helicity ID or row index.
    fn born_selected(
        &mut self,
        momenta: Vec<String>,
        point_ids: Vec<u64>,
        scales: Vec<String>,
        resolved: bool,
        strategy: Option<&str>,
        selection: &Selection,
    ) -> RusticolResult<Evaluation> {
        let accuracy = {
            let Engine::V4(native) = &self.engine;
            native.metadata().color_accuracy.clone()
        };
        if selection.colour_selected() {
            return Err(RusticolError::compatibility(if accuracy == "lc" {
                BORN_FLOWS.to_owned()
            } else {
                no_flows(&accuracy)
            }));
        }
        if resolved && selection.helicity_by_point.is_some() {
            return Err(invalid(
                "helicity_by_point selects ordinary evaluations; a resolved evaluation \
                 takes global helicity ids",
            ));
        }
        let axes = self.observables.clone().map_err(|reason| {
            RusticolError::compatibility(format!("{NO_OBSERVABLES}: {reason}"))
        })?;
        let rows = axes.rows.len();
        let kept = match selection.helicities() {
            Some(ids) => Some(helicity_rows(&axes.rows, ids)?),
            None => None,
        };
        if let Some(by_point) = &selection.helicity_by_point
            && let Some((point, &row)) = by_point
                .iter()
                .enumerate()
                .find(|(_, row)| **row as usize >= rows)
        {
            return Err(RusticolError::selector(format!(
                "helicity_by_point[{point}] = {row} is out of range; choose a zero-based \
                 helicity row below {rows}"
            )));
        }
        let inputs = self.prepare_inputs(momenta, point_ids, scales)?;
        if let Some(strategy) = strategy {
            v4_r2_strategy(strategy)?;
        }
        let reduction = self.reduce_on(reduction_policy::RESOLVED)?;
        if self.checks.enabled() && self.checks.cross_backend() {
            self.cross_available(&reduction)?;
        }
        let monitoring = Monitoring {
            checks: None,
            requested: self.checks.enabled(),
        };
        let (shape, poles, arithmetic) = (self.shape, self.poles, self.arithmetic);
        let Engine::V4(native) = &mut self.engine;
        let request = InducedRequest {
            shape: Shape {
                born_orders: axes.born_orders,
                ..shape
            },
            poles,
            arithmetic,
            reduction: &reduction,
            stability: monitoring.not_selected(),
        };
        if resolved {
            let kept = kept.expect("a resolved selection names helicity ids");
            return Evaluation::born_resolved_selected(native, inputs, &axes, &kept, request);
        }
        let by_point: Vec<Vec<usize>> = match (&kept, &selection.helicity_by_point) {
            (Some(kept), _) => vec![kept.clone(); inputs.point_ids.len()],
            (None, Some(by_point)) => by_point.iter().map(|&row| vec![row as usize]).collect(),
            (None, None) => unreachable!("an unselected evaluation returned before"),
        };
        Evaluation::born_selected(native, inputs, &by_point, request)
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
            RusticolError::compatibility(reduction_policy::cross_refusal(&message))
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

/// One evaluation's outcome of a point: its failure reasons when it failed
/// (the per-point fallback's test, [`with_fallback`]) and, when the
/// evaluation ran the fallback guard's check with it, that check's relative
/// estimate (no second re-test).
pub(crate) trait PointOutcome: Sized {
    fn reasons(&self) -> Option<&[String]>;
    fn failed(reasons: Vec<String>) -> Self;
    fn guard_estimate(&self) -> Option<f64> {
        None
    }
}

impl<T> PointOutcome for Result<T, Vec<String>> {
    fn reasons(&self) -> Option<&[String]> {
        self.as_ref().err().map(Vec::as_slice)
    }
    fn failed(reasons: Vec<String>) -> Self {
        Err(reasons)
    }
}

/// A point of an evaluation that ran the per-point fallback's guard check
/// with it (`V2Runtime::evaluate_extended_guarded`): its outcome and, when
/// it succeeded, the guard's estimate.
impl<T> PointOutcome for (Result<T, Vec<String>>, Option<f64>) {
    fn reasons(&self) -> Option<&[String]> {
        self.0.reasons()
    }
    fn failed(reasons: Vec<String>) -> Self {
        (Err(reasons), None)
    }
    fn guard_estimate(&self) -> Option<f64> {
        self.1
    }
}

impl PointOutcome for Monitored {
    fn reasons(&self) -> Option<&[String]> {
        self.0.reasons()
    }
    fn failed(reasons: Vec<String>) -> Self {
        (Err(reasons), None, None)
    }
}

/// A point outcome as the call returns it, without the guard's estimate
/// (the outcomes of [`with_fallback`]; m-lit lit-fallback-guard).
pub(crate) trait IntoPoint {
    type Point;
    fn into_point(self) -> Self::Point;
}

impl<T> IntoPoint for Result<T, Vec<String>> {
    type Point = Self;
    fn into_point(self) -> Self {
        self
    }
}

impl<T> IntoPoint for (Result<T, Vec<String>>, Option<f64>) {
    type Point = Result<T, Vec<String>>;
    fn into_point(self) -> Self::Point {
        self.0
    }
}

/// An evaluation's points with no guard estimate (a call's primary
/// evaluation beside its fallback's guarded one, [`with_fallback`]).
pub(crate) fn unguarded<T>(
    points: RusticolResult<Vec<T>>,
) -> RusticolResult<Vec<(T, Option<f64>)>> {
    Ok(points?.into_iter().map(|point| (point, None)).collect())
}

/// One point the per-point fallback evaluated again: the backend that served
/// it (None: it failed there too, or its guard rejected it), the reason
/// (`reduction_policy::fallback_served`, the fallback's failure, or
/// `reduction_policy::fallback_rejected`) and, for a point the fallback
/// served, its guard's estimate (m-lit lit-fallback-guard).
pub(crate) struct FallbackPoint {
    effective: Option<String>,
    reason: String,
    estimate: Option<f64>,
}

/// The points of a batch the per-point fallback evaluated again, per index in
/// the batch ([`FallbackPoint`]).
#[derive(Default)]
pub(crate) struct Fallbacks {
    fallback: Option<String>,
    points: BTreeMap<usize, FallbackPoint>,
}

impl Fallbacks {
    /// The record of the diagnostics: per point id the backend that served
    /// the point (null: failed on both, or rejected by its guard), the reason
    /// and the guard's relative estimate (null: the guard did not run, in
    /// binary64 or on a point the fallback failed, or its estimate is not a
    /// number).
    fn record(&self, ids: &[u64]) -> serde_json::Value {
        serde_json::Value::Object(
            self.points
                .iter()
                .map(|(&index, point)| {
                    (
                        ids[index].to_string(),
                        serde_json::json!({
                            "effective": point.effective,
                            "reason": point.reason,
                            "guard_estimate": point.estimate,
                        }),
                    )
                })
                .collect(),
        )
    }
    /// The indices of the points the fallback served.
    fn served(&self) -> impl Iterator<Item = (usize, &str)> {
        self.points
            .iter()
            .filter(|(_, point)| point.effective.is_some())
            .map(|(&index, point)| (index, point.reason.as_str()))
    }
}

/// The values at `indices` (None: every value, borrowed).
fn pick<'a, T: Clone>(values: &'a [T], indices: Option<&[usize]>) -> std::borrow::Cow<'a, [T]> {
    match indices {
        None => std::borrow::Cow::Borrowed(values),
        Some(indices) => indices.iter().map(|&i| values[i].clone()).collect(),
    }
}

/// `evaluate` of a batch (`None`: every point; or the points at the given
/// indices) on the call's effective backend, then, for an `"auto"` call
/// with a per-point fallback (`Choice::fallback`, m-lit F3 (c): TIRust for
/// a loop-induced process, NinjaRS behind it), the points it refused or
/// failed (also in the enabled binary64 monitoring's own evaluations)
/// again on the fallback, from their original exact inputs, at the call's
/// carrier, after the primary call: each served point takes the fallback's
/// outcome, a point the fallback fails too fails with both backends'
/// reasons; both recorded per point ([`Fallbacks`]). In the extended
/// carriers (`reduction_policy::fallback_guarded`: double-double and
/// arbitrary precision, USER 2026-11-09 ~11:53; a binary64 fallback point
/// is served unguarded, with no estimate) every point the fallback serves
/// is GUARDED (m-lit lit-fallback-guard, DECISIONS 2026-11-09 05:56 (1) and
/// 11:53; the Python API's `loops.v2.reduction.with_fallback`): the
/// fallback's evaluation of the served points (`Some(indices)`) runs the
/// cached reduction-direction re-test at the call's carrier with them and
/// returns the guard's estimate of the outputs the call returns
/// ([`PointOutcome::guard_estimate`]; DECISIONS 2026-11-09 14:30 Q7: the
/// summed value, `V2Runtime::evaluate_extended_guarded`, or each resolved,
/// correlated or selected output of the call's rows only against its own
/// magnitude, `V2Runtime::evaluate_induced_call_guarded`; one evaluation per
/// served point); the estimate is recorded, and a
/// point whose estimate exceeds the guard's limit at the runtime's target
/// (`V2Runtime::target`; `reduction_policy::fallback_accepted`: the target
/// never below `FALLBACK_FLOOR`) fails with the one reason
/// `reduction_policy::fallback_rejected` (a point the guard's re-test
/// fails, the fallback failed; `arithmetic` names the carrier and decides
/// whether the guard runs). A primary backend that refuses the
/// call's plans (TIRust's capability query of a family, a typed
/// `UnsupportedRuntimeCapability` error before any point) leaves all `count`
/// points to the fallback. A call without a fallback (an explicit backend, a
/// Born-interfered or a TIR-only process) runs `evaluate` once, with nothing
/// else.
pub(crate) fn with_fallback<P: PointOutcome>(
    native: &mut V2Runtime,
    choice: &Choice,
    count: usize,
    arithmetic: Arithmetic,
    mut evaluate: impl FnMut(&mut V2Runtime, Option<&[usize]>) -> RusticolResult<Vec<P>>,
) -> RusticolResult<(Vec<P>, Fallbacks)> {
    let mut points = match evaluate(native, None) {
        Ok(points) => points,
        Err(error)
            if choice.fallback.is_some()
                && error.kind()
                    == rusticol_core::RusticolErrorKind::UnsupportedRuntimeCapability =>
        {
            (0..count)
                .map(|_| P::failed(vec![error.to_string()]))
                .collect()
        }
        Err(error) => return Err(error),
    };
    let mut record = Fallbacks {
        fallback: choice.fallback.clone(),
        points: BTreeMap::new(),
    };
    let Some(fallback) = choice.fallback.as_deref() else {
        return Ok((points, record));
    };
    let failed: Vec<usize> = (0..points.len())
        .filter(|&index| points[index].reasons().is_some())
        .collect();
    if failed.is_empty() {
        return Ok((points, record));
    }
    // The guard runs in the extended carriers only (USER 2026-11-09 ~11:53),
    // with the fallback's own evaluation of the served points (its outcome
    // carries the estimate of the outputs the call returns).
    let checking = reduction_policy::fallback_guarded(arithmetic.name());
    let again = native.on_other(|native| evaluate(native, Some(&failed)))?;
    if again.len() != failed.len() {
        return Err(invalid("the fallback evaluation lost points"));
    }
    let guarded: Vec<Option<f64>> = again
        .iter()
        .map(|point| match point.reasons() {
            Some(_) => None,
            None => point.guard_estimate(),
        })
        .collect();
    let target = native.target();
    for ((&index, point), guarded) in failed.iter().zip(again).zip(guarded) {
        let first = points[index].reasons().unwrap_or_default().to_vec();
        let outcome = match (point.reasons(), guarded) {
            (Some(second), _) => Err(second.to_vec()),
            (None, Some(estimate)) if checking => Ok(Some(estimate)),
            (None, None) if !checking => Ok(None),
            (None, _) => {
                return Err(invalid(
                    "a served fallback point without its guard's estimate (the guard runs in the \
                     extended carriers only)",
                ));
            }
        };
        let second = match outcome {
            Ok(Some(estimate)) if !reduction_policy::fallback_accepted(estimate, target) => {
                let reason = reduction_policy::fallback_rejected(
                    &choice.effective,
                    fallback,
                    &first.join("; "),
                    arithmetic.name(),
                    estimate,
                    target,
                );
                record.points.insert(
                    index,
                    FallbackPoint {
                        effective: None,
                        reason: reason.clone(),
                        estimate: Some(estimate),
                    },
                );
                points[index] = P::failed(vec![reason]);
                continue;
            }
            Ok(estimate) => {
                record.points.insert(
                    index,
                    FallbackPoint {
                        effective: Some(fallback.to_owned()),
                        reason: reduction_policy::fallback_served(
                            &choice.effective,
                            fallback,
                            &first.join("; "),
                        ),
                        estimate,
                    },
                );
                points[index] = point;
                continue;
            }
            Err(second) => second,
        };
        let second: Vec<String> = second
            .iter()
            .map(|reason| reduction_policy::fallback_failed(fallback, reason))
            .collect();
        record.points.insert(
            index,
            FallbackPoint {
                effective: None,
                reason: second.join("; "),
                estimate: None,
            },
        );
        let mut reasons = first;
        reasons.extend(second);
        points[index] = P::failed(reasons);
    }
    Ok((points, record))
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
                // A fallback point's cross-backend check is not run (m-lit
                // R6): its other backend is the one that failed it.
                let plain = checks.without_cross_backend();
                let (mut monitored, fallbacks) = with_fallback(
                    native,
                    reduction,
                    momenta.len(),
                    arithmetic,
                    |native, indices| {
                        let checks = if indices.is_some() { &plain } else { checks };
                        native.evaluate_checked(
                            &pick(&momenta, indices),
                            &pick(&scales, indices),
                            poles.v2(),
                            checks,
                        )
                    },
                )?;
                if checks.cross_backend() {
                    for (index, reason) in fallbacks.served() {
                        if let Some(estimate) = &mut monitored[index].1 {
                            estimate.not_run.push((CROSS.to_owned(), reason.to_owned()));
                        }
                    }
                }
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
                    inputs, shape, poles, arithmetic, points, reduction, &fallbacks, stability,
                )
            }
            (Arithmetic::Double, None) => {
                let (points, fallbacks) = with_fallback(
                    native,
                    reduction,
                    momenta.len(),
                    arithmetic,
                    |native, indices| {
                        native.evaluate(
                            &pick(&momenta, indices),
                            &pick(&scales, indices),
                            poles.v2(),
                        )
                    },
                )?;
                // Checks set but not run in binary64: a split pole selection.
                let stability = monitoring.record(SPLIT_MONITORING);
                Self::ordinary(
                    inputs, shape, poles, arithmetic, points, reduction, &fallbacks, stability,
                )
            }
            _ => {
                let (points, fallbacks) = with_fallback(
                    native,
                    reduction,
                    momenta.len(),
                    arithmetic,
                    // The fallback's points run with the guard's cached
                    // check in the call's carrier (one evaluation per
                    // served point, as the Python API's).
                    |native, indices| match indices {
                        None => Ok(native
                            .evaluate_extended(&momenta, &scales, poles.v2(), arithmetic.v2())?
                            .into_iter()
                            .map(|point| (point, None))
                            .collect()),
                        Some(_) => native.evaluate_extended_guarded(
                            &pick(&momenta, indices),
                            &pick(&scales, indices),
                            poles.v2(),
                            arithmetic.v2(),
                        ),
                    },
                )?;
                let points = points.into_iter().map(|(point, _)| point).collect();
                let stability = monitoring.not_run();
                Self::ordinary(
                    inputs, shape, poles, arithmetic, points, reduction, &fallbacks, stability,
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
        fallbacks: &Fallbacks,
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
            fallbacks,
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
            helicity_ids: Vec::new(),
            color_ids: Vec::new(),
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
                let (points, fallbacks) = with_fallback(
                    native,
                    reduction,
                    momenta.len(),
                    arithmetic,
                    |native, indices| {
                        native.evaluate_columns(
                            "resolved",
                            &pick(&momenta, indices),
                            &pick(&scales, indices),
                            &[],
                            poles.v2(),
                        )
                    },
                )?;
                Self::resolved(
                    inputs, shape, poles, arithmetic, axes, &keys, points, reduction, &fallbacks,
                    stability, None,
                )
            }
            _ => {
                let (points, fallbacks) = with_fallback(
                    native,
                    reduction,
                    momenta.len(),
                    arithmetic,
                    // The fallback's points run with the guard's check of
                    // the outputs the call returns (one evaluation per
                    // served point).
                    |native, indices| match indices {
                        None => unguarded(native.evaluate_columns_extended(
                            "resolved",
                            &momenta,
                            &scales,
                            &[],
                            poles.v2(),
                            arithmetic.v2(),
                        )),
                        Some(_) => native.evaluate_columns_guarded(
                            "resolved",
                            &pick(&momenta, indices),
                            &pick(&scales, indices),
                            &[],
                            poles.v2(),
                            arithmetic.v2(),
                        ),
                    },
                )?;
                let points = points.into_iter().map(|(point, _)| point).collect();
                Self::resolved(
                    inputs, shape, poles, arithmetic, axes, &keys, points, reduction, &fallbacks,
                    stability, None,
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
        fallbacks: &Fallbacks,
        stability: serde_json::Value,
        kept: Option<&[usize]>,
    ) -> RusticolResult<Self> {
        // The rows of the layout: every row, or a resolved selection's kept
        // rows (ascending), whose columns alone were formed.
        let every: Vec<usize> = (0..axes.rows.len()).collect();
        let kept = kept.unwrap_or(&every);
        let helicities = kept.len();
        let colours = axes.colour_ids.len();
        // Column index of each (helicity row, colour component).
        let mut index = vec![usize::MAX; helicities * colours];
        for (column, (row, colour)) in keys.iter().enumerate() {
            if let Some(position) = kept.iter().position(|r| r == row) {
                index[position * colours + colour] = column;
            }
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
            fallbacks,
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
            helicity_ids: kept.iter().map(|i| format!("h{i}")).collect(),
            color_ids: axes.colour_ids.clone(),
            diagnostics_json,
            decimal: decimal_values(arithmetic, born, virtual_values),
        })
    }

    /// A loop-induced resolved call with its selection
    /// ([`Runtime::evaluate_selected`]): the outputs of `call`
    /// (`V2Runtime::evaluate_induced_call`, row-major over `rows`, then the
    /// `components`) in the layout point/order/helicity/colour/(epsilon),
    /// no Born.
    fn induced_resolved(
        native: &mut V2Runtime,
        inputs: LoopInputs,
        call: &InducedCall,
        rows: &[usize],
        components: Vec<String>,
        request: InducedRequest<'_>,
    ) -> RusticolResult<Self> {
        let shape = request.shape;
        let (momenta, scales) = exact_inputs(&inputs, shape.external_count)?;
        let keys: Vec<(usize, usize)> = (0..rows.len())
            .flat_map(|row| (0..components.len()).map(move |colour| (row, colour)))
            .collect();
        let axes = Observables {
            rows: vec![Vec::new(); rows.len()],
            colour_ids: components,
            born_orders: 0,
            anchors: 0,
        };
        let mut evaluation = match request.arithmetic {
            Arithmetic::Double => {
                let (points, fallbacks) = with_fallback(
                    native,
                    request.reduction,
                    momenta.len(),
                    request.arithmetic,
                    |native, indices| {
                        native.evaluate_induced_call(
                            call,
                            &pick(&momenta, indices),
                            &pick(&scales, indices),
                            request.poles.v2(),
                        )
                    },
                )?;
                Self::resolved(
                    inputs,
                    Shape {
                        helicities: rows.len(),
                        virtual_color_components: axes.colour_ids.len(),
                        ..shape
                    },
                    request.poles,
                    request.arithmetic,
                    &axes,
                    &keys,
                    points,
                    request.reduction,
                    &fallbacks,
                    request.stability,
                    None,
                )?
            }
            arithmetic => {
                let (points, fallbacks) = with_fallback(
                    native,
                    request.reduction,
                    momenta.len(),
                    request.arithmetic,
                    // The fallback's points run with the guard's check of
                    // the call's outputs (its rows only).
                    |native, indices| match indices {
                        None => unguarded(native.evaluate_induced_call_extended(
                            call,
                            &momenta,
                            &scales,
                            request.poles.v2(),
                            arithmetic.v2(),
                        )),
                        Some(_) => native.evaluate_induced_call_guarded(
                            call,
                            &pick(&momenta, indices),
                            &pick(&scales, indices),
                            &[],
                            request.poles.v2(),
                            arithmetic.v2(),
                        ),
                    },
                )?;
                let points = points.into_iter().map(|(point, _)| point).collect();
                Self::resolved(
                    inputs,
                    Shape {
                        helicities: rows.len(),
                        virtual_color_components: axes.colour_ids.len(),
                        ..shape
                    },
                    request.poles,
                    request.arithmetic,
                    &axes,
                    &keys,
                    points,
                    request.reduction,
                    &fallbacks,
                    request.stability,
                    None,
                )?
            }
        };
        evaluation.helicity_ids = rows.iter().map(|row| format!("h{row}")).collect();
        Ok(evaluation)
    }

    /// A loop-induced helicity selection ([`Runtime::evaluate_selected`]):
    /// per point the sum over its kept rows (`by_point`), in the ordinary
    /// layout. Points sharing their rows are evaluated in one batch (each
    /// point's values are those of its own batch: the executor and the tree
    /// fill run every lane alone), in their order.
    fn induced_selected(
        native: &mut V2Runtime,
        inputs: LoopInputs,
        by_point: &[Vec<usize>],
        request: InducedRequest<'_>,
    ) -> RusticolResult<Self> {
        let shape = request.shape;
        let (momenta, scales) = exact_inputs(&inputs, shape.external_count)?;
        let mut groups: Vec<(&Vec<usize>, Vec<usize>)> = Vec::new();
        for (point, rows) in by_point.iter().enumerate() {
            match groups.iter_mut().find(|(kept, _)| *kept == rows) {
                Some((_, points)) => points.push(point),
                None => groups.push((rows, vec![point])),
            }
        }
        fn summed<N: Clone>(point: ColumnPoint<N>) -> V2Point<N> {
            point.map(|columns| {
                let column = columns.into_iter().next().expect("one selection output");
                (
                    Vec::new(),
                    column
                        .virtual_values
                        .into_iter()
                        .map(|laurent| laurent.into_iter().map(|z| z.re).collect())
                        .collect(),
                )
            })
        }
        let ids =
            |rows: &[usize]| -> Vec<String> { rows.iter().map(|r| format!("h{r}")).collect() };
        macro_rules! run {
            ($evaluate:expr) => {{
                let mut points: Vec<Option<_>> = (0..by_point.len()).map(|_| None).collect();
                // The per-point fallback of each group, by the call's point
                // index (m-lit F3 (c)).
                let mut fallbacks = Fallbacks {
                    fallback: request.reduction.fallback.clone(),
                    points: BTreeMap::new(),
                };
                for (rows, members) in &groups {
                    let call = InducedCall::Selection {
                        rows: (*rows).clone(),
                    };
                    let part_momenta: Vec<Vec<Rational>> =
                        members.iter().map(|&p| momenta[p].clone()).collect();
                    let part_scales: Vec<Rational> =
                        members.iter().map(|&p| scales[p].clone()).collect();
                    let (found, part) = with_fallback(
                        native,
                        request.reduction,
                        members.len(),
                        request.arithmetic,
                        |native, indices| {
                            $evaluate(
                                native,
                                &call,
                                indices.is_some(),
                                &pick(&part_momenta, indices),
                                &pick(&part_scales, indices),
                            )
                        },
                    )?;
                    for (local, record) in part.points {
                        fallbacks.points.insert(members[local], record);
                    }
                    for (&point, value) in members.iter().zip(found) {
                        points[point] = Some(summed(value.into_point()));
                    }
                }
                let points: Vec<_> = points
                    .into_iter()
                    .map(|point| point.expect("every point is in a group"))
                    .collect();
                let mut evaluation = Self::ordinary(
                    inputs,
                    shape,
                    request.poles,
                    request.arithmetic,
                    points,
                    request.reduction,
                    &fallbacks,
                    request.stability.clone(),
                )?;
                evaluation.diagnostics_json = with_selection(
                    &evaluation.diagnostics_json,
                    &by_point.iter().map(|rows| ids(rows)).collect::<Vec<_>>(),
                )?;
                Ok(evaluation)
            }};
        }
        let poles = request.poles.v2();
        match request.arithmetic {
            Arithmetic::Double => {
                run!(|native: &mut V2Runtime,
                      call: &InducedCall,
                      _fallback: bool,
                      m: &[Vec<Rational>],
                      sc: &[Rational]| native
                    .evaluate_induced_call(call, m, sc, poles))
            }
            arithmetic => {
                // The fallback's points run with the guard's check of the
                // group's selection (its rows only).
                run!(|native: &mut V2Runtime,
                      call: &InducedCall,
                      fallback: bool,
                      m: &[Vec<Rational>],
                      sc: &[Rational]| if fallback {
                    native.evaluate_induced_call_guarded(call, m, sc, &[], poles, arithmetic.v2())
                } else {
                    unguarded(native.evaluate_induced_call_extended(
                        call,
                        m,
                        sc,
                        poles,
                        arithmetic.v2(),
                    ))
                })
            }
        }
    }
}

impl Evaluation {
    /// A Born-interfered ordinary selection ([`Runtime::born_selected`]):
    /// per group of points sharing their kept rows one native call
    /// (`V2Runtime::evaluate_selection`, the masks of one call on the
    /// selection template), each point's Born per order and virtual per
    /// sector in the ordinary layout, the selection's helicity IDs per point
    /// in the diagnostics.
    fn born_selected(
        native: &mut V2Runtime,
        inputs: LoopInputs,
        by_point: &[Vec<usize>],
        request: InducedRequest<'_>,
    ) -> RusticolResult<Self> {
        let shape = request.shape;
        let (momenta, scales) = exact_inputs(&inputs, shape.external_count)?;
        let mut groups: Vec<(&Vec<usize>, Vec<usize>)> = Vec::new();
        for (point, rows) in by_point.iter().enumerate() {
            match groups.iter_mut().find(|(kept, _)| *kept == rows) {
                Some((_, points)) => points.push(point),
                None => groups.push((rows, vec![point])),
            }
        }
        fn one<N: Clone>(point: ColumnPoint<N>) -> V2Point<N> {
            point.map(|columns| {
                let column = columns.into_iter().next().expect("one selection output");
                (
                    column.born.into_iter().map(|z| z.re).collect(),
                    column
                        .virtual_values
                        .into_iter()
                        .map(|laurent| laurent.into_iter().map(|z| z.re).collect())
                        .collect(),
                )
            })
        }
        let ids =
            |rows: &[usize]| -> Vec<String> { rows.iter().map(|r| format!("h{r}")).collect() };
        macro_rules! run {
            ($evaluate:expr) => {{
                let mut points: Vec<Option<_>> = (0..by_point.len()).map(|_| None).collect();
                let mut fallbacks = Fallbacks {
                    fallback: request.reduction.fallback.clone(),
                    points: BTreeMap::new(),
                };
                for (rows, members) in &groups {
                    let kept: BTreeSet<usize> = rows.iter().copied().collect();
                    let part_momenta: Vec<Vec<Rational>> =
                        members.iter().map(|&p| momenta[p].clone()).collect();
                    let part_scales: Vec<Rational> =
                        members.iter().map(|&p| scales[p].clone()).collect();
                    let (found, part) = with_fallback(
                        native,
                        request.reduction,
                        members.len(),
                        request.arithmetic,
                        |native, indices| {
                            $evaluate(
                                native,
                                &kept,
                                &pick(&part_momenta, indices),
                                &pick(&part_scales, indices),
                            )
                        },
                    )?;
                    for (local, record) in part.points {
                        fallbacks.points.insert(members[local], record);
                    }
                    for (&point, value) in members.iter().zip(found) {
                        points[point] = Some(one(value));
                    }
                }
                let points: Vec<_> = points
                    .into_iter()
                    .map(|point| point.expect("every point is in a group"))
                    .collect();
                let mut evaluation = Self::ordinary(
                    inputs,
                    shape,
                    request.poles,
                    request.arithmetic,
                    points,
                    request.reduction,
                    &fallbacks,
                    request.stability.clone(),
                )?;
                evaluation.diagnostics_json = with_selection(
                    &evaluation.diagnostics_json,
                    &by_point.iter().map(|rows| ids(rows)).collect::<Vec<_>>(),
                )?;
                Ok(evaluation)
            }};
        }
        let poles = request.poles.v2();
        match request.arithmetic {
            Arithmetic::Double => {
                run!(|native: &mut V2Runtime,
                      kept: &BTreeSet<usize>,
                      m: &[Vec<Rational>],
                      sc: &[Rational]| native.evaluate_selection(
                    Some(kept),
                    None,
                    m,
                    sc,
                    poles
                ))
            }
            arithmetic => {
                run!(|native: &mut V2Runtime,
                      kept: &BTreeSet<usize>,
                      m: &[Vec<Rational>],
                      sc: &[Rational]| native.evaluate_selection_extended(
                    Some(kept),
                    None,
                    m,
                    sc,
                    poles,
                    arithmetic.v2()
                ))
            }
        }
    }

    /// A Born-interfered resolved selection ([`Runtime::born_selected`]):
    /// the kept rows' resolved columns only (`V2Runtime::
    /// evaluate_resolved_selection`), in the layout point/order/helicity/
    /// colour/(epsilon) over the kept rows, with their helicity IDs.
    fn born_resolved_selected(
        native: &mut V2Runtime,
        inputs: LoopInputs,
        axes: &Observables,
        kept: &[usize],
        request: InducedRequest<'_>,
    ) -> RusticolResult<Self> {
        let shape = Shape {
            helicities: kept.len(),
            ..request.shape
        };
        let (momenta, scales) = exact_inputs(&inputs, shape.external_count)?;
        let rows: BTreeSet<usize> = kept.iter().copied().collect();
        let poles = request.poles.v2();
        let precision = request.arithmetic.v2();
        let mut keys = Vec::new();
        macro_rules! run {
            ($variant:ident) => {{
                let (points, fallbacks) = with_fallback(
                    native,
                    request.reduction,
                    momenta.len(),
                    request.arithmetic,
                    |native, indices| {
                        let (found, points) = native.evaluate_resolved_selection(
                            Some(&rows),
                            None,
                            &pick(&momenta, indices),
                            &pick(&scales, indices),
                            poles,
                            precision,
                        )?;
                        keys = found;
                        match points {
                            ResolvedPoints::$variant(points) => Ok(points),
                            _ => Err(invalid("a resolved selection of another carrier")),
                        }
                    },
                )?;
                Self::resolved(
                    inputs,
                    shape,
                    request.poles,
                    request.arithmetic,
                    axes,
                    &keys,
                    points,
                    request.reduction,
                    &fallbacks,
                    request.stability.clone(),
                    Some(kept),
                )
            }};
        }
        match request.arithmetic {
            Arithmetic::Double => run!(Double),
            _ => run!(Decimal),
        }
    }
}

/// The shared inputs of a loop-induced call's assembly.
struct InducedRequest<'a> {
    shape: Shape,
    poles: Poles,
    arithmetic: Arithmetic,
    reduction: &'a Choice,
    stability: serde_json::Value,
}

/// The record of a point the kinematics settings reach but do not improve
/// (the setting "off"; the Python API's `kinematics.off_record`).
fn kinematics_off_record(setting: &str) -> serde_json::Value {
    serde_json::json!({
        "requested": setting,
        "fallback": true,
        "status": "off",
        "produced": null,
        "attempts": [],
        "exactness": null,
        "iterations": null,
        "scale": null,
        "moved": 0.0,
        "warning": null,
        "warning_text": null,
    })
}

/// The diagnostics record with each point's kinematics improvement record
/// (m-lit lit-psimprove: `"kinematics"`, by point in input order, the
/// Python API's `VirtualLaneDiagnostic.kinematics`; absent when no record
/// applies, a binary64 call without the binary64 switch).
pub(crate) fn with_kinematics(
    diagnostics: &mut String,
    records: Option<Vec<serde_json::Value>>,
) -> RusticolResult<()> {
    let Some(records) = records else {
        return Ok(());
    };
    let mut record: serde_json::Value = serde_json::from_str(diagnostics).map_err(serialization)?;
    record["kinematics"] = serde_json::Value::Array(records);
    *diagnostics = serde_json::to_string(&record).map_err(serialization)?;
    Ok(())
}

/// The diagnostics record with the selection's helicity IDs per point
/// (`"selection": {"helicity_ids": [[...] per point]}`).
fn with_selection(diagnostics: &str, ids: &[Vec<String>]) -> RusticolResult<String> {
    let mut record: serde_json::Value = serde_json::from_str(diagnostics).map_err(serialization)?;
    record["selection"] = serde_json::json!({ "helicity_ids": ids });
    serde_json::to_string(&record).map_err(serialization)
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
const SPLIT_MONITORING: &str = "the native SDKs monitor the pole selections \"all\" and \
    \"none\" (the Python API estimates the \"uv\" and \"ir\" selections)";
const SELECTION_MONITORING: &str = "the native SDKs monitor the ordinary evaluations without \
    a selection (the Python API estimates the helicity selections and the resolved and \
    correlated outputs)";

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
    fn not_selected(&self) -> serde_json::Value {
        self.record(SELECTION_MONITORING)
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
pub(crate) fn diagnostics(
    requested: &[u64],
    successful: &[u64],
    failures: &BTreeMap<u64, Vec<String>>,
    poles: Poles,
    arithmetic: Arithmetic,
    reduction: &Choice,
    fallbacks: &Fallbacks,
    stability: serde_json::Value,
) -> RusticolResult<String> {
    serde_json::to_string(&serde_json::json!({
        // The stability monitoring (`Runtime::set_stability_checks`): null
        // without checks, else per successful point its estimate (the
        // Python API's `VirtualStabilityEstimate`), or why none is formed.
        "stability": stability,
        // The reduction backend of the call (the Python API's
        // `VirtualReduction`): requested, effective and why; the per-point
        // fallback (m-lit F3 (c): null without one) and how many points it
        // evaluated again.
        "reduction": {
            "requested": reduction.requested,
            "effective": reduction.effective,
            "reason": reduction.reason,
            "fallback": fallbacks.fallback,
            "fallback_points": fallbacks.points.len(),
            // The call's unchecked setting (m-lit lit-unchecked).
            "unchecked": reduction.unchecked,
        },
        // Per point the fallback evaluated again (by point id): the backend
        // that served it (null: it failed there too, with both backends'
        // reasons in "failures") and the reason. Every other point is the
        // call's effective backend's.
        "fallbacks": fallbacks.record(requested),
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
