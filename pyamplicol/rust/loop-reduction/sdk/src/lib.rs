// SPDX-License-Identifier: 0BSD
//! Source-only standalone virtual SDK. The safe Rust and C interfaces share the
//! existing native executor; neither links Python or implements loop reduction.
mod correlated;
mod ffi;
pub use correlated::{Complex64, CorrelatedEvaluation, CorrelatedShape, SpinSources};
pub use ffi::*;
use rusticol_core::loop_execution::LoopInputs;
use rusticol_core::loop_artifact::{LOOP_RUNTIME_CAPABILITY, LOOP_V2_RUNTIME_CAPABILITY};
pub use rusticol_core::source_bindings::SourceOverrides;
use rusticol_core::{RusticolError, RusticolResult};
#[cfg(test)]
use rusticol_loop_reduction::runtime::{Batch, Data};
use rusticol_loop_reduction::v2_runtime::{
    V2Poles, V2Runtime, decimal_parameter, is_v2_artifact, named_parameters,
};
pub use rusticol_loop_reduction::verification::{ArithmeticSpec, VerificationPolicy};
use serde::Serialize;
use std::{collections::BTreeMap, path::Path};

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
}

/// The runtime behind an SDK handle: a plan v4 process with native point
/// inputs (`v2_runtime`, WP3.1 I6). The plan v1-v3 runtime of the removed
/// legacy one-loop engine is refused at open (`LEGACY_ENGINE_REMOVED`).
enum Engine {
    V4(Box<V2Runtime>),
}

const V4_ONLY_ORDINARY: &str = "plan v4 SDK runtimes serve ordinary (unresolved) \
    binary64 virtuals with model R2; resolved, correlated, verified and \
    rescued evaluations go through the Python API";

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
    color_components: Vec<String>,
    history_ids: Vec<String>,
    spin_classes: Vec<Vec<usize>>,
}

impl Runtime {
    pub fn open(
        path: impl AsRef<Path>,
        process: Option<&str>,
        parameters: Option<&SourceOverrides>,
        r2_strategy: &str,
    ) -> RusticolResult<Self> {
        Self::open_with_policy(path, process, parameters, r2_strategy, None)
    }
    /// Retain an explicit policy before exact source parameters are realized.
    /// A per-call policy replaces this default only for that evaluation.
    pub fn open_with_policy(
        path: impl AsRef<Path>,
        process: Option<&str>,
        parameters: Option<&SourceOverrides>,
        r2_strategy: &str,
        policy: Option<&VerificationPolicy>,
    ) -> RusticolResult<Self> {
        if is_v2_artifact(path.as_ref())? {
            return Self::open_v4(path, process, parameters, r2_strategy, policy);
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
        policy: Option<&VerificationPolicy>,
    ) -> RusticolResult<Self> {
        v4_r2_strategy(r2_strategy)?;
        if policy.is_some() {
            return Err(invalid(V4_ONLY_ORDINARY));
        }
        let mut native = V2Runtime::open(path, process, None)?;
        if let Some(values) = parameters {
            native.set_parameters(real_overrides(values)?)?;
        }
        let shape = Shape {
            external_count: native.external_count(),
            born_orders: native.born_orders(),
            virtual_orders: native.sectors(),
            helicities: native.helicities(),
            born_color_components: 1,
            virtual_color_components: 1,
            epsilon_count: 3,
            resolved_available: 0,
        };
        Ok(Self {
            color_components: vec![native.metadata().color_accuracy.clone()],
            engine: Engine::V4(Box::new(native)),
            shape,
            poles: Poles::All,
            history_ids: vec![],
            spin_classes: vec![],
        })
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }
    /// The pole selection of later evaluations (default all): "all",
    /// "none", "uv" or "ir", the Python API's four (`Poles`).
    pub fn set_poles(&mut self, poles: Poles) -> RusticolResult<()> {
        self.poles = poles;
        Ok(())
    }
    pub fn poles(&self) -> Poles {
        self.poles
    }
    /// Plan v4 runtimes retain no verification policy (escalation is the
    /// caller's, through the Python API).
    pub fn verification_policy(&self) -> Option<&VerificationPolicy> {
        None
    }
    /// Existing artifact metadata includes exact order pairings and schemes.
    pub fn metadata_json(&self) -> RusticolResult<String> {
        let Engine::V4(native) = &self.engine;
        native.metadata_json()
    }
    pub fn capabilities_json(&self) -> RusticolResult<String> {
        let Engine::V4(native) = &self.engine;
        let metadata = native.metadata();
        serde_json::to_string(&serde_json::json!({
                "contribution_kind":"virtual", "execution_mode":"loop_recurrence",
                "engine":"v2", "plan_sha256":native.plan_sha256(),
                "point_inputs":"native-v2",
                "arithmetic":["double"], "epsilon_powers":[0,-1,-2],
                "pole_selections":["all","none","uv","ir"],
                "poles":self.poles.name(),
                "resolved":false, "correlations":false,
                "correlation_histories":Vec::<String>::new(),
                "spin_correlation_classes":Vec::<Vec<usize>>::new(),
                "correlated_resolved":false,
                "color_accuracy":metadata.color_accuracy,
                "color_contraction":metadata.color_contraction,
                "color_components":self.color_components,
                "color_component_semantics":"contracted colour",
                "available_r2_strategies":["model"],
                "r2_strategy":"model", "explicit_verification":false,
                "source_only":true,
                "retained_verification_policy":serde_json::Value::Null,
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
    /// so higher arithmetic retries preserve the original C/Rust input numbers.
    pub fn evaluate(
        &mut self,
        momenta: &[f64],
        point_ids: &[u64],
        scales: &[f64],
        resolved: bool,
        strategy: Option<&str>,
        policy: Option<&VerificationPolicy>,
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
            policy,
        )
    }
    /// Decimal literals remain exact input snapshots; the returned carrier is
    /// binary64, including any explicitly verified/rescued values.
    pub fn evaluate_decimal(
        &mut self,
        momenta: Vec<String>,
        point_ids: Vec<u64>,
        scales: Vec<String>,
        resolved: bool,
        strategy: Option<&str>,
        policy: Option<&VerificationPolicy>,
    ) -> RusticolResult<Evaluation> {
        let inputs = self.prepare_inputs(momenta, point_ids, scales)?;
        let Engine::V4(native) = &mut self.engine;
        if let Some(strategy) = strategy {
            v4_r2_strategy(strategy)?;
        }
        if resolved || policy.is_some() {
            return Err(invalid(V4_ONLY_ORDINARY));
        }
        Evaluation::from_v4(native, inputs, self.shape, self.poles)
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

impl Evaluation {
    /// One ordinary binary64 batch of a plan v4 process: Born per order and
    /// the complete virtual per sector and epsilon power (with `Poles::None`
    /// the finite part, the poles quiet NaNs), failed points without rows.
    fn from_v4(
        native: &mut V2Runtime,
        inputs: LoopInputs,
        shape: Shape,
        poles: Poles,
    ) -> RusticolResult<Self> {
        use rusticol_core::loop_v2::dress::parse_rational;
        let width = 4 * shape.external_count;
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
        let points = native.evaluate(&momenta, &scales, poles.v2())?;
        let mut successful_point_ids = Vec::new();
        let mut statuses = Vec::with_capacity(points.len());
        let mut failures = BTreeMap::new();
        let mut born = Vec::new();
        let mut virtual_values = Vec::new();
        for (id, point) in inputs.point_ids.iter().zip(points) {
            match point {
                Ok((born_values, sectors)) => {
                    successful_point_ids.push(*id);
                    statuses.push(PointStatus::AccuracyNotEstablished);
                    born.extend(born_values);
                    for laurent in sectors {
                        let finite = laurent.len();
                        virtual_values.extend(laurent);
                        // Absent poles (finite-only).
                        virtual_values.extend((finite..shape.epsilon_count).map(|_| f64::NAN));
                    }
                }
                Err(reasons) => {
                    statuses.push(PointStatus::Failed);
                    failures.insert(*id, reasons);
                }
            }
        }
        let diagnostics_json = serde_json::to_string(&serde_json::json!({
            "requested_point_ids":inputs.point_ids,
            "successful_point_ids":successful_point_ids,
            "failures":failures, "verification":[],
            "r2_strategy":"model", "engine":"v2", "poles":poles.name(),
            "primary_color_contraction":"direct",
        }))
        .map_err(serialization)?;
        Ok(Self {
            requested_point_ids: inputs.point_ids,
            successful_point_ids,
            statuses,
            born,
            virtual_values,
            shape,
            resolved: false,
            diagnostics_json,
        })
    }
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
