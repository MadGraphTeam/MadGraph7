// SPDX-License-Identifier: 0BSD
//! Source-only standalone virtual SDK. The safe Rust and C interfaces share the
//! existing native executor; neither links Python or implements loop reduction.
mod correlated;
mod ffi;
pub use correlated::{Complex64, CorrelatedEvaluation, CorrelatedShape, SpinSources};
pub use ffi::*;
use rusticol_core::loop_execution::{CompleteLoopResult, LoopInputs};
use rusticol_core::loop_runtime::LoopRuntimePlan;
pub use rusticol_core::source_bindings::SourceOverrides;
use rusticol_core::{RusticolError, RusticolResult};
use rusticol_loop_reduction::runtime::{Batch, Data};
use rusticol_loop_reduction::v2_runtime::{V2Poles, V2Runtime, is_v2_artifact};
use rusticol_loop_reduction::verification::VerificationDiagnostic;
pub use rusticol_loop_reduction::verification::{ArithmeticSpec, VerificationPolicy};
use rusticol_loop_reduction::virtual_runtime::VirtualRuntime;
use serde::Serialize;
use std::{collections::BTreeMap, path::Path, sync::Arc};

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

/// The runtime behind an SDK handle: a plan v1-v3 loop runtime, or a plan v4
/// process with native point inputs (`v2_runtime`, WP3.1 I6).
enum Engine {
    V3(VirtualRuntime),
    V4(Box<V2Runtime>),
}

const V4_ONLY_ORDINARY: &str = "plan v4 SDK runtimes serve ordinary (unresolved) \
    binary64 virtuals with model R2; resolved, correlated, verified and \
    rescued evaluations go through the Python API";

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
    born_anchored_color: bool,
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
        let native = VirtualRuntime::open_with_policy(
            path,
            process,
            "double",
            None,
            parameters,
            r2_strategy,
            policy,
        )?;
        let plan: LoopRuntimePlan =
            serde_json::from_str(&native.plan_metadata_json()?).map_err(serialization)?;
        let shape = Shape {
            external_count: native.external_count(),
            born_orders: native.metadata().orders.born.len(),
            virtual_orders: native.metadata().orders.r#virtual.len(),
            helicities: plan.helicities.len(),
            born_color_components: plan.resolved_color_component_count(),
            virtual_color_components: plan.resolved_color_component_count(),
            epsilon_count: 3,
            resolved_available: u32::from(native.metadata().resolved),
        };
        let color_components = plan.born_color_components.as_ref().map_or_else(
            || vec![native.metadata().color_accuracy.clone()],
            |partition| partition.ids.clone(),
        );
        let history_ids = native
            .selected_histories()
            .iter()
            .map(|h| h.id.clone())
            .collect();
        let spin_classes = native.selected_spin_classes().to_vec();
        Ok(Self {
            engine: Engine::V3(native),
            shape,
            poles: Poles::All,
            color_components,
            born_anchored_color: plan.born_color_components.is_some(),
            history_ids,
            spin_classes,
        })
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
            born_anchored_color: false,
            history_ids: vec![],
            spin_classes: vec![],
        })
    }
    /// The plan v1-v3 runtime (plan v4 serves ordinary virtuals only).
    fn v3(&self) -> RusticolResult<&VirtualRuntime> {
        match &self.engine {
            Engine::V3(native) => Ok(native),
            Engine::V4(_) => Err(invalid(V4_ONLY_ORDINARY)),
        }
    }
    fn v3_mut(&mut self) -> RusticolResult<&mut VirtualRuntime> {
        match &mut self.engine {
            Engine::V3(native) => Ok(native),
            Engine::V4(_) => Err(invalid(V4_ONLY_ORDINARY)),
        }
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }
    /// The pole selection of later evaluations (default all): "all",
    /// "none", "uv" or "ir", the Python API's four (`Poles`). A plan v4
    /// runtime serves all four; plan v1-v3 runtimes evaluate every pole.
    pub fn set_poles(&mut self, poles: Poles) -> RusticolResult<()> {
        if poles != Poles::All && matches!(self.engine, Engine::V3(_)) {
            return Err(invalid(
                "plan v1-v3 SDK runtimes evaluate every pole (poles \"all\")",
            ));
        }
        self.poles = poles;
        Ok(())
    }
    pub fn poles(&self) -> Poles {
        self.poles
    }
    pub fn verification_policy(&self) -> Option<&VerificationPolicy> {
        match &self.engine {
            Engine::V3(native) => native.verification_policy(),
            Engine::V4(_) => None,
        }
    }
    /// Existing artifact metadata includes exact order pairings and schemes.
    pub fn metadata_json(&self) -> RusticolResult<String> {
        match &self.engine {
            Engine::V3(native) => native.metadata_json(),
            Engine::V4(native) => native.metadata_json(),
        }
    }
    pub fn capabilities_json(&self) -> RusticolResult<String> {
        if let Engine::V4(native) = &self.engine {
            let metadata = native.metadata();
            return serde_json::to_string(&serde_json::json!({
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
            .map_err(serialization);
        }
        let native = self.v3()?;
        serde_json::to_string(&serde_json::json!({
            "contribution_kind":"virtual", "execution_mode":"loop_recurrence",
            "arithmetic":["double"], "epsilon_powers":[0,-1,-2],
            "resolved": native.metadata().resolved, "correlations":native.metadata().correlations,
            "correlation_histories":self.history_ids,
            "spin_correlation_classes":self.spin_classes,
            "correlated_resolved":false,
            "correlated_layout":"point/history/order/epsilon; complex binary64",
            "color_accuracy":native.metadata().color_accuracy,
            "color_contraction":native.color_contraction_metadata(),
            "color_components":self.color_components,
            "color_component_semantics":if self.born_anchored_color {
                "signed canonical Born tensor groups, potentially overcomplete; not sampling probabilities"
            } else { "contracted colour" },
            "available_r2_strategies": native.metadata().available_r2_strategies,
            "r2_strategy":"model", "explicit_verification":true,
            "source_only":true,
            "retained_verification_policy":self.verification_policy().map(|p| serde_json::json!({
                "higher":p.higher, "relative_tolerance":p.relative_tolerance,
                "absolute_tolerance":p.absolute_tolerance, "verify_successful":p.verify_successful
            })),
            "source_binding_rescue":"configured policy retains exact inputs after a numerical primary failure; complete-result agreement remains required",
            "normalization":"physical couplings and inclusive initial spin/color average; resolved helicities retain that average"
        })).map_err(serialization)
    }
    /// Update model parameters. A plan v4 process validates the whole
    /// mapping first, as the Python API's `set_model_parameters`: unknown
    /// names, the renormalisation scale's parameter (set per point by
    /// `mu_r2`) and a value other than the one generation fixed are refused
    /// with a typed `ModelParameter` error, and a refused update changes
    /// nothing (its revision is always 0).
    pub fn update_parameters(&mut self, values: &SourceOverrides) -> RusticolResult<u64> {
        match &mut self.engine {
            Engine::V3(native) => native.update_parameters(values),
            Engine::V4(native) => {
                native.set_parameters(real_overrides(values)?)?;
                Ok(0)
            }
        }
    }
    pub fn clear(&mut self) {
        if let Engine::V3(native) = &mut self.engine {
            native.clear();
        }
    }
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
        if let Engine::V4(native) = &mut self.engine {
            if let Some(strategy) = strategy {
                v4_r2_strategy(strategy)?;
            }
            if resolved || policy.is_some() {
                return Err(invalid(V4_ONLY_ORDINARY));
            }
            return Evaluation::from_v4(native, inputs, self.shape, self.poles);
        }
        let effective = policy.or_else(|| self.verification_policy()).cloned();
        let native = self.v3_mut()?;
        let (result, verification) = match effective.as_ref() {
            Some(policy) => {
                let output = native.evaluate_verified(inputs, strategy, resolved, policy)?;
                (output.result, output.diagnostics)
            }
            None => (native.evaluate_partial(inputs, strategy, resolved)?, vec![]),
        };
        Evaluation::from_native(result, verification, self.shape, resolved)
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

/// Real model parameter overrides: a plan v4 process takes real external
/// parameters, an exactly zero imaginary part (any exact spelling), refused
/// otherwise in the Python API's words (`artifact._real_parameters`).
fn real_overrides(values: &SourceOverrides) -> RusticolResult<Vec<(&str, &str)>> {
    use rusticol_core::loop_v2::dress::parse_rational;
    values
        .iter()
        .map(|(name, [re, im])| match parse_rational(im) {
            Ok(imaginary) if imaginary.is_zero() => Ok((name.as_str(), re.as_str())),
            _ => Err(RusticolError::model_parameter(format!(
                "parameter '{name}': v2 loop runtimes take real external parameters"
            ))),
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
    fn from_native(
        result: CompleteLoopResult<Arc<Batch>>,
        verification: Vec<VerificationDiagnostic>,
        shape: Shape,
        resolved: bool,
    ) -> RusticolResult<Self> {
        let helicities = if resolved { shape.helicities } else { 1 };
        let born_colors = if resolved {
            shape.born_color_components
        } else {
            1
        };
        let virtual_colors = if resolved {
            shape.virtual_color_components
        } else {
            1
        };
        let rows_per_point = product(&[helicities, born_colors])?;
        if born_colors != virtual_colors
            || rows_per_point == 0
            || result.born.rows() != product(&[result.point_ids.len(), rows_per_point])?
        {
            return Err(invalid("native resolved point identity shape differs"));
        }
        let successful_point_ids = result.point_ids.clone();
        let (statuses, diagnostics_json) = diagnostics(&result, &verification)?;
        let born = pack(
            &result.born,
            successful_point_ids.len(),
            shape.born_orders,
            helicities,
            born_colors,
            1,
        )?;
        let virtual_values = pack(
            &result.components[0],
            successful_point_ids.len(),
            shape.virtual_orders,
            helicities,
            virtual_colors,
            3,
        )?;
        Ok(Self {
            requested_point_ids: result.requested_point_ids,
            successful_point_ids,
            statuses,
            born,
            virtual_values,
            shape,
            resolved,
            diagnostics_json,
        })
    }
}

fn diagnostics(
    result: &CompleteLoopResult<Arc<Batch>>,
    verification: &[VerificationDiagnostic],
) -> RusticolResult<(Vec<PointStatus>, String)> {
    let verified: BTreeMap<_, _> = verification.iter().map(|v| (v.point_id, v)).collect();
    let successful_point_ids = &result.point_ids;
    let successes: std::collections::BTreeSet<_> = successful_point_ids.iter().copied().collect();
    let statuses = result
        .requested_point_ids
        .iter()
        .map(|id| match verified.get(id) {
            Some(v) if v.status == "accuracy-not-established" => {
                PointStatus::AccuracyNotEstablished
            }
            Some(v) if v.status == "rescued" => PointStatus::Rescued,
            Some(v) if v.accuracy_established => PointStatus::Verified,
            _ if successes.contains(id) => PointStatus::Success,
            _ => PointStatus::Failed,
        })
        .collect();
    let diagnostics_json = serde_json::to_string(&serde_json::json!({
        "requested_point_ids":result.requested_point_ids,
        "successful_point_ids":successful_point_ids,
        "failures":result.failures, "verification":verification,
        "reduction_statuses":result.statuses,
        "source_revision":result.source_revision, "r2_strategy":result.r2_strategy,
        "primary_color_contraction":result.primary_color_contraction,
    }))
    .map_err(serialization)?;
    Ok((statuses, diagnostics_json))
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
