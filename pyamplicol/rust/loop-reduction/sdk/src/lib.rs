// SPDX-License-Identifier: 0BSD
//! Source-only standalone virtual SDK. The safe Rust and C interfaces share the
//! existing native executor; neither links Python or implements loop reduction.
mod ffi;
pub use ffi::*;
use rusticol_core::loop_execution::{CompleteLoopResult, LoopInputs};
use rusticol_core::loop_runtime::LoopRuntimePlan;
pub use rusticol_core::source_bindings::SourceOverrides;
use rusticol_core::{RusticolError, RusticolResult};
use rusticol_loop_reduction::runtime::{Batch, Data};
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

pub struct Runtime {
    native: VirtualRuntime,
    shape: Shape,
    color_components: Vec<String>,
}

impl Runtime {
    pub fn open(
        path: impl AsRef<Path>,
        process: Option<&str>,
        parameters: Option<&SourceOverrides>,
        r2_strategy: &str,
    ) -> RusticolResult<Self> {
        let native = VirtualRuntime::open(path, process, "double", None, parameters, r2_strategy)?;
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
        Ok(Self {
            native,
            shape,
            color_components,
        })
    }
    pub fn shape(&self) -> Shape {
        self.shape
    }
    /// Existing artifact metadata includes exact order pairings and schemes.
    pub fn metadata_json(&self) -> RusticolResult<String> {
        self.native.metadata_json()
    }
    pub fn capabilities_json(&self) -> RusticolResult<String> {
        serde_json::to_string(&serde_json::json!({
            "contribution_kind":"virtual", "execution_mode":"loop_recurrence",
            "arithmetic":["double"], "epsilon_powers":[0,-1,-2],
            "resolved": self.native.metadata().resolved, "correlations":false,
            "color_accuracy":self.native.metadata().color_accuracy,
            "color_components":self.color_components,
            "color_component_semantics":if self.native.metadata().color_accuracy == "lc" {
                "signed canonical Born tensor groups, potentially overcomplete; not sampling probabilities"
            } else { "contracted colour" },
            "available_r2_strategies": self.native.metadata().available_r2_strategies,
            "r2_strategy":"model", "explicit_verification":true,
            "source_only":true,
            "verification_limitations":"source-global parameter binding errors precede subset rescue",
            "normalization":"physical couplings and inclusive initial spin/color average; resolved helicities retain that average"
        })).map_err(serialization)
    }
    pub fn update_parameters(&self, values: &SourceOverrides) -> RusticolResult<u64> {
        self.native.update_parameters(values)
    }
    pub fn clear(&mut self) {
        self.native.clear();
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
        mut scales: Vec<String>,
        resolved: bool,
        strategy: Option<&str>,
        policy: Option<&VerificationPolicy>,
    ) -> RusticolResult<Evaluation> {
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
        let inputs = LoopInputs {
            momenta,
            point_ids,
            scales_squared: scales,
            model_values: vec![],
        };
        let (result, verification) = match policy {
            Some(policy) => {
                let output = self
                    .native
                    .evaluate_verified(inputs, strategy, resolved, policy)?;
                (output.result, output.diagnostics)
            }
            None => (
                self.native.evaluate_partial(inputs, strategy, resolved)?,
                vec![],
            ),
        };
        Evaluation::from_native(result, verification, self.shape, resolved)
    }
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
    fn from_native(
        result: CompleteLoopResult<Arc<Batch>>,
        verification: Vec<VerificationDiagnostic>,
        shape: Shape,
        resolved: bool,
    ) -> RusticolResult<Self> {
        let verified: BTreeMap<_, _> = verification.iter().map(|v| (v.point_id, v)).collect();
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
        let successes: std::collections::BTreeSet<_> =
            successful_point_ids.iter().copied().collect();
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
        let diagnostics_json = serde_json::to_string(&serde_json::json!({
            "requested_point_ids":result.requested_point_ids,
            "successful_point_ids":successful_point_ids,
            "failures":result.failures, "verification":verification,
            "reduction_statuses":result.statuses,
            "source_revision":result.source_revision, "r2_strategy":result.r2_strategy,
        }))
        .map_err(serialization)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use symbolica::domains::rational::Rational;
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
