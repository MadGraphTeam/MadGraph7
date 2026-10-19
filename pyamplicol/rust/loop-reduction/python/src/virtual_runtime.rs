// SPDX-License-Identifier: 0BSD
//! Python ownership/conversion only; complete execution is one native call.
use crate::{
    error,
    physical::{NativeCompleteLoopResult, NativeSourceState, SpinVectors, spin_inputs},
};
use pyo3::{create_exception, exceptions::PyValueError, prelude::*};
use rusticol_core::{
    RusticolError, RusticolErrorKind,
    loop_execution::{LoopInputs, LoopObservable},
    source_bindings::SourceOverrides,
};
use rusticol_loop_reduction::verification::{
    ArithmeticSpec, VerificationPolicy, VerifiedLoopResult,
};
use rusticol_loop_reduction::virtual_runtime::VirtualRuntime;

#[pyclass(frozen)]
pub(super) struct NativeVerifiedLoopResult {
    #[pyo3(get)]
    result: Py<NativeCompleteLoopResult>,
    #[pyo3(get)]
    diagnostics_json: String,
}
impl NativeVerifiedLoopResult {
    pub(super) fn bind(py: Python<'_>, value: VerifiedLoopResult) -> PyResult<Self> {
        Ok(Self {
            diagnostics_json: value.diagnostics_json().map_err(error)?,
            result: Py::new(py, NativeCompleteLoopResult(value.result))?,
        })
    }
}
pub(super) fn verification_policy(
    higher: Vec<(String, Option<u32>)>,
    relative: String,
    absolute: String,
    verify_successful: bool,
) -> VerificationPolicy {
    VerificationPolicy {
        higher: higher
            .into_iter()
            .map(|(arithmetic, precision)| ArithmeticSpec {
                arithmetic,
                precision,
            })
            .collect(),
        relative_tolerance: relative,
        absolute_tolerance: absolute,
        verify_successful,
    }
}

/// The existing verification policy transported without a second policy schema.
pub(super) type StabilityInput = (Vec<(String, Option<u32>)>, String, String, bool);
pub(super) fn input_policy(input: StabilityInput) -> VerificationPolicy {
    verification_policy(input.0, input.1, input.2, input.3)
}
pub(super) fn policy_input(policy: &VerificationPolicy) -> StabilityInput {
    (
        policy
            .higher
            .iter()
            .map(|x| (x.arithmetic.clone(), x.precision))
            .collect(),
        policy.relative_tolerance.clone(),
        policy.absolute_tolerance.clone(),
        policy.verify_successful,
    )
}

create_exception!(_loop_reduction, VirtualArtifactError, PyValueError);
create_exception!(_loop_reduction, VirtualCompatibilityError, PyValueError);
create_exception!(_loop_reduction, VirtualEvaluationError, PyValueError);

fn native_error(error: RusticolError) -> PyErr {
    match error.kind() {
        RusticolErrorKind::Artifact
        | RusticolErrorKind::Security
        | RusticolErrorKind::Integrity
        | RusticolErrorKind::Serialization => VirtualArtifactError::new_err(error.to_string()),
        RusticolErrorKind::Compatibility
        | RusticolErrorKind::UnsupportedPrecision
        | RusticolErrorKind::UnsupportedRuntimeCapability => {
            VirtualCompatibilityError::new_err(error.to_string())
        }
        RusticolErrorKind::InvalidArgument | RusticolErrorKind::ModelParameter => {
            PyValueError::new_err(error.to_string())
        }
        _ => VirtualEvaluationError::new_err(error.to_string()),
    }
}

#[pyclass]
struct NativeVirtualRuntime(VirtualRuntime);

#[pymethods]
impl NativeVirtualRuntime {
    #[new]
    #[pyo3(signature=(path, *, process=None, arithmetic="double", precision=None, parameters=None, r2_strategy="auto", stability=None))]
    fn new(
        py: Python<'_>,
        path: &str,
        process: Option<&str>,
        arithmetic: &str,
        precision: Option<u32>,
        parameters: Option<SourceOverrides>,
        r2_strategy: &str,
        stability: Option<StabilityInput>,
    ) -> PyResult<Self> {
        let policy = stability.map(input_policy);
        py.detach(|| {
            VirtualRuntime::open_with_policy(
                path,
                process,
                arithmetic,
                precision,
                parameters.as_ref(),
                r2_strategy,
                policy.as_ref(),
            )
        })
        .map(Self)
        .map_err(native_error)
    }
    #[getter]
    fn contribution_kind(&self) -> &'static str {
        "virtual"
    }
    #[getter]
    fn execution_mode(&self) -> &'static str {
        "loop_recurrence"
    }
    #[getter]
    fn process_id(&self) -> &str {
        self.0.process_id()
    }
    #[getter]
    fn arithmetic(&self) -> &str {
        self.0.arithmetic()
    }
    #[getter]
    fn precision(&self) -> Option<u32> {
        self.0.precision()
    }
    #[getter]
    fn precision_bits(&self) -> u32 {
        self.0.precision_bits()
    }
    #[getter]
    fn requested_r2_strategy(&self) -> &str {
        self.0.requested_r2_strategy()
    }
    #[getter]
    fn r2_strategy(&self) -> &'static str {
        "model"
    }
    #[getter]
    fn source_revision(&self) -> PyResult<u64> {
        self.0.source_revision().map_err(native_error)
    }
    #[getter]
    fn stability(&self) -> Option<StabilityInput> {
        self.0.verification_policy().map(policy_input)
    }
    #[getter]
    fn source_state(&self) -> NativeSourceState {
        NativeSourceState {
            owner: self.0.source_owner().clone(),
            backend: self.0.backend().clone(),
            validation: self.0.source_binding_validation().clone(),
        }
    }
    #[getter]
    fn parameter_overrides(&self) -> PyResult<SourceOverrides> {
        self.0.parameter_overrides().map_err(native_error)
    }
    fn metadata_json(&self) -> PyResult<String> {
        self.0.metadata_json().map_err(native_error)
    }
    #[getter]
    fn color_contraction_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.0.color_contraction_metadata()).map_err(crate::error)
    }
    fn plan_metadata_json(&self) -> PyResult<String> {
        self.0.plan_metadata_json().map_err(native_error)
    }
    fn set_model_parameters(&self, py: Python<'_>, parameters: SourceOverrides) -> PyResult<u64> {
        py.detach(|| self.0.update_parameters(&parameters))
            .map_err(native_error)
    }
    fn clear(&mut self, py: Python<'_>) {
        py.detach(|| self.0.clear());
    }

    #[pyo3(signature=(*, arithmetic="double", precision=None, parameters=None, r2_strategy=None, stability=None))]
    fn rebind(
        &self,
        py: Python<'_>,
        arithmetic: &str,
        precision: Option<u32>,
        parameters: Option<SourceOverrides>,
        r2_strategy: Option<&str>,
        stability: Option<StabilityInput>,
    ) -> PyResult<Self> {
        let policy = stability.map(input_policy);
        py.detach(|| {
            self.0.rebind_with_policy(
                arithmetic,
                precision,
                parameters.as_ref(),
                r2_strategy,
                policy.as_ref(),
            )
        })
        .map(Self)
        .map_err(native_error)
    }

    #[pyo3(signature=(momenta, scales_squared, higher, relative_tolerance, absolute_tolerance, *, point_ids=None, r2_strategy=None, resolved=false, verify_successful=false))]
    fn evaluate_verified(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        higher: Vec<(String, Option<u32>)>,
        relative_tolerance: String,
        absolute_tolerance: String,
        point_ids: Option<Vec<u64>>,
        r2_strategy: Option<&str>,
        resolved: bool,
        verify_successful: bool,
    ) -> PyResult<NativeVerifiedLoopResult> {
        self.verified_mode(
            py,
            momenta,
            scales_squared,
            higher,
            relative_tolerance,
            absolute_tolerance,
            point_ids,
            r2_strategy,
            LoopObservable::from_resolved(resolved),
            verify_successful,
            None,
        )
    }

    #[pyo3(signature=(momenta, scales_squared, higher, relative_tolerance, absolute_tolerance, *, point_ids=None, r2_strategy=None, verify_successful=false, spin_vectors=None))]
    fn evaluate_directed_verified(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        higher: Vec<(String, Option<u32>)>,
        relative_tolerance: String,
        absolute_tolerance: String,
        point_ids: Option<Vec<u64>>,
        r2_strategy: Option<&str>,
        verify_successful: bool,
        spin_vectors: Option<SpinVectors>,
    ) -> PyResult<NativeVerifiedLoopResult> {
        self.verified_mode(
            py,
            momenta,
            scales_squared,
            higher,
            relative_tolerance,
            absolute_tolerance,
            point_ids,
            r2_strategy,
            LoopObservable::Directed,
            verify_successful,
            spin_vectors,
        )
    }

    #[pyo3(signature=(momenta, scales_squared, *, point_ids=None, r2_strategy=None))]
    fn evaluate(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        point_ids: Option<Vec<u64>>,
        r2_strategy: Option<&str>,
    ) -> PyResult<NativeCompleteLoopResult> {
        self.evaluate_mode(
            py,
            momenta,
            scales_squared,
            point_ids,
            r2_strategy,
            LoopObservable::Inclusive,
            false,
            None,
        )
    }

    #[pyo3(signature=(momenta, scales_squared, *, point_ids=None, r2_strategy=None))]
    fn evaluate_resolved(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        point_ids: Option<Vec<u64>>,
        r2_strategy: Option<&str>,
    ) -> PyResult<NativeCompleteLoopResult> {
        self.evaluate_mode(
            py,
            momenta,
            scales_squared,
            point_ids,
            r2_strategy,
            LoopObservable::Resolved,
            false,
            None,
        )
    }

    #[pyo3(signature=(momenta, scales_squared, *, point_ids=None, r2_strategy=None, resolved=false))]
    fn evaluate_partial(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        point_ids: Option<Vec<u64>>,
        r2_strategy: Option<&str>,
        resolved: bool,
    ) -> PyResult<NativeCompleteLoopResult> {
        self.evaluate_mode(
            py,
            momenta,
            scales_squared,
            point_ids,
            r2_strategy,
            LoopObservable::from_resolved(resolved),
            true,
            None,
        )
    }
    #[pyo3(signature=(momenta, scales_squared, *, point_ids=None, r2_strategy=None, partial=false, spin_vectors=None))]
    fn evaluate_directed(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        point_ids: Option<Vec<u64>>,
        r2_strategy: Option<&str>,
        partial: bool,
        spin_vectors: Option<SpinVectors>,
    ) -> PyResult<NativeCompleteLoopResult> {
        self.evaluate_mode(
            py,
            momenta,
            scales_squared,
            point_ids,
            r2_strategy,
            LoopObservable::Directed,
            partial,
            spin_vectors,
        )
    }
}

impl NativeVirtualRuntime {
    fn verified_mode(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        higher: Vec<(String, Option<u32>)>,
        relative_tolerance: String,
        absolute_tolerance: String,
        point_ids: Option<Vec<u64>>,
        r2_strategy: Option<&str>,
        observable: LoopObservable,
        verify_successful: bool,
        spin_vectors: Option<SpinVectors>,
    ) -> PyResult<NativeVerifiedLoopResult> {
        if momenta.iter().any(|p| p.len() != self.0.external_count()) {
            return Err(error("physical momentum leg shape differs"));
        }
        let ids = point_ids.unwrap_or_else(|| (0..momenta.len() as u64).collect());
        if ids.len() != momenta.len() {
            return Err(error("physical point identity count differs"));
        }
        let inputs = LoopInputs {
            spin_sources: spin_inputs(spin_vectors, momenta.len())?,
            momenta: momenta.into_iter().flatten().flatten().collect(),
            scales_squared,
            point_ids: ids,
            model_values: vec![],
        };
        let policy = verification_policy(
            higher,
            relative_tolerance,
            absolute_tolerance,
            verify_successful,
        );
        let value = py
            .detach(|| {
                if observable == LoopObservable::Directed {
                    self.0
                        .evaluate_directed_verified(inputs, r2_strategy, &policy)
                } else {
                    self.0.evaluate_verified(
                        inputs,
                        r2_strategy,
                        observable == LoopObservable::Resolved,
                        &policy,
                    )
                }
            })
            .map_err(native_error)?;
        NativeVerifiedLoopResult::bind(py, value)
    }

    fn evaluate_mode(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        point_ids: Option<Vec<u64>>,
        r2_strategy: Option<&str>,
        observable: LoopObservable,
        partial: bool,
        spin_vectors: Option<SpinVectors>,
    ) -> PyResult<NativeCompleteLoopResult> {
        if momenta
            .iter()
            .any(|point| point.len() != self.0.external_count())
        {
            return Err(error("physical momentum leg shape differs"));
        }
        let ids = point_ids.unwrap_or_else(|| (0..momenta.len() as u64).collect());
        if ids.len() != momenta.len() {
            return Err(error("physical point identity count differs"));
        }
        let input = LoopInputs {
            spin_sources: spin_inputs(spin_vectors, momenta.len())?,
            momenta: momenta.into_iter().flatten().flatten().collect(),
            scales_squared,
            point_ids: ids,
            model_values: Vec::new(),
        };
        py.detach(|| {
            if observable == LoopObservable::Directed {
                self.0.evaluate_directed(input, r2_strategy, partial)
            } else if partial {
                self.0
                    .evaluate_partial(input, r2_strategy, observable == LoopObservable::Resolved)
            } else if observable == LoopObservable::Resolved {
                self.0.evaluate_resolved(input, r2_strategy)
            } else {
                self.0.evaluate(input, r2_strategy)
            }
        })
        .map(NativeCompleteLoopResult)
        .map_err(native_error)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeVirtualRuntime>()?;
    module.add_class::<NativeVerifiedLoopResult>()?;
    module.add(
        "VirtualArtifactError",
        module.py().get_type::<VirtualArtifactError>(),
    )?;
    module.add(
        "VirtualCompatibilityError",
        module.py().get_type::<VirtualCompatibilityError>(),
    )?;
    module.add(
        "VirtualEvaluationError",
        module.py().get_type::<VirtualEvaluationError>(),
    )
}
