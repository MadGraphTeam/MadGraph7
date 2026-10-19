// SPDX-License-Identifier: 0BSD
//! Experimental complete native execution of validated legacy loop PACBIN plans.
use crate::virtual_runtime::{StabilityInput, input_policy, policy_input};
use crate::{batch::NativeBatch, error};
use pyo3::prelude::*;
use rusticol_core::loop_execution::{
    CompleteLoopResult, LoopExecutor, LoopInputs, LoopLaneFailure, LoopObservable,
};
use rusticol_core::loop_runtime::{LoadedLoopRuntime, load_loop_runtime_pacbin};
use rusticol_core::source_bindings::{SourceOverrides, SourceOwner, SourceSnapshot};
use rusticol_loop_reduction::runtime::{Batch, physical::PhysicalBackend};
use rusticol_loop_reduction::verification::{
    ArithmeticSpec, SourceBindingValidation, VerificationPolicy,
};
use std::collections::BTreeMap;
use std::sync::Arc;

pub(crate) type SpinVectors = BTreeMap<usize, Vec<[[String; 2]; 4]>>;

pub(crate) fn spin_inputs(
    vectors: Option<SpinVectors>,
    points: usize,
) -> PyResult<BTreeMap<usize, Vec<[String; 2]>>> {
    vectors
        .unwrap_or_default()
        .into_iter()
        .map(|(leg, rows)| {
            if rows.len() != 1 && rows.len() != points {
                return Err(error(
                    "spin vectors require one row or one row per physical point",
                ));
            }
            let values = if rows.len() == 1 {
                (0..points).flat_map(|_| rows[0].iter().cloned()).collect()
            } else {
                rows.into_iter().flatten().collect()
            };
            Ok((leg, values))
        })
        .collect()
}

#[pyclass]
struct NativeLoopProgram {
    executor: LoopExecutor<PhysicalBackend>,
    bits: u32,
    arithmetic: String,
    precision: Option<u32>,
    verification: rusticol_loop_reduction::verification::VerificationWorkspace,
    policy: Option<VerificationPolicy>,
    binding_validation: SourceBindingValidation,
}
impl NativeLoopProgram {
    fn bind(
        loaded: Arc<LoadedLoopRuntime>,
        arithmetic: &str,
        precision: Option<u32>,
        coefficient_engine: &str,
        reduction_engine: &str,
        parameters: Option<&SourceOverrides>,
        policy: Option<&VerificationPolicy>,
    ) -> PyResult<Self> {
        let backend =
            PhysicalBackend::new(arithmetic, precision, coefficient_engine, reduction_engine)
                .map_err(error)?;
        let bits = backend.precision_bits();
        let binding_validation = SourceBindingValidation::new(
            &ArithmeticSpec {
                arithmetic: arithmetic.into(),
                precision,
            },
            policy,
        )
        .map_err(error)?;
        let executor = if matches!(
            loaded.plan().model_binding,
            rusticol_core::loop_runtime::ModelBinding::Source { .. }
        ) {
            let owner = binding_validation
                .owner(loaded, &backend, parameters)
                .map_err(error)?;
            LoopExecutor::with_source_owner(owner, backend).map_err(error)?
        } else {
            if policy.is_some() {
                return Err(error(
                    "configured stability requires source-owned exact parameters",
                ));
            }
            LoopExecutor::with_parameters(loaded, backend, parameters).map_err(error)?
        };
        Ok(Self {
            executor,
            bits,
            arithmetic: arithmetic.into(),
            precision,
            verification: Default::default(),
            policy: policy.cloned(),
            binding_validation,
        })
    }
    fn verified_observable(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        higher: Vec<(String, Option<u32>)>,
        relative_tolerance: String,
        absolute_tolerance: String,
        point_ids: Option<Vec<u64>>,
        observable: LoopObservable,
        verify_successful: bool,
        spin_vectors: Option<SpinVectors>,
    ) -> PyResult<crate::virtual_runtime::NativeVerifiedLoopResult> {
        if momenta
            .iter()
            .any(|p| p.len() != self.executor.loaded().plan().state_legs.len())
        {
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
        let policy = crate::virtual_runtime::verification_policy(
            higher,
            relative_tolerance,
            absolute_tolerance,
            verify_successful,
        );
        let spec = rusticol_loop_reduction::verification::ArithmeticSpec {
            arithmetic: self.arithmetic.clone(),
            precision: self.precision,
        };
        let value = py
            .detach(|| {
                rusticol_loop_reduction::verification::evaluate_observable_verified(
                    &mut self.executor,
                    &mut self.verification,
                    inputs,
                    &spec,
                    &policy,
                    observable,
                )
            })
            .map_err(error)?;
        crate::virtual_runtime::NativeVerifiedLoopResult::bind(py, value)
    }
    fn evaluate_observable(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        model_values: Option<Vec<String>>,
        point_ids: Option<Vec<u64>>,
        snapshot: Option<&NativeSourceSnapshot>,
        observable: LoopObservable,
        partial: bool,
        spin_vectors: Option<SpinVectors>,
    ) -> PyResult<NativeCompleteLoopResult> {
        if self.policy.is_some() {
            return Err(error(
                "configured stability requires verified evaluation to preserve rescue diagnostics",
            ));
        }
        if self.executor.source_owner().is_some() && model_values.is_some() {
            return Err(error(
                "source model parameters require named owner updates, not positional model_values",
            ));
        }
        let snapshot = snapshot.map(|value| value.0.clone());
        let legs = self.executor.loaded().plan().state_legs.len();
        if momenta.iter().any(|point| point.len() != legs) {
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
            model_values: model_values.unwrap_or_default(),
            point_ids: ids,
        };
        py.detach(|| {
            if partial {
                self.executor
                    .evaluate_observable_partial_with_snapshot(input, snapshot, observable)
            } else {
                self.executor
                    .evaluate_observable_with_snapshot(input, snapshot, observable)
            }
        })
        .map(NativeCompleteLoopResult)
        .map_err(error)
    }
}
#[pymethods]
impl NativeLoopProgram {
    #[new]
    #[pyo3(signature=(path, arithmetic="double", precision=None, coefficient_engine="interpreted", reduction_engine="generic", parameters=None, stability=None))]
    fn new(
        py: Python<'_>,
        path: &str,
        arithmetic: &str,
        precision: Option<u32>,
        coefficient_engine: &str,
        reduction_engine: &str,
        parameters: Option<SourceOverrides>,
        stability: Option<StabilityInput>,
    ) -> PyResult<Self> {
        let policy = stability.map(input_policy);
        py.detach(|| {
            Self::bind(
                Arc::new(load_loop_runtime_pacbin(path).map_err(error)?),
                arithmetic,
                precision,
                coefficient_engine,
                reduction_engine,
                parameters.as_ref(),
                policy.as_ref(),
            )
        })
    }
    #[pyo3(signature=(arithmetic="double", precision=None, coefficient_engine="interpreted", reduction_engine="generic", parameters=None, stability=None))]
    fn rebind(
        &self,
        py: Python<'_>,
        arithmetic: &str,
        precision: Option<u32>,
        coefficient_engine: &str,
        reduction_engine: &str,
        parameters: Option<SourceOverrides>,
        stability: Option<StabilityInput>,
    ) -> PyResult<Self> {
        let replacement = stability.map(input_policy);
        let policy = replacement.as_ref().or(self.policy.as_ref());
        let captured = self
            .executor
            .source_owner()
            .map(|owner| owner.snapshot())
            .transpose()
            .map_err(error)?;
        let mut overrides = captured
            .as_ref()
            .map(|snapshot| snapshot.overrides().clone());
        if let Some(changes) = parameters {
            overrides
                .get_or_insert_with(SourceOverrides::new)
                .extend(changes);
        }
        py.detach(|| {
            Self::bind(
                self.executor.loaded().clone(),
                arithmetic,
                precision,
                coefficient_engine,
                reduction_engine,
                overrides.as_ref(),
                policy,
            )
        })
    }
    #[getter]
    fn precision_bits(&self) -> u32 {
        self.bits
    }
    #[getter]
    fn arithmetic(&self) -> &str {
        &self.arithmetic
    }
    #[getter]
    fn r2_strategy(&self) -> &str {
        &self.executor.loaded().plan().r2_strategy
    }
    fn metadata_json(&self) -> PyResult<String> {
        self.executor.loaded().metadata_json().map_err(error)
    }
    #[getter]
    fn stability(&self) -> Option<StabilityInput> {
        self.policy.as_ref().map(policy_input)
    }
    #[getter]
    fn source_state(&self) -> Option<NativeSourceState> {
        self.executor.source_owner().map(|owner| NativeSourceState {
            owner: owner.clone(),
            backend: self.executor.backend().clone(),
            validation: self.binding_validation.clone(),
        })
    }
    #[pyo3(signature=(momenta, scales_squared, higher, relative_tolerance, absolute_tolerance, *, point_ids=None, resolved=false, verify_successful=false))]
    fn evaluate_verified(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        higher: Vec<(String, Option<u32>)>,
        relative_tolerance: String,
        absolute_tolerance: String,
        point_ids: Option<Vec<u64>>,
        resolved: bool,
        verify_successful: bool,
    ) -> PyResult<crate::virtual_runtime::NativeVerifiedLoopResult> {
        self.verified_observable(
            py,
            momenta,
            scales_squared,
            higher,
            relative_tolerance,
            absolute_tolerance,
            point_ids,
            LoopObservable::from_resolved(resolved),
            verify_successful,
            None,
        )
    }

    #[pyo3(signature=(momenta, scales_squared, model_values=None, point_ids=None, snapshot=None, *, resolved=false, partial=false))]
    fn evaluate(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        model_values: Option<Vec<String>>,
        point_ids: Option<Vec<u64>>,
        snapshot: Option<&NativeSourceSnapshot>,
        resolved: bool,
        partial: bool,
    ) -> PyResult<NativeCompleteLoopResult> {
        self.evaluate_observable(
            py,
            momenta,
            scales_squared,
            model_values,
            point_ids,
            snapshot,
            LoopObservable::from_resolved(resolved),
            partial,
            None,
        )
    }
    #[pyo3(signature=(momenta, scales_squared, model_values=None, point_ids=None, snapshot=None, *, partial=false, spin_vectors=None))]
    fn evaluate_directed(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        model_values: Option<Vec<String>>,
        point_ids: Option<Vec<u64>>,
        snapshot: Option<&NativeSourceSnapshot>,
        partial: bool,
        spin_vectors: Option<SpinVectors>,
    ) -> PyResult<NativeCompleteLoopResult> {
        self.evaluate_observable(
            py,
            momenta,
            scales_squared,
            model_values,
            point_ids,
            snapshot,
            LoopObservable::Directed,
            partial,
            spin_vectors,
        )
    }
    #[pyo3(signature=(momenta, scales_squared, higher, relative_tolerance, absolute_tolerance, *, point_ids=None, verify_successful=false, spin_vectors=None))]
    fn evaluate_directed_verified(
        &mut self,
        py: Python<'_>,
        momenta: Vec<Vec<[String; 4]>>,
        scales_squared: Vec<String>,
        higher: Vec<(String, Option<u32>)>,
        relative_tolerance: String,
        absolute_tolerance: String,
        point_ids: Option<Vec<u64>>,
        verify_successful: bool,
        spin_vectors: Option<SpinVectors>,
    ) -> PyResult<crate::virtual_runtime::NativeVerifiedLoopResult> {
        self.verified_observable(
            py,
            momenta,
            scales_squared,
            higher,
            relative_tolerance,
            absolute_tolerance,
            point_ids,
            LoopObservable::Directed,
            verify_successful,
            spin_vectors,
        )
    }
}

#[pyclass]
pub(super) struct NativeCompleteLoopResult(pub(super) CompleteLoopResult<Arc<Batch>>);
#[pyclass(frozen, get_all)]
pub(super) struct NativeLoopFailure {
    point_id: u64,
    helicity: Option<usize>,
    order_index: Option<usize>,
    family_index: Option<usize>,
    external_leg: Option<usize>,
    color_component_index: Option<usize>,
    history_index: Option<usize>,
    direction: Option<usize>,
    stage: String,
    detail: String,
}
impl From<&LoopLaneFailure> for NativeLoopFailure {
    fn from(value: &LoopLaneFailure) -> Self {
        Self {
            point_id: value.point_id,
            helicity: value.helicity,
            order_index: value.order_index,
            family_index: value.family_index,
            external_leg: value.external_leg,
            color_component_index: value.color_component_index,
            history_index: value.history_index,
            direction: value.direction,
            stage: value.stage.clone(),
            detail: value.detail.clone(),
        }
    }
}
#[pymethods]
impl NativeCompleteLoopResult {
    #[getter]
    fn primary_color_contraction_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.0.primary_color_contraction).map_err(error)
    }
    #[getter]
    fn observable_kind(&self) -> &'static str {
        if self.0.directed.is_some() {
            "directed-complex"
        } else {
            "ordinary-interference"
        }
    }
    #[getter]
    fn spin_legs(&self) -> Vec<usize> {
        self.0
            .directed
            .as_ref()
            .map(|x| x.spin_legs.clone())
            .unwrap_or_default()
    }
    #[getter]
    fn history_ids(&self) -> Vec<String> {
        self.0
            .directed
            .as_ref()
            .map(|x| x.history_ids.clone())
            .unwrap_or_default()
    }
    #[getter]
    fn directed_first(&self) -> Option<NativeBatch> {
        self.0
            .directed
            .as_ref()
            .map(|x| NativeBatch(x.first.as_ref().clone()))
    }
    #[getter]
    fn directed_second(&self) -> Option<NativeBatch> {
        self.0
            .directed
            .as_ref()
            .map(|x| NativeBatch(x.second.as_ref().clone()))
    }
    #[getter]
    fn requested_point_ids(&self) -> Vec<u64> {
        self.0.requested_point_ids.clone()
    }
    #[getter]
    fn failures(&self) -> Vec<NativeLoopFailure> {
        self.0
            .failures
            .iter()
            .map(NativeLoopFailure::from)
            .collect()
    }
    #[getter]
    /// Successful original physical point IDs. Numerical batch rows are dense;
    /// resolved helicity/colour axes are not encoded in these opaque IDs.
    fn point_ids(&self) -> Vec<u64> {
        self.0.point_ids.clone()
    }
    #[getter]
    fn r2_strategy(&self) -> &str {
        &self.0.r2_strategy
    }
    #[getter]
    fn source_revision(&self) -> Option<u64> {
        self.0.source_revision
    }
    #[getter]
    fn born(&self) -> NativeBatch {
        NativeBatch(self.0.born.as_ref().clone())
    }
    #[getter]
    fn component_batches(&self) -> Vec<NativeBatch> {
        self.0
            .components
            .iter()
            .map(|b| NativeBatch(b.as_ref().clone()))
            .collect()
    }
    #[getter]
    fn totals(&self) -> NativeBatch {
        NativeBatch(self.0.totals.as_ref().clone())
    }
    #[getter]
    fn statuses(&self) -> Vec<Vec<(u64, String)>> {
        self.0.statuses.clone()
    }
}
/// Separate owner handle: updates do not borrow mutable process workspaces.
#[pyclass]
pub(super) struct NativeSourceState {
    pub(super) owner: Arc<SourceOwner<PhysicalBackend>>,
    pub(super) backend: PhysicalBackend,
    pub(super) validation: SourceBindingValidation,
}
#[pymethods]
impl NativeSourceState {
    fn snapshot(&self) -> PyResult<NativeSourceSnapshot> {
        self.owner
            .snapshot()
            .map(NativeSourceSnapshot)
            .map_err(error)
    }
    fn update(
        &self,
        py: Python<'_>,
        parameters: SourceOverrides,
    ) -> PyResult<NativeSourceSnapshot> {
        py.detach(|| {
            self.validation
                .update(&self.owner, &self.backend, &parameters)
        })
        .map(NativeSourceSnapshot)
        .map_err(error)
    }
}
#[pyclass(frozen)]
struct NativeSourceSnapshot(Arc<SourceSnapshot<Arc<Batch>>>);
#[pymethods]
impl NativeSourceSnapshot {
    #[getter]
    fn revision(&self) -> u64 {
        self.0.revision()
    }
    #[getter]
    fn overrides(&self) -> SourceOverrides {
        self.0.overrides().clone()
    }
    #[getter]
    fn binding_failure_stage(&self) -> Option<String> {
        self.0
            .binding_failure()
            .map(|failure| failure.stage.clone())
    }
    #[getter]
    fn arguments(&self) -> PyResult<NativeBatch> {
        self.0
            .arguments()
            .map(|values| NativeBatch(values.as_ref().clone()))
            .map_err(error)
    }
    #[getter]
    fn values(&self) -> PyResult<NativeBatch> {
        self.0
            .values()
            .map(|values| NativeBatch(values.as_ref().clone()))
            .map_err(error)
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<NativeLoopProgram>()?;
    module.add_class::<NativeCompleteLoopResult>()?;
    module.add_class::<NativeLoopFailure>()?;
    module.add_class::<NativeSourceState>()?;
    module.add_class::<NativeSourceSnapshot>()
}
