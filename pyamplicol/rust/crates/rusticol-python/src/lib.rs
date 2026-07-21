// SPDX-License-Identifier: 0BSD

#[cfg(feature = "numpy")]
mod eager_lowering;

#[cfg(feature = "numpy")]
use numpy::{PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyDict, PyList};
use rusticol_core::{
    ColorAccuracy, ColorComponent as CoreColorComponent, ModelParameter as CoreModelParameter,
    NativeResolvedEvaluation, NativeRuntime, NativeRuntimeProfile, ParameterKind, ParticleRole,
    ProcessPhysics as CoreProcessPhysics, ReductionKind, RusticolError as CoreError,
    RusticolErrorKind, preflight_prepared_kernel_pack, runtime_target_info,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

create_exception!(_rusticol, RusticolError, PyException);
create_exception!(_rusticol, ArtifactError, RusticolError);
create_exception!(_rusticol, CompatibilityError, RusticolError);
create_exception!(_rusticol, EvaluationError, RusticolError);
create_exception!(_rusticol, SelectorError, EvaluationError);
create_exception!(_rusticol, ModelParameterError, EvaluationError);

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct TargetInfo {
    triple: String,
    cpu_features: Vec<String>,
}

fn python_error(error: CoreError) -> PyErr {
    let message = error.to_string();
    match error.kind() {
        RusticolErrorKind::InvalidArgument => PyValueError::new_err(message),
        RusticolErrorKind::Artifact
        | RusticolErrorKind::Security
        | RusticolErrorKind::Integrity
        | RusticolErrorKind::Serialization => ArtifactError::new_err(message),
        RusticolErrorKind::Compatibility
        | RusticolErrorKind::UnsupportedPrecision
        | RusticolErrorKind::UnsupportedRuntimeCapability => CompatibilityError::new_err(message),
        RusticolErrorKind::Selector => SelectorError::new_err(message),
        RusticolErrorKind::ModelParameter => ModelParameterError::new_err(message),
        RusticolErrorKind::Evaluation | RusticolErrorKind::Internal => {
            EvaluationError::new_err(message)
        }
        _ => EvaluationError::new_err(message),
    }
}

fn require_raw_f64_precision(precision: u32) -> Result<(), CoreError> {
    if precision == 16 {
        return Ok(());
    }
    Err(CoreError::compatibility(format!(
        "raw pyamplicol._rusticol supports only precision=16 for Symbolica-free direct SymJIT evaluation; precision={precision} must be routed through the public Python Symbolica executor"
    )))
}

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct ExternalParticle {
    index: usize,
    label: usize,
    name: String,
    pdg_id: i32,
    state: String,
    momentum_slot: usize,
}

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct HelicityConfiguration {
    id: String,
    index: usize,
    values: Vec<i32>,
    computed: bool,
    structural_zero: bool,
    representative_id: String,
    coefficient: f64,
}

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct ColorFlow {
    id: String,
    index: usize,
    word: Vec<usize>,
    computed: bool,
    representative_id: String,
    coefficient: f64,
}

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct ContractedColorComponent {
    id: String,
    index: usize,
    description: String,
}

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct ModelParameter {
    name: String,
    kind: String,
    default_real: f64,
    default_imaginary: f64,
    mutable: bool,
}

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct ReductionGroup {
    id: String,
    representative_helicity_id: String,
    representative_color_id: String,
    physical_helicity_ids: Vec<String>,
    physical_color_ids: Vec<String>,
}

#[pyclass(module = "pyamplicol._rusticol", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
struct PhysicsReduction {
    kind: String,
    groups: Vec<ReductionGroup>,
}

impl From<&CoreModelParameter> for ModelParameter {
    fn from(parameter: &CoreModelParameter) -> Self {
        Self {
            name: parameter.name.clone(),
            kind: parameter_kind_name(parameter.kind).to_string(),
            default_real: parameter.default_real,
            default_imaginary: parameter.default_imaginary,
            mutable: parameter.mutable,
        }
    }
}

#[pyclass(module = "pyamplicol._rusticol", frozen, skip_from_py_object)]
#[derive(Clone)]
struct ProcessPhysics {
    process_id: String,
    process: String,
    color_accuracy: String,
    helicity_coverage: String,
    color_coverage: String,
    color_kind: String,
    structural_zero_helicity_count: usize,
    external_particles: Vec<ExternalParticle>,
    helicities: Vec<HelicityConfiguration>,
    color_flows: Vec<ColorFlow>,
    contracted_color_components: Vec<ContractedColorComponent>,
    reduction: PhysicsReduction,
    model_parameters: Vec<ModelParameter>,
    selector_capabilities: Vec<String>,
}

impl ProcessPhysics {
    fn from_core(physics: &CoreProcessPhysics) -> Self {
        let external_particles = physics
            .external_particles
            .iter()
            .map(|particle| ExternalParticle {
                index: particle.index,
                label: particle.label,
                name: particle.particle.clone(),
                pdg_id: particle.pdg,
                state: match particle.role {
                    ParticleRole::Initial => "incoming",
                    ParticleRole::Final => "outgoing",
                }
                .to_string(),
                momentum_slot: particle.momentum_slot,
            })
            .collect();
        let helicities = physics
            .helicities
            .iter()
            .map(|helicity| HelicityConfiguration {
                id: helicity.id.clone(),
                index: helicity.index,
                values: helicity.values.clone(),
                computed: helicity.computed,
                structural_zero: helicity.structural_zero,
                representative_id: helicity.representative_id.clone(),
                coefficient: helicity.coefficient,
            })
            .collect();
        let color_flows = physics
            .color_components
            .iter()
            .filter_map(|component| match component {
                CoreColorComponent::LcFlow(flow) => Some(ColorFlow {
                    id: flow.id.clone(),
                    index: flow.index,
                    word: flow.word.clone(),
                    computed: flow.computed,
                    representative_id: flow.representative_id.clone(),
                    coefficient: flow.coefficient,
                }),
                CoreColorComponent::ContractedColor(_) => None,
            })
            .collect();
        let contracted_color_components = physics
            .color_components
            .iter()
            .filter_map(|component| match component {
                CoreColorComponent::ContractedColor(color) => Some(ContractedColorComponent {
                    id: color.id.clone(),
                    index: color.index,
                    description: color.description.clone(),
                }),
                CoreColorComponent::LcFlow(_) => None,
            })
            .collect();
        let mut selector_capabilities = Vec::new();
        if physics.selectors.helicity {
            selector_capabilities.push("helicity".to_string());
        }
        if physics.selectors.color_flow {
            selector_capabilities.push("color_flow".to_string());
        }
        if physics.selectors.contracted_color {
            selector_capabilities.push("contracted_color".to_string());
        }
        let reduction = PhysicsReduction {
            kind: reduction_kind_name(physics.reduction.kind).to_string(),
            groups: physics
                .reduction
                .groups
                .iter()
                .map(|group| ReductionGroup {
                    id: group.id.clone(),
                    representative_helicity_id: group.representative_helicity_id.clone(),
                    representative_color_id: group.representative_color_id.clone(),
                    physical_helicity_ids: group.physical_helicity_ids.clone(),
                    physical_color_ids: group.physical_color_ids.clone(),
                })
                .collect(),
        };
        Self {
            process_id: physics.process_id.clone(),
            process: physics.process.clone(),
            color_accuracy: color_accuracy_name(physics.color_accuracy).to_string(),
            helicity_coverage: physics.coverage.helicities.clone(),
            color_coverage: physics.coverage.color.clone(),
            color_kind: physics.coverage.color_kind.clone(),
            structural_zero_helicity_count: physics.coverage.structural_zero_helicity_count,
            external_particles,
            helicities,
            color_flows,
            contracted_color_components,
            reduction,
            model_parameters: physics.model_parameters.iter().map(Into::into).collect(),
            selector_capabilities,
        }
    }
}

#[pymethods]
impl ProcessPhysics {
    #[getter]
    fn process_id(&self) -> &str {
        &self.process_id
    }

    #[getter]
    fn process(&self) -> &str {
        &self.process
    }

    #[getter]
    fn color_accuracy(&self) -> &str {
        &self.color_accuracy
    }

    #[getter]
    fn helicity_coverage(&self) -> &str {
        &self.helicity_coverage
    }

    #[getter]
    fn color_coverage(&self) -> &str {
        &self.color_coverage
    }

    #[getter]
    fn color_kind(&self) -> &str {
        &self.color_kind
    }

    #[getter]
    fn structural_zero_helicity_count(&self) -> usize {
        self.structural_zero_helicity_count
    }

    #[getter]
    fn external_particles(&self) -> Vec<ExternalParticle> {
        self.external_particles.clone()
    }

    #[getter]
    fn helicities(&self) -> Vec<HelicityConfiguration> {
        self.helicities.clone()
    }

    #[getter]
    fn color_flows(&self) -> Vec<ColorFlow> {
        self.color_flows.clone()
    }

    #[getter]
    fn contracted_color_components(&self) -> Vec<ContractedColorComponent> {
        self.contracted_color_components.clone()
    }

    #[getter]
    fn reduction(&self) -> PhysicsReduction {
        self.reduction.clone()
    }

    #[getter]
    fn model_parameters(&self) -> Vec<ModelParameter> {
        self.model_parameters.clone()
    }

    #[getter]
    fn selector_capabilities(&self) -> Vec<String> {
        self.selector_capabilities.clone()
    }
}

#[pyclass(module = "pyamplicol._rusticol", unsendable)]
struct ResolvedEvaluation {
    values: Py<PyAny>,
    totals: Py<PyAny>,
    shape: (usize, usize, usize),
    helicity_ids: Vec<String>,
    color_ids: Vec<String>,
    color_accuracy: String,
    precision: u32,
}

#[pymethods]
impl ResolvedEvaluation {
    #[getter]
    fn values(&self, py: Python<'_>) -> Py<PyAny> {
        self.values.clone_ref(py)
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        self.shape
    }

    #[getter]
    fn helicity_ids(&self) -> Vec<String> {
        self.helicity_ids.clone()
    }

    #[getter]
    fn color_ids(&self) -> Vec<String> {
        self.color_ids.clone()
    }

    #[getter]
    fn color_flow_ids(&self) -> Vec<String> {
        if self.color_accuracy == "lc" {
            self.color_ids.clone()
        } else {
            Vec::new()
        }
    }

    #[getter]
    fn color_accuracy(&self) -> &str {
        &self.color_accuracy
    }

    #[getter]
    fn precision(&self) -> u32 {
        self.precision
    }

    fn total(&self, py: Python<'_>) -> Py<PyAny> {
        self.totals.clone_ref(py)
    }
}

#[pyclass(module = "pyamplicol._rusticol", unsendable)]
struct Runtime {
    runtime: NativeRuntime,
}

// These signatures intentionally mirror the keyword-rich public Python ABI.
#[allow(clippy::too_many_arguments)]
#[pymethods]
impl Runtime {
    #[new]
    #[pyo3(signature=(artifact, *, process=None, model_parameters=None, mute_warnings=false))]
    fn new(
        artifact: &Bound<'_, PyAny>,
        process: Option<&str>,
        model_parameters: Option<&Bound<'_, PyAny>>,
        mute_warnings: bool,
    ) -> PyResult<Self> {
        let path = path_from_python(artifact)?;
        let mut runtime = NativeRuntime::load(path, process, None).map_err(python_error)?;
        if let Some(mapping) = model_parameters {
            let values = parse_model_parameters(mapping)?;
            runtime
                .set_model_parameters(&values)
                .map_err(python_error)?;
        }
        if mute_warnings {
            runtime.mute_warnings();
        }
        Ok(Self { runtime })
    }

    #[staticmethod]
    #[pyo3(signature=(artifact, *, process=None, model_parameters=None, mute_warnings=false))]
    fn load(
        artifact: &Bound<'_, PyAny>,
        process: Option<&str>,
        model_parameters: Option<&Bound<'_, PyAny>>,
        mute_warnings: bool,
    ) -> PyResult<Self> {
        Self::new(artifact, process, model_parameters, mute_warnings)
    }

    #[getter]
    fn process(&self) -> String {
        self.runtime.metadata().process
    }

    #[getter]
    fn process_id(&self) -> String {
        self.runtime.metadata().process_key
    }

    #[getter]
    fn color_accuracy(&self) -> String {
        self.runtime.metadata().color_accuracy
    }

    #[getter]
    fn physics(&self) -> ProcessPhysics {
        ProcessPhysics::from_core(self.runtime.process_physics())
    }

    fn metadata_json(&self) -> PyResult<String> {
        self.runtime.metadata_json().map_err(python_error)
    }

    fn physics_json(&self) -> PyResult<String> {
        self.runtime.physics_json().map_err(python_error)
    }

    fn _exact_runtime_state_json(&self) -> PyResult<String> {
        self.runtime
            .exact_runtime_state_json()
            .map_err(python_error)
    }

    #[pyo3(signature=(momenta, *, helicities=None, color_flows=None, helicity_by_point=None, color_flow_by_point=None, precision=16))]
    fn evaluate(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        helicities: Option<Vec<String>>,
        color_flows: Option<Vec<String>>,
        helicity_by_point: Option<Vec<u32>>,
        color_flow_by_point: Option<Vec<u32>>,
        precision: u32,
    ) -> PyResult<Py<PyAny>> {
        require_raw_f64_precision(precision).map_err(python_error)?;
        let (values, point_count) = parse_f64_momenta(momenta, self.runtime.external_count())?;
        let values = self
            .runtime
            .evaluate_f64_with_selectors(
                &values,
                point_count,
                helicities.as_deref(),
                color_flows.as_deref(),
                helicity_by_point.as_deref(),
                color_flow_by_point.as_deref(),
            )
            .map_err(python_error)?;
        let result = PyList::new(py, values)?.into_any().unbind();
        self.emit_warnings(py)?;
        Ok(result)
    }

    #[pyo3(signature=(momenta, repetitions, *, helicities=None, color_flows=None, helicity_by_point=None, color_flow_by_point=None, precision=16))]
    fn _benchmark_f64_wall_time(
        &mut self,
        momenta: &Bound<'_, PyAny>,
        repetitions: usize,
        helicities: Option<Vec<String>>,
        color_flows: Option<Vec<String>>,
        helicity_by_point: Option<Vec<u32>>,
        color_flow_by_point: Option<Vec<u32>>,
        precision: u32,
    ) -> PyResult<f64> {
        require_raw_f64_precision(precision).map_err(python_error)?;
        let (values, point_count) = parse_f64_momenta(momenta, self.runtime.external_count())?;
        self.runtime
            .benchmark_f64_wall_time_with_selectors(
                &values,
                point_count,
                repetitions,
                helicities.as_deref(),
                color_flows.as_deref(),
                helicity_by_point.as_deref(),
                color_flow_by_point.as_deref(),
            )
            .map_err(python_error)
    }

    #[pyo3(signature=(momenta, *, helicities=None, color_flows=None, helicity_by_point=None, color_flow_by_point=None, precision=16, include_values=false))]
    fn profile(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        helicities: Option<Vec<String>>,
        color_flows: Option<Vec<String>>,
        helicity_by_point: Option<Vec<u32>>,
        color_flow_by_point: Option<Vec<u32>>,
        precision: u32,
        include_values: bool,
    ) -> PyResult<Py<PyAny>> {
        require_raw_f64_precision(precision).map_err(python_error)?;
        let (values, point_count) = parse_f64_momenta(momenta, self.runtime.external_count())?;
        let profiled = self
            .runtime
            .evaluate_f64_profile_with_selectors(
                &values,
                point_count,
                helicities.as_deref(),
                color_flows.as_deref(),
                helicity_by_point.as_deref(),
                color_flow_by_point.as_deref(),
            )
            .map_err(python_error)?;
        let result = runtime_profile_to_python(py, &profiled.profile, point_count)?;
        if include_values {
            result.set_item("values", PyList::new(py, profiled.values)?)?;
        }
        self.emit_warnings(py)?;
        Ok(result.into_any().unbind())
    }

    #[pyo3(signature=(momenta, repetitions, *, helicities=None, color_flows=None, helicity_by_point=None, color_flow_by_point=None, precision=16, include_values=false))]
    fn profile_repeated(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        repetitions: usize,
        helicities: Option<Vec<String>>,
        color_flows: Option<Vec<String>>,
        helicity_by_point: Option<Vec<u32>>,
        color_flow_by_point: Option<Vec<u32>>,
        precision: u32,
        include_values: bool,
    ) -> PyResult<Py<PyAny>> {
        require_raw_f64_precision(precision).map_err(python_error)?;
        let (values, point_count) = parse_f64_momenta(momenta, self.runtime.external_count())?;
        let profiled = self
            .runtime
            .evaluate_f64_profile_repeated_with_selectors(
                &values,
                point_count,
                repetitions,
                helicities.as_deref(),
                color_flows.as_deref(),
                helicity_by_point.as_deref(),
                color_flow_by_point.as_deref(),
            )
            .map_err(python_error)?;
        let measured_points = point_count
            .checked_mul(repetitions)
            .ok_or_else(|| PyValueError::new_err("profile point count overflowed"))?;
        let result = runtime_profile_to_python(py, &profiled.profile, measured_points)?;
        if include_values {
            result.set_item("values", PyList::new(py, profiled.values)?)?;
        }
        self.emit_warnings(py)?;
        Ok(result.into_any().unbind())
    }

    #[pyo3(signature=(momenta, *, helicities=None, color_flows=None, helicity_by_point=None, color_flow_by_point=None, precision=16, include_values=false))]
    fn evaluate_profile(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        helicities: Option<Vec<String>>,
        color_flows: Option<Vec<String>>,
        helicity_by_point: Option<Vec<u32>>,
        color_flow_by_point: Option<Vec<u32>>,
        precision: u32,
        include_values: bool,
    ) -> PyResult<Py<PyAny>> {
        self.profile(
            py,
            momenta,
            helicities,
            color_flows,
            helicity_by_point,
            color_flow_by_point,
            precision,
            include_values,
        )
    }

    #[pyo3(signature=(momenta, *, helicities=None, color_flows=None, precision=16))]
    fn evaluate_resolved(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        helicities: Option<Vec<String>>,
        color_flows: Option<Vec<String>>,
        precision: u32,
    ) -> PyResult<ResolvedEvaluation> {
        require_raw_f64_precision(precision).map_err(python_error)?;
        let accuracy = self.runtime.metadata().color_accuracy;
        let (values, point_count) = parse_f64_momenta(momenta, self.runtime.external_count())?;
        let resolved = self
            .runtime
            .evaluate_resolved_f64(
                &values,
                point_count,
                helicities.as_deref(),
                color_flows.as_deref(),
            )
            .map_err(python_error)?;
        let result = resolved_f64_to_python(py, resolved, accuracy, precision)?;
        self.emit_warnings(py)?;
        Ok(result)
    }

    fn evaluate_with_prec(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        decimal_digit_precision: u32,
    ) -> PyResult<Py<PyAny>> {
        self.evaluate(py, momenta, None, None, None, None, decimal_digit_precision)
    }

    #[pyo3(signature=(momenta, decimal_digit_precision, helicities=None, color_flows=None))]
    fn evaluate_resolved_with_prec(
        &mut self,
        py: Python<'_>,
        momenta: &Bound<'_, PyAny>,
        decimal_digit_precision: u32,
        helicities: Option<Vec<String>>,
        color_flows: Option<Vec<String>>,
    ) -> PyResult<ResolvedEvaluation> {
        self.evaluate_resolved(
            py,
            momenta,
            helicities,
            color_flows,
            decimal_digit_precision,
        )
    }

    fn set_model_parameters(&mut self, mapping: &Bound<'_, PyAny>) -> PyResult<()> {
        let values = parse_model_parameters(mapping)?;
        self.runtime
            .set_model_parameters(&values)
            .map_err(python_error)
    }

    fn set_model_parameter(&mut self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = parse_complex_value(value, name)?;
        self.runtime
            .set_model_parameter(name, value.0, value.1)
            .map_err(python_error)
    }

    fn mute_warnings(&mut self) {
        self.runtime.mute_warnings();
    }

    fn unmute_warnings(&mut self) {
        self.runtime.unmute_warnings();
    }

    fn take_warnings(&mut self) -> Vec<String> {
        self.runtime.take_warnings()
    }
}

impl Runtime {
    fn emit_warnings(&mut self, py: Python<'_>) -> PyResult<()> {
        let pending = self.runtime.take_warnings();
        if pending.is_empty() {
            return Ok(());
        }
        let warnings = py.import("warnings")?;
        for message in pending {
            warnings.call_method1("warn", (message,))?;
        }
        Ok(())
    }
}

fn color_accuracy_name(value: ColorAccuracy) -> &'static str {
    value.as_str()
}

fn reduction_kind_name(value: ReductionKind) -> &'static str {
    match value {
        ReductionKind::LcDiagonal => "lc-diagonal",
        ReductionKind::ContractedColor => "contracted-color",
    }
}

fn parameter_kind_name(value: ParameterKind) -> &'static str {
    match value {
        ParameterKind::Normalization => "normalization",
        ParameterKind::Mass => "mass",
        ParameterKind::Width => "width",
        ParameterKind::Coupling => "coupling",
        ParameterKind::External => "external",
        ParameterKind::Derived => "derived",
    }
}

fn path_from_python(value: &Bound<'_, PyAny>) -> PyResult<PathBuf> {
    value.extract::<PathBuf>().map_err(|error| {
        PyTypeError::new_err(format!(
            "artifact must be a string or path-like object: {error}"
        ))
    })
}

fn parse_model_parameters(mapping: &Bound<'_, PyAny>) -> PyResult<BTreeMap<String, (f64, f64)>> {
    let items = mapping.call_method0("items").map_err(|_| {
        PyTypeError::new_err("model_parameters must be a mapping from names to numbers")
    })?;
    let mut result = BTreeMap::new();
    for item in items.try_iter()? {
        let item = item?;
        let name = item.get_item(0)?.extract::<String>()?;
        if name.is_empty() {
            return Err(PyValueError::new_err(
                "model parameter names must not be empty",
            ));
        }
        let value = item.get_item(1)?;
        if result
            .insert(name.clone(), parse_complex_value(&value, &name)?)
            .is_some()
        {
            return Err(PyValueError::new_err(format!(
                "duplicate model parameter {name:?}"
            )));
        }
    }
    Ok(result)
}

fn parse_complex_value(value: &Bound<'_, PyAny>, name: &str) -> PyResult<(f64, f64)> {
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(format!(
            "model parameter {name:?} must be numeric, not bool"
        )));
    }
    if let Ok(real) = value.extract::<f64>() {
        return Ok((real, 0.0));
    }
    let real = value.getattr("real")?.extract::<f64>().map_err(|_| {
        PyTypeError::new_err(format!(
            "model parameter {name:?} must be a real or complex number"
        ))
    })?;
    let imaginary = value.getattr("imag")?.extract::<f64>().map_err(|_| {
        PyTypeError::new_err(format!(
            "model parameter {name:?} must be a real or complex number"
        ))
    })?;
    Ok((real, imaginary))
}

fn parse_f64_momenta(
    momenta: &Bound<'_, PyAny>,
    expected_legs: usize,
) -> PyResult<(Vec<f64>, usize)> {
    #[cfg(feature = "numpy")]
    if let Ok(array) = momenta.extract::<PyReadonlyArray3<'_, f64>>() {
        let shape = array.shape();
        if shape[1] != expected_legs || shape[2] != 4 {
            return Err(PyValueError::new_err(format!(
                "momenta array has shape ({}, {}, {}), expected (points, {expected_legs}, 4)",
                shape[0], shape[1], shape[2]
            )));
        }
        if shape[0] == 0 {
            return Err(PyValueError::new_err(
                "momenta must contain at least one point",
            ));
        }
        if let Ok(values) = array.as_slice() {
            return Ok((values.to_vec(), shape[0]));
        }
    }

    let mut values = Vec::new();
    let mut point_count = 0;
    for (point_index, point) in momenta.try_iter()?.enumerate() {
        let point = point?;
        let mut leg_count = 0;
        for (leg_index, leg) in point.try_iter()?.enumerate() {
            let leg = leg?;
            let components = leg
                .try_iter()?
                .map(|component| component?.extract::<f64>())
                .collect::<PyResult<Vec<_>>>()?;
            if components.len() != 4 {
                return Err(PyValueError::new_err(format!(
                    "momenta point {point_index} leg {leg_index} has {} components, expected 4",
                    components.len()
                )));
            }
            values.extend(components);
            leg_count += 1;
        }
        if leg_count != expected_legs {
            return Err(PyValueError::new_err(format!(
                "momenta point {point_index} has {leg_count} external legs, expected {expected_legs}"
            )));
        }
        point_count += 1;
    }
    if point_count == 0 {
        return Err(PyValueError::new_err(
            "momenta must contain at least one point",
        ));
    }
    Ok((values, point_count))
}

fn resolved_f64_to_python(
    py: Python<'_>,
    resolved: NativeResolvedEvaluation,
    color_accuracy: String,
    precision: u32,
) -> PyResult<ResolvedEvaluation> {
    let shape = resolved.shape();
    let values = nested_f64_values(py, &resolved.values, shape.1, shape.2)?;
    let totals = PyList::new(py, resolved.totals())?.into_any().unbind();
    Ok(ResolvedEvaluation {
        values,
        totals,
        shape,
        helicity_ids: resolved.helicity_ids,
        color_ids: resolved.color_ids,
        color_accuracy,
        precision,
    })
}

fn nested_f64_values(
    py: Python<'_>,
    values: &[f64],
    helicity_count: usize,
    color_count: usize,
) -> PyResult<Py<PyAny>> {
    let point_width = helicity_count
        .checked_mul(color_count)
        .ok_or_else(|| EvaluationError::new_err("resolved shape overflow"))?;
    if point_width == 0 || values.len() % point_width != 0 {
        return Err(EvaluationError::new_err(
            "resolved value buffer does not match its shape",
        ));
    }
    let mut points = Vec::with_capacity(values.len() / point_width);
    for point in values.chunks(point_width) {
        let mut helicities = Vec::with_capacity(helicity_count);
        for colors in point.chunks(color_count) {
            helicities.push(PyList::new(py, colors.iter().copied())?.into_any().unbind());
        }
        points.push(PyList::new(py, helicities)?.into_any().unbind());
    }
    Ok(PyList::new(py, points)?.into_any().unbind())
}

fn runtime_profile_to_python<'py>(
    py: Python<'py>,
    profile: &NativeRuntimeProfile,
    point_count: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let payload = PyDict::new(py);
    payload.set_item("points", point_count)?;
    payload.set_item("wall_time_s", profile.total_s)?;
    payload.set_item("source_fill_time_s", profile.source_fill_s)?;
    payload.set_item("momentum_setup_time_s", profile.momentum_setup_s)?;
    payload.set_item("stage_input_pack_time_s", profile.stage_input_pack_s)?;
    payload.set_item(
        "stage_evaluator_call_time_s",
        profile.stage_evaluator_call_s,
    )?;
    payload.set_item("stage_evaluator_time_s", profile.stage_evaluator_s)?;
    payload.set_item("output_assign_time_s", profile.output_assign_s)?;
    payload.set_item(
        "amplitude_input_pack_time_s",
        profile.amplitude_input_pack_s,
    )?;
    payload.set_item(
        "amplitude_evaluator_call_time_s",
        profile.amplitude_evaluator_call_s,
    )?;
    payload.set_item("amplitude_evaluator_time_s", profile.amplitude_evaluator_s)?;
    payload.set_item("reduction_time_s", profile.reduction_s)?;
    payload.set_item(
        "stage_input_pack_by_stage_time_s",
        profile.stage_input_pack_by_stage_s.clone(),
    )?;
    payload.set_item(
        "stage_evaluator_call_by_stage_time_s",
        profile.stage_evaluator_call_by_stage_s.clone(),
    )?;
    payload.set_item(
        "stage_output_assign_by_stage_time_s",
        profile.stage_output_assign_by_stage_s.clone(),
    )?;
    payload.set_item("eager_initialize_time_s", profile.eager_initialize_s)?;
    payload.set_item("eager_gather_time_s", profile.eager_gather_s)?;
    payload.set_item("eager_kernel_call_time_s", profile.eager_kernel_call_s)?;
    payload.set_item(
        "eager_invocation_scatter_time_s",
        profile.eager_invocation_scatter_s,
    )?;
    payload.set_item("eager_finalization_time_s", profile.eager_finalization_s)?;
    payload.set_item(
        "eager_scatter_finalization_time_s",
        profile.eager_scatter_finalization_s,
    )?;
    payload.set_item("eager_closure_time_s", profile.eager_closure_s)?;
    payload.set_item("eager_reduction_time_s", profile.eager_reduction_s)?;
    payload.set_item("eager_copy_out_time_s", profile.eager_copy_out_s)?;
    payload.set_item("selector_planner_time_s", profile.selector_planner_s)?;
    payload.set_item("selector_gather_time_s", profile.selector_gather_s)?;
    payload.set_item("selector_scatter_time_s", profile.selector_scatter_s)?;
    payload.set_item("selector_plan_kind", &profile.selector_plan_kind)?;
    payload.set_item("selector_group_sizes", profile.selector_group_sizes.clone())?;
    payload.set_item(
        "selector_reordered_point_count",
        profile.selector_reordered_point_count,
    )?;
    payload.set_item("selector_simd_lane_width", profile.selector_simd_lane_width)?;
    payload.set_item("selector_simd_occupancy", profile.selector_simd_occupancy)?;
    Ok(payload)
}

#[pyfunction]
fn abi_version() -> u32 {
    NativeRuntime::ABI_VERSION
}

#[pyfunction]
fn package_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyfunction]
fn target_info() -> TargetInfo {
    let target = runtime_target_info();
    TargetInfo {
        triple: target.triple,
        cpu_features: target.cpu_features,
    }
}

#[pyfunction]
fn _preflight_eager_kernel_pack(manifest_path: PathBuf, payload_root: PathBuf) -> PyResult<usize> {
    preflight_prepared_kernel_pack(&manifest_path, &payload_root).map_err(python_error)
}

#[pymodule]
fn _rusticol(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("RusticolError", module.py().get_type::<RusticolError>())?;
    module.add("ArtifactError", module.py().get_type::<ArtifactError>())?;
    module.add(
        "CompatibilityError",
        module.py().get_type::<CompatibilityError>(),
    )?;
    module.add("EvaluationError", module.py().get_type::<EvaluationError>())?;
    module.add("SelectorError", module.py().get_type::<SelectorError>())?;
    module.add(
        "ModelParameterError",
        module.py().get_type::<ModelParameterError>(),
    )?;
    module.add_class::<Runtime>()?;
    module.add_class::<ProcessPhysics>()?;
    module.add_class::<ExternalParticle>()?;
    module.add_class::<HelicityConfiguration>()?;
    module.add_class::<ColorFlow>()?;
    module.add_class::<ContractedColorComponent>()?;
    module.add_class::<ReductionGroup>()?;
    module.add_class::<PhysicsReduction>()?;
    module.add_class::<ModelParameter>()?;
    module.add_class::<ResolvedEvaluation>()?;
    module.add_class::<TargetInfo>()?;
    module.add_function(wrap_pyfunction!(abi_version, module)?)?;
    module.add_function(wrap_pyfunction!(package_version, module)?)?;
    module.add_function(wrap_pyfunction!(target_info, module)?)?;
    module.add_function(wrap_pyfunction!(_preflight_eager_kernel_pack, module)?)?;
    #[cfg(feature = "numpy")]
    module.add_function(wrap_pyfunction!(
        eager_lowering::_lower_eager_runtime_v1,
        module
    )?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusticol_core::{
        ContractedColor as CoreContractedColor, Coverage, ExternalParticle as CoreExternalParticle,
        Helicity, LcColorFlow, Reduction, ReductionGroup as CoreReductionGroup, ReductionKind,
        SelectorCapabilities,
    };

    #[test]
    fn raw_runtime_accepts_only_f64_precision() {
        assert!(require_raw_f64_precision(16).is_ok());
        let error = require_raw_f64_precision(32).unwrap_err();
        assert_eq!(error.kind(), RusticolErrorKind::Compatibility);
        assert!(error.to_string().contains("only precision=16"));
        assert!(
            error
                .to_string()
                .contains("public Python Symbolica executor")
        );
    }

    #[test]
    fn process_physics_maps_the_canonical_python_metadata_fields() {
        let mut physics = CoreProcessPhysics {
            schema_version: 1,
            kind: "pyamplicol-resolved-physics".to_string(),
            process_id: "p0".to_string(),
            process: "d d~ > z".to_string(),
            color_accuracy: ColorAccuracy::Lc,
            coverage: Coverage {
                helicities: "complete".to_string(),
                color: "complete".to_string(),
                color_kind: "physical-lc-flows".to_string(),
                structural_zero_helicity_count: 0,
            },
            external_particles: vec![
                core_particle(0, "d", 1, ParticleRole::Initial),
                core_particle(1, "d~", -1, ParticleRole::Initial),
                core_particle(2, "z", 23, ParticleRole::Final),
            ],
            helicities: vec![Helicity {
                id: "h0".to_string(),
                index: 0,
                values: vec![-1, 1, 0],
                computed: true,
                structural_zero: false,
                representative_id: "h0".to_string(),
                coefficient: 2.0,
            }],
            color_components: vec![CoreColorComponent::LcFlow(LcColorFlow {
                id: "c0".to_string(),
                index: 0,
                word: vec![1, 2],
                computed: true,
                representative_id: "c0".to_string(),
                coefficient: 3.0,
            })],
            reduction: Reduction {
                kind: ReductionKind::LcDiagonal,
                groups: vec![CoreReductionGroup {
                    id: "group:0".to_string(),
                    representative_helicity_id: "h0".to_string(),
                    representative_color_id: "c0".to_string(),
                    physical_helicity_ids: vec!["h0".to_string()],
                    physical_color_ids: vec!["c0".to_string()],
                }],
            },
            model_parameters: vec![CoreModelParameter {
                name: "alpha_s".to_string(),
                kind: ParameterKind::Coupling,
                default_real: 0.118,
                default_imaginary: 0.0,
                mutable: true,
            }],
            selectors: SelectorCapabilities {
                helicity: true,
                color_flow: true,
                contracted_color: false,
            },
            extensions: BTreeMap::new(),
        };

        physics.validate().unwrap();
        let mapped = ProcessPhysics::from_core(&physics);

        assert_eq!(mapped.process_id, "p0");
        assert_eq!(mapped.process, "d d~ > z");
        assert_eq!(mapped.color_accuracy, "lc");
        assert_eq!(mapped.helicity_coverage, "complete");
        assert_eq!(mapped.color_coverage, "complete");
        assert_eq!(mapped.color_kind, "physical-lc-flows");
        assert_eq!(mapped.structural_zero_helicity_count, 0);
        assert_eq!(mapped.external_particles[0].name, "d");
        assert_eq!(mapped.external_particles[0].pdg_id, 1);
        assert_eq!(mapped.external_particles[0].state, "incoming");
        assert_eq!(mapped.external_particles[2].state, "outgoing");
        assert_eq!(mapped.helicities[0].representative_id, "h0");
        assert_eq!(mapped.helicities[0].coefficient, 2.0);
        assert_eq!(mapped.color_flows[0].word, vec![1, 2]);
        assert_eq!(mapped.color_flows[0].coefficient, 3.0);
        assert!(mapped.contracted_color_components.is_empty());
        assert_eq!(mapped.reduction.kind, "lc-diagonal");
        assert_eq!(mapped.reduction.groups[0].id, "group:0");
        assert_eq!(mapped.reduction.groups[0].physical_helicity_ids, vec!["h0"]);
        assert_eq!(mapped.model_parameters[0].kind, "coupling");
        assert_eq!(mapped.model_parameters[0].default_real, 0.118);
        assert!(mapped.model_parameters[0].mutable);
        assert_eq!(mapped.selector_capabilities, vec!["helicity", "color_flow"]);

        physics.color_accuracy = ColorAccuracy::Full;
        physics.coverage.color = "contracted".to_string();
        physics.coverage.color_kind = "contracted-color".to_string();
        physics.color_components = vec![CoreColorComponent::ContractedColor(CoreContractedColor {
            id: "contracted".to_string(),
            index: 0,
            description: "fully contracted color".to_string(),
        })];
        physics.reduction.kind = ReductionKind::ContractedColor;
        physics.reduction.groups[0].representative_color_id = "contracted".to_string();
        physics.reduction.groups[0].physical_color_ids = vec!["contracted".to_string()];
        physics.selectors.color_flow = false;
        physics.validate().unwrap();

        let mapped = ProcessPhysics::from_core(&physics);
        assert!(mapped.color_flows.is_empty());
        assert_eq!(mapped.contracted_color_components[0].id, "contracted");
        assert_eq!(mapped.selector_capabilities, vec!["helicity"]);
    }

    #[test]
    fn target_info_is_canonical_and_matches_the_core() {
        let mapped = target_info();
        let core = runtime_target_info();
        assert_eq!(mapped.triple, core.triple);
        assert_eq!(mapped.cpu_features, core.cpu_features);
        assert!(mapped.cpu_features.windows(2).all(|pair| pair[0] < pair[1]));
    }

    fn core_particle(
        index: usize,
        name: &str,
        pdg: i32,
        role: ParticleRole,
    ) -> CoreExternalParticle {
        CoreExternalParticle {
            index,
            label: index + 1,
            particle: name.to_string(),
            pdg,
            role,
            momentum_slot: index,
            momentum_components: ["E", "px", "py", "pz"].map(str::to_string),
        }
    }
}
