// SPDX-License-Identifier: 0BSD

#[cfg(feature = "f64-compiled")]
use super::evaluator::native_direct::LoadedNativeDirectExecutor;
use super::evaluator::recurrence_closure_direct::execute_closure_reduce_rows;
use super::evaluator::recurrence_intrinsic_direct::{
    FEYNMAN_VECTOR_PROPAGATOR_TEMPLATE, LoadedRecurrenceIntrinsicDirectExecutor,
    RecurrenceIntrinsicScale, WEYL_PROPAGATOR_NEGATIVE_TEMPLATE, WEYL_PROPAGATOR_POSITIVE_TEMPLATE,
    execute_identity_finalization_rows,
};
use super::evaluator::recurrence_source_direct::{
    DirectSourceDispatchDomainSpec, LoadedDirectSourceExecutor,
};
#[cfg(feature = "f64-symjit")]
use super::evaluator::symjit_direct::{
    LoadedSymjitDirectExecutor, SymjitDirectPlaneProjection, SymjitDirectScalarProjection,
};
use super::{PreparedKernelManifest, PreparedKernelPackManifest};
use crate::artifact::EvaluatorPayloadStore;
use crate::recurrence::direct_backend::{
    DirectExecutorCatalog, DirectExecutorHandle, DirectUnionSourceDispatchHandle,
};
use crate::recurrence::{DirectExecutorRole, DirectRecurrencePlan, SemanticDigest};
use crate::{RusticolError, RusticolResult, VerifiedArtifact};
#[cfg(feature = "f64-symjit")]
use sha2::{Digest, Sha256};
use std::path::Path;
#[cfg(feature = "f64-symjit")]
use std::path::PathBuf;
#[cfg(feature = "f64-symjit")]
use symjit::{
    DirectApplicationMetadata, DirectDestinationOperation as SymjitDestinationOperation,
    DirectInputBinding,
};

#[cfg(feature = "f64-symjit")]
use super::eager_manifest::RecurrenceDirectPlaneProjectionManifest;
use super::eager_manifest::{
    RecurrenceDirectParameterBindingManifest, RecurrenceDirectScalarProjectionManifest,
    RecurrenceDirectTemplateManifest,
};

/// Authenticated identity of one loaded Direct-Arena recurrence executor catalog.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct NativeRecurrenceDirectBackendSummary {
    pub(super) prepared_kernel_pack_digest: String,
    pub(super) direct_template_catalog_digest: String,
    pub(super) backend: String,
    pub(super) target_triple: String,
    pub(super) target_portable: bool,
    pub(super) executor_count: usize,
}

/// Context owners backing a loaded Direct-Arena executor catalog.
///
/// Handles stored in `catalog` address immutable heap allocations owned by
/// `source`, `symjit`, and `native`. Those owners therefore must remain alive
/// for every direct runtime call.
pub(super) struct NativeRecurrenceDirectExecutorBackend {
    catalog: DirectExecutorCatalog,
    owners: NativeRecurrenceDirectExecutorOwners,
}

/// Immutable context ownership retained beside the native recurrence scheduler.
pub(super) struct NativeRecurrenceDirectExecutorOwners {
    source: Option<LoadedDirectSourceExecutor>,
    _intrinsics: Vec<LoadedRecurrenceIntrinsicDirectExecutor>,
    #[cfg(feature = "f64-symjit")]
    _symjit: Vec<LoadedSymjitDirectExecutor>,
    #[cfg(feature = "f64-compiled")]
    _native: Vec<LoadedNativeDirectExecutor>,
    summary: NativeRecurrenceDirectBackendSummary,
}

impl NativeRecurrenceDirectExecutorBackend {
    /// Load Direct-Arena executors from a verified artifact payload store.
    ///
    /// `source_domains` is derived from the process runtime metadata. Its
    /// indices must match `DirectSourceRow::source_template_or_dispatch_domain`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_from_verified_artifact(
        manifest_json: &[u8],
        artifact: &VerifiedArtifact,
        payload_root: impl AsRef<Path>,
        plan: &DirectRecurrencePlan,
        expected_prepared_pack_digest: &str,
        expected_catalog_digest: &str,
        source_domains: Vec<DirectSourceDispatchDomainSpec>,
    ) -> RusticolResult<Self> {
        let payloads = artifact.evaluator_payload_store(payload_root.as_ref())?;
        Self::load_from_store(
            manifest_json,
            &payloads,
            plan,
            expected_prepared_pack_digest,
            expected_catalog_digest,
            source_domains,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_from_store(
        manifest_json: &[u8],
        payloads: &EvaluatorPayloadStore,
        plan: &DirectRecurrencePlan,
        expected_prepared_pack_digest: &str,
        expected_catalog_digest: &str,
        source_domains: Vec<DirectSourceDispatchDomainSpec>,
    ) -> RusticolResult<Self> {
        #[cfg(not(any(feature = "f64-symjit", feature = "f64-compiled")))]
        let _ = payloads;
        let pack: PreparedKernelPackManifest =
            serde_json::from_slice(manifest_json).map_err(|error| {
                RusticolError::serialization(format!(
                    "could not parse prepared Direct-Arena kernel pack: {error}"
                ))
            })?;
        pack.validate()?;
        let direct = pack.recurrence_direct_template_catalog(
            expected_prepared_pack_digest,
            expected_catalog_digest,
        )?;
        if direct.catalog_digest != plan.direct_template_catalog_digest().to_string() {
            return Err(RusticolError::integrity(
                "prepared Direct-Arena catalog digest does not match the recurrence plan",
            ));
        }
        let source_needed = direct
            .templates
            .iter()
            .any(|template| template.role == "source");
        let source = if source_needed {
            Some(LoadedDirectSourceExecutor::load(source_domains)?)
        } else {
            if !source_domains.is_empty() {
                return Err(RusticolError::integrity(
                    "Direct-Arena source domains were supplied to a catalog without source executors",
                ));
            }
            None
        };
        let mut intrinsics = Vec::new();
        #[cfg(feature = "f64-symjit")]
        let mut symjit = Vec::new();
        #[cfg(feature = "f64-compiled")]
        let mut native = Vec::new();
        let mut handles = Vec::with_capacity(direct.templates.len());
        for template in &direct.templates {
            let handle = match template.payload_binding.kind.as_str() {
                "rusticol-intrinsic" if template.role == "contribution" => {
                    let loaded = load_contribution_intrinsic(template)?;
                    let handle = loaded.handle();
                    intrinsics.push(loaded);
                    handle
                }
                "rusticol-intrinsic" => load_intrinsic_handle(template, source.as_ref())?,
                "prepared-direct-call" => match template.backend.as_str() {
                    "jit" => {
                        #[cfg(feature = "f64-symjit")]
                        {
                            let loaded = load_symjit_executor(template, payloads)?;
                            let handle = loaded.handle();
                            symjit.push(loaded);
                            handle
                        }
                        #[cfg(not(feature = "f64-symjit"))]
                        {
                            return Err(RusticolError::compatibility(
                                "Direct-Arena JIT recurrence execution requires the f64-symjit feature",
                            ));
                        }
                    }
                    "cpp" | "asm" => {
                        #[cfg(feature = "f64-compiled")]
                        {
                            let loaded = load_native_executor(template, &pack, payloads)?;
                            let handle = loaded.handle();
                            native.push(loaded);
                            handle
                        }
                        #[cfg(not(feature = "f64-compiled"))]
                        {
                            return Err(RusticolError::compatibility(
                                "Direct-Arena C++/ASM recurrence execution requires the f64-compiled feature",
                            ));
                        }
                    }
                    other => {
                        return Err(RusticolError::compatibility(format!(
                            "unsupported Direct-Arena prepared-call backend {other:?}"
                        )));
                    }
                },
                other => {
                    return Err(RusticolError::compatibility(format!(
                        "unsupported Direct-Arena executor binding {other:?}"
                    )));
                }
            };
            if handle.role() != direct_role(&template.role)? {
                return Err(RusticolError::integrity(format!(
                    "Direct-Arena executor {} resolved with the wrong role",
                    template.direct_executor_id
                )));
            }
            handles.push(handle);
        }
        let digest = semantic_digest(&direct.catalog_digest, "Direct-Arena catalog")?;
        let catalog = DirectExecutorCatalog::new(plan, digest, handles)?;
        let summary = NativeRecurrenceDirectBackendSummary {
            prepared_kernel_pack_digest: direct.prepared_kernel_pack_digest.clone(),
            direct_template_catalog_digest: direct.catalog_digest.clone(),
            backend: direct.backend.clone(),
            target_triple: direct.target_triple.clone(),
            target_portable: direct.portable,
            executor_count: direct.templates.len(),
        };
        Ok(Self {
            catalog,
            owners: NativeRecurrenceDirectExecutorOwners {
                source,
                _intrinsics: intrinsics,
                #[cfg(feature = "f64-symjit")]
                _symjit: symjit,
                #[cfg(feature = "f64-compiled")]
                _native: native,
                summary,
            },
        })
    }

    /// Split the lightweight handle catalog from the contexts that it addresses.
    ///
    /// The caller must retain `owners` for at least as long as `catalog` can be
    /// invoked. Contexts are boxed, so moving either returned value cannot
    /// invalidate a handle.
    pub(super) fn into_parts(
        self,
    ) -> (DirectExecutorCatalog, NativeRecurrenceDirectExecutorOwners) {
        (self.catalog, self.owners)
    }
}

impl NativeRecurrenceDirectExecutorOwners {
    pub(super) fn summary(&self) -> &NativeRecurrenceDirectBackendSummary {
        &self.summary
    }

    pub(super) fn union_source_dispatch(&self) -> RusticolResult<DirectUnionSourceDispatchHandle> {
        self.source
            .as_ref()
            .map(LoadedDirectSourceExecutor::union_handle)
            .ok_or_else(|| {
                RusticolError::integrity(
                    "all-flow-union recurrence backend has no SourceIR dispatcher",
                )
            })
    }
}

#[cfg(feature = "f64-compiled")]
fn load_native_executor(
    template: &RecurrenceDirectTemplateManifest,
    pack: &PreparedKernelPackManifest,
    payloads: &EvaluatorPayloadStore,
) -> RusticolResult<LoadedNativeDirectExecutor> {
    let binding = &template.payload_binding;
    let library_path = binding.source_application_path.as_deref().ok_or_else(|| {
        RusticolError::artifact("native Direct-Arena prepared call has no library payload")
    })?;
    let exported_symbol = binding.native_entry_point.as_deref().ok_or_else(|| {
        RusticolError::artifact("native Direct-Arena prepared call has no exported symbol")
    })?;
    let prepared_kernel_id = binding.prepared_kernel_id.ok_or_else(|| {
        RusticolError::artifact("native Direct-Arena prepared call has no kernel ID")
    })?;
    let kernel = pack
        .kernels
        .iter()
        .find(|kernel| kernel.kernel_id == prepared_kernel_id)
        .ok_or_else(|| {
            RusticolError::integrity(format!(
                "native Direct-Arena prepared kernel {prepared_kernel_id} is absent"
            ))
        })?;
    let binding_coupling = native_binding_coupling(binding, kernel)?;
    LoadedNativeDirectExecutor::load(
        payloads.physical_path(library_path)?,
        direct_role(&template.role)?,
        prepared_kernel_id,
        exported_symbol,
        binding_coupling,
    )
}

#[cfg(feature = "f64-compiled")]
fn native_binding_coupling(
    binding: &super::eager_manifest::RecurrenceDirectPayloadBindingManifest,
    kernel: &PreparedKernelManifest,
) -> RusticolResult<Option<(f64, f64)>> {
    let expected_parameter_bindings =
        kernel.input_contracts.len().checked_mul(2).ok_or_else(|| {
            RusticolError::artifact("native Direct-Arena input count exceeds usize")
        })?;
    if binding.parameter_bindings.len() != expected_parameter_bindings {
        return Err(RusticolError::integrity(format!(
            "native Direct-Arena kernel {} has {} parameter bindings, expected {}",
            kernel.kernel_id,
            binding.parameter_bindings.len(),
            expected_parameter_bindings,
        )));
    }

    let mut coupling_re = None;
    let mut coupling_im = None;
    for (input_index, input) in kernel.input_contracts.iter().enumerate() {
        let destination = match input.role.as_str() {
            "coupling-real" => &mut coupling_re,
            "coupling-imag" => &mut coupling_im,
            _ => continue,
        };
        if destination.is_some() {
            return Err(RusticolError::integrity(format!(
                "native Direct-Arena kernel {} repeats {:?}",
                kernel.kernel_id, input.role
            )));
        }
        let base = input_index * 2;
        *destination = Some(native_literal_binding(
            binding,
            base,
            &format!("kernel {} {} input", kernel.kernel_id, input.role),
        )?);
        let imaginary = native_literal_binding(
            binding,
            base + 1,
            &format!("kernel {} {} imaginary input", kernel.kernel_id, input.role),
        )?;
        if imaginary != 0.0 {
            return Err(RusticolError::integrity(format!(
                "native Direct-Arena kernel {} coupling input has a nonzero imaginary scalar lane",
                kernel.kernel_id
            )));
        }
    }
    if coupling_re.is_none() && coupling_im.is_none() {
        return Ok(None);
    }
    Ok(Some((
        coupling_re.unwrap_or(0.0),
        coupling_im.unwrap_or(0.0),
    )))
}

#[cfg(feature = "f64-compiled")]
fn native_literal_binding(
    binding: &super::eager_manifest::RecurrenceDirectPayloadBindingManifest,
    parameter_index: usize,
    label: &str,
) -> RusticolResult<f64> {
    let scalar_index = match binding.parameter_bindings.get(parameter_index) {
        Some(RecurrenceDirectParameterBindingManifest::Scalar { index }) => *index,
        _ => {
            return Err(RusticolError::integrity(format!(
                "native Direct-Arena {label} is not bound to a scalar"
            )));
        }
    };
    let value = match binding.scalar_projections.get(scalar_index as usize) {
        Some(RecurrenceDirectScalarProjectionManifest::Literal { value }) => *value,
        _ => {
            return Err(RusticolError::integrity(format!(
                "native Direct-Arena {label} is not an immutable literal"
            )));
        }
    };
    if !value.is_finite() {
        return Err(RusticolError::integrity(format!(
            "native Direct-Arena {label} is non-finite"
        )));
    }
    Ok(value)
}

fn load_contribution_intrinsic(
    template: &RecurrenceDirectTemplateManifest,
) -> RusticolResult<LoadedRecurrenceIntrinsicDirectExecutor> {
    let runtime_template = template
        .payload_binding
        .runtime_template
        .as_deref()
        .ok_or_else(|| RusticolError::artifact("contribution intrinsic has no template"))?;
    let scale = match template.payload_binding.scalar_projections.as_slice() {
        [
            RecurrenceDirectScalarProjectionManifest::IntrinsicScale {
                constant_real_bits,
                constant_imag_bits,
                parameter_index,
            },
        ] => RecurrenceIntrinsicScale::new(
            f64::from_bits(*constant_real_bits),
            f64::from_bits(*constant_imag_bits),
            *parameter_index,
        )?,
        _ => {
            return Err(RusticolError::integrity(
                "contribution intrinsic must carry exactly one intrinsic scale",
            ));
        }
    };
    LoadedRecurrenceIntrinsicDirectExecutor::load_runtime_template(runtime_template, scale)
}

fn load_intrinsic_handle(
    template: &RecurrenceDirectTemplateManifest,
    source: Option<&LoadedDirectSourceExecutor>,
) -> RusticolResult<DirectExecutorHandle> {
    let runtime_template = template
        .payload_binding
        .runtime_template
        .as_deref()
        .ok_or_else(|| RusticolError::artifact("Direct-Arena intrinsic has no runtime template"))?;
    match template.role.as_str() {
        "source" if runtime_template.starts_with("rusticol.source-fill.") => {
            let source = source.ok_or_else(|| {
                RusticolError::integrity("Direct-Arena source executor owner is absent")
            })?;
            let handle = source.handle();
            Ok(DirectExecutorHandle::Source {
                call: handle.call,
                context: handle.context,
            })
        }
        "finalization" if runtime_template == "rusticol.identity-finalize-in-place.v1" => {
            Ok(DirectExecutorHandle::Finalization {
                call: execute_identity_finalization_rows,
                context: std::ptr::null::<std::ffi::c_void>(),
            })
        }
        "finalization"
            if matches!(
                runtime_template,
                WEYL_PROPAGATOR_POSITIVE_TEMPLATE
                    | WEYL_PROPAGATOR_NEGATIVE_TEMPLATE
                    | FEYNMAN_VECTOR_PROPAGATOR_TEMPLATE
            ) =>
        {
            LoadedRecurrenceIntrinsicDirectExecutor::finalization_handle(runtime_template)
        }
        "closure" if runtime_template.starts_with("rusticol.closure-reduce.v1:") => {
            Ok(DirectExecutorHandle::Closure {
                call: execute_closure_reduce_rows,
                context: std::ptr::null::<std::ffi::c_void>(),
            })
        }
        _ => Err(RusticolError::compatibility(format!(
            "unsupported Direct-Arena intrinsic {runtime_template:?}"
        ))),
    }
}

#[cfg(feature = "f64-symjit")]
fn load_symjit_executor(
    template: &RecurrenceDirectTemplateManifest,
    payloads: &EvaluatorPayloadStore,
) -> RusticolResult<LoadedSymjitDirectExecutor> {
    let binding = &template.payload_binding;
    let source_path = binding.source_application_path.as_deref().ok_or_else(|| {
        RusticolError::artifact("Direct-Arena prepared call has no source application")
    })?;
    let source = payloads.source(source_path)?;
    let bytes = source.read()?;
    let expected_sha = binding
        .source_application_sha256
        .as_deref()
        .ok_or_else(|| RusticolError::artifact("Direct-Arena source application has no digest"))?;
    let actual_sha = format!("{:x}", Sha256::digest(bytes.as_ref()));
    if actual_sha != expected_sha {
        return Err(RusticolError::integrity(format!(
            "Direct-Arena source application {} has digest {actual_sha}, expected {expected_sha}",
            source.display_name()
        )));
    }
    let role = direct_role(&template.role)?;
    let operation = direct_destination_operation(&template.destination_operation)?;
    let parameter_bindings = binding
        .parameter_bindings
        .iter()
        .map(|binding| match *binding {
            RecurrenceDirectParameterBindingManifest::Plane { index } => {
                DirectInputBinding::Plane(index)
            }
            RecurrenceDirectParameterBindingManifest::Scalar { index } => {
                DirectInputBinding::Scalar(index)
            }
        })
        .collect();
    let metadata = DirectApplicationMetadata::new(
        operation,
        binding.state_plane_indices.clone(),
        parameter_bindings,
        binding.input_plane_count,
        binding.scalar_input_count,
        binding.output_alias_inputs.clone(),
    )
    .map_err(|error| {
        RusticolError::integrity(format!(
            "invalid Direct-Arena SymJIT metadata for executor {}: {error}",
            template.direct_executor_id
        ))
    })?;
    let input_planes = binding
        .input_plane_projections
        .iter()
        .copied()
        .map(plane_projection)
        .collect();
    let scalars = binding
        .scalar_projections
        .iter()
        .copied()
        .map(scalar_projection)
        .collect::<RusticolResult<Vec<_>>>()?;
    LoadedSymjitDirectExecutor::load_prepared_application_bytes(
        bytes.as_ref(),
        PathBuf::from(source.display_name()),
        binding
            .source_application_abi
            .as_deref()
            .unwrap_or_default(),
        role,
        metadata,
        input_planes,
        scalars,
    )
}

#[cfg(feature = "f64-symjit")]
fn plane_projection(
    projection: RecurrenceDirectPlaneProjectionManifest,
) -> SymjitDirectPlaneProjection {
    match projection {
        RecurrenceDirectPlaneProjectionManifest::ParentCurrent {
            parent,
            component,
            imaginary,
        } => SymjitDirectPlaneProjection::ParentCurrent {
            parent,
            component,
            imaginary,
        },
        RecurrenceDirectPlaneProjectionManifest::Momentum {
            operand,
            lorentz_component,
        } => SymjitDirectPlaneProjection::Momentum {
            operand,
            lorentz_component,
        },
        RecurrenceDirectPlaneProjectionManifest::DestinationCurrent {
            component,
            imaginary,
        } => SymjitDirectPlaneProjection::DestinationCurrent {
            component,
            imaginary,
        },
        RecurrenceDirectPlaneProjectionManifest::DestinationAmplitude {
            component,
            imaginary,
        } => SymjitDirectPlaneProjection::DestinationAmplitude {
            component,
            imaginary,
        },
    }
}

#[cfg(feature = "f64-symjit")]
fn scalar_projection(
    projection: RecurrenceDirectScalarProjectionManifest,
) -> RusticolResult<SymjitDirectScalarProjection> {
    Ok(match projection {
        RecurrenceDirectScalarProjectionManifest::ExactFactor { imaginary } => {
            SymjitDirectScalarProjection::ExactFactor { imaginary }
        }
        RecurrenceDirectScalarProjectionManifest::Parameter { index, imaginary } => {
            SymjitDirectScalarProjection::Parameter { index, imaginary }
        }
        RecurrenceDirectScalarProjectionManifest::Literal { value } => {
            if !value.is_finite() {
                return Err(RusticolError::artifact(
                    "Direct-Arena literal scalar projection is not finite",
                ));
            }
            SymjitDirectScalarProjection::Literal(value)
        }
        RecurrenceDirectScalarProjectionManifest::IntrinsicScale { .. } => {
            return Err(RusticolError::integrity(
                "intrinsic scale cannot be used by a prepared SymJIT callable",
            ));
        }
    })
}

fn direct_role(role: &str) -> RusticolResult<DirectExecutorRole> {
    match role {
        "source" => Ok(DirectExecutorRole::Source),
        "contribution" => Ok(DirectExecutorRole::Contribution),
        "finalization" => Ok(DirectExecutorRole::Finalization),
        "closure" => Ok(DirectExecutorRole::Closure),
        other => Err(RusticolError::compatibility(format!(
            "unsupported Direct-Arena executor role {other:?}"
        ))),
    }
}

#[cfg(feature = "f64-symjit")]
fn direct_destination_operation(value: &str) -> RusticolResult<SymjitDestinationOperation> {
    match value {
        "initialize" => Ok(SymjitDestinationOperation::Initialize),
        "add" => Ok(SymjitDestinationOperation::Add),
        "finalize-in-place" => Ok(SymjitDestinationOperation::FinalizeInPlace),
        "closure-add" => Ok(SymjitDestinationOperation::ClosureAdd),
        other => Err(RusticolError::compatibility(format!(
            "unsupported Direct-Arena destination operation {other:?}"
        ))),
    }
}

fn semantic_digest(value: &str, label: &str) -> RusticolResult<SemanticDigest> {
    let bytes = value.as_bytes();
    if bytes.len() != 64 {
        return Err(RusticolError::artifact(format!(
            "{label} is not a SHA-256 digest"
        )));
    }
    let mut digest = [0_u8; 32];
    for (index, pair) in bytes.chunks_exact(2).enumerate() {
        let high = hex_nibble(pair[0]).ok_or_else(|| {
            RusticolError::artifact(format!("{label} is not lowercase hexadecimal"))
        })?;
        let low = hex_nibble(pair[1]).ok_or_else(|| {
            RusticolError::artifact(format!("{label} is not lowercase hexadecimal"))
        })?;
        digest[index] = (high << 4) | low;
    }
    SemanticDigest::new(digest)
        .map_err(|_| RusticolError::artifact(format!("{label} must not be all zero")))
}

fn hex_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}
