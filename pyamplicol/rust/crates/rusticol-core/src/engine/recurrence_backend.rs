// SPDX-License-Identifier: 0BSD

use super::PreparedKernelPackManifest;
use super::evaluator::recurrence_closure_direct::execute_closure_reduce_rows;
use super::evaluator::recurrence_source_direct::{
    DirectSourceDispatchDomainSpec, LoadedDirectSourceExecutor,
};
#[cfg(feature = "f64-symjit")]
use super::evaluator::symjit_direct::{
    LoadedSymjitDirectExecutor, SymjitDirectPlaneProjection, SymjitDirectScalarProjection,
    execute_identity_finalization_rows,
};
use crate::artifact::EvaluatorPayloadStore;
use crate::recurrence::direct_backend::{
    DirectExecutorCatalog, DirectExecutorHandle, DirectUnionSourceDispatchHandle,
};
use crate::recurrence::{DirectExecutorRole, DirectRecurrencePlan, SemanticDigest};
use crate::{RusticolError, RusticolResult, VerifiedArtifact};
#[cfg(feature = "f64-symjit")]
use sha2::{Digest, Sha256};
#[cfg(feature = "f64-symjit")]
use std::ffi::c_void;
use std::path::Path;
#[cfg(feature = "f64-symjit")]
use std::path::PathBuf;
#[cfg(feature = "f64-symjit")]
use std::ptr;
#[cfg(feature = "f64-symjit")]
use symjit::{
    DirectApplicationMetadata, DirectDestinationOperation as SymjitDestinationOperation,
    DirectInputBinding,
};

use super::eager_manifest::RecurrenceDirectTemplateManifest;
#[cfg(feature = "f64-symjit")]
use super::eager_manifest::{
    RecurrenceDirectParameterBindingManifest, RecurrenceDirectPlaneProjectionManifest,
    RecurrenceDirectScalarProjectionManifest,
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
/// `source` and `symjit`. Those owners therefore must remain alive for every
/// direct runtime call.
pub(super) struct NativeRecurrenceDirectExecutorBackend {
    catalog: DirectExecutorCatalog,
    owners: NativeRecurrenceDirectExecutorOwners,
}

/// Immutable context ownership retained beside the native recurrence scheduler.
pub(super) struct NativeRecurrenceDirectExecutorOwners {
    source: Option<LoadedDirectSourceExecutor>,
    #[cfg(feature = "f64-symjit")]
    symjit: Vec<LoadedSymjitDirectExecutor>,
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
        #[cfg(not(feature = "f64-symjit"))]
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
        #[cfg(feature = "f64-symjit")]
        let mut symjit = Vec::new();
        let mut handles = Vec::with_capacity(direct.templates.len());
        for template in &direct.templates {
            let handle = match template.payload_binding.kind.as_str() {
                "rusticol-intrinsic" => load_intrinsic_handle(template, source.as_ref())?,
                "prepared-direct-call" => {
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
                #[cfg(feature = "f64-symjit")]
                symjit,
                summary,
            },
        })
    }

    pub(super) fn catalog(&self) -> &DirectExecutorCatalog {
        &self.catalog
    }

    pub(super) fn summary(&self) -> &NativeRecurrenceDirectBackendSummary {
        &self.owners.summary
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

    #[cfg(test)]
    fn owner_counts(&self) -> (usize, usize) {
        (usize::from(self.source.is_some()), {
            #[cfg(feature = "f64-symjit")]
            {
                self.symjit.len()
            }
            #[cfg(not(feature = "f64-symjit"))]
            {
                0
            }
        })
    }
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
            #[cfg(feature = "f64-symjit")]
            {
                Ok(DirectExecutorHandle::Finalization {
                    call: execute_identity_finalization_rows,
                    context: ptr::null::<c_void>(),
                })
            }
            #[cfg(not(feature = "f64-symjit"))]
            {
                Err(RusticolError::compatibility(
                    "Direct-Arena identity finalization is unavailable without f64-symjit",
                ))
            }
        }
        "closure" if runtime_template.starts_with("rusticol.closure-reduce.v1:") => {
            Ok(DirectExecutorHandle::Closure {
                call: execute_closure_reduce_rows,
                context: std::ptr::null::<c_void>(),
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
