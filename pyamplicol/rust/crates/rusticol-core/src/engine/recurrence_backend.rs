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
    LoadedSymjitDirectExecutor, SymjitDirectParameterBinding, SymjitDirectPlaneProjection,
    SymjitDirectScalarProjection,
};
use super::{PreparedKernelManifest, PreparedKernelPackManifest};
use crate::artifact::EvaluatorPayloadStore;
use crate::recurrence::direct_backend::{
    DirectContributionExecutionMetadata, DirectContributionFanoutExecutorHandle,
    DirectExecutorCatalog, DirectExecutorHandle, DirectUnionSourceDispatchHandle,
};
use crate::recurrence::on_the_fly::{
    OnTheFlyExecutorKeyV1, OnTheFlyPreparedExecutorResolver, OnTheFlyProcessSeedV1,
    OnTheFlySelectedQueryTraceV1, OnTheFlyStructuralTraceV1, ResolvedOnTheFlyExecutor,
    authenticated_prepared_executor_binding,
};
use crate::recurrence::template::ValidatedRecurrenceTemplateInput;
use crate::recurrence::{
    DIRECT_NONE_U32, DirectExecutorRole, DirectRecurrencePlan, PreparedDirectExecutorBinding,
    PreparedDirectExecutorCatalog, SemanticDigest,
};
use crate::{RusticolError, RusticolResult, VerifiedArtifact};
#[cfg(feature = "f64-symjit")]
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
#[cfg(any(test, feature = "on-the-fly-test-support"))]
use std::collections::BTreeSet;
use std::path::Path;
#[cfg(feature = "f64-symjit")]
use std::path::PathBuf;

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

/// Model-level ownership of one authenticated prepared-executor catalog.
///
/// Source executors are deliberately absent: their contexts contain crossed
/// source domains and therefore belong to a process/lane binding.  Every
/// other handle addresses immutable contexts owned by this pool and can be
/// shared by independent selector-local traces without constructing a direct
/// recurrence plan.
pub(super) struct NativeRecurrencePreparedExecutorPool {
    handles: Box<[Option<DirectExecutorHandle>]>,
    bindings: BTreeMap<(DirectExecutorRole, u32), NativePreparedExecutorBinding>,
    executor_roles: Box<[DirectExecutorRole]>,
    contribution_metadata: Box<[Option<DirectContributionExecutionMetadata>]>,
    contribution_fanout: Box<[Option<DirectContributionFanoutExecutorHandle>]>,
    identity_finalizer_id: Option<u32>,
    direct_template_catalog_digest: SemanticDigest,
    recurrence_template_catalog_digest: SemanticDigest,
    compiled_model_digest: SemanticDigest,
    prepared_kernel_pack_digest: SemanticDigest,
    _intrinsics: Vec<LoadedRecurrenceIntrinsicDirectExecutor>,
    #[cfg(feature = "f64-symjit")]
    _symjit: Vec<LoadedSymjitDirectExecutor>,
    #[cfg(feature = "f64-compiled")]
    _native: Vec<LoadedNativeDirectExecutor>,
    summary: NativeRecurrenceDirectBackendSummary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct NativePreparedExecutorBinding {
    executor_id: u32,
    parent_permutation: [u8; 2],
}

/// Process/lane-owned crossed-source dispatch binding.
///
/// This value must outlive every source handle obtained from it and must be
/// dropped before the model-level prepared pool during lane teardown.
pub(super) struct OnTheFlySourceDomainBinding {
    source: Option<LoadedDirectSourceExecutor>,
}

/// Owning semantic resolver used by the private on-the-fly interpreter.
///
/// The copied direct handles address heap-stable contexts owned by `sources`
/// and `pool`. Declaration order is intentional: resolved handles disappear
/// first, then process-bound source contexts, then the model-level pool.
pub(super) struct NativeOnTheFlyPreparedExecutorResolver {
    resolved: BTreeMap<OnTheFlyExecutorKeyV1, ResolvedOnTheFlyExecutor>,
    pending_resolved: Option<BTreeMap<OnTheFlyExecutorKeyV1, ResolvedOnTheFlyExecutor>>,
    sources: OnTheFlySourceDomainBinding,
    pool: NativeRecurrencePreparedExecutorPool,
}

impl NativeOnTheFlyPreparedExecutorResolver {
    fn effective_resolved(&self) -> &BTreeMap<OnTheFlyExecutorKeyV1, ResolvedOnTheFlyExecutor> {
        self.pending_resolved.as_ref().unwrap_or(&self.resolved)
    }

    pub(super) fn semantic_executor_binding_count(&self) -> RusticolResult<u32> {
        u32::try_from(self.effective_resolved().len()).map_err(|_| {
            RusticolError::integrity("on-the-fly semantic executor binding count exceeds u32")
        })
    }
}

#[cfg(any(test, feature = "on-the-fly-test-support"))]
impl NativeOnTheFlyPreparedExecutorResolver {
    pub(super) fn distinct_prepared_executor_count(&self) -> RusticolResult<u32> {
        let direct_executor_ids = self
            .effective_resolved()
            .values()
            .map(|resolved| resolved.direct_executor_id)
            .collect::<BTreeSet<_>>();
        u32::try_from(direct_executor_ids.len())
            .map_err(|_| RusticolError::integrity("on-the-fly prepared executor count exceeds u32"))
    }
}

/// Immutable context ownership retained beside the native recurrence scheduler.
pub(super) struct NativeRecurrenceDirectExecutorOwners {
    // Field order documents the required destruction order. Rust drops fields
    // in declaration order: process-bound source contexts before model pool.
    source_binding: OnTheFlySourceDomainBinding,
    pool: NativeRecurrencePreparedExecutorPool,
}

impl NativeRecurrenceDirectExecutorBackend {
    /// Load Direct-Arena executors from a verified artifact payload store.
    ///
    /// `source_domains` is derived from the process runtime metadata. Its
    /// indices must match `DirectSourceRow::source_template_or_dispatch_domain`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_from_verified_artifact(
        pack: &mut PreparedKernelPackManifest,
        artifact: &VerifiedArtifact,
        payload_root: impl AsRef<Path>,
        plan: &DirectRecurrencePlan,
        expected_prepared_pack_digest: &str,
        expected_catalog_digest: &str,
        source_domains: Vec<DirectSourceDispatchDomainSpec>,
    ) -> RusticolResult<Self> {
        let payloads = artifact.evaluator_payload_store(payload_root.as_ref())?;
        Self::load_from_store(
            pack,
            &payloads,
            plan,
            expected_prepared_pack_digest,
            expected_catalog_digest,
            source_domains,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn load_from_store(
        pack: &mut PreparedKernelPackManifest,
        payloads: &EvaluatorPayloadStore,
        plan: &DirectRecurrencePlan,
        expected_prepared_pack_digest: &str,
        expected_catalog_digest: &str,
        source_domains: Vec<DirectSourceDispatchDomainSpec>,
    ) -> RusticolResult<Self> {
        let pool = NativeRecurrencePreparedExecutorPool::load_for_direct_plan_from_validated_pack(
            pack,
            payloads,
            plan,
            expected_prepared_pack_digest,
            expected_catalog_digest,
        )?;
        let source_binding = pool.bind_source_domains(source_domains)?;
        let catalog = pool.bind_direct_plan(plan, &source_binding)?;
        Ok(Self {
            catalog,
            owners: NativeRecurrenceDirectExecutorOwners {
                source_binding,
                pool,
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

impl NativeRecurrencePreparedExecutorPool {
    /// Load and authenticate model-level prepared executors without binding a
    /// process source domain or a recurrence plan.
    pub(super) fn load_from_store(
        manifest_json: &[u8],
        payloads: &EvaluatorPayloadStore,
        expected_prepared_pack_digest: &str,
        expected_catalog_digest: &str,
    ) -> RusticolResult<Self> {
        #[cfg(not(any(feature = "f64-symjit", feature = "f64-compiled")))]
        let _ = payloads;
        let mut pack: PreparedKernelPackManifest =
            serde_json::from_slice(manifest_json).map_err(|error| {
                RusticolError::serialization(format!(
                    "could not parse prepared Direct-Arena kernel pack: {error}"
                ))
            })?;
        pack.validate()?;
        Self::load_from_validated_pack(
            &mut pack,
            payloads,
            None,
            expected_prepared_pack_digest,
            expected_catalog_digest,
        )
    }

    /// Load only the executor contexts referenced by one authenticated direct
    /// plan while retaining the complete prepared-catalog identity and index
    /// domain. The caller has already validated `pack` at the artifact boundary.
    fn load_for_direct_plan_from_validated_pack(
        pack: &mut PreparedKernelPackManifest,
        payloads: &EvaluatorPayloadStore,
        plan: &DirectRecurrencePlan,
        expected_prepared_pack_digest: &str,
        expected_catalog_digest: &str,
    ) -> RusticolResult<Self> {
        Self::load_from_validated_pack(
            pack,
            payloads,
            Some(plan),
            expected_prepared_pack_digest,
            expected_catalog_digest,
        )
    }

    fn load_from_validated_pack(
        pack: &mut PreparedKernelPackManifest,
        payloads: &EvaluatorPayloadStore,
        plan: Option<&DirectRecurrencePlan>,
        expected_prepared_pack_digest: &str,
        expected_catalog_digest: &str,
    ) -> RusticolResult<Self> {
        #[cfg(not(any(feature = "f64-symjit", feature = "f64-compiled")))]
        let _ = payloads;
        let direct = pack.take_recurrence_direct_template_catalog(
            expected_prepared_pack_digest,
            expected_catalog_digest,
        )?;
        let digest = semantic_digest(&direct.catalog_digest, "Direct-Arena catalog")?;
        if plan.is_some_and(|plan| digest != plan.direct_template_catalog_digest()) {
            return Err(RusticolError::integrity(
                "prepared Direct-Arena catalog digest does not match the recurrence plan",
            ));
        }
        let required_executors = plan
            .map(|plan| required_direct_executor_mask(plan, direct.templates.len()))
            .transpose()?;
        let mut intrinsics = Vec::new();
        #[cfg(feature = "f64-symjit")]
        let mut symjit = Vec::new();
        #[cfg(feature = "f64-compiled")]
        let mut native = Vec::new();
        let mut handles = Vec::with_capacity(direct.templates.len());
        let mut bindings = BTreeMap::new();
        let mut executor_roles = Vec::with_capacity(direct.templates.len());
        let mut contribution_metadata = Vec::with_capacity(direct.templates.len());
        let mut contribution_fanout = Vec::with_capacity(direct.templates.len());
        let mut identity_finalizer_id = None;
        for template in &direct.templates {
            let mut fanout = None;
            let role = direct_role(&template.role)?;
            let parent_permutation: [u8; 2] = template
                .payload_binding
                .contribution_parent_permutation
                .as_slice()
                .try_into()
                .map_err(|_| {
                    RusticolError::integrity(
                        "prepared Direct-Arena executor has an invalid parent permutation",
                    )
                })?;
            if bindings
                .insert(
                    (role, template.evaluator_binding_id),
                    NativePreparedExecutorBinding {
                        executor_id: template.direct_executor_id,
                        parent_permutation,
                    },
                )
                .is_some()
            {
                return Err(RusticolError::integrity(
                    "prepared Direct-Arena catalog repeats a semantic executor binding",
                ));
            }
            if template.payload_binding.kind == "rusticol-intrinsic"
                && template.payload_binding.runtime_template.as_deref()
                    == Some("rusticol.identity-finalize-in-place.v1")
                && identity_finalizer_id
                    .replace(template.direct_executor_id)
                    .is_some()
            {
                return Err(RusticolError::integrity(
                    "prepared Direct-Arena catalog repeats the identity finalizer",
                ));
            }
            let handle = instantiate_selected_executor(
                required_executors.as_deref(),
                template.direct_executor_id,
                || {
                    let handle = match template.payload_binding.kind.as_str() {
                        "rusticol-intrinsic" if template.role == "contribution" => {
                            let loaded = load_contribution_intrinsic(template)?;
                            let handle = loaded.handle();
                            fanout = loaded.contribution_fanout_handle();
                            intrinsics.push(loaded);
                            Some(handle)
                        }
                        "rusticol-intrinsic" if template.role == "source" => None,
                        "rusticol-intrinsic" => Some(load_intrinsic_handle(template)?),
                        "prepared-direct-call" => match template.backend.as_str() {
                            "jit" => {
                                #[cfg(feature = "f64-symjit")]
                                {
                                    let loaded = load_symjit_executor(template, pack, payloads)?;
                                    let handle = loaded.handle();
                                    symjit.push(loaded);
                                    Some(handle)
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
                                    let loaded = load_native_executor(template, pack, payloads)?;
                                    let handle = loaded.handle();
                                    native.push(loaded);
                                    Some(handle)
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
                    Ok(handle)
                },
            )?;
            if let Some(handle) = handle
                && handle.role() != role
            {
                return Err(RusticolError::integrity(format!(
                    "Direct-Arena executor {} resolved with the wrong role",
                    template.direct_executor_id
                )));
            }
            handles.push(handle);
            contribution_fanout.push(fanout);
            executor_roles.push(role);
            contribution_metadata.push(if role == DirectExecutorRole::Contribution {
                Some(DirectContributionExecutionMetadata::new(
                    template.destination_component_count,
                    !template
                        .payload_binding
                        .exact_factor_scalar_slots
                        .is_empty(),
                )?)
            } else {
                None
            });
        }
        let recurrence_template_catalog_digest = semantic_digest(
            &direct.recurrence_template_catalog_digest,
            "recurrence template catalog",
        )?;
        let compiled_model_digest =
            semantic_digest(&direct.compiled_model_digest, "compiled model")?;
        let prepared_kernel_pack_digest =
            semantic_digest(&direct.prepared_kernel_pack_digest, "prepared kernel pack")?;
        let summary = NativeRecurrenceDirectBackendSummary {
            prepared_kernel_pack_digest: direct.prepared_kernel_pack_digest.clone(),
            direct_template_catalog_digest: direct.catalog_digest.clone(),
            backend: direct.backend.clone(),
            target_triple: direct.target_triple.clone(),
            target_portable: direct.portable,
            executor_count: direct.templates.len(),
        };
        Ok(Self {
            handles: handles.into_boxed_slice(),
            bindings,
            executor_roles: executor_roles.into_boxed_slice(),
            contribution_metadata: contribution_metadata.into_boxed_slice(),
            contribution_fanout: contribution_fanout.into_boxed_slice(),
            identity_finalizer_id,
            direct_template_catalog_digest: digest,
            recurrence_template_catalog_digest,
            compiled_model_digest,
            prepared_kernel_pack_digest,
            _intrinsics: intrinsics,
            #[cfg(feature = "f64-symjit")]
            _symjit: symjit,
            #[cfg(feature = "f64-compiled")]
            _native: native,
            summary,
        })
    }

    pub(super) fn bind_source_domains(
        &self,
        source_domains: Vec<DirectSourceDispatchDomainSpec>,
    ) -> RusticolResult<OnTheFlySourceDomainBinding> {
        let source_needed = self
            .bindings
            .keys()
            .any(|(role, _)| *role == DirectExecutorRole::Source);
        let source = if source_needed {
            Some(LoadedDirectSourceExecutor::load(source_domains)?)
        } else {
            if !source_domains.is_empty() {
                return Err(RusticolError::integrity(
                    "source domains were supplied to a prepared catalog without source executors",
                ));
            }
            None
        };
        Ok(OnTheFlySourceDomainBinding { source })
    }

    /// Re-express the already authenticated prepared-pack mappings through
    /// the shared semantic resolver used by selector-local lowering.  This is
    /// a compact model-level catalog, not a process recurrence plan.
    pub(super) fn prepared_direct_catalog(&self) -> RusticolResult<PreparedDirectExecutorCatalog> {
        let mut bindings = self
            .bindings
            .iter()
            .map(|(&(role, evaluator_binding_id), binding)| {
                PreparedDirectExecutorBinding::evaluator_with_parent_permutation(
                    role,
                    evaluator_binding_id,
                    binding.executor_id,
                    binding.parent_permutation,
                )
            })
            .collect::<Vec<_>>();
        if let Some(executor_id) = self.identity_finalizer_id {
            bindings.push(PreparedDirectExecutorBinding::identity_finalizer(
                executor_id,
            ));
        }
        PreparedDirectExecutorCatalog::new(self.direct_template_catalog_digest, bindings)
    }

    pub(super) fn into_on_the_fly_resolver(
        self,
        sources: OnTheFlySourceDomainBinding,
    ) -> NativeOnTheFlyPreparedExecutorResolver {
        NativeOnTheFlyPreparedExecutorResolver {
            resolved: BTreeMap::new(),
            pending_resolved: None,
            sources,
            pool: self,
        }
    }

    fn resolve_handle(
        &self,
        sources: &OnTheFlySourceDomainBinding,
        role: DirectExecutorRole,
        evaluator_binding_id: u32,
    ) -> RusticolResult<(u32, DirectExecutorHandle, [u8; 2])> {
        let binding = self
            .bindings
            .get(&(role, evaluator_binding_id))
            .copied()
            .ok_or_else(|| {
                RusticolError::integrity(format!(
                    "prepared recurrence catalog has no {role:?} semantic binding {evaluator_binding_id}"
                ))
            })?;
        let handle = match role {
            DirectExecutorRole::Source => sources.source_handle()?,
            _ => self
                .handles
                .get(binding.executor_id as usize)
                .and_then(|handle| *handle)
                .ok_or_else(|| {
                    RusticolError::integrity(format!(
                        "prepared recurrence executor {} is not loaded",
                        binding.executor_id,
                    ))
                })?,
        };
        if handle.role() != role {
            return Err(RusticolError::integrity(format!(
                "prepared recurrence executor {} has role {:?}, expected {role:?}",
                binding.executor_id,
                handle.role()
            )));
        }
        Ok((binding.executor_id, handle, binding.parent_permutation))
    }

    fn identity_finalizer_handle(&self) -> RusticolResult<(u32, DirectExecutorHandle)> {
        let executor_id = self.identity_finalizer_id.ok_or_else(|| {
            RusticolError::integrity("prepared recurrence catalog has no identity finalizer")
        })?;
        let handle = self
            .handles
            .get(executor_id as usize)
            .and_then(|handle| *handle)
            .ok_or_else(|| RusticolError::integrity("identity finalizer is not loaded"))?;
        Ok((executor_id, handle))
    }

    fn bind_direct_plan(
        &self,
        plan: &DirectRecurrencePlan,
        sources: &OnTheFlySourceDomainBinding,
    ) -> RusticolResult<DirectExecutorCatalog> {
        if self.direct_template_catalog_digest != plan.direct_template_catalog_digest() {
            return Err(RusticolError::integrity(
                "prepared Direct-Arena catalog digest does not match the recurrence plan",
            ));
        }
        let required_executors = required_direct_executor_mask(plan, self.handles.len())?;
        let mut handles = Vec::with_capacity(self.handles.len());
        for (executor_id, handle) in self.handles.iter().copied().enumerate() {
            let selected = required_executors[executor_id];
            let handle = match (handle, selected) {
                (Some(handle), _) => Some(handle),
                (None, false) => None,
                (None, true) => {
                    let role = self
                        .executor_roles
                        .get(executor_id)
                        .copied()
                        .ok_or_else(|| {
                            RusticolError::integrity("prepared executor role is absent")
                        })?;
                    if role != DirectExecutorRole::Source {
                        return Err(RusticolError::integrity(format!(
                            "referenced prepared non-source executor {executor_id} has no loaded handle"
                        )));
                    }
                    Some(sources.source_handle()?)
                }
            };
            handles.push(handle);
        }
        DirectExecutorCatalog::new_sparse_with_metadata_and_fanout(
            plan,
            self.direct_template_catalog_digest,
            handles,
            self.contribution_metadata.to_vec(),
            self.contribution_fanout.to_vec(),
        )
    }
}

fn required_direct_executor_mask(
    plan: &DirectRecurrencePlan,
    catalog_executor_count: usize,
) -> RusticolResult<Vec<bool>> {
    if plan.direct_executor_count() as usize != catalog_executor_count {
        return Err(RusticolError::integrity(format!(
            "prepared Direct-Arena catalog has {catalog_executor_count} executors, expected {}",
            plan.direct_executor_count()
        )));
    }
    let mut required = vec![false; catalog_executor_count];
    for descriptor in plan.row_groups() {
        if descriptor.direct_executor_id == DIRECT_NONE_U32 {
            continue;
        }
        let slot = required
            .get_mut(descriptor.direct_executor_id as usize)
            .ok_or_else(|| {
                RusticolError::integrity(format!(
                    "recurrence row group references absent prepared executor {}",
                    descriptor.direct_executor_id
                ))
            })?;
        *slot = true;
    }
    Ok(required)
}

fn instantiate_selected_executor<T>(
    required_executors: Option<&[bool]>,
    executor_id: u32,
    instantiate: impl FnOnce() -> RusticolResult<Option<T>>,
) -> RusticolResult<Option<T>> {
    let selected = required_executors.map_or(Ok(true), |required| {
        required.get(executor_id as usize).copied().ok_or_else(|| {
            RusticolError::integrity(format!(
                "prepared executor {executor_id} is outside the authenticated catalog"
            ))
        })
    })?;
    if selected { instantiate() } else { Ok(None) }
}

impl NativeOnTheFlyPreparedExecutorResolver {
    pub(super) fn clear_resolved_bindings(&mut self) {
        self.pending_resolved = None;
        self.resolved.clear();
    }

    pub(super) fn discard_pending_resolved_bindings(&mut self) {
        self.pending_resolved = None;
    }

    pub(super) fn commit_pending_resolved_bindings(&mut self) -> RusticolResult<()> {
        let pending = self.pending_resolved.take().ok_or_else(|| {
            RusticolError::internal("on-the-fly family retention has no pending semantic bindings")
        })?;
        self.resolved = pending;
        Ok(())
    }

    pub(super) fn bind_on_the_fly_trace(
        &mut self,
        templates: &ValidatedRecurrenceTemplateInput,
        direct: &PreparedDirectExecutorCatalog,
        seed: &OnTheFlyProcessSeedV1,
        trace: &mut OnTheFlyStructuralTraceV1,
    ) -> RusticolResult<()> {
        self.begin_on_the_fly_family_binding(templates, direct)?;
        self.extend_on_the_fly_family_binding(templates, direct, seed, std::iter::once(trace))
    }

    pub(super) fn bind_on_the_fly_family(
        &mut self,
        templates: &ValidatedRecurrenceTemplateInput,
        direct: &PreparedDirectExecutorCatalog,
        seed: &OnTheFlyProcessSeedV1,
        selected: &mut [OnTheFlySelectedQueryTraceV1],
    ) -> RusticolResult<()> {
        self.begin_on_the_fly_family_binding(templates, direct)?;
        self.extend_on_the_fly_family_binding(
            templates,
            direct,
            seed,
            selected.iter_mut().map(|selected| &mut selected.trace),
        )
    }

    /// Begin a streamed semantic-binding transaction without disturbing the
    /// last successfully committed family.
    pub(super) fn begin_on_the_fly_family_binding(
        &mut self,
        templates: &ValidatedRecurrenceTemplateInput,
        direct: &PreparedDirectExecutorCatalog,
    ) -> RusticolResult<()> {
        self.pending_resolved = None;
        let summary = templates.summary();
        if summary.catalog_digest != self.pool.recurrence_template_catalog_digest
            || summary.compiled_model_digest != self.pool.compiled_model_digest
            || summary.prepared_kernel_pack_digest != self.pool.prepared_kernel_pack_digest
        {
            return Err(RusticolError::integrity(
                "on-the-fly recurrence templates do not belong to the loaded prepared pack",
            ));
        }
        if direct.direct_template_catalog_digest() != self.pool.direct_template_catalog_digest {
            return Err(RusticolError::integrity(
                "on-the-fly direct-template catalog does not match the loaded prepared pack",
            ));
        }
        self.pending_resolved = Some(BTreeMap::new());
        Ok(())
    }

    /// Extend the current streamed transaction by one bounded trace chunk.
    /// Failure drops only the uncommitted map; the active map remains intact.
    pub(super) fn extend_on_the_fly_family_binding<'trace>(
        &mut self,
        templates: &ValidatedRecurrenceTemplateInput,
        direct: &PreparedDirectExecutorCatalog,
        seed: &OnTheFlyProcessSeedV1,
        traces: impl IntoIterator<Item = &'trace mut OnTheFlyStructuralTraceV1>,
    ) -> RusticolResult<()> {
        let mut resolved = self.pending_resolved.take().ok_or_else(|| {
            RusticolError::internal(
                "on-the-fly streamed binding extension has no pending transaction",
            )
        })?;
        let authenticate = authenticated_prepared_executor_binding(templates, direct)?;
        let mut trace_count = 0_u32;
        for trace in traces {
            trace_count = trace_count.checked_add(1).ok_or_else(|| {
                RusticolError::integrity("on-the-fly trace-family count exceeds u32")
            })?;
            if seed.template_catalog_digest() != self.pool.recurrence_template_catalog_digest
                || seed.model_digest() != self.pool.compiled_model_digest
                || seed.prepared_pack_digest() != self.pool.prepared_kernel_pack_digest
                || seed.direct_catalog_digest() != self.pool.direct_template_catalog_digest
                || trace.seed_digest() != seed.semantic_digest()
            {
                return Err(RusticolError::integrity(
                    "on-the-fly compact process identity does not match the loaded prepared pack",
                ));
            }
            let mut parent_permutations = BTreeMap::new();
            for key in trace.executor_keys() {
                let resolved_executor = if let Some(resolved) = resolved.get(&key).copied() {
                    resolved
                } else {
                    let expected = authenticate(key)?;
                    let (direct_executor_id, handle, parent_permutation) =
                        if let Some(evaluator_binding_id) = key.evaluator_binding_id() {
                            self.pool.resolve_handle(
                                &self.sources,
                                key.role(),
                                evaluator_binding_id,
                            )?
                        } else {
                            let (executor_id, handle) = self.pool.identity_finalizer_handle()?;
                            (executor_id, handle, [0, 1])
                        };
                    if direct_executor_id != expected.direct_executor_id
                        || parent_permutation != expected.parent_permutation
                    {
                        return Err(RusticolError::integrity(
                            "prepared pool and authenticated direct-template mapping disagree",
                        ));
                    }
                    let resolved_executor = ResolvedOnTheFlyExecutor {
                        direct_executor_id,
                        handle,
                        parent_permutation,
                    };
                    resolved.insert(key, resolved_executor);
                    resolved_executor
                };
                parent_permutations.insert(key, resolved_executor.parent_permutation);
            }
            trace.bind_prepared_executor_rows(&parent_permutations)?;
        }
        if trace_count == 0 {
            return Err(RusticolError::invalid_argument(
                "on-the-fly prepared executor family is empty",
            ));
        }
        self.pending_resolved = Some(resolved);
        Ok(())
    }
}

impl OnTheFlySourceDomainBinding {
    fn source_handle(&self) -> RusticolResult<DirectExecutorHandle> {
        let source = self.source.as_ref().ok_or_else(|| {
            RusticolError::integrity("on-the-fly source-domain binding is absent")
        })?;
        let handle = source.handle();
        Ok(DirectExecutorHandle::Source {
            call: handle.call,
            context: handle.context,
        })
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

impl OnTheFlyPreparedExecutorResolver for NativeOnTheFlyPreparedExecutorResolver {
    fn resolve(&self, key: OnTheFlyExecutorKeyV1) -> RusticolResult<ResolvedOnTheFlyExecutor> {
        self.effective_resolved().get(&key).copied().ok_or_else(|| {
            RusticolError::integrity(
                "on-the-fly operation has no exact authenticated prepared executor",
            )
        })
    }

    fn invalidate_row_tables(&self) -> RusticolResult<()> {
        self.pool.invalidate_on_the_fly_row_tables()
    }
}

impl NativeRecurrencePreparedExecutorPool {
    fn invalidate_on_the_fly_row_tables(&self) -> RusticolResult<()> {
        #[cfg(feature = "f64-symjit")]
        for executor in &self._symjit {
            executor.invalidate_row_tables()?;
        }
        Ok(())
    }
}

impl NativeRecurrenceDirectExecutorOwners {
    pub(super) fn summary(&self) -> &NativeRecurrenceDirectBackendSummary {
        &self.pool.summary
    }

    pub(super) fn internal_traffic_bytes(&self) -> (u64, u64) {
        #[cfg(feature = "f64-symjit")]
        {
            self.pool
                ._symjit
                .iter()
                .map(LoadedSymjitDirectExecutor::internal_traffic_bytes)
                .fold(
                    (0_u64, 0_u64),
                    |(scratch, broadcast), (next_scratch, next_broadcast)| {
                        (
                            scratch.saturating_add(next_scratch),
                            broadcast.saturating_add(next_broadcast),
                        )
                    },
                )
        }
        #[cfg(not(feature = "f64-symjit"))]
        {
            (0, 0)
        }
    }

    pub(super) fn union_source_dispatch(&self) -> RusticolResult<DirectUnionSourceDispatchHandle> {
        self.source_binding.union_source_dispatch()
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
        payloads.load_native_library(library_path)?,
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
) -> RusticolResult<DirectExecutorHandle> {
    let runtime_template = template
        .payload_binding
        .runtime_template
        .as_deref()
        .ok_or_else(|| RusticolError::artifact("Direct-Arena intrinsic has no runtime template"))?;
    match template.role.as_str() {
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
    pack: &PreparedKernelPackManifest,
    payloads: &EvaluatorPayloadStore,
) -> RusticolResult<LoadedSymjitDirectExecutor> {
    let binding = &template.payload_binding;
    let expected_compression = symjit_plane_compression(binding, pack)?;
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
    validate_direct_destination_policy(role, &template.destination_operation)?;
    let parameter_bindings = binding
        .parameter_bindings
        .iter()
        .map(|binding| match *binding {
            RecurrenceDirectParameterBindingManifest::Plane { index } => {
                SymjitDirectParameterBinding::Plane { index }
            }
            RecurrenceDirectParameterBindingManifest::Scalar { index } => {
                SymjitDirectParameterBinding::Broadcast { index }
            }
        })
        .collect::<Vec<_>>();
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
        template.optimization_level,
        expected_compression,
        role,
        parameter_bindings,
        input_planes,
        scalars,
        binding.output_alias_inputs.clone(),
    )
}

#[cfg(feature = "f64-symjit")]
fn symjit_plane_compression(
    binding: &super::eager_manifest::RecurrenceDirectPayloadBindingManifest,
    pack: &PreparedKernelPackManifest,
) -> RusticolResult<bool> {
    let prepared_kernel_id = binding
        .prepared_kernel_id
        .ok_or_else(|| RusticolError::artifact("Direct-Arena prepared call has no kernel ID"))?;
    let kernel = pack
        .kernels
        .iter()
        .find(|kernel| kernel.kernel_id == prepared_kernel_id)
        .ok_or_else(|| {
            RusticolError::integrity(format!(
                "Direct-Arena prepared kernel {prepared_kernel_id} is absent"
            ))
        })?;
    kernel
        .f64_evaluator_manifest
        .get("plane_application")
        .and_then(serde_json::Value::as_object)
        .and_then(|plane| plane.get("compression"))
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| {
            RusticolError::compatibility(format!(
                "Direct-Arena prepared JIT kernel {prepared_kernel_id} has no authenticated \
                 SymJIT compression setting; regenerate the prepared model"
            ))
        })
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
fn validate_direct_destination_policy(role: DirectExecutorRole, value: &str) -> RusticolResult<()> {
    let expected = match role {
        DirectExecutorRole::Source => "initialize",
        DirectExecutorRole::Contribution => "add",
        DirectExecutorRole::Finalization => "finalize-in-place",
        DirectExecutorRole::Closure => "closure-add",
    };
    if value == expected {
        Ok(())
    } else {
        Err(RusticolError::integrity(format!(
            "Direct-Arena destination operation {value:?} does not match role {role:?}; expected {expected:?}"
        )))
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

#[cfg(test)]
pub(in crate::engine) mod on_the_fly_adapter_tests {
    use super::*;
    use crate::engine::evaluator::recurrence_source_direct::{
        DirectSourceDispatchKey, DirectSourceDispatchVariantSpec, DirectSourceOrientation,
        DirectSourceTemplateSpec, DirectSourceWavefunctionFamily,
    };
    use crate::recurrence::on_the_fly::{
        OnTheFlyStructuralInterpreter, OnTheFlyWorkspaceV1, scalar_adapter_test_seed,
        scalar_adapter_test_trace,
    };
    use crate::recurrence::{
        PreparedDirectExecutorBinding, PreparedDirectExecutorCatalog, validated_template_fixture,
    };
    use std::cell::Cell;
    use std::ptr;

    #[test]
    fn unreferenced_symjit_executor_loader_is_not_invoked_by_plan_filter() {
        let required = [true, false, true];
        let calls = Cell::new(0_u32);
        let skipped = instantiate_selected_executor(Some(&required), 1, || {
            calls.set(calls.get() + 1);
            Ok(Some(7_u32))
        })
        .unwrap();
        assert_eq!(skipped, None);
        assert_eq!(calls.get(), 0);

        let loaded = instantiate_selected_executor(Some(&required), 2, || {
            calls.set(calls.get() + 1);
            Ok(Some(9_u32))
        })
        .unwrap();
        assert_eq!(loaded, Some(9));
        assert_eq!(calls.get(), 1);

        let full_catalog = instantiate_selected_executor(None, 99, || {
            calls.set(calls.get() + 1);
            Ok(Some(13_u32))
        })
        .unwrap();
        assert_eq!(full_catalog, Some(13));
        assert_eq!(calls.get(), 2);

        let propagated = instantiate_selected_executor(Some(&required), 0, || {
            Err::<Option<u32>, _>(RusticolError::artifact("selected payload is absent"))
        })
        .unwrap_err();
        assert_eq!(propagated.kind(), crate::RusticolErrorKind::Artifact);

        let error =
            instantiate_selected_executor(Some(&required), 3, || Ok(Some(11_u32))).unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
    }

    pub(in crate::engine) fn digest(seed: u8) -> SemanticDigest {
        SemanticDigest::new([seed; 32]).unwrap()
    }

    pub(in crate::engine) fn direct_catalog(
        direct_digest: SemanticDigest,
    ) -> PreparedDirectExecutorCatalog {
        PreparedDirectExecutorCatalog::new(
            direct_digest,
            vec![
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Source, 0, 0),
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Closure, 3, 1),
                PreparedDirectExecutorBinding::identity_finalizer(2),
            ],
        )
        .unwrap()
    }

    pub(in crate::engine) fn complete_scalar_direct_catalog(
        direct_digest: SemanticDigest,
    ) -> PreparedDirectExecutorCatalog {
        PreparedDirectExecutorCatalog::new(
            direct_digest,
            vec![
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Source, 0, 0),
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Closure, 3, 1),
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Closure, 4, 1),
                PreparedDirectExecutorBinding::identity_finalizer(2),
            ],
        )
        .unwrap()
    }

    pub(in crate::engine) fn prepared_pool(
        templates: &ValidatedRecurrenceTemplateInput,
        direct_digest: SemanticDigest,
    ) -> NativeRecurrencePreparedExecutorPool {
        let summary = templates.summary();
        NativeRecurrencePreparedExecutorPool {
            handles: vec![
                None,
                Some(DirectExecutorHandle::Closure {
                    call: execute_closure_reduce_rows,
                    context: ptr::null(),
                }),
                Some(DirectExecutorHandle::Finalization {
                    call: execute_identity_finalization_rows,
                    context: ptr::null(),
                }),
            ]
            .into_boxed_slice(),
            bindings: BTreeMap::from([
                (
                    (DirectExecutorRole::Source, 0),
                    NativePreparedExecutorBinding {
                        executor_id: 0,
                        parent_permutation: [0, 1],
                    },
                ),
                (
                    (DirectExecutorRole::Closure, 3),
                    NativePreparedExecutorBinding {
                        executor_id: 1,
                        parent_permutation: [0, 1],
                    },
                ),
            ]),
            executor_roles: vec![
                DirectExecutorRole::Source,
                DirectExecutorRole::Closure,
                DirectExecutorRole::Finalization,
            ]
            .into_boxed_slice(),
            contribution_metadata: vec![None, None, None].into_boxed_slice(),
            contribution_fanout: vec![None, None, None].into_boxed_slice(),
            identity_finalizer_id: Some(2),
            direct_template_catalog_digest: direct_digest,
            recurrence_template_catalog_digest: summary.catalog_digest,
            compiled_model_digest: summary.compiled_model_digest,
            prepared_kernel_pack_digest: summary.prepared_kernel_pack_digest,
            _intrinsics: Vec::new(),
            #[cfg(feature = "f64-symjit")]
            _symjit: Vec::new(),
            #[cfg(feature = "f64-compiled")]
            _native: Vec::new(),
            summary: NativeRecurrenceDirectBackendSummary {
                prepared_kernel_pack_digest: summary.prepared_kernel_pack_digest.to_string(),
                direct_template_catalog_digest: direct_digest.to_string(),
                backend: "test-intrinsic".into(),
                target_triple: "test".into(),
                target_portable: true,
                executor_count: 3,
            },
        }
    }

    pub(in crate::engine) fn complete_scalar_prepared_pool(
        templates: &ValidatedRecurrenceTemplateInput,
        direct_digest: SemanticDigest,
    ) -> NativeRecurrencePreparedExecutorPool {
        let mut pool = prepared_pool(templates, direct_digest);
        pool.bindings.insert(
            (DirectExecutorRole::Closure, 4),
            NativePreparedExecutorBinding {
                executor_id: 1,
                parent_permutation: [0, 1],
            },
        );
        pool
    }

    pub(in crate::engine) fn source_domains() -> Vec<DirectSourceDispatchDomainSpec> {
        vec![DirectSourceDispatchDomainSpec {
            variants: vec![DirectSourceDispatchVariantSpec {
                key: DirectSourceDispatchKey::SpinStateClass(50_000),
                template: DirectSourceTemplateSpec {
                    spin_state_class: 50_000,
                    family: DirectSourceWavefunctionFamily::Scalar,
                    orientation: DirectSourceOrientation::SelfConjugate,
                    helicity: 0,
                    chirality: 0,
                    mass_parameter_index: None,
                },
            }],
        }]
    }

    #[test]
    fn genuine_scalar_sources_nonunit_identity_and_closure_execute_to_two() {
        let templates = validated_template_fixture();
        let summary = templates.summary();
        let direct_digest = digest(40);
        let direct = direct_catalog(direct_digest);
        let seed = scalar_adapter_test_seed(
            summary.compiled_model_digest,
            summary.catalog_digest,
            summary.prepared_kernel_pack_digest,
            direct_digest,
        )
        .unwrap();
        let mut trace = scalar_adapter_test_trace(&templates, &seed).unwrap();
        trace.test_insert_identity_finalizer(direct_digest);
        let pool = prepared_pool(&templates, direct_digest);
        let sources = pool.bind_source_domains(source_domains()).unwrap();
        let mut resolver = pool.into_on_the_fly_resolver(sources);
        resolver
            .bind_on_the_fly_trace(&templates, &direct, &seed, &mut trace)
            .unwrap();
        let work = trace.execution_work_census().unwrap();
        assert_eq!(work.logical_current_count, 2);
        assert_eq!(work.resident_current_count, 2);
        assert_eq!(work.resident_current_component_count, 2);
        assert_eq!(work.source_operation_count, 2);
        assert_eq!(work.contribution_operation_count, 0);
        assert_eq!(work.finalization_operation_count, 1);
        assert_eq!(work.closure_operation_count, 1);
        assert_eq!(work.total_kernel_application_count, 4);
        assert_eq!(resolver.semantic_executor_binding_count().unwrap(), 3);
        assert_eq!(resolver.distinct_prepared_executor_count().unwrap(), 3);
        let mut workspace = OnTheFlyWorkspaceV1::new(&trace, 1).unwrap();
        workspace
            .fill_momenta_from_external(&trace, &[0.0; 8], 1)
            .unwrap();
        OnTheFlyStructuralInterpreter::execute(&trace, &resolver, &mut workspace, 1).unwrap();
        assert_eq!(workspace.amplitude(0).unwrap(), (2.0, 0.0));
    }

    #[test]
    fn adapter_rejects_tampered_operation_and_identity_keys() {
        let templates = validated_template_fixture();
        let summary = templates.summary();
        let direct_digest = digest(40);
        let direct = direct_catalog(direct_digest);
        let seed = scalar_adapter_test_seed(
            summary.compiled_model_digest,
            summary.catalog_digest,
            summary.prepared_kernel_pack_digest,
            direct_digest,
        )
        .unwrap();
        let pool = prepared_pool(&templates, direct_digest);
        let sources = pool.bind_source_domains(source_domains()).unwrap();
        let mut resolver = pool.into_on_the_fly_resolver(sources);

        let mut operation_tamper = scalar_adapter_test_trace(&templates, &seed).unwrap();
        operation_tamper.test_tamper_first_operation_semantic_digest(digest(99));
        assert!(
            resolver
                .bind_on_the_fly_trace(&templates, &direct, &seed, &mut operation_tamper)
                .is_err()
        );

        let mut identity_tamper = scalar_adapter_test_trace(&templates, &seed).unwrap();
        identity_tamper.test_insert_identity_finalizer(digest(99));
        assert!(
            resolver
                .bind_on_the_fly_trace(&templates, &direct, &seed, &mut identity_tamper)
                .is_err()
        );
        assert_eq!(resolver.semantic_executor_binding_count().unwrap(), 0);

        let mut retained = scalar_adapter_test_trace(&templates, &seed).unwrap();
        resolver
            .bind_on_the_fly_trace(&templates, &direct, &seed, &mut retained)
            .unwrap();
        assert_eq!(resolver.semantic_executor_binding_count().unwrap(), 2);
    }

    #[test]
    fn semantic_binding_candidates_replace_exactly_only_after_commit() {
        let templates = validated_template_fixture();
        let summary = templates.summary();
        let direct_digest = digest(40);
        let direct = direct_catalog(direct_digest);
        let seed = scalar_adapter_test_seed(
            summary.compiled_model_digest,
            summary.catalog_digest,
            summary.prepared_kernel_pack_digest,
            direct_digest,
        )
        .unwrap();
        let pool = prepared_pool(&templates, direct_digest);
        let sources = pool.bind_source_domains(source_domains()).unwrap();
        let mut resolver = pool.into_on_the_fly_resolver(sources);

        let mut first = scalar_adapter_test_trace(&templates, &seed).unwrap();
        resolver
            .bind_on_the_fly_trace(&templates, &direct, &seed, &mut first)
            .unwrap();
        assert_eq!(resolver.semantic_executor_binding_count().unwrap(), 2);
        resolver.commit_pending_resolved_bindings().unwrap();

        let mut second = scalar_adapter_test_trace(&templates, &seed).unwrap();
        second.test_insert_identity_finalizer(direct_digest);
        resolver
            .bind_on_the_fly_trace(&templates, &direct, &seed, &mut second)
            .unwrap();
        assert_eq!(resolver.semantic_executor_binding_count().unwrap(), 3);
        resolver.discard_pending_resolved_bindings();
        assert_eq!(resolver.semantic_executor_binding_count().unwrap(), 2);

        let mut malformed = scalar_adapter_test_trace(&templates, &seed).unwrap();
        malformed.test_insert_identity_finalizer(digest(99));
        assert!(
            resolver
                .bind_on_the_fly_trace(&templates, &direct, &seed, &mut malformed)
                .is_err()
        );
        assert_eq!(resolver.semantic_executor_binding_count().unwrap(), 2);

        let mut committed_second = scalar_adapter_test_trace(&templates, &seed).unwrap();
        committed_second.test_insert_identity_finalizer(direct_digest);
        resolver
            .bind_on_the_fly_trace(&templates, &direct, &seed, &mut committed_second)
            .unwrap();
        resolver.commit_pending_resolved_bindings().unwrap();
        assert_eq!(resolver.semantic_executor_binding_count().unwrap(), 3);

        let mut revisited_first = scalar_adapter_test_trace(&templates, &seed).unwrap();
        resolver
            .bind_on_the_fly_trace(&templates, &direct, &seed, &mut revisited_first)
            .unwrap();
        resolver.commit_pending_resolved_bindings().unwrap();
        assert_eq!(resolver.semantic_executor_binding_count().unwrap(), 2);
    }

    #[test]
    fn streamed_semantic_binding_failure_rolls_back_every_prior_chunk() {
        let templates = validated_template_fixture();
        let summary = templates.summary();
        let direct_digest = digest(40);
        let direct = direct_catalog(direct_digest);
        let seed = scalar_adapter_test_seed(
            summary.compiled_model_digest,
            summary.catalog_digest,
            summary.prepared_kernel_pack_digest,
            direct_digest,
        )
        .unwrap();
        let pool = prepared_pool(&templates, direct_digest);
        let sources = pool.bind_source_domains(source_domains()).unwrap();
        let mut resolver = pool.into_on_the_fly_resolver(sources);

        let mut committed = scalar_adapter_test_trace(&templates, &seed).unwrap();
        resolver
            .bind_on_the_fly_trace(&templates, &direct, &seed, &mut committed)
            .unwrap();
        resolver.commit_pending_resolved_bindings().unwrap();
        assert_eq!(resolver.semantic_executor_binding_count().unwrap(), 2);

        let mut first_chunk = scalar_adapter_test_trace(&templates, &seed).unwrap();
        first_chunk.test_insert_identity_finalizer(direct_digest);
        resolver
            .begin_on_the_fly_family_binding(&templates, &direct)
            .unwrap();
        resolver
            .extend_on_the_fly_family_binding(
                &templates,
                &direct,
                &seed,
                std::iter::once(&mut first_chunk),
            )
            .unwrap();
        assert_eq!(resolver.semantic_executor_binding_count().unwrap(), 3);

        let mut malformed_second_chunk = scalar_adapter_test_trace(&templates, &seed).unwrap();
        malformed_second_chunk.test_insert_identity_finalizer(digest(99));
        let error = resolver
            .extend_on_the_fly_family_binding(
                &templates,
                &direct,
                &seed,
                std::iter::once(&mut malformed_second_chunk),
            )
            .unwrap_err();
        assert_eq!(error.kind(), crate::RusticolErrorKind::Integrity);
        assert_eq!(resolver.semantic_executor_binding_count().unwrap(), 2);
        assert!(resolver.commit_pending_resolved_bindings().is_err());
        assert_eq!(resolver.semantic_executor_binding_count().unwrap(), 2);
    }

    #[test]
    fn adapter_rejects_missing_binding_and_each_compact_catalog_identity_mismatch() {
        let templates = validated_template_fixture();
        let summary = templates.summary();
        let direct_digest = digest(40);
        let direct = direct_catalog(direct_digest);
        let pool = prepared_pool(&templates, direct_digest);
        let sources = pool.bind_source_domains(source_domains()).unwrap();
        let mut resolver = pool.into_on_the_fly_resolver(sources);
        let seed = scalar_adapter_test_seed(
            summary.compiled_model_digest,
            summary.catalog_digest,
            summary.prepared_kernel_pack_digest,
            direct_digest,
        )
        .unwrap();
        let mut trace = scalar_adapter_test_trace(&templates, &seed).unwrap();
        let missing = PreparedDirectExecutorCatalog::new(
            direct_digest,
            vec![
                PreparedDirectExecutorBinding::evaluator(DirectExecutorRole::Source, 0, 0),
                PreparedDirectExecutorBinding::identity_finalizer(1),
            ],
        )
        .unwrap();
        assert!(
            resolver
                .bind_on_the_fly_trace(&templates, &missing, &seed, &mut trace)
                .is_err()
        );

        for (model, template, pack, direct_id) in [
            (
                digest(90),
                summary.catalog_digest,
                summary.prepared_kernel_pack_digest,
                direct_digest,
            ),
            (
                summary.compiled_model_digest,
                digest(90),
                summary.prepared_kernel_pack_digest,
                direct_digest,
            ),
            (
                summary.compiled_model_digest,
                summary.catalog_digest,
                digest(90),
                direct_digest,
            ),
            (
                summary.compiled_model_digest,
                summary.catalog_digest,
                summary.prepared_kernel_pack_digest,
                digest(90),
            ),
        ] {
            let mismatched = scalar_adapter_test_seed(model, template, pack, direct_id).unwrap();
            let mut mismatched_trace = scalar_adapter_test_trace(&templates, &mismatched).unwrap();
            assert!(
                resolver
                    .bind_on_the_fly_trace(&templates, &direct, &mismatched, &mut mismatched_trace,)
                    .is_err()
            );
        }
    }
}
