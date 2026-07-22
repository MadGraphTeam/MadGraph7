// SPDX-License-Identifier: 0BSD

use super::{PreparedEvaluatorBackend, PreparedKernelPackManifest};
use crate::artifact::EvaluatorPayloadStore;
use crate::{
    EagerKernelBackend, EagerKernelCall, EagerKernelSpec, RusticolError, RusticolResult,
    VerifiedArtifact,
};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Authenticated identity of the prepared kernel pack used by recurrence execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeRecurrenceKernelBackendSummary {
    pub manifest_sha256: String,
    pub backend: String,
    pub target_triple: String,
    pub target_cpu_features: Vec<String>,
    pub target_portable: bool,
    pub kernel_count: usize,
}

/// Prepared-kernel evaluator lane used by the native recurrence runtime.
///
/// Construction validates the prepared-pack manifest before loading evaluators.
/// Recurrence execution therefore shares the eager runtime's evaluator loader,
/// capability checks, target checks, and packet ABI rather than maintaining a
/// second backend implementation.
pub struct NativeRecurrenceKernelBackend {
    backend: PreparedEvaluatorBackend,
    kernel_specs: Vec<EagerKernelSpec>,
    summary: NativeRecurrenceKernelBackendSummary,
}

impl NativeRecurrenceKernelBackend {
    /// Load a prepared pack whose evaluator payloads live below `payload_root`.
    pub fn load(manifest_json: &[u8], payload_root: impl AsRef<Path>) -> RusticolResult<Self> {
        let payloads = EvaluatorPayloadStore::directory(payload_root.as_ref());
        Self::load_from_store(manifest_json, &payloads)
    }

    /// Load a prepared pack from a verified artifact's loose or PACBIN payload store.
    pub fn load_from_verified_artifact(
        manifest_json: &[u8],
        artifact: &VerifiedArtifact,
        payload_root: impl AsRef<Path>,
    ) -> RusticolResult<Self> {
        let payloads = artifact.evaluator_payload_store(payload_root.as_ref())?;
        Self::load_from_store(manifest_json, &payloads)
    }

    fn load_from_store(
        manifest_json: &[u8],
        payloads: &EvaluatorPayloadStore,
    ) -> RusticolResult<Self> {
        let manifest: PreparedKernelPackManifest =
            serde_json::from_slice(manifest_json).map_err(|error| {
                RusticolError::serialization(format!(
                    "could not parse prepared recurrence kernel pack: {error}"
                ))
            })?;
        manifest.validate()?;
        let kernel_specs = manifest.kernel_specs()?;
        let backend = PreparedEvaluatorBackend::load_from_store(&manifest, payloads)?;
        let summary = NativeRecurrenceKernelBackendSummary {
            manifest_sha256: format!("{:x}", Sha256::digest(manifest_json)),
            backend: manifest.backend.clone(),
            target_triple: manifest.target.target_triple.clone(),
            target_cpu_features: manifest.target.cpu_features.clone(),
            target_portable: manifest.target.portable,
            kernel_count: kernel_specs.len(),
        };
        Ok(Self {
            backend,
            kernel_specs,
            summary,
        })
    }

    /// Return recurrence-compatible kernel contracts detached from manifest storage.
    pub fn kernel_specs(&self) -> Vec<EagerKernelSpec> {
        self.kernel_specs.clone()
    }

    pub fn summary(&self) -> &NativeRecurrenceKernelBackendSummary {
        &self.summary
    }
}

impl EagerKernelBackend for NativeRecurrenceKernelBackend {
    fn evaluate_batch(&mut self, call: EagerKernelCall<'_>) -> RusticolResult<()> {
        self.backend.evaluate_batch(call)
    }
}
