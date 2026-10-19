// SPDX-License-Identifier: 0BSD

//! Additive native virtual handle over the verified process-artifact boundary.
//! The shared core owns complete execution; this consumer supplies its existing
//! NinjaRS backend. No Python interpreter or generation code is involved.

use crate::runtime::{Batch, physical::PhysicalBackend};
use rusticol_core::loop_artifact::{LoadedLoopArtifact, LoopExecutionMetadata, open_loop_artifact};
use rusticol_core::loop_execution::{CompleteLoopResult, LoopExecutor, LoopInputs};
use rusticol_core::source_bindings::{SourceOverrides, SourceOwner};
use rusticol_core::{RusticolError, RusticolResult};
use std::{path::Path, sync::Arc};

pub struct VirtualRuntime {
    artifact: Arc<LoadedLoopArtifact>,
    backend: PhysicalBackend,
    owner: Arc<SourceOwner<PhysicalBackend>>,
    executor: Option<LoopExecutor<PhysicalBackend>>,
    arithmetic: String,
    precision: Option<u32>,
    requested_strategy: String,
}

fn strategy(requested: &str) -> RusticolResult<()> {
    match requested {
        "auto" | "model" => Ok(()),
        "direct" => Err(RusticolError::unsupported_runtime_capability(
            "direct R2",
            "this artifact contains only the complete model-rule R2 branch",
        )),
        _ => Err(RusticolError::invalid_argument(
            "r2_strategy must be auto, model, or direct",
        )),
    }
}

impl VirtualRuntime {
    pub fn open(
        path: impl AsRef<Path>,
        process: Option<&str>,
        arithmetic: &str,
        precision: Option<u32>,
        parameters: Option<&SourceOverrides>,
        r2_strategy: &str,
    ) -> RusticolResult<Self> {
        crate::runtime::register_scalar_primitives();
        Self::bind(
            Arc::new(open_loop_artifact(path, process)?),
            arithmetic,
            precision,
            parameters,
            r2_strategy,
        )
    }

    fn bind(
        artifact: Arc<LoadedLoopArtifact>,
        arithmetic: &str,
        precision: Option<u32>,
        parameters: Option<&SourceOverrides>,
        r2_strategy: &str,
    ) -> RusticolResult<Self> {
        strategy(r2_strategy)?;
        let engine = if arithmetic == "double" {
            "fast"
        } else {
            "generic"
        };
        let backend = PhysicalBackend::new(arithmetic, precision, "interpreted", engine)
            .map_err(|e| RusticolError::invalid_argument(e.to_string()))?;
        let executor =
            LoopExecutor::with_parameters(artifact.runtime.clone(), backend.clone(), parameters)?;
        let owner = executor.source_owner().cloned().ok_or_else(|| {
            RusticolError::compatibility("public virtual handles require source-owned parameters")
        })?;
        Ok(Self {
            artifact,
            backend,
            owner,
            executor: Some(executor),
            arithmetic: arithmetic.into(),
            precision,
            requested_strategy: r2_strategy.into(),
        })
    }

    pub fn metadata(&self) -> &LoopExecutionMetadata {
        &self.artifact.metadata
    }
    pub fn metadata_json(&self) -> RusticolResult<String> {
        serde_json::to_string(&serde_json::json!({
            "artifact_id": self.artifact.artifact_id,
            "process_id": self.process_id(),
            "expression": self.artifact.selection.public_expression,
            "external_pdgs": self.artifact.selection.external_pdgs,
            "initial_average": self.artifact.runtime.plan().initial_average,
            "physics": self.metadata(),
            "plan": self.artifact.runtime.plan(),
        }))
        .map_err(|e| RusticolError::serialization(e.to_string()))
    }
    pub fn plan_metadata_json(&self) -> RusticolResult<String> {
        self.artifact.runtime.metadata_json()
    }
    pub fn arithmetic(&self) -> &str {
        &self.arithmetic
    }
    pub fn precision(&self) -> Option<u32> {
        self.precision
    }
    pub fn external_count(&self) -> usize {
        self.artifact.runtime.plan().state_legs.len()
    }
    pub fn process_id(&self) -> &str {
        &self.artifact.selection.process.id
    }
    pub fn precision_bits(&self) -> u32 {
        self.backend.precision_bits()
    }
    pub fn requested_r2_strategy(&self) -> &str {
        &self.requested_strategy
    }
    pub fn source_revision(&self) -> RusticolResult<u64> {
        Ok(self.owner.snapshot()?.revision())
    }
    pub fn parameter_overrides(&self) -> RusticolResult<SourceOverrides> {
        Ok(self.owner.snapshot()?.overrides().clone())
    }
    pub fn source_owner(&self) -> &Arc<SourceOwner<PhysicalBackend>> {
        &self.owner
    }
    pub fn backend(&self) -> &PhysicalBackend {
        &self.backend
    }
    pub fn update_parameters(&self, changes: &SourceOverrides) -> RusticolResult<u64> {
        Ok(self.owner.update(&self.backend, changes)?.revision())
    }

    /// Clear process numerical workspaces; exact parameter state and existing
    /// result buffers remain valid. The next call binds fresh native workspaces.
    pub fn clear(&mut self) {
        self.executor = None;
    }

    pub fn rebind(
        &self,
        arithmetic: &str,
        precision: Option<u32>,
        parameters: Option<&SourceOverrides>,
        r2_strategy: Option<&str>,
    ) -> RusticolResult<Self> {
        let mut exact = self.parameter_overrides()?;
        if let Some(changes) = parameters {
            exact.extend(changes.clone());
        }
        Self::bind(
            self.artifact.clone(),
            arithmetic,
            precision,
            Some(&exact),
            r2_strategy.unwrap_or(&self.requested_strategy),
        )
    }

    pub fn evaluate(
        &mut self,
        inputs: LoopInputs,
        r2_strategy: Option<&str>,
    ) -> RusticolResult<CompleteLoopResult<Arc<Batch>>> {
        strategy(r2_strategy.unwrap_or(&self.requested_strategy))?;
        if !inputs.model_values.is_empty() {
            return Err(RusticolError::invalid_argument(
                "source model parameters require named atomic updates",
            ));
        }
        if self.executor.is_none() {
            self.executor = Some(LoopExecutor::with_source_owner(
                self.owner.clone(),
                self.backend.clone(),
            )?);
        }
        self.executor
            .as_mut()
            .expect("workspaces bound")
            .evaluate(inputs)
    }
}
