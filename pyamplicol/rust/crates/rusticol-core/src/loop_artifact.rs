// SPDX-License-Identifier: 0BSD

//! Checked virtual-process lane in the ordinary schema-v3 artifact envelope.
//! Scheme records are explicit declarations; loading does not certify physics.

use crate::loop_runtime::{
    LoadedLoopRuntime, ModelBinding, OrderLayout, PROJECTED_FAMILY_CONVENTION,
    PROJECTED_IDENTITY_CONVENTION, StateLeg, load_loop_runtime_reader,
};
use crate::{ArtifactSelection, PayloadRole, RusticolError, RusticolResult, VerifiedArtifact};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

pub const LOOP_RUNTIME_CAPABILITY: &str = "rusticol.loop-runtime.v3";
pub const LOOP_RUNTIME_KIND: &str = "pyamplicol-runtime-loop-execution";
pub const EPSILON_CONVENTION: &str = "Cepsilon=(4*pi)^epsilon/Gamma(1-epsilon)";
pub const ORDINARY_INTERFERENCE: &str = "2*Re(conj(Born)*loop)";
pub const DIRECTED_INTERFERENCE: &str = "Bdagger K_BL L + Ldagger K_LB B";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoopSchemeDeclaration {
    pub dimensional_regularization: String,
    pub external_states: String,
    pub gamma5: String,
    pub renormalization: String,
    pub mass_scheme: String,
    pub finite_restoration: String,
    pub model_convention: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoopExecutionMetadata {
    pub scheme: LoopSchemeDeclaration,
    pub orders: OrderLayout,
    pub r2_strategy: String,
    pub available_r2_strategies: Vec<String>,
    pub arithmetic: Vec<String>,
    pub source_owned: bool,
    pub contribution_kind: String,
    pub execution_mode: String,
    pub epsilon_powers: Vec<i32>,
    pub epsilon_convention: String,
    pub dimension: String,
    pub interference: String,
    pub color_accuracy: String,
    pub scheme_status: String,
    pub correlations: bool,
    pub resolved: bool,
    pub partial_results: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Execution {
    schema_version: u32,
    kind: String,
    process_id: String,
    contribution_kind: String,
    execution_mode: String,
    required_runtime_capabilities: Vec<String>,
    runtime_path: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Physics {
    schema_version: u32,
    kind: String,
    process_id: String,
    metadata: LoopExecutionMetadata,
    initial_average: u64,
    state_legs: Vec<StateLeg>,
    helicities: Vec<Vec<i32>>,
}

/// All execution bytes stay owned by the validated PACBIN reader. No producer
/// model, graph, filesystem source import or tree numerical lane is retained.
pub struct LoadedLoopArtifact {
    pub artifact_id: String,
    pub selection: ArtifactSelection,
    pub metadata: LoopExecutionMetadata,
    pub runtime: Arc<LoadedLoopRuntime>,
}

fn require(condition: bool, message: &str) -> RusticolResult<()> {
    if condition {
        Ok(())
    } else {
        Err(RusticolError::compatibility(format!(
            "loop artifact: {message}"
        )))
    }
}
fn same<T: Serialize, U: Serialize>(left: &T, right: &U) -> RusticolResult<bool> {
    Ok(
        serde_json::to_value(left).map_err(|e| RusticolError::serialization(e.to_string()))?
            == serde_json::to_value(right)
                .map_err(|e| RusticolError::serialization(e.to_string()))?,
    )
}
fn parse<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> RusticolResult<T> {
    serde_json::from_slice(bytes)
        .map_err(|e| RusticolError::serialization(format!("loop artifact: {e}")))
}

pub fn open_loop_artifact(
    path: impl AsRef<Path>,
    requested_process: Option<&str>,
) -> RusticolResult<LoadedLoopArtifact> {
    load_loop_artifact(&VerifiedArtifact::open(path)?, requested_process)
}

pub fn load_loop_artifact(
    artifact: &VerifiedArtifact,
    requested_process: Option<&str>,
) -> RusticolResult<LoadedLoopArtifact> {
    require(
        artifact.manifest().runtime.required_runtime_capabilities == [LOOP_RUNTIME_CAPABILITY]
            && artifact
                .manifest()
                .processes
                .iter()
                .all(|p| p.required_runtime_capabilities == [LOOP_RUNTIME_CAPABILITY]),
        "tree and virtual processes cannot share an artifact yet",
    )?;
    let selection = artifact.select_process(requested_process)?;
    let entries = crate::engine::verified_execution_payloads(artifact)?;
    let (_, path, bytes) = entries
        .into_iter()
        .find(|(p, _, _)| p.id == selection.process.id)
        .ok_or_else(|| RusticolError::integrity("execution set omits selected virtual process"))?;
    let bytes = match bytes {
        Some(bytes) => bytes,
        None => artifact.read_payload(&path)?,
    };
    let execution: Execution = parse(&bytes)?;
    require(
        execution.schema_version == 3
            && execution.kind == LOOP_RUNTIME_KIND
            && execution.process_id == selection.process.id
            && execution.contribution_kind == "virtual"
            && execution.execution_mode == "loop_recurrence",
        "unsupported virtual execution declaration",
    )?;
    require(
        execution.required_runtime_capabilities == [LOOP_RUNTIME_CAPABILITY]
            && selection.process.required_runtime_capabilities
                == execution.required_runtime_capabilities,
        "execution and process runtime capabilities disagree",
    )?;
    require(
        selection.process.aliases.is_empty(),
        "only independent virtual processes are currently admitted",
    )?;
    let physics_path = &selection.process.physics_path;
    let physics_payload = artifact.payload(physics_path)?;
    require(
        physics_payload.role == PayloadRole::RuntimePhysics
            && physics_payload.process_id.as_deref() == Some(selection.process.id.as_str()),
        "physics payload ownership disagrees with selected process",
    )?;
    let physics: Physics = parse(&artifact.read_payload(physics_path)?)?;
    require(
        physics.schema_version == 1
            && physics.kind == "pyamplicol-loop-physics"
            && physics.process_id == selection.process.id,
        "invalid virtual physics metadata",
    )?;
    let root = artifact
        .root()
        .join(Path::new(&path).parent().unwrap_or(Path::new("")));
    let store = artifact.evaluator_payload_store(&root)?;
    let logical_path = store.logical_path(&execution.runtime_path)?;
    let runtime_payload = artifact.payload(&logical_path)?;
    require(
        runtime_payload.role == PayloadRole::EvaluatorState
            && runtime_payload.process_id.as_deref() == Some(selection.process.id.as_str()),
        "runtime payload ownership disagrees with selected process",
    )?;
    let runtime = Arc::new(load_loop_runtime_reader(
        store.source(&execution.runtime_path)?.pacbin()?,
    )?);
    let plan = runtime.plan();
    if selection.remaps_public_process() {
        let relation = selection.process_group.as_ref();
        let member = plan.admitted_flavor_members.as_ref().and_then(|members| {
            members
                .iter()
                .find(|member| member.process_id == selection.requested_id)
        });
        require(
            selection.alias.is_none()
                && selection
                    .external_permutation
                    .iter()
                    .copied()
                    .eq(0..plan.state_legs.len())
                && relation.is_some_and(|r| {
                    !r.conjugated && r.factor_numerator == 1 && r.factor_denominator == 1
                })
                && member.is_some_and(|m| {
                    m.expression == selection.public_expression
                        && m.external_pdgs == selection.external_pdgs
                }),
            "selected loop group member lacks matching immutable flavour admission",
        )?;
    }
    let metadata = &physics.metadata;
    require(
        selection.process.color_accuracy == metadata.color_accuracy
            && metadata.color_accuracy
                == plan
                    .color_projection
                    .as_ref()
                    .map_or("full", |p| p.accuracy.as_str()),
        "declared colour accuracy differs from the saved projection",
    )?;
    require(
        metadata.contribution_kind == "virtual"
            && metadata.execution_mode == "loop_recurrence"
            && metadata.epsilon_powers == [0, -1, -2]
            && metadata.epsilon_convention == EPSILON_CONVENTION
            && metadata.dimension == "D=4-2*epsilon"
            && metadata.interference
                == if metadata.correlations {
                    DIRECTED_INTERFERENCE
                } else {
                    ORDINARY_INTERFERENCE
                }
            && matches!(metadata.color_accuracy.as_str(), "full" | "nlc" | "lc")
            && (metadata.color_accuracy != "lc"
                || (metadata.resolved && plan.born_color_components.is_some())
                || (metadata.correlations
                    && !metadata.resolved
                    && plan.born_color_components.is_none()
                    && plan.directed_projection.as_ref().is_some_and(|p| {
                        matches!(
                            p.convention.as_str(),
                            PROJECTED_IDENTITY_CONVENTION | PROJECTED_FAMILY_CONVENTION
                        )
                    })))
            && metadata.scheme_status == "declared-not-certified"
            && metadata.correlations == plan.directed_projection.is_some()
            && metadata.resolved == plan.helicity_projection.is_some()
            && !metadata.partial_results,
        "unsupported contribution, convention or feature declaration",
    )?;
    require(
        metadata.arithmetic == ["double", "double-double", "arbitrary"],
        "unsupported arithmetic declaration",
    )?;
    // Public source artifacts start with the complete effective-rule strategy.
    // The internal plan interface remains available for bounded direct controls.
    require(
        metadata.source_owned
            && matches!(plan.model_binding, ModelBinding::Source { .. })
            && metadata.r2_strategy == "model"
            && metadata.available_r2_strategies == ["model"]
            && plan.r2_strategy == "model"
            && !plan.dimensional_numerators,
        "public virtual artifacts require source-owned model R2 and UV",
    )?;
    let scheme = &metadata.scheme;
    require(
        [
            &scheme.dimensional_regularization,
            &scheme.external_states,
            &scheme.gamma5,
            &scheme.renormalization,
            &scheme.mass_scheme,
            &scheme.finite_restoration,
            &scheme.model_convention,
        ]
        .iter()
        .all(|x| !x.trim().is_empty()),
        "scheme declarations must be explicit and nonempty",
    )?;
    require(
        same(&metadata.orders, &plan.order_layout)?
            && selection.process.external_pdgs == plan.external_pdgs
            && physics.initial_average == plan.initial_average
            && physics.helicities == plan.helicities
            && same(&physics.state_legs, &plan.state_legs)?,
        "public order, state or normalization metadata disagrees with execution plan",
    )?;
    Ok(LoadedLoopArtifact {
        artifact_id: artifact.manifest().artifact_id.clone(),
        selection,
        metadata: physics.metadata,
        runtime,
    })
}
