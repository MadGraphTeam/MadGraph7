// SPDX-License-Identifier: 0BSD

//! Checked virtual-process lane in the ordinary schema-v3 artifact envelope.
//! Scheme records are explicit declarations; loading does not certify physics.
//!
//! Two runtime capabilities share the envelope: the legacy loop runtime
//! (`rusticol.loop-runtime.v3`, [`load_loop_artifact`]) and the shared-recursion
//! engine v2 (`rusticol.loop-runtime.v4`, [`load_loop_v2_artifact`], DESIGN.md
//! §12.1), whose PACBIN holds one plan v4 member.

use crate::loop_runtime::{
    LoadedLoopRuntime, ModelBinding, OrderLayout, PROJECTED_FAMILY_CONVENTION,
    PROJECTED_IDENTITY_CONVENTION, StateLeg, load_loop_runtime_reader,
};
use crate::{ArtifactSelection, PayloadRole, RusticolError, RusticolResult, VerifiedArtifact};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

pub const LOOP_RUNTIME_CAPABILITY: &str = "rusticol.loop-runtime.v3";
pub const LOOP_V2_RUNTIME_CAPABILITY: &str = "rusticol.loop-runtime.v4";
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
        !artifact
            .manifest()
            .runtime
            .required_runtime_capabilities
            .iter()
            .any(|capability| capability == LOOP_V2_RUNTIME_CAPABILITY),
        "plan v4 processes (rusticol.loop-runtime.v4) are admitted by \
         load_loop_v2_artifact; this runtime evaluates them once trees, Born \
         amplitudes and counterterm scalars run natively (WP3.1)",
    )?;
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
                && relation.is_some_and(|r| {
                    !r.conjugated && r.factor_numerator == 1 && r.factor_denominator == 1
                })
                && member.is_some_and(|m| {
                    m.expression == selection.public_expression
                        && m.external_pdgs == selection.external_pdgs
                        && m.permutation
                            .clone()
                            .unwrap_or_else(|| (0..plan.state_legs.len()).collect::<Vec<_>>())
                            == selection.external_permutation
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

// ------------------------------------------------------------------ plan v4

/// Coupling orders as `(name, power)` pairs in canonical order.
pub type CouplingOrders = Vec<(String, u32)>;

/// The (Born, loop) amplitude orders of one squared-order sector.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoopV2BornLoopPair {
    pub born: CouplingOrders,
    #[serde(rename = "loop")]
    pub loop_orders: CouplingOrders,
}

/// One squared coupling-order sector of the virtual: a separate partial result.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoopV2Sector {
    pub order: CouplingOrders,
    pub pairs: Vec<LoopV2BornLoopPair>,
}

/// Declarations of one plan v4 process (`generation/loop_artifact.py`
/// `LoopV2ExecutionMetadata`, DESIGN.md §12.1). `buckets` lists
/// `(propagators, rank, families)` per NinjaRS reduction bucket and
/// `color_degrees` the leading N_c degree kept per sector at LC/NLC.
/// `color_correlations` are the declared directed colour histories (by id,
/// besides the Born metric "born"), `spin_correlations` the declared spin
/// classes (one-based legs) and `resolved` that helicity/colour resolved
/// output is served (always: observable columns are contracted directly, and
/// `color_contraction` is "auto" or "direct" since generation refuses the
/// forced seed FFT). `poles` lists the pole selections an evaluation may
/// request (`PoleSelection`): every pole ("all") or the finite part only
/// ("none"); the UV and IR selections are added with the pole split.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoopV2ExecutionMetadata {
    pub scheme: LoopSchemeDeclaration,
    pub born_orders: Vec<CouplingOrders>,
    pub sectors: Vec<LoopV2Sector>,
    pub families: usize,
    pub buckets: Vec<(usize, usize, usize)>,
    pub color_accuracy: String,
    pub color_degrees: Option<BTreeMap<String, Option<i64>>>,
    pub color_contraction: String,
    pub engine: String,
    pub schedule: String,
    pub r2_strategy: String,
    pub arithmetic: Vec<String>,
    pub color_correlations: Vec<String>,
    pub spin_correlations: Vec<Vec<usize>>,
    pub resolved: bool,
    pub poles: Vec<String>,
    pub stability_checks: Vec<String>,
    pub accuracy_retest: String,
    pub rescue_policy: String,
    pub contribution_kind: String,
    pub execution_mode: String,
    pub epsilon_powers: Vec<i32>,
    pub epsilon_convention: String,
    pub dimension: String,
    pub interference: String,
    pub scheme_status: String,
}

/// One external leg of a plan v4 process, in momentum order.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LoopV2StateLeg {
    pub pdg: i32,
    pub incoming: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionV4 {
    schema_version: u32,
    kind: String,
    process_id: String,
    contribution_kind: String,
    execution_mode: String,
    engine: String,
    required_runtime_capabilities: Vec<String>,
    runtime_path: String,
    plan_sha256: String,
    point_inputs: String,
    request: serde_json::Value,
}

/// How a plan v4 process obtains its per-point inputs (trees, Born
/// amplitudes, counterterm scalars): rebuilt at load from the artifact's
/// compiled model and generation request (Python, until WP3.1 stores them as
/// native programs under a new value of this field and further PACBIN
/// members; the plan member is unchanged by that addition).
pub const LOOP_V2_POINT_INPUTS: &str = "regenerated-at-load";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PhysicsV4 {
    schema_version: u32,
    kind: String,
    process_id: String,
    metadata: LoopV2ExecutionMetadata,
    external_pdgs: Vec<i32>,
    state_legs: Vec<LoopV2StateLeg>,
    helicities: Vec<Vec<i32>>,
}

/// A plan v4 process admitted from its artifact. `plan_json` is the stored
/// plan (its SHA-256 is `plan_sha256`); `request` is the generation recipe.
pub struct LoadedLoopV2Artifact {
    pub artifact_id: String,
    pub selection: ArtifactSelection,
    pub metadata: LoopV2ExecutionMetadata,
    pub state_legs: Vec<LoopV2StateLeg>,
    pub helicities: Vec<Vec<i32>>,
    pub request: serde_json::Value,
    pub plan_sha256: String,
    pub plan_json: Vec<u8>,
    #[cfg(feature = "loop-v2")]
    pub program: Arc<crate::loop_v2::plan::LoopProgramV4>,
}

pub fn open_loop_v2_artifact(
    path: impl AsRef<Path>,
    requested_process: Option<&str>,
) -> RusticolResult<LoadedLoopV2Artifact> {
    load_loop_v2_artifact(&VerifiedArtifact::open(path)?, requested_process)
}

/// Admit one plan v4 process: the envelope, its execution and physics
/// declarations, the authenticated plan member and its digest, and (with the
/// `loop-v2` feature) the validated program, whose families, reduction
/// buckets, legs, helicity rows and sectors must equal the declarations.
pub fn load_loop_v2_artifact(
    artifact: &VerifiedArtifact,
    requested_process: Option<&str>,
) -> RusticolResult<LoadedLoopV2Artifact> {
    use sha2::{Digest, Sha256};

    require(
        artifact.manifest().runtime.required_runtime_capabilities == [LOOP_V2_RUNTIME_CAPABILITY]
            && artifact
                .manifest()
                .processes
                .iter()
                .all(|p| p.required_runtime_capabilities == [LOOP_V2_RUNTIME_CAPABILITY]),
        "plan v4 processes cannot share an artifact with other runtimes",
    )?;
    let selection = artifact.select_process(requested_process)?;
    require(
        selection.process.aliases.is_empty() && !selection.remaps_public_process(),
        "plan v4 admits independent virtual processes only",
    )?;
    let entries = crate::engine::verified_execution_payloads(artifact)?;
    let (_, path, bytes) = entries
        .into_iter()
        .find(|(p, _, _)| p.id == selection.process.id)
        .ok_or_else(|| RusticolError::integrity("execution set omits selected virtual process"))?;
    let bytes = match bytes {
        Some(bytes) => bytes,
        None => artifact.read_payload(&path)?,
    };
    let execution: ExecutionV4 = parse(&bytes)?;
    require(
        execution.schema_version == 4
            && execution.kind == LOOP_RUNTIME_KIND
            && execution.process_id == selection.process.id
            && execution.contribution_kind == "virtual"
            && execution.execution_mode == "loop_recurrence"
            && execution.engine == "v2"
            && execution.point_inputs == LOOP_V2_POINT_INPUTS
            && execution.request.is_object(),
        "unsupported plan v4 execution declaration",
    )?;
    require(
        execution.required_runtime_capabilities == [LOOP_V2_RUNTIME_CAPABILITY]
            && selection.process.required_runtime_capabilities
                == execution.required_runtime_capabilities,
        "execution and process runtime capabilities disagree",
    )?;
    let physics_path = &selection.process.physics_path;
    let physics_payload = artifact.payload(physics_path)?;
    require(
        physics_payload.role == PayloadRole::RuntimePhysics
            && physics_payload.process_id.as_deref() == Some(selection.process.id.as_str()),
        "physics payload ownership disagrees with selected process",
    )?;
    let physics: PhysicsV4 = parse(&artifact.read_payload(physics_path)?)?;
    require(
        physics.schema_version == 2
            && physics.kind == "pyamplicol-loop-physics"
            && physics.process_id == selection.process.id
            && physics.external_pdgs == selection.process.external_pdgs
            && physics.state_legs.len() == physics.external_pdgs.len()
            && physics
                .state_legs
                .iter()
                .zip(&physics.external_pdgs)
                .all(|(leg, pdg)| leg.pdg == *pdg),
        "invalid plan v4 physics metadata",
    )?;
    let metadata = &physics.metadata;
    validate_v2_metadata(metadata)?;
    require(
        selection.process.color_accuracy == metadata.color_accuracy,
        "declared colour accuracy differs from the process record",
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
    let reader = store.source(&execution.runtime_path)?.pacbin()?;
    let plan_json = crate::loop_runtime::read_loop_program_v4_json(&reader)?;
    require(
        format!("{:x}", Sha256::digest(&plan_json)) == execution.plan_sha256,
        "the stored plan v4 differs from its declared digest",
    )?;
    #[cfg(feature = "loop-v2")]
    let program = {
        let program = crate::loop_v2::plan::LoopProgramV4::load(&plan_json)?;
        let buckets: Vec<(usize, usize, usize)> = program
            .buckets()
            .into_iter()
            .map(|((n, r), families)| (n, r, families.len()))
            .collect();
        require(
            program.families.len() == metadata.families
                && buckets == metadata.buckets
                && metadata
                    .spin_correlations
                    .iter()
                    .flatten()
                    .all(|leg| *leg <= program.legs)
                && program.legs == physics.state_legs.len()
                && program.rows == physics.helicities.len()
                && program.columns == metadata.sectors.len()
                && physics
                    .helicities
                    .iter()
                    .all(|row| row.len() == program.legs),
            "plan v4 families, buckets, legs, helicity rows or sectors disagree \
             with the declarations",
        )?;
        Arc::new(program)
    };
    Ok(LoadedLoopV2Artifact {
        artifact_id: artifact.manifest().artifact_id.clone(),
        selection,
        metadata: physics.metadata,
        state_legs: physics.state_legs,
        helicities: physics.helicities,
        request: execution.request,
        plan_sha256: execution.plan_sha256,
        plan_json,
        #[cfg(feature = "loop-v2")]
        program,
    })
}

fn validate_v2_metadata(metadata: &LoopV2ExecutionMetadata) -> RusticolResult<()> {
    require(
        metadata.engine == "v2"
            && metadata.contribution_kind == "virtual"
            && metadata.execution_mode == "loop_recurrence"
            && metadata.epsilon_powers == [0, -1, -2]
            && metadata.epsilon_convention == EPSILON_CONVENTION
            && metadata.dimension == "D=4-2*epsilon"
            && metadata.interference == ORDINARY_INTERFERENCE
            && metadata.scheme_status == "declared-not-certified"
            && metadata.schedule == "backward"
            && metadata.poles == ["all", "none"]
            && metadata.accuracy_retest == "reduction-direction"
            && metadata.rescue_policy == "caller"
            && metadata.stability_checks == ["rotation", "reduction-direction"]
            && matches!(metadata.color_contraction.as_str(), "auto" | "direct"),
        "unsupported contribution, convention or feature declaration",
    )?;
    require(
        metadata.arithmetic == ["double", "double-double", "arbitrary"],
        "unsupported arithmetic declaration",
    )?;
    require(
        metadata.resolved,
        "v2 processes contract seed columns directly and serve resolved output",
    )?;
    let mut histories: Vec<&String> = metadata.color_correlations.iter().collect();
    histories.sort();
    histories.dedup();
    require(
        histories.len() == metadata.color_correlations.len()
            && histories
                .iter()
                .all(|id| !id.is_empty() && id.as_str() != "born")
            && metadata.spin_correlations.iter().all(|legs| {
                !legs.is_empty() && legs.windows(2).all(|w| w[0] < w[1]) && legs[0] > 0
            }),
        "colour histories are distinct non-Born ids; spin classes are increasing \
         one-based legs",
    )?;
    require(
        metadata.r2_strategy == "model",
        "plan v4 artifacts carry source-owned model R2 and UV",
    )?;
    require(
        matches!(metadata.color_accuracy.as_str(), "full" | "nlc" | "lc")
            && metadata.color_degrees.as_ref().map_or(
                metadata.color_accuracy == "full",
                |degrees| {
                    metadata.color_accuracy != "full" && degrees.len() == metadata.sectors.len()
                },
            ),
        "colour degrees are declared exactly per sector at LC and NLC",
    )?;
    require(
        !metadata.born_orders.is_empty()
            && !metadata.sectors.is_empty()
            && metadata.sectors.iter().all(|s| !s.pairs.is_empty())
            && metadata.families > 0
            && metadata.buckets.iter().map(|b| b.2).sum::<usize>() == metadata.families,
        "plan v4 needs Born orders, sectors and reduction buckets covering every family",
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
    )
}
