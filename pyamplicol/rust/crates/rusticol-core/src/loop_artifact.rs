// SPDX-License-Identifier: 0BSD

//! Checked virtual-process lane in the ordinary schema-v3 artifact envelope.
//! Scheme records are explicit declarations; loading does not certify physics.
//!
//! The shared-recursion engine v2 (`rusticol.loop-runtime.v4`,
//! [`load_loop_v2_artifact`], DESIGN.md §12.1) publishes them; its PACBIN
//! holds one plan v4 member. The removed legacy engine's capability
//! (`LOOP_RUNTIME_CAPABILITY`, `rusticol.loop-runtime.v3`) is only recognised
//! to refuse its artifacts with the reason.

use crate::{ArtifactSelection, PayloadRole, RusticolError, RusticolResult, VerifiedArtifact};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

pub const LOOP_RUNTIME_CAPABILITY: &str = "rusticol.loop-runtime.v3";
pub use crate::engine::LOOP_V2_RUNTIME_CAPABILITY;
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


fn require(condition: bool, message: &str) -> RusticolResult<()> {
    if condition {
        Ok(())
    } else {
        Err(RusticolError::compatibility(format!(
            "loop artifact: {message}"
        )))
    }
}
fn parse<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> RusticolResult<T> {
    serde_json::from_slice(bytes)
        .map_err(|e| RusticolError::serialization(format!("loop artifact: {e}")))
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
/// output is served (always: observable columns are contracted directly).
/// `color_contraction` is the requested seed contraction ("auto", "direct"
/// or "fft"; DESIGN.md §7.3): the certified pairs of the stored seed Fourier
/// record are contracted by the symmetric-group FFT under "auto" and "fft".
/// `poles` lists the pole selections an evaluation may
/// request (`PoleSelection`, DESIGN.md §12.4): every pole ("all"), the finite
/// part only ("none"), and the ultraviolet ("uv") or infrared ("ir") parts:
/// always all four ([`LOOP_V2_POLE_SELECTIONS`]).
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

/// The pole selections a plan v4 process declares (one artifact serves every
/// selection). No older declaration is admitted (AGENTS.md: no backward
/// compatibility of internal artifact formats).
pub const LOOP_V2_POLE_SELECTIONS: [&str; 4] = ["all", "none", "uv", "ir"];

#[cfg(feature = "loop-v2")]
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
    /// SHA-256 of the companion plan of the certified-scaleless families
    /// (`LOOP_PROGRAM_V4_SCALELESS_MEMBER`); null without such families. The
    /// key is required (an absent key is not read as null: no compatibility
    /// of internal artifact formats, AGENTS.md).
    #[serde(deserialize_with = "required_nullable")]
    scaleless_plan_sha256: Option<String>,
    point_inputs: String,
    request: serde_json::Value,
}

/// A nullable key that must be present: serde reads an absent `Option` field
/// as `None` unless the field has its own deserializer.
#[cfg(feature = "loop-v2")]
fn required_nullable<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Option::<String>::deserialize(deserializer)
}

/// Native point inputs (WP3.1), how every plan v4 process obtains its
/// per-point inputs (trees, Born amplitudes, counterterm and field
/// components): the PACBIN also holds the process's native
/// program (`loops/v2/native_process.py`): its assembly record and tree
/// program (zlib JSON, kind `LoopRuntimePlan`), its parameter evaluator and
/// kernel evaluators (saved exact Symbolica states, for DoubleFloat and
/// arbitrary precision) and their binary64 SymJIT plane applications
/// (storage-v4 IR, kind `SymjitApplication`, recompiled at load), so a load
/// regenerates nothing and binary64 touches no Symbolica state.
pub const LOOP_V2_NATIVE_POINT_INPUTS: &str = "native-v2";

/// Former point-input declarations, refused with a request to regenerate:
/// `native-v1` stored no plane applications; `regenerated-at-load` rebuilt
/// the inputs at load from the artifact's compiled model and generation
/// request (a stored process never regenerates, REQUIREMENTS_ADDENDUM G).
pub const LOOP_V2_RETIRED_POINT_INPUTS: [&str; 2] = ["native-v1", "regenerated-at-load"];

/// The PACBIN members of native point inputs.
pub const LOOP_V2_NATIVE_RECORD_MEMBER: &str = "point-inputs/native-process.json.zlib";
pub const LOOP_V2_NATIVE_TREES_MEMBER: &str = "point-inputs/tree-program.json.zlib";
pub const LOOP_V2_NATIVE_PARAMETERS_MEMBER: &str = "point-inputs/parameters.state";
pub const LOOP_V2_NATIVE_PARAMETERS_PLANE_MEMBER: &str = "point-inputs/parameters.symjit";
/// The interaction kernels' saved evaluators and plane applications,
/// `kernel-0.state` / `kernel-0.symjit` onwards.
pub const LOOP_V2_NATIVE_KERNEL_PREFIX: &str = "point-inputs/kernel-";
/// The certified-scaleless companion's tree program (the "uv" and "ir" pole
/// selections run it): the same members under this prefix
/// (`tree-program.json.zlib`, `parameters.state`, `parameters.symjit`,
/// `kernel-0.state` / `kernel-0.symjit` onwards).
pub const LOOP_V2_NATIVE_SCALELESS_PREFIX: &str = "point-inputs/scaleless/";

/// The native point inputs of a plan v4 process
/// ([`LOOP_V2_NATIVE_POINT_INPUTS`]): the assembly record and the tree
/// program (JSON), the interaction kernels' and the parameter program's
/// saved exact evaluators and their binary64 SymJIT plane applications
/// (`loop_v2::trees::TreeProgram::load` order: the kernels, then the
/// parameter program).
#[cfg(feature = "loop-v2")]
pub struct LoopV2NativeInputs {
    pub record_json: Vec<u8>,
    pub trees_json: Vec<u8>,
    pub evaluators: Vec<Vec<u8>>,
    pub planes: Vec<Vec<u8>>,
    /// The certified-scaleless companion's tree program
    /// ([`LOOP_V2_NATIVE_SCALELESS_PREFIX`]), stored exactly when the
    /// process stores its companion plan (`scaleless_plan_json`).
    pub scaleless: Option<LoopV2NativeTrees>,
}

/// One stored tree program: its JSON and its kernels' and parameter
/// program's saved exact evaluators and plane applications (the
/// `loop_v2::trees::TreeProgram::load` order).
#[cfg(feature = "loop-v2")]
pub struct LoopV2NativeTrees {
    pub trees_json: Vec<u8>,
    pub evaluators: Vec<Vec<u8>>,
    pub planes: Vec<Vec<u8>>,
}

#[cfg(feature = "loop-v2")]
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
/// `scaleless_*` is the companion plan of the certified-scaleless families
/// that the "uv" and "ir" pole selections run (DESIGN.md §12.4), when the
/// process has such families.
#[cfg(feature = "loop-v2")]
pub struct LoadedLoopV2Artifact {
    pub artifact_id: String,
    pub selection: ArtifactSelection,
    pub metadata: LoopV2ExecutionMetadata,
    pub state_legs: Vec<LoopV2StateLeg>,
    pub helicities: Vec<Vec<i32>>,
    pub request: serde_json::Value,
    pub plan_sha256: String,
    pub plan_json: Vec<u8>,
    pub program: Arc<crate::loop_v2::plan::LoopProgramV4>,
    pub scaleless_plan_sha256: Option<String>,
    pub scaleless_plan_json: Option<Vec<u8>>,
    pub scaleless_program: Option<Arc<crate::loop_v2::plan::LoopProgramV4>>,
    /// The execution payload's path and its PACBIN's runtime path, and the
    /// declared point inputs ([`read_loop_v2_native_inputs`] reads the
    /// native ones on demand: the Python runtime reads them itself).
    pub execution_path: String,
    pub runtime_path: String,
    pub point_inputs: String,
}

#[cfg(feature = "loop-v2")]
pub fn open_loop_v2_artifact(
    path: impl AsRef<Path>,
    requested_process: Option<&str>,
) -> RusticolResult<LoadedLoopV2Artifact> {
    load_loop_v2_artifact(&VerifiedArtifact::open(path)?, requested_process)
}

/// Admit one plan v4 process: the envelope, its execution and physics
/// declarations, the plan member and its digest, and the validated program,
/// whose families, reduction buckets, legs, helicity rows and sectors must
/// equal the declarations (the `loop-v2` feature: plan v4 needs the v2
/// executor, and so does its zlib member).
#[cfg(feature = "loop-v2")]
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
    // Internal formats are not kept backward compatible (AGENTS.md).
    require(
        !LOOP_V2_RETIRED_POINT_INPUTS.contains(&execution.point_inputs.as_str()),
        "the artifact's point inputs are no longer supported (native-v1 stored \
         no SymJIT plane applications; regenerated-at-load regenerated the \
         process at load); regenerate the artifact",
    )?;
    require(
        execution.schema_version == 4
            && execution.kind == LOOP_RUNTIME_KIND
            && execution.process_id == selection.process.id
            && execution.contribution_kind == "virtual"
            && execution.execution_mode == "loop_recurrence"
            && execution.engine == "v2"
            && execution.point_inputs == LOOP_V2_NATIVE_POINT_INPUTS
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
    if execution.point_inputs == LOOP_V2_NATIVE_POINT_INPUTS {
        use crate::pacbin::PacbinMemberKind;
        for (member, kind) in [
            (
                LOOP_V2_NATIVE_RECORD_MEMBER,
                PacbinMemberKind::LoopRuntimePlan,
            ),
            (
                LOOP_V2_NATIVE_TREES_MEMBER,
                PacbinMemberKind::LoopRuntimePlan,
            ),
            (
                LOOP_V2_NATIVE_PARAMETERS_MEMBER,
                PacbinMemberKind::SymbolicaExactState,
            ),
            (
                LOOP_V2_NATIVE_PARAMETERS_PLANE_MEMBER,
                PacbinMemberKind::SymjitApplication,
            ),
        ] {
            require(
                reader.member(member)?.kind() == kind,
                "native point inputs member of the wrong kind",
            )?;
        }
    }
    require(
        format!("{:x}", Sha256::digest(&plan_json)) == execution.plan_sha256,
        "the stored plan v4 differs from its declared digest",
    )?;
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
    // The companion plan of the certified-scaleless families: declared by
    // digest exactly when stored.
    let scaleless_plan_json = crate::loop_runtime::read_loop_program_v4_scaleless_json(&reader)?;
    require(
        scaleless_plan_json.is_some() == execution.scaleless_plan_sha256.is_some(),
        "the companion plan of certified-scaleless families is stored exactly \
         when declared",
    )?;
    let scaleless_program = match (&scaleless_plan_json, &execution.scaleless_plan_sha256) {
        (Some(json), Some(digest)) => {
            require(
                format!("{:x}", Sha256::digest(json)) == *digest,
                "the stored companion plan v4 differs from its declared digest",
            )?;
            let companion = crate::loop_v2::plan::LoopProgramV4::load(json)?;
            require(
                !companion.families.is_empty()
                    && companion.legs == program.legs
                    && companion.rows == program.rows
                    && companion.columns == program.columns,
                "the companion plan's legs, helicity rows or sectors disagree \
                 with the process plan",
            )?;
            Some(Arc::new(companion))
        }
        _ => None,
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
        program,
        scaleless_plan_sha256: execution.scaleless_plan_sha256,
        scaleless_plan_json,
        scaleless_program,
        execution_path: path,
        runtime_path: execution.runtime_path,
        point_inputs: execution.point_inputs,
    })
}

/// The native point inputs of an admitted plan v4 process (its declared
/// `point_inputs` is [`LOOP_V2_NATIVE_POINT_INPUTS`]; `None` otherwise):
/// read from the process's PACBIN, the zlib JSON members inflated.
#[cfg(feature = "loop-v2")]
pub fn read_loop_v2_native_inputs(
    artifact: &VerifiedArtifact,
    loaded: &LoadedLoopV2Artifact,
) -> RusticolResult<Option<LoopV2NativeInputs>> {
    use crate::pacbin::PacbinMemberKind;
    use std::io::Read;

    if loaded.point_inputs != LOOP_V2_NATIVE_POINT_INPUTS {
        return Ok(None);
    }
    let root = artifact.root().join(
        Path::new(&loaded.execution_path)
            .parent()
            .unwrap_or(Path::new("")),
    );
    let store = artifact.evaluator_payload_store(&root)?;
    let reader = store.source(&loaded.runtime_path)?.pacbin()?;
    let member = |name: &str, kind: PacbinMemberKind| -> RusticolResult<Vec<u8>> {
        require(
            reader.member(name)?.kind() == kind,
            "native point inputs member of the wrong kind",
        )?;
        Ok(reader.member_bytes(name)?.to_vec())
    };
    let inflate = |name: &str| -> RusticolResult<Vec<u8>> {
        let mut json = Vec::new();
        flate2::read::ZlibDecoder::new(member(name, PacbinMemberKind::LoopRuntimePlan)?.as_slice())
            .read_to_end(&mut json)
            .map_err(|error| {
                RusticolError::integrity(format!("{name} is not a zlib stream: {error}"))
            })?;
        Ok(json)
    };
    // The tree program under `prefix` (`point-inputs/` or the companion's).
    let trees = |prefix: &str| -> RusticolResult<LoopV2NativeTrees> {
        let mut evaluators = Vec::new();
        let mut planes = Vec::new();
        loop {
            let state = format!("{prefix}kernel-{}.state", evaluators.len());
            if reader.member(&state).is_err() {
                break;
            }
            let plane = format!("{prefix}kernel-{}.symjit", evaluators.len());
            evaluators.push(member(&state, PacbinMemberKind::SymbolicaExactState)?);
            planes.push(member(&plane, PacbinMemberKind::SymjitApplication)?);
        }
        evaluators.push(member(
            &format!("{prefix}parameters.state"),
            PacbinMemberKind::SymbolicaExactState,
        )?);
        planes.push(member(
            &format!("{prefix}parameters.symjit"),
            PacbinMemberKind::SymjitApplication,
        )?);
        Ok(LoopV2NativeTrees {
            trees_json: inflate(&format!("{prefix}tree-program.json.zlib"))?,
            evaluators,
            planes,
        })
    };
    let main = trees(LOOP_V2_NATIVE_INPUTS_PREFIX)?;
    let scaleless = match loaded.scaleless_plan_json {
        Some(_) => Some(trees(LOOP_V2_NATIVE_SCALELESS_PREFIX)?),
        None => None,
    };
    Ok(Some(LoopV2NativeInputs {
        record_json: inflate(LOOP_V2_NATIVE_RECORD_MEMBER)?,
        trees_json: main.trees_json,
        evaluators: main.evaluators,
        planes: main.planes,
        scaleless,
    }))
}

/// The prefix of the process's own tree program members.
#[cfg(feature = "loop-v2")]
const LOOP_V2_NATIVE_INPUTS_PREFIX: &str = "point-inputs/";

#[cfg(feature = "loop-v2")]
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
            && metadata.accuracy_retest == "reduction-direction"
            && metadata.rescue_policy == "caller"
            && metadata.stability_checks == ["rotation", "reduction-direction"]
            && matches!(
                metadata.color_contraction.as_str(),
                "auto" | "direct" | "fft"
            ),
        "unsupported contribution, convention or feature declaration",
    )?;
    require(
        metadata.poles == LOOP_V2_POLE_SELECTIONS,
        "unsupported pole selections (plan v4 serves all, none, uv and ir): \
         regenerate the artifact",
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
