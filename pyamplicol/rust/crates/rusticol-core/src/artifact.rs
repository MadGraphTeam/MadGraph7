// SPDX-License-Identifier: 0BSD

use crate::pacbin::{PacbinMemberKind, PacbinReader};
use crate::{
    ARTIFACT_MANIFEST_FILE, C_ABI_VERSION, COMPILED_MODEL_SCHEMA_VERSION,
    PROCESS_ARTIFACT_SCHEMA_VERSION, PYTHON_API_VERSION, RUNTIME_PHYSICS_SCHEMA_VERSION,
    RuntimeCapability, RusticolError, RusticolResult, TOML_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_json::value::RawValue;
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
#[cfg(test)]
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;
const EVALUATOR_PAYLOAD_CONTAINER_EXTENSION: &str = "evaluator_payload_container";
const EVALUATOR_PAYLOAD_CONTAINER_KIND: &str = "pyamplicol-evaluator-payload-container";
const EVALUATOR_PAYLOAD_CONTAINER_STORAGE_ABI: &str = "pacbin-v1";
const SUPPORTED_ARTIFACT_TARGETS: [&str; 3] = [
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ArtifactKind {
    PyamplicolProcess,
    PyamplicolProcessSet,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VersionSet {
    pub python_api: u32,
    pub toml: u32,
    pub compiled_model: u32,
    pub process_artifact: u32,
    pub runtime_physics: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbolica_serialization: Option<String>,
    pub c_abi: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub triple: String,
    pub cpu_features: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Producer {
    pub distribution: String,
    pub version: String,
    pub versions: VersionSet,
    pub target: Target,
    #[serde(default)]
    pub git_revision: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelSourceKind {
    BuiltInSm,
    Ufo,
    UfoJson,
    CompiledModel,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactModel {
    pub name: String,
    pub source_kind: ModelSourceKind,
    pub content_sha256: String,
    pub compiled_schema_version: u32,
    #[serde(default)]
    pub restriction: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ConfigurationAdjustment {
    pub path: String,
    pub reason: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactConfiguration {
    pub toml_schema_version: u32,
    pub requested_path: String,
    pub effective_path: String,
    pub adjustments: Vec<ConfigurationAdjustment>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessAlias {
    pub id: String,
    pub expression: String,
    pub external_pdgs: Vec<i32>,
    pub external_permutation: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactProcess {
    pub id: String,
    pub expression: String,
    pub color_accuracy: String,
    pub external_pdgs: Vec<i32>,
    pub physics_path: String,
    pub required_runtime_capabilities: Vec<String>,
    pub aliases: Vec<ProcessAlias>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRuntime {
    pub engine: String,
    pub engine_version: String,
    pub evaluator_manifest_path: String,
    pub required_runtime_capabilities: Vec<String>,
    #[serde(default)]
    pub api_bundle_path: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum PayloadRole {
    ConfigurationRequested,
    ConfigurationEffective,
    CompiledModel,
    RuntimePhysics,
    EvaluatorManifest,
    EvaluatorState,
    ModelParameters,
    ValidationMomenta,
    ApiSource,
    ApiBuildFile,
    SdkMetadata,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Payload {
    pub path: String,
    pub role: PayloadRole,
    pub media_type: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub executable: bool,
    #[serde(default)]
    pub target: Option<Target>,
    #[serde(default)]
    pub process_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Dependency {
    pub name: String,
    pub version: String,
    pub source: String,
    pub license: String,
    #[serde(default)]
    pub content_sha256: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub patch_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifest {
    pub schema_version: u32,
    pub kind: ArtifactKind,
    pub artifact_id: String,
    pub created_utc: String,
    pub producer: Producer,
    pub model: ArtifactModel,
    pub configuration: ArtifactConfiguration,
    pub processes: Vec<ArtifactProcess>,
    #[serde(default)]
    pub default_process_id: Option<String>,
    pub runtime: ArtifactRuntime,
    pub payloads: Vec<Payload>,
    pub dependencies: Vec<Dependency>,
    #[serde(default)]
    pub extensions: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactSelection {
    pub process: ArtifactProcess,
    pub requested_id: String,
    pub alias: Option<ProcessAlias>,
}

#[derive(Clone, Debug)]
pub struct VerifiedArtifact {
    root: PathBuf,
    manifest_path: PathBuf,
    manifest: ArtifactManifest,
    payloads: BTreeMap<String, Payload>,
    evaluator_payload_container: Option<Arc<PacbinReader>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvaluatorPayloadContainerExtension {
    kind: String,
    schema_version: u32,
    storage_abi: String,
    path: String,
    member_count: u64,
    unpacked_size_bytes: u64,
    index_sha256: String,
}

/// One evaluator payload resolved from a legacy loose file or a packed member.
#[derive(Clone, Debug)]
pub(crate) enum EvaluatorPayloadSource {
    File(PathBuf),
    Packed {
        container: Arc<PacbinReader>,
        logical_path: String,
    },
}

impl EvaluatorPayloadSource {
    pub(crate) fn read(&self) -> RusticolResult<Cow<'_, [u8]>> {
        match self {
            Self::File(path) => fs::read(path).map(Cow::Owned).map_err(|error| {
                RusticolError::artifact(format!(
                    "could not read evaluator payload {}: {error}",
                    path.display()
                ))
            }),
            Self::Packed {
                container,
                logical_path,
            } => container.member_bytes(logical_path).map(Cow::Borrowed),
        }
    }

    pub(crate) fn display_name(&self) -> String {
        match self {
            Self::File(path) => path.display().to_string(),
            Self::Packed { logical_path, .. } => format!("evaluators.pacbin:{logical_path}"),
        }
    }
}

/// A path-scoped resolver shared by compiled and eager evaluator loaders.
#[derive(Clone, Debug)]
pub(crate) struct EvaluatorPayloadStore {
    artifact_root: PathBuf,
    relative_root: PathBuf,
    container: Option<Arc<PacbinReader>>,
}

impl EvaluatorPayloadStore {
    pub(crate) fn directory(root: &Path) -> Self {
        Self {
            artifact_root: root.to_path_buf(),
            relative_root: root.to_path_buf(),
            container: None,
        }
    }

    pub(crate) fn source(&self, value: &str) -> RusticolResult<EvaluatorPayloadSource> {
        let relative = confined_evaluator_path(value)?;
        let path = self.relative_root.join(relative);
        let logical_path = artifact_logical_path(&self.artifact_root, &path)?;
        if let Some(container) = &self.container
            && container.member(&logical_path).is_ok()
        {
            return Ok(EvaluatorPayloadSource::Packed {
                container: container.clone(),
                logical_path,
            });
        }
        Ok(EvaluatorPayloadSource::File(path))
    }

    pub(crate) fn physical_path(&self, value: &str) -> RusticolResult<PathBuf> {
        match self.source(value)? {
            EvaluatorPayloadSource::File(path) => Ok(path),
            EvaluatorPayloadSource::Packed { logical_path, .. } => {
                Err(RusticolError::compatibility(format!(
                    "native evaluator library {logical_path:?} cannot be loaded from pacbin storage"
                )))
            }
        }
    }
}

impl VerifiedArtifact {
    /// Verify an artifact directory or a direct v3 manifest path.
    ///
    /// Every payload is size- and SHA-256-checked before this function returns.
    pub fn open(path: impl AsRef<Path>) -> RusticolResult<Self> {
        Self::open_with_manifest_preflight(path, |_| Ok(()))
    }

    pub(crate) fn open_with_manifest_preflight(
        path: impl AsRef<Path>,
        preflight: impl FnOnce(&ArtifactManifest) -> RusticolResult<()>,
    ) -> RusticolResult<Self> {
        let requested = path.as_ref();
        reject_symlink_chain(requested)?;
        let (root, manifest_path) = locate_manifest(requested)?;
        reject_symlink_chain(&manifest_path)?;
        let metadata = fs::metadata(&manifest_path).map_err(|error| {
            RusticolError::artifact(format!(
                "could not inspect artifact manifest {}: {error}",
                manifest_path.display()
            ))
        })?;
        if !metadata.is_file() {
            return Err(RusticolError::security(format!(
                "artifact manifest {} is not a regular file",
                manifest_path.display()
            )));
        }
        if metadata.len() > MAX_MANIFEST_BYTES {
            return Err(RusticolError::security(format!(
                "artifact manifest {} exceeds the {} byte limit",
                manifest_path.display(),
                MAX_MANIFEST_BYTES
            )));
        }
        let bytes = fs::read(&manifest_path).map_err(|error| {
            RusticolError::artifact(format!(
                "could not read artifact manifest {}: {error}",
                manifest_path.display()
            ))
        })?;
        let header: Value = serde_json::from_slice(&bytes).map_err(|error| {
            RusticolError::serialization(format!(
                "could not parse artifact manifest {} as JSON: {error}",
                manifest_path.display()
            ))
        })?;
        let schema_version = header
            .get("schema_version")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if matches!(schema_version, 1 | 2) {
            return Err(RusticolError::compatibility(format!(
                "process artifact schema v{schema_version} is unsupported and unsafe to migrate; regenerate it with `pyamplicol generate` to produce schema v3"
            )));
        }
        if schema_version != u64::from(PROCESS_ARTIFACT_SCHEMA_VERSION) {
            return Err(RusticolError::compatibility(format!(
                "unsupported process artifact schema {schema_version}; this runtime requires schema v{PROCESS_ARTIFACT_SCHEMA_VERSION}"
            )));
        }
        validate_artifact_identity(&header, &bytes)?;
        reject_forbidden_nulls(&header)?;
        let manifest: ArtifactManifest = serde_json::from_slice(&bytes).map_err(|error| {
            RusticolError::serialization(format!(
                "artifact manifest {} does not conform to schema v3: {error}",
                manifest_path.display()
            ))
        })?;
        validate_manifest(&manifest)?;
        preflight(&manifest)?;

        let mut payloads = BTreeMap::new();
        let mut portable_paths = BTreeSet::new();
        for payload in &manifest.payloads {
            validate_relative_path(&payload.path, "payload path")?;
            if payload.path == ARTIFACT_MANIFEST_FILE {
                return Err(RusticolError::security(format!(
                    "{ARTIFACT_MANIFEST_FILE} is reserved for the artifact manifest"
                )));
            }
            let portable = payload.path.to_ascii_lowercase();
            if !portable_paths.insert(portable) {
                return Err(RusticolError::security(format!(
                    "duplicate or case-colliding payload path {:?}",
                    payload.path
                )));
            }
            if payloads
                .insert(payload.path.clone(), payload.clone())
                .is_some()
            {
                return Err(RusticolError::security(format!(
                    "duplicate payload path {:?}",
                    payload.path
                )));
            }
        }
        validate_references(&manifest, &payloads)?;
        validate_artifact_tree(&root, &payloads)?;
        for payload in payloads.values() {
            validate_payload(&root, payload)?;
        }
        let evaluator_payload_container =
            load_evaluator_payload_container(&root, &manifest, &payloads)?;
        Ok(Self {
            root,
            manifest_path,
            manifest,
            payloads,
            evaluator_payload_container,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    pub fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    pub fn select_process(&self, requested: Option<&str>) -> RusticolResult<ArtifactSelection> {
        self.manifest.select_process(requested)
    }

    pub fn payload(&self, path: &str) -> RusticolResult<&Payload> {
        self.payloads.get(path).ok_or_else(|| {
            RusticolError::security(format!("artifact path {path:?} is not a declared payload"))
        })
    }

    pub fn read_payload(&self, path: &str) -> RusticolResult<Vec<u8>> {
        let payload = self.payload(path)?;
        validate_payload(&self.root, payload)?;
        fs::read(self.root.join(path)).map_err(|error| {
            RusticolError::artifact(format!("could not read payload {path:?}: {error}"))
        })
    }

    pub(crate) fn payload_path(&self, path: &str) -> RusticolResult<PathBuf> {
        let payload = self.payload(path)?;
        validate_payload(&self.root, payload)?;
        Ok(self.root.join(path))
    }

    pub(crate) fn evaluator_payload_store(
        &self,
        relative_root: &Path,
    ) -> RusticolResult<EvaluatorPayloadStore> {
        if !relative_root.starts_with(&self.root) {
            return Err(RusticolError::security(
                "evaluator payload root escapes the artifact root",
            ));
        }
        Ok(EvaluatorPayloadStore {
            artifact_root: self.root.clone(),
            relative_root: relative_root.to_path_buf(),
            container: self.evaluator_payload_container.clone(),
        })
    }

    pub(crate) fn has_evaluator_payload(&self, path: &str) -> RusticolResult<bool> {
        if let Some(payload) = self.payloads.get(path) {
            return Ok(payload.role == PayloadRole::EvaluatorState);
        }
        Ok(self
            .evaluator_payload_container
            .as_ref()
            .is_some_and(|container| container.member(path).is_ok()))
    }
}

fn load_evaluator_payload_container(
    root: &Path,
    manifest: &ArtifactManifest,
    payloads: &BTreeMap<String, Payload>,
) -> RusticolResult<Option<Arc<PacbinReader>>> {
    let Some(raw) = manifest
        .extensions
        .get(EVALUATOR_PAYLOAD_CONTAINER_EXTENSION)
    else {
        return Ok(None);
    };
    let extension: EvaluatorPayloadContainerExtension = serde_json::from_value(raw.clone())
        .map_err(|error| {
            RusticolError::artifact(format!(
                "artifact extension {EVALUATOR_PAYLOAD_CONTAINER_EXTENSION:?} is invalid: {error}"
            ))
        })?;
    if extension.kind != EVALUATOR_PAYLOAD_CONTAINER_KIND
        || extension.schema_version != 1
        || extension.storage_abi != EVALUATOR_PAYLOAD_CONTAINER_STORAGE_ABI
    {
        return Err(RusticolError::compatibility(format!(
            "unsupported evaluator payload container kind/version/ABI: {:?}/{}/{}",
            extension.kind, extension.schema_version, extension.storage_abi
        )));
    }
    validate_relative_path(&extension.path, "evaluator payload container path")?;
    validate_sha256(
        &extension.index_sha256,
        "evaluator payload container index_sha256",
    )?;
    let payload = payloads.get(&extension.path).ok_or_else(|| {
        RusticolError::security(format!(
            "evaluator payload container {:?} is not a declared payload",
            extension.path
        ))
    })?;
    if payload.role != PayloadRole::EvaluatorState
        || payload.media_type != "application/octet-stream"
        || payload.process_id.is_some()
    {
        return Err(RusticolError::artifact(
            "evaluator payload container must be a root evaluator-state octet-stream payload",
        ));
    }
    let reader = PacbinReader::open_trusted(root.join(&extension.path))?;
    let index = reader.index();
    if index.version() != 1
        || u64::try_from(index.members().len()).unwrap_or(u64::MAX) != extension.member_count
        || hex_digest(index.index_sha256()) != extension.index_sha256
    {
        return Err(RusticolError::integrity(
            "evaluator payload container metadata disagrees with its authenticated index",
        ));
    }
    let unpacked_size = index.members().iter().try_fold(0_u64, |total, member| {
        total
            .checked_add(member.length())
            .ok_or_else(|| RusticolError::integrity("packed evaluator size exceeds u64"))
    })?;
    if unpacked_size != extension.unpacked_size_bytes {
        return Err(RusticolError::integrity(
            "evaluator payload container unpacked size disagrees with its index",
        ));
    }
    for member in index.members() {
        validate_relative_path(member.logical_path(), "packed evaluator logical path")?;
        if payloads.contains_key(member.logical_path())
            || member.logical_path() == extension.path
            || !matches!(
                member.kind(),
                PacbinMemberKind::SymjitApplication | PacbinMemberKind::SymbolicaExactState
            )
        {
            return Err(RusticolError::integrity(format!(
                "invalid packed evaluator member {:?}",
                member.logical_path()
            )));
        }
    }
    Ok(Some(Arc::new(reader)))
}

fn confined_evaluator_path(value: &str) -> RusticolResult<&Path> {
    let path = Path::new(value);
    if value.is_empty()
        || value.contains('\\')
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(RusticolError::security(format!(
            "evaluator payload path {value:?} is not a confined relative path"
        )));
    }
    Ok(path)
}

fn artifact_logical_path(root: &Path, path: &Path) -> RusticolResult<String> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| RusticolError::security("evaluator payload path escapes the artifact root"))?;
    let logical = relative
        .to_str()
        .ok_or_else(|| RusticolError::security("evaluator payload path is not valid UTF-8"))?;
    Ok(logical.replace(std::path::MAIN_SEPARATOR, "/"))
}

fn hex_digest(bytes: &[u8]) -> String {
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(result, "{byte:02x}");
    }
    result
}

impl ArtifactManifest {
    pub(crate) fn select_process(
        &self,
        requested: Option<&str>,
    ) -> RusticolResult<ArtifactSelection> {
        let selected_id = if let Some(requested) = requested {
            requested
        } else if let Some(default) = self.default_process_id.as_deref() {
            default
        } else if self.processes.len() == 1 {
            self.processes[0].id.as_str()
        } else {
            return Err(RusticolError::selector(
                "this process-set artifact has no default process; select a process or alias id",
            ));
        };
        for process in &self.processes {
            if process.id == selected_id {
                return Ok(ArtifactSelection {
                    process: process.clone(),
                    requested_id: selected_id.to_string(),
                    alias: None,
                });
            }
            if let Some(alias) = process.aliases.iter().find(|alias| alias.id == selected_id) {
                return Ok(ArtifactSelection {
                    process: process.clone(),
                    requested_id: selected_id.to_string(),
                    alias: Some(alias.clone()),
                });
            }
        }

        let requested_expression = normalize_process_expression(selected_id);
        let mut expression_matches = Vec::new();
        for process in &self.processes {
            if normalize_process_expression(&process.expression) == requested_expression {
                expression_matches.push(ArtifactSelection {
                    process: process.clone(),
                    requested_id: process.id.clone(),
                    alias: None,
                });
            }
            for alias in &process.aliases {
                if normalize_process_expression(&alias.expression) == requested_expression {
                    expression_matches.push(ArtifactSelection {
                        process: process.clone(),
                        requested_id: alias.id.clone(),
                        alias: Some(alias.clone()),
                    });
                }
            }
        }
        match expression_matches.len() {
            1 => return Ok(expression_matches.pop().expect("one expression match")),
            count if count > 1 => {
                let mut matching_ids = expression_matches
                    .iter()
                    .map(|selection| selection.requested_id.as_str())
                    .collect::<Vec<_>>();
                matching_ids.sort_unstable();
                return Err(RusticolError::selector(format!(
                    "process expression {selected_id:?} is ambiguous; select one of these stable ids: {}",
                    matching_ids.join(", ")
                )));
            }
            _ => {}
        }
        Err(RusticolError::selector(format!(
            "unknown process id, alias id, or concrete process expression {selected_id:?}"
        )))
    }
}

fn normalize_process_expression(expression: &str) -> String {
    expression
        .split_whitespace()
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

fn validate_artifact_identity(manifest: &Value, bytes: &[u8]) -> RusticolResult<()> {
    let claimed = manifest
        .get("artifact_id")
        .and_then(Value::as_str)
        .ok_or_else(|| RusticolError::artifact("artifact_id must be a SHA-256 digest"))?;
    validate_sha256(claimed, "artifact_id")?;
    let computed = compute_artifact_id_from_bytes(bytes)?;
    if computed != claimed {
        return Err(RusticolError::integrity(format!(
            "artifact manifest identity digest mismatch: found {claimed}, computed {computed}"
        )));
    }
    Ok(())
}

fn compute_artifact_id_from_bytes(bytes: &[u8]) -> RusticolResult<String> {
    let mut fields: BTreeMap<String, Box<RawValue>> =
        serde_json::from_slice(bytes).map_err(|error| {
            RusticolError::serialization(format!(
                "could not preserve artifact manifest JSON for identity hashing: {error}"
            ))
        })?;
    if fields.remove("artifact_id").is_none() {
        return Err(RusticolError::artifact(
            "artifact manifest has no artifact_id field",
        ));
    }
    let mut canonical = serde_json::to_vec(&fields).map_err(|error| {
        RusticolError::serialization(format!(
            "could not canonicalize artifact manifest identity fields: {error}"
        ))
    })?;
    canonical.push(b'\n');
    Ok(format!("{:x}", Sha256::digest(&canonical)))
}

#[cfg(test)]
fn compute_artifact_id(manifest: &Value) -> RusticolResult<String> {
    let mut content = manifest.clone();
    let object = content
        .as_object_mut()
        .ok_or_else(|| RusticolError::artifact("artifact manifest must be a JSON object"))?;
    object.remove("artifact_id");
    let mut canonical = String::new();
    write_python_canonical_json(&content, &mut canonical)?;
    canonical.push('\n');
    Ok(format!("{:x}", Sha256::digest(canonical.as_bytes())))
}

#[cfg(test)]
fn write_python_canonical_json(value: &Value, output: &mut String) -> RusticolResult<()> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(true) => output.push_str("true"),
        Value::Bool(false) => output.push_str("false"),
        Value::Number(number) => output.push_str(&python_number(number)),
        Value::String(value) => write_python_json_string(value, output),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_python_canonical_json(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            for (index, key) in keys.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_python_json_string(key, output);
                output.push(':');
                write_python_canonical_json(&values[*key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

#[cfg(test)]
fn python_number(number: &serde_json::Number) -> String {
    let rendered = number.to_string();
    let Some((mantissa, exponent)) = rendered.split_once('e') else {
        return rendered;
    };
    let (sign, digits) = if let Some(digits) = exponent.strip_prefix('-') {
        ('-', digits)
    } else if let Some(digits) = exponent.strip_prefix('+') {
        ('+', digits)
    } else {
        ('+', exponent)
    };
    format!("{mantissa}e{sign}{digits:0>2}")
}

#[cfg(test)]
fn write_python_json_string(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            '\u{0020}'..='\u{007e}' => output.push(character),
            character if u32::from(character) <= 0xffff => {
                let _ = write!(output, "\\u{:04x}", u32::from(character));
            }
            character => {
                let scalar = u32::from(character) - 0x1_0000;
                let high = 0xd800 + (scalar >> 10);
                let low = 0xdc00 + (scalar & 0x3ff);
                let _ = write!(output, "\\u{high:04x}\\u{low:04x}");
            }
        }
    }
    output.push('"');
}

fn reject_forbidden_nulls(manifest: &Value) -> RusticolResult<()> {
    if manifest
        .get("runtime")
        .and_then(Value::as_object)
        .is_some_and(|runtime| !runtime.contains_key("api_bundle_path"))
    {
        return Err(RusticolError::artifact(
            "runtime.api_bundle_path is required and must be a relative path or null",
        ));
    }
    for (collection, keys) in [
        ("payloads", &["target", "process_id"] as &[&str]),
        (
            "dependencies",
            &["content_sha256", "revision", "patch_sha256"] as &[&str],
        ),
    ] {
        let Some(items) = manifest.get(collection).and_then(Value::as_array) else {
            continue;
        };
        for (index, item) in items.iter().enumerate() {
            for key in keys {
                if item.get(*key).is_some_and(Value::is_null) {
                    return Err(RusticolError::artifact(format!(
                        "{collection}[{index}].{key} may be omitted but may not be null"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn locate_manifest(requested: &Path) -> RusticolResult<(PathBuf, PathBuf)> {
    let metadata = fs::metadata(requested).map_err(|error| {
        RusticolError::artifact(format!(
            "could not inspect artifact path {}: {error}",
            requested.display()
        ))
    })?;
    if metadata.is_file() {
        if requested.file_name().and_then(|name| name.to_str()) != Some(ARTIFACT_MANIFEST_FILE) {
            return Err(RusticolError::artifact(format!(
                "artifact manifest must be named {ARTIFACT_MANIFEST_FILE}"
            )));
        }
        let manifest = requested.canonicalize().map_err(|error| {
            RusticolError::artifact(format!(
                "could not resolve artifact manifest {}: {error}",
                requested.display()
            ))
        })?;
        let root = manifest.parent().ok_or_else(|| {
            RusticolError::artifact("artifact manifest has no containing directory")
        })?;
        return Ok((root.to_path_buf(), manifest));
    }
    if !metadata.is_dir() {
        return Err(RusticolError::security(format!(
            "artifact path {} is neither a regular file nor a directory",
            requested.display()
        )));
    }
    let root = requested.canonicalize().map_err(|error| {
        RusticolError::artifact(format!(
            "could not resolve artifact directory {}: {error}",
            requested.display()
        ))
    })?;
    let manifest = root.join(ARTIFACT_MANIFEST_FILE);
    if manifest.exists() {
        Ok((root, manifest))
    } else {
        Err(RusticolError::artifact(format!(
            "artifact directory does not contain {ARTIFACT_MANIFEST_FILE}"
        )))
    }
}

fn validate_manifest(manifest: &ArtifactManifest) -> RusticolResult<()> {
    validate_sha256(&manifest.artifact_id, "artifact_id")?;
    validate_datetime(&manifest.created_utc)?;
    if manifest.producer.distribution != "pyamplicol" {
        return Err(RusticolError::compatibility(format!(
            "unsupported artifact producer {:?}; expected pyamplicol",
            manifest.producer.distribution
        )));
    }
    if !compatible_distribution_version(&manifest.producer.version)
        || !compatible_distribution_version(&manifest.runtime.engine_version)
    {
        return Err(RusticolError::compatibility(format!(
            "artifact producer/runtime version {}/{} is incompatible with Rusticol {}",
            manifest.producer.version,
            manifest.runtime.engine_version,
            env!("CARGO_PKG_VERSION")
        )));
    }
    let versions = &manifest.producer.versions;
    let expected = [
        ("python API", versions.python_api, PYTHON_API_VERSION),
        ("TOML", versions.toml, TOML_SCHEMA_VERSION),
        (
            "compiled model",
            versions.compiled_model,
            COMPILED_MODEL_SCHEMA_VERSION,
        ),
        (
            "process artifact",
            versions.process_artifact,
            PROCESS_ARTIFACT_SCHEMA_VERSION,
        ),
        (
            "runtime physics",
            versions.runtime_physics,
            RUNTIME_PHYSICS_SCHEMA_VERSION,
        ),
        ("C ABI", versions.c_abi, C_ABI_VERSION),
    ];
    for (name, found, required) in expected {
        if found != required {
            return Err(RusticolError::compatibility(format!(
                "artifact {name} version {found} is incompatible with required version {required}; regenerate the artifact"
            )));
        }
    }
    if manifest.runtime.engine != "rusticol" {
        return Err(RusticolError::compatibility(format!(
            "unsupported runtime engine {:?}",
            manifest.runtime.engine
        )));
    }
    validate_target(&manifest.producer.target, "producer")?;
    for payload in &manifest.payloads {
        if let Some(target) = &payload.target {
            validate_payload_target(&manifest.producer.target, target, &payload.path)?;
            validate_target(target, &format!("payload {}", payload.path))?;
        }
    }
    if let Some(revision) = &manifest.producer.git_revision
        && (revision.len() != 40
            || !revision
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()))
    {
        return Err(RusticolError::artifact(
            "producer.git_revision must be 40 lowercase hexadecimal characters",
        ));
    }
    validate_sha256(&manifest.model.content_sha256, "model content_sha256")?;
    if manifest.model.name.is_empty()
        || manifest
            .model
            .restriction
            .as_ref()
            .is_some_and(String::is_empty)
    {
        return Err(RusticolError::artifact(
            "model name and any model restriction must be non-empty",
        ));
    }
    if manifest.model.compiled_schema_version != COMPILED_MODEL_SCHEMA_VERSION {
        return Err(RusticolError::compatibility(format!(
            "compiled model schema {} is incompatible with required schema {}",
            manifest.model.compiled_schema_version, COMPILED_MODEL_SCHEMA_VERSION
        )));
    }
    if manifest.configuration.toml_schema_version != TOML_SCHEMA_VERSION {
        return Err(RusticolError::compatibility(format!(
            "configuration TOML schema {} is incompatible with required schema {}",
            manifest.configuration.toml_schema_version, TOML_SCHEMA_VERSION
        )));
    }
    validate_relative_path(
        &manifest.configuration.requested_path,
        "requested configuration path",
    )?;
    validate_relative_path(
        &manifest.configuration.effective_path,
        "effective configuration path",
    )?;
    for (index, adjustment) in manifest.configuration.adjustments.iter().enumerate() {
        if adjustment.path.is_empty() || adjustment.reason.is_empty() {
            return Err(RusticolError::artifact(format!(
                "configuration adjustment {index} requires non-empty path and reason"
            )));
        }
    }
    validate_relative_path(
        &manifest.runtime.evaluator_manifest_path,
        "evaluator manifest path",
    )?;
    let runtime_capabilities = validate_runtime_capabilities(
        &manifest.runtime.required_runtime_capabilities,
        "runtime.required_runtime_capabilities",
    )?;
    if let Some(path) = &manifest.runtime.api_bundle_path {
        validate_relative_path(path, "API bundle path")?;
    }
    if manifest.processes.is_empty() {
        return Err(RusticolError::artifact(
            "artifact must contain at least one process",
        ));
    }
    match manifest.kind {
        ArtifactKind::PyamplicolProcess if manifest.processes.len() != 1 => {
            return Err(RusticolError::artifact(
                "pyamplicol-process artifacts must contain exactly one process",
            ));
        }
        _ => {}
    }
    let mut public_ids = BTreeSet::new();
    let mut process_capabilities = BTreeSet::new();
    for process in &manifest.processes {
        validate_public_id(&process.id, "process id")?;
        validate_relative_path(&process.physics_path, "runtime physics path")?;
        if process.expression.is_empty() || process.external_pdgs.len() < 3 {
            return Err(RusticolError::artifact(format!(
                "process {} has an empty expression or fewer than three external particles",
                process.id
            )));
        }
        if !matches!(process.color_accuracy.as_str(), "lc" | "nlc" | "full") {
            return Err(RusticolError::artifact(format!(
                "process {} has unsupported color accuracy {:?}",
                process.id, process.color_accuracy
            )));
        }
        process_capabilities.extend(validate_runtime_capabilities(
            &process.required_runtime_capabilities,
            &format!("process {:?}.required_runtime_capabilities", process.id),
        )?);
        if !public_ids.insert(&process.id) {
            return Err(RusticolError::artifact(format!(
                "duplicate public process id {:?}",
                process.id
            )));
        }
        for alias in &process.aliases {
            validate_public_id(&alias.id, "process alias id")?;
            if alias.expression.is_empty() || !public_ids.insert(&alias.id) {
                return Err(RusticolError::artifact(format!(
                    "duplicate or invalid process alias id {:?}",
                    alias.id
                )));
            }
            validate_permutation(
                &alias.external_permutation,
                process.external_pdgs.len(),
                &alias.id,
            )?;
            if alias.external_permutation[0] != 0 || alias.external_permutation[1] != 1 {
                return Err(RusticolError::artifact(format!(
                    "alias {:?} may only permute final-state particles",
                    alias.id
                )));
            }
            let mut expected_external_pdgs = vec![0; process.external_pdgs.len()];
            for (representative_index, alias_index) in
                alias.external_permutation.iter().copied().enumerate()
            {
                expected_external_pdgs[alias_index] = process.external_pdgs[representative_index];
            }
            if alias.external_pdgs != expected_external_pdgs {
                return Err(RusticolError::artifact(format!(
                    "alias {:?} external_pdgs {:?} does not match external_permutation {:?}; expected {:?}",
                    alias.id,
                    alias.external_pdgs,
                    alias.external_permutation,
                    expected_external_pdgs,
                )));
            }
        }
    }
    if runtime_capabilities != process_capabilities {
        return Err(RusticolError::artifact(
            "runtime.required_runtime_capabilities must equal the union of process capability declarations",
        ));
    }
    if let Some(default) = &manifest.default_process_id
        && !public_ids.contains(default)
    {
        return Err(RusticolError::artifact(format!(
            "default process id {default:?} does not identify a process or alias"
        )));
    }
    if manifest.payloads.is_empty() {
        return Err(RusticolError::artifact(
            "artifact must declare at least one payload",
        ));
    }
    let mut dependencies = BTreeSet::new();
    for dependency in &manifest.dependencies {
        if dependency.name.is_empty()
            || dependency.version.is_empty()
            || dependency.source.is_empty()
            || dependency.license.is_empty()
            || !dependencies.insert(dependency.name.to_ascii_lowercase())
        {
            return Err(RusticolError::artifact(format!(
                "invalid or duplicate dependency {:?}",
                dependency.name
            )));
        }
        for (name, value) in [
            ("content_sha256", dependency.content_sha256.as_deref()),
            ("patch_sha256", dependency.patch_sha256.as_deref()),
        ] {
            if let Some(value) = value {
                validate_sha256(value, &format!("dependency {} {name}", dependency.name))?;
            }
        }
        if dependency.revision.as_ref().is_some_and(String::is_empty) {
            return Err(RusticolError::artifact(format!(
                "dependency {} revision must be non-empty when present",
                dependency.name
            )));
        }
    }
    Ok(())
}

fn compatible_distribution_version(version: &str) -> bool {
    canonical_distribution_version(version)
        == canonical_distribution_version(env!("CARGO_PKG_VERSION"))
}

fn canonical_distribution_version(version: &str) -> String {
    version.replace("-dev.", ".dev")
}

fn validate_references(
    manifest: &ArtifactManifest,
    payloads: &BTreeMap<String, Payload>,
) -> RusticolResult<()> {
    require_payload_role(
        payloads,
        &manifest.configuration.requested_path,
        PayloadRole::ConfigurationRequested,
        None,
    )?;
    require_payload_role(
        payloads,
        &manifest.configuration.effective_path,
        PayloadRole::ConfigurationEffective,
        None,
    )?;
    require_payload_role(
        payloads,
        &manifest.runtime.evaluator_manifest_path,
        PayloadRole::EvaluatorManifest,
        None,
    )?;
    for process in &manifest.processes {
        require_payload_role(
            payloads,
            &process.physics_path,
            PayloadRole::RuntimePhysics,
            Some(&process.id),
        )?;
    }
    if let Some(api_path) = &manifest.runtime.api_bundle_path {
        let prefix = format!("{}/", api_path.trim_end_matches('/'));
        if !payloads.keys().any(|path| path.starts_with(&prefix)) {
            return Err(RusticolError::artifact(format!(
                "API bundle path {api_path:?} contains no declared payload"
            )));
        }
    }
    Ok(())
}

fn require_payload_role(
    payloads: &BTreeMap<String, Payload>,
    path: &str,
    role: PayloadRole,
    process_id: Option<&str>,
) -> RusticolResult<()> {
    let payload = payloads.get(path).ok_or_else(|| {
        RusticolError::security(format!(
            "referenced artifact path {path:?} is not a declared payload"
        ))
    })?;
    if payload.role != role {
        return Err(RusticolError::security(format!(
            "referenced payload {path:?} has role {:?}, expected {:?}",
            payload.role, role
        )));
    }
    if let (Some(expected), Some(found)) = (process_id, payload.process_id.as_deref())
        && expected != found
    {
        return Err(RusticolError::artifact(format!(
            "payload {path:?} belongs to process {found:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn validate_artifact_tree(root: &Path, payloads: &BTreeMap<String, Payload>) -> RusticolResult<()> {
    let declared = payloads.keys().map(String::as_str).collect::<BTreeSet<_>>();
    validate_artifact_tree_directory(root, root, &declared)
}

fn validate_artifact_tree_directory(
    root: &Path,
    directory: &Path,
    declared: &BTreeSet<&str>,
) -> RusticolResult<()> {
    let entries = fs::read_dir(directory).map_err(|error| {
        RusticolError::security(format!(
            "could not inspect artifact directory {}: {error}",
            directory.display()
        ))
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            RusticolError::security(format!("could not inspect artifact tree entry: {error}"))
        })?;
        let path = entry.path();
        let relative = artifact_relative_path(root, &path)?;
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            RusticolError::security(format!(
                "could not inspect artifact tree entry {relative:?}: {error}"
            ))
        })?;
        let file_type = metadata.file_type();
        if file_type.is_symlink() {
            return Err(RusticolError::security(format!(
                "artifact tree contains a symlink at {relative:?}"
            )));
        }
        if file_type.is_dir() {
            validate_artifact_tree_directory(root, &path, declared)?;
            continue;
        }
        if !file_type.is_file() {
            return Err(RusticolError::security(format!(
                "artifact tree contains a non-regular entry at {relative:?}"
            )));
        }
        if !declared.contains(relative.as_str()) && metadata_is_executable(&metadata) {
            return Err(RusticolError::security(format!(
                "artifact tree contains an undeclared executable at {relative:?}"
            )));
        }
    }
    Ok(())
}

fn artifact_relative_path(root: &Path, path: &Path) -> RusticolResult<String> {
    let relative = path.strip_prefix(root).map_err(|_| {
        RusticolError::security("artifact tree entry escapes the canonical artifact root")
    })?;
    let mut parts = Vec::new();
    for component in relative.components() {
        let std::path::Component::Normal(part) = component else {
            return Err(RusticolError::security(
                "artifact tree contains a non-normal path component",
            ));
        };
        parts.push(
            part.to_str().ok_or_else(|| {
                RusticolError::security("artifact tree contains a non-UTF-8 path")
            })?,
        );
    }
    Ok(parts.join("/"))
}

fn metadata_is_executable(metadata: &fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        metadata.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        let _ = metadata;
        false
    }
}

fn validate_payload(root: &Path, payload: &Payload) -> RusticolResult<()> {
    validate_sha256(&payload.sha256, &format!("payload {} sha256", payload.path))?;
    if payload.media_type.is_empty() {
        return Err(RusticolError::artifact(format!(
            "payload {:?} has an empty media type",
            payload.path
        )));
    }
    if let Some(process_id) = &payload.process_id {
        validate_public_id(process_id, "payload process id")?;
    }
    if payload.role == PayloadRole::EvaluatorState && payload.target.is_none() {
        return Err(RusticolError::artifact(format!(
            "evaluator-state payload {:?} is missing required target metadata",
            payload.path
        )));
    }
    if payload.executable
        && !matches!(
            payload.role,
            PayloadRole::EvaluatorState | PayloadRole::ApiSource | PayloadRole::ApiBuildFile
        )
    {
        return Err(RusticolError::security(format!(
            "payload {:?} has role {:?}, which may not be executable",
            payload.path, payload.role
        )));
    }
    let path = root.join(&payload.path);
    reject_symlink_chain(&path)?;
    let canonical = path.canonicalize().map_err(|error| {
        RusticolError::security(format!(
            "could not resolve payload {:?}: {error}",
            payload.path
        ))
    })?;
    if !canonical.starts_with(root) {
        return Err(RusticolError::security(format!(
            "payload {:?} escapes the artifact root",
            payload.path
        )));
    }
    let metadata = fs::metadata(&canonical).map_err(|error| {
        RusticolError::security(format!(
            "could not inspect payload {:?}: {error}",
            payload.path
        ))
    })?;
    if !metadata.is_file() {
        return Err(RusticolError::security(format!(
            "payload {:?} is not a regular file",
            payload.path
        )));
    }
    if metadata.len() != payload.size_bytes {
        return Err(RusticolError::integrity(format!(
            "payload {:?} has size {}, expected {}",
            payload.path,
            metadata.len(),
            payload.size_bytes
        )));
    }
    #[cfg(unix)]
    {
        let executable = metadata_is_executable(&metadata);
        if executable != payload.executable {
            return Err(RusticolError::security(format!(
                "payload {:?} executable mode is {}, but the manifest declares {}",
                payload.path, executable, payload.executable
            )));
        }
    }
    let file = File::open(&canonical).map_err(|error| {
        RusticolError::artifact(format!(
            "could not open payload {:?}: {error}",
            payload.path
        ))
    })?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = reader.read(&mut buffer).map_err(|error| {
            RusticolError::artifact(format!(
                "could not hash payload {:?}: {error}",
                payload.path
            ))
        })?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let actual = format!("{:x}", digest.finalize());
    if actual != payload.sha256 {
        return Err(RusticolError::integrity(format!(
            "payload {:?} has SHA-256 {actual}, expected {}",
            payload.path, payload.sha256
        )));
    }
    Ok(())
}

fn validate_relative_path(value: &str, description: &str) -> RusticolResult<()> {
    if value.is_empty()
        || value.starts_with('/')
        || value.ends_with('/')
        || value.contains('\\')
        || value.contains('\0')
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
        || Path::new(value).is_absolute()
    {
        return Err(RusticolError::security(format!(
            "{description} {value:?} is not a normalized confined relative path"
        )));
    }
    Ok(())
}

fn reject_symlink_chain(path: &Path) -> RusticolResult<()> {
    let mut ancestors = path.ancestors().collect::<Vec<_>>();
    ancestors.reverse();
    for ancestor in ancestors {
        let Ok(metadata) = fs::symlink_metadata(ancestor) else {
            continue;
        };
        if metadata.file_type().is_symlink() {
            return Err(RusticolError::security(format!(
                "artifact path {} contains a symlink at {}",
                path.display(),
                ancestor.display()
            )));
        }
    }
    Ok(())
}

fn validate_sha256(value: &str, description: &str) -> RusticolResult<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(RusticolError::artifact(format!(
            "{description} must be 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

fn validate_public_id(value: &str, description: &str) -> RusticolResult<()> {
    let valid = !value.is_empty()
        && value.len() <= 255
        && value.bytes().enumerate().all(|(index, byte)| {
            byte.is_ascii_alphanumeric()
                || (index > 0 && matches!(byte, b'.' | b'_' | b':' | b'+' | b',' | b'~' | b'-'))
        });
    if !valid {
        return Err(RusticolError::artifact(format!(
            "invalid {description} {value:?}"
        )));
    }
    Ok(())
}

fn validate_datetime(value: &str) -> RusticolResult<()> {
    let bytes = value.as_bytes();
    let digits = |start: usize, length: usize| -> Option<u32> {
        let slice = bytes.get(start..start + length)?;
        if !slice.iter().all(u8::is_ascii_digit) {
            return None;
        }
        slice.iter().try_fold(0_u32, |value, digit| {
            value.checked_mul(10)?.checked_add(u32::from(*digit - b'0'))
        })
    };
    let year = digits(0, 4);
    let month = digits(5, 2);
    let day = digits(8, 2);
    let hour = digits(11, 2);
    let minute = digits(14, 2);
    let second = digits(17, 2);
    let date_valid = match (year, month, day) {
        (Some(year), Some(month @ 1..=12), Some(day)) => {
            let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
            let maximum = match month {
                2 if leap => 29,
                2 => 28,
                4 | 6 | 9 | 11 => 30,
                _ => 31,
            };
            (1..=maximum).contains(&day)
        }
        _ => false,
    };
    let time_valid = matches!(hour, Some(0..=23))
        && matches!(minute, Some(0..=59))
        && matches!(second, Some(0..=60));
    let separators_valid = bytes.get(4) == Some(&b'-')
        && bytes.get(7) == Some(&b'-')
        && matches!(bytes.get(10), Some(b'T' | b't'))
        && bytes.get(13) == Some(&b':')
        && bytes.get(16) == Some(&b':');
    let mut offset = 19;
    if bytes.get(offset) == Some(&b'.') {
        offset += 1;
        let fractional_start = offset;
        while bytes.get(offset).is_some_and(u8::is_ascii_digit) {
            offset += 1;
        }
        if offset == fractional_start {
            offset = bytes.len() + 1;
        }
    }
    let zone_valid = match bytes.get(offset) {
        Some(b'Z' | b'z') => offset + 1 == bytes.len(),
        Some(b'+' | b'-') => {
            offset + 6 == bytes.len()
                && bytes.get(offset + 3) == Some(&b':')
                && matches!(digits(offset + 1, 2), Some(0..=23))
                && matches!(digits(offset + 4, 2), Some(0..=59))
        }
        _ => false,
    };
    let valid = bytes.len() >= 20 && date_valid && time_valid && separators_valid && zone_valid;
    if !valid {
        return Err(RusticolError::artifact(format!(
            "created_utc {value:?} is not an RFC 3339 date-time"
        )));
    }
    Ok(())
}

fn validate_permutation(values: &[usize], size: usize, alias_id: &str) -> RusticolResult<()> {
    if values.len() != size {
        return Err(RusticolError::artifact(format!(
            "alias {alias_id:?} permutation has length {}, expected {size}",
            values.len()
        )));
    }
    let found = values.iter().copied().collect::<BTreeSet<_>>();
    let expected = (0..size).collect::<BTreeSet<_>>();
    if found != expected {
        return Err(RusticolError::artifact(format!(
            "alias {alias_id:?} permutation is not a complete zero-based permutation"
        )));
    }
    Ok(())
}

fn validate_runtime_capabilities(
    values: &[String],
    description: &str,
) -> RusticolResult<BTreeSet<String>> {
    if values.is_empty() {
        return Err(RusticolError::artifact(format!(
            "{description} must contain at least one capability"
        )));
    }
    let capabilities = values.iter().cloned().collect::<BTreeSet<_>>();
    if capabilities.len() != values.len() {
        return Err(RusticolError::artifact(format!(
            "{description} must not contain duplicates"
        )));
    }
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(RusticolError::artifact(format!(
            "{description} must be sorted"
        )));
    }
    let known = [
        RuntimeCapability::CompiledColorTopologyLanesV1,
        RuntimeCapability::CompiledHelicityDualLaneV1,
        RuntimeCapability::CompiledHelicityPrimaryRecurrenceV1,
        RuntimeCapability::CompiledHelicitySelectorUnionV1,
        RuntimeCapability::CompiledRuntimeSelectorsV1,
        RuntimeCapability::EagerDagComplexF64V1,
        RuntimeCapability::EagerLcTopologyReplayComplexF64V1,
        RuntimeCapability::SymjitApplicationComplexF64V1,
        RuntimeCapability::SymbolicaLegacyJitContainerComplexF64V1,
        RuntimeCapability::SymbolicaCompiledCppComplexF64V1,
        RuntimeCapability::SymbolicaCompiledAsmComplexF64V1,
    ]
    .map(RuntimeCapability::as_str)
    .into_iter()
    .collect::<BTreeSet<_>>();
    let unknown = capabilities
        .iter()
        .filter(|capability| !known.contains(capability.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        return Err(RusticolError::artifact(format!(
            "{description} contains unsupported capabilities: {}",
            unknown.join(", ")
        )));
    }
    Ok(capabilities)
}

fn validate_target(target: &Target, description: &str) -> RusticolResult<()> {
    let current = current_target_triple();
    if !SUPPORTED_ARTIFACT_TARGETS.contains(&current) {
        return Err(RusticolError::compatibility(format!(
            "Rusticol process artifacts are not supported on runtime target {current:?}"
        )));
    }
    if target.triple != current {
        return Err(RusticolError::compatibility(format!(
            "{description} target {:?} is incompatible with runtime target {current:?}",
            target.triple
        )));
    }
    let required = normalized_cpu_features(target, description)?;
    let available = detected_cpu_features().into_iter().collect::<BTreeSet<_>>();
    let unavailable = required.difference(&available).cloned().collect::<Vec<_>>();
    if !unavailable.is_empty() {
        return Err(RusticolError::compatibility(format!(
            "{description} requires unavailable CPU features {unavailable:?}"
        )));
    }
    Ok(())
}

fn validate_payload_target(
    producer: &Target,
    payload: &Target,
    payload_path: &str,
) -> RusticolResult<()> {
    if payload.triple != producer.triple {
        return Err(RusticolError::compatibility(format!(
            "payload {payload_path:?} target {:?} does not match producer target {:?}",
            payload.triple, producer.triple
        )));
    }
    let producer_features = normalized_cpu_features(producer, "producer")?;
    let payload_features = normalized_cpu_features(payload, &format!("payload {payload_path}"))?;
    if payload_features != producer_features {
        return Err(RusticolError::compatibility(format!(
            "payload {payload_path:?} CPU features {payload_features:?} do not match producer CPU features {producer_features:?}"
        )));
    }
    Ok(())
}

fn normalized_cpu_features(target: &Target, description: &str) -> RusticolResult<BTreeSet<String>> {
    let mut features = BTreeSet::new();
    let mut previous: Option<&str> = None;
    for feature in &target.cpu_features {
        let canonical = !feature.is_empty()
            && feature.bytes().enumerate().all(|(index, byte)| {
                byte.is_ascii_lowercase()
                    || byte.is_ascii_digit()
                    || (index > 0 && matches!(byte, b'.' | b'-'))
            });
        if !canonical {
            return Err(RusticolError::artifact(format!(
                "{description} target CPU feature {feature:?} is not a canonical feature ID"
            )));
        }
        if previous.is_some_and(|value| value >= feature.as_str()) {
            return Err(RusticolError::artifact(format!(
                "{description} target CPU features must be sorted and unique"
            )));
        }
        previous = Some(feature);
        features.insert(feature.clone());
    }
    Ok(features)
}

/// Return the current Rusticol target and every CPU feature it can verify at runtime.
///
/// An empty feature list on an artifact means the architecture's baseline ISA. Native
/// evaluator producers use this detected, canonical list as a conservative requirement.
pub fn runtime_target_info() -> Target {
    Target {
        triple: current_target_triple().to_string(),
        cpu_features: detected_cpu_features(),
    }
}

fn current_target_triple() -> &'static str {
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        "aarch64-apple-darwin"
    }
    #[cfg(all(target_arch = "x86_64", target_os = "macos"))]
    {
        "x86_64-apple-darwin"
    }
    #[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
    {
        "x86_64-unknown-linux-gnu"
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_os = "macos"),
        all(target_arch = "x86_64", target_os = "macos"),
        all(target_arch = "x86_64", target_os = "linux", target_env = "gnu")
    )))]
    {
        "unsupported-target"
    }
}

fn detected_cpu_features() -> Vec<String> {
    let mut features = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        let detected = [
            ("adx", std::is_x86_feature_detected!("adx")),
            ("aes", std::is_x86_feature_detected!("aes")),
            ("avx", std::is_x86_feature_detected!("avx")),
            ("avx2", std::is_x86_feature_detected!("avx2")),
            ("avx512bf16", std::is_x86_feature_detected!("avx512bf16")),
            (
                "avx512bitalg",
                std::is_x86_feature_detected!("avx512bitalg"),
            ),
            ("avx512bw", std::is_x86_feature_detected!("avx512bw")),
            ("avx512cd", std::is_x86_feature_detected!("avx512cd")),
            ("avx512dq", std::is_x86_feature_detected!("avx512dq")),
            ("avx512f", std::is_x86_feature_detected!("avx512f")),
            ("avx512ifma", std::is_x86_feature_detected!("avx512ifma")),
            ("avx512vbmi", std::is_x86_feature_detected!("avx512vbmi")),
            ("avx512vbmi2", std::is_x86_feature_detected!("avx512vbmi2")),
            ("avx512vl", std::is_x86_feature_detected!("avx512vl")),
            ("avx512vnni", std::is_x86_feature_detected!("avx512vnni")),
            (
                "avx512vpopcntdq",
                std::is_x86_feature_detected!("avx512vpopcntdq"),
            ),
            ("bmi1", std::is_x86_feature_detected!("bmi1")),
            ("bmi2", std::is_x86_feature_detected!("bmi2")),
            ("cmpxchg16b", std::is_x86_feature_detected!("cmpxchg16b")),
            ("f16c", std::is_x86_feature_detected!("f16c")),
            ("fma", std::is_x86_feature_detected!("fma")),
            ("fxsr", std::is_x86_feature_detected!("fxsr")),
            ("gfni", std::is_x86_feature_detected!("gfni")),
            ("lzcnt", std::is_x86_feature_detected!("lzcnt")),
            ("movbe", std::is_x86_feature_detected!("movbe")),
            ("pclmulqdq", std::is_x86_feature_detected!("pclmulqdq")),
            ("popcnt", std::is_x86_feature_detected!("popcnt")),
            ("rdrand", std::is_x86_feature_detected!("rdrand")),
            ("rdseed", std::is_x86_feature_detected!("rdseed")),
            ("rtm", std::is_x86_feature_detected!("rtm")),
            ("sha", std::is_x86_feature_detected!("sha")),
            ("sse", std::is_x86_feature_detected!("sse")),
            ("sse2", std::is_x86_feature_detected!("sse2")),
            ("sse3", std::is_x86_feature_detected!("sse3")),
            ("sse4.1", std::is_x86_feature_detected!("sse4.1")),
            ("sse4.2", std::is_x86_feature_detected!("sse4.2")),
            ("ssse3", std::is_x86_feature_detected!("ssse3")),
            ("vaes", std::is_x86_feature_detected!("vaes")),
            ("vpclmulqdq", std::is_x86_feature_detected!("vpclmulqdq")),
            ("xsave", std::is_x86_feature_detected!("xsave")),
            ("xsavec", std::is_x86_feature_detected!("xsavec")),
            ("xsaveopt", std::is_x86_feature_detected!("xsaveopt")),
            ("xsaves", std::is_x86_feature_detected!("xsaves")),
        ];
        features.extend(
            detected
                .into_iter()
                .filter(|(_, available)| *available)
                .map(|(name, _)| name.to_string()),
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        let detected = [
            ("aes", std::arch::is_aarch64_feature_detected!("aes")),
            ("bf16", std::arch::is_aarch64_feature_detected!("bf16")),
            ("bti", std::arch::is_aarch64_feature_detected!("bti")),
            ("crc", std::arch::is_aarch64_feature_detected!("crc")),
            ("dit", std::arch::is_aarch64_feature_detected!("dit")),
            (
                "dotprod",
                std::arch::is_aarch64_feature_detected!("dotprod"),
            ),
            ("dpb", std::arch::is_aarch64_feature_detected!("dpb")),
            ("dpb2", std::arch::is_aarch64_feature_detected!("dpb2")),
            ("f32mm", std::arch::is_aarch64_feature_detected!("f32mm")),
            ("f64mm", std::arch::is_aarch64_feature_detected!("f64mm")),
            ("fcma", std::arch::is_aarch64_feature_detected!("fcma")),
            ("fhm", std::arch::is_aarch64_feature_detected!("fhm")),
            ("flagm", std::arch::is_aarch64_feature_detected!("flagm")),
            ("fp", std::arch::is_aarch64_feature_detected!("fp")),
            ("fp16", std::arch::is_aarch64_feature_detected!("fp16")),
            (
                "frintts",
                std::arch::is_aarch64_feature_detected!("frintts"),
            ),
            ("i8mm", std::arch::is_aarch64_feature_detected!("i8mm")),
            ("jsconv", std::arch::is_aarch64_feature_detected!("jsconv")),
            ("lse", std::arch::is_aarch64_feature_detected!("lse")),
            ("lse2", std::arch::is_aarch64_feature_detected!("lse2")),
            ("mte", std::arch::is_aarch64_feature_detected!("mte")),
            ("neon", std::arch::is_aarch64_feature_detected!("neon")),
            ("paca", std::arch::is_aarch64_feature_detected!("paca")),
            ("pacg", std::arch::is_aarch64_feature_detected!("pacg")),
            ("rand", std::arch::is_aarch64_feature_detected!("rand")),
            ("rcpc", std::arch::is_aarch64_feature_detected!("rcpc")),
            ("rcpc2", std::arch::is_aarch64_feature_detected!("rcpc2")),
            ("rdm", std::arch::is_aarch64_feature_detected!("rdm")),
            ("sb", std::arch::is_aarch64_feature_detected!("sb")),
            ("sha2", std::arch::is_aarch64_feature_detected!("sha2")),
            ("sha3", std::arch::is_aarch64_feature_detected!("sha3")),
            ("sm4", std::arch::is_aarch64_feature_detected!("sm4")),
            ("ssbs", std::arch::is_aarch64_feature_detected!("ssbs")),
            ("sve", std::arch::is_aarch64_feature_detected!("sve")),
            ("sve2", std::arch::is_aarch64_feature_detected!("sve2")),
            (
                "sve2-aes",
                std::arch::is_aarch64_feature_detected!("sve2-aes"),
            ),
            (
                "sve2-bitperm",
                std::arch::is_aarch64_feature_detected!("sve2-bitperm"),
            ),
            (
                "sve2-sha3",
                std::arch::is_aarch64_feature_detected!("sve2-sha3"),
            ),
            (
                "sve2-sm4",
                std::arch::is_aarch64_feature_detected!("sve2-sm4"),
            ),
            ("tme", std::arch::is_aarch64_feature_detected!("tme")),
        ];
        features.extend(
            detected
                .into_iter()
                .filter(|(_, available)| *available)
                .map(|(name, _)| name.to_string()),
        );
    }
    features.sort();
    features
}

#[cfg(test)]
#[path = "artifact_tests.rs"]
pub(crate) mod tests;
