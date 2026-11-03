// SPDX-License-Identifier: 0BSD

//! Opt-in coherent Born correlations. Ordinary reductions never enter this lane.

use super::*;

/// One literal complex four-vector broadcasts; otherwise supply one per point.
pub type NativeSpinCorrelationVectors = BTreeMap<usize, Vec<[[f64; 2]; 4]>>;

#[derive(Clone, Debug)]
pub struct NativeCorrelatedRequest {
    pub color_correlation: String,
    /// None inherits the runtime setter; an empty map restores physical sources.
    pub spin_vectors: Option<NativeSpinCorrelationVectors>,
}

impl Default for NativeCorrelatedRequest {
    fn default() -> Self {
        Self {
            color_correlation: "born".into(),
            spin_vectors: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeCorrelatedEvaluation {
    pub point_count: usize,
    pub request_count: usize,
    /// Request-major: values[request * point_count + point].
    pub values: Vec<[f64; 2]>,
}

/// Spin vectors of [`NativeDecimalCorrelatedRequest`]: per leg, four complex
/// components `[real, imaginary]` as decimal strings, parsed at the working
/// precision. One vector broadcasts; otherwise supply one per point.
pub type NativeDecimalSpinCorrelationVectors = BTreeMap<usize, Vec<[[String; 2]; 4]>>;

/// One request of [`NativeRuntime::evaluate_correlated_many_with_precision`],
/// as [`NativeCorrelatedRequest`] with decimal spin vectors.
#[derive(Clone, Debug)]
pub struct NativeDecimalCorrelatedRequest {
    pub color_correlation: String,
    /// None inherits the runtime setter (its binary64 vectors, exactly); an
    /// empty map restores physical sources.
    pub spin_vectors: Option<NativeDecimalSpinCorrelationVectors>,
}

impl Default for NativeDecimalCorrelatedRequest {
    fn default() -> Self {
        Self {
            color_correlation: "born".into(),
            spin_vectors: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeDecimalCorrelatedEvaluation {
    pub point_count: usize,
    pub request_count: usize,
    /// Request-major: values[request * point_count + point] = [real, imaginary].
    pub values: Vec<[String; 2]>,
    pub decimal_digits: u32,
}

#[derive(Deserialize)]
struct Catalogue {
    schema_version: u32,
    complete_source_basis: bool,
    processes: BTreeMap<String, ProcessCatalogue>,
}

#[derive(Deserialize)]
struct ProcessCatalogue {
    declarations: Declarations,
    spin_legs: Vec<usize>,
    matrices: Vec<MatrixRecord>,
    coherent_groups: Vec<GroupRecord>,
}

#[derive(Deserialize)]
struct Declarations {
    color_correlations: Vec<Declaration>,
    spin_correlations: Vec<Vec<usize>>,
}

#[derive(Deserialize)]
struct Declaration {
    id: String,
    bra: Vec<Value>,
    ket: Vec<Value>,
}

#[derive(Deserialize)]
struct GroupRecord {
    group_id: i64,
    helicities: Vec<i32>,
    color_sector_id: i64,
    color_word: Vec<i64>,
}

#[derive(Deserialize)]
struct MatrixRecord {
    id: String,
    order: usize,
    bra: Vec<Value>,
    ket: Vec<Value>,
    convention: String,
    color_accuracy: String,
    output_legs: Value,
    sector_ids: Vec<i64>,
    storage: String,
    includes_color_factor: bool,
    entries: Vec<EntryRecord>,
}

#[derive(Deserialize)]
struct EntryRecord {
    left_sector_id: i64,
    right_sector_id: i64,
    weight: WeightRecord,
}

#[derive(Deserialize)]
struct WeightRecord {
    real: [Value; 2],
    imag: [Value; 2],
}

struct Matrix {
    id: String,
    sectors: BTreeSet<i64>,
    entries: Vec<(i64, i64, [f64; 2])>,
    /// The exact weights of `entries` (aligned): [real, imaginary] as
    /// [numerator, denominator] integer strings, for the double-double and
    /// arbitrary-precision contraction.
    #[cfg_attr(not(feature = "symbolica-runtime"), allow(dead_code))]
    exact_weights: Vec<[[String; 2]; 2]>,
}

struct Group {
    helicity_id: String,
    helicities: Vec<i32>,
    sector: i64,
    roots: Vec<usize>,
}

pub(super) struct CorrelatedRuntime {
    catalogue_json: String,
    matrices: Vec<Matrix>,
    spin_classes: BTreeSet<Vec<usize>>,
    known_helicities: BTreeSet<String>,
    groups: Vec<Group>,
    vectors: NativeSpinCorrelationVectors,
}

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

fn malformed(message: impl Into<String>) -> RusticolError {
    RusticolError::integrity(format!(
        "invalid correlated process output: {}",
        message.into()
    ))
}

fn validate_catalogue_artifact_id(loaded: &str, reopened: &str) -> RusticolResult<()> {
    if loaded != reopened {
        return Err(malformed(
            "process output changed since the runtime was loaded; reload the runtime before correlated evaluation",
        ));
    }
    Ok(())
}

/// A matrix coefficient's integer numerator or denominator as its decimal text.
fn exact_integer_text(value: &Value) -> RusticolResult<String> {
    let text = match value {
        Value::String(text) => text.clone(),
        Value::Number(number) if number.is_i64() || number.is_u64() => number.to_string(),
        _ => return Err(malformed("matrix coefficient is not an integer")),
    };
    let digits = text.strip_prefix('-').unwrap_or(&text);
    if digits.is_empty() || !digits.bytes().all(|value| value.is_ascii_digit()) {
        return Err(malformed("matrix coefficient is not an integer"));
    }
    Ok(text)
}

fn exact_rational_text(parts: &[Value; 2]) -> RusticolResult<[String; 2]> {
    Ok([
        exact_integer_text(&parts[0])?,
        exact_integer_text(&parts[1])?,
    ])
}

fn rational_f64(parts: &[Value; 2]) -> RusticolResult<f64> {
    let integer = |value: &Value| -> RusticolResult<f64> {
        exact_integer_text(value)?
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite())
            .ok_or_else(|| malformed("matrix coefficient exceeds binary64 range"))
    };
    let denominator = integer(&parts[1])?;
    if denominator == 0.0 {
        return Err(malformed("zero matrix denominator"));
    }
    let value = integer(&parts[0])? / denominator;
    if !value.is_finite() {
        return Err(malformed("matrix coefficient exceeds binary64 range"));
    }
    Ok(value)
}

impl CorrelatedRuntime {
    fn load(native: &NativeRuntime) -> RusticolResult<Self> {
        let identity = native
            .external_permutation
            .iter()
            .copied()
            .eq(0..native.runtime.external_count);
        if let Some(group) = native.process_group.as_deref()
            && (group.conjugated || !identity)
        {
            return Err(RusticolError::compatibility(format!(
                "correlated evaluation of process-group member {:?} through its representative {:?} is available only for an identity leg order without charge conjugation; generate the process explicitly",
                native.process_key, group.representative_process_key
            )));
        }
        if !matches!(native.execution_lane, NativeExecutionLane::Compiled) || !identity {
            return Err(RusticolError::compatibility(
                "correlations require compiled process output with identity external ordering",
            ));
        }
        let artifact = VerifiedArtifact::open(&native.root)?;
        validate_catalogue_artifact_id(&native.artifact_id, &artifact.manifest().artifact_id)?;
        let extension = artifact
            .manifest()
            .extensions
            .get("correlators")
            .ok_or_else(|| {
                RusticolError::compatibility("correlations must be declared at generation")
            })?;
        if extension.get("schema_version").and_then(Value::as_u64) != Some(1) {
            return Err(malformed("unsupported catalogue version"));
        }
        let path = extension
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed("missing catalogue path"))?;
        let mut payload: Catalogue = serde_json::from_slice(&artifact.read_payload(path)?)
            .map_err(|error| malformed(format!("catalogue: {error}")))?;
        if payload.schema_version != 1 || !payload.complete_source_basis {
            return Err(malformed("correlations require a complete source basis"));
        }
        let process = payload
            .processes
            .remove(&native.representative_process_id)
            .ok_or_else(|| malformed("selected process is absent from catalogue"))?;
        Self::from_process(process, &native.runtime, native.process_physics()?)
    }

    fn from_process(
        process: ProcessCatalogue,
        runtime: &ExecutionRuntime,
        physics: &ProcessPhysicsV1,
    ) -> RusticolResult<Self> {
        if runtime.lc_topology_replay_enabled
            || runtime.color_topology_replay_enabled
            || runtime.helicity_recurrence.is_some()
            || runtime.helicity_sum_runtime.is_some()
        {
            return Err(malformed(
                "correlated amplitudes must not use physical-helicity reductions",
            ));
        }
        let spin_classes: BTreeSet<_> =
            process.declarations.spin_correlations.into_iter().collect();
        let mut spin_legs = BTreeSet::new();
        for class in &spin_classes {
            if class.is_empty()
                || class.windows(2).any(|pair| pair[0] >= pair[1])
                || class
                    .iter()
                    .any(|&leg| leg == 0 || leg > runtime.external_count)
            {
                return Err(malformed("invalid joint spin-correlation class"));
            }
            spin_legs.extend(class.iter().copied());
        }
        if spin_legs.iter().copied().collect::<Vec<_>>() != process.spin_legs {
            return Err(malformed("inconsistent declared spin legs"));
        }
        for leg in spin_legs {
            let sources: Vec<_> = runtime
                .sources
                .iter()
                .filter(|source| source.leg_label == leg)
                .collect();
            if sources.is_empty()
                || sources.iter().any(|source| {
                    !matches!(
                        source.source_ir.wavefunction_family,
                        GenericWavefunctionFamilyManifest::Vector
                    ) || source.source_ir.component_dimension != 4
                        || source
                            .value_slot
                            .component_stop
                            .checked_sub(source.value_slot.component_start)
                            != Some(4)
                })
            {
                return Err(malformed(
                    "spin correlation requires retained four-component vector sources",
                ));
            }
        }
        let mut declarations = BTreeMap::from([("born".to_owned(), (Vec::new(), Vec::new()))]);
        for declaration in process.declarations.color_correlations {
            if declaration.id.is_empty()
                || declarations
                    .insert(declaration.id, (declaration.bra, declaration.ket))
                    .is_some()
            {
                return Err(malformed("repeated correlation declaration"));
            }
        }
        let mut matrices = Vec::new();
        let mut catalogue = Vec::new();
        for matrix in process.matrices {
            if matrix.convention != "su3-literal-tau-physical-charge-all-outgoing-v1"
                || !["lc", "nlc", "full"].contains(&matrix.color_accuracy.as_str())
                || matrix.storage != "sparse-directed"
                || !matrix.includes_color_factor
                || matrix.order != matrix.bra.len()
                || matrix.order != matrix.ket.len()
                || declarations.remove(&matrix.id) != Some((matrix.bra.clone(), matrix.ket.clone()))
            {
                return Err(malformed("unsupported or inconsistent colour matrix"));
            }
            let sectors: BTreeSet<_> = matrix.sector_ids.iter().copied().collect();
            if sectors.is_empty()
                || sectors.len() != matrix.sector_ids.len()
                || sectors.iter().any(|&id| id < 0)
            {
                return Err(malformed("invalid matrix sector IDs"));
            }
            let mut pairs = BTreeSet::new();
            let mut entries = Vec::new();
            let mut exact_weights = Vec::new();
            for entry in matrix.entries {
                if !sectors.contains(&entry.left_sector_id)
                    || !sectors.contains(&entry.right_sector_id)
                    || !pairs.insert((entry.left_sector_id, entry.right_sector_id))
                {
                    return Err(malformed("invalid or repeated matrix entry"));
                }
                entries.push((
                    entry.left_sector_id,
                    entry.right_sector_id,
                    [
                        rational_f64(&entry.weight.real)?,
                        rational_f64(&entry.weight.imag)?,
                    ],
                ));
                exact_weights.push([
                    exact_rational_text(&entry.weight.real)?,
                    exact_rational_text(&entry.weight.imag)?,
                ]);
            }
            catalogue.push(serde_json::json!({"id":matrix.id,"order":matrix.order,"bra":matrix.bra,"ket":matrix.ket,
                "color_accuracy":matrix.color_accuracy,"output_legs":matrix.output_legs,"sector_ids":matrix.sector_ids}));
            matrices.push(Matrix {
                id: matrix.id,
                sectors,
                entries,
                exact_weights,
            });
        }
        if !declarations.is_empty() {
            return Err(malformed("missing colour matrices"));
        }
        let mut known_helicities = BTreeSet::new();
        let mut helicities = BTreeMap::new();
        let mut computed = BTreeSet::new();
        for helicity in &physics.helicities {
            if helicity.id != helicity.representative_id
                || helicity.values.len() != runtime.external_count
                || helicity.computed == helicity.structural_zero
                || helicity.coefficient != if helicity.computed { 1.0 } else { 0.0 }
                || !known_helicities.insert(helicity.id.clone())
                || helicities
                    .insert(helicity.values.clone(), helicity.id.clone())
                    .is_some()
            {
                return Err(malformed(
                    "correlations require complete unquotiented helicities",
                ));
            }
            if helicity.computed {
                computed.insert(helicity.id.clone());
            }
        }
        let amplitude = runtime
            .amplitude_stage
            .as_ref()
            .ok_or_else(|| malformed("missing amplitude stage"))?;
        if amplitude
            .raw_sum_weights
            .iter()
            .any(|&weight| weight != 1.0)
            || amplitude
                .raw_sum_all_sector_weights
                .iter()
                .any(|&weight| weight != 1.0)
        {
            return Err(malformed("correlated roots have folded weights"));
        }
        let raw: BTreeMap<_, _> = amplitude
            .raw_sum_groups
            .iter()
            .map(|group| (group.id, group))
            .collect();
        let mut seen = BTreeSet::new();
        let mut roots = BTreeSet::new();
        let mut groups: Vec<Group> = Vec::new();
        let mut physical = BTreeMap::new();
        let mut words = BTreeMap::new();
        for record in process.coherent_groups {
            let helicity_id = helicities
                .get(&record.helicities)
                .filter(|id| computed.contains(*id))
                .ok_or_else(|| malformed("unknown coherent helicity"))?;
            let group = raw
                .get(&record.group_id)
                .ok_or_else(|| malformed("unknown coherent group"))?;
            if !seen.insert(record.group_id)
                || group.weight != 1.0
                || group.all_sector_weight != 1.0
            {
                return Err(malformed("repeated or folded coherent group"));
            }
            if group
                .indices
                .iter()
                .any(|&index| index >= runtime.amplitude_output_count || !roots.insert(index))
            {
                return Err(malformed("invalid coherent root mapping"));
            }
            let key = (record.helicities.clone(), record.color_sector_id);
            let position = if let Some(&position) = physical.get(&key) {
                if words.get(&key) != Some(&record.color_word) {
                    return Err(malformed("inconsistent coherent colour word"));
                }
                position
            } else {
                let position = groups.len();
                physical.insert(key.clone(), position);
                words.insert(key, record.color_word);
                groups.push(Group {
                    helicity_id: helicity_id.clone(),
                    helicities: record.helicities,
                    sector: record.color_sector_id,
                    roots: Vec::new(),
                });
                position
            };
            groups[position].roots.extend(group.indices.iter().copied());
        }
        if seen.len() != raw.len()
            || roots.len() != runtime.amplitude_output_count
            || groups
                .iter()
                .map(|group| group.helicity_id.clone())
                .collect::<BTreeSet<_>>()
                != computed
        {
            return Err(malformed("incomplete coherent amplitude coverage"));
        }
        let mut basis: BTreeMap<&str, BTreeSet<i64>> = BTreeMap::new();
        for group in &groups {
            basis
                .entry(&group.helicity_id)
                .or_default()
                .insert(group.sector);
        }
        if matrices
            .iter()
            .any(|matrix| basis.values().any(|sectors| sectors != &matrix.sectors))
        {
            return Err(malformed(
                "coherent groups do not cover the colour matrices",
            ));
        }
        let catalogue_json =
            serde_json::to_string(&catalogue).map_err(|error| malformed(error.to_string()))?;
        Ok(Self {
            catalogue_json,
            matrices,
            spin_classes,
            known_helicities,
            groups,
            vectors: BTreeMap::new(),
        })
    }

    fn validate_spin_class<V>(&self, vectors: &BTreeMap<usize, V>) -> RusticolResult<()> {
        let legs: Vec<_> = vectors.keys().copied().collect();
        if !legs.is_empty() && !self.spin_classes.contains(&legs) {
            return Err(invalid(
                "this joint spin-correlation class was not declared at generation",
            ));
        }
        Ok(())
    }

    fn validate_vectors(
        &self,
        vectors: &NativeSpinCorrelationVectors,
        point_count: Option<usize>,
    ) -> RusticolResult<()> {
        self.validate_spin_class(vectors)?;
        for batch in vectors.values() {
            if batch.is_empty()
                || point_count.is_some_and(|count| batch.len() != 1 && batch.len() != count)
                || batch
                    .iter()
                    .flatten()
                    .flatten()
                    .any(|value| !value.is_finite())
            {
                return Err(invalid(
                    "spin vectors must be finite, with one vector or one per phase-space point",
                ));
            }
        }
        Ok(())
    }

    /// The groups of the selected helicities, one placeholder helicity per
    /// replaced leg (the keys of `vectors`).
    fn selected_groups<V>(
        &self,
        ids: Option<&[String]>,
        vectors: &BTreeMap<usize, V>,
    ) -> Vec<usize> {
        let selected = |group: &Group| ids.is_none_or(|ids| ids.contains(&group.helicity_id));
        let spectator = |group: &Group| {
            group
                .helicities
                .iter()
                .enumerate()
                .filter(|(leg, _)| !vectors.contains_key(&(leg + 1)))
                .map(|(_, &value)| value)
                .collect::<Vec<_>>()
        };
        let mut representatives: BTreeMap<Vec<i32>, &Vec<i32>> = BTreeMap::new();
        for group in self.groups.iter().filter(|group| selected(group)) {
            representatives
                .entry(spectator(group))
                .and_modify(|values| {
                    if group.helicities < **values {
                        *values = &group.helicities;
                    }
                })
                .or_insert(&group.helicities);
        }
        self.groups
            .iter()
            .enumerate()
            .filter(|(_, group)| {
                selected(group)
                    && representatives.get(&spectator(group)) == Some(&&group.helicities)
            })
            .map(|(index, _)| index)
            .collect()
    }
}

impl NativeRuntime {
    fn ensure_correlated(&mut self) -> RusticolResult<()> {
        if self.correlated.is_none() {
            self.correlated = Some(CorrelatedRuntime::load(self)?);
        }
        Ok(())
    }

    pub fn available_color_correlations_json(&mut self) -> RusticolResult<String> {
        self.ensure_correlated()?;
        Ok(self
            .correlated
            .as_ref()
            .expect("initialized")
            .catalogue_json
            .clone())
    }

    pub fn color_correlation_ids(&mut self) -> RusticolResult<Vec<String>> {
        self.ensure_correlated()?;
        Ok(self
            .correlated
            .as_ref()
            .expect("initialized")
            .matrices
            .iter()
            .map(|matrix| matrix.id.clone())
            .collect())
    }

    pub fn set_spin_correlation_vectors_f64(
        &mut self,
        vectors: NativeSpinCorrelationVectors,
    ) -> RusticolResult<()> {
        self.ensure_correlated()?;
        let correlated = self.correlated.as_mut().expect("initialized");
        correlated.validate_vectors(&vectors, None)?;
        correlated.vectors = vectors;
        Ok(())
    }

    /// Evaluate all requested complex bilinears, sharing each spin assignment's amplitudes.
    pub fn evaluate_correlated_many_f64(
        &mut self,
        momenta: &[f64],
        point_count: usize,
        requests: &[NativeCorrelatedRequest],
        helicity_ids: Option<&[String]>,
    ) -> RusticolResult<NativeCorrelatedEvaluation> {
        self.ensure_correlated()?;
        #[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
        {
            let mut evaluation = evaluate_many(self, momenta, point_count, requests, helicity_ids)?;
            self.apply_process_group_factor(evaluation.values.as_flattened_mut());
            Ok(evaluation)
        }
        #[cfg(not(any(feature = "f64-compiled", feature = "f64-symjit")))]
        {
            let _ = (momenta, point_count, requests, helicity_ids);
            Err(RusticolError::compatibility(
                "native correlations require binary64 compiled-stage support",
            ))
        }
    }
}

#[cfg(any(feature = "f64-compiled", feature = "f64-symjit"))]
fn evaluate_many(
    native: &mut NativeRuntime,
    momenta: &[f64],
    point_count: usize,
    requests: &[NativeCorrelatedRequest],
    helicity_ids: Option<&[String]>,
) -> RusticolResult<NativeCorrelatedEvaluation> {
    let expected = point_count
        .checked_mul(native.runtime.external_count)
        .and_then(|n| n.checked_mul(4));
    if point_count == 0
        || expected != Some(momenta.len())
        || momenta.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "correlated momenta must be finite with shape [point][external leg][4]",
        ));
    }
    let correlated = native.correlated.as_ref().expect("initialized");
    if helicity_ids.is_some_and(|ids| {
        ids.iter()
            .any(|id| !correlated.known_helicities.contains(id))
    }) {
        return Err(invalid("unknown correlated helicity ID"));
    }
    let size = point_count
        .checked_mul(requests.len())
        .ok_or_else(|| invalid("correlated result shape overflow"))?;
    let mut assignments: Vec<&NativeSpinCorrelationVectors> = Vec::new();
    let mut keys = BTreeMap::new();
    let mut routes = Vec::new();
    for request in requests {
        let matrix = correlated
            .matrices
            .iter()
            .position(|matrix| matrix.id == request.color_correlation)
            .ok_or_else(|| {
                invalid(format!(
                    "unknown colour correlation ID {:?}",
                    request.color_correlation
                ))
            })?;
        let vectors = request.spin_vectors.as_ref().unwrap_or(&correlated.vectors);
        correlated.validate_vectors(vectors, Some(point_count))?;
        // Canonicalize broadcast and per-point forms; preserve signed zero.
        let key: Vec<_> = vectors
            .iter()
            .map(|(&leg, batch)| {
                (
                    leg,
                    (0..point_count)
                        .flat_map(|point| {
                            batch[if batch.len() == 1 { 0 } else { point }]
                                .into_iter()
                                .flatten()
                                .map(f64::to_bits)
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
        let assignment = *keys.entry(key).or_insert_with(|| {
            assignments.push(vectors);
            assignments.len() - 1
        });
        routes.push((assignment, matrix));
    }
    let mut values = vec![[0.0; 2]; size];
    if requests.is_empty() {
        return Ok(NativeCorrelatedEvaluation {
            point_count,
            request_count: 0,
            values,
        });
    }
    let selected: Vec<_> = assignments
        .iter()
        .map(|vectors| correlated.selected_groups(helicity_ids, vectors))
        .collect();
    let runtime = &mut native.runtime;
    let direct = runtime.compiled_direct_runtime.as_mut().ok_or_else(|| {
        RusticolError::compatibility(
            "native correlations require retained generic Direct-Arena compiled stages",
        )
    })?;
    let batch = F64MomentumBatchView::from_contiguous_prevalidated(
        momenta,
        point_count,
        runtime.external_count,
        None,
    )?;
    let mut point_start = 0;
    while point_start < point_count {
        let point_stop = (point_start + direct.tile_capacity()).min(point_count);
        let count = point_stop - point_start;
        // Memo lifetime is exactly one point tile: momenta and parameters cannot change.
        let mut memo = compiled_direct_prototype::CorrelatedStageMemo::default();
        for (assignment, vectors) in assignments.iter().enumerate() {
            direct.begin_tile_from_inputs(
                batch.subview(point_start, point_stop)?,
                &runtime.sources,
                None,
                runtime.external_count,
                &runtime.particle_masses,
                &runtime.momentum_slots,
                &runtime.external_is_initial,
                &runtime.model_parameter_values_f64,
            )?;
            direct.replace_correlated_sources(&runtime.sources, vectors, point_start, count)?;
            if assignments.len() > 1 {
                direct.evaluate_correlated_reusing(count, &mut memo)?;
            } else {
                direct.evaluate_all(count)?;
            }
            let planes = direct.amplitude_planes()?;
            let mut amplitudes = vec![[0.0; 2]; selected[assignment].len() * count];
            let mut by_helicity: BTreeMap<&str, BTreeMap<i64, usize>> = BTreeMap::new();
            for (position, &index) in selected[assignment].iter().enumerate() {
                let group = &correlated.groups[index];
                by_helicity
                    .entry(&group.helicity_id)
                    .or_default()
                    .insert(group.sector, position);
                for &root in &group.roots {
                    let (re, im) = planes.plane_unchecked(root);
                    for point in 0..count {
                        amplitudes[position * count + point][0] += re[point];
                        amplitudes[position * count + point][1] += im[point];
                    }
                }
            }
            for (request, &(_, matrix)) in routes
                .iter()
                .enumerate()
                .filter(|(_, (id, _))| *id == assignment)
            {
                for sectors in by_helicity.values() {
                    for &(left, right, weight) in &correlated.matrices[matrix].entries {
                        let a = sectors[&left] * count;
                        let b = sectors[&right] * count;
                        for point in 0..count {
                            let [ar, ai] = amplitudes[a + point];
                            let [br, bi] = amplitudes[b + point];
                            let wr = ar * weight[0] + ai * weight[1];
                            let wi = ar * weight[1] - ai * weight[0];
                            let value = &mut values[request * point_count + point_start + point];
                            value[0] += wr * br - wi * bi;
                            value[1] += wr * bi + wi * br;
                        }
                    }
                }
                for value in &mut values
                    [request * point_count + point_start..request * point_count + point_stop]
                {
                    value[0] *= runtime.normalization_factor;
                    value[1] *= runtime.normalization_factor;
                }
            }
        }
        point_start = point_stop;
    }
    Ok(NativeCorrelatedEvaluation {
        point_count,
        request_count: requests.len(),
        values,
    })
}

/// Guard bits of the wider `Float` in which exact colour weights are formed
/// before they are rounded to the working precision.
#[cfg(feature = "symbolica-runtime")]
const EXACT_WEIGHT_GUARD_BITS: u32 = 64;

#[cfg(feature = "symbolica-runtime")]
impl NativeRuntime {
    /// [`Self::evaluate_correlated_many_f64`] in double-double
    /// (`decimal_digits == 32`) or arbitrary precision (`ceil(decimal_digits ·
    /// log2 10)` bits; 16 selects binary64), with decimal momenta
    /// (`[point][external leg][4]`) and spin vectors, as
    /// [`Self::evaluate_with_precision`]. The stages, the spin-vector source
    /// replacement, the exact colour weights of the declared correlations and
    /// the contraction run at the working precision; the normalization and the
    /// process-group factor are the artifact's, as in
    /// [`Self::evaluate_with_precision`]. Values are request-major
    /// (`values[request * point_count + point]`), formatted with
    /// `decimal_digits` digits.
    pub fn evaluate_correlated_many_with_precision(
        &mut self,
        momenta: &[String],
        point_count: usize,
        decimal_digits: u32,
        requests: &[NativeDecimalCorrelatedRequest],
        helicity_ids: Option<&[String]>,
    ) -> RusticolResult<NativeDecimalCorrelatedEvaluation> {
        if decimal_digits == 0 {
            return Err(RusticolError::unsupported_precision(
                "precision must be a positive number of decimal digits",
            ));
        }
        self.ensure_correlated()?;
        if point_count == 0 {
            return Err(invalid(
                "correlated momenta must be finite with shape [point][external leg][4]",
            ));
        }
        let values = if decimal_digits == 16 {
            let parse = |value: &String| {
                value.parse::<f64>().map_err(|error| {
                    invalid(format!(
                        "could not parse f64 correlated input {value:?}: {error}"
                    ))
                })
            };
            let momenta = momenta
                .iter()
                .map(parse)
                .collect::<RusticolResult<Vec<_>>>()?;
            let requests = requests
                .iter()
                .map(|request| {
                    let spin_vectors = request
                        .spin_vectors
                        .as_ref()
                        .map(|vectors| {
                            vectors
                                .iter()
                                .map(|(&leg, batch)| {
                                    let batch = batch
                                        .iter()
                                        .map(|vector| {
                                            let mut parsed = [[0.0; 2]; 4];
                                            for (target, [re, im]) in parsed.iter_mut().zip(vector)
                                            {
                                                *target = [parse(re)?, parse(im)?];
                                            }
                                            Ok(parsed)
                                        })
                                        .collect::<RusticolResult<Vec<_>>>()?;
                                    Ok((leg, batch))
                                })
                                .collect::<RusticolResult<NativeSpinCorrelationVectors>>()
                        })
                        .transpose()?;
                    Ok(NativeCorrelatedRequest {
                        color_correlation: request.color_correlation.clone(),
                        spin_vectors,
                    })
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            let evaluation =
                self.evaluate_correlated_many_f64(&momenta, point_count, &requests, helicity_ids)?;
            native_runtime::format_decimal_values(
                evaluation.values.into_iter().flatten().collect(),
                decimal_digits,
            )
        } else if decimal_digits == 32 {
            self.ensure_exact_model_parameters()?;
            let batch = self.prepare_double_batch(momenta, point_count)?;
            let mut values =
                evaluate_many_generic::<DoubleFloat>(self, &batch, requests, helicity_ids, None)?;
            self.apply_process_group_factor_high_precision(&mut values, None);
            native_runtime::format_decimal_values(values, decimal_digits)
        } else {
            self.ensure_exact_model_parameters()?;
            let binary_precision = decimal_digits_to_bits(decimal_digits);
            let batch = self.prepare_float_batch(momenta, point_count, binary_precision)?;
            let mut values = evaluate_many_generic::<Float>(
                self,
                &batch,
                requests,
                helicity_ids,
                Some(binary_precision),
            )?;
            self.apply_process_group_factor_high_precision(&mut values, Some(binary_precision));
            native_runtime::format_decimal_values(values, decimal_digits)
        };
        let mut values = values.into_iter();
        let values = std::iter::from_fn(|| Some([values.next()?, values.next()?])).collect();
        Ok(NativeDecimalCorrelatedEvaluation {
            point_count,
            request_count: requests.len(),
            values,
            decimal_digits,
        })
    }
}

/// Spin vectors of one assignment at the working precision, per leg.
#[cfg(feature = "symbolica-runtime")]
type GenericSpinVectors<T> = BTreeMap<usize, Vec<[Complex<T>; 4]>>;

/// The generic-precision counterpart of [`evaluate_many`]: flattened
/// request-major `[real, imaginary]` values, before the process-group factor.
#[cfg(feature = "symbolica-runtime")]
fn evaluate_many_generic<T>(
    native: &mut NativeRuntime,
    batch: &[Vec<[T; 4]>],
    requests: &[NativeDecimalCorrelatedRequest],
    helicity_ids: Option<&[String]>,
    binary_precision: Option<u32>,
) -> RusticolResult<Vec<T>>
where
    T: RusticolHighPrecisionNumber,
    Complex<T>: Real + EvaluationDomain,
{
    let point_count = batch.len();
    if batch
        .iter()
        .flatten()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(invalid(
            "correlated momenta must be finite with shape [point][external leg][4]",
        ));
    }
    let correlated = native.correlated.as_ref().expect("initialized");
    if helicity_ids.is_some_and(|ids| {
        ids.iter()
            .any(|id| !correlated.known_helicities.contains(id))
    }) {
        return Err(invalid("unknown correlated helicity ID"));
    }
    // Spin-vector components at the working precision (106 bits for
    // double-double): decimal inputs keep their digits, the setter's binary64
    // vectors convert exactly.
    let bits = binary_precision.unwrap_or(106);
    let parse = |text: &str| -> RusticolResult<T> {
        Float::parse(text, Some(bits))
            .ok()
            .filter(Float::is_finite)
            .map(|value| T::from_float(&value, binary_precision))
            .ok_or_else(|| {
                invalid(format!(
                    "spin-vector component {text:?} is not a finite decimal number"
                ))
            })
    };
    let mut assignments: Vec<GenericSpinVectors<T>> = Vec::new();
    let mut keys: BTreeMap<Vec<(usize, Vec<String>)>, usize> = BTreeMap::new();
    let mut routes = Vec::new();
    for request in requests {
        let matrix = correlated
            .matrices
            .iter()
            .position(|matrix| matrix.id == request.color_correlation)
            .ok_or_else(|| {
                invalid(format!(
                    "unknown colour correlation ID {:?}",
                    request.color_correlation
                ))
            })?;
        // Canonicalize broadcast and per-point forms: the key repeats each
        // point's input text (binary64 setter values by their bits).
        let per_point = |len: usize, point: usize| if len == 1 { 0 } else { point };
        let (key, vectors): (Vec<_>, GenericSpinVectors<T>) = match &request.spin_vectors {
            Some(decimal) => {
                correlated.validate_spin_class(decimal)?;
                let mut key = Vec::new();
                let mut vectors = BTreeMap::new();
                for (&leg, values) in decimal {
                    if values.is_empty() || (values.len() != 1 && values.len() != point_count) {
                        return Err(invalid(
                            "spin vectors must be finite, with one vector or one per phase-space point",
                        ));
                    }
                    key.push((
                        leg,
                        (0..point_count)
                            .flat_map(|point| {
                                values[per_point(values.len(), point)]
                                    .iter()
                                    .flatten()
                                    .cloned()
                            })
                            .collect(),
                    ));
                    let parsed = values
                        .iter()
                        .map(|vector| {
                            let mut components = Vec::with_capacity(4);
                            for [re, im] in vector {
                                components.push(Complex::new(parse(re)?, parse(im)?));
                            }
                            Ok(components
                                .try_into()
                                .unwrap_or_else(|_| unreachable!("four components")))
                        })
                        .collect::<RusticolResult<Vec<[Complex<T>; 4]>>>()?;
                    vectors.insert(leg, parsed);
                }
                (key, vectors)
            }
            None => {
                correlated.validate_vectors(&correlated.vectors, Some(point_count))?;
                let key = correlated
                    .vectors
                    .iter()
                    .map(|(&leg, values)| {
                        (
                            leg,
                            (0..point_count)
                                .flat_map(|point| {
                                    values[per_point(values.len(), point)]
                                        .iter()
                                        .flatten()
                                        .map(|value| format!("0x{:016x}", value.to_bits()))
                                })
                                .collect(),
                        )
                    })
                    .collect();
                let vectors = correlated
                    .vectors
                    .iter()
                    .map(|(&leg, values)| {
                        let converted = values
                            .iter()
                            .map(|vector| {
                                vector.map(|[re, im]| {
                                    Complex::new(
                                        T::from_f64_constant(re, binary_precision),
                                        T::from_f64_constant(im, binary_precision),
                                    )
                                })
                            })
                            .collect();
                        (leg, converted)
                    })
                    .collect();
                (key, vectors)
            }
        };
        let assignment = *keys.entry(key).or_insert_with(|| {
            assignments.push(vectors);
            assignments.len() - 1
        });
        routes.push((assignment, matrix));
    }
    let mut values = vec![T::new_zero(); 2 * point_count * requests.len()];
    if requests.is_empty() {
        return Ok(values);
    }
    // Exact colour weights at the working precision, once per used matrix.
    let weight_bits = bits + EXACT_WEIGHT_GUARD_BITS;
    let rational = |[numerator, denominator]: &[String; 2]| -> RusticolResult<T> {
        let parse = |text: &str| {
            Float::parse(text, Some(weight_bits))
                .map_err(|error| malformed(format!("matrix coefficient {text:?}: {error}")))
        };
        Ok(T::from_float(
            &(parse(numerator)? / &parse(denominator)?),
            binary_precision,
        ))
    };
    let mut weights: BTreeMap<usize, Vec<(i64, i64, T, T)>> = BTreeMap::new();
    for &(_, matrix) in &routes {
        if weights.contains_key(&matrix) {
            continue;
        }
        let record = &correlated.matrices[matrix];
        let entries = record
            .entries
            .iter()
            .zip(&record.exact_weights)
            .map(|(&(left, right, _), [real, imag])| {
                Ok((left, right, rational(real)?, rational(imag)?))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        weights.insert(matrix, entries);
    }
    let selected: Vec<_> = assignments
        .iter()
        .map(|vectors| correlated.selected_groups(helicity_ids, vectors))
        .collect();
    let runtime = &mut native.runtime;
    let normalization = T::from_f64_constant(runtime.normalization_factor, binary_precision);
    for (assignment, vectors) in assignments.iter().enumerate() {
        // Rank-one replacement of every source of a replaced leg, with the
        // source's crossing phase (±1 or ±i, exact), as
        // `replace_correlated_sources` of the binary64 lane.
        let replacements: Vec<_> = runtime
            .sources
            .iter()
            .filter_map(|source| {
                let batch = vectors.get(&source.leg_label)?;
                let [pr, pi] = source.applied_crossing.phase;
                Some((
                    source.value_slot.component_start,
                    source.value_slot.component_stop,
                    batch,
                    T::from_f64_constant(pr, binary_precision),
                    T::from_f64_constant(pi, binary_precision),
                ))
            })
            .collect();
        let (state, _profile) = runtime.evaluate_generic_stages_with_sources(
            batch,
            binary_precision,
            |point, row| {
                for (start, stop, batch, pr, pi) in &replacements {
                    if stop.checked_sub(*start) != Some(4) || *stop > row.len() {
                        return Err(malformed(
                            "spin correlation requires retained four-component vector sources",
                        ));
                    }
                    let vector = &batch[if batch.len() == 1 { 0 } else { point }];
                    for (target, value) in row[*start..*stop].iter_mut().zip(vector) {
                        *target = Complex::new(
                            value.re.clone() * pr.clone() - value.im.clone() * pi.clone(),
                            value.re.clone() * pi.clone() + value.im.clone() * pr.clone(),
                        );
                    }
                }
                Ok(())
            },
        )?;
        let (outputs, _, _) = runtime
            .amplitude_stage
            .as_mut()
            .ok_or_else(|| malformed("missing amplitude stage"))?
            .evaluate_outputs_generic(point_count, &state, binary_precision)?;
        let output_length = outputs.len() / point_count;
        let mut amplitudes = vec![complex_zero::<T>(); selected[assignment].len() * point_count];
        let mut by_helicity: BTreeMap<&str, BTreeMap<i64, usize>> = BTreeMap::new();
        for (position, &index) in selected[assignment].iter().enumerate() {
            let group = &correlated.groups[index];
            by_helicity
                .entry(&group.helicity_id)
                .or_default()
                .insert(group.sector, position);
            for &root in &group.roots {
                for point in 0..point_count {
                    let value = &outputs[point * output_length + root];
                    let amplitude = &mut amplitudes[position * point_count + point];
                    amplitude.re += value.re.clone();
                    amplitude.im += value.im.clone();
                }
            }
        }
        for (request, &(_, matrix)) in routes
            .iter()
            .enumerate()
            .filter(|(_, (id, _))| *id == assignment)
        {
            for point in 0..point_count {
                let (mut real, mut imag) = (T::new_zero(), T::new_zero());
                for sectors in by_helicity.values() {
                    for (left, right, wr, wi) in &weights[&matrix] {
                        let a = &amplitudes[sectors[left] * point_count + point];
                        let b = &amplitudes[sectors[right] * point_count + point];
                        // conj(a) w b
                        let xr = a.re.clone() * wr.clone() + a.im.clone() * wi.clone();
                        let xi = a.re.clone() * wi.clone() - a.im.clone() * wr.clone();
                        real += xr.clone() * b.re.clone() - xi.clone() * b.im.clone();
                        imag += xr * b.im.clone() + xi * b.re.clone();
                    }
                }
                let slot = 2 * (request * point_count + point);
                values[slot] = real * normalization.clone();
                values[slot + 1] = imag * normalization.clone();
            }
        }
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lazy_catalogue_cannot_bind_to_replaced_process_output() {
        assert!(validate_catalogue_artifact_id("loaded-output", "loaded-output").is_ok());
        assert!(validate_catalogue_artifact_id("loaded-output", "replacement-output").is_err());
    }

    #[test]
    fn exact_weights_reject_lossy_or_nonfinite_encodings() {
        assert_eq!(
            rational_f64(&[Value::from("-3"), Value::from("2")]).unwrap(),
            -1.5
        );
        for parts in [
            [Value::from("1e2"), Value::from(1)],
            [Value::from(1.2), Value::from(1)],
            [Value::from(1), Value::from(0)],
        ] {
            assert!(rational_f64(&parts).is_err());
        }
    }

    #[test]
    fn spin_validation_is_joint_finite_and_shape_checked() {
        let runtime = CorrelatedRuntime {
            catalogue_json: "[]".into(),
            matrices: vec![],
            spin_classes: BTreeSet::from([vec![3], vec![3, 4]]),
            known_helicities: BTreeSet::new(),
            groups: vec![],
            vectors: BTreeMap::new(),
        };
        let vector = [[1.0, 0.0]; 4];
        assert!(
            runtime
                .validate_vectors(&BTreeMap::from([(3, vec![vector])]), Some(2))
                .is_ok()
        );
        assert!(
            runtime
                .validate_vectors(&BTreeMap::from([(4, vec![vector])]), Some(2))
                .is_err()
        );
        assert!(
            runtime
                .validate_vectors(&BTreeMap::from([(3, vec![vector; 3])]), Some(2))
                .is_err()
        );
        assert!(
            runtime
                .validate_vectors(&BTreeMap::from([(3, vec![[[f64::NAN, 0.0]; 4]])]), None)
                .is_err()
        );
    }

    #[test]
    fn spin_replacements_select_one_placeholder_and_sum_spectators() {
        let mut runtime = CorrelatedRuntime {
            catalogue_json: "[]".into(),
            matrices: vec![],
            spin_classes: BTreeSet::from([vec![1]]),
            known_helicities: BTreeSet::new(),
            groups: vec![],
            vectors: BTreeMap::new(),
        };
        for h1 in [-1, 1] {
            for h2 in [-1, 1] {
                for sector in [0, 2] {
                    runtime.groups.push(Group {
                        helicity_id: format!("{h1},{h2}"),
                        helicities: vec![h1, h2],
                        sector,
                        roots: vec![],
                    });
                }
            }
        }
        assert_eq!(
            runtime
                .selected_groups(None, &NativeSpinCorrelationVectors::new())
                .len(),
            8
        );
        let vectors = BTreeMap::from([(1, vec![[[0.0, 0.0]; 4]])]);
        let selected = runtime.selected_groups(None, &vectors);
        assert_eq!(selected.len(), 4);
        assert!(
            selected
                .iter()
                .all(|&index| runtime.groups[index].helicities[0] == -1)
        );
        let ids = vec!["1,-1".to_owned()];
        let selected = runtime.selected_groups(Some(&ids), &vectors);
        assert_eq!(selected.len(), 2);
        assert!(
            selected
                .iter()
                .all(|&index| runtime.groups[index].helicities == [1, -1])
        );
    }
}
