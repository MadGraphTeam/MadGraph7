// SPDX-License-Identifier: 0BSD

use super::*;

fn process_string<'a>(
    process: &'a crate::recurrence::process::OwnedRecurrenceProcessInput,
    string_id: u32,
    label: &str,
) -> RusticolResult<&'a str> {
    let range = process
        .string_ranges
        .get(string_id as usize)
        .copied()
        .ok_or_else(|| integrity(format!("{label} string is absent")))?;
    let bytes = &process.string_bytes[range.as_usize_range(process.string_bytes.len(), label)?];
    std::str::from_utf8(bytes)
        .map_err(|error| integrity(format!("{label} string is not UTF-8: {error}")))
}

fn process_factor(
    process: &crate::recurrence::process::OwnedRecurrenceProcessInput,
    factor_id: u32,
    label: &str,
) -> RusticolResult<ExactComplexRational> {
    let row = process
        .exact_factors
        .get(factor_id as usize)
        .ok_or_else(|| integrity(format!("{label} factor is absent")))?;
    if row.id != factor_id {
        return Err(integrity(format!(
            "{label} factor catalog is not canonical"
        )));
    }
    ExactComplexRational::parse_parts(
        process_string(process, row.real_numerator_string_id, label)?,
        process_string(process, row.real_denominator_string_id, label)?,
        process_string(process, row.imag_numerator_string_id, label)?,
        process_string(process, row.imag_denominator_string_id, label)?,
    )
    .map_err(|error| integrity(format!("{label} exact factor is invalid: {error}")))
}

fn process_sequence<'a>(
    process: &'a crate::recurrence::process::OwnedRecurrenceProcessInput,
    sequence_id: u32,
    label: &str,
) -> RusticolResult<&'a [u32]> {
    let range = process
        .u32_sequence_ranges
        .get(sequence_id as usize)
        .copied()
        .ok_or_else(|| integrity(format!("{label} sequence is absent")))?;
    Ok(&process.u32_sequence_values
        [range.as_usize_range(process.u32_sequence_values.len(), label)?])
}

fn process_digest(
    process: &crate::recurrence::process::OwnedRecurrenceProcessInput,
    digest_id: u32,
    label: &str,
) -> RusticolResult<SemanticDigest> {
    let row = process
        .digest_catalog
        .get(digest_id as usize)
        .copied()
        .ok_or_else(|| integrity(format!("{label} digest is absent")))?;
    if row.id != digest_id {
        return Err(integrity(format!(
            "{label} digest catalog is not canonical"
        )));
    }
    SemanticDigest::new(row.value)
        .map_err(|error| integrity(format!("{label} digest is invalid: {error}")))
}

/// Model-neutral source family needed by the private direct-source adapter.
///
/// This deliberately stays independent of the evaluator module so the
/// recurrence seed remains a compact authenticated physics contract rather
/// than a loaded-executor object.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum OnTheFlySourceWavefunctionFamilyV1 {
    Scalar = 0,
    WeylFermion = 1,
    DiracFermion = 2,
    Vector = 3,
    Spin2 = 4,
}

/// Model-neutral particle orientation retained by one source state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub(crate) enum OnTheFlySourceOrientationV1 {
    Particle = 0,
    Antiparticle = 1,
    SelfConjugate = 2,
}

/// One authenticated source execution row exposed to the eventual engine
/// adapter. Source-template IDs may be sparse; callers must retain them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlySourceExecutionSpecV1 {
    pub(crate) source_slot: u32,
    pub(crate) source_template_id: u32,
    pub(crate) spin_state_class: i32,
    pub(crate) family: OnTheFlySourceWavefunctionFamilyV1,
    pub(crate) orientation: OnTheFlySourceOrientationV1,
    pub(crate) helicity: i32,
    pub(crate) chirality: i32,
    pub(crate) prepared_mass_parameter_slot: Option<u32>,
}

/// One concrete selected-helicity source state retained in the compact seed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlySourceStateV1 {
    pub(super) state_index: u32,
    pub(super) public_helicity: i32,
    pub(super) source_helicity: i32,
    pub(super) source_template_id: u32,
    pub(super) current_state_template_id: u32,
    pub(super) source_semantic_digest: SemanticDigest,
    pub(super) current_state_semantic_digest: SemanticDigest,
    pub(super) momentum_sign: i32,
    pub(super) crossing_phase: ExactComplexRational,
    pub(super) spin_state: i32,
    pub(super) chirality: i32,
    pub(super) flavour_flow: Box<[i32]>,
    pub(super) quantum_number_flow_id: u32,
    pub(super) color_seed_proof_digest: SemanticDigest,
    pub(super) source_family: OnTheFlySourceWavefunctionFamilyV1,
    pub(super) source_orientation: OnTheFlySourceOrientationV1,
    pub(super) prepared_mass_parameter_slot: Option<u32>,
}

impl OnTheFlySourceStateV1 {
    #[allow(clippy::too_many_arguments)]
    fn new(
        state_index: u32,
        public_helicity: i32,
        source_helicity: i32,
        source_template_id: u32,
        current_state_template_id: u32,
        source_semantic_digest: SemanticDigest,
        current_state_semantic_digest: SemanticDigest,
        momentum_sign: i32,
        crossing_phase: ExactComplexRational,
        spin_state: i32,
        chirality: i32,
        flavour_flow: Vec<i32>,
        quantum_number_flow_id: u32,
        color_seed_proof_digest: SemanticDigest,
        source_family: OnTheFlySourceWavefunctionFamilyV1,
        source_orientation: OnTheFlySourceOrientationV1,
        prepared_mass_parameter_slot: Option<u32>,
    ) -> RusticolResult<Self> {
        if source_template_id == MISSING_U32 || current_state_template_id == MISSING_U32 {
            return Err(invalid("source state reserves the missing u32 sentinel"));
        }
        if momentum_sign != -1 && momentum_sign != 1 {
            return Err(invalid("source momentum sign must be -1 or +1"));
        }
        if crossing_phase.is_zero() || flavour_flow.is_empty() {
            return Err(invalid(
                "source state requires a nonzero crossing phase and flavour ancestry",
            ));
        }
        Ok(Self {
            state_index,
            public_helicity,
            source_helicity,
            source_template_id,
            current_state_template_id,
            source_semantic_digest,
            current_state_semantic_digest,
            momentum_sign,
            crossing_phase,
            spin_state,
            chirality,
            flavour_flow: flavour_flow.into_boxed_slice(),
            quantum_number_flow_id,
            color_seed_proof_digest,
            source_family,
            source_orientation,
            prepared_mass_parameter_slot,
        })
    }

    pub(crate) const fn state_index(&self) -> u32 {
        self.state_index
    }

    pub(crate) const fn public_helicity(&self) -> i32 {
        self.public_helicity
    }
}

fn authenticated_source_family(
    catalog: &TemplateCatalog<'_>,
    source: SourceRow,
    current_state: CurrentStateRow,
) -> RusticolResult<OnTheFlySourceWavefunctionFamilyV1> {
    let family = catalog.string(
        source.wavefunction_family_string_id,
        "source wavefunction family",
    )?;
    let statistics = ParticleStatistics::try_from(current_state.statistics)?;
    match (family, statistics, current_state.dimension) {
        ("scalar", ParticleStatistics::Boson, 1) => Ok(OnTheFlySourceWavefunctionFamilyV1::Scalar),
        ("fermion", ParticleStatistics::Fermion, 2) => {
            Ok(OnTheFlySourceWavefunctionFamilyV1::WeylFermion)
        }
        ("fermion", ParticleStatistics::Fermion, 4) => {
            Ok(OnTheFlySourceWavefunctionFamilyV1::DiracFermion)
        }
        ("vector", ParticleStatistics::Boson, 4) => Ok(OnTheFlySourceWavefunctionFamilyV1::Vector),
        ("spin2", ParticleStatistics::Boson, 16) => Ok(OnTheFlySourceWavefunctionFamilyV1::Spin2),
        ("ghost" | "auxiliary", _, _) => Err(invalid(
            "on-the-fly source execution does not support ghost or auxiliary external states",
        )),
        _ => Err(integrity(
            "authenticated source family, statistics, and component dimension disagree",
        )),
    }
}

fn authenticated_source_orientation(
    current_state: CurrentStateRow,
) -> RusticolResult<OnTheFlySourceOrientationV1> {
    Ok(
        match CurrentOrientation::try_from(current_state.orientation)? {
            CurrentOrientation::Particle => OnTheFlySourceOrientationV1::Particle,
            CurrentOrientation::Antiparticle => OnTheFlySourceOrientationV1::Antiparticle,
            CurrentOrientation::SelfConjugate => OnTheFlySourceOrientationV1::SelfConjugate,
        },
    )
}

fn authenticated_prepared_mass_slot(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    source: SourceRow,
    current_state: CurrentStateRow,
) -> RusticolResult<Option<u32>> {
    if source.mass_parameter_id != current_state.mass_parameter_id {
        return Err(integrity(
            "authenticated source and current state disagree on their mass parameter",
        ));
    }
    if source.mass_parameter_id == MISSING_U32 {
        return Ok(None);
    }
    let parameter = authenticated
        .template()
        .input()
        .parameters
        .get(source.mass_parameter_id as usize)
        .copied()
        .ok_or_else(|| integrity("authenticated source mass parameter is absent"))?;
    if parameter.id != source.mass_parameter_id {
        return Err(integrity(
            "authenticated source mass-parameter catalog is not canonical",
        ));
    }
    let matches = authenticated
        .process()
        .input()
        .parameter_projection
        .iter()
        .filter(|row| row.parameter_template_id == source.mass_parameter_id && row.component == 0)
        .collect::<Vec<_>>();
    let [projection] = matches.as_slice() else {
        return Err(integrity(format!(
            "authenticated source mass parameter has {} real prepared projections, expected exactly one",
            matches.len(),
        )));
    };
    let prepared = projection.prepared_parameter_id().ok_or_else(|| {
        invalid("authenticated source mass parameter has no prepared execution slot")
    })?;
    if parameter.prepared_parameter_id != prepared
        || prepared as usize >= authenticated.template().input().parameters.len()
    {
        return Err(integrity(
            "authenticated source mass projection disagrees with the prepared parameter catalog",
        ));
    }
    Ok(Some(prepared))
}

/// All concrete source alternatives for one external source slot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlySourceAnchorV1 {
    pub(super) source_slot: u32,
    pub(super) external_label: u32,
    pub(super) color_role: i32,
    pub(super) states: Box<[OnTheFlySourceStateV1]>,
}

impl OnTheFlySourceAnchorV1 {
    fn new(
        source_slot: u32,
        external_label: u32,
        color_role: i32,
        mut states: Vec<OnTheFlySourceStateV1>,
    ) -> RusticolResult<Self> {
        if states.is_empty() {
            return Err(invalid("source anchor has no concrete state"));
        }
        states.sort_unstable_by_key(|state| state.state_index);
        if states
            .windows(2)
            .any(|pair| pair[0].state_index == pair[1].state_index)
        {
            return Err(invalid("source anchor repeats a state index"));
        }
        Ok(Self {
            source_slot,
            external_label,
            color_role,
            states: states.into_boxed_slice(),
        })
    }

    pub(super) fn selected(
        &self,
        state_index: u32,
        public_helicity: i32,
    ) -> RusticolResult<&OnTheFlySourceStateV1> {
        self.states
            .iter()
            .find(|state| {
                state.state_index == state_index && state.public_helicity == public_helicity
            })
            .ok_or_else(|| {
                invalid(format!(
                    "source slot {} has no state {state_index} with public helicity {public_helicity}",
                    self.source_slot
                ))
            })
    }
}

/// One authenticated open-line pairing and permutation proof.
///
/// `fermion_parity` is retained as proof identity but is not multiplied at
/// closure: canonical evaluator input-exchange factors already apply that
/// sign exactly once, matching the established recurrence builder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlyPairingLineageV1 {
    pub(super) rule_id: u32,
    pub(super) endpoint_pairs: Box<[[u32; 2]]>,
    pub(super) source_slot_permutation: Box<[u32]>,
    pub(super) source_lineage: Box<[u32]>,
    pub(super) fermion_parity: i32,
    pub(super) proof_digest: SemanticDigest,
}

impl OnTheFlyPairingLineageV1 {
    fn from_authenticated_rule(
        catalog: crate::recurrence::process::ValidatedFermionPairingCatalog<'_>,
        rule: crate::recurrence::process::FermionPairingRuleRow,
    ) -> RusticolResult<Self> {
        let endpoint_pairs = catalog
            .endpoint_pairings(rule)?
            .iter()
            .map(|pair| {
                [
                    pair.fundamental_source_slot,
                    pair.antifundamental_source_slot,
                ]
            })
            .collect::<Vec<_>>();
        for pair in &endpoint_pairs {
            if pair[0] == pair[1] {
                return Err(invalid("pairing lineage joins one source slot to itself"));
            }
        }
        if endpoint_pairs.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid("pairing lineage repeats an endpoint pair"));
        }
        let mut endpoints = BTreeSet::new();
        if endpoint_pairs
            .iter()
            .flatten()
            .any(|endpoint| !endpoints.insert(*endpoint))
        {
            return Err(invalid("pairing lineage reuses one endpoint"));
        }
        let source_slot_permutation = catalog.source_slot_permutation(rule)?.to_vec();
        let mut sorted_permutation = source_slot_permutation.clone();
        sorted_permutation.sort_unstable();
        if sorted_permutation != (0..catalog.summary().source_count()).collect::<Vec<_>>() {
            return Err(integrity(
                "authenticated pairing source-slot permutation is not complete",
            ));
        }
        if !matches!(rule.fermion_parity, -1 | 1) {
            return Err(integrity(
                "authenticated pairing rule has a non-binary fermion parity",
            ));
        }
        Ok(Self {
            rule_id: rule.rule_id,
            endpoint_pairs: endpoint_pairs.into_boxed_slice(),
            source_slot_permutation: source_slot_permutation.into_boxed_slice(),
            source_lineage: catalog.lineage(rule)?.to_vec().into_boxed_slice(),
            fermion_parity: rule.fermion_parity,
            proof_digest: SemanticDigest::new(rule.proof_digest).map_err(|error| {
                integrity(format!(
                    "authenticated pairing proof digest is invalid: {error}"
                ))
            })?,
        })
    }
}

fn authenticated_source_anchors(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
) -> RusticolResult<Vec<OnTheFlySourceAnchorV1>> {
    let process = authenticated.process().input();
    let selected_source_mode = authenticated.process().summary().selected_source_mode();
    let selected_source_states = process
        .selected_source_coverage
        .iter()
        .map(|row| (row.source_slot, row.source_state_index))
        .collect::<BTreeSet<_>>();
    let templates = authenticated.template();
    let input = templates.input();
    let catalog = TemplateCatalog::new(input)?;
    let mut anchors = Vec::new();
    anchors
        .try_reserve_exact(process.external_legs.len())
        .map_err(|error| invalid(format!("source-anchor allocation failed: {error}")))?;
    for (source_slot, leg) in process.external_legs.iter().copied().enumerate() {
        if leg.source_slot as usize != source_slot {
            return Err(integrity(
                "authenticated external-leg source slots are not canonical",
            ));
        }
        let range = leg.source_state_range.as_usize_range(
            process.source_states.len(),
            "on-the-fly authenticated source states",
        )?;
        let mut states = Vec::new();
        states
            .try_reserve_exact(range.len())
            .map_err(|error| invalid(format!("source-state allocation failed: {error}")))?;
        let mut color_role = None;
        for process_state in process.source_states[range]
            .iter()
            .copied()
            .filter(|state| {
                !selected_source_mode
                    || selected_source_states.contains(&(leg.source_slot, state.state_index))
            })
        {
            let source = *input
                .sources
                .get(process_state.source_template_id as usize)
                .ok_or_else(|| integrity("authenticated source template is absent"))?;
            let current_state = *input
                .current_states
                .get(process_state.current_state_template_id as usize)
                .ok_or_else(|| integrity("authenticated current-state template is absent"))?;
            if source.id != process_state.source_template_id
                || source.state_template_id != process_state.current_state_template_id
                || current_state.id != process_state.current_state_template_id
                || source.spin_state != process_state.spin_state
                || current_state.chirality != process_state.chirality
            {
                return Err(integrity(
                    "authenticated process source row differs from its template contract",
                ));
            }
            match color_role {
                None => color_role = Some(current_state.color_representation),
                Some(previous) if previous == current_state.color_representation => {}
                Some(_) => {
                    return Err(integrity(
                        "one external source anchor mixes color representations",
                    ));
                }
            }
            let seed = catalog.source_seed(source)?;
            let source_family = authenticated_source_family(&catalog, source, current_state)?;
            let source_orientation = authenticated_source_orientation(current_state)?;
            let prepared_mass_parameter_slot =
                authenticated_prepared_mass_slot(authenticated, source, current_state)?;
            states.push(OnTheFlySourceStateV1::new(
                process_state.state_index,
                process_state.public_helicity,
                source.helicity,
                source.id,
                current_state.id,
                catalog.digest(source.semantic_digest_id, "source semantic")?,
                catalog.digest(current_state.semantic_digest_id, "current-state semantic")?,
                process_state.momentum_sign,
                process_factor(
                    process,
                    process_state.crossing_phase_factor_id,
                    "source crossing phase",
                )?,
                process_state.spin_state,
                process_state.chirality,
                catalog
                    .flavour_flow(source.flavour_flow_id, "source flavour flow")?
                    .to_vec(),
                source.quantum_number_flow_id,
                seed.proof_digest(),
                source_family,
                source_orientation,
                prepared_mass_parameter_slot,
            )?);
        }
        anchors.push(OnTheFlySourceAnchorV1::new(
            leg.source_slot,
            leg.public_label,
            color_role.ok_or_else(|| integrity("authenticated source anchor is empty"))?,
            states,
        )?);
    }
    Ok(anchors)
}

/// One exact public-flow selector derived from the validated process payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlyPublicFlowV1 {
    pub(super) flow_id: u32,
    pub(super) construction_sector_id: u32,
    pub(super) target_components: Box<[LCColorComponent]>,
    pub(super) closure_anchor_slot: u32,
    pub(super) source_slot_permutation: Box<[u32]>,
    /// Authenticated squared-output reducer metadata. Raw amplitudes do not
    /// apply this value.
    pub(super) reduction_weight: ExactComplexRational,
    pub(super) pairing_rule_id: Option<u32>,
    pub(super) pairing_proof_digest: Option<SemanticDigest>,
    pub(super) pairing_source_slot_permutation: Box<[u32]>,
    pub(super) pairing_source_lineage: Box<[u32]>,
    pub(super) pairing_fermion_parity: i32,
    pub(super) closure_proof_digest: SemanticDigest,
    pub(super) semantic_digest: SemanticDigest,
}

impl OnTheFlyPublicFlowV1 {
    fn compute_digest(&self) -> RusticolResult<SemanticDigest> {
        let mut hash = Sha256::new();
        hash.update(ON_THE_FLY_PUBLIC_FLOW_DOMAIN);
        hash.update(self.flow_id.to_le_bytes());
        hash.update(self.construction_sector_id.to_le_bytes());
        hash.update(self.closure_anchor_slot.to_le_bytes());
        hash_exact(&mut hash, self.reduction_weight);
        hash.update(self.pairing_rule_id.unwrap_or(MISSING_U32).to_le_bytes());
        match self.pairing_proof_digest {
            None => hash.update([0]),
            Some(digest) => {
                hash.update([1]);
                hash_digest(&mut hash, digest);
            }
        }
        hash.update(self.pairing_fermion_parity.to_le_bytes());
        hash_len(
            &mut hash,
            self.pairing_source_slot_permutation.len(),
            "public-flow pairing source permutation",
        )?;
        for source_slot in &self.pairing_source_slot_permutation {
            hash.update(source_slot.to_le_bytes());
        }
        hash_len(
            &mut hash,
            self.pairing_source_lineage.len(),
            "public-flow pairing source lineage",
        )?;
        for lineage in &self.pairing_source_lineage {
            hash.update(lineage.to_le_bytes());
        }
        hash_digest(&mut hash, self.closure_proof_digest);
        hash_len(
            &mut hash,
            self.source_slot_permutation.len(),
            "public-flow source permutation",
        )?;
        for source_slot in &self.source_slot_permutation {
            hash.update(source_slot.to_le_bytes());
        }
        hash_len(
            &mut hash,
            self.target_components.len(),
            "public-flow target components",
        )?;
        for component in &self.target_components {
            hash.update([component.kind() as u8]);
            hash_len(
                &mut hash,
                component.source_slots().len(),
                "public-flow component word",
            )?;
            for source_slot in component.source_slots() {
                hash.update(source_slot.to_le_bytes());
            }
        }
        final_digest(hash)
    }
}

fn mapped_public_component(
    kind: LCColorComponentKind,
    source_slots: &[u32],
    source_slot_permutation: &[u32],
) -> RusticolResult<LCColorComponent> {
    LCColorComponent::new(
        kind,
        source_slots
            .iter()
            .map(|source_slot| {
                source_slot_permutation
                    .get(*source_slot as usize)
                    .copied()
                    .ok_or_else(|| integrity("public-flow source permutation is out of range"))
            })
            .collect::<RusticolResult<Vec<_>>>()?,
    )
}

fn authenticated_public_flows(
    authenticated: &AuthenticatedRecurrenceBuilderInput,
    pairing_lineages: &[OnTheFlyPairingLineageV1],
) -> RusticolResult<Vec<OnTheFlyPublicFlowV1>> {
    use crate::recurrence::process::ProcessLCSectorKind;

    let validated = authenticated.process();
    let process = validated.input();
    let selected_flow_ids = process
        .selected_public_flow_coverage
        .iter()
        .map(|row| row.flow_id)
        .collect::<BTreeSet<_>>();
    let selected_flow_mode = validated.summary().selected_flow_mode();
    let mut result = Vec::new();
    for flow in process
        .public_lc_flows
        .iter()
        .copied()
        .filter(|flow| !selected_flow_mode || selected_flow_ids.contains(&flow.flow_id))
    {
        let sector = process
            .physical_lc_sectors
            .get(flow.construction_sector_id as usize)
            .copied()
            .ok_or_else(|| integrity("public-flow construction sector is absent"))?;
        if sector.sector_id != flow.construction_sector_id {
            return Err(integrity(
                "public-flow construction sector catalog is not canonical",
            ));
        }
        let source_slot_permutation = process_sequence(
            process,
            flow.source_slot_permutation_sequence_id,
            "public-flow source permutation",
        )?
        .to_vec();
        let source_count = process.external_legs.len();
        if source_slot_permutation.len() != source_count {
            return Err(integrity(
                "authenticated public-flow permutation has the wrong source count",
            ));
        }
        let mut sorted_permutation = source_slot_permutation.clone();
        sorted_permutation.sort_unstable();
        if sorted_permutation
            != (0..checked_u32(source_count, "public-flow source count")?).collect::<Vec<_>>()
        {
            return Err(integrity(
                "authenticated public-flow source permutation is not complete",
            ));
        }
        let closure_anchor_slot = source_slot_permutation
            .get(sector.closure_source_slot as usize)
            .copied()
            .ok_or_else(|| integrity("public-flow closure anchor is out of range"))?;
        let mut target_components = Vec::new();
        match sector.kind()? {
            ProcessLCSectorKind::Singlet => {}
            ProcessLCSectorKind::SingleTrace => {
                target_components.push(LCColorComponent::new(
                    LCColorComponentKind::Trace,
                    process_sequence(process, flow.word_sequence_id, "public LC flow word")?
                        .to_vec(),
                )?);
            }
            ProcessLCSectorKind::OpenLines => {
                let range = sector.open_string_range.as_usize_range(
                    process.lc_open_strings.len(),
                    "public-flow construction open strings",
                )?;
                for row in &process.lc_open_strings[range] {
                    let mut word = vec![row.fundamental_source_slot];
                    word.extend_from_slice(process_sequence(
                        process,
                        row.adjoint_sequence_id,
                        "public-flow open-string adjoints",
                    )?);
                    word.push(row.antifundamental_source_slot);
                    target_components.push(mapped_public_component(
                        LCColorComponentKind::OpenString,
                        &word,
                        &source_slot_permutation,
                    )?);
                }
            }
        }
        target_components.sort_unstable();
        if target_components.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(integrity(
                "authenticated public flow repeats one target component",
            ));
        }
        let mut target_endpoint_pairs = target_components
            .iter()
            .filter(|component| component.kind() == LCColorComponentKind::OpenString)
            .map(|component| {
                let slots = component.source_slots();
                Ok([
                    *slots
                        .first()
                        .ok_or_else(|| integrity("public open string has no first endpoint"))?,
                    *slots
                        .last()
                        .ok_or_else(|| integrity("public open string has no last endpoint"))?,
                ])
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        target_endpoint_pairs.sort_unstable();
        let pairing = if pairing_lineages.is_empty() {
            None
        } else {
            let matches = pairing_lineages
                .iter()
                .filter(|lineage| {
                    if target_endpoint_pairs.is_empty() {
                        pairing_lineages.len() == 1
                    } else {
                        lineage.endpoint_pairs.as_ref() == target_endpoint_pairs.as_slice()
                    }
                })
                .collect::<Vec<_>>();
            let [lineage] = matches.as_slice() else {
                return Err(invalid(format!(
                    "public flow {} selects {} authenticated pairing rules, expected exactly one",
                    flow.flow_id,
                    matches.len(),
                )));
            };
            Some(*lineage)
        };
        let reduction_weight = process_factor(
            process,
            flow.reduction_weight_factor_id,
            "public-flow reduction weight",
        )?;
        if reduction_weight.is_zero() {
            return Err(integrity("public-flow reduction weight is zero"));
        }
        let mut flow = OnTheFlyPublicFlowV1 {
            flow_id: flow.flow_id,
            construction_sector_id: flow.construction_sector_id,
            target_components: target_components.into_boxed_slice(),
            closure_anchor_slot,
            source_slot_permutation: source_slot_permutation.into_boxed_slice(),
            reduction_weight,
            pairing_rule_id: pairing.map(|lineage| lineage.rule_id),
            pairing_proof_digest: pairing.map(|lineage| lineage.proof_digest),
            pairing_source_slot_permutation: pairing
                .map(|lineage| lineage.source_slot_permutation.to_vec())
                .unwrap_or_default()
                .into_boxed_slice(),
            pairing_source_lineage: pairing
                .map(|lineage| lineage.source_lineage.to_vec())
                .unwrap_or_default()
                .into_boxed_slice(),
            pairing_fermion_parity: pairing.map_or(1, |lineage| lineage.fermion_parity),
            closure_proof_digest: process_digest(
                process,
                sector.closure_proof_digest_id,
                "physical-sector closure proof",
            )?,
            semantic_digest: SemanticDigest::new([1; 32])?,
        };
        flow.semantic_digest = flow.compute_digest()?;
        result.push(flow);
    }
    result.sort_unstable_by_key(|flow| flow.flow_id);
    if result.is_empty()
        || result
            .windows(2)
            .any(|pair| pair[0].flow_id == pair[1].flow_id)
    {
        return Err(integrity(
            "authenticated public-flow selector catalog is empty or non-canonical",
        ));
    }
    Ok(result)
}

/// Compact immutable process input for one on-the-fly lane.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlyProcessSeedV1 {
    pub(super) process_digest: SemanticDigest,
    pub(super) model_digest: SemanticDigest,
    pub(super) template_catalog_digest: SemanticDigest,
    pub(super) prepared_pack_digest: SemanticDigest,
    pub(super) direct_catalog_digest: SemanticDigest,
    pub(super) normalization_semantic_digest: SemanticDigest,
    pub(super) normalization_convention: Box<str>,
    pub(super) source_anchors: Box<[OnTheFlySourceAnchorV1]>,
    pub(super) public_flows: Box<[OnTheFlyPublicFlowV1]>,
    pub(super) coupling_limits: Box<[Option<u32>]>,
    pub(super) pairing_catalog_digest: Option<SemanticDigest>,
    pub(super) pairing_topology_digest: Option<SemanticDigest>,
    pub(super) pairing_semantic_digest: Option<SemanticDigest>,
    pub(super) pairing_lineages: Box<[OnTheFlyPairingLineageV1]>,
    pub(super) semantic_digest: SemanticDigest,
}

impl OnTheFlyProcessSeedV1 {
    pub(crate) fn from_authenticated_process(
        authenticated: &AuthenticatedRecurrenceBuilderInput,
        direct_catalog_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        if authenticated.process().summary().strategy() != RecurrenceStrategy::TopologyReplay {
            return Err(invalid(
                "the private on-the-fly slice requires selected-flow topology-replay input",
            ));
        }
        let process = authenticated.process();
        let templates = authenticated.template();
        let normalization = process
            .input()
            .normalization
            .first()
            .copied()
            .ok_or_else(|| integrity("authenticated process normalization is absent"))?;
        if process.input().normalization.len() != 1
            || process_factor(
                process.input(),
                normalization.factor_id,
                "process normalization",
            )? != ExactComplexRational::ONE
        {
            return Err(integrity(
                "on-the-fly raw-amplitude contract requires the authenticated process normalization factor to be exact one",
            ));
        }
        let normalization_semantic_digest = process_digest(
            process.input(),
            normalization.semantic_digest_id,
            "process normalization semantic",
        )?;
        let normalization_convention = process_string(
            process.input(),
            normalization.convention_string_id,
            "process normalization convention",
        )?
        .to_owned()
        .into_boxed_str();
        let mut source_anchors = authenticated_source_anchors(authenticated)?;
        let coupling_limits = process
            .input()
            .coupling_limits
            .iter()
            .map(|row| Some(row.maximum))
            .collect::<Vec<_>>();
        let pairing = process.fermion_pairing_catalog();
        let mut pairing_lineages = pairing
            .map(|catalog| {
                catalog
                    .rules()
                    .iter()
                    .copied()
                    .map(|rule| OnTheFlyPairingLineageV1::from_authenticated_rule(catalog, rule))
                    .collect::<RusticolResult<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        pairing_lineages.sort_unstable_by_key(|lineage| lineage.rule_id);
        let public_flows = authenticated_public_flows(authenticated, &pairing_lineages)?;
        let pairing_summary = process.fermion_pairing_summary();
        if source_anchors.len() < 2 || coupling_limits.is_empty() {
            return Err(invalid(
                "process seed requires at least two sources and explicit coupling limits",
            ));
        }
        if coupling_limits.iter().any(Option::is_none) {
            return Err(invalid(
                "on-the-fly coupling limits must be explicit for every model order",
            ));
        }
        source_anchors.sort_unstable_by_key(|anchor| anchor.source_slot);
        for (index, anchor) in source_anchors.iter().enumerate() {
            if anchor.source_slot as usize != index {
                return Err(invalid(
                    "source anchors must form the canonical dense slot domain",
                ));
            }
        }
        for lineage in &pairing_lineages {
            for endpoint in lineage.endpoint_pairs.iter().flatten() {
                if *endpoint as usize >= source_anchors.len() {
                    return Err(invalid(
                        "pairing lineage endpoint is outside the source domain",
                    ));
                }
            }
        }

        if pairing_summary.is_some() != !pairing_lineages.is_empty() {
            return Err(integrity(
                "authenticated pairing summary and rules disagree",
            ));
        }
        let mut result = Self {
            process_digest: process.semantic_identity().process_digest(),
            model_digest: templates.summary().compiled_model_digest,
            template_catalog_digest: templates.summary().catalog_digest,
            prepared_pack_digest: templates.summary().prepared_kernel_pack_digest,
            direct_catalog_digest,
            normalization_semantic_digest,
            normalization_convention,
            source_anchors: source_anchors.into_boxed_slice(),
            public_flows: public_flows.into_boxed_slice(),
            coupling_limits: coupling_limits.into_boxed_slice(),
            pairing_catalog_digest: pairing_summary.map(|summary| summary.columnar_digest()),
            pairing_topology_digest: pairing_summary.map(|summary| summary.topology_digest()),
            pairing_semantic_digest: pairing_summary.map(|summary| summary.semantic_digest()),
            pairing_lineages: pairing_lineages.into_boxed_slice(),
            // Replaced immediately below by the digest over every field.
            semantic_digest: SemanticDigest::new([1; 32])?,
        };
        result.semantic_digest = result.compute_digest()?;
        Ok(result)
    }

    fn compute_digest(&self) -> RusticolResult<SemanticDigest> {
        let mut hash = Sha256::new();
        hash.update(ON_THE_FLY_SEED_DOMAIN);
        for digest in [
            self.process_digest,
            self.model_digest,
            self.template_catalog_digest,
            self.prepared_pack_digest,
            self.direct_catalog_digest,
            self.normalization_semantic_digest,
        ] {
            hash_digest(&mut hash, digest);
        }
        hash_len(
            &mut hash,
            self.normalization_convention.len(),
            "normalization convention",
        )?;
        hash.update(self.normalization_convention.as_bytes());
        hash_len(&mut hash, self.source_anchors.len(), "source anchors")?;
        for anchor in &self.source_anchors {
            hash.update(anchor.source_slot.to_le_bytes());
            hash.update(anchor.external_label.to_le_bytes());
            hash.update(anchor.color_role.to_le_bytes());
            hash_len(&mut hash, anchor.states.len(), "source states")?;
            for state in &anchor.states {
                hash.update(state.state_index.to_le_bytes());
                hash.update(state.public_helicity.to_le_bytes());
                hash.update(state.source_helicity.to_le_bytes());
                hash.update(state.source_template_id.to_le_bytes());
                hash.update(state.current_state_template_id.to_le_bytes());
                hash_digest(&mut hash, state.source_semantic_digest);
                hash_digest(&mut hash, state.current_state_semantic_digest);
                hash.update(state.momentum_sign.to_le_bytes());
                hash_exact(&mut hash, state.crossing_phase);
                hash.update(state.spin_state.to_le_bytes());
                hash.update(state.chirality.to_le_bytes());
                hash_len(&mut hash, state.flavour_flow.len(), "source flavour flow")?;
                for flavour in &state.flavour_flow {
                    hash.update(flavour.to_le_bytes());
                }
                hash.update(state.quantum_number_flow_id.to_le_bytes());
                hash_digest(&mut hash, state.color_seed_proof_digest);
                hash.update([state.source_family as u8]);
                hash.update([state.source_orientation as u8]);
                match state.prepared_mass_parameter_slot {
                    None => hash.update([0]),
                    Some(slot) => {
                        hash.update([1]);
                        hash.update(slot.to_le_bytes());
                    }
                }
            }
        }
        hash_len(&mut hash, self.coupling_limits.len(), "coupling limits")?;
        for limit in &self.coupling_limits {
            hash.update(limit.unwrap_or(MISSING_U32).to_le_bytes());
        }
        hash_len(&mut hash, self.public_flows.len(), "public flows")?;
        for flow in &self.public_flows {
            hash_digest(&mut hash, flow.semantic_digest);
        }
        for digest in [
            self.pairing_catalog_digest,
            self.pairing_topology_digest,
            self.pairing_semantic_digest,
        ] {
            match digest {
                None => hash.update([0]),
                Some(value) => {
                    hash.update([1]);
                    hash_digest(&mut hash, value);
                }
            }
        }
        hash_len(&mut hash, self.pairing_lineages.len(), "pairing lineages")?;
        for lineage in &self.pairing_lineages {
            hash.update(lineage.rule_id.to_le_bytes());
            hash_digest(&mut hash, lineage.proof_digest);
            hash_len(
                &mut hash,
                lineage.endpoint_pairs.len(),
                "pairing endpoint pairs",
            )?;
            for pair in &lineage.endpoint_pairs {
                hash.update(pair[0].to_le_bytes());
                hash.update(pair[1].to_le_bytes());
            }
            hash_len(
                &mut hash,
                lineage.source_slot_permutation.len(),
                "pairing source permutation",
            )?;
            for slot in &lineage.source_slot_permutation {
                hash.update(slot.to_le_bytes());
            }
            hash_len(
                &mut hash,
                lineage.source_lineage.len(),
                "pairing source lineage",
            )?;
            for line in &lineage.source_lineage {
                hash.update(line.to_le_bytes());
            }
            hash.update(lineage.fermion_parity.to_le_bytes());
        }
        final_digest(hash)
    }

    pub(crate) const fn semantic_digest(&self) -> SemanticDigest {
        self.semantic_digest
    }

    /// Iterate the exact source execution contracts without exposing seed
    /// storage or importing evaluator-specific types into recurrence.
    pub(crate) fn source_execution_specs(
        &self,
    ) -> impl Iterator<Item = OnTheFlySourceExecutionSpecV1> + '_ {
        self.source_anchors.iter().flat_map(|anchor| {
            anchor
                .states
                .iter()
                .map(move |state| OnTheFlySourceExecutionSpecV1 {
                    source_slot: anchor.source_slot,
                    source_template_id: state.source_template_id,
                    spin_state_class: state.spin_state,
                    family: state.source_family,
                    orientation: state.source_orientation,
                    helicity: state.public_helicity,
                    chirality: state.chirality,
                    prepared_mass_parameter_slot: state.prepared_mass_parameter_slot,
                })
        })
    }
}
