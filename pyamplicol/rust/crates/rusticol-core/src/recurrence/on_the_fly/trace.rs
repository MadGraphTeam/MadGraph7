// SPDX-License-Identifier: 0BSD

use super::sweep::*;
use super::*;

/// Authenticated model operation addressed by one semantic executor key.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub(crate) enum OnTheFlyOperationKindV1 {
    Source = 0,
    Transition = 1,
    Propagator = 2,
    Closure = 3,
    IdentityFinalizer = 4,
}

/// Plan-independent semantic address of one prepared Direct-Arena executor.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct OnTheFlyExecutorKeyV1 {
    direct_catalog_digest: SemanticDigest,
    role: DirectExecutorRole,
    operation_kind: OnTheFlyOperationKindV1,
    operation_id: Option<u32>,
    operation_semantic_digest: SemanticDigest,
    evaluator_binding_id: Option<u32>,
    evaluator_binding_semantic_digest: SemanticDigest,
}

impl OnTheFlyExecutorKeyV1 {
    #[allow(clippy::too_many_arguments)]
    fn new(
        direct_catalog_digest: SemanticDigest,
        role: DirectExecutorRole,
        operation_kind: OnTheFlyOperationKindV1,
        operation_id: Option<u32>,
        operation_semantic_digest: SemanticDigest,
        evaluator_binding_id: Option<u32>,
        evaluator_binding_semantic_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        let expected_role = match operation_kind {
            OnTheFlyOperationKindV1::Source => DirectExecutorRole::Source,
            OnTheFlyOperationKindV1::Transition => DirectExecutorRole::Contribution,
            OnTheFlyOperationKindV1::Propagator | OnTheFlyOperationKindV1::IdentityFinalizer => {
                DirectExecutorRole::Finalization
            }
            OnTheFlyOperationKindV1::Closure => DirectExecutorRole::Closure,
        };
        if role != expected_role {
            return Err(invalid("semantic executor key has an inconsistent role"));
        }
        let identity = operation_kind == OnTheFlyOperationKindV1::IdentityFinalizer;
        if identity != operation_id.is_none() || identity != evaluator_binding_id.is_none() {
            return Err(invalid(
                "only an identity finalizer may omit operation and evaluator IDs",
            ));
        }
        Ok(Self {
            direct_catalog_digest,
            role,
            operation_kind,
            operation_id,
            operation_semantic_digest,
            evaluator_binding_id,
            evaluator_binding_semantic_digest,
        })
    }

    pub(crate) const fn direct_catalog_digest(self) -> SemanticDigest {
        self.direct_catalog_digest
    }

    pub(crate) const fn role(self) -> DirectExecutorRole {
        self.role
    }

    pub(crate) const fn operation_kind(self) -> OnTheFlyOperationKindV1 {
        self.operation_kind
    }

    pub(crate) const fn operation_id(self) -> Option<u32> {
        self.operation_id
    }

    pub(crate) const fn operation_semantic_digest(self) -> SemanticDigest {
        self.operation_semantic_digest
    }

    pub(crate) const fn evaluator_binding_id(self) -> Option<u32> {
        self.evaluator_binding_id
    }

    pub(crate) const fn evaluator_binding_semantic_digest(self) -> SemanticDigest {
        self.evaluator_binding_semantic_digest
    }
}

#[derive(Clone, Debug)]
pub(super) enum OnTheFlyTraceOperationV1 {
    Source {
        key: OnTheFlyExecutorKeyV1,
        row: DirectSourceRow,
    },
    Contribution {
        key: OnTheFlyExecutorKeyV1,
        row: DirectContributionRow,
    },
    Finalization {
        key: OnTheFlyExecutorKeyV1,
        row: DirectFinalizationRow,
    },
    Closure {
        key: OnTheFlyExecutorKeyV1,
        row: DirectClosureRow,
    },
}

impl OnTheFlyTraceOperationV1 {
    pub(super) const fn key(&self) -> OnTheFlyExecutorKeyV1 {
        match self {
            Self::Source { key, .. }
            | Self::Contribution { key, .. }
            | Self::Finalization { key, .. }
            | Self::Closure { key, .. } => *key,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlyWorkspaceLayoutV1 {
    pub(super) source_count: u32,
    pub(super) lorentz_component_count: u16,
    pub(super) parameter_count: u32,
    pub(super) current_component_count: u32,
    pub(super) amplitude_component_count: u32,
    pub(super) momentum_form_count: u32,
    pub(super) exact_factor_count: u32,
}

impl OnTheFlyWorkspaceLayoutV1 {
    pub(crate) const fn source_count(self) -> u32 {
        self.source_count
    }

    pub(crate) const fn current_component_count(self) -> u32 {
        self.current_component_count
    }

    pub(crate) const fn amplitude_component_count(self) -> u32 {
        self.amplitude_component_count
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct OnTheFlyStructuralProofV1 {
    current_count: u32,
    contribution_count: u32,
    closure_count: u32,
    constructed_current_count: u32,
    constructed_contribution_count: u32,
    current_multiset_digest: SemanticDigest,
    contribution_multiset_digest: SemanticDigest,
    closure_multiset_digest: SemanticDigest,
    owner_digest: SemanticDigest,
    semantic_digest: SemanticDigest,
}

impl OnTheFlyStructuralProofV1 {
    pub(crate) const fn current_count(self) -> u32 {
        self.current_count
    }

    pub(crate) const fn contribution_count(self) -> u32 {
        self.contribution_count
    }

    pub(crate) const fn closure_count(self) -> u32 {
        self.closure_count
    }

    pub(crate) const fn semantic_digest(self) -> SemanticDigest {
        self.semantic_digest
    }

    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(super) const fn current_multiset_digest(self) -> SemanticDigest {
        self.current_multiset_digest
    }

    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(super) const fn contribution_multiset_digest(self) -> SemanticDigest {
        self.contribution_multiset_digest
    }

    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(super) const fn closure_multiset_digest(self) -> SemanticDigest {
        self.closure_multiset_digest
    }
}

/// Immutable selected-query trace.  Exact current keys remain available for
/// development parity joins; numeric IDs are deliberately query-local.
#[derive(Clone, Debug)]
pub(crate) struct OnTheFlyStructuralTraceV1 {
    pub(super) seed_digest: SemanticDigest,
    pub(super) query_digest: SemanticDigest,
    pub(super) current_keys: Box<[CurrentCoreKey]>,
    pub(super) current_colors: Box<[DynamicLCColorState]>,
    pub(super) operations: Box<[OnTheFlyTraceOperationV1]>,
    pub(super) momentum_forms: Box<[CanonicalMomentumLinearForm]>,
    pub(super) exact_factors: Box<[ExactComplexRational]>,
    pub(super) pairing_owner: ResolvedPairingOwnerV1,
    pub(super) layout: OnTheFlyWorkspaceLayoutV1,
    pub(super) proof: OnTheFlyStructuralProofV1,
    pub(super) semantic_digest: SemanticDigest,
    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(super) current_component_ranges: Box<[[u32; 2]]>,
    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(super) current_semantic_digests: Box<[SemanticDigest]>,
}

impl OnTheFlyStructuralTraceV1 {
    pub(crate) const fn proof(&self) -> OnTheFlyStructuralProofV1 {
        self.proof
    }

    pub(crate) fn current_keys(&self) -> &[CurrentCoreKey] {
        &self.current_keys
    }

    pub(crate) const fn layout(&self) -> OnTheFlyWorkspaceLayoutV1 {
        self.layout
    }

    pub(crate) fn momentum_forms(&self) -> &[CanonicalMomentumLinearForm] {
        &self.momentum_forms
    }

    pub(crate) fn exact_factors(&self) -> &[ExactComplexRational] {
        &self.exact_factors
    }

    pub(crate) const fn semantic_digest(&self) -> SemanticDigest {
        self.semantic_digest
    }

    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(crate) fn current_component_range(&self, current_id: u32) -> RusticolResult<[u32; 2]> {
        self.current_component_ranges
            .get(current_id as usize)
            .copied()
            .ok_or_else(|| invalid("observed current ID is outside the trace"))
    }

    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    pub(crate) fn current_semantic_digest(
        &self,
        current_id: u32,
    ) -> RusticolResult<SemanticDigest> {
        self.current_semantic_digests
            .get(current_id as usize)
            .copied()
            .ok_or_else(|| invalid("observed current ID is outside the trace"))
    }
}

pub(super) fn hash_current_key(
    key: &CurrentCoreKey,
    color: &DynamicLCColorState,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-on-the-fly-current-key-v1\0");
    hash_digest(&mut hash, key.catalog_digest());
    hash.update([match key.node_kind() {
        RecurrenceNodeKind::Source => 0,
        RecurrenceNodeKind::Current => 1,
    }]);
    hash.update(key.current_state_template_id().to_le_bytes());
    hash_len(
        &mut hash,
        key.support_source_slots().len(),
        "current support",
    )?;
    for slot in key.support_source_slots() {
        hash.update(slot.to_le_bytes());
    }
    hash_len(&mut hash, key.momentum().terms().len(), "current momentum")?;
    for term in key.momentum().terms() {
        hash.update(term.source_slot.to_le_bytes());
        hash.update(term.coefficient.to_le_bytes());
    }
    hash.update(key.spin_state_class().to_le_bytes());
    hash_len(
        &mut hash,
        key.helicity_identity().local_source_states().len(),
        "current source ancestry",
    )?;
    for assignment in key.helicity_identity().local_source_states() {
        hash.update(assignment.source_slot().to_le_bytes());
        hash.update(assignment.state_index().to_le_bytes());
    }
    hash_len(
        &mut hash,
        key.flavour_flow().len(),
        "current flavour ancestry",
    )?;
    for value in key.flavour_flow() {
        hash.update(value.to_le_bytes());
    }
    hash.update(key.quantum_number_flow_id().to_le_bytes());
    hash_len(
        &mut hash,
        key.coupling_orders().len(),
        "current coupling orders",
    )?;
    for value in key.coupling_orders() {
        hash.update(value.to_le_bytes());
    }
    match key.source_binding() {
        CurrentSourceBinding::None => hash.update([0]),
        CurrentSourceBinding::FixedTemplate(id) => {
            hash.update([1]);
            hash.update(id.to_le_bytes());
        }
        CurrentSourceBinding::RuntimeDispatch { .. } => {
            return Err(integrity(
                "selected on-the-fly current unexpectedly uses runtime source dispatch",
            ));
        }
    }
    hash.update(
        key.propagator_template_id()
            .unwrap_or(MISSING_U32)
            .to_le_bytes(),
    );
    hash.update(color.output_color_shape_id().to_le_bytes());
    hash_len(
        &mut hash,
        color.components().len(),
        "current color components",
    )?;
    for component in color.components() {
        hash.update([component.kind() as u8]);
        hash_len(&mut hash, component.source_slots().len(), "color component")?;
        for slot in component.source_slots() {
            hash.update(slot.to_le_bytes());
        }
    }
    hash_len(
        &mut hash,
        color.result_port_bindings().len(),
        "current color ports",
    )?;
    for binding in color.result_port_bindings() {
        hash.update(binding.component_index().to_le_bytes());
        hash.update([binding.endpoint() as u8]);
    }
    final_digest(hash)
}

pub(super) fn multiset_digest(
    domain: &[u8],
    mut rows: Vec<SemanticDigest>,
) -> RusticolResult<SemanticDigest> {
    rows.sort_unstable();
    let mut hash = Sha256::new();
    hash.update(domain);
    hash_len(&mut hash, rows.len(), "proof rows")?;
    for row in rows {
        hash_digest(&mut hash, row);
    }
    final_digest(hash)
}

pub(super) fn contribution_proof_digest(
    result_digest: SemanticDigest,
    parent_digests: [SemanticDigest; 2],
    key: &ContributionKey,
    factor: ExactComplexRational,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-on-the-fly-contribution-row-v1\0");
    hash_digest(&mut hash, result_digest);
    hash_digest(&mut hash, parent_digests[0]);
    hash_digest(&mut hash, parent_digests[1]);
    hash.update(key.transition_template_id().to_le_bytes());
    hash.update(key.result_state_template_id().to_le_bytes());
    hash.update(key.quantum_flow_witness_id().to_le_bytes());
    hash.update(
        key.color_witness_term_id()
            .color_contraction_template_id()
            .to_le_bytes(),
    );
    hash.update(key.color_witness_term_id().witness_ordinal().to_le_bytes());
    hash_digest(&mut hash, key.runtime_coupling_binding_digest());
    hash.update(key.output_projection_id().to_le_bytes());
    hash_exact(&mut hash, factor);
    final_digest(hash)
}

fn closure_proof_digest(
    parent_digests: [SemanticDigest; 2],
    closure: &PendingClosure,
) -> RusticolResult<SemanticDigest> {
    let mut hash = Sha256::new();
    hash.update(b"pyamplicol-on-the-fly-closure-row-v1\0");
    hash_digest(&mut hash, parent_digests[0]);
    hash_digest(&mut hash, parent_digests[1]);
    hash.update(closure.key.closure_template_id.to_le_bytes());
    hash.update(
        closure
            .key
            .quantum_flow_template_id
            .unwrap_or(MISSING_U32)
            .to_le_bytes(),
    );
    hash.update(
        closure
            .key
            .color_witness_term_id
            .color_contraction_template_id()
            .to_le_bytes(),
    );
    hash.update(
        closure
            .key
            .color_witness_term_id
            .witness_ordinal()
            .to_le_bytes(),
    );
    hash_exact(&mut hash, closure.factor);
    hash_len(
        &mut hash,
        closure.component_coefficients.len(),
        "closure component factors",
    )?;
    for coefficient in &closure.component_coefficients {
        hash_exact(&mut hash, *coefficient);
    }
    final_digest(hash)
}

fn intern_factor(
    factor: ExactComplexRational,
    factors: &mut Vec<ExactComplexRational>,
    ids: &mut BTreeMap<ExactComplexRational, u32>,
) -> RusticolResult<u32> {
    if let Some(id) = ids.get(&factor).copied() {
        return Ok(id);
    }
    let id = checked_u32(factors.len(), "exact factor count")?;
    factors.push(factor);
    ids.insert(factor, id);
    Ok(id)
}

fn intern_momentum(
    form: &CanonicalMomentumLinearForm,
    forms: &mut Vec<CanonicalMomentumLinearForm>,
    ids: &mut BTreeMap<CanonicalMomentumLinearForm, u32>,
) -> RusticolResult<u32> {
    if let Some(id) = ids.get(form).copied() {
        return Ok(id);
    }
    let id = checked_u32(forms.len(), "momentum form count")?;
    forms.push(form.clone());
    ids.insert(form.clone(), id);
    Ok(id)
}

fn operation_key(
    seed: &OnTheFlyProcessSeedV1,
    role: DirectExecutorRole,
    kind: OnTheFlyOperationKindV1,
    operation_id: u32,
    operation_semantic_digest: SemanticDigest,
    evaluator_binding_id: u32,
    evaluator_binding_semantic_digest: SemanticDigest,
) -> RusticolResult<OnTheFlyExecutorKeyV1> {
    OnTheFlyExecutorKeyV1::new(
        seed.direct_catalog_digest,
        role,
        kind,
        Some(operation_id),
        operation_semantic_digest,
        Some(evaluator_binding_id),
        evaluator_binding_semantic_digest,
    )
}

fn source_executor_key(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    seed: &OnTheFlyProcessSeedV1,
    source_template_id: u32,
) -> RusticolResult<OnTheFlyExecutorKeyV1> {
    let source = templates
        .input()
        .sources
        .get(source_template_id as usize)
        .ok_or_else(|| integrity("source executor template is absent"))?;
    let binding = templates
        .input()
        .evaluator_bindings
        .get(source.evaluator_binding_id as usize)
        .ok_or_else(|| integrity("source executor binding is absent"))?;
    operation_key(
        seed,
        DirectExecutorRole::Source,
        OnTheFlyOperationKindV1::Source,
        source.id,
        catalog.digest(source.semantic_digest_id, "source semantic")?,
        binding.id,
        catalog.digest(binding.semantic_digest_id, "source evaluator semantic")?,
    )
}

fn transition_executor_key(
    seed: &OnTheFlyProcessSeedV1,
    transition: &PreparedTransition,
) -> RusticolResult<OnTheFlyExecutorKeyV1> {
    operation_key(
        seed,
        DirectExecutorRole::Contribution,
        OnTheFlyOperationKindV1::Transition,
        transition.row.id,
        transition.transition_semantic_digest,
        transition.row.evaluator_binding_id,
        transition.evaluator_binding_digest,
    )
}

fn propagator_executor_key(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    seed: &OnTheFlyProcessSeedV1,
    propagator_template_id: u32,
) -> RusticolResult<OnTheFlyExecutorKeyV1> {
    let row = templates
        .input()
        .propagators
        .get(propagator_template_id as usize)
        .ok_or_else(|| integrity("propagator executor template is absent"))?;
    let binding = templates
        .input()
        .evaluator_bindings
        .get(row.evaluator_binding_id as usize)
        .ok_or_else(|| integrity("propagator executor binding is absent"))?;
    operation_key(
        seed,
        DirectExecutorRole::Finalization,
        OnTheFlyOperationKindV1::Propagator,
        row.id,
        catalog.digest(row.semantic_digest_id, "propagator semantic")?,
        binding.id,
        catalog.digest(binding.semantic_digest_id, "propagator evaluator semantic")?,
    )
}

fn closure_executor_key(
    seed: &OnTheFlyProcessSeedV1,
    closure: &PreparedClosure,
) -> RusticolResult<OnTheFlyExecutorKeyV1> {
    operation_key(
        seed,
        DirectExecutorRole::Closure,
        OnTheFlyOperationKindV1::Closure,
        closure.row.id,
        closure.closure_semantic_digest,
        closure.row.evaluator_binding_id,
        closure.evaluator_binding_digest,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn lower_trace(
    templates: &ValidatedRecurrenceTemplateInput,
    catalog: &TemplateCatalog<'_>,
    seed: &OnTheFlyProcessSeedV1,
    query: &DecodedLcQueryV1,
    colors: &DynamicLCColorStateInterner,
    currents: &[PendingCurrent],
    closures: &[PendingClosure],
    pairing_owner: &ResolvedPairingOwnerV1,
    transitions: &BTreeMap<(u32, u32), Vec<PreparedTransition>>,
    prepared_closures: &BTreeMap<(u32, u32), Vec<PreparedClosure>>,
    live: &BTreeSet<u32>,
    constructed_contribution_count: usize,
) -> RusticolResult<OnTheFlyStructuralTraceV1> {
    let input = templates.input();
    let mut old_to_new = BTreeMap::new();
    let mut current_keys = Vec::with_capacity(live.len());
    let mut current_colors = Vec::with_capacity(live.len());
    let mut component_bases = BTreeMap::new();
    let mut component_counts = BTreeMap::new();
    let mut current_component_count = 0u32;
    for old_id in live.iter().copied() {
        let current = currents
            .get(old_id as usize)
            .ok_or_else(|| integrity("live current is absent"))?;
        let state = input
            .current_states
            .get(current.key.current_state_template_id() as usize)
            .ok_or_else(|| integrity("live current state template is absent"))?;
        let count = u16::try_from(state.dimension)
            .map_err(|_| invalid("on-the-fly current component count exceeds u16"))?;
        let new_id = checked_u32(current_keys.len(), "live current count")?;
        old_to_new.insert(old_id, new_id);
        component_bases.insert(old_id, current_component_count);
        component_counts.insert(old_id, count);
        current_component_count = current_component_count
            .checked_add(u32::from(count))
            .ok_or_else(|| invalid("on-the-fly current arena exceeds u32 components"))?;
        current_keys.push(current.key.clone());
        current_colors.push(
            colors
                .get(current.key.dynamic_lc_color_state_id())
                .ok_or_else(|| integrity("live current color state is absent"))?
                .clone(),
        );
    }

    let mut momentum_forms = Vec::new();
    let mut momentum_ids = BTreeMap::new();
    for old_id in live.iter().copied() {
        intern_momentum(
            currents[old_id as usize].key.momentum(),
            &mut momentum_forms,
            &mut momentum_ids,
        )?;
    }
    let mut exact_factors = Vec::new();
    let mut factor_ids = BTreeMap::new();
    intern_factor(
        ExactComplexRational::ONE,
        &mut exact_factors,
        &mut factor_ids,
    )?;
    let mut operations = Vec::new();

    for old_id in live
        .iter()
        .copied()
        .filter(|old_id| currents[*old_id as usize].key.node_kind() == RecurrenceNodeKind::Source)
    {
        let current = &currents[old_id as usize];
        let source_template_id = match current.key.source_binding() {
            CurrentSourceBinding::FixedTemplate(id) => *id,
            _ => return Err(integrity("selected source has no fixed template binding")),
        };
        operations.push(OnTheFlyTraceOperationV1::Source {
            key: source_executor_key(templates, catalog, seed, source_template_id)?,
            row: DirectSourceRow {
                source_slot: current.key.support_source_slots()[0],
                destination_component_base: component_bases[&old_id],
                momentum_form_id: momentum_ids[current.key.momentum()],
                source_template_or_dispatch_domain: source_template_id,
                spin_state_class: current.key.spin_state_class(),
                exact_factor_id: intern_factor(
                    current
                        .source_factor
                        .ok_or_else(|| integrity("source current has no exact factor"))?,
                    &mut exact_factors,
                    &mut factor_ids,
                )?,
                selector_domain_id: 0,
            },
        });
    }

    let maximum_stage = live
        .iter()
        .map(|id| currents[*id as usize].stage)
        .max()
        .unwrap_or(0);
    let mut contribution_proof_rows = Vec::new();
    let current_digest_by_old = live
        .iter()
        .copied()
        .map(|old_id| {
            let current = &currents[old_id as usize];
            Ok((
                old_id,
                hash_current_key(
                    &current.key,
                    colors
                        .get(current.key.dynamic_lc_color_state_id())
                        .ok_or_else(|| integrity("proof current color state is absent"))?,
                )?,
            ))
        })
        .collect::<RusticolResult<BTreeMap<_, _>>>()?;

    for stage in 1..=maximum_stage {
        for old_id in live
            .iter()
            .copied()
            .filter(|old_id| currents[*old_id as usize].stage == stage)
        {
            let current = &currents[old_id as usize];
            let mut initialized = false;
            for (pending, factor) in current.contributions.iter().filter(|(pending, factor)| {
                !factor.is_zero()
                    && pending
                        .parent_current_ids
                        .iter()
                        .all(|parent| live.contains(parent))
            }) {
                let transition_id = pending.key.transition_template_id();
                let prepared = transitions
                    .values()
                    .flatten()
                    .find(|prepared| prepared.row.id == transition_id)
                    .ok_or_else(|| integrity("prepared live transition disappeared"))?;
                let parent_ids = pending.parent_current_ids;
                operations.push(OnTheFlyTraceOperationV1::Contribution {
                    key: transition_executor_key(seed, prepared)?,
                    row: DirectContributionRow {
                        parent0_component_base: component_bases[&parent_ids[0]],
                        parent1_component_base_or_sentinel: component_bases[&parent_ids[1]],
                        parent0_momentum_form_id: momentum_ids
                            [currents[parent_ids[0] as usize].key.momentum()],
                        parent1_momentum_form_id_or_sentinel: momentum_ids
                            [currents[parent_ids[1] as usize].key.momentum()],
                        destination_component_base: component_bases[&old_id],
                        exact_factor_id: intern_factor(
                            *factor,
                            &mut exact_factors,
                            &mut factor_ids,
                        )?,
                        selector_domain_id: 0,
                        flags: if initialized {
                            0
                        } else {
                            DIRECT_CONTRIBUTION_FLAG_INITIALIZE_DESTINATION
                        },
                    },
                });
                initialized = true;
                contribution_proof_rows.push(contribution_proof_digest(
                    current_digest_by_old[&old_id],
                    [
                        current_digest_by_old[&parent_ids[0]],
                        current_digest_by_old[&parent_ids[1]],
                    ],
                    &pending.key,
                    *factor,
                )?);
            }
            if !initialized {
                return Err(integrity(
                    "live non-source current has no live contribution",
                ));
            }
            if let Some(propagator_id) = current.key.propagator_template_id() {
                let component_count = component_counts[&old_id];
                operations.push(OnTheFlyTraceOperationV1::Finalization {
                    key: propagator_executor_key(templates, catalog, seed, propagator_id)?,
                    row: DirectFinalizationRow {
                        component_base: component_bases[&old_id],
                        component_count,
                        momentum_form_id: momentum_ids[current.key.momentum()],
                        exact_factor_id: factor_ids[&ExactComplexRational::ONE],
                        selector_domain_id: 0,
                        flags: 0,
                    },
                });
            }
        }
    }

    let mut closure_proof_rows = Vec::new();
    for closure in closures {
        let prepared = prepared_closures
            .values()
            .flatten()
            .find(|prepared| prepared.row.id == closure.key.closure_template_id)
            .ok_or_else(|| integrity("prepared live closure disappeared"))?;
        let parent_ids = closure.key.parent_current_ids;
        let parent0_count = component_counts[&parent_ids[0]];
        let parent1_count = component_counts[&parent_ids[1]];
        if parent0_count != parent1_count
            || closure.component_coefficients.len() != usize::from(parent0_count)
        {
            return Err(integrity(
                "closure component coefficients do not match parent dimensions",
            ));
        }
        let component_factor_start = checked_u32(exact_factors.len(), "closure factor start")?;
        exact_factors.extend_from_slice(&closure.component_coefficients);
        operations.push(OnTheFlyTraceOperationV1::Closure {
            key: closure_executor_key(seed, prepared)?,
            row: DirectClosureRow {
                parent0_component_base: component_bases[&parent_ids[0]],
                parent1_component_base_or_sentinel: component_bases[&parent_ids[1]],
                parent0_momentum_form_id: momentum_ids
                    [currents[parent_ids[0] as usize].key.momentum()],
                parent1_momentum_form_id_or_sentinel: momentum_ids
                    [currents[parent_ids[1] as usize].key.momentum()],
                amplitude_destination_id: 0,
                exact_factor_id: intern_factor(
                    closure.factor,
                    &mut exact_factors,
                    &mut factor_ids,
                )?,
                component_factor_start,
                component_count: parent0_count,
                selector_domain_id: 0,
                flags: 0,
            },
        });
        closure_proof_rows.push(closure_proof_digest(
            [
                current_digest_by_old[&parent_ids[0]],
                current_digest_by_old[&parent_ids[1]],
            ],
            closure,
        )?);
    }

    let current_multiset_digest = multiset_digest(
        b"pyamplicol-on-the-fly-current-multiset-v1\0",
        current_digest_by_old.values().copied().collect(),
    )?;
    let contribution_multiset_digest = multiset_digest(
        b"pyamplicol-on-the-fly-contribution-multiset-v1\0",
        contribution_proof_rows,
    )?;
    let closure_multiset_digest = multiset_digest(
        b"pyamplicol-on-the-fly-closure-multiset-v1\0",
        closure_proof_rows,
    )?;
    let mut owner_hash = Sha256::new();
    owner_hash.update(b"pyamplicol-on-the-fly-owner-v1\0");
    hash_digest(&mut owner_hash, seed.semantic_digest());
    hash_digest(&mut owner_hash, query.semantic_digest());
    owner_hash.update(query.closure_anchor_slot.to_le_bytes());
    hash_digest(&mut owner_hash, query.selector_digest);
    match pairing_owner.proof_digest {
        None => owner_hash.update([0]),
        Some(digest) => {
            owner_hash.update([1]);
            hash_digest(&mut owner_hash, digest);
        }
    }
    let owner_digest = final_digest(owner_hash)?;
    let mut proof_hash = Sha256::new();
    proof_hash.update(ON_THE_FLY_PROOF_DOMAIN);
    for digest in [
        current_multiset_digest,
        contribution_multiset_digest,
        closure_multiset_digest,
        owner_digest,
    ] {
        hash_digest(&mut proof_hash, digest);
    }
    let live_contribution_count = operations
        .iter()
        .filter(|operation| matches!(operation, OnTheFlyTraceOperationV1::Contribution { .. }))
        .count();
    for value in [
        live.len(),
        live_contribution_count,
        closures.len(),
        currents.len(),
        constructed_contribution_count,
    ] {
        proof_hash.update(checked_u32(value, "structural proof count")?.to_le_bytes());
    }
    let proof = OnTheFlyStructuralProofV1 {
        current_count: checked_u32(live.len(), "live current count")?,
        contribution_count: checked_u32(live_contribution_count, "live contribution count")?,
        closure_count: checked_u32(closures.len(), "live closure count")?,
        constructed_current_count: checked_u32(currents.len(), "constructed current count")?,
        constructed_contribution_count: checked_u32(
            constructed_contribution_count,
            "constructed contribution count",
        )?,
        current_multiset_digest,
        contribution_multiset_digest,
        closure_multiset_digest,
        owner_digest,
        semantic_digest: final_digest(proof_hash)?,
    };
    let exact_factor_count = checked_u32(exact_factors.len(), "exact factor count")?;
    let layout = OnTheFlyWorkspaceLayoutV1 {
        source_count: checked_u32(query.selected_sources.len(), "selected source count")?,
        lorentz_component_count: 4,
        parameter_count: checked_u32(templates.input().parameters.len(), "parameter count")?,
        current_component_count,
        amplitude_component_count: 1,
        momentum_form_count: checked_u32(momentum_ids.len(), "momentum form count")?,
        exact_factor_count,
    };
    let mut trace_hash = Sha256::new();
    trace_hash.update(b"pyamplicol-on-the-fly-structural-trace-v1\0");
    hash_digest(&mut trace_hash, seed.semantic_digest());
    hash_digest(&mut trace_hash, query.semantic_digest());
    hash_digest(&mut trace_hash, proof.semantic_digest());
    for value in [
        layout.source_count,
        u32::from(layout.lorentz_component_count),
        layout.parameter_count,
        layout.current_component_count,
        layout.amplitude_component_count,
        layout.momentum_form_count,
        layout.exact_factor_count,
    ] {
        trace_hash.update(value.to_le_bytes());
    }
    let semantic_digest = final_digest(trace_hash)?;
    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    let current_component_ranges = live
        .iter()
        .copied()
        .map(|old_id| {
            [
                component_bases[&old_id],
                u32::from(component_counts[&old_id]),
            ]
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    #[cfg(any(test, feature = "on-the-fly-test-support"))]
    let current_semantic_digests = live
        .iter()
        .copied()
        .map(|old_id| current_digest_by_old[&old_id])
        .collect::<Vec<_>>()
        .into_boxed_slice();
    Ok(OnTheFlyStructuralTraceV1 {
        seed_digest: seed.semantic_digest(),
        query_digest: query.semantic_digest(),
        current_keys: current_keys.into_boxed_slice(),
        current_colors: current_colors.into_boxed_slice(),
        operations: operations.into_boxed_slice(),
        momentum_forms: momentum_forms.into_boxed_slice(),
        exact_factors: exact_factors.into_boxed_slice(),
        pairing_owner: pairing_owner.clone(),
        layout,
        proof,
        semantic_digest,
        #[cfg(any(test, feature = "on-the-fly-test-support"))]
        current_component_ranges,
        #[cfg(any(test, feature = "on-the-fly-test-support"))]
        current_semantic_digests,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> SemanticDigest {
        SemanticDigest::new([byte; 32]).unwrap()
    }

    #[test]
    fn logical_batch_capacity_belongs_to_workspace_not_structural_trace() {
        let trace = OnTheFlyStructuralTraceV1 {
            seed_digest: digest(1),
            query_digest: digest(2),
            current_keys: Box::new([]),
            current_colors: Box::new([]),
            operations: Box::new([]),
            momentum_forms: vec![
                CanonicalMomentumLinearForm::new(vec![MomentumTerm {
                    source_slot: 0,
                    coefficient: 1,
                }])
                .unwrap(),
            ]
            .into_boxed_slice(),
            exact_factors: vec![ExactComplexRational::ONE].into_boxed_slice(),
            pairing_owner: ResolvedPairingOwnerV1 {
                endpoint_pairs: Box::new([]),
                proof_digest: None,
                source_slot_permutation: Box::new([]),
                source_lineage: Box::new([]),
                fermion_parity: 1,
            },
            layout: OnTheFlyWorkspaceLayoutV1 {
                source_count: 1,
                lorentz_component_count: 4,
                parameter_count: 0,
                current_component_count: 1,
                amplitude_component_count: 1,
                momentum_form_count: 1,
                exact_factor_count: 1,
            },
            proof: OnTheFlyStructuralProofV1 {
                current_count: 0,
                contribution_count: 0,
                closure_count: 0,
                constructed_current_count: 0,
                constructed_contribution_count: 0,
                current_multiset_digest: digest(3),
                contribution_multiset_digest: digest(4),
                closure_multiset_digest: digest(5),
                owner_digest: digest(6),
                semantic_digest: digest(7),
            },
            semantic_digest: digest(8),
            current_component_ranges: Box::new([]),
            current_semantic_digests: Box::new([]),
        };
        let small = OnTheFlyWorkspaceV1::new(&trace, 1).unwrap();
        let large = OnTheFlyWorkspaceV1::new(&trace, 65).unwrap();
        assert_ne!(small.point_stride(), large.point_stride());
        assert_eq!(trace.semantic_digest(), digest(8));
    }
}
