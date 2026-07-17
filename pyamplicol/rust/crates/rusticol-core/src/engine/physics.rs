// SPDX-License-Identifier: 0BSD

use super::*;

impl PhysicsRuntime {
    pub(super) fn new(manifest: ProcessPhysicsV1) -> RusticolResult<Self> {
        manifest.validate()?;
        let helicity_index_by_id = manifest
            .helicities
            .iter()
            .enumerate()
            .map(|(index, helicity)| (helicity.id.clone(), index))
            .collect::<BTreeMap<_, _>>();
        let color_index_by_id = manifest
            .color_components
            .iter()
            .enumerate()
            .map(|(index, color)| (color.id().to_string(), index))
            .collect::<BTreeMap<_, _>>();
        let mut reduction_by_group_id = BTreeMap::new();
        for group in &manifest.reduction.groups {
            let group_id = parse_reduction_group_id(&group.id)?;
            let mut expected_helicity_weight = None;
            for id in &group.physical_helicity_ids {
                let index = helicity_index_by_id[id];
                let coefficient = manifest.helicities[index].coefficient;
                if let Some(expected) = expected_helicity_weight {
                    if coefficient.to_bits() != expected {
                        return Err(RusticolError::artifact(format!(
                            "resolved reduction group {} has inconsistent helicity weights",
                            group.id
                        )));
                    }
                } else {
                    expected_helicity_weight = Some(coefficient.to_bits());
                }
            }
            if reduction_by_group_id
                .insert(group_id, group.clone())
                .is_some()
            {
                return Err(RusticolError::artifact(format!(
                    "resolved reduction groups map to duplicate evaluator id {group_id}"
                )));
            }
        }
        Ok(Self {
            manifest,
            helicity_index_by_id,
            color_index_by_id,
            reduction_by_group_id,
        })
    }

    pub(super) fn selected_helicity_indices(
        &self,
        ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<Vec<usize>> {
        self.select_indices(ids, &self.helicity_index_by_id, "helicity")
    }

    pub(super) fn selected_color_indices(
        &self,
        ids: Option<&BTreeSet<String>>,
    ) -> RusticolResult<Vec<usize>> {
        self.select_indices(ids, &self.color_index_by_id, "color component")
    }

    pub(super) fn select_indices(
        &self,
        ids: Option<&BTreeSet<String>>,
        available: &BTreeMap<String, usize>,
        kind: &str,
    ) -> RusticolResult<Vec<usize>> {
        let Some(ids) = ids else {
            return Ok((0..available.len()).collect());
        };
        let mut indices = Vec::with_capacity(ids.len());
        for id in ids {
            let index = available.get(id).ok_or_else(|| {
                RusticolError::selector(format!("unknown resolved {kind} id {id:?}"))
            })?;
            indices.push(*index);
        }
        indices.sort_unstable();
        Ok(indices)
    }

    pub(super) fn has_contracted_color_axis(&self) -> bool {
        matches!(
            self.manifest.color_components.as_slice(),
            [PhysicsColorComponentV1::ContractedColor(_)]
        )
    }

    pub(super) fn color_is_computed(&self, index: usize) -> bool {
        match &self.manifest.color_components[index] {
            PhysicsColorComponentV1::LcFlow(flow) => flow.computed,
            PhysicsColorComponentV1::ContractedColor(_) => true,
        }
    }

    pub(super) fn normalized_helicity_weights(
        &self,
        group: &crate::ReductionGroup,
    ) -> RusticolResult<Vec<(usize, f64)>> {
        let mut weights = Vec::with_capacity(group.physical_helicity_ids.len());
        let mut total = 0.0;
        for id in &group.physical_helicity_ids {
            let index = self.helicity_index_by_id[id];
            let coefficient = self.manifest.helicities[index].coefficient;
            total += coefficient;
            weights.push((index, coefficient));
        }
        if !total.is_finite() || total <= 0.0 {
            return Err(RusticolError::artifact(format!(
                "resolved reduction group {} has no positive helicity weight",
                group.id
            )));
        }
        for (_, weight) in &mut weights {
            *weight /= total;
        }
        Ok(weights)
    }

    pub(super) fn normalized_member_weights(
        &self,
        group: &crate::ReductionGroup,
    ) -> RusticolResult<Vec<(usize, usize, f64)>> {
        let mut weights =
            Vec::with_capacity(group.physical_helicity_ids.len() * group.physical_color_ids.len());
        let mut total = 0.0;
        for helicity_id in &group.physical_helicity_ids {
            let helicity_index = self.helicity_index_by_id[helicity_id];
            let helicity_weight = self.manifest.helicities[helicity_index].coefficient;
            for color_id in &group.physical_color_ids {
                let color_index = self.color_index_by_id[color_id];
                let weight =
                    helicity_weight * self.manifest.color_components[color_index].coefficient();
                total += weight;
                weights.push((helicity_index, color_index, weight));
            }
        }
        if !total.is_finite() || total <= 0.0 {
            return Err(RusticolError::artifact(format!(
                "resolved reduction group {} has no positive member weight",
                group.id
            )));
        }
        for (_, _, weight) in &mut weights {
            *weight /= total;
        }
        Ok(weights)
    }

    pub(super) fn lc_resolved_replay_plan(
        &self,
        mappings: &LcTopologyReplayMappings,
        replay_weights: &[f64],
    ) -> RusticolResult<LcResolvedReplayPlan> {
        if self.manifest.color_accuracy.as_str() != "lc" {
            return Err(RusticolError::artifact(
                "LC topology replay requires LC resolved physics metadata",
            ));
        }
        if mappings.len() != replay_weights.len() {
            return Err(RusticolError::artifact(
                "LC topology replay mappings and weights have inconsistent lengths",
            ));
        }

        let helicity_count = self.manifest.helicities.len();
        let color_count = self.manifest.color_components.len();
        let mut relevant_helicities = BTreeSet::new();
        let mut relevant_colors = BTreeSet::new();
        for reduction in self.reduction_by_group_id.values() {
            for id in &reduction.physical_helicity_ids {
                relevant_helicities.insert(self.helicity_index_by_id[id]);
            }
            for id in &reduction.physical_color_ids {
                relevant_colors.insert(self.color_index_by_id[id]);
            }
        }
        if relevant_helicities.is_empty() || relevant_colors.is_empty() {
            return Err(RusticolError::artifact(
                "LC topology replay has no materialized resolved members",
            ));
        }

        let helicity_index_by_values = self
            .manifest
            .helicities
            .iter()
            .enumerate()
            .map(|(index, helicity)| (helicity.values.clone(), index))
            .collect::<BTreeMap<_, _>>();
        if helicity_index_by_values.len() != helicity_count {
            return Err(RusticolError::artifact(
                "resolved physics contains duplicate helicity vectors",
            ));
        }
        let mut color_index_by_word = BTreeMap::new();
        for (index, color) in self.manifest.color_components.iter().enumerate() {
            let PhysicsColorComponentV1::LcFlow(flow) = color else {
                return Err(RusticolError::artifact(
                    "LC topology replay encountered a contracted color component",
                ));
            };
            if color_index_by_word
                .insert(flow.word.clone(), index)
                .is_some()
            {
                return Err(RusticolError::artifact(
                    "resolved physics contains duplicate LC flow words",
                ));
            }
        }

        let mut entries = Vec::with_capacity(mappings.len());
        for (mapping, replay_weight) in mappings.iter().zip(replay_weights.iter().copied()) {
            validate_public_label_permutation(mapping, self.manifest.external_particles.len())?;
            let mut helicity_targets = BTreeMap::new();
            for source_index in &relevant_helicities {
                let source = &self.manifest.helicities[*source_index];
                let target_values = permute_public_helicity(&source.values, mapping);
                let target_index = *helicity_index_by_values.get(&target_values).ok_or_else(|| {
                    RusticolError::compatibility(format!(
                        "resolved physics is missing replayed helicity vector {target_values:?}; regenerate the artifact with complete topology-replay reductions"
                    ))
                })?;
                let target = &self.manifest.helicities[target_index];
                if source.coefficient.to_bits() != target.coefficient.to_bits() {
                    return Err(RusticolError::artifact(format!(
                        "LC topology replay maps helicity {} to {} with a different reduction coefficient",
                        source.id, target.id
                    )));
                }
                helicity_targets.insert(*source_index, target_index);
            }

            let mut color_targets = BTreeMap::new();
            for source_index in &relevant_colors {
                let PhysicsColorComponentV1::LcFlow(source) =
                    &self.manifest.color_components[*source_index]
                else {
                    unreachable!("LC color axes checked above");
                };
                let target_word = permute_public_color_word(&source.word, mapping);
                let target_words = replay_public_color_words(&target_word, replay_weight)?;
                let mut target_coefficients = BTreeMap::new();
                for word in target_words {
                    let target_index = *color_index_by_word.get(&word).ok_or_else(|| {
                        RusticolError::compatibility(format!(
                            "resolved physics is missing replayed LC flow word {word:?}; regenerate the artifact with replay-to-public-flow reductions"
                        ))
                    })?;
                    let coefficient = self.manifest.color_components[target_index].coefficient();
                    if !coefficient.is_finite() || coefficient <= 0.0 {
                        return Err(RusticolError::artifact(format!(
                            "replayed LC flow {} has no positive reduction coefficient",
                            self.manifest.color_components[target_index].id()
                        )));
                    }
                    target_coefficients
                        .entry(target_index)
                        .or_insert(coefficient);
                }
                let total = target_coefficients.values().sum::<f64>();
                if !total.is_finite() || total <= 0.0 {
                    return Err(RusticolError::artifact(
                        "LC topology replay has no positive public-flow reduction weight",
                    ));
                }
                color_targets.insert(
                    *source_index,
                    target_coefficients
                        .into_iter()
                        .map(|(target_index, coefficient)| {
                            (target_index, replay_weight * coefficient / total)
                        })
                        .collect::<Vec<_>>(),
                );
            }

            let mut routes = Vec::new();
            for (source_helicity, target_helicity) in &helicity_targets {
                for source_color in &relevant_colors {
                    for (target_color, weight) in &color_targets[source_color] {
                        routes.push(LcResolvedReplayRoute {
                            source_index: source_helicity * color_count + source_color,
                            target_index: target_helicity * color_count + target_color,
                            weight: *weight,
                        });
                    }
                }
            }
            entries.push(LcResolvedReplayEntry { routes });
        }
        Ok(LcResolvedReplayPlan {
            entries,
            helicity_count,
            color_count,
        })
    }
}

impl ExecutionRuntime {
    pub(super) fn remap_lc_topology_replay_public_labels(
        &mut self,
        representative_to_public: &[usize],
    ) -> RusticolResult<()> {
        if !self.lc_topology_replay_enabled {
            return Ok(());
        }
        if representative_to_public.len() != self.external_count
            || representative_to_public
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
                != (0..self.external_count).collect::<BTreeSet<_>>()
        {
            return Err(RusticolError::artifact(
                "process alias has an invalid public-label permutation for LC topology replay",
            ));
        }
        self.lc_topology_replay_public_mappings = self
            .lc_topology_replay_mappings
            .iter()
            .map(|mapping| {
                mapping
                    .iter()
                    .map(|(representative, sector)| {
                        (
                            representative_to_public[*representative],
                            representative_to_public[*sector],
                        )
                    })
                    .collect()
            })
            .collect();
        Ok(())
    }
}

fn validate_public_label_permutation(
    mapping: &[(usize, usize)],
    external_count: usize,
) -> RusticolResult<()> {
    let mut representatives = BTreeSet::new();
    let mut sectors = BTreeSet::new();
    for (representative, sector) in mapping {
        if *representative >= external_count || *sector >= external_count {
            return Err(RusticolError::artifact(
                "LC topology replay public-label mapping references an out-of-range external leg",
            ));
        }
        if !representatives.insert(*representative) || !sectors.insert(*sector) {
            return Err(RusticolError::artifact(
                "LC topology replay public-label mapping is not one-to-one",
            ));
        }
    }
    if representatives != sectors {
        return Err(RusticolError::artifact(
            "LC topology replay public-label mapping is not a permutation of its support",
        ));
    }
    Ok(())
}

fn permute_public_helicity(values: &[i32], mapping: &[(usize, usize)]) -> Vec<i32> {
    let mut target = values.to_vec();
    for (representative, sector) in mapping {
        target[*sector] = values[*representative];
    }
    target
}

fn permute_public_color_word(word: &[usize], mapping: &[(usize, usize)]) -> Vec<usize> {
    let labels = mapping.iter().copied().collect::<BTreeMap<_, _>>();
    word.iter()
        .map(|label| {
            label
                .checked_sub(1)
                .and_then(|index| labels.get(&index).copied())
                .map(|index| index + 1)
                .unwrap_or(*label)
        })
        .collect()
}

fn replay_public_color_words(
    word: &[usize],
    replay_weight: f64,
) -> RusticolResult<Vec<Vec<usize>>> {
    const WEIGHT_TOLERANCE: f64 = 1.0e-12;
    if (replay_weight - 1.0).abs() <= WEIGHT_TOLERANCE {
        return Ok(vec![word.to_vec()]);
    }
    if (replay_weight - 2.0).abs() > WEIGHT_TOLERANCE {
        return Err(RusticolError::compatibility(format!(
            "resolved LC topology replay cannot expand sector weight {replay_weight}; schema-v3 replay metadata defines public-flow expansion only for unit sectors and folded trace-reflection weight two"
        )));
    }
    if word.len() < 2 {
        return Ok(vec![word.to_vec()]);
    }
    let mut reflected = Vec::with_capacity(word.len());
    reflected.push(word[0]);
    reflected.extend(word[1..].iter().rev().copied());
    if reflected == word {
        Ok(vec![word.to_vec()])
    } else {
        Ok(vec![word.to_vec(), reflected])
    }
}
