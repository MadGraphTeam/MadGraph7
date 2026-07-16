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
}
