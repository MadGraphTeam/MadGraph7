// SPDX-License-Identifier: 0BSD

use std::collections::{BTreeSet, HashMap};

use crate::{RusticolError, RusticolResult};

use super::{DynamicLCColorStateId, ExactComplexRational, SemanticDigest};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

/// One ordered component in a dynamic partial LC color forest.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum LCColorComponentKind {
    OpenString = 0,
    AdjointSegment = 1,
    Trace = 2,
}

impl TryFrom<u8> for LCColorComponentKind {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::OpenString),
            1 => Ok(Self::AdjointSegment),
            2 => Ok(Self::Trace),
            _ => Err(invalid(format!(
                "unsupported LC color component kind {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum LCColorComponentRole {
    Active = 0,
    Passive = 1,
    None = 2,
}

impl TryFrom<u8> for LCColorComponentRole {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Active),
            1 => Ok(Self::Passive),
            2 => Ok(Self::None),
            _ => Err(invalid(format!(
                "unsupported LC color component role {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum LCColorSourceSeedOperation {
    Empty = 0,
    Singleton = 1,
}

impl TryFrom<u8> for LCColorSourceSeedOperation {
    type Error = RusticolError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Empty),
            1 => Ok(Self::Singleton),
            _ => Err(invalid(format!(
                "unsupported LC source seed operation {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct LCColorSourceSeed {
    operation: LCColorSourceSeedOperation,
    output_color_shape_id: u32,
    component_kind: Option<LCColorComponentKind>,
    component_role: LCColorComponentRole,
    proof_digest: SemanticDigest,
}

impl LCColorSourceSeed {
    pub fn new(
        operation: LCColorSourceSeedOperation,
        output_color_shape_id: u32,
        component_kind: Option<LCColorComponentKind>,
        component_role: LCColorComponentRole,
        proof_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        match operation {
            LCColorSourceSeedOperation::Empty
                if component_kind.is_some() || component_role != LCColorComponentRole::None =>
            {
                return Err(invalid(
                    "an empty LC source seed cannot declare a component",
                ));
            }
            LCColorSourceSeedOperation::Singleton
                if component_kind.is_none() || component_role == LCColorComponentRole::None =>
            {
                return Err(invalid(
                    "a singleton LC source seed requires component kind and role",
                ));
            }
            _ => {}
        }
        Ok(Self {
            operation,
            output_color_shape_id,
            component_kind,
            component_role,
            proof_digest,
        })
    }

    pub fn instantiate(self, source_slot: u32) -> RusticolResult<DynamicLCColorState> {
        match self.operation {
            LCColorSourceSeedOperation::Empty => {
                DynamicLCColorState::new(self.output_color_shape_id, None, Vec::new())
            }
            LCColorSourceSeedOperation::Singleton => {
                let component = LCColorComponent::new(
                    self.component_kind
                        .expect("validated singleton component kind"),
                    vec![source_slot],
                )?;
                let active = (self.component_role == LCColorComponentRole::Active).then_some(0);
                DynamicLCColorState::new(self.output_color_shape_id, active, vec![component])
            }
        }
    }

    pub const fn proof_digest(self) -> SemanticDigest {
        self.proof_digest
    }
}

/// Canonical ordered colored-source word for one LC component.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct LCColorComponent {
    kind: LCColorComponentKind,
    source_slots: Box<[u32]>,
}

impl LCColorComponent {
    pub fn new(kind: LCColorComponentKind, source_slots: Vec<u32>) -> RusticolResult<Self> {
        if source_slots.is_empty() {
            return Err(invalid(
                "an LC color component cannot have an empty source word",
            ));
        }
        if source_slots.len() > u32::MAX as usize {
            return Err(invalid("LC color component exceeds the u32 ABI domain"));
        }
        let mut seen = BTreeSet::new();
        if source_slots.iter().any(|slot| !seen.insert(*slot)) {
            return Err(invalid(
                "an LC color component cannot contain a source slot more than once",
            ));
        }
        let source_slots = if kind == LCColorComponentKind::Trace {
            canonical_trace_rotation(source_slots)
        } else {
            source_slots
        };
        Ok(Self {
            kind,
            source_slots: source_slots.into_boxed_slice(),
        })
    }

    pub const fn kind(&self) -> LCColorComponentKind {
        self.kind
    }

    pub fn source_slots(&self) -> &[u32] {
        &self.source_slots
    }

    fn reversed(&self) -> RusticolResult<Self> {
        Self::new(self.kind, self.source_slots.iter().rev().copied().collect())
    }
}

/// Internable dynamic LC state, independent of physical sector identity.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DynamicLCColorState {
    output_color_shape_id: u32,
    active_component_index: Option<u32>,
    components: Box<[LCColorComponent]>,
}

impl DynamicLCColorState {
    pub fn new(
        output_color_shape_id: u32,
        active_component_index: Option<u32>,
        components: Vec<LCColorComponent>,
    ) -> RusticolResult<Self> {
        if components.len() > u32::MAX as usize {
            return Err(invalid("LC color forest exceeds the u32 ABI domain"));
        }
        let mut seen = BTreeSet::new();
        for component in &components {
            if component
                .source_slots()
                .iter()
                .any(|slot| !seen.insert(*slot))
            {
                return Err(invalid(
                    "a dynamic LC color state cannot duplicate colored source slots",
                ));
            }
        }
        if let Some(index) = active_component_index
            && index as usize >= components.len()
        {
            return Err(invalid(
                "the active LC color component index is outside the color forest",
            ));
        }
        Ok(Self {
            output_color_shape_id,
            active_component_index,
            components: components.into_boxed_slice(),
        })
    }

    pub const fn output_color_shape_id(&self) -> u32 {
        self.output_color_shape_id
    }

    pub fn components(&self) -> &[LCColorComponent] {
        &self.components
    }

    pub const fn active_component_index(&self) -> Option<u32> {
        self.active_component_index
    }

    pub fn active_component(&self) -> Option<&LCColorComponent> {
        self.active_component_index
            .map(|index| &self.components[index as usize])
    }

    fn reversed(&self) -> RusticolResult<Self> {
        let components = self
            .components
            .iter()
            .rev()
            .map(LCColorComponent::reversed)
            .collect::<RusticolResult<Vec<_>>>()?;
        let active_component_index = self
            .active_component_index
            .map(|index| self.components.len() as u32 - 1 - index);
        Self::new(
            self.output_color_shape_id,
            active_component_index,
            components,
        )
    }
}

/// Owns the only mapping from semantic LC color states to recurrence IDs.
#[derive(Clone, Debug, Default)]
pub struct DynamicLCColorStateInterner {
    states: Vec<DynamicLCColorState>,
    ids: HashMap<DynamicLCColorState, DynamicLCColorStateId>,
}

impl DynamicLCColorStateInterner {
    pub fn intern(&mut self, state: DynamicLCColorState) -> RusticolResult<DynamicLCColorStateId> {
        if let Some(id) = self.ids.get(&state) {
            return Ok(*id);
        }
        let raw_id = u32::try_from(self.states.len())
            .map_err(|_| invalid("dynamic LC color-state interner exceeds u32"))?;
        let id = DynamicLCColorStateId::from_interner(raw_id);
        self.states.push(state.clone());
        self.ids.insert(state, id);
        Ok(id)
    }

    pub fn get(&self, id: DynamicLCColorStateId) -> Option<&DynamicLCColorState> {
        self.states.get(id.get() as usize)
    }

    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

/// Exact operation performed by one compiler-certified LC transition term.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
pub enum LCColorComponentOperation {
    ConcatenateJoin = 0,
    ConcatenateKeep = 1,
    InheritLeft = 2,
    InheritRight = 3,
    Empty = 4,
    Close = 5,
}

/// Executable color-state witness emitted by the model compiler.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct LCColorTransitionWitness {
    input_permutation: [u8; 2],
    reverse_parent_mask: u8,
    operation: LCColorComponentOperation,
    result_component_kind: Option<LCColorComponentKind>,
    result_component_role: LCColorComponentRole,
    result_color_shape_id: Option<u32>,
    exact_factor: ExactComplexRational,
    proof_digest: SemanticDigest,
}

impl LCColorTransitionWitness {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_permutation: [u8; 2],
        reverse_parent_mask: u8,
        operation: LCColorComponentOperation,
        result_component_kind: Option<LCColorComponentKind>,
        result_component_role: LCColorComponentRole,
        result_color_shape_id: Option<u32>,
        exact_factor: ExactComplexRational,
        proof_digest: SemanticDigest,
    ) -> RusticolResult<Self> {
        if input_permutation != [0, 1] && input_permutation != [1, 0] {
            return Err(invalid(
                "LC color witness input permutation must be [0, 1] or [1, 0]",
            ));
        }
        if reverse_parent_mask > 0b11 {
            return Err(invalid(
                "LC color witness has an invalid parent-reversal mask",
            ));
        }
        if exact_factor.is_zero() {
            return Err(invalid(
                "an executable LC color witness cannot have zero factor",
            ));
        }
        match operation {
            LCColorComponentOperation::Close => {
                if result_component_role != LCColorComponentRole::None
                    || result_color_shape_id.is_some()
                {
                    return Err(invalid(
                        "an LC closure witness cannot declare an active result color state",
                    ));
                }
            }
            LCColorComponentOperation::ConcatenateJoin => {
                if result_component_kind.is_none()
                    || result_component_role == LCColorComponentRole::None
                    || result_color_shape_id.is_none()
                {
                    return Err(invalid(
                        "an LC join witness requires result component, role, and shape contracts",
                    ));
                }
            }
            LCColorComponentOperation::InheritLeft | LCColorComponentOperation::InheritRight => {
                if result_component_kind.is_some()
                    || result_component_role != LCColorComponentRole::Active
                    || result_color_shape_id.is_none()
                {
                    return Err(invalid(
                        "an LC inheritance witness requires an active result role and shape",
                    ));
                }
            }
            _ => {
                if result_component_kind.is_some()
                    || result_component_role != LCColorComponentRole::None
                    || result_color_shape_id.is_none()
                {
                    return Err(invalid(
                        "a passive LC witness requires only a result shape contract",
                    ));
                }
            }
        }
        Ok(Self {
            input_permutation,
            reverse_parent_mask,
            operation,
            result_component_kind,
            result_component_role,
            result_color_shape_id,
            exact_factor,
            proof_digest,
        })
    }

    pub const fn exact_factor(&self) -> ExactComplexRational {
        self.exact_factor
    }

    pub const fn proof_digest(&self) -> SemanticDigest {
        self.proof_digest
    }

    pub fn apply(
        &self,
        left: &DynamicLCColorState,
        right: &DynamicLCColorState,
    ) -> RusticolResult<Option<DynamicLCColorState>> {
        let mut parents = [left.clone(), right.clone()];
        for (index, parent) in parents.iter_mut().enumerate() {
            if self.reverse_parent_mask & (1 << index) != 0 {
                *parent = parent.reversed()?;
            }
        }
        let [first, second] = self.input_permutation;
        let left = &parents[first as usize];
        let right = &parents[second as usize];
        if self.operation == LCColorComponentOperation::Close {
            self.closed_components(left, right)?;
            return Ok(None);
        }

        let (mut components, active_component_index) = match self.operation {
            LCColorComponentOperation::ConcatenateJoin => {
                let components =
                    concatenate_join(left, right, self.result_component_kind.unwrap())?;
                let active = (self.result_component_role == LCColorComponentRole::Active)
                    .then_some(components.len() as u32 - 1);
                (components, active)
            }
            LCColorComponentOperation::ConcatenateKeep => {
                if left.active_component().is_some() || right.active_component().is_some() {
                    return Err(invalid(
                        "concatenate-keep requires two passive LC color forests",
                    ));
                }
                (canonical_passive_union(left, right), None)
            }
            LCColorComponentOperation::InheritLeft => inherit_active(left, right, "inherit-left")?,
            LCColorComponentOperation::InheritRight => {
                inherit_active(right, left, "inherit-right")?
            }
            LCColorComponentOperation::Empty => {
                if !left.components().is_empty() || !right.components().is_empty() {
                    return Err(invalid("empty LC operation requires two empty forests"));
                }
                (Vec::new(), None)
            }
            LCColorComponentOperation::Close => unreachable!(),
        };
        components.shrink_to_fit();
        Ok(Some(DynamicLCColorState::new(
            self.result_color_shape_id.unwrap(),
            active_component_index,
            components,
        )?))
    }

    pub fn closed_components(
        &self,
        left: &DynamicLCColorState,
        right: &DynamicLCColorState,
    ) -> RusticolResult<Vec<LCColorComponent>> {
        if self.operation != LCColorComponentOperation::Close {
            return Err(invalid(
                "closed-components is available only for an LC closure witness",
            ));
        }
        let mut parents = [left.clone(), right.clone()];
        for (index, parent) in parents.iter_mut().enumerate() {
            if self.reverse_parent_mask & (1 << index) != 0 {
                *parent = parent.reversed()?;
            }
        }
        let [first, second] = self.input_permutation;
        let left = &parents[first as usize];
        let right = &parents[second as usize];
        match (left.active_component(), right.active_component()) {
            (Some(_), Some(_)) => concatenate_join(
                left,
                right,
                self.result_component_kind.ok_or_else(|| {
                    invalid("a colored LC closure requires a certified result component kind")
                })?,
            ),
            (None, None) if self.result_component_kind.is_none() => {
                Ok(canonical_passive_union(left, right))
            }
            (None, None) => Err(invalid(
                "a passive LC closure cannot declare a joined component kind",
            )),
            _ => Err(invalid(
                "an LC closure must consume either two active components or two passive forests",
            )),
        }
    }
}

fn concatenate_join(
    left: &DynamicLCColorState,
    right: &DynamicLCColorState,
    result_kind: LCColorComponentKind,
) -> RusticolResult<Vec<LCColorComponent>> {
    let Some(left_active_index) = left.active_component_index() else {
        return Err(invalid("LC color join requires an active left component"));
    };
    let Some(right_active_index) = right.active_component_index() else {
        return Err(invalid("LC color join requires an active right component"));
    };
    let left_active = &left.components()[left_active_index as usize];
    let right_active = &right.components()[right_active_index as usize];
    let joined = left_active
        .source_slots()
        .iter()
        .chain(right_active.source_slots())
        .copied()
        .collect();
    let mut components = left
        .components()
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != left_active_index as usize)
        .map(|(_, component)| component.clone())
        .chain(
            right
                .components()
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != right_active_index as usize)
                .map(|(_, component)| component.clone()),
        )
        .collect::<Vec<_>>();
    components.sort_unstable();
    components.push(LCColorComponent::new(result_kind, joined)?);
    Ok(components)
}

fn canonical_passive_union(
    left: &DynamicLCColorState,
    right: &DynamicLCColorState,
) -> Vec<LCColorComponent> {
    let mut components = left
        .components()
        .iter()
        .chain(right.components())
        .cloned()
        .collect::<Vec<_>>();
    components.sort_unstable();
    components
}

fn inherit_active(
    active_parent: &DynamicLCColorState,
    passive_parent: &DynamicLCColorState,
    operation: &str,
) -> RusticolResult<(Vec<LCColorComponent>, Option<u32>)> {
    let Some(active_index) = active_parent.active_component_index() else {
        return Err(invalid(format!(
            "{operation} requires an active component in the inherited parent"
        )));
    };
    if passive_parent.active_component().is_some() {
        return Err(invalid(format!(
            "{operation} requires the other parent to have no active component"
        )));
    }
    let active = active_parent.components()[active_index as usize].clone();
    let mut components = active_parent
        .components()
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != active_index as usize)
        .map(|(_, component)| component.clone())
        .chain(passive_parent.components().iter().cloned())
        .collect::<Vec<_>>();
    components.sort_unstable();
    components.push(active);
    let active_index = components.len() as u32 - 1;
    Ok((components, Some(active_index)))
}

fn canonical_trace_rotation(source_slots: Vec<u32>) -> Vec<u32> {
    let Some((best, _)) = (0..source_slots.len())
        .map(|offset| {
            let rotated = source_slots[offset..]
                .iter()
                .chain(&source_slots[..offset])
                .copied()
                .collect::<Vec<_>>();
            (rotated, offset)
        })
        .min()
    else {
        return source_slots;
    };
    best
}
