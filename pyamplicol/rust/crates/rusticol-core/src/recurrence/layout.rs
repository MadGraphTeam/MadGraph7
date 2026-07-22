// SPDX-License-Identifier: 0BSD

use std::fmt;

use crate::{RusticolError, RusticolResult};

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

/// A nonzero SHA-256 semantic digest.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct SemanticDigest([u8; 32]);

impl SemanticDigest {
    pub fn new(bytes: [u8; 32]) -> RusticolResult<Self> {
        if bytes == [0; 32] {
            return Err(invalid("semantic digest must not be all zero"));
        }
        Ok(Self(bytes))
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl fmt::Display for SemanticDigest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Workload-specific LC recurrence schedule strategy.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u32)]
pub enum RecurrenceStrategy {
    TopologyReplay = 0,
    AllFlowUnion = 1,
}

impl RecurrenceStrategy {
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TopologyReplay => "topology-replay",
            Self::AllFlowUnion => "all-flow-union",
        }
    }
}

impl TryFrom<u32> for RecurrenceStrategy {
    type Error = RusticolError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::TopologyReplay),
            1 => Ok(Self::AllFlowUnion),
            _ => Err(invalid(format!(
                "unknown recurrence strategy discriminant {value}"
            ))),
        }
    }
}

impl fmt::Display for RecurrenceStrategy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Semantic kind of recurrence state.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u32)]
pub enum RecurrenceNodeKind {
    Source = 0,
    Current = 1,
}

impl TryFrom<u32> for RecurrenceNodeKind {
    type Error = RusticolError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Source),
            1 => Ok(Self::Current),
            _ => Err(invalid(format!(
                "unknown recurrence node-kind discriminant {value}"
            ))),
        }
    }
}

/// One canonical term in a momentum linear form.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MomentumTerm {
    pub source_slot: u32,
    pub coefficient: i32,
}

/// Canonical source-slot-ordered momentum linear form.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CanonicalMomentumLinearForm(Box<[MomentumTerm]>);

impl CanonicalMomentumLinearForm {
    pub fn new(terms: Vec<MomentumTerm>) -> RusticolResult<Self> {
        validate_sequence_len("momentum terms", terms.len())?;
        let mut previous = None;
        for (index, term) in terms.iter().enumerate() {
            if term.coefficient == 0 {
                return Err(invalid(format!(
                    "momentum term {index} has zero coefficient"
                )));
            }
            if let Some(previous) = previous
                && previous >= term.source_slot
            {
                return Err(invalid(format!(
                    "momentum terms are not strictly source-slot ordered at row {index}"
                )));
            }
            previous = Some(term.source_slot);
        }
        Ok(Self(terms.into_boxed_slice()))
    }

    pub fn terms(&self) -> &[MomentumTerm] {
        &self.0
    }
}

/// Exact semantic identity before contribution-vector comparison.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct CurrentCoreKey {
    catalog_digest: SemanticDigest,
    node_kind: RecurrenceNodeKind,
    current_state_template_id: u32,
    support_source_slots: Box<[u32]>,
    momentum: CanonicalMomentumLinearForm,
    spin_state: i32,
    flavour_flow: Box<[i32]>,
    quantum_number_flow: Box<[i32]>,
    coupling_orders: Box<[u32]>,
    source_template_id: Option<u32>,
    propagator_template_id: Option<u32>,
}

impl CurrentCoreKey {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        catalog_digest: SemanticDigest,
        node_kind: RecurrenceNodeKind,
        current_state_template_id: u32,
        support_source_slots: Vec<u32>,
        momentum: CanonicalMomentumLinearForm,
        spin_state: i32,
        flavour_flow: Vec<i32>,
        quantum_number_flow: Vec<i32>,
        coupling_orders: Vec<u32>,
        source_template_id: Option<u32>,
        propagator_template_id: Option<u32>,
    ) -> RusticolResult<Self> {
        validate_sequence_len("support source slots", support_source_slots.len())?;
        validate_sequence_len("flavour flow", flavour_flow.len())?;
        validate_sequence_len("quantum-number flow", quantum_number_flow.len())?;
        validate_sequence_len("coupling orders", coupling_orders.len())?;
        validate_strict_u32_sequence("support source slots", &support_source_slots)?;
        if node_kind == RecurrenceNodeKind::Source && source_template_id.is_none() {
            return Err(invalid(
                "source recurrence key requires a source template id",
            ));
        }
        if node_kind == RecurrenceNodeKind::Current && source_template_id.is_some() {
            return Err(invalid(
                "non-source recurrence key must not carry a source template id",
            ));
        }
        Ok(Self {
            catalog_digest,
            node_kind,
            current_state_template_id,
            support_source_slots: support_source_slots.into_boxed_slice(),
            momentum,
            spin_state,
            flavour_flow: flavour_flow.into_boxed_slice(),
            quantum_number_flow: quantum_number_flow.into_boxed_slice(),
            coupling_orders: coupling_orders.into_boxed_slice(),
            source_template_id,
            propagator_template_id,
        })
    }

    pub const fn catalog_digest(&self) -> SemanticDigest {
        self.catalog_digest
    }

    pub const fn node_kind(&self) -> RecurrenceNodeKind {
        self.node_kind
    }

    pub const fn current_state_template_id(&self) -> u32 {
        self.current_state_template_id
    }

    pub fn support_source_slots(&self) -> &[u32] {
        &self.support_source_slots
    }

    pub const fn momentum(&self) -> &CanonicalMomentumLinearForm {
        &self.momentum
    }

    pub const fn spin_state(&self) -> i32 {
        self.spin_state
    }

    pub fn flavour_flow(&self) -> &[i32] {
        &self.flavour_flow
    }

    pub fn quantum_number_flow(&self) -> &[i32] {
        &self.quantum_number_flow
    }

    pub fn coupling_orders(&self) -> &[u32] {
        &self.coupling_orders
    }

    pub const fn source_template_id(&self) -> Option<u32> {
        self.source_template_id
    }

    pub const fn propagator_template_id(&self) -> Option<u32> {
        self.propagator_template_id
    }
}

/// Exact identity of one contribution before coefficient aggregation.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ContributionKey {
    transition_template_id: u32,
    parent_value_class_ids: Box<[u32]>,
    parent_state_template_ids: Box<[u32]>,
    parent_momenta: Box<[CanonicalMomentumLinearForm]>,
    result_state_template_id: u32,
    quantum_flow_witness_id: u32,
    color_flow_rule_id: u32,
    runtime_coupling_binding_digest: SemanticDigest,
    output_projection_id: u32,
}

impl ContributionKey {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        transition_template_id: u32,
        parent_value_class_ids: Vec<u32>,
        parent_state_template_ids: Vec<u32>,
        parent_momenta: Vec<CanonicalMomentumLinearForm>,
        result_state_template_id: u32,
        quantum_flow_witness_id: u32,
        color_flow_rule_id: u32,
        runtime_coupling_binding_digest: SemanticDigest,
        output_projection_id: u32,
    ) -> RusticolResult<Self> {
        let parent_count = parent_value_class_ids.len();
        if parent_count == 0 {
            return Err(invalid(
                "recurrence contribution requires at least one parent",
            ));
        }
        if parent_state_template_ids.len() != parent_count || parent_momenta.len() != parent_count {
            return Err(invalid(format!(
                "recurrence contribution parent columns have lengths {parent_count}, {}, and {}",
                parent_state_template_ids.len(),
                parent_momenta.len()
            )));
        }
        u32::try_from(parent_count)
            .map_err(|_| invalid("recurrence contribution parent count exceeds u32"))?;
        Ok(Self {
            transition_template_id,
            parent_value_class_ids: parent_value_class_ids.into_boxed_slice(),
            parent_state_template_ids: parent_state_template_ids.into_boxed_slice(),
            parent_momenta: parent_momenta.into_boxed_slice(),
            result_state_template_id,
            quantum_flow_witness_id,
            color_flow_rule_id,
            runtime_coupling_binding_digest,
            output_projection_id,
        })
    }

    pub const fn transition_template_id(&self) -> u32 {
        self.transition_template_id
    }

    pub fn parent_value_class_ids(&self) -> &[u32] {
        &self.parent_value_class_ids
    }

    pub fn parent_state_template_ids(&self) -> &[u32] {
        &self.parent_state_template_ids
    }

    pub fn parent_momenta(&self) -> &[CanonicalMomentumLinearForm] {
        &self.parent_momenta
    }

    pub const fn result_state_template_id(&self) -> u32 {
        self.result_state_template_id
    }

    pub const fn quantum_flow_witness_id(&self) -> u32 {
        self.quantum_flow_witness_id
    }

    pub const fn color_flow_rule_id(&self) -> u32 {
        self.color_flow_rule_id
    }

    pub const fn runtime_coupling_binding_digest(&self) -> SemanticDigest {
        self.runtime_coupling_binding_digest
    }

    pub const fn output_projection_id(&self) -> u32 {
        self.output_projection_id
    }
}

fn validate_strict_u32_sequence(label: &str, values: &[u32]) -> RusticolResult<()> {
    let mut previous = None;
    for (index, value) in values.iter().copied().enumerate() {
        if let Some(previous) = previous
            && previous >= value
        {
            return Err(invalid(format!(
                "{label} are not strictly ordered at row {index}"
            )));
        }
        previous = Some(value);
    }
    Ok(())
}

fn validate_sequence_len(label: &str, length: usize) -> RusticolResult<u32> {
    u32::try_from(length).map_err(|_| invalid(format!("{label} length {length} exceeds u32")))
}
