// SPDX-License-Identifier: 0BSD

//! Immutable shared native loop execution data in the existing PACBIN envelope.
//! Public runtimes and SDKs use the same descriptor and evaluator boundary.

use crate::pacbin::{PacbinMemberKind, PacbinReader};
use crate::{RusticolError, RusticolResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const LOOP_RUNTIME_MEMBER: &str = "schedule/loop-runtime-v3.json";
type Order = Vec<(String, u32)>;

macro_rules! record {
    ($name:ident { $($field:ident : $ty:ty),* $(,)? }) => {
        #[derive(Clone, Debug, Deserialize, Serialize)]
        #[serde(deny_unknown_fields)]
        pub struct $name { $(pub $field: $ty),* }
    };
}
record!(StateLeg {
    index: usize,
    kind: String,
    incoming: bool,
    massless: bool,
    offset: usize
});
record!(BornSector { order: Order, pairs: Vec<(Order, Order)> });
record!(VirtualPair {
    born: Order,
    r#loop: Order
});
record!(VirtualSector { order: Order, pairs: Vec<VirtualPair> });
record!(OrderLayout { born: Vec<BornSector>, r#virtual: Vec<VirtualSector>, correction_names: Vec<String> });
record!(Schedule { states: Vec<String>, input_slots: Vec<Vec<usize>>, parameter_count: usize, output_slots: Vec<usize> });
record!(ScheduleGroup { schedules: Vec<usize>, input_columns: Vec<Vec<usize>>, representatives: Vec<usize>, representative_for: Vec<usize> });
record!(FamilyKinematics {
    context: String, mass_output_count: usize, on_shell: Vec<[usize; 2]>
});
record!(Family {
    index: usize, graph_indices: Vec<usize>, sector_indices: Vec<usize>,
    powers: Vec<[u32; 6]>, propagators: usize, mass_columns: Vec<usize>,
    exact_state: String, mass_state: String, projector_state: String,
    numerators: [String; 3], rational_numerators: [String; 3],
    kinematics: Option<FamilyKinematics>
});
record!(ScalarOwner {
    external_names: Vec<String>, external_types: Vec<String>,
    outputs: Vec<(String, i32)>, fixed_parameters: Vec<(String, [String; 2])>,
    values_state: String, defaults_state: Option<String>, guards_state: Option<String>
});
record!(ScalarScaleView {
    outputs: Vec<(String, i32)>, scale_parameter: String, scale_index: usize,
    scale_state: String, values_state: String
});
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelBinding {
    Legacy,
    Source {
        owner: ScalarOwner,
        ordinary_scale: Option<ScalarScaleView>,
        counterterm_scale: ScalarScaleView,
        model_output_indices: Vec<usize>,
        initial_overrides: BTreeMap<String, [String; 2]>,
    },
}
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ScalarBinding {
    Evaluator { state: String },
    SourceColumns { columns: Vec<usize> },
}
impl ScalarBinding {
    pub fn legacy_state(&self) -> Option<&str> {
        match self {
            Self::Evaluator { state } => Some(state),
            Self::SourceColumns { .. } => None,
        }
    }
}
record!(Counterterm { kind: String, schedule: usize, scalar: ScalarBinding, input_columns: Vec<usize> });
record!(DualBornWeights {
    kind: String, state: String, columns: Vec<[usize; 3]>,
    term_graphs: Vec<usize>, family_columns: Vec<Vec<usize>>
});
record!(HelicityProjection {
    born: String, families: Vec<String>, counterterms: String,
    dual_born_weights: Option<DualBornWeights>
});
// The existing ColorCorrelator JSON is retained as column metadata. Its
// operations are compiled into the dependency evaluator states at generation.
record!(DirectedHistory {
    id: String, bra: Vec<serde_json::Value>, ket: Vec<serde_json::Value>
});
record!(DirectedProjection {
    kind: String, convention: String, color_legs: Vec<(usize, i32)>,
    histories: Vec<DirectedHistory>, born: String, families: Vec<String>,
    postprocess: String, counterterms: String, assemble: String, sum_order_values: String,
    dual_born_weights: Option<DualBornWeights>
});
record!(SpinProjection {
    legs: Vec<usize>, helicities: Vec<Vec<i32>>, phases: Vec<String>,
    source_contracts: Vec<serde_json::Value>, directed: DirectedProjection
});
// Projector instructions already contain this fixed-Nf projection evaluated at Nc=3.
// Degrees are aligned with the corresponding squared coupling order catalogue.
record!(ColorProjection {
    accuracy: String, born_degrees: Vec<Option<i64>>, virtual_degrees: Vec<Option<i64>>
});
record!(BornColorComponents {
    kind: String, ids: Vec<String>, members: Vec<Vec<usize>>
});
record!(AdmittedFlavorMember {
    process_id: String, expression: String, external_pdgs: Vec<i32>, species: Vec<(i32, i32)>
});
record!(LoopRuntimePlan {
    schema: u32, r2_strategy: String, dimensional_numerators: bool,
    external_pdgs: Vec<i32>, state_legs: Vec<StateLeg>, helicities: Vec<Vec<i32>>, initial_average: u64,
    model_inputs: Vec<String>, model_count: usize, model_binding: ModelBinding,
    bound_positions: BTreeMap<String, usize>,
    order_layout: OrderLayout, schedules: Vec<Schedule>, born: ScheduleGroup, loops: ScheduleGroup,
    families: Vec<Family>, bare_evaluators: BTreeMap<String, String>,
    currents: Vec<Counterterm>, fields: Vec<ScalarBinding>, model_evaluators: BTreeMap<String, String>,
    helicity_projection: Option<HelicityProjection>, color_projection: Option<ColorProjection>,
    born_color_components: Option<BornColorComponents>,
    directed_projection: Option<DirectedProjection>, spin_projections: Option<Vec<SpinProjection>>,
    admitted_flavor_members: Option<Vec<AdmittedFlavorMember>>,
    dual_born_weights: Option<DualBornWeights>
});

impl DualBornWeights {
    fn validate(
        &self,
        directions: usize,
        born_width: usize,
        loop_widths: &[usize],
        sectors: usize,
        families: &[Family],
        shape: &mut impl FnMut(&str, usize, usize) -> RusticolResult<()>,
    ) -> RusticolResult<()> {
        require(
            self.kind == "exact-dual-born-v1",
            "unsupported dual Born weight schedule",
        )?;
        require(
            !self.columns.is_empty()
                && unique(&self.columns)
                && !self.term_graphs.is_empty()
                && indices(&self.term_graphs, loop_widths.len())
                && self.family_columns.len() == families.len(),
            "invalid dual Born weight column/family layout",
        )?;
        for (graph, &width) in loop_widths.iter().enumerate() {
            require(
                self.term_graphs.iter().filter(|&&g| g == graph).count() <= width,
                "dual Born term count exceeds graph outputs",
            )?;
        }
        require(
            self.columns.iter().all(|&[direction, sector, term]| {
                direction < directions && sector < sectors && term < self.term_graphs.len()
            }),
            "dual Born weight key out of range",
        )?;
        shape(&self.state, born_width, self.columns.len())?;
        for (family, selected) in families.iter().zip(&self.family_columns) {
            let expected = self
                .columns
                .iter()
                .enumerate()
                .filter_map(|(column, &[_, sector, term])| {
                    (family.sector_indices.contains(&sector)
                        && family.graph_indices.contains(&self.term_graphs[term]))
                    .then_some(column)
                })
                .collect::<Vec<_>>();
            require(
                *selected == expected,
                "dual Born family selection differs from exact keys",
            )?;
        }
        Ok(())
    }
}

impl LoopRuntimePlan {
    pub fn directed_projections(&self) -> impl Iterator<Item = (&DirectedProjection, usize)> {
        self.directed_projection
            .iter()
            .map(|p| (p, self.helicities.len()))
            .chain(
                self.spin_projections
                    .iter()
                    .flatten()
                    .map(|p| (&p.directed, p.helicities.len())),
            )
    }

    /// Signed Born anchors are emitted only by the resolved LC projection.
    pub fn resolved_color_component_count(&self) -> usize {
        self.born_color_components
            .as_ref()
            .map_or(1, |p| p.ids.len())
    }

    pub fn resolved_rows_per_point(&self) -> RusticolResult<usize> {
        product(self.helicities.len(), self.resolved_color_component_count())
    }
}

/// Keeps authenticated immutable payload ownership independent of any evaluator workspace.
pub struct LoadedLoopRuntime {
    plan: LoopRuntimePlan,
    reader: PacbinReader,
}
impl LoadedLoopRuntime {
    pub fn plan(&self) -> &LoopRuntimePlan {
        &self.plan
    }
    pub fn metadata_json(&self) -> RusticolResult<String> {
        serde_json::to_string(&self.plan).map_err(|e| invalid(e.to_string()))
    }
    pub fn evaluator_state(&self, name: &str) -> RusticolResult<&[u8]> {
        require(
            self.reader.member(name)?.kind() == PacbinMemberKind::SymbolicaExactState,
            "member is not an exact evaluator",
        )?;
        self.reader.member_bytes(name)
    }
    pub fn evaluator_states(&self) -> impl Iterator<Item = (&str, &[u8])> {
        self.reader
            .members()
            .iter()
            .filter(|m| m.kind() == PacbinMemberKind::SymbolicaExactState)
            .map(|m| {
                (
                    m.logical_path(),
                    self.reader
                        .member_bytes(m.logical_path())
                        .expect("validated member"),
                )
            })
    }
}
fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::compatibility(format!("loop runtime plan: {}", message.into()))
}
fn require(condition: bool, message: &str) -> RusticolResult<()> {
    if condition {
        Ok(())
    } else {
        Err(invalid(message))
    }
}
fn sum(mut values: impl Iterator<Item = usize>) -> RusticolResult<usize> {
    values.try_fold(0usize, |a, b| {
        a.checked_add(b)
            .ok_or_else(|| invalid("dimension overflow"))
    })
}
fn product(a: usize, b: usize) -> RusticolResult<usize> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("dimension overflow"))
}
fn indices(values: &[usize], width: usize) -> bool {
    values.iter().all(|&x| x < width)
}
fn unique<T: Ord + Clone>(values: &[T]) -> bool {
    values.iter().cloned().collect::<BTreeSet<_>>().len() == values.len()
}
fn order(value: &Order) -> bool {
    value.iter().all(|(n, p)| !n.is_empty() && *p > 0) && value.windows(2).all(|w| w[0].0 < w[1].0)
}

fn scalar_outputs(outputs: &[(String, i32)]) -> bool {
    !outputs.is_empty() && unique(outputs) && outputs.iter().all(|(name, _)| !name.is_empty())
}

fn decimal_pair(value: &[String; 2], real: bool) -> RusticolResult<()> {
    use crate::source_scalar::decimal_rational;
    decimal_rational(&value[0]).map_err(invalid)?;
    let imaginary = decimal_rational(&value[1]).map_err(invalid)?;
    require(
        !real || imaginary == symbolica::domains::rational::Rational::from(0),
        "real source input has an imaginary component",
    )
}

impl ScalarOwner {
    fn validate(
        &self,
        shape: &mut impl FnMut(&str, usize, usize) -> RusticolResult<()>,
    ) -> RusticolResult<()> {
        let inputs = self.external_names.len();
        require(
            unique(&self.external_names)
                && self.external_names.iter().all(|name| !name.is_empty())
                && self.external_types.len() == inputs
                && self
                    .external_types
                    .iter()
                    .all(|kind| matches!(kind.as_str(), "real" | "complex"))
                && scalar_outputs(&self.outputs),
            "invalid source scalar names/types/outputs",
        )?;
        let fixed_names = self
            .fixed_parameters
            .iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        require(
            unique(&fixed_names)
                && fixed_names
                    .iter()
                    .all(|name| !name.is_empty() && !self.external_names.contains(name)),
            "source fixed/adjustable parameter names overlap or repeat",
        )?;
        for (_, value) in &self.fixed_parameters {
            decimal_pair(value, false)?;
        }
        shape(&self.values_state, inputs, product(2, self.outputs.len())?)?;
        match (&self.defaults_state, inputs) {
            (None, 0) => (),
            (Some(state), count) if count > 0 => shape(state, 0, product(2, count)?)?,
            _ => return Err(invalid("source default program/input shape differs")),
        }
        let real = self
            .external_types
            .iter()
            .filter(|kind| *kind == "real")
            .count();
        match (&self.guards_state, real) {
            (None, 0) => (),
            (Some(state), count) if count > 0 => shape(state, inputs, product(2, count)?)?,
            _ => return Err(invalid("source type-guard program/input shape differs")),
        }
        Ok(())
    }
}

impl ScalarScaleView {
    fn validate(
        &self,
        owner: &ScalarOwner,
        shape: &mut impl FnMut(&str, usize, usize) -> RusticolResult<()>,
    ) -> RusticolResult<()> {
        require(
            self.scale_index < owner.external_names.len()
                && owner.external_names[self.scale_index] == self.scale_parameter
                && owner.external_types[self.scale_index] == "real"
                && scalar_outputs(&self.outputs),
            "invalid source scale selection",
        )?;
        shape(&self.scale_state, 1, 4)?;
        shape(
            &self.values_state,
            owner.external_names.len(),
            product(2, self.outputs.len())?,
        )
    }
}

fn pair_order(left: &Order, right: &Order, squared: &Order) -> bool {
    let mut powers = BTreeMap::<&str, u32>::new();
    for (name, power) in left.iter().chain(right) {
        let entry = powers.entry(name.as_str()).or_default();
        let Some(total) = entry.checked_add(*power) else {
            return false;
        };
        *entry = total;
    }
    powers
        .into_iter()
        .eq(squared.iter().map(|(name, power)| (name.as_str(), *power)))
}

pub fn load_loop_runtime_pacbin(path: impl AsRef<Path>) -> RusticolResult<LoadedLoopRuntime> {
    load_loop_runtime_reader(PacbinReader::open(path)?)
}

/// Consume a reader whose bytes have already been authenticated by its owner.
/// Artifact callers retain their checked file handle through reader creation.
pub fn load_loop_runtime_reader(reader: PacbinReader) -> RusticolResult<LoadedLoopRuntime> {
    require(
        reader.member(LOOP_RUNTIME_MEMBER)?.kind() == PacbinMemberKind::LoopRuntimePlan,
        "wrong plan member kind",
    )?;
    let bytes = reader.member_bytes(LOOP_RUNTIME_MEMBER)?;
    let plan: LoopRuntimePlan =
        serde_json::from_slice(bytes).map_err(|e| invalid(e.to_string()))?;
    require(plan.schema == 3, "unsupported schema")?;
    require(
        (plan.r2_strategy == "model" && !plan.dimensional_numerators)
            || (plan.r2_strategy == "direct" && plan.dimensional_numerators),
        "inconsistent R2 strategy/dimensional numerator selection",
    )?;
    let mut sizes = BTreeMap::new();
    for member in reader.members() {
        if member.logical_path() == LOOP_RUNTIME_MEMBER {
            continue;
        }
        require(
            member.kind() == PacbinMemberKind::SymbolicaExactState,
            "unexpected payload member kind",
        )?;
        let evaluator = crate::engine::decode_exact_evaluator_state(
            reader.member_bytes(member.logical_path())?,
        )?;
        sizes.insert(
            member.logical_path().to_owned(),
            (evaluator.get_input_len(), evaluator.get_output_len()),
        );
    }
    plan.validate(&sizes)?;
    Ok(LoadedLoopRuntime { plan, reader })
}

impl LoopRuntimePlan {
    fn validate(&self, sizes: &BTreeMap<String, (usize, usize)>) -> RusticolResult<()> {
        let mut used = BTreeSet::new();
        let mut shape = |name: &str, input: usize, output: usize| -> RusticolResult<()> {
            require(
                sizes.get(name) == Some(&(input, output)),
                &format!("evaluator {name} shape/reference differs"),
            )?;
            used.insert(name.to_owned());
            Ok(())
        };
        if let Some(projection) = &self.color_projection {
            require(
                self.r2_strategy == "model"
                    && matches!(projection.accuracy.as_str(), "full" | "lc" | "nlc")
                    && projection.born_degrees.len() == self.order_layout.born.len()
                    && projection.virtual_degrees.len() == self.order_layout.r#virtual.len(),
                "colour projection accuracy/order layout differs or strategy is unsupported",
            )?;
        }
        let legs = self.state_legs.len();
        require(
            legs > 1 && !self.helicities.is_empty() && self.initial_average > 0,
            "empty state layout or normalization",
        )?;
        require(
            self.external_pdgs.len() == legs && self.external_pdgs.iter().all(|&pdg| pdg != 0),
            "external particle layout differs",
        )?;
        if let Some(members) = &self.admitted_flavor_members {
            require(
                self.r2_strategy == "model"
                    && !self.dimensional_numerators
                    && matches!(self.model_binding, ModelBinding::Source { .. })
                    && self.color_projection.is_none(),
                "flavour transport requires source-owned full colour and model R2",
            )?;
            let mut ids = BTreeSet::new();
            for member in members {
                let map: BTreeMap<_, _> = member.species.iter().copied().collect();
                let targets: BTreeSet<_> = map.values().copied().collect();
                // Identity-port source transport keeps every selected vector
                // species fixed. Its typed SourceIR is checked below as usual.
                let source_ports_fixed = self.spin_projections.iter().flatten().all(|class| {
                    class.legs.iter().all(|leg| {
                        leg.checked_sub(1)
                            .and_then(|index| self.external_pdgs.get(index))
                            .is_some_and(|pdg| map.get(pdg).unwrap_or(pdg) == pdg)
                    })
                });
                require(
                    source_ports_fixed
                        && !member.process_id.trim().is_empty()
                        && !member.expression.trim().is_empty()
                        && ids.insert(&member.process_id)
                        && member.external_pdgs.len() == legs
                        && map.len() == member.species.len()
                        && targets == map.keys().copied().collect()
                        && map.iter().all(|(&a, &b)| {
                            a != 0
                                && a != b
                                && a.signum() == b.signum()
                                && a.checked_neg()
                                    .zip(b.checked_neg())
                                    .is_some_and(|(a, b)| map.get(&a) == Some(&b))
                        })
                        && self
                            .external_pdgs
                            .iter()
                            .zip(&member.external_pdgs)
                            .all(|(a, b)| map.get(a).unwrap_or(a) == b),
                    "invalid admitted identity-species flavour member",
                )?;
            }
        }
        if self.r2_strategy == "direct" {
            require(
                self.external_pdgs == [21, 21, 21, 21]
                    && self.order_layout.born.len() == 1
                    && self.order_layout.r#virtual.len() == 1
                    && self.order_layout.born[0].order == [("QCD".into(), 4)]
                    && self.order_layout.born[0].pairs
                        == [(vec![("QCD".into(), 2)], vec![("QCD".into(), 2)])]
                    && self.order_layout.r#virtual[0].order == [("QCD".into(), 6)]
                    && self.order_layout.r#virtual[0].pairs.len() == 1
                    && self.order_layout.r#virtual[0].pairs[0].born == [("QCD".into(), 2)]
                    && self.order_layout.r#virtual[0].pairs[0].r#loop == [("QCD".into(), 4)]
                    && self.order_layout.correction_names == ["QCD"]
                    && self.state_legs.iter().enumerate().all(|(i, leg)| {
                        leg.massless
                            && leg.incoming == (i < 2)
                            && leg.kind == if i < 2 { "epsilon" } else { "epsilon_bar" }
                    }),
                "direct R2 requires the certified QCD four-gluon layout",
            )?;
        }
        let mut state_width = 0;
        for (index, leg) in self.state_legs.iter().enumerate() {
            require(
                leg.index == index && leg.offset == state_width,
                "state leg order/offset differs",
            )?;
            require(
                matches!(
                    leg.kind.as_str(),
                    "scalar" | "u" | "v" | "u_bar" | "v_bar" | "epsilon" | "epsilon_bar"
                ),
                "unsupported state kind",
            )?;
            state_width += if leg.kind == "scalar" { 1 } else { 4 };
        }
        for helicities in &self.helicities {
            require(helicities.len() == legs, "helicity leg count differs")?;
            for (leg, &h) in self.state_legs.iter().zip(helicities) {
                let valid = if leg.kind == "scalar" {
                    h == 0
                } else if !leg.massless && matches!(leg.kind.as_str(), "epsilon" | "epsilon_bar") {
                    (-1..=1).contains(&h)
                } else {
                    h == -1 || h == 1
                };
                require(valid, "invalid physical helicity")?;
            }
        }
        require(unique(&self.helicities), "duplicate helicity lane")?;
        require(
            unique(&self.model_inputs) && self.model_inputs.iter().all(|x| !x.is_empty()),
            "invalid model input names",
        )?;
        let bound_width = sum([product(4, legs)?, state_width, self.model_count].into_iter())?;
        require(
            self.bound_positions.values().all(|&x| x < bound_width),
            "bound column out of range",
        )?;
        require(
            !self.order_layout.born.is_empty() && !self.order_layout.r#virtual.is_empty(),
            "empty order layout",
        )?;
        require(
            self.order_layout.born.iter().all(|s| {
                order(&s.order)
                    && !s.pairs.is_empty()
                    && unique(&s.pairs)
                    && s.pairs
                        .iter()
                        .all(|(a, b)| order(a) && order(b) && pair_order(a, b, &s.order))
            }) && self.order_layout.r#virtual.iter().all(|s| {
                order(&s.order)
                    && !s.pairs.is_empty()
                    && unique(
                        &s.pairs
                            .iter()
                            .map(|p| (&p.born, &p.r#loop))
                            .collect::<Vec<_>>(),
                    )
                    && s.pairs.iter().all(|p| {
                        order(&p.born)
                            && order(&p.r#loop)
                            && pair_order(&p.born, &p.r#loop, &s.order)
                    })
            }),
            "invalid order layout",
        )?;
        require(
            unique(
                &self
                    .order_layout
                    .born
                    .iter()
                    .map(|s| &s.order)
                    .collect::<Vec<_>>(),
            ) && unique(
                &self
                    .order_layout
                    .r#virtual
                    .iter()
                    .map(|s| &s.order)
                    .collect::<Vec<_>>(),
            ) && !self.order_layout.correction_names.is_empty()
                && self
                    .order_layout
                    .correction_names
                    .iter()
                    .all(|s| !s.is_empty())
                && unique(&self.order_layout.correction_names),
            "duplicate/invalid order labels or correction names",
        )?;
        let mut schedule_widths = Vec::new();
        for schedule in &self.schedules {
            require(
                schedule.states.len() == schedule.input_slots.len(),
                "schedule node/input counts differ",
            )?;
            let mut width = schedule.parameter_count;
            for (name, inputs) in schedule.states.iter().zip(&schedule.input_slots) {
                let &(input_count, output_count) = sizes
                    .get(name)
                    .ok_or_else(|| invalid("missing schedule evaluator"))?;
                require(
                    inputs.len() == input_count && indices(inputs, width),
                    "schedule input shape/forward reference differs",
                )?;
                shape(name, input_count, output_count)?;
                width = sum([width, output_count].into_iter())?;
            }
            require(
                indices(&schedule.output_slots, width),
                "schedule output out of range",
            )?;
            schedule_widths.push(schedule.output_slots.len());
        }
        let group = |group: &ScheduleGroup| -> RusticolResult<Vec<usize>> {
            let n = group.schedules.len();
            require(
                n > 0 && group.input_columns.len() == n && group.representative_for.len() == n,
                "schedule group shape differs",
            )?;
            require(
                unique(&group.representatives) && indices(&group.representatives, n),
                "invalid schedule representatives",
            )?;
            let representatives = group
                .representatives
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            let mut widths = Vec::new();
            for (i, (&schedule, columns)) in
                group.schedules.iter().zip(&group.input_columns).enumerate()
            {
                require(
                    schedule < self.schedules.len(),
                    "schedule index out of range",
                )?;
                require(
                    columns.len() == self.schedules[schedule].parameter_count
                        && indices(columns, bound_width),
                    "schedule binding columns differ",
                )?;
                let r = group.representative_for[i];
                require(
                    representatives.contains(&r) && group.representative_for[r] == r,
                    "invalid schedule representative map",
                )?;
                // A saved sharing proof must retain equal bytecode AND ordered columns.
                require(
                    group.schedules[r] == schedule && group.input_columns[r] == *columns,
                    "unproven schedule sharing",
                )?;
                widths.push(schedule_widths[schedule]);
            }
            Ok(widths)
        };
        let born_width = sum(group(&self.born)?.into_iter())?;
        let loop_widths = group(&self.loops)?;
        let h = self.helicities.len();
        let b = self.order_layout.born.len();
        let v = self.order_layout.r#virtual.len();
        if let Some(weights) = &self.dual_born_weights {
            weights.validate(1, born_width, &loop_widths, v, &self.families, &mut shape)?;
        }
        if let Some(classes) = &self.spin_projections {
            let ordinary = self
                .directed_projection
                .as_ref()
                .ok_or_else(|| invalid("spin classes require ordinary directed kernels"))?;
            require(!classes.is_empty(), "empty saved spin class catalogue")?;
            require(
                unique(&classes.iter().map(|p| &p.legs).collect::<Vec<_>>()),
                "duplicate saved joint spin class",
            )?;
            for class in classes {
                require(
                    !class.legs.is_empty()
                        && class.legs.windows(2).all(|pair| pair[0] < pair[1])
                        && class.legs.iter().all(|&leg| {
                            leg > 0
                                && leg <= legs
                                && self.state_legs[leg - 1].massless
                                && matches!(
                                    self.state_legs[leg - 1].kind.as_str(),
                                    "epsilon" | "epsilon_bar"
                                )
                        }),
                    "saved spin class requires ordered physical massless vector ports",
                )?;
                require(
                    class.source_contracts.len() == class.legs.len(),
                    "spin source contract count differs",
                )?;
                for (&leg, source) in class.legs.iter().zip(&class.source_contracts) {
                    crate::engine::validate_loop_vector_source(
                        leg,
                        source,
                        self.external_pdgs[leg - 1],
                    )?;
                    require(
                        self.state_legs[leg - 1].kind
                            == if self.state_legs[leg - 1].incoming {
                                "epsilon"
                            } else {
                                "epsilon_bar"
                            },
                        "spin source orientation differs from physical state port",
                    )?;
                }
                let mut seen = BTreeSet::new();
                let spectators = self
                    .helicities
                    .iter()
                    .filter(|row| {
                        seen.insert(
                            row.iter()
                                .enumerate()
                                .filter_map(|(leg, &value)| {
                                    (!class.legs.contains(&(leg + 1))).then_some(value)
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                require(
                    class.helicities == spectators,
                    "spin projection spectator helicities differ",
                )?;
                require(
                    class.phases.len() == class.legs.len(),
                    "spin phase count differs",
                )?;
                for phase in &class.phases {
                    shape(phase, 4, 4)?;
                }
                require(
                    class.directed.color_legs == ordinary.color_legs
                        && class.directed.histories.len() == ordinary.histories.len()
                        && class
                            .directed
                            .histories
                            .iter()
                            .zip(&ordinary.histories)
                            .all(|(a, b)| a.id == b.id && a.bra == b.bra && a.ket == b.ket),
                    "spin and ordinary directed history columns differ",
                )?;
            }
        }
        for (projection, projection_h) in self.directed_projections() {
            if let Some(weights) = &projection.dual_born_weights {
                weights.validate(
                    product(2, projection.histories.len())?,
                    born_width,
                    &loop_widths,
                    v,
                    &self.families,
                    &mut shape,
                )?;
            }
            require(
                projection.kind == "directed-complex"
                    && projection.convention == "su3-literal-tau-physical-charge-all-outgoing-v1"
                    && self.r2_strategy == "model"
                    && self.helicity_projection.is_none()
                    && self.born_color_components.is_none()
                    && self
                        .color_projection
                        .as_ref()
                        .is_none_or(|p| p.accuracy == "full"),
                "unsupported directed projection kind/convention or spin/resolved/colour/R2 domain",
            )?;
            require(
                projection.color_legs.len() == legs
                    && projection.color_legs.iter().enumerate().all(
                        |(index, &(label, representation))| {
                            label == index && matches!(representation, 1 | 3 | -3 | 8)
                        },
                    ),
                "directed colour ports differ from physical leg ordering",
            )?;
            require(
                !projection.histories.is_empty()
                    && projection.histories.iter().all(|history| {
                        !history.id.trim().is_empty() && history.bra.len() == history.ket.len()
                    })
                    && unique(
                        &projection
                            .histories
                            .iter()
                            .map(|history| &history.id)
                            .collect::<Vec<_>>(),
                    )
                    && projection.families.len() == self.families.len(),
                "directed history/family column layout differs",
            )?;
            shape(
                &projection.born,
                product(projection_h, born_width)?,
                product(projection.histories.len(), b)?,
            )?;
        }
        let lookup = |map: &BTreeMap<String, String>, name: &str| -> RusticolResult<String> {
            map.get(name)
                .cloned()
                .ok_or_else(|| invalid(format!("missing evaluator {name}")))
        };
        require(
            self.model_evaluators.len() == 3,
            "unexpected model evaluator keys",
        )?;
        let source_width = match &self.model_binding {
            ModelBinding::Legacy => {
                require(
                    self.bare_evaluators.len() == 5,
                    "unexpected legacy evaluator keys",
                )?;
                shape(
                    &lookup(&self.bare_evaluators, "model_evaluator")?,
                    self.model_inputs.len(),
                    self.model_count,
                )?;
                None
            }
            ModelBinding::Source {
                owner,
                ordinary_scale,
                counterterm_scale,
                model_output_indices,
                initial_overrides,
            } => {
                require(
                    self.r2_strategy == "model" && !self.dimensional_numerators,
                    "source dimensional/direct execution is not certified",
                )?;
                require(
                    self.bare_evaluators.len() == 4
                        && !self.bare_evaluators.contains_key("model_evaluator"),
                    "source scalar owner cannot mix legacy evaluator bindings",
                )?;
                owner.validate(&mut shape)?;
                require(
                    // Source model_inputs are qualified display labels only.
                    // The owner alone binds names/order; model_output_indices
                    // selects its checked values for the physical schedules.
                    self.model_inputs.len() == owner.external_names.len()
                        && model_output_indices.len() == self.model_count
                        && model_output_indices.iter().all(|&index| {
                            index < owner.outputs.len() && owner.outputs[index].1 == 0
                        }),
                    "source model output/input selection differs",
                )?;
                for (name, value) in initial_overrides {
                    let index = owner
                        .external_names
                        .iter()
                        .position(|external| external == name)
                        .ok_or_else(|| invalid("source override is not an adjustable parameter"))?;
                    decimal_pair(value, owner.external_types[index] == "real")?;
                }
                if let Some(view) = ordinary_scale {
                    view.validate(owner, &mut shape)?;
                    require(
                        view.outputs == owner.outputs
                            && view.values_state == owner.values_state
                            && view.scale_index == counterterm_scale.scale_index,
                        "ordinary source scale view differs from owner outputs",
                    )?;
                }
                counterterm_scale.validate(owner, &mut shape)?;
                Some(counterterm_scale.outputs.len())
            }
        };
        shape(
            &lookup(&self.bare_evaluators, "born_projector")?,
            product(h, born_width)?,
            b,
        )?;
        shape(&lookup(&self.bare_evaluators, "zero_poles")?, 0, 3)?;
        shape(
            &lookup(&self.bare_evaluators, "sum_order_values")?,
            sum([b, product(3, v)?].into_iter())?,
            4,
        )?;
        require(
            !self.families.is_empty()
                && unique(&self.families.iter().map(|f| f.index).collect::<Vec<_>>()),
            "empty/duplicate family indices",
        )?;
        let mut integral_width = 0;
        let components = self.resolved_color_component_count();
        if let Some(partition) = &self.born_color_components {
            require(
                partition.kind == "canonical-born-tensor-partition"
                    && !partition.ids.is_empty()
                    && partition.ids.iter().all(|id| !id.is_empty())
                    && unique(&partition.ids)
                    && partition.members.len() == components
                    && partition.members.iter().all(|row| !row.is_empty())
                    && self.helicity_projection.is_some()
                    && self
                        .color_projection
                        .as_ref()
                        .is_some_and(|p| p.accuracy == "lc"),
                "invalid signed Born colour component declaration",
            )?;
            let mut members: Vec<_> = partition.members.iter().flatten().copied().collect();
            members.sort_unstable();
            require(
                members == (0..born_width).collect::<Vec<_>>(),
                "Born colour components do not partition every Born column exactly once",
            )?;
        }
        if let Some(projection) = &self.helicity_projection {
            if let Some(weights) = &projection.dual_born_weights {
                weights.validate(
                    components,
                    born_width,
                    &loop_widths,
                    v,
                    &self.families,
                    &mut shape,
                )?;
            }
            require(
                projection.families.len() == self.families.len(),
                "helicity family count differs",
            )?;
            shape(&projection.born, born_width, product(components, b)?)?;
        }
        for (family_index, family) in self.families.iter().enumerate() {
            let n = family.propagators;
            require(
                n > 0
                    && !family.powers.is_empty()
                    && !family.graph_indices.is_empty()
                    && indices(&family.graph_indices, loop_widths.len()),
                "invalid family dimensions",
            )?;
            require(
                !family.sector_indices.is_empty()
                    && unique(&family.sector_indices)
                    && indices(&family.sector_indices, v),
                "invalid family sectors",
            )?;
            require(
                indices(&family.mass_columns, self.model_count),
                "family mass column out of range",
            )?;
            require(
                family.powers.iter().all(|p| {
                    (self.dimensional_numerators || (p[4] == 0 && p[5] == 0)) && p[5] <= 2
                }),
                "unsupported numerator epsilon/dimensional sector",
            )?;
            require(
                self.dimensional_numerators || family.rational_numerators.iter().all(|p| p == "0"),
                "ordinary plan contains dimensional remainder",
            )?;
            shape(
                &family.exact_state,
                product(4, legs)?,
                sum([product(4, n)?, product(n, n)?].into_iter())?,
            )?;
            let mass_outputs = if let Some(kinematics) = &family.kinematics {
                require(
                    matches!(kinematics.context.as_str(), "raw" | "physical")
                        && kinematics.mass_output_count >= n,
                    "invalid family kinematic context/mass output count",
                )?;
                require(
                    kinematics.context == "physical" || kinematics.on_shell.is_empty(),
                    "raw kinematics contains physical on-shell assignments",
                )?;
                let mut assignments = BTreeMap::new();
                for &[entry, mass] in &kinematics.on_shell {
                    require(
                        entry < product(n, n)?
                            && entry / n != entry % n
                            && mass < kinematics.mass_output_count
                            && assignments.insert(entry, mass).is_none(),
                        "invalid/duplicate on-shell matrix assignment",
                    )?;
                }
                require(
                    assignments.iter().all(|(&entry, mass)| {
                        assignments.get(&((entry % n) * n + entry / n)) == Some(mass)
                    }),
                    "on-shell matrix assignments are not symmetric",
                )?;
                kinematics.mass_output_count
            } else {
                n
            };
            shape(&family.mass_state, family.mass_columns.len(), mass_outputs)?;
            let loops = sum(family.graph_indices.iter().map(|&i| loop_widths[i]))?;
            shape(
                &family.projector_state,
                product(
                    h,
                    sum([
                        self.dual_born_weights
                            .as_ref()
                            .map_or(born_width, |w| w.family_columns[family_index].len()),
                        loops,
                    ]
                    .into_iter())?,
                )?,
                product(family.sector_indices.len(), family.powers.len())?,
            )?;
            for (projection, projection_h) in self.directed_projections() {
                shape(
                    &projection.families[family_index],
                    product(
                        projection_h,
                        sum([
                            projection
                                .dual_born_weights
                                .as_ref()
                                .map_or(born_width, |w| w.family_columns[family_index].len()),
                            loops,
                        ]
                        .into_iter())?,
                    )?,
                    product(
                        product(2, projection.histories.len())?,
                        product(family.sector_indices.len(), family.powers.len())?,
                    )?,
                )?;
            }
            if let Some(projection) = &self.helicity_projection {
                shape(
                    &projection.families[family_index],
                    sum([
                        projection
                            .dual_born_weights
                            .as_ref()
                            .map_or(born_width, |w| w.family_columns[family_index].len()),
                        loops,
                    ]
                    .into_iter())?,
                    product(
                        components,
                        product(family.sector_indices.len(), family.powers.len())?,
                    )?,
                )?;
            }
            integral_width =
                sum([integral_width, product(3, family.sector_indices.len())?].into_iter())?;
        }
        shape(
            &lookup(&self.bare_evaluators, "postprocess")?,
            integral_width,
            product(3, v)?,
        )?;
        let mut scalar_width = |binding: &ScalarBinding| -> RusticolResult<usize> {
            match binding {
                ScalarBinding::Evaluator { state } => {
                    require(
                        source_width.is_none(),
                        "source scalar columns cannot mix legacy evaluators",
                    )?;
                    let &(_, outputs) = sizes
                        .get(state)
                        .ok_or_else(|| invalid("missing CT scalar evaluator"))?;
                    require(
                        outputs > 0 && outputs % 2 == 0,
                        "CT scalar value/diagnostic pairing differs",
                    )?;
                    shape(
                        state,
                        sum([self.model_inputs.len(), 1].into_iter())?,
                        outputs,
                    )?;
                    Ok(outputs / 2)
                }
                ScalarBinding::SourceColumns { columns } => {
                    let width = source_width
                        .ok_or_else(|| invalid("source columns have no scalar owner"))?;
                    require(
                        !columns.is_empty() && indices(columns, width),
                        "source scalar column out of range",
                    )?;
                    Ok(columns.len())
                }
            }
        };
        let mut amplitude_width = born_width;
        for current in &self.currents {
            require(
                matches!(current.kind.as_str(), "R2" | "UVtree" | "UVloop" | "UVmass")
                    && (self.r2_strategy != "direct" || current.kind != "R2"),
                "unsupported or mixed counterterm R2 kind",
            )?;
            require(
                current.schedule < self.schedules.len(),
                "CT schedule out of range",
            )?;
            let coefficients = scalar_width(&current.scalar)?;
            require(
                current.input_columns.len() == self.schedules[current.schedule].parameter_count
                    && indices(
                        &current.input_columns,
                        sum([bound_width, coefficients].into_iter())?,
                    ),
                "CT binding columns differ",
            )?;
            amplitude_width =
                sum([amplitude_width, schedule_widths[current.schedule]].into_iter())?;
        }
        for field in &self.fields {
            require(
                scalar_width(field)? == 1,
                "external field scalar width differs",
            )?;
            amplitude_width += 1;
        }
        shape(
            &lookup(&self.model_evaluators, "projector")?,
            product(h, amplitude_width)?,
            product(9, v)?,
        )?;
        if let Some(projection) = &self.helicity_projection {
            require(
                self.r2_strategy == "model",
                "helicity projection requires model R2",
            )?;
            shape(
                &projection.counterterms,
                amplitude_width,
                product(components, product(9, v)?)?,
            )?;
        }
        shape(
            &lookup(&self.model_evaluators, "assemble")?,
            product(12, v)?,
            product(6, v)?,
        )?;
        shape(
            &lookup(&self.model_evaluators, "sum_order_values")?,
            sum([b, product(18, v)?].into_iter())?,
            19,
        )?;
        for (projection, projection_h) in self.directed_projections() {
            let histories = projection.histories.len();
            let width = product(histories, product(3, v)?)?;
            shape(
                &projection.postprocess,
                product(product(2, histories)?, integral_width)?,
                product(3, width)?,
            )?;
            shape(
                &projection.counterterms,
                product(projection_h, amplitude_width)?,
                product(6, width)?,
            )?;
            shape(&projection.assemble, product(8, width)?, product(7, width)?)?;
            shape(
                &projection.sum_order_values,
                product(histories, sum([b, product(24, v)?].into_iter())?)?,
                product(histories, 25)?,
            )?;
        }
        require(used.len() == sizes.len(), "unreferenced evaluator payload")?;
        Ok(())
    }
}
