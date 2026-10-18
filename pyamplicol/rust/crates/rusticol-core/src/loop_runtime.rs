// SPDX-License-Identifier: 0BSD

//! Graph-free, immutable loop execution data in the existing PACBIN envelope.
//! This bounded internal loader is not public artifact dispatch or an SDK runner.

use crate::pacbin::{PacbinMemberKind, PacbinReader};
use crate::{RusticolError, RusticolResult};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const LOOP_RUNTIME_MEMBER: &str = "schedule/loop-runtime-v1.json";
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
record!(Family {
    index: usize, graph_indices: Vec<usize>, sector_indices: Vec<usize>,
    powers: Vec<[u32; 6]>, propagators: usize, mass_columns: Vec<usize>,
    exact_state: String, mass_state: String, projector_state: String,
    numerators: [String; 3], rational_numerators: [String; 3]
});
record!(Counterterm { schedule: usize, coefficient_state: String, input_columns: Vec<usize> });
record!(LoopRuntimePlan {
    schema: u32, r2_strategy: String, dimensional_numerators: bool,
    state_legs: Vec<StateLeg>, helicities: Vec<Vec<i32>>, initial_average: u64,
    model_inputs: Vec<String>, model_count: usize, bound_positions: BTreeMap<String, usize>,
    order_layout: OrderLayout, schedules: Vec<Schedule>, born: ScheduleGroup, loops: ScheduleGroup,
    families: Vec<Family>, bare_evaluators: BTreeMap<String, String>,
    currents: Vec<Counterterm>, fields: Vec<String>, model_evaluators: BTreeMap<String, String>
});

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
    let reader = PacbinReader::open(path)?;
    require(
        reader.member(LOOP_RUNTIME_MEMBER)?.kind() == PacbinMemberKind::LoopRuntimePlan,
        "wrong plan member kind",
    )?;
    let bytes = reader.member_bytes(LOOP_RUNTIME_MEMBER)?;
    let plan: LoopRuntimePlan =
        serde_json::from_slice(bytes).map_err(|e| invalid(e.to_string()))?;
    require(plan.schema == 1, "unsupported schema")?;
    require(
        plan.r2_strategy == "model" && !plan.dimensional_numerators,
        "only ordinary full-colour model-R2 runtime plans are supported",
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
        let legs = self.state_legs.len();
        require(
            legs > 1 && !self.helicities.is_empty() && self.initial_average > 0,
            "empty state layout or normalization",
        )?;
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
        let lookup = |map: &BTreeMap<String, String>, name: &str| -> RusticolResult<String> {
            map.get(name)
                .cloned()
                .ok_or_else(|| invalid(format!("missing evaluator {name}")))
        };
        require(
            self.bare_evaluators.len() == 5 && self.model_evaluators.len() == 3,
            "unexpected assembly evaluator keys",
        )?;
        shape(
            &lookup(&self.bare_evaluators, "model_evaluator")?,
            self.model_inputs.len(),
            self.model_count,
        )?;
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
        for family in &self.families {
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
                    p[4] == 0
                        && p[5] == 0
                        && p[..4].iter().map(|&x| u64::from(x)).sum::<u64>() <= n as u64 + 1
                }),
                "unsupported numerator rank/sector",
            )?;
            require(
                family.rational_numerators.iter().all(|p| p == "0"),
                "ordinary plan contains dimensional remainder",
            )?;
            shape(
                &family.exact_state,
                product(4, legs)?,
                sum([product(4, n)?, product(n, n)?].into_iter())?,
            )?;
            shape(&family.mass_state, family.mass_columns.len(), n)?;
            let loops = sum(family.graph_indices.iter().map(|&i| loop_widths[i]))?;
            shape(
                &family.projector_state,
                product(h, sum([born_width, loops].into_iter())?)?,
                product(family.sector_indices.len(), family.powers.len())?,
            )?;
            integral_width =
                sum([integral_width, product(3, family.sector_indices.len())?].into_iter())?;
        }
        shape(
            &lookup(&self.bare_evaluators, "postprocess")?,
            integral_width,
            product(3, v)?,
        )?;
        let mut amplitude_width = born_width;
        for current in &self.currents {
            require(
                current.schedule < self.schedules.len(),
                "CT schedule out of range",
            )?;
            let &(inputs, outputs) = sizes
                .get(&current.coefficient_state)
                .ok_or_else(|| invalid("missing CT scalar evaluator"))?;
            require(
                outputs % 2 == 0,
                "CT scalar value/diagnostic pairing differs",
            )?;
            shape(
                &current.coefficient_state,
                self.model_inputs.len() + 1,
                outputs,
            )?;
            require(
                inputs == self.model_inputs.len() + 1
                    && current.input_columns.len()
                        == self.schedules[current.schedule].parameter_count
                    && indices(&current.input_columns, bound_width + outputs / 2),
                "CT binding columns differ",
            )?;
            amplitude_width =
                sum([amplitude_width, schedule_widths[current.schedule]].into_iter())?;
        }
        for field in &self.fields {
            shape(field, self.model_inputs.len() + 1, 2)?;
            amplitude_width += 1;
        }
        shape(
            &lookup(&self.model_evaluators, "projector")?,
            product(h, amplitude_width)?,
            product(9, v)?,
        )?;
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
        require(used.len() == sizes.len(), "unreferenced evaluator payload")?;
        Ok(())
    }
}
