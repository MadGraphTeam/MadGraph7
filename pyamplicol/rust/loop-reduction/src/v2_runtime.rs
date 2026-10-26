// SPDX-License-Identifier: 0BSD

//! Plan v4 processes evaluated without Python (WP3.1 I6, the SDK's v4 path).
//!
//! A plan v4 artifact with native point inputs (`point_inputs = "native-v2"`,
//! `rusticol_core::loop_artifact::LoopV2NativeInputs`) stores, next to its
//! plan, the process's tree program (`loops/v2/native_trees.py`: the trees,
//! the Born root closure, the counterterm program and closure, the external
//! fields and the parameter program, with their saved exact evaluators and
//! binary64 SymJIT plane applications) and an assembly record
//! (`loops/v2/native_process.py`). [`V2Runtime`] evaluates a batch from those
//! alone, in binary64, through the plane applications: no Python and no
//! Symbolica state (so no licence, REQUIREMENTS_ADDENDUM G (e)):
//!
//! 1. the tree program binds the external model parameters (the request's
//!    fixed parameters over the caller's over the defaults; the scale
//!    parameter is `sqrt(mu_R²)` of each point) and forms every point's block
//!    rows, Born components, `|A0|²` per squared Born order and counterterm
//!    and field components (`loop_v2::trees::TreeProgram`);
//! 2. the plan's executor dresses and reduces every family
//!    (`loop_execution::v2::V2Executor`, NinjaRS through
//!    [`crate::runtime::V2ReductionBackend`]);
//! 3. the components are assembled per sector exactly as
//!    `loops/v2/assemble._PointEvaluation._assemble` does for the Python
//!    API's native lane (`plan_export.NativeExecutor(inputs="native")`), in
//!    the same binary64 operations and order: the bare virtual
//!    `2 Re(i/(16π² avg) Σ lanes)` summed over the lanes in the executor's
//!    order, `uv = uv_local + uv_external_fields`,
//!    `virtual = (bare + uv) + r2`. Results are therefore bitwise those of
//!    `pyamplicol.Runtime.load(...).evaluate` on the same exact inputs.
//!
//! Every pole selection of REQUIREMENTS_ADDENDUM I is served ([`V2Poles`]):
//! "all", "none" (finite-only: the pole-only tree and counterterm work and
//! the reduction's pole columns skipped natively) and the split selections
//! "uv" and "ir". A split selection reduces every lane with its UV residue
//! (`[finite, 1/ε, 1/ε², 1/ε_UV]`) and runs the stored companion of the
//! certified-scaleless families (its plan and tree program, read with the
//! process; only their UV residues are kept, their combined integrals
//! vanishing), then splits every component by origin as
//! `loops/v2/poles.split_sector` / `select_sector` do, in the same binary64
//! operations and order (`assemble._PointEvaluation.set_lanes` and
//! `_assemble`): bitwise the Python API's values. Catani's operator is not
//! needed for the selection (it is the Python API's separate IR-pole
//! prediction, `PointResult.ir_poles`, not a part of the values), so it is not
//! formed here. The stored correlated and resolved columns are not served
//! here (the Python API evaluates them, without regenerating).

use crate::runtime::{PoleSelection, V2Reducer, V2ReductionBackend};
use rusticol_core::VerifiedArtifact;
use rusticol_core::loop_artifact::{
    LOOP_V2_RUNTIME_CAPABILITY, LoadedLoopV2Artifact, LoopV2ExecutionMetadata, LoopV2NativeTrees,
    load_loop_v2_artifact, read_loop_v2_native_inputs,
};
use rusticol_core::loop_execution::v2::{LaneKey, Poles, V2Executor, V2Options};
use rusticol_core::loop_v2::dress::{V2Batch, parse_rational};
use rusticol_core::loop_v2::trees::{TreeProgram, TreeRuntime};
use rusticol_core::{RusticolError, RusticolResult};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;
use symbolica::domains::rational::Rational;

/// The native assembly record's schema (`native_process.SCHEMA`).
pub const NATIVE_PROCESS_SCHEMA: u32 = 2;
/// SIMD lanes per chunk, as the Python API's executor (`plan_export.DEFAULT_LANES`).
pub const V2_LANES: usize = 16;

/// The `uv_external_fields` component (`loops/v2/fields.FIELD_COMPONENT`).
const FIELD_COMPONENT: &str = "uv_external_fields";
/// Laurent powers in coefficient order (finite, 1/ε, 1/ε²).
const LAURENT_POWERS: [i32; 3] = [0, -1, -2];
/// The bare virtual's loop factor denominator (`conventions.LOOP_FACTOR_DENOMINATOR`).
const LOOP_FACTOR_DENOMINATOR: f64 = 16.0;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("plan v4 runtime: {}", message.into()))
}

type Orders = Vec<(String, i64)>;

/// Coupling orders compared as sets of (name, power) with zero powers dropped.
fn canonical(orders: &[(String, i64)]) -> Orders {
    let mut out: Orders = orders.iter().filter(|(_, p)| *p != 0).cloned().collect();
    out.sort();
    out
}

#[derive(Deserialize)]
struct NativeTreesRecord {
    squared_orders: Vec<Orders>,
    counterterm_outputs: Vec<(String, Orders, i32)>,
    #[serde(flatten)]
    _rest: BTreeMap<String, serde_json::Value>,
}

/// The assembly record of a native process (`native_process.NativeProcess`).
#[derive(Deserialize)]
struct NativeRecord {
    schema: u32,
    /// (label, initial, PDG code) per leg, by label.
    legs: Vec<(usize, bool, i32)>,
    /// (sector, its (Born, loop) order pairs), in the plan's column order.
    sectors: Vec<(Orders, Vec<(Orders, Orders)>)>,
    average: i64,
    fixed_parameters: BTreeMap<String, String>,
    trees: NativeTreesRecord,
    #[serde(flatten)]
    _rest: BTreeMap<String, serde_json::Value>,
}

/// One evaluated point: the Born per declared Born order and the complete
/// virtual per declared sector, `[finite, 1/ε, 1/ε²]` (finite only without
/// poles), or the executor's failures.
pub type V2Point = Result<(Vec<f64>, Vec<Vec<f64>>), Vec<String>>;

/// The pole selection of an evaluation (REQUIREMENTS_ADDENDUM I,
/// `loops/v2/poles.py`): the renormalised virtual's poles (`All`), the
/// finite part alone (`None`), the bare loop's UV poles (`Uv`: the virtual's
/// poles are the unrenormalised amplitude's UV poles, its finite part the
/// renormalised one's) or the renormalised virtual's IR poles (`Ir`). The
/// finite part is identical in every selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum V2Poles {
    #[default]
    All,
    None,
    Uv,
    Ir,
}

impl V2Poles {
    fn poles(self) -> bool {
        self != Self::None
    }

    fn split(self) -> bool {
        matches!(self, Self::Uv | Self::Ir)
    }

    /// The executor's lane columns.
    fn lanes(self) -> Poles {
        match self {
            Self::All => Poles::All,
            Self::None => Poles::None,
            Self::Uv | Self::Ir => Poles::Split,
        }
    }
}

/// The certified-scaleless companion of a process (its stored plan and tree
/// program), which only the split selections run; built on first use.
struct Companion {
    trees: TreeProgram,
    runtime: TreeRuntime<f64>,
    executor: V2Executor,
}

/// Laurent coefficients `[finite, 1/ε, 1/ε²]` added coefficient-wise
/// (`assemble.Laurent.__add__`).
fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// The selected virtual of one sector in a split selection, from its
/// combined bare Laurent `bare`, the bare loop's UV Laurent `bare_uv`
/// (finite part unused) and the counterterm components: `poles.split_sector`
/// then `poles.select_sector`, operation for operation.
fn split_virtual(
    mode: V2Poles,
    bare: [f64; 3],
    bare_uv: [f64; 3],
    r2: [f64; 3],
    uv_local: [f64; 3],
    fields: [f64; 3],
) -> [f64; 3] {
    // The UV (single, double) poles per component; the field factors carry
    // what makes the renormalised virtual UV finite.
    let field_uv = [
        -(bare_uv[1] + uv_local[1] + r2[1]),
        -(bare_uv[2] + uv_local[2] + r2[2]),
    ];
    let part = |value: [f64; 3], uv: [f64; 2]| -> [f64; 3] {
        match mode {
            V2Poles::Uv => [value[0], uv[0], uv[1]],
            _ => [value[0], value[1] - uv[0], value[2] - uv[1]],
        }
    };
    let bare_part = part(bare, [bare_uv[1], bare_uv[2]]);
    let r2_part = part(r2, [r2[1], r2[2]]);
    let local_part = part(uv_local, [uv_local[1], uv_local[2]]);
    let field_part = part(fields, field_uv);
    let uv = add(local_part, field_part);
    let virtual_part = add(add(bare_part, uv), r2_part);
    match mode {
        // The bare loop's UV poles with the renormalised finite part.
        V2Poles::Uv => [virtual_part[0], bare_part[1], bare_part[2]],
        _ => virtual_part,
    }
}

/// A plan v4 process with native point inputs, evaluated in binary64.
pub struct V2Runtime {
    artifact: LoadedLoopV2Artifact,
    record: NativeRecord,
    trees: TreeProgram,
    runtime: TreeRuntime<f64>,
    executor: V2Executor,
    backend: V2ReductionBackend,
    /// The stored companion's tree program (none: the process has no
    /// certified-scaleless families) and, once a split selection ran, the
    /// companion itself.
    scaleless_inputs: Option<LoopV2NativeTrees>,
    scaleless: Option<Companion>,
    threads: Option<usize>,
    /// The caller's model parameter values (exact).
    parameters: BTreeMap<String, Rational>,
    /// Per declared Born order its squared Born order in the tree program
    /// (none: zero), per declared sector its record sector.
    born_index: Vec<Option<usize>>,
    sector_index: Vec<usize>,
}

/// Whether the artifact at `path` serves its processes through the plan v4
/// runtime (every process requires the v2 runtime capability).
pub fn is_v2_artifact(path: impl AsRef<Path>) -> RusticolResult<bool> {
    let artifact = rusticol_core::VerifiedArtifact::open(path)?;
    Ok(artifact.manifest().runtime.required_runtime_capabilities == [LOOP_V2_RUNTIME_CAPABILITY])
}

impl V2Runtime {
    /// Open one process (`process`: its id or expression; none: the only
    /// one) of a plan v4 artifact with native point inputs. `threads` bounds
    /// the executor's threads (none: every core).
    pub fn open(
        path: impl AsRef<Path>,
        process: Option<&str>,
        threads: Option<usize>,
    ) -> RusticolResult<Self> {
        let verified = VerifiedArtifact::open(path)?;
        let artifact = load_loop_v2_artifact(&verified, process)?;
        let native = read_loop_v2_native_inputs(&verified, &artifact)?.ok_or_else(|| {
            invalid("the process has no native point inputs; regenerate the artifact")
        })?;
        let record: NativeRecord = serde_json::from_slice(&native.record_json)
            .map_err(|e| invalid(format!("native process record: {e}")))?;
        if record.schema != NATIVE_PROCESS_SCHEMA {
            return Err(invalid(
                "unsupported native process schema; regenerate the artifact",
            ));
        }
        let program = artifact.program.clone();
        let trees = TreeProgram::load(
            &native.trees_json,
            &native.evaluators,
            Some(&native.planes),
            &program,
        )?;
        if !trees.has_counterterms() {
            return Err(invalid("the tree program has no counterterm components"));
        }
        let runtime = trees.runtime(&0.0)?;
        let executor = V2Executor::new(program, V2_LANES, threads)?;
        if native.scaleless.is_some() != artifact.scaleless_program.is_some() {
            return Err(invalid(
                "the companion's tree program is stored exactly with its plan",
            ));
        }
        // The Python API's binary64 backend (`v2_executor.backend_of`).
        let reducer =
            V2Reducer::new("double", None, "fast", true).map_err(|e| invalid(e.to_string()))?;
        let metadata = &artifact.metadata;
        let squared: Vec<Orders> = record
            .trees
            .squared_orders
            .iter()
            .map(|o| canonical(o))
            .collect();
        let born_index = metadata
            .born_orders
            .iter()
            .map(|order| {
                squared
                    .iter()
                    .position(|o| *o == canonical(&order_pairs(order)))
            })
            .collect();
        let sectors: Vec<Orders> = record.sectors.iter().map(|(s, _)| canonical(s)).collect();
        let sector_index = metadata
            .sectors
            .iter()
            .map(|sector| {
                let wanted = canonical(&order_pairs(&sector.order));
                sectors
                    .iter()
                    .position(|s| *s == wanted)
                    .ok_or_else(|| invalid("a declared sector is not in the native record"))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        if record.legs.len() != program_legs(&artifact) {
            return Err(invalid("the native record's legs differ from the plan's"));
        }
        Ok(Self {
            artifact,
            record,
            trees,
            runtime,
            executor,
            backend: V2ReductionBackend {
                reducer,
                masters: PoleSelection::BOTH,
            },
            scaleless_inputs: native.scaleless,
            scaleless: None,
            threads,
            parameters: BTreeMap::new(),
            born_index,
            sector_index,
        })
    }

    /// The companion of the certified-scaleless families, built on its
    /// first use (`plan_export.NativeExecutor.scaleless`): its plan's
    /// columns are the process's sectors.
    fn companion(&mut self) -> RusticolResult<Option<&mut Companion>> {
        if self.scaleless.is_none() {
            let (Some(stored), Some(program)) =
                (&self.scaleless_inputs, &self.artifact.scaleless_program)
            else {
                return Ok(None);
            };
            if program.columns != self.record.sectors.len() {
                return Err(invalid("the companion's columns are not the sectors"));
            }
            let trees = TreeProgram::load(
                &stored.trees_json,
                &stored.evaluators,
                Some(&stored.planes),
                program,
            )?;
            if trees.parameter_names() != self.trees.parameter_names() {
                return Err(invalid(
                    "the companion's model parameters differ from the process's",
                ));
            }
            let runtime = trees.runtime(&0.0)?;
            let executor = V2Executor::new(program.clone(), V2_LANES, self.threads)?;
            self.scaleless = Some(Companion {
                trees,
                runtime,
                executor,
            });
        }
        Ok(self.scaleless.as_mut())
    }

    /// Whether the process stores the companion of certified-scaleless
    /// families (the split selections run it).
    pub fn has_companion(&self) -> bool {
        self.scaleless_inputs.is_some()
    }

    pub fn metadata(&self) -> &LoopV2ExecutionMetadata {
        &self.artifact.metadata
    }

    /// The declared metadata as JSON (orders, sectors, scheme).
    pub fn metadata_json(&self) -> RusticolResult<String> {
        serde_json::to_string(&self.artifact.metadata)
            .map_err(|e| RusticolError::serialization(e.to_string()))
    }

    pub fn external_count(&self) -> usize {
        self.record.legs.len()
    }

    pub fn helicities(&self) -> usize {
        self.artifact.helicities.len()
    }

    pub fn born_orders(&self) -> usize {
        self.born_index.len()
    }

    pub fn sectors(&self) -> usize {
        self.sector_index.len()
    }

    pub fn plan_sha256(&self) -> &str {
        &self.artifact.plan_sha256
    }

    /// Set the caller's model parameter values (exact decimal or fraction
    /// strings) by name; the request's fixed parameters still take
    /// precedence, as in the Python API.
    pub fn set_parameters<'a>(
        &mut self,
        values: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> RusticolResult<()> {
        let names = self.trees.parameter_names();
        for (name, value) in values {
            if !names.iter().any(|known| known == name) {
                return Err(invalid(format!("unknown model parameter {name:?}")));
            }
            let exact = parse_rational(value).map_err(|e| invalid(e.message()))?;
            self.parameters.insert(name.to_owned(), exact);
        }
        Ok(())
    }

    /// The external parameter values of a batch (`native_trees.
    /// batch_parameters`): fixed over the caller's over the defaults.
    fn parameter_values(&self) -> RusticolResult<Vec<Rational>> {
        let names = self.trees.parameter_names();
        let defaults = self.trees.parameter_defaults();
        names
            .iter()
            .zip(defaults)
            .map(|(name, default)| {
                if let Some(value) = self.record.fixed_parameters.get(name) {
                    return parse_rational(value).map_err(|e| invalid(e.message()));
                }
                Ok(self
                    .parameters
                    .get(name)
                    .cloned()
                    .unwrap_or_else(|| default.clone()))
            })
            .collect()
    }

    /// Evaluate a batch: `momenta` holds per point the physical momenta of
    /// every leg (`4 × legs` exact decimals, label order), `scales` mu_R² per
    /// point, `mode` the pole selection. Points fail alone.
    pub fn evaluate(
        &mut self,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
    ) -> RusticolResult<Vec<V2Point>> {
        let poles = mode.poles();
        let legs = self.record.legs.len();
        if momenta.len() != scales.len() {
            return Err(invalid("one mu_R² per point"));
        }
        if momenta.is_empty() {
            return Ok(Vec::new());
        }
        if momenta.iter().any(|point| point.len() != 4 * legs) {
            return Err(invalid("momenta need 4 components per leg"));
        }
        if scales.iter().any(|scale| *scale <= Rational::from(0)) {
            return Err(invalid("mu_R² must be positive"));
        }
        // All-incoming momenta: final legs negated.
        let incoming: Vec<Vec<Rational>> = momenta
            .iter()
            .map(|point| {
                point
                    .chunks_exact(4)
                    .zip(&self.record.legs)
                    .flat_map(|(p, &(_, initial, _))| {
                        p.iter()
                            .map(move |x| if initial { x.clone() } else { -x.clone() })
                    })
                    .collect()
            })
            .collect();
        let values = self.parameter_values()?;
        let zero = 0.0_f64;
        let bound = self.trees.bind(&mut self.runtime, &values, scales, &zero)?;
        let inputs = self.trees.inputs(
            &mut self.runtime,
            self.executor.plan(),
            &bound.trees,
            &incoming,
            &bound.constants,
            poles,
            &zero,
        )?;
        let batch = V2Batch {
            momenta: incoming.clone(),
            scales: scales.to_vec(),
            masses: bound.masses,
            // The on-shell external invariants (as the Python binding's).
            leg_masses: bound.leg_masses,
            constants: bound.constants,
            block_rows: inputs.block_rows,
            born: inputs.born,
        };
        let options = V2Options {
            poles: mode.lanes(),
            ..V2Options::default()
        };
        let (results, _report) =
            self.executor
                .evaluate(&batch, &zero, &mut self.backend, options)?;
        let born = inputs
            .born_squared
            .ok_or_else(|| invalid("the tree program forms no |A0|²"))?;
        let counterterms = inputs
            .counterterms
            .ok_or_else(|| invalid("the tree program forms no counterterm components"))?;
        // The split selections also run the companion of the certified-
        // scaleless families on its own tree inputs (schedule B), for the UV
        // residues of its lanes (`plan_export.NativeExecutor.reduce`).
        let twin_results = if mode.split() && self.companion()?.is_some() {
            let twin = self.scaleless.as_mut().expect("the companion is built");
            let bound = twin.trees.bind(&mut twin.runtime, &values, scales, &zero)?;
            let twin_inputs = twin.trees.inputs(
                &mut twin.runtime,
                twin.executor.plan(),
                &bound.trees,
                &incoming,
                &bound.constants,
                true,
                &zero,
            )?;
            let twin_batch = V2Batch {
                momenta: incoming,
                scales: scales.to_vec(),
                masses: bound.masses,
                leg_masses: bound.leg_masses,
                constants: bound.constants,
                block_rows: twin_inputs.block_rows,
                born: twin_inputs.born,
            };
            let (twin_results, _report) =
                twin.executor
                    .evaluate(&twin_batch, &zero, &mut self.backend, options)?;
            if twin_results.len() != results.len() {
                return Err(invalid("the companion evaluated another batch"));
            }
            Some(twin_results)
        } else {
            None
        };
        let width = if poles { 3 } else { 1 };
        let lane_width = mode.lanes().width();
        let average = self.record.average as f64;
        // Python: numbers.pair(0, 1) * (1 / (16 * pi * pi * average)).
        let factor =
            1.0 / (LOOP_FACTOR_DENOMINATOR * std::f64::consts::PI * std::f64::consts::PI * average);
        // 2 * (complex(0, factor) * sum).real
        let loop_factor = |(re, im): (f64, f64)| 2.0 * (0.0 * re - factor * im);
        let sectors = self.record.sectors.len();
        let outputs: Vec<(String, usize, i32)> = self
            .record
            .trees
            .counterterm_outputs
            .iter()
            .map(|(component, sector, power)| {
                let wanted = canonical(sector);
                let s = self
                    .record
                    .sectors
                    .iter()
                    .position(|(o, _)| canonical(o) == wanted)
                    .ok_or_else(|| invalid("a counterterm output's sector is not in the record"))?;
                Ok((component.clone(), s, *power))
            })
            .collect::<RusticolResult<_>>()?;
        let zero_pair = (0.0, 0.0);
        let mut out = Vec::with_capacity(results.len());
        for (index, result) in results.into_iter().enumerate() {
            let twin = twin_results.as_ref().map(|twin| &twin[index]);
            let mut failures = result.failures;
            if let Some(twin) = twin {
                failures.extend(twin.failures.iter().cloned());
            }
            if !failures.is_empty() {
                out.push(Err(failures));
                continue;
            }
            // Per sector the lanes' combined Laurent (and in a split
            // selection their UV Laurent `(finite, 1/ε_UV, 0)`), summed in
            // the executor's order, the companion's lanes after the
            // process's (`set_lanes`: their combined integrals vanish, only
            // their UV residues are kept).
            let mut sums: Vec<Option<Vec<(f64, f64)>>> = vec![None; sectors];
            let mut uv_sums: Vec<Option<[(f64, f64); 3]>> = vec![None; sectors];
            let mut lanes: Vec<(&LaneKey, &[_], bool)> = result
                .families
                .iter()
                .map(|(key, value)| (key, value.as_slice(), false))
                .collect();
            if let Some(twin) = twin {
                lanes.extend(
                    twin.families
                        .iter()
                        .map(|(key, value)| (key, value.as_slice(), true)),
                );
            }
            for (key, value, vanishing) in lanes {
                if key.column >= sectors || value.len() != lane_width {
                    return Err(invalid("an executor lane outside the record's sectors"));
                }
                let terms = sums[key.column].get_or_insert_with(|| vec![zero_pair; width]);
                for (k, term) in terms.iter_mut().enumerate() {
                    let (re, im) = if vanishing {
                        zero_pair
                    } else {
                        (value[k].re, value[k].im)
                    };
                    *term = (term.0 + re, term.1 + im);
                }
                if mode.split() {
                    let finite = if vanishing {
                        zero_pair
                    } else {
                        (value[0].re, value[0].im)
                    };
                    let uv = (value[3].re, value[3].im);
                    let terms = uv_sums[key.column].get_or_insert([zero_pair; 3]);
                    for (term, (re, im)) in terms.iter_mut().zip([finite, uv, zero_pair]) {
                        *term = (term.0 + re, term.1 + im);
                    }
                }
            }
            // (counterterm component, sector, power) -> value.
            let mut local: BTreeMap<(&str, usize, i32), f64> = BTreeMap::new();
            for ((component, s, power), value) in outputs.iter().zip(&counterterms[index]) {
                local.insert((component.as_str(), *s, *power), *value);
            }
            let component = |name: &str, s: usize, k: usize| -> f64 {
                local
                    .get(&(name, s, LAURENT_POWERS[k]))
                    .copied()
                    .unwrap_or(0.0)
            };
            let laurent = |name: &str, s: usize| -> [f64; 3] {
                [
                    component(name, s, 0),
                    component(name, s, 1),
                    component(name, s, 2),
                ]
            };
            let mut virtual_values = Vec::with_capacity(self.sector_index.len());
            for &s in &self.sector_index {
                let bare = |k: usize| sums[s].as_ref().map_or(0.0, |terms| loop_factor(terms[k]));
                if mode.split() {
                    let bare_uv = |k: usize| uv_sums[s].map_or(0.0, |terms| loop_factor(terms[k]));
                    let selected = split_virtual(
                        mode,
                        [bare(0), bare(1), bare(2)],
                        [bare_uv(0), bare_uv(1), bare_uv(2)],
                        laurent("r2", s),
                        laurent("uv_local", s),
                        laurent(FIELD_COMPONENT, s),
                    );
                    virtual_values.push(selected.to_vec());
                    continue;
                }
                let mut values = Vec::with_capacity(width);
                for k in 0..width {
                    let uv = component("uv_local", s, k) + component(FIELD_COMPONENT, s, k);
                    values.push(bare(k) + uv + component("r2", s, k));
                }
                virtual_values.push(values);
            }
            let born_values = self
                .born_index
                .iter()
                .map(|found| found.map_or(0.0, |k| born[index][k]))
                .collect();
            out.push(Ok((born_values, virtual_values)));
        }
        Ok(out)
    }
}

fn order_pairs(orders: &[(String, u32)]) -> Orders {
    orders
        .iter()
        .map(|(name, power)| (name.clone(), i64::from(*power)))
        .collect()
}

fn program_legs(artifact: &LoadedLoopV2Artifact) -> usize {
    artifact.program.legs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orders(pairs: &[(&str, i64)]) -> Orders {
        pairs.iter().map(|(n, p)| ((*n).to_owned(), *p)).collect()
    }

    #[test]
    fn orders_compare_as_sets_without_zero_powers() {
        assert_eq!(
            canonical(&orders(&[("QED", 0), ("QCD", 2), ("NP", 1)])),
            orders(&[("NP", 1), ("QCD", 2)])
        );
        assert!(canonical(&orders(&[("QCD", 0)])).is_empty());
        // The plan's declared (unsigned) orders meet the record's the same way.
        let declared = vec![("QED".to_owned(), 0_u32), ("QCD".to_owned(), 4)];
        assert_eq!(
            canonical(&order_pairs(&declared)),
            canonical(&orders(&[("QCD", 4)]))
        );
    }

    #[test]
    fn native_record_reads_the_assembly_fields_and_keeps_the_rest() {
        // `native_process.NativeProcess`'s record: the fields the assembly
        // reads; any other key (the tree program's own, observables) passes.
        let json = r#"{
            "schema": 2,
            "legs": [[1, true, 21], [2, true, 21], [3, false, 21], [4, false, 21]],
            "sectors": [[[["QCD", 6]], [[[["QCD", 2]], [["QCD", 4]]]]]],
            "average": 256,
            "fixed_parameters": {"MB": "0"},
            "trees": {
                "squared_orders": [[["QCD", 4], ["QED", 0]]],
                "counterterm_outputs": [["r2", [["QCD", 6]], 0], ["uv_local", [["QCD", 6]], -1]],
                "program": {"anything": [1, 2]}
            },
            "observables": null
        }"#;
        let record: NativeRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.schema, NATIVE_PROCESS_SCHEMA);
        assert_eq!(record.legs.len(), 4);
        assert!(record.legs[0].1 && !record.legs[3].1);
        assert_eq!(record.sectors.len(), 1);
        assert_eq!(canonical(&record.sectors[0].0), orders(&[("QCD", 6)]));
        assert_eq!(record.average, 256);
        assert_eq!(record.fixed_parameters["MB"], "0");
        assert_eq!(
            canonical(&record.trees.squared_orders[0]),
            orders(&[("QCD", 4)])
        );
        assert_eq!(record.trees.counterterm_outputs[1].0, "uv_local");
        assert_eq!(record.trees.counterterm_outputs[1].2, -1);
        // A field the assembly needs is required.
        let missing = json.replace(r#""average": 256,"#, "");
        assert!(serde_json::from_str::<NativeRecord>(&missing).is_err());
    }

    #[test]
    fn split_selections_split_every_component_as_the_python_api() {
        // Exactly representable values, so the identities below are exact.
        let bare = [10.0, -6.0, 4.0];
        let bare_uv = [10.0, 3.0, 0.0];
        let r2 = [0.5, 0.0, 0.0];
        let uv_local = [1.0, -2.0, 0.0];
        let fields = [0.25, 0.75, 0.0];
        let finite = bare[0] + (uv_local[0] + fields[0]) + r2[0];
        let uv = split_virtual(V2Poles::Uv, bare, bare_uv, r2, uv_local, fields);
        let ir = split_virtual(V2Poles::Ir, bare, bare_uv, r2, uv_local, fields);
        // The finite part is the renormalised virtual's in every selection.
        assert_eq!(uv[0], finite);
        assert_eq!(ir[0], finite);
        // "uv": the bare loop's UV poles (the double pole is infrared).
        assert_eq!(&uv[1..], &[3.0, 0.0]);
        // "ir": the renormalised virtual's poles less their UV parts; the
        // field factors make the renormalised virtual UV finite, so they are
        // its combined poles: bare + uv_local + fields + r2.
        let combined = [
            bare[1] + (uv_local[1] + fields[1]) + r2[1],
            bare[2] + (uv_local[2] + fields[2]) + r2[2],
        ];
        assert_eq!(&ir[1..], &combined);
        // A finite-only or combined selection is not split here.
        assert!(!V2Poles::All.split() && !V2Poles::None.split());
        assert!(V2Poles::Uv.split() && V2Poles::Ir.split());
        assert_eq!(V2Poles::Ir.lanes(), Poles::Split);
        assert_eq!(V2Poles::None.lanes(), Poles::None);
        assert!(!V2Poles::None.poles() && V2Poles::Uv.poles());
    }

    #[test]
    fn a_path_without_an_artifact_is_refused() {
        let missing = std::env::temp_dir().join(format!(
            "rusticol-v2-runtime-no-artifact-{}",
            std::process::id()
        ));
        assert!(!missing.exists());
        assert!(is_v2_artifact(&missing).is_err());
        assert!(V2Runtime::open(&missing, None, Some(1)).is_err());
    }
}
