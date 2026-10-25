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
//! Catani's pole prediction is not formed (the native lane has none), and
//! correlated and resolved observables are not served (their columns are not
//! stored natively yet). The pole selection is all or none (finite-only); the
//! split selections ("uv", "ir") go through the Python API.

use crate::runtime::{PoleSelection, V2Reducer, V2ReductionBackend};
use rusticol_core::VerifiedArtifact;
use rusticol_core::loop_artifact::{
    LOOP_V2_RUNTIME_CAPABILITY, LoadedLoopV2Artifact, LoopV2ExecutionMetadata,
    load_loop_v2_artifact, read_loop_v2_native_inputs,
};
use rusticol_core::loop_execution::v2::{Poles, V2Executor, V2Options};
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

/// A plan v4 process with native point inputs, evaluated in binary64.
pub struct V2Runtime {
    artifact: LoadedLoopV2Artifact,
    record: NativeRecord,
    trees: TreeProgram,
    runtime: TreeRuntime<f64>,
    executor: V2Executor,
    backend: V2ReductionBackend,
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
            invalid(
                "the process has no native point inputs (point_inputs = \
                 \"regenerated-at-load\"): it is evaluated through the Python API",
            )
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
            parameters: BTreeMap::new(),
            born_index,
            sector_index,
        })
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
    /// point. Points fail alone.
    pub fn evaluate(
        &mut self,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        poles: bool,
    ) -> RusticolResult<Vec<V2Point>> {
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
            momenta: incoming,
            scales: scales.to_vec(),
            masses: bound.masses,
            // The on-shell external invariants (as the Python binding's).
            leg_masses: bound.leg_masses,
            constants: bound.constants,
            block_rows: inputs.block_rows,
            born: inputs.born,
        };
        let options = V2Options {
            poles: if poles { Poles::All } else { Poles::None },
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
        let width = if poles { 3 } else { 1 };
        let average = self.record.average as f64;
        // Python: numbers.pair(0, 1) * (1 / (16 * pi * pi * average)).
        let factor =
            1.0 / (LOOP_FACTOR_DENOMINATOR * std::f64::consts::PI * std::f64::consts::PI * average);
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
        let mut out = Vec::with_capacity(results.len());
        for (index, result) in results.into_iter().enumerate() {
            if !result.failures.is_empty() {
                out.push(Err(result.failures));
                continue;
            }
            // Bare per sector: the lanes summed in the executor's order.
            let mut sums: Vec<Option<Vec<(f64, f64)>>> = vec![None; sectors];
            for (key, value) in &result.families {
                if key.column >= sectors || value.len() != width {
                    return Err(invalid("an executor lane outside the record's sectors"));
                }
                let terms = sums[key.column].get_or_insert_with(|| vec![(0.0, 0.0); width]);
                for (term, z) in terms.iter_mut().zip(value) {
                    *term = (term.0 + z.re, term.1 + z.im);
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
            let mut virtual_values = Vec::with_capacity(self.sector_index.len());
            for &s in &self.sector_index {
                let mut laurent = Vec::with_capacity(width);
                for k in 0..width {
                    // 2 * (complex(0, factor) * sum).real
                    let bare = sums[s].as_ref().map_or(0.0, |terms| {
                        let (re, im) = terms[k];
                        2.0 * (0.0 * re - factor * im)
                    });
                    let uv = component("uv_local", s, k) + component(FIELD_COMPONENT, s, k);
                    laurent.push(bare + uv + component("r2", s, k));
                }
                virtual_values.push(laurent);
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
