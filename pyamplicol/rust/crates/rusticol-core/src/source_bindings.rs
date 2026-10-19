// SPDX-License-Identifier: 0BSD

//! Checked source scalar ownership over saved dependency evaluators. This module
//! supplies no scalar algebra or renormalization prescription: exact defaults,
//! value/diagnostic pairs and scale substitution are the validated saved programs.

use crate::loop_execution::LoopBackend;
use crate::loop_runtime::{LoadedLoopRuntime, ModelBinding, ScalarOwner, ScalarScaleView};
use crate::source_scalar::decimal_rational;
use crate::{RusticolError, RusticolResult};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, RwLock};
use symbolica::domains::rational::Rational;

pub type SourceOverrides = BTreeMap<String, [String; 2]>;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(message)
}

/// Immutable values and original exact overrides from one atomic owner revision.
/// Fields are private so a retained snapshot cannot be changed after publication.
pub struct SourceSnapshot<T> {
    arguments: T,
    values: T,
    overrides: SourceOverrides,
    revision: u64,
    identity: Arc<()>,
}
impl<T> SourceSnapshot<T> {
    pub fn arguments(&self) -> &T {
        &self.arguments
    }
    pub fn values(&self) -> &T {
        &self.values
    }
    pub fn overrides(&self) -> &SourceOverrides {
        &self.overrides
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
}

struct OwnerWorkspace<B: LoopBackend> {
    values: B::Evaluator,
    defaults: Option<B::Evaluator>,
    guards: Option<B::Evaluator>,
}

/// One mutable global parameter owner. Writers serialize evaluator scratch; a
/// snapshot reader only holds the state lock long enough to clone its Arc. All
/// checks finish before publishing, so failed updates preserve the old revision.
pub struct SourceOwner<B: LoopBackend> {
    loaded: Arc<LoadedLoopRuntime>,
    identity: Arc<()>,
    workspace: Mutex<OwnerWorkspace<B>>,
    current: RwLock<Arc<SourceSnapshot<B::Batch>>>,
}

fn source_record(loaded: &LoadedLoopRuntime) -> RusticolResult<&ScalarOwner> {
    match &loaded.plan().model_binding {
        ModelBinding::Source { owner, .. } => Ok(owner),
        ModelBinding::Legacy => Err(invalid(
            "source scalar owner requires a source model binding",
        )),
    }
}

fn checked_overrides(record: &ScalarOwner, changes: &SourceOverrides) -> RusticolResult<()> {
    for (name, value) in changes {
        if record
            .fixed_parameters
            .iter()
            .any(|(fixed, _)| fixed == name)
        {
            return Err(invalid(format!(
                "parameter fixed during generation cannot be updated: {name}"
            )));
        }
        let index = record
            .external_names
            .iter()
            .position(|external| external == name)
            .ok_or_else(|| invalid(format!("unknown external parameter {name:?}")))?;
        decimal_rational(&value[0]).map_err(invalid)?;
        let imaginary = decimal_rational(&value[1]).map_err(invalid)?;
        if record.external_types[index] == "real" && imaginary != Rational::from(0) {
            return Err(invalid(format!(
                "real external parameter {name:?} has an imaginary part"
            )));
        }
    }
    Ok(())
}

impl<B: LoopBackend> OwnerWorkspace<B> {
    fn evaluate(
        &mut self,
        record: &ScalarOwner,
        backend: &B,
        overrides: &SourceOverrides,
    ) -> RusticolResult<(B::Batch, B::Batch)> {
        checked_overrides(record, overrides)?;
        let empty = backend.ingress_complex(&[], 1, 0, vec![0])?;
        let arguments = if let Some(defaults) = &mut self.defaults {
            let missing: Vec<_> = record
                .external_names
                .iter()
                .enumerate()
                .filter_map(|(index, name)| (!overrides.contains_key(name)).then_some(index))
                .collect();
            let defaults = if missing.is_empty() {
                None
            } else {
                let value = B::evaluate(defaults, &empty)?;
                B::require_finite(
                    &B::select(
                        &value,
                        missing.iter().flat_map(|&i| [2 * i, 2 * i + 1]).collect(),
                    )?,
                    "model external defaults and diagnostics",
                    1,
                )?;
                Some(value)
            };
            let columns = record
                .external_names
                .iter()
                .enumerate()
                .map(|(index, name)| {
                    if let Some(value) = overrides.get(name) {
                        backend.ingress_complex(std::slice::from_ref(value), 1, 1, vec![0])
                    } else {
                        B::select(
                            defaults.as_ref().expect("missing values evaluated"),
                            vec![2 * index],
                        )
                    }
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            B::concat(&columns.iter().collect::<Vec<_>>())?
        } else {
            empty
        };
        B::require_finite(&arguments, "model external inputs", 1)?;
        if let Some(guards) = &mut self.guards {
            B::require_finite(
                &B::evaluate(guards, &arguments)?,
                "real model input type",
                1,
            )?;
        }
        let values = B::evaluate(&mut self.values, &arguments)?;
        B::require_finite(&values, "model scalar values and active diagnostics", 1)?;
        let values = B::select(&values, (0..record.outputs.len()).map(|i| 2 * i).collect())?;
        Ok((arguments, values))
    }
}

impl<B: LoopBackend> SourceOwner<B> {
    pub(crate) fn loaded(&self) -> &Arc<LoadedLoopRuntime> {
        &self.loaded
    }

    /// Overlay named parameters on the exact overrides captured in the artifact.
    /// The validated descriptor is the only cold declaration/bytecode source.
    pub fn new(
        loaded: Arc<LoadedLoopRuntime>,
        backend: &B,
        parameters: Option<&SourceOverrides>,
    ) -> RusticolResult<Self> {
        let record = source_record(&loaded)?;
        let ModelBinding::Source {
            initial_overrides, ..
        } = &loaded.plan().model_binding
        else {
            unreachable!("source binding checked");
        };
        let mut overrides = initial_overrides.clone();
        if let Some(changes) = parameters {
            overrides.extend(
                changes
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone())),
            );
        }
        let evaluator = |name: &str| backend.evaluator(loaded.evaluator_state(name)?);
        let mut workspace = OwnerWorkspace::<B> {
            values: evaluator(&record.values_state)?,
            defaults: record
                .defaults_state
                .as_deref()
                .map(evaluator)
                .transpose()?,
            guards: record.guards_state.as_deref().map(evaluator).transpose()?,
        };
        let (arguments, values) = workspace.evaluate(record, backend, &overrides)?;
        let identity = Arc::new(());
        let snapshot = SourceSnapshot {
            arguments,
            values,
            overrides,
            revision: 0,
            identity: identity.clone(),
        };
        Ok(Self {
            loaded,
            identity,
            workspace: Mutex::new(workspace),
            current: RwLock::new(Arc::new(snapshot)),
        })
    }

    pub fn snapshot(&self) -> RusticolResult<Arc<SourceSnapshot<B::Batch>>> {
        self.current
            .read()
            .map(|snapshot| snapshot.clone())
            .map_err(|_| invalid("source scalar snapshot lock poisoned"))
    }

    /// Reject a foreign retained state before starting any process workspaces.
    pub fn require_snapshot(&self, snapshot: &SourceSnapshot<B::Batch>) -> RusticolResult<()> {
        if !Arc::ptr_eq(&self.identity, &snapshot.identity) {
            return Err(invalid("snapshot belongs to a different global state"));
        }
        Ok(())
    }

    /// Merge with one current exact revision, then publish all fields together.
    pub fn update(
        &self,
        backend: &B,
        changes: &SourceOverrides,
    ) -> RusticolResult<Arc<SourceSnapshot<B::Batch>>> {
        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| invalid("source scalar update lock poisoned"))?;
        let previous = self.snapshot()?;
        let mut overrides = previous.overrides.clone();
        overrides.extend(
            changes
                .iter()
                .map(|(name, value)| (name.clone(), value.clone())),
        );
        let (arguments, values) =
            workspace.evaluate(source_record(&self.loaded)?, backend, &overrides)?;
        let revision = previous
            .revision
            .checked_add(1)
            .ok_or_else(|| invalid("source scalar revision overflows"))?;
        let candidate = Arc::new(SourceSnapshot {
            arguments,
            values,
            overrides,
            revision,
            identity: self.identity.clone(),
        });
        *self
            .current
            .write()
            .map_err(|_| invalid("source scalar snapshot lock poisoned"))? = candidate.clone();
        Ok(candidate)
    }
}

/// A separate scale workspace tied to exactly one owner, with no update state of
/// its own. A complete process call passes its captured snapshot to both views.
pub struct SourceScale<B: LoopBackend> {
    identity: Arc<()>,
    scale: B::Evaluator,
    values: B::Evaluator,
    scale_index: usize,
    input_count: usize,
    output_count: usize,
}
impl<B: LoopBackend> SourceScale<B> {
    fn new(owner: &SourceOwner<B>, backend: &B, record: &ScalarScaleView) -> RusticolResult<Self> {
        let evaluator = |name: &str| backend.evaluator(owner.loaded.evaluator_state(name)?);
        Ok(Self {
            identity: owner.identity.clone(),
            scale: evaluator(&record.scale_state)?,
            values: evaluator(&record.values_state)?,
            scale_index: record.scale_index,
            input_count: source_record(&owner.loaded)?.external_names.len(),
            output_count: record.outputs.len(),
        })
    }
    pub fn ordinary(owner: &SourceOwner<B>, backend: &B) -> RusticolResult<Option<Self>> {
        let ModelBinding::Source { ordinary_scale, .. } = &owner.loaded.plan().model_binding else {
            unreachable!("owner has a validated source binding");
        };
        ordinary_scale
            .as_ref()
            .map(|record| Self::new(owner, backend, record))
            .transpose()
    }
    pub fn counterterms(owner: &SourceOwner<B>, backend: &B) -> RusticolResult<Self> {
        let ModelBinding::Source {
            counterterm_scale, ..
        } = &owner.loaded.plan().model_binding
        else {
            unreachable!("owner has a validated source binding");
        };
        Self::new(owner, backend, counterterm_scale)
    }
    pub fn evaluate(
        &mut self,
        snapshot: &SourceSnapshot<B::Batch>,
        mu_r2: &B::Batch,
    ) -> RusticolResult<B::Batch> {
        if !Arc::ptr_eq(&self.identity, &snapshot.identity) {
            return Err(invalid(
                "scale snapshot belongs to a different global state",
            ));
        }
        if B::columns(mu_r2) != 1 {
            return Err(invalid("scales require one mu_r2 column"));
        }
        B::require_finite(mu_r2, "renormalization scales", 1)?;
        // Even an empty evaluation checks carrier and precision compatibility.
        let scale = B::evaluate(&mut self.scale, mu_r2)?;
        B::require_finite(&scale, "positive real renormalization scales", 1)?;
        let arguments = B::gather(
            &snapshot.arguments,
            vec![0; B::rows(&scale)],
            B::row_ids(&scale).to_vec(),
        )?;
        let columns = (0..self.input_count)
            .map(|index| {
                if index == self.scale_index {
                    B::select(&scale, vec![2])
                } else {
                    B::select(&arguments, vec![index])
                }
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let values = B::evaluate(
            &mut self.values,
            &B::concat(&columns.iter().collect::<Vec<_>>())?,
        )?;
        B::require_finite(
            &values,
            "scale-dependent scalar values and active diagnostics",
            1,
        )?;
        B::select(&values, (0..self.output_count).map(|i| 2 * i).collect())
    }
}
