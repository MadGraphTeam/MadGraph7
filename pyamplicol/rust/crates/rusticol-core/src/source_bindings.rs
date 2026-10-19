// SPDX-License-Identifier: 0BSD

//! Checked source scalar ownership over saved dependency evaluators. This module
//! supplies no scalar algebra or renormalization prescription: exact defaults,
//! value/diagnostic pairs and scale substitution are the validated saved programs.

use crate::loop_execution::{LoopBackend, LoopLaneFailure, finite_rows};
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

/// An exact source declaration whose numerical realization is unavailable in
/// one carrier. Structural/schema errors never enter this retryable category.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceBindingFailure {
    pub stage: String,
}
impl SourceBindingFailure {
    pub fn error(&self) -> RusticolError {
        RusticolError::evaluation(format!("{} has nonfinite numerical values", self.stage))
    }
}

struct SourceValues<T> {
    arguments: T,
    values: T,
}

enum SourceRealization<T> {
    Ready(SourceValues<T>),
    Failed(SourceBindingFailure),
}
impl<T> SourceRealization<T> {
    fn values(&self) -> RusticolResult<&SourceValues<T>> {
        match self {
            Self::Ready(values) => Ok(values),
            Self::Failed(failure) => Err(failure.error()),
        }
    }
    fn failure(&self) -> Option<&SourceBindingFailure> {
        match self {
            Self::Ready(_) => None,
            Self::Failed(failure) => Some(failure),
        }
    }
}

/// Immutable original exact overrides and realization from one atomic revision.
/// Fields are private so a retained snapshot cannot be changed after publication.
pub struct SourceSnapshot<T> {
    realization: SourceRealization<T>,
    overrides: SourceOverrides,
    revision: u64,
    identity: Arc<()>,
}
impl<T> SourceSnapshot<T> {
    pub fn arguments(&self) -> RusticolResult<&T> {
        Ok(&self.realization.values()?.arguments)
    }
    pub fn values(&self) -> RusticolResult<&T> {
        Ok(&self.realization.values()?.values)
    }
    pub fn binding_failure(&self) -> Option<&SourceBindingFailure> {
        self.realization.failure()
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
    // A valid empty batch checks arithmetic compatibility even when this
    // carrier cannot realize the current model. It contains no model values.
    carrier: B::Batch,
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
    fn new(loaded: &LoadedLoopRuntime, backend: &B) -> RusticolResult<Self> {
        let record = source_record(loaded)?;
        let evaluator = |name: &str| backend.evaluator(loaded.evaluator_state(name)?);
        Ok(Self {
            values: evaluator(&record.values_state)?,
            defaults: record
                .defaults_state
                .as_deref()
                .map(evaluator)
                .transpose()?,
            guards: record.guards_state.as_deref().map(evaluator).transpose()?,
        })
    }

    fn evaluate(
        &mut self,
        record: &ScalarOwner,
        backend: &B,
        overrides: &SourceOverrides,
    ) -> RusticolResult<SourceRealization<B::Batch>> {
        checked_overrides(record, overrides)?;
        // Opaque evaluator/ingress errors remain hard errors. Only explicit
        // nonfinite rows are numerical outcomes; no exception text is parsed.
        macro_rules! finite {
            ($batch:expr, $stage:expr) => {
                if !B::nonfinite_rows($batch).is_empty() {
                    return Ok(SourceRealization::Failed(SourceBindingFailure {
                        stage: $stage.into(),
                    }));
                }
            };
        }
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
                finite!(
                    &B::select(
                        &value,
                        missing.iter().flat_map(|&i| [2 * i, 2 * i + 1]).collect(),
                    )?,
                    "model external defaults and diagnostics"
                );
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
        finite!(&arguments, "model external inputs");
        if let Some(guards) = &mut self.guards {
            finite!(&B::evaluate(guards, &arguments)?, "real model input type");
        }
        let values = B::evaluate(&mut self.values, &arguments)?;
        finite!(&values, "model scalar values and active diagnostics");
        let values = B::select(&values, (0..record.outputs.len()).map(|i| 2 * i).collect())?;
        Ok(SourceRealization::Ready(SourceValues { arguments, values }))
    }
}

impl<B: LoopBackend> SourceOwner<B> {
    pub fn loaded(&self) -> &Arc<LoadedLoopRuntime> {
        &self.loaded
    }

    /// Rebind an already admitted exact snapshot, recording this carrier's
    /// numerical outcome even when it cannot realize the declaration. Ownership
    /// is checked before any replay; no later live parameter state is consulted.
    /// This is for independent evaluation attempts, not parameter admission.
    pub fn from_snapshot(
        backend: &B,
        original_owner: &Self,
        original: &SourceSnapshot<B::Batch>,
    ) -> RusticolResult<Self> {
        original_owner.require_snapshot(original)?;
        let target = Self::new_with_alternate(
            original_owner.loaded.clone(),
            backend,
            Some(original.overrides()),
            |_, _| Ok(()),
        )?;
        let replay = target.reconstruct_snapshot(backend, original_owner, original)?;
        *target
            .current
            .write()
            .map_err(|_| invalid("source scalar snapshot lock poisoned"))? = replay;
        Ok(target)
    }

    /// Overlay named parameters on the exact overrides captured in the artifact.
    /// The validated descriptor is the only cold declaration/bytecode source.
    pub fn new(
        loaded: Arc<LoadedLoopRuntime>,
        backend: &B,
        parameters: Option<&SourceOverrides>,
    ) -> RusticolResult<Self> {
        Self::new_with_alternate(loaded, backend, parameters, |_, failure| {
            Err(failure.error())
        })
    }

    /// Publish an exact declaration only after the primary realization or an
    /// explicitly configured alternate validates the complete candidate. The
    /// consumer owns alternate arithmetic; higher values never fill primary
    /// buffers. Schema errors bypass the alternate entirely.
    pub fn new_with_alternate(
        loaded: Arc<LoadedLoopRuntime>,
        backend: &B,
        parameters: Option<&SourceOverrides>,
        alternate: impl FnOnce(&SourceOverrides, &SourceBindingFailure) -> RusticolResult<()>,
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
        let mut workspace = OwnerWorkspace::<B>::new(&loaded, backend)?;
        let realization = workspace.evaluate(record, backend, &overrides)?;
        if let Some(failure) = realization.failure() {
            alternate(&overrides, failure)?;
        }
        let carrier = backend.ingress_complex(&[], 1, 0, vec![0])?;
        let identity = Arc::new(());
        let snapshot = SourceSnapshot {
            realization,
            overrides,
            revision: 0,
            identity: identity.clone(),
        };
        Ok(Self {
            loaded,
            identity,
            carrier,
            workspace: Mutex::new(workspace),
            current: RwLock::new(Arc::new(snapshot)),
        })
    }

    /// Validate the complete exact map in another carrier without creating or
    /// publishing an owner. No artifact defaults are replaced by rounded data.
    /// A numerical failure is returned as data; schema/executor errors stay hard.
    pub fn validate_exact(
        loaded: &LoadedLoopRuntime,
        backend: &B,
        overrides: &SourceOverrides,
    ) -> RusticolResult<Option<SourceBindingFailure>> {
        let mut workspace = OwnerWorkspace::<B>::new(loaded, backend)?;
        Ok(workspace
            .evaluate(source_record(loaded)?, backend, overrides)?
            .failure()
            .cloned())
    }

    pub(crate) fn require_backend(&self, backend: &B) -> RusticolResult<()> {
        let empty = backend.ingress_complex(&[], 1, 0, vec![0])?;
        B::concat(&[&self.carrier, &empty])?;
        Ok(())
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

    /// Re-evaluate a captured exact declaration in this owner's arithmetic.
    /// This does not merge with or publish the target owner's mutable state.
    pub fn reconstruct_snapshot(
        &self,
        backend: &B,
        original_owner: &Self,
        original: &SourceSnapshot<B::Batch>,
    ) -> RusticolResult<Arc<SourceSnapshot<B::Batch>>> {
        original_owner.require_snapshot(original)?;
        if !Arc::ptr_eq(&self.loaded, &original_owner.loaded) {
            return Err(invalid(
                "snapshot reconstruction requires the same saved program",
            ));
        }
        let mut workspace = self
            .workspace
            .lock()
            .map_err(|_| invalid("source scalar reconstruction lock poisoned"))?;
        self.require_backend(backend)?;
        let realization =
            workspace.evaluate(source_record(&self.loaded)?, backend, original.overrides())?;
        Ok(Arc::new(SourceSnapshot {
            realization,
            overrides: original.overrides.clone(),
            revision: original.revision,
            identity: self.identity.clone(),
        }))
    }

    /// Merge with one current exact revision, then publish all fields together.
    pub fn update(
        &self,
        backend: &B,
        changes: &SourceOverrides,
    ) -> RusticolResult<Arc<SourceSnapshot<B::Batch>>> {
        self.update_with_alternate(backend, changes, |_, failure| Err(failure.error()))
    }

    /// The alternate sees one complete candidate while this owner's writer is
    /// serialized. It must not reenter this owner. Any failure leaves the current
    /// exact revision untouched, including when every configured carrier fails.
    pub fn update_with_alternate(
        &self,
        backend: &B,
        changes: &SourceOverrides,
        alternate: impl FnOnce(&SourceOverrides, &SourceBindingFailure) -> RusticolResult<()>,
    ) -> RusticolResult<Arc<SourceSnapshot<B::Batch>>> {
        self.require_backend(backend)?;
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
        let realization = workspace.evaluate(source_record(&self.loaded)?, backend, &overrides)?;
        if let Some(failure) = realization.failure() {
            alternate(&overrides, failure)?;
        }
        let revision = previous
            .revision
            .checked_add(1)
            .ok_or_else(|| invalid("source scalar revision overflows"))?;
        let candidate = Arc::new(SourceSnapshot {
            realization,
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
        let (values, failures) = self.evaluate_partial(snapshot, mu_r2)?;
        if !failures.is_empty() {
            return Err(crate::RusticolError::evaluation(format!(
                "source scale evaluation failed: {failures:?}"
            )));
        }
        Ok(values)
    }

    /// Preserve numerical row failures while keeping invalid structure and
    /// foreign snapshots as errors. The caller retains original exact scales.
    pub fn evaluate_partial(
        &mut self,
        snapshot: &SourceSnapshot<B::Batch>,
        mu_r2: &B::Batch,
    ) -> RusticolResult<(B::Batch, Vec<LoopLaneFailure>)> {
        if !Arc::ptr_eq(&self.identity, &snapshot.identity) {
            return Err(invalid(
                "scale snapshot belongs to a different global state",
            ));
        }
        if B::columns(mu_r2) != 1 {
            return Err(invalid("scales require one mu_r2 column"));
        }
        let mut failures = Vec::new();
        let mu_r2 = finite_rows::<B>(mu_r2, "renormalization scales", &mut failures)?;
        // Even an empty evaluation checks carrier and precision compatibility.
        let scale = B::evaluate(&mut self.scale, &mu_r2)?;
        let scale = finite_rows::<B>(
            &scale,
            "positive real renormalization scales",
            &mut failures,
        )?;
        let arguments = B::gather(
            snapshot.arguments()?,
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
        let values = finite_rows::<B>(
            &values,
            "scale-dependent scalar values and active diagnostics",
            &mut failures,
        )?;
        Ok((
            B::select(&values, (0..self.output_count).map(|i| 2 * i).collect())?,
            failures,
        ))
    }
}
