// SPDX-License-Identifier: 0BSD

//! The stored observable columns of a plan v4 process, evaluated without
//! Python: colour- and spin-correlated and helicity-resolved virtuals.
//!
//! A process generated with its observable columns
//! (`loops/v2/native_observables.StoredObservables`) stores two column sets,
//! `correlated` (the inclusive Born metric `born` and every declared colour
//! history) and `resolved` (one column per Born helicity row, and at leading
//! colour per Born basis element), each as the seed records of its plan over
//! the process plan, its interference programs and its plan Born components,
//! with the spin-source slots of the trees. [`V2Runtime::evaluate_columns`]
//! evaluates a set exactly as the Python API's `StoredColumns.evaluate` does
//! (one executor of the set's plan with the process's tree program and the
//! set's columns, the companion's in the split selections, the columns'
//! interferences at the IR regulator 0 for a model with pole regulators)
//! and assembles every column as `observables._Assembly.result`, in the
//! carrier's arithmetic ([`Arith`]): bitwise the Python API's binary64
//! values, equal `Decimal` values in double-double and arbitrary precision.
//! A spin source replaces a declared vector leg's wavefunction natively (the
//! leg keeps its lowest helicity row, `StoredColumns._sources`).

use super::*;

/// The stored observables record's schema (`native_observables.SCHEMA`).
const OBSERVABLES_SCHEMA: u32 = 2;
/// The column-set kinds (`native_observables.KINDS`).
pub const COLUMN_SETS: [&str; 2] = ["correlated", "resolved"];
/// The formed column sets of the selections (lit-fc-hel (G5)): the
/// selection template ([`template::SelectionTemplate`]) and the latest
/// resolved selection's columns ([`template::resolved_subset`]).
const SELECTION_SETS: [&str; 2] = [template::TEMPLATE, "resolved-selection"];
/// The inclusive Born metric's history id (`observables.INCLUSIVE`).
pub const INCLUSIVE: &str = "born";

#[derive(Deserialize)]
struct ColumnRecord {
    key: serde_json::Value,
    history: Option<String>,
}

#[derive(Deserialize)]
struct SetRecord {
    columns: Vec<ColumnRecord>,
    lanes: Vec<(serde_json::Value, Orders, usize, usize)>,
    plan_components: Vec<Option<usize>>,
    companion_components: Option<Vec<Option<usize>>>,
}

#[derive(Deserialize)]
struct Sources {
    #[serde(default)]
    companion: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ObservablesRecord {
    schema: u32,
    rows: Vec<Vec<(usize, i32)>>,
    basis: usize,
    spin_classes: Vec<Vec<usize>>,
    resolution: String,
    sets: BTreeMap<String, SetRecord>,
    sources: Sources,
}

/// One stored column set: its record, its plan's overlay and interference
/// programs, and its runners (process and companion), built on first use.
struct ColumnSet {
    record: SetRecord,
    overlay: Vec<u8>,
    columns_json: Vec<u8>,
    companion_overlay: Option<Vec<u8>>,
    runner: Option<Runner>,
    twin: Option<Runner>,
}

/// The output columns of a set: per column its key, whether it is directed
/// (a history with a bra: complex outputs), and per sector its (direct,
/// swapped) lane columns.
struct ColumnView {
    directed: bool,
    lanes: Vec<(usize, usize)>,
}

/// The stored observable columns of a process.
pub(super) struct Observables {
    record: ObservablesRecord,
    sets: BTreeMap<String, ColumnSet>,
    /// The declared colour histories (id, directed), in the request's order.
    histories: Vec<(String, bool)>,
    /// The stored observables record as read (the selections' formation,
    /// [`template`]).
    raw: serde_json::Value,
    /// The selection template, formed on the first selection.
    template: Option<template::SelectionTemplate>,
    /// The kept (rows, basis elements) of the latest resolved selection's
    /// columns (the set `resolved-selection`).
    subset: Option<SubsetKey>,
}

/// The kept rows and basis elements of a resolved selection (None keeps
/// every one of that axis).
type SubsetKey = (Option<Vec<usize>>, Option<Vec<usize>>);

/// One leading-colour flow of the stored catalogue (`selection.Flow`): its
/// ID and its Born basis element (None: no Born amplitude of that flow).
#[derive(Clone, Debug, Deserialize)]
pub struct StoredFlow {
    pub id: String,
    pub basis: Option<usize>,
}

/// One evaluated column of one point: the Born per public squared Born
/// order and the virtual per sector (`[finite, 1/ε, 1/ε²]`, finite only
/// without poles), complex (a real column's imaginary parts are zero), and
/// whether the column is real (an undirected history).
#[derive(Clone, Debug)]
pub struct ColumnValues<N> {
    pub born: Vec<C<N>>,
    pub virtual_values: Vec<Vec<C<N>>>,
    pub real: bool,
}

/// One evaluated point: per requested column its values, or the failures.
pub type ColumnPoint<N> = Result<Vec<ColumnValues<N>>, Vec<String>>;

impl Observables {
    /// The stored observables of a process (`artifact.read_native_process`'s
    /// observable members), refused when the process stores none or its
    /// colour contraction requires the seed FFT (`FORCED_FFT_OBSERVABLES`).
    pub(super) fn read(
        verified: &VerifiedArtifact,
        artifact: &LoadedLoopV2Artifact,
    ) -> RusticolResult<Self> {
        if artifact.metadata.color_contraction == "fft" {
            return Err(RusticolError::compatibility(
                "observable columns are contracted directly (design §7.3: no \
                 correlated or resolved seed FFT); generate the slice with colour \
                 contraction 'auto' or 'direct'",
            ));
        }
        let mut members = read_loop_v2_observables(verified, artifact)?;
        let record = members.remove("observables.json").ok_or_else(|| {
            RusticolError::compatibility(
                "the stored process has no observable columns (an older build's \
                 artifact); regenerate the artifact",
            )
        })?;
        let raw: serde_json::Value = serde_json::from_slice(&record)
            .map_err(|e| invalid(format!("stored observables: {e}")))?;
        let record: ObservablesRecord = serde_json::from_slice(&record)
            .map_err(|e| invalid(format!("stored observables: {e}")))?;
        if record.schema != OBSERVABLES_SCHEMA {
            return Err(invalid(
                "unsupported stored observables schema; regenerate the artifact",
            ));
        }
        let mut sets = BTreeMap::new();
        let ObservablesRecord {
            schema,
            rows,
            basis,
            spin_classes,
            resolution,
            sets: set_records,
            sources,
        } = record;
        for (kind, set) in set_records {
            let member = |name: String| {
                members
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| invalid(format!("stored observables lack {name}")))
            };
            let overlay = member(format!("observables-{kind}-seeds.json"))?;
            let columns_json = member(format!("observables-{kind}-columns.json"))?;
            let companion_overlay = members
                .get(&format!("observables-{kind}-companion-seeds.json"))
                .cloned();
            sets.insert(
                kind,
                ColumnSet {
                    record: set,
                    overlay,
                    columns_json,
                    companion_overlay,
                    runner: None,
                    twin: None,
                },
            );
        }
        // The declared histories and whether each is directed (a history
        // with a bra, `observables.Column.directed`), from the request's
        // correlators (`artifact.correlators_from_record`).
        let histories = artifact
            .request
            .get("correlators")
            .and_then(|c| c.get("color_correlations"))
            .and_then(|h| h.as_array())
            .map(|items| {
                items
                    .iter()
                    .map(|item| {
                        let id = item.get("id").and_then(|v| v.as_str()).unwrap_or_default();
                        let bra = item
                            .get("bra")
                            .and_then(|v| v.as_array())
                            .is_some_and(|bra| !bra.is_empty());
                        (id.to_owned(), bra)
                    })
                    .collect()
            })
            .unwrap_or_default();
        Ok(Self {
            record: ObservablesRecord {
                schema,
                rows,
                basis,
                spin_classes,
                resolution,
                sets: BTreeMap::new(),
                sources,
            },
            sets,
            histories,
            raw,
            template: None,
            subset: None,
        })
    }

    /// The Born helicity rows of the resolved columns (per row its
    /// (0-based leg, helicity) pairs).
    pub(super) fn rows(&self) -> &[Vec<(usize, i32)>] {
        &self.record.rows
    }

    /// The Born colour basis elements.
    pub(super) fn basis(&self) -> usize {
        self.record.basis
    }

    pub(super) fn spin_classes(&self) -> &[Vec<usize>] {
        &self.record.spin_classes
    }

    pub(super) fn resolution(&self) -> &str {
        &self.record.resolution
    }

    /// The column keys of a set, in its stored order (`correlated`: the
    /// history ids; `resolved`: `[row, basis element]`).
    pub(super) fn keys(&self, kind: &str) -> RusticolResult<Vec<serde_json::Value>> {
        Ok(self
            .set(kind)?
            .record
            .columns
            .iter()
            .map(|c| c.key.clone())
            .collect())
    }

    fn set(&self, kind: &str) -> RusticolResult<&ColumnSet> {
        self.sets
            .get(kind)
            .ok_or_else(|| invalid(format!("the process stores no {kind} columns")))
    }

    /// Whether a column is directed (a declared history with a bra).
    fn directed(&self, history: Option<&str>) -> RusticolResult<bool> {
        match history {
            None | Some(INCLUSIVE) => Ok(false),
            Some(id) => self
                .histories
                .iter()
                .find(|(known, _)| known == id)
                .map(|(_, directed)| *directed)
                .ok_or_else(|| invalid(format!("undeclared colour correlation {id:?}"))),
        }
    }

    /// A point's native spin sources (`StoredColumns._sources`): per source
    /// leg its literal vector (exact rationals) and the Born rows the point
    /// keeps (each source leg at its lowest helicity row).
    fn sources<T: Native>(
        &self,
        spin: &BTreeMap<usize, [(Rational, Rational); 4]>,
        zero: &T,
    ) -> RusticolResult<PointSources<T>>
    where
        Complex<T>: Real + EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        let legs: Vec<usize> = spin.keys().copied().collect();
        if !self.record.spin_classes.contains(&legs) {
            return Err(invalid(format!(
                "spin legs {legs:?} are not a declared spin class"
            )));
        }
        let rows = &self.record.rows;
        let mut fixed = BTreeMap::new();
        for &leg in &legs {
            let lowest = rows
                .iter()
                .flat_map(|row| row.iter().filter(|(i, _)| *i + 1 == leg).map(|(_, h)| *h))
                .min()
                .ok_or_else(|| invalid(format!("leg {leg} has no helicity rows")))?;
            fixed.insert(leg, lowest);
        }
        let keep = rows
            .iter()
            .map(|row| {
                fixed.iter().all(|(leg, h)| {
                    row.iter()
                        .find(|(i, _)| *i + 1 == *leg)
                        .is_some_and(|(_, value)| value == h)
                })
            })
            .collect();
        let vectors = spin
            .iter()
            .map(|(leg, components)| {
                let lift = |(re, im): &(Rational, Rational)| {
                    Complex::new(zero.from_rational(re), zero.from_rational(im))
                };
                (
                    *leg,
                    [
                        lift(&components[0]),
                        lift(&components[1]),
                        lift(&components[2]),
                        lift(&components[3]),
                    ],
                )
            })
            .collect();
        Ok(PointSources {
            vectors,
            rows: Some(keep),
            basis: None,
        })
    }
}

impl V2Runtime {
    /// The leading-colour flows of the stored catalogue (`color_flows`; none
    /// at next-to-leading and full colour), in the tree runtime's order.
    pub fn colour_flows(&mut self) -> RusticolResult<Vec<StoredFlow>> {
        let observables = self.stored_observables()?;
        match observables.raw.get("color_flows") {
            None | Some(serde_json::Value::Null) => Ok(Vec::new()),
            Some(flows) => serde_json::from_value(flows.clone())
                .map_err(|e| invalid(format!("stored colour flows: {e}"))),
        }
    }

    /// Form the selection template (once) or the resolved selection's
    /// columns of `rows` and `basis` (replacing the previous ones) as a
    /// column set ([`template`]); returns the template's masks of the
    /// selection, or None for a resolved selection.
    fn formed_set(
        &mut self,
        resolved: bool,
        rows: Option<&BTreeSet<usize>>,
        basis: Option<&BTreeSet<usize>>,
    ) -> RusticolResult<Option<template::SelectionMasks>> {
        let plan = self.artifact.plan_json.clone();
        let companion_plan = self.artifact.scaleless_plan_json.clone();
        let observables = self.stored_observables()?;
        let base = observables.set("resolved")?;
        let stored_record =
            |set: &template::FormedColumns, base: &ColumnSet| -> RusticolResult<SetRecord> {
                Ok(SetRecord {
                    columns: set
                        .outputs
                        .iter()
                        .map(|column| ColumnRecord {
                            key: column.key.clone(),
                            history: None,
                        })
                        .collect(),
                    lanes: set
                        .lanes
                        .iter()
                        .map(|(c, sector, direct, swapped)| {
                            let orders: Orders = serde_json::from_value(sector.clone())
                                .map_err(|e| invalid(format!("a lane sector: {e}")))?;
                            Ok((set.outputs[*c].key.clone(), orders, *direct, *swapped))
                        })
                        .collect::<RusticolResult<_>>()?,
                    plan_components: base.record.plan_components.clone(),
                    companion_components: base.record.companion_components.clone(),
                })
            };
        let column_set = |set: &template::FormedColumns, base: &ColumnSet| {
            Ok::<_, RusticolError>(ColumnSet {
                record: stored_record(set, base)?,
                overlay: set.overlay.clone().into_bytes(),
                columns_json: set.columns.clone().into_bytes(),
                companion_overlay: set.companion_overlay.clone().map(String::into_bytes),
                runner: None,
                twin: None,
            })
        };
        // The record's JSON text: the formation reads it typed (its bulk
        // borrowed), not as the held values.
        let record = serde_json::to_vec(&observables.raw)
            .map_err(|e| invalid(format!("the observables record: {e}")))?;
        let records = template::ResolvedRecords {
            record: &record,
            overlay: &base.overlay,
            companion_overlay: base.companion_overlay.as_deref(),
            columns: &base.columns_json,
            plan: &plan,
            companion_plan: companion_plan.as_deref(),
        };
        if resolved {
            let key = (
                rows.map(|r| r.iter().copied().collect()),
                basis.map(|b| b.iter().copied().collect()),
            );
            if observables.subset.as_ref() != Some(&key) {
                let set = template::resolved_subset(&records, rows, basis)?;
                let formed = column_set(&set, base)?;
                observables
                    .sets
                    .insert(SELECTION_SETS[1].to_owned(), formed);
                observables.subset = Some(key);
            }
            return Ok(None);
        }
        if observables.template.is_none() {
            let formed = template::SelectionTemplate::form(&records)?;
            let set = column_set(&formed.set, base)?;
            observables.sets.insert(SELECTION_SETS[0].to_owned(), set);
            observables.template = Some(formed);
        }
        let formed = observables.template.as_ref().expect("the template");
        formed.masks(rows, basis).map(Some)
    }

    /// The Born-interfered selection of the Born rows `rows` and (leading
    /// colour) the basis elements `basis` (None keeps every one) in binary64
    /// (lit-fc-hel; the Python API's `evaluate(helicities=...,
    /// color_flows=...)` on a stored process): per point the selection's
    /// one column, the masks of one call on the selection template (only
    /// the kept rows' trees, links and lanes are formed: the executor's row
    /// liveness), the template columns of several row classes summed in
    /// their order. Points fail alone.
    pub fn evaluate_selection(
        &mut self,
        rows: Option<&BTreeSet<usize>>,
        basis: Option<&BTreeSet<usize>>,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
    ) -> RusticolResult<Vec<ColumnPoint<f64>>> {
        let masks = self.formed_set(false, rows, basis)?.expect("the masks");
        let points = self.columns_batch(
            SELECTION_SETS[0],
            momenta,
            scales,
            &[],
            mode,
            V2Precision::Double,
            &0.0,
            &Binary64,
            Some(&masks),
        )?;
        Ok(summed(points, &masks.wanted, &Binary64))
    }

    /// [`V2Runtime::evaluate_selection`] in double-double or arbitrary
    /// precision (the Python API's `Decimal` values).
    pub fn evaluate_selection_extended(
        &mut self,
        rows: Option<&BTreeSet<usize>>,
        basis: Option<&BTreeSet<usize>>,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
        precision: V2Precision,
    ) -> RusticolResult<Vec<ColumnPoint<Decimal>>> {
        let masks = self.formed_set(false, rows, basis)?.expect("the masks");
        let arith = DecimalArith::new(precision.working_digits());
        let points = match precision {
            V2Precision::Double => {
                return Err(invalid(
                    "binary64 selections are V2Runtime::evaluate_selection's",
                ));
            }
            V2Precision::DoubleDouble => self.columns_batch(
                SELECTION_SETS[0],
                momenta,
                scales,
                &[],
                mode,
                precision,
                &DoubleFloat::from(0.0),
                &arith,
                Some(&masks),
            )?,
            V2Precision::Arbitrary(_) => {
                let zero = Float::with_val(precision.bits()?, 0);
                self.columns_batch(
                    SELECTION_SETS[0],
                    momenta,
                    scales,
                    &[],
                    mode,
                    precision,
                    &zero,
                    &arith,
                    Some(&masks),
                )?
            }
        };
        Ok(summed(points, &masks.wanted, &arith))
    }

    /// The resolved columns of a resolved selection (the Python API's
    /// `evaluate_resolved(helicities=..., color_flows=...)`): per point the
    /// kept rows' (and leading-colour basis elements') columns only, each
    /// its stored resolved column's value; with their (row, colour
    /// component) keys ([`V2Runtime::resolved_keys`]'s numbering).
    #[allow(clippy::type_complexity)]
    pub fn evaluate_resolved_selection(
        &mut self,
        rows: Option<&BTreeSet<usize>>,
        basis: Option<&BTreeSet<usize>>,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        mode: V2Poles,
        precision: V2Precision,
    ) -> RusticolResult<(Vec<(usize, usize)>, ResolvedPoints)> {
        self.formed_set(true, rows, basis)?;
        let keys = self.set_keys(SELECTION_SETS[1])?;
        let points = match precision {
            V2Precision::Double => ResolvedPoints::Double(self.columns_batch(
                SELECTION_SETS[1],
                momenta,
                scales,
                &[],
                mode,
                precision,
                &0.0,
                &Binary64,
                None,
            )?),
            V2Precision::DoubleDouble => {
                let arith = DecimalArith::new(precision.working_digits());
                ResolvedPoints::Decimal(self.columns_batch(
                    SELECTION_SETS[1],
                    momenta,
                    scales,
                    &[],
                    mode,
                    precision,
                    &DoubleFloat::from(0.0),
                    &arith,
                    None,
                )?)
            }
            V2Precision::Arbitrary(_) => {
                let arith = DecimalArith::new(precision.working_digits());
                let zero = Float::with_val(precision.bits()?, 0);
                ResolvedPoints::Decimal(self.columns_batch(
                    SELECTION_SETS[1],
                    momenta,
                    scales,
                    &[],
                    mode,
                    precision,
                    &zero,
                    &arith,
                    None,
                )?)
            }
        };
        Ok((keys, points))
    }

    /// Per column of a set (stored order) its (row, colour component)
    /// indices on the stored rows (the key `[row, basis element]`).
    fn set_keys(&mut self, kind: &str) -> RusticolResult<Vec<(usize, usize)>> {
        let observables = self.stored_observables()?;
        let rows = observables.rows().to_vec();
        observables
            .keys(kind)?
            .iter()
            .map(|key| {
                let (row, colour): (Vec<(usize, i32)>, Option<usize>) =
                    serde_json::from_value(key.clone())
                        .map_err(|e| invalid(format!("a resolved column key: {e}")))?;
                let row = rows
                    .iter()
                    .position(|known| *known == row)
                    .ok_or_else(|| invalid("a resolved column of an unknown helicity row"))?;
                Ok((row, colour.unwrap_or(0)))
            })
            .collect()
    }

    /// The process's stored observables (read on first use).
    fn stored_observables(&mut self) -> RusticolResult<&mut Observables> {
        if self.observables.is_none() {
            let verified = VerifiedArtifact::open(&self.path)?;
            self.observables = Some(Observables::read(&verified, &self.artifact)?);
        }
        Ok(self.observables.as_mut().expect("the observables"))
    }

    /// The colour histories of the correlated columns, in their stored
    /// order (`born` first): the ids `evaluate_columns("correlated", ...)`
    /// returns.
    pub fn correlation_histories(&mut self) -> RusticolResult<Vec<String>> {
        if self.induced.is_some() {
            return self.induced_histories();
        }
        Ok(self
            .stored_observables()?
            .keys("correlated")?
            .iter()
            .map(|key| key.as_str().unwrap_or_default().to_owned())
            .collect())
    }

    /// The declared joint spin-correlation classes (1-based legs).
    pub fn spin_correlation_classes(&mut self) -> RusticolResult<Vec<Vec<usize>>> {
        if self.induced.is_some() {
            return self.induced_spin_classes();
        }
        Ok(self.stored_observables()?.spin_classes().to_vec())
    }

    /// The resolved columns' Born helicity rows (per row its (0-based leg,
    /// helicity) pairs), the Born colour basis elements and the resolution
    /// (`helicity`, or `helicity-colour` at leading colour, whose colour axis
    /// holds one entry per basis element).
    #[allow(clippy::type_complexity)]
    pub fn resolved_axes(&mut self) -> RusticolResult<(Vec<Vec<(usize, i32)>>, usize, String)> {
        if let Some(induced) = &self.induced {
            // A loop-induced process resolves its helicity rows with one
            // colour-contracted component (`induced_observables.CONTRACTED`).
            return Ok((self.induced_rows(), induced.size(), "helicity".to_owned()));
        }
        let observables = self.stored_observables()?;
        Ok((
            observables.rows().to_vec(),
            observables.basis(),
            observables.resolution().to_owned(),
        ))
    }

    /// Per resolved column (stored order) its Born helicity row and colour
    /// component indices (the component of a helicity resolution is 0):
    /// the column keys `[row, basis element]` on the stored rows.
    pub fn resolved_keys(&mut self) -> RusticolResult<Vec<(usize, usize)>> {
        if self.induced.is_some() {
            return Ok((0..self.induced_rows().len()).map(|row| (row, 0)).collect());
        }
        self.set_keys("resolved")
    }

    /// The squared Born orders of the observable outputs
    /// (`V2ObservablesRuntime._default_born_orders`: the sums of the tree
    /// program's Born amplitude orders, sorted) as indices of the declared
    /// Born orders' squared orders in the tree program.
    fn column_born_orders(&self) -> RusticolResult<Vec<Orders>> {
        let amplitudes = self
            .record
            .trees
            ._rest
            .get("born_orders")
            .ok_or_else(|| invalid("the native record has no Born amplitude orders"))?;
        let amplitudes: Vec<Orders> = serde_json::from_value(amplitudes.clone())
            .map_err(|e| invalid(format!("Born amplitude orders: {e}")))?;
        let mut squared: Vec<Orders> = Vec::new();
        for a in &amplitudes {
            for b in &amplitudes {
                let mut sum: BTreeMap<String, i64> = BTreeMap::new();
                for (name, power) in a.iter().chain(b) {
                    *sum.entry(name.clone()).or_default() += power;
                }
                let order = canonical(&sum.into_iter().collect::<Vec<_>>());
                if !squared.contains(&order) {
                    squared.push(order);
                }
            }
        }
        squared.sort();
        Ok(squared)
    }

    /// The public sectors of the observable outputs: the record's sectors
    /// in their order (`V2ObservablesRuntime._sectors`).
    pub fn column_sectors(&self) -> Vec<Orders> {
        self.record
            .sectors
            .iter()
            .map(|(sector, _)| canonical(sector))
            .collect()
    }

    /// The squared Born orders of the observable outputs (their Born axis).
    pub fn column_born_axis(&self) -> RusticolResult<Vec<Orders>> {
        if self.induced.is_some() {
            // No Born (RULED Q36).
            return Ok(Vec::new());
        }
        self.column_born_orders()
    }

    /// Evaluate the columns of the stored set `kind` (`correlated` or
    /// `resolved`) in binary64: per point every column of the set, in its
    /// stored order. `spins` is empty (no spin sources) or holds per point
    /// the spin-source vectors by 1-based leg (exact rationals, one declared
    /// spin class). Points fail alone.
    pub fn evaluate_columns(
        &mut self,
        kind: &str,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        spins: &[BTreeMap<usize, [(Rational, Rational); 4]>],
        mode: V2Poles,
    ) -> RusticolResult<Vec<ColumnPoint<f64>>> {
        if self.induced.is_some() {
            return self.induced_columns(
                kind,
                momenta,
                scales,
                spins,
                mode,
                V2Precision::Double,
                &0.0,
                &Binary64,
            );
        }
        self.columns_batch(
            kind,
            momenta,
            scales,
            spins,
            mode,
            V2Precision::Double,
            &0.0,
            &Binary64,
            None,
        )
    }

    /// [`V2Runtime::evaluate_columns`] in double-double or arbitrary
    /// precision, assembled in Python's `Decimal` arithmetic at the working
    /// digits (equal values to the Python API's).
    pub fn evaluate_columns_extended(
        &mut self,
        kind: &str,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        spins: &[BTreeMap<usize, [(Rational, Rational); 4]>],
        mode: V2Poles,
        precision: V2Precision,
    ) -> RusticolResult<Vec<ColumnPoint<Decimal>>> {
        let arith = DecimalArith::new(precision.working_digits());
        if self.induced.is_some() {
            return match precision {
                V2Precision::Double => Err(invalid(
                    "binary64 column evaluations are V2Runtime::evaluate_columns'",
                )),
                V2Precision::DoubleDouble => self.induced_columns(
                    kind,
                    momenta,
                    scales,
                    spins,
                    mode,
                    precision,
                    &DoubleFloat::from(0.0),
                    &arith,
                ),
                V2Precision::Arbitrary(_) => {
                    let zero = Float::with_val(precision.bits()?, 0);
                    self.induced_columns(
                        kind, momenta, scales, spins, mode, precision, &zero, &arith,
                    )
                }
            };
        }
        match precision {
            V2Precision::Double => Err(invalid(
                "binary64 column evaluations are V2Runtime::evaluate_columns'",
            )),
            V2Precision::DoubleDouble => self.columns_batch(
                kind,
                momenta,
                scales,
                spins,
                mode,
                precision,
                &DoubleFloat::from(0.0),
                &arith,
                None,
            ),
            V2Precision::Arbitrary(_) => {
                let zero = Float::with_val(precision.bits()?, 0);
                self.columns_batch(
                    kind, momenta, scales, spins, mode, precision, &zero, &arith, None,
                )
            }
        }
    }

    /// The runners of a column set (process and, in the split selections,
    /// companion), built on first use (`StoredColumns._executor`): the
    /// set's plan, the process's (companion's) tree program with the set's
    /// plan Born components and columns (the companion's: its spin-source
    /// slots only). No seed FFT (`StoredColumns` attaches none).
    fn column_runners(&mut self, kind: &str, twin: bool) -> RusticolResult<()> {
        let threads = self.threads;
        let process = self.process_inputs.clone();
        let companion = self.scaleless_inputs.clone();
        let base = self.artifact.plan_json.clone();
        let companion_base = self.artifact.scaleless_plan_json.clone();
        let observables = self.stored_observables()?;
        let sources = observables.record.sources.companion.clone();
        let set = observables
            .sets
            .get_mut(kind)
            .ok_or_else(|| invalid(format!("the process stores no {kind} columns")))?;
        if !twin && set.runner.is_none() {
            let plan = Arc::new(LoopProgramV4::load_overlaid(
                &base,
                &set.overlay,
                rusticol_core::loop_v2::plan::FormPolicy::Rule,
            )?);
            let mut trees = TreeProgram::load_with_components(
                &process.trees_json,
                &process.evaluators,
                Some(&process.planes),
                &plan,
                Some(set.record.plan_components.clone()),
            )?;
            trees.set_columns(&set.columns_json)?;
            set.runner = Some(Runner::new(plan, trees, None, threads)?);
        }
        if twin && set.twin.is_none() {
            let (Some(overlay), Some(stored), Some(base)) =
                (&set.companion_overlay, companion, companion_base)
            else {
                return Err(invalid("the column set stores no companion seeds"));
            };
            let plan = Arc::new(LoopProgramV4::load_overlaid(
                &base,
                overlay,
                rusticol_core::loop_v2::plan::FormPolicy::Rule,
            )?);
            let mut trees = TreeProgram::load_with_components(
                &stored.trees_json,
                &stored.evaluators,
                Some(&stored.planes),
                &plan,
                set.record.companion_components.clone(),
            )?;
            let columns = serde_json::json!({
                "cells": [], "programs": [], "columns": [],
                "trees": sources.unwrap_or_else(|| serde_json::json!([])),
                "counterterms": [],
            });
            trees
                .set_columns(&serde_json::to_vec(&columns).map_err(|e| invalid(e.to_string()))?)?;
            set.twin = Some(Runner::new(plan, trees, None, threads)?);
        }
        Ok(())
    }

    /// One batch of a column set: the native inputs and interferences with
    /// the spin sources, the executor run (and the companion's in a split
    /// selection), and every column assembled (`StoredColumns._evaluations`
    /// and `observables._Assembly.result`).
    #[allow(clippy::too_many_arguments)]
    fn columns_batch<T: Native, A: Lift<T>>(
        &mut self,
        kind: &str,
        momenta: &[Vec<Rational>],
        scales: &[Rational],
        spins: &[BTreeMap<usize, [(Rational, Rational); 4]>],
        mode: V2Poles,
        precision: V2Precision,
        zero: &T,
        arith: &A,
        masks: Option<&template::SelectionMasks>,
    ) -> RusticolResult<Vec<ColumnPoint<A::N>>>
    where
        Complex<T>: Real + EvaluationDomain,
        TreeRuntime<T>: Send,
    {
        self.check_batch(momenta, scales)?;
        if !COLUMN_SETS.contains(&kind) && !SELECTION_SETS.contains(&kind) {
            return Err(invalid(format!("unknown column set {kind:?}")));
        }
        if !spins.is_empty() && spins.len() != momenta.len() {
            return Err(invalid("one spin source assignment per point"));
        }
        if masks.is_some() && !spins.is_empty() {
            return Err(invalid("a selection takes no spin sources"));
        }
        // The output columns, their lanes per sector and the born orders.
        let sectors = self.column_sectors();
        let born_axis = self.column_born_orders()?;
        let squared: Vec<Orders> = self
            .record
            .trees
            .squared_orders
            .iter()
            .map(|o| canonical(o))
            .collect();
        let born_of: Vec<Option<usize>> = born_axis
            .iter()
            .map(|order| squared.iter().position(|o| o == order))
            .collect();
        let outputs: Vec<(String, usize, i32)> =
            self.record
                .trees
                .counterterm_outputs
                .iter()
                .map(|(component, sector, power)| {
                    let wanted = canonical(sector);
                    let s = sectors.iter().position(|o| *o == wanted).ok_or_else(|| {
                        invalid("a counterterm output's sector is not in the record")
                    })?;
                    Ok((component.clone(), s, *power))
                })
                .collect::<RusticolResult<_>>()?;
        let observables = self.stored_observables()?;
        let set = observables.set(kind)?;
        let views = set
            .record
            .columns
            .iter()
            .map(|column| {
                let lanes = sectors
                    .iter()
                    .map(|sector| {
                        set.record
                            .lanes
                            .iter()
                            .find(|(key, s, _, _)| *key == column.key && canonical(s) == *sector)
                            .map(|(_, _, direct, swapped)| (*direct, *swapped))
                            .ok_or_else(|| invalid("a column has no lanes in a sector"))
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                Ok(ColumnView {
                    directed: observables.directed(column.history.as_deref())?,
                    lanes,
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let has_twin = set.companion_overlay.is_some();
        let sources = if spins.is_empty() {
            None
        } else {
            Some(
                spins
                    .iter()
                    .map(|spin| {
                        if spin.is_empty() {
                            Ok(PointSources {
                                vectors: Vec::new(),
                                rows: None,
                                basis: None,
                            })
                        } else {
                            observables.sources(spin, zero)
                        }
                    })
                    .collect::<RusticolResult<Vec<_>>>()?,
            )
        };
        if momenta.is_empty() {
            return Ok(Vec::new());
        }
        let poles = mode.poles();
        let bits = precision.bits()?;
        let lanes = precision.lanes(momenta.len());
        let incoming = self.incoming(momenta);
        let values = self.parameter_values()?;
        let regulated = self.regulated_values(&values);
        self.column_runners(kind, false)?;
        let twin_runs = mode.split() && has_twin;
        if twin_runs {
            self.column_runners(kind, true)?;
        }
        self.backend(precision)?;
        let options = V2Options {
            poles: mode.lanes(),
            ..V2Options::default()
        };
        let observables = self.observables.as_mut().expect("the observables");
        let set = observables.sets.get_mut(kind).expect("the set");
        let runner = set.runner.as_mut().expect("the set's runner");
        let call = masks.map(|masks| masks.call(0));
        let (bound, inputs, live) = match &call {
            // A selection's masks on the template (lit-fc-hel (G5)).
            Some(call) => runner.fill_masked(
                precision, lanes, &values, scales, &incoming, poles, call, &options, zero,
            )?,
            None => {
                let (bound, inputs) = runner.fill(
                    precision,
                    lanes,
                    &values,
                    scales,
                    &incoming,
                    poles,
                    sources.as_deref(),
                    zero,
                )?;
                (bound, inputs, RowLiveness::none())
            }
        };
        let columns = inputs
            .columns
            .ok_or_else(|| invalid("the stored tree program forms no observable columns"))?;
        let batch = v2_batch(
            incoming.clone(),
            scales,
            bound,
            inputs.block_rows,
            inputs.born,
        );
        let backend = self.backends.get_mut(&precision).expect("the backend");
        let results = match &call {
            Some(call) => {
                runner.evaluate_masked(lanes, &batch, zero, backend, options, call, &live)?
            }
            None => runner.evaluate(lanes, &batch, zero, backend, options)?,
        };
        // A model with pole regulators: the split selections also form every
        // column's interferences with the IR regulator at 0 (one more pass
        // of the tree program, no loop dressing or reduction).
        let columns_uv = match (&regulated, mode.split()) {
            (Some(regulated), true) => {
                // A selection: its rows and basis elements (the Python
                // binding's `tree_columns(rows=, basis=)`, the whole tree
                // program).
                let masked = call.as_ref().and_then(|call| {
                    template::CallMasks {
                        rows: call.rows.clone(),
                        basis: call.basis.clone(),
                        ..Default::default()
                    }
                    .sources::<T>(None, incoming.len())
                });
                let (_, inputs) = runner.fill(
                    precision,
                    lanes,
                    regulated,
                    scales,
                    &incoming,
                    true,
                    masked.as_deref().or(sources.as_deref()),
                    zero,
                )?;
                Some(inputs.columns.ok_or_else(|| {
                    invalid("the stored tree program forms no observable columns")
                })?)
            }
            _ => None,
        };
        let twin_results = if twin_runs {
            let twin = set.twin.as_mut().expect("the set's companion");
            let twin_call = masks.map(|masks| masks.call(1));
            let (bound, inputs, live) = match &twin_call {
                Some(call) => twin.fill_masked(
                    precision, lanes, &values, scales, &incoming, poles, call, &options, zero,
                )?,
                None => {
                    let (bound, inputs) = twin.fill(
                        precision,
                        lanes,
                        &values,
                        scales,
                        &incoming,
                        poles,
                        sources.as_deref(),
                        zero,
                    )?;
                    (bound, inputs, RowLiveness::none())
                }
            };
            let batch = v2_batch(incoming, scales, bound, inputs.block_rows, inputs.born);
            let twin_results = match &twin_call {
                Some(call) => {
                    twin.evaluate_masked(lanes, &batch, zero, backend, options, call, &live)?
                }
                None => twin.evaluate(lanes, &batch, zero, backend, options)?,
            };
            if twin_results.len() != results.len() {
                return Err(invalid("the companion evaluated another batch"));
            }
            Some(twin_results)
        } else {
            None
        };
        let width = if poles { 3 } else { 1 };
        let lane_width = mode.lanes().width();
        let factor = loop_factor(arith, self.record.average);
        let inverse = arith.div(&arith.int(1), &arith.real_int(self.record.average));
        let zero_c = C::new(arith.zero(), arith.zero());
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
            // Per lane column its summed Laurent (`omega`), and in a split
            // selection the combined and UV sums (`_Assembly.result`), over
            // the executor's lanes in order, the companion's after them
            // (vanishing but for their UV residues).
            let mut omega: BTreeMap<usize, Vec<C<A::N>>> = BTreeMap::new();
            let mut combined: BTreeMap<usize, Vec<C<A::N>>> = BTreeMap::new();
            let mut ultraviolet: BTreeMap<usize, Vec<C<A::N>>> = BTreeMap::new();
            let mut lanes_of: Vec<(&LaneKey, &[Complex<T>], bool)> = result
                .families
                .iter()
                .map(|(key, value)| (key, value.as_slice(), false))
                .collect();
            if let Some(twin) = twin {
                lanes_of.extend(
                    twin.families
                        .iter()
                        .map(|(key, value)| (key, value.as_slice(), true)),
                );
            }
            for (key, value, vanishing) in lanes_of {
                if value.len() != lane_width {
                    return Err(invalid("an executor lane of another width"));
                }
                let lifted = value
                    .iter()
                    .map(|z| arith.complex(z, bits))
                    .collect::<RusticolResult<Vec<_>>>()?;
                let k = key.column;
                if mode.split() {
                    // `set_lanes`: a vanishing lane keeps its UV residue only.
                    let whole = combined.entry(k).or_insert_with(|| vec![zero_c.clone(); 3]);
                    for (i, term) in whole.iter_mut().enumerate() {
                        let value = if vanishing { &zero_c } else { &lifted[i] };
                        *term = cadd(arith, term, value);
                    }
                    let uv = ultraviolet
                        .entry(k)
                        .or_insert_with(|| vec![zero_c.clone(); 3]);
                    let finite = if vanishing { &zero_c } else { &lifted[0] };
                    uv[0] = cadd(arith, &uv[0], finite);
                    uv[1] = cadd(arith, &uv[1], &lifted[3]);
                } else {
                    let terms = omega
                        .entry(k)
                        .or_insert_with(|| vec![zero_c.clone(); width]);
                    for (i, term) in terms.iter_mut().enumerate() {
                        *term = cadd(arith, term, &lifted[i]);
                    }
                }
            }
            // `_locals`: per column the Born per squared order and the local
            // components per (component, sector, power).
            #[allow(clippy::type_complexity)]
            let locals = |values: &[Complex<T>]| -> RusticolResult<(
                Vec<C<A::N>>,
                BTreeMap<(&str, usize, i32), C<A::N>>,
            )> {
                let lifted = values
                    .iter()
                    .map(|z| arith.complex(z, bits))
                    .collect::<RusticolResult<Vec<_>>>()?;
                let born = lifted[..squared.len()].to_vec();
                let mut local = BTreeMap::new();
                for (k, (component, s, power)) in outputs.iter().enumerate() {
                    if !poles && *power != LAURENT_POWERS[0] {
                        continue;
                    }
                    local.insert(
                        (component.as_str(), *s, *power),
                        lifted[squared.len() + k].clone(),
                    );
                }
                Ok((born, local))
            };
            let mut point = Vec::with_capacity(views.len());
            for (c, view) in views.iter().enumerate() {
                let (born, local) = locals(&columns[index][c])?;
                let local_uv = columns_uv
                    .as_ref()
                    .map(|columns| locals(&columns[index][c]).map(|(_, local)| local))
                    .transpose()?;
                let real = !view.directed;
                // factor * x + (factor * y).conjugate()
                let bare_of = |lanes: &BTreeMap<usize, Vec<C<A::N>>>,
                               direct: usize,
                               swapped: usize|
                 -> Vec<C<A::N>> {
                    let none = vec![zero_c.clone(); width];
                    let a = lanes.get(&direct).unwrap_or(&none);
                    let b = lanes.get(&swapped).unwrap_or(&none);
                    a.iter()
                        .zip(b)
                        .map(|(x, y)| {
                            cadd(
                                arith,
                                &cmul(arith, &factor, x),
                                &conj(arith, &cmul(arith, &factor, y)),
                            )
                        })
                        .collect()
                };
                let laurent = |values: Vec<C<A::N>>| -> Vec<C<A::N>> {
                    if real {
                        values
                            .into_iter()
                            .map(|z| C::new(z.re, arith.zero()))
                            .collect()
                    } else {
                        values
                    }
                };
                let component = |table: &BTreeMap<(&str, usize, i32), C<A::N>>,
                                 name: &str,
                                 s: usize,
                                 count: usize|
                 -> Vec<C<A::N>> {
                    LAURENT_POWERS[..count]
                        .iter()
                        .map(|power| {
                            table
                                .get(&(name, s, *power))
                                .cloned()
                                .unwrap_or_else(|| zero_c.clone())
                        })
                        .collect()
                };
                let mut virtual_values = Vec::with_capacity(sectors.len());
                for (s, (direct, swapped)) in view.lanes.iter().enumerate() {
                    let bare = bare_of(
                        if mode.split() { &combined } else { &omega },
                        *direct,
                        *swapped,
                    );
                    let r2 = component(&local, "r2", s, width);
                    let uv_local = component(&local, "uv_local", s, width);
                    let fields = component(&local, FIELD_COMPONENT, s, width);
                    if mode.split() {
                        let parts = |values: Vec<C<A::N>>| -> [C<A::N>; 3] {
                            let mut values = laurent(values).into_iter();
                            [
                                values.next().expect("finite"),
                                values.next().expect("single pole"),
                                values.next().expect("double pole"),
                            ]
                        };
                        let bare_uv = parts(bare_of(&ultraviolet, *direct, *swapped));
                        let counterterms_uv = local_uv.as_ref().map(|table| {
                            (
                                parts(component(table, "r2", s, 3)),
                                parts(component(table, "uv_local", s, 3)),
                                parts(component(table, FIELD_COMPONENT, s, 3)),
                            )
                        });
                        let (bare, r2, uv_local, fields) =
                            (parts(bare), parts(r2), parts(uv_local), parts(fields));
                        // The split is coefficient-wise: on the real and the
                        // imaginary parts alike.
                        let side = |pick: fn(&C<A::N>) -> A::N| {
                            let three = |v: &[C<A::N>; 3]| [pick(&v[0]), pick(&v[1]), pick(&v[2])];
                            split_virtual(
                                arith,
                                mode,
                                three(&bare),
                                three(&bare_uv),
                                three(&r2),
                                three(&uv_local),
                                three(&fields),
                                counterterms_uv
                                    .as_ref()
                                    .map(|(a, b, c)| (three(a), three(b), three(c))),
                            )
                        };
                        // A real column splits its real parts only (its
                        // Laurent values are `value.real`).
                        let re = side(|z| z.re.clone());
                        let im = if real {
                            [arith.zero(), arith.zero(), arith.zero()]
                        } else {
                            side(|z| z.im.clone())
                        };
                        virtual_values.push(
                            re.into_iter()
                                .zip(im)
                                .map(|(re, im)| C::new(re, im))
                                .collect(),
                        );
                        continue;
                    }
                    let values: Vec<C<A::N>> = (0..width)
                        .map(|k| {
                            let uv = cadd(arith, &uv_local[k], &fields[k]);
                            cadd(arith, &cadd(arith, &bare[k], &uv), &r2[k])
                        })
                        .collect();
                    virtual_values.push(laurent(values));
                }
                // The Born: value.real * inverse, or value * inverse.
                let born_values = born_of
                    .iter()
                    .map(|found| match found {
                        None => zero_c.clone(),
                        Some(k) => {
                            if real {
                                C::new(arith.mul(&born[*k].re, &inverse), arith.zero())
                            } else {
                                arith.times_real(&born[*k], &inverse)
                            }
                        }
                    })
                    .collect();
                point.push(ColumnValues {
                    born: born_values,
                    virtual_values,
                    real,
                });
            }
            out.push(Ok(point));
        }
        Ok(out)
    }
}

/// The points of a resolved selection ([`V2Runtime::evaluate_resolved_selection`])
/// in binary64 or in `Decimal` (double-double and arbitrary precision).
pub enum ResolvedPoints {
    Double(Vec<ColumnPoint<f64>>),
    Decimal(Vec<ColumnPoint<Decimal>>),
}

/// A selection's one output per point: the template columns `wanted` (row
/// classes with a kept row) summed in their order (`observables.
/// _sum_results`: Born, then every sector's Laurent coefficients, left to
/// right).
fn summed<A: Arith>(
    points: Vec<ColumnPoint<A::N>>,
    wanted: &[usize],
    arith: &A,
) -> Vec<ColumnPoint<A::N>> {
    points
        .into_iter()
        .map(|point| {
            point.map(|columns| {
                let mut picked = wanted.iter().map(|&c| columns[c].clone());
                let mut total = picked.next().expect("a selection keeps a template column");
                for other in picked {
                    for (a, b) in total.born.iter_mut().zip(&other.born) {
                        *a = cadd(arith, a, b);
                    }
                    for (sector, extra) in
                        total.virtual_values.iter_mut().zip(&other.virtual_values)
                    {
                        for (a, b) in sector.iter_mut().zip(extra) {
                            *a = cadd(arith, a, b);
                        }
                    }
                }
                vec![total]
            })
        })
        .collect()
}
