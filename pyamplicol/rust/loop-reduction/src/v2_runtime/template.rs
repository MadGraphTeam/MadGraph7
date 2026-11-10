// SPDX-License-Identifier: 0BSD

//! The evaluation-time selections of a stored Born-interfered process, one
//! formation for the Python API and the native SDKs (lit-fc-hel (G5);
//! DESIGN §8.5 "Full-colour selection").
//!
//! A stored process serves every helicity (and, at leading colour, colour
//! flow) selection with one column set, its selection template
//! ([`SelectionTemplate`], MF-MASKS), formed once from its stored resolved
//! records without regeneration and without any Symbolica operation (JSON
//! over the records): rows whose resolved columns read the same seed-weight
//! column per (colour component, sector) form a row class (one in practice);
//! per row class and sector one seed column whose weights are every colour
//! component's seed-weight column concatenated in seed-weight-column order,
//! and one interference column of the class's rows. The colour components
//! are the Born basis elements at leading colour and the one contracted
//! component at next-to-leading and full colour (the resolved columns'
//! `basis` is null there), so the same formation serves every accuracy and
//! forms exactly the leading-colour template it formed before. A selection
//! is then the masks of one call ([`SelectionTemplate::masks`]): the kept
//! Born rows, the kept bra basis elements (leading colour), the seeds' kept
//! plan Born components and the families with a kept seed term. A resolved
//! selection reads the stored resolved columns of its kept rows (and basis
//! elements) only ([`resolved_subset`]), so an unselected row's column is
//! never formed. The Python API (`loops/v2/native_observables.py`, through
//! the PyO3 binding) and the SDK runtime ([`V2Runtime`]) call these
//! functions, so their binary64 values are bitwise equal by construction.

use super::*;
use serde_json::value::RawValue;
use serde_json::{Value, json};
use std::collections::BTreeSet;

/// A row's seed-weight column per (colour component, sector, plan): rows
/// with equal ones form a row class.
type Signature = BTreeMap<(Option<usize>, usize, usize), usize>;

/// The key prefix of the template's columns (one per row class).
pub const TEMPLATE: &str = "selection-template";

/// The stored records a formation reads: the observables record
/// (`observables.json`, its JSON text: read typed, its bulk borrowed rather
/// than held as JSON values), the resolved set's seed records (and its
/// companion's), its interference programs, and the process (companion)
/// plan (JSON or the stored binary plan).
pub struct ResolvedRecords<'a> {
    pub record: &'a [u8],
    pub overlay: &'a [u8],
    pub companion_overlay: Option<&'a [u8]>,
    pub columns: &'a [u8],
    pub plan: &'a [u8],
    pub companion_plan: Option<&'a [u8]>,
}

/// One output column of a formed set: its key (JSON) and the Born rows and
/// basis elements it keeps (None: every one).
#[derive(Clone, Debug, PartialEq)]
pub struct FormedColumn {
    pub key: Value,
    pub rows: Option<Vec<usize>>,
    pub basis: Option<Vec<usize>>,
}

/// A formed column set: its seed records over the process (companion) plan
/// (`native_observables.ObservableSet`), its interference programs, its
/// output columns and per (output column, sector) its (direct, swapped)
/// seed columns, the sector as stored.
#[derive(Clone, Debug)]
pub struct FormedColumns {
    pub overlay: String,
    pub companion_overlay: Option<String>,
    pub columns: String,
    pub outputs: Vec<FormedColumn>,
    pub lanes: Vec<(usize, Value, usize, usize)>,
}

/// The masks of one call on one plan (`V2Executor.evaluate`'s `rows`,
/// `basis`, `components` and `families`; None keeps every one): the kept
/// Born helicity `rows` seed and interfere (every other row's Born
/// components vanish, as a spin source's fixed row does); the observable
/// columns' interferences keep the bra Born `basis` elements only; the
/// seeds' bra keeps the plan Born `components` only (the others read zero);
/// only the plan `families` kept are dressed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CallMasks {
    pub rows: Option<Vec<bool>>,
    pub basis: Option<Vec<bool>>,
    pub components: Option<Vec<bool>>,
    pub families: Option<Vec<bool>>,
}

impl CallMasks {
    pub fn is_empty(&self) -> bool {
        self.rows.is_none()
            && self.basis.is_none()
            && self.components.is_none()
            && self.families.is_none()
    }

    /// Each point's sources with the kept rows (and the point's own) and the
    /// kept basis elements; points without sources get an empty one.
    pub fn sources<T>(
        &self,
        sources: Option<Vec<PointSources<T>>>,
        points: usize,
    ) -> Option<Vec<PointSources<T>>> {
        if self.rows.is_none() && self.basis.is_none() {
            return sources;
        }
        let mut sources = sources.unwrap_or_else(|| {
            (0..points)
                .map(|_| PointSources {
                    vectors: Vec::new(),
                    rows: None,
                    basis: None,
                })
                .collect()
        });
        for point in &mut sources {
            if let Some(kept) = &self.rows {
                point.rows = Some(match point.rows.take() {
                    Some(own) => own.iter().zip(kept).map(|(a, b)| *a && *b).collect(),
                    None => kept.clone(),
                });
            }
            if self.basis.is_some() {
                point.basis.clone_from(&self.basis);
            }
        }
        Some(sources)
    }

    /// Zero the plan Born components the seeds' bra does not keep (`born`:
    /// per point `rows x components`).
    pub fn mask_born<T: Real>(&self, born: &mut [Vec<Complex<T>>], zero: &T) -> RusticolResult<()> {
        let Some(kept) = &self.components else {
            return Ok(());
        };
        for point in born {
            if kept.is_empty() || point.len() % kept.len() != 0 {
                return Err(invalid("one component flag per plan Born component"));
            }
            for row in point.chunks_exact_mut(kept.len()) {
                for (value, keep) in row.iter_mut().zip(kept) {
                    if !keep {
                        *value = Complex::new(zero.clone(), zero.clone());
                    }
                }
            }
        }
        Ok(())
    }
}

/// One selection's masks on a template ([`SelectionTemplate::masks`]): the
/// kept rows and basis elements, per plan (process, companion) the seeds'
/// kept plan Born components and the dressed families, and the template
/// columns (row classes) with a kept row.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectionMasks {
    pub rows: Option<Vec<bool>>,
    pub basis: Option<Vec<bool>>,
    pub components: Vec<Option<Vec<bool>>>,
    pub families: Vec<Option<Vec<bool>>>,
    pub wanted: Vec<usize>,
}

impl SelectionMasks {
    /// The masks of plan `plan`'s call (0 the process, 1 the companion,
    /// whose tree program has no columns: no basis mask).
    pub fn call(&self, plan: usize) -> CallMasks {
        CallMasks {
            rows: self.rows.clone(),
            basis: if plan == 0 { self.basis.clone() } else { None },
            components: self.components.get(plan).cloned().flatten(),
            families: self.families.get(plan).cloned().flatten(),
        }
    }
}

/// The selection template of a stored process (module docs).
#[derive(Clone, Debug)]
pub struct SelectionTemplate {
    pub set: FormedColumns,
    /// The row classes (their Born rows, ascending), one template column
    /// each.
    pub classes: Vec<Vec<usize>>,
    rows: usize,
    basis: usize,
    /// The resolved colour axis: the basis elements at leading colour, the
    /// contracted component (None) otherwise.
    axis: Vec<Option<usize>>,
    /// Per plan: the plan Born components of each colour component's seed
    /// weights, the components each seed class reads, the plan's Born
    /// component count, and per family the seed classes its leaves read
    /// (None: a forward or merged program, which reads every family).
    basis_components: Vec<BTreeMap<Option<usize>, BTreeSet<usize>>>,
    class_components: Vec<Vec<BTreeSet<usize>>>,
    components: Vec<usize>,
    family_classes: Vec<Option<Vec<BTreeSet<usize>>>>,
}

/// The records a formation reads of a plan (its seed classes and keys, its
/// forward or merged program, its families' leaf seeds): typed, so the rest
/// of the plan (steps, contents, edges, ...) is skipped while reading
/// rather than held as JSON values (the formation's peak memory).
#[derive(serde::Deserialize)]
struct PlanView {
    #[serde(default)]
    seed_classes: Option<Box<RawValue>>,
    #[serde(default)]
    seeds: Option<Box<RawValue>>,
    #[serde(default)]
    forward: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    merged: Option<serde::de::IgnoredAny>,
    #[serde(default)]
    families: Vec<FamilyView>,
}

#[derive(serde::Deserialize)]
struct FamilyView {
    leaves: Vec<LeafView>,
}

#[derive(serde::Deserialize)]
struct LeafView {
    seed: usize,
}

impl PlanView {
    fn seed_classes(&self) -> RusticolResult<&RawValue> {
        self.seed_classes
            .as_deref()
            .ok_or_else(|| invalid("a stored record lacks \"seed_classes\""))
    }

    fn seeds(&self) -> RusticolResult<&RawValue> {
        self.seeds
            .as_deref()
            .ok_or_else(|| invalid("a stored record lacks \"seeds\""))
    }
}

/// A plan's [`PlanView`] (a stored binary plan member is read as its JSON).
fn plan_view(bytes: &[u8]) -> RusticolResult<PlanView> {
    use rusticol_core::loop_v2::plan::binary::{PlanModel, is_binary};
    let read =
        |bytes: &[u8]| serde_json::from_slice(bytes).map_err(|e| invalid(format!("plan: {e}")));
    if is_binary(bytes) {
        return read(&PlanModel::from_binary(bytes)?.json());
    }
    read(bytes)
}

/// The part of the observables record a formation reads, typed and
/// borrowed: the Born rows and basis count, and the resolved set's columns
/// and lanes (their keys, rows and sectors as the stored JSON text).
#[derive(serde::Deserialize)]
struct RecordView<'a> {
    #[serde(borrow)]
    rows: Vec<&'a RawValue>,
    basis: usize,
    #[serde(borrow)]
    sets: SetsView<'a>,
}

#[derive(serde::Deserialize)]
struct SetsView<'a> {
    #[serde(borrow)]
    resolved: ResolvedView<'a>,
}

#[derive(serde::Deserialize)]
struct ResolvedView<'a> {
    #[serde(borrow)]
    columns: Vec<ItemView<'a>>,
    /// (key, sector, direct, swapped seed column).
    #[serde(borrow)]
    lanes: Vec<(&'a RawValue, &'a RawValue, usize, usize)>,
}

/// One stored resolved column.
#[derive(serde::Deserialize)]
struct ItemView<'a> {
    #[serde(borrow)]
    key: &'a RawValue,
    #[serde(borrow)]
    rows: Vec<&'a RawValue>,
    basis: Option<Vec<usize>>,
}

impl ItemView<'_> {
    /// Its colour component: its Born basis element at leading colour, None
    /// (the contracted component) otherwise.
    fn basis(&self) -> RusticolResult<Option<usize>> {
        match self.basis.as_deref() {
            None => Ok(None),
            Some([b]) => Ok(Some(*b)),
            Some(_) => Err(invalid("a resolved column has one basis element")),
        }
    }

    /// Its one Born row (the stored JSON).
    fn row(&self) -> RusticolResult<&str> {
        match self.rows.as_slice() {
            [row] => Ok(row.get()),
            _ => Err(invalid("a resolved column has one Born row")),
        }
    }

    /// Its key as a JSON value.
    fn key(&self) -> RusticolResult<Value> {
        value_of(self.key)
    }
}

/// A borrowed stored JSON text as a value.
fn value_of(raw: &RawValue) -> RusticolResult<Value> {
    serde_json::from_str(raw.get()).map_err(|e| invalid(format!("a stored record: {e}")))
}

/// One seed-weight term: (plan Born component, real part, imaginary part as
/// stored).
type Term<'a> = (usize, &'a RawValue, &'a RawValue);

/// A seed class: per seed-weight column its terms.
#[derive(Clone, serde::Deserialize, serde::Serialize)]
struct SeedClassView<'a> {
    #[serde(borrow)]
    columns: Vec<Vec<Term<'a>>>,
}

impl<'a> SeedClassView<'a> {
    /// Its seed-weight column `w`.
    fn column(&self, w: usize) -> RusticolResult<&[Term<'a>]> {
        self.columns
            .get(w)
            .map(Vec::as_slice)
            .ok_or_else(|| invalid("a seed-weight column out of range"))
    }
}

/// A seed key's seed classes (the class index of each `[class, re, im]`).
#[derive(serde::Deserialize)]
struct SeedKeyView {
    classes: Vec<(usize, serde::de::IgnoredAny, serde::de::IgnoredAny)>,
}

/// A column set's seed records (`native_observables.ObservableSet` overlay),
/// typed and borrowed.
#[derive(serde::Deserialize)]
struct OverlayView<'a> {
    #[serde(borrow, default)]
    seed_classes: Option<Vec<SeedClassView<'a>>>,
    #[serde(borrow, default)]
    seeds: Option<&'a RawValue>,
    columns: usize,
    #[serde(borrow, default)]
    column_rows: Option<Vec<&'a RawValue>>,
    #[serde(default)]
    column_weights: Option<Vec<usize>>,
    born_components: usize,
}

impl OverlayView<'_> {
    fn own_seeds(&self) -> bool {
        self.seed_classes.is_some() && self.seeds.is_some()
    }
}

/// A formed set's seed records over a plan, as the loader reads them.
#[derive(serde::Serialize)]
struct FormedOverlay<'a> {
    /// None: the plan's own seed records (`plan_weights`).
    #[serde(skip_serializing_if = "Option::is_none")]
    seed_classes: Option<Vec<SeedClassView<'a>>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seeds: Option<&'a RawValue>,
    columns: usize,
    column_rows: Value,
    column_weights: Option<Vec<usize>>,
    born_components: usize,
}

/// A formed set's seed columns as the plan's own seed-weight columns, when
/// it can read them without seed records of its own: the stored set stores
/// none (it reads the plan's, e.g. a full-colour resolved set), each formed
/// column reads exactly one seed-weight column `weights[c]`, and they cover
/// the plan's columns as the loader's `column_weights` rule requires (the
/// highest + 1 = the plan's seed-weight column count, at most the formed
/// column count). The set then reads the plan's seed records in place
/// rather than a copy of them (u u~ > g g g g g at full colour: 1.9 M terms,
/// about 260 MB once loaded), with the same terms in the same order.
fn plan_weights(
    overlay: &OverlayView<'_>,
    seed_records: &[SeedClassView<'_>],
    weights: impl IntoIterator<Item = Option<usize>>,
) -> Option<Vec<usize>> {
    if overlay.own_seeds() {
        return None;
    }
    let weights: Vec<usize> = weights.into_iter().collect::<Option<_>>()?;
    let width = seed_records.first().map_or(0, |class| class.columns.len());
    let covered = weights.iter().max().map(|w| w + 1);
    (covered == Some(width) && width <= weights.len()).then_some(weights)
}

/// The seed records a formation reads for one plan: the set's own, else
/// the plan's.
fn seed_records<'b, 'a: 'b>(
    overlay: &'b OverlayView<'a>,
    plan: Option<&'b PlanSeeds<'b>>,
) -> RusticolResult<(&'b [SeedClassView<'b>], &'b RawValue)> {
    match (&overlay.seed_classes, overlay.seeds) {
        (Some(classes), Some(keys)) => Ok((classes.as_slice(), keys)),
        _ => {
            let plan = plan.ok_or_else(|| invalid("a stored record lacks \"seed_classes\""))?;
            Ok((plan.classes.as_slice(), plan.keys))
        }
    }
}

/// A plan's seed records, borrowed from its [`PlanView`].
struct PlanSeeds<'p> {
    classes: Vec<SeedClassView<'p>>,
    keys: &'p RawValue,
}

impl<'p> PlanSeeds<'p> {
    fn of(plan: &'p PlanView) -> RusticolResult<Self> {
        let classes = serde_json::from_str(plan.seed_classes()?.get())
            .map_err(|e| invalid(format!("a plan's seed classes: {e}")))?;
        Ok(Self {
            classes,
            keys: plan.seeds()?,
        })
    }
}

/// The interference programs of a column set: its records (passed through)
/// and its columns.
struct ColumnsView<'a> {
    top: BTreeMap<&'a str, &'a RawValue>,
    columns: Vec<&'a RawValue>,
}

impl<'a> ColumnsView<'a> {
    fn read(bytes: &'a [u8]) -> RusticolResult<Self> {
        let what =
            |e: serde_json::Error| invalid(format!("the resolved interference programs: {e}"));
        let top: BTreeMap<&'a str, &'a RawValue> = serde_json::from_slice(bytes).map_err(what)?;
        let columns = top
            .get("columns")
            .ok_or_else(|| invalid("the interference programs lack 'columns'"))?;
        let columns = serde_json::from_str(columns.get()).map_err(what)?;
        Ok(Self { top, columns })
    }

    /// The records with these interference columns.
    fn with_columns(&self, columns: &impl serde::Serialize) -> RusticolResult<String> {
        let what = |e: serde_json::Error| invalid(format!("the interference programs: {e}"));
        let columns =
            RawValue::from_string(serde_json::to_string(columns).map_err(what)?).map_err(what)?;
        let mut top = self.top.clone();
        top.insert("columns", &columns);
        serde_json::to_string(&top).map_err(what)
    }
}

/// One interference column record: the fields the template reads.
#[derive(serde::Deserialize)]
struct ColumnRecordView<'a> {
    #[serde(borrow)]
    program: &'a RawValue,
    #[serde(borrow)]
    cells: &'a RawValue,
    #[serde(borrow)]
    swapped: &'a RawValue,
}

/// A template interference column.
#[derive(serde::Serialize)]
struct InterferenceColumn<'a> {
    program: &'a RawValue,
    cells: &'a RawValue,
    swapped: &'a RawValue,
    rows: Option<Vec<usize>>,
    basis: Option<Vec<usize>>,
}

/// The stored resolved set's records: its columns, the process rows and
/// basis count, and per resolved column and sector its seed column.
struct Resolved<'a> {
    items: Vec<ItemView<'a>>,
    /// The resolved lanes: (item, sector as stored, direct, swapped).
    lanes: Vec<(usize, &'a RawValue, usize, usize)>,
    rows: usize,
    basis: usize,
    /// Per item, its row index.
    row: Vec<usize>,
    /// The sectors (stored JSON) in their first appearance.
    sectors: Vec<&'a RawValue>,
    /// Per (item, sector index): its seed column.
    seed_column: BTreeMap<(usize, usize), usize>,
    overlays: Vec<OverlayView<'a>>,
    /// Per plan: each seed column's seed-weight column.
    weight_of: Vec<Vec<usize>>,
}

impl<'a> Resolved<'a> {
    fn read(records: &ResolvedRecords<'a>) -> RusticolResult<Self> {
        let record: RecordView<'a> = serde_json::from_slice(records.record)
            .map_err(|e| invalid(format!("the observables record: {e}")))?;
        let ResolvedView {
            columns: items,
            lanes: stored,
        } = record.sets.resolved;
        let rows: BTreeMap<&str, usize> = record
            .rows
            .iter()
            .enumerate()
            .map(|(h, row)| (row.get(), h))
            .collect();
        let row = items
            .iter()
            .map(|item| {
                rows.get(item.row()?)
                    .copied()
                    .ok_or_else(|| invalid("a resolved column of an unknown Born row"))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let by_key: BTreeMap<&str, usize> = items
            .iter()
            .enumerate()
            .map(|(k, item)| (item.key.get(), k))
            .collect();
        let mut sectors: Vec<&'a RawValue> = Vec::new();
        let mut seed_column = BTreeMap::new();
        let mut lanes = Vec::with_capacity(stored.len());
        for (key, sector, direct, swapped) in stored {
            let k = *by_key
                .get(key.get())
                .ok_or_else(|| invalid("a lane of an unknown resolved column"))?;
            lanes.push((k, sector, direct, swapped));
            if direct != swapped {
                return Err(invalid(
                    "a resolved column reads one seed column per sector",
                ));
            }
            let s = match sectors.iter().position(|known| known.get() == sector.get()) {
                Some(s) => s,
                None => {
                    sectors.push(sector);
                    sectors.len() - 1
                }
            };
            seed_column.insert((k, s), direct);
        }
        let basis = record.basis;
        let rows = record.rows.len();
        let mut overlays = vec![records.overlay];
        overlays.extend(records.companion_overlay);
        let overlays = overlays
            .into_iter()
            .map(|bytes| {
                serde_json::from_slice::<OverlayView<'a>>(bytes).map_err(|e| {
                    invalid(format!(
                        "the resolved set stores no seed records of its own: {e}"
                    ))
                })
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let weight_of = overlays
            .iter()
            .map(|overlay| match &overlay.column_weights {
                Some(weights) if !weights.is_empty() => weights.clone(),
                _ => (0..overlay.columns).collect(),
            })
            .collect();
        Ok(Self {
            items,
            lanes,
            rows,
            basis,
            row,
            sectors,
            seed_column,
            overlays,
            weight_of,
        })
    }

    fn seed_column(&self, item: usize, sector: usize) -> RusticolResult<usize> {
        self.seed_column
            .get(&(item, sector))
            .copied()
            .ok_or_else(|| invalid("a resolved column has no lane in a sector"))
    }
}

/// The plans of a formation (process, companion), as JSON.
fn plans(records: &ResolvedRecords<'_>, count: usize) -> RusticolResult<Vec<PlanView>> {
    let mut out = vec![plan_view(records.plan)?];
    if count > 1 {
        let companion = records
            .companion_plan
            .ok_or_else(|| invalid("a companion seed record without the companion plan"))?;
        out.push(plan_view(companion)?);
    }
    Ok(out)
}

impl SelectionTemplate {
    /// Form the template of a stored process (module docs) from its
    /// records: pure JSON, no Symbolica operation.
    pub fn form(records: &ResolvedRecords<'_>) -> RusticolResult<Self> {
        let resolved = Resolved::read(records)?;
        let every = resolved.rows;
        let found_axis: BTreeSet<Option<usize>> = resolved
            .items
            .iter()
            .map(ItemView::basis)
            .collect::<RusticolResult<_>>()?;
        if found_axis.contains(&None) && found_axis.len() > 1 {
            return Err(invalid(
                "a resolved set mixes basis elements and contracted columns",
            ));
        }
        let axis: Vec<Option<usize>> = if found_axis.contains(&None) {
            vec![None]
        } else {
            (0..resolved.basis).map(Some).collect()
        };
        let plan_count = resolved.overlays.len();
        // Per row: the seed-weight column of each (colour component, sector,
        // plan); rows with equal ones form a row class.
        let mut weights: Vec<Signature> = vec![BTreeMap::new(); every];
        let mut seen = vec![false; every];
        for (k, item) in resolved.items.iter().enumerate() {
            let b = item.basis()?;
            let h = resolved.row[k];
            seen[h] = true;
            for s in 0..resolved.sectors.len() {
                let i = resolved.seed_column(k, s)?;
                for (p, of) in resolved.weight_of.iter().enumerate() {
                    let w = *of
                        .get(i)
                        .ok_or_else(|| invalid("a seed column without its weights"))?;
                    weights[h].insert((b, s, p), w);
                }
            }
        }
        if seen.iter().any(|seen| !seen) {
            return Err(invalid("the resolved set lacks a Born row"));
        }
        let mut classes: Vec<(&Signature, Vec<usize>)> = Vec::new();
        for (h, signature) in weights.iter().enumerate() {
            match classes.iter_mut().find(|(known, _)| *known == signature) {
                Some((_, hs)) => hs.push(h),
                None => classes.push((signature, vec![h])),
            }
        }
        let records_json = ColumnsView::read(records.columns)?;
        let read = records_json
            .columns
            .iter()
            .map(|column| {
                serde_json::from_str::<ColumnRecordView<'_>>(column.get())
                    .map_err(|e| invalid(format!("an interference column: {e}")))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let programs: BTreeSet<(&str, &str, &str)> = read
            .iter()
            .map(|column| {
                (
                    column.program.get(),
                    column.cells.get(),
                    column.swapped.get(),
                )
            })
            .collect();
        if programs.len() != 1 {
            return Err(invalid("the resolved columns read different programs"));
        }
        let (program, cells, swapped) = (read[0].program, read[0].cells, read[0].swapped);
        let mut outputs = Vec::new();
        let mut interference = Vec::new();
        let mut lanes = Vec::new();
        let mut seed_rows: Vec<Option<Vec<usize>>> = Vec::new();
        // Per plan and output seed column: its seed-weight columns in order.
        let mut template_weights: Vec<Vec<Vec<usize>>> = vec![Vec::new(); plan_count];
        for (c, (signature, hs)) in classes.iter().enumerate() {
            let kept_rows = (hs.len() != every).then(|| hs.clone());
            outputs.push(FormedColumn {
                key: json!([TEMPLATE, c]),
                rows: kept_rows.clone(),
                basis: None,
            });
            interference.push(InterferenceColumn {
                program,
                cells,
                swapped,
                rows: kept_rows.clone(),
                basis: None,
            });
            for (s, sector) in resolved.sectors.iter().enumerate() {
                lanes.push((c, value_of(sector)?, seed_rows.len(), seed_rows.len()));
                seed_rows.push(kept_rows.clone());
                for (p, columns) in template_weights.iter_mut().enumerate() {
                    let ws: BTreeSet<usize> = signature
                        .iter()
                        .filter(|((_, lab, q), _)| *lab == s && *q == p)
                        .map(|(_, &w)| w)
                        .collect();
                    columns.push(ws.into_iter().collect());
                }
            }
        }
        let classes: Vec<Vec<usize>> = classes.into_iter().map(|(_, hs)| hs).collect();
        let plans = plans(records, plan_count)?;
        let plan_seeds = resolved
            .overlays
            .iter()
            .zip(&plans)
            .map(|(overlay, plan)| {
                (!overlay.own_seeds())
                    .then(|| PlanSeeds::of(plan))
                    .transpose()
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let mut basis_components = Vec::new();
        let mut class_components = Vec::new();
        let mut components = Vec::new();
        let mut family_classes = Vec::new();
        let mut formed = Vec::new();
        for (p, (overlay, plan)) in resolved.overlays.iter().zip(&plans).enumerate() {
            let (seed_records, keys) = seed_records(overlay, plan_seeds[p].as_ref())?;
            let mut owner: BTreeMap<usize, Option<usize>> = BTreeMap::new();
            let mut per_basis: BTreeMap<Option<usize>, BTreeSet<usize>> =
                axis.iter().map(|&b| (b, BTreeSet::new())).collect();
            // Each (colour component, seed-weight column) of the plan once
            // (rows of a class, and sectors, repeat them).
            let pairs: BTreeSet<(Option<usize>, usize)> = weights
                .iter()
                .flat_map(|signature| signature.iter())
                .filter(|((_, _, q), _)| *q == p)
                .map(|(&(b, _, _), &w)| (b, w))
                .collect();
            for &(b, w) in &pairs {
                for record in seed_records {
                    for &(component, _, _) in record.column(w)? {
                        if *owner.entry(component).or_insert(b) != b {
                            return Err(invalid(
                                "a plan Born component carries the seed weights of two \
                                 Born basis elements; the selection template masks its \
                                 seeds by component",
                            ));
                        }
                        per_basis.entry(b).or_default().insert(component);
                    }
                }
            }
            basis_components.push(per_basis);
            let mut class_columns = Vec::new();
            let mut reads = Vec::new();
            for record in seed_records {
                let mut merged = Vec::new();
                let mut read = BTreeSet::new();
                for ws in &template_weights[p] {
                    let mut terms = Vec::new();
                    for &w in ws {
                        terms.extend_from_slice(record.column(w)?);
                    }
                    let distinct: BTreeSet<usize> = terms.iter().map(|term| term.0).collect();
                    if distinct.len() != terms.len() {
                        return Err(invalid(
                            "a seed class repeats a plan Born component across the \
                             resolved seed weights",
                        ));
                    }
                    read.extend(distinct);
                    merged.push(terms);
                }
                class_columns.push(SeedClassView { columns: merged });
                reads.push(read);
            }
            class_components.push(reads);
            let born_components = overlay.born_components;
            components.push(born_components);
            let reads_every = plan.forward.is_some() || plan.merged.is_some();
            if reads_every {
                family_classes.push(None);
            } else {
                let keys: Vec<SeedKeyView> = serde_json::from_str(keys.get())
                    .map_err(|e| invalid(format!("the seed keys: {e}")))?;
                let mut per_family = Vec::with_capacity(plan.families.len());
                for family in &plan.families {
                    let mut classes_read = BTreeSet::new();
                    for leaf in &family.leaves {
                        let key = keys
                            .get(leaf.seed)
                            .ok_or_else(|| invalid("a leaf of an unknown seed key"))?;
                        classes_read.extend(key.classes.iter().map(|class| class.0));
                    }
                    per_family.push(classes_read);
                }
                family_classes.push(Some(per_family));
            }
            let any_rows = seed_rows.iter().any(Option::is_some);
            let in_place = plan_weights(
                overlay,
                seed_records,
                template_weights[p]
                    .iter()
                    .map(|ws| (ws.len() == 1).then(|| ws[0])),
            );
            let (class_columns, keys) = match in_place {
                Some(_) => (None, None),
                None => (Some(class_columns), Some(keys)),
            };
            let overlay = FormedOverlay {
                seed_classes: class_columns,
                seeds: keys,
                columns: seed_rows.len(),
                column_rows: if any_rows {
                    json!(seed_rows)
                } else {
                    Value::Null
                },
                column_weights: in_place,
                born_components,
            };
            formed.push(
                serde_json::to_string(&overlay)
                    .map_err(|e| invalid(format!("a formed seed record: {e}")))?,
            );
        }
        let columns = records_json.with_columns(&interference)?;
        let mut formed = formed.into_iter();
        let overlay = formed.next().expect("the process plan");
        let companion_overlay = formed.next();
        Ok(Self {
            set: FormedColumns {
                overlay,
                companion_overlay,
                columns,
                outputs,
                lanes,
            },
            classes,
            rows: every,
            basis: resolved.basis,
            axis,
            basis_components,
            class_components,
            components,
            family_classes,
        })
    }

    /// The Born helicity rows of the process.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// The per-call masks of a selection keeping the Born rows `rows` and
    /// (leading colour) the Born basis elements `basis` (None keeps every
    /// one of that axis): its kept rows and basis elements, per plan the
    /// seeds' kept Born components (those of the kept basis elements) and
    /// the families with a leaf on a seed class with a kept term (the
    /// others' seeds vanish), and the template columns with a kept row.
    pub fn masks(
        &self,
        rows: Option<&BTreeSet<usize>>,
        basis: Option<&BTreeSet<usize>>,
    ) -> RusticolResult<SelectionMasks> {
        let kept_rows: Vec<usize> = (0..self.rows)
            .filter(|h| rows.is_none_or(|rows| rows.contains(h)))
            .collect();
        let kept_basis: Vec<Option<usize>> = self
            .axis
            .iter()
            .copied()
            .filter(|b| basis.is_none_or(|basis| b.is_some_and(|b| basis.contains(&b))))
            .collect();
        if kept_rows.is_empty() || kept_basis.is_empty() {
            return Err(invalid("a selection keeps no resolved column"));
        }
        let row_mask = (kept_rows.len() != self.rows).then(|| {
            (0..self.rows)
                .map(|h| kept_rows.contains(&h))
                .collect::<Vec<_>>()
        });
        let wanted = self
            .classes
            .iter()
            .enumerate()
            .filter(|(_, hs)| {
                hs.iter()
                    .any(|&h| row_mask.as_ref().is_none_or(|mask| mask[h]))
            })
            .map(|(c, _)| c)
            .collect();
        let every = kept_basis.len() == self.axis.len();
        let mut components = Vec::new();
        let mut families = Vec::new();
        for (p, per_basis) in self.basis_components.iter().enumerate() {
            let kept: BTreeSet<usize> = kept_basis
                .iter()
                .flat_map(|b| per_basis.get(b).into_iter().flatten().copied())
                .collect();
            components.push(
                (!every).then(|| (0..self.components[p]).map(|k| kept.contains(&k)).collect()),
            );
            let Some(family_classes) = &self.family_classes[p] else {
                families.push(None);
                continue;
            };
            let live: BTreeSet<usize> = self.class_components[p]
                .iter()
                .enumerate()
                .filter(|(_, read)| !read.is_disjoint(&kept))
                .map(|(c, _)| c)
                .collect();
            let flags: Vec<bool> = family_classes
                .iter()
                .map(|classes| !classes.is_disjoint(&live))
                .collect();
            families.push((!flags.iter().all(|&f| f)).then_some(flags));
        }
        let chosen: BTreeSet<Option<usize>> = kept_basis.into_iter().collect();
        Ok(SelectionMasks {
            rows: row_mask,
            basis: (!every).then(|| (0..self.basis).map(|b| chosen.contains(&Some(b))).collect()),
            components,
            families,
            wanted,
        })
    }
}

/// The stored resolved columns of a resolved selection (lit-fc-hel (G3)):
/// the columns of the kept Born `rows` and (leading colour) the kept basis
/// elements only (None keeps every one of that axis), each with its stored
/// seed column, seed-weight column and interference program, so the set's
/// plan seeds (and its load keeps, `column_rows`) the kept rows only. The
/// column keys and values are the resolved set's.
pub fn resolved_subset(
    records: &ResolvedRecords<'_>,
    rows: Option<&BTreeSet<usize>>,
    basis: Option<&BTreeSet<usize>>,
) -> RusticolResult<FormedColumns> {
    let resolved = Resolved::read(records)?;
    let mut kept = Vec::new();
    for (k, item) in resolved.items.iter().enumerate() {
        let b = item.basis()?;
        let row_kept = rows.is_none_or(|rows| rows.contains(&resolved.row[k]));
        let basis_kept = basis.is_none_or(|basis| b.is_some_and(|b| basis.contains(&b)));
        if row_kept && basis_kept {
            kept.push(k);
        }
    }
    if kept.is_empty() {
        return Err(invalid("a selection keeps no resolved column"));
    }
    // The kept columns' seed columns, numbered in their order of use.
    let mut used: Vec<usize> = Vec::new();
    let mut lanes = Vec::new();
    let number = |used: &mut Vec<usize>, column: usize| -> usize {
        match used.iter().position(|&c| c == column) {
            Some(i) => i,
            None => {
                used.push(column);
                used.len() - 1
            }
        }
    };
    let position: BTreeMap<usize, usize> = kept.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    for &(k, sector, direct, swapped) in &resolved.lanes {
        let Some(&output) = position.get(&k) else {
            continue;
        };
        let direct = number(&mut used, direct);
        let swapped = number(&mut used, swapped);
        lanes.push((output, value_of(sector)?, direct, swapped));
    }
    // The plans' seed records, read only where a set stores none of its own.
    let plans = if resolved.overlays.iter().all(OverlayView::own_seeds) {
        Vec::new()
    } else {
        plans(records, resolved.overlays.len())?
    };
    let plan_seeds = resolved
        .overlays
        .iter()
        .enumerate()
        .map(|(p, overlay)| {
            (!overlay.own_seeds())
                .then(|| PlanSeeds::of(&plans[p]))
                .transpose()
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let mut formed = Vec::new();
    for (p, (overlay, weights)) in resolved
        .overlays
        .iter()
        .zip(&resolved.weight_of)
        .enumerate()
    {
        // The kept seed columns' seed-weight columns, renumbered in their
        // order of use: each seed class keeps those weight columns only, so
        // the set's plan reads as many weight columns as it keeps (the
        // loader's `column_weights` rule).
        let mut kept_weights: Vec<usize> = Vec::new();
        let column_weights = used
            .iter()
            .map(|&c| {
                let w = *weights
                    .get(c)
                    .ok_or_else(|| invalid("a seed column without its seed-weight column"))?;
                Ok(number(&mut kept_weights, w))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let (seed_records, keys) = seed_records(overlay, plan_seeds[p].as_ref())?;
        let in_place = plan_weights(
            overlay,
            seed_records,
            used.iter().map(|&c| weights.get(c).copied()),
        );
        let (classes, keys, column_weights) = match in_place {
            Some(in_place) => (None, None, in_place),
            None => {
                let classes = seed_records
                    .iter()
                    .map(|record| {
                        let columns = kept_weights
                            .iter()
                            .map(|&w| record.column(w).map(<[_]>::to_vec))
                            .collect::<RusticolResult<Vec<_>>>()?;
                        Ok(SeedClassView { columns })
                    })
                    .collect::<RusticolResult<Vec<_>>>()?;
                (Some(classes), Some(keys), column_weights)
            }
        };
        let column_rows = match &overlay.column_rows {
            None => Value::Null,
            Some(rows) => Value::Array(
                used.iter()
                    .map(|&c| {
                        value_of(
                            rows.get(c)
                                .ok_or_else(|| invalid("a seed column without its rows"))?,
                        )
                    })
                    .collect::<RusticolResult<_>>()?,
            ),
        };
        let formed_overlay = FormedOverlay {
            seed_classes: classes,
            seeds: keys,
            columns: used.len(),
            column_rows,
            column_weights: Some(column_weights),
            born_components: overlay.born_components,
        };
        formed.push(
            serde_json::to_string(&formed_overlay)
                .map_err(|e| invalid(format!("a formed seed record: {e}")))?,
        );
    }
    let columns_json = ColumnsView::read(records.columns)?;
    let picked = kept
        .iter()
        .map(|&k| {
            columns_json
                .columns
                .get(k)
                .copied()
                .ok_or_else(|| invalid("a resolved column without its interference"))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let columns = columns_json.with_columns(&picked)?;
    let outputs = kept
        .iter()
        .map(|&k| {
            let item = &resolved.items[k];
            Ok(FormedColumn {
                key: item.key()?,
                rows: Some(vec![resolved.row[k]]),
                basis: item.basis()?.map(|b| vec![b]),
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let mut formed = formed.into_iter();
    Ok(FormedColumns {
        overlay: formed.next().expect("the process plan"),
        companion_overlay: formed.next(),
        columns,
        outputs,
        lanes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A full-colour process of two Born rows (a contracted resolved column
    /// each, both reading seed-weight column 0) with one seed class and one
    /// family, as `native_observables.StoredObservables` stores it.
    fn records() -> (Value, Value, Value, Value) {
        let key = |h: i32| json!([[[0, h]], null]);
        let record = json!({
            "rows": [[[0, -1]], [[0, 1]]],
            "basis": 2,
            "sets": {"resolved": {
                "columns": [
                    {"key": key(-1), "rows": [[[0, -1]]], "basis": null},
                    {"key": key(1), "rows": [[[0, 1]]], "basis": null},
                ],
                "lanes": [[key(-1), [["QCD", 2]], 0, 0], [key(1), [["QCD", 2]], 1, 1]],
            }},
        });
        let overlay = json!({
            "columns": 2, "column_rows": [[0], [1]], "column_weights": [0, 0],
            "born_components": 2,
        });
        let columns = json!({"cells": [], "programs": [], "columns": [
            {"program": 0, "cells": 0, "swapped": null, "rows": [0], "basis": null},
            {"program": 0, "cells": 0, "swapped": null, "rows": [1], "basis": null},
        ]});
        let plan = json!({
            "seed_classes": [{"columns": [[[0, "1", "0"], [1, "2", "0"]]]}],
            "seeds": [{"classes": [[0, "1", "0"]]}],
            "families": [{"leaves": [{"seed": 0}]}],
        });
        (record, overlay, columns, plan)
    }

    #[test]
    fn a_full_colour_template_masks_rows_only_and_a_subset_keeps_its_columns() {
        let (record, overlay, columns, plan) = records();
        let (overlay, columns, plan) = (overlay.to_string(), columns.to_string(), plan.to_string());
        let record = record.to_string();
        let records = ResolvedRecords {
            record: record.as_bytes(),
            overlay: overlay.as_bytes(),
            companion_overlay: None,
            columns: columns.as_bytes(),
            plan: plan.as_bytes(),
            companion_plan: None,
        };
        let template = SelectionTemplate::form(&records).unwrap();
        // One row class: one seed column of every row with the contracted
        // component's weights, one interference column of every row.
        assert_eq!(template.classes, vec![vec![0, 1]]);
        let formed: Value = serde_json::from_str(&template.set.overlay).unwrap();
        assert_eq!(formed["columns"], json!(1));
        assert_eq!(formed["column_rows"], Value::Null);
        // The stored set reads the plan's seed records, and so does the
        // template: in place, its one column the plan's seed-weight column 0.
        assert_eq!(formed.get("seed_classes"), None);
        assert_eq!(formed["column_weights"], json!([0]));
        let interference: Value = serde_json::from_str(&template.set.columns).unwrap();
        assert_eq!(interference["columns"][0]["rows"], Value::Null);
        assert_eq!(template.set.lanes, vec![(0, json!([["QCD", 2]]), 0, 0)]);
        // A selection masks the rows only; no basis element is selectable.
        let one: BTreeSet<usize> = [1].into();
        let masks = template.masks(Some(&one), None).unwrap();
        assert_eq!(masks.rows, Some(vec![false, true]));
        assert_eq!((masks.basis.clone(), masks.wanted.clone()), (None, vec![0]));
        assert_eq!(
            (masks.components.clone(), masks.families.clone()),
            (vec![None], vec![None])
        );
        assert_eq!(template.masks(None, None).unwrap().rows, None);
        let basis: BTreeSet<usize> = [0].into();
        assert!(template.masks(None, Some(&basis)).is_err());
        let call = masks.call(0);
        let sources = call.sources::<f64>(None, 2).unwrap();
        assert!(
            sources
                .iter()
                .all(|s| s.rows == Some(vec![false, true]) && s.basis.is_none())
        );
        // The resolved selection of row 1: its column alone.
        let subset = resolved_subset(&records, Some(&one), None).unwrap();
        assert_eq!(subset.outputs.len(), 1);
        assert_eq!(subset.outputs[0].key, json!([[[0, 1]], null]));
        assert_eq!(subset.outputs[0].rows, Some(vec![1]));
        let formed: Value = serde_json::from_str(&subset.overlay).unwrap();
        assert_eq!(formed["column_rows"], json!([[1]]));
        assert_eq!(formed["column_weights"], json!([0]));
        assert_eq!(formed.get("seed_classes"), None);
        assert_eq!(subset.lanes, vec![(0, json!([["QCD", 2]]), 0, 0)]);
        let interference: Value = serde_json::from_str(&subset.columns).unwrap();
        assert_eq!(
            interference["columns"],
            json!([
                {"program": 0, "cells": 0, "swapped": null, "rows": [1], "basis": null}
            ])
        );
    }

    #[test]
    fn a_leading_colour_subset_keeps_only_its_seed_weight_columns() {
        // Two Born rows x two basis elements, each resolved column reading
        // its basis element's seed-weight column (the set's own seed records).
        let key = |h: i32, b: usize| json!([[[0, h]], [b]]);
        let mut items = Vec::new();
        let mut lanes = Vec::new();
        let mut interference = Vec::new();
        for (k, (h, b)) in [(-1, 0), (-1, 1), (1, 0), (1, 1)].into_iter().enumerate() {
            items.push(json!({"key": key(h, b), "rows": [[[0, h]]], "basis": [b]}));
            lanes.push(json!([key(h, b), [["QCD", 2]], k, k]));
            interference.push(json!({"program": 0, "cells": 0, "swapped": null,
                "rows": [usize::from(h > 0)], "basis": [b]}));
        }
        let record = json!({
            "rows": [[[0, -1]], [[0, 1]]],
            "basis": 2,
            "sets": {"resolved": {"columns": items, "lanes": lanes}},
        });
        let overlay = json!({
            "columns": 4, "column_rows": [[0], [0], [1], [1]], "column_weights": [0, 1, 0, 1],
            "seed_classes": [{"columns": [[[0, "1", "0"]], [[1, "3", "0"]]]}],
            "seeds": [{"classes": [[0, "1", "0"]]}],
            "born_components": 2,
        })
        .to_string();
        let columns = json!({"cells": [], "programs": [], "columns": interference}).to_string();
        let record = record.to_string();
        let records = ResolvedRecords {
            record: record.as_bytes(),
            overlay: overlay.as_bytes(),
            companion_overlay: None,
            columns: columns.as_bytes(),
            plan: b"",
            companion_plan: None,
        };
        // Row 1, basis element 1: the last column alone, its seed-weight
        // column renumbered to the set's only one.
        let (one, second): (BTreeSet<usize>, BTreeSet<usize>) = ([1].into(), [1].into());
        let subset = resolved_subset(&records, Some(&one), Some(&second)).unwrap();
        assert_eq!(subset.outputs.len(), 1);
        assert_eq!(subset.outputs[0].key, key(1, 1));
        assert_eq!(subset.outputs[0].basis, Some(vec![1]));
        let formed: Value = serde_json::from_str(&subset.overlay).unwrap();
        assert_eq!(formed["columns"], json!(1));
        assert_eq!(formed["column_rows"], json!([[1]]));
        assert_eq!(formed["column_weights"], json!([0]));
        assert_eq!(
            formed["seed_classes"],
            json!([{"columns": [[[1, "3", "0"]]]}])
        );
        // Both rows of basis element 0: two seed columns, one weight column.
        let first: BTreeSet<usize> = [0].into();
        let subset = resolved_subset(&records, None, Some(&first)).unwrap();
        let formed: Value = serde_json::from_str(&subset.overlay).unwrap();
        assert_eq!(formed["column_weights"], json!([0, 0]));
        assert_eq!(
            formed["seed_classes"],
            json!([{"columns": [[[0, "1", "0"]]]}])
        );
        assert_eq!(
            subset.lanes,
            vec![
                (0, json!([["QCD", 2]]), 0, 0),
                (1, json!([["QCD", 2]]), 1, 1)
            ]
        );
    }

    #[test]
    fn call_masks_zero_the_unkept_born_components() {
        let masks = CallMasks {
            components: Some(vec![true, false]),
            ..Default::default()
        };
        let one = Complex::new(1.0, 2.0);
        let mut born = vec![vec![one, one, one, one]];
        masks.mask_born(&mut born, &0.0).unwrap();
        let zero = Complex::new(0.0, 0.0);
        assert_eq!(born, vec![vec![one, zero, one, zero]]);
        let mut odd = vec![vec![one; 3]];
        assert!(masks.mask_born(&mut odd, &0.0).is_err());
        assert!(CallMasks::default().is_empty());
    }
}
