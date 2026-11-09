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
use serde_json::{Value, json};
use std::collections::BTreeSet;

/// The key prefix of the template's columns (one per row class).
pub const TEMPLATE: &str = "selection-template";

/// The stored records a formation reads: the observables record
/// (`observables.json`), the resolved set's seed records (and its
/// companion's), its interference programs, and the process (companion)
/// plan (JSON or the stored binary plan).
pub struct ResolvedRecords<'a> {
    pub record: &'a Value,
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

fn parse(bytes: &[u8], what: &str) -> RusticolResult<Value> {
    serde_json::from_slice(bytes).map_err(|e| invalid(format!("{what}: {e}")))
}

/// A plan as JSON (a stored binary plan member is read as its JSON).
fn plan_value(bytes: &[u8]) -> RusticolResult<Value> {
    use rusticol_core::loop_v2::plan::binary::{PlanModel, is_binary};
    if is_binary(bytes) {
        return parse(&PlanModel::from_binary(bytes)?.json(), "plan");
    }
    parse(bytes, "plan")
}

fn usize_of(value: &Value, what: &str) -> RusticolResult<usize> {
    value
        .as_u64()
        .map(|v| v as usize)
        .ok_or_else(|| invalid(format!("{what} is not an index")))
}

fn array<'a>(value: &'a Value, what: &str) -> RusticolResult<&'a Vec<Value>> {
    value
        .as_array()
        .ok_or_else(|| invalid(format!("{what} is not a list")))
}

fn field<'a>(value: &'a Value, name: &str) -> RusticolResult<&'a Value> {
    value
        .get(name)
        .ok_or_else(|| invalid(format!("a stored record lacks {name:?}")))
}

/// A resolved column's colour component: its Born basis element at leading
/// colour, None (the contracted component) otherwise.
fn basis_of(item: &Value) -> RusticolResult<Option<usize>> {
    match field(item, "basis")? {
        Value::Null => Ok(None),
        basis => {
            let items = array(basis, "a resolved column's basis")?;
            match items.as_slice() {
                [b] => Ok(Some(usize_of(b, "a basis element")?)),
                _ => Err(invalid("a resolved column has one basis element")),
            }
        }
    }
}

/// A resolved column's one Born row.
fn row_of(item: &Value) -> RusticolResult<&Value> {
    match array(field(item, "rows")?, "a resolved column's rows")?.as_slice() {
        [row] => Ok(row),
        _ => Err(invalid("a resolved column has one Born row")),
    }
}

/// The stored resolved set's records: its columns, the process rows and
/// basis count, and per resolved column and sector its seed column.
struct Resolved<'a> {
    items: &'a Vec<Value>,
    rows: &'a Vec<Value>,
    basis: usize,
    /// Per item, its row index.
    row: Vec<usize>,
    /// The sectors (stored JSON) in their first appearance.
    sectors: Vec<Value>,
    /// Per (item, sector index): its seed column.
    seed_column: BTreeMap<(usize, usize), usize>,
    overlays: Vec<serde_json::Map<String, Value>>,
    /// Per plan: each seed column's seed-weight column.
    weight_of: Vec<Vec<usize>>,
}

impl<'a> Resolved<'a> {
    fn read(records: &ResolvedRecords<'_>, record: &'a Value) -> RusticolResult<Self> {
        let meta = field(field(record, "sets")?, "resolved")?;
        let items = array(field(meta, "columns")?, "the resolved columns")?;
        let rows = array(field(record, "rows")?, "the Born rows")?;
        let basis = usize_of(field(record, "basis")?, "the basis count")?;
        let row = items
            .iter()
            .map(|item| {
                let found = row_of(item)?;
                rows.iter()
                    .position(|known| known == found)
                    .ok_or_else(|| invalid("a resolved column of an unknown Born row"))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let by_key: BTreeMap<String, usize> = items
            .iter()
            .enumerate()
            .map(|(k, item)| Ok((field(item, "key")?.to_string(), k)))
            .collect::<RusticolResult<_>>()?;
        let mut sectors: Vec<Value> = Vec::new();
        let mut seed_column = BTreeMap::new();
        for lane in array(field(meta, "lanes")?, "the resolved lanes")? {
            let lane = array(lane, "a lane")?;
            let [key, sector, direct, swapped] = lane.as_slice() else {
                return Err(invalid("a lane is (key, sector, direct, swapped)"));
            };
            if direct != swapped {
                return Err(invalid(
                    "a resolved column reads one seed column per sector",
                ));
            }
            let s = match sectors.iter().position(|known| known == sector) {
                Some(s) => s,
                None => {
                    sectors.push(sector.clone());
                    sectors.len() - 1
                }
            };
            let k = *by_key
                .get(&key.to_string())
                .ok_or_else(|| invalid("a lane of an unknown resolved column"))?;
            seed_column.insert((k, s), usize_of(direct, "a seed column")?);
        }
        let mut overlays = vec![records.overlay];
        overlays.extend(records.companion_overlay);
        let overlays = overlays
            .into_iter()
            .map(|bytes| match parse(bytes, "the resolved seed records")? {
                Value::Object(map) => Ok(map),
                _ => Err(invalid(
                    "the resolved set stores no seed records of its own",
                )),
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let weight_of = overlays
            .iter()
            .map(|overlay| match overlay.get("column_weights") {
                Some(Value::Array(weights)) if !weights.is_empty() => weights
                    .iter()
                    .map(|w| usize_of(w, "a column weight"))
                    .collect::<RusticolResult<Vec<_>>>(),
                _ => Ok((0..usize_of(
                    overlay
                        .get("columns")
                        .ok_or_else(|| invalid("seed records lack 'columns'"))?,
                    "the seed column count",
                )?)
                    .collect()),
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        Ok(Self {
            items,
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
fn plans(records: &ResolvedRecords<'_>, count: usize) -> RusticolResult<Vec<Value>> {
    let mut out = vec![plan_value(records.plan)?];
    if count > 1 {
        let companion = records
            .companion_plan
            .ok_or_else(|| invalid("a companion seed record without the companion plan"))?;
        out.push(plan_value(companion)?);
    }
    Ok(out)
}

impl SelectionTemplate {
    /// Form the template of a stored process (module docs) from its
    /// records: pure JSON, no Symbolica operation.
    pub fn form(records: &ResolvedRecords<'_>) -> RusticolResult<Self> {
        let resolved = Resolved::read(records, records.record)?;
        let every = resolved.rows.len();
        let found_axis: BTreeSet<Option<usize>> = resolved
            .items
            .iter()
            .map(basis_of)
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
        let mut weights: Vec<BTreeMap<(Option<usize>, usize, usize), usize>> =
            vec![BTreeMap::new(); every];
        let mut seen = vec![false; every];
        for (k, item) in resolved.items.iter().enumerate() {
            let b = basis_of(item)?;
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
        let mut classes: Vec<(&BTreeMap<_, _>, Vec<usize>)> = Vec::new();
        for (h, signature) in weights.iter().enumerate() {
            match classes.iter_mut().find(|(known, _)| *known == signature) {
                Some((_, hs)) => hs.push(h),
                None => classes.push((signature, vec![h])),
            }
        }
        let mut records_json = match parse(records.columns, "the resolved interference programs")? {
            Value::Object(map) => map,
            _ => return Err(invalid("the resolved interference programs are no record")),
        };
        let programs: BTreeSet<String> = array(
            records_json
                .get("columns")
                .ok_or_else(|| invalid("the interference programs lack 'columns'"))?,
            "the interference columns",
        )?
        .iter()
        .map(|column| {
            json!([
                column.get("program"),
                column.get("cells"),
                column.get("swapped")
            ])
            .to_string()
        })
        .collect();
        if programs.len() != 1 {
            return Err(invalid("the resolved columns read different programs"));
        }
        let first = &records_json["columns"][0];
        let (program, cells, swapped) = (
            first.get("program").cloned().unwrap_or(Value::Null),
            first.get("cells").cloned().unwrap_or(Value::Null),
            first.get("swapped").cloned().unwrap_or(Value::Null),
        );
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
            interference.push(json!({
                "program": program,
                "cells": cells,
                "swapped": swapped,
                "rows": kept_rows,
                "basis": Value::Null,
            }));
            for (s, sector) in resolved.sectors.iter().enumerate() {
                lanes.push((c, sector.clone(), seed_rows.len(), seed_rows.len()));
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
        let mut basis_components = Vec::new();
        let mut class_components = Vec::new();
        let mut components = Vec::new();
        let mut family_classes = Vec::new();
        let mut formed = Vec::new();
        for (p, (overlay, plan)) in resolved.overlays.iter().zip(&plans).enumerate() {
            let (seed_records, keys) = match (overlay.get("seed_classes"), overlay.get("seeds")) {
                (Some(classes @ Value::Array(_)), Some(keys)) => (classes, keys),
                _ => (field(plan, "seed_classes")?, field(plan, "seeds")?),
            };
            let seed_records = array(seed_records, "the seed classes")?;
            let terms_of = |record: &Value, w: usize| -> RusticolResult<Vec<Value>> {
                let columns = array(field(record, "columns")?, "a seed class's columns")?;
                Ok(array(
                    columns
                        .get(w)
                        .ok_or_else(|| invalid("a seed-weight column out of range"))?,
                    "a seed-weight column",
                )?
                .clone())
            };
            let component_of = |term: &Value| -> RusticolResult<usize> {
                usize_of(
                    term.get(0).ok_or_else(|| invalid("a seed weight term"))?,
                    "a plan Born component",
                )
            };
            let mut owner: BTreeMap<usize, Option<usize>> = BTreeMap::new();
            let mut per_basis: BTreeMap<Option<usize>, BTreeSet<usize>> =
                axis.iter().map(|&b| (b, BTreeSet::new())).collect();
            for signature in &weights {
                for (&(b, _s, q), &w) in signature {
                    if q != p {
                        continue;
                    }
                    for record in seed_records {
                        for term in terms_of(record, w)? {
                            let component = component_of(&term)?;
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
                        terms.extend(terms_of(record, w)?);
                    }
                    let distinct: BTreeSet<usize> = terms
                        .iter()
                        .map(component_of)
                        .collect::<RusticolResult<_>>()?;
                    if distinct.len() != terms.len() {
                        return Err(invalid(
                            "a seed class repeats a plan Born component across the \
                             resolved seed weights",
                        ));
                    }
                    read.extend(distinct);
                    merged.push(Value::Array(terms));
                }
                class_columns.push(json!({ "columns": merged }));
                reads.push(read);
            }
            class_components.push(reads);
            let born_components = overlay
                .get("born_components")
                .cloned()
                .ok_or_else(|| invalid("seed records lack 'born_components'"))?;
            components.push(usize_of(&born_components, "the Born component count")?);
            let reads_every = plan.get("forward").is_some_and(|v| !v.is_null())
                || plan.get("merged").is_some_and(|v| !v.is_null());
            if reads_every {
                family_classes.push(None);
            } else {
                let keys = array(keys, "the seed keys")?;
                let families = array(field(plan, "families")?, "the plan families")?;
                let mut per_family = Vec::with_capacity(families.len());
                for family in families {
                    let mut classes_read = BTreeSet::new();
                    for leaf in array(field(family, "leaves")?, "a family's leaves")? {
                        let seed = usize_of(field(leaf, "seed")?, "a leaf seed")?;
                        let key = keys
                            .get(seed)
                            .ok_or_else(|| invalid("a leaf of an unknown seed key"))?;
                        for class in array(field(key, "classes")?, "a seed key's classes")? {
                            let c = class.get(0).ok_or_else(|| invalid("a seed key's class"))?;
                            classes_read.insert(usize_of(c, "a seed class")?);
                        }
                    }
                    per_family.push(classes_read);
                }
                family_classes.push(Some(per_family));
            }
            let any_rows = seed_rows.iter().any(Option::is_some);
            formed.push(json!({
                "seed_classes": class_columns,
                "seeds": keys,
                "columns": seed_rows.len(),
                "column_rows": if any_rows { json!(seed_rows) } else { Value::Null },
                "column_weights": Value::Null,
                "born_components": born_components,
            }));
        }
        records_json.insert("columns".to_owned(), Value::Array(interference));
        let mut formed = formed.into_iter().map(|value| value.to_string());
        let overlay = formed.next().expect("the process plan");
        let companion_overlay = formed.next();
        Ok(Self {
            set: FormedColumns {
                overlay,
                companion_overlay,
                columns: Value::Object(records_json).to_string(),
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
    let resolved = Resolved::read(records, records.record)?;
    let mut kept = Vec::new();
    for (k, item) in resolved.items.iter().enumerate() {
        let b = basis_of(item)?;
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
    let meta = field(field(records.record, "sets")?, "resolved")?;
    let position: BTreeMap<usize, usize> = kept.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    let by_key: BTreeMap<String, usize> = resolved
        .items
        .iter()
        .enumerate()
        .map(|(k, item)| Ok((field(item, "key")?.to_string(), k)))
        .collect::<RusticolResult<_>>()?;
    for lane in array(field(meta, "lanes")?, "the resolved lanes")? {
        let lane = array(lane, "a lane")?;
        let [key, sector, direct, swapped] = lane.as_slice() else {
            return Err(invalid("a lane is (key, sector, direct, swapped)"));
        };
        let k = by_key
            .get(&key.to_string())
            .ok_or_else(|| invalid("a lane of an unknown resolved column"))?;
        let Some(&output) = position.get(k) else {
            continue;
        };
        let direct = number(&mut used, usize_of(direct, "a seed column")?);
        let swapped = number(&mut used, usize_of(swapped, "a seed column")?);
        lanes.push((output, sector.clone(), direct, swapped));
    }
    let mut formed = Vec::new();
    for (overlay, weights) in resolved.overlays.iter().zip(&resolved.weight_of) {
        let mut out = serde_json::Map::new();
        for key in ["seed_classes", "seeds"] {
            if let Some(value) = overlay.get(key) {
                out.insert(key.to_owned(), value.clone());
            }
        }
        let column_rows = match overlay.get("column_rows") {
            None | Some(Value::Null) => Value::Null,
            Some(Value::Array(rows)) => Value::Array(
                used.iter()
                    .map(|&c| {
                        rows.get(c)
                            .cloned()
                            .ok_or_else(|| invalid("a seed column without its rows"))
                    })
                    .collect::<RusticolResult<_>>()?,
            ),
            Some(_) => return Err(invalid("column_rows is not a list")),
        };
        out.insert("columns".to_owned(), json!(used.len()));
        out.insert("column_rows".to_owned(), column_rows);
        out.insert(
            "column_weights".to_owned(),
            json!(used.iter().map(|&c| weights[c]).collect::<Vec<_>>()),
        );
        out.insert(
            "born_components".to_owned(),
            overlay
                .get("born_components")
                .cloned()
                .ok_or_else(|| invalid("seed records lack 'born_components'"))?,
        );
        formed.push(Value::Object(out).to_string());
    }
    let mut columns_json = match parse(records.columns, "the resolved interference programs")? {
        Value::Object(map) => map,
        _ => return Err(invalid("the resolved interference programs are no record")),
    };
    let all = array(
        columns_json
            .get("columns")
            .ok_or_else(|| invalid("the interference programs lack 'columns'"))?,
        "the interference columns",
    )?
    .clone();
    let picked = kept
        .iter()
        .map(|&k| {
            all.get(k)
                .cloned()
                .ok_or_else(|| invalid("a resolved column without its interference"))
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    columns_json.insert("columns".to_owned(), Value::Array(picked));
    let outputs = kept
        .iter()
        .map(|&k| {
            let item = &resolved.items[k];
            Ok(FormedColumn {
                key: field(item, "key")?.clone(),
                rows: Some(vec![resolved.row[k]]),
                basis: basis_of(item)?.map(|b| vec![b]),
            })
        })
        .collect::<RusticolResult<Vec<_>>>()?;
    let mut formed = formed.into_iter();
    Ok(FormedColumns {
        overlay: formed.next().expect("the process plan"),
        companion_overlay: formed.next(),
        columns: Value::Object(columns_json).to_string(),
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
        let records = ResolvedRecords {
            record: &record,
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
        assert_eq!(
            formed["seed_classes"][0]["columns"],
            json!([[[0, "1", "0"], [1, "2", "0"]]])
        );
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
