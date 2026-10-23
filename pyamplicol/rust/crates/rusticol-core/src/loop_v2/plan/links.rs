// SPDX-License-Identifier: 0BSD

//! Derived helicity rows and row links of the schedule-B steps (WP3.1 I11).
//!
//! The exporter (`plan_export._Exporter._step`) forms them from row tuples:
//! a leaf's rows are the Born rows; a step's output rows are the source rows
//! restricted to the legs it leaves open (`out`), numbered first-seen over
//! its edges and their source rows; an edge links a source row to its
//! content's block row of the same helicities on the content's legs. A plan
//! carries the Born rows (per leg its helicities, rows in product order) and
//! per content its legs and row helicities, so the loader derives the rows
//! and links instead of reading them (they are the bulk of a large plan). A
//! plan that still writes them (`export_plan(links=True)`) is checked.
//!
//! A row restricted to a leg set is the pair (legs, mixed-radix key): the
//! key sums each of its legs' helicity index times the leg's stride in the
//! Born product, so a Born row's key is its index and two restricted rows are
//! equal exactly when the exporter's tuples are.

use super::{ContentFile, FamilyFile, PlanFile, invalid, require};
use crate::RusticolResult;
use std::collections::HashMap;

/// (legs, key) of a helicity row restricted to `legs`.
pub(super) type Row = (u64, u64);

/// The Born rows as a product of the legs' helicities.
pub(super) struct RowSpace {
    stride: Vec<u64>,
    radix: Vec<u64>,
    digits: Vec<HashMap<i32, u64>>,
    all: u64,
    rows: usize,
}

impl RowSpace {
    pub(super) fn new(file: &PlanFile) -> RusticolResult<Self> {
        let legs = file.legs;
        require(
            file.helicities.len() == legs,
            "plan helicities per leg differ from its legs",
        )?;
        let mut stride = vec![0; legs];
        let mut radix = vec![0; legs];
        let mut size: u64 = 1;
        for leg in (0..legs).rev() {
            let helicities = &file.helicities[leg];
            require(
                !helicities.is_empty() && helicities.windows(2).all(|w| w[0] < w[1]),
                "a leg's helicities are not sorted and distinct",
            )?;
            stride[leg] = size;
            radix[leg] = helicities.len() as u64;
            size = size
                .checked_mul(radix[leg])
                .ok_or_else(|| invalid("Born rows overflow"))?;
        }
        require(
            size == file.rows as u64,
            "the Born rows are not the product of the legs' helicities",
        )?;
        let digits = file
            .helicities
            .iter()
            .map(|helicities| {
                helicities
                    .iter()
                    .enumerate()
                    .map(|(digit, &h)| (h, digit as u64))
                    .collect()
            })
            .collect();
        Ok(Self {
            stride,
            radix,
            digits,
            all: (1u64 << legs) - 1,
            rows: file.rows,
        })
    }

    fn restrict(&self, (legs, key): Row, mask: u64) -> Row {
        let legs = legs & mask;
        let mut restricted = 0;
        let mut bits = legs;
        while bits != 0 {
            let leg = bits.trailing_zeros() as usize;
            restricted += key / self.stride[leg] % self.radix[leg] * self.stride[leg];
            bits &= bits - 1;
        }
        (legs, restricted)
    }

    /// Block row index of each row of a content, by its restricted row.
    pub(super) fn content_rows(&self, content: &ContentFile) -> RusticolResult<HashMap<Row, u32>> {
        let mask = content.mask;
        require(mask & !self.all == 0, "content legs outside the process")?;
        require(
            content.row_helicities.len() == content.rows,
            "content row helicities differ from its rows",
        )?;
        let mut out = HashMap::with_capacity(content.rows);
        for (index, helicities) in content.row_helicities.iter().enumerate() {
            require(
                helicities.len() == mask.count_ones() as usize,
                "a content row off its legs",
            )?;
            let mut key = 0;
            let mut bits = mask;
            for h in helicities {
                let leg = bits.trailing_zeros() as usize;
                let digit = self.digits[leg]
                    .get(h)
                    .ok_or_else(|| invalid("a content row helicity outside the Born rows"))?;
                key += digit * self.stride[leg];
                bits &= bits - 1;
            }
            let index = u32::try_from(index).map_err(|_| invalid("too many content rows"))?;
            require(
                out.insert((mask, key), index).is_none(),
                "repeated content row",
            )?;
        }
        Ok(out)
    }
}

/// Per step its output rows and per edge its flat row links
/// `(target, source position, block)`.
pub(super) struct DerivedSteps {
    pub(super) rows: Vec<usize>,
    pub(super) links: Vec<Vec<Vec<u32>>>,
}

/// The rows and row links of one family's steps, in the exporter's order.
pub(super) fn derive(
    family: &FamilyFile,
    file: &PlanFile,
    space: &RowSpace,
    content_rows: &[HashMap<Row, u32>],
) -> RusticolResult<DerivedSteps> {
    let mut outputs: Vec<Option<Vec<Row>>> = vec![None; family.steps.len()];
    let born: Vec<Row> = (0..space.rows as u64).map(|key| (space.all, key)).collect();
    let mut derived = DerivedSteps {
        rows: Vec::with_capacity(family.steps.len()),
        links: Vec::with_capacity(family.steps.len()),
    };
    for (index, step) in family.steps.iter().enumerate() {
        let mut targets: HashMap<Row, u32> = HashMap::new();
        let mut order: Vec<Row> = Vec::new();
        let mut step_links = Vec::with_capacity(step.edges.len());
        for edge in &step.edges {
            let content = file
                .factors
                .get(edge.factor)
                .ok_or_else(|| invalid("edge factor index"))?
                .0;
            let blocks = content_rows
                .get(content)
                .ok_or_else(|| invalid("factor content index"))?;
            let mask = file.contents[content].mask;
            let source: &[Row] = match (edge.leaf, edge.child) {
                (leaf, -1) if leaf >= 0 => &born,
                (-1, child) if child >= 0 && (child as usize) < index => outputs[child as usize]
                    .as_deref()
                    .ok_or_else(|| invalid("a step output feeds two edges"))?,
                _ => return Err(invalid("edge needs one leaf or one earlier child")),
            };
            let mut links = Vec::new();
            for (position, &row) in source.iter().enumerate() {
                if let Some(&block) = blocks.get(&space.restrict(row, mask)) {
                    let key = space.restrict(row, step.out);
                    let next = u32::try_from(order.len()).map_err(|_| invalid("too many rows"))?;
                    let target = *targets.entry(key).or_insert_with(|| {
                        order.push(key);
                        next
                    });
                    let position = u32::try_from(position).map_err(|_| invalid("too many rows"))?;
                    links.extend([target, position, block]);
                }
            }
            step_links.push(links);
        }
        // A child object feeds one edge: release its rows.
        for edge in &step.edges {
            if edge.child >= 0 {
                outputs[edge.child as usize] = None;
            }
        }
        derived.rows.push(order.len());
        outputs[index] = Some(order);
        derived.links.push(step_links);
    }
    Ok(derived)
}
