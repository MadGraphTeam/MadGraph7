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
//!
//! **Dressed rows** (amplitude mode, route R2; M-LIT LIT-R2): a leaf's one
//! row is the empty row (no leg), and an edge links each source row to every
//! row of its content, the target being their union (the rule of schedule F,
//! `plan_export.forward`): a step's rows are the helicities of the legs it
//! has dressed (those it leaves open, `out`, excluded), numbered first-seen
//! over its edges, their source rows and the content rows in order. A trace
//! step's rows cover every leg, so each is a Born row: its key.

use super::hash::WordMap;
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
    /// Per leg the key bits of its digit when every leg's helicity count is
    /// a power of two (then a row restricts by one mask: [`Self::key_bits`]).
    bits: Option<Vec<u64>>,
    all: u64,
    rows: usize,
}

impl RowSpace {
    pub(super) fn new(file: &PlanFile) -> RusticolResult<Self> {
        let legs = file.legs;
        let all = super::super::bounds::check_legs(legs).map_err(invalid)?;
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
        // Power-of-two radices: each stride is a power of two too (a product
        // of them), so a digit is the key's bits `(radix - 1) * stride`.
        let bits = radix.iter().all(|r| r.is_power_of_two()).then(|| {
            radix
                .iter()
                .zip(&stride)
                .map(|(r, s)| (r - 1) * s)
                .collect()
        });
        Ok(Self {
            stride,
            radix,
            digits,
            bits,
            all,
            rows: file.rows,
        })
    }

    /// The key bits of the legs in `mask` (power-of-two radices only, else
    /// `None`): a row restricted to `mask` keeps exactly these bits of its
    /// key ([`Self::restrict_with`]).
    fn key_bits(&self, mask: u64) -> Option<u64> {
        let bits = self.bits.as_ref()?;
        let mut out = 0;
        let mut legs = mask & self.all;
        while legs != 0 {
            out |= bits[legs.trailing_zeros() as usize];
            legs &= legs - 1;
        }
        Some(out)
    }

    /// [`Self::restrict`] with `bits = self.key_bits(mask)`: a row's key
    /// holds digits of its own legs only, so with power-of-two radices the
    /// restricted key is the key masked (the same row, no division).
    #[inline]
    fn restrict_with(&self, row: Row, mask: u64, bits: Option<u64>) -> Row {
        match bits {
            Some(bits) => (row.0 & mask, row.1 & bits),
            None => self.restrict(row, mask),
        }
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
    pub(super) fn content_rows(&self, content: &ContentFile) -> RusticolResult<WordMap<Row, u32>> {
        let mask = content.mask;
        require(mask & !self.all == 0, "content legs outside the process")?;
        require(
            content.row_helicities.len() == content.rows,
            "content row helicities differ from its rows",
        )?;
        let mut out = WordMap::with_capacity_and_hasher(content.rows, Default::default());
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
/// `(target, source position, block)`; with dressed rows, per trace step the
/// Born row of each of its rows.
pub(super) struct DerivedSteps {
    pub(super) rows: Vec<usize>,
    pub(super) links: Vec<Vec<Vec<u32>>>,
    pub(super) born: Vec<Option<Vec<u32>>>,
}

/// Each content's rows in block-row order (the inverse of
/// [`RowSpace::content_rows`]): the rows a dressed edge links every source
/// row to.
pub(super) fn content_lists(content_rows: &[std::sync::Arc<WordMap<Row, u32>>]) -> Vec<Vec<Row>> {
    content_rows
        .iter()
        .map(|rows| {
            let mut list = vec![(0, 0); rows.len()];
            for (&row, &index) in rows.iter() {
                list[index as usize] = row;
            }
            list
        })
        .collect()
}

/// The rows and row links of one family's steps, in the exporter's order:
/// seeded (`dressed` None), or with dressed rows over `dressed`, each
/// content's rows in block-row order ([`content_lists`]; module docs: a leaf
/// seeds the empty row, each source row of an edge is joined with every row
/// of its content, a step's rows cover exactly the legs it has dressed, so a
/// trace step's rows are Born rows). One walk serves both: only a leaf's
/// rows and the rows an edge links a source row to depend on the rule.
pub(super) fn derive(
    family: &FamilyFile,
    file: &PlanFile,
    space: &RowSpace,
    content_rows: &[std::sync::Arc<WordMap<Row, u32>>],
    dressed: Option<&[Vec<Row>]>,
) -> RusticolResult<DerivedSteps> {
    let mut outputs: Vec<Option<Vec<Row>>> = vec![None; family.steps.len()];
    let leaf: Vec<Row> = match dressed {
        Some(_) => vec![(0, 0)],
        None => (0..space.rows as u64).map(|key| (space.all, key)).collect(),
    };
    let mut derived = DerivedSteps {
        rows: Vec::with_capacity(family.steps.len()),
        links: Vec::with_capacity(family.steps.len()),
        born: Vec::with_capacity(if dressed.is_some() {
            family.steps.len()
        } else {
            0
        }),
    };
    for (index, step) in family.steps.iter().enumerate() {
        // The legs a dressed step's rows cover.
        let covered = space.all & !step.out;
        let out_bits = space.key_bits(step.out);
        let mut targets: WordMap<Row, u32> = WordMap::default();
        let mut order: Vec<Row> = Vec::new();
        let mut step_links = Vec::with_capacity(step.edges.len());
        for edge in &step.edges {
            let content = file
                .factors
                .get(edge.factor)
                .ok_or_else(|| invalid("edge factor index"))?
                .0;
            require(content < content_rows.len(), "factor content index")?;
            let mask = file.contents[content].mask;
            let mask_bits = space.key_bits(mask);
            let source: &[Row] = match (edge.leaf, edge.child) {
                (leaf_index, -1) if leaf_index >= 0 => &leaf,
                (-1, child) if child >= 0 && (child as usize) < index => outputs[child as usize]
                    .as_deref()
                    .ok_or_else(|| invalid("a step output feeds two edges"))?,
                _ => return Err(invalid("edge needs one leaf or one earlier child")),
            };
            // Seeded: at most one link per source row; dressed: one per
            // (source row, content row).
            let mut links = Vec::with_capacity(
                3 * source.len() * dressed.map_or(1, |lists| lists[content].len()),
            );
            let mut link = |row: Row, position: usize, block: u32| -> RusticolResult<()> {
                let next = u32::try_from(order.len()).map_err(|_| invalid("too many rows"))?;
                let target = *targets.entry(row).or_insert_with(|| {
                    order.push(row);
                    next
                });
                let position = u32::try_from(position).map_err(|_| invalid("too many rows"))?;
                links.extend([target, position, block]);
                Ok(())
            };
            for (position, &row) in source.iter().enumerate() {
                match dressed {
                    // The block row of the same helicities on the content's
                    // legs, restricted to the legs the step leaves open.
                    None => {
                        if let Some(&block) =
                            content_rows[content].get(&space.restrict_with(row, mask, mask_bits))
                        {
                            link(
                                space.restrict_with(row, step.out, out_bits),
                                position,
                                block,
                            )?;
                        }
                    }
                    // Every row of the content, joined with the source row.
                    Some(lists) => {
                        let (legs, key) = row;
                        require(
                            legs & mask == 0 && legs | mask == covered,
                            "a dressed edge does not dress the step's legs",
                        )?;
                        for (block, &(_, block_key)) in lists[content].iter().enumerate() {
                            let block =
                                u32::try_from(block).map_err(|_| invalid("too many rows"))?;
                            link((covered, key + block_key), position, block)?;
                        }
                    }
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
        if dressed.is_some() {
            derived.born.push(if step.trace >= 0 {
                require(covered == space.all, "a dressed trace step with open legs")?;
                Some(
                    order
                        .iter()
                        .map(|&(_, key)| u32::try_from(key).map_err(|_| invalid("too many rows")))
                        .collect::<RusticolResult<_>>()?,
                )
            } else {
                None
            });
        }
        outputs[index] = Some(order);
        derived.links.push(step_links);
    }
    Ok(derived)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row space of legs with these helicity counts (no plan file).
    fn space(radix: &[u64]) -> RowSpace {
        let legs = radix.len();
        let mut stride = vec![0; legs];
        let mut size = 1;
        for leg in (0..legs).rev() {
            stride[leg] = size;
            size *= radix[leg];
        }
        let bits = radix.iter().all(|r| r.is_power_of_two()).then(|| {
            radix
                .iter()
                .zip(&stride)
                .map(|(r, s)| (r - 1) * s)
                .collect()
        });
        RowSpace {
            stride,
            radix: radix.to_vec(),
            digits: Vec::new(),
            bits,
            all: (1 << legs) - 1,
            rows: size as usize,
        }
    }

    #[test]
    fn a_masked_restriction_is_the_divided_one() {
        for radix in [
            &[2, 2, 2, 2, 2][..],
            &[4, 2, 1, 8],
            &[3, 2, 2],
            &[2, 3, 1, 2],
        ] {
            let space = space(radix);
            assert_eq!(
                space.bits.is_some(),
                radix.iter().all(|r| r.is_power_of_two())
            );
            for key in 0..space.rows as u64 {
                for first in 0..=space.all {
                    // A Born row and a row already restricted to `first`.
                    for row in [(space.all, key), space.restrict((space.all, key), first)] {
                        for mask in 0..=space.all {
                            assert_eq!(
                                space.restrict_with(row, mask, space.key_bits(mask)),
                                space.restrict(row, mask),
                                "radix {radix:?} row {row:?} mask {mask:b}"
                            );
                        }
                    }
                }
            }
        }
    }
}
