// SPDX-License-Identifier: 0BSD

//! Observable columns of plan v4 (DESIGN.md §7.7, §8.4, §8.5).
//!
//! Every observable of the v2 engine is a set of seed columns on one plan:
//! directed colour histories, Born colour components and per-sector partials
//! differ only in the exact seed weights the plan carries per column, and spin
//! sources and the asymmetric Ward mode only in the batch's Born components and
//! block rows. Helicity-resolved outputs are the one exception the seeds see:
//! a resolved column keeps the seeds of its own Born helicity rows and is zero
//! on every other row ([`ColumnRows`], the plan's optional `column_rows`).

use crate::{RusticolError, RusticolResult};

/// Per observable column the Born helicity rows its seeds keep.
#[derive(Clone, Debug, Default)]
pub struct ColumnRows {
    /// `rows × columns` flags, row major; `None` keeps every row of every
    /// column (the inclusive and correlated columns).
    keep: Option<Vec<bool>>,
    columns: usize,
}

impl ColumnRows {
    /// Rows per column as exported (`None` for a column keeping every row).
    pub fn new(
        rows: usize,
        columns: usize,
        file: Option<&[Option<Vec<usize>>]>,
    ) -> RusticolResult<Self> {
        let Some(file) = file else {
            return Ok(Self {
                keep: None,
                columns,
            });
        };
        if file.len() != columns {
            return Err(invalid("one row set per observable column"));
        }
        let mut keep = vec![false; rows * columns];
        for (column, kept) in file.iter().enumerate() {
            match kept {
                None => (0..rows).for_each(|row| keep[row * columns + column] = true),
                Some(kept) => {
                    for &row in kept {
                        if row >= rows {
                            return Err(invalid("column row index"));
                        }
                        keep[row * columns + column] = true;
                    }
                }
            }
        }
        Ok(Self {
            keep: Some(keep),
            columns,
        })
    }

    /// Whether some column leaves some Born helicity row unseeded.
    pub fn restricts(&self) -> bool {
        self.keep
            .as_ref()
            .is_some_and(|keep| keep.iter().any(|&kept| !kept))
    }

    /// Whether `column` seeds Born helicity row `row`.
    #[inline]
    pub fn keeps(&self, row: usize, column: usize) -> bool {
        self.keep
            .as_ref()
            .is_none_or(|keep| keep[row * self.columns + column])
    }
}

fn invalid(message: &str) -> RusticolError {
    RusticolError::compatibility(format!("loop v2 plan: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_default_to_every_row_and_select_resolved_columns() {
        let all = ColumnRows::new(3, 2, None).unwrap();
        assert!((0..3).all(|row| all.keeps(row, 0) && all.keeps(row, 1)));
        assert!(!all.restricts());
        assert!(!ColumnRows::new(3, 1, Some(&[None])).unwrap().restricts());
        let resolved = ColumnRows::new(3, 2, Some(&[None, Some(vec![2])])).unwrap();
        assert!((0..3).all(|row| resolved.keeps(row, 0)));
        assert_eq!(
            (0..3).map(|row| resolved.keeps(row, 1)).collect::<Vec<_>>(),
            [false, false, true]
        );
        assert!(resolved.restricts());
        assert!(ColumnRows::new(3, 2, Some(&[None])).is_err());
        assert!(ColumnRows::new(3, 1, Some(&[Some(vec![3])])).is_err());
    }
}
