// SPDX-License-Identifier: 0BSD

//! Interned coupling-order vectors (`loops/v2/families.py` `Orders`).
//!
//! An order vector is a list of `(name, power)` with names given as ids of the
//! generator input's sorted name table and zero powers dropped, exactly as
//! `add_orders` normalises them. Ids are assigned first-seen; comparisons of
//! the vectors themselves (`Vec<(u16, i64)>`) follow Python's tuple order
//! because name ids are ranks of the sorted names (any number of names up to
//! the `u16` id width, which the input's deserialisation checks).

use super::FastMap;

pub type OrderVec = Vec<(u16, i64)>;

/// Interned order vectors with memoised sums.
#[derive(Debug, Default, Clone)]
pub struct OrderSpace {
    vectors: Vec<OrderVec>,
    index: FastMap<OrderVec, u32>,
    sums: FastMap<(u32, u32), u32>,
}

impl OrderSpace {
    pub fn new() -> Self {
        let mut space = Self::default();
        space.index.insert(Vec::new(), Self::EMPTY);
        space.vectors.push(Vec::new());
        space
    }

    /// The id of the empty order vector.
    pub const EMPTY: u32 = 0;

    /// The id of `orders` (normalised); a new vector beyond the `u32` id
    /// width is refused, never wrapped.
    pub fn intern(&mut self, mut orders: OrderVec) -> Result<u32, String> {
        orders.retain(|&(_, power)| power != 0);
        orders.sort_unstable();
        if let Some(&id) = self.index.get(&orders) {
            return Ok(id);
        }
        let id = crate::narrow::checked_str::<u32>(self.vectors.len(), "coupling-order vector id")?;
        self.index.insert(orders.clone(), id);
        self.vectors.push(orders);
        Ok(id)
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn get(&self, id: u32) -> &OrderVec {
        &self.vectors[id as usize]
    }

    /// `add_orders(a, b)`.
    pub fn add(&mut self, a: u32, b: u32) -> Result<u32, String> {
        if a == Self::EMPTY {
            return Ok(b);
        }
        if b == Self::EMPTY {
            return Ok(a);
        }
        let key = if a <= b { (a, b) } else { (b, a) };
        if let Some(&id) = self.sums.get(&key) {
            return Ok(id);
        }
        let (x, y) = (&self.vectors[a as usize], &self.vectors[b as usize]);
        let mut out = Vec::with_capacity(x.len() + y.len());
        let (mut i, mut j) = (0, 0);
        while i < x.len() || j < y.len() {
            if j == y.len() || (i < x.len() && x[i].0 < y[j].0) {
                out.push(x[i]);
                i += 1;
            } else if i == x.len() || y[j].0 < x[i].0 {
                out.push(y[j]);
                j += 1;
            } else {
                out.push((x[i].0, x[i].1 + y[j].1));
                i += 1;
                j += 1;
            }
        }
        let id = self.intern(out)?;
        self.sums.insert(key, id);
        Ok(id)
    }

    /// `orders_within(orders, cap)`: every power at most the cap's (absent = 0).
    pub fn within(&self, orders: u32, cap: &OrderVec) -> bool {
        self.vectors[orders as usize].iter().all(|&(name, power)| {
            power
                <= cap
                    .iter()
                    .find(|&&(other, _)| other == name)
                    .map_or(0, |&(_, bound)| bound)
        })
    }

    /// Whether a name with a non-zero power is one of `names` (sorted).
    pub fn names_meet(&self, orders: u32, names: &[u16]) -> bool {
        self.vectors[orders as usize]
            .iter()
            .any(|(name, _)| names.binary_search(name).is_ok())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_are_normalised_and_memoised() {
        let mut space = OrderSpace::new();
        assert_eq!(space.intern(Vec::new()).unwrap(), OrderSpace::EMPTY);
        assert_eq!(space.intern(vec![(4, 0)]).unwrap(), OrderSpace::EMPTY);
        let a = space.intern(vec![(1, 2), (0, 1)]).unwrap();
        let b = space.intern(vec![(0, -1), (2, 1)]).unwrap();
        let c = space.add(a, b).unwrap();
        assert_eq!(space.get(c), &vec![(1, 2), (2, 1)]);
        assert_eq!(space.add(b, a).unwrap(), c);
        assert_eq!(space.add(OrderSpace::EMPTY, a).unwrap(), a);
        assert!(space.within(c, &vec![(1, 2), (2, 3)]));
        assert!(!space.within(c, &vec![(1, 1), (2, 3)]));
        assert!(space.names_meet(c, &[2]));
        assert!(space.names_meet(c, &[0, 1]));
        assert!(!space.names_meet(c, &[0, 3]));
        // Name ids past any bitmask width.
        let wide = space.intern(vec![(70, 1), (300, 2)]).unwrap();
        assert!(space.names_meet(wide, &[300]));
        assert!(!space.names_meet(wide, &[6, 44]));
    }
}
