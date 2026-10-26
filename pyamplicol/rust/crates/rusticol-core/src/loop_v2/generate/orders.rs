// SPDX-License-Identifier: 0BSD

//! Interned coupling-order vectors (`loops/v2/families.py` `Orders`).
//!
//! An order vector is a list of `(name, power)` with names given as ids of the
//! generator input's sorted name table and zero powers dropped, exactly as
//! `add_orders` normalises them. Ids are assigned first-seen; comparisons of
//! the vectors themselves (`Vec<(u16, i64)>`) follow Python's tuple order
//! because name ids are ranks of the sorted names.

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
        space.intern(Vec::new());
        space
    }

    /// The id of the empty order vector.
    pub const EMPTY: u32 = 0;

    pub fn intern(&mut self, mut orders: OrderVec) -> u32 {
        orders.retain(|&(_, power)| power != 0);
        orders.sort_unstable();
        if let Some(&id) = self.index.get(&orders) {
            return id;
        }
        let id = u32::try_from(self.vectors.len()).expect("order vectors fit u32");
        self.index.insert(orders.clone(), id);
        self.vectors.push(orders);
        id
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
    pub fn add(&mut self, a: u32, b: u32) -> u32 {
        if a == Self::EMPTY {
            return b;
        }
        if b == Self::EMPTY {
            return a;
        }
        let key = if a <= b { (a, b) } else { (b, a) };
        if let Some(&id) = self.sums.get(&key) {
            return id;
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
        let id = self.intern(out);
        self.sums.insert(key, id);
        id
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

    /// Bitset of the names with a non-zero power.
    pub fn names(&self, orders: u32) -> u64 {
        self.vectors[orders as usize]
            .iter()
            .fold(0, |bits, &(name, _)| bits | (1u64 << name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_are_normalised_and_memoised() {
        let mut space = OrderSpace::new();
        let a = space.intern(vec![(1, 2), (0, 1)]);
        let b = space.intern(vec![(0, -1), (2, 1)]);
        let c = space.add(a, b);
        assert_eq!(space.get(c), &vec![(1, 2), (2, 1)]);
        assert_eq!(space.add(b, a), c);
        assert_eq!(space.add(OrderSpace::EMPTY, a), a);
        assert!(space.within(c, &vec![(1, 2), (2, 3)]));
        assert!(!space.within(c, &vec![(1, 1), (2, 3)]));
        assert_eq!(space.names(c), 0b110);
    }
}
