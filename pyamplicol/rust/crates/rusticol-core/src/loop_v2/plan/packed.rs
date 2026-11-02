// SPDX-License-Identifier: 0BSD

//! Packed row links of the split parts (M-FINAL MF-8G-2 item 4).
//!
//! The split parts' row links are the bulk of a loaded plan (0.95 of 1.33
//! GiB for g g > 5g at leading colour as `RowLink` triples of `u32`). A
//! part's links are stored column by column (outputs, inputs, blocks), each
//! column at the narrowest width (1, 2 or 4 bytes) its largest value fits:
//! object rows and term rows below 256 take one byte each, as for every
//! one-column plan up to eight legs of two helicities, so a link takes 3
//! bytes instead of 12; larger values widen their column (no cap). The
//! dressing unpacks a part into a scratch list before its kernel
//! ([`PackedLinks::unpack_into`]): a widening copy per part and lane chunk,
//! small beside the kernel's lane work per link, and the kernels read the
//! same links in the same order (values bitwise unchanged).

use crate::loop_v2::kernels::RowLink;

/// Row links packed column by column (module docs).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PackedLinks {
    len: usize,
    /// Bytes per value of the output, input and block columns.
    widths: [u8; 3],
    bytes: Box<[u8]>,
}

fn width(max: u32) -> u8 {
    if max <= u32::from(u8::MAX) {
        1
    } else if max <= u32::from(u16::MAX) {
        2
    } else {
        4
    }
}

/// Set `field` of each link to its value in `column` (`w` bytes each).
fn fill(out: &mut [RowLink], column: &[u8], w: u8, field: impl Fn(&mut RowLink) -> &mut u32) {
    match w {
        1 => out
            .iter_mut()
            .zip(column)
            .for_each(|(l, &v)| *field(l) = u32::from(v)),
        2 => out
            .iter_mut()
            .zip(column.chunks_exact(2))
            .for_each(|(l, v)| *field(l) = u32::from(u16::from_le_bytes([v[0], v[1]]))),
        _ => out
            .iter_mut()
            .zip(column.chunks_exact(4))
            .for_each(|(l, v)| *field(l) = u32::from_le_bytes([v[0], v[1], v[2], v[3]])),
    }
}

impl PackedLinks {
    /// The links `links()` yields, packed (two passes: the column widths,
    /// then the values; nothing else is allocated).
    pub fn collect<I: Iterator<Item = RowLink>>(links: impl Fn() -> I) -> Self {
        let mut len = 0usize;
        let mut max = [0u32; 3];
        for link in links() {
            len += 1;
            max[0] = max[0].max(link.output);
            max[1] = max[1].max(link.input);
            max[2] = max[2].max(link.block);
        }
        let widths = max.map(width);
        let starts = [
            0,
            len * usize::from(widths[0]),
            len * usize::from(widths[0] + widths[1]),
        ];
        let total = len * usize::from(widths[0] + widths[1] + widths[2]);
        let mut bytes = vec![0u8; total].into_boxed_slice();
        for (k, link) in links().enumerate() {
            for (c, value) in [link.output, link.input, link.block]
                .into_iter()
                .enumerate()
            {
                let w = usize::from(widths[c]);
                let at = starts[c] + k * w;
                bytes[at..at + w].copy_from_slice(&value.to_le_bytes()[..w]);
            }
        }
        Self { len, widths, bytes }
    }

    /// The links of a list, packed.
    pub fn from_links(links: &[RowLink]) -> Self {
        Self::collect(|| links.iter().copied())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Column `c`'s bytes.
    fn column(&self, c: usize) -> &[u8] {
        let start: usize = self.widths[..c].iter().map(|&w| usize::from(w)).sum();
        let w = usize::from(self.widths[c]);
        &self.bytes[start * self.len..(start + w) * self.len]
    }

    /// Link `k`.
    pub fn get(&self, k: usize) -> RowLink {
        let value = |c: usize| {
            let w = usize::from(self.widths[c]);
            let mut word = [0u8; 4];
            word[..w].copy_from_slice(&self.column(c)[k * w..(k + 1) * w]);
            u32::from_le_bytes(word)
        };
        RowLink {
            output: value(0),
            input: value(1),
            block: value(2),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = RowLink> + '_ {
        (0..self.len).map(|k| self.get(k))
    }

    /// The links unpacked into `out` (cleared first; its capacity is reused).
    pub fn unpack_into<'a>(&self, out: &'a mut Vec<RowLink>) -> &'a [RowLink] {
        out.clear();
        out.resize(
            self.len,
            RowLink {
                output: 0,
                input: 0,
                block: 0,
            },
        );
        fill(out, self.column(0), self.widths[0], |l| &mut l.output);
        fill(out, self.column(1), self.widths[1], |l| &mut l.input);
        fill(out, self.column(2), self.widths[2], |l| &mut l.block);
        out
    }

    /// Keep the links `keep` accepts, in order (`keep` is called twice per
    /// link: it must be a function of the link).
    pub fn retain(&mut self, mut keep: impl FnMut(&RowLink) -> bool) {
        let old = std::mem::take(self);
        let keep = std::cell::RefCell::new(&mut keep);
        *self = Self::collect(|| old.iter().filter(|link| (keep.borrow_mut())(link)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link(output: u32, input: u32, block: u32) -> RowLink {
        RowLink {
            output,
            input,
            block,
        }
    }

    fn triples(links: &[RowLink]) -> Vec<(u32, u32, u32)> {
        links.iter().map(|l| (l.output, l.input, l.block)).collect()
    }

    #[test]
    fn links_pack_at_the_narrowest_widths_and_unpack_unchanged() {
        let cases = [
            vec![],
            vec![link(0, 0, 0)],
            vec![link(3, 255, 1), link(200, 7, 0)],
            vec![
                link(256, 1, 65_535),
                link(0, 70_000, 2),
                link(u32::MAX, 0, 65_536),
            ],
        ];
        let widths = [[1, 1, 1], [1, 1, 1], [1, 1, 1], [4, 4, 4]];
        let mut scratch = vec![link(9, 9, 9); 5];
        for (links, expected) in cases.iter().zip(widths) {
            let packed = PackedLinks::from_links(links);
            assert_eq!(packed.len(), links.len());
            assert_eq!(packed.widths, expected);
            assert_eq!(
                packed.bytes.len(),
                links.len() * 3 * usize::from(expected[0])
            );
            assert_eq!(triples(packed.unpack_into(&mut scratch)), triples(links));
            assert_eq!(triples(&packed.iter().collect::<Vec<_>>()), triples(links));
        }
        let mixed = PackedLinks::from_links(&[link(300, 2, 1), link(1, 3, 70_000)]);
        assert_eq!(mixed.widths, [2, 1, 4]);
        assert_eq!(mixed.get(1).block, 70_000);
    }

    #[test]
    fn retain_keeps_the_accepted_links_in_order_and_narrows() {
        let mut packed = PackedLinks::from_links(&[link(1, 2, 3), link(400, 5, 6), link(7, 8, 9)]);
        assert_eq!(packed.widths, [2, 1, 1]);
        packed.retain(|l| l.output < 100);
        assert_eq!(
            triples(&packed.iter().collect::<Vec<_>>()),
            [(1, 2, 3), (7, 8, 9)]
        );
        assert_eq!(packed.widths, [1, 1, 1]);
        packed.retain(|_| false);
        assert!(packed.is_empty());
    }
}
