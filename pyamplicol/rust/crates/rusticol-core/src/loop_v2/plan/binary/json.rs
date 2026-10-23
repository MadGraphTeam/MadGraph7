// SPDX-License-Identifier: 0BSD

//! The plan v4 JSON exactly as `plan_export.ExportedPlan.json` writes it:
//! `json.dumps(program, separators=(",", ":"))` with Python's defaults, keys
//! in the exporter's insertion order.
//!
//! - integers in decimal, `true`/`false`, `null` for `None`;
//! - strings with `ensure_ascii`: `"` and `\` escaped, `\n \r \t \b \f` in
//!   their short forms, every other character outside `' '..='~'` as a
//!   lowercase `\uXXXX` (UTF-16 surrogate pairs above U+FFFF);
//! - optional keys as the exporter writes them: a step's `rows` and an
//!   edge's `links` only when written (after the other keys, as appended),
//!   `column_rows` only when some column keeps a subset of the Born rows;
//!   `forward`, `merged`, a relabelling, a merged edge's weight and a merged
//!   state's closure as `null` when absent.

use super::super::{
    ClosureFile, ContentFile, FamilyFile, KernelFile, MergedFile, PlanFile, StateFile, TermFile,
};

struct Out(Vec<u8>);

impl Out {
    fn raw(&mut self, bytes: &[u8]) {
        self.0.extend_from_slice(bytes);
    }

    fn byte(&mut self, byte: u8) {
        self.0.push(byte);
    }

    fn uint(&mut self, mut value: u64) {
        let mut digits = [0u8; 20];
        let mut at = digits.len();
        loop {
            at -= 1;
            digits[at] = b'0' + (value % 10) as u8;
            value /= 10;
            if value == 0 {
                break;
            }
        }
        self.raw(&digits[at..]);
    }

    fn int(&mut self, value: i64) {
        if value < 0 {
            self.byte(b'-');
        }
        self.uint(value.unsigned_abs());
    }

    fn usize(&mut self, value: usize) {
        self.uint(value as u64);
    }

    fn boolean(&mut self, value: bool) {
        self.raw(if value { b"true" } else { b"false" });
    }

    fn string(&mut self, text: &str) {
        self.byte(b'"');
        for c in text.chars() {
            match c {
                '"' => self.raw(b"\\\""),
                '\\' => self.raw(b"\\\\"),
                '\n' => self.raw(b"\\n"),
                '\r' => self.raw(b"\\r"),
                '\t' => self.raw(b"\\t"),
                '\u{8}' => self.raw(b"\\b"),
                '\u{c}' => self.raw(b"\\f"),
                ' '..='~' => self.byte(c as u8),
                _ => {
                    let mut units = [0u16; 2];
                    for unit in c.encode_utf16(&mut units) {
                        self.raw(format!("\\u{unit:04x}").as_bytes());
                    }
                }
            }
        }
        self.byte(b'"');
    }

    /// `"name":` preceded by `{` for the first key and `,` otherwise.
    fn key(&mut self, first: bool, name: &str) {
        self.byte(if first { b'{' } else { b',' });
        self.byte(b'"');
        self.raw(name.as_bytes());
        self.raw(b"\":");
    }

    fn list<T>(&mut self, items: &[T], mut item: impl FnMut(&mut Self, &T)) {
        self.byte(b'[');
        for (k, value) in items.iter().enumerate() {
            if k > 0 {
                self.byte(b',');
            }
            item(self, value);
        }
        self.byte(b']');
    }

    fn usizes(&mut self, items: &[usize]) {
        self.list(items, |out, &v| out.usize(v));
    }

    fn option<T>(&mut self, value: &Option<T>, item: impl FnOnce(&mut Self, &T)) {
        match value {
            Some(value) => item(self, value),
            None => self.raw(b"null"),
        }
    }

    fn relabel(&mut self, relabel: &Option<(i8, u64)>) {
        self.option(relabel, |out, &(sign, mask)| {
            out.byte(b'[');
            out.int(i64::from(sign));
            out.byte(b',');
            out.uint(mask);
            out.byte(b']');
        });
    }

    fn pair(&mut self, [re, im]: &[String; 2]) {
        self.byte(b'[');
        self.string(re);
        self.byte(b',');
        self.string(im);
        self.byte(b']');
    }

    fn weights(&mut self, items: &[(usize, String, String)]) {
        self.list(items, |out, (index, re, im)| {
            out.byte(b'[');
            out.usize(*index);
            out.byte(b',');
            out.string(re);
            out.byte(b',');
            out.string(im);
            out.byte(b']');
        });
    }

    fn inputs(&mut self, items: &[(String, usize, usize)]) {
        self.list(items, |out, (kind, a, b)| {
            out.byte(b'[');
            out.string(kind);
            out.byte(b',');
            out.usize(*a);
            out.byte(b',');
            out.usize(*b);
            out.byte(b']');
        });
    }
}

/// The plan JSON of `file`.
pub(super) fn write(file: &PlanFile) -> Vec<u8> {
    let mut out = Out(Vec::new());
    let o = &mut out;
    o.key(true, "schema");
    o.uint(u64::from(file.schema));
    o.key(false, "legs");
    o.usize(file.legs);
    o.key(false, "rows");
    o.usize(file.rows);
    o.key(false, "helicities");
    o.list(&file.helicities, |out, leg| {
        out.list(leg, |out, &h| out.int(i64::from(h)))
    });
    o.key(false, "columns");
    o.usize(file.columns);
    o.key(false, "average");
    o.uint(file.average);
    o.key(false, "born_components");
    o.usize(file.born_components);
    o.key(false, "constants");
    o.usize(file.constants);
    o.key(false, "masses");
    o.usize(file.masses);
    o.key(false, "block_rows");
    o.usizes(&file.block_rows);
    o.key(false, "kernels");
    o.list(&file.kernels, kernel);
    o.key(false, "contents");
    o.list(&file.contents, content);
    o.key(false, "factors");
    o.list(&file.factors, |out, &(content, shift)| {
        out.byte(b'[');
        out.usize(content);
        out.byte(b',');
        out.uint(shift);
        out.byte(b']');
    });
    o.key(false, "seed_classes");
    o.list(&file.seed_classes, |out, seed| {
        out.key(true, "columns");
        out.list(&seed.columns, |out, column| out.weights(column));
        out.byte(b'}');
    });
    o.key(false, "seeds");
    o.list(&file.seeds, |out, key| {
        out.key(true, "classes");
        out.weights(&key.classes);
        out.byte(b'}');
    });
    o.key(false, "families");
    o.list(&file.families, family);
    o.key(false, "forward");
    o.option(&file.forward, |out, forward| {
        out.key(true, "states");
        out.list(&forward.states, state);
        out.byte(b'}');
    });
    o.key(false, "merged");
    o.option(&file.merged, merged);
    if let Some(columns) = &file.column_rows {
        o.key(false, "column_rows");
        o.list(columns, |out, rows| {
            out.option(rows, |out, rows| out.usizes(rows))
        });
    }
    o.byte(b'}');
    out.0
}

fn kernel(out: &mut Out, kernel: &KernelFile) {
    out.key(true, "inputs");
    out.inputs(&kernel.inputs);
    out.key(false, "outputs");
    out.list(&kernel.outputs, |out, terms| {
        out.list(terms, |out, (constant, monomial)| {
            out.byte(b'[');
            out.usize(*constant);
            out.byte(b',');
            out.list(monomial, |out, &(input, power)| {
                out.byte(b'[');
                out.usize(input);
                out.byte(b',');
                out.uint(u64::from(power));
                out.byte(b']');
            });
            out.byte(b']');
        })
    });
    out.byte(b'}');
}

fn content(out: &mut Out, content: &ContentFile) {
    out.key(true, "d_in");
    out.usize(content.d_in);
    out.key(false, "d_out");
    out.usize(content.d_out);
    out.key(false, "degree");
    out.usize(content.degree);
    out.key(false, "rows");
    out.usize(content.rows);
    out.key(false, "mask");
    out.uint(content.mask);
    out.key(false, "row_helicities");
    out.list(&content.row_helicities, |out, row| {
        out.list(row, |out, &h| out.int(i64::from(h)))
    });
    out.key(false, "fused");
    out.list(&content.fused, |out, entry| {
        out.list(entry, |out, &v| out.uint(u64::from(v)))
    });
    out.key(false, "terms");
    out.list(&content.terms, term);
    out.byte(b'}');
}

fn term(out: &mut Out, term: &TermFile) {
    out.key(true, "slots");
    out.list(&term.slots, |out, &mask| out.uint(mask));
    out.key(false, "dimensions");
    out.usizes(&term.dimensions);
    out.key(false, "bindings");
    out.list(&term.bindings, |out, binding| {
        out.key(true, "kernel");
        out.usize(binding.kernel);
        out.key(false, "inputs");
        out.inputs(&binding.inputs);
        out.key(false, "entries");
        out.usizes(&binding.entries);
        out.byte(b'}');
    });
    out.key(false, "vertex");
    out.int(term.vertex);
    out.key(false, "split");
    out.list(&term.split, |out, split| {
        out.key(true, "kernel");
        out.usize(split.kernel);
        out.key(false, "inputs");
        out.inputs(&split.inputs);
        out.key(false, "entries");
        out.list(&split.entries, |out, &(a, b)| {
            out.byte(b'[');
            out.usize(a);
            out.byte(b',');
            out.usize(b);
            out.byte(b']');
        });
        out.byte(b'}');
    });
    out.key(false, "rows");
    out.list(&term.rows, |out, (row, blocks)| {
        out.byte(b'[');
        out.usize(*row);
        out.byte(b',');
        out.usizes(blocks);
        out.byte(b']');
    });
    out.byte(b'}');
}

fn family(out: &mut Out, family: &FamilyFile) {
    out.key(true, "sequence");
    out.list(&family.sequence, |out, &(mask, mass)| {
        out.byte(b'[');
        out.uint(mask);
        out.byte(b',');
        out.usize(mass);
        out.byte(b']');
    });
    out.key(false, "rank");
    out.usize(family.rank);
    out.key(false, "orders");
    out.usizes(&family.orders);
    out.key(false, "leaves");
    out.list(&family.leaves, |out, leaf| {
        out.key(true, "seed");
        out.usize(leaf.seed);
        out.key(false, "weight");
        out.string(&leaf.weight);
        out.byte(b'}');
    });
    out.key(false, "steps");
    out.list(&family.steps, |out, step| {
        out.key(true, "seeded");
        out.boolean(step.seeded);
        out.key(false, "trace");
        out.int(step.trace);
        out.key(false, "out");
        out.uint(step.out);
        out.key(false, "d_alpha");
        out.usize(step.d_alpha);
        out.key(false, "d_out");
        out.usize(step.d_out);
        out.key(false, "degree");
        out.usize(step.degree);
        out.key(false, "rank_in");
        out.usize(step.rank_in);
        out.key(false, "edges");
        out.list(&step.edges, |out, edge| {
            out.key(true, "leaf");
            out.int(edge.leaf);
            out.key(false, "child");
            out.int(edge.child);
            out.key(false, "factor");
            out.usize(edge.factor);
            if let Some(links) = &edge.links {
                out.key(false, "links");
                out.list(links, |out, &v| out.uint(u64::from(v)));
            }
            out.byte(b'}');
        });
        if let Some(rows) = step.rows {
            out.key(false, "rows");
            out.usize(rows);
        }
        out.byte(b'}');
    });
    out.byte(b'}');
}

fn state(out: &mut Out, state: &StateFile) {
    out.key(true, "parent");
    out.int(state.parent);
    out.key(false, "factor");
    out.usize(state.factor);
    out.key(false, "seeded");
    out.boolean(state.seeded);
    out.key(false, "rows");
    out.usize(state.rows);
    out.key(false, "d_alpha");
    out.usize(state.d_alpha);
    out.key(false, "d_out");
    out.usize(state.d_out);
    out.key(false, "degree");
    out.usize(state.degree);
    out.key(false, "rank");
    out.usize(state.rank);
    out.key(false, "links");
    out.list(&state.links, |out, &v| out.uint(u64::from(v)));
    out.key(false, "closures");
    out.list(&state.closures, closure);
    out.byte(b'}');
}

fn closure(out: &mut Out, closure: &ClosureFile) {
    out.key(true, "family");
    out.usize(closure.family);
    out.key(false, "accumulator");
    out.usize(closure.accumulator);
    out.key(false, "seed");
    out.usize(closure.seed);
    out.key(false, "weight");
    out.string(&closure.weight);
    out.key(false, "rows");
    out.list(&closure.rows, |out, &v| out.int(v));
    out.key(false, "relabel");
    out.relabel(&closure.relabel);
    out.byte(b'}');
}

fn merged(out: &mut Out, merged: &MergedFile) {
    out.key(true, "families");
    out.usizes(&merged.families);
    out.key(false, "entries");
    out.list(&merged.entries, |out, entry| {
        out.key(true, "seed");
        out.usize(entry.seed);
        out.key(false, "weight");
        out.pair(&entry.weight);
        out.byte(b'}');
    });
    out.key(false, "states");
    out.list(&merged.states, |out, state| {
        out.key(true, "seeded");
        out.boolean(state.seeded);
        out.key(false, "rows");
        out.usize(state.rows);
        out.key(false, "d_alpha");
        out.usize(state.d_alpha);
        out.key(false, "d_out");
        out.usize(state.d_out);
        out.key(false, "degree");
        out.usize(state.degree);
        out.key(false, "rank_in");
        out.usize(state.rank_in);
        out.key(false, "edges");
        out.list(&state.edges, |out, edge| {
            out.key(true, "entry");
            out.int(edge.entry);
            out.key(false, "parent");
            out.int(edge.parent);
            out.key(false, "factor");
            out.usize(edge.factor);
            out.key(false, "links");
            out.list(&edge.links, |out, &v| out.uint(u64::from(v)));
            out.key(false, "weight");
            out.option(&edge.weight, |out, weight| out.pair(weight));
            out.byte(b'}');
        });
        out.key(false, "closure");
        out.option(&state.closure, |out, closure| {
            out.key(true, "family");
            out.usize(closure.family);
            out.key(false, "accumulator");
            out.usize(closure.accumulator);
            out.key(false, "relabel");
            out.relabel(&closure.relabel);
            out.byte(b'}');
        });
        out.byte(b'}');
    });
    out.key(false, "chunks");
    out.usizes(&merged.chunks);
    out.byte(b'}');
}

#[cfg(test)]
mod tests {
    use super::Out;

    #[test]
    fn strings_follow_ensure_ascii() {
        let mut out = Out(Vec::new());
        out.string("a\"b\\c\n\r\t\u{8}\u{c}\u{1}\u{7f}\u{e9}\u{1f600}/~ ");
        assert_eq!(
            String::from_utf8(out.0).unwrap(),
            r#""a\"b\\c\n\r\t\b\f\u0001\u007f\u00e9\ud83d\ude00/~ ""#
        );
    }

    #[test]
    fn integers_in_decimal() {
        let mut out = Out(Vec::new());
        out.int(i64::MIN);
        out.byte(b' ');
        out.uint(u64::MAX);
        out.byte(b' ');
        out.int(0);
        assert_eq!(
            String::from_utf8(out.0).unwrap(),
            "-9223372036854775808 18446744073709551615 0"
        );
    }
}
