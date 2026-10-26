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
//!   `column_rows` only when some column keeps a subset of the Born rows,
//!   `column_weights` only when columns share a seed-weight column;
//!   `forward`, `merged`, a relabelling, a merged edge's weight and a merged
//!   state's closure as `null` when absent;
//! - every other key always.
//!
//! Every record is destructured whole (no `..`), so a field added to the
//! loader's file form does not compile until the writer writes it.

use super::super::{
    BindingFile, ClosureFile, ContentFile, EdgeFile, EntryFile, FamilyFile, ForwardFile,
    KernelFile, LeafFile, MergedClosureFile, MergedEdgeFile, MergedFile, MergedStateFile, PlanFile,
    SeedFile, SeedKeyFile, SplitFile, StateFile, StepFile, TermFile,
};

/// The keys the writer omits when their value is absent, as (record, key):
/// the plan's `column_rows` and `column_weights`, a step's `rows` and an
/// edge's `links`, where `json.dumps` of a record holding them as `None`
/// would write `null`. The writer writes every other key of every record, an
/// absent optional value as `null`. A records reader that accepts only what this writer writes back
/// byte for byte (`V2PlanModel.from_program`) refuses `None` under these keys
/// and the absence of any other key. The record is the file form's type name
/// (`serde`'s struct name).
pub const OMITTED_WHEN_ABSENT: [(&str, &str); 4] = [
    ("PlanFile", "column_rows"),
    ("PlanFile", "column_weights"),
    ("StepFile", "rows"),
    ("EdgeFile", "links"),
];

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
    let PlanFile {
        schema,
        legs,
        rows,
        helicities,
        columns,
        average,
        born_components,
        constants,
        masses,
        block_rows,
        kernels,
        contents,
        factors,
        seed_classes,
        seeds,
        families,
        forward,
        merged: merged_section,
        column_rows,
        column_weights,
    } = file;
    let mut out = Out(Vec::new());
    let o = &mut out;
    o.key(true, "schema");
    o.uint(u64::from(*schema));
    o.key(false, "legs");
    o.usize(*legs);
    o.key(false, "rows");
    o.usize(*rows);
    o.key(false, "helicities");
    o.list(helicities, |out, leg| {
        out.list(leg, |out, &h| out.int(i64::from(h)))
    });
    o.key(false, "columns");
    o.usize(*columns);
    o.key(false, "average");
    o.uint(*average);
    o.key(false, "born_components");
    o.usize(*born_components);
    o.key(false, "constants");
    o.usize(*constants);
    o.key(false, "masses");
    o.usize(*masses);
    o.key(false, "block_rows");
    o.usizes(block_rows);
    o.key(false, "kernels");
    o.list(kernels, kernel);
    o.key(false, "contents");
    o.list(contents, content);
    o.key(false, "factors");
    o.list(factors, |out, &(content, shift)| {
        out.byte(b'[');
        out.usize(content);
        out.byte(b',');
        out.uint(shift);
        out.byte(b']');
    });
    o.key(false, "seed_classes");
    o.list(seed_classes, |out, SeedFile { columns }| {
        out.key(true, "columns");
        out.list(columns, |out, column| out.weights(column));
        out.byte(b'}');
    });
    o.key(false, "seeds");
    o.list(seeds, |out, SeedKeyFile { classes }| {
        out.key(true, "classes");
        out.weights(classes);
        out.byte(b'}');
    });
    o.key(false, "families");
    o.list(families, family);
    o.key(false, "forward");
    o.option(forward, |out, ForwardFile { states }| {
        out.key(true, "states");
        out.list(states, state);
        out.byte(b'}');
    });
    o.key(false, "merged");
    o.option(merged_section, merged);
    if let Some(columns) = column_rows {
        o.key(false, "column_rows");
        o.list(columns, |out, rows| {
            out.option(rows, |out, rows| out.usizes(rows))
        });
    }
    if let Some(weights) = column_weights {
        o.key(false, "column_weights");
        o.usizes(weights);
    }
    o.byte(b'}');
    out.0
}

fn kernel(out: &mut Out, KernelFile { inputs, outputs }: &KernelFile) {
    out.key(true, "inputs");
    out.inputs(inputs);
    out.key(false, "outputs");
    out.list(outputs, |out, terms| {
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
    let ContentFile {
        d_in,
        d_out,
        degree,
        rows,
        mask,
        row_helicities,
        fused,
        terms,
    } = content;
    out.key(true, "d_in");
    out.usize(*d_in);
    out.key(false, "d_out");
    out.usize(*d_out);
    out.key(false, "degree");
    out.usize(*degree);
    out.key(false, "rows");
    out.usize(*rows);
    out.key(false, "mask");
    out.uint(*mask);
    out.key(false, "row_helicities");
    out.list(row_helicities, |out, row| {
        out.list(row, |out, &h| out.int(i64::from(h)))
    });
    out.key(false, "fused");
    out.list(fused, |out, entry| {
        out.list(entry, |out, &v| out.uint(u64::from(v)))
    });
    out.key(false, "terms");
    out.list(terms, term);
    out.byte(b'}');
}

fn term(out: &mut Out, term: &TermFile) {
    let TermFile {
        slots,
        dimensions,
        bindings,
        vertex,
        split,
        rows,
    } = term;
    out.key(true, "slots");
    out.list(slots, |out, &mask| out.uint(mask));
    out.key(false, "dimensions");
    out.usizes(dimensions);
    out.key(false, "bindings");
    out.list(bindings, |out, binding| {
        let BindingFile {
            kernel,
            inputs,
            entries,
        } = binding;
        out.key(true, "kernel");
        out.usize(*kernel);
        out.key(false, "inputs");
        out.inputs(inputs);
        out.key(false, "entries");
        out.usizes(entries);
        out.byte(b'}');
    });
    out.key(false, "vertex");
    out.int(*vertex);
    out.key(false, "split");
    out.list(split, |out, split| {
        let SplitFile {
            kernel,
            inputs,
            entries,
        } = split;
        out.key(true, "kernel");
        out.usize(*kernel);
        out.key(false, "inputs");
        out.inputs(inputs);
        out.key(false, "entries");
        out.list(entries, |out, &(a, b)| {
            out.byte(b'[');
            out.usize(a);
            out.byte(b',');
            out.usize(b);
            out.byte(b']');
        });
        out.byte(b'}');
    });
    out.key(false, "rows");
    out.list(rows, |out, (row, blocks)| {
        out.byte(b'[');
        out.usize(*row);
        out.byte(b',');
        out.usizes(blocks);
        out.byte(b']');
    });
    out.byte(b'}');
}

fn family(out: &mut Out, family: &FamilyFile) {
    let FamilyFile {
        sequence,
        rank,
        orders,
        leaves,
        steps,
    } = family;
    out.key(true, "sequence");
    out.list(sequence, |out, &(mask, mass)| {
        out.byte(b'[');
        out.uint(mask);
        out.byte(b',');
        out.usize(mass);
        out.byte(b']');
    });
    out.key(false, "rank");
    out.usize(*rank);
    out.key(false, "orders");
    out.usizes(orders);
    out.key(false, "leaves");
    out.list(leaves, |out, LeafFile { seed, weight }| {
        out.key(true, "seed");
        out.usize(*seed);
        out.key(false, "weight");
        out.string(weight);
        out.byte(b'}');
    });
    out.key(false, "steps");
    out.list(steps, step);
    out.byte(b'}');
}

fn step(out: &mut Out, step: &StepFile) {
    let StepFile {
        seeded,
        trace,
        out: open,
        d_alpha,
        d_out,
        degree,
        rank_in,
        edges,
        rows,
    } = step;
    out.key(true, "seeded");
    out.boolean(*seeded);
    out.key(false, "trace");
    out.int(*trace);
    out.key(false, "out");
    out.uint(*open);
    out.key(false, "d_alpha");
    out.usize(*d_alpha);
    out.key(false, "d_out");
    out.usize(*d_out);
    out.key(false, "degree");
    out.usize(*degree);
    out.key(false, "rank_in");
    out.usize(*rank_in);
    out.key(false, "edges");
    out.list(edges, |out, edge| {
        let EdgeFile {
            leaf,
            child,
            factor,
            links,
        } = edge;
        out.key(true, "leaf");
        out.int(*leaf);
        out.key(false, "child");
        out.int(*child);
        out.key(false, "factor");
        out.usize(*factor);
        if let Some(links) = links {
            out.key(false, "links");
            out.list(links, |out, &v| out.uint(u64::from(v)));
        }
        out.byte(b'}');
    });
    if let Some(rows) = rows {
        out.key(false, "rows");
        out.usize(*rows);
    }
    out.byte(b'}');
}

fn state(out: &mut Out, state: &StateFile) {
    let StateFile {
        parent,
        factor,
        seeded,
        rows,
        d_alpha,
        d_out,
        degree,
        rank,
        links,
        closures,
    } = state;
    out.key(true, "parent");
    out.int(*parent);
    out.key(false, "factor");
    out.usize(*factor);
    out.key(false, "seeded");
    out.boolean(*seeded);
    out.key(false, "rows");
    out.usize(*rows);
    out.key(false, "d_alpha");
    out.usize(*d_alpha);
    out.key(false, "d_out");
    out.usize(*d_out);
    out.key(false, "degree");
    out.usize(*degree);
    out.key(false, "rank");
    out.usize(*rank);
    out.key(false, "links");
    out.list(links, |out, &v| out.uint(u64::from(v)));
    out.key(false, "closures");
    out.list(closures, closure);
    out.byte(b'}');
}

fn closure(out: &mut Out, closure: &ClosureFile) {
    let ClosureFile {
        family,
        accumulator,
        seed,
        weight,
        rows,
        relabel,
    } = closure;
    out.key(true, "family");
    out.usize(*family);
    out.key(false, "accumulator");
    out.usize(*accumulator);
    out.key(false, "seed");
    out.usize(*seed);
    out.key(false, "weight");
    out.string(weight);
    out.key(false, "rows");
    out.list(rows, |out, &v| out.int(v));
    out.key(false, "relabel");
    out.relabel(relabel);
    out.byte(b'}');
}

fn merged(out: &mut Out, merged: &MergedFile) {
    let MergedFile {
        families,
        entries,
        states,
        chunks,
    } = merged;
    out.key(true, "families");
    out.usizes(families);
    out.key(false, "entries");
    out.list(entries, |out, EntryFile { seed, weight }| {
        out.key(true, "seed");
        out.usize(*seed);
        out.key(false, "weight");
        out.pair(weight);
        out.byte(b'}');
    });
    out.key(false, "states");
    out.list(states, merged_state);
    out.key(false, "chunks");
    out.usizes(chunks);
    out.byte(b'}');
}

fn merged_state(out: &mut Out, state: &MergedStateFile) {
    let MergedStateFile {
        seeded,
        rows,
        d_alpha,
        d_out,
        degree,
        rank_in,
        edges,
        closure,
    } = state;
    out.key(true, "seeded");
    out.boolean(*seeded);
    out.key(false, "rows");
    out.usize(*rows);
    out.key(false, "d_alpha");
    out.usize(*d_alpha);
    out.key(false, "d_out");
    out.usize(*d_out);
    out.key(false, "degree");
    out.usize(*degree);
    out.key(false, "rank_in");
    out.usize(*rank_in);
    out.key(false, "edges");
    out.list(edges, |out, edge| {
        let MergedEdgeFile {
            entry,
            parent,
            factor,
            links,
            weight,
        } = edge;
        out.key(true, "entry");
        out.int(*entry);
        out.key(false, "parent");
        out.int(*parent);
        out.key(false, "factor");
        out.usize(*factor);
        out.key(false, "links");
        out.list(links, |out, &v| out.uint(u64::from(v)));
        out.key(false, "weight");
        out.option(weight, |out, weight| out.pair(weight));
        out.byte(b'}');
    });
    out.key(false, "closure");
    out.option(closure, |out, closure| {
        let MergedClosureFile {
            family,
            accumulator,
            relabel,
        } = closure;
        out.key(true, "family");
        out.usize(*family);
        out.key(false, "accumulator");
        out.usize(*accumulator);
        out.key(false, "relabel");
        out.relabel(relabel);
        out.byte(b'}');
    });
    out.byte(b'}');
}

#[cfg(test)]
mod tests {
    use super::super::super::{EdgeFile, PlanFile, StepFile};
    use super::{OMITTED_WHEN_ABSENT, Out};

    #[test]
    fn omitted_keys_name_their_records() {
        let records = [
            std::any::type_name::<PlanFile>(),
            std::any::type_name::<PlanFile>(),
            std::any::type_name::<StepFile>(),
            std::any::type_name::<EdgeFile>(),
        ];
        for (&(record, _), name) in OMITTED_WHEN_ABSENT.iter().zip(records) {
            assert_eq!(name.rsplit("::").next(), Some(record));
        }
    }

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
