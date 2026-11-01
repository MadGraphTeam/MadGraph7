// SPDX-License-Identifier: 0BSD

//! Symbol renaming in Symbolica's binary export (M-FINAL MF-RUST).
//!
//! A canonical stage (`loops/v2/assemble.py::_canonical_stage`) renames a
//! stage's own symbols onto canonical ones. On text (`_renamed_text`) that
//! costs a print and a parse of every expression in the Python wheel (the
//! parse alone 18 s of the 634 MB parameter program of an electroweak 2->3
//! export). Symbolica's export (`Expression.save`) writes the expression with
//! a table of the symbols it uses, and its import (`Expression.load`) maps
//! every table entry by name onto the importing state's symbols and
//! normalises the expression there. Renaming the table entries therefore
//! renames the symbols wherever they occur, without touching the expression:
//! [`rename_export`] rewrites an export file (brotli stream, Symbolica's
//! export format 6) entry by entry, replacing each renamed symbol's entry by
//! the entry of its target as this crate's state exports it (a plain symbol:
//! no attributes, tags, aliases or data), and copies everything else byte
//! for byte. The target must be a plain symbol in the importing state too
//! (the canonical symbols are), or the import refuses the conflict. This
//! crate builds the Symbolica revision of the Python wheel, so both write
//! and read the same format; the format version is checked against this
//! crate's own export.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use symbolica::atom::{Atom, AtomCore, UserData};
use symbolica::parser::ParseSettings;

use crate::{RusticolError, RusticolResult};

const MAGIC: u32 = 0x3787_1367;
const FULL_STATE: u8 = 1;
const BUFFER: usize = 4096;
const WINDOW: u32 = 22;

fn invalid(message: impl Into<String>) -> RusticolError {
    RusticolError::invalid_argument(format!("symbol rename: {}", message.into()))
}

fn io(error: std::io::Error) -> RusticolError {
    invalid(error.to_string())
}

/// A reader that keeps a copy of every byte it reads.
struct Tee<'a, R: Read> {
    inner: &'a mut R,
    copy: Vec<u8>,
}

impl<R: Read> Read for Tee<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buffer)?;
        self.copy.extend_from_slice(&buffer[..n]);
        Ok(n)
    }
}

fn read_exact<const N: usize>(source: &mut impl Read) -> RusticolResult<[u8; N]> {
    let mut bytes = [0u8; N];
    source.read_exact(&mut bytes).map_err(io)?;
    Ok(bytes)
}

fn read_u32(source: &mut impl Read) -> RusticolResult<u32> {
    Ok(u32::from_le_bytes(read_exact::<4>(source)?))
}

fn read_u16(source: &mut impl Read) -> RusticolResult<u16> {
    Ok(u16::from_le_bytes(read_exact::<2>(source)?))
}

fn read_string(source: &mut impl Read) -> RusticolResult<String> {
    let length = read_u32(source)? as usize;
    let mut bytes = vec![0u8; length];
    source.read_exact(&mut bytes).map_err(io)?;
    String::from_utf8(bytes).map_err(|e| invalid(e.to_string()))
}

/// One symbol entry (`Symbol::export`): its full name and its bytes.
fn read_entry(source: &mut impl Read) -> RusticolResult<(String, Vec<u8>)> {
    let mut tee = Tee {
        inner: source,
        copy: Vec::new(),
    };
    let name = read_string(&mut tee)?;
    let _namespace = read_string(&mut tee)?;
    let _flags = read_exact::<1>(&mut tee)?;
    let _extra_flags = read_u32(&mut tee)?;
    for _ in 0..read_u16(&mut tee)? {
        read_string(&mut tee)?; // tags
    }
    for _ in 0..read_u16(&mut tee)? {
        read_string(&mut tee)?; // aliases
    }
    UserData::read(&mut tee).map_err(io)?;
    let _exportable = read_exact::<1>(&mut tee)?;
    Ok((name, tee.copy))
}

/// This crate's export header: (format version, entry bytes of `symbol`).
fn own_entry(name: &str) -> RusticolResult<(u16, Vec<u8>)> {
    let atom = Atom::parse(name, "python", ParseSettings::default())
        .map_err(|e| invalid(format!("cannot parse the symbol {name:?}: {e}")))?;
    let Atom::Var(var) = &atom else {
        return Err(invalid(format!("{name:?} is not a symbol")));
    };
    let symbol = var.get_symbol();
    if symbol.get_name() != name {
        return Err(invalid(format!("{name:?} is not a fully qualified symbol name")));
    }
    let mut header = Vec::new();
    atom.export(&mut header).map_err(io)?;
    let mut source = header.as_slice();
    if read_u32(&mut source)? != MAGIC {
        return Err(invalid("this crate's export has an unknown magic number"));
    }
    let version = read_u16(&mut source)?;
    let mut entry = Vec::new();
    symbol.export(&mut entry).map_err(io)?;
    Ok((version, entry))
}

/// What [`rename_export`] did: the renamed entries and all entries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenameCounts {
    pub renamed: usize,
    pub symbols: usize,
}

/// Rewrite the Symbolica export `input` (brotli, as `Expression.save` writes
/// it) into `output` (brotli of `quality`), the entries of the symbols named
/// by `renames` (old, new fully qualified names) replaced by their targets'.
pub fn rename_export(
    input: &Path,
    output: &Path,
    renames: &[(String, String)],
    quality: u32,
) -> RusticolResult<RenameCounts> {
    let _guard = crate::licence::unlock();
    let mut targets: HashMap<&str, Vec<u8>> = HashMap::with_capacity(renames.len());
    let mut version = None;
    let mut entries: HashMap<&str, Vec<u8>> = HashMap::new();
    for (old, new) in renames {
        let entry = match entries.get(new.as_str()) {
            Some(entry) => entry.clone(),
            None => {
                let (own, entry) = own_entry(new)?;
                version = Some(own);
                entries.insert(new.as_str(), entry.clone());
                entry
            }
        };
        if targets.insert(old.as_str(), entry).is_some() {
            return Err(invalid(format!("{old:?} is renamed twice")));
        }
    }
    let file = File::open(input).map_err(io)?;
    let mut source = brotli::Decompressor::new(BufReader::new(file), BUFFER);
    let file = File::create(output).map_err(io)?;
    let mut dest = brotli::CompressorWriter::new(BufWriter::new(file), BUFFER, quality, WINDOW);
    let magic = read_u32(&mut source)?;
    if magic != MAGIC {
        return Err(invalid("the input is not a Symbolica export"));
    }
    let format = read_u16(&mut source)?;
    let version = match version {
        Some(version) => version,
        None => own_entry("symbolica::𝜋")?.0,
    };
    if format != version {
        return Err(invalid(format!(
            "export format {format} is not this crate's ({version})"
        )));
    }
    let flag = read_exact::<1>(&mut source)?[0];
    let symbols = u64::from_le_bytes(read_exact::<8>(&mut source)?);
    dest.write_all(&magic.to_le_bytes()).map_err(io)?;
    dest.write_all(&format.to_le_bytes()).map_err(io)?;
    dest.write_all(&[flag]).map_err(io)?;
    dest.write_all(&symbols.to_le_bytes()).map_err(io)?;
    let mut counts = RenameCounts {
        renamed: 0,
        symbols: symbols as usize,
    };
    for _ in 0..symbols {
        if flag != FULL_STATE {
            let index = read_exact::<4>(&mut source)?;
            dest.write_all(&index).map_err(io)?;
        }
        let (name, bytes) = read_entry(&mut source)?;
        match targets.get(name.as_str()) {
            Some(entry) => {
                dest.write_all(entry).map_err(io)?;
                counts.renamed += 1;
            }
            None => dest.write_all(&bytes).map_err(io)?,
        }
    }
    // Finite fields, variable lists and the expression: unchanged.
    std::io::copy(&mut source, &mut dest).map_err(io)?;
    dest.flush().map_err(io)?;
    let mut inner = dest.into_inner();
    inner.flush().map_err(io)?;
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renamed_entries_import_as_their_targets() {
        let _guard = crate::licence::unlock();
        let dir = std::env::temp_dir().join(format!("mfrust-rename-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let input = dir.join("in.dat");
        let output = dir.join("out.dat");
        let expression = Atom::parse(
            "mfrust_old::f(mfrust_old::a, mfrust_old::b^2) + mfrust_old::a*mfrust_keep::k",
            "python",
            ParseSettings::default(),
        )
        .unwrap();
        {
            let file = File::create(&input).unwrap();
            let mut writer = brotli::CompressorWriter::new(BufWriter::new(file), BUFFER, 0, WINDOW);
            expression.export(&mut writer).unwrap();
        }
        let renames = vec![
            ("mfrust_old::a".to_owned(), "mfrust_new::x0".to_owned()),
            ("mfrust_old::f".to_owned(), "mfrust_new::x1".to_owned()),
            ("mfrust_old::unused".to_owned(), "mfrust_new::x2".to_owned()),
        ];
        let counts = rename_export(&input, &output, &renames, 0).unwrap();
        assert_eq!(counts.renamed, 2);
        let file = File::open(&output).unwrap();
        let mut reader = brotli::Decompressor::new(BufReader::new(file), BUFFER);
        let imported = Atom::import(&mut reader, None).unwrap();
        let expected = Atom::parse(
            "mfrust_new::x1(mfrust_new::x0, mfrust_old::b^2) + mfrust_new::x0*mfrust_keep::k",
            "python",
            ParseSettings::default(),
        )
        .unwrap();
        assert_eq!(imported, expected);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
