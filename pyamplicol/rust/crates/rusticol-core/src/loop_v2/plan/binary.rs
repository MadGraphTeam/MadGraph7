// SPDX-License-Identifier: 0BSD

//! The plan v4 model, its byte-identical JSON writer and its binary columnar
//! member (WP3.1 package D2, `workspace/reports/engine-v2/wp31/RUST-MIGRATION.md`
//! §3.D, PLAN I11).
//!
//! [`PlanModel`] holds a plan in the loader's own file form, so a plan read
//! from either encoding goes through the same validation and derivations
//! ([`LoopProgramV4::load`]; the schedule-B row links stay derived at load).
//!
//! - **JSON** ([`PlanModel::json`]): the bytes of
//!   `json.dumps(program, separators=(",", ":"))` of `plan_export.export_plan`,
//!   keys in the exporter's insertion order (`json` module). The plan holds no
//!   floating-point value: every number is an integer of the loader's types and
//!   every exact weight a `p/q` string, so Python's float formatting never
//!   arises.
//! - **Binary member** ([`PlanModel::binary`], [`LoopProgramV4::load_binary`]):
//!   every field of every record is one column, in a fixed traversal order;
//!   list lengths and option flags are columns too, and strings (exact weights,
//!   kernel input kinds) are numbers into a string table in first-use order.
//!   Each column is stored at the narrowest of 0, 1, 2, 4 or 8 bytes per value
//!   after subtracting its minimum, as values or as differences of successive
//!   values, whichever is narrower. The per-content lists that repeat across
//!   contents (row helicities, fused entries, term dimensions, binding and
//!   split inputs and entries) are stored once: their reference column holds
//!   the number of the list's first use, and a list not seen before takes the
//!   next number and follows in full. The member holds plan data only (no path,
//!   no host or build information), is deterministic, and decodes to the same
//!   model, so `binary → model → binary` is bitwise.
//!
//! Layout (little endian): magic `LOOPPLAN`, format `u32`, plan schema `u32`,
//! column count `u32`, `u32` zero; then per column (in the order of `Col`) a
//! width `u8` (0, 1, 2, 4 or 8), a delta flag `u8`, `u16` and `u32` zero, the
//! value count `u64`, the base `i64`, and `count × width` payload bytes padded
//! to a multiple of eight. Writing and reading share one traversal
//! (`visit_plan` over a `Stream`), so the two cannot disagree on the order.

mod json;

use super::{
    BindingFile, ClosureFile, ContentFile, EdgeFile, EntryFile, FamilyFile, FormPolicy,
    ForwardFile, KernelFile, LeafFile, LoopProgramV4, MergedClosureFile, MergedEdgeFile,
    MergedFile, MergedStateFile, PLAN_SCHEMA, PlanFile, SeedFile, SeedKeyFile, SplitFile,
    StateFile, StepFile, TermFile, invalid, require,
};
use crate::RusticolResult;
use sha2::{Digest, Sha256};
use std::any::Any;
use std::collections::HashMap;
use std::hash::Hash;

/// The first bytes of a binary plan member.
pub const MAGIC: [u8; 8] = *b"LOOPPLAN";
/// The binary member's format (one version; another is refused).
pub const FORMAT: u32 = 1;

/// Whether `bytes` start like a binary plan member (else they are JSON).
pub fn is_binary(bytes: &[u8]) -> bool {
    bytes.starts_with(&MAGIC)
}

/// A plan v4 in the loader's file form: the input of [`LoopProgramV4`]'s
/// validation and derivations, written as the exporter's JSON or as the
/// binary columnar member.
pub struct PlanModel {
    file: PlanFile,
}

impl PlanModel {
    /// The model of a plan v4 JSON (`ExportedPlan.json()`), parsed as the
    /// loader parses it (unknown fields refused).
    pub fn from_json(json: &[u8]) -> RusticolResult<Self> {
        let file = serde_json::from_slice(json).map_err(|error| invalid(error.to_string()))?;
        Ok(Self { file })
    }

    /// The model of a binary plan member ([`Self::binary`]).
    pub fn from_binary(bytes: &[u8]) -> RusticolResult<Self> {
        let mut reader = Reader::new(bytes)?;
        let mut file = PlanFile::default();
        visit_plan(&mut reader, &mut file)?;
        reader.finish()?;
        Ok(Self { file })
    }

    /// The plan JSON, byte-identical to the exporter's.
    pub fn json(&self) -> Vec<u8> {
        json::write(&self.file)
    }

    /// The binary columnar member. The traversal is shared with the reader,
    /// hence `&mut`; writing leaves the model unchanged.
    pub fn binary(&mut self) -> RusticolResult<Vec<u8>> {
        let mut writer = Writer::default();
        visit_plan(&mut writer, &mut self.file)?;
        Ok(writer.finish())
    }

    /// The validated program with its derived tables (`LoopProgramV4::load`).
    pub fn program(self, policy: FormPolicy) -> RusticolResult<LoopProgramV4> {
        LoopProgramV4::from_file(self.file, policy)
    }
}

impl LoopProgramV4 {
    /// [`Self::load`] from a binary plan member ([`PlanModel::binary`]).
    pub fn load_binary(bytes: &[u8]) -> RusticolResult<Self> {
        Self::load_binary_with(bytes, FormPolicy::Rule)
    }

    /// [`Self::load_binary`] with an explicit kernel-form policy.
    pub fn load_binary_with(bytes: &[u8], policy: FormPolicy) -> RusticolResult<Self> {
        PlanModel::from_binary(bytes)?.program(policy)
    }
}

/// SHA-256 of a loaded program's validated and derived tables (its `Debug`
/// form, streamed, every field included): equal tables give equal digests
/// and any differing value changes it. A test tool for comparing the JSON and
/// binary loads; no artifact carries it.
pub fn program_digest(program: &LoopProgramV4) -> String {
    struct Hashing(Sha256);
    impl std::fmt::Write for Hashing {
        fn write_str(&mut self, text: &str) -> std::fmt::Result {
            self.0.update(text.as_bytes());
            Ok(())
        }
    }
    let mut hashing = Hashing(Sha256::new());
    std::fmt::write(&mut hashing, format_args!("{program:?}"))
        .expect("formatting a program into a hasher cannot fail");
    hashing
        .0
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

// ------------------------------------------------------------------ columns

macro_rules! columns {
    ($($name:ident),* $(,)?) => {
        /// One column of the binary member per record field, list length and
        /// option flag, in member order.
        #[derive(Clone, Copy, Debug)]
        #[repr(u16)]
        enum Col {
            $($name),*
        }
        /// The column names (diagnostics).
        const COLUMN_NAMES: &[&str] = &[$(stringify!($name)),*];
    };
}

columns! {
    // string table
    StringLen, StringByte,
    // header scalars
    Schema, Legs, Rows, Columns, Average, BornComponents, Constants, Masses,
    HelicityLegs, HelicityLen, Helicity,
    BlockRowLen, BlockRow,
    // kernels
    KernelLen, KernelInputLen, KernelInputKind, KernelInputA, KernelInputB,
    KernelOutputLen, KernelTermLen, KernelTermConstant, MonomialLen, MonomialInput,
    MonomialPower,
    // contents
    ContentLen, ContentDIn, ContentDOut, ContentDegree, ContentRows, ContentMask,
    RowHelicityRef, RowHelicityLen, RowHelicityLegs, RowHelicity, FusedRef, FusedLen, Fused0, Fused1, Fused2, Fused3,
    Fused4, Fused5, TermLen, SlotLen, Slot, DimensionRef, DimensionLen, Dimension,
    BindingLen, BindingKernel, BindingInputRef, BindingInputLen, BindingInputKind, BindingInputA, BindingInputB,
    BindingEntryRef, BindingEntryLen, BindingEntry, TermVertex, SplitLen, SplitKernel,
    SplitInputRef, SplitInputLen, SplitInputKind, SplitInputA, SplitInputB, SplitEntryRef,
    SplitEntryLen, SplitEntryA, SplitEntryB,
    TermRowLen, TermRow, TermRowBlockLen, TermRowBlock,
    // factors
    FactorLen, FactorContent, FactorShift,
    // seed classes and seed keys
    SeedClassLen, SeedColumnLen, SeedWeightLen, SeedWeightBorn, SeedWeightRe, SeedWeightIm,
    SeedKeyLen, SeedKeyClassLen, SeedKeyClass, SeedKeyRe, SeedKeyIm,
    // families (schedule B)
    FamilyLen, SequenceLen, SequenceMask, SequenceMass, FamilyRank, FamilyOrderLen,
    FamilyOrder, LeafLen, LeafSeed, LeafWeight, StepLen, StepSeeded, StepTrace, StepOut,
    StepRowsFlag, StepRows, StepDAlpha, StepDOut, StepDegree, StepRankIn, EdgeLen, EdgeLeaf,
    EdgeChild, EdgeFactor, EdgeLinksFlag, EdgeLinkLen, EdgeLink,
    // forward (schedule F)
    ForwardFlag, StateLen, StateParent, StateFactor, StateSeeded, StateRows, StateDAlpha,
    StateDOut, StateDegree, StateRank, StateLinkLen, StateLink, ClosureLen, ClosureFamily,
    ClosureAccumulator, ClosureSeed, ClosureWeight, ClosureRowLen, ClosureRow,
    ClosureRelabelFlag, ClosureRelabelSign, ClosureRelabelMask,
    // merged (schedule M)
    MergedFlag, MergedFamilyLen, MergedFamily, EntryLen, EntrySeed, EntryRe, EntryIm,
    MergedStateLen, MergedSeeded, MergedRows, MergedDAlpha, MergedDOut, MergedDegree,
    MergedRankIn, MergedEdgeLen, MergedEdgeEntry, MergedEdgeParent, MergedEdgeFactor,
    MergedEdgeWeightFlag, MergedEdgeRe, MergedEdgeIm, MergedEdgeLinkLen, MergedEdgeLink,
    MergedClosureFlag, MergedClosureFamily, MergedClosureAccumulator,
    MergedClosureRelabelFlag, MergedClosureRelabelSign, MergedClosureRelabelMask, ChunkLen,
    Chunk,
    // column rows
    ColumnRowsFlag, ColumnRowsLen, ColumnRowFlag, ColumnRowLen, ColumnRow,
}

// ------------------------------------------------------------------ streams

/// One direction of the shared traversal: the writer records the visited
/// values, the reader overwrites them with the member's.
trait Stream {
    const READ: bool;
    fn int(&mut self, col: Col, value: &mut i64) -> RusticolResult<()>;
    fn string(&mut self, col: Col, value: &mut String) -> RusticolResult<()>;
    /// A list length: written as `len`, or read (and bounded).
    fn len(&mut self, col: Col, len: usize) -> RusticolResult<usize>;
    /// Per reference column, its table of shared lists (`shared`).
    fn tables(&mut self) -> &mut Vec<Option<Box<dyn Any>>>;
}

/// The table of shared lists of one reference column: by list when writing
/// (`HashMap<Vec<T>, i64>`), by number when reading (`Vec<Vec<T>>`).
fn table<S: Stream, X: Default + 'static>(stream: &mut S, refs: Col) -> &mut X {
    let tables = stream.tables();
    if tables.is_empty() {
        tables.resize_with(COLUMN_NAMES.len(), || None);
    }
    tables[refs as usize]
        .get_or_insert_with(|| Box::new(X::default()))
        .downcast_mut::<X>()
        .expect("one list type per reference column")
}

fn num<S: Stream, T>(stream: &mut S, col: Col, value: &mut T) -> RusticolResult<()>
where
    T: Copy + TryInto<i64> + TryFrom<i64>,
{
    let mut wide: i64 = (*value)
        .try_into()
        .map_err(|_| invalid(format!("{col:?} out of the binary member's range")))?;
    stream.int(col, &mut wide)?;
    *value =
        T::try_from(wide).map_err(|_| invalid(format!("binary plan: {col:?} out of range")))?;
    Ok(())
}

fn flag<S: Stream>(stream: &mut S, col: Col, value: &mut bool) -> RusticolResult<()> {
    let mut wide = i64::from(*value);
    stream.int(col, &mut wide)?;
    *value = match wide {
        0 => false,
        1 => true,
        _ => return Err(invalid(format!("binary plan: {col:?} is not a flag"))),
    };
    Ok(())
}

fn list<S: Stream, T: Default>(
    stream: &mut S,
    col: Col,
    items: &mut Vec<T>,
    mut visit: impl FnMut(&mut S, &mut T) -> RusticolResult<()>,
) -> RusticolResult<()> {
    let len = stream.len(col, items.len())?;
    if S::READ {
        items.clear();
        items.resize_with(len, T::default);
    }
    for item in items.iter_mut() {
        visit(stream, item)?;
    }
    Ok(())
}

fn nums<S: Stream, T>(stream: &mut S, len: Col, col: Col, items: &mut Vec<T>) -> RusticolResult<()>
where
    T: Copy + Default + TryInto<i64> + TryFrom<i64>,
{
    list(stream, len, items, |s, item| num(s, col, item))
}

/// A list stored once at its first use and referenced by number afterwards.
fn shared<S: Stream, T>(
    stream: &mut S,
    refs: Col,
    len: Col,
    items: &mut Vec<T>,
    visit: impl FnMut(&mut S, &mut T) -> RusticolResult<()>,
) -> RusticolResult<()>
where
    T: Clone + Default + Eq + Hash + 'static,
{
    if S::READ {
        let mut id = 0i64;
        stream.int(refs, &mut id)?;
        let known: &mut Vec<Vec<T>> = table(stream, refs);
        match usize::try_from(id) {
            Ok(k) if k < known.len() => {
                items.clone_from(&known[k]);
                Ok(())
            }
            Ok(k) if k == known.len() => {
                list(stream, len, items, visit)?;
                table::<S, Vec<Vec<T>>>(stream, refs).push(items.clone());
                Ok(())
            }
            _ => Err(invalid(format!("binary plan: {refs:?} reference"))),
        }
    } else {
        let known: &mut HashMap<Vec<T>, i64> = table(stream, refs);
        let next = known.len() as i64;
        let mut id = match known.get(items.as_slice()) {
            Some(&id) => id,
            None => {
                known.insert(items.clone(), next);
                next
            }
        };
        stream.int(refs, &mut id)?;
        if id == next {
            list(stream, len, items, visit)?;
        }
        Ok(())
    }
}

fn option<S: Stream, T: Default>(
    stream: &mut S,
    col: Col,
    value: &mut Option<T>,
    visit: impl FnOnce(&mut S, &mut T) -> RusticolResult<()>,
) -> RusticolResult<()> {
    let mut present = value.is_some();
    flag(stream, col, &mut present)?;
    if S::READ {
        *value = present.then(T::default);
    }
    match value {
        Some(inner) => visit(stream, inner),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------- traversal

type Input = (String, usize, usize);

/// One kernel or binding input `(kind, a, b)`.
fn input<S: Stream>(s: &mut S, [kind, a, b]: [Col; 3], item: &mut Input) -> RusticolResult<()> {
    let (k, x, y) = item;
    s.string(kind, k)?;
    num(s, a, x)?;
    num(s, b, y)
}

fn visit_plan<S: Stream>(s: &mut S, file: &mut PlanFile) -> RusticolResult<()> {
    num(s, Col::Schema, &mut file.schema)?;
    num(s, Col::Legs, &mut file.legs)?;
    num(s, Col::Rows, &mut file.rows)?;
    list(s, Col::HelicityLegs, &mut file.helicities, |s, leg| {
        nums(s, Col::HelicityLen, Col::Helicity, leg)
    })?;
    num(s, Col::Columns, &mut file.columns)?;
    num(s, Col::Average, &mut file.average)?;
    num(s, Col::BornComponents, &mut file.born_components)?;
    num(s, Col::Constants, &mut file.constants)?;
    num(s, Col::Masses, &mut file.masses)?;
    nums(s, Col::BlockRowLen, Col::BlockRow, &mut file.block_rows)?;
    list(s, Col::KernelLen, &mut file.kernels, visit_kernel)?;
    list(s, Col::ContentLen, &mut file.contents, visit_content)?;
    list(
        s,
        Col::FactorLen,
        &mut file.factors,
        |s, (content, shift)| {
            num(s, Col::FactorContent, content)?;
            num(s, Col::FactorShift, shift)
        },
    )?;
    list(s, Col::SeedClassLen, &mut file.seed_classes, |s, seed| {
        list(s, Col::SeedColumnLen, &mut seed.columns, |s, column| {
            list(s, Col::SeedWeightLen, column, |s, (born, re, im)| {
                num(s, Col::SeedWeightBorn, born)?;
                s.string(Col::SeedWeightRe, re)?;
                s.string(Col::SeedWeightIm, im)
            })
        })
    })?;
    list(s, Col::SeedKeyLen, &mut file.seeds, |s, key| {
        list(
            s,
            Col::SeedKeyClassLen,
            &mut key.classes,
            |s, (class, re, im)| {
                num(s, Col::SeedKeyClass, class)?;
                s.string(Col::SeedKeyRe, re)?;
                s.string(Col::SeedKeyIm, im)
            },
        )
    })?;
    list(s, Col::FamilyLen, &mut file.families, visit_family)?;
    option(s, Col::ForwardFlag, &mut file.forward, |s, forward| {
        list(s, Col::StateLen, &mut forward.states, visit_state)
    })?;
    option(s, Col::MergedFlag, &mut file.merged, visit_merged)?;
    option(
        s,
        Col::ColumnRowsFlag,
        &mut file.column_rows,
        |s, columns| {
            list(s, Col::ColumnRowsLen, columns, |s, rows| {
                option(s, Col::ColumnRowFlag, rows, |s, rows| {
                    nums(s, Col::ColumnRowLen, Col::ColumnRow, rows)
                })
            })
        },
    )
}

fn visit_kernel<S: Stream>(s: &mut S, kernel: &mut KernelFile) -> RusticolResult<()> {
    list(s, Col::KernelInputLen, &mut kernel.inputs, |s, item| {
        input(
            s,
            [Col::KernelInputKind, Col::KernelInputA, Col::KernelInputB],
            item,
        )
    })?;
    list(s, Col::KernelOutputLen, &mut kernel.outputs, |s, terms| {
        list(s, Col::KernelTermLen, terms, |s, (constant, monomial)| {
            num(s, Col::KernelTermConstant, constant)?;
            list(s, Col::MonomialLen, monomial, |s, (input, power)| {
                num(s, Col::MonomialInput, input)?;
                num(s, Col::MonomialPower, power)
            })
        })
    })
}

fn visit_content<S: Stream>(s: &mut S, content: &mut ContentFile) -> RusticolResult<()> {
    num(s, Col::ContentDIn, &mut content.d_in)?;
    num(s, Col::ContentDOut, &mut content.d_out)?;
    num(s, Col::ContentDegree, &mut content.degree)?;
    num(s, Col::ContentRows, &mut content.rows)?;
    num(s, Col::ContentMask, &mut content.mask)?;
    let rows = &mut content.row_helicities;
    shared(
        s,
        Col::RowHelicityRef,
        Col::RowHelicityLen,
        rows,
        |s, row| nums(s, Col::RowHelicityLegs, Col::RowHelicity, row),
    )?;
    shared(
        s,
        Col::FusedRef,
        Col::FusedLen,
        &mut content.fused,
        |s, entry| {
            const FUSED: [Col; 6] = [
                Col::Fused0,
                Col::Fused1,
                Col::Fused2,
                Col::Fused3,
                Col::Fused4,
                Col::Fused5,
            ];
            for (col, value) in FUSED.into_iter().zip(entry.iter_mut()) {
                num(s, col, value)?;
            }
            Ok(())
        },
    )?;
    list(s, Col::TermLen, &mut content.terms, visit_term)
}

fn visit_term<S: Stream>(s: &mut S, term: &mut TermFile) -> RusticolResult<()> {
    nums(s, Col::SlotLen, Col::Slot, &mut term.slots)?;
    let dimensions = &mut term.dimensions;
    shared(
        s,
        Col::DimensionRef,
        Col::DimensionLen,
        dimensions,
        |s, d| num(s, Col::Dimension, d),
    )?;
    list(s, Col::BindingLen, &mut term.bindings, visit_binding)?;
    num(s, Col::TermVertex, &mut term.vertex)?;
    list(s, Col::SplitLen, &mut term.split, visit_split)?;
    list(s, Col::TermRowLen, &mut term.rows, |s, (row, blocks)| {
        num(s, Col::TermRow, row)?;
        nums(s, Col::TermRowBlockLen, Col::TermRowBlock, blocks)
    })
}

fn visit_binding<S: Stream>(s: &mut S, binding: &mut BindingFile) -> RusticolResult<()> {
    const INPUT: [Col; 3] = [
        Col::BindingInputKind,
        Col::BindingInputA,
        Col::BindingInputB,
    ];
    num(s, Col::BindingKernel, &mut binding.kernel)?;
    let inputs = &mut binding.inputs;
    shared(
        s,
        Col::BindingInputRef,
        Col::BindingInputLen,
        inputs,
        |s, item| input(s, INPUT, item),
    )?;
    let entries = &mut binding.entries;
    shared(
        s,
        Col::BindingEntryRef,
        Col::BindingEntryLen,
        entries,
        |s, e| num(s, Col::BindingEntry, e),
    )
}

fn visit_split<S: Stream>(s: &mut S, split: &mut SplitFile) -> RusticolResult<()> {
    const INPUT: [Col; 3] = [Col::SplitInputKind, Col::SplitInputA, Col::SplitInputB];
    num(s, Col::SplitKernel, &mut split.kernel)?;
    let inputs = &mut split.inputs;
    shared(
        s,
        Col::SplitInputRef,
        Col::SplitInputLen,
        inputs,
        |s, item| input(s, INPUT, item),
    )?;
    let entries = &mut split.entries;
    shared(
        s,
        Col::SplitEntryRef,
        Col::SplitEntryLen,
        entries,
        |s, (a, b)| {
            num(s, Col::SplitEntryA, a)?;
            num(s, Col::SplitEntryB, b)
        },
    )
}

fn visit_family<S: Stream>(s: &mut S, family: &mut FamilyFile) -> RusticolResult<()> {
    list(
        s,
        Col::SequenceLen,
        &mut family.sequence,
        |s, (mask, mass)| {
            num(s, Col::SequenceMask, mask)?;
            num(s, Col::SequenceMass, mass)
        },
    )?;
    num(s, Col::FamilyRank, &mut family.rank)?;
    nums(s, Col::FamilyOrderLen, Col::FamilyOrder, &mut family.orders)?;
    list(
        s,
        Col::LeafLen,
        &mut family.leaves,
        |s, leaf: &mut LeafFile| {
            num(s, Col::LeafSeed, &mut leaf.seed)?;
            s.string(Col::LeafWeight, &mut leaf.weight)
        },
    )?;
    list(
        s,
        Col::StepLen,
        &mut family.steps,
        |s, step: &mut StepFile| {
            flag(s, Col::StepSeeded, &mut step.seeded)?;
            num(s, Col::StepTrace, &mut step.trace)?;
            num(s, Col::StepOut, &mut step.out)?;
            option(s, Col::StepRowsFlag, &mut step.rows, |s, rows| {
                num(s, Col::StepRows, rows)
            })?;
            num(s, Col::StepDAlpha, &mut step.d_alpha)?;
            num(s, Col::StepDOut, &mut step.d_out)?;
            num(s, Col::StepDegree, &mut step.degree)?;
            num(s, Col::StepRankIn, &mut step.rank_in)?;
            list(
                s,
                Col::EdgeLen,
                &mut step.edges,
                |s, edge: &mut EdgeFile| {
                    num(s, Col::EdgeLeaf, &mut edge.leaf)?;
                    num(s, Col::EdgeChild, &mut edge.child)?;
                    num(s, Col::EdgeFactor, &mut edge.factor)?;
                    option(s, Col::EdgeLinksFlag, &mut edge.links, |s, links| {
                        nums(s, Col::EdgeLinkLen, Col::EdgeLink, links)
                    })
                },
            )
        },
    )
}

fn visit_state<S: Stream>(s: &mut S, state: &mut StateFile) -> RusticolResult<()> {
    num(s, Col::StateParent, &mut state.parent)?;
    num(s, Col::StateFactor, &mut state.factor)?;
    flag(s, Col::StateSeeded, &mut state.seeded)?;
    num(s, Col::StateRows, &mut state.rows)?;
    num(s, Col::StateDAlpha, &mut state.d_alpha)?;
    num(s, Col::StateDOut, &mut state.d_out)?;
    num(s, Col::StateDegree, &mut state.degree)?;
    num(s, Col::StateRank, &mut state.rank)?;
    nums(s, Col::StateLinkLen, Col::StateLink, &mut state.links)?;
    list(
        s,
        Col::ClosureLen,
        &mut state.closures,
        |s, closure: &mut ClosureFile| {
            num(s, Col::ClosureFamily, &mut closure.family)?;
            num(s, Col::ClosureAccumulator, &mut closure.accumulator)?;
            num(s, Col::ClosureSeed, &mut closure.seed)?;
            s.string(Col::ClosureWeight, &mut closure.weight)?;
            nums(s, Col::ClosureRowLen, Col::ClosureRow, &mut closure.rows)?;
            option(
                s,
                Col::ClosureRelabelFlag,
                &mut closure.relabel,
                |s, (sign, mask)| {
                    num(s, Col::ClosureRelabelSign, sign)?;
                    num(s, Col::ClosureRelabelMask, mask)
                },
            )
        },
    )
}

fn visit_merged<S: Stream>(s: &mut S, merged: &mut MergedFile) -> RusticolResult<()> {
    nums(
        s,
        Col::MergedFamilyLen,
        Col::MergedFamily,
        &mut merged.families,
    )?;
    list(
        s,
        Col::EntryLen,
        &mut merged.entries,
        |s, entry: &mut EntryFile| {
            num(s, Col::EntrySeed, &mut entry.seed)?;
            let [re, im] = &mut entry.weight;
            s.string(Col::EntryRe, re)?;
            s.string(Col::EntryIm, im)
        },
    )?;
    list(
        s,
        Col::MergedStateLen,
        &mut merged.states,
        |s, state: &mut MergedStateFile| {
            flag(s, Col::MergedSeeded, &mut state.seeded)?;
            num(s, Col::MergedRows, &mut state.rows)?;
            num(s, Col::MergedDAlpha, &mut state.d_alpha)?;
            num(s, Col::MergedDOut, &mut state.d_out)?;
            num(s, Col::MergedDegree, &mut state.degree)?;
            num(s, Col::MergedRankIn, &mut state.rank_in)?;
            list(
                s,
                Col::MergedEdgeLen,
                &mut state.edges,
                |s, edge: &mut MergedEdgeFile| {
                    num(s, Col::MergedEdgeEntry, &mut edge.entry)?;
                    num(s, Col::MergedEdgeParent, &mut edge.parent)?;
                    num(s, Col::MergedEdgeFactor, &mut edge.factor)?;
                    option(
                        s,
                        Col::MergedEdgeWeightFlag,
                        &mut edge.weight,
                        |s, [re, im]| {
                            s.string(Col::MergedEdgeRe, re)?;
                            s.string(Col::MergedEdgeIm, im)
                        },
                    )?;
                    nums(
                        s,
                        Col::MergedEdgeLinkLen,
                        Col::MergedEdgeLink,
                        &mut edge.links,
                    )
                },
            )?;
            option(
                s,
                Col::MergedClosureFlag,
                &mut state.closure,
                |s, closure: &mut MergedClosureFile| {
                    num(s, Col::MergedClosureFamily, &mut closure.family)?;
                    num(s, Col::MergedClosureAccumulator, &mut closure.accumulator)?;
                    option(
                        s,
                        Col::MergedClosureRelabelFlag,
                        &mut closure.relabel,
                        |s, (sign, mask)| {
                            num(s, Col::MergedClosureRelabelSign, sign)?;
                            num(s, Col::MergedClosureRelabelMask, mask)
                        },
                    )
                },
            )
        },
    )?;
    nums(s, Col::ChunkLen, Col::Chunk, &mut merged.chunks)
}

// ------------------------------------------------------------------ writing

#[derive(Default)]
struct Writer {
    columns: Vec<Vec<i64>>,
    strings: Vec<String>,
    ids: HashMap<String, i64>,
    tables: Vec<Option<Box<dyn Any>>>,
}

impl Writer {
    fn column(&mut self, col: Col) -> &mut Vec<i64> {
        if self.columns.is_empty() {
            self.columns = vec![Vec::new(); COLUMN_NAMES.len()];
        }
        &mut self.columns[col as usize]
    }

    fn finish(mut self) -> Vec<u8> {
        let strings = std::mem::take(&mut self.strings);
        for text in &strings {
            let len = text.len() as i64;
            self.column(Col::StringLen).push(len);
            let bytes = text.bytes().map(i64::from);
            self.column(Col::StringByte).extend(bytes);
        }
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT.to_le_bytes());
        out.extend_from_slice(&PLAN_SCHEMA.to_le_bytes());
        out.extend_from_slice(&(COLUMN_NAMES.len() as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for k in 0..COLUMN_NAMES.len() {
            let values = self.columns.get(k).map(Vec::as_slice).unwrap_or(&[]);
            encode_column(&mut out, values);
        }
        out
    }
}

impl Stream for Writer {
    const READ: bool = false;

    fn int(&mut self, col: Col, value: &mut i64) -> RusticolResult<()> {
        self.column(col).push(*value);
        Ok(())
    }

    fn string(&mut self, col: Col, value: &mut String) -> RusticolResult<()> {
        let id = match self.ids.get(value.as_str()) {
            Some(&id) => id,
            None => {
                let id = self.strings.len() as i64;
                self.strings.push(value.clone());
                self.ids.insert(value.clone(), id);
                id
            }
        };
        self.column(col).push(id);
        Ok(())
    }

    fn len(&mut self, col: Col, len: usize) -> RusticolResult<usize> {
        let mut wide = len as i64;
        self.int(col, &mut wide)?;
        Ok(len)
    }

    fn tables(&mut self) -> &mut Vec<Option<Box<dyn Any>>> {
        &mut self.tables
    }
}

/// Narrowest width (bytes) of values spanning `span` above their base.
fn width_of(span: u128) -> u8 {
    match span {
        0 => 0,
        1..=0xff => 1,
        0x100..=0xffff => 2,
        0x1_0000..=0xffff_ffff => 4,
        _ => 8,
    }
}

/// `(width, base)` of `values` as offsets from their minimum.
fn plain_form(values: impl Iterator<Item = i128> + Clone) -> (u8, i128) {
    let min = values.clone().min().unwrap_or(0);
    let max = values.max().unwrap_or(0);
    (width_of((max - min) as u128), min)
}

fn encode_column(out: &mut Vec<u8>, values: &[i64]) {
    let plain = values.iter().map(|&v| i128::from(v));
    let deltas = values.iter().scan(0i128, |previous, &v| {
        let delta = i128::from(v) - *previous;
        *previous = i128::from(v);
        Some(delta)
    });
    let (plain_width, plain_base) = plain_form(plain.clone());
    let (delta_width, delta_base) = plain_form(deltas.clone());
    // Narrower differences span < 2^32, so their base is an i64.
    let use_delta = delta_width < plain_width && i64::try_from(delta_base).is_ok();
    let (width, base, delta) = if use_delta {
        (delta_width, delta_base, true)
    } else {
        (plain_width, plain_base, false)
    };
    out.push(width);
    out.push(u8::from(delta));
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(values.len() as u64).to_le_bytes());
    let base64 = i64::try_from(base).expect("a column base is one of its values or deltas");
    out.extend_from_slice(&base64.to_le_bytes());
    if width > 0 {
        let start = out.len();
        let stored: Box<dyn Iterator<Item = i128>> = if delta {
            Box::new(deltas)
        } else {
            Box::new(plain)
        };
        for value in stored {
            let offset = (value - base) as u64;
            out.extend_from_slice(&offset.to_le_bytes()[..width as usize]);
        }
        let written = out.len() - start;
        out.resize(out.len() + (8 - written % 8) % 8, 0);
    }
}

// ------------------------------------------------------------------ reading

struct Cursor<'a> {
    payload: &'a [u8],
    width: usize,
    base: i64,
    delta: bool,
    count: usize,
    position: usize,
    previous: i64,
}

impl Cursor<'_> {
    fn next(&mut self, col: usize) -> RusticolResult<i64> {
        if self.position >= self.count {
            return Err(invalid(format!(
                "binary plan: column {} is exhausted",
                COLUMN_NAMES[col]
            )));
        }
        let at = self.position * self.width;
        let offset = match self.width {
            0 => 0,
            1 => u64::from(self.payload[at]),
            2 => u64::from(u16::from_le_bytes([self.payload[at], self.payload[at + 1]])),
            4 => u64::from(u32::from_le_bytes(
                self.payload[at..at + 4].try_into().expect("four bytes"),
            )),
            _ => u64::from_le_bytes(self.payload[at..at + 8].try_into().expect("eight bytes")),
        };
        self.position += 1;
        let stored = i128::from(self.base) + i128::from(offset);
        let value = if self.delta {
            i128::from(self.previous) + stored
        } else {
            stored
        };
        let value = i64::try_from(value).map_err(|_| {
            invalid(format!(
                "binary plan: column {} overflows",
                COLUMN_NAMES[col]
            ))
        })?;
        self.previous = value;
        Ok(value)
    }
}

struct Reader<'a> {
    cursors: Vec<Cursor<'a>>,
    strings: Vec<String>,
    /// Values in the member: no list is longer.
    budget: usize,
    tables: Vec<Option<Box<dyn Any>>>,
}

fn take<'a>(bytes: &mut &'a [u8], n: usize) -> RusticolResult<&'a [u8]> {
    if bytes.len() < n {
        return Err(invalid("binary plan: truncated member"));
    }
    let (head, rest) = bytes.split_at(n);
    *bytes = rest;
    Ok(head)
}

fn u32_at(bytes: &mut &[u8]) -> RusticolResult<u32> {
    Ok(u32::from_le_bytes(
        take(bytes, 4)?.try_into().expect("four bytes"),
    ))
}

fn u64_at(bytes: &mut &[u8]) -> RusticolResult<u64> {
    Ok(u64::from_le_bytes(
        take(bytes, 8)?.try_into().expect("eight bytes"),
    ))
}

impl<'a> Reader<'a> {
    fn new(member: &'a [u8]) -> RusticolResult<Self> {
        let mut bytes = member;
        require(
            take(&mut bytes, 8)? == MAGIC,
            "binary plan: not a binary plan member",
        )?;
        require(
            u32_at(&mut bytes)? == FORMAT,
            "binary plan: unsupported format",
        )?;
        require(
            u32_at(&mut bytes)? == PLAN_SCHEMA,
            "unsupported plan schema",
        )?;
        require(
            u32_at(&mut bytes)? as usize == COLUMN_NAMES.len(),
            "binary plan: column count of another format",
        )?;
        require(
            u32_at(&mut bytes)? == 0,
            "binary plan: reserved header field",
        )?;
        let mut cursors = Vec::with_capacity(COLUMN_NAMES.len());
        let mut budget = 0usize;
        for _ in COLUMN_NAMES {
            let head = take(&mut bytes, 8)?;
            let width = usize::from(head[0]);
            require(
                matches!(width, 0 | 1 | 2 | 4 | 8)
                    && head[1] <= 1
                    && head[2..].iter().all(|&b| b == 0),
                "binary plan: column header",
            )?;
            let count = usize::try_from(u64_at(&mut bytes)?)
                .map_err(|_| invalid("binary plan: column length"))?;
            let base = u64_at(&mut bytes)? as i64;
            let size = count
                .checked_mul(width)
                .ok_or_else(|| invalid("binary plan: column length"))?;
            let payload = take(&mut bytes, size)?;
            take(&mut bytes, (8 - size % 8) % 8)?;
            budget = budget.saturating_add(count);
            cursors.push(Cursor {
                payload,
                width,
                base,
                delta: head[1] == 1,
                count,
                position: 0,
                previous: 0,
            });
        }
        require(bytes.is_empty(), "binary plan: trailing bytes")?;
        let mut reader = Self {
            cursors,
            strings: Vec::new(),
            budget,
            tables: Vec::new(),
        };
        let count = reader.cursors[Col::StringLen as usize].count;
        let mut strings = Vec::with_capacity(count);
        for _ in 0..count {
            let len = reader.next(Col::StringLen)?;
            let len = usize::try_from(len).map_err(|_| invalid("binary plan: string length"))?;
            let mut text = Vec::with_capacity(len.min(reader.budget));
            for _ in 0..len {
                let byte = reader.next(Col::StringByte)?;
                text.push(u8::try_from(byte).map_err(|_| invalid("binary plan: string byte"))?);
            }
            strings.push(String::from_utf8(text).map_err(|_| invalid("binary plan: string"))?);
        }
        reader.strings = strings;
        Ok(reader)
    }

    fn next(&mut self, col: Col) -> RusticolResult<i64> {
        self.cursors[col as usize].next(col as usize)
    }

    /// Every column must be read to its end.
    fn finish(self) -> RusticolResult<()> {
        for (k, cursor) in self.cursors.iter().enumerate() {
            if cursor.position != cursor.count {
                return Err(invalid(format!(
                    "binary plan: column {} has unread values",
                    COLUMN_NAMES[k]
                )));
            }
        }
        Ok(())
    }
}

impl Stream for Reader<'_> {
    const READ: bool = true;

    fn int(&mut self, col: Col, value: &mut i64) -> RusticolResult<()> {
        *value = self.next(col)?;
        Ok(())
    }

    fn string(&mut self, col: Col, value: &mut String) -> RusticolResult<()> {
        let id = self.next(col)?;
        let text = usize::try_from(id)
            .ok()
            .and_then(|id| self.strings.get(id))
            .ok_or_else(|| invalid(format!("binary plan: {col:?} string id")))?;
        value.clone_from(text);
        Ok(())
    }

    fn len(&mut self, col: Col, _len: usize) -> RusticolResult<usize> {
        let len = self.next(col)?;
        usize::try_from(len)
            .ok()
            .filter(|&len| len <= self.budget)
            .ok_or_else(|| invalid(format!("binary plan: {col:?} list length")))
    }

    fn tables(&mut self) -> &mut Vec<Option<Box<dyn Any>>> {
        &mut self.tables
    }
}

// ------------------------------------------------------------ empty records

/// The traversal fills fresh records when reading.
macro_rules! empty_records {
    ($($name:ident { $($field:ident),* $(,)? }),* $(,)?) => {
        $(impl Default for $name {
            fn default() -> Self {
                Self { $($field: Default::default()),* }
            }
        })*
    };
}

empty_records! {
    PlanFile {
        schema, legs, rows, helicities, columns, average, born_components, constants, masses,
        block_rows, kernels, contents, factors, seed_classes, seeds, families, forward, merged,
        column_rows,
    },
    KernelFile { inputs, outputs },
    ContentFile { d_in, d_out, degree, rows, mask, row_helicities, fused, terms },
    TermFile { slots, dimensions, bindings, vertex, split, rows },
    SplitFile { kernel, inputs, entries },
    BindingFile { kernel, inputs, entries },
    SeedFile { columns },
    SeedKeyFile { classes },
    FamilyFile { sequence, rank, orders, leaves, steps },
    LeafFile { seed, weight },
    StepFile { seeded, trace, out, rows, d_alpha, d_out, degree, rank_in, edges },
    EdgeFile { leaf, child, factor, links },
    ForwardFile { states },
    StateFile { parent, factor, seeded, rows, d_alpha, d_out, degree, rank, links, closures },
    ClosureFile { family, accumulator, seed, weight, rows, relabel },
    MergedFile { families, entries, states, chunks },
    EntryFile { seed, weight },
    MergedStateFile { seeded, rows, d_alpha, d_out, degree, rank_in, edges, closure },
    MergedEdgeFile { entry, parent, factor, weight, links },
    MergedClosureFile { family, accumulator, relabel },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn columns_pick_the_narrowest_form() {
        let mut out = Vec::new();
        encode_column(&mut out, &[7, 7, 7]);
        assert_eq!(out[0], 0, "a constant column stores no payload");
        assert_eq!(out.len(), 24);
        out.clear();
        encode_column(&mut out, &[1_000_000, 1_000_001, 1_000_002, 1_000_003]);
        assert_eq!((out[0], out[1]), (1, 0), "values above their minimum");
        out.clear();
        encode_column(&mut out, &[70_000, 140_000, 210_000, 280_001]);
        assert_eq!((out[0], out[1]), (1, 1), "differences of an arithmetic run");
        out.clear();
        encode_column(&mut out, &[i64::MIN, i64::MAX]);
        assert_eq!((out[0], out[1]), (8, 0));
    }

    #[test]
    fn refuses_other_members() {
        assert!(PlanModel::from_binary(b"LOOPPLAX").is_err());
        let mut member = MAGIC.to_vec();
        member.extend_from_slice(&(FORMAT + 1).to_le_bytes());
        assert!(PlanModel::from_binary(&member).is_err());
    }
}
