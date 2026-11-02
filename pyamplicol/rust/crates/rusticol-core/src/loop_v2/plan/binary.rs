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
//! - **Binary member** ([`PlanModel::binary`], [`read_binary`]):
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
//! column count `u32`, `u32` zero, the decoded size `u64` and the padding
//! length `u64`; then per column (in the order of `Col`) a width `u8` (0, 1,
//! 2, 4 or 8), a delta flag `u8`, `u16` and `u32` zero, the value count
//! `u64`, the base `i64`, and `count × width` payload bytes padded to a
//! multiple of eight; then the padding's zero bytes. Writing and reading share
//! one traversal (`visit_plan` over a `Stream`), so the two cannot disagree on
//! the order.
//!
//! **Bounded decoding.** The member is a compressed encoding (a width-0
//! column holds any count of one value without payload, a shared list is
//! copied at every reference), so its decoded size is not a function of its
//! length alone. The member therefore declares the size of its decoded model:
//! the bytes of every allocation the decoding makes (list buffers at their
//! item size, string bytes, the string table, the copies of shared lists and
//! their table), each plus [`ALLOCATION`] bytes. The reader refuses a member
//! that declares more than [`DECODED_PER_BYTE`] bytes per member byte, charges
//! every allocation against the declaration *before* making it (a list's
//! length before its buffer, a copy before it is made) and refuses the member
//! at the first charge beyond it; at the end the charges must equal the
//! declaration. A member of `n` bytes thus decodes into at most
//! `DECODED_PER_BYTE × n` bytes (plus the allocator's rounding) in time
//! linear in `n`, whatever it declares; a decompression bomb is refused with
//! an error before it allocates. The writer computes the same charges in the
//! same traversal and pads a member whose plan shares more than the bound
//! allows (zero bytes after the columns), so every plan has a member and none
//! is refused (no plan written today needs padding: 18-43 decoded bytes per
//! member byte at five to seven legs).
//!
//! **Parallel decoding.** The top-level lists whose records hold no shared
//! list (seed classes, seed keys, families) decode in parallel chunks when
//! the read has more than one thread and the list at least
//! [`PARALLEL_RECORDS`] records: a sequential skip pass reads, checks and
//! charges every value of the list without building records (so the bound
//! holds before any of its records allocates) and notes the cursors' state at
//! each chunk's start; the chunks then decode from those states, each
//! required to end at the next one's with exactly the charges the pass made.
//! The model is the sequential decoding's for any thread count. A read
//! decodes on one thread where an allocation can fail before the host's
//! memory is exhausted (an address-space or data limit, strict overcommit:
//! `read_threads`): the parallel decode adds per-thread state the probes do
//! not see (the workers' malloc arenas, created lazily, and rayon's queues),
//! so only the sequential decode, which charges, probes and makes fallibly
//! every allocation, keeps the fail-closed contract there. Elsewhere the
//! per-chunk state is still allocated fallibly, and a section whose chunks
//! cannot allocate it, or fail, is decoded again sequentially from its start.
//! A thread pool that cannot start makes every read of the process decode on
//! one thread.
//!
//! **Bounded derivation.** The loader's derivations
//! ([`LoopProgramV4::from_file`], shared with the JSON load) size tables by
//! plan scalars (the row links per Born row, the product of the legs'
//! helicity counts, expanded over the observable columns; the split parts per
//! content term; the monomial tables per (rank, degree); the column-row flags;
//! the vertex table; a decimal weight's power of ten) and by the plan's lists
//! at several times their decoded size (segment layouts and their sets per
//! fused entry, vertex tables per split entry, parsed weights, the schedules'
//! structs, maps and B-trees). The member pays for them as for its decoded
//! size: `derive::bound` computes, in one pass over the decoded model, an
//! upper bound of the bytes those derivations allocate (in all, and held at
//! their peak) and of the iterations of their scalar-driven loops
//! ([`DeriveBound`]), after refusing the values they would trust beyond the
//! plan's tables (vertex ids, family and step ranks, fused-entry q-degrees,
//! decimal powers; `derive` module docs). [`PlanModel::program`] refuses a
//! member whose bound exceeds [`DERIVED_PER_BYTE`] per member byte, and
//! probes the bound's peak fallibly before deriving (with its [`margin`]: an
//! eighth more and 16 MiB); the writer pads a plan that needs more (none
//! written today: a few hundred bytes per member byte at five to seven legs).
//! A member of `n` bytes thus makes the loader allocate at most
//! `DERIVED_PER_BYTE × n` bytes beyond its decoded model, and a plan whose
//! derivations the allocator cannot grant is refused with an error. A probe
//! is a check, not a reservation (it allocates and releases at once), and it
//! discriminates under an address-space limit (`RLIMIT_AS`): under Linux's
//! default overcommit without one it succeeds for any size up to about the
//! host's RAM plus swap, where the per-member-byte bounds are the guard. The
//! decoded model is probed likewise before decoding, with the thread pool
//! started first. The production JSON load (`LoopProgramV4::load`, `plan.rs`)
//! takes the same refusals and probe through [`PlanModel::program`], without
//! the per-byte bound (a JSON cannot be padded), and the derivations reserve
//! their scalar-sized tables fallibly.
//!
//! Reading fails closed with an error, never a panic or an allocation abort:
//! headers, widths, payload sizes and the padding are checked, every value is
//! checked against its field's type, every column must be read to its end, a
//! string takes its bytes from what the byte column still holds, the list
//! reservations and copies are fallible (`TryCopy`), the string table is
//! shared as decoded (one fixed-size `Arc` header besides it), and the thread
//! pool is started fallibly (`read_threads`: a pool that cannot start leaves
//! the read on one thread, never rayon's panic). This holds for [`read_binary`]
//! ([`PlanModel::from_binary`] and [`PlanModel::program`]); the test tool
//! [`program_digest`] is outside it (it formats the loaded program, a
//! multi-precision weight as decimal digits, beyond the derivations' bound).

mod derive;
mod json;

pub use derive::{DERIVED_PER_BYTE, DeriveBound, RANKS, margin};
pub use json::OMITTED_WHEN_ABSENT;

use super::{
    BindingFile, ClosureFile, ContentFile, EdgeFile, EntryFile, FamilyFile, FormPolicy,
    ForwardFile, KernelFile, LeafFile, LoopProgramV4, MergedClosureFile, MergedEdgeFile,
    MergedFile, MergedStateFile, PLAN_SCHEMA, PlanFile, SeedFile, SeedKeyFile, SplitFile,
    StateFile, StepFile, TermFile, invalid, require,
};
use crate::RusticolResult;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::any::Any;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

/// The first bytes of a binary plan member.
pub const MAGIC: [u8; 8] = *b"LOOPPLAN";
/// The binary member's format (one version; another is refused). Format 3
/// adds the plan's `column_weights` (three columns after the column rows).
pub const FORMAT: u32 = 3;

/// The most bytes of decoded model a member may declare per member byte
/// (module docs, "Bounded decoding").
pub const DECODED_PER_BYTE: u64 = 128;

/// The bytes charged per allocation of the decoded model besides its payload
/// (the allocator's bookkeeping); an empty list or string allocates nothing.
pub const ALLOCATION: usize = 16;

/// The header's length: magic, format, schema, column count, reserved, the
/// decoded size and the padding length.
const HEADER: usize = 40;

/// The fewest records of a top-level list the reader decodes in parallel
/// chunks ([`Reader::records`]); shorter lists decode sequentially.
const PARALLEL_RECORDS: usize = 256;

/// Whether `bytes` start like a binary plan member (else they are JSON).
pub fn is_binary(bytes: &[u8]) -> bool {
    bytes.starts_with(&MAGIC)
}

/// The member's column names, in member order (diagnostics and tests).
pub fn column_names() -> &'static [&'static str] {
    COLUMN_NAMES
}

/// A plan v4 in the loader's file form: the input of [`LoopProgramV4`]'s
/// validation and derivations, written as the exporter's JSON or as the
/// binary columnar member.
pub struct PlanModel {
    file: PlanFile,
    /// The length of the member it was decoded from (none for a JSON).
    member: Option<u64>,
}

impl PlanModel {
    /// The model of a plan v4 JSON (`ExportedPlan.json()`), parsed as the
    /// loader parses it (unknown fields refused).
    pub fn from_json(json: &[u8]) -> RusticolResult<Self> {
        let file = serde_json::from_slice(json).map_err(|error| invalid(error.to_string()))?;
        Ok(Self { file, member: None })
    }

    /// The model of a plan's records read by any `serde` deserializer of the
    /// loader's file form (unknown fields refused, as [`Self::from_json`]):
    /// the exporter's Python records without a JSON in between (the
    /// `V2PlanModel.from_program` binding).
    pub fn from_records<'de, D: serde::Deserializer<'de>>(records: D) -> Result<Self, D::Error> {
        Ok(Self {
            file: serde::Deserialize::deserialize(records)?,
            member: None,
        })
    }

    /// The model of a binary plan member ([`Self::binary`]).
    pub fn from_binary(bytes: &[u8]) -> RusticolResult<Self> {
        let mut reader = Reader::new(bytes)?;
        let mut file = PlanFile::fresh();
        visit_plan(&mut reader, &mut file)?;
        reader.finish()?;
        Ok(Self {
            file,
            member: Some(bytes.len() as u64),
        })
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
        // A plan the bound refuses gets no padding for it: its reader
        // refuses it alike.
        let derived = derive::bound(&self.file).map_or(0, |bound| bound.cost());
        writer.finish(derived)
    }

    /// An upper bound of what the loader's derivations allocate and iterate
    /// for this plan, after refusing the scalars they would trust beyond the
    /// plan's tables (module docs, "Bounded derivation").
    pub fn derive_bound(&self) -> RusticolResult<DeriveBound> {
        derive::bound(&self.file)
    }

    /// The validated program with its derived tables (`LoopProgramV4::load`),
    /// once the derivations are bounded: a model decoded from a member must
    /// not need more than [`DERIVED_PER_BYTE`] per member byte, and the
    /// allocator must grant the bound's peak (a fallible probe, with its
    /// margin) before deriving (module docs, "Bounded derivation").
    pub fn program(self, policy: FormPolicy) -> RusticolResult<LoopProgramV4> {
        // A plan of another schema is refused as such before its bound.
        require(self.file.schema == PLAN_SCHEMA, "unsupported plan schema")?;
        let bound = derive::bound(&self.file)?;
        if let Some(member) = self.member {
            require(
                bound.cost() <= DERIVED_PER_BYTE.saturating_mul(member),
                "binary plan: the loader's derivations exceed the member's bound",
            )?;
        }
        derive::reserve("the loader's derivations need", bound.memory)?;
        LoopProgramV4::from_file(self.file, policy)
    }

    /// [`Self::program`] without the probe of the derivations' peak (the
    /// bound's refusals kept): `LoopProgramV4::load_unprobed`.
    pub(super) fn program_unprobed(self, policy: FormPolicy) -> RusticolResult<LoopProgramV4> {
        require(self.file.schema == PLAN_SCHEMA, "unsupported plan schema")?;
        derive::bound(&self.file)?;
        LoopProgramV4::from_file(self.file, policy)
    }
}

/// The validated program of a binary plan member ([`PlanModel::binary`])
/// with the loader's default kernel-form policy ([`LoopProgramV4::load`]'s):
/// the one reader of the member, for the loader and for an artifact's PACBIN
/// plan member. It decodes within the member's declared size and derives
/// within the member's derivation bound (module docs), else it fails with an
/// error. Re-exported as `loop_v2::plan::read_binary` (RUST-MIGRATION.md
/// §3.D).
pub fn read_binary(member: &[u8]) -> RusticolResult<LoopProgramV4> {
    read_binary_with(member, FormPolicy::Rule)
}

/// [`read_binary`] with an explicit kernel-form policy
/// ([`LoopProgramV4::load_with`]'s).
pub fn read_binary_with(member: &[u8], policy: FormPolicy) -> RusticolResult<LoopProgramV4> {
    PlanModel::from_binary(member)?.program(policy)
}

/// SHA-256 of a loaded program's validated and derived tables (its `Debug`
/// form, streamed, every field included): equal tables give equal digests
/// and any differing value changes it. A test tool for comparing the JSON and
/// binary loads; no artifact carries it. Not bounded by the member: the
/// `Debug` form writes a multi-precision weight in decimal (GMP's conversion,
/// through C `malloc`, several times the number's size), so a plan with a
/// weight of 10^8 digits takes it about 20 s and, under an address-space
/// limit the load itself fits in, can abort the process in GMP. Probes of the
/// reader's fail-closed contract read without it (`v2_plan_load` with
/// `digest=False`).
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
    RowHelicityRef, RowHelicityLen, RowHelicityLegs, RowHelicity, FusedRef, FusedLen, Fused0,
    Fused1, Fused2, Fused3,
    Fused4, Fused5, TermLen, SlotLen, Slot, DimensionRef, DimensionLen, Dimension,
    BindingLen, BindingKernel, BindingInputRef, BindingInputLen, BindingInputKind, BindingInputA,
    BindingInputB,
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
    // column weights
    ColumnWeightsFlag, ColumnWeightsLen, ColumnWeight,
}

// ------------------------------------------------------------------ streams

/// One direction of the shared traversal: the writer records the visited
/// values, the reader overwrites them with the member's.
trait Stream {
    const READ: bool;
    fn int(&mut self, col: Col, value: &mut i64) -> RusticolResult<()>;
    fn string(&mut self, col: Col, value: &mut String) -> RusticolResult<()>;
    /// A list length: written as `len`, or read.
    fn len(&mut self, col: Col, len: usize) -> RusticolResult<usize>;
    /// Charges `bytes` of decoded model (module docs, "Bounded decoding"):
    /// summed when writing; when reading, refused beyond the member's
    /// declared size, before the allocation they stand for is made.
    fn charge(&mut self, col: Col, bytes: usize) -> RusticolResult<()>;
    /// Per reference column, its table of shared lists (`shared`).
    fn tables(&mut self) -> &mut Vec<Option<Box<dyn Any>>>;
    /// Whether the reader only reads and charges, without building records
    /// (the skip pass of [`Reader::records`]).
    fn skipping(&self) -> bool {
        false
    }
    /// A top-level list whose records hold no shared list: [`list`], which
    /// the reader may decode in parallel chunks ([`Reader::records`]).
    fn records<T, F>(&mut self, col: Col, items: &mut Vec<T>, visit: F) -> RusticolResult<()>
    where
        Self: Sized,
        T: Fresh + Send,
        F: Fn(&mut Self, &mut T) -> RusticolResult<()> + Sync,
    {
        list(self, col, items, visit)
    }
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

#[inline]
fn num<S: Stream, T>(stream: &mut S, col: Col, value: &mut T) -> RusticolResult<()>
where
    T: Copy + TryInto<i64> + TryFrom<i64>,
{
    let mut wide: i64 = if S::READ {
        0
    } else {
        (*value)
            .try_into()
            .map_err(|_| invalid(format!("{col:?} out of the binary member's range")))?
    };
    stream.int(col, &mut wide)?;
    *value =
        T::try_from(wide).map_err(|_| invalid(format!("binary plan: {col:?} out of range")))?;
    Ok(())
}

#[inline]
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

#[inline]
fn list<S: Stream, T: Fresh>(
    stream: &mut S,
    col: Col,
    items: &mut Vec<T>,
    visit: impl FnMut(&mut S, &mut T) -> RusticolResult<()>,
) -> RusticolResult<()> {
    let len = stream.len(col, items.len())?;
    // A crafted length is refused here, beyond the declared size, before its
    // buffer is reserved (and a damaged one at its first exhausted column).
    stream.charge(col, allocation(len, size_of::<T>()))?;
    fill(stream, col, len, items, visit)
}

/// The records of a list whose length `len` is read and charged.
#[inline]
fn fill<S: Stream, T: Fresh>(
    stream: &mut S,
    col: Col,
    len: usize,
    items: &mut Vec<T>,
    mut visit: impl FnMut(&mut S, &mut T) -> RusticolResult<()>,
) -> RusticolResult<()> {
    if !S::READ {
        return items.iter_mut().try_for_each(|item| visit(stream, item));
    }
    if stream.skipping() {
        // Read and charged only: a scratch record, no buffer.
        for _ in 0..len {
            visit(stream, &mut T::fresh())?;
        }
        return Ok(());
    }
    *items = reserved(len, col)?;
    // Each record is filled in place in the reserved buffer.
    for _ in 0..len {
        items.push(T::fresh());
        if let Some(item) = items.last_mut() {
            visit(stream, item)?;
        }
    }
    Ok(())
}

#[inline]
fn nums<S: Stream, T>(stream: &mut S, len: Col, col: Col, items: &mut Vec<T>) -> RusticolResult<()>
where
    T: Copy + Fresh + TryInto<i64> + TryFrom<i64>,
{
    list(stream, len, items, |s, item| num(s, col, item))
}

/// A list stored once at its first use and referenced by number afterwards.
/// Every reference copies the list, so each is charged the copy's size (the
/// list's [`TryCopy::heap`], kept in the table), and a first use also its
/// table entry: a member cannot reference one long list from any number of
/// records beyond its declared size.
#[inline]
fn shared<S: Stream, T>(
    stream: &mut S,
    refs: Col,
    len: Col,
    items: &mut Vec<T>,
    visit: impl FnMut(&mut S, &mut T) -> RusticolResult<()>,
) -> RusticolResult<()>
where
    T: TryCopy + Fresh + Eq + Hash + 'static,
{
    // The table entry of a list (the reader's `(list, heap)`, doubling).
    let entry = 2 * size_of::<(Vec<T>, usize)>();
    if stream.skipping() {
        return Err(invalid("binary plan: a shared list in a parallel section"));
    }
    if S::READ {
        let mut id = 0i64;
        stream.int(refs, &mut id)?;
        let known: &mut Vec<(Vec<T>, usize)> = table(stream, refs);
        let first = known.len();
        match usize::try_from(id) {
            Ok(k) if k < first => {
                let heap = known[k].1;
                stream.charge(refs, heap)?;
                let known: &mut Vec<(Vec<T>, usize)> = table(stream, refs);
                *items = known[k].0.try_copy(refs)?;
                Ok(())
            }
            Ok(k) if k == first => {
                list(stream, len, items, visit)?;
                let heap = items.heap();
                stream.charge(refs, heap.saturating_add(entry))?;
                let copy = items.try_copy(refs)?;
                let known: &mut Vec<(Vec<T>, usize)> = table(stream, refs);
                known.try_reserve(1).map_err(|_| too_large(refs))?;
                known.push((copy, heap));
                Ok(())
            }
            _ => Err(invalid(format!("binary plan: {refs:?} reference"))),
        }
    } else {
        let known: &mut HashMap<Vec<T>, (i64, usize)> = table(stream, refs);
        let next = known.len() as i64;
        let (mut id, heap) = match known.get(items.as_slice()) {
            Some(&known) => known,
            None => {
                let heap = items.heap();
                known.insert(items.try_copy(refs)?, (next, heap));
                (next, heap)
            }
        };
        stream.int(refs, &mut id)?;
        if id == next {
            list(stream, len, items, visit)?;
            stream.charge(refs, heap.saturating_add(entry))
        } else {
            stream.charge(refs, heap)
        }
    }
}

#[inline]
fn option<S: Stream, T: Fresh>(
    stream: &mut S,
    col: Col,
    value: &mut Option<T>,
    visit: impl FnOnce(&mut S, &mut T) -> RusticolResult<()>,
) -> RusticolResult<()> {
    let mut present = value.is_some();
    flag(stream, col, &mut present)?;
    if S::READ {
        *value = present.then(T::fresh);
    }
    match value {
        Some(inner) => visit(stream, inner),
        None => Ok(()),
    }
}

// ------------------------------------------------------- fresh and copied

/// An empty value for a record the reader is about to fill. This module's own
/// trait (not `Default`), so it never collides with a `Default` the loader's
/// file types may derive; the records' impls list every field
/// (`fresh_records!`), so a field added to a file type without a column fails
/// to compile here.
trait Fresh {
    fn fresh() -> Self;
}

macro_rules! fresh_scalars {
    ($($ty:ty),* $(,)?) => {
        $(impl Fresh for $ty {
            fn fresh() -> Self {
                <$ty>::default()
            }
        })*
    };
}

fresh_scalars!(bool, i8, i32, i64, u32, u64, usize, String);

impl<T> Fresh for Vec<T> {
    fn fresh() -> Self {
        Vec::new()
    }
}

impl<T> Fresh for Option<T> {
    fn fresh() -> Self {
        None
    }
}

impl<T: Fresh, const N: usize> Fresh for [T; N] {
    fn fresh() -> Self {
        std::array::from_fn(|_| T::fresh())
    }
}

impl<A: Fresh, B: Fresh> Fresh for (A, B) {
    fn fresh() -> Self {
        (A::fresh(), B::fresh())
    }
}

impl<A: Fresh, B: Fresh, C: Fresh> Fresh for (A, B, C) {
    fn fresh() -> Self {
        (A::fresh(), B::fresh(), C::fresh())
    }
}

#[cold]
#[inline(never)]
fn too_large(col: Col) -> crate::RusticolError {
    invalid(format!(
        "binary plan: {col:?} decodes to more than the allocator grants"
    ))
}

/// The bytes charged for one allocation of `len` items of `size` bytes:
/// none for an empty list or string (no allocation), else the payload plus
/// [`ALLOCATION`] (saturating: an unrepresentable size exceeds any member).
#[inline]
fn allocation(len: usize, size: usize) -> usize {
    if len == 0 {
        0
    } else {
        len.saturating_mul(size.max(1)).saturating_add(ALLOCATION)
    }
}

/// An empty list with room for exactly `len` items, or an error.
#[inline]
fn reserved<T>(len: usize, col: Col) -> RusticolResult<Vec<T>> {
    let mut items = Vec::new();
    items.try_reserve_exact(len).map_err(|_| too_large(col))?;
    Ok(items)
}

/// A copy whose every allocation is fallible: of a shared list at each
/// reference and of a table string at each use (a crafted member can make
/// these copies outgrow memory; the error names the column).
trait TryCopy: Sized {
    fn try_copy(&self, col: Col) -> RusticolResult<Self>;

    /// The bytes [`allocation`] charges for a copy (zero for plain values).
    fn heap(&self) -> usize;

    /// The bytes charged for copies of the items of a list of `Self`.
    fn heap_list(items: &[Self]) -> usize {
        items
            .iter()
            .fold(0usize, |sum, item| sum.saturating_add(item.heap()))
    }

    /// A copy of a list of `Self` (item by item; `memcpy` for plain values).
    fn try_copy_list(items: &[Self], col: Col) -> RusticolResult<Vec<Self>> {
        let mut copy = reserved(items.len(), col)?;
        for item in items {
            copy.push(item.try_copy(col)?);
        }
        Ok(copy)
    }
}

macro_rules! copy_plain {
    ($($ty:ty),* $(,)?) => {
        $(impl TryCopy for $ty {
            fn try_copy(&self, _col: Col) -> RusticolResult<Self> {
                Ok(*self)
            }

            fn heap(&self) -> usize {
                0
            }

            fn heap_list(_items: &[Self]) -> usize {
                0
            }

            fn try_copy_list(items: &[Self], col: Col) -> RusticolResult<Vec<Self>> {
                let mut copy = reserved(items.len(), col)?;
                copy.extend_from_slice(items);
                Ok(copy)
            }
        })*
    };
}

copy_plain!(i32, u32, usize, [u32; 6], (usize, usize));

/// Sets `value` to a copy of `text` whose allocation is fallible: the hot
/// path of the decoding (one copy per string use, 3.8 M at u u~ > g g g g g).
/// It allocates as `str::to_owned` does, with the allocator's refusal as an
/// error: `String::try_reserve_exact` is the safe equivalent, 12-20 % slower
/// on the whole decode at seven legs (its out-of-line growth path).
#[inline]
fn copy_text(text: &str, value: &mut String, col: Col) -> RusticolResult<()> {
    let len = text.len();
    if len == 0 {
        value.clear();
        return Ok(());
    }
    let layout = std::alloc::Layout::array::<u8>(len).map_err(|_| too_large(col))?;
    // SAFETY: `layout` has a non-zero size (`len > 0`).
    let data = unsafe { std::alloc::alloc(layout) };
    if data.is_null() {
        return Err(too_large(col));
    }
    // SAFETY: `data` is a fresh allocation of the global allocator with the
    // layout of `len` bytes, which `String` deallocates for capacity `len`;
    // it receives the `len` bytes of `text`, so it holds UTF-8 of length
    // `len` and does not overlap `text`.
    unsafe {
        std::ptr::copy_nonoverlapping(text.as_ptr(), data, len);
        *value = String::from_raw_parts(data, len, len);
    }
    Ok(())
}

impl TryCopy for String {
    fn try_copy(&self, col: Col) -> RusticolResult<Self> {
        let mut copy = String::new();
        copy_text(self, &mut copy, col)?;
        Ok(copy)
    }

    fn heap(&self) -> usize {
        allocation(self.len(), 1)
    }
}

impl<T: TryCopy> TryCopy for Vec<T> {
    fn try_copy(&self, col: Col) -> RusticolResult<Self> {
        T::try_copy_list(self, col)
    }

    fn heap(&self) -> usize {
        allocation(self.len(), size_of::<T>()).saturating_add(T::heap_list(self))
    }
}

impl<A: TryCopy, B: TryCopy, C: TryCopy> TryCopy for (A, B, C) {
    fn try_copy(&self, col: Col) -> RusticolResult<Self> {
        Ok((
            self.0.try_copy(col)?,
            self.1.try_copy(col)?,
            self.2.try_copy(col)?,
        ))
    }

    fn heap(&self) -> usize {
        self.0
            .heap()
            .saturating_add(self.1.heap())
            .saturating_add(self.2.heap())
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

/// The member's traversal of a plan. Every record is destructured whole (no
/// `..`), so a field added to the loader's file form does not compile until
/// the member encodes it.
fn visit_plan<S: Stream>(s: &mut S, file: &mut PlanFile) -> RusticolResult<()> {
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
        merged,
        column_rows,
        column_weights,
    } = file;
    num(s, Col::Schema, schema)?;
    num(s, Col::Legs, legs)?;
    num(s, Col::Rows, rows)?;
    list(s, Col::HelicityLegs, helicities, |s, leg| {
        nums(s, Col::HelicityLen, Col::Helicity, leg)
    })?;
    num(s, Col::Columns, columns)?;
    num(s, Col::Average, average)?;
    num(s, Col::BornComponents, born_components)?;
    num(s, Col::Constants, constants)?;
    num(s, Col::Masses, masses)?;
    nums(s, Col::BlockRowLen, Col::BlockRow, block_rows)?;
    list(s, Col::KernelLen, kernels, visit_kernel)?;
    list(s, Col::ContentLen, contents, visit_content)?;
    list(s, Col::FactorLen, factors, |s, (content, shift)| {
        num(s, Col::FactorContent, content)?;
        num(s, Col::FactorShift, shift)
    })?;
    s.records(
        Col::SeedClassLen,
        seed_classes,
        |s, SeedFile { columns }| {
            list(s, Col::SeedColumnLen, columns, |s, column| {
                list(s, Col::SeedWeightLen, column, |s, (born, re, im)| {
                    num(s, Col::SeedWeightBorn, born)?;
                    s.string(Col::SeedWeightRe, re)?;
                    s.string(Col::SeedWeightIm, im)
                })
            })
        },
    )?;
    s.records(Col::SeedKeyLen, seeds, |s, SeedKeyFile { classes }| {
        list(s, Col::SeedKeyClassLen, classes, |s, (class, re, im)| {
            num(s, Col::SeedKeyClass, class)?;
            s.string(Col::SeedKeyRe, re)?;
            s.string(Col::SeedKeyIm, im)
        })
    })?;
    s.records(Col::FamilyLen, families, visit_family)?;
    option(s, Col::ForwardFlag, forward, |s, ForwardFile { states }| {
        list(s, Col::StateLen, states, visit_state)
    })?;
    option(s, Col::MergedFlag, merged, visit_merged)?;
    option(s, Col::ColumnRowsFlag, column_rows, |s, columns| {
        list(s, Col::ColumnRowsLen, columns, |s, rows| {
            option(s, Col::ColumnRowFlag, rows, |s, rows| {
                nums(s, Col::ColumnRowLen, Col::ColumnRow, rows)
            })
        })
    })?;
    option(s, Col::ColumnWeightsFlag, column_weights, |s, weights| {
        nums(s, Col::ColumnWeightsLen, Col::ColumnWeight, weights)
    })
}

fn visit_kernel<S: Stream>(s: &mut S, kernel: &mut KernelFile) -> RusticolResult<()> {
    let KernelFile { inputs, outputs } = kernel;
    list(s, Col::KernelInputLen, inputs, |s, item| {
        input(
            s,
            [Col::KernelInputKind, Col::KernelInputA, Col::KernelInputB],
            item,
        )
    })?;
    list(s, Col::KernelOutputLen, outputs, |s, terms| {
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
    num(s, Col::ContentDIn, d_in)?;
    num(s, Col::ContentDOut, d_out)?;
    num(s, Col::ContentDegree, degree)?;
    num(s, Col::ContentRows, rows)?;
    num(s, Col::ContentMask, mask)?;
    shared(
        s,
        Col::RowHelicityRef,
        Col::RowHelicityLen,
        row_helicities,
        |s, row| nums(s, Col::RowHelicityLegs, Col::RowHelicity, row),
    )?;
    shared(s, Col::FusedRef, Col::FusedLen, fused, |s, entry| {
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
    })?;
    list(s, Col::TermLen, terms, visit_term)
}

fn visit_term<S: Stream>(s: &mut S, term: &mut TermFile) -> RusticolResult<()> {
    let TermFile {
        slots,
        dimensions,
        bindings,
        vertex,
        split,
        rows,
    } = term;
    nums(s, Col::SlotLen, Col::Slot, slots)?;
    shared(
        s,
        Col::DimensionRef,
        Col::DimensionLen,
        dimensions,
        |s, d| num(s, Col::Dimension, d),
    )?;
    list(s, Col::BindingLen, bindings, visit_binding)?;
    num(s, Col::TermVertex, vertex)?;
    list(s, Col::SplitLen, split, visit_split)?;
    list(s, Col::TermRowLen, rows, |s, (row, blocks)| {
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
    let BindingFile {
        kernel,
        inputs,
        entries,
    } = binding;
    num(s, Col::BindingKernel, kernel)?;
    shared(
        s,
        Col::BindingInputRef,
        Col::BindingInputLen,
        inputs,
        |s, item| input(s, INPUT, item),
    )?;
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
    let SplitFile {
        kernel,
        inputs,
        entries,
    } = split;
    num(s, Col::SplitKernel, kernel)?;
    shared(
        s,
        Col::SplitInputRef,
        Col::SplitInputLen,
        inputs,
        |s, item| input(s, INPUT, item),
    )?;
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
    let FamilyFile {
        sequence,
        rank,
        orders,
        leaves,
        steps,
    } = family;
    list(s, Col::SequenceLen, sequence, |s, (mask, mass)| {
        num(s, Col::SequenceMask, mask)?;
        num(s, Col::SequenceMass, mass)
    })?;
    num(s, Col::FamilyRank, rank)?;
    nums(s, Col::FamilyOrderLen, Col::FamilyOrder, orders)?;
    list(s, Col::LeafLen, leaves, |s, LeafFile { seed, weight }| {
        num(s, Col::LeafSeed, seed)?;
        s.string(Col::LeafWeight, weight)
    })?;
    list(s, Col::StepLen, steps, visit_step)
}

fn visit_step<S: Stream>(s: &mut S, step: &mut StepFile) -> RusticolResult<()> {
    let StepFile {
        seeded,
        trace,
        out,
        d_alpha,
        d_out,
        degree,
        rank_in,
        edges,
        rows,
    } = step;
    flag(s, Col::StepSeeded, seeded)?;
    num(s, Col::StepTrace, trace)?;
    num(s, Col::StepOut, out)?;
    option(s, Col::StepRowsFlag, rows, |s, rows| {
        num(s, Col::StepRows, rows)
    })?;
    num(s, Col::StepDAlpha, d_alpha)?;
    num(s, Col::StepDOut, d_out)?;
    num(s, Col::StepDegree, degree)?;
    num(s, Col::StepRankIn, rank_in)?;
    list(s, Col::EdgeLen, edges, |s, edge: &mut EdgeFile| {
        let EdgeFile {
            leaf,
            child,
            factor,
            links,
        } = edge;
        num(s, Col::EdgeLeaf, leaf)?;
        num(s, Col::EdgeChild, child)?;
        num(s, Col::EdgeFactor, factor)?;
        option(s, Col::EdgeLinksFlag, links, |s, links| {
            nums(s, Col::EdgeLinkLen, Col::EdgeLink, links)
        })
    })
}

fn visit_state<S: Stream>(s: &mut S, state: &mut StateFile) -> RusticolResult<()> {
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
    num(s, Col::StateParent, parent)?;
    num(s, Col::StateFactor, factor)?;
    flag(s, Col::StateSeeded, seeded)?;
    num(s, Col::StateRows, rows)?;
    num(s, Col::StateDAlpha, d_alpha)?;
    num(s, Col::StateDOut, d_out)?;
    num(s, Col::StateDegree, degree)?;
    num(s, Col::StateRank, rank)?;
    nums(s, Col::StateLinkLen, Col::StateLink, links)?;
    list(
        s,
        Col::ClosureLen,
        closures,
        |s, closure: &mut ClosureFile| {
            let ClosureFile {
                family,
                accumulator,
                seed,
                weight,
                rows,
                relabel,
            } = closure;
            num(s, Col::ClosureFamily, family)?;
            num(s, Col::ClosureAccumulator, accumulator)?;
            num(s, Col::ClosureSeed, seed)?;
            s.string(Col::ClosureWeight, weight)?;
            nums(s, Col::ClosureRowLen, Col::ClosureRow, rows)?;
            option(s, Col::ClosureRelabelFlag, relabel, |s, (sign, mask)| {
                num(s, Col::ClosureRelabelSign, sign)?;
                num(s, Col::ClosureRelabelMask, mask)
            })
        },
    )
}

fn visit_merged<S: Stream>(s: &mut S, merged: &mut MergedFile) -> RusticolResult<()> {
    let MergedFile {
        families,
        entries,
        states,
        chunks,
    } = merged;
    nums(s, Col::MergedFamilyLen, Col::MergedFamily, families)?;
    list(
        s,
        Col::EntryLen,
        entries,
        |s, EntryFile { seed, weight }| {
            num(s, Col::EntrySeed, seed)?;
            let [re, im] = weight;
            s.string(Col::EntryRe, re)?;
            s.string(Col::EntryIm, im)
        },
    )?;
    list(s, Col::MergedStateLen, states, visit_merged_state)?;
    nums(s, Col::ChunkLen, Col::Chunk, chunks)
}

fn visit_merged_state<S: Stream>(s: &mut S, state: &mut MergedStateFile) -> RusticolResult<()> {
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
    flag(s, Col::MergedSeeded, seeded)?;
    num(s, Col::MergedRows, rows)?;
    num(s, Col::MergedDAlpha, d_alpha)?;
    num(s, Col::MergedDOut, d_out)?;
    num(s, Col::MergedDegree, degree)?;
    num(s, Col::MergedRankIn, rank_in)?;
    list(
        s,
        Col::MergedEdgeLen,
        edges,
        |s, edge: &mut MergedEdgeFile| {
            let MergedEdgeFile {
                entry,
                parent,
                factor,
                links,
                weight,
            } = edge;
            num(s, Col::MergedEdgeEntry, entry)?;
            num(s, Col::MergedEdgeParent, parent)?;
            num(s, Col::MergedEdgeFactor, factor)?;
            option(s, Col::MergedEdgeWeightFlag, weight, |s, [re, im]| {
                s.string(Col::MergedEdgeRe, re)?;
                s.string(Col::MergedEdgeIm, im)
            })?;
            nums(s, Col::MergedEdgeLinkLen, Col::MergedEdgeLink, links)
        },
    )?;
    option(
        s,
        Col::MergedClosureFlag,
        closure,
        |s, closure: &mut MergedClosureFile| {
            let MergedClosureFile {
                family,
                accumulator,
                relabel,
            } = closure;
            num(s, Col::MergedClosureFamily, family)?;
            num(s, Col::MergedClosureAccumulator, accumulator)?;
            option(
                s,
                Col::MergedClosureRelabelFlag,
                relabel,
                |s, (sign, mask)| {
                    num(s, Col::MergedClosureRelabelSign, sign)?;
                    num(s, Col::MergedClosureRelabelMask, mask)
                },
            )
        },
    )
}

// ------------------------------------------------------------------ writing

#[derive(Default)]
struct Writer {
    columns: Vec<Vec<i64>>,
    strings: Vec<String>,
    ids: HashMap<String, i64>,
    tables: Vec<Option<Box<dyn Any>>>,
    /// The decoded size: the charges of the traversal (`Stream::charge`).
    decoded: u64,
}

impl Writer {
    fn column(&mut self, col: Col) -> &mut Vec<i64> {
        if self.columns.is_empty() {
            self.columns = vec![Vec::new(); COLUMN_NAMES.len()];
        }
        &mut self.columns[col as usize]
    }

    /// The member, padded so that it pays for its decoded size and for its
    /// derivations' bound `derived`.
    fn finish(mut self, derived: u64) -> RusticolResult<Vec<u8>> {
        let strings = std::mem::take(&mut self.strings);
        let mut decoded = self.decoded;
        let mut charge = |bytes: usize| decoded = decoded.saturating_add(bytes as u64);
        charge(allocation(strings.len(), size_of::<String>()));
        for text in &strings {
            charge(allocation(text.len(), 1));
            let len = text.len() as i64;
            self.column(Col::StringLen).push(len);
            let bytes = text.bytes().map(i64::from);
            self.column(Col::StringByte).extend(bytes);
        }
        let mut body = Vec::new();
        for k in 0..COLUMN_NAMES.len() {
            let values = self.columns.get(k).map(Vec::as_slice).unwrap_or(&[]);
            encode_column(&mut body, values);
        }
        // Zero bytes after the columns when the plan shares more than the
        // bound allows, so its member declares at most DECODED_PER_BYTE bytes
        // per member byte, or derives more than DERIVED_PER_BYTE per member
        // byte (none for the plans written today).
        let unpadded = (HEADER + body.len()) as u64;
        let least = decoded
            .div_ceil(DECODED_PER_BYTE)
            .max(derived.div_ceil(DERIVED_PER_BYTE));
        let padding = least.saturating_sub(unpadded).next_multiple_of(8);
        let size = usize::try_from(unpadded.saturating_add(padding))
            .ok()
            .filter(|&size| size <= isize::MAX as usize)
            .ok_or_else(|| {
                invalid("binary plan: the plan needs a member beyond any address space")
            })?;
        let mut out = Vec::new();
        out.try_reserve_exact(size).map_err(|_| {
            invalid("binary plan: the member's padding exceeds what the allocator grants")
        })?;
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&FORMAT.to_le_bytes());
        out.extend_from_slice(&PLAN_SCHEMA.to_le_bytes());
        out.extend_from_slice(&(COLUMN_NAMES.len() as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&decoded.to_le_bytes());
        out.extend_from_slice(&padding.to_le_bytes());
        out.extend_from_slice(&body);
        out.resize(size, 0);
        Ok(out)
    }
}

impl Stream for Writer {
    const READ: bool = false;

    fn int(&mut self, col: Col, value: &mut i64) -> RusticolResult<()> {
        self.column(col).push(*value);
        Ok(())
    }

    fn string(&mut self, col: Col, value: &mut String) -> RusticolResult<()> {
        self.decoded = self
            .decoded
            .saturating_add(allocation(value.len(), 1) as u64);
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

    fn charge(&mut self, _col: Col, bytes: usize) -> RusticolResult<()> {
        self.decoded = self.decoded.saturating_add(bytes as u64);
        Ok(())
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

#[derive(Clone, Copy)]
struct Cursor<'a> {
    payload: &'a [u8],
    width: usize,
    base: i64,
    delta: bool,
    count: usize,
    position: usize,
    previous: i64,
}

/// Why a column yields no value.
enum Short {
    Exhausted,
    Overflow,
}

impl Cursor<'_> {
    /// The next value: base plus the stored offset, plus the previous value
    /// for a column of differences, in checked `i64` arithmetic (a value of
    /// a valid member is an `i64`, and so is each stored difference).
    #[inline]
    fn next(&mut self) -> Result<i64, Short> {
        let at = self.position;
        if at >= self.count {
            return Err(Short::Exhausted);
        }
        self.position = at + 1;
        let bytes = self.payload;
        let offset = match self.width {
            0 => 0,
            1 => u64::from(bytes[at]),
            2 => u64::from(u16::from_le_bytes([bytes[2 * at], bytes[2 * at + 1]])),
            4 => {
                let word: [u8; 4] = bytes[4 * at..4 * at + 4].try_into().expect("four bytes");
                u64::from(u32::from_le_bytes(word))
            }
            _ => u64::from_le_bytes(bytes[8 * at..8 * at + 8].try_into().expect("eight bytes")),
        };
        let stored = self
            .base
            .checked_add_unsigned(offset)
            .ok_or(Short::Overflow)?;
        let value = if self.delta {
            self.previous.checked_add(stored).ok_or(Short::Overflow)?
        } else {
            stored
        };
        self.previous = value;
        Ok(value)
    }
}

struct Reader<'a> {
    cursors: Vec<Cursor<'a>>,
    strings: Arc<Vec<String>>,
    budget: Budget,
    tables: Vec<Option<Box<dyn Any>>>,
    /// In the skip pass of [`Reader::records`].
    skip: bool,
    /// The threads the read decodes with ([`read_threads`]).
    threads: usize,
}

/// The decoded size a member declares and the part of it charged so far.
struct Budget {
    declared: u64,
    used: u64,
}

impl Budget {
    /// Charges `bytes`, refused beyond the declared size.
    #[inline]
    fn charge(&mut self, col: Col, bytes: usize) -> RusticolResult<()> {
        match self.used.checked_add(bytes as u64) {
            Some(used) if used <= self.declared => {
                self.used = used;
                Ok(())
            }
            _ => Err(beyond_declared(col)),
        }
    }
}

/// Whether an allocation of this process can fail before the host's memory
/// is exhausted: under an address-space or data-segment limit (`RLIMIT_AS`,
/// `RLIMIT_DATA`) or strict overcommit accounting (Linux
/// `vm.overcommit_memory = 2`). Elsewhere an allocation the reader's probes
/// granted does not fail later (Linux's default overcommit; a cgroup limit
/// kills, it does not refuse).
fn allocations_can_fail() -> bool {
    #[cfg(unix)]
    {
        let limited = |resource| {
            let mut limit = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            // SAFETY: `getrlimit` writes the `rlimit` it is given.
            let status = unsafe { libc::getrlimit(resource, &mut limit) };
            status != 0 || limit.rlim_cur != libc::RLIM_INFINITY
        };
        if limited(libc::RLIMIT_AS) || limited(libc::RLIMIT_DATA) {
            return true;
        }
    }
    #[cfg(target_os = "linux")]
    {
        use std::io::Read;
        let mut mode = [0u8; 1];
        let strict = std::fs::File::open("/proc/sys/vm/overcommit_memory")
            .and_then(|mut file| file.read_exact(&mut mode))
            .is_ok_and(|()| mode[0] == b'2');
        if strict {
            return true;
        }
    }
    false
}

/// Whether the reader found the global thread pool up when it first needed
/// it ([`read_threads`]): started by it or by another component before.
static GLOBAL_POOL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// The reader's record of the global thread pool: `None` before a read
/// needed it (or where every read is sequential: [`read_threads`]), else
/// whether it was up. A diagnostic: a test that makes the pool's start fail
/// (no thread may be spawned) checks that the start failed, so that its
/// reads ran on one thread for that reason.
#[doc(hidden)]
pub fn read_pool() -> Option<bool> {
    GLOBAL_POOL.get().copied()
}

/// The threads a read decodes with (module docs, "Parallel decoding"): one
/// where an allocation can fail ([`allocations_can_fail`]), since only the
/// sequential decode charges, probes and makes fallibly every allocation it
/// makes (the parallel one adds per-thread state: the workers' malloc arenas,
/// created lazily after the probes, rayon's queues and the chunks' cursors);
/// else inside a pool's worker that pool's, else the global pool's, started
/// first. rayon starts the global pool lazily at its first use and panics
/// there when a worker cannot be spawned, and never starts it after a failed
/// start (any later use panics too): the reader starts it itself, and a
/// pool that cannot start, or whose start failed before (outside this
/// reader, revealed by one caught use), makes every read of the process
/// decode on one thread, never a refusal or rayon's panic. A pool another
/// component started first is used as it is.
fn read_threads() -> usize {
    if allocations_can_fail() {
        return 1;
    }
    if rayon::current_thread_index().is_some() {
        return rayon::current_num_threads();
    }
    let started =
        *GLOBAL_POOL.get_or_init(|| match rayon::ThreadPoolBuilder::new().build_global() {
            Ok(()) => true,
            Err(error) => match std::error::Error::source(&error) {
                Some(_) => false,
                None => std::panic::catch_unwind(rayon::current_num_threads).is_ok(),
            },
        });
    if started {
        rayon::current_num_threads()
    } else {
        1
    }
}

#[cold]
#[inline(never)]
fn beyond_declared(col: Col) -> crate::RusticolError {
    invalid(format!(
        "binary plan: {col:?} decodes beyond the member's declared size"
    ))
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
        let declared = u64_at(&mut bytes)?;
        require(
            declared <= DECODED_PER_BYTE.saturating_mul(member.len() as u64),
            "binary plan: declared decoded size beyond the member's bound",
        )?;
        let padding = usize::try_from(u64_at(&mut bytes)?)
            .map_err(|_| invalid("binary plan: padding length"))?;
        // The allocator must grant the declared model before decoding starts
        // (a probe, with its margin), the thread pool being up first.
        let threads = read_threads();
        derive::reserve("binary plan: the decoded model needs", declared)?;
        let mut cursors = Vec::with_capacity(COLUMN_NAMES.len());
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
        require(
            take(&mut bytes, padding)?.iter().all(|&b| b == 0),
            "binary plan: padding",
        )?;
        require(bytes.is_empty(), "binary plan: trailing bytes")?;
        let mut reader = Self {
            cursors,
            strings: Arc::new(Vec::new()),
            budget: Budget { declared, used: 0 },
            tables: Vec::new(),
            skip: false,
            threads,
        };
        // The string table, charged as the writer charges it; a string takes
        // its bytes from what the byte column still holds.
        let count = reader.cursors[Col::StringLen as usize].count;
        reader
            .budget
            .charge(Col::StringLen, allocation(count, size_of::<String>()))?;
        let mut strings = reserved(count, Col::StringLen)?;
        for _ in 0..count {
            let len = reader.next(Col::StringLen)?;
            let byte_cursor = &reader.cursors[Col::StringByte as usize];
            let left = byte_cursor.count - byte_cursor.position;
            let len = usize::try_from(len)
                .ok()
                .filter(|&len| len <= left)
                .ok_or_else(|| invalid("binary plan: string length"))?;
            reader.budget.charge(Col::StringByte, allocation(len, 1))?;
            let mut text: Vec<u8> = reserved(len, Col::StringByte)?;
            for _ in 0..len {
                let byte = reader.next(Col::StringByte)?;
                text.push(u8::try_from(byte).map_err(|_| invalid("binary plan: string byte"))?);
            }
            strings.push(String::from_utf8(text).map_err(|_| invalid("binary plan: string"))?);
        }
        // Shared as decoded (a fixed-size Arc header, no copy of the table).
        reader.strings = Arc::new(strings);
        Ok(reader)
    }

    #[inline]
    fn next(&mut self, col: Col) -> RusticolResult<i64> {
        self.cursors[col as usize].next().map_err(|short| {
            let name = COLUMN_NAMES[col as usize];
            invalid(match short {
                Short::Exhausted => format!("binary plan: column {name} is exhausted"),
                Short::Overflow => format!("binary plan: column {name} overflows"),
            })
        })
    }

    /// Every column must be read to its end, and the charges must equal the
    /// declared decoded size.
    fn finish(self) -> RusticolResult<()> {
        for (k, cursor) in self.cursors.iter().enumerate() {
            if cursor.position != cursor.count {
                return Err(invalid(format!(
                    "binary plan: column {} has unread values",
                    COLUMN_NAMES[k]
                )));
            }
        }
        require(
            self.budget.used == self.budget.declared,
            "binary plan: declared decoded size",
        )
    }

    /// The cursors' positions and running values and the charges so far, or
    /// `None` when the allocator refuses the list (the parallel decode then
    /// gives way to the sequential one).
    fn state(&self) -> Option<State> {
        let mut cursors = Vec::new();
        cursors.try_reserve_exact(self.cursors.len()).ok()?;
        cursors.extend(self.cursors.iter().map(|c| (c.position, c.previous)));
        Some((cursors, self.budget.used))
    }

    /// Whether the cursors and the charges are at `state`.
    fn at(&self, (cursors, used): &State) -> bool {
        self.budget.used == *used
            && self.cursors.len() == cursors.len()
            && self
                .cursors
                .iter()
                .zip(cursors)
                .all(|(c, &(position, previous))| c.position == position && c.previous == previous)
    }

    /// Places the cursors and the charges at a [`Self::state`].
    fn restore(&mut self, (cursors, used): &State) {
        for (cursor, &(position, previous)) in self.cursors.iter_mut().zip(cursors) {
            cursor.position = position;
            cursor.previous = previous;
        }
        self.budget.used = *used;
    }

    /// The `len` records of a top-level list in parallel chunks (module docs,
    /// "Parallel decoding"): `Ok(true)` once decoded, `Ok(false)` when a
    /// chunk's state cannot be allocated or a chunk fails (the caller decodes
    /// the section sequentially from its start), an error when the skip pass
    /// refuses the member.
    fn parallel<T, F>(
        &mut self,
        col: Col,
        len: usize,
        items: &mut Vec<T>,
        visit: &F,
    ) -> RusticolResult<bool>
    where
        T: Fresh + Send,
        F: Fn(&mut Self, &mut T) -> RusticolResult<()> + Sync,
    {
        let threads = self.threads;
        let chunk = len.div_ceil(8 * threads).max(1);
        // Skip pass: every value of the section read, checked and charged
        // against the declaration before any record allocates, and the
        // cursors' state at each chunk's start.
        let mut starts: Vec<State> = Vec::new();
        if starts.try_reserve_exact(len.div_ceil(chunk) + 1).is_err() {
            return Ok(false);
        }
        self.skip = true;
        for k in 0..len {
            if k % chunk == 0 {
                let Some(state) = self.state() else {
                    self.skip = false;
                    return Ok(false);
                };
                starts.push(state);
            }
            if let Err(error) = visit(self, &mut T::fresh()) {
                self.skip = false;
                return Err(error);
            }
        }
        self.skip = false;
        let Some(end) = self.state() else {
            return Ok(false);
        };
        starts.push(end);
        // The chunks, decoded in parallel from their start states: each must
        // end at the next one's with exactly the charges the pass made.
        let Ok(buffer) = reserved(len, col) else {
            return Ok(false);
        };
        *items = buffer;
        items.resize_with(len, T::fresh);
        let (cursors, strings) = (&self.cursors, &self.strings);
        let decoded = items
            .par_chunks_mut(chunk)
            .zip(starts.par_windows(2))
            .try_for_each(|(records, window)| {
                let [start, end] = window else {
                    return Err(());
                };
                let mut copy = Vec::new();
                copy.try_reserve_exact(cursors.len()).map_err(drop)?;
                copy.extend_from_slice(cursors);
                let mut reader = Reader {
                    cursors: copy,
                    strings: Arc::clone(strings),
                    budget: Budget {
                        declared: end.1,
                        used: 0,
                    },
                    tables: Vec::new(),
                    skip: false,
                    threads,
                };
                reader.restore(start);
                for record in records {
                    visit(&mut reader, record).map_err(drop)?;
                }
                if reader.at(end) { Ok(()) } else { Err(()) }
            });
        Ok(decoded.is_ok())
    }
}

/// The cursors' positions and running values, and the charges, at one point
/// of a read ([`Reader::state`]).
type State = (Vec<(usize, i64)>, u64);

impl Stream for Reader<'_> {
    const READ: bool = true;

    #[inline]
    fn int(&mut self, col: Col, value: &mut i64) -> RusticolResult<()> {
        *value = self.next(col)?;
        Ok(())
    }

    #[inline]
    fn string(&mut self, col: Col, value: &mut String) -> RusticolResult<()> {
        let id = self.next(col)?;
        let text = usize::try_from(id)
            .ok()
            .and_then(|id| self.strings.get(id))
            .ok_or_else(|| invalid(format!("binary plan: {col:?} string id")))?;
        self.budget.charge(col, allocation(text.len(), 1))?;
        if self.skip {
            return Ok(());
        }
        copy_text(text, value, col)
    }

    #[inline]
    fn len(&mut self, col: Col, _len: usize) -> RusticolResult<usize> {
        let len = self.next(col)?;
        usize::try_from(len).map_err(|_| invalid(format!("binary plan: {col:?} list length")))
    }

    #[inline]
    fn charge(&mut self, col: Col, bytes: usize) -> RusticolResult<()> {
        self.budget.charge(col, bytes)
    }

    fn tables(&mut self) -> &mut Vec<Option<Box<dyn Any>>> {
        &mut self.tables
    }

    fn skipping(&self) -> bool {
        self.skip
    }

    fn records<T, F>(&mut self, col: Col, items: &mut Vec<T>, visit: F) -> RusticolResult<()>
    where
        T: Fresh + Send,
        F: Fn(&mut Self, &mut T) -> RusticolResult<()> + Sync,
    {
        let len = self.len(col, 0)?;
        self.budget.charge(col, allocation(len, size_of::<T>()))?;
        if self.threads == 1 || self.skip || len < PARALLEL_RECORDS {
            return fill(self, col, len, items, visit);
        }
        // The section's start, where the sequential decode takes over when
        // the parallel one cannot run.
        let Some(section) = self.state() else {
            return fill(self, col, len, items, visit);
        };
        if self.parallel(col, len, items, &visit)? {
            return Ok(());
        }
        *items = Vec::new();
        self.restore(&section);
        fill(self, col, len, items, visit)
    }
}

// ------------------------------------------------------------ empty records

/// The traversal fills fresh records when reading.
macro_rules! fresh_records {
    ($($name:ident { $($field:ident),* $(,)? }),* $(,)?) => {
        $(impl Fresh for $name {
            fn fresh() -> Self {
                Self { $($field: Fresh::fresh()),* }
            }
        })*
    };
}

fresh_records! {
    PlanFile {
        schema, legs, rows, helicities, columns, average, born_components, constants, masses,
        block_rows, kernels, contents, factors, seed_classes, seeds, families, forward, merged,
        column_rows, column_weights,
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
    StepFile { seeded, trace, out, d_alpha, d_out, degree, rank_in, edges, rows },
    EdgeFile { leaf, child, factor, links },
    ForwardFile { states },
    StateFile { parent, factor, seeded, rows, d_alpha, d_out, degree, rank, links, closures },
    ClosureFile { family, accumulator, seed, weight, rows, relabel },
    MergedFile { families, entries, states, chunks },
    EntryFile { seed, weight },
    MergedStateFile { seeded, rows, d_alpha, d_out, degree, rank_in, edges, closure },
    MergedEdgeFile { entry, parent, factor, links, weight },
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
    fn cursors_refuse_exhausted_and_overflowing_columns() {
        let cursor = |payload: &'static [u8], width, base, delta, count| Cursor {
            payload,
            width,
            base,
            delta,
            count,
            position: 0,
            previous: 0,
        };
        let mut plain = cursor(&[0, 255], 1, -3, false, 2);
        assert!(matches!(plain.next(), Ok(-3)));
        assert!(matches!(plain.next(), Ok(252)));
        assert!(matches!(plain.next(), Err(Short::Exhausted)));
        let mut wide = cursor(&[1], 1, i64::MAX, false, 1);
        assert!(matches!(wide.next(), Err(Short::Overflow)));
        let mut deltas = cursor(&[0, 0, 1, 0], 2, i64::MAX - 1, true, 2);
        assert!(matches!(deltas.next(), Ok(v) if v == i64::MAX - 1));
        assert!(matches!(deltas.next(), Err(Short::Overflow)));
        let mut words = cursor(&[1, 0, 0, 0, 0, 0, 0, 128], 8, i64::MIN, false, 1);
        assert!(matches!(words.next(), Ok(1)));
    }

    /// The header offset of every column of a member, in member order.
    fn column_offsets(member: &[u8]) -> Vec<usize> {
        let mut at = HEADER;
        let mut offsets = Vec::new();
        for _ in COLUMN_NAMES {
            offsets.push(at);
            let width = usize::from(member[at]);
            let count = u64::from_le_bytes(member[at + 8..at + 16].try_into().unwrap());
            let size = width * count as usize;
            at += 24 + size + (8 - size % 8) % 8;
        }
        let padding = u64::from_le_bytes(member[32..40].try_into().unwrap());
        assert_eq!(at + padding as usize, member.len());
        offsets
    }

    fn header(member: &[u8], at: usize) -> u64 {
        u64::from_le_bytes(member[at..at + 8].try_into().unwrap())
    }

    fn empty_member() -> Vec<u8> {
        PlanModel {
            file: PlanFile::fresh(),
            member: None,
        }
        .binary()
        .unwrap()
    }

    /// Rewrites column `col`'s count and base (a width-0 column: no payload).
    fn declare(member: &mut [u8], offsets: &[usize], col: Col, count: u64, base: u64) {
        let at = offsets[col as usize];
        assert_eq!(member[at], 0, "{col:?} holds no payload");
        member[at + 8..at + 16].copy_from_slice(&count.to_le_bytes());
        member[at + 16..at + 24].copy_from_slice(&base.to_le_bytes());
    }

    /// The most a member of its length may declare.
    fn declare_most(member: &mut [u8]) {
        let most = DECODED_PER_BYTE * member.len() as u64;
        member[24..32].copy_from_slice(&most.to_le_bytes());
    }

    #[test]
    fn crafted_counts_fail_closed() {
        let member = empty_member();
        assert!(PlanModel::from_binary(&member).is_ok());
        let offsets = column_offsets(&member);
        let patch = |edits: &[(Col, usize, u64)], most: bool| {
            let mut crafted = member.clone();
            for &(col, field, value) in edits {
                let at = offsets[col as usize] + field;
                crafted[at..at + 8].copy_from_slice(&value.to_le_bytes());
            }
            if most {
                declare_most(&mut crafted);
            }
            crafted
        };
        for most in [false, true] {
            // Width-0 columns hold no payload: a column without values claims
            // 2^40, and the one content length (width 0) becomes 2^36 or 2^62.
            for length in [1u64 << 36, 1 << 62] {
                let edits = [(Col::Chunk, 8, 1 << 40), (Col::ContentLen, 16, length)];
                let error = PlanModel::from_binary(&patch(&edits, most)).err().unwrap();
                assert!(error.to_string().contains("declared size"), "{error}");
            }
            // A string table of 2^40 one-byte strings without bytes, and one
            // string longer than the byte column.
            let edits = [(Col::StringLen, 8, 1 << 40), (Col::StringLen, 16, 1)];
            assert!(PlanModel::from_binary(&patch(&edits, most)).is_err());
            let edits = [
                (Col::StringLen, 8, 1),
                (Col::StringLen, 16, 1 << 40),
                (Col::StringByte, 8, 4),
            ];
            assert!(PlanModel::from_binary(&patch(&edits, most)).is_err());
            // One string of 2^40 bytes, all from a width-0 byte column.
            let edits = [
                (Col::StringLen, 8, 1),
                (Col::StringLen, 16, 1 << 40),
                (Col::StringByte, 8, 1 << 40),
                (Col::StringByte, 16, u64::from(b'1')),
            ];
            let error = PlanModel::from_binary(&patch(&edits, most)).err().unwrap();
            assert!(error.to_string().contains("declared size"), "{error}");
        }
        // A declaration beyond the bound is refused at the header, and the
        // charges must equal the declaration.
        let mut crafted = member.clone();
        let beyond = DECODED_PER_BYTE * crafted.len() as u64 + 1;
        crafted[24..32].copy_from_slice(&beyond.to_le_bytes());
        let error = PlanModel::from_binary(&crafted).err().unwrap();
        assert!(error.to_string().contains("bound"), "{error}");
        let mut crafted = member.clone();
        declare_most(&mut crafted);
        let error = PlanModel::from_binary(&crafted).err().unwrap();
        assert!(
            error.to_string().contains("declared decoded size"),
            "{error}"
        );
    }

    /// Width-0 columns declare `n` contents that all reference one shared
    /// list of `m` empty rows (a member of a few kB, the evidence audit's
    /// 2.4e11-byte decompression bomb at n = 10^6, m = 10^4): the charges of
    /// the first content's list, or of the copies, exceed what the member may
    /// declare, so it is refused before it allocates them, whatever it
    /// declares.
    #[test]
    fn shared_list_bombs_are_refused_before_they_allocate() {
        let member = empty_member();
        let offsets = column_offsets(&member);
        for (n, m) in [(1_000_000u64, 10_000u64), (100_000, 100_000), (2_000, 30)] {
            for most in [false, true] {
                let mut crafted = member.clone();
                declare(&mut crafted, &offsets, Col::ContentLen, 1, n);
                for col in [
                    Col::ContentDIn,
                    Col::ContentDOut,
                    Col::ContentDegree,
                    Col::ContentRows,
                    Col::ContentMask,
                    Col::RowHelicityRef,
                    Col::FusedRef,
                    Col::TermLen,
                ] {
                    declare(&mut crafted, &offsets, col, n, 0);
                }
                declare(&mut crafted, &offsets, Col::RowHelicityLen, 1, m);
                declare(&mut crafted, &offsets, Col::RowHelicityLegs, m, 0);
                declare(&mut crafted, &offsets, Col::FusedLen, 1, 0);
                if most {
                    declare_most(&mut crafted);
                }
                assert_eq!(crafted.len(), member.len());
                let error = PlanModel::from_binary(&crafted).err().unwrap();
                assert!(error.to_string().contains("declared size"), "{error}");
            }
        }
    }

    /// The same structure written by the writer (`n` contents sharing one
    /// list of `m` empty rows) shares more than the bound allows: the writer
    /// pads the member, which then decodes to the same model.
    #[test]
    fn plans_that_share_beyond_the_bound_are_padded() {
        let (n, m) = (500, 400);
        let mut file = PlanFile::fresh();
        file.contents = (0..n)
            .map(|_| {
                let mut content = ContentFile::fresh();
                content.row_helicities = vec![Vec::new(); m];
                content
            })
            .collect();
        let mut model = PlanModel { file, member: None };
        let member = model.binary().unwrap();
        let (decoded, padding) = (header(&member, 24), header(&member, 32));
        assert!(padding > 0, "the plan needs padding");
        assert!(decoded <= DECODED_PER_BYTE * member.len() as u64);
        assert!(decoded > DECODED_PER_BYTE * (member.len() as u64 - padding));
        let again = PlanModel::from_binary(&member).unwrap();
        assert_eq!(again.json(), model.json());
        // Without its padding (and the declaration the bound then allows) it
        // is refused.
        let mut cut = member[..member.len() - padding as usize].to_vec();
        cut[32..40].copy_from_slice(&0u64.to_le_bytes());
        declare_most(&mut cut);
        assert!(PlanModel::from_binary(&cut).is_err());
        // A damaged padding byte is refused.
        let mut damaged = member.clone();
        *damaged.last_mut().unwrap() = 1;
        assert!(PlanModel::from_binary(&damaged).is_err());
        // An allocation the allocator refuses is an error, not an abort, and
        // the fallible copies equal their originals.
        assert!(reserved::<u64>(usize::MAX / 2, Col::Chunk).is_err());
        let rows = vec![vec![1i32, -1], vec![]];
        assert_eq!(rows.try_copy(Col::RowHelicityRef).unwrap(), rows);
        assert_eq!(rows.heap(), allocation(2, 24) + allocation(2, 4));
        let inputs = vec![("loop".to_string(), 1usize, 2usize)];
        assert_eq!(inputs.try_copy(Col::BindingInputRef).unwrap(), inputs);
    }

    /// Every truncation of a member is refused.
    #[test]
    fn truncated_members_are_refused() {
        let mut file = PlanFile::fresh();
        file.helicities = vec![vec![-1, 1]; 4];
        file.block_rows = vec![3, 1, 4];
        file.factors = vec![(0, 2), (1, 7)];
        let member = PlanModel { file, member: None }.binary().unwrap();
        assert!(PlanModel::from_binary(&member).is_ok());
        for len in 0..member.len() {
            assert!(PlanModel::from_binary(&member[..len]).is_err(), "{len}");
        }
    }

    /// A plan whose seed classes, seed keys and families are long enough for
    /// the parallel decode, with strings, nested lists and options.
    fn parallel_plan() -> PlanFile {
        let mut file = PlanFile::fresh();
        let weight = |k: usize| format!("{}/{}", k % 7, 1 + k % 5);
        file.seed_classes = (0..600)
            .map(|k| SeedFile {
                columns: vec![
                    (0..k % 9)
                        .map(|t| (t, weight(k + t), "0".to_string()))
                        .collect(),
                ],
            })
            .collect();
        file.seeds = (0..700)
            .map(|k| SeedKeyFile {
                classes: (0..k % 4).map(|c| (c, weight(c), weight(k))).collect(),
            })
            .collect();
        file.families = (0..900)
            .map(|k| {
                let mut family = FamilyFile::fresh();
                family.sequence = vec![(1 << (k % 5), k % 3)];
                family.rank = k % 4;
                family.leaves = (0..k % 6)
                    .map(|j| LeafFile {
                        seed: j,
                        weight: weight(j * k),
                    })
                    .collect();
                let mut step = StepFile::fresh();
                step.trace = k as i64 - 450;
                step.rows = (k % 2 == 0).then_some(k);
                step.edges = vec![EdgeFile {
                    leaf: -1,
                    child: k as i64,
                    factor: k % 11,
                    links: (k % 3 == 0).then(|| vec![1, 2, 3]),
                }];
                family.steps = vec![step];
                family
            })
            .collect();
        file
    }

    /// The parallel decode equals the sequential one (one thread), and a
    /// damaged member fails closed on it as well (RUST-MIGRATION.md rule 5).
    #[test]
    fn parallel_decode_equals_the_sequential_one() {
        let mut model = PlanModel {
            file: parallel_plan(),
            member: None,
        };
        let member = model.binary().unwrap();
        let json = model.json();
        let pool = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        };
        let (one, four) = (pool(1), pool(4));
        let sequential = one.install(|| PlanModel::from_binary(&member).unwrap());
        let parallel = four.install(|| PlanModel::from_binary(&member).unwrap());
        assert_eq!(sequential.json(), json);
        assert_eq!(parallel.json(), json);
        let mut again = parallel;
        assert_eq!(again.binary().unwrap(), member);
        let offsets = column_offsets(&member);
        let payload = offsets[Col::SeedClassLen as usize]..member.len();
        for at in payload.step_by(41) {
            let mut damaged = member.clone();
            damaged[at] ^= 0x5a;
            let results = [&one, &four].map(|pool| {
                pool.install(|| PlanModel::from_binary(&damaged).map(|model| model.json()))
            });
            match results {
                [Ok(a), Ok(b)] => assert_eq!(a, b, "byte {at}"),
                [Err(_), Err(_)] => {}
                _ => panic!("byte {at}: one and four threads disagree"),
            }
        }
    }

    /// The member of `file` padded for its decoded size and for derivations
    /// of `derived` bytes (the writer's padding with another bound).
    fn member_paying(file: &PlanFile, derived: u64) -> Vec<u8> {
        let mut writer = Writer::default();
        let mut file = PlanModel {
            file: json_copy(file),
            member: None,
        };
        visit_plan(&mut writer, &mut file.file).unwrap();
        writer.finish(derived).unwrap()
    }

    fn json_copy(file: &PlanFile) -> PlanFile {
        serde_json::from_slice(&json::write(file)).unwrap()
    }

    /// `legs` two-helicity legs, one family reading them all at once through
    /// `edges` leaf edges into a content on no leg (each edge links every
    /// Born row), `columns` observable columns.
    fn born_rows_plan(legs: usize, edges: usize, columns: usize) -> PlanFile {
        let mut file = PlanFile::fresh();
        (file.schema, file.legs, file.rows, file.columns) = (PLAN_SCHEMA, legs, 1 << legs, columns);
        file.helicities = vec![vec![-1, 1]; legs];
        let mut content = ContentFile::fresh();
        (content.rows, content.row_helicities) = (1, vec![Vec::new()]);
        content.terms = vec![TermFile::fresh()];
        file.contents = vec![content];
        file.factors = vec![(0, 0)];
        let mut family = FamilyFile::fresh();
        (family.sequence, family.rank) = (vec![((1 << legs) - 1, 0)], 1);
        let mut step = StepFile::fresh();
        step.seeded = true;
        step.edges = (0..edges)
            .map(|_| EdgeFile {
                leaf: 0,
                child: -1,
                factor: 0,
                links: None,
            })
            .collect();
        family.steps = vec![step];
        file.families = vec![family];
        file
    }

    /// Plans of a few kB whose derivations would take terabytes or crash:
    /// 2^40 Born rows, 2^30 columns, 2^20 x 2^12 column-row flags, a 2^40-th
    /// vertex, ranks of 10^6 or 2^63, a forward table of degree 10^6, fused
    /// exponents of 2^20 (a stack overflow in the layouts' monomial index),
    /// wrapping to 0 or beyond any rank, decimal powers of ten of 4 × 10^9
    /// and 5 × 10^9 (a numerica panic). Their unpadded members decode, and
    /// their reader refuses them before the derivations allocate (this
    /// process would abort otherwise).
    #[test]
    fn derivation_bombs_are_refused_before_they_allocate() {
        let mut bombs = vec![
            ("rows", born_rows_plan(40, 1, 1)),
            ("columns", born_rows_plan(4, 8, 1 << 30)),
        ];
        let mut flags = born_rows_plan(20, 0, 1 << 12);
        flags.families.clear();
        flags.column_rows = Some(vec![None; 1 << 12]);
        bombs.push(("column rows", flags));
        let mut vertex = born_rows_plan(4, 1, 1);
        vertex.contents[0].terms[0].vertex = 1 << 40;
        bombs.push(("vertex", vertex));
        let mut rank = born_rows_plan(4, 1, 1);
        rank.families[0].rank = 1_000_000;
        bombs.push(("family rank", rank));
        let mut step = born_rows_plan(4, 1, 1);
        (
            step.families[0].steps[0].rank_in,
            step.families[0].steps[0].degree,
        ) = (1 << 62, 1 << 62);
        bombs.push(("step rank", step));
        let mut merged = born_rows_plan(4, 1, 1);
        let mut state = MergedStateFile::fresh();
        state.rank_in = 1_000_000;
        merged.merged = Some(MergedFile {
            families: vec![0],
            entries: Vec::new(),
            states: vec![state],
            chunks: vec![1],
        });
        bombs.push(("merged rank", merged));
        let mut forward = born_rows_plan(4, 1, 1);
        let mut state = StateFile::fresh();
        (state.parent, state.degree, state.rank) = (-1, 1_000_000, 1_000_000);
        forward.forward = Some(ForwardFile {
            states: vec![state],
        });
        bombs.push(("forward degree", forward));
        // A fused entry's exponent recursed on by the layouts' monomial index
        // before the loader checks its content's degree (a stack overflow at
        // 2^20), exponents whose u32 sum wraps to 0, and a degree beyond any
        // family's rank that its unread content allows.
        for (name, exponents, degree) in [
            ("fused exponent", [0, 0, 0, 1 << 20], 0),
            ("fused wrap", [0, 1 << 31, 1 << 31, 0], 0),
            ("fused rank", [0, 0, 0, RANKS as u32], 1000),
        ] {
            let mut fused = born_rows_plan(4, 1, 1);
            let mut unread = ContentFile::fresh();
            (unread.d_in, unread.d_out, unread.degree) = (1, 1, degree);
            let [e0, e1, e2, e3] = exponents;
            unread.fused = vec![[0, 0, e0, e1, e2, e3]];
            fused.contents.push(unread);
            bombs.push((name, fused));
        }
        // A decimal weight's power of ten: 10^(4 × 10^9) is 1.7 GB to
        // compute; beyond 2^32 numerica panics.
        for (name, weight) in [
            ("decimal power", "1e4000000000"),
            ("power 2^32", "1e5000000000"),
        ] {
            let mut power = born_rows_plan(4, 1, 1);
            power.seed_classes = vec![SeedFile {
                columns: vec![vec![(0, weight.to_owned(), "0".to_owned())]],
            }];
            bombs.push((name, power));
        }
        for (name, file) in &bombs {
            let member = member_paying(file, 0);
            assert!(member.len() < 8192, "{name}: {}", member.len());
            let decoded = PlanModel::from_binary(&member).unwrap();
            assert_eq!(decoded.json(), json::write(file), "{name}");
            let error = read_binary(&member).err().unwrap().to_string();
            let expected = match *name {
                "vertex" => "vertex id",
                "family rank" => "family rank",
                "step rank" => "step above the family rank",
                "merged rank" => "a merged state above",
                "fused exponent" | "fused wrap" | "fused rank" => "q-degree above its content's",
                "power 2^32" => "power of ten beyond 2^32",
                _ => "exceed the member's bound",
            };
            assert!(error.contains(expected), "{name}: {error}");
        }
    }

    /// A plan whose derivations need more than its member pays for gets a
    /// padded member, which the reader accepts; without its padding it is
    /// refused. A bound the allocator cannot reserve is an error.
    #[test]
    fn members_pay_for_their_derivations() {
        let file = born_rows_plan(16, 4, 16);
        let mut model = PlanModel {
            file: json_copy(&file),
            member: None,
        };
        let bound = model.derive_bound().unwrap();
        let member = model.binary().unwrap();
        let padding = header(&member, 32);
        assert!(padding > 0, "the plan needs padding");
        let len = member.len() as u64;
        assert!(bound.cost() <= DERIVED_PER_BYTE * len);
        assert!(bound.cost() > DERIVED_PER_BYTE * (len - padding));
        assert_eq!(
            PlanModel::from_binary(&member)
                .unwrap()
                .derive_bound()
                .unwrap(),
            bound
        );
        // Past the bound, the loader's own validation decides (this plan's
        // content has no fused entry).
        let error = read_binary(&member).err().unwrap().to_string();
        assert!(!error.contains("bound"), "{error}");
        let cut = member_paying(&file, 0);
        assert!(cut.len() as u64 == len - padding);
        let error = read_binary(&cut).err().unwrap().to_string();
        assert!(error.contains("exceed the member's bound"), "{error}");
        // The empty plan derives next to nothing, and is no bomb.
        let empty = PlanModel {
            file: PlanFile::fresh(),
            member: None,
        };
        assert!(empty.derive_bound().unwrap().cost() < 1024);
        for bytes in [u64::MAX, 1 << 62] {
            assert!(derive::reserve("a test needs", bytes).is_err(), "{bytes}");
        }
        assert!(derive::reserve("a test needs", 1 << 20).is_ok());
    }

    #[test]
    fn refuses_other_members() {
        assert!(PlanModel::from_binary(b"LOOPPLAX").is_err());
        let mut member = MAGIC.to_vec();
        member.extend_from_slice(&(FORMAT + 1).to_le_bytes());
        assert!(PlanModel::from_binary(&member).is_err());
    }
}
