// SPDX-License-Identifier: 0BSD

//! PACBIN members of plan v4 loop programs (the v2 engine). The plan v3
//! runtime of the removed legacy one-loop engine (descriptor, loader and
//! evaluator boundary) was deleted with it (MQ-LEGACY-REMOVE, 2026-10-29).

#[cfg(feature = "loop-v2")]
use crate::pacbin::{PacbinMemberKind, PacbinReader};
#[cfg(feature = "loop-v2")]
use crate::{RusticolError, RusticolResult};

#[cfg(feature = "loop-v2")]
fn require(condition: bool, message: &str) -> RusticolResult<()> {
    if condition {
        Ok(())
    } else {
        Err(RusticolError::compatibility(format!("loop runtime plan: {message}")))
    }
}

/// PACBIN member of a v2 loop program (plan v4, DESIGN.md §3.8, §12.1): the
/// zlib stream of the JSON that `loops/v2/plan_export.py` writes
/// (`loops/v2/artifact.py` publishes it). Artifacts store the binary member
/// instead ([`LOOP_PROGRAM_V4_BINARY_MEMBER`]); this one is read where no
/// binary member is stored (artifacts written before it).
pub const LOOP_PROGRAM_V4_MEMBER: &str = "schedule/loop-plan-v4.json.zlib";

/// PACBIN member of the companion plan v4 of a process's certified-scaleless
/// families (`LoopSlice.scaleless`, DESIGN.md §12.4): only the "uv" and "ir"
/// pole selections run it. Absent when the process has no such family.
pub const LOOP_PROGRAM_V4_SCALELESS_MEMBER: &str = "schedule/loop-plan-v4-scaleless.json.zlib";

/// The plan v4 as its binary columnar member (`loop_v2::plan::binary`,
/// `ExportedPlan.binary()`), zlib stream: the plan's stored form (M-FINAL
/// MF-8G-1 item 5); `loop_v2::plan::LoopProgramV4::load` reads either form.
pub const LOOP_PROGRAM_V4_BINARY_MEMBER: &str = "schedule/loop-plan-v4.bin.zlib";

/// The companion plan's binary member (as [`LOOP_PROGRAM_V4_BINARY_MEMBER`]).
pub const LOOP_PROGRAM_V4_SCALELESS_BINARY_MEMBER: &str =
    "schedule/loop-plan-v4-scaleless.bin.zlib";

#[cfg(feature = "loop-v2")]
fn has_member(reader: &PacbinReader, name: &str) -> bool {
    reader
        .members()
        .iter()
        .any(|member| member.logical_path() == name)
}

/// The stored plan v4 of an authenticated PACBIN reader: its binary member
/// where stored, else its JSON (its only caller is the v4 admission,
/// `loop_artifact::load_loop_v2_artifact`, which digests these bytes and
/// validates them as a `loop_v2::plan::LoopProgramV4`). The member has the
/// plan kind of the v3 runtime; the JSON's schema field selects version 4.
#[cfg(feature = "loop-v2")]
pub fn read_loop_program_v4_json(reader: &PacbinReader) -> RusticolResult<Vec<u8>> {
    if has_member(reader, LOOP_PROGRAM_V4_BINARY_MEMBER) {
        return read_plan_v4_member(reader, LOOP_PROGRAM_V4_BINARY_MEMBER);
    }
    read_plan_v4_member(reader, LOOP_PROGRAM_V4_MEMBER)
}

/// The stored companion plan v4 (its binary member, else
/// `LOOP_PROGRAM_V4_SCALELESS_MEMBER`), or None when the PACBIN has neither.
#[cfg(feature = "loop-v2")]
pub fn read_loop_program_v4_scaleless_json(
    reader: &PacbinReader,
) -> RusticolResult<Option<Vec<u8>>> {
    for name in [
        LOOP_PROGRAM_V4_SCALELESS_BINARY_MEMBER,
        LOOP_PROGRAM_V4_SCALELESS_MEMBER,
    ] {
        if has_member(reader, name) {
            return read_plan_v4_member(reader, name).map(Some);
        }
    }
    Ok(None)
}

#[cfg(feature = "loop-v2")]
fn read_plan_v4_member(reader: &PacbinReader, name: &str) -> RusticolResult<Vec<u8>> {
    use std::io::Read;

    require(
        reader.member(name)?.kind() == PacbinMemberKind::LoopRuntimePlan,
        "wrong v4 program member kind",
    )?;
    let mut json = Vec::new();
    flate2::read::ZlibDecoder::new(reader.member_bytes(name)?)
        .read_to_end(&mut json)
        .map_err(|error| {
            RusticolError::integrity(format!("plan v4 member is not a zlib stream: {error}"))
        })?;
    Ok(json)
}
