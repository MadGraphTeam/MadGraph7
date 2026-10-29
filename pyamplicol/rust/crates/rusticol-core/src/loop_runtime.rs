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
/// (`loops/v2/artifact.py` publishes it).
pub const LOOP_PROGRAM_V4_MEMBER: &str = "schedule/loop-plan-v4.json.zlib";

/// PACBIN member of the companion plan v4 of a process's certified-scaleless
/// families (`LoopSlice.scaleless`, DESIGN.md §12.4): only the "uv" and "ir"
/// pole selections run it. Absent when the process has no such family.
pub const LOOP_PROGRAM_V4_SCALELESS_MEMBER: &str = "schedule/loop-plan-v4-scaleless.json.zlib";

/// The plan v4 JSON of an authenticated PACBIN reader (its only caller is the
/// v4 admission, `loop_artifact::load_loop_v2_artifact`, which validates it as
/// a `loop_v2::plan::LoopProgramV4`). The member has the plan kind of the v3
/// runtime; the JSON's schema field selects version 4.
#[cfg(feature = "loop-v2")]
pub fn read_loop_program_v4_json(reader: &PacbinReader) -> RusticolResult<Vec<u8>> {
    read_plan_v4_member(reader, LOOP_PROGRAM_V4_MEMBER)
}

/// The companion plan v4 JSON (`LOOP_PROGRAM_V4_SCALELESS_MEMBER`), or None
/// when the PACBIN has no such member.
#[cfg(feature = "loop-v2")]
pub fn read_loop_program_v4_scaleless_json(
    reader: &PacbinReader,
) -> RusticolResult<Option<Vec<u8>>> {
    if !reader
        .members()
        .iter()
        .any(|member| member.logical_path() == LOOP_PROGRAM_V4_SCALELESS_MEMBER)
    {
        return Ok(None);
    }
    read_plan_v4_member(reader, LOOP_PROGRAM_V4_SCALELESS_MEMBER).map(Some)
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
