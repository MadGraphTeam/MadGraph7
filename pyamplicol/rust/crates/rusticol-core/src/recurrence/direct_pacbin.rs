// SPDX-License-Identifier: 0BSD

//! PACBIN publication and authenticated loading for direct-plan v2.

use std::path::Path;

use super::direct_codec::{decode_recurrence_direct_plan_v2, encode_recurrence_direct_plan_v2};
use super::direct_plan::DirectRecurrencePlan;
use crate::pacbin::{
    PacbinMemberKind, PacbinReader, PacbinWriteMember, PacbinWriteOptions, write_pacbin_atomic,
};
use crate::{RusticolError, RusticolResult};

pub const RECURRENCE_DIRECT_PLAN_MEMBER: &str = "plan/recurrence-direct-plan-v2.bin";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceDirectPacbinMetadata {
    pub container_size: u64,
    pub member_count: u64,
    pub unpacked_size_bytes: u64,
    pub index_sha256: [u8; 32],
    pub plan_payload_size: u64,
    pub plan_sha256: [u8; 32],
}

pub fn write_recurrence_direct_plan_pacbin(
    destination: impl AsRef<Path>,
    plan: &DirectRecurrencePlan,
) -> RusticolResult<RecurrenceDirectPacbinMetadata> {
    let payload = encode_recurrence_direct_plan_v2(plan)?;
    let member = PacbinWriteMember::from_bytes(
        RECURRENCE_DIRECT_PLAN_MEMBER,
        PacbinMemberKind::RecurrenceDirectPlan,
        &payload,
    )?;
    let index = write_pacbin_atomic(destination, [member], PacbinWriteOptions::default())?;
    let indexed = index
        .members()
        .first()
        .ok_or_else(|| RusticolError::artifact("direct recurrence PACBIN has no plan member"))?;
    Ok(RecurrenceDirectPacbinMetadata {
        container_size: index.file_size(),
        member_count: index.members().len() as u64,
        unpacked_size_bytes: index.members().iter().map(|member| member.length()).sum(),
        index_sha256: *index.index_sha256(),
        plan_payload_size: indexed.length(),
        plan_sha256: *indexed.sha256(),
    })
}

pub fn load_recurrence_direct_plan_pacbin(
    source: impl AsRef<Path>,
) -> RusticolResult<DirectRecurrencePlan> {
    let reader = PacbinReader::open(source)?;
    if reader.members().len() != 1 {
        return Err(RusticolError::compatibility(format!(
            "direct recurrence PACBIN must contain exactly one member, found {}",
            reader.members().len()
        )));
    }
    let member = reader.member(RECURRENCE_DIRECT_PLAN_MEMBER).map_err(|_| {
        RusticolError::compatibility(
            "unsupported recurrence payload; regenerate with direct-plan v2",
        )
    })?;
    if member.kind() != PacbinMemberKind::RecurrenceDirectPlan {
        return Err(RusticolError::compatibility(
            "direct recurrence plan has the wrong PACBIN member kind",
        ));
    }
    decode_recurrence_direct_plan_v2(reader.member_bytes(RECURRENCE_DIRECT_PLAN_MEMBER)?)
}

#[cfg(test)]
#[path = "direct_pacbin_tests.rs"]
mod tests;
