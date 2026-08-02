// SPDX-License-Identifier: 0BSD

//! PACBIN publication and authenticated loading for direct-plan v2.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

#[cfg(test)]
use super::direct_codec::encode_recurrence_direct_plan_v2;
use super::direct_codec::{
    decode_recurrence_direct_plan_v2, encode_recurrence_direct_plan_v2_to_writer,
};
use super::direct_plan::DirectRecurrencePlan;
use crate::pacbin::{
    PACBIN_DEFAULT_CHUNK_SIZE, PacbinMemberKind, PacbinReader, PacbinWriteMember,
    PacbinWriteOptions, create_temporary_file, write_pacbin_atomic,
};
use crate::{RusticolError, RusticolResult};
use sha2::{Digest, Sha256};

pub const RECURRENCE_DIRECT_SCHEDULE_MEMBER: &str = "schedule/recurrence-direct-schedule-v2.bin";
pub const RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER: &str =
    "proof/recurrence-color-projection-v1.bin";
const COLOR_PROJECTION_CERTIFICATE_BODY_MAGIC: &[u8] = b"PYAMP-COLOR-PROJECTION-BODY-V1\0\0";
const COLOR_PROJECTION_CERTIFICATE_MAGIC: &[u8] = b"PYAMP-COLOR-PROJECTION-CERT-V1\0";

struct TemporaryPlanPayload(PathBuf);

impl Drop for TemporaryPlanPayload {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecurrenceDirectPacbinMetadata {
    pub container_size: u64,
    pub member_count: u64,
    pub unpacked_size_bytes: u64,
    pub index_sha256: [u8; 32],
    pub plan_payload_size: u64,
    pub plan_sha256: [u8; 32],
    pub projection_certificate_payload_size: Option<u64>,
    pub projection_certificate_sha256: Option<[u8; 32]>,
}

pub fn write_recurrence_direct_plan_pacbin(
    destination: impl AsRef<Path>,
    plan: &DirectRecurrencePlan,
) -> RusticolResult<RecurrenceDirectPacbinMetadata> {
    write_recurrence_direct_plan_pacbin_with_projection_certificate(destination, plan, None)
}

pub fn write_recurrence_direct_plan_pacbin_with_projection_certificate(
    destination: impl AsRef<Path>,
    plan: &DirectRecurrencePlan,
    projection_certificate: Option<&[u8]>,
) -> RusticolResult<RecurrenceDirectPacbinMetadata> {
    let destination = destination.as_ref();
    let parent = destination
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let (payload_path, payload_file) = create_temporary_file(destination, parent)?;
    let payload_guard = TemporaryPlanPayload(payload_path);
    let mut payload_writer = BufWriter::with_capacity(PACBIN_DEFAULT_CHUNK_SIZE, payload_file);
    let encoded_payload_size =
        encode_recurrence_direct_plan_v2_to_writer(plan, &mut payload_writer)?;
    payload_writer.flush().map_err(|error| {
        RusticolError::artifact(format!(
            "could not flush recurrence direct-plan temporary payload {}: {error}",
            payload_guard.0.display()
        ))
    })?;
    drop(payload_writer);
    let plan_member = PacbinWriteMember::from_path(
        RECURRENCE_DIRECT_SCHEDULE_MEMBER,
        PacbinMemberKind::RecurrenceDirectPlan,
        &payload_guard.0,
    )?;
    let mut members = vec![plan_member];
    if let Some(certificate) = projection_certificate {
        validate_recurrence_color_projection_certificate(certificate)?;
        members.push(PacbinWriteMember::from_bytes(
            RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER,
            PacbinMemberKind::RecurrenceColorProjectionCertificate,
            certificate,
        )?);
    }
    let index = write_pacbin_atomic(destination, members, PacbinWriteOptions::default())?;
    let indexed = index
        .members()
        .iter()
        .find(|member| member.logical_path() == RECURRENCE_DIRECT_SCHEDULE_MEMBER)
        .ok_or_else(|| RusticolError::artifact("direct recurrence PACBIN has no plan member"))?;
    let projection_indexed = index
        .members()
        .iter()
        .find(|member| member.logical_path() == RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER);
    if indexed.length() != encoded_payload_size {
        return Err(RusticolError::artifact(format!(
            "direct recurrence PACBIN plan length differs from the streamed payload: encoded={encoded_payload_size}, indexed={}",
            indexed.length()
        )));
    }
    Ok(RecurrenceDirectPacbinMetadata {
        container_size: index.file_size(),
        member_count: index.members().len() as u64,
        unpacked_size_bytes: index.members().iter().map(|member| member.length()).sum(),
        index_sha256: *index.index_sha256(),
        plan_payload_size: indexed.length(),
        plan_sha256: *indexed.sha256(),
        projection_certificate_payload_size: projection_indexed.map(|member| member.length()),
        projection_certificate_sha256: projection_indexed.map(|member| *member.sha256()),
    })
}

fn validated_hex_identity<'a>(
    value: &'a str,
    expected_len: usize,
    label: &str,
) -> RusticolResult<&'a [u8]> {
    if value.len() != expected_len
        || !value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(RusticolError::invalid_argument(format!(
            "{label} must be {expected_len} lowercase hexadecimal characters"
        )));
    }
    Ok(value.as_bytes())
}

fn validate_projection_certificate_body(body: &[u8]) -> RusticolResult<()> {
    let minimum = COLOR_PROJECTION_CERTIFICATE_BODY_MAGIC.len() + 4 + 32;
    if body.len() < minimum
        || !body.starts_with(COLOR_PROJECTION_CERTIFICATE_BODY_MAGIC)
        || body[COLOR_PROJECTION_CERTIFICATE_BODY_MAGIC.len()
            ..COLOR_PROJECTION_CERTIFICATE_BODY_MAGIC.len() + 4]
            != 1_u32.to_le_bytes()
    {
        return Err(RusticolError::integrity(
            "recurrence color-projection certificate body has invalid framing",
        ));
    }
    let digest_start = body.len() - 32;
    let actual: [u8; 32] = Sha256::digest(&body[..digest_start]).into();
    if body[digest_start..] != actual {
        return Err(RusticolError::integrity(
            "recurrence color-projection certificate body digest mismatch",
        ));
    }
    Ok(())
}

pub fn bind_recurrence_color_projection_certificate(
    body: &[u8],
    source_revision: &str,
    native_build_inputs_sha256: &str,
) -> RusticolResult<Vec<u8>> {
    validate_projection_certificate_body(body)?;
    let source_revision = validated_hex_identity(
        source_revision,
        40,
        "projection certificate source revision",
    )?;
    let native_build_inputs = validated_hex_identity(
        native_build_inputs_sha256,
        64,
        "projection certificate native-build-input digest",
    )?;
    let mut payload = Vec::new();
    payload.extend_from_slice(COLOR_PROJECTION_CERTIFICATE_MAGIC);
    payload.extend_from_slice(&1_u32.to_le_bytes());
    payload.extend_from_slice(
        &u32::try_from(source_revision.len())
            .map_err(|_| {
                RusticolError::invalid_argument(
                    "projection certificate source revision length exceeds u32",
                )
            })?
            .to_le_bytes(),
    );
    payload.extend_from_slice(source_revision);
    payload.extend_from_slice(
        &u32::try_from(native_build_inputs.len())
            .map_err(|_| {
                RusticolError::invalid_argument(
                    "projection certificate native digest length exceeds u32",
                )
            })?
            .to_le_bytes(),
    );
    payload.extend_from_slice(native_build_inputs);
    payload.extend_from_slice(
        &u64::try_from(body.len())
            .map_err(|_| {
                RusticolError::invalid_argument(
                    "projection certificate structural body exceeds u64",
                )
            })?
            .to_le_bytes(),
    );
    payload.extend_from_slice(body);
    let digest: [u8; 32] = Sha256::digest(&payload).into();
    payload.extend_from_slice(&digest);
    Ok(payload)
}

pub(crate) fn validate_recurrence_color_projection_certificate(
    payload: &[u8],
) -> RusticolResult<()> {
    let fixed_prefix = COLOR_PROJECTION_CERTIFICATE_MAGIC.len() + 4;
    if payload.len() < fixed_prefix + 4 + 40 + 4 + 64 + 8 + 32
        || !payload.starts_with(COLOR_PROJECTION_CERTIFICATE_MAGIC)
        || payload[COLOR_PROJECTION_CERTIFICATE_MAGIC.len()..fixed_prefix] != 1_u32.to_le_bytes()
    {
        return Err(RusticolError::integrity(
            "recurrence color-projection certificate has invalid framing",
        ));
    }
    let mut offset = fixed_prefix;
    let take_u32 = |payload: &[u8], offset: &mut usize| -> RusticolResult<usize> {
        let end = offset
            .checked_add(4)
            .ok_or_else(|| RusticolError::integrity("projection certificate offset overflow"))?;
        let bytes: [u8; 4] = payload
            .get(*offset..end)
            .ok_or_else(|| RusticolError::integrity("projection certificate is truncated"))?
            .try_into()
            .map_err(|_| RusticolError::integrity("projection certificate u32 is malformed"))?;
        *offset = end;
        Ok(u32::from_le_bytes(bytes) as usize)
    };
    let revision_len = take_u32(payload, &mut offset)?;
    let revision_end = offset
        .checked_add(revision_len)
        .ok_or_else(|| RusticolError::integrity("projection certificate revision overflows"))?;
    let revision = payload
        .get(offset..revision_end)
        .ok_or_else(|| RusticolError::integrity("projection certificate revision is truncated"))?;
    let revision = std::str::from_utf8(revision)
        .map_err(|_| RusticolError::integrity("projection certificate revision is not UTF-8"))?;
    validated_hex_identity(revision, 40, "projection certificate source revision")?;
    offset = revision_end;
    let native_len = take_u32(payload, &mut offset)?;
    let native_end = offset.checked_add(native_len).ok_or_else(|| {
        RusticolError::integrity("projection certificate native digest overflows")
    })?;
    let native = payload.get(offset..native_end).ok_or_else(|| {
        RusticolError::integrity("projection certificate native digest is truncated")
    })?;
    let native = std::str::from_utf8(native).map_err(|_| {
        RusticolError::integrity("projection certificate native digest is not UTF-8")
    })?;
    validated_hex_identity(
        native,
        64,
        "projection certificate native-build-input digest",
    )?;
    offset = native_end;
    let body_len_end = offset
        .checked_add(8)
        .ok_or_else(|| RusticolError::integrity("projection certificate body length overflows"))?;
    let body_len_bytes: [u8; 8] = payload
        .get(offset..body_len_end)
        .ok_or_else(|| RusticolError::integrity("projection certificate body length is absent"))?
        .try_into()
        .map_err(|_| RusticolError::integrity("projection certificate body length is malformed"))?;
    offset = body_len_end;
    let body_len = usize::try_from(u64::from_le_bytes(body_len_bytes))
        .map_err(|_| RusticolError::integrity("projection certificate body exceeds usize"))?;
    let body_end = offset
        .checked_add(body_len)
        .ok_or_else(|| RusticolError::integrity("projection certificate body overflows"))?;
    let body = payload
        .get(offset..body_end)
        .ok_or_else(|| RusticolError::integrity("projection certificate body is truncated"))?;
    validate_projection_certificate_body(body)?;
    if body_end.checked_add(32) != Some(payload.len()) {
        return Err(RusticolError::integrity(
            "projection certificate has trailing or missing bytes",
        ));
    }
    let actual: [u8; 32] = Sha256::digest(&payload[..body_end]).into();
    if payload[body_end..] != actual {
        return Err(RusticolError::integrity(
            "recurrence color-projection certificate digest mismatch",
        ));
    }
    Ok(())
}

pub fn load_recurrence_direct_plan_pacbin(
    source: impl AsRef<Path>,
) -> RusticolResult<DirectRecurrencePlan> {
    let reader = PacbinReader::open(source)?;
    if !matches!(reader.members().len(), 1 | 2) {
        return Err(RusticolError::compatibility(format!(
            "direct recurrence PACBIN must contain one plan and at most one projection certificate, found {} members",
            reader.members().len()
        )));
    }
    let member = reader
        .member(RECURRENCE_DIRECT_SCHEDULE_MEMBER)
        .map_err(|_| {
            RusticolError::compatibility(
                "unsupported recurrence payload; regenerate with direct-plan v2",
            )
        })?;
    if member.kind() != PacbinMemberKind::RecurrenceDirectPlan {
        return Err(RusticolError::compatibility(
            "direct recurrence plan has the wrong PACBIN member kind",
        ));
    }
    if reader.members().len() == 2 {
        let certificate = reader
            .member(RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER)
            .map_err(|_| {
                RusticolError::compatibility(
                    "direct recurrence PACBIN has an unsupported second member",
                )
            })?;
        if certificate.kind() != PacbinMemberKind::RecurrenceColorProjectionCertificate {
            return Err(RusticolError::compatibility(
                "recurrence color-projection certificate has the wrong PACBIN member kind",
            ));
        }
        validate_recurrence_color_projection_certificate(
            reader.member_bytes(RECURRENCE_COLOR_PROJECTION_CERTIFICATE_MEMBER)?,
        )?;
    }
    decode_recurrence_direct_plan_v2(reader.member_bytes(RECURRENCE_DIRECT_SCHEDULE_MEMBER)?)
}

#[cfg(test)]
#[path = "direct_pacbin_tests.rs"]
mod tests;
