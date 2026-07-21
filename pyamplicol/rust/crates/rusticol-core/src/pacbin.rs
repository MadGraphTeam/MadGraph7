// SPDX-License-Identifier: 0BSD

//! Strict reader for portable pyAmpliCol evaluator containers.
//!
//! `pacbin-v1` stores fixed-size metadata, aligned uncompressed member
//! payloads, a sorted variable-width index, and a footer. The index and every
//! payload are authenticated with SHA-256. This module is intentionally
//! independent of artifact loading: it owns one container as immutable bytes
//! and provides indexed borrowed slices without extracting files.

use crate::{RusticolError, RusticolResult};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs::File;
#[cfg(not(unix))]
use std::io::Read;
use std::ops::Deref;
#[cfg(unix)]
use std::os::fd::AsRawFd;
use std::path::Path;
#[cfg(unix)]
use std::ptr::NonNull;

pub const PACBIN_VERSION: u16 = 1;
pub const PACBIN_ALIGNMENT: u32 = 64;
pub const PACBIN_MAX_MEMBERS: u64 = 1_000_000;
pub const PACBIN_MAX_PATH_BYTES: u32 = 4096;
pub const PACBIN_MAX_INDEX_BYTES: u64 = 256 * 1024 * 1024;

const HEADER_MAGIC: &[u8; 8] = b"PACBIN\0\0";
const INDEX_MAGIC: &[u8; 8] = b"PACIDX\0\0";
const FOOTER_MAGIC: &[u8; 8] = b"PACEND\0\0";
const SUPPORTED_FLAGS: u32 = 0;
const HEADER_SIZE: usize = 64;
const INDEX_HEADER_SIZE: usize = 32;
const INDEX_ENTRY_SIZE: usize = 56;
const FOOTER_SIZE: usize = 64;
const INDEX_ALIGNMENT: u64 = 8;

/// Portable payload kinds defined by `pacbin-v1`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u16)]
pub enum PacbinMemberKind {
    SymjitApplication = 1,
    SymbolicaExactState = 2,
    NativeLibrary = 3,
}

impl PacbinMemberKind {
    fn parse(value: u16) -> RusticolResult<Self> {
        match value {
            1 => Ok(Self::SymjitApplication),
            2 => Ok(Self::SymbolicaExactState),
            3 => Ok(Self::NativeLibrary),
            _ => Err(compatibility(format!(
                "unknown pacbin member kind: {value}"
            ))),
        }
    }
}

/// Validated metadata for one member.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PacbinMember {
    logical_path: String,
    kind: PacbinMemberKind,
    offset: u64,
    length: u64,
    sha256: [u8; 32],
}

impl PacbinMember {
    pub fn logical_path(&self) -> &str {
        &self.logical_path
    }

    pub fn kind(&self) -> PacbinMemberKind {
        self.kind
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn length(&self) -> u64 {
        self.length
    }

    pub fn sha256(&self) -> &[u8; 32] {
        &self.sha256
    }
}

/// Validated top-level container metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PacbinIndex {
    version: u16,
    index_offset: u64,
    index_sha256: [u8; 32],
    file_size: u64,
    members: Vec<PacbinMember>,
}

impl PacbinIndex {
    pub fn version(&self) -> u16 {
        self.version
    }

    pub fn index_offset(&self) -> u64 {
        self.index_offset
    }

    pub fn index_sha256(&self) -> &[u8; 32] {
        &self.index_sha256
    }

    pub fn file_size(&self) -> u64 {
        self.file_size
    }

    pub fn members(&self) -> &[PacbinMember] {
        &self.members
    }
}

/// An authenticated, indexed `pacbin-v1` container.
///
/// Files are mapped read-only on Unix so large evaluator packs remain outside
/// the Rust heap. Containers constructed from bytes retain owned storage.
#[derive(Debug)]
pub struct PacbinReader {
    bytes: PacbinStorage,
    index: PacbinIndex,
}

#[derive(Debug)]
enum PacbinStorage {
    Owned(Box<[u8]>),
    #[cfg(unix)]
    Mapped(ReadOnlyMmap),
}

impl AsRef<[u8]> for PacbinStorage {
    fn as_ref(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            #[cfg(unix)]
            Self::Mapped(bytes) => bytes,
        }
    }
}

#[cfg(unix)]
struct ReadOnlyMmap {
    pointer: NonNull<u8>,
    length: usize,
}

#[cfg(unix)]
impl std::fmt::Debug for ReadOnlyMmap {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReadOnlyMmap")
            .field("length", &self.length)
            .finish_non_exhaustive()
    }
}

#[cfg(unix)]
impl ReadOnlyMmap {
    fn map(file: &File, path: &Path, length: usize) -> RusticolResult<Self> {
        if length == 0 {
            return Err(RusticolError::artifact(format!(
                "cannot map empty pacbin container {}",
                path.display()
            )));
        }
        // SAFETY: the file descriptor remains valid for the duration of mmap,
        // the non-zero length was obtained from file metadata, and the mapping
        // is private/read-only. The mapping itself owns no borrowed File state.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(RusticolError::artifact(format!(
                "could not memory-map pacbin container {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            )));
        }
        let pointer = NonNull::new(pointer.cast::<u8>()).ok_or_else(|| {
            RusticolError::artifact(format!(
                "memory-mapping pacbin container {} returned a null pointer",
                path.display()
            ))
        })?;
        Ok(Self { pointer, length })
    }
}

#[cfg(unix)]
impl Deref for ReadOnlyMmap {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        // SAFETY: mmap established a read-only mapping of exactly `length`
        // bytes and this object keeps it alive until Drop.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }
}

#[cfg(unix)]
unsafe impl Send for ReadOnlyMmap {}
#[cfg(unix)]
unsafe impl Sync for ReadOnlyMmap {}

#[cfg(unix)]
impl Drop for ReadOnlyMmap {
    fn drop(&mut self) {
        // SAFETY: this pointer/length pair came from the successful mmap in
        // `map` and is unmapped exactly once here.
        unsafe {
            libc::munmap(self.pointer.as_ptr().cast(), self.length);
        }
    }
}

impl PacbinReader {
    /// Open and fully authenticate one container file.
    pub fn open(path: impl AsRef<Path>) -> RusticolResult<Self> {
        Self::open_with_payload_verification(path, true)
    }

    /// Open a container whose enclosing artifact payload digest was already
    /// authenticated. Structural and index validation remain mandatory, while
    /// duplicate per-member hashing is skipped.
    pub(crate) fn open_trusted(path: impl AsRef<Path>) -> RusticolResult<Self> {
        Self::open_with_payload_verification(path, false)
    }

    fn open_with_payload_verification(
        path: impl AsRef<Path>,
        verify_payloads: bool,
    ) -> RusticolResult<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|error| {
            RusticolError::artifact(format!(
                "could not open pacbin container {}: {error}",
                path.display()
            ))
        })?;
        let expected_length = file
            .metadata()
            .map_err(|error| {
                RusticolError::artifact(format!(
                    "could not inspect pacbin container {}: {error}",
                    path.display()
                ))
            })?
            .len();
        let capacity = usize::try_from(expected_length).map_err(|_| {
            RusticolError::artifact(format!(
                "pacbin container {} is too large for this platform",
                path.display()
            ))
        })?;
        #[cfg(unix)]
        let storage = PacbinStorage::Mapped(ReadOnlyMmap::map(&file, path, capacity)?);
        #[cfg(not(unix))]
        let storage = {
            let read_limit = expected_length.checked_add(1).ok_or_else(|| {
                RusticolError::artifact(format!(
                    "pacbin container {} exceeds the supported file size",
                    path.display()
                ))
            })?;
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(capacity).map_err(|error| {
                RusticolError::artifact(format!(
                    "could not allocate {capacity} bytes for pacbin container {}: {error}",
                    path.display()
                ))
            })?;
            file.take(read_limit)
                .read_to_end(&mut bytes)
                .map_err(|error| {
                    RusticolError::artifact(format!(
                        "could not read pacbin container {}: {error}",
                        path.display()
                    ))
                })?;
            if bytes.len() != capacity {
                return Err(RusticolError::artifact(format!(
                    "pacbin container {} changed while being read: expected {capacity} bytes, read {}",
                    path.display(),
                    bytes.len()
                )));
            }
            PacbinStorage::Owned(bytes.into_boxed_slice())
        };
        Self::from_storage(storage, verify_payloads)
    }

    /// Own and fully authenticate already-read container bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> RusticolResult<Self> {
        Self::from_storage(PacbinStorage::Owned(bytes.into_boxed_slice()), true)
    }

    fn from_storage(bytes: PacbinStorage, verify_payloads: bool) -> RusticolResult<Self> {
        let index = parse_and_validate(bytes.as_ref(), verify_payloads)?;
        Ok(Self { bytes, index })
    }

    pub fn index(&self) -> &PacbinIndex {
        &self.index
    }

    pub fn members(&self) -> &[PacbinMember] {
        self.index.members()
    }

    /// Resolve a member by normalized logical path in logarithmic time.
    pub fn member(&self, logical_path: &str) -> RusticolResult<&PacbinMember> {
        let normalized = normalize_logical_path(logical_path)?;
        self.index
            .members
            .binary_search_by(|member| member.logical_path.as_bytes().cmp(normalized.as_bytes()))
            .map(|index| &self.index.members[index])
            .map_err(|_| {
                RusticolError::invalid_argument(format!("unknown pacbin member: {normalized}"))
            })
    }

    /// Borrow a complete authenticated member without unpacking it.
    pub fn member_bytes(&self, logical_path: &str) -> RusticolResult<&[u8]> {
        let member = self.member(logical_path)?;
        member_slice(self.bytes.as_ref(), member.offset, member.length)
    }

    /// Borrow a bounded range relative to one authenticated member.
    pub fn member_range(
        &self,
        logical_path: &str,
        offset: u64,
        length: u64,
    ) -> RusticolResult<&[u8]> {
        let member = self.member(logical_path)?;
        if offset > member.length || length > member.length - offset {
            return Err(RusticolError::invalid_argument(
                "member read exceeds indexed payload bounds",
            ));
        }
        let absolute_offset = member
            .offset
            .checked_add(offset)
            .ok_or_else(|| RusticolError::invalid_argument("member read offset exceeds u64"))?;
        member_slice(self.bytes.as_ref(), absolute_offset, length)
    }

    /// Re-authenticate all member payloads held by this reader.
    pub fn verify_payloads(&self) -> RusticolResult<()> {
        verify_payload_digests(self.bytes.as_ref(), &self.index.members)
    }

    /// Return the total owned container size.
    pub fn container_size(&self) -> usize {
        self.bytes.as_ref().len()
    }
}

/// Normalize a portable ASCII member path using the Python codec's POSIX rules.
///
/// Generated evaluator paths are ASCII. Rejecting non-ASCII input is
/// intentional: without weakening Python's NFC and full Unicode case-folding
/// collision guarantees, the runtime accepts the complete generated-path
/// domain and fails closed for paths outside it.
pub fn normalize_logical_path(value: &str) -> RusticolResult<String> {
    normalize_logical_path_impl(value).map_err(RusticolError::invalid_argument)
}

fn normalize_logical_path_impl(value: &str) -> Result<String, String> {
    if value.is_empty() || value.contains('\0') {
        return Err("pacbin logical path must be non-empty and contain no NUL".to_string());
    }
    if !value.is_ascii() {
        return Err(
            "pacbin logical path must use the portable ASCII evaluator-path subset".to_string(),
        );
    }
    if value.contains('\\') {
        return Err("pacbin logical path must use POSIX '/' separators".to_string());
    }
    if value.starts_with('/') {
        return Err("pacbin logical path must be relative".to_string());
    }

    let mut parts = Vec::new();
    for part in value.split('/') {
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." {
            return Err("pacbin logical path must not contain '..'".to_string());
        }
        parts.push(part);
    }
    if parts.is_empty() {
        return Err("pacbin logical path must name a member".to_string());
    }
    Ok(parts.join("/"))
}

fn parse_and_validate(bytes: &[u8], verify_payloads: bool) -> RusticolResult<PacbinIndex> {
    let minimum_size = HEADER_SIZE
        .checked_add(INDEX_HEADER_SIZE)
        .and_then(|value| value.checked_add(FOOTER_SIZE))
        .expect("pacbin fixed structure sizes fit usize");
    if bytes.len() < minimum_size {
        return Err(integrity("truncated pacbin container"));
    }
    let file_size =
        u64::try_from(bytes.len()).map_err(|_| integrity("pacbin file size exceeds u64"))?;

    let header = checked_slice(bytes, 0, HEADER_SIZE, "pacbin header")?;
    if &header[0..8] != HEADER_MAGIC {
        return Err(integrity("invalid pacbin header magic"));
    }
    let version = u16_at(header, 8, "pacbin header version")?;
    let header_size = u16_at(header, 10, "pacbin header size")?;
    let header_flags = u32_at(header, 12, "pacbin header flags")?;
    validate_contract("header", version, header_size, HEADER_SIZE, header_flags)?;
    let alignment = u32_at(header, 16, "pacbin payload alignment")?;
    if alignment != PACBIN_ALIGNMENT {
        return Err(compatibility(format!(
            "unsupported pacbin payload alignment: {alignment}"
        )));
    }
    if u32_at(header, 20, "pacbin header reserved field")? != 0
        || header[40..64].iter().any(|value| *value != 0)
    {
        return Err(integrity("pacbin header reserved fields must be zero"));
    }
    let index_offset = u64_at(header, 24, "pacbin index offset")?;
    let member_count = u64_at(header, 32, "pacbin member count")?;
    if index_offset % u64::from(PACBIN_ALIGNMENT) != 0 {
        return Err(integrity("pacbin index offset is not payload-aligned"));
    }

    let footer_offset = bytes
        .len()
        .checked_sub(FOOTER_SIZE)
        .ok_or_else(|| integrity("truncated pacbin footer"))?;
    let footer_offset_u64 =
        u64::try_from(footer_offset).map_err(|_| integrity("pacbin footer offset exceeds u64"))?;
    if index_offset < HEADER_SIZE as u64 || index_offset >= footer_offset_u64 {
        return Err(integrity("pacbin index offset is out of bounds"));
    }

    let footer = checked_slice(bytes, footer_offset, FOOTER_SIZE, "pacbin footer")?;
    if &footer[0..8] != FOOTER_MAGIC {
        return Err(integrity("invalid pacbin footer magic"));
    }
    let footer_version = u16_at(footer, 8, "pacbin footer version")?;
    let footer_size = u16_at(footer, 10, "pacbin footer size")?;
    let footer_flags = u32_at(footer, 12, "pacbin footer flags")?;
    validate_contract(
        "footer",
        footer_version,
        footer_size,
        FOOTER_SIZE,
        footer_flags,
    )?;
    if u64_at(footer, 16, "pacbin footer index offset")? != index_offset {
        return Err(integrity(
            "pacbin footer index offset disagrees with header",
        ));
    }
    if u64_at(footer, 24, "pacbin footer member count")? != member_count {
        return Err(integrity(
            "pacbin footer member count disagrees with header",
        ));
    }
    if member_count > PACBIN_MAX_MEMBERS {
        return Err(integrity(format!(
            "pacbin member count exceeds limit: {member_count}"
        )));
    }
    let expected_index_digest: [u8; 32] = footer[32..64]
        .try_into()
        .map_err(|_| integrity("truncated pacbin footer digest"))?;

    let index_offset_usize = usize::try_from(index_offset)
        .map_err(|_| integrity("pacbin index offset exceeds platform bounds"))?;
    let available_index_bytes = footer_offset_u64 - index_offset;
    validate_index_bounds(member_count, available_index_bytes)?;
    let index_bytes = checked_slice(
        bytes,
        index_offset_usize,
        usize::try_from(available_index_bytes)
            .map_err(|_| integrity("pacbin index size exceeds platform bounds"))?,
        "pacbin index",
    )?;
    let actual_index_digest: [u8; 32] = Sha256::digest(index_bytes).into();

    let index_header = checked_slice(index_bytes, 0, INDEX_HEADER_SIZE, "pacbin index header")?;
    if &index_header[0..8] != INDEX_MAGIC {
        return Err(integrity("invalid pacbin index magic"));
    }
    let index_version = u16_at(index_header, 8, "pacbin index version")?;
    let index_header_size = u16_at(index_header, 10, "pacbin index size")?;
    let index_flags = u32_at(index_header, 12, "pacbin index flags")?;
    validate_contract(
        "index",
        index_version,
        index_header_size,
        INDEX_HEADER_SIZE,
        index_flags,
    )?;
    if u64_at(index_header, 24, "pacbin index reserved field")? != 0 {
        return Err(integrity("pacbin index reserved field must be zero"));
    }
    if u64_at(index_header, 16, "pacbin index member count")? != member_count {
        return Err(integrity("pacbin index member count disagrees with header"));
    }

    let member_capacity = usize::try_from(member_count)
        .map_err(|_| integrity("pacbin member count exceeds platform bounds"))?;
    let mut members = Vec::new();
    members
        .try_reserve_exact(member_capacity)
        .map_err(|error| {
            integrity(format!(
                "could not allocate pacbin index for {member_count} members: {error}"
            ))
        })?;
    let mut seen_paths = BTreeSet::new();
    let mut seen_folded_paths = BTreeSet::new();
    let mut previous_path_bytes: Option<Vec<u8>> = None;
    let mut cursor = INDEX_HEADER_SIZE;
    for _ in 0..member_count {
        let entry = checked_slice(index_bytes, cursor, INDEX_ENTRY_SIZE, "pacbin index entry")?;
        let path_length = u32_at(entry, 0, "pacbin member path length")?;
        if path_length > PACBIN_MAX_PATH_BYTES {
            return Err(integrity(format!(
                "pacbin member path exceeds size limit: {path_length} bytes"
            )));
        }
        let kind = PacbinMemberKind::parse(u16_at(entry, 4, "pacbin member kind")?)?;
        let entry_flags = u16_at(entry, 6, "pacbin member flags")?;
        if entry_flags != SUPPORTED_FLAGS as u16 {
            return Err(compatibility(format!(
                "unknown pacbin member flags: {entry_flags}"
            )));
        }
        let offset = u64_at(entry, 8, "pacbin member offset")?;
        let length = u64_at(entry, 16, "pacbin member length")?;
        let sha256: [u8; 32] = entry[24..56]
            .try_into()
            .map_err(|_| integrity("truncated pacbin member digest"))?;
        cursor = cursor
            .checked_add(INDEX_ENTRY_SIZE)
            .ok_or_else(|| integrity("pacbin index cursor overflow"))?;
        let path_length_usize = usize::try_from(path_length)
            .map_err(|_| integrity("pacbin member path length exceeds platform bounds"))?;
        let path_bytes =
            checked_slice(index_bytes, cursor, path_length_usize, "pacbin member path")?;
        let logical_path = std::str::from_utf8(path_bytes)
            .map_err(|_| integrity("pacbin member path is not valid UTF-8"))?;
        let normalized = normalize_logical_path_impl(logical_path).map_err(integrity)?;
        if normalized != logical_path {
            return Err(integrity(format!(
                "pacbin member path is not canonical: {logical_path:?}"
            )));
        }
        let folded = logical_path.to_ascii_lowercase();
        if !seen_paths.insert(logical_path.to_string()) {
            return Err(integrity(format!(
                "duplicate pacbin member path: {logical_path}"
            )));
        }
        if !seen_folded_paths.insert(folded) {
            return Err(integrity(format!(
                "case-colliding pacbin member path: {logical_path}"
            )));
        }
        if previous_path_bytes
            .as_deref()
            .is_some_and(|previous| path_bytes <= previous)
        {
            return Err(integrity("pacbin index paths are not strictly sorted"));
        }
        previous_path_bytes = Some(path_bytes.to_vec());
        cursor = cursor
            .checked_add(path_length_usize)
            .ok_or_else(|| integrity("pacbin index cursor overflow"))?;
        let record_length = (INDEX_ENTRY_SIZE as u64)
            .checked_add(u64::from(path_length))
            .ok_or_else(|| integrity("pacbin index record length exceeds u64"))?;
        let padding = padding_length(record_length, INDEX_ALIGNMENT);
        let padding_usize = usize::try_from(padding)
            .map_err(|_| integrity("pacbin index padding exceeds platform bounds"))?;
        let padding_bytes =
            checked_slice(index_bytes, cursor, padding_usize, "pacbin index padding")?;
        if padding_bytes.iter().any(|value| *value != 0) {
            return Err(integrity("pacbin index padding must be zero"));
        }
        cursor = cursor
            .checked_add(padding_usize)
            .ok_or_else(|| integrity("pacbin index cursor overflow"))?;
        members.push(PacbinMember {
            logical_path: logical_path.to_string(),
            kind,
            offset,
            length,
            sha256,
        });
    }
    if cursor != index_bytes.len() {
        return Err(integrity("pacbin index has trailing or missing bytes"));
    }
    if actual_index_digest != expected_index_digest {
        return Err(integrity("pacbin index digest mismatch"));
    }

    validate_canonical_payload_layout(bytes, &members, index_offset)?;
    if verify_payloads {
        verify_payload_digests(bytes, &members)?;
    }
    Ok(PacbinIndex {
        version,
        index_offset,
        index_sha256: actual_index_digest,
        file_size,
        members,
    })
}

fn validate_contract(
    label: &str,
    version: u16,
    encoded_size: u16,
    expected_size: usize,
    flags: u32,
) -> RusticolResult<()> {
    if version != PACBIN_VERSION {
        return Err(compatibility(format!(
            "unsupported pacbin {label} version: {version}"
        )));
    }
    if usize::from(encoded_size) != expected_size {
        return Err(compatibility(format!(
            "unsupported pacbin {label} size: {encoded_size}"
        )));
    }
    if flags != SUPPORTED_FLAGS {
        return Err(compatibility(format!(
            "unknown pacbin {label} flags: {flags}"
        )));
    }
    Ok(())
}

fn validate_index_bounds(member_count: u64, available_index_bytes: u64) -> RusticolResult<()> {
    if available_index_bytes > PACBIN_MAX_INDEX_BYTES {
        return Err(integrity(format!(
            "pacbin index exceeds size limit: {available_index_bytes} bytes"
        )));
    }
    let minimum_record_size = (INDEX_ENTRY_SIZE as u64)
        .checked_add(INDEX_ALIGNMENT)
        .expect("pacbin minimum record size fits u64");
    if member_count > available_index_bytes / minimum_record_size {
        return Err(integrity("pacbin member count cannot fit in index"));
    }
    Ok(())
}

fn validate_canonical_payload_layout(
    bytes: &[u8],
    members: &[PacbinMember],
    index_offset: u64,
) -> RusticolResult<()> {
    let mut expected_offset = HEADER_SIZE as u64;
    for member in members {
        member.offset.checked_add(member.length).ok_or_else(|| {
            integrity(format!(
                "pacbin member range overflows u64: {}",
                member.logical_path
            ))
        })?;
        let unaligned_offset = expected_offset;
        let padding = padding_length(unaligned_offset, u64::from(PACBIN_ALIGNMENT));
        expected_offset = expected_offset
            .checked_add(padding)
            .ok_or_else(|| integrity("pacbin payload alignment exceeds u64"))?;
        if member.offset < expected_offset {
            return Err(integrity(format!(
                "overlapping pacbin member payload: {}",
                member.logical_path
            )));
        }
        if member.offset > expected_offset {
            return Err(integrity(format!(
                "non-canonical pacbin member gap: {}",
                member.logical_path
            )));
        }
        if member.offset % u64::from(PACBIN_ALIGNMENT) != 0 {
            return Err(integrity(format!(
                "unaligned pacbin member payload: {}",
                member.logical_path
            )));
        }
        let end = member.offset + member.length;
        if end > index_offset {
            return Err(integrity(format!(
                "pacbin member payload is out of bounds: {}",
                member.logical_path
            )));
        }
        validate_zero_region(bytes, unaligned_offset, padding, "pacbin payload padding")?;
        expected_offset = end;
    }

    let trailing_padding = padding_length(expected_offset, u64::from(PACBIN_ALIGNMENT));
    let canonical_index_offset = expected_offset
        .checked_add(trailing_padding)
        .ok_or_else(|| integrity("pacbin payload region exceeds u64"))?;
    if canonical_index_offset != index_offset {
        if canonical_index_offset > index_offset {
            return Err(integrity("pacbin payloads overlap the index"));
        }
        return Err(integrity(
            "pacbin payload region has a non-canonical trailing gap",
        ));
    }
    validate_zero_region(
        bytes,
        expected_offset,
        trailing_padding,
        "pacbin payload padding",
    )?;
    Ok(())
}

fn verify_payload_digests(bytes: &[u8], members: &[PacbinMember]) -> RusticolResult<()> {
    for member in members {
        let payload = member_slice(bytes, member.offset, member.length)?;
        let digest: [u8; 32] = Sha256::digest(payload).into();
        if digest != member.sha256 {
            return Err(integrity(format!(
                "pacbin member digest mismatch: {}",
                member.logical_path
            )));
        }
    }
    Ok(())
}

fn member_slice(bytes: &[u8], offset: u64, length: u64) -> RusticolResult<&[u8]> {
    let start = usize::try_from(offset)
        .map_err(|_| integrity("pacbin member offset exceeds platform bounds"))?;
    let length = usize::try_from(length)
        .map_err(|_| integrity("pacbin member length exceeds platform bounds"))?;
    checked_slice(bytes, start, length, "pacbin member payload")
}

fn validate_zero_region(bytes: &[u8], offset: u64, length: u64, label: &str) -> RusticolResult<()> {
    let start = usize::try_from(offset)
        .map_err(|_| integrity(format!("{label} offset exceeds platform bounds")))?;
    let length = usize::try_from(length)
        .map_err(|_| integrity(format!("{label} length exceeds platform bounds")))?;
    let region = checked_slice(bytes, start, length, label)?;
    if region.iter().any(|value| *value != 0) {
        return Err(integrity(format!("{label} must be zero")));
    }
    Ok(())
}

fn checked_slice<'a>(
    bytes: &'a [u8],
    offset: usize,
    length: usize,
    label: &str,
) -> RusticolResult<&'a [u8]> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| integrity(format!("{label} range overflow")))?;
    bytes
        .get(offset..end)
        .ok_or_else(|| integrity(format!("truncated {label}")))
}

fn u16_at(bytes: &[u8], offset: usize, label: &str) -> RusticolResult<u16> {
    let value: [u8; 2] = checked_slice(bytes, offset, 2, label)?
        .try_into()
        .map_err(|_| integrity(format!("truncated {label}")))?;
    Ok(u16::from_le_bytes(value))
}

fn u32_at(bytes: &[u8], offset: usize, label: &str) -> RusticolResult<u32> {
    let value: [u8; 4] = checked_slice(bytes, offset, 4, label)?
        .try_into()
        .map_err(|_| integrity(format!("truncated {label}")))?;
    Ok(u32::from_le_bytes(value))
}

fn u64_at(bytes: &[u8], offset: usize, label: &str) -> RusticolResult<u64> {
    let value: [u8; 8] = checked_slice(bytes, offset, 8, label)?
        .try_into()
        .map_err(|_| integrity(format!("truncated {label}")))?;
    Ok(u64::from_le_bytes(value))
}

fn padding_length(position: u64, alignment: u64) -> u64 {
    (alignment - position % alignment) % alignment
}

fn integrity(message: impl Into<String>) -> RusticolError {
    RusticolError::integrity(message)
}

fn compatibility(message: impl Into<String>) -> RusticolError {
    RusticolError::compatibility(message)
}

#[cfg(test)]
#[path = "pacbin_tests.rs"]
mod tests;
