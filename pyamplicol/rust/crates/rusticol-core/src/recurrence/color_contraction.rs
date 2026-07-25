// SPDX-License-Identifier: 0BSD

//! Process-owned fixed-width recurrence color-contraction payload.

use std::collections::{BTreeMap, BTreeSet};
use std::iter::FusedIterator;

use sha2::{Digest, Sha256};

use super::exact::{ExactComplexRational, ExactRational};
use crate::{RusticolError, RusticolResult};

pub const RECURRENCE_COLOR_CONTRACTION_CODEC_ABI: &str =
    "pyamplicol-recurrence-color-contraction-v3";

const MAGIC: &[u8; 8] = b"PACRCLR3";
const VERSION: u32 = 3;
const HEADER_BYTES: usize = 120;
const ENTRY_BYTES: usize = 36;
const EXACT_FACTOR_BYTES: usize = 64;
const MAX_PAYLOAD_BYTES: usize = 8 * 1024 * 1024 * 1024;
const MAX_FACTOR_RANK: u32 = 16;
const ZERO_SECTOR_OWNER: u32 = u32::MAX;
const FLAG_INCLUDES_COLOR_FACTOR: u32 = 1 << 0;
const KNOWN_FLAGS: u32 = FLAG_INCLUDES_COLOR_FACTOR;

fn malformed(message: impl Into<String>) -> RusticolError {
    RusticolError::artifact(format!(
        "recurrence color-contraction codec: {}",
        message.into()
    ))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum RecurrenceColorAccuracy {
    Nlc = 1,
    Full = 2,
}

impl RecurrenceColorAccuracy {
    fn decode(value: u32) -> RusticolResult<Self> {
        match value {
            1 => Ok(Self::Nlc),
            2 => Ok(Self::Full),
            _ => Err(malformed(format!(
                "unknown color-accuracy discriminant {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum RecurrenceColorStorage {
    Expanded = 1,
    Repeated = 2,
}

impl RecurrenceColorStorage {
    fn decode(value: u32) -> RusticolResult<Self> {
        match value {
            1 => Ok(Self::Expanded),
            2 => Ok(Self::Repeated),
            _ => Err(malformed(format!(
                "unknown color-contraction storage discriminant {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RawColorContractionEntry {
    pub left_group_id: u32,
    pub right_group_id: u32,
    pub weight_re: f64,
    pub weight_im: f64,
    pub symmetry_factor: f64,
    pub exact_factor_id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanonicalColorContractionEntry {
    pub left_group_id: u32,
    pub right_group_id: u32,
    pub left_destination_id: u32,
    pub right_destination_id: u32,
    pub weight_re: f64,
    pub weight_im: f64,
    pub symmetry_factor: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuntimeColorContractionEntry {
    pub left_destination_id: u32,
    pub right_destination_id: u32,
    pub coefficient_re: f64,
    pub coefficient_im: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FactorizedColorContractionKind {
    KleinFourWalsh,
    ElementaryAbelianWalsh,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FactorizedColorContraction {
    kind: FactorizedColorContractionKind,
    rank: u32,
    coset_count: usize,
    coset_indices: Vec<u32>,
}

impl FactorizedColorContraction {
    pub fn kind(&self) -> FactorizedColorContractionKind {
        self.kind
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }

    pub fn subgroup_order(&self) -> usize {
        1usize << self.rank
    }

    pub fn coset_count(&self) -> usize {
        self.coset_count
    }

    pub fn coset(&self, index: usize) -> Option<&[u32]> {
        let order = self.subgroup_order();
        let start = index.checked_mul(order)?;
        self.coset_indices.get(start..start.checked_add(order)?)
    }

    pub fn coset_indices(&self) -> &[u32] {
        &self.coset_indices
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuntimeFactorizedColorContractionEntry {
    pub left_group_index: u32,
    pub right_group_index: u32,
    pub coefficient_re: f64,
    pub coefficient_im: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeFactorizedColorContraction {
    subgroup_order: usize,
    cosets: Vec<Vec<u32>>,
    entries: Vec<RuntimeFactorizedColorContractionEntry>,
    amplitude_scale: f64,
}

impl RuntimeFactorizedColorContraction {
    pub fn subgroup_order(&self) -> usize {
        self.subgroup_order
    }

    pub fn cosets(&self) -> &[Vec<u32>] {
        &self.cosets
    }

    pub fn entries(&self) -> &[RuntimeFactorizedColorContractionEntry] {
        &self.entries
    }

    pub fn amplitude_scale(&self) -> f64 {
        self.amplitude_scale
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RecurrenceColorContraction {
    accuracy: RecurrenceColorAccuracy,
    storage: RecurrenceColorStorage,
    includes_color_factor: bool,
    group_count: u32,
    sector_count: u32,
    component_count: u32,
    local_group_count: u32,
    destination_count: u32,
    entries: Vec<RawColorContractionEntry>,
    exact_factors: Vec<ExactComplexRational>,
    ordered_group_ids: Vec<u32>,
    destination_by_group: Vec<u32>,
    sector_by_group: Vec<u32>,
    component_by_group: Vec<u32>,
    owner_by_sector: Vec<u32>,
    ordered_destination_ids: Vec<u32>,
    factorization: Option<FactorizedColorContraction>,
    runtime_factorization: Option<RuntimeFactorizedColorContraction>,
}

impl RecurrenceColorContraction {
    pub fn accuracy(&self) -> RecurrenceColorAccuracy {
        self.accuracy
    }

    pub fn storage(&self) -> RecurrenceColorStorage {
        self.storage
    }

    pub fn includes_color_factor(&self) -> bool {
        self.includes_color_factor
    }

    pub fn group_count(&self) -> u32 {
        self.group_count
    }

    pub fn sector_count(&self) -> u32 {
        self.sector_count
    }

    pub fn component_count(&self) -> u32 {
        self.component_count
    }

    pub fn local_group_count(&self) -> u32 {
        self.local_group_count
    }

    pub fn destination_count(&self) -> u32 {
        self.destination_count
    }

    pub fn entries(&self) -> &[RawColorContractionEntry] {
        &self.entries
    }

    pub fn exact_factors(&self) -> &[ExactComplexRational] {
        &self.exact_factors
    }

    pub fn ordered_group_ids(&self) -> &[u32] {
        &self.ordered_group_ids
    }

    pub fn destination_by_group(&self) -> &[u32] {
        &self.destination_by_group
    }

    pub fn sector_by_group(&self) -> &[u32] {
        &self.sector_by_group
    }

    pub fn component_by_group(&self) -> &[u32] {
        &self.component_by_group
    }

    pub fn owner_by_sector(&self) -> &[u32] {
        &self.owner_by_sector
    }

    pub fn active_sector_count(&self) -> usize {
        self.owner_by_sector
            .iter()
            .enumerate()
            .filter(|(sector, owner)| *sector == **owner as usize)
            .count()
    }

    pub fn factorization(&self) -> Option<&FactorizedColorContraction> {
        self.factorization.as_ref()
    }

    pub fn runtime_factorization(&self) -> Option<&RuntimeFactorizedColorContraction> {
        self.runtime_factorization.as_ref()
    }

    pub fn ordered_destination_id(
        &self,
        local_group_index: usize,
        component_index: usize,
    ) -> Option<u32> {
        let ordered_index = local_group_index
            .checked_mul(self.component_count as usize)?
            .checked_add(component_index)?;
        self.ordered_destination_ids.get(ordered_index).copied()
    }

    pub fn logical_entry_count(&self) -> usize {
        match self.storage {
            RecurrenceColorStorage::Expanded => self.entries.len(),
            RecurrenceColorStorage::Repeated => self.entries.len() * self.component_count as usize,
        }
    }

    /// Iterate canonical logical entries without allocating.
    pub fn canonical_logical_entries(&self) -> CanonicalColorContractionEntries<'_> {
        CanonicalColorContractionEntries {
            plan: self,
            next_index: 0,
            entry_count: self.logical_entry_count(),
        }
    }

    /// Iterate symmetry-folded Direct-Arena rows without allocating.
    pub fn runtime_entries(&self) -> RuntimeColorContractionEntries<'_> {
        RuntimeColorContractionEntries {
            inner: self.canonical_logical_entries(),
        }
    }
}

pub struct CanonicalColorContractionEntries<'a> {
    plan: &'a RecurrenceColorContraction,
    next_index: usize,
    entry_count: usize,
}

impl Iterator for CanonicalColorContractionEntries<'_> {
    type Item = CanonicalColorContractionEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.entry_count {
            return None;
        }
        let logical_index = self.next_index;
        self.next_index += 1;
        let (entry, left_group_id, right_group_id) = match self.plan.storage {
            RecurrenceColorStorage::Expanded => {
                let entry = *self.plan.entries.get(logical_index)?;
                (entry, entry.left_group_id, entry.right_group_id)
            }
            RecurrenceColorStorage::Repeated => {
                let template_count = self.plan.entries.len();
                if template_count == 0 {
                    return None;
                }
                let component_index = logical_index / template_count;
                let entry = *self.plan.entries.get(logical_index % template_count)?;
                let component_count = self.plan.component_count as usize;
                let left_index = entry.left_group_id as usize * component_count + component_index;
                let right_index = entry.right_group_id as usize * component_count + component_index;
                (
                    entry,
                    *self.plan.ordered_group_ids.get(left_index)?,
                    *self.plan.ordered_group_ids.get(right_index)?,
                )
            }
        };
        Some(CanonicalColorContractionEntry {
            left_group_id,
            right_group_id,
            left_destination_id: self.plan.destination_by_group[left_group_id as usize],
            right_destination_id: self.plan.destination_by_group[right_group_id as usize],
            weight_re: entry.weight_re,
            weight_im: entry.weight_im,
            symmetry_factor: entry.symmetry_factor,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.entry_count.saturating_sub(self.next_index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for CanonicalColorContractionEntries<'_> {}
impl FusedIterator for CanonicalColorContractionEntries<'_> {}

pub struct RuntimeColorContractionEntries<'a> {
    inner: CanonicalColorContractionEntries<'a>,
}

impl Iterator for RuntimeColorContractionEntries<'_> {
    type Item = RuntimeColorContractionEntry;

    fn next(&mut self) -> Option<Self::Item> {
        let raw = self.inner.next()?;
        Some(RuntimeColorContractionEntry {
            left_destination_id: raw.left_destination_id,
            right_destination_id: raw.right_destination_id,
            coefficient_re: raw.weight_re * raw.symmetry_factor,
            coefficient_im: raw.weight_im * raw.symmetry_factor,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl ExactSizeIterator for RuntimeColorContractionEntries<'_> {}
impl FusedIterator for RuntimeColorContractionEntries<'_> {}

/// Decode and fully validate a caller-authenticated process payload.
pub fn decode_recurrence_color_contraction_v3(
    bytes: &[u8],
) -> RusticolResult<RecurrenceColorContraction> {
    if bytes.len() > MAX_PAYLOAD_BYTES.saturating_add(HEADER_BYTES) {
        return Err(malformed("payload exceeds the 8 GiB format limit"));
    }
    let mut reader = Reader::new(bytes);
    if reader.take(8, "magic")? != MAGIC {
        return Err(malformed("invalid payload magic"));
    }
    if reader.u32("version")? != VERSION {
        return Err(malformed("unsupported payload version"));
    }
    if reader.u32("header size")? as usize != HEADER_BYTES {
        return Err(malformed("header size does not match codec v3"));
    }
    let storage = RecurrenceColorStorage::decode(reader.u32("storage")?)?;
    let accuracy = RecurrenceColorAccuracy::decode(reader.u32("color accuracy")?)?;
    let flags = reader.u32("flags")?;
    if flags & !KNOWN_FLAGS != 0 {
        return Err(malformed("payload declares unknown flags"));
    }
    let group_count = reader.u32("group count")?;
    let sector_count = reader.u32("sector count")?;
    let component_count = reader.u32("component count")?;
    let local_group_count = reader.u32("local group count")?;
    let destination_count = reader.u32("destination count")?;
    let factor_kind = reader.u32("factorization kind")?;
    let factor_rank = reader.u32("factorization rank")?;
    if reader.u32("entry stride")? as usize != ENTRY_BYTES {
        return Err(malformed("entry stride does not match codec v3"));
    }
    if reader.u32("exact factor stride")? as usize != EXACT_FACTOR_BYTES {
        return Err(malformed("exact factor stride does not match codec v3"));
    }
    let entry_count = reader.count("entry count")?;
    let exact_factor_count = reader.count("exact factor count")?;
    let coset_count = reader.count("coset count")?;
    let coset_index_count = reader.count("coset index count")?;
    let declared_logical_entry_count = reader.count("logical entry count")?;
    let owner_map_count = reader.count("physical sector owner map count")?;
    let payload_bytes = reader.count("payload byte count")?;

    if group_count == 0 || sector_count == 0 || component_count == 0 || destination_count == 0 {
        return Err(malformed(
            "group, sector, component, and destination counts must be positive",
        ));
    }
    if owner_map_count != sector_count as usize {
        return Err(malformed(
            "physical sector owner map count does not match sector_count",
        ));
    }
    let expected_payload_bytes = entry_count
        .checked_mul(ENTRY_BYTES)
        .and_then(|value| value.checked_add(exact_factor_count.checked_mul(EXACT_FACTOR_BYTES)?))
        .and_then(|value| value.checked_add((group_count as usize).checked_mul(16)?))
        .and_then(|value| value.checked_add(owner_map_count.checked_mul(4)?))
        .and_then(|value| value.checked_add(coset_index_count.checked_mul(4)?))
        .ok_or_else(|| malformed("payload byte count overflows usize"))?;
    if payload_bytes != expected_payload_bytes
        || payload_bytes > MAX_PAYLOAD_BYTES
        || bytes.len() != HEADER_BYTES + payload_bytes
    {
        return Err(malformed(
            "declared payload size does not match its fixed-width sections",
        ));
    }

    let entry_domain = match storage {
        RecurrenceColorStorage::Expanded => group_count,
        RecurrenceColorStorage::Repeated => local_group_count,
    };
    let mut entries = Vec::with_capacity(entry_count);
    let mut seen_entry_pairs = BTreeSet::new();
    for index in 0..entry_count {
        let entry = RawColorContractionEntry {
            left_group_id: reader.u32("entry left group ID")?,
            right_group_id: reader.u32("entry right group ID")?,
            weight_re: reader.f64("entry real weight")?,
            weight_im: reader.f64("entry imaginary weight")?,
            symmetry_factor: reader.f64("entry symmetry factor")?,
            exact_factor_id: reader.u32("entry exact factor ID")?,
        };
        validate_raw_entry(index, entry, entry_domain, &mut seen_entry_pairs)?;
        entries.push(entry);
    }
    let mut exact_factors = Vec::with_capacity(exact_factor_count);
    for index in 0..exact_factor_count {
        let factor = ExactComplexRational::new(
            ExactRational::new(
                reader.i128("exact real numerator")?,
                reader.i128("exact real denominator")?,
            )
            .map_err(|error| malformed(format!("exact color factor {index} real part: {error}")))?,
            ExactRational::new(
                reader.i128("exact imaginary numerator")?,
                reader.i128("exact imaginary denominator")?,
            )
            .map_err(|error| {
                malformed(format!(
                    "exact color factor {index} imaginary part: {error}"
                ))
            })?,
        );
        exact_factors.push(factor);
    }
    if entries
        .iter()
        .any(|entry| entry.exact_factor_id as usize >= exact_factors.len())
    {
        return Err(malformed(
            "entry references an out-of-bounds exact color factor",
        ));
    }
    for (index, entry) in entries.iter().copied().enumerate() {
        validate_exact_matches_f64(index, entry, exact_factors[entry.exact_factor_id as usize])?;
    }
    let ordered_group_ids = reader.u32_vec(group_count as usize, "ordered group ID")?;
    let destination_by_group =
        reader.u32_vec(group_count as usize, "Direct-Arena destination ID")?;
    let sector_by_group = reader.u32_vec(group_count as usize, "group physical-sector ID")?;
    let component_by_group = reader.u32_vec(group_count as usize, "group resolved-helicity ID")?;
    let owner_by_sector = reader.u32_vec(owner_map_count, "physical sector owner ID")?;
    let coset_indices = reader.u32_vec(coset_index_count, "factorization coset index")?;
    if !reader.is_finished() {
        return Err(malformed("payload contains trailing bytes"));
    }

    validate_permutation(&ordered_group_ids, group_count, "ordered group map")?;
    validate_destination_map(&destination_by_group, destination_count)?;
    validate_group_identities(
        &sector_by_group,
        &component_by_group,
        sector_count,
        component_count,
    )?;
    validate_sector_owners(&owner_by_sector, &sector_by_group, sector_count)?;

    let expected_logical_entry_count = match storage {
        RecurrenceColorStorage::Expanded => {
            if local_group_count != 0
                || factor_kind != 0
                || factor_rank != 0
                || coset_count != 0
                || coset_index_count != 0
            {
                return Err(malformed(
                    "expanded storage is mixed with repeated/factorized fields",
                ));
            }
            validate_expanded_components(&entries, &component_by_group)?;
            entry_count
        }
        RecurrenceColorStorage::Repeated => {
            if component_count < 2 {
                return Err(malformed(
                    "repeated storage requires at least two components",
                ));
            }
            if local_group_count == 0
                || local_group_count.checked_mul(component_count) != Some(group_count)
            {
                return Err(malformed(
                    "repeated local, sector, component, and group counts are inconsistent",
                ));
            }
            validate_repeated_group_identities(
                &ordered_group_ids,
                &sector_by_group,
                &component_by_group,
                local_group_count,
                component_count,
            )?;
            entry_count
                .checked_mul(component_count as usize)
                .ok_or_else(|| malformed("logical entry count overflows usize"))?
        }
    };
    if declared_logical_entry_count != expected_logical_entry_count {
        return Err(malformed(
            "declared logical entry count is inconsistent with storage",
        ));
    }

    let factorization = decode_factorization(
        storage,
        factor_kind,
        factor_rank,
        coset_count,
        coset_indices,
        local_group_count,
        &entries,
    )?;
    let runtime_factorization = factorization
        .as_ref()
        .map(|factorization| {
            build_runtime_factorization(factorization, local_group_count, &entries)
        })
        .transpose()?;
    let ordered_destination_ids = ordered_group_ids
        .iter()
        .map(|group_id| destination_by_group[*group_id as usize])
        .collect();

    Ok(RecurrenceColorContraction {
        accuracy,
        storage,
        includes_color_factor: flags & FLAG_INCLUDES_COLOR_FACTOR != 0,
        group_count,
        sector_count,
        component_count,
        local_group_count,
        destination_count,
        entries,
        exact_factors,
        ordered_group_ids,
        destination_by_group,
        sector_by_group,
        component_by_group,
        owner_by_sector,
        ordered_destination_ids,
        factorization,
        runtime_factorization,
    })
}

/// Return the caller-owned deterministic SHA-256 digest of canonical bytes.
pub fn recurrence_color_contraction_digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn validate_raw_entry(
    index: usize,
    entry: RawColorContractionEntry,
    group_count: u32,
    seen_pairs: &mut BTreeSet<(u32, u32)>,
) -> RusticolResult<()> {
    if entry.left_group_id >= group_count || entry.right_group_id >= group_count {
        return Err(malformed(format!(
            "entry {index} references an out-of-bounds group"
        )));
    }
    if entry.left_group_id > entry.right_group_id {
        return Err(malformed(format!(
            "entry {index} is not canonical upper triangular"
        )));
    }
    if !seen_pairs.insert((entry.left_group_id, entry.right_group_id)) {
        return Err(malformed(format!(
            "entry {index} duplicates a canonical group pair"
        )));
    }
    if !entry.weight_re.is_finite()
        || !entry.weight_im.is_finite()
        || !entry.symmetry_factor.is_finite()
    {
        return Err(malformed(format!(
            "entry {index} contains a non-finite f64"
        )));
    }
    if !(entry.weight_re * entry.symmetry_factor).is_finite()
        || !(entry.weight_im * entry.symmetry_factor).is_finite()
    {
        return Err(malformed(format!(
            "entry {index} overflows after symmetry folding"
        )));
    }
    Ok(())
}

fn validate_exact_matches_f64(
    index: usize,
    entry: RawColorContractionEntry,
    factor: ExactComplexRational,
) -> RusticolResult<()> {
    let actual = [
        entry.weight_re * entry.symmetry_factor,
        entry.weight_im * entry.symmetry_factor,
    ];
    let expected = [
        factor.real().numerator() as f64 / factor.real().denominator() as f64,
        factor.imag().numerator() as f64 / factor.imag().denominator() as f64,
    ];
    for (component, actual, expected) in ["real", "imaginary"]
        .into_iter()
        .zip(actual)
        .zip(expected)
        .map(|((component, actual), expected)| (component, actual, expected))
    {
        let tolerance = f64_ulp(actual).max(f64_ulp(expected));
        if (actual - expected).abs() > tolerance {
            return Err(malformed(format!(
                "entry {index} {component} f64 coefficient disagrees with its exact color factor",
            )));
        }
    }
    Ok(())
}

fn f64_ulp(value: f64) -> f64 {
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let magnitude = value.abs();
    if !magnitude.is_finite() {
        return f64::INFINITY;
    }
    f64::from_bits(magnitude.to_bits() + 1) - magnitude
}

fn validate_permutation(values: &[u32], count: u32, label: &str) -> RusticolResult<()> {
    let mut seen = vec![false; count as usize];
    for value in values {
        let Some(slot) = seen.get_mut(*value as usize) else {
            return Err(malformed(format!("{label} contains an out-of-bounds ID")));
        };
        if *slot {
            return Err(malformed(format!("{label} contains a duplicate ID")));
        }
        *slot = true;
    }
    if seen.iter().any(|value| !value) {
        return Err(malformed(format!("{label} is not a complete permutation")));
    }
    Ok(())
}

fn validate_destination_map(values: &[u32], destination_count: u32) -> RusticolResult<()> {
    let mut seen = BTreeSet::new();
    for value in values {
        if *value >= destination_count {
            return Err(malformed(
                "destination map references an out-of-bounds Direct-Arena destination",
            ));
        }
        if !seen.insert(*value) {
            return Err(malformed(
                "destination map contains a duplicate Direct-Arena destination",
            ));
        }
    }
    Ok(())
}

fn validate_expanded_components(
    entries: &[RawColorContractionEntry],
    component_by_group: &[u32],
) -> RusticolResult<()> {
    for entry in entries {
        if component_by_group[entry.left_group_id as usize]
            != component_by_group[entry.right_group_id as usize]
        {
            return Err(malformed(
                "expanded entry couples groups from different components",
            ));
        }
    }
    Ok(())
}

fn validate_group_identities(
    sector_by_group: &[u32],
    component_by_group: &[u32],
    sector_count: u32,
    component_count: u32,
) -> RusticolResult<()> {
    if sector_by_group.len() != component_by_group.len() {
        return Err(malformed("group identity maps have different lengths"));
    }
    let mut identities = BTreeSet::new();
    for (group_id, (&sector_id, &component_id)) in
        sector_by_group.iter().zip(component_by_group).enumerate()
    {
        if sector_id >= sector_count || component_id >= component_count {
            return Err(malformed(format!(
                "group {group_id} identity references an out-of-bounds sector or component"
            )));
        }
        if !identities.insert((sector_id, component_id)) {
            return Err(malformed(
                "group identity maps repeat a sector/component pair",
            ));
        }
    }
    Ok(())
}

fn validate_sector_owners(
    owner_by_sector: &[u32],
    sector_by_group: &[u32],
    sector_count: u32,
) -> RusticolResult<()> {
    if owner_by_sector.len() != sector_count as usize {
        return Err(malformed("physical sector owner map has the wrong length"));
    }
    let mut fixed_points = BTreeSet::new();
    for (sector_id, owner_id) in owner_by_sector.iter().copied().enumerate() {
        if owner_id == ZERO_SECTOR_OWNER {
            continue;
        }
        if owner_id >= sector_count
            || owner_id as usize > sector_id
            || owner_by_sector[owner_id as usize] != owner_id
        {
            return Err(malformed(format!(
                "physical sector {sector_id} has an invalid canonical owner",
            )));
        }
        if owner_id as usize == sector_id {
            fixed_points.insert(owner_id);
        }
    }
    let active = sector_by_group.iter().copied().collect::<BTreeSet<_>>();
    if active != fixed_points {
        return Err(malformed(
            "active recurrence sectors are not exactly the authenticated owner sectors",
        ));
    }
    Ok(())
}

fn validate_repeated_group_identities(
    ordered_group_ids: &[u32],
    sector_by_group: &[u32],
    component_by_group: &[u32],
    local_group_count: u32,
    component_count: u32,
) -> RusticolResult<()> {
    for local_group in 0..local_group_count as usize {
        let start = local_group
            .checked_mul(component_count as usize)
            .ok_or_else(|| malformed("repeated group identity offset overflows usize"))?;
        let stop = start
            .checked_add(component_count as usize)
            .ok_or_else(|| malformed("repeated group identity range overflows usize"))?;
        let group_ids = ordered_group_ids
            .get(start..stop)
            .ok_or_else(|| malformed("repeated group identity range is truncated"))?;
        let first = *group_ids
            .first()
            .ok_or_else(|| malformed("repeated group identity row is empty"))?;
        let sector_id = sector_by_group[first as usize];
        for (component_id, group_id) in group_ids.iter().copied().enumerate() {
            if sector_by_group[group_id as usize] != sector_id
                || component_by_group[group_id as usize] != component_id as u32
            {
                return Err(malformed(
                    "repeated group identities are not local-color-major/component-minor",
                ));
            }
        }
    }
    Ok(())
}

fn decode_factorization(
    storage: RecurrenceColorStorage,
    factor_kind: u32,
    factor_rank: u32,
    coset_count: usize,
    coset_indices: Vec<u32>,
    local_group_count: u32,
    entries: &[RawColorContractionEntry],
) -> RusticolResult<Option<FactorizedColorContraction>> {
    if factor_kind == 0 {
        if factor_rank != 0 || coset_count != 0 || !coset_indices.is_empty() {
            return Err(malformed(
                "factorization-none carries rank or coset metadata",
            ));
        }
        return Ok(None);
    }
    if storage != RecurrenceColorStorage::Repeated {
        return Err(malformed(
            "factorized metadata requires repeated color storage",
        ));
    }
    let kind = match factor_kind {
        1 => {
            if factor_rank != 2 {
                return Err(malformed("Klein-four factorization must have rank two"));
            }
            FactorizedColorContractionKind::KleinFourWalsh
        }
        2 => {
            if !(3..=MAX_FACTOR_RANK).contains(&factor_rank) {
                return Err(malformed(format!(
                    "elementary-Abelian factorization rank must be in [3, {MAX_FACTOR_RANK}]"
                )));
            }
            FactorizedColorContractionKind::ElementaryAbelianWalsh
        }
        _ => {
            return Err(malformed(format!(
                "unknown factorization discriminant {factor_kind}"
            )));
        }
    };
    let subgroup_order = 1usize
        .checked_shl(factor_rank)
        .ok_or_else(|| malformed("factorization subgroup order overflows usize"))?;
    if coset_count == 0
        || coset_count.checked_mul(subgroup_order) != Some(coset_indices.len())
        || coset_indices.len() != local_group_count as usize
    {
        return Err(malformed(
            "factorization coset shape does not match local groups and rank",
        ));
    }
    validate_permutation(&coset_indices, local_group_count, "factorization coset map")?;
    validate_walsh_invariance(&coset_indices, coset_count, subgroup_order, entries)?;
    Ok(Some(FactorizedColorContraction {
        kind,
        rank: factor_rank,
        coset_count,
        coset_indices,
    }))
}

fn validate_walsh_invariance(
    coset_indices: &[u32],
    coset_count: usize,
    subgroup_order: usize,
    entries: &[RawColorContractionEntry],
) -> RusticolResult<()> {
    let mut matrix = BTreeMap::new();
    for entry in entries {
        if entry.weight_im != 0.0 {
            return Err(malformed(
                "factorized color contraction requires real weights",
            ));
        }
        let mut coefficient = entry.weight_re * entry.symmetry_factor;
        if entry.left_group_id != entry.right_group_id {
            coefficient *= 0.5;
        }
        if !coefficient.is_finite() {
            return Err(malformed(
                "factorized color contraction matrix coefficient is not finite",
            ));
        }
        matrix.insert((entry.left_group_id, entry.right_group_id), coefficient);
    }
    let matrix_value = |left: u32, right: u32| {
        let pair = if left <= right {
            (left, right)
        } else {
            (right, left)
        };
        matrix.get(&pair).copied().unwrap_or(0.0)
    };
    for left_coset_index in 0..coset_count {
        let left_start = left_coset_index * subgroup_order;
        let left_coset = &coset_indices[left_start..left_start + subgroup_order];
        for right_coset_index in 0..coset_count {
            let right_start = right_coset_index * subgroup_order;
            let right_coset = &coset_indices[right_start..right_start + subgroup_order];
            for left_index in 0..subgroup_order {
                for right_index in 0..subgroup_order {
                    let actual = matrix_value(left_coset[left_index], right_coset[right_index]);
                    let expected =
                        matrix_value(left_coset[0], right_coset[left_index ^ right_index]);
                    if actual != expected {
                        return Err(malformed(
                            "factorization cosets are inconsistent with the canonical color matrix",
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn build_runtime_factorization(
    factorization: &FactorizedColorContraction,
    local_group_count: u32,
    entries: &[RawColorContractionEntry],
) -> RusticolResult<RuntimeFactorizedColorContraction> {
    let subgroup_order = factorization.subgroup_order();
    let cosets = (0..factorization.coset_count())
        .map(|index| {
            factorization
                .coset(index)
                .expect("validated factorization coset")
                .to_vec()
        })
        .collect::<Vec<_>>();
    let mut matrix = BTreeMap::new();
    for entry in entries {
        let mut coefficient = entry.weight_re * entry.symmetry_factor;
        if entry.left_group_id != entry.right_group_id {
            coefficient *= 0.5;
        }
        matrix.insert((entry.left_group_id, entry.right_group_id), coefficient);
    }
    let matrix_value = |left: u32, right: u32| {
        let pair = if left <= right {
            (left, right)
        } else {
            (right, left)
        };
        matrix.get(&pair).copied().unwrap_or(0.0)
    };

    let amplitude_scale = match factorization.kind() {
        FactorizedColorContractionKind::KleinFourWalsh => 0.5,
        FactorizedColorContractionKind::ElementaryAbelianWalsh => 1.0,
    };
    let weight_scale = match factorization.kind() {
        FactorizedColorContractionKind::KleinFourWalsh => 1.0,
        FactorizedColorContractionKind::ElementaryAbelianWalsh => 1.0 / subgroup_order as f64,
    };
    let mut transformed_entries = Vec::new();
    for left_coset_index in 0..cosets.len() {
        for right_coset_index in left_coset_index..cosets.len() {
            let left_coset = &cosets[left_coset_index];
            let right_coset = &cosets[right_coset_index];
            let mut weights = (0..subgroup_order)
                .map(|subgroup_index| matrix_value(left_coset[0], right_coset[subgroup_index]))
                .collect::<Vec<_>>();
            walsh_butterfly_f64(&mut weights);
            for (character_index, weight) in weights.into_iter().enumerate() {
                let mut coefficient = weight * weight_scale;
                if left_coset_index != right_coset_index {
                    coefficient *= 2.0;
                }
                if coefficient == 0.0 {
                    continue;
                }
                if !coefficient.is_finite() {
                    return Err(malformed(
                        "runtime factorized color coefficient is not finite",
                    ));
                }
                transformed_entries.push(RuntimeFactorizedColorContractionEntry {
                    left_group_index: left_coset[character_index],
                    right_group_index: right_coset[character_index],
                    coefficient_re: coefficient,
                    coefficient_im: 0.0,
                });
            }
        }
    }
    if cosets
        .iter()
        .flat_map(|coset| coset.iter())
        .any(|group| *group >= local_group_count)
    {
        return Err(malformed(
            "runtime factorization references an out-of-bounds local group",
        ));
    }
    Ok(RuntimeFactorizedColorContraction {
        subgroup_order,
        cosets,
        entries: transformed_entries,
        amplitude_scale,
    })
}

fn walsh_butterfly_f64(values: &mut [f64]) {
    debug_assert!(values.len().is_power_of_two());
    let mut stride = 1;
    while stride < values.len() {
        for start in (0..values.len()).step_by(stride * 2) {
            for offset in 0..stride {
                let left = values[start + offset];
                let right = values[start + stride + offset];
                values[start + offset] = left + right;
                values[start + stride + offset] = left - right;
            }
        }
        stride *= 2;
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize, label: &str) -> RusticolResult<&'a [u8]> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| malformed(format!("{label} offset overflows usize")))?;
        let result = self.bytes.get(self.offset..end).ok_or_else(|| {
            malformed(format!(
                "truncated {label} at byte {}: need {count}, have {}",
                self.offset,
                self.bytes.len().saturating_sub(self.offset)
            ))
        })?;
        self.offset = end;
        Ok(result)
    }

    fn u32(&mut self, label: &str) -> RusticolResult<u32> {
        Ok(u32::from_le_bytes(
            self.take(4, label)?.try_into().expect("checked read"),
        ))
    }

    fn u64(&mut self, label: &str) -> RusticolResult<u64> {
        Ok(u64::from_le_bytes(
            self.take(8, label)?.try_into().expect("checked read"),
        ))
    }

    fn i128(&mut self, label: &str) -> RusticolResult<i128> {
        Ok(i128::from_le_bytes(
            self.take(16, label)?.try_into().expect("checked read"),
        ))
    }

    fn f64(&mut self, label: &str) -> RusticolResult<f64> {
        Ok(f64::from_le_bytes(
            self.take(8, label)?.try_into().expect("checked read"),
        ))
    }

    fn count(&mut self, label: &str) -> RusticolResult<usize> {
        let value = self.u64(label)?;
        usize::try_from(value).map_err(|_| malformed(format!("{label} exceeds usize")))
    }

    fn u32_vec(&mut self, count: usize, label: &str) -> RusticolResult<Vec<u32>> {
        (0..count).map(|_| self.u32(label)).collect()
    }

    fn is_finished(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestWire {
        storage: u32,
        accuracy: u32,
        flags: u32,
        group_count: u32,
        sector_count: u32,
        component_count: u32,
        local_group_count: u32,
        destination_count: u32,
        factor_kind: u32,
        factor_rank: u32,
        entries: Vec<RawColorContractionEntry>,
        exact_factors: Vec<ExactComplexRational>,
        ordered_group_ids: Vec<u32>,
        destination_by_group: Vec<u32>,
        sector_by_group: Vec<u32>,
        component_by_group: Vec<u32>,
        owner_by_sector: Vec<u32>,
        cosets: Vec<Vec<u32>>,
        logical_entry_count: u64,
    }

    impl TestWire {
        fn expanded() -> Self {
            Self {
                storage: 1,
                accuracy: 1,
                flags: FLAG_INCLUDES_COLOR_FACTOR,
                group_count: 2,
                sector_count: 2,
                component_count: 1,
                local_group_count: 0,
                destination_count: 12,
                factor_kind: 0,
                factor_rank: 0,
                entries: vec![
                    RawColorContractionEntry {
                        left_group_id: 0,
                        right_group_id: 0,
                        weight_re: 3.0,
                        weight_im: 0.0,
                        symmetry_factor: 1.0,
                        exact_factor_id: 0,
                    },
                    RawColorContractionEntry {
                        left_group_id: 0,
                        right_group_id: 1,
                        weight_re: 2.0,
                        weight_im: 0.5,
                        symmetry_factor: 2.0,
                        exact_factor_id: 1,
                    },
                ],
                exact_factors: vec![
                    ExactComplexRational::new(
                        ExactRational::new(3, 1).unwrap(),
                        ExactRational::ZERO,
                    ),
                    ExactComplexRational::new(
                        ExactRational::new(4, 1).unwrap(),
                        ExactRational::new(1, 1).unwrap(),
                    ),
                ],
                ordered_group_ids: vec![0, 1],
                destination_by_group: vec![7, 9],
                sector_by_group: vec![0, 1],
                component_by_group: vec![0, 0],
                owner_by_sector: vec![0, 1],
                cosets: Vec::new(),
                logical_entry_count: 2,
            }
        }

        fn repeated_k4() -> Self {
            Self {
                storage: 2,
                accuracy: 2,
                flags: FLAG_INCLUDES_COLOR_FACTOR,
                group_count: 8,
                sector_count: 4,
                component_count: 2,
                local_group_count: 4,
                destination_count: 20,
                factor_kind: 1,
                factor_rank: 2,
                entries: (0..4)
                    .map(|index| RawColorContractionEntry {
                        left_group_id: index,
                        right_group_id: index,
                        weight_re: 1.0,
                        weight_im: 0.0,
                        symmetry_factor: 1.0,
                        exact_factor_id: 0,
                    })
                    .collect(),
                exact_factors: vec![ExactComplexRational::ONE],
                ordered_group_ids: vec![0, 4, 1, 5, 2, 6, 3, 7],
                destination_by_group: vec![8, 9, 10, 11, 12, 13, 14, 15],
                sector_by_group: vec![0, 1, 2, 3, 0, 1, 2, 3],
                component_by_group: vec![0, 0, 0, 0, 1, 1, 1, 1],
                owner_by_sector: vec![0, 1, 2, 3],
                cosets: vec![vec![0, 1, 2, 3]],
                logical_entry_count: 8,
            }
        }

        fn repeated_k4_multicoset() -> Self {
            let diagonal_blocks = [[2.0, 1.0, 0.0, -1.0], [3.0, 0.5, -0.5, 1.5]];
            let cross_block = [1.0, 2.0, -1.0, 0.25];
            let matrix_value = |left: usize, right: usize| {
                let left_coset = left / 4;
                let right_coset = right / 4;
                let xor = (left % 4) ^ (right % 4);
                if left_coset == right_coset {
                    diagonal_blocks[left_coset][xor]
                } else {
                    cross_block[xor]
                }
            };
            let mut entries = Vec::new();
            let mut exact_factors = Vec::new();
            for left in 0..8 {
                for right in left..8 {
                    let weight = matrix_value(left, right);
                    if weight == 0.0 {
                        continue;
                    }
                    let symmetry = if left == right { 1.0 } else { 2.0 };
                    let coefficient = weight * symmetry;
                    let exact_factor_id = exact_factors.len() as u32;
                    exact_factors.push(ExactComplexRational::new(
                        ExactRational::from_f64_exact(coefficient).unwrap(),
                        ExactRational::ZERO,
                    ));
                    entries.push(RawColorContractionEntry {
                        left_group_id: left as u32,
                        right_group_id: right as u32,
                        weight_re: weight,
                        weight_im: 0.0,
                        symmetry_factor: symmetry,
                        exact_factor_id,
                    });
                }
            }
            let logical_entry_count = (2 * entries.len()) as u64;
            Self {
                storage: 2,
                accuracy: 2,
                flags: FLAG_INCLUDES_COLOR_FACTOR,
                group_count: 16,
                sector_count: 8,
                component_count: 2,
                local_group_count: 8,
                destination_count: 16,
                factor_kind: 1,
                factor_rank: 2,
                entries,
                exact_factors,
                ordered_group_ids: (0..16).collect(),
                destination_by_group: (0..16).collect(),
                sector_by_group: (0..8).flat_map(|sector| [sector, sector]).collect(),
                component_by_group: (0..8).flat_map(|_| [0, 1]).collect(),
                owner_by_sector: (0..8).collect(),
                cosets: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
                logical_entry_count,
            }
        }

        fn encode(&self) -> Vec<u8> {
            let flattened_cosets = self.cosets.iter().flatten().copied().collect::<Vec<_>>();
            let payload_bytes = self.entries.len() * ENTRY_BYTES
                + self.exact_factors.len() * EXACT_FACTOR_BYTES
                + self.ordered_group_ids.len() * 4
                + self.destination_by_group.len() * 4
                + self.sector_by_group.len() * 4
                + self.component_by_group.len() * 4
                + self.owner_by_sector.len() * 4
                + flattened_cosets.len() * 4;
            let mut bytes = Vec::with_capacity(HEADER_BYTES + payload_bytes);
            bytes.extend_from_slice(MAGIC);
            for value in [
                VERSION,
                HEADER_BYTES as u32,
                self.storage,
                self.accuracy,
                self.flags,
                self.group_count,
                self.sector_count,
                self.component_count,
                self.local_group_count,
                self.destination_count,
                self.factor_kind,
                self.factor_rank,
                ENTRY_BYTES as u32,
                EXACT_FACTOR_BYTES as u32,
            ] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            for value in [
                self.entries.len() as u64,
                self.exact_factors.len() as u64,
                self.cosets.len() as u64,
                flattened_cosets.len() as u64,
                self.logical_entry_count,
                self.owner_by_sector.len() as u64,
                payload_bytes as u64,
            ] {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            assert_eq!(bytes.len(), HEADER_BYTES);
            for entry in &self.entries {
                bytes.extend_from_slice(&entry.left_group_id.to_le_bytes());
                bytes.extend_from_slice(&entry.right_group_id.to_le_bytes());
                bytes.extend_from_slice(&entry.weight_re.to_le_bytes());
                bytes.extend_from_slice(&entry.weight_im.to_le_bytes());
                bytes.extend_from_slice(&entry.symmetry_factor.to_le_bytes());
                bytes.extend_from_slice(&entry.exact_factor_id.to_le_bytes());
            }
            for factor in &self.exact_factors {
                for value in [
                    factor.real().numerator(),
                    factor.real().denominator(),
                    factor.imag().numerator(),
                    factor.imag().denominator(),
                ] {
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
            for value in self
                .ordered_group_ids
                .iter()
                .chain(&self.destination_by_group)
                .chain(&self.sector_by_group)
                .chain(&self.component_by_group)
                .chain(&self.owner_by_sector)
                .chain(&flattened_cosets)
            {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            bytes
        }
    }

    fn error_contains(bytes: &[u8], expected: &str) {
        let error = decode_recurrence_color_contraction_v3(bytes).unwrap_err();
        assert!(
            error.message().contains(expected),
            "expected {expected:?} in {:?}",
            error.message()
        );
    }

    #[test]
    fn expanded_payload_preserves_raw_and_derives_runtime_entries() {
        let bytes = TestWire::expanded().encode();
        let plan = decode_recurrence_color_contraction_v3(&bytes).unwrap();
        assert_eq!(plan.accuracy(), RecurrenceColorAccuracy::Nlc);
        assert_eq!(plan.storage(), RecurrenceColorStorage::Expanded);
        assert!(plan.includes_color_factor());
        assert_eq!(plan.sector_count(), 2);
        assert_eq!(plan.component_count(), 1);
        assert_eq!(plan.ordered_group_ids(), [0, 1]);
        assert_eq!(plan.destination_by_group(), [7, 9]);
        let raw = plan.canonical_logical_entries().collect::<Vec<_>>();
        assert_eq!(raw[1].weight_re, 2.0);
        assert_eq!(raw[1].weight_im, 0.5);
        assert_eq!(raw[1].symmetry_factor, 2.0);
        let runtime = plan.runtime_entries().collect::<Vec<_>>();
        assert_eq!(runtime[1].left_destination_id, 7);
        assert_eq!(runtime[1].right_destination_id, 9);
        assert_eq!(runtime[1].coefficient_re, 4.0);
        assert_eq!(runtime[1].coefficient_im, 1.0);
        assert_eq!(
            recurrence_color_contraction_digest(&bytes),
            recurrence_color_contraction_digest(&bytes)
        );
    }

    #[test]
    fn repeated_k4_payload_expands_logical_rows_without_runtime_allocation() {
        let bytes = TestWire::repeated_k4().encode();
        let plan = decode_recurrence_color_contraction_v3(&bytes).unwrap();
        assert_eq!(plan.storage(), RecurrenceColorStorage::Repeated);
        assert_eq!(plan.logical_entry_count(), 8);
        let factor = plan.factorization().unwrap();
        assert_eq!(
            factor.kind(),
            FactorizedColorContractionKind::KleinFourWalsh
        );
        assert_eq!(factor.coset(0), Some(&[0, 1, 2, 3][..]));
        let runtime_factor = plan.runtime_factorization().unwrap();
        assert_eq!(runtime_factor.subgroup_order(), 4);
        assert_eq!(runtime_factor.cosets(), [vec![0, 1, 2, 3]]);
        assert_eq!(runtime_factor.amplitude_scale(), 0.5);
        assert_eq!(runtime_factor.entries().len(), 4);
        assert!(runtime_factor.entries().iter().all(|entry| {
            entry.left_group_index == entry.right_group_index
                && entry.coefficient_re == 1.0
                && entry.coefficient_im == 0.0
        }));
        assert_eq!(plan.ordered_destination_id(0, 0), Some(8));
        assert_eq!(plan.ordered_destination_id(0, 1), Some(12));
        assert_eq!(plan.ordered_destination_id(3, 1), Some(15));
        let mut entries = plan.canonical_logical_entries();
        assert_eq!(entries.len(), 8);
        assert_eq!(entries.next().unwrap().left_group_id, 0);
        assert_eq!(entries.next().unwrap().left_group_id, 1);
        assert_eq!(entries.next().unwrap().left_group_id, 2);
        assert_eq!(entries.next().unwrap().left_group_id, 3);
        assert_eq!(entries.next().unwrap().left_group_id, 4);
        assert_eq!(entries.len(), 3);
    }

    #[test]
    fn repeated_multicoset_k4_preserves_the_nontrivial_quadratic_form() {
        let plan =
            decode_recurrence_color_contraction_v3(&TestWire::repeated_k4_multicoset().encode())
                .unwrap();
        let amplitudes = [0.5, -1.25, 2.0, 0.75, -0.4, 1.1, 0.2, -0.9];
        let direct = plan
            .entries()
            .iter()
            .map(|entry| {
                entry.weight_re
                    * entry.symmetry_factor
                    * amplitudes[entry.left_group_id as usize]
                    * amplitudes[entry.right_group_id as usize]
            })
            .sum::<f64>();

        let factorized = plan.runtime_factorization().unwrap();
        assert_eq!(factorized.cosets().len(), 2);
        let mut transformed = [0.0; 8];
        for coset in factorized.cosets() {
            let mut values = coset
                .iter()
                .map(|index| amplitudes[*index as usize])
                .collect::<Vec<_>>();
            walsh_butterfly_f64(&mut values);
            for (index, value) in coset.iter().zip(values) {
                transformed[*index as usize] = value * factorized.amplitude_scale();
            }
        }
        let transformed_value = factorized
            .entries()
            .iter()
            .map(|entry| {
                entry.coefficient_re
                    * transformed[entry.left_group_index as usize]
                    * transformed[entry.right_group_index as usize]
            })
            .sum::<f64>();
        assert!((direct - transformed_value).abs() <= 32.0 * f64::EPSILON);
    }

    #[test]
    fn decoder_rejects_mixed_duplicate_out_of_bounds_and_nonfinite_data() {
        let mut mixed = TestWire::expanded();
        mixed.factor_kind = 1;
        mixed.factor_rank = 2;
        mixed.cosets = vec![vec![0, 1, 2, 3]];
        error_contains(&mixed.encode(), "mixed");

        let mut duplicate = TestWire::expanded();
        duplicate.entries.push(duplicate.entries[0]);
        duplicate.logical_entry_count += 1;
        error_contains(&duplicate.encode(), "duplicates");

        let mut out_of_bounds = TestWire::expanded();
        out_of_bounds.entries[0].left_group_id = 2;
        error_contains(&out_of_bounds.encode(), "out-of-bounds");

        let mut nonfinite = TestWire::expanded();
        nonfinite.entries[0].weight_re = f64::NAN;
        error_contains(&nonfinite.encode(), "non-finite");

        let mut duplicate_destination = TestWire::expanded();
        duplicate_destination.destination_by_group = vec![7, 7];
        error_contains(&duplicate_destination.encode(), "duplicate Direct-Arena");
    }

    #[test]
    fn decoder_rejects_inconsistent_factorization_map_and_matrix() {
        let mut duplicate_coset = TestWire::repeated_k4();
        duplicate_coset.cosets[0][3] = 2;
        error_contains(&duplicate_coset.encode(), "duplicate ID");

        let mut non_invariant = TestWire::repeated_k4();
        non_invariant.entries[0].weight_re = 2.0;
        non_invariant.entries[0].exact_factor_id = 1;
        non_invariant.exact_factors.push(ExactComplexRational::new(
            ExactRational::new(2, 1).unwrap(),
            ExactRational::ZERO,
        ));
        error_contains(&non_invariant.encode(), "inconsistent");
    }

    #[test]
    fn decoder_rejects_incomplete_or_noncanonical_sector_ownership() {
        let mut missing_owner = TestWire::expanded();
        missing_owner.owner_by_sector = vec![0, 0];
        error_contains(&missing_owner.encode(), "owner sectors");

        let mut forward_owner = TestWire::expanded();
        forward_owner.owner_by_sector = vec![1, 1];
        error_contains(&forward_owner.encode(), "invalid canonical owner");
    }

    #[test]
    fn decoder_rejects_cross_component_expanded_entries_and_trailing_bytes() {
        let mut cross_component = TestWire::expanded();
        cross_component.sector_count = 1;
        cross_component.component_count = 2;
        cross_component.sector_by_group = vec![0, 0];
        cross_component.component_by_group = vec![0, 1];
        cross_component.owner_by_sector = vec![0];
        error_contains(&cross_component.encode(), "different components");

        let mut trailing = TestWire::expanded().encode();
        trailing.push(0);
        error_contains(&trailing, "fixed-width sections");
    }
}
