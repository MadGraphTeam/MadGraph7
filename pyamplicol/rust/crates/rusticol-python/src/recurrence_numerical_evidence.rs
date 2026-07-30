// SPDX-License-Identifier: 0BSD

//! Fail-closed authentication of recurrence numerical-current evidence.
//!
//! Python supplies complete raw candidate and independent-verification captures.
//! This module independently reconstructs their schema, probe contexts, exact
//! rational residuals, decision census, certificates, and current mappings
//! before the native Direct-Arena lowering is allowed to apply any reuse.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use rusticol_core::recurrence::process::ValidatedRecurrenceProcessInput;
use rusticol_core::recurrence::template::{ParameterKind, ValidatedRecurrenceTemplateInput};
use rusticol_core::recurrence::{
    RecurrenceNumericalCurrentMapping, RecurrenceNumericalRelationEvidence,
    RecurrenceRelationDiscoveryMode,
};
use rusticol_core::{RusticolError, RusticolResult};
use serde_json::{Map as JsonMap, Value as JsonValue};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::io::{self, Write};
use std::str::FromStr;

use crate::recurrence::{
    canonical_json_bytes, canonical_json_sha256, hex_digest, invalid, json_array, json_bool,
    json_field, json_nonempty_string, json_object, json_string, json_u32, require_json_fields,
    require_json_string_value, semantic_digest_from_hex,
};

type ExactProbeComplex = (BigRational, BigRational);

const MAX_RAW_EVIDENCE_BYTES: usize = 157_853_696;
const MAX_RAW_JSON_DEPTH: usize = 32;
const MAX_RAW_JSON_STRING_BYTES: usize = 65_536;
const MAX_RAW_JSON_STRUCTURAL_TOKENS: usize = 8_000_000;
const MAX_RAW_RESIDENT_BYTES: usize = 1 << 30;
const RAW_PRE_DOM_FIXED_BYTES: usize = 32 * 1024 * 1024;
const RAW_PRE_DOM_WIRE_COPIES: usize = 2;
const RAW_PRE_DOM_BYTES_PER_TOKEN: usize = 80;
const RAW_CAPTURE_BYTES_PER_SCALAR: usize = 320;
const RAW_CAPTURE_BYTES_PER_ROW: usize = 512;
const RAW_PRODUCER_BYTES_PER_SCALAR: usize = 640;
const RAW_PRODUCER_BYTES_PER_ROW: usize = 512;
const MIN_RAW_EVIDENCE_WIRE_BYTES: usize = 1 << 20;

struct CanonicalJsonMatcher<'a> {
    expected: &'a [u8],
    offset: usize,
    matches: bool,
}

impl Write for CanonicalJsonMatcher<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        let end = self.offset.saturating_add(buffer.len());
        if end > self.expected.len() || self.expected[self.offset..end] != *buffer {
            self.matches = false;
        }
        self.offset = end;
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn validate_canonical_json_bytes(
    value: &JsonValue,
    expected: &[u8],
    context: &str,
) -> RusticolResult<()> {
    let mut matcher = CanonicalJsonMatcher {
        expected,
        offset: 0,
        matches: true,
    };
    serde_json::to_writer(&mut matcher, value)
        .map_err(|error| invalid(format!("could not canonicalize {context}: {error}")))?;
    if !matcher.matches || matcher.offset != expected.len() {
        return Err(invalid(format!("{context} is not canonical ASCII JSON")));
    }
    Ok(())
}

fn validate_raw_json_lexical_budget(bytes: &[u8]) -> RusticolResult<usize> {
    let mut in_string = false;
    let mut escaped = false;
    let mut string_bytes = 0_usize;
    let mut depth = 0_usize;
    let mut structural_tokens = 0_usize;
    for &byte in bytes {
        if !byte.is_ascii() {
            return Err(invalid(
                "numerical relation evidence must be canonical ASCII JSON",
            ));
        }
        if in_string {
            string_bytes = string_bytes
                .checked_add(1)
                .ok_or_else(|| invalid("raw numerical JSON string length overflows usize"))?;
            if string_bytes > MAX_RAW_JSON_STRING_BYTES {
                return Err(invalid(
                    "numerical relation evidence JSON string exceeds its lexical boundary",
                ));
            }
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
                structural_tokens += 1;
            }
            continue;
        }
        match byte {
            b'"' => {
                in_string = true;
                string_bytes = 0;
            }
            b'{' | b'[' => {
                depth += 1;
                structural_tokens += 1;
                if depth > MAX_RAW_JSON_DEPTH {
                    return Err(invalid(
                        "numerical relation evidence JSON nesting exceeds its boundary",
                    ));
                }
            }
            b'}' | b']' => {
                depth = depth.saturating_sub(1);
                structural_tokens += 1;
            }
            b',' | b':' => structural_tokens += 1,
            _ => {}
        }
        if structural_tokens > MAX_RAW_JSON_STRUCTURAL_TOKENS {
            return Err(invalid(
                "numerical relation evidence JSON token count exceeds its boundary",
            ));
        }
    }
    if in_string || depth != 0 {
        return Err(invalid(
            "numerical relation evidence JSON lexical structure is incomplete",
        ));
    }
    Ok(structural_tokens)
}

fn validate_pre_dom_resident_budget(
    raw_byte_count: usize,
    structural_token_count: usize,
) -> RusticolResult<()> {
    // serde_json materializes an owned DOM.  Bound that allocation before
    // from_slice: two wire residents, 80 bytes per lexically counted value /
    // delimiter token, and 32 MiB of parser/fixed headroom.  The independent
    // post-DOM geometry check below bounds BigRational capture residents.
    let resident_bytes = raw_byte_count
        .checked_mul(RAW_PRE_DOM_WIRE_COPIES)
        .and_then(|wire| {
            structural_token_count
                .checked_mul(RAW_PRE_DOM_BYTES_PER_TOKEN)
                .and_then(|dom| wire.checked_add(dom))
        })
        .and_then(|dynamic| dynamic.checked_add(RAW_PRE_DOM_FIXED_BYTES))
        .ok_or_else(|| invalid("raw numerical pre-DOM resident bound overflows usize"))?;
    if resident_bytes > MAX_RAW_RESIDENT_BYTES {
        return Err(invalid(
            "raw numerical evidence exceeds the explicit pre-DOM 1 GiB resident memory envelope",
        ));
    }
    Ok(())
}

fn validate_combined_raw_resident_budget(
    raw_byte_count: usize,
    structural_token_count: usize,
    scalar_count: usize,
    row_count: usize,
) -> RusticolResult<()> {
    // At capture parsing time the input Vec and Python bytes, serde DOM, and
    // independently constructed BigRational/String capture graph coexist.
    // Keep all of those residents in one checked 1 GiB envelope.
    let resident_bytes = raw_byte_count
        .checked_mul(RAW_PRE_DOM_WIRE_COPIES)
        .and_then(|wire| {
            structural_token_count
                .checked_mul(RAW_PRE_DOM_BYTES_PER_TOKEN)
                .and_then(|dom| wire.checked_add(dom))
        })
        .and_then(|total| {
            scalar_count
                .checked_mul(RAW_CAPTURE_BYTES_PER_SCALAR)
                .and_then(|capture| total.checked_add(capture))
        })
        .and_then(|total| {
            row_count
                .checked_mul(RAW_CAPTURE_BYTES_PER_ROW)
                .and_then(|rows| total.checked_add(rows))
        })
        .and_then(|dynamic| dynamic.checked_add(RAW_PRE_DOM_FIXED_BYTES))
        .ok_or_else(|| invalid("raw numerical combined resident bound overflows usize"))?;
    if resident_bytes > MAX_RAW_RESIDENT_BYTES {
        return Err(invalid(
            "raw numerical evidence exceeds the combined 1 GiB resident memory envelope",
        ));
    }
    Ok(())
}

fn raw_evidence_shape_wire_limit(scalar_count: usize, row_count: usize) -> RusticolResult<usize> {
    // Authenticate the producer-side shape budget independently.  Python
    // reserves all capture/DOM/exact scalar residents conservatively here;
    // the native lexical and combined-DOM checks remain separate bounds.
    let resident_without_wire = scalar_count
        .checked_mul(RAW_PRODUCER_BYTES_PER_SCALAR)
        .and_then(|scalars| {
            row_count
                .checked_mul(RAW_PRODUCER_BYTES_PER_ROW)
                .and_then(|rows| scalars.checked_add(rows))
        })
        .and_then(|dynamic| dynamic.checked_add(RAW_PRE_DOM_FIXED_BYTES))
        .ok_or_else(|| invalid("raw numerical producer resident bound overflows usize"))?;
    let minimum_resident = MIN_RAW_EVIDENCE_WIRE_BYTES
        .checked_mul(RAW_PRE_DOM_WIRE_COPIES)
        .and_then(|wire| resident_without_wire.checked_add(wire))
        .ok_or_else(|| invalid("raw numerical producer wire reserve overflows usize"))?;
    if minimum_resident > MAX_RAW_RESIDENT_BYTES {
        return Err(invalid(
            "raw numerical capture geometry leaves no minimum canonical wire reserve inside the 1 GiB resident memory envelope",
        ));
    }
    Ok(MAX_RAW_EVIDENCE_BYTES
        .min((MAX_RAW_RESIDENT_BYTES - resident_without_wire) / RAW_PRE_DOM_WIRE_COPIES))
}

#[derive(Clone, Debug)]
struct RawProbeValue {
    real_text: String,
    imaginary_text: String,
    value: ExactProbeComplex,
}

#[derive(Clone, Debug)]
struct RawNumericalCapture {
    point_sha256s: Vec<String>,
    kinematic_sha256s: Vec<String>,
    parameter_context_sha256s: Vec<String>,
    parameter_contexts: Vec<Vec<BigRational>>,
    observations: BTreeMap<u32, Vec<RawProbeValue>>,
    dimensions: BTreeMap<u32, u16>,
    observation_batch_sha256: String,
    capture_contract_sha256: String,
}

#[derive(Clone, Debug)]
struct RawSourceCurrent {
    current_id: u32,
    is_source: bool,
    contract_key: Vec<u8>,
    dimension: u16,
}

#[derive(Clone, Debug)]
struct RawSourceSemantics {
    value: JsonValue,
    process_id: String,
    physical_pdgs: Vec<i32>,
    strategy: String,
    selector_schedule: JsonValue,
    currents: Vec<RawSourceCurrent>,
}

#[derive(Clone, Debug)]
struct DerivedNumericalRelation {
    mapping: RecurrenceNumericalCurrentMapping,
    certificate: JsonValue,
    mapping_json: JsonValue,
}

#[derive(Clone, Debug)]
struct RawNumericalCandidateIndex {
    observation_index: usize,
    scalar_component: usize,
    entries: Vec<(BigRational, u32)>,
}

#[derive(Clone, Debug)]
struct RawNumericalDerivation {
    relations: Vec<DerivedNumericalRelation>,
    numerical_candidate_count: usize,
    verification_rejected_count: usize,
    rejected_hypothesis_count: usize,
    tested_hypothesis_count: usize,
    theoretical_pair_hypothesis_count: usize,
    screened_pair_hypothesis_count: usize,
    zero_hypothesis_count: usize,
    candidate_index_contract_count: usize,
    exhaustive_fallback_contract_count: usize,
    decision_sha256: String,
    rejection_decision_sha256: String,
}

#[derive(Clone, Debug)]
struct AuthenticatedRuntimeParameterSlot {
    default: BigRational,
    probe_policy: &'static str,
}

#[derive(Clone, Debug)]
pub(super) struct AuthenticatedRuntimeParameterContract {
    schema: JsonValue,
    schema_sha256: String,
    slots: Vec<AuthenticatedRuntimeParameterSlot>,
    process_id: String,
    physical_pdgs: Vec<i32>,
}

#[derive(Clone, Debug)]
struct RelationResiduals {
    maximum_absolute: BigRational,
    maximum_relative: BigRational,
    maximum_ratio: BigRational,
}

pub(super) fn authenticated_runtime_parameter_contract(
    process: &ValidatedRecurrenceProcessInput,
    template: &ValidatedRecurrenceTemplateInput,
) -> RusticolResult<AuthenticatedRuntimeParameterContract> {
    const ABI: &str = "pyamplicol-recurrence-runtime-parameter-schema-v2";
    const MATERIAL_POLICY: &str = "native-template-default-perturbed-v1";
    const DERIVED_POLICY: &str = "derived-overwritten-fixed-zero-v1";
    let input = process.input();
    let parameter_contracts = template.parameter_kind_and_default_factors()?;
    let mut parameters = Vec::with_capacity(input.parameter_projection.len());
    let mut slots = Vec::with_capacity(input.parameter_projection.len());
    for (index, row) in input.parameter_projection.iter().enumerate() {
        if row.runtime_slot as usize != index {
            return Err(invalid(
                "authenticated runtime parameter projection is not dense",
            ));
        }
        let range = input
            .string_ranges
            .get(row.runtime_name_string_id as usize)
            .copied()
            .ok_or_else(|| invalid("runtime-parameter name string range is absent"))?
            .as_usize_range(
                input.string_bytes.len(),
                "runtime-parameter name string range",
            )?;
        let runtime_name = std::str::from_utf8(&input.string_bytes[range])
            .map_err(|_| invalid("runtime-parameter name is not UTF-8"))?;
        let (kind, default_factor) = parameter_contracts
            .get(row.parameter_template_id as usize)
            .copied()
            .ok_or_else(|| invalid("runtime-parameter projection references an absent template"))?;
        let (default, default_binary64, probe_policy) = if kind == ParameterKind::Derived {
            (BigRational::zero(), 0.0_f64, DERIVED_POLICY)
        } else {
            let factor = default_factor.ok_or_else(|| {
                invalid("material runtime parameter has no authenticated template default")
            })?;
            let exact = match row.component {
                0 => factor.real(),
                1 => factor.imag(),
                _ => {
                    return Err(invalid(
                        "runtime-parameter component is not real or imaginary",
                    ));
                }
            };
            let rational = BigRational::new(
                BigInt::from(exact.numerator()),
                BigInt::from(exact.denominator()),
            );
            let binary64 = rational.to_f64().ok_or_else(|| {
                invalid("authenticated runtime-parameter default is outside binary64")
            })?;
            if !binary64.is_finite() {
                return Err(invalid(
                    "authenticated runtime-parameter default is not finite binary64",
                ));
            }
            (
                exact_binary64_rational(binary64)?,
                binary64,
                MATERIAL_POLICY,
            )
        };
        parameters.push(serde_json::json!({
            "runtime_slot": row.runtime_slot,
            "runtime_name": runtime_name,
            "parameter_template_id": row.parameter_template_id,
            "prepared_parameter_id": row.prepared_parameter_id(),
            "component": row.component,
            "default_binary64": python_f64_hex_signed(default_binary64),
            "probe_policy": probe_policy,
        }));
        slots.push(AuthenticatedRuntimeParameterSlot {
            default,
            probe_policy,
        });
    }
    let schema = serde_json::json!({
        "abi": ABI,
        "parameters": parameters,
    });
    let schema_sha256 =
        canonical_json_sha256(&schema, "native recurrence runtime-parameter schema")?;
    Ok(AuthenticatedRuntimeParameterContract {
        schema,
        schema_sha256,
        slots,
        process_id: process.summary().process_id().to_owned(),
        physical_pdgs: input
            .external_legs
            .iter()
            .map(|leg| leg.physical_pdg)
            .collect(),
    })
}

pub(super) fn parse_numerical_relation_evidence(
    bytes: &[u8],
    runtime_parameters: &AuthenticatedRuntimeParameterContract,
) -> RusticolResult<RecurrenceNumericalRelationEvidence> {
    // The global byte ceiling protects serde before the source shape is
    // available.  Once parsed, the authenticated scalar/row geometry derives
    // a possibly smaller wire allowance inside the same 1 GiB envelope.
    const ABI: &str = "pyamplicol-recurrence-numerical-current-evidence-v3";
    const RELATION_SET_ABI: &str = "pyamplicol-authenticated-numerical-current-relation-set-v2";
    if bytes.is_empty() {
        return Err(invalid("numerical relation evidence must not be empty"));
    }
    if bytes.len() > MAX_RAW_EVIDENCE_BYTES {
        return Err(invalid(format!(
            "numerical relation evidence exceeds the explicit {MAX_RAW_EVIDENCE_BYTES}-byte generation boundary"
        )));
    }
    let structural_token_count = validate_raw_json_lexical_budget(bytes)?;
    validate_pre_dom_resident_budget(bytes.len(), structural_token_count)?;
    let value: JsonValue = serde_json::from_slice(bytes)
        .map_err(|error| invalid(format!("numerical relation evidence is not JSON: {error}")))?;
    validate_canonical_json_bytes(&value, bytes, "numerical relation evidence")?;
    let object = json_object(&value, "numerical relation evidence")?;
    require_json_fields(
        object,
        &[
            "abi",
            "requested_mode",
            "schedule_semantic_digest",
            "baseline_runtime_layout_digest",
            "source_semantics",
            "source_semantics_sha256",
            "runtime_parameter_schema",
            "runtime_parameter_schema_sha256",
            "candidate_capture",
            "verification_capture",
            "certificate_algorithm",
            "certificate_set_sha256",
            "precision_digits",
            "probe_count",
            "verification_probe_count",
            "relative_tolerance_binary64",
            "absolute_tolerance_binary64",
            "seed",
            "candidate_index",
            "numerical_candidate_count",
            "verification_rejected_count",
            "rejected_hypothesis_count",
            "tested_hypothesis_count",
            "decision_sha256",
            "rejection_decision_sha256",
            "certificates",
            "mappings",
        ],
        "numerical relation evidence",
    )?;
    require_json_string_value(object, "abi", ABI, "numerical relation evidence")?;
    let requested_mode = RecurrenceRelationDiscoveryMode::parse(json_string(
        object,
        "requested_mode",
        "numerical relation evidence mode",
    )?)?;
    if requested_mode == RecurrenceRelationDiscoveryMode::Off {
        return Err(invalid("off mode cannot carry numerical relation evidence"));
    }
    let schedule_semantic_digest = evidence_sha256(
        object,
        "schedule_semantic_digest",
        "numerical relation schedule digest",
    )?;
    let baseline_runtime_layout_digest = evidence_sha256(
        object,
        "baseline_runtime_layout_digest",
        "numerical relation baseline layout digest",
    )?;
    let source_semantics_sha256 = evidence_sha256(
        object,
        "source_semantics_sha256",
        "numerical relation source semantics",
    )?;
    let source = parse_raw_source_semantics(
        json_field(object, "source_semantics", "numerical relation evidence")?,
        &source_semantics_sha256,
        &schedule_semantic_digest,
        &baseline_runtime_layout_digest,
        runtime_parameters,
    )?;
    let runtime_parameter_schema_sha256 = evidence_sha256(
        object,
        "runtime_parameter_schema_sha256",
        "recurrence runtime parameter schema",
    )?;
    let runtime_parameter_count = validate_raw_runtime_parameter_schema(
        json_field(
            object,
            "runtime_parameter_schema",
            "numerical relation evidence",
        )?,
        &runtime_parameter_schema_sha256,
        runtime_parameters,
    )?;
    let certificate_algorithm = json_nonempty_string(
        object,
        "certificate_algorithm",
        "numerical relation certificate algorithm",
    )?
    .to_owned();
    if certificate_algorithm != rusticol_core::recurrence::NUMERICAL_RELATION_CERTIFICATE_ALGORITHM
    {
        return Err(invalid(
            "numerical relation evidence uses an unsupported certificate algorithm",
        ));
    }
    let precision_digits = json_u32(object, "precision_digits", "numerical relation precision")?;
    let probe_count = json_u32(
        object,
        "probe_count",
        "numerical relation candidate probe count",
    )?;
    let verification_probe_count = json_u32(
        object,
        "verification_probe_count",
        "numerical relation verification probe count",
    )?;
    let (scalar_count, row_count, shape_wire_limit) = validate_raw_evidence_geometry(
        &source,
        probe_count,
        verification_probe_count,
        runtime_parameter_count,
    )?;
    if bytes.len() > shape_wire_limit {
        return Err(invalid(format!(
            "numerical relation evidence exceeds its authenticated {shape_wire_limit}-byte shape-dependent wire boundary"
        )));
    }
    validate_combined_raw_resident_budget(
        bytes.len(),
        structural_token_count,
        scalar_count,
        row_count,
    )?;
    let relative_tolerance_hex = json_string(
        object,
        "relative_tolerance_binary64",
        "numerical relation relative tolerance encoding",
    )?;
    let absolute_tolerance_hex = json_string(
        object,
        "absolute_tolerance_binary64",
        "numerical relation absolute tolerance encoding",
    )?;
    let (relative_tolerance, relative_exact) = parse_nonnegative_python_f64_hex(
        relative_tolerance_hex,
        "numerical relation relative tolerance encoding",
    )?;
    let (absolute_tolerance, absolute_exact) = parse_nonnegative_python_f64_hex(
        absolute_tolerance_hex,
        "numerical relation absolute tolerance encoding",
    )?;
    if relative_tolerance == 0.0 && absolute_tolerance == 0.0 {
        return Err(RusticolError::integrity(
            "numerical relation tolerances cannot both be zero",
        ));
    }
    if json_string(
        object,
        "relative_tolerance_binary64",
        "numerical relation relative tolerance encoding",
    )? != python_f64_hex(relative_tolerance)
        || json_string(
            object,
            "absolute_tolerance_binary64",
            "numerical relation absolute tolerance encoding",
        )? != python_f64_hex(absolute_tolerance)
    {
        return Err(RusticolError::integrity(
            "numerical relation tolerance encoding disagrees with its binary64 value",
        ));
    }
    let seed = json_u64(object, "seed", "numerical relation seed")?;
    let candidate = parse_raw_numerical_capture(
        json_field(object, "candidate_capture", "numerical relation evidence")?,
        "candidate",
        "candidate-current-probes-v1",
        precision_digits,
        probe_count,
        &source,
        &source_semantics_sha256,
        &runtime_parameter_schema_sha256,
        runtime_parameter_count,
        seed,
    )?;
    let verification = parse_raw_numerical_capture(
        json_field(
            object,
            "verification_capture",
            "numerical relation evidence",
        )?,
        "verification",
        "independent-verification-current-probes-v1",
        precision_digits,
        verification_probe_count,
        &source,
        &source_semantics_sha256,
        &runtime_parameter_schema_sha256,
        runtime_parameter_count,
        seed,
    )?;
    validate_independent_raw_captures(&candidate, &verification, runtime_parameters, seed)?;

    let derivation = derive_raw_numerical_relations(
        &source,
        &candidate,
        &verification,
        precision_digits,
        seed,
        &relative_exact,
        &absolute_exact,
        json_string(
            object,
            "relative_tolerance_binary64",
            "numerical relation relative tolerance encoding",
        )?,
        json_string(
            object,
            "absolute_tolerance_binary64",
            "numerical relation absolute tolerance encoding",
        )?,
        &runtime_parameter_schema_sha256,
        &certificate_algorithm,
    )?;
    let supplied_decision_sha256 = evidence_sha256(
        object,
        "decision_sha256",
        "numerical relation decision chain",
    )?;
    if supplied_decision_sha256 != derivation.decision_sha256 {
        return Err(RusticolError::integrity(format!(
            "numerical relation decision digest was not derived from every raw hypothesis: \
             supplied {supplied_decision_sha256}, derived {}",
            derivation.decision_sha256
        )));
    }
    let supplied_rejection_sha256 = evidence_sha256(
        object,
        "rejection_decision_sha256",
        "numerical relation rejection decision chain",
    )?;
    if supplied_rejection_sha256 != derivation.rejection_decision_sha256 {
        return Err(RusticolError::integrity(
            "numerical relation rejection digest was not derived from every rejected raw hypothesis",
        ));
    }
    for (field, expected) in [
        (
            "numerical_candidate_count",
            derivation.numerical_candidate_count,
        ),
        (
            "verification_rejected_count",
            derivation.verification_rejected_count,
        ),
        (
            "tested_hypothesis_count",
            derivation.tested_hypothesis_count,
        ),
        (
            "rejected_hypothesis_count",
            derivation.rejected_hypothesis_count,
        ),
    ] {
        if evidence_usize(object, field, "numerical relation evidence counter")? != expected {
            return Err(RusticolError::integrity(format!(
                "numerical relation evidence {field} was not derived from raw observations"
            )));
        }
    }
    validate_candidate_index_claim(
        json_field(object, "candidate_index", "numerical relation evidence")?,
        &derivation,
    )?;
    let supplied_certificates =
        json_array(object, "certificates", "numerical relation certificates")?;
    let supplied_mappings = json_array(object, "mappings", "numerical relation mappings")?;
    let derived_certificates = derivation
        .relations
        .iter()
        .map(|relation| relation.certificate.clone())
        .collect::<Vec<_>>();
    let derived_mappings = derivation
        .relations
        .iter()
        .map(|relation| relation.mapping_json.clone())
        .collect::<Vec<_>>();
    if supplied_certificates != derived_certificates || supplied_mappings != derived_mappings {
        return Err(RusticolError::integrity(
            "numerical relation certificates or mappings were not derived from raw observations",
        ));
    }
    let relation_set = serde_json::json!({
        "abi": RELATION_SET_ABI,
        "certificates": derived_certificates,
        "mappings": derived_mappings,
    });
    let actual_certificate_set_sha256 =
        canonical_json_sha256(&relation_set, "numerical relation certificate set")?;
    let certificate_set_sha256 = evidence_sha256(
        object,
        "certificate_set_sha256",
        "numerical relation certificate set",
    )?;
    if certificate_set_sha256 != actual_certificate_set_sha256 {
        return Err(RusticolError::integrity(
            "numerical relation certificate-set digest does not match native raw-evidence replay",
        ));
    }
    Ok(RecurrenceNumericalRelationEvidence {
        requested_mode,
        schedule_semantic_digest,
        baseline_runtime_layout_digest,
        source_semantics_sha256,
        runtime_parameter_schema_sha256,
        candidate_observation_batch_sha256: candidate.observation_batch_sha256,
        verification_observation_batch_sha256: verification.observation_batch_sha256,
        decision_sha256: derivation.decision_sha256,
        rejection_decision_sha256: derivation.rejection_decision_sha256,
        certificate_algorithm,
        certificate_set_sha256,
        precision_digits,
        probe_count,
        verification_probe_count,
        relative_tolerance,
        absolute_tolerance,
        seed,
        numerical_candidate_count: derivation.numerical_candidate_count,
        verification_rejected_count: derivation.verification_rejected_count,
        rejected_hypothesis_count: derivation.rejected_hypothesis_count,
        tested_hypothesis_count: derivation.tested_hypothesis_count,
        mappings: derivation
            .relations
            .into_iter()
            .map(|relation| relation.mapping)
            .collect(),
    })
}

fn parse_raw_source_semantics(
    value: &JsonValue,
    claimed_sha256: &str,
    schedule_semantic_digest: &str,
    baseline_runtime_layout_digest: &str,
    authenticated_process: &AuthenticatedRuntimeParameterContract,
) -> RusticolResult<RawSourceSemantics> {
    const ABI: &str = "pyamplicol-recurrence-numerical-current-source-v3";
    let object = json_object(value, "recurrence numerical source semantics")?;
    require_json_fields(
        object,
        &[
            "abi",
            "process_id",
            "strategy",
            "schedule_semantic_digest",
            "baseline_runtime_layout_digest",
            "selector_schedule",
            "currents",
        ],
        "recurrence numerical source semantics",
    )?;
    require_json_string_value(object, "abi", ABI, "recurrence numerical source semantics")?;
    if canonical_json_sha256(value, "recurrence numerical source semantics")? != claimed_sha256 {
        return Err(RusticolError::integrity(
            "recurrence numerical source-semantics digest does not match its raw payload",
        ));
    }
    require_json_string_value(
        object,
        "schedule_semantic_digest",
        schedule_semantic_digest,
        "recurrence numerical source semantics",
    )?;
    require_json_string_value(
        object,
        "baseline_runtime_layout_digest",
        baseline_runtime_layout_digest,
        "recurrence numerical source semantics",
    )?;
    require_json_string_value(
        object,
        "process_id",
        &authenticated_process.process_id,
        "recurrence numerical source semantics",
    )?;
    if authenticated_process.physical_pdgs.is_empty() {
        return Err(invalid(
            "authenticated recurrence process has no physical external particles",
        ));
    }
    let strategy =
        json_nonempty_string(object, "strategy", "recurrence numerical source semantics")?
            .to_owned();
    if !matches!(
        strategy.as_str(),
        "topology-replay" | "all-flow-union" | "contracted-color-union"
    ) {
        return Err(invalid("raw recurrence source has an unsupported strategy"));
    }
    let current_values = json_array(object, "currents", "recurrence numerical source currents")?;
    let mut currents = Vec::with_capacity(current_values.len());
    for (index, current_value) in current_values.iter().enumerate() {
        let context = format!("recurrence numerical source current {index}");
        let current = json_object(current_value, &context)?;
        require_json_fields(current, &["current_id", "is_source", "contract"], &context)?;
        let current_id = json_u32(current, "current_id", &context)?;
        if current_id as usize != index {
            return Err(invalid(format!(
                "{context} has non-canonical current ID {current_id}"
            )));
        }
        let is_source = json_field(current, "is_source", &context)?
            .as_bool()
            .ok_or_else(|| invalid(format!("{context} source flag is not boolean")))?;
        let contract = json_field(current, "contract", &context)?.clone();
        let contract_values = contract
            .as_array()
            .ok_or_else(|| invalid(format!("{context} contract is not an array")))?;
        if contract_values.len() != 9 {
            return Err(invalid(format!("{context} contract width is not nine")));
        }
        let dimension_u64 = contract_values[3]
            .as_u64()
            .ok_or_else(|| invalid(format!("{context} dimension is not unsigned")))?;
        let dimension = u16::try_from(dimension_u64)
            .map_err(|_| invalid(format!("{context} dimension exceeds u16")))?;
        if dimension == 0 {
            return Err(invalid(format!("{context} dimension is zero")));
        }
        currents.push(RawSourceCurrent {
            current_id,
            is_source,
            contract_key: canonical_json_bytes(&contract, &context)?,
            dimension,
        });
    }
    Ok(RawSourceSemantics {
        value: value.clone(),
        process_id: authenticated_process.process_id.clone(),
        physical_pdgs: authenticated_process.physical_pdgs.clone(),
        strategy,
        selector_schedule: json_field(
            object,
            "selector_schedule",
            "recurrence numerical source semantics",
        )?
        .clone(),
        currents,
    })
}

fn validate_candidate_index_claim(
    value: &JsonValue,
    derivation: &RawNumericalDerivation,
) -> RusticolResult<()> {
    const ALGORITHM: &str = "complete-contract-anchor-tolerance-window-v1";
    const MAX_SCREENED_HYPOTHESES: usize = 1_000_000;
    let object = json_object(value, "raw numerical candidate index")?;
    require_json_fields(
        object,
        &[
            "algorithm",
            "completeness",
            "contract_count",
            "exhaustive_fallback_contract_count",
            "theoretical_pair_hypothesis_count",
            "screened_pair_hypothesis_count",
            "zero_hypothesis_count",
            "screened_hypothesis_budget",
            "budget_classification",
            "nearest_rejected_scope",
        ],
        "raw numerical candidate index",
    )?;
    require_json_string_value(
        object,
        "algorithm",
        ALGORITHM,
        "raw numerical candidate index",
    )?;
    require_json_string_value(
        object,
        "completeness",
        "complete-within-configured-tolerance",
        "raw numerical candidate index",
    )?;
    require_json_string_value(
        object,
        "budget_classification",
        "within-authenticated-budget",
        "raw numerical candidate index",
    )?;
    require_json_string_value(
        object,
        "nearest_rejected_scope",
        "zero-and-tolerance-window-screened-hypotheses",
        "raw numerical candidate index",
    )?;
    for (field, expected) in [
        ("contract_count", derivation.candidate_index_contract_count),
        (
            "exhaustive_fallback_contract_count",
            derivation.exhaustive_fallback_contract_count,
        ),
        (
            "theoretical_pair_hypothesis_count",
            derivation.theoretical_pair_hypothesis_count,
        ),
        (
            "screened_pair_hypothesis_count",
            derivation.screened_pair_hypothesis_count,
        ),
        ("zero_hypothesis_count", derivation.zero_hypothesis_count),
        ("screened_hypothesis_budget", MAX_SCREENED_HYPOTHESES),
    ] {
        if evidence_usize(object, field, "raw numerical candidate index")? != expected {
            return Err(RusticolError::integrity(format!(
                "raw numerical candidate index {field} was not independently derived"
            )));
        }
    }
    Ok(())
}

fn validate_raw_evidence_geometry(
    source: &RawSourceSemantics,
    probe_count: u32,
    verification_probe_count: u32,
    runtime_parameter_count: usize,
) -> RusticolResult<(usize, usize, usize)> {
    let row_count = source
        .currents
        .len()
        .checked_mul(2)
        .and_then(|count| count.checked_add(runtime_parameter_count))
        .ok_or_else(|| invalid("raw numerical row count overflows usize"))?;
    let component_count = source.currents.iter().try_fold(0_usize, |total, current| {
        total
            .checked_add(usize::from(current.dimension))
            .ok_or_else(|| invalid("raw numerical component count overflows usize"))
    })?;
    let point_count = usize::try_from(probe_count)
        .ok()
        .and_then(|candidate| {
            usize::try_from(verification_probe_count)
                .ok()
                .and_then(|verification| candidate.checked_add(verification))
        })
        .ok_or_else(|| invalid("raw numerical probe count overflows usize"))?;
    let observation_scalar_count = component_count
        .checked_mul(point_count)
        .and_then(|count| count.checked_mul(2))
        .ok_or_else(|| invalid("raw numerical scalar count overflows usize"))?;
    let parameter_scalar_count = runtime_parameter_count
        .checked_mul(point_count)
        .ok_or_else(|| invalid("raw parameter scalar count overflows usize"))?;
    let scalar_count = observation_scalar_count
        .checked_add(parameter_scalar_count)
        .ok_or_else(|| invalid("raw total scalar count overflows usize"))?;
    let wire_limit = raw_evidence_shape_wire_limit(scalar_count, row_count)?;
    Ok((scalar_count, row_count, wire_limit))
}

fn validate_raw_runtime_parameter_schema(
    value: &JsonValue,
    claimed_sha256: &str,
    expected: &AuthenticatedRuntimeParameterContract,
) -> RusticolResult<usize> {
    if value != &expected.schema || claimed_sha256 != expected.schema_sha256 {
        return Err(RusticolError::integrity(
            "recurrence runtime parameter schema/defaults do not match the authenticated native template input",
        ));
    }
    Ok(expected.slots.len())
}

#[allow(clippy::too_many_arguments)]
fn parse_raw_numerical_capture(
    value: &JsonValue,
    label: &str,
    domain: &str,
    precision_digits: u32,
    expected_point_count: u32,
    source: &RawSourceSemantics,
    source_semantics_sha256: &str,
    runtime_parameter_schema_sha256: &str,
    runtime_parameter_count: usize,
    seed: u64,
) -> RusticolResult<RawNumericalCapture> {
    const ABI: &str = "pyamplicol-recurrence-current-observation-capture-v2";
    const BATCH_ABI: &str = "pyamplicol-recurrence-current-observation-batch-v2";
    const PARAMETER_CONTEXT_ABI: &str = "pyamplicol-recurrence-parameter-context-v1";
    let context = format!("{label} raw numerical capture");
    let object = json_object(value, &context)?;
    require_json_fields(
        object,
        &[
            "abi",
            "precision_digits",
            "point_count",
            "point_sha256s",
            "kinematic_sha256s",
            "parameter_contexts",
            "parameter_context_sha256s",
            "context_sha256s",
            "points",
            "current_count",
            "runtime_parameter_schema_sha256",
            "source_semantics_sha256",
            "observation_batch_sha256",
            "capture_contract_sha256",
            "context_policy",
            "complete_current_components",
            "point_major",
            "current_dimensions_sha256",
            "evaluator",
            "kinematic_binary64",
            "selector_contexts",
            "current_dimensions",
            "observations",
        ],
        &context,
    )?;
    require_json_string_value(object, "abi", ABI, &context)?;
    if json_u32(object, "precision_digits", &context)? != precision_digits
        || json_u32(object, "point_count", &context)? != expected_point_count
        || expected_point_count < 2
    {
        return Err(invalid(format!("{context} probe geometry is inconsistent")));
    }
    require_json_string_value(
        object,
        "runtime_parameter_schema_sha256",
        runtime_parameter_schema_sha256,
        &context,
    )?;
    require_json_string_value(
        object,
        "source_semantics_sha256",
        source_semantics_sha256,
        &context,
    )?;
    require_json_string_value(
        object,
        "evaluator",
        "recurrence-direct-plan-decimal-symbolica-exact",
        &context,
    )?;
    if !json_bool(object, "complete_current_components", &context)?
        || !json_bool(object, "point_major", &context)?
    {
        return Err(invalid(format!(
            "{context} does not contain complete point-major current components"
        )));
    }
    let expected_policy = match source.strategy.as_str() {
        "topology-replay" => "seeded-replay-target-per-physical-point-v1",
        "all-flow-union" => "seeded-helicity-per-selector-domain-and-physical-point-v1",
        "contracted-color-union" => "fixed-contracted-source-schedule-v1",
        _ => return Err(invalid("raw recurrence source strategy is unsupported")),
    };
    require_json_string_value(object, "context_policy", expected_policy, &context)?;

    let point_values = json_array(object, "points", &context)?;
    let point_sha256s = parse_sha256_array(object, "point_sha256s", &context)?;
    let kinematic_values = json_array(object, "kinematic_binary64", &context)?;
    let kinematic_sha256s = parse_sha256_array(object, "kinematic_sha256s", &context)?;
    let selector_contexts = json_array(object, "selector_contexts", &context)?;
    let selector_context_sha256s = parse_sha256_array(object, "context_sha256s", &context)?;
    let parameter_values = json_array(object, "parameter_contexts", &context)?;
    let parameter_context_sha256s =
        parse_sha256_array(object, "parameter_context_sha256s", &context)?;
    let point_count = expected_point_count as usize;
    if [
        point_values.len(),
        point_sha256s.len(),
        kinematic_values.len(),
        kinematic_sha256s.len(),
        selector_contexts.len(),
        selector_context_sha256s.len(),
        parameter_values.len(),
        parameter_context_sha256s.len(),
    ]
    .into_iter()
    .any(|count| count != point_count)
    {
        return Err(invalid(format!(
            "{context} does not cover every ordered probe slot"
        )));
    }
    if point_sha256s.iter().collect::<BTreeSet<_>>().len() != point_count
        || kinematic_sha256s.iter().collect::<BTreeSet<_>>().len() != point_count
    {
        return Err(invalid(format!(
            "{context} physical probes are not distinct"
        )));
    }

    let mut parameter_contexts = Vec::with_capacity(point_count);
    for point_index in 0..point_count {
        let point_context = format!("{context} point {point_index}");
        let actual_point_sha256 = canonical_json_sha256(
            &point_values[point_index],
            &format!("{point_context} record"),
        )?;
        if point_sha256s[point_index] != actual_point_sha256 {
            return Err(RusticolError::integrity(format!(
                "{point_context} digest does not match its raw point"
            )));
        }
        validate_raw_kinematic(
            &point_values[point_index],
            &kinematic_values[point_index],
            source,
            domain_seed(seed, domain, point_index),
            &point_context,
        )?;
        let actual_kinematic_sha256 = canonical_json_sha256(
            &kinematic_values[point_index],
            &format!("{point_context} binary64 kinematics"),
        )?;
        if kinematic_sha256s[point_index] != actual_kinematic_sha256 {
            return Err(RusticolError::integrity(format!(
                "{point_context} kinematic digest does not match its raw binary64 values"
            )));
        }
        let expected_selector = expected_selector_context(source, seed, domain, point_index)?;
        if selector_contexts[point_index] != expected_selector
            || selector_context_sha256s[point_index]
                != canonical_json_sha256(&expected_selector, &format!("{point_context} selector"))?
        {
            return Err(RusticolError::integrity(format!(
                "{point_context} selector context is stale or non-deterministic"
            )));
        }
        let raw_parameters = parameter_values[point_index]
            .as_array()
            .ok_or_else(|| invalid(format!("{point_context} parameters are not an array")))?;
        if raw_parameters.len() != runtime_parameter_count {
            return Err(invalid(format!(
                "{point_context} parameters do not cover every runtime slot"
            )));
        }
        let parsed_parameters = raw_parameters
            .iter()
            .enumerate()
            .map(|(slot, value)| {
                let text = value.as_str().ok_or_else(|| {
                    invalid(format!("{point_context} parameter {slot} is not a string"))
                })?;
                parse_canonical_decimal(text, &format!("{point_context} parameter {slot}"))
            })
            .collect::<RusticolResult<Vec<_>>>()?;
        let parameter_payload = serde_json::json!({
            "abi": PARAMETER_CONTEXT_ABI,
            "domain": domain,
            "point_index": point_index,
            "point_sha256": point_sha256s[point_index],
            "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
            "values": raw_parameters,
        });
        if parameter_context_sha256s[point_index]
            != canonical_json_sha256(
                &parameter_payload,
                &format!("{point_context} parameter context"),
            )?
        {
            return Err(RusticolError::integrity(format!(
                "{point_context} parameter-context digest does not match its raw values"
            )));
        }
        parameter_contexts.push(parsed_parameters);
    }

    let dimension_values = json_array(object, "current_dimensions", &context)?;
    let observation_values = json_array(object, "observations", &context)?;
    if dimension_values.len() != source.currents.len()
        || observation_values.len() != source.currents.len()
        || json_u32(object, "current_count", &context)? as usize != source.currents.len()
    {
        return Err(invalid(format!(
            "{context} current census does not match source semantics"
        )));
    }
    let mut dimensions = BTreeMap::new();
    let mut observations = BTreeMap::new();
    let mut canonical_current_rows = Vec::with_capacity(source.currents.len());
    for (index, source_current) in source.currents.iter().enumerate() {
        let row_context = format!("{context} current {index}");
        let dimension_row = json_object(&dimension_values[index], &row_context)?;
        require_json_fields(dimension_row, &["current_id", "dimension"], &row_context)?;
        let observation_row = json_object(&observation_values[index], &row_context)?;
        require_json_fields(
            observation_row,
            &["current_id", "dimension", "values"],
            &row_context,
        )?;
        let current_id = json_u32(dimension_row, "current_id", &row_context)?;
        let dimension = json_u32(dimension_row, "dimension", &row_context)?;
        if current_id != source_current.current_id
            || json_u32(observation_row, "current_id", &row_context)? != current_id
            || dimension != u32::from(source_current.dimension)
            || json_u32(observation_row, "dimension", &row_context)? != dimension
        {
            return Err(invalid(format!(
                "{row_context} dimension or ID disagrees with source semantics"
            )));
        }
        let raw_values = json_array(observation_row, "values", &row_context)?;
        let expected_width = point_count
            .checked_mul(dimension as usize)
            .ok_or_else(|| invalid(format!("{row_context} width overflows usize")))?;
        if raw_values.len() != expected_width {
            return Err(invalid(format!(
                "{row_context} observation width is {}, expected {expected_width}",
                raw_values.len()
            )));
        }
        let mut parsed_values = Vec::with_capacity(expected_width);
        for (component, raw_value) in raw_values.iter().enumerate() {
            let pair = raw_value
                .as_array()
                .filter(|pair| pair.len() == 2)
                .ok_or_else(|| {
                    invalid(format!(
                        "{row_context} observation {component} is not a complex pair"
                    ))
                })?;
            let real_text = pair[0].as_str().ok_or_else(|| {
                invalid(format!(
                    "{row_context} observation {component} real part is not a string"
                ))
            })?;
            let imaginary_text = pair[1].as_str().ok_or_else(|| {
                invalid(format!(
                    "{row_context} observation {component} imaginary part is not a string"
                ))
            })?;
            parsed_values.push(RawProbeValue {
                real_text: real_text.to_owned(),
                imaginary_text: imaginary_text.to_owned(),
                value: (
                    parse_canonical_decimal(
                        real_text,
                        &format!("{row_context} observation {component} real part"),
                    )?,
                    parse_canonical_decimal(
                        imaginary_text,
                        &format!("{row_context} observation {component} imaginary part"),
                    )?,
                ),
            });
        }
        dimensions.insert(current_id, source_current.dimension);
        observations.insert(current_id, parsed_values);
        canonical_current_rows.push(observation_values[index].clone());
    }
    let dimensions_object = dimensions
        .iter()
        .map(|(current_id, dimension)| {
            (
                current_id.to_string(),
                JsonValue::from(u64::from(*dimension)),
            )
        })
        .collect::<JsonMap<_, _>>();
    if evidence_sha256(object, "current_dimensions_sha256", &context)?
        != canonical_json_sha256(
            &JsonValue::Object(dimensions_object),
            &format!("{context} dimensions"),
        )?
    {
        return Err(RusticolError::integrity(format!(
            "{context} dimension digest does not match the raw census"
        )));
    }
    let observation_batch_sha256 = evidence_sha256(object, "observation_batch_sha256", &context)?;
    let batch_payload = serde_json::json!({
        "abi": BATCH_ABI,
        "source_semantics_sha256": source_semantics_sha256,
        "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
        "point_sha256s": point_sha256s,
        "selector_context_sha256s": selector_context_sha256s,
        "parameter_context_sha256s": parameter_context_sha256s,
        "currents": canonical_current_rows,
    });
    if observation_batch_sha256
        != canonical_json_sha256(&batch_payload, &format!("{context} observation batch"))?
    {
        return Err(RusticolError::integrity(format!(
            "{context} observation-batch digest does not match raw observations"
        )));
    }
    let capture_contract_sha256 = evidence_sha256(object, "capture_contract_sha256", &context)?;
    let capture_payload = serde_json::json!({
        "abi": ABI,
        "precision_digits": precision_digits,
        "source_semantics_sha256": source_semantics_sha256,
        "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
        "point_sha256s": point_sha256s,
        "kinematic_sha256s": kinematic_sha256s,
        "selector_context_sha256s": selector_context_sha256s,
        "parameter_context_sha256s": parameter_context_sha256s,
        "current_dimensions": dimension_values,
        "observation_batch_sha256": observation_batch_sha256,
        "context_policy": expected_policy,
    });
    if capture_contract_sha256
        != canonical_json_sha256(&capture_payload, &format!("{context} contract"))?
    {
        return Err(RusticolError::integrity(format!(
            "{context} contract digest does not match raw capture"
        )));
    }
    Ok(RawNumericalCapture {
        point_sha256s,
        kinematic_sha256s,
        parameter_context_sha256s,
        parameter_contexts,
        observations,
        dimensions,
        observation_batch_sha256,
        capture_contract_sha256,
    })
}

fn parse_sha256_array(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<Vec<String>> {
    json_array(object, field, context)?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let text = value
                .as_str()
                .ok_or_else(|| invalid(format!("{context} {field}[{index}] is not a string")))?;
            semantic_digest_from_hex(text, &format!("{context} {field}[{index}]"))?;
            Ok(text.to_owned())
        })
        .collect()
}

fn validate_raw_kinematic(
    point: &JsonValue,
    kinematic: &JsonValue,
    source: &RawSourceSemantics,
    expected_seed: u64,
    context: &str,
) -> RusticolResult<()> {
    let point_object = json_object(point, &format!("{context} point record"))?;
    require_json_fields(
        point_object,
        &[
            "schema_version",
            "kind",
            "process_id",
            "process",
            "seed",
            "available",
            "error",
            "points",
        ],
        &format!("{context} point record"),
    )?;
    if json_u32(point_object, "schema_version", context)? != 1
        || json_string(point_object, "kind", context)? != "pyamplicol-rusticol-validation-momenta"
        || !json_bool(point_object, "available", context)?
        || !json_field(point_object, "error", context)?.is_null()
    {
        return Err(invalid(format!(
            "{context} validation point is unavailable"
        )));
    }
    require_json_string_value(point_object, "process_id", &source.process_id, context)?;
    json_nonempty_string(point_object, "process", context)?;
    if json_u64(point_object, "seed", context)? != expected_seed {
        return Err(RusticolError::integrity(format!(
            "{context} validation-point seed is stale"
        )));
    }
    let point_sets = json_array(point_object, "points", context)?;
    if point_sets.len() != 1 {
        return Err(invalid(format!(
            "{context} validation point must contain one momentum set"
        )));
    }
    let particles = point_sets[0]
        .as_array()
        .ok_or_else(|| invalid(format!("{context} particle set is not an array")))?;
    let kinematic_rows = kinematic
        .as_array()
        .ok_or_else(|| invalid(format!("{context} kinematics are not an array")))?;
    if particles.len() != kinematic_rows.len()
        || particles.len() != source.physical_pdgs.len()
        || particles.is_empty()
    {
        return Err(invalid(format!(
            "{context} point, process, and binary64 kinematic widths disagree"
        )));
    }
    for (particle_index, ((particle, raw_row), expected_pdg)) in particles
        .iter()
        .zip(kinematic_rows)
        .zip(&source.physical_pdgs)
        .enumerate()
    {
        let particle_context = format!("{context} particle {particle_index}");
        let particle = json_object(particle, &particle_context)?;
        require_json_fields(particle, &["pdg", "momentum"], &particle_context)?;
        let actual_pdg = particle
            .get("pdg")
            .and_then(JsonValue::as_i64)
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| invalid(format!("{particle_context} PDG is not an i32")))?;
        if actual_pdg != *expected_pdg {
            return Err(RusticolError::integrity(format!(
                "{particle_context} PDG does not match the authenticated recurrence process"
            )));
        }
        let momentum = json_array(particle, "momentum", &particle_context)?;
        let binary = raw_row
            .as_array()
            .filter(|row| row.len() == 4)
            .ok_or_else(|| {
                invalid(format!(
                    "{particle_context} binary64 momentum is not four-wide"
                ))
            })?;
        if momentum.len() != 4 {
            return Err(invalid(format!(
                "{particle_context} decimal momentum is not four-wide"
            )));
        }
        for component in 0..4 {
            let decimal = momentum[component].as_str().ok_or_else(|| {
                invalid(format!(
                    "{particle_context} momentum component {component} is not a string"
                ))
            })?;
            parse_canonical_decimal(
                decimal,
                &format!("{particle_context} momentum component {component}"),
            )?;
            let binary_value = decimal.parse::<f64>().map_err(|_| {
                invalid(format!(
                    "{particle_context} momentum component {component} is not binary64"
                ))
            })?;
            if !binary_value.is_finite()
                || binary[component].as_str() != Some(python_f64_hex_signed(binary_value).as_str())
            {
                return Err(RusticolError::integrity(format!(
                    "{particle_context} decimal and binary64 momentum disagree at component {component}"
                )));
            }
        }
    }
    Ok(())
}

fn expected_selector_context(
    source: &RawSourceSemantics,
    seed: u64,
    domain: &str,
    point_index: usize,
) -> RusticolResult<JsonValue> {
    let selector_seed = domain_seed(seed, domain, point_index);
    let schedule = json_object(
        &source.selector_schedule,
        "recurrence numerical selector schedule",
    )?;
    match source.strategy.as_str() {
        "topology-replay" => {
            require_json_fields(
                schedule,
                &["policy", "replay_targets"],
                "recurrence topology selector schedule",
            )?;
            require_json_string_value(
                schedule,
                "policy",
                "seeded-replay-target-per-physical-point-v1",
                "recurrence topology selector schedule",
            )?;
            let targets = json_array(
                schedule,
                "replay_targets",
                "recurrence topology selector schedule",
            )?;
            if targets.is_empty() {
                return Err(invalid("recurrence topology selector schedule is empty"));
            }
            let target_index = selector_seed as usize % targets.len();
            let target = json_object(&targets[target_index], "recurrence replay target")?;
            require_json_fields(
                target,
                &[
                    "target_index",
                    "public_flow_id",
                    "representative_id",
                    "selector_domain_id",
                ],
                "recurrence replay target",
            )?;
            if json_u32(target, "target_index", "recurrence replay target")? as usize
                != target_index
            {
                return Err(invalid("recurrence replay target index is not canonical"));
            }
            Ok(serde_json::json!({
                "strategy": source.strategy,
                "target_index": target_index,
                "public_flow_id": json_u32(target, "public_flow_id", "recurrence replay target")?,
                "representative_id": json_u32(target, "representative_id", "recurrence replay target")?,
                "selector_domain_id": json_u32(target, "selector_domain_id", "recurrence replay target")?,
            }))
        }
        "all-flow-union" => {
            require_json_fields(
                schedule,
                &["policy", "resolved_helicities"],
                "recurrence union selector schedule",
            )?;
            require_json_string_value(
                schedule,
                "policy",
                "seeded-helicity-per-selector-domain-and-physical-point-v1",
                "recurrence union selector schedule",
            )?;
            let helicities = json_array(
                schedule,
                "resolved_helicities",
                "recurrence union selector schedule",
            )?;
            let mut by_domain = BTreeMap::<u32, Vec<(usize, u32)>>::new();
            for (index, value) in helicities.iter().enumerate() {
                let helicity = json_object(value, "recurrence resolved helicity")?;
                require_json_fields(
                    helicity,
                    &["helicity_index", "helicity_id", "selector_domain_id"],
                    "recurrence resolved helicity",
                )?;
                if json_u32(helicity, "helicity_index", "recurrence resolved helicity")? as usize
                    != index
                {
                    return Err(invalid(
                        "recurrence resolved-helicity index is not canonical",
                    ));
                }
                by_domain
                    .entry(json_u32(
                        helicity,
                        "selector_domain_id",
                        "recurrence resolved helicity",
                    )?)
                    .or_default()
                    .push((
                        index,
                        json_u32(helicity, "helicity_id", "recurrence resolved helicity")?,
                    ));
            }
            if by_domain.is_empty() {
                return Err(invalid("recurrence union selector schedule is empty"));
            }
            let selected = by_domain
                .into_iter()
                .map(|(selector_domain_id, rows)| {
                    let (helicity_index, helicity_id) = rows[selector_seed as usize % rows.len()];
                    serde_json::json!({
                        "selector_domain_id": selector_domain_id,
                        "helicity_index": helicity_index,
                        "helicity_id": helicity_id,
                    })
                })
                .collect::<Vec<_>>();
            Ok(serde_json::json!({
                "strategy": source.strategy,
                "selector_domain_helicities": selected,
            }))
        }
        "contracted-color-union" => {
            require_json_fields(
                schedule,
                &["policy", "fixed_source_schedule"],
                "recurrence contracted selector schedule",
            )?;
            require_json_string_value(
                schedule,
                "policy",
                "fixed-contracted-source-schedule-v1",
                "recurrence contracted selector schedule",
            )?;
            if !json_bool(
                schedule,
                "fixed_source_schedule",
                "recurrence contracted selector schedule",
            )? {
                return Err(invalid(
                    "recurrence contracted source schedule is not fixed",
                ));
            }
            Ok(serde_json::json!({
                "strategy": source.strategy,
                "fixed_source_schedule": true,
            }))
        }
        _ => Err(invalid("recurrence selector strategy is unsupported")),
    }
}

fn validate_independent_raw_captures(
    candidate: &RawNumericalCapture,
    verification: &RawNumericalCapture,
    runtime_parameters: &AuthenticatedRuntimeParameterContract,
    seed: u64,
) -> RusticolResult<()> {
    if candidate.dimensions != verification.dimensions
        || candidate
            .observations
            .keys()
            .ne(verification.observations.keys())
        || !sets_are_disjoint(&candidate.point_sha256s, &verification.point_sha256s)
        || !sets_are_disjoint(
            &candidate.kinematic_sha256s,
            &verification.kinematic_sha256s,
        )
        || !sets_are_disjoint(
            &candidate.parameter_context_sha256s,
            &verification.parameter_context_sha256s,
        )
    {
        return Err(RusticolError::integrity(
            "candidate and verification raw captures are not independent",
        ));
    }
    let expected_candidate = deterministic_parameter_contexts(
        &runtime_parameters.slots,
        candidate.parameter_contexts.len(),
        seed,
        "candidate-current-parameter-probes-v1",
        true,
    )?;
    let expected_verification = deterministic_parameter_contexts(
        &runtime_parameters.slots,
        verification.parameter_contexts.len(),
        seed,
        "independent-verification-current-parameter-probes-v1",
        false,
    )?;
    if candidate.parameter_contexts != expected_candidate
        || verification.parameter_contexts != expected_verification
    {
        return Err(RusticolError::integrity(
            "raw recurrence parameter contexts are stale or non-deterministic",
        ));
    }
    Ok(())
}

fn sets_are_disjoint(left: &[String], right: &[String]) -> bool {
    let left = left.iter().collect::<BTreeSet<_>>();
    right.iter().all(|value| !left.contains(value))
}

fn deterministic_parameter_contexts(
    slots: &[AuthenticatedRuntimeParameterSlot],
    count: usize,
    seed: u64,
    domain: &str,
    include_defaults: bool,
) -> RusticolResult<Vec<Vec<BigRational>>> {
    (0..count)
        .map(|probe_index| {
            if include_defaults && probe_index == 0 {
                return Ok(slots.iter().map(|slot| slot.default.clone()).collect());
            }
            slots
                .iter()
                .enumerate()
                .map(|(parameter_index, slot)| {
                    if slot.probe_policy == "derived-overwritten-fixed-zero-v1" {
                        return Ok(slot.default.clone());
                    }
                    let digest = Sha256::digest(
                        format!("{seed}:{domain}:{probe_index}:{parameter_index}").as_bytes(),
                    );
                    let unsigned = u64::from_be_bytes(
                        digest[..8]
                            .try_into()
                            .map_err(|_| invalid("parameter digest width is not eight"))?,
                    );
                    let signed = i128::from(unsigned) - (1_i128 << 63);
                    let signed = if signed == 0 { 1 } else { signed };
                    let scale = slot.default.abs().max(BigRational::one());
                    Ok(&slot.default
                        + scale * BigRational::new(BigInt::from(signed), BigInt::one() << 67_usize))
                })
                .collect()
        })
        .collect()
}

fn domain_seed(seed: u64, domain: &str, index: usize) -> u64 {
    let digest = Sha256::digest(format!("{seed}:{domain}:{index}").as_bytes());
    u64::from_be_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 prefix is eight bytes"),
    )
}

fn parse_canonical_decimal(value: &str, context: &str) -> RusticolResult<BigRational> {
    const MAX_DECIMAL_BYTES: usize = 16_384;
    if value.is_empty()
        || value.len() > MAX_DECIMAL_BYTES
        || value.starts_with('+')
        || value == "-0"
        || value.contains(['e', 'E'])
    {
        return Err(invalid(format!(
            "{context} is not a canonical finite decimal"
        )));
    }
    let (negative, unsigned) = value
        .strip_prefix('-')
        .map_or((false, value), |rest| (true, rest));
    let (integer, fraction) = unsigned
        .split_once('.')
        .map_or((unsigned, None), |(integer, fraction)| {
            (integer, Some(fraction))
        });
    if integer.is_empty()
        || !integer.bytes().all(|byte| byte.is_ascii_digit())
        || integer.len() > 1 && integer.starts_with('0')
        || fraction.is_some_and(|fraction| {
            fraction.is_empty()
                || !fraction.bytes().all(|byte| byte.is_ascii_digit())
                || fraction.ends_with('0')
        })
    {
        return Err(invalid(format!(
            "{context} is not a canonical finite decimal"
        )));
    }
    let fraction = fraction.unwrap_or("");
    if negative && integer == "0" && fraction.bytes().all(|byte| byte == b'0') {
        return Err(invalid(format!("{context} is negative zero")));
    }
    let digits = format!("{integer}{fraction}");
    let mut numerator = BigInt::from_str(&digits)
        .map_err(|_| invalid(format!("{context} decimal numerator is invalid")))?;
    if negative {
        numerator = -numerator;
    }
    let denominator = BigInt::from(10_u8).pow(
        u32::try_from(fraction.len())
            .map_err(|_| invalid(format!("{context} decimal scale exceeds u32")))?,
    );
    Ok(BigRational::new(numerator, denominator))
}

fn exact_binary64_rational(value: f64) -> RusticolResult<BigRational> {
    if !value.is_finite() {
        return Err(invalid("binary64 value is not finite"));
    }
    if value == 0.0 {
        return Ok(BigRational::zero());
    }
    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, exponent) = if exponent_bits == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, exponent_bits - 1023 - 52)
    };
    let mut numerator = BigInt::from(significand);
    let mut denominator = BigInt::one();
    if exponent >= 0 {
        numerator <<= exponent as usize;
    } else {
        denominator <<= (-exponent) as usize;
    }
    if negative {
        numerator = -numerator;
    }
    Ok(BigRational::new(numerator, denominator))
}

fn parse_nonnegative_python_f64_hex(
    text: &str,
    context: &str,
) -> RusticolResult<(f64, BigRational)> {
    if text.starts_with('-') || text.len() > 24 {
        return Err(invalid(format!(
            "{context} is not a canonical nonnegative finite binary64 encoding"
        )));
    }
    let (mantissa, exponent_text) = text.split_once('p').ok_or_else(|| {
        invalid(format!(
            "{context} is not a canonical nonnegative finite binary64 encoding"
        ))
    })?;
    let (leading, fraction_text) = mantissa
        .strip_prefix("0x")
        .and_then(|value| value.split_once('.'))
        .ok_or_else(|| {
            invalid(format!(
                "{context} is not a canonical nonnegative finite binary64 encoding"
            ))
        })?;
    if fraction_text.is_empty()
        || fraction_text.len() > 13
        || !fraction_text.bytes().all(|byte| byte.is_ascii_hexdigit())
        || !matches!(leading, "0" | "1")
    {
        return Err(invalid(format!(
            "{context} is not a canonical nonnegative finite binary64 encoding"
        )));
    }
    let exponent = i32::from_str(exponent_text).map_err(|_| {
        invalid(format!(
            "{context} is not a canonical nonnegative finite binary64 encoding"
        ))
    })?;
    let parsed_fraction = u64::from_str_radix(fraction_text, 16).map_err(|_| {
        invalid(format!(
            "{context} is not a canonical nonnegative finite binary64 encoding"
        ))
    })?;
    let padding = 4 * (13 - fraction_text.len());
    let fraction = parsed_fraction
        .checked_shl(
            u32::try_from(padding)
                .map_err(|_| invalid(format!("{context} fraction padding is invalid")))?,
        )
        .ok_or_else(|| invalid(format!("{context} fraction is too wide")))?;
    let bits = match leading {
        "1" if (-1022..=1023).contains(&exponent) => {
            let exponent_bits = u64::try_from(exponent + 1023)
                .map_err(|_| invalid(format!("{context} exponent is invalid")))?;
            (exponent_bits << 52) | fraction
        }
        "0" if exponent == -1022 && fraction != 0 => fraction,
        "0" if exponent == 0 && fraction == 0 => 0,
        _ => {
            return Err(invalid(format!(
                "{context} is not a canonical nonnegative finite binary64 encoding"
            )));
        }
    };
    let value = f64::from_bits(bits);
    if !value.is_finite() || python_f64_hex(value) != text {
        return Err(invalid(format!(
            "{context} is not a canonical nonnegative finite binary64 encoding"
        )));
    }
    let exact = exact_binary64_rational(value)?;
    Ok((value, exact))
}

fn python_f64_hex_signed(value: f64) -> String {
    let sign = if value.is_sign_negative() { "-" } else { "" };
    format!("{sign}{}", python_f64_hex(value.abs()))
}

fn raw_probe_scalar(value: &RawProbeValue, scalar_component: usize) -> &BigRational {
    if scalar_component == 0 {
        &value.value.0
    } else {
        &value.value.1
    }
}

fn build_raw_numerical_candidate_indexes(
    source: &RawSourceSemantics,
    candidate: &RawNumericalCapture,
) -> RusticolResult<BTreeMap<Vec<u8>, RawNumericalCandidateIndex>> {
    let mut members_by_contract = BTreeMap::<Vec<u8>, Vec<u32>>::new();
    for current in &source.currents {
        members_by_contract
            .entry(current.contract_key.clone())
            .or_default()
            .push(current.current_id);
    }
    members_by_contract
        .into_iter()
        .map(|(contract, members)| {
            let first = members
                .first()
                .and_then(|current_id| candidate.observations.get(current_id))
                .ok_or_else(|| invalid("raw numerical candidate index has no members"))?;
            if first.is_empty()
                || members.iter().any(|current_id| {
                    candidate
                        .observations
                        .get(current_id)
                        .is_none_or(|values| values.len() != first.len())
                })
            {
                return Err(invalid(
                    "raw numerical candidate index has inconsistent widths",
                ));
            }
            let mut best_choice = (0_usize, 0_usize);
            let mut best_score = (0_usize, 0_usize);
            for observation_index in 0..first.len() {
                for scalar_component in 0..2 {
                    let mut unique = BTreeSet::<BigRational>::new();
                    let mut nonzero = 0_usize;
                    for current_id in &members {
                        let value = candidate
                            .observations
                            .get(current_id)
                            .and_then(|values| values.get(observation_index))
                            .ok_or_else(|| {
                                invalid("raw numerical candidate index value is absent")
                            })?;
                        let scalar = raw_probe_scalar(value, scalar_component);
                        unique.insert(scalar.clone());
                        nonzero += usize::from(!scalar.is_zero());
                    }
                    let score = (unique.len(), nonzero);
                    if score > best_score
                        || score == best_score
                            && (observation_index, scalar_component) < best_choice
                    {
                        best_score = score;
                        best_choice = (observation_index, scalar_component);
                    }
                }
            }
            let mut entries = members
                .iter()
                .map(|current_id| {
                    let value = candidate
                        .observations
                        .get(current_id)
                        .and_then(|values| values.get(best_choice.0))
                        .ok_or_else(|| invalid("raw numerical candidate index value is absent"))?;
                    Ok((raw_probe_scalar(value, best_choice.1).clone(), *current_id))
                })
                .collect::<RusticolResult<Vec<_>>>()?;
            entries.sort();
            Ok((
                contract,
                RawNumericalCandidateIndex {
                    observation_index: best_choice.0,
                    scalar_component: best_choice.1,
                    entries,
                },
            ))
        })
        .collect()
}

fn raw_numerical_tolerance_window_ids(
    index: &RawNumericalCandidateIndex,
    current_values: &[RawProbeValue],
    current_id: u32,
    relation_kind: &str,
    relative_tolerance: &BigRational,
    absolute_tolerance: &BigRational,
) -> RusticolResult<BTreeSet<u32>> {
    if !matches!(relation_kind, "equal" | "opposite") {
        return Err(invalid(
            "raw numerical candidate index relation is unsupported",
        ));
    }
    let selected = current_values
        .get(index.observation_index)
        .ok_or_else(|| invalid("raw numerical candidate index observation is absent"))?;
    if relative_tolerance >= &BigRational::one() {
        return Ok(index
            .entries
            .iter()
            .filter_map(|(_value, representative_id)| {
                (*representative_id < current_id).then_some(*representative_id)
            })
            .collect());
    }
    let selected_scalar = raw_probe_scalar(selected, index.scalar_component);
    let target = if relation_kind == "equal" {
        selected_scalar.clone()
    } else {
        -selected_scalar
    };
    // For complex-pair infinity norms X (current), Y (representative), and D
    // (residual), passing gives D <= a+r*max(X,Y) and reverse triangle gives
    // Y <= X+D.  Hence r<1 implies D <= (a+rX)/(1-r); the indexed scalar
    // residual is <= D, so this window cannot discard a passing relation.
    let current_scale = selected.value.0.abs().max(selected.value.1.abs());
    let radius = (absolute_tolerance + relative_tolerance * current_scale)
        / (BigRational::one() - relative_tolerance);
    let lower_value = &target - &radius;
    let upper_value = target + radius;
    let lower = index
        .entries
        .partition_point(|(value, _current_id)| value < &lower_value);
    let upper = index
        .entries
        .partition_point(|(value, _current_id)| value <= &upper_value);
    Ok(index.entries[lower..upper]
        .iter()
        .filter_map(|(_value, representative_id)| {
            (*representative_id < current_id).then_some(*representative_id)
        })
        .collect())
}

#[allow(clippy::too_many_arguments)]
fn derive_raw_numerical_relations(
    source: &RawSourceSemantics,
    candidate: &RawNumericalCapture,
    verification: &RawNumericalCapture,
    precision_digits: u32,
    seed: u64,
    relative_tolerance: &BigRational,
    absolute_tolerance: &BigRational,
    relative_tolerance_hex: &str,
    absolute_tolerance_hex: &str,
    runtime_parameter_schema_sha256: &str,
    certificate_algorithm: &str,
) -> RusticolResult<RawNumericalDerivation> {
    const DECISION_CHAIN_ABI: &str = "pyamplicol-recurrence-numerical-decision-chain-v1";
    const REJECTION_CHAIN_ABI: &str = "pyamplicol-recurrence-rejected-numerical-decision-chain-v1";
    const MAX_SCREENED_HYPOTHESES: usize = 1_000_000;
    let candidate_indexes = build_raw_numerical_candidate_indexes(source, candidate)?;
    let mut prior_by_contract = BTreeMap::<Vec<u8>, Vec<u32>>::new();
    let mut derived = Vec::new();
    let mut numerical_candidate_count = 0_usize;
    let mut verification_rejected_count = 0_usize;
    let mut tested_hypothesis_count = 0_usize;
    let mut theoretical_pair_hypothesis_count = 0_usize;
    let mut screened_pair_hypothesis_count = 0_usize;
    let mut zero_hypothesis_count = 0_usize;
    let source_semantics_sha256 =
        canonical_json_sha256(&source.value, "raw numerical source semantics")?;
    let mut decision_chain = canonical_json_sha256(
        &serde_json::json!({
            "abi": DECISION_CHAIN_ABI,
            "source_semantics_sha256": source_semantics_sha256,
            "candidate_capture_sha256": candidate.capture_contract_sha256,
            "verification_capture_sha256": verification.capture_contract_sha256,
        }),
        "raw numerical decision-chain root",
    )?;
    let mut rejection_chain = canonical_json_sha256(
        &serde_json::json!({
            "abi": REJECTION_CHAIN_ABI,
            "source_semantics_sha256": source_semantics_sha256,
            "candidate_capture_sha256": candidate.capture_contract_sha256,
            "verification_capture_sha256": verification.capture_contract_sha256,
        }),
        "raw numerical rejection-chain root",
    )?;
    for current in &source.currents {
        let prior = prior_by_contract
            .entry(current.contract_key.clone())
            .or_default();
        if current.is_source {
            prior.push(current.current_id);
            continue;
        }
        let current_values = candidate
            .observations
            .get(&current.current_id)
            .ok_or_else(|| invalid("raw numerical current observations are absent"))?;
        let candidate_index = candidate_indexes
            .get(&current.contract_key)
            .ok_or_else(|| invalid("raw numerical candidate index contract is absent"))?;
        let equal_representatives = raw_numerical_tolerance_window_ids(
            candidate_index,
            current_values,
            current.current_id,
            "equal",
            relative_tolerance,
            absolute_tolerance,
        )?;
        let opposite_representatives = raw_numerical_tolerance_window_ids(
            candidate_index,
            current_values,
            current.current_id,
            "opposite",
            relative_tolerance,
            absolute_tolerance,
        )?;
        theoretical_pair_hypothesis_count = theoretical_pair_hypothesis_count
            .checked_add(
                prior
                    .len()
                    .checked_mul(2)
                    .ok_or_else(|| invalid("theoretical pair count overflows usize"))?,
            )
            .ok_or_else(|| invalid("theoretical pair count overflows usize"))?;
        screened_pair_hypothesis_count = screened_pair_hypothesis_count
            .checked_add(equal_representatives.len())
            .and_then(|count| count.checked_add(opposite_representatives.len()))
            .ok_or_else(|| invalid("screened pair count overflows usize"))?;
        zero_hypothesis_count = zero_hypothesis_count
            .checked_add(1)
            .ok_or_else(|| invalid("zero hypothesis count overflows usize"))?;
        if screened_pair_hypothesis_count
            .checked_add(zero_hypothesis_count)
            .is_none_or(|count| count > MAX_SCREENED_HYPOTHESES)
        {
            return Err(RusticolError::integrity(
                "raw numerical candidate index exceeds the authenticated screened-hypothesis budget",
            ));
        }
        let mut hypotheses = vec![("zero", None)];
        for representative_id in equal_representatives
            .union(&opposite_representatives)
            .copied()
        {
            if equal_representatives.contains(&representative_id) {
                hypotheses.push(("equal", Some(representative_id)));
            }
            if opposite_representatives.contains(&representative_id) {
                hypotheses.push(("opposite", Some(representative_id)));
            }
        }
        for (relation_kind, representative_id) in hypotheses {
            tested_hypothesis_count = tested_hypothesis_count
                .checked_add(1)
                .ok_or_else(|| invalid("numerical hypothesis count overflows usize"))?;
            let candidate_residuals = relation_residuals(
                relation_kind,
                current.current_id,
                representative_id,
                candidate,
                relative_tolerance,
                absolute_tolerance,
            )?;
            if candidate_residuals.maximum_ratio > BigRational::one() {
                decision_chain = advance_decision_chain(
                    &decision_chain,
                    current.current_id,
                    representative_id,
                    relation_kind,
                    &candidate_residuals,
                    None,
                    false,
                )?;
                rejection_chain = advance_rejection_chain(
                    &rejection_chain,
                    current.current_id,
                    representative_id,
                    relation_kind,
                    &candidate_residuals,
                    "candidate-observations-not-equal",
                )?;
                continue;
            }
            numerical_candidate_count = numerical_candidate_count
                .checked_add(1)
                .ok_or_else(|| invalid("numerical candidate count overflows usize"))?;
            let verification_residuals = relation_residuals(
                relation_kind,
                current.current_id,
                representative_id,
                verification,
                relative_tolerance,
                absolute_tolerance,
            )?;
            if verification_residuals.maximum_ratio > BigRational::one() {
                verification_rejected_count = verification_rejected_count
                    .checked_add(1)
                    .ok_or_else(|| invalid("verification rejection count overflows usize"))?;
                decision_chain = advance_decision_chain(
                    &decision_chain,
                    current.current_id,
                    representative_id,
                    relation_kind,
                    &candidate_residuals,
                    Some(&verification_residuals),
                    false,
                )?;
                rejection_chain = advance_rejection_chain(
                    &rejection_chain,
                    current.current_id,
                    representative_id,
                    relation_kind,
                    &verification_residuals,
                    "independent-verification-rejected-candidate",
                )?;
                continue;
            }
            let execution_representative_id = if relation_kind == "zero" {
                prior.first().copied().unwrap_or(current.current_id)
            } else {
                representative_id
                    .ok_or_else(|| invalid("non-zero numerical relation has no representative"))?
            };
            derived.push(build_derived_numerical_relation(
                source,
                current,
                representative_id,
                execution_representative_id,
                relation_kind,
                candidate,
                verification,
                &candidate_residuals,
                &verification_residuals,
                precision_digits,
                seed,
                relative_tolerance_hex,
                absolute_tolerance_hex,
                runtime_parameter_schema_sha256,
                certificate_algorithm,
            )?);
            decision_chain = advance_decision_chain(
                &decision_chain,
                current.current_id,
                representative_id,
                relation_kind,
                &candidate_residuals,
                Some(&verification_residuals),
                true,
            )?;
            break;
        }
        prior.push(current.current_id);
    }
    let rejected_hypothesis_count = tested_hypothesis_count
        .checked_sub(derived.len())
        .ok_or_else(|| invalid("raw numerical rejected-hypothesis count underflows"))?;
    let decision_sha256 = canonical_json_sha256(
        &serde_json::json!({
            "abi": DECISION_CHAIN_ABI,
            "chain_tail_sha256": decision_chain,
            "tested_hypothesis_count": tested_hypothesis_count,
            "theoretical_pair_hypothesis_count": theoretical_pair_hypothesis_count,
            "screened_pair_hypothesis_count": screened_pair_hypothesis_count,
            "zero_hypothesis_count": zero_hypothesis_count,
            "numerical_candidate_count": numerical_candidate_count,
            "verification_rejected_count": verification_rejected_count,
            "rejected_hypothesis_count": rejected_hypothesis_count,
            "certified_relation_count": derived.len(),
        }),
        "raw numerical decision-chain census",
    )?;
    let rejection_decision_sha256 = rejection_chain;
    Ok(RawNumericalDerivation {
        relations: derived,
        numerical_candidate_count,
        verification_rejected_count,
        rejected_hypothesis_count,
        tested_hypothesis_count,
        theoretical_pair_hypothesis_count,
        screened_pair_hypothesis_count,
        zero_hypothesis_count,
        candidate_index_contract_count: candidate_indexes.len(),
        exhaustive_fallback_contract_count: if relative_tolerance >= &BigRational::one() {
            candidate_indexes.len()
        } else {
            0
        },
        decision_sha256,
        rejection_decision_sha256,
    })
}

fn advance_decision_chain(
    previous: &str,
    current_id: u32,
    representative_id: Option<u32>,
    relation_kind: &str,
    candidate: &RelationResiduals,
    verification: Option<&RelationResiduals>,
    selected: bool,
) -> RusticolResult<String> {
    let row = serde_json::json!({
        "current_id": current_id,
        "representative_id": representative_id,
        "relation_kind": relation_kind,
        "candidate_maximum_absolute_residual": rational_string(&candidate.maximum_absolute),
        "candidate_maximum_relative_residual": rational_string(&candidate.maximum_relative),
        "candidate_maximum_tolerance_ratio": rational_string(&candidate.maximum_ratio),
        "candidate_accepted": candidate.maximum_ratio <= BigRational::one(),
        "verification_maximum_absolute_residual": verification.map(|value| rational_string(&value.maximum_absolute)),
        "verification_maximum_relative_residual": verification.map(|value| rational_string(&value.maximum_relative)),
        "verification_maximum_tolerance_ratio": verification.map(|value| rational_string(&value.maximum_ratio)),
        "verification_accepted": verification.map(|value| value.maximum_ratio <= BigRational::one()),
        "selected": selected,
    });
    let previous = semantic_digest_from_hex(previous, "raw numerical decision-chain tail")?;
    let mut digest = Sha256::new();
    digest.update(previous.as_bytes());
    digest.update(canonical_json_bytes(&row, "raw numerical decision row")?);
    Ok(hex_digest(digest.finalize()))
}

fn advance_rejection_chain(
    previous: &str,
    current_id: u32,
    representative_id: Option<u32>,
    relation_kind: &str,
    residuals: &RelationResiduals,
    reason: &str,
) -> RusticolResult<String> {
    let row = serde_json::json!({
        "current_id": current_id,
        "representative_id": representative_id,
        "relation_kind": relation_kind,
        "reason": reason,
        "maximum_absolute_residual": rational_string(&residuals.maximum_absolute),
        "maximum_relative_residual": rational_string(&residuals.maximum_relative),
        "maximum_tolerance_ratio": rational_string(&residuals.maximum_ratio),
    });
    let previous = semantic_digest_from_hex(previous, "raw numerical rejection-chain predecessor")?;
    let mut digest = Sha256::new();
    digest.update(previous.as_bytes());
    digest.update(canonical_json_bytes(
        &row,
        "raw numerical rejected hypothesis",
    )?);
    Ok(hex_digest(digest.finalize()))
}

fn relation_residuals(
    relation_kind: &str,
    current_id: u32,
    representative_id: Option<u32>,
    capture: &RawNumericalCapture,
    relative_tolerance: &BigRational,
    absolute_tolerance: &BigRational,
) -> RusticolResult<RelationResiduals> {
    let current = capture
        .observations
        .get(&current_id)
        .ok_or_else(|| invalid("raw numerical current observations are absent"))?;
    let representative = representative_id
        .map(|representative_id| {
            capture
                .observations
                .get(&representative_id)
                .ok_or_else(|| invalid("raw representative observations are absent"))
        })
        .transpose()?;
    if relation_kind != "zero"
        && representative.is_none_or(|representative| representative.len() != current.len())
    {
        return Err(invalid("raw numerical relation width is invalid"));
    }
    let mut maximum_absolute = BigRational::zero();
    let mut maximum_relative = BigRational::zero();
    let mut maximum_ratio = BigRational::zero();
    for (index, current_value) in current.iter().enumerate() {
        let zero = (BigRational::zero(), BigRational::zero());
        let representative_value = if relation_kind == "zero" {
            &zero
        } else {
            &representative
                .expect("checked non-zero representative")
                .get(index)
                .ok_or_else(|| invalid("raw representative observation is absent"))?
                .value
        };
        let (right_real, right_imaginary) = match relation_kind {
            "zero" | "equal" => (
                representative_value.0.clone(),
                representative_value.1.clone(),
            ),
            "opposite" => (
                -representative_value.0.clone(),
                -representative_value.1.clone(),
            ),
            _ => return Err(invalid("raw numerical relation kind is unsupported")),
        };
        let difference = (&current_value.value.0 - right_real)
            .abs()
            .max((&current_value.value.1 - right_imaginary).abs());
        let scale = current_value
            .value
            .0
            .abs()
            .max(current_value.value.1.abs())
            .max(representative_value.0.abs())
            .max(representative_value.1.abs());
        let allowed = absolute_tolerance + relative_tolerance * &scale;
        let (relative, ratio) = if difference.is_zero() {
            (BigRational::zero(), BigRational::zero())
        } else {
            if allowed <= BigRational::zero() {
                return Err(invalid("raw numerical tolerance is not positive"));
            }
            (
                &difference / scale.max(absolute_tolerance.clone()),
                &difference / allowed,
            )
        };
        maximum_absolute = maximum_absolute.max(difference);
        maximum_relative = maximum_relative.max(relative);
        maximum_ratio = maximum_ratio.max(ratio);
    }
    Ok(RelationResiduals {
        maximum_absolute,
        maximum_relative,
        maximum_ratio,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_derived_numerical_relation(
    source: &RawSourceSemantics,
    current: &RawSourceCurrent,
    representative_id: Option<u32>,
    execution_representative_id: u32,
    relation_kind: &str,
    candidate: &RawNumericalCapture,
    verification: &RawNumericalCapture,
    candidate_residuals: &RelationResiduals,
    verification_residuals: &RelationResiduals,
    precision_digits: u32,
    seed: u64,
    relative_tolerance_hex: &str,
    absolute_tolerance_hex: &str,
    runtime_parameter_schema_sha256: &str,
    certificate_algorithm: &str,
) -> RusticolResult<DerivedNumericalRelation> {
    let factor = match relation_kind {
        "equal" => rusticol_core::recurrence::ExactComplexRational::ONE,
        "opposite" => rusticol_core::recurrence::ExactComplexRational::ONE.checked_neg()?,
        "zero" => rusticol_core::recurrence::ExactComplexRational::ZERO,
        _ => return Err(invalid("raw numerical relation kind is unsupported")),
    };
    let factor_integer = match relation_kind {
        "equal" => serde_json::json!([1, 0]),
        "opposite" => serde_json::json!([-1, 0]),
        "zero" => serde_json::json!([0, 0]),
        _ => unreachable!(),
    };
    let candidate_observations_sha256 = relation_observation_sha256(
        current.current_id,
        representative_id,
        relation_kind,
        candidate,
    )?;
    let verification_observations_sha256 = relation_observation_sha256(
        current.current_id,
        representative_id,
        relation_kind,
        verification,
    )?;
    let probe_contract = serde_json::json!({
        "algorithm": certificate_algorithm,
        "source_semantics_sha256": canonical_json_sha256(
            &source.value,
            "raw numerical source semantics",
        )?,
        "current_id": current.current_id,
        "representative_id": representative_id,
        "execution_representative_id": execution_representative_id,
        "relation_kind": relation_kind,
        "precision_digits": precision_digits,
        "seed": seed,
        "candidate_domain": "candidate-current-probes-v1",
        "verification_domain": "independent-verification-current-probes-v1",
        "relative_tolerance_binary64": relative_tolerance_hex,
        "absolute_tolerance_binary64": absolute_tolerance_hex,
        "candidate_probe_count": candidate.point_sha256s.len(),
        "verification_probe_count": verification.point_sha256s.len(),
        "current_dimension": current.dimension,
        "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
        "candidate_capture_sha256": candidate.capture_contract_sha256,
        "verification_capture_sha256": verification.capture_contract_sha256,
        "candidate_observations_sha256": candidate_observations_sha256,
        "verification_observations_sha256": verification_observations_sha256,
    });
    let probe_contract_sha256 =
        canonical_json_sha256(&probe_contract, "raw numerical probe contract")?;
    let proof = serde_json::json!({
        "algorithm": certificate_algorithm,
        "source_semantics_sha256": probe_contract["source_semantics_sha256"],
        "current_id": current.current_id,
        "representative_id": representative_id,
        "execution_representative_id": execution_representative_id,
        "relation_kind": relation_kind,
        "precision_digits": precision_digits,
        "seed": seed,
        "candidate_domain": "candidate-current-probes-v1",
        "verification_domain": "independent-verification-current-probes-v1",
        "relative_tolerance_binary64": relative_tolerance_hex,
        "absolute_tolerance_binary64": absolute_tolerance_hex,
        "candidate_probe_count": candidate.point_sha256s.len(),
        "verification_probe_count": verification.point_sha256s.len(),
        "current_dimension": current.dimension,
        "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
        "candidate_capture_sha256": candidate.capture_contract_sha256,
        "verification_capture_sha256": verification.capture_contract_sha256,
        "candidate_observations_sha256": candidate_observations_sha256,
        "verification_observations_sha256": verification_observations_sha256,
        "proof_kind": "authenticated-numerical",
        "factor_integer": factor_integer,
        "candidate_maximum_absolute_residual": rational_string(&candidate_residuals.maximum_absolute),
        "candidate_maximum_relative_residual": rational_string(&candidate_residuals.maximum_relative),
        "candidate_maximum_tolerance_ratio": rational_string(&candidate_residuals.maximum_ratio),
        "verification_maximum_absolute_residual": rational_string(&verification_residuals.maximum_absolute),
        "verification_maximum_relative_residual": rational_string(&verification_residuals.maximum_relative),
        "verification_maximum_tolerance_ratio": rational_string(&verification_residuals.maximum_ratio),
        "probe_contract_sha256": probe_contract_sha256,
    });
    let proof_sha256 = canonical_json_sha256(&proof, "raw numerical relation proof")?;
    let certificate = serde_json::json!({
        "algorithm": certificate_algorithm,
        "proof_kind": "authenticated-numerical",
        "relation_kind": relation_kind,
        "current_id": current.current_id,
        "representative_id": representative_id,
        "execution_representative_id": execution_representative_id,
        "factor_integer": factor_integer,
        "source_semantics_sha256": probe_contract["source_semantics_sha256"],
        "precision_digits": precision_digits,
        "seed": seed,
        "relative_tolerance_binary64": relative_tolerance_hex,
        "absolute_tolerance_binary64": absolute_tolerance_hex,
        "candidate_probe_count": candidate.point_sha256s.len(),
        "verification_probe_count": verification.point_sha256s.len(),
        "current_dimension": current.dimension,
        "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
        "candidate_capture_sha256": candidate.capture_contract_sha256,
        "verification_capture_sha256": verification.capture_contract_sha256,
        "candidate_maximum_absolute_residual": rational_string(&candidate_residuals.maximum_absolute),
        "candidate_maximum_relative_residual": rational_string(&candidate_residuals.maximum_relative),
        "candidate_maximum_tolerance_ratio": rational_string(&candidate_residuals.maximum_ratio),
        "verification_maximum_absolute_residual": rational_string(&verification_residuals.maximum_absolute),
        "verification_maximum_relative_residual": rational_string(&verification_residuals.maximum_relative),
        "verification_maximum_tolerance_ratio": rational_string(&verification_residuals.maximum_ratio),
        "candidate_observations_sha256": candidate_observations_sha256,
        "verification_observations_sha256": verification_observations_sha256,
        "probe_contract_sha256": probe_contract_sha256,
        "proof_sha256": proof_sha256,
    });
    let mapping_json = serde_json::json!({
        "current_id": current.current_id,
        "representative_id": representative_id,
        "execution_representative_id": execution_representative_id,
        "relation_kind": relation_kind,
        "factor_integer": factor_integer,
        "current_dimension": current.dimension,
        "certificate_proof_sha256": proof_sha256,
        "candidate_observations_sha256": candidate_observations_sha256,
        "verification_observations_sha256": verification_observations_sha256,
    });
    Ok(DerivedNumericalRelation {
        mapping: RecurrenceNumericalCurrentMapping {
            current_id: current.current_id,
            representative_id,
            execution_representative_id,
            relation_kind: relation_kind.to_owned(),
            factor,
            current_dimension: current.dimension,
            certificate_proof_sha256: proof_sha256,
            candidate_observations_sha256,
            verification_observations_sha256,
        },
        certificate,
        mapping_json,
    })
}

fn relation_observation_sha256(
    current_id: u32,
    representative_id: Option<u32>,
    relation_kind: &str,
    capture: &RawNumericalCapture,
) -> RusticolResult<String> {
    let current_values = raw_probe_values_json(
        capture
            .observations
            .get(&current_id)
            .ok_or_else(|| invalid("raw relation current observations are absent"))?,
    );
    let representative_values = representative_id
        .map(|representative_id| {
            capture
                .observations
                .get(&representative_id)
                .map(|values| JsonValue::Array(raw_probe_values_json(values)))
                .ok_or_else(|| invalid("raw relation representative observations are absent"))
        })
        .transpose()?
        .unwrap_or(JsonValue::Null);
    canonical_json_sha256(
        &serde_json::json!({
            "abi": "pyamplicol-recurrence-relation-observation-v2",
            "capture_contract_sha256": capture.capture_contract_sha256,
            "current_id": current_id,
            "representative_id": representative_id,
            "relation_kind": relation_kind,
            "current_dimension": capture.dimensions[&current_id],
            "current_values": current_values,
            "representative_values": representative_values,
        }),
        "raw numerical relation observations",
    )
}

fn raw_probe_values_json(values: &[RawProbeValue]) -> Vec<JsonValue> {
    values
        .iter()
        .map(|value| serde_json::json!([value.real_text.as_str(), value.imaginary_text.as_str()]))
        .collect()
}

fn rational_string(value: &BigRational) -> String {
    if value.denom().is_one() {
        value.numer().to_string()
    } else {
        format!("{}/{}", value.numer(), value.denom())
    }
}

fn evidence_sha256(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<String> {
    let value = json_string(object, field, context)?;
    semantic_digest_from_hex(value, context)?;
    Ok(value.to_owned())
}

fn evidence_usize(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<usize> {
    json_field(object, field, context)?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid(format!("{context} must be a nonnegative usize")))
}

fn json_u64(
    object: &JsonMap<String, JsonValue>,
    field: &str,
    context: &str,
) -> RusticolResult<u64> {
    json_field(object, field, context)?
        .as_u64()
        .ok_or_else(|| invalid(format!("{context} must be a nonnegative u64")))
}

fn python_f64_hex(value: f64) -> String {
    let bits = value.to_bits();
    if bits & 0x7fff_ffff_ffff_ffff == 0 {
        return "0x0.0p+0".to_owned();
    }
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (leading, exponent) = if exponent_bits == 0 {
        (0, -1022)
    } else {
        (1, exponent_bits - 1023)
    };
    format!("0x{leading}.{fraction:013x}p{exponent:+}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusticol_core::recurrence::RecurrenceRelationDiscoveryOptions;
    use serde_json::json;

    fn digest(seed: u8) -> String {
        format!("{seed:02x}").repeat(32)
    }

    fn empty_runtime_parameter_contract() -> AuthenticatedRuntimeParameterContract {
        let schema = json!({
            "abi": "pyamplicol-recurrence-runtime-parameter-schema-v2",
            "parameters": [],
        });
        AuthenticatedRuntimeParameterContract {
            schema_sha256: canonical_json_sha256(&schema, "empty runtime parameter schema")
                .unwrap(),
            schema,
            slots: Vec::new(),
            process_id: "raw-v3-test".to_owned(),
            physical_pdgs: vec![1],
        }
    }

    fn parse_fixture(bytes: &[u8]) -> RusticolResult<RecurrenceNumericalRelationEvidence> {
        parse_numerical_relation_evidence(bytes, &empty_runtime_parameter_contract())
    }

    fn refresh_numerical_certificate_set_digest(value: &mut JsonValue) {
        let object = value.as_object().unwrap();
        let relation_set = json!({
            "abi": "pyamplicol-authenticated-numerical-current-relation-set-v2",
            "certificates": object["certificates"].clone(),
            "mappings": object["mappings"].clone(),
        });
        let digest =
            canonical_json_sha256(&relation_set, "test numerical relation certificate set")
                .unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("certificate_set_sha256".to_owned(), json!(digest));
    }

    fn raw_capture_fixture(
        label: &str,
        domain: &str,
        point_base: u32,
        source: &RawSourceSemantics,
        source_semantics_sha256: &str,
        runtime_parameter_schema_sha256: &str,
        seed: u64,
    ) -> (JsonValue, RawNumericalCapture) {
        let points = (0..4_u32)
            .map(|index| {
                let value = point_base + index;
                json!({
                    "schema_version": 1,
                    "kind": "pyamplicol-rusticol-validation-momenta",
                    "process_id": "raw-v3-test",
                    "process": "1 -> 1",
                    "seed": domain_seed(seed, domain, index as usize),
                    "available": true,
                    "error": null,
                    "points": [[{
                        "pdg": 1,
                        "momentum": [value.to_string(), "0", "0", "0"],
                    }]],
                })
            })
            .collect::<Vec<_>>();
        let point_sha256s = points
            .iter()
            .map(|point| canonical_json_sha256(point, "fixture point").unwrap())
            .collect::<Vec<_>>();
        let kinematic_binary64 = (0..4_u32)
            .map(|index| {
                json!([[
                    python_f64_hex(f64::from(point_base + index)),
                    "0x0.0p+0",
                    "0x0.0p+0",
                    "0x0.0p+0",
                ]])
            })
            .collect::<Vec<_>>();
        let kinematic_sha256s = kinematic_binary64
            .iter()
            .map(|row| canonical_json_sha256(row, "fixture kinematics").unwrap())
            .collect::<Vec<_>>();
        let selector_contexts = (0..4)
            .map(|_| {
                json!({
                    "strategy": "contracted-color-union",
                    "fixed_source_schedule": true,
                })
            })
            .collect::<Vec<_>>();
        let selector_context_sha256s = selector_contexts
            .iter()
            .map(|row| canonical_json_sha256(row, "fixture selector").unwrap())
            .collect::<Vec<_>>();
        let parameter_contexts = vec![json!([]); 4];
        let parameter_context_sha256s = (0..4)
            .map(|point_index| {
                canonical_json_sha256(
                    &json!({
                        "abi": "pyamplicol-recurrence-parameter-context-v1",
                        "domain": domain,
                        "point_index": point_index,
                        "point_sha256": point_sha256s[point_index],
                        "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
                        "values": [],
                    }),
                    "fixture parameter context",
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let observation_scalars = [
            [1_i64, 1, 1, 1],
            [2, 3, 4, 5],
            [2, 3, 4, 5],
            [-2, -3, -4, -5],
            [0, 0, 0, 0],
            [7, 8, 9, 11],
        ];
        let observations = observation_scalars
            .iter()
            .enumerate()
            .map(|(current_id, values)| {
                json!({
                    "current_id": current_id,
                    "dimension": 1,
                    "values": values
                        .iter()
                        .map(|value| json!([value.to_string(), "0"]))
                        .collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>();
        let dimensions = (0..observation_scalars.len())
            .map(|current_id| json!({"current_id": current_id, "dimension": 1}))
            .collect::<Vec<_>>();
        let observation_batch_sha256 = canonical_json_sha256(
            &json!({
                "abi": "pyamplicol-recurrence-current-observation-batch-v2",
                "source_semantics_sha256": source_semantics_sha256,
                "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
                "point_sha256s": point_sha256s,
                "selector_context_sha256s": selector_context_sha256s,
                "parameter_context_sha256s": parameter_context_sha256s,
                "currents": observations,
            }),
            "fixture observation batch",
        )
        .unwrap();
        let capture_contract_sha256 = canonical_json_sha256(
            &json!({
                "abi": "pyamplicol-recurrence-current-observation-capture-v2",
                "precision_digits": 96,
                "source_semantics_sha256": source_semantics_sha256,
                "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
                "point_sha256s": point_sha256s,
                "kinematic_sha256s": kinematic_sha256s,
                "selector_context_sha256s": selector_context_sha256s,
                "parameter_context_sha256s": parameter_context_sha256s,
                "current_dimensions": dimensions,
                "observation_batch_sha256": observation_batch_sha256,
                "context_policy": "fixed-contracted-source-schedule-v1",
            }),
            "fixture capture contract",
        )
        .unwrap();
        let dimensions_object = (0..observation_scalars.len())
            .map(|current_id| (current_id.to_string(), json!(1)))
            .collect::<JsonMap<_, _>>();
        let value = json!({
            "abi": "pyamplicol-recurrence-current-observation-capture-v2",
            "precision_digits": 96,
            "point_count": 4,
            "point_sha256s": point_sha256s,
            "kinematic_sha256s": kinematic_sha256s,
            "parameter_contexts": parameter_contexts,
            "parameter_context_sha256s": parameter_context_sha256s,
            "context_sha256s": selector_context_sha256s,
            "points": points,
            "current_count": observation_scalars.len(),
            "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
            "source_semantics_sha256": source_semantics_sha256,
            "observation_batch_sha256": observation_batch_sha256,
            "capture_contract_sha256": capture_contract_sha256,
            "context_policy": "fixed-contracted-source-schedule-v1",
            "complete_current_components": true,
            "point_major": true,
            "current_dimensions_sha256": canonical_json_sha256(
                &JsonValue::Object(dimensions_object),
                "fixture dimensions",
            ).unwrap(),
            "evaluator": "recurrence-direct-plan-decimal-symbolica-exact",
            "kinematic_binary64": kinematic_binary64,
            "selector_contexts": selector_contexts,
            "current_dimensions": dimensions,
            "observations": observations,
        });
        let parsed = parse_raw_numerical_capture(
            &value,
            label,
            domain,
            96,
            4,
            source,
            source_semantics_sha256,
            runtime_parameter_schema_sha256,
            0,
            seed,
        )
        .unwrap();
        (value, parsed)
    }

    fn canonical_numerical_relation_evidence() -> JsonValue {
        canonical_numerical_relation_evidence_with_tolerances(1.0e-70, 1.0e-80)
    }

    fn canonical_numerical_relation_evidence_with_tolerances(
        relative_tolerance: f64,
        absolute_tolerance: f64,
    ) -> JsonValue {
        let seed = 0x5059_414d_u64;
        let source_semantics = json!({
            "abi": "pyamplicol-recurrence-numerical-current-source-v3",
            "process_id": "raw-v3-test",
            "strategy": "contracted-color-union",
            "schedule_semantic_digest": digest(40),
            "baseline_runtime_layout_digest": digest(41),
            "selector_schedule": {
                "policy": "fixed-contracted-source-schedule-v1",
                "fixed_source_schedule": true,
            },
            "currents": (0..6_u32).map(|current_id| json!({
                "current_id": current_id,
                "is_source": current_id == 0,
                "contract": [
                    if current_id == 0 { 0 } else { 1 },
                    1, 7, 1, 0, 0, u32::MAX, u32::MAX, ["1", "1", "0", "1"],
                ],
            })).collect::<Vec<_>>(),
        });
        let source_semantics_sha256 =
            canonical_json_sha256(&source_semantics, "fixture source semantics").unwrap();
        let source = parse_raw_source_semantics(
            &source_semantics,
            &source_semantics_sha256,
            &digest(40),
            &digest(41),
            &empty_runtime_parameter_contract(),
        )
        .unwrap();
        let runtime_parameter_schema = json!({
            "abi": "pyamplicol-recurrence-runtime-parameter-schema-v2",
            "parameters": [],
        });
        let runtime_parameter_schema_sha256 = canonical_json_sha256(
            &runtime_parameter_schema,
            "fixture runtime parameter schema",
        )
        .unwrap();
        let (candidate_value, candidate) = raw_capture_fixture(
            "candidate",
            "candidate-current-probes-v1",
            1,
            &source,
            &source_semantics_sha256,
            &runtime_parameter_schema_sha256,
            seed,
        );
        let (verification_value, verification) = raw_capture_fixture(
            "verification",
            "independent-verification-current-probes-v1",
            101,
            &source,
            &source_semantics_sha256,
            &runtime_parameter_schema_sha256,
            seed,
        );
        validate_independent_raw_captures(
            &candidate,
            &verification,
            &empty_runtime_parameter_contract(),
            seed,
        )
        .unwrap();
        let derivation = derive_raw_numerical_relations(
            &source,
            &candidate,
            &verification,
            96,
            seed,
            &exact_binary64_rational(relative_tolerance).unwrap(),
            &exact_binary64_rational(absolute_tolerance).unwrap(),
            &python_f64_hex(relative_tolerance),
            &python_f64_hex(absolute_tolerance),
            &runtime_parameter_schema_sha256,
            rusticol_core::recurrence::NUMERICAL_RELATION_CERTIFICATE_ALGORITHM,
        )
        .unwrap();
        assert_eq!(
            derivation
                .relations
                .iter()
                .map(|relation| relation.mapping.relation_kind.as_str())
                .collect::<Vec<_>>(),
            ["equal", "opposite", "zero"],
        );
        let certificates = derivation
            .relations
            .iter()
            .map(|relation| relation.certificate.clone())
            .collect::<Vec<_>>();
        let mappings = derivation
            .relations
            .iter()
            .map(|relation| relation.mapping_json.clone())
            .collect::<Vec<_>>();
        let mut evidence = json!({
            "abi": "pyamplicol-recurrence-numerical-current-evidence-v3",
            "requested_mode": "certified-reuse",
            "schedule_semantic_digest": digest(40),
            "baseline_runtime_layout_digest": digest(41),
            "source_semantics": source_semantics,
            "source_semantics_sha256": source_semantics_sha256,
            "runtime_parameter_schema": runtime_parameter_schema,
            "runtime_parameter_schema_sha256": runtime_parameter_schema_sha256,
            "candidate_capture": candidate_value,
            "verification_capture": verification_value,
            "certificate_algorithm":
                rusticol_core::recurrence::NUMERICAL_RELATION_CERTIFICATE_ALGORITHM,
            "certificate_set_sha256": digest(0),
            "precision_digits": 96,
            "probe_count": 4,
            "verification_probe_count": 4,
            "relative_tolerance_binary64": python_f64_hex(relative_tolerance),
            "absolute_tolerance_binary64": python_f64_hex(absolute_tolerance),
            "seed": seed,
            "candidate_index": {
                "algorithm": "complete-contract-anchor-tolerance-window-v1",
                "completeness": "complete-within-configured-tolerance",
                "contract_count": derivation.candidate_index_contract_count,
                "exhaustive_fallback_contract_count":
                    derivation.exhaustive_fallback_contract_count,
                "theoretical_pair_hypothesis_count":
                    derivation.theoretical_pair_hypothesis_count,
                "screened_pair_hypothesis_count":
                    derivation.screened_pair_hypothesis_count,
                "zero_hypothesis_count": derivation.zero_hypothesis_count,
                "screened_hypothesis_budget": 1_000_000,
                "budget_classification": "within-authenticated-budget",
                "nearest_rejected_scope":
                    "zero-and-tolerance-window-screened-hypotheses",
            },
            "numerical_candidate_count": derivation.numerical_candidate_count,
            "verification_rejected_count": derivation.verification_rejected_count,
            "rejected_hypothesis_count": derivation.rejected_hypothesis_count,
            "tested_hypothesis_count": derivation.tested_hypothesis_count,
            "decision_sha256": derivation.decision_sha256,
            "rejection_decision_sha256":
                derivation.rejection_decision_sha256,
            "certificates": certificates,
            "mappings": mappings,
        });
        refresh_numerical_certificate_set_digest(&mut evidence);
        evidence
    }

    #[test]
    fn numerical_relation_evidence_recomputes_every_digest_layer() {
        let evidence = canonical_numerical_relation_evidence();
        let encoded = canonical_json_bytes(&evidence, "test numerical evidence").unwrap();
        let parsed = parse_fixture(&encoded).unwrap();
        assert_eq!(parsed.mappings.len(), 3);
        assert_eq!(
            parsed.rejected_hypothesis_count,
            parsed.tested_hypothesis_count - parsed.mappings.len(),
        );
        assert!(parsed.rejected_hypothesis_count > parsed.verification_rejected_count);

        let mut stale_set = evidence.clone();
        stale_set["certificates"][0]["current_dimension"] = json!(3);
        let stale_bytes =
            canonical_json_bytes(&stale_set, "stale numerical certificate set").unwrap();
        assert!(
            parse_fixture(&stale_bytes)
                .unwrap_err()
                .to_string()
                .contains("not derived from raw observations")
        );

        let mut stale_proof = evidence.clone();
        stale_proof["certificates"][0]["candidate_maximum_absolute_residual"] = json!("1e-90");
        refresh_numerical_certificate_set_digest(&mut stale_proof);
        let stale_bytes = canonical_json_bytes(&stale_proof, "stale numerical proof").unwrap();
        assert!(
            parse_fixture(&stale_bytes)
                .unwrap_err()
                .to_string()
                .contains("not derived from raw observations")
        );

        let mut stale_mapping = evidence.clone();
        stale_mapping["mappings"][0]["current_dimension"] = json!(3);
        refresh_numerical_certificate_set_digest(&mut stale_mapping);
        let stale_bytes = canonical_json_bytes(&stale_mapping, "stale numerical mapping").unwrap();
        assert!(
            parse_fixture(&stale_bytes)
                .unwrap_err()
                .to_string()
                .contains("not derived from raw observations")
        );

        let mut stale_tolerance = evidence;
        stale_tolerance["relative_tolerance_binary64"] = json!("0x1.0p+0");
        let stale_bytes =
            canonical_json_bytes(&stale_tolerance, "stale numerical tolerance").unwrap();
        assert!(
            parse_fixture(&stale_bytes)
                .unwrap_err()
                .to_string()
                .contains("canonical nonnegative finite binary64")
        );

        let mut stale_index = canonical_numerical_relation_evidence();
        stale_index["candidate_index"]["screened_pair_hypothesis_count"] = json!(99);
        let stale_bytes =
            canonical_json_bytes(&stale_index, "stale numerical candidate index").unwrap();
        assert!(
            parse_fixture(&stale_bytes)
                .unwrap_err()
                .to_string()
                .contains("was not independently derived")
        );
    }

    #[test]
    fn raw_probe_identity_is_bound_to_the_authenticated_process_and_seed_domain() {
        fn mutate_first_candidate_point(
            mut evidence: JsonValue,
            field: &str,
            value: JsonValue,
        ) -> Vec<u8> {
            evidence["candidate_capture"]["points"][0][field] = value;
            let digest = canonical_json_sha256(
                &evidence["candidate_capture"]["points"][0],
                "tampered candidate point",
            )
            .unwrap();
            evidence["candidate_capture"]["point_sha256s"][0] = json!(digest);
            canonical_json_bytes(&evidence, "tampered probe identity").unwrap()
        }

        let wrong_id = mutate_first_candidate_point(
            canonical_numerical_relation_evidence(),
            "process_id",
            json!("unrelated-process"),
        );
        assert!(
            parse_fixture(&wrong_id)
                .unwrap_err()
                .to_string()
                .contains("process_id")
        );

        let wrong_pdg = mutate_first_candidate_point(
            canonical_numerical_relation_evidence(),
            "points",
            json!([[{
                "pdg": -1,
                "momentum": ["1", "0", "0", "0"],
            }]]),
        );
        assert!(
            parse_fixture(&wrong_pdg)
                .unwrap_err()
                .to_string()
                .contains("authenticated recurrence process")
        );

        let wrong_seed =
            mutate_first_candidate_point(canonical_numerical_relation_evidence(), "seed", json!(0));
        assert!(
            parse_fixture(&wrong_seed)
                .unwrap_err()
                .to_string()
                .contains("validation-point seed is stale")
        );
    }

    #[test]
    fn numerical_relation_evidence_accepts_canonical_binary64_tolerance_boundaries() {
        for (relative, absolute) in [
            (1.0e-8, 1.0e-9),
            (f64::from_bits(1), 0.0),
            (f64::MIN_POSITIVE, f64::from_bits(1)),
            (0.25, 0.0),
        ] {
            let evidence =
                canonical_numerical_relation_evidence_with_tolerances(relative, absolute);
            let encoded = canonical_json_bytes(&evidence, "boundary tolerance evidence").unwrap();
            let parsed = parse_fixture(&encoded).unwrap();
            assert_eq!(parsed.relative_tolerance.to_bits(), relative.to_bits());
            assert_eq!(parsed.absolute_tolerance.to_bits(), absolute.to_bits());
        }
        let (maximum, exact) =
            parse_nonnegative_python_f64_hex(&python_f64_hex(f64::MAX), "maximum").unwrap();
        assert_eq!(maximum, f64::MAX);
        assert_eq!(exact, exact_binary64_rational(f64::MAX).unwrap());
    }

    #[test]
    fn numerical_relation_evidence_rejects_nonfinite_negative_zero_and_noncanonical_tolerances() {
        for invalid_tolerance in [
            "nan",
            "inf",
            "-0x0.0p+0",
            "0x1.0p-3",
            "0X1.0000000000000P-3",
        ] {
            let mut evidence = canonical_numerical_relation_evidence();
            evidence["relative_tolerance_binary64"] = json!(invalid_tolerance);
            let encoded = canonical_json_bytes(&evidence, "invalid tolerance evidence").unwrap();
            assert!(
                parse_fixture(&encoded)
                    .unwrap_err()
                    .to_string()
                    .contains("canonical nonnegative finite binary64")
            );
        }
        let mut zero = canonical_numerical_relation_evidence();
        zero["relative_tolerance_binary64"] = json!("0x0.0p+0");
        zero["absolute_tolerance_binary64"] = json!("0x0.0p+0");
        let encoded = canonical_json_bytes(&zero, "zero tolerance evidence").unwrap();
        assert!(
            parse_fixture(&encoded)
                .unwrap_err()
                .to_string()
                .contains("cannot both be zero")
        );
    }

    #[test]
    fn raw_candidate_index_is_complete_at_equal_and_opposite_boundaries() {
        fn rational(numerator: i128, denominator: i128) -> BigRational {
            BigRational::new(BigInt::from(numerator), BigInt::from(denominator))
        }
        fn probe(value: BigRational) -> RawProbeValue {
            RawProbeValue {
                real_text: rational_string(&value),
                imaginary_text: "0".to_owned(),
                value: (value, BigRational::zero()),
            }
        }
        let large = BigRational::from_integer(BigInt::from(10_u8).pow(100));
        let values = [
            (0_u32, rational(100, 9), large.clone()),
            (1, rational(-100, 9), -large.clone()),
            (
                2,
                BigRational::from_integer(BigInt::from(1_000)),
                BigRational::zero(),
            ),
            (4, BigRational::from_integer(BigInt::from(10)), large),
        ];
        let observations = values
            .into_iter()
            .map(|(current_id, selected, unrelated)| {
                (current_id, vec![probe(selected), probe(unrelated)])
            })
            .collect::<BTreeMap<_, _>>();
        let source = RawSourceSemantics {
            value: json!({}),
            process_id: "candidate-index-test".to_owned(),
            physical_pdgs: vec![1],
            strategy: "contracted-color-union".to_owned(),
            selector_schedule: json!({}),
            currents: [0_u32, 1, 2, 4]
                .into_iter()
                .map(|current_id| RawSourceCurrent {
                    current_id,
                    is_source: false,
                    contract_key: vec![1],
                    dimension: 2,
                })
                .collect(),
        };
        let capture = RawNumericalCapture {
            point_sha256s: Vec::new(),
            kinematic_sha256s: Vec::new(),
            parameter_context_sha256s: Vec::new(),
            parameter_contexts: Vec::new(),
            observations,
            dimensions: BTreeMap::new(),
            observation_batch_sha256: digest(80),
            capture_contract_sha256: digest(81),
        };
        let indexes = build_raw_numerical_candidate_indexes(&source, &capture).unwrap();
        let index = indexes.get(&vec![1]).unwrap();
        assert_eq!((index.observation_index, index.scalar_component), (0, 0));
        let current = capture.observations.get(&4).unwrap();
        let relative = rational(1, 10);
        let absolute = BigRational::zero();
        let boundary_residual = rational(10, 9);
        assert_eq!(
            (&current[0].value.0 - &capture.observations[&0][0].value.0).abs(),
            boundary_residual,
        );
        assert_eq!(
            &relative * capture.observations[&0][0].value.0.abs(),
            boundary_residual,
        );
        // The unrelated 10^100 observation is exact under the signed
        // relations and must not enlarge the selected pair's window.
        assert_eq!(
            raw_numerical_tolerance_window_ids(index, current, 4, "equal", &relative, &absolute,)
                .unwrap(),
            BTreeSet::from([0]),
        );
        assert_eq!(
            raw_numerical_tolerance_window_ids(
                index, current, 4, "opposite", &relative, &absolute,
            )
            .unwrap(),
            BTreeSet::from([1]),
        );
    }

    #[test]
    fn numerical_relation_options_reject_stale_probe_and_counter_contracts() {
        let evidence = canonical_numerical_relation_evidence();
        let encoded = canonical_json_bytes(&evidence, "test numerical evidence").unwrap();
        let parsed = parse_fixture(&encoded).unwrap();
        let options = RecurrenceRelationDiscoveryOptions::new(
            RecurrenceRelationDiscoveryMode::CertifiedReuse,
            96,
            4,
            4,
            1.0e-70,
            1.0e-80,
            0x5059_414d,
            "lc",
        )
        .unwrap();
        options
            .clone()
            .with_numerical_evidence(parsed.clone())
            .unwrap();

        let mut stale_probe = parsed.clone();
        stale_probe.probe_count = 5;
        assert!(
            options
                .clone()
                .with_numerical_evidence(stale_probe)
                .unwrap_err()
                .to_string()
                .contains("probe contract")
        );

        let mut stale_counters = parsed;
        stale_counters.verification_rejected_count = 1;
        assert!(
            options
                .with_numerical_evidence(stale_counters)
                .unwrap_err()
                .to_string()
                .contains("candidate counters")
        );
    }

    #[test]
    fn raw_numerical_replay_rejects_context_nonfinite_and_decision_tampering() {
        let evidence = canonical_numerical_relation_evidence();

        let mut selector_tamper = evidence.clone();
        selector_tamper["candidate_capture"]["selector_contexts"][0]["fixed_source_schedule"] =
            json!(false);
        let error =
            parse_fixture(&canonical_json_bytes(&selector_tamper, "selector tamper").unwrap())
                .unwrap_err()
                .to_string();
        assert!(error.contains("selector context"));

        let mut nonfinite = evidence.clone();
        nonfinite["candidate_capture"]["observations"][1]["values"][0][0] = json!("NaN");
        let error =
            parse_fixture(&canonical_json_bytes(&nonfinite, "nonfinite raw observation").unwrap())
                .unwrap_err()
                .to_string();
        assert!(error.contains("canonical finite decimal"));

        let mut stale_decision = evidence;
        stale_decision["decision_sha256"] = json!(digest(99));
        let error =
            parse_fixture(&canonical_json_bytes(&stale_decision, "stale decision digest").unwrap())
                .unwrap_err()
                .to_string();
        assert!(error.contains("decision digest"));

        let mut stale_rejection = canonical_numerical_relation_evidence();
        stale_rejection["rejection_decision_sha256"] = json!(digest(98));
        let error = parse_fixture(
            &canonical_json_bytes(&stale_rejection, "stale rejection digest").unwrap(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("rejection digest"));
    }

    #[test]
    fn raw_numerical_tolerance_boundary_and_parameter_domains_are_exact() {
        let defaults = vec![AuthenticatedRuntimeParameterSlot {
            default: BigRational::new(BigInt::from(3), BigInt::from(2)),
            probe_policy: "native-template-default-perturbed-v1",
        }];
        let candidate = deterministic_parameter_contexts(
            &defaults,
            4,
            17,
            "candidate-current-parameter-probes-v1",
            true,
        )
        .unwrap();
        let verification = deterministic_parameter_contexts(
            &defaults,
            4,
            17,
            "independent-verification-current-parameter-probes-v1",
            false,
        )
        .unwrap();
        assert_eq!(candidate[0], vec![defaults[0].default.clone()]);
        assert!(
            candidate
                .iter()
                .all(|row| !verification.iter().any(|other| row == other))
        );

        let value = |numerator: i64, denominator: i64| RawProbeValue {
            real_text: rational_string(&BigRational::new(
                BigInt::from(numerator),
                BigInt::from(denominator),
            )),
            imaginary_text: "0".to_owned(),
            value: (
                BigRational::new(BigInt::from(numerator), BigInt::from(denominator)),
                BigRational::zero(),
            ),
        };
        let capture = |current_value: RawProbeValue| RawNumericalCapture {
            point_sha256s: vec![digest(1)],
            kinematic_sha256s: vec![digest(2)],
            parameter_context_sha256s: vec![digest(4)],
            parameter_contexts: vec![vec![]],
            observations: BTreeMap::from([(0, vec![value(1, 1)]), (1, vec![current_value])]),
            dimensions: BTreeMap::from([(0, 1), (1, 1)]),
            observation_batch_sha256: digest(5),
            capture_contract_sha256: digest(6),
        };
        let boundary = relation_residuals(
            "equal",
            1,
            Some(0),
            &capture(value(2, 1)),
            &BigRational::zero(),
            &BigRational::one(),
        )
        .unwrap();
        assert_eq!(boundary.maximum_ratio, BigRational::one());
        let outside = relation_residuals(
            "equal",
            1,
            Some(0),
            &capture(value(2_000_001, 1_000_000)),
            &BigRational::zero(),
            &BigRational::one(),
        )
        .unwrap();
        assert!(outside.maximum_ratio > BigRational::one());
    }

    #[test]
    fn authenticated_parameter_defaults_reject_shifted_neighborhoods() {
        fn contract(default: i64) -> AuthenticatedRuntimeParameterContract {
            let default = BigRational::from_integer(BigInt::from(default));
            let schema = json!({
                "abi": "pyamplicol-recurrence-runtime-parameter-schema-v2",
                "parameters": [{
                    "runtime_slot": 0,
                    "runtime_name": "mass",
                    "parameter_template_id": 0,
                    "prepared_parameter_id": 0,
                    "component": 0,
                    "default_binary64": python_f64_hex_signed(default.to_f64().unwrap()),
                    "probe_policy": "native-template-default-perturbed-v1",
                }],
            });
            AuthenticatedRuntimeParameterContract {
                schema_sha256: canonical_json_sha256(&schema, "test runtime parameter schema")
                    .unwrap(),
                schema,
                slots: vec![AuthenticatedRuntimeParameterSlot {
                    default,
                    probe_policy: "native-template-default-perturbed-v1",
                }],
                process_id: "parameter-test".to_owned(),
                physical_pdgs: vec![1],
            }
        }

        let authenticated = contract(3);
        let shifted = contract(5);
        assert!(
            validate_raw_runtime_parameter_schema(
                &shifted.schema,
                &shifted.schema_sha256,
                &authenticated,
            )
            .unwrap_err()
            .to_string()
            .contains("authenticated native template input")
        );
    }

    #[test]
    fn native_parameter_default_hex_preserves_negative_binary64_signs() {
        assert_eq!(python_f64_hex_signed(-1.5), "-0x1.8000000000000p+0");
        assert_eq!(
            python_f64_hex_signed(f64::from_bits(0x8000_0000_0000_0001)),
            "-0x0.0000000000001p-1022"
        );
        assert_eq!(python_f64_hex_signed(0.0), "0x0.0p+0");
    }

    #[test]
    fn derived_parameter_probe_slots_are_fixed_and_tampering_fails() {
        let contract = AuthenticatedRuntimeParameterContract {
            schema: json!({}),
            schema_sha256: digest(90),
            slots: vec![AuthenticatedRuntimeParameterSlot {
                default: BigRational::zero(),
                probe_policy: "derived-overwritten-fixed-zero-v1",
            }],
            process_id: "parameter-test".to_owned(),
            physical_pdgs: vec![1],
        };
        let candidate_contexts = deterministic_parameter_contexts(
            &contract.slots,
            4,
            17,
            "candidate-current-parameter-probes-v1",
            true,
        )
        .unwrap();
        let verification_contexts = deterministic_parameter_contexts(
            &contract.slots,
            4,
            17,
            "independent-verification-current-parameter-probes-v1",
            false,
        )
        .unwrap();
        assert!(
            candidate_contexts
                .iter()
                .chain(&verification_contexts)
                .all(|row| row == &[BigRational::zero()])
        );
        let capture = |parameter_contexts, base: u8| RawNumericalCapture {
            point_sha256s: (0..4).map(|index| digest(base + index)).collect(),
            kinematic_sha256s: (0..4).map(|index| digest(base + 10 + index)).collect(),
            parameter_context_sha256s: (0..4).map(|index| digest(base + 20 + index)).collect(),
            parameter_contexts,
            observations: BTreeMap::new(),
            dimensions: BTreeMap::new(),
            observation_batch_sha256: digest(base + 30),
            capture_contract_sha256: digest(base + 31),
        };
        let mut candidate = capture(candidate_contexts, 1);
        let verification = capture(verification_contexts, 101);
        validate_independent_raw_captures(&candidate, &verification, &contract, 17).unwrap();
        candidate.parameter_contexts[1][0] = BigRational::one();
        assert!(
            validate_independent_raw_captures(&candidate, &verification, &contract, 17)
                .unwrap_err()
                .to_string()
                .contains("stale or non-deterministic")
        );
    }

    #[test]
    fn parameter_rows_are_included_in_raw_memory_preflight() {
        let source = RawSourceSemantics {
            value: json!({}),
            process_id: "memory-preflight-test".to_owned(),
            physical_pdgs: vec![1],
            strategy: "contracted-color-union".to_owned(),
            selector_schedule: json!({}),
            currents: Vec::new(),
        };
        assert!(
            validate_raw_evidence_geometry(&source, 2, 2, 400_000)
                .unwrap_err()
                .to_string()
                .contains("resident memory envelope")
        );
    }

    #[test]
    fn lexical_caps_bound_serde_dom_before_materialization() {
        let maximum_pre_dom_resident = RAW_PRE_DOM_FIXED_BYTES
            + MAX_RAW_EVIDENCE_BYTES * RAW_PRE_DOM_WIRE_COPIES
            + MAX_RAW_JSON_STRUCTURAL_TOKENS * RAW_PRE_DOM_BYTES_PER_TOKEN;
        assert_eq!(maximum_pre_dom_resident, 989_261_824);
        assert!(maximum_pre_dom_resident < MAX_RAW_RESIDENT_BYTES);
        validate_pre_dom_resident_budget(MAX_RAW_EVIDENCE_BYTES, MAX_RAW_JSON_STRUCTURAL_TOKENS)
            .unwrap();

        let first_token_count_over_envelope =
            (MAX_RAW_RESIDENT_BYTES - RAW_PRE_DOM_FIXED_BYTES) / RAW_PRE_DOM_BYTES_PER_TOKEN + 1;
        assert!(
            validate_pre_dom_resident_budget(0, first_token_count_over_envelope)
                .unwrap_err()
                .to_string()
                .contains("pre-DOM 1 GiB")
        );
    }

    #[test]
    fn combined_dom_and_exact_capture_residents_fail_closed() {
        let raw_byte_count = 150_000_000;
        let structural_token_count = 6_000_000;
        let scalar_count = 800_000;
        let row_count = 20_000;

        assert!(raw_byte_count < MAX_RAW_EVIDENCE_BYTES);
        assert!(structural_token_count < MAX_RAW_JSON_STRUCTURAL_TOKENS);
        validate_pre_dom_resident_budget(raw_byte_count, structural_token_count).unwrap();
        assert!(
            validate_combined_raw_resident_budget(
                raw_byte_count,
                structural_token_count,
                scalar_count,
                row_count,
            )
            .unwrap_err()
            .to_string()
            .contains("combined 1 GiB"),
            "separately valid DOM and exact-capture bounds must not be added after allocation"
        );
    }

    #[test]
    fn a_like_raw_capture_passes_the_combined_native_memory_model() {
        // Cross-model counterpart of Python's canonical synthetic study:
        // 17,000 four-component currents over 4+4 probes encode to
        // 128,293,862 bytes.  Each capture row contributes 107 lexical
        // tokens; the two observation arrays add their separators, yielding
        // 216 * current_count + 17 total tokens.
        let raw_byte_count = 128_293_862;
        let structural_token_count = 216 * 17_000 + 17;
        let scalar_count = 1_088_000;
        let row_count = 34_000;
        assert_eq!(structural_token_count, 3_672_017);
        validate_pre_dom_resident_budget(raw_byte_count, structural_token_count).unwrap();
        validate_combined_raw_resident_budget(
            raw_byte_count,
            structural_token_count,
            scalar_count,
            row_count,
        )
        .unwrap();

        let first_token_count_over_combined_envelope = (MAX_RAW_RESIDENT_BYTES
            - RAW_PRE_DOM_FIXED_BYTES
            - raw_byte_count * RAW_PRE_DOM_WIRE_COPIES
            - scalar_count * RAW_CAPTURE_BYTES_PER_SCALAR
            - row_count * RAW_CAPTURE_BYTES_PER_ROW)
            / RAW_PRE_DOM_BYTES_PER_TOKEN
            + 1;
        assert!(first_token_count_over_combined_envelope < MAX_RAW_JSON_STRUCTURAL_TOKENS);
        validate_pre_dom_resident_budget(raw_byte_count, first_token_count_over_combined_envelope)
            .unwrap();
        assert!(
            validate_combined_raw_resident_budget(
                raw_byte_count,
                first_token_count_over_combined_envelope,
                scalar_count,
                row_count,
            )
            .is_err()
        );
    }

    #[test]
    fn real_a_shape_has_the_same_dynamic_wire_boundary_as_python() {
        let current_count = 15_834_usize + 1_240;
        let component_count = 15_834_usize * 4 + 1_240 * 6;
        let point_count = 4_usize + 4;
        let runtime_parameter_count = 10_usize;
        let scalar_count =
            component_count * point_count * 2 + runtime_parameter_count * point_count;
        let row_count = current_count * 2 + runtime_parameter_count;

        assert_eq!((current_count, component_count), (17_074, 70_776));
        assert_eq!((scalar_count, row_count), (1_132_496, 34_158));
        let wire_limit = raw_evidence_shape_wire_limit(scalar_count, row_count).unwrap();
        assert_eq!(wire_limit, 148_950_528);
        assert!(115_356_478 < wire_limit, "configured 96-digit estimate");
        assert!(133_475_134 < wire_limit, "conservative 112-char estimate");
        assert!(151_593_790 > wire_limit, "128-char estimate must fail");

        let without_parameter_metadata =
            raw_evidence_shape_wire_limit(scalar_count - 80, row_count - 10).unwrap();
        assert_eq!(without_parameter_metadata, 148_978_688);
        assert_eq!(without_parameter_metadata - wire_limit, 28_160);
    }

    #[test]
    fn shape_wire_budget_rejects_overflow_and_missing_wire_reserve() {
        assert!(
            raw_evidence_shape_wire_limit(usize::MAX, usize::MAX)
                .unwrap_err()
                .to_string()
                .contains("overflows usize")
        );
        let consumes_envelope =
            (MAX_RAW_RESIDENT_BYTES - RAW_PRE_DOM_FIXED_BYTES) / RAW_PRODUCER_BYTES_PER_SCALAR;
        assert!(
            raw_evidence_shape_wire_limit(consumes_envelope, 0)
                .unwrap_err()
                .to_string()
                .contains("wire reserve")
        );
        assert_eq!(
            raw_evidence_shape_wire_limit(0, 0).unwrap(),
            MAX_RAW_EVIDENCE_BYTES
        );
    }

    #[test]
    fn excessive_depth_and_string_width_are_rejected_lexically() {
        let excessive_depth = vec![b'['; MAX_RAW_JSON_DEPTH + 1];
        assert!(
            validate_raw_json_lexical_budget(&excessive_depth)
                .unwrap_err()
                .to_string()
                .contains("nesting")
        );

        let mut excessive_string = Vec::with_capacity(MAX_RAW_JSON_STRING_BYTES + 3);
        excessive_string.push(b'"');
        excessive_string.extend(std::iter::repeat_n(b'a', MAX_RAW_JSON_STRING_BYTES + 1));
        excessive_string.push(b'"');
        assert!(
            validate_raw_json_lexical_budget(&excessive_string)
                .unwrap_err()
                .to_string()
                .contains("string exceeds")
        );
    }
}
