# Benchmark policy v2: corrected model helpers and original timing reference

Status: design only. No measurement protocol, numerical implementation, old
run, or acceptance result is changed by this document.

## Minimal architecture

Keep `ninjars_benchmark.py`, `ninjars_benchmark_workers.py`, both Fortran
drivers, the existing v1 proof and every existing run unchanged. The frozen
seven-target runner already validates a target against its own calibration;
it does not compare original and candidate physics during timing. Therefore
deterministic wrong original QP coefficients require **no runtime bypass**.

Add three small tools:

1. `ninjars_benchmark_policy_v2.py`: validates the correction/source graph,
   representative proof, complete timing series and corrected-reference
   series; recomputes comparisons and publishes a content-addressed v2
   dataset manifest.
2. `ninjars_benchmark_references_v2.py`: constructs isolated original outputs
   with exactly the two verified model-source corrections, rebuilds the model
   and executable, and collects all points with the existing worker primitives.
3. `ninjars_report_v2.py`: reads only a validated v2 dataset manifest, independently
   recomputes numerical/performance conclusions, and produces the report
   snapshot. It may reuse pure formatting/statistics helpers from v1.

The collection protocol remains `persistent-worker-v1`; the **acceptance policy**
is `madloop-helper-correction-policy-v2`. Keeping these two identities separate
avoids claiming that timer boundaries changed. A new
`representative-policy-validation-v2` proof is mandatory: the old v1 proof
alone cannot validate the new source roles or corrected model implementation.

Do not add `--ignore-failures`, an expected-point list, tolerance overrides or
any “accept known QP failures” option.

## Source roles and required data

| Role | Backend set | Source | Use |
|---|---|---|---|
| `original` | 1, 6, 7 | Untouched archived revision `27147b3300f556a75661a886e92bdbe00558ee25` and its original model code | Seven-target timing series; original discrepancies retained |
| `candidate_corrected` | 1, 6, 7, 8 | Final candidate, with corrected generated helpers | Seven-target timing series and correctness |
| `original_helper_corrected` | 1, 6, 7 | Isolated original generated outputs with only the verified model corrections | Correctness references; timings diagnostic only |

The main timing series remains exactly the seven existing targets, at least
seven contiguous repetitions, with adjacent original Ninja/candidate NinjaRS
measurements and alternating order. The only performance denominator is
`original/6`; `original_helper_corrected/6` can never substitute for it.

The additional three corrected references need all 100 fixed points, all
Born/Laurent/split-order rows, and at least two retained worker commands after
initialization/warmup. Their purpose is numerical comparison, so they do not
need seven performance repetitions. Fresh-versus-reused parity on the
representative processes remains a separate mandatory proof.

Every role uses identical physical parameter cards, phase-space streams,
scales, numerical thresholds, rotations and pole settings. The existing
allowed backend-selector differences remain explicit. No point or coefficient
is omitted because it uses QP.

## Policy and manifest fields

The policy file is a content-addressed input, not a free-form runtime override.
Its digest is calculated over canonical JSON excluding its own digest field.
Illustrative schema:

```json
{
  "schema": "madloop-helper-correction-policy-v2",
  "policy_identity_sha256": "<digest>",
  "collection_protocol": {
    "schema": "persistent-worker-v1",
    "identity_sha256": "<frozen descriptor digest>"
  },
  "correction_identity_sha256": "<verified correction manifest digest>",
  "configuration_sha256": "<retained cases/configuration digest>",
  "timing_roles": {
    "original": [1, 6, 7],
    "candidate_corrected": [1, 6, 7, 8]
  },
  "correctness_roles": {
    "original_helper_corrected": [1, 6, 7],
    "candidate_corrected": [1, 6, 7, 8]
  },
  "performance_pair": ["original/6", "candidate_corrected/8"],
  "points_per_case": 100,
  "minimum_timing_repetitions": 7,
  "minimum_reference_commands": 2,
  "relative_tolerance": 1e-7,
  "scale_tolerance": 1e-10,
  "maximum_runtime_ratio": 1.05
}
```

The actual supported policy must reject changes to these required roles,
point count, tolerances and performance pair. A different scientific policy
requires a separately reviewed version, not permissive JSON interpretation.

The dataset manifest records:

- Policy, configuration, correction and new representative-proof hashes.
- Immutable original run directory/hash and each original raw measurement
  hash; corrected references have separate run directories and hashes.
- Explicit role-to-run/target mappings. A role cannot be assigned merely by a
  caller-supplied label: its source/build identity must satisfy the rules below.
- Per-case completeness of the seven timing targets and three reference
  targets; every command/repetition and row stream must be present.
- Original archive identity, generation identity, full generated source
  manifest, relevant library/model archive hashes, executable and driver
  hashes, compiler flags, physical card/points hashes and numerical settings.
- Recomputed gate records and hashes of their complete comparison data.
- `structural_status`, `corrected_agreement`, `performance`, and
  `original_diagnostics` as separate fields. Publication of a failed gate is
  allowed; publication does not mean acceptance.

All v2 publications go under a new namespace, for example
`docs/benchmarkfolder/policy-v2/<dataset-digest>/`. Existing `data/`,
`agreement-*.json`, `legacy-agreement-*.json`, promotions and v1 run manifests
remain untouched. Preserve historical report snapshots/PDFs before rendering
the requested final top-level `benchmark.pdf` from v2. A small
`current-policy-v2.json` may point to the chosen immutable dataset; it cannot
overwrite original evidence.

## Exact model-source correction validation

The correction manifest must account for **both**:

- `Source/MODEL/model_functions.f`
- `Source/MODEL/mp_intparam_definition.inc`

An implementation that fixes helpers but forgets the generated internal
parameters is invalid. For each file present in the original model, record
its original bytes/hash, corrected bytes/hash and exact transformation/patch.
An absence claim is valid only when that path is actually absent. An unchanged
postimage is permitted only when independent fresh generation proves that the
file contains no affected construct. For the demonstrated EW3 case, the
required changed-file set is exactly both files.

The manifest additionally binds:

1. The original exporter/parser revision and source hashes.
2. The corrected `export_v4.py` and `ufo_expression_parsers.py` hashes and
   their exact correction diff.
3. Fresh CLI command cards and before/after generation identities that prove
   the model-source delta. Both generations must use the same UFO source,
   restrictions and physical cards. The proof must include the complete
   `Source/MODEL` source manifest, not just a grep or selected-file diff.
   Verify the delta for each generated model/control used by the final
   dataset; the successful EW3 proof alone does not establish that the parser
   correction has no other generated-source effects.
4. Independent helper tests, including binary128-distinguishable axis inputs,
   general complex inputs, both logarithm sheet shifts, and mass/width
   trajectory construction, with evidence hashes.

Construct each corrected-original reference from its recorded original
generated output, not from candidate subprocess code. Before replacing the
worker driver, compare the entire generated source manifest:

- The two specified model postimages must equal their independently verified
  expected corrected bytes.
- Every other source/include/make file must equal the original preimage.
- No extra source file, deletion, unrecorded edit or unresolved link is allowed.
- Local symlink relocation is recorded and must preserve the resolved source
  bytes. Writable model paths must resolve inside the isolated reference.
- The physical `param_card.dat` must be byte-identical, including the actual
  SETPARA search-order check already implemented by the frozen worker.

A filename allowlist by itself is insufficient: arbitrary changes inside
either allowed file must fail the expected-postimage/hash check.

Force recompilation of the model objects affected by either source, including
objects that include `mp_intparam_definition.inc`; rebuild the model archive
and relink the subprocess with unchanged compiler flags and original
DHELAS/reduction libraries. Do not rely on copied timestamps to rebuild an
archive. Retain model source/object/archive hashes and the link command.

The reference's source identity is explicitly
`original-generated-output+verified-model-correction`, containing its original
generation parent and correction digest. It must never claim to be an
unmodified verified archive. After installing the frozen worker driver, verify
that the additional driver delta is exactly the expected generated driver
bytes and retain the new manifest/executable identity.

The same correction digest must identify the final candidate model code.
Fresh candidate output must contain the corresponding expected corrected
model postimages. Candidate Fortran optimizations outside the model are
allowed only in the candidate role and retain their own generation identity.

## Collection and validation integration

The seven timed targets use the existing `run_case`, `parse_outputs`,
`validate_repetition` and cleanup/failure paths unchanged. Their historical
storage keys `baseline` and `candidate` map to `original` and
`candidate_corrected` only through the checked v2 dataset manifest.

The reference collector may reuse `Worker`, `parse_outputs`,
`validate_param_card` and `numerical_parity`. It must use the original
MadLoop card template for original-code references. Its separate workspace
can use `baseline` as an internal collector key, but the enclosing manifest
must expose `original_helper_corrected` and its derived source identity.
It must not write references into the original output directory or the main
timing-data namespace.

Existing `checked_series(minimum=7, require_core=True)` validates the main
timing series. A reference series can use `checked_series(minimum=2,
require_core=False)` plus strict v2 checks requiring exactly original
backends 1/6/7, the complete point/order stream and the correction identity.
Its timings are never passed to the performance gate.

New representative-proof validation binds:

- The frozen collection protocol digest and the new policy/correction digests.
- Fresh/reused parity for all four DP backend types, legacy QP1/QP6 and own
  QP8, loop-induced and multiple-split-order cases, with at least two retained
  commands.
- All three source roles, including corrected original QP and corrected
  candidate QP. Role-specific source/model/library/executable identities are
  part of each proof record.
- Every retained raw file hash. Existing v1 proof bytes may be retained as
  historical evidence but cannot satisfy this new proof.

No cross-role coefficient comparison belongs inside the timed command loop.
Malformed streams, unsuccessful statuses, nonfinite values, missing accuracy,
changed inputs, within-target drift, bad shutdown or incomplete repetitions
remain collection failures. The helper defect does not excuse a `4xx`
status or missing point. Failed attempts are retained by the existing audit
mechanism; complete failed-physics datasets can still be published honestly.
In particular, an original EW3 Collier `410` remains a structural blocker if
it persists under the final common settings. Correcting the QP helpers does
not excuse that original runtime failure. Any stricter common accuracy policy
must be separately recorded and applied to all roles; otherwise the affected
case remains incomplete.

## Correctness, original discrepancies and performance

Compute all 21 pairwise comparisons among the seven **corrected** targets
(candidate1/6/7/8 and corrected-original1/6/7). Use every point/order and all
four Born/Laurent coefficients. The common point scale `S` is the largest
Born/Laurent magnitude in this corrected cohort; it must not be inflated
using a wrong original value. The original requested tolerance is unchanged.
The combined corrected gate passes only if every comparison passes.

Also compute, over every point/order/coefficient:

- Original backend versus its corrected-original counterpart.
- Original backend versus the corresponding corrected candidate backend.
- The original Ninja/candidate NinjaRS comparison used alongside the literal
  performance denominator.

These original comparisons retain ordinary `passed`/`failed` outcomes and
complete mismatch records, not an expected-failure exemption. The report can
attribute a discrepancy to the demonstrated correction when the
corrected-original and corrected-candidate checks agree; the numerical
failure itself remains visible. No list of known failing points is needed.

The performance validator reuses `runtime_statistics` only after verifying
all original7-target series/pairing rules: identical physical inputs and
settings, one pinned CPU, matching run/repetition pairing, at least seven
samples, immutable executables, and complete v2 proof/source bindings. The
95% interval must satisfy the unchanged 1.05 limit. An unresolved interval,
failed performance result or corrected numerical failure remains outstanding.

## Report touchpoints

The current `ninjars_report.py:correctness` counts only the four candidate
backends; it cannot alone express the new mandatory corrected-original checks.
The v2 reader must independently recompute the seven-corrected-target gate
and verify any retained summary against raw records.

Preserve existing timing-card styling and the literal original-Ninja ratio.
Add a compact provenance/method panel and a per-case table distinguishing:

- Four corrected candidate backends and corrected-original regression.
- Original-source numerical discrepancies, with counts and evidence links.
- Original-Ninja performance ratio/interval and corrected gate status.

The report's overall correctness count must require the complete corrected
cohort and verified correction/proof, not just candidate agreement. An
unverified or missing corrected reference is pending/invalid, never passed.
Original failed comparisons remain visible even when corrected correctness
and performance both pass.

## Required failure tests before use

Use small retained fixtures and fake workers; these tests need no expensive
physics or timing run.

1. Candidate/source-role spoofing: a candidate copy cannot masquerade as
   corrected-original; original archive identity and all non-model source
   bytes must match.
2. Missing one correction, wrong postimage, a third modified model/subprocess
   file, deletion, extra source, escaping symlink or changed parameter card
   must reject the reference.
3. A stale model archive/executable, missing forced model rebuild provenance
   or changed original reduction/DHELAS dependency must reject it.
4. Old v1 proof, stale policy/correction hash, changed proof file, missing
   source role or missing real QP/loop-induced/split-order coverage must reject
   v2 acceptance.
5. A deterministic original/corrected numerical mismatch must not abort
   collection and must remain `failed` in diagnostics. No corrected failure
   may be reclassified by that diagnostic outcome.
6. Missing/duplicated/reordered point or order, dropped failing point,
   nonfinite coefficient, invalid status, missing accuracy or target drift
   must still reject data.
7. A corrected-original mismatch with otherwise agreeing candidates must fail
   the combined gate. Test the common scale independently so an erroneous
   original magnitude cannot relax corrected comparisons.
8. Missing reference backend or its second command, mixed correction versions,
   mixed executable/source hashes or stale numerical summary must reject or
   remain pending.
9. Substituting corrected-original timing for the literal original denominator,
   using unpaired samples, missing repetitions or mixed protocols/CPUs/settings
   must reject the performance gate.
10. Complete datasets with a failed corrected numerical or performance gate
    may be published with that failure retained; they cannot produce an
    overall pass. All old files must remain byte-identical after publication.
11. The v2 report must reproduce every original failure and independently
    recompute corrected comparisons and confidence intervals; forged summary
    booleans and wrong role labels must fail.
12. The optional completed-case extraction mechanism must preserve the failed
    parent and select every complete original prefix case, with the matching
    corrected references; it cannot cherry-pick passing points or backends.

After these tests and independent review, refresh the representative proof,
freeze policy/source identities, and collect final corrected-candidate runs.
Keep the helper implementation milestone separate from this policy milestone.
