# Isolated native Float256 study

The fixed numerical study and supplemental state controls pass. Request timing
is prepared and remains held until allocation's complete-process study finishes
and the root grants a quiet window. No native dispatch change has been applied
to production; its default and public entry points remain interpreted.

All arms use the same corrected signed-zero parser, original generic tensor,
113-bit recovered input values widened to 256 bits, scalar provider and
on-shell threshold. The pinned NinjaRS, oneloopmaster and Symbolica revisions
remain those in `prepared.json`. The allocation experiment is separate.

## Fixed numerical results

| Check | Result |
|---|---|
| Ordinary corrected-transport tests | 18 passed |
| Historical requests per policy | 1,392; all successful and finite |
| Output fields per policy | 11,136 |
| Interpreted versus historical ASCII controls | Exact |
| Denominator-only versus interpreted | Exact values, precision, signs and ASCII |
| Full-native raw value differences | 958 requests; 5,122 scalar fields |
| Full-native precision differences at 256 bits | 8 requests; 16 scalar fields, each one bit lower |
| Full-native sign differences at the same working precision | None |
| Full-native ASCII differences | 4 last-digit fields, representing two duplicated requests |
| Returned binary128 values and zero signs versus interpreted | All 11,136 fields exact for each native policy |
| Worst full-native Laurent difference/common norm at 256 bits | 2.4567604308e-71 |
| Discrepancy controls at 256 and 384 bits | All 958 requests plus the retained complex-mass pentagon |
| Worst 256-to-384 Laurent difference/common norm | 1.0132439743e-63 |
| Worst full-native versus interpreted difference/common norm at 384 bits | 4.9692043982e-110 |
| Full Laurent convergence and finite=CC+rational screens | Passed |
| Analytic tadpole and bubble controls | Passed for all policies at both precisions |
| Supplemental retained-state/late-entry tensor-count errors | All 12 passed |

The value screen is the preregistered 1e-31 common Laurent-component norm. It
assesses the binary128 return contract on these inputs, and does not establish
correct rounding of every internal 256-bit operation or replace the generated
MadLoop physics and stability campaign. The raw precision and ASCII differences
are retained, even though independently rounding the return records to 113 bits
gives identical represented binary128 values. The rounding audit checks the
normal binary128 exponent range and zero signs separately.

The original short-shape fixture caught a missing imaginary component panic and
replaced the private state. Its results remain intact and are labelled as panic
recovery. The separate supplemental fixture supplies two complete complex
coefficients to a rank-zero tensor. It reaches the returned tensor-count error,
retains the actual entry, and proves exact fresh-state recovery without a
panic reset, for both an existing entry and an entry created by the failed call.

## Retained history sensitivity

Rerunning only the selected discrepant requests in fresh contexts changes tiny
raw finite/CC values in 20 requests, common to all three policies. Statuses,
formatted ASCII output, tracked precision and signs remain exact. The largest
change/common norm is 5.6659306854e-65. The complete input-order hashes and all
20 original/selected/384-bit observations are in `history-diagnostic.json`.
This is disclosed as microscopic internal history sensitivity; exact raw
history independence is not claimed.

The source audit found no stale numerator or accumulated-result path. A
plausible separate cause is scalar-master expression construction using plain
Symbolica symbol ordering, while NinjaRS kernels deliberately use positional
evaluators. The full stream first constructs a bubble; the selected stream
first constructs a triangle, changing symbol-registration history. This is a
source hypothesis, not a traced proof of the differing instructions. The
separately reviewed Symbolica operand-precision fix was not adopted here.

The physical pentagon's tiny imaginary residuals change sign between 256 and
384 bits while shrinking by about 40 decimal digits. Interpreted and
denominator-only values match the pre-existing `convergence.json` expectations.
These are vanishing nonzero cancellation residuals, not negative-zero changes
or a new native branch difference.

Full native corrections use a scaled complex reciprocal, whereas interpreted
evaluation normally uses the direct norm-squared inverse. Their numerical
operation ordering and tracked precision may therefore differ. Source review
explains this mechanism and its precision propagation; it does not attribute
each changed scalar field to an instrumented individual instruction.

## Build and pause provenance

The original compiler was resumed after two user-requested pauses. Its shell
and Python parent had exited; exact Cargo/rustc process identities were checked
before resuming. `build-recovery-result.json` binds Cargo's successful JSON
completion and artifact hash, and explicitly records that the lost parent's
wait status is unavailable. No paused interval is a performance sample.

The supplemental test-only relink unexpectedly reused the original executable
disambiguator despite a separate local manifest path. The original executable
was preserved before final link. The controller's collision assertion remains
as a failed packaging check; the successful new artifact was copied separately
and the original was restored byte for byte. Both records are retained in
`supplement/`. The supplemental binary passed the same 18 ordinary tests and
the 12 new state controls. No additional compile was used to hide the collision.

## Files and reproduction

- `PLAN.md`, `prepared.json` and `study.patch`: original immutable source scope.
- `run-*/`: complete original stage outputs and source/runtime bindings.
- `independent/`: offline independent focused, replay and convergence audits.
- `transport_equivalence.py`: exact-rational binary128 return comparison.
- `SCREENING-PLAN.md`: fixed seven-triplet request study; this is not full
  MadLoop evaluation or a final performance gate.
- `supplement/`: separate source, build recovery, ordinary tests and 12 controls.
- `run_supplement.py`, `screen_preflight.py`, `analyze_screening.py`: independently
  reviewed screening controller, host qualification and fixed statistics.

Large raw and derived files will be archived only after sampling finishes,
using `archive_large.py` with deterministic gzip and both compressed and
original hashes. Its materializer restores the original paths for the frozen
analyzers. No raw observations or failed attempts are discarded.
