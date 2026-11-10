# Retained passing revision-2 component fixture

Attempt 1 completed with `passed=true`. All 5,376 bulk original/candidate pairs
passed the prescribed finite-upper mirroring policy, caller-input-word checks
and IEEE-state comparisons. The four configurations each contain 1,344 pairs:
binary64 and binary128 with historical flags and with the separately labeled O3
supplement. All four rounding modes were exercised, with no skipped flags.

All 48 supported trap comparisons passed. In the 32 invalid-trap pairs with
upper/lower signaling NaNs, both components returned and passed their recorded
matrix/state comparisons. The other 16 pairs agreed by both terminating with
SIGFPE under inexact trapping; those pairs have no post-call value or state
evidence. The two negative controls discriminated: removing the whole-matrix
finite guard caused 360 matrix-policy failures; removing the equality guard
caused 72. These expected control failures are retained in full.

All 124 owned commands were reaped: 92 returned zero and 32 terminated with
SIGFPE. The 139 protected input identities matched before and after the attempt.
The frozen source proposal remains unchanged, including its historical
pre-execution metadata. The prior revision's failed qualification remains
separately retained and hash-bound.

The frozen plan carries stale copied byte sizes for seven changed v2 source
inputs; two index rows omit sizes. Its authoritative SHA-256 identities all
match the frozen sources. The runner checks those hashes, so this discrepancy
does not alter the executed source identity or the recorded outcome.
`plan-size-observations.json` records the historical sizes and the frozen source
sizes explicitly. The initial packaging preflight rejected the stale size
metadata before creating payloads; its helper and receipt are preserved under
`preflight-failure-1/`. Only this new retention helper was corrected. The plan,
source indexes and scientific execution were not changed or rerun.

This result qualifies the tested **extracted components and synthetic corpus**
under the recorded compiler/environment. It does not establish full adapter
transport, actual model initialization, reducer behavior, physical amplitudes,
QP transport/cache-key parity, rescue rates or full 100-point acceptance. No
reducer or amplitude was invoked by these fixtures. There is no performance
claim. Matrix-policy agreement permits the specified finite lower-triangle
replacement; it does not mean every output matrix equals the original matrix.

The package contains exact execution/progress JSON and all 124 command stdout
files, including compiler/linker logs, loader listings, bulk/mutation raw-word
frames and trap diagnostics. Gzip uses empty filenames, mtime zero and level six;
the index records both raw and compressed hashes/sizes. The execution manifest
binds six executable hashes and actual command/process metadata. The separate
artifact metadata additionally records ten object hashes observed during
retention. No binary, library, cache or core dump is copied.

Default verification checks retained bytes and their joins to the execution
record. `--reparse` also runs only the unchanged, hash-bound analyzer's Python
integer/text functions against retained gzip logs, requiring exact agreement
with all four bulk results, 48 trap comparisons and two negative controls.
The analyzer is hashed before and after replay. Neither command invokes the
fixture runner, compiler or a native workload. They use this package and frozen
repository-relative bindings; ignored build trees and installed libraries are
unnecessary. Historical absolute paths remain record identifiers.

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/ew-invariant-symmetry-prototype/v2/fortran-fixtures/qualification-1/package.py
python3 -I -B docs/benchmarkfolder/experiments/ew-invariant-symmetry-prototype/v2/fortran-fixtures/qualification-1/package.py --reparse
```

`--package` retains the completed attempt once and checks protected inputs and
artifact identities before/after retention. Any partial packaging failure is
preserved without automatic retries. Verification success and component success
are reported separately; full adapter/physical qualification remains false.

Execution SHA-256:
`bc0049e3c2212f71d3372ff772ec1b47737ff73b9fc4c205530691ab8ba4dc91`.
Frozen analyzer SHA-256:
`602fec89e78ccd22105d21fb323e93608ab35ce0e86b01c751dfeb2933c329ed`.
