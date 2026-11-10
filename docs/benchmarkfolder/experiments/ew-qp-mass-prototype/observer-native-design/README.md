This is the **source-only, uncompiled observer stage** of the private EW QP
mass-transport experiment. The parent 40-file package remains unchanged.
Nothing in this directory authorizes compilation or numerical execution.

The proposed comparison is original/candidate × explicit NinjaRS/TIRust on
the same retained ordinary and threshold inputs. Both arms retain the original
256-bit TIRust library and conditioning guard. Fixed `CTModeRun=4` is an
unchecked transport diagnostic; neither a returned 120 nor cross-reducer
agreement establishes physical accuracy or checked QP acceptance.

## Exact changes and observations

The generated observer adds calls to four source files in each arm:
`CT_interface.f`, `TIRUST_interface.f`, `mp_compute_loop_coefs.f`, and the
matched rebuild of `loop_CT_calls_1.f`. The latter receives no observer edits;
its only original/candidate difference is the frozen paired-mass prototype.
Every pre-existing source line remains in order. New driver, snapshot and
observer files replace the prior interactive benchmark driver. The inherited
`ew_trace.o` remains linked with its original inactive state; no call enables
its decimal, first-occurrence-only logger.

The snapshots record:

- Actual model MP storage immediately after the existing `MP_UPDATE_AS_PARAM`,
  its incoming repaired point, and the true QP-integrand policy.
- Every reached group, split order, CT mode, actual selected reducer and call
  ordinal; all original DP mass arguments/squares, plus candidate MP arguments
  and its already-computed QP squares. The original observer never calculates
  a QP square to fill a comparison field.
- Canonical repaired `MP_PS`, currently consumed `MP_P`, numerator coefficients,
  helper/adapter entry momenta, and actual before/after direction ordering.
- Prepared shifts, complex kinematic matrix, supplied real invariants, masses,
  scale, native return status, errors and successful outputs. Failed native
  outputs are not inspected. Outer results and stability flags cover early
  adapter returns too.
- Public amplitude, accuracy, status and host FPE flags at entry/exit. Native
  and group records also retain the observed IEEE exception flags.

`TRANSFER` extracts binary64/binary128 storage into 64-bit integer words.
Output uses hexadecimal integer formatting, including native error-message
bytes. No new float conversion, square, norm, Gram computation or decimal
floating output is introduced by the observer. Each observer call saves and
restores its own IEEE status, and records any exception-flag change caused
inside that call. Restoration never spans model or reducer arithmetic and
never rewrites MadLoop's FPE common block. One million retained records is a
hard output budget: overflow marks the journal incomplete and continues the
original calculation; analysis then rejects the capture.

## Build and runtime isolation

`prepared.json` binds 210 regular private snapshots: generated sources,
unchanged numerical objects, include/module closure, archives, shared
libraries, cards and resources. No generated makefile is invoked. Direct
compiler recipes rebuild seven objects and link once per arm using the exact
historical GNU compiler and flags; existing optimized polynomial code stays
unchanged. The build is sequential on CPU 8 with a **single 180-second deadline**
for both arms, no retries, and full partial-result preservation. Shared
Ninja/COLLIER libraries and the compiler-resolved static libgcc ordering are
preserved. The private TIRust DSO is `49af12f0…`; installed libraries are never
rewritten.

The runner permits four processes, each at most **120 seconds including
startup and loader checks**, and at most 480 seconds collectively. CPU 8 and
single-thread OMP/BLAS/Rayon settings are fixed. All backend 8/9 work takes the
shared `.tirust-build/numerical-tests.lock`; these scripts fail rather than
queue automatically if the lane is occupied. The historical first-call and
warmup sequence repeats only the two saved points. Every startup status is
retained; the external deadline also bounds the 1002-call warmup-loop ceiling, in addition to the one bootstrap call.
There is no retry, new point set, fixed-mode stability promotion, performance
measurement or automatic checked phase.

Each of the four cases gets its own regular resource/card/input copies. The
only permitted input edit selects explicit backend 8 or 9 and reasserts the
frozen finite/QP settings. All copied inputs must remain byte-identical after
execution; newly created diagnostic/filter outputs are recorded separately.
The runner verifies the loaded path and hash of every private SONAME and
records all other resolved runtime libraries. It requires identical runtime
dependencies across all four cases. A positive exit code alone cannot make a
case complete: output, input and build identity checks must also pass.

## Offline comparison and limits

`analyze.py` requires complete uncapped journals with all 200 group records for
each point, rejects malformed shapes and failed-output reads, and compares
nonmass inputs bitwise. It distinguishes unchanged original DP inputs from
candidate QP squares. Changed masses and their dependent kinematic matrices
are reported separately. Cross-backend scale conventions remain unchanged:
TIRust squares after promotion, while NinjaRS promotes the DP square. This
pre-existing difference is explicitly classified rather than silently fixed.

Retained historical fixed-QP output tokens provide round-trip binary64
amplitude/accuracy/status controls for both original reducers. The old TIRust
record also supplies host FPE flags; equivalent NinjaRS flags were not
retained, so their parity is not claimed. Observer interpretation requires
these available original controls and exact nonmass input checks to pass.
The reported symmetric relative differences are diagnostics, not a substitute
for MadLoop's acceptance test. All refusals, changed statuses and unresolved
threshold discrepancies remain visible.

A separately reviewed automatic `CTModeRun=-1` phase is still required before
any checked-QP conclusion. Stable DP may never enter QP; any diagnostic forcing
of the DP-rejection branch would require a new reviewed source change.

## Source-only checks and later reproduction

```sh
source tools/tirust_env.sh
python -B docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/verify.py --working-root "$PWD"
python -B docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/test_design.py
python -B docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/build.py --show-review-hash
```

`verify.py` also works in a detached copy containing this source index and the
parent package. The optional working-root check verifies private native files
by hash only. Neither verifier compiles or executes a reducer. The build/run
scripts' default and `--show-review-hash` modes only display metadata. Their
explicit `--execute-reviewed-source SHA256` option binds the preparation and
both scripts; possessing this hash is not authorization to run them.
