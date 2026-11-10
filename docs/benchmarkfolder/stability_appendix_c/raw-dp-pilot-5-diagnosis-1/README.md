# First TIRust failures in the closed raw stability pilot

This is a read-only diagnosis of the admitted data capsule `ebd8f1fa…`.
It changes no numerical source, flag, card, point, plot or acceptance policy.
The observations concern host forced-DP calls to the card-selected backend;
actual reducer-internal precision and independent true error remain unobserved.

| Process | First observed group failure | Groups | Tensor topology | DP software flag |
|---|---|---|---|---|
| `u u~ > z g g` | Primary point 25, method 1, call 78 | 59, 61 | Five massless propagators, rank 4, 70 coefficients | false → true |
| `u u~ > z z g g` | **Bootstrap** point 2, method 0, call 2 | 408, 422 | Six massless propagators, rank 5, 126 coefficients | false → true |

The generated calls associate the first pair with loop diagrams 90/92 and
wavefunction orders `[1,2,5,4,3]` / `[1,2,5,3,4]`. The second pair corresponds
to diagrams 538/552 and orders `[1,2,5,3,6,4]` / `[1,2,5,3,4,6]`. These are
massless **internal** propagators; the external Z bosons remain massive.
The exact call excerpts and full source hashes are in `source-excerpts.json`.
Generated-source hashes were joined to the full numerical compilation records
retained in the admitted preparation receipt.

For both first failures, active numerator components are finite and nonzero,
the native group stability flag is false, and the six saved loop-result words
are all positive zero. No recorded call in either lane reports nonfinite
numerator or loop-result components. The zero result does **not** establish
that the internal tensor calculation remained finite: the failure handler
deliberately replaces failed results with zero.

The Fortran adapter begins with `STABLE=.FALSE.`. Unsupported inputs,
initialization/cache-budget failures, a momentum-conservation guard, a nonzero
bridge status, or a nonfinite tensor can return without setting it true.
The caller sets `FPE_IN_DP_REDUCTION=.TRUE.` and returns a zero contribution.
A later nonfinite numerator contraction also sets that flag and zeros the
result. Rust additionally converts unstable/master-failure lanes and nonfinite
packed tensors into failure statuses. The pilot did **not** record the bridge
status, its `LAST_ERROR`, intermediate tensors or hardware IEEE exception flags.
Therefore the exact Rust/adapter failure reason cannot be recovered from these
observations. `FPE_IN_DP_REDUCTION` is a software failure latch, not proof of a
hardware floating-point exception.

The latch is initialized false once. The bound MadLoop source resets it when
automatic stability handling advances to another backend/precision. That path
requires `CTModeRun=-1`; the positive forced modes used here bypass it. Neither
the passive pilot observer nor the event cache clearing resets the flag.
Consequently a later call with stable active groups can retain the earlier
failure flag. The recorded before/after bits distinguish this effect:

- Zgg has 15 primary calls with failed groups, at only five physical points:
  25 (59/61), 34 (61), 45 (63), 70 (59/61), and 84 (58/63/70).
  These comprise 27 failed-group observations across the three probes.
  Another **213 primary calls have no failed groups and only the latched flag**.
  Point 26/method 1/call 81 is the first such example. There is exactly one
  observed DP false→true transition and no QP transition.
- ZZgg has four bootstrap calls with failed groups (35 group observations).
  Its first primary call, point 1/method 1/call 6, has no failed group but
  inherits the bootstrap latch. The first primary group failure is point
  2/method 1/call 9, again groups 408/422. Of the 254 completed primary calls,
  212 have failed groups (2,438 observations at 71 points), and 42 have only
  the latched flag. There is one DP transition, already in bootstrap call 2.

The first ZZgg bootstrap failure returns initialization status 130 and emits
no W03 warning in that call. Its group snapshot still records the failure;
absence of a warning is insufficient. The first matching primary group failure
does emit W03 and returns unchecked 110. All 300 saved Zgg and 254 saved ZZgg
primary order-zero rows return 110 with finite saved amplitude coefficients.
This cannot certify a reduction whose failed contributions were replaced by
zero, nor can it establish backend identity or accuracy.

ZZgg then reaches its bounded execution deadline. The journal ends with an
unmatched before-call record for point 85/method 3/call 260; no matching group
summary, amplitude row or call-end marker was collected. That interrupted call
is not a completed numerical failure. The retained timeout and incomplete
stream leave all 100 intended ZZgg/TIRust point metrics unavailable under the
frozen policy. Zgg retains 24 available and 76 unavailable metrics; the latter
are **not 76 independent new reducer failures**. The published plots correctly
retain unknown bounds and mask the corresponding Ninja ratios.

`diagnosis.json` includes selected exact journal/console lines with original
line numbers, complete count reconciliations and the terminal failure fields.
Reproduce those derived observations without running numerical code:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/stability_appendix_c/raw-dp-pilot-5-diagnosis-1/diagnose.py --data-root docs/benchmarkfolder/stability_appendix_c/raw-dp-pilot-5-1
```

Add `--verify` to compare the fresh read-only derivation with `diagnosis.json`
and print a compact verification result.

`standalone-replay-proposal.md` describes a small future diagnostic. It is
source-only and has not been compiled or executed. COLLIER and its reported
cache bug are outside this diagnosis and remain unchanged.
