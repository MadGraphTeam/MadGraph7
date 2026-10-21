# Unapplied production candidates and final proof scope

Both patches are source-only proposals. `git apply --check` passed against the
production files bound by `prepared.json`; neither patch was applied or built.
Root selects one only after the already frozen numerical/performance studies.

| Candidate | Factory selection | Validation evidence |
|---|---|---|
| `denominators.patch` | `.with_native_denominators()` | Exact raw values, tracked precision, signs and ASCII for all 1,392 fixed requests |
| `full.patch` | `.with_native_kernels()` | Same binary128 return values and signs for every fixed request; disclosed tiny raw/precision/ASCII differences with 256/384 convergence |

Each changes only `vendor/ninjars/Cargo.toml` and the existing factory creation
in `vendor/ninjars/src/lib.rs`. The additional `native-kernels` feature is
required for either builder method and implies pinned upstream `rust-codegen`.
The validated private study already resolves these features with the identical
Cargo.lock; no pin, lockfile, scalar provider or public API change is needed.

The builder methods are infallible policy selections. Existing factory and
`into_amplitude` errors still map to the same explicit error status through the
existing contained worker path. The upstream partial catalog controls native
coverage and retains its unsupported-family interpreter fallback, including
rank-n+2 triangle-to-bubble corrections. Do not add bridge-side silent retries
or replace failures with cached output.

The production context cache remains keyed by `(n,rank)` at its existing
precision. The selected builder runs when each entry is created, including a
late topology/rank. Numerical inputs, tensor coefficients, result resets,
geometry, master objects and the scalar threshold `1e15 / 2^112` remain exactly
the existing implementation. No environment variable, mutable policy field,
research selector, timing harness or attribution replacement is included.
Upstream source and `UPSTREAM.md` attribution stay intact.

Root should apply the selected hunk while preserving any separately approved
panic-contract test declaration, tensor optimization or numerical fix added
after this proposal's baseline. The candidate source snapshots are evidence,
not files to copy over newer production wholesale.

## One final production proof binary

Keep these production test modules as they stand:

- `src/transport_contract_tests.rs`: binary128 value/zero-sign parsing contract.
- `src/guarded_tests.rs` and `src/retry_tests.rs`: the existing DP/compensated
  contracts and recovery controls.
- Root's separately prepared `src/panic_contract_tests.rs`, once applied.
- The four allocation-contract tests, if that tensor change is promoted.
- Existing `tests/data/convergence.json` and the ordinary rank-five convergence
  fixture in `lib.rs`: retain their original values and comparisons.

Use the production `MpState` directly for final retained MP proofs. Do **not**
promote `StudyKernelPolicy`, `with_study_policy`, test backend fields, the
three-policy harness, fixed timing helper or its controllers. A final proof
module needs no policy parameter: it exercises whichever production factory
root selected, plus the actual final tensor and parser in the same binary.

Exact existing source entry points to adapt are:

| Final proof entry point | Source already reviewed and executed | Minimal adaptation |
|---|---|---|
| `mp_backend_proof_tests::all_stored_requests_preserve_binary128_returns` | `source/src/native_study_tests.rs::all_requests_three_policies`; existing `vendor/ninjars/src/transport_replay_tests.rs::all_stored_requests_retain_outputs_after_transport_fix` | One `MpState::new`, all 1,392 original records in order; retain every original expected ASCII field and every actual field/status; compare represented binary128 values and zero signs without rewriting the historical strings |
| `mp_backend_proof_tests::all_stored_requests_converge_at_256_and_384` | `source/src/native_study_tests.rs::discrepancies_at_256_and_384`; `analyze.py::convergence` | One production state per precision, the same full original stream at both precisions, plus original complex-mass pentagon; no discrepancy-selection flag or alternative policy; preserve raw exact dyadics/precision/sign and apply the fixed 1e-31 common-Laurent norm screen and 1e-60 finite-split screen |
| `mp_backend_proof_tests::analytic_poles_reuse_and_returned_shape_errors` | `source/src/native_study_tests.rs::policies_rebind_fail_recover_and_converge`; `supplement/source/src/native_supplement_tests.rs::returned_shape_error_retains_entries_and_late_policy` | Remove the policy loop, keep 256/384, all five analytic fixtures and both complete-complex-coefficient count errors; verify existing and late entries, returned errors, no stale outputs, unchanged inputs and fresh-state recovery |

The existing captured historical replay asserts byte-exact ASCII output and
must remain unchanged as historical evidence. It passes denominator-only on
this capture. For full native it deliberately reports the four known last-digit
ASCII changes, although all represented binary128 values are exact. Do not
claim that historical exact-ASCII test passes for full native, weaken its
assertion silently, or substitute new expected strings. The separately named
binary128-return proof states the actual Fortran return contract and retains
all ASCII differences for review.

The all-request convergence proof should use the full stream at both
precisions rather than a selected subset, so the new final comparison does not
introduce different topology-construction histories. The already observed
microscopic common scalar-master history sensitivity remains disclosed; this
does not establish exact raw 256-bit history independence or correct rounding.

`final-tests.patch` now implements that promotion map as one generic test-only
module, `final_precision_tests.rs`, without a policy selector. It adds one
ordinary analytic/reuse/error test, one ignored combined historical replay
and all-request256/384 convergence test, and one ignored replay of the existing
seven physical fixtures in all three modes. This source is prepared and has not
been compiled or executed; numerical_review owns its independent review and
the single final binary's build/run integration.

The exact new ignored entry point is
`final_precision_tests::retained_requests_replay_and_converge`. It requires
`NINJARS_CONVERGENCE_INPUT` to name the original1,392-recordJSONL capture and
`NINJARS_CONVERGENCE_OUTPUT` to name a new, nonexistent raw JSONL destination.
It retains1,392 replay rows followed by1,392 convergence rows and one summary.
Each observed result includes raw represented dyadics, tracked precision,
sign, status, original outputs and input-immutability checks. Failed records
remain in the stream before the final assertion; it never rewrites historical
expected ASCII fields. The summary stdout prefix is
`NINJARS_CONVERGENCE_CAPTURE_SUMMARY `, which is deliberately distinct from
the installed-archive reader's exact existing `NINJARS_CONVERGENCE ` sentinel.

The ordinary entry point is
`final_precision_tests::analytic_poles_reuse_and_returned_shape_errors`. It
exercises the selected backend at both precisions through the five fixed
fixtures. The original ordinary pentagon test and the ten mandatory test names
remain unmodified and continue to satisfy `tools/ninjars_precision_v2.py`.

The other ignored entry point is
`final_precision_tests::seven_physical_fixtures_all_modes`. It embeds the seven
unchanged fixture/reference pairs already used by `tests/replay.py` and runs
all21lanes through native DP, compensated DP and generic MP in the same final
library-test binary. It prints63raw `NINJARS_FIXTURE_RECORD ` rows and one
`NINJARS_FIXTURE_SUMMARY ` record. The existing1e-9 comparison and MP rational
basis convention are preserved; no second example executable is needed.

Retain each failed proof before any repair. The final generated-process
campaign remains required even if every Rust proof passes.
