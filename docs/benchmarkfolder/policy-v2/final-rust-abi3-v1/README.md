# Final ABI 3 Rust proof: retry prepared

This controller collects the final selected production policy. Its first attempt
at `c2b48e988fe6e3c6ba9403b67ec4d26fa5fa9ccb` is preserved in `proof-v1`:
compilation found four test-only borrowed-operand errors and no tests or numerical
calls ran. Root applied the owned-left fixes. The retry controller's Python AST
and `--help` readiness checks passed on CPU 6; its new cache option remains
unexecuted. Root must first commit the corrected source and controller on
`madloop_ninjars`, provide that exact revision and
the immutable original replay-input SHA256, and release compilation plus the MP
slot. The current preparation is not acceptance evidence.

The controller builds **one** release libtest target with `cargo test --release
--locked --lib --no-run --features fenv-test-support --message-format=json`,
`CARGO_BUILD_JOBS=1`, on CPU 190. It builds only committed `vendor/ninjars` bytes
from a fresh Git archive under `.benchmark-work`; sibling reference clones are
never written. It includes the entire tracked bridge map, including interfaces,
Cargo files, Makefile, all test modules and their fixture data. An unexpected
archived symlink is rejected rather than silently omitted. The controller itself
must also match the supplied commit. Each numerical command is serialized with
`--test-threads=1` and checked for complete child-process exit before the next.

The same exact executable runs three commands:

1. All ordinary tests, including transport, panic recovery, allocation oracles
   actually selected for production, analytic poles/state reuse and the existing
   sole pentagon `NINJARS_CONVERGENCE` sentinel. The ordinary sentinel's eight
   ASCII outputs are checked against 384 bits with the existing `1e-31` bound
   normalized only by the six complete Laurent fields.
2. Ignored exact test
   `final_precision_tests::seven_physical_fixtures_all_modes`: seven unchanged
   physical fixtures, 21 lanes and 63 native-DP/compensated-DP/generic-MP calls.
   The existing cross-backend threshold and basis-dependent MP rational
   diagnostic convention are preserved. No replay example is separately built.
3. Ignored exact test
   `final_precision_tests::retained_requests_replay_and_converge`: all 1392
   original requests, in identical full history at 256 and 384 bits. Every raw
   result/status, tracked precision and sign is retained. The controller
   independently recomputes complete-Laurent and rational convergence plus the
   finite=CC+rational identities from exact dyadic fractions. Physical errors
   use the six complete Laurent components; basis-dependent CC cannot enlarge
   the scale. It reports ASCII changes separately from the Rust test's original
   binary128-value/signed-zero replay assertion and never rewrites expectations.

Nonzero internal sign-change counts remain explicit review items. A passing
convergence bound alone does not establish exact internal history independence,
correct rounding or semantic equivalence of raw signed cancellation residuals.
No timing or generated-process acceptance claim is made by this unit proof.

After the explicit release, root can run from the repository root:

```sh
source .ninjars-build/dev-env.sh
taskset -c 190 python -B \
  docs/benchmarkfolder/policy-v2/final-rust-abi3-v1/collect.py \
  --revision FINAL_FULL_COMMIT \
  --replay-input docs/benchmarkfolder/experiments/generic-tensor-plan/capture/actual-mp-requests.jsonl.gz \
  --replay-input-sha256 5e6fd2c0bad620460af7742e19d4f34b1f2ce281af22e3623001c7ba74411312 \
  --replay-uncompressed-sha256 1fafb19d4b534c3e222e5b265dfd8b139f262d3353f847a5c39955c468fa9f4d \
  --work .benchmark-work/policy-v2/final-rust-abi3-v1-work-v2 \
  --report docs/benchmarkfolder/policy-v2/final-rust-abi3-v1/proof-v2 \
  --cargo-target-dir .benchmark-work/policy-v2/final-rust-abi3-v1/target \
  --resources-released
```

Every work/report namespace must be fresh. Failures and partial raw captures are
preserved, with a failed attempt record and a manifest; no failure can produce a
successful `unit-proof.json`. Build products stay private. The executable can
be removed after all readers have consumed the successful proof: its exact hash
and build record remain retained, as required by the existing precision reader.
The input hashes above are the existing native preparation's capture record.
Only an authorized collection decodes the retained gzip into private working
storage, checking both compressed and exact original uncompressed bytes before
any engine runs. This does not generate or change phase-space/numerator inputs.

`--cargo-target-dir` optionally reuses only a disposable dependency/build cache
under repository `.benchmark-work`; otherwise it defaults to `WORK/target`.
`request.json` records its actual resolved target/build directories and whether
the directory already existed. Every retry still creates a fresh committed
source archive, working directory and report and records the actual resulting
executable. Cargo validates fingerprints for the final changed test crate and
locked dependencies. Reusing the failed first attempt's cache does not modify
its source snapshot, logs or sealed failure manifest. This is a cached build,
not a measurement of clean compilation time.

## Proof contract shared with install/export and precision readers

`unit-proof.json` uses `schema: final-rust-abi3-v1`, exact `revision`, `exit_code`,
`stdout`, `stderr`, `build_record`, `executable`, and `bridge_source_sha256`.
All bound records have repository-relative `path`, `sha256`, and `bytes`.
The top stdout/stderr are the ordinary-run logs alone, keeping exactly one
`NINJARS_CONVERGENCE` prefix for the unchanged
`tools/ninjars_precision_v2.py:unit_evidence` reader. The build record repeats
the exact executable, revision and complete source map. Source-map keys are
relative to `vendor/ninjars`, and their key set must exactly match the final
installation source archive.

`named_runs` has `ordinary`, `fixtures`, `replay` and `convergence`. Each carries
its own exit status, source map, revision, exact executable and stdout/stderr,
`observed_test_names` (all libtest names, including ignored tests),
`passed_test_names`, `ignored_test_names` and actual
`tests: {passed, failed, ignored, filtered_out}` counts. Replay and convergence
refer to the same successful third command and raw capture. They are not counted
as two executed commands. `total_executed_test_passes` is the ordinary passing
count plus the two actually executed ignored tests; it is never a maximum or a
sum that counts the shared capture twice.

The fixture summary reports `fixtures=7`, `lanes=21`, `calls=63`,
`failed_calls=0`, `passed=true`. Replay summary fields are `passed`, `records`,
`fields`, `historical_binary128_failures`,
`historical_ascii_changed_records`, `historical_ascii_changed_fields`.
Convergence summary fields are `passed`, `records`, `fields_per_precision`,
`production_bits`, `reference_bits`, `tolerance`, `failed_records`,
`sign_changed_records`, `raw_comparisons_recomputed` and the stated physical
normalization/split convention. The source test's claims are checked against
retained exact raw records, including original input sequence and raw sign
inventories. The source test separately asserts the original binary128 outputs
and signed zeros; the controller does not claim a second binary128 decoder.

`proof-manifest.json` has schema `final-rust-abi3-proof-manifest-v1`, `revision`,
`files: {REPORT_RELATIVE_PATH: {sha256, bytes}}` and `count`; it covers all
retained report files except itself. Install/export preparation accepts explicit
`--revision FULL_COMMIT --rust-proof PATH_TO_UNIT_PROOF.json
--rust-proof-manifest-sha256 SHA256_OF_PROOF_MANIFEST`, and verifies the map
against its own final source archive. Existing historical source and evidence
seals are preserved. The pending source module must be applied and compiled
before any current ordinary-test count or final numerical outcome can be known.
