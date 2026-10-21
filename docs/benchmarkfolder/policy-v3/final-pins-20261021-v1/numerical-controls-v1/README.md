# Final dependency numerical controls

This capsule retains the exact observations for NinjaRS
`aa6a6307dd081dc3714f9accebc0df9e5561ae73` and oneloopmaster
`8ec256fc26cb361b20e10f59c19beb35717ef32c`. The 45 tested bridge files
match MadGraph commit `856e76e669618ec8d0f799b9c8e6f86819dcd986` and the
archive installed by the actual `install ninjars --update` command. Original
build/test records retain their earlier dirty checkout identity; a separate
source equivalence receipt establishes the later committed revision.

The focused bridge controls pass. These observations do not replace the
full MadLoop process, stability, or timing campaigns.

## Results and retained failures

- The ordinary Rust suite passed all 24 tests with `fenv-test-support`,
  including directed rounding, binary128 transport, transactional output,
  reinitialization, invalid input, complex masses, and 256/384-bit convergence.
- The actual installed archive passed 64 C ABI calls: 63 physical fixture
  evaluations in native DP, compensated DP, and MP, plus a complex pentagon
  containing information beyond binary64. Its full generated-process runtime
  identity is `a7577a9a33292ffb5d675ccf1b9c3667db3ab3613d77ccd270f1efc9e9802008`.
- Every one of 1392 retained requests was replayed, in its original order, at
  both 256 and 384 bits. All succeeded. Complete Laurent, CC, and R1 values
  independently converge within `1e-31` of the common Laurent scale, and
  `finite = CC + R1` holds within `1e-60`. The largest complete Laurent
  difference is `2.731e-62` of that scale.
- The original Rust replay test **exited 101** because 468 outputs differ from
  historical binary128 values (482 ASCII differences). This remains a failed
  historical identity test. The separate exact-arithmetic convergence result
  passes. The largest historical complete Laurent change is `1.476e-34` of
  the common Laurent scale; the separate basis-dependent R1 diagnostic changes
  by as much as `4.784` times that scale. MadLoop consumes the complete Laurent
  result; it does not add the returned R1 diagnostic again. Historical R1 is
  not an independent physical oracle.
- All 24 soft-D0 permutations passed at both precisions, with a `1e-13`
  independent binary64 reference comparison and `1e-31` permutation agreement,
  including failure sentinels and successful provider reuse after rejection. Additional retained
  controls cover 27 calls at ranks `n+2`, complex masses, and the compiled
  Fortran interface.
- The direct scalar-provider planar sentinel test **exited 101**: `ns9210`
  incorrectly succeeds at 256 bits for both signed-zero lips despite mutually
  converged 512/1024-bit references. `ns11857` safely rejects at 256 and 512
  bits without touching output. All four comparisons are retained. On an
  exact real-momentum realization of `ns9210`, the full reduction bridge's four
  DP calls agree with the higher-precision scalar reference within `3.034e-13`
  (required `1e-10`), and both MP calls reject without committing output.
  This does not resolve the underlying direct scalar-provider finding.

Tracked precision is an internal estimate, not a certified error bound.
Sixteen of 22,272 ASCII output fields round to a neighboring binary128 value
relative to rounding the exact raw Float directly. Their largest difference
is `4.196e-36` of the common Laurent scale. Input recovery of the original
113-bit values is tested separately. Forty-digit printed references support
the stated `1e-31` gate; they cannot independently reproduce the scalar test's
reported internal 151-digit agreement.

## Read-only verification

Run from the MadGraph repository root with Python 3. No compiler, Rust
library, numerical engine, output writer, or private build directory is used:

```sh
python3 tools/ninjars_final_numerical.py \
  docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/numerical-controls-v1/manifest-v2.json
python3 tools/ninjars_precision_v2.py validate \
  docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/installed-precision/run-v1/result-portable.json
python3 -m unittest tests.unit_tests.various.test_ninjars_final_numerical
```

The numerical API is
`ninjars_final_numerical.verify(manifest_path, installation=None, inputs=None)`.
It checks every artifact's bytes and digest, decoded gzip digests, all 45
source files, the concrete installation's source/archive/interface inventory,
the unchanged build and process custody, all 2785 raw replay records, exact
rational convergence and historical differences, and preservation of both
failed diagnostics. Passing another installation checks its complete source
and installed-file inventories against this capsule. `inputs`, when supplied,
collects the checked file digests for report provenance. Checks remain enabled
under optimized Python.

`manifest-v2.json` binds all observations and the verifier version;
`verification-v2.json` is a retained view that the verifier recomputes. The
initial `manifest.json`, `verification.json`, and `observations/verifier.py`
remain unchanged as historical authority. Version 2 adds explicit comparisons
against original metadata to verify that only declared relocation fields
changed. Files
under `observations/` preserve original bytes and original paths in metadata.
Large streams use deterministic gzip with both compressed and decoded hashes.
`source-v2` retains the earlier focused-test source; `source-v4` is the final
complete source inventory. The direct provider, exact geometry, and independent
audit receipts are separate from full MadLoop acceptance.

The installed proof's original `prepared.json`, `result.json`, raw logs and
process handoff are unchanged. `prepared-portable.json` and
`result-portable.json` explicitly derive a metadata-only relocation of the
same 64-call observation. Their transitive ordinary-suite logs and build
records are retained here. Compiler outputs and executables are identified by
their actual build hashes; ELF build products are intentionally not shipped.

## Numerical reproduction

Rebuild the retained final source with the committed Cargo lockfile and the
documented Linux toolchain. Use one evaluation thread and an exclusive CPU
for scalar and MP tests. `vendor/ninjars/Makefile`'s `check` target enables
`fenv-test-support`; the release test profile used here was also recorded.

To repeat the complete replay, decompress the manifest's
`original-replay.jsonl` artifact into a fresh working file. Set
`NINJARS_CONVERGENCE_INPUT` to that file and
`NINJARS_CONVERGENCE_OUTPUT` to a nonexistent output file inside the repository.
Run the release Rust test with:

```sh
cargo test --release --locked --features fenv-test-support \
  --manifest-path vendor/ninjars/Cargo.toml \
  final_precision_tests::retained_requests_replay_and_converge -- \
  --ignored --exact --test-threads=1 --nocapture
```

The retained historical-identity assertion intentionally remains present, so
with these dependency pins this run exits 101 after writing all requests and
the convergence footer. Preserve that result and independently verify the
new raw stream; do not replace historical outputs or filter failed records.
The separately ignored soft-box and planar diagnostic tests are in
`vendor/ninjars/src/upstream_master_tests.rs`. The planar diagnostic is a
known failed provider regression, not a passing ordinary test.

New actual installed-archive repetitions can use
`tools/ninjars_precision_v2.py`'s explicit `prepare`/`execute` workflow with a
new report/work namespace and a compiled installation. Do not overwrite these
observations. Recorded cold durations are diagnostics only and are excluded
from the warmed MadLoop performance comparison.
