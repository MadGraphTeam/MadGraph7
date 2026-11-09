# Stability audit of the retained N4R5 stress failure

**The default MadLoop stability check misses correlated DP errors at this
synthetic tensor point. Explicit QP resolves the discrepancy.** This is retained
failure evidence, not an all-pass qualification or a physical-amplitude result.
The production bridge, default cards, and benchmark policy were unchanged by
this audit. The initial bridge DSO was the debug build identified in
`build-metadata.json`; the `release-*` records replay the same audit against the
installed release DSO, SHA256
`49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12`.
Both profiles give the same native diagnostic values. No runtime-performance
conclusion follows.

The fixture is `li-rows.json LI-32 N4-massive R4 row 57 R=5 lane 44` in the pinned
TIRust `gate_passed_lanes.json` (SHA256
`4ebf900f99805d7894a7c484f1755c896307cea97e032942937b88cd3f3459e8`).
The previously recorded primary DP finite symmetric error was `3.479475e-6`,
above the independent `3e-6` acceptance bound. The raw C-ABI audit additionally
tests propagator reversal, cache bypass, exact coordinate rotations, and a
permutation placing the nearly zero shift first.

`build_harness.py` compiles the actual `TIRUST_interface.inc`, `SWITCH_ORDER`,
kinematic-matrix helpers, and `COMPUTE_ACCURACY` from the current MadLoop
templates. A small diagnostic driver contracts the returned moments in
MadLoop coefficient order. This initial harness executes CT modes 1/2 and 4/5,
including the actual check-pass flag; it does **not** execute the complete
physical `SLOOPMATRIX` flow. The native accuracy routine was also used to verify
every independent Python estimator calculation.

For the generated adapter, the nearly zero final shift is placed first and
removed by the exact polynomial change `q'=q+p0`, `N'(q')=N(q'-p0)` before
binary64 rounding. The loop increments are cyclic shift differences. QP
promotes the same binary64 inputs, reconstructs invariants in QP, and contracts
in QP. The entire transformed input is in `raw-and-native.json`; this comparison
does not assume that binary64 and QP kinematic reconstruction are identical.

| Native adapter result | Finite coefficient |
| --- | --- |
| DP original, CT1 | `524.1991612252314 + 166.3023424926655 i` |
| DP reversed, CT2 | `524.1909523498261 + 166.32034801927102 i` |
| QP original/reversed, CT4/5 | `524.0351881183658 + 166.5453035500552 i` |
| Supplied reference | `524.0351881183657 + 166.545303550055 i` |

The native primary DP finite symmetric error against the same-input QP result
is `2.665154525962104e-4`. The QP whole-lane relative error against the supplied
binary64 reference is `4.1672708817260766e-16`.

The tensor is complex; MadLoop's matrix-element estimator consumes real
numbers. This audit therefore tests its real and imaginary projections
separately, with all three Laurent coefficients. It adds neither a physical
Born amplitude nor R2/counterterms. For each component the production routine
uses the range and midpoint across evaluations, then weights the relative
ranges by the absolute midpoints and divides by one third of the absolute sum
of the three midpoints.

| Native real-valued projection | Reversal accuracy | Default `1e-3` | Diagnostic `1e-6` |
| --- | ---: | --- | --- |
| Real part | `3.7983320394e-5` | Accepts | Requests rescue |
| Imaginary part | `2.1708268565e-4` | Accepts | Requests rescue |

Adding both ordinary MadLoop axis rotations changes these accuracies only at
rounding level. Both default-accepted midpoint estimates fail the independent
`3e-6` truth gate. Explicit QP original/reversal/rotations return identical
binary64 report values, so their reported accuracy is zero. The `1e-6` column
is a diagnostic application of the actual estimator; the subsequent automatic
dispatch experiment below observes the rescue itself. The physical-process campaign must independently satisfy its
truth gate; these projections do not guarantee physical-amplitude accuracy.

Cache correlation is excluded for the raw reversal diagnostic: the check-pass,
cache-disabled, and ordinary-cache reversed results agree exactly. Check-pass
and cache-disabled calls leave all cache counters unchanged. Repeated primary
calls hit the relevant cache and reproduce the primary result. The finite
discrepancy is therefore present when the numerical check bypasses caches.

Full-DP precision routing was audited in `vendor/tirust/src/lib.rs:100` and
the pinned `crates/iregi-rust/src/dispatch/embedded.rs:69`. The bridge uses
`TensorWorkspace::new_embedded(..., 0.0f64, EmbedderOptions::default())`:
`method_choice=true`, `partial_rescue=false`, `precision_tiers=false`.
The method-choice workspace retains the `f64` prototype. The optional
double-double partial rescue and 106/256/512-bit precision tiers are disabled;
the served full-DP counts do not conceal higher-precision numerical rescue.
This statement concerns the full-DP path, not the separate pole-only engine.
The reference and patched copies of `embedded.rs` have the same SHA256,
`c4e392171b243c408c4a2802f30e3a90938c2c6cfc70c85d994ff426cce201c4`.

At capture, `cases.json` selected `CTModeRun=-1` and left the default
`MLStabThres=1e-3`, `NRotations_DP=0`, `NRotations_QP=0`. The benchmark worker
forces stability checks and calls `SLOOPMATRIX_THRES(...,-1D0,...)`, retaining
the card threshold. The independent truth tolerance `3e-6` and ordinary target
`1e-8` are different criteria. Any later benchmark-policy change must be
recorded separately and applied equally to baseline and candidate.

`raw-initial.*` preserves the first raw-only run. `raw-and-native.*` adds the
native adapter and native estimator. The latter contracts QP raw tensor data
with 80-digit Decimal arithmetic before rounding its report; the older
66-fixture checker instead contracted rounded binary64 tensor values. That
explains minor differences in the quoted QP error without changing the finding.
`adjudication.json` records assertions, classifications, and all relevant
source hashes. `build.log` records exact compiler/linker arguments. No binary
is stored in this directory.

To reproduce from the MadGraph root with the qualified debug DSO available:

```sh
source tools/tirust_env.sh
export PYTHONDONTWRITEBYTECODE=1
python docs/benchmarkfolder/qualification/tirust-bridge/stability-audit/build_harness.py
flock .tirust-build/numerical-tests.lock python docs/benchmarkfolder/qualification/tirust-bridge/stability-audit/check_stability.py --output .tirust-build/stability-audit/replay.json
python docs/benchmarkfolder/qualification/tirust-bridge/stability-audit/adjudicate.py .tirust-build/stability-audit/replay.json .tirust-build/stability-audit/replay-adjudication.json
```

Use new output filenames on each run. The shared lock serializes numerical
processes under the existing Symbolica license configuration.

The later release replay uses `build_release_harness.py`, then the same
`check_stability.py` with
`--library HEPTools/tirust/lib/libmadloop_tirust.so` and
`--adapter .tirust-build/stability-audit-release/libtirust_stability_audit.so`.
`adjudicate_release.py` executes the unchanged assertions and labels the
result as a release correctness replay. These commands do not modify the
installed prefix. The benchmark's later adoption of a shared `1e-6` threshold
is recorded independently; the original default-policy failure remains here.

The subsequent **release automatic-dispatch experiment** is recorded in
`host-dispatch-release.json`, with its independent assertions in
`host-dispatch-adjudication.json`. `build_host_rescue.py` extracts the production
control block at `loop_matrix_standalone.inc:1713–2006`, including the original
GOTOs, DP/QP decisions, rotations, FPE handling and H/T/U finalization. It uses
the production `COMPUTE_ACCURACY`, `SET_N_EVALS`, `SET_RET_CODE_U`, and backend
availability/QP-index initialization from an actual generated all-backend
process. Only the physics evaluation at labels 200/300 is replaced with the
real rank-five Fortran adapter callback. The rendered block was checked against
the template; source excerpts, compiler commands and hashes are packaged in
`host-dispatch-index.json`.

All eight combinations of threshold, projection and zero/two rotations were
executed against the installed release DSO named above. At `1e-3`, all four
cases return `219` (stable DP) and retain the inaccurate DP results. At `1e-6`,
all four cases execute `[1,2,4,5]`, or `[1,2,1,1,4,5,4,4]` with two rotations,
and return `323` (successful QP rescue). Every backend callback succeeds; the
rescued host accuracy is zero, and the final results exactly equal the
separately computed same-input QP result. Relative errors against the supplied
fixture are `2.1694504548e-16` for the real projection and
`1.1945816650e-15` for the imaginary projection.

This is observed execution of the existing automatic rescue algorithm, not an
inference from its tolerance formula. The complete physical helicity/filter
initialization pipeline remains outside this synthetic experiment. It supports
the separately frozen shared `1e-6` benchmark policy; it does not replace the
physical campaign or prove that this threshold detects every unstable point.
Production defaults remain unchanged. To replay the dispatch experiment:

```sh
PYTHONDONTWRITEBYTECODE=1 python docs/benchmarkfolder/qualification/tirust-bridge/stability-audit/build_host_rescue.py --native-library .tirust-build/stability-audit-release/libtirust_stability_audit.so --build-dir .tirust-build/host-rescue-replay
PYTHONDONTWRITEBYTECODE=1 python docs/benchmarkfolder/qualification/tirust-bridge/stability-audit/run_host_rescue.py --library .tirust-build/host-rescue-replay/libtirust_host_rescue.so --native-report docs/benchmarkfolder/qualification/tirust-bridge/stability-audit/release-raw-and-native.json --output .tirust-build/host-rescue-replay/results.json
PYTHONDONTWRITEBYTECODE=1 python docs/benchmarkfolder/qualification/tirust-bridge/stability-audit/adjudicate_host.py .tirust-build/host-rescue-replay/results.json .tirust-build/host-rescue-replay/adjudication.json
```

The dispatch runner acquires the shared numerical lock itself. Its build also
requires the actual generated process used for the recorded eligibility data;
the exact path and hash are in the build metadata. No native binary is packaged.
