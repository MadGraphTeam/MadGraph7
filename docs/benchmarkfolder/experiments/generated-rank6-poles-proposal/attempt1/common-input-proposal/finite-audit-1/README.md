# Read-only finite audit of the rank-six common-input diagnostic

The N=6, rank-six finite disagreement remains open. The retained data establish a disagreement with a comparator that missed its requested accuracy; they do not establish which implementation is wrong. No numerical code, cache policy, acceptance threshold, original evidence or reference repository was changed. This audit performed only text/ELF inspection and offline analysis; no library was loaded or evaluated.

The closed original execution and its unchanged analyzer are retained in [`../runtime1`](../runtime1/README.md), index `b25130768bc8a5f8daabbcfe820160755b41b4dd2e558cd0b1a87aa49e1f6ec7`. All thirty actual rows use bit-exact common effective shifts, invariants, complex squared masses and scale. This audit first replayed that analyzer and required identical `results.json` before extracting the exact binary128 coefficient words. `rows.json.gz` preserves all thirty parsed rows; `analysis.json` records all 122 suspect finite real/imaginary parts rather than only the largest example.

## What the reported residual means

For fixture 108, TIRust QP full actions 3007/3009/3011 and COLLIER actions 3013/3014/3015 have the same finite coefficient words across all three UV/IR selectors within each backend. Each of the three full finite comparisons fails the existing `3e-6` criterion for the same 122 of 420 real/imaginary parts: 2 real and 120 imaginary. The actual worst finite symmetric residual is **0.0013285037839680765**, at coefficient 161, powers `(3,1,1,1)`, imaginary part:

| Backend | Exact-input finite imaginary value, shortened for display |
|---|---:|
| TIRust QP, action 3007 | −7.2782192842873164104e−12 |
| COLLIER, action 3013 | −7.2975832931658345641e−12 |

Their absolute difference is about `1.9364e-14`. The two real failures are coefficient 55, powers `(1,1,1,1)`, residual `5.8186660e-6`, and coefficient 111, powers `(1,1,1,2)`, residual `3.4446017e-6`. Thus the issue cannot be dismissed solely as the relative comparison of a tiny imaginary part.

| Tensor rank | Coefficients | Finite real failures | Finite imaginary failures |
|---|---:|---:|---:|
| 0 | 1 | 0 | 0 |
| 1 | 4 | 0 | 0 |
| 2 | 10 | 0 | 0 |
| 3 | 20 | 0 | 11 |
| 4 | 35 | 1 | 24 |
| 5 | 56 | 1 | 31 |
| 6 | 84 | 0 | 54 |

The original aggregate maximum residual **1** comes from the simple-pole slot: TIRust returns exactly zero, while the COLLIER adapter subtracts separately evaluated finite tensors to extract UV residues and obtains tiny nonzero values for all 210 coefficients. For this massive N=6 tensor, even rank six is UV convergent (`4 + 6 − 2×6 = −2`); all internal masses are nonzero, so no IR residue is expected either. The largest COLLIER simple-pole/finite ratio is `7.3561e-7`, which passes the analyzer's existing finite-scaled pole test at `3e-6`. The maximum symmetric pole residual is not the finite disagreement's size. No criterion was relaxed.

All QP full/partial pole words agree exactly. TIRust DP explicitly refuses all six N=6 requests on the same Gram-conditioning condition. Fixture 109, N=2/rank six, has all 210 UV witnesses and its finite comparisons pass; this checks an additional mapping path, but does not qualify the finite N=6 result. These fixtures are standalone synthetic tensors, not full amplitudes or physical phase-space stress tests.

## Comparator accuracy and IEEE state

The currently mapped COLLIER DSO is SHA-256 `6ee2403dd120f9b0f5a551a7d16412ddfd46d97f29af6f61f0d09fc62db80d92`. Its initialized ELF symbol `__collier_global_MOD_version_cll` contains the five literal bytes `1.2.9` (file offset 3677864); the installed source/build DSO has the identical hash. `collier-identity.json` retains the read-only ELF proof. Source version/CMake/installation evidence is independently bound in `bindings.json` and quoted with line numbers in `source-excerpts.json`. No runtime version query was executed.

Every N=6 COLLIER row reports error flag 0 and accuracy flag **−1**. COLLIER 1.2.9 `collier_init.F90:2534–2537` sets −1 when any propagated per-rank estimate exceeds the requested accuracy, and −2 above critical accuracy. This run uses default mode 1 and required accuracy `1e-8`; critical accuracy defaults to `0.1`. Therefore −1 is an actual requested-accuracy miss, not a sentinel meaning accuracy is unavailable above five points, and it does not certify `3e-6` agreement. Error flag 0 supplies no independent accuracy guarantee.

The flags are event-cumulative. This driver starts an event before each adapter invocation; a mode-0 invocation then contains base, UV, IR1 and IR2 evaluations. Existing evidence does not retain `TNtenErr` by rank and by inner call, so the exact source of the −1 cannot be identified. Mode 1 does not perform the independent COLI/DD comparison offered by mode 3. Disabling MadLoop's `COLLIERUseInternalStabilityTest` does not turn off COLLIER's own propagated estimates; it leaves the adapter's `TIRCOEFSERRORS` zeroed. The driver's existing `CERR==0 && CACC>=-1` rule accepts these requested-accuracy misses. This audit preserves that rule and records its scientific limit.

The fresh process initially has no bad IEEE flags. The first COLLIER N=6 call changes the recorded mask from 32 (inexact) to 37 (inexact, invalid and divide-by-zero). Subsequent calls inherit those sticky flags, so their unchanged masks do not establish fresh exception freedom. No flags were cleared and no trap/rounding settings were changed.

## Known COLLIER cache issue: bound context, no fix

The user's supplied note is read-only context:

`/shared/external/madloop_ninjars/madgraph7_madloop_ninjars/.ninjars-build/upstream-poles-20261023/ninjars/COLLIER_CACHE_BUG.md`

SHA-256: `3d7052026dfcd86331814e2e746ac1924f226cce9c41cc74e5a3f955bc38f7d0` (8710 bytes).

It documents COLLIER **1.2.10**, global-cache **N=7/rank 9**, wrong results from the third event and accuracy flag **−2**; a rank-promotion/call-position explanation remains a hypothesis. Its tested N≤6 suite agreed within `1.4e-7`, and the note explicitly says cached 1.2.9 was not checked.

The present standalone common-input fixture explicitly sets `COLLIERGlobalCache=0`, leaves all cache-active counters zero, resets the library and takes the no-cache `InitEvent_cll()` path. It does not create or enable the global cache. This differs from ordinary campaign cards, which use `COLLIERGlobalCache=-1`; no claim about their cache correctness follows from this standalone audit. The documented cached N=7/rank-9 issue is **not a demonstrated cause** of this uncached 1.2.9 N=6/rank-six discrepancy. Per the user's instruction, leave the COLLIER cache implementation unchanged and do not contact its authors.

## Narrow conclusion and remaining evidence

The lower ranks agree much better, the mismatch grows with tensor rank, and identical finite words across selectors argue against a selector-specific extraction or an overall rank-independent normalization/sign discrepancy. This pattern is consistent with numerical conditioning or cancellation in a high-rank reduction; it is an inference, not a diagnosis of either implementation. The exact-input check rules out the previously observed input-policy discrepancy for these two fixtures. The all-nonzero mapping witness alone cannot certify the permutation or the finite values.

Before assigning blame or accepting this N=6/rank-six finite fixture, retain a same-input comparison whose accuracy is independently established. Useful discriminating evidence includes per-rank/per-inner-call COLLIER error estimates with the cache still disabled, an independently qualified reducer or analytic/numerical representation, and genuine precision-convergence checks of TIRust's finite components. A success status or a higher-precision retry by itself is not truth evidence. Preserve the existing IEEE observations and all failures. None of these additional numerical evaluations was performed by this audit.

Portable verification from the MadGraph working directory:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/common-input-proposal/finite-audit-1/audit.py
```

This verifies owned hashes, the retained runtime index, the raw-record-to-snapshot word/state join, and every derived finite/pole summary. Add `--live` only to check original workspace/installed-library/external-note paths as well. Portable verification does not load native code or require those live paths. The original runtime package has its own full frozen-parser replay.
