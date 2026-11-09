# Ordered Fortran prefix sums: isolated experiment

This experiment measures a small Fortran preprocessing kernel and a generated
tensor-reduction wrapper with a **stubbed numerical reducer**. It does not
measure physical amplitudes or establish the campaign's runtime acceptance.

The selected candidate replaces repeatedly summing each momentum prefix with
one cumulative sum. Each prefix retains precisely the original order of
additions, including the initial positive zero:

```fortran
      DO I=0,3
        PREFIX_SUM=0.D0
        DO J=1,NLOOPLINE-1
          PREFIX_SUM=PREFIX_SUM+PL(I,J)
          P_TMP(I,J)=PREFIX_SUM
        ENDDO
      ENDDO
```

The quadruple-precision version declares `PREFIX_SUM` in the same precision as
`PL` and uses `0.0_16`. Assigning the first input directly would omit the initial
addition and can change signed-zero behavior under IEEE rounding modes. The
conservation sum in column zero, thresholds, and independent stability checks
are untouched. For six propagators the prefix additions decrease from 60 to 20.

## Reproduce

Run from the MadGraph repository root. GNU Fortran with `REAL*16` and Python 3
are required; the development environment supplies both. All compilation and
executables remain in the ignored build directory.

```sh
source tools/tirust_env.sh
python3 docs/benchmarkfolder/experiments/fortran-prefix/run.py \
  --build-dir .tirust-build/fortran-prefix-refined \
  --output .tirust-build/fortran-prefix-refined/results.json \
  --seconds 0.25 --pairs 7 --suite focused
```

The focused suite tests propagator counts 3 and 6. Use `--suite all` for counts
2 through 6, mixed hits/misses, combined experimental changes, and the historical
wrapper comparison. `--cpu` selects an allowed logical CPU; by default the last
CPU in the process affinity mask is used. `--compiler` selects the compiler.
Frozen source checksums are checked before compiling. The reproduction script
does not import MadGraph and needs no reduction libraries.

## Numerical checks

For each optimization level, 481,536 prefix cases compare exact output bit
patterns and the five IEEE exception flags (invalid, overflow, underflow,
inexact, and divide-by-zero). They cover:

- Double and genuine Fortran quadruple precision, propagator counts 1–6.
- Round-to-nearest, upward, downward, and toward-zero rounding modes.
- Positive/negative zero, normal and subnormal extremes, infinities, and quiet
  and signaling NaNs. The refined fixture uses each precision's own extremes.
- 10,000 deterministic random raw-bit matrices per size and rounding mode,
  using a fixed xorshift seed of 1234567, plus 32 structured edge cases.

Every timed candidate/reference pair must also have the same printed checksum.
That wrapper check uses one balanced finite momentum configuration for each
size; it is not a physical phase-space or amplitude regression.

## Timing method and limits

The frozen wrappers come from `TIR_interface.inc`, rendered with
`TIRCaching=True`, `AmplitudeReduction=True`, one loop group, one squared order,
and maximum coefficient capacity 210. Rank equals propagator count in timed
wrapper calls. The external COLLIER routine is replaced with a separate
deterministic Fortran stub that fills all 210 coefficients. There is no link-time
optimization. The contraction and outer Fortran cache remain real generated
code, but reducer cost and cache-miss balance are artificial.

Both `-O0` and `-O3` are measured with
`-fPIC -ffixed-line-length-132 -ffree-line-length-none`; no fast-math flags are
used. The generated pilot's ordinary TIR compilation had no explicit
optimization flag, so `-O0` is relevant; the `-O3` series exposes dependence on
compiler optimization. This experiment does not propose changing production
compiler flags.

Seven alternating reference/candidate pairs use process CPU time on a single
logical CPU, with a calibrated target of 0.25 seconds per batch in the refined
series. Other builds were active on the same host. CPU frequency and contention
remain uncontrolled, and complete-wrapper differences are sometimes noisy.
Raw paired seconds per call, counts, compiler identity, commands, source hashes,
and paired log-ratio statistics are retained in JSON. These are exploratory
microbenchmarks, distinct from the required warmed full-amplitude paired
campaign. The earlier exploratory series used 0.035-second batches.

## Frozen variants and provenance

`sources.json` records source hashes and baseline commit
`9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`. `sources/` contains all Fortran and
include files needed for reproduction:

| Variant | Change relative to the frozen current wrapper |
|---|---|
| `current` | Snapshot before the prefix optimization |
| `ordered` | Only the zero-seeded scalar prefix optimization |
| `early` | Move mass conversion and prefix/PDEN setup behind cache miss; keep momentum-conservation checks before lookup |
| `linear` | Exploratory combination of `early` and a previous-prefix-array implementation |
| `base` | Starting-revision wrapper text rendered with shared current include declarations |

The `base` fixture is for source comparison and optional wrapper-only timing.
It is **not** the historical full-MadLoop performance baseline: the reducer is
stubbed and include declarations are shared with the other variants.

## Measured kernel results

The refined run used GNU Fortran 15.2.0 on an AMD EPYC 9754, logical CPU 383.
Both optimization levels passed all 481,536 bit/exception checks with zero
failures. `results-refined.json` contains the complete raw measurements.

| Optimization | Propagators | Precision | Original median (ns) | Ordered median (ns) | Median paired ratio |
|---|---:|---|---:|---:|---:|
| `-O0` | 3 | DP | 91.4 | 65.0 | 0.678 |
| `-O0` | 3 | QP | 289.8 | 191.7 | 0.677 |
| `-O0` | 6 | DP | 235.5 | 103.3 | 0.443 |
| `-O0` | 6 | QP | 1361.9 | 472.8 | 0.347 |
| `-O3` | 3 | DP | 55.5 | 44.2 | 0.809 |
| `-O3` | 3 | QP | 220.4 | 156.1 | 0.705 |
| `-O3` | 6 | DP | 66.5 | 46.7 | 0.703 |
| `-O3` | 6 | QP | 1058.9 | 375.6 | 0.356 |

The ratio column is the median of seven paired ratios, not the ratio of the
two marginal medians. The isolated prefixes consistently improved. Complete
stubbed-wrapper changes were much smaller and less consistent: for example,
the prefix-only rank-six all-miss wrapper at `-O0` had a paired ratio of 0.967,
while the rank-three all-hit wrapper at `-O3` had a ratio of 1.013. This supports
testing the small arithmetic change in real amplitudes, not a claim of an
already qualified amplitude speedup.

## Cache decision and whole-amplitude follow-up

Do not broaden tensor caching solely because a Born process has multiple
helicities. `LoopProcessExporterFortranSA.get_context` enables it for
amplitude-level reduction (loop-induced or requested color-flow output) or more
than one squared coupling order. Ordinary Born code merges numerator
coefficients over helicities before reducing each loop group once per squared
order.

In the generated `qcd_n1_uux_z` pilot under
`.benchmark-work/tirust/pilots/no-ninjars`, `loop_matrix.f` lines 970–1040 contain
the helicity loop and numerator construction; line 1079 calls reduction after
that loop. `loop_CT_calls_1.f` line 102 contains its one rank-two triangle call.
This static control-flow evidence predicts no additional primary-pass reuse
from enabling the outer cache for this single-order Born process. Adding cache
copies could instead cost time. Call counts and actual cache hits should be
instrumented on representative larger processes before changing that policy.

The cache-preprocessing experiment is not selected for production: gains depend
on hit frequency, misses can regress, and moving the conservation check itself
would remove an existing safeguard. The isolated prefix candidate still needs
full-amplitude numerical validation and the agreed per-process paired runtime
checks. Neither a kernel speedup nor a stubbed wrapper result establishes those
requirements.
