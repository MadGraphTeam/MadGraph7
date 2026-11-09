# Private compact DP kernel experiment

These sources are isolated rank-2x1 and rank-3x1 proposals. No production
generator, exporter, or public updater was changed. The private routines retain
the generated compact updater's complete output initialization, zero-B skip,
coefficient lookup, and M,K,I,L arithmetic order; only arithmetic I is restricted
to J. Admission scans all active real/imaginary input components using the
existing binary64 exponent guard and requires nearest rounding, disabled traps,
the exact four-component shape, and caller-captured sticky INEXACT.

`prepare.py` emits the unchanged public controls and private sources.
`run.py` compares against those public controls using the frozen 24-fixture IEEE
corpus: all shapes 0..4, four rounding modes, all pre-existing IEEE flag masks,
signed zeros, exceptional values and exponent boundaries. Two compiler settings
per rank (`-O3 -frounding-math`, additionally `-fno-automatic`) pass 1,536,000
comparisons in total. Tests assert actual fast/fallback admission, consumed bits,
full fallback output, off-diagonal initialization, unchanged inputs, flags and
rounding. The production polynomial.inc CREATE_LOOP_COEFS and generated MERGE_WL
also pass for both ranks, with caller O0 / polynomial O3, nontrivial Born/color
weights, and flags recorded before and after the consumer. Those synthetic tests
use LOOPMAXCOEFS=210.

The first independently written loop reference changed L/I traversal and
produced different NaN bits on an exceptional fallback fixture. `attempt1/`
retains the failure and its source. The successful reference calls the unchanged
public compact updater directly; it is not an independent arithmetic formula.
`captured-build-attempt1/` retains an initial rejected Fortran TRANSFER assignment;
RESHAPE corrected its rank before any captured-input execution.

`captured.py` replays the four exact operand records under `../capture/attempt3/`.
Those records capture IEEE flags in the actual caller, all with INEXACT set and
traps disabled. Earlier recorder-scope flags are not used. Replay explicitly
selects nearest rounding (the capture header did not record the rounding mode).
All four pass consumed-bit, complete IEEE-flag, and unchanged-input comparisons.
`captured-input-audit.json` independently checks the active raw binary64 operand
bounds. Replay uses the matching LOOPMAXCOEFS=70 and VERTEXMAXCOEFS=5.

The final timings use O3/fPIC without bounds checking, CPU 10, 100 warm-up calls,
then seven alternating pairs of 500 calls per captured operand set. Only the
private variant pays IEEE_GET_FLAG on every call; all private guards and output
initialization are included. The longest measured pass is 0.004027 seconds.

| Rank, sample | Dense median ns/call | Private median ns/call | Median paired private/dense |
|---|---:|---:|---:|
| 2x1, 1 | 2677.032 | 1367.786 | 0.510934 |
| 2x1, 2 | 3222.114 | 1497.586 | 0.468891 |
| 3x1, 1 | 7354.430 | 2906.412 | 0.407062 |
| 3x1, 2 | 7264.990 | 2928.652 | 0.394326 |

These short warm repeated-input measurements have no confidence interval and
are not amplitude or acceptance benchmarks. They do not establish a production
speed gain. The earlier synthetic timings in `results.json` include bounds
checking and must not be substituted for the final captured-input comparisons
in `captured-results.json`. Whole-source coefficient-table and consumer proof
is retained separately under `../proof/`.

Run from the MadGraph root, in an authorized quiet window:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. taskset -c 10 python3 docs/benchmarkfolder/experiments/dp-higher-rank/kernels/prepare.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. taskset -c 10 python3 docs/benchmarkfolder/experiments/dp-higher-rank/kernels/run.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. taskset -c 10 python3 docs/benchmarkfolder/experiments/dp-higher-rank/kernels/captured.py
```

Commands, return codes, stdout/stderr, source hashes, and all individual timing
observations are retained. Compiled binaries remain only in ignored scratch.
