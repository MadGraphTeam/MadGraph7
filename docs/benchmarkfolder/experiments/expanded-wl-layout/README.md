# Expanded wavefunction update traversal

The isolated `run.py` transformation changes the expanded `UPDATE_WL` routines
from an outer output-component `J` loop to an inner `J` loop, for both clearing
and arithmetic. For every fixed output `(J,coefficient,I)`, contributions still
arrive in ascending `K` order, and every emitted arithmetic statement remains
unchanged. Clearing touches exactly the same active output region. Input and
output arrays must be distinct, as in the existing conforming Fortran callers.

An independent audit split the generated files at subroutine boundaries and
confirmed that the transformation matched only expanded `UPDATE_WL` routines:
ten DP/QP routines in `ew-2` and twelve in `qcd-2`. A production implementation
should change the structured generator rather than apply this experimental
regular expression to arbitrary Fortran.

## Exact numerical check

Run from the repository root:

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/experiments/expanded-wl-layout/check_exact.py
```

The script reads the original generator from revision `d47f0bc4f`, so it remains
reproducible after the production generator changes. It compiles the original
generated routines and the transformed routines together using GNU Fortran
15.3.0, `-O3`, and bounds checking. It compares the
complete output arrays as integer bit patterns, including inactive regions
initialized with sentinels, and checks that inputs were not modified.

All **11,250 DP and 11,250 QP comparisons passed**. Coverage includes left ranks
0–4, right ranks 0–2, every combination of wavefunction dimensions 0–4, dense
and sparse complex inputs, scales from `1e-100` to `1e100`, all-zero inputs,
alternating cancellation, and signed zeros. `exact-regression.json` retains
compiler flags, source hashes, counts, and output.

## Exploratory full-process evidence

The two copies use identical generated candidate physics code and original
Ninja, differing only in the expanded routines. The existing compact-kernel
optimization is present in both copies. Every run evaluates the same 100 points
30 times and checks all numerical output rows, accuracy estimates, and statuses
against one retained reference. Each timed run returned DP code `216` for all
3,000 calls.

| Process | Interchanged/original median | Paired bootstrap 95% interval |
| --- | ---: | --- |
| `ew-2` | 0.933690 | [0.894414, 0.959123] |
| `qcd-2` | 0.970802 | [0.942238, 0.980781] |

Both nine-pair intervals support a runtime improvement in these experiments.
The calculations use the benchmark's paired ratio-of-medians bootstrap with
10,000 resamples and seed 410083, with a strict benefit threshold of 1.0.
The raw samples, generated patches, and independently computed statistics are
retained beside this note.

The observed run copied the then-current candidate outputs before expanded
traversal was promoted. `observed-run.py` preserves the exact script used for
those observations; `observed-run-provenance.json` identifies the retained
copies and labels the later provenance snapshot explicitly.

The maintained `run.py` first selects the retained experiment's original copy,
then the archived `outputs/pre-final-candidate` output, and only then an
unchanged current candidate. It verifies that the expanded routines still have
the original loop layout and that compact routines already have their shared
optimization. Thus a regenerated optimized candidate cannot silently become
the experiment's original variant. To reproduce into fresh directories:

```sh
python docs/benchmarkfolder/experiments/expanded-wl-layout/run.py \
  --cases ew-2,qcd-2 --cpu 4 --repetitions 9 \
  --work-dir .benchmark-work/experiments/expanded-wl-layout-repeat \
  --report-dir docs/benchmarkfolder/experiments/expanded-wl-layout/repeat
```

An explicit `--source-root .benchmark-work/outputs/pre-final-candidate` can select
the archived case directories. Existing experimental copies and evidence are
never overwritten. New runs retain source selection and polynomial hashes,
generated-source manifests, and executable hashes.

Other tasks were active on the host. These measurements supported promoting the
generator change; they are not final core benchmark acceptance results and do
not establish improvements for every process.

The production generator now applies the expanded traversal directly.
`tests/unit_tests/various/test_q_polynomial_compiled.py` adds an independent
runtime convolution using the legacy per-output accumulation order, rather
than importing this experiment's transformation. Its 11,250 DP and 11,250 QP
bit-pattern comparisons passed, together with the existing compact-kernel
checks and all 41 polynomial tests. Filtered optimized exporter comparisons for
`gg_ttx`, `ddx_ttx`, `dux_mumvmxg`, and `gg_wmtbx` also passed after updating the
affected polynomial snapshots.
