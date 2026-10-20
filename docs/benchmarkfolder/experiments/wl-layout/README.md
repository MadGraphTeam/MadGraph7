# Compact wavefunction update loop order

The executable profiles in `../fortran-profile` identify compact `UPDATE_WL`
polynomial contractions as the main Fortran cost. This experiment swaps the
innermost `L,I` loops to `I,L`, traversing polynomial coefficients within a
fixed last-index array plane. The outer `J,M,K` order and all arithmetic remain
unchanged. For each fixed output `(J,position,I)`, the sequence of contributing
`(M,K,L)` entries is identical, including zero-coefficient skipping and clearing
of the complete output array. The optimization applies to generated compact DP
and QP routines.

`run.py` copies the complete original baseline output, modifies only the copied
`polynomial.f`, and builds and runs both copies. It preserves the untouched
baseline output. Each variant uses original Ninja, the same 100 stored phase
points and normal DP/stability checking. Every invocation checks that the full
printed Born/Laurent coefficients, accuracies and statuses are byte-identical
and that every timed status is `216`.

From the repository root, with the benchmark baseline outputs already generated:

```sh
source .ninjars-build/dev-env.sh
python3 docs/benchmarkfolder/experiments/wl-layout/run.py \
  --cases qcd-2,qcd-3 --cpu 2 --repetitions 7
python3 -m unittest tests.unit_tests.various.test_q_polynomial_compiled
```

The compiled regression compares the current generated routines against an
independent implementation of the legacy accumulation order. It covers left
ranks 0–4, right ranks 0–2, all three wavefunction dimensions 1–4, dense complex
inputs, scales from `1e-20` to `1e20`, and sparse updater coefficients. All 2,880
DP and 2,880 QP full-array comparisons passed at `-O3` with bounds checking.

The retained exploratory measurements used seven alternating pairs on CPU 2.
Other builds were running on the shared host:

| Case | Original median, ms/point | Interchanged median, ms/point | Ratio |
| --- | ---: | ---: | ---: |
| `qcd-2`: `g g > t t~ [virt=QCD]` | 2.264197445 | 2.129129825 | 0.940346 |
| `qcd-3`: `u u~ > z g g [virt=QCD]` | 37.20367625 | 32.32222553 | 0.868791 |

All 100 point outputs agreed exactly in every run. These observations justify
testing the production generator change, but do not establish a final speed
claim: timings varied substantially during concurrent builds. Retention requires
isolated paired repetitions and the full benchmark's correctness and performance
gates. The harness's paired bootstrap (10,000 resamples, seed 410083) gives 95%
ratio intervals `[0.913864, 1.033940]` and `[0.398845, 1.111719]`, respectively;
both leave a strict speed-benefit gate unresolved. The JSON records preserve each
sample, status count, point-stream hash and exploratory statistics;
the patch files preserve the exact generated-code changes. `production-template.patch`
records the original three-line generator change (the applied version also has
an explanatory Python comment).

The subsequent retained persistent-worker experiment
`../persistent-workers/ninja-parity-final-20261020` gives a qcd-2
candidate/original median ratio of `0.91643` and paired 95% interval
`[0.91278, 0.92128]` over seven alternating blocks. Fresh and reused processes
return identical full numerical rows. This later evidence resolves the
representative compact-kernel benefit and supports retaining the production
change. It remains an exploratory observation on a shared host, with uncertainty
covering warmed blocks rather than process initialization. It does not replace
the complete eight-process legacy regression or NinjaRS performance gates.
