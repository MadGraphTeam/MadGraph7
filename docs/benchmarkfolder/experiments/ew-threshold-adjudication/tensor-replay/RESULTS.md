# Matched tensor replay results

The bounded replay completed all three TIRust tensor calls and all 105 NinjaRS
unit-numerator calls. TIRust reproduced the original Gram-matrix rejection for
threshold point 2, group 5. NinjaRS returned status zero for all 35 units of
that group. There is no successful TIRust tensor for that case, so no numerical
comparison against a zero or partial TIRust array is made.

Both engines succeeded for ordinary point 1/group 5 and threshold point 2/group
24. Every printed TIRust coefficient for those two groups exactly matches the
historical parent trace. All three cases are four-propagator, rank-three
integrals, with 1, 4, 10 and 20 unit coefficients of degrees zero through three.
Group 5 has squared masses `[mt², 0, mt², mt²]`; group 24 has four zero masses.

| Point/group | TIRust status | NinjaRS successful units | Complex coefficient comparisons |
|---|---:|---:|---:|
| Threshold 2/5 | 1, unstable Gram matrix | 35/35 | 0; rejected tensor excluded |
| Ordinary 1/5 | 0 | 35/35 | 105 |
| Threshold 2/24 | 0 | 35/35 | 105 |

The failed group's native error reproduces conditioning
`5.22407811366341444758736094099234985181085186192010198472166824e-30`
for Gram indices `[0, 1, 2, 3]`. The replay retains the historical guard and
256-bit working precision; it applies no precision or threshold adjustment.
Successful NinjaRS status is evidence of a returned tensor coefficient, not an
independent accuracy certificate for the near-singular input.

The table reports the largest complex differences over each group. Relative
difference means `abs(a-b)/max(abs(a),abs(b))`, with zero–zero set to zero.
Absolute and relative maxima need not occur at the same coefficient.

| Point/group | Slot | Maximum absolute difference | Maximum relative difference |
|---|---|---:|---:|
| Ordinary 1/5 | finite | 5.58e-19 | 9.47e-17 |
| Ordinary 1/5 | single pole | 7.69e-20 | 1.57e-16 |
| Ordinary 1/5 | double pole | 0 | 0 |
| Threshold 2/24 | finite | 4.15e-22 | 2.30e-11 |
| Threshold 2/24 | single pole | 5.15e-23 | 4.14e-4 |
| Threshold 2/24 | double pole | 3.20e-24 | approximately 1 |

The scalar coefficient (`unit=0`) agrees exactly in every Laurent slot for
both successful groups. Large relative pole differences occur among tensor
coefficients close to zero. For example, threshold group 24/unit 29 has a
single pole near `-1.06e-19`, differing by `4.39e-23`; unit 21 has a NinjaRS
double pole `3.20e-24` while the TIRust value is about `1.89e-66`.
These differences remain recorded, including all complex values. They are not
dismissed by the scalar agreement or by the native success statuses. Absolute
values across different tensor degrees carry different units and must be
interpreted with their numerator coefficients; those original coefficients
are not present in this diagnostic. No amplitude reconstruction, whole-process
accuracy resolution or acceptance claim follows from these unit tensors.

`metrics.json` records coefficient identities, both complex values, scalar
comparisons and maxima. Its optional count above `3e-6` uses the complex
max-norm metric stated here, which differs from the campaign's
`abs(a-b)/abs(a+b)` regression metric. The threshold control has 14 single-pole
and 25 double-pole unit ratios above that diagnostic cutoff; all are retained.

Both engines consume exactly the captured TIRust shifts, invariant matrix,
complex squared masses and scale. The common scale deliberately differs from
the production NinjaRS DP-square transport, as specified in the frozen
protocol. Input echoes match every captured decimal token, including signed
zeros. The independently audited generated map, odd-degree signs, coefficient
counts and complete output grids all pass the frozen analyzer. TIRust caching
is disabled, with zero entries/bytes and one bypass per call. The two engines
share oneloopmaster ancestry at different revisions, limiting scalar
independence.

The original `README.md`, Fortran driver, preparation script, analyzer and
source manifest remain unchanged from source-only preparation. Their manifest
hash is `97b93227e0475f7bb0037434d041e49d190aa25dcf1b786e26870dc173a4f4a4`.
The new launcher links the private historical wrapper/module and DSO, plus the
unchanged pinned NinjaRS archive. `compilation.json` records commands and
object/executable hashes. `execution.json`, `ldd.txt` and `process-maps.txt`
bind the actual loaded historical DSO and runtime libraries. Protected
sources, inputs and archives match before and after; only private replay
artifacts were created. The process exits zero, and the complete frozen
validator passes.

Build and execution were each bounded to 180 seconds. Execution held the
shared numerical lock on CPU 9 and completed in about 4.83 seconds. That value
includes setup and all diagnostic output, while an unrelated installer build
ran on other CPUs. It is timing-ineligible and is not a benchmark result.

The initial source/input preparation commands are preserved in `README.md`.
After restoring the exact historical runtime files in `runtime-snapshot.json`
and preparing a fresh private work directory, the authorized native and
analysis commands from the repository root were:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/run.py build
python3 -B docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/run.py run --cpu 9
python3 -B docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/analyze.py
python3 -B docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/summarize.py
```

The launcher deliberately refuses to overwrite existing build/run evidence.
The historical binary snapshots and executable live in ignored local working
directories; the tracked sources, hashes and raw output preserve this run's
evidence. No additional native evaluation or higher-precision build was made.
