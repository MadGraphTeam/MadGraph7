# UPDATE_WL sparsity investigation

This directory retains isolated diagnostic and prototype work. The qualified
production generator change and its checks are described below. The benchmark
driver, protocol source and previously generated final candidate outputs are
unchanged by these experiments.
The completed diagnostics and process trials use original Ninja (backend 6),
so they do not create a Symbolica worker. These are exploratory results,
separate from the main acceptance benchmark series.

## Completed measurement

`profile.py` copied the final generated QCD 2→3 output, added counters and
`SYSTEM_CLOCK` calls to its polynomial routines, and evaluated the same 100
retained points on CPU 2. Counters are reset immediately before the timed pass.
The extra scans count zero coefficient blocks and nonfinite inputs outside the
individual updater body timers. These timings are diagnostic: instrumentation
adds overhead and the process shares the host with other work.

All 200 physical-order rows reproduce the preflight Ninja6 coefficients,
accuracy estimates, and return codes exactly: 98 DP points and two successful
QP rescues. Full results, sources, build/run logs, status counts, and hashes are
retained in `qcd-3-*`. `profile-command.log` is the invoking script's output.

| Updater lane | Original body time in the 100-point pass |
| --- | ---: |
| DP, all routines | 2.2642 s |
| QP, all routines | 4.8261 s |
| QP, compact routines | 3.3671 s |
| QP, expanded routines | 1.4590 s |

The instrumented full pass took 9.9985 s. These figures do not establish an
optimization speedup. In particular, the body timer excludes the diagnostic
input scans but their effect on caching cannot be removed retrospectively.

The busiest expanded QP routine, `UPDATE_WL_1_1`, takes 0.8561 s and receives
34.98% entirely zero B blocks. Expanded ranks `(2,0)` and `(3,0)` receive 50.45%
and 55.0% zero blocks. No nonfinite A input was observed. Weighting these
fractions by measured body cost suggests roughly 0.6 s of arithmetic work in
this pass, before the cost of any new guards. It is an opportunity estimate,
not a measured gain.

Reproduce the completed diagnostic in a fresh work directory:

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/experiments/expanded-wl-sparsity/profile.py --case qcd-3 --cpu 2
```

The script deliberately refuses to overwrite its previous copy. It compares
against the retained `preflight-qcd3-final-generator` Ninja6 rows.

## Completed compact follow-up

`profile_compact.py` adds a separate count of A coefficients that are exactly
zero **after** the compact updater's existing B-zero filter. It records both
individual zero A terms and entirely zero A coefficient blocks, weighting by
the number of nonzero, finite B coefficients. It records nonfinite B terms
separately. This distinguishes new removable work from multiplication already
skipped by the current compact implementation. The run retained in
`compact-a-followup/` again matches all 200 point/order rows exactly. It finds:

| QP routine | Products after existing B skip | Exact-zero A fraction | Whole zero A block fraction |
| --- | ---: | ---: | ---: |
| `(2,1)` | 13,992,120 | 43.159% | 27.911% |
| `(3,1)` | 7,896,000 | 42.057% | 30.638% |

There were no nonfinite B products. Their original instrumented bodies took
2.1692 s and 1.1991 s respectively.

## Isolated prototypes and exact regression

`prototype.py` provides two generated-QP transformations:

- Expanded: precompute all-zero B blocks whose matching A entries are finite,
  then skip those blocks while preserving every retained arithmetic statement
  and K accumulation order.
- Compact: classify exact-zero finite A coefficients once per call, then skip
  their products only when the existing nonzero B updater is finite. The
  existing B-zero skip and M/K accumulation order are preserved.

Both fast paths require IEEE round-to-nearest and finite active operands.
Directed rounding and nonfinite operands call a separate, untouched dense
helper. This explicit fallback matters: the first prototype only disabled
skips on those inputs, but inserting guards changed compiler-generated NaN
bit patterns. `prototype-v1.py` and `exact-regression-v1.json` preserve that
failed version and its evidence.

The fourth version passes all 210,000 compiled QP comparisons in
`exact-regression.json`: 150,000 expanded and 60,000 compact comparisons,
each with complete output/input bit patterns and all IEEE exception flags.
Tests cover ranks 0–4 by 0–2 for expanded routines, ranks 2–4 by 1–2 for
compact routines, dimensions 0–4, dense/sparse inputs, cancellation, signed
zero, quiet NaN/Inf, subnormal and overflow cases, all four rounding modes,
and both `-O3` and `-O3 -frounding-math`. The compact prototype applies to the
existing vertex-coefficient-first loop form only.

An additional 24,000 comparisons in `signaling-regression.json` exercise
signaling NaNs in A with zero B, and in B with zero A, including empty active
dimensions. Earlier versions 2 and 3 classified inputs with floating-point
IEEE intrinsics, which raised INVALID even in an empty-output call where the
original did no arithmetic. Those failures and source versions are retained.
Version 4 instead inspects the integer exponent bits of IEEE binary128;
unsupported REAL16 representations take the untouched dense path. Both the
full suite and the signaling suite pass with complete output/input bit
patterns and exception flags, under both compiler flag sets.

`run.py` compares the unchanged output with compact, expanded and combined
variants in copied generated outputs. It preserves all 100-point values,
accuracy estimates, row statuses and timed rescue counts, along with source
hashes and patches. These short trials run alongside other host work and
cannot establish an acceptance speedup. Longer isolated paired trials and
full controls are still required before production promotion. The source of
the initial three-pair run is retained verbatim as `run-initial.py`; the
current runner additionally supports shared experiment helpers and snapshots
all its source dependencies.

The initial version-2 three-pair QCD3 trial retained exact values, accuracy and
98 DP / 2 QP statuses in every pass. Median original-relative time ratios were
0.89721 for compact, 0.95300 for expanded and 0.85019 combined. These are
concurrent-host observations with three samples, not a statistically accepted
speedup. The combined version-4 quiet measurements are now complete.

## Quiet paired measurements and production checks

Seven alternating pairs on the reserved CPU 207 use the original generated
model helpers, unchanged numerical settings and retained point streams.
Every output row, accuracy estimate and status count must equal the retained
original bytes, not merely the other variant in the same pair.

| Case | Original median s/point | Combined median s/point | Ratio | Paired 95% interval |
| --- | ---: | ---: | ---: | ---: |
| QCD3 | 0.0968992 | 0.0806043 | 0.831836 | [0.806229, 0.877321] |
| EW2 | 0.000791341 | 0.000797904 | 1.008293 | [0.974855, 1.017739] |

QCD3 uses one timed 100-point traversal per invocation, with 98 DP216 and two
QP328 rescues. EW2 uses 30 traversals, all 3,000 evaluations DP216. Both cases
retain all 200 printed point/order rows exactly. The intervals use the frozen
paired-bootstrap statistic with 10,000 resamples. QCD3 passes the strict
improvement test (upper bound below 1); EW2 passes the 1.05 regression limit.
These are isolated optimization measurements, not substitutes for the final
four-backend core benchmark.

Results and full source/executable/input hashes are in
`qcd3-v4-quiet-seven-pair/` and `ew2-v4-quiet-seven-pair-retry1/`.
The first EW2 setup found no compact updater and stopped before timing.
Its failure record is retained in `ew2-v4-quiet-seven-pair/`; the retry
handles an absent updater category without changing the Fortran prototype.

The production generator now emits the same statements directly. Thirty
rank/mode comparisons confirm equality with measured version 4, apart from
one additional compile-time check of the binary128 representation of 1.0.
Ninety source comparisons confirm unchanged DP, alternate compact-order and
explicit dense bodies. The new independent compiled regression performs
30,000 complete bit/input/IEEE comparisons, including forced rejection of an
unsupported layout. All 43 polynomial tests and four optimized exporter
polynomial comparisons pass. The ten current polynomial snapshots are updated;
five historical pre-module layouts are unchanged.

## Structural sparsity

There is a possible generator-level route, but the current rank-only updater
interface does not carry the information it needs. The HELAS call writer emits
`UPDATE_WL_<mother rank>_<vertex rank>` after each ALOHA vertex call
(`madgraph/iolibs/helas_call_writers.py`, around line 1997); the exporter chooses
expanded versus compact solely from those ranks
(`madgraph/loop/loop_exporters.py`, around line 2633).

The generated ALOHA vertices contain provably literal zeros. For example,
QCD3 `Source/DHELAS/MP_FFV1L2_1.f` sets `COEFF(3,1,1)` and
`COEFF(4,1,1)` to `0Q0`, with related derivative components also zero. Its
constant coefficients for the same rows instead contain `M1`, so complete
zero blocks additionally depend on a massless call. Other zeros depend on
helicity or cancellations and cannot be inferred from the rank pair alone.

Specializing on vertex identity, a proven literal-zero mask, and the incoming
wavefunction's symbolic support could remove runtime branches. That needs
new exporter metadata, mask propagation through the wavefunction graph, and
care over reusable model parameters. Even a structural zero must retain a
fallback for nonfinite multiplication if exact legacy failure behavior is
required. The measured runtime-guard approach is a smaller first experiment;
structural specialization remains a separate, unimplemented proposal.
