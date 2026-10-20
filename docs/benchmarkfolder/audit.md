# NinjaRS–MadLoop integration audit

The numerical and performance results in `benchmark.pdf` are generated from retained
measurements. This note documents code-level findings; it does not substitute for
the full process comparison or performance gate.

## Implemented checks and fixes

- The Fortran interface carries MadLoop's supplied, thresholded S matrix, loop
  momentum convention, odd-rank coefficient signs, complex masses, Laurent order
  and `1/(16 pi²)` normalization. Both amplitude reduction and Born interference
  are covered by compiled recording-backend tests.
- DP uses NinjaRS's binary64 fast engine. Higher precision transports actual
  binary128 values as 40-significant-digit ASCII records, recovers their represented
  113-bit values, then widens to 256-bit arithmetic. The scalar on-shell threshold
  follows original Ninja's binary128 convention rather than becoming tighter merely
  because the working precision is higher.
- The audit found that the generic higher-precision auxiliary box sampling scale
  was fixed at one. It now uses the maximum of one, absolute supplied invariants
  and absolute real squared masses, matching NinjaRS's pinned generic tensor harness.
- Retained amplitudes reset their per-call output/status state. Tests change
  coefficients, mass order and renormalization scale on reused contexts, including
  spacelike `(0,m²)`, `(m²,0)` and `(0,0)` bubbles, and compare fresh contexts and DP.
- Rust panics are contained at the C boundary or in the retained rescue worker.
  Output buffers are committed only on success. The Fortran adapter rejects
  unsuccessful status and nonfinite output, including overflow on conversion to DP.
- Context destruction joins the shared rescue worker when the last context is
  released. The Fortran `EXIT_MADLOOP` hook calls `ninjars_finalize` only for exports
  with NinjaRS available. Shared scheduling allows multiple C contexts without
  violating the restricted Symbolica license's single-active-thread allowance.
- Existing reduction return integers are preserved. NinjaRS uses `(H,U)=(2,8)`
  in DP and `(3,2)` in QP, with dedicated statistics counters. Card parsing supports
  all eight distinct library IDs plus its terminating zero.

## Verification locations

`tests/unit_tests/loop/test_ninjars_fortran.py` contains four compiled tests, including
present/absent module cleanup and color-flow combinations. These tests passed during
the integration audit. `vendor/ninjars/src/lib.rs` contains bridge and scalar reuse
tests; all nine passed, including ABI2 buffer-length validation, after the
sampling-scale and shared-worker fixes. The full
process benchmark remains independently governed by the data and status tables in
the generated report.

The benchmark driver checks actual return codes on every timed evaluation.
Calibration is separate, output is outside the timed region, and the initialization
clock includes a complete warm pass after helicity filtering. The report excludes
calibration and hides final timings until the required repetition count is reached.
Its correctness calculation independently compares all backend pairs and all four
coefficients in every point/order slot, rather than trusting a pass flag alone.

## Rational-term diagnostic convention

The arbitrary-precision four-photon rank-four box replay agrees with original
Ninja on finite, single-pole and double-pole results, while its separate rational
diagnostic differs by approximately `0.006740638578186683`. This separation is
basis dependent, as documented by the pinned NinjaRS revision
`0d0d7e7a8033431b60a8580731084742e0655cda`:

- `crates/ninjars-cli/tests/fast_engine.rs:1–9` compares the fast engine's separate
  rational and cut-constructible parts with Ninja, rather than with the generic
  engine, because the cut bases differ.
- `crates/ninjars-cli/README.md:260–265` states that the generic engine's split
  can differ while the Laurent coefficients agree.
- `crates/ninjars-symbolica/tests/fast_parity.rs:286–289` identifies the different
  distribution of some rank-at-least-`n` bubble terms, with identical totals.
- `validation/performance/shared_provider/PHYSICAL.md:18` validates the total
  Laurent coefficients and each reducer's own `finite = cut-constructible +
  rational` identity; it does not claim separate-term equality across bases.

The MadLoop interface uses only the three complete Laurent coefficients and
discards the local `R1` diagnostic after each call, exactly as its original Ninja
interface does. No rational term is added a second time. The public bridge
diagnostic should therefore be interpreted in its engine's residue basis.
Full Laurent agreement remains mandatory; this convention does not justify a
discrepancy in an observable coefficient.

## Scratch-array experiment: not retained

Copies of the generated original `qcd-1` and `qcd-2` outputs were modified to use
bounded coefficient scratch arrays and fused copying/sign/conversion. Production
templates and the untouched benchmark outputs were not changed. Both variants
produced byte-identical printed coefficients, accuracies and status codes on all
100 points in normal DP/stability mode and explicitly selected QP mode.

Seven alternating exploratory DP repetitions, pinned to CPU 2, gave:

| Process | Original, µs/point | Scratch, µs/point | Scratch / original |
| --- | ---: | ---: | ---: |
| `u u~ > z [virt=QCD]` | 44.9522 | 45.0179 | 1.00146 |
| `g g > t t~ [virt=QCD]` | 1365.9906 | 1362.9724 | 0.99779 |

Other builds were running on the shared host. These are exploratory observations,
not acceptance measurements, and show no demonstrated end-to-end improvement
above noise. The production change was therefore not retained. One QP validation
run per variant gave `2.14782 → 2.15077 ms/point` and
`102.889 → 103.290 ms/point`, respectively; no QP speed claim follows from one run.
QP was selected directly for this regression check, so its status `120` records
an unchecked QP evaluation rather than a stability acceptance.

The separately rendered template test compares every backend input and result
for amplitude and interference modes, DP/QP, rank sequence `2,0,3,1,0`, and
direction modes `1,2,4,5`; both variants agree exactly. Experiment scripts, diffs
and timing JSON are retained under `experiments/ninja-scratch` alongside this
report, with generated build trees kept in `.benchmark-work`.

## Profile-driven wavefunction update optimization

Executable sampling of the original `qcd-2` and `qcd-3` outputs identified compact
`UPDATE_WL` contractions as the dominant Fortran cost. Their leading rank-two
updates accounted for 40.08% and 32.29% of executable self samples, respectively.
The qcd-3 rank-three update added 18.41%. These percentages exclude dynamically
linked reducer/master-integral code and do not represent total wall time.
The profiles, CSV and reproduction script are retained in
`experiments/fortran-profile`.

The generator now interchanges only the innermost coefficient and last-array-index
loops in compact updates, keeping each output's arithmetic accumulation order
unchanged. Compiled tests compare the generated DP and QP routines with the legacy
order across rank pairs, all dimensions from one to four, sparse coefficients and
large/small complex inputs: 2,880 full-array comparisons per precision are exact.
Copied generated outputs also agree exactly on all printed quantities and statuses
at all 100 stored qcd-2 and qcd-3 points.

Seven alternating exploratory pairs on CPU 2 gave median ratios of `0.940346`
for qcd-2 and `0.868791` for qcd-3, relative to their unmodified copies. Concurrent
builds caused substantial noise, so these are preliminary observations, not accepted
speed claims. Paired-bootstrap 95% ratio intervals of `[0.913864, 1.033940]` and
`[0.398845, 1.111719]` leave a strict speed-benefit gate unresolved. Production
retention is conditional on isolated repeated timing and
the full process gates. Scripts, generated diffs and sample-level evidence are
retained in `experiments/wl-layout`.

## Limits and optimization follow-up

Code review and scalar tests cannot establish correctness on all high-rank,
ill-conditioned, massive or electroweak inputs. Native failures that abort a process
cannot be converted into a normal Rust error by `catch_unwind`. The report therefore
retains missing/failed comparisons and rescue usage explicitly.

Future Fortran optimizations require profile evidence, original-backend regression
comparisons and isolated timing measurements. The scratch experiment above is
retained as a negative result, not claimed as an implemented improvement.
