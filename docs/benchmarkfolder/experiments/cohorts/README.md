# Precision cohorts: isolated diagnostic

These runs measure subsets of the original ordinary point stream with the
natural automatic rescue policy. They are **not performance acceptance data**.
The diagnostic adds per-call integer bookkeeping and finite-value checks inside
its timed loop, and its host-load snapshots can include concurrent compiler or
other jobs. The acceptance executables and their timings remain unchanged.

The worker first warms the complete original 100-point stream, then classifies
all 100 points in another untimed traversal. It freezes original point indices
before any calibration. Timed calls continue to use the original point array,
cards, helicity/color sums, and automatic stability/rescue settings. Every call
records its accepted status and the four immediately sampled, absolute
NinjaRS retry counters. Schema 2 additionally records both host reduction FPE
flags, without resetting them. A migration, nonfinite result, negative accuracy,
inconsistent counter, or reported reduction failure invalidates a pure-cohort
timing. Original membership is never rewritten to hide a migration.

The `dp` label means accepted binary64 arithmetic without a host precision
rescue or observed NinjaRS compensated retry. It includes ordinary stability
checks and backend method selection. The TIRust embedded options disable hidden
precision tiers and partial rescues but allow same-precision method selection.
The `compensated_dp` label identifies a successful NinjaRS compensated retry
that still returns status 218. The `qp` label includes failed DP work followed
by natural QP rescue. The difference between these cohort times is not an
incremental rescue cost: the subsets contain different physical points.
Backend comparisons require identical original point IDs, such as the
intersection of their frozen DP subsets. Accepted-status histograms alone do
not count internal attempts. False FPE flags do not prove numerical accuracy.
Snapshots follow public `SLOOPMATRIX_THRES` calls; they do not reconstruct every
internal bootstrap invocation or every internal algorithm attempt.

The driver is compiled and linked in an ignored private clone. Numerical
objects, model/DHELAS archives, external libraries, compiler policy, physical
cards, and exact decimal point records are checked before and after. Each run
retains its source driver, protocol copies, expanded driver-only build commands,
executable hash, runtime identities, frozen membership, all per-call traces,
calibration batches, seven alternating-order timing repetitions, and failures.
Directories are append-only. Reparse does not require native execution.

From the MadGraph root:

```sh
source tools/tirust_env.sh
python3 -m unittest tests.unit_tests.various.test_tirust_benchmark_cohorts
python3 tools/tirust_benchmark_cohorts.py run \
  --case qcd_n1_uux_z --backend 9 --label a-new-label --cpu 2 --timeout 600
python3 tools/tirust_benchmark_cohorts.py validate --label a-new-label
```

`pilot-qcd-n1-tirust` retains an initial build-preflight refusal; no numerical
result was produced. `pilot-qcd-n1-tirust-2` is the successful schema-1 initial
qualification: all 100 points stayed in DP, with no observed NinjaRS counters or
cohort migrations. Its seven diagnostic batches have a median 70.5688 µs per
point. It predates FPE-flag tracing and must not be represented as observing
those flags. Its exact protocol and results remain unchanged and reparse
successfully with the explicit legacy schema.

Schema-2 qualification is recorded in separate directories. This diagnostic
does not change the authoritative failed original n1 performance comparison.
