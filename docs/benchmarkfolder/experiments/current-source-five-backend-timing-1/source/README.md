# Current-source descriptive five-backend timing

This source proposes three fixed cases (`qcd_n1_uux_z`, `qcd_n2_gg_ttx`,
`ew_n2_epem_mupmum`) from `entry-latch-cache-guard-1`, using the registered
0fcd source bytes and eb3d installed TIRust. The separate new four-lepton
preparation is not part of this invocation. No previous timing is promoted.

`--preflight` hashes retained/current inputs only. `--prepare` additionally
copies fifteen private exports, without compilation, relinking, loading a
reducer, model setup, or amplitude calls. Each keeps the original executable,
objects, libraries, generated sources, physical card and exact 100 points.
Only private numerical cards/backend selection and scratch/filter files may
change. Every native worker starts a new full 100-point learning history;
the standard configure helper deletes all HelFilter and LoopFilter files.

`--execute` requires the reviewed source-index SHA and separately reviewed
prepared specification SHA. It owns one child per case and five persistent
workers, evaluated serially under the common numerical lock. All cases use
nonexclusive CPU12, the existing 3,600-second worker policy, automatic
precision, MLStabThres=1e-6 and LoopPoleMode=0. Both COLLIER pole flags are true
and its pole cache remains false. All other registered cards/cache settings
are unchanged. No CPU is reserved; other host activity is recorded.

All five complete learning traversals must pass the existing 3e-6 numerical
comparator against retained same-case CutTools data before calibration.
The 1e-8 ordinary target is reported independently. Two retained calibration
batches (one traversal, then a 2-second target) set each backend's fixed loop
count. Ten cycles follow: five cyclic rotations of 1,6,7,8,9, then each rotation
reversed. Every backend occurs twice in every position. Each timed batch
contains an integer number of complete 100-point traversals. The unchanged
worker retains coefficients for the last full traversal and status counts for
every timed call; numerical comparison is required for every retained batch.
Earlier traversals within a batch have status/precision counts, not retained
coefficient buffers. No failed result buffer is used.

The clock is Fortran SYSTEM_CLOCK wall elapsed divided by 100 times loops.
Summaries retain all ten samples, median, unscaled MAD, extrema, actual batch
durations and the median of ten within-cycle ratios to NinjaRS8. A 2-second
target is calibration, not a guaranteed minimum; actual durations remain
visible. There is no confidence interval or <=1.05 acceptance claim. READY's
initialization interval includes first amplitude plus full learning and
excludes process startup/SETPARA. Backend/precision usage means final accepted
results, not internal attempts; FPE flags and internal retry counts are not
observed by this worker.

Each case has an all-in 600-second bound, including a reserved 60 seconds for
cleanup; the invocation has an all-in 1,800-second bound. Lock wait, source
guards, loader queries, native work, raw copying, postchecks and result writes
are included. Private preparation is outside this measurement budget. Existing
reviewed absolute alarms and subreaper/identity-checked cleanup cover detached
workers. Cleanup uncertainty or protected-input drift stops admission. There
are no retries, reduced cohorts or substituted points. Missing/time-limited
cases remain unavailable. Postdeadline administrative failure publication is
explicitly outside the bound and cannot qualify a measurement.

Before Fortran startup, the exact actual ELF PT_INTERP is queried against
prebound providers. The historical ldd foreign-loader alias is retained as
query history and excluded explicitly from actual-provider claims. Each live
worker's /proc maps is checked before/after timing against the actual closure,
its unchanged private executable, and the pinned locale archive. Source,
object, compilation, card, point, native/config and runtime receipts are kept.

Run from repository cwd after `source tools/tirust_env.sh`:

```
python3 -B .tirust-build/current-source-five-backend-timing-controller-1/run.py --preflight --source SOURCE_SHA
python3 -B .tirust-build/current-source-five-backend-timing-controller-1/run.py --prepare --source SOURCE_SHA
python3 -B .tirust-build/current-source-five-backend-timing-controller-1/run.py --execute --source SOURCE_SHA --prepared PREPARED_SHA
```

Native execution is a separate admission after the prepared specification is
reviewed and the four-lepton preparation has closed. This source does not
grant that admission. Closed data belong to
`docs/benchmarkfolder/experiments/current-source-five-backend-timing-1`;
private exports remain under the matching ignored work directory.
