# Provisional shared-host timing proposal

This is an unexecuted proposal. Hua-Sheng will perform precise final timings on
his local machine, possibly with an agent. Shared-host observations cannot
qualify the 1.05 performance bound. The existing `timing-qualification-1`
quiet-window controller remains preserved and unrun.

The ordinary numerical campaign is unchanged: all eleven processes, baseline
CutTools plus candidate backends 1, 6, 7, 8 and 9, and all 100 saved points under
the registered physics and stability settings. Its completion is a prerequisite
for this timing proposal; it is not replaced by provisional timing checks.

## Fixed scope

Use the existing full-amplitude worker and exact compiled exports in private
copies, with private cards and filters and immutable shared libraries. Keep the
full original 100-point learning history, complete helicity/color sums and full
100-point timed traversals. No point subset, extra fixed-precision mode, changed
numerator, lower stability requirement or modified Fortran driver is proposed.

Each completed comparison has three paired repetitions with alternating order
baseline/candidate, candidate/baseline, baseline/candidate. Calibration chooses
an integer number of complete 100-point traversals, targeting one second per
batch. Retain the exact traversals, duration, initialization-plus-warmup interval,
all values and final accepted backend/precision status counts. A batch longer
than the target is retained as measured; no short subset substitutes for it.

A comparison that cannot finish initialization, calibration and all six batches
within its administrative cap is unavailable for the requested comparison.
Retain every completed and partial observation, the timeout phase and cleanup
record. Partial results may inform diagnostics, but do not become a completed
three-pair result. Do not retry timed-out or slow cases automatically.

## Predetermined schedule and bounds

Prepare private copies without native evaluation before launching the bounded
controller. The launch budget is 1,800 seconds total, including lock waiting,
post-preparation verification, initialization, calibration, measurements and
cleanup. Enforce the deadline in the parent supervisor, allowing a reserved
cleanup interval and stopping admission early enough to keep that interval.

Schedule all eleven chain comparisons first, in this order:

1. `qcd_n1_uux_z`
2. `ew_n1_udx_wp`
3. `qcd_n2_gg_ttx`
4. `ew_n2_epem_mupmum`
5. `li_gg_h`
6. `mixed_uux_ga`
7. `li_gg_hh`
8. `qcd_n3_uux_zgg`
9. `ew_n3_epem_ttxa`
10. `qcd_n4_uux_zzgg`
11. `ew_n4_epem_mupmumaa`

Propose a 120-second all-in cap for each chain comparison, below the permitted
180-second maximum, so all eleven can receive an admission attempt within the
overall budget. Exact cancellation/cleanup allowances must be validated before
execution; unresolved cleanup blocks further admission. A comparison waiting
for another approved numerical job can exhaust its cap without starting a
worker. This is unavailable evidence, not an accuracy failure.

Then attempt standalone comparisons in backend order 1, 6, 7, 8, 9, first for
`qcd_n1_uux_z`, then `ew_n2_epem_mupmum`. Continue with the remaining cases in the
chain order above, omitting those two already covered cases, while the fixed
overall budget permits. Each later comparison has at most 180 seconds, reduced
by the remaining overall allowance. Every unattempted schedule row remains
explicitly `not_run_budget`; no case is silently dropped.

Chain choices come from the saved baseline generation record and candidate
prefixing of backend 9. Standalone backends 1/6/7/8 compare the same explicit
backend between baseline and candidate; candidate 9 is reported alongside
baseline CutTools 1. The latter is descriptive, not an additional speed gate.

## Provenance and reporting

Before launch, freeze the controller, worker/parser/helper sources, complete
ordered schedule, campaign manifest, source/build/executable/library hashes,
point and card hashes, comparison caps, CPU affinity and thread environment.
Use a nonexclusive allowed CPU affinity only; do not reserve CPUs, request a
quiet window or stop other users' work. Continue to serialize every process
using backend 8 or 9 on the common numerical lock. Record host-load snapshots;
shared-host interference remains a limitation even when a sample looks quiet.

Retain raw timings and coefficient/status streams and reparse them with the
saved protocol. Apply the same numerical comparator and report its result
separately from duration completeness. Report absolute seconds per point,
medians and descriptive paired ratios, plus the actual accepted DP/QP counts.
Do not produce a confidence-bound acceptance verdict, `passed` speed flag,
formal campaign gate or claim that three pairs establish precise performance.
The result schema must set `performance_acceptance_eligible=false` everywhere.
Initialization, failures, unavailable cases and local TODOs remain visible.

The controller and its schemas require root review and focused timeout,
provenance and reporting tests before any native launch. No measurement has
been authorized by this document alone.
