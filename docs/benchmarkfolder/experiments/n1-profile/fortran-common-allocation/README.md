# Common TIR workspace allocation experiments

These isolated variants change only a clone's generated `TIR_interface.f`.
Production templates, frozen campaign outputs, compiler flags and reduction
libraries remain unchanged. The completed comparison establishes no speed
benefit, so neither variant is proposed for production.

`tir-alloc` changes the three local arrays `PCT`, `PDEN` and `M2LCT` to
allocatables. Their original bounds are allocated only for backends other
than 9, after the existing invalid-backend rejection. TIRust never references
these arrays. Every returning path explicitly deallocates an allocated
workspace: Golem's early return, TIRust refusal (normally unallocated), and
ordinary completion. Explicit cleanup also covers `-fno-automatic` builds.
The same single allocation creates all three arrays, and no backend receives
an unallocated dummy argument. Existing STOP paths terminate the process.

`switch-guard` keeps the public DP/QP signatures and original integer CTMODE
and topology-size checks. Only an actual reversal enters a private worker,
which contains the original automatic arrays and unchanged reversal body.
This preserves copy order, in-place mutations and floating-point operations;
it does not replace reversal with swaps or omit dummy-PDEN writes. `combined`
applies both independent changes.

The shared saved cache, pole-mode and library state stay in TIRLOOP. Legacy
preprocessing and conservation checks still occur before a cache hit is
consulted; refusal and contraction order are unchanged. Generic backends are
not restricted to TIRust's six-propagator domain. No new persistent scratch
or reentrancy guarantee is introduced into the existing COMMON-based serial
workflow.

A read-only audit of the frozen n1 object finds three malloc/free pairs in
TIRLOOP and three in each SWITCH_ORDER routine before its original guards.
For the observed two DP calls per n1 amplitude, `tir-alloc` can remove six
pairs and `switch-guard` three further pairs from CTMODE1. CTMODE2 retains its
original arrays. This is an allocation-count hypothesis, not a speed estimate.
The old momentum-stage measurement was 0.785 microseconds per amplitude and
included both reversal calls plus other preparation; common TIR entry
allocations were outside those stage brackets. The experiment is unlikely to
explain the full prior 9–11% runtime gap without additional evidence.

Preparation verifies frozen sources, compile recipes, injected environment,
static dependencies, and exact reconstruction of the original executable
before replacing only `TIR_interface.o`. Example commands, after coordination
with other numerical jobs:

```sh
source tools/tirust_env.sh
python3 docs/benchmarkfolder/experiments/n1-profile/fortran-common-allocation/prepare_variant.py \
  --name combined-build --variant combined --build
```

The separately retained stub corpus exercises actual rendered routines with
recording backend stubs, including cache hits, backend/mode changes, refusal,
Golem return and repeated calls. Its checks are distinct from physical
full-amplitude parity. `run_common.py` is a local fork of the frozen diagnostic
triad runner; it accepts only an isolated `TIR_interface.o` replacement, uses
the same explicitly selected DSO for original and variant, and requires the
shared numerical lock. All its timings are diagnostic, not acceptance results.

`stub-parity.json` records 36 passing builds: original plus the three variants,
under O0, O3 and O3 with static locals, for cached interference, cached
amplitudes and uncached amplitudes. Each build exercises 645 TIR calls and
432 DP/QP switch cases (38,772 calls total). Deliberately omitted cleanup and
an incorrect QP switch guard both fail the corpus. The original source cannot
compile with a constant cache size of zero because its unused ELSE branch
contains `MOD(...,0)`; that separate baseline limitation is recorded, not
changed by this experiment. `template-used.inc` and `check-layout-used.py`
freeze the original executed sources. The current replay runner defaults to
the frozen template and a fresh work directory; `--template` permits a
separate regression experiment after a future template correction.

`combined-native-parity-1` compares the combined variant with the original
for all five explicit reducers on the same 100 physical n1 points. All 500
original/variant pairs match coefficients, returned status, accuracy and
precision histograms exactly. Each reducer serves all 100 points in DP;
this physical sample does not establish QP amplitude coverage. Both outputs
load the same frozen split-workspace TIRust DSO (`1c368801…`). The host policy
and generated process are the historical qualified-attempt-3 snapshots,
not a newly generated campaign. These checks establish local equivalence;
they do not replace a later full campaign for any proposed production change.

`combined-triads-1` retains seven alternated, warmed triads on CPU 9 using
the same split-workspace DSO for the original and combined variants. Each
measured run evaluates the 100 frozen points 200 times; calibration uses the
same count. All original/combined coefficient bits match. The median times
are 89.000 microseconds for baseline NinjaRS, 98.546 for the original TIRust
adapter and 99.622 for the combined allocation variant. The paired geometric
mean combined/original ratio is 1.01120, with a 95% Student-t interval on
paired log ratios of [0.99373, 1.02898]. This does not establish a benefit.
The original/baseline ratio is 1.10966 [1.07749, 1.14278], and the
combined/baseline ratio is 1.12208 [1.08807, 1.15716]; these comparisons are
diagnostic only and do not replace the campaign's acceptance gate.

The native slot was serialized, but the shared host was not quiet: another
agent had an installer build assigned to CPUs 2–5, and even simple shell
reads stalled during the final part of the observation window. Separately,
the parent agent reported an `uptime` observation at 22:45:07 UTC with load
averages 28347.78, 14255.10 and 5659.77. That parent-reported observation is
additional context, not an execution-time sample recorded by this runner.
The retained per-run host snapshots and raw timings are unchanged. These
conditions further limit interpretation; the result does not establish a
universal regression or permit a performance acceptance claim.

No decomposition timing was run after this combined result. The individual
`tir-alloc` and `switch-guard` builds remain source/compile/stub-qualified
experiments only. The production template remains unchanged by this work.
