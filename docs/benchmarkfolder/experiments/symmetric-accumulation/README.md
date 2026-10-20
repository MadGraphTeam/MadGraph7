# Symmetric accumulation experiments

This directory contains isolated research prompted by the recursive symmetric
tensor construction in [arXiv:1604.01363](https://arxiv.org/pdf/1604.01363).
`paper-review.md` distinguishes that paper's projector recurrence from MadLoop's
already independent monomial coefficients. None of the experiments below
changes the production generator or the frozen benchmark protocol.

For a linear vertex, each output coefficient can be gathered from at most five
predecessors. `gather.py` proves that its predecessor map contains every original
scatter operation exactly once and preserves the original `M,K` summation order
of each output. The compact updater is the target; the expanded rank-one updater
has a different grouped arithmetic order and is not transformed.

## Retained rejected variants

`table-gather-v1/` retains the table-driven gather and its complete generated
sources. All 24,000 compiled DP/QP output-bit, input-bit and exception-flag
comparisons passed. Its complete guarded kernel was slower in the synthetic
trials: roughly 2.18–2.81 times the original DP kernel and 1.03–1.10 times the QP
kernel. It was rejected without a full-process timing run.

`unrolled-gather-v1/` retains a separate variant with literal predecessor indices.
Its 24,000 comparisons passed. DP remained substantially slower; QP ranged from
0.98 to 1.15 times the original in these trials, without a consistent benefit.
It was also rejected. The retained execution note records a deliberate pause
during one correctness run while the independent NLO export defect was fixed.

The early unbounded-DP exceptional-input mismatch and a rejected scalar
`ASSOCIATE` classifier under `-fno-automatic` are retained in
`rejected-unbounded-dp/` and `rejected-scalar-associate/`. The bounded implementation
uses integer layout checks and the accepted array `ASSOCIATE` form.

## Separate exact-product sharing

`shared.py` retains the original scatter traversal. It caches only an identical
complex product that is used again:

- Full updaters require equal contracted index `K`, equal `L,I`, and identical
  complex bits for the vertex operand.
- Terminal updaters also require equal `J`, because their consumed diagonal has
  `I=J`.
- Every original output addition remains, in its original order. The first
  occurrence performs the original multiplication; later uses copy its complete
  complex result.
- A source-derived group is accepted only after a runtime bit-equality check,
  including signed zeros. Unsupported shapes, layouts, rounding modes,
  exceptional/extreme operands and enabled traps use the original routines.
- Finite operand exponent bounds prevent overflow, underflow and invalid
  operations in the accepted path. Repeating an identical multiplication cannot
  add a new sticky flag once its first occurrence has executed. All traps must
  be disabled before multiplication reuse is allowed.

`shared-product-v1/` contains the initial full-QP rank-two/rank-three trial for
the `MP_FFV1L2_1` source map. All 12,000 exact comparisons passed, as did five
enabled-trap fallback controls. Seven alternating synthetic kernel trials gave
ratios 0.837–0.853 with both dense and sparse inputs, including guards, dispatch
and cache clearing. Those runs shared the host with other work: they establish
a promising kernel experiment, not a full-process speedup or an acceptance
result.

The generalized full/terminal checks in `shared-product-v2/` passed 228,000 exact
comparisons and 130 enabled-trap controls across all 26 rank-two/rank-three
map/terminal combinations. `check_shared_general.py` exercises every observed
nonempty same-operand map.
`prototype.py` verifies the matching current ALOHA source expressions, binds each
original routine to the rendered generator, appends private helpers, and changes
only selected QP call names. The entire original polynomial source remains an
unchanged prefix. The complete QCD3 trace in `qcd3-sharing-trace-v2/` reproduces
every printed coefficient, accuracy and status for all 100 points exactly.
Statuses remain 98 DP (`216`) and two QP (`328`). All 7,296 selected runtime calls
used sharing; none fell back. `trace-summary.json` binds the raw comparisons and
counter data. These are sampled external outputs, not a claim that the process
driver exposes every internal binary128 intermediate. The concurrent,
instrumented timings are diagnostic only.

`shared-product-production-flags/` passed another 72,000 exact comparisons and
40 enabled-trap controls using the actual standalone and NLO polynomial flags.
Those runs omit `-frounding-math` and `-fcheck=all`, and include NLO's
`-fno-automatic`, closing the difference between the original strict test flags
and production compilation. Rank-two and rank-four full/terminal kernels are
covered. The broader all-map rank-four campaign is retained separately in
`shared-product-rank4-v3/`: all 21 actual QCD4 map variants passed 186,000 exact
comparisons and 105 enabled-trap controls.

The uninstrumented process timing is separate from the structural trace.
`quiet_shared.py` retains prospective core selection, topology and periodic CPU
counters. CPU3 was rejected before timing because an unrelated long-running
process occupied it. CPU206 was selected prospectively as idle and has no online
SMT partner. The explicitly allowed corrected-original reference worker remains
on the separate CPU207; the host is not claimed to be exclusively reserved.
The seven-pair experiment retains every pair and requires the trace's exact
100-point outputs and status histogram.
It completed with all comparisons exact, but did not demonstrate a process
speedup: the shared/original median ratio is 0.998449, with paired bootstrap
95% interval [0.966772, 1.026978]. The lazy-cache version is not promoted.
`qcd3-sharing-quiet-v2/result.json` retains all fourteen measurements and the
fixed 10,000-resample calculation. Monitor data include the complete 301-second
build/initialization/evaluation period; they do not establish exclusive host
ownership.

`inventory.json` binds the earlier QCD3/EW2 source inventory.
`final-inventory.json` binds the final fixed QCD3/QCD4 CLI exports. The final
QCD3 static potential is 35,900 reusable products out of 255,220 after literal
vertex-zero skips (14.1%). QCD4 has 494,200 out of 3,226,340 (15.3%), including 136
rank-four terminal calls. These are static counts before runtime numerator-zero
skips; they are not measured executed-product savings. The separate QCD4
CutTools warmup profile in `policy-v2/reference-diagnostics/qcd4-ct-warmup-20261020`
motivates extending the rank coverage, but it does not establish a process
speedup.

## Reproduction

Run from the repository root using the pinned development environment:

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/experiments/symmetric-accumulation/inventory_final.py
python docs/benchmarkfolder/experiments/symmetric-accumulation/check_shared_general.py \
  --tag NEW_UNIQUE_SAFETY_ID
python docs/benchmarkfolder/experiments/symmetric-accumulation/run_shared.py \
  --case qcd-3 --variants shared-trace --run-id NEW_UNIQUE_ID \
  --repetitions 1 --loops 1 --cpu 3
```

The exact-suite script intentionally refuses to overwrite its report/work directories.
Retained versioned source snapshots reproduce historical variants; the current
research scripts may add later controls. Process trials use original Ninja
backend 6, unchanged 100-point input streams, common strict stability settings,
and complete coefficient/accuracy/status comparisons. They do not start a
NinjaRS/Symbolica worker. A coordinated quiet run and an uninstrumented process
comparison remain prerequisites for any retained performance optimization.
The first-round files are immutable and listed in `SHA256SUMS.json`. Subsequent
static-owner experiments live in the separate sibling `symmetric-static-owner/`
directory and do not replace these results.
