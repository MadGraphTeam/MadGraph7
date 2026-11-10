# Fortran performance follow-up audit

This is a source-bound audit, with **no implementation, build, native execution,
new measurement or performance benefit claim**. Production sources, generated
campaigns and active provisional results were not changed or inspected. The
private symmetry-v2 adapter is the starting point for any later adapter
experiment; its qualification does not automatically qualify another variant.
`audit.json` binds every source and historical observation used here.

The earlier n1 profile recorded two N=3/rank=2 DP calls per amplitude. Its raw
stage medians were 0.750 microseconds for initialization/budget, 0.785 for
ordered momenta, 0.701 for the kinematic matrix, 0.120 for invariant conversion,
8.196 for the wrapper/Rust call, 0.878 for mapping/validation and 0.586 for
contraction/final checks. These are historical, instrumented attribution
numbers, **not costs measured for symmetry-v2 or estimated removable work**.
Allocation and IEEE routine-entry costs lie outside these brackets. The new
symmetry guard did not exist in that profile.

## Two experiments to consider

1. **Use the generated coefficient-count table in common TIRLOOP.**
   `TIR_interface.inc:233–236` still computes the cumulative coefficient count
   with an integer loop. `q_polynomial.py:192–194` already generates `NCOEF_R`;
   the TIRust adapter uses it at lines 75, 170 and 242. For ranks within the
   generated table, use that lookup and retain the original integer loop as
   the fallback outside its bounds. The certified rank-0..6 counts are
   1, 5, 15, 35, 70, 126 and 210.

   The hypothesis is fewer integer operations on every common TIR call,
   including Fortran tensor-cache hits. No floating-point expression, tensor
   traversal, contraction sum, cache lookup or failure branch needs to move.
   Measure the count stage/call frequency first, then warmed full-amplitude
   candidate/control ratios; a positive microbenchmark alone is insufficient.
   Expected benefit is small and currently unknown. Risks are out-of-range
   indexing and a new generated-module build dependency, especially clean
   parallel builds and OLP subdirectories. Verify identical counts and exact
   coefficient/status/input/IEEE parity across all five backends, cached and
   uncached modes, rank boundaries and the existing common-interface corpus.

2. **Consider an integer-only fast exit for a bitwise symmetric matrix.**
   The private v2 adapter first transfers all invariant words to integer storage,
   scans every exponent field, then compares off-diagonal floating values
   (`v2/TIRUST_interface.f:100–114` and `243–257`). After the same `TRANSFER`,
   an integer scan could bypass the existing repair block only when every
   off-diagonal pair has identical bits. Any unequal pair must execute the
   complete unchanged v2 finite guard and repair. This is a proposal, not a
   patch. It may add more work than it removes and has no measured justification
   yet; profile the qualified v2 guard before preparing a candidate.

   In the proposed bypass case there is no possible matrix write: finite equal
   pairs remain equal, and v2 skips all floating comparisons if any entry is
   nonfinite. Bit equality preserves NaN payloads and signed-zero inputs;
   bitwise unequal signed zeros must take the unchanged fallback. N=1 has no
   pairs. Keep every kinematic arithmetic statement, upper-triangle convention,
   input mutation and later finite-result check unchanged. Measure the actual
   bypass rate and guard time separately from uninstrumented full-amplitude
   paired timing; do not infer a bypass rate from prior native refusal counts.
   Require the complete existing v2 component/trap/negative-control fixture
   against v2 as control, all rounding modes and entry flags, both precisions,
   nonfinite diagonal/off-diagonal values, signed zeros, subnormals and enabled
   traps. Standard IEEE parity does not by itself qualify arbitrary processor
   modes for denormal operands. Retain physical amplitude/status/accuracy and
   primary/check/cache parity, including the EW n3 full-100 sample.

Both experiments require a new isolated source/build identity and a root review
before implementation. Change one mechanism at a time. Promotion requires
numerical qualification and measured full-amplitude benefit. Final precise
performance acceptance remains Hua-Sheng's local-machine TODO.

## Other opportunities examined

- Automatic scratch allocation is real, but pointer scratch, sequence-associated
  scratch and common allocation changes already had diagnostic intervals
  spanning no benefit. Adapter-only `-O3` did too. Repeating these without a
  new attribution result has no current justification.
- DP and QP contractions already traverse the contiguous coefficient dimension
  in scalar summation order (`TIR_interface.inc:320–328`,
  `TIRUST_interface.inc:241–248`). Reassociation, fusion with conversion or a
  cache-layout rewrite would need a separate numerical/IEEE contract analysis.
- A repeated budget request already avoids cache clearing in Rust when the
  value is unchanged (`vendor/tirust/src/lib.rs:660–682`), but still acquires the
  context mutex. Fortran memoization is not merely removal of an assignment:
  context recreation, failed setters/poisoned contexts and public error-message
  clearing must remain observable (`mtirust.f90:109–120,178–190`). The historical
  initialization/budget stage does not isolate that mutex's cost. No such edit
  is recommended without attribution and a lifecycle/error contract.
- Backend 9 already skips common legacy momentum preprocessing before its
  Fortran cache lookup. Moving the generic lookup earlier would skip legacy
  validation/arithmetic and alter input/IEEE behavior. Numerical cache reuse
  across independent stability checks must remain prohibited. The rejected
  early-doomed-DP-pass audit is not reopened here.
- Keep both DP and QP raw finite-result checks. The earlier proposed DP scan
  removal was rejected because a successful but faulty provider must still be
  rejected by the public diagnostic.
- COLLIER cache repair is outside this audit and the user's requested scope.

Replay only the source/hash and historical-summary checks, without a compiler,
loader or numerical worker:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/fortran-performance-followup-audit-1/verify.py
```
