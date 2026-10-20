# Further Fortran quadruple-precision experiments

These are isolated experiments, not production changes or final performance
acceptance data. They use copied generated processes and original Ninja (backend
6), so they do not consume the NinjaRS/Symbolica worker slot. The original
generated outputs, physical cards and retained point streams remain untouched.

## Evidence and limits

`analyze.py` decodes the retained QCD3 point-44 profile. It finds 842 Fortran QP
numerator samples among 1,911 samples, dominated by `UPDATE_WL_2_1` (329),
`UPDATE_WL_3_1` (189) and `UPDATE_WL_1_1` (126). The profiled executable predates
the sparse QP production optimization. These counts identify useful code to
inspect; they do **not** measure the fraction remaining in the final candidate.
Only four numerator samples are in `CREATE_LOOP_COEFS`, and one has a memory
clearing/copying call chain. The final generated polynomial kernels contain no
dynamic allocation.

`source_inventory.py` analyzes the final generated QCD3 coefficient graph.
Its 388 update nodes are all needed; 224 are terminal inputs to the diagonal
`MERGE_WL` trace. There are 223 repeated predecessor uses, so a per-node cache
could avoid some repeated classification scans, but it would need explicit
helicity/phase-point invalidation and additional storage. No such cache has been
implemented. The terminal nodes account for 912,350 of 1,238,879 dense complex
products per helicity. Omitting only their unconsumed off-diagonal components
would remove at most 55.21% of this dense count. Existing zero skipping makes
that an upper opportunity estimate, not a runtime prediction.

## Rejected classifier experiment

`prototype.py` removes redundant finite tests after the existing finite guard
and replaces QP equality with integer zero classification on that proven path.
The actual fixed-form transformed kernels passed 30,000 output/input-bit and
IEEE-flag comparisons, including all rounding modes, signed zeros, exceptional
operands, empty dimensions and a rejected-layout fallback.

The three alternating full QCD3 pairs in
`qcd3-classifier-three-pair-retry1/result.json` reproduce every printed
coefficient, accuracy estimate, per-row status and timed status count exactly.
The ratio of medians is **1.08244**, with prototype/original times of
0.085543/0.079028 seconds per point. These exploratory measurements were not
quiet acceptance runs; they demonstrate no benefit, so this change is not
proposed for production. The initial build failure and the earlier logical-only
test are retained separately rather than overwritten.

## Private terminal updater prototype

`terminal.py` proves the entire generated multiple-precision wavefunction graph
before redirecting a call. It rejects unknown consumers, nonunique producers,
input/output aliasing, missing predecessors, unclassified COMMON accesses and
unexpected merge code. It redirects only terminal nodes consumed by the
diagonal expression `WL(I,J,I)` in `MERGE_WL`. Existing public full updaters and
all nonterminal calls remain available and unchanged.

The private helper preserves clearing and the complete arithmetic sequence for
every consumed diagonal component. Unconsumed off-diagonal scratch components
are intentionally not materialized. This is therefore a generated-DAG
optimization, not a drop-in replacement for an arbitrary direct updater call.
The QCD3 graph has 224 eligible calls; the mixed-order control has 151.

The fast path requires recognized IEEE binary128 storage, nearest rounding,
an already-set INEXACT flag, disabled INEXACT halting, finite active operands
with each nonzero component's unbiased exponent in [-1000,1000], at most 16
wavefunction components and at most 65,536 coefficients on each side.
Otherwise the original public updater runs.

Every admitted component is an integer multiple of 2^-1112 and has magnitude
below 2^1001. Products therefore lie on a grid no finer than 2^-2224.
Even the conservative bound of 2^36 accumulated complex products leaves
component magnitudes below 2^2040, including rounding; cancellation cannot
create values below that grid. Thus the omitted
operations cannot raise overflow, underflow, division-by-zero or invalid.
They may raise INEXACT, which is already sticky and cannot trap under the
guard. Integer operand classification does not signal on a signaling NaN.
The original full arithmetic remains the fallback outside this domain.

GNU Fortran's IEEE procedure entry clears local sticky flags and merges them
back on return. Consequently the helper cannot discover the caller's existing
INEXACT flag by querying after entry. The prototype samples it immediately
before the private call and passes it as a logical argument. This adds an IEEE
entry/exit boundary to each affected coefficient batch. The explicit runtime
fixture in `ieee-entry.f90` verifies that INEXACT halting remains enabled through
nested procedure entries (`T T T T`), while the sticky flag is local
(`T F F T`) and restored to the caller. The generated processes contain no
existing flag queries that this new batch-local state would intercept.

`terminal-exact.json` records 48,000 comparisons against the independent legacy
arithmetic, with both expanded and compact kernels at `-O3` and with
`-frounding-math`. Each fixture confirms that both the fast and fallback paths
actually execute. `terminal-bounds.json` adds 36,000 checks at both admitted
exponent boundaries, immediately excluded binades, cancellation cases and the
largest value in the upper admitted binade. Consumed diagonal bits, unchanged
input bits and cumulative IEEE flags agree; fallback calls also compare the
full output.

The full-process three-pair retry is retained in
`qcd3-terminal-three-pair-retry1`. The initial
`qcd3-terminal-three-pair` attempt stopped at compilation because the
experimental renderer dropped fixed-form statement labels; its failure log is
retained and no amplitude evaluation ran. The revised adapter replaces only
individual call spans and preserves every other original statement and label.
All three retry pairs reproduce every printed coefficient, accuracy estimate,
per-row status and timed status count exactly: each traversal has 98 stable DP
points (216) and two successful QP rescues (328). The ratio of medians is 0.48434,
with terminal/original times of 0.19577/0.40420 seconds per point. CPU2 received
roughly one quarter to one third of a CPU during parts of this run, and
availability varied; these numbers are exploratory and cannot establish a
retention decision. `check_terminal_counts.py` instruments a separate copy and
resets counters before the final traversal to measure actual fast/fallback use
without changing the timing executables.
The successful trace in `qcd3-terminal-counts-retry1` records **10,752 fast
calls and zero fallbacks**, with all numerical rows and statuses exactly equal
to the original. The earlier counter setup failure is retained separately; it
rejected an indentation mismatch before compilation or evaluation.
`check_mixed.py` separately exercises all mixed-order physical rows with
`CTModeRun=4` and private-path counters. This is explicitly an unchecked QP
diagnostic, not stability acceptance or a timing benchmark. Full-process
validation and coordinated quiet measurements are still required before
considering production use.

Two mixed-order attempts are retained as incomplete diagnostics:
`mixed-terminal-forced-qp` and `mixed-terminal-v2-first2`. Their long execution
was caused by the driver, not an established cost of mixed-order QP evaluation.
The generic driver's forced stability check remapped the deliberately
unchecked `CTModeRun=4` answer to initialization status 140 on every call, so
the warmup condition could never complete. The cancellation records and
superseding diagnosis retain this distinction. The corrected diagnostic uses
`FORCE_STABILITY_CHECK(.FALSE.)`, explicitly requires unchecked-QP status 120,
and bounds warmup by `max(20,2*npoints)` calls. The first two unchanged points
then completed with all five output rows per point exactly equal and 3,624
private fast calls, zero fallbacks, across its diagnostic traversals. This
prefix check is retained separately as `mixed-terminal-v2-first2-fixed-driver`;
it is not a replacement for the complete ten-point control.
The subsequent `mixed-terminal-v2-full10-fixed-driver` completed all ten
unchanged points: all 50 coefficient/accuracy/status rows and timed status
counts match exactly, with 18,120 private fast calls and zero fallbacks across
the diagnostic traversals. Both variants finished initialization and their
ten-point warm pass within 13 bounded calls. All rows retain unchecked-QP
status 120; this establishes structural numerical parity, not stability.

The v2 prototype now applies the committed `ASSOCIATE` finite-classifier fix
equally to its baseline and terminal copies, retaining the preparation hashes.
Its bounded classifier also uses association-only scratch. The separately
retained `terminal-associate-v2.json` passes **108,000** comparisons, including
the actual NLO `-fno-automatic` flags. Earlier v1 results retain their exact
earlier source snapshots; they are not relabeled as v2 timing evidence.

`qcd3-terminal-v2-quiet-seven` contains seven alternating pairs on the reserved
CPU207 with the other task-owned builds and numerical jobs paused. Every
100-point traversal matches the retained unmodified-updater numerical rows
and status counts exactly (98 stable DP points, two QP rescues). Median time
falls from 0.0810265 to 0.0663484 seconds per point: ratio 0.818848, paired
bootstrap 95% interval [0.787189, 0.837055]. The comparison baseline is the
current sparse-QP generated candidate with the shared classifier storage fix,
using original Ninja6; this is evidence for the isolated terminal optimization,
not the overall NinjaRS-versus-original acceptance gate.
The companion `ew2-terminal-v2-quiet-seven` uses all 100 unchanged EW2 points,
30 timed traversals and seven alternating pairs. Every numerical row and
status agrees exactly; all 3,000 timed evaluations per run are stable DP
(216). Its ratio is 0.998689, with 95% interval [0.983533, 1.006879], passing
the 1.05 regression limit. Median times are 0.000772966 and 0.000771952
seconds per point. These measurements support a production port with the
same guards and arithmetic; the final generated port still needs its own
correctness and integration checks.

## Reproduction

Run from the repository root with the pinned toolchain:

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/experiments/fortran-qp-followup/analyze.py
python docs/benchmarkfolder/experiments/fortran-qp-followup/source_inventory.py
python docs/benchmarkfolder/experiments/fortran-qp-followup/check_exact.py
python docs/benchmarkfolder/experiments/fortran-qp-followup/check_terminal.py
python docs/benchmarkfolder/experiments/fortran-qp-followup/check_terminal_bounds.py
python docs/benchmarkfolder/experiments/fortran-qp-followup/check_ieee_entry.py
python docs/benchmarkfolder/experiments/fortran-qp-followup/check_terminal_v2.py
python docs/benchmarkfolder/experiments/fortran-qp-followup/run.py \
  --case qcd-3 --variants terminal --run-id NEW-UNIQUE-RUN-ID \
  --repetitions 3 --loops 1 --cpu 2
```

Some fixture scripts deliberately refuse an existing work directory, preserving
earlier evidence. The runner retains copied source inputs, complete generated
source manifests, exact patches, executable hashes, point hashes and raw rows.
It compares every printed value and status byte-for-byte. Timing includes the
normal full amplitude evaluation; initialization plus a warm traversal is
reported separately. The existing runner assumes the core 100-point streams,
so it must not be used unchanged to time a shorter focused-control stream.
