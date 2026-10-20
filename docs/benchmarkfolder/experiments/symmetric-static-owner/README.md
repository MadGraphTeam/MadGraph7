# Static owners for repeated QP products

This separate experiment follows the immutable first-round evidence in
`../symmetric-accumulation/`. Its lazy cache had a quiet full-process median ratio
0.99845, with 95% interval [0.96677, 1.02698]; it was not promoted.

`static_owner.py` keeps the same bounded, finite, nearest-rounding domain and
original scatter arithmetic. A positive map entry identifies the first
lexicographic `(J,M,K)` member of a source-derived product group. That operation
writes the cache; later negative entries read it. Dispatch is outside the
`I,L` loops. There is no per-product readiness flag or clearing.

The write-before-read argument is per cache cell. Members share `K` and identical
complex `B` bits, including signed zeros, while `A` is unchanged. Their zero
predicates therefore agree for each `L,I`. Full updaters traverse the same
`I=1..LCUT_SIZE` range. Terminal groups also fix `J`, so their diagonal range
`I=J..min(J,LCUT_SIZE)` agrees. Empty ranges have no readers. The first member
performs the same original multiplication at the same point in the traversal;
every original addition remains in order. The private process transformation
must continue to prove separate input/output wavefunctions and an independent
vertex operand.

The optional folded-guard variant removes a second finite/bounded scan only
after the first integer classification has accepted exactly those unchanged
`A/B` slices. Finite checks used to fill zero masks are then redundant too.
Dimension, rounding, terminal entry-INEXACT and trap guards remain. The prototype
also checks support for every halting query before issuing it.

`prove_owners.py` enumerates actual QCD3/QCD4 group maps, ranks 2–4, every
coefficient, all legal cut sizes and both zero-predicate outcomes. Compiled
arithmetic, flags, trap and production-flag controls remain required. No
production source changes or performance claim are made by this proof.

## Completed checks

The source proof covers 102 map/rank/terminal combinations and 1,939,200
coefficient/traversal/predicate cases. `safety-representative-v1/` passes 72,000
exact output/input-bit and IEEE-flag comparisons plus 40 enabled-trap fallback
controls. It covers ranks two/four, full/terminal kernels and actual standalone
and NLO flags, including `-fno-automatic`. `ownership-stress-v1/` adds 4,000
successive calls with changing sparsity and all legal cut sizes: all 4,000 enter
the optimized path, none fall back, and all bits and flags agree.

`kernel-timing-v1/` compares original, lazy cache, static owner alone and static
owner with folded guards in seven rotated synthetic trials at ranks two–four.
The folded variant takes 0.828–0.848 of the original full-kernel time and
0.925–0.937 of the original terminal-kernel time. Relative to the lazy cache,
the incremental change is only about 1–3%. The whole guard/dispatch path is
included. These concurrent-host kernel measurements establish no full-process
benefit.

`count_products.py` instruments a separate copied process to count original
eligible products, unshared/owner multiplies, reuses and A/B zero-skipped slots
per helper. It also records whether each eligible product's vertex operand has
only a real component, only an imaginary component, or both. All counters are
outside production and their instrumented runtime is not performance evidence.
The complete unchanged 100-point stream and exact external output/status
comparison remain mandatory for the diagnostic trace.

The trace completed with all printed rows, accuracies and statuses exact
(98 DP and two QP), 7,296 accepted calls and zero fallback. It counts 5,503,000
eligible original complex products, of which 916,524 (16.65%) are reused.
Original B/A zero skips already remove 10,339,680 and 3,069,320 product slots.
Among eligible products, 32.29% have a purely real or purely imaginary vertex
operand. Counts apply only to the selected private helpers.

## Full evaluation profile and limitations

`full-evaluation-profile-v1/` samples five complete 100-point timed traversals
of the unchanged original polynomial, after initialization and untimed
numerical checks. The only driver change is a pre-timer stdin gate for attaching
`perf`. Numerical outputs and all 500 statuses agree with the retained reference.

The recorder exceeded its 30-second finalization wait after numerical completion,
leaving a zero data-length header. The raw stream contains 6,980 complete event
records (6,476 samples and the target exit), with no partial tail. The failure and
original bytes remain preserved. A separately recorded recovery changes the
header's data length and clears optional feature flags whose sections were never
finalized; every event byte is unchanged. Decoding disables inline source lookup
after a slow `addr2line` attempt. Lossless deterministic gzip files retain both
raw and derived streams, with compressed/decompressed hashes.

Cycle-weighted rank categories occupy 18.33% of sampled total evaluation work
for QP full rank2 / terminal rank2 / terminal rank3 combined. These include all
vertices using those routines, so they are an upper bound for the selected
sharing calls. QP expanded rank1 adds 6.35%. DP updater ranks2/3/1 occupy
17.62% / 9.45% / 4.99%; reduction accounts for 12.01%. These are diagnostic
cycle samples on a shared host, not exact stage times or a performance gate.
The small total share explains why a selected-product improvement has limited
whole-process potential. No additional seven-pair timing or production port is
justified for static owners alone.

This completed stage is immutable and bound by `SHA256SUMS.json`; its verifier
checks every retained file and both decompressed profile streams. The larger
changed-rounding derivative-row factorization proceeds separately in
`../symmetric-factorization/`.
