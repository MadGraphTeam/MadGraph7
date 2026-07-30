# pyAmpliCol Mac M3 performance investigations

This logbook summarizes the accepted reproductions and generic fixes in the
final study-integration lineage. Historical campaign entries are retained only
where they provide a before-fix baseline; timing tables identify an older exact
measurement source when it differs from final `main`.

Unless explicitly marked otherwise:

- `recurrence` means the built-in SM recurrence evaluator with JIT O2;
- `compiled` means the built-in SM compiled evaluator with JIT O3;
- wall time is the outer repeated-evaluation boundary;
- recurrence, compiled, and eager expose an accumulated evaluator-total clock;
- recurrence alone additionally exposes its narrow inner execution
  attribution; and
- structural comparisons use currents, interaction/contribution evaluations,
  roots, and sources, with known source-representation rows identified
  separately from produced work.

The user explicitly requested parallel execution without quiet-CPU gating.
Studies therefore do not wait for an idle host. Each attempt records its
measurement context, but observed concurrency is not used to postpone work.

## Status

Studies (a)--(f) are closed. Their process-agnostic fixes are integrated, and
the dedicated Z table has a terminal result for every requested cell through
`n=9`. Default-on numerical current reuse is integrated for compiled, eager,
and recurrence generation.

## (a) NLC/full `g g -> t t~ + 3g`

Closed. The historical 100x/60x result was stale, and the remaining compiled
gap was a generic native-call tiling issue.

Historical committed measurements (untrusted; pyAmpliCol source `9b7357f`):

| Colour | Implementation | Generation [s] | Wall [µs/pt] | Evaluator total [µs/pt] | pyAmpliCol/AmpliCol wall |
|---|---|---:|---:|---:|---:|
| NLC | Original AmpliCol | 37.465 | 1,442.08 | not exposed | 1.00 |
| NLC | pyAmpliCol recurrence JIT O2 | 41.551 | 169,487.12 | not exposed | 117.53 |
| Full | Original AmpliCol | 35.336 | 2,710.23 | not exposed | 1.00 |
| Full | pyAmpliCol recurrence JIT O2 | 42.082 | 177,229.50 | not exposed | 65.39 |

The fresh reproduction below disproves the historical 100x/60x regression.
This early `certified-reuse` run was a no-op audit; later high-precision
validation on the same recurrence structure certified six relations, described
below. Timing differences between `off` and `certified-reuse` were measured
during concurrent host activity and are not attributed to an optimization.

| Colour | Implementation / mode | Relation policy | Generation [s] | Wall [µs/pt] | Execution attribution [µs/pt] | Evaluator total [µs/pt] | Wall / fresh AmpliCol |
|---|---|---|---:|---:|---:|---:|---:|
| NLC | Original AmpliCol | N/A | 23.914 | 1,521.36 | 1,521.36 | not exposed | 1.000 |
| NLC | recurrence JIT O2 | off | 33.831 | 2,019.98 | 1,626.23 | not exposed | 1.328 |
| NLC | recurrence JIT O2 | certified-reuse | 34.425 | 2,131.05 | 1,662.05 | not exposed | 1.401 |
| NLC | compiled JIT O3 | off | 157.668 | 5,452.47 | not separately exposed | 5,452.47 | 3.584 |
| NLC | compiled JIT O3 | certified-reuse | 168.722 | 6,032.00 | not separately exposed | 6,032.00 | 3.964 |
| Full | Original AmpliCol | N/A | 39.677 | 2,896.88 | 2,896.88 | not exposed | 1.000 |
| Full | recurrence JIT O2 | off | 34.883 | 2,127.92 | 1,524.86 | not exposed | 0.735 |
| Full | recurrence JIT O2 | certified-reuse | 35.051 | 2,064.57 | 1,486.77 | not exposed | 0.713 |
| Full | compiled JIT O3 | off | 157.461 | 7,287.97 | not separately exposed | 7,287.97 | 2.516 |
| Full | compiled JIT O3 | certified-reuse | 166.543 | 7,515.19 | not separately exposed | 7,515.19 | 2.595 |

Recurrence uses 17,074 currents, 87,300 contribution evaluations, and 15,360
amplitude destinations, versus legacy's 37,680 active currents, 159,840
kernel evaluations, and identical 15,360 destinations. Compiled has 21,434
currents, 117,580 raw interactions, 46,180 evaluation groups, and the same
destinations. Thus both pyAmpliCol schedules are structurally better than
legacy.

The remaining compiled/recurrence gap was execution organization, not missed
current reuse. The compiled reducer exposed 208 leaf invocations and the old
generic `tile=2` policy issued 104 native calls per point. The generic adaptive
tiling fix uses authenticated phase-local scratch/output footprints and selects
`tile=16` for this shape. It contains no process, multiplicity, model, or colour
special case.

Post-fix runtime measurements reused the generated artifacts; generation was
not remeasured for this runtime-only tiling study:

| Colour | Implementation / mode | Batch | Generation [s] | Wall [µs/pt] | Evaluator total [µs/pt] | Wall / fresh AmpliCol | Wall / recurrence |
|---|---|---:|---:|---:|---:|---:|---:|
| NLC | Original AmpliCol | 18 | 23.914 | 1,521.36 | not exposed | 1.000 | 0.753 |
| NLC | recurrence JIT O2 | 18 | 33.831 | 2,019.98 | not exposed | 1.328 | 1.000 |
| NLC | compiled JIT O3, adaptive tile 16 | 18 | not remeasured | 1,844 | 1,844 | 1.212 | 0.913 |
| NLC | compiled JIT O3, adaptive tile 16 | 128 | not remeasured | 1,396 | 1,396 | 0.918 | 0.691 |
| Full | Original AmpliCol | 18 | 39.677 | 2,896.88 | not exposed | 1.000 | 1.361 |
| Full | recurrence JIT O2 | 18 | 34.883 | 2,127.92 | not exposed | 0.735 | 1.000 |
| Full | compiled JIT O3, adaptive tile 16 | 18 | not remeasured | 2,763 | 2,763 | 0.954 | 1.298 |
| Full | compiled JIT O3, adaptive tile 16 | 128 | not remeasured | 2,192 | 2,192 | 0.757 | 1.030 |

Thus “generic fragmentation” meant an unnecessarily fragmented native-call
schedule shared by any compiled evaluator with this footprint pattern.
Modified chunking did help: the adaptive policy removes most of the call
overhead and makes compiled O3 competitive with recurrence and legacy without
altering DAG structure or numerical results.

Later 96-digit recurrence validation certified four zero, one equal, and one
opposite relation, reducing 87,300 evaluations to 87,294 with identical
outputs. This valid but tiny saving was not the cause of the historical
slowdown.

**Conclusion:** the historical 100x/60x regression is closed. Recurrence is
structurally better than legacy and runs at 1.33x legacy for NLC and 0.74x for
full colour. At batch 128, generic adaptive tiling brings compiled O3 to 0.92x
and 0.76x legacy respectively.

## (b) LC selected-flow `d d~ -> t t~ + 4g`

Closed at measurement source
`20130aeebee8f3cff2c1305ca65fc0fbda4110b7`, an ancestor of current `main`.
The earlier conclusion that this process required a deeper representation
redesign was wrong: its structural excess came from using the terminal label
of a public LC word as a private recursion-closure sink.

Before the fix, the fresh reproduction was:

| Implementation / mode | Generation [s] | Wall [µs/pt] | Evaluator total [µs/pt] | Structure: currents / interactions / roots / sources | Wall / AmpliCol | Wall / recurrence |
|---|---:|---:|---:|---:|---:|---:|
| Original AmpliCol | 28.452 | 9.418653 | not exposed | 377 / 1,590 / 128 / 15 (superseded active-probe census) | 1.0000 | 0.3895 |
| recurrence JIT O2 | 6.991655 | 24.180016 | 22.829824 | 652 / 3,532 / 128 / 16 | 2.5672 | 1.0000 |
| compiled JIT O3 | 52.066506 | 26.527843 | 26.384622 | 652 / 3,532 / 128 / 16 | 2.8165 | 1.0971 |

The bounded post-merge rerun used the same selected public flow
`(2,5,6,7,8,4,3,1)`, helicity sum, batch 128, and a one-second warmed timing
target. No quiet-CPU condition was imposed.

| Implementation / mode | Generation [s] | Wall [µs/pt] | Evaluator total [µs/pt] | Structure: currents / interactions / roots / sources | Wall / AmpliCol | Wall / recurrence |
|---|---:|---:|---:|---:|---:|---:|
| Original AmpliCol | 11.660 | 9.221 | not exposed | 378 / 1,590 / 128 / 16 | 1.000 | 0.659 |
| recurrence JIT O2 | 29.895 | 13.985 | 12.966 | 378 / 1,590 / 128 / 16 | 1.517 | 1.000 |
| compiled JIT O3 | 30.376 | 11.596 | not separately exposed | 378 / 1,590 / 128 / 16 | 1.258 | 0.829 |

The structure column is the physical selected-flow closure census before
backend-specific, structurally proven helicity materialization. Compiled O3
subsequently materializes an even smaller proven schedule. The corrected raw
legacy-module audit has 16 source declarations and 16 generated `ext_*`
slots, exactly matching pyAmpliCol; the earlier apparent fifteenth legacy
source came from the active-probe accounting boundary. The supposed
“additional pyAmpliCol source” therefore never existed and was not the cause
of the mismatch.

The generic fix keeps the public flow word unchanged, reconstructs complete
open-string blocks, cyclically rotates those blocks from the block containing
the lowest canonical external source slot, and closes on the final block
endpoint. For this case the private closure traversal is
`(3,1,2,5,6,7,8,4)`. The rule is validated, invariant under public-label
relabeling and crossing bijections, and is shared by the generic DAG,
recurrence projection, and native recurrence lowering; it contains no process
or multiplicity special case.

At the fresh oracle point, recurrence differs from original AmpliCol by
`8.82e-15` relative, compiled differs from AmpliCol by `1.71e-15`, and compiled
differs from recurrence by `1.05e-14`. The default 96-digit numerical-current
pass again returns `no_certified_numerical_relation` and applies no numerical
mapping. That negative result is now expected: structural parity comes from
the canonical closure rule, not from weakening numerical-current
certification.

**Conclusion:** exact structural parity is restored: legacy and pyAmpliCol
both use 378 currents, 1,590 interactions, 128 roots, and 16 sources. Compiled
O3 is 1.26x legacy and recurrence is 1.52x. The numerical detector correctly
finds no relation because the former excess was a closure-representation bug,
not a numerical current identity.

## (c) LC `d d~ -> u u~ s s~ + (n-4)g`

Closed after fresh selected-flow, helicity-sum reproduction. The reported
slowdowns were stale. Times below are microseconds per point;
“execution” is the recurrence schedule attribution, while “evaluator total” is
the warmed accumulated evaluator wall envelope.

| n | Implementation / mode | Generation [s] | Wall [µs/pt] | Execution [µs/pt] | Evaluator total [µs/pt] | Wall / AmpliCol |
|---:|---|---:|---:|---:|---:|---:|
| 4 | Original AmpliCol, selected flow | 4.749 | 0.227298 | 0.227298 | not separately exposed | 1.000 |
| 4 | recurrence JIT O2, complete artifact + runtime selector | 5.324 | 0.683683 | 0.405073 | 0.683683 | 3.008 |
| 4 | compiled JIT O3, complete artifact + runtime selector | 3.950 | 0.500570 | not separately exposed | 0.500570 | 2.202 |
| 6 | Original AmpliCol, selected flow | 14.438 | 1.106783 | 1.106783 | not separately exposed | 1.000 |
| 6 | recurrence JIT O2 before fix, complete artifact + runtime selector | 30.646 | 3.613397 | 2.141222 | 3.613397 | 3.265 |
| 6 | compiled JIT O3, complete artifact + runtime selector | 59.094 | 1.517343 | not separately exposed | 1.517343 | 1.371 |
| 6 | recurrence JIT O2 after fix, generation-specialized sector | 4.954 | 2.747183 | 2.362211 | 2.747183 | 2.482 |

The structural comparison is:

| n / mode | Sources | Currents | Interactions / contributions | Destinations | Difference from AmpliCol |
|---|---:|---:|---:|---:|---|
| 4 AmpliCol | 9 | 32 | 37 | 4 | reference |
| 4 recurrence selected target | 10 | 33 | 37 | 4 | one reachable source-representation current; produced work and destinations match |
| 6 AmpliCol | 13 | 136 | 283 | 16 | reference |
| 6 recurrence before fix, representative shared by two public flows | 14 | 140 | 262 | 16 requested (32 closure rows across both flows) | +4 currents, but 21 fewer contributions |
| 6 recurrence after fix, generation-specialized sector | 14 | 116 | 246 | 16 | 20 fewer currents and 37 fewer contributions |

The generic cause was that recurrence generation with an explicit selected
colour sector still projected the complete 72-sector colour plan and only
masked the requested public flow afterward. This retained an unrequested
representative alias/closure domain and made a genuinely specialized artifact
fail with `replay covers 1 of 72 physical sectors`. The fix projects the
already-restricted colour plan and remaps its artifact-local sector IDs densely
before lowering. It contains no process, multiplicity, particle, or sector-ID
special case.

The fixed n=6 artifact agrees exactly with the corresponding flow of the
complete artifact across all 256 helicities at precision 32. Its default
numerical-current pass certifies and applies no additional relation, so the
improvement is entirely structural rather than a tolerance-based reuse.

**Conclusion:** dense generation-time projection removes unrelated colour
sectors without a process-specific rule. At `n=4`, physical work and
destinations match legacy apart from one source-representation row; at `n=6`,
recurrence performs 20 fewer currents and 37 fewer contributions than legacy.
No additional numerical relation is certified.

## (d) LC `d d~ -> Z Z Z + 6g`

Fresh post-merge measurements at source `20130ae` do not reproduce the
reported `2.3x` recurrence slowdown. The default recurrence path is 1.158x
slower than original AmpliCol. It found no numerical-current relation, so the
default-on and explicit opt-out artifacts have identical matrices and
structures.

| Implementation / mode | Generation [s] | Wall [µs/pt] | Execution [µs/pt] | Evaluator total [µs/pt] | Wall / AmpliCol |
|---|---:|---:|---:|---:|---:|
| Original AmpliCol | 299.858 | 467.467 | 467.467 | not exposed | 1.000 |
| recurrence JIT O2, default certified reuse | 41.514* | 541.474 | 534.614 | 541.474 | 1.158 |
| compiled JIT O3, before fix | 144.721 | 6,203.550 | not exposed | 6,203.550 | 13.271 |
| compiled JIT O3, phase-local tile fix | not remeasured† | 669.993 | not exposed | 669.993 | 1.433 |

`*` The default run's generation timer was lost after generation when an
unrelated selector-derivation point failed its threshold. The 41.514 s value is
the comparable opt-out generation, whose artifact has the same matrix and
structural census. `†` The post-fix runtime reused the already-generated O3
artifact; generation was deliberately not repeated.

All paths perform the same physical work:

| Implementation | Sources | Produced currents | Total currents | Interaction evaluations | Roots |
|---|---:|---:|---:|---:|---:|
| Original AmpliCol | 24 | 16,700 | 16,724 | 128,158 | 3,456 |
| recurrence / compiled generic DAG | 25 | 16,700 | 16,725 | 128,158 | 3,456 |

The one-current difference is the known source-row representation convention,
not missed or duplicated interaction work. Default recurrence inspected all
16,725 currents and tested 16,896 equal/opposite/zero hypotheses, but found,
certified, and applied zero relations. The opt-out result is consequently
identical.

The fresh matrix elements are `2.950518076490481e-27` for AmpliCol,
`2.950518076465731e-27` for recurrence, and
`2.9505180764657654e-27` for compiled O3. The relative differences are
8.39e-12 between recurrence and AmpliCol, 8.38e-12 between compiled and
AmpliCol, and 1.17e-14 between compiled and recurrence. The compiled result is
bit-for-bit unchanged by the fix, and its resolved-sum validation has a maximum
relative difference of 3.65e-16.

The genuine anomaly was instead compiled Direct-Arena tiling. One cache
footprint was shared by total evaluation and resolved/routed reduction. The
cold resolved tensor has 6,912 helicities times 720 colours, and reducer
overhead raises its authenticated footprint to 4,980,101 scalars per point.
Applying that footprint to selected-total evaluation forced a one-point tile,
even though the selected-total path never materializes the resolved tensor.
Smaller evaluator output chunks could not help because this reduction footprint
dominated the tile calculation.

The generic fix gives total execution and resolved/routed reduction separate
cache-local tile capacities while retaining the same hard workspace bound.
Selected-total evaluation now uses its leaf/source footprint; resolved and
routed paths continue to use the full authenticated reduction footprint. On
this artifact the total tile rises from 1 to 32 and Direct-Arena calls fall
from 79.0 to 2.46875 per point. Compiled O3 improves by 9.259x to 669.993
µs/point, 1.237x slower than recurrence and 1.433x slower than AmpliCol, with
no process-specific rule or numerical change.

**Conclusion:** the reported 2.3x recurrence slowdown is not reproduced:
recurrence is 1.158x legacy. Separating selected-total and resolved-reduction
tile footprints improves compiled O3 from 6,203.6 to 670.0 µs/point, or 1.433x
legacy, with unchanged structure and numerical output. Numerical discovery
correctly finds no missing reuse.

## (e) LC `d d~ -> u u~ s s~ c c~ + (n-6)g`

Closed at measurement source
`20130aeebee8f3cff2c1305ca65fc0fbda4110b7`, an ancestor of current `main`.
Original AmpliCol is outside its supported quark-line scope, so no legacy
timing or structural ratio is fabricated. Both multiplicities generate and run
in recurrence JIT O2 and compiled JIT O3.

The selected-flow measurements used a one-second warmed target with no
quiet-CPU condition. The compiled timing rows were refreshed once from the
already generated artifacts after separating the outer wall clock from the
native accumulated evaluator clock; generation was not repeated.

| n | Mode | Generation [s] | Wall [µs/pt] | Evaluator total [µs/pt] | Currents / interactions / roots / sources | Wall / recurrence |
|---:|---|---:|---:|---:|---:|---:|
| 6 | recurrence JIT O2 | 10.989 | 2.453 | 1.266 | 75 / 129 / 8 / 14 | 1.000 |
| 6 | compiled JIT O3 | 37.451 | 1.325 | 1.324 | 75 / 129 / 8 / 14 | 0.540 |
| 7 | recurrence JIT O2 | 70.228 | 6.032 | 3.600 | 165 / 445 / 16 / 16 | 1.000 |
| 7 | compiled JIT O3 | 262.466 | 6.192 | 6.185 | 165 / 445 / 16 / 16 | 1.027 |

The recurrence evaluator totals are its direct execution attributions.
Compiled narrow phase attribution remains correctly unavailable; its distinct
outer wall and native evaluator-total values are the two supported clocks.
The uncontrolled `n=7` compiled sample has 20.3% relative standard error, but
its central wall value is still within 2.7% of recurrence.

Recurrence and compiled use exactly the same selected physical flows and have
exact structural parity at both multiplicities. Their matrix elements are
`6.85754823014535e-17` versus `6.85754823014541e-17` at `n=6`, and
`3.442610849692029e-19` versus `3.442610849692051e-19` at `n=7`: relative
differences of `8.81e-15` and `6.43e-15`.

Default numerical-current discovery ran during generation in every relevant
lane: recurrence primary plus compiled primary and helicity-sum. Every lane
returned `no_certified_numerical_relation`, with zero certified and zero
applied relations. Thus the detector found no missed equal, opposite, or zero
current reuse for these selected sectors; the unoptimized baseline mapping was
retained.

The closest fresh supported three-quark-line comparator is process 13 at
`n=6`, `d d~ -> u u~ s s~ g g`, which also has eight external legs:

| Process | Mode | Generation [s] | Wall [µs/pt] | Evaluator total [µs/pt] | Currents / interactions / roots / sources |
|---|---|---:|---:|---:|---:|
| supported three-line comparator | recurrence JIT O2 | 4.954 | 2.747 | 2.362 | 116 / 246 / 16 / 14 |
| study E four-line process | recurrence JIT O2 | 10.989 | 2.453 | 1.266 | 75 / 129 / 8 / 14 |
| supported three-line comparator | compiled JIT O3 | 59.094 | not separately measured | 1.517 | 140 / 262 / 32 / 14 |
| study E four-line process | compiled JIT O3 | 37.451 | 1.325 | 1.324 | 75 / 129 / 8 / 14 |

Study E is therefore structurally smaller and faster than this closest
same-leg-count supported comparator: recurrence is faster at both boundaries,
and compiled has the smaller evaluator total, rather than merely succeeding at
an unreasonable cost.

One generic report bug initially hid this success: selector derivation called
all physical values below the unrelated absolute `1e-15` validation tolerance
zero. It now selects the largest finite strictly nonzero resolved component,
independent of process scale, while still failing closed when all components
are exactly zero or any value is non-finite. A second generic report fix gives
compiled/eager measurements separate outer wall and native evaluator-total
clocks. Neither fix contains a process, multiplicity, model, or backend special
case.

**Conclusion:** both four-quark-line cases generate and run normally.
Recurrence and compiled have exact structural parity—75/129/8/14 at `n=6`
and 165/445/16/16 at `n=7`—and agree numerically to better than `7e-15`
relative. Performance is comparable to or better than the nearest supported
same-leg-count process, and no additional numerical reuse is found.

## (f) Built-in SM dedicated `d d~ -> Z +` gluon ladder

The [final table](docs/performance_reports/macbook_M3/z_table/z_table.pdf) is
complete through `n=9`. The erroneous one-time AmpliCol start-up cost is no
longer charged to `n=1`: selected-flow AmpliCol generation is 2.18 s at both
`n=1` and `n=2`, rather than the stale 18.203 s value. Every successful
pyAmpliCol runtime exposes wall and evaluator-total timing separately; legacy
AmpliCol does not expose evaluator-total timing.

Native ASM/C++ rows for `n>6` are declarative static N/A with reason
`native-backend-generation-cap-n6-v1`. All 12 cells remain visible in the
table, but planning creates no attempt or compiler directory for them.

The `n=7` tier is:

| Flow setup | Implementation | Generation [s] | Wall [µs/pt] | Evaluator total [µs/pt] | Wall / AmpliCol |
|---|---|---:|---:|---:|---:|
| selected flow, helicity sum | Original AmpliCol | 5.080 | 37.774 | not exposed | 1.000 |
| selected flow, helicity sum | compiled JIT O1 | 17.5 | 49.4 | 49.4 | 1.31 |
| selected flow, helicity sum | compiled JIT O3 | 16.635 | 46.043 | 46.043 | 1.22 |
| selected flow, helicity sum | eager-DAG JIT O2 | 3.35 | 83.4 | 83.4 | 2.21 |
| selected flow, helicity sum | recurrence JIT O2 | 7.041 | 37.900 | 37.081 | 1.00 |
| all flows, single helicity | Original AmpliCol | 1.232 | 315.177 | not exposed | 1.000 |
| all flows, single helicity | compiled JIT O1 | 646 | 258 | 258 | 0.819 |
| all flows, single helicity | compiled JIT O3 | 656.946 | 231.565 | 231.565 | 0.735 |
| all flows, single helicity | eager-DAG JIT O2 | 2,964.444 | 905.216 | 905.216 | 2.87 |
| all flows, single helicity | recurrence JIT O2 | 23.810 | 319.417 | 315.055 | 1.01 |

The long `n=7` eager all-flow generation completed in 49.4 minutes below both
caps. Recurrence and compiled O3 are close for selected flow; for all flows
compiled O3 is 27.5% faster than AmpliCol and recurrence is within 1.3%.

### `n=8`

| Workload | Setup | Generation status/time | Wall [µs/pt] | Evaluator total [µs/pt] | Ratio vs AmpliCol (gen / wall) | Peak guard/RSS [GB] | Outcome |
|---|---|---:|---:|---:|---:|---:|---|
| selected flow, hel-sum | AmpliCol | 10.352 s | 100.863 | not exposed | 1.00× / 1.00× | — | OK |
| selected flow, hel-sum | compiled JIT O1 | 126.377 s | 218.105 | 217.812 | 12.2× / 2.16× | 1.961 / 1.961 | OK |
| selected flow, hel-sum | compiled JIT O3 | 76.350 s | 180.044 | 179.923 | 7.38× / 1.79× | 1.983 / 1.983 | OK |
| selected flow, hel-sum | eager-DAG JIT O2 | 49.239 s | 718.401 | 716.418 | 4.76× / 7.12× | 1.456 / 1.456 | OK |
| selected flow, hel-sum | recurrence JIT O2 | 66.736 s | 93.872 | 93.872 | 6.45× / 0.931× | 0.635 / 0.635 | OK |
| all flows, single hel | AmpliCol | 19.114 s | 8,209.519 | not exposed | n/c / 1.00× | 3.709 / — | OK |
| all flows, single hel | compiled JIT O1 | 30 GB cap at 2,638.855 s | — | — | n/c / — | 30.023 / 6.563 | guarded N/A |
| all flows, single hel | compiled JIT O3 | 30 GB cap at 2,776.558 s | — | — | n/c / — | 30.026 / 6.884 | guarded N/A |
| all flows, single hel | eager-DAG JIT O2 | 30 GB cap at 2,753.840 s | — | — | n/c / — | 30.019 / 6.952 | guarded N/A |
| all flows, single hel | recurrence JIT O2 | 1,263.412 s | 6,156.947 | 6,156.947 | n/c / 0.750× | 1.347 / 1.347 | OK; resolved/pointwise/cross-layout numerical parity |

### `n=9`

| Workload | Setup | Generation status/time | Wall [µs/pt] | Evaluator total [µs/pt] | Ratio vs AmpliCol (gen / wall) | Peak guard/RSS [GB] | Outcome |
|---|---|---:|---:|---:|---:|---:|---|
| selected flow, hel-sum | AmpliCol | 63.432 s | 430.430 | not exposed | 1.00× / 1.00× | — | OK |
| selected flow, hel-sum | compiled JIT O1 | 516.756 s | 458.012 | 455.750 | 8.15× / 1.06× | 21.499 / 14.170 | OK |
| selected flow, hel-sum | compiled JIT O3 | 511.035 s | 453.164 | 451.693 | 8.06× / 1.05× | 21.655 / 13.207 | OK |
| selected flow, hel-sum | eager-DAG JIT O2 | 112.792 s | 530.922 | 529.102 | 1.78× / 1.23× | 11.573 / 9.005 | OK |
| selected flow, hel-sum | recurrence JIT O2 | 814.364 s | 227.275 | 227.275 | 12.8× / 0.528× | 7.845 / 5.178 | OK |
| all flows, single hel | AmpliCol | 616.264 s | 105,293.508 | not exposed | n/c / 1.00× | 26.111 / 9.579 | OK |
| all flows, single hel | compiled JIT O1 | 30 GB cap at 2,099.235 s | — | — | n/c / — | 30.047 / 9.411 | guarded N/A |
| all flows, single hel | compiled JIT O3 | 30 GB cap at 2,049.774 s | — | — | n/c / — | 30.008 / 8.988 | guarded N/A |
| all flows, single hel | eager-DAG JIT O2 | 30 GB cap at 2,119.015 s | — | — | n/c / — | 30.024 / 8.886 | guarded N/A |
| all flows, single hel | recurrence JIT O2 | 1 h cap at 3,600.437 s | — | — | n/c / — | 2.320 / 1.229 | guarded N/A |

The strongest runtime result is recurrence: 7% faster than AmpliCol for `n=8`
selected-flow, 47% faster for `n=9` selected-flow, and 25% faster for the
completed `n=8` all-flow case. The remaining issue is generation scalability:
`n=8` all-flow compiled/eager modes hit 30 GB, while `n=9` all-flow
compiled/eager modes hit 30 GB after about 34--35 minutes and recurrence hits
the one-hour cap at only 2.32 GB. Each is an independently observed terminal
result rather than a status propagated from another mode.

**Conclusion:** the table is complete through `n=9` under the requested
30 GB/one-hour limits. Successful rows report both pyAmpliCol timing clocks,
all numerical checks pass, and every uncompleted row records the independently
reached resource cap.

## Integrated corrective measures

All corrective code described in studies (a)--(f) is integrated in the final
study-integration lineage. None of the fixes selects a process, multiplicity,
model, or colour accuracy by name.

| Area | Generic correction | Result |
|---|---|---|
| LC closure | Derive the private closure traversal from complete open-string blocks | Restores exact study-(b) parity in generic-DAG and recurrence generation |
| Selected colour sectors | Project and densely renumber the already-restricted colour plan before lowering | Removes unrelated sectors in study (c); `n=6` recurrence becomes structurally better than legacy |
| Compiled execution | Choose tiles from phase-local total/reduction footprints | Removes study-(a) call fragmentation and cuts study-(d) compiled runtime from 6,203.6 to 670.0 µs/point |
| Reporting | Select the largest finite nonzero probe component and expose separate outer-wall/evaluator-total clocks | Makes tiny physical amplitudes measurable and restores both clocks for recurrence, compiled, and eager |
| Numerical reuse | Run bounded high-precision equal/opposite/zero discovery during generation | Applies every certified relation by default in compiled, eager, and recurrence; a public opt-out restores the unoptimized schedule |

Numerical current reuse is an optimization safety net, not a substitute for
structural canonicalization. Candidate relations must pass independent
high-precision verification before application. An applied relation without an
exact structural proof emits one warning per artifact. If discovery returns
`no_certified_numerical_relation`, no mapping and no warning are produced.
`--no-numerical-current-reuse` (or configuration mode `off`) is the public
opt-out.

The study-(b), (c), (d), and (e) selected sectors correctly return no
certified relation after their structural fixes. In the large study-(a)
recurrence case, the detector certified four zero, one equal, and one opposite
relation and reduced 87,300 evaluations to 87,294 with identical outputs; this
small saving was not the cause of the original timing anomaly.

Large recurrence evidence no longer requires candidate and verification
Decimal graphs to coexist in memory. Current `main` captures them sequentially
into temporary row stores, preserves complete global discovery and full-vector
verification, and transports canonical evidence in a bounded compressed
envelope whose length and digest are checked before use. Temporary stores close
on both success and lowering/validation errors.

The study-(a) NLC/full validation used 146.8 MB of raw evidence and certified
the same six relations in both colour modes. Compiled and eager additionally
use a generic current-ID-keyed high-precision partition, preventing unrelated
output-bundle rounding from contaminating relation validation while leaving
production lowering and public tolerances unchanged.

The final `n=8` all-flow recurrence artifact has 38,581 currents and 286,294
contributions. It generated in 1,263.412 s at 1.347 GB peak guarded memory,
inspected every current, applied zero relations, and passed resolved,
pointwise, and cross-layout numerical parity.

## Final cross-study assessment

No scoped process retains a structural-parity defect. The only unexpected
extra source in study (b) was removed by a generic private-closure rule, the
study-(c) selected sectors are now equal to or smaller than legacy, and studies
(a), (d), and (e) contain no missed structural optimization. The default
numerical pass remains a generic safety net across all generation modes and
uses every independently certified equal, opposite, or zero relation unless
the user explicitly opts out.

No additional structural investigation is required to close studies (a)--(f).
The remaining future optimization opportunity is generation scalability for
the largest all-flow Z cells; it does not indicate a current/interactions
parity failure and is therefore outside the corrective scope of these studies.
