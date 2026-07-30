# pyAmpliCol Mac M3 performance investigations

This document records focused reproductions made from current `main`
(`202b3fd91a40a6e0422320053796fd71b5b2cd48`). The measurements committed in
the earlier Mac campaign are treated only as hypotheses and historical
baselines. Every current result must retain its source identity, resolved
configuration, selectors, deterministic phase-space point, numerical
validation, and raw evidence path.

Unless explicitly marked otherwise:

- `recurrence` means the built-in SM recurrence evaluator with JIT O2;
- `compiled` means the built-in SM compiled evaluator with JIT O3;
- wall time is the primary repeated native evaluation boundary;
- evaluator total is reported separately when the runtime exposes it;
- interaction/current counts are compared with original AmpliCol; and
- numerical-current evidence is used to audit missing or invalid recycling.

The user explicitly requested parallel execution without quiet-CPU gating.
Studies therefore do not wait for an idle host. Each attempt records its
measurement context, but observed concurrency is not used to postpone work.

## Checkpoint status

Work is paused at the requested clean-main checkpoint. The generic fixes and
evidence reviewed so far are integrated, but the final current-main timing
reruns for (a) and (b) remain pending, (c)--(e) have not started, and (f) is
complete only through `n=7`. No `n=8` or `n=9` study-F measurement was launched;
those cells remain explicitly N/A rather than being filled from stale or
unauthenticated evidence.

## (a) NLC/full `g g -> t t~ + 3g`

Paused at the requested checkpoint; the generic corrective work is integrated,
but the final integrated-source timing rerun remains pending.

Historical committed measurements (untrusted; pyAmpliCol source `9b7357f`):

| Colour | Implementation | Generation [s] | Wall [µs/pt] | Evaluator total [µs/pt] | pyAmpliCol/AmpliCol wall |
|---|---|---:|---:|---:|---:|
| NLC | Original AmpliCol | 37.465 | 1,442.08 | not exposed | 1.00 |
| NLC | pyAmpliCol recurrence JIT O2 | 41.551 | 169,487.12 | not exposed | 117.53 |
| Full | Original AmpliCol | 35.336 | 2,710.23 | not exposed | 1.00 |
| Full | pyAmpliCol recurrence JIT O2 | 42.082 | 177,229.50 | not exposed | 65.39 |

Fresh current-main measurements disprove the historical 100x/60x regression.
The `certified-reuse` rows are an explicit no-op audit: both representations
and both accuracies reported zero candidates/certificates, so structural counts
were unchanged. Timing differences between `off` and `certified-reuse` were
measured during concurrent host activity and are not attributed to an
optimization.

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
legacy, and no current-recycling fix is justified for this process.

The remaining compiled/recurrence gap was execution organization, not missed
current reuse. The compiled reducer exposed 208 leaf invocations and the old
generic `tile=2` policy issued 104 native calls per point. The generic adaptive
tiling fix uses authenticated phase-local scratch/output footprints and selects
`tile=16` for this shape. It contains no process, multiplicity, model, or colour
special case.

Provisional post-fix runtime measurements (generation-time remeasurement on the
integrated source is still pending):

| Colour | Implementation / mode | Batch | Generation [s] | Wall [µs/pt] | Evaluator total [µs/pt] | Wall / fresh AmpliCol | Wall / recurrence |
|---|---|---:|---:|---:|---:|---:|---:|
| NLC | Original AmpliCol | 18 | 23.914 | 1,521.36 | not exposed | 1.000 | 0.753 |
| NLC | recurrence JIT O2 | 18 | 33.831 | 2,019.98 | not exposed | 1.328 | 1.000 |
| NLC | compiled JIT O3, adaptive tile 16 | 18 | pending integrated rerun | 1,844 | 1,844 | 1.212 | 0.913 |
| NLC | compiled JIT O3, adaptive tile 16 | 128 | pending integrated rerun | 1,396 | 1,396 | 0.918 | 0.691 |
| Full | Original AmpliCol | 18 | 39.677 | 2,896.88 | not exposed | 1.000 | 1.361 |
| Full | recurrence JIT O2 | 18 | 34.883 | 2,127.92 | not exposed | 0.735 | 1.000 |
| Full | compiled JIT O3, adaptive tile 16 | 18 | pending integrated rerun | 2,763 | 2,763 | 0.954 | 1.298 |
| Full | compiled JIT O3, adaptive tile 16 | 128 | pending integrated rerun | 2,192 | 2,192 | 0.757 | 1.030 |

Thus “generic fragmentation” meant an unnecessarily fragmented native-call
schedule shared by any compiled evaluator with this footprint pattern.
Modified chunking did help: the adaptive policy removes most of the call
overhead and makes compiled O3 competitive with recurrence and legacy without
altering DAG structure or numerical results.

## (b) LC selected-flow `d d~ -> t t~ + 4g`

Closed on post-merge `main`
(`20130aeebee8f3cff2c1305ca65fc0fbda4110b7`). The earlier conclusion that
this process required a deeper representation redesign was wrong: its
structural excess came from using the terminal label of a public LC word as a
private recursion-closure sink.

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

## (c) LC `d d~ -> u u~ s s~ + (n-4)g`

Closed after fresh selected-flow, helicity-sum reproduction. The old 6.2x and
9.6x recurrence ratios were stale. Times below are microseconds per point;
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
| compiled JIT O3, phase-local tile fix | 144.721† | 669.993 | not exposed | 669.993 | 1.433 |

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

## (e) LC `d d~ -> u u~ s s~ c c~ + (n-6)g`

Closed on exact runtime `main`
(`20130aeebee8f3cff2c1305ca65fc0fbda4110b7`). Original AmpliCol is outside
its supported quark-line scope, so no legacy timing or structural ratio is
fabricated. Both multiplicities generate and run in recurrence JIT O2 and
compiled JIT O3.

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
| supported three-line comparator | recurrence JIT O2 | 30.646 | 3.613 | 2.141 | 140 / 262 / 32 / 14 |
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

## (f) Built-in SM dedicated `d d~ -> Z +` gluon ladder

Paused at the requested checkpoint after completing the authenticated table
through `n=7`. The live staging PDF is
`.agent-work/perf-z-table-f/.artifacts/performance-investigations/f/publication-staging/z_table.pdf`;
the final reviewed result will be copied to
`docs/performance_reports/macbook_M3/z_table/z_table.pdf`.

All declared cells through `n=7` now have an authenticated terminal state.
The erroneous initial `n=1` original-AmpliCol selected-flow generation time of
18.203 s was quarantined because it included one-time setup. A fresh prewarmed
attempt measures 2.18 s, consistent with the independently prewarmed `n=2`
value of 2.18 s.

ASM O3 and C++ O3 generation will not be attempted for `n>6`. Their 12 cells
for `n=7..9` remain declared but are resolved by catalog planning as static
policy N/A before source authentication, worker launch, compiler invocation,
or attempt-directory creation.

The refreshed `n=7` tier is:

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

The long `n=7` eager all-flow generation completed in 49.4 minutes with an
11.79 GiB outer peak, so it is retained under the requested one-hour/30 GiB
generation limits. Recurrence and compiled O3 are close for selected flow;
for all flows compiled O3 is 27.5% faster than AmpliCol and recurrence is
within 1.3%. The current PDF has been rendered at 180 dpi and independently
inspected on all four pages; tables, evaluator-total columns, ratios, native
static-N/A cells, and footnotes are legible without clipping or overlap.
The campaign was then stopped at the user's requested checkpoint before any
new study-F contract or `n=8`/`n=9` cell was launched. Those cells therefore
remain explicitly N/A rather than being populated from unauthenticated
historical attempts.

The first `n=8` all-flow eager-DAG JIT O2 attempt is not a timing result.
Immutable attempt `3e21ee19-fc35-4559-a60a-30860f9ba6a4` exited unexpectedly
after 2,874.38 s (47.91 min) while its authenticated phase state was still
`generation`; no evaluator or result was produced. The supervisor recorded
`phase_state_error` (`worker exited before closing its generation interval`)
and a 12.04 GiB peak process-tree RSS. System swap fell by roughly 9 GiB
immediately after the worker exited, but system-wide swap cannot authenticate
a per-process footprint. The cell is therefore classified as failed under
macOS resource pressure with `RSS-authenticated; physical-footprint
inconclusive`, rather than as either a valid measurement or a proven breach of
the 30 GiB cap. It will not be retried until the generic Darwin watchdog also
records and enforces process-tree physical footprint.

## Integrated corrective measures

At the requested wrap-up checkpoint, all independently reviewed fixes were
fast-forwarded to local `main` at `38fc47d`. This includes the generic adaptive
compiled tiling, compiled/eager/recurrence numerical-current certification and
replay, and the Darwin dual-metric/Z-table policy-wrapper work. A real
`just dev-install` completed successfully on that source. The bounded merged
gate passed 164 Python tests and all 20 Rust raw-evidence
replay/tamper/context/memory tests. Final integrated-source A/B timing reruns
remain pending and are not silently replaced by provisional measurements.

The generic numerical-current implementation now has the following contract:

| Generation mode | Default behavior | Explicit opt-out | Applied relation kinds | Replay behavior |
|---|---|---|---|---|
| compiled JIT | bounded certified reuse | `--no-numerical-current-reuse` or `mode = "off"` | equal, opposite, zero | persisted evidence and mapping; stale/tampered context fails closed |
| eager JIT | bounded certified reuse | `--no-numerical-current-reuse` or `mode = "off"` | equal, opposite, zero | persisted evidence and mapping; stale/tampered context fails closed |
| recurrence JIT | bounded certified reuse | `--no-numerical-current-reuse` or `mode = "off"` | equal, opposite, zero | native Rust independently recomputes classifications and mappings from authenticated raw samples |

Exact structural certificates remain preferred. A numerically certified
relation is nevertheless applied by default when it passes the deterministic
candidate and independent verification points at the configured absolute and
relative tolerances. Exactly one actionable warning is emitted per artifact
when an applied relation set lacks an exact structural proof. A deterministic
`no_certified_numerical_relation` outcome, such as study (b), applies no
mapping, emits no proof-less warning, and leaves default-on and opt-out
evaluators numerically identical.

For recurrence, the evidence identity binds every varied runtime parameter,
the canonical process and external PDGs, deterministic point seeds and
kinematics, selector/routing context, tolerances, candidate and verification
captures, full acceptance/rejection census, and the resulting schedule.
Removing or nulling the rejected-candidate diagnostics now fails closed.
Loading an artifact never silently rediscovers or changes the relation
mapping.

The final rebuilt native-extension replay passed 9/9 cases, covering a genuine
nonzero raw relation, charged-current process aliases, zero-certificate
default/off PACBIN identity, and builtin/UFO × LC/NLC/full. The focused Python
gate passed 173 tests with one intentional 30-GiB guarded skip. Separate Rust
gates passed 37 recurrence-manifest, 16 raw-evidence, 24 lowering, 5 direct
backend, and 3 relation tests. At the study-(a) scale, the complete candidate
index reduced approximately 66,093,200 exhaustive pair hypotheses to 19,262
screened hypotheses without changing the result; 399,392 exact-Fraction
property cases found no false negatives.

A subsequent real-shape integration reproduction exposed a gate that the
synthetic scale test missed: both recurrence NLC and full study-(a) runs
initially failed closed before allocation because the actual 17,074-current
component geometry exceeded the old one-GiB raw-evidence resident model. No
timing from those failed attempts is accepted. This blocked integration until
the evidence path was made generically bounded for the actual component
geometry without disabling discovery or weakening full-vector verification.

The exact geometry is 15,834 four-component currents plus 1,240 six-component
currents, or 70,776 components. Four candidate plus four verification points
produce 1,132,496 scalar slots and 34,158 rows. This is only 2.95% above the
old independent 1.1-million-scalar cutoff; it does not intrinsically exceed the
combined one-GiB model. The base resident estimate is 775,840,768 bytes, leaving
148,950,528 bytes for each of the two peak wire copies. The exact configured
96-character estimate is 115,356,478 bytes (and the conservative 112-character
estimate is 133,475,134 bytes), so the generic correction is to derive the wire
allowance from each shape and retain the same combined cap rather than reject
on the unrelated fixed scalar count. A 128-character estimate of 151,593,790
bytes remains correctly outside the bound.

The exact real-A recurrence rerun at provisional source `fc9fa19` confirmed
that the dynamic Python limit is necessary but not sufficient. Capture and
encoding succeeded with 146,798,789 raw-evidence bytes, 2,151,739 bytes below
the derived 148,950,528-byte wire ceiling, and found six relations. Native
authentication then rejected before applying them because the canonical JSON
DOM and independently parsed rational graph would exceed their combined 1-GiB
resident model. The 108.78-second failed run reached 1,170,915,328 bytes
maximum RSS and 1,045,942,144 bytes peak physical footprint. No constant or
tolerance is being relaxed: the remaining generic work is to consume the
authenticated raw samples without co-resident full JSON and BigRational
graphs, while preserving independent Rust reconstruction of the entire
classification and mapping.

That generic streaming correction now passes the exact NLC case at clean
source `dafcf99`. Rust consumes borrowed canonical observation rows, retains a
small authenticated offset/index structure, and independently reconstructs
the complete candidate/rejection census, certificates, and mappings. Python
drops the full Decimal capture graphs after canonical encoding and later
recaptures the bound baseline plan for strict application validation.

| Colour | Source/mode | Raw evidence | Persisted replay | Certified relations | Evaluations before → after | Application check | Elapsed [s] | Max RSS | Peak physical footprint |
|---|---|---:|---:|---|---:|---|---:|---:|---:|
| NLC | `dafcf99`, recurrence JIT O2, default-on | 146,798,789 B | 32,562 B | zero: 16, 17, 32, 33; opposite: 34→15; equal: 35→14 | 87,300 → 87,294 | 17,074 currents / 283,104 components; all residuals exactly zero; identical batch digest | 255.94 | 1,023,328,256 B | 699,960,640 B |
| Full | `dafcf99`, recurrence JIT O2, default-on | 146,798,789 B | 32,562 B | zero: 16, 17, 32, 33; opposite: 34→15; equal: 35→14 | 87,300 → 87,294 | 17,074 currents / 283,104 components; all residuals exactly zero; identical batch digest | 250.74 | 956,678,144 B | 710,544,896 B |

The lane is `authenticated-numerical-applied` and the native status is
`exact-certified-applied`; the raw 146.8 MB evidence is not persisted in the
artifact. The independent census tested 19,258 hypotheses from 66,076,140
theoretical pairs, screened 2,202 pair hypotheses plus 17,060 zero hypotheses,
certified six, rejected 19,252, and had zero verification rejections. The
full-colour run reproduces the same relation set and census. Strict post-run
invariant checks and `Runtime.load` pass for both artifacts. Their retained
paths are `.artifacts/real-a-nlc-v3.akohkg/artifact` and
`.artifacts/real-a-full-v3.cSxpPL/artifact` in the recurrence validation
worktree.

The corresponding real compiled NLC/full runs also produced no accepted
timing. After roughly four minutes of concurrent generation, both failed
closed in the shared helicity-sum lane when post-rewrite validation found that
current 202 changed by `4e-11`, far above the authenticated relation tolerance
(about `1.5e-76` for that value). The four implicated certificates—currents
15, 16, 21, and 23—are nevertheless genuine six-component zero relations:
candidate and independent-verification residuals are exactly zero. Applying
them suppresses 624 downstream rows and eliminates 264 complete evaluator
groups (74,260 to 73,996 evaluations). An exact recursive trace finds no
changed row or eliminated group in current 202's complete physical and
compiled dependency closure. Its producers and all upstream rows are
byte-for-byte unchanged. A fresh rebuild of the original 21,434-current DAG
reproduces current 202 exactly, whereas rebuilding the relation-applied joint
output bundle retains the `4e-11` difference even with Symbolica optimization
iterations set to zero. The cause is therefore output-bundle-dependent
finite-precision evaluation, not a bad zero certificate or an optimizer-iteration
setting.

A generic current-ID-keyed high-precision partition control removes that
coupling. On the exact real-A NLC case it rediscovers the same four zero
relations and 624 suppressions, then validates all 21,434 currents and 362,784
components with exactly zero absolute, relative, and tolerance residual; the
reference and applied observation-batch digests are identical. The two large
7,680-current stages take about 11.6 and 12.9 seconds to build per capture
session, and the full diagnostic reaches validation in roughly 1.5 minutes.
The partition ABI and digest are now authenticated against the runtime schema,
the focused equal/opposite/zero/default-on/opt-out/tamper tests pass, and the
implementation passed independent integration review. Production
compiled/eager lowering, public settings, and relation tolerances are
unchanged.
