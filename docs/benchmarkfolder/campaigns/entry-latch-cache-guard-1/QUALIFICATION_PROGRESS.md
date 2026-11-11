# Current-source shared-host qualification

The generated Fortran at source revision `0fcd2cd873d61d134837e1d9e87afb13a47824fd`
has completed all eleven ordinary process comparisons. Each completed
process uses the same 100 saved physical points, starting-revision CutTools,
and candidate backends 1, 6, 7, 8 and 9. All 66 completed lanes pass the
`3e-6` regression criterion; 54 also pass the separate `1e-8` ordinary-point
target. In QCD `u u~ > z g g`, candidate backends 6/7/8/9 each retain 56
scalar target misses, covering 28 saved points with duplicated finite entries
across split-order rows.
The largest symmetric finite residual is approximately `2.80642065e-6`.
This comparison does not assign independent numerical truth to CutTools.

In QCD `u u~ > z z g g`, candidate backends 6/7/8/9 retain respectively
88/80/80/78 finite scalar target misses, covering 44/40/40/39 distinct saved
points. The largest symmetric finite residual is approximately
`4.15091160e-7`. All six lanes still pass the agreed `3e-6` criterion.

In EW `e+ e- > mu+ mu- a a`, candidate backends 6/7/8/9 retain respectively
12/10/14/8 scalar target misses. Their finite/single-pole/double-pole counts
are 10/2/0, 8/0/2, 10/2/2 and 8/0/0. TIRust's finite misses occur at saved
points 14, 81, 85 and 87, with duplicated split-order entries; its maximum
symmetric finite residual is `1.143649376375392e-7`. These comparisons preserve
the CutTools reference without assigning it independent numerical truth.

Accepted final statuses identify the selected backend: TIRust uses DP at 87
QCD three-particle points and QP at 13. QCD four-particle counts are:

| Lane | Accepted DP | Accepted QP |
|---|---:|---:|
| Baseline CutTools and candidate CutTools, each | 88 | 12 |
| Candidate Ninja | 97 | 3 |
| Candidate COLLIER | 100 | 0 |
| Candidate NinjaRS | 96 | 4 |
| Candidate TIRust | 14 | 86 |

EW four-particle counts are:

| Lane | Accepted DP | Accepted QP |
|---|---:|---:|
| Baseline CutTools and candidate CutTools, each | 93 | 7 |
| Candidate Ninja, COLLIER and NinjaRS, each | 100 | 0 |
| Candidate TIRust | 40 | 60 |

All other completed lanes report DP.
These statuses do not instrument internal retries or floating-point exception
flags.

| Process | Closed evidence | Independent audit |
|---|---|---|
| QCD `u u~ > z` | [n1 closure](n1-qualification-1/closure-index.json) | Passed |
| EW `u d~ > w+` | [n1 closure](n1-qualification-1/closure-index.json) | Passed |
| `g g > h` | [checkpoint](remaining-qualification-1/li_gg_h/checkpoint.json) | Passed |
| QCD `g g > t t~` | [checkpoint](remaining-qualification-1/qcd_n2_gg_ttx/checkpoint.json) | Passed |
| EW `e+ e- > mu+ mu-` | [checkpoint](remaining-qualification-1/ew_n2_epem_mupmum/checkpoint.json) | Passed |
| Mixed `u u~ > g a` | [checkpoint](remaining-qualification-1/mixed_uux_ga/checkpoint.json) | Passed |
| `g g > h h` | [checkpoint](remaining-qualification-1/li_gg_hh/checkpoint.json) | Passed |
| QCD `u u~ > z g g` | [checkpoint](remaining-qualification-1/qcd_n3_uux_zgg/checkpoint.json) | Passed; target misses retained |
| EW `e+ e- > t t~ a` | [checkpoint](remaining-qualification-1/ew_n3_epem_ttxa/checkpoint.json) | Passed |
| QCD `u u~ > z z g g` | [checkpoint](remaining-qualification-1/qcd_n4_uux_zzgg/checkpoint.json) | Passed; target misses retained |
| EW `e+ e- > mu+ mu- a a` | [combined checkpoint](ew-n4-continuation-1/checkpoint.json) | Passed; target misses and interruption retained |

Each continuation checkpoint binds its raw protocols, saved inputs, build and
source receipts, resolved runtime-library hashes, immutable stage receipts and
numerical summaries. Candidate builds are fresh; baseline numerical objects
remain unchanged. Runtime evidence uses resolved `ldd` paths and file hashes,
not sampled worker memory maps. The [continuation source index](remaining-qualification-1/source-index.json)
freezes its supervisor and dependencies. The growing supervisor status is not
part of a closed scientific checkpoint.

For EW three-particle production, all 375 inherited baseline objects, 426
generated source pins and compiler flags match the unchanged starting-revision
donor. Its original full-build receipt has not been located; the retained proof
establishes unchanged inherited objects and the current worker relink. The
candidate has a fresh full build.
The same original-receipt limitation applies to QCD four-particle production,
whose proof preserves 155 inherited baseline objects and 206 generated source
pins; its candidate also has a fresh full build.
For EW four-particle production, the proof preserves 320 inherited baseline
objects and 371 generated source pins with the same original-receipt limitation;
its candidate has a fresh full build with 372 generated source pins.

The EW four-particle checkpoint combines five unchanged, completed original
lanes with a separate fresh 100-point TIRust lane. The user requested a clean
stop at point 74 of the original TIRust attempt before updating Codex. That
partial stream and administrative `InterruptedError` remain preserved and
unselected. The fresh stream starts from the original bootstrap and warmup;
the streams were not concatenated. Its controller exited zero, the reviewed
cleanup recorded no surviving owned processes, and the closed packet passed
independent raw, source, build, input and runtime audits.

Historical accuracy failures and raw stability plots are preserved;
these ordinary-point passes do not close the stress/CMS accuracy items.
The 34-page `benchmark.pdf` remains the separately frozen nine-case
`progress2` snapshot; it has not been regenerated for this checkpoint.

No precise timing acceptance is claimed. Hua-Sheng's [local run TODO](timing-handoff-1/LOCAL_TIMING_TODO.md)
still requires a fresh local installation and all 66 local numerical lanes,
followed by the paired per-process timing campaign. Shared-host numerical
qualification does not replace that local deployment check.

The [pure continuation verifier](../../experiments/closed-continuation-verifier-1/README.md)
and the [combined EW verifier](ew-n4-continuation-1/verify.py) reparse only
hash-bound closed checkpoint payloads. Their explicit path/SHA APIs keep target
misses, DP/QP counts and the original interruption as outcomes and perform no
native execution.
