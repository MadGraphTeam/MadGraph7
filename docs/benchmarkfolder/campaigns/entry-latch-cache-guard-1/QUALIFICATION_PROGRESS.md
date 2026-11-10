# Current-source shared-host qualification

The generated Fortran at source revision `0fcd2cd873d61d134837e1d9e87afb13a47824fd`
has completed seven of the eleven ordinary process comparisons. Each completed
process uses the same 100 saved physical points, starting-revision CutTools,
and candidate backends 1, 6, 7, 8 and 9. All 42 completed lanes pass the
`3e-6` regression criterion and the separate `1e-8` ordinary-point target.
All accepted final statuses identify the selected backend in DP. These statuses
do not instrument internal retries or floating-point exception flags.

| Process | Closed evidence | Independent audit |
|---|---|---|
| QCD `u u~ > z` | [n1 closure](n1-qualification-1/closure-index.json) | Passed |
| EW `u d~ > w+` | [n1 closure](n1-qualification-1/closure-index.json) | Passed |
| `g g > h` | [checkpoint](remaining-qualification-1/li_gg_h/checkpoint.json) | Passed |
| QCD `g g > t t~` | [checkpoint](remaining-qualification-1/qcd_n2_gg_ttx/checkpoint.json) | Passed |
| EW `e+ e- > mu+ mu-` | [checkpoint](remaining-qualification-1/ew_n2_epem_mupmum/checkpoint.json) | Passed |
| Mixed `u u~ > g a` | [checkpoint](remaining-qualification-1/mixed_uux_ga/checkpoint.json) | Passed |
| `g g > h h` | [checkpoint](remaining-qualification-1/li_gg_hh/checkpoint.json) | Passed |

Each continuation checkpoint binds its raw protocols, saved inputs, build and
source receipts, resolved runtime-library hashes, immutable stage receipts and
numerical summaries. Candidate builds are fresh; baseline numerical objects
remain unchanged. Runtime evidence uses resolved `ldd` paths and file hashes,
not sampled worker memory maps. The [continuation source index](remaining-qualification-1/source-index.json)
freezes its supervisor and dependencies. The growing supervisor status is not
part of a closed scientific checkpoint.

The two three-particle and two four-particle processes remain pending in this
snapshot. Historical accuracy failures and raw stability plots are preserved;
these ordinary-point passes do not close the stress/CMS accuracy items.

No precise timing acceptance is claimed. Hua-Sheng's [local run TODO](timing-handoff-1/LOCAL_TIMING_TODO.md)
still requires a fresh local installation and all 66 local numerical lanes,
followed by the paired per-process timing campaign. Shared-host numerical
qualification does not replace that local deployment check.
