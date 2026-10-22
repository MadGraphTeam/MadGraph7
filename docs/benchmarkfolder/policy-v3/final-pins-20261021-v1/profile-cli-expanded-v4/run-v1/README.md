# Actual Ninja / NinjaRS MG7 CLI profiles

This study completed the declared 72 real `bin/madgraph` profile commands:
12 Standard Model processes, three fresh timing sessions per tool, Ninja6
and NinjaRS8. The initial command for each process/tool sampled 20 CLI
stability points; subsequent `-reuse 0` commands measured timing again and
retained the same stability cache. Every cache has 20 DP entries and no QP
retry entry. Those entries are not timed rescue counters or a matched-momentum
random stability dataset.

`result-list.json` lists the actual descriptors in declared execution order.
`final-validation-v1/publication-result.json` records the successful existing
strict-reader pass, exact controller coverage, retained input hashes and
scope limitations. Both controllers exited zero. All recorded process
identities and isolated groups were absent at the terminal cleanup check.
The separate failed predecessor pilots are preserved as historical failures;
their timings have not been promoted into this study.

The MG7 CLI reports total MadLoop time, DP numerator-construction time and
their difference, the reduction estimate. The difference includes scalar
masters, callbacks and assembly. It is not a direct reduction stopwatch.
The returned exact wall-time expressions, actual adaptive point counts,
printed rounding, selected helicity/filter, cards, source snapshots and
runtime hashes remain in each raw descriptor and timing metadata. The
three-session empirical bootstrap does not carry a small-sample coverage
guarantee. Different adaptive counts/RNG prefixes do not establish identical
sampled momenta.

QCD1 and EW1 changed the selected physical helicity and filter between the
initial and reused sessions, in the same way for both tools. The unchanged
scope comparator therefore rejects their aggregate three-session ratios.
The other ten cases have identical captured scopes across all six sessions.
All selected helicity weights and all raw timing estimates are positive.
All 72 successful observations remain visible; no session was replaced.

The literal-original, full-helicity, rescue-on LIGHT6 seven-pair acceptance
study is a separate measurement. These CLI profiles do not certify that
performance gate.

`generated-source-closure-v1/closure.json` binds 2,177 content-addressed
source text files covering every generated-source manifest entry in the
72 descriptors. It retains no executable, object, compiler module or native
archive. Static MG7 source is pinned by the committed specification; the
exact electroweak UFO is independently retained in
`docs/benchmarkfolder/policy-v2/ew-model-source-final-v1`.

The optional local 41.7 MB mirrored array of reader return values and its
draft receipt are excluded from publication. Original descriptors, full raw
inventories, the concise successful receipt and verified input hash map are
retained instead. Private build directories remain untouched.
