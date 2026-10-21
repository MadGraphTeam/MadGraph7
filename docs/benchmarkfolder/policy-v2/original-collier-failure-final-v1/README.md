# Reproduce the original Collier QCD4 calibration failures

This capsule preserves the real failed original-helper corrected calibration
for `u u~ > z z g g [virt=QCD]`. Its original 281 payloads, evidence seal,
source snapshots and root review are unchanged under `historical/`. Their
relative path strings retain their historical meaning. `input-index.json`
adds explicit mappings to the byte-identical local copies. No source
archive, executable or library binary is included.

Run using only Python's standard library:

```sh
python -B docs/benchmarkfolder/policy-v2/original-collier-failure-final-v1/reproduce.py
```

The directory can be copied elsewhere and the script run there. It reads
only local capsule files. All bound hashes are verified before source/card
interpretation or comparison arithmetic. No live worker, library, reference
clone, model import, network, compiler or amplitude call is needed.

Exit0 means **the expected failures were reproduced**. The output explicitly
keeps `scientific_agreement_passed=false` and
`stability_acceptance_passed=false`. It does not turn a failed campaign into
a successful one.

The actual Collier-only card selects backend7, `MLStabThres=1e-9`, internal
stability checking, both Laurent poles and required accuracy `1e-12`.
The captured source/card/build bindings and no-selected-QP warning are
retained. Collier returns410 at physical points17 and52; its reported
estimates are `3.6714977813117075e-9` and `8.116512431646654e-9`.
The frozen ordinary parser rejects those results. Cache-reset warnings are
preserved and are not established as the cause.

The offline coefficient comparison retains all three original backends
(CutTools1, Ninja6, Collier7), 100 physical points, total plus sole split
order, and all four Born/Laurent columns. It recomputes all2400 scalar
comparisons. The common point scale is the largest absolute Born/Laurent
magnitude across all three backends and both orders. The unchanged
tolerance is `1e-7*max(abs(a),abs(b)) + 1e-10*S`.

A separate finite-coefficient discrepancy between CutTools and Collier at
point99 reaches `1.15789685641515` times tolerance. Its two failed rows are
the duplicated total and sole split order: one physical event. Complete,
finite raw coefficients do not establish stability acceptance.

This evidence is a failed original calibration, not a final repetition,
NinjaRS comparison, analytic oracle, timing measurement or complete campaign
result. No core dataset identity is fabricated. The historical executable
and library hashes remain provenance records; reproducing the postprocessing
does not rerun or independently revalidate those binaries.
