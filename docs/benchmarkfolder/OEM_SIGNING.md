# TIRust OEM signing and rebuild

Upstream signed revision: `10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1`.
MadGraph build inputs and observations: [qualification summary](qualification/tirust-oem-signing-1/summary.json).
Signed MadGraph qualification: **passed**. The separate merged upstream-main suites are in progress.

TIRust's core crate, `iregi_rust`, embeds the Symbolica library activation
issued for it. The signed offline expiry is 1 March 2028. Private RAII guards
activate the key during core plan construction and reduction, including calls
into `iregi-masters` and oneloopmaster on that call chain. External calling
threads each enter through the guarded API. Return, error and panic unwinding
release the scope; unrelated host Symbolica operations remain subject to their
own licensing requirements. Standalone oneloopmaster calls and graph/numerator
preparation outside the TIRust core are outside this activation. The Symbolica
library agreement determines the authorized scope.

Existing installed libraries require rebuilding. From the MadGraph repository
root, use the normal MG command interface:

```text
./bin/madgraph
MG7> install tirust --update
```

A separate installation prefix can preserve a previous installation for
comparison. Explicitly select the newly installed library for any qualification
or benchmark run. Successful reuse of an old installation does not prove that
it contains the signed revision. `--source` and `--oneloop-source` remain
available; their source-tree and revision provenance must identify the intended
copies. NinjaRS and the pinned Symbolica/oneloopmaster revisions are unchanged.

| Check | Result to fill from actual execution |
|---|---|
| Fresh-process activation without `SYMBOLICA_LICENSE`, real master fallback, scoped host restriction, calling threads, error and panic cleanup | Passed; real master fallback, Float256, calling threads, error and panic cleanup, restricted host before/after. |
| Existing concurrent workspace numerical test | Passed without a personal key. |
| Core/masters and default-member Rust suites; core examples | Signed predecessor: core/masters 661 passes, default workspace 707 passes; each has 31 existing ignores. All 45 examples compiled in both builds. |
| Upstream benchmark PDF reproducibility | Byte-identical after metadata update; measured data unchanged. |
| Fresh installed bridge identity and signed source/module hashes | Remote signed revision and OEM module hash verified; DSO `85ee0cdf860e…`. |
| Installed Fortran smoke, genuine QP transport and poles-only support | Passed installed `--require-poles`: DP, binary128 last bit, UV/IR, full/poles-only, caches and lifecycle. |
| Bridge ABI/export tests; installation rollback and tamper checks | 11 release bridge tests, export audit, 63 Python tests and 66 subtests passed. |
| Coexistence with unchanged NinjaRS | Z and ZZgg (2→4), one saved physical point each: all five reducers plus rescue chain pass both 3e-6 and 1e-8 criteria. |

The pinned signed commit is published on upstream `hs/symbolica-oem`, based on `854b79`. Newer upstream triangle-rotation changes are preserved in a separate local merge awaiting its required full suites before the main push. MadGraph pins the validated signed predecessor to preserve its numerical baseline.

In the 2→4 check, standalone TIRust was accepted in **QP**; the TIRust-first mixed chain was accepted by **NinjaRS DP**. Z production used native TIRust DP. Actual rescue behavior is retained in the summary; these are coefficient/loading checks, not timing acceptance.

The signing change adds activation boundaries; numerical bodies in the pinned revision are unchanged.
That source comparison and interface tests do not qualify new full-amplitude
accuracy or performance. The 44-page benchmark and its measured campaign remain
bound to their recorded predecessor source. They are retained unchanged.
Physical-stress accuracy failures, broader experimental qualifications and the
formal eight-process runtime gates remain open. Hua-Sheng's precise local
campaign must use a newly frozen signed-source installation and matching
handoff; the predecessor's frozen local TODO is historical input, not automatic
acceptance of the signing successor.
