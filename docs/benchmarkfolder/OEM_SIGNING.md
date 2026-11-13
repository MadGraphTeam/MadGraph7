# TIRust OEM signing and rebuild

Upstream signed revision: `10ce34ed8ad5b6f5f989a0e220eb3a244a71d1a1`.
MadGraph build inputs and observations: [qualification summary](https://github.com/MadGraphTeam/MadGraph7/tree/madloop_tirust-evidence-2026-11-13/docs/benchmarkfolder/qualification/tirust-oem-signing-1/summary.json).
Signed MadGraph qualification: **passed**. Scoped activation is also published
on [TIRust main](https://github.com/huashengshao/TIRust/commit/de60dc480fec6475d2373a533c64d1c40ce23b2e),
following its [separate completed functional validation](https://github.com/MadGraphTeam/MadGraph7/tree/madloop_tirust-evidence-2026-11-13/docs/benchmarkfolder/qualification/tirust-oem-main-publication-1/README.md).

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

| Check | Recorded result |
|---|---|
| Fresh-process activation without `SYMBOLICA_LICENSE`, real master fallback, scoped host restriction, calling threads, error and panic cleanup | Passed; real master fallback, Float256, calling threads, error and panic cleanup, restricted host before/after. |
| Existing concurrent workspace numerical test | Passed without a personal key. |
| Core/masters and default-member Rust suites; core examples | Signed predecessor: core/masters 661 passes, default workspace 707 passes; each has 31 existing ignores. All 45 examples compiled in both builds. |
| Upstream benchmark PDF reproducibility | Byte-identical after metadata update; measured data unchanged. |
| Fresh installed bridge identity and signed source/module hashes | Remote signed revision and OEM module hash verified; DSO `85ee0cdf860e…`. |
| Installed Fortran smoke, genuine QP transport and poles-only support | Passed installed `--require-poles`: DP, binary128 last bit, UV/IR, full/poles-only, caches and lifecycle. |
| Bridge ABI/export tests; installation rollback and tamper checks | 11 release bridge tests, export audit, 63 Python tests and 66 subtests passed. |
| Coexistence with unchanged NinjaRS | Z and ZZgg (2→4), one saved physical point each: all five reducers plus rescue chain pass both 3e-6 and 1e-8 criteria. |

The pinned signed commit, based on `854b79`, is now an ancestor of published
upstream main `de60dc480fec6475d2373a533c64d1c40ce23b2e`. The merged upstream
source passed 37 preflight, 662 core/masters and 708 default-member tests;
core and default each retain 31 ignored cases. Its metadata-only publication
child preserves those numerical sources. MadGraph keeps the validated signed
predecessor pin to preserve its numerical baseline; newer upstream
triangle-rotation changes are retained on main without changing this pin.

In the 2→4 check, standalone TIRust was accepted in **QP**; the TIRust-first mixed chain was accepted by **NinjaRS DP**. Z production used native TIRust DP. Actual rescue behavior is retained in the summary; these are coefficient/loading checks, not timing acceptance.

The signing change adds activation boundaries; numerical bodies in the pinned revision are unchanged.
That source comparison and interface tests do not qualify new full-amplitude
accuracy or performance. The report's measured campaigns remain bound to their
recorded sources. This publication leaves the current PDF, ledger and historical
evidence unchanged.
Physical-stress accuracy failures, broader experimental qualifications and the
formal eight-process runtime gates remain open. Hua-Sheng can use the new
[signed-source local handoff](https://github.com/MadGraphTeam/MadGraph7/tree/madloop_tirust-evidence-2026-11-13/docs/benchmarkfolder/campaigns/tirust-oem-signed-1/timing-handoff-1/LOCAL_TIMING_TODO.md)
for local installation, fresh numerical qualification, paired timings and report
generation. Its eleven cases, 66 numerical lanes and eight main runtime gates
are registered with all results pending. The predecessor's frozen local TODO
remains historical input; its qualifications do not transfer to this source.
