# TIRust acceptance evidence audit

This audit reads retained evidence and adds four pure Python metadata tests. It
does not compile, execute native reducers, alter production, rerun a benchmark,
or modify previous evidence indexes. The ordinary 100-point campaign, formal
timing, and physical/EW/complex-mass stress work remain with their existing
owners. Evidence links below distinguish tested scope from remaining gaps.

The current isolated installation here means revision
`059bb9c979194bfda783cad83443223def8df280`, DSO `eb3d67fd…` and Fortran archive
`0c86349a…`, as recorded in its [installation qualification](../../qualification/tirust-install/qp-transport-cache/README.md).
Historical generated integration used DSO `49af12f0…`; historical mixed-link
tests used debug TIRust libraries. These identities are not interchangeable.

| Requirement | Retained evidence and scope | Remaining item or limit |
|---|---|---|
| Accelerated partial evaluation through N6/rank6 | [Native integration log](../../qualification/poles-only/tensor-integration-tests.log) passes all nine tests. [Core unit log](../../qualification/poles-only/tensor-unit-tests.log) names the no-finite-evaluator and finite-scratch-omission tests. [Patch source](../../../../vendor/tirust/patches/0001-direct-poles-only.patch#L1086) compares full/partial for N1–6, rank6, f64/DD/113-bit MPFR; cumulative tensors include lower ranks. [Scalar tests](../../qualification/poles-only/scalar-regression.log) cover generated pole kernels. | This is native-core coverage. [Generated diagnostics](../poles-diagnostics/README.md) cover historical N6/rank2 and N4/rank4, with 12 explicit DP refusals and all 54 QP pairs passing. No retained current-artifact **generated N6/rank6 partial** result was found. The independent [210-component mapping test](../../../../tests/unit_tests/loop/test_tirust_fortran.py#L18) supplies complementary mapping coverage, not that missing end-to-end case. |
| Installation and rollback | [Current CLI installation](../../qualification/tirust-install/qp-transport-cache/README.md) fetched pinned sources without overrides and passed actual native smoke. [Earlier clean-cache installation](../../qualification/tirust-install/clean-download/index.json) is separately identified. [Transaction audit](../../qualification/tirust-install/configuration-transaction-audit.json) records 66 tests. | Rollback cases deliberately inject mocked compiler/publication/configuration failures. They are not observations of a real failed native compilation. Current installation reused caches and does not inherit the earlier clean-cache claim. |
| Missing/incompatible libraries | Existing [missing/corrupt/replaced-artifact tests](../../../../tests/unit_tests/various/test_tirust_install.py#L168) use fixture installation files. New [retained test patch](metadata-tests.patch) exercises actual metadata readers. | **Two open schema failures:** JSON `abi_version: true` and `link_order_version: true` are accepted. Four tests/25 mutation cases: 23 reject correctly, two fail. Production remains unchanged; details below. No native loader-failure experiment was performed by this audit. |
| Relocated internal export | [Historical release manifest](../../qualification/madloop-adapter/release/manifest.json) records actual relocation, removal of the source prefix, normal generated builds, and six internal-output checks. [Procedure](../../qualification/madloop-adapter/README.md#L25) identifies `$ORIGIN` lookup and supported internal dependencies. | This is real historical 49af relocation evidence, not a current eb3d relocation run. No duplicate relocation job was started. |
| Mixed backend initialization/coexistence | [Four serial TIRust/NinjaRS cases](../../qualification/tirust-bridge/mixed-link/serial-results.json) cover both link orders and both initialization orders. [Comparison-library evidence](../../qualification/comparison-backends/README.md) separately covers 1/6/7/8. [Historical generated external checks](../../qualification/madloop-adapter/README.md#L15) select all five reducers. | Debug/historical results remain labeled. Forced positive CT modes do not qualify automatic stability. Current campaign evidence must bind its own installed libraries; it must not inherit these binary identities. |
| Historical statistics compatibility | [Tests](../../../../tests/unit_tests/various/test_ninjars_statistics.py#L110) compile actual LO/NLO tally/writer fragments and test legacy XML, optional counters and aliases. Current test-source SHA matches commit `101cf438c58907be071b1907defe650222b54b54`. [Retained aggregate log](../../qualification/madloop-adapter/debug/final-unit.log) reports 92 passes. | Exact historical statistics invocation and executed-source binding were not recovered. The 92-pass log has no module names. Source identity alone cannot attribute that run to these tests. No statistics rerun occurred. |
| Cache failures, budgets and isolation | Current [90-call DP replay](../../qualification/tirust-install/qp-transport-cache/release-validation.json), [65-call public Fortran parity](../../qualification/tirust-install/qp-transport-cache/success-parity/results.json), and [actual QP transport boundary](../../qualification/tirust-install/qp-transport-cache/archive-tests/real-boundary-analysis.json) cover successful results, bypass/budget behavior and failure cleanup. [Historical generated cache diagnostics](../poles-diagnostics/final-qcd-n1-cache/README.md) cover topology exclusion, independent checks and full/partial/precision stores. | The 28 malformed transport and seven clear-failure cases use an independent mock C provider linked to the actual installed archive. The separately labeled huge-mass boundary uses the real provider. Numerical-cache accounting is not a process-RSS/plan/workspace memory bound. |
| Event/subprocess lifecycle | Current [attempt5 results](../subprocess-cache-lifecycle/attempt5/results.json) show all four actual workers passing: plain/observed × cache disabled/enabled. Each covers 19 physical calls, 24 DP/QP key requests, automatic clearing, context recreation and exact parity. Actual exit status is retained per worker under [execution](../subprocess-cache-lifecycle/attempt5/execution/native-observed-cache-all.json). | Owner packaging and independent review were pending at this audit snapshot. This coverage is no longer classified as missing. Rank6 partial evaluation and timing are outside this bounded protocol. |
| Scale and scaleless poles | [Native partial tests](../../qualification/poles-only/tensor-integration-tests.log) cover changed scales/rotations and scaleless residues. Current [replay source](../../qualification/tirust-install/qp-transport-cache/validate_release.py#L280) mutates scale/mass/shifts and analytically checks all 210 rank≤6 scaleless UV/IR moments at line359. Current lifecycle records scale/MZ variation and DP/QP coefficient-change witnesses. | Raw replay mutations allow explicit DP refusal and preserve it. Physical MZ variation changes external mass/invariants; internal complex-mass variation belongs to the separate tensor keys. |

The new metadata regression run is retained verbatim in
[metadata-tests.txt](metadata-tests.txt), with command, source hashes and scope
in [audit.json](audit.json). All installation subprocesses were replaced by the
existing fixture callback; no compiler, network fetch or native smoke executed.
The fixture contents do not represent a real loadable installation.

To reproduce the known red tests, apply `metadata-tests.patch` to a scratch copy
of the recorded base revision `47c480b72e3fd99bde5780b5986fb1314ef6f9e4`. The
unmodified test module must first match SHA256
`e92c4526030d6fa242b8be00a477c354376e6c0ebb58ff18d9b8492e0fca61db`; the
patched module must match `audit.json` before running its exact `test_run.argv`
with the recorded environment. The working-tree test module is intentionally
unstaged while the production fix waits; the committed patch is the portable
source of this regression evidence. Do not apply the separate production
proposal when reproducing the retained failures.

For the two failures, the real [ABI guard](../../../../madgraph/various/tirust_install.py#L54)
and [link-order guard](../../../../madgraph/various/tirust_install.py#L64) compare
values with integer 1. Python considers `True == 1`, so `read_config` does not
raise. The regression expects a JSON integer version, not a boolean. Failing
cases stop at that assertion; they do not independently establish later
availability checks. No expected-failure decorator or implementation fix masks
the issue. Any production correction requires coordination with the currently
frozen numerical campaign.

[The proposed two-guard patch](proposed-strict-version-types.patch) is text only
and has **not been applied or tested**. It proposes an explicit schema policy:
both version fields must be JSON integers, excluding booleans. Under that policy
JSON `1.0` would also be rejected because it decodes to a float. The current
tests do not classify or test integral floats; accepting `1.0` as a compatible
numeric spelling remains a policy choice until the proposed strict-integer
schema is adopted. The two observed failures concern JSON booleans only.

Historical statistics recovery searched the portable adapter manifests/logs,
their original `.tirust-build/adapter-export` logs and packaging/runners, the
documentation tree, and the statistics file's Git history. Original and portable
`final-unit.log` are byte-identical. The debug manifest hashes that log but records
a dirty tree and no statistics command/source identity. Commit/source equality
was recovered; execution provenance remains open. The bounded search scope and
hashes are recorded in `audit.json`; absence here is not a claim that no external
terminal history could contain the command.
