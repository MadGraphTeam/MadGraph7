# Retained legacy validation and preparation

This snapshot retains completed representative evidence, every failed or
cancelled attempt in the same collection namespaces, and the eight completed
build-only reference preparations. It is **not** a complete representative
proof or an acceptance result. Fresh final-candidate coverage and the final
eight-process correctness/performance collection remain outstanding.

| Namespace, relative to `docs/benchmarkfolder` | Retained scope |
|---|---|
| `policy-v2/representative-runs` | Original and historical candidate fresh/persistent comparisons, including strict forced-Ninja failures and an explicitly cancelled obsolete-candidate run. |
| `policy-v2/reference-runs` | Corrected-original representative comparisons; also the initial parallel MODEL-build failure and the failed forced-Ninja attempt. |
| `policy-v2/runtime-bindings-20261020` | Derived bindings and copied module/link metadata. Original run bytes are unchanged. Exact IREGI/no-IREGI signatures remain separate. |
| `policy-v2/legacy-coverage-status.json` | The 12 eligible legacy runs: 26 targets, 12 independently covered role/backend/runtime groups, complete DP/QP and loop-induced/mixed-order coverage for both legacy roles. Candidate proof remains pending. |
| `policy-v2/specs`, `representative-point-streams`, `representative-qcd1-ct-qp.json` | Exact collection specifications, full near-singular point-stream alias, and explicitly stricter representative-only QCD1 configuration. Core inputs were not changed. |
| `policy-v2/reference-preparation` | All eight clean corrected-original MODEL/subprocess rebuilds. No numerical evaluations were started by this preparation. |
| `policy-v2/reference-collection-plan` | Exact eight-case command/configuration, held for coordinated launch. The frozen ordinary collector will repeat preparation. |
| `policy-v2/final-controls-checklist.md` | Outstanding final-source controls; historical results are not recertified. |
| `experiments/persistent-workers/ninja-parity-20261020` | Initial completed QCD1/QCD2 Ninja worker experiment: four targets, fresh parity and seven blocks each. It predates final status/protocol-source retention and is historical only. Its absent run-status is preserved. |
| `experiments/persistent-workers/ninja-parity-audit-20261020` | Rerun of that four-target experiment after parameter-card shadow and failure-cleanup fixes; complete recorded status, still an obsolete exploratory protocol. |

Both early Ninja experiment directories were collected by the reproducibility
agent and independently reviewed by the Fortran audit agent. Neither contributes
to v2 proof or final timing. Their original controller hashes differ from the
frozen final controller and remain visible.

Five v2 attempts failed or were cancelled: the initial reference MODEL build,
three strict ordinary forced-Ninja attempts (original, historical candidate,
corrected-original; status 420), and the obsolete candidate QCD3 cancellation.
The exact errors, raw outputs and cleanup records are retained in their run
directories. They are excluded from proof rather than repaired or relabelled.
`representative-runs/README.md` explains the complete alternate representative
streams; no failed points were filtered from a run.

`manifest.json` hashes every retained file in these namespaces, including this
guide and `verify.py`; its own identity uses canonical JSON without the identity
field. `validation.json` records the read-only checks performed at sealing.
Nothing here changes production source or previously frozen run files.

From the repository, verify the package without compiling or evaluating physics:

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/policy-v2/retention-20261020/verify.py
```

The verifier checks retained bytes, all 17 completed v2 run bindings and full
series, fresh parity, all five failure statuses, historical prototype parity,
and build-preparation identities. It does not promote evidence or launch work.
