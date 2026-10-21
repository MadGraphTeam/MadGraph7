# Held persistent Ninja6/NinjaRS8 timing launch

`launch.py` accepts only the exact reviewed core `de65bb6b...` and loop
`56cee094...` specifications. Both use seven adjacent alternating pairs, 100
fixed points, full helicity sums, poles and rescue, and CPU207 with the four
thread limits set to one. Existing report/work namespaces are rejected.

The root orchestrator must first retain release receipts for every other
numerical, profiling and compiler job. The launcher requires both the existing
collector's exact handoff and a separate bound quiet receipt containing:

```json
{
  "schema": "final-timing-quiet-handoff-v1",
  "root_authorized": true,
  "stage": "core",
  "cpu": 207,
  "quiet_cpu207": true,
  "all_other_numerical_and_compiler_work_released": true,
  "specification_sha256": "exact reviewed specification digest",
  "collector_handoff_sha256": "exact collector handoff digest",
  "release_evidence": [{"path": "retained root release proof", "sha256": "its digest"}]
}
```

This example is documentation, not an execution authorization. For the loop
stage use `"stage": "loop"` and its distinct exact collector handoff. After the
root records the actual quiet proof, launch one retained session from the
repository root:

```sh
source .ninjars-build/dev-env.sh
taskset -c 6,7,207 python3 -B \
  docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/timing-launch/launch.py core \
  --handoff /absolute/path/to/actual-collector-handoff.json \
  --quiet-proof /absolute/path/to/actual-root-quiet-proof.json \
  --attempt docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/timing-launch/core-launch-v1 \
  --execute-numerical
```

Normal tool-session yielding and conversation compaction do not require a new
launch. Keep that same controller running to completion. Its separate process
group, start ticks, logs, exact handoffs and terminal result are retained.
Interruptions during process creation are deferred until durable ownership is
recorded. Failure cleanup checks exact identities; a successful execution receipt
also requires matched exit-zero reaps and absent child/controller groups.

The collectors retain every completed command's raw output and dispatch entry.
They do not implement in-place resumption after process death: preserve partial
attempts, never silently append, restart or label incomplete pairs complete. A
separately reviewed continuation design would be required after interruption.
The launcher has no automatic retry and makes no scientific pass decision.

Diagnostic scheduling evidence projects about 13.4 hours for the core queue
before clean-build/source-check overhead, with QCD4 dominating. The final
NinjaRS HH100 cost is still unknown. These are planning estimates from different
CPU/source roles, not paired benchmark results; see the bound post-core queue
readiness record. Actual confidence intervals must satisfy the unchanged 1.05
gate. The private MG7 CLI patch remains held until all source-guarded campaigns
finish.
