# Inactive preparation attempts

These three historical folders contain preparation/protocol stops, not completed
numerical comparisons. Their original `status: failed` fields describe the
runner outcome. They do not establish reducer inaccuracy or a runtime failure
of the physical benchmark.

| Historical label | Retained outcome | Files |
|:--|:--|--:|
| `strict-policy-ew-n3` | All six lanes rejected a worker-template/compilation mismatch during preflight | 14 |
| `strict-policy-ew-n4` | All six lanes rejected a worker-template/compilation mismatch during preflight | 14 |
| `strict-policy-qcd-n4` | Baseline/candidate CutTools workers administratively stopped before READY; four remaining candidate lanes rejected a template mismatch | 20 |

The audit covers 18 lane records: sixteen reported template preflight mismatches
and two administrative stops. Each root result agrees exactly with its lane's
failure record. All recorded performance eligibility flags are false. No folder
contains completed coefficient rows, point/status streams, or timing batches.
The requested 100 points per lane are a configuration value, not a count of
completed evaluations.

The two QCD CutTools lanes retain startup output and input manifests. Their
recorded point/card identities agree; worker templates reproduce the recorded
driver hashes, and recorded link-input hashes agree before/after compilation.
The baseline revision is `9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`; the candidate
revision is `c91cb85f3aa942c90299b3518bb1f195f55334a5`. These historical identity
records do not revalidate the original binaries or physical card/point bytes,
which are not copied into these folders. The administrative notes explicitly
state that no numerical failure is inferred. The old worker did not retain an
actionable warmup point/status trace, so this audit cannot assign a failing
physical point or determine how many evaluations occurred before interruption.
The candidate manifest identifies a linked TIRust library, but its selected
reducer is CutTools; linkage is not evidence that TIRust evaluated these points.

The sixteen mismatch lanes retain no per-lane input/build/runtime manifests or
worker output. Their exact conflicting compilation/template hashes are therefore
not available here. The frozen Python guard confirms the reported error is a
preflight template-identity rejection, and its source and all frozen protocol
hashes match their specifications. Current binaries and later worker templates
are not substituted to fill missing evidence.
Both historical worker versions use the same version-one READY schema, so
their distinct source hashes identify the different warmup protocols.

`inactive-preparation-audit.json` inventories all 48 original files by relative
path, byte length and SHA256, and hashes this note and the audit script. The
script verifies the retained identities and stopping-point classifications
without importing the historical controllers or executing native libraries.
It reads only the three exact labels above. Active `*-warmup-v2` folders and
their results are outside this audit.

From the repository root, reproduce the audit without changing any evidence:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/numerical/audit_inactive_preparations.py
```

The initial JSON was created with `--write`, which refuses to overwrite an
existing audit. No original folder, campaign summary, or native result was
modified or rerun.
