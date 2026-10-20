# Profile controller smoke-test evidence

This package covers only the three `profile-controller-*` runs listed in
`manifest.json`. It excludes future final-candidate profiles and all performance
acceptance data.

| Run suffix | Outcome |
|---|---|
| `original-qcd1-20261020` | Failed because the supplied `perf` executable path was mistyped. The initialized worker was cleaned up; the error, log, inputs and exact controller revision remain retained. |
| `original-qcd1-retry1-20261020` | Earlier successful controller revision: three full-stream blocks, 409 CPU-clock samples. |
| `original-qcd1-retry2-20261020` | Final reviewed controller revision: three full-stream blocks, 405 CPU-clock samples. |

Each successful run uses the unchanged bound original Ninja QCD1 worker and all
100 retained points. Each block traverses the stream 300 times: 30,000
evaluations per block, 90,000 per run. All 200 physical point/order rows captured
from each final traversal match the bound reference exactly, including the
accuracy estimates and stable status 216. There are no numerical or rescue
exceptions hidden by the profile classification.

The final controller binds the exact source/role/backend, executable, source
manifest, reduction libraries, physical card and effective MadLoop card. It
retains raw outputs before parsing and comparison, checks identities after
execution, and records worker/profiler failures while cleaning up its workers. The frozen
benchmark collector and reference run are unchanged.

The sampling interval excludes initialization/warmup but includes worker output
serialization between commands and profiler overhead. Stack categories use the
first recognized ancestor; unmatched work remains `other`. These are diagnostic
CPU-clock samples, not exclusive stage timers or acceptance measurements.

`raw-perf/` retains exact copies of both successful perf recordings, matched to
their recorded digests; the original run directories are unchanged. Their
already-symbolized `perf-stacks.txt` files permit offline category recomputation
without disposable generated executables. Re-decoding raw perf recordings may
require those executables and the matching toolchain.

`manifest.json` seals this guide, the validator, the three original evidence
directories and raw recordings. `validation.json` records independent raw-block
parity and stack-count checks. To repeat these read-only checks from the
repository:

```sh
source .ninjars-build/dev-env.sh
export TMPDIR="$PWD/.ninjars-build/tmp"
python -B docs/benchmarkfolder/policy-v2/profiles/controller-evidence-20261020/verify.py
```
