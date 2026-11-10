# Completed source preparation

All eleven private baseline reuses and eleven fresh candidate CLI/source builds
completed successfully on CPUs 12–15 with four build jobs. The three serialized
stages and their exact commands, dates, exit codes and console hashes are in
`status.json` and `specification.json`. Baseline reuse first proved the untouched
old executable could be relinked byte-identically, then rebuilt only the benchmark
driver. Candidate numerical objects were rebuilt from freshly exported source.

The frozen index binds all 22 generation/compilation records, compiler logs,
private baseline proofs, source identities, eleven actual candidate CLI commands,
point/card inputs, configuration snapshots, 22 mass-setup logs, and this verifier.
`results.json` records the independent post-build source/executable/dependency and
object checks. The original four configuration files remained byte-identical.
Every candidate configuration-changing CLI command uses `--no_save`, and every
new CLI child uses `PYTHONHASHSEED=0`; original baseline export seeds remain
honestly unrecorded.

The registered exporter revision is b33801545f1361fd6a86e3d1d125560abb367870.
The first two candidate generation records observed 63b09e4581c29193847f5511cb864c0ce7f3c795,
and the rest observed 560497178d44883638e61520189a89adfc1b9d5e after a documentation
commit. All eleven have the exact same registered exporter file hash. The
observed revisions are retained without relabeling or unnecessary rebuilds.

No amplitude or reducer evaluations ran. Each mandatory `check --masses` call
executes parameter setup and writes external masses, then stops before the worker
reaches amplitude initialization. Source preparation does not establish numerical
agreement or performance acceptance. Build elapsed times are not benchmark times.
The supervisor completed normally; its original interrupted-child cleanup caveat
was never exercised and its executed source is preserved unchanged.

From the repository root, verify retained portable evidence without compilation:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/campaigns/split-dp-terminal-1/preparation-1/verify.py
```

Add `--live` to also rehash the currently retained ignored outputs, dependencies,
original configurations and masses. It performs no compiler or reducer calls.
These live checks describe source preparation and should run before numerical
workers deliberately create private cards/filter state. The numerical supervisor
performs its own per-run checks afterward. Numerical qualification remains a
separately reviewed and authorized stage.
