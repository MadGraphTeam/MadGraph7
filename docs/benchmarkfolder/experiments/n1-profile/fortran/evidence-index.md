# n1 Fortran attribution evidence

All measurements here are diagnostic. The authoritative runtime gate is
`../../../runs/qcd_n1_uux_z/tirust-first-attempt-3/` and remains failed.

| Files | Purpose |
|---|---|
| `README.md`, `summary.json`, `analyse.py` | Reproduction, interpretation, and reproducible summary of the retained observations |
| `prepare.py`, `ml_profile.f90`, `run.py`, `control_pair.py` | Isolated preparation, timers, stage matrix, and uninstrumented paired controls |
| `original/`, `instrumented/`, `diffs/` | Exact generated source before/after instrumentation and reviewable changes |
| `preparation.json`, `built-artifacts.json` | Source, compiler, card and actual native artifact identities |
| `compile-objects.log`, `compile-link.log` | Successful Fortran compilation and final link commands |
| `static-adapter-audit.json` | Exact object hashes and static allocation/free call sites outside the adapter's internal timer brackets |
| `protocol/identity.json` | Original frozen/captured worker and helper hashes |
| `protocol/profile-run-used.py`, `protocol/control-pair-used.py` | Exact historical controllers corresponding to recorded hashes |
| `protocol/profile-campaign-captured.py`, `protocol/tirust_benchmark.py` | Captured and frozen campaign helpers; AST identity proof is in `summary.json` |
| `smoke-1/`, `smoke-1.log` | Initial 100-point pristine/instrumented numerical smoke |
| `attribution-1/`, `attribution-1.log` | All 18 stage runs, numerical outputs, return codes, clocks, profiles, loader resolution and host-load snapshots |
| `original-controls-1/`, `original-controls-1.log` | Two original-binary calibration batches and seven alternating pairs with numerical comparisons |
| `SHA256SUMS` | SHA-256 for every file in this evidence folder, excluding this manifest itself and bytecode caches |

No failed numerical observations were discarded. Batch stage timings are
inclusive and their timer overhead is retained. The matrix's three repetitions
are descriptive; original controls have seven statistical pairs but are not
qualified acceptance runs. Paths and compiler identities describe the pinned
local environment; no credentials or private keys are included.
