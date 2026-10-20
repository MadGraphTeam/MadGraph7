# Latest user-requested pause — 2026-10-21 (Europe/Zurich)

Work is paused again at the user's immediate request. Do not resume any job
until explicitly instructed. Latest pushed milestone before this checkpoint:
`ffbca3d46` (fresh signed-zero installation and copied SA/NLO checks pass).

New results saved since the previous pause:

- Existing NinjaRS MG7 suites: 113 tests passed. Fourteen new upstream/interface
  tests are written; 25 focused tests including existing installer/statistics
  checks pass. The new files remain saved but uncommitted pending root review:
  `tools/ninjars_upstream.py` and
  `tests/unit_tests/various/test_ninjars_{upstream,integration_boundaries}.py`.
- Corrected parser plus allocation tensor: all 22 tests, 21 golden lanes and
  both full 1,392-request replays pass exactly. The uncommitted 123-file evidence
  seal is `6043ed79a1f0bd5bd0936db988a3e6ba2205fd21fe780de510a99ff016d20d26`
  in `experiments/ninjars-mp-tensor-allocation-transport-v1`. No production
  tensor change, static-library build or process timing was started.
- Native Float256 compilation is still incomplete; its tests/replay did not
  start. The four exact process identities are stopped in
  `.benchmark-work/pause-native-20261021-second.json`.
- The broad polynomial/exporter/HELAS regression suite is suspended before its
  final result. Some IO comparisons fail and require investigation; do not
  report this suite as passed. Its controller and process descendants, the
  reference collector/worker, and the read-only upstream watcher are stopped
  in `.benchmark-work/pause-20261021-second.json`. Already-exited zombie
  children are explicitly classified separately from stopped live jobs.
- The reuse census and supplemental loop-induced/threshold plan are saved and
  sealed. Both upstream main branches still match the original pins; the new
  checker and retained live checkpoint confirm no update is pending.

Resume-time timeout caveats below still apply. The current native and reference
roots have been stopped a second time; use these newest records. Preserve
uncommitted evidence and source; do not blindly clean untracked files. All
agents are idle. All eight-process gates, final optimization decisions, expanded
stability/loop-induced benchmarks and the final report remain outstanding.

## Historical first pause — 2026-10-20 22:07 UTC

Do not resume work or suspended jobs until the user explicitly resumes the goal.
All work stays in this repository on `madloop_ninjars`.

The latest tested milestone before this checkpoint is `9b97a94c4`, pushed to
`origin/madloop_ninjars`. It retains the independently verified 21-pair MP tensor
callback result: allocation/original ratio 0.907561, 95% interval
[0.898288, 0.912750], with exact values, precision and signed zeros throughout.
This remains an isolated kernel result. Production Fortran's earlier QP sharing
gain is 3.19% in the retained QCD3 study; DP sharing showed no gain.
The full eight-process correctness/performance gates and final report are open.

The signed-zero transport correction is committed as `9f13d10cc`. Its fresh
actual-CLI installation passed, including compiled Fortran DP/compensated/MP
smokes. Private archive SHA256:
`832fc0d1198814b9cc80a65e47bc7aa5dc204f7d25701f206c910cd188fbe80b`.
Standalone and NLO CLI exports both completed successfully in
`docs/benchmarkfolder/policy-v2/install-export-abi3-zero-sign-v1/`.
Relocation, hidden-installation runtime checks and evidence sealing have not run.
The main `HEPTools/ninjars` installation still has its earlier archive identity.

Suspended jobs are preserved with SIGSTOP, not killed or restarted:

- Corrected-original reference collector 2004654 and QCD4 CutTools worker
  2962065, plus monitor 2328586 and log follower 2017168. Exact commands and PID
  start ticks: `.benchmark-work/pause-20261020T2207Z.json`.
- Native Float256 prototype compile: shell 2962687, controller 2962690,
  cargo 2962840 and rustc 3063368. Record:
  `docs/benchmarkfolder/experiments/ninjars-native-float256-v1/pause-20261020.json`.
  No study tests/replay ran.
- Corrected-parser plus allocation prototype compile: controller 3040051,
  cargo 3048274 and compiler 3049337. Record:
  `.benchmark-work/experiments/ninjars-mp-tensor-allocation-transport-v1/pause-20261020.json`.
  No combined tests/replay ran.
- The offline reuse/sparsity audit has no live child; its partial scope is saved
  in `docs/benchmarkfolder/experiments/ninjars-reuse-census-v1/PAUSED.md`.

On resume, inspect process identities and retained records before sending any
signal. PIDs can be reused. Compile-controller timeout clocks include suspended
wall time: resume compiler/cargo descendants first, let them finish, then resume
the controller so it observes an exited child before checking its deadline.
The reference collector has a 24-hour timeout; inspect its exact wait/protocol
state and deadline separately. Preserve any expired/interrupted attempt and use
a fresh namespace if necessary. Initialization or timing intervals spanning this
pause must not be represented as performance measurements.

Next work: finish the fresh copied SA/NLO runtime checks, coordinate the single
MP engine slot for prototype correctness gates, evaluate native Float256
dispatch without changing precision/thresholds, finish the reuse census, and
continue the original reference/final benchmark collection. Unsealed preparation
and live logs remain on disk without being presented as validated evidence.
