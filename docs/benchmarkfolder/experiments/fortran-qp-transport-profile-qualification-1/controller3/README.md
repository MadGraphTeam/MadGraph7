# Private QP profile controller3 — RUN-only source proposal

This additive successor has not executed native work. It retains controller2
source `c5e6eeb2…`, successful BUILD terminal `90ef4a7b…`, build result
`58d5e4e2…`, and failed mode0 terminal `b7066be2…` / index `a8d8e1c7…`
unchanged. The failed mode stopped before loader/worker/model setup: private
copying dereferenced the process-local resource-directory symlink, adding a
fourth card path with exactly the same bytes. This is a preparation inventory
failure, not a numerical failure. Modes1/2 were never attempted.

The sole admission correction requires the original three card paths and one
explicit alias:

- `Cards/MadLoopParams.dat`
- `SubProcesses/MadLoopParams.dat`
- `SubProcesses/MadLoop5_resources/MadLoopParams.dat`
- `SubProcesses/P0_uux_zgg/MadLoop5_resources/MadLoopParams.dat`

All four must have SHA256
`5300bd6d7f3c4553c3402ab7a044e8fbca3c646f451a5f586215933bfa113b85`.
Missing/changed original paths, a missing/changed alias or any other extra path
fails. All four observed hashes remain in before/after receipts; no unknown path
is silently normalized. The file-backed regression is the actual retained
three-versus-four failure inventory, including mutation/extra/missing controls.

The original process resource link is `../MadLoop5_resources`. The built private
copy has an independent dereferenced directory. Every fresh clone excludes all
`*HelFilter.dat` and `*LoopFilter.dat` files throughout the tree, including the
private alias, before the original setup/bootstrap/full100 learning sequence.
Thus no old filter is imported or shortened learning admitted. Exact complete
steady warmup parity remains mandatory; this source proposal does not establish
runtime filter equivalence by itself.

BUILD is disabled in the CLI and supervisor. The reviewed BUILD2 executable and
objects are reused without relink/recompilation. `source-compatibility.json`
pins the original17-payload Fortran proposal `049c26fb…`, the exact unchanged
worker_flags and loader helpers, all five actual compiled source files, exact
makefile derivation, executable, donor source/build/result and original resource
alias. The donor's source hash remains `c5e6eeb2…`; it is never rewritten as the
new controller hash. Its actual LOAD/prelink/object/source/runtime checks are
replayed read-only before new workers. Source3 has a distinct working directory
and append-only mode receipts; old failed paths remain immutable.

Native scope is unchanged: three fresh mode0→1→2 workers, each the full100 saved
ordinary QCD n3 learning traversal followed by exactly once each point1 (219DP)
and point16 (323QP). Both orders' coefficient and accuracy words/status must
match retained expected results and mode0. Actual IEEE flag/halting/rounding
pre/post observations must match across modes; sticky flags are never cleared.
No RUN timing command, fixed-QP policy, new point, optimization, library change,
cache insertion or Rust build is permitted. Counter/query/clock/key guards and
actual PT_INTERP/provider/maps checks are unchanged from reviewed controller2.

Each mode has300 seconds total, including mutex waiting/checks/copying/full
learning/anchors/parsing and60 seconds reserved cleanup; total900 seconds for
three stages. CPU12 is nonexclusive. The same native mutex and pinned
StageDeadline/OwnedChild/SignalRequests apply, with16GiB address space,
512MiB per-file and zero core limits. Uncertain cleanup or the first failure
stops remaining modes; no automatic retry. Cleanup describes observed,
identity-checked descendants, not a kernel proof against every possible
spawn/reparent race. Administrative failure records can be written after a cap
and cannot establish a passed measurement.

From the repository root, after source review, use an explicit new source hash
and the unchanged reviewed BUILD2 terminal hash:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/fortran-qp-transport-profile-controller-3/controller.py \
  --preflight --source REVIEWED_SOURCE3_SHA256 \
  --reviewed-build 90ef4a7b47c0a795b9f2bd80a4c4b059d1930af4a5ad05e2e68ba70b3c516a11
# Only after separate RUN3 admission:
python3 -B docs/benchmarkfolder/experiments/fortran-qp-transport-profile-controller-3/controller.py \
  --run --source REVIEWED_SOURCE3_SHA256 \
  --reviewed-build 90ef4a7b47c0a795b9f2bd80a4c4b059d1930af4a5ad05e2e68ba70b3c516a11
```

Pure tests invoke no compiler, loader or reducer:

```sh
python3 -B -m pytest -q -p no:cacheprovider \
  docs/benchmarkfolder/experiments/fortran-qp-transport-profile-controller-3
```

Original controller1 failure and controller2 build/failure remain pinned ancestry.
Only successfully closed, parity-qualified modes may support descriptive
stage/counter/repeat accounting. No cache benefit or precise performance
acceptance is inferred; Hua-Sheng's precise local timing TODO remains unchanged.
