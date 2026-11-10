# Stress supervisor2: tracked CLI entrypoint

This is the additive mechanical successor to sealed supervisor1 (`dbfa3ca37a05f7f7e7f2f91872902adf249106968314192a273ab804a24ff1bf`). The original source proposal1, supervisor1, make fixture, failed preparation1, cards and point streams remain unchanged. The closed failure is retained portably in `../physical-stress-current-preparation-failure-1/` (index `a62dca850eeb2b150d8e34324048127e8aa8e9b4037c874a66db46c1a964f6e6`).

Preparation1 stopped before MG CLI entry because it invoked nonexistent `bin/mg5_aMC`. Both exact archived revisions track `bin/madgraph`. Successor2 invokes that file only after verifying its revision and SHA256 against the complete materialized Git inventory; this admission is recorded before the CLI process, then checked again afterward. The scientific inputs, algorithms, worker, build recipes, caps and ownership/linker helpers are unchanged.

The only other changes are fresh work/report names `physical-stress-current-2` / `physical-stress-current-qualification-2`, eight CLI cards with only their private model/output namespace changed, and `XDG_STATE_HOME=WORK/xdg/state` so incidental MG7 history stays inside this private workspace. Previously XDG_STATE_HOME was inherited or absent (the latter would use ~/.mg7); it is now explicit and included in each owned command receipt. HOME is never changed. `successor.patch` gives the full executable-supervisor diff; native_job.py and all six ownership helpers remain byte-identical to supervisor1.

From `/shared/external/madloop_tirust/madgraph7_madloop_tirust`, after `source tools/tirust_env.sh`:

```sh
python -B docs/benchmarkfolder/experiments/physical-stress-current-supervisor-2/run.py
python -B docs/benchmarkfolder/experiments/physical-stress-current-supervisor-2/run.py prepare --execute
```

The first command only prints the plan. Root alone schedules actual execution; no retry is admitted merely by creating this packet. Preparation still means eight fresh CLI exports and full MODEL/DHELAS/process objects, CPUs12–15 nonexclusive/jobs4, aggregate3600seconds with nested600-second caps. It never invokes a mass reader, loader or physics executable. Subsequent `preflight --execute` and `run --execute` require separate coordinator admission and the previous stage to pass. All native limits and scientific failure/provenance rules in supervisor1 README apply unchanged, including 24singleton requests, strict host checks, W03 rejection, current EW geometry-oracle qualification still open, and no timing claim.

Twenty-six focused tests passed, including two actual read-only `git cat-file blob <revision>:bin/madgraph` checks, literal namespace-only comparison of all eight CLI cards, missing/mutated entrypoint rejection, and private state/HOME/receipt checks. Existing22 source/parser/mocked-cleanup tests also pass. The test suite invokes no MG CLI, compiler, loader or amplitude. Independent narrow review of the final source, eight command deltas and environment isolation passed. Test-history preserves the initial over-specific __main__ assertion and the empty mocked-environment recorder failure; these were test/protocol-development failures, never scientific runs. The original supervisor1 real-GNU-make fixture still applies because MAKE_ADDENDUM is unchanged.

No successor2 preparation or native execution has run at sealing. The packet is append-only after sealing.
