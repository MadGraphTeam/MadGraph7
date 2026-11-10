# Current-source physical stress supervisor

This packet implements sealed `../physical-stress-current-source-proposal-1/` (index `5478869fa719db000df7df825202f3ae827080d1f00b43f38bad1506f5ab54ac`). That proposal remains unchanged. This packet has executed **source checks and mocked tests only**: no CLI generation, compiler, ELF loader, mass reader or amplitude command has run.

The four original two-point streams and physical cards, all five explicit candidate reducers, and baseline CutTools give 24 requests. Candidate source is revision `2c13246ffa63aa357ad1e29e5e79aff63d88dc7e`, generator inventory `a90f80545696a648dbcfdfb8de73b1e319364edf2b93d02d4458c7b9656b8b19`, campaign manifest `ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0`. Baseline source is `9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`. All nine declared installed TIRust artifacts are pinned, including the `eb3d67fd...` DSO; existing other reducer archives and metadata are checked before and after. No adaptive bridge or changed scientific policy is admitted.

## Execution, separately admitted by the coordinator

Run from `/shared/external/madloop_tirust/madgraph7_madloop_tirust` after `source tools/tirust_env.sh`.

```sh
python -B docs/benchmarkfolder/experiments/physical-stress-current-supervisor-1/run.py
python -B docs/benchmarkfolder/experiments/physical-stress-current-supervisor-1/run.py prepare --execute
python -B docs/benchmarkfolder/experiments/physical-stress-current-supervisor-1/run.py preflight --execute
python -B docs/benchmarkfolder/experiments/physical-stress-current-supervisor-1/run.py run --execute
```

The first command only prints a plan. Preparation requires fresh `.tirust-build/physical-stress-current-1` and `docs/benchmarkfolder/experiments/physical-stress-current-qualification-1` directories. Every actual stage is append-only, with no retry. A prior successful stage is required for the next one. Root schedules and executes the stages; an agent must not duplicate them.

Preparation has an inclusive 3,600-second cap, nonexclusive CPUs 12–15 and four compiler jobs. It archives each exact Git revision into an isolated clone, copies read-only prebuilt external CutTools/IREGI dependency artifacts, copies the pinned EW model, performs eight actual CLI exports, and compiles every MODEL, DHELAS and process object afresh. Each CLI and process build has a nested 600-second cap. Inputs, source inventories, compiler recipes, object hashes, prebound transitive linker scripts/libraries and actual link maps are retained. Installed dependencies and shared configuration must remain unchanged. **Preparation never invokes `--masses`, a loader, or a physics executable.**

Preflight is a separately admitted 600-second stage. For each of the eight binaries it invokes the actual ELF PT_INTERP loader with `--list`, then `./check --masses`, and checks the masses against the original point metadata. Each binary has an inclusive 60-second bound and mass execution at most 30 seconds. This is native setup, not amplitude qualification.

The final separately admitted stage makes a fresh private copy of each matching completed export for each request, applies the original strict card values with exactly one selected reducer, removes only its private generated filters, and preserves all stdout. Each worker executes `taskset -c 12 ./check <sealed-selected.dat> <FIFO>` through the frozen Worker protocol, serially on nonexclusive CPU 12. Warmup and traversal each retain the original 180-second protocol bounds; the whole request has a 380-second inclusive bound with a cleanup reserve. Total native execution allowance is 9,120 seconds. Only backend 8/9 requests acquire `.tirust-build/numerical-tests.lock`; lock waiting is separately bounded at 3,600 seconds per acquisition and starts before the worker deadline. No CPU reservation or timing acceptance is claimed.

`ownership/` is copied byte-for-byte from the successful current lifecycle proposal2: a single OwnedChild owner performs subprocess creation, signals, waiting, cleanup and receipts. SIGALRM expiry is deferred during Popen ownership transfer, without passing a blocked mask to children. Descendants are killed/reaped on timeout; any unproven cleanup stops the stage. Physics rejections, finite-output failures, W03 omissions and cleanly closed bounded worker timeouts remain failed lanes and do not suppress later lanes. Unexpected runtime identities, controller failures and cleanup failures stop. A stage abort records the attempted/deferred request and explicitly lists all unfinished requests. No failure is retried automatically.

## Two mechanical implementation amendments to proposal1

The sealed source proposal lists one `make --always-make -j4 check`. The implementation first builds `check_sa.o` and the original `$(PROCESS)` through the additive `stress.mk` target, freezes their exact hashes and transitive linker input closure, then invokes the unchanged original `check` link target with a `-Wl,-Map,stress.map` addition to LDFLAGS. MODEL/DHELAS and all numerical objects still receive full fresh builds and the identical effective compiler policy. This split establishes pre-link provenance; it changes no scientific source or arithmetic flag.

The sealed proposal's illustrative stdin ends `STOP 2 0`. The retained Worker actually sends `RUN 1 1` followed by `STOP 0 0`. The implementation preserves that already-qualified protocol unchanged and records its actual bytes explicitly in every specification. STOP sequence numbers do not enter any amplitude; no claim is made that these illustrative strings are byte-identical.

Card serialization copies the frozen values literally into each clone's generated card without importing shared mutable MadLoopParam Python code. The four TIRust-only fields absent from the baseline template are explicitly recorded as not applicable; every other required field must exist. Point bytes, card values, finite/pole settings, QP-integrand choices, independent stability checks and rescue policy remain exactly as declared in proposal1.

## Scientific interpretation and source validation

Every absolute ELF mapping must match its preflight byte hash and mapped inode/device. Deleted, unknown or changed mappings are fatal. The sole allowed non-ELF file class is immutable Nix `lib/locale/locale-archive` mapped read-only; its header, size, hash, device and inode are retained. At least one complete expected runtime map is required. No observer thread polls or reaps the worker.

Only complete parser-validated streams, nonnegative finite accuracy at most `1e-6`, checked backend statuses, no `##W03`, clean outer execution and admitted runtime maps may enter the existing `3e-6` comparison. Fixed-QP status 120 is not accepted. Raw logs and partial/nonfinite outputs are retained even when no coefficient comparison is allowed. A checked status or reducer agreement is not an independent physical oracle. In particular, the historical EW analytic pole evidence used its recorded repaired geometry; this unchanged driver does not capture the current repaired point, so that oracle remains **not requalified**. There are no performance results or stress-acceptance inheritance in this packet.

`test_supervisor.py` contains 22 focused source/parser/mocked-process tests, all passed in `tests-attempt2.log`. `tests-attempt1.log` preserves one initial mock-clock exhaustion failure; its source is retained in `draft-history/`. No real child was launched by these tests. `source-validation.json` rechecks the 99 proposal inputs, 27 installed dependency files and literal card rendering against both source templates. The independent review records the initial actionable findings and their resolved source hashes.

Recheck source tests with:

```sh
python -B docs/benchmarkfolder/experiments/physical-stress-current-supervisor-1/test_supervisor.py
```

`source-index.json` lists every retained payload by size and SHA256, excluding itself. Do not modify this packet after sealing; retain any subsequent correction in an additive successor.
