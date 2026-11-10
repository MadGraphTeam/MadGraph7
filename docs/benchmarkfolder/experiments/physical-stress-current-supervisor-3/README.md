# Stress supervisor3: create the export parent

This is an additive mechanical successor to supervisor2 (index `d9563f387845c889390c3111ba3589028e1b1617ddd7156703b25d2eb103e6d0`). Source2 and its closed failed preparation remain unchanged. The failed attempt is retained portably at `../physical-stress-current-preparation-failure-2/`, index `8b26c2a8b7817dc6378dde69b212c6a94b17c1d7de60087b03e74574929e9748`.

Preparation2 proved that tracked bin/madgraph dispatches the command script and generates the process. Export then failed because the frozen exporter uses os.mkdir(output) and the supervisor had not created output.parent. MG caught the exception and returned0. This failure is independent of --plain, which only changes output coloring. The unchanged local benchmark generator already creates its output parent, so the defect was isolated to this private supervisor.

Successor3 asserts that the output path does not exist (including dangling links), then creates only its parents before CLI invocation. The CLI still creates its output directory. It additionally retains and rejects the known fatal console markers `Error detected`, `interrupted in sub-command`, and `Traceback (most recent call last)`, and requires the expected card, process sources/makefile, make_opts and MODEL/DHELAS layout before any driver replacement or compilation. The full diff is successor.patch.

Only those mechanical admission checks and fresh work/evidence namespace3 differ from source2. The eight CLI command scripts change only their private output/model namespace from2 to3. The native worker, six ownership/linker helpers, make targets, physical points/cards, model/coupling settings, strict host checks, selected reducers, precision/rescue policy and bounds are unchanged. All original failures and seals remain intact.

From `/shared/external/madloop_tirust/madgraph7_madloop_tirust`, after `source tools/tirust_env.sh`:

```sh
python -B docs/benchmarkfolder/experiments/physical-stress-current-supervisor-3/run.py
python -B docs/benchmarkfolder/experiments/physical-stress-current-supervisor-3/run.py prepare --execute
```

Default is dry-run. Root alone schedules actual preparation, in fresh `.tirust-build/physical-stress-current-3` and `experiments/physical-stress-current-qualification-3`. Its scope remains8actual CLI exports/full object builds, nonexclusive CPUs12–15/jobs4, inclusive3600seconds and nested600secondCLI/buildcaps. No mass reader/ELF loader/amplitude is run during preparation. Separate `preflight --execute` and `run --execute` need later coordinator admission and prior stage success. Native bounds/commonlock/status/W03/runtime-map gates and no-current-EW-geometry-oracle caveat from supervisor1/2 apply unchanged. There is no scientific retry or performance claim in this source packet.

All29 tests passed. New tests reproduce actual temporary-directory os.mkdir failure with an absent parent, prove the correction creates only the parent and permits the exporter mkdir, reject existing outputs, reject the three fatal diagnostics even with a nominal zero exit, and require complete generated-file layout. The previous actual read-only Git entrypoint checks and exact scientific-command/namespace comparisons remain. Two Git blob reads are the only real external commands in this test suite; no CLI, compiler, loader or amplitude ran. Independent narrow source review passed. The unchanged MAKE_ADDENDUM remains covered by supervisor1's real GNU make fixture.

This packet is sealed before any successor3 execution. Do not modify it after sealing; preserve later outcomes separately.
