# Current-source subprocess/cache lifecycle replay — source proposal

This packet prepares a fresh replay of the successful `subprocess-cache-lifecycle/attempt5` protocol for `invariant-symmetry-1`. It contains no new generation, build, library loading, amplitude evaluation or performance result. The default command only prints the plan. The coordinator must admit each `--execute` stage separately.

This is source successor2. Proposal1 and its failed preparation remain unchanged. The first actual build rejected an unbound GCC runtime linker script before any loader/native step. Its complete portable failure packet is `../subprocess-cache-lifecycle-current-failed-preparation-1/`, index `22deeb09e727fb4f687e3c78c2bc215afd86cca81a933b74cf259597453c296b`.

The sole build-policy correction binds the pinned compiler wrapper's declarative link flags and resolves explicit plus the already-used GNU Fortran implicit library roots against the declared, compiler and wrapper search directories. A restricted parser recursively follows `INPUT`, `GROUP` and `AS_NEEDED` members, including relative SONAME files and `-l` archives; `OUTPUT_FORMAT` and absolute `SEARCH_DIR` metadata are supported. Unknown syntax, missing members and changed inputs fail before linking. All candidate paths are hashed before link; multiple explicitly declared directories may provide the same name. The unchanged actual-map policy rejects any selected input absent from those pre-bound bytes. No actual-map entry is added retrospectively and no scientific source/card/call changes occur.

`linker-prebinding.json` will retain wrapper sources, search paths, requested roots, transitive script edges and exact hashes before the first lifecycle link. Its digest and all bound input hashes are checked again at runtime admission. `linker-diagnosis.json` is an explicitly retrospective read-only reconstruction from the already closed failed map and retained compiler query answers: the successor covers all five previously omitted selected files (libgcc script, SONAME, archive, quadmath and C++ runtime). This offline diagnosis neither links nor qualifies an executable. `diagnosis-attempt1.json` retains the initially incomplete offline root-list observation and its correction.

The full fresh CLI/materialize/build stages must run in new `-2` namespaces; no executable or object from failed preparation1 is reused. The existing external qualified libraries remain the same pinned inputs. Source tests include missing transitive dependencies, unsupported scripts, changed/unknown selected inputs, wrapper flags, implicit Fortran roots and immutable closure admission. Native scope remains held pending separate coordinator admission.

The registered campaign manifest is SHA256 `ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0`. The private MG source comes from commit `2c13246ffa63aa357ad1e29e5e79aff63d88dc7e`; its registered generator inventory is `a90f80545696a648dbcfdfb8de73b1e319364edf2b93d02d4458c7b9656b8b19`. Every registered generator/numerical source is checked against the live repository before and after stages. The explicit native prefix remains `.tirust-build/qp-transport-install/mg/HEPTools/tirust`, with DSO `eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9`, wrapper archive `0c86349a4fd0dadd918476731e899529ce1c7b17f5892a4af2c22dd762250c1a`, and metadata `25182cfb31e68f8ff6246717e6819f0ec86232de88742f6e47e73676e6a41af9`. The full manifest and installation inventory are checked, including the compiler module/interface/header. Hashing a file here does not load it.

The actual CLI recipe generates independent `u u~ > z [virt=QCD] @101` and `d d~ > z [virt=QCD] @202` subprocesses using `loop_sm`, scalar optimized output and external dependencies. It must produce distinct prefixes with matching tensor capacities. The output is entirely new. Model, DHELAS and process numerical objects are rebuilt from their generated sources; no historical generated executable or numerical object is reused. Previously qualified external reducer archives are copied or linked with before/after hashes. The build records effective compiler flags, exact link recipes, actual `LOAD` inputs, the ELF `PT_INTERP` loader's `--list` output, and hashes of the linked and mapped files.

The two Fortran drivers, CLI template and numerical analyzer are exact copies of the successful predecessor. `lineage.json` records their committed origins. The new C diff only copies `/proc/self/maps` at observer open/close, outside amplitude and ABI wrappers and inside the existing IEEE-environment invariance witnesses. It is applied to both plain and observed executables. Runtime maps at both boundaries must equal the actual loader's dependency inventory plus the executable. No numerical or cache wrapper changes are introduced.

The card-only smoke must pass before the four physical workers. It verifies nominal → alternate MZ=92 → nominal model reading with the required `ident_card.dat` layout and the reader's 20-character numeric-token limit. It must make zero observed reducer exports. Each plain/observed × cache0/cache−1 worker then runs the unchanged sequence: bounded initialization, 19 physical A/B/A amplitude calls and 24 explicit DP/QP cache-key calls. Cards explicitly select backend9, automatic `CTModeRun=-1`, initialization mode1, `MLStabThres=1e-6`, full finite/pole evaluation, genuine-QP integrand enabled and 256MiB cache budget. The original analyzer checks numerical parity, complete records, context recreation, per-process plan identities, 38 cache boundaries, cache enabled/disabled behavior, model changes, precision/mode/key separation, and observer IEEE/rounding invariance. An omitted-contribution `##W03` warning fails before comparison even if the printed result looks usable.

This remains a narrow real-SM DP physical lifecycle proof. The explicit QP key probes promote their DP inputs exactly; they are not a physical high-rank/QP/EW/poles-only amplitude campaign. A future successful replay supports current-source lifecycle status only in the demonstrated scope. All durations are diagnostic and ineligible for performance acceptance.

Run from the repository root after sourcing the pinned environment:

```sh
source tools/tirust_env.sh
python -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-source-proposal-2/run.py
python -B -m unittest discover -s docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-source-proposal-2 -p 'test_*.py' -v
```

After coordinator admission, run the following commands in order. Stop on the first failure; no command retries or overwrites an existing stage.

```sh
python -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-source-proposal-2/run.py freeze --execute
python -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-source-proposal-2/run.py materialize --execute
python -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-source-proposal-2/run.py generate --execute
python -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-source-proposal-2/run.py prepare-build --execute
python -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-source-proposal-2/run.py build --execute
python -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-source-proposal-2/run.py smoke --execute
python -B docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-source-proposal-2/run.py run --execute
```

All children use CPU12 nonexclusively, one compiler job and one numerical thread. Each preparation/build stage has an inclusive 600s cap; smoke has 30s; the four-worker stage has 750s with inclusive 180s per worker. Ten seconds are reserved for owned-child cleanup. Child session/descendant ownership, deferred signals, terminate/kill escalation, logs and failures use the reviewed `OwnedChild`/`SignalRequests` helper. SIGALRM is not blocked across child creation. Smoke/run acquire `.tirust-build/numerical-tests.lock`; waiting for that scheduling lock starts no child and is outside the execution deadline. No CPU reservation or acceptance timing is requested.

Fresh work goes only to `.tirust-build/subprocess-cache-lifecycle-current-2/`. Execution evidence goes only to `docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-qualification-2/`. The protocol, rendered drivers, generated sources, configuration, numerical objects, libraries and runtime input cards are bound before/after their relevant stages. Scientific outputs and failed attempts are append-only. Production files, the registered campaign, its formal inputs/results, old lifecycle evidence and the canonical report are not modified.

The source tests exercise parser rejection, exact copied provenance, rendered-source mutation, stage query separation, nested deadlines, cleanup failure retention, default no-execution behavior, warning completeness, alternate-card tokens, actual-loader selection and the closed predecessor smoke records. Synthetic parser inputs are tests, not measurements. The independent review found and then verified fixes for the initial query-name collision and omitted rendered-driver checks; its exact scope and source hash are retained in `audit.json`.

The separate report-refresh proposal remains at `../report-current-source-refresh-proposal-1/`. The parent has archived the prior 19-page report at `../../history/report-before-invariant-symmetry-qualification/`, manifest `3ac7e02340a1148f9c22d9d7cec2e0b81493c429d9b3ac264e616ff2bb5d9dec`, verifying all 4,294 inputs against commit `584dae7875`. This replay does not alter that archive or transfer its older timing claims to the new source.
