# Private current-source preparation supervisor

Prepared source only. The coordinator owns actual execution. No build, linker,
CLI generation, parameter probe or reducer was launched while preparing this
supervisor. Default invocation prints the exact schedule without activating the
campaign, changing affinity or writing execution evidence:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/run.py
```

After source review and the coordinator's scheduling admission, the exact command
is:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1/run.py --execute
```

Run from the repository root. The immutable campaign manifest is
`ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0`.
The fixed nonexclusive CPU affinity is 12–15 with four compiler jobs. It does not
reserve CPUs or imply a quiet host. Each serialized stage has a 3600-second cap,
with ten seconds reserved inside that allowance for cleanup. These are source
preparation caps; the campaign's per-case numerical worker deadlines are unchanged.

The stages are:

1. `reuse-baseline`: the existing benchmark API proves original baseline object
   relinks reproduce the executable exactly, then copies source and objects.
2. `build-baseline`: the existing API relinks private workers and requires all
   other proven numerical objects to remain unchanged.
3. `generate-candidate`: actual MG CLI exports for all eleven cases from the
   registered current source, with `PYTHONHASHSEED=0` and `set --no_save`.
4. `assert-declared-cards`: all 22 actual card hashes and generation bindings must
   equal the manifest's explicit predecessor physical-input ancestry.
5. `build-candidate`: cards are checked again before compilation; all MODEL,
   DHELAS and process objects are freshly source-built by the existing API.
   Cards, source identity, configuration and baseline/candidate effective compiler
   policies are checked afterward.

The installed reducer archives/DSO are linked without rebuilding Rust. Mandatory
`check --masses` probes execute SETPARA and stop before amplitude initialization.
No amplitude or reducer evaluation is admitted. Build durations are not benchmark
timings. Model pins, cards, manifests, numerical policy, summary, handoff 3 and
shared tools are not modified by this supervisor's source preparation.

`run.py` derives from the frozen old split campaign preparation controller.
`process_cleanup.py` is copied byte-for-byte from its frozen numerical controller.
`OwnedChild` retains observed descendant process/session/start identities;
`SignalRequests` defers INT/TERM across the spawn handoff. ALRM is also deferred
across that handoff, then bounds synchronous verification as well as child waiting.
Every owned child is cleaned up before another stage. Survivor or cleanup errors
fail the run. The helper's ownership scope is observed descendants; it makes no
claim to capture an unobserved process that detaches and disappears between polls.

Execution is append-only: existing outputs/specification reject a second attempt.
All status/events, exact commands, source/configuration snapshots, exit codes and
console hashes remain in this directory. Configuration drift is retained without
repair. Failures stop the sequence. At the deadline, no new unbounded hash capture
begins; any unavailable final snapshot is explicit, and only small administrative
closure may be published afterward. There is no automatic retry or resume.

`source-index.json` freezes this source, the unchanged cleanup helper and protocol
inputs. `test_run.py` uses mocked child execution and temporary files only. The
retained dry-run and tests do not qualify scientific results.
