# Draft full-matrix numerical qualification

No numerical job has been authorized or launched by this draft. The coordinator
must review the completed source-preparation evidence before using `--execute`.
Without that flag the supervisor only prints its eleven commands.

The unchanged `tools/tirust_benchmark_numerical.py --warmup-only` route runs each
case on baseline CutTools and candidate backends 1, 6, 7, 8 and 9. It retains the
final complete steady 100-point traversal and introduces no timed loop or extra
classification traversal. The supervisor schedules the nine lower-multiplicity
cases first, then QCD n4 and EW n4. It retains every failed case and continues
through the full matrix unless source, configuration or protocol drift prevents
safe continuation. Neither passing numerical preparation nor this supervisor
authorizes timing automatically.

The frozen manifest supplies each worker's 3,600/14,400-second administrative
limit. Exact source/executable/card/library/point identities are checked by the
existing worker protocol. Per-case observations include accepted return-status
and backend/precision counts. **FPE flags and internal retry counts are not
observed by this worker** and remain null in the supervisor summary. They must
not be inferred from a DP return status.

Run only after coordinator clearance, from the repository root:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/campaigns/split-dp-terminal-1/numerical-qualification-1/run.py --cpu 9 --execute
```

One supervised command runs at a time. The coordinator excludes other native
work; the existing common lock remains mandatory inside backend 8/9 workers.
An outer acquisition of that same lock would deadlock the unchanged runner and
is deliberately absent. CPU, source/protocol hashes, machine state, commands,
inputs and registered deadlines are recorded before the first numerical child.
Each case has a separate console log and the existing append-only raw evidence.

SIGINT/SIGTERM requests are deferred until the controller handle is recorded.
On interruption the supervisor first sends SIGINT to the Python controller so
its Worker/ExitStack can close the detached native worker. After a bounded grace
period it can terminate only process groups observed in that controller's
descendant tree, checking PID, process-start time and session identity before
each signal. Any observed survivors are retained as a cleanup failure and block
further admission. This is an observed-descendant guarantee, not proof against
an unobserved child spawned and reparented between samples.

`test_process_cleanup.py` uses only fake Python controllers and sleeping detached
workers. Its startup/wait tests interrupt after those processes are observed;
they do not claim an empirical test of the Popen-assignment race. The deferred
signal handler addresses that assignment window by recording requests without
raising inside Popen. The remaining tests exercise forced fallback and refusal
to signal a process group whose recorded start identity no longer matches.
