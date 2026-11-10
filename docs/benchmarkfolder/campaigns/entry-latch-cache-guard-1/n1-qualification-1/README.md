# Two-case n1 numerical qualification

This is a supervision adapter for the existing benchmark CLI. It admits only
the main QCD `u u~ > z [virt=QCD]` and EW `u d~ > w+ [virt=QED]` cases in
`entry-latch-cache-guard-1`. It does not change physics, workers, numerical
parsers, cards, model pins, points or precision policy. No timing is authorized.

From the repository root, after `source tools/tirust_env.sh`:

```sh
python3 -B docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/n1-qualification-1/run.py
python3 -B docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/n1-qualification-1/run.py --preflight
```

Both commands are source-only and create no execution attempt. Preflight hashes
the actual registered source, native installation, frozen UFO model, point and
card inputs, predecessor baseline source identity and configurations. It does
not spawn CLI, compiler or reducer processes or activate campaign discovery.

Only after explicit coordinator admission:

```sh
python3 -B docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/n1-qualification-1/run.py --execute
```

The immutable manifest is
`4a27dda7a0977cd1b0ed0d779b30c403da50968bf494aa96d31cf18fc9fe2076`.
Its generator source is production `0fcd2cd873`, with unchanged installed eb3d
TIRust. Predecessor and historical failures remain in their original namespaces.

The five serialized stages use existing CLI commands:

| Stage | Bound including cleanup | Work |
|---|---:|---|
| `reuse-baseline` | 600 seconds | Private copies from the immediate predecessor after exact object relink proof |
| `generate-candidate` | 900 seconds | Fresh actual MG CLI exports for both cases |
| `build-both` | 1200 seconds | Baseline worker-only relinks; full fresh candidate MODEL/DHELAS/process builds |
| `numerical-qcd_n1_uux_z` | 1800 seconds | Baseline CutTools plus candidate 1/6/7/8/9 |
| `numerical-ew_n1_udx_wp` | 1800 seconds | Baseline CutTools plus candidate 1/6/7/8/9 |

The final 60 seconds of every bound are reserved for cleanup. The numerical
stage bound includes mutex waiting and pre/postchecks. It is a separate
administrative abort limit: the manifest's 3600-second per-worker timeout is
unchanged. An interrupted or incomplete 100-point traversal cannot pass.

Build affinity is nonexclusive CPUs 12–15, four jobs. Numerical workers use
nonexclusive CPU 12 and the existing shared backend8/9 mutex. No CPU is
reserved. Generation uses `PYTHONHASHSEED=0` and `set --no_save`. Every actual
physical card must match inherited hashes; both materialized default chains
remain `8|6|7|1`. The numerical lanes explicitly select their single backend.
Mandatory `check --masses` setup during build exits before amplitudes.

The full100 warmup-only route retains the complete final steady traversal,
coefficients, statuses, final accepted DP/QP counts, refusals and 1e-8 target
misses. The 3e-6 regression criterion, strict 1e-6 stability setting, full finite
terms and UV/IR poles are inherited unchanged. No extra timed `RUN` command is
issued. FPE flags and internal attempts are not claimed observed.

The adapter imports exact hash-pinned historical `StageDeadline`, `OwnedChild`
and `SignalRequests` implementations. It also calls the pinned numerical
supervisor's baseline-proof verifier and complete raw-result replay directly;
neither historical `main()` runs. Only namespace constants and stage deadline
parameters are adapted. Source/configuration/card/build/runtime drift, numerical
failure or cleanup uncertainty stops all later admissions. There is no retry.
Cleanup ownership is the reviewed helper's observed descendant/session/start
identity scope, not a claim about unobserved processes between samples.

`specification.json`, per-stage console/configuration/card/build/replay evidence
and `status.json` are append-only execution evidence. Only small administrative
failure status may be published after an exhausted deadline; no new hashing,
copying, replay or native work occurs then. A successful result qualifies these
two numerical cases only, never the full eleven-case matrix or performance.
