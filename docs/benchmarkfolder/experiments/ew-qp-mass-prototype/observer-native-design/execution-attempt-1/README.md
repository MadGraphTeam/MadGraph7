The direct Fortran build succeeded, but all four authorized fixed-QP diagnostic processes stopped during initial model-card loading. There are no observed amplitude or tensor results from this attempt. Each executable returned zero; the runner correctly classified the missing observer outputs as incomplete.

`build.json` records 16 successful compiler/link commands and 18 output hashes. The sum of command durations was 12.00636942544952 seconds under the 180-second build bound. The original and candidate executables were distinct private outputs; no installed library or production source was changed. `run.json` records original/candidate × backend 8/9, CPU 8, the 120-second per-process and 480-second total bounds, exact loaded library identities, and the single-thread environment. These startup durations are not performance measurements.

All four `stdout.log` files contain the same missing `/ident_card.dat` warning. The retained generated reader in `../../sources/reference-03-lha_read.f` first directly opens the requested card (line 451), derives its directory by stripping the basename (lines 454–455), and forms the identification-card path (line 149). The bare `param_card.dat` argument produces an empty directory and therefore `/ident_card.dat`. Its missing-file branch issues a bare `STOP` (line 480). The frozen driver calls `SETPARA` at line 10, before opening `observer-run.log` at line 24 and before any amplitude call. The identification card was present under `MadLoop5_resources`, beside an identical copy of the physical parameter card, but that directory was not supplied to the reader.

`raw/` preserves every small runtime file, including the empty generated `param.log` files. `failure-classification.json` records the setup-failure classification, and `offline-analysis.json` is the unchanged frozen analyzer's rejection of the incomplete run. This is not a native reducer refusal, successful numerical parity, checked-QP result, or threshold-accuracy qualification. Fixed `CTModeRun=4` remains an unchecked transport diagnostic even if a future attempt completes.

Commands consumed exactly once, from the MadGraph repository root after sourcing `tools/tirust_env.sh`:

```sh
python -B docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/build.py --execute-reviewed-source a5b646275e03fce2815dfd13cb603a6d4223926db5c57d175cf7f7b98ac5cc1d
python -B docs/benchmarkfolder/experiments/ew-qp-mass-prototype/observer-native-design/run.py --execute-reviewed-source a5b646275e03fce2815dfd13cb603a6d4223926db5c57d175cf7f7b98ac5cc1d
```

The accompanying source index remains the unchanged source-only snapshot; this appended execution index records what subsequently happened. `verify.py` checks this package independently, including every raw runtime hash and the incomplete outcome. Native binaries and libraries are represented by hashes, not copied into the report. No automatic retry was performed.
