# Fresh candidate exports for policy v2

`regenerate.py` generates all eight core processes and all five processes in
`controls.json` through the actual `bin/madgraph` CLI, then builds their standalone
checks. Its output root is
`.benchmark-work/policy-v2/generation/outputs/candidate`; it does not replace the
original benchmark outputs or the earlier candidate outputs.

Run from the repository root in a checkout without this generation namespace:

```sh
source .ninjars-build/dev-env.sh
python docs/benchmarkfolder/policy-v2/generation/regenerate.py \
  > docs/benchmarkfolder/policy-v2/generation/run.log 2>&1
```

The script refuses to overwrite an existing series. It uses two independent CLI
jobs, each with four build workers. Generation and build durations are operational
measurements under concurrent work, not performance results. The compilation
duration describes the clean subprocess build; generation also includes the
exporter's Source-library construction.

`status.json` records every case, including any failure, its generation-time git
revision and complete tracked generator identity, command/build metadata, masses,
physical answer dimensions, and versioned GNU binary128 runtime imports.
`summary.json` is a compact index produced by `summarize.py` after completion.
`commands/` and `logs/` retain the actual CLI cards and generation/build logs.
`inputs/candidate/` retains the frozen harness's compiler, source, card, and
executable manifests.

For each case, the generated parameter card must equal the untouched original
card byte for byte. The original point stream and its metadata are copied, and
their count, hash and external masses are checked against the new executable.
The only execution is `check --masses`, which exits before calling a loop
amplitude. This series performs no numerical benchmark, warm-up, reduction or
Symbolica-worker initialization.

`model-deltas/<case>/manifest.json` covers every generated `Source/MODEL` Fortran
source and include. The exact changed preimages, postimages and unified patches
are retained beside it. Only `model_functions.f` and
`mp_intparam_definition.inc` may differ: QCD exports change the first file;
electroweak exports change both. `manifests/` and
`protected-inputs-before.json` retain the complete generated source inventories,
including the separate MadLoop/Fortran optimization changes outside MODEL.
`summary.json` enumerates those additional changes without treating them as
part of the model correction.

The script validates the installed NinjaRS ABI/QP/link metadata and hashes the
installed numerical libraries and tracked bridge sources before and after the
series. It also runs
`python tools/ninjars_setup_references.py --verify-model-only`, which verifies the
retained top-level UFO Python inventory and its original content digest without
repinning. Full original and earlier candidate source inventories, together with
every file under the existing top-level report `inputs/`, `data/` and `points/`,
must remain unchanged at the end.

The copied configuration files document the original process and input
definitions. They are generation inputs, not policy-v2 acceptance settings:
the separate v2 collection configuration applies its reviewed common stability
settings to all source roles.
