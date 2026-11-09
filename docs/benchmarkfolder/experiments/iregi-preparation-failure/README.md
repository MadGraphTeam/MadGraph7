# Preserved IREGI preparation failure

`cli-output.txt` records an early baseline export of
`u u~ > z [virt=QCD]` that failed while preparing IREGI, before an amplitude
worker existed. The CLI requested the ignored baseline's IREGI directory, but
the nested make command entered the main checkout's IREGI directory. The
retained compiler error is a missing `global.mod` while compiling
`binary_tree.f90`.

The frozen baseline's wrapper makefile uses the inherited `PWD` variable to
construct this path. `isolated-build-output.txt` preserves the subsequent build
inside the intended baseline tree; it reaches `ranlib libiregi.a`. Later
successful export/build records bind the resulting baseline dependency. The
harness now sets child `PWD` consistently with its actual working directory.
The observed directory mismatch is established by the log; the records do not
independently establish every concurrency condition behind the missing module.

These two logs do not include a complete original environment snapshot or
separate machine-readable exit-status record. They are retained preparation
evidence, not numerical accuracy or performance observations. No coefficients,
phase-space samples or timing acceptance can be inferred from this folder.

Recompute their hashes and source cross-checks with
`python3 docs/benchmarkfolder/experiments/audit_inactive_preparations.py` from the
repository root after sourcing `tools/tirust_env.sh`.
