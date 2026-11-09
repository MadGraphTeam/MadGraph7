# Baseline object replay qualification

The original QCD 2→1 baseline worker's 18 process object files were relinked
without recompilation, using the original library paths and unchanged old
driver. The temporary executable reproduced its recorded SHA-256 exactly:

`2b3d88c1d350c0cd68f13242468b5cf7228d31399d4d2605a707c9b4a79e6b4d`

The original executable, source inventory, process objects, archives and runtime
libraries remained unchanged. The operation was pinned to CPU 9; it ran only a
linker, with no amplitude evaluation or runtime measurement. The executable is
kept in the ignored `.benchmark-work/tirust/reuse-proof-qualification/` tree.

`qcd-n1/proof.json` preserves the exact make dry-run command, original link
command, redirected proof command, object hashes and before/after executable
hashes. `make.log`, `link.log`, the exact helper source and `qualification.json`
retain the build and dependency evidence. Run the helper through the sourced
benchmark environment with a new empty destination to reproduce the check;
the helper refuses numerical compilation and requires byte-identical output.

This qualifies the proof mechanism on one real export. Each later baseline
reuse must pass its own proof before those objects are copied into a campaign.
