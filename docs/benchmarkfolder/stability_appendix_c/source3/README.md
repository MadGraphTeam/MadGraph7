# Raw-DP stability pilot source3

This source-only successor preserves the four processes, five reducers, 100
common cut-qualified points, 6,000 forced-DP probes, numerical policy and limits
of [source2](../source2/README.md). The Fortran worker, point generator and
statistical routines are byte-identical to source2. No source3 preparation,
compiler, loader query, point freezing or amplitude has run.

Source2 preparation 15086 failed before any child launch because it equated
two distinct Git metadata fields. Every actual closed generation receipt
records revision 584dae7875..., whereas the frozen production manifest records
2c13246.... Their complete source-file inventories are identical, with frozen
content hash a90f805.... Source3 pins both exact revisions in their own roles
and joins the entire generation inventory to the manifest; neither revision
is relabelled. Both full-rebuild flags and both exact campaign bindings remain
required. [Failure2 receipts](../failed-preparation-2/README.md) retain all 20
unexecuted requests, separately from the unchanged earlier attempts.

Before sealing, the source3 controller's read-only preflight passed on ALL
FOUR actual closed donors under audit-hook guards forbidding writes, child
processes and dynamic loading. It checks receipt hashes and roles, generated
scientific sources, executable/cards/dependencies/points/masses, alias graphs,
COMMON and rank layouts, rendered driver, compiler hash and donor ELF
interpreter. It launches no compiler, loader or amplitude and creates no
attempt work directory. The [actual proof](actual-readonly-preflight.json) records all four passed cases
and no denied operations. Replay it with `python3 -B verify_actual_donors.py`
from this directory; redirect output to a new file. The 113 pure/mock/source
tests also passed. Mock tests alone do not substitute for actual preflight. Actual compile/loader preparation and
native stability evaluation remain separate root-admitted stages.

```
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/stability_appendix_c/source3/pilot.py prepare --execute
```

Only after a real passed preparation terminal and checkpoint review:

```
python3 -B docs/benchmarkfolder/stability_appendix_c/source3/pilot.py run --execute
```

All writes belong to fresh `.tirust-build/stability-appendix-c-pilot-3`.
Source1/source2, their failed work directories and production files remain
unchanged. The exact two dangling-link guard from source2 remains intact.
