# Raw-DP stability pilot source4

This successor corrects setup failures in source3's closed raw run. All 20
requests stopped before native execution because independent card readback
rejected the canonical footer `/* End of param file */`. The complete closed
[preparation and failure capsule](../closed-preparation-and-failed-run-3/README.md)
preserves every failure and the 400 frozen physical inputs. No stability
measurement was obtained by that attempt.

Source4 accepts the exact terminal footer and validates end state, rejecting
duplicate headers, malformed or nonterminal footers and unrelated extra
values. The caller declares the precise private card write destinations
before mutation; a writer or readback error therefore retains its primary
failure while the guard still checks all other files.

The passed source3 preparation is inherited explicitly by checkpoint hash,
source seal and protocol. Its four verified master trees, driver executables,
runtime closures and frozen points remain read-only. Inheritance creates a
fresh work4 checkpoint without compiling, loading a library, querying a
loader or regenerating points. Fresh numerical workers are copied from
those masters only during the separately admitted run.

All 20 requests, 100 common points per process, 6,000 forced-DP probes,
controls and limits remain fixed. The declared schedule runs the 12 workers
using backends 1/6/7 first, then the eight workers using 8/9 that require the
shared lock. Both groups retain case and backend order. No observed numerical
result influenced this ordering.

Only root-admitted stages may create the new
`.tirust-build/stability-appendix-c-pilot-4` namespace. Source3 and all prior
attempt evidence remain unchanged. Results are unchecked forced-DP
consistency diagnostics, with effective card-selected backend identity;
they are not independent true-error or precise-runtime measurements.

From the repository directory, after source review:

```
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/stability_appendix_c/source4/pilot.py inherit --execute
```

Only after root verifies the matching inherited checkpoint:

```
python3 -B docs/benchmarkfolder/stability_appendix_c/source4/pilot.py run --execute
```

Before sealing, the exact source4 parser passed all 20 retained card strings,
and guarded read-only admission passed the actual four inherited preparations.
The [actual proof](actual-readonly-inputs.json) records zero forbidden attempts
and an absent work4 directory. All 122 pure/mock/source tests passed.
