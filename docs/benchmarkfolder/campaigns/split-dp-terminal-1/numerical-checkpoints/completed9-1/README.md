# Nine completed ordinary numerical cases

This append-only checkpoint covers the nine closed cases preceding the two n4
cases in the registered numerical schedule. All 54 lanes contain the complete
final steady 100-point warmup traversal and pass the existing 3e-6 regression
criterion against baseline CutTools. The stricter 1e-8 target misses remain
explicitly recorded; a cross-backend discrepancy alone does not establish truth.

The index binds every file in those nine completed raw case directories, their
closed console logs, static supervisor specification and source, manifest,
configuration, exact point/card/build provenance and the preparation index.
`closed-cases.json` is a snapshot of only those nine completed status entries.
The mutable supervisor status and active QCD n4/EW n4 raw directories are
intentionally outside this checkpoint. No active artifact was changed.

`verify.py` reparses every retained warmup stream with the original hash-bound
parser and recomputes every comparison. It checks before/after runtime, cards,
points and executable identities against retained compilation records, without
repeating broad live build hashing, running a compiler or calling a reducer.
Precision counts describe the final accepted amplitude return. FPE flags and
internal retry counts were not observed and remain null.

This is partial ordinary numerical qualification: nine of eleven cases. It is
not a full-matrix result, timing evidence, stress qualification or goal completion.
All observation seconds remain null and no warmup time is promoted to a benchmark.

From the repository root:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/campaigns/split-dp-terminal-1/numerical-checkpoints/completed9-1/verify.py
```
