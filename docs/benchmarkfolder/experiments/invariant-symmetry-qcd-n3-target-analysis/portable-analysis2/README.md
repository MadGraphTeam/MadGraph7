# Portable closed QCD n3 analysis

This additive capsule reproduces the sealed `attempt1` analysis without a
checkout, campaign directory, installed library, or native execution. It
contains the **exact seven closed input byte streams** required by that
analysis, compressed with gzip level 9, an empty embedded filename, and
`mtime=0`. The current closed result is included even though the full campaign
capsule has not yet been published. No active four-body result or mutable
campaign summary is included.

The result is unchanged: all registered `3e-6` checks pass; the same 28 points
miss `1e-8` exclusively in the finite coefficient. The largest CutTools/TIRust
symmetric difference is `2.806420647e-6` at point 93; the largest non-CutTools
difference is `8.595156396e-8` at point 39. All 1,200 accepted rows match the
previous closed campaign exactly. Reducer correlations and historical
unchecked status-120 diagnostics are not independent finite-term truth.

## Reproduce from any directory

With Python 3.9 or newer, copy this entire directory and run:

```sh
python3 -B /path/to/portable-analysis2/verify.py
python3 -B /path/to/portable-analysis2/relocation_check.py
```

The first command checks every indexed payload, proves the reader-only source
change, decompresses and verifies the seven inputs, and reproduces the exact
stored `analysis.json`. The second makes a clean temporary copy, starts
isolated Python, and installs a read-only audit guard before loading the
portable reader/analyzer. Every file open outside the copied directory is
forbidden, including the original workspace; write opens are also forbidden.
Two deliberate external-read probes must be blocked, and replay must then
complete without further denied reads. It does not run a compiler, reducer,
loader or network request. `verification.json` records that check and the
unchanged result hash. The check can be repeated after relocation.

## Exact lineage and bounds

The original attempt is unchanged. Its index is copied as
`attempt1-index.json`, SHA-256
`7f8661e635664374023e057dedf45b6232c13407d01f62ee10605e16bbebdcd0`.
`attempt1-analyze.py` and `attempt1-README.md` are exact copies bound by that
index; `analysis.json` is the original analysis bytes, SHA-256
`e0dcc8460091021db3089f56b181d4b184b4f11df1302fd14b4a383bd1568778`.
The copied original analyzer is inert provenance; replay executes `analyze.py`.

`reader-only.patch` records the complete analyzer change: import the portable
reader, remove the repository-root lookup, and replace the three-line
repository file read with `read_input(HERE, name, PINS)`. `verify.py` requires
the adapted source and patch to equal that exact transformation. No numerical,
statistical, status, comparison or serialization code changes. The result's
original source names remain historical identifiers, never filesystem paths
opened by the portable reader.

`input-map.json` maps those seven identifiers to unique simple relative gzip
filenames. Both compressed and uncompressed sizes/hashes are checked. The
reader rejects symlinks, unindexed files, duplicate JSON keys, missing/extra
inputs, unsafe names, concatenated/truncated gzip members and decoded length
mismatches. Decompression has an exact expected-output bound and a 64 MiB
maximum per input. `index.json` binds every regular payload and rejects extra
files. The index itself is the external publication hash supplied on handoff.

This capsule retains the complete scope and limitations of `attempt1`.
Historical QP provenance is retained context, not newly requalified. The
separate `qp-specification1` remains a source-only proposal: no part of its
16-worker scope is authorized or implemented here.
