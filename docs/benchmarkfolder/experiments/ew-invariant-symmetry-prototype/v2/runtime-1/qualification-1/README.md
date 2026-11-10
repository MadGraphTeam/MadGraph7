# Qualified full100 EW revision-2 replay

The exact frozen EW n3 stream completed with 100 accepted DP results (status
219), compared with 82 accepted DP and 18 QP results in the original replay.
All six reference comparisons passed the recorded comparator at both 3e-6 and
the ordinary 1e-8 target, including its existing small-pole criterion. All raw
input words matched. Previously accepted DP answer and accuracy words were
exactly preserved. The 101-call public trace includes bootstrap call zero and
all 100 points; it has no IEEE-state differences and no W03 warning lines.

The worker returned zero and was reaped after READY/BYE. Actual process mappings
were captured before STOP and matched the candidate executable and dependency
hashes, including the installed eb3d bridge. The completed qualification records
unchanged protected inputs. This result applies to this candidate and this EW
stream. It does not update the older 11-case campaign, qualify newly generated
production exports, or establish a timing/performance result.

All actual runtime text files are retained, including points, cards/resources,
public/raw protocol, stdout, warmup, maps, execution, qualification and detailed
comparison results. The original stream comes from its existing frozen portable
archive. All six references and the exact parser sources are retained too.
Deterministic gzip payloads have raw and compressed byte hashes/sizes; binaries,
libraries, caches and core dumps are not copied. Build and component evidence
remain separately indexed and bound.

Default verification checks the portable byte inventory and execution joins.
`--reparse` materializes these exact retained sources and inputs in a temporary
repository layout, then calls the frozen runtime `analyze()` with its frozen
public parser and benchmark comparison functions. No source or import text is
rewritten, and source/input hashes are checked afterward. The helper supplied
to `analyze()` exposes only its parser factory, with no worker-launch methods.
The replay must reproduce `results.json` exactly. Neither verification mode
launches a compiler, loader, reducer or worker. Repository bindings use retained
source/report paths; ignored build trees and installed binaries are unnecessary.

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/ew-invariant-symmetry-prototype/v2/runtime-1/qualification-1/package.py
python3 -I -B docs/benchmarkfolder/experiments/ew-invariant-symmetry-prototype/v2/runtime-1/qualification-1/package.py --reparse
```

`--package` retains the completed attempt once and checks current protected
identities before/after retention. Failed retention is preserved without an
automatic retry. Frozen runtime/source evidence remains unchanged.

Results SHA-256: `803e76b3d767e0a108a9056b1251edc6769b7abf3c27816cf05ce9c58cab6d55`.
Qualification SHA-256: `95ef387372667b0aaa18e9399abdf0ed17759d3a937cddc0beafbf422cc1fb7d`.
