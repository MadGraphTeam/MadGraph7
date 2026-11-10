# Portable six-arm numerical evidence

The six-arm replay completed with 101 public calls in each arm, exact public
bit-word and IEEE-state parity, validated loaded runtimes and unchanged protected
inputs. All workers exited successfully and the shared numerical lock was
released. This package retains that completed evidence and the required
independent supplemental audit. There is no timing or performance conclusion.

The source is the `cause-2-qualified/controller-native/attempt1` directory under
the private build workspace. All actual regular run records are retained: outer
and per-arm execution metadata, public bit words and IEEE states, warmup, stdout,
native JSONL, protocol bytes, process maps, point/card/resource files, generated
resource outputs and both final analyses. Host events and frame envelopes remain
inside each original native JSONL stream. The original/no-op arm has no native
journal. FIFO endpoints are represented by their exact retained protocol bytes;
temporary/cache directories are excluded explicitly.

Identical raw text is stored once in content-addressed gzip payloads. `index.json`
maps every original relative filename to its payload, raw SHA-256 and raw size,
and separately pins each compressed file. Compression uses an empty filename,
mtime zero and level six. No text, integer word, status, command, environment or
runtime path is reformatted. Executables, libraries, objects, archives and
dependency caches are not copied. The existing completed build package is bound
by its immutable index and payload identities; its large build artifact maps and
prior failed build records are not duplicated.

The compact summary keeps bootstrap separate from the 100 ordinary points and
reports DP/QP acceptance counts, native status/error observations, host accuracy
threshold observations and contraction-rejection counts from the bound analyses.
These observations can coexist; they are not unique-cause assignments.
The QCD engine reports 454 higher-precision attempts, with 320 successes and 134
refusals. These are reported engine counters, not proof of exclusively binary64
arithmetic in other routes. Immediate native error classes are reported separately:
method-selection failure details can differ from an earlier Gram-matrix error.

The package also binds the frozen controller, analyzer, protocol, source snapshot
and original source records through repository-relative paths. The completed
`numerical-audit/index.json`, its script and results are mandatory bindings; the
results are additionally retained as `supplemental/results.json`. An absent or
unfinished audit cannot silently produce a final package. Numerical failure
records remain failure records. The helper does not promote provisional cause
attribution, interpret elapsed time, or make a performance claim.

Default invocation verifies portable bytes, inventory, execution/output/analysis
joins and repository source/build/audit bindings. It does not need the ignored
workspace, native binaries or caches. This is **byte and inventory verification**,
not a fresh semantic replay of the frozen public/native parsers. The semantic
results come from the retained controller analyses and supplemental audit. No
parser is modified or imported with rewritten filesystem functions.

Optional offline semantic rechecking can materialize the inventory into a fresh
private directory under the repository root: for every logical path except `supplemental/`, decompress its
referenced gzip payload to that path, checking the recorded raw size and SHA.
Preserve all files, including `execution.json`, per-arm points/cards/resources and
native streams. Then run the unchanged `python3 -B analyze.py --case CASE --original
MATERIALIZED/CASE/original --disabled MATERIALIZED/CASE/disabled --enabled
MATERIALIZED/CASE/enabled` from the repository. That analyzer also requires the
existing historical `candidate-9/raw/{observation.json,warmup.dat}` evidence for
each case in the registered campaign; those repository files are already bound
by the frozen protocol/source preparation. The supplemental audit's live-workspace
admission wrapper is separate from this optional parser replay. Materialization
and semantic replay are not performed by the default verifier. For this direct
script invocation, omit `-I`: Python must include the analyzer's script directory
to import its unchanged sibling `protocol.py`.

After completion and source review, from the repository cwd:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/dp-rescue-cause-observer/numerical-qualification-1/package.py --package --execution-sha256 c308d6bdf48db24bc428e0c556618e4191fe07535720708f954b62331376f8b4 --audit-index-sha256 a43855eeb3fcee83c99068258e6f49be86bc352eeb8d37b292a801f4140fed06
python3 -I -B docs/benchmarkfolder/experiments/dp-rescue-cause-observer/numerical-qualification-1/package.py
```

The first command refuses an existing index or packaging-start receipt. A failed
copy preserves its partial files and failure receipt; there is no automatic retry
or replacement. The second command is read-only. Authorization remains the root's
explicit instruction; supplying hashes does not authorize a numerical launch.

`preflight-failure-1/` preserves the first helper and its metadata-check failure:
the audit's `size_bytes` field initially was compared with `bytes`. That failure
created no payloads; the corrected helper normalizes the size field while leaving
the frozen audit and numerical records unchanged.
