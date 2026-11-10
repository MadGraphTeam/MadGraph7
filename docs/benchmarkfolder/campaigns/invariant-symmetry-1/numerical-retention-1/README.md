# Portable closed-case numerical retention proposal

This source-only tool targets campaign `invariant-symmetry-1`, manifest
`ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0`,
and numerical supervisor session 63790. Creating or testing this proposal
consumes no live numerical payload. Its default command prints a plan only.

From the repository root:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/campaigns/invariant-symmetry-1/numerical-retention-1/retain.py
python3 -B docs/benchmarkfolder/campaigns/invariant-symmetry-1/numerical-retention-1/test_retain.py
```

A later explicit `seal` command requires a JSON closed-case confirmation with
schema `madloop-tirust-closed-numerical-confirmation-v1`, campaign binding,
`supervisor_session_id: 63790`, the exact supervisor specification SHA-256,
`supervisor_source_index_sha256:
fde535243092336b3988a902a88aaff8223d6d3dfb37cdfe68e7fc4cd152e243`, and
`closed_cases`: the selected, unchanged completed supervisor entries in their
registered order. Entries must have status `passed` or `failed`, an integer
return code and a finish timestamp. A passed entry must carry its successful
raw replay marker. Full eleven-case retention also requires
`terminal_confirmation` with `observed_terminal: true`, `session_id: 63790`
and the actually observed integer `exit_code`. A subset may use null for this
field. A full-matrix passing flag additionally requires that observed exit code
to be zero; eleven passing case entries cannot erase a failed supervisor. The
coordinator creates this confirmation from closed evidence; the
retainer never reads the mutable numerical supervisor `status.json`.

After that concrete admission, the command is:

```sh
python3 -B docs/benchmarkfolder/campaigns/invariant-symmetry-1/numerical-retention-1/retain.py seal \
  --confirmation PATH_TO_CLOSED_CASE_CONFIRMATION.json \
  --destination docs/benchmarkfolder/campaigns/invariant-symmetry-1/numerical-retention-1/sealed-1
```

The destination is created exclusively. Failed seals leave their failure
record; they are never silently overwritten or retried. No real seal is part
of this proposal's source-only tests.

The input allowlist contains only the selected closed case trees and console
logs; immutable campaign/model/source/point metadata; the declared physical-card
ancestry; selected process generation, compilation, card and build-log records;
and source/preparation/supervisor records bound by the numerical specification.
Baseline generation ancestry follows only declared predecessor manifests and
selected baseline generation/compilation/CLI files. These records and any bound
relink proof are retained independently of whether a numerical lane succeeded.
Current installation paths
inside JSON are data, never filesystem inputs. Shared immutable point files
for all eleven cases remain part of the manifest closure, but unselected
numerical experiment directories are neither inspected nor copied. Source
files and closed tree membership are rechecked before publication. Symlinks,
nonregular files, binary/native artifacts and paths outside the allowlist fail
closed. No ELF executable, archive or shared library is packaged.

The package contains a deterministic, content-addressed gzip JSON payload.
Each logical UTF-8 file retains its exact bytes, byte count and SHA-256; repeated
content is stored once. The index separately binds the outer source/summary/
payload files. It lists every actual source input and selected tree. No
newline normalization or decoded/re-encoded JSON replaces the raw evidence.

Offline verification uses only this package:

```sh
python3 -B PATH_TO_PACKAGE/retain.py verify --package PATH_TO_PACKAGE \
  --index-sha256 EXACT_PUBLISHED_INDEX_SHA256
```

`read_retained_json(path)` returns only an indexed member after all payload
hash checks. `replay_completed()` returns the exact reparsed supervisor result
for each selected case, including failed or partial cases. It extracts pure
definitions from the frozen original numerical supervisor and original
warmup/comparator/status parsers into a temporary text-only layout. It also
runs a frozen copy of the existing independent numerical report collector.
Neither replay imports the live campaign harness, executes a compiler or
reducer, nor opens paths from recorded runtime metadata. Relocation and absence
of the original workspace therefore do not change replay.

Both passes preserve the six lanes, the complete final steady 100-point
traversal, every Born/split-order column, initialization copies, READY data,
input/postcheck bindings, actual accepted backend/precision counts and the
existing `3e-6` comparison criterion. The ordinary `1e-8` target remains
separate. A failed controller cannot become qualified merely because complete
raw rows exist. Failure markers and partial results remain in the payload.
An integrity mismatch stops sealing or verification rather than inventing a
successful row.

Partial checkpoint cases outside the selection remain pending. Even a full
passing ordinary matrix makes no timing, stress, performance-acceptance or
overall-goal claim. Warmup `cold_seconds` is retained as initialization data;
timing coefficients remain null. Final accepted DP/QP counts do not enumerate
internal retries. FPE flags and internal retry counts remain unobserved.

`campaign_report_numerical.py`, `numerical_supervisor.py` and the seven
`protocol-*` files are frozen copies, not edits to their originals.
`synthetic_fixture.py` derives from the existing report collector's fixture;
its output is synthetic text used only by the focused tests.
