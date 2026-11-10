# Private public-API DP failure diagnostics: source proposal

This packet proposes a private build and, after its post-link review, one
twelve-call replay of six already captured Zgg requests. It contains no new
native result. Production code, installed libraries, reference repositories,
old captures and their original failure status remain unchanged.

The closed predecessor is `../closed-2/index.json`, SHA256
`1e26738f4489b55e85ac059c963fec0c75dbc33f8cb4bd208f59a8676477cc06`.
Six Zgg inputs were captured, and six ZZgg inputs remain missing. The latter
are excluded from this diagnostic and are never reconstructed. Each frozen
Zgg input is evaluated twice, in its own newly created context, for twelve
DP calls total. All twelve prior calls returned status 1 without reading
their failed tensor buffers.

The question is why embedded method choice also refused these requests.
Pinned TIRust `dispatch/embedded.rs` appends a failed method-choice status to
`last_diagnostics` while retaining the original primary failure in the
returned lane status. Consequently the unchanged EB3 boundary message can
report the initial Gram refusal while the fallback failed for another reason.
This is a source finding, not a newly measured fallback diagnosis.

## The sole bridge addition

`bridge_patch.py` derives `private-lib.rs` from the exact installed bridge
source `455529b9…`. `restore(candidate)` reproduces every original byte.
The patch resets one private TLS string at DP entry, then copies public
metadata immediately after the existing full-DP `reduce_batch` returns,
before the unchanged status mapping and failed-workspace removal. It records
the actual returned status/flags, workspace precision, default embedded
options, lane routes, engine execution summary and `last_diagnostics`.
It never reads tensor values or a best-effort failed result. Early `Err`
metadata is explicitly labelled as potentially partial or previous.

The unique private export is:

```c
size_t ml_tirust_private_dp_diagnostics_v1(unsigned char *out, size_t capacity);
```

It copies owned TLS bytes, returns required bytes including NUL, and does
not call the regular guard or alter `LAST_ERROR`. The snapshot has a 65536
byte ceiling and an explicit truncation marker; the replay rejects missing,
stale or truncated snapshots. Rust Debug text is tied to these exact source
pins; this is not a new supported numerical ABI.

Existing public APIs do not expose all per-subset variant masks, condition
estimates or recurrence operands. This first stage adds no upstream observer.
The frozen request words are retained in `bindings.json`; they are request
operands and must not be described as observed recurrence operands. If the
public metadata is insufficient, any deeper instrumentation requires a
separate reviewed proposal.

## Source, build and cache identities

`bindings.json` pins the six input files and all their integer words,
unchanged original C source, installed bridge source, resolved Cargo lock,
original installation metadata, checked poles-only patch, source tree digests,
tool bytes, 148 existing Cargo source-cache roots (13222 files), and the
original fourteen runtime providers. The pristine TIRust tree contains 463
files and has digest `5b60209f…`; the fourteen-file pole-kernel tree has
digest `2d449d05…`. These exactly match the unchanged EB3 installation receipt.
The full dependency revisions are TIRust
`854b79dcb7fffcb35e92b4a8a3dd443369b798c4`, oneloopmaster
`bd703d09a78f7849a6b86a07096e6f222fc01193`, and Symbolica/Numerica
`dfe6bd97b974b1028dc63071ea165136cd2b0ef5`.

The private Cargo manifest changes only `[lib].name` to
`madloop_tirust_private_diag`. Package name/version, release profile,
dependency revisions and features remain unchanged. The installed resolved
lock `c4f59f29…` must stay byte-identical. The private source copy receives
the existing exact poles-only patch and generated kernels, preserving the
installed full/poles/cache implementations. No installed or reference path
is a build destination. No numerical option, guard, precision or cache
policy is changed.

`build.py build` is the proposed **build-only** entry point, requiring root
admission before invocation. It exclusively creates
`.tirust-build/tirust-private-diagnostics-1`, refuses an existing destination,
checks source/tool/runtime pins, uses nonexclusive CPUs 16–17 only after
checking affinity, limits each build process to 4 GiB and all commands to a
shared 900-second wall deadline, and retains command logs and terminal status.
It uses a fresh private Cargo target/build cache. Existing downloaded source
caches are read under their frozen tree identities, with complete source-file
inventories before/after; an offline cache miss or drift is a retained failure.
No network fallback, dependency update or automatic retry is permitted.

Cargo is invoked with `build --locked --offline --release --features poles-only`.
The C compiler keeps `-std=c11 -O2 -fPIC`; the original Fortran link driver
and link flags/libraries are reused, replacing only the bridge link name/path
and rpath with the private library, and adding a link map. The old static
Fortran wrapper remains in the original link list; the C driver invokes the
bridge directly. Source trees and locked cache trees must match before/after.
New DSO/executable hashes are deliberately `null` until observed after a real
build; `build-terminal.json` records their actual bytes. This controller does
not run the driver, use a loader, import the DSO, or claim runtime qualification.

## Separate native replay admission

After a successful private build, root must review the actual post-link
artifacts, ELF dependencies, symbol resolution and runtime launch controller.
The new uniquely named private DSO replaces EB3 only; every other original
provider must retain its exact path/hash. In particular the original glibc
2.40 interpreter, Fortran, C++, GCC and quadmath providers remain unchanged.
No incompatible or unexpectedly resolved provider can be silently omitted
from the expected set. A private SONAME alias directory, if required, must
be derived from actual ELF metadata and bound by an exact link inventory.
The installed EB3 DSO must never be loaded alongside the private replacement.

The proposed next replay is one process, at most twelve numerical evaluations,
60 seconds including loader checks, 4 GiB, nonexclusive CPU 16 if available,
and no retry. It acquires the existing
`.tirust-build/numerical-tests.lock` nonblockingly for the entire numerical
interval; a busy lock means no launch, not competing numerical execution.
Outer Python/compiler processes have no injected `LD_*`; only the target
exec receives its reviewed environment. A strict separate-process loader trace
must pass before entry, followed by mandatory sampled `/proc` map checks.
Those samples are observational; they are not a pre-entry map barrier or a
race-proof isolation claim. All source/input/executable/dependency/alias
pins are checked immediately before the launch and after its terminal status.

`driver_patch.py` derives `private-replay.c` from the unchanged standalone
driver, retaining its immediate boundary error copy and success-only output
read. It adds only a later private getter copy plus exact six-input selection.
The external launcher must also enforce all six byte hashes in `bindings.json`
before starting; metadata checks inside the C parser are not sufficient to
establish frozen input identity. `protocol.py` requires all thirty lifecycle
records, exact original boundary status/error bytes, fresh/repeat counts,
fresh full-DP snapshots, confirmed context creation/destruction, and no
serialized failed tensor output. Unexpected results are retained and fail
validation; they are not discarded or silently reclassified.

All compiler/build/loader/native logs, statuses, errors, maps, snapshot bytes,
input copies, link maps and source/cache inventories must be retained in an
append-only closed packet, including failures. Later pure replay reconstructs
the diagnostic summary without running code. This small failure diagnostic
does not establish scalar/tensor/amplitude accuracy, resolve the stress/CMS
acceptance gates, qualify QP, or measure runtime acceptance.

## Pure review commands

Run from the required checkout:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/diagnostics-bridge-proposal-1/verify.py
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/diagnostics-bridge-proposal-1/run_pure.py
```

The tests deny child processes, explicit native loading and writes. They
qualify derivation/validation logic only. The separate build command, to be
run only after root admission, is:

```sh
python3 -B docs/benchmarkfolder/experiments/tirust-first-failure-capture/diagnostics-bridge-proposal-1/build.py build
```
