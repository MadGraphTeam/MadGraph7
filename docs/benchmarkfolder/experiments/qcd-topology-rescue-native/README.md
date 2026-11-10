# Private QCD topology and coordinate replay controller

This directory contains the bounded native controller for the independently
reviewed source design in `../qcd-topology-rescue-prototype/`. Its frozen design
index is `0af03d980acdac032abcb767713bf5906fa7fa53cf8b3c57da75b764281e3015`.
The controller is prepared for review; no compilation or replay has occurred at
this source freeze. Execution needs the parent agent's final clearance.

The selected arms are `topology-b`, `original-c`, and `topology-c`, each with
backend 9 and backend 8 once. The existing `original-b` executable, exact
observed outputs, IEEE flags, and rejected stress status 420 remain the control
in `../qcd-loop-kinematics-observer-native/`. There is no redundant control run.
No production or installed-library bytes are modified. This is a diagnostic,
with no performance or physical-acceptance claim.

`common.py` validates the exact 19-file source design, the exact 25-file original
native evidence inventory, the nine named frozen inputs, and the merged
historical build/execution/runtime pins. The union has 220 unique protected
paths before adding this controller's source inventory. It rejects conflicting
hashes and validates files before any compiler or reducer launch.

`build.py` constructs its recipe directly from the pinned original native link
command. All 24 objects retain their order. The seven explicitly declared
observer/adapter/host/driver/Born object paths are replaced by the selected
arm's private compiled objects. A topology arm appends `shell_classes.o` before
the unchanged library arguments. The original compiler, numerical flags,
include directories, and exact library arguments/order are retained; only the
private module output/include directory changes. The Born object's historical
flags are separately checked against the corresponding original recipe.
All compile sources are copied from the frozen arm manifest. No source is
transformed during compilation. No Rust build occurs.

The compiler receives an explicit environment allowlist: the recorded tool
`PATH`, private `TMPDIR`, C locale, and one OpenMP thread. Nix flag hooks,
compiler/include/library search overrides, loader preloads, and arbitrary
inherited variables are absent. The historical builds did not retain their
environment, so historical equality here means recorded command arguments and
protected input bytes, not a claim about unrecorded environment settings.
Every private source/card copy must match its frozen expected hash before each
compiler command and afterward; a self-hashed changed copy cannot qualify.

`plans.json` records all three exact recipes. `--plan-only` is read-only and
does not launch a compiler. The real builder requires equality with that
reviewed recipe and binds `source-index.json`, including the finalized runner,
into its before/after protected-input manifest. Build outputs live only in
`.tirust-build/qcd-topology-rescue-native/attempt1/<arm>/`; portable build logs
and manifests go to `attempt1/<arm>/`. An existing arm directory prevents a
second build. A compile/link failure is retained and stops that invocation.

`run.py` requires a successful immutable arm build and the frozen numerical
card for the selected backend. It uses the existing ordinary/stress/ordinary
driver and physical input bytes. Each real process is pinned to CPU 11 with
one thread and runs under `.tirust-build/numerical-tests.lock`. A process-group
wall limit of 180 seconds kills all descendants on timeout. There is no retry.
The original installed TIRust DSO is required at its original path with SHA256
`49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12`.
All resolved runtime files and protected source/object/card bytes are checked
before and after execution. The numerical policy remains automatic checked
mode `CTModeRun=-1`, `MLStabThres=1e-6`, `OSThres=1e-8`, existing 256-bit TIRust
working precision and guard, and the original parameter/repair prescriptions
except the explicitly isolated B/C handoff.

The native environment is also a recorded positive allowlist: known library
directories, single-thread controls, C locale, a private temporary directory,
the shared Symbolica lock directory and existing home directory for license
lookup. It excludes inherited loader and Fortran exception-policy overrides.
The complete resolved 15-file runtime map must equal the historical map.

The ordinary first/repeated outputs, statuses and named IEEE records must match
the historical control exactly. Stress outputs and IEEE flags may change as a
result of the intended arithmetic changes; they are retained verbatim and do
not receive an automatic success label. Observer health must be zero at every
point. Full trace decoding, native-status and topology classification checks,
exact C handoff, all nonmatrix inputs, finite outputs, and genuine current-call
checked-QP evidence remain separate analysis gates. Finite Born values or fewer
tensor refusals do not establish a checked physical amplitude. If stress still
returns 420 it remains rejected. A C-arm result is explicitly a changed
effective-coordinate comparison, not an unchanged Cartesian-input claim.

After final clearance, run from the MadGraph repository root with the pinned
environment sourced. Each command is a separate one-shot action; stop and
retain evidence after any failure rather than repeating it:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-native/verify_source.py
PYTHONDONTWRITEBYTECODE=1 python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-native/build.py --arm topology-b
PYTHONDONTWRITEBYTECODE=1 python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-native/build.py --arm original-c
PYTHONDONTWRITEBYTECODE=1 python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-native/build.py --arm topology-c
PYTHONDONTWRITEBYTECODE=1 python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-native/run.py --arm topology-b --backend 9
PYTHONDONTWRITEBYTECODE=1 python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-native/run.py --arm topology-b --backend 8
PYTHONDONTWRITEBYTECODE=1 python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-native/run.py --arm original-c --backend 9
PYTHONDONTWRITEBYTECODE=1 python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-native/run.py --arm original-c --backend 8
PYTHONDONTWRITEBYTECODE=1 python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-native/run.py --arm topology-c --backend 9
PYTHONDONTWRITEBYTECODE=1 python -B docs/benchmarkfolder/experiments/qcd-topology-rescue-native/run.py --arm topology-c --backend 8
```

The original source design's `native_authorized=false` records its historical
source-only stage and is deliberately unchanged. Later execution authorization
and any attempted process are recorded in this separate namespace. No failure
or later source correction overwrites a frozen design or previous attempt.

`check_runner_source.py` exercises the comparison guards using retained text
and integer trace words only. Its four positive cases and 36 rejected mutations
are recorded in `source-checks.json` and reproduced by `verify_source.py`.
These checks launch no compiler, reducer, live `ldd`, or timeout subprocess;
they do not claim to test operating-system timeout or signal delivery.
The process launcher defers SIGINT/SIGTERM/SIGHUP exceptions during process
creation until it owns the child handle, then restores the previous handlers
and kills/reaps the process group on any interruption or timeout.
