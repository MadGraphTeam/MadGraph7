# Bounded private QP profile controller2 — source only

This is a separate successor to controller1 source25ae013c… and its failed
build index8814dfb6… / terminal53daedd3…. The previous attempt stopped
before compiling because its dry-run parser rejected an unselected `ln … ||
echo …` prerequisite. All original source and failure bytes remain unchanged.
Only selected five Fortran compile commands and the final check link are now
validated/executed. Unselected prerequisite text is ignored; selected compound
commands, new source operands, unknown compiler prefixes, duplicates or missing
commands still fail. The exact failed dry-run text is a file-backed regression
fixture. Physics, module source, cards, policy, caps and the all-source compiler
audit are unchanged. There has been no successor compilation or native action.

This additive capsule implements the unexecuted proposal sealed by
`fortran-qp-transport-profile-1/index.json` SHA256
`049c26fbf460e0102269bdd237a71fd8d2438160f0fa47afeb57be59aa218ba2`.
The original 17 payloads and index remain unchanged. No compilation, loader
query, model setup or amplitude has been performed by this capsule. It adds
supervision, provenance and read-only IEEE observations, not an optimization.

The only case is the closed current-source QCD `u u~ > z g g` export, backend9,
production `0fcd2cd873`, native `eb3d67fd…`, unchanged cards and100 saved points.
After its complete original learning traversal, each fresh mode0/1/2 worker
makes exactly two additional automatic-policy calls: point1 (219 DP), point16
(323 QP). Both orders' coefficient/accuracy words and status must equal the
sealed expected results. The complete100 steady warmup rows must equal the
closed original observation. No RUN command, fixed precision, repeated timing
batch, cache implementation, changed stability threshold or new point is used.

BUILD and RUN are separate coordinator admissions. Every invocation requires
an explicit reviewed `--source` index hash. RUN additionally requires the
reviewed **build/terminal.json** hash; this terminal binds build-result.json,
which binds recipes, prelink inventory, actual LOADs, all objects, executable,
private sources, provider identities and protected inputs. Merely finding a
self-generated build result does not admit RUN. Stage directories are exclusive
and immutable after closure; failures stop later modes without retry.

The existing pinned StageDeadline, OwnedChild and SignalRequests supervise one
controller child per stage. Compiler/query subprocesses inherit that owned
session; each retained command records its parent/session, return and output.
Worker's own session and cleanup remain unchanged. The inherited cleanup covers
observed descendants with start-identity checks; it does not claim a kernel-level
proof against an unobserved child created and reparented between samples.
Cleanup uncertainty stops admission. Failed raw files remain at the recorded
private working path even if parsing/copying never completes.

Caps include lock waiting, preparation, checks, commands, parsing, copying and
cleanup:180 seconds BUILD,300 seconds per mode,60 seconds cleanup reserved in
each,1,080 seconds for the four stages. Nonexclusive CPU12 is not a reservation.
BUILD and the entire native mode (including loader queries) acquire the same
`.tirust-build/numerical-tests.lock`. The owned child and descendants inherit
16GiB address-space,512MiB per-file and zero-core limits; actual limits are
retained before work. The Fortran shadow buffer keeps its original64MiB cap.
There is no silent truncation. Administrative failure publication can occur
after a deadline and cannot establish a measurement pass.

BUILD privately copies sources and numerical objects, dereferences mutable
source/include links and leaves only `lib` as a read-only original reference.
Every new `.f90` is included in the actual expanded recipe inventory. It executes
only five compile commands (qp_profile, private mtirust, TIR_interface,
TIRUST_interface, check_sa), then one link. Generated units retain their exact
recorded flags; the two private modules use the recorded installed module
`-O3 -fPIC` policy. Every other numerical object must remain byte-identical.
Private mtirust.o precedes the archive. `nm` and the actual link map must show
exactly one definition of every MTIRUST symbol and no pulled original MTIRUST
archive member. No Rust/model/HELAS/polynomial compilation is allowed.

The reused CMS helpers prebind explicit/implicit inputs, compiler/wrapper data,
CRT objects and transitive linker scripts before linking. Every actual map LOAD
and PT_INTERP must be prebound before a loader call. The actual ELF interpreter
performs `--list`; all resolved providers must exactly equal the frozen expected
closure. The historical generic-ldd receipt remains unchanged: its explicit
foreign query-interpreter alias (glibc2.42) is excluded from the expected actual
process, whose ELF PT_INTERP is glibc2.40. This is a disclosed read-only
derivation, checked by the new actual-interpreter query and process maps.
No generic ldd result is passed off as actual process mappings.

`input-pins.json` binds source/build/card/point/runtime/tool bytes and the original
object map. The original exporter inventory's two null entries are pre-existing
broken symlinks: their exact targets are separately pinned; a new resolved file
or changed link fails. Neither historical absence nor source identity is reset.

IEEE flag, halting and rounding queries surround only the two selected calls,
without clearing or setting any state. Actual pre/post states must match mode0.
Unsupported observations remain−1. Sticky true→true does not prove a newly raised
exception. Native counters retain statuses/epochs/generations and explicit
absent-context pre-clear snapshots; invalid counters or query failures stop
interpretation. Clock rates, scopes, ticks and derived seconds must be valid and
finite. Overlapping stage clocks are not summed. Mode2 counts exact potential
within-event repeats only, with full key/eligibility checks; no budget simulation,
cache benefit, precise performance acceptance or speedup is inferred.

From the repository root, using the exact hashes supplied after source review:

```sh
source tools/tirust_env.sh
python3 -B docs/benchmarkfolder/experiments/fortran-qp-transport-profile-controller-2/controller.py \
  --preflight --source REVIEWED_SOURCE_SHA256
# Only after BUILD admission:
python3 -B docs/benchmarkfolder/experiments/fortran-qp-transport-profile-controller-2/controller.py \
  --build --source REVIEWED_SOURCE_SHA256
# Only after reviewing the closed build and granting RUN admission:
python3 -B docs/benchmarkfolder/experiments/fortran-qp-transport-profile-controller-2/controller.py \
  --run --source REVIEWED_SOURCE_SHA256 --reviewed-build REVIEWED_BUILD_TERMINAL_SHA256
```

Pure tests execute no compiler, loader, Fortran executable or reducer:

```sh
python3 -B -m pytest -q -p no:cacheprovider \
  docs/benchmarkfolder/experiments/fortran-qp-transport-profile-controller-1
```

Results, if later admitted, use this capsule's append-only `build/` and `mode-N/`
receipts; private artifacts/raw data live under
`.tirust-build/fortran-qp-transport-profile-controller-2/`. Final precise timings
remain Hua-Sheng's local TODO. The original source proposal remains historical
and unexecuted evidence until a separate actual execution closure says otherwise.
