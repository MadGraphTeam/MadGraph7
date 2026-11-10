# Checked Born and repair controls: 51 native cases passed

All five isolated GNU Fortran control executables completed successfully on
CPU14. Their raw logs contain all **51 expected PASS records**, each exactly
once, and one `ALL PASS` terminator per executable. The common controller used
one inclusive 60-second deadline, retained its actual commands and environment,
and verified unchanged source and compiled-artifact hashes before and after.
The recorded 0.114 seconds is controller wall time, not a performance benchmark.

| Fixture | Passed cases | Purpose |
|---|---:|---|
| Scalar, 3 external legs | 10 | Scalar adapter; all-flavor helicity shortcut |
| Scalar, 5 external legs | 10 | Scalar adapter; distinct flavor learning |
| Split orders, 3 external legs | 11 | All split slots and learning-buffer checks |
| Split orders, 5 external legs | 11 | Hidden orders and distinct flavor learning |
| Checked repair preparation | 9 | Explicit method outcomes and staged adoption |

The Born cases execute the exact reviewed transaction aggregate and the
original aggregate, using controlled MATRIX contributions. Finite native
output bits and learning state agree. The drivers also check independently
tabulated values, valid zeros, invalid-flavor early returns, fixed helicities,
NaN/infinity, beam-weight/sum/normalization overflow, excluded helicities, and
unchosen split-order outputs. Rejected calls restore every flavor of `NTRY`
and `GOODHEL`, pending `HELRESET`, and the wrapper's prior `USERHEL`. They
preserve the input point, an entry sticky flag, rounding and trap settings.
Fixture-only read-only observer ENTRY additions expose private state; removing
them recovers the exact reviewed and original aggregate bytes.

The repair cases execute the exact checked wrapper and PREPARE host, with
controlled original method helpers. They check primary and alternate success,
both-method failure, nonphysical input, supported method choices and rejected
unsupported selection. Raw errors, helper order, explicit outcome/adoption,
unchanged DP input and caller sentinel preservation are checked. The expected
`ERROR::`/`WARNING::` messages in `repair.log` come from deliberately injected
failure branches; the driver verifies their outcomes and finishes successfully.

This qualifies these controlled Fortran paths with GNU Fortran 15.2.0,
`-O0 -fcheck=all -fno-fast-math`, and disabled traps on entry. It does **not**
qualify numerical kinematic repair, physical amplitudes, all IEEE classes,
enabled-trap termination, other rounding modes, optimized performance, or the
broader integration. The parent report contains the separate physical-process
evidence and any failures. No reducer library is used by these five fixtures.

`original/` preserves the exact prepared templates, rendered Fortran, stubs,
drivers, source manifest, declared commands, expected cases, controller,
complete native report and five raw logs. Its README and `prepared_only`
fields are historical preparation records; `original/native.json` supplies
the later native result. `generators/` and `support/` retain the exact bound
generator inputs and process owner. Original repository-relative identities
are recorded in `source-map.json`; original absolute command paths remain
unchanged in their historical records.

`build/compile-proof.json` contains the exact five successful compiler events,
their source bindings, environment and compiled-artifact hashes. The exact
compiler-log suffix for those jobs is preserved. Full parent build report,
log and input inventory identities are retained by SHA256; this subset does
not reconstruct unrelated parent build bytes. It cross-checks those fields
against the complete control native report. Executables, object files and
module files are represented **only by original path and hash**.

Portable verification requires only this directory and Python's standard
library. It reads no ignored working files and does not compile, load or run
native code:

```sh
python3 -B docs/benchmarkfolder/experiments/born-effective-point-native/controls/verify.py
```

The verifier checks the complete text inventory, original-copy hashes, source
and observer identities, successful compiler events and command/log bindings,
unchanged native artifact identities, the effective single-process commands,
shared deadline, and every raw PASS record. An optional `--live-root /path/to/repo`
also checks still-existing original files and binary hashes. The default works
after the original working directories have been deleted, or after this
directory has been copied elsewhere. No binary/pickle/pycache is packaged.

The preserved `original/compiler-jobs.json` contains the exact declared build
commands. The actual commands in `build/compile-proof.json` use the pinned
compiler and the parent's shared build budget. For independent compilation
from this portable bundle, the following command sequence uses the same five
Fortran source triplets and flags, writing only to a new ignored directory:

```sh
source tools/tirust_env.sh
bundle=docs/benchmarkfolder/experiments/born-effective-point-native/controls/original
replay=.tirust-build/born-effective-controls-replay
for name in scalar_n3 scalar_n5 split_n3 split_n5 repair; do
  mkdir -p "$replay/$name"
  unit=aggregate.f
  if [ "$name" = repair ]; then unit=wrapper.f; fi
  taskset -c 14 gfortran -O0 -g -fcheck=all -fbacktrace -fno-fast-math \
    -ffixed-line-length-none -ffree-line-length-none \
    -J "$replay/$name" -I "$replay/$name" -I "$bundle/sources/$name" \
    "$bundle/sources/$name/stubs.f90" "$bundle/sources/$name/$unit" \
    "$bundle/sources/$name/driver.f90" -o "$replay/$name/control.exe" || exit
done
```

The executable names and expected stdout are listed in `original/expected.json`.
A new replay should run them serially, retain new logs/exit codes, and enforce
one shared deadline. Historical `original/run.py` documents the exact reviewed
process owner, cleanup, hash admission and output checks; it is retained without
rewriting its original workspace paths. A replay is new evidence and must not
overwrite this bundle or be substituted for the recorded run.
