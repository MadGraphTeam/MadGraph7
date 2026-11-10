# Proposed release-Fortran adaptive qualification

This additive source package reuses the frozen debug-Fortran fixture, numerical
objects, observer, optional telemetry shim, analyzer and released-ABI1 control.
It changes the adaptive executable's selected library from debug `63298bdd…`
to the optimized private DSO
`c5de24e6c41a90caf9e34684392d0193588522f26e6bbac646bdf74e5f151cb7`.
No link, loader or numerical process has been run by this package. Build and
native stages require separate root admission. The original debug source,
build and any later debug native attempt remain immutable.

The selected release artifact is
`.tirust-build/adaptive-qp-rescue-release-build/attempt1/runtime/libmadloop_tirust_adaptive_private.so`,
bound by release-build index
`e417c11182884afed742daed880aef138a0daa92189b554db844279b946e4dbe`.
Its numerical sources are the exact qualified adaptive revision 3. The library
has no SONAME, so the link uses this exact absolute filename. Actual runtime
mapping remains a later, independent requirement.

The original Fortran source index is `80f83d53…`, portable build index
`7d92500b…`, and build record `a213c73a…`. All five reusable objects are checked
against that frozen build record, copied byte-for-byte to the separate work
directory, and checked again. No Fortran or C source is recompiled. The legacy
control executable is also copied byte-for-byte. Only the adaptive executable
is relinked: its five object paths, output/map paths and selected DSO/RUNPATH
change. Compiler driver, ordered objects, wrapper flags, pinned glibc 2.42
interpreter and other runtime paths remain exactly as in the original link.
`release-plan.json` retains both complete commands and every changed argument.
`controller.patch` records the full source-control changes.

Every file under `source/` except `controller.py` is byte-identical to the
frozen debug package, including the Fortran driver, C forwarding observer,
all binary128 fixture words/records and analyzer. The expected control matrix
is unchanged: one ordinary QP call in the legacy worker; three ordinary,
three captured N4R3 full, three captured poles-only and one deliberately invalid
pole-mode call in the adaptive worker. Telemetry/error/cache/check/generation
assertions and exact 113-bit transport checks are unchanged. The fixed captured
N4R3 point is a control-flow fixture, not physical truth. No new test point,
threshold, tensor input, precision policy or scientific tolerance is introduced.

The proposed build is one serial relink under a 120-second cap and 4 GiB address
space on CPU 12 nonexclusively; no CPU is reserved. It uses the original
reviewed owned-process supervisor and retains the link map and exact input,
object, executable and library hashes. The work/output namespace is
`.tirust-build/adaptive-qp-release-fortran-qualification/{build1,run1}`.
There is no automatic retry or modification of a debug artifact.

The later native protocol is copied unchanged: two fresh workers under the
common nonblocking numerical lock; each has 180 seconds/4 GiB on CPU 12; exact
interpreter dependency listings get 30 seconds each. The bounded startup
handshake precedes tensor work and permits actual `/proc` ELF mapping capture.
Both forwarding wrapper flags remain present; the optional weak getter itself
is not wrapped. Full records, errors, mappings and output words are retained.
No stage is launched merely by source preparation or review.

This remains a standalone tensor transport and telemetry test, without Born,
numerator or physical amplitude construction. Release completion must not be
inferred from debug qualification, and a later tensor pass must not be treated
as mixed-tier physical-stress acceptance or production promotion. Final precise
timings remain Hua-Sheng's local TODO; elapsed times here only record resource use.

Source verification only:

```
python -B docs/benchmarkfolder/experiments/adaptive-qp-release-fortran-qualification/source/controller.py verify
```

The same command with `build` would perform the single relink; `run` would
perform the two native workers. Neither is authorized by this source package.
