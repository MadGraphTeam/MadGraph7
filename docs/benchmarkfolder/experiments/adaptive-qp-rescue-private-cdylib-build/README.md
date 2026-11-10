# Private adaptive QP shared library: compile/link evidence

The exact adaptive revision3 source compiled into a private debug shared
library in 98.08120 seconds on CPU11 with one Cargo job, inside the reviewed
inclusive300-second process cap. No source correction was needed. No library
was loaded, test or reducer executed, installation performed, or production
change made. The earlier d22953ed test binary and all four protected historical,
current and private512 DSOs retained their hashes.

The isolated artifact is:

```
.tirust-build/adaptive-qp-rescue-cdylib-build/attempt1/runtime/libmadloop_tirust_adaptive_private.so
```

SHA256: `63298bdd242b99dd46d654debe6304f5b4fdff955d5082d8c50c3b2ff01ae9ea`

This is an unoptimized debug artifact, 542,062,696 bytes including debug
information. It is suitable for later private correctness work, subject to
runtime qualification; no performance inference is supported. Its exact
source index remains `f5550b87…`, and locked dependencies plus the successful
test-build environment were reused. Cargo built the cdylib and rlib, without
the test runner. Exact argv, environment, tool versions, source/artifact hashes
and compiler cleanup are retained in attempt1/execution.json.

Pure ELF-byte inspection finds13 prefixed bridge exports, including
ml_tirust_qp_precision_statistics. The ordinary Cargo cdylib has no SONAME.
Its private filename is intentionally distinct from released libmadloop_tirust.
Future private Fortran linkage must bind the actual selected artifact rather
than assume an installed-library name. Recorded dependencies are libgcc_s,
libm, libc and ld-linux, with Nix glibc2.42-61/GCC15.2 runtime search paths.
This metadata does not prove which runtime a future process will load or
establish compatibility with an older executable interpreter.

No binary is copied into the report. bindings.json records the isolated
artifact by path, size and hash; source/controller/log copies are exact.
The reviewed ELF reader is copied from the earlier frozen design and reads
bytes only. Runtime loading, the queued20 Rust tests, Fortran transport and
IEEE/trap behavior, mixed-tier physical comparisons and actual-tier telemetry
remain separate gates. This compile/link outcome does not change their status.

Run `python -B verify.py` for portable inventory and build/provenance checks.
Add --live to rehash protected sources, the test binary, frozen DSOs and the
new private artifact and to reproduce ELF metadata without loading it.
