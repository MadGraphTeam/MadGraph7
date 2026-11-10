# Adaptive bridge compile-only controller

This append-only wrapper leaves the frozen d0d59b7b source package and its
original proposed protocol unchanged. Root authorized exactly one compile-only
attempt on CPU 11, one Cargo job, with an inclusive 900-second cap, after the
cause-2 compilation releases that CPU. No Rust tests or amplitude run is
authorized by this controller.

The command differs only by replacing GNU timeout's extra ten-second kill grace
with the already reviewed runner2 supervisor (69f76225). Its source inventory is
authenticated before import. The same owned-process helper supervises short
tool-version queries. The supervisor keeps child-group ownership through
signal-deferred cleanup and enforces its deadline before accepting an exit.

The frozen source and 295 external pins are validated before and after the
attempt. A positive recorded environment allowlist excludes compiler injection
and dynamic-loader overrides. The caller first sources tools/tirust_env.sh.
Cargo uses the existing caches, locked/offline mode and --no-run. A successful
build records the exact test executable hash without executing it. The attempt
directory cannot be reused; any failure is retained and requires review.

For a later separately authorized test, the same helper can impose an inclusive
180-second cap on the exact recorded binary under the shared numerical lock,
CPU 11 and one test thread. That future test command is not implemented or run
here. Frozen runner2's actual amplitude replay still has priority when its
natural lock boundary becomes available.
