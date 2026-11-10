# Proposed private adaptive release build

This controller is prepared for review only. It has not run a compiler, test,
reducer or dynamic-library loader. The root agent must admit the build.

The exact qualified revision 3 source remains at
`.tirust-build/adaptive-qp-rescue-prototype3`, source index
`f5550b8719e76e177ad6b73d456cc655815770eb7092a8fabdc776c415c03213`.
No numerical source, Cargo manifest or lockfile changes are proposed. Its
existing release profile uses opt-level 3, thin LTO and one codegen unit.
The exact private crate name remains `madloop_tirust_adaptive_private`.

Proposed command, using the pinned Cargo/toolchain and positive environment
from the successful compile-only qualification:

```
cargo build --release --manifest-path .tirust-build/adaptive-qp-rescue-prototype3/Cargo.toml --locked --offline --features poles-only --lib --message-format=json
```

`CARGO_BUILD_JOBS=1`. The existing shared Cargo home, target and build directory
are reused. No CPU is reserved and no affinity override is applied. The single
900-second monotonic budget includes Cargo/Rust version inspection and the
build, enforced by the already reviewed runner2 owned-process supervisor.
The supervisor reaps the complete owned process group on timeout/interruption.
There is no automatic retry or budget extension and no test command.

The source-only cache inspection records 25 existing release rlibs for
Symbolica, OneLOop, core and pole packages (about 1.01 GB). None of the observed
iregi-rust release artifacts identifies the exact `pole-development` path,
so Cargo may rebuild path-dependent core packages. Availability is not proof
of cache freshness or exact reuse. The actual Cargo JSON log must establish
which artifacts were selected and whether they were fresh. Locked/offline
execution prohibits dependency revision changes or network fetching.

The output attempt is `.tirust-build/adaptive-qp-rescue-release-build/attempt1`.
On success the exact release artifact is copied to its own
`runtime/libmadloop_tirust_adaptive_private.so` and recorded by hash, size and
Cargo profile. The already qualified test binary and debug 63298bdd library,
all source bindings and frozen historical/current DSOs are checked before and
after. No install prefix is modified. The release artifact is not loaded.

Actual Fortran optional-telemetry and 113-bit transport qualification is owned
by acceptance_audit and currently binds the immutable debug 63298bdd artifact.
This build does not duplicate those fixtures or transfer their result to a new
binary. Any later release-native proof must explicitly bind the new artifact.
Physical mixed-tier qualification and production promotion remain separate.
Final precise performance measurements are Hua-Sheng's local TODO; this build
does not establish performance acceptance on the shared host.

`python -B verify.py` performs only source/hash/AST checks. Once separately
authorized, `python -B run.py` performs the one bounded compilation.
