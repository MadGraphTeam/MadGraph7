# Source-only guarded adaptive release build

This prepares one private shared-library build from the exact source and
dependency lineage of adaptive release DSO
`c5de24e6c41a90caf9e34684392d0193588522f26e6bbac646bdf74e5f151cb7`.
**No compiler, linker, loader, reducer, or amplitude has been executed for
this proposal.** Root admission is required before its one build attempt.
The earlier 27-file zero-guard proposal remains unchanged.

The existing bridge and patched TIRust trees are copied under:

```text
.tirust-build/cayley-guarded-adaptive-build-proposal-1/sources/
  adaptive-qp-rescue-prototype3/
  pole-development/
```

Preserving sibling names leaves every Cargo manifest and lockfile
byte-identical. The bridge lock SHA-256 remains
`0d09649d4005c3a905dd605407d29e5b2818cd2c574d608d34a7f1dff3d4a2d3`.
The `madloop-pole-kernels` symlink is materialized as private regular files
with identical contents. Its original vendor target is only read.
The full copied input inventory has 499 files, approximately 49.2 MB;
`source-copies.json` binds every original and copied file separately.

There is exactly one executable-source change: the signed-zero guard from
the [separate tiny-fixture proposal](../cayley-zero-denominator-guard-proposal-1/README.md).
The actual existing patched TIRust `gram.rs` has SHA-256
`4565a5075e6abae10dc062be4119a206855fa533eeab09ea1767cdd4e9631858`;
the privately guarded copy is
`572fff44788d7a419151065a75af86a028cc1477987ec70dc0f47d70894a4dda`.
Its prior pole-development patches remain unchanged. The baseline is not
silently substituted with the differently hashed pristine reference file.

The old successful release receipt is retained, including its 441
protected-input bindings, effective environment and compiler selection.
All still match during preparation. oneloopmaster remains pinned to
`bd703d09a78f7849a6b86a07096e6f222fc01193`; Symbolica remains pinned to
`dfe6bd97b974b1028dc63071ea165136cd2b0ef5`. NinjaRS is untouched.

## Proposed one-attempt protocol

After root admission, run from the MadGraph repository root:

```sh
source tools/tirust_env.sh
python3 -I -B docs/benchmarkfolder/experiments/cayley-guarded-adaptive-build-proposal-1/build.py --root-admitted
```

The existing reviewed process-group supervisor is reused. One 600-second
budget includes source preflight, cache copying, the release build,
artifact copying/hashing, and final protected-input closure. Every child
is owned and reaped; there is no retry or CPU reservation. An absolute
`ITIMER_REAL` alarm covers those operations and successful receipt
publication. Failure-only metadata may be retained after child cleanup
and expiry; it cannot turn that attempt into a successful build.

On admission only, the controller copies the existing 2.73 GB release
cache into its own `build-cache/release` using `cp --archive
--reflink=auto`. This permits a copy-on-write clone where supported and
ordinary private copies otherwise; it creates no hard links to seed
files. The seed cache is never used as the new writable build directory.
The final Cargo target directory is also private. Cargo source downloads
remain the existing cached `CARGO_HOME`, with offline and locked mode.
Cache availability does not guarantee reuse; actual Cargo records will
report freshness.

The effective compiler/environment stays the old release environment,
except for these two output paths. No Rust flags or compiler wrappers are
added. The precise executable paths and file hashes are in `plan.json`.
The build is the original command with the copied manifest:

```sh
prlimit --as=4294967296 --core=0 -- taskset -c 11 cargo build --release \
  --manifest-path .tirust-build/cayley-guarded-adaptive-build-proposal-1/sources/adaptive-qp-rescue-prototype3/Cargo.toml \
  --locked --offline --features poles-only --lib --message-format=json
```

Settings remain Rust 1.97.1, GNU 15.2 toolchain, release opt-level 3,
thin LTO, codegen-units 1, one Cargo job, and one thread for numerical
dependencies. CPU11 affinity is nonexclusive. The additional resource
cap is **4 GiB virtual address space per copy/build process**, inherited
by children; it is not an aggregate RSS measurement. A resource or
deadline failure is retained and stops the attempt.

Successful build acceptance requires the actual core and bridge Cargo
records to show recompilation from the private source manifests, with
release optimization and no test target. All path dependency manifests
must resolve within the private source copy. The one emitted shared
library is hashed and copied into `attempt1/runtime`; it is never loaded
or installed by this controller. Original source, vendor, reference,
existing libraries, and current generated numerical campaigns remain
protected. Future masked/trapped and forced-QP physical driver runs are
separate work requiring a bound artifact and separate admission.

For a read-only preflight with no compiler or native execution:

```sh
python3 -I -B docs/benchmarkfolder/experiments/cayley-guarded-adaptive-build-proposal-1/verify_source.py
```

Full reducer DP/QP behavior, physical accuracy, stability checks, cache
behavior and performance remain unqualified until those later runs.
