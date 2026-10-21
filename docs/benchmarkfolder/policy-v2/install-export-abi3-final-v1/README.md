# Final source installation and relocated exports

This harness is prepared as source only while the final optimization decisions
are measured. No final revision, archive, installation or result is claimed by
its presence. It adapts the successful `install-export-abi3-zero-sign-v1`
harness without changing that historical evidence.

`prepare.py` requires an explicit committed revision and the sealed final Rust
proof. It archives that revision, binds every archived bridge source byte to
the one-executable Rust proof, and records the actual generator inputs under
`madgraph`, `aloha`, `Template`, `bin`, `models` and `input`. The proof follows
the existing `unit_evidence` record format: bound ordinary stdout/stderr,
build record, executable and complete `bridge_source_sha256`, extended with
the named ordinary/fixture/replay/convergence runs. Counts and test names come
from retained libtest output. Replay and convergence use the same final
ignored-run capture; historical output changes and sign differences remain
explicit. Installation and linkage checks do not declare those differences
numerically equivalent.

The actual repository CLI installs pinned dependencies without sibling source
overrides into the archived checkout's private `HEPTools` prefix. It retains
the installer invocation
`cargo rustc --release --locked --lib -- --print native-static-libs` and the
compiled Fortran DP/compensated-DP/binary128 smoke. The prefix and Cargo cache
stay inside this repository. Installation does not update the main working
checkout's `HEPTools/ninjars`.

Actual CLI cards then generate standalone and FKS/NLO outputs with internal
dependencies. Runtime validation relocates both outputs, checks their copied
archive/interface/link metadata as regular files, hides the private
installation, and preserves the standalone DP110/QP120/stable218 checks against
the retained CutTools coefficients. NLO compilation retains twenty pole
cancellations, the soft/collinear check and GNU binary128 helper linkage.
These are fixed-order checks, with no shower claim. Python checks run against
the archived final source, including the installer transaction, absent-backend,
statistics and upstream checks; their actual names and count are retained.

After the final candidate and one-binary Rust proof are committed and root
schedules the build/evaluation slot, run from the repository root:

```sh
source .ninjars-build/dev-env.sh
taskset -c "$NINJARS_FINAL_CPU" python -B \
  docs/benchmarkfolder/policy-v2/install-export-abi3-final-v1/prepare.py \
  --revision "$NINJARS_FINAL_REVISION" --cpu "$NINJARS_FINAL_CPU" \
  --rust-proof "$NINJARS_FINAL_RUST_PROOF" \
  --rust-proof-manifest-sha256 "$NINJARS_FINAL_RUST_MANIFEST_SHA256"
taskset -c "$NINJARS_FINAL_CPU" python -B \
  docs/benchmarkfolder/policy-v2/install-export-abi3-final-v1/export.py
taskset -c "$NINJARS_FINAL_CPU" python -B \
  docs/benchmarkfolder/policy-v2/install-export-abi3-final-v1/runtime_validator.py \
  --build --hide-installation --run --collect
taskset -c "$NINJARS_FINAL_CPU" python -B \
  docs/benchmarkfolder/policy-v2/install-export-abi3-final-v1/focused_validator.py
python -B docs/benchmarkfolder/policy-v2/install-export-abi3-final-v1/seal.py --live
```

`NINJARS_FINAL_RUST_PROOF` is the final controller's `unit-proof.json`;
the explicit SHA256 is that directory's `proof-manifest.json`. This harness
does not build a second test binary. A failed run remains in this namespace;
it is not overwritten or replaced by a successful claim. The verifier reads
retained files and hashes only, and never compiles or evaluates a library.

After the private installation/export proof passes, root separately publishes
the validated archive, interfaces and metadata into the main working checkout's
`HEPTools`, recording the copy provenance and actual main archive SHA256 before
final core/supplemental generation. The old main `1199bc84...` and corrected
private `832fc0d1...` installations remain historical evidence. A private
success does not imply that either main runtime or final benchmark provenance
has changed. Final eight-process, loop-induced, stability and performance
acceptance remain separate gates.
