This package retains the completed **release adaptive-QP standalone Fortran tensor diagnostic** from `.tirust-build/adaptive-qp-release-fortran-qualification/run1`. It is not an amplitude, Born, kinematic-repair, production-build, or performance qualification.

The original execution and result JSON files are copied byte for byte. All 34 other runtime files are UTF-8 text compressed losslessly with deterministic gzip. `copies.json` binds compressed bytes, uncompressed bytes and their original paths. `evidence-index.json` binds every package file except itself. No ELF objects, executables or shared libraries are included. Their recorded hashes and actual captured loader/mapping evidence remain in `execution.json` and the raw streams.

The frozen analyzer verifies the exact binary128 input/output transport, thirteen input components requiring more than binary64 precision, successful full and partial 256-to-512 tensor escalation, cache/check namespace boundaries, invalid-mode error preservation, context recreation and optional-getter compatibility with the released bridge. Ordinary tensor output agrees between the legacy and adaptive bridges. Partial finite values remain unevaluated zero ABI placeholders. Last dispatched tensor precision is not accepted amplitude precision; `successes_256` and `accepted_amplitude_tier` remain null.

From the repository root, replay with Python 3 and its standard library:

```bash
python3 -B docs/benchmarkfolder/experiments/adaptive-qp-release-fortran-qualification-v2/runtime1/replay.py
python3 -B -m unittest discover -s docs/benchmarkfolder/experiments/adaptive-qp-release-fortran-qualification-v2/runtime1 -p test_replay.py -v
```

The entire `runtime1` directory can also be copied elsewhere and its `replay.py` executed directly. Replay checks the exact file inventory and hashes, original execution/build/source bindings, original worker exits and mapped runtime closure, decompresses into a temporary directory, and runs only the frozen Python analyzer. It requires no Rust, Fortran, native libraries, original absolute paths, or host binaries. It compares the regenerated result object and its exact serialized SHA-256 with the original result. It does not rerun the controller, compiler, loader, or reduction.

`frozen/source-index.json`, every indexed source, and the entire retained `build1` evidence package preserve their existing bytes. Their earlier admission/compile-only wording describes those earlier stages; this separate runtime package does not rewrite their history. The runtime record contains the later successful calls. Build and runtime elapsed values are supervisory records, not benchmark timings. Actual native library bytes are recorded evidence, not redistributed or independently rehashed by portable replay.

Origin SHA-256 bindings:

- Execution: `51170427b184026ddc0a695a9509e43a647b5067a3227831bf537e184f60b1c3`.
- Results: `1db3228443e5bcc37d104541e608b62aba36b17000d1c8096bff6d838f6d644f`.
- Source index: `f85e28784762a70806398ab5b878f93f01c0a1ca51475d960c50036a9ad7995f`.
- Build record: `f27a190f04804e8dc13a9d534e8558ac4eab95c2881cf97cf0aaaefb12260005`.

This run observed the separately built release DSO `c5de24e6c41a90caf9e34684392d0193588522f26e6bbac646bdf74e5f151cb7`. It does not infer native release behavior from the debug run. The v2 source index and retained release link record are bound independently; the release build evidence index is `f6c4fd59be05127bfe3e98beb7480f61f2802225f31e4834d34ee661a75008a8`. Earlier source/build indices under `frozen/ancestry` preserve the reused-object history. Identical analyzer result JSON between debug and release is an observed standalone diagnostic result, not an amplitude or runtime equivalence claim.
