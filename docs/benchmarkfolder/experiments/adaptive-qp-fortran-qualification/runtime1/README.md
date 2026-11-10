This package retains the completed **debug adaptive-QP standalone Fortran tensor diagnostic** from `.tirust-build/adaptive-qp-fortran-qualification/run1`. It is not an amplitude, Born, kinematic-repair, production-build, or performance qualification.

The original execution and result JSON files are copied byte for byte. All 34 other runtime files are UTF-8 text compressed losslessly with deterministic gzip. `copies.json` binds compressed bytes, uncompressed bytes and their original paths. `evidence-index.json` binds every package file except itself. No ELF objects, executables or shared libraries are included. Their recorded hashes and actual captured loader/mapping evidence remain in `execution.json` and the raw streams.

The frozen analyzer verifies the exact binary128 input/output transport, thirteen input components requiring more than binary64 precision, successful full and partial 256-to-512 tensor escalation, cache/check namespace boundaries, invalid-mode error preservation, context recreation and optional-getter compatibility with the released bridge. Ordinary tensor output agrees between the legacy and adaptive bridges. Partial finite values remain unevaluated zero ABI placeholders. Last dispatched tensor precision is not accepted amplitude precision; `successes_256` and `accepted_amplitude_tier` remain null.

From the repository root, replay with Python 3 and its standard library:

```bash
python3 -B docs/benchmarkfolder/experiments/adaptive-qp-fortran-qualification/runtime1/replay.py
python3 -B -m unittest discover -s docs/benchmarkfolder/experiments/adaptive-qp-fortran-qualification/runtime1 -p test_replay.py -v
```

The entire `runtime1` directory can also be copied elsewhere and its `replay.py` executed directly. Replay checks the exact file inventory and hashes, original execution/build/source bindings, original worker exits and mapped runtime closure, decompresses into a temporary directory, and runs only the frozen Python analyzer. It requires no Rust, Fortran, native libraries, original absolute paths, or host binaries. It compares the regenerated result object and its exact serialized SHA-256 with the original result. It does not rerun the controller, compiler, loader, or reduction.

`frozen/source-index.json`, every indexed source, and the entire retained `build1` evidence package preserve their existing bytes. Their earlier admission/compile-only wording describes those earlier stages; this separate runtime package does not rewrite their history. The runtime record contains the later successful calls. Build and runtime elapsed values are supervisory records, not benchmark timings. Actual native library bytes are recorded evidence, not redistributed or independently rehashed by portable replay.

Origin SHA-256 bindings:

- Execution: `2fe072df782064edb9fb4f435d96e17041e20c211d178a79cd094175aaf6aa48`.
- Results: `1db3228443e5bcc37d104541e608b62aba36b17000d1c8096bff6d838f6d644f`.
- Source index: `80f83d5384ada33b609ac94bb9f9be935eb92e56ee706f979494e4d053fcc9b5`.
- Build record: `a213c73aadab7d898e1a3699ab7c1c4ae09dae8afa1c7916a8e6671ebfb14b15`.
