<!-- SPDX-License-Identifier: 0BSD -->

# Release Status

This documentation targets standalone pyAmpliCol `0.1.0` while recording the
current implementation boundary. It must be updated as integration closes
these gaps.

## Present Now

- Immutable schema-v1 configuration classes, field registry, TOML loading,
  path resolution, ordered overrides, and recorded clamps.
- CLI parsing and typed dispatch for `generate`, `evaluate`, `benchmark`,
  `inspect`, and `model inspect|compile|processes`.
- Interactive/noninteractive Symbolica trial and hobbyist request commands.
- Public model/process request types and lazy service facades.
- Non-writing `Generator.plan()` and CLI `generate --dry-run` validation.
- Transactional schema-v3 generation with payload hashing, typed runtime
  physics metadata, process aliases, validation points, and append/replace
  modes.
- One generated root `API/` bundle containing Python, C++17, and Fortran 2008
  resolved-evaluation drivers and deterministic validation data.
- A lazy Python-to-Rusticol adapter for total/resolved evaluation, selectors,
  precision control, model-parameter updates, warnings, and typed physics
  metadata.
- Typed runtime benchmarking of the optimized summed evaluation path.
- `examples list|copy|run`, `config template|resolve`, `doctor`, and
  installed-wheel `self-test` commands.
- Built-in SM plus packaged `sm`, `scalars`, and `scalar_gravity` UFO/JSON
  assets.
- Rusticol core, Python extension source, C ABI, C++17 wrapper, Fortran 2008
  module, and wheel-owned SDK discovery source.
- Direct SymJIT application payloads and explicit runtime capabilities, so the
  default JIT artifact has a Symbolica-independent f64 execution path while
  precision fallback and non-JIT backends remain distinguishable.
- Lazy Python non-f64 execution through Symbolica's retained evaluator states,
  including resolved LC/NLC/full reduction and live model-parameter updates.
- A portable 64-bit-little-endian installed-package physics fixture whose
  saved SymJIT MIR is retargeted at wheel build time, then loaded and compiled
  for the host without importing Symbolica. The self-test checks total and
  resolved values from the resulting schema-v3 JIT artifact.
- macOS arm64 candidate-wheel integration for mixed LC, single-process LC,
  NLC, and full color, with matching resolved metadata and values across the
  Python, C++17, and Fortran 2008 runners.
- The current macOS arm64 candidate wheel
  (`sha256:403675588bca05e8e28cb40ada116859189655e7f48f8d9b6e99c7fc455faeca`)
  passes isolated installation, native-SDK smoke tests, the Symbolica-blocked
  f64 self-test, and all 31 installed integration tests. Its audited candidate
  sdist has SHA-256
  `ce662c9262626e0b14e0a86968ab818015234813e2311996e4ab5fc469f18d41`.

## Integration Gates

- Packaged model assets do not have a public name-to-resource resolver;
  external model inputs must be filesystem paths. Copied examples provide a
  checkout-independent helper that materializes the packaged assets.
- Installed-wheel Python/native SDK integration is green on macOS arm64 but has
  not completed on macOS x86_64 or manylinux x86_64.
- The independent legacy-Fortran physics ladder and the documented
  generation/runtime/RSS regression gates are not yet automated in this
  standalone repository.
- Non-JIT and arbitrary-precision fallback coverage still needs the final
  release-platform pass.
- The post-parity model-independence pass remains open: generic lowering must
  replace or explicitly isolate unexplained PDG/name-specific rules and pass
  relabeled-PDG and reordered-tensor invariance tests.
- Final clean-clone artifact parity, independent audits, and reviewed commits
  remain internal release work.

## Publication Gates

- `pyamplicol==0.1.0` has not been published.
- The exact Symbolica/SymJIT build remains a patched candidate, and
  `ufo-model-loader==0.1.7` is recorded as awaiting publication.
- Release dependency and artifact hashes are not yet marked verified; the
  hashes above identify only the current non-publishable candidate artifacts.
- macOS arm64, macOS x86_64, and manylinux x86_64 targets remain planned rather
  than release-validated.
- Strict release builds fail closed on these conditions. Candidate wheels are
  explicitly non-publishable and cannot be uploaded by the release tooling.

The authoritative machine-readable state is
`dependencies/release-lock.toml`; this page summarizes it for users.

## Documentation Convention

Commands that only parse, resolve, copy resources, or inspect a source tree may
be tested without a native extension. Generation requires the contributor
candidate dependencies until compatible published releases are available.
`self-test` is successful only for a built wheel whose extension, native SDK,
and host-retargeted direct-SymJIT physics fixture pass their installed-package
checks. Native-driver examples are validated locally on macOS arm64; the other
release targets remain gated. No packaged example assumes a parent-repository
path.
