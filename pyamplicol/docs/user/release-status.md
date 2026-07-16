<!-- SPDX-License-Identifier: 0BSD -->

# Release Status

This guide targets standalone pyAmpliCol `0.1.0`. The package is not yet
published on PyPI.

## Implemented

- Schema-v1 TOML, direct CLI, ordered overrides, typed Python configuration,
  and recorded license/resource adjustments.
- External UFO/JSON model loading and canonical compiled-model schema v7,
  including packaged `sm`, `scalars`, and `scalar_gravity` assets.
- Multiprocess planning and transactional schema-v3 generation with append,
  replace, payload integrity, and deterministic validation-point support.
- The primary external JSON `p p > Z j j` workflow and stable concrete process
  names such as `p_p_to_z_j_j_4`.
- Total/resolved runtime evaluation, selectors, f64 and Python precision
  control, atomic UFO parameter updates, warnings, and typed physics metadata.
- One generated root API bundle with Python, dependency-free Rust 2021, C++17,
  and Fortran 2008 drivers.
- Rusticol core, Python extension, C ABI v1, C++ wrapper, Fortran module source,
  and target-specific static SDK discovery through `rusticol-config`.
- A Symbolica-independent direct SymJIT application capability that executes at
  f64 without importing Symbolica or consulting its runtime license state.
- `examples list|copy|run`, `config template|resolve`, `doctor`, and installed
  `self-test` utilities.
- Standard source distribution, wheel, `twine check`, platform audit,
  clean-install, and Trusted Publishing workflows.

## Remaining Integration Gates

- The complete installed-wheel matrix must pass on macOS arm64, macOS x86_64,
  and manylinux x86_64 for CPython 3.11 and 3.14.
- Non-JIT and Python precision fallback coverage needs its final platform pass.
- The independent legacy-Fortran physics ladder and documented generation,
  runtime, artifact-size, and memory regression gates must complete in the
  release workflow.
- Built-in SM compatibility code is isolated under `models.builtin`, and shared
  generation uses structural particle and color roles. Exact normalized
  color/Lorentz certificates now recover built-in graph topology across the
  documented n<=4 external-SM family ladder. The remaining model-independence
  gate is to complete typed source/crossing/propagator/contraction records and
  the relabeled-PDG and reordered-tensor adversarial fixtures.
- External `generate --dry-run` currently requires a previously compiled model
  or populated model cache; it does not compile a trusted source as a planning
  side effect.
- The public model API has no package-name resource resolver; `examples copy`
  materializes packaged assets as ordinary filesystem inputs.

## Publication Gates

- The exact published Symbolica Python/Rust combination and SymJIT application
  compatibility remain unverified in the release dependency contract.
- `ufo-model-loader==0.1.7` is the verified published loader input.
- Every supported wheel target must complete clean installation, Python
  self-test, generated Python/Rust/C++/Fortran driver tests, and native SDK
  audits.
- Candidate artifacts are marked non-publishable and are rejected by the
  publication workflow.

Strict release builds fail closed on these conditions. The authoritative
machine-readable state is `dependencies/release-lock.toml`; contributor-only
state is not a release fallback.

Publishing is manual. A protected GitHub environment receives already
validated wheels and one source distribution through PyPI Trusted Publishing;
the publishing job does not rebuild or modify package files.
