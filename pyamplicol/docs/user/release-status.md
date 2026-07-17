<!-- SPDX-License-Identifier: 0BSD -->

# Release Status

This guide targets standalone pyAmpliCol `0.1.0`. The package is not yet
published on PyPI.

## Implemented

- Schema-v1 TOML, direct CLI, ordered overrides, typed Python configuration,
  and recorded license/resource adjustments.
- External UFO/JSON model loading, canonical compiled-model schema 9, and
  model-compiler policy version 11, including packaged `sm`, `scalars`, and
  `scalar_gravity` assets plus an immutable public compiled-model handle that
  keeps compiler IR private.
- Multiprocess planning and transactional schema-v3 generation with append,
  replace, payload integrity, and deterministic validation-point support.
- The primary external JSON `p p > Z j j` workflow and stable concrete process
  names such as `p_p_to_z_j_j_4`.
- Total/resolved runtime evaluation, selectors, f64 and Python precision
  control, atomic UFO parameter updates, warnings, and typed physics metadata.
- One generated root API bundle with Python, dependency-free Rust 2021, C++17,
  and Fortran 2008 drivers.
- Rusticol core, Python extension, C ABI v1, safe Rust source wrapper, C++
  wrapper, Fortran module source, and target-specific static SDK discovery
  through `rusticol-config`.
- Symbolica-independent f64 execution for direct SymJIT applications and
  target-compatible ASM/C++ compiled evaluators.
- `examples list|copy|run`, `config template|resolve`, `doctor`, and installed
  `self-test` utilities.
- Artifact `inspect` with stable-ID or exact-expression process selection, and
  adaptive `profile`/`benchmark` timing with statistical uncertainty and native
  Rusticol phase breakdowns.
- Workflow definitions for candidate artifacts, retained source distributions,
  audited wheels, clean installed-wheel tests, and OIDC Trusted Publishing.

## Current CI And Artifact Workflows

The automatic **Tests** workflow runs a lightweight configuration API matrix
on every push to `main` and on pull requests, covering CPython 3.11 through
3.14. A separate dependency-minimal CPython 3.11 job checks public API and CLI
contracts, import laziness, repository policy, deployment harness logic, SDK
configuration, and release metadata. Pull requests and manual dispatch
additionally run one guarded Ubuntu candidate job covering selected
unit/integration tests, generation, the Python/Rust/C++/Fortran APIs, Rust
checks, the native SDK, self-test, and checkout-independent examples. Keeping
the candidate job off ordinary `main` pushes prevents rapid documentation or
report checkpoints from repeatedly cancelling the same expensive dependency
build.

The full **Candidate artifacts** workflow is `workflow_dispatch` only. It runs
release-tool preflight, the complete candidate source gate and an isolated
candidate deployment on Ubuntu, then builds and audits non-publishable candidate
artifacts for macOS arm64, macOS x86_64, and manylinux x86_64. These jobs use the
pinned contributor candidate inputs; their outputs are explicitly marked as
candidates, retained for inspection, and cannot be promoted to release files.

The **Validated release artifacts** workflow is also manually dispatched. Its
defined release path verifies published dependency inputs, runs the full source
gate and independent Fortran oracle, retains one source distribution, builds
the three target wheels from that source distribution, tests installed wheels
on CPython 3.11 and 3.14, and collects the unchanged package files. It currently
fails closed while the release dependency contract below remains unverified.

## Remaining Integration Gates

- The complete installed-wheel matrix must pass on macOS arm64, macOS x86_64,
  and manylinux x86_64 for CPython 3.11 and 3.14.
- Non-JIT and Python precision fallback coverage needs its final platform pass.
- The independent legacy-Fortran physics ladder and documented generation,
  runtime, artifact-size, and memory regression gates must complete in the
  release workflow.
- Built-in SM compatibility code is isolated under `models.builtin`, while
  shared generation uses structural particle and color roles. Exact normalized
  color/Lorentz certificates recover built-in graph topology across the
  documented UFO-SM `n<=4` family ladder. Typed source, crossing, and
  propagator records drive wavefunctions, crossing phases, mass class, gauge,
  numerator/denominator, auxiliary, and Goldstone policy without SM-PDG
  fallbacks. Runtime cards may vary nonzero masses but must regenerate an
  artifact when a particle changes between massive and massless state spaces.
  Artifacts predating the typed source or propagator contracts request
  regeneration explicitly. Default and model-supplied UFO propagators are
  distinguished by normalized expressions rather than object names. The
  compiled-model schema now persists authoritative tensor-ordering and
  colored-contact proof records. Compiler policy 10 derives the fundamental
  `3 x 3bar x 8` Fierz singlet-subtraction auxiliary from exact UFO tensors,
  which restores built-in/external current, interaction-evaluation, root, and
  sector topology through the validated three-quark-line ladder without names
  or conventional PDGs. Compiler policy 11 additionally records every declared
  Goldstone's massive-vector relationship from normalized spin, color,
  quantum-number, mass-expression, and propagator metadata; ambiguous matches
  fail during compilation and custom propagators keep their Goldstones
  explicit. Relabeled-PDG, renamed-particle, and reordered-UFO-inventory gates
  preserve topology and tensor-order identities. A scalar-gravity source with
  transposed spin-2 axes and renamed Lorentz dummy indices compiles to identical
  canonical kernels and numerical component values. Current
  two-structure-constant contact lowering preserves explicit source scalar
  prefactors separately from canonicalized permutation parity, rejects
  residual color tensors, and validates model-owned direct and closure
  contractions in compiled-model schema 9. LC topology replay is reconstructed
  onto public helicity/color axes for f64 and exact evaluation before selectors
  are applied.
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
- TestPyPI and PyPI Trusted Publisher registrations and the protected GitHub
  environments that authorize their OIDC identities are not configured yet.

Strict release builds fail closed on these conditions. The authoritative
machine-readable state is `dependencies/release-lock.toml`; contributor-only
state is not a release fallback.

The defined publishing workflow is manual and accepts only a successful
default-branch **Validated release artifacts** run. It downloads the already
validated three wheels and one source distribution and does not rebuild or
modify them. Before it is operational, maintainers must create protected
`testpypi` and `pypi` GitHub environments with the intended approval policy and
register matching Trusted Publishers with TestPyPI and PyPI so those OIDC
claims are accepted.
