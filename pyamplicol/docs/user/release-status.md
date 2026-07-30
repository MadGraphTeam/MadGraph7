<!-- SPDX-License-Identifier: 0BSD -->

# Release Status

This guide targets standalone pyAmpliCol `0.1.0`. The package is not yet
published on PyPI.

## Implemented

- Schema-v1 TOML, direct CLI, ordered overrides, typed Python configuration,
  and recorded license/resource adjustments.
- External UFO/JSON model loading, canonical compiled-model schema 9, and
  model-compiler policy version 13, including packaged `sm`, `scalars`, and
  `scalar_gravity` assets plus an immutable public compiled-model handle that
  keeps compiler IR private.
- Multiprocess planning and transactional schema-v3 generation with append,
  replace, payload integrity, and deterministic validation-point support.
- The primary external JSON `p p > Z j j` workflow and stable concrete process
  names such as `p_p_to_z_j_j_4`.
- Total/resolved runtime evaluation, selectors, f64 and Python precision
  control, atomic UFO parameter updates, warnings, and typed physics metadata.
- One generated root API bundle with Python, C11, dependency-free Rust 2021,
  C++17, and Fortran 2008 drivers.
- Rusticol core, Python extension, C ABI v1, safe Rust source wrapper, C++
  wrapper, Fortran module source, and target-specific static SDK discovery
  through `rusticol-config`.
- Symbolica-independent f64 execution for direct SymJIT applications and
  target-compatible ASM/C++ compiled evaluators.
- Default prepared-model recurrence execution and opt-in eager execution with
  the portable, wheel-owned
  built-in-SM `built-in-sm-jit-o2` kernel pack, compact process invocation
  tables or recurrence schedules, runtime flow/helicity selectors, and fixed
  reusable workspaces. Compiled DAG execution remains an explicit mode.
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
gate and a fresh numerical comparison against the pinned independent Fortran
implementation, retains one source distribution, builds the three target
wheels from that source distribution, tests installed wheels on CPython 3.11
and 3.14, and collects the unchanged package files. The
dependency contract is now patchless and pinned to a verified immutable
SymJIT fork revision. Exact release-mode prepared-model packs are checked in
under `release_assets/prepared_models` and projected into release builds. A
successful exact-main run of this complete three-platform workflow remains the
final artifact validation gate.

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

- The exact Symbolica 2.2.0 Python/Rust combination and immutable SymJIT fork
  revision are verified in the release dependency contract, with no local
  patches or installer source rewrites.
- The checked-in built-in-SM release packs carry the exact release dependency
  identity and pass the fail-closed package audit when projected from
  `release_assets/prepared_models`.
- `ufo-model-loader==0.1.7` is the verified published loader input.
- Every supported wheel target must complete clean installation, Python
  self-test, generated Python/Rust/C++/Fortran driver tests, and native SDK
  audits.
- Candidate artifacts are marked non-publishable and are rejected by the
  publication workflow.
- Protected `testpypi` and `pypi` GitHub environments exist. Matching
  TestPyPI/PyPI Trusted Publisher or pending-publisher registrations must still
  be confirmed before the first publication.

The SymJIT dependency is an immutable Git source override, not an applied
source patch. It is valid for Python wheel and source-distribution builds, but
those source builds require access to the pinned Git revision. Publishing
pyAmpliCol's internal Rust crates independently through crates.io would still
require a compatible SymJIT release on crates.io.

Strict release builds fail closed on these conditions. The authoritative
machine-readable state is `dependencies/release-lock.toml`; contributor-only
state is not a release fallback.

The defined publishing workflow is manual and downloads the package files from
the explicitly selected **Validated release artifacts** run. It checks the
actual three-wheel/one-source-distribution inventory but does not re-parse or
authenticate GitHub workflow/job metadata, and it does not rebuild or modify
the files. Before its first use, maintainers must confirm that the existing
`testpypi` and `pypi` GitHub environments have the intended approval policy and
register matching Trusted Publishers with TestPyPI and PyPI so those OIDC
claims are accepted.
