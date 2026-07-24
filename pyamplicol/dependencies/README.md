# Dependency Modes

pyAmpliCol has two deliberately separate dependency modes.

## Release Mode

Release-equivalent builds use only versions published on PyPI and crates.io.
They never apply patches or reference a local checkout. The exact required
versions and compatibility state are recorded in `release-lock.toml`.

`tools/release/check_dependencies.py` is a hard release gate. At present,
strict release mode is expected to fail because published Symbolica
serialization/runtime compatibility has not been marked verified.

## Candidate Development Mode

`just dev-install` uses immutable Symbolica/GammaLoop source revisions and the
checksummed SymJIT source archive listed in `contributor-lock.toml`.
It applies only the target-, revision-, and SHA-256-pinned patches listed in
that lock. Candidate mode exists for development and physics validation only.
It installs the verified published `ufo-model-loader==0.1.7` wheel directly
from the hash-locked runtime closure. Artifacts produced in this mode record
the candidate revisions, patch identity, and resulting source-tree identity
and are not eligible for PyPI publication.

The contributor build uses the checksummed upstream source archive for SymJIT
2.21.1 at revision `48197f32536c894b51ef25b2cf05ddd05c22675f`.
The installer verifies and applies the tracked AArch64 compressed-funclet patch
before changing the crate target from `cdylib` to `rlib` so Symbolica can
consume it as a Rust dependency. It then verifies the complete configured tree
against the lock. The patch enables relative funclet calls for scalar, SIMD,
and fast-complex AArch64 code and makes funclet emission deterministic; its
tests cover evaluation parity and storage-v3 round trips. The build uses
Symbolica and symbolica-community at the immutable planned-release revisions
recorded in the lock. GammaLoop is pinned to the merged main revision that
provides Spenso's cached symbolic-parallelism policy. Spynso3 initializes that
policy in `Auto` mode, checking the license once and keeping symbolic tensor
reductions serial for restricted users or parallel for licensed users. The
matching Symbolica, symbolica-community, and SymJIT versions must be published
with the required fixes before a strict PyPI build can replace the currently
verified release pins.

The contributor-only SymJIT patch and planned Symbolica revision remain
upstream release blockers. The Symbolica revision also has a blocker for
pyAmpliCol's compiled-complex C++/CUDA path: complex constants can be emitted
with nested scalar wrappers that do not compile as `std::complex<double>`.
Publication requires upstream fixes or an explicit exclusion of the affected
capabilities; the release dependency gate remains unchanged and fails closed.

The original Fortran AmpliCol checkout is optional, developer-only, and used
only as an independent validation and benchmarking reference. The pinned
`amplicol_with_patches` branch removes the unnecessary LHAPDF
link from its direct color probe, exposes complete recursion-kind diagnostics,
and reports each LC contraction-row partition. This
resolves the physical color component only for a genuinely single-flow case;
multi-flow fixtures use the rows only to verify the complete per-helicity
aggregate. None modifies
amplitude physics. `just legacy-physics` builds that probe and checks the
tracked low-multiplicity LC/NLC/full fixture, including every physical
helicity and every independently resolvable color component, against the
pinned Fortran implementation.

The pinned upstream revision contains no `LICENSE` or `COPYING` file. That fact
is recorded in contributor-only provenance, not the release dependency lock.
The checkout and its developer-only branch are never redistributed in a wheel
or sdist.
