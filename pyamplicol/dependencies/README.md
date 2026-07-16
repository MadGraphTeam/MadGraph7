# Dependency Modes

pyAmpliCol has two deliberately separate dependency modes.

## Release Mode

Release-equivalent builds use only versions published on PyPI and crates.io.
They never apply patches or reference a local checkout. The exact required
versions and compatibility state are recorded in `release-lock.toml`.

`tools/release/check_dependencies.py` is a hard release gate. At present,
strict release mode is expected to fail because the validated
`ufo-model-loader` 0.1.7 revision is not yet available on PyPI and the
published Symbolica serialization/runtime compatibility has not been marked
verified.

## Candidate Development Mode

`just dev-install` uses immutable source revisions and the checksummed patches
listed in `release-lock.toml`. Candidate mode exists for development and physics
validation only. Artifacts produced in this mode record the candidate
revisions and are not eligible for PyPI publication.

The managed `ufo-model-loader` patch makes sparse JSON restrictions match UFO
restriction-card semantics and evaluates unrestricted serialized models from
their declared external defaults. It is applied locally to the pinned public
revision; contributor setup never depends on an unpublished upstream branch.

The original Fortran AmpliCol checkout is optional, developer-only, and used
only as an independent validation and benchmarking reference. Three narrow,
checksummed diagnostic patches remove the unnecessary LHAPDF link from its
direct color probe and expose complete recursion-kind diagnostics. A third
checksummed diagnostic patch reports each LC contraction-row partition. This
resolves the physical color component only for a genuinely single-flow case;
multi-flow fixtures use the rows only to verify the complete per-helicity
aggregate. None modifies
amplitude physics. `just legacy-physics` builds that probe and checks the
tracked low-multiplicity LC/NLC/full fixture, including every physical
helicity and every independently resolvable color component, against the
pinned Fortran implementation.

The pinned upstream revision contains no `LICENSE` or `COPYING` file, so the
release lock records its license as `NOASSERTION`. The checkout is never
redistributed in a wheel or sdist. The three pyAmpliCol-authored diagnostic
patches are part of this 0BSD source distribution; that does not assert or
replace a license for the upstream Fortran source to which a developer applies
them locally.
