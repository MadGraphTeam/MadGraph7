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
only as an independent validation and benchmarking reference.
