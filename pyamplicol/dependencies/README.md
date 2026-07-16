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

`just dev-install` uses immutable Symbolica/SymJIT source revisions and the
checksummed Symbolica patches listed in `contributor-lock.toml`. Candidate mode
exists for development and physics validation only. It installs the verified
published `ufo-model-loader==0.1.7` wheel directly from the hash-locked runtime
closure. Artifacts produced in this mode record the candidate revisions and
are not eligible for PyPI publication.

The contributor build uses unpatched SymJIT 2.20.2 at
`1d9e7b104c29d612981b1d59cae0cfe8fbf9a4d1`. On macOS arm64 the retained
requested-level-3 wrong-result probe agrees with requested level 1 to
`3.47e-16`. The revision also adds the
production vector-emitter safeguard that prevents the previously observed
oversized stack adjustment. The deliberately synthetic probe that calls the
low-level stack helper with a frame above 16 MiB still exceeds AArch64's
immediate range, but that manufactured route is no longer reachable from the
generated evaluator path. No local SymJIT patch is applied.

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
