# Dependency Modes

pyAmpliCol has two deliberately separate dependency modes.

## Release Mode

Release-equivalent builds use exact versions published on PyPI and crates.io,
except for SymJIT, which is redirected from crates.io to one immutable commit
of `ValentinHirschi/symjit_changes_for_pyamplicol`. The exact Git repository,
branch, revision, upstream PR, versions, and compatibility state are recorded
in `release-lock.toml`. Release builds never apply source patches or reference
a local checkout.

`tools/release/check_dependencies.py` is a hard release gate. It verifies the
workspace-level SymJIT source override and fully resolved Cargo lock, including
the full Git revision, before release artifacts are built.

## Candidate Development Mode

`just dev-install` uses immutable Symbolica/GammaLoop source revisions and the
checksummed SymJIT source archive listed in `contributor-lock.toml`.
That archive is generated from the same fork revision used by release mode, so
the active contributor patch list is empty. Candidate mode exists for
development and physics validation only.
It installs the verified published `ufo-model-loader==0.1.7` wheel directly
from the hash-locked runtime closure. Artifacts produced in this mode record
the candidate revisions and resulting source-tree identity and are not
eligible for PyPI publication.

The contributor build uses the checksummed fork archive for SymJIT 2.21.1 at
revision `60a9d66fbfb2181d36a5747c389714eccc187244` on branch
`pyamplicol-generic-direct-apis`. The fork contains the ordered generic SymJIT
commit series and is an `rlib`-only pyAmpliCol integration branch. The
installer verifies and uses the pristine archive tree without rewriting it or
applying patches. The changes provide deterministic AArch64 compressed
funclets, generic direct split-plane applications, a generic table-driven
direct application on AArch64 and x86-64, and safe lowering for stored direct
applications whose internal spills or mapped outputs retain scratch
registers. The direct
contracts expose overwrite/accumulate, live/before-write input, and
identity/complex-scalar policies; Rusticol owns the mapping from pyAmpliCol
recurrence roles to those policies. The previously unreleased direct contracts
are reset to storage v1 and table binding v1, with no pyAmpliCol-specific
compatibility loader. Direct bytecode is trusted input rather than a hostile
payload, while ordinary shape, range, and alias checks remain part of the safe
calling contract. The build uses Symbolica and symbolica-community at the
immutable planned-release revisions recorded in the lock. GammaLoop is pinned
to the merged main revision that provides Spenso's
cached symbolic-parallelism policy. Spynso3 initializes that policy in `Auto`
mode, checking the license once and keeping symbolic tensor reductions serial
for restricted users or parallel for licensed users.

Symbolica 2.2.0 is pinned from PyPI and crates.io. SymJIT is pinned through the
workspace's `[patch.crates-io]` table so Symbolica and Rusticol resolve the
same fork revision rather than compiling two copies. This is suitable for
pyAmpliCol's precompiled Python-wheel publication workflow because no Rust
crate is published to crates.io. Once the upstream PR is released, the
workspace override can be replaced by an exact crates.io version without
changing Rusticol's generic API use.

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
