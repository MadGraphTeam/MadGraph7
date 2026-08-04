<!-- SPDX-License-Identifier: 0BSD -->

# Release Status

Version `0.1.1` is tagged as the immutable
[`v0.1.1` source snapshot](https://github.com/mg5amcnlo/pyamplicol/tree/v0.1.1).
It is available for testing from
[TestPyPI](https://test.pypi.org/project/pyamplicol/0.1.1/) but has not yet been
uploaded to PyPI.

The [validated release-artifacts
workflow](https://github.com/mg5amcnlo/pyamplicol/actions/workflows/release-artifacts.yml)
retains one source distribution and three `cp311-abi3` wheels:

- macOS 11 or newer on Apple silicon;
- macOS 11 or newer on x86-64;
- manylinux 2.28 x86-64.

Each wheel completed the full installed Python, C11, C++17, Fortran 2008, and
Rust 2021 API deployment on CPython 3.11. CPython 3.14 received a focused abi3
installation, import, metadata, and direct-runtime smoke test. The workflow
also completed its source preflight and independent Fortran physics oracle.

Performance campaigns are separate manual measurements and are not run in
release CI. The repository retains only two rendered PDFs, indexed in the
[performance report documentation](../performance_reports/README.md); use
`pyamplicol profiling-campaign copy DEST --force` to create or reset a
self-contained campaign and reproduce the report format directly. Its visible
`DEST/campaign_artifacts/` state moves with the campaign and never reuses
legacy repository-level `.artifacts` state; the reset removes only its local
state and managed generated outputs while preserving unrelated destination
files.

Release dependencies use the official `siravan/symjit-crate` 2.22.0 repository
at immutable revision `d8abfeeb4db98c13cdcf9dd39cf3e795fd5001a7`. There is no
private SymJIT fork or local patch. pyAmpliCol has no LHAPDF dependency.

TestPyPI publication completed through its Trusted Publisher. The remaining
publication action is to select the final successful validated-artifact run
whose head SHA exactly matches the intended PyPI publication commit, then pass
that run ID to the same manual publishing workflow with the `pypi` destination.
Publication does not rebuild the artifacts.
