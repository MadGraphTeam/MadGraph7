<!-- SPDX-License-Identifier: 0BSD -->

# Release Status

Version `0.1.0` is tagged as an immutable archival source snapshot at commit
`863a228915ebe236551b31849a1bad3dc2cb12d9`. It has not yet been uploaded to
PyPI or TestPyPI.

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
release CI. The repository retains only their four rendered PDFs; use
`pyamplicol profiling-campaign copy DEST --force` with a new or empty `DEST`
to create a clean campaign.

Release dependencies use the official `siravan/symjit-crate` 2.22.0 repository
at immutable revision `d8abfeeb4db98c13cdcf9dd39cf3e795fd5001a7`. There is no
private SymJIT fork or local patch. pyAmpliCol has no LHAPDF dependency.

The remaining publication action is to select a successful validated-artifact
run whose head SHA exactly matches the intended publication commit, then pass
that run ID to the manual publishing workflow for TestPyPI or PyPI. The first
upload may require confirming the corresponding Trusted Publisher registration
and environment approval policy; it does not rebuild the artifacts.
