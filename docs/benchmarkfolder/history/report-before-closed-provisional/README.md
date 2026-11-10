# Preserved closed11 report before provisional timings

The 14-page canonical PDF, exact build ledger, context and changed report sources
are preserved byte for byte from commit `772b15ed39cc8c1c21a3ecbc8e616e1fb3d97071`. The copied
canonical README is `canonical-README.md`. The closed11 selection, dossier and
verification record remain unchanged in the campaign report-snapshots directory.
Every original report input is identified by `report-build.json` and can be
reconstructed from that commit at `docs/benchmarkfolder/<ledger input path>`.
The archive manifest verifies each compact copy against its exact Git blob.

The retained closed11 verification record states that the original report passed
`--verify-only`. This preservation does not rerun that old renderer with later
helper sources. Reconstruct its pinned Git inputs before reproducing the old
build. No scientific evaluation or timing is performed by this archive.
