# Preserved closed10 report

The canonical 14-page PDF, exact build ledger, context and modified Typst
sources are preserved byte for byte from commit `ffa22845749ec0f52f1da2daedddd3787c91290b`.
The closed10 selection/dossier remain unchanged in the campaign report-snapshots
directory. Every original report input is available from that Git commit at
`docs/benchmarkfolder/<ledger input path>` and is pinned by report-build.json.
The archive manifest confirms the compact copies against the same Git blobs.
The canonical report passed --verify-only immediately before preservation.

Reconstruct all ledger inputs from the recorded commit before replaying the old
renderer; do not substitute later campaign summaries or Typst sources. This
archive performs no numerical evaluation and contains no new measurements.
