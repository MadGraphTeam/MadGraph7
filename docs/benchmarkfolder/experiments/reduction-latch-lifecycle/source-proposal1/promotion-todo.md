# Inputs requiring successor handling if promoted

* Apply only `proposal.patch` to the optimized template, then add public-entry
  lifecycle coverage. Keep direct-adapter tests' failure-latching semantics.
  Existing `tests/unit_tests/loop/test_ninjars_fortran.py` outer-wrapper coverage
  and `test_tirust_fortran.py` adapter coverage remain useful regressions.
* Preserve the precision-stream initialization sequence required by
  `tools/tirust_benchmark_precision_stream.py:130–139`. The new assignments go
  immediately before it. Keep its driver from clearing host flags.
* Re-run campaign/local drift and historical-snapshot tests in
  `test_tirust_benchmark_campaign.py` and `test_tirust_benchmark_local.py`.
  Their dynamic checks do not need historical pins rewritten.
* Create a new campaign/source identity, generated matrix/object receipts and
  qualification. The two assignments change numerical source identity.
  `invariant-symmetry-1` manifests, source snapshots and evidence stay historical.
* Publish a successor local handoff and precision-stream instructions. Update
  live navigation in `docs/benchmarkfolder/README.md:126–139` and the selector in
  `campaign_report_refresh.py:359–361`, which currently chooses handoff4.
  `PRECISION_STREAM_TODO.md:16–19` currently links handoff3. Existing selected
  report/dossier/context/ledger pins must stay unchanged; use a successor packet.
* A successor raw-pilot classifier must distinguish inherited pre-public-call
  flags from current-call failures for the exact qualified entry-reset source.
  The historical `stability_appendix_c/pilot.py:866` rejects before=true even if
  after=false and therefore cannot classify this successor unchanged. Preserve
  its old parser/evidence. Cohorts already classify post-call flags; precision
  streams retain before/after observations separately.
* Keep old qualification template hashes historical. Add a new scoped lifecycle
  item to `qualification.json` only after actual qualification. Do not relabel
  previous source proofs or measurements as covering the successor.
