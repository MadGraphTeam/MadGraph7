Current native-tier work follows the [final local validation TODO](LOCAL_FINAL_VALIDATION_TODO.md).
Status `219` identifies accepted DP transport and can include internal tensor
precision rescues. Classifying a current point as native DP requires qualified
per-point route evidence and Fortran coefficient-cache provenance. The frozen
`0fcd2cd873d6` addendum below remains historical; its results do not qualify the
current source.

---

# Hua-Sheng's local precision-cohort timing TODO

This is an addendum to the preserved
[local timing handoff](campaigns/entry-latch-cache-guard-1/timing-handoff-1/LOCAL_TIMING_TODO.md)
for production `0fcd2cd873d61d134837e1d9e87afb13a47824fd` and baseline
`9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`. The original handoff remains
unchanged. Precise timings are to be measured on Hua-Sheng's local machine.

The handoff's `forward campaign` command produces warmed full-stream timings
and final accepted backend/precision histograms. It does not produce separately
timed native-DP and rescued cohorts. **Qualifying a local cohort adapter and
running those diagnostics remain TODO.** No executable local adapter command
is provided yet; the existing cohort tool is not wired into that local workflow.

For each of CutTools, Ninja, NinjaRS, COLLIER and TIRust:

- Classify the saved physical points using observed accepted backend and
  precision. Freeze the exact point IDs and sample counts before timing.
  Retain empty, refused and migrating cohorts explicitly, along with observed
  status/precision usage. Final accepted status does not identify every
  internal reduction attempt.
- Keep the registered physical inputs, automatic rescue policy and independent
  stability checks unchanged. A rescued-cohort timing includes unsuccessful DP
  work and the natural rescue; forcing QP is a different measurement.
- Warm each frozen cohort, then measure at least seven alternating paired
  repetitions with batches of approximately one second or longer. Keep the
  compared point IDs fixed and report initialization separately. Retain
  failures and status migrations; do not select a favorable replacement sample.
- Label the results as diagnostic native-DP or rescued-cohort timings, with
  their sample counts and classification scope. Coefficient agreement and
  accepted status do not establish independent numerical truth.

These diagnostics supplement the full-stream campaign. Formal performance
acceptance still requires the upper 95% paired candidate/baseline runtime bound
to be at most `1.05` for every one of the eight main processes, comparing the
TIRust-first chain to the materialized starting revision's default chain.
Existing-backend regressions remain separate. Cohort timings do not replace
that gate or complete the overall integration goal.
