# Local final validation TODO (Hua-Sheng)

The current agent-ready task is the newest campaign's local timing handoff.
[`CURRENT_STATUS.md`](CURRENT_STATUS.md) names that campaign; today it is
[`campaigns/tirust-a5d4afbf-1/timing-handoff-1/LOCAL_TIMING_TODO.md`](campaigns/tirust-a5d4afbf-1/timing-handoff-1/LOCAL_TIMING_TODO.md).

It covers:
- a fresh `install tirust`;
- all 66 numerical lanes;
- the precise paired timing gates (upper 95% paired candidate/baseline bound
  ≤ 1.05 for each of the eight main processes, against baseline
  `9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`);
- separate existing-backend regressions;
- the new backend-9 precision cohorts;
- the local report.

`tirust-a5d4afbf-1` is the final-pin campaign (TIRust `main` `a5d4afb`). It
replaces the rehearsal `tirust-9a781998-1`, whose handoff stays in its campaign
directory.

Earlier versions of this file, for `tirust-native-tiers-1` and before, are in
git history.
