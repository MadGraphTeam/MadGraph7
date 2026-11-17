# Local final validation TODO (Hua-Sheng)

The current agent-ready task is the newest campaign's local timing handoff.
[`CURRENT_STATUS.md`](CURRENT_STATUS.md) names that campaign; today it is
[`campaigns/tirust-2b9ba961-1/timing-handoff-1/LOCAL_TIMING_TODO.md`](campaigns/tirust-2b9ba961-1/timing-handoff-1/LOCAL_TIMING_TODO.md).

It covers:
- a fresh `install tirust`;
- all 66 numerical lanes;
- the precise paired timing gates (upper 95% paired candidate/baseline bound
  ≤ 1.05 for each of the eight main processes, against baseline
  `9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f`);
- separate existing-backend regressions;
- the new backend-9 precision cohorts;
- the local report.

`tirust-2b9ba961-1` (TIRust `main` `2b9ba961`, bridge method choice off) replaces
`tirust-a5d4afbf-1` and the rehearsal `tirust-9a781998-1`, whose handoffs stay in
their campaign directories. It has one open blocking defect whose bridge fix may
change TIRust's cost; if a newer campaign exists by the time you measure, time that
one.

Earlier versions of this file, for `tirust-native-tiers-1` and before, are in
git history.
