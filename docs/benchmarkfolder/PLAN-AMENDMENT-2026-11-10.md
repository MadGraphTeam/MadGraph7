# Performance measurement handoff — user amendment, 2026-11-10

Hua-Sheng has directed that the final precise timings will be performed on his
local machine, possibly by an agent. Work on the current shared host should
continue on a best-effort basis without reserving CPUs. This instruction
supersedes the original requirement to establish final performance acceptance
on the current host before delivery.

The numerical campaign, frozen baseline, backend comparisons, physical stress
requirements and implementation checks remain required. The current ordinary
campaign continues with its unchanged 100-point streams and numerical policy.
Completed and failed experiments remain part of the evidence.

Current-host timings must be labelled provisional, with actual inputs,
initialization, sample sizes, elapsed times, backend/precision usage and host
conditions recorded. They cannot establish final performance acceptance.
Nonexclusive CPU affinity may make a run reproducible; it is not a reservation.
Do not request an exclusive CPU set or launch the previously prepared
quiet-window reservation controller. Preserve that controller as an unexecuted
historical protocol.

## TODO for Hua-Sheng on his local machine

- Rebuild the final delivered candidate and baseline
  `9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f` with identical pinned models,
  compiler flags, cards, point sets, helicity/color sums and rescue policy.
- Replay numerical qualification before timing. Use all 100 saved physical
  points per main process; do not substitute the provisional timing subset.
- Run warmed full-amplitude, single-core alternating paired timings with at
  least seven pairs and approximately one second per timed batch. Report
  initialization separately and retain actual accepted backend and precision.
- Evaluate the original upper 95% paired confidence-bound requirement
  (candidate/baseline no greater than 1.05) for every main TIRust-first chain
  and the existing-backend regression comparisons. Run the standalone and
  control comparisons, and distinguish native DP from rescued results.
- Retain failures and regenerate the Typst report from local raw measurements.
  A provisional shared-host speedup is not a substitute for this acceptance.

The final delivery must link a concrete local reproduction/handoff document,
including an agent-ready task and commands for the final candidate revision.
Until those local measurements exist, the report must show precise performance
acceptance as **deferred to Hua-Sheng**, not passed. The assigned goal remains
active while implementation, numerical qualification and delivery work remain.
