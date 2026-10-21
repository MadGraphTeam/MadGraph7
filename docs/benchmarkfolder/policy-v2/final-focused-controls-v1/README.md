# Final complex-mass and mixed-order controls

These fresh candidate measurements use the unchanged representative worker,
protocol, ten-point streams and `controls.json` settings. Each target has two
complete persistent commands and a separate fresh-process parity command.
The short durations are correctness diagnostics, not performance acceptance.

`specs/complex.json` selects all four backends from the actual complex-mass CLI
export in `../final-complex-generation-v1`. `specs/mixed-legacy.json` selects
CutTools, Ninja and Collier from the actual mixed-order CLI export in
`../final-representative-generation-v1`. The already completed NinjaRS mixed
measurement in `../final-representative-v1/runs/mixed` is reused rather than
evaluated again. It contains four loop orders and three Born orders.

Generation belongs to `b0ea105a04afa34304d437d2956e4d4672360670`, numerical
source to `d05b59f3277ce0d727cb2816d8b7ad45107e58c4`, and the installed archive
has SHA256 `800a941faf3cdfdab764fdffe1d5ab73f84984a786aa194596b753af6b6f5d42`.
Every candidate's complete reduction descriptor is
`8b454519cb6818ef1fafe37378930c276f7e85394a2333ebc9fdb00e75726cff`.
Documentation-only orchestration revisions are recorded separately. The
collector checks the actual numerical source and installed runtime before
and after collection.

The executed commands were:

```sh
source .ninjars-build/dev-env.sh
taskset -c 190 python -B docs/benchmarkfolder/policy-v2/final-focused-controls-v1/collect.py complex \
  --authorization 'Root focused numerical handoff 2026-10-21'
taskset -c 188 python -B docs/benchmarkfolder/policy-v2/final-focused-controls-v1/collect.py mixed-legacy \
  --authorization 'Root focused numerical handoff 2026-10-21'
```

`ledgers/` retains controller PID, start ticks and process group, exact child
commands, six-hour limits, results and child-exit checks. The two commands ran
in independent processes; only the complex collection used NinjaRS. The
initial source-preparation shell lacked `python` on PATH and exited before
writing specifications or launching any engine; sourcing the retained tool
environment resolved that operational preflight issue.

Cross-backend and original-helper comparisons use the existing comparison
policy and common-point scale. The separately retained
`../final-focused-reference-reuse-v1/receipt.json` validates only the relevant
original-helper references. Its historical complex data has no fresh-process
parity measurement, and this report does not claim one for those old data.
No old candidate rows are reused.
