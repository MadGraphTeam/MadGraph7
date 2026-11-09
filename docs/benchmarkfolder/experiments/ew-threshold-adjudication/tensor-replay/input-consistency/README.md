# Consistency of captured tensor inputs

This arithmetic audit uses the exact round-trip decimal records from the
completed tensor replay. It rounds each token to binary128 with integer
arithmetic, then evaluates every squared shift difference as an exact rational
number. It invokes no reducer and changes no input. The script and input
hashes are recorded in `summary.json`.

The threshold box in group 5 has a maximum difference of
`8.8448359747417284164e-13 GeV²` between a supplied invariant and the invariant
derived from its captured shifts. The ordinary group-5 control has the same
mass-shell convention difference. The successfully served massless threshold
control, group 24, differs by at most `9.9193e-30 GeV²`.

The group-5 mismatch is consistent with the independently measured difference
between a promoted DP squared mass and the square of the promoted mass used
in QP phase-space repair. Relative to ordinary hard scales it is tiny, but
the threshold fixture has `s_tt - 4m_t²` of only about `6.5e-10 GeV²`.
The original MadLoop on-shell policy intentionally snaps invariants; this
audit does not propose changing that policy or the frozen point.

Matching every argument supplied to two reducer APIs does not establish
exact consistency between their supplied momentum vectors and invariants.
If a subsequent higher-precision replay disagrees, internal uses of these
two representations and scalar classification must also be examined. This
audit alone identifies no specific recurrence as the cause of a coefficient
disagreement and certifies no amplitude result.

Reproduce from the repository root:

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/input-consistency/audit.py
```
