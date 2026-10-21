# Final allocation optimization decision study

The complete fresh study passes its two predeclared decision gates. The QCD3
allocation/original median ratio is **0.9685184**, with paired 95% interval
**[0.9568780, 0.9936397]**, over exactly 21 pairs. Median full MadLoop evaluation
falls from 126.4152 to 122.4354 ms per point, a 3.15% reduction. EW2's ratio is
**0.9976396**, interval **[0.9947157, 0.9993387]**, over exactly 7 pairs, passing
the at-most-1.05 regression gate. No optional samples or exclusions were used.

All 56 fresh workers exited zero. Every physical coefficient, reported stability
estimate, return status and raw numerical row is exactly identical across both
arms and all repetitions. The offline validator reparses every raw result and
checks 44,800 physical components. QCD3 has 98 DP results and 2 QP rescues per
100 calls in every stream; EW2 has 100 DP results and no rescue. Full numerator,
reduction, scalar masters, stability checks and actual rescue remain timed.

This is a full-process **optimization decision**, not the final eight-process
acceptance comparison with literal-original MadLoop and Ninja. The current
NinjaRS runtime still needs that separate benchmark campaign. No final 5% gate
or project-completion claim follows from this result.

The experiment uses the exact previously validated generic allocation tensor,
with no new arithmetic, precision, threshold, provider or rescue change. It
reuses byte-identical copied worker executables and unchanged generated Fortran,
cards, points, libraries, metadata and compiler fingerprints. CPU234 passed the
three fixed idle preflights. All workers use one thread. Original/archive and
allocation/archive identities are respectively 832fc0d1… and 366ae91d….

The earlier v1 study remains historical and interrupted. Its 12 pairs plus one
original arm are never reused in this study. Recovery verified exact stopped
PID/start/command identities and retired both jobs via pidfd SIGKILL, without
SIGCONT. The fresh study has a new root handoff, output namespace and frozen
bindings. All old retained evidence remains unchanged.

`frozen-bindings.json` binds the plan, copied workers, unchanged controllers and
immutable comparison helper. `timings/attempt-1/` contains the full handoff,
preflights, host records, every raw worker output, completion ledger and analysis.
`independent-recovery-audit.*` and the independent measurement audit retain the
separate review. `post-run-ready.json` proves the same worker identities remain
after measurement. `production-patch-confirmation.json` identifies the held
reviewable patch: measured release body plus 4 ordinary dense-oracle tests and
2 ignored capture integration tests, without touching lib.rs.

Root alone applies a final production patch, runs the resulting combined Rust
and MG7 regression tests, rebuilds and installs the runtime, performs the complete
correctness/stability/performance campaign, and commits or pushes. No production
source was changed by this isolated study.
