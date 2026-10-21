# Lightweight final study

The latest user direction narrows the final numerical study to Ninja (6) and
NinjaRS (8), using inexpensive Standard Model examples. Preserve the existing
high multiplicity correctness evidence and completed prefix controls. The final
random sample is limited to the cases below; the separate CLI profile has the
wider scope stated later in this note.

| Process family | Random-study cases | Helicity selection |
| --- | --- | --- |
| QCD | `qcd-1`, `qcd-2`, `qcd-3` | Full helicity and color sum |
| Electroweak | `ew-1`, `ew-2`, `ew-3` | Full helicity and color sum |
| Loop induced | `supplement-gg-h`, `supplement-gg-hh` | The preregistered all-minus gluon tuple; scalar helicities zero |

The final random sample uses the frozen 100 accepted configurations per case,
with the same seeds, cuts, masses, physical cards and point files for both
tools. Keep direction reversal and Lorentz rotation, per-reduction failure
recording, native-DP guards, initialization diagnostics, and exact resource
custody. Two-to-one samples describe boosted frames at their on-shell invariant
energy. They do not describe a continuum of invariant masses. Loop-induced
random panels use the declared single helicity; the main loop-induced
correctness and normal timing evaluations retain their full sums.

These existing 100 configurations are the final sample size. No larger stream,
sample-size map or generator extension is required. Historical artifact names
that contain "pilot" remain unchanged. Report the actual denominator, empirical
resolution and zero-event confidence bound. A 100-point sample has resolution
`1/N = 1%` and a one-sided 95% zero-event bound of approximately 2.95%; it cannot
support claims about a rarer instability rate. The paper's much larger sample
counts are methodological context, not adopted requirements.

The latest CLI profile scope is the eight core QCD/EW processes with one to four
final-state particles, plus loop-induced `gg -> H`, `HH`, `HHH` and `HHHH`.
Use three repetitions per tool, emphasizing reduction time with numerator and
total times retained as references. Report the exact initialization and timing
boundaries, stability/rescue policy, numerator/reduction treatment and measured uncertainty.
The diagnostic wall, WARM and DONE durations in the instrumented forced-DP
sample are retained for resource planning. They are not performance measurements.
Warm work grows with the requested point count because the driver requires
consecutive post-initialization passes over the selected stream.

Profiling the additional cases does not extend the random sample. No additional
full random stream is authorized for `qcd-4`, `ew-4`, or either
stress case. The unfinished private deadline-adapter and sample-extension
drafts are held and unused. Their existence is not execution authorization.
The final report records measurements and numerical agreement for the selected
lightweight scope; an exhaustive eight-process runtime threshold is no longer
a completion condition under the latest user direction.

The original fixed-point rescued correctness campaign, the new native-DP
self-consistency study, and the shared-point cross-backend Born/Laurent
diagnostic remain distinct evidence. Failed or nonfinite observations remain
visible, initialization safety is reported separately, and no result is
selected according to whether it is successful.

## Scope update — 2026-10-21

The user's latest profiling request covers twelve processes: the original eight
core QCD/EW cases through two-to-four kinematics, plus loop-induced `gg -> H`,
`HH`, `HHH` and `HHHH`. Use three repetitions for Ninja (6) and NinjaRS (8),
with reduction time as the main comparison and numerator and total time as
references. This is a separate profiling campaign. The original six-case
`N=10` timing study with full helicity and color sums and the eight-case
`N=100` random stability study retain their existing inputs and results. The
new profile cases do not authorize additional random stability samples.
