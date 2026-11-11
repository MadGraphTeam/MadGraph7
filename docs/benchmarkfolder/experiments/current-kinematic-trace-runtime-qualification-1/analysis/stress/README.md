# Six admitted stress input/status decodes

Root's explicit review `2954d77a73bf5a31ff97b4e8b4f52b39989b5a62be17fd0e301c8a157c2b2484`
admits exactly six closed enabled traces from the failed numerical closure
`df0987b3f0c20dd26c4a903ff7c4acc7fb6cce4e433a50071a82918194553e9e`.
The review was authenticated before parsing and is retained verbatim here.
The unchanged sealed `df0c1113…` decoder ran with the explicit low-word-first
GNU x86_64 ABI authenticated by root. All six invocations exited zero; there
were no parser failures or decoder edits. Receipt/logs retain each command.

The trace records are input construction and native integer status only.
`KT_RAW` means the intermediate kinematic difference/matrix input; it is **not**
a returned tensor coefficient. No failed tensor outputs, other run payloads,
library, compiler, new native job, or Git operation were used in this analysis.

| Input/status trace | Selected groups | Native status | Critical input observation |
|---|---|---|---|
| CMS point20 mode1 | 3,4,8,12,24 | 1 for all five | RAW ≈−2.5e−7; ZERO branch; native invariant 0 |
| CMS point20 mode4 | 3,4,8,12,24 | 1 for all five | RAW ≈−2.5e−7; ZERO branch; native invariant 0 |
| CMS point21 mode1 | 3,4,8,12,24 | 1 for all five | RAW ≈−2.5e−11; ZERO branch; native invariant 0 |
| CMS point21 mode4 | 3,4,8,12,24 | 1 for all five | RAW ≈−2.5e−11; ZERO branch; native invariant 0 |
| EW point2 mode1 | 5 | 1 | MASSI projection onto observed DP-squared mass1 |
| EW point2 mode4 | 5 | 1 | MASSI projection onto the same widened DP-squared mass1 |

All 22 selected calls have residual, matrix, native-input and native-return
boundaries. Each returns status 1, so the selected calls themselves record a
nonzero native status; this is not inferred solely from global amplitude flags.
Exact decoded inputs and before/after branch snapshots are retained in each
output and in `summary.json`. No accuracy or tensor-completeness claim follows.

For CMS pair(2,4), the captured conservation residual is zero and the raw
spacelike difference remains nonzero in DP and QP. Reference normalization is
250000 and the observed DP OSTHRES is about 1e−8. The original ZERO branch
overwrites the matrix entry so the invariant supplied to TIRust is exactly zero
for every selected group. This establishes the projection and subsequent
status 1. It does **not** establish that the projection caused the refusal:
there was no counterfactual threshold/routing experiment or failure-cause trace.

For EW pair(1,2), original MASSI is taken. DP RAW is approximately
30032.88999999998850; QP RAW is approximately 30032.8900000000039404.
Both delivered invariants equal the real part of the observed DP-squared mass1,
30032.8900000000030559, unchanged by QP widening. Pair(3,4), the source-bound
external photon control, takes ZERO: DP RAW≈1.97019e−18 and QP RAW=0, with a
delivered invariant 0 in both cases. These projected inputs coexist with native
status 1. Which input or other conditioning property causes refusal remains
unknown from these traces.

Root reported bootstrap FPE flags (0,0), measurement mode1 flags (1,0), and
measurement mode4 flags (0,1) across these six triples. Those global flags are
explicitly attributed to root's admission, not reconstructed from selected-call
inputs. Root's exact unmodified/disabled/enabled equality preserves the failed
numerical gate; it does not convert the stress amplitudes into qualified results.

`summarize.py` reads only this package's decoded JSON outputs and review/receipt
and recomputes the summary. All scalar values are stored exactly as integer
words and rational hex ratios. No thresholds, model inputs, scientific sources
or sealed records were changed; no new numerical evaluations were performed.
