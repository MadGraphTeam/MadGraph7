#import "style.typ": *
#let data = json("ew-qualification-data.json")
#show: body => report-style(body, date: "12 November 2026", revision: "unapplied mass candidate · control 0fcd2cd873d6")
#set heading(numbering: none)

= Electroweak mass-interface qualification

#note(fill: warm)[*Ordinary DP regression passes; the threshold failure remains open.*
  The QP-mass source patch is unapplied. These controls establish neither
  independent physical accuracy nor performance acceptance.]

Both studies use #raw(data.process) in the pinned #raw(data.model) model, with TIRust backend 9, cache disabled and normal stability handling at #raw("MLStabThres=1e-6"). Fresh CLI generation, generated-source correspondence and compilation are retained for the original integration and the mass prototype. The comparison concerns this one EW process.

== Ordinary sample: 100 accepted DP points per arm

#table-style[
  #table(columns: (2.9fr, 1.35fr, 1.35fr), align: (left, right, right),
    inset: 5pt, stroke: table-stroke, fill: table-fill,
    [Observed per arm], [Original], [Mass candidate],
    [Physical points], [100], [100],
    [Accepted DP / QP points], [100 / 0], [100 / 0],
    [Public accepted status], [219], [219],
    [Returned coefficient / accuracy slots], [800 / 200], [800 / 200],
  )
]

All returned coefficient, accuracy, status, history and observed IEEE/software-state words agree exactly. The pair passes both the #raw("3e-6") regression gate and #raw("1e-8") target with zero pair difference. Its maximum reported accuracy estimate is #raw("1.0098069303873654e-7"), below the checked threshold. These are agreement and estimator observations, not an independent finite-answer oracle. No accepted QP point or changed native QP-mass input is observed; unobserved internal QP attempts are not excluded.

Each worker retains 101 initialization calls, including one scientifically unqualified bootstrap. The remaining 100-point warmup and final sample each return 200 overall/sole-order rows; those duplicate rows are not additional physical points. Explicit #raw("LoopPoleMode=0") evaluates full sums and restores stored filter state after each call. These records do *not* qualify persistent filter learning.

== Threshold / soft-photon fixture: both arms refuse

#table-style[
  #table(columns: (2.9fr, 1.35fr, 1.35fr), align: (left, right, right),
    inset: 5pt, stroke: table-stroke, fill: table-fill,
    [Target and observation], [Original], [Mass candidate],
    [Ordinary control], [Accepted 219], [Accepted 219],
    [Saved threshold fixture], [*Refused 420*], [*Refused 420*],
    [Reported threshold accuracy], [1], [1],
    [QP software latch before → after], [0 → 1], [0 → 1],
  )
]

The ordinary control and complete 100-point initialization records agree exactly. At the threshold, four changed words in failed return buffers are retained only as diagnostics. They are *not accepted coefficients* and receive no numerical accuracy, improvement or degradation comparison. Caller momentum/scale words remain unchanged and observed IEEE vectors agree, but neither is a witness of repaired or native reducer inputs. The mass correction does not resolve this stress failure.

The earlier pre-READY refusal and corrected pole-policy readiness check are preserved. Five later policy snapshots per arm retain the source-derived full-sum bypass and stored initialization flags; those flags are not learned filters. Native QP inputs, independent physical truth, genuine >53-bit physical masses, broader stress/cache coverage and promotion remain unqualified. No timings are added; Hua-Sheng’s precise local performance acceptance remains pending.

#text(7.7pt, fill: ink2)[Evidence: ordinary index #raw("48a8b22cdb65…"), threshold index #raw("df6f4e453f57…"). Exact identities and pure reproduction: #raw("ew-qualification-selection.json"), #raw("derive_evidence.py"). Style follows preserved pyAmpliCol revision #raw("7266c1600f6e…").]
