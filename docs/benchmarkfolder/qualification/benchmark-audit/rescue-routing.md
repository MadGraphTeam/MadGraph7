# Frozen automatic rescue routing audit

For the warmed automatic policy `CTModeInit=1`, `CTModeRun=-1`, and
`MLReductionLib=9|8|6|7|1`, a TIRust DP refusal or failed amplitude stability
test does **not** dispatch TIRust QP before NinjaRS DP. The nominal amplitude
sequence is **9 DP → 8 DP → 6 DP → 7 DP → 1 DP**, stopping at an accepted
result. Only exhaustion of that DP list starts **9 QP → 8 QP → 6 QP → 1 QP**.
COLLIER has no separate QP entry in this generated dispatch table.

This is a source audit, not a measurement of which attempts any saved point
actually executed. No executable, card, original evidence, or generated source
was changed, and no numerical or compiler jobs were launched for this audit.

## Frozen source identity

Audited export root, relative to the repository:
`.benchmark-work/tirust/outputs/candidate/qcd_n4_uux_zzgg/`.

- Generator revision: `c91cb85f3aa942c90299b3518bb1f195f55334a5`.
- Generator source inventory SHA-256:
  `74116e4bb2ba7abdb24a3dbcdb51cfae6547f05c3e276d3e17f5d5dab2b8d7a5`.
- Recorded generator `dirty`: `false`.

| File under the export root | SHA-256 |
| --- | --- |
| `generation.json` | `19f8e2e800f4be6a326b595055c143b7ab810cd3327530eeccbe8548b4a7adae` |
| `SubProcesses/P0_uux_zzgg/loop_matrix.f` | `a07804ebcb4088c8b82da923b630f845c3d4339d4bacbccdca976308ecdc437a` |
| `SubProcesses/P0_uux_zzgg/TIR_interface.f` | `05ae4e14af288945845ba259f8f185b7257d113fa5ca8b49e80d8b43c4ef3904` |
| `SubProcesses/P0_uux_zzgg/CT_interface.f` | `046d23bc913bdbec4adee0608d479ad79e798141f5fbb2ddaff17a04912c832e` |
| `SubProcesses/MadLoopCommons.f` | `279e750387957413087fa0a5eedb2e291a48dc719af045e31ab3fa7b8b5963fd` |

## Routing evidence

Line numbers below refer to those exact generated files.

1. `loop_matrix.f:299–304` declares QP support for IDs 1, 6, 8, and 9.
   Lines 592–615 remove unavailable libraries and construct `INDEX_QP_TOOLS`
   in the remaining configured order. Lines 899–918 select initialization
   mode or steady automatic mode 1; lines 956–959 begin at the first library.
2. `CT_interface.f:1212` derives `DOING_QP` from `CTMODE>=4`.
   Lines 1281–1283 call `MP_TIRUST_LOOP` only when that condition holds;
   ordinary backend 9 reaches `TIRLOOP` at lines 1284–1288.
3. `TIR_interface.f:236–242` maps a failed DP tensor reduction to
   `FPE_IN_DP_REDUCTION=.TRUE.`, zero output, and return. Lines 254–261 apply
   the same treatment to nonfinite contracted results.
4. `loop_matrix.f:1412–1491` performs independent amplitude checks in
   automatic mode. At lines 1591–1594, a DP failure flag forces accuracy to
   one. Failed acceptance advances `I_LIB` at line 1635. If another DP
   library remains, lines 1652–1666 reset checks and retry it without
   changing precision. Only the end-of-list branch at lines 1636–1651
   selects `INDEX_QP_TOOLS(1)`, enables `DOING_QP_EVALS`, and sets mode 4.
   Failed QP acceptance advances the QP list at lines 1511–1572.

## Scope and exceptions

- A DP refusal is not necessarily an immediate jump: the same backend's
  direction/rotation checks can finish before the amplitude-level decision.
  NinjaRS additionally has one compensated DP retry at
  `loop_matrix.f:1608–1630`, before advancing to the next backend; this does
  not select QP.
- Topology selection is separate from numerical failure rescue.
  `TIR_interface.f:641–699` searches only within the current precision,
  wrapping through the configured list when a topology is unsupported; it
  cannot promote a DP attempt to QP. `MadLoopCommons.f:515–521` accepts
  TIRust for one through six propagators and rank zero through six.
  Outside that domain, an amplitude nominated for one backend can use
  another backend for particular integrals. Final amplitude return codes
  therefore do not enumerate every internal reduction attempt.
- With an explicit backend-9-only list, exhaustion of DP occurs immediately
  after that backend fails, so TIRust QP can follow TIRust DP. Standalone-9
  timings consequently do not determine the cost of the five-backend chain.
- Initialization normally uses `CTModeInit` and bypasses the warmed
  automatic acceptance block (`loop_matrix.f:899–901`, `1412`). Choosing
  initialization or fixed run mode at least 4 takes the direct multiple
  precision path at lines 1012–1023; that is outside the audited strict
  policy. `FORCE_STABILITY_CHECK` can trigger a separate initialization
  recheck at lines 3663–3671, marking it with an initialization return code.
  Under the strict policy that recheck starts in automatic DP and retains
  the same DP-before-QP order.
- These anchors establish MadLoop dispatch and its explicit TIRust QP
  entrypoint. They do not infer runtime precision usage or internal scalar
  provider algorithms from the final accepted amplitude status.
