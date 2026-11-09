The retained QCD soft/collinear point exposes an avoidable double-precision Born failure. It is **not exactly singular in the supplied binary64 inputs**. This private Born-only diagnostic reproduces the original NaN and obtains a finite same-input quadruple-precision result. It does not yet qualify an automatically checked full-loop amplitude.

The generated `born_matrix.f:567` calls `FFV1_2` with the incoming quark and soft outgoing gluon. The internal momentum is their difference. Exact rational arithmetic on the original binary64 components gives its invariant as approximately `-5.0008825204128851e-14 GeV²`. Forming that momentum in binary64 rounds its energy and longitudinal component to exactly the same value, `499.9999` (hex `0x1.f3fff97247454p+8`). The generated left-associative squared norm then becomes zero. Merely factoring the norm after forming that DP momentum gives about `-1e-20`, so it cannot recover the lost information.

The private executable links the unchanged generated Born object, DHELAS archive and model archive. It calls the original DP Born calculation, then precision-lifted copies of the existing `GET_AMP`, `GET_JAMP` and `GET_MATRIX` routines. These copies call the already generated MP DHELAS functions. Inputs are first read into binary64 and then promoted exactly to binary128 **before** wavefunction or momentum arithmetic. Couplings and the Z mass are also exact promotions of their DP values. No phase-space repair, direct decimal-to-QP reinterpretation, reducer or Symbolica call occurs.

| Check | Retained result |
|---|---|
| Original ordinary DP Born | `3.9576388462893338e-5` |
| Ordinary QP Born | `3.9576388462893354e-5` |
| Original stress DP Born | NaN, IEEE invalid flag set |
| Same-input stress QP Born | `4.3074648942609748518e19` |
| Same-input QP propagator denominator | `-5.0008825204128863e-14` |
| Exchange identical final gluons | QP Born bit-identical |
| Exact cyclic spatial-axis permutation | QP Born agrees within `4.5e-16` relative |

Every QP helicity contribution is finite, and no invalid, divide-by-zero or overflow flag is raised. The diagnostic sums all 48 helicities, uses the existing two color structures, original color matrix `(16,-4,16)/3`, averaging/identical-particle denominator 72 and broken-symmetry factor. It is limited to this unpolarized single-split-order process. The source transformation is retained verbatim as `derivation.patch`; numerical and identity checks are in `analyze.py` and `analysis.json`.

The existing loop-QP route computes MP Born helicity amplitudes for interference but never replaces the DP `ANS(0,:)` calculated earlier in `loop_matrix.f`. Forced-QP loop follow-ups consequently retain the Born NaN. An automatic fallback must run immediately after the ordinary Born call and before setting the reference `REF`. It must publish every rescued Born split order and their selected sum, use the rescued value for `REF`, and expose it consistently to callers that normalize poles or finite terms by the Born. Simply replacing the displayed total after loop evaluation would leave filtering and normalization incoherent.

A production fallback should preserve the finite ordinary DP path exactly; only nonfinite Born results would trigger the extra QP calculation. It must retain helicity selection, beam polarization, flavor, split-order masks and averaging semantics. It must isolate the failed DP Born's IEEE flags from subsequent checks while preserving caller flags, and must fail explicitly if the QP result or its DP conversion is nonfinite. Genuine loop instability still requires the ordinary checked rescue chain; a rescued Born alone cannot turn an unchecked or rejected loop result into acceptance. The private follow-up prototype is separate from this frozen evidence folder.

`prepare.py`, `run.py` and `analyze.py` reproduce the successful experiment from the MG root after sourcing `tools/tirust_env.sh`. Compilation and execution were pinned to CPU 11. Original source/object/archive/card hashes were checked before and after. Two preparation path failures and a first native setup that exited at a missing `ident_card.dat` are retained. That first setup's process exit code was zero but no Born rows were produced; `native-attempt1/adjudication.json` explicitly invalidates it. The successful runner verifies all 14 expected result rows. Native binaries remain in ignored scratch directories; the evidence index contains only portable source, metadata and logs.
