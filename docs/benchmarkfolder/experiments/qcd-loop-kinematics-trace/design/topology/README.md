# Exact routing supplement for the historical QCD export

This is a source-only audit of `.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg`. It does not compile, execute a reduction, calculate a Born amplitude, change a threshold, or modify that export. The retained source hashes identify the historical files; the conclusions are not automatically transferable to another generated process.

## Momentum signs and the composite channel

Write the physical external momenta as `p1,p2 -> p3,p4,p5` and define all-incoming momenta `k=(p1,p2,-p3,-p4,-p5)`. The generated tree calls in `SubProcesses/P0_uux_zgg/helas_calls_ampb_1.f:101–105` and its MP counterpart at lines 86–90 pass signs `+1,-1,+1,+1,+1`. The actual DHELAS assignments establish `W(i)%P=-k_i`:

| Routine in `Source/DHELAS/aloha_functions.f` | DP assignment lines | MP assignment lines | Rule |
| --- | --- | --- | --- |
| `IXXXXX` | 107–110 | 383–386 | `fi%P=-nsf*p` |
| `OXXXXX` | 538–541 | 780–783 | `fo%P=nsf*p` |
| `VXXXXX` | 1497–1500 | 1601–1604 | `vc%P=nsv*p` |

The loop tree calls construct W12 from W1 and W5 (`helas_calls_ampb_1.f:122`, `mp_helas_calls_ampb_1.f:107`). Both `Source/DHELAS/FFV1_2.f:20` and `MP_FFV1_2.f:20` assign the sum of the mother `%P` fields. Therefore **`W12%P=-p1+p5`**. The generated `CT_interface.f` constructs each corner by negating its grouped mother momenta, for example lines 1230–1240 and 1703–1715. A W12 corner is consequently **`p1-p5=k1+k5`**, a composite channel regardless of its numerical invariant.

Explicit W12 calls occur in groups **31, 32, 34, 35, 38, 39, 40**, at `loop_CT_calls_1.f:182–207`. The final integer argument is the group ID; preceding comments list diagram numbers, which are a different identifier. Explicit `PAIRING` arguments must also be respected: a corner can contain several wavefunctions.

## Group 43 distinguishes three cases

`loop_CT_calls_1.f:215–216` calls `LOOP_3(1,5,24,...,I_SO,43)`. W24 combines W11 and W4 (`helas_calls_ampb_1.f:254`), and W11 combines W2 and W3 (line 117). Its three corner incidences are therefore `{1}`, `{5}`, `{2,3,4}`.

The TIRust QP adapter stores `q1=sum(C_j)`, `q2=C1`, `q3=C1+C2` (`TIRUST_interface.f:169–194`). In particular, its first shift is the computed conservation residual, not an imposed zero. Integer incidence gives:

| Propagator difference | Formal expression in the all-incoming basis | Classification |
| --- | --- | --- |
| `q2-q3` | `-k5` | Direct external leg 5, massless |
| `q1-q2` | `k2+k3+k4+k5` | External leg 1 only modulo total conservation |
| `q1-q3` | `k2+k3+k4` | Composite, equivalent to `-(p1-p5)` modulo conservation |

These are formal routing identities. Floating sums can retain rounding and conservation residuals, including in the direct-leg case. Keep the unreduced integer vector, captured momenta, and canonical class separately. For CT2/5 with more than two propagators, `TIR_interface.f:341–377` maps corners to `C'_j=-C_(N+1-j)` and reverses mass entries 2 through N; for N<=2 it returns unchanged. The same transformation must be applied to the incidence metadata.

## Reproduction and scope

`audit_routing.py` parses the fixed-form source calls, retains the wavefunction and group incidence, checks DP/MP routing consistency and the DHELAS momentum rules, and binds the source bytes. Its JSON artifact is `routing.json`; the supplementary file manifest is `evidence-index.json`. Run the parser from the MadGraph repository root; no MadGraph import or native evaluation is required. The retained invocation and checks are recorded with the artifact.

```sh
source tools/tirust_env.sh
PYTHONDONTWRITEBYTECODE=1 python3 docs/benchmarkfolder/experiments/qcd-loop-kinematics-trace/design/topology/audit_routing.py --verify docs/benchmarkfolder/experiments/qcd-loop-kinematics-trace/design/topology/routing.json
```

Use `--output NEW_FILE.json` instead of `--verify` to retain a fresh result; existing files are never overwritten. An alternative export can be selected with `--export-root`, but this is a deliberately narrow historical audit that fails if its expected source forms or group counts change.

The retained result checks all **35 DP/MP wavefunction producers** and **116 loop groups**, giving **477 unordered propagator pairs**: 178 massless external-leg channels, 64 Z channels, and 235 composite channels. Literal difference subsets have sizes 1/2/3/4 with counts **146/114/121/96**. The separate forward-arc counts are **220/167/68/22**; those use the other representative for pairs involving the first shift. In particular, 22 is not the number of literal four-leg differences in the adapter.

The integer classification treats a one-leg subset or its four-leg complement as a single external leg; two- and three-leg subsets remain composite. A subset size is used only after checking that each group partitions all five legs exactly once. Counts describe static source coverage, not runtime calls: filtering, repeated visits, rescue and rotations change the latter. A numerical observer must retain both ordered matrix entries because independently applied threshold branches can overwrite previous assignments.

Topology cannot establish numerical equivalence between **B**, the exact promotion of the original DP event used by the private Born rescue, and **C**, the post-`SET_MP_PS` repaired QP event used to build the loop numerator. The class `p1-p5` is the same at B and C; its invariant and the corresponding Born value may differ substantially near collinearity. The parent design requires separate B/C channel values and a separately authorized consistent Born comparison, also preserving any rotated input D. This supplement supplies neither those numerical values nor evidence that a clamp caused the rejected amplitude. No frozen Born evidence has been changed.
